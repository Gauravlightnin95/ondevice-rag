"""
Stage 3 — Passage validator and locator.

Two separate questions, deliberately kept apart:

  1. Does the gold passage appear verbatim in its source document?
     Reported as a statistic. It measures extraction fidelity, not question quality —
     Oracle-RAG feeds the passage as written in the eval set, so whitespace and glyph
     drift never reaches the model.

  2. Can the passage be located to a chunk?
     This is the gate. Stage 4's Recall@k needs each gold passage's chunk_id; a passage
     that cannot be located means no retrieval recall for that question.

Locating is done against the document string rebuilt with chunk.py's own helpers, so a
located character span maps to chunk IDs by offset overlap rather than by re-matching
text. Reimplementing that normalisation here would let the offsets drift silently.

The eval set is never modified. Passages are not edited to match extraction artifacts:
writing PUA glyphs and missing spaces into the frozen passages would encode extraction
bugs into the artifact and make Oracle-RAG feed the model corrupted text.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\ingest\\validate_passages.py
"""

import json
import re
import statistics
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk import build_document, split_pages  # noqa: E402  same normalisation as stage 1

CORPUS = Path("corpus/extracted")
CHUNKS = Path("corpus/chunks.jsonl")
EVALSET = Path("evalset/evalset_v1.jsonl")
OUT = Path("evalset/passage_locations.jsonl")

# Below this, a token-aligned window is rejected rather than accepted. For Recall@k a
# wrong chunk is worse than a missing one: it corrupts the metric instead of showing
# up as a gap.
DENSITY_FLOOR = 0.30
MIN_SEGMENT_TOKENS = 5

# A passage counts as located only if this fraction of its tokens landed inside an
# emitted span. Guards against a long passage being "located" by one stray fragment.
COVERAGE_FLOOR = 0.60

# Hard-coded by qid, never detected by heuristic. A heuristic that mislabelled a
# genuinely broken passage as a redaction would hide it permanently, and stage 4 would
# never locate its chunk. These still have to locate — the whitelist governs how they
# are reported, not whether they pass the gate.
PRIVACY_REDACTED = {
    "Students-GrievanceRedressal-Cell::Q1":
        "names and roll number removed per README section 4",
    "Students-GrievanceRedressal-Cell::Q3":
        "names and designations removed per README section 4",
}

# Character substitution table. Recorded in docs/ingestion_notes.md.
SUBS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2010": "-", "\u2011": "-", "\u2212": "-",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00a0": " ", "\u2026": "...",
}
# Deleted outright: private-use glyphs (Symbol/Wingdings bullets survive extraction as
# U+F0B7 and friends), the replacement char where a smart quote failed to decode, and
# the quote characters themselves — U+FFFD stands in for a quote the extractor lost, so
# deleting both sides is the only symmetric fix.
DROP = set("'\"")


def is_dropped(ch):
    return "\uf000" <= ch <= "\uf8ff" or ch == "\ufffd" or ch in DROP


def compact(text):
    """Fold to a comparison form, keeping a map back to original character offsets.

    Returns (compacted, index) where index[i] is the offset in `text` that produced
    compacted[i]. All whitespace is removed rather than collapsed, which resolves the
    document's glued words ('belowtimings' against 'below timings') and hyphenated line
    breaks ('extra- curricular' against 'extra-curricular') under one rule.
    """
    out, index = [], []
    for i, ch in enumerate(text):
        if ch.isspace() or is_dropped(ch):
            continue
        for sub in SUBS.get(ch, ch):
            out.append(sub.lower())
            index.append(i)
    return "".join(out), index


TOKEN = re.compile(r"[0-9a-z]+")


def tokenise(text):
    """Alphanumeric tokens with their character spans in `text`.

    PUA glyphs, the replacement char and punctuation are non-alphanumeric, so they act
    as separators and drop out for free — which is what makes bullets and broken quotes
    invisible to the aligner.
    """
    folded = "".join(SUBS.get(ch, ch) if len(SUBS.get(ch, ch)) == 1 else ch
                     for ch in text).lower()
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(folded)]


def locate_exact(passage, doc_compact, doc_index, doc_text):
    """Contiguous match after folding. Returns (char_start, char_end) or None."""
    needle, _ = compact(passage)
    if not needle:
        return None
    at = doc_compact.find(needle)
    if at < 0:
        return None
    return doc_index[at], doc_index[at + len(needle) - 1] + 1


def align_window(ptokens, dtokens, postings):
    """Tightest window of dtokens containing ptokens in order.

    Anchored on the passage's rarest token so the candidate starts stay few — anchoring
    on the first token would scan every occurrence of a word like 'the'.
    Returns (start_idx, end_idx, density) or None.
    """
    if not ptokens:
        return None
    anchor = min(range(len(ptokens)),
                 key=lambda i: len(postings.get(ptokens[i], ())))
    spots = postings.get(ptokens[anchor], ())
    if not spots:
        return None

    best = None
    for spot in spots:
        i = spot
        ok = True
        for w in ptokens[anchor + 1:]:                      # forward from the anchor
            nxt = postings.get(w)
            if not nxt:
                ok = False
                break
            k = bisect_left(nxt, i + 1)
            if k >= len(nxt):
                ok = False
                break
            i = nxt[k]
        if not ok:
            continue
        end = i + 1

        j = spot
        for w in reversed(ptokens[:anchor]):                # backward from the anchor
            prev = postings.get(w)
            if not prev:
                ok = False
                break
            k = bisect_left(prev, j) - 1
            if k < 0:
                ok = False
                break
            j = prev[k]
        if not ok:
            continue
        start = j

        density = len(ptokens) / (end - start)
        if best is None or density > best[2]:
            best = (start, end, density)
        if density >= 0.999:
            break
    return best


def locate_aligned(passage, dtokens, postings):
    """Locate by token alignment, splitting into segments where density collapses.

    A passage the annotator assembled from several places in the document cannot match
    one tight window. Rather than widening the window until it matches something, take
    the longest prefix that holds above the density floor, emit it, and continue with
    the remainder. That yields one span per real location.

    A token that cannot start an acceptable segment is skipped rather than failing the
    whole passage: the AICTE approval tables extract column-wise, so a row read across
    the table has its figures scattered document-wide. Dropping one token and carrying
    on locates the parts that are real and reports how much was covered, instead of
    discarding a passage that is mostly locatable.

    Returns (spans, densities, coverage) where coverage is the fraction of passage
    tokens that landed inside an emitted span.
    """
    ptokens = [t for t, _, _ in tokenise(passage)]
    total = len(ptokens)
    spans, densities, rest, covered = [], [], ptokens, 0

    while rest:
        hit = align_window(rest, dtokens, postings)
        if hit and hit[2] >= DENSITY_FLOOR:
            take = len(rest)
        else:
            lo, hi, take = MIN_SEGMENT_TOKENS, len(rest), 0
            while lo <= hi:                                  # longest acceptable prefix
                mid = (lo + hi) // 2
                probe = align_window(rest[:mid], dtokens, postings)
                if probe and probe[2] >= DENSITY_FLOOR:
                    take, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if not take:
                rest = rest[1:]                              # unplaceable token, skip it
                continue
            hit = align_window(rest[:take], dtokens, postings)

        start, end, density = hit
        spans.append((dtokens[start][1], dtokens[end - 1][2]))
        densities.append(density)
        covered += take
        rest = rest[take:]

    return spans, densities, (covered / total if total else 0.0)


def page_of(spans, pos):
    """Page containing a character offset, using stage 1's own page span table."""
    for start, end, page in spans:
        if start <= pos <= end:
            return page
    return None


def chunks_for(spans, chunk_rows):
    """Every chunk whose character span overlaps any located span."""
    hits = []
    for row in chunk_rows:
        for lo, hi in spans:
            if row["char_start"] < hi and row["char_end"] > lo:
                hits.append(row)
                break
    return hits


def classify(passage, doc_text):
    """How faithful the passage is to the extracted text. Reporting only, never gating."""
    squash = lambda s: " ".join(s.split())
    if squash(passage) in squash(doc_text):
        return "L0 strict verbatim"
    folded = "".join("" if is_dropped(c) else SUBS.get(c, c) for c in passage)
    doc_folded = "".join("" if is_dropped(c) else SUBS.get(c, c) for c in doc_text)
    if squash(folded) in squash(doc_folded):
        return "L1 char-normalised"
    if compact(passage)[0] in compact(doc_text)[0]:
        return "L2 whitespace-insensitive"
    return "L3 gap-tolerant"


def main():
    records = [json.loads(l) for l in EVALSET.open(encoding="utf-8")]
    by_qid = {r["qid"]: r for r in records}

    missing = [q for q in PRIVACY_REDACTED if q not in by_qid]
    assert not missing, f"whitelisted qid no longer in the eval set: {missing}"

    chunk_rows = defaultdict(list)
    for line in CHUNKS.open(encoding="utf-8"):
        row = json.loads(line)
        chunk_rows[row["doc_id"]].append(row)

    docs, page_spans = {}, {}
    for path in sorted(CORPUS.glob("*.txt")):
        text, spans = build_document(split_pages(path.read_text(encoding="utf-8"),
                                                path.name))
        comp, index = compact(text)
        tokens = tokenise(text)
        postings = defaultdict(list)
        for i, (tok, _, _) in enumerate(tokens):
            postings[tok].append(i)
        docs[path.stem.lower()] = (text, comp, index, tokens, postings)
        page_spans[path.stem.lower()] = spans

    jobs = []
    for r in records:
        for field, doc_key, page_key in (("passage", "source_doc_id", "page_list"),
                                         ("passage2", "source2_doc_id", "page2_list")):
            if r.get(field) and r.get(doc_key):
                jobs.append((r, field, r[doc_key], r[page_key]))

    out, layers, unlocated, uncorroborated, all_density = [], Counter(), [], [], []

    for r, field, doc_id, page_list in jobs:
        passage = r[field]
        text, comp, index, tokens, postings = docs[doc_id]
        layer = classify(passage, text)
        layers[layer] += 1

        hit = locate_exact(passage, comp, index, text)
        if hit:
            spans, densities, coverage = [hit], [1.0], 1.0
        else:
            spans, densities, coverage = locate_aligned(passage, tokens, postings)

        hits = chunks_for(spans, chunk_rows[doc_id]) if spans else []
        if not hits or coverage < COVERAGE_FLOOR:
            unlocated.append((r["qid"], field, doc_id, passage, coverage))
            continue

        pages = sorted({p for h in hits for p in h["pages"]})
        located_page = page_of(page_spans[doc_id], spans[0][0])
        # The pages the passage itself sits on. Narrower than chunk_pages, which covers
        # whole chunks and can span up to 7 pages — this is the page-in-corpus answer.
        passage_pages = sorted({
            pg for lo, hi in spans
            for start, end, pg in page_spans[doc_id]
            if start < hi and end > lo
        } | {located_page} - {None})
        corroborated = (not page_list) or bool(set(pages) & set(page_list))
        offset = (located_page - min(page_list)) if page_list and located_page else None
        if not corroborated:
            uncorroborated.append((r["qid"], field, doc_id, located_page, page_list, offset))
        all_density.extend(densities)

        out.append({
            "qid": r["qid"],
            "field": field,
            "type": r["type"],
            "doc_id": doc_id,
            "layer": layer,
            "density": round(min(densities), 4),
            "coverage": round(coverage, 4),
            "n_spans": len(spans),
            "spans": [[a, b] for a, b in spans],
            "chunk_ids": [h["chunk_id"] for h in hits],
            "chunk_pages": pages,
            "passage_pages": passage_pages,
            "located_page": located_page,
            "page_list": page_list,
            "page_offset": offset,
            "page_corroborated": corroborated,
            "privacy_redacted": r["qid"] in PRIVACY_REDACTED,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for record in out:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- report: counts and IDs, with passage text only as the specified 80-char head
    print(f"{len(jobs)} non-null passages "
          f"({sum(1 for j in jobs if j[1] == 'passage')} passage, "
          f"{sum(1 for j in jobs if j[1] == 'passage2')} passage2)\n")

    print("Extraction fidelity — how faithful each passage is to the extracted text")
    for name in ("L0 strict verbatim", "L1 char-normalised",
                 "L2 whitespace-insensitive", "L3 gap-tolerant"):
        print(f"  {name:28} {layers.get(name, 0):3d}")
    strict = layers.get("L0 strict verbatim", 0)
    print(f"  strict verbatim rate: {strict}/{len(jobs)} "
          f"({100 * strict / len(jobs):.1f}%) — reported, not gating")

    print("\nMismatches against strict verbatim (qid, source, first 80 chars)")
    shown = 0
    for r, field, doc_id, _ in jobs:
        layer = classify(r[field], docs[doc_id][0])
        if layer == "L0 strict verbatim":
            continue
        shown += 1
        head = " ".join(r[field].split())[:80]
        print(f"  {r['qid']}  [{field}]  {doc_id}")
        print(f"      {layer}  |  {head!r}")
    print(f"  {shown} mismatches listed")

    multi = [o for o in out if o["n_spans"] > 1]
    multi_chunk = [o for o in out if len(o["chunk_ids"]) > 1]
    straddle = [o for o in multi_chunk if o["n_spans"] == 1]
    print(f"\nMulti-span population — stage 4 needs this before defining Recall@k")
    print(f"  spans per passage  : {dict(sorted(Counter(o['n_spans'] for o in out).items()))}")
    print(f"  chunks per passage : "
          f"{dict(sorted(Counter(len(o['chunk_ids']) for o in out).items()))}")
    print(f"  passages with more than one gold chunk   : {len(multi_chunk)}, of which:")
    print(f"    one span straddling a chunk boundary   : {len(straddle)}  "
          f"— the passage is in ONE place and merely crosses a cut, so any-hit is the")
    print(f"                                             only sensible rule for these")
    print(f"    genuinely composite, several locations  : {len(multi_chunk) - len(straddle)}  "
          f"— any-hit vs all-hit is a real choice here")
    print(f"  passages resolving to more than one span : {len(multi)}")
    print(f"  by question type (composite only)        : "
          f"{dict(sorted(Counter(o['type'] for o in multi).items()))}")
    for o in multi:
        print(f"    {o['qid']:<44} [{o['field']}] type {o['type']}  "
              f"{o['n_spans']} spans, {len(o['chunk_ids'])} chunks, "
              f"coverage {o['coverage']:.0%}")

    if all_density:
        print(f"\nAlignment density: min {min(all_density):.2f}, "
              f"median {statistics.median(all_density):.2f} (floor {DENSITY_FLOOR})")

    print("\nPrivacy redactions — the redacted form is the correct state, do not fix")
    for qid, why in PRIVACY_REDACTED.items():
        rec = next((o for o in out if o["qid"] == qid), None)
        state = f"located, {len(rec['chunk_ids'])} chunk(s)" if rec else "NOT LOCATED"
        print(f"  {qid}\n      {why}\n      {state}")

    print(f"\nPage corroboration: {len(out) - len(uncorroborated)}/{len(out)} passages land on "
          f"a chunk covering their recorded page")
    if uncorroborated:
        per_doc = defaultdict(list)
        for qid, field, doc_id, located, want, offset in uncorroborated:
            per_doc[doc_id].append(offset)
        print("  Offsets of the failures, by document (located page - recorded page):")
        for doc_id in sorted(per_doc):
            spread = Counter(o for o in per_doc[doc_id] if o is not None)
            print(f"    {doc_id:<38} {len(per_doc[doc_id]):>3} failing  {dict(sorted(spread.items()))}")
        print("  A consistent non-zero offset means the eval set recorded the page number")
        print("  printed on the page while extraction numbers physical pages from 1 — the")
        print("  location is right and the recorded page is shifted by the front matter.")
        for qid, field, doc_id, located, want, offset in uncorroborated:
            print(f"    {qid} [{field}] located p{located}, eval set says {want} "
                  f"(offset {offset:+d})" if offset is not None else
                  f"    {qid} [{field}] located p{located}, eval set says {want}")

    print(f"\nLocatability gate: {len(out)}/{len(jobs)} passages map to at least one chunk "
          f"(coverage floor {COVERAGE_FLOOR})")
    for qid, field, doc_id, passage, coverage in unlocated:
        head = " ".join(passage.split())[:80]
        print(f"  UNLOCATED {qid} [{field}] {doc_id}  coverage {coverage:.0%}\n      {head!r}")

    print(f"\n{len(out)} located passages -> {OUT}")

    if unlocated:
        sys.exit(1)


if __name__ == "__main__":
    main()
