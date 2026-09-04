"""
Stage 1 — Ingestion. Chunk the extracted corpus into corpus/chunks.jsonl.

Fixed parameters from README section 6. This is a frozen-retriever step: re-running on
the same corpus must produce a byte-identical file, so everything here is deterministic
and every invariant is asserted rather than assumed.

The 512-token budget is measured in bge-small-en-v1.5 WordPiece tokens, not Qwen BPE.
bge's own context window is 512, so chunking in its tokenizer is what guarantees no
chunk is silently truncated at embed time in stage 4. See docs/ingestion_notes.md.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\ingest\\chunk.py
"""

import json
import re
import statistics
from pathlib import Path

from transformers import AutoTokenizer

CORPUS = Path("corpus/extracted")
EVALSET = Path("evalset")
OUT = Path("corpus/chunks.jsonl")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512
OVERLAP = 64
STRIDE = MAX_TOKENS - OVERLAP  # 448

# Verified totals — see README section 4. A mismatch means the corpus moved.
EXPECT_DOCS = 29
EXPECT_PAGES = 507

PAGE_MARKER = re.compile(r"^===== PAGE (\d+) =====$", re.M)


def normalise(page_text):
    """Deterministic whitespace cleanup for one page.

    PyMuPDF leaves a trailing space on nearly every line and emits long runs of blank
    lines. WordPiece discards whitespace, so this changes no token count — but the same
    text is later fed to the LLM, where padding would inflate the prefill-token metric.
    Intra-line spacing is preserved so table rows stay readable.
    """
    lines = [ln.rstrip(" \t") for ln in page_text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def split_pages(raw, name):
    """Split on the page markers into [(page_no, page_text)], asserting 1..N contiguous."""
    parts = PAGE_MARKER.split(raw)
    # parts[0] is whatever precedes the first marker; extract.py emits only a newline.
    assert not parts[0].strip(), f"{name}: text before the first page marker"

    pages = []
    for i in range(1, len(parts), 2):
        pages.append((int(parts[i]), parts[i + 1]))

    numbers = [n for n, _ in pages]
    assert numbers == list(range(1, len(pages) + 1)), \
        f"{name}: page numbers are not contiguous from 1: {numbers[:10]}..."
    return pages


def build_document(pages):
    """Join normalised pages into one string plus a [(start, end, page_no)] span table."""
    chunks_of_text, spans, cursor = [], [], 0
    for page_no, page_text in pages:
        body = normalise(page_text)
        if not body:
            # A page with no text layer still occupies a position; record an empty span
            # so page accounting stays honest, but contribute nothing to the token stream.
            spans.append((cursor, cursor, page_no))
            continue
        if chunks_of_text:
            chunks_of_text.append("\n")
            cursor += 1
        start = cursor
        chunks_of_text.append(body)
        cursor += len(body)
        spans.append((start, cursor, page_no))
    return "".join(chunks_of_text), spans


def pages_for(spans, char_start, char_end):
    """Every page whose span overlaps [char_start, char_end)."""
    hit = [p for s, e, p in spans if s < char_end and e > char_start]
    if not hit:
        # Only reachable if a chunk lands entirely inside an empty page's zero-width span.
        hit = [next(p for s, e, p in spans if s <= char_start <= e)]
    return hit


def chunk_document(doc, spans, doc_id, source_file, tok):
    """Slide a MAX_TOKENS window with STRIDE step over the document's token stream."""
    enc = tok(doc, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = enc["offset_mapping"]
    n_tokens = len(offsets)
    if n_tokens == 0:
        return []

    records, index, start = [], 0, 0
    while True:
        end = min(start + MAX_TOKENS, n_tokens)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        span_pages = pages_for(spans, char_start, char_end)

        records.append({
            "chunk_id": f"{doc_id}::c{index:04d}",
            "doc_id": doc_id,
            "source_file": source_file,
            "chunk_index": index,
            "text": doc[char_start:char_end],
            "n_tokens": end - start,
            "source_page": span_pages[0],
            "page_start": span_pages[0],
            "page_end": span_pages[-1],
            "pages": span_pages,
            "char_start": char_start,
            "char_end": char_end,
            "token_start": start,
            "token_end": end,
        })

        if end >= n_tokens:
            break
        index += 1
        start += STRIDE

    return records


def check_windows(records, name):
    """Assert the stride and overlap held across every consecutive pair in one document.

    A gap check alone would pass an off-by-one stride: no text is lost, but the overlap
    silently becomes 63 or 65 and every downstream chunk boundary shifts. Since each
    window starts stride-aligned and every non-final window is a full MAX_TOKENS, the
    overlap is arithmetically always OVERLAP — so this asserts equality, not a tolerance.
    Equality also proves the loop never emitted a final stub already covered by its
    predecessor, which would show up here as a wrong overlap.
    """
    for prev, cur in zip(records, records[1:]):
        step = cur["token_start"] - prev["token_start"]
        assert step == STRIDE, f"{name}: stride {step} != {STRIDE} at {cur['chunk_id']}"

        overlap = prev["token_end"] - cur["token_start"]
        assert overlap == OVERLAP, \
            f"{name}: overlap {overlap} != {OVERLAP} at {cur['chunk_id']}"

        assert cur["char_start"] <= prev["char_end"], \
            f"{name}: character gap before {cur['chunk_id']}"


def eval_sources():
    """Distinct source/source2 values from evalset/*.md, normalised to bare stems.

    Top level only — evalset/notes/ is prose and is never parsed (README section 5).
    Read-only; the eval set is frozen.
    """
    field = re.compile(r"^- \*\*source2?:\*\* (.+)$", re.M)
    found = set()
    for md in sorted(EVALSET.glob("*.md")):
        for value in field.findall(md.read_text(encoding="utf-8")):
            value = value.strip()
            if value and value != "null":
                found.add(Path(value).stem.lower())
    return sorted(found)


def main():
    files = sorted(CORPUS.glob("*.txt"))
    assert len(files) == EXPECT_DOCS, f"expected {EXPECT_DOCS} documents, found {len(files)}"

    print(f"Tokenizer: {EMBED_MODEL}")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    assert tok.is_fast, "a fast tokenizer is required for offset mapping"

    all_records, pages_seen, per_doc = [], 0, []

    for path in files:
        raw = path.read_text(encoding="utf-8")
        pages = split_pages(raw, path.name)
        pages_seen += len(pages)

        doc, spans = build_document(pages)
        doc_id = path.stem.lower()
        records = chunk_document(doc, spans, doc_id, path.name, tok)
        check_windows(records, path.name)

        all_records.extend(records)
        per_doc.append((path.name, len(pages), len(records)))

    assert pages_seen == EXPECT_PAGES, f"expected {EXPECT_PAGES} pages, saw {pages_seen}"

    ids = [r["chunk_id"] for r in all_records]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    assert all(r["n_tokens"] <= MAX_TOKENS for r in all_records), "chunk over the token budget"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Summary is counts only. Chunk text is never printed: two documents carry names,
    # mobile numbers, roll numbers and a live-looking credential (README section 4).
    print(f"\n{'document':<48} {'pages':>6} {'chunks':>7}")
    for name, n_pages, n_chunks in per_doc:
        print(f"  {name:<46} {n_pages:>6} {n_chunks:>7}")

    sizes = [r["n_tokens"] for r in all_records]
    print(f"\n{len(files)} documents, {pages_seen} pages, {len(all_records)} chunks -> {OUT}")
    print(f"tokens per chunk: min {min(sizes)}, median {int(statistics.median(sizes))}, "
          f"max {max(sizes)}")

    doc_ids = {r["doc_id"] for r in all_records}
    sources = eval_sources()
    matched = [s for s in sources if s in doc_ids]
    missing = [s for s in sources if s not in doc_ids]

    print(f"\nEval-set source coverage: {len(matched)}/{len(sources)} distinct sources "
          f"match a chunked document")
    for source in missing:
        print(f"  UNMATCHED  {source}")
    if missing:
        print("  These need an alias map to their corpus filenames before stage 3 can")
        print("  validate their gold passages. Reported here, not fixed — stage 2/3 work.")


if __name__ == "__main__":
    main()
