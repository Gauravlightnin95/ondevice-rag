"""
Stage 2 — Eval set loader. Parse evalset/*.md into evalset/evalset_v1.jsonl.

Only two line forms are read, per README section 5:
    ^## Q<n>
    ^- **field:** value
Everything else on every line is skipped. evalset/notes/ is prose commentary and is
never opened — the glob is top level, so it cannot be reached even by accident.

The eval set is frozen, so this is read-only over the markdown. Every count that the
README states is asserted here rather than trusted: a silent parser drop would show up
downstream as a quietly smaller grid, not as an error.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\ingest\\load_evalset.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

EVALSET = Path("evalset")
CORPUS = Path("corpus/extracted")
CHUNKS = Path("corpus/chunks.jsonl")
OUT = EVALSET / "evalset_v1.jsonl"

# README section 5. Asserted, not assumed.
EXPECT_PER_TYPE = {1: 46, 2: 50, 3: 51, 4: 25, 5: 29, 6: 22}
EXPECT_TOTAL = 223

REQUIRED = ["type", "question", "answer", "passage", "page", "source",
            "annotator", "contaminated"]
TYPE4_EXTRA = ["passage2", "page2", "source2"]
NULLABLE = ["passage", "page", "source"]  # type 5 sets all three to null

HEADER = re.compile(r"^## Q(\d+)\s*$")
FIELD = re.compile(r"^- \*\*([a-z0-9_]+):\*\* (.*)$")
PAGE_MARKER = re.compile(r"^===== PAGE (\d+) =====$", re.M)


def doc_id_for(source):
    """Eval-set source name -> chunk doc_id. Same rule as stage 1: lowercased stem."""
    if source is None:
        return None
    return Path(source).stem.lower()


def parse_pages(value):
    """'29-31' -> [29,30,31];  '2, 5, 7' -> [2,5,7];  null -> []."""
    if value is None:
        return []
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_file(path):
    """Read one markdown file into a list of raw field dicts, order-independent.

    Field order varies between questions — _conflicts.md Q1 runs passage, passage2,
    page, source, source2, page2 — so fields are accumulated into a dict and never
    read positionally.
    """
    questions, current, headers = [], None, 0
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = HEADER.match(line)
        if m:
            headers += 1
            current = {"_md": path.name, "_n": int(m.group(1))}
            questions.append(current)
            continue
        m = FIELD.match(line)
        if m and current is not None:
            current[m.group(1)] = m.group(2).strip()
    return questions, headers


def build_record(raw):
    """One parsed question -> the output record, with nulls and page lists resolved."""
    qtype = int(raw["type"])
    stem = Path(raw["_md"]).stem

    def value(key):
        v = raw.get(key)
        return None if v == "null" else v

    page = value("page")
    source = value("source")

    record = {
        "qid": f"{stem}::Q{raw['_n']}",
        "source_md": raw["_md"],
        "q_num": raw["_n"],
        "type": qtype,
        "question": raw["question"],
        "answer": raw["answer"],
        "passage": value("passage"),
        "page": page,
        "page_list": parse_pages(page),
        "source": source,
        "source_doc_id": doc_id_for(source),
    }

    # Non-type-4 records omit these keys rather than padding them with nulls.
    if qtype == 4:
        page2 = value("page2")
        source2 = value("source2")
        record.update({
            "passage2": value("passage2"),
            "page2": page2,
            "page2_list": parse_pages(page2),
            "source2": source2,
            "source2_doc_id": doc_id_for(source2),
        })

    record["annotator"] = raw["annotator"]
    record["contaminated"] = raw["contaminated"]
    return record


def page_counts():
    """doc_id -> page count, from the ===== PAGE N ===== markers.

    Counted from the extracted text rather than from chunks.jsonl: a trailing page with
    no text layer produces no chunk, so chunk data would understate the real page count.
    """
    counts = {}
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        counts[path.stem.lower()] = len(PAGE_MARKER.findall(text))
    return counts


def report(records, headers_seen, failures):
    """Counts and IDs only — passage text is never printed (README section 4)."""
    by_type = Counter(r["type"] for r in records)

    print("Questions per type")
    for qtype in sorted(EXPECT_PER_TYPE):
        want = EXPECT_PER_TYPE[qtype]
        got = by_type.get(qtype, 0)
        flag = "OK " if got == want else "BAD"
        print(f"  {flag} type {qtype}   {got:>3} (expected {want})")
    print(f"      total    {len(records):>3} (expected {EXPECT_TOTAL})")

    # 1 + 2. Counts and header parity. A drop here means the parser lost questions.
    assert headers_seen == len(records), \
        f"{headers_seen} '## Q' headers but {len(records)} records emitted"
    for qtype, want in EXPECT_PER_TYPE.items():
        assert by_type.get(qtype, 0) == want, \
            f"type {qtype}: {by_type.get(qtype, 0)} questions, expected {want}"
    assert len(records) == EXPECT_TOTAL

    # 3. qid uniqueness.
    qids = [r["qid"] for r in records]
    assert len(qids) == len(set(qids)), "duplicate qid"

    # 4. Type 4 carries the second passage, and nothing else does.
    with_extra = {r["qid"] for r in records if "passage2" in r}
    type4 = {r["qid"] for r in records if r["type"] == 4}
    assert with_extra == type4, f"passage2 present on non-type-4: {with_extra ^ type4}"
    assert all(all(k in r for k in ("passage2", "page2", "source2"))
               for r in records if r["type"] == 4), "type 4 missing a second-source field"

    # 5. Type 5 nulls, and only type 5.
    nulled = {r["qid"] for r in records if r["source"] is None}
    type5 = {r["qid"] for r in records if r["type"] == 5}
    assert nulled == type5, f"null source outside type 5: {nulled ^ type5}"
    assert all(r["passage"] is None and r["page"] is None
               for r in records if r["type"] == 5), "type 5 with a non-null passage or page"

    # 6. Required fields survived the parse.
    for r in records:
        for key in ("question", "answer", "annotator", "contaminated"):
            assert r[key], f"{r['qid']}: empty {key}"

    print(f"\nqid scheme: {records[0]['qid']}  ...  {records[-1]['qid']}")
    print(f"type 4 with a second source: {len(type4)}   type 5 with nulls: {len(type5)}")

    # 8. Page forms, printed for review before the artifact is frozen.
    forms = {}
    for r in records:
        for raw_key, list_key in (("page", "page_list"), ("page2", "page2_list")):
            v = r.get(raw_key)
            if v and not v.isdigit():
                forms[v] = r[list_key]
    print(f"\nNon-integer page forms ({len(forms)} distinct), parsed:")
    for v in sorted(forms, key=lambda s: (len(s), s)):
        print(f"  {v!r:<12} -> {forms[v]}")

    if failures:
        print("\n" + "=" * 66)
        print("FAILED CHECK — page reference outside its document")
        for line in failures:
            print(f"  {line}")
        print("=" * 66)


def main():
    files = sorted(EVALSET.glob("*.md"))  # top level only; notes/ is unreachable
    print(f"Reading {len(files)} markdown files from {EVALSET}/ (notes/ not parsed)")

    raws, headers_seen = [], 0
    for path in files:
        parsed, headers = parse_file(path)
        raws.extend(parsed)
        headers_seen += headers

    for raw in raws:
        missing = [k for k in REQUIRED if k not in raw]
        assert not missing, f"{raw['_md']} Q{raw['_n']}: missing {missing}"
        if raw["type"] == "4":
            missing = [k for k in TYPE4_EXTRA if k not in raw]
            assert not missing, f"{raw['_md']} Q{raw['_n']}: type 4 missing {missing}"

    records = [build_record(r) for r in raws]

    # 7. Every non-null source must join to a chunked document.
    chunk_docs = {json.loads(line)["doc_id"]
                  for line in CHUNKS.open(encoding="utf-8")}
    referenced = Counter()
    for r in records:
        for key in ("source_doc_id", "source2_doc_id"):
            if r.get(key):
                referenced[r[key]] += 1
    unmatched = sorted(d for d in referenced if d not in chunk_docs)

    # 9. Every page number must fall inside its own source document.
    counts = page_counts()
    failures = []
    for r in records:
        for list_key, doc_key in (("page_list", "source_doc_id"),
                                  ("page2_list", "source2_doc_id")):
            doc = r.get(doc_key)
            if not doc:
                continue
            total = counts.get(doc)
            for page in r.get(list_key, []):
                if total is None or not 1 <= page <= total:
                    failures.append(
                        f"{r['qid']}: {list_key[:-5]} {page} but {doc} has {total} pages")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    report(records, headers_seen, failures)

    print(f"\nSource join: {len(referenced) - len(unmatched)}/{len(referenced)} distinct "
          f"sources match a chunked document ({sum(referenced.values())} references)")
    for doc in unmatched:
        print(f"  UNMATCHED  {doc}")
    assert not unmatched, f"eval-set sources with no chunked document: {unmatched}"

    print(f"Page bounds: {len(failures)} references outside their document")
    print(f"\n{len(records)} questions -> {OUT}")

    # The output is still written on a bounds failure so stage 3 has something to work
    # with, but the run must not read as clean.
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
