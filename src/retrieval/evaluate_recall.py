"""
Stage 4 — Retrieval evaluation. Recall@k per question type.

Gold chunks come from evalset/passage_locations.jsonl, produced by Stage 3's aligner.
They are NOT re-derived here: a naive substring search would fail on the 76 passages
whose extracted text carries PUA bullet glyphs, lost inter-word spaces or column-wise
table order, and would silently under-report recall.

Type 5 has no gold passage and is excluded, leaving 194 questions.

No single aggregate figure is reported. Multi-hop retrieval is a different problem from
single-hop lookup and an all-types average hides exactly what H2 predicts.

Scoring rules (see docs/ingestion_notes.md):
  any-hit   primary for every type — did top-k contain ANY gold chunk
  all-hit   reported for the 15 composite passages, whose gold text is spread across
            several locations
  both-hit  reported for the 25 type-4 questions, which carry two gold passages, split
            by whether those sit in different documents (24) or the same one (1)

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\retrieval\\evaluate_recall.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import CANDIDATE_DEPTH, RRF_K, search, search_single  # noqa: E402

EVALSET = Path("evalset/evalset_v1.jsonl")
LOCATIONS = Path("evalset/passage_locations.jsonl")
OUT = Path("results/retrieval_recall.json")

KS = (1, 3, 5, 10)
MAX_K = max(KS)
TYPES = (1, 2, 3, 4, 6)          # 5 excluded: no gold passage


def load_questions():
    ev = {r["qid"]: r for r in (json.loads(l) for l in EVALSET.open(encoding="utf-8"))}
    per_passage = defaultdict(list)
    for line in LOCATIONS.open(encoding="utf-8"):
        r = json.loads(line)
        per_passage[r["qid"]].append(r)

    questions = []
    for qid, records in per_passage.items():
        q = ev[qid]
        if q["type"] == 5:
            continue
        questions.append({
            "qid": qid,
            "type": q["type"],
            "question": q["question"],
            # one gold set per passage; a type-4 question has two
            "passage_gold": [set(r["chunk_ids"]) for r in records],
            "gold": {c for r in records for c in r["chunk_ids"]},
            "composite": any(r["n_spans"] > 1 for r in records),
            "dual": len(records) == 2,
            "cross_doc": len({r["doc_id"] for r in records}) > 1,
        })
    return sorted(questions, key=lambda q: q["qid"])


def pct(hit, total):
    return f"{100.0 * hit / total:5.1f}" if total else "    -"


def main():
    questions = load_questions()
    print(f"{len(questions)} questions with a gold passage (type 5 excluded)")
    print(f"retrieval: RRF k={RRF_K} over top-{CANDIDATE_DEPTH} dense + top-{CANDIDATE_DEPTH} BM25\n")

    for i, q in enumerate(questions, 1):
        fused = [h[0] for h in search(q["question"], k=MAX_K)]
        dense, sparse = search_single(q["question"], k=MAX_K)
        q["fused"], q["dense"], q["sparse"] = fused, dense, sparse
        if i % 25 == 0 or i == len(questions):
            print(f"  retrieving {i}/{len(questions)}")

    def rate(subset, k, ranking="fused", mode="any"):
        hit = 0
        for q in subset:
            top = set(q[ranking][:k])
            if mode == "any":
                hit += bool(top & q["gold"])
            elif mode == "all":
                hit += q["gold"].issubset(top)
            elif mode == "both":
                hit += all(top & g for g in q["passage_gold"])
        return hit, len(subset)

    report = {"n_questions": len(questions), "ks": list(KS), "by_type": {},
              "supplementary": {}, "single_retriever": {}}

    # ---- primary: any-hit per type, fused
    print("Recall@k — any-hit, fused (primary)")
    print(f"  {'type':<6} {'n':>4}  " + "  ".join(f"@{k:<4}" for k in KS))
    for t in TYPES:
        sub = [q for q in questions if q["type"] == t]
        cells = []
        for k in KS:
            h, n = rate(sub, k)
            cells.append(pct(h, n))
            report["by_type"].setdefault(str(t), {})[f"any@{k}"] = round(h / n, 4)
        report["by_type"][str(t)]["n"] = len(sub)
        print(f"  {t:<6} {len(sub):>4}  " + "  ".join(f"{c}%" for c in cells))
    print("  (no aggregate row: an average over types hides the multi-hop result)")

    # ---- dense vs sparse vs fused, per type, so fusion has to earn its place
    print("\nRetriever comparison — any-hit @10")
    print(f"  {'type':<6} {'n':>4}  {'dense':>7}  {'bm25':>7}  {'fused':>7}")
    for t in TYPES:
        sub = [q for q in questions if q["type"] == t]
        row = {}
        for name in ("dense", "sparse", "fused"):
            h, n = rate(sub, 10, ranking=name)
            row[name] = round(h / n, 4)
        report["single_retriever"][str(t)] = row
        print(f"  {t:<6} {len(sub):>4}  {100*row['dense']:6.1f}%  "
              f"{100*row['sparse']:6.1f}%  {100*row['fused']:6.1f}%")

    # ---- supplementary: all-hit on the composite passages
    comp = [q for q in questions if q["composite"]]
    print(f"\nAll-hit — the {len(comp)} composite passages (gold spread across locations)")
    print(f"  by type: " + ", ".join(
        f"{t}:{sum(1 for q in comp if q['type']==t)}" for t in TYPES
        if any(q['type'] == t for q in comp)))
    print(f"  {'':<6} {'n':>4}  " + "  ".join(f"@{k:<4}" for k in KS))
    cells_any, cells_all = [], []
    for k in KS:
        ha, n = rate(comp, k, mode="any")
        hl, _ = rate(comp, k, mode="all")
        cells_any.append(pct(ha, n))
        cells_all.append(pct(hl, n))
        report["supplementary"][f"composite_any@{k}"] = round(ha / n, 4)
        report["supplementary"][f"composite_all@{k}"] = round(hl / n, 4)
    print(f"  {'any':<6} {len(comp):>4}  " + "  ".join(f"{c}%" for c in cells_any))
    print(f"  {'all':<6} {len(comp):>4}  " + "  ".join(f"{c}%" for c in cells_all))

    # ---- supplementary: both-hit on type 4, split by cross-document
    dual = [q for q in questions if q["dual"]]
    cross = [q for q in dual if q["cross_doc"]]
    same = [q for q in dual if not q["cross_doc"]]
    print(f"\nBoth-sources hit — the {len(dual)} type-4 questions (two gold passages each)")
    print(f"  {'':<22} {'n':>4}  " + "  ".join(f"@{k:<4}" for k in KS))
    for label, sub, key in (("all type 4", dual, "type4"),
                            ("cross-document", cross, "type4_cross_doc"),
                            ("same document", same, "type4_same_doc")):
        cells = []
        for k in KS:
            h, n = rate(sub, k, mode="both")
            cells.append(pct(h, n))
            report["supplementary"][f"{key}_both@{k}"] = round(h / n, 4) if n else None
        note = "  (sample of one)" if len(sub) == 1 else ""
        print(f"  {label:<22} {len(sub):>4}  " + "  ".join(f"{c}%" for c in cells) + note)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUT}")
    print("\nContext: retrieval is from a pool of 491 chunks — a far smaller haystack than")
    print("the web-scale settings of the work being compared against. High Recall@10 here")
    print("is not evidence that retrieval is solved.")


if __name__ == "__main__":
    main()
