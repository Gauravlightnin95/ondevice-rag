"""
Stage 8 — Blind batched grading of the inference grid.

Accuracy, groundedness and citation correctness in one call per batch. Rows are shuffled
with a fixed seed and batched by question type, so the judge never sees which model
produced an answer, which device ran it, or which condition it came from. Blindness is
enforced on the rendered prompt by rubric.assert_blind, not merely by convention.

EMPTY OUTPUTS ARE NEVER SENT TO THE JUDGE. They score 0 deterministically and carry
empty_output=true. The rule matters most on type 5: "correct" there means stating the
answer is absent, and silence is not a refusal - it carries no evidence the model
recognised absence. Under any "did not hallucinate" reading, a model emitting nothing
would score 100% on type 5 and beat every model that correctly says "not stated". Scoring
them here rather than in the judge also removes ~1,000 calls and removes the risk of the
judge confabulating a rationale for blank text.

"0 by silence" and "0 by hallucination" are different failures - one a capability limit,
the other a safety one - so they are tagged and reported separately. README section 7's
refusal rate counts explicit refusals only.

Modes:
    batchcheck  same rows graded batched and unbatched, paired, before the full run
    full        grade everything not already done

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\grade.py --mode batchcheck
    .venv\\Scripts\\python.exe src\\analysis\\grade.py --mode full
    .venv\\Scripts\\python.exe src\\analysis\\grade.py --mode full --resume results\\grades_....jsonl
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge import Judge, assert_no_key_in                      # noqa: E402
from rubric import assert_blind, build_item, build_prompt      # noqa: E402

RESULTS = Path("results")
EVALSET = Path("evalset/evalset_v1.jsonl")
CHUNKS = Path("corpus/chunks.jsonl")
BATCH = 18
SEED = 20260906


def latest(pattern):
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} in {RESULTS}")
    return files[-1]


def load_rows():
    """Grid rows joined to gold answers and supplied context, contaminated ones dropped."""
    ev = {r["qid"]: r for r in (json.loads(l) for l in EVALSET.open(encoding="utf-8"))}
    chunk_text = {json.loads(l)["chunk_id"]: json.loads(l)["text"]
                  for l in CHUNKS.open(encoding="utf-8")}

    bad = set()
    cpath = RESULTS / "contaminated_qids.json"
    if cpath.exists():
        bad = set(json.loads(cpath.read_text(encoding="utf-8"))["contaminated_qids"])
        print(f"excluding {len(bad)} contaminated questions")
    else:
        print("WARNING: contaminated_qids.json missing — run screen_contamination first")

    rows = []
    for line in latest("grid_full_*.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["error"] or r["qid"] in bad:
            continue
        q = ev[r["qid"]]
        # Context the answering system actually saw, so groundedness is graded against it.
        ctx = []
        if r["condition"] == "rag":
            ctx = [(c, chunk_text[c]) for c in r["retrieved_chunk_ids"]]
        elif r["condition"] == "oracle":
            if r["oracle_context_source"] == "retrieved_chunk":
                ctx = [(c, chunk_text[c]) for c in r["oracle_chunk_ids"]]
            elif q["type"] == 4:
                ctx = [("passage_1", q["passage"]), ("passage_2", q["passage2"])]
            else:
                ctx = [("passage_1", q["passage"])]
        rows.append({
            "row_id": f"{r['model']}|{r['device']}|{r['condition']}|{r['qid']}",
            "qid": r["qid"], "type": r["type"], "model": r["model"],
            "device": r["device"], "condition": r["condition"],
            "question": q["question"], "gold": q["answer"],
            "answer": (r["cleaned_output"] or "").strip(),
            "context": ctx,
            "gold_chunk_ids": r["gold_chunk_ids"],
        })
    return rows


def score_empty(row):
    """Deterministic 0 for an empty answer. See the module docstring for why type 5 too."""
    return {"row_id": row["row_id"], "qid": row["qid"], "type": row["type"],
            "model": row["model"], "device": row["device"],
            "condition": row["condition"], "accuracy": 0, "groundedness": None,
            "cited": [], "citation_correct": None, "empty_output": True,
            "judged": False, "reason": "empty cleaned output — scored 0 without the judge"}


def grade_batch(j, qtype, rows, batch_index=None):
    """One call. Returns score dicts keyed by row_id.

    The judge is given OPAQUE per-batch ids, never row_id. row_id is
    "<model>|<device>|<condition>|<qid>", so passing it as the item id would put the model
    name, the device and the condition in front of the judge on every item - which
    assert_blind caught the first time this ran. The mapping back is local to this call.
    """
    opaque = {f"i{n}": r for n, r in enumerate(rows)}
    items = [build_item(oid, r["question"], r["gold"], r["answer"], r["context"])
             for oid, r in opaque.items()]
    prompt = build_prompt(qtype, items)
    assert_blind(prompt)
    scores, _ = j.ask_json(prompt, max_output_tokens=32768)
    lookup = {str(s.get("id")): s for s in scores}
    out = []
    for pos, (oid, r) in enumerate(opaque.items()):
        s = lookup.get(oid, {})
        cited = s.get("cited") or []
        cit = None
        if r["context"]:
            cit = (None if not cited
                   else bool(set(cited) & set(r["gold_chunk_ids"])))
        out.append({"row_id": r["row_id"], "qid": r["qid"], "type": r["type"],
                    "model": r["model"], "device": r["device"],
                    "condition": r["condition"],
                    "accuracy": s.get("accuracy"),
                    "groundedness": s.get("groundedness"),
                    "cited": cited, "citation_correct": cit,
                    "empty_output": False, "judged": True,
                    "batch_pos": pos, "batch_index": batch_index,
                    "reason": (s.get("reason") or "")[:200]})
    return out


def run_full(j, rows, out, done):
    empties = [r for r in rows if not r["answer"] and r["row_id"] not in done]
    if empties:
        with out.open("a", encoding="utf-8", newline="\n") as fh:
            for r in empties:
                g = score_empty(r)
                done[r["row_id"]] = g
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"{len(empties)} empty outputs scored 0 without a judge call")

    todo = [r for r in rows if r["answer"] and r["row_id"] not in done]
    rng = random.Random(SEED)
    rng.shuffle(todo)                       # ordering must carry no signal
    by_type = defaultdict(list)
    for r in todo:
        by_type[r["type"]].append(r)

    total = sum(len(v) for v in by_type.values())
    n_batches = sum((len(v) + BATCH - 1) // BATCH for v in by_type.values())
    print(f"{total} rows to grade in {n_batches} batches of {BATCH} "
          f"(~{n_batches*j.min_interval/60:.0f} min at {60/j.min_interval:.0f} RPM)")

    bi = 0
    for qtype, group in sorted(by_type.items()):
        for s in range(0, len(group), BATCH):
            chunk = group[s:s + BATCH]
            rng.shuffle(chunk)              # randomise position within the batch
            bi += 1
            try:
                res = grade_batch(j, qtype, chunk, batch_index=bi)
            except Exception as exc:
                print(f"  batch {bi} (type {qtype}) failed: {j.safe(exc)[:140]}")
                continue
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                for g in res:
                    done[g["row_id"]] = g
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
            if bi % 10 == 0 or bi == n_batches:
                print(f"  {bi}/{n_batches} batches, {len(done)} graded")


def run_batchcheck(j, rows, out):
    """Same rows graded batched and unbatched, paired.

    The real daily quota is 500, not 1,500 (measured:
    GenerateRequestsPerDayPerProjectPerModel-FreeTier). Unbatched grading of ~4,300 rows is
    therefore ~9 days rather than ~3, so batching is once again load-bearing for the
    budget and not only a convenience. That does not change what this check measures, but
    it does change what a divergence costs: if batched and unbatched disagree, the honest
    options are to grade unbatched over nine days or to reduce the batch size and pay for
    it in time. The verdict below reports the fact; it does not get to assume the
    convenient answer.

    Weighted toward types 4 and 5: those need the judge to notice something ABSENT - a
    conflict between passages, or that the answer is not in the corpus - which is the
    judgement most likely to degrade when attention is split across 18 rows, and most
    likely for a lite model to abandon in favour of "did it answer the question". Types 1
    and 3 are near-mechanical comparisons and would show high agreement even if batching
    hurt the cases that matter.
    """
    # Sized to the MEASURED daily quota, not the assumed one. The real limit is
    # GenerateRequestsPerDayPerProjectPerModel-FreeTier = 500, not 1,500. The unbatched
    # arm costs one call per row, so a 200-row check would consume 40% of a day's budget
    # and leave too little for the ~240 calls the full batched grade needs. 100 rows still
    # detects a meaningful divergence, and the weighting to types 4 and 5 is preserved
    # because that is where the divergence would actually appear.
    want = {4: 30, 5: 30, 1: 10, 2: 10, 3: 10, 6: 10}
    rng = random.Random(SEED)
    by_type = defaultdict(list)
    for r in rows:
        if r["answer"]:
            by_type[r["type"]].append(r)
    sample = []
    for t, n in want.items():
        pool = sorted(by_type[t], key=lambda r: r["row_id"])
        sample.extend(rng.sample(pool, min(n, len(pool))))
    print(f"batch-vs-unbatched on {len(sample)} paired rows "
          f"(weighted to types 4 and 5)\n")

    batched, unbatched = {}, {}
    bt = defaultdict(list)
    for r in sample:
        bt[r["type"]].append(r)
    bi = 0
    for qtype, group in sorted(bt.items()):
        for s in range(0, len(group), BATCH):
            chunk = group[s:s + BATCH]
            bi += 1
            for g in grade_batch(j, qtype, chunk, batch_index=bi):
                batched[g["row_id"]] = g
    print(f"  batched pass done ({bi} calls)")

    for i, r in enumerate(sample, 1):
        for g in grade_batch(j, r["type"], [r], batch_index=None):
            unbatched[g["row_id"]] = g
        if i % 25 == 0 or i == len(sample):
            print(f"  unbatched {i}/{len(sample)}")

    pairs = [(batched[k]["accuracy"], unbatched[k]["accuracy"], batched[k])
             for k in batched if k in unbatched
             and batched[k]["accuracy"] is not None
             and unbatched[k]["accuracy"] is not None]

    agree = sum(1 for b, u, _ in pairs if b == u)
    diffs = [b - u for b, u, _ in pairs]
    print(f"\n{'='*66}")
    print(f"paired rows compared: {len(pairs)}")
    print(f"exact agreement     : {agree}/{len(pairs)} = {100*agree/len(pairs):.1f}%")
    print(f"mean score difference (batched - unbatched): {statistics.mean(diffs):+.3f}")
    print(f"  -> {'batched is more lenient' if statistics.mean(diffs) > 0.05 else 'batched is harsher' if statistics.mean(diffs) < -0.05 else 'no systematic direction'}")
    print("\nby type:")
    for t in sorted({p[2]['type'] for p in pairs}):
        sub = [(b, u) for b, u, g in pairs if g["type"] == t]
        a = sum(1 for b, u in sub if b == u)
        d = statistics.mean(b - u for b, u in sub)
        print(f"  type {t}: {a}/{len(sub)} agree ({100*a/len(sub):>5.1f}%), "
              f"mean diff {d:+.3f}")

    print("\nposition effect within batch (mean accuracy by position):")
    bypos = defaultdict(list)
    for b, u, g in pairs:
        if g.get("batch_pos") is not None:
            bypos[g["batch_pos"] // 6].append(b)
    for band in sorted(bypos):
        v = bypos[band]
        print(f"  positions {band*6}-{band*6+5}: n={len(v):>3}  mean {statistics.mean(v):.3f}")

    rate = agree / len(pairs)
    mean_d = statistics.mean(diffs)
    verdict = ("BATCH" if rate >= 0.90 and abs(mean_d) <= 0.05 else "GRADE UNBATCHED")
    print(f"\nVERDICT: {verdict}")
    if verdict != "BATCH":
        print("  Batching changes the measurement, and the quota no longer requires it")
        print("  (30 RPM / 1,500 RPD covers ~4,300 rows unbatched). Grade unbatched.")

    payload = {"verdict": verdict,
               "n_pairs": len(pairs), "exact_agreement": agree / len(pairs),
               "mean_score_diff_batched_minus_unbatched": statistics.mean(diffs),
               "by_type": {str(t): {
                   "agreement": sum(1 for b, u, g in pairs if g["type"] == t and b == u)
                                / max(sum(1 for _, _, g in pairs if g["type"] == t), 1),
                   "mean_diff": statistics.mean([b - u for b, u, g in pairs
                                                 if g["type"] == t] or [0])}
                   for t in sorted({p[2]['type'] for p in pairs})},
               "position_bands": {str(k): statistics.mean(v) for k, v in sorted(bypos.items())}}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "batchcheck"], required=True)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = load_rows()
    print(f"{len(rows)} grid rows after exclusions "
          f"({sum(1 for r in rows if not r['answer'])} empty answers)\n")
    j = Judge()

    if args.mode == "batchcheck":
        run_batchcheck(j, rows, RESULTS / f"batchcheck_{ts}.json")
    else:
        out = args.resume or RESULTS / f"grades_{ts}.jsonl"
        done = {}
        if args.resume and out.exists():
            for line in out.open(encoding="utf-8"):
                g = json.loads(line)
                done[g["row_id"]] = g
            print(f"resuming: {len(done)} already graded")
        run_full(j, rows, out, done)
        meta = {"judge_model": j.model, "judge_model_version": j.model_version,
                "batch_size": BATCH, "seed": SEED, "graded": len(done),
                "calls": j.calls, "retries": j.retries}
        mp = out.with_suffix(".meta.json")
        mp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        assert_no_key_in(mp)
        assert_no_key_in(out)
        print(f"\n{len(done)} grades -> {out}")
        print(f"meta -> {mp}  (model name and version only, no key)")


if __name__ == "__main__":
    main()
