"""
Stage 8, Gate 1 — Contamination screening. Runs BEFORE any grading.

README section 13: every question run closed-book; anything answered correctly with no
documents is testing memorised knowledge, not retrieval, and is removed. Expect to lose
10-20%, mostly types 1 and 3.

Grading rows for questions that are about to be deleted wastes calls and produces accuracy
figures that shift afterwards, which is why this is a gate rather than a later filter.

Design decisions worth stating:

  * TYPE 5 IS EXCLUDED. There is no answer to memorise. A judge "correctly" answering a
    type-5 question means refusing - correct behaviour, not contamination. Screening them
    would delete precisely the questions that test refusal.

  * QUESTIONS ARE ASKED ONE PER CALL, not batched. Batching would put several questions
    from the same corpus in one context, letting them cue each other, which would inflate
    apparent contamination. The judge's unaided recall is the thing being measured.

  * THREE PASSES, DELETE ON MAJORITY. A single lucky generation should not remove a
    question from a frozen eval set.

  * DELETE ON SCORE 2 ONLY. Score 1 is flagged for review: partial recall of a figure is
    weaker evidence of memorisation than a clean hit.

The eval set is NOT modified. Contaminated qids are written to a filter list that every
downstream analysis honours, so the frozen artifact stays intact and the decision stays
reversible. Editing evalset/ would trigger README section 12's v1.1 rule.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\screen_contamination.py
    .venv\\Scripts\\python.exe src\\analysis\\screen_contamination.py --resume results\\contamination_....jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge import Judge, assert_no_key_in                      # noqa: E402
from rubric import assert_blind, build_item, build_prompt      # noqa: E402

EVALSET = Path("evalset/evalset_v1.jsonl")
RESULTS = Path("results")
PASSES = 3
GRADE_BATCH = 12

# NEUTRAL BY DESIGN. The first version of this prompt said "Do not guess. Do not
# speculate. A wrong confident answer is worse than admitting ignorance." That produced
# 92.1% "I do not know" and a 2.1% contamination rate - which measured what the model
# produces WHEN DISCOURAGED FROM TRYING, not what it can recall. Contamination screening
# has to measure recall at its ceiling, so the question is asked plainly and the grader,
# not the prompt, decides whether the answer was right.
#
# The opposite bias - forcing an answer and rewarding a lucky hallucination - is handled
# downstream: deletion requires a majority score of 2 against the gold answer, which an
# invented figure will not reach.
ANSWER_PROMPT = """Answer this question about an Indian higher-education institution.

Give your best answer from your own knowledge, in one or two sentences.
If you have no relevant knowledge at all, say: I do not know.

Question: {question}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--passes", type=int, default=PASSES,
                    help="run fewer passes first to measure a prompt change cheaply")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.resume or RESULTS / f"contamination_{ts}.jsonl"

    questions = [json.loads(l) for l in EVALSET.open(encoding="utf-8")]
    # Type 5 excluded: nothing to memorise, and refusing is correct behaviour there.
    screened = [q for q in questions if q["type"] != 5]
    print(f"{len(screened)} questions screened ({len(questions) - len(screened)} type-5 "
          f"excluded — no answer to memorise)")

    done = {}
    if args.resume and out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            done[(r["qid"], r["pass"])] = r
        print(f"resuming: {len(done)} (qid, pass) results already present")

    j = Judge()
    print(f"judge: {j.model} at {60/j.min_interval:.0f} RPM\n")

    # ---- pass 1..N: answer each question closed-book, one per call
    for p in range(args.passes):
        todo = [q for q in screened if (q["qid"], p) not in done]
        if not todo:
            print(f"pass {p+1}/{args.passes}: complete")
            continue
        print(f"pass {p+1}/{args.passes}: {len(todo)} answers "
              f"(~{len(todo)*j.min_interval/60:.0f} min)")
        for i, q in enumerate(todo, 1):
            try:
                text, _ = j.ask(ANSWER_PROMPT.format(question=q["question"]),
                                max_output_tokens=2048)
                err = None
            except Exception as exc:
                text, err = "", j.safe(exc)
            row = {"qid": q["qid"], "type": q["type"], "pass": p,
                   "question": q["question"], "gold": q["answer"],
                   "judge_answer": text.strip(), "error": err}
            done[(q["qid"], p)] = row
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}")

    # ---- grade the judge's own closed-book answers against gold, batched by type
    print("\ngrading closed-book answers against gold...")
    graded_path = out.with_name(out.stem + "_graded.jsonl")
    graded = {}
    if graded_path.exists():
        for line in graded_path.open(encoding="utf-8"):
            r = json.loads(line)
            graded[(r["qid"], r["pass"])] = r

    by_type = defaultdict(list)
    for (qid, p), row in sorted(done.items()):
        if (qid, p) not in graded and not row["error"]:
            by_type[row["type"]].append(row)

    for qtype, rows in sorted(by_type.items()):
        for s in range(0, len(rows), GRADE_BATCH):
            chunk = rows[s:s + GRADE_BATCH]
            items = [build_item(f"{r['qid']}#{r['pass']}", r["question"], r["gold"],
                                r["judge_answer"] or "(no answer)") for r in chunk]
            prompt = build_prompt(qtype, items)
            assert_blind(prompt)
            try:
                scores, _ = j.ask_json(prompt, max_output_tokens=16384)
            except Exception as exc:
                print(f"    type {qtype} batch failed: {j.safe(exc)[:120]}")
                continue
            lookup = {str(x.get("id")): x for x in scores}
            with graded_path.open("a", encoding="utf-8", newline="\n") as fh:
                for r in chunk:
                    sc = lookup.get(f"{r['qid']}#{r['pass']}", {})
                    g = {"qid": r["qid"], "pass": r["pass"], "type": r["type"],
                         "accuracy": sc.get("accuracy"), "reason": sc.get("reason", "")}
                    graded[(r["qid"], r["pass"])] = g
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"    type {qtype}: {min(s+GRADE_BATCH, len(rows))}/{len(rows)}")

    # ---- majority verdict
    per_q = defaultdict(list)
    for (qid, p), g in graded.items():
        if g.get("accuracy") is not None:
            per_q[qid].append(g["accuracy"])

    contaminated, review = [], []
    for q in screened:
        scores = per_q.get(q["qid"], [])
        if not scores:
            continue
        twos = sum(1 for s in scores if s == 2)
        ones = sum(1 for s in scores if s == 1)
        if twos > len(scores) / 2:
            contaminated.append({"qid": q["qid"], "type": q["type"], "scores": scores})
        elif twos + ones > len(scores) / 2:
            review.append({"qid": q["qid"], "type": q["type"], "scores": scores})

    verdict = {
        "judge_model": j.model, "judge_model_version": j.model_version,
        "passes": args.passes, "screened": len(screened),
        "rule": "delete on majority score 2; score 1 flagged for review; type 5 excluded",
        "contaminated_qids": sorted(c["qid"] for c in contaminated),
        "contaminated_detail": contaminated,
        "review_detail": review,
    }
    (RESULTS / "contaminated_qids.json").write_text(
        json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    n = len(contaminated)
    print(f"\n{'='*66}")
    print(f"CONTAMINATED: {n}/{len(screened)} = {100*n/len(screened):.1f}% "
          f"(README predicts 10-20%)")
    print(f"flagged for review (majority >=1): {len(review)}")
    print("\nby type:")
    ct = Counter(c["type"] for c in contaminated)
    tt = Counter(q["type"] for q in screened)
    for t in sorted(tt):
        print(f"  type {t}: {ct.get(t,0):>3}/{tt[t]:<3} = {100*ct.get(t,0)/tt[t]:>5.1f}%")
    print(f"\n-> results/contaminated_qids.json  (evalset/ NOT modified)")
    assert_no_key_in(RESULTS / "contaminated_qids.json")
    print(f"judge calls: {j.calls}, retries: {j.retries}")


if __name__ == "__main__":
    main()
