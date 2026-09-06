"""
Stage 8 — Verify the judge can fail, before any grade is trusted.

A judge that scores everything 1 is not measuring anything. This constructs cases where
the correct score is known by design and checks the judge returns it:

  gold answers VERBATIM                    -> must score 2
  deliberately wrong answers               -> must score 0
  type 4 given one confident figure        -> must score 0 (the figure IS in the corpus,
                                              which is exactly the trap)
  type 5 given a confident plausible answer-> must score 0 (hallucination)
  type 5 given an explicit refusal         -> must score 2

The last three are the ones that matter. Types 1/2/3/6 are near-mechanical comparisons any
judge handles; types 4 and 5 invert the usual "did it answer" instinct and are where a
generic judge silently does the wrong thing.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\verify_judge.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from judge import Judge                                        # noqa: E402
from rubric import assert_blind, build_item, build_prompt      # noqa: E402

EVALSET = Path("evalset/evalset_v1.jsonl")
WRONG = "The figure is 47 units, as stated on page 12 of the regulations."
CONFIDENT_T5 = ("The minimum required is 65%, as specified in the academic "
                "regulations under the attendance policy.")
REFUSAL = ("That information is not stated anywhere in the provided documents. "
           "The corpus does not specify it.")


def main():
    qs = [json.loads(l) for l in EVALSET.open(encoding="utf-8")]
    by_type = defaultdict(list)
    for q in qs:
        by_type[q["type"]].append(q)

    cases = []          # (label, qtype, question, gold, answer, expected)
    for t in (1, 2, 3, 6):
        for q in by_type[t][:2]:
            cases.append((f"type{t} verbatim gold", t, q["question"], q["answer"],
                          q["answer"], 2))
            cases.append((f"type{t} wrong answer", t, q["question"], q["answer"],
                          WRONG, 0))
    for q in by_type[4][:3]:
        cases.append(("type4 both sides (gold)", 4, q["question"], q["answer"],
                      q["answer"], 2))
        # One side, stated confidently. The figure is real; the failure is the confidence.
        one_side = q["passage"].strip()[:200]
        cases.append(("type4 ONE figure confidently", 4, q["question"], q["answer"],
                      one_side, 0))
    for q in by_type[5][:3]:
        cases.append(("type5 explicit refusal", 5, q["question"], q["answer"],
                      REFUSAL, 2))
        cases.append(("type5 confident hallucination", 5, q["question"], q["answer"],
                      CONFIDENT_T5, 0))

    j = Judge()
    print(f"judge: {j.model}   {len(cases)} constructed cases\n")

    results = []
    by_t = defaultdict(list)
    for i, c in enumerate(cases):
        by_t[c[1]].append((i, c))
    for qtype, group in sorted(by_t.items()):
        items = [build_item(str(i), c[2], c[3], c[4]) for i, c in group]
        prompt = build_prompt(qtype, items)
        assert_blind(prompt)
        scores, _ = j.ask_json(prompt, max_output_tokens=16384)
        lookup = {str(s.get("id")): s for s in scores}
        for i, c in group:
            got = lookup.get(str(i), {}).get("accuracy")
            results.append((c[0], c[5], got, lookup.get(str(i), {}).get("reason", "")))

    print(f"{'case':<34}{'expect':>7}{'got':>5}   verdict")
    ok = 0
    for label, exp, got, reason in results:
        good = got == exp
        ok += good
        print(f"  {label:<32}{exp:>7}{str(got):>5}   {'ok' if good else 'MISMATCH'}"
              + ("" if good else f"  ({reason[:60]})"))

    print(f"\n{ok}/{len(results)} constructed cases scored as designed")

    spread = {r[2] for r in results}
    print(f"distinct scores returned: {sorted(x for x in spread if x is not None)}")
    if len(spread) < 2:
        raise SystemExit("FAIL: judge returned a single score for everything — "
                         "it is not discriminating")

    critical = [r for r in results if "type4 ONE" in r[0] or "type5 confident" in r[0]
                or "type5 explicit" in r[0]]
    cok = sum(1 for r in critical if r[1] == r[2])
    print(f"critical type-4/type-5 cases: {cok}/{len(critical)}")
    if cok < len(critical):
        raise SystemExit("FAIL: judge does not apply the type-4/type-5 criteria — "
                         "the rubric is not landing and grades would be wrong on the "
                         "questions H2 depends on")
    print("\nPASS — judge discriminates, and applies the inverted type-4/5 criteria")


if __name__ == "__main__":
    main()
