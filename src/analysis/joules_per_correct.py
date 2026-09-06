"""
Stage 8 — Joules per correct answer. README section 7's headline metric.

    accuracy        = mean(score) / 2          partial credit on the 0/1/2 rubric
    strict_accuracy = fraction scoring 2
    J per correct   = mean joules per query / accuracy

Reported BOTH ways and per cell, never aggregated. Partial and strict diverge most exactly
where the 0.6B sits, and an aggregate over question types would hide the H2 result.

Stage 7 found the 0.6B is the MOST expensive model per query on iGPU - 22.7 J against the
8B's 7.0 J - because it generates the full 512-token cap in every condition while the 8B
stops at ~37. If its accuracy is also lowest, joules per correct answer diverges far more
sharply than joules per query. That is H1's central claim and this is where it holds or
does not.

A cell whose accuracy rounds to zero reports "infinite" rather than a number divided by
something tiny. That is the honest rendering of "spends energy, answers nothing", and it
is a real outcome here rather than a hypothetical.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\joules_per_correct.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grade import latest                                       # noqa: E402

RESULTS = Path("results")
MODELS = ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-8b")
CONDS = ("closed_book", "oracle", "rag")


def main():
    grades = {}
    for line in latest("grades_*.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        grades[g["row_id"]] = g

    energy = {}
    for line in (RESULTS / "energy_per_query.jsonl").open(encoding="utf-8"):
        e = json.loads(line)
        energy[f"{e['model']}|{e['device']}|{e['condition']}|{e['qid']}"] = e

    joined = []
    for rid, g in grades.items():
        e = energy.get(rid)
        if e and g.get("accuracy") is not None:
            joined.append((g, e))
    print(f"{len(joined)} rows with both a grade and an energy estimate\n")

    cells = defaultdict(list)
    for g, e in joined:
        cells[(g["device"], g["model"], g["condition"])].append((g, e))

    out = {}
    print(f"{'dev':<5}{'model':<12}{'cond':<12}{'n':>5}{'acc':>7}{'strict':>8}"
          f"{'J/query':>9}{'J/correct':>11}{'J/strict':>11}")
    for key in sorted(cells):
        rows = cells[key]
        scores = [g["accuracy"] for g, _ in rows]
        acc = statistics.mean(scores) / 2
        strict = sum(1 for s in scores if s == 2) / len(scores)
        jq = statistics.mean(e["e_total_j"] for _, e in rows)
        jpc = jq / acc if acc > 0.005 else None
        jps = jq / strict if strict > 0.005 else None
        out["|".join(key)] = {
            "n": len(rows), "accuracy": round(acc, 4),
            "strict_accuracy": round(strict, 4),
            "joules_per_query": round(jq, 3),
            "joules_per_correct": round(jpc, 2) if jpc else "infinite",
            "joules_per_strict_correct": round(jps, 2) if jps else "infinite",
            "empty_outputs": sum(1 for g, _ in rows if g.get("empty_output")),
        }
        print(f"{key[0]:<5}{key[1]:<12}{key[2]:<12}{len(rows):>5}{acc:>7.3f}{strict:>8.3f}"
              f"{jq:>9.2f}"
              f"{(f'{jpc:>11.1f}' if jpc else f'{'infinite':>11}')}"
              f"{(f'{jps:>11.1f}' if jps else f'{'infinite':>11}')}")

    # per question type, since an average across types hides H2
    print(f"\n\nAccuracy by question type (partial credit), iGPU:")
    print(f"{'model':<12}{'cond':<12}" + "".join(f"{'t'+str(t):>8}" for t in (1,2,3,4,5,6)))
    bytype = defaultdict(list)
    for g, e in joined:
        bytype[(g["device"], g["model"], g["condition"], g["type"])].append(g["accuracy"])
    for m in MODELS:
        for c in CONDS:
            cells_ = []
            for t in (1, 2, 3, 4, 5, 6):
                v = bytype.get(("GPU", m, c, t), [])
                cells_.append(f"{statistics.mean(v)/2:>8.2f}" if v else f"{'-':>8}")
            print(f"{m:<12}{c:<12}" + "".join(cells_))

    (RESULTS / "joules_per_correct.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> results/joules_per_correct.json")


if __name__ == "__main__":
    main()
