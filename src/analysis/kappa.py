"""
Stage 8 — Cohen's kappa: blind worksheet export, then join and compute.

THE HUMAN GRADES FIRST. Grading after seeing the judge's scores would measure agreement
with a prior the annotator has already been anchored to, inflating kappa by an unknown
amount - and an inflated kappa is worse than none, because it is reported as validation.

The ordering is enforced mechanically rather than by discipline:

  export   writes a worksheet with NO judge score, model, device or condition, and salted
           row ids that cannot be matched back to the grid by eye
  compute  refuses to run until the filled worksheet exists, and only then reads the
           judge's scores for those rows

Sample: 300 rows stratified by type x model, WEIGHTED TOWARD TYPES 4 AND 5. Those need the
judge to notice something absent - a conflict between passages, or that an answer is not in
the corpus - which is where a lite judge is likeliest to fall back on "did it answer the
question", and therefore where kappa most needs statistical power. Types 1 and 3 are
near-mechanical comparisons that would show high agreement regardless.

README section 7 asks for 20% (1,070 rows). At 30-60s of careful grading each that is 9-18
hours for one annotator. 300 rows gives an overall kappa with a 95% CI near +/-0.08-0.10 -
enough to separate "substantial" from "moderate", which is the claim the paper needs - at
about 2.5-5 hours. Per-type kappa is reported only WITH its wide CI, never as a headline.

LIMITATION, stated rather than implied: the same person wrote the gold answers and grades
this sample. Kappa here measures judge-versus-annotator agreement where the annotator
authored the reference. It is not independent inter-annotator agreement, and it is an
upper bound on what independent double-grading would give.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\kappa.py --mode export
    .venv\\Scripts\\python.exe src\\analysis\\kappa.py --mode compute
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grade import latest, load_rows                            # noqa: E402
from rubric import FORBIDDEN_PATTERNS                          # noqa: E402

RESULTS = Path("results")
WORKSHEET = RESULTS / "kappa_worksheet.csv"
# The annotator's returned file. Either name is accepted; the first that exists wins.
FILLED_CANDIDATES = (RESULTS / "kappa_grades.csv",
                     RESULTS / "kappa_worksheet_filled.csv")
FILLED = next((p for p in FILLED_CANDIDATES if p.exists()), FILLED_CANDIDATES[0])
KEYMAP = RESULTS / "kappa_keymap.json"          # salted id -> row_id, never shown to grader
SEED = 20260906
N = 300
# Weighted to types 4 and 5: see the module docstring.
WEIGHTS = {4: 75, 5: 75, 1: 40, 2: 40, 3: 40, 6: 30}


def salt(row_id):
    return "K" + hashlib.sha256(f"{SEED}:{row_id}".encode()).hexdigest()[:10].upper()


def export():
    rows = [r for r in load_rows() if r["answer"]]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    rng = random.Random(SEED)
    sample = []
    for t, want in WEIGHTS.items():
        pool = sorted(by_type.get(t, []), key=lambda r: r["row_id"])
        # stratify within type across the four models so all four are covered
        by_model = defaultdict(list)
        for r in pool:
            by_model[r["model"]].append(r)
        per = max(want // max(len(by_model), 1), 1)
        picked = []
        for m in sorted(by_model):
            picked.extend(rng.sample(by_model[m], min(per, len(by_model[m]))))
        rng.shuffle(picked)
        sample.extend(picked[:want])
    rng.shuffle(sample)                       # order carries no signal either

    keymap = {salt(r["row_id"]): r["row_id"] for r in sample}
    KEYMAP.write_text(json.dumps(keymap, indent=2) + "\n", encoding="utf-8")

    with WORKSHEET.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "gold_answer", "model_answer", "context",
                    "your_accuracy_0_1_2"])
        for r in sample:
            ctx = "\n\n".join(f"[{cid}] {text}" for cid, text in r["context"])
            w.writerow([salt(r["row_id"]), r["question"], r["gold"], r["answer"],
                        ctx, ""])

    # The worksheet must be as blind as the judge's prompt.
    import re
    text = WORKSHEET.read_text(encoding="utf-8").lower()
    hits = [p for p in FORBIDDEN_PATTERNS if re.search(p, text)]
    assert not hits, f"worksheet is not blind, matched: {hits}"

    print(f"exported {len(sample)} rows -> {WORKSHEET}")
    print(f"  by type : {dict(sorted(Counter(r['type'] for r in sample).items()))}")
    print(f"  by model: {dict(sorted(Counter(r['model'] for r in sample).items()))}")
    print(f"  blind   : no judge score, model, device or condition present")
    print(f"\nFill in your_accuracy_0_1_2 and save as:\n  {FILLED}")
    print("Grade before looking at any judge output for these rows — kappa measures")
    print("agreement, and agreement with a score you have already seen is not agreement.")


def kappa(a, b, categories=(0, 1, 2)):
    """Cohen's kappa, unweighted."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in categories)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def boot_ci(a, b, n=2000, seed=0):
    rng = random.Random(seed)
    idx = range(len(a))
    ks = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        k, _ = kappa([a[i] for i in s], [b[i] for i in s])
        ks.append(k)
    ks.sort()
    return ks[int(0.025 * n)], ks[int(0.975 * n)]


def compute():
    if not FILLED.exists():
        raise SystemExit(
            f"{FILLED} not found.\n"
            "The human grades come FIRST. Run --mode export, grade the worksheet, save it\n"
            "as the filled file, then run compute. This refusal is the mechanism that\n"
            "stops kappa being measured against a judge score already seen.")

    keymap = json.loads(KEYMAP.read_text(encoding="utf-8"))
    human = {}
    with FILLED.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("your_accuracy_0_1_2") or "").strip()
            if v != "":
                human[r["id"]] = int(v)
    missing = len(keymap) - len(human)
    if missing > 0:
        print(f"WARNING: {missing} of {len(keymap)} rows ungraded — kappa uses the "
              f"{len(human)} that are filled")

    grades = {}
    for line in latest("grades_*.jsonl").open(encoding="utf-8"):
        g = json.loads(line)
        grades[g["row_id"]] = g

    pairs, meta = [], []
    for sid, score in human.items():
        rid = keymap.get(sid)
        g = grades.get(rid)
        if g and g.get("accuracy") is not None:
            pairs.append((score, g["accuracy"]))
            meta.append(g)
    if not pairs:
        raise SystemExit("no overlap between the worksheet and the judge's grades")

    h = [p[0] for p in pairs]
    m = [p[1] for p in pairs]
    k, po = kappa(h, m)
    lo, hi = boot_ci(h, m)

    # Binary kappa, collapsing 1 into 0. The human grades are close to binary - only 17 of
    # 292 are 1 - so a three-way disagreement can reflect where the boundary between
    # "partially correct" and "wrong" was drawn rather than a disagreement about whether
    # the answer was right. Binary kappa asks the question the headline accuracy actually
    # rests on: did the judge and the human agree the answer was fully correct?
    hb = [1 if x == 2 else 0 for x in h]
    mb = [1 if x == 2 else 0 for x in m]
    kb, pob = kappa(hb, mb, categories=(0, 1))
    lob, hib = boot_ci(hb, mb)

    jm = json.loads(latest("grades_*.meta.json").read_text(encoding="utf-8"))
    print(f"Cohen's kappa — judge vs human, {len(pairs)} rows")
    print(f"  judge model      : {jm.get('judge_model')} "
          f"({jm.get('judge_model_version')})")
    print(f"\n  THREE-WAY (0/1/2)")
    print(f"    raw agreement  : {po:.3f}")
    print(f"    kappa          : {k:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"\n  BINARY (2 vs not-2, collapsing 1 into 0)")
    print(f"    raw agreement  : {pob:.3f}")
    print(f"    kappa          : {kb:.3f}   95% CI [{lob:.3f}, {hib:.3f}]")
    print(f"    human 1s       : {sum(1 for x in h if x == 1)}/{len(h)} "
          f"— near-binary, which is why this is reported")

    # ---- direction of disagreement
    print(f"\n  DIRECTION OF DISAGREEMENT")
    print(f"    mean human score : {statistics.mean(h):.3f}")
    print(f"    mean judge score : {statistics.mean(m):.3f}  "
          f"(judge - human = {statistics.mean(m)-statistics.mean(h):+.3f})")
    lenient = sum(1 for x, y in zip(h, m) if y > x)
    harsher = sum(1 for x, y in zip(h, m) if y < x)
    print(f"    judge scored HIGHER than human: {lenient} rows")
    print(f"    judge scored LOWER  than human: {harsher} rows")
    if lenient + harsher:
        skew = lenient / (lenient + harsher)
        print(f"    of disagreements, {100*skew:.0f}% are the judge being more lenient")
        verdict_dir = ("judge is systematically MORE LENIENT" if skew > 0.65 else
                       "judge is systematically HARSHER" if skew < 0.35 else
                       "no systematic direction")
        print(f"    -> {verdict_dir}")
    else:
        skew, verdict_dir = None, "no disagreements"

    print(f"\n    confusion (rows = human, cols = judge):")
    print(f"      {'':>6}" + "".join(f"{'J'+str(c):>6}" for c in (0, 1, 2)))
    for hv in (0, 1, 2):
        row = [sum(1 for x, y in zip(h, m) if x == hv and y == jv) for jv in (0, 1, 2)]
        print(f"      {'H'+str(hv):>6}" + "".join(f"{v:>6}" for v in row))

    print("\n  per type (indicative only — CIs are wide at this n):")
    for t in sorted({g["type"] for g in meta}):
        sub = [(x, y) for (x, y), g in zip(pairs, meta) if g["type"] == t]
        if len(sub) < 10:
            continue
        kt, pot = kappa([s[0] for s in sub], [s[1] for s in sub])
        lot, hit = boot_ci([s[0] for s in sub], [s[1] for s in sub], n=1000)
        print(f"    type {t}: n={len(sub):>3}  kappa {kt:>6.3f}  "
              f"95% CI [{lot:.3f}, {hit:.3f}]")

    verdict = ("SUBSTANTIAL" if k >= 0.61 else "MODERATE" if k >= 0.41 else "WEAK")
    print()
    print(f"\n  verdict: {verdict}")
    if k < 0.6:
        print("  kappa below 0.6 — README section 7's fallback applies: tighten the")
        print("  rubric or move to human grading. This is a decision on the measured")
        print("  number, not something to explain away.")

    out = {"judge_model": jm.get("judge_model"),
           "judge_model_version": jm.get("judge_model_version"),
           "n": len(pairs),
           "kappa_three_way": k, "ci95_three_way": [lo, hi], "raw_agreement": po,
           "kappa_binary": kb, "ci95_binary": [lob, hib],
           "raw_agreement_binary": pob,
           "human_ones": sum(1 for x in h if x == 1),
           "mean_human": statistics.mean(h), "mean_judge": statistics.mean(m),
           "judge_higher_rows": lenient, "judge_lower_rows": harsher,
           "lenience_skew": skew, "direction": verdict_dir,
           "verdict": verdict,
           "limitation": ("The annotator authored the gold answers. This is "
                          "judge-versus-annotator agreement, not independent "
                          "inter-annotator agreement, and is an upper bound on what "
                          "independent double-grading would produce.")}
    (RESULTS / "kappa.json").write_text(json.dumps(out, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"\n-> results/kappa.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "compute"], required=True)
    a = ap.parse_args()
    export() if a.mode == "export" else compute()
