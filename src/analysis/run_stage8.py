"""
Stage 8 orchestrator — budget-gated phases.

The daily cap is 500 measured calls, and a phase that stalls part-way spends calls on work
that has to be redone. So each phase is checked against the REMAINING budget before it
starts, and a phase that cannot finish is not begun.

Order matters:

  A  contamination re-grade   ~17 calls   must precede grading; the neutral-prompt run
                                          only graded 12 of 194 before the quota died
  B  batch check             ~106 calls   its verdict decides how everything else is graded
  C  full grading            ~239 calls   only started if the whole phase fits
  D  kappa                      0 calls   local join, needs the human grades already filed

Phase C is deliberately all-or-nothing on the budget check. It is resumable, so a stall is
recoverable, but restarting mid-grade means the batch composition differs between runs and
the batch-position analysis stops being interpretable.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\run_stage8.py
    .venv\\Scripts\\python.exe src\\analysis\\run_stage8.py --wait-for-reset
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import Budget, Judge, RPD                           # noqa: E402

PY = sys.executable
RESULTS = Path("results")

PHASES = [
    ("A  contamination re-grade", 17,
     [PY, "src/analysis/screen_contamination.py", "--passes", "1", "--resume"]),
    ("B  batch check", 106,
     [PY, "src/analysis/grade.py", "--mode", "batchcheck"]),
    ("C  full grading", 239,
     [PY, "src/analysis/grade.py", "--mode", "full"]),
]


def quota_available():
    """One probe. Success also rolls the ledger over if it was marked exhausted."""
    try:
        Judge().ask("Reply with exactly: OK", max_output_tokens=16)
        return True, None
    except Exception as exc:
        return False, str(exc)[:160]


def show(b, label=""):
    print(f"  budget: {b.used}/{RPD} used, {b.remaining} remaining"
          + (f"   {label}" if label else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-reset", action="store_true",
                    help="poll until the daily quota rolls over, then run")
    ap.add_argument("--poll-minutes", type=int, default=20)
    ap.add_argument("--max-wait-hours", type=float, default=14)
    args = ap.parse_args()

    b = Budget()
    print(f"Stage 8 orchestrator — cap {RPD}/day")
    show(b, f"(exhausted at {b.state.get('exhausted_at')})"
            if b.state.get("exhausted_at") else "")

    ok, err = quota_available()
    if not ok and args.wait_for_reset:
        deadline = time.time() + args.max_wait_hours * 3600
        print(f"\nquota unavailable; polling every {args.poll_minutes} min "
              f"for up to {args.max_wait_hours:g}h")
        while time.time() < deadline:
            time.sleep(args.poll_minutes * 60)
            ok, err = quota_available()
            print(f"  {time.strftime('%H:%M:%S')}  "
                  f"{'RESET — quota available' if ok else 'still limited'}")
            if ok:
                break
    if not ok:
        raise SystemExit(f"\nquota still unavailable: {err}\n"
                         "All phases are resumable; rerun after the daily reset.")

    b = Budget()
    print()
    show(b, "after reset probe")

    for label, cost, cmd in PHASES:
        b = Budget()
        print(f"\n{'='*66}\n{label}   estimated {cost} calls")
        show(b)
        if not b.can_afford(cost):
            print(f"  SKIPPED — {b.remaining} remaining cannot finish a {cost}-call "
                  f"phase (margin included).")
            print("  Not started: a partial phase spends calls on work that must be "
                  "redone.")
            break

        # phase A resumes into whatever contamination file already exists
        run = list(cmd)
        if run[-1] == "--resume":
            existing = sorted(p for p in RESULTS.glob("contamination_2*.jsonl")
                              if "graded" not in p.name)
            if existing:
                run.append(str(existing[-1]))
            else:
                run = run[:-1]

        t0 = time.time()
        proc = subprocess.run(run, capture_output=True, text=True)
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-14:])
        print(tail)
        if proc.returncode != 0:
            print(f"  phase exited {proc.returncode}; "
                  f"{(proc.stderr or '').strip().splitlines()[-1:] or ['']}"[:200])
        print(f"  ({time.time()-t0:.0f}s)")

        after = Budget()
        show(after, f"— phase used ~{after.used - b.used}")

        # Phase B's verdict decides how phase C grades. Surface it explicitly.
        if label.startswith("B"):
            bc = sorted(RESULTS.glob("batchcheck_*.json"))
            if bc:
                v = json.loads(bc[-1].read_text(encoding="utf-8"))
                print(f"\n  BATCH CHECK VERDICT: {v.get('verdict')}   "
                      f"agreement {v.get('exact_agreement', 0):.3f}, "
                      f"mean diff {v.get('mean_score_diff_batched_minus_unbatched', 0):+.3f}")
                if v.get("verdict") != "BATCH":
                    print("  Batched grading changes the measurement. At 500/day the "
                          "unbatched\n  alternative is ~9 days, so this needs a decision "
                          "before phase C runs.")
                    break

    print(f"\n{'='*66}")
    final = Budget()
    show(final, "final")
    print("\nkappa (phase D, 0 calls) once grades exist:")
    print("  .venv\\Scripts\\python.exe src\\analysis\\kappa.py --mode compute")


if __name__ == "__main__":
    main()
