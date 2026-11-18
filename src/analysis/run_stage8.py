"""
Stage 8 orchestrator — budget-gated phases.

The daily cap is 500 measured calls, and a phase that stalls part-way spends calls on work
that has to be redone. So each phase is checked against the REMAINING budget before it
starts, and a phase that cannot finish is not begun.

Order matters:

  A  contamination passes 2-3  ~435 calls  the majority rule needs 3 passes; pass 1 alone
                                           would delete from a frozen eval set on one roll
  C  full grading UNBATCHED   ~4326 calls  the batch check said batching changes the
                                           measurement, so one call per row
  D  kappa                       0 calls   local join, needs the human grades already filed

Phase B (the batch check) has run and is not repeated. Its verdict was GRADE UNBATCHED:
87.0% agreement with a -0.030 mean difference overall, but agreement by batch position ran
0.625 / 0.750 / 0.333 - collapsing in the last third of each batch. See
docs/grading_notes.md.

Budget gating differs by phase. Phase A is all-or-nothing because a partial set of passes
cannot support a majority verdict. Phase C is NOT: unbatched grading has no batches, so
every row is independent, a stall costs only time, and --resume continues exactly where it
stopped. Demanding all 4,326 calls up front would simply stop the job ever starting.

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
from judge import (Budget, Judge, RPD, _load_keys,             # noqa: E402
                   reconcile_budget)

PY = sys.executable
RESULTS = Path("results")

# (label, estimated calls, command, all_or_nothing)
#
# Phase C is NO LONGER all-or-nothing. That rule existed because restarting a BATCHED
# grade mid-run changes batch composition and breaks the position analysis. Unbatched
# grading has no batches: every row is an independent call, so a stall costs only time and
# --resume picks up exactly where it stopped. Requiring the full 4,326 calls up front
# would just prevent the job from ever starting.
PHASES = [
    ("A  contamination passes 2-3", 435,
     [PY, "src/analysis/screen_contamination.py", "--passes", "3", "--resume"], True),
    ("C  full grading (UNBATCHED)", 4326,
     [PY, "src/analysis/grade.py", "--mode", "full", "--batch-size", "1", "--resume"],
     False),
]


def quota_available():
    """One probe. Success also rolls the ledger over if it was marked exhausted."""
    try:
        Judge().ask("Reply with exactly: OK", max_output_tokens=16)
        return True, None
    except Exception as exc:
        return False, str(exc)[:160]


KEY_NAMES = [n for n, _ in _load_keys()]


def show(b, label=""):
    live = [n for n in KEY_NAMES if not b.is_exhausted(n)]
    print(f"  ring: {b.total_remaining(KEY_NAMES)} calls remaining across "
          f"{len(live)}/{len(KEY_NAMES)} live keys"
          + (f"   {label}" if label else ""))
    for n in KEY_NAMES:
        flag = "  EXHAUSTED" if b.is_exhausted(n) else ""
        print(f"      {n:<20} {b.used(n):>4}/{RPD}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-for-reset", action="store_true",
                    help="poll until the daily quota rolls over, then run")
    ap.add_argument("--poll-minutes", type=int, default=20)
    ap.add_argument("--max-wait-hours", type=float, default=14)
    args = ap.parse_args()

    b = Budget()
    print(f"Stage 8 orchestrator — {RPD}/day per key, {len(KEY_NAMES)} keys")
    show(b)

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

    for label, cost, cmd, all_or_nothing in PHASES:
        # Clear stale exhaustion flags before gating. The ledger only self-heals on a
        # successful call, but the gate runs first, so after a rollover it would refuse a
        # phase the ring can afford.
        b = reconcile_budget()
        print(f"\n{'='*66}\n{label}   estimated {cost} calls"
              + ("" if all_or_nothing else "  (resumable — partial progress is kept)"))
        show(b)
        need = cost if all_or_nothing else 50
        if not b.can_afford(KEY_NAMES, need):
            print(f"  SKIPPED — {b.total_remaining(KEY_NAMES)} remaining, needs {need}.")
            print("  Not started: a partial phase spends calls on work that must be "
                  "redone." if all_or_nothing else
                  "  Ring is spent; rerun after a key rolls over.")
            break

        # phase A resumes into whatever contamination file already exists
        run = list(cmd)
        if run[-1] == "--resume":
            pat = ("contamination_2*.jsonl" if "screen_contamination.py" in run[1]
                   else "grades_*.jsonl")
            existing = sorted(p for p in RESULTS.glob(pat)
                              if "graded" not in p.name and "meta" not in p.name)
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
        show(after, f"— phase used ~{b.total_remaining(KEY_NAMES) - after.total_remaining(KEY_NAMES)}")

    print(f"\n{'='*66}")
    final = Budget()
    show(final, "final")
    print("\nkappa (phase D, 0 calls) once grades exist:")
    print("  .venv\\Scripts\\python.exe src\\analysis\\kappa.py --mode compute")


if __name__ == "__main__":
    main()
