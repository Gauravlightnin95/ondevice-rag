"""
Stage 7 — Instrumented energy measurement.

Measures energy as a function of WORK (prefill tokens, decode tokens) per (model, device),
so the resulting model can be applied to the work the Stage 6 grid actually logged. A pure
re-run pass cannot do that: Stage 6 found only 17.6% of iGPU/NPU pairs agree, so a re-run
produces a different generation with a different token count, and its energy would not
correspond to the row Stage 8 grades.

Prefill and decode are separated WITHOUT needing sub-100ms sampling. Instead of trying to
catch the first token, the same prompt is run at several max_new_tokens values:

    E(n_prompt, m_tokens) = E_prefill(n_prompt) + m * e_decode

so a regression over m gives per-token decode energy as the slope and prefill energy as
the intercept. That is robust to the accumulator's resolution and gives the model's
coefficients directly, rather than inferring them from a phase split.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\inference\\measure_energy.py --mode verify
    .venv\\Scripts\\python.exe src\\inference\\measure_energy.py --mode components
    .venv\\Scripts\\python.exe src\\inference\\measure_energy.py --mode sweep
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "retrieval"))

import prompts as P                                             # noqa: E402
from pipeline import build_prompt, make_config, make_pipeline   # noqa: E402
from power import (RAILS, EnergyReader, UNITS_PER_JOULE,        # noqa: E402
                   delta, measure_idle, power_state)
from run_grid import (GRID_DEVICES, MODELS, NPU_MAX_PROMPT_LEN,  # noqa: E402
                      load_chunk_text, load_questions)

RESULTS = Path("results")

# Prefill lengths, clustered around the ~1,000-token transition where Stage 6's NPU TTFT
# steps (651->659->2062 ms across the 300-1000 / 1000-2500 boundary). Spreading these
# evenly across 49-4192 would leave the step unresolved, and the step is the thing that
# decides whether a linear form is defensible.
PREFILL_POINTS = [49, 143, 384, 640, 896, 1152, 1536, 2048, 2905, 4192]
DECODE_POINTS = [1, 16, 48, 96, 192, 384]
REPS = 3
FIXED_PROMPT_TOKENS = 143          # decode sweep holds prefill at the Oracle-RAG median

# Gate A established that iGPU TTFT is thermally sensitive by up to ~50%: the same 8B
# closed-book query measured 83 ms in the grid, 127 ms immediately after a CPU load
# ladder, and 81 ms once the machine had settled. Energy follows power follows clocks, so
# measuring a hot machine would inflate whichever cells happened to run while hot and bias
# the comparison between them. Two controls:
#   - a cooldown before each (model, device) block, held until package power settles
#   - randomised order of sweep points WITHIN a block, so any residual drift becomes noise
#     spread across prefill lengths rather than a systematic ramp along them
COOLDOWN_S = 45
COOLDOWN_SETTLE_W = 3.0            # proceed once package power is back near the ~2.4 W idle

# The PDH energy accumulator quantises at exactly 0.1 J (measured: GCD of 60 short idle
# deltas). Single-call measurements are therefore dominated by the tick, so each point
# repeats distinct prompts until the window clears TARGET_WINDOW_J.
TARGET_WINDOW_J = 5.0
MIN_REPS = 1                       # a single expensive call can already clear the tick
MAX_REPS = 200
POOL = 12                          # distinct prompts per length, rotated within a window


def build_padded_prompt(tok, blob, target_tokens, offset_frac=0.0):
    """A prompt of ~target_tokens built from real corpus text starting at offset_frac.

    Real text rather than repeated filler: tokenisation and therefore compute behaviour
    depend on content, and filler would not represent the grid's actual prompts.

    offset_frac exists because EVERY measurement must use a DISTINCT prompt sharing no
    prefix with any other. The stateful OpenVINO pipeline reuses KV cache across
    generate() calls, and truncating one blob at different lengths makes every prompt a
    prefix of the longer ones. Measured before this was fixed: the same 2,904-token prompt
    cost 222 ms, then 36 ms, then 12.9 ms on successive runs — an 17x collapse that would
    have been read as prefill energy saturating with length, and would have chosen the
    wrong functional form.
    """
    start = int(len(blob) * offset_frac) % max(len(blob) - 1, 1)
    window = blob[start:] + blob[:start]            # rotate so every offset is unique
    lo, hi, best = 0, len(window), ""
    while lo < hi:                                  # binary search on character length
        mid = (lo + hi + 1) // 2
        candidate = P.build_user_content(P.RAG, "What does the document state?",
                                         passages=[window[:mid]])
        n = len(tok.encode(build_prompt(tok, candidate)).input_ids.data[0])
        if n <= target_tokens:
            best, lo = candidate, mid
        else:
            hi = mid - 1
    return build_prompt(tok, best or P.build_user_content(
        P.CLOSED_BOOK, "What does the document state?"))


def cooldown(reader, max_wait=180.0):
    """Wait until package power settles, so a hot machine does not inflate the block.

    Returns the settled idle baseline, which is also the per-block idle used for marginal
    energy — measuring it here rather than once globally means it tracks thermal state.
    """
    t0 = time.time()
    while True:
        idle = measure_idle(reader, 10.0)
        w = idle["pkg_per_s"] / UNITS_PER_JOULE
        if w <= COOLDOWN_SETTLE_W or time.time() - t0 > max_wait:
            return idle, w, time.time() - t0
        time.sleep(5.0)


def measure_batched(reader, calls, idle_per_s, target_j=TARGET_WINDOW_J,
                    min_reps=MIN_REPS, max_reps=MAX_REPS):
    """Energy per call, averaged over enough calls to clear the accumulator's tick.

    The PDH energy counter quantises at exactly 0.1 J. A single short call is far below
    that — a 49-token prefill costs ~0.02 J, so one measurement is mostly quantisation
    noise and even a 2 J call carries 5% tick error. Repeating until the window
    accumulates target_j pushes the tick down to a few percent of the total.

    `calls` is an iterable of zero-arg callables, each using a DISTINCT prompt: repeating
    one prompt would be served from KV cache and measure nothing.
    """
    it = iter(calls)
    a = reader.read()
    n, last = 0, None
    while n < max_reps:
        try:
            last = next(it)()
        except StopIteration:
            break
        n += 1
        if n >= min_reps:  # check after every call so costly points stop early
            b = reader.read()
            if (b["pkg"] - a["pkg"]) / UNITS_PER_JOULE >= target_j:
                break
    b = reader.read()
    d = delta(a, b)
    out = {"seconds": d["seconds"] / n, "reps": n, "window_seconds": d["seconds"],
           "result": last}
    for rail in list(RAILS) + ["residual"]:
        total = d[f"{rail}_j"]
        marginal = total - (idle_per_s[f"{rail}_per_s"] / UNITS_PER_JOULE) * d["seconds"]
        out[f"{rail}_j"] = total / n
        out[f"{rail}_marginal_j"] = marginal / n
    out["tick_error_pct"] = 100 * 0.1 / max(d["pkg_j"], 1e-9)
    return out


def sweep(out_path):
    blob = "\n\n".join(load_chunk_text().values())
    rows = []
    rng = random.Random(0)
    with EnergyReader() as reader:
        for device in GRID_DEVICES:
            for model in MODELS:
                print(f"\n=== {model} on {device} ===")
                idle, w, waited = cooldown(reader)
                print(f"  cooled to {w:.2f} W after {waited:.0f}s  "
                      f"(residual {idle['residual_per_s']/UNITS_PER_JOULE:.2f} W)")

                mpl = NPU_MAX_PROMPT_LEN if device == "NPU" else None
                t0 = time.time()
                pipe = make_pipeline(MODELS[model], device, max_prompt_len=mpl)
                tok = pipe.get_tokenizer()
                print(f"  compiled in {time.time()-t0:.1f}s")

                # Interleave prefill and decode points in randomised order so residual
                # thermal drift lands as noise across the sweep rather than as a ramp
                # correlated with prompt length — which would look exactly like a
                # super-linear prefill cost and corrupt the functional-form choice.
                plan = ([("prefill", n, r) for n in PREFILL_POINTS for r in range(REPS)]
                        + [("decode", m, r) for m in DECODE_POINTS for r in range(REPS)])
                rng.shuffle(plan)

                # Prompt pools are built ONCE, up front, never inside a measured window:
                # build_padded_prompt runs a tokenising binary search, and that CPU work
                # would otherwise be charged to inference energy.
                lengths = sorted(set(PREFILL_POINTS) | {FIXED_PROMPT_TOKENS})
                print(f"  building prompt pools ({len(lengths)} lengths x {POOL} each)...")
                pool = {L: [build_padded_prompt(tok, blob, L, offset_frac=(k + 1) / (POOL + 1))
                            for k in range(POOL)] for L in lengths}

                for i, (kind, size, rep) in enumerate(plan, 1):
                    length = size if kind == "prefill" else FIXED_PROMPT_TOKENS
                    if kind == "prefill":
                        cfg = make_config(1)
                    else:
                        cfg = make_config(size)
                        cfg.ignore_eos = True      # force the full length, else the slope
                                                   # is fitted over uncontrolled lengths

                    # Rotate through the pool so no prompt repeats until POOL calls later,
                    # far beyond anything the pipeline holds in KV cache.
                    def calls(prompts=pool[length], cfg=cfg, start=rng.randrange(POOL)):
                        j = 0
                        while True:
                            p = prompts[(start + j) % len(prompts)]
                            yield lambda p=p, c=cfg: pipe.generate([p], c)
                            j += 1

                    m = measure_batched(reader, calls(), idle)
                    pm = m.pop("result").perf_metrics
                    rows.append({"kind": kind, "model": model, "device": device,
                                 "target": size, "rep": rep,
                                 "prefill_tokens": int(pm.get_num_input_tokens()),
                                 "generated_tokens": int(pm.get_num_generated_tokens()),
                                 "idle_pkg_w": w, "order": i, **m})
                    if i % 12 == 0 or i == len(plan):
                        print(f"    {i}/{len(plan)}  (last window {m['reps']} reps, "
                              f"tick err {m['tick_error_pct']:.1f}%)")

                del pipe
                with out_path.open("w", encoding="utf-8", newline="\n") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

                for kind, pts in (("prefill", PREFILL_POINTS), ("decode", DECODE_POINTS)):
                    cell = [r for r in rows if r["model"] == model
                            and r["device"] == device and r["kind"] == kind]
                    line = "  ".join(
                        f"{p}:{statistics.median([r['pkg_marginal_j'] for r in cell if r['target']==p] or [0]):.1f}J"
                        for p in pts)
                    print(f"    {kind:<8} {line}")
    return rows


def components(out_path):
    """Retrieval-side energy: embedding, FAISS, BM25, fusion.

    README section 7 counts the embedding model and index as part of RAG's cost, not just
    the LLM. Each is far below the accumulator's noise floor for a single call, so each is
    measured over many repetitions and divided.
    """
    from retrieve import _load, embed_query, search
    _load()
    questions = [q["question"] for q in load_questions()][:50]
    reps = 200
    rows = []
    with EnergyReader() as reader:
        idle = measure_idle(reader, 8.0)
        print(f"idle pkg {idle['pkg_per_s']/UNITS_PER_JOULE:.2f} W\n")

        def bench(label, fn):
            fn()                                   # warm
            m = measure(reader, lambda: [fn() for _ in range(reps)], idle)
            m.pop("result")
            per = {k: v / reps for k, v in m.items() if k.endswith("_j")}
            rows.append({"component": label, "reps": reps,
                         "seconds_total": m["seconds"], **per})
            print(f"  {label:<22} {per['pkg_marginal_j']*1000:>8.3f} mJ/call  "
                  f"({m['seconds']/reps*1000:.2f} ms)")

        bench("query_embedding", lambda: embed_query(questions[0]))
        bench("search_full_rrf", lambda: search(questions[0], k=5))
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def verify():
    """Per-device discrimination: idle must separate from a real model workload."""
    blob = "\n\n".join(load_chunk_text().values())
    print("Discrimination per device — idle vs a real inference workload\n")
    with EnergyReader() as reader:
        for device in GRID_DEVICES:
            idle = measure_idle(reader, 6.0)
            iw = idle["pkg_per_s"] / UNITS_PER_JOULE
            mpl = NPU_MAX_PROMPT_LEN if device == "NPU" else None
            pipe = make_pipeline(MODELS["qwen3-1.7b"], device, max_prompt_len=mpl)
            tok = pipe.get_tokenizer()
            prompt = build_padded_prompt(tok, blob, 2905)
            cfg = make_config(128)
            cfg.ignore_eos = True
            a = reader.read()
            pipe.generate([prompt], cfg)
            b = reader.read()
            d = delta(a, b)
            lw = d["pkg_j"] / d["seconds"]
            rw = d["residual_j"] / d["seconds"]
            ri = idle["residual_per_s"] / UNITS_PER_JOULE
            print(f"  {device}:  idle {iw:5.2f} W -> load {lw:5.2f} W  ratio {lw/iw:.2f}x  "
                  f"{'PASS' if lw/iw > 1.2 else 'FAIL'}")
            print(f"        residual {ri:5.2f} W -> {rw:5.2f} W  ratio {rw/ri:.2f}x")
            del pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "components", "verify"], required=True)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ps = power_state()
    print(f"power state: {'AC' if ps.get('ac') else 'BATTERY'}   "
          f"units/J: {UNITS_PER_JOULE:,.0f}")
    if not ps.get("ac"):
        raise SystemExit("ABORT: measure on AC. Battery power limits change both timing "
                         "and energy, and the grid is being characterised on AC.")

    if args.mode == "verify":
        verify()
    elif args.mode == "components":
        components(RESULTS / f"energy_components_{ts}.jsonl")
    else:
        out = RESULTS / f"energy_sweep_{ts}.jsonl"
        rows = sweep(out)
        print(f"\n{len(rows)} measurements -> {out}")


if __name__ == "__main__":
    main()
