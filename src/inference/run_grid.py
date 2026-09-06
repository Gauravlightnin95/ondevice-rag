"""
Stage 6 — Inference grid runner. 4 models x 3 conditions x 223 questions.

Three gates run before the grid is allowed to start, cheapest first, because the risk here
is not a wrong number but spending hours producing an unusable log:

    gate 0  freeze verification          seconds     abort on DRIFTED
    gate 1  smoke     20q x 1 model x 3   60 infer   asserts the wiring is right
    gate 2  agreement 20q x 2 models x 3 120 infer   justifies the single-device grid
    gate 3  full grid                   2676 infer

Rows are appended and flushed one at a time, so a crash costs the current query rather than
the run. --resume skips work already present in a partial file.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\inference\\run_grid.py --mode smoke
    .venv\\Scripts\\python.exe src\\inference\\run_grid.py --mode agreement
    .venv\\Scripts\\python.exe src\\inference\\run_grid.py --mode full
    .venv\\Scripts\\python.exe src\\inference\\run_grid.py --mode full --resume results\\grid_....jsonl
"""

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "retrieval"))

import prompts as P                                              # noqa: E402
from pipeline import (build_prompt, clean, count_tokens,         # noqa: E402
                      make_config, make_pipeline, thinking_tokens)
from power import power_state                                    # noqa: E402

MODELS = {
    "qwen3-0.6b": "models/qwen3-0.6b-cw",
    "qwen3-1.7b": "models/qwen3-1.7b-cw",
    "qwen3-4b": "models/qwen3-4b-cw",
    "qwen3-8b": "models/qwen3-8b-cw",
}
# Gate 2 showed the same model on the same byte-identical prompt gives DIFFERENT factual
# answers per device under greedy decoding, so device is an AXIS of the experiment, not a
# caveat. iGPU runs first so the primary results exist early; NPU follows via --resume.
# CPU is dropped: at 2703 ms/token (README section 3) it is not a deployment target for an
# on-device framing, and Gate 2 already characterises it.
GRID_DEVICE = "GPU"
GRID_DEVICES = ["GPU", "NPU"]
AGREEMENT_DEVICES = ["NPU", "GPU", "CPU"]
AGREEMENT_MODELS = ["qwen3-8b", "qwen3-0.6b"]   # the two ends of the failure spectrum
SMOKE_MODEL = "qwen3-1.7b"

K = 5                               # fixed a priori for the main grid; H3 sweeps it later
MAX_NEW_TOKENS = 512                # uniform across models — varying it would confound size

# NPU-only, frozen alongside k=5. The NPU statically allocates for a declared maximum
# prompt; at the default 1024 it cannot run k=5 RAG at all (prompts are 2889-4192 tokens).
# 4608 covers the largest observed RAG prompt with ~10% headroom. Sized for THIS grid, not
# for the later H3 k-sweep: the allocation tax scales with the buffer, so declaring 8192
# now would inflate the NPU latency this grid reports in order to serve a different
# experiment. The sweep gets its own configuration.
NPU_MAX_PROMPT_LEN = 4608
N_PROBE = 20

EVALSET = Path("evalset/evalset_v1.jsonl")
LOCATIONS = Path("evalset/passage_locations.jsonl")
CHUNKS = Path("corpus/chunks.jsonl")
RESULTS = Path("results")


# --------------------------------------------------------------------------- data

def load_questions():
    return [json.loads(l) for l in EVALSET.open(encoding="utf-8")]


def load_gold():
    """qid -> {'all': set(chunk_id), 'per_passage': [set, ...]} from stage 3."""
    gold = defaultdict(lambda: {"all": set(), "per_passage": []})
    for line in LOCATIONS.open(encoding="utf-8"):
        r = json.loads(line)
        gold[r["qid"]]["all"].update(r["chunk_ids"])
        gold[r["qid"]]["per_passage"].append(set(r["chunk_ids"]))
    return gold


def load_chunk_text():
    return {json.loads(l)["chunk_id"]: json.loads(l)["text"]
            for l in CHUNKS.open(encoding="utf-8")}


def probe_questions(questions, n=N_PROBE, seed=0):
    """Fixed stratified sample, weighted toward types 4 and 5 where length varies most."""
    want = {4: 6, 5: 6, 1: 2, 2: 2, 3: 2, 6: 2}
    by_type = defaultdict(list)
    for q in sorted(questions, key=lambda r: r["qid"]):
        by_type[q["type"]].append(q)
    rng = random.Random(seed)
    picked = []
    for t, count in want.items():
        picked.extend(rng.sample(by_type[t], min(count, len(by_type[t]))))
    return sorted(picked, key=lambda r: r["qid"])[:n]


# --------------------------------------------------------------------------- run

def run_one(pipe, tok, config, record, condition, retrieve_fn, chunk_text, gold,
            model_name, model_path, device, proc, no_think=True):
    """One inference. Returns the log row."""
    retrieved, passages, source, oracle_ids = [], None, P.SRC_NONE, []

    if condition == P.RAG:
        hits = retrieve_fn(record["question"], K)
        retrieved = [cid for cid, _ in hits]
        passages = [text for _, text in hits]
        source = P.SRC_RETRIEVED_TOPK
    elif condition == P.ORACLE:
        passages, source, oracle_ids = P.oracle_context(record, retrieve_fn)

    user = P.build_user_content(condition, record["question"], passages=passages)
    prompt = build_prompt(tok, user, no_think=no_think)

    g = gold.get(record["qid"], {"all": set(), "per_passage": []})
    top = set(retrieved)
    per = [bool(top & s) for s in g["per_passage"]]

    row = {
        "run_id": None, "model": model_name, "model_path": model_path,
        "device": device, "condition": condition,
        "k": K if condition == P.RAG else None,
        "qid": record["qid"], "type": record["type"],
        "retrieved_chunk_ids": retrieved,
        "gold_chunk_ids": sorted(g["all"]),
        "gold_hit_any": bool(top & g["all"]) if retrieved else None,
        "passage1_hit": per[0] if len(per) > 0 and retrieved else None,
        "passage2_hit": per[1] if len(per) > 1 and retrieved else None,
        "both_source_hit": (all(per) if len(per) > 1 else None) if retrieved else None,
        "oracle_context_source": source,
        "oracle_chunk_ids": oracle_ids,
        "no_think": no_think,
        "npu_max_prompt_len": NPU_MAX_PROMPT_LEN if device == "NPU" else None,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "error": None,
    }

    row["rss_before_mb"] = round(proc.memory_info().rss / 2**20, 1)
    try:
        t0 = time.perf_counter()
        # The prompt is passed as a LIST: generate(str, ...) returns a bare str with no
        # perf_metrics, so TTFT, prefill and the true generated-token count are only
        # available via the DecodedResults that a list argument returns.
        out = pipe.generate([prompt], config)
        wall_ms = (time.perf_counter() - t0) * 1000

        raw = out.texts[0]
        pm = out.perf_metrics
        n_gen = int(pm.get_num_generated_tokens())
        # Round first, then derive, so tpot_ms == wall_ms / generated_tokens holds exactly
        # against the stored values and the log is self-auditable.
        wall_ms = round(wall_ms, 3)

        row.update({
            "raw_output": raw,
            "cleaned_output": clean(raw),
            "raw_tokens": count_tokens(raw, tok),          # re-encoded, as test_npu.py did
            "generated_tokens": n_gen,                     # runtime truth, TPOT denominator
            "thinking_tokens": thinking_tokens(raw, tok),
            "prefill_tokens": int(pm.get_num_input_tokens()),
            "finish_reason": (str(out.finish_reasons[0])
                              if getattr(out, "finish_reasons", None) else None),
            "ttft_ms": round(float(pm.get_ttft().mean), 3),
            "generate_duration_ms": round(float(pm.get_generate_duration().mean), 3),
            # README section 7: elapsed over ACTUAL tokens generated, so a model that
            # refuses early does not look artificially fast.
            "tpot_ms": round(wall_ms / n_gen, 3) if n_gen else None,
            "tpot_runtime_ms": round(float(pm.get_tpot().mean), 3),
            "wall_ms": wall_ms,
        })
    except Exception as exc:
        row.update({"raw_output": "", "cleaned_output": "", "raw_tokens": None,
                    "generated_tokens": None, "thinking_tokens": None,
                    "prefill_tokens": None, "finish_reason": None, "ttft_ms": None,
                    "generate_duration_ms": None, "tpot_ms": None,
                    "tpot_runtime_ms": None, "wall_ms": None,
                    "error": f"{type(exc).__name__}: {exc}"})

    row["rss_after_mb"] = round(proc.memory_info().rss / 2**20, 1)
    mi = proc.memory_info()
    row["peak_wset_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 2**20, 1)
    return row


def key_of(row):
    return (row["model"], row["condition"], row["qid"], row["k"], row["device"],
            row.get("no_think", True))


def recompute(path):
    """Re-derive cleaned_output and thinking_tokens from raw_output, in place.

    Both are pure functions of raw_output, so a corrected clean() can be applied to an
    existing log without re-running a single inference. Used when the 8B's dropped
    </think> was found to make README section 2's clean() delete real answers.
    """
    import openvino_genai as ov_genai
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    toks = {}
    changed = 0
    for r in rows:
        if r.get("error") or not r.get("raw_output"):
            continue
        if r["model"] not in toks:
            toks[r["model"]] = ov_genai.Tokenizer(MODELS[r["model"]])
        tok = toks[r["model"]]
        new_clean = clean(r["raw_output"])
        new_think = thinking_tokens(r["raw_output"], tok)
        if new_clean != r.get("cleaned_output") or new_think != r.get("thinking_tokens"):
            changed += 1
        r["cleaned_output"] = new_clean
        r["thinking_tokens"] = new_think
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"recomputed {len(rows)} rows in {path}; {changed} changed")
    return rows


def execute(jobs, out_path, run_id, done, retrieve_fn, chunk_text, gold, no_think=True):
    """jobs: [(model_name, device, [(record, condition), ...]), ...] — model outer."""
    proc = psutil.Process()
    written, t_start = 0, time.time()
    # AC vs battery changes power limits and therefore timings. Stage 6 did not record it,
    # which is why Stage 7 opens with a check that its numbers were not throttled. Logged
    # per row from here on: an uncontrolled variable that was not recorded is worse than
    # one that was, because it cannot be checked later.
    ps = power_state()

    for model_name, device, work in jobs:
        pending = [(r, c) for r, c in work
                   if (model_name, c, r["qid"], K if c == P.RAG else None,
                       device, no_think) not in done]
        if not pending:
            print(f"  {model_name} on {device}: already complete, skipping")
            continue

        print(f"\n=== {model_name} on {device} — {len(pending)} inferences ===")
        t0 = time.time()
        mpl = NPU_MAX_PROMPT_LEN if device == "NPU" else None
        pipe = make_pipeline(MODELS[model_name], device, max_prompt_len=mpl)
        tok = pipe.get_tokenizer()
        config = make_config(MAX_NEW_TOKENS)
        print(f"  loaded and compiled in {time.time() - t0:.1f}s"
              + (f" (MAX_PROMPT_LEN={mpl})" if mpl else ""))

        for i, (record, condition) in enumerate(pending, 1):
            row = run_one(pipe, tok, config, record, condition, retrieve_fn,
                          chunk_text, gold, model_name, MODELS[model_name], device, proc,
                          no_think=no_think)
            row["run_id"] = run_id
            row["power_state"] = "AC" if ps.get("ac") else "battery"
            with out_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            written += 1
            if i % 10 == 0 or i == len(pending):
                rate = (time.time() - t_start) / written
                print(f"    {i}/{len(pending)}  {rate:.1f}s/query")

        del pipe
    return written


# --------------------------------------------------------------------------- gates

def gate_freeze(meta):
    sys.path.insert(0, str(HERE.parent / "retrieval"))
    from verify_frozen import verify
    print("Gate 0 — frozen retriever")
    report = verify(verbose=False)
    print(f"  {report.verdict}")
    meta["freeze"] = report.as_dict()
    if report.verdict == "DRIFTED":
        print(report.summary())
        raise SystemExit("ABORT: retriever drifted — results would not be comparable.")
    if report.verdict == "ARTIFACTS_MISSING":
        raise SystemExit("ABORT: frozen retriever artifacts missing — rebuild first.")
    if report.verdict == "BEHAVIOURALLY_IDENTICAL":
        print("  hashes moved but rankings held; 'changed' recorded in the run meta")
    return report


def check_smoke(rows):
    """Assertions, not just a run. A smoke test that cannot fail proves nothing."""
    fails = []
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    for cond in P.CONDITIONS:
        if not by_cond[cond]:
            fails.append(f"no rows for condition {cond}")
            continue
        if any(r["error"] for r in by_cond[cond]):
            bad = next(r for r in by_cond[cond] if r["error"])
            fails.append(f"{cond}: error on {bad['qid']}: {bad['error']}")
        if any(not r["cleaned_output"] for r in by_cond[cond]):
            n = sum(1 for r in by_cond[cond] if not r["cleaned_output"])
            fails.append(f"{cond}: {n} rows with empty cleaned output")

    # closed-book must receive no context, ever
    if any(r["retrieved_chunk_ids"] for r in by_cond[P.CLOSED_BOOK]):
        fails.append("closed_book rows carry retrieved chunks — the condition is not isolated")
    if any(r["oracle_context_source"] != P.SRC_NONE for r in by_cond[P.CLOSED_BOOK]):
        fails.append("closed_book rows carry oracle context")

    if any(not r["retrieved_chunk_ids"] for r in by_cond[P.RAG]):
        fails.append("rag rows with no retrieved chunks")

    # prefill must order closed-book < oracle < rag, or context is not reaching the prompt
    med = {c: sorted(r["prefill_tokens"] for r in by_cond[c] if r["prefill_tokens"])
           for c in P.CONDITIONS}
    mid = {c: v[len(v) // 2] if v else None for c, v in med.items()}
    if any(v is None for v in mid.values()):
        # Report the real problem rather than raising on a None comparison. A check that
        # crashes on bad data is weaker than one that names it.
        fails.append(f"no prefill_tokens recorded for some conditions: {mid} "
                     f"— check the 'error' field on the rows")
    elif not (mid[P.CLOSED_BOOK] < mid[P.ORACLE] < mid[P.RAG]):
        fails.append(f"prefill not ordered closed<oracle<rag: {mid}")

    # oracle context rules
    for r in by_cond[P.ORACLE]:
        if r["type"] == 4 and r["oracle_context_source"] != P.SRC_GOLD_PAIR:
            fails.append(f"type 4 oracle {r['qid']} not a gold pair")
        if r["type"] == 5 and r["oracle_context_source"] != P.SRC_RETRIEVED:
            fails.append(f"type 5 oracle {r['qid']} not a retrieved chunk")
        if r["type"] not in (4, 5) and r["oracle_context_source"] != P.SRC_GOLD:
            fails.append(f"type {r['type']} oracle {r['qid']} not a gold passage")

    # TPOT denominator must be the runtime's generated-token count
    for r in rows:
        if r["generated_tokens"] and r["tpot_ms"]:
            want = round(r["wall_ms"] / r["generated_tokens"], 3)
            if abs(want - r["tpot_ms"]) > 1e-6:
                fails.append(f"{r['qid']}: tpot denominator is not generated_tokens")
                break

    keys = [key_of(r) for r in rows]
    if len(keys) != len(set(keys)):
        fails.append("duplicate (model, condition, qid, k, device) keys")

    thinking = sum(1 for r in rows if (r["thinking_tokens"] or 0) > 0)
    return fails, {"n": len(rows), "prefill_median": mid, "rows_with_thinking": thinking}


def normalise(text):
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


REFUSAL = re.compile(r"not (stated|specified|mentioned|given|provided|available|found)"
                     r"|does not (contain|specify|state|mention)"
                     r"|no (information|mention|answer)|cannot (be )?(found|determined)"
                     r"|unable to (find|determine)|isn't (stated|specified)",
                     re.IGNORECASE)


def agreement(rows):
    """Four measures. Exact match alone would mislead — README section 3's observed
    divergence was verbosity (GPU 171 tokens vs NPU 6), not a different answer."""
    by_case = defaultdict(dict)
    for r in rows:
        by_case[(r["model"], r["condition"], r["qid"])][r["device"]] = r

    tally = Counter()
    detail = []
    for case, per_dev in sorted(by_case.items()):
        if len(per_dev) < 2:
            continue
        outs = {d: (r["cleaned_output"] or "") for d, r in per_dev.items()}
        vals = list(outs.values())
        qtype = per_dev[next(iter(per_dev))]["type"]

        # Empty outputs must not count as trivial agreement. If every device produced
        # nothing the devices do "agree", but on nothing — that is the 0.6B failing to
        # answer at all, and folding it into the headline would inflate every measure.
        # It is counted separately and excluded. If only SOME are empty they genuinely
        # disagree, and containment must not pass on "" being a substring of everything.
        n_empty = sum(1 for v in vals if not normalise(v))
        if n_empty == len(vals):
            tally["all_empty"] += 1
            detail.append({"model": case[0], "condition": case[1], "qid": case[2],
                           "type": qtype, "all_empty": True,
                           "lengths": {d: len(v) for d, v in outs.items()}})
            continue

        a = len(set(vals)) == 1
        b = len({normalise(v) for v in vals}) == 1
        srt = sorted(vals, key=len)
        shortest = normalise(srt[0])
        c = False if n_empty else (
            b or (bool(shortest) and all(shortest in normalise(v) for v in vals)))
        if n_empty:
            d = False                      # a device that answered nothing did not agree
        elif qtype == 5:
            d = len({bool(REFUSAL.search(v)) for v in vals}) == 1
        elif qtype in (1, 3):
            d = len({frozenset(numbers(v)) for v in vals}) == 1
        else:
            d = c
        for name, ok in (("exact", a), ("normalised", b), ("containment", c),
                         ("substantive", d)):
            tally[name] += bool(ok)
        tally["cases"] += 1
        detail.append({"model": case[0], "condition": case[1], "qid": case[2],
                       "type": qtype, "exact": a, "normalised": b,
                       "containment": c, "substantive": d,
                       "lengths": {d_: len(v) for d_, v in outs.items()}})
    return tally, detail


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "agreement", "full", "recompute",
                                       "diagnostic", "acheck"], required=True)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--device", choices=["GPU", "NPU", "CPU"],
                    help="full mode: run one device only (default: GPU then NPU)")
    args = ap.parse_args()

    if args.mode == "recompute":
        if not args.resume:
            raise SystemExit("--mode recompute needs --resume <path to jsonl>")
        recompute(args.resume)
        return

    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.resume or RESULTS / f"grid_{args.mode}_{ts}.jsonl"
    meta_path = out_path.with_suffix(".meta.json")
    run_id = out_path.stem

    meta = {"run_id": run_id, "mode": args.mode, "started": datetime.now().isoformat(),
            "k": K, "max_new_tokens": MAX_NEW_TOKENS,
            "npu_max_prompt_len": NPU_MAX_PROMPT_LEN,
            "power_state": power_state(),
            "environment": {"python": platform.python_version()}}
    gate_freeze(meta)

    questions = load_questions()
    gold = load_gold()
    chunk_text = load_chunk_text()

    from retrieve import search

    def retrieve_fn(query, k):
        return [(cid, chunk_text[cid]) for cid, *_ in search(query, k=k)]

    done = set()
    if args.resume and out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            done.add(key_of(json.loads(line)))
        print(f"resuming: {len(done)} rows already present in {out_path}")

    if args.mode == "smoke":
        probe = probe_questions(questions)
        work = [(q, c) for q in probe for c in P.CONDITIONS]
        jobs = [(SMOKE_MODEL, GRID_DEVICE, work)]
    elif args.mode == "agreement":
        # 2 models x 20 questions x 3 devices = 120 inferences, ONE condition.
        # Closed-book isolates device effects on generation from any retrieval-context
        # variance, and matches the bare-question setting where README section 3 observed
        # the divergence. LIMITATION, recorded in docs: this does not test whether the
        # long-prefill RAG prompts (2902 tokens vs 46) agree across devices. Covering all
        # three conditions would cost ~4.5h because the 8B runs at 2703 ms/token on CPU.
        probe = probe_questions(questions)
        work = [(q, P.CLOSED_BOOK) for q in probe]
        jobs = [(m, d, work) for m in AGREEMENT_MODELS for d in AGREEMENT_DEVICES]
    elif args.mode == "acheck":
        # Gate A. Stage 6's power state was never recorded, so if any of it ran on battery
        # the TTFT/TPOT findings — including the ~650 ms NPU floor — could be throttling
        # artifacts. Re-run a fixed subset on AC and compare medians. The 8B because it is
        # the most power-hungry model and therefore the most sensitive throttling detector,
        # and the model whose numbers carry the headline.
        probe = probe_questions(questions)
        work = [(q, c) for q in probe for c in P.CONDITIONS]
        jobs = [("qwen3-8b", d, work) for d in GRID_DEVICES]
    elif args.mode == "diagnostic":
        # 0.6B without /no_think, Oracle-RAG, iGPU. STRICTLY OUTSIDE the main grid: it
        # uses a different prompt and must never enter the size comparison. It exists to
        # disambiguate "too small to reason" (which supports H2) from "too small to follow
        # the /no_think instruction" (which does not).
        probe = probe_questions(questions)
        work = [(q, P.ORACLE) for q in probe]
        jobs = [("qwen3-0.6b", GRID_DEVICE, work)]
    else:
        work = [(q, c) for q in questions for c in P.CONDITIONS]
        devices = [args.device] if args.device else GRID_DEVICES
        # device outer, so iGPU completes before NPU begins
        jobs = [(m, d, work) for d in devices for m in MODELS]

    total = sum(len(w) for _, _, w in jobs)
    print(f"\n{args.mode}: {total} inferences -> {out_path}")

    if args.mode == "diagnostic":
        # Both arms, so the comparison is MATCHED: same model, same 20 questions, same
        # condition, same device, differing only in the /no_think suffix. Running only the
        # no-/no_think arm would confound the prompt with the Oracle-RAG context, since
        # the Gate 2 baseline was closed-book.
        written = execute(jobs, out_path, run_id, done, retrieve_fn, chunk_text, gold,
                          no_think=True)
        written += execute(jobs, out_path, run_id, done, retrieve_fn, chunk_text, gold,
                           no_think=False)
    else:
        written = execute(jobs, out_path, run_id, done, retrieve_fn, chunk_text, gold,
                          no_think=True)
    rows = [json.loads(l) for l in out_path.open(encoding="utf-8")]

    meta["finished"] = datetime.now().isoformat()
    meta["rows"] = len(rows)
    meta["written_this_run"] = written

    if args.mode == "smoke":
        fails, stats = check_smoke(rows)
        print(f"\n--- smoke checks ---")
        print(f"  rows {stats['n']}  prefill median {stats['prefill_median']}")
        print(f"  rows with thinking tokens: {stats['rows_with_thinking']}")
        for f in fails:
            print(f"  FAIL {f}")
        meta["smoke"] = {"failures": fails, "stats": stats}
        ok_rows = [r for r in rows if not r["error"]]
        if ok_rows:
            per = sum(r["wall_ms"] for r in ok_rows) / len(ok_rows) / 1000
            print(f"\n  mean {per:.1f}s/query on {SMOKE_MODEL}")
            print(f"  extrapolated full grid (2676, scaling by model size): "
                  f"~{per * 2676 * 1.8 / 3600:.1f}h")
        if fails:
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            raise SystemExit("SMOKE TEST FAILED — do not start the grid")
        print("\n  all smoke checks passed")

    elif args.mode == "agreement":
        tally, detail = agreement(rows)
        n = tally["cases"]
        print(f"\n--- device agreement over {n} cases ---")
        for name in ("exact", "normalised", "containment", "substantive"):
            print(f"  {name:<14} {tally[name]:>4}/{n}  {100*tally[name]/n:5.1f}%")
        dump = RESULTS / f"device_agreement_{ts}.jsonl"
        with dump.open("w", encoding="utf-8", newline="\n") as fh:
            for d in detail:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  per-case detail -> {dump}")
        print(f"  all {len(rows)} outputs in {out_path} for inspection")
        meta["agreement"] = {k: v for k, v in tally.items()}

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} rows -> {out_path}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
