"""
Stage 7 — Fit the energy model, choose its functional form, apply it to the grid.

Per (model, device):
    E_decode  = b * generated_tokens                    (slope from the decode sweep)
    E_prefill = f(prefill_tokens)                       (form CHOSEN, not assumed)

The prefill form is selected by held-out error among three candidates, because Stage 6's
NPU timing is flat to ~1,000 tokens and then steps — 651 -> 659 -> 2,062 ms — which is what
static block allocation produces, not a line. A linear fit through that would misestimate
at both ends, and it would land on exactly the prefill-cost comparison this stage exists to
make. A wrong functional form gives confident numbers with STRUCTURED error, which is worse
than noisy ones.

    linear   a*n + c
    block    c + a*ceil(n/B)        static allocation, B fitted
    power    c + a*n^p              saturating or super-linear

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\analysis\\energy_model.py
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("results")
RAIL = "pkg_marginal_j"          # marginal package energy: total minus idle
RESID = "residual_marginal_j"


def latest(pattern):
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"no {pattern} in {RESULTS} — run measure_energy.py first")
    return files[-1]


# ----------------------------------------------------------------- candidate forms

def fit_linear(n, e):
    A = np.vstack([n, np.ones_like(n)]).T
    (a, c), *_ = np.linalg.lstsq(A, e, rcond=None)
    return {"form": "linear", "a": float(a), "c": float(c)}, lambda x, a=a, c=c: a * x + c


def fit_block(n, e):
    """c + a*ceil(n/B). B searched over plausible allocation granularities."""
    best = None
    for B in (64, 128, 256, 320, 512, 640, 768, 1024, 1280, 1536, 2048):
        blocks = np.ceil(n / B)
        A = np.vstack([blocks, np.ones_like(n)]).T
        (a, c), *_ = np.linalg.lstsq(A, e, rcond=None)
        resid = float(np.sum((A @ [a, c] - e) ** 2))
        if best is None or resid < best[0]:
            best = (resid, B, float(a), float(c))
    _, B, a, c = best
    return ({"form": "block", "a": a, "c": c, "B": B},
            lambda x, a=a, c=c, B=B: c + a * np.ceil(np.asarray(x) / B))


def fit_power(n, e):
    """c + a*n^p, p searched on a grid to keep the fit linear in (a, c)."""
    best = None
    for p in np.arange(0.2, 2.01, 0.05):
        A = np.vstack([n ** p, np.ones_like(n)]).T
        (a, c), *_ = np.linalg.lstsq(A, e, rcond=None)
        resid = float(np.sum((A @ [a, c] - e) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(p), float(a), float(c))
    _, p, a, c = best
    return ({"form": "power", "a": a, "c": c, "p": p},
            lambda x, a=a, c=c, p=p: c + a * np.asarray(x, dtype=float) ** p)


FORMS = {"linear": fit_linear, "block": fit_block, "power": fit_power}


def choose_form(n, e, rep, apply_range):
    """Held-out selection among forms that are PHYSICALLY VALID over the applied range.

    Held-out MAE alone is not sufficient. Fitted over 38-4192 tokens, a linear form can
    win on average error while predicting NEGATIVE energy at the short end — measured:
    -0.437 J at 49 tokens for the 4B on GPU. Closed-book (49) and Oracle-RAG (143) are
    exactly where two thirds of the grid sits, so that form would be confidently wrong
    precisely where it is used most, which is the structured-error failure this selection
    exists to avoid. Energy cannot be negative, so such a form is disqualified outright
    rather than merely penalised.
    """
    train = rep < max(rep)
    if train.sum() < 4 or (~train).sum() < 2:
        train = np.ones_like(rep, dtype=bool)
    probe = np.array(sorted(apply_range), dtype=float)
    scored = {}
    for name, fitter in FORMS.items():
        params, fn = fitter(n[train], e[train])
        pred = np.asarray(fn(n[~train]), dtype=float)
        err = np.abs(pred - e[~train])
        _, full_fn = fitter(n, e)
        worst = float(np.min(np.asarray(full_fn(probe), dtype=float)))
        scored[name] = {"params": params,
                        "heldout_mae_j": float(np.mean(err)),
                        "heldout_mape": float(np.mean(err / np.maximum(e[~train], 1e-9)) * 100),
                        "min_predicted_j": worst,
                        "valid": worst >= 0.0}
    valid = [k for k in scored if scored[k]["valid"]]
    best = min(valid or list(scored), key=lambda k: scored[k]["heldout_mae_j"])
    params, fn = FORMS[best](n, e)          # refit on everything once chosen
    return best, params, fn, scored


def main():
    sweep = [json.loads(l) for l in latest("energy_sweep_*.jsonl").open(encoding="utf-8")]
    comp = [json.loads(l) for l in latest("energy_components_*.jsonl").open(encoding="utf-8")]
    grid_path = latest("grid_full_*.jsonl")
    grid = [json.loads(l) for l in grid_path.open(encoding="utf-8")]
    # The prefill lengths the model will actually be asked to predict.
    apply_range = sorted({r["prefill_tokens"] for r in grid if not r["error"]})

    embed_j = next(c[RAIL] for c in comp if c["component"] == "query_embedding")
    search_j = next(c[RAIL] for c in comp if c["component"] == "search_full_rrf")
    print(f"retrieval components: embedding {embed_j*1000:.1f} mJ, "
          f"full search+rrf {search_j*1000:.1f} mJ "
          f"(index-only {max(search_j-embed_j,0)*1000:.1f} mJ)\n")

    models = {}
    print("Functional form selection — held-out MAE per candidate\n")
    print(f"{'model':<12}{'dev':<5}{'chosen':<8}{'linear':>10}{'block':>10}{'power':>10}"
          f"{'B':>7}{'decode J/tok':>14}   disqualified (negative)")

    for device in sorted({r["device"] for r in sweep}):
        for model in sorted({r["model"] for r in sweep}):
            pre = [r for r in sweep if r["kind"] == "prefill"
                   and r["model"] == model and r["device"] == device]
            dec = [r for r in sweep if r["kind"] == "decode"
                   and r["model"] == model and r["device"] == device]
            if not pre or not dec:
                continue

            n = np.array([r["prefill_tokens"] for r in pre], dtype=float)
            e = np.array([r[RAIL] for r in pre], dtype=float)
            rep = np.array([r["rep"] for r in pre])
            best, params, fn, scored = choose_form(n, e, rep, apply_range)

            # decode slope: energy above the 1-token case, per extra generated token
            m = np.array([r["generated_tokens"] for r in dec], dtype=float)
            de = np.array([r[RAIL] for r in dec], dtype=float)
            A = np.vstack([m, np.ones_like(m)]).T
            (b, c0), *_ = np.linalg.lstsq(A, de, rcond=None)

            models[f"{model}|{device}"] = {
                "prefill": params, "prefill_scored": scored,
                "decode_j_per_token": float(b), "decode_intercept_j": float(c0),
            }
            bad = ",".join(k for k, v in scored.items() if not v["valid"]) or "-"
            print(f"{model:<12}{device:<5}{best:<8}"
                  f"{scored['linear']['heldout_mae_j']:>10.3f}"
                  f"{scored['block']['heldout_mae_j']:>10.3f}"
                  f"{scored['power']['heldout_mae_j']:>10.3f}"
                  f"{params.get('B',''):>7}{b:>14.4f}   {bad}")

    # ------------------------------------------------------------------ apply
    def predict(model, device, prefill_tokens, gen_tokens):
        key = f"{model}|{device}"
        if key not in models:
            return None
        p = models[key]["prefill"]
        if p["form"] == "linear":
            ep = p["a"] * prefill_tokens + p["c"]
        elif p["form"] == "block":
            ep = p["c"] + p["a"] * math.ceil(prefill_tokens / p["B"])
        else:
            ep = p["c"] + p["a"] * prefill_tokens ** p["p"]
        ed = models[key]["decode_j_per_token"] * gen_tokens
        return max(ep, 0.0), max(ed, 0.0)

    out_rows, cells = [], defaultdict(list)
    for r in grid:
        if r["error"]:
            continue
        got = predict(r["model"], r["device"], r["prefill_tokens"], r["generated_tokens"])
        if not got:
            continue
        ep, ed = got
        # Retrieval energy follows oracle_context_source, not the condition name: type-5
        # Oracle-RAG retrieves a chunk and therefore pays it too.
        retrieves = r["oracle_context_source"] in ("retrieved_top_k", "retrieved_chunk")
        er = search_j if retrieves else 0.0
        row = {"run_id": r["run_id"], "model": r["model"], "device": r["device"],
               "condition": r["condition"], "qid": r["qid"], "type": r["type"],
               "prefill_tokens": r["prefill_tokens"],
               "generated_tokens": r["generated_tokens"],
               "e_prefill_j": round(ep, 4), "e_decode_j": round(ed, 4),
               "e_retrieval_j": round(er, 6),
               "e_total_j": round(ep + ed + er, 4),
               "retrieval_counted": retrieves}
        out_rows.append(row)
        cells[(r["device"], r["model"], r["condition"])].append(row)

    out = RESULTS / "energy_per_query.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n\nJoules per query per cell — marginal package energy, prefill/decode split\n")
    print(f"{'device':<5}{'model':<12}{'condition':<12}{'n':>5}"
          f"{'prefill J':>11}{'decode J':>10}{'retr J':>9}{'total J':>10}")
    summary = {}
    for key in sorted(cells):
        rs = cells[key]
        s = {"n": len(rs),
             "prefill_j": statistics.median(r["e_prefill_j"] for r in rs),
             "decode_j": statistics.median(r["e_decode_j"] for r in rs),
             "retrieval_j": statistics.median(r["e_retrieval_j"] for r in rs),
             "total_j": statistics.median(r["e_total_j"] for r in rs)}
        summary["|".join(key)] = s
        print(f"{key[0]:<5}{key[1]:<12}{key[2]:<12}{s['n']:>5}"
              f"{s['prefill_j']:>11.2f}{s['decode_j']:>10.2f}"
              f"{s['retrieval_j']*1000:>8.1f}m{s['total_j']:>10.2f}")

    (RESULTS / "energy_model.json").write_text(
        json.dumps({"models": models, "components": {"embedding_j": embed_j,
                                                     "search_j": search_j},
                    "grid": grid_path.name}, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "joules_per_cell.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(out_rows)} rows -> {out}")
    print("model -> results/energy_model.json   cells -> results/joules_per_cell.json")


if __name__ == "__main__":
    main()
