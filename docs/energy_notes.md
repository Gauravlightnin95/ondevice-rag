# Energy notes — Stage 7

Scripts: `src/inference/power.py`, `measure_energy.py`, `src/analysis/energy_model.py`.
Outputs: `results/energy_sweep_*.jsonl`, `energy_components_*.jsonl`, `energy_model.json`,
`energy_per_query.jsonl`, `joules_per_cell.json`.

**Every figure here is an estimate, reported as a relative comparison.** Lunar Lake has no NPU
power sensor, so NPU energy is a residual, and the residual contains more than the NPU.

## Measurement — Windows PDH RAPL, not HWiNFO

README §7 originally named HWiNFO. It is not installed, and Windows exposes the same RAPL
registers natively as PDH performance counters. README §7 has been **updated to name PDH** rather
than carrying a deviation note: the spec named HWiNFO before the counters were checked, and
leaving the spec wrong is worse than changing it.

| Rail | Counter |
|---|---|
| Package | `\Energy Meter(RAPL_Package0_PKG)\Energy` |
| IA (cores) | `_PP0` |
| GT (graphics) | `_PP1` |
| DRAM | `_DRAM` |

The deciding property is that `Energy` is a **monotonic accumulator**, so per-query energy is an
exact difference between two reads rather than a sampled integral. HWiNFO's free-version CSV
logging is ~1 s granularity and many queries here are shorter than that — the 0.6B closed-book
TTFT is 47 ms. No HWiNFO cross-check was run: both tools read the same registers, so agreement
would prove the reader works, not that the residual estimates NPU power correctly.

`Power Meter (0)` reads a constant 17912 on this machine — a dead sensor, unused.

### Units

**Nanojoules**, established three ways, because the raw unit is undocumented and getting it wrong
would scale every absolute figure:

1. The sibling `Time` counter is unambiguously milliseconds — it advanced 8,016 over a measured
   8.00 s window.
2. At 1e9 the package idles at **2.43 W** and a saturating 8-process load reads **8.3 W**, both
   plausible for a 258V.
3. The alternative implied by integrating the PDH `Power` counter (2.86e8) would put idle at 8.5 W
   and a `math.sqrt` loop at 29 W, near TDP. Not credible — `Power` is the unreliable reference.

Linearity confirmed: implied watts constant within 6.5% across 2 s / 4 s / 8 s windows.

### What the residual contains

```
residual = PKG − PP0 − PP1 = uncore + system agent + NPU
```

**PDH does not separate system agent**, so the residual is an upper bound on NPU power, not a
measurement of it. This is a named limitation.

Evidence the residual does track NPU work: under load the NPU's residual rises **5.09×**
(0.75 → 3.82 W) while its package rises only 2.38×. On iGPU the two move together (2.69× vs
2.66×), as expected when the work is on a rail that *is* measured.

### Conditions

All measurement on **AC** (`power.py` aborts on battery), reported as **SoC energy, not system
energy** — RAPL excludes display, SSD and WiFi. `power_state` is now logged per grid row.

README §7's "baseline idle around 13 W" does not match this machine: **2.4–2.9 W package**. The
13 W figure predates the measurement and likely included the display.

## Gate A — Stage 6's timings stand

Stage 6 never recorded power state, so its TTFT/TPOT findings were checked against a 120-inference
AC re-run of the 8B. The check fired, but the cause was **thermal, not power state**:

| iGPU 8B TTFT | grid | hot re-run | cooled re-run |
|---|---|---|---|
| closed-book | 83.0 ms | 126.7 (+53%) | **81.5 (−1.8%)** |
| oracle | 107.2 ms | 146.9 (+37%) | **100.8 (−6.0%)** |
| rag | 1540.1 ms | 2044.2 (+33%) | **1701.1 (+10.5%)** |

Three lines of evidence: the grid has no mid-run discontinuity (max 9.8% half-to-half drift, and
*decreasing* — cache warming, not a power step), NPU never moved more than 4%, and the cooled
re-run matches. The inflation came from a CPU load ladder run minutes before.

**iGPU TTFT is thermally sensitive by ~50%**, so the sweep cools to a settled package power before
each block and randomises point order within it.

## Two measurement artifacts caught before they reached results

**KV cache contamination.** Sweep prompts were initially truncations of one corpus blob, making
each a prefix of the longer ones; the stateful pipeline served them from cache. The same
2,904-token prompt cost **222 → 36 → 12.9 ms** on successive runs. That reads exactly like prefill
energy saturating with length and would have selected the wrong functional form on the one
comparison this stage exists to make. Fixed by giving every measurement a distinct, non-overlapping
prompt (rotated corpus offsets, pool of 12 per length); repeats then hold at ~232 ms / 2.0 J.

**Quantisation.** The accumulator ticks at exactly **0.1 J** (GCD of 60 short idle deltas). A
49-token prefill costs ~0.02 J — below one tick, so single-call measurements were mostly noise.
Each point now batches distinct prompts until the window clears 5 J. All 384 sweep windows landed
under 10% tick error; median 1.9%.

## Functional form — chosen, not assumed

Prefill energy was fitted three ways per (model, device) and selected on held-out error, with a
hard constraint: **a form predicting negative energy anywhere in the applied range is
disqualified.** That constraint is not cosmetic — fitted across 38–4,192 tokens, the linear form
won on average error for the 4B and 8B on iGPU while predicting **−0.437 J at 49 tokens**, and
49/143 tokens is where two thirds of the grid sits. It would have been confidently wrong exactly
where it was used most.

| Model | Device | Chosen | linear MAE | block MAE | power MAE | B | Disqualified |
|---|---|---|---|---|---|---|---|
| 0.6B | iGPU | power | 0.079 | 0.085 | **0.057** | | linear |
| 1.7B | iGPU | power | 0.175 | 0.161 | **0.151** | | linear |
| 4B | iGPU | block | 0.278 | **0.399** | 0.496 | 512 | linear |
| 8B | iGPU | power | 0.578 | 0.666 | **0.601** | | linear |
| 0.6B | NPU | **block** | 0.558 | **0.221** | 0.284 | **1024** | – |
| 1.7B | NPU | **block** | 0.783 | **0.208** | 0.373 | **1024** | – |
| 4B | NPU | **block** | 1.859 | **0.588** | 1.152 | **1024** | – |
| 8B | NPU | **block** | 3.379 | **0.940** | 1.952 | **1024** | – |

**All four NPU models independently choose the block form with B = 1024**, beating linear by 2.5–3.6×
on held-out error. 1024 is exactly OpenVINO GenAI's default NPU prompt granularity — the same
static allocation that made k=5 RAG fail at the default `MAX_PROMPT_LEN` and that produces the
~650 ms TTFT floor. **Three independent observations converge on one mechanism: NPU prefill cost
is quantised in 1,024-token blocks, in time and in energy.**

iGPU shows no such structure: no 1024 block anywhere, and linear disqualified on all four models
for going negative at short prompts.

Decode is linear in generated tokens everywhere, and **per-token decode energy is monotonic in
model size on both devices** — iGPU 0.0442 / 0.0599 / 0.1089 / 0.1793 J/token, NPU 0.0417 / 0.0787 /
0.1302 / 0.2360.

## Joules per query per cell

Marginal package energy (idle subtracted), median per cell. Retrieval is 77.3 mJ, charged by
`oracle_context_source` rather than condition name, so type-5 Oracle-RAG pays it too.

| Device | Model | Condition | Prefill J | Decode J | Retr mJ | **Total J** |
|---|---|---|---|---|---|---|
| iGPU | 0.6B | closed-book | 0.10 | 22.63 | 0 | **22.73** |
| iGPU | 0.6B | Oracle-RAG | 0.11 | 22.63 | 0 | **22.74** |
| iGPU | 0.6B | RAG | 1.56 | 22.63 | 77.3 | **24.25** |
| iGPU | 1.7B | closed-book | 0.07 | 1.20 | 0 | **1.27** |
| iGPU | 1.7B | Oracle-RAG | 0.10 | 1.20 | 0 | **1.33** |
| iGPU | 1.7B | RAG | 2.10 | 1.02 | 77.3 | **3.24** |
| iGPU | 4B | closed-book | 0.01 | 6.10 | 0 | **6.11** |
| iGPU | 4B | Oracle-RAG | 0.01 | 4.46 | 0 | **4.59** |
| iGPU | 4B | RAG | 5.76 | 6.53 | 77.3 | **12.80** |
| iGPU | 8B | closed-book | 0.41 | 6.64 | 0 | **7.04** |
| iGPU | 8B | Oracle-RAG | 0.48 | 4.48 | 0 | **5.06** |
| iGPU | 8B | RAG | 7.70 | 5.02 | 77.3 | **13.10** |
| NPU | 0.6B | closed-book | 0.78 | 21.33 | 0 | **22.10** |
| NPU | 0.6B | Oracle-RAG | 0.78 | 21.33 | 0 | **22.10** |
| NPU | 0.6B | RAG | 3.67 | 21.33 | 77.3 | **25.08** |
| NPU | 1.7B | closed-book | 1.56 | 0.63 | 0 | **2.19** |
| NPU | 1.7B | Oracle-RAG | 1.56 | 1.18 | 0 | **2.74** |
| NPU | 1.7B | RAG | 6.33 | 40.29 | 77.3 | **46.69** |
| NPU | 4B | closed-book | 3.35 | 7.03 | 0 | **10.39** |
| NPU | 4B | Oracle-RAG | 3.35 | 4.95 | 0 | **8.30** |
| NPU | 4B | RAG | 14.24 | 7.03 | 77.3 | **22.65** |
| NPU | 8B | closed-book | 5.94 | 8.26 | 0 | **14.20** |
| NPU | 8B | Oracle-RAG | 5.94 | 5.90 | 0 | **11.84** |
| NPU | 8B | RAG | 23.01 | 5.90 | 77.3 | **30.40** |

### Reading these

**The 0.6B is the most expensive model per query on iGPU** — 22.7 J against the 8B's 7.0 J, a 3.2×
inversion of the size ordering. Not a measurement error: the 0.6B generates the full 512-token cap
in every condition on both devices, while the 1.7B and 8B stop at 8–37 tokens. Per *token* the
ordering is strictly monotonic. **Per query, the smallest model costs the most, because it never
stops talking.** That is a direct H1 result and it survives into joules per correct answer, where
the 0.6B's accuracy is also lowest.

**RAG's prefill cost is the dominant device difference.** On iGPU RAG prefill is 5.8–7.7 J for the
larger models; on NPU it is 14.2–23.0 J, roughly 3×. Prefill is 59–76% of the whole NPU RAG query.

**Retrieval itself is negligible per query** — 77.3 mJ, under 1% of every cell. But 94% of that is
the **embedding model** (72.4 mJ); FAISS + BM25 + RRF together are ~5 mJ. H1's "RAG adds an
embedding model and an index" is a RAM cost far more than an energy cost: Stage 6 measured the
resident footprint at 1.9–6.3 GB.

**NPU 1.7B RAG is an outlier at 46.7 J** — the highest cell in the grid, above even the NPU 8B.
Its decode is 40.3 J because on NPU that model generates the full 512-token cap under RAG (median
512) while on iGPU it stops at 17. This is Stage 6's device divergence showing up in energy.

## Validation

| Check | Result |
|---|---|
| Idle vs load discrimination | iGPU 2.66×, NPU 2.38× — **PASS** |
| Accumulator linearity | 6.5% spread across 2/4/8 s windows |
| Tick error | 0 of 384 windows above 10%; median 1.9% |
| Additivity — prefill + decode vs measured full query | median **3.2%**, max 10.8% |
| Per-token energy monotonic in model size | **yes**, both devices |
| Held-out MAPE of the chosen form | NPU 6.8–10.3%; iGPU 14.5–56.9% |

The iGPU held-out error is worse in *relative* terms because iGPU prefill energies are small in
absolute terms — the 4B's 56.9% MAPE corresponds to 0.399 J MAE against values as low as 0.01 J.
Where it matters (RAG prefill, 5.8–7.7 J) the relative error is far smaller. Reported rather than
smoothed over.

## Limitations

- The residual is **uncore + system agent + NPU**, not NPU alone. PDH cannot separate system agent.
- SoC energy only. Display, SSD, WiFi excluded.
- Energy is **modelled** from measured coefficients, not measured per grid row. That is deliberate:
  re-running would produce different generations (Stage 6 found only 17.6% iGPU/NPU agreement), so
  measured re-run energy would not correspond to the rows Stage 8 grades.
- Absolute joules carry the nanojoule unit assumption. Relative comparisons do not.
- Joules per correct answer needs Stage 8.
