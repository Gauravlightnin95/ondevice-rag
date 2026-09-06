# Inference notes — Stage 6

Scripts: `src/inference/pipeline.py`, `prompts.py`, `run_grid.py`.
Outputs: `results/grid_{mode}_{ts}.jsonl` + `.meta.json` (gitignored).

The grid is 4 models × 3 conditions × 223 questions = **2,676 inferences**. The risk is not any
one wrong number, it is spending hours producing an unusable log — so three gates run first, and
the runner is resumable.

## Fixed parameters

| | |
|---|---|
| `k` (RAG condition) | **5** — the literature default, so the headline stays comparable |
| `max_new_tokens` | **512**, uniform across all four models |
| Decoding | greedy, `do_sample=False` (README §8) |
| Grid devices | **iGPU and NPU**, both across the full eval set — device is an axis, see Gate 2 |
| `MAX_PROMPT_LEN` (NPU only) | **4608** — mandatory for RAG; see the static-allocation tax below |
| Thinking mode | `/no_think` appended to every user message, uniform |

`max_new_tokens` is uniform deliberately. Varying it by model size would confound the size axis,
the same reasoning README §2 gives for keeping `/no_think` uniform.

## Conditions

The three differ **only** in the context block; system prompt, user-message shape and `/no_think`
are identical across conditions and models.

| Condition | Context | Median prefill (measured) |
|---|---|---|
| `closed_book` | none at all | 46 tokens |
| `oracle` | gold passage(s) | 152 tokens |
| `rag` | top-5 retrieved chunks | 2,902 tokens |

Closed-book receives **no context block**, not an empty one, and `build_user_content` asserts it is
never passed passages. A closed-book cell that silently received context would invalidate the whole
comparison, so the smoke test checks the ordering above as well.

## Oracle-RAG rules

| Type | Context | `oracle_context_source` |
|---|---|---|
| 1, 2, 3, 6 | the gold `passage` | `gold_passage` |
| 4 | **both** `passage` and `passage2` | `gold_passage_pair` |
| 5 | the **top-1 retrieved chunk** | `retrieved_chunk` |

**Type-5 selection rule.** Type 5 has `passage: null` by definition, so there is no gold text and
README §5 requires "topically-related passages that do not contain the answer". The rule is: query
the frozen retriever and supply the single top-ranked chunk. It introduces nothing above the LLM;
it cannot contain the answer *by construction*, since `verify_absent.py` established the answer is
absent from the whole corpus so no chunk can hold it; and top-1 rather than top-5 preserves
Oracle-RAG's small-context character — handing type 5 the full k=5 set would make its oracle cell
byte-identical to its RAG cell and the comparison vacuous.

### Type-5 Oracle-RAG is not the same condition as the other types

Types 1–4 and 6 receive a **human-verified gold passage**, median 242 characters. Type 5 receives a
**retrieved chunk**, ~2,447 characters, plausible but wrong. Those are categorically different
objects.

**Consequently, type-5 accuracy under Oracle-RAG is not directly comparable to the other types
along the type axis within that condition.** A reader scanning the Oracle-RAG column will see a
uniform-looking row that is not measuring the same thing. It remains valid for the purpose it
exists to serve — measuring refusal behaviour when plausible-looking context *is* present, which
README §5 argues is the harder and more informative test than supplying nothing.

`oracle_context_source` is logged on every row so analysis partitions on the actual context kind
rather than inferring it from the type.

### Context-size asymmetry

Oracle-RAG supplies a median 242 characters; RAG at k=5 supplies ~12,200 — a ~50× difference,
visible in the prefill table above (152 vs 2,902 tokens). Oracle-RAG therefore isolates reasoning
*and* removes most of the prefill, so its TTFT, prefill tokens and joules-per-query are not a
"same workload, better retrieval" contrast against RAG. That is inherent to the design (README §1),
not a defect, but the paper has to state it.

## Two corrections to `test_npu.py`'s helpers

`test_npu.py` is **not modified** — it is the record of the hardware characterisation behind
README §3. But two of its helpers are reimplemented in `pipeline.py`, because using them as-is
would corrupt the 0.6B's numbers:

**`clean()`** has two regexes; README §2 specifies **three**. The missing one is
`re.sub(r"<think>.*$", "", flags=re.DOTALL)`, for an **unclosed** tag. README §3 records that the
0.6B "opens a `<think>` tag and never closes it" — with the two-regex version that model's entire
output survives as if it were an answer.

**`thinking_tokens()`** matches only `<think>(.*?)</think>`, so an unclosed block counts as **zero**
thinking tokens for output that is *entirely* thinking. README §2 is explicit that thinking tokens
are a real energy cost and hiding them is not acceptable. The version here counts the unclosed tail.

The smoke test asserts at least one row carries non-zero thinking tokens, so this path cannot go
dead silently. Measured on the 1.7B smoke run: **58 of 60 rows** had thinking tokens despite
`/no_think`.

## A bug the smoke test caught in 60 seconds

`pipe.generate(prompt, config)` with a **string** prompt returns a bare `str` — no `perf_metrics`,
so no TTFT, no prefill count, no true generated-token count. All 60 smoke rows failed with
`AttributeError: 'str' object has no attribute 'perf_metrics'`.

The fix is to pass a **list**: `pipe.generate([prompt], config)` returns `DecodedResults` carrying
`perf_metrics` and `finish_reasons`. Had the smoke test not run first, this would have surfaced
after four hours of grid time with 2,676 empty rows.

A second, smaller finding: the first version of the smoke check *crashed* on that broken data
(`TypeError` comparing `None`) instead of reporting it. A check that dies on bad input is weaker
than one that names it, so it now reports "no prefill_tokens recorded — check the error field".

## Metric definitions

**TPOT** is `tpot_ms = wall_ms / generated_tokens` — elapsed over **actual tokens generated**, per
README §7, so a model that refuses early does not look artificially fast. `wall_ms` is rounded
before the division, so the invariant `tpot_ms == wall_ms / generated_tokens` holds exactly against
the stored values and the log is self-auditable.

The runtime's own `get_tpot()` is logged alongside as `tpot_runtime_ms`; it uses a different
definition ((elapsed − TTFT)/(n−1)). Both plus their components are logged, so either can be
recomputed later without re-running anything.

**`generated_tokens`** comes from `perf_metrics.get_num_generated_tokens()`, the runtime's true
count — not from re-encoding the decoded text, which is an approximation because
detokenise→retokenise need not round-trip. It is the TPOT denominator, so it has to be exact. The
re-encoded count is retained as `raw_tokens` for continuity with `test_npu.py`'s reporting.

**Peak RSS** uses `psutil` `peak_wset`, a true Windows process peak. It is **monotonic within a
process** and cannot be reset, so a per-query "peak" is really peak-so-far; `rss_before_mb`,
`rss_after_mb` and `peak_wset_mb` are all logged, and the per-model figure README §7 wants is the
value at the end of that model's block.

## Retrieval and the frozen index

Retrieval goes through `retrieve.search()` only, never a reimplementation. `search()` resolves to
`index/frozen/v1` by default, so the grid cannot read an unfrozen index by omission.

Every row carries `passage1_hit`, `passage2_hit` and `both_source_hit` against the Stage 3 gold
map. At k=5 this makes the H2 distinction recoverable **from the log alone**, with no re-running of
retrieval: *the model held both sides of the contradiction and failed to notice* versus *the model
only ever saw one side*. Stage 4 measured type-4 both-source recall at 16% @3 and 84% @10, so both
populations are well represented at k=5.

## Gates

| Gate | Cost | Purpose |
|---|---|---|
| 0 freeze verification | seconds | abort on `DRIFTED`; report written to `.meta.json` |
| 1 smoke — 20q × 1 model × 3 conditions | 60 inferences | assert the wiring before spending hours |
| 2 device agreement — 20q × 2 models × 3 devices | 120 inferences | justify the single-device grid |
| 3 full grid | 2,676 inferences | the experiment |

The smoke test **asserts** rather than merely running. All seven assertions were verified to fire
by deliberately corrupting a row set: closed-book carrying context, a RAG row with no chunks,
prefill ordering collapsing, a type-4 oracle given one passage, a wrong TPOT denominator, a
duplicate key, and an empty cleaned output — each was caught.

Measured on the 1.7B: **0.8 s/query**, extrapolating to roughly 1–1.5 h for the full grid.

**Crash safety.** Rows are appended and flushed one at a time. `--resume <path>` rebuilds the
done-set keyed on `(model, condition, qid, k, device)` and skips completed work. Verified by
truncating a 60-row run to 25 and resuming: it ran exactly the missing 35 and finished at 60 with
no duplicate keys.

## The `/no_think` output patterns — three, not two

README §2's `clean()` deletes everything after an unclosed `<think>`. Measured against 138 real
outputs there are **three** distinct patterns, and the second is destroyed by that rule:

| Pattern | Model | `finish_reason` | Meaning | Correct action |
|---|---|---|---|---|
| `<think>…</think>` | 1.7B, 90 rows | STOP | normal closed block | strip |
| `<think>\n\n` + text | 8B, 31 rows | STOP, 9–113 tok | **empty** block, closer dropped by the decoder — the text is **the answer** | keep the text |
| `<think>\n` + text | 0.6B, 25 rows | LENGTH, 512 tok | genuine rambling, never closed, no answer produced | strip all |

The signature separates them with **zero overlap**, and `clean()` keys on the `/no_think` token
pattern (blank line = empty block) rather than on model identity — cleaning that varied by model
size would confound the size axis.

**This is a deliberate deviation from README §2's literal code.** Applying that version destroyed
**31 of the 8B's 60 answers** — on the model carrying the headline comparison — while leaving
`raw_output` intact. The corrected rule recovers all 31 with **zero regressions**: no row that was
non-empty became empty.

`cleaned_output` and `thinking_tokens` are pure functions of `raw_output`, so `--mode recompute`
re-derives them in place on an existing log without re-running a single inference.

The 0.6B's 25 rows are not a bug to fix: it hits the 512-token cap producing reasoning and never
answers, which is README §3's documented behaviour, confirmed and quantified.

## Device agreement — Gate 2 result: the gate FAILED

120 inferences, closed-book, 20 questions × {8B, 0.6B} × {NPU, GPU, CPU}. Zero errors.
`prompt_sha256` is identical across devices for every case, so the prompts are byte-identical.

**Decision rule, registered before the numbers existed:** exact low + substantive high would justify
a single-device grid. Substantive low means accuracy must be reported per device and the grid scope
changes.

| | 8B | 0.6B |
|---|---|---|
| exact | 15.0% | 0.0% |
| normalised | 15.0% | 0.0% |
| containment | 20.0% | 0.0% |
| **substantive** | **50.0%** | **0.0%** |
| all-empty cases (excluded) | 0 | 9 |

Classifying the 8B's 20 cases directly:

| | n |
|---|---|
| identical | 3 |
| same answer, verbosity differs | 1 |
| **different numeric answer** | **7** |
| **different wording / claim** | **9** |

**Only 4 of 20 agree in any meaningful sense.** This is not the verbosity effect README §3
observed — these are different factual claims from the same model on the same byte-identical
prompt under greedy decoding:

```
Final_Mandatory_Disclosure::Q4   NPU 650 sq m   GPU 4,500 sq ft   CPU 1,750 sq m
_conflicts::Q10                  NPU 15         GPU 30            CPU 150
PART-A-ACADEMIC-REGULATIONS::Q10 NPU/GPU "first two weeks"        CPU "first three weeks"
```

The 0.6B is worse: 19 of 20 cases involve at least one device producing nothing at all (9 all
three, 10 some). It is effectively non-functional under `/no_think` on this hardware.

### An inflated first reading, and why

The first pass reported containment 80% and substantive 90% for the 8B. Those numbers were wrong
twice over: `clean()` was emptying 31 rows, and the agreement measures then treated an empty string
as trivially agreeing — `""` is a substring of everything, and two empty outputs share an empty
number set. All-empty cases are now counted separately and excluded; partially-empty cases count as
disagreement. Recorded here because the inflated numbers would have justified exactly the wrong
decision.

### Consequence: device is an axis, not a caveat

**Absolute accuracy is a property of (model, device), not of model.** On-device inference work
generally treats accuracy as a property of the model; this says it is a property of the model and
the accelerator, and 20 questions is too thin to carry that claim.

So the grid runs on **iGPU and NPU across the full eval set** — 5,352 inferences, ~11.3 h measured
— with device reported as an experimental axis. iGPU runs first so the primary results exist early;
NPU follows through the resume mechanism.

**CPU is dropped.** At 2,703 ms/token (README §3) it is not a deployment target for an on-device
framing, and Gate 2 already characterises it. Estimated cost was ~10.2 h of the 21.5 h total.

Running both remaining devices over the whole eval set makes two things testable that a
single-device grid could not answer: whether divergence concentrates **by model size** (the 8B sits
past the NPU's 3–4 GB fast-memory boundary) and **by question type** (numerical questions look most
affected — 7 of the 8B's 20 cases differed numerically).

Measured per-device cost, closed-book, 20 questions:

| | 0.6B | 8B |
|---|---|---|
| iGPU | 4.7 s | 2.5 s |
| NPU | 7.6 s | 4.2 s |
| CPU | 6.1 s | 11.0 s |

## The NPU static-allocation tax — a fourth hidden cost of RAG (H1)

**At OpenVINO GenAI's default configuration, the NPU cannot run k=5 RAG on this corpus at all.**
Every one of the 892 NPU RAG rows failed:

```
Stateful LLM pipeline on NPU may only process prompts or hold chat history
up to 1024 tokens. 2921 is passed. Set the "MAX_PROMPT_LEN" config option.
```

RAG prompts here are 2,889–4,192 tokens. Closed-book (median 49) and Oracle-RAG (median 143) run
fine, so the failure is specific to retrieved context. Those 892 failures are preserved as evidence
in `results/npu_default_config_failure.jsonl` rather than discarded — the default-configuration
failure is a result.

The NPU pipeline statically allocates for a **declared maximum** prompt length. Raising it to 4608
fixes RAG, and it is not free. Measured on the 1.7B, closed-book:

| `MAX_PROMPT_LEN` | Compile | Median short-prompt latency |
|---|---|---|
| default (1024) | 2.8 s | 436.8 ms |
| **4608** | **63.3 s** (22×) | **496.1 ms** (+13.6%) |

**This is an H1 finding, not a configuration note.** H1 names RAG's hidden costs as the embedding
model, the RAM-resident index, and inflated prefill. This is a fourth, and it is *invisible to any
prefill measurement*: supporting k=5 RAG on the NPU costs **13.6% more latency on every short
prompt and 22× compile time — a tax paid even by queries that carry no retrieved context at all.**
A closed-book query on an NPU configured for RAG is slower than the same query on an NPU that is
not, and no prefill-token count would ever reveal it.

### Why the whole NPU half was re-run

Because the tax applies to short prompts, the 1,784 NPU rows already completed at the default were
**not comparable** to anything run at 4608. Re-running RAG only would have handed NPU closed-book
and Oracle-RAG a 13.6% configuration advantage over NPU RAG — landing the confound precisely on
RAG's prefill penalty, the number this project exists to measure. All 2,676 NPU rows were re-run
under one configuration (~3.5 h) so the condition axis on NPU differs only by prefill length.

`MAX_PROMPT_LEN` is logged per row as `npu_max_prompt_len` and in the run meta, so the
configuration is recoverable from the results.

### Why 4608 and not 8192

4608 covers the largest observed RAG prompt (4,192) with ~10% headroom, and is frozen alongside
k=5. Sizing for the later H3 k-sweep would have made this grid's NPU latency reflect a buffer that
exists only for a different experiment — and since the tax scales with the allocated buffer, that
would inflate the very number being reported. The sweep gets its own configuration.

**For the H3 sweep:** if the static-allocation tax scales with `MAX_PROMPT_LEN`, that scaling is
worth measuring directly. "Supporting larger k costs latency on *every* query, not just large-k
queries" is a stronger version of this H1 finding than the single data point at 4608, and the sweep
is the natural place to establish it.

### The tax is a TTFT floor, not a percentage

The full grid shows the cost is far larger than the 13.6% the isolated benchmark suggested, and
structurally different. **NPU TTFT is near-flat with respect to actual prompt length** — it tracks
the declared buffer, not the prompt:

| Prefill bucket | n | iGPU TTFT | NPU TTFT | NPU/iGPU |
|---|---|---|---|---|
| 0–100 | 988 | 68.1 ms | 651.8 ms | **9.57×** |
| 100–300 | 616 | 90.6 ms | 582.1 ms | 6.42× |
| 300–1,000 | 180 | 84.0 ms | 659.5 ms | 7.85× |
| 2,500–3,200 | 796 | 816.9 ms | 2,317.2 ms | **2.84×** |
| 3,200–4,300 | 88 | 1,059.8 ms | 3,340.6 ms | 3.15× |

Going from 49 to ~2,900 prefill tokens — 59× more — multiplies iGPU TTFT by **12.0×** but NPU TTFT
by only **3.6×**. A 49-token closed-book prompt on the NPU costs **28%** of what a 2,900-token RAG
prompt costs; on iGPU it costs 8%.

So the NPU pays a **~650 ms TTFT floor on every query** once configured for RAG. This refines
README §3's "NPU is much worse at prefill than decode": relative to iGPU the NPU is worst on
**short** prompts (9.57×), not long ones (2.84×), because the fixed floor dominates when there is
little real prefill to amortise it against. A closed-book query on a RAG-configured NPU is paying
for retrieval it never used.

## Two-device grid — complete

5,352 rows, **zero errors**, all 24 cells (2 devices × 4 models × 3 conditions) at 223.

### Device divergence across the full eval set

2,676 paired iGPU/NPU cases, same model, same byte-identical prompt, greedy decoding:

| | n | % |
|---|---|---|
| DIFFERENT wording | 829 | 31.0 |
| **DIFFERENT numeric** | **630** | **23.5** |
| one device empty | 506 | 18.9 |
| identical | 353 | 13.2 |
| both empty | 239 | 8.9 |
| same answer, verbosity differs | 119 | 4.4 |

**Only 17.6% agree.** Gate 2's 20-question result was not a small-sample artifact — it holds across
the whole eval set and is the stronger form of the claim.

### Divergence concentrates by model size — in the opposite direction to expectation

Counting only cases where **both** devices produced output, so the 0.6B's empties do not distort it:

| Model | scored | agree % | numeric diffs |
|---|---|---|---|
| 0.6B | 194 | **10.8%** | 96 |
| 1.7B | 408 | 21.8% | 192 |
| 4B | 660 | 28.8% | 162 |
| 8B | 669 | **32.9%** | 149 |

Agreement **rises monotonically with model size**. The expectation was that the 8B would diverge
most, being the only model past the NPU's 3–4 GB fast-memory boundary; the data says the opposite —
the 8B is the most device-stable model in the grid and the 0.6B the least. Larger models appear to
have sharper next-token distributions, so numerical differences in the accelerator flip the argmax
less often.

By question type, agreement is lowest on **type 6** (long-form synthesis, 1.9%) — the longest
outputs, where a single early divergence cascades — and **type 3** carries the most numeric
disagreements (207), which does match the expectation that numerical questions are most exposed.

By condition: Oracle-RAG agrees most (28.3%), closed-book least (9.3%).

### Resource contrast (medians)

| Condition | Device | Prefill | TTFT ms | TPOT ms | Wall s |
|---|---|---|---|---|---|
| closed-book | iGPU | 49 | 67.1 | 27.6 | 1.70 |
| closed-book | NPU | 49 | 651.8 | 54.3 | 3.17 |
| Oracle-RAG | iGPU | 143 | 88.7 | 28.7 | 1.29 |
| Oracle-RAG | NPU | 143 | 654.4 | 56.7 | 2.66 |
| RAG | iGPU | 2,905 | 842.7 | 55.3 | 3.32 |
| RAG | NPU | 2,905 | 2,624.6 | 74.4 | 9.61 |

## The 0.6B diagnostic — and a reversal

"The 0.6B fails" was ambiguous between *too small to reason* (which supports H2) and *too small to
follow the `/no_think` instruction* (which does not). Only a matched comparison can separate them,
so: 0.6B, Oracle-RAG, iGPU, the same 20 questions, run **both** with and without `/no_think` —
40 inferences, strictly **outside** the main grid because the no-`/no_think` arm uses a different
prompt and must never enter the size comparison.

A first version of this diagnostic ran only the no-`/no_think` arm and compared it against the Gate
2 closed-book baseline. That was confounded — it varied the prompt *and* the condition at once, and
would have credited the context's effect to the instruction. Rerun matched:

| | with `/no_think` | without |
|---|---|---|
| answered (non-empty) | **15/20** | **10/20** |
| median thinking tokens | **0** | **485** |
| hit the 512-token cap | 18/20 | 16/20 |

**This reverses the earlier reading.** `/no_think` does not break the 0.6B — it *helps* it. Removing
it makes the model spend 485 tokens reasoning and answer less often. Against the Gate 2 closed-book
baseline (2/20 answered, 511 median thinking tokens), the picture is:

| 0.6B, iGPU | answered | median thinking tokens |
|---|---|---|
| closed-book, `/no_think` | 2/20 | 511 |
| Oracle-RAG, `/no_think` | 15/20 | 0 |

**The 0.6B's failure is condition-dependent, not instruction-dependent.** Given context it complies
with `/no_think` and answers 15 of 20. Given none it burns the whole budget reasoning and produces
almost nothing.

That matters for how H2 is stated. The 0.6B is *not* "too small to follow the format instruction",
and the earlier framing in this document — that it is effectively non-functional under `/no_think` —
was wrong and is corrected here. Its closed-book collapse is about parametric knowledge, and the
open question H2 actually poses is whether retrieval closes that gap on reasoning-bound types,
which the grid measures directly.

Note it still hits the 512-token cap 18/20 times *with* zero thinking tokens — it rambles inside the
answer rather than inside a think block. The token cost is real either way, which is the H1 energy
point, and is why the cap was not raised: a model needing 1,000+ tokens to answer is itself the
result, not something to engineer around.
