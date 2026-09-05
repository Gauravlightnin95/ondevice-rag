# On-Device Document QA — Retrieval or Parameters?

Research project measuring where retrieval-augmented generation stops compensating for
reduced model size, under a fixed resource budget on consumer hardware.

**This README is the build specification. Read all of it before writing any code.**

Current state: the evaluation set and corpus are complete and frozen. The next task is the
retrieval pipeline. See section 10 for what to build and section 11 for what not to.

---

## 1. What this project tests

Existing papers show that a small model with retrieval can match a much larger model. They
establish this by parameter count, on server hardware, reporting accuracy only.

This project re-runs that comparison with honest cost accounting — peak RAM including the
embedding model and vector index, time-to-first-token including inflated prefill, and
**joules per correct answer** — on a consumer laptop, with accuracy broken out by question
type rather than averaged.

### Hypotheses

**H1 (resource-matched crossover).** When cost is measured in peak RAM and joules per correct
answer rather than parameter count, the small-plus-retrieval advantage shrinks. Retrieval adds
an embedding model, a RAM-resident index, and a prompt 10–100× longer than the bare question.

**H2 (task boundary).** Retrieval closes the gap on knowledge-bound questions (single-hop
lookup) but not on reasoning-bound questions (multi-hop synthesis, numerical reasoning,
conflicting sources). Retrieval supplies facts, not reasoning capacity.

**H3 (distractibility scales inversely with size).** As the proportion of irrelevant retrieved
context rises, accuracy degrades faster for smaller models, so the optimal number of retrieved
chunks `k` is smaller for small models than the common default.

### Three experimental conditions

| Condition | Model receives | Isolates |
|---|---|---|
| Closed-book | Question only | What parameters alone know |
| RAG | Question + top-k retrieved chunks | The deployable system |
| Oracle-RAG | Question + known-correct gold passage | Pure reasoning, retrieval removed |

**Oracle-RAG is the most important design decision in this project.** Without it, a low score
is ambiguous — did retrieval fail to find the passage, or did the model have the passage and
still get it wrong? With it, every failure is attributable.

This is why every question carries an exact gold passage, and why those passages must match
the corpus text verbatim. See section 5.

---

## 2. Verified environment — do not re-investigate

All of this was tested on the target hardware. Treat it as settled fact.

### Hardware
- Intel Core Ultra 7 258V (Lunar Lake)
- NPU, integrated GPU and CPU all available and working
- NPU fast-memory region is roughly 3–4 GB; throughput degrades past it

### Software
- Windows, PowerShell, venv at `.venv`
- Python 3.12.10
- OpenVINO 2026.3.1-22476

### Model zoo

All four models exported from source with **identical** settings:

```
optimum-cli export openvino -m Qwen/Qwen3-0.6B --task text-generation-with-past \
  --weight-format int4 --sym --ratio 1.0 --group-size -1 models/qwen3-0.6b-cw
```

Same command for `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`.

| Folder | INT4 size | vs 3–4 GB NPU region |
|---|---|---|
| `models/qwen3-0.6b-cw` | ~0.4 GB | well inside |
| `models/qwen3-1.7b-cw` | ~1.0 GB | inside |
| `models/qwen3-4b-cw` | ~2.4 GB | inside |
| `models/qwen3-8b-cw` | ~4.7 GB | outside |

Identical quantisation across all four is deliberate. Pre-converted community checkpoints use
inconsistent schemes and are **not** comparable across sizes — this was tested and confirmed
(the pre-converted 4B produced garbled output on NPU). Never mix them in.

### OpenVINO GenAI API quirks — these cost real time to discover

```python
# Cache property differs by device. Passing NPUW_CACHE_DIR to GPU or CPU raises.
if device == "NPU":
    pipe = ov_genai.LLMPipeline(path, device, NPUW_CACHE_DIR=".npucache")
else:
    pipe = ov_genai.LLMPipeline(path, device, CACHE_DIR=".ovcache")

# add_generation_prompt is POSITIONAL, not a keyword argument.
# Template variables go in extra_context, NOT chat_template_kwargs.
prompt = tok.apply_chat_template(messages, True, extra_context={...})

# Some exports need the template set explicitly before use
try:
    tok.set_chat_template(tok.chat_template)
except Exception:
    pass
```

### Thinking mode — must stay off

Qwen3 has hybrid thinking. Left on it emits long reasoning blocks, which wrecks TPOT, inflates
joules per query, and breaks determinism.

**Append `/no_think` to the user message.** This is the uniform setting for all four models. Do
not vary it by model size — that would confound the size axis.

`extra_context={"enable_thinking": False}` was tested and is unreliable. Use `/no_think`.

Always log **raw token count** and **thinking token count** separately from the cleaned answer.
Thinking tokens are a real energy cost and they vary by model and device. Stripping them
silently hides that.

```python
def clean(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)   # unclosed tag
    text = re.sub(r"</?think>", "", text)
    return text.strip()
```

---

## 3. Findings from hardware characterisation

These are results, not bugs. Do not try to fix them.

**NPU is much worse at prefill than at decode.** On the 8B with a 4-token output (so almost
pure prefill), NPU took 1.91s against GPU 0.41s — about 4.7× slower. On longer generations the
decode gap is only ~1.7×. RAG is overwhelmingly prefill, so the NPU is weakest exactly where
RAG costs most. This is a headline result.

**iGPU beats NPU across the board.** Do not build any framing on the NPU winning.

**The 0.6B cannot reliably follow formatting instructions.** It ignores `/no_think`, opens a
`<think>` tag and never closes it, and hallucinated that the capital of France is Lille.
Relevant to H2 and to type 5 refusal behaviour. Record it, do not fix it.

**Greedy decoding is not deterministic across devices.** With `do_sample=False`, the same model
and prompt produced different outputs on different devices. On the 8B, GPU emitted 171 tokens
while NPU and CPU emitted 6.

This means the assumption that accuracy is device-independent is **unsafe**. Plan a check: run
~20 questions across all three devices and report the agreement rate. If they agree, the main
grid can run on one device and that choice is justified. If not, accuracy must be reported per
device.

### Device compatibility table (Qwen3-8B, warm cache)

| Device | Compile | Generate | ms/token |
|---|---|---|---|
| NPU | 4.8s | 1.86s | 465.7 |
| iGPU | 1.8s | 0.39s | 96.3 |
| CPU | 8.4s | 10.81s | 2703.5 |

All four models compile and run on all three devices. The fallback plan in the original
proposal is not needed.

---

## 4. Corpus — frozen, do not modify

`corpus/raw/` holds **29 PDFs, 507 pages**. `corpus/extracted/` holds the matching `.txt`
files with `===== PAGE N =====` markers.

Both directories are gitignored. `corpus/manifest.csv` is committed and is the provenance
record.

Screening applied: 55 documents collected, 26 rejected. Rejection reasons were scanned PDFs
with no text layer, OCR-corrupted text, byte-identical duplicates, and non-English content.
The screening script is `cleanup.py`.

**Do not add, remove or re-extract documents.** Every gold passage in the eval set was copied
from the current extracted text. Re-extracting with different settings would break them
silently.

### Privacy constraints

Two documents contain personal data:

- `Final_Mandatory_Disclosure.pdf` — tables of named faculty, staff and students with personal
  mobile numbers and roll numbers
- `NurqOlFS...pdf` (Galgotias handbook) — a live-looking Wi-Fi credential

No eval-set question draws on those sections, and gold passages were written to exclude names
and identifiers even where they sit in the same table.

**Consequences for any code you write:** do not surface those fields in logs, results tables or
outputs. When the eval set is released publicly, release the questions plus
`corpus/manifest.csv` with source URLs — not the extracted corpus text.

---

## 5. Evaluation set — frozen, do not modify

`evalset/` holds **223 questions** as Markdown, one file per source document plus two special
files.

| Type | Count | Description |
|---|---|---|
| 1 | 46 | Single-hop extractive |
| 2 | 50 | Multi-hop synthesis |
| 3 | 51 | Numerical / tabular |
| 4 | 25 | Conflicting sources |
| 5 | 29 | Absent answer |
| 6 | 22 | Long-form synthesis |

Counts are uneven against the nominal 30-per-type target. This is intentional and reported as
such. Bootstrap confidence intervals are computed per type, so unequal cells only widen
intervals on the smaller types.

### Question format

```markdown
## Q1
- **type:** 1
- **question:** What is the minimum attendance a student must maintain?
- **answer:** 75%
- **passage:** A student is expected to maintain full attendance in all courses...
- **page:** 29
- **source:** PART-A-ACADEMIC-REGULATIONS.pdf
- **annotator:** GK
- **contaminated:** unchecked
```

### Parser rules

- Read only lines matching `^## Q` and `^- \*\*field:\*\* value`. Skip everything else.
- Files under `evalset/notes/` are prose commentary. **Never parse them.**
- Files beginning with `_` have no single source document. Do not infer source from filename;
  read it from each question's fields.

### Type 4 — extra fields

Conflicting-source questions carry a second passage:

```markdown
- **passage:** <passage from first document>
- **passage2:** <passage from second document>
- **page:** 31
- **page2:** 32
- **source:** PART-A-ACADEMIC-REGULATIONS.pdf
- **source2:** galgotias-handbook-2026-27.pdf
```

**Oracle-RAG must supply both passages for these questions.** A correct answer requires
noticing that the corpus states two different things.

These live in `evalset/_conflicts.md`. Two sub-kinds, worth reporting separately:
- True contradictions within one institution's own documents (e.g. Part A says no minimum
  attendance for MTE, the 2026-27 handbook says 50%)
- Cross-source differences between institutions, where both are correct locally but a
  confident single-figure answer is still wrong

### Type 5 — nulls

Absent-answer questions have `passage: null`, `page: null`, `source: null`. There is no source
document; that is the definition of the type. The loader must accept nulls in those fields.

**Oracle-RAG needs an explicit rule for type 5.** Supply topically-related passages that do not
contain the answer, rather than nothing. That is the harder and more informative test of
refusal behaviour. State the choice in the paper.

These live in `evalset/_absent.md`. Absence was verified by regex search across all 29 extracted
documents (`verify_absent.py`).

Screening history, 36 candidates down to 29:

- 36 type 5 candidates drafted
- `verify_absent.py` flagged 6. Of those, 3 were deleted as genuine — the answer really was in the
  corpus — and 3 kept after manual review as false positives
- Q25 was deleted later, during passage assembly, when the MTE weightage of 30% turned up in the
  handbook
- 3 more trimmed to reach the target

Question numbering in `_absent.md` preserves the original candidate numbers, so it has holes at
9, 13, 25, 27, 28, 34 and 35. That is expected — IDs point at a fixed question rather than
shifting when one is removed.

---

## 6. Retrieval pipeline — the frozen-retriever rule

Everything above the LLM is built once, frozen, and never touched again. Only the LLM and the
retrieval condition vary. This is what protects the experiment from confounds.

Fixed parameters:

- **Chunking:** 512 tokens, 64 token overlap
- **Chunk IDs:** `{filename}::c{index:04d}` — must be stable and deterministic. Re-running
  chunking on the same document must produce identical IDs.
- **Embedding model:** `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB, MIT licensed)
- **Dense index:** FAISS, exact search
- **Sparse index:** BM25
- **Fusion:** reciprocal rank fusion

Record page numbers during chunking. `source_page` is needed for citation-correctness checks.

---

## 7. Metrics

### Quality
- **Accuracy** — rubric-graded 0/1/2 by an LLM judge, validated against human grading
- **Groundedness** — is the answer supported by the passages supplied
- **Citation correctness** — did it cite the right chunk
- **Refusal rate** — on type 5, did it correctly say the answer is absent
- **Judge agreement** — Cohen's κ against human grading on a 20% subset

Do not use exact string match. It punishes small models for formatting rather than for being
wrong.

### Resource
- **Peak RSS** — model + index + embedding model, measured together
- **Prefill tokens** — tokens in the final prompt
- **TTFT** — time to first token
- **TPOT** — time per output token, computed as elapsed / actual tokens generated
- **Thinking tokens** — logged separately, see section 2
- **Joules per query** — whole pipeline including embedding and search
- **Joules per correct answer** — energy ÷ accuracy. **This is the headline metric.**

TPOT must divide by actual tokens generated. Total wall-clock time is not comparable across
devices — a model that refuses early looks artificially fast. This was observed during hardware
testing.

### NPU power measurement

Lunar Lake has no dedicated NPU power sensor. Power is estimated as a residual: total system
power minus IA, GT, DRAM and system-agent rails, logged via HWiNFO. Baseline idle is around
13 W. Document this method in the paper as a stated limitation.

---

## 8. Statistical protocol

- Greedy decoding (`do_sample=False`) for the main grid
- Bootstrap confidence intervals resampled over the question set, 2,000 resamples
- Paired tests (paired bootstrap or Wilcoxon signed-rank), since every model sees the same
  questions
- Judge run 3× on a subset, self-agreement reported

**If the 95% confidence intervals for two configurations overlap, no difference has been
shown. Say so plainly.**

---

## 9. Project structure

```
ondevice-rag/
├── README.md                  # this file
├── requirements.txt
├── .gitignore
├── models/                    # gitignored, ~9 GB
│   ├── qwen3-0.6b-cw/
│   ├── qwen3-1.7b-cw/
│   ├── qwen3-4b-cw/
│   └── qwen3-8b-cw/
├── corpus/
│   ├── raw/                   # gitignored — 29 PDFs
│   ├── extracted/             # gitignored — 29 .txt
│   ├── rejected/              # gitignored — 26 screened out
│   └── manifest.csv           # committed
├── evalset/                   # committed — 223 questions
│   ├── _absent.md             # type 5
│   ├── _conflicts.md          # type 4
│   ├── <document>.md          # one per source document
│   └── notes/                 # prose — never parsed
├── src/
│   ├── ingest/                # chunking
│   ├── retrieval/             # embed, FAISS, BM25, fusion
│   ├── inference/             # grid runner, telemetry
│   └── analysis/              # grading, statistics, figures
├── results/                   # gitignored
├── docs/
└── *.py                       # existing scripts, see below
```

### Existing scripts — do not rewrite

| Script | Purpose |
|---|---|
| `check.py` | lists OpenVINO devices |
| `download_model.py` | pulls pre-converted models (superseded by self-conversion) |
| `extract.py` | PDF → text with page markers |
| `cleanup.py` | corpus screening, moves rejects |
| `verify_absent.py` | verifies type 5 answers are absent |
| `apply_passages.py` | one-off, already run |
| `test_npu.py` | device characterisation harness — reuse its pipeline setup |

`test_npu.py` already contains working device setup, cache handling, `/no_think` prompting and
token counting. Reuse that code rather than writing it again.

---

## 10. What to build next, in order

**Stage 1 — Ingestion.** Chunk the 29 extracted documents at 512 tokens with 64 overlap.
Deterministic chunk IDs. Preserve page numbers. Output to `corpus/chunks.jsonl`.

**Stage 2 — Eval set loader.** Parse `evalset/*.md` into `evalset/evalset_v1.jsonl`. Handle
type 4 second passages and type 5 nulls. Skip `evalset/notes/`.

**Stage 3 — Passage validator.** For each question, confirm the gold passage appears in the
extracted text of its source document. Report any that do not match. Whitespace may be
normalised for comparison; wording may not.

**Stage 4 — Retrieval.** Embed chunks with bge-small, build FAISS and BM25 indexes, implement
reciprocal rank fusion. Report Recall@k per question type — a single aggregate hides that
multi-hop retrieval is a different problem.

**Stage 5 — Freeze the retriever.** Tag it. Nothing above the LLM changes after this point.

**Stage 6 — Inference grid.** 4 models × 3 conditions × 223 questions. Log raw output, cleaned
output, raw token count, thinking token count, prefill tokens, TTFT, TPOT, peak RSS.

**Stage 7 — Telemetry and energy.** Per-query joules via the residual method in section 7.

**Stage 8 — Grading harness.** LLM judge, rubric 0/1/2, Cohen's κ against human grading on a
20% subset.

Each stage is checkable before the next begins. Do not skip ahead.

---

## 11. Do not build yet

- Anything past stage 5 until stages 1–4 are verified working
- The offline demo application (week 11 in the project plan)
- Figures and statistical analysis
- Rerankers, fine-tuning, additional models, additional embedding models — these go in
  "future work"

Scope creep is the single biggest risk to this project. Every addition costs time needed for
analysis and writing.

---

## 12. Hard rules

**Do not modify `evalset/` or `corpus/`.** Both are frozen. Changes become v1.1 as new files
and every affected result is re-run.

**Do not re-extract the corpus.** Gold passages were copied from the current extraction.

**Do not screen contamination with a model from the grid.** Using the 8B as both judge and
subject is circular. Screen with an external model, or document the limitation explicitly.

**Offline at inference time.** Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so silent
fallback-to-download becomes a loud error. Grading and contamination screening may run online
afterwards on logged outputs — offline is a property of the measured system, not the analysis
harness. State this in the paper.

**Reproducibility.** Record OpenVINO version, NPU driver version, model commit hashes and exact
export commands in `docs/hardware_notes.md`.

---

## 13. Outstanding manual work (not for Claude Code)

- **Human verification** of gold passages against source PDFs. The `annotator: GK` field
  asserts this has happened. It is in progress.
- **Contamination screening** — every question run closed-book, anything answered without
  documents deleted. Expect to lose 10–20%, mostly types 1 and 3.

EvalSet v1.0 is tagged only after both are complete.
