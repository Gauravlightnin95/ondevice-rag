# Ingestion notes

Stage 1 (chunking) and Stage 2 (eval set loading). Both produce frozen artifacts, so the
decisions here are recorded rather than left implicit in the code.

---

# Stage 1 — chunking

Decisions frozen when `corpus/chunks.jsonl` was first built. README §6 makes chunking a
frozen-retriever step, so changing anything here invalidates every result above it.

Script: `src/ingest/chunk.py`. Output: `corpus/chunks.jsonl` (gitignored — see Privacy below).

## Corpus rename, after the first build

Five documents were renamed from upload-hash filenames to descriptive ones after `chunks.jsonl`
was first generated:

| Was | Now |
|---|---|
| `CmlIGo2zF4aPg8zCk5WSpcVB23p6C7Ftw8kJB641` | `galgotias-grievance-policy` |
| `Iq4uYzH0okL24sa9FWIuY5rV90odO9Ni2r828Jx1` | `galgotias-academic-monitoring-v3` |
| `NurqOlFSCg6EZdah90R9vyYUlAy0ghsHgjx2wltc` | `galgotias-handbook-2026-27` |
| `YdoUIjxvgRKYHcWx0AmXssMbQQlj6t6KKdE1Bj5c` | `galgotias-it-policy` |
| `brochure-(details of programmes)` | `amity-cg-brochure` |

Because `doc_id` is derived from the filename, every affected `chunk_id` changed with it. Chunk
count (491), per-document counts and page total (507) are unchanged — the rename touched names
only, not content, so this is not a re-extraction and README §12 is not in play.

This closed the source-alias gap Stage 1 reported: all 13 distinct eval-set sources now join to a
`doc_id` directly, and no alias map is needed.

## Frozen parameters

| Parameter | Value | Source |
|---|---|---|
| Chunk size | 512 tokens | README §6 |
| Overlap | 64 tokens | README §6 |
| Stride | 448 tokens | derived |
| Tokenizer | `BAAI/bge-small-en-v1.5` | decision below |
| Tokenizer commit | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | resolved at first run |

## Deviation from README §6 — chunk ID format

README §6 specifies `{filename}::c{index:04d}`. **What was built is
`{lowercased stem}::c{index:04d}`** — e.g. `part-a-academic-regulations::c0000`, not
`PART-A-ACADEMIC-REGULATIONS.txt::c0000`.

Reason: chunking reads `.txt` files, but the eval set's `source:` fields name `.pdf` files. A
literal `{filename}` would embed whichever extension the chunker happened to read, forcing every
downstream join to rewrite the ID. The bare stem joins to both. Lowercasing additionally absorbs
`GLBITM-IDP24-29.PDF`, the one corpus file whose extension case differs from its eval-set
reference.

Approved as a deliberate, documented deviation. Chunk IDs are now frozen in this form.

## Tokenizer — bge-small, not Qwen3

The 512-token budget is measured in bge-small-en-v1.5 WordPiece tokens, even though the Qwen3
tokenizers are already local and the LLM's prefill accounting uses Qwen BPE.

bge-small's context window is 512 tokens. Chunking in Qwen BPE would produce chunks of roughly
565–615 WordPiece tokens, and bge would silently truncate 10–17% off the tail of every chunk at
embed time. FAISS would then index less text than BM25 scores, and less than the LLM is shown in
the RAG prompt — a confound baked into a retriever that is about to be frozen and never touched
again.

Chunks are cut at 512 tokens with `add_special_tokens=False`, so 512 is the *content* budget.
bge prepends `[CLS]` and appends `[SEP]` at embed time, pushing the sequence to 514; the two
tokens beyond the window sit inside the 64-token overlap and are carried by the next chunk, so
nothing is lost from the index.

Cost of this choice: the tokenizer was downloaded once at build time. README §12 scopes offline
operation to inference, not the build; the commit hash above pins it for reproducibility.

## Whitespace normalisation

Applied per page before tokenisation, deterministically:

- trailing spaces and tabs stripped from each line
- runs of 3+ newlines collapsed to 2
- leading/trailing blank lines dropped per page
- pages joined with a single `\n`

Intra-line spacing is preserved so table rows stay readable.

WordPiece discards whitespace, so this changes **no** token counts and no chunk boundaries. It
exists because the same text is later placed in the LLM prompt, where PyMuPDF's per-line trailing
spaces and blank-line runs would inflate the prefill-token metric — a reported figure feeding
joules per query.

Consequence: `char_start` / `char_end` are offsets into the **normalised** document, not into the
raw `.txt`. Normalisation is a pure function of the source file, so the exact text is always
recoverable by re-running `normalise()`.

## Page attribution

Chunks routinely cross page boundaries — 407 of 491 span more than one page, up to 7 pages for
one chunk in a slide-style document. Each record therefore carries:

- `source_page` — the page containing `char_start`; this is README §6's citation-correctness field
- `page_start`, `page_end`, `pages[]` — the full span

A page with no text layer contributes a zero-width span, so page accounting stays exact (507)
without injecting phantom content into the token stream.

## Privacy

`corpus/chunks.jsonl` is gitignored. It carries verbatim extracted text, including the named
faculty/staff/student rows with mobile and roll numbers in `Final_Mandatory_Disclosure.pdf` and
the Wi-Fi credential in the Galgotias handbook (README §4). `.gitignore` previously covered only
`corpus/raw|extracted|rejected/`, not a file sitting directly in `corpus/` — the rule was added
before the file was first generated.

The chunker's console summary prints counts only, never chunk text.

## Verified at build time

- 29 documents, 507 pages — asserted against README §4, not assumed
- 491 chunks; tokens per chunk min 71 / median 512 / max 512
- All `chunk_id` unique; no chunk over 512 tokens
- Stride exactly 448 and overlap exactly 64 between every consecutive pair within a document,
  asserted on token indices. A gap check alone would pass an off-by-one stride — no text lost,
  but the overlap silently 63 or 65 and every boundary shifted. Equality also proves the
  final-window break never emits a stub already contained in its predecessor.
- No character gaps between consecutive chunks
- Determinism: built, hashed, deleted, rebuilt — SHA-256 identical
  (`82EDF4B50D9934B7947CE8B68249C16168F77D68772AC20339F6913DA9E71097`)

## Resolved — eval-set source aliases

Stage 1's first run reported 5 of the 13 distinct eval-set `source` values with no matching corpus
document stem. The corpus rename above resolved this: all 13 now join directly, confirmed by
Stage 2's source-join check. No alias map is needed.

---

# Stage 2 — eval set loader

Script: `src/ingest/load_evalset.py`. Output: `evalset/evalset_v1.jsonl` (gitignored — it embeds
gold passages copied from the extracted corpus, so it falls under the same non-redistribution rule
as `corpus/extracted/`. README §4's public release is the questions plus `manifest.csv`, which is a
passage-stripped export and a separate deliberate step).

## Question ID — `{md stem}::Q{n}`

e.g. `part-a-academic-regulations::Q7`, `_absent::Q36`, `_conflicts::Q3`. Mirrors Stage 1's
`chunk_id` convention.

Numbering restarts at Q1 in each of the 15 markdown files, so `## Q1` alone is not unique — 15
questions share it. `_absent.md` additionally has holes (see README §5), so the ID deliberately
preserves the original number rather than renumbering: an ID always points at the same markdown
block. Renumbering would silently reassign IDs whenever a question is added or removed, which would
invalidate inference results already logged against them.

## Page fields — raw string plus parsed list

21 of the 219 page references are not plain integers. 15 distinct forms, in two shapes: ranges
(`1-2`, `29-31`, `3-8`) and comma lists (`2, 5, 7`, `6, 8`).

Each record therefore carries both `page` (the frozen text verbatim) and `page_list` (parsed
ints), and `page2`/`page2_list` for type 4. The list intersects directly with a chunk's `pages[]`
for citation-correctness; keeping the raw string means the artifact still reproduces the eval set
exactly. Type 5 gets `page: null`, `page_list: []`.

`source_doc_id` is precomputed as the lowercased extension-free stem, so no downstream stage
re-derives the join key into `chunks.jsonl`. Non-type-4 records omit the `*2` keys entirely rather
than padding them with nulls.

## Correction to `evalset/_conflicts.md` Q3 — 2026-09-04

| | |
|---|---|
| Field | `page2` on `_conflicts::Q3` |
| Was | `55` |
| Now | `51` |
| Found by | the Stage 2 page-bounds check, not by review |

`PART-A-ACADEMIC-REGULATIONS` has 53 pages, so `page2: 55` could not refer to anything. The
`passage2` text ("*the grievant may appeal to the Vice Chancellor within 7 calendar days*") was
located in `corpus/extracted/PART-A-ACADEMIC-REGULATIONS.txt` under `===== PAGE 51 =====`. The
question's other page reference, `page: 49`, was verified correct and left alone.

Only that one field changed — no question, answer, passage or count was touched, and the type-4
total stays 25. The correction was made before the loader's first run, so no artifact ever carried
page 55. Freezing the eval set means changes are deliberate and recorded; it does not mean errors
survive.

The bounds check and its non-zero exit path remain in the script unchanged. A check that exists
only while it is failing is worthless for Stage 3 onward, so it was verified to still fire: run
against a scratch copy with `page2: 999`, it named `_conflicts::Q3` and exited 1.

## Verified at build time

- 223 questions, per type exactly 46 / 50 / 51 / 25 / 29 / 22 — hard assert, since a silent parser
  drop would surface downstream as a quietly smaller grid rather than an error
- `## Q` header count equals emitted record count — catches a question absorbed into its predecessor
- All 223 `qid` unique
- 25 type-4 questions carry `passage2`/`page2`/`source2`, and no other type does
- 29 type-5 questions have `passage`/`page`/`source` null, and no other type does
- All 13 distinct sources (219 references) join to a `doc_id` in `chunks.jsonl`
- Every parsed page number falls within its own source document's marker count
- Determinism: built, hashed, deleted, rebuilt — SHA-256 identical
  (`996CAA4776DC8D7A39B35359433303A6953200B6850AB9AFDEFCB3464F29BE60`)

Field order varies between questions — `_conflicts.md` Q1 runs passage, passage2, page, source,
source2, page2 — so the parser accumulates into a dict and never reads fields positionally. No
field value wraps onto a continuation line anywhere in the eval set, which is what makes the
strict `^- \*\*field:\*\* value` rule safe: it cannot silently truncate a long passage.

---

# Stage 3 — passage validator / locator

Script: `src/ingest/validate_passages.py`. Output: `evalset/passage_locations.jsonl` (gitignored;
derived from two gitignored artifacts).

## Two questions, kept apart

**Extraction fidelity** — does the gold passage appear verbatim in its source? Reported as a
statistic, never gating. Oracle-RAG feeds the passage *as written in the eval set*, so whitespace
and glyph drift never reaches the model.

**Locatability** — can the passage be mapped to a chunk? This gates. Stage 4's Recall@k needs each
gold passage's `chunk_id`; an unlocatable passage means no retrieval recall for that question.

Conflating the two would have been the expensive mistake here: a strict-verbatim gate would have
blocked Stage 4 on 77 passages that are all perfectly locatable.

## Result

| | |
|---|---|
| Non-null passages | 219 (194 `passage` + 25 `passage2`) |
| **Strict verbatim** | **143 / 219 (65.3%)** — reported, not gating |
| **Locatable to ≥1 chunk** | **219 / 219** — the gate, passing |
| Passages containing text absent from their source | **0** |

Fidelity layers: L0 strict 143, L1 char-normalised 3, L2 whitespace-insensitive 7, L3 gap-tolerant
66. The 65.3% verbatim rate is a property of PyMuPDF extraction, not of the questions, and is worth
stating in the paper as the extraction-fidelity figure.

## Why passages fail strict verbatim

None is annotator paraphrase. Every one was traced:

- **PUA bullet glyphs** — Symbol/Wingdings bullets survive extraction as `U+F0B7`, or as stray
  `o` / `y` tokens between passage fragments
- **`U+FFFD`** where a smart quote failed to decode: `Dean�s office` vs `Dean's office`
- **Hyphenated line breaks** — `extra-\ncurricular` extracts as `extra- curricular`
- **Lost inter-word spaces** — the document contains `belowtimings`, `tothecampus`
- **Interleaved list numbers and table headers** between passage fragments
- **Column-wise table extraction** — AICTE approval tables read row-wise by the annotator come out
  column-wise, scattering the figures document-wide
- **Privacy redactions** — exactly 2, below

## Character substitution table

Applied to both sides before comparison.

| Input | Becomes |
|---|---|
| `U+F000`–`U+F8FF` (private use) | deleted |
| `U+FFFD` replacement char | deleted |
| `'` `"` and curly quotes `‘ ’ ‚ ‛ “ ” „` | deleted |
| `– — ― ‐ ‑ −` | `-` |
| `ﬀ ﬁ ﬂ ﬃ ﬄ` | `ff fi fl ffi ffl` |
| `U+00A0` | space |
| `…` | `...` |

Quotes are deleted rather than mapped because `U+FFFD` stands in for a quote the extractor lost;
deleting on both sides is the only symmetric fix.

The **L2** layer strips whitespace entirely rather than collapsing it. That resolves the glued
words and the hyphenated line breaks under one rule instead of two special cases.

## Locating — minimum-window alignment with density

Locating never uses the fidelity layers. It runs against the document string rebuilt with
`chunk.py`'s own `split_pages` and `build_document`, so a located character span maps to chunk IDs
by **offset overlap** rather than by re-matching text — verified, all 491 chunks reproduce
byte-identically from the rebuilt offsets.

For passage tokens `P` and document tokens `D`, find the shortest window of `D` containing `P` in
order, anchored on `P`'s rarest token, and score it `density = len(P) / window_length`. Density is
the quality measure a plain gap-window lacks: ≈1.0 is clean, 0.5 means half the window is
interleaved bullets and table headers. **Floor 0.30** — a window below it is rejected rather than
accepted, because for Recall@k a *wrong* chunk is worse than a missing one: it corrupts the metric
silently instead of showing up as a gap.

Composite passages fall out of the same mechanism: take the longest prefix holding above the floor,
emit it, continue with the remainder. A token that cannot start any acceptable segment is skipped
rather than failing the whole passage — the scattered table figures above would otherwise discard
passages that are 90%+ locatable. **Coverage floor 0.60** on the fraction of passage tokens landing
inside an emitted span guards against a long passage being "located" by one stray fragment.

An earlier naive sentence-splitting version fragmented mid-table and made things worse; the
minimum-window formulation replaced it.

## Privacy whitelist — hard-coded, two qids

```
Students-GrievanceRedressal-Cell::Q1   names and roll number removed per README section 4
Students-GrievanceRedressal-Cell::Q3   names and designations removed per README section 4
```

Reported under their own heading: *the redacted form is the correct state, do not fix*. Hard-coded
by ID, never by heuristic — a heuristic that mislabelled a genuinely broken passage as a redaction
would hide it permanently and Stage 4 would never locate its chunk. The whitelist governs
*reporting*, not the gate: both still locate (2 chunks each, coverage 100%), and if either ever
failed to locate that would still be a gate failure. The script asserts both qids still exist, so a
future renumbering cannot leave a stale exemption in place.

## Correction to page / page2 fields — 2026-09-05

**55 field values corrected across 5 markdown files.** Authorised; the second deliberate change to
the eval set after the Q3 `page2` fix.

**What was wrong.** Page numbers for four documents were taken from each document's own table of
contents, which lists the page number *printed on the page*, while the corpus numbers physical
pages from 1. The gap is the front matter. The gold passages were always in the right place — only
the recorded page numbers were shifted.

**How it was found.** Stage 3's page-corroboration check, which compares the located chunk's pages
against the recorded page. It uses a signal the matcher never sees, so it catches exactly this.

**Why the fields were corrected rather than the scoring.** The `page` field now means
page-in-corpus, with no translation layer. README §7's citation-correctness metric scores whether
the model cited the right page; against the uncorrected fields a correct handbook citation would
have been marked wrong 32 times out of 37. An offset table was deliberately not built — see below
for why it could not have worked anyway.

**Source of truth** was each passage's own located character spans, not `chunk_pages`. A chunk
spans up to 7 pages, so `chunk_pages` would have widened the field rather than corrected it. The
validator now emits `passage_pages` for this: the pages the passage itself sits on.

| File | Corrections | Offsets applied |
|---|---|---|
| `galgotias-handbook-2026-27.md` | 19 | +4 |
| `_conflicts.md` (`page2`, handbook-sourced; `page`, part-b) | 18 | +4, +1 |
| `PART-B-Examination-Rules-Regulations.md` | 13 | +1 in 10, **−1** in 1, **+2** in 1 |
| `GLBITM-IDP24-29.md` | 3 | +3 |
| `galgotias-it-policy.md` | 2 | +1 |

**The offsets are not constant per document.** part-b runs +1 for ten questions but −1 for Q17 and
+2 for Q18; the handbook is +4 for 32 questions and +5 for 2. A per-document offset table would
have silently mis-corrected those. Each field was set from its own passage's location instead.

**Six discrepancies were deliberately left alone**, and are not errors:

- Five where the recorded pages are a *subset* of the located pages — the passage genuinely starts
  on the recorded page and runs onto the next. `ABESEC_EOA_2026-27::Q6`/`Q7`,
  `galgotias-academic-monitoring-v3::Q13`, `HOSTEL-RULE-BOOKLET::Q17`,
  `Students-GrievanceRedressal-Cell::Q3`.
- `boardof-studies-2024-25::Q7`, recorded `3-8`, located on `1, 3, 4, 6, 8` across 5 spans. The
  annotator's contiguous range is a sounder description of a passage read across one table than my
  fragment set is; "correcting" it would have injected aligner noise into the frozen eval set.

Every replacement asserted the existing value matched what the correction expected before writing,
so no field was overwritten on a guess. CRLF line endings were preserved in all five files.

**After the correction:** Stage 2 re-run — 223 questions, 46/50/51/25/29/22, page bounds 0
violations, exit 0. Stage 3 re-run — **page corroboration 219/219**, locatability 219/219, exit 0.
Against the stricter precise-page comparison, disjoint cases went 54 → **0**.

## Multi-chunk gold passages — Recall@k rule, decided

70 passages map to more than one chunk, and they split into two populations that must not be
treated as one:

- **55 have a single span straddling a chunk boundary.** The passage is in one place and merely
  crosses a cut — the 64-token overlap makes this common.
- **15 are genuinely composite**, assembled from several locations. Types 2 (4), 3 (6) and 6 (5).

**Decided rule:**

- **Any-hit is the primary Recall@k**, for comparability with published work.
- **The 55 boundary-straddling passages are any-hit only.** The passage exists in one place; which
  side of a cut it lands on is an artifact of chunking, not a retrieval outcome.
- **The 15 composite passages are additionally reported under all-hit**, separately — never
  averaged into the headline.

The reason for the split report is H2. Those 15 are exactly types 2, 3 and 6 — multi-hop synthesis,
numerical/tabular and long-form synthesis. All-hit recall on them measures whether retrieval
actually supplies the *whole* evidence set that multi-hop reasoning needs, rather than one fragment
of it. Averaging that into a single any-hit figure would inflate recall precisely on the questions
H2 is about, and would hide the distinction the hypothesis is testing.

The 15 qids are listed in the Stage 3 report output. `passage_locations.jsonl` carries `n_spans`
and `chunk_ids` per passage, so Stage 4 can partition the two populations without re-deriving them.

## Verified at build time

- 219/219 passages locate to at least one chunk; exit 0
- 219/219 page corroboration after the page correction above
- Determinism: built, hashed, deleted, rebuilt — SHA-256 identical
- Located spans round-trip: slicing the rebuilt document at the reported offsets reproduces the
  passage, at L0, L3-single-span and L3-multi-span alike, and every reported `chunk_id` genuinely
  overlaps a reported span
- **The checks were proven to fail.** Against scratch copies (mutation confirmed present in the
  file before trusting any result — literal string replacement, never an anchored regex, since a
  `$`-anchored regex silently no-opped on CRLF during Stage 2):
  - one word corrupted → strict verbatim drops 143→142 and the qid is listed as a mismatch, while
    the gate still passes, which is correct: the passage remains locatable
  - passage fully corrupted → `UNLOCATED`, coverage 0%, exit 1

Nothing under `evalset/` or `corpus/` was modified. Passages were deliberately **not** edited to
match extraction artifacts: writing PUA glyphs and missing spaces into the frozen eval set would
encode extraction bugs into the artifact and make Oracle-RAG feed the model corrupted text.
