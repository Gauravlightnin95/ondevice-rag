# Ingestion notes — Stage 1

Decisions frozen when `corpus/chunks.jsonl` was first built. README §6 makes chunking a
frozen-retriever step, so changing anything here invalidates every result above it.

Script: `src/ingest/chunk.py`. Output: `corpus/chunks.jsonl` (gitignored — see Privacy below).

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

## Open issue for Stage 2/3 — eval-set source aliases

5 of the 13 distinct eval-set `source` values do not match any corpus document stem:

```
amity-cg-brochure
galgotias-academic-monitoring-v3
galgotias-grievance-policy
galgotias-handbook-2026-27
galgotias-it-policy
```

These are almost certainly the hash-named corpus files — README §4 identifies
`NurqOlFSCg6EZdah90R9vyYUlAy0ghsHgjx2wltc.pdf` as the Galgotias handbook. An alias map from
eval-set source name to corpus filename is required before Stage 3 can validate those gold
passages. Stage 1 reports the gap; it does not resolve it.
