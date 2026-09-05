# Frozen retriever — v1

Frozen 2026-09-05 at commit `625218c55cbe`, tagged `retriever-v1`.

README §6: everything above the LLM is built once, frozen, and never touched again. Only
the LLM and the retrieval condition vary after this point. That is what keeps model-size
effects separable from retrieval effects across the 4x3x223 grid.

## What is frozen

| | |
|---|---|
| Documents / pages / chunks | 29 / 507 / 491 |
| Chunk size | 512 tokens |
| Overlap | 64 tokens (stride 448) |
| Chunk tokenizer | `BAAI/bge-small-en-v1.5` |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding commit | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Dimensions | 384 |
| Pooling | cls (not mean pooling) |
| Normalisation | l2 — inner product is cosine |
| Max length | 512 |
| Query prefix | `Represent this sentence for searching relevant passages: ` |
| Prefix applied to | **queries only** — never to indexed chunks |
| Dense index | `faiss.IndexFlatIP` (exact, no approximation) |
| Sparse index | `rank_bm25.BM25Okapi`, k1=1.5, b=0.75 |
| BM25 tokenisation | lowercase [0-9a-z]+, no stemming, no stopwords |
| Fusion | RRF, k=60, ties broken on chunk_id |
| Candidate depth | top-100 from each retriever before fusion |

Every one of these was fixed a priori. None was selected by comparing Recall on the eval
set — see `docs/ingestion_notes.md` for why that matters to the claim in README §6.

## What is NOT frozen

The LLM (4 models) and the retrieval condition (closed-book, RAG, Oracle-RAG). Those are
the experiment's only variables. `k`, the number of chunks placed in the prompt, is also a
variable — H3 predicts the optimum is smaller for smaller models.

## Environment it was frozen on

Python 3.12.10, torch 2.13.0+cpu, faiss 1.15.0,
transformers 5.5.4, numpy 2.4.6.

## Files

`FREEZE.md` and `fixture.json` are tracked in git and covered by the tag. The artifacts
themselves — `faiss.index`, `bm25.pkl`, `embeddings.npy`, `chunk_ids.json`, `chunks.jsonl` —
are gitignored: `bm25.pkl` holds the tokenised corpus and `chunks.jsonl` the raw text, both
carrying the personal data README §4 excludes from redistribution. The tag pins the
contract, not the data; the verifier proves local artifacts satisfy it.

A fresh clone therefore has the contract and no artifacts. Rebuild them with:

```
.venv\Scripts\python.exe src\retrieval\build_index.py
.venv\Scripts\python.exe src\retrieval\freeze.py
```

## Verify

```
.venv\Scripts\python.exe src\retrieval\verify_frozen.py
```

Checks 194 probe queries — every eval-set question with a gold passage — against their
recorded top-10, plus all five artifact hashes. Verdicts: `BYTE_IDENTICAL`,
`BEHAVIOURALLY_IDENTICAL` (hashes moved, rankings did not), `DRIFTED`, `ARTIFACTS_MISSING`.
Stage 6 calls this before starting the grid and aborts on `DRIFTED`.
