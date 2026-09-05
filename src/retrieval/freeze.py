"""
Stage 5 — Freeze the retriever.

Snapshots the built index into index/frozen/<version>/ and writes the freeze contract:
FREEZE.md for a human, fixture.json for verify_frozen.py.

The binaries cannot go into git — bm25.pkl stores the tokenised corpus and chunks.jsonl
the raw text, both carrying the personal data README section 4 excludes from
redistribution. So the CONTRACT is what gets committed and tagged: every parameter,
every artifact hash, and the expected top-10 for all 194 gold-passage questions. The
tag pins the contract; verify_frozen.py proves the local binaries still satisfy it.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\retrieval\\freeze.py
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrieve as R                                    # noqa: E402
from build_index import (BM25_B, BM25_K1, EMBED_MODEL,  # noqa: E402
                         EMBED_REVISION, MAX_LENGTH, QUERY_PREFIX)

VERSION = "v1"
INDEX = Path("index")
FROZEN = INDEX / "frozen" / VERSION
CHUNKS = Path("corpus/chunks.jsonl")
EVALSET = Path("evalset/evalset_v1.jsonl")
CORPUS = Path("corpus/extracted")

PROBE_K = 10
BINARIES = ["faiss.index", "bm25.pkl", "embeddings.npy", "chunk_ids.json", "chunks.jsonl"]

# Chunking parameters live in chunk.py; restated here so the contract is self-contained.
CHUNK_MAX_TOKENS, CHUNK_OVERLAP = 512, 64


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    if FROZEN.exists():
        raise SystemExit(
            f"{FROZEN} already exists. A freeze that can be silently rewritten is not a\n"
            f"freeze — cutting a new version is a deliberate act. Bump VERSION to v2 if\n"
            f"the retriever genuinely changed, or delete {FROZEN} if this one was never used.")

    for name in ("faiss.index", "bm25.pkl", "embeddings.npy", "chunk_ids.json",
                 "manifest.json"):
        if not (INDEX / name).exists():
            raise SystemExit(f"missing {INDEX / name} — run build_index.py first")

    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    chunk_ids = json.loads((INDEX / "chunk_ids.json").read_text(encoding="utf-8"))
    questions = [json.loads(l) for l in EVALSET.open(encoding="utf-8")]
    probes_src = [q for q in questions if q["type"] != 5]

    print(f"Freezing retriever as {VERSION}")
    print(f"  {len(chunk_ids)} chunks, {len(probes_src)} probe questions")

    # ---- snapshot the artifacts
    FROZEN.mkdir(parents=True)
    for name in ("faiss.index", "bm25.pkl", "embeddings.npy", "chunk_ids.json"):
        shutil.copy2(INDEX / name, FROZEN / name)
    shutil.copy2(CHUNKS, FROZEN / "chunks.jsonl")     # self-contained freeze

    # ---- capture expected rankings through the REAL search path
    print(f"  running {len(probes_src)} probes through search()...")
    probes = []
    for i, q in enumerate(probes_src, 1):
        top = [hit[0] for hit in R.search(q["question"], k=PROBE_K)]
        probes.append({"qid": q["qid"], "question": q["question"], "top_k": top})
        if i % 50 == 0 or i == len(probes_src):
            print(f"    {i}/{len(probes_src)}")

    n_pages = sum(
        p.read_text(encoding="utf-8").count("===== PAGE ")
        for p in CORPUS.glob("*.txt"))

    fixture = {
        "freeze_version": VERSION,
        "frozen_at": date.today().isoformat(),
        "git_commit": git_commit(),
        "corpus": {
            "n_documents": len(list(CORPUS.glob("*.txt"))),
            "n_pages": n_pages,
            "n_chunks": len(chunk_ids),
        },
        "chunking": {
            "max_tokens": CHUNK_MAX_TOKENS,
            "overlap": CHUNK_OVERLAP,
            "stride": CHUNK_MAX_TOKENS - CHUNK_OVERLAP,
            "tokenizer": EMBED_MODEL,
        },
        "embedding": {
            "model": EMBED_MODEL,
            "revision": EMBED_REVISION,
            "dim": manifest["embed_dim"],
            "pooling": manifest["pooling"],
            "normalisation": manifest["normalisation"],
            "max_length": MAX_LENGTH,
        },
        "query_prefix": QUERY_PREFIX,
        "query_prefix_applied_to": "queries only",
        "dense_index": manifest["dense_index"],
        "sparse": {
            "impl": manifest["sparse_index"],
            "k1": BM25_K1,
            "b": BM25_B,
            "tokenisation": manifest["bm25_tokenisation"],
        },
        "fusion": {
            "method": "rrf",
            "k": R.RRF_K,
            "candidate_depth": R.CANDIDATE_DEPTH,
            "tie_break": "chunk_id",
        },
        "hashes": {name.split(".")[0] if name != "faiss.index" else "faiss_index":
                   sha256_file(FROZEN / name) for name in BINARIES},
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "faiss": __import__("faiss").__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": __import__("numpy").__version__,
        },
        "probes": {"n": len(probes), "k": PROBE_K, "expected": probes},
    }

    assert_text_free(fixture, questions, chunk_ids)

    (FROZEN / "fixture.json").write_text(
        json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (FROZEN / "FREEZE.md").write_text(freeze_md(fixture), encoding="utf-8")

    print(f"\nFrozen to {FROZEN}/")
    print(f"  tracked : FREEZE.md, fixture.json")
    print(f"  ignored : {', '.join(BINARIES)}")
    print(f"\nVerify with:  .venv\\Scripts\\python.exe src\\retrieval\\verify_frozen.py")


def assert_text_free(fixture, questions, chunk_ids):
    """No corpus or passage text may reach the committed fixture.

    Asserted as a positive whitelist: every probe question must be a verbatim eval-set
    question, and every expected result must be a known chunk_id. A whitelist cannot be
    fooled by text that merely looks safe, which a blacklist scan can.
    """
    valid_questions = {q["question"] for q in questions}
    valid_ids = set(chunk_ids)

    for probe in fixture["probes"]["expected"]:
        assert probe["question"] in valid_questions, \
            f"{probe['qid']}: probe text is not a verbatim eval-set question"
        for cid in probe["top_k"]:
            assert cid in valid_ids, f"{probe['qid']}: '{cid}' is not a known chunk_id"

    # Belt and braces: no window of any chunk's text may appear in the serialised fixture.
    blob = json.dumps(fixture, ensure_ascii=False)
    for line in (FROZEN / "chunks.jsonl").open(encoding="utf-8"):
        text = json.loads(line)["text"]
        for start in range(0, max(1, len(text) - 60), 60):
            window = " ".join(text[start:start + 60].split())
            if len(window) >= 40:
                assert window not in blob, f"chunk text leaked into fixture: {window[:50]!r}"

    print(f"  fixture is text-free: {len(fixture['probes']['expected'])} probes, "
          f"questions and chunk_ids only")


def freeze_md(f):
    return f"""# Frozen retriever — {f['freeze_version']}

Frozen {f['frozen_at']} at commit `{(f['git_commit'] or 'unknown')[:12]}`, tagged `retriever-{f['freeze_version']}`.

README §6: everything above the LLM is built once, frozen, and never touched again. Only
the LLM and the retrieval condition vary after this point. That is what keeps model-size
effects separable from retrieval effects across the 4x3x223 grid.

## What is frozen

| | |
|---|---|
| Documents / pages / chunks | {f['corpus']['n_documents']} / {f['corpus']['n_pages']} / {f['corpus']['n_chunks']} |
| Chunk size | {f['chunking']['max_tokens']} tokens |
| Overlap | {f['chunking']['overlap']} tokens (stride {f['chunking']['stride']}) |
| Chunk tokenizer | `{f['chunking']['tokenizer']}` |
| Embedding model | `{f['embedding']['model']}` |
| Embedding commit | `{f['embedding']['revision']}` |
| Dimensions | {f['embedding']['dim']} |
| Pooling | {f['embedding']['pooling']} (not mean pooling) |
| Normalisation | {f['embedding']['normalisation']} — inner product is cosine |
| Max length | {f['embedding']['max_length']} |
| Query prefix | `{f['query_prefix']}` |
| Prefix applied to | **{f['query_prefix_applied_to']}** — never to indexed chunks |
| Dense index | `{f['dense_index']}` (exact, no approximation) |
| Sparse index | `{f['sparse']['impl']}`, k1={f['sparse']['k1']}, b={f['sparse']['b']} |
| BM25 tokenisation | {f['sparse']['tokenisation']} |
| Fusion | {f['fusion']['method'].upper()}, k={f['fusion']['k']}, ties broken on {f['fusion']['tie_break']} |
| Candidate depth | top-{f['fusion']['candidate_depth']} from each retriever before fusion |

Every one of these was fixed a priori. None was selected by comparing Recall on the eval
set — see `docs/ingestion_notes.md` for why that matters to the claim in README §6.

## What is NOT frozen

The LLM (4 models) and the retrieval condition (closed-book, RAG, Oracle-RAG). Those are
the experiment's only variables. `k`, the number of chunks placed in the prompt, is also a
variable — H3 predicts the optimum is smaller for smaller models.

## Environment it was frozen on

Python {f['environment']['python']}, torch {f['environment']['torch']}, faiss {f['environment']['faiss']},
transformers {f['environment']['transformers']}, numpy {f['environment']['numpy']}.

## Files

`FREEZE.md` and `fixture.json` are tracked in git and covered by the tag. The artifacts
themselves — `faiss.index`, `bm25.pkl`, `embeddings.npy`, `chunk_ids.json`, `chunks.jsonl` —
are gitignored: `bm25.pkl` holds the tokenised corpus and `chunks.jsonl` the raw text, both
carrying the personal data README §4 excludes from redistribution. The tag pins the
contract, not the data; the verifier proves local artifacts satisfy it.

A fresh clone therefore has the contract and no artifacts. Rebuild them with:

```
.venv\\Scripts\\python.exe src\\retrieval\\build_index.py
.venv\\Scripts\\python.exe src\\retrieval\\freeze.py
```

## Verify

```
.venv\\Scripts\\python.exe src\\retrieval\\verify_frozen.py
```

Checks {f['probes']['n']} probe queries — every eval-set question with a gold passage — against their
recorded top-{f['probes']['k']}, plus all five artifact hashes. Verdicts: `BYTE_IDENTICAL`,
`BEHAVIOURALLY_IDENTICAL` (hashes moved, rankings did not), `DRIFTED`, `ARTIFACTS_MISSING`.
Stage 6 calls this before starting the grid and aborts on `DRIFTED`.
"""


if __name__ == "__main__":
    main()
