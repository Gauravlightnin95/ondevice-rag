"""
Stage 4 — Build the frozen retriever: embeddings, FAISS and BM25.

Every parameter here is fixed a priori and recorded in the manifest. None was chosen by
looking at a Recall number: selecting retriever settings by their score on the eval set
they are then measured on would void the frozen-retriever claim in README section 6.
See docs/ingestion_notes.md.

Both indexes are built from ONE ordered chunk list in one pass. If the dense and sparse
sides ever indexed different chunk universes, RRF would fuse rankings over mismatched
sets and still emit chunk_ids that look perfectly valid — a wrong Recall number with
nothing visibly broken. That is asserted here, not assumed.

Run from C:\\ondevice-rag
    .venv\\Scripts\\python.exe src\\retrieval\\build_index.py
"""

import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

CHUNKS = Path("corpus/chunks.jsonl")
OUT = Path("index")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_LENGTH = 512
BATCH = 32

# Applied to QUERIES ONLY, never to indexed chunks. The bge model card specifies this
# asymmetry for short-query-to-long-passage retrieval; applying it to both sides would
# be a silent bug that still produces plausible numbers. Lives here so retrieve.py and
# the index are guaranteed to agree on it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

BM25_K1 = 1.5
BM25_B = 0.75
TOKEN = re.compile(r"[0-9a-z]+")


def bm25_tokenise(text):
    """Lowercase alphanumeric tokens. No stemming, no stopword removal — frozen."""
    return TOKEN.findall(text.lower())


def embed(texts, tok, model, desc):
    """CLS pooling then L2 normalise — bge's own recipe, not mean pooling."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH,
                  return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy())
        print(f"\r  {desc}: {min(i + BATCH, len(texts))}/{len(texts)}", end="")
    print()
    return np.vstack(out).astype(np.float32)


def main():
    # ---- one ordered chunk list; everything downstream indexes off this exact order
    rows = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    chunk_ids = [r["chunk_id"] for r in rows]
    texts = [r["text"] for r in rows]
    print(f"{len(rows)} chunks from {CHUNKS}")

    assert len(set(chunk_ids)) == len(chunk_ids), "duplicate chunk_id in chunks.jsonl"

    # Stage 1 cut chunks at 512 bge tokens with add_special_tokens=False, so a full chunk
    # plus [CLS]/[SEP] is 514 and the encoder drops 2 tokens from its tail. For a
    # non-final chunk those 2 sit in the next chunk's 64-token overlap. For a document's
    # LAST chunk there is no next chunk, so they would leave the index entirely.
    # Currently no final chunk is full-length; assert it rather than trust it.
    last = {}
    for r in rows:
        if r["chunk_index"] > last.get(r["doc_id"], (-1, None))[0]:
            last[r["doc_id"]] = (r["chunk_index"], r)
    full_finals = [r["chunk_id"] for _, r in last.values() if r["n_tokens"] == MAX_LENGTH]
    assert not full_finals, (
        f"final chunk(s) at exactly {MAX_LENGTH} tokens would lose 2 tokens from the "
        f"index with no following overlap to recover them: {full_finals}")
    print(f"  no document's final chunk is {MAX_LENGTH} tokens — 0 tokens lost to "
          f"encoder truncation")

    # ---- dense
    print(f"\nEmbedding with {EMBED_MODEL} @ {EMBED_REVISION[:8]}")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    model = AutoModel.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION).eval()
    vectors = embed(texts, tok, model, "chunks")

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"embeddings not L2 normalised: {norms.min()}..{norms.max()}"
    assert vectors.shape == (len(rows), 384), f"unexpected shape {vectors.shape}"

    index = faiss.IndexFlatIP(vectors.shape[1])       # exact search, no approximation
    index.add(vectors)

    # ---- sparse, from the SAME list in the SAME order
    corpus = [bm25_tokenise(t) for t in texts]
    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)

    # ---- the assertion this file exists for
    assert len(corpus) == vectors.shape[0] == index.ntotal == len(chunk_ids), (
        f"chunk universe mismatch: bm25={len(corpus)} dense={vectors.shape[0]} "
        f"faiss={index.ntotal} ids={len(chunk_ids)}")
    assert [r["chunk_id"] for r in rows] == chunk_ids, "id list drifted from the chunk rows"
    assert all(bm25_tokenise(texts[i]) == corpus[i] for i in range(0, len(corpus), 50)), \
        "bm25 corpus does not correspond to the same texts, position for position"

    # The prefix must never have reached the indexed side.
    assert not any(QUERY_PREFIX in t for t in texts), \
        "query instruction prefix found in indexed chunk text — it is queries-only"

    id_hash = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()
    vec_hash = hashlib.sha256(vectors.tobytes()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "embeddings.npy", vectors)
    faiss.write_index(index, str(OUT / "faiss.index"))
    (OUT / "bm25.pkl").write_bytes(pickle.dumps(bm25))
    (OUT / "chunk_ids.json").write_text(json.dumps(chunk_ids, indent=0), encoding="utf-8")

    manifest = {
        "n_chunks": len(rows),
        "embed_model": EMBED_MODEL,
        "embed_revision": EMBED_REVISION,
        "embed_dim": int(vectors.shape[1]),
        "pooling": "cls",
        "normalisation": "l2",
        "max_length": MAX_LENGTH,
        "query_prefix": QUERY_PREFIX,
        "query_prefix_applied_to": "queries only",
        "dense_index": "faiss.IndexFlatIP",
        "sparse_index": "rank_bm25.BM25Okapi",
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "bm25_tokenisation": "lowercase [0-9a-z]+, no stemming, no stopwords",
        "chunk_ids_sha256": id_hash,
        "embeddings_sha256": vec_hash,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")

    print(f"\ndense : {vectors.shape[0]} x {vectors.shape[1]}  faiss.ntotal={index.ntotal}")
    print(f"sparse: {len(corpus)} documents, "
          f"{sum(len(c) for c in corpus)} tokens, "
          f"{len(set(t for c in corpus for t in c))} vocabulary")
    print(f"chunk universe identical across both retrievers: {len(chunk_ids)} ids")
    print(f"chunk_ids sha256 : {id_hash[:16]}...")
    print(f"embeddings sha256: {vec_hash[:16]}...")
    print(f"\nwritten to {OUT}/")


if __name__ == "__main__":
    main()
