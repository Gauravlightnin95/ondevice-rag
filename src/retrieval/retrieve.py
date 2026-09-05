"""
Stage 4 — The frozen retrieval path: dense + BM25 + reciprocal rank fusion.

This is the ONE place retrieval happens. The Stage 6 inference grid imports `search`
from here rather than re-implementing it, so it cannot quietly diverge on the query
prefix, the candidate depth or the fusion constant.

Loading verifies the chunk id list against the manifest hash: the dense and sparse
indexes must be over the identical chunk universe or the fused ranking is meaningless
while still returning valid-looking chunk_ids.

    from retrieve import search
    hits = search("What is the minimum attendance?", k=10)
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

INDEX = Path("index")

# Fixed a priori, not tuned on Recall. RRF constant is Cormack et al.'s conventional 60.
CANDIDATE_DEPTH = 100
RRF_K = 60


@lru_cache(maxsize=1)
def _load():
    manifest = json.loads((INDEX / "manifest.json").read_text(encoding="utf-8"))
    chunk_ids = json.loads((INDEX / "chunk_ids.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX / "faiss.index"))
    bm25 = pickle.loads((INDEX / "bm25.pkl").read_bytes())

    import hashlib
    got = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()
    assert got == manifest["chunk_ids_sha256"], \
        "chunk_ids.json does not match the manifest hash — rebuild the index"
    assert index.ntotal == len(chunk_ids) == bm25.corpus_size, (
        f"chunk universe mismatch at load: faiss={index.ntotal} "
        f"ids={len(chunk_ids)} bm25={bm25.corpus_size}")

    tok = AutoTokenizer.from_pretrained(manifest["embed_model"],
                                        revision=manifest["embed_revision"])
    model = AutoModel.from_pretrained(manifest["embed_model"],
                                      revision=manifest["embed_revision"]).eval()
    return manifest, chunk_ids, index, bm25, tok, model


def embed_query(text):
    """Embed a query WITH the instruction prefix. Chunks are embedded without it."""
    manifest, _, _, _, tok, model = _load()
    enc = tok([manifest["query_prefix"] + text], padding=True, truncation=True,
              max_length=manifest["max_length"], return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy().astype(np.float32)


def _ranked(scores, depth):
    """Indices of the top `depth` scores, best first."""
    top = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
    return top[np.argsort(-scores[top])]


def search(query, k=10, depth=CANDIDATE_DEPTH):
    """Fused retrieval.

    Returns [(chunk_id, rrf_score, dense_rank, sparse_rank)] best first, length <= k.
    Ranks are 1-based; None means the chunk was outside that retriever's candidate list.
    """
    _, chunk_ids, index, bm25, _, _ = _load()

    _, dense_idx = index.search(embed_query(query), depth)
    dense_order = [int(i) for i in dense_idx[0] if i >= 0]

    from build_index import bm25_tokenise
    sparse_scores = np.asarray(bm25.get_scores(bm25_tokenise(query)), dtype=np.float64)
    sparse_order = [int(i) for i in _ranked(sparse_scores, depth)]

    dense_rank = {idx: r for r, idx in enumerate(dense_order, 1)}
    sparse_rank = {idx: r for r, idx in enumerate(sparse_order, 1)}

    fused = {}
    for ranks in (dense_rank, sparse_rank):
        for idx, rank in ranks.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

    # Tie-break on chunk_id so the ordering is total and reproducible.
    order = sorted(fused, key=lambda i: (-fused[i], chunk_ids[i]))[:k]
    return [(chunk_ids[i], fused[i], dense_rank.get(i), sparse_rank.get(i))
            for i in order]


def search_single(query, k=10, depth=CANDIDATE_DEPTH):
    """Dense-only and BM25-only rankings, for showing whether fusion earns its place."""
    _, chunk_ids, index, bm25, _, _ = _load()

    _, dense_idx = index.search(embed_query(query), depth)
    dense = [chunk_ids[int(i)] for i in dense_idx[0] if i >= 0][:k]

    from build_index import bm25_tokenise
    scores = np.asarray(bm25.get_scores(bm25_tokenise(query)), dtype=np.float64)
    sparse = [chunk_ids[int(i)] for i in _ranked(scores, depth)][:k]
    return dense, sparse
