"""
Stage 6 — Prompt construction for the three experimental conditions.

README section 1:
    closed-book  question only                    what parameters alone know
    rag          question + top-k retrieved       the deployable system
    oracle       question + known-correct passage pure reasoning, retrieval removed

The three differ ONLY in the context block. The system prompt, the user-message shape and
the /no_think suffix are identical across all conditions and all four models — varying any
of them by model size would confound the size axis.

Oracle-RAG context, README section 5:
    types 1,2,3,6   the gold passage
    type 4          BOTH passages; a correct answer requires noticing the corpus disagrees
    type 5          the top-1 retrieved chunk (see ORACLE_TYPE5 below)
"""

CLOSED_BOOK = "closed_book"
RAG = "rag"
ORACLE = "oracle"
CONDITIONS = (CLOSED_BOOK, RAG, ORACLE)

# oracle_context_source values, so analysis can partition without inferring from type
SRC_NONE = "none"
SRC_GOLD = "gold_passage"
SRC_GOLD_PAIR = "gold_passage_pair"
SRC_RETRIEVED = "retrieved_chunk"
SRC_RETRIEVED_TOPK = "retrieved_top_k"

INSTRUCTION = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say so."
)


def _context_block(passages):
    parts = []
    for i, text in enumerate(passages, 1):
        parts.append(f"[{i}] {text.strip()}")
    return "\n\n".join(parts)


def build_user_content(condition, question, *, passages=None):
    """The user message, before pipeline.build_prompt appends /no_think.

    Closed-book receives no context block at all — not an empty one — so there is no path
    by which context can leak into the condition that is supposed to isolate parameters.
    """
    if condition == CLOSED_BOOK:
        assert not passages, "closed-book must receive no context"
        return f"Question: {question}"

    assert passages, f"{condition} requires context"
    return (f"{INSTRUCTION}\n\n"
            f"Context:\n{_context_block(passages)}\n\n"
            f"Question: {question}")


def oracle_context(record, retrieve_fn):
    """Context for the Oracle-RAG condition. Returns (passages, source, chunk_ids).

    Type 5 has passage: null by definition, so there is no gold text to supply. README
    section 5 requires topically-related passages that do not contain the answer, and the
    rule here is the single top-ranked chunk from the frozen retriever:

      - it introduces nothing new above the LLM, the retriever is already frozen
      - it cannot contain the answer BY CONSTRUCTION: verify_absent.py established the
        answer is absent from the whole corpus, so no chunk can hold it
      - top-1 rather than top-k keeps Oracle-RAG's small-context character. Handing type 5
        the full k=5 set would make its oracle cell identical to its rag cell.

    NOTE the resulting asymmetry, recorded in docs/inference_notes.md: types 1-4 and 6 get
    a human-verified gold passage (median 242 chars); type 5 gets a retrieved chunk
    (~2447 chars, plausible but wrong). Those are categorically different objects, so
    type-5 accuracy under Oracle-RAG is NOT comparable to other types along the type axis.
    It is valid for its own purpose: refusal behaviour when plausible context is present.
    """
    if record["type"] == 5:
        # retrieve_fn(question, k) -> [(chunk_id, text), ...]
        hits = retrieve_fn(record["question"], 1)
        return [hits[0][1]], SRC_RETRIEVED, [hits[0][0]]

    if record["type"] == 4:
        return ([record["passage"], record["passage2"]], SRC_GOLD_PAIR, [])

    return ([record["passage"]], SRC_GOLD, [])
