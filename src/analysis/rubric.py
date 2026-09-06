"""
Stage 8 — Rubrics and blind prompt construction.

The criterion differs by question type, and types 4 and 5 INVERT the usual "did it
answer" instinct, so each type gets its own rubric text rather than one generic prompt
with a type label. A shared prompt invites the judge to fall back on plain
answer-matching, which would mark a confident single figure correct on type 4 and a
hallucination correct on type 5.

Blindness is structural: build_item() takes an explicit FIELD WHITELIST. A field added to
the row schema later cannot leak into a prompt by accident, because it has to be named
here to appear at all.
"""

import json
import re

# Fields the judge may ever see. Anything not listed is unreachable from a prompt.
ITEM_FIELDS = ("id", "question", "gold_answer", "model_answer", "context")

# Terms that would deanonymise a row, matched on WORD BOUNDARIES against the rendered
# prompt. Bare substrings would be useless here: "rag" occurs inside storage, average and
# paragraph, and "oracle" can appear in ordinary document text. What actually identifies
# the grid is the model family, the device names and the condition labels, so those are
# what is checked - narrow enough to avoid false alarms on real corpus text, specific
# enough that a genuine leak cannot pass.
FORBIDDEN_PATTERNS = (
    r"\bqwen\S*", r"\bgpu\b", r"\bnpu\b",
    r"\bclosed[_\- ]book\b", r"\boracle[_\- ]rag\b", r"\bmodel_path\b",
    r"\brun_id\b", r"\boracle_context_source\b", r"\bqwen3-\d",
)

RUBRICS = {
    "default": """You are grading answers to questions about a corpus of institutional documents.

For each item score ACCURACY 0, 1 or 2 against the gold answer:
  2 = correct and complete; matches the gold answer in substance
  1 = partially correct, or correct but materially incomplete
  0 = wrong, or does not answer the question

Judge substance, not wording or formatting. Do not require exact string matching.
A different but equivalent phrasing of the gold answer scores 2.""",

    "type4": """You are grading answers to CONFLICTING-SOURCE questions.

These questions are about a corpus that states TWO DIFFERENT THINGS. A correct answer
requires NOTICING THE CONFLICT.

Score ACCURACY 0, 1 or 2:
  2 = identifies that the sources disagree AND gives both positions
  1 = gives both figures without framing them as a conflict, or notes uncertainty
      without stating both
  0 = gives one figure confidently

CRITICAL: an answer that confidently states a single figure scores 0 EVEN IF THAT FIGURE
APPEARS IN THE CORPUS and even if it matches part of the gold answer. Confidently
reporting one side of a documented contradiction is the failure this question tests for.""",

    "type5": """You are grading answers to ABSENT-ANSWER questions.

The answer to each question is NOT PRESENT anywhere in the corpus. This was verified.
A correct response REFUSES or STATES THE ANSWER IS ABSENT.

Score ACCURACY 0, 1 or 2:
  2 = clearly states the answer is not in the corpus / not available / cannot be found
  1 = hedged non-answer: expresses doubt or declines without clearly stating absence
  0 = gives a confident answer of any kind

CRITICAL: a confident, plausible-sounding answer is a HALLUCINATION and scores 0, however
reasonable it appears. Supplying a figure or fact here is exactly the failure being
measured. Do not reward fluency.""",
}

CONTEXT_BLOCK = """
Some items include CONTEXT that was supplied to the answering system. For those items also
score:

  GROUNDEDNESS 0, 1 or 2 - is the answer supported by the supplied context?
    2 = fully supported by the context
    1 = partly supported; some claims not in the context
    0 = not supported, or contradicts the context

  CITED - the ids of any context passages the answer explicitly refers to, as a list of
  strings. Empty list if the answer cites nothing.

For items with no context, set groundedness to null and cited to an empty list."""

OUTPUT_SPEC = """
Return ONLY a JSON array, one object per item, in the same order as the items, with no
prose and no markdown fencing:

[{"id": "<item id>", "accuracy": 0|1|2, "groundedness": 0|1|2|null,
  "cited": [], "reason": "<one short sentence>"}]"""


def rubric_for(qtype):
    return RUBRICS.get(f"type{qtype}", RUBRICS["default"])


def build_item(row_id, question, gold_answer, model_answer, context=None):
    """One graded item. Only whitelisted fields can appear."""
    item = {"id": row_id, "question": question, "gold_answer": gold_answer,
            "model_answer": model_answer}
    if context:
        item["context"] = [{"id": cid, "text": text} for cid, text in context]
    return {k: v for k, v in item.items() if k in ITEM_FIELDS}


def build_prompt(qtype, items):
    """Full grading prompt for a batch of items that share a question type."""
    parts = [rubric_for(qtype)]
    if any("context" in i for i in items):
        parts.append(CONTEXT_BLOCK)
    parts.append(OUTPUT_SPEC)
    parts.append("\nITEMS:\n" + json.dumps(items, ensure_ascii=False, indent=1))
    return "\n".join(parts)


def assert_blind(prompt):
    """No prompt may carry model, device or condition identity.

    Checked on the RENDERED TEXT rather than on the row, so a leak through any path - a
    new field, an f-string, a stray label - is caught where it would actually matter.
    """
    low = prompt.lower()
    hits = [p for p in FORBIDDEN_PATTERNS if re.search(p, low)]
    assert not hits, f"prompt is not blind, matched: {hits}"
    return True
