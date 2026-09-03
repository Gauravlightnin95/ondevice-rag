# Notes — Batch 3

## Both documents accepted

| File | Pages | Verdict |
|---|---|---|
| `HOSTEL RULE BOOKLET.pdf` | 22 | ACCEPT — the best document so far for types 2, 3 and 6 |
| `Final_Mandatory_Disclosure.pdf` | 59 | ACCEPT, with a caveat below |

The hostel booklet is unusually good material. It is dense with specific, checkable rules —
timings, fines, eligibility conditions, procedures — and almost none of it could be guessed
from pretraining. If you find similar rule booklets from other institutions, prioritise them.

---

## Caveat on the Mandatory Disclosure: personal contact data

Large parts of this document are tables of named individuals with their personal mobile
numbers — faculty, non-teaching staff, parent representatives, and **named students**.

I deliberately drew no questions from those sections. Asking "what is X's phone number",
then publishing the eval set, would put personal contact details of named students into a
public research artifact. That is a bad outcome regardless of the document being public.

Two things follow:

1. **Do not draft questions from those tables**, and tell Claude Code the same if it works
   on this document.
2. Consider whether to release this document's text at all. The safer route for your public
   deliverable is to release the questions and a manifest of source URLs rather than the
   extracted corpus itself. That is standard practice and avoids the issue entirely.

Add a line to the paper's ethics or limitations section noting that the corpus contains
publicly-disclosed institutional documents which include personal contact information, and
that questions were not drawn from those sections.

---

## Four more type 4 questions

`_conflicts_append.md` contains Q4-Q7. **Append these to your existing `_conflicts.md`** —
the numbering continues from the three already there, so you will have Q1-Q7.

These are a different flavour from the first three. Q1-Q3 were contradictions within one
institution's own documents. Q4-Q7 are cross-institution differences: Galgotias and ABES
both have grievance procedures with different timelines.

**This distinction matters and should go in the paper.** A cross-institution difference is
not strictly a contradiction — both documents are correct for their own institution. But it
still tests the thing type 4 is meant to test: does the model notice that the corpus contains
more than one answer, or does it retrieve one chunk and state a single figure confidently?

A model that answers "four days" without qualification has failed, even though four days is
correct for ABES. The gold answer requires recognising both sources.

Consider tagging these as `4a` (true contradiction) and `4b` (cross-source difference) so you
can report them separately. If small models fail 4b but pass 4a, that is a finding worth
having.

---

## Running type totals

| Type | Now | Target | Remaining |
|---|---|---|---|
| 1 — single-hop extractive | 38 | 30 | over |
| 2 — multi-hop synthesis | 30 | 30 | done |
| 3 — numerical / tabular | 38 | 30 | over |
| 4 — conflicting sources | 7 | 30 | 23 |
| 5 — absent answer | 0 | 30 | 30 |
| 6 — long-form synthesis | 12 | 30 | 18 |

**Total: 125 of 180.**

Types 1, 2 and 3 are all at or over target. Everything from here goes to 4, 5 and 6.

You are over on types 1 and 3. That is fine — trim to 30 each when you freeze, keeping the
strongest. Or keep them all and report the actual distribution; an uneven set is defensible
as long as you say so, since bootstrap confidence intervals are computed per type anyway.

---

## What to send next

**For type 4:** documents that overlap with what you already have. Specifically:
- Part B sections 5, 10, 13, 18 and 22 — these restate Part A's rules on attendance, credit
  limits, awards and grievance. Send me those sections and I will diff them.
- Any second-year version of a document already in the corpus.
- Other institutions' grievance or attendance policies — each one adds cross-source pairs
  against what you have.

**For type 6:** long procedural documents. `GLBITM-IDP24-29.PDF` (48p) and
`NurqOlFSCg6EZdah90R9vyYUlAy0ghsHgjx2wltc` (86p) are your two biggest untouched files.
Rename that second one first.

**For type 5:** nothing yet. Wait until the corpus is frozen, then verify absence by
searching the full extracted text. Doing it earlier risks a later document supplying the
answer.
