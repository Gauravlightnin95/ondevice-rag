# Notes — ABESEC_EOA_2026-27.pdf

**Source:** AICTE Extension of Approval letter, ABES Engineering College, AY 2026-27
**Pages:** 5 | **Text layer:** yes (Helvetica, clean) | **Drafted by:** claude-verified

---

## Table passages need re-copying

Q6, Q7 and Q9 have `passage: TODO_MULTIROW`. They span several table rows, so
the passage has to be assembled from the extracted text once ingestion exists.

Q4, Q5, Q8, Q10, Q11 have passages written in tidy form. The real extracted text
will look messier — PDF table columns collapse when converted to text, breaking
words across lines. Re-copy these from the extraction output before freezing.

If a gold passage does not appear verbatim in the corpus, Oracle-RAG silently
feeds the model text that does not exist. That is the failure this note prevents.

---

## Deliberately not asked

Available in the document but likely contaminated — a model may know them from
pretraining rather than from the text:

- Year of establishment (2000) — on the college website
- Affiliating university (AKTU) — widely known
- College address — widely known

Write them if you want, but expect the screen to delete them.

---

## Type coverage

| Type | Count | IDs |
|---|---|---|
| 1 — single-hop extractive | 6 | Q1, Q2, Q3, Q4, Q10, Q12, Q13 |
| 2 — multi-hop synthesis | 1 | Q9 (weak — stays within one document) |
| 3 — numerical / tabular | 5 | Q5, Q6, Q7, Q8, Q11 |
| 4 — conflicting sources | 0 | needs a second document |
| 5 — absent answer | 0 | not document-specific |
| 6 — long-form synthesis | 1 | Q14 |

Running totals so far: type 1 = 6, type 2 = 1, type 3 = 5, type 6 = 1.
Target is 30 per type.

---

## Next document to find

**The 2025-26 EoA letter for ABES.** Same document, previous year. The two state
different intake figures for the same courses, which gives genuine type-4
conflicting-source questions from real documents rather than manufactured ones.

Type 4 is the hardest type to source, so this is high priority. Ask whoever is
collecting to look for previous-year versions of every document they find.
