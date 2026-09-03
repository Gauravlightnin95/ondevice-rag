# Notes — Batch 4

## The unnamed file is the Galgotias Student's Handbook 2026-27

`NurqOlFSCg6EZdah90R9vyYUlAy0ghsHgjx2wltc.pdf` → rename to **`galgotias-handbook-2026-27.pdf`**

This is the most valuable document you have sent, for one reason: **it restates the same
rules as Part A and Part B, but seven years later, and the numbers have changed.**

Part A and Part B are dated 2019-20. The Handbook is 2026-27. Same university, same topics,
different answers. That produces genuine contradictions rather than the cross-institution
differences we were relying on before.

Eight of them are in `_conflicts_append2.md` (Q8-Q15). The strongest:

| Rule | Part A / B (2019-20) | Handbook (2026-27) |
|---|---|---|
| MTE attendance | no minimum criterion | 50% |
| Summer term attendance | below 90% barred | at least 75% |
| Summer term limit | maximum 4 courses | 16 credits, 40 contact hrs/week |
| Summer eligibility | detained students only | failed **or** detained |
| Attendance relaxation | VC, fixed 15% | Examination Committee, no fixed % |
| Family grounds | calamity in immediate family | death, parents & siblings only |
| Debarred list | School notice board | university ERP |
| Medical notice | via Program Chair, consent in a week | email dean immediately, proof in a week |

The MTE one is the cleanest. Part A says explicitly that *no* minimum applies; the Handbook
sets 50%. A model cannot reconcile those — it has to notice both.

**Append `_conflicts_append2.md` to `_conflicts.md`.** Numbering continues, giving Q1-Q15.

---

## This is now your type 4 engine

The pattern is: **find the same rule stated in two documents of different vintage.** You
have three Galgotias documents covering overlapping ground (Part A, Part B, Handbook) plus
the standalone grievance policy. Sections not yet diffed:

- Credit limits — Part A section 13, Part B section 6.8, Handbook Chapter 2
- Awards and medals — Part A section 14, Part B section 18
- Graduation policy — Handbook Chapter 3, no Part A/B equivalent yet found
- Discipline and code of conduct — Part A section 20, Part B, Handbook Chapter 7
- Hostel rules — Handbook Chapter 11 vs the ABES hostel booklet
- Grievance redressal — Handbook Chapter 7 vs the standalone policy and Part A section 22

Send me the Handbook's Chapter 2 (Credit System), Chapter 3 (Graduation Policy) and
Chapter 7 (Discipline and Grievance Redressal) as extracted text and I can likely find
another eight to twelve conflicts. That would close type 4 entirely.

---

## GL Bajaj IDP — accepted but thin

`GLBITM-IDP24-29.pdf` is a strategic planning document. It is 48 pages of intentions and
committee structures rather than checkable rules, so the question yield is low — four
questions, mostly type 6. Strategy documents generally make weak eval material: the prose
is abstract, and questions drawn from it tend to have vague answers that an LLM judge will
grade inconsistently.

Not a rejection, but do not prioritise similar documents.

Q1 is marked `TODO_MULTIROW` — the Board of Governors composition is an eleven-item list
spanning a page break. Assemble the passage from the extracted text.

---

## Running type totals

| Type | Now | Target | Remaining |
|---|---|---|---|
| 1 | 40 | 30 | over |
| 2 | 38 | 30 | over |
| 3 | 40 | 30 | over |
| 4 | 15 | 30 | 15 |
| 5 | 0 | 30 | 30 |
| 6 | 17 | 30 | 13 |

**Total: 150 of 180.**

Types 1, 2 and 3 are all over target. Only 4, 5 and 6 remain.

---

## Type 5 — plan it now, write it after freezing

Type 5 is the last untouched type and needs a different method. The answer must be genuinely
absent from the entire corpus, which can only be verified once the corpus is frozen.

The method: pick a topic the corpus nearly covers, then grep the full extracted text for the
answer to confirm absence.

From what is now in the corpus, plausible candidates:

- The actual hostel fee amount. The ABES hostel booklet gives a URL for the fee structure but
  no figure, so the corpus discusses hostel fees without stating any.
- The academic calendar dates. Multiple documents reference an academic calendar; none in the
  corpus appears to contain the dates themselves.
- The current Vice Chancellor's name at Galgotias. Documents refer to the VC's powers
  throughout without naming the holder.
- The number of hostel rooms or total hostel capacity at ABES. Hostels are named and rules
  given, but no capacity figure appears.
- Marks required for a specific letter grade. Part B gives grade points per letter grade but
  not the mark ranges that map to them.

That last one is the best kind of type 5 question: it looks answerable from a table the model
can see, and the model must notice the table gives grade points rather than mark ranges. A
small model is likely to invent ranges.

Do not write these yet. Freeze first, then verify each by searching.
