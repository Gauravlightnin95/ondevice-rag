# Notes — Batch 1: Galgotias PART-A and PART-B

**Documents:** PART-A-ACADEMIC-REGULATIONS.pdf (53p), PART-B-Examination-Rules-Regulations.pdf (60p)
**Institution:** Galgotias University
**Questions produced:** 22 + 23 = 45
**Drafted by:** claude-verified

---

## Running type totals (including the earlier EoA file)

| Type | Now | Target | Remaining |
|---|---|---|---|
| 1 — single-hop extractive | 27 | 30 | 3 |
| 2 — multi-hop synthesis | 10 | 30 | 20 |
| 3 — numerical / tabular | 16 | 30 | 14 |
| 4 — conflicting sources | 0 | 30 | 30 |
| 5 — absent answer | 0 | 30 | 30 |
| 6 — long-form synthesis | 3 | 30 | 27 |

Type 1 is nearly done. **Stop writing type 1 questions** — they are the easiest to produce
and you already have enough. Future batches should target types 2, 3 and 6.

---

## Type 4 is available in this batch and worth building

Part A and Part B overlap heavily. Both contain sections on attendance, credit limits,
course registration, adding/dropping, additional learning, awards and medals, code of
conduct, and the grievance appeal committee. Where the two documents state the same rule
in different words or with different numbers, that is a **genuine conflicting-source pair**
from two real documents — exactly what type 4 needs, and much stronger than a manufactured
conflict.

To find them, diff the overlapping sections. Candidates worth checking:

- Medal ordering: Part A lists silver as second topper and bronze as third. Part B's index
  lists 18.2 Bronze before 18.3 Silver. Check the body text of both.
- Attendance: Part A gives the 75%/60% table and the VC's 15% relaxation. Part B section 5
  and section 10 (Detention) restate attendance rules. Check whether the numbers match.
- Credit limits: Part A section 13 and Part B section 6.8 cover the same ground.
- Grievance procedure: Part A section 22 and Part B section 22.

I did not draft these because confirming a real conflict requires reading both passages
side by side. Send me the specific overlapping sections and I will write the type 4
questions properly.

---

## Type 5 needs a separate approach

Type 5 questions have no gold passage because the answer is genuinely absent from the
corpus. They are not drawn from any one document, so they go in `evalset/_absent.md`.

The right way to write them: pick topics the corpus *nearly* covers but does not. From
these two documents, plausible type 5 questions would be things like the fee amount for a
specific programme, the name of the current Vice Chancellor, or the exact dates of the
academic calendar — all adjacent to what the documents discuss without being stated.

Verify absence by searching the full extracted corpus before accepting each one. Do this
once the corpus is frozen, not before, or a later document may accidentally supply the
answer.

---

## Passage verification

Two entries are marked `TODO_MULTIPAGE` (Part A Q22, Part B Q23). Both are long-form
synthesis questions whose supporting text spans several pages. Assemble the passage from
the extracted text once ingestion exists.

All other passages were copied from the extracted `.txt` files, so they match what the
pipeline will see — with one caveat: I collapsed runs of whitespace and line breaks. Your
chunker will likely do the same, but confirm the normalisation matches before freezing. If
your chunker preserves raw line breaks, these passages will not match verbatim.

Before freezing, grep the evalset for `TODO_MULTIPAGE` and `TODO_MULTIROW` and make sure
none remain.

---

## Quality notes on this batch

The extracted text contains OCR-style run-together words in places (`Thestudentalsowillhave...`,
`shell be out of 100`). These are in the source PDF, not extraction errors. I avoided
drawing passages from the worst-affected regions. If your retriever performs oddly on
Part A pages 33-34, this is why.

Part B is version 1.3 and states an applicability date. If you later obtain version 1.2 or
an earlier edition, that is another strong type 4 source.

---

## Next documents to send

Highest value remaining, in order:

1. `Final_Mandatory_Disclosure.pdf` (59p) — table-heavy, good for type 3
2. `HOSTEL RULE BOOKLET.pdf` (22p) — specific rules, good for types 2 and 6
3. `GLBITM-IDP24-29.PDF` (48p) — institutional plan, good for type 6
4. The four `Format-Public-Self-Disclosure` variants — compare them for type 4 conflicts

Send 2-3 at a time.
