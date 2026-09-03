# Notes — Batch 5

## Verdicts

| File | Verdict | Reason |
|---|---|---|
| `Iq4uYzH0okL24sa9FWIuY5rV90odO9Ni2r828Jx1` | **ACCEPT** — best of the batch | Galgotias Academic Monitoring System v3 |
| `YdoUIjxvgRKYHcWx0AmXssMbQQlj6t6KKdE1Bj5c` | ACCEPT | Galgotias IT Policy, 37p |
| `Students-GrievanceRedressal-Cell` | PARTIAL ACCEPT | see caution below |
| `QTMRVE2p2ESUcuqLIumE4wZjNrb0UudSoTWZPDZx` | HOLD | Galgotias Vision 2030 — strategy doc, thin like the IDP |
| `RnD-Office-Order-asperUGC` | HOLD | GL Bajaj R&D Cell — committee list, minimal rules |
| `UGCPublicSelf-Disclosure` | HOLD | Malla Reddy Vishwavidyapeeth — usable, not yet drafted |
| `O8f3RRPKGAbRzkxJMlUgWGbioTUGqJQU8cSCHnfV` | **REJECT** | byte-identical duplicate of PART-B (md5 verified) |
| `nss_team_merged` | **REJECT** | garbled Devanagari OCR |

**Delete `O8f3RRPK...`.** I checked the md5 hash against `PART-B-Examination-Rules-Regulations.txt`
and they match exactly — same file, different name. Two identical copies in the index would
give the retriever duplicate chunks competing for the same slot and distort Recall@k.

---

## Rename before loading

- `Iq4uYzH0okL24sa9FWIuY5rV90odO9Ni2r828Jx1.pdf` → `galgotias-academic-monitoring-v3.pdf`
- `YdoUIjxvgRKYHcWx0AmXssMbQQlj6t6KKdE1Bj5c.pdf` → `galgotias-it-policy.pdf`

---

## Caution on Students-GrievanceRedressal-Cell

Two problems, so I drew only five questions from it.

**Contamination.** Most of the document is the UGC (Redressal of Grievances of Students)
Regulations 2023 quoted verbatim — a nationally published gazette notification, reproduced
across dozens of institutional websites and legal databases. Questions drawn from those
clauses would likely be answerable closed-book and deleted by your contamination screen. I
took questions only from the GL Bajaj-specific parts: the office order reference, the
committee composition, the contact address.

**Personal data.** The committee table includes a named student with their roll number
(`Roll No. 2201921520057`). Same concern as the ABES Mandatory Disclosure. I wrote no
question touching that row, and Q1 and Q3 are phrased around categories and departments
rather than names. Keep it that way.

The document is still useful because it is a **fourth grievance procedure** in the corpus,
alongside Galgotias's standalone policy, Part A section 22, and the ABES Mandatory
Disclosure. That gives more cross-source type 4 material once you send overlapping sections.

---

## The Academic Monitoring System is the standout

`galgotias-academic-monitoring-v3.pdf` is excellent eval material for the same reason the
hostel booklet was: specific, checkable, consequence-bearing rules. Salary deductions per
default, three-day substitution limits, 24-hour response windows, 6:30 PM reporting
deadlines. Fourteen questions from ten pages, which is the best density so far.

It is also **version 3, revised March 2025**, which means versions 1 and 2 exist. If you can
obtain either, that is another same-institution different-vintage pair — the most reliable
type 4 source you have found.

---

## Running type totals

| Type | Now | Target | Remaining |
|---|---|---|---|
| 1 | 45 | 30 | over |
| 2 | 51 | 30 | over |
| 3 | 47 | 30 | over |
| 4 | 15 | 30 | **15** |
| 5 | 0 | 30 | **30** |
| 6 | 23 | 30 | **7** |

**Total: 181.** You have passed 180 in raw count, but the distribution is wrong — four types
are over and two are short.

If you trim types 1, 2 and 3 to 30 each, the total is 135, and you still need 15 type 4 and
30 type 5.

---

## What is actually left

**Type 4 — 15 more.** I still have not seen the handbook chapters I asked for. Send
`galgotias-handbook-2026-27.txt` again and I will focus on Chapters 2, 3, 7 and 11, which
overlap with Part A sections 13, 20 and 22, Part B sections 6.8 and 22, and the ABES hostel
booklet. That should close the type.

**Type 5 — 30, and this is now the critical path.** Nothing can be written until the corpus
is frozen. The good news is that freezing is close: you have 636 usable pages, three
rejections identified in this batch, and no remaining must-have documents.

Proposed sequence:
1. Delete the duplicate and the rejects, rename the hash files
2. Re-run the checker to confirm the final page count
3. Freeze the corpus and tag it
4. Send me the frozen file list, and I will write 30 type 5 candidates plus a grep script to
   verify each answer genuinely is absent

**Type 6 — 7 more.** The three HOLD documents above can cover this if needed. Say the word
and I will draft from `UGCPublicSelf-Disclosure` and `QTMRVE2p` (Vision 2030).
