# Notes — Batch 2

## Verdicts on the seven files

| File | Verdict | Reason |
|---|---|---|
| `CmlIGo2zF4aPg8zCk5WSpcVB23p6C7Ftw8kJB641` | ACCEPT | Galgotias grievance policy, clean text |
| `brochure-(details of programmes)` | ACCEPT | Amity CG brochure, number-rich |
| `boardof-studies-2024-25` | ACCEPT | Amity CG, limited but usable |
| `CSE` | HOLD | ABES CSE newsletter — promotional prose, low question yield |
| `b1ybIzNKrD64qGcjopMBWNg6P4ovFluDX1iBbvQR` | REJECT | Same document as CmlIGo2z, OCR-broken |
| `constitution-of-school-level-iqac-committee` | REJECT | Bad OCR (`Intemal`, `Proflndrani SinghRai`, `APoorva`) |
| `BalanceSheetFY201920` | REJECT | Severe OCR (`Audit repoft under section l2A(b) ofth. ,r"rrr"-`) |

**Delete `b1ybIzNK...`** — it is a duplicate of `CmlIGo2z...` in broken form. Keeping both
would put two versions of the same policy in the index, one of them corrupt.

---

## Rename these before loading

The `source` field is written into every question, so hash filenames must be fixed:

- `CmlIGo2zF4aPg8zCk5WSpcVB23p6C7Ftw8kJB641.pdf` → `galgotias-grievance-policy.pdf`
- `brochure-(details of programmes).pdf` → `amity-cg-brochure.pdf`

The question files already use the new names. Rename the PDFs and the extracted `.txt`
files to match, or the validator will fail.

---

## Type 4 is now open — this is the important result

`_conflicts.md` contains the first three type 4 questions, and they are real conflicts
between real documents, not manufactured ones:

1. **Appeal deadline to the VC** — grievance policy says "within a week", Part A says
   "within 14 calendar days".
2. **Who chairs the appeal body** — grievance policy says Pro Vice Chancellor, Part A says
   a Senior Professor/Senior Dean.
3. **Discrimination appeal window** — Part A itself gives 15 calendar days in section 22.0
   and 7 calendar days in section 22.5.

Q3 is a conflict *within a single document*, which is arguably even better: it cannot be
explained away as one document simply being older than the other.

**This is the pattern to exploit.** Galgotias publishes the same rules in at least three
places — the standalone grievance policy, Part A section 22, and Part B section 22. Every
overlapping topic is a candidate. Still to check: attendance rules (Part A section 11 vs
Part B sections 5 and 10), credit limits (Part A section 13 vs Part B section 6.8), awards
and medals (Part A section 14 vs Part B section 18), code of conduct, and additional
learning.

Send me Part B section 22 and the award sections and I will extract more.

Note the schema extension: type 4 entries carry `passage2`, `source2` and `page2`. Your
loader will need to handle these, and Oracle-RAG must supply both passages for these
questions.

---

## Running type totals

| Type | Now | Target | Remaining |
|---|---|---|---|
| 1 — single-hop extractive | 34 | 30 | done |
| 2 — multi-hop synthesis | 19 | 30 | 11 |
| 3 — numerical / tabular | 26 | 30 | 4 |
| 4 — conflicting sources | 3 | 30 | 27 |
| 5 — absent answer | 0 | 30 | 30 |
| 6 — long-form synthesis | 6 | 30 | 24 |

Total: 88 of 180.

Type 1 is over target — I will stop writing them. Type 3 is nearly done. **The work from
here is types 4, 5 and 6.**

---

## On CSE.txt

ABES CSE department newsletter (INSPERIA, Jan–July 2024). Clean text, but the content is
messages from the Director and HOD, event listings and testimonials. Very little that is
factual and checkable, and much of it is the kind of promotional prose that produces vague
questions.

I can draw perhaps 3-4 weak questions from it. Not worth the corpus slot unless you are
short. If you want it in, say so and I will write them.

---

## Corpus composition

You now have documents from ABES, Galgotias and Amity Chhattisgarh. That is fine — the
research needs uncontaminated institutional prose, not one institution. But record the
composition in the paper: number of documents per institution, and total pages. It is the
kind of detail a reviewer asks for.

---

## Next: type 5

Type 5 questions have no gold passage and belong in `evalset/_absent.md`. They cannot be
written from a single document — absence must be verified against the whole corpus.

Do this only once the corpus is frozen. Then search the full extracted text for each
candidate answer to confirm it genuinely is not there. Good candidates are topics adjacent
to what the corpus covers: specific fee amounts, named current office-holders, exact
calendar dates, hostel capacity figures.
