# Notes — Batch 6 (final drafting batch)

## Type 4 is now closed

`_conflicts_append3.md` contains Q16-Q25. Append to `_conflicts.md` for **25 type 4 questions**.
Five short of 30, but see the note on distribution below — 25 is defensible and you have
better uses for your remaining time.

The ten new conflicts:

| Rule | Older document | Handbook 2026-27 |
|---|---|---|
| Semester credit load | 15-30 (Part A) | 20-26 |
| Medal attendance | 60% (Part A) | 75% |
| Silver medal scope | each School, cohort >30/50 (Part A) | each Program, no threshold |
| Best outgoing award | Achiever's Award, Rs 11000 (Part A) | Chancellor Trophy, no cash |
| Dropping courses | any registered course, 10 days (Part A) | additional courses only, first year banned |
| Additional learning | 12/8/4 course awards (Part A) | max 6 credits, 2 courses |
| Hostel allotment | merit + first come (ABES) | first come only |
| Hostel attendance | staggered by gender/year (ABES) | single 10:00 pm |
| Alcohol penalty | Rs 5000 fine (ABES) | no fine specified |
| First year credits | min 15, no special rule (Part A) | fixed 20, no dropping |

**Q16 and Q17 are the strongest.** Both are bare numerical contradictions with no room for
interpretation — 15-30 against 20-26, and 60% against 75%. A model must either notice both
or state one confidently and be wrong.

---

## Type 6 is now closed

`galgotias-handbook-append.md` contains Q12-Q19. **Append to your existing
`galgotias-handbook-2026-27.md`**, which already has Q1-Q11.

Five new type 6 (credit definitions, graduation, Category I offences, Category II offences,
hostel admission) plus three type 2. That takes type 6 to 28.

---

## Final type distribution

| Type | Count | Target |
|---|---|---|
| 1 — single-hop extractive | 45 | 30 |
| 2 — multi-hop synthesis | 54 | 30 |
| 3 — numerical / tabular | 47 | 30 |
| 4 — conflicting sources | 25 | 30 |
| 5 — absent answer | 0 | 30 |
| 6 — long-form synthesis | 28 | 30 |

**199 written, 0 type 5.**

On trimming: your project document specifies 30 per type, and the cleanest thing is to cut
types 1, 2 and 3 back to 30 each, keeping the strongest. That gives 30/30/30/25/30/28 = 173.

But an uneven set is defensible if you report it. Bootstrap confidence intervals are computed
per type anyway, so unequal cell sizes cost you nothing statistically — they only mean
slightly wider intervals on the smaller types. State the actual distribution in the paper and
move on. Do not spend a week manufacturing five more type 4 questions.

If you do want to trim, cut type 1 hardest. It is the easiest type and the least informative,
since single-hop extraction is exactly where you expect small models to succeed.

---

## Naming — resolve this before loading

I have used descriptive filenames throughout (`galgotias-handbook-2026-27.pdf`,
`galgotias-academic-monitoring-v3.pdf`, `galgotias-it-policy.pdf`,
`galgotias-grievance-policy.pdf`, `amity-cg-brochure.pdf`).

You said you would keep the original hash names. **Either is fine, but the `source` field
must match the filename on disk exactly.** Pick one and do a find-and-replace across
`evalset/*.md`. Check `_conflicts.md` carefully since those entries have both `source` and
`source2`.

Also replace `annotator: XX` with your initials across all files.

---

## What happens now

Drafting is done. Nothing more to send until the corpus is frozen.

**Freeze sequence:**

1. Delete `O8f3RRPK...` (byte-identical duplicate of Part B, md5 verified) and
   `nss_team_merged` (garbled Devanagari)
2. Move every SCAN+OCR and NO TEXT file to `corpus/rejected/`
3. Re-run the checker for the final page count
4. Fill in `corpus/manifest.csv`: filename, title, institution, source URL, date downloaded,
   page count
5. Tag the corpus frozen and stop adding to it

**Then send me the frozen file list** — filenames and page counts only, no text. I will write
30 type 5 questions plus a grep script that verifies each answer genuinely appears nowhere in
the corpus.

Type 5 must come last. A document added after those questions are written could silently
supply an answer, turning a correct refusal into a wrong one and corrupting the type without
any visible error.

---

## After type 5

The remaining work on the eval set is contamination screening. Every question runs
closed-book through a model outside your grid; anything answered correctly without documents
is deleted. Budget for losing 10-20% of the set, mostly from types 1 and 3 where the facts
are most generic.

Then freeze as EvalSet v1.0 and move to the pipeline build.
