"""
Verify that type 5 questions really have no answer in the corpus.

Each entry lists search terms. If a term is FOUND, that question is suspect —
the corpus may contain the answer after all, which would make an honest refusal
the wrong answer and silently corrupt the type.

Run from C:\\ondevice-rag
"""

import re
from pathlib import Path

CORPUS = Path("corpus/extracted")

# question id -> terms that would indicate the answer IS present
CHECKS = {
    "Q1  grade mark ranges":      [r"90\s*-\s*100", r"9[05]\s*and above", r"marks?\s+range"],
    "Q2  B+ percentage":          [r"B\+.{0,40}\d{2}\s*%", r"7[05]\s*-\s*7\d"],
    "Q3  pass mark":              [r"pass(ing)?\s+marks?", r"minimum\s+\d{2}\s*%\s*to\s*pass"],
    "Q4  hostel fee amount":      [r"hostel\s+fee.{0,60}(Rs\.?|₹)\s*[\d,]{4,}", r"(Rs\.?|₹)\s*[\d,]{4,}.{0,40}hostel"],
    "Q5  tuition fee amount":     [r"tuition\s+fee.{0,60}(Rs\.?|₹)\s*[\d,]{4,}", r"annual\s+fee.{0,40}[\d,]{5,}"],
    "Q6  semester start date":    [r"semester\s+(begins|starts|commences)\s+on", r"(commencement|start)\s+of\s+(the\s+)?semester\s*:?\s*\d"],
    "Q7  convocation date":       [r"convocation.{0,40}\d{1,2}(st|nd|rd|th)?\s+\w+\s+20\d\d", r"convocation\s+(date|day)\s*:?\s*\d"],
    "Q8  total rooms":            [r"total\s+(number\s+of\s+)?rooms", r"\d{3}\s+rooms"],
    "Q9  hostel capacity":        [r"hostel.{0,30}capacity", r"capacity.{0,30}hostel", r"\d{3,4}\s*seater"],
    "Q10 library hours":          [r"library.{0,40}(open|timing|hours).{0,20}\d", r"\d{1,2}[:.]\d\d\s*(am|AM).{0,30}library"],
    "Q11 VC name":                [r"Vice\s+Chancellor\s*[:,-]\s*(Dr|Prof|Mr|Ms)", r"(Dr|Prof)\.?\s+[A-Z]\w+.{0,20}Vice\s+Chancellor"],
    "Q12 Registrar name":         [r"Registrar\s*[:,-]\s*(Dr|Prof|Mr|Ms)", r"(Dr|Prof|Mr)\.?\s+[A-Z]\w+.{0,20}Registrar"],
    "Q13 highest package":        [r"highest\s+(package|salary|ctc)", r"(package|CTC).{0,20}(LPA|lakh)"],
    "Q14 placement percent":      [r"\d{1,3}\s*%\s+(students\s+)?placed", r"placement\s+(percentage|record).{0,20}\d"],
    "Q15 hostel apply deadline":  [r"last\s+date.{0,40}hostel", r"hostel.{0,40}last\s+date"],
    "Q16 revaluation fee":        [r"revaluation.{0,40}(Rs\.?|₹)\s*\d", r"(Rs\.?|₹)\s*\d+.{0,30}revaluation"],
    "Q17 summer term fee":        [r"summer\s+(term|semester).{0,40}(Rs\.?|₹)\s*\d", r"(Rs\.?|₹)\s*\d+.{0,40}summer\s+(term|semester)"],
    "Q18 max degree duration":    [r"maximum\s+(duration|period).{0,40}\d+\s+(year|semester)", r"within\s+\d+\s+years?\s+of\s+admission"],
    "Q19 total enrolment":        [r"total\s+(number\s+of\s+)?students\s+enrolled", r"enrolment.{0,20}\d{3,}"],
    "Q20 SFR value":              [r"student\s*[- ]?\s*faculty\s+ratio\s*[:=-]?\s*\d", r"ratio\s+of\s+\d+\s*:\s*\d+"],
    "Q21 mess fee":               [r"mess\s+(fee|charges).{0,40}(Rs\.?|₹)\s*\d", r"(Rs\.?|₹)\s*[\d,]{3,}.{0,30}mess"],
    "Q22 late fee fine":          [r"late\s+(fee|payment).{0,40}(Rs\.?|₹)\s*\d", r"(fine|penalty).{0,30}late\s+payment"],
    "Q23 total credits degree":   [r"total\s+(of\s+)?\d{2,3}\s+credits", r"\d{3}\s+credits\s+(to|for)\s+(complete|graduate|award)"],
    "Q24 MTE duration":           [r"MTE.{0,40}\d\s*(hour|hrs)", r"duration.{0,30}(examination|exam).{0,20}\d\s*hour"],
    "Q25 MTE weightage":          [r"MTE.{0,30}\d{2}\s*(marks|%)", r"weightage.{0,30}MTE"],
    "Q26 faculty leave days":     [r"\d+\s+days?\s+(of\s+)?(casual|earned|annual)\s+leave", r"leave\s+entitlement"],
    "Q27 wifi password":          [r"password\s*[:=]\s*\S", r"wi-?fi.{0,30}password"],
    "Q28 books return count":     [r"return\s+all\s+(library\s+)?books", r"\d+\s+books.{0,30}return"],
    "Q29 graduation CGPA":        [r"minimum\s+CGPA\s+of\s+\d(\.\d)?\s+(to|for)\s+(graduate|award|complete)", r"CGPA\s+of\s+\d(\.\d)?\s+for\s+(the\s+)?degree"],
    "Q30 reservation percent":    [r"\d{1,2}\s*%.{0,20}(SC|ST|OBC|EWS)", r"(SC|ST|OBC)\s*[-–]\s*\d{1,2}\s*%"],
    "Q31 withdrawal notice":      [r"notice.{0,30}withdraw", r"withdraw.{0,40}\d+\s+days?\s+notice"],
    "Q32 faculty teaching hours": [r"\d{1,2}\s+hours?\s+per\s+week.{0,30}(teaching|faculty)", r"teaching\s+load.{0,20}\d"],
    "Q33 scholar stipend":        [r"stipend.{0,40}(Rs\.?|₹)\s*\d", r"(Rs\.?|₹)\s*[\d,]{4,}.{0,30}(stipend|fellowship)"],
    "Q34 script retention":       [r"answer\s+(script|sheet).{0,40}(retain|preserve|destroy)", r"retention.{0,30}(script|answer)"],
    "Q35 parent visiting hours":  [r"visit(ing|or).{0,30}(hours|timing)", r"parents?.{0,30}visit.{0,20}\d"],
    "Q36 GRC quorum":             [r"quorum.{0,40}[Gg]rievance", r"[Gg]rievance.{0,40}quorum"],
}

files = sorted(CORPUS.glob("*.txt"))
print(f"Searching {len(files)} files in {CORPUS}\n")

blobs = {}
for f in files:
    blobs[f.name] = f.read_text(encoding="utf-8", errors="ignore")

clean, suspect = [], []

for qid, patterns in CHECKS.items():
    hits = []
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for name, text in blobs.items():
            m = rx.search(text)
            if m:
                s = max(0, m.start() - 50)
                ctx = " ".join(text[s:m.end() + 50].split())
                hits.append((name, ctx))
                break
    if hits:
        suspect.append((qid, hits))
    else:
        clean.append(qid)

print("=== CLEAN (answer genuinely absent) ===")
for q in clean:
    print(f"  OK    {q}")

print(f"\n=== SUSPECT (check these manually) ===")
for qid, hits in suspect:
    print(f"\n  !!    {qid}")
    for name, ctx in hits[:2]:
        print(f"        {name}")
        print(f"        ...{ctx}...")

print(f"\n{len(clean)} clean, {len(suspect)} suspect, {len(CHECKS)} total")