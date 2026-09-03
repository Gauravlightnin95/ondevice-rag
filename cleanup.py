import fitz, re, shutil
from pathlib import Path

raw = Path("corpus/raw")
ext = Path("corpus/extracted")
rej = Path("corpus/rejected")

DRY_RUN = False   # set False to actually move

# Files to keep regardless of automatic verdict.
# The brochure has full-page background images that trip the scan detector,
# but its extracted text is clean and 15 questions were drawn from it.
KEEP = {
    "brochure-(details of programmes).pdf",
}

# Known duplicates and manual rejects
BY_NAME = {
    "O8f3RRPKGAbRzkxJMlUgWGbioTUGqJQU8cSCHnfV.pdf": "duplicate of PART-B",
    "nss_team_merged.pdf": "garbled Devanagari OCR",
    "b1ybIzNKrD64qGcjopMBWNg6P4ovFluDX1iBbvQR.pdf": "duplicate of CmlIGo2z, broken OCR",
    "PART-B-Examination-Rules-Regulations (1).pdf": "duplicate of PART-B",
}


def verdict(path):
    if path.name in KEEP:
        return None
    if path.name in BY_NAME:
        return BY_NAME[path.name]

    doc = fitz.open(path)
    pages = len(doc)
    text = ""
    scan_pages = 0
    for page in doc:
        text += page.get_text()
        parea = abs(page.rect)
        for img in page.get_images(full=True):
            try:
                r = page.get_image_rects(img[0])
                if r and abs(r[0]) > parea * 0.5:
                    scan_pages += 1
                    break
            except Exception:
                pass
    doc.close()

    chars = len(text.strip())
    if chars < 200 * pages:
        return "NO TEXT"
    if scan_pages > pages * 0.5:
        return "SCAN+OCR"

    bad = len(re.findall(r"\b(lf|ln|lt|lssue|ls)\b", text))
    bad += len(re.findall(r"[a-z][\]\[\\|]{1,2}[a-z]", text))
    if bad > pages * 2:
        return "BAD OCR"

    return None


rej.mkdir(parents=True, exist_ok=True)
kept = []
removed = []

for pdf in sorted(raw.glob("*.pdf")):
    try:
        why = verdict(pdf)
    except Exception as e:
        why = f"ERROR {e}"

    if why:
        removed.append((pdf.name, why))
        if not DRY_RUN:
            shutil.move(str(pdf), str(rej / pdf.name))
            txt = ext / (pdf.stem + ".txt")
            if txt.exists():
                txt.unlink()
    else:
        d = fitz.open(pdf)
        kept.append((pdf.name, len(d)))
        d.close()

print("DRY RUN - nothing moved\n" if DRY_RUN else "MOVED\n")

print("--- REMOVING ---")
for name, why in removed:
    print(f"  {name[:45]:<45} {why}")

print("\n--- KEEPING ---")
for name, pages in kept:
    print(f"  {name[:55]:<55} {pages:>3}p")

total = sum(p for _, p in kept)
print(f"\nRemoving: {len(removed)} files")
print(f"Keeping:  {len(kept)} files, {total} pages")