import fitz, re
from pathlib import Path

def check(path):
    doc = fitz.open(path)
    pages = len(doc)
    text = ""
    scanned_pages = 0

    for page in doc:
        text += page.get_text()
        parea = abs(page.rect)
        for img in page.get_images(full=True):
            try:
                r = page.get_image_rects(img[0])
                if r and abs(r[0]) > parea * 0.5:
                    scanned_pages += 1
                    break
            except Exception:
                pass
    doc.close()

    chars = len(text.strip())
    if chars < 200 * pages:
        return pages, chars, "NO TEXT"
    if scanned_pages > pages * 0.5:
        return pages, chars, "SCAN+OCR"

    # OCR artefacts: l/I confusion, stray brackets inside words
    bad = len(re.findall(r"\b(lf|ln|lt|lssue|ls)\b", text))
    bad += len(re.findall(r"[a-z][\]\[\\|]{1,2}[a-z]", text))
    if bad > pages * 2:
        return pages, chars, "BAD OCR"

    return pages, chars, "OK"

total = 0
for pdf in sorted(Path("corpus/raw").glob("*.pdf")):
    p, c, verdict = check(pdf)
    if verdict == "OK":
        total += p
    print(f"{pdf.name[:42]:<42} {p:>3}p {c:>7}ch  {verdict}")

print(f"\nUsable pages: {total}  (target 400-600)")