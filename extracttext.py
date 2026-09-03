import fitz
from pathlib import Path

src = Path("corpus/raw")
out = Path("corpus/extracted")
out.mkdir(parents=True, exist_ok=True)

for pdf in sorted(src.glob("*.pdf")):
    doc = fitz.open(pdf)
    pages = len(doc)
    parts = []
    for i, page in enumerate(doc, 1):
        parts.append(f"\n===== PAGE {i} =====\n")
        parts.append(page.get_text())
    text = "".join(parts)
    doc.close()

    chars = len(text.strip())
    if chars < 200 * pages:
        print(f"SKIP (scanned)  {pdf.name}  {pages}p {chars}ch")
        continue

    target = out / (pdf.stem + ".txt")
    target.write_text(text, encoding="utf-8")
    print(f"OK              {pdf.name}  {pages}p {chars}ch  -> {target.name}")