"""
Apply the filled passages from filled_passages_batchA/B/C.md into evalset/*.md

Each batch file has headings like:
    ## PART-A-ACADEMIC-REGULATIONS.md - line 215 (Q22, attendance summary)
followed by a fenced block containing the replacement line.

This script reads those, checks the target line still contains a TODO marker,
and replaces it.

Put the three batch files in C:\\ondevice-rag (next to this script) and run from there.
"""

import re
from pathlib import Path

EVALSET = Path("evalset")
BATCHES = ["filled_passages_batchA.md",
           "filled_passages_batchB.md",
           "filled_passages_batchC.md"]

DRY_RUN = False   # set False to actually write

# "## <file>.md <dash> line <N>"  — tolerate em dash, en dash or hyphen
HEADING = re.compile(r"^##\s+(\S+\.md)\s*[—–-]\s*line\s+(\d+)", re.IGNORECASE)


def parse_batch(path):
    """Yield (target_file, line_number, replacement_line)."""
    if not path.exists():
        print(f"  (missing: {path.name})")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = HEADING.match(lines[i])
        if not m:
            i += 1
            continue

        target, lineno = m.group(1), int(m.group(2))

        # find the next fenced block and take the passage line from it
        j = i + 1
        replacement = None
        while j < len(lines):
            if lines[j].startswith("## "):          # ran into next heading
                break
            if lines[j].strip().startswith("```"):
                k = j + 1
                buf = []
                while k < len(lines) and not lines[k].strip().startswith("```"):
                    buf.append(lines[k])
                    k += 1
                for b in buf:
                    if b.strip().startswith("- **passage:**"):
                        replacement = b.rstrip("\n")
                        break
                j = k
                break
            j += 1

        if replacement:
            yield target, lineno, replacement
        else:
            print(f"  !! no passage block found for {target} line {lineno}")
        i = j + 1


edits = []
print("Reading batch files...")
for name in BATCHES:
    print(f"  {name}")
    for item in parse_batch(Path(name)):
        edits.append(item)

print(f"\nFound {len(edits)} replacements\n")

# group by file so each file is read and written once
by_file = {}
for target, lineno, repl in edits:
    by_file.setdefault(target, []).append((lineno, repl))

applied, skipped = 0, 0

for target, items in sorted(by_file.items()):
    path = EVALSET / target
    if not path.exists():
        print(f"MISSING FILE  {target}  ({len(items)} edits skipped)")
        skipped += len(items)
        continue

    lines = path.read_text(encoding="utf-8").splitlines()

    for lineno, repl in sorted(items):
        idx = lineno - 1
        if idx < 0 or idx >= len(lines):
            print(f"  !! {target} line {lineno} out of range")
            skipped += 1
            continue

        current = lines[idx]
        if "TODO_MULTI" not in current:
            print(f"  !! {target} line {lineno} does not contain a TODO marker")
            print(f"     found: {current.strip()[:70]}")
            skipped += 1
            continue

        lines[idx] = repl
        applied += 1
        print(f"  OK {target} line {lineno}  ({len(repl)} chars)")

    if not DRY_RUN:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\n{'DRY RUN - nothing written' if DRY_RUN else 'WRITTEN'}")
print(f"{applied} applied, {skipped} skipped, {len(edits)} total")