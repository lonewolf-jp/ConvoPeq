import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "evidence/phase-d101-8-step5-hmax-gcontract-derivation.md"
with open(path, "r") as f:
    content = f.read()

lines = content.split("\n")
result = []
prev_was_table_row = False

for i, line in enumerate(lines):
    stripped = line.strip()

    # Detect table separator row: | --- | --- | ... or |---|---|
    is_separator = bool(re.match(r"^\|[\s\-:]+\|$", stripped))
    is_table_row = bool(re.match(r"^\|.*\|$", stripped)) and not is_separator
    is_other = not is_table_row and not is_separator

    if is_separator and prev_was_table_row:
        # Count columns from previous line
        prev_parts = result[-1].split("|")
        if len(prev_parts) > 2:
            col_count = len(prev_parts) - 2  # subtract leading/trailing empty from split
        else:
            col_count = 2
        # Use compact separator
        result.append("|" + "|".join(["---"] * col_count) + "|")
    elif is_table_row and prev_was_table_row:
        # Compact the data row: strip leading/trailing spaces in each cell
        parts = line.split("|")
        if line.strip().startswith("|") and line.strip().endswith("|"):
            parts = parts[1:-1]  # remove empty leading/trailing
            compact_parts = [p.strip() for p in parts]
            result.append("|" + "|".join(compact_parts) + "|")
        else:
            result.append(line)
    elif is_table_row:
        # First row of a table (header) - compact it too
        parts = line.split("|")
        if line.strip().startswith("|") and line.strip().endswith("|"):
            parts = parts[1:-1]
            compact_parts = [p.strip() for p in parts]
            result.append("|" + "|".join(compact_parts) + "|")
        else:
            result.append(line)
    else:
        result.append(line)

    prev_was_table_row = is_table_row or is_separator

with open(path, "w") as f:
    f.write("\n".join(result))
print("Tables compacted")
