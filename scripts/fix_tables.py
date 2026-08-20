import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "evidence/phase-d101-8-step5-hmax-gcontract-derivation.md"
with open(path, "r") as f:
    content = f.read()

lines = content.split("\n")
result = []
prev_is_table_row = False

for i, line in enumerate(lines):
    stripped = line.strip()
    is_sep = bool(re.match(r"^\|[\s\-:|]+\|$", stripped))
    if is_sep and prev_is_table_row:
        # Count columns from the separator itself
        raw_cols = [c for c in stripped.split("|") if c.strip() != ""]
        col_count = len(raw_cols)
        result.append("|" + "|".join(["---"] * col_count) + "|")
    else:
        result.append(line)
    prev_is_table_row = bool(re.match(r"^\|", stripped)) and not is_sep

with open(path, "w") as f:
    f.write("\n".join(result))
print("Tables fixed")
