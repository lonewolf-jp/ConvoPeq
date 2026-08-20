import re, sys

path = sys.argv[1]
with open(path, "r") as f:
    content = f.read()

lines = content.split("\n")
result = []

# Track previous line type
prev_line_type = "none"  # none, heading, text, table, list, code, blank

for i, line in enumerate(lines):
    stripped = line.strip()

    # Classify line type
    if stripped == "":
        line_type = "blank"
    elif re.match(r"^#{1,6}\s", stripped):
        line_type = "heading"
    elif re.match(r"^\|.*\|$", stripped) and not re.match(r"^\|[\s\-:]+\|$", stripped):
        line_type = "table"
    elif re.match(r"^\|[\s\-:]+\|$", stripped) and result and re.match(r"^\|.*\|$", result[-1].strip()):
        line_type = "table_sep"
    elif re.match(r"^\|[\s\-:]+\|$", stripped):
        line_type = "table_sep"
    elif re.match(r"^(-|\*|\d+\.)\s", stripped):
        line_type = "list"
    elif stripped.startswith("```"):
        line_type = "code"
    else:
        line_type = "text"

    # Add blank line before headings if previous was not blank
    if line_type == "heading" and prev_line_type not in ("blank", "none"):
        result.append("")

    # Add blank line before lists if previous was not blank
    if line_type == "list" and prev_line_type not in ("blank", "none", "list"):
        result.append("")

    # Add blank line before tables if previous was not blank
    if line_type == "table" and prev_line_type not in ("blank", "none", "table", "table_sep"):
        result.append("")

    # Add blank line after tables (when next non-blank is not table)
    # This will be handled in next iteration

    result.append(line)
    prev_line_type = line_type

# Join and handle trailing newline
content_out = "\n".join(result)

# Ensure single trailing newline
content_out = content_out.rstrip("\n") + "\n"

with open(path, "w") as f:
    f.write(content_out)

# Now fix specific issues:
with open(path, "r") as f:
    content = f.read()

# Fix MD026: Trailing punctuation in headings - remove colons from headings
content = re.sub(r'^(#{1,6}\s+.*):$', r'\1', content, flags=re.MULTILINE)

# Fix the table at line 323 (5 columns but 4-col separator) - add 5th column separator
content = content.replace(
    "|#|\n|---|---|---|---|\n",
    "|#|File|Line|Context|Tick Call|\n|---|---|---|---|---|\n"
)

with open(path, "w") as f:
    f.write(content)

print("Lint fixes applied")
