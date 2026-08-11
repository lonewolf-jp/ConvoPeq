# 一時デバッグスクリプト: INTEGRATED_BUG_FIX.md 再編成（print デバッグ付き）
import re, sys

SRC = "doc/work88/big_bug/INTEGRATED_BUG_FIX.md"
print("reading...", flush=True)
text = open(SRC, encoding="utf-8").read()
lines = text.split("\n")
print(f"read {len(lines)} lines", flush=True)

# ヘッダ
header = []
i = 0
while i < len(lines) and not lines[i].startswith("## 1. 概要"):
    header.append(lines[i]); i += 1
print(f"header: {len(header)} lines (i={i})", flush=True)

# §1
sec1 = []
while i < len(lines) and not lines[i].startswith("## 2. 改修対象バグ一覧"):
    sec1.append(lines[i]); i += 1
print(f"sec1: {len(sec1)} lines (i={i})", flush=True)

# §2.1+2.2 / 2.3
sec2_1, sec2_3 = [], []
in_sec23 = False
while i < len(lines) and not lines[i].startswith("## 3. 詳細改修設計"):
    if lines[i].startswith("### 2.3"):
        in_sec23 = True
    (sec2_3 if in_sec23 else sec2_1).append(lines[i]); i += 1
print(f"sec2_1: {len(sec2_1)} / sec2_3: {len(sec2_3)} (i={i})", flush=True)

# §3
sec3_header = []
if i < len(lines) and lines[i].startswith("## 3. 詳細改修設計"):
    sec3_header.append(lines[i]); i += 1
sec3_design, sec3_appendix = [], []
appendix_ids = {"3-18", "3-21", "3-22", "3-23", "3-28", "3-36"}
while i < len(lines) and not lines[i].startswith("## 4. 実装順序"):
    m = re.match(r"^### (3-\d+)\.", lines[i])
    if m:
        sec_id = m.group(1)
        j = i
        while j < len(lines) and not re.match(r"^### (3-\d+)\.", lines[j]):
            j += 1
        block = lines[i:j]
        (sec3_appendix if sec_id in appendix_ids else sec3_design).append(f"  {sec_id}: {len(block)} lines")
        print(f"  section {sec_id}: {len(block)} lines (i={i}->j={j})", flush=True)
        i = j
        continue
    i += 1
print(f"sec3_design: {len(sec3_design)} sections / sec3_appendix: {len(sec3_appendix)} sections (i={i})", flush=True)

# §4-6
sec4_6 = []
while i < len(lines):
    sec4_6.append(lines[i]); i += 1
print(f"sec4_6: {len(sec4_6)} lines (i={i})", flush=True)

print("writing...", flush=True)
# 再構築（デバッグではファイル書き込みを抑制）
print("DONE - sections parsed OK", flush=True)
