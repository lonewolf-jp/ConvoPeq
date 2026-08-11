# 一時再編成スクリプト: INTEGRATED_BUG_FIX.md を「設計」(有効バグ) と「Appendix」(それ以外) に再構成
# セクション番号は保持し、配置のみ変更する
import re

SRC = "doc/work88/big_bug/INTEGRATED_BUG_FIX.md"
text = open(SRC, encoding="utf-8").read()
lines = text.split("\n")

# ---- ヘッダ（1-10行: タイトル〜改修方針〜---） ----
# ファイル先頭から "## 1. 概要" の直前まで
header = []
i = 0
while i < len(lines) and not lines[i].startswith("## 1. 概要"):
    header.append(lines[i])
    i += 1

# ---- §1 概要（## 1. 概要 〜 ## 2. 改修対象バグ一覧 の直前） ----
sec1 = []
while i < len(lines) and not lines[i].startswith("## 2. 改修対象バグ一覧"):
    sec1.append(lines[i])
    i += 1

# ---- §2（### 2.1 即時改修 〜 ## 3. 詳細改修設計 の直前） ----
sec2_start = i
sec2_1 = []   # 2.1 + 2.2（有効バグ一覧）
sec2_3 = []   # 2.3 現状維持一覧（Appendix へ）
in_sec23 = False
while i < len(lines) and not lines[i].startswith("## 3. 詳細改修設計"):
    line = lines[i]
    if line.startswith("### 2.3"):
        in_sec23 = True
    if in_sec23:
        sec2_3.append(line)
    else:
        sec2_1.append(line)
    i += 1

# ---- §3 詳細改修設計（## 3. 〜 ## 4. 実装順序 の直前） ----
sec3_header = []   # "## 3. 詳細改修設計" 見出し行
sec3_design = []   # 有効バグセクション（設計部）
sec3_appendix = [] # 現状維持・監査セクション（Appendix へ）
if i < len(lines) and lines[i].startswith("## 3. 詳細改修設計"):
    sec3_header.append(lines[i])
    i += 1

# 現状維持・監査記録セクション（Appendix へ移動するもの）
appendix_section_ids = {"3-18", "3-21", "3-22", "3-23", "3-28", "3-36"}

while i < len(lines) and not lines[i].startswith("## 4. 実装順序"):
    line = lines[i]
    m = re.match(r"^### (3-\d+)\.", line)
    if m:
        # このセクションの開始
        sec_id = m.group(1)
        # セクションの終了位置を探す（次の ### 3-XX まで）— j = i+1 から開始（i 自身は見出しなのでスキップ）
        j = i + 1
        while j < len(lines) and not re.match(r"^### (3-\d+)\.", lines[j]):
            j += 1
        block = lines[i:j]
        if sec_id in appendix_section_ids:
            sec3_appendix.extend(block)
        else:
            sec3_design.extend(block)
        i = j
        continue
    i += 1

# ---- §4 実装順序〜§6（## 4. 〜 末尾） ----
sec4_6 = []
while i < len(lines):
    sec4_6.append(lines[i])
    i += 1

# ---- 出力構成 ----
out = []
out.extend(header)
out.append("")
out.append("---")
out.append("")
out.append("# 設計 — 有効バグの改修設計")
out.append("")
out.extend(sec1)          # §1 概要
out.append("")
out.extend(sec2_1)        # §2.1 + 2.2 有効バグ一覧
out.append("")
out.extend(sec3_header)   # ## 3. 詳細改修設計
out.extend(sec3_design)   # 有効バグの詳細設計
out.append("")
out.extend(sec4_6)        # §4 実装順序 / §5 検証計画 / §6 実施上の注意

out.append("")
out.append("---")
out.append("")
out.append("# Appendix — 現状維持・監査記録")
out.append("")
out.append("以下のバグは §10 / 別視点調査（2026-08-11）の結果、現状が正しい / 実害なし / 監査記録のみと確定されたものです。改修対象外（または防御的強化・将来対応）として、設計（本文）から分離して記載します。")
out.append("")
out.extend(sec2_3)        # §2.3 現状維持一覧
out.append("")
out.append("## A. 現状維持・監査記録バグの設計")
out.append("")
out.extend(sec3_appendix) # 現状維持・監査バグの詳細設計

result = "\n".join(out)
open(SRC, "w", encoding="utf-8").write(result)

# 検証
new_lines = result.split("\n")
print(f"元: {len(lines)} 行 → 新: {len(new_lines)} 行")
print("=== 新構造の見出し ===")
for idx, l in enumerate(new_lines, 1):
    if l.startswith("# ") or l.startswith("## ") or l.startswith("### "):
        print(f"{idx}: {l[:70]}")
