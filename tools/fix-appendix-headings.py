# 一時スクリプト: Appendix 内の見出しを整理（CRLF 維持）
path = "doc/work88/big_bug/INTEGRATED_BUG_FIX.md"
with open(path, "rb") as f:
    data = f.read().decode("utf-8")
lines = data.split("\r\n")  # CRLF で分割

# 587 行目（1-indexed）: ### 2.3 → ## A. 現状維持バグ一覧
# 600 行目（1-indexed）: ## A. → ## B. 現状維持・監査記録バグの設計
targets = {
    587: "## A. 現状維持バグ一覧",
    600: "## B. 現状維持・監査記録バグの設計",
}
for ln, new in targets.items():
    idx = ln - 1  # 0-indexed
    old = lines[idx]
    print(f"L{ln}: [{old[:60]}] -> [{new[:60]}]")
    lines[idx] = new

result = "\r\n".join(lines)
with open(path, "wb") as f:
    f.write(result.encode("utf-8"))
print("DONE")
