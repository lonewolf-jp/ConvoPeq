# P1-5-IR-P2-E — BLOCK-RESOLUTION AUDIT（OS=1x SIGSEGV）

**作成**: 2026-09-22 / work113 P1-5-IR Phase 2-E
**基準 source**: HEAD `1e9e63e3`（ConvoPeq.md Generated 2026-09-22）相当

---

## 1. Latest ConvoPeq identity

- 参照版: `ConvoPeq.md` Generated **2026-09-22 06:20:11**（HEAD と一致）
- 対象: `--buzz-os` SIGSEGV の read-only isolation（production/test 境界を跨がない）
- 範囲: SIGSEGV 発生条件の特定（OS=1 独自の幾何を除外済み）

## 2. 現行の構造的差分（読み取り済み）

| field | OS=1 | OS=2 | 備考 |
| --- | --- | ---: | ---: |
| probe vehicle | SIGSEGV | SIGSEGV | 同型 |
| oversampling factor | 1 | 2 | 明示 |
| IR length | 96000 | 192000 | OS に比例 |
| publication seq | 8 | 8 | 変化なし |
| conv generation | 9 → 14 | — | IR swap 経路 |
| report idx | 8 | 8 | 変化なし |

**SIGSEGV root cause**: UNKNOWN（probe log のみでは不整合を証明できない）。したがって「production OS=1 独自のバグ」とは断定しない。SIGSEGV の原因は調査中。

---

## 3. 既存 instrumentation による OS=1/OS=2 差分（読み取り済み）

| field | OS=1 | OS=2 | 備考 |
| --- | --- | ---: | ---: |
| probe vehicle | SIGSEGV | SIGSEGV | 同型 |
| OS factor | 1 | 2 | 明示 |
| publication seq | 8 | 8 | 変化なし |
| conv generation | 9 → 14 | — | IR swap 経路 |
| report idx | 8 | 8 | 変化なし |

**SIGSEGV の thread 内訳は特定できていない（debugger 情報が初回障害で失効）。したがって「production OS=1 bug」とは確定しない。**

---

## 6. Next authorization request

- x96dbg crash attribution を**実行する**には、SIGSEGV 時の例外ハンドラ/レジスタ/スタックを取得する必要がある。
- 本 audit は SIGSEGV 発生条件の特定までを範囲とし、**production 変更は一切行わない**。

**SIGSEGV root cause は UNKNOWN（調査中）**。修正は次の承認を待つ。
