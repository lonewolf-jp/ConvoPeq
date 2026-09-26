# P1-5-IR-P2-E — BLOCK-RESOLUTION AUDIT（read-only）

- **作成**: 2026-09-22 / work113
- **スコープ**: Step 3-E-3（`--buzz-os` SIGSEGV の最小切り分け・production 変更なし）
- **成果物**: 本レポート（実装はまだしない）

---

## 1. 前提（P1-5-IR-P2 の現状）

- **FLAG 設定**: `--buzz-os`（SIGSEGV 側の既存フラグ・K2 設定維持）
- **測定系**: AudioEngineHarness（test harness）— production/test の切り分けは維持済み
- **測定範囲**: SIGSEGV 発生条件の特定（OS=1 独自の幾何不整合を除外済み）

---

## 2. 現行 source での OS=1/OS=2 の構造的差分（読み取り済み）

| field | OS=1 | OS=2 | 備考 |
| --- | --- | ---: | ---: |
| probe vehicle | SIGSEGV | SIGSEGV | 同型 |
| OS factor | 1 | 2 | 明示 |
| IR length | 96000 | 192000 | OS に比例 |
| publication | RCU/RCU | RCU | 変化なし |
| conv generation | 9 → 14 | — | IR swap 経路 |
| publication idx | 8 | 8 | 変化なし |

**SIGSEGV 発生位置の仮説**（実測）:
1. **production 変数の swap 遅延**（probe 側は SC=OFF のまま・`--buzz-order` が未反映）
2. **signal generator / IR swap の非同期待機不足**
3. **harness 側の tap lifetime**

→ **SIGSEGV の直接原因は特定できていない**。SIGSEGV の取得は次の承認を待つ。

---

## 6. Next authorization request

- x96dbg crash attribution を**実行する**には `AudioEngineHarness.exe` の x96dbg 配下で SIGSEGV 時の例外ハンドラ/レジスタ/スタックを取得する必要がある。
- 本 audit は SIGSEGV 発生時の crash dump 取得までを範囲とし、**production 変更は一切行わない**。

**SIGSEGV root cause は UNKNOWN（調査中）**。修正は次の承認を待つ。
