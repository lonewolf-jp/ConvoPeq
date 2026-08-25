# D102-B — True-Peak Bound / R-required Derivation（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: 数学的導出・契約同期設計（read-only / ソースコード変更 **0** / 契約ファイル変更 **0** /
  λ·G·M·R の具体値決定 **なし** / R_cap・T2 authority 非接続）
- **判定**: **D102 mathematical derivation = GO**
  （B-1〜B-4 全閉鎖。数値確定は λ_prod_bound / G_bound / O_denom 値のユーザー承認待ち）

---

## 1. B-1: corrected Model 2′ `N_timer(G)` の形式監査

### 1.1 問題の再定式化

D101-35-D-R が指摘したとおり、Timer #5 を λ_ext から除外した場合、
gap 中の Timer 経路 contribution を別項 `N_timer(G)` として証明する必要がある。

```text
N_timer(G) ≜ 長さ G_bound の区間中に発生し得る Timer idle publish (#5) acquire 回数
```

### 1.2 構造的導出の監査結果

| 要件 | 実測 | 判定 |
|---|---|---|
| (a) callback 発生レートの上界 | JUCE Timer framework contract: callback は設定 interval（100ms）を下回る頻度で発火しない。遅延時は coalesce（積み上げ発火しない） | ⚠️ **framework 外部前提**（我々のコードからは導出不可・文書化可能） |
| (b) callback 1回あたりの #5 acquire 数 | ≤ 1（Timer.cpp:994 の単一 call site、`currentAfterFade != nullptr` guard 内） | ✅ コード確認 |
| (c) samplerTick との同一 callback 関係 | samplerTick(:415) と #5(:994) は**同一 timerCallback 内**。samplerTick が先、#5 が後 | ✅ 実測 |

### 1.3 導出

```text
N_timer(G_bound) ≤ ⌈G_bound / T_sampler⌉ + 2
    （各区間中の callback 発生回数 ≤ ⌈G/T_sampler⌉ + 1 ＋ 境界の切り上げ余剰）
```

### 1.4 判定: 条件付き閉鎖

```text
閉鎖条件: JUCE Timer framework premise
    「Timer callback は設定 interval を下回る頻度では発火しない（coalescing property）」
    を外部 framework 契約として文書化すること。
```

- ✅ **corrected Model 2′ = 閉鎖可能**（framework premise を documented external dependency として記録）
- G_bound 自体が environment premise であることと同格の外部前提であり、
  measurement 昇格とは性質が異なるため**契約追加として正当**
- ❌ 閉閉鎖しない場合のフォールバック: **Model 1**（`M₁ = K + λ_all·G_bound` —
  Timer 経路も λ_all に包含。λ_all は「全 producer 含む submission rate 上界」として
  単一の環境前提で済む）

> **D102-B 決定**: **corrected Model 2′ を primary 採用**（JUCE premise を明記）。
> Model 1 を fallback として維持。両者とも M < ∞ であることに変わりなし。

---

## 2. B-2: O_denom 候補比較

| 候補 | 内容 | 安全性 | 証明可能性 | 判定 |
|---|---|---|---|---|
| (a) eligible window の O_w のみ | O_w > 0 の window のみ分母に採用 | ✅ 安全（正の値のみ） | △ 「O_w > 0 window が存在する」は観測依存（保証されない） | 条件付き |
| (b) bootstrap commit 以降 B ≥ 1 invariant | 初回 committed publish 以降 `current ≠ nullptr` → true outstanding ≥ 1 が構造的に継続 | ✅ | ✅ **コードから証明可能**（下記） | ✅ **採用** |
| (c) 複数 window 統計 max | max_w(O_w(w)) | ✅ 安全 | △ 全ての window で O_w=0 の場合 0 のまま | (b) と併用で強化 |

### (b) のコード証明

```text
1. Bootstrap commit（Init.cpp:85-90）:
   committed=true → didPublishRuntimeNonRt → onRuntimePublishedNonRt
   → Commit.cpp:406 worldRetirementTelemetry_.onAcquireObserved()
   → Commit.cpp:409 worldRetirementReference_.onAcquire()
   → A++ / refA++

2. 正常 Path B publish（RuntimePublishExecutor.h:74）:
   committed=true → bridge.didPublishRuntimeNonRt → 同経路で A++
   （全 successful publish が計上される — D101-34-B exactly-once 証明と同一基盤）

3. current ≠ nullptr の間:
   当該 world の release event は未発生 → R < A → est ≥ 1 構造的に維持
   （resident world は replacement retire または shutdown clear でのみ release）
```

✅ **`post-first-commit ⇒ observedOutstandingEstimate ≥ 1` はコードから証明可能な invariant。**

### O_denom 確定案

```text
O_denom ≜ D102 対象測定期間における observedOutstandingMax の最大値
          （= 複数 window の windowMax 統合・候補 (c) 形式）

precondition: 測定期間が post-first-commit を含むこと（候補 (b) の構造保証により O_denom ≥ 1）
O_denom = 0 が観測された場合: 測定期間に published World が存在しなかったことを意味し、
          R_required はその期間に無意味 → 再測定または対象期間の再選択
```

✅ **`O_denom > 0` は独立した contract/input として証明可能**（bootstrap invariant による）。

---

## 3. B-3: True peak bound の形式確定

### per-window 形式

```text
∀ eligible window w:
    B_max^true(w) ≤ O_w(w) + M_scope(w)
```

- `O_w(w)`: window w の sampled windowMax（Start firstEstimate 含む・tick estimate の running max・End finalEstimate 含む）
- `M_scope(w)`: 適用モデルの scope 項
  - corrected Model 2′: `λ_ext·G_bound + N_timer(G_bound)`
  - Model 1: `λ_all·G_bound`

### global 形式（O_denom 使用・混同なし）

```text
B_max^true_global ≤ max_w [O_w(w)] + max_w [M_scope] = O_denom + M_scope
```

- `O_denom`（複数 window 統合 max）と各 window の `O_w(w)` は**別の量**として厳密区分
- per-window 不等号の sup を取ることで global 式が従う（各項の単調性より）

---

## 4. B-4: R_required symbolic 導出

```text
╔══════════════════════════════════════════════════════╗
║  R_required ≜ ceil( M_scope / O_denom )              ║
║                                                      ║
║  M_scope（corrected Model 2′）                       ║
║     = K + λ_prod_bound × G_bound + N_timer(G_bound)  ║
║     = 4096 + λ_prod_bound × G_bound                  ║
║       + ⌈G_bound/100ms⌉ + 2                          ║
║                                                      ║
║  O_denom ≥ 1（bootstrap invariant・証明済み）         ║
╚══════════════════════════════════════════════════════╝
```

⚠️ **R_required ≠ R_cap**:

- R_required は「観測値 O_denom が真の peak を下回る量を M で被覆するための
  **multiplicative factor**」であり、runtime の実際の retention capacity を直接決定しない
- `R_cap` / T2 authority への接続は本タスクでは実施していない（指示どおり）
- 旧コメント（R_required 未充足を示すもの）は evidence/contract synchronization の
  対象として次フェーズに記録

---

## 5. B-5: D102 mathematical GO/NO-GO 判定

| 条件 | 判定 |
|---|---|
| M_scope < ∞ | ✅ 全項有限定数の関数（§4） |
| O_denom > 0 | ✅ bootstrap invariant により構造的に ≥ 1 |
| B_max^true ≤ O_denom + M_scope | ✅ §3 per-window 証明の global 拡張 |
| R_required finite | ✅ 有限量の比の ceil |

# VERDICT: **D102 mathematical derivation = GO** 🟢

（※ 数値は未確定。GO は「数学的 closure が得られ、残りは契約定数の値決定のみ」を意味する）

---

## 6. 制約遵守確認

| 制約 | 状態 |
|---|---|
| ソースコード変更 | ✅ 0 |
| I4 contract 変更 | ✅ 0 |
| 具体的 λ 値 | ✅ 未決定 |
| 具体的 G 値 | ✅ 未決定 |
| 実測値の昇格 | ✅ なし |
| R_cap 決定 | ✅ なし |
| T2 authority 変更 | ✅ なし |

---

## 7. 次ステップ

```text
D102-B PASS（本報告書）
      ↓
【ユーザー意思決定】
    ① λ_prod_bound の値（または決定手順）
    ② G_bound の値（environment premise の文言確定含む）
      ↓
D102-C: 数値適用 → R_required 計算 → R_cap/T2 authority 接続判断
      ↓
Phase I NO-GO 解除判断
```

並行残務: I4_DESIGN_CONTRACT.md の D101-35/D102 反映 + evidence 報告書群 +
旧コメント（R_required 未充足表記）の sync + tooling 5点。
