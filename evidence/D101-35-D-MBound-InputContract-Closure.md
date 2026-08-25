# D101-35-D — M-Bound Input Contract Closure & Symbolic Derivation（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: 設計・証明確定（ソースコード変更 **0** / 契約実装 **0** / 具体値決定 **なし** / D102 GO-NO-GO 未判定）
- **判定**: **PASS**（D-1〜D-6 全ステップ完了。C の proof gap（window boundary case）を閉鎖）
- **基準**: ConvoPeq.md 16:25 版内容＋ fresh trace（ソース未变更のため現行最新と同一）

---

## 1. D-1: C proof の形式閉鎖（window boundary / baseline continuity）

### 1.1 指摘された proof gap

D101-35-C の証明は「O_w(w) ≥ est(t*)、t* は window 内の tick」を前提した。
τ が window 開始直後で t* が前 window の tick になるケースを明示扱いしていなかった。

### 1.2 実コード確認（ISRWorldRetirementTelemetry.h:192-213 `beginWindow`）

```cpp
const auto a0 = acquireObserved();          // ← 先に A を読む
const auto r0 = releaseObserved();          // ← 次に R を読む
firstEstimate = a0 − r0;
publishAtomic(windowMax_, firstEstimate);   // ★ windowMax 初期値 = firstEstimate
```

✅ **指示の修正方針どおり、`beginWindow()` は windowMax を firstEstimate = A0 − R0 で初期化する**
（0 リセットではない → baseline continuity 構造的に存在）。

### 1.3 形式閉鎖（virtual observation point 定理）

**観測点集合の定義**: 各 window w の観測点 ≜ {Start} ∪ {w 内の全 samplerTick} ∪ {End snapshot}。

| 性質 | 根拠 |
|---|---|
| 観測点は G_bound 間隔以下で反復する | environment premise（G_bound 定義そのもの — tick 間隔 ≤ G_bound、Start/End も tick で駆動） |
| windowMax(w) ≥ 全観測点での obs-value | Start: firstEstimate（:210）/ tick・End: `updateWindowMax(estimate)`（:233 / :250 確認済み） |

**閉鎖された境界 case**: ピーク τ ∈ w に対し、o ≜ τ 以前で最後の観測点とする。

```text
(i) τ − o ≤ G_bound
    （o の次の観測点 o' は o + G_bound 以内に存在し、o が「τ 以前の最後」である以上 τ < o'）

(ii) B(τ) − ObsVal(o) ≤ K + λ·(τ − o) ≤ K + λ·G_bound
     （B の増加は execute のみ。executes(o,τ] ≤ backlog(o) + arrivals(o,τ] ≤ K + λ(τ−o)）

(iii) O_w(w) ≥ ObsVal(o) ≥ B(o)
      ∴ P − O_w(w) ≤ K + λ·G_bound   … 前 window の tick に依存しない閉じ方 ✅
```

これで D101-35-C の証明は **window boundary case を含めて完全閉鎖**。
（t* が window 内に存在する通常ケースは §2 の元の議論がそのまま成立。）

### 1.4 残留 formal caveat（誠実な記録 — OBSERVATION）

A0/R0（および各 tick の a/r ペア）は**別個の atomic load**のため、2 load 間に割り込んだ
event により baseline/estimate が ±ずれ得る（adversarial scheduling 下で形式的には非有限）。

- **方向分析**: beginWindow は A→R 順のため、skew は baseline を **過大方向**に逸脱させる
  （pending release が反映前のため）→ O_w を押し上げ → E_w を縮小 = **safe 方向**。
  sampleWindow/closeWindow も同順（a 先 / r 後）で整合。
- 実務モデル: 隣接 load 数命令間の割込みイベント ≒ 0〜僅少。
- **将来強化候補**: R 先読み→A 后読みの逆順スナップショット+再検証ループ、または
  単一 128bit スナップショット（CMPXCHG16B）への統合。M 導出の前提にはしない。

---

## 2. D-2: G_bound Contract Candidate（文言確定・値は TBD）

```text
【定義】
G ≜ ある event（acquire / release）が発生してから、その event が
   sampler observation（samplerTick による estimate 計算）に反映されるまでの時間

【G_bound Contract Candidate】
environment premise:
    AudioEngine を取り巻く実行環境は、いかなる場合でも
    G ≤ G_bound < ∞ を満たすことを保証する。
    （G_bound には公称 cadence 100ms、message thread のスケジューリング遅延、
      missed tick、tick 内処理順序のすべてを包含する。）

【明文化】
nominal cadence ≠ G_bound。
100ms は T_sampler（公称値）であり、G_bound の根拠としては使用しない。
```

分類: **純粋な caller/environment contract として成立**（実装変更不要。
enforcement watchdog は将来オプション）。finite M 導出には **YES**。

---

## 3. D-3: λ scope decision（意思決定表）

### 検証した internal producer の構造 bound

| internal producer | 構造 bound | 強度 |
|---|---|---|
| Timer idle publish (#5) | ✅ timerCallback 内実行 → ≤ 1/tick ≒ 10/s | **構造的（hard）** |
| rebuild 完了駆動 publish | △ rebuild dispatch policy 依存（build 完了率 = 外部要因連動） | 半構造 |
| Recovery publish | △ quarantine 発生率依存（外部 build 失敗連動）。ShuttingDown gate + tryAdmit で shutdown 時は閉鎖済み | 半構造 |

### Trade-off 表

| | Model 1（conservative 全包含） | Model 2′（採用推奨: Timer 構造 bound 分離 + 他は全て λ スコープ） |
|---|---|---|
| 有限性 | ✅ M < ∞ | ✅ M < ∞ |
| R_required のタイトネス | 緩い（λ に内部自動系も上乗せ） | **タイト**（Timer #5 は構造 bound 列挙で除外） |
| proof maintenance | 軽い（λ 一本） | 中（Timer #5 の timerCallback 構造依存を証明として保持） |
| 失効リスク | 低（λ を大きめに取れば不変） | Timer 実行モデル変更時に sub-bound の再証明が必要 |

### 決定（D101-35-D としての確定）

> **Model 2′ を採用する。**
>
> - λ_prod_bound の scope = **Timer callback 内包経路以外の全 publication producer**
>   （PrepareToPlay ×2 / Transition / ReleaseResources:175 / rebuild 完了駆動 / Recovery）
> - Timer idle publish (#5) は「≤ 1 per timerCallback tick」の構造 bound として別枠管理
> - Model 1 へのフォールバック条件: Timer 実行モデル変更時

---

## 4. D-4: M_bound symbolic 固定

```text
╔══════════════════════════════════════════════╗
║  M_bound ≜ K + λ_prod_bound × G_bound        ║
║         = 4096 + λ_prod_bound × G_bound      ║
║  （K = 4096 はコード定数・他は抽象有限契約定数）║
╚══════════════════════════════════════════════╝
```

- 具体値は未決定（λ_prod_bound / G_bound の値決定は次フェーズ）
- `M_bound < ∞` は §1 の定理により保証

---

## 5. D-5: O_w contract / positivity condition

D102 の `R_required = ceil(M_bound / O_w)` は O_w > 0 を要求する。

| 項目 | 確定 |
|---|---|
| O_w の定義（D102 入力用） | **reference measurement window（Start〜End）の windowMax** = `max(firstEstimate, 各 tick estimate, finalEstimate)`。単一時点の現在値ではなく window 内保証値 |
| positivity 条件 | `O_w ≥ 1` を D102 evaluation precondition とする。根拠: O_w = 0 ⟺ 測定 window 中一度も published World が commit されなかった（engine 非稼働相当）であり、R_required は意味をなさない |
| O_w = 0 時の扱い | **D102 deferred**（その window で GO/NO-GO 判定を行わず、World commit を含む window で再測定）。「O_w=0 でも安全」と解解釈しない |
| 追加契約 | measurement window には ≥1 の acquire が含まれること（test protocol 側で担保） |

---

## 6. D-6: D102 GO/NO-GO 入力表

| Input | Status |
|---|---|
| K | **4096 / fixed（コード定数）** |
| G_bound | finite contract candidate / **wording 確定・value TBD** |
| λ_prod_bound | finite contract / **scope 確定（Model 2′）・value TBD** |
| M_bound | **`4096 + λ_prod_bound × G_bound`（symbolic・< ∞ 証明済み）** |
| O_w | measurement window の windowMax / **precondition: ≥ 1** |
| R_required | `ceil(M_bound / O_w)`（symbolic） |
| B_max^true | D102 derivation（未着手） |
| D102 | GO / NO-GO **未判定** |

---

## 7. Gate 判定

| Gate | 内容 | 判定 |
|---|---|---|
| D-1 | window boundary / baseline continuity の形式閉鎖 | ✅ §1（virtual observation point 定理 + beginWindow 実装一致） |
| D-2 | G_bound contract wording | ✅ §2（nominal cadence ≠ G_bound 明記） |
| D-3 | λ scope decision | ✅ §3 Model 2′ 採用（trade-off 表付き） |
| D-4 | M_bound symbolic | ✅ §4 |
| D-5 | O_w contract / positivity | ✅ §5 |
| D-6 | D102 input table | ✅ §6 |
| 制約 | コード 0 / 契約実装 0 / 具体値 0 / D102 未判定 | ✅ 全遵守 |

# VERDICT: D101-35-D = **PASS**

---

## 8. 次ステップ

```text
D101-35-D PASS（本報告書）
      ↓
【ユーザー意思決定】 λ_prod_bound / G_bound の具体値（または導出手順の承認）
      ↓
D102: B_max^true 導出 + R_required 計算 + GO/NO-GO 判定
```

並行残務: I4_DESIGN_CONTRACT.md 更新 + evidence 報告書群（D101-33-F/34-A/B/C/D/35-A/B′/C/D）+
O-1（R6 メソッド名表記）のコミット判断。
