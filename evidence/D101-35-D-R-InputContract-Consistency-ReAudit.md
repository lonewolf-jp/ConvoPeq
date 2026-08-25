# D101-35-D-R — Input Contract Consistency Re-Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only re-audit（ソースコード変更 **0** / 契約ファイル変更 **0** / 具体値決定 **なし**）
- **判定**: **PASS（修正込み）** — D101-35-D の結論のうち **2 点に REQUIRED 級の不整合を確認し、
  本報告書で訂正済み**。訂正後の契約入力は D102 へ引き渡し可能。
- **基準**: ConvoPeq.md 16:25 版内容＋ fresh source trace（Timer.cpp / WorldRetirementMeasurementTests.cpp）

---

## 1. 再監査の結果サマリ

| 項目 | D101-35-D の記述 | 再監査判定 |
|---|---|---|
| D-3 Model 2′ | 「Timer #5 を λ から除外、M = K + λG のまま」 | ❌ **不整合確認** — 証明式の arrivals 項に Timer 経路が含まれるため式が未閉鎖。→ §2 で再導出 |
| D-5 O_w ≥ 1 前提 | 「window に acquire が存在 ⇒ O_w ≥ 1」 | ❌ **撤回** — acquire+release が同一 tick 間隔内で完結すると O_w = 0 のまま（現行設計の想定ケース）。→ §3 で分離再定義 |
| その他（D-1/D-2/D-4/D-6） | — | ✅ 変更なし・維持 |

---

## 2. R1: Model 2′ λ scope / M 式整合性

### 2.1 不整合の内容（認定）

D101-33-C の証明式:

```text
executes(o, τ] ≤ backlog(o) + arrivals(o, τ]
```

の `arrivals` は「execute に至る publication arrival の**全体**」である。
Timer idle publish (#5) も execute に至る arrival である以上、これを λ から除外したまま
`M ≤ K + λ_ext·G` と書くことは**証明式と scope の不一致**。

また「timerCallback 内最大 1 回」という構造事実だけでは、G 区間中の callback 発生回数を
ゼロにはできない（callback は ~100ms ごとに発生するため、G 区間中の #5 貢献は有限だが
別項として明示が必要）。

### 2.2 修正後の Model 2′ 定式化（閉鎖形）

```text
N_timer(G) ≜ 長さ G の区間中に timerCallback が実行され得る回数の上界
           = ⌈G_bound / T_sampler⌉ + 1   （T_sampler = 100ms 公称 cadence）
             ＊ callback 1 回につき idle publish (#5) は高々 1 acquire

修正後 Model 2′:
    M₂′ ≤ K + λ_ext·G_bound + N_timer(G_bound)
        = 4096 + λ_ext·G_bound + ⌈G_bound/100ms⌉ + 1      （すべて有限）
```

- `λ_ext` の scope: Timer callback 内包経路**以外**の全 publication producer
  （PrepareToPlay ×2 / Transition / ReleaseResources:175 / rebuild 完了駆動 / Recovery）
- Timer 経路は構造事実（callback cadence）のみから有限化され、**外部 rate 契約に依存しない**
  ← ここが Model 2′ を維持する価値

### 2.3 Model 比較（修正後）

| | 式 | proof burden | 境界タイトネス |
|---|---|---|---|
| **Model 1（corrected）** | M ≤ K + λ_all·G_bound | 最小（λ_all 1 本の契約で完結。Timer 経路も包含） | λ_all に内部自動系も含めるため緩い |
| **Model 2′（corrected）** | M ≤ K + λ_ext·G_bound + N_timer(G_bound) | 中（callback cadence 構造の証明を保持） | **タイト**（Timer #5 が構造定数項に分離） |

両者とも有限。**D102 の R_required = ceil(M/O_w) を最小化する観点では Model 2′（corrected）が有利。**
証明構造の単純さを優先するなら Model 1。

> **D101-35-D-R の推奨**: 数式としては両者とも閉鎖完了。D102 へは
> **corrected Model 2′ を primary、Model 1 を fallback** として提出することを推奨する
> （最終選択は D102 冒頭の意思決定事項）。

---

## 3. R2: O_w denominator semantics 再固定

### 3.1 「acquire 存在 ⇒ O_w ≥ 1」の撤回

機構（実コード構造から確認）:

```text
windowMax は Start 時 firstEstimate(=A0−R0) で初期化され、以降各 tick の
estimate = signedWide(A) − signedWide(R) でのみ更新される。

acquire と release が同一 tick 間隔内で完結した場合:
    A と R が同量増加 → estimate 不変 → ピークは windowMax に反映されない
    ∴ O_w = 0 のまま T_w = 1, E_w = 1 が成立し得る
```

これは現行設計が**意図的に測定対象としているケース**である
（`WorldRetirementMeasurementTests.cpp` 先頭コメント: 「Burst（本命）: sampler interval
より短い時間幅に retire を集中（T_w > O_w を意図的に生成できるか）」:300,
「O_w(sampled) < T_w(event-driven max) を意図的に生成できる」:61）。

✅ **「measurement window に acquire が存在 ⇒ O_w ≥ 1」を正式に撤回する。**

### 3.2 O_w と R_required denominator の分離

| 量 | 定義 | 用途 |
|---|---|---|
| O_w | sampled windowMax（measurement 出力） | E_w 測定・M妥当性評価の**観測値**。0 も有効値 |
| **O_denom**（新設・D102 入力） | R_required の分母として契約可能な outstanding baseline | D102 で定義・確定 |

O_denom の候補（D102 で決定）:

```text
(a) O_w そのもの（ただし O_w > 0 の window に限る）
(b) 定常 resident 下限: engine 稼働中は published runtime が常在するため
    bootstrap commit 以降は B ≥ 1 が構造的に成立する点を利用した下限
(c) 複数 window の統合統計（例: 複数 window の O_w 最大値）
```

本監査での確定事項: **`R_required = ceil(M/O_w)` が適用可能なのは
「O_w > 0 かつ window が定常状態を含む」という eligible 条件を満たす window のみ**。
O_w = 0 の burst window は denominator として使用しない（E_w の測定記録としてのみ保持）。

---

## 4. R3: `B_max^true ≤ O_w + M` の適用条件再固定

D101-33-C 証明（§2 参照・再掲）により、任意 window w 内のピーク P について:

```text
P − O_w(w) ≤ K + λ_scope·G_bound + N_timer(G_bound)（scope に応じた項のみ）
```

が成立する。よって:

```text
B_max^true(w) ≤ O_w(w) + M_scope     … 全 window で成立（M が scope を正しく覆う場合）
```

**適用条件（D102 入力として固定）**:

1. 同一 windowId の O_w と M を比較する（D95 既存契約）
2. M は window 内ピークを覆う scope であること
   - Model 1: λ_all·G_bound（全 producer 含む）
   - corrected Model 2′: λ_ext·G_bound + N_timer(G_bound)
3. O_w(w) > 0 であること（denominator eligibility、R3 §3）
4. Q2 producer join 前提（clear 後の swap-in なし）は従来どおり前提

---

## 5. D102 への引き渡し条件（更新版）

```text
D101-35-C 証明            ✅（M ≤ K + λG 形式・Case B 閉鎖済み）
D101-35-D 入力契約        ✅（G_bound wording / O_w positivity）
D101-35-D-R 本監査        ✅ Model 2′ 式修正 + O_w/denominator 分離 + 適用条件固定
      ↓
D102-B 冒頭の意思決定:
    ① Model 1（λ_all 単純式）vs corrected Model 2′（λ_ext + N_timer）の選択
    ② O_denom の定義選択（候補 a/b/c）
      ↓
D102-B: B_max^true 導出 → R_required 計算 → GO/NO-GO
```

---

## 6. Gate 判定

| Gate | 内容 | 判定 |
|---|---|---|
| R-1 | Model 2′ の式不整合を認定・修正式を導出 | ✅ §2 |
| R-2 | O_w = 0 が有効 burst case であることを確認・「⇒O_w≥1」を撤回 | ✅ §3 |
| R-3 | O_w と denominator の分離 + eligible 条件定義 | ✅ §3/§4 |
| R-4 | B_max^true ≤ O_w + M 適用条件の再固定 | ✅ §4 |
| R-5 | コード変更 0 | ✅ |
| R-6 | 契約ファイル変更 0 | ✅ |
| R-7 | 具体値決定 0 | ✅ |
| R-8 | R/R_cap/T2 authority 非接続 | ✅ |

# VERDICT: D101-35-D-R = **PASS**（修正込み・D102 引き渡し可能）

---

## 7. 補足

- 本再監査により D101-35-D の結論は **2 点訂正**された。訂正後の契約入力は本報告書が
  最新の authoritative record となる（I4_DESIGN_CONTRACT.md への同期は次の文書タスクで実施）。
- 判定ルール遵守: 実測値の昇格なし / queue capacity からの直接 rate bound 導出なし /
  「直列だから有限」推論の不使用（T_exec 下限の不存在を Case B として明示）。
