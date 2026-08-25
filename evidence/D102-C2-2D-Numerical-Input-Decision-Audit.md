# D102-C2-2D — Numerical Input Decision Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS** — 全入力の決定根拠・決定権者・必要証拠が閉じており、未決定値が明確に切り分け済み
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容（ソース未変更）＋ 20:08 版 I4 契約同期反映確認

---

## 1. D-G01 基準化

```
ConvoPeq.md: 2026-08-25 20:08 再生成版を確認
  （C2-2C の契約同期が反映されている = authoritative 最新版）
git status --short -- src → 変更なし（D101-33-C 以降ソース未变更）
git diff --check          → エラーなし
```

File Library の古いスナップショット（⌈G/100ms⌉+2 等を含む版）は使用せず、
ローカル実ファイルの 20:08 版を唯一のコード基準とすることを確認。

---

## 2. Gate 1: λ_prod_bound admissibility

### 2.1 producer inventory 再確認（8 sites）

| # | producer | call site | 定常状態での発生 | λ 加算 |
|---|---|---|---|---|
| P-1 | PrepareToPlay #1 | :155 | lifecycle 時のみ | ❌ scope 外 |
| P-2 | PrepareToPlay #2 | :277 | 同上 | ❌ scope 外 |
| P-3 | Timer #5 | :994 | ✅ 定常的に発生（≤1/tick ≒ 10/s） | ❌ N_timer 項へ分離 |
| P-4 | Transition #6 | :25 | ユーザー/自動遷移時 | ⚠️ workload contract 必要 |
| P-5 | ReleaseResources | :175 | shutdown 中のみ | ❌ scope 外 |
| P-6 | deferred resubmit | PubExec:53-66 | 初回失敗時の再試行 | ❌ 二重計上除外確定 |
| P-7 | Orchestrator Path A | Orchestrator 経由 | ユーザー publish 要求時 | ⚠️ workload contract 必要 |
| P-8 | Recovery | quarantine 検出時 | build failure 連動 | ⚠️ workload contract 必要 |

### 2.2 各項の admissibility 判定

| 項目 | workload upper bound 宣言可能か | 根拠 | 判定 |
|---|---|---|---|
| λ_transition | ✅ 可能 — DSP transition はユーザー/オートメーション要求駆動。運用仕様で「同時 transition 数 ≤ N」「最小間隔 ≥ M秒」等を宣言可能 | CoordinatorLoop 直列化により acquire は排他的。transition 発生率は運用設計で制御可能なパラメータ | ✅ admissible |
| λ_user_publish | ✅ 可能 — UI/API 経由の publish 要求レート。INV-PUB-1 により CoordinatorLoop で直列化されるため、要求率が如何に高くても commit rate は loop 処理能力以下 | queue backpressure（4096 full → reject）が構造的な上限 | ✅ admissible |
| λ_recovery | ✅ 可能 — quarantine 検出は build failure 連動であり、build failure 自体が外部要因（不正 IR 等）。Recovery gate（Coordinator.cpp:825 ShuttingDown check）により shutdown 中は遮断済み | quarantine 発生率は運用環境依存だが、workload contract として上限を宣言可能 | ✅ admissible |
| lifecycle（P-1/P-2/P-5） | steady-state scope 外として除外してよいか | ✅ **してよい** — prepareToPlay/releaseResources は engine ライフサイクル遷移時のみ発火し、定常運転中には発生しない。measurement window も定常区間に限定する方針（D101-35-A §1.3）と整合 | ✅ 除外確定 |
| deferred resubmit | 二重計上なしで除外できるか | ✅ **できる** — 失敗 attempt は onAcquireObserved を発火しないため acquire 会計に寄与しない。resubmit 成功時の acquire は元 world の obligation として一度のみ計上 | ✅ 除外確定 |

### 2.3 λ_prod_bound 構成の確定

```text
λ_prod_bound = λ_transition + λ_user_publish + λ_recovery

各項:
  λ_transition   … workload contract（運用仕様で宣言可能・値はユーザー決定）
  λ_user_publish … workload contract（queue backpressure 4096 + INV-PUB-1 直列化で
                    構造的上限あり・具体的値は運用仕様で決定）
  λ_recovery     … workload contract（quarantine policy 依存・ShuttingDown gate 済み）

加算の正当化: CoordinatorLoop 直列化により acquires は排他的処理。
Σ(個別上界) ≥ 合成レート は保守的正しい上界。
```

✅ **Gate 1 = PASS** — 全項 admissible。値は D102-C2 でユーザー/運用仕様から決定。

---

## 3. Gate 2: G_bound admissibility

### 3.1 構成要素の basis 分類

| 要素 | 値の出所 | basis 区分 |
|---|---|---|
| T_sampler = 100ms | `startTimer(100)` Init.cpp:121 / `kExpectedTickIntervalUs = 100'000` Telemetry.h:311 | **source constant** ✅ |
| K_starve | environment contract — **ユーザー/製品仕様で決定必須**。例示値 5s の昇格は禁止 | environment premise |
| δ_processing | samplerTick 内 atomic ops の処理時間。**K_starve premise に包含可能**（同一 message thread 上の処理のため K_starve ≤ の場合 δ_processing ≤ K_starve が自明に従う） | subsumed |

### 3.2 starvation 依存関係の明示

```text
unbounded message-thread starvation
    ↓
G_bound = ∞
    ↓
finite M proof invalid
    ↓
D101 有限性証明が無意味化
```

この依存関係を断つためには K_starve を environment premise として採用する必要がある。

### 3.3 K_starve を正式 environment premise として採用できるか

| 観点 | 判定 |
|---|---|
| 技術的整合性 | ✅ JUCE Timer coalescing property 下、message thread stall は timer callback の遅延として現れる。K_starve はその最大許容遅延の宣言であり、JUCE 動作と矛盾しない |
| 検知可能性 | ✅ missedTickCount watchdog により超過を検知可能（将来実装・現在は記録のみ） |
| 製品要件との整合 | △ modal dialog 等で message thread が停滞するシナリオでの妥当性は製品設計判断。本監査では技術的境界の明示のみ実施 |

✅ **K_starve を正式な environment premise として採用可能**。
値の決定はユーザー/製品設計者の意思決定事項（例示 5s の昇格は禁止・指示遵守）。

✅ **Gate 2 = PASS**。

---

## 4. Gate 3: δ_processing の扱い

| Option | 内容 | 判定 |
|---|---|---|
| (a) 実測値を採用 | 実測データが現状不存在 | ❌ 選択不可 |
| (b) 実測 + 安全マージン → 契約値 | 実測後に適用可能。現時点では data 不足 | △ 将来選択肢 |
| (c) 0 扱い | 「atomic ops だけなので無視」は証明ではない | ❌ 指示どおり不採用 |
| **(d) K_starve premise への帰着** | δ_processing は samplerTick の message thread 上処理時間であり、K_starve premise（message thread callback 遅延の最大許容値）に**論理的に包含される** | ✅ **採用** |

```text
確定: δ_processing は独立した TBD 項目から除外し、
      K_starve premise に包含されることを契約上明記する。

根拠: samplerTick の処理は message thread 上で実行されるため、
      message thread 自体が K_starve 以上停滞しないという premise の下では
      δ_processing > K_starve となることは構造的に不可能。
      独立項として扱う必要がない。
```

✅ **Gate 3 = PASS**（δ_processing = K_starve 包含・独立 TBD 項目から除外）。

---

## 5. Gate 4: O_denom measurement readiness

| 項目 | 状態 |
|---|---|
| eligibility protocol | ✅ 機械化済み（D101-34-B/C 確定・E1-E4 条件） |
| 取得経路 | ✅ `telemetry.lastClosedSnapshot().snap.windowMax` |
| production data | ❌ 未取得（実稼働 window での測定が必要） |
| harness data | △ WorldRetirementMeasurementTests で partial 取得可能 |
| positivity guarantee | ✅ bootstrap invariant により post-first-commit で est ≥ 1 構造保証 |

→ **O_denom = measurement required**（protocol ready・データ収集は次フェーズ）。

---

## 6. Gate 5: numerical decision matrix

| Input | Basis | Candidate value | Evidence | Admissible? |
|---|---|---:|---|---|
| K | source constant | 4096 | Coordinator.h:693 | ✅ A |
| T_sampler | source/framework | 100 ms | startTimer(100) / kExpectedTickIntervalUs | ✅ A |
| N_timer(G) | derived function | ⌊G/T_sampler⌋ + 1 | JUCE premise + math | ✅ A |
| λ_transition | workload contract | TBD | 運用仕様（未定義） | 🔲 B |
| λ_user_publish | workload contract | TBD | 運用仕様（未定義） | 🔲 B |
| λ_recovery | workload contract | TBD | 運用仕様（未定義） | 🔲 B |
| K_starve | environment contract | TBD | 製品設計判断 | 🔲 B |
| δ_processing | ~~measurement~~ → **subsumed into K_starve** | —（独立項廃止） | — | ✅ 解決 |
| G_bound | derived | TBD | K_starve + T_sampler 確定後 | 🔲 B 依存 |
| O_denom | measurement | TBD | eligible campaign | 🔲 C |
| M₂′ | derived | TBD | 全入力確定後 | 🔲 B+C 依存 |
| R_required | derived | TBD | M₂′ + O_denom 確定後 | 🔲 依存 |

**A = 今すぐ確定可能 / B = 契約者の意思決定が必要 / C = 測定が必要**

---

## 7. Gate 6: C2-2 へ進める条件の分離

### A. 今すぐ決定可能（確定済み）

```text
K               = 4096                          （source constant）
T_sampler       = 100 ms                        （source constant）
N_timer(G)      = ⌊G / T_sampler⌋ + 1           （mathematical function）
δ_processing    → K_starve premise に包含（独立項廃止）
M_boundary      = 0（M_gap に吸収・D95 固定点1）
```

### B. 契約者の意思決定が必要（D102-C2 冒頭で実施）

```text
λ_transition    … 運用仕様による transition 完了率上界の宣言
λ_user_publish  … 運用仕様による publish 要求率上界の宣言
λ_recovery      … 運用仕様による quarantine 発生率上界の宣言
K_starve        … message thread 最大許容遅延の宣言（例: 5s は参考値）
```

### C. 測定が必要

```text
O_denom         … 実稼働 telemetry からの eligible window 抽出
                  （protocol ready: lastClosedSnapshot().snap.windowMax）
δ_processing    … K_starve 包含により独立測定不要（Gate 3 で解決）
```

---

## 8. 最重要確認：数値の勝手な仮定なし

| 禁止項目 | 遵守 |
|---|---|
| `λ_timer = 10/s` を λ_prod_bound に加算 | ✅ 不使用（Timer #5 は N_timer 項に構造的分離済み） |
| `K_starve = 5s` の正式採用 | ✅ 例示値として維持・正式値はユーザー決定 |
| `G_bound = 100ms` 採用 | ✅ 不使用（nominal cadence ≠ bound） |
| `O_denom = 1` の硬直採用 | ✅ 不使用（bootstrap invariant による下限は証明済みだが値は測定） |
| `M₂′` / `R_required` の数値 | ✅ 未採用 |

authoritative formula は引き続き:

```text
M₂′ = 4096 + λ_prod_bound × G_bound + ⌊G_bound / T_sampler⌋ + 1
R_required = 1 + ceil(M₂′ / O_denom)
```

---

## 9. 成果物チェックリスト

1. ✅ **λ producer contract admissibility table**（§2.1/§2.2）
2. ✅ **G_bound premise/admissibility table**（§3.1/§3.3）
3. ✅ **δ_processing decision**（§4 — K_starve 包含により独立項廃止）
4. ✅ **O_denom measurement readiness audit**（§5 — protocol ready/data pending）
5. ✅ **numerical decision matrix**（§6 — A/B/C 分類完了）
6. ✅ **数値採用可否の判定**（全項明記・具体値ゼロ）
7. ✅ **C2-2 に進めるためにユーザーが決定すべき値の明示**（§7 B 項4件）

# VERDICT: D102-C2-2D = **PASS**

（各数値の決定根拠・決定権者・必要証拠が閉じており、未決定値が明確に切り分けられている）

---

## 10. 次ステップ: D102-C2-2 — Numerical Value Determination

```text
D102-C2-2D PASS（本報告書）
      ↓
【ユーザー意思決定】B 項4件の値確定
    λ_transition, λ_user_publish, λ_recovery, K_starve
      ↓
【測定】O_denom（eligible campaign から取得）
      ↓
D102-C2-2: M₂′ / R_required 数値計算
      ↓
D102-C3: R_cap / T2 authority compatibility audit
      ↓
D102-C4: retention-capacity GO/NO-GO → Phase I NO-GO 解除判断
```
