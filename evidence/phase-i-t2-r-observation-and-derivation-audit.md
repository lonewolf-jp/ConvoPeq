# Phase I-T2/R 事前監査 — 観測・決定監査（R_required 導出可能性）

- **日付**: 2026-08-19
- **対象**: Phase I-T2/R（`REPAIR_PLAN2-dash2.md §1.2` / `doc/work88/I4_DESIGN_CONTRACT.md D63-D100`）— T1 telemetry が収集した観測データから `R_required` を有限に導出可能かの事前監査
- **目的**: T1 telemetry の integrity を検証し、収集済み観測データが `M`（安全側上界）の数学的バインド（D101）を満たすかを判定する。**R_required の確定・ReservationExhausted の実生成・Phase I 本実装は行わない**（監査のみ）
- **判定**: **B. R = UNDETERMINED** — telemetry は正常だが、M の数学的バインド（D101）未着手 + 観測データ不足により R_required は導出不能

---

## 1. T1 telemetry integrity（T1 の実装・検証状況）

### 1.1 実装済み T1 コンポーネント

| コンポーネント | ファイル | 実装状況 | 検証 |
| --- | --- | --- | --- |
| `ObservationWindowTag`（Normal/Stall/Shutdown/Catastrophic） | `ISRWorldRetirementTelemetry.h:11` | ✅ | D76 CLOSED |
| `MeasurementState`（Idle/StartRequested/Running/EndRequested/Closed） | `ISRWorldRetirementTelemetry.h:32` | ✅ | D91 CLOSED（CAS transition / single owner） |
| `MeasurementSnapshot`（windowId / startAcquire / startRelease / endAcquire / endRelease / finalEstimate / windowMax / timestamps / sampleCount / maxSamplingGapUs / missedTickCount / counterWrapped / valid） | `ISRWorldRetirementTelemetry.h:42` | ✅ | D91 CLOSED（trivially copyable / atomic publish） |
| `samplerTick`（window transition owner） | `ISRWorldRetirementTelemetry.h:165` | ✅ | D91 CLOSED（MessageThread / 100ms cadence） |
| `lastClosedSnapshot`（immutable export） | `ISRWorldRetirementTelemetry.h:195` | ✅ | D91 CLOSED（valid flag / atomic read） |
| `observedOutstandingEstimate`（signedWide(A) - signedWide(R)） | `ISRWorldRetirementTelemetry.h:108` | ✅ | D82 CLOSED（signed/wide-domain arithmetic） |
| `observedOutstandingMax`（Non-RT sampler only） | `ISRWorldRetirementTelemetry.h:120` | ✅ | D83 CLOSED（sampler 責務分離） |
| `WorldRetirementReferenceObserver`（T_w = reference max） | `ISRWorldRetirementReference.h` | ✅ | D94/D95/D99 CLOSED（event-driven / Non-RT / T1 と分離） |
| 100ms sampler（timerCallback） | `AudioEngine.Timer.cpp:435` | ✅ | D83 CLOSED（A/R loads → signedWide → estimate → max → window tag） |
| evidence export（`world_retirement_telemetry.json`） | `AudioEngine.Commit.cpp:723-760` | ✅ | D82 CLOSED（export 命名: observedOutstandingEstimate / Max） |

### 1.2 T1 integrity の検証

- **A/R counter arithmetic**: `O = signedWide(A) - signedWide(R)`（unsigned wraparound 回避・D82.1 CLOSED）。`observedOutstandingEstimate` / `observedOutstandingMax` という export 命名で authority の `outstanding` / `held_count` と明確に区別（D82.3 CLOSED）。
- **window reset**: `requestMeasurementStart`（CAS Idle→StartRequested）/ `requestMeasurementEnd`（CAS Running→EndRequested）/ `samplerTick`（StartRequested→Running→EndRequested→Closed→Idle）。CAS-based・single owner（sampler）。重複 Start/End は既存 window を変更しない（D91.1 上書き契約）。**window reset 正常動作確認済み**。
- **duplicate / stale snapshot**: `lastClosedSnapshot()` は atomic publish された `MeasurementSnapshot` を読み取り、`valid` フラグで区別。`waitForClosed`（test harness）は `valid != 0` で待機。**duplicate/stale なし**（D91 基準 10）。
- **RT safety**: RT path は acquire observation（atomic fetch_add のみ）・sampler/B_max 更新は Non-RT（timer）。allocation/lock/logging/I/O なし（D83 #8 CLOSED）。
- **T1 実装レビュー**: D83（10 項目突合）= CLOSED。D99（unit test 29/29 PASS）。D100（burst test harness 実済）。

**結論: T1 telemetry に integrity の問題はない**（verdict C ではない）。

## 2. 観測データの収集状況（observation sufficiency）

### 2.1 収集済み window

| 条件 | window 数 | duration | O_w | T_w | E_w | sampleCount | maxSamplingGapUs | missedTick |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal | 1 | ~1.2s（8×150ms） | 3 | 4 | 1 | 21 | 100959 | 0 |
| burst | 1 | ~3s（20×150ms） | 3 | 4 | 1 | 41 | 101132 | 0 |
| jitter | 1 | ~1.1s（10 irregular） | 13 | 13 | 0 | 19 | 101102 | 3 |

- **window 総数**: 3（各条件 1 window）—**sustained observation period ではない**（D100.2）
- **distribution**: normal/burst/jitter 各 1 回ずつ。統計的サンプリング不足。
- **production data**: `evidence/world_retirement_telemetry.json` は**存在しない**（production observation 未実施）。唯一のデータは test harness（`WorldRetirementMeasurementTests.cpp`）の 3 window。

### 2.2 window reset / duplicate / stale

- window reset: CAS-based transition（D91）—正常。
- duplicate/stale: `valid` flag + atomic publish（D91）—正常。
- **問題なし**（T1 integrity は健全）。

## 3. T2 の定義（R_required の導出元）

### 3.1 R_required の形式（D63）

$$R > R_{\text{baseline}} + B_{\max}(T_{\text{reservation\_stall}}) + M$$

| 項目 | 定義 | 現状 |
| --- | --- | --- |
| `R_baseline` | 通常時 outstanding の P99.9（statistical baseline） | **OPEN**（実測不足・通常 ≈ 0〜2 推測のみ） |
| `B_max(T)` | 期間 T 内の最大累積増分（sliding window） | **実装可能**（D64: 8 メトリクスに B_max(T) 含む） |
| `T_reservation_stall` | reservation が解放されない時間（drain stall / reader stall） | **OPEN**（T_deferral=5s と分離済み・D63） |
| `M` | safety margin（policy headroom） | **OPEN（D101 未着手）** |

### 3.2 M の定義（D94.5 / D101.2）

$$M \geq \sup_k \Delta_k^{\text{growth}} + E^{\text{obs}}$$

- $\Delta_k^{\text{growth}} = \max_{t \in [t_k, t_{k+1}]} [B_{\text{true}}(t) - B_{\text{true}}(t_k)]$ — 観測間隔内増加量
- $E^{\text{obs}} = \sup_k [B_{\text{true}}(t_k) - B_{\text{obs}}(t_k)]$ — sampler の underestimation 上界
- **M は測定値ではない** — sampling interval / burst duration / A/R read order / counter wraparound / measurement duration から**数学的に導出**する必要がある（D82.4）

### 3.3 その他の T2 要件

| 要件 | 定義 | 現状 |
| --- | --- | --- |
| `O_w + M` の safety margin | $B_{\max}^{\text{true}} \leq O_w + M$ | **OPEN（D101）** |
| `T_w ≥ O_w` の継続確認 | reference は sampler 以上に観測 | ✅ D100.5 実証済み（E_w ≥ 0） |
| reclaim latency の含め方 | D101.3 #7（delayed release） | **OPEN**（未測定） |

## 4. 観測値と保証値の分離（measured vs guaranteed）

```text
B_max^true           — 数学上の真値（観測不能）
    │
    ├── reference observer
    │       └── T_w = B_max^reference  — 観測値（真値ではない・D100.5）
    └── 100ms sampler
            └── O_w = B_max^observed   — 観測値（sampled maximum）
E_w = T_w - O_w       — 観測値の差（診断のみ・M の根拠にしない）
    ↓
M ≥ sup Δ_k^growth + E^obs  — 数学的安全上界（未導出・D101）
    ↓
R_required = R_baseline + B_max(T_stall) + M  — 最終保証値（未導出）
```

**コードでの分離確認**（`AudioEngine.Commit.cpp:757-760`）:

- `observedOutstandingEstimate` / `observedOutstandingMax` — measured（D82.3 命名で authority と区別）
- `referenceMax`（T_w）— measured（reference observer は観測器・D94）
- `Ew`（E_w = T_w - O_w）— **診断のみ**（コメント: "E_w = T_w - O_w は診断のみ（M の安全側根拠にしない)"）

**結論: 観測値と保証値の分離は正しく実装されている**。E_w を M に誤用していない。

## 5. R の算出可能性（R derivability）

### 5.1 現在の telemetry で R_required を導出できるか

**NO** — 以下の理由:

1. **M が未導出**（D101 未着手）。D94.5/D100.5 は明示的に「M = max(E_w) で終了しない」と禁止している。実測 E_w=1 から M=1 を決定づけることは**禁止**されている。
2. **R_baseline が未確定**（P99.9(N_res)・統計的 baseline・sustained observation 必要）。
3. **reclaim latency 未測定**（D101.3 #7: "delayed release — reclaim latency が outstanding peak を増幅するケース" = OPEN）。
4. **B_max(T) の envelope 未証明**（D82.5: `B_max^true ≤ B_max^observed + M` = OPEN）。

### 5.2 追加 instrumentation の必要性

| 必要な観測 | 現状 | 備考 |
| --- | --- | --- |
| reclaim latency（release イベント → deleter 実行までの時間） | ❌ 未観測 | D101.3 #7（delayed release） |
| sustained window（複数 window / production workload） | ❌ 3 single-window test のみ | D64.4 observation window |
| burst duration / event rate | ⚠️ 部分的（D100 burst test） | D101.3 #4（acquire/increase envelope） |
| sampler gap / missed tick | ✅ 測定中（maxSamplingGapUs / missedTickCount） | D101.3 #3 |
| A/R read order | ✅ signedWide（D82.1） | D101.3 #2 |

### 5.3 μ_burst / burst duration / reclaim latency の bound 充足性

- **μ_burst（burst rate）**: D100 burst test で 20 publish / 150ms interval 実測済みだが、**bound 未証明**（D101.3 #4 OPEN）。
- **burst duration**: D100.5 で「retirement burst の継続時間が sampler interval より短いかが M の支配要因」と指摘されているが、**数学的上界未導出**。
- **reclaim latency**: **完全に未観測**（D101.3 #7 OPEN）。reclaim 遅延が outstanding peak を増幅するケース（delayed release）の bound がない。

## 6. 観測不足の分析（observation insufficiency）

| # | 不足項目 | 現状 | 影響 |
| --- | --- | --- | --- |
| 1 | **window 数不足** | 3 single-window test | 統計的信頼性なし（P99.9 baseline 不可） |
| 2 | **duration 不足** | 各 ~1-3s | sustained observation period 未実施 |
| 3 | **burst 不足** | 1 burst condition（20 publish） | burst envelope の bound 未証明（D101.3 #4） |
| 4 | **jitter 不足** | 1 jitter condition（10 publish） | sampler gap / missed tick の統計不足 |
| 5 | **release/reclaim latency 不足** | 未観測 | D101.3 #7（delayed release）— **M 証明の最大のギャップ** |
| 6 | **production workload 不足** | test harness のみ | production burst / jitter / stall の観測なし |
| 7 | **M の数学的バインド（D101）** | **未着手（セクション不存在）** | **R_required 導出の根本的ブロック** |

## 7. 判定: **B. R = UNDETERMINED**

### 判定理由

1. **T1 telemetry integrity は健全**（D83 CLOSED / D99 29/29 PASS / D100 E_w > 0 実証）。verdict C（telemetry 自体の問題）ではない。
2. **M の数学的バインド（D101）が未着手** — D100.5 は「M の数学的バインドは D101 へ引継ぐ」としているが、**D101 セクションは I4_DESIGN_CONTRACT.md に存在しない**（grep `^## D101` → 0 件）。D82.5 の `B_max^true ≤ B_max^observed + M` は **OPEN（実測方法から証明）** のまま。
3. **観測データ不足** — 3 single-window test runs のみ。production observation 未実施（`world_retirement_telemetry.json` なし）。reclaim latency 完全未観測。
4. **D94.5/D100.5 の禁止**: `M = max(E_w)` は**禁止**。E_w=1 は「O_w ≠ T_w となる実行条件が存在する」の証明にすぎず、M の安全側上界ではない。

### R_required 導出のブロック要因

```text
R_required = R_baseline + B_max(T_stall) + M
                     │              │
                     │              └── D101 未着手（数学的バインド証明）
                     └── P99.9 baseline（sustained observation 不足）
```

**M が未導出であるため、R_required は現在の telemetry から導出不能**。これは telemetry の問題（C）ではなく、**proof obligation の未充足（B）**。

## 8. 推奨アクション（production code 変更なし）

### 即座の次ステップ（D101 実装）

1. **D101 セクションを I4_DESIGN_CONTRACT.md に記述** — 10-step proof（D101.3）:
   - #1 Reference completeness（4-tier）
   - #2 State equation（B = A - R）
   - #3 Sampler gap（G = sup(t_{k+1} - t_k)）
   - #4 Acquire/increase envelope（event rate / burst duration τ_b / μ_burst）
   - #5 Single burst bound
   - #6 Multiple acquire in one interval
   - #7 Delayed release（reclaim latency）
   - #8 Shutdown / quarantine / deferred deletion
   - #9 Finite M proof（sup Δ_k^growth + E^obs < ∞）
   - #10 D102 gate（finite M が証明できれば T2 GO）

2. **reclaim latency の instrumentation** — release event（terminal deleter）から実際の destroy までの時間を測定する T1/D101 instrumentation を追加（D101.3 #7）。

3. **sustained observation periodの実施** — production workload での複数 window 収集（normal / burst / jitter / stall / shutdown）。

### ゲート

- **R_required の確定**: D101 が完了し、有限の M が数学的に証明された後。
- **ReservationExhausted の実生成**: T2（R gate）実装後。
- **Phase I 本実装**: ユーザー最終 GO 後（D70: T1 は測定装置のみ・T2 は authority 機構）。

---

## 付録: 監査のスコープ

- **変更した production code**: なし（本監査は audit-only）
- **ゲート**: `R_required` の確定・`ReservationExhausted` の実生成・Phase I 本実装は行わない
- **参照**: `doc/work88/I4_DESIGN_CONTRACT.md`（D63-D100, D82, D83, D94, D100）, `D2_IMPL_CHECKLIST.md`（Phase I ステータス）, `src/audioengine/ISRWorldRetirementTelemetry.h`, `src/audioengine/ISRWorldRetirementReference.h`, `src/audioengine/AudioEngine.Timer.cpp:435`, `src/audioengine/AudioEngine.Commit.cpp:723-760`, `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp`, `CMakeLists.txt:1682`
