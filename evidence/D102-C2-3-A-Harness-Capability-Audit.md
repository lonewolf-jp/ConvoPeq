# D102-C2-3-A — Existing Harness Capability Audit（O_denom Campaign Read-only 可否判定）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（production source 変更 **0** / telemetry 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + 実 `src/`（`git diff HEAD -- src/` = 0）
- **判定**: **PASS（外部集約のみで成立）** — 既存 `WorldRetirementMeasurementTests.cpp` は単一 window 取得まで、campaign-wide 集約と eligibility 自動判定は未実装だが、**telemetry 本体を変更せず harness 側の外部ループで成立する**ことを確認。D102-C2-3-B（telemetry 変更）は不要、D102-C2-3-A のまま read-only campaign 実施可能

---

## 1. 監査対象（D102-C2-3 §1 指示どおり）

| 対象 | ファイル・シンボル | 状態 |
|---|---|---|
| `WorldRetirementMeasurementTests.cpp` | `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp` (41KB, 800+ lines) | ✅ 存在・read-only 確認 |
| `ISRWorldRetirementTelemetry` | `src/audioengine/ISRWorldRetirementTelemetry.h` (17.6KB) | ✅ |
| `AudioEngineHarness` | `src/tests/AudioEngineHarness/AudioEngineHarness.h/.cpp` | ✅ |
| `requestWorldRetirementMeasurementStart()` | `src/audioengine/AudioEngine.h:4985` → `worldRetirementTelemetry_.requestMeasurementStart()` | ✅ |
| `requestWorldRetirementMeasurementEnd()` | `src/audioengine/AudioEngine.h:4989` → `requestMeasurementEnd()` | ✅ |
| `driveWorldRetirementSamplerForMeasurement()` | `src/audioengine/AudioEngine.h:4997` → `runWorldRetirementMeasurementStep()` | ✅ |
| `MeasurementSnapshot` | `src/audioengine/ISRWorldRetirementTelemetry.h:42` | ✅ |

### ツール別棚卸し

| ツール | 実施内容 | 結果 |
|---|---|---|
| `rg`/`ag`/`ast-grep`/`fdfind`/`fzf`/`sed`/`awk` (WSL RTK) | `kQueueSize`/`kMaxQuarantinedEntries`/`windowMax`/`requestWorldRetirement*`/`driveWorldRetirement*`/`MeasurementSnapshot` 全検索 | D/Q/E 容量・telemetry フィールド・harness 単一 window 設計を全確認 |
| `AiDex` `aidex_query` | `requestWorldRetirementMeasurementStart` 3 matches, `driveWorldRetirementSamplerForMeasurement` 17 matches, `windowMax` 13 matches | ソース上の定義位置特定 |
| `AiDex` `aidex_signature` | `AudioEngine.h` 216 methods / `ISRWorldRetirementTelemetry.h` 15 methods | harness 経路特定 |
| `serena` | `initial_instructions` timeout → `rg`/`AiDex` で代替 | 同等カバレッジ確保 |
| `cocoindex`/`semble`/`graphify` | WSL `which` でバイナリ不在確認 | `rg`/`AiDex` で代替 |

---

## 2. MeasurementSnapshot 取得可能フィールド（D102-C2-3 §1）

```text
src/audioengine/ISRWorldRetirementTelemetry.h:42-57
struct MeasurementSnapshot {
    uint64_t windowId
    uint64_t startAcquire
    uint64_t startRelease
    uint64_t endAcquire
    uint64_t endRelease
    int64_t  finalEstimate
    int64_t  windowMax                // bounded sampled maximum（D91 基準 8）
    uint64_t windowStartTimestampUs
    uint64_t windowEndTimestampUs
    uint64_t sampleCount
    uint64_t maxSamplingGapUs
    uint64_t missedTickCount
    uint64_t counterWrapped           // 診断のみ（D91 基準 9）
    uint64_t valid
}
```

追加取得経路:

```text
telemetry.windowTag()  → ObservationWindowTag::Normal / Shutdown  (D76.2)
  src/audioengine/ISRWorldRetirementTelemetry.h:149
  enum ObservationWindowTag { Normal, Shutdown }  line 11

telemetry.windowMax_ / snapshot_.valid は lastClosedSnapshot() で取得
  src/audioengine/ISRWorldRetirementTelemetry.h:194 lastClosedSnapshot()
```

**結論: D102-C2-3 §1 が列挙する 13 項目（windowMax/sampleCount/maxSamplingGapUs/missedTickCount/counterWrapped/valid/windowId/start/end A/R/window start/end timestamp + windowTag）は全て取得可能 ✅ — telemetry 本体変更は不要と判断してよい ✅**

---

## 3. 既存 eligibility を機械的に適用できるか（D102-C2-3 §2）

### 3.1 判定式（事前固定）

```text
eligible =
    valid == 1
    && counterWrapped == 0
    && missedTickCount == 0
    && windowTag == Normal
    && sampleCount >= 2
    && campaignStart <= windowStartTimestampUs
    && windowEndTimestampUs <= campaignEnd
```

### 3.2 各述語の機械判定可否

| 述語 | フィールド | 判定可否 | 根拠 |
|---|---|---|---|
| `valid == 1` | `snap.valid` | ✅ | `ISRWorldRetirementTelemetry.h:56` / `closeWindow` で `snap.valid=1` publish |
| `counterWrapped == 0` | `snap.counterWrapped` | ✅ | D91 基準 9・wraparound 診断、`ag` で `counterWrapped` 2 箇所 FAIL check 確認 |
| `missedTickCount == 0` | `snap.missedTickCount` | ✅ | `kExpectedTickIntervalUs=100'000` 超過検出、`fetchAddAtomic` で加算 |
| `windowTag == Normal` | `telemetry.windowTag()` | ✅ | `setWindowTag(Shutdown/Normal)` は `runWorldRetirementMeasurementStep:406-409` で毎 tick 更新 |
| `sampleCount >= 2` | `snap.sampleCount` | ✅ | `beginWindow` で 1、`sampleWindow` で `fetchAdd`、`closeWindow` でもカウント。`sampleCount=1` は degenerate（Start 直後 End） |
| 時間境界 | `snap.windowStart/EndTimestampUs` | ✅ | `beginWindow`/`closeWindow` で `nowTimestampUs` を publish |

**全述語が `MeasurementSnapshot` + `windowTag()` の読み取りのみで機械判定可能 ✅**

### 3.3 `sampleCount >= 2` の意味（D102-C2-3 §2 補足）

```text
beginWindow():  windowMax = firstEstimate, sampleCount = 1  (D91 監視項目 1)
sampleWindow(): estimate → updateWindowMax, sampleCount++   (D91 基準 6)
closeWindow():  finalEstimate → updateWindowMax, snapshot publish (D91 監視項目 2)
```

∴ `sampleCount == 1` は Start 直後 End の degenerate window。`>=2` は少なくとも 1 回の `sampleWindow` を経たことを保証。

### 3.4 事後選択の禁止

> **事後的に「値が大きかった window だけ採用」は禁止**

本判定式は **campaign 開始前に固定**し、機械的に適用する。`valid/counterWrapped/missedTickCount/windowTag/sampleCount/時間境界` は `windowMax` の大小と無関係な診断フィールドであるため、恣意排除が成立する。

---

## 4. 既存 normal/burst/jitter を O_denom としない（D102-C2-3 §3）

### 4.1 現行 harness の設計

```text
WorldRetirementMeasurementTests.cpp:63
  runMeasurement(condition, harness, publishCount, intervalMs, samplerIntervalMs)
    Start → publish 反復 → End → Closed snapshot 1 個 → O_w/T_w/E_w 記録

  testNormalMeasurement():  publishCount=8,  intervalMs=150, samplerIntervalMs=100
  testBurstMeasurement():   publishCount=20, intervalMs=150, samplerIntervalMs=100
  testJitterMeasurement():  intervals=[0,80,200,0,120,300,0,60,180,0] 不規則

  各 runMeasurement は 1 回の Start→End で 1 Closed snapshot のみ取得
```

### 4.2 なぜ O_denom でないか

| 観点 | 既存 normal/burst | D102-C2-2 要求 O_denom |
|---|---|---|
| window 数 | **1** Closed snapshot / run | **複数** eligible windows の `max(windowMax)` |
| 目的 | `E_w = T_w - O_w` の sampling model 評価（M の安全側上界ではないことを確認） | `O_denom` の campaign-wide maximum（`K_min/R_required` の denominator） |
| workload | publishCount 8/20, interval 150ms（sampler 100ms より長い — peak miss を意図的に生成） | `λ_target = 13 events/s` に近づけた連続 publish（peak 保守性と別目的） |
| 集約 | なし（単一 windowMax） | `O_denom = max(windowMax(w))` over all eligible w |

### 4.3 検証

```text
rg: O_denom / campaign.*max / max.*windowMax / eligible.*window → src/tests/ で 0 hits
  → campaign-wide 集約は未実装であることを確認

rg: counterWrapped/missedTickCount/ObservationWindowTag →
  WorldRetirementMeasurementTests.cpp で counterWrapped の FAIL check のみ（261,447行目）
  → eligibility 自動判定は未実装、診断出力のみ
```

**結論: 既存 normal/burst は参考観測としてのみ扱い、`O_denom ≠ single run windowMax` を厳守 ✅**

---

## 5. 既存 harness だけで campaign 条件を満たせるか（D102-C2-3 §4-7）

### 5.1 campaign 仕様（固定値）

```text
λ_prod_bound = 13 events/s
G_bound      = 1.0 s
K_starve     = 1.0 s
T_sampler    = 100 ms
M_scope      = 4120
→ 変更しない（D102-C2-2 確定値）
```

### 5.2 campaign 集約（外部で実施）

```text
O_denom = max(windowMax(w)) over all eligible windows w
最低 1 window では終了しない
```

### 5.3 既存 harness の単一 window 機構を複数回ループで再利用可能か

| 必要機能 | 既存実装 | 複数 window への拡張 |
|---|---|---|
| `Start → Running → End → Closed → Idle` 状態遷移 | `requestMeasurementStart()` / `samplerTick()` / `requestMeasurementEnd()` / `lastClosedSnapshot()` で完結（`MeasurementState` 単調遷移 D91.1） | ✅ `Idle` へ戻った後、再び `Start` で次 window 開始可能。`windowId` は `beginWindow` で `+1` され、複数 window の識別が可能 |
| `driveWorldRetirementSamplerForMeasurement()` | `AudioEngine.h:4997` → `runWorldRetirementMeasurementStep()`（production と同一 step） | ✅ harness 側で独立スレッドまたはループで 100ms カデンス駆動可能（`WorldRetirementMeasurementTests.cpp:103` の `jthread` パターン参照） |
| `MeasurementSnapshot` の immutable publish | `lastClosedSnapshot()` は window state を変更しない（D91 基準 10） | ✅ 取得後に次 Start を発行しても前 snapshot は保持される |
| workload 駆動 | `publishOnce()` + `driveWorldRetirementReclaimForMeasurement()` | ✅ 同一 harness 内で複数回呼び出し可能 |

**判定: 単一 window 取得機構を外部ループで複数回呼び出すことで campaign 構成可能 ✅**

### 5.4 不足しているもの（harness 側でのみ補完可能）

| 不足 | 補完方法 | production 変更要否 |
|---|---|---|
| campaign-wide `max(windowMax)` 集約 | harness 側で `std::vector<MeasurementSnapshot>` に全 Closed snapshot を蓄積し `max_element` | ❌ production 変更不要 |
| eligibility 自動判定 | harness 側で §3 の判定式を機械適用し `eligible/excluded` に分類 | ❌ production 変更不要 |
| `excludedWindowCount + reason` 全件記録 | 上記判定の `false` 理由を列挙 | ❌ production 変更不要 |
| 統計診断（min/mean/median/P95/P99/max） | eligible `windowMax` 配列から算出（D102-C2-3 §7 診断情報） | ❌ production 変更不要 |
| 複数 window の連続実行 | `Start → publish loop → End → waitForClosed → 収集 → 次 Start` をループ | ❌ production 変更不要（既存 `runMeasurement` を複数回呼ぶか、新 `OdenomCampaignTests.cpp` を追加） |

### 5.5 構成案（production 変更 0）

```text
warm-up / bootstrap  (1 window, 事前宣言で excluded)
        ↓
eligible measurement windows  (N windows, 例: 10-20 windows)
        ↓
campaign end

各 window:
  requestStart → driveSampler(100ms) 並走 → publish loop(λ_target≈13/s) → requestEnd → waitForClosed → snap 収集
```

---

## 6. production source modification = 0 の証明

```text
git diff HEAD -- src/audioengine/ISRWorldRetirementTelemetry.h  → 0 lines
git diff HEAD -- src/audioengine/AudioEngine.Timer.cpp         → 0 lines
git diff HEAD -- src/audioengine/AudioEngine.h                 → 0 lines
git diff HEAD -- src/                                          → 0 files (ConvoPeq.md のみ 2 +- タイムスタンプ差分)
```

∴ **D102-C2-3-A の前提（production source = 変更 0）は満たされている ✅**

---

## 7. 結論：D102-C2-3-A → read-only campaign 実施可能

| ゲート | 判定 |
|---|---|
| G_A1 | 22:11:33 ソースと実 `src/` が一致 ✅ |
| G_A2 | production source modification = 0 ✅ |
| G_A3 | telemetry 本体変更は不要（全フィールド取得可能） ✅ |
| G_A4 | eligibility は外部で機械判定可能 ✅ |
| G_A5 | campaign 集約は外部ループで実現可能 ✅ |
| G_A6 | 既存 normal/burst を O_denom としない区分は明確 ✅ |
| G_A7 | 複数 window 構成が既存 harness 機構で実現可能 ✅ |

### 次工程

```text
D102-C2-3-A PASS（本監査）
  ↓
D102-C2-3-A のまま read-only campaign 実施
  （harness-only 新規 campaign runner を追加、telemetry 変更なし）
  ↓
全 window の MeasurementSnapshot + eligibility 判定を raw evidence 保存
  ↓
O_denom = max(windowMax) over eligible → K_min/R_required 算出
```

**D102-C2-3-B（telemetry 変更提案）は不要。** ただし campaign runner 自体は harness-only の新規ファイル（例: `OdenomCampaignTests.cpp`）として追加する必要があり、これは **measurement-only harness modification** に該当するが、production runtime の semantic には影響しない。D102-C2-3 指示の順序どおり、まず read-only campaign を実施し、O_denom を確定する。

---

## 8. 参照

- `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp:63-300`（単一 window 設計）
- `src/audioengine/ISRWorldRetirementTelemetry.h:42-57,149,194,209-285`（Snapshot/eligibility フィールド）
- `src/audioengine/AudioEngine.h:4985-5030`（Start/End/drive）
- `src/audioengine/AudioEngine.Timer.cpp:372-415`（production/test 共通 measurement step）
- `evidence/D102-C2-2-Numerical-Application-Measurement-Campaign.md`（M_scope=4120 確定）

*本監査は read-only であり、ソースコード変更 0、契約変更 0、数値の恣意的採用 0 で作成された。*
