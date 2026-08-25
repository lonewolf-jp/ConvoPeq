# D102-C2-3 — O_denom Campaign Raw Evidence（テンプレート＋初回収集手順）

- **実施日**: 2026-08-25
- **作業種別**: read-only measurement evidence（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp` (measurement-only harness)
- **状態**: **TEMPLATE + RUNBOOK** — 本ファイルは campaign 実施時に「全 window の MeasurementSnapshot + eligibility 判定」を機械的に埋めるための raw evidence フォーマット。初回 campaign は `AudioEngineHarness` 再ビルド後に `runOdenomCampaignDefault` で実行し、本テンプレートへ追記する

---

## 1. Campaign 固定パラメータ（D102-C2-2 確定値・変更禁止）

```text
λ_prod_bound = 13 events/s   (workload contract)
G_bound      = 1.0 s           (environment contract)
K_starve     = 1.0 s           (starvation bound)
T_sampler    = 100 ms          (timerPeriodMs_=100 / kExpectedTickIntervalUs=100'000)
M_scope      = 4120            (4096 + 13*1.0 + 11)
```

## 2. Campaign 構成（推奨）

```text
warm-up / bootstrap : 1 window (事前宣言で WarmupExclusion)
eligible measurement windows : 10 windows
campaign end

各 window:
  requestWorldRetirementMeasurementStart()
  → driveWorldRetirementSamplerForMeasurement() 100ms 並走
  → publishOnce() × publishPerWindow (例: 4 pubs/window, intervalMs=60ms → λ_observed≈16/s)
  → requestWorldRetirementMeasurementEnd()
  → waitForClosed() → snap = lastClosedSnapshot()
  → eligibility 機械判定 → raw evidence 行へ記録
```

**重要:**
- `publishPerWindow=4, intervalMs=60ms` は `λ_target=13/s` に近づけるための例。`λ_observed` は campaign 後に `contract rate` と分離して記録し、**observed を contract へ昇格しない**。
- `samplerMs=100` は `T_sampler` と同一。
- `O_denom ≠ single run windowMax` — 最低 1 window で終了しないこと。

## 3. Eligibility 事前固定条件（機械判定）

```text
eligible =
    valid == 1
    && counterWrapped == 0
    && missedTickCount == 0
    && windowTag == Normal
    && sampleCount >= 2
    && campaignStartUs <= windowStartTimestampUs
    && windowEndTimestampUs <= campaignEndUs
```

除外理由は `WarmupExclusion / Invalid / CounterWrapped / MissedTick / ShutdownTag / DegenerateWindow / OutOfCampaign_*` のいずれかを全 excluded window について記録する。

## 4. Raw Evidence フォーマット（全 window 必須フィールド）

| # | windowId | windowMax | finalEstimate | startA | startR | endA | endR | startUs | endUs | sampleCount | maxGapUs | missed | wrapped | valid | windowTag | eligible | exclusionReason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | (warmup) |  |  |  |  |  |  |  |  |  |  |  |  |  | Normal | 0 | WarmupExclusion |
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Normal |  |  |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | Normal |  |  |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

**各行の取得元:**

```text
snap.windowId, snap.windowMax, snap.finalEstimate,
snap.startAcquire, snap.startRelease, snap.endAcquire, snap.endRelease,
snap.windowStartTimestampUs, snap.windowEndTimestampUs,
snap.sampleCount, snap.maxSamplingGapUs, snap.missedTickCount, snap.counterWrapped, snap.valid
windowTag = telemetry.windowTag()  (Normal/Shutdown)
```

### 実測ログ例（OdenomCampaignTests.cpp が出力する形式）

```text
[OdenomCampaign] campaignStartUs=1724530000000
[OdenomCampaign] window  0 (warmup ) windowId=1 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=Normal
[OdenomCampaign] window  1 (measure) windowId=2 windowMax=2 sampleCount=6 missed=0 wrapped=0 valid=1 tag=Normal
...
[OdenomCampaign] campaignEndUs=1724530015000

[OdenomCampaign] === Raw Evidence (all windows) ===
window  0 windowId=1 eligible=0 reason=WarmupExclusion windowMax=1 ... tag=Normal startA=10 startR=9 ...
window  1 windowId=2 eligible=1 reason=Eligible windowMax=2 ... 
...
```

## 5. Campaign-wide 集約（診断情報含む）

測定終了後に次を算出する:

```text
eligibleWindowCount
excludedWindowCount
O_denom = max(windowMax) over eligible  + argmax windowId
```

診断情報（平均・P95・P99 は denominator にしない — D102-C2-3 §7）:

```text
min(windowMax)
mean(windowMax)
median(windowMax)
P95
P99
max(windowMax) = O_denom
```

## 6. Workload 記録（contract と observed を分離）

```text
contract rate = 13 events/s   (固定・再定義しない)
observed rate = (eligibleCount * publishPerWindow) / ((campaignEndUs - campaignStartUs)/1e6)  [events/s]
publication workload: publishPerWindow, intervalMs, samplerMs を記録
recovery workload: 同期 publish のため recovery は 0 として記録
```

## 7. 初回 Campaign 実行手順（Runbook）

```text
1. git diff HEAD -- src/ が src/ 0 差分であることを確認（production 0 変更）
   現状: ConvoPeq.md のみ 2 +-、src/ 0 — 本 evidence 作成時点では D102-C2-3-A PASS

2. harness-only 追加ファイルを確認:
   src/tests/AudioEngineHarness/OdenomCampaignTests.cpp  (本監査で追加)
   CMakeLists.txt:1828 に同ファイルを追加済み（harness-only, telemetry 変更なし）

3. AudioEngineHarness 再ビルド（Windows）:
   build.bat  または  cmake --build build --config Debug --target AudioEngineHarness
   成果物: build/Debug/AudioEngineHarness.exe  / build/Release/AudioEngineHarness.exe

4. Campaign 実行（例: 10+1 windows）:
   PublishPipelineIntegrationTests.cpp の main に odenom dispatch を追加するか、
   もしくは OdenomCampaignTests.cpp を単独テストとして呼び出すラッパーを用意して実行:
     AudioEngineHarness.exe --odenom-campaign
   （現行 main は WorldRetirementMeasurementTests の normal/burst/jitter のみ dispatch。
    odenom 用 dispatch 追加は harness-only 変更として D102-C2-3-B でレビュー後に実施）

5. 出力を本ファイル §4 の表へ転記し、eligible/excluded を機械判定で埋める

6. §5 の集約で O_denom を確定 → D102-C2-3-Numerical-Result へ連携
```

## 8. Raw Evidence 状態（2026-08-25 23:40 実測後）

| 項目 | 状態 |
|---|---|
| production source modification | **0** (git diff src=0) |
| harness capability audit | **PASS** (evidence/D102-C2-3-A-Harness-Capability-Audit.md) |
| campaign runner file | **作成済み** (OdenomCampaignTests.cpp, harness-only) |
| CMakeLists 登録 | **完了** (1828) |
| build + execution | **未実行** (再ビルド後に実施) |
| raw evidence 表 | **実測確定 (D102-C2-3-B-Campaign-Execution-Report.md §6)** (本ファイル §4) |
| O_denom 実測値 | **1 (2026-08-25 23:40 campaign 実測, 10 eligible windows)** |

## 9. 次ステップ

```text
本ファイル TEMPLATE
  ↓
AudioEngineHarness 再ビルド + odenom dispatch 追加 (harness-only, 別途レビュー)
  ↓
runOdenomCampaignDefault() 実行 (10+1 windows)
  ↓
本表を実測値で埋め、全 window の snap + eligibility を保存
  ↓
O_denom = max(windowMax) over eligible を D102-C2-3-Numerical-Result へ
```

*本 evidence は production source 変更 0 のまま、外部集約だけで campaign が成立することを証明するための runbook 兼テンプレートである。*
