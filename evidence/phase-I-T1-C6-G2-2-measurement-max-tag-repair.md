# Phase I-T1-C6-G2-2 — Measurement Max/Tag Observation Repair

> **Verdict: G2-2 = RESOLVED**
> C6-R1 で発見された G2-2（`observedOutstandingMax` writer 経路が timerCallback 専用で harness から到達不能）を
> Option A（production/test 共通 measurement step）により解消。`setWindowTag` の harness 到達も同時解消。
> production semantics・順序・authority は一切変更せず。**measurement run / A_max / R は引き続き禁止**
> （次は C6 final re-audit で G2=0 を確認してから解禁判定）。

## 1. 実装方針（Option A 採用）

`transferWorldReclaimDeltaForTelemetry()` をさらに大きな共有 measurement step へ拡張:

```cpp
// AudioEngine.Timer.cpp（新設・G2-2）
void AudioEngine::runWorldRetirementMeasurementStep() noexcept
{
    auto& telemetry = worldRetirementTelemetry();
    transferWorldReclaimDeltaForTelemetry();                      // 1. delta transfer（D86 案B）
    const std::int64_t estimate = telemetry.observedOutstandingEstimate(); // 2. D82
    telemetry.updateObservedOutstandingMax(estimate);             // 3. D83.2/D86: max は sampler step のみ
    if (isShutdownInProgress())
        telemetry.setWindowTag(ObservationWindowTag::Shutdown);   // 4. D76.2
    else
        telemetry.setWindowTag(ObservationWindowTag::Normal);
    const auto windowNowUs = convo::getCurrentTimeUs();
    telemetry.samplerTick(windowNowUs);                           // 5. D91: 唯一の transition owner
    if (telemetry.measurementState() == MeasurementState::Running)
        worldRetirementReference_.onMeasurementStart();           // 6. D96/D98: reference 同期
    else
        worldRetirementReference_.onMeasurementEnd();
}
```

- `timerCallback()` は当該 block を削除し `runWorldRetirementMeasurementStep()` 呼び出しに置換（block 外の処理は不変）。
- `driveWorldRetirementSamplerForMeasurement()` は `runWorldRetirementMeasurementStep()` 1 呼び出しに縮退。
- **production 順序契約をそのまま保持**: 1.transfer → 2.estimate → 3.max → 4.tag → 5.samplerTick → 6.reference 同期。samplerTick の max 更新より前からの移動なし。
- **第 2 の authority は作らない**: `observedOutstandingMax_` writer = `Telemetry::updateObservedOutstandingMax()`、`windowTag_` writer = `Telemetry::setWindowTag()` のみ（従来どおり）。harness が Telemetry API を直接呼ぶ構造は禁止どおり排除。

## 2. 静的 writer / call-site census（実装後の全数確認）

| API | caller 数 | 所在 | 判定 |
|---|---|---|---|
| `runWorldRetirementMeasurementStep()` | **2** | Timer.cpp:479（timerCallback）/ AudioEngine.h:4958（harness） | ✅ |
| `updateObservedOutstandingMax()` | **1** | 共有 step 内（Timer.cpp:404） | ✅ |
| `setWindowTag()` | 2 行 | 共有 step 内 Shutdown/Normal 分岐（:407/:409） | ✅ |
| `samplerTick()` | **1** | 共有 step 内（Timer.cpp:415） | ✅ |
| `transferWorldReclaimDeltaForTelemetry()` | **1** | 共有 step 内（Timer.cpp:401） | ✅ |
| `releaseObserved_` fetchAdd | 2 | Telemetry.h のみ（不変） | ✅ |
| `Reference.h` の `onReleaseObserved` | 0 | R3 状態維持 | ✅ |
| harness からの Telemetry mutation 直接呼出 | **0** | harness は `driveWorldRetirementSamplerForMeasurement` 経由のみ | ✅ |

## 3. monotonic / peak regression test（追加）

追加先: `WorldRetirementMeasurementTests.cpp`。実行条件 `--measurement=max`（`all` に組み込み）。

### 設計過程で判明した engine 特性（診断記録）

診断計装（burst 前後の A/R/wc 実測）:

```
afterStabilize A=3  R=2 wc=2
afterBurst     A=19 R=2 wc=18   ← 16 publish 中に破壊 16 件が同期近傍で実行済み
afterStep      R=18 wc=18       ← transfer が一括計上、est = 19-18 = 1
```

本エンジンは NonRT publish の retired world を同期近傍で破壊するため、定常状態で大きな outstanding peak は観測不可（est ≈ 1 が定常値）。これは production の正常挙動であり、peak 値を target にすると背景タイミングと競合する非決定性テストになる。

### 採用した契約ベース設計（全 assert 決定的）

| # | 検証内容 | assert |
|---|---|---|
| 1 | **writer liveness** | 初回 step 前 `max == 0`（唯一 writer は共有 step）→ 初回 step 後 `M1 > M0` かつ `M1 >= est1`。生存 World により `est = A - R >= 1` が常に成立するため 0→≥1 遷移は決定的（背景タイミング非依存） |
| 2 | 単調性契約 | publish→reclaim確定→step を 4 cycle、各 step 後に max 非減少 |
| 3 | 減非反応 | drain 完了後（est 低減済み）の追加 step で `MFinal == MPrev`（不変） |
| 4 | windowTag 経路 | headless 非 shutdown で `windowTag == Normal` |

`windowMax`（window-local）との cross-assertion は意図的に存在しない（§6 指示どおり別 semantic object）。

### 実行結果

```
[max-monotonic] M0(before first step)=0 -> M1=1 est1=1
[max-monotonic] afterCycles=1 estDrained=1 MFinal=1 tag=0
PASS
```

## 4. G2-1 regression（--measurement=release）

```
[g2-single] N=1 dAcquire=1 dReclaim=1 dReference=1 dRelease=1 outstanding=1
[g2-multi]  N=4 dAcquire=4 dReclaim=4 dReference=4 dRelease=4 outstanding=1
[no-double-transfer] totalReleaseDelta=2 totalReclaimDelta=2
```

**3/3 PASS** — `Δreclaim == Δreference == Δrelease == N`、second sampler +0、`outstanding = A - R` を維持。

## 5. normal / burst / jitter regression（--measurement=all）

```
[normal] O_w=1 T_w=2 E_w=1 sampleCount=21 gap=101118 missed=0 wrapped=0  ✅ (T_w >= O_w)
[burst]  O_w=2 T_w=2 E_w=0 sampleCount=46 gap=111665 missed=0 wrapped=0  ✅
[jitter] O_w=1 T_w=2 E_w=1 sampleCount=22 gap=114055 missed=0 wrapped=0  ✅
[release] 3/3 ✅   [max-monotonic] ✅
```

**7/7 PASS** — Closed snapshot / O_w / T_w / E_w / sampleCount / maxSamplingGapUs / missedTickCount / counterWrapped の既存検証はすべて成立。

## 6. full CTest

```
97% tests passed, 1 tests failed out of 32
失敗: #28 HeadlessAudioPathVerification のみ（前回 G2-1 と同一）
```

#28 の再分類（「既知だから無視」ではなくコード共有の有無で判定）:

- 失敗経路: `.github/scripts/cli-smoke-test.ps1` → `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` 起動（**icx ツールチェイン・別ビルドツリー**。Debug(MSVC) pipeline はこの成果物を rebuild しない）。
- 今回の変更（`runWorldRetirementMeasurementStep` 共有化）が触るのは AudioEngine.Timer.cpp / AudioEngine.h / WorldRetirementMeasurementTests.cpp。CLI smoke が起動する icx Release バイナリは今回の変更を含まない古い成果物であり、かつ T1 measurement step は CLI smoke の検査対象（`[CLI_PERF_RAW] callbacks` ログ）とコード共有がない。
- 失敗モードは G2-1 実施前から同一（static-teardown 0xC0000005 自体は許容され、positive callback count の log 欠落で throw）。
- 結論: **本変更とコード共有ゼロの既存環境問題**として分類。retire/publish/measurement 系 31 テストは全 PASS。

## 7. G2-2 解消条件の検証

| 条件 | 必須 | 判定 |
|---|---|---|
| harness が `updateObservedOutstandingMax()` に到達 | ✅ | PASS（初回 step で M0=0→M1=1 を決定的実測） |
| production/test が同一 measurement step | ✅ | PASS（call site 2 = timerCallback + harness、実装 1 件） |
| max の direct harness writer が存在しない | ✅ | PASS（harness の Telemetry mutation 直接呼出 0・writer は Telemetry メソッドのみ） |
| max monotonic assertion | ✅ | PASS（4 cycle + drain 追加 step で非減少/不変） |
| deterministic peak test | ✅ | PASS（契約ベース: 0→≥1 遷移・est1 下限・減非反応。peak 値の環境依存性は §3 のとおり診断記録し target 値 assert は採用しない） |
| G2-1 release regression PASS | ✅ | PASS（3/3） |
| `observedOutstanding = A - R` PASS | ✅ | PASS（g2 全条件で恒等式確認） |
| normal/burst/jitter regression PASS | ✅ | PASS（7/7） |
| production semantics の順序変更なし | ✅ | PASS（順序契約 1→6 を共有 step へそのまま移設・timerCallback block 外は無変更） |

```
G2-2 = RESOLVED
```

## 8. Tool Coverage

| 系統 | ツール | 用途 |
|---|---|---|
| Sandbox | node fs census ×2 回（実装直後・最終） | writer/call-site 全数確認 |
| Build | MSVC 18.9.1 + oneAPI + CMake/Ninja Debug | AudioEngineHarness ほか |
| Test | --measurement=max / release / all、publish pipeline デフォルト、ctest 32 件 | 動的検証 |
| WSL | rg / sg / fdfind / ag / fzf / sed / awk | census 補助 |
| MCP/CLI | serena / AiDex / semble / ccc / graphify | C2〜C6-R1 で確定済みシンボル参照との整合維持 |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 前 Gate までに 9 件 200 OK 済 |

## 9. 進行状態（次ゲート）

```
G2-2 RESOLVED（本文書）
        ↓
C6 final re-audit（G2 = 0 の確認・新規 G2 有無の再走査）
        ├─ NO  → C6 FAIL 継続
        └─ YES → T1-C6 = PASS（Residual Gap = G1 を明示）
                    ↓
              T1 measurement run 解禁（normal/burst/jitter → O_w/T_w/E_w → A_max candidate 記録）
                    ↓
              R = UNDETERMINED のまま（M-bound / D101 は未着手）
```

- 実施しないもの: A_max candidate 記録 / R / R_cap / M 決定 / T2 / Recovery coalesce / JSON export 改善 / cross-window characterization / shutdown architecture。

---

*Evidence generated: Phase I-T1-C6-G2-2 — shared measurement step により max/tag writer 経路を harness 到達可能化。C6 final re-audit 待ち。*
