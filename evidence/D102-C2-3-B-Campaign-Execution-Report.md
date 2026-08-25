# D102-C2-3-B/C — O_denom Campaign 実行報告（production source 0-diff / Debug 再ビルド / warmup1+10 実測）

- **実施日**: 2026-08-25 23:40 (JST)
- **作業種別**: read-only campaign execution（production source 変更 **0** / harness-only 追加のみ）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + 実 `src/`
- **Build**: `AudioEngineHarness.exe` **Debug** 60,021,248 bytes, 2026-08-25 23:40:41, `cmake --build build --config Debug --target AudioEngineHarness` PASS

---

## 0. 固定値（変更禁止・再確認）

```text
λ_prod_bound = 13 events/s
G_bound      = 1.0 s
K_starve     = 1.0 s
T_sampler    = 100 ms (kExpectedTickIntervalUs=100'000)
M_scope      = 4120 (=4096+13*1.0+11)
R_cap,bounded = 5120 (=4096(D)+512(Q)+512(E))
```

`src/DeferredDeletionQueue.h:262 kQueueSize=4096`, `src/audioengine/RetireQuarantineStore.h:65 kMax=512`, `src/audioengine/ISRWorldRetirementTelemetry.h:311 kExpectedTickIntervalUs=100'000` 全一致

## 1. Campaign runner 実行可能性検証（D102-C2-3 §1）

| 項目 | 状態 |
|---|---|
| `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp` | 存在 (19KB), `runOdenomCampaignDefault()` 定義, `publishPerWindow=4` `intervalMs=60` `samplerMs=100` `totalWindows=10` `warmupWindows=1` 構成可能 ✅ |
| `CMakeLists.txt:1829` | `OdenomCampaignTests.cpp` 追加済み ✅ |
| dispatch `--odenom-campaign` | **当初未 dispatch**を確認 → **harness-only 修正**として `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp:357,365,435` に `odenomCampaign` 追加（production `src/audioengine/` 変更 0） ✅ |
| 検証順序 | 現状 build/dispatch 検証 → 不足を harness-only ゲートで実施 → の順序を遵守 ✅ |

## 2. Production 0-diff gate（campaign 直前）

```text
=== git diff HEAD -- src/audioengine (production) ===
0 lines

=== git diff HEAD --stat -- src/ ===
src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp | 21 +++++  (harness-only)

=== git status --short ===
 M CMakeLists.txt
 M ConvoPeq.md (timestamp 1 line)
 M src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
?? src/tests/AudioEngineHarness/OdenomCampaignTests.cpp

=== git diff --stat HEAD ===
CMakeLists.txt 1+
ConvoPeq.md 2+-
PublishPipelineIntegrationTests.cpp 21+
```

- **production `src/audioengine/` 差分 0** — gate **PASS**（harness-only 変更のみ）
- `src/DeferredDeletionQueue.h` / `RetireQuarantineStore.h` / `ISRWorldRetirementTelemetry.h` / `AudioEngine.*` 全て 0 diff

## 3. Debug 再ビルド

```text
cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl  → PASS (38s)
cmake --build build --config Debug --target AudioEngineHarness → PASS
  [144/145] Linking CXX executable Debug\AudioEngineHarness.exe
  60,021,248 bytes @ 2026-08-25 23:40:41
```

- `build configuration = Debug` / `target = AudioEngineHarness` / `build result = PASS` — raw evidence に記録 ✅

## 4. Campaign 実行（固定構成）

```text
.\build\Debug\AudioEngineHarness.exe --odenom-campaign

構成:
  Window 0: warm-up/bootstrap → excluded
  Window 1..10: measurement → eligible判定対象
  各 measurement window:
    requestWorldRetirementMeasurementStart()
    sampler 100ms 並走 (jthread)
    publish ×4, interval 60ms
    requestWorldRetirementMeasurementEnd()
    wait Closed → lastClosedSnapshot()
```

**実行ログ保存:** `evidence/OdenomCampaign_console_2026-08-25.log`

## 5. Campaign Raw Output（原文）

```text
[OdenomCampaign] campaignStartUs=104614058528
[OdenomCampaign] window  0 (warmup ) windowId=1 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  1 (measure) windowId=2 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  2 (measure) windowId=3 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  3 (measure) windowId=4 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  4 (measure) windowId=5 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  5 (measure) windowId=6 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  6 (measure) windowId=7 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  7 (measure) windowId=8 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  8 (measure) windowId=9 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window  9 (measure) windowId=10 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] window 10 (measure) windowId=11 windowMax=1 sampleCount=5 missed=0 wrapped=0 valid=1 tag=normal
[OdenomCampaign] campaignEndUs=104623794884
```

## 6. 全 Closed Window Raw Evidence（eligibility 機械適用）

| # | windowId | windowMax | finalEst | startA | startR | endA | endR | startUs | endUs | sampleCount | maxGapUs | missed | wrapped | valid | windowTag | eligible | exclusionReason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | 1 | 1 | 1 | 3 | 2 | 7 | 6 | 104614062257 | 104614574176 | 5 | 111728 | 0 | 0 | 1 | Normal | 0 | WarmupExclusion |
| 1 | 2 | 1 | 1 | 7 | 6 | 11 | 10 | 104614936861 | 104615449431 | 5 | 110971 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 2 | 3 | 1 | 1 | 11 | 10 | 15 | 14 | 104615832006 | 104616344166 | 5 | 112108 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 3 | 4 | 1 | 1 | 15 | 14 | 19 | 18 | 104616725494 | 104617238009 | 5 | 112469 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 4 | 5 | 1 | 1 | 19 | 18 | 23 | 22 | 104617593647 | 104618106442 | 5 | 111703 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 5 | 6 | 1 | 1 | 23 | 22 | 27 | 26 | 104618491059 | 104619003560 | 5 | 112023 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 6 | 7 | 1 | 1 | 27 | 26 | 31 | 30 | 104619384848 | 104619896713 | 5 | 111323 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 7 | 8 | 1 | 1 | 31 | 30 | 35 | 34 | 104620249292 | 104620762160 | 5 | 111687 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 8 | 9 | 1 | 1 | 35 | 34 | 39 | 38 | 104621143386 | 104621655071 | 5 | 110709 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 9 | 10 | 1 | 1 | 39 | 38 | 43 | 42 | 104622035367 | 104622547675 | 5 | 111675 | 0 | 0 | 1 | Normal | 1 | Eligible |
| 10 | 11 | 1 | 1 | 43 | 42 | 47 | 46 | 104622900283 | 104623412808 | 5 | 110914 | 0 | 0 | 1 | Normal | 1 | Eligible |

**Eligibility 判定式（事前固定・変更なし）:**

```text
eligible = valid==1 && counterWrapped==0 && missedTickCount==0
           && windowTag==Normal && sampleCount>=2
           && campaignStartUs<=windowStart && windowEnd<=campaignEndUs
```

- **windowMax の大小を理由に除外していない**（post-selection 禁止を遵守）
- excluded も全件保存（window 0 WarmupExclusion）

### Campaign-wide 集約

```text
eligibleWindowCount = 10
excludedWindowCount = 1
O_denom = max(windowMax over eligible) = 1
argmax windowId = 2 (全 eligible が 1 のため先頭)

診断 (denominator にしない):
  min=1 mean=1.00 median=1.00 P95=1 P99=1 max=1
```

## 7. Workload 分離記録

```text
contract rate = 13 events/s (固定・再定義しない)
observed rate = (eligibleCount * publishPerWindow) / ((campaignEndUs - campaignStartUs)/1e6)
              = (10 * 4) / (9736356 / 1e6)
              = 40 / 9.736356
              = 4.11 events/s

publication workload: publishPerWindow=4, intervalMs=60, samplerMs=100
recovery workload: 0 (同期 publish のため)
sampler cadence: 100ms (kExpectedTickIntervalUs=100'000)
starvation condition: なし (K_starve=1.0s 未到達)
```

- **observed 4.11/s を contract 13/s の根拠として再定義していない** — 別々に記録 ✅

## 8. O_denom 確定後の即時計算

```text
O_denom    = 1 (実測 campaign-wide maximum)
K_min      = ceil(4120 / 1) = 4120
R_required = 1 + 4120 = 4121
R_cap,bounded = 5120
Terminal dependency = max(0, 4121-5120) = 0
```

```text
O>=1 ⇒ K_min<=4120 ⇒ R_required<=4121 < 5120 ⇒ Terminal dependency 0
```

## 9. PASS/FAIL 判定順序（G3-G15）

| Gate | 条件 | 結果 |
|---|---|---|
| G3 | campaign start/end recorded | **PASS** 104614058528 → 104623794884 |
| G4 | all Closed windows recorded | **PASS** 11 windows |
| G5 | eligibility mechanically applied | **PASS** |
| G6 | all exclusions + reasons recorded | **PASS** 1 excluded (WarmupExclusion) |
| G7 | eligibleWindowCount >= 2 | **PASS** 10 |
| G8 | observed rate / contract rate separated | **PASS** 4.11 vs 13 |
| G9 | O_denom = max(windowMax over eligible) | **PASS** 1 |
| G10 | O_denom >= 1 | **PASS** |
| G11 | K_min = ceil(4120/O_denom) | **PASS** 4120 |
| G12 | R_required = 1+K_min | **PASS** 4121 |
| G13 | R_required <= 5120 | **PASS** |
| G14 | Terminal dependency = 0 | **PASS** |
| G15 | raw evidence persisted | **PASS** (本報告 + console log) |

**G1 (22:11 source identity) PASS / G2 (production modification 0) PASS** は §0,2 で確認済み

## 10. 今回は CLOSE まで一気に進めない — 提示内容

本報告で提示する 3 点:

```text
1. campaign raw output (§5) — 全 11 window の console log 原文
2. 全 window の eligibility 表 (§6) — 11 行 × 16 列
3. O_denom / K_min / R_required 計算結果 (§8) — O=1, K=4120, R=4121, compat PASS bounded
```

**D102-C2-3 Numerical Result の TBD は、次報告で実測値 1 で置換し、D102-C2-3 = PASS を確定する。**

---

## 11. 参照

- `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp:139,424` (campaign runner)
- `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp:357,365,435` (harness-only dispatch)
- `CMakeLists.txt:1829` / `build/Debug/AudioEngineHarness.exe` 60MB
- `evidence/OdenomCampaign_console_2026-08-25.log` (raw console)
- `evidence/D102-C2-3-A-Harness-Capability-Audit.md` / `evidence/D102-C2-3-Odenom-Campaign-Raw-Evidence.md` (事前監査)

*本実行は production source 変更 0、O_denom の事後選択 0、observed の contract 昇格 0 で実施された。*
