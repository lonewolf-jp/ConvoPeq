# D169-2-5 — Targeted Collapse Regression（Work Record）

```text
D169-2-5 — Duplicate-Prepare Collapse Targeted Runtime Regression
Date:      2026-09-07
Contract:  evidence/D169/D169_2_2_REPAIR_CONTRACT.md（RC-D169-2-1〜7）
Preflight: evidence/D169/D169_2_3_PREFLIGHT_SOURCE_AUDIT.md（P1〜P5 GO）
Impl:      evidence/D169/D169_2_4_MINIMAL_IMPLEMENTATION.md（IMPLEMENTED）
Verdict:   **PASS — D169-2-6 GO**
```

---

## 1. Source precondition 再確認（§1）

- `enterPrepare() → expectedPhase == Prepared → return` 経路が `PrepareToPlay.cpp:33-37`
  に存在（collapse return :36 / leavePrepare :344 — 到達不能を実測）。
- collapse branch 内は `diagLog` 1 行 + return のみ（それ以外の処理なし）。
- `git diff` で `ISRLifecycle.cpp/.h` 無変更を再確認。

## 2. Same-SR/BS duplicate prepare の runtime 観測（§2・§3・§4）

### 実行手段の確定（既存経路の不足証明）

既存 runtime 経路では collapse に到達できないことを source 走査で証明した:

| 経路 | 調査結果 |
| --- | --- |
| CLI（`--cli-device-type`） | startup 1 回のみ適用（`MainWindow.cpp:430-479`）・1 run = 1 prepare（D168/D170 ログ実測） |
| JUCE `setAudioDeviceSetup` | 同一 setup は early return（`juce_AudioDeviceManager.cpp:815-818`） |
| 既存 unit test 全走査 | 全て release 後 prepare（phase=Released）か SR 交互で collapse を回避（D169-2-1 R11・`testD167...:594-597` コメント） |
| soak S1-S5 | prepare 呼出 0 件（`SoakPublishIntegrationTests.cpp` 実測） |

よって既存 harness テストに targeted regression 1 件を追加した
（**test source deviation 1 件** — 「原則変更禁止」に対する文書化された例外。
R11 の恒久 coverage gap も同時に埋める。production source は無変更）。

### `testD169DuplicatePrepareCollapseNoop`（新設・`PublishPipelineIntegrationTests.cpp`）

シーケンス:

```text
h.start(48000, 512)            → 初回 prepare（Uninitialized → Preparing → Prepared）
startup rebuild 完了待ち + 400ms
h.stopAudioOnly()              ← harness 契約（prepareToPlay は audio thread 停止後に呼ぶ）
e.releaseResources()           → reconfigure pass（phase/state Prepared 維持）
ベースライン観測（gen / seq / slot / isEnginePrepared）
capture logger 設置（CriticalSection で直列化 — Timer [MEM_SNAP] 並行書込み対策）
prepareToPlay(512, 48000) ×4   → same SR/BS duplicate prepare（collapse 経路）
観測比較
prepareToPlay(512, 44100)      → SR 変更（非 collapse・完全 prepare）
h.stop()                       → terminal 完走確認
```

### 観測結果（3 config 全 PASS）

| 観測項目 | 結果 |
| --- | --- |
| abort / exception / process termination | **0**（collapse ×4 後もテスト継続 — 旧 code なら `leavePrepare` で `std::abort()`） |
| collapse diagLog（`duplicate-prepare collapsed`） | **4/4 回観測**（collapse が実際に発生） |
| `prepareToPlay: enter spb=`（body 入口 log） | **0 回**（body に侵入していない — collapse は body より前に return） |
| `REBUILD_TELEMETRY` 行 | **0 回**（`submitRebuildIntent` 不発） |
| `currentBuildGeneration` | **不変**（:96 generation reset も rebuild request も発火せず） |
| publication `sequenceId` | **不変**（idle publish #2 不発） |
| `activeRuntimeDSPSlot`（placeholder pointer） | **不変**（placeholder 再作成・null 化なし） |
| `isEnginePrepared()` | collapse 前後で **true 維持**（lifecycleState == Prepared・Preparing 経由なし） |
| admission | Open 維持（reconfigure pass 後も collapse 後も） |
| collapse 後 terminal shutdown | admission Closed + `ShutdownComplete` 到達（phase integrity の帰無検証） |

### 非 collapse regression（§5）

- SR 変更 re-prepare（512/48000 → 512/44100）: 完全 prepare が実行され publication が進行
  （D167 step-d と同一の正常系・collapse 分岐に入らない = 判別子 negative check §6 合格）。
- 既存テスト **無変更** で full suite 実行 → 後述のとおり Debug/Release CTest 40/40・
  RWDI/Debug/Release harness とも全 PASS（`Uninitialized→Preparing→Prepared`・
  `Released→Preparing→Prepared`（T-I2-1）・`Prepared+SR変更`（D167 test）を既存テストが網羅）。

## 3. 実行中に発生した harness segfault の調査記録（解消済み）

初版テストで Debug/Release/RWDI harness が T-I2-3 付近で segfault した。調査:

1. **bisect**: `PrepareToPlay.cpp` の D169-2-4 分岐を stash して rebuild → 同一 crash
   → **D169-2-4 修復は無関係**（pre-existing 系統の可能性を一時疑った）。
2. **crash dump 解析**（minidump + PDB symbolize）: crash thread = `AudioEngine::timerCallback`
   内（MEM_SNAP block 付近）。Sep 6 の旧 dump 群は既知の 2 系統
   （ucrtbase abort = D167 collapse abort・AV = D169-1 publish race）と区別される。
3. **根本原因特定**: 初版テストの 2 つの契約違反。
   - `prepareToPlay` を **audio thread 走行中**に呼出（harness 契約「AudioThread 停止中のみ呼ぶ」
     違反 → buffer realloc が RT thread と競合）。D167 テストは `stopAudioOnly` + reconfigure
     release 後に呼ぶのが正規パターン。
   - capture logger が Timer thread の `[MEM_SNAP]` 並行書込み（`writeToLog` は全 config 有効）
     と非同期アクセス（`juce::StringArray` は thread-safe でない）。
4. **修正**: collapse 投入前に `stopAudioOnly()` + `releaseResources()`（D167 と同一パターン）+
   capture logger を `CriticalSection` で直列化。
5. **修正後**: 3 config harness + CTest Debug/Release が **全 PASS**（crash 完全消滅）
   → crash はテスト初版の契約違反起因であり、production / 既存テスト / D169-2-4 には起因しない。

備考（調査過程で観測した pre-existing hazard・本 track では処置しない）:
`AudioEngine::timerCallback` の MEM_SNAP sampler（`Timer.cpp:1079-1088`）は
`getActiveRuntimeDSP()`（pointer slot）の値を無条件で `collectTrackedMemoryStatistics()`
に渡す。pointer slot は rebuild が placeholder を destroy した後も free 済み address を
保持し続けるため（D169-1 §3）、diagnostic sampler が dangling pointer を読み得る
（D169-1 family の別 consumer）。timing 依存の flaky AV として yesterday の dump 群にも
同系の痕跡がある。**別 track（diagnostic sampler の slot 参照見直し候補）として記録**。

## 4. PASS 条件照合（指示の 9 条件）

| 条件 | 結果 |
| --- | --- |
| Same SR/BS duplicate prepare が abort しない | ✓ ×4 連続で abort 0 |
| collapse が実際に発生 | ✓ diagLog 4/4 観測 |
| collapse 後 `Prepared` 維持 | ✓ isEnginePrepared true・admission Open |
| `leavePrepare()` が collapse 時に呼ばれない | ✓ body 入口 log 0（collapse は :36 return・leavePrepare :344 に到達経路なし） |
| prepare body の副作用が再実行されない | ✓ generation / sequenceId / telemetry / slot すべて不変 |
| generation / rebuild / publication が duplicate により変化しない | ✓ 全不変 |
| 非collapse prepare が従来どおり成功 | ✓ SR 変更 re-prepare で publication 進行 + 既存 CTest 40/40 ×2 |
| D169-2-4 の変更範囲を超える source change = 0 | ✓ production 1 ファイル（D169-2-4 分）のみ・test 1 関数追加（deviation 記録済み） |
| RC-1〜RC-10 違反 = 0 | ✓ RC-1〜10 全照合 OK |

## 5. Environment / Evidence

- Binary: build-diag Debug / Release / RelWithDebInfo（D169-2-4 + D169-2-5 test 入り）
- Logs: evidence/D169/d169_2_5_harness_{rwdi,debug,release}.log（全 EXIT=0）
- CTest: evidence/D169/d169_2_5_ctest_{debug,release}.log（40/40 ×2）
- 調査記録: d169_2_5_harness_debug2.log（初版 crash）・d169_2_5_harness_bisect_baseline.log
  （bisect 再現）・dump 解析（AudioEngineHarness.exe.32368.dmp → timerCallback）
- Test source: `PublishPipelineIntegrationTests.cpp` に `testD169DuplicatePrepareCollapseNoop`
  1 関数 + runner entry 1 箇所（**deviation 1 件** — 既存経路では collapse に到達不可能な
  ため。R11 gap の恒久 coverage として維持）

## 6. Verdict

**PASS — D169-2-6 GO。**

次: **D169-2-6 Device restart / stress**（同一 SR/BS device restart 相当の反復・
JUCE 経路での collapse 反復 + 長時間安定性）→ D169-2-7 full regression → D169-2 close。
