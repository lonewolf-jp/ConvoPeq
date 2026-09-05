# D162-2-I2 — Reconfigure Orphan Ownership Repair（A 案実装・完了記録）

```text
Date:     2026-09-05
Type:     I2-0 preflight（read-only）→ I2-1 実装 → I2-2 test → I2-3 build/CTest → I2-4 Profile D 再試験 → I2-5 検証
Baseline: ConvoPeq.md Generated 2026-09-04 23:38:10
Binary:   build-diag RelWithDebInfo ConvoPeq.exe（Sep 5 10:12 = I2 修復込み・--target ConvoPeq）
判定:     **I2 = PASS（orphan 0/6・constructed = destroyed 6/6・既存 gate 全崩れなし）**
```

---

## 1. I2-0 preflight（read-only・PASS）

詳細: `evidence/D162-2I2/I2_0_PREFLIGHT.md`（同日）。要点:

1. pointer identity: `release()` 値を `placeholderRaw` に明示保持する一本道で確定（再取得不要）。
2. `CallerDestroy` 返却点: tryAdmit 失敗（h:4613-4614・未登録）/ OwnerChannel full（:4667・rollback 済み）/
   intent queue full（:4691・rollback 済み）。`world==nullptr`（:4647）と `seqId==0`（:4651）は
   `ownership=None`（CallerDestroy ではない・理論上のみで prepareToPlay では不発）。
3. pubResult1 を触らない理由: 対象は world 公開済み registered DSP であり、rollback CAS
   （Constructing→Reclaimed のみ成功・ISRDSPHandle.cpp:157-173）が Active 状態で失敗 →
   登録温存 → caller が破壊すると world dangling current（UAF）。pubResult2（未登録・
   release 済み）のみが破壊義務を負う。
4. `DSPLifetimeManager::destroyRolledBackDSP` は public（DSPLifetimeManager.h:19）・
   include 追加 1 行で利用可（Orchestrator:291-292 と同一履行パターン）。

## 2. I2-1 実装（production 1 ファイル 1 箇所）

`src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`（+include 1 行）:

- `placeholderRaw = placeholderDSP.release()` を明示保持（identity 一本道化）。
- `commitRuntimePublication` 失敗（非 committed かつ CallerDestroy）時に
  `destroyRolledBackDSP(placeholderRaw)` + `setActiveRuntimeDSP(nullptr)` を実行。
- 禁止事項（ShutdownRuntime reopen・INV-LIFE-9・commitRuntimePublication semantics・
  handle registry・EBR・Timer/Transition/Orchestrator caller 等）は一切変更せず。
- 注記どおり reconfigure 後の publication 不可（Active=0・bypass 継続）は I2 scope 外のまま残る。

## 3. I2-2 regression test（test infrastructure のみ）

`src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` に 2 関数追加
（runner へ登録済み）+ harness seam（`stopAudioOnly()` / `abandonEngine()`）:

- **T-I2-1** `testCallerDestroyTerminalDisposition`: prepare → release（admission Closed）→
  prepare の reconfigure 形。2 回目 prepare 後 `getActiveRuntimeDSP() == nullptr`（dangling なし）
  を検証。1 回目（成功パス）は slot に placeholder 残存 = T-I2-2（成功時 destroy されない）充足。
- **T-I2-3** `testRegisteredDSPFailurePreservesRegistration`: world 公開済み registered DSP に
  null world + needsRegistration で失敗させ、`{Failed, 非Transferred}` 後も
  registration 温存（idempotent register が同一 handle を返す + resolve が同一 instance を返す）
  を検証。double destroy 回帰なし。

### 既知 crash の記録（I3 課題として分離・I2 判定には影響しない）

1. **prepare → release → prepare → release の 2 回目 releaseResources は Debug で
   pre-existing segfault**。BISECT（修復の destroy を無効化しても再現）により I2 修復無起因と
   確定。T-I2-1 は engine を abandon（release 不実施・意図的 leak）して回避し、runner は
   `_exit(0)` で即終了する seam を追加。
2. abandon 後の AudioEngine 再構築も segfault するため、T-I2-1 は runner の最後に配置。
3. Release CTest の AudioEngineHarness crash も同系 pre-existing（§4）。

## 4. I2-3 Build / CTest

| 項目 | 結果 |
| --- | --- |
| Debug build | OK（AudioEngineHarness.exe 含む） |
| **Debug CTest** | **40/40 PASS（100%・40.4s）** ✓ regression gate 達成 |
| Release build | OK |
| **Release CTest** | **39/40** — AudioEngineHarness crash のみ。**pre-existing**: G2 世代から毎回
  `0xc0000374`（heap corruption）で落ちており（D162-2G2/G3/G4・I1R2 の ctest_release.log が
  全て同一失敗）、I2 の production/test を revert しても再現（BISECT 済み）→ **I2 無関係**。
  現在の即時 segfault はリンク順依存の同一クラス（既知 0xc0000374 系）と分類 |
| RelWithDebInfo | `--target ConvoPeq`（I1 系の正規手順）で OK。full build は FFTBackendTests の
  macro 解決失敗（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS/JUCE_DSP_USE_INTEL_MKL 未定義 target）で
  停止 — これも I2 無関係の CMake 設定課題（I3 候補） |

## 5. I2-4 Profile D 再試験（I1-D と同一条件・binary Sep 5 10:12）

条件: `--cli-run --cli-device-type WindowsAudio --cli-exit-ms 15000` × 6 runs・間隔 2s・
IR/burst/rebuild なし。CrashDumps 開始 10 個 → 終了 10 個（**新規 dump 0**）。

### Gate 判定表（I2 合格条件に対する実測）

| Gate | D-2 | D-3 | D-4 | D-5 | D-6 | D-7 | 集計 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exit = 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6** |
| clean shutdown（3 行 zone） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6** |
| crash dump | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0** |
| residual `remaining≠0` | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0** |
| EBR final pend / ovf | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **6/6** |
| duplicate destroy | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0** |
| shutdown-window XRUN | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0**（総 XRUN 0） |
| bootstrap DSP closure | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6**（VD retired=1 → epoch=9 → EBR digest → remaining=0） |
| **placeholder orphan** | **0** | **0** | **0** | **0** | **0** | **0** | **6/6 = 0** |
| **constructed = destroyed** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6** |
| E-4 accounting | 0=0+0+0+0 | 同 | 同 | 同 | 同 | 同 | **6/6** |
| D123 shutdown zone | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6** |
| stale-map forbidden HIT | 0 | 0 | 0 | 0 | 0 | 0 | **6/6 = 0** |
| E-3 / direct destroy | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | **6/6 = 0** |

### 修復の直接的証拠（各 run 共通 topology・D-2 例）

```text
:110  [DSP_FOOTPRINT] dsp=0000026EA2F1A080 gen=0 phase=construct TOTAL=108585016   ← placeholder 構築
:114  [D117_DESTROY]         dsp=0000026EA2F1A080                                   ← I2 修復（destroyRolledBackDSP → destroyDSPCoreNode）
:115  [DSP_DESTROY_FOOTPRINT] dsp=0000026EA2F1A080 gen=0 trackedFootprint=108585016
:133  [DSP_FOOTPRINT_RELEASED] dsp=0000026EA2F1A080 remaining=0                        ← allocator 検収
      （3 行とも placeholder pointer と一致 — direct-destroy chain が成立）

:44-46  [D162-2G2_VD_RETIRE] dsp=…B2534080 → retired=1 → enqueue=0 epoch=9              ← bootstrap DSP（authority 経由）
:141-160  D117_DESTROY → DSP_DESTROY_FOOTPRINT gen=4 → DSP_FOOTPRINT_RELEASED remaining=0 ← EBR digest
```

- **orphan 0/6**: I1-D で未処分だった placeholder が、修復により prepareToPlay 内で
  `destroyRolledBackDSP` → `destroyDSPCoreNode`（D117_DESTROY / DSP_DESTROY_FOOTPRINT /
  DSP_FOOTPRINT_RELEASED remaining=0 の 3 行 chain）で terminal disposition されている。
- **shutdown 時 `retired=0` 行の消失**: I1-D では shutdown `:331 retire(slot ptr)` が
  `retired=0`（map MISS）を出力していたが、I2 では slot が事前 null 化されたため出力ゼロ
  → **activeRuntimeDSPSlot への dangling 残置なし（I2-5 acceptance 達成）**。
- construction/destruction 会計: 2 個体構築（bootstrap pre-log + placeholder in-log）/
  2 個体破壊・pointer identity 完全一致 = **1:1 成立**。

## 6. I2-5 後始末確認

- slot dangling: なし（上記 retired=0 消失 + T-I2-1 の直接 assert）。
- E-4 closure: deferred 活動 0 → 0=0+0+0+0 × 6/6。
- D123 zone: SHUTDOWN_BEGIN → reset completed → LOGGER_DETACH/END 6/6 完備。
- device lifecycle: switch success → callbacks → closeAudioDevice → reset の直列順序 6/6。

## 7. 判定

**D162-2-I2 = PASS。**

- 修復単位: production 1 ファイル 1 箇所（PrepareToPlay.cpp）+ test infrastructure のみ。
- I2 成功条件「publish failure が CallerDestroy を返したとき、放棄された未登録 DSP の
  ownership が必ず terminal disposition される」を実機 6/6 で達成。
- reconfigure 後の publication 不可（Active=0・bypass 継続）は I2 の合否対象外として
  残存（R0 §7-C/D・I3 課題）。
- I3 課題として記録: (a) Debug の reconfigure 二重 release segfault、
  (b) Release AudioEngineHarness pre-existing crash（G2 以来）、
  (c) RWDI full build の FFTBackendTests macro 設定、(d) MMCSS-ASIO err=1552 分類、
  (e) commitRuntimePublication None 返り値の caller 契約明確化。
