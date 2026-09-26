# P1-5-IR-P2 — Step 5-AL / P3-5-R30: Recovery-origin Vehicle Implementation / Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R30）
- **種別**: test-only implementation＋build gate＋run gate。**production 変更 0**・新 counter 0・新 getter 0・
  Intent 変更 0・`recoveryObligationId` semantics 変更 0・deferred/retry/admission 変更 0・CMake 0。
- **判定**: **R30-B（STOP-E）**。D1 vehicle は実装・build 成功し、
  **Recovery-origin publish を 3/3 episode 再現**（seqΔ=1／coordΔ=0／cmtΔ=0／drpΔ=0）。
  ただし R30 §7 の `waitForDrain()` 検証は**成立不能**（baseline でも false・全 audit 成分 0）
  ことが判明 → STOP-E。原因は recovery 経路ではなく**観測子（shutdown 契約 API の誤用）**。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF
R16 take／R19 B2／R22 commit／R27 origin plumbing／R10 accessor／T6/T9／R23 vehicle 保持
ConvoPeq.md 再生成済み（2026-09-23 23:04:35・generator output_sourcecode_markdown.py）
production diff = R27 のみ（8 files・+155/−12・R30 追加 0）
CMake = clean（変更 0）
```

- R30 §1 が参照を求める `ConvoPeq(3).md` は本環境に存在しない（探索 root 全域・R29 と同じ）。
  最新 `ConvoPeq.md` を source authority とした。

## 2. Latest ConvoPeq Source Reconciliation（PASS）

R30 §2 の必須 API を live source で再確認（R29 の D1 設計と一致・変更なし）：

| API | 位置 | 確認 |
| --- | --- | --- |
| `AudioEngine::submitRecoveryIntent` | AudioEngine.h:4686 | `inline void (DSPHandle, const RuntimeBuildSnapshot&) noexcept`・public |
| `AudioEngine::getCurrentBuildSnapshotForRecovery` | AudioEngine.h:4677 | `RuntimeBuildSnapshot () const noexcept`・public |
| `AudioEngine::registerDSPHandleForRuntime` | AudioEngine.h:4520 | public（既存 test が使用） |
| `AudioEngine::hasAuthoritativePublishedRuntime` | AudioEngine.h:3853 | public |
| `AudioEngine::observePublishedWorld` | AudioEngine.h:1193 | public |
| `AudioEngine::waitForDrain` | AudioEngine.h:1601 | `bool (int=2000, int=2)`・public |
| `AudioEngine::isFullyDrained` | AudioEngine.h:1600 | public |
| `RuntimeIntentCoordinator::submitRecoveryRequest` | Coordinator.h:565 | private（AudioEngine member 経由） |
| `RuntimeIntentCoordinator::resolveRecoveryObligation` | Coordinator.h:573 | private |
| `RuntimeIntentCoordinator::isFullyDrained` | Coordinator.h:221 | `liveLogicalRecoveryObligationCount()==0` を含む（cpp:561） |
| `recoveryObligationId` | PublishPayload h:732／搬送 AudioEngine.h:4888 | R27 plumbing のまま |
| `coordinatorTakeCount_` | h:974／ProcessIntent.cpp:61-62 | Main-only writer 1 |
| `getRebuildCommitEnqueueCount` / `getLastCommittedPublicationSequence` / `getRuntimeLifecycleDiagnostics` | AudioEngine.h:2084 / 1744 / 1732 | public |
| `AudioEngine::collectDrainAudit` | AudioEngine.h:1604 | public（R30 で診断に使用） |

- 差分・契約矛盾なし → 実装に進んだ。

## 3. D1 Implementation Scope

R29 D1 をそのまま実装（test-only）：

```text
--p1-recovery-origin
  1. h.start() → authoritative published Runtime 待機（observePublishedWorld && hasAuthoritativePublishedRuntime）
  2. handle = registerDSPHandleForRuntime(activeDSP)（非 null）
  3. baseline drain sample（sleepPump(1000) 後）
  4. control（trigger なし 5000 ms）→ dSeq を観測
  5. 各 episode:
       pre read（cmt/coord/seq/drp + take/bld/req/que/dup/blo）
       snapshot = getCurrentBuildSnapshotForRecovery(); snapshot.sealed = true
       submitRecoveryIntent(handle, snapshot)   ← 1 episode 1 回
       seq advancement 待機（≤20 s）
       waitForDrain(20 s) + drain audit（pre/post）
       post read + delta
  6. audio 停止後の drain 追加診断（stopAudioOnly → waitForDrain）
  7. h.stop()
```

- 戻り値のみで publish 成功と判定しない（trigger acceptance と publish completion を分離）。
- `cmt` は topology/context 観測（Recovery の counter として扱わない）。
- 新 logger/prefix family なし（`[P1REC]` は既存 `emitLine`）。

## 4. Production / Test / CMake Diff Boundary

| 対象 | 変更 |
| --- | --- |
| production source（src/audioengine） | **0**（diff は R27 の8ファイルのみ・+155/−12 は R27 分） |
| test source | 2 既存 TU：`P1PolyphaseGainCharacterization.cpp`（vehicle 本体）＋`PublishPipelineIntegrationTests.cpp`（decl 1 行＋dispatch 1 行） |
| CMake | **0**（`git status` clean・既存 AudioEngineHarness target 内） |
| counter / getter / Intent / semantics / deferred / retry / admission | 0 |

- R30 §1 の禁止（production API 追加・counter・getter・Intent 変更）すべて遵守。

## 5. Vehicle Command

```text
build/Release/AudioEngineHarness.exe --p1-recovery-origin
（生成: build.bat 相当（vcvarsall x64＋oneAPI setvars intel64）→ cmake --build build --config Release --target AudioEngineHarness）
```

## 6. Precondition Verification（PASS）

```text
[P1REC] precondition ok slot=254 gen=1
```

- authoritative published Runtime 成立、active DSP handle 取得（slot=254・generation=1）。
- PRECONDITION FAILURE は発生せず（Recovery 経路の問題と混同する余地なし）。

## 7. Recovery Episode Execution

- D1 trigger は **1 episode につき 1 回**・逐次（同時多発なし）。
- 各 episode で seq advancement を確認後、次 episode へ（coalescing 回避）。

### Control（trigger なし・5000 ms）

```text
[P1REC] control_notrigger waitMs=5000 dSeq=0
```

- **自発的 publication は 0**。したがって episode の seq 前進は trigger 起因（window 粒度）。

## 8. cmt / coord / seq / drp Observations

Run: `--p1-recovery-origin`（Release/OFF・R27 binary から再 build）

| episode | dSeq | dCoord | dCmt | dDrp | dTake | dBld | seqAdvanced |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep1 | **+1** | **0** | **0** | 0 | 0 | 0 | 1 |
| ep2 | **+1** | **0** | **0** | 0 | 0 | 0 | 1 |
| ep3 | **+1** | **0** | **0** | 0 | 0 | 0 | 1 |

絶対値（ep2 pre → ep3 post で確認）:

```text
pre  req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=3 blo=0
post req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=4 blo=0
... ep3 後 seq=6（各 episode で seq のみ +1・cmt/coord/take/bld 不変）
```

- `cmt`（Main-side rebuild commit-enqueue）は **増加しない** → main-site rebuild 経由ではない。
- `coord`（Main-origin pop）は **増加しない** → R27 origin plumbing の設計どおり Recovery は Main bucket 外。
- `take`/`bld` も不変 → Main-side build 経由ではない。
- `drp` は 0（commit 側 monotonicity reject なし）。
- 再現性：baseline settle 後の 3 run（r30_recovery3 / 4 / 5）で同一構造（ep ごと seq+1・coord 0・cmt 0）。

### 注意（per-task attribution は依然として不能）

```text
seqΔ=1 かつ cmt/coord/bld 不変＋control dSeq=0 は Recovery-origin publish と整合するが、
「その seq が Recovery publish である」per-task 証明は既存観測では不可（R28 §9.2・§12）。
本 Step の主張は window 粒度（既知 trigger ＋ control 0）に留める。
```

## 9. waitForDrain Observation（**成立せず＝STOP-E**）

```text
drain_baseline（trigger 前）: fullyDrained=0 pendPub=0 pendRetire=0 xfade=0 routerPending=0
                               deferred=0 quarRes=0 activeWorlds=1 published=3 retired=2
                               activeReaders=1 stuckReaders=0 overflowRes=0
drain_pre/post（各 episode）: fullyDrained=0（全 audit 成分 0 のまま）
drain_after_audio_stop      : drainOk=0 fullyDrained=0 activeReaders=0（他は 0）
```

- **trigger 前（baseline）から `isFullyDrained()==false`**、かつ `collectDrainAudit()` の
  audit 成分はすべて 0。audio thread 停止後（activeReaders=0）でも false。
- `isFullyDrained()` の残 predicate は `pendingReclaimHandles_` と
  `runtimePublicationBridge_.isFullyDrained()` の内部条件であり、`collectDrainAudit` は
  これを露出しない（本 Step では新 getter 追加禁止のため分解不能）。
- `waitForDrain` は自身の contract を明記している：
  `Threading.cpp:218「waitForDrain は AudioStopped 以降でのみ呼ばれる」`＋
  `jassert(phase ∈ {AudioStopped … ShutdownComplete})`（:221-229）。
  内部 caller も `releaseResources`（:253／:615）のみ。
- → **`waitForDrain` / `isFullyDrained` は run-time 観測子ではない**。
  R29 §11 の「drain で exactly-once を間接観測」という設計前提は**誤り**（本 Step で確定）。
- これは Recovery 経路の欠陥ではない（baseline・control でも同一）。

## 10. Repeated Episode Results

```text
episodes=3／ok=0（ok は seqAdvanced && drained && dCoord==0 の複合条件）
  seqAdvanced: 3/3 PASS
  dCoord==0 : 3/3 PASS
  drained    : 0/3（§9 のとおり観測子自体が不成立）
```

- 複合条件 `ok` は drain のため 0 だが、**Recovery-origin publish の再現（主目的）は 3/3**。

## 11. Failure / STOP Analysis

```text
STOP-A（submitRecoveryIntent が説明不能な false）      : 非該当（trigger は毎回受理・publish 到達）
STOP-B（admission 成立後 Builder へ進まない）          : 非該当（seq 前進＝Builder→publish 到達）
STOP-C（Builder まで進むが publish されない seqΔ=0）    : 非該当（seqΔ=+1 ×3）
STOP-D（Recovery なのに coordΔ>0）                    : 非該当（coordΔ=0 ×3）
STOP-E（waitForDrain が成功しない）                   : **該当**（§9）
STOP-F（二重 terminalization を示唆する crash/assert）  : 非該当（crash/assert/exception 0）
STOP-G（vehicle 実装のため production を変更したく）    : 非該当（production 変更 0）
```

- STOP-E の原因分類：**観測子の設計誤り**（shutdown 契約 API を run-time で使用）。
  recovery obligation の terminalization 不全・double-resolve を示す証拠はない
  （crash 0／seq 前進／baseline でも同一）。
- 追加観測（初回 run の知見）：baseline settle を行わない初回 run では ep1 が
  **起動直後の main-site rebuild publish** と混入（dCmt=1/dCoord=1/dBld=1）。
  したがって `sleepPump(1000)` の baseline settle＋trigger なし control は必須。

## 12. D105 Limitation

```text
D105 dynamic winner identity = UNOBSERVABLE（R28/R29 の結論を維持）
  - resolveRecoveryObligation は liveCount CAS 単一 authority（source 不変）
  - `won` 戻り値は両 call site で破棄・専用 counter/getter なし（R30 でも追加せず）
  - Route A/B の動的 winner を既存観測で識別する手段は本 Step でも得られていない
exactly-once terminalization の間接観測
  - R29 の想定（waitForDrain）は §9 により不成立 → 現時点で run-time の代替観測は未確立
  - 候補（R31 設計事項）: engine shutdown 契約内での drain（releaseResources → waitForDrain →
    finalizeShutdown）を終端観測子として使う／または別 design gate で observability を追加
```

- 本 Step では winner・terminalization のいずれも**判定しない**。

## 13. R30 Gate

```text
R30-A： 非該当
  [OK] Recovery trigger  D1 が Full AudioEngine から実行可能（PASS）
  [OK] admission        Recovery request 成立（seq 前進で間接確認）
  [OK] publish          seqΔ > 0（3/3）
  [OK] origin observation Recovery publish で coordΔ == 0（3/3）
  [OK] Main counter     Recovery によって coord が増加しない（3/3）
  [NG] drain            waitForDrain() 成功 → 不成立（§9・STOP-E）
  [OK] repeat           3 episode で再現
  [OK] stability        crash/assert 0
  [OK] production diff  0
  [OK] new counter 0／new getter 0／Intent 変更 0／recovery semantics 変更 0
R30-B： ADOPTED
  D1 vehicle は実装・build・実行に成功し、主目的（Recovery-origin publish の Full AudioEngine 実走）を
  3/3 episode で達成。しかし R30 §7 の drain 観測子が run-time では成立不能
  （baseline から isFullyDrained()==false・全 audit 成分 0・audio 停止後も false）と確定 → STOP-E。
  原因は recovery 経路ではなく観測子設計（R29 §11 の前提誤り）。
R30-C： 非該当（contract 矛盾・Recovery id の 0 化・coord 混入なし）
```

- 主目的 PASS／§7 観測子 FAIL の分離を明記する（R30-A を R28-A へ繰り上げない）。
- source は R27 production＋R30 test vehicle を保持する。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。

## 14. Next Step

```text
R30-B → R31 は「Recovery-origin Full-pipeline Measurement」ではなく、
        先に「terminalization 観測子の再設計 gate」を置くことを推奨：

  R31-A 候補: shutdown 契約内 drain を終端観測子として使う vehicle 改訂
              （episodes → h.stop()/releaseResources → 内部 waitForDrain/finalizeShutdown の
               結果を観測。isFullyDrained が shutdown では成立し得るかを実測）
  R31-B 候補: D105 winner/per-task observability の design gate（R28-B 残件・別 gate）

  - Recovery-origin publish の生成自体は R30 で確立済み（本 Step の成果）。
  - control（trigger なし dSeq=0）と 3 episode 再現は R31 の測定基盤として再利用可能。
```

- R31 へ進む前に、本 Step の STOP-E を解消する観測子設計を確定すること。
