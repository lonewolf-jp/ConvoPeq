# P1-5-IR-P2 — Step 5-AO / P3-5-R33-A: Shutdown Drain Retire/Reclaim Cause Isolation Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R33-A）
- **種別**: **read-only source / diagnostic audit**。
  production 0・CMake 0・API/getter/counter 0・Recovery/Shutdown semantics 0・Retire/Reclaim 実装変更 0・
  test-only vehicle 変更 0。既存 evidence / 既存ログのみ使用。
- **判定**: **R33-A-PASS**（限定付き）。
  `waitForDrain` の全 predicate を source から展開し、timeout を発生させる predicate を
  **`retireDepth == 0`（`routerPendingRetire` = DeferredDeletionQueue.sizeApprox()）** と特定。
  その残留は **VerifyDrained の terminal disposition が「現在 epoch」で enqueue した EBR entry** であり、
  `reclaim` の `isOlder(entry.epoch, minReaderEpoch)` 条件と `waitForDrain` loop が
  **epoch を進めない**ことの合成で説明できる（source 内に同一機構の既存記述あり）。
  **Recovery 起因ではない**（default harness でも同一 `blockingReason=Unknown`）。
  限定: **個体識別**（どの World/DSP か）は per-entry observability が無いため未確定 → 必要なら R33-C へ。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持
ConvoPeq.md = 2026-09-23 23:33:38（R32 で再生成・本 Step でも同じものを authority とする）
production diff = R27 のみ／test diff = R23+R30+R32 vehicle／CMake clean
R33-A の差分 = 本ドキュメントのみ（read-only）
```

R32 で取得済みの一次データ（本 Step の入力）：

```text
D3 = vehicle（Recovery 3 episodes）
  terminal : phase=ShutdownComplete / completed=1 / violations=0 / blockingReason=Unknown(9)
  audit    : fullyDrained=0 routerPending=2 pendPub=0 pendRetire=0 xfade=0 deferred=0
             quarRes=0 activeWorlds=1 published=6 retired=5 readers=0 stuck=0 overflowRes=0
  trace    : sh3_pendingRetire=2
D0 = default harness（Recovery 0）
  trace    : phase=ShutdownComplete / blockingReason=Unknown(9) / violations=0 / sh3_pendingRetire=1
```

## 2. Latest ConvoPeq Source Reconciliation

R31/R32 で確認した shutdown chain が最新 source でも不変であることを再確認（差分なし）：

```text
releaseResources()（terminal pass）
  :96  transitionTo(AudioStopped)／:109 closeAdmission()
  :237-238 shutdownCoordinatorLoop() + stopRebuildThread()
  :250-256 joinProducers retry（timeout 時 :253 waitForDrain(100,1)）
  :259 ObserverDrained／:264 RetireClosed／:265 EpochSettled
  :277 escalateAllRetires(Critical)
  :285-311 Graceful Drain（≤5000 ms・毎 tick publishEpoch()+tryReclaim()）
  :376 transitionTo(ReclaimComplete)                ← ★ ここで「reclaim 完了」ではない（§4）
  :383 transitionTo(EmergencyDrain)
  :475 transitionTo(VerifyDrained)
  :502-517 active/fading handle retire + tryShutdownQuiescentReclaim
  :554-555 if readers==0 → drainAllQuarantineStore()
  :537-543 world clear → retirePublishedRuntimeWorldNonRt(clearedWorld,true)
  :568-588 DSPLifetimeManager::retire(activeDSPToDestroy / fadingDSPToDestroy)   ← ★ EBR enqueue
  :599-600 clearDeferredForShutdown()／:605-606 resetRedriveBudget()
  :615 waitForDrain(2000,2)                          ← ★ 本 Step の対象
  :616 timedOut=!drainedWithinBudget
  :622 drainPendingRetireIntentsForShutdown()
  :624-633 if (timedOut) markTimedOut(Unknown)       ← blockingReason=9 の発生源
  :654-662 if (!drained||!isFullyDrained()) → drainDeferredRetireQueues(true)+tryReclaim
  :664 finalizeShutdown(timedOut)／:744 transitionTo(ShutdownComplete)
```

- `markTimedOut` の caller は **この :632 の 1 箇所のみ**（`markFailed` は caller 0）。
  ⇒ `blockingReason != None` は timeout の十分条件（§3）。

## 3. waitForDrain Predicate Decomposition

`AudioEngine::waitForDrain`（Threading.cpp:215-247）は `while (!isFullyDrained())` で
`drainDeferredRetireQueues(true)` を回すだけである。`isFullyDrained`（Threading.cpp:204-212）＋
Coordinator 側（Coordinator.cpp:506-561）を完全展開すると：

```text
DrainPredicate =
  P1  !hasDeferredCommit
  P2  pendingReclaimEmpty                 （pendingReclaimHandles_.empty()）
  P3  retireDepth == 0                    （routerPendingRetire = pendingRetireCount()+fallbackQueueDepth_）
  P4  lifetimeRetireIntentPending == 0
  P5  ringResident == 0                   （OverflowRing resident）
  P6  dspQuarantineResident == 0
  P7  retireQuarantineResident == 0
  P8  terminalReclaimResident == 0
  --- 以下 coordinator（runtimePublicationBridge_.isFullyDrained()）---
  P9  swapPending_ == false
  P10 intentQueue_.sizeApprox() == 0
  P11 observeDeferredRing_ == 0
  P12 quarantineFallbackQueue_.sizeApprox() == 0
  P13 recoveryIntentQueue_ == 0
  P14 retireBacklogCount_ == 0
  P15 publicationBacklogCount_ == 0
  P16 publicationIntentResidencyCount_ == 0
  P17 pendingIntentCount_ == 0
  P18 reclaimInFlightCount_ == 0
  P19 quarantineIntentResidencyCount_ == 0
  P20 quarantineRingResidencyCount_ == 0
  P21 !recoveryAdmissionPending_
  P22 liveLogicalRecoveryObligationCount() == 0
```

### predicate 別の状態（R32 の実測 + source）

| predicate | D3 vehicle | D0 default | 観測/根拠 | timeout 原因候補 |
| --- | ---: | ---: | --- | --- |
| P1 deferred publication | 0 | 0 | `collectDrainAudit.deferredPublish` | ✗ |
| **P2 pendingReclaimEmpty** | **?** | **?** | collectDrainAudit **非露出**（Threading.cpp:198-202） | **候補** |
| **P3 retireDepth (routerPending)** | **2** | **1** | `collectDrainAudit.routerPendingRetire` | **★ 主候補** |
| P4 lifetimeRetireIntentPending | 0 | 0 | `collectDrainAudit.pendingRetire` | ✗ |
| P5 ringResident | 0 | 0 | `collectDrainAudit.overflowRingResident` | ✗ |
| P6/P7 quarantine | 0 | 0 | `collectDrainAudit.quarantineResident` | ✗ |
| P8 terminalReclaimResident | ? | ? | collectDrainAudit **非露出** | 候補（弱） |
| P9 swapPending_ | ? | ? | 非露出 | ✗（shutdown 中 producer 停止） |
| P10-P22 coordinator | ? | ? | 非露出（P22 は「Recovery obligation」） | ✗（P1/P15 等は観測 0 と整合） |
| crossfade | 0 | 0 | `collectDrainAudit.activeCrossfadeCount` | ✗ |
| reader | 0 | 0 | `activeReaders=0` / `stuckReaders=0` | ✗ |

**注意（測定時点）**: `collectDrainAudit` の値は `h.stop()` 復帰**後**（ShutdownComplete 後）の
スナップショットである。:615 の時点の値そのものではない。したがって「P3 が :615 で非 0 だった」ことは
直接観測していない。§5-§6 の source 追跡が経路を確定する。

## 4. Shutdown Phase-by-Phase Drain State（ReclaimComplete の意味）

| phase（行） | routerPending | pendingReclaimHandles | epoch | readers | activeWorlds | 備考 |
| --- | --- | --- | --- | --- | --- | --- |
| Running | 変動 | 変動 | audio/publish ごと前進 | ≥1（audio） | 1 | — |
| AudioStopped (:96) | 凍結開始 | — | — | audio 停止済 | 1 | audio thread は harness が先に停止 |
| StopWorkers (:236) | 凍結 | 凍結 | — | 0 | 1 | Coordinator/Rebuild join |
| **Graceful Drain (:285-311, ≤5 s)** | **→ 0 可能** | → 0 可能 | **毎 tick 前進 (:305)** | 0 | 1 | epoch 前進があるため drain 可能 |
| **ReclaimComplete (:376)** | ここまでで 0 近傍 | — | — | 0 | 1 | **ただし以降で増え得る** |
| EmergencyDrain (:383) | 条件付 tryReclaim | — | — | 0 | 1 | 通常 skip |
| **VerifyDrained (:475)** | **増加し得る** | **増加し得る** | **前進しない** | 0 | 1 | :537-588 で World/DSP の terminal disposition → enqueue |
| **waitForDrain (2000) (:615)** | **0 にできない**（epoch-equal） | 再試行のみ | **前進しない** | 0 | 1 | loop は `drainDeferredRetireQueues` のみ |
| finalize/ShutdownComplete (:664/:744) | timedOut なら残留 | 残留 | — | 0 | 1 | dtor で最終回収（§5-c） |

**§Step2 の核心への回答**:

```text
「ReclaimComplete(:376) に到達した時点で reclaim が完全終了した」とは言えない。
  理由: :376 は Graceful Drain(:285-311) の直後に置かれ、
        その後に実行される VerifyDrained の terminal disposition (:537-588) が
        **再度 EBR entry を enqueue する**。
        :376 の名前は「graceful drain 区間の終了」を意味し、
        shutdown 全体の reclaim 完了を意味しない。
```

## 5. routerPending Lifecycle

### a. 定義

```text
routerPending = m_retireRouter->pendingRetireCount() + fallbackQueueDepth_
  pendingRetireCount() = EpochDomain::deferredDeletionQueue.sizeApprox()   (EpochDomain.h:430-433)
  fallbackQueueDepth_  = drainDeferredRetireQueues が 0 を publish（Retire.cpp:136/139）→ 実質 0
  sizeApprox() = enqueuePos - dequeuePos（単調カウンタ差分・approximate）
```

### b. increment（enqueue）— 誰が入れるか

```text
DSPLifetimeManager::retire(dsp, epoch) / retireByHandle(handle)（DSPLifetimeManager.cpp:40-137）
  → router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode,
                              epoch = router_->currentEpoch(),  DeletionEntryType::Generic)
Retire 経路の主体:
  1. publish 後の旧 World retire（通常運転）
  2. releaseResources VerifyDrained の terminal disposition:
       :537-543  clearPublishedRuntimeSnapshotsNonRt → retirePublishedRuntimeWorldNonRt(clearedWorld,true)
       :568-588  DSPLifetimeManager::retire(activeDSPToDestroy) / retire(fadingDSPToDestroy)
     ⇒ **すべて epoch = currentEpoch() で enqueue**
  3. dtor の同種 disposition（CtorDtor.cpp:263-269）
```

### c. decrement（consume）— いつ消えるか

```text
DeferredDeletionQueue::reclaim(minReaderEpoch)（DeferredDeletionQueue.h:110-178）
  条件: isOlder(entry.epoch, minReaderEpoch) == true  … すなわち entry.epoch < minReaderEpoch
  FIFO 厳守: 先頭が reclaim 不可なら即 break（後続も消えない）
minReaderEpoch = EpochDomain::getMinReaderEpoch()（EpochDomain.h:214-248）
  = min を currentEpoch() で初期化し、depth>0 の reader が居る場合のみその epoch まで下げる
  ⇒ **active reader 0 のとき minReaderEpoch == currentEpoch**
⇒ entry.epoch == currentEpoch で enqueue された entry は
   **globalEpoch が 1 回前進するまで絶対に reclaim できない**。
```

### d. shutdown 中に decrement 条件が実行されるか

```text
publishEpoch()（= globalEpoch++）の呼び出し箇所は以下のみ:
  AudioEngine.CtorDtor.cpp:228   dtor 冒頭
  AudioEngine.CtorDtor.cpp:249   dtor graceful drain の毎 tick
  AudioEngine.Processing.ReleaseResources.cpp:305  releaseResources graceful drain の毎 tick
  AudioEngine.Processing.ReleaseResources.cpp:323  releaseResources graceful-timeout 経路
  AudioEngine.Publication.cpp:18/28（publish/advanceRetireEpoch 経由・通常運転）
  AudioEngine.h:5306（テスト用 helper driveWorldRetirementReclaimForMeasurement）
⇒ **:376 ReclaimComplete 以降に releaseResources 内で publishEpoch は呼ばれない**。
⇒ :568-588 で enqueue された entry は :615 waitForDrain(2000) の間に
   epoch 前進が起きず、`deferredDeletionQueue.reclaim` で消えない。
⇒ waitForDrain の loop（Threading.cpp:235-244）は `drainDeferredRetireQueues(true)` のみを呼び、
   `AudioEngine.Retire.cpp` に publishEpoch は **存在しない**（grep 0 件）。
```

**同一機構の既存記述（in-repo 裏付け）** — `AudioEngine.h:5302-5303`:

```text
//   publish 時に markRetireEpoch() で epoch は進行済みだが、enqueue 時点の epoch は「現在 epoch」のため、
//   もう一度 publishEpoch() で進めてから tryReclaim しないと isOlder(entry.epoch, minReaderEpoch) が偽になる。
```

⇒ この現象は **新規 defect ではなく、source に既知・明記された epoch 整合の性質**である。

### e. 最終回収（安全性）

```text
~AudioEngine（CtorDtor.cpp）
  :228 publishEpoch()                       ← epoch 前進
  :241-251 graceful drain（毎 tick publishEpoch()+tryReclaim()）→ ここで全件 reclaim 可能
  :286 m_retireRouter->drainAll()（readers==0 時・epoch 非依存の強制 drain）
  :291 drainAllQuarantineStore()（stuck reader fallback・epoch 非依存）
⇒ releaseResources が timeout で抜けても、dtor が必ず強制回収する（pointer ownership は保持）。
```

## 6. pendingReclaimHandles_ Lifecycle

```text
生成（push_back）:
  AudioEngine.h:4611-4615/4621-4624  requestReclaimHandle(): epoch 不安全 or requestReclaim false
  AudioEngine.Retire.cpp:119-131    drainDeferredRetireQueues 再試行でまだ不安全
移動（swap out → 再試行）:
  AudioEngine.Retire.cpp:93-133     pending へ swap → isRetired かつ retireEpoch<minReaderEpoch なら
                                    requestReclaim → 失敗/不安全なら再 push_back
消去:
  requestReclaim（Coordinator）成功時に slot が Reclaimed になり、その handle は再登録されない
読み（drain 判定）:
  AudioEngine.Threading.cpp:198-202 pendingReclaimHandles_.empty()  … isFullyDrained の P2
```

**独立性（Step1-B）**:

```text
- pendingReclaimHandles_ は「handle 参照」であり、DeferredDeletionQueue（pointer ownership）とは別系統。
- P2 != empty かつ P3 == 0 は起こり得る（handle の slot が Reclaimed 化待ちでも EBR entry がない）。
- P3 != 0 かつ P2 == empty も起こり得る（EBR entry はあるが handle は既に Reclaimed）。
⇒ 両者は独立であり、collectDrainAudit が P3 のみを露出する（P2 は非露出）。
  本 Step では P2 を観測できないため、P3 を主候補として扱い、P2 は R33-C 候補として残す。
```

## 7. Default vs Recovery Vehicle Comparison

| 指標 | D3 vehicle（Recovery 3） | D0 default（Recovery 0） | 取得元 |
| --- | ---: | ---: | --- |
| routerPending（post-shutdown） | 2 | 1 | vehicle: collectDrainAudit／default: shutdown_trace.sh3_pendingRetire |
| blockingReason | Unknown(9) | Unknown(9) | collectResult／trace |
| phase / completed / violations | ShutdownComplete / 1 / 0 | ShutdownComplete / 1 / 0 | 同上 |
| publishedCount | 6 | **非露出**（§7 注） | collectDrainAudit |
| retiredCount | 5 | **非露出** | 同上 |
| activeWorldCount | 1 | **非露出** | 同上 |
| published − retired | 1 == activeWorlds ✅ | — | 整合（World 会計の破綻なし） |
| activeReaders / stuck | 0 / 0 | 0 / 0 | collectDrainAudit／trace sh1/sh4=0 |

**§7 注（正直な制約）**: default path は `published/retired/activeWorlds` を post-shutdown に
露出しない。R33-A は test-only vehicle 変更を原則禁止しているため、**D0 側は routerPending と
trace（sh1-sh6）に限定**される。`evidence/world_lifecycle_audit.json` は直近の
**別 engine**（publishedCount=1）の残置であり、D0 の値ではない（比較に使用しない）。

- `published − retired == activeWorlds`（6−5=1）は World 会計の整合を示す
  （= World リークや二重 retire の**観測上の兆候はない**。doubleRetire の直接観測は P8 非露出のため不能）。

## 8. Residual Item Origin

source から特定できる**クラスと機構**：

```text
residual = DeferredDeletionQueue 内の EBR entry
  うち、VerifyDrained terminal disposition で epoch=currentEpoch として enqueue されたもの。
構成要素（enqueue 箇所）:
  (i)  clearPublishedRuntimeSnapshotsNonRt が返す clearedWorld
       → retirePublishedRuntimeWorldNonRt(clearedWorld, true)（ReleaseResources.cpp:537-543）
  (ii) DSPLifetimeManager::retire(activeDSPToDestroy)（:570-578）
  (iii) DSPLifetimeManager::retire(fadingDSPToDestroy)（:579-587）
D0=1 / D3=2 の差は (i)〜(iii) のうち非 null だった本数と、
  graceful drain（:285-311, epoch 前進あり）で先に消えた本数の合計差として説明できる。
```

**個体識別（which item）は未確定**：

```text
理由: DeferredDeletionQueue の entry は ptr/deleter/type を持つが、
  collectDrainAudit / isFullyDrained は size のみを露出し、per-entry の identity を
  持たない。retire ごとの entry 計数・種別（World/Generic）の観測点も無い。
⇒ 「1 件が World か DSP か」「どの generation か」は production observability 無しには確定不能。
  これは R33-A の適用範囲外（R33-C: observability gate）。
```

## 9. Recovery Causality Separation

```text
Recovery obligation（T1）と shutdown residual（T2/T3）は別系統である。
  系統1: recovery obligation  = RecoveryAdmissionTable.liveCount_（Coordinator 内部）
         → isFullyDrained の P22 としてのみ関与。R32 では episode ごとに resolve 済み
           （3 episode が seq+1 で完遂、P1/P15 等の観測は 0 と整合）。
  系統2: shutdown residual    = DeferredDeletionQueue（EBR pointer ownership）
         → P3。terminal disposition で enqueue される。

分離の決定的根拠:
  D0（Recovery 0）でも blockingReason=Unknown(9)（= markTimedOut）が発生しており、
  timeout の発生自体は Recovery の有無に依存しない。
  D3 の +1（2 vs 1）は「terminal disposition で非 null だった DSP/World が 1 本多い」
  ことの反映であり、Recovery 専用の retire 経路ではない
  （recovery publish も通常 publish と同じ PublicationExecutor→publish 経路で、
   旧 World を retire するだけ）。
⇒ Recovery-origin publish が shutdown drain residual の原因である、という因果は
   source 上も実測上も支持されない。
```

## 10. Production Change Boundary

| 項目 | 判定 |
| --- | --- |
| production fix の必要性 | **不要**（分類完了。timeout は設計が許容し、dtor が強制回収） |
| production source 変更 | **0**（本 Step では行わない） |
| 新 getter / counter | **0** |
| shutdown / Retire / Reclaim semantics 変更 | **0** |
| Recovery semantics 変更 | **0** |
| waitForDrain(2000) 拡張 | **行わない**（R33-A 禁止事項） |
| force reclaim / emergency drain 変更 | **行わない** |

- 分類に必要な情報は既存 source と既存 evidence で足りた（§2-§9）。
- 個体識別や P2/P8 の分解まで求める場合のみ observability が必要 → R33-C（実装しない）。

## 11. R33-A Gate

```text
[1] waitForDrain の全 predicate を source から列挙              … PASS（P1-P22・§3）
[2] timeout を発生させる predicate の特定                       … PASS（P3 retireDepth / routerPending）
[3] routerPending の生成元の特定                                … PASS（enqueueWithRetry・epoch=currentEpoch・§5b）
[4] routerPending の消費条件の特定                              … PASS（reclaim の isOlder + FIFO・§5c/d）
[5] pendingReclaimHandles の lifecycle の特定                    … PASS（§6・生成/移動/消去/読み）
[6] default の residual の説明                                  … PASS（クラス・機構）／個体は R33-C
[7] Recovery 3 episodes の +1 の説明                            … PASS（クラス・機構）／個体は R33-C
[8] Recovery obligation と shutdown residual の分離             … PASS（§9）
[9] production 修正なしでの原因分類                             … PASS（§10）
→ R33-A-PASS（限定: 個体識別 require R33-C observability）
```

- 原因の一段落（source 検証済み）:

```text
releaseResources は :376 ReclaimComplete の後（VerifyDrained :537-588）で
World/DSP の terminal disposition を **epoch = currentEpoch** で EBR enqueue する。
しかし :376 以降に publishEpoch() は呼ばれず、minReaderEpoch は readers==0 のため
currentEpoch に張り付く。したがって entry.epoch == minReaderEpoch となり
isOlder(...) が偽 → :615 waitForDrain(2000) の loop（epoch 前進なし）では reclaim できない
→ timedOut → markTimedOut(Unknown) → blockingReason=9。
（時間を延ばしても消えない。消すには epoch 前進が必要 — §5d の設計事実。）
```

## 12. R33-B/C Decision

```text
R33-A-PASS のため、次は R33-B（契約明確化）を選ぶのが妥当。
R33-C（observability 追加）は「個体識別」または「P2/P8 分解」を要件とする場合のみ。

R33-B 候補（契約明確化・実装は次段で判断）:
  B1. `ReleaseResources` の ReclaimComplete(:376) の意味を
      「graceful drain 区間の終端」と明記し、「shutdown 全体の reclaim 完了」ではないことを固定する。
  B2. VerifyDrained の terminal disposition（:537-588）は
      「epoch 前進を伴わないため waitForDrain では消えない」ことを契約として明記する
      （AudioEngine.h:5302-5303 の記述を shutdown 本体コメントへ展開）。
  B3. `blockingReason != None` は「graceful drain 未完（設計許容）」を示す診断であり、
      ownership 失敗ではないことを §15-P-4-7 の結論と結び、テストの成功条件に
      `completed` 単独を使わない（blockingReason を併記する）方針を固定する。

R33-C 候補（observability・実装しない）:
  C1. per-entry identity（World/Generic・generation）の観測点。
  C2. pendingReclaimHandles_ の件数露出（P2）。
  C3. terminalReclaimResident と retire 種別の分解（P8）。
```

## 13. Next Step

```text
R33-A-PASS（限定付き）で停止。
次は R33-B（契約明確化 gate）を推奨。実装（production）には進まない。
R31-C（per-episode T1 / E2 winner）と Full-pipeline Measurement は引き続き開始しない。
```

- source は R27 production＋R23/R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。
