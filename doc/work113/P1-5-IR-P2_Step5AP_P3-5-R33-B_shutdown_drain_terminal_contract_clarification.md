# P1-5-IR-P2 — Step 5-AP / P3-5-R33-B: Shutdown Drain / Terminal Contract Clarification Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-R33-B）
- **種別**: **contract / read-only gate**。
  production 0・test 0・CMake 0・getter/counter 0・shutdown semantics 0・
  epoch/reclaim 0・Recovery 0・Full-pipeline measurement 禁止・R31-C observability実装 禁止。
- **判定**: **R33-B-PASS**（contract clarification のみ。実装変更なし）。
- **入力**: R33-A-PASS（Step5AO）の原因機構確定を受け、その shutdown state の**解釈**を契約として固定する。
- **STOP 後停止**: 本 gate 完了後は実装に進まず STOP。R33-C は別 gate 判断。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（R33-A と同一）
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（R33-A 継承）
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持（変更なし）
ConvoPeq.md = 2026-09-23 23:33:38, Length 5535334（R33-A の authority と同一物を再確認）
  Get-Item LastWriteTime: 2026/09/23 23:33:38（本 Step で実測）
production diff = R27 のみ／test diff = R23+R30+R32 vehicle／CMake clean（R33-A 継承・本 Step 無変更）
R33-B の差分 = 本ドキュメントのみ（read-only）
```

ConvoPeq.md 照合（最新 source 再確認・必須）：

```text
ConvoPeq.md:18117  const bool drainedWithinBudget = waitForDrain(2000, 2);
ConvoPeq.md:18134  shutdownRuntime_.markTimedOut(reason);
ConvoPeq.md:18246  shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase::ShutdownComplete);
ConvoPeq.md:35336  result.completed = (phase == ShutdownComplete)（collectResult 定義）
```

⇒ 指示書の前提「`waitForDrain(2000,2)` 後に `markTimedOut()` があり、その後の terminal transition が存在する」は
最新 source でも成立（src では ReleaseResources.cpp:615 → :632 → :744 の順）。

---

## 2. Latest ConvoPeq Source Reconciliation

R33-A §2 の shutdown chain が最新 source でも不変であることを再確認（差分なし・行番号は src 実測）：

```text
releaseResources()（terminal pass）
  :95   setShutdownPhase(StopAcceptingWork) + :96 transitionTo(AudioStopped)
  :109  closeAdmission()
  :236-238 shutdownCoordinatorLoop() + stopRebuildThread()
  :250-256 joinProducers retry（timeout 時 :253 waitForDrain(100,1)）
  :259  transitionTo(ObserverDrained)／:264 RetireClosed／:265 EpochSettled
  :277  escalateAllRetires(Critical)
  :285-311 Graceful Drain（≤5000 ms・毎 tick :305 publishEpoch() + :306 tryReclaim()）
  :375  drainDeferredRetireQueues(true)
  :376  transitionTo(ReclaimComplete)                ← ★ B1 の対象
  :383  transitionTo(EmergencyDrain)                ← 常に経由（単一遷移・処理は条件付き :384）
  :431-472 Quarantine 全スロット強制解放（PR2・Phase 3）
  :475  transitionTo(VerifyDrained)
  :502-517 active/fading handle retire + tryShutdownQuiescentReclaim
  :537-543 world clear → retirePublishedRuntimeWorldNonRt(clearedWorld,true)  ← ★ 再 enqueue
  :554-555 if readers==0 → drainAllQuarantineStore()
  :568-588 DSPLifetimeManager::retire(active/fadingDSPToDestroy)              ← ★ EBR enqueue
  :599-600 clearDeferredForShutdown()／:605-606 resetRedriveBudget()
  :615  waitForDrain(2000,2)                          ← ★ B2 の対象
  :616  timedOut=!drainedWithinBudget
  :622  drainPendingRetireIntentsForShutdown()
  :624-633 if (timedOut) markTimedOut(Unknown)       ← blockingReason=9 の発生源（§7）
  :654-662 if (!drained||!isFullyDrained()) → drainDeferredRetireQueues(true)+tryReclaim
  :664  finalizeShutdown(timedOut)                    ← SnapshotCoordinator 二段構え（§8）
  :677-697 OwnerChannel residual → terminal retire chain
  :724  markShutdownComplete()（Coordinator 側）
  :744  transitionTo(ShutdownComplete)                ← ★ timeout 後も到達し得る（§8）
  :745  emitShutdownTrace()
```

- `markTimedOut` の caller は src 全体で **ReleaseResources.cpp:632 の 1 箇所のみ**
 （`markFailed` は caller 0。grep `markTimedOut\(|markFailed\(` で確認）。
- `transitionTo(ReclaimComplete/EmergencyDrain/VerifyDrained/ShutdownComplete)` は
  ReleaseResources.cpp の上記箇所のみ（AudioEngine 側 ShutdownPhase と ISRShutdown 側 phase は別 enum・§3 注）。
- R33-A の原因 chain（VerifyDrained → currentEpoch enqueue → epoch 不進 → P3 ≠ 0 →
  timeout → markTimedOut(Unknown) → finalize → ShutdownComplete）は最新 source と一致。

### R31-AM 記述の訂正（source 内矛盾ではなく文書の supersede）

R31（Step5AM §4 相当、lines 160-165）は「phase が TimedOut になったら
`transitionTo(ShutdownComplete)` は許可されず TimedOut に留まる ⇒ completed==false」と記述していた。
これは **現行 source と整合しない**ため R33-B で撤回する：

```text
ISRShutdown.cpp:124-152 transitionTo:
  allowed = (t==c || t==c+1) だが、t>c+1 でもスキップ区間が terminal のみなら許可。
  VerifyDrained(7) → markTimedOut は transitionTo を使わず直接 store で TimedOut(8)。
  TimedOut(8) → ShutdownComplete(10) は i=9 (Failed) が terminal のみ → allowed=true。
ISRShutdown.cpp:201-206 isTerminalPhase = {ShutdownComplete, TimedOut, Failed}。
ReleaseResources.cpp:744 は timedOut の有無に関わらず無条件に transitionTo(ShutdownComplete)。
```

⇒ **timeout 後も ShutdownComplete に到達し得る**（R32 実測 D3/D0 がその実例）。
R31-AM の該当段落は R33-B により superseded とする。STOP 条件 R33-B-C2（source 内矛盾）には当たらない
（矛盾は旧文書と source の間であり、source 内に矛盾はない）。

---

## 3. ReclaimComplete Contract（B1）

### 3.1 契約文

```text
ReclaimComplete
    =
    Graceful Drain 区間（:285-311）の終端マーカー

    ≠ shutdown 全体の ownership / EBR reclaim 完了
```

### 3.2 source 根拠

1. **:376 は final empty-state assertion ではない**。
   :376 の前後いずれにも `jassert(isFullyDrained())` / `jassert(pendingRetireCount()==0)` は存在しない。
   直後 :383 で無条件に EmergencyDrain へ遷移し、:431-472 の quarantine 解放、:475 の VerifyDrained が続く。
2. **ReclaimComplete 後に retire enqueue が存在する**（合法 sequence）：
```text
   ReclaimComplete(:376)
     → EmergencyDrain(:383, 通常 diagnostic only)
     → VerifyDrained(:475)
     → :537-543 clearPublishedRuntimeSnapshotsNonRt → retirePublishedRuntimeWorldNonRt
     → :570-588 DSPLifetimeManager::retire(active/fadingDSPToDestroy)
                 → router_->enqueueWithRetry(dsp, destroyDSPCoreNode,
                                             epoch = router_->currentEpoch(), Generic)
                    （DSPLifetimeManager.cpp:54-61／:120-124）
     → :599-600 clearDeferredForShutdown（deferred slot の authority 処分・EBR 経由）
     → :677-697 OwnerChannel drainAllNonRt → enqueueDeferredDeleteNonRtWithResult
                 → shutdownReclaim（Terminal 経由）
     → waitForDrain(:615)
```
3. AudioEngine 側 `ShutdownPhase`（AudioEngine.h:2804）と ISR `ShutdownPhase`
   （ISRShutdown.h:38-54）は別 enum であり、:376 は後者（`convo::isr::`）の遷移である。
   AudioEngine 側は :267 で `DrainRetire` に入ったまま :376 に対応する独自 phase を持たない
   ⇒ :376 を「shutdown 全体の reclaim 完了」と読む source 上の裏付けはない。

### 3.3 B1 判定

```text
[PASS] ReclaimComplete 後にも retire enqueue が存在する（:537-543／:570-588／:677-697）。
[PASS] ReclaimComplete は final empty-state assertion ではない（assert なし・後続 phase あり）。
```

---

## 4. VerifyDrained Terminal-Disposition Contract

```text
VerifyDrained(:475) の terminal disposition（:502-588）は
「epoch 前進を伴わない EBR enqueue」である。
したがってその直後の waitForDrain(:615) だけでは消えない entry を
合法的に生成し得る。これは defect ではなく設計上の性質である。
```

source 根拠：

- :502-517 handle retire（slot 遷移・冪等）＋ :570-588 authority retire（EBR enqueue・epoch=currentEpoch）。
- :537-543 world clear → `retirePublishedRuntimeWorldNonRt(clearedWorld,true)`
  → `enqueueDeferredDeleteNonRt` → shutdown 中は P-4 経路で `shutdownReclaim`
  （AudioEngine.h:4466-4476）→ Terminal 保持または即時破棄。
- in-repo の同一機構の既存記述（AudioEngine.h:5302-5303）：
```text
//   publish 時に markRetireEpoch() で epoch は進行済みだが、enqueue 時点の epoch は「現在 epoch」のため、
//   もう一度 publishEpoch() で進めてから tryReclaim しないと isOlder(entry.epoch, minReaderEpoch) が偽になる。
```
⇒ R33-A §5d の結論を契約として再掲する。**実装変更はしない**。

---

## 5. Epoch Advancement Contract（B2）

### 5.1 契約文

```text
VerifyDrained の terminal disposition によって current epoch に enqueue された entry は、
epoch advance がない限り、直後の waitForDrain() だけでは reclaim 完了しない。
```

### 5.2 source による適合確認（変更ではなく固定）

1. **reclaim 条件**（DeferredDeletionQueue.h:132／:257-260）：
```text
reclaim(entry) iff isOlder(entry.epoch, minReaderEpoch)
isOlder(a,b) = (int64_t)(a-b) < 0   ⇔ a < b
FIFO 厳守: 先頭が不可なら即 break（:171-175）。後続が reclaimable でも消えない。
```
2. **readers==0 の minReaderEpoch**（EpochDomain.h:214-248）：
```text
minEpoch = currentEpoch() で初期化し、depth>0 の非 quarantine reader のみで下げる。
⇒ active reader 0 のとき minReaderEpoch == currentEpoch。
```
3. **enqueue epoch**（DSPLifetimeManager.cpp:54-56／:120）：
```text
epoch = (publicationEpoch>0) ? publicationEpoch : router_->currentEpoch()
VerifyDrained 経路は publicationEpoch=0 呼び（:577／:586 retire(dsp)）⇒ currentEpoch。
```
4. したがって `entry.epoch == minReaderEpoch` ⇒ `isOlder == false` ⇒ reclaim 不可。
   消すには `publishEpoch()`（EpochDomain.h:193-202 globalEpoch++）が 1 回必要。
5. **:376 以降 releaseResources 内で publishEpoch は呼ばれない**：
   publishEpoch 呼出箇所は CtorDtor.cpp:228/249（dtor）、ReleaseResources.cpp:305/323
   （graceful drain 本体・timeout 経路）、Publication.cpp:18/28（通常運転）のみ。
   Retire.cpp の `drainDeferredRetireQueues` 内に publishEpoch は存在しない（grep 0 件）。
6. **waitForDrain 自体が epoch advancement mechanism ではない**（Threading.cpp:215-247）：
```text
while (!isFullyDrained()) { drainDeferredRetireQueues(true); sleep; }
drainDeferredRetireQueues（Retire.cpp:45-139）= tryReclaim + m_coordinator.reclaim(minReaderEpoch)
  + pending handle 再試行。publishEpoch を含まない。
⇒ 時間を延ばしても epoch-equal entry は消えない（R33-A §5d と同一結論）。
```

### 5.3 B2 判定

```text
[PASS] currentEpoch enqueue は epoch advance なしでは reclaim できない（§5.2-1〜4）。
[PASS] waitForDrain 自体が epoch advancement mechanism ではない（§5.2-6）。
[PASS] 既存実装はこの契約に適合している（変更不要・固定のみ）。
```

---

## 6. waitForDrain Contract

```text
waitForDrain(timeoutMs, pollIntervalMs)（Threading.cpp:215-247）
  = isFullyDrained()（Threading.cpp:153-213 + Coordinator.cpp:511-562 の P1-P22 展開は R33-A §3）
    が真になるまでの bounded poll。drain を「促進」するものではなく「観測」するものである。
  - epoch を進めない。reader を閉じない。admission を閉じない。
  - timeout（false return）は drain predicate のいずれかが残留したことの観測であり、
    ownership 喪失の証明ではない。
  - :253 waitForDrain(100,1)（joinProducers retry）と :615 waitForDrain(2000,2)（VerifyDrained）は
    同一関数・別 budget。いずれも epoch 非前進。
```

jassert 契約（Threading.cpp:221-229）：AudioStopped 以降でのみ呼ぶこと。
VerifyDrained からの呼出（:615）は契約内。`waitForDrainSignalOrTimeout`（Threading.cpp:259-264）は
CV 待機であり本 gate の対象外（:615 はこちらを使っていない）。

---

## 7. blockingReason Contract

### 7.1 定義域

ISRShutdown.h:59-71：

```text
None=0, PendingPublication, PendingRetire, ActiveCrossfade, DeferredPublish,
QuarantineResident, RouterPendingRetire, ReaderActive, ActiveBuilder, Unknown(=9)
```

`RuntimeDrainAudit::BlockingReason`（RuntimeDrainAudit.h:53-63）は別 enum（診断表示用・最大 7=Unknown）であり、
shutdown terminal diagnostic である `ShutdownBlockingReason` と混同しないこと。
`getPrimaryBlockingReason()`（RuntimeDrainAudit.h:65-74）は audit 快照からの推定であり、
:626-632 の markTimedOut reason 選択とは独立（後者は collectDrainAudit の stuckReader/rebuildThread のみ参照）。

### 7.2 `blockingReason = shutdown terminal diagnostic state`

- 保存は `markTimedOut`（ISRShutdown.cpp:72-112）／`markFailed`（:114-122）のみ。
  いずれも `transitionTo` をバイパスする直接 store（phase 上書き＋ blockingReason 保存＋統計）。
- `markTimedOut(reason=Unknown)`／`markFailed(reason=Unknown)` の default は Unknown
  （ISRShutdown.h:217-218）。
- 唯一の production caller は ReleaseResources.cpp:632：
```text
reason = Unknown;
if (stuckReaderCount>0) reason = ReaderActive;
else if (rebuildThreadIsRunning) reason = ActiveBuilder;
markTimedOut(reason);
```
⇒ **R33-A の具体例で `blockingReason=Unknown` は「VerifyDrained／waitForDrain の timeout が発生し、
  かつ stuck reader でも ActiveBuilder でもなかった」ことのみを示す。**

### 7.3 限定（推測禁止）

```text
blockingReason != None ⇒ drain/terminal condition に問題または timeout があった（timeout evidence）。
blockingReason != None ⇏ ownership failure。
blockingReason != None ⇏ Recovery failure。
blockingReason=Unknown ⇏ 原因特定（「Unknown の原因」をさらに推測しない）。
```

根拠：blockingReason の書込み条件は timedOut（:624）のみであり、pointer ownership／Recovery obligation の
状態を直接観測していない。Recovery 因果分離は R33-A §9（D0 でも同一 Unknown）を継承する。

### 7.4 B 判定

```text
[PASS] blockingReason != None を timeout evidence として扱える（書込み条件が timedOut のみ）。
[PASS] blockingReason=Unknown を ownership failure と解釈しない（source 上の書込み情報に所有権の述語なし）。
```

---

## 8. ShutdownComplete vs FullyDrained

### 8.1 契約表（R33-B 固定）

| 状態 | 意味 |
| --- | --- |
| `ShutdownComplete + blockingReason=None` | nominal shutdown completion／drain success |
| `ShutdownComplete + blockingReason!=None` | terminal state reached **with shutdown blocking/timeout evidence**（R32 D3 が実例） |
| `TimedOut` | timeout terminal indication。ただし現行 transition 契約上、後続 `ShutdownComplete` があり得る（§2 訂正） |
| `Failed` | failure indication（現行 caller 0・到達経路なし） |
| `completed=true` 単独 | **成功判定不可**（§8.2） |

### 8.2 `completed=true` は drain success の十分条件ではない

`collectResult`（ISRShutdown.cpp:161-181）：

```text
result.completed = (phase_ == ShutdownPhase::ShutdownComplete)   // のみ
result.blockingReason = blockingReason_（completed と独立）
result.transitionViolations／lateCallback／postStopEnqueue も独立フィールド
```

⇒ `completed` は phase の別名であり、drain predicate・blockingReason を見ていない。
R32 実測が反例を具体化する：

```text
phase=ShutdownComplete／completed=true／violations=0／
blockingReason=Unknown／fullyDrained=false（routerPending=2, P3≠0）
```

したがって今後の test／vehicle／gate では `phase／completed／blockingReason／
transitionViolations` をセットで扱い、drain 主張には `isFullyDrained()` または
`collectDrainAudit` の独立観測を要求する。**禁止事項**：今後の gate で `completed==true` を
drain 成功と解釈すること（指示書 §B3 の禁止を本契約として継承）。

### 8.3 finalizeShutdown の位置づけ

SnapshotCoordinator::finalizeShutdown（SnapshotCoordinator.h:62-72）は
`timedOut=true の場合も retire は実行（reclaim のみスキップ）`の二段構え。
⇒ timeout 後の ShutdownComplete 到達は「retire 済み・reclaim 未完」の状態を含み得る。
これは §8.1 の `ShutdownComplete + blockingReason!=None` 行と整合する。

Coordinator 側 `markShutdownComplete`（Coordinator.cpp:568-579）は
`isFullyDrained()==false` なら `Faulted` に遷移するが、これは Coordinator state であり
engine の `ShutdownPhase::ShutdownComplete`（:744）とは別状態機械である。
両者を同一視しないこと。

---

## 9. T1 / T2 / T3 Contract Reconciliation

### 9.1 定義（R33-B 固定・新規 API なし）

```text
T3a = Shutdown state machine reached ShutdownComplete
      （getPhase()==ShutdownComplete ⇔ collectResult().completed==true）
T3b = shutdown completed without blocking/timeout
      （T3a ＋ blockingReason==None ＋ drain predicate 成立。
        drain predicate の具体的構成は isFullyDrained() の P1-P22（R33-A §3）とし、
        新規 API は作らない）
T2  = runtime drain predicate satisfied（isFullyDrained()==true）
T1  = recovery obligations terminalized
      （liveLogicalRecoveryObligationCount()==0。Coordinator.cpp:561 の P22。
        resolve は RecoveryAdmissionTable::resolve の単一 −1 authority・D152 T3）
```

### 9.2 旧解釈の撤回

R31（Step5AM §5 lines 184-193）の旧解釈：

```text
T3 ⊇ T2 ⊇ T1（述語の包含）。T3 成立 ⇒ VerifyDrained 時点で T1 成立が entails。
```

は、少なくとも `phase==ShutdownComplete／completed==true` だけでは成立しない。
R32 D3 が反例：

```text
T3a（ShutdownComplete＋completed=1）は真だが、
T2（isFullyDrained）は P3≠0 により偽。
```

したがって：

```text
T3a ↛ T2（明示的に撤回・本 gate で固定）
T3a ↛ T1（T2 を経由しないため、T3a 単独から T1 の terminalization は entails されない。
          R31 の「T3 経由の T1 含意」記述は completed-only 解釈としては撤回。
          T1 の成立自体は別途 liveCount==0 で判定する）
T3b ⇒ T2（定義により。T3b は drain predicate 成立を含む）
T2 ⇒ T1（isFullyDrained が P22 liveCount==0 を含むため。Coordinator.cpp:561）
```

### 9.3 T3b の drain predicate 選択（source 確認済み）

T3b の「drain predicate の成立」は `AudioEngine::isFullyDrained()`（Layer 1 実測＋
Coordinator `isFullyDrained()` の P9-P22 委譲）とする。`collectDrainAudit().isAllZero()` は
監査ログ専用（RuntimeDrainAudit.h:77-84「shutdown 完了判定には使用しない」）であり、
T3b の authority にはしない。P2（pendingReclaimHandles）・P8（terminalReclaimResident）は
isFullyDrained の述語だが collectDrainAudit 非露出（§11）であるため、T3b 判定に使う場合は
isFullyDrained() を呼ぶこと（audit 快照の代用不可）。

---

## 10. Existing Test Assertion Audit（read-only・修正なし）

`completed==true` を成功条件として使っている箇所の列挙（修正はしない）：

1. **PublishPipelineIntegrationTests.cpp:657-663（D167 terminal closure）**：
```text
const auto result = collectResult(Healthy, 0);
if (!result.completed) FAIL;              // blockingReason 未参照
if (result.transitionViolations != 0) FAIL;
```
   - 前段 :647-651 は `getPhase()!=ShutdownComplete` のみ。
   - コメント :652-656 は「ShutdownComplete 到達＋admission Closed＋completed が terminal closure の契約」
     と明記し、`isFullyDrained()` を「広い判定」として意図的に外している。
   - ⇒ R32 型状態（ShutdownComplete＋completed＋TV=0＋blockingReason=Unknown＋fullyDrained=false）を
     **PASS と認識できてしまう**。既存 assertion は shutdown success の十分条件ではない（記録のみ）。

2. **同 :825-830（D169-2-5 terminal）／:986-991（D169-2-6 terminal）**：
```text
admission Closed ＋ phase ShutdownComplete のみ。completed／blockingReason／drain いずれも未参照。
```
   ⇒ 1 よりさらに弱い closure 判定（記録のみ）。

3. **P1PolyphaseGainCharacterization.cpp:1278-1283（R30/R32 vehicle terminalOk）**：
```text
terminalOk = phaseComplete && termResult.completed && (violations==0)
```
   - blockingReason は emit（:1268）のみで判定に不使用。
   - drain audit は emit のみで「補助・主判定には使わない」（:1275-1276）と明記。
   - ⇒ R32 D3（blockingReason=Unknown／fullyDrained=0）は `terminalOk=1` になる。
     **既存 vehicle の terminalOk は drain success の十分条件ではない**（記録のみ）。

4. **ISRSemanticValidationTests.cpp（coordinator 層）**：`markShutdownComplete` の drained／not-drained
   テスト（:352-448）は Coordinator state 機械の判定であり、engine の `completed` とは別層。
   本監査の対象外（混同しないこと）。

5. **StuckReaderFallbackDrainTests.cpp**：router 層の drain 不変条件テスト。
   engine terminal assertion を含まない（対象外）。

結論（記録のみ・test 変更なし）：

```text
既存 test の completed-only（＋phase＋TV）assertion は shutdown success の十分条件ではない。
R32 型状態を PASS と認識できてしまう箇所が少なくとも 3 系統（D167／D169-2-5・2-6／R30-32 vehicle）存在する。
```

---

## 11. Production Change Boundary

| 項目 | 判定 |
| --- | --- |
| production source 変更 | **0**（行わない・行っていない） |
| test 変更 | **0** |
| CMake 変更 | **0** |
| getter/counter 追加 | **0** |
| shutdown semantics 変更 | **0** |
| epoch/reclaim 変更 | **0** |
| Recovery 変更 | **0** |
| Full-pipeline measurement | **禁止・未実施** |
| R31-C observability 実装 | **禁止・未実施** |
| publishEpoch 追加 | **行わない**（R33-A 禁止を継承） |
| waitForDrain 変更 | **行わない** |

検証手段は Read／Grep のみ。ビルド・テスト実行・計測は行っていない（read-only gate のため不要）。

---

## 12. R33-B Gate

```text
[1] ReclaimComplete が shutdown 全体の reclaim 完了を意味しない        … PASS（§3）
[2] VerifyDrained 後に terminal disposition が再 enqueue する          … PASS（§3-§4）
[3] currentEpoch enqueue は epoch advance なしでは reclaim できない    … PASS（§5）
[4] waitForDrain 自体が epoch advancement mechanism ではない           … PASS（§5-§6）
[5] timeout 後も ShutdownComplete に到達し得る                         … PASS（§2 訂正＋§8）
[6] completed=true は drain success の十分条件ではない                 … PASS（§8）
[7] blockingReason != None を timeout evidence として扱える            … PASS（§7）
[8] blockingReason=Unknown を ownership failure と解釈しない           … PASS（§7）
[9] ShutdownComplete と FullyDrained を別概念として固定できる          … PASS（§8）
[10] T3a ⇒ T2 の旧解釈を撤回できる                                    … PASS（§9・R32 反例）
[11] 既存 test の completed-only assertion を特定できる                … PASS（§10・3系統）
[12] production 変更なし                                               … PASS（§11）
→ R33-B-PASS（contract clarification のみ）
```

STOP 条件の確認：

```text
R33-B-C1（contract が一意に決まらない）：該当なし。全契約を source 行で一意に固定。
R33-B-C2（意味に source 内矛盾）：該当なし。旧文書（R31-AM）と source の差は supersede で解消。
R33-B-C3（clarification に production API 変更が必要）：該当なし。新規 API なしで固定。
R33-B-C4（test 成功条件修正に production semantics 変更が必要）：該当なし。R33-B は監査のみ。
```

---

## 13. R33-C Residual（実装しない・必要性が残った場合に別 gate）

R33-A の限定事項は依然として非露出（本 gate で確認・変更なし）：

```text
P2 pendingReclaimHandles_：
  isFullyDrained の述語（Threading.cpp:198-202）だが、
  collectDrainAudit（Threading.cpp:109-151）は露出しない。
P8 terminalReclaimResident：
  isFullyDrained の述語（Threading.cpp:190-191）だが、collectDrainAudit は露出しない。
  （RuntimeHealthMonitor snapshot :636 および backpressure telemetry :1769-1770 では読めるが、
   shutdown terminal evidence 経路ではない）
per-entry World/DSP identity：
  DeferredDeletionQueue entry（ptr/deleter/type・DeferredDeletionQueue.h:219-224 sizeApprox のみ露出）。
  per-entry の identity／generation の観測点なし。
generation：
  ShutdownRuntime shutdownGeneration_（ISRShutdown.h:351）は Proof identity 用であり、
  EBR entry generation の観測ではない。
```

R33-C を開く条件（いずれも本 gate では判断しない）：

```text
Option 1: R33-C observability design gate（per-entry identity／P2／P8 の露出設計）
Option 2: R31-C Recovery per-episode T1／E2 observability design
Option 3: P3-5 terminal contract closure 後に Full-pipeline Measurement
```

---

## 14. Next Step

```text
R33-A-PASS
    │
    ▼
R33-B-PASS（本 gate・contract clarification・production/test/CMake 変更 0）
    │
    ├── 次工程判断（本 gate では選択しない）
    │     Option 1: R33-C observability design gate
    │     Option 2: R31-C Recovery per-episode T1/E2 observability design
    │     Option 3: terminal contract closure 後の Full-pipeline Measurement
    │
    └── STOP（本 gate はここで停止・実装に進まない）
```

- `completed == true` を drain 成功と解釈することは今後の gate で禁止（§8 契約）。
- `publishEpoch()` 追加・`waitForDrain()` 変更・force reclaim／emergency drain 変更は行わない
  （Retire → Epoch Domain → Reclaim → Delete と Shutdown の Drain → Reclaim → Verify の責務分離を維持）。
- R31-C（per-episode T1／E2 winner）と Full-pipeline Measurement は引き続き開始しない。
- source は R27 production＋R23/R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
