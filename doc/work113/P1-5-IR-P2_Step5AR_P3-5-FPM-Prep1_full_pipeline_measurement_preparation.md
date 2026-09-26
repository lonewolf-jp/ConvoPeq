# P1-5-IR-P2 — Step 5-AR / P3-5-FPM-Prep1: Full-pipeline Measurement Preparation Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-FPM-Prep1）
- **種別**: **design / read-only gate**。
  production 0・test 0・CMake 0・getter/counter 0・shutdown semantics 0・
  epoch/reclaim 0・Recovery semantics 0・R33-C 0・R31-C 0・実計測 0。
- **判定**: **FPM-PREP-PASS**（測定プロトコル固定のみ。実装・実行なし）。
- **入力**: R33-A-PASS（原因機構）＋ R33-B-PASS（解釈契約）＋ R34-PASS（判定 C／P3-5 CLOSED）。
- **STOP 後停止**: 本 gate 完了後は実装・実計測に進まず STOP。実計測 GO 判断は次の別 gate。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（R33-A／R33-B／R34 と同一）
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（継承）
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持（変更なし）
ConvoPeq.md = 2026-09-23 23:33:38, Length 5535334（R33-A の authority と同一物を再確認）
  実測: (Get-Item ConvoPeq.md).LastWriteTime 2026-09-23 23:33:38（本 Step で再実測）
production diff = R27 のみ／test diff = R23+R30+R32 vehicle／CMake clean（継承・本 Step 無変更）
FPM-PREP-1 の差分 = 本ドキュメントのみ（read-only）
```

検証手段は Read／Grep のみ。ビルド・テスト実行・計測は行っていない。

---

## 2. Latest ConvoPeq.md Reconciliation

指示の 7 ファイル × A-G 経路を src 実測で再照合（いずれも R33-A/B/R34 の主張と一致・変更なし）：

### A. publish 経路（Publication.cpp）

```text
markRetireEpoch() :16-19 → m_retireRouter->publishEpoch()
advanceRetireEpoch() :26-29 → m_retireRouter->publishEpoch()（ReleaseResources.cpp:263 の呼出元）
currentRetireEpoch() :21-24 → currentEpoch()（観測のみ）
makePublishDecisionSnapshot（:50-101）：oldHandle==null は crossfade skip（B4）。
  crossfade 要否は CrossfadeAuthority.evaluate＋Critical 抑制。
```

### B. crossfade 経路

```text
AudioEngine.h:2508 crossfadeRuntime_（CrossfadeRuntime）
ReleaseResources.cpp:132 reconfigure pass reset／:198 terminal pass Audio 停止後 reset／
  :407-410 EmergencyDrain 要求時のみ force reset／:731 activeCrossfadeCount = isPending()?1:0
Threading.cpp:119 collectDrainAudit.activeCrossfadeCount = isPending()?1:0
DSPTransition.h:143 start／:152 complete（publish 側の遷移駆動）
```

### C. retire 経路（Retire.cpp・DSPLifetimeManager.cpp）

```text
drainDeferredRetireQueues（Retire.cpp:45-139）= tryReclaim＋m_coordinator.reclaim(minReaderEpoch)
  ＋pending handle 再試行。publishEpoch を含まない（R33-B §5 継承）。
DSPLifetimeManager::retire(dsp, publicationEpoch)（:40-77）：
  epoch = (publicationEpoch>0) ? publicationEpoch : router_->currentEpoch()
  → enqueueWithRetry(dsp, destroyDSPCoreNode, epoch, Generic)
retireByHandle（:79-137）：epoch = router_->currentEpoch()（:120）。
  VerifyDrained 経路（ReleaseResources.cpp:577／586 retire(dsp)）は publicationEpoch=0 呼び
  ⇒ currentEpoch enqueue（R33-A §5b 継承）。
```

### D. shutdown 経路（ReleaseResources.cpp）

```text
:615 waitForDrain(2000,2) → :616 timedOut → :622 drainPendingRetireIntentsForShutdown
→ :624-633 markTimedOut（唯一の production caller・markFailed caller 0）
→ :654-662 timeout 後も safe tryReclaim のみ（drainAll 禁止）
→ :664 finalizeShutdown(timedOut)（二段構え・timedOut でも retire 実行）
→ :724 markShutdownComplete（Coordinator 側・別状態機械）
→ :744 transitionTo(ShutdownComplete)（無条件・terminal-only skip で TimedOut 後も許可）
```

### E. isFullyDrained()（T2 authority）

```text
AudioEngine::isFullyDrained（Threading.cpp:153-213）：
  P1 !hasDeferredCommit／P2 pendingReclaimHandles_.empty()／P3 retireDepth==0／
  P4 lifetimeRetireIntentPending==0／P5 ringResident==0／P6 dspQuarantineResident==0／
  P7 retireQuarantineResident==0／P8 terminalReclaimResident==0／
  ＋ runtimePublicationBridge_.isFullyDrained()（Coordinator P9-P22 委譲）
ShutdownScheduler::isFullyDrained（Coordinator.cpp:511-562）：
  P9 swapPending／P10-P12 queue empty／P13 recoveryIntentQueue／P14-P20 backlog/residency／
  P21 !recoveryAdmissionPending_／P22 liveLogicalRecoveryObligationCount()==0（:561）
```

### F. collectResult()（ISRShutdown.cpp:161-181）

```text
result.completed = (phase_ == ShutdownPhase::ShutdownComplete)（のみ）
result.blockingReason／transitionViolations／lateCallback／postStopEnqueue は独立フィールド。
⇒ completed は phase の別名。drain predicate を見ていない（R33-B §8 継承）。
```

### G. collectDrainAudit()（Threading.cpp:109-151）

```text
露出：pendingPublication／pendingRetire／activeCrossfadeCount／routerPendingRetire／
  deferred／quarantine／World（active／published／retired）／readers／stuck／
  healthState／EBR visibility／overflowRingResident。
非露出：P2 pendingReclaimHandles_／P8 terminalReclaimResident（isFullyDrained の述語だが
  audit 快照に含まれない・R33-A/B 継承）。
isAllZero() は監査ログ専用（RuntimeDrainAudit.h:77-84「shutdown 完了判定には使用しない」）。
```

⇒ R34 契約の前提に現行 source の変更なし。再照合 PASS。

---

## 3. Terminal Contract（R34 継承・再固定）

```text
T3a = getPhase()==ShutdownComplete ⇔ collectResult().completed==true
T3b = T3a ＋ blockingReason==None ＋ isFullyDrained()==true
T2  = isFullyDrained()==true
T1  = liveLogicalRecoveryObligationCount()==0（P22）
T3b ⇒ T2 ⇒ T1／T3a ↛ T2／T3a ↛ T1
ShutdownComplete ≠ FullyDrained
```

### terminal success authority（本 gate 固定）

```text
T3b =
    phase == ShutdownComplete
 && completed == true
 && blockingReason == None
 && transitionViolations == 0
 && isFullyDrained() == true
```

同一 shutdown episode について 5 点を同時に取得できる構成を vehicle に要求する
（§4）。`completed` 単独・`phase` 単独・R30/R32 型 `phase && completed && violations==0` の
いずれも success 判定として禁止（§3 の禁止コード形を継承）。

---

## 4. Measurement Vehicle Design

### 4.1 設計原則

- 既存 D167／D169／R30-R32 vehicle を terminal-success vehicle としてそのまま再利用しない
  （R34 §8：いずれも R32 型状態を PASS 認識し得る）。
- 新 vehicle は **同一 shutdown episode** について T3b 5 点セットを取得する。
  取得子はいずれも既存 public API（新規 getter／counter なし）：
```text
phase:                e.isrShutdownRuntime().getPhase()            （ISRShutdown.h:199）
completed:            e.isrShutdownRuntime().collectResult(h,0)    （ISRShutdown.h:224）
blockingReason:       同上 .blockingReason                        （同上）
transitionViolations: 同上 .transitionViolations                   （同上）
isFullyDrained:       e.isFullyDrained()                           （AudioEngine.h:1600）
```
- `collectDrainAudit()`（AudioEngine.h:1604）は diagnostic evidence として同時取得するが、
  T3b の authority にはしない（§8 契約）。

### 4.2 禁止判定形（vehicle 設計契約）

```cpp
// × 禁止：completed 単独
if (result.completed)
    success = true;
// × 禁止：phase 単独
if (phase == ShutdownPhase::ShutdownComplete)
    success = true;
// × 禁止：R30/R32 型
terminalOk = phaseComplete && completed && (violations == 0);
```

正形は T3b 5 点セットの conjunction のみ（§3）。

---

## 5. M0 Control

```text
入力：Recovery なし／publication 最小（startup rebuild のみ）／crossfade 最小／retire 最小
手順：h.start(48000, 512) → startup rebuild settle（seq 前進＋backlog 0 確認）→ h.stop()
取得：T3b 5 点セット＋drain audit（§8 schema）
目的：shutdown terminal baseline
```

注意：R32 で Control（D0・Recovery 0）でも `Unknown` が出た実績があるため、
M0 は「正常系基準」ではなく **baseline（比較の起点）** として扱う。
M0 で `INVALID-TERMINAL` が出ることは vehicle 設計上許容し、その場合も run 分類に従い記録する
（§9）。baseline の INVALID は「測定値が悪い」ではなく terminal contract 未達の事実である。

---

## 6. M1 Single Recovery

```text
入力：Recovery episode = 1（R30 D1 方式・submitRecoveryIntent 直呼び・test-only）
手順：h.start → settle → episode 1 件（pre snap → submit → seq 前進待ち＋dCoord=0・dCmt=0・
      dTake=0・dBld=0 確認・R30 §8 期待形）→ settle → h.stop()
取得：episode 観測（§8 補助 counter）＋T3b 5 点セット＋drain audit
目的：Recovery が入った場合でも T3b を正しく判定できること
```

制約：R31-C の per-episode E2 winner は本 vehicle の必須条件にしない（R34 §6 継承）。
episode 観測は R30 既存 snap（req／que／dup／take／bld／cmt／coord／drp／seq／blo・
P1PolyphaseGainCharacterization.cpp:1071-1101）の再利用で足りる。

---

## 7. M2 Full Pipeline

```text
入力：複数 publish（SR 変更 re-prepare 等）＋crossfade＋retire＋Recovery＋shutdown の組合せ
手順：h.start → settle → publish 群（各 publish で seq 前進確認）→ Recovery episode 群
      → crossfade settle（isPending()==false 確認）→ h.stop()
取得：publish 群の補助 counter 軌跡＋T3b 5 点セット＋drain audit
目的：実運用に近い publication → retire → shutdown の一連の pipeline integrity
```

範囲：測定対象は **pipeline integrity** のみ（§10）：

```text
Input → Intent → Build → Publication → Crossfade → Retire → Shutdown → Terminal verification
```

WORK105 の conv garbage／NUC／limiter attribution は本 gate の成功条件に混ぜない。
M2 は integrity vehicle であり、音質・原因帰属 vehicle ではない。

---

## 8. Measurement Evidence Schema

各 run について以下を 1 record として保存する（形式は JSON lines を推奨・確定は実装 gate）：

### Shutdown（authority・T3b 5 点＋付帯）

```text
phase                ：getPhase() の int 値＋名（ShutdownComplete 期待）
completed            ：collectResult().completed（0/1）
blockingReason       ：collectResult().blockingReason（int＋名・None=0 期待）
transitionViolations  ：collectResult().transitionViolations（0 期待）
lateCallback          ：collectResult().lateCallbackCount（参考）
postStopEnqueue       ：collectResult().postStopEnqueueCount（参考）
```

### Drain（authority＋diagnostic の区別付き）

```text
[T2 authority]
isFullyDrained        ：e.isFullyDrained()（0/1・T3b の第5条件）
[diagnostic evidence]
routerPendingRetire   ：audit.routerPendingRetire（P3）
pendingPublication    ：audit.pendingPublication（P1）
pendingRetire         ：audit.pendingRetire（P4）
activeCrossfadeCount  ：audit.activeCrossfadeCount
deferredPublish       ：audit.deferredPublish
quarantineResident    ：audit.quarantineResident
activeWorlds          ：audit.activeWorldCount
published             ：audit.publishedCount
retired               ：audit.retiredCount
activeReaders         ：audit.activeReaderCount
stuckReaders          ：audit.stuckReaderCount
overflowRingResident  ：audit.overflowRingResident
```

契約（R33-B/R34 継承）：

```text
isFullyDrained()   → T2 authority
collectDrainAudit() → diagnostic evidence（authority ではない）
```

### 補助 counter（R27/R30/R32 既存・補助証拠のみ）

```text
publication sequence（getLastCommittedPublicationSequence）
coordinator take（getCoordinatorTakeCount）／commit（getRebuildCommitEnqueueCount）／
build（getRebuildBuildResultCount）／take（getRebuildTakeCount）／backlog／drop 等
```

`counter == 期待値` だけで shutdown success を判定しない。terminal authority は T3b のみ（§6 指示継承）。

---

## 9. VALID / INVALID / ABORTED / CRASH Classification

### VALID

```text
T3b == true（5 点セット全成立）
```

### INVALID-TERMINAL

```text
T3a == true かつ T3b == false
```

例：

```text
ShutdownComplete／completed=true／blockingReason=Unknown／isFullyDrained=false
→ INVALID-TERMINAL
```

これは「測定値が悪かった」ではなく **terminal contract を満たしていない run** である。
M0 baseline で INVALID が出る場合も同分類（§5）。

### ABORTED

vehicle 自体が所定の measurement sequence を完遂しなかった場合
（例：startup 失敗・episode の seq 前進待ち timeout・settle 未達で h.stop() に進めない）。

### CRASH / PROCESS FAILURE

プロセス異常終了等（0xC0000005 等・exit code 非 0・harness 応答喪失）。
terminal 5 点セットが取得できないため T3b 判定不能として分類する。

---

## 10. Existing Vehicle Reuse Boundary（変更せず・分類のみ）

R34 §8 の分類を継承し、measurement 用途での再利用範囲を確定する：

| vehicle | 再利用可 | 再利用不可 |
| --- | --- | --- |
| D167 terminal（PublishPipelineIntegrationTests.cpp） | admission closure 部分（Closed／tryAdmit reject） | terminal success 判定（blockingReason・drain 未参照） |
| D169-2-5／2-6 | collapse regression／stress 部分 | terminal success 判定（completed 未参照） |
| R30/R32 vehicle（P1PolyphaseGainCharacterization.cpp） | episode 観測（seq／coord／cmt／take／bld snap・drain emit） | terminalOk 判定（R30/R32 型・T3b 非充足） |
| R30 STOP-E 撤去対象（Running 中 waitForDrain 等） | なし（R31 契約違反のため再利用禁止） | 全体 |

新 measurement vehicle は T3b 5 点セット取得部を新規設計とし、
episode 観測部のみ既存 snap 関数（`p1RecReadSnap`／`p1RecDrainFields` 相当）を流用可とする
（流用は実装 gate の判断・本 gate ではコード変更なし）。

---

## 11. Prohibited Changes（本 gate で禁止・いずれも未実施）

```text
× production 変更 × test 変更 × CMake 変更 × getter／counter 追加
× shutdown 変更 × waitForDrain 変更 × publishEpoch 追加 × reclaim 変更 × Recovery 変更
× R33-C 実装 × R31-C 実装 × 実計測 × buzz 測定 × limiter attribution × NUC attribution
× measurement vehicle 作成のための既存 test source 書換え
```

必要な変更が見つかった場合は STOP して別の実装 gate へ分離する（本 gate では該当なし・§12）。

---

## 12. FPM-PREP Gate

```text
[1] 最新 ConvoPeq.md との source reconciliation PASS … PASS（§2・7 ファイル A-G・差分なし）
[2] T3b 5 点セット固定 … PASS（§3・terminal success authority）
[3] measurement vehicle 設計確定 … PASS（§4・既存 public API のみ・禁止判定形付き）
[4] M0/M1/M2 scenario 確定 … PASS（§5-§7・pipeline integrity 範囲）
[5] measurement validity 分類確定 … PASS（§9・VALID／INVALID-TERMINAL／ABORTED／CRASH）
[6] shutdown evidence schema 確定 … PASS（§8・authority／diagnostic 区別付き）
[7] diagnostic と authority の区別確定 … PASS（§8・isFullyDrained／collectDrainAudit）
[8] 既存 vehicle の再利用範囲確定 … PASS（§10・terminal 判定は再利用不可）
[9] production/test/CMake 変更 0 … PASS（§11・本ドキュメントのみ）
[10] 実計測 0 … PASS（実行なし）
→ FPM-PREP-PASS
```

---

## 13. Next Gate

```text
R33-A PASS → R33-B PASS → R34 PASS／P3-5 CLOSED → FPM-PREP-1 PASS（本 gate）
    │
    └── Full-pipeline Measurement 実行 gate（別 gate）
          - 本 gate の vehicle 設計・T3b 契約・§8 schema に従い初めて実計測
          - M0→M1→M2 の順に実行し、各 run を §9 分類で記録
          - buzz／limiter／NUC attribution は含まない
```

- source は R27 production＋R23/R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
- 本 gate はここで停止する。
