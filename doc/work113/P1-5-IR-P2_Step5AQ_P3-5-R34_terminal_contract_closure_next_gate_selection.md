# P1-5-IR-P2 — Step 5-AQ / P3-5-R34: Terminal Contract Closure / Next-Gate Selection Audit

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-R34）
- **種別**: **read-only audit gate**。
  production 0・test 0・CMake 0・getter/counter 0・shutdown semantics 0・
  epoch/reclaim 0・Recovery semantics 0・observability implementation 0・
  Full-pipeline measurement 0（実計測なし）。
- **判定**: **判定 C — P3-5 terminal contract CLOSED**。
  R33-C は不要。R31-C は P3-5 closure の前提ではない（独立 track として残置）。
  次は **Full-pipeline Measurement 準備 gate**（本 gate では実計測を開始しない）。
- **入力**: R33-A-PASS（Step5AO・原因機構確定）＋ R33-B-PASS（Step5AP・解釈契約固定）。
- **STOP 後停止**: 本 gate 完了後は実装に進まず STOP。R33-C／R31-C／実計測はいずれも開始しない。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（R33-A／R33-B と同一）
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（継承）
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持（変更なし）
ConvoPeq.md = 2026-09-23 23:33:38, Length 5535334（R33-A の authority と同一物を再確認）
  実測: (Get-Item ConvoPeq.md).LastWriteTime 2026-09-23 23:33:38（本 Step で再実測）
production diff = R27 のみ／test diff = R23+R30+R32 vehicle／CMake clean（継承・本 Step 無変更）
R34 の差分 = 本ドキュメントのみ（read-only）
```

検証手段は Read／Grep のみ。ビルド・テスト実行・計測は行っていない。

---

## 2. Latest ConvoPeq.md Reconciliation

最新 source aggregate を基準にする条件に従い、R33-B の行番号主張を src 実測で再照合（差分なし）：

```text
ReleaseResources.cpp:615  waitForDrain(2000,2)
                   :616  timedOut = !drainedWithinBudget
                   :622  drainPendingRetireIntentsForShutdown()
                   :624-633 if (timedOut) markTimedOut(Unknown／ReaderActive／ActiveBuilder)
                   :664  m_coordinator.finalizeShutdown(timedOut)
                   :744  transitionTo(ShutdownComplete)（timedOut 有無に関わらず無条件）
                   :745  emitShutdownTrace()
ISRShutdown.cpp:168-169  completed = (phase_ == ShutdownComplete)（collectResult のみ）
ISRShutdown.cpp:124-152  transitionTo（terminal-only skip 許可）
ISRShutdown.cpp:201-206  isTerminalPhase = {ShutdownComplete, TimedOut, Failed}
Threading.cpp:153-213    isFullyDrained（Layer 1 実測＋ bridge 委譲）
Coordinator.cpp:511-562  ShutdownScheduler::isFullyDrained（P9-P22・P22 = liveCount==0 は :561）
Threading.cpp:215-247    waitForDrain（bounded poll・publishEpoch なし）
```

Shutdown Pipeline の `Drain → Reclaim → Verify` 構造も確認：

```text
Drain:    :285-311 Graceful Drain（毎 tick publishEpoch＋tryReclaim）
Reclaim:  :375 drainDeferredRetireQueues(true) → :376 ReclaimComplete
          :383 EmergencyDrain（通常 diagnostic only）→ :431-472 quarantine 解放
Verify:   :475 VerifyDrained → :502-588 terminal disposition（再 enqueue）
          → :615 waitForDrain → :632 markTimedOut → :664 finalize → :744 ShutdownComplete
```

⇒ R33-A/B の chain は最新 source と一致。ConvoPeq.md 系の内容とも矛盾なし。

---

## 3. R33-B Contract Revalidation

### A. Shutdown terminal（R33-B §8）

```text
ShutdownComplete ≠ FullyDrained
completed == true は shutdown 成功の十分条件ではない
```

再確認（src）：

- `collectResult`（ISRShutdown.cpp:161-181）は `completed = (phase==ShutdownComplete)` のみ。
  `blockingReason`／drain predicate を見ていない。`completed` は phase の別名。
- R32 実測が反例として有効：
```text
phase=ShutdownComplete／completed=true／violations=0／
blockingReason=Unknown／fullyDrained=false（routerPending=2・P3≠0）
```
- ⇒ A 契約は維持。再検証 PASS。

### B. timeout 後の terminal state（R33-B §2・§8）

```text
waitForDrain() → timeout → markTimedOut(Unknown)
  → finalizeShutdown(true) → ShutdownComplete
ShutdownComplete + blockingReason != None は合法な terminal state
```

再確認（src）：

- :615-616 timeout 判定 → :624-633 markTimedOut（唯一の production caller・`markFailed` caller 0）。
- :654-662 timeout 後も safe tryReclaim のみ（drainAll 禁止）。
- :664 `finalizeShutdown(timedOut)` は timedOut=true でも retire 実行（SnapshotCoordinator.h:62-72 二段構え）。
- :744 `transitionTo(ShutdownComplete)` は無条件。`transitionTo` の terminal-only skip
  （ISRShutdown.cpp:134-143）により TimedOut(8)→ShutdownComplete(10) は allowed
  （i=9 Failed は terminal のみ）。
- ⇒ B 契約は維持。再検証 PASS。

### C. drain predicate（R33-B §9・R33-A §3）

- `AudioEngine::isFullyDrained()`（Threading.cpp:204-212）＋
  `ShutdownScheduler::isFullyDrained()`（Coordinator.cpp:527-561）の P1-P22 展開は不変。
- **P22 = `liveLogicalRecoveryObligationCount() == 0`**（Coordinator.cpp:561・D105-R13）。
  したがって `T2 ⇒ T1` は維持（isFullyDrained が P22 を含むため）。
- 一方 `T3a ⇒ T2`／`T3a ⇒ T1` は不成立（R32 D3 が反例・§4）。
- 固定関係は以下（R33-B §9 継承）：
```text
T3a ↛ T2
T3a ↛ T1
T3b ⇒ T2 ⇒ T1
```
- ⇒ C 契約は維持。再検証 PASS。

---

## 4. T1 / T2 / T3 Closure

定義（R33-B §9・新規 API なし・本 gate で再確定）：

```text
T3a = Shutdown state machine reached ShutdownComplete
      （getPhase()==ShutdownComplete ⇔ collectResult().completed==true）
T3b = shutdown completed without blocking/timeout
      （T3a ＋ blockingReason==None ＋ isFullyDrained()==true）
T2  = runtime drain predicate satisfied（isFullyDrained()==true）
T1  = recovery obligations terminalized（liveLogicalRecoveryObligationCount()==0・P22）
```

含意の closure（source 行付き）：

```text
T3b ⇒ T2：定義により（T3b は drain predicate 成立を含む）。
T2 ⇒ T1：Coordinator.cpp:561 の P22 により。isFullyDrained==true なら liveCount==0。
T3a ↛ T2：R32 D3 が反例（T3a 真・P3≠0 により T2 偽）。
T3a ↛ T1：T2 を経由しないため、T3a 単独から per-episode でも shutdown-wide でも
          terminalization は entails されない。R31 の「T3 経由の T1 含意」記述は
          completed-only 解釈としては撤回済み（R33-B §9）。
```

注意（混同防止）：

- engine の `ShutdownPhase::ShutdownComplete`（:744）と Coordinator の
  `markShutdownComplete`（Coordinator.cpp:568-579・drained でなければ Faulted）は別状態機械。
  両者を同一視しない（R33-B §8.3 継承）。
- `collectDrainAudit().isAllZero()` は監査ログ専用（RuntimeDrainAudit.h:77-84
  「shutdown 完了判定には使用しない」）であり、T3b の authority にはしない。
  T3b 判定は `isFullyDrained()` を呼ぶこと（audit 快照の代用不可）。

---

## 5. P2 / P8 / Per-Entry Observability Necessity（Q1・Q2）

R33-A/B で非露出と確認されたもの（本 gate で再確認・変更なし）：

```text
P2 pendingReclaimHandles_：isFullyDrained の述語（Threading.cpp:198-202）だが
  collectDrainAudit（Threading.cpp:109-151）は露出しない。
P8 terminalReclaimResident：isFullyDrained の述語（Threading.cpp:190-191）だが
  collectDrainAudit は露出しない。
per-entry World/DSP identity：DeferredDeletionQueue は sizeApprox のみ露出。
  per-entry の identity／generation の観測点なし。
generation：ShutdownRuntime shutdownGeneration_ は Proof identity 用であり、
  EBR entry generation の観測ではない。
```

### Q1 — P3-5 の現在の目的を満たすために P2／P8／per-entry identity が必要か？

**不要。** 理由：

1. P3-5 の残存目的は terminal contract の closure（shutdown success criterion の固定）であり、
   「どの entry が残ったか」の個体識別ではない。
2. T3b の drain 主張には `isFullyDrained()` の真偽（P2／P8 を**含む** boolean）で足りる。
   P2／P8 の**分解**は不要。`isFullyDrained()` は既存 public API として呼べる。
3. R33-A で timeout の機構（currentEpoch enqueue → epoch 不進 → P3≠0 → timeout →
   Unknown → ShutdownComplete）は source レベルで確定済み。個体識別は因果の追加証明にならない。

### Q2 — 個別 entry identity を観測しなければならない未解決命題が残っているか？

**残っていない。** 理由：

1. R33-A の限定（「1 件が World か DSP か」「どの generation か」は未確定）は
   R33-C 候補として記録されたが、terminal contract の成立条件ではない。
2. 残留のクラスと機構は特定済み（(i) clearedWorld／(ii) activeDSP／(iii) fadingDSP のいずれか・
   D0=1／D3=2 の差は非 null 本数＋graceful drain 消化差で説明可能）。
3. 「観測できない」ことと「次工程に必須」なことは別（指示 §3 の区別を採用）。
   本 gate では後者に該当しないと判定する。

⇒ **R33-C observability は不要（判定 B は不採用）。** ここでも実装しない。

---

## 6. R31-C Necessity（Q3・§4 境界明確化）

### Shutdown-wide question と Recovery per-episode question の分離

```text
Shutdown-wide question（P3-5・本 gate の範囲）：
  T2 = isFullyDrained()（P22 shutdown-wide T1 を含む）
  判定子は既存 public API のみで足りる。

Recovery per-episode question（R31-C の範囲・本 gate では実装しない）：
  各 Recovery episode について
    T1：その episode の obligation が terminalized されたか
    E2：terminalization winner は誰か
  shutdown-wide T1 == 0 が確認できても、各 episode の T1／E2 が証明できたことにはならない。
```

### Q3 — episode 単位の T1／E2 winner identity を必要とする理由が残っているか？

P3-5 terminal contract closure の前提としては **残っていない。** 理由：

1. shutdown-wide T1（P22・`liveLogicalRecoveryObligationCount()==0`）は
   `isFullyDrained()` の述語として既存 predicate に存在する（Coordinator.cpp:561）。
2. R33-A §9 で Recovery obligation 系統と shutdown residual 系統の分離は確定済み
   （D0 でも同一 Unknown＝Recovery 非起因）。shutdown-wide の因果は閉じている。
3. per-episode T1／E2 は R31 が R31-C（従）として限定した独立命題
   （Step5AM §10-§12：per-episode 識別・E2 観測は不能・新 observability が必要）。
   これは P3-5 closure のブロッカーではなく、Recovery exactly-once 証明の別 track である。

⇒ **R31-C は P3-5 closure の先行条件ではない。** R31-C の必要性自体は否定しないが、
次工程としては選択しない（独立 track として残置）。**判定 A は不採用。**

---

## 7. Full-Pipeline Measurement Readiness

実計測は実行しない。開始条件の判定のみ（指示 §5）：

Full-pipeline Measurement 開始に必要な 3 点：

```text
P3-5 terminal contract：CLOSED（本 gate §3-§4 で再検証 PASS）
shutdown success criterion：固定済み（下記 T3b set）
measurement result validity：vehicle が T3b set を満たす場合のみ有効（§8）
```

shutdown success criterion（T3b set・最低要件）：

```text
phase == ShutdownComplete
completed == true
blockingReason == None
transitionViolations == 0
isFullyDrained() == true（drain state の authority・audit 快照の代用不可）
```

特に `phase == ShutdownComplete` だけを measurement vehicle の成功条件にしてはならない
（R33-B §8 の禁止を継承）。`phase／completed／blockingReason／transitionViolations／
drain state` の 5 点セットが確定していることが開始条件であり、本 gate で確定した。

⇒ 開始条件の定義は完了。実計測自体は本 gate では開始しない（次 gate 以降の準備 gate で扱う）。

---

## 8. Existing Vehicle Contract Compatibility（変更せず・分類のみ）

対象 4 系統（R33-B §10 の監査を継承・test 修正は禁止）：

| vehicle | 現行 assertion | 分類 |
| --- | --- | --- |
| D167 terminal（PublishPipelineIntegrationTests.cpp:657-663） | phase＋completed＋TV（blockingReason 未参照・isFullyDrained 意図的外し） | **terminal success vehicle として不十分**。admission closure vehicle としては再利用可 |
| D169-2-5（:825-830） | admission Closed＋phase のみ | **terminal success vehicle として不十分**（1 より弱い）。collapse regression としては再利用可 |
| D169-2-6（:986-991） | admission Closed＋phase のみ | **同上**。stress vehicle としては再利用可 |
| R30/R32 vehicle terminalOk（P1PolyphaseGainCharacterization.cpp:1278-1283） | phase＋completed＋TV（blockingReason・drain は emit のみ） | **drain success vehicle として不十分**。R32 D3 は terminalOk=1 になる。episode 観測（seq／coord／cmt）部分は再利用可 |

結論：既存 vehicle はいずれも R32 型状態（ShutdownComplete＋completed＋TV=0＋
blockingReason=Unknown＋fullyDrained=false）を PASS 認識し得るため、
**そのままでは terminal success vehicle（T3b 判定）として再利用不可**。
次工程の measurement vehicle は T3b 5 点セットを満たす新規または拡張 vehicle が必要
（設計は次 gate・本 gate では行わない）。

---

## 9. GO / STOP Decision

```text
判定 A（R33-C 不要・R31-C 必要 → 次に R31-C）：不採用
  R31-C（per-episode T1／E2）は独立命題であり、P3-5 closure の先行条件ではない（§6）。
判定 B（R33-C 必要 → 次に R33-C design gate）：不採用
  P2／P8／per-entry identity の未観測が P3-5 の残存命題を解くために必要ではない（§5）。
判定 C（terminal contract closure 済み → Full-pipeline Measurement 準備 gate）：ADOPTED
  P3-5 の目的＋shutdown terminal semantics＋measurement success criteria は
  既存 source／既存 API のみで十分に固定された（§3・§4・§7）。
  R33-C／R31-C による追加観測は closure に必須でない。
```

STOP 条件の確認：

```text
C1（contract が一意に決まらない）：該当なし（§3 で全契約を source 行で再検証）。
C2（意味に source 内矛盾）：該当なし（R33-B で supersede 済み・本 gate で再発なし）。
C3（clarification に production API 変更が必要）：該当なし（新規 API なし）。
C4（test 成功条件修正に production semantics 変更が必要）：該当なし（監査のみ・修正なし）。
```

---

## 10. Next Gate

```text
R33-A PASS → R33-B PASS → R34（本 gate・判定 C・P3-5 terminal contract CLOSED）
    │
    └── Full-pipeline Measurement 準備 gate（別 gate・実計測は含まない）
          - T3b 5 点セットを満たす measurement vehicle の設計
          - 既存 vehicle（§8）の再利用範囲の確定
          - R33-C／R31-C は本線の前提にしない（独立 track として残置）
```

R34 で禁止したこと（厳守・いずれも未実施）：

```text
× R33-C 実装 × R31-C 実装 × getter／counter 追加 × test assertion 修正
× publishEpoch 追加 × waitForDrain 変更 × force reclaim 変更
× Shutdown semantics 変更 × Recovery 変更 × Full-pipeline measurement
× buzz／limiter／NUC 調査への復帰
```

- source は R27 production＋R23/R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
- 本 gate はここで停止する。
