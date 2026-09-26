# P1-5-IR-P2 — Step 5-AN / P3-5-R32: Recovery-origin Terminal Observation / Build・Run Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R32）
- **種別**: test-only implementation＋build gate＋run gate。**production 変更 0・CMake 0・
  new counter/getter 0・Intent/semantics 変更 0**。R27 production diff は不変で保持。
- **判定**: **R32-B（STOP）**。
  R30 の STOP-E は **vehicle の観測順序の誤りとして閉じた**（episode 内 Running 中 `waitForDrain` を撤去し、
  `h.stop()` 後の契約的観測へ置換）。control / Recovery 3/3 / `h.stop()` 復帰 / nominal terminal
  （`ShutdownComplete`・`completed=1`・`violations=0`）はすべて成立した。
  **しかし `blockingReason=Unknown` が `markTimedOut` の実行（= VerifyDrained の `waitForDrain(2000)` timeout）
  を証明**し、`completed` は drain 成功を意味しないことが判明した → R32 §9/§12 に従い R32-A としない。
  当該 timeout は **default harness（Recovery なし）でも同値**（別 run の `shutdown_trace.json` で確認）
  → **Recovery-origin 起因ではなく、既存の shutdown-drain 残留**。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF
R16/R19/R22/R27 counters／R23 vehicle／R30 D1 vehicle 保持
ConvoPeq.md 再生成済み（2026-09-23 23:33:38・generator output_sourcecode_markdown.py）
production diff = R27 のみ（8 files・+155/−12）／CMake clean
R32 差分 = test-only（既存 TU）＋ 本ドキュメントのみ
```

- **`ConvoPeq(3).md` は本環境に存在しない**（C:/VSC_Project 配下・探索 root 全域を再探索）。
  生成物 `ConvoPeq.md` を最新 source authority とした（R31 の記述を維持）。

## 2. Latest ConvoPeq Source Reconciliation

R32 §2 の対象を **最新 md の実コードで再確認**（R31 の順序が不変であること）：

```text
releaseResources()（terminal pass は requestTerminalRelease 必須）
  :96  transitionTo(AudioStopped)
  :109 closeAdmission()
  :237-238 shutdownCoordinatorLoop() / stopRebuildThread()
  :250-256 joinProducers() retry（outstanding()>0 の間 waitForDrain(100,1)）
  :259 ObserverDrained → :264 RetireClosed → :265 EpochSettled
  :376 ReclaimComplete → :383 EmergencyDrain → :475 VerifyDrained
  :615 waitForDrain(2000, 2)          ← ここが drain の主判定
  :616 timedOut = !drainedWithinBudget
  :632 if (timedOut) markTimedOut(reason)   ← reason は既定 Unknown（stuck/rebuilder 以外）
  :654 if (!drained || !isFullyDrained()) → drainDeferredRetireQueues + tryReclaim
  :664 finalizeShutdown(timedOut)
  :744 transitionTo(ShutdownComplete)
```

確認結果（R31 からの差分）：

```text
[不変] waitForDrain / isFullyDrained / collectDrainAudit / markTimedOut / collectResult / getPhase
[不変] 上記の phase 遷移順序
[新規確認] transitionTo() の terminal-skip 規則（ISRShutdown.cpp:124-151）:
   allowed = (t == c || t == c+1) 。t > c+1 のときは「間の状態が全て terminal」なら許可。
   ⇒ TimedOut(8) → ShutdownComplete(10) は **許可される**（間の 9=Failed は terminal）。
   ⇒ R31 §4 の記述「TimedOut 後は phase は TimedOut のまま留まる」は **誤り**（本 Step で訂正）。
```

- `markTimedOut` の caller は **ReleaseResources.cpp:632 の 1 箇所のみ**（`if (timedOut)` ガード）。
  `markFailed` の caller は **0**。
- ⇒ **`blockingReason != None` は「drain timeout が発生した」ことの十分な証拠**。

## 3. R30 Vehicle Delta（test-only・`--p1-recovery-origin`）

削除（R32 §4 の要求）：

```text
- episode 内: emitLine(drain_pre) / waitForDrain(kWaitMs) / emitLine(drain_post)
- baseline: drain_baseline（Running 中の waitForDrain 系観測）
- 末尾診断: stopAudioOnly() → waitForDrain(10000) → drain_after_audio_stop
→ vehicle 内に waitForDrain の **呼び出しは 0**（コメントのみ残置）
```

保持（R32 §5・R30 で確立した構造を変更しない）：

```text
start → authoritative Runtime 待機 → registerDSPHandleForRuntime
→ baseline settle(sleepPump 1000) → control(trigger なし 5000ms)
→ episode ×3（submitRecoveryIntent ×1 → seq advancement 待機 → pre/post 差分）
```

追加（R32 §6-8）：

```text
episode 完了後: sleepPump(500) → h.stop()
→ e.isrShutdownRuntime().getPhase() / collectResult(h,0) / admissionState()
→ phase / completed / blockingReason / transitionViolations / admissionClosed /
  lateCallbacks / postStopEnqueue を emit
→ collectDrainAudit() を 1 回だけ diagnostic evidence として取得（補助・主判定に使わない）
```

## 4. Production / Test / CMake Diff Boundary

| 対象 | 変更 |
| --- | --- |
| production source（src/audioengine） | **0**（diff は R27 の8ファイルのみ） |
| CMake | **0**（`git status` clean） |
| test source | 既存 TU のみ：`P1PolyphaseGainCharacterization.cpp`（vehicle 修正）／`PublishPipelineIntegrationTests.cpp`（dispatch は R30 で追加済み・変更なし） |
| new counter / getter / public API / Intent / semantics / admission / deferred / retry / shutdown production | 0 |

## 5. Build Gate（PASS）

```text
条件: Release／CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF／vcvarsall x64＋oneAPI setvars intel64
cmake --build build --config Release --target AudioEngineHarness
→ BUILD_EXIT=0（error 0 行）
```

## 6. Control Gate（PASS）

```text
[P1REC] control_notrigger waitMs=5000 dSeq=0
```

- trigger なし 5 s で **自発 publication 0** → 以降の episode の seq 前進が trigger 起因であることを
  window 粒度で裏付ける（R30 の baseline settle を復元して制御外の起動直後 publish を除外した）。

## 7. Recovery Episode Gate（3/3 PASS）

| episode | dSeq | dCoord | dCmt | dTake | dBld | dDrp | seqAdvanced |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep1 | +1 | 0 | 0 | 0 | 0 | 0 | 1 |
| ep2 | +1 | 0 | 0 | 0 | 0 | 0 | 1 |
| ep3 | +1 | 0 | 0 | 0 | 0 | 0 | 1 |

絶対値（seq のみ 3→4→5→6・cmt/coord/take/bld は不変）：

```text
pre  req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=N   blo=0
post req=1 que=1 dup=0 take=1 bld=1 cmt=1 coord=2 drp=0 seq=N+1 blo=0
```

- Recovery-origin publish を 3/3 再現（R30 と同型）。`coord`（Main-origin pop）は不変。
- `cmt`/`take`/`bld` 不変 → main-site rebuild 経由でない。

## 8. Shutdown Terminal Observation（nominal PASS・ただし timeout が内在）

`h.stop()` 後の既存 public API 観測：

```text
[P1REC] terminal phase=10 phaseComplete=1 phaseTimeout=0 phaseFailed=0
        completed=1 blockingReason=9 violations=0
        admissionClosed=1 lateCallbacks=0 postStopEnqueue=0
```

| 項目 | 値 | 判定 |
| --- | --- | --- |
| `phase` | 10 = `ShutdownComplete` | PASS |
| `completed` | true | PASS |
| `transitionViolations` | 0 | PASS |
| `admissionState` | Closed | PASS |
| `lateCallbackCount` / `postStopEnqueueCount` | 0 / 0 | PASS |
| **`blockingReason`** | **9 = `Unknown`** | **timeout 発生（§2 の markTimedOut 唯一 caller）** |

- `blockingReason=Unknown(9)` は `markTimedOut(Unknown)` が実行された（= `waitForDrain(2000,2)` が
  timeout した）ことの証拠。`blockingReason` の初期値は `None(0)` であり、書込みは markTimedOut/Failed のみ。
- `transitionTo(TimedOut → ShutdownComplete)` が terminal-skip で許可されるため、
  **timeout 後も `phase` は ShutdownComplete / `completed=true` になり得る**（§2 の新規確認）。
  したがって **`completed` 単独では drain 成功を意味しない**。

## 9. Drain Audit（補助 evidence）

```text
[P1REC] terminal_drain_audit fullyDrained=0 pendPub=0 pendRetire=0 xfade=0
        routerPending=2 deferred=0 quarRes=0 activeWorlds=1 published=6 retired=5
        activeReaders=0 stuckReaders=0 overflowRes=0
```

- shutdown 後も `routerPending=2`（retire 残留）・`fullyDrained=0`。
- 対応する `evidence/shutdown_trace.json`（terminal pass が :745 で出力）:

```text
vehicle run : phase=ShutdownComplete / blockingReason=Unknown(9) / violations=0
              sh3_pendingRetire=2 / sh2_activeCrossfade=0 / sh1=sh4=sh5=sh6=0
```

- readers=0・crossfade=0・publication=0・deferred=0・quarantine=0 であるため、残留は
  **retire/reclaim 系**（routerPendingRetire / pendingReclaimHandles）。

## 10. Failure Classification（R32 §10 の分離）

```text
停止点 = VerifyDrained の drain predicate（:615 waitForDrain(2000,2)）
  - publication : 0（pendPub=0）
  - reader      : 0（activeReaders=0 / stuckReaders=0）
  - builder     : 終了済み（rebuildThreadIsRunning=false が jassert :613 で要求される）
  - reclaim/retire : 残留あり（routerPending=2）
  - recovery obligation : 当該残留の原因ではない（下記 separation）
```

**Recovery 起因かの分離（同一条件・別 run・既存 evidence）**

```text
vehicle run（Recovery 3 episodes）  : blockingReason=Unknown(9) / sh3_pendingRetire=2
default harness（Recovery 0）        : blockingReason=Unknown(9) / sh3_pendingRetire=1
```

- Recovery episode が **0 の default harness でも同一の timeout（blockingReason=Unknown）** が発生。
- ⇒ 当該 drain timeout は **Recovery-origin vehicle 起因ではない**（pre-existing / 環境依存の
  shutdown-drain 残留）。Recovery defect とは断定しない（R32 §11 の指示どおり）。
- 副次：D167（`completed` のみを assert）が PASS していても、`blockingReason` を見ないため
  timeout を検出していない（既存 test の観測範囲の限界）。

## 11. R32 Gate

```text
R32-A の列挙条件:
  [OK] control dSeq = 0
  [OK] Recovery 3/3
  [OK] seq +1 ×3
  [OK] coord 0 ×3
  [OK] cmt 0 ×3
  [OK] take 0 ×3
  [OK] bld 0 ×3
  [OK] h.stop() 正常復帰（例外なく return）
  [OK] phase = ShutdownComplete
  [OK] completed = true
  [OK] transitionViolations = 0
  → 列挙条件はすべて成立。

R32-B（採用・STOP）:
  §9 が禁じる「timeout を通常完了と混同する」構造が実在した。
  blockingReason=Unknown(9) が markTimedOut（VerifyDrained の drain timeout）を証明し、
  §2 の terminal-skip により phase/completed は ShutdownComplete/true に上書きされている。
  ⇒ drain は成功しておらず、R31 の T3 ⟹ T1（isFullyDrained の liveCount==0 含意）は
     本 run では成立しない。
  ただし timeout は Recovery 非起因（§10 の分離）であり、Recovery-origin publish の成立
  （control 0・episode 3/3・coord 不変）と、契約的 terminal 観測の成立（phase/completed/
  violations/admission）自体は確認できた。

R32-C（production 変更が必要になった場合・即STOP）: 非該当（production 変更 0）
```

- R30 の STOP-E は **観測子の選択・順序の誤りとして閉じた**（episode 内 waitForDrain 撤去 →
  `h.stop()` 後の契約的観測で terminal state は取得可能）。
- ただし「drain 完了を伴う T3」は本環境では得られず、**T3 の nominal 達成と drain 達成は別**であることを確定。

## 12. R31-C Residual Limitation

```text
本 Step で確定した追加制約（R31 の訂正を含む）:
  C1. R31 §4 の「TimedOut 後は TimedOut のまま」は誤り。transitionTo の terminal-skip により
      TimedOut → ShutdownComplete が成立する。⇒ `completed` は drain 成功の指標にならない。
      信頼できる非-timeout 指標は `blockingReason == None(0)`（本環境では未観測）。
  C2. drain が timeout する場合、`isFullyDrained()`（= liveCount==0 を含む）は成立しておらず、
      T3 から T1 を entail できない ⇒ E1（terminalization）の集約観測は **drain 成功時のみ**有効。
  C3. per-episode T1 識別 / E2（二重 terminalization 不存在）は依然として観測不能
      （winner/liveCount が非露出）。R31-C の残件として維持（本 Step では実装しない）。
  C4. shutdown-drain timeout 自体（readers/refcount ではなく retire/reclaim 残留）は
      Recovery 非起因の既存事象。原因分離は別 work item（本 Step の scope 外）。
```

## 13. Next Step

```text
R32-B → 次は実装でも R31 Full-pipeline Measurement でもなく、以下を分離した gate を推奨:

  (a) shutdown-drain 残留（retire/reclaim / routerPendingRetire）の原因分離
      — pre-existing・Recovery 非起因であることは本 Step で確定済み。
      — R33-A 候補: drain predicate のどの成分が timeout を生むかの source/diagnostic 監査
        （`pendingReclaimHandles_` は collectDrainAudit 非露出 → 必要なら observability gate を分離）。
  (b) terminalization 観測子の信頼化（blockingReason を含む契約の明確化）
      — `completed` 単独を成功条件にしない。`blockingReason == None` を併用する。
      — これは source（ShutdownRuntime）の contract clarification 候補（R31-D 相当）。
  (c) per-episode T1 / E2 winner observability（R31-C 残件）は独立 gate。

  R33 の scope は (a)/(b)/(c) のどれを先にするかを、本 Step の実測を基に改めて決める。
  R32-A が出ていないため、R31 Full-pipeline Measurement には進まない。
```

- source は R27 production＋R30/R32 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。
