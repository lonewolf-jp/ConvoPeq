# STG-8 Concrete Defect Discovery（2026-09-28・read-only）

## Authority（調査開始時）
- SHA-256: `955582FBF3ABBEE532E67EF908DF083D807F34DE550D85EA89A76FA93DB0732C`
- size: 5,589,694 B / Generated: 2026-09-28 14:33:45 / HEAD: `29dbdca9`
- `--check`: FRESH / NEWER_SRC_COUNT: 0
- 判定基準: `doc/Practical Stable ISR Bridge Runtime.md`
  （RT no wait/lock/allocation/delete/decision、Coordinator sole authority、
  Retire through Epoch、RuntimeWorld immutable after publish、
  Overflow != silent loss、Shutdown = complete drain）

## 除外（再候補化なし）
STG-1 / STG-2 / STG-4-1 / STG-6-D1 / STG-7-D1、および既知 P3 double-representation
（`Transport(O)∧Durable(O)` 型）。以下 D1–D3 はいずれも新規 variant
（Live + 無代表 / deferred-slot-vs-table 終端欠落）であり P3 と重ならない。

## 方法
- Domain A（deferred/async lifecycle）: 探索 subagent＋所有者による現行ソース実読検証。
- Domain B（retire/epoch/ownership）・Domain C（shutdown completeness）:
  subagent 2 件が環境起因で失敗したため所有者が直接実読（bounded probe）。
- production / test の変更・commit なし。push NOT AUTHORIZED 継続。

---

## STG-8-D1 — Deferred-discard が recovery obligation を終端化しない（Live slot leak）

1. defect statement:
   `processDeferredAdmission()` の Discard 分岐（および RedriveBudgetExhausted 分岐）は
   deferred DSP を authority retire し slot を破棄するが、slot が保持していた
   `recoveryObligationId != 0` を単一 Completion Authority に通さない。
   Coordinator table 上の当該 obligation は `Live` のまま残留し、transport 表現も
   durable 表現も持たない（pop 済み・slot 消滅）ため adjudicate・redrive の双方に
   永遠に拾われない。
2. source location:
   `src/audioengine/RuntimePublicationOrchestrator.cpp:916-938`（Discard）、
   `:885-901`（RedriveBudgetExhausted）、判定生産 `src/audioengine/PublicationAdmission.cpp:77-88`、
   迂回された authority `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1036-1069`。
3. triggering precondition:
   deferred slot が recovery request（`req.recoveryObligationId != 0`）を保持。
   `testFadingRuntimePresent` フックで決定的に再現可能。
4. exact execution sequence:
   1. recovery build が `RebuildDispatch.cpp:1085/:1165` から
      `enqueuePublicationIntentForRuntimeCommit(...,oblId=O)`（O≠0）。
   2. `Commit.cpp:831` が O を `req` に搬送 → `submitPublishRequest` →
      `DeferredFadingActive` → `enqueueDeferred`。slot=O、table=`Live(O,Transport)`
     （Builder が pop 済みで queue は空）。
   3. 新規 `requestRebuild` が `rebuildRequestGeneration` を進める。
   4. wake により `processDeferredAdmission(false)` →
      `evaluateDeferred` が generation 不一致で `{Discard,StaleDiscard}`（TTL/sequence も同型）。
   5. `:934-936` が DSP retire＋view discard のみ。`resolveRecoveryObligation` 呼び出しなし。
5. actual behavior: `liveLogicalRecoveryObligationCount` 不変（終端イベントで ΔL=0）、
   `W(O)={Live,pending=0,delivery=Transport}` が無代表で残留。
   expected: `ISRRuntimePublicationCoordinator.h:367-377`＋`Coordinator.cpp:1023-1035` の
   「Retry 以外は単一 Authority で終端」、D105-R5-9 MUST-1、
   および `Orchestrator.cpp:369-372` 自身の記述
   「admission 失敗した recovery obligation は必ず resolve すること
   （さもなくば Live slot が leak し 32 を消費し尽くす）」。
   Ready-path の前例（`:398` StaleSuperseded 終端）と対称を欠く。
   violated invariant: 単一 −1 authority、`L≤32` 会計、
   `isFullyDrained` の `L==0` 要件（`Coordinator.cpp:561`）。
6. deterministic counterexample（timing なし）:
   fading 有効→recovery O1/gen1 を defer（L=1）→`requestRebuild` で gen2→
   `processDeferredAdmission` 直呼びで Discard。`hasDeferred==false` だが L==1 のまま
  （期待 0）、`peekLifecycleForTest(O1)` は Live のまま。32 回繰返しで L=32、
   33 回目の `submitRecoveryRequest` は `false`（capacity exhausted）。
7. impact: 容量 leak（discard 毎に 1）。回復 admission 枯渇→quarantine DSP 再建不能
   （stale world に停滞）。`isFullyDrained()==false` 永続→shutdown `waitForDrain` timeout。
   DSP 自体は retire 済みのためメモリ leak ではなく liveness/capacity の欠陥。
8. minimal repair boundary: `processDeferredAdmission` の Discard＋RedriveBudgetExhausted
   分岐のみ。snapshot 採取後に O≠0 なら `resolveRecoveryObligation(id, StaleSuperseded)`
  （`discardReason==ShutdownDiscard` 時は `ShutdownDiscarded`）。Coordinator 変更なし。
9. regression test proposal: fading フック有効下で recovery deferred O1/gen1→generation bump→
   `processDeferredAdmission` 実行→`!hasDeferred && L==0 &&
   peekLifecycleForTest(O1)==ResolvedStaleSuperseded` を assert。
   shutdown 文脈の第二 case は `ShutdownDiscarded` を assert。
10. why existing tests do not catch it:
    `DeferredFlowIntegrationTests.cpp` は recovery 非含有（`recoveryObligation|Obligation|liveLogical`
    の参照 0 件）の Ready-path＋slot drain のみを検証。`PublicationAdmissionTests` は
    `evaluateDeferred` の Decision 純粋判定のみ。C11–C16 は Orchestrator-Discard×Coordinator-L
    を扱わない。
11. （11=上記 10 に含む。欠番としないため再掲: テスト追加は harness 内 subtest で足り、
    CTest 新規登録は不要。）
12. authority identity: 上記 Authority 節と同一（SHA `955582FB…0732C` / HEAD `29dbdca9`）。

---

## STG-8-D2 — Deferred-overwrite が旧 recovery obligation を孤立させる（無終端）

1. defect statement:
   `enqueueDeferred()` の overwrite は旧 deferred DSP を retire するが、
   obligation identity `(generation,obligationId)` を無終端で置換する。
   旧 O_old は `Live+Transport` のまま残り、DSP は EBR 行き・slot 消滅で無代表となる。
2. source location:
   `src/audioengine/RuntimePublicationOrchestrator.cpp:498-507`（旧 DSP retire）、
   `:513-521`（identity 置換）、`:551-579`（slot 置換）。
3. triggering precondition: slot 占有中に別 key の新規 `DeferredFadingActive`
  （通常 gen または別 recovery）。単一 owner（RebuildThread）直列で race 不要。
4. exact execution sequence:
   1. fading 有効下で recovery O_old/gen1 が defer（slot=O_old、L=1）。
   2. 新規 build gen2（oblId=0 等）が `submitPublishRequest→DeferredFadingActive→enqueueDeferred`。
   3. `:471-473` overwrite カウント、`:506` 旧 DSP retire、`:516-521` identity 上書き、
      `:551` slot 全置換。旧 O_old への `resolveRecoveryObligation` なし。
5. actual behavior: 旧 DSP 廃棄済み・旧 O_old は Live のまま L を消費し無代表。
   新 O_new も Live のため 1 slot に対し L=2。
   expected: 別 obligation の追い出しは単一 authority で `StaleSuperseded` 終端
   （MUST-4 stale-generation 終端）。同ファイル D132 コメントは DSP のみを扱い L を扱わない。
   violated invariant: D1 と同一（単一 −1、`L≤32`、`isFullyDrained L==0`）。
   P3 とは相違（1 O の二重表現ではなく、1 slot に対する 2 Live O）。
6. deterministic counterexample:
   fading 有効→recovery O1/gen1 defer（L=1）→通常 gen2 defer。
   結果 `hasDeferred==true`・slot=gen2 のまま L==1 残留（期待 0）、
   `peekLifecycleForTest(O1)==Live`。交互 32 回で L==32。
7. impact: D1 と同型だが UI burst（crossfade 中の overwrite が通常系）のため枯渇が高速。
   回復 starvation＋shutdown drain timeout。
8. minimal repair boundary: `enqueueDeferred` の `:551` 置換前のみ。旧 slot の O≠0 かつ
   新 req と異なる場合 `resolveRecoveryObligation(oldId, StaleSuperseded)`
  （冪等・競合終端勝ちは no-op）。retire 経路・identity  logic 変更なし。
9. regression test proposal: recovery O1 で slot 占有→通常 gen2 で上書き→
   `L==0 && peekLifecycleForTest(O1)==ResolvedStaleSuperseded && hasDeferred==true`
   を assert（新規 request 保全）。既存 overwrite-count assertion 不変。
10. why existing tests do not catch it:
    overwrite テストは `deferredOverwriteCount` と DSP 処分（D132）のみを assert し L を見ない。
    C16 coalesce テストは同 key の ΔL=0 のみで別 key 追い出し終端を扱わない。
11. （テスト追加は harness 内 subtest、CTest 新規登録不要。）
12. authority identity: 上記 Authority 節と同一。

---

## STG-8-D3 — `RejectedPressure` の transport recovery が無 signal・無 rearm・無 redrive で停滞

1. defect statement:
   `submitPublishRequest` の `RejectedPressure` は transport recovery を Live 維持するが
   （Retry 意味として正しい）、`RejectedNotFinalized` と異なり failure signal を post せず、
   rearm guard（durable Building＋id 一致）も transport には不発のため、
   `Live+Transport+pending=0` の無 queue・無 durable 状態で adjudicate・redrive の
   双方から永久に外れる（外部再 submit までの間、自動 liveness なし）。
2. source location:
   `src/audioengine/RuntimePublicationOrchestrator.cpp:421-437`（Pressure、signal なし）、
   対照 `:403-420`（NotFinalized、`:415` で signal）、
   guard `src/audioengine/ISRRuntimePublicationCoordinator.cpp:1181-1188`、
   停滞点 `:1128-1129`（adjudicate skip）＋`:1204-1205`（redrive skip）。
3. triggering precondition: transport recovery（push 成功・`delivery=Transport`）＋
   `evaluate` が `RejectedPressure`（Health Critical/Degraded `PublicationAdmission.cpp:27-36`、
   または throttle `retirePressurePublicationThrottleActive_` `:40-48`）。
4. exact execution sequence:
   1. `submitRecoveryRequest(O)`→push 成功→`casDelivery(Transport)`、L=1。Builder wake。
   2. Builder が pop（`popRecoveryRequest` `:1434-1447` は delivery 不変）＋build＋
      `enqueuePublicationIntentForRuntimeCommit(...,O)`→`submitPublishRequest`→`RejectedPressure`。
   3. `:429` `resolveIfRecovery(Retry)` は no-op（`:1043-1044`、Live 維持・正しい）。
   4. `:431` `rearmRecoveryRetry(O)` は durable 非 Building で no-op。
   5. `:436` 新造 DSP retire。`W(O)={Live,pending=0,delivery=Transport}`、queue 空、durable 空。
   6. 以後の `runCoordinatorPhase` は adjudicate skip・redrive skip。pressure 解除後も自動再開なし。
5. actual behavior: 圧力解除後も redrive されず（`recoveryRetryRedriveCount` 増加なし）。
   expected: 分岐コメント MUST-2 の「delivery re-drive is deferred」を満たすには
   `delivery=None` 化＋redrive 可能化が必要。`NotFinalized` 前例（`:414-415` signal post）が
   正面の対称形。D105-R5-10 liveness（「delivery 表現なき Live は全て re-drive」）の前提
   「Transport⇒queue が intent を保持」が pop 後に偽になる。
   violated invariant: D105-R5-10 liveness＋`L≤32`（shutdown まで leak）。
6. deterministic counterexample:
   Health Critical 強制（または throttle 有効）→transport recovery O（L=1、Transport）→
   Builder 1 駆動で `RejectedPressure`・DSP retire→pressure 解除→adjudicate＋redrive 100 tick。
   `delivery==Transport` のまま・redrive 0 件・L==1（期待: None 化→再 Transport＋同 obligationId
   の Builder 再 intent）。
7. impact: 圧力 episode 中の quarantine recovery が圧力解除後に自動再建されない
  （stale/muted convolution が次回手動 rebuild まで継続）。繰返しで L→32。
   DSP は retire 済みでメモリ面は健全（liveness のみの欠陥）。
8. minimal repair boundary: `RejectedPressure` 分岐のみ。O≠0 かつ rearm 不発の場合
   `postRecoveryFailureSignal(O)`（`NotFinalized` と同一の冪等 CAS 保証。
   `kMaxObligationConsecutiveFailures=4→ResolvedFailed` の budget を保存するため
   直接 `casDelivery(None)` より signal を優先）。
9. regression test proposal: transport recovery＋Critical 強制で `submitPublishRequest`→
   L==1 維持を確認→adjudicate＋redrive で `delivery==None`→再 Transport と同 oblId 再 intent を
   assert（既存 NotFinalized→redrive テストの鏡像）。
10. why existing tests do not catch it:
    pressure テストは非 recovery publish の Decision 判定のみ、または durable rearm 経路。
    `Pressure × transport-recovery × redrive-eligibility` の組合せテストなし。
    R18 exhaustion テストは signal 経路の `markTransientFailure` を使用し、
    唯一の Retry-without-signal 経路である Pressure 分岐を通らない。
11. （テスト追加は harness 内 subtest、CTest 新規登録不要。）
12. authority identity: 上記 Authority 節と同一。

---

## Domain B 探査（bounded probe・欠陥なし）
- `DSPLifetimeManager::retire/retireByHandle`（`DSPLifetimeManager.cpp:35-137`）:
  map-erase（一括 findAndErase）＋miss 時 no-op のため二重 retire は構造的に不可。
  quarantine fallback は `enqueueWithRetry` 内部移送＋追加処置禁止で二重移送回避。
  E-2/E-3 の pointer-value retirement 廃止済み。重複 authority の実証可能パスなし。
- `resolveRecoveryObligation`（`Coordinator.cpp:1036-1069`）: id-based resolve＋Retry no-op。
  却下: ABA の懸念は id 単調性＋full-word CAS（G-4.3/D152 T5）の設計で排除されており、
  反例を構成できず。
- 結論: **Domain B: NO CONCRETE DEFECT**（本探査範囲）。

## Domain C 探査（bounded probe・欠陥なし）
- `releaseResources`（`:237-238` join → `:250-256` producer join 待機 → `:262-265` epoch →
  `:279-304` graceful drain 5s → `:339-375` final drain → `:393/:600` deferred 無条件 clear →
  `:615` 2s verify drain → `:654-677` residual OwnerChannel drain）＋
  `stopRebuildThread` 内 discard-all（`RebuildDispatch.cpp:825-839`:
  `discardRecoveryRequestsOnShutdown` が全 Live を ShutdownDiscarded 化）。
  D1–D3 の停滞物は shutdown で閉鎖されるため C への波及なし。
- `~AudioEngine`（`CtorDtor.cpp:103-329`）: multi-drain（D＋Q＋E＋T、stuck-reader fallback 含む）、
  EBR 残留の観測のみ assert（Debug）、Faulted 診断。budget 超過は `markTimedOut(reason)`＋
  audit＋safe-tryReclaim で観測可能（silent loss なし）。
- 却下した suspicious 3 件:
  1. `waitForDrain(100,1)` 予算超過 `break`（`:250-256`）— 後段の無条件 drain 群が残存物を
     処理するため具体的遺失物を特定できず。設計意見に留まるため却下。
  2. `clearDeferredForShutdown` の obligation 非終端（`:591-652`）— discard-all が先行閉鎖
     するため shutdown 文脈では無害。却下（mid-run 文脈は D1 の射程）。
  3. dtor/releaseResources 二経路 — 双方に drain 群があり、到達不能ステップを特定できず。却下。
- 結論: **Domain C: NO CONCRETE DEFECT**（本探査範囲）。

## 総合
```text
STG-8-D1  Concrete Defect（D1・deferred-discard 終端欠落）
STG-8-D2  Concrete Defect（D2・deferred-overwrite 終端欠落）
STG-8-D3  Concrete Defect（D3・RejectedPressure 無 signal 停滞）
```
3 件とも repair boundary は `RuntimePublicationOrchestrator.cpp` の当該分岐のみ。
**実装には進まず、Owner の Repair GO を待つ。**
