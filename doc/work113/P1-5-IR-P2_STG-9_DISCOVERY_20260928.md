# STG-9 Concrete Defect Discovery（2026-09-28・read-only）

> **Mode**: read-only / discovery only。production source・test source・CMake・`ConvoPeq.md` の変更なし。
> **Commit**: なし。**Push**: NOT AUTHORIZED。
> 本書は **Case A**（証明可能候補の提示と STOP）で終了する。Owner GO なしに修正へ進まない。

---

## 0. Authority 検証（再確認）

| 項目 | 値 |
| --- | --- |
| authority file | `C:\VSC_Project\ConvoPeq\ConvoPeq.md` |
| SHA-256 | `615940DC90D73CED680081A8FBD0B491DBDC376788C0469B0A59AF7D2F6F2E3E3B` |
| size | 5,612,326 B |
| file mtime | 2026-09-28 16:18:11 |
| git HEAD | `b0b4694161817e705b21534777a2b2cf8085f0f0` |
| NEWER_SRC_COUNT（authority より新しい `src/**` ソース） | **0** |

判定: **FRESH**。user 指定 SHA と完全一致。`ConvoPeq.md` は本セッションで一切変更していない。

注記（user 記述と実態の相違、記録のみ）:
- `ConvoPeq(9).md` は main repo ルートに存在しない。実 authority は `ConvoPeq.md`。
- worktree 内の `ConvoPeq.md` は 5,489,292 B / SHA `025EE1E6…` / mtime 15:58:53。差分は `Generated` タイムスタンプ行のみで **本文は同一**。`src/` も worktree と main repo で差分ゼロ（無関係な `.pyc` 1 件のみ）。

---

## 1. 判定基準（`doc/Practical Stable ISR Bridge Runtime.md` より）

| 種別 | 破られる不変条件 |
| --- | --- |
| **Liveness** | 活性。処理・drain・recovery が finitely many step 内に完了する |
| **Safety** | 安全。RT reader が dereference 可能な对象的生存、UAF なし |
| **Ownership** |  所有権。単一 authority・二重解放なし・リークなし |
| **Authority** | 権限。Coordinator が reclaim の一本化点、RuntimeWorld の不変性 |

優先順位: P0 = ユーザーデータ消失 / crash / UAF / permanent strand、P1 = lifecycle・ownership・recovery・state corruption、P2 = RT 契約 / persistence / semantic inconsistency、P3 = robustness・diagnostics・boundedness。

**昇格基準**: 「怪しそう」断定は不可。source 上 disprove 不能な反例が 1 つでも存在し、既存 test がそれを捕捉し得ない場合に限り Concrete Defect として昇格する。

---

## 2. 探索範囲

| Domain | 対象 | 結果 |
| --- | --- | --- |
| A | Retire / EBR / Epoch（retire router、DSPHandleRuntime、reclaim authority、drain predicate） | **STG-9-D1 発見** |
| B | Publication admission / deferred slot / recovery obligation | 欠陥なし（STG-8 D1–D3 は既に CLOSED・再オープンせず） |
| C | Shutdown drain 順序・terminalization | 欠陥なし（drain predicate の**観測・記述**に関する所見のみ。述語そのものは Domain A で STG-9-D1 として扱う） |
| D | Loader / IR lifecycle / preset | 欠陥なし（STG-4-1・STG-6-D1・STG-7-D1 は既に CLOSED） |

副次手段: `cppcheck 2.22`（`--project=`・audioengine 9 ファイル + SnapshotCoordinator、実質的無所見）、subagent 2 本（Domain A / Domain C）の報告は**すべて所有者自身が実ソースで一次検証**して採否を決定した（subagent の CONFIRMED 申告を鵜呑みにしていない）。

---

## 3. STG-9-D1（昇格）: `reclaimInFlightCount_` が identity set と非対称に増減し、`isFullyDrained()` を恒久 false にする

### 3.1 Defect statement

`RuntimeIntentCoordinator::reclaimInFlightCount_` は「保留中（deferred）reclaim の**試行回数**」として加算されるが、`onReclaimEnd()` は「**同一 identity の reclaim 成功**」で 1 回だけ減算される。**両者を結ぶ不変条件が存在せず**、pending identity を破棄する全経路で `onReclaimEnd()` が呼ばれない。結果、`reclaimInFlightCount_` は一方向に増加しうるラッチ-counter となり、`ShutdownScheduler::isFullyDrained()` の述語 `reclaimInFlightCount_ == 0` を**永久に**偽に保つ。

`Shutdown = complete drain`（Practical Stable ISR Bridge Runtime の終端契約）が破られる。

**破る不変条件: Liveness（shutdown drain predicate の永久 strand）＋ Ownership（counter と identity set の台帳乖離）＋ Authority（drain 完了判定の authority が 2 つの非等価な源の論理積に依拠し、その 1 つが壊れている）**

**優先順位: P1**
- 判定根拠: strand 自体は**不可逆**（同 process 内で二度と解除されない）であり user 定義の P0「permanent strand」に字義では該当する。
- ただし**観測される被害範囲が限定**されているため P1 を採用する:
  - audio thread は stop 済み（RT 経路への影響なし）。
  - user データ消失・crash・UAF は無い（`reclaimShutdownQuiescent` は epoch gate を bypass するが *shutdown 確定後* のみ、且つ Terminal/T2 destroy は別経路で進行する）。
  - 観測結果は (a) `waitForDrain(2000, 2)` が budget 満了で false を返す（shutdown ごとに 2 秒の sleep ループ）、(b) `markShutdownComplete()` が `CoordinatorState::Faulted` を書込む、(c) `~AudioEngine` が `[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete` を出力する。
- Owner が P0 と判定すべきなら上位格上げであり、本書はその再判定を妨げない。

### 3.2 Exact source location

| # | file:line | 内容 |
| --- | --- | --- |
| L1 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:211-213` | `onReclaimBegin()` = `fetchAdd(reclaimInFlightCount_, 1)` |
| L2 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:224-230` | `onReclaimEnd()` = `old>0` ガード付き `fetchSub(1)`。**underflow は Faulted 化しない（no-op）** |
| L3 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:700-708` | `reclaimNormal` の **defer 分岐のみ**で `onReclaimBegin()`（**唯一の非対称 `+1`**）。`return false` |
| L4 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:710-717` | `reclaimNormal` の **成功分岐のみ**で `onReclaimEnd()`（**唯一の非対称 `-1`**） |
| L5 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:748-778` | `reclaimShutdownQuiescent` が `handleRuntime.reclaim(handle)`（L776）で `Reclaimed` にするが **`onReclaimEnd()` を一度も呼ばない** |
| L6 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:784-809` | `QuarantineService::executeQuarantine` が `handleRuntime.quarantine()`（L794）で `Quarantined` にする。counter に触れない |
| L7 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:542` | `ShutdownScheduler::isFullyDrained()`: `&& consumeAtomic(reclaimInFlightCount_) == 0` |
| L8 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:568-579` | `markShutdownComplete()`: `isFullyDrained()` false → `publishAtomic(state_, CoordinatorState::Faulted)`（L577） |
| L9 | `src/audioengine/AudioEngine.Retire.cpp:98-133` | pending identity の再試行ループ。**L109 `if (isRetired(handle))` の else 側（entry 破棄）に `onReclaimEnd()` が無い** |
| L10 | `src/audioengine/AudioEngine.h:4602-4629` | `requestReclaimHandle`。L4606-4607 で pre-check → L4612 `requestReclaim` false → L4614-4618 push |
| L11 | `src/audioengine/AudioEngine.h:4560-4591` | `retireDSPHandleForRuntime`（L4586 retire → L4587 `requestReclaimHandle`） |
| L12 | `src/audioengine/AudioEngine.Threading.cpp:193-212` | `AudioEngine::isFullyDrained()`。L205 `pendingReclaimEmpty` と L212 `runtimePublicationBridge_.isFullyDrained()` を**論理積**で要求 |
| L13 | `src/audioengine/AudioEngine.Threading.cpp:215-247` | `waitForDrain` = `while(!isFullyDrained()) { drainDeferredRetireQueues(true); … }`（L235-241、budget 超過で `return false`） |
| L14 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:499-517` | `dspHandleRuntime_.retire(activeHandle)`（L504）→ `tryShutdownQuiescentReclaim(activeHandle)`（L507）= **sink L5 の決定的な到達点**。`fadingHandle` も L513-514 同様 |
| L15 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-632` | `waitForDrain(2000, 2)` → `timedOut` → `shutdownRuntime_.markTimedOut(reason)` |
| L16 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:724` | `runtimePublicationBridge_.markShutdownComplete()` = **sink L8** |
| L17 | `src/audioengine/AudioEngine.Threading.cpp:63-106` | `AudioEngine::quarantineSlot`。L98 `lifetimeMgr.retire(dsp)`、**L102 `dspHandleRuntime_.quarantineSlot(slot)`**（= sink L6 相当の state 遷移） |
| L18 | `src/audioengine/AudioEngine.Commit.cpp:620-644` | retire deferral 閾値超過時に `quarantineSlot(pendingSlot, generation, RetireDeferralTimeout)` を呼ぶ（L623 / L643） |
| L19 | `src/audioengine/AudioEngine.Commit.cpp:662-681` | quarantine 解放時 `dspHandleRuntime_.destroyQuarantineSlot(qslot, 0)`（L678）— counter に触れない |
| L20 | `src/audioengine/ISRDSPHandle.cpp:122-127` | `DSPHandleRuntime::retire()` は **state を無条件に `Retired` へ上書き**（generation check なし） |
| L21 | `src/audioengine/ISRDSPHandle.cpp:129-148` | `reclaim()`: `Reclaimed` 化 + free-list push（counter 非反映） |
| L22 | `src/audioengine/ISRDSPHandle.cpp:176-182` / `:198-252` | `quarantineSlot()` / `destroyQuarantineSlot()`（counter 非反映） |
| L23 | `src/core/EpochDomain.h:214-248` | `getMinReaderEpoch()` は `minEpoch = currentEpoch()` で初期化し、active reader の epoch でのみ下限を下げる。**active reader 0 なら `minReaderEpoch == currentEpoch()`** |
| L24 | `src/core/EpochDomain.h:193-202` | `publishEpoch()` = `fetchAdd(globalEpoch, 1)`。呼出元は `AudioEngine.Publication.cpp:18`（`advanceEpoch`）／`:28`（`advanceRetireEpoch`）、`SnapshotCoordinator.cpp:91/109`（`startFade` / `switchImmediate`）、`EQProcessor.Core.cpp:75`、`ConvolverProcessor.StateAndUI.cpp:1087`、`ConvolverProcessor.LoadPipeline.cpp:811`、`AudioEngine.Processing.ReleaseResources.cpp:263/305` |
| L25 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:285-311` | graceful drain ループが **10 ms 周期・最大 5000 ms（最大 500 回）で `m_retireRouter->publishEpoch()` を呼ぶ**（L305）。shutdown 中に `globalEpoch` を能動的に前進させる |

**`reclaimInFlightCount_` の mutator は L1 と L2 の 2 つだけ**（`grep onReclaimBegin|onReclaimEnd` で production 側 3 箇所のみ確認: `AudioEngine.Retire.cpp:62/66` と `:356/359` は同一スコープ内の balanced pair で、`:705` と `:716` が非対称 pair）。**reset / reconcile / clamp は production 側に一切存在しない**（`finalizeShutdown` も触らない）。

### 3.3 Preconditions

1. `DSPHandleRuntime` に有効 handle が 1 つ存在する（`create()`）。
2. NonRT スレッド（CoordinatorLoop / MessageThread / RebuildThread）が `retireDSPHandleForRuntime`（L11）または `DSPLifetimeManager::retireByHandle`（`DSPLifetimeManager.cpp:114-115`）経由で `requestReclaimHandle` を呼ぶ。
3. caller 側 pre-check（L10 の L4606-4607）が `retireEpoch < minReaderEpoch` を**満たす**（= TOCTOU window に入る）。
4. `reclaimNormal` 内部の再確認（L4 の L698-699）では `retireEpoch >= minReaderEpoch` に**反転している**。
   - 実装上の必然性: `EpochDomain::getMinReaderEpoch()`（L23）は active reader 0 なら `currentEpoch()` と等しい。よって pre-check が真になるには「pre-check の 2 連続 load の間に `currentEpoch()` が 1 前進し、かつ全 active reader がそれより新 epoch を持つ」必要がある。
   - **この事象は production で予期されている**: `AudioEngine.h:4600-4601` と `AudioEngine.Retire.cpp:116` の両コメントが「事前チェック後に epoch が進み unsafe になった場合、false が返る → 保留リストへ再登録（TOCTOU 対策）」と明記し、その false 戻り用の分岐を専用に実装している。
   - **発生率について（断定しない）**: `globalEpoch` は L24 に列挙した多くの NonRT 経路から前進する。特に graceful drain ループ（L25）は **10 ms 周期で最大 500 回** `publishEpoch()` を呼ぶため、shutdown 付近では `currentEpoch()` の前進が TOCTOU window を狭く埋める。本書は「必ず発生する」とは主張せず、**production が明示的に前提としている事象が 1 回でも起きた場合の本当の defect** として主張する。
5. その後、当該 handle が `Retired` 以外の state へ移る（quarantine / 他経路の reclaim）。

### 3.4 Reproduction / counter-example（source 上 disprove 不能）

**前提イベント E1（TOCTOU deferral、1 回）**
```
t0: NonRT: requestReclaimHandle(H)
      retireEpoch  = router.currentEpoch()      = E          (AudioEngine.h:4606)
      minReader    = router.minReaderEpoch()   = E+2        (AudioEngine.h:4607)   → pre-check 真
    runtimePublicationBridge_.requestReclaim(H, …)          (AudioEngine.h:4612)
t1: 他スレッド: publishEpoch()  → globalEpoch = E+1          (EpochDomain.h:199)
t2: NonRT: reclaimNormal:
      handleRuntime.retire(H)                               (Coordinator.cpp:695)  → state=Retired
      retireEpoch  = router.currentEpoch()      = E+1        (Coordinator.cpp:698)
      minReader    = router.minReaderEpoch()   = E+1        (Coordinator.cpp:699)  ← t1 の影響
      retireEpoch >= minReaderEpoch  →  TRUE                 (Coordinator.cpp:700)
      onReclaimBegin()   → reclaimInFlightCount_ = 1        (Coordinator.cpp:705)  ★+1
      return false                                               (Coordinator.cpp:707)
t3: NonRT: pendingReclaimHandles_ ← {H}                     (AudioEngine.h:4617)
```

**後続イベント E2（sink。いずれを選んでもよい）**
```
E2-a (quarantine sink — L17/L18/L6):
     onRuntimeRetiredNonRt の retire-deferral 閾値超過
       → quarantineSlot(pendingSlot, …, RetireDeferralTimeout)   (Commit.cpp:623 / :643)
       → dspHandleRuntime_.quarantineSlot(slot) → state = Quarantined   (Threading.cpp:102)

E2-b (shutdown quiescent sink — L5/L14 — releaseResources が必ず到達する終端経路):
     releaseResources:
       dspHandleRuntime_.retire(activeHandle)                  (ReleaseResources.cpp:504)
       tryShutdownQuiescentReclaim(activeHandle)               (ReleaseResources.cpp:507)
         → reclaimShutdownQuiescent:
             handleRuntime.retire(activeHandle)                 (Coordinator.cpp:770)
             handleRuntime.reclaim(activeHandle)                (Coordinator.cpp:776) → state=Reclaimed
             ★ onReclaimEnd() は存在しない（L748-778 全体を読み切った上で確認）
```

**後続イベント E3（discard sink — L9）**
```
    waitForDrain(2000, 2)                                      (ReleaseResources.cpp:615)
      while (!isFullyDrained()) { drainDeferredRetireQueues(true); … }   (Threading.cpp:235-237)
        drainDeferredRetireQueues:
          pending.swap(pendingReclaimHandles_)                 (Retire.cpp:96)   → H を取り出す
          if (dspHandleRuntime_.isRetired(H))                  (Retire.cpp:109)
              …  ← state は Quarantined / Reclaimed なので FALSE
          ★ else 側が存在しない。entry は破棄され onReclaimEnd() も呼ばれない
                 → pendingReclaimHandles_ は空になる（Layer-1 の P2 は充足する）
```

**到達する final state**
```
pendingReclaimHandles_        == []     (空)
reclaimInFlightCount_         == 1      (永久)
→ ShutdownScheduler::isFullyDrained() は L542 で false           (Coordinator.cpp:542)
→ AudioEngine::isFullyDrained() は L212 の委譲で false            (Threading.cpp:212)
→ waitForDrain は L240 で budget 超過し false を返す（2000 ms）  (Threading.cpp:240)
→ shutdownRuntime_.markTimedOut(...)                              (ReleaseResources.cpp:632)
→ markShutdownComplete() → isFullyDrained() false → Faulted      (Coordinator.cpp:577)
→ [FAULT] ~AudioEngine: coordinator in Faulted state …           (CtorDtor.cpp:310-312)
```

**不可逆性の証明**: `reclaimInFlightCount_` の mutator は `onReclaimBegin`（+1, L1）と `onReclaimEnd`（−1 ガード付き, L2）**のみ**。`+1` の発生源は L3 のみ。`−1` の発生源は L4 のみ。E2/E3 のいずれの sink も L4 を通らないため、以後 `reclaimInFlightCount_` を 0 に戻す実行可能な経路が存在しない。`onReclaimEnd` の `old > 0` ガード（L2/L226-228）は underflow を Faulted 化しないため、この誤差を検出も回復もしない。`finalizeShutdown`（`ReleaseResources.cpp:664`）も counter に触れない。→ **同 process 内の全後続 drain 試行が同一理由で失敗する。**

### 3.5 Expected invariant（破られるもの）

- **EI-1（Liveness）**: shutdown 完了条件 `isFullyDrained()` は、全 reclaim work が完了すれば真になる。実 reclaim work は E3 時点で**完全に完了している**（H は `Reclaimed`、free-list へ返却済み、DSPCore 破棄は Terminal/T2 経路で進行済）のに述語が偽である。
- **EI-2（Ownership）**: 「`reclaimInFlightCount_` の値 == `pendingReclaimHandles_` に残る deferred identity 数」という等価性が不変であるべき。本 defect はこれを破る（counter 1 / identity 0）。
  - なお `AudioEngine.Threading.cpp:193-196` のコメントは**逆向きの不整合（counter 過少）**しか認識していない:「`reclaimInFlightCount_` だけでは複数 pending reclaim を正確に数えられない（A pending + B pending で A 成功 → count=0 なのに B が残る）」。**過多方向（count>0 なのに identity 0）は未認識**であり、`pendingReclaimEmpty`（L205）を**論理積**で足すこと（OR でなく AND）が、L542 の独立 predicate と非対称な二重計上を生む。
- **EI-3（Authority）**: drain 完了の判定 authority は単一の観測可能な実測集合であるべき。現状は `reclaimInFlightCount_`（近似 counter・L1/L2 が非対称）と `pendingReclaimHandles_`（authoritative identity set・L9/L10 が非対称）の**論理積**であり、両者の間に **reconciliation が存在しない**。片方の非対称性は false-positive（drain 遅延）、他方の非対称性は **false-negative（drain 恒久不能）** を生む。判定の独立性が保証されない。

### 3.6 Actual state transition

```
[正常]  H: Active → Retired →(epoch 安全)→ Reclaimed     reclaimInFlightCount_: 0 → 0
[欠陥]  H: Active → Retired →(TOCTOU deferral +1)→ [pending 登録]
              → (sink) Quarantined / Reclaimed          reclaimInFlightCount_: 0 → 1  ★ 以後 -1 経路なし
              → pending entry 破棄（onReclaimEnd 無し）
              → reclaimInFlightCount_: 1（恒久）
[結果]  isFullyDrained(): true であるべき ⟹ 恒久 false
        markShutdownComplete(): Bootstrapping ⟹ Faulted
```

### 3.7 Observable failure

| # | 観測点 | 観測値 |
| --- | --- | --- |
| O1 | `AudioEngine::isFullyDrained()`（`AudioEngine.h:1600` public） | 恒久 `false`（全 Layer-1 実測が 0、`pendingReclaimHandles_` も空） |
| O2 | `AudioEngine::waitForDrain(2000, 2)`（`ReleaseResources.cpp:615`） | budget 満了で `false`。shutdown ごとに 2 秒の sleep ループ |
| O3 | `ShutdownRuntime::getPhase()` / blocking reason | `TimedOut`（`markTimedOut(Unknown)` — `stuckReaderCount==0` かつ `rebuildThreadIsRunning==false` なので `Unknown`） |
| O4 | `RuntimeIntentCoordinator::getState()` | `CoordinatorState::Faulted`（`Coordinator.cpp:577`） |
| O5 | ログ `~AudioEngine` | `[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — residual intents may remain in System 1 queues`（`CtorDtor.cpp:311-312`）。**本文が誤誘導**（残存 intent は 0、起因は counter 誤差） |
| O6 | ログ `releaseResources` | `[DIAG] releaseResources: drain timeout reached, performing safe tryReclaim (drainAll skipped)`（`ReleaseResources.cpp:657`） |
| O7 | `reclaimInFlightCount_` | `getReclaimInFlightCount()`（`Coordinator.h:219`）が 0 に戻らない |
| O8 | `collectDrainAudit()` | **全 component 0**。O1 の原因を露出しない（`pendingReclaimHandles_` と coordinator 内部は非露出 — R33-A §11 と同じ構造的盲点） |

診断可能性: **O8 が特に問題**。既存の唯一の shutdown 診断スナップショット（`collectDrainAudit`）が「全 0」を報告するため、O1 の原因を既存 instrumentation から特定できない。`[FAULT]` ログの文言も原因を誤る。

### 3.8 Why recovery / drain does not fix it

1. `waitForDrain`（`Threading.cpp:235-247`）は `drainDeferredRetireQueues(true)` を回すだけである。E3 で `pendingReclaimHandles_` は既に空であり、**回す対象が残っていない**。
2. `drainDeferredRetireQueues` の L109 の else 分岐は `onReclaimEnd()` を含まない。**drain を何度回しても counter は減らない**。
3. `reclaimShutdownQuiescent`（`Coordinator.cpp:748-778`）と `DSPHandleRuntime::quarantineSlot` / `destroyQuarantineSlot` は Coordinator counter を触らない。
4. `onReclaimEnd` の `old > 0` no-op ガードは**過多を是正しない**。ガードは underflow（過少）対策であり、本 defect は過多である。
5. `setReclaimInFlightCount` は D101-32-D で**削除済み**（`Coordinator.cpp:277-280`）。reset/reconcile 用の production 手段が**存在しない**。
6. `finalizeShutdown`（`ReleaseResources.cpp:664`）は counter を触らない。
7. `markShutdownComplete` は `Faulted` への**一方向**遷移であり、回復経路を持たない（`Coordinator.cpp:568-579`）。

### 3.9 Capacity / ownership / lifecycle impact

- **Ownership**: `reclaimInFlightCount_` は「保留 reclaim の件数」という所有権会計の**台帳**である。台帳が実体（`pendingReclaimHandles_`）より大きいまま凍結する。RSS 上のリーク（DSPCore / slot ）は本 defect 直接の成果物**ではない**（`Reclaimed` 化と free-list 返却は `DSPHandleRuntime::reclaim` / `destroyQuarantineSlot` が正常単体で実行する）。異常なのは **drain 判定の台帳**である。
- **Capacity**: `drainDeferredRetireQueues` は `waitForDrain` の 2 ms ポーリングで最大 1000 回（`boundedTimeoutMs=2000` / `boundedPollIntervalMs=2`）回転する。その間 slot は既に解放済だが shutdown は進行しない。RT は停止済みなので audio glitch はない。
- **Lifecycle**: `CoordinatorState::Faulted` が process 寿命 동안持続。`markShutdownComplete` は `ShuttingDown` 以外の state では no-op（`Coordinator.cpp:570-572`）なので、**再 publish で復帰できない**。AudioEngine の再初期化（同一 process 内で engine を作り直す運用）でも counter は新 Coordinator で 0 になるが、同一 engine の再 drain は回復しない。
- **Bounding**: 影響は bounded（1 イベントにつき +1、最大 2^64 まで増加しうるが実質 1〜少数）。**無制限の資源消費はない**。これが P0 ではなく P1 とした主因。

### 3.10 Existing test coverage

**カバーしているもの（すべて balanced ケース）**

| test | file:line | 内容 |
| --- | --- | --- |
| `testInv3_1*` | `src/tests/invariant_INV3_INV5.cpp:132-141` | 単発成功。`getReclaimInFlightCount() == 0`（L140） |
| `testInv3_2ReclaimDeferredThenSucceeds` | `src/tests/invariant_INV3_INV5.cpp:151-192` | defer 1 回（`== 1`, L176）→ epoch 安全化 → 成功（`== 0`, L186）。**`TestEpochProvider` で currentEpoch / minReaderEpoch を独立設定するため TOCTOU を再現できない** |
| （1 件） | `src/tests/invariant_INV3_INV5.cpp:282` | 4 例目の `getReclaimInFlightCount() == 0` |

**カバーしていないもの（= 本 defect の uncovered 空間）**

1. **TOCTOU deferral**（caller pre-check 真 → 内部再確認偽）を再現する装置が production にも test にも存在しない。`TestEpochProvider` は 2 値を独立に固定するため、2 連続 load の間の `publishEpoch()` を再現できない。
2. `reclaimShutdownQuiescent`（`invariant_INV3_INV5.cpp:277 / 687 / 726 / 754`）は `getReclaimInFlightCount()` を**一度も assert していない**。L5 の「counter に触れない」性質は未検証。
3. `AudioEngine::quarantineSlot` → `dspHandleRuntime_.quarantineSlot`（`Threading.cpp:102`）が pending reclaim identity の state を `Retired` から外し、かつ pending entry を破棄する経路の test が**存在しない**。
4. `AudioEngine::drainDeferredRetireQueues` を直接駆動する test が**存在しない**（`grep drainDeferredRetireQueues src/tests` = 0 件）。
5. `reclaimInFlightCount_ > 0` かつ `pendingReclaimHandles_.empty()` の状態で `isFullyDrained()` が false になること（= 偽の terminal strand）を検査する test が**存在しない**。
6. `AudioEngine::isFullyDrained()` は `P1PolyphaseGainCharacterization.cpp:1107 / 1323` で**観測記録されるのみ**（`fullyDrained` フィールド）。assert は無い。
7. `PublishPipelineIntegrationTests.cpp:656-658` は terminal success 判定から `isFullyDrained()` を**意図的に除外**している（「`isFullyDrained()` は recovery obligation / intent residency を含む広い判定であり」）。したがって shutdown の正常系 gate は本 defect を検出できない。
8. `drainDeferredRetireQueues` は `reclaimInFlightCount_` を読むが、`collectDrainAudit()` は露出しない（L3.7 の O8）。

**結論: 既存 42/42 CTest は本 defect を捕捉できない。**

### 3.11 Minimal repair boundary（提案のみ・本 STG では実装しない）

本 STG は read-only であり、以下は**所有者が GO した後の最小境界の提案**に留める（実装・改変は行わない）。

**R-0（前提の明示化・最小）**: `reclaimInFlightCount_` の意味論を「deferred **identity** 数」から「deferred **試行** 数」へ再定義するか、あるいは「retry-unaware counter」である旨を `ISRRuntimePublicationCoordinator.h:993-994` に明記する。現状のコメント（L222-223「意味論 = 保留中（deferred）の reclaim 数」）は実装と一致していない。

**R-1（推奨・局所的）**: identity を単位に `+1` / `-1` を行う。最小形は
- `+1` を `reclaimNormal` の defer 分岐（L705）から**呼び出し元**（`AudioEngine.h:4612-4618` / `AudioEngine.Retire.cpp:117-123`）へ移し、**pending への push と同一の临界点でペアにする**（re-push しない drop では `+1` しない）。または
- `pendingReclaimHandles_` に `ReclaimIdentity` として登録済みを明示し、`reclaimNormal` の defer 分岐で「同一 identity が既に登録済みなら `+1` しない」ガードを入れる（dedup 化）。

**R-2（sink 対策・必要）**: pending entry を破棄する **全 sink** で `onReclaimEnd()` を対で呼ぶ。対象:
- `AudioEngine.Retire.cpp:109` の `else`（entry 破棄）— **最重要**
- `reclaimShutdownQuiescent`（`Coordinator.cpp:776`）— reclaim-in-flight に未解除の identity があれば `onReclaimEnd()` で対にする
- `DSPHandleRuntime::quarantineSlot` / `destroyQuarantineSlot` 呼び出し側（`Threading.cpp:102` / `Commit.cpp:678`）

**R-3（predicate 強化）**: `AudioEngine::isFullyDrained()`（`Threading.cpp:204-212`）の L205 `pendingReclaimEmpty` と L212 の委譲は、**互いに独立した 2 つの非等価な源の論理積**である。`reclaimInFlightCount_` を authority にするなら identity set をその**唯一**の source of truth とし（`ISRLifetimeProof.h:60` は既に「`pendingReclaimHandles_` を本 identity set へ昇格する」ことを示唆している）、counter 述語を削除するか、両者の reconcile を drain 時に行う。

**R-4（診断）**: `collectDrainAudit()`（`AudioEngine.Threading.cpp:109+`）に `reclaimInFlightCount` を露出する。現状は「全 audit component 0 なのに `isFullyDrained()==false`」という**診断不能な状態**を実運用で生成しうる。`CtorDtor.cpp:311-312` の `[FAULT]` ログ文言も原因を誤誘導するため、counter 値を含めるべき。

**影響ファイル（推定）**: `ISRRuntimePublicationCoordinator.cpp` / `.h`、`AudioEngine.h`、`AudioEngine.Retire.cpp`、`AudioEngine.Threading.cpp`（+ テスト）。`ConvoPeq.md` の更新も必要（invariant 契約の記述が実装と乖離: `ISRRuntimePublicationCoordinator.cpp:216-223` の「意味論 = 保留中（deferred）の reclaim 数」という記述が identity 単位でないことを 반영していない）。

### 3.12 ISR / authority impact

- **RT 影響: なし。** `reclaimInFlightCount_` の mutator（`onReclaimBegin` / `onReclaimEnd`）は `fetchAdd` / `consume+fetchSub` の atomic のみ（L212/L225-227）。RT 経路から直接は呼ばれない。`publishEpoch()`（`EpochDomain.h:199`）も `fetchAdd` のみ。本 defect の全コードは NonRT である。
- **Practical Stable ISR Bridge Runtime 準拠**: 本 defect の修正は **RT no-wait / no-lock / no-alloc / no-delete / no-decision** を 위반しない。`pendingReclaimHandlesMutex_`（`Retire.cpp:95/119/127`）は既に NonRT のみ。R-1/R-2 の修正も NonRT 限定。
- **Authority 影響（重大）**:
  - **Coordinator sole authority for reclaim**: `reclaimInFlightCount_` は Coordinator が reclaim の唯一の accounting authority であるべき値（`ISRRuntimePublicationCoordinator.h:993-994`「production wired が authority のため KEEP」）。しかし authority の**implementer**（`reclaimNormal`）と**invalidator**（`DSPHandleRuntime` の state 遷移 + pending list の drop）が**別モジュール・別ファイル**にあり、両者の整合を担保する機構が無い。→ authority 境界が実装上 leak している。
  - **RuntimeWorld immutable**: 本 defect は RuntimeWorld の不変性には触及しない（`RuntimePublishWorld` の state は変更しない）。
  - **Overflow ≠ silent loss**: 本 defect は silent loss の**逆**（silent **over**-count → silent drain failure）。同種の「観測されない state corruption」に属する。
  - **Shutdown = complete drain**: **最も直接的に破る契約**。「drain 完了」は真の完了を反映しない。

### 3.13 Regression scope（修正時に影響を受ける範囲）

| 範囲 | 詳細 |
| --- | --- |
| 直接 | `onReclaimBegin` / `onReclaimEnd` の全 caller（`AudioEngine.Retire.cpp:62/66`、`:356/359`；`Coordinator.cpp:705/716`）。特に balanced pair 群（`:62/:66`、`:356/:359`）を誤って触ると無関係な回帰を招く |
| 影響する不変条件 | INV-3-1（単発成功で count 0 のまま）、INV-X3-5（`pendingReclaimHandles_` が reclaim pending の source of truth）、INV-D162-8（EBR residual）、`T2 = isFullyDrained()`（R33-B §「terminal contract」）、`markShutdownComplete` の Bootstrapping/Faulted 分岐 |
| 影響する観測 | `collectDrainAudit()`（R-4 で追加する場合、全 shutdown ログの書式が変わる）、`RuntimeDrainAudit` 系 snapshot、evidence JSON（`emitEvidenceTickNonRt` が `isFullyDrained` を間接含む） |
| 影響する test | `invariant_INV3_INV5.cpp`（140/176/186/282 の 4 assertion の期待値が変わる可能性。**R-2 を採ると `testInvX3_4ReclaimModeQuiescent`（:204-）の Permit 経路で counter を触るため新規 assert 追加が要る**）、`ISRSemanticValidationTests.cpp:330-450`（`isFullyDrained` / `markShutdownComplete` の状態遷移を検証）、`PublishPipelineIntegrationTests.cpp:539-663`（terminal success 経路） |
| 影響しない | RT audio callback、IR loader、preset、publication admission、recovery obligation table（STG-8 D1–D3 の領域）、EpochDomain の grace period 判定そのもの（`minReaderEpoch` の意味論は不変） |
| 既存 CLOSED 項目への影響 | **なし**。STG-1 / STG-2 / STG-4-1 / STG-6-D1 / STG-7-D1 / STG-8-D1 / STG-8-D2 / STG-8-D3 と failure path もコード領域も非交差 |

---

## 4. 調査したが昇格しなかった候補（記録）

昇格基準（source 上 disprove 不能な反例 + 既存 test が捕捉不能）を満たさなかったもの。**いずれも defect として主張しない。**

| # | 候補 | 却下理由 |
| --- | --- | --- |
| R-1 | `QuarantineService::executeQuarantine`（`Coordinator.cpp:784-809`）が `DSPLifetimeManager::retire` を呼ばず DSPCore を破棄しない | **設計意図が明示されている**（L805-806「Audit 失敗時も State は変更しない。Rollback 廃止。New World Publish が復旧担う」＋ `Threading.cpp:82-96` の orphan 意味論確定コメント (a)–(d)）。quarantine 解放は `Commit.cpp:677-679` の 3 系統で必ず走る。状態不変ではない |
| R-2 | `AudioEngine.Commit.cpp:492` `slot = world->generation % 256u` と `DSPHandle.slot`（1..255）の名前空間衝突 | `RetireIntent` は world generation から**導出される**値であり `DSPHandle.slot` とは別 namespace（`intent.generation` も同時に保持し、`emitIntent(pendingSlot, pending.generation)` と `lifetime()` 側が token 付きで照合）。衝突的后果を source 上で示せなかった |
| R-3 | `ISRRetire.cpp:54-56` が `reinjectRetryCount = 0` を hardcode し `kMaxReinjectRetries = 10`（`Coordinator.cpp:315/343`）が到達不能 | **bounded diagnostics**（P3）。retry budget の未使用は資源消費・安全性に影響しない |
| R-4 | `SnapshotCoordinator::quarantineRetireSink`（`SnapshotCoordinator.cpp:28-29`）の `assert(false)` が NDEBUG で消える | `quarantineRetire` が false = RetireQuarantineStore が full。drop しないこと自体は**意図的**（L19-20「store full 時も deleter を実行しない（capacity exhaustion は health escalation で先行検知）」）。assert 消滅は diagnostics 欠落（P3）。store full 自体は health escalation で先行検知される契約 |
| R-5 | `ownerChannel().drainAllNonRt`（`ReleaseResources.cpp:677-693`）が epoch を前進させるため `terminalReclaim` が store のみ | epoch 前進は**安全側**（grace period を短くする）であり、内部不変の反転ではない。`AudioEngine.h:4471-4473` のコメントとの字面不一致は**コメントの陳腐化**（P3 documentation）であり、runtime の state corruption は示せなかった |
| R-6 | CoordinatorLoop join 後に `submitRecoveryIntent` が `discardRecoveryRequestsOnShutdown` 通過後に新 obligation を作る | `discardRecoveryRequestsOnShutdown`（`Coordinator.cpp:1457`）の producer join 順序を source 上で最後まで切り分けられなかった（= 仮説段階）。**Concrete Defect には昇格しない** |
| R-7 | epoch wraparound / tombstone 会計 / TerminalReclaim 二重 enqueue / `escalateAllRetires` | 既 refute（`isOlder` は epoch 順序比較であり wrap-around に対して閉じている、tombstone は SPSC で単一 consumer、TerminalReclaim は `isSwapPending` ゲートで排他） |
| R-8 | `resetFadeStateAndRetireTarget` が未使用 private 関数 | cppcheck の唯一の指摘。**reachability なし**。dead code は defect ではない |

---

## 5. 既存 CLOSED 項目の再オープン検査

| CLOSED 項目 | 内容 | STG-9 との交差 |
| --- | --- | --- |
| STG-1 | bypass staging clobber | なし（preset/bypass 領域） |
| STG-2 | order-stale headroom | なし（ordering 領域） |
| STG-3 | NO CONCRETE DEFECT FOUND | なし |
| STG-4 / STG-4-1 | Loader / IR lifecycle | なし（IR ファイル path） |
| STG-5 | CLOSED（欠陥なし） | なし |
| STG-6-D1 | sync preset load の silent drop（short IR × upsampling） | なし（upsampling 経路） |
| STG-7-D1 | `requestLoadState` が `autoGainStagingEnabled` を復元しない | なし（gain staging flag） |
| STG-8-D1 | Deferred-discard が recovery obligation を終端化しない（Live slot leak） | **なし**。STG-8-D1 は `RuntimePublicationOrchestrator` の deferred slot × `RecoveryAdmissionTable` の obligation 終端化。STG-9-D1 は `reclaimInFlightCount_`（Coordinator 内部 counter）× `pendingReclaimHandles_`（AudioEngine member）× drain predicate。**failure path・ファイル・Symbol が非交差** |
| STG-8-D2 | Deferred-overwrite が旧 recovery obligation を孤立させる | なし（同上） |
| STG-8-D3 | `RejectedPressure` の transport recovery が停滞 | なし（intent queue rearm/redrive） |

`doc/work113/*STG-*` 全 26 ファイルに対する `reclaimInFlightCount_|onReclaimBegin` grep = **0 件**。STG-9-D1 は既存のどの記録とも**同一 failure path の別名ではない**。

**Closed items re-opened: 0**

---

## 6. ISR-authority audit

| 検査項目 | 結果 |
| --- | --- |
| RT no-wait | **遵守**。本 defect の全コードは NonRT。`onReclaimBegin`/`onReclaimEnd` は atomic のみ |
| RT no-lock | **遵守**。`pendingReclaimHandlesMutex_` は NonRT のみ（`Retire.cpp:95/119/127`） |
| RT no-alloc | **遵守**。`pending` vector の swap は NonRT 限定 |
| RT no-delete | **遵守**。DSPCore 物理削除は `enqueueWithRetry`（EBR）が担当。`reclaimShutdownQuiescent` は slot 状態遷移のみ |
| RT no-decision | **遵守** |
| Coordinator sole reclaim authority | **破る（本 defect の Authority 側面）**。`reclaimInFlightCount_`（Coordinator が保持）と `DSPHandleRuntime` の state 遷移（別モジュール）が、`onReclaimEnd` を共有せずに同じ identity の寿命を管理している |
| Retire through Epoch | **遵守**。本 defect は epoch gate の判定ロジック（`minReaderEpoch` の意味論）を変更しない。壊すのはその**後**の accounting |
| RuntimeWorld immutable | **遵守**。`RuntimePublishWorld` の state は変更しない |
| Overflow ≠ silent loss | 本 defect は silent loss の**逆方向**（silent over-count → silent drain failure）。同種の無観測状態生成に関与 |
| Shutdown = complete drain | **破る**。drain 完了が真の完了を反映しない |

**ISR-authority 違反: 1 件**（Coordinator sole reclaim authority の accounting 一貫性）。RT 構造の違反: 0 件。

---

## 7. 終了

**Case A — STG-9 STOP（Concrete Defect 提示）**

- 昇格: **STG-9-D1 1 件**（P1: `reclaimInFlightCount_` の非対称 counter → `isFullyDrained()` 恒久 false → `CoordinatorState::Faulted`）
- 却下: 8 候補（§4）
- 既存 CLOSED 項目の再オープン: **0 件**
- ISR-authority 違反: 1 件（accounting 一貫性。RT 構造違反 0 件）
- 本 STG は read-only。production / test / CMake / `ConvoPeq.md` の変更なし。commit なし。push なし。

**Owner の GO なしに修正へ進まない。**
