# Phase I-T1-D101-2.5O-3A — Accepted→Execution Data-Flow Census

> **Verdict: CENSUS COMPLETE — O→E binding candidate をコードから抽出 / OwnerChannelKey{seq, epoch, mappedGen} が最強の既存 E 側 key / correlationId は E pipeline に流入しない / B1〜B4 の正式判定は O-3B/C/D へ**
> **コード変更 0 / 新 permit・token・ID・deadline・timeout・field 追加 0 / evaluate・evaluateDeferred 変更 0**

- **一次ソース**: `ConvoPeq.md` Generated 2026-08-22 15:43:20 版 + `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}` / `src/audioengine/PublicationExecutor.{h,cpp}` / `src/audioengine/RuntimePublishExecutor.h` / `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` / `src/audioengine/RuntimeWorldAuthority.h` / `src/audioengine/AudioEngine.h:4499-4620` / `src/audioengine/SequenceArithmetic.h`
- **位置づけ**: O-3（Execution Binding Contract Closure）の第 1 段階。設計案は作らず、現行 execution data-flow をコード行単位で全追跡し binding candidate のみを抽出する

---

## O-3-A: Accepted → Execution 実コード経路（行レベル全追跡）

### Main path（同期 transaction）

```text
[1] trySubmit(req) → trySubmitImpl(req)                      Orchestrator.cpp:34-38
[2] admission_.evaluate(req, engine_, pubCtx) → Accepted     cpp:48-55
    （evaluate() 必須・バイパス禁止 — M-C1 PASS の主体）
[3] correlationId = nextCorrelationId()                      cpp:57
    stateOwner_.onSubmitted(shortValue)                      cpp:58
    telemetryRecorder_.recordProgress(Submitted, nowUs)      cpp:66-70
[4] resolveDSPHandle(req.newDSP) / oldHandle 解決            cpp:76-84
[5] RuntimeBuilder build → builder 内部で
    reserveRuntimePublicationIdentity()                      RuntimeBuilder.cpp:63,165
    → {generation, worldId, publicationSequence} を world に stamp
[6] FrozenRuntimeWorld wrap                                  cpp:~200
    aligned_make_unique<FrozenRuntimeWorld>(
      aligned_unique_ptr<RuntimeState>(const_cast<RuntimeState*>(worldOwner.release())))
[7] executor_.publish(engine_, std::move(frozen),
      req.newDSP, oldHandle)                                 cpp:~215
    = publishImpl(..., waitForReceipt=true)                  PublicationExecutor.cpp:8-13
[8] publishImpl:
    frozen->releaseState() → rawState                        PublicationExecutor.cpp:17-30
    worldGen = rawState->generation / worldId = rawState->worldId 保存
    stateOwner = aligned_unique_ptr<RuntimeState>(rawState)
[9] engine.commitRuntimePublication(                         PublicationExecutor.cpp:40-48
      std::move(stateOwner),
      RegistrationContext::alreadyRegistered(existingHandle),
      oldHandle)
    ※ deferred resubmit 経路は enqueueRuntimePublicationFireAndForget
      （waitForReceipt=false — fire-and-forget）
[10] commitRuntimePublication（async facade）                AudioEngine.h:4499-
     Producer 唯一の publish 入口:
       a. ownerChannel().push(owner, OwnerChannelKey{seq, epoch, mappedGeneration})
       b. registry().registerPublish(seqId, sealedWorld)
       c. intentQueue_ へ Publish intent enqueue（payload は enqueue 時固定）
       d. waitForPublishReceipt(seqId, 250ms) で block        AudioEngine.h:4609-4619
     seqId = world->publication.sequenceId                     SequenceArithmetic.h:33
[11] CoordinatorLoop (1ms tick): processIntent →
     kDispatchTable[Publish] → PublishIntentHandler →
     PublishExecutor::executePublish(authority, intent, ctx)   ProcessIntent.cpp Step 5-2/5-3
[12] executePublish 本体                                      RuntimePublishExecutor.h:31-
     a. owner = authority.ownerChannel().take(
          OwnerChannelKey{intent.sequenceId,
                          epoch, mappedGeneration})           SOLE ownership・moved exactly once
     b. newWorld fallback chain:
        owner.get() → registry.lookup(intent.sequenceId)
        → p.newWorld（sealed snapshot 最終 fallback）
     c. sealRecursively()（publish 前 immutable 化）
     d. authority.publish(std::move(owner),
          PublishMetadata{p.boundary, p.version,
            intent.sequenceId, p.epoch, p.mappedGeneration},
          &committed)
        → 内部: prevWorld=observe() → coordinator_.commit(bake+monotonicity)
        → Faulted check → writeAccess_.publishAndSwap(next)   物理 LP（INV-X4-3）
     e. Execution tail:
        bridge.didPublishRuntimeNonRt(*newWorld)
        bridge.willRetireRuntimeNonRt(oldWorld)
        bridge.retirePublishedRuntimeWorldNonRt(oldWorld)
     f. registry().unregister(intent.sequenceId)
     g. onPublishCompleted → advanceRetireEpoch →
        orchestrator_.onPublishCommitted(intent.sequenceId)
[13] onPublishCommitted(seqId):                              Orchestrator.cpp:313-323
     m_lastObservedSequence 公開 → notifyPublishReceipt(seqId)
     → producer が waitForPublishReceipt から起床
[14] trySubmitImpl 後処理:
     isCommitted(result.stage) 判定 → 成功時 onPublished +
     recordProgress(Published) → return Accepted
     失敗時 destroyRolledBackDSP + onExecutorFailed →
     RejectedShutdown / RejectedPublishFailure
```

### 経路の構造的事実

- **Accepted return 時点で物理 swap は原則完了している**: waitForReceipt=true のため、trySubmitImpl が Accepted を返すのは receipt 受信後。ただし 250ms timeout という race window が存在する（receipt timeout 後に intent が遅延 commit する可能性 — B4 評価の UNPROVEN 領域として記録）
- **fire-and-forget モードが実在**: deferred resubmit は所有権移譲のみで待たない。「enqueue 済み + 所有権移譲済みなので次 tick で commit」— このモードでは Accepted return と物理 swap の順序が異なる
- **newWorld 解決の 3 段 fallback**: owner → registry lookup → sealed snapshot。同一 intent に対する world 解決源が複数存在する（B3 評価の入力）

---

## Binding Candidate 抽出表

| Candidate | 生成点 | 消費点 | binding 強度評価 |
| --------- | ------ | ------ | ---------------- |
| **OwnerChannelKey{sequenceId, epoch(u32), mappedGeneration}** | producer push（commitRuntimePublication） | executePublish take() | **最強の既存候補** — key 一致による単一スロット transfer、「moved exactly once」INVARIANT コメント付き |
| intent.sequenceId | intent enqueue | take key / registry lookup/unregister / onPublishCommitted / receipt watermark | execution identity thread — E 側で一貫 |
| PendingPublishRegistry entry {seqId→world} | producer registerPublish | executePublish lookup + unregister | gap handle（非 authoritative fallback） |
| p.newWorld（intent payload 内 sealed snapshot） | enqueue 時固定 | 最終 fallback | immutable snapshot |
| DSPHandle existingHandle / oldHandle | trySubmitImpl resolve | RegistrationContext / retire intent | DSP lifecycle identity — O identity ではない |
| worldGen / worldId（rawState field） | builder stamp | publish 後 timing history/log のみ | artifact metadata |
| correlationId | post-Accepted（cpp:57） | **telemetry/stateOwner のみ — executor/intent pipeline に流入しない** | O 側進捗相関。O→E binding key としては機能しない |

### 決定的観察

1. **correlationId は E pipeline に到達しない**: trySubmitImpl 内で消費され、executor_.publish 引数にも intent payload にも含まれない。O 側（post-Accepted request）と E 側（sequenceId family）は**異なる ID 系**であり、現行に両者を結合する authoritative field は存在しない
2. **object continuity は実在するが契約ではない**: 同一 RuntimeState object が request→frozen→rawState→stateOwner→ownerChannel→publish→store と移動する（sealRecursively は immutable flag のみ）。preservation の証拠にはなるが、指示どおり pointer identity 単独を binding identity とは採用しない
3. **E 側 identity は 3 値組 {sequenceId, epoch, mappedGeneration} に収束**: OwnerChannelKey がこの組を key とし、take-once invariant により単一 execution への一意性を構造的に示唆する（正式証明は O-3C）

---

## B1〜B4 予備観察（正式判定は O-3B/C/D）

| 条件 | 予備観察 | 正式判定 |
| ---- | -------- | -------- |
| B1 Existence（O admitted ⇒ ∃E） | Accepted 後は必ず一度 commitRuntimePublication/enqueue を呼ぶ（手続き的 ∃E candidate） | O-3C で判定 |
| B2 Uniqueness（∃!E） | OwnerChannelKey take-once + FIFO intent queue が示唆。但し fire-and-forget モードと receipt timeout race が検証対象 | O-3C で判定 |
| B3 Preservation（O target = E target） | 同一 object continuity が実在。fallback chain の意味論確認が必要 | O-3C で判定 |
| B4 Terminality（E 後の disposition 対応） | terminal 点: Success(receipt) / RejectedNotFinalized(build fail) / RejectedShutdown・RejectedPublishFailure(executor fail)。receipt timeout race と ownership disposition（CallerDestroy→destroyRolledBackDSP）の整理が必要 | O-3D で判定 |

## Accepted ≠ Execution ≠ Published の分離確認（指示 §3）

```text
Accepted ≠ Execution : evaluate 通過は build/validate/commit 前である（cpp:48-55 vs 76-215）
Accepted ≠ Published : 物理 swap は CoordinatorLoop 上の executePublish で発生（非同期区間あり）
                       ※ main path は receipt 待ちにより事実上同期だが、fire-and-forget 経路と
                         receipt timeout race により「Accepted ⇒ Published」は状態機械として未証明
```

post-Accepted failure 分岐と O 所有権の行き先（census 結果）:

| 分岐 | 発生点 | O の行き先 |
| ---- | ------ | ---------- |
| payload 生成失敗 | build fail → RejectedNotFinalized | destroyRolledBackDSP（rollback済み Handle 回収） |
| Executor 拒否 | frozen null / releaseState null → PublishFailed | CallerDestroy disposition |
| publish 失敗 | !isCommitted(result.stage) | destroyRolledBackDSP + RejectedPublishFailure / shutdown 中なら RejectedShutdown |
| shutdown | admission-publish 間 race（15-P-6） | RejectedShutdown 分類 |
| deferred | DeferredFadingActive → control-lane（NOT Accepted — 別 domain） | O 対象外（O-1/O-2 確立済み） |
| stale discard | deferred lane 内 TTL/gen/seq | O 対象外（control-lane） |

## Deferred との混同禁止の再確認（指示 §4）

```text
C1 Accepted ── main execution domain ──→ E
Deferred    ── control-lane pre-admission state（consume 後 C1 再通過）
```

DeferredPublishSlot を O の execution binding storage として扱わない — O-2 で確定した意味領域分離を維持。

## trySubmit() return semantics の精査（指示 §6）

header 契約: 「Accepted: 全処理完了 / Deferred: 保留 / Rejected*: 却下」

- main path: Accepted return は receipt 受信後 = 物理 swap 後が原則（同期 transaction）
- 但し「同期的な一つの execution transaction」内部に別の ownership/state transition が存在する:
  producer push → intent enqueue → CoordinatorLoop take → authority.publish → execution tail → receipt
- fire-and-forget 経路では「Accepted 相当の return」と物理 swap の順序が逆転し得る
- **結論: return semantics は順序契約であって O→E identity binding の証明ではない（指示の指摘どおり）**

## Payload 変換チェーンの target 保持確認（指示 §7）

```text
request → [builder] → RuntimeState(+identity stamp)
       → FrozenRuntimeWorld wrap
       → releaseState → rawState
       → aligned_unique_ptr<RuntimeState>(stateOwner)
       → ownerChannel push（同一 object）
       → executePublish take（同一 object）
       → authority.publish → store swap（同一 object）
```

- 変換は wrap/unwrap のみで target RuntimeState の置換は発生しない（object continuity census 済み）
- 但し fallback chain（registry.lookup / p.newWorld）が起動する場合、owner が取れないケースの target が別 source から来る — B3 の正式検証対象

## 判定表（O-3A 成果物）

| Obligation → Execution property  | 判定 |
| -------------------------------- | ---- |
| O identity                       | UNPROVEN（correlationId は E に届かず、O 側 authoritative ID 不在 — O-1 継承） |
| E identity                       | PASS（candidate 確定: {sequenceId, epoch, mappedGeneration} 3 値組 — OwnerChannelKey/intent/registry/receipt で一貫） |
| O→E binding key                  | ABSENT（既存 field の組で O と E を authoritative に結合するものなし。object continuity は証拠だが契約ではない） |
| Accepted→E existence             | UNPROVEN（手続き的 1:1 は確認 — B1 正式判定は O-3C） |
| O→E uniqueness                   | UNPROVEN（take-once invariant が示唆 — B2 正式判定は O-3C） |
| target preservation              | UNPROVEN（object continuity 実在、fallback chain 意味論の検証が必要 — B3 は O-3C） |
| failure disposition preservation | UNPROVEN（terminal 点 6 分岐を census 済み — B4 は O-3D） |
| shutdown interaction             | UNPROVEN（15-P-6 race 分岐を確認 — O-3D） |
| deferred interaction             | SEPARATE（control-lane との意味領域分離を維持 — O-1/O-2 確定の再確認） |
| duplicate execution possibility  | UNPROVEN（receipt timeout race + fire-and-forget モードが検証対象 — O-3C/D） |
| new token required               | YES candidate（既存 field のみでの binding 成立は ABSENT 判定により困難 — 但し新 token 設計は禁止中、O-3B/C の検証結果待ち） |

## Current Fixed State & Next

```text
D101-2.5N     CONDITIONAL/CANDIDATE（N-C2-B 唯一候補）
D101-2.5O-1   OPEN（definition/birth 条件付き固定 / identity UNRESOLVED）
D101-2.5O-2   OPEN（D_obligation UNASSIGNED / Expired dead enum 確定）
D101-2.5O-3A  CENSUS COMPLETE — data-flow 全追跡 / binding candidate 抽出完了 /
              E identity candidate 確定 / O→E binding key = ABSENT 暫定判定

A_max=1 / T_w=2 / max(E_w)=1 observed only / M/R/R_cap/B_max^true/T2 UNDETERMINED 維持
```

**次フェーズ: D101-2.5O-3B/C** — O/E identity・binding candidate の深掘り監査（uniqueness/preservation proof）→ **O-3D**（failure/shutdown/rejected path の ownership conservation）→ **O-3E**（Execution Binding Contract Closure 判定表確定）

---

## Tool Coverage

| 系統 | ツール | 実行内容 | 結果 |
| ---- | ------ | -------- | ---- |
| WSL | rg 15.1.0 | commitRuntimePublication 全 caller census（production/tests/facade）/ OwnerChannelKey take/push / registry lookup-unregister | 経路網羅 |
| WSL | cat / sed 4.9 | PublicationExecutor.{h,cpp} 全文 / RuntimePublishExecutor.h 1-130 行 / ProcessIntent.cpp 60-160 行 | 行レベル追跡の本体照合 |
| Sandbox | ctx_batch_execute | 4 コマンド並列（executor full / commit callers / publish executor / process intent）+ queries 抽出 | 1 往復で全データ収集 |
| 継承 | Phase N/O-1/O-2 の結果 | reserveRuntimePublicationIdentity 採番点 / correlationId 意味論 / slot 分離 | 再利用 |
| 文献 | crossbeam-epoch / rigtorp 等 9 系統 | 前ステップまでに 200 OK 済の知識を再利用（O-3A は社内 data-flow census で完結） | 追加調査不要と判断 |
