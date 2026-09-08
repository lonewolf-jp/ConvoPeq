# D176 — RuntimeWorld / Retire / Shutdown / Recovery 総合 Invariant Audit（evidence）

- 日付: 2026-09-08
- Type: **read-only structural audit** — production source 0 / test source 0 / CMake 0 / build・CTest 0
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-08 20:32:26`（NEWER_SRC_COUNT=0 FRESH）・実装 commit 54ba7b40
- 方針: D175 報告書を根拠にせず、**現行ソースの call-chain / writer / state transition を一次証拠**として構造を再証明する
- 原則基準: Practical Stable ISR Bridge Runtime — RT は観測・実行に限定 / Retire → Epoch → Reclaim → Delete の NonRT 隔離 / Publish・Retire authority 単一化 / Publish 後 immutable

---

## D176-0 — Source Authority【PASS】

- `ConvoPeq.md` Generated **2026-09-08 20:32:26** / NEWER_SRC_COUNT=0 FRESH（D175-3 再生成版）
- 実装 commit 54ba7b40（D172-3）以降の source 差分 = AudioEngine.h コメント 1 箇所のみ（D175-2）

## D176-1 — RuntimeWorld / Publication Authority【PASS — 単一 gateway 実測】

### Publish writer の全箇所（一次証拠）

**物理 store swap（`publishAndSwap`）の実行箇所は 2 つのみ、いずれも `RuntimeWorldAuthority` 内部:**

| # | 箇所 | 役割 |
|---|---|---|
| W1 | `RuntimeWorldAuthority::publish()`（RuntimeWorldAuthority.h:312 `writeAccess_.publishAndSwap(next)`） | 唯一の物理 publish gateway（INV-X4-3・commit-before-swap ordering = Test 7） |
| W2 | `clearPublishedRuntimeSnapshotsNonRt()`（同 :326 `publishAndSwap(nullptr)`） | shutdown clear（別 semantic — 二十次レビュー §15・`shutdownClearRequested_` gate） |

- `commitRuntimePublication()`（AudioEngine.h:4750）は async facade であり、実 swap は行わない。呼び出し側 5 箇所（PrepareToPlay:173/300・ReleaseResources:216・Transition:25・Timer:994）+ PublicationExecutor.cpp:53 — **すべて enqueue → CoordinatorLoop → PublishExecutor::executePublish に収束**
- `PublishExecutor::executePublish`（RuntimePublishExecutor.h:20-）が唯一の Execution gateway（HANDLER-1）: **OwnerChannel take（単一移動）→ `sealRecursively()`（publish 前 immutable 化 PR-5）→ `authority.publish()`（bake → swap）→ `bridge.didPublish → willRetire → retirePublishedRuntimeWorldNonRt(oldWorld)`** の順序を実測
- `publish()` 内部: validate（owner null / seqId==0 → 消費して nullptr）→ `coordinator_.commit()`（bake・monotonicity check・Faulted なら swap しない）→ swap → oldWorld を caller へ返す（bake 完了前に観測されない）
- identity 対応: `{world, publication}` は CW-8 型（`observePublishedObservation` — 単一 acquire load）で取得・`newWorld->publication.sequenceId/epoch/mappedRuntimeGeneration` が metadata として bake される（AudioEngine.h:4690-4694）
- **旧 authority `currentWorld_` 復活: 0 件**（コメント言及のみ・src 実測）。commit は `prevWorld = runtimeStore_.observe()` を明示依存（CW-3b）

**判定**: `Build → Validate(seal) → Bake(commit) → Swap(publishAndSwap) → Observe` 以外の World mutation 経路は存在しない。

## D176-2 — Retire Authority【PASS — 二重 retire の構造的排除】

### old World の retire 経路（一次証拠）

- `publishAndSwap` は atomic exchange で oldWorld を**正確に 1 回**返す（RuntimeStore.h:40-52）→ `executePublish` が committed==true のときのみ `retirePublishedRuntimeWorldNonRt(oldWorld, false)` を呼ぶ → oldWorld が retire 対象になるのは **swap 成功時に限り 1 回**。
- `retirePublishedRuntimeWorldNonRt`（AudioEngine.h:3623-3639）: PRECONDITION `W ∈ PublishedDomain` → `enqueueDeferredDeleteNonRt(world, deleter, DeletionEntryType::World)`。
- `DeletionEntryType::World` を付ける箇所は **src 全体で 1 箇所のみ**（AudioEngine.h:3635）— World 型 terminal deletion の単一 entry point。
- 未公開 world の rollback は別関数 `retireRejectedRuntimeWorldNonRt`（`W ∉ PublishedDomain`・`DeletionEntryType::Generic` — ReleaseResources.cpp:659 comment で PRECONDITION 分離を明示）。
- shutdown clear の oldWorld も同一関数経由（ReleaseResources.cpp:535・CtorDtor.cpp:268 — `clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true)`）。
- DSPCore の retire は `DSPLifetimeManager::retire` → `retireDSPHandleForRuntime`（台帳解除 primitive）→ `enqueueWithRetry`。台帳解除と物理破壊は分離（AudioEngine.h:4332-4348 契約）。

### 二重 retire 防御の構造

1. oldWorld の取得が `publishAndSwap` の戻り値**のみ**（他に World ポインタを取得する経路がない — observe 系は `const RuntimeState*` で非所有）。
2. `enqueueDeferredDeleteNonRtWithResult` の不変式「**caller → authority への ownership transfer が成立してから return**」（AudioEngine.h:4291・ISRRetireRouter.cpp:25「enqueueWithRetry() never returns with ptr unowned」— 失敗時は quarantine 移送）。
3. `isShutdownInProgress()` 時は `shutdownReclaim`（TerminalReclaimAuthority）へ移送 — ptr を捨てない（P-4）。

**判定**: 同一 RuntimeWorld への Retire authority 二重発生経路は実測で存在しない（swap 戻り値が唯一の retire 権利者）。

## D176-3 — Epoch / Reclaim / Delete Chain【PASS — ownership 逆流なし】

実測 chain:

```text
retire(world)                enqueueDeferredDeleteNonRt → markRetireEpoch() → enqueueWithRetry/Quarantine
   ↓ retireEpoch             entry.epoch = enqueue 時の currentEpoch
   ↓ reader protection       EpochDomain: enterReader（enter 時 epoch を pin）→ exitReader（kInactiveEpoch）
   ↓ quiescence             getMinReaderEpoch = active reader の pin 最小値（active 0 → currentEpoch）
   ↓ reclaim eligibility    isOlder(entry.epoch, minReaderEpoch) == true のみ（Router/Quarantine/Terminal 共通条件）
   ↓ delete                 drain: lock 下で抽出 → lock 外で deleter 実行（reentrancy-safe）
                              → world deleter: unseal → ~RuntimePublishWorld → aligned_free
                              → DSP deleter: ~DSPCore → aligned_free（destroyDSPCoreNode）
```

- **ownership 逆流（reclaim 後の再 enqueue・delete 後の参照再取得）の経路**: 実測で不存在。queue から外れるのは deleter 実行時のみで、deleter 実行後のエントリは `Entry{}` に clear・resident counter decrement。reclaim 遅延時は epoch-unsafe のまま保持（破壊しない）・re-ownership も再手渡しも発生しない。
- `ReclaimProof` / `ReclaimPermit` / `ShutdownRuntimeIdentity`（ISRLifetimeProof.h）: shutdown 経路の quiescence proof → Permit → `reclaimShutdownQuiescent`（AudioEngine.h:4410-4414）で epoch gate を形式的に通過させる（D172-2 P2 で再確認済み）。
- `worldReclaimCount`（T1/D86）= terminal deleter 実行数（primary + quarantine 合算）— release observation の一次情報源。

## D176-4 — RT Boundary【PASS — helper 追跡まで実施】

RT entry: `AudioEngineProcessor::audioDeviceIOCallback` → `AudioEngine::getNextAudioBlock`（AudioBlock.cpp:27）。

- **RT path ファイル群（AudioBlock / BlockDouble / DSPCoreFloat / DSPCoreDouble / DSPCoreIO）の forbidden-op scan**: `delete / aligned_free / malloc / make_unique / std::mutex / lock_guard / writeToLog / retirePublished / destroyDSPCore / destroyRolledBack` = **実 0 件**（唯一の writeToLog 言及 2 件は「RT-safe: LockFreeRingBuffer (DiagEvent)」の comment）
- **helper 経由の NonRT-only operation 到達**: RT の DSP 解決は `readAudioRuntimeView()`（単一 callback authority view）→ `makeRuntimeReadHandle`（**epoch pin** 付与）→ `resolveActiveRuntimeDSPFromRuntimeWorldOnly`（const world 読み取り）→ `dsp->process()` 等の member 呼び出しのみ。解決 helper 自体は publish/retire/delete を呼ばない純 resolver（D172-2 P1 実測と同一）。
- RT から `worldAuthority_.publish` / retire / reclaim に到達する call chain: 実測 0 件。publish は NonRT producer（Builder/Timer/Transition/ReleaseResources）→ intentQueue → CoordinatorLoop のみ。
- MMCSS 登録は RT callback 内だが logging は diagnostic guard + thread_local（AudioBlock.cpp:44-53 実測）・寿命管理とは無関係。

## D176-5 — Shutdown Full Pipeline【PASS — 7 phase 実測】

source 実測（ShutdownPhase enum・dtor シーケンス — D172-3 runtime log とも一致）:

```text
StopAcceptingWork  requestShutdown・closeAdmission（Early Close Convergence）
    ↓
StopAudio          stopTimer（Timer/MessageThread 停止）・Audio callback は isShutdownInProgress で clear-active
    ↓
StopWorkers        RetryScheduler shutdown（stop→clear/discard→join）→ shutdownCoordinatorLoop → stopRebuildThread
    ↓                 （producer 停止後に worker join — 新規 intent 生成停止）
DrainRetire        drainDeferredRetireQueues(true)・tryReclaim
    ↓
ForceEpochAdvance  （期間前: setShutdownPhase(DRAIN_RETIRE) までに publish 不在を確立）
    ↓
（clear）          clearPublishedRuntimeSnapshotsNonRt → shutdownReclaim（TerminalReclaimAuthority）
    ↓                 15-P-5: quiescence 確立時のみ drainAll（stuck reader 時は epoch-gated drain に委譲）
（final drain）    m_epochDomain.drainAll()（D のみ・live reader なし）+ drainAllQuarantineStore()（Q+E+T・epoch 非依存・AudioThread 停止後のみ契約）
    ↓
markShutdownComplete
    ↓
（Faulted 状態チェック → shutdown 異常を [FAULT] ログ）
Destroy
```

- **shutdown 中の新規 ownership 生成**: producer 停止後に drain する順序で排除。`enqueueDeferredDeleteNonRtWithResult` は shutdown 中でも ptr を捨てず ShutdownReclaimAuthority へ移送（P-4）— ownership 取り残し構造なし。
- **timeout fallback**: `waitForPublishReceipt` timeout は「所有権移譲済み（Transferred）」として扱い rollback しない（X2 §6.2 — double ownership / double publish の構造的排除）。stuck reader 時は強制 drain を skip し epoch-gated drain に委譲（15-P-5）。
- `isFullyDrained()`: Layer 1 = `pendingReclaimHandles_.empty()` + retire queue pending = 0（INV-X3-5）— 実測値を直接判定する設計（dash2 §1.4 B0-7）。
- 事後検証: D169-2-6/7（restart 50 cycles ×3 config・full regression 40/40 ×3）・D162-2-I1（26/26 exit 0x0・zone clean）の runtime evidence が本実測順序の動的裏付け。

## D176-6 — Recovery / Retire の交差点【PASS — invariant 非侵蝕】

- Recovery obligation は **rebuild intent のみ**を生成（QuarantineIntentHandler → `submitRecoveryIntent` → Builder Work Queue）— RuntimeWorld / DSPCore の lifetime を直接所有しない。
- obligation terminalization は `RecoveryOutcome → ObligationState`（Coordinator.cpp:1018-1061・単一 Completion Authority・D105-R5-9/R18 idempotent）で完結 — retire authority（Router）とは別 authority。`DeletionEntryType` に obligation を触る経路なし。
- transient failure / redrive: `settlePendingRecoveryAdmission`（durable slot 戻し）+ `markTransientFailure`（delivery=None + redrive wake）— いずれも「次サイクルの再 build」を指示するのみで retire/delete を呼ばない。
- shutdown discard（RecoveryOutcome::ShutdownDiscarded → `recoveryObligationShutdownDiscardCount_`）と retire/delete は二重化されていない（obligation の台帳 closure のみ・DSP/World の物理破壊は既存 retire pipeline 専用）。
- D105-R18: obligation counter と durable-slot sub-state は**別フィールドの独立線形化**（Dual-LP）— D174-1 B で実測済み。
- same obligation / same world の double ownership: Recovery 発行は quarantine 成功（`stateChanged==true`）時 1 回のみ（六次レビュー修正・HANDLER-1）。

## D176-7 — Authority Matrix

| Resource / Decision | 唯一の Authority | Reader | Writer | Delete/Terminal |
|---|---|---|---|---|
| RuntimeWorld（物理） | `RuntimeWorldAuthority::publish`（INV-X4-2/3） | `runtimeStore_.observe()`（全 reader・const） | executePublish のみ（seal→bake→swap） | `retirePublishedRuntimeWorldNonRt`（DeletionEntryType::World 唯一点・Router drain） |
| Publication identity | `coordinator_.commit`（bake） | CW-8 `observePublishedObservation` | bake within publish() | —（identity は world と運命共通） |
| Crossfade | `CrossfadeAuthority`（Decision Snapshot enqueue 時固定） | RT process（構成読み取り） | commit 前の Decision のみ | — |
| Retire（DSP） | `DSPLifetimeManager::retire` | telemetry counters | retireDSPHandleForRuntime（台帳解除 primitive） | Router enqueue → deleter |
| Retire（World） | `retirePublishedRuntimeWorldNonRt`（swap 戻り値が唯一の権利源） | — | 同左 | 同左 |
| Epoch | `EpochDomain`（m_epochDomain） | enterReader/exitReader・getMinReaderEpoch | publishEpoch（publish tail・markRetireEpoch） | —（protect 専用） |
| Reclaim | `ISRRetireRouter::tryReclaim / drain`（isOlder 条件単一） | telemetry | CoordinatorLoop / RecoveryAction / shutdown | → Deferred Delete |
| Deferred Delete | Router（Q/E/T + TerminalReclaimAuthority + OverflowRing） | pendingRetireCount 等 telemetry | enqueueWithRetry / shutdownReclaim / quarantine | deleter 実行（lock 外） |
| Recovery obligation | `ISRRuntimePublicationCoordinator`（Completion Authority・D105-R5-9） | orchestrator telemetry | intent 発行（quarantine 1 回） | ObligationState terminal（World/DSP 破壊は別 authority） |
| Shutdown admission | `ISRShutdown`（AdmissionState PackedState・単一 32-bit CAS） | isShutdownInProgress 等 | closeAdmission / reopen（Closing） | ShutdownDiscarded（obligation 台帳のみ） |
| Retry scheduling | `RetryScheduler`（time/ordering のみ） | pendingCount/rejectCount | Site 3 :1311 のみ | reject（capacity/shutdown・捨てない対象なし） |

`?` 残留項目: **なし**。

## D176-8 — Ownership Conservation Proof

World 1 個の遷移（一次証拠ベース）:

```text
Build ownership     Builder が aligned_unique_ptr<RuntimeState> 生成（FrozenRuntimeWorld）
    ↓ enqueue         commitRuntimePublication — 所有権 Transferred（P-4 不変式・timeout でも返却されない）
    ↓ publication     executePublish: OwnerChannel take（単一移動）→ seal → publish()（swap）→ oldWorld 1 回
    ↓ active          RuntimeStore::current（immutable・reader は const のみ）
    ↓ retire          swap 戻り値 oldWorld → retirePublishedRuntimeWorldNonRt（1 回）
    ↓ epoch-protected Router Q または Quarantine/Terminal store（epoch unsafe の間保持）
    ↓ reclaim         isOlder 成立 → drain 抽出（Entry clear・resident decrement）
    ↓ delete          unseal → ~RuntimePublishWorld → aligned_free（T1 worldReclaimCount +1）
```

### 禁止状態のチェック結果

| 禁止状態 | 実測判定 |
|---|---|
| Owner=0 なのに queue に存在 | **不可能** — enqueue は所有権移転を前提（never returns with ptr unowned・失敗は quarantine 移送） |
| Owner=1 なのに 2 container に存在 | **不可能** — swap の atomic exchange で oldWorld が正確に 1 回手を離れ・移転先は単一（Router）|
| Retired + Active | **不可能** — retired world は swap で current から外れた個体のみ（current は新 world） |
| Retired + Deleted | **不可能** — retire は enqueue のみ・削除は Router drain の単一地点・二重 dequeue 構造なし |
| Deleted + reachable | **不可能** — reader は observe（current のみ）+ epoch pin で保護・deleted 後の参照経路なし（D172-1 の slot は D172-3 で dereference 廃止済み） |
| RT-held + reclaimed | **不可能** — RT は epoch pin 中（enter→exit）であり minReaderEpoch が回収を遅延 |
| same World → two retire admissions | **不可能** — swap 戻り値が唯一の retire 権利源・`DeletionEntryType::World` 付与箇所は src で 1 箇所のみ |

## D176-9 — Findings / Severity

**blocking finding: 0 件。** 観測した非 blocking 記録事項:

| # | 記録 | 深刻度 | 処置 |
|---|---|---|---|
| N-1 | `makeRuntimeReadHandle` の world observe → enter 順序（μs 窓・Case 3）は理論上残存（D172-2 P2-4 既知境界） | INFO | 既知境界のまま維持・D172 closure で scope 外確定済み・着手理由なし |
| N-2 | `Faulted 状態 after markShutdownComplete` は [FAULT] ログのみで recovery action なし（D172-3 run2 log で観測された実績あり） | INFO | 観測系は health monitor に存在・shutdown 自体は完了する。telemetry 監視対象として維持 |
| N-3 | `reclaim()` 名の API が複数層（EpochDomain reclaimRetired / DeletionQueue / SnapshotRetireManager / Coordinator）に存在するが、いずれも同条件（isOlder）の別ストアへの適用であり authority 二重化ではない | INFO | 命名上の混同注意のみ・実害なし |

## Final Decision

> ## **D176 PASS — 総合 invariant は閉じている**
>
> - 全 lifetime path が **Retire → Epoch → Reclaim → Delete** に閉じている（単一 entry point・単一 reclaim 条件・ownership 逆流経路 0 件）
> - **RT → NonRT 境界 violation なし**（RT path forbidden-op 0 件・helper 追跡でも到達経路なし）
> - **Publish / Retire authority 単一**（publish gateway = RuntimeWorldAuthority::publish のみ・World 型 deletion entry point 1 箇所のみ・currentWorld_ 復活 0）
> - **Shutdown が ownership を取り残さない**（P-4 移送不変式・quiescence 条件付き drainAll・stuck reader fallback・timeout でも Transferred 維持）
> - **Recovery は lifetime authority を奪わない**（obligation = rebuild intent のみ・Completion Authority 分離・D105-R18 Dual-LP）
>
> **次工程への推奨**: D176 PASS により個別 defect track の起票義務はゼロ。以後は通常開発サイクル（D159 ハンドオフどおり）に復帰し、D174 登録の trigger 条件（RuntimeBuilder.h:118-124）成立時のみ BuildError Phase-II を再評価する。

## 実測コマンド系譜（主要分）

```bash
rg -n "publishAndSwap" src/                                   # 物理 swap 2 箇所（publish / shutdown clear）
rg -n "commitRuntimePublication\(" src/audioengine/           # facade 呼び出し 5+1 箇所 → executePublish 収束
sed -n '260,330p' src/audioengine/RuntimeWorldAuthority.h     # publish()（validate→bake→swap→oldWorld）
cat src/audioengine/RuntimePublishExecutor.h                 # executePublish（take→seal→publish→retire old→advanceEpoch）
sed -n '3600,3660p' src/audioengine/AudioEngine.h            # retirePublished/Rejected（World 型 唯一点 :3635）
sed -n '4290,4330p' src/audioengine/AudioEngine.h            # P-4 ownership transfer 不変式
sed -n '52,80p' src/audioengine/ISRRetireRouter.cpp          # isOlder reclaim 条件（共通）
grep -cE "delete|aligned_free|malloc|make_unique|mutex|lock_guard|writeToLog" src/audioengine/AudioEngine.Processing.{AudioBlock,BlockDouble,DSPCoreFloat,DSPCoreDouble,DSPCoreIO}.cpp   # RT scan = 実 0
rg -n "enum class ShutdownPhase" + dtor trace                # 7 phase 実測
rg -n "ShutdownDiscarded|postRecoveryFailureSignal|D105-R18" # obligation/retire 分離
sed -n '40,52p' src/core/RuntimeStore.h                      # publishAndSwap（atomic exchange・oldWorld 1 回）
```
