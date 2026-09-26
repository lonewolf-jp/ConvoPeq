# P1-5-IR-P2 — Step 5-AK / P3-5-R29: Recovery-origin Vehicle Feasibility / D105 Dynamic Verification Design Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R29）
- **種別**: read-only design audit。**production 0・test 0・CMake 0・counter 0・getter 0・API semantic 0**。
  build 0・run 0。Recovery vehicle の実装自体は行わない（設計 gate）。
- **判定**: **R29-B（Path B）**。
  既存 Full AudioEngine vehicle で Recovery-origin publish を発生させる決定論的手段は存在しない。
  最小 test-only 入口（D1）は **production 変更 0** で定義可能（public API のみ）。
  ただし Quarantine 段は test から再現不能（state 再現は可能）→ R30 implementation gate へ。

---

## 1. State Freeze（PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（CMakeCache BOOL=OFF）
保持: R16 take／R19 build-result／R22 commit／R27 origin plumbing／R10 accessor／T6/T9／R23 vehicle／R28-B findings
ConvoPeq.md = 2026-09-23 22:41:29（5,525,709 B・generator output_sourcecode_markdown.py・R27 state を内包）
binary = R27（SHA-256 425158012318dde3…）・R29 では build/run しない
working tree = R27 state（本 Step の差分は本ドキュメントのみ）
```

- R29 §1 が参照を求める `ConvoPeq(3).md` / `ConvoPeq(2).md` は本環境に存在しない（探索 root 全域）。
  最新 `ConvoPeq.md`（R27 state 内包）を source authority とした。

## 2. Latest ConvoPeq Source Reconciliation（PASS）

R29 §1 の必須実装を live source で確認（すべて R27 state）：

| 実装 | 位置 | 備考 |
| --- | --- | --- |
| `AudioEngine::submitRecoveryIntent` | AudioEngine.h:4686 | **public**（QuarantineIntentHandler が非 friend で呼ぶ） |
| `AudioEngine::getCurrentBuildSnapshotForRecovery` | AudioEngine.h:4677 | **public**（ProcessIntent.cpp:145 が非 friend で呼ぶ） |
| `RuntimeIntentCoordinator::submitRecoveryRequest` | Coordinator.cpp:873 | **private**（AudioEngine の member access 経由） |
| `RuntimeIntentCoordinator::resolveRecoveryObligation` | Coordinator.cpp:1031 | **private** |
| `QuarantineIntentHandler::handle` | ProcessIntent.cpp:111 | `executeQuarantine` → `submitRecoveryIntent` |
| `RuntimePublicationOrchestrator::onPublishCommitted` | Orchestrator.cpp:347 | 355 notify → 358 resolve |
| `PublicationExecutor::publish / publishImpl` | PublicationExecutor.cpp:8 / :17 | `waitForReceipt=true` のみ |
| `AudioEngine::trySubmitImpl` | Orchestrator.cpp:41 | 283 `executor_.publish` → 324 Route A resolve |
| `AudioEngine::resetReceipt` | Timer.cpp:2004（decl AudioEngine.h:1232） | **function exists** ✅／**caller 0** |
| `RuntimeIntentCoordinator::submitQuarantine` | Coordinator.cpp:1474（decl h:241） | **public（coordinator 上）**／AudioEngine passthrough なし |
| `coordinatorTakeCount_` | h:974／writer ProcessIntent.cpp:61-62 | Main-only（R27） |
| `recoveryObligationId` | PublishPayload h:732／搬送 AudioEngine.h:4888 | R27 plumbing |

### R28 記述の精密化（R29 §1 の要求）

- **`resetReceipt()`：function exists ≠ vehicle can invoke it**
  - 関数は存在し **public**（AudioEngine.h:1232）。
  - ただし **caller 0**（def＋decl のみ）。かつ動作には private `pendingReceipt_` が populated である
    必要がある（empty なら即 return）。`pendingReceipt_` の writer は `storeReceipt`（AudioEngine.h:1207・
    DSPTransition.h:136 から呼ばれる）のみで、これは crossfade/fading publish 時に発生する。
    → **関数は呼べるが、意味のある動作を test から決定論的に作れない**。
- **`liveLogicalRecoveryObligationCount()`：component observability ≠ Full AudioEngine observability**
  - 本 getter は RuntimeIntentCoordinator 上に存在し、component tests
    （ISRSemanticValidationTests 等）が直接駆動する。
  - ただし **AudioEngine からの passthrough が存在しない**（`AudioEngine.h` に caller/accessor 0）。
    Full AudioEngine Harness からは直接不可視。R28 の「harness から不可視」は
    **Full AudioEngine Harness 限定の caller 有無**として成立する。
  - ★ ただし本 Step で重要な副次発見：`RuntimeIntentCoordinator::isFullyDrained()` は
    `liveLogicalRecoveryObligationCount() == 0` を predicate に含み（Coordinator.cpp:561）、
    `AudioEngine::isFullyDrained()`（Threading.cpp:153→:212）と `waitForDrain()`（AudioEngine.h:1601・public）
    経由で **Full AudioEngine から間接的に観測可能**（§11）。

## 3. Recovery Path Actual Topology（一本に固定）

```text
[production trigger] quarantine 検出
  Timer.cpp:1975   retirePublishedDSP CAS mismatch → submitQuarantine(PublishViolation)
  Timer.cpp:2013   resetReceipt → submitQuarantine(ReceiptReset)          ← caller 0（dead）
  Commit.cpp:623/643 quarantineSlot(RetireDeferralTimeout)                ← private・要実DSP slot
        ↓
RuntimeIntentCoordinator::submitQuarantine（h:241 public）
        ↓ intentQueue_.push(QuarantineIntent)
CoordinatorLoop: processIntent（ProcessIntent.cpp:47）
        ↓ QuarantineIntentHandler::handle（:111）
        ↓ QuarantineService::executeQuarantine（Coordinator.cpp:784）
           状態変更条件: !handle.isNull() && handle.slot > 0 → stateChanged=true
        ↓ if (stateChanged && !handle.isNull())（:137-141）
AudioEngine::submitRecoveryIntent(handle, buildSource)（AudioEngine.h:4686 public）
  │ 前提: hasAuthoritativePublishedRuntime()（:4695）— false なら absorb（obligation なし）
  │ buildSource = getCurrentBuildSnapshotForRecovery()（:139・sealed snapshot）
        ↓
RuntimeIntentCoordinator::submitRecoveryRequest（private・cpp:873）
  │ shutdown gate（:878）／capacity L<32（:969）／coalesce（:954）
  │ → recoveryAdmissions_.tryInsert → oblId != 0（:977）／obligationId = oblId（:990）
  │ → pendingIntentCount_++（:995）→ recoveryIntentQueue_.push（:996）capacity 256
        ↓ recoveryPending=true + rebuildCV.notify_all（AudioEngine.h:4723/4725）
Builder Loop（RebuildDispatch.cpp:1014 popRecoveryRequest）
  │ gate: !qHandle.isNull() && recovery->buildSource.sealed（:1020）
  │ build → warmup → enqueuePublicationIntentForRuntimeCommit(
  │        dspToCommit, recoveryGeneration, recoverySnapshot, {}, {}, {},
  │        recovery->obligationId)          ← :1085（transport）／:1165（durable）
        ↓
enqueuePublicationIntentForRuntimeCommit（Commit.cpp:800）
  → submitPublishRequest → trySubmitImpl → executor_.publish(..., req.recoveryObligationId)
  → commitRuntimePublication(...) → intent.payload.publish.recoveryObligationId != 0（AudioEngine.h:4888）
  → IntentQueue → Coordinator pop（coord は加算**しない**・ProcessIntent.cpp:61）
  → executePublish → publish → seq bump → onPublishCommitted → resolveRecoveryObligation
```

- Recovery-origin publish は **`submitRecoveryIntent` が唯一の起点**（生存 caller は QuarantineIntentHandler のみ）。
- obligation 生成条件：`hasAuthoritativePublishedRuntime()==true` かつ L<32 かつ非 ShuttingDown。
- build 成立条件：handle 非 null ＋ `buildSource.sealed==true`。

## 4. Candidate A — resetReceipt

```text
resetReceipt()（AudioEngine.h:1232 public・Timer.cpp:2004）
  if (!pendingReceipt_.has_value()) return;    ← empty なら no-op
  if (!pendingReceipt_->handle.isNull())
      submitQuarantine(handle, ReceiptReset, ...)   → QuarantineIntent → recovery
```

| 判定項目 | 結果 |
| --- | --- |
| Full AudioEngine | ✅（engine 直接） |
| 実 DSP slot | ✅（pendingReceipt_ の handle・要 populated） |
| quarantine stateChanged | ✅（handle.slot>0 なら true） |
| Recovery obligation 生成 | ✅（submitRecoveryIntent 経由・条件付き） |
| `recoveryObligationId != 0` | ✅ |
| Builder まで到達 | ✅ |
| Publish まで到達 | ✅ |
| deterministic | ❌ `pendingReceipt_` が crossfade publish 依存（test から設定不可・private） |
| repeatable | ❌ 同上（タイミング依存） |
| existing API のみ | ✅（resetReceipt は public） |
| new production code | 不要 |
| new test-only entry 必要 | 不要（呼ぶだけ）だが **pendingReceipt_ 前提を満たせない** |
| D105 dynamic verification 可能 | ❌（episode を発生できなければ不可） |

- **不成立**：public 関数だが前提（private `pendingReceipt_`）を test から決定論的に作れない。
  `storeReceipt` は crossfade 完了時にのみ呼ばれる（DSPTransition.h:136）。quiescent harness では発生しない。

## 5. Candidate B — RetireDeferralTimeout

| 判定項目 | 結果 |
| --- | --- |
| Full AudioEngine | ✅（Commit.cpp 経由） |
| 実 DSP slot | 要（`pending.dspSlot != UINT32_MAX`） |
| quarantine stateChanged | ✅ |
| Recovery obligation 生成 | ✅ |
| `recoveryObligationId != 0` | ✅ |
| Builder まで到達 | ✅ |
| Publish まで到達 | ✅ |
| deterministic | ❌ 閾値（`hasExceededDeferralThresholds`）超過が条件・負荷依存 |
| repeatable | ❌ |
| existing API のみ | ✅（内部経路だが test から発火手段なし） |
| new production code | 不要 |
| new test-only entry 必要 | **必要**（発火には実 DSP slot pending retire ＋ 閾値超過の誘発 = 新機構） |
| D105 dynamic verification 可能 | ❌ |

- **不成立**：R28 の障害（`quarantineSlot` private／`RegistrationContext::none()` は slot なし）は不変。
  既存 test vehicle に「実 DSP slot を pending retire にして閾値を超えさせる」経路は **存在しない**
  （R29 では新規 DSP-slot injection mechanism を作らない）。
- `quarantineSlot`（AudioEngine.Threading.cpp:63・**public**）は存在するが、これは
  QuarantineIntent を発行せず **直接隔離**（truth store + projection + retire）するだけであり、
  recovery を起動しない（`submitQuarantine` 非経由）。

## 6. Candidate C — PublishViolation

```text
retirePublishedDSP（Timer.cpp:1937）
  currentHandle = dspHandleRuntime_.getFadingRuntimeDSPHandle()
  if (currentHandle == pendingReceipt_->handle) → Normal Retire
  else → submitQuarantine(PublishViolation)     ← mismatch 時のみ
```

| 判定項目 | 結果 |
| --- | --- |
| Full AudioEngine | ✅ |
| 実 DSP slot | ✅ |
| quarantine stateChanged | ✅（mismatch handle が有効なら） |
| Recovery obligation 生成 | ✅ |
| `recoveryObligationId != 0` | ✅ |
| Builder まで到達 | ✅ |
| Publish まで到達 | ✅ |
| deterministic | ❌ **CAS mismatch は race**（fading handle と receipt handle のずれ） |
| repeatable | ❌ |
| existing API のみ | ✅（内部経路） |
| new production code | 不要 |
| new test-only entry 必要 | **必要**（handle/generation を意図的にずらす既存 vehicle は**存在しない**） |
| D105 dynamic verification 可能 | ❌ |

- **不成立**：「CAS failure を起こせる可能性がある」は R29 §3 により **GO にしない**。
  決定論的 mismatch を作る既存 test API は無い（`pendingReceipt_` は private・
  `retirePublishedDSP` は public でない）。

## 7. Candidate D — test-only direct trigger（2 sub-variant）

### D1 — `AudioEngine::submitRecoveryIntent(handle, sealedSnapshot)` 直呼び

| 判定項目 | 結果 |
| --- | --- |
| Full AudioEngine | ✅ |
| 実 DSP slot | ✅（`registerDSPHandleForRuntime(activeDSP)` で取得・public・既存 test 前例 :316） |
| quarantine stateChanged | ❌ **Quarantine 段をスキップ**（state 変更なし） |
| Recovery obligation 生成 | ✅ |
| `recoveryObligationId != 0` | ✅ |
| Builder まで到達 | ✅ |
| Publish まで到達 | ✅ |
| deterministic | ✅（hasAuthoritativePublishedRuntime ＋ sealed ＋ generation 一致で決定的） |
| repeatable | ✅（episode 逐次実行：publish 後 obligation が terminal 化 → 次の submit が新 obligation。※同一 target の同時 submit は coalesce） |
| existing API のみ | ✅（submitRecoveryIntent / getCurrentBuildSnapshotForRecovery / registerDSPHandleForRuntime / observePublishedWorld / hasAuthoritativePublishedRuntime すべて **public**） |
| new production code | **0** |
| new test-only entry 必要 | ✅（test 1 点＝harness TU に function 追加。CMake は既存 TU 追加なら 0） |
| D105 dynamic verification 可能 | 部分（§11：exactly-once は drain で間接可・winner identity は不可） |

### D2 — `RuntimeIntentCoordinator::submitQuarantine(...)` 直呼び（Quarantine 段を含む）

| 判定項目 | 結果 |
| --- | --- |
| Full AudioEngine | ✅（engine 経由） |
| 実 DSP slot | ✅ |
| quarantine stateChanged | ✅（handle.slot>0） |
| Recovery obligation 生成 | ✅（QuarantineIntentHandler → submitRecoveryIntent） |
| `recoveryObligationId != 0` | ✅ |
| Builder まで到達 | ✅ |
| Publish まで到達 | ✅ |
| deterministic | ✅（intent 経路・非 race） |
| repeatable | ✅ |
| existing API のみ | △（`submitQuarantine` は coordinator の public だが **AudioEngine からの accessor が無い**） |
| new production code | **要 1 行**（friend 宣言 or passthrough） |
| new test-only entry 必要 | ✅（TestAccess クラス＋test 1 点） |
| D105 dynamic verification 可能 | 部分（D1 と同等） |

- **friend 前例**：`DeferredPublicationTestAccess`（AudioEngine.h:3866）・
  `LatencyDelayWiringTestAccess`（:3870）・`RetireGraceSemanticsTestAccess`（ISRRetireRouter.h:423）
  という TestAccess-friend パターンが既に存在する。D2 はこの前例に 1 行追加で追随可能。
- **D1 を上回る点**：Quarantine 段（R29 §5 の stage 1）を含む。
- **D1 を下回る点**：production 変更 1 行が必要（R29 §9 Path B は production 0 を優先）。

## 8. Full AudioEngine Reachability Matrix

| 項目 | A resetReceipt | B RetireTimeout | C PublishViolation | D1 submitRecoveryIntent | D2 submitQuarantine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full AudioEngine | ✅ | ✅ | ✅ | ✅ | ✅ |
| 実 DSP slot | 条件付 | ✅ | ✅ | ✅ | ✅ |
| quarantine stateChanged | ✅ | ✅ | ✅ | ❌（skip） | ✅ |
| Recovery obligation 生成 | 条件付 | ✅ | ✅ | ✅ | ✅ |
| `recoveryObligationId != 0` | 条件付 | ✅ | ✅ | ✅ | ✅ |
| Builder まで到達 | 条件付 | ✅ | ✅ | ✅ | ✅ |
| Publish まで到達 | 条件付 | ✅ | ✅ | ✅ | ✅ |
| deterministic | ❌ | ❌ | ❌ | ✅ | ✅ |
| repeatable | ❌ | ❌ | ❌ | ✅ | ✅ |
| existing API のみ | ✅ | ✅ | ✅ | ✅ | △（accessor 無） |
| new production code | 0 | 0 | 0 | **0** | **1 行（friend）** |
| new test-only entry 必要 | 前提不能 | ✅ | ✅ | ✅（最小1点） | ✅ |
| D105 dynamic verification | ❌ | ❌ | ❌ | 部分 | 部分 |

- 「順位付け」ではなく実装条件の比較。A/B/C は**前提を test から作れない**点で不成立。
- D1 と D2 は共に成立可能。差分は「production 0 vs 1 行」と「Quarantine 段の有無」。

## 9. Recovery-origin Publish Success Conditions（最低成立条件の固定）

最低 1 episode で以下を満たすこと（R29 §5）：

```text
[1] Quarantine（state）             … D1 は skip／D2 は含む
[2] Recovery admission              … submitRecoveryRequest が true（obligation 生成）
[3] obligationId = non-zero         … recoveryAdmissions_.tryInsert → oblId != 0
[4] Recovery request（transport）   … recoveryIntentQueue_.push 成功
[5] Builder                         … popRecoveryRequest → build 成功
[6] trySubmitImpl                   … admission Accepted
[7] PublicationExecutor             … publish 成功
[8] Publish Intent                  … intent.payload.publish.recoveryObligationId != 0
[9] Coordinator pop                 … PublishIntentHandler
[10] Recovery-origin pop            … coord は加算されない
[11] publish completion             … seq bump
[12] resolveRecoveryObligation      … terminal 化（liveCount_ → 0）
```

- 「Recovery component test PASS」ではなく **Full AudioEngine recovery-origin publish** であることが条件。
- [1] を除く [2]–[12] は公開 API と既存観測のみで構成可能（D1）。
- [3]/[8] は既存の内部状態だが、publish 到達＋`coord` 非加算＋drain で**間接的に**成立を確認できる。

## 10. coord / cmt / seq / drp Observability

```text
同一 window で採取する（R23 vehicle の emitQDelta が既に提供）:
  cmt  = getRebuildCommitEnqueueCount()   （main-site commit-enqueue 累積）
  coord= getCoordinatorTakeCount()        （Main-origin pop 累積・Recovery 非加算）
  seq  = getLastCommittedPublicationSequence()
  drp  = getRuntimeLifecycleDiagnostics().lastDroppedGeneration
  （補助: take/bld/req/que/dup/blo）
```

Recovery episode の構造的期待（R29 §6）：

```text
Recovery pop が発生したなら coord は増加しない（recoveryObligationId != 0 のため）
```

- ★ **`coordΔ == 0` から Recovery pop を証明してはならない**。`coord` は Main-origin 専用の
  aggregate counter であり、recovery を計数しないのは仕様（R26-A）。absence-of-increment は
  「Recovery が起きた」ことの証明ではない（主証拠は seq 到達 ＋ 当該 episode の既知入力）。
- 本設計では、Recovery episode を**同一 vehicle・同一 window**で観測することで
  「seq が進み coord が進まない」という**必要条件**を検証する。

## 11. D105 Dynamic Winner Feasibility

### Route の実時間関係（R29 §8 の要求：正確な記述）

```text
Producer（RebuildThread）:
  trySubmitImpl
   → executor_.publish(...)                                 [PublicationExecutor.cpp:8]
     → publishImpl(waitForReceipt=true)                     [cpp:17]
       → commitRuntimePublication(...)                      [AudioEngine.h:4920]
         → enqueueRuntimePublicationFireAndForget(...)      [AudioEngine.h:4787]（enqueue 後 return）
         → waitForPublishReceipt(seqId, 250ms)              [AudioEngine.h:4942]  ← ここで block
   ← publish が return
   → Route A: resolveRecoveryObligation(req.id, Published)  [Orchestrator.cpp:324]

CoordinatorLoop（ISR）:
  PublishExecutor::executePublish
   → ... commit/swap ...
   → onPublishCommitted(seq, intent.payload.publish.recoveryObligationId)  [RuntimePublishExecutor.h:114]
     → notifyPublishReceipt(seqId)                          [Orchestrator.cpp:355] ← producer を起床
     → Route B: resolveRecoveryObligation(id, Published)    [Orchestrator.cpp:358]
```

- `executor_.publish` は `waitForReceipt=true` 固定（`publish()` の唯一 caller は Orchestrator.cpp:283、
  `publishImpl` の唯一 caller は `publish()`）→ **producer は receipt まで block**。
- 355（notify）と 358（Route B resolve）は CoordinatorLoop 上で隣接。producer は 355 で起床し、
  324（Route A）へ進む。したがって **Route A と Route B が同一 obligation に対して順不同で attempt し得る**。
  → **race は実在**（R27 静的監査の結論を再確認）。
- authority は既存どおり `resolveRecoveryObligation` → `RecoveryAdmissionTable::resolve` の
  **full-word CAS（Live→terminal）一箇所のみ**（Coordinator.cpp:508-531・:1031）。
  敗者は idempotent no-op（:528）。

### dynamic 検証可能性

```text
Winner identity（どちらが CAS を獲ったか）= UNOBSERVABLE
  - `won` 戻り値は両 site で破棄（:324／:358 は文として呼ぶのみ）
  - 専用 counter/getter は存在しない（R29 §7 により追加禁止）
  - liveCount_ の public passthrough なし（Full AudioEngine）
Exact-once terminalization = 間接的に観測可能
  - RuntimeIntentCoordinator::isFullyDrained() は liveCount()==0 を predicate に含む（Coordinator.cpp:561）
  - AudioEngine::isFullyDrained()（Threading.cpp:212）→ waitForDrain()（AudioEngine.h:1601 public）
  - したがって episode 後に drain が成立すれば「obligation が一度だけ terminal 化した」ことと整合。
    double-resolve による liveCount_ 破壊（underflow）や未 terminal は drain 不成立として現れる。
Contract 違反の動的検出 = crash/assert 0 の確認（fatal assert は Timer.cpp:1988 等）
```

- **単に resolve が 2 箇所あることを dynamic winner と呼ばない**（R29 §8）。winner identity は
  本 Step の観測では閉じられない（R28-B の残件そのもの）。

## 12. Minimal Test-only Vehicle Design

### 推奨：D1（production 0）

```text
入口: AudioEngine::submitRecoveryIntent(handle, buildSource)  ← すべて public・新 API 不要
配置: 既存 harness TU（例 P1PolyphaseGainCharacterization.cpp）へ test-only function 追加
      → 既存 target 内のため CMake 追加 0
手順（test-only）:
  1. h.start() → wait: observePublishedWorld()!=nullptr && engine.current!=nullptr
  2. handle = engine.registerDSPHandleForRuntime(activeDSP)（非 null を assert）
  3. snapshot = engine.getCurrentBuildSnapshotForRecovery(); snapshot.sealed = true;
  4. pre read: cmt/coord/seq/drp（+ take/bld/req/que）
  5. engine.submitRecoveryIntent(handle, snapshot)
  6. wait: seq 増加（timeout 付き）／ pending 収束
  7. post read: cmt/coord/seq/drp
  8. 検証:
     - seq 前進（Recovery-origin publish 到達）
     - coord 不変（Recovery pop は Main bucket 外）
     - waitForDrain(timeout) 成功（exactly-once terminalization・間接）
     - crash/assert 0
  9. repeat: 手順 4–8 を N 回（episode 逐次 → 各回で新 obligation）
```

### 補足：D2（production 1 行・Quarantine 段を含む）

```text
friend class RecoveryOriginTestAccess;（AudioEngine.h の既存 TestAccess 群に 1 行追加）
+ RecoveryOriginTestAccess::submitQuarantine(engine, handle, reason)
    → engine.runtimePublicationBridge_.submitQuarantine(
          handle, reason, engine.dspHandleRuntime(), engine.dspQuarantineManager(), epoch)
手順は D1 と同じ（段 1 が QuarantineIntentHandler 経由になる）。
```

- どちらも **per-task generation linkage を持ち込まない**（R29 §10）。
- D1 は Quarantine state を伴わないが、Recovery-origin publish の成立条件 [2]–[12] は満たす。
  Quarantine state を伴わせたい場合は「`quarantineSlot(slot,gen,reason)`（public）→
  `submitRecoveryIntent`」の順で state を再現することも可能（設計選択・R30 で確定）。

## 13. Production / Test / CMake Change Boundary

| 項目 | D1 | D2 |
| --- | --- | --- |
| production source | **0** | 1 行（friend） |
| test source | 1 点（test-only function・既存 TU） | 1 点＋TestAccess クラス |
| CMake | 0（既存 TU）／1 行（新 TU の場合） | 0／1 行 |
| counter/getter | 0 | 0 |
| Intent／recoveryObligationId／deferred／admission／retry／quarantine policy | 変更 0 | 変更 0 |
| per-task linkage | 追加しない | 追加しない |

- R29 §10 の禁止（production instrumentation／new counter／new getter／Intent 変更／
  semantics 変更／per-task linkage／F・R 本測定／P3-1-D／帰属／R28-B→R28-A 繰り上げ）を遵守。
- 本 Step では実装 0（設計のみ）。R30 で D1（または D2）を実装する。

## 14. R29 Gate

```text
R29-A： 非該当
  [PASS] 最新 ConvoPeq source reconciliation
  [PASS] Full AudioEngine recovery path を一本に固定（§3）
  [PASS] Recovery-origin vehicle の具体的入口を特定（§7 D1/D2・§12）
  [PASS] obligationId != 0 を publish まで保持（R27 §3 静的＋§9 条件）
  [PASS] Recovery pop が coord へ加算されないことを検証可能（§10・必要条件として）
  [PASS] cmt/coord/seq/drp の観測境界を維持（§10）
  [PASS] D105 Route A / Route B の dynamic 検証方法を定義（§11：exactly-once は drain で間接・winner は UNOBSERVABLE）
  [PASS] production 変更不要/必要の境界を明確化（§13）
  [FAIL] 「既存 vehicle で Recovery-origin publish を実際に発生させられる」根拠
        → §4–§8 のとおり**既存 vehicle では不可能**（A/B/C は前提を test から作れない）
R29-B： ADOPTED（Path B）
  既存 vehicle では Recovery-origin publish を決定論的に発生できない。
  最小 test-only 入口（D1：submitRecoveryIntent 直呼び）を定義：
    production 変更 0／test 最小 1 点／CMake 0 または 1 行。
  D105 dynamic は exactly-once（drain 経由）まで。winner identity は UNOBSERVABLE（R28-B 残件）。
R29-C： 非該当（contract 矛盾なし。R27/R28 の source finding と整合）
```

- source は R27 vehicle のまま（本 Step の差分は本ドキュメントのみ）。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。

## 15. Next Step

```text
Path B → R30 = Recovery-origin Vehicle Implementation Gate
  - D1（production 0）を第一候補として実装
  - D2（Quarantine 段含む・production 1 行 friend）を第二候補
  - 実装後に build gate → R31 で Recovery-origin full-pipeline measurement
    （cmt/coord/seq/drp window・coord 非加算・drain による exactly-once）
  - D105 winner identity は別途 observability gate（R28-B 残件）として分離
```

- R29 では vehicle 実装・build・run を行わない。
