# REPAIR PLAN 2 — 未実装・実装途中・将来実装項目の改修計画書

**日付:** 2026-08-08
**対象:** `doc/work88/REPAIR_PLAN.md` の実装チェックリスト（:236-257）・修正優先順位（:2446-2468）から抽出した残作業
**検証方法:** ソースコード実測（2026-08-08）。REPAIR_PLAN.md のステータス記載は一部古いため、本計画書で最新化した。
**位置づけ:** REPAIR_PLAN3.md が BUG-011〜046 の**検証**を担うのに対し、本計画書は残作業の**改修案（詳細設計）**を担う。

---

## 0. ステータス最新化サマリ（REPAIR_PLAN.md 記載との乖離）

> **三次レビュー最終判定（2026-08-08）:**
> | 項目 | 判定 | 概要 |
> |------|------|------|
> | BUG-014（MmcssPolicy enum atomic 化） | ✅ GO | 前版 `const char*` は pointed-to lifetime 未保証 → enum atomic 化は正しい。**underlying type は `uint8_t` を明示**（`enum class MmcssPolicy : uint8_t`） |
> | BUG-028（RT-only LinearRamp） | ✅ GO | RT 専有化は ISR の「RT は観測主体」原則に合致。NonRT→`dryScaleTarget_` atomic publish、RT→LinearRamp のみ（第3章） |
> | BUG-015/027（RetireQuarantineStore） | ✅ GO 条件付き | EBR 二重化の注意。**容量枯渇時の第2段 overflow（HealthEvent / shutdown escalation）が必要**（第2章追記） |
> | BUG-060 / BUG-061 | ✅ 変更不要 | 実測で fetchSub 単一アトミック / mutex 保護を確認済み |
> | FUTURE-3（recovery build） | ✅ GO 条件付き | contextEpoch 追加は良い。**epoch→sealedSnapshot の lifetime 保証を追加定義**（snapshotKey / 値コピー）。**四次実測追記: IR data は snapshot に無く現在の processor から取得 → semantic は「現在のユーザー構成を再 build + quarantined 除外」（レビュー指摘 A）**（第4章） |
> | SHUTDOWN-7 | ✅ GO（現案） | 完了条件への組込みは不要（join 済み）。**順序不変条件 + 防御 jassert** として明文化（第5章） |
> | FUTURE-10 MPSC 化 | 🔴 NO-GO（現案） | **「bounded CAS retry → drop」は撤回**。Publish/Quarantine/Recovery の drop は runtime state transition 喪失 = 致命的。**per-type admission policy に変更**（第6章） |
> | RecoveryIntentHandler | ✅ GO（再設計後） | enqueue-only + Builder Work Queue 分離が正しい。Recovery は共通 Intent Queue に入れない（第7章） |
> | Shutdown Pipeline 検証 | ✅ GO | 必須。シャットダウン順序を実測（第8章） |
> | FUTURE-5 MemoryPool / FUTURE-6 HandleTable | ⏸ 保留 | 現時点では優先度低。ISR semantic completion → soak/TSan → performance の順（第11・12章） |
> | **PublishReceiptWaiter（FIFO 前提）** | 🔴 要対応 | `complete()` は `seqId > lastCompleted_` の単調更新（AudioEngine.h:3566-3573）で、**FIFO 完了を前提**（:3564 コメント明記）。MPSC 化 + Publish の deferred 退避で out-of-order 完了が起きると `waitFor(101)` が誤成功（第6章追記・第14章 MUST） |
> | **retireDSPHandleForRuntime の shutdownReclaim** | 🔴 要対応 | transitional 措置が `DSPLifetimeManager::retire` で **reclaim→enqueue の順序逆転**を生む（AudioEngine.h:4156-4159 / ISRDSPHandle.h:158）。enqueue 失敗時は Reclaimed のまま deferred delete に入らず**リーク**。`requestReclaim` に一本化（第2章追記） |
> | **RetireEnqueueResult** | ✅ 再利用 | `Success/QueuePressure/QueueFull/Shutdown` が既存（ISRAuthorityClass.h:25-30）。tryEnqueue() 返り値としてそのまま使用（第2章） |
> | **RuntimePublishSpecification** | ✅ 確認 | `TopologyPart{activeDSP, fadingDSP}` を含む — 単なる handle map ではない（RuntimeBuilder.h:19-22）。Recovery は buildSource 値コピー + spec 除外で対応（第4章） |

> **五次レビュー最終判定（2026-08-08）:**
> | 項目 | 判定 | 概要 |
> |------|------|------|
> | BUG-014（MmcssPolicy enum atomic 化） | ✅ GO | **`static_assert(std::is_trivially_copyable_v<MmcssPolicy>)` を追加**（`std::atomic<enum>` の lock-free 性は enum が trivially copyable であることに依存）。underlying type `uint8_t` 明示（第1章反映済み） |
> | BUG-015/027（RetireQuarantineStore） | ✅ GO（条件付き） | **`RetireQuarantineStore` は `ISRRetireRouter` の **Policy lane 配下**に単一配置（ISRRetireRouter は stateless dispatcher であり内部状態を持たない — `ISRRetireRouter.h:34-44`参照）** — SnapshotCoordinator / DSPLifetimeManager はストアを直接保持せず **Router API（`quarantineRetire()`）経由**で移送（Authority Singularization）。Retire authority は 1 個のまま（第2章反映済み） |
> | BUG-028（CrossfadeRuntime リセット） | ✅ GO | 実測(2026-08-08): `complete()`（:95-103）は stale フラグ（`useDryAsOld_`/`firstIrDryPending_`/`firstIrDryDone_`/`queuedFadeTimeSec_`/`fadeStartTimestampUs_`）を **atomic publish で reset する** — これ自体は RT 安全。実際の問題は **dryScaleGain_（LinearRamp，非atomic）への NonRT 直接 `setCurrentAndTargetValue` 操作**であり、現コードは complete()（h:118）で `dryScaleGain_.setCurrentAndTargetValue(1.0)` を呼んでおり、これも NonRT→LinearRamp data race（二次レビュー追加発見）。start() は `dryScaleGain_` に触れないが `gain_.setCurrentAndTargetValue(0.0)`（h:41）で race を起こす。**五次レビュー §8 の「reset 禁止」は実コードを誤解**: reset は atomic で安全だが **`start()` が `gain_.setCurrentAndTargetValue(0.0)` を呼んでおり、NonRT→LinearRamp data race の根因**。`gain_.setCurrentAndTargetValue(0.0)` は `LinearRamp::totalSteps` と `currentValue` の両方を NonRT から書き込み、RT の `getNextValue()` と race する。改修案: **`start()` から `gain_.setCurrentAndTargetValue(0.0)` を削除**（BUG-028 fix — Ch.3 line 524 反映済み）。`gain_.reset()` のみ NonRT で実行（`totalSteps` のみ）、RT 側 `armCrossfadeIfPending`（`AudioEngine.h:3878`）の `setTargetValue(1.0)` で fade-in target を設定する。順序 invariant（complete() atomic batch publish → AudioThread `getDryScaleGain().getNextValue()` 消責）をテストで固定（第3章反映済み） |
> | FUTURE-3（recovery build） | ⚠️ 条件付き GO | `buildSource`（`RuntimeBuildSnapshot` 値コピー）は正しい。**`ConvolverProcessor::BuildSnapshot` は `juce::File`/`juce::String` を含み trivially copyable でないため内包せず（案 i）、build 時に `uiConvolverProcessor.captureBuildSnapshot()` から取得**（第4章コード例を五次レビューで修正済み） |
> | SHUTDOWN-7 | ✅ 実装 → **順序検証テストへ変更** | join 済みのためコード変更不要。`stopRebuildThread(join)` → `waitForDrain` の順序を `shutdown_trace.json` で**検証**する（第5章・第13章 Phase 6） |
> | FUTURE-10（共通 Intent Queue） | ⚠️ 条件付き GO | per-type admission policy（drop 禁止）+ **producer hole テスト** + `PublishReceiptWaiter` の monotonic-completion 維持 + **Recovery は Builder Work Queue 分離維持**（第6章反映済み） |
> | RecoveryIntentHandler | ✅ GO（enqueue-only 限り） | **enqueue-only である限り** GO。Decision/World 書換禁止（HANDLER-1）を厳守（第7章） |
> | FUTURE-5 / FUTURE-6 | ⏸ 後回し | 現時点での後回しで正しい。ISR 完成系 → soak/TSan → performance の順（第11・12章） |

> **五次レビュー推奨実装順（第13章反映済み）:** Phase 0（**不変条件テスト先行（INV-1〜7））**→ 1(BUG-014) → 2(BUG-028) → 3(BUG-015/027) → 4(FUTURE-3) → 5(FUTURE-10 MPSC) → 6(shutdown 検証) → 7(Recovery queue 整理) → 8(FUTURE-5/6)

> **Phase 0 不変条件（INV-1〜7）**（詳細は第13章 ※ 不変条件一覧）:
> - **INV-1**: RuntimeWorld authority — Builder 生成, Publish は Coordinator authority, retire は Coordinator authority. World immutable after publish.
> - **INV-2**: LinearRamp ownership — `dryScaleGain_`/`gain_` は RT のみ (`reset()` は Audio Thread 停止後のみ). NonRT は `dryScaleTarget_` atomic publish のみ.
> - **INV-3**: Retire ownership — retire → EBR → reclaim 順序. `directDelete` なし. `RetireQuarantineStore` は `ISRRetireRouter` の Policy lane 配下に単一配置（stateless dispatcher 配下、NOT internal）.
> - **INV-4**: Recovery semantic — Recovery = "quarantined 除外した現在の authoritative configuration の再構築"（過去 World の rollback ではない). IR は `transferIRStateFrom(engine.getConvolverProcessor())` で現在値取得.
> - **INV-5**: Intent loss — Publish/Quarantine/Recovery は drop 禁止. Observe は 3 層 fallback.
> - **INV-6**: Shutdown — admission gate closed → stop producers → join workers → drain → reclaim → verify の順序不変.
> - **INV-7**: MPSC ordering — sequenceId assignment → reservation → publication → consumption → completion の 4 順序を分離. Producer hole は seq 番号で検知.

> **三次レビュー必須修正（2026-08-08）:**
> 1. **FUTURE-10: 「bounded CAS retry → drop」を撤回** — Publish / Quarantine / Recovery は drop 禁止（drop は telemetry loss ではなく **Runtime state transition そのものの喪失**）。Intent type ごとの overflow policy（admission policy）を採用（第6章）。> 2. **FUTURE-3: `contextEpoch → sealedSnapshot` の lifetime を明文化** — epoch が分かってもその epoch の snapshot が存在する保証はない（過去 World は retire/reclaim 済みの可能性）。build input の lifetime を明示管理（第4章）。
> 3. **RetireQuarantineStore: capacity exhaustion policy を追加** — Store full 時に `delete` は絶対しない。HealthEvent / shutdown escalation へ（第2章）。
> 4. **MPSC ring は SPSC ring の小改造ではなく独立した bounded MPSC primitive として検証** — producer hole と memory ordering をテスト（第6章）。
> 5. **PublishReceiptWaiter の monotonic-completion invariant を維持** — MPSC 化・deferred 退避でも `complete()` の単調更新を保つ（out-of-order で `waitFor` が誤成功しない）（第6章追記・第14章 MUST）。
> 6. **retire 順序の逆転を修正** — `DSPLifetimeManager::retire` は reclaim→enqueue の順（AudioEngine.h:4156-4159）で、enqueue 失敗時に Reclaimed のまま deferred delete に入らずリーク。enqueue →（成功時のみ）reclaim / 失敗時は RetireQuarantineStore 移送に変更。`shutdownReclaim` transitional 措置は削除（第2章）。

> **四次実測照合（2026-08-08）— 全14章の根拠コードを再実測し、計画書の行番号・挙動記述が実コードと一致することを確認:**
> | 章 | 照合結果 | 実測で確認した根拠コード |
> |----|----------|---------------------------|
> | 1. BUG-014 | ✅ 一致 | `currentDeviceTypeName_`（AudioEngine.h:2356, setter h:2348）／RT パス 3 箇所（AudioBlock.cpp:56, BlockDouble.cpp:60, ReleaseResources.cpp:81）／`getCurrentMmcssPolicy`（Mmcss.cpp:54-64） |
> | 2. BUG-015/027 | ✅ 一致 + 逆転確認 | `RetireEnqueueResult`（ISRAuthorityClass.h:25-30）／`SnapshotCoordinator.h:88` 捕捉+TODO, `:100` 未捕捉／`DSPLifetimeManager.cpp:49-53` ignoreUnused／`ISRRetireRouter.cpp:154-157` TODO／**`retireDSPHandleForRuntime`（h:4141-4163）が retire(:4155)→shutdownReclaim(:4159) を先に実行し、呼び出し元 `DSPLifetimeManager::retire` で enqueueWithRetry → reclaim→enqueue 逆転を再確認** |
> | 3. BUG-028 | ✅ 一致 | `start()`（CrossfadeRuntime.h:38-51）は `dryScaleGain_` 非操作／`complete()`（:95-103）は stale フラグ atomic reset のみ — **しかし h:118 で `dryScaleGain_.setCurrentAndTargetValue(1.0)` が呼ばれ NonRT→LinearRamp race（二次レビュー追加発見）**／`reset()`（:117-118）のみ `setCurrentAndTargetValue(1.0)`／RT パス（AudioBlock.cpp:442, BlockDouble.cpp:421） |
> | 4. FUTURE-3 | ✅ 一致 + Producer/Consumer 未接続 | `RecoveryIntent`（h:151-159）は **buildSource を含まない** — 実際の struct は3フィールドのみ: `DSPHandle handle` / `PublicationEpoch epoch` / `uint64_t intentId`（trivially_copyable + standard_layout static_assert 実装済み at h:156-159）。**buildSource フィールドは FUTURE-3 Phase 4 のコード追加予定（未実装）**。コード例（h:605）で示す `recovery->buildSource` は将来設計であり、現在の struct には存在しない。`submitRecoveryRequest`/`popRecoveryRequest`（cpp:648-672）実装済み／`recoveryIntentQueue_`（h:394, kRecoveryIntentQueueCapacity=256）／**プロダクションコード上の呼び出し元不在（単体テスト ISRSemanticValidationTests.cpp:595-596 のみ。Builder Loop 未接続）**。**buildSource フィールドは FUTURE-3 Phase 4 の設計追加（未実装）。`buildSource` 設計（RuntimeBuildSnapshot 値コピー）は第4章で正しいが、**現コードには未実装**。IR data 供給元の実測: `RuntimeBuilder::build()`（cpp:443-447）は `applyBuildSnapshot`（metadata のみ）→ `transferIRStateFrom(engine.getConvolverProcessor())`（現在の `uiConvolverProcessor` から AudioBuffer コピー。ConvolverProcessor.h:1130-1147）。`RuntimeBuildSnapshot`（RuntimeBuildTypes.h:48-66）に IR AudioBuffer は無い → Recovery semantic = 現在のユーザー構成を再 build（レビュー指摘 A）** |
> | 5. SHUTDOWN-7 | ✅ 一致 | `rebuildThreadIsRunning`（AudioEngine.h:2530, atomic false 初期値）／RebuildDispatch.cpp:799(true)/:1066(false)／`rebuildWorkerRunning`（RuntimeBuilder.cpp:284 const bool = false ハードコード）／ReleaseResources.cpp:75 requestShutdown → :188 StopWorkers → :189 shutdownCoordinatorLoop → :190 stopRebuildThread → :191 ObserverDrained → :430 waitForDrain の順序 |
> | 6. FUTURE-10 | ✅ 一致 | `intentQueue_`（h:398-399, LockFreeRingBuffer<Intent,4096>）／`observeIntentQueue_`（h:376）/`observeFallbackQueue_`（h:380）/`recoveryIntentQueue_`（h:394）／LockFreeRingBuffer SPSC 専用（LockFreeRingBuffer.h:2,34）／`enqueuePublicationIntent`（h:227-240, push は :240）full 時 false→caller が ownerChannel 回収（AudioEngine.h:4296-4299）／PublishReceiptWaiter（h:3565-3587）`complete()` は `seqId > lastCompleted_` のみ |
> | 7. RecoveryIntentHandler | 🔴 NO-GO（未実装） | `ISRIntentDispatcher.h:43-44` handle は **no-op `{}`** — Builder Work Queue へ転送未実装。kDispatchTable 1:1 registration は済み。`static_assert(DispatcherHasNoDecision)`（:76-83）実装済み。Phase 4 (FUTURE-3) 実装必須。 |
> | 8. Shutdown Pipeline | ✅ 実装確認 | `ISRShutdown.cpp`（transitionTo:108-136, markTimedOut:56-96, markFailed:98-106, emitShutdownTrace:168-303, setBoundedTeardownCounters:305-314）／`RuntimeDrainAudit.h`（getPrimaryBlockingReason:65-74, isAllZero:77-84）／`waitForDrain`（Threading.cpp:138-170）/`isFullyDrained`（:114-136）/`collectDrainAudit`（:70-112）／SHUTDOWN-1〜6 の各契約が isFullyDrained/collectDrainAudit に実装済み |
> | 9. BUG-060 | ✅ 一致 | `ISRRetireRuntimeEx.cpp:215-222` fetchSubAtomic + previous==0 回復／`EpochDomain.h:399-402` `isOlder`（static_cast<int64_t>(a-b)<0） |
> | 10. BUG-061 | ✅ 一致 | `ISRDSPQuarantine.h:83-86` auditMutex_ + auditLog_／`.cpp` 全アクセス lock_guard（:37,59,85,119,145,177）／compactAuditLogLocked（:158）は lock 内からのみ呼ばれる |
> | 11. FUTURE-5 | ✅ 一致 | `ISRDSPHandle.h:115` MAX_DSP_SLOTS=256／`:176` `std::array<DSPRegistrySlot, 256> registry_`／`ISRDSPHandle.cpp:41-42` 線形スキャンで空き slot 検索（O(n)） |
> | 12. FUTURE-6 | ✅ 一致 | `AudioEngine.h:4109,4146,4174,4187` runtimeDSPHandleMapMutex_／:4600-4601 `std::unordered_map`／:4182-4193 `eraseByHandle` O(n) 線形スキャン |

> **二次レビュー最終判定（2026-08-08）:**
> | 項目 | 判定 | 概要 |
> |------|------|------|
> | BUG-014（MmcssPolicy enum atomic 化） | ✅ GO | `publishAtomic`/`consumeAtomic` 経由 + `static_assert` ロックフリー（第1章） |
> | BUG-015/027（RetireQuarantineStore 新設） | ✅ GO（条件付き） | `EpochDomain::isOlder` 使用・directDelete 禁止・retire/physical delete 責務分離（第2章） |
> | BUG-028（CrossfadeRuntime リセット） | 🔴 NO-GO → 再設計 | 旧案は RT 稼働中に LinearRamp（非 atomic）へ NonRT 書き込み = データレース導入。**RT 専有化**（第3章） |
> | FUTURE-3（recovery build 接続） | 🔴 NO-GO → 再設計 | `resolve()` は Quarantined を拒否 → quarantinedHandle 単独 payload では build 不能。**contextEpoch 引当を追加**（第4章） |
> | SHUTDOWN-7（VerifyDrained） | 🔴 NO-GO → 再設計 | `stopRebuildThread(join)` が `waitForDrain` より前 → 完了条件への組込みは冗長。**shutdown 順序確定**（第5章） |
> | FUTURE-10 前提 0（intentQueue_ MPSC 化） | ✅ GO | CAS retry は**有界**（kMaxProducerRetries 超えで drop）（第6章） |
> | FUTURE-10 Observe 統合 | ✅ GO（MPSC 完成後） | cross-type FIFO 順序保証テストを追加（第6章） |
> | FUTURE-10 Recovery 統合 + RecoveryIntentHandler | 🔴 NO-GO → 再設計 | intentQueue_ 再 enqueue で無限循環。**Intent Queue と Builder Work Queue 分離**（第6・7章） |
> | BUG-060 / BUG-061 | ✅ 実装済み | 実測で fetchSub 単一アトミック / mutex 保護を確認（第9・10章） |
> | FUTURE-5 + FUTURE-6 | ✅ GO（後回し） | Storage Policy。ISR 完成後に一括（第11・12章） |

REPAIR_PLAN.md のチェックリストは 2026-07-31 時点の記載であり、その後（特に f0d6406 "Deferred Admission Integration"）で解決済みの項目がある。実コードで再確認した結果:

| REPAIR_PLAN.md 記載 | 実コード確認結果（2026-08-08） | 判定 |
|---------------------|-------------------------------|------|
| 5. FUTURE-9: BUG-052 `hasDeferred_` → atomic **未完了** | `RuntimePublicationOrchestrator.h:249` で `std::atomic<bool> hasDeferred_{false}` 済み。f0d6406 で解決 | ✅ 解決済み |
| 13-2: `deferredSlot_` mutex 保護 HIGH | Single Thread Owner 契約（jassert スレッドガード付き）により non-atomic `deferredSlot_` の並行アクセスは構造的に排除済み | ✅ 設計で解決 |
| 6.1 FUTURE-10: `kDispatchTable` 1:1 + `static_assert(DispatcherHasNoDecision)` | `ISRIntentDispatcher.h:58-89` で実装済み | ✅ 実装済み |
| 6.2 FUTURE-10: Handler = Executor のみ | Observe/Publish/Quarantine は実装済み。**`RecoveryIntentHandler::handle` は no-op `{}`（`ISRIntentDispatcher.h:44`）** — Builder Work Queue 転送未実装 | 🔴 NO-GO（Phase 4 実装必須） |
| 13-2: BUG-054 `onPublishCompleted` に oldHandle 渡す | `DSPTransition.h:49-53` に `oldHandle` 引数あり、`RuntimePublishExecutor.h:77` が `p.decision.oldHandle` を渡している | ✅ 実装済み |
| 2.3 FUTURE-3: recovery-world build 未着手 | transport（`submitRecoveryRequest`→`popRecoveryRequest` 1-hop）は `ISRRuntimePublicationCoordinator.cpp:660-668` で完了。**Builder Loop の recovery-world build 接続が未着手** | 🔄 実装途中 |
| 13-1: BUG-028 CrossflareRuntime reset | `start()` に `startDelayBlocks_`/`dryHoldSamples_` リセット追加済み。**実測: `gain_.reset()`（h:40, AudioEngine.Init.cpp:19）も NonRT から `LinearRamp::totalSteps` を書き換えるため RT→LinearRamp data race の候補**（二次レビュー追加発見）。`dryScaleGain_` は RT 専有（RT-only ownership）に確定 — しかし `complete()`（h:118）で `dryScaleGain_.setCurrentAndTargetValue(1.0)` が呼ばれ NonRT→LinearRamp race も NO-GO。`start()` も `gain_.setCurrentAndTargetValue(0.0)`（h:41）で race。`gain_.reset()` は RT 停止後または RT パスの `armCrossfadeIfPending` 内で行わなければならない。旧案（start/complete で `setCurrentAndTargetValue`）は RT 稼働中のデータレースを生むため NO-GO → 第3章で再設計 | 🔴 P0 NO-GO（再設計） |
| 13-1: BUG-014 `currentDeviceTypeName_` | RT パス（`AudioEngine.Processing.AudioBlock.cpp:55` 等）が `getCurrentMmcssPolicy()` 経由で juce::String を直接参照。文字列は MMCSS ポリシー導出専用。`getAudioDeviceTypeName()` 0 コーラー | 🔴 未修正（enum atomic 化を第1章に設計） |
| 13-1: BUG-015/027 `enqueueWithRetry` | 失敗 = `EpochDomain` の `enqueueRetire` queue full（= DeferredDeletionQueue / Vyukov MPSC bounded queue full）。queue full は RT Reader が参照中の可能性 → `directDelete` は UAF 発生。**実測: `ISRRetireRouter` は stateless EpochDomain wrapper（no RetireQuarantineStore）。`enqueueWithRetry` 失敗時は `QueuePressure` を返し caller に退避を委ねる**（`ISRRetire.cpp:38-68` の `emitRetireIntent` 内で fallback queue / overflowRing_ への退避）。**RetireQuarantineStore は実在せず（`ISR_BUG.md:115` 設計のみ）**。`retireDSPHandleForRuntime`（AudioEngine.h:4141-4163）は `retire → shutdownReclaim`（epoch ゲートなし）を即座に実行する Transitional bug を有。Phase 4 で修正必須（BUG-015/027 fix）。 | 🔴 未修正（退避ストア新設を第2章に設計） |
| FUTURE-10: `intentQueue_` の Producer | `LockFreeRingBuffer` は SPSC 専用だが、push は Builder（`AudioEngine.h:4296`）/ Timer（`cpp:696`）/ CoordinatorLoop（`h:4236`）の**複数スレッド** = 既に MPSC 実態（潜在競合） | ⚠️ 前提 0 として MPSC 化必須 |

本計画書が扱う残作業:

| 区分 | 項目 | 状態 |
|------|------|------|
| バグ修正 | BUG-014 / BUG-015 / BUG-027 / BUG-028（残り） | 🔴 未修正・部分 |
| バグ修正 | BUG-060 / BUG-061 | ✅ 実装済み（二次レビューで訂正） |
| FUTURE-3 | recovery-world build の Builder Loop 接続（**payload に `buildSource`（RuntimeBuildSnapshot 値コピー）を追加 — 現 RecoveryIntent は handle/epoch/intentId のみ）** | 🔴 P0 NO-GO（再設計） |
| FUTURE-9 | SHUTDOWN-7: shutdown 順序確定（完了条件への組込みは撤回） | 🔴 P0 NO-GO（再設計） |
| FUTURE-10 | 共通 Intent Queue 一本化（Observe のみ。**Recovery は Builder Work Queue 分離**） | ☐ 未着手（※ `intentQueue_` は既に MPSC 実態 — 前提 0 必須） |
| FUTURE-10 | `RecoveryIntentHandler::handle` 実装（enqueue-only + Builder Work Queue 転送） | 🔴 P0 NO-GO（再設計） |
| 検証 | Shutdown Pipeline 検証（SHUTDOWN-1〜7 と Intent Queue の対応） | ☐ 未着手 |
| FUTURE-5 | MemoryPool化 | ☐ 未着手 |
| FUTURE-6 | Handle Table 完全移行 | ☐ 未着手 |

> **2026-08-08 再調査による訂正（REPAIR_PLAN.md 記載との乖離）:**
> - BUG-014: 前版の `std::atomic<const char*>` 方式は RT 読み取り中の UAF リスクあり → **`MmcssPolicy` enum の atomic 化**に変更（第1章）。
> - BUG-015/027: `directDelete` は queue full（= RT 参照中）時に UAF → **`RetireQuarantineStore` 新設 + EBR 安全削除**に変更（第2章）。**実測: `ISRRetireRouter` は stateless wrapper — `enqueueWithRetry` 失敗時は `QueuePressure` を返す。caller が退避を処理すべき。`retireDSPHandleForRuntime`（AudioEngine.h:4159）は `shutdownReclaim` を即座に呼ぶ transitional bug。**
> - FUTURE-10: `intentQueue_` は既に複数 Producer（Builder/Timer/CoordinatorLoop）で **MPSC 実態**。`LockFreeRingBuffer` は SPSC 専用のため **前提 0（MPSC 化）が必須**（第6章）。
> - SHUTDOWN-7: 対象は `rebuildWorkerRunning`（常 false ハードコード）でなく **`rebuildThreadIsRunning`**。タイムアウト時 reason は**実装済み**。~~完了条件への組込み~~ → **二次レビューで撤回**（waitForDrain 時点で Builder は join 済みのため冗長。第5章参照）。

> **2026-08-08 二次レビュー（P0/P1 指摘）による改訂:**
> - **BUG-028 → NO-GO（再設計）**: 旧案は `start()`/`complete()`（NonRT）で `dryScaleGain_.setCurrentAndTargetValue(1.0)` を追加するものだったが、`dryScaleGain_` は `convo::LinearRamp`（非 atomic）で **RT パス（`AudioBlock.cpp:442`/`BlockDouble.cpp:421`）が `getNextValue()` を実行中に書き込む = データレースを新規導入する**。`LinearRamp` は `reset()`/`setCurrentAndTargetValue()` を **NonRT 専用**（`DspNumericPolicy.h:310-313`）と宣言しており、RT 稼働中の NonRT 書き込みは契約違反。→ **`dryScaleGain_`/`gain_` は RT 専有（RT-only ownership）に確定**。NonRT は `dryScaleTarget_`（atomic）のみ publish し、LinearRamp への直接操作は RT パスのみが行う（第3章）。
> - **FUTURE-3 → NO-GO（payload 不備）**: `RecoveryIntent` の payload は `quarantinedHandle` のみだが、**`DSPHandleRuntime::resolve()` は Quarantined state を `{nullptr,false,false}` で拒否する**（`ISRDSPHandle.cpp:69`）。すなわち Builder Loop が quarantinedHandle から DSP spec（build 入力）を**復元できない**。Recovery World を build するには、quarantined 対象の **build spec を別経路（sealedSnapshot / RuntimeBuildSnapshot）から引き当てる**設計が必要（第4章）。
> - **FUTURE-10 / RecoveryIntentHandler → NO-GO（循環）**: 第7章の案 A（Handler が `submitRecoveryRequest` を呼ぶ）を第6章の「`submitRecoveryRequest` を `intentQueue_.push` に変更」と組み合わせると、**Recovery Intent → intentQueue_ → Dispatcher → Handler → submitRecoveryRequest → intentQueue_ の無限循環**になる。**Intent Queue（CoordinatorLoop が消費）と Builder Work Queue（Builder Loop が消費）を分離**し、Recovery は Builder Work Queue 側のみで流す（第6章・第7章）。
> - **SHUTDOWN-7 → NO-GO（順序未確定のまま完了条件へ組込むのは誤り）**: 実測では `stopRebuildThread()`（join で Builder 完全終了）が `waitForDrain()` **より前に**呼ばれている（`ReleaseResources.cpp:190` → `:430`）。すなわち waitForDrain 時点で Builder は既に停止済みであり、完了条件へ `rebuildThreadIsRunning` を追加するのは**冗長**。先に shutdown 順序（Builder 停止要求 → Builder join → waitForDrain）を確定する（第5章）。
> - **BUG-061 → ✅ 実装済みに訂正**: `auditLog_` は `mutable std::mutex auditMutex_` で既に保護済み（`ISRDSPQuarantine.h:86`、`.cpp` 全アクセス lock）。**RT パスからの参照はゼロ**（`quarantineActiveFlags_` のみ RT が atomic access）。第10章の設計案は撤回し、実装済みとして文書化。
> - **BUG-060 → ✅ 実装済みに訂正**: `EpochControl::reclaim` は既に **fetchSub 単一アトミック + 回復**で TOCTOU を排除済み（`ISRRetireRuntimeEx.cpp:214-222`）。`cur - last` の unsigned underflow パターンは対象コードに存在しない。第9章の設計案は撤回。
> - **MmcssPolicy atomic**: 第1章の設計は `publishAtomic`/`consumeAtomic`（`AtomicAccess.h`）経由で実装すること（直接 `store`/`load` を書かない）。`std::atomic<MmcssPolicy>` のロックフリー性は `static_assert` で担保する。

実装順序の原則（REPAIR_PLAN.md:230 踏襲）: ISR 完成系を優先 → Storage Policy（FUTURE-5/6）は最後。

---

## 1. BUG-014: `currentDeviceTypeName_` juce::String CoW race

**状態:** 🔴 P0 未修正
**対象:** `src/audioengine/AudioEngine.h:2338,2348-2349,2356`, `src/audioengine/AudioEngine.Mmcss.cpp:54-64`, `src/audioengine/AudioEngine.CtorDtor.cpp:89`

### 現状
```cpp
// AudioEngine.h:2348-2349, 2356
void setAudioDeviceTypeName(const juce::String& type) noexcept { currentDeviceTypeName_ = type; }
[[nodiscard]] const juce::String& getAudioDeviceTypeName() const noexcept { return currentDeviceTypeName_; }
juce::String currentDeviceTypeName_;
```
`juce::String` は CoW（Copy-on-Write）。Audio Thread が `getAudioDeviceTypeName()` の参照を保持したまま Message Thread が `setAudioDeviceTypeName()` で再代入すると、参照カウント更新のデータ競合（UB）になる。

呼出し実測（2026-08-08 再調査）:
- **RT 読み取り**: `getCurrentMmcssPolicy()` が `AudioEngine.Processing.AudioBlock.cpp:56` / `BlockDouble.cpp:60` / `ReleaseResources.cpp:81` の **RT パス**から呼ばれ、`AudioEngine.Mmcss.cpp:56` で `const auto& type = currentDeviceTypeName_;` として直接参照。
- **書き込み**: `setAudioDeviceTypeName` は `DeviceSettings.cpp:1133,1260`（NonRT のみ、`dev->getTypeName()` の結果）。
- `getAudioDeviceTypeName()` は **0 コーラー**（検証済み）。
- 用途は **MMCSS ポリシー enum の導出のみ**（文字列 "WASAPI"/"ASIO"/"DirectSound" → `MmcssPolicy`）。

### 改修案（詳細設計）: 文字列を残さず **enum の atomic 化**（レビュー指摘反映）
> 前版の `std::atomic<const char*>` + `exchange`/`delete[]` 方式は **RT 読み取り中の UAF** を生むため撤回。
> `getCurrentMmcssPolicy()` は RT パスで `load` → `strstr` する。その間に Message Thread が `exchange` + `delete[] old` を実行すると、RT が**解放済みポインタ**を `strstr` で走査 = UAF。acquire/release は「RT が読み終える前に delete しない」ことを保証しない（deleter は RT の完了を待たない）。
>
> 代わりに、**文字列比較そのものを NonRT の setter で行い、RT に公開するのは `MmcssPolicy` enum（trivially copyable）のみ**にする。これでポインタ寿命問題・CoW・アロケーションが全て消える。

```cpp
// AudioEngine.h — メンバ変更
//   MmcssPolicy は既存 enum class : uint8_t（AudioEngine.h:2338）→ underlying type 明示済み。
//   三次レビュー: underlying type を uint8_t で固定し、atomic 化はロックフリー保証。
std::atomic<MmcssPolicy> currentMmcssPolicy_{MmcssPolicy::None};
// 五次レビュー: enum 自体の trivially-copyable を static_assert で保証する
//   （`std::atomic<enum>` の lock-free 性は enum が trivially copyable であることに依存）。
static_assert(std::is_trivially_copyable_v<MmcssPolicy>,
    "BUG-014: MmcssPolicy must be trivially copyable (plain enum class : uint8_t).");
static_assert(std::atomic<MmcssPolicy>::is_always_lock_free,
    "BUG-014: MmcssPolicy must be lock-free (trivially copyable enum). RT path does a plain atomic load.");

// 設定（Message Thread からのみ。文字列比較は NonRT 側で完結）
void setAudioDeviceTypeName(const juce::String& type) noexcept
{
    MmcssPolicy p = MmcssPolicy::None;
    if (type.containsIgnoreCase("WASAPI") || type.containsIgnoreCase("Windows Audio"))
        p = MmcssPolicy::JuceManaged;
    else if (type.containsIgnoreCase("ASIO"))
        p = MmcssPolicy::SelfManagedProAudio;
    else if (type.containsIgnoreCase("DirectSound"))
        p = MmcssPolicy::SelfManagedPlayback;
    convo::publishAtomic(currentMmcssPolicy_, p, std::memory_order_release);
}

// 取得（RT 読みでも安全。1 byte atomic load のみ）
[[nodiscard]] MmcssPolicy getCurrentMmcssPolicy() const noexcept
{
    return convo::consumeAtomic(currentMmcssPolicy_, std::memory_order_acquire);
}
```

**二次レビュー P1 補強（実装時の必須事項）:**
1. `currentMmcssPolicy_` への**全ての読み書きは `AtomicAccess.h` の `publishAtomic` / `consumeAtomic` wrapper 経由で行う**（直接 `store`/`load` を書かない）。`AtomicAccess.h` は既存の atomic アクセス規約（release/acquire + 必要に応じた fence 統一）をカプセル化しており、`AudioEngine.h` 内の他 atomic（`rebuildThreadIsRunning` 等）と同じ流儀に揃える。`static_assert(std::atomic<MmcssPolicy>::is_always_lock_free)` で RT パスのロックフリーをコンパイル時に保証する。
2. **NonRT thread contract の文明化**: `setAudioDeviceTypeName()` に `jassert(juce::MessageManager::getInstance()->isThisTheMessageThread())` を追加 — 既存コードベースのパターン（`AudioEngine.Init.cpp:146`, `AudioEngine.RebuildDispatch.cpp:510` と同型）。設計文書だけの契約にしない。setter が Message Thread 以外から呼ばれることを防ぐ。

**削除:** `currentDeviceTypeName_`（juce::String）と `getAudioDeviceTypeName()` は**両方削除**（0 コーラー確認済み）。`setAudioDeviceTypeName(const juce::String&)` のシグネチャは**維持**（呼出し元 `DeviceSettings.cpp:1133,1260` の変更を不要にする）が、本体を enum 変換に置換。`AudioEngine.Mmcss.cpp:54-64` の `containsIgnoreCase` 比較ロジックは setter へ移動し、`getCurrentMmcssPolicy()` は上記の atomic load 1 行になる。

**注意（デストラクタ）:** ポインタ管理が無くなったため、`~AudioEngine` の `delete[]` cleanup は不要。`AudioEngine.CtorDtor.cpp:89` は**変更不要**（対象を外す）。

### 検証
- **四次実測照合（2026-08-08）:** `currentDeviceTypeName_`（`AudioEngine.h:2356`, setter `h:2348`）を RT パス 3 箇所（`AudioBlock.cpp:56`, `BlockDouble.cpp:60`, `ReleaseResources.cpp:81`）が `getCurrentMmcssPolicy()`（`Mmcss.cpp:54-64`）経由で参照することを実測。enum atomic 化後は `static_assert(std::atomic<MmcssPolicy>::is_always_lock_free)` + `publishAtomic`/`consumeAtomic` で実装する。
- `rg "currentDeviceTypeName_|getAudioDeviceTypeName"` で参照が 0 になることを確認
- `rg "getCurrentMmcssPolicy"` で全呼出し（RT 3 箇所 + Mmcss.cpp 内部）が enum 比較のままコンパイル通ること
- ASan/TSan CI（`.github/workflows/sanitizer-ci.yml`）で MMCSS 有効時のレース消滅を確認
- 手動: デバイス種別切替（ASIO↔WASAPI）でクラッシュ・ヘタリ音なし

### リスク
低。enum 変換ロジックは setter へ移動するだけ（比較条件は不変）。RT パスは atomic load 1 回に単純化され、むしろ従来より軽い。唯一の注意点は `setAudioDeviceTypeName` が「セッション開始時に 1 度だけ」の既存契約（AudioEngine.h:2345 コメント）を維持すること。再設定（デバイス切替）があっても atomic なので安全。

---

## 2. BUG-015 + BUG-027: `enqueueWithRetry` failure recovery

**状態:** 🟡 P1 未修正（`// ★ Future: RuntimeHealthMonitor へ通知` の TODO のみ）
**対象:** `src/core/SnapshotCoordinator.h:100,158,160` / `src/core/SnapshotCoordinator.cpp:38,94` / `src/audioengine/ISRRetireRouter.cpp:154` / `src/audioengine/DSPLifetimeManager.cpp:49,90`

### 現状
`SnapshotCoordinator::enqueueWithRetry` は 2 系統ある:
- **(A) `SnapshotCoordinator::enqueueWithRetry`（static, bool 返却）**: `SnapshotCoordinator.cpp:38,94` で返り値を未チェック。失敗時（retire epoch 不一致等）に snapshot がリークする経路が残る。`SnapshotCoordinator.h:88,90` に `// ★ Future: RuntimeHealthMonitor へ通知` の TODO。
- **(B) `ISRRetireRouter::enqueueWithRetry`（メンバ, RetireEnqueueResult 返却）**: `ISRRetireRouter.cpp:154` は TODO のみ。`DSPLifetimeManager.cpp:49,90` は `ignoreUnused` で返り値を握り潰している。

### 実測調査（2026-08-08）— 失敗の意味論と directDelete の危険性

**`DeferredDeletionQueue::enqueue` の失敗条件は queue full のみ**（`DeferredDeletionQueue.h:100` `return false; // Full`）。epoch による拒否は enqueue 時に**存在しない**（epoch 判定は reclaim 時に `isOlder(entry.epoch, minReaderEpoch)` で行う）。

**queue full の意味:** `reclaim()` は FIFO 先頭からしか解放しない（`DeferredDeletionQueue.h:135` "現在の dequeue 先頭と一致した時だけ削除する"）。つまり queue full は「先頭エントリの epoch がまだ `minReaderEpoch` に達していない（= RT Reader が参照中）」可能性が高い。この状況で **`directDelete`（即時解放）は RT 参照中のオブジェクトを破壊する = UAF**。

**レビュー指摘の妥当性:** `RetireQuarantineStore` は実在せず（`doc/work59/ISR_BUG.md:115` で「新設」設計されたまま）。実装上の quarantine 機構は `DSPQuarantineManager::quarantineHandle`（DSP ハンドル用）+ `EpochDomain::quarantineReader`（Reader スロット用）だが、**retire エントリの退避ストアは存在しない**。

### 改修案（詳細設計）: `directDelete` 撤回 → **RetireQuarantineStore 新設 + EBR 安全削除**

カテゴリ分類（**五次レビュー §5 反映 — Authority Singularization**）:
- **`RetireQuarantineStore` は `ISRRetireRouter` の **Policy lane** 配下に単一配置（ISRRetireRouter は stateless dispatcher — `ISRRetireRouter.h:34-44` 参照）**。既存の overflow 退避構造（`ISRRetire.cpp:50-77` の fallback queue → `overflowRing_` への退避試行。`RetireOverflowRing` は `ISRRetire.h:17` 前方宣言・`ISRRetireOverflowRing.h` 完全定義）の **NonRT 側拡張**として Router 配下に置く。
- **SnapshotCoordinator も DSPLifetimeManager も別の退避 authority を持たない**（Retire authority は 1 個のまま — Router 配下に Queue と QuarantineStore）。退避ストアへの移送は**必ず Router API 経由**（`retireQuarantine()`）で行い、各クラスが `m_retireQuarantine` を直接保持する案は**撤回**。
- **Category A（SnapshotCoordinator）**: 呼出し元（`startFade`/`completeFade`/`switchImmediate`/`retireCurrentAndTarget`）はすべて NonRT。失敗時は**Router 経由で退避ストアへ移送**。
- **Category B（ISRRetireRouter/DSPLifetimeManager）**: Router 内部の退避ストアへ移送。追加リトライ不要（内部で tryReclaim + 2 回試行済み）。

**RetireQuarantineStore — DOES NOT EXIST in current codebase**. `ISRRetireRouter` (ISRRetireRouter.cpp:97-131) is a stateless EpochDomain wrapper — it delegates `enqueueRetire` to `provider_->enqueueRetire()` and when that fails (QueuePressure), returns `QueuePressure` to the caller. There is NO quarantine fallback store. The fallback mechanism is at `ISRRetire.cpp:38-68` (in `emitRetireIntent` — the Vyukov MPSC bounded queue has a mutex-protected `fallbackQueue_` and an `overflowRing_`). A new `RetireQuarantineStore` would need to be **created**, not placed under the stateless router.

```cpp
// ISRRetireRouter 内部メンバ（五次レビュー §5 — Authority Singularization）。
//   既存の overflow 構造（ISRRetire.cpp:50-75 の overflowRing_ 退避試行）の NonRT 側拡張として
//   Router 配下に置く。SnapshotCoordinator / DSPLifetimeManager からは Router API
//   （retireQuarantine()）経由でのみ移送し、各クラスはストアを直接保持しない。
// src/DeferredDeletionQueue.h の DeletionEntry と同等のフィールドを保持（ownership transfer の完全性）。
// 保持フィールド: { ptr, deleter, epoch, type, publicationSequenceId, generation, reason, enqueueTimeUs }
//   - type/generation/publicationSequenceId: DeferredDeletionQueue::DeletionEntry と同型（退避エントリの EBR safe drain を epoch+type+generation で判定）
//   - ptr/deleter/epoch/type: 既存 DeferredDeletionQueue enqueue の引数と 1:1 対応（退避 → drain → deleter call が既存 EBR パスと同一の epoch 判定を使う）
//   - reason/enqueueTimeUs: 診断用
// 設計: ISR_BUG.md:110-119 の quarantineRetire 移送先。永続リークを防ぎつつ、
//       RT 参照中のオブジェクトは解放しない（EBR 安全削除）。
struct QuarantinedEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId = 0;  // ★ 因果追跡（DeferredDeletionQueue と同型）
    uint64_t generation = 0;             // ★ 世代追跡（DSPLifetimeManager::currentRetiringGeneration_ と同型）
    const char* reason = nullptr;
    uint64_t enqueueTimeUs = 0;
};
static_assert(std::is_trivially_copyable_v<QuarantinedEntry>,
    "QuarantinedEntry must be trivially copyable for lock-free compatible storage");

class RetireQuarantineStore {
public:
    bool quarantine(void* ptr, void (*deleter)(void*), uint64_t epoch,
                    DeletionEntryType type, const char* reason,
                    uint64_t publicationSequenceId = 0, uint64_t generation = 0) noexcept;
        // 戻り値 false = store full（呼出し元は deleter を実行してはならない）。
        //   type/generation/publicationSequenceId を保持し、drain 時の epoch safe-check に利用。
        //   参照: DeferredDeletionQueue::enqueue(ptr, deleter, epoch, type, pubSeqId, generation)
    // 定期 drain（Timer/CoordinatorLoop の tryReclaim 直後）: epoch < minReaderEpoch になった
    // エントリのみ deleter 実行。それ以外は保持継続。
    void drain(uint64_t minReaderEpoch,
               const std::function<bool(uint64_t, uint64_t)>& isOlderFn);
    // Shutdown 専用: 全強制解放（Audio Thread 停止後 — destroyForShutdown と同契約）
    void drainAllUnsafe();
    [[nodiscard]] size_t residentCount() const noexcept;
 private:
    mutable std::mutex mtx_;
    // ★ 2026-08-08 三次レビュー追記: std::vector は noexec 保証下で allocation を引き起こすため
    // **std::array + 固定 capacity** に変更。kMaxQuarantinedEntries=512 を上限とし、
    // push_back は行わず index にて配置（allocation ゼロ）。store full 時は false を返す。
    static constexpr std::size_t kMaxQuarantinedEntries = 512;
    std::array<QuarantinedEntry, kMaxQuarantinedEntries> entries_;
    std::size_t size_ = 0;
};
```

**二次レビュー P1 補強（実装時の必須事項）:**

1. **epoch 比較は既存 `EpochDomain::isOlder(a, b)`（`EpochDomain.h:399-402`, `static_cast<int64_t>(a - b) < 0`）を使用**。`entry.epoch < minReaderEpoch` の単純比較はエポック wraparound 時に誤判定するため、`drain` 内は `if (isOlderFn(entry.epoch, minReaderEpoch))` で判定する（`isOlderFn` を `RetireQuarantineStore::drain` に注入 — Ch.2 の `quarantine()`/`drain()` シグネチャ参照）。`RetireQuarantineStore` は `EpochDomain` を include せず、`std::function<bool(uint64_t,uint64_t)>` として比較述語を注入するか、`EpochProvider` 経由で `isOlder` に委譲する（`ISRRetireRouter` が `EpochDomain` 完全型を隠蔽する既存パターン `ISRRetireRouter.h:48` を踏襲）。
2. **directDelete 禁止を厳守**: `drainAllUnsafe()` は Audio Thread 停止後に**のみ**呼ばれる。drain 中に RT が参照し得る場合（epoch 未達）は必ず保持継続。deleter の即時実行は本ストア内では一切行わない（`RetireQuarantineStore` は deleter を実行する場所であって、呼出し元で deleter を直接実行してはならない）。
3. **retire（論理退役）と physical delete（物理解放）の責務分離を再確認**: `DeferredDeletionQueue`（EBR 安全削除・先頭から順次解放）と `RetireQuarantineStore`（queue full 時の退避 + 定期 drain）は**別物**。前者は「正常経路の EBR」、後者は「退避経路の EBR」であり、両者が同じ epoch 判定（`isOlder`）を使うことで整合する。retire 直後の deleter 実行は両者とも行わない（EBR の原則）。entry は `DeletionEntry` と同等のフィールド（ptr/deleter/epoch/type/publicationSequenceId/generation）を保持し、退避 → drain → deleter call が既存 EBR パスと同一の epoch 判定を使う（INV-3）。
4. **`residentCount()` を既存 backpressure テレメトリに統合**（`AudioEngine.Timer.cpp:1133` の quarantine 表示）— 滞留が増えるケースの監視を維持。

**三次レビュー必須追加 — capacity exhaustion policy（2026-08-08）:**

`RetireQuarantineStore` は「delete できないものを安全に保持する」ストアであり、「delete できないものを別の場所に押し付ける」ストアではない。しかし無制限に増加させるとメモリ unbounded になるため、**第2段 overflow（backpressure）方針**を必須とする。

```
DeferredDeletionQueue
       ↓ full
RetireQuarantineStore          (kMaxQuarantinedEntries, 例: 512)
       ↓ high watermark
HealthEvent → shutdown escalation
```

- **容量は固定（allocation-free)**（`kMaxQuarantinedEntries = 512`。RT 参照中オブジェクトは通常 100ms オーダーで解放されるため、過剰な backlog は異常系）。**`std::vector::push_back` は `noexcept` 違反を引き起こす**（allocation 失敗 → `std::bad_alloc` → terminate）。`std::array<QuarantinedEntry, 512>` + index 配置により **allocation ゼロ** を保証する。store full 時は push を拒否し false を返す（deleter は caller が管理）。
- **Store full 時に `delete` は絶対しない**（三次レビュー: "quarantine store full時に delete は絶対にしない"）。候補は (a) 容量を十分大きく固定、(b) allocation failure 時は shutdown/health escalation、(c) 別の永続退避領域、(d) fatal diagnostic + controlled shutdown。本設計では **(b) HealthEvent + controlled shutdown** を採用。
- **high watermark 監視**: `residentCount() >= kMaxQuarantinedEntries`（または滞留時間が閾値を超過）で `ISRHealthState` を Degraded → Critical へ移行させ、`PublicationAdmission`（既存 `setHealthStateRef`）で publish を抑制しつつ、`RuntimeHealthMonitor` のヘルスイベントとして報告（`ISRRuntimeSemanticSchema.h` の health 系と統合）。
- **backpressure の継承**: `ISRRetireRouter` の既存 overflow 構造（`ISRRetire.cpp:50-75` の `overflowRing_` push 失敗 → `overflowCount_++` + `onHealthEvent`）と同じ方針を踏襲。`RetireQuarantineStore` は「queue full → 退避」の最終受け皿であり、その上（store full）は **health escalation** のみ。
- **アサート**: `quarantine()` が store full で拒否された場合、`jassert` で異常を検出 + `RuntimeHealthMonitor` に `Critical` を通知（retire リークは EBR 破綻 = 即停止すべき）。

```cpp
// 三次レビュー追記 — capacity policy 反映
static constexpr std::size_t kMaxQuarantinedEntries = 512;
std::size_t residentCount() const noexcept;   // 既存
bool quarantine(void* ptr, void (*deleter)(void*), uint64_t epoch,
                DeletionEntryType type, const char* reason,
                uint64_t publicationSequenceId = 0, uint64_t generation = 0) noexcept;
    // 戻り値 false = store full（呼出し元は deleter を実行してはならない。
    //   type/generation/publicationSequenceId を保持し、drain 時の epoch safe-check に利用。
    //   本設計では quarantine() が false を返さないよう health escalation が先行する）
```

**適用（Category A — SnapshotCoordinator。五次レビュー §5: `m_retireQuarantine` 直接保持は撤回、Router API 経由で移送）:**
```cpp
// SnapshotCoordinator.cpp:38 (startFade) — 返り値チェック + Router 経由で退避ストア移送
const auto result = enqueueWithRetry(*m_epochProvider, oldTarget, snapshotDeleter, retireEpoch);
if (!result) {
    // 五次レビュー: SnapshotCoordinator は退避ストアを直接保持しない。
    //   provider（= ISRRetireRouter）経由で Router 内部の RetireQuarantineStore へ移送する。
    //   接続点: IEpochProvider に退避委譲 API（quarantineRetire）を追加するか、
    //   SnapshotCoordinator に Router 参照（m_epochProvider を ISRRetireRouter* へ拡張）を注入する。
    //   type/generation/publicationSequenceId は SnapshotCoordinator が保持していないため 0 を渡す（Snapshot の世代は epoch で判定）。
    m_epochProvider->quarantineRetire(oldTarget, snapshotDeleter, retireEpoch,
        DeletionEntryType::Generic, "startFade:queueFull");
    // 将来: RuntimeHealthMonitor へイベント通知（TODO 解決）
}
```
```cpp
// SnapshotCoordinator.h:88,90 (switchImmediate) 同様にチェックを追加
if (!enqueueWithRetry(*m_epochProvider, oldSnap, snapshotDeleter, newEpoch))
    m_epochProvider->quarantineRetire(oldSnap, snapshotDeleter, newEpoch,
        DeletionEntryType::Generic, "switchImmediate:queueFull");
```

**適用（Category B — ISRRetireRouter / DSPLifetimeManager。五次レビュー §5: Router 内部ストアへ一本化）:**
```cpp
// ISRRetireRouter.cpp:154 — TODO → Router 内部の退避ストアへ移送（directDelete しない）
const auto enq = enqueueWithRetry(provider, snap, deleter, retireEpoch);
if (enq != RetireEnqueueResult::Success) {
    // type/generation/publicationSequenceId を退避エントリに引き継ぐ（DeferredDeletionQueue と同型）
    m_retireQuarantine->quarantine(snap, deleter, retireEpoch, DeletionEntryType::Generic,
        "enqueueWithRetry:QueuePressure", /*publicationSequenceId=*/0, /*generation=*/currentRetiringGeneration_);
    // 五次レビュー: RetireQuarantineStore は Router 内部メンバ（m_retireQuarantine）として
    //   単一配置。外部クラスから直接触らず、Router API（quarantineRetire）経由で委譲する。
}
```
```cpp
// DSPLifetimeManager.cpp:49,90 — ignoreUnused を削除して Router API 経由で退避へ
auto result = router_.enqueueWithRetry(ptr, deleter, epoch, DeletionEntryType::Generic,
    /*publicationSequenceId=*/0, /*generation=*/currentRetiringGeneration_);
if (result != convo::isr::RetireEnqueueResult::Success)
    router_.quarantineRetire(ptr, deleter, epoch, DeletionEntryType::Generic,
        "DSPLifetimeManager:QueuePressure", /*generation=*/currentRetiringGeneration_);
```

**drain タイミング:** `RetireQuarantineStore::drain(minReaderEpoch, isOlderFn)` を既存の定期 drain 点（`AudioEngine.Timer.cpp` の tryReclaim 呼出し直後、および `ISRRetireRouter::tryReclaim` 内）に 1 行追加。`isOlderFn` は `EpochDomain::isOlder` を委譲（二次レビュー P1 補強①参照 — `EpochDomain::isOlder(a, b)` = `static_cast<int64_t>(a - b) < 0`、wraparound 対応）。shutdown 時は `AudioEngine.Processing.ReleaseResources.cpp` の drain セクション（`m_retireRouter->unquarantineAllReaders()` 付近）で `drainAllUnsafe()` を実行。

**三次レビュー必須追記① — retire 順序の逆転（DSPLifetimeManager::retire 実測 2026-08-08）:**

`DSPLifetimeManager::retire()`（:38-53）の実行順序を実測すると **reclaim が enqueueWithRetry より先に実行されている**:

```cpp
// DSPLifetimeManager.cpp:38-53（実測）— retire 順序に逆転あり
retireDSPHandleForRuntime(handle);   // ① map 削除 + retired 遷移 + reclaim 実行（AudioEngine.h:4156-4159）
enqueueWithRetry(...);               // ② deferred delete（① の後）
juce::ignoreUnused(result);          // ③ 失敗を握り潰し
```

- `retireDSPHandleForRuntime`（AudioEngine.h:4156-4159）は `dspHandleRuntime_.shutdownReclaim(handle)` を呼ぶ。`ISRDSPHandle.h:158` の `shutdownReclaim` は **`reclaim(handle)` の単なるエイリアス**であり、正規パス `Coordinator::requestReclaim`（ISRRuntimePublicationCoordinator.cpp:581、DELETE-2/3 の executeRetire → waitReaders → executeReclaim 順序）を**バイパス**している。
- **重大: reclaim が enqueue 成功より先に実行される**。`enqueueWithRetry` が queue full で失敗した場合、DSP は **Reclaimed 済み**で deferred delete に入らない = **リーク**（Reclaimed なので RT の `resolve()` は nullptr を返す = UAF ではないが、DSPCore* は二度と解放されない）。
- **修正方針**: ① retire（論理退役 + map 削除）→ ② enqueueWithRetry（deferred delete）→ ③ 成功時のみ reclaim（または `Coordinator::requestReclaim` 経由の正常パス）。**enqueue 失敗時は RetireQuarantineStore へ移送し、epoch 安全確認後に deleter 実行**（本設計の EBR 原則と整合）。`shutdownReclaim` の transitional 措置は **削除**（`requestReclaim` に一本化。DELETE-1 の本来意図へ回帰）。
- `RetireEnqueueResult`（`ISRAuthorityClass.h:25-30`）は **`Success / QueuePressure / QueueFull / Shutdown` が既に存在**する。三次レビュー推奨の `tryEnqueue()` 返り値（`{Stored, StoreFull, InvalidEpoch, Shutdown}`）とほぼ同型のため、**新設 enum は不要 — 既存を再利用**する。Category B の `retire` 失敗判定は `result != RetireEnqueueResult::Success` で統一（`retire` の epoch 拒否は既に enqueue 側で処理済み — `Shutdown` は受入不可、`QueuePressure`/`QueueFull` は退避移送）。

**重要（directDelete の禁止）:** 本設計は **いかなる失敗経路でも deleter を即時実行しない**。deleter は `RetireQuarantineStore::drain`（safe-epoch 到達確認後）か `drainAllUnsafe`（Audio Thread 停止後）でのみ実行される。

### 検証
- **四次実測照合（2026-08-08）:** 本改修の根拠を再実測。
  - `RetireEnqueueResult`（`Success/QueuePressure/QueueFull/Shutdown`）は既存（`ISRAuthorityClass.h:25-30`）。
  - 既存呼出しの戻り値利用状況: `SnapshotCoordinator.h:88` は `if (!result)` で捕捉＋TODO、`:100` は**未捕捉**／`DSPLifetimeManager.cpp:49-53` は `juce::ignoreUnused(result)`／`ISRRetireRouter.cpp:154-157` は捕捉するが TODO コメントのみ。
  - **reclaim→enqueue 逆転を再確認**: `retireDSPHandleForRuntime`（`AudioEngine.h:4141-4163`）が `retire(:4155)`→`shutdownReclaim(:4159)` を**先に**実行し、呼び出し元 `DSPLifetimeManager::retire`（`DSPLifetimeManager.cpp:33-58`）で `enqueueWithRetry(:49)` が後続する。enqueue 失敗時は Reclaimed のまま deferred delete に入らずリーク。
  - `enqueueWithRetry`（`ISRRetireRouter.cpp:161-186`）: 初期 enqueue → `tryReclaim` → 再 enqueue（最大 2 回）→ 全失敗時 `QueuePressure` を返す実装を確認。
- 単体: `RetireQuarantineStore` の「queue full を強制 → quarantine → epoch 前進 → drain で deleter 実行」を確認
- 単体: 「queue full 中に RT が参照継続 → drain が解放しない → RT 離脱後に解放」を TSan で確認（UAF なし）
- **単体（三次レビュー追記）: store full 強制 → deleter が一切実行されない + HealthEvent 発火 + `ISRHealthState` 遷移を確認**
- **単体（三次レビュー追記）: `DSPLifetimeManager::retire` が enqueue 失敗時に退避移送し、Reclaimed 済みでリークしないこと（reclaim→enqueue 逆転が存在しないこと）を確認**
- ソーク: `SoakPublishIntegrationTests` で retire 経路のリークゼロを確認
- shutdown 時: `drainAllUnsafe` が Audio Thread 停止後にのみ呼ばれる契約を診断ログで確認

### リスク
中。`RetireQuarantineStore` が新規コード（~60 行 + 接続点 4 箇所）。ただし既存の `DSPQuarantineManager`/`DeferredFreeThread` と同パターンの NonRT mutex 保護なので RT 影響なし。`directDelete` 案より安全（UAF が構造的に排除される）が、退避ストアの滞留が増えるケース（queue full が続く）は監視が必要 — `residentCount()` を既存の backpressure テレメトリ（`AudioEngine.Timer.cpp:1133` の quarantine 表示）に統合する。**三次レビュー: store full → delete は絶対禁止。capacity 枯渇は HealthEvent → controlled shutdown へ**（capacity exhaustion policy 追記参照）。

---

## 3. BUG-028: CrossfadeRuntime リセット（残り）— 二次レビュー NO-GO → LinearRamp の RT/NonRT ownership 確定

**状態:** 🟡 実装前（五次レビュー §8 訂正: complete() は stale フラグ reset する — これ自体は RT 安全。**未実装**: `start()` の `gain_.setCurrentAndTargetValue(0.0)` を削除（RT データレース） + `complete()` に `dryScaleTarget_/startDelayBlocks_/dryHoldSamples_` の atomic publish を追加。`dryScaleGain_` は RT 専有に確定）
**対象:** `src/audioengine/CrossfadeRuntime.h:38-51, 95-103, 143-144` / `src/DspNumericPolicy.h:312-345` / `src/audioengine/AudioEngine.h:3883, 3917-3920` / `src/audioengine/AudioEngine.Init.cpp:19-22`

### 現状（実測 2026-08-08 — 二次レビューで再検証）

`dryScaleGain_` / `gain_` は `convo::LinearRamp`（**非 atomic**、4 フィールド構造体）。アクセス主体は以下の通り:

| メソッド | スレッド規約（`src/DspNumericPolicy.h:311-315`） |
|---|---|
| `reset()` | **NonRT 専用**（prepareToPlay 等） |
| `setCurrentAndTargetValue()` | **NonRT 専用**（ASSERT_NON_RT_THREAD） |
| `setTargetValue()` / `getNextValue()` / `skip()` | **RT 専用**（ASSERT_AUDIO_THREAD） |
| `applyImmediateValueRT()` | **RT 専用**（世代カウンタ同期保証時のみ） |

**アクセス実測:**
- **RT**: `AudioEngine.Processing.AudioBlock.cpp:442` / `BlockDouble.cpp:421` が `getDryScaleGain().getNextValue()` を**毎サンプル**呼ぶ。
- **RT**: `AudioEngine.h:3883`（activate）が `getDryScaleGain().setTargetValue(prepared.dryScaleTarget)` を呼ぶ。
- **RT**: `AudioEngine.h:3917-3920`（`finalizeCrossfadeMixPath`）が `dryScaleGain_.current/target/step/remaining` を**直接書き換え**。
- **NonRT**: `AudioEngine.Init.cpp:19-22` が `gain_.reset(48000.0, 0.03)` + `gain_.setCurrentAndTargetValue(1.0)` + `dryScaleGain_.reset(48000.0, 0.060)` + `dryScaleGain_.setCurrentAndTargetValue(1.0)`。
- **NonRT**: `CrossfadeRuntime::reset()`（:117-118）が `gain_`/`dryScaleGain_` に `setCurrentAndTargetValue(1.0)`。

**旧案の欠陥（二次レビュー指摘）:** REPAIR_PLAN3.md:310-322 の案は `start()`/`complete()` 内で `dryScaleGain_.setCurrentAndTargetValue(1.0)` を呼ぶもの。だが `start()` は publish 完了時（RT が `getNextValue()` を実行中）に呼ばれ得る **NonRT** スレッド（CoordinatorLoop/Builder）から発火する。**LinearRamp は非 atomic** なので、RT が `current`/`step`/`remaining` を読んでいる最中に NonRT が書き換えると**データレース（UB）**。`setCurrentAndTargetValue` の `ASSERT_NON_RT_THREAD` は「呼び出しスレッド」しか検証せず、「RT が同時に読んでいないこと」は保証しない。**旧案は新規のデータレースを導入する = NO-GO。**

### 改修案（詳細設計）: RT 専有（RT-only ownership）に確定

LinearRamp の操作は **RT 専有**とする。NonRT は LinearRamp に一切触れず、**atomic な `dryScaleTarget_`（既存）のみ publish** する。リセットは **RT パスの `finalizeCrossfadeMixPath` 内**で行う（既存 `AudioEngine.h:3917-3920` が既に RT で直接リセットしている — これを正規経路として確立）。

```cpp
// === CrossfadeRuntime.h ===
// start(): dryScaleGain_ は触らない（RT 専有）。gain_ の setCurrentAndTargetValue は
//   RT が getGain().getNextValue() を読む可能性があるため **データレース** — start() は
//   gain_.reset() のみ（NonRT、Audio Thread 停止前または publish 完了直後）し、
//   setCurrentAndTargetValue(0.0) は RT 側の armCrossfadeIfPending (AudioEngine.h:3878) で
//   setTargetValue(1.0) として代替する（RT 専用操作）。dryScaleTarget_ の publish は start() ではなく
//   complete() で行う（fade 終了後に dryScale を 1.0 リセット）。
void start(double fadeTimeSec, double sampleRate) noexcept
{
    gain_.reset(sampleRate, std::max(0.001, fadeTimeSec));
    // ★ BUG-028 fix: setCurrentAndTargetValue(0.0) を削除。gain_.reset() → dryScaleGain_.setCurrentAndTargetValue(1.0) も
    //   complete()（h:118）で NonRT から呼ばれていたため削除（RT data race）。gain_.reset() は
    //   NonRT で totalSteps のみ設定（current/target/step/remaining は未変更）。ただし reset() 自体も
    //   RT が isSmoothing()/getNextValue() を読む可能性があるため、将来は RT 停止後または RT パスの
    //   armCrossflareIfPending 内で行う（applyImmediateValueRT への移行を前提）。
    ...既存の atomic publish 群...         // pending_/queuedFadeTimeSec_/useDryAsOld_/firstIrDryPending_/startDelayBlocks_/dryHoldSamples_/fadeStartTimestampUs_
}

// complete(): stale フラグ reset は **維持**（実測 :95-103 一致 — atomic publish なので RT 安全）。
//   ★ 五次レビュー §8 訂正（2026-08-08 実測による）: 実コード complete() は
//   useDryAsOld_/firstIrDryPending_/firstIrDryDone_/queuedFadeTimeSec_/fadeStartTimestampUs_ を
//   **reset している**。これ自体は atomic publish で RT 安全だが、五次レビュー §8 指示の
//   「stale フラグ reset 禁止」は **実コードを誤解**（reset は安全だが dryScaleTarget_/startDelayBlocks_/
//   dryHoldSamples_ が欠落）。実際の危険は dryScaleGain_（LinearRamp）への NonRT 書き込みであり、
//   現コードは complete()（h:118）で dryScaleGain_.setCurrentAndTargetValue(1.0) を呼び、start() で gain_.setCurrentAndTargetValue(0.0)（h:41）を呼ぶ — いずれも NonRT→LinearRamp data race（二次レビュー追加発見）。
//   呼んでおり、gain_ も LinearRamp のため NonRT→RT data race となる（BUG-028 根因）。
//   改修案: complete() は stale フラグ reset を維持 + dryScaleTarget_/startDelayBlocks_/dryHoldSamples_
//   の publish を追加（:95-103 は dryScaleTarget_/startDelayBlocks_/dryHoldSamples_ を publish しない未実装）。
void complete() noexcept
{
    convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);   // 【追加】未実装
    convo::publishAtomic(startDelayBlocks_, 0, std::memory_order_release);    // 【追加】未実装
    convo::publishAtomic(dryHoldSamples_, 0, std::memory_order_release);      // 【追加】未実装
    ...既存の atomic publish 群...                                              // pending_/useDryAsOld_/firstIrDryPending_/firstIrDryDone_/queuedFadeTimeSec_/fadeStartTimestampUs_ reset (atomic, RT 安全)
    // ★ 五次レビュー §8: dryScaleGain_.setCurrentAndTargetValue(1.0) は追加しない。
    //   順序 invariant: complete()（NonRT, atomic batch publish）→ AudioThread が次回
    //   getDryScaleGain().getNextValue()（RT）で状態を消費。reset は atomic で原子的。
}

// reset(): shutdown 時（RT 完全停止後）のみ LinearRamp を直接操作してよい。
//   現状の :117-118 は保持（Audio Thread 停止後の呼び出しが保証されているため）。
```

**RT パス（正規リセット経路）:**
```cpp
// AudioEngine.h:3917-3920（finalizeCrossfadeMixPath）— 既存の直接フィールド代入を維持。
//   これは RT 内の操作なので LinearRamp 契約上も正しい。applyImmediateValueRT(1.0) への
//   置換も可能だが、RT 内で完結しているため挙動は同等。保守的には現状維持。
crossfadeRuntime_.getDryScaleGain().current = 1.0;
crossfadeRuntime_.getDryScaleGain().target = 1.0;
crossfadeRuntime_.getDryScaleGain().step = 0.0;
crossfadeRuntime_.getDryScaleGain().remaining = 0;
```

**`AudioEngine.h:3883`（activate）**: 既に RT から `setTargetValue(prepared.dryScaleTarget)` を呼んでおり、これは正しい RT 専有操作。`prepared.dryScaleTarget` は NonRT の atomic `dryScaleTarget_` から取られる（`AudioEngine.h:2907,3008`）。**RT 専有ルールと整合**。

**`AudioEngine.Init.cpp:19-22`**: `gain_.reset(48000.0, 0.03)` + `setCurrentAndTargetValue(1.0)` + `dryScaleGain_.reset(48000.0, 0.060)` + `setCurrentAndTargetValue(1.0)` は **prepareToPlay 相当（RT 未起動）**。LinearRamp 契約上正当。`reset()` は `totalSteps` 確定に必須なので**削除しない**（RT 側は `setTargetValue` で使用）。

### 検証
- **四次実測照合（2026-08-08）:** `CrossfadeRuntime.h` を再実測。
  - `start()`（:38-51）は `gain_.reset(sampleRate, ...)`（h:40）+ `gain_.setCurrentAndTargetValue(0.0)`（h:41）を行い、`dryScaleGain_` には触れない。**しかし `gain_.reset()` も NonRT から `LinearRamp::totalSteps` を書き換えるため、RT data race の候補**（二次レビュー追加発見）。`start()` は `dryScaleTarget_` を **atomic publish しない**（未実装）。`gain_.setCurrentAndTargetValue(0.0)` は prepareToPlay 相当の NonRT 操作（:372）。
   - `complete()`（:95-103）は stale フラグ atomic reset のみ — **しかし h:118 で `dryScaleGain_.setCurrentAndTargetValue(1.0)` が呼ばれ NonRT→LinearRamp race（二次レビュー追加発見）**。**stale フラグ（`useDryAsOld_`/`firstIrDryPending_`/`firstIrDryDone_`/`queuedFadeTimeSec_`/`fadeStartTimestampUs_`）を atomic publish で reset する**（実測 2026-08-08） — これ自体は RT 安全。

  - `reset()`（:117-118）のみ `gain_`/`dryScaleGain_` に `setCurrentAndTargetValue(1.0)`（Audio Thread 停止後）。
  - RT パス: `AudioBlock.cpp:442`/`BlockDouble.cpp:421` が `getDryScaleGain().getNextValue()` を毎サンプル呼ぶ。
  - NonRT→RT 境界: `AudioEngine.h:3883` activate が `setTargetValue(prepared.dryScaleTarget)`、`AudioEngine.Init.cpp:19-22` が初期化（NonRT, Audio スレッド停止中）。`crossflareRuntime_.getGain().reset(48000.0, 0.03)` + `setCurrentAndTargetValue(1.0)` および `getDryScaleGain().reset(48000.0, 0.060)` + `setCurrentAndTargetValue(1.0)` — これらは `initialize()` (AudioEngine.Init.cpp:13) にて Audio スレッド未開始前のため RT セーフ。
- RT パス `AudioEngine.Processing.AudioBlock.cpp:442` / `BlockDouble.cpp:421` の `getDryScaleGain()` がフェードサイクル間で 1.0 に戻ることをソークで確認
- **TSan で `dryScaleGain_` への RT↔NonRT 同時アクセスがゼロであること**（旧案を適用すると必ず検出される）
- **五次レビュー §8（順序 invariant テスト）**: `start()` が `dryScaleTarget_` を atomic publish することをテストで固定。`complete()` は stale フラグを **atomic publish で reset する**（実測 :95-103 — `useDryAsOld_/firstIrDryPending_/firstIrDryDone_/queuedFadeTimeSec_/fadeStartTimestampUs_` を publish）。reset は atomic なので RT 安全だが、**`dryScaleGain_`（LinearRamp）への NonRT 書き込みは一切行わない**ことを固定。`complete()` 発火後、AudioThread（`finalizeCrossfadeMixPath` / `getDryScaleGain().getNextValue()`）が complete event を消費するまで状態が変わらない（complete() は atomic batch publish で原子的）ことを順序検証テストで固定。
- `rg "dryScaleGain_|getDryScaleGain|getDryScaleGain"` で NonRT 側の直接 LinearRamp 操作を列挙する。**実測: complete()（h:118）で `dryScaleGain_.setCurrentAndTargetValue(1.0)` が呼ばれているため、これを除外する**（BUG-028 fix）。NonRT からの LinearRamp 操作は `Init.cpp:19-22` および `CrossflareRuntime::reset()`（RT 停止後）のみ許容する。
- デバイス切替 × 連続フェードで stale gain による音量段差が出ないこと
- `rg "dryScaleGain_|getDryScaleGain"` で NonRT 側の直接操作が `Init.cpp`/`CrossfadeRuntime::reset`（RT 停止時）のみであることを確認

**五次レビュー §8 追加 — complete() の block-boundary semantic consistency (P1 semantic hardening):**
- `complete()` は stale フラグ（`pending_`/`useDryAsOld_`/`firstIrDryPending_`/`firstIrDryDone_`/`queuedFadeTimeSec_`/`fadeStartTimestampUs_`/`dryScaleTarget_`/`startDelayBlocks_`/`dryHoldSamples_`）を**個別の atomic publish**で reset する。各 atomic store は独立しており、** transactional snapshot（すべて同時に可視）は保証されない** — 理論上 NonRT が `pending_=false` を publish した直後に RT が観測し、`dryScaleTarget_=1.0` はまだ観測していない中間状態が存在する。
- **data race ではない**（各 atomic は release/acquire で整合）。しかし **logical state consistency（block-boundary invariant）** は保証されない。RT は `pending_=false` を見て「crossfade not pending」と判断するが、`dryScaleTarget_` がまだ 1.0 でない間、`getDryScaleGain().getNextValue()` が stale な dryScaleGain_ を読み得る。
- **解決案（P1 hardening）**: Crossfade state に `uint64_t crossfadeGeneration`（または既存 `CrossfadeId`）を追加し、`start()`/`complete()` の各 atomic publish に続いて `generation` を 1 increment して publish。RT は **block boundary**（fade cycle 開始/完了）で `generation` の変化を検知し、一貫した atomic batch を「consumed」とする — `generation` が変わった時点で RT は stale フラグ群の新旧をリセットできる。
  ```cpp
  // CrossfadeRuntime.h — P1 hardening (optional)
  void complete() noexcept {
      convo::publishAtomic(dryScaleTarget_, 1.0, std::memory_order_release);
      // ... existing stale flag resets ...
      convo::publishAtomic(crossfadeGeneration_, crossfadeGeneration_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
      // RT: getCrossfadeGeneration() が前回と異なれば stale フラグ群を再読み込み（block-boundary consistency）
  }
  ```
- **検証**: soak で `crossfadeGeneration` が予期せず 2 回変化する（中断なき reset）ケースの absence + TSan で `dryScaleGain_` への RT↔NonRT 同時アクセスゼロ + **block-boundary trace**で `complete()` publish 完了 → AudioThread 次 block で `getDryScaleGain().getNextValue()` 消費の順序が決定的であることを確認。

### リスク
**ConvolverProcessor の LinearRamp は独立（BUG-028 対象外に明言）:** `ConvolverProcessor.h:910` (`latencySmoother`)、`:935` (`crossfadeGain`)、`:945` (`mixSmoother`) — CrossflareRuntime の `gain_`/`dryScaleGain_` とは**別個の LinearRamp**。`ConvolverProcessor.Lifecycle.cpp:370-386` で NonRT（prepareToPlay）から `reset()`/`setCurrentAndTargetValue()` されるが、RT スレッドからは独自のアクセスパターン（`ConvolverProcessor.Runtime.cpp:281-601`）で管理される。INV-2（RT-only LinearRamp ownership）は CrossflareRuntime にのみ適用。**ConvolverProcessor の LinearRamp は、将来的な独立した RT-safety 検証が必要だが、BUG-028 の対象外**。

低（LinearRamp は触らない）。**旧案（dryScaleGain_.setCurrentAndTargetValue の NonRT 追加）を実装すると TSan が必ず失敗する**ため、本改訂（dryScaleGain_ 操作封じ + dryScaleTarget_ atomic publish のみ）は実装の前提条件。`start()`/`complete()` は dryScaleGain_ に一切触れず, publish するのは atomic な dryScaleTarget_/stale フラグのみ (RT 安全)。**`start()` の `gain_.setCurrentAndTargetValue(0.0)` を削除**（BUG-028 fix）: しかし `gain_.reset()`（h:40）も NonRT から `LinearRamp::totalSteps` を書き換える — RT が `isSmoothing()` を読み、`getNextValue()` で `step`/`remaining` を消費中に race する可能性がある。**`gain_.reset()` は RT 停止後または RT パスの `armCrossflareIfPending` 内で行わなければならない**。**追加発見（2026-08-08 二次レビュー）: `dryScaleGain_.setCurrentAndTargetValue(1.0)` が `complete()`（h:118）で呼ばれている — RT パス（`AudioBlock.cpp:442`, `BlockDouble.cpp:421`）が `getDryScaleGain().getNextValue()` を毎サンプル呼ぶため、**これも data race**。`complete()` は `dryScaleGain_` に一切触れず、atomic `dryScaleTarget_` publish のみ。順序 invariant（complete() atomic batch publish → AudioThread 次 block で `getDryScaleGain().getNextValue()` 消費）を soak trace で固定する。

---

## 4. FUTURE-3 残り: recovery-world build の Builder Loop 接続 — 二次レビュー NO-GO → build spec を値コピーで payload に内包

**状態:** 🟡 P0 再設計中 → 三次レビュー反映済み（snapshot lifetime を値コピーで構造的に解決。GO 条件付き）→ 四次実測（2026-08-08）で **IR data 供給元を確定**（`RuntimeBuilder::build()` は IR 実体を `transferIRStateFrom(engine.getConvolverProcessor())` で取得 = **Recovery semantic は「現在のユーザー構成を再構築」（レビュー指摘 A）**。後述）
**対象:** `src/audioengine/ISRRuntimePublicationCoordinator.cpp:646-668`（pop 側）、`src/audioengine/ISRDSPHandle.cpp:56-72`（resolve）、`src/audioengine/RuntimeBuildTypes.h:48-66`（RuntimeBuildSnapshot POD）、`src/audioengine/RuntimeBuilder.h:136,146,192`（sealedSnapshot）、`src/audioengine/RuntimeBuilder.cpp:428-469`（`build()` — IR data 供給元）、`src/ConvolverProcessor.h:1130-1147`（`transferIRStateFrom`）、Builder Loop（`AudioEngine.RebuildDispatch.cpp`）、`src/audioengine/AudioEngine.h:2546`（runtimeBuildSnapshot 保持）

### 現状（実測 2026-08-08 — 二次レビューで再検証）

- `submitRecoveryRequest()` は定義が `ISRRuntimePublicationCoordinator.cpp:648`、`recoveryIntentQueue_.push(intent)`（transport-only enqueue）は `:660`。payload は `{ quarantinedHandle, currentEpoch, id }` のみ（`ISRRuntimePublicationCoordinator.cpp:652-658`）。
- `popRecoveryRequest()`（定義 `:665-668`）は実装済み。
- **Builder Loop 側の消費が未着手**: `popRecoveryRequest()` の結果を Builder へ渡し、Recovery RuntimeWorld を build → validate → publish する経路が未接続。

**二次レビュー指摘（核心）:** `DSPHandleRuntime::resolve()` は **Quarantined state を `{nullptr, false, false}` で拒否**する（`ISRDSPHandle.cpp:69`）:

```cpp
if (state == DSPState::Reclaimed || state == DSPState::Quarantined) {
    return { nullptr, false, false };
}
```

すなわち **quarantinedHandle から DSPCore* を解決できない**。旧案（第4章）の `isQuarantined(qHandle)` → build は、**Recovery World の build 入力（DSP spec / IR データ）をどこから取得するか未定義**のままだった。Recovery の本質は「問題 DSP を検出 → 新しい World を build し直す」ことだが、build には quarantined DSP の**元の構成情報**（または除外情報）が必要。

**Recovery build の入力（実測による引当）:**
- build の正規入力は `RuntimeBuildSnapshot* sealedSnapshot` + `spec`（`RuntimeBuilder.h:136,146,192`）。`sealedSnapshot` は Orchestrator が現在 World から収集する（`:37`）。
- Recovery の意味論は「quarantined DSP を除外（または差し替え）して再 build」。**quarantined DSP 自体の spec は通常の build 経路が持つものと同じ**（新 World は既存 World の spec 集合 + quarantined 除外で表現される）。

### 四次実測追記（2026-08-08）: IR data 供給元の確定 — Recovery build の semantic は「現在のユーザー構成を再構築」（レビュー指摘 A）

**実測の核心:** `RuntimeBuildSnapshot` は **IR AudioBuffer を所有しない**（metadata/fingerprint のみ）。`RuntimeBuilder::build()`（`RuntimeBuilder.cpp:428-469`）の実装は、IR 実体を **現在の processor** からコピーする:

```cpp
// RuntimeBuilder.cpp:443-447 — recovery/rebuild 用 DSPCore の生成
runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
runtime->convolverRt().setVisualizationEnabled(false);
runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
// Transfer actual IR data (applyBuildSnapshot only copies metadata, not the AudioBuffer)
runtime->convolverRt().transferIRStateFrom(engine.getConvolverProcessor());
```

- `applyBuildSnapshot`（`ConvolverProcessor.h:477`, `.cpp:271`）は **metadata のみ**コピー（`BuildSnapshot` 構造体 — `ConvolverProcessor.h:68-103` に IR AudioBuffer は含まれない。`irFile`/`irName`/`irLength`/`currentIRScale` 等の表示・復元用メタデータのみ）。
- `transferIRStateFrom(engine.getConvolverProcessor())`（`ConvolverProcessor.h:1130-1147`）が **現在の `uiConvolverProcessor`**（`AudioEngine.h:1219`）の `IRState` から AudioBuffer をコピーする。
- `RuntimeBuildSnapshot` の `rebuildFingerprint.irIdentityHash`（= structuralHash）と `convolverFingerprint` は **同一性検証用**（`isRuntimeBuildSnapshotSealedAndCompatible`、`RuntimeBuildTypes.h:309-337`）であり、IR data の実体ではない。

**semantic の帰結（レビュー指摘 A = 実装の実態）:**
1. Recovery build は quarantined DSP の**過去の IR 実体を復元できない** — snapshot に IR AudioBuffer が存在しないため。build 時点の**現在のユーザー構成**（現在の `uiConvolverProcessor` の IR + snapshot の metadata/fingerprint）を再構築する。
2. `buildSource`（sealedSnapshot 値コピー）は「どの構成で build すべきか」の metadata を運ぶ役割であり、IR data の供給元ではない。IR は常に `engine.getConvolverProcessor()`（現在の UI processor）から取られる。
3. したがって第4章の設計（`buildSource` = `RuntimeBuildSnapshot` 値コピー）は **metadata/fingerprint の輸送としては正しい**が、**「Recovery は quarantined DSP の構成を復元する」という文脈での過大表明はしない**。Recovery World は「現在のユーザー構成を再 build + quarantined 除外」という semantic に確定する。**セマンティックの明確化（2026-08-08 追加）: この semantic は A (Rebuild from current config) であり、B (snapshot at quarantine time) ではない。`buildSource` は quarantined DSP の過去の IR 実体を復元するためのものではなく、**quarantined DSP を除外した現在の configuration を指す metadata**である。IR data は `RuntimeBuilder::build()` が `transferIRStateFrom(engine.getConvolverProcessor())` で現在の `uiConvolverProcessor` から取得するため、Recovery World の IR は**現在のユーザー構成の IR**となる（過去の IR ではない）。**

**`buildRuntimePublishWorld` 側（World 構築）も整合確認済み**（`RuntimeBuilder.cpp:179-426`）:
- `worldOwner->topology.runtimeUuid` は `spec.topology.activeDSP`（`current`）から導出（:237）— DSP 実体ポインタは `spec` 経由。
- `dspProjection` は sealedSnapshot の値（irLoaded/irFinalized/structuralHash/oversamplingFactor/sampleRate/baseLatencySamples）を写像（:256-262）— metadata のみ。
- 実際の DSP 処理は `build()` で生成された DSPCore（IR は現在の processor から転送済み）が担う。**World は「どの DSPCore を公開するか」の authority であり、IR data を所有しない**点で一貫している。

### 改修案（詳細設計）: `RecoveryIntent` を「build spec 引当可能」にする — 三次レビュー反映（snapshot lifetime の明文化）

**三次レビュー指摘（核心）:** `contextEpoch` だけでは過去 snapshot の存在が保証されない。`resolveSnapshotForEpoch(100)` は、その epoch の World が既に retire/reclaim 済みなら **何を返すか未定義**。epoch から過去 World を逆引きする設計は lifetime が保証されないため、**build input の lifetime を明示的に管理**する必要がある。裸ポインタの `RuntimeBuildSnapshot*` を Intent に入れるのは NO-GO。

**確定した設計（実測に基づく）:**

`RuntimeBuildSnapshot` は **POD 値**（`captureRuntimeBuildSnapshot`/`sealRuntimeBuildSnapshot` で生成、`BuildInput` を値で内包 — `AudioEngine.RebuildDispatch.cpp:83-146`, `RuntimeBuildTypes.h:48-66`）。trivially copyable であり、**RecoveryIntent に値コピーで埋め込める**。これにより:

1. **epoch 逆引きが不要** — `resolveSnapshotForEpoch` を新設しない。Recovery 発行時に quarantined DSP の **sealedSnapshot を値コピーして payload に載せる**（transport metadata と build input を分離 — 既存 publish 経路の `OwnerChannel` パターンと同じ発想）。
2. **lifetime 問題が構造的に消滅** — ポインタではなく値を持つため、過去 World が retire されても build 入力は無傷。
3. **四次実測追記（2026-08-08）— `buildSource` の意味を限定する**: `buildSource`（`RuntimeBuildSnapshot` 値コピー）は **metadata/fingerprint の輸送**であり、**IR AudioBuffer は含まれない**（前述）。IR 実体は `RuntimeBuilder::build()` が `transferIRStateFrom(engine.getConvolverProcessor())` で現在の `uiConvolverProcessor` から取得する（`RuntimeBuilder.cpp:447`, `ConvolverProcessor.h:1130-1147`）。**Recovery semantic = 現在のユーザー構成を再 build + quarantined 除外**（レビュー指摘 A）。

```cpp
// ISRRuntimePublicationCoordinator.h — RecoveryIntent の payload 拡張（三次レビュー反映 + 四次/五次確定）
//   quarantinedHandle だけでは resolve 不能（ISRDSPHandle.cpp:69）。build 入力は値コピーで保持。
struct RecoveryIntent {
    DSPHandle quarantinedHandle;          // 既存: 除外対象
    PublicationEpoch contextEpoch;        // 既存: 引当時の currentWorld エポック（診断・FIFO 用）
    uint64_t   id;                        // 既存
    // ★ 三次レビュー: sealedSnapshot を値コピーで内包（POD、trivially copyable）。
    //   Builder はこの値から Recovery World を build し、quarantinedHandle を spec から除外する。
    //   RuntimeBuildSnapshot が trivially copyable であることは static_assert で保証。
    convo::RuntimeBuildSnapshot buildSource;
    // ★ 五次レビュー（案 i 確定）: ConvolverProcessor::BuildSnapshot は内包しない。
    //   juce::File/juce::String（ConvolverProcessor.h:92-93）を含むため trivially copyable でなく、
    //   static_assert が成立しない。build() の第 2 入力（convolverBuildSnapshot）は build 時に
    //   現在の uiConvolverProcessor.captureBuildSnapshot() から取得する（RebuildDispatch.cpp:910-911 と同型）。
};
static_assert(std::is_trivially_copyable_v<convo::RuntimeBuildSnapshot>,
    "FUTURE-3: RuntimeBuildSnapshot must be trivially copyable to embed in RecoveryIntent");
```

> **四次実測注記（2026-08-08）:** `ConvolverProcessor::BuildSnapshot` は `juce::File irFile` / `juce::String irName` を含む（`ConvolverProcessor.h:92-93`）。`juce::File`/`juce::String` は **trivially copyable ではない**（参照カウント付き）。したがって上記の `static_assert(trivially_copyable)` は **`ConvolverProcessor::BuildSnapshot` では成立しない**。実装時は次のいずれかを採用する:
> - **案 i（推奨）: `ConvolverProcessor::BuildSnapshot` を `RecoveryIntent` に入れず、`lookupSealedSnapshot` の引当元（`DSPHandleRuntime` スロット / registry）で **`RuntimeBuildSnapshot` + `ConvolverProcessor::BuildSnapshot` のペア**を保持し、Recovery build はこのペアを参照する。ただし quarantined DSP の convolver は resolve 不能（`ISRDSPHandle.cpp:69`）のため、**現在の `uiConvolverProcessor.captureBuildSnapshot()` を build 時に取得**する（Recovery semantic = 現在のユーザー構成のため整合）。
> - **案 ii: `juce::File`/`juce::String` を `std::uint64_t` ハッシュ等に置換した POD 専用 `BuildSnapshotLite` を定義**し、`RecoveryIntent` に内包する（fingerprint 照合 + build に必要な数値 metadata のみ）。ただし `build()` の引数は `ConvolverProcessor::BuildSnapshot` そのもののため、`BuildSnapshotLite` → `BuildSnapshot` の再構築が必要。
> - **判定:** 実装コストと POD 性質維持のバランスから **案 i を採用**。`RecoveryIntent` は `buildSource`（`RuntimeBuildSnapshot`）のみを内包し、convolver metadata は **build 時に現在の `uiConvolverProcessor` から `captureBuildSnapshot()`** で取得する（RebuildDispatch.cpp:910-911 の既存呼び出しが `task.convolverBuildSnapshot` を渡しているのと同型）。これにより `static_assert(trivially_copyable)` は `RuntimeBuildSnapshot` のみで維持できる。

**`submitRecoveryRequest` の発行側（quarantine 検出 → Recovery 発行の接続）:**

現状 `submitRecoveryRequest` は呼出し元ゼロ（`RecoveryIntentHandler::handle` が no-op）。quarantine 検出時（`QuarantineIntentHandler` / `QuarantineService::executeQuarantine`）に、**quarantined DSP の build spec を引当してから** Recovery Intent を発行する。build spec の引当元は、quarantine 検出時点の `RuntimeBuildSnapshot`（DSP 登録時に capture/seal 済みのものを `DSPHandleRuntime` スロットか registry が保持）を値コピーする。

```cpp
// QuarantineService::executeQuarantine 完了後（Recovery 発行）
//   quarantined DSP の sealedSnapshot を引当 → 値コピーで payload に載せる
const convo::RuntimeBuildSnapshot snap = handleRuntime.lookupSealedSnapshot(qHandle); // 新規 API
runtimePublicationBridge_.submitRecoveryRequest(qHandle, currentEpoch, snap);
```

**Builder Loop 側の消費（`:RebuildDispatch.cpp`）:**
```cpp
if (runtimeOrchestrator_ != nullptr) {
    while (auto recovery = runtimeOrchestrator_->popRecoveryRequest()) {
        // BUILDER-STATE: BuildSession RAII 内で PendingMap 集約 → build → validate → publish
        const auto& qHandle = recovery->quarantinedHandle;
        // 1) Admission: 消費時点で quarantinedHandle の実在性・Quarantine 状態を検証（RECOVERY-6）
        if (!isQuarantined(qHandle))
            continue;  // 二重解放・無効ハンドルは無視
        // 2) build 入力は payload 内の buildSource（値コピー済み）— epoch 逆引き不要
        // 3) spec 集合から quarantinedHandle を除外 → Recovery World build
        // ★ 五次レビュー §15: currentSpecs() は「現在の UI state / RuntimeWorld / Pending Builder state」の
        //   いずれを指すか曖昧。Recovery build は Builder 側で **RecoveryBuildInput** として正規化し、
        //   Authoritative build specification + Recovery exclusion set にする方が安全。
        auto recoverySpec = normalizeRecoveryBuildInput(currentSpecs(), recovery->buildSource);
        recoverySpec.erase(qHandle);  // 除外（または差し替え）
        // 4) ★ 四次追記: build() は 2 入力。buildInput は buildSource.buildInput、
        //    convolverBuildSnapshot は build 時に現在の uiConvolverProcessor から
        //    captureBuildSnapshot() で取得（Recovery semantic = 現在のユーザー構成のため整合。
        //    RebuildDispatch.cpp:910-911 の既存呼び出しと同型）。
        //    BuildSession RAII 開始 → recovery world build（&recovery->buildSource + recoverySpec）
        // 5) PublicationValidator で validate（通常経路と同一・RECOVERY-2）
        // 6) submitPublishRequest / publishWorld（Immutable Publish）
    }
}
```

**設計方針（三次レビュー反映）:**
- 専用 Recovery Worker は設けない（既存 Builder Loop が消費）— 維持。
- **`RecoveryIntent` は buildSource（sealedSnapshot 値）を payload に内包**。`resolveSnapshotForEpoch` は**新設しない**（epoch 逆引きの lifetime 問題を構造的に回避）。resolve 不能な quarantinedHandle に依存しない。
- **`buildSource` は metadata/fingerprint の輸送であって IR data の輸送ではない（四次実測確定）**: IR 実体は `RuntimeBuilder::build()` が現在の `engine.getConvolverProcessor()` から `transferIRStateFrom` で取得。Recovery World の音響は「現在のユーザー構成 + quarantined 除外」で決まる。
- **`build()` の第 2 入力（`convolverBuildSnapshot`）は build 時に現在の `uiConvolverProcessor.captureBuildSnapshot()` から取得（四次追記）**: `ConvolverProcessor::BuildSnapshot` は `juce::File`/`juce::String` を含み trivially copyable でないため `RecoveryIntent` に内包しない（案 i 採用）。buildInput は `buildSource.buildInput`、convolver metadata は現在の UI processor から取得する — これは `RebuildDispatch.cpp:910-911`（rebuildThreadLoop が `task.convolverBuildSnapshot` を渡す）と同型のパターンで、Recovery semantic = 現在のユーザー構成と整合。
- **`OwnerChannel` パターン（B2）との整合**: 既存 publish は「transport metadata（`PublishPayload`）と build input lifetime（`OwnerChannel` + `PendingPublishRegistry`）を分離」。Recovery も同じ原則で「transport metadata（quarantinedHandle + epoch）と build input（sealedSnapshot 値）を分離」。ただし sealedSnapshot は POD 値なので **lifetime 管理用チャネルは不要**（値コピーが lifetime-safe）。
- PendingMap は BuildSession RAII に閉じ込め、全終了経路で破棄保証（BUILDER-STATE）— 維持。
- **RECOVERY-5 の「quarantinedHandle のみを payload」は撤回**。quarantinedHandle は「除外対象の識別子」であり、build 入力ではない。build 入力は payload 内 sealedSnapshot。

**INV-4: Recovery semantic — formalize (五次レビュー §12):**
> **RECOVERY-SEMANTIC-001**
> Recovery は過去 RuntimeWorld の復元（rollback）ではない。
> Recovery は、quarantined component を除外した**現在の authoritative configuration**から新しい RuntimeWorld を再構築する（reconstruction）。
>
> `buildSource`（`RuntimeBuildSnapshot` 値コピー）は **historical world snapshot** ではなく、**Recovery admission 晇点で確定した build metadata/fingerprint** である。IR 実体は snapshot に内包されず、`RuntimeBuilder::build()` が `transferIRStateFrom(engine.getConvolverProcessor())` で**現在の `uiConvolverProcessor`** から取得する。

### 検証
- **四次実測照合（2026-08-08）:** `RecoveryIntent`/`submitRecoveryRequest`/`popRecoveryRequest` を再実測。
  - `RecoveryIntent`（`ISRRuntimePublicationCoordinator.h:151-159`）は `trivially_copyable` + `standard_layout` を static_assert で保証（LockFreeRingBuffer 転送可能）。
  - `submitRecoveryRequest`（`cpp:648-662`）は `recoveryIntentQueue_.push`（transport-only）のみ。`popRecoveryRequest`（`cpp:664-672`）は Builder 消費前提の pop。
  - `recoveryIntentQueue_`（`h:394`, `kRecoveryIntentQueueCapacity=256`）は専用キューとして存在。
  - **Producer/Consumer 未接続**: `submitRecoveryRequest` と `popRecoveryRequest` の**プロダクションコード上の呼び出し元が存在しない**（呼び出しは単体テスト `ISRSemanticValidationTests.cpp:595-596` のみ。Builder Loop への接続が未着手）＝「P0 再設計中」の現状と整合。
  - **IR data 供給元の実測**（2026-08-08）: `RuntimeBuilder::build()`（`RuntimeBuilder.cpp:443-447`）は `applyBuildSnapshot`（metadata のみ）→ `transferIRStateFrom(engine.getConvolverProcessor())`（現在の `uiConvolverProcessor` から AudioBuffer コピー）。`RuntimeBuildSnapshot`（`RuntimeBuildTypes.h:48-66`）に IR AudioBuffer は存在しない。`isRuntimeBuildSnapshotSealedAndCompatible`（:309-337）は fingerprint/metadata の比較のみ。
  - **`build()` の 2 入力の実測**（2026-08-08）: `RuntimeBuilder::build(const BuildInput&, const ConvolverProcessor::BuildSnapshot&)`（`RuntimeBuilder.cpp:428-429`）は buildInput と convolverBuildSnapshot の 2 入力。rebuildThreadLoop は `task.runtimeBuildSnapshot.buildInput` + `task.convolverBuildSnapshot` を渡す（`RebuildDispatch.cpp:910-911`）。`ConvolverProcessor::BuildSnapshot` は `juce::File`/`juce::String` を含む（`ConvolverProcessor.h:92-93`）ため trivially copyable でなく、`RecoveryIntent` には内包せず build 時に `uiConvolverProcessor.captureBuildSnapshot()` から取得（案 i）。
- `RecoveryIntentHandler` を経由した quarantine → recovery publish → 新 World 反映の一連を AudioEngineHarness の Integration Test で検証
- **Recovery build が quarantined DSP を正しく除外（spec 集合から erase）することを検証**
- **buildSource の値コピーが世代をまたいで有効であることを検証**（Recovery 発行後に過去 World を retire しても build が成立する — lifetime テスト）
- BUILDER-STATE の例外安全性（build 失敗時に PendingMap が残留しない）テスト
- **resolve が Quarantined を拒否しても build が成立することをテスト**（payload 引当の有効性）
- **IR data semantic の検証（四次追記）**: Recovery build 後の DSPCore が `transferIRStateFrom` 経由で現在の `uiConvolverProcessor` の IR を保持すること（fingerprint 比較 `irIdentityHash == 現在 structuralHash` で確認）。snapshot の `irIdentityHash` が現在 IR と不一致の場合、**Recovery は現在のユーザー構成で build される**（過去 IR 復元は semantic 外）ことを診断ログで確認。

### リスク
中。Recovery world build は既存 Builder 経路の拡張であり、**buildSource 引当 + spec 除外**を正しく実装しないと quarantined 済み DSP が再 publish される（二重 publish）。RECOVERY-6 の検証を必須とする。`lookupSealedSnapshot` は新規 API（`DSPHandleRuntime` に追加）— quarantined DSP の sealedSnapshot を保持できるよう、DSP 登録時に `RuntimeBuildSnapshot` をスロットへ格納する既存構造（`AudioEngine.h:2546` の `runtimeBuildSnapshot` フィールド等）との整合を確認してから実装。**三次レビュー: epoch 逆引き（`resolveSnapshotForEpoch`）は新設しない — snapshot の lifetime 問題を構造的に回避するため。**
**四次追記（IR data semantic のリスク）:** Recovery は「現在のユーザー構成」を build するため、quarantine 検出時点と build 時点で IR が変わっていた場合（ユーザーが IR を差し替えた等）、Recovery World は**最新 IR**で build される。これは semantic 指摘 A の帰結であり、**設計として受け入れる**（過去 IR 復元は snapshot にデータが無いため不可能）。fingerprint 不一致は診断ログで検出可能（`irIdentityHash` vs 現在 `structuralHash`）。**IR AudioBuffer を snapshot に載せる拡張は今回対象外**（メモリ増大・POD 性質喪失のため）— 必要なら将来 FUTURE として別途設計。**四次追記（convolver metadata のリスク）:** `build()` の第 2 入力（`convolverBuildSnapshot`）を現在の `uiConvolverProcessor.captureBuildSnapshot()` から取得する案 i は、**Recovery 消費時点の UI 状態**を反映する（quarantine 検出時点ではない）。buildInput（`buildSource.buildInput`）と convolver metadata（現在値）の間に**世代差**が生じ得るが、`finalizeRuntimeBuildSnapshot` が正規化（`sampleRate`/`blockSize` 等を buildInput から再導出）するため実害は限定的。fingerprint 照合（`isRuntimeBuildSnapshotSealedAndCompatible`）で乖離を検出し、診断ログに記録する。

---

## 5. FUTURE-9 残り: SHUTDOWN-7（`VerifyDrained` の rebuild worker チェック）— 二次レビュー NO-GO → shutdown 順序確定

**状態:** 🔴 P0 NO-GO（完了条件への組込みは冗長。先に shutdown 順序を確定）
**対象:** `src/audioengine/ISRShutdown.cpp`（VerifyDrained phase）, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:188-196,430`, `src/audioengine/AudioEngine.h:2530`（`rebuildThreadIsRunning`）

### 現状（実測 2026-08-08 — 二次レビューで再検証）

- REPAIR_PLAN.md が参照する **`rebuildWorkerRunning`（`RuntimeBuilder.cpp:284`）は `const bool rebuildWorkerRunning = false;` と常時 false のハードコード** = シャットダウン判定に使えない無意味フィールド。
- **実体は `rebuildThreadIsRunning`**（`AudioEngine.h:2530` `std::atomic<bool>`）。`AudioEngine.RebuildDispatch.cpp:799`（true, worker 起動）・`:1066`（false, worker 終了）で publish される。
- **VerifyDrained は既に `rebuildThreadIsRunning` をチェック済み**: `AudioEngine.Processing.ReleaseResources.cpp:439-440` で `else if (convo::consumeAtomic(rebuildThreadIsRunning, ...)) reason = ShutdownBlockingReason::ActiveBuilder;` — ただし **タイムアウト時（`timedOut`）の診断 reason 決定のみ**で、`waitForDrain` の**完了条件**には入っていない。

**二次レビュー指摘（核心）— shutdown 順序の実測:**
`stopRebuildThread()`（`rebuildThread.join()` を含む）は **`waitForDrain()` より前に**呼ばれている:

```
ReleaseResources.cpp:75   runtimePublicationBridge_.requestShutdown();   // 先に停止要求を発行
ReleaseResources.cpp:188  setShutdownPhase(StopWorkers)
ReleaseResources.cpp:189  shutdownCoordinatorLoop();     // ★ FUTURE-9: Coordinator Worker を join（先に停止）
ReleaseResources.cpp:190  stopRebuildThread();          // join() で Builder 完全終了（RebuildDispatch.cpp:781-782）
ReleaseResources.cpp:191  transitionTo(ObserverDrained)
...
ReleaseResources.cpp:430  waitForDrain(2000, 2);        // この時点で rebuildThreadIsRunning は必ず false
```

すなわち **waitForDrain 実行時には Builder・Coordinator 双方が join 済み**であり、`rebuildThreadIsRunning` を完了条件に追加しても**常に false で冗長**（条件として機能しない）。完了条件に加えるべきは Builder の実行状態ではなく、**シャットダウン順序そのもの**（停止要求 → CoordinatorLoop join → Builder join → waitForDrain）が守られていること。

### 改修案（詳細設計）: shutdown 順序の確定 + 完了条件は「フェーズゲート」で担保

```cpp
// AudioEngine.Processing.ReleaseResources.cpp — 順序の確定（既存実装を契約として明文化）
//
//   不変条件 SHUTDOWN-ORDER:
//     (1) requestShutdown() を先に呼ぶ（runtimePublicationBridge_.requestShutdown() :75）
//     (2) shutdownCoordinatorLoop() が Coordinator Worker を join する（:189 — FUTURE-9）
//     (3) stopRebuildThread() が rebuildThread.join() で Builder を完全終了させる（:190）
//     (4) その後にのみ waitForDrain() を呼ぶ（:430）
//   これにより waitForDrain の「Builder・Coordinator が動いていない」前提が構造的に保証される。
//   完了条件への rebuildThreadIsRunning 追加は行わない（join 済みで常に false — 冗長）。

// 防御的アサート（デバッグビルド）:
//   jassert(!convo::consumeAtomic(rebuildThreadIsRunning, std::memory_order_acquire));
//   「waitForDrain 時点で Builder は終了済み」を実行時に検証（SHUTDOWN-ORDER の契約破れ検出）。
```

**既存のタイムアウト時 reason 分岐（:437-440）は維持**（`markTimedOut(reason)` の ActiveBuilder 報告）。ただしこれは**タイムアウト診断**であり、正常系の完了条件ではない（join 済みのため理論上発生しないが、未来の順序変更・追加経路に備えた安全網として残す）。

**追加（完了条件ゲート）:** `waitForDrain` の先頭に `ASSERT_NON_RT_THREAD` + フェーズゲート（`AudioStopped` 以降）は既にある（`AudioEngine.Threading.cpp:140-150`）。SHUTDOWN-ORDER の (1)(2)(3)(4) はこのフェーズ遷移（`StopWorkers` → `ObserverDrained` → ... → `waitForDrain`）で間接的に担保されている。**コード変更は不要** — 設計書に順序契約を明文化し、既存実装を検証する。

**★★ 2026-08-08 三次レビュー追加 — 2つの ShutdownPhase enum が共存中**:
- `AudioEngine::ShutdownPhase` (AudioEngine.h:2479-2503): `Running/StopAcceptingWork/StopAudio/StopWorkers/ForceEpochAdvance/DrainRetire/Destroy` — AudioEngine 内部ライフサイクル
- `convo::isr::ShutdownPhase` (ISRShutdown.h:25-41): `Running/AudioStopped/ObserverDrained/RetireClosed/EpochSettled/ReclaimComplete/EmergencyDrain/VerifyDrained/TimedOut/Failed/ShutdownComplete` — ISR シャットダウンランタイム

CtorDtor.cpp:96-229 は `AudioEngine::ShutdownPhase` を `setShutdownPhase()` で set する。同時に `ReleaseResources.cpp:73-520` は `shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase)` も呼ぶ。**二つの enum は手動で対応づけられている**（StopAcceptingWork↔AudioStopped, StopWorkers↔ObserverDrained, など）が、**対応表がテストで固定されていない** — 将来 enum 追加時に不整合発生。`AudioEngine.shutdownPhase` と `shutdownRuntime_.phase` の同期を invariant テストで固定すべき。

### 検証
- **四次実測照合（2026-08-08）:** shutdown 順序の根拠コードを再実測。
  - `rebuildThreadIsRunning`（`AudioEngine.h:2530`, `std::atomic<bool>{false}`）が `RebuildDispatch.cpp:799`(true)/`:1066`(false) で publish されることを確認。
  - `rebuildWorkerRunning`（`RuntimeBuilder.cpp:284`）は `const bool rebuildWorkerRunning = false;` と**常時 false のハードコード**（判定に使えない）を確認。
  - `ReleaseResources.cpp` の順序: `:75 requestShutdown` → `:188 StopWorkers` → `:189 shutdownCoordinatorLoop` → `:190 stopRebuildThread` → `:191 ObserverDrained` → `:430 waitForDrain(2000,2)`。**waitForDrain 時点で Builder/Coordinator は join 済み**。
  - タイムアウト時 reason 分岐（`:437-440`）が `stuckReaderCount>0`→`ReaderActive`、`rebuildThreadIsRunning==true`→`ActiveBuilder` であることを確認。
- シャットダウン中に rebuild worker が publish を続けている状況を再現し、**`stopRebuildThread` が join で完全終了後に `waitForDrain` が始まる**ことを `shutdown_trace.json` で確認
- `rebuildThreadIsRunning` が `waitForDrain` 開始時点で必ず false であることを診断ログで確認
- タイムアウト時（2 秒超過）でも `markTimedOut(ActiveBuilder)` が既存通り動作すること
- フェーズ遷移 `StopWorkers → ObserverDrained → ... → waitForDrain` の順序が崩れた場合に jassert で検出されること

### リスク
低（コード変更はほぼ不要。順序契約の明文化 + 防御的 jassert）。`rebuildThreadIsRunning` への追加アクセスは行わないため、既存のシャットダウンタイムアウト（`waitForDrain(2000, 2)`）との相互作用はない。**`rebuildWorkerRunning`（常 false の旧フィールド）は本設計から除外**（誤って使うと常に false でチェックが無意味になる）。

---

## 6. FUTURE-10: 共通 Intent Queue 一本化（Observe の `intentQueue_` 統合。Recovery は Builder Work Queue 分離）

**状態:** ☐ 未着手（`intentQueue_` は存在するが Observe/Recovery は別キュー）
**対象:** `src/audioengine/ISRRuntimePublicationCoordinator.h:397-400` / `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:33-45` / `AudioEngine.Timer.cpp:896,1029,1568`

### 現状
- `intentQueue_`（`LockFreeRingBuffer<Intent, kIntentQueueCapacity>`）は存在し、Publish/Quarantine はここを通る。
- **Observe** は `observeIntentQueue_`/`observeFallbackQueue_`（専用 SPSC）のまま（`ProcessIntent.cpp:33-38`）。
- **Recovery** は `recoveryIntentQueue_`（専用）のまま（`ISRRuntimePublicationCoordinator.cpp:660`）。
- `ProcessIntent.cpp:40-45` のコメントで「Observe stays on its dedicated SPSC rings — DoD #4/#7 deferred to FUTURE-10」と明記。

### 実測調査（2026-08-08）— `intentQueue_` は既に **MPSC 実態**（SPSC リングで潜在競合）

`LockFreeRingBuffer` は **SPSC 専用**（`src/LockFreeRingBuffer.h:9` "SPSCロックフリーリングバッファ"、`pop()` 注記 "Do NOT rely on this pattern for multi-producer/consumer scenarios"）。しかし `intentQueue_` への push は **複数スレッド**から行われる:

| Producer スレッド | 経路 | 箇所 |
|---|---|---|
| Builder/Rebuild スレッド | `RebuildDispatch.cpp:1053` → `enqueuePublicationIntentForRuntimeCommit` → `submitPublishRequest` → orchestrator → `enqueuePublicationIntent`（`intentQueue_.push`） | `AudioEngine.h:4296` |
| Timer スレッド | `Timer.cpp:1788,1826` → `submitQuarantine` → `intentQueue_.push` | `ISRRuntimePublicationCoordinator.cpp:696` |
| CoordinatorLoop（deferred resubmit） | `enqueueRuntimePublicationFireAndForget` → `enqueuePublicationIntent` | `AudioEngine.h:4236` |
| その他 NonRT | `commitRuntimePublication`（Timer.cpp:918 / PrepareToPlay.cpp:155,277 / ReleaseResources.cpp:175 / Transition.cpp:25）→ fire-and-forget | — |

Consumer は CoordinatorLoop 単一。**SPSC 前提のリングを複数 Producer が同時 push** すると、`writeIndex` の更新が `readIndex` 観測と競合し、エントリ破損・要素消失のデータ競合になり得る。

**FUTURE-10 の正しい解釈:** 計画の「Observe の SPSC 専用リングを MPSC の `intentQueue_` に統合」は「既に MPSC 実態のキューへ、さらに Producer を増やす」ことを意味する。**`LockFreeRingBuffer` の MPSC 化（または Producer を CoordinatorLoop に集約する Single-Producer 化）が前提**となる。

### 改修案（詳細設計）— 三次レビュー反映: Intent Queue と Builder Work Queue の分離 + per-type admission policy

**二次レビュー指摘（核心）:** 第7章の案 A（RecoveryIntentHandler が `submitRecoveryRequest` を呼ぶ）を、旧第6章の「`submitRecoveryRequest()` を `intentQueue_.push(Intent{type=Recovery})` に変更」と組み合わせると、**無限循環**になる:

```
Recovery Intent → intentQueue_ → CoordinatorLoop(processIntent) → RecoveryIntentHandler
                → submitRecoveryRequest → intentQueue_.push(Recovery) → (再び CoordinatorLoop が pop) → ...
```

Recovery は「Builder Loop が build する作業」であり、**CoordinatorLoop が pop して Handler が再 enqueue する Dispatcher 経路に載せてはならない**。**Intent Queue（CoordinatorLoop が消費）と Builder Work Queue（Builder Loop が消費）を明確に分離**する。

**分離の原則:**
- **Intent Queue** (`intentQueue_`): CoordinatorLoop が pop → Dispatcher → Handler。**Publish / Quarantine / Observe** が対象。
- **Builder Work Queue** (`recoveryIntentQueue_`): **Builder Loop が pop** する専用キュー。**Recovery** のみが対象。CoordinatorLoop の Dispatcher を経由しない。

FUTURE-10 の最終形に合わせ、Observe も `Intent` 型（tagged-union payload）で `intentQueue_` に一本化する。**Recovery は統一しない**（Builder Work Queue のまま分離維持 — 循環回避のため）。

**三次レビュー必須修正① — 「bounded CAS retry → drop」を撤回（2026-08-08）:**

旧案の「CAS retry が `kMaxProducerRetries` を超えたら push を drop する」は **NO-GO**。Intent は telemetry ではなく **runtime state transition の運搬手段**であり、`Publish` を drop すると「World B build 完了 → キュー drop → World A が公開されたまま」となり、**runtime state transition そのものが失われる**。`Quarantine` を drop すると「bad DSP が使用可能なまま」となる（安全問題）。したがって:

- **Observe drop**（観測情報・後発の Observe で補完可能）と **Publish/Quarantine drop**（state transition そのもの）を同じ policy で扱ってはならない。
- **drop ではなく admission policy を採用**: per-intent-type の overflow policy を定義し、最終的にデータ喪失を避ける（`Practical Stable ISR Bridge` の原則 — overflow はデータ喪失に直結してはならない）。

| Intent | Queue full 時（三次レビュー確定 policy） |
|--------|------------------------------------------|
| **Publish** | **絶対 drop 禁止**。既存の `enqueuePublicationIntent` は false 返却時に呼出し元が `ownerChannel().take(key)` で所有権を回収（`AudioEngine.h:4296-4299`、take は `:4299`）→ **retry / deferred 経路**へ。既存 `PublicationAdmission::evaluateDeferred`（Deferred Publish View, `PublicationAdmission.h:64-102`）で再試行。 |
| **Quarantine** | **絶対 drop 禁止**。quarantine 検出は安全要件（bad DSP のアクセス禁止）。queue full 時は **dedicated fallback ring**（Observe と同型の `observeFallbackQueue_` 相当、または quarantine 専用 3 層）へ。それも full なら HealthEvent → `ISRHealthState::Critical`。 |
| **Recovery** | **drop 禁止。ただし coalesce 可能**（同一 quarantinedHandle の複数 Recovery は 1 件にマージ — 同一 DSP の再 build は重複無意味）。Builder Work Queue 側の overflow は HealthEvent。 |
| **Observe** | 条件付き drop / coalesce 可能。既存の 3 層 fallback（`observeIntentQueue_` → `observeFallbackQueue_` → `observeDeferredRing_` → drop + `onHealthEvent`）を維持。後発 Observe で補完可能なため、最後の drop は許容（`ISRRuntimePublicationCoordinator.cpp:551-575`）。 |
| **Diagnostic** | drop 可（存在する場合）。 |

**三次レビュー必須修正④ — MPSC ring は独立 bounded MPSC primitive として実装・検証（2026-08-08）:**

`LockFreeRingBuffer`（SPSC）の CAS 小改造は **避ける**。SPSC 前提のリングに MPSC を足すと、producer hole（Producer A が slot N 予約、Producer B が slot N+1 予約、Consumer が N+1 を読もうとした時 A が未書き込み）や `writeIndex`/`readIndex` の観測順序が破綻する。

```cpp
// 新設: MpscBoundedRing（既存 DeferredDeletionQueue と同型の Vyukov bounded アルゴリズム）
//   ★ 2026-08-08 実測: MpscBoundedRing は**未実装**（source tree に 0 件）。SPSC 前提の LockFreeRingBuffer.h:25-89 は SPSC のみ（line 67: 'Do NOT use with multi-producer/consumer scenarios'）。MpscBoundedRing は Phase 0/Phase 5 の前提 0 として**新規実装が必要** — 独立検証してから統合する。
//   DeferredDeletionQueue は既に Vyukov bounded MPMC（kQueueSize=4096, lock-free）として実装済み
//   （src/DeferredDeletionQueue.h:51-56）。intentQueue_ の MPSC 化はこのアルゴリズムを流用し、
//   SPSC 前提の LockFreeRingBuffer は触らない。
template<class T, size_t Capacity>
class MpscBoundedRing {
    // seq 配列 + entry 配列 + CAS による slot 予約（DeferredDeletionQueue.h と同一方式）
    // producer reservation → slot ownership → payload publication → consumer visibility の順序を保証
    // 有界: Capacity 超過時は満杯を返す（呼出し元が per-type admission policy を適用）
};
```

- **`LockFreeRingBuffer` は SPSC 用途のまま維持**（`ISRRetireOverflowRing.h:39-56` の SPSC 前提コメントを尊重）。
- **`MpscBoundedRing` は単体テスト → stress test → TSan → 既存 Intent Queue 置換**の順で検証（三次レビュー: "Unit Test → Stress Test → TSan → 既存 Intent Queue へ置換"）。
- **必要最小テスト**（三次レビュー列挙）: 1 producer / 2 producers / N producers / producer contention / queue full / consumer lag / **producer hole** / FIFO / cross-type FIFO / sequence monotonicity / **shutdown while producers active**。
- **producer hole の扱い**: Consumer は「自身の slot が書き込み完了するまで次の slot を読まない」（seq 番号で検知）。`DeferredDeletionQueue.h:119-124` の dequeue と同一パターン。
- **memory ordering**: `enqueuePos` の CAS は acq_rel（`DeferredDeletionQueue.h:83-102` と同一）、payload 書込み後に seq release（consumer acquire と HB）。`AtomicAccess.h` の `publishAtomic`/`consumeAtomic`/`compareExchangeAtomic` を使用。
- **INV-7: MPSC 4 ordering types の分離**（五次レビュー §20）: MPSC queue の FIFO を単に「queue FIFO」と定義しない。以下 4 つの順序を明示する:
  1. **sequenceId assignment order** — producer が `fetch_add` で seqId を取得する順序（Intent 生成順）。`PublishReceiptWaiter` の `complete()` とは別 — Intent の内部 seqId ではなく、publish の seqId。
  2. **reservation order** — producer が `MpscBoundedRing` の slot を CAS で予約する順序（Vyukov の enqueuePos CAS）。複数 producer の間で reservation が競合する → **producer hole** が発生。
  3. **publication order** — producer が payload を slot に書き込み seq release する順序。reservation と異なる可能性あり（Producer A が slot N を予約したが B が slot N+1 を先に書き込む）。
  4. **completion order** — `PublishReceiptWaiter::complete(seqId)` が呼ばれる順序（`seqId > lastCompleted_` 単調更新前提）。**`completion order ≠ pop order` ではない** — pop は reservation order で行われるが、`complete()` は publish の処理完了時に呼ばれる。**Publish は deferred へ退避できないかぎり completion order = seqId order が保たれる必要あり**（§840-857）。
  - Consumer は **reservation order**（slot seq）で pop し、**publication order**（payload visibility）で検証する（seq mismatch → producer hole → skip）。`DeferredDeletionQueue.h:119-124` の dequeue と同一パターン。
  - **shutdown admission gate**（INV-6 関連）: MPSC 化後、producer がアドレス空間で予約を完了した瞬間と shutdown が始まった瞬間の race が発生する可能性あり。`shutdown admission gate` を明示する: `requestShutdown()` → admission gate closed（`isAdmissionOpen` atomic false）→ producer の `push()` は gate が閉じていることを検知して false を返す → **entry 消失なし**（producer は所有権を取り戻して retry/deferred）。gate closed 後に予約済みの entry は drain 時に必ず消費される。

0. **（前提・必須）`intentQueue_` の SPSC→MPSC 化**（`MpscBoundedRing` 新設・独立検証）:
   - 案 X（推奨）: `MpscBoundedRing` を**独立実装**（`DeferredDeletionQueue` の Vyukov アルゴリズムを流用）し、`intentQueue_` を置換。RT パス（Audio Thread）からは push されないため、CAS retry は許容（ただし **drop はしない** — full 時は false を返し、呼出し元が per-type admission policy を適用）。
   - **前提 0 が未完了なら統合しない**（段階的移行の第一歩を「前提 0」とする）。

1. **`Intent` の payload に Observe を統合**（Recovery は統合しない）: `Intent{ type, payload }` の variant に `ObservePayload`（既に `intent.payload.observe` は定義済み）を完備。**`Recovery` 型は IntentType から削除しない**（`kDispatchTable` の 1:1 total mapping を維持）が、Dispatcher 経由で流れないように **`RecoveryIntentHandler::handle` は enqueue-only（第7章案 A'）で、`recoveryIntentQueue_`（Builder Work Queue）へ転送**する。これにより循環は構造的に排除される。
2. **push 側の置換**:
   - `AudioEngine.Timer.cpp:896,1029,1568` の `submitObserve(fadingHandle)` → `submitObserve` 実装を `intentQueue_.push(Intent{type=Observe, handle})` に変更
   - **`submitRecoveryRequest()` は `intentQueue_` に変更しない**（`recoveryIntentQueue_` のまま — Builder Work Queue 分離のため）。
   - **`submitQuarantine`（`cpp:696`）は drop をやめる**: `intentQueue_.push` が false を返したら quarantine 専用 fallback ring へ（三次レビュー policy 表に従う）。現状の「saturate 時に静かに drop」（`cpp:700` コメント）は撤回。**Severity: P0** — Quarantine intent が drop されると quarantined DSP が永久に retire されず、RT からアクセス不能なメモリが残存する (use-after-free via stale handle)。
3. **pop 側の置換**（`ProcessIntent.cpp:33-45`）: `observeIntentQueue_`/`observeFallbackQueue_` の while-pop を廃止し、`intentQueue_` の単一 while-pop に統一。Observe の epoch フィルタ（`dispatchObserve`）は Dispatcher 層に維持。**`recoveryIntentQueue_` の pop は Builder Loop 側（`RebuildDispatch.cpp`）のままで、ProcessIntent からは pop しない**。
4. **Observe Deferred Ring は維持**: `observeDeferredRing_`/`drainObserveDeferred`（FUTURE-8）は overflow 専用の分離されたリングとして残す（`ProcessIntent.cpp:47`）。
5. **`recoveryIntentQueue_` は Builder Work Queue として維持**: 専用キュー・`RecoveryIntent` 型・`popRecoveryRequest` API は**残す**（第4章の buildSource 値コピー拡張を反映）。旧案の「`recoveryIntentQueue_` 削除 + `intentQueue_` の `type=Recovery` 参照」は**撤回**（RECOVERY-3 の「共通 Queue が望ましい」は Builder Work Queue 分離の下で再解釈 — Intent Queue と Builder Work Queue は**別概念**であり、Recovery は後者に属する）。

**循環のコンパイル時検証:** `RecoveryIntentHandler` が `submitRecoveryRequest` を呼ぶ場合、その引数先が `intentQueue_` でないことをコードレビュー + テストで担保（`kDispatchTable` の Recovery スロットは enqueue-only であり、CoordinatorLoop が pop した Intent を再 enqueue しない）。

**三次レビュー必須追記② — PublishReceiptWaiter の FIFO 前提（AudioEngine.h:3564-3587 実測 2026-08-08）:**

`PublishReceiptWaiter::complete()` は **`seqId > lastCompleted_` のときのみ** `lastCompleted_` を更新する（**単調増加前提**）:

```cpp
// AudioEngine.h:3564（実測）— 順序性前提がコードコメントに明記
// executePublish は intentQueue_ を SPSC で処理するため seqId は enqueue 順で処理される（順序性前提）。
// ただし PublishReceiptWaiter.lastCompleted_ は high-water mark semantic（if (seqId > lastCompleted_) lastCompleted_ = seqId）。
// 後の seqId が先に完了すると、先の seqId の waitFor() が即座に true を返す — strict per-seqId tracking ではなく high-water mark。
// intentQueue_ が SPSC（single consumer = CoordinatorLoop）であるため実際の処理順序は enqueue 順 = seqId 順だが、ReceiptWaiter はこの前提に依存する。FPS 化（FUTURE-10）後もこの不変条件を維持する必要あり。
void complete(seqId) { if (seqId > lastCompleted_) lastCompleted_ = seqId; }
bool waitFor(seqId) { return seqId <= lastCompleted_; }   // :3575-3582
```

- **MPSC 化の影響**: `MpscBoundedRing` は slot 予約順（CAS）で pop されるため、**単一 Consumer（CoordinatorLoop）消費の限り pop 順序 = 予約順序 = seqId 順序**が保たれ、FIFO 完了は維持される。ただし **per-type admission policy の「Publish を deferred 経路（`PublicationAdmission::evaluateDeferred`）へ退避」** や **cross-type の再順序**があると、`seq=101` の Publish が遅延し `seq=102` が先に `complete()` される可能性がある。
- **危険シナリオ**: `complete(102)` が先に実行され `lastCompleted_=102` になると、未完了の `waitFor(101)` が **`seqId <= lastCompleted_` の判定で誤成功**する。
- **修正方針（MUST）**:
  1. **Publish は deferred 経路でも seqId 順序を保持**（`PublicationAdmission::evaluateDeferred` の退避は「先頭から順」に限定し、cross-type の reorder で Publish の相対順序を壊さない）。Observe は観測情報なので順序が崩れても許容 — **Publish のみ順序不変が要件**。
  2. **MPSC 化の受入条件に「PublishReceiptWaiter の monotonic-completion invariant 維持」を MUST として追加**（第14章）。「Publish が deferred へ退避 → 後続 seqId が先に complete → waitFor 誤成功」を stress test で検出する。
  3. 代替（Publish の順序不変が保てない場合のみ）: `PublishReceiptWaiter` を **out-of-order 完了対応**（`lastCompleted_` 高水位 + 個別完了ビットマップ / セット）に拡張。ただし本設計では (1)(2) を優先 — キュー側で順序を保証する方が waiter の単調性を維持でき、既存契約の変更が最小。

### 検証
- **四次実測照合（2026-08-08）:** FUTURE-10 の根拠コードを再実測。
  - `intentQueue_`（`ISRRuntimePublicationCoordinator.h:398-399`, `LockFreeRingBuffer<Intent, kIntentQueueCapacity=4096>`）が SPSC リングである一方、push 元は複数スレッド（RebuildDispatch.cpp:1053 → h:4296 / Timer.cpp:1788,1826 / CoordinatorLoop）＝ **MPSC 実態**を確認。
  - `observeIntentQueue_`（h:376）/`observeFallbackQueue_`（h:380）/`recoveryIntentQueue_`（h:394）が種別別に存在。
  - `LockFreeRingBuffer` は SPSC 専用（`LockFreeRingBuffer.h:2,34` コメント明記）。`ISRRetireOverflowRing.h:11-19` も SPSC 前提。
  - `enqueuePublicationIntent`（`ISRRuntimePublicationCoordinator.h:227-240`、push は `:240`）は full 時 false を返し、**呼出し元が `ownerChannel().take(key)` で所有権を回収**する設計を確認（drop ではない）。
  - `PublishReceiptWaiter::complete()`（h:3565-3587）は `seqId > lastCompleted_` のときのみ更新（単調増加前提）。
- DoD #4（単一 intentQueue — Observe/Publish/Quarantine）+ #7（cross-type FIFO）達成を Integration Test で確認
- **前提 0 の MPSC 化（三次レビュー必須テスト一式）**: 1/N producers・producer contention・queue full・consumer lag・**producer hole**・FIFO・cross-type FIFO・sequence monotonicity・**shutdown while producers active** を単体 → stress → TSan の順で確認（エントリ消失・破損なし）
- **per-type admission policy（三次レビュー必須）**: `Publish`/`Quarantine`/`Recovery` の drop が**ゼロ**であることを stress test でカウント確認（overflow 時は fallback/retry/coalesce へ振り分け）
- **cross-type FIFO 順序保証**: `Observe → Publish → Quarantine` の push 順が pop 順と一致すること（`sequenceId` 単調増加を利用した順序検証テスト）
- **PublishReceiptWaiter monotonic-completion invariant（三次レビュー必須）**: Publish を deferred へ退避した後も `complete()` が seqId 単調更新を維持し、`waitFor(101)` が `complete(102)` 後に誤成功しないこと（deferred 退避を含む stress test で検証）
- Observe の overflow 経路（`observeDeferredRing_`）が無傷で動作すること
- 既存 `kDispatchTable` の 1:1 mapping + `static_assert(DispatcherHasNoDecision)` が引き続き通ること
- **Recovery 循環の不在**: Recovery Intent が Dispatcher 経由で再 enqueue されないこと（stress test でキュー滞留がゼロ）
- **`submitQuarantine` の drop 廃止**: quarantine 検出が満杯時に fallback ring / HealthEvent へ正しく振り分けられること（bad DSP が使用可能のまま残らないこと）
- **shutdown admission gate (INV-6)**: `requestShutdown()` 後、producer の `push()` が gate closed を検知して false を返し、entry 消失なし（既に予約済みの entry は drain 時に消費される）。stress test で "shutdown while producers active" + "producer reservation race" のエントリ消失・破損なしを確認

### リスク
高（挙動変更を伴う大規模リファクタリング）。**前提 0（SPSC→MPSC）が欠けると、統合自体が既存の潜在競合を増幅する**。**実測: current code は Observe を intentQueue_ に統合していない** — `processIntent()` (ISRRuntimePublicationCoordinator_ProcessIntent.cpp:34-51) は Observe を dedicated SPSC ring (observeIntentQueue_/observeFallbackQueue_) から先に**全件 drain**し、**その後** intentQueue_ を処理する。この順序により、Observe Intent の大量到達が Publish/Quarantine/Recovery の processing を delay させる可能性がある（OBSERVE HOL BLOCKING）。MPSC unification（FUTURE-10）で intentQueue_ に統合する場合、cross-type priority ordering（Publish/Quarantine/Recovery > Observe）を導入する必要がある — あるいは Observe を coalescing/fallback で処理し、intentQueue_ からは除外する。**現行 code (ProcessIntent.cpp:44-45) は Observe を dedicated ring から先に drain し intentQueue_ は後** — この順序は Observe flood が Publish/Recovery を delay する HOL blocking を引き起こす。**FUTURE-10 実装時の必須検証: intentQueue_ に統合した場合、round-robin processing（Observe N : Common 1 ratio）または priority queue を導入すること**。**段階的移行**: まず **前提 0（MpscBoundedRing 独立実装 + 単体/stress/TSan 検証）を 1 コミット** → Observe を統合して 1 コミット → 挙動確認、の **2 段階**を推奨。**Recovery は Builder Work Queue として分離維持のまま**（第7章 — 統合すると循環するため、統合しない）。**三次レビュー: drop policy を実装すると「lock-free 化した代わりに Runtime Intent を失う」新種の semantic failure を導入するため、per-type admission policy（表）を厳守。

---

## 7. FUTURE-10: `RecoveryIntentHandler::handle` 実装 — 二次レビュー NO-GO → enqueue-only + Builder Work Queue 分離

**状態:** 🔴 P0 NO-GO（案 A は intentQueue_ 再 enqueue で循環を生む。Builder Work Queue 分離に変更）
**対象:** `src/audioengine/ISRIntentDispatcher.h:43-44` / `src/audioengine/ISRRuntimePublicationCoordinator.cpp:648-672`

### 現状
```cpp
struct RecoveryIntentHandler final : IntentHandler {
    void handle(const Intent&, IntentHandlerContext&) const noexcept override {} // A3 Step 5: → Recovery path
};
```
`kDispatchTable` には登録済み（`ISRIntentDispatcher.h:61`）だが handle が空。FUTURE-10 の DispatchTable 経路で Recovery Intent が流れても**何も起きない**。

### 二次レビュー指摘（核心）— 案 A の循環構造
旧案 A は `submitRecoveryRequest(intent.payload.recovery.handle)` を呼ぶもので、第6章の「`submitRecoveryRequest` を `intentQueue_.push` に変更」と合体すると無限循環になる。**Recovery は Dispatcher 経路（CoordinatorLoop pop）に載せない**。

### 改修案（詳細設計）: enqueue-only + Builder Work Queue 転送（循環の構造的排除）

```cpp
// ISRRuntimePublicationCoordinator.cpp — RecoveryIntentHandler::handle
//   Handler は Decision/World 書換禁止（HANDLER-1）。Recovery は Builder の作業なので、
//   Builder Work Queue（recoveryIntentQueue_）へ enqueue するのみ。
//   ★ 循環排除: intentQueue_（CoordinatorLoop が pop）に再 enqueue しない。
//     Dispatcher で pop された Intent の「処理結果」を別キュー（Builder Work Queue）へ
//     運ぶため、Dispatcher 自身へのフィードバックにはならない。
void RuntimePublicationCoordinator::handleRecoveryIntent(RecoveryPayload& payload) noexcept
{
    // Builder Work Queue へ転送（第6章: Recovery は統一しない・分離維持）
    submitRecoveryRequest(payload.handle);   // recoveryIntentQueue_.push のみ（現行実装のまま）
}

// ISRIntentDispatcher.h
struct RecoveryIntentHandler final : IntentHandler {
    void handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept override {
        // enqueue-only。intentQueue_ へは返さない（循環防止）。
        // ctx.engine.runtimeOrchestrator_->submitRecoveryRequest(intent.payload.recovery.buildSource);
        //   → 実装は第4章の buildSource 値コピー（snapshot lifetime 構造的解決）を反映
        //   → 四次追記: convolverBuildSnapshot は内包しない（juce::File/String 非 POD）。
        //     build 時に現在の uiConvolverProcessor.captureBuildSnapshot() から取得（第4章案 i）。
    }
};
```

**方針:**
- **案 A'（採用）: Handler は Builder Work Queue への enqueue-only**。`submitRecoveryRequest` は `recoveryIntentQueue_`（Builder Work Queue）への push のまま変更しない。第4章の payload 拡張（`RuntimeBuildSnapshot buildSource` 値コピー）を反映。
- **案 B（不採用）: Handler を削除し直接 Builder Loop 消費** — `kDispatchTable` の 1:1 total mapping 要件（`static_assert`）と矛盾するため不採用。Handler は enqueue-only で維持し、`static_assert(DispatcherHasNoDecision)` を保つ。
- **循環の構造的排除**: Handler は pop 元（intentQueue_）と異なるキュー（recoveryIntentQueue_）に書くため、Dispatcher のループに再流入しない。`submitRecoveryRequest` が `intentQueue_` を書く実装は**禁止**（コードレビュー + テストで担保）。

### 検証
- **四次実測照合（2026-08-08）:** `ISRIntentDispatcher.h` を再実測。
  - `RecoveryIntentHandler::handle`（:43-44）は空スタブ（no-op）のまま。
  - `kDispatchTable`（:58-63）に `Recovery` スロット登録済み（1:1 total mapping）。1:1 は `static_assert(std::size(kDispatchTable) == kIntentTypeCount)`（:64-66）で保証。
  - `static_assert(DispatcherHasNoDecision)`（`HandlerIsStateless` は :71-74、`DispatcherHasNoDecision` は :76-80、static_assert は :81-83。`HandlerIsStateless` は polymorphic + sizeof==sizeof(void*) + default-constructible で検証）が維持されている。
- `kDispatchTable[Recovery]` 経由で `submitRecoveryRequest` が呼ばれる単体テスト（enqueue 先が `recoveryIntentQueue_` であること）
- **循環不在テスト**: Recovery Intent を 10 万回 enqueue → pop しても intentQueue_ の滞留がゼロ（Dispatcher に再流入しない）
- `static_assert(DispatcherHasNoDecision)`（ハンドラが状態を持たない）を維持

### リスク
低〜中。Handler を enqueue-only + Builder Work Queue 分離にすれば HANDLER-1 契約に整合し、Dispatcher の純粋性を保つ。`submitRecoveryRequest` の引数先が `recoveryIntentQueue_` のままであることが循環回避の要件 — 実装時は第6章と併せて確認する。

---

## 8. 検証: Shutdown Pipeline 検証（SHUTDOWN-1〜7）

**状態:** ☐ 未着手（検証自体）
**対象:** `src/audioengine/ISRShutdown.cpp` / `RuntimeDrainAudit.h` / evidence（`shutdown_trace.json` 等）

### 現状
- SHUTDOWN-1〜6 の設計契約は REPAIR_PLAN.md:1369-1420 に記載済み。
- SHUTDOWN-7（ActiveBuilder）は第5章で**再設計**（タイムアウト時 reason 決定は実装済み。完了条件への組込みは撤回 → shutdown 順序確定が本実装）。
- `evidence/shutdown_trace.json` の記録経路は存在する（`ISRShutdown.cpp:168-303` `emitShutdownTrace` — アトミック置換・rename リトライ・fallback 付き）。

### 改修案（詳細設計）
以下の対応表に沿って、**各契約のコード上の実装有無を 1:1 で検証**し、不足を埋める。**四次実測照合（2026-08-08）の結果、各契約は下記の通り `isFullyDrained`/`collectDrainAudit`/`waitForDrain` に実装済み**であることを確認:

| 契約 | 検証内容 | 状態 | 実測での実装箇所 |
|------|----------|------|------------------|
| SHUTDOWN-1 | 全 Queue Drain（Intent/Publish/Observe/Recovery） | ✅ 実装確認 | `isFullyDrained`（AudioEngine.Threading.cpp:114-136）が `pendingIntentCount`/`publicationBacklogCount`/`retireBacklogCount`/`fallbackDepth`/`overflowRingResident`/`quarantineResident` を集計 |
| SHUTDOWN-2 | Pending Publication が 0 になるまで待機 | ✅ 実装確認 | `getPublicationBacklogCount`（collectDrainAudit:78）＋ `setPublicationBacklogCount`（isFullyDrained:118） |
| SHUTDOWN-3 | Retire Queue Drain | ✅ 実装確認 | `pendingIntentCount`（collectDrainAudit:79）＋ `routerPendingRetire`（:81-82, ring+fallback 合計）＋ `setRetireBacklogCount`（isFullyDrained:123） |
| SHUTDOWN-4 | Advance Epoch 後に Active Crossfade が 0 | ✅ 実装確認 | `crossfadeRuntime_.isPending()`（collectDrainAudit:80）→ `activeCrossfadeCount` |
| SHUTDOWN-5 | Reader が epoch 域を離脱 | ✅ 実装確認 | `m_retireRouter->detectStuckReaders(10)`（collectDrainAudit:74, 1回のみ呼び出し）→ `stuckReaderCount`（:97）。タイムアウト時 reason は `ReaderActive`（ReleaseResources.cpp:437-438） |
| SHUTDOWN-6 | 共通 Intent Queue と Shutdown の連携 | ✅ 実装確認 | `requestShutdown`（ReleaseResources.cpp:75）→ `shutdownCoordinatorLoop`（:189, processIntent 停止）→ `isFullyDrained` の `pendingIntentCount` 集計 |
| SHUTDOWN-7 | ActiveBuilder 検出（第5章の実装） | 🔴 再設計（タイムアウト時 reason は既存 `ReleaseResources.cpp:439-440`。完了条件への組込みは撤回 → shutdown 順序確定が本実装） | — |

**検証方法:** シャットダウン誘発テスト（`AudioEngineHarness`）で各フェーズの `blockingReason` を観測し、`shutdown_trace.json` と突き合わせる。`VerifyDrained` が無限待ちしないこと（タイムアウト付き）も確認。

**四次実測の追加確認事項:**
- `waitForDrain(2000, 2)`（ReleaseResources.cpp:430）は `isFullyDrained()` をポーリング（AudioEngine.Threading.cpp:138-170、ポーリング本体は `:158` の `while (!isFullyDrained())`）し、タイムアウト時は false を返す（無限待ちなし）。
- `emitShutdownTrace` は `blockingReasonStats` を JSON に出力し `verified` フラグ（violations==0 && boundedComplete）を付与（ISRShutdown.cpp:260-276）。
- SHUTDOWN-1〜6 の完了条件は `RuntimeDrainAudit::isAllZero()`（RuntimeDrainAudit.h:77-84）と `getPrimaryBlockingReason()`（:65-74）に集約されており、`shutdown_trace.json` の `verified` と 1:1 対応する。

### リスク
低（検証主体）。

---

## 9. BUG-060: ISRRetireRuntimeEx::reclaim の TOCTOU — ✅ 実装済みに訂正

**状態:** ✅ 実装済み（二次レビューで実測確認）
**対象:** `src/audioengine/ISRRetireRuntimeEx.cpp:205-225`

### 現状（実測 2026-08-08 — 二次レビューで再検証）

REPAIR_PLAN.md:2465 の記載は「reclaim 処理で該当ハンドルの age/エポック差分を unsigned で減算する際、値が負になるケースで巨大数になる underflow リスク」だったが、**実コードでは既に解消済み**:

```cpp
// ISRRetireRuntimeEx.cpp:205-225 — EpochControl::reclaim
if (previousLane == RetireLane::Quarantine)
{
    // BUG-060: TOCTOU 排除 — check-then-act ではなく fetchSub 単一アトミック + 回復
    const auto previous = convo::fetchSubAtomic(
        quarantineResidentCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    if (previous == 0) {
        // 減らしすぎた（既に 0）ので元に戻す（UINT64_MAX ラップ防止）
        convo::fetchAddAtomic(quarantineResidentCount_, static_cast<std::uint64_t>(1), std::memory_order_acq_rel);
    }
}
```

**確認結果:**
- `cur - last` の unsigned 差分パターンは `ISRRetireRuntimeEx.cpp` に存在しない（`rg "cur - last|lastReclaimEpoch"` 0 件）。
- TOCTOU は **fetchSub 単一アトミック + previous==0 回復**で排除済み（check-then-act をやめている）。
- **既存 `EpochDomain::isOlder(a, b)`**（`EpochDomain.h:399-402`, `static_cast<int64_t>(a - b) < 0`）がエポック wraparound 対応の比較として利用可能。

### 改修案
**コード変更は不要**（実装済み）。ただし REPAIR_PLAN.md:2465 の記述が古いため、本計画書で「実装済み」に訂正する。将来、reclaim の age 比較を追加する場合は `EpochDomain::isOlder`（wraparound 対応）を使うこと — `cur < last` の単純比較はエポック wraparound で誤判定するため使用禁止。

### 検証
- **四次実測照合（2026-08-08）:** `ISRRetireRuntimeEx.cpp:215-222` に `fetchSubAtomic(quarantineResidentCount_, 1, acq_rel)` + `previous==0` 時の `fetchAddAtomic` 回復（UINT64_MAX ラップ防止）を確認。`EpochDomain.h:399-402` に `isOlder`（`static_cast<int64_t>(a-b)<0`）を確認。
- 既存の reclaim パスが fetchSub 単一アトミックであることを単体テストで確認（TOCTOU 再発防止）
- エポック wraparound を再現するテストで `EpochDomain::isOlder` が正しく比較すること

### リスク
なし（実装済み。本計画書の訂正のみ）。

---

## 10. BUG-061: `auditLog_` mutex 保護 — ✅ 実装済みに訂正

**状態:** ✅ 実装済み（二次レビューで実測確認）
**対象:** `src/audioengine/ISRDSPQuarantine.h:83-86` / `src/audioengine/ISRDSPQuarantine.cpp`

### 現状（実測 2026-08-08 — 二次レビューで再検証）

REPAIR_PLAN.md:2466 の記載は「`std::vector` ベースの監査ログに並行アクセスがあるとデータ競合」だったが、**実コードでは既に mutex 保護済み**:

```cpp
// ISRDSPQuarantine.h:83-86
// ★ BUG-061: auditLog_ は複数 NonRT スレッド（Timer / Message）から
//   同時アクセスされるため、全アクセスを mutex で保護する
mutable std::mutex auditMutex_;
std::vector<Entry> auditLog_;
```

`ISRDSPQuarantine.cpp` の全アクセス（`quarantineHandle` :38、`reclaimSlot` :60,87、`getEntry` :120、`getMaxEntryAgeSec` :146、`compactAuditLogLocked` :162-171）が `std::lock_guard<std::mutex> lock(auditMutex_);` で保護されている。

**RT パスからの参照はゼロ**（`rg "auditLog_"` で `.cpp` の lock 内のみ。RT 側は `quarantineActiveFlags_`（atomic）のみ参照）。

### 改修案
**コード変更は不要**（実装済み）。REPAIR_PLAN.md:2466 の記述が古いため、本計画書で「実装済み」に訂正する。`quarantineActiveFlags_` の atomic アクセス（RT）と `auditLog_` の mutex アクセス（NonRT）の責務分離が契約として成立していることを文書化。

### 検証
- **四次実測照合（2026-08-08）:** `ISRDSPQuarantine.h:83-86` に `mutable std::mutex auditMutex_` + `std::vector<Entry> auditLog_` を確認。`.cpp` の全アクセス（:37,59,85,119,145,177）が `std::lock_guard<std::mutex> lock(auditMutex_)` で保護され、`compactAuditLogLocked`（:158）は lock 内（:72,152,178）からのみ呼ばれることを確認。
- 並行 append + snapshot の TSan テスト（既存実装に対する回帰確認）
- RT パスからの `auditLog_` アクセスが存在しないこと（`rg` で確認）

### リスク
なし（実装済み。本計画書の訂正のみ）。

---

## 11. FUTURE-5: MemoryPool化（`registry_` → `registryPool_`）

**状態:** ☐ 未着手（Storage Policy のため**実装順序は最後**）
**対象:** `src/audioengine/DSPHandleRuntime`（registry_）関連

### 現状（実測 2026-08-08 — 二次レビューで再検証）
REPAIR_PLAN.md:255, 614-704 の設計のみ。**ただし `registry_` は既に固定 256 slot の `std::array` として実装済み**（`ISRDSPHandle.h:115` `MAX_DSP_SLOTS = 256`、`:176` `std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_{}`）— 文書の「現状の動的確保」という前提は**古い**（旧 REPAIR_PLAN.md 時点の記述）。残る作業は「フリーリスト（O(1) pop）」と「generation タグで freed-slot 再利用の stale 検出」の整理・明確化のみ。

### 改修案（詳細設計）
REPAIR_PLAN.md:614-704 の FUTURE-5 設計に従う。要点:
1. ~~`std::array<Slot, 256>` の固定プール + フリーリスト~~ → 固定配列は**実装済み**（`ISRDSPHandle.h:176`）。フリーリスト（O(1) 確保）を**追加**する（現在は `ISRDSPHandle.cpp:41` の線形スキャンで空き slot を探す — O(n)）。
2. 非 RT コンテキストからの確保禁止（アサート）
3. generation タグで freed-slot 再利用の stale アクセスを検出（`ISRDSPHandle.cpp:31,176` で generation フィールドは既に存在 — 保存・検証の完全化）
4. slot 取得は RT-bounded（フリーリスト pop は O(1)）
5. FUTURE-6 の HandleTable と連携（slot index を handle にエンコード — `DSPHandle::slot` に既にエンコード済み）

### 検証
- **四次実測照合（2026-08-08）:** `ISRDSPHandle.h:115`（`MAX_DSP_SLOTS = 256`）/`:176`（`std::array<DSPRegistrySlot, MAX_DSP_SLOTS> registry_`）を確認。`ISRDSPHandle.cpp:41-42` の `for (slot = 1; slot < MAX_DSP_SLOTS; ++slot)` 線形スキャンで空き slot を探す（O(n)）現状を確認。
- RT パスからの割当/解放がロックなし O(1) であることをベンチマーク + 割当回数上限テスト
- 256 slot 枯渇時の動作（失敗 or 明確なエラー）

### リスク
中〜高（Storage Policy 全面変更）。**ISR 完成系（第4〜7章）完了後に実施**。

---

## 12. FUTURE-6: Handle Table 完全移行（`runtimeDSPHandleMap_` → HandleTable）

**状態:** ☐ 未着手（Storage Policy のため**実装順序は最後**）
**対象:** `src/audioengine/DSPHandleRuntime`（`runtimeDSPHandleMap_`）関連

### 現状
REPAIR_PLAN.md:256, 705-790 の設計のみ。`std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_`（`AudioEngine.h:4601`）の linear scan（`eraseByHandle` O(n)）を HandleTable（forward O(1) hash + reverse O(1) dense array）に置換する。`AudioEngine.h:4183` のコメント通り「MAX_DSP_SLOTS=256 のため問題なし」の O(n) が現状（書き込みは `runtimeDSPHandleMapMutex_` で保護済み — `AudioEngine.h:4109,4146,4174,4187`、読みは RT パスで lock なし）。

### 改修案（詳細設計）
REPAIR_PLAN.md:705-790 の FUTURE-6 設計に従う。要点:
1. forward: `DSPCore*` → slot index の O(1) hash
2. reverse: slot index → `DSPCore*` の O(1) dense array
3. `eraseByHandle` を O(n) linear scan → O(1) に置換
4. FUTURE-5 の slot 配置と 1:1 対応（handle に slot index を埋め込む）

### 検証
- **四次実測照合（2026-08-08）:** `AudioEngine.h:4109,4146,4174,4187` に `std::lock_guard<std::mutex> lock(runtimeDSPHandleMapMutex_)`（書き込み経路）を確認。`:4600-4601` に `std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_` 定義。`eraseByHandle`（:4182-4193）が `for` 線形スキャン（O(n)）で、読みは RT パス lock なし、を確認。
- 既存 `eraseByHandle` の全呼出しの置換
- ソークで retire 経路の O(1) 化を計測

### リスク
中〜高（Storage Policy 全面変更）。FUTURE-5 との整合が必要なため**両者を同じコミット単位**で実施する。

---

## 12.5 Phase 0: Invariant Test Plan (INV-1〜7)

**目的:** 7 不変条件（INV-1〜7）を既存コードに対してテストで固定し、各 Phase の回帰を即検出できる土台とする。各 invariant に対応するテストは `src/tests/` に `invariant_*.cpp` の形式で追加。

### INV-1: RuntimeWorld authority
- **TSan テスト**: `RuntimeWorld` の `publish()` を Message Thread から呼んだ場合、Audio Thread が `world->processor` への参照を取得している間に `publish()` が完了し `world` ポインタがスワップされると、`publish()` 側が stale world を観測しないこと。
- **Unit テスト**: `DSPLifetimeManager::retire` が `ISRRetireRouter::enqueueWithRetry`（Coordinator authority）経由でのみ retire を行う。`Builder/Rebuild スレッド`が直接 `publishEpoch()` を呼ばないことを `jassert` で検証（実装: `AudioEngine.CtorDtor.cpp:187,204`、`ReleaseResources.cpp:230,253`）

### INV-2: LinearRamp ownership
- **TSan テスト**: Audio Thread が `dryScaleGain_->getNextValue()` を call 中、NonRT スレッドが `dryScaleTarget_` atomic publish を行う。TSan で `dryScaleGain_`/`gain_` への NonRT 同時アクセスゼロ。
- **Unit テスト**: `setAudioDeviceTypeName()` に `jassert(MessageManager::isThisTheMessageThread())` が付いていること (`/fix_phase0` ブランチ実装済み — Ch.1 を参照）。

### INV-3: Retire ownership
- **Unit テスト**: `RetireQuarantineStore::quarantine()` が `true`(success) を返した場合のみ deleter を呼出し元が実行しないこと。`drain()` が deleter を実行するのは epoch safe 到達後のみ。
- **TSan テスト**: `drain()` 中に RT スレッドが同じ entry を参照していないこと。

### INV-4: Recovery semantic
- **Unit テスト**: Recovery build 後の `DSPCore` の `irIdentityHash` が `uiConvolverProcessor` の `structuralHash` と一致すること (`transferIRStateFrom` 確認）。`quarantinedHandle` が buildSource から除外されていること。
- **Integration テスト**: quarantinedHandle が resolve 不可能でも Recovery build が成功すること（snapshot 値コピーによる）。
- **Compatibility failure policy（2026-08-08 実測追加）**: `finalizeRuntimeBuildSnapshot()` の generation mismatch 検出時は **古い snapshot をそのまま publish する fallback を禁止**。Compatibility failure 時は: (a) discard current recovery build, (b) capture newest authoritative `RuntimeBuildSnapshot`, (c) rebuild。この policy を `RuntimePublicationOrchestrator::processDeferredAdmission()` または `Builder Loop` に実装する。

### INV-5: Intent loss
- **ObserveIntent classification — 重要な再確認（2026-08-08 実測）**: `ObserveIntentHandler::handle`（`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:67-71`）は `ctx.lifetimeMgr.retireByHandle(intent.payload.observe.handle)` を呼ぶ — Pure telemetry ではなく **state-affecting (retire)**。したがって Observe の drop は handle leak を直接起こさない（`retireByHandle` は EpochDomain で idempotent）。しかし drop により retire が遅延する可能性があり、RetireQuarantineStore の overflow と同じ Health escalation path に乗せるべきではない（retire は既に EBR で保護済み）。Observe drop の補完は 3 層 fallback（primary → fallback → deferred）で十分。
- **Integration テスト**: `intentQueue_`（MPSC 化後）の full 時に Publish/Quarantine が drop されないこと。Observe は 3 層 fallback で補完可能。
- **Stress テスト**: 10 万 iteration で Intent drop カウンタがゼロ（Publish/Quarantine）、Observe は fallback で補完率 100%。

### INV-6: Shutdown ordering
- **Integration テスト**: `shutdown_trace.json` で `requestShutdown()` → gate closed → `push()` false → drain 消費の順序が保証されること。
- **Unit テスト**: shutdown admission gate の `isAdmissionOpen` atomic が `requestShutdown()` 後に `false` になること。gate closed 後の `push()` が `false` を返し、entry 消失なし（drain 時に消費）。

### INV-7: MPSC ordering
- **Unit テスト**: `MpscBoundedRing` の 4 順序 (sequenceId assignment → reservation → publication → completion) が個別に検証されること。
  - **reservation order = seqId order**: pop 順序が CAS reservation 順と一致（`seqId` 単調増加で検証）。
  - **producer hole**: Consumer が reservation した slot が publication されるまで待機（seq mismatch → skip）。
- **Integration テスト**: `PublishReceiptWaiter::complete()` の `seqId > lastCompleted_` 単調更新が MPSC 化後も維持され、`waitFor(101)` が `complete(102)` 後に誤成功しないこと。Publish は deferred へ退避しても seqId 順序を保持すること。
- **Stress テスト**: 10M enqueue/dequeue で sequence monotonicity 破綻ゼロ + cross-type FIFO 順序保証。

### Test infrastructure
- **Framework**: `src/tests/invariant/invariant_test.cpp` — Google Test ベース。`AudioEngineHarness` で RT スレッド + NonRT スレッドをシミュレート。
- **TSan**: `src/tests/invariant/tsan_invariant_*` — 各 invariant の TSan がゼロであること。
- **コードカバレッジ**: `gcov` で Ch.2-7 の各 invariant 関連コードが 100% カバーされる。
- **CI ゲート**: Phase 0 テストは `invariant-gate` CI ジョブで実行。**すべて PASS しない限り Phase 1 以降の実装を認めない**。

---

REPAIR_PLAN.md:2469-2487（BUG 優先）と ISR 完成系優先の原則を統合。**三次レビューにより P0 項目の「設計修正」が先行**し、**五次レビューにより「Phase 0 で既存不変条件をテスト固定してから改修」「SHUTDOWN-7 は実装ではなく順序検証テスト」**となる。各フェーズが独立コミット:

| Phase | 項目 | 区分 | 理由 |
|-------|------|------|------|
| **0** | **既存不変条件のテスト固定 + Phase 0 不変条件（INV-1〜7）の文書化**: 設計修正を先行確定。以下の 7 不変条件（INV-1〜7）を既存コードに対してテストで固定し、各 Phase の回帰を即検出できる土台とする。**設計修正（BUG-014 enum 化 / BUG-028 RT 専有化 / BUG-015/027 Router 配下 RetireQuarantineStore / FUTURE-3 buildSource 値コピー / FUTURE-10 per-type admission）を章 0 の判定表を緑にする。 | 設計 | **INV-1** RuntimeWorld authority（Builder 生成・Immutable Publish / retire は Coordinator）。**INV-2** LinearRamp ownership（dryScaleGain_/gain_ は RT のみ操作。reset() は Audio Thread 停止後のみ）。**INV-3** Retire ownership（retire → EBR → reclaim 順序。directDelete なし。RetireQuarantineStore は Router 配下単一配置）。**INV-4** Recovery semantic（Recovery = quarantined 除外した現在の authoritative configuration の再構築。過去 World の rollback ではない）。**INV-5** Intent loss（Publish/Quarantine/Recovery は drop 禁止。Observe は 3 層 fallback）。**INV-6** Shutdown（admission gate closed → stop producers → join workers → drain → reclaim → verify の順序不変）。**INV-7** MPSC ordering（sequenceId assignment → reservation → publication → consumption → completion の 4 順序を分離。Producer hole は seq 番号で検知）。 |
| **1** | **BUG-014**（enum atomic 化 + `publishAtomic`/`consumeAtomic` + underlying type `uint8_t` 明示 + `static_assert(std::is_trivially_copyable_v<MmcssPolicy>)` + `static_assert(is_always_lock_free)`） | バグ修正 | 五次レビュー推奨順 1 番目。P0 レース先回り。文字列比較を setter へ移動するだけで RT パスが atomic load 1 回に。コミット単位: 1 コミット |
| **2** | **BUG-028 再設計の適用**（`dryScaleGain_` への NonRT 直接操作を atomic publish に置換。LinearRamp は触らず、RT 側リセット経路の明文化。`complete()` は stale フラグを atomic publish で reset（実測 :95-103 一致）+ `dryScaleTarget_`/`startDelayBlocks_`/`dryHoldSamples_` の publish を**追加**（未実装）。**`start()` の `gain_.setCurrentAndTargetValue(0.0)` を**削除**（BUG-028 fix — LinearRamp への NonRT 書き込みを排除。RT 側 `armCrossfadeIfPending` が `setTargetValue` で代替）**）。**P1 hardening: `crossfadeGeneration` カウンタで block-boundary semantic consistency を強化**（五次レビュー §8 追加 — complete() の複数 atomic publish が block boundary で一致して消費されることを検証）） | バグ修正 | **P0 NO-GO 解消**。旧案のまま実装すると TSan が必ず失敗。コミット単位: 1 コミット |
| **3** | **BUG-015 + BUG-027**（`RetireQuarantineStore` を **ISRRetireRouter 配下に新設**（Authority Singularization — SnapshotCoordinator/DSPLifetimeManager は直接保持せず Router API `quarantineRetire()` 経由）。epoch 比較は `EpochDomain::isOlder`、directDelete 禁止、retire/physical delete 責務分離。**QuarantinedEntry は DeletionEntry と同等フィールド（ptr/deleter/epoch/type/publicationSequenceId/generation）を保持— ownership transfer の完全性。capacity exhaustion policy: store full で delete は絶対禁止 → HealthEvent + `ISRHealthState` Degraded→Critical 遷移 + controlled shutdown**。**retire 順序逆転の修正: enqueue → 成功時のみ reclaim、失敗時は退避移送。`shutdownReclaim` transitional 措置は削除し `requestReclaim` に一本化**） | バグ修正 | retire リーク防止 + UAF 排除。新規コード ~60 行 + 接続 4 点。五次レビュー §5: Retire authority は 1 個のまま（Router 配下に Queue と QuarantineStore）。コミット単位: 1 コミット |
| **4** | **FUTURE-3 再設計の適用**（RecoveryIntent に buildSource 値コピー。quarantinedHandle 単独 payload を撤回。sealedSnapshot + spec 除外で build。Builder Loop 接続。**案 i 確定: `ConvolverProcessor::BuildSnapshot` は trivially copyable でないため内包せず、build 時に `uiConvolverProcessor.captureBuildSnapshot()` から取得。buildSource は metadata/fingerprint 輸送であり IR data は `transferIRStateFrom(engine.getConvolverProcessor())` で現在の `uiConvolverProcessor` から取得 — semantic は「現在のユーザー構成を再 build」（四次/五次実測確定）**）。**INV-4 形式化（RECOVERY-SEMANTIC-001）**: Recovery = quarantined 除外した現在の authoritative configuration の再構築（rollback ではない）。`currentSpecs()` を Builder 側 `normalizeRecoveryBuildInput()` で正規化** | ISR | **P0 NO-GO 解消**。resolve 不能な quarantinedHandle に依存せず、値コピーした snapshot で build。IR 実体は snapshot に存在しないため現在の processor から転送する（`RuntimeBuilder.cpp:447`）。コミット単位: 1 コミット |
| **5** | **FUTURE-10 前提 0/0b: `MpscBoundedRing` 独立実装 + per-type admission policy**（DeferredDeletionQueue の Vyukov bounded アルゴリズム流用。SPSC 前提の `LockFreeRingBuffer` は触らない）+ 単体 → stress → TSan の必要最小テスト一式（1/N producers・contention・full・consumer lag・producer hole・FIFO・cross-type FIFO・sequence monotonicity・**shutdown admission gate**）。**Publish/Quarantine/Recovery は drop 禁止** — fallback ring / retry / coalesce / HealthEvent へ振分。Observe は 3 層 fallback 維持。`submitQuarantine` の静かな drop は撤回。**INV-7: 4 順序を分離検証**（sequenceId assignment / reservation / publication / completion）。**INV-6: shutdown admission gate**（gate closed 後の producer push は false、entry 消失なし） | ISR | **三次レビュー必須**。MPSC は独立 bounded primitive として実装・検証してから統合。**PublishReceiptWaiter の FIFO 前提（AudioEngine.h:3564）を考慮し、MpscBoundedRing の pop 順序 = 予約順序（seqId 順）をテストで保証**。Publish の deferred 退避は seqId 順序を保持（`evaluateDeferred` は先頭から順）。コミット単位: 1 コミット |
| **6** | **Shutdown Pipeline 検証（SHUTDOWN-1〜7） + MPSC admission gate**: **SHUTDOWN-7 は「コード実装」ではなく「順序検証テスト」として実施**（`stopRebuildThread(join)` → `waitForDrain` の順序が `shutdown_trace.json`で保証されること。`rebuildThreadIsRunning==false` の完了条件組込みは撤回済み）。**INV-6: shutdown admission gate（`requestShutdown()` → gate closed → producer push false → drain 消費）**を MPSC 導入後に検証 | 検証 | 五次レビュー: SHUTDOWN-7 は join 済みでコード変更不要のため、**実装フェーズから検証フェーズへ移動**（レビュー推奨順 6）。ISR 完成系の受け入れ。コミット単位: 1 コミット |
| **7** | **FUTURE-10 Observe 統合（段階1）** + **Recovery は統合しない**（Builder Work Queue 分離を確定。`recoveryIntentQueue_` 維持）+ **RecoveryIntentHandler::handle 実装（enqueue-only + Builder Work Queue 転送）** | ISR | 大規模リファクタ。cross-type FIFO 順序保証テスト + Recovery 循環不在テストを追加。**Recovery queue 整理（レビュー推奨順 7）**。コミット単位: 2 コミット（統合 / 分離確定） |
| **8** | **FUTURE-5 + FUTURE-6** | Storage Policy | ISR 完成後に一括。**FUTURE-5 は固定配列化済み（`registry_` = `std::array<...,256>`）のため、残るはフリーリスト O(1) 化 + generation 検証の完全化**。コミット単位: 各 1 コミット |

**削除（実装済み）:** ~~BUG-060 / BUG-061~~ → 二次レビューで「実装済み」確認済み。コード変更不要。

**コミット単位の提案:** 各 Phase 1 コミット。特に FUTURE-10 は「Phase 5（MpscBoundedRing 独立実装 + admission policy）」「Phase 7（Observe 統合 / Recovery 分離）」の 2 段階に分離し、途中状態でもビルド・テストが通るようにする（Deferred Admission Integration と同じ方針）。**Phase 0 は既存不変条件のテスト固定 + 設計修正の確定（五次レビュー: 改修前に不変条件を締める）**で、BUG-028 / FUTURE-3 / SHUTDOWN-7 / FUTURE-10 の設計修正を先行確定させてから Phase 1 に進む。****SHUTDOWN-7: admission gate は既実装済み** — `PublicationAdmission.cpp:11-12` で `isShutdownInProgress()` チェック + `Decision::RejectedShutdown` リターン。`submitQuarantine`（`ISRRuntimePublicationCoordinator.cpp:674-701`）も `isShutdownInProgress()` をチェックしない — **queue full 時の silent drop が更に悪化**（shutdown 中の Quarantine intent が失われる）。

**★★ 2026-08-08 追加 — 2つの ShutdownPhase enum が共存中**:
- `AudioEngine::ShutdownPhase` (AudioEngine.h:2479): `Running/StopAcceptingWork/StopAudio/StopWorkers/ForceEpochAdvance/DrainRetire/Destroy`
- `convo::isr::ShutdownPhase` (ISRShutdown.h:25): `Running/AudioStopped/ObserverDrained/RetireClosed/EpochSettled/ReclaimComplete/EmergencyDrain/VerifyDrained/TimedOut/Failed/ShutdownComplete`

CtorDtor.cpp:96-229 は `AudioEngine::ShutdownPhase` を set し、同時に `ReleaseResources.cpp:73-520` は `shutdownRuntime_.transitionTo(convo::isr::ShutdownPhase)` も呼ぶ。二つの enum は手動で対応づけられているが、**対応表がテストで固定されていない** — enum 追加時の不整合リスク。invariant テストで `AudioEngine.shutdownPhase` と `shutdownRuntime_.phase` の同期を検証すること。`IntentType::Publish` の admission は `PublishExecutor → PublicationAdmission` 経路でゲートされている。`Quarantine`/`Recovery`/`Observe` は各々 `isShutdownInProgress()` チェックが別途存在する（`AudioEngine.Commit.cpp:195`, `AudioEngine.Timer.cpp:658,694,741,744,1116`, `AudioEngine.Threading.cpp:220`）。**実装済みだが、統一的不変条件としてテストで固定する必要あり** — gate closed 後の producer push は false を返し、entry 涧失なし。**。

---

## 14. 受け入れ条件（全体）— 三次レビュー反映

### 基盤
- `ctest` 全件 PASS（実測: CMakeLists.txt の `add_test` は現行 29 件（2026-08-08 時点）。「23/23」は過去の PASS 記録 — 本計画の Phase 適用後は全 add_test が PASS すること）＋ 追加 Integration Test
- `git diff --check` クリーン
- ASan/TSan CI green（`sanitizer-ci.yml`）
- `static_assert(DispatcherHasNoDecision)` / `kDispatchTable` 1:1 が維持される
- ソーク（`soak-ci.yml`）で retire/レース系の新規リークなし

### RT 安全（三次レビュー必須）
- **BUG-014**: `rg "currentDeviceTypeName_|getAudioDeviceTypeName"` が 0 件（enum 化の完全適用）+ `static_assert(std::is_trivially_copyable_v<MmcssPolicy>)` と `static_assert(std::atomic<MmcssPolicy>::is_always_lock_free)` が両方通る（underlying type `uint8_t` 固定）+ `currentMmcssPolicy_` への全アクセスが `publishAtomic`/`consumeAtomic` 経由
- **BUG-028**: TSan で `dryScaleGain_`/`gain_` への RT↔NonRT 同時アクセスが**ゼロ**。**`start()` の `gain_.setCurrentAndTargetValue(0.0)` を削除** + **`complete()` の `dryScaleGain_.setCurrentAndTargetValue(1.0)`（h:118）も削除**（BUG-028 fix — LinearRamp への NonRT 書き込みを排除）。`gain_.reset()` のみ NonRT で実行（totalSteps のみ設定）。RT 側 `armCrossfadeIfPending`（`AudioEngine.h:3878,4014`）が `setTargetValue`/`getNextValue` で fade を駆動。**complete() は stale フラグ（`useDryAsOld_`/`firstIrDryPending_`/`firstIrDryDone_` 等）を atomic publish で reset する**（実測 :95-103 — RT 安全）+ **`dryScaleTarget_`/`startDelayBlocks_`/`dryHoldSamples_` の atomic publish を追加**（未実装）。NonRT 側の LinearRamp（`dryScaleGain_`/`gain_`）への**直接 `setCurrentAndTargetValue` 操作は `Init.cpp`/`CrossfadeRuntime::reset`（RT 停止後）のみ**。**complete() 自体の stale reset は RT 安全**（atomic）。**Block-boundary semantic consistency**: `crossfadeGeneration`（P1 hardening）が追加されれば、`complete()` の複数 atomic publish が block boundary で一貫して消費されることを soak trace で検証する（五次レビュー §8 追加）。

- RT パス（Audio Thread callback）で lock・malloc・CoW 参照・`delete[]` が**新規導入されていない**こと（`MpscBoundedRing` の push/pop は RT パスから呼ばれない）

### キュー正しさ（三次レビュー必須 — Phase 6/7 受入）
- **MPSC primitive 単体テスト一式**: 1 producer / 2 producers / N producers / producer contention / queue full / consumer lag / **producer hole** / FIFO / cross-type FIFO / sequence monotonicity / **shutdown while producers active** が全て PASS
- **per-type admission policy**: stress test で `Publish`/`Quarantine`/`Recovery` の drop が**ゼロ**（overflow は fallback ring / retry / coalesce / HealthEvent へ正しく振分。データ喪失なし）
- **`submitQuarantine` の drop 廃止**: quarantine 検出が満杯時に fallback ring / HealthEvent へ正しく振り分けられること（bad DSP が使用可能のまま残らないこと）
- **cross-type FIFO 順序保証**: `Observe → Publish → Quarantine` の push 順が pop 順と一致（`sequenceId` 単調増加検証）
- **PublishReceiptWaiter monotonic-completion invariant**: `complete()` の `seqId > lastCompleted_` 単調更新が MPSC 化・admission policy（deferred 退避含む）後も維持され、**out-of-order 完了で `waitFor(101)` が誤成功しない**こと（`AudioEngine.h:3564-3587`。stress test で検証）
- **FUTURE-10 循環**: Recovery Intent を 10 万回 enqueue→pop しても intentQueue_ 滞留ゼロ（Dispatcher に再流入しない）

### EBR / メモリ安全性（三次レビュー必須）
- **BUG-015/027**: `RetireQuarantineStore::residentCount()` がソーク中に単調増加しない（drain が追いつく）+ **capacity exhaustion 時は HealthEvent 発火 + `ISRHealthState` Degraded→Critical 遷移 + controlled shutdown**（store full でも deleter は一切実行されない = **UAF を構造的に排除**。directDelete がコードに存在しないこと）。**五次レビュー §5: `RetireQuarantineStore` が `ISRRetireRouter` 配下に単一配置され、SnapshotCoordinator / DSPLifetimeManager がストアを直接保持していないこと**（Router API `quarantineRetire()` 経由でのみ移送）
- **retire 順序**: `DSPLifetimeManager::retire` が **enqueue →（成功時のみ）reclaim** の順であり、reclaim→enqueue の逆転**がコードに存在する**。`retireDSPHandleForRuntime`（AudioEngine.h:4155-4159）は `dspHandleRuntime_.retire(handle)` → `dspHandleRuntime_.shutdownReclaim(handle)`（= `reclaim()`）を**即座**に実行する。**その後** AudioEngine.Commit.cpp:4083 または DSPLifetimeManager.cpp:49 で `enqueueWithRetry` が呼ばれる — すなわち **reclaim → enqueue** の逆転順序が実コードで確認された。対照的に `DSPLifetimeManager::retireByHandle`（cpp:84-90）は `retire → enqueue` の**正しい順序**を使用する。`retireDSPHandleForRuntime`（transitional path）を `DSPLifetimeManager::retireByHandle` に一本化すべき。。enqueue 失敗時は RetireQuarantineStore へ移送され、Reclaimed のままリークしないこと
- **RetireEnqueueResult 再利用**: `tryEnqueue` が新設 enum ではなく既存 `RetireEnqueueResult`（ISRAuthorityClass.h:25-30）を使用していること
- **FUTURE-3**: Recovery build が quarantined DSP を buildSource から正しく除外 + **resolve 不能な quarantinedHandle でも build 成立**（値コピーした snapshot での build の有効性テスト）。quarantinedHandle への lifetime 依存（resolve ベース設計）が**コードに存在しない**こと。**IR data semantic: Recovery build 後の DSPCore が `transferIRStateFrom(engine.getConvolverProcessor())` 経由で現在の `uiConvolverProcessor` の IR を保持すること（`irIdentityHash == 現在 structuralHash` で確認）。snapshot に IR AudioBuffer を載せる実装が**存在しない**こと（POD 性質維持）**
- TSan ソークで EBR フェーズ境界のデータ競合ゼロ

### シャットダウン（三次/五次レビュー必須 — Phase 6/9 受入）
- **SHUTDOWN-7（五次レビュー: 実装 → 順序検証テストへ変更）**: `shutdown_trace.json` で `stopRebuildThread(join)` → `waitForDrain` の順序が保証 + `waitForDrain` 開始時点で `rebuildThreadIsRunning==false`。**コード変更なしで順序契約が成立すること自体を検証**（`ReleaseResources.cpp:75 requestShutdown → :188 StopWorkers → :189 shutdownCoordinatorLoop → :190 stopRebuildThread → :191 ObserverDrained → :430 waitForDrain` の実測順序が維持される）
- **SHUTDOWN-1〜7 対応表** の各項目が検証結果と紐づく（Phase 9）
- **MPSC shutdown admission gate (INV-6)**: `requestShutdown()` 後、producer の `push()` が gate closed を検知して false を返し、entry 消失なし（既に予約済みの entry は drain 時に消費される）。"shutdown while producers active" + "producer reservation race" のエントリ消失・破損なしを Phase 6 単体テスト + Integration Test で確認
- **MPSC 4 ordering types (INV-7)**: sequenceId assignment → reservation → publication → completion の 4 順序が個別に検証されている。`MpscBoundedRing` の pop 順序 = reservation order（seqId 順）が保証されていること

### 完成度
- **FUTURE-5 + FUTURE-6** 適用後: `registry_` → `registryPool_`、`runtimeDSPHandleMap_` → HandleTable への移行が完了し、対応する旧構造がコードベースから消えていること
  - **FUTURE-5 は実装済み部分を正確に切り分け**: 固定 256 slot 配列（`ISRDSPHandle.h:176`）は既存のまま維持し、フリーリスト O(1) 確保 + generation タグ stale 検出の完全化のみを残作業とする（`registry_` の線形スキャン確保 `ISRDSPHandle.cpp:41` の撤廃を含む）
