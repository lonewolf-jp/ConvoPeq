# ISR Rebuild Admission - Phase 0（Telemetry only）実装タスク分解

作成日: 2026-05-23
参照元: `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
方針: **挙動変更なし（制御ロジック不変更）**、観測性のみ先行導入

> 追記（2026-05-23）: Phase 1 着手として `submitRebuildIntent(...)` を実装し、
> `timerCallback()` の DeferredStructural / DeferredFinalizeAware 解放経路、
> `changeListenerCallback(uiEqEditor)`、`prepareToPlay()` 非MT分岐、
> `rebuildThreadLoop()` の retry 経路まで同APIへ移行済み（挙動は維持）。
>
> 追記（2026-05-23）: Phase 3 適用として、`Replaceable` 限定の latest-wins
> （debounce window 内 same-as-pending suppress）を有効化済み。
>
> 追記（2026-05-23）: Phase 4 着手として、`DeferredFinalizeAware` に
> timeout（2秒）監視を導入し、超過時は `REBUILD_FORCED_DISPATCH` を出力して
> `MustExecute` rebuild を強制発行する経路を追加済み。
>
> 追記（2026-05-23）: Phase 5 着手として、`requestRebuild(double,int)` の旧重複抑止
> （`recent_duplicate` / `pending_duplicate`）が `MustExecute` intent を落とさないよう、
> `submitRebuildIntent(..., MustExecute)` 経路では duplicate suppress をバイパスする実装を追加済み。
>
> 追記（2026-05-23）: Phase 5 継続として、`MustExecute` 経路では
> `DeferredStructuralWindow` の legacy suppress もバイパスするよう調整し、
> timeout 後 forced dispatch の最終実行保証を強化した。
>
> 追記（2026-05-23）: 方針明文化として、`MixedPhaseIntermediate` suppress は
> `MustExecute` でも維持（縮退しない）とする。これは重複抑止ではなく、
> progressive mixed-phase の中間状態 publish を防ぐ安全ガードとして扱う。
>
> 追記（2026-05-23）: Phase 5 継続として、`submitRebuildIntent` 内の
> delegate/defer/nonMT 系 telemetry の `policy` 表示を `N/A` 固定から
> 実際の `collapsePolicy`（`Replaceable` / `MustExecute`）反映へ統一した。
>
> 追記（2026-05-23）: Phase 5 残件対応として、`requestRebuild(double,int)` の suppress 分岐に
> 「`phase5_reduce_target`（縮退対象）」/「`phase5_keep_target`（維持対象）」タグを
> コメント・DIAGログ・telemetry付帯文字列で明示し、段階削除レビューの判別コストを削減した。
>
> 追記（2026-05-23）: Phase 5 の段階縮退として、`phase5_reduce_target` のうち
> `RecentDuplicate` suppress（200ms近傍の直近重複抑止）を削除し、
> duplicate 系は `PendingDuplicate` merge のみを維持する構成へ移行した。
>
> 追記（2026-05-23）: Phase 5 の段階縮退として、`phase5_reduce_target` のうち
> `DeferredStructuralWindow` suppress を削除し、該当条件でも request は
> 通常の queue 判定（TaskQueued / PendingDuplicate）へ進む構成へ移行した。
>
> 追記（2026-05-23）: `PendingDuplicate` は段階縮退候補から除外し、
> pending queue の過負荷抑止（backpressure）を担う `phase5_keep_target` として
> merge 扱いを維持する方針に固定した。

---

## 1. Phase 0 のスコープ定義

### 1.1 目的

- rebuild 経路を `request -> decision -> dispatch` で追跡可能にする。
- suppress / defer の理由をログで一意に判別可能にする。

### 1.2 非目的（禁止）

- suppress 条件の追加・削除
- pending task 判定ロジック変更
- defer 条件／タイミング変更
- `submitRebuildIntent` への全入口一括移行（段階移行中）

---

## 2. 対象関数（関数単位）

Phase 0 で触る関数は以下に限定する。

1. `AudioEngine::convolverParamsChanged(ConvolverProcessor*)`
   - File: `src/audioengine/AudioEngine.UIEvents.cpp`
2. `AudioEngine::requestRebuild(convo::RebuildKind)`
   - File: `src/audioengine/AudioEngine.RebuildDispatch.cpp`
3. `AudioEngine::handleAsyncUpdate()`
   - File: `src/audioengine/AudioEngine.RebuildDispatch.cpp`
4. `AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)`
   - File: `src/audioengine/AudioEngine.RebuildDispatch.cpp`
5. `AudioEngine::timerCallback()`
   - File: `src/audioengine/AudioEngine.Timer.cpp`
6. `AudioEngine::enqueueSnapshotCommand()`
   - File: `src/audioengine/AudioEngine.Init.cpp`

補助追加（必要最小限）:

- `AudioEngine` private helper（ログ共通化）
  - 例: `logRebuildTelemetry(...)`（命名は実装時に既存規約へ合わせる）
- telemetry counters（既存 debug counter 拡張のみ。新しい制御分岐は禁止）

---

## 3. ログイベント仕様（Phase 0実装版）

最終計画書のイベント名をそのまま採用する。

- `REBUILD_REQUESTED`
- `REBUILD_MERGED`（Phase 0では「would-merge」相当の観測ログとして出してよい）
- `REBUILD_SUPPRESSED`
- `REBUILD_DEFERRED`
- `REBUILD_FORCED_DISPATCH`（Phase 0時点で未使用なら未出力で可、予約名だけ確保）
- `REBUILD_DISPATCHED`

共通フィールド（ログ payload）:

- `intentId`（Phase 0は暫定採番で可。例: monotonically increasing）
- `reason`
- `class`（Phase 0は暫定カテゴリで可）
- `policy`（Phase 0は `N/A` 可）
- `hash`
- `fingerprint`
- `finalizeState`
- `decision`
- `latencyMs`（計測可能箇所のみ）

### 3.1 実装済み定義一覧（固定値スキーマ）

実装参照: `src/audioengine/AudioEngine.h`
（`RebuildTelemetryEvent / Reason / Class / Policy / Decision` と `toTelemetry*String`）

#### Event（固定）

- `Requested` → `REBUILD_REQUESTED`
- `Merged` → `REBUILD_MERGED`
- `Suppressed` → `REBUILD_SUPPRESSED`
- `Deferred` → `REBUILD_DEFERRED`
- `ForcedDispatch` → `REBUILD_FORCED_DISPATCH`
- `Dispatched` → `REBUILD_DISPATCHED`

#### Class（固定）

- `NA` → `N/A`
- `Structural` → `Structural`
- `FinalizeAware` → `FinalizeAware`
- `Snapshot` → `Snapshot`

#### Policy（固定）

- `NA` → `N/A`
- `Replaceable` → `Replaceable`
- `MustExecute` → `MustExecute`

#### Decision（固定）

- `Accepted` → `accepted`
- `Suppressed` → `suppressed`
- `Deferred` → `deferred`
- `Dispatched` → `dispatched`
- `Merged` → `merged`
- `Dropped` → `dropped`
- `Released` → `released`

#### Reason（固定）

- `ConvolverParamsChanged` → `convolver_params_changed`
- `MixedPhaseIntermediate` → `mixed_phase_intermediate`
- `HashDedup` → `hash_dedup`
- `PreparedIRApplyWindow` → `prepared_ir_apply_window`
- `SnapshotEnqueueFailed` → `snapshot_enqueue_failed`
- `SnapshotEnqueued` → `snapshot_enqueued`
- `RequestRebuildKindEntry` → `requestRebuild_kind_entry`
- `UiEqEditorChangeListener` → `ui_eq_editor_change_listener`
- `PrepareToPlayNonMt` → `prepare_to_play_non_mt`
- `RebuildThreadWarmupRetry` → `rebuild_thread_warmup_retry`
- `ShutdownInProgress` → `shutdown_in_progress`
- `KindFiltered` → `kind_filtered`
- `DelegateRequestRebuildSrBs` → `delegate_requestRebuild_sr_bs`
- `MissingSrBs` → `missing_sr_bs`
- `NonMtTriggerAsync` → `non_mt_trigger_async`
- `NonMtAlreadyPending` → `non_mt_already_pending`
- `AsyncBridgeConsume` → `async_bridge_consume`
- `AsyncBridgeDelegateSrBs` → `async_bridge_delegate_sr_bs`
- `AsyncBridgeMissingSrBs` → `async_bridge_missing_sr_bs`
- `RequestRebuildSrBs` → `requestRebuild_sr_bs`
- `DeferredStructuralWindow` → `deferred_structural_window`
- `TaskQueued` → `task_queued`
- `RecentDuplicate` → `recent_duplicate`
- `PendingDuplicate` → `pending_duplicate`
- `DeferredStructuralDue` → `deferred_structural_due`
- `DeferredStructuralRebuildRequested` → `deferred_structural_rebuild_requested`
- `DeferredFinalizeReady` → `deferred_finalize_ready`
- `DeferredFinalizeRebuildRequested` → `deferred_finalize_rebuild_requested`
- `EnqueueSnapshotCommand` → `enqueue_snapshot_command`
- `SnapshotIntentDebounced` → `snapshot_intent_debounced`
- `SnapshotCommandBufferFull` → `snapshot_command_buffer_full`
- `SnapshotCommandQueued` → `snapshot_command_queued`
- `SnapshotCommandBufferFullNonMt` → `snapshot_command_buffer_full_non_mt`
- `SnapshotCommandQueuedNonMt` → `snapshot_command_queued_non_mt`
- `SameAsPendingWouldMerge` → `same_as_pending_would_merge`

### 3.2 関数 × Reason マッピング（実装反映済み）

> 対象コードは 2026-05-23 時点の `AudioEngine` 実装に一致。

| 関数 | ファイル | 出力する `Reason`（enum） | 文字列表現 |
| --- | --- | --- | --- |
| `AudioEngine::convolverParamsChanged(ConvolverProcessor*)` | `src/audioengine/AudioEngine.UIEvents.cpp` | `ConvolverParamsChanged`, `MixedPhaseIntermediate`, `HashDedup`, `PreparedIRApplyWindow`, `SnapshotEnqueueFailed`, `SnapshotEnqueued` | `convolver_params_changed`, `mixed_phase_intermediate`, `hash_dedup`, `prepared_ir_apply_window`, `snapshot_enqueue_failed`, `snapshot_enqueued` |
| `AudioEngine::changeListenerCallback(juce::ChangeBroadcaster*)`（`uiEqEditor` 分岐） | `src/audioengine/AudioEngine.UIEvents.cpp` | `UiEqEditorChangeListener` | `ui_eq_editor_change_listener` |
| `AudioEngine::prepareToPlay(...)`（非MT分岐） | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | `PrepareToPlayNonMt` | `prepare_to_play_non_mt` |
| `AudioEngine::requestRebuild(convo::RebuildKind)` | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `RequestRebuildKindEntry`, `ShutdownInProgress`, `KindFiltered`, `DelegateRequestRebuildSrBs`, `MissingSrBs`, `NonMtTriggerAsync`, `NonMtAlreadyPending` | `requestRebuild_kind_entry`, `shutdown_in_progress`, `kind_filtered`, `delegate_requestRebuild_sr_bs`, `missing_sr_bs`, `non_mt_trigger_async`, `non_mt_already_pending` |
| `AudioEngine::handleAsyncUpdate()` | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `AsyncBridgeConsume`, `AsyncBridgeDelegateSrBs`, `AsyncBridgeMissingSrBs` | `async_bridge_consume`, `async_bridge_delegate_sr_bs`, `async_bridge_missing_sr_bs` |
| `AudioEngine::rebuildThreadLoop()`（warmup retry） | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `RebuildThreadWarmupRetry` | `rebuild_thread_warmup_retry` |
| `AudioEngine::submitRebuildIntent(...)`（Phase 2: logging-only） | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `SameAsPendingWouldMerge` | `same_as_pending_would_merge` |
| `AudioEngine::requestRebuild(double sampleRate, int samplesPerBlock)` | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `RequestRebuildSrBs`, `ShutdownInProgress`, `MixedPhaseIntermediate`, `TaskQueued`, `PendingDuplicate` | `requestRebuild_sr_bs`, `shutdown_in_progress`, `mixed_phase_intermediate`, `task_queued`, `pending_duplicate` |
| `AudioEngine::timerCallback()`（defer解放経路） | `src/audioengine/AudioEngine.Timer.cpp` | `DeferredStructuralDue`, `DeferredStructuralRebuildRequested`, `DeferredFinalizeReady`, `DeferredFinalizeRebuildRequested` | `deferred_structural_due`, `deferred_structural_rebuild_requested`, `deferred_finalize_ready`, `deferred_finalize_rebuild_requested` |
| `AudioEngine::enqueueSnapshotCommand()` | `src/audioengine/AudioEngine.Init.cpp` | `EnqueueSnapshotCommand`, `SnapshotIntentDebounced`, `SnapshotCommandBufferFull`, `SnapshotCommandQueued`, `SnapshotCommandBufferFullNonMt`, `SnapshotCommandQueuedNonMt` | `enqueue_snapshot_command`, `snapshot_intent_debounced`, `snapshot_command_buffer_full`, `snapshot_command_queued`, `snapshot_command_buffer_full_non_mt`, `snapshot_command_queued_non_mt` |

### 3.3 関数 × Reason × Event/Decision（3軸マッピング）

| 関数 | Reason | Event | Decision |
| --- | --- | --- | --- |
| `convolverParamsChanged` | `ConvolverParamsChanged` | `Requested` | `Accepted` |
| `convolverParamsChanged` | `MixedPhaseIntermediate` | `Suppressed` | `Suppressed` |
| `convolverParamsChanged` | `HashDedup` | `Suppressed` | `Suppressed` |
| `convolverParamsChanged` | `PreparedIRApplyWindow` | `Deferred` | `Deferred` |
| `convolverParamsChanged` | `SnapshotEnqueueFailed` | `Suppressed` | `Dropped` |
| `convolverParamsChanged` | `SnapshotEnqueued` | `Dispatched` | `Dispatched` |
| `changeListenerCallback(uiEqEditor)` | `UiEqEditorChangeListener` | `Requested` | `Accepted` |
| `prepareToPlay(non-MT)` | `PrepareToPlayNonMt` | `Requested` | `Accepted` |
| `requestRebuild(convo::RebuildKind)` | `RequestRebuildKindEntry` | `Requested` | `Accepted` |
| `requestRebuild(convo::RebuildKind)` | `ShutdownInProgress` | `Suppressed` | `Suppressed` |
| `requestRebuild(convo::RebuildKind)` | `KindFiltered` | `Suppressed` | `Suppressed` |
| `requestRebuild(convo::RebuildKind)` | `DelegateRequestRebuildSrBs` | `Dispatched` | `Dispatched` |
| `requestRebuild(convo::RebuildKind)` | `MissingSrBs` | `Deferred` | `Deferred` |
| `requestRebuild(convo::RebuildKind)` | `NonMtTriggerAsync` | `Dispatched` | `Dispatched` |
| `requestRebuild(convo::RebuildKind)` | `NonMtAlreadyPending` | `Merged` | `Merged` |
| `handleAsyncUpdate` | `AsyncBridgeConsume` | `Requested` | `Accepted` |
| `handleAsyncUpdate` | `AsyncBridgeDelegateSrBs` | `Dispatched` | `Dispatched` |
| `handleAsyncUpdate` | `AsyncBridgeMissingSrBs` | `Deferred` | `Deferred` |
| `rebuildThreadLoop(retry)` | `RebuildThreadWarmupRetry` | `Requested` | `Accepted` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs` | `Requested` | `Accepted` |
| `requestRebuild(double,int)` | `ShutdownInProgress` | `Suppressed` | `Suppressed` |
| `requestRebuild(double,int)` | `MixedPhaseIntermediate` | `Suppressed` | `Suppressed` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs → TaskQueued` | `Requested → Dispatched` | `Accepted → Dispatched` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs → PendingDuplicate` | `Requested → Merged` | `Accepted → Merged` |
| `timerCallback` | `DeferredStructuralDue → DeferredStructuralRebuildRequested` | `Deferred → Dispatched` | `Released → Dispatched` |
| `timerCallback` | `DeferredFinalizeReady → DeferredFinalizeRebuildRequested` | `Deferred → Dispatched` | `Released → Dispatched` |
| `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotIntentDebounced` | `Requested → Suppressed` | `Accepted → Suppressed` |
| `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotCommandQueued / SnapshotCommandBufferFull` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` |
| `enqueueSnapshotCommand(NonMT)` | `EnqueueSnapshotCommand → SnapshotCommandQueuedNonMt / SnapshotCommandBufferFullNonMt` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` |

### 3.4 関数 × Reason × Class × Policy × Event/Decision（5軸マッピング）

| 関数 | Reason | Class | Policy | Event | Decision | ソース（file:line） |
| --- | --- | --- | --- | --- | --- | --- |
| `convolverParamsChanged` | `ConvolverParamsChanged` | `NA` | `NA` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.UIEvents.cpp:36` |
| `convolverParamsChanged` | `MixedPhaseIntermediate` | `NA` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.UIEvents.cpp:53` |
| `convolverParamsChanged` | `HashDedup` | `Structural` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.UIEvents.cpp:99` |
| `convolverParamsChanged` | `PreparedIRApplyWindow` | `Structural` | `NA` | `Deferred` | `Deferred` | `src/audioengine/AudioEngine.UIEvents.cpp:131` |
| `convolverParamsChanged` | `SnapshotEnqueueFailed` | `Structural` | `NA` | `Suppressed` | `Dropped` | `src/audioengine/AudioEngine.UIEvents.cpp:150` |
| `convolverParamsChanged` | `SnapshotEnqueued` | `Structural` | `NA` | `Dispatched` | `Dispatched` | `src/audioengine/AudioEngine.UIEvents.cpp:162` |
| `changeListenerCallback(uiEqEditor)` | `UiEqEditorChangeListener` | `Structural` | `Replaceable` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.UIEvents.cpp:22` |
| `prepareToPlay(non-MT)` | `PrepareToPlayNonMt` | `Structural` | `Replaceable` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:241` |
| `requestRebuild(convo::RebuildKind)` | `RequestRebuildKindEntry` | `Structural` | `Replaceable` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:198` |
| `requestRebuild(convo::RebuildKind)` | `ShutdownInProgress` | `NA` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:61` |
| `requestRebuild(convo::RebuildKind)` | `KindFiltered` | `NA` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:70` |
| `requestRebuild(convo::RebuildKind)` | `DelegateRequestRebuildSrBs` | `Structural` | `NA` | `Dispatched` | `Dispatched` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:89` |
| `requestRebuild(convo::RebuildKind)` | `MissingSrBs` | `FinalizeAware` | `NA` | `Deferred` | `Deferred` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:102` |
| `requestRebuild(convo::RebuildKind)` | `NonMtTriggerAsync` | `Structural` | `NA` | `Dispatched` | `Dispatched` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:118` |
| `requestRebuild(convo::RebuildKind)` | `NonMtAlreadyPending` | `Structural` | `NA` | `Merged` | `Merged` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:130` |
| `handleAsyncUpdate` | `AsyncBridgeConsume` | `Structural` | `NA` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:152` |
| `handleAsyncUpdate` | `AsyncBridgeDelegateSrBs` | `Structural` | `NA` | `Dispatched` | `Dispatched` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:165` |
| `handleAsyncUpdate` | `AsyncBridgeMissingSrBs` | `FinalizeAware` | `NA` | `Deferred` | `Deferred` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:178` |
| `rebuildThreadLoop(retry)` | `RebuildThreadWarmupRetry` | `Structural` | `Replaceable` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:599` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs` | `Structural` | `NA` | `Requested` | `Accepted` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:201` |
| `requestRebuild(double,int)` | `ShutdownInProgress` | `Structural` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:219` |
| `requestRebuild(double,int)` | `MixedPhaseIntermediate` | `Structural` | `NA` | `Suppressed` | `Suppressed` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:268` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs → TaskQueued` | `Structural` | `NA` | `Requested → Dispatched` | `Accepted → Dispatched` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:201,372` |
| `requestRebuild(double,int)` | `RequestRebuildSrBs → PendingDuplicate` | `Structural` | `NA` | `Requested → Merged` | `Accepted → Merged` | `src/audioengine/AudioEngine.RebuildDispatch.cpp` |
| `timerCallback` | `【共通維持】DeferredStructuralDue → DeferredStructuralRebuildRequested` | `Structural` | `NA → Replaceable` | `Deferred → Dispatched` | `Released → Dispatched` | `src/audioengine/AudioEngine.Timer.cpp:207,215` |
| `timerCallback` | `【共通維持】DeferredFinalizeReady → DeferredFinalizeRebuildRequested` | `FinalizeAware` | `NA → MustExecute` | `Deferred/ForcedDispatch → Dispatched` | `Released/Dispatched → Dispatched` | `src/audioengine/AudioEngine.Timer.cpp:291,305,315` |
| `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotIntentDebounced` | `Snapshot` | `NA` | `Requested → Suppressed` | `Accepted → Suppressed` | `src/audioengine/AudioEngine.Init.cpp:90,137` |
| `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotCommandQueued / SnapshotCommandBufferFull` | `Snapshot` | `NA` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` | `src/audioengine/AudioEngine.Init.cpp:90,166,153` |
| `enqueueSnapshotCommand(NonMT)` | `EnqueueSnapshotCommand → SnapshotCommandQueuedNonMt / SnapshotCommandBufferFullNonMt` | `Snapshot` | `NA` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` | `src/audioengine/AudioEngine.Init.cpp:90,192,182` |

補足:

- 同じ `Reason` が複数関数から出る設計を許容している（例: `MixedPhaseIntermediate`, `ShutdownInProgress`）。
- Phase 1 以降の `submitRebuildIntent(...)` 導入時は、本表を起点に `Reason -> Class/Policy` の責務整理を行う。
- Deferred 系は現行で `timerCallback` 単一入口のため、5軸表では `【共通維持】` ラベルのみを残し、
   入口分離の議論は 3.5.1（圧縮表）へ集約した。

### 3.5 共通Reasonの分類（入口共通 / 入口分離）

分類基準（今回適用）:

1. **入口共通でよい**
   - 判定条件が「入口に依存しない同一ガード条件」である。
   - 横断集計（運用SLO・異常率監視）で意味がぶれない。
2. **入口分離すべき**
   - Reason が「どの入口で発火したか」を含意しないと分析価値が落ちる。
   - 同じガード名でも、入口ごとに対処方針が異なる。

| Reason | 分類 | 判定 | 根拠 | 該当ソース（file:line） |
| --- | --- | --- | --- | --- |
| `ShutdownInProgress` | 入口共通 | **共通維持** | シャットダウン中抑止は入口非依存のグローバルガード。運用上は総量監視が主目的。 | `src/audioengine/AudioEngine.RebuildDispatch.cpp:68`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:234` |
| `KindFiltered` | 入口共通 | **共通維持** | `RebuildKind::None/Runtime` のフィルタは admission 入口の共通規約。入口別分解の利得が低い。 | `src/audioengine/AudioEngine.RebuildDispatch.cpp:77` |
| `MixedPhaseIntermediate` | 入口共通 | **共通維持** | 同一ドメイン条件（progressive mixed-phase 中間状態）での抑止。入口横断で同一指標として扱う価値が高い。 | `src/audioengine/AudioEngine.UIEvents.cpp:56`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:283` |
| `DeferredStructuralDue` / `DeferredStructuralRebuildRequested` | 入口共通 | **共通維持** | defer解放の2段遷移（`Deferred -> Dispatched`）を表すペア。現状は `timerCallback` 単一入口で完結し、入口別に分割する利得が低い。 | `src/audioengine/AudioEngine.Timer.cpp:207,217` |
| `DeferredFinalizeReady` / `DeferredFinalizeRebuildRequested` | 入口共通 | **共通維持** | finalize-aware解放の2段遷移ペア。現状は単一入口で、系列として監視する方が運用しやすい。 | `src/audioengine/AudioEngine.Timer.cpp:273,283` |
| `EnqueueSnapshotCommand` / `SnapshotIntentDebounced` | 入口共通 | **共通維持** | Snapshot intent 受付→debounce抑止の系列。MT入口に限定され、系列監視の価値が高い。 | `src/audioengine/AudioEngine.Init.cpp:90,137` |
| `SnapshotCommandQueued` / `SnapshotCommandBufferFull` | 入口分離候補 | **現状分離維持（MT）** | MT経路での admission 成否（queued/full）を示す。将来入口増加時は `...FromMT` 命名を検討。 | `src/audioengine/AudioEngine.Init.cpp:166,153` |
| `SnapshotCommandQueuedNonMt` / `SnapshotCommandBufferFullNonMt` | 入口分離 | **分離維持（NonMT）** | NonMTブリッジ経路の admission 成否。MT系と混在させるとボトルネック分析が難化。 | `src/audioengine/AudioEngine.Init.cpp:192,182` |
| `RequestRebuildKindEntry` / `UiEqEditorChangeListener` / `PrepareToPlayNonMt` / `RebuildThreadWarmupRetry` | 入口分離 | **分離維持** | 入口起点の追跡が主目的。共通化すると「どこから来た intent か」が埋没する。 | `src/audioengine/AudioEngine.RebuildDispatch.cpp:198`, `src/audioengine/AudioEngine.UIEvents.cpp:22`, `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp:241`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:599` |

Deferred* の入口分離候補（将来）:

- 現在は `timerCallback` 単一入口のため **共通維持**。
- 将来、defer 解放を別入口（例: 明示コマンド経路）からも実行するようになった場合に限り、
   `Deferred*` を `...FromTimer` / `...FromCommand` のように入口分離する。

### 3.5.1 Deferred* 圧縮整理（5軸表を簡潔化するための補助表）

| 系列 | 現行分類 | 5軸表での表記方針 | 分離候補（将来のみ） | 判定トリガー |
| --- | --- | --- | --- | --- |
| `DeferredStructuralDue → DeferredStructuralRebuildRequested` | **共通維持** | 1行集約（`【共通維持】` 付与） | `DeferredStructuralDueFromTimer` / `DeferredStructuralDueFromCommand` | timer以外の解放入口が実装された時点 |
| `DeferredFinalizeReady → DeferredFinalizeRebuildRequested` | **共通維持** | 1行集約（`Deferred/ForcedDispatch` を同列で表記） | `DeferredFinalizeReadyFromTimer` / `DeferredFinalizeReadyFromCommand` | finalize解放に複数入口が生じた時点 |

簡潔化ルール:

1. **単一入口（現行）**: `Deferred*` は共通維持として 5軸表を最小行数で保持する。
2. **複数入口（将来）**: 入口が増えた時だけ `...From<entry>` 命名へ分離し、通常時は過分割しない。
3. **運用抽出優先**: 系列監視（`S-DEF-01/02/03`）が読めることを優先し、入口識別は必要発生時に追加する。

運用メモ:

- 現時点では `ShutdownInProgress` は **共通維持** が妥当。
- 将来、入口別のシャットダウン挙動差分を追う必要が出た場合のみ、`ShutdownInProgress*` を入口別に分離する（現段階では過分割を避ける）。

### 3.6 系列サマリー（レビュー/運用向け）

> 3.3/3.4 の圧縮系列を「確認順」で再掲。詳細は各表の file:line を参照。

| 系列ID | 入口 | 主要Reason遷移 | 代表Event遷移 | 代表Decision遷移 | 期待する終端 |
| --- | --- | --- | --- | --- | --- |
| S-REQ-01 | `requestRebuild(double,int)` | `RequestRebuildSrBs → TaskQueued` | `Requested → Dispatched` | `Accepted → Dispatched` | task queue へ投入 |
| S-REQ-02 | `requestRebuild(double,int)` | `RequestRebuildSrBs → PendingDuplicate` | `Requested → Merged` | `Accepted → Merged` | 重複抑止（再投入なし） |
| S-DEF-01 | `timerCallback` | `DeferredStructuralDue → DeferredStructuralRebuildRequested` | `Deferred → Dispatched` | `Released → Dispatched` | defer解放後の再発行 |
| S-DEF-02 | `timerCallback` | `DeferredFinalizeReady → DeferredFinalizeRebuildRequested` | `Deferred → Dispatched` | `Released → Dispatched` | finalize-aware解放 |
| S-DEF-03 | `timerCallback` | `DeferredFinalizeRebuildRequested(timeout)` | `ForcedDispatch → Requested` | `Dispatched → Accepted` | finalize defer timeout による強制再発行 |
| S-SNAP-01 | `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotIntentDebounced` | `Requested → Suppressed` | `Accepted → Suppressed` | intent debounce |
| S-SNAP-02 | `enqueueSnapshotCommand(MT)` | `EnqueueSnapshotCommand → SnapshotCommandQueued / SnapshotCommandBufferFull` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` | queue投入 or full drop |
| S-SNAP-03 | `enqueueSnapshotCommand(NonMT)` | `EnqueueSnapshotCommand → SnapshotCommandQueuedNonMt / SnapshotCommandBufferFullNonMt` | `Requested → Dispatched / Suppressed` | `Accepted → Dispatched / Dropped` | NonMT経路の成否判定 |

補足:

- 監査時はまず `S-REQ-*` / `S-DEF-*` / `S-SNAP-*` の系列IDで期待遷移を確認し、差分があれば 3.4 の file:line へ降りる。
- これにより「表を全部読む」より先に、異常系列を最短で特定できる。

### 3.7 S-* 系列ID対応 ログ抽出テンプレート（運用定型）

前提:

- 対象ログは `REBUILD_TELEMETRY` 行を含むテキスト（アプリログ/標準出力保存）
- 例では PowerShell の `Select-String` を使用

#### 共通テンプレート

- 単一系列抽出（時系列確認）
  - `Select-String -Path <logfile> -Pattern '<pattern1>\|<pattern2>\|<pattern3>' | Sort-Object LineNumber`
- 件数確認（系列の偏り検知）
  - `Select-String -Path <logfile> -Pattern '<pattern>' | Measure-Object`

#### 系列ID別テンプレート

| 系列ID | 目的 | 推奨パターン（Regex） |
| --- | --- | --- |
| `S-REQ-01` | 正常 queue 投入系列 | `event=REBUILD_REQUESTED.*reason=requestRebuild_sr_bs\|event=REBUILD_DISPATCHED.*reason=task_queued` |
| `S-REQ-02` | 重複抑止系列（Pending） | `event=REBUILD_REQUESTED.*reason=requestRebuild_sr_bs\|event=REBUILD_MERGED.*reason=pending_duplicate` |
| `S-DEF-01/02/03`（共通維持） | Deferred 系列の統合抽出（`timerCallback` 単一入口） | `event=REBUILD_DEFERRED.*reason=deferred_structural_due\|event=REBUILD_DISPATCHED.*reason=deferred_structural_rebuild_requested\|event=REBUILD_DEFERRED.*reason=deferred_finalize_ready\|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested\|event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested\|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested` |
| `S-SNAP-01` | Snapshot debounce 系列 | `event=REBUILD_REQUESTED.*reason=enqueue_snapshot_command\|event=REBUILD_SUPPRESSED.*reason=snapshot_intent_debounced` |
| `S-SNAP-02` | Snapshot MT 成否系列 | `event=REBUILD_REQUESTED.*reason=enqueue_snapshot_command\|event=REBUILD_DISPATCHED.*reason=snapshot_command_queued\|event=REBUILD_SUPPRESSED.*reason=snapshot_command_buffer_full` |
| `S-SNAP-03` | Snapshot NonMT 成否系列 | `event=REBUILD_REQUESTED.*reason=enqueue_snapshot_command\|event=REBUILD_DISPATCHED.*reason=snapshot_command_queued_non_mt\|event=REBUILD_SUPPRESSED.*reason=snapshot_command_buffer_full_non_mt` |

#### S-DEF 共通維持テンプレ（短文化版）

> 現行は Deferred 系が `timerCallback` 単一入口のため、S-DEF-01/02/03 は「まず統合抽出してから系列別に数える」運用を標準とする。

```powershell
$SDefPattern = 'event=REBUILD_DEFERRED.*reason=deferred_structural_due|event=REBUILD_DISPATCHED.*reason=deferred_structural_rebuild_requested|event=REBUILD_DEFERRED.*reason=deferred_finalize_ready|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested|event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested'

# 1) S-DEF 全系列を時系列で一括抽出
Select-String -Path <logfile> -Pattern $SDefPattern | Sort-Object LineNumber

# 2) 系列別件数（統合抽出の後段でカウント）
$sdef01 = (Select-String -Path <logfile> -Pattern 'reason=deferred_structural_due|reason=deferred_structural_rebuild_requested').Count
$sdef02 = (Select-String -Path <logfile> -Pattern 'reason=deferred_finalize_ready|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested').Count
$sdef03 = (Select-String -Path <logfile> -Pattern 'event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested').Count
```

#### すぐ使える例（PowerShell）

- `S-REQ-02`（重複抑止）の時系列抽出
  - `Select-String -Path <logfile> -Pattern 'event=REBUILD_REQUESTED.*reason=requestRebuild_sr_bs\|event=REBUILD_MERGED.*reason=pending_duplicate' | Sort-Object LineNumber`
- `S-SNAP-03`（NonMT）の件数確認
  - `Select-String -Path <logfile> -Pattern 'reason=snapshot_command_queued_non_mt\|reason=snapshot_command_buffer_full_non_mt' | Measure-Object`
- `S-DEF-01/02/03`（共通維持）の一括抽出: `Select-String -Path <logfile> -Pattern 'event=REBUILD_DEFERRED.*reason=deferred_structural_due\|event=REBUILD_DISPATCHED.*reason=deferred_structural_rebuild_requested\|event=REBUILD_DEFERRED.*reason=deferred_finalize_ready\|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested\|event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested\|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested' | Sort-Object LineNumber`

#### 閾値テンプレート（5分窓）

前提:

- `<windowLog>` は「直近5分だけ」を含むログファイル（または5分ローテーション単位ファイル）
- `N` は許容上限（運用で調整）

`buffer_full_non_mt > N` なら警告（S-SNAP-03）:

```powershell
$N = 20
$count = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_buffer_full_non_mt').Count
if ($count -gt $N) {
   Write-Warning "S-SNAP-03 threshold exceeded: buffer_full_non_mt=$count (> $N)"
} else {
   Write-Output "OK: buffer_full_non_mt=$count (<= $N)"
}
```

抑止率テンプレート（任意）:

```powershell
$queued = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_queued_non_mt').Count
$full = (Select-String -Path <windowLog> -Pattern 'reason=snapshot_command_buffer_full_non_mt').Count
$total = $queued + $full
if ($total -gt 0) {
   $ratio = [math]::Round(($full / $total) * 100, 2)
   Write-Output "S-SNAP-03 full_ratio=${ratio}% ($full/$total)"
}
```

`S-REQ-02`（PendingDuplicate比率）閾値テンプレート:

```powershell
$WarnRatio = 60.0   # [%] 閾値（運用で調整）
$mergedCount = (Select-String -Path <windowLog> -Pattern 'reason=pending_duplicate').Count
$requested = (Select-String -Path <windowLog> -Pattern 'reason=requestRebuild_sr_bs').Count

if ($requested -gt 0) {
   $ratio = [math]::Round(($mergedCount / $requested) * 100, 2)
   if ($ratio -gt $WarnRatio) {
      Write-Warning "S-REQ-02 threshold exceeded: pending_duplicate_ratio=${ratio}% (> ${WarnRatio}%)"
   } else {
      Write-Output "OK: S-REQ-02 pending_duplicate_ratio=${ratio}% (<= ${WarnRatio}%)"
   }
}
```

`S-DEF-01/02`（Deferred→Dispatched 解放遅延）閾値テンプレート:

```powershell
# 前提: ログ各行に "HH:mm:ss.fff" 形式の時刻が含まれること
$WarnDelayMs = 3000  # [ms] 閾値（運用で調整）

$deferred = Select-String -Path <windowLog> -Pattern 'reason=deferred_structural_due|reason=deferred_finalize_ready'
$dispatched = Select-String -Path <windowLog> -Pattern 'reason=deferred_structural_rebuild_requested|reason=deferred_finalize_rebuild_requested'

$toMs = {
   param([string]$line)
   if ($line -match '(?<ts>\d{2}:\d{2}:\d{2}\.\d{3})') {
      return [datetime]::ParseExact($Matches['ts'], 'HH:mm:ss.fff', $null)
   }
   return $null
}

$maxDelay = -1
foreach ($d in $deferred) {
   $dt = & $toMs $d.Line
   if ($null -eq $dt) { continue }

   $next = $dispatched | Where-Object { $_.LineNumber -gt $d.LineNumber } | Select-Object -First 1
   if ($null -eq $next) { continue }

   $nt = & $toMs $next.Line
   if ($null -eq $nt) { continue }

   $delay = ($nt - $dt).TotalMilliseconds
   if ($delay -gt $maxDelay) { $maxDelay = $delay }
}

if ($maxDelay -gt $WarnDelayMs) {
   Write-Warning "S-DEF threshold exceeded: max_release_delay_ms=$maxDelay (> $WarnDelayMs)"
} else {
   Write-Output "OK: S-DEF max_release_delay_ms=$maxDelay (<= $WarnDelayMs)"
}
```

`S-DEF-03`（Finalize defer timeout 強制発行回数）閾値テンプレート:

```powershell
$WarnForced = 5   # [count / 5min] 閾値（運用で調整）
$forced = (Select-String -Path <windowLog> -Pattern 'event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested').Count

if ($forced -gt $WarnForced) {
   Write-Warning "S-DEF-03 threshold exceeded: forced_dispatch_count=$forced (> $WarnForced)"
} else {
   Write-Output "OK: S-DEF-03 forced_dispatch_count=$forced (<= $WarnForced)"
}
```

`S-DEF-03`（ForcedDispatch 発生率）閾値テンプレート（任意）:

```powershell
$WarnRatio = 10.0  # [%] 閾値（運用で調整）
$forced = (Select-String -Path <windowLog> -Pattern 'event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested').Count
$finalizeReq = (Select-String -Path <windowLog> -Pattern 'reason=deferred_finalize_rebuild_requested').Count

if ($finalizeReq -gt 0) {
   $ratio = [math]::Round(($forced / $finalizeReq) * 100, 2)
   if ($ratio -gt $WarnRatio) {
      Write-Warning "S-DEF-03 threshold exceeded: forced_dispatch_ratio=${ratio}% (> ${WarnRatio}%)"
   } else {
      Write-Output "OK: S-DEF-03 forced_dispatch_ratio=${ratio}% (<= ${WarnRatio}%)"
   }
}
```

時刻抽出が難しい場合の簡易フォールバック（行差分）:

```powershell
$WarnLineGap = 200  # 行差分の閾値（運用で調整）

$deferred = Select-String -Path <windowLog> -Pattern 'reason=deferred_structural_due|reason=deferred_finalize_ready'
$dispatched = Select-String -Path <windowLog> -Pattern 'reason=deferred_structural_rebuild_requested|reason=deferred_finalize_rebuild_requested'

$maxGap = -1
foreach ($d in $deferred) {
   $next = $dispatched | Where-Object { $_.LineNumber -gt $d.LineNumber } | Select-Object -First 1
   if ($null -eq $next) { continue }
   $gap = $next.LineNumber - $d.LineNumber
   if ($gap -gt $maxGap) { $maxGap = $gap }
}

if ($maxGap -gt $WarnLineGap) {
   Write-Warning "S-DEF fallback threshold exceeded: max_line_gap=$maxGap (> $WarnLineGap)"
} else {
   Write-Output "OK: S-DEF max_line_gap=$maxGap (<= $WarnLineGap)"
}
```

運用ルール（推奨）:

- まず `S-REQ-01` と `S-REQ-02` の比率を確認し、重複抑止が急増していないか監視する。
- `S-SNAP-03` の `buffer_full_non_mt` が閾値超過した場合は、NonMT 発火頻度と command buffer 容量を優先点検する。
- `S-DEF-*` は共通維持前提で「統合抽出 → 系列別カウント」の順に確認する。
- `S-DEF-01/02` は Deferred→Dispatched の遅延（ms）または行差分が閾値超過していないかを併せて監視する。
- `S-DEF-03` は `REBUILD_FORCED_DISPATCH` の回数/発生率を監視し、timeout 強制発行の常態化を早期検知する。
- 異常系列を検出したら、3.4 の `file:line` へ降りて該当分岐を確認する。

関連Runbook（1ページ）:

- `doc/work/ISR_Rebuild_Admission_Runbook_Index_2026-05-23.md`
- `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md`
- `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`
- `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md`

---

## 4. 変更順（実装オーダー）

### Step 1: 共通テレメトリ出力面の追加（関数外影響なし）

対象:

- `AudioEngine` 内にログ整形 helper を追加。
- event 名、decision 名を enum/string で固定化。

完了判定:

- 既存ログと混在しても grep で telemetry 行だけ抽出可能。

### Step 2: 入口 request ログの挿入

対象関数:

- `convolverParamsChanged`
- `requestRebuild(convo::RebuildKind)`
- `enqueueSnapshotCommand`

挿入ポイント:

- 「要求受理直後」
- 「hash dedup / debounce による抑止判定直前・直後」

出力イベント:

- `REBUILD_REQUESTED`
- `REBUILD_SUPPRESSED`

### Step 3: 非MT→MTブリッジ観測ログ

対象関数:

- `handleAsyncUpdate`

挿入ポイント:

- `StructuralFromNonMT` clear 成功時
- sr/bs 不足で `DeferredFinalizeAware` へフォールバックする分岐

出力イベント:

- `REBUILD_REQUESTED`（bridge受理）
- `REBUILD_DEFERRED`

### Step 4: dispatch 本線ログ

対象関数:

- `requestRebuild(double, int)`

挿入ポイント:

- duplicate pending 判定
- recent duplicate 判定
- queued 成功
- deferred structural window suppress
- intermediate mixed-phase suppress

出力イベント:

- `REBUILD_SUPPRESSED`
- `REBUILD_DISPATCHED`（queued 時）
- `REBUILD_MERGED`（Phase 0では「同一判定観測」のみ。挙動は変えない）

### Step 5: timer 側 defer 解放ログ

対象関数:

- `timerCallback`

挿入ポイント:

- `DeferredStructural` due 到達で rebuild 発行時
- `DeferredFinalizeAware` 条件成立で rebuild 発行時

出力イベント:

- `REBUILD_DEFERRED`（待機状態の観測）
- `REBUILD_DISPATCHED`（解放後の実発行）

### Step 6: 既存 debug counter 連携

対象:

- 既存 `debugRebuildDispatch*` 系カウンタに telemetry の event/decision 集計を補完

注意:

- Counter は観測用途のみ。判定条件に使わない。

---

## 5. 関数別タスク詳細（作業単位）

### 5.1 `convolverParamsChanged`

- Task A1: 入口 request telemetry 追加
- Task A2: hash dedup 抑止に `REBUILD_SUPPRESSED` 追加
- Task A3: deferred structural へ遷移する分岐に `REBUILD_DEFERRED` 追加

検証観点:

- 同一 IR 操作で hash dedup が suppress として出る
- prepared IR apply 直後は deferred として出る

### 5.2 `requestRebuild(convo::RebuildKind)` + `handleAsyncUpdate`

- Task B1: NonMT request 受理ログ
- Task B2: MT bridge での消費ログ
- Task B3: sr/bs 不足時の defer ログ

検証観点:

- AudioThread起点相当の要求が bridge 経由で1本の系列として追える

### 5.3 `requestRebuild(double, int)`

- Task C1: duplicate pending / recent duplicate の suppress telemetry
- Task C2: queued 成功時 dispatch telemetry
- Task C3: structural defer window / mixed-phase suppress telemetry

検証観点:

- BLOCKED 系と queued 系が必ずどちらか一方で記録される
- 既存挙動（queue可否）が変更されない

### 5.4 `timerCallback`

- Task D1: DeferredStructural due 到達時 telemetry
- Task D2: DeferredFinalizeAware 解放時 telemetry

検証観点:

- defer解除のタイミングと dispatch 発行が同一 trace で確認できる

### 5.5 `enqueueSnapshotCommand`

- Task E1: debounce identical intent suppress telemetry
- Task E2: command buffer full 時の suppress telemetry（drop観測）

検証観点:

- identical snapshot intent は suppress として明示される

---

## 6. 検証観点（テスト観測シナリオ）

### S1: UI burst（スライダ連続操作）

期待:

- `REBUILD_REQUESTED` が複数
- suppress（dedup/debounce）が可視化
- dispatch は必要回数のみ

### S2: IR load 直後（prepared apply）

期待:

- `REBUILD_DEFERRED` が先に出る
- due 到達後に `REBUILD_DISPATCHED`

### S3: finalize-aware defer

期待:

- finalize 条件未成立中は defer 観測
- 条件成立後に dispatch

### S4: duplicate pending / recent duplicate

期待:

- 両 suppress reason が区別される
- queue 挙動は変更なし

---

## 7. 受入基準（Phase 0専用）

### 7.1 観測性

- [ ] 1操作について requested/suppressed/deferred/dispatched が追える
- [ ] suppress reason が `pending duplicate` / `recent duplicate` / `debounce` / `mixed-phase` で識別可能

### 7.2 非退行（挙動不変）

- [ ] suppress/dispatch の判定条件にコード変更がない
- [ ] rebuild回数が既存から有意に変動しない（±5%以内を目安、同一操作比較）

### 7.3 品質ゲート

- [ ] Debug build pass
- [ ] Release build pass
- [ ] Strict Atomic Dot-Call Scan pass

---

## 8. 実行・確認手順（このリポの task に準拠）

1. `Debug` build
2. `Release` build
3. `Strict Atomic Dot-Call Scan`
4. UI burst / IR load / finalize defer の手動操作でログ採取

ログ抽出観点:

- `REBUILD_` プレフィックスで抽出できること
- 1操作の trace が event 時系列で追えること

---

## 9. リスクと回避策

### リスク R1: ログ追加により可読性低下

回避:

- telemetry ログを固定プレフィックス化
- 既存 `[DIAG]` と分離

### リスク R2: 計測変数が制御へ混入

回避:

- telemetry 専用変数は `decision` に使わない
- review 観点に「観測変数が分岐条件に使われていないこと」を追加

### リスク R3: 意図せぬ性能影響

回避:

- 文字列組み立てを最小化
- 必要に応じて debug flag で出力抑制（ただし Phase 0では観測優先）

---

## 10. Phase 1への引き渡しメモ

Phase 0完了後、以下を材料として Phase 1 へ進む。

- suppress/defer の実測頻度
- 入口ごとの request 発生源分布
- duplicate 判定の実運用ヒット率

これにより `submitRebuildIntent(...)` 導入時のマッピング（reason/class/policy）を安全に確定できる。

---

## 11. 実装PR 1本目の粒度（着手順・コミット分割案）

本PRは **Phase 0 telemetry only を完結** させる。
原則: 「1コミット1目的」「挙動変更ゼロ」「レビューで差分意図が即読できること」。

### 11.1 PR 1 の対象ファイル（最小）

- `src/audioengine/AudioEngine.h`（telemetry helper/最小メンバ追加が必要な場合のみ）
- `src/audioengine/AudioEngine.UIEvents.cpp`
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- `src/audioengine/AudioEngine.Timer.cpp`
- `src/audioengine/AudioEngine.Init.cpp`
- `doc/work/ISR_Rebuild_Admission_Phase0_実装タスク分解_2026-05-23.md`（実績反映）

### 11.2 着手関数順（推奨）

依存が少なく副作用が小さい順に着手する。

1. `enqueueSnapshotCommand()`
   - debounce suppress の観測点として最も独立性が高い。
2. `convolverParamsChanged(...)`
   - UI 入口の request/suppress/defer 観測を追加。
3. `requestRebuild(convo::RebuildKind)`
   - NonMT→MTブリッジ前段の request 観測を追加。
4. `handleAsyncUpdate()`
   - ブリッジ消費点（defer遷移含む）を観測。
5. `requestRebuild(double, int)`
   - duplicate suppress / queued dispatch の主幹観測を追加。
6. `timerCallback()`
   - DeferredStructural / DeferredFinalizeAware 解放時の dispatch 観測を追加。

### 11.3 コミット分割案（PR 1 内）

#### Commit 1: telemetry 基盤追加（共通）

対象:

- 共通ログ整形 helper（event/decision の固定フォーマット）
- 必要最小限の enum/string 追加

含めないもの:

- 各関数への実ログ挿入

レビューポイント:

- ログ文言・キーの安定性
- 制御分岐への影響ゼロ

#### Commit 2: 入口系ログ（UI / Snapshot / NonMT request）

対象関数:

- `enqueueSnapshotCommand()`
- `convolverParamsChanged(...)`
- `requestRebuild(convo::RebuildKind)`

レビューポイント:

- request/suppress/defer の区分が意図通り
- suppress条件自体は不変

#### Commit 3: ブリッジ + dispatch本線ログ

対象関数:

- `handleAsyncUpdate()`
- `requestRebuild(double, int)`

レビューポイント:

- pending duplicate / recent duplicate の識別可否
- queued と blocked の排他性（同一事象で二重記録しない）

#### Commit 4: timer defer 解放ログ

対象関数:

- `timerCallback()`

レビューポイント:

- defer観測とdispatch観測が時系列で追えること
- 既存 timer の rebuild 発火条件を変えていないこと

#### Commit 5: 検証ログ採取結果 + 文書更新

対象:

- 本文書の「検証結果」追記（運用ログの抜粋）
- 必要なら telemetry キー名の微修正（挙動非変更）

レビューポイント:

- 受入基準（7章）を満たす証跡が揃っていること

### 11.4 PR 1 の Definition of Ready（着手前チェック）

- [ ] telemetry キー（`intentId/reason/class/policy/hash/fingerprint/finalizeState/decision/latencyMs`）を固定
- [ ] 既存 `[DIAG]` ログとの住み分けプレフィックスを固定
- [ ] 「挙動変更なし」をコードレビュー観点に明記

### 11.5 PR 1 の Definition of Done（マージ条件）

- [ ] 受入基準 7.1 / 7.2 / 7.3 を満たす
- [ ] Debug / Release / Strict Atomic Dot-Call Scan 通過
- [ ] 4シナリオ（S1〜S4）のログ証跡を添付
- [ ] 次PR（Phase 1: `submitRebuildIntent` 導入）に必要な観測データが揃っている

以上（PR 1スコープ定義完了）。
