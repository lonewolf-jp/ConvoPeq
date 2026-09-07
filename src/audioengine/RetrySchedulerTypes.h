#pragma once
// RetrySchedulerTypes.h — D-5-2 Step 1: enum-only extraction from AudioEngine.h
// Contains the 3 rebuild telemetry semantic types used by submitRebuildIntent().
// RebuildKind remains in core/RebuildTypes.h.
// This header includes only <cstdint>; RebuildKind is NOT redefined here.

#include <cstdint>

enum class RebuildTelemetryReason : uint8_t
{
    ConvolverParamsChanged,
    MixedPhaseIntermediate,
    HashDedup,
    PreparedIRApplyWindow,
    SnapshotEnqueueFailed,
    SnapshotEnqueued,
    RequestRebuildKindEntry,
    UiEqEditorChangeListener,
    PrepareToPlayNonMt,
    RebuildThreadWarmupRetry,
    ShutdownInProgress,
    KindFiltered,
    DelegateRequestRebuildSrBs,
    MissingSrBs,
    NonMtTriggerAsync,
    NonMtAlreadyPending,
    AsyncBridgeConsume,
    AsyncBridgeDelegateSrBs,
    AsyncBridgeMissingSrBs,
    RequestRebuildSrBs,
    DeferredStructuralWindow,
    TaskQueued,
    RecentDuplicate,
    PendingDuplicate,
    DeferredStructuralDue,
    DeferredStructuralRebuildRequested,
    DeferredFinalizeReady,
    DeferredFinalizeRebuildRequested,
    EnqueueSnapshotCommand,
    SnapshotIntentDebounced,
    SnapshotCommandBufferFull,
    SnapshotCommandQueued,
    SnapshotCommandBufferFullNonMt,
    SnapshotCommandQueuedNonMt,
    RetirePressureSevere,
    SameAsPendingWouldMerge,
    // ★ D167-5: tryAdmit 失敗（admission Closing/Closed）の会計用。Build 経路の
    //   REQUESTED(accepted) → 無出力消滅を Suppressed(AdmissionClosed) として記録する
    //   （D166 §5 observability defect 修復）。末尾追加 — 既存値の再番号付けなし。
    AdmissionClosed
};

enum class RebuildTelemetryClass : uint8_t
{
    NA,
    Structural,
    FinalizeAware,
    Snapshot
};

enum class RebuildTelemetryPolicy : uint8_t
{
    NA,
    Replaceable,
    MustExecute
};
