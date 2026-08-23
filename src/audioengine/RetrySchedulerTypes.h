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
    SameAsPendingWouldMerge
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
