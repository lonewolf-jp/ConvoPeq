# Retire Matrix

- Entry
  - Caller:
  - Target API:
  - Eligibility Check:
  - State Transition:
  - Thread:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Caller: `retireRuntimePublication(world)` processing loop
  - Target API: `retireRuntimeEx_.canTransitionRetirePendingToFree` -> reclaim/quarantine
  - Eligibility Check: graceCompleted + pendingIntentOwned + authoritativeOwnershipReleased
  - State Transition: RetirePending -> Free/Reclaim or Quarantine
  - Thread: Control/Message
  - Evidence: grep(`AudioEngine.Commit.cpp:350-367`) + Serena(pattern lines 350+...) + CodeGraph(read `AudioEngine.Commit.cpp:350-367`)
