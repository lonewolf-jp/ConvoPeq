# Publication Matrix

- Entry
  - Caller:
  - Target API:
  - State Transition:
  - Visibility Boundary:
  - Thread:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Caller: `runPublicationPrecheckNonRt` / commit path
  - Target API: publication precheck + `retireRuntimePublication(...)`
  - State Transition: precheck -> publish/rollback -> retire intent emission
  - Visibility Boundary: publishAtomic release / consume acquire（intent/backlog metrics）
  - Thread: Control/Message
  - Evidence: grep(`AudioEngine.Commit.cpp:38,258,271-274`) + Serena(pattern lines around precheck/retire) + CodeGraph(read `AudioEngine.Commit.cpp:260-420`)
