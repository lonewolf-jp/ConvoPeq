# Authority Matrix

- Entry
  - Authority Entity:
  - Owner:
  - Authoritative Writer:
  - Authoritative Reader:
  - Update Phase:
  - Allowed Dependencies:
  - Forbidden Dependencies:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Authority Entity: `RuntimeReadView::graph` pointer exposure
  - Owner: AudioEngine runtime view layer
  - Authoritative Writer: `RuntimePublishView` constructor wiring (`AudioEngine.h`)
  - Authoritative Reader: Commit/Processing/Timer call sites via `getRuntimeGraph(...)`
  - Update Phase: runtime view construction/read phase
  - Allowed Dependencies: semantic read contractsへ段階収束（計画準拠）
  - Forbidden Dependencies: 直接authority判定への逆流（RuntimeGraph/EngineRuntime/DSPCore）
  - Evidence: Serena(`AudioEngine.h` RuntimePublishView/RuntimeReadView defs) + grep(call sites) + CodeGraph(read snippet 1388-1438)

- Entry
  - Authority Entity: `RuntimeState::semanticHash`
  - Owner: RuntimeWorld build path (`buildRuntimePublishWorld`)
  - Authoritative Writer: `buildRuntimePublishWorld` 内の semanticHash assignment 群
  - Authoritative Reader: shadow compare / publication precheck / diagnostics
  - Update Phase: publication build/commit
  - Allowed Dependencies: generation/topology/execution/routing/publication/overlap/retire semantic fields
  - Forbidden Dependencies: EngineRuntime direct state, RuntimeGraph direct authority, DSPCore internal mutable state
  - Evidence: grep(`AudioEngine.h:193`, `2550+`) + Serena(semanticHash assignment lines) + CodeGraph(read `AudioEngine.h` around RuntimeState)
