# Construction Matrix

- Entry
  - Entity:
  - Constructor/Factory/Builder:
  - Publish Path:
  - Retire Path:
  - Owner:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Entity: `RuntimePublishView`
  - Constructor/Factory/Builder: `RuntimePublishView(convo::EpochDomain&, int, const RuntimeGraph*, const TransitionState&)`
  - Publish Path: `makeRuntimePublishView(...)` (`AudioEngine.h`)
  - Retire Path: n/a (view object lifetime)
  - Owner: AudioEngine runtime view helper
  - Evidence: Serena(symbol search) + grep(`AudioEngine.h:1396+`, `2192+`) + CodeGraph(read snippet 1388-1438)

- Entry
  - Entity: `RuntimeReadView`
  - Constructor/Factory/Builder: `RuntimeReadView(RuntimePublishView&&, convo::ObservedRuntime&&)`
  - Publish Path: `makeRuntimeReadView(...)` -> `readAudioRuntimeView/readControlRuntimeView`
  - Retire Path: n/a (scope-bound object)
  - Owner: AudioEngine runtime view helper
  - Evidence: Serena(symbol search) + grep(`AudioEngine.h:1417+`, `2209+`, `2277+`, `2282+`) + CodeGraph(read snippet 1388-1438)

- Entry
  - Entity: `RuntimeState` / `RuntimePublishWorld`
  - Constructor/Factory/Builder: `RuntimeState(BuilderToken)` + `RuntimeState::createForBuilder(...)`
  - Publish Path: `buildRuntimePublishWorld(...)` returns world used by publication coordinator
  - Retire Path: `retireRuntimePublication(...)` + retireRuntimeEx intent/reclaim/quarantine path
  - Owner: AudioEngine runtime publication pipeline
  - Evidence: grep(`AudioEngine.h:117,127,136`, `buildRuntimePublishWorld`) + Serena(pattern lines 116-137, 2446+) + CodeGraph(read `AudioEngine.Commit.cpp:260-420`)
