# Observe Matrix

- Entry
  - Observer:
  - Source:
  - Access Pattern:
  - Lifetime Guard:
  - Thread:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Observer: Audio block processing
  - Source: `getRuntimeGraph(runtimeReadViewRef)`
  - Access Pattern: direct pointer dereference path
  - Lifetime Guard: runtime read view scope + observed snapshot
  - Thread: Audio
  - Evidence: grep(`AudioEngine.Processing.AudioBlock.cpp:122`) + Serena(pattern result) + CodeGraph(`AudioEngine.h` RuntimeReadView)

- Entry
  - Observer: Timer diagnostics
  - Source: `getRuntimeGraph(runtimeReadView)`
  - Access Pattern: direct runtime graph observation
  - Lifetime Guard: control read view scope
  - Thread: Message/Timer
  - Evidence: grep(`AudioEngine.Timer.cpp:15`) + Serena(pattern result) + CodeGraph(`AudioEngine.h` read helpers)
