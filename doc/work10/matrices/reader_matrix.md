# Reader Matrix

- Entry
  - Component:
  - Symbol/File:
  - Reads From:
  - Read Type: direct / projection / semantic API
  - Thread: Audio / Message / Worker
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Component: AudioEngine Commit Path
  - Symbol/File: `AudioEngine::commitNewDSP` (`src/audioengine/AudioEngine.Commit.cpp`)
  - Reads From: `RuntimeReadView -> getRuntimeGraph(runtimeReadView)`
  - Read Type: direct
  - Thread: Message/Control
  - Evidence: grep(`AudioEngine.Commit.cpp:632,683,719`) + Serena(pattern search) + CodeGraph(read `AudioEngine.h:1417-1434`)

- Entry
  - Component: Audio Processing Path
  - Symbol/File: `AudioEngine.Processing.AudioBlock.cpp`
  - Reads From: `getRuntimeGraph(runtimeReadViewRef)`
  - Read Type: direct
  - Thread: Audio
  - Evidence: grep(`AudioEngine.Processing.AudioBlock.cpp:122`) + Serena(pattern search) + CodeGraph(read `AudioEngine.h:2292`)
