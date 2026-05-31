# Crossfade Matrix

- Entry
  - Caller:
  - Authority Source:
  - Update Phase:
  - Recovery Path:
  - Thread:
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Caller: `AudioEngine::commitNewDSP`
  - Authority Source: `computeCrossfadeContext(...)` + handle/crossfade runtime helpers
  - Update Phase: commit publication transition
  - Recovery Path: `retireRuntimeEx_.requestRollback()` on semantic mismatch path
  - Thread: Control/Message
  - Evidence: grep(`AudioEngine.Commit.cpp:626,907,940-942,219`) + Serena(pattern result) + CodeGraph(read commit section)

- Entry
  - Caller: crossfade pending publication flags
  - Authority Source: `dspCrossfadePending`, `dspCrossfadeUseDryAsOld`, `firstIrDryCrossfadePending`
  - Update Phase: immediate smooth transition start/stop
  - Recovery Path: pending flag clear and fallback retire path
  - Thread: Control with Audio visibility boundary
  - Evidence: grep(`AudioEngine.Commit.cpp:696-699,747`) + Serena(pattern result) + CodeGraph(commit snippet)
