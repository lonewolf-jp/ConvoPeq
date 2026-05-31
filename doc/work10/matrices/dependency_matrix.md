# Dependency Matrix

- Entry
  - Subject:
  - Depends On:
  - Dependency Kind: semantic / projection / runtime
  - Direction Valid: yes / no
  - Cycle Risk: low / med / high
  - Evidence: grep + CodeGraph + Serena

- Entry
  - Subject: `RuntimeState::kFieldDescriptors` (`AudioEngine.h`)
  - Depends On: `RuntimeState` authority fields + `PublicationSemantic::validateDescriptorSet()`
  - Dependency Kind: semantic
  - Direction Valid: yes
  - Cycle Risk: med（descriptor count=9 と実体フィールド群の乖離監査が必要）
  - Evidence: grep(`AudioEngine.h:195,207,209`) + Serena(pattern lines 194-209) + CodeGraph(read `AudioEngine.h` descriptor/validate section)
