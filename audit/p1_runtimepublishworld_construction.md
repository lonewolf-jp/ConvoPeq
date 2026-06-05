# P1 Phase1-A 前準備: RuntimePublishWorld 生成箇所全列挙

**AUDIT_DATE**: 2026-06-05
**AUDITOR**: GitHub Copilot (AI Assistant)
**SEARCH_METHOD**: grep (src/**) for `buildRuntimePublishWorld`
**STATUS**: 完了

---

## RuntimePublishWorld 型

`AudioEngine.h:274` — `using RuntimePublishWorld = RuntimeState;`（型エイリアス）

生成は `RuntimeBuilder::buildRuntimePublishWorld()` 経由のみ。デフォルトコンストラクタは禁止（static_assert）。

## 生成箇所一覧

### 1. `AudioEngine.Init.cpp`

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(nullptr, /* runtime */ ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 2. `AudioEngine.Processing.ReleaseResources.cpp:129`

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(nullptr,
    nullptr, convo::TransitionPolicy::SmoothOnly, 0.0, false, nullptr);
coordinator.publishWorld(std::move(worldOwner));
```

### 3. `AudioEngine.Commit.cpp:905` — applyRuntimeCommitFromIntent → commitNewDSP

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(nextDSP, nullptr,
    convo::TransitionPolicy::SmoothOnly, 0.0, false, &sealedSnapshot);
coordinator.publishWorld(std::move(worldOwner));
```

### 4. `AudioEngine.Commit.cpp:942` — changeActiveRuntimeDSP/retireAllDSPAndPublishEmptyWorld

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 5. `AudioEngine.Commit.cpp:997` — publishHardReset

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 6. `AudioEngine.Commit.cpp:1052` — processWithActiveDSPBeforeReset

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(atomicCurrent, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 7. `AudioEngine.Commit.cpp:1221` — commitNewDSP (deferred commit 内)

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(newDSP, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 8. `AudioEngine.Commit.cpp:1400` — refreshActiveRuntime

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(current, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 9. `AudioEngine.Processing.PrepareToPlay.cpp:132`

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentForPublish, ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 10. `AudioEngine.Processing.PrepareToPlay.cpp:244`

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(getActiveRuntimeDSP(), ...);
coordinator.publishWorld(std::move(worldOwner));
```

### 11. `AudioEngine.Timer.cpp:405`

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade, ...);
coordinator.publishWorld(std::move(worldOwner));
```

---

## 総括

- `RuntimePublishWorld` の生成は全11箇所、すべて `RuntimeBuilder::buildRuntimePublishWorld()` 経由。
- 生成後は常に `coordinator.publishWorld()` に渡され、直接の手動構築は存在しない。
- `AudioEngine::publishWorld()` ラッパーを経由する箇所と、coordinator 直接呼び出しの箇所が混在している。
