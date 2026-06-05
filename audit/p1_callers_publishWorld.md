# P1 Phase1-A 前準備: publishWorld 呼び出し元全列挙

**AUDIT_DATE**: 2026-06-05
**AUDITOR**: GitHub Copilot (AI Assistant)
**SEARCH_METHOD**: grep (src/**), codegraph (caller graph)
**STATUS**: 完了

---

## publishWorld 定義

`AudioEngine.h:2754` — `AudioEngine::publishWorld(convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner)`
→ `coordinator.publishWorld(std::move(worldOwner))` に委譲。

## 呼び出し元一覧

### 1. `AudioEngine.Init.cpp:54` — ブートストラップ

```cpp
coordinator.publishWorld(std::move(bootstrapWorld));
```

コンテキスト: `AudioEngine::init()` 内。初回の RuntimePublishWorld 生成後、coordinator 経由で即時 publish。

### 2. `AudioEngine.Processing.ReleaseResources.cpp:134` — リソース解放

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: `releaseResources()` 内。null DSP の RuntimePublishWorld をビルドして publish。

### 3. `AudioEngine.Commit.cpp:911` — commitNewDSP（通常 commit）

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: `applyRuntimeCommitFromIntent()` → `commitNewDSP()` 内。新 DSP の通常 publish。

### 4. `AudioEngine.Commit.cpp:948` — changeActiveRuntimeDSP/retireAllDSPAndPublishEmptyWorld

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: 全 DSP retire 後の空ワールド publish。

### 5. `AudioEngine.Commit.cpp:1011` — publishHardReset

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: ハードリセット時の publish。

### 6. `AudioEngine.Commit.cpp:1052` — processWithActiveDSPBeforeReset

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: リセット前処理後の publish。

### 7. `AudioEngine.Commit.cpp:1221` — commitNewDSP（deferred commit 内）

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: `appendPublicationIntentForCommitConsumer` からの defer commit 後、ワールド publish。

### 8. `AudioEngine.Commit.cpp:1400` — refreshActiveRuntime

```cpp
auto worldOwner = worldBuilder.buildRuntimePublishWorld(current, ...);
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: アクティブ runtime リフレッシュ。

### 9. `AudioEngine.Processing.PrepareToPlay.cpp:132` — prepareToPlay（初期化）

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: `prepareToPlay()` 内、初回設定時の publish。

### 10. `AudioEngine.Processing.PrepareToPlay.cpp:244` — prepareToPlay（再設定）

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: `prepareToPlay()` 内、再設定時の publish。

### 11. `AudioEngine.Timer.cpp:405` — タイマー駆動 publish

```cpp
coordinator.publishWorld(std::move(worldOwner));
```

コンテキスト: Timer callback 内、fade 完了後の publish。

---

## 総括

- `publishWorld` への全呼び出しは `coordinator.publishWorld()` 経由で統一されている。
- `AudioEngine::publishWorld()` ラッパー自体は `AudioEngine.h:2754` で定義され、`coordinator.publishWorld()` に委譲するのみ。
- `AudioEngine::publishWorld()` ラッパーを経由しない `coordinator.publishWorld()` 直接呼び出しが 11 箇所存在する。これらは Phase1-B でラッパー統一要検討。
- Phase1-B 完全削除時は、ラッパー `AudioEngine::publishWorld()` も含めて全呼び出しを coordinator 直接呼び出しに統合するか、あるいはラッパー自体を coordinator.publishWorld() への委譲維持という判断も可能。
