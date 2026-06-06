# publishWorld() 呼び出し分類

作成日: 2026-06-06
ベース: grep "publishWorld(" 調査結果

---

## 分類結果

| # | ファイル | 行 | カテゴリ | 説明 |
|---|---|---|---|---|
| 1 | `AudioEngine.Init.cpp` | 54 | **LifecyclePublication** | Bootstrap: 初期化時の初回 publish |
| 2 | `AudioEngine.Processing.ReleaseResources.cpp` | 134 | **LifecyclePublication** | Shutdown: リソース解放時の publish |
| 3 | `AudioEngine.Processing.PrepareToPlay.cpp` | 137 | **TransitionPublication** | PrepareToPlay: サンプルレート変更時の特殊 publish |
| 4 | `AudioEngine.Processing.PrepareToPlay.cpp` | 249 | **TransitionPublication** | PrepareToPlay: 2箇所目の特殊 publish |
| 5 | `AudioEngine.Timer.cpp` | 411 | **RuntimePublication** | Timer: 定期処理からの publish (通常DSP切替) |
| 6 | `AudioEngine.Commit.cpp` | 618 | **RuntimePublication** | Commit: RebuildDispatch 経由の publish |
| 7 | `PublicationExecutor::publish()` | (内部) | **内部委譲** | Executor 内蔵の coordinator.publishWorld() |
| 8-11 | `PartialPublicationRejectTests.cpp` | 181,189,237,242,293 | **(テスト)** | テストコード、本番対象外 |

## カテゴリ別方針 (PR-4)

| カテゴリ | 該当数 | 方針 |
|---|---|---|
| **RuntimePublication** | 2 (Timer, Commit) | `submitPublishRequest()` 経由に変更 |
| **LifecyclePublication** | 2 (Bootstrap, Shutdown) | 専用経路維持 (通常RuntimePublicationと分離) |
| **TransitionPublication** | 2 (PrepareToPlay) | 専用経路維持 (通常RuntimePublicationと分離) |
| **内部委譲** | 1 (Executor) | Coordinator経由維持 (store直結は行わない) |

## PR-4 での変更対象

- **RuntimePublication カテゴリ**: `AudioEngine.Timer.cpp:411` と `AudioEngine.Commit.cpp:618` の publishWorld() 直接呼び出しを `submitPublishRequest()` 経由に変更
- **デッドコード削除**: `AudioEngine::publishWorld()` を削除 (AudioEngine.h)
- **Lifecycle/TransitionPublication**: 現状維持 (専用経路として分離)
