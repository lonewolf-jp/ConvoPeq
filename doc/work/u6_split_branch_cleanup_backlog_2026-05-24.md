# U6 splitマクロ分岐整理: Deferred Cleanup Backlog

作成日: 2026-05-24
対象: `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_*` の常時ON前提分岐

## このPR（safe subset）で実施したこと

- `src/audioengine/AudioEngine.Cache.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CACHE_MANAGER)` を削除
- `src/audioengine/AudioEngine.CtorDtor.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CTOR_DTOR)` を削除
- `src/audioengine/AudioEngine.Fifo.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_FIFO_UI)` を削除
- `src/audioengine/AudioEngine.Globals.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_GLOBALS)` を削除
- `src/audioengine/AudioEngine.Timer.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_TIMER_CALLBACK)` を削除
- `src/audioengine/AudioEngine.Init.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_INIT_LIFECYCLE)` を削除
- `src/audioengine/AudioEngine.EQResponse.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_CALC_EQ_RESPONSE)` を削除
- `src/audioengine/AudioEngine.Parameters.cpp`
  - 外側の `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PARAMETERS)` を削除
  - 内側の `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_*` 分岐は維持
- `src/audioengine/AudioEngine.Snapshot.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_SNAPSHOT_CREATE)` を削除
- `src/audioengine/AudioEngine.StateIO.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_LOAD/GET` を削除
- `src/audioengine/AudioEngine.Threading.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_THREADING_*` を削除
- `src/audioengine/AudioEngine.UIEvents.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_UI_EVENTS)` を削除
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_AUDIO_BLOCK)` を削除
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_BLOCK_DOUBLE)` を削除
- `src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_TO_BUFFER)` を削除
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_FLOAT)` を削除
- `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_IO)` を削除
- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_TO_BUFFER/DSP_DOUBLE/DSP_IO` を削除
- `src/audioengine/AudioEngine.Processing.Latency.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_LATENCY_QUERY/DSP_LATENCY` を削除
- `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_DSP_PREPARE)` を削除
- `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_PREPARE_TO_PLAY)` を削除
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
  - ファイル全体を包む `#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_RELEASE_RESOURCES)` を削除
- `src/audioengine/AudioEngine.Processing.Snapshot.cpp`
  - ガードで囲まれていた補助namespace/関数定義の `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_PROCESSING_SNAPSHOT` を削除
- `src/audioengine/AudioEngine.Commit.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_COMMIT_PREPARE/COMMIT_EXECUTE` を削除
- `src/audioengine/AudioEngine.Learning.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_*` を削除
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - 関数単位ガード `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_REBUILD_DISPATCH/REBUILD_EXECUTE` を削除

上記26件はいずれも **CMakeで常時 `=1` 定義** される no-op ガードであり、意味不変で除去可能。

## 後続PRへ回す項目（deferred）

現時点で、この台帳に記載していた deferred 項目（Commit/Learning/RebuildDispatch）は解消済み。

## 次の進め方（提案）

1. ファイル全体を包むだけのガードを優先除去（意味不変を担保しやすい）
2. 部分ブロック分岐は機能ごとにPR分割して整理
3. 各PRで Debug/Release ビルドとRT制約レビューを必須化

## 補足

- `AudioEngine.StateIO.cpp` は `AudioEngine.Parameters.cpp` 側の `!defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_STATEIO_*)` フォールバック実装と重複しないことを維持。
- `AudioEngine.Threading.cpp` は実体定義を一本化し、常時ONマクロ依存を減らした。
