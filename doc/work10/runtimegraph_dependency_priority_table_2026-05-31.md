# RuntimeGraph 依存解消 優先順位表（2026-05-31）

## 判定基準

- **確定漏れ**: 詳細設計の DoD / 統治規約に照らして、実装上まだ残すべきでない `RuntimeGraph` 依存。
- **詳細設計どおり実装済み**: `RuntimeGraph` の authority 判定依存が消え、`RuntimeReadView` / `TransitionState` / 非所有スロットへ収束済み。

## 優先順位表

| 優先度 | ファイル | 残存の性質 | 対応結果 |
| --- | --- | --- | --- |
| 1 | `src/audioengine/AudioEngine.Commit.cpp` | commit / crossfade / retire の中核。`RuntimeGraph` を使った current/fading 判定が残りやすい | **解消済み** |
| 2 | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | prepare 時の current/fading publication 整合 | **解消済み** |
| 3 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | shutdown / clear 時の current/fading 整合 | **解消済み** |
| 4 | `src/audioengine/AudioEngine.CtorDtor.cpp` | 初期化 / 終了時の slot 整合 | **解消済み** |
| 5 | `src/audioengine/AudioEngine.Learning.cpp` | learning dispatch の current DSP 取得 | **解消済み** |
| 6 | `src/audioengine/AudioEngine.Processing.Latency.cpp` | latency query の current DSP 取得 | **解消済み** |
| 7 | `src/audioengine/AudioEngine.Processing.Snapshot.cpp` | snapshot 経路の不要な `RuntimeGraph` hint | **解消済み** |
| 8 | `src/audioengine/AudioEngine.h` | projection helper / diagnostic helper 収束 | **解消済み** |

## 監査メモ

- `getRuntimeGraph(...)` / `runtimeGraph->...` / `RuntimeGraph* runtimeGraphHint` の `src/audioengine` 配下残存はゼロ化。
- `RuntimePublishView` は `sampleRateHz` と `TransitionState` のみを保持し、graph authority を持たない。
- `RuntimeGraph` は publish world の構築・診断用の内部データとしてのみ残存。

## 検証結果

- `get_errors`: 対象ファイルすべて **No errors found**
- `grep`: `src/audioengine/**/*.cpp` / `src/audioengine/**/*.h` における `getRuntimeGraph(` / `runtimeGraph->` / `RuntimeGraph* runtimeGraphHint` 残存 **なし**
