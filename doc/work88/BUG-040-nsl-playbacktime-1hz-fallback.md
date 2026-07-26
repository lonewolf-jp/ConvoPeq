# BUG-040: NoiseShaperLearner 再生時間計算が 1 Hz にフォールバックし学習が異常終了

## 発見日
2026-07-26

## ファイル
`src/NoiseShaperLearner.cpp:1164-1168`

## 問題
`drainCaptureQueue()` で accumulatedPlaybackSeconds を計算する際のサンプルレートフォールバック:

```cpp
const int playbackSampleRateHz = (session.sampleRateHz > 0)
    ? session.sampleRateHz
    : ((block.sampleRateHz > 0) ? block.sampleRateHz : 1);
accumulatedPlaybackSeconds += static_cast<double>(block.numSamples)
    / static_cast<double>(playbackSampleRateHz);
```

`session.sampleRateHz` と `block.sampleRateHz` が両方 0 以下の場合、`playbackSampleRateHz = 1` となる。

結果:
- `accumulatedPlaybackSeconds` が `block.numSamples` 秒ずつ増加（通常 512〜4096 秒/ブロック）
- ほぼ瞬時に `targetSeconds`（Shortest=10s, Short=30s など）を超える
- 学習が**開始直後に完了・保存**され、係数が不十分なまま確定する

### 通常動作
- 正常時: `session.sampleRateHz` は `captureSessionSignature()` でエンジンから取得される
- 異常系: エンジン未初期化・再構築中・サンプルレート未設定の状態で学習が始まった場合に発生

### リスク評価
- **重大度**: MEDIUM（正常時は発生しないが、条件成立時に無音の学習結果を保存）
- **発生頻度**: エンジン状態遷移中のレアケース

### 修正方針
`playbackSampleRateHz` のフォールバック値を 1 から `AudioEngine::getDefaultSampleRate()` または 48000 に変更する。
またはフォールバック時はブロックをスキップしてカウンタを進めない。
