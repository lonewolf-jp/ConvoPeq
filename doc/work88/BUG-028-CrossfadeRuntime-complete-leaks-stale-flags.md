# BUG-028: CrossfadeRuntime::complete() が useDryAsOld_ / firstIrDryPending_ をリセットしない

## 発見日
2026-07-26

## ファイル
`src/audioengine/ISR/CrossfadeRuntime.h:93-98`

## 問題
`CrossfadeRuntime::complete()` は以下のフィールドをリセットしない:
- `useDryAsOld_` (std::atomic<bool>)
- `firstIrDryPending_` (std::atomic<bool>)
- `firstIrDryDone_` (plain bool)

一方 `start()`（line 38-51）はこれらのフィールドをゼロ初期化する:
```cpp
void start(double fadeTimeSec, double sampleRate) noexcept
{
    convo::publishAtomic(useDryAsOld_, false, std::memory_order_release);
    convo::publishAtomic(firstIrDryPending_, false, std::memory_order_release);
    firstIrDryDone_ = false;
    ...
}
```

`complete()` はこれらのリセットを行わない:
```cpp
void complete() noexcept
{
    convo::publishAtomic(pending_, false, std::memory_order_release);
    convo::publishAtomic(queuedFadeTimeSec_, 0.030, std::memory_order_release);
    convo::publishAtomic(fadeStartTimestampUs_, 0, std::memory_order_release);
    // useDryAsOld_, firstIrDryPending_, firstIrDryDone_ はそのまま！
}
```

### 問題となるシナリオ

`DSPTransition::onPublishCompleted()`（DSPTransition.h:108-112）は、
クロスフェード不要の遷移で `complete()` を `start()` なしで直接呼ぶ:
```cpp
} else if (oldDSP != nullptr) {
    engine_.crossfadeRuntime_.complete();   // ★ start() を経由しない！
    lifetime.retire(oldDSP);
}
```

もし以前のクロスフェードで `useDryAsOld_ = true` が設定されたままだと、
次回 `AudioEngine::refreshCrossfadePreparedSnapshotFromAtomics()` が
`useDryAsOld()` の古い値を読んでしまい、アクティブでないクロスフェードの
Dry/Wet 混在ロジックが誤って適用される。

### 影響
- クロスフェード終了後も stale な `useDryAsOld_` が残る
- Audio Thread が不適切な Dry 信号混合を行う可能性 → 可聴な出力歪み
- `refreshCrossfadePreparedSnapshotFromAtomics()` は `isPending()` をチェックしない

### リスク評価
- **重大度**: HIGH — 可聴なオーディオ歪み（Dry/Wet 混合率異常）
- **発生頻度**: 低（特定の遷移シーケンスが必要）
- **検出性**: 難（ノイズとして現れる場合あり）

### 修正方針
`complete()` で以下のフィールドもリセットする:
```cpp
convo::publishAtomic(useDryAsOld_, false, std::memory_order_release);
convo::publishAtomic(firstIrDryPending_, false, std::memory_order_release);
firstIrDryDone_ = false;
```
