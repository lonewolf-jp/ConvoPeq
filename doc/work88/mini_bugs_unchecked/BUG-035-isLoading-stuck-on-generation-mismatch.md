# BUG-035: applyComputedIR() の世代不一致で isLoading が true に固着

## 発見日
2026-07-26

## ファイル
`src/ConvolverProcessor.LoadPipeline.cpp:329-334`

## 問題
`applyComputedIR()` が世代不一致を検出すると、`isLoading` を `false` に戻さずに早期 return する:

```cpp
if (!convolverStateGeneration.isCurrentGeneration(generation))
{
    DBG("[Pipeline] generation mismatch, discarding result");
    // ★ isLoading を false に戻していない！
    return;
}
```

以降:
- UI にローディングスピナーが永続表示される
- `loadIR()` の `activeLoader` と `isLoading` の排他チェックが機能しない
- `isRebuilding` が固着する可能性もある
- プロセッサを破棄して再作成するまで回復不能

### 通常パス
正常系では `finalizeNUCEngineOnMessageThread` 末尾で `isLoading = false` が設定される。

### 影響
- **CRITICAL**: 一度世代不一致が発生するとシステムが完全に "Loading" 状態でスタック
- ユーザーはプロセッサの再作成以外に回復手段がない

### リスク評価
- **重大度**: CRITICAL — 回復不能な固着状態
- **発生頻度**: 低（世代不一致は稀だが、発生すると決定的）
- **影響**: UI フリーズ/操作不能

### 修正方針
早期 return の直前に `isLoading = false` を設定する:
```cpp
if (!convolverStateGeneration.isCurrentGeneration(generation))
{
    convo::publishAtomic(isLoading, false, std::memory_order_release);
    return;
}
```
