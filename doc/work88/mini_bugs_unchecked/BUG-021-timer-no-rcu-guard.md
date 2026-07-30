# BUG-021: timerCallback が RCU reader なしで engine にアクセス

## 発見日
2026-07-26

## ファイル
- `src/convolver/ConvolverProcessor.Lifecycle.cpp:150-169`
  - `loadActiveEngine()` で engine を取得し、`nucConvolvers[ch]->getRingOverflowCount()` / `resetRingOverflowCount()` にアクセス
  - 周囲に RCU reader guard なし
  - 比較対象: `process()` (Runtime.cpp:211) は `RCUReaderGuard` あり、`refreshLatency()` (Runtime.cpp:90-94) は `GlobalGuard` あり

## 問題
`timerCallback()` はエンジンポインタを RCU reader 保護なしで読み取り、メンバ関数を呼び出している。
コードベースの他箇所（`process`, `refreshLatency`, `reset`）はすべて RCU reader guard を使用しており、一貫性を欠く。

## リスク評価
**現状**: LOW — 現在のスレッドモデルでは Message Thread serialization により実害なし。
- `timerCallback()` は Message Thread で実行される
- `switchEngineOnMessageThread()` も Message Thread で実行される
- メッセージキューにより直列化されるため、timerCallback は常に有効な engine を参照する

**潜在的リスク**: MEDIUM — 以下の変更があった場合に UAF が顕在化する:
- timer 機構が Message Thread 以外でコールバックするよう変更された場合
- engine スワップが別スレッドに移動した場合
- 新しい concurrent engine accessor が追加された場合

## 修正
関数冒頭に RCU reader guard を追加する:

```cpp
void ConvolverProcessor::timerCallback()
{
    // ★ RCU reader guard: engine ポインタの有効性を保護
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(3); }
        ~GlobalGuard() { cp.exitGlobalReader(3); }
    } guard(*this);

    ...
    auto* conv = loadActiveEngine(std::memory_order_acquire);
    ...
}
```
