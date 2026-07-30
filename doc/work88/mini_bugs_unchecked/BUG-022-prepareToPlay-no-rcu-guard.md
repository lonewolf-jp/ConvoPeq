# BUG-022: prepareToPlay が RCU reader なしで engine データにアクセス

## 発見日
2026-07-26

## ファイル
`src/convolver/ConvolverProcessor.Lifecycle.cpp:228-274`

## 問題
`prepareToPlay()` (Lifecycle.cpp:228) は `loadActiveEngine()` で engine を取得し、
そのデータ（`conv->irDataLength`, `conv->irData[0]`, `conv->irData[1]`, `conv->irLatency`, `conv->storedScale` など）を
RCU reader 保護なしで読み取っている。その後、この engine を `exchangeActiveEngine(newConv)` で置き換える。

比較対象:
- `process()`: `RCUReaderGuard` あり (Runtime.cpp:211)
- `refreshLatency()`: `GlobalGuard` あり (Runtime.cpp:90-94)
- `reset()`: `GlobalGuard` あり (Lifecycle.cpp:462-466)
- `getLatencyBreakdown()`: `GlobalGuard` あり (StateAndUI.cpp:676-680)
- `shareConvolutionEngineFrom()`: `GlobalGuard` あり (StateAndUI.cpp:416-420)

`prepareToPlay()` だけが関数全体を RCU 未保護で実行している。

## リスク評価

### シナリオ

`prepareToPlay()` は以下の処理を行う:
1. 既存 engine の `irData` / `irDataLength` を読み取る（lines 246-249）
2. 新しい engine を構築し、既存 engine のパラメータで初期化（lines 265-268）
3. `exchangeActiveEngine(newConv)` で新しい engine を公開（line 272）
4. `retireStereoConvolver(oldConv, retireEpoch)` で古い engine を deferred delete に委譲（line 274）

`prepareToPlay()` はセッション開始時などに呼ばれ、その間は engine の concurrent swap は発生しない。
しかし、この関数で読み取る engine は `exchangeActiveEngine` の戻り値と同一であり、
同じ関数内で生成された新しい engine が完了するまで、古い engine から読み取ったデータに依存する。
RCU 保護がない場合、コンカレントリクレームが発生するとデータ競合となる。

### 現状

- すべての caller は Message Thread 上で動作する
- `prepareToPlay` も Message Thread 上の排他制御下で動作するため、実害はない
- `retireStereoConvolver` は deferred delete を使用するため、すぐに解放はされない

### 将来的リスク

Message Thread 以外から呼ばれた場合、または engine のコンカレントアクセスが導入された場合に
UAF / データ競合として顕在化する。

## 修正

関数冒頭に RCU reader guard を追加する（`process` / `refreshLatency` / `reset` と同パターン）:

```cpp
void ConvolverProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(*this);

    ...
    auto* conv = loadActiveEngine(std::memory_order_acquire);
    ...
}
```
