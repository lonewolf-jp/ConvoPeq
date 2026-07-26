# BUG-036: finalizeNUCEngineOnMessageThread で init() 失敗時に irL/irR がリーク

## 発見日
2026-07-26

## ファイル
`src/ConvolverProcessor.LoadPipeline.cpp:616-618`

## 問題
以下のコードで `irL.release()` と `irR.release()` が `init()` よりも先に評価される:

```cpp
if (newConv->init(irL.release(), irR.release(), ...)) // line 616-618
```

`init()` が `false` を返した場合:
- `release()` が発行した生ポインタは `init()` に渡される
- `init()` は失敗し、ポインタの所有権を取得しない
- `ScopedAlignedPtr` は既に `release()` 済みで nullptr を保持
- 生ポインタはどちらの側からも解放されず **メモリリーク**

さらに `init()` が例外を投げた場合も同様（ただし `noexcept` が宣言されていれば発生しない）。

### 影響
- 各 IR ロード失敗ごとに `numPartitions × fftSize` バイト（数十MB〜数百MB）のメモリリーク
- 連続した IR ロード失敗でメモリ消費が増大

### リスク評価
- **重大度**: CRITICAL — リーク量が大きく、OOM の原因となる
- **発生頻度**: 低（init 失敗は稀だが、発生時は決定的）
- **検出性**: 難（メモリ使用量の増加としてのみ現れる）

### 修正方針
`.release()` を `.get()` に変更し、成功時にのみ解放する:
```cpp
double* irLRaw = irL.get();
double* irRRaw = irR.get();
if (newConv->init(irLRaw, irRRaw, sampleRate, partitionSize, ...))
{
    irL.release();
    irR.release();
    ...
}
```
