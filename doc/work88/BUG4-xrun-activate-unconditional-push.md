# BUG 4: xRunBuffer ACTIVATE イベント無条件プッシュによるサイレントデータ損失

- **発見日**: 2026-07-25
- **重要度**: Critical
- **影響**: ACTIVATE 検出イベントがキュー満杯時に通知なしで消失する

## 場所

| ファイル | 行 |
|---------|-----|
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 605 |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 572 |
| `src/audioengine/AudioEngine.h` | 2486-2487 (キュー定義) |

## 現象

ACTIVATE 検出（RuntimeWorld generation 変化）時に `xRunBuffer.push(ev)` の戻り値をチェックしていない。
隣接する XRUN イベントのプッシュ（`AudioBlock.cpp:579`, `BlockDouble.cpp:546`）では
正しく `if (!xRunBuffer.push(ev))` で失敗を検出し `xRunDropCount` をインクリメントしている。

```cpp
// AudioBlock.cpp:579 — 正しい（戻り値をチェック）
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

```cpp
// AudioBlock.cpp:605 — バグ（戻り値を破棄）
xRunBuffer.push(ev);
```

```cpp
// BlockDouble.cpp:546 — 正しい（戻り値をチェック）
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

```cpp
// BlockDouble.cpp:572 — バグ（戻り値を破棄）
xRunBuffer.push(ev);
```

## キュー仕様

- `LockFreeRingBuffer<XRunEvent, 64>` — SPSC リングバッファ
- Capacity = 64（`kXRunBufferCapacity`）
- `push()` はキュー満杯時に `false` を返す（`LockFreeRingBuffer.h:38`）
- コンシューマーは Timer スレッド（`AudioEngine.Timer.cpp:1317`）

## 影響

- ACTIVATE イベント喪失：RuntimeWorld の generation 変化が Timer 側で検出できない
- 喪失時の指標がない：同じキューを使う XRUN イベントは `xRunDropCount` で監視可能だが、
  ACTIVATE イベントのドロップは無通知
- 結果として、Timer スレッドの RuntimeWorld 世代追跡が不正確になる
- SPSC なので他のイベントへの影響はないが、診断・監視の死角となる

## 修正方針

他の XRUN プッシュと同様に戻り値をチェックし、失敗時はドロップカウンターを
インクリメントする。

```cpp
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

必要であれば専用の `activateDropCount` を追加することも検討。
