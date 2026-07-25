現在の設計方針（Practical Stable ISR Bridge Runtime、および「Publication Domain は RuntimeWorld のみ」という前提）に従うなら、`rtActiveStructureShadow` や `rtDeferredBandResetMask` も **P0-1 と同じ考え方で「個別atomic化」するのが最も整合性が高い**と考えます。

添付ソースを見る限りでは、これらは以下のように利用されています。

* `rtActiveStructureShadow`

  * Worker同期で `syncedStructure` を代入
  * RT処理中にも `requestedMode` 等で更新
  * RT側で読み取り分岐に使用
* `rtDeferredBandResetMask`

  * Worker同期で代入
  * RT側で `|=` による追加
  * RT側で読取り→0クリア→残件再設定

したがって **Worker と RT の双方が書き込む shadow** という点では、AGC shadow と同じ性質です。

---

# 推奨案（ISR設計との整合を優先）

## 1. rtActiveStructureShadow

### 現状

```
Worker
    rtActiveStructureShadow = syncedStructure;

RT
    auto mode = rtActiveStructureShadow;

RT
    rtActiveStructureShadow = requestedMode;
```

これは Data Race になります。

---

### 修正

```
std::atomic<FilterStructure> rtActiveStructureShadow {
    FilterStructure::Serial
};
```

読込み

```cpp
auto mode =
    rtActiveStructureShadow.load(std::memory_order_relaxed);
```

書込み

```cpp
rtActiveStructureShadow.store(
    syncedStructure,
    std::memory_order_relaxed);
```

RT側

```cpp
rtActiveStructureShadow.store(
    requestedMode,
    std::memory_order_relaxed);
```

---

### memory_order

`FilterStructure` は単独値です。

他の状態と一緒に同期していないため

```
memory_order_relaxed
```

で十分です。

---

## 2. rtDeferredBandResetMask

こちらは少し事情が違います。

現状は

```
Worker
    mask = syncedMask;

RT
    mask |= xxx;

RT
    tmp = mask;
    mask = 0;
```

となっています。

これは

```
load
OR
store
```

ではありません。

実質

```
read
modify
write
```

です。

つまり

```
mask |= value;
```

はRMWです。

---

### 最適案

```
std::atomic<uint32_t> rtDeferredBandResetMask{0};
```

Worker

```cpp
store(mask, relaxed);
```

RT OR追加

```cpp
fetch_or(bits, relaxed);
```

RT取得

```cpp
uint32_t mask =
    exchange(0, relaxed);
```

つまり

```cpp
rtDeferredBandResetMask.fetch_or(bits,
                                 std::memory_order_relaxed);

auto mask =
    rtDeferredBandResetMask.exchange(
        0,
        std::memory_order_relaxed);
```

になります。

これなら

```
OR
取得
クリア
```

がatomicになります。

---

# 3. 他shadowも同じ分類で整理

ソースを見る限りでは、shadowは次の3種類に分類できます。

| 種類                          | 推奨修正                                 |
| --------------------------- | ------------------------------------ |
| bool / enum / scalar shadow | atomic<T>                            |
| bitmask                     | atomic<uint32_t> + fetch_or/exchange |
| 複数値snapshot                 | RuntimeWorldへ統合（ShadowState禁止）       |

---

# 4. ISRとの整合

この方法なら

* RuntimeWorldは増えない
* Epoch不要
* retire不要
* delete不要
* Publication Domainは増えない

という現在のISR設計を維持できます。

---

# 5. ただし `rtDeferredBandResetMask` は設計改善の余地がある

一点だけ、`rtDeferredBandResetMask` はAGC shadowより設計上の課題が大きいです。

これは単なる「共有状態」ではなく、

> **イベントキューを1つのbitmaskで代用している**

構造になっています。

そのため、atomic化でData Raceは解消できますが、

* Workerが `store()` で新しいマスクを書き込む
* RTが `fetch_or()` で追加する
* WorkerとRTが同時更新する

というケースでは、更新の上書き・消失が起きないかを個別に検証する必要があります。

もし「Worker側の更新」と「RT側の追加」が独立したイベントとして保持されることが要件であれば、`store()` は適切ではなく、Worker側も `fetch_or()` を用いるか、専用のイベントキューへ置き換える設計の方が堅牢です。

## 総合評価

この設計書に追記する改修案としては、以下が最も一貫しています。

* **`rtActiveStructureShadow`**：`std::atomic<FilterStructure>` 化し、`load/store(memory_order_relaxed)` を使用。
* **`rtDeferredBandResetMask`**：`std::atomic<uint32_t>` 化し、`fetch_or()` と `exchange(0)` によるRMWへ変更する。ただし、Worker側が現在 `store()` を行っている実装については、イベント消失の可能性がないかソースレベルで追加検証を行い、必要に応じてWorker側も `fetch_or()` または専用イベントキューへ変更する。これが実装前に確認すべき重要事項です。
