以下は、`plan.md` に基づいて AI に詳細設計・実装を行わせる際に、必須とすべき「AI実装遵守事項」です。
特に ConvoPeq のような RT-safe / immutable runtime / RCU 系 DSP エンジンでは、通常の CRUD 系アプリケーション向け AI コーディングと全く異なる制約が必要です。

---

# ConvoPeq AI実装遵守事項

## 1. 基本原則

AI は「機能追加」ではなく、以下の不変条件維持を最優先とすること。

* IR-1 レンダーフェーズ不変性
* IR-2 単一公開世界
* IR-3 所有権固定
* IR-4 RT 隠蔽同期禁止
* IR-5 エポック一貫観測
* IR-6 所有権移転
* IR-7 クロスフェード隔離

機能改善・最適化・可読性改善よりも、これら不変条件を優先する。

---

# 2. AI が絶対に行ってはいけないこと

## 2.1 Audio Thread 内への以下の追加を禁止

以下を Audio Thread / processBlock / DSP process path に追加してはならない。

### 禁止事項

* `std::mutex`
* `std::lock_guard`
* `std::unique_lock`
* `std::shared_ptr`
* `std::weak_ptr`
* `std::function`
* `dynamic_cast`
* `typeid`
* `new`
* `delete`
* `malloc`
* `free`
* `aligned_malloc`
* `std::vector::resize`
* `std::string`
* `JUCE String`
* `Logger::writeToLog`
* `DBG`
* `std::pow`
* `std::exp`
* `std::log`
* `std::sin`
* `std::cos`
* `std::async`
* `MessageManagerLock`

---

## 2.2 「便利そうだから」で mutable 化禁止

以下を禁止。

```cpp
mutable
const_cast
```

による immutable runtime 回避。

---

## 2.3 publish 後 object の変更禁止

以下を禁止。

```cpp
activeDSP->xxx = ...
currentDSP.load()->...
```

のような publish 後 mutation。

---

## 2.4 「部分修正」で既存 invariant を壊すことを禁止

AI は局所最適で以下を行ってはならない。

### 禁止例

* rebuild bypass
* shortcut path
* temporary mutable cache
* lazy initialization
* singleton state escape
* shared scratch reuse

---

# 3. Audio Thread 実装規約

## 3.1 Audio Thread は consume only

Audio Thread は：

```text
prepared immutable state
```

を消費するのみ。

---

## 3.2 Audio Thread が許可される処理

### 許可

* lock-free atomic load
* SIMD math
* preallocated buffer access
* fixed-size stack allocation
* POD state update
* precomputed coefficient use

---

## 3.3 Audio Thread で禁止される概念

### 禁止

* 初期化
* ownership transfer
* topology mutation
* buffer resize
* cache generation
* thread creation
* publication
* rebuild

---

# 4. Atomic / Publication 規約

## 4.1 atomic の直接使用禁止

AI は：

```cpp
.store(... memory_order_xxx)
.load(... memory_order_xxx)
```

を直接書いてはならない。

---

## 必須

必ず project helper を使用。

```cpp
publishAtomic(...)
consumeAtomic(...)
```

---

## 理由

publication semantics の統一。

---

# 4.2 relaxed 使用禁止ルール

AI は以下以外で：

```cpp
memory_order_relaxed
```

を使ってはならない。

### relaxed 許可対象

* meter
* statistic counter
* debug value
* telemetry

---

# 4.3 publication domain 分裂禁止

以下を禁止。

```cpp
atomic A 更新
atomic B 更新
```

で一貫スナップショットを構成すること。

---

## 必須

単一 immutable snapshot publish。

---

# 5. Ownership / Lifetime 規約

## 5.1 shared ownership 禁止

DSP runtime object に：

```cpp
shared_ptr
weak_ptr
```

禁止。

---

## 許可

* unique ownership
* epoch reclamation
* explicit transfer

---

# 5.2 raw pointer 保持禁止

AI は長寿命 object に：

```cpp
T*
```

を保存してはならない。

---

## 許可される raw pointer

### 一時参照のみ

```cpp
auto* dsp = consumeAtomic(currentDSP);
```

。

---

# 5.3 RCUReader 規約

RCUReader は：

* copy禁止
* move禁止
* thread-affine

。

AI は wrapper 化しない。

---

# 6. Crossfade / Runtime Transition 規約

## 6.1 crossfade 中の mutable state 共有禁止

旧 runtime/new runtime 間で：

* scratch buffer
* latency buffer
* meter state
* temporary FFT buffer

共有禁止。

---

# 6.2 transition 中の ownership 明示

AI は：

```text
old owner
new owner
retire timing
```

をコードコメントで必ず明示。

---

# 7. SIMD / Buffer 規約

## 7.1 AVX load/store 変更禁止

AI は：

```cpp
loadu → load
```

変更を勝手に行ってはならない。

---

## 必須

アライン保証証明がある場合のみ。

---

# 7.2 SIMD 最適化より invariant 優先

以下禁止。

```text
unsafe alias optimization
manual prefetch
undefined alignment assumption
```

---

# 8. メモリ確保規約

## 8.1 aligned_malloc 直接使用禁止

必須：

```cpp
aligned_make_unique<T>()
```

。

---

# 8.2 placement new 直接使用禁止

AI が：

```cpp
new(ptr) T
```

を書くことを禁止。

---

## 例外

aligned allocator helper 内のみ。

---

# 9. 状態機械規約

## 9.1 bool フラグ増殖禁止

AI は：

```cpp
bool shuttingDown;
bool rebuilding;
bool stopping;
```

のような状態管理を追加禁止。

---

## 必須

enum state machine。

---

# 9.2 phase transition をコメント化

以下を mandatory。

```cpp
// Transition:
// Running -> Stopping
```

。

---

# 10. 実装前解析義務

AI は修正前に：

---

## 必須解析

### 1.

call graph

### 2.

ownership flow

### 3.

publication edge

### 4.

thread affinity

### 5.

crossfade coexistence

---

## 未解析状態で修正禁止

「その場修正」を禁止。

---

# 11. パッチ生成規約

## 11.1 AI は diff 単位で出力

禁止：

* 全ファイル再生成
* unrelated formatting
* include reorder

---

## 必須

最小差分。

---

# 11.2 invariant コメント必須

AI が runtime publication を変更する場合：

```cpp
// IR-2:
// Atomic world switch.
```

のように invariant を明示。

---

# 12. テスト規約

AI は修正ごとに：

---

## 必須

### RT-safe 検証

* allocation
* lock
* libm
* hidden init

混入有無。

---

### ownership 検証

* double free
* UAF
* retire ordering

。

---

### publication 検証

* acquire/release edge
* partial visibility

。

---

# 13. AI が優先すべき思想

AI は：

```text
性能最適化
```

より：

```text
ownership algebra
publication semantics
immutable purity
phase isolation
```

を優先すること。

---

# 14. 最重要原則

AI は：

```text
「動くようにする」
```

のではなく、

```text
「形式的不変条件を壊さない」
```

ことを目的とする。

ConvoPeq は通常アプリケーションではなく：

```text
deterministic concurrent DSP runtime
```

であることを常に前提とする。
