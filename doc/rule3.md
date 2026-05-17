# ConvoPeq バグ修正時 AI 実装遵守事項（必須）

本修正群は、

* RT安全性
* RCU/Epoch lifetime
* Immutable Snapshot Runtime
* JUCE 非同期寿命管理
* lock-free determinism

に深く関係する。

そのため AI に単純な「局所修正」を許可してはならない。

以下を必須遵守事項として与えるべきです。

---

# 0. 最重要原則

AI は:

```text
「警告を消す修正」
```

をしてはならない。

必ず:

```text
ownership
threading
lifetime
publish ordering
snapshot authority
```

を解析した上で修正すること。

局所パッチ禁止。

---

# 1. 修正前に必ず実施する解析

AI は各修正前に必ず:

## 1-1. 呼び出し経路解析

最低でも:

```text
caller
callee
thread
ownership
publish timing
destruction timing
```

を追跡すること。

特に:

* Audio Thread
* Message Thread
* Worker Thread
* Rebuild Thread
* Epoch reclaim thread

を区別すること。

---

## 1-2. RT到達性解析

以下を必須化:

```text
このコードは Audio Thread から到達可能か
```

を必ず明示判定。

推測禁止。

call chain を追跡すること。

---

## 1-3. Lifetime 解析

必須確認:

```text
誰が所有するか
いつ解放されるか
どの thread で retire/reclaim されるか
```

特に:

* callAsync
* lambda capture
* listener callback
* DeferredDeletionQueue
* RCU/Epoch

では必須。

---

## 1-4. Snapshot authority 解析

以下を必ず列挙:

```text
authoritative source
shadow cache
derived cache
RT local state
```

混同禁止。

---

# 2. 修正時の絶対禁止事項

---

# 2-1. Audio Thread に mutex 導入禁止

禁止:

```cpp
std::mutex
CriticalSection
MessageManagerLock
WaitableEvent
condition_variable
future.get()
```

Audio Thread 到達可能箇所では全面禁止。

---

# 2-2. Audio Thread に allocation 禁止

禁止:

```cpp
new
delete
make_shared
shared_ptr refcount increment
vector resize
String allocation
Logger
std::function heap allocation
```

RT path では禁止。

---

# 2-3. CAS retry loop を RT に導入禁止

禁止:

```cpp
compare_exchange_weak loop
spin loop
unbounded retry
```

Audio Thread 上禁止。

lock-free ではなく:

```text
wait-free bounded
```

を要求。

---

# 2-4. 「atomic を増やして解決」禁止

AI が最もやりがちな誤修正:

```cpp
std::atomic<bool> safeFlag;
```

等。

禁止。

shadow state を増やすだけ。

必ず:

```text
single authority
```

へ統合。

---

# 2-5. WeakReference の濫用禁止

JUCE UAF 修正で:

```cpp
WeakReference everywhere
```

は禁止。

本当に必要な非同期境界のみ。

---

# 2-6. RCU guard 範囲の縮小禁止

禁止:

```cpp
load
↓
guard
```

必ず:

```cpp
guard
↓
load
↓
use
```

順。

---

# 2-7. 「とりあえず shared_ptr」禁止

DSP/RT 系で:

```cpp
shared_ptr everywhere
```

は禁止。

理由:

* atomic refcount
* cache ping-pong
* RT jitter

を発生。

---

# 3. Critical 修正ごとの専用遵守事項

---

# C1 callAsync UAF 修正

## 必須事項

AI は:

```text
どの thread が lambda を enqueue し
どの thread が execute し
owner がいつ destroy されるか
```

を必ず解析。

---

## 修正規則

### 許可

* WeakReference
* SafePointer
* AsyncUpdater
* MessageManager::callAsync weak capture

### 禁止

```cpp
if (this == nullptr)
```

等の無意味修正。

---

## 推奨

最優先:

```text
AsyncUpdater 化
```

callAsync 濫用回避。

---

# C2 RCUReaderGuard

---

## 必須事項

AI は:

```text
loadCurrentState()
```

内部実装まで読むこと。

もし内部で epoch pin 済みなら:

```text
修正不要
```

の可能性がある。

推測禁止。

---

## 禁止

guard を広げすぎて:

```text
Audio Thread work 増加
```

させること。

---

# C3 CMA-ES NaN

---

## 必須事項

AI は:

```text
NaN がどこから来るか
```

を追跡。

特に:

* division
* sqrt
* log
* eigen decomposition
* denormal flush

確認。

---

## 禁止

単なる:

```cpp
if (std::isnan(x)) x = 0;
```

禁止。

これは optimizer 崩壊を隠すだけ。

---

## 必須

NaN 検出時:

```text
covariance reset
sigma reset
sample discard
generation rollback
```

を明示設計。

---

# C4 LockFreeRingBuffer

---

## 最重要項目

AI は必ず:

```text
SPMC
MPSC
MPMC
SPSC
```

どれが本当に必要か解析。

---

## 禁止

RT thread に:

```cpp
compare_exchange loop
```

残すこと。

---

## 必須

RT path は:

```text
single store-release
bounded write
fixed latency
```

のみ。

---

## 許可される構造

推奨順:

### 最優先

```text
SPSC ring
```

### 次善

```text
double buffer publish
```

---

## 禁止

「lock-free だから安全」という説明。

ConvoPeq 規約では:

```text
wait-free bounded RT
```

必須。

---

# 4. High 修正群の遵守事項

---

# H3 shadow atomic 廃止

---

## 必須

AI は各 atomic を:

```text
authoritative
derived cache
RT local state
legacy mutable state
```

へ分類。

---

## 禁止

snapshot と atomic の:

```text
dual authority
```

維持。

---

## 必須

最終的に:

```text
GlobalSnapshot only
```

へ統合。

---

# H4 rebuild request

---

## 必須

rebuild trigger を:

```text
snapshot publication
```

へ統一。

---

## 禁止

setter 内から:

```cpp
requestDebouncedRebuild();
```

呼ぶこと。

---

## 必須

変更経路を:

```text
UI
↓
SnapshotCommand
↓
Snapshot publish
↓
Rebuild worker
↓
Runtime publish
```

へ統一。

---

# H5 listeners.call

---

## 必須

listener callback が:

```text
parameter mutation
snapshot mutation
rebuild request
```

を行っていないか確認。

---

## 禁止

callback 内から:

```cpp
setX()
```

再入。

---

## 推奨

listener は:

```text
UI invalidation only
```

に限定。

---

# 5. RT安全性規約

AI に必ず与えるべき。

---

# Audio Thread 完全禁止一覧

以下禁止:

* mutex
* malloc/new
* shared_ptr atomic refcount
* compare_exchange retry
* filesystem
* Logger
* sleep
* future/promise
* virtual allocation
* std::string growth
* vector resize
* exception throw

---

# Audio Thread 許可一覧

許可:

* fixed-size array
* preallocated memory
* store/load atomic
* bounded memcpy
* SIMD
* branch-light DSP

---

# 6. Immutable Runtime 規約

AI に必須提示。

---

# publish 後 mutate 禁止

publish 後 Runtime は:

```text
logical immutable
```

であること。

---

# RT mutable 許可範囲

許可されるのは:

```text
DSP history
filter z-state
ramp accumulator
envelope follower
```

等の:

```text
thread-local DSP state only
```

---

# 禁止

publish 後:

* topology mutation
* parameter mutation
* rebuild mutation
* runtime flag mutation

禁止。

---

# 7. 修正時の提出義務

AI に必須化すべき。

---

# 修正ごとに提出

## 必須項目

### 1. thread analysis

```text
どの thread が読むか
どの thread が書くか
```

---

### 2. ownership analysis

```text
誰が所有するか
いつ解放されるか
```

---

### 3. RT safety analysis

```text
Audio Thread 到達可能か
```

---

### 4. publish ordering

```text
release/acquire ordering
```

---

### 5. regression risk

```text
この修正で壊れる可能性
```

---

# 8. AI に禁止すべき危険行動

---

# 禁止1

```text
「ビルドが通るのでOK」
```

禁止。

---

# 禁止2

```text
警告が消えたのでOK
```

禁止。

---

# 禁止3

```text
atomic を足して同期
```

禁止。

---

# 禁止4

```text
shared_ptr 化して安全化
```

禁止。

---

# 禁止5

```text
Audio Thread に lock 導入
```

禁止。

---

# 禁止6

```text
RCU/Epoch を理解せず修正
```

禁止。

---

# 9. 最終検証義務

AI に必須化。

---

# 修正後に必ず確認

## 1.

```text
Audio Thread wait-free 性
```

維持。

---

## 2.

```text
Snapshot authority singularity
```

維持。

---

## 3.

```text
publish-after-free
```

なし。

---

## 4.

```text
reclaim ordering
```

正しい。

---

## 5.

```text
Message thread callback lifetime
```

安全。

---

# 最重要

AI に最も強く禁止すべきなのは:

```text
「局所修正で済ませること」
```

です。

ConvoPeq は現在:

```text
Immutable Runtime 移行途中
```

なので、

1箇所修正すると:

* snapshot
* rebuild
* epoch
* RT safety
* ownership

へ波及します。

したがって AI には必ず:

```text
全 call chain
全 ownership
全 thread boundary
```

解析を義務化すべきです。
