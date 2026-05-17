# ConvoPeq

# Immutable Snapshot Runtime 移行実装

# AI 実装規約書 v1.0

## 目的

本規約は、ConvoPeq を

```text
Deterministic Immutable DSP Runtime System
```

へ完全移行するために、AI が詳細設計・実装・リファクタリング・レビューを行う際の強制遵守事項を定義する。

本規約は:

* アーキテクチャ不変条件
* RT deterministic 制約
* Immutable RuntimeWorld 原則
* RCU/Epoch reclamation 原則
* JUCE plugin safety
* x64 cache coherence
* ABA/UAF 安全性

を保護するためのものである。

AI は本規約を「推奨」ではなく:

```text
強制契約
```

として扱うこと。

---

# 1. 最上位不変条件（IR: Immutable Runtime Invariants）

---

## IR-1

# Runtime publish 後 mutate 完全禁止

publish 後の RuntimeWorld に対する:

* 書き込み
* sync
* patch
* update
* propagation
* repair
* lazy init

を全面禁止。

禁止例:

```cpp
runtime->table[i] = x;
syncRuntimeGraph(...);
updateCoefficients(...);
```

---

## IR-2

# Audio Thread は read-only

Audio Thread は:

```cpp
const RuntimeWorld*
```

のみ読む。

Audio Thread で禁止:

* runtime mutation
* command execution
* parameter propagation
* rebuild
* cache update
* smoothing reconfiguration
* coeff regeneration

---

## IR-3

# RuntimeWorld は完全 immutable

RuntimeWorld 配下オブジェクトは:

```cpp
const
immutable
trivially stable
```

でなければならない。

禁止:

```cpp
mutable
std::atomic inside runtime
lazy cache
mutable lookup
```

---

## IR-4

# Source of Truth は単一

parameter/state/configuration の source of truth は:

```text
GlobalSnapshot
```

のみ。

禁止:

```cpp
m_currentXxx
live runtime cache
shadow parameter state
duplicated atomics
```

---

## IR-5

# Runtime は完全 materialized

publish 前に:

* coeff
* FIR plans
* AGC tables
* smoothing seeds
* lookup tables
* latency plans

を完全生成すること。

RT thread 上で:

```cpp
generate
update
resize
prepare
```

禁止。

---

## IR-6

# Runtime transition も immutable

crossfade/fade/transition state machine を mutable にしてはならない。

transition は:

```cpp
ImmutableTransitionPlan
```

として publish する。

---

## IR-7

# Retire 後アクセス禁止

retire 済 runtime へのアクセスは禁止。

RCU/Epoch の reader contract を破ってはならない。

---

# 2. RuntimeWorld 設計規約

---

## 2.1 RuntimeWorld の責務

RuntimeWorld は DSP 実行に必要な:

* graph
* tables
* coeffs
* latency
* routing
* transitions
* AGC
* smoothers

を完全に含むこと。

---

## 2.2 RuntimeWorld の禁止事項

禁止:

```cpp
mutable
shared mutable ownership
post-build sync
atomic fields
MessageManager access
JUCE UI object refs
```

---

## 2.3 所有権

許可:

```cpp
std::unique_ptr<const T>
std::shared_ptr<const T>
```

禁止:

```cpp
raw owning ptr
manual delete
shared mutable ptr
```

---

# 3. Audio Thread 規約

---

## 3.1 RT thread 完全禁止事項

Audio Thread 上で禁止:

| 禁止                  | 理由                 |
| ------------------- | ------------------ |
| malloc/new          | nondeterministic   |
| free/delete         | allocator lock     |
| std::function       | hidden alloc       |
| virtual dispatch    | unpredictable      |
| mutex               | priority inversion |
| condition_variable  | blocking           |
| logging             | hidden lock        |
| exceptions          | unwind             |
| libm                | latency variance   |
| compare_exchange    | cache ping-pong    |
| exchange            | locked op          |
| fetch_add(acq_rel)  | heavy RMW          |
| filesystem          | blocking           |
| JUCE MessageManager | UI dependency      |

---

## 3.2 atomic 制約

Audio Thread で許可される atomic は:

```cpp
load(memory_order_acquire)
store(memory_order_release)
```

のみを原則とする。

禁止:

```cpp
exchange
compare_exchange
fetch_add
fetch_sub
```

---

## 3.3 RT thread の責務

RT thread は:

```cpp
load runtime ptr
↓
process immutable runtime
↓
return
```

のみ。

---

# 4. Snapshot 規約

---

## 4.1 Snapshot は完全値型

GlobalSnapshot は:

```cpp
trivially copyable semantics
immutable semantics
```

を維持すること。

---

## 4.2 Snapshot build

parameter 更新時は:

```text
new snapshot build
↓
new runtime build
↓
publish
```

のみ許可。

---

## 4.3 partial update 禁止

禁止:

```cpp
patch snapshot
incremental mutation
runtime-side patch
```

---

# 5. Builder Thread 規約

---

## 5.1 RuntimeBuilder の責務

Builder thread は:

* coeff generation
* table generation
* FIR build
* AGC build
* transition planning
* latency planning

を全て完了する。

---

## 5.2 Build complete before publish

publish 前に:

```text
Runtime fully materialized
```

でなければならない。

---

## 5.3 Builder のみ mutable 許可

mutable operations は builder thread 内のみ許可。

---

# 6. Crossfade 規約

---

## 6.1 mutable fade state 禁止

禁止:

```cpp
fadePending
runtimeCrossfadePending
snapshotAlpha
```

---

## 6.2 immutable transition

許可:

```cpp
ImmutableTransitionPlan
```

のみ。

---

# 7. DSP API 規約

---

## 7.1 process API

最終形:

```cpp
process(const RuntimeWorld&)
```

のみ。

禁止:

```cpp
process(runtime, mutableState)
```

---

## 7.2 DSP state mutation 禁止

DSP processing 中に:

```cpp
updateCoeff()
rebuildTable()
syncState()
```

禁止。

---

# 8. RCU/Epoch 規約

---

## 8.1 Reader contract

reader enter/exit は完全対称でなければならない。

禁止:

```cpp
silent enter failure
nested imbalance
underflow
```

---

## 8.2 Retire contract

retire object は:

```text
all readers exited
```

後のみ reclaim。

---

## 8.3 ABA 防止

全 retire object は:

```cpp
generation/version
```

を持つこと。

---

# 9. Memory 規約

---

## 9.1 allocator

RT path allocator 禁止。

---

## 9.2 aligned allocation

aligned allocation は:

```text
non-RT only
```

を強制。

---

## 9.3 lifetime

runtime lifetime は:

```text
publish
↓
reader protected
↓
retire
↓
epoch reclaim
```

を厳守。

---

# 10. JUCE 規約

---

## 10.1 MessageManager

RT thread から:

```cpp
MessageManager
callAsync
AsyncUpdater
```

禁止。

---

## 10.2 Thread ownership

JUCE object は:

```text
message thread ownership
```

を維持。

---

# 11. AI 実装時の強制レビュー項目

AI はコード生成時、毎回以下を自己監査すること。

---

## 11.1 immutable 違反監査

確認:

* publish後 mutate がないか
* sync API がないか
* runtime cache がないか
* shadow state がないか

---

## 11.2 RT safety 監査

確認:

* allocation がないか
* lock がないか
* RMW atomic がないか
* libm がないか
* logging がないか

---

## 11.3 ownership 監査

確認:

* raw ownership がないか
* UAF がないか
* retire/reclaim ordering が正しいか

---

## 11.4 transition 監査

確認:

* mutable fade state がないか
* crossfade pending がないか

---

# 12. AI による禁止行為

AI は以下を絶対に行ってはならない。

---

## 12.1 応急処置禁止

禁止:

```cpp
mutex追加
atomic追加
sleep追加
retry loop追加
volatile追加
```

---

## 12.2 mutable cache 追加禁止

禁止:

```cpp
currentValue
cachedValue
runtimeShadow
syncState
```

---

## 12.3 部分 rebuild 禁止

禁止:

```cpp
partial runtime patch
incremental graph mutation
live runtime repair
```

---

# 13. 実装順序規約

AI は以下順序を厳守。

| 順序 | 作業                         |
| -- | -------------------------- |
| 1  | RuntimeCommandQueue 削除     |
| 2  | runtime mutation setter 削除 |
| 3  | sync API 削除                |
| 4  | DSPExecutionState 削除       |
| 5  | process API 統一             |
| 6  | immutable transition 導入    |
| 7  | currentXXX atomic 削除       |
| 8  | RuntimeWorld 統合            |
| 9  | legacy cleanup             |

順序違反禁止。

---

# 14. 最終完成条件

以下を満たした時のみ完成と見なす。

---

## 完成条件

```text
RT thread never mutates runtime
```

```text
single immutable RuntimeWorld publish
```

```text
all runtime transitions immutable
```

```text
no runtime sync APIs
```

```text
no mutable DSP execution cache
```

```text
no RuntimeCommandQueue
```

```text
no shadow parameter state
```

---

# 15. AI 最終遵守命令

AI は常に以下を最優先すること。

優先順位:

```text
Determinism
> RT safety
> Immutable correctness
> Ownership correctness
> Throughput
> Convenience
> Minimal patch size
```

局所最適修正は禁止。

常に:

```text
architectural correctness
```

を優先すること。
