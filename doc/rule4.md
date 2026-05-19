# ConvoPeq ISR改修プロジェクト

## AI専用実装統制規約 v1.0（Immutable Runtime System完全移行用）

本規約は、ConvoPeq を
**“Immutable DSP Graph + Immutable Runtime State Machine”**
へ完全移行する際に、AI に設計・実装・修正を行わせるための強制規約である。

本規約の目的は以下：

* ISR不変条件の破壊防止
* AIによる局所修正の暴走防止
* mutable coordination plane の再発防止
* memory ordering 崩壊防止
* ownership 崩壊防止
* RT安全性の保証
* snapshot coherence の保証
* architectural drift の防止

以下に違反する実装は、コンパイル可能であっても不合格とする。

---

# 1. 最上位原則（絶対不変）

## 1-1. RuntimeState が唯一の publish unit

AIは以下を絶対に破ってはならない。

```cpp
std::atomic<RuntimeState*> current;
```

のみが Runtime 可視状態の同期点である。

禁止事項：

* RuntimeState 外の mutable runtime parameter
* split atomic publication
* side-channel atomic
* bypass shadow
* runtime flag singleton
* transition completion flag
* runtime global mutable cache state

許可される mutable state は：

* Audio Thread ローカル状態
* UI Thread ローカル状態
* epoch internals
* allocator internals

のみ。

---

## 1-2. RuntimeState 外で runtime coherence を形成してはならない

禁止：

```cpp
std::atomic<int> fadeSamples;
std::atomic<double> fadeTime;
```

許可：

```cpp
struct RuntimeParameterSnapshot {
    FadeSpec fade;
};
```

理由：

ISR の coherence unit は RuntimeState 全体のみ。

---

## 1-3. RuntimeState は immutable

publish 後に変更禁止。

禁止：

```cpp
state->params.mix = x;
```

許可：

```cpp
RuntimeState* newState = cloneAndModify(oldState);
publish(newState);
```

---

## 1-4. RT thread は ownership を変更してはならない

RT thread 内禁止：

* shared_ptr copy/reset
* weak_ptr lock
* intrusive_ptr addRef/release
* malloc/free/new/delete
* vector resize
* unordered_map insert
* string allocation

RT thread は：

* immutable object の参照
* fixed-capacity buffer 操作
* relaxed atomic local state 更新

のみ許可。

---

# 2. Epoch / RCU 規約

---

## 2-1. Epoch authority は単一

禁止：

* EpochManager
* EpochCore
* local epoch timeline
* subsystem-local reclamation

唯一許可：

```cpp
EpochDomain
```

のみ。

---

## 2-2. retire 権限は RetireManager のみ

禁止：

```cpp
delete ptr;
free(ptr);
SnapshotFactory::destroy(ptr);
```

唯一許可：

```cpp
retireManager.retire(ptr);
```

AIは delete を追加してはならない。

---

## 2-3. Runtime object は epoch 管理必須

以下は必ず epoch 管理：

* RuntimeState
* DSPCore
* Transition
* ConvolverState
* EQCoeffCache
* runtime snapshot objects

例外：

* 永続 singleton
* static FFT plans
* startup-only immutable tables

---

## 2-4. retire は publish 後のみ

禁止：

```cpp
retire(oldState);
publish(newState);
```

必須順序：

```text
publish(newState)
HB
retire(oldState)
```

---

## 2-5. Epoch reader registration を省略禁止

Audio Thread が RuntimeState を読む箇所は必ず：

```cpp
EpochReaderGuard guard(domain);
```

または同等機構で保護。

---

# 3. Memory Order 規約

---

## 3-1. memory_order を推測してはならない

AIは禁止：

```cpp
memory_order_seq_cst をなんとなく追加
```

または禁止：

```cpp
全部 relaxed
```

各 atomic は causality.md の HB に従う。

---

## 3-2. publish は release

```cpp
current.store(newState, std::memory_order_release);
```

固定。

---

## 3-3. observe は acquire

```cpp
RuntimeState* s =
    current.load(std::memory_order_acquire);
```

固定。

---

## 3-4. reclaim comparison は acquire

reclaimer が epoch を読む際：

```cpp
load(memory_order_acquire)
```

必須。

---

## 3-5. RT-local state のみ relaxed 許可

例：

```cpp
remainingSamples.store(x, memory_order_relaxed);
```

ただし：

* 他スレッド非共有
* publication semantics 不要

である場合のみ。

---

# 4. Transition 規約

---

## 4-1. Transition は immutable

禁止：

```cpp
transition->remaining--;
```

許可：

```cpp
struct Transition {
    ...
};

struct ActiveTransition {
    const Transition* immutable;
    int remainingSamples;
};
```

---

## 4-2. superseded transition を destroy 禁止

禁止：

```cpp
abortFade();
destroy(oldTransition);
```

必須：

```cpp
retire(oldTransition);
```

---

## 4-3. completion flag 禁止

禁止：

```cpp
atomic<bool> fadeCompleted;
```

理由：

lost wakeup を引き起こす。

---

## 4-4. overlapping transition 禁止

PublicationCoordinator が linearization を保証。

AIは禁止：

* transition 同時進行
* multi-fade active
* bypass side transition

---

# 5. PublicationCoordinator 規約

---

## 5-1. commit queue を mutable coordination にしてはならない

禁止：

```cpp
std::queue<Commit>
```

理由：

queue state 自体が coordination plane になる。

---

## 5-2. append-only log を使用

許可：

```cpp
PublicationLog
```

のみ。

---

## 5-3. retire authority 集約

retire を実行可能なのは：

```cpp
PublicationCoordinator
↓
RetireManager
```

経路のみ。

---

## 5-4. RuntimeStore へ直接 publish 禁止

禁止：

```cpp
runtimeStore.publish(...)
```

を任意箇所から呼ぶこと。

唯一許可：

```cpp
PublicationCoordinator
```

経由。

---

# 6. Snapshot 設計規約

---

## 6-1. RuntimeState は shallow immutable aggregate

内部構造：

```cpp
RuntimeState {
    const DSPTopologySnapshot* topology;
    const RuntimeParameterSnapshot* params;
    const TransitionSnapshot* transition;
}
```

内部ポインタも immutable。

---

## 6-2. sub-snapshot 個別 publish 禁止

禁止：

```cpp
publish(params);
publish(topology);
```

理由：

publish coherence 崩壊。

---

## 6-3. topology mutation 禁止

DSP graph mutation は：

* rebuild
* clone
* publish

のみ。

---

# 7. RT安全規約

---

## 7-1. Audio Thread 禁止操作

絶対禁止：

* lock
* wait
* condition_variable
* heap allocation
* delete/free
* file IO
* logger
* printf
* exception throw
* dynamic_cast
* virtual ownership mutation

---

## 7-2. RT thread は immutable graph traversal のみ

RT thread の責務：

* snapshot observe
* DSP execute
* fade interpolation
* local state update

のみ。

---

## 7-3. branchy coordination logic 禁止

RT thread に：

* transaction arbitration
* ownership negotiation
* reclamation logic

を書いてはならない。

---

# 8. AI設計変更規約

---

## 8-1. AIは局所修正禁止

禁止：

* 単一関数のみ修正
* compile pass 優先
* 部分的 atomic patch

必須：

* causality chain 全体解析
* ownership chain 全体解析
* HB chain 全体解析

---

## 8-2. 修正前に必須提出物

AIは実装前に必ず以下を提示：

### 必須

1. 変更理由
2. ISR不変条件への影響
3. happens-before 変化
4. ownership 変化
5. retire/reclaim 変化
6. RT-safe 判定
7. publish coherence 判定
8. supersession 安全性
9. shutdown 安全性

これが欠けた実装は禁止。

---

## 8-3. 修正後に必須提出物

### 必須

* 新HB図
* 新ownership図
* retire flow
* reclaim flow
* RT safety summary
* added atomic一覧
* memory_order justification

---

# 9. 禁止ワード規約（AI暴走防止）

AIが以下を提案した場合は原則不合格：

* “簡単に”
* “とりあえず”
* “mutexで保護”
* “shared_ptrで管理”
* “atomic<bool>で通知”
* “queueで後回し”
* “thread-safeだから問題ない”
* “seq_cstに変更”
* “GC的に”
* “後でcleanup”

---

# 10. 実装単位規約

---

## 10-1. フェーズ跨ぎ実装禁止

AIは：

* Phase 1 の途中で Phase 3 修正
* Epoch統一前に Transition redesign

をしてはならない。

---

## 10-2. 各Phase完了条件

AIは各フェーズごとに：

* removed mutable states
* removed atomics
* removed ownership paths
* removed destruction paths

を列挙。

---

# 11. レビュー規約

---

## 11-1. “動く” は合格条件ではない

以下を全て満たす必要：

* ISR invariant preserved
* HB valid
* retire safe
* reclaim safe
* RT safe
* publish coherent
* ownership acyclic
* no mutable coordination plane

---

## 11-2. 新mutable state追加禁止

AIが新たな：

```cpp
atomic<bool>
atomic<int>
global singleton
mutable runtime cache
```

を追加した場合、
ISR違反として扱う。

---

# 12. 最終完成条件

ConvoPeq ISR完成条件：

## 必須条件

* RuntimeStore の atomic pointer のみが同期点
* Runtime parameter side-channel がゼロ
* Runtime mutable singleton がゼロ
* direct destroy path がゼロ
* dual epoch がゼロ
* RT ownership mutation がゼロ
* transition overlap がゼロ
* split publication がゼロ
* publish coherence violation がゼロ

---

# 13. AIへの最終指示

AIは以下を絶対に優先すること：

1. compile 成功より ISR invariant
2. 部分修正より causality preservation
3. パフォーマンスより correctness
4. convenience より ownership safety
5. mutable coordination 排除
6. publish coherence 維持
7. RT deterministic behavior

ISR は「RCUを使っている設計」ではない。

ISR とは：

```text
Immutable publish
+
single causality timeline
+
epoch-safe reclamation
+
immutable runtime coherence
+
RT-local mutable execution only
```

を満たす runtime system のみを指す。
