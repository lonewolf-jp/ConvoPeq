# ConvoPeq ISR移行 実装厳守規約（Phase別）

本規約は、提示された詳細設計書を実装へ落とし込む際に、AI・人間実装者双方が厳守すべき実装統制規約である。
目的は「局所修正によるISR不変条件破壊」を防止することにある。

---

# 0. 全Phase共通 絶対禁止事項

以下は全Phase共通で禁止する。

## 0.1 ISR不変条件破壊

禁止:

* publish後RuntimeState書換
* RuntimeState外mutable runtime state追加
* split publication
* side-channel atomic追加
* RuntimeState部分更新
* RuntimeState部分retire
* RuntimeState部分reclaim

禁止例:

```cpp
runtime->mix = newMix;
runtime->transition.remaining -= n;
atomic<bool> fadeDone;
```

---

## 0.2 RT違反

Audio Thread上で禁止:

* `new/delete`
* `malloc/free`
* `shared_ptr` copy/reset
* lock
* wait
* file I/O
* logger
* exception throw
* MessageManagerLock
* publish
* retire
* reclaim

---

## 0.3 atomic乱立禁止

禁止:

```cpp
foo.store(x);
bar.load();
```

許可:

```cpp
publishRuntimeState(...);
observeRuntimeState(...);
enqueueRetire(...);
```

atomicはhelper API以外から直接操作禁止。

---

## 0.4 legacy coexistence禁止

移行中でも禁止:

* 新旧runtime併存
* 新旧epoch併存
* old/new transition系混在
* mutable/immutable混在

必ず:

```text
移設 → 旧経路即削除
```

を1変更単位で実施する。

---

# 1. 実装単位規約

## 1.1 1変更 = 1責務

1 PR / 1 patch で複数Phase混在禁止。

禁止例:

* Phase1 + Phase3同時
* Epoch + Transition同時
* RuntimeStore + EQ cache同時

---

## 1.2 「動くようにする修正」禁止

ISR違反を隠す暫定修正禁止。

禁止例:

```cpp
std::mutex guard;
std::atomic<bool> workaround;
```

---

## 1.3 failure atomicity 必須

publish失敗時:

* current維持
* partially initialized state破棄
* retire登録禁止

禁止:

```cpp
publish(partialState);
if (failed) rollback();
```

---

# 2. Phase 0 実装規約（HB仕様固定）

# 2.1 仕様未固定実装禁止

以下が文書化されるまで実装禁止:

* HB chain
* memory_order
* reclaim inequality
* append linearization
* epoch advancement

---

## 2.2 memory_order 推測禁止

禁止:

```cpp
memory_order_relaxed // なんとなく速そう
```

必ず仕様書根拠をコメントで併記。

許可例:

```cpp
// publish HB observe
store(memory_order_release);
```

---

## 2.3 seq_cst禁止

`seq_cst` による「とりあえず安全化」禁止。

理由:

* HB設計崩壊を隠蔽する
* 因果関係未設計を覆い隠す

---

# 3. Phase 1 規約（Epoch統合）

# 3.1 dual epoch禁止

以下の併存禁止:

* EpochManager
* EpochCore
* g_deletionQueue
* 新EpochDomain

移行時は:

```text
接続 → 切替 → 旧削除
```

を同一変更内で完了。

---

## 3.2 ReaderSlot即時free禁止

禁止:

```cpp
delete slot;
```

必ず:

```text
retire -> epoch-safe reclaim
```

---

## 3.3 recursionDepth不整合禁止

必須:

* underflow assert
* overflow fail-fast
* nested observe整合

禁止:

```cpp
--depth;
if (depth < 0) ...
```

符号付き依存禁止。

---

## 3.4 drainAll前提違反禁止

Audio停止だけでdrain禁止。

必須:

* registration close
* worker stop
* join
* active reader zero

---

# 4. Phase 2 規約（Transition線形化）

# 4.1 mutable transition禁止

禁止:

```cpp
transition.remainingSamples--;
transition.finished = true;
```

mutable進行状態は:

```cpp
ActiveTransition
```

のみ。

---

## 4.2 abort path禁止

禁止:

```cpp
abortFade();
delete oldTransition;
```

必ず supersede。

---

## 4.3 completion side-channel禁止

禁止:

```cpp
atomic<bool> completed;
```

完了条件は:

```text
remaining == 0
&& transitionId一致
&& completionLatched == false
```

のみ。

---

## 4.4 PublicationLog shortcut禁止

禁止:

* bypass append
* direct consume
* direct reclaim
* reorder consume

---

## 4.5 MPSC helping省略禁止

禁止:

```cpp
if (next != nullptr)
    return;
```

必須:

```cpp
tail correction helping
```

---

## 4.6 append前node再利用禁止

禁止:

```cpp
reuseNode->next = nullptr;
```

recycled node禁止。

---

# 5. Phase 3 規約（Snapshot包含）

# 5.1 side-channel移植忘れ禁止

mutable runtime stateを残したままsnapshot化禁止。

必須確認:

* bypass
* mix
* latency
* fade state
* IR state
* cache state

---

## 5.2 split publication禁止

禁止:

```cpp
publish(runtime);
publish(params);
```

publish単位は:

```text
RuntimeState only
```

---

## 5.3 RuntimeState外coherence禁止

禁止:

```cpp
runtime + atomic mix + atomic bypass
```

runtime coherence形成はcurrentのみ。

---

# 6. Phase 4 規約（Cache寿命統合）

# 6.1 RT shared_ptr禁止

禁止:

```cpp
auto p = sharedCache;
```

RTは:

```cpp
const EQCoeffCache*
```

のみ。

---

## 6.2 cache global ownership禁止

禁止:

```cpp
global cache registry
singleton cache lifetime
```

寿命はsnapshot従属のみ。

---

## 6.3 cache partial retire禁止

cache単独寿命禁止。

cache寿命:

```text
RuntimeState と一致
```

---

# 7. Phase 5 規約（責務分離）

# 7.1 God object延命禁止

SnapshotCoordinatorへ新責務追加禁止。

移行中でも禁止:

* utility追加
* temporary routing
* facade集中

---

## 7.2 publish authority漏洩禁止

禁止:

```cpp
runtimeStore.publish(...)
```

許可:

```cpp
publicationCoordinator.commit(...)
```

のみ。

---

## 7.3 retire authority分散禁止

禁止:

```cpp
retire(oldState);
epoch.retire(...);
delete old;
```

必ず:

```text
PublicationCoordinator
 -> RetireManager
 -> EpochDomain
```

---

# 8. Phase 6 規約（最適化）

# 8.1 correctness未完最適化禁止

Phase0-5完了前:

* structural sharing禁止
* pooling禁止
* partial snapshot禁止

---

## 8.2 部分共有証明必須

sharing導入時は:

* publish coherence
* retire correctness
* reclaim correctness
* ABA safety

を証明文書化必須。

---

## 8.3 benchmark無し最適化禁止

必須:

* rebuild latency
* retired bytes
* reclaim cadence
* RT callback jitter

定量比較。

---

# 9. コーディング規約

# 9.1 RuntimeState const化

publish後変更防止のため:

```cpp
const RuntimeState*
```

を徹底。

---

## 9.2 observer裸ポインタ禁止

禁止:

```cpp
auto* r = observe();
```

必須:

```cpp
ObservedRuntime observed = observeRuntimeState();
```

---

## 9.3 move-only厳守

ObservedRuntime:

* copy ctor禁止
* copy assign禁止

static_assert必須。

---

## 9.4 constructor escape禁止

禁止:

```cpp
global = this;
```

fully-constructed before publish。

---

## 9.5 helper外memory_order禁止

memory_order指定はhelper内部のみ。

---

# 10. レビュー規約

# 10.1 「何を削除したか」を必須化

レビュー提出時必須:

* removed atomics
* removed mutable states
* removed ownership paths
* removed destruction paths

追加機能だけの報告禁止。

---

## 10.2 HB説明必須

atomic変更時必須:

* release元
* acquire先
* HB chain
* safe reclaim根拠

---

## 10.3 retire flow図必須

retire関連変更時必須:

```text
publish
 -> retire enqueue
 -> epoch safe
 -> reclaim
 -> destroy
```

---

# 11. CI / lint 規約

# 11.1 atomic dot-call禁止

CIで以下検出時fail:

```text
.store(
.load(
.exchange(
.compare_exchange
```

helper外禁止。

---

## 11.2 RT禁止API検査

RT pathでfail:

* mutex
* shared_ptr
* malloc
* new
* delete
* printf
* Logger
* filesystem
* MessageManagerLock

---

## 11.3 observe leak検査

検出対象:

* guardなし保存
* callback外保持
* cross-thread move

---

# 12. 実装者行動規約

# 12.1 「局所最適」禁止

修正時は常に確認:

* HB
* ownership
* retire
* reclaim
* RT-safe
* publish coherence

局所修正のみ禁止。

---

## 12.2 「一時的mutable」禁止

禁止:

```cpp
あとでimmutable化する
```

ISRでは中間状態が最終破綻原因になる。

---

## 12.3 「コンパイル優先」禁止

優先順位:

```text
ISR invariant
 > memory safety
 > coherence
 > RT safety
 > compile success
 > performance
```

---

# 13. 最終受入基準

以下がゼロでなければ不合格。

* runtime side-channel
* split publication
* dual epoch
* direct destroy
* RT ownership mutation
* mutable runtime state
* completion side-channel
* transition overlap
* callback multiple observe
* unsafe reclaim inequality
* partially initialized publish
* stale ReaderSlot reuse
* non-helper atomic operation
* shared_ptr on RT
* reclaim outside EpochDomain
* publish outside PublicationCoordinator
