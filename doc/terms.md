# AI実装担当者 遵守事項

## ConvoPeq 「Immutable Runtime Graph + Per-thread DSP State」移行プロジェクト

本書は、AI に詳細設計・実装・リファクタリングを行わせる際に、必ず遵守させるべき事項を定義する。
対象は：

* architecture design
* DSP implementation
* memory management
* threading
* runtime transition
* JUCE integration
* lock-free safety
* reclamation
* SIMD optimization
* offline rendering

を含む全領域である。

このプロジェクトでは、

```text
「動くコード」
```

ではなく、

```text
「長期運用で破綻しないRT-safe architecture」
```

を最優先とする。

---

# 1. 最重要原則

---

# 1.1 「mutable DSP object」を増やさない

AI が最も破壊しやすい点。

禁止：

```cpp
class ConvolverProcessor
{
    void process(...)
    {
        internalHistory.push(...);
    }
};
```

理由：

```text
runtime object と execution state が混在する
```

ため。

---

# 1.2 immutable graph に mutable state を絶対に入れない

禁止：

```cpp
struct RuntimeGraph
{
    mutable ScratchBuffer scratch;
};
```

禁止：

```cpp
const RuntimeGraph&
↓
内部で lazy init
```

禁止：

```cpp
static cache
```

禁止：

```cpp
thread_local inside graph node
```

---

# 1.3 process() は純DSP実行のみ

process() 内で禁止：

* allocation
* lock
* vector resize
* logging
* file IO
* atomic ownership mutation
* shared_ptr graph rebuild
* FFT plan creation
* coeff recomputation
* IR loading

---

# 1.4 Audio thread は ownership を持たない

禁止：

```cpp
shared_ptr reset
unique_ptr delete
retire enqueue
```

audio thread 内実行。

許可：

```cpp
raw pointer read only
```

---

# 1.5 「prepare/reset/process」のJUCE流儀をそのまま持ち込まない

AI は JUCE 的 OOP を再導入しやすい。

禁止：

```cpp
processor.prepare();
processor.reset();
processor.process();
```

最終形：

```cpp
process(graph, state, block);
```

---

# 2. AI設計時の必須分類

AI は実装前に、全メンバを必ず以下へ分類する。

---

# 2.1 immutable runtime data

例：

* FIR coeff
* FFT partition metadata
* EQ coefficients
* routing topology
* oversampling topology
* latency metadata

特徴：

```text
publish後変更禁止
```

---

# 2.2 per-thread mutable state

例：

* overlap-add history
* delay line
* biquad z state
* smoothing memory
* temporary FFT buffer
* scratch arena

特徴：

```text
audio execution専用
```

---

# 2.3 UI state

例：

* visualization cache
* editor selection
* analyzer display
* preview waveform

---

# 2.4 worker-thread resource

例：

* FFT plan cache
* IR decode temp
* file load scratch

---

# 2.5 明示分類義務

AI は新規 member を追加する際：

```cpp
// IMMUTABLE_RUNTIME
// DSP_THREAD_STATE
// UI_ONLY
// WORKER_ONLY
```

を必須コメントとして付与。

分類不能なら実装禁止。

---

# 3. AIが絶対にやってはいけないこと

---

# 3.1 mutable singleton

禁止：

```cpp
GlobalFFTCache::instance()
```

---

# 3.2 hidden static state

禁止：

```cpp
static std::vector<float> scratch;
```

---

# 3.3 thread_local依存

原則禁止。

理由：

* host thread migration
* offline render
* future parallel rendering
* nested graph execution

で破綻。

---

# 3.4 Audio thread 内 shared_ptr ownership mutation

禁止：

```cpp
std::shared_ptr<T> local = atomicLoad(...);
```

ownership increment が RT unsafe。

---

# 3.5 「RCUだからmutableでもOK」理論

完全禁止。

RCU は：

```text
lifetime safety
```

のみ。

thread safety を保証しない。

---

# 3.6 lazy initialization

禁止：

```cpp
if (!initialized)
    init();
```

process 中。

---

# 3.7 prepare() 時 realloc 前提

禁止。

prepare は：

```text
max block size reservation
```

のみ。

実際の mutable execution memory は：

```text
DSPExecutionState allocator
```

が管理。

---

# 4. AIが必ず維持すべき不変条件

---

# 4.1 RuntimeGraph は publish 後 immutable

AI は：

```cpp
const RuntimeGraph*
```

を受けたら、

```text
絶対に変更不可
```

とみなす。

---

# 4.2 DSPExecutionState は thread confined

複数 audio thread 共有禁止。

---

# 4.3 process() は deterministic

同一：

* graph
* state
* input

なら必ず同一結果。

---

# 4.4 rebuild thread と audio thread 分離

禁止：

```cpp
audio thread rebuild
```

---

# 4.5 Runtime publish は atomic pointer swap のみ

禁止：

* deep copy
* ownership rebuild
* runtime mutation

---

# 5. AIによる詳細設計時の必須出力

AI に設計させる場合、必ず以下を出力させる。

---

# 5.1 ownership table

例：

| Object            | Owner          | Thread        | Lifetime        |
| ----------------- | -------------- | ------------- | --------------- |
| RuntimeGraph      | RuntimeManager | Builder→Audio | immutable       |
| DSPExecutionState | AudioEngine    | Audio         | thread confined |

---

# 5.2 mutation table

例：

| Member         | Mutable | Thread     |
| -------------- | ------- | ---------- |
| coeffs         | no      | N/A        |
| overlapHistory | yes     | audio only |

---

# 5.3 RT safety table

例：

| Function | RT Safe | Why              |
| -------- | ------- | ---------------- |
| process  | yes     | no alloc/no lock |
| rebuild  | no      | heap allocation  |

---

# 5.4 destruction table

AI は destruction path を必ず提示。

特に：

* fade end
* queued runtime discard
* offline reset
* plugin shutdown

---

# 5.5 allocation table

必須。

例：

| Allocation   | Thread  | Timing       |
| ------------ | ------- | ------------ |
| FFTPlan      | Builder | rebuild only |
| ScratchArena | prepare | once         |

---

# 6. AIレビュー時の必須検査項目

AI に self-review させる。

---

# 6.1 mutable leakage check

確認：

```text
immutable graph 内に mutable state が混入していないか
```

---

# 6.2 RT allocation check

確認：

```text
process 内 allocation 0
```

---

# 6.3 ownership mutation check

確認：

```text
audio thread shared_ptr mutation 0
```

---

# 6.4 reclamation correctness check

確認：

```text
UAF path が存在しないか
```

---

# 6.5 offline rendering check

確認：

```text
thread_local 依存がないか
```

---

# 6.6 host reentrancy check

確認：

```text
nested process 呼び出しで壊れないか
```

---

# 7. AI実装時の推奨ワークフロー

---

# Step 1

現行コード解析

必須：

* member分類
* thread分析
* ownership分析
* mutation分析

---

# Step 2

state split design

```text
immutable
vs
per-thread mutable
```

を明示。

---

# Step 3

destruction設計

最重要。

AI は destruction を軽視しやすい。

---

# Step 4

process API redesign

最初に固定。

後から変えると破綻。

---

# Step 5

crossfade redesign

runtime ownership ではなく：

```text
state duplication
```

へ変更。

---

# Step 6

RT audit

必須：

* lock
* alloc
* atomic contention
* ownership mutation

全排除。

---

# 8. AIに禁止すべき「安易な修正」

---

# 8.1 mutex追加

AI は race を mutex で塞ぎやすい。

audio thread mutex 禁止。

---

# 8.2 atomic乱用

atomic は RT-safe ではない。

特に：

* shared_ptr atomic
* refcount atomic
* contention-heavy atomic

は禁止。

---

# 8.3 virtual dispatch増殖

audio thread polymorphism 多用禁止。

node graph は：

* enum dispatch
* CRTP
* static polymorphism

優先。

---

# 8.4 「一時的mutable cache」

最終的に必ず腐敗源になる。

禁止。

---

# 9. AIに要求すべきコードコメント

AI に必ず以下を書かせる。

---

# immutable graph

```cpp
// Immutable after publish.
// Audio thread may access lock-free.
```

---

# DSP state

```cpp
// Thread-confined DSP execution state.
// Never shared across audio threads.
```

---

# RT-safe function

```cpp
// RT-safe:
// - no allocation
// - no locks
// - no ownership mutation
```

---

# 10. AIが理解すべき本質

このプロジェクトの本質は：

```text
「mutable object を安全化する」
```

ではない。

本質は：

```text
「runtime object から mutable execution state を完全分離する」
```

ことである。

---

# 11. AIへの最終要求

AI は常に：

```text
これは
- immutable runtime dataか？
- per-thread execution stateか？
- worker resourceか？
- UI stateか？
```

を明示分類しなければならない。

分類できない実装は禁止。

---

# 12. 実装レビュー時の最重要質問

AI実装レビュー時は、必ず以下を確認する。

---

## Q1

```text
この mutable state はなぜ immutable graph に存在しているのか？
```

---

## Q2

```text
この ownership mutation は audio thread で発生しないか？
```

---

## Q3

```text
crossfade は graph 切替か？
state duplicationか？
```

---

## Q4

```text
offline render で thread_local が破綻しないか？
```

---

## Q5

```text
prepare/reset に hidden allocation はないか？
```

---

# 13. 最終的にAIが到達すべき構造

```text
RuntimeGraph
    完全 immutable
        ↓

DSPExecutionState
    thread confined mutable

        ↓

process(graph, state, block)
```

これ以外の方向へ戻る実装は、原則として拒否すべきである。
