# ConvoPeq 現行ソースコード問題点整理

## — ISR Bridge Runtime 観点 詳細監査 —

以下は、現行 `ConvoPeq.md` のコードベースを ISR Bridge Runtime の観点から精査した結果を、構造的に整理したものである。

単なる「コード臭」ではなく、

> 長時間実運用で破綻要因になりうる runtime semantic 問題

を中心に分類している。

---

# 1. 根本問題

ConvoPeq 現行コードの本質問題は：

> Runtime meaning source が単一化されていない

ことである。

これは単なる設計美の問題ではなく、以下を引き起こす：

* stale observe
* mixed-generation execution
* semantic drift
* duplicate authority
* retire ambiguity
* transition divergence

つまり：

> 「同時に複数の runtime truth が存在しうる」

という構造問題である。

---

# 2. 問題分類一覧

| 分類                           | 危険度      | 状態  |
| ---------------------------- | -------- | --- |
| Observe Path Duplication     | Critical | 未解決 |
| Authority Duplication        | Critical | 未解決 |
| RuntimeWorld 非自己完結           | Critical | 未解決 |
| Generation Semantic 混在       | Critical | 未解決 |
| Crossfade Semantic 分散        | Critical | 未解決 |
| Runtime Ownership 混乱         | Critical | 未解決 |
| Retire Governance 不完全        | High     | 未解決 |
| Partial Publication          | High     | 未解決 |
| Legacy Mutable Runtime 共存    | Critical | 未解決 |
| Semantic Validation 不足       | High     | 未解決 |
| Executor Leakage             | High     | 未解決 |
| Runtime Topology 分散          | High     | 未解決 |
| Runtime Visibility State 混在  | High     | 未解決 |
| Publication Coordinator 責務逸脱 | High     | 未解決 |
| Long-run Soak 安定性不足          | Critical | 未解決 |

---

# 3. Observe Path Duplication（最重要）

---

## 問題概要

AudioThread が runtime を複数経路から観測している。

---

## 現在の observe source

現状コードでは AudioThread が：

```cpp
RuntimeWorld
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
preparedCrossfade
TransitionState
RuntimeGraph.activeNode
RuntimeGraph.fadingNode
```

等を経由して runtime semantic を取得している。

---

## 問題本質

ISR Runtime の原則は：

```cpp
const RuntimeWorld* world
```

のみが observable runtime source であること。

しかし現状は：

> observe path が並列多重化

している。

---

## 実害

以下が発生しうる：

### mixed-generation observe

```text
RuntimeWorld = generation 101
activeRuntimeDSPSlot = generation 102
fadingRuntimeDSPSlot = generation 100
```

---

### stale fade observe

fade semantic が world と executor で不一致。

---

### transition divergence

crossfade 完了タイミングが複数 semantic source に分散。

---

## 危険度

Critical。

これは ISR Runtime の最重要違反。

---

# 4. Authority Duplication

---

## 問題概要

runtime authority が単一でない。

---

## 現在の authority source

現状：

```cpp
RuntimeWorld
RuntimePublicationCoordinator
AudioEngine mutable state
TransitionState
runtimeVersion
generation
transitionId
```

が authority 的意味を持っている。

---

## 問題本質

runtime meaning authority が：

```text
single-source-of-truth
```

になっていない。

---

## 実害

### rebuild semantic split

publish 時：

* coordinator semantic
* engine semantic
* transition semantic

が部分的にズレる。

---

### retire authority ambiguity

どの generation が authoritative なのか不明確になる。

---

### visibility race

observable state と actual state が乖離。

---

# 5. RuntimeWorld 非自己完結

---

## 問題概要

RuntimeWorld が immutable semantic runtime ではない。

---

## 現状

RuntimeWorld は：

```cpp
void* activeNode;
void* fadingNode;
```

等の external object reference を保持。

所有権は AudioEngine 側。

---

## 問題本質

RuntimeWorld が：

> observer snapshot

に過ぎない。

---

## ISR 的完成像

本来は：

```cpp
RuntimeWorld
    owns semantic runtime
```

である必要がある。

---

## 実害

### external semantic dependency

world 単独では runtime semantic が成立しない。

---

### retire mismatch

world lifetime と DSP lifetime が分離。

---

### stale pointer risk

retire timing により dangling semantic が発生しうる。

---

# 6. Generation Semantic 混在

---

## 問題概要

runtime identity が単一でない。

---

## 現状

以下が混在：

```cpp
generation
runtimeVersion
transitionId
closureId
```

---

## 問題本質

semantic identity collapse 未達。

---

## 特に危険

```cpp
runtimeVersion != 0
    ? runtimeVersion
    : generation
```

fallback semantic。

---

## 実害

### generation drift

runtime semantic が複数 generation source に依存。

---

### diagnostic semantic pollution

本来 diagnostic の runtimeVersion が semantic authority に混入。

---

### overlap reject inconsistency

transition overlap 判定が generation semantic に依存できない。

---

# 7. Crossfade Semantic 分散

---

## 問題概要

crossfade が observable semantic state になっている。

---

## 現状分散先

```cpp
RuntimeGraph
EngineRuntime
TransitionState
FadeAccumulator
CrossfadeAuthorityRuntime
preparedCrossfade
```

---

## 問題本質

crossfade が：

> implementation detail

ではなく：

> runtime semantic

になっている。

---

## 実害

### double retire

fade completion semantic が複数箇所に存在。

---

### overlap semantic divergence

crossfade overlap reject が不整合。

---

### orphan runtime

fade semantic drift により retire 漏れ。

---

# 8. Runtime Ownership 混乱

---

## 問題概要

runtime ownership が collapse していない。

---

## 現状

```cpp
RuntimeWorld -> observe
AudioEngine -> own
RetireRuntime -> reclaim
TransitionState -> visibility
```

---

## 問題本質

ownership semantic が分裂。

---

## 実害

### lifetime ambiguity

どこが authoritative owner か不明。

---

### reclaim ambiguity

retire と ownership が分離。

---

### hidden dependency

world semantic が external mutable state に依存。

---

# 9. Retire Governance 不完全

---

## 問題概要

retire overload handling が未完成。

---

## 現状

```cpp
erase(retiredWorlds_.begin())
```

silent truncate。

---

## 問題本質

pressure semantic 不在。

---

## 実害

### hidden reclaim loss

silent retire drop。

---

### long-run backlog collapse

高頻度 rebuild 時に reclaim semantic が破綻。

---

### soak instability

数時間後に retire drift。

---

# 10. Partial Publication

---

## 問題概要

runtime semantic が atomic publish されていない。

---

## 現状

publish semantic が：

```cpp
graph
fade
transition
runtime slots
```

へ分散。

---

## 問題本質

field-level publish が存在。

---

## 実害

### partial visibility

AudioThread が中間状態を観測。

---

### semantic tearing

world semantic と fade semantic が不一致。

---

# 11. Legacy Mutable Runtime 共存

---

## 問題概要

ISR Runtime と legacy mutable runtime が同時存在。

---

## 現状

以下が依然 active：

```cpp
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
legacy transitions
legacy atomics
legacy visibility state
```

---

## 問題本質

Bridge Runtime が：

> overlay architecture

になっている。

---

## 実害

### semantic competition

ISR と legacy が runtime authority を競合。

---

### rebuild divergence

一部 rebuild が mutable path を通る。

---

### hidden runtime branching

semantic source が複数化。

---

# 12. Validation 不足

---

## 現状 validator が検出できるもの

* cycle
* ownership
* mutability
* dependency

---

## 検出できないもの

* observe duplication
* authority duplication
* mixed-generation semantic
* partial publication
* crossfade semantic overlap
* executor leakage

---

## 問題本質

semantic correctness verifier が不足。

---

# 13. Executor Leakage

---

## 問題概要

executor-local state が observable runtime へ漏れている。

---

## 例

```cpp
dspCrossfadePending
queuedFadeTimeSec
latencyResetPending
```

---

## 問題本質

executor ephemeral state が runtime semantic 化。

---

## 実害

### runtime semantic inflation

runtime meaning が executor implementation に依存。

---

### replay instability

runtime reproducibility が崩壊。

---

# 14. Publication Coordinator 責務逸脱

---

## 現状

Coordinator が：

```cpp
retiredWorlds_
version_
swapPending_
visibility semantic
```

を持つ。

---

## 問題本質

Coordinator が：

> semantic runtime container

化している。

---

## 本来

Coordinator は：

```text
publication arbitration only
```

であるべき。

---

# 15. Runtime Topology 分散

---

## 問題概要

runtime topology が複数構造に分散。

---

## 現状

```cpp
RuntimeGraph
EngineRuntime
TransitionState
DSP slots
Execution ordering
```

---

## 実害

### topology semantic drift

runtime graph semantic が単一でない。

---

### inconsistent execution

execution order と transition semantic が分離。

---

# 16. Long-run Soak 不安定

---

## 現状評価

短時間運用では比較的安定。

しかし：

| 条件                | 評価 |
| ----------------- | -- |
| 長時間運用             | 危険 |
| 高頻度 rebuild       | 危険 |
| IR reload storm   | 危険 |
| automation storm  | 危険 |
| sample-rate churn | 危険 |

---

## 主因

### semantic drift accumulation

小さな semantic 不一致が累積。

---

### retire backlog drift

reclaim semantic が長時間で崩壊。

---

### mixed-generation visibility

長時間で stale observe が発生。

---

# 17. 本質的総括

現行 ConvoPeq は：

> ISR Runtime が完成した状態

ではない。

正確には：

> legacy mutable runtime の上に ISR semantic layer を追加し始めた段階

である。

つまり：

* publication abstraction はある
* validation layer もある
* retire runtime もある

しかし：

> runtime meaning source collapse

が未達。

---

# 18. 最重要改善対象

優先順位順：

## 最優先

1. Observe Path Collapse
2. Authority Collapse
3. RuntimeWorld Self-contained化

---

## 次点

4. Crossfade Executor-local化
5. Retire Governance
6. Legacy Runtime Removal

---

# 19. 最終評価

現状コードは：

* 「ISR風 runtime」
  ではあるが、
* 「Single Authoritative Observable Runtime」

ではない。

最大問題は：

> runtime semantic が単一 source に collapse していないこと

である。

これが：

* semantic drift
* stale observe
* mixed-generation runtime
* reclaim ambiguity
* long-run instability

の根本原因になっている。
