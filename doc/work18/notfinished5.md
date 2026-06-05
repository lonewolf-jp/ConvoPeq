最新版 ConvoPeq.md を見る限り、Practical Stable ISR Bridge Runtime の「骨格」はほぼ移行済みですが、**まだ未達成と判断できる項目が複数残っています。**

なお、以下は単なる理想論ではなく、

* 実運用で破綻しにくい
* Authority の単一化
* Semantic/Execution 分離
* RuntimeWorld 単一権威

という Practical Stable ISR Bridge Runtime の目的に照らした評価です。

---

# ① Legacy Commit Path がまだ残存

## 1. 未達成内容

旧 PublicationIntent 系アーキテクチャが完全削除されていない。

Practical Stable ISR Bridge Runtime の最終形では

```text
Builder
 ↓
Coordinator.publishWorld()
 ↓
RuntimeStore
```

のみになるべきですが、

現在は

```text
pendingCommit
 ↓
processPendingCommit
 ↓
applyRuntimeCommitFromIntent
```

経路が残っています。

---

## 2. 該当箇所

### processPendingCommit

```cpp
applyRuntimeCommitFromIntent(dsp, gen, snap);
```



---

### applyRuntimeCommitFromIntent

```cpp
void AudioEngine::applyRuntimeCommitFromIntent(...)
```

---

### 監査計画でも削除対象

```text
applyRuntimeCommitFromIntent 残存 → CI Error
```

---

## 3. あるべき姿

```text
RuntimeBuilder
     ↓
RuntimePublishWorld
     ↓
RuntimePublicationCoordinator.publishWorld()
```

のみ。

AudioEngine が commit orchestration を持たない。

---

## 4. 改修方法

* processPendingCommit 削除
* applyRuntimeCommitFromIntent 削除
* commit intent 構造削除
* publishWorld() 直結

---

# ② RuntimeBuildSnapshot がまだ Authority 化されている

## 1. 未達成内容

sealedSnapshot が RuntimeWorld 構築の権威として使われている。

Practical Stable ISR Bridge Runtime では

```text
RuntimeWorld
↑
唯一の権威
```

であるべき。

---

## 2. 該当箇所

RuntimeBuilder

```cpp
if (sealedSnapshot != nullptr)
```

```cpp
const auto& sealedBuildInput =
    sealedSnapshot->buildInput;
```

```cpp
worldOwner->routing.processingOrder =
    sealedBuildInput.processingOrder;
```

```cpp
worldOwner->routing.eqBypassed =
    sealedBuildInput.eqBypassed;
```

```cpp
worldOwner->routing.convBypassed =
    sealedBuildInput.convBypassed;
```



---

## 3. あるべき姿

```text
RuntimeBuilder
   ↓
現在の Engine State
   ↓
RuntimeWorld
```

Snapshot は診断用途のみ。

---

## 4. 改修方法

* buildRuntimePublishWorld() から sealedSnapshot 引数削除
* RuntimeBuildSnapshot を診断専用へ降格
* RuntimeWorld を直接構築

---

# ③ Retire Authority が完全単一化されていない可能性

## 1. 未達成内容

Coordinator が retire を持つ一方、

AudioEngine 側も retireDSP() を実行している。

Authority が分散している可能性がある。

---

## 2. 該当箇所

```cpp
retireDSP(prev);
```

---

```cpp
retireDSP(newDSP);
```



---

Coordinator 側

```cpp
willRetireRuntimeNonRt(...)
```

```cpp
retireRuntimePublishWorldNonRt(...)
```

---

## 3. あるべき姿

```text
Coordinator
    ↓
Retire
```

のみ。

AudioEngine が retire を直接決定しない。

---

## 4. 改修方法

* retireDSP() 呼び出し元を全監査
* retire enqueue を Coordinator 経由へ統一
* AudioEngine は retire request のみ生成

---

# ④ Crossfade Decision が AudioEngine に残存

## 1. 未達成内容

Crossfade 判定が RuntimeWorld の外に存在。

---

## 2. 該当箇所

```cpp
CrossfadeContext
```

```cpp
needsCrossfade
```

```cpp
oldHasIR
```

```cpp
newHasIR
```

```cpp
fadeTimeSec
```

---

## 3. あるべき姿

```text
RuntimeWorld
    ↓
TransitionPolicy
    ↓
Executor
```

Execution 側のみ参照。

---

## 4. 改修方法

CrossfadeContext を

```cpp
RuntimeTransitionExecutor
```

へ移動。

publish 判定から切り離す。

---

# ⑤ Active/Fading DSP Slot が依然として中心的存在

## 1. 未達成内容

RuntimeWorld が主権であるべきなのに、

依然として

```cpp
activeRuntimeDSPSlot
```

```cpp
fadingRuntimeDSPSlot
```

が強く使われている。

---

## 2. 該当箇所

```cpp
activeRuntimeDSPSlot
```

```cpp
fadingRuntimeDSPSlot
```

```cpp
getActiveRuntimeDSP()
```

```cpp
exchangeFadingRuntimeDSP()
```



---

さらに

```cpp
DSPCore* atomicCurrent =
    getActiveRuntimeDSP();
```

---

## 3. あるべき姿

Semantic Decision:

```text
RuntimeWorld
```

のみ。

Execution Access:

```text
getActiveRuntimeDSP()
```

許容。

---

## 4. 改修方法

P11 Audit 実施。

呼び出し元を全て調査。

Semantic 分岐に使われていれば排除。

---

# ⑥ RuntimeStore Authority がまだ完全封鎖されていない

## 1. 未達成内容

Coordinator が唯一の Store 更新主体である保証が確認できない。

---

## 2. 該当箇所

```cpp
RuntimePublishStore runtimeStore;
```

```cpp
makeRuntimePublicationCoordinator()
```

---

監査計画でも

```text
RuntimeStore Authority Audit
```

が残っている。

---

## 3. あるべき姿

```text
RuntimeStore
↑
Coordinatorのみ
```

---

## 4. 改修方法

* publish API を private 化
* friend Coordinator 化
* 全 caller を監査

---

# ⑦ RuntimeWorld Construction Authority が未完成

## 1. 未達成内容

Builder が主権になったが、

RuntimePublishWorld を他所で生成できる可能性が残る。

---

## 2. 該当箇所

Builder 専用生成保証が見当たらない。

監査計画でも

```text
RuntimeWorld Construction Audit
```

が残っている。



---

## 3. あるべき姿

```text
RuntimeBuilder
のみ生成可能
```

---

## 4. 改修方法

```cpp
RuntimePublishWorld
```

コンストラクタ private。

```cpp
friend RuntimeBuilder;
```

化。

---

# ⑧ Observe Path 単一化が未証明

## 1. 未達成内容

RuntimeWorld が唯一の観測経路になったことがソース上からまだ証明できない。

---

## 2. 該当箇所

監査対象として

```cpp
getActiveRuntimeDSP
exchangeFadingRuntimeDSP
resolve*RuntimeWorldOnly
```

が残っている。

---

## 3. あるべき姿

Audio Thread の意味判断は

```text
RuntimeWorld
```

のみ。

---

## 4. 改修方法

serena/codegraph による全 caller 監査。

---

# 優先順位

実運用で重要度順に並べると、

| 優先 | 項目                              |
| -- | ------------------------------- |
| S  | Legacy Commit Path 削除           |
| S  | Snapshot Authority 削除           |
| S  | Retire Authority 単一化            |
| A  | Observe Path 監査                 |
| A  | Active/Fading DSP の Semantic 排除 |
| A  | Crossfade Semantic Leakage 排除   |
| B  | RuntimeStore 封鎖                 |
| B  | RuntimeWorld Construction 封鎖    |

特に **① Legacy Commit Path** と **② Snapshot Authority** は、提示されている Practical Stable ISR Bridge Runtime 計画の完了条件と照らしても、最新版 ConvoPeq.md 上で未達成が明確に確認できる最重要項目です。



前回までに提示した

* Legacy Commit Path
* Snapshot Authority
* Retire Authority
* Crossfade Decision
* Active/Fading DSP Semantic依存
* RuntimeStore Authority
* RuntimeWorld Construction Authority
* Observe Path

以外で、ConvoPeq.md の Practical Stable ISR Bridge Runtime 完了判定を厳密に見ると、さらに監査対象となる箇所があります。

ただし今回は、

**「まだ未達成である可能性が高いもの」**
のみを挙げます。

---

# ⑨ RuntimePublicationCoordinator が完全な State Machine Authority になっていない可能性

## 1. 未達成内容

Coordinator が導入されているものの、

State 遷移の決定権が AudioEngine 側に残っている可能性があります。

Practical Stable ISR Bridge Runtime では

```text
Coordinator
  ↓
Publishing
Transitioning
Pressure
ShuttingDown
Faulted
```

の状態遷移を Coordinator が単独で管理するべきです。

---

## 2. 該当箇所

Coordinator 側

```cpp
PublicationState
```

```cpp
transitionTo(...)
```

```cpp
setFaulted(...)
```

など。

一方で AudioEngine 側に

```cpp
isCrossfading()
```

```cpp
isTransitioning()
```

```cpp
pendingCommit
```

などが残っている場合、

State Authority が分散している可能性があります。

---

## 3. あるべき姿

```text
Coordinator
    ↓
PublicationState
```

唯一。

AudioEngine は

```text
request
```

のみ送る。

---

## 4. 改修方法

* state変更APIをCoordinatorに集約
* AudioEngineの状態変更コード削除
* Coordinator外からstate変更不可化

---

# ⑩ RuntimeWorld が Immutable Contract を完全達成していない可能性

## 1. 未達成内容

RuntimeWorld 公開後の可変性が残っている可能性があります。

Practical Stable ISR Bridge Runtime では

```text
build
 ↓
freeze
 ↓
publish
```

です。

---

## 2. 該当箇所

典型的には

```cpp
world->routing.xxx = ...
```

```cpp
world->semantic.xxx = ...
```

のような公開後更新。

RuntimePublishWorld の public フィールド。

---

## 3. あるべき姿

```cpp
const RuntimePublishWorld
```

相当。

Publish後変更不能。

---

## 4. 改修方法

* mutable field削減
* getter化
* Builderのみ書込可能化

---

# ⑪ RuntimeSemantic と RuntimeExecution の境界が未完全

## 1. 未達成内容

Semantic と Execution の責務分離がまだ完全ではない可能性があります。

---

## 2. 該当箇所

典型例

```cpp
DSPCore
```

を参照して

```cpp
if (...)
```

判定する箇所。

---

また

```cpp
hasIR()
```

```cpp
getLatency()
```

```cpp
getConvolutionState()
```

などが

意味判断に使われるケース。

---

## 3. あるべき姿

Semantic判定は

```text
RuntimeWorld
```

のみ。

DSPCore参照は禁止。

---

## 4. 改修方法

P11監査を拡張。

以下を全検索。

```text
hasIR
latency
convolution
getActiveRuntimeDSP
DSPCore*
```

---

# ⑫ RuntimeGraph が依然として Semantic Source になっている可能性

## 1. 未達成内容

RuntimeWorld 導入後も

RuntimeGraph が意味判断ソースとして使われている可能性があります。

---

## 2. 該当箇所

検索対象

```cpp
RuntimeGraph
```

```cpp
routingGraph
```

```cpp
graphNode
```

---

特に

```cpp
if (graph...)
```

は要注意。

---

## 3. あるべき姿

```text
RuntimeGraph
↓
Execution only
```

---

Semantic は

```text
RuntimeWorld
```

のみ。

---

## 4. 改修方法

RuntimeGraph caller監査。

Decision利用を禁止。

---

# ⑬ Rebuild Admission が RuntimeWorld のみで完結していない可能性

## 1. 未達成内容

Rebuild必要判定が

RuntimeWorld以外の状態に依存している可能性があります。

---

## 2. 該当箇所

典型的には

```cpp
lastBuildState
```

```cpp
previousDSP
```

```cpp
activeDSP
```

比較。

---

## 3. あるべき姿

```text
Intent
 ↓
BuildInput
 ↓
Validator
```

のみ。

---

## 4. 改修方法

RebuildAdmission周辺の全分岐監査。

---

# ⑭ EpochDomain が Semantic Authority を持っている可能性

## 1. 未達成内容

EpochDomain は本来

```text
Execution lifetime
```

管理のみ。

---

しかし

```cpp
epoch
generation
activation
```

の意味判断に利用されている可能性があります。

---

## 2. 該当箇所

検索対象

```cpp
EpochDomain
```

```cpp
activationEpoch
```

```cpp
currentEpoch
```

---

## 3. あるべき姿

```text
GenerationSemantic
```

が唯一の意味権威。

---

## 4. 改修方法

EpochDomain利用箇所を全監査。

Semantic分岐排除。

---

# ⑮ RuntimePublicationValidator が Authority Validator まで担っている

## 1. 未達成内容

Validator が Validation を超えて Decision を持つ可能性があります。

---

## 2. 該当箇所

```cpp
validatePublication(...)
```

内部。

---

例えば

```cpp
if (...)
{
    rewrite(...)
}
```

など。

---

## 3. あるべき姿

Validator は

```text
Accept
Reject
```

のみ。

---

## 4. 改修方法

Validator監査。

修正処理禁止。

---

# ⑯ AudioEngine が Runtime Orchestrator のまま残っている

## 1. 未達成内容

最大の構造問題候補です。

---

Practical Stable ISR Bridge Runtime の最終形では

```text
AudioEngine
   ↓
Facade
```

になるべきです。

---

しかし現状は

```cpp
applyRuntimeCommitFromIntent
processPendingCommit
publishSmoothTransitionState
startImmediateSmoothTransition
replaceFadingRuntimeDSPAndRetirePrevious
```

などが残っています。

---

## 2. 該当箇所

AudioEngine.cpp 全体。

---

## 3. あるべき姿

```text
AudioEngine
    ↓
RuntimeBuilder
    ↓
Coordinator
    ↓
Executor
```

---

## 4. 改修方法

段階的解体。

AudioEngineをFacade化。

---

# 優先度順

前回までの指摘を除いた追加監査候補としては、

| 優先 | 項目                                   |
| -- | ------------------------------------ |
| S  | ⑯ AudioEngine Runtime Orchestrator残存 |
| S  | ⑪ Semantic/Execution境界未完成            |
| A  | ⑨ Coordinator State Authority        |
| A  | ⑩ RuntimeWorld Immutable化            |
| A  | ⑬ Rebuild Admission完全RuntimeWorld化   |
| A  | ⑭ EpochDomain Semantic漏洩             |
| B  | ⑫ RuntimeGraph Semantic漏洩            |
| B  | ⑮ ValidatorのDecision化                |

特に **⑯・⑪・⑨** は、Practical Stable ISR Bridge Runtime の最終目標である「Authorityの単一化」と「Semantic/Execution分離」に直接関わるため、前回指摘分を除けば最も重点的に監査すべき候補です。
