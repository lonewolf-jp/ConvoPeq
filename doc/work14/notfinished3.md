最新版 ConvoPeq.md 全体を再監査しない限り「漏れなく提示」はできません。今回のファイル検索で確認できた範囲から判断すると、Practical Stable ISR Bridge Runtime の到達条件に対して、依然として未達または未証明な箇所は少なくとも以下があります。

重要なのは、

> 「実装済み」と「Verifierで保証されている」は別

という点です。

Practical Stable ISR Runtime の最終到達条件は

* Observe単一
* Authority単一
* Publication単一
* Generation単一
* Retire単一

です。

現在のコードは Coordinator / RuntimeStore / RuntimeWorld が実装されていますが、以下はまだ未達または未証明です。

---

# 1. AudioEngine が依然として Publication Authority の一部を保持

## 1-1. 未達内容

本来の完成形は

```text
RuntimeBuilder
    ↓
RuntimeWorld
    ↓
Coordinator.publish(world)
```

です。

しかし現状は

```cpp
engine_->runPublicationPrecheckNonRt(...)
engine_->onRuntimePublishedNonRt(...)
engine_->onRuntimeRetiredNonRt(...)
engine_->enqueueDeferredDeleteNonRt(...)
```

が RuntimePublicationBridge 内から呼ばれています。 

つまり Publication の最終 Authority が Coordinator に完全移譲されていません。

---

## 1-2. 該当箇所

RuntimePublicationBridge

```cpp
engine_->runPublicationPrecheckNonRt(...)
engine_->onRuntimePublishedNonRt(...)
engine_->onRuntimeRetiredNonRt(...)
engine_->enqueueDeferredDeleteNonRt(...)
```



---

## 1-3. あるべき姿

```text
RuntimeBuilder
    ↓
RuntimeWorld
    ↓
Coordinator
    ↓
Store
```

AudioEngine は Consumer のみ。

```text
AudioEngine
      ↓
observe()
```

のみ許可。

---

## 1-4. 改修方法

Bridge 内の

```cpp
runPublicationPrecheckNonRt
onRuntimePublishedNonRt
onRuntimeRetiredNonRt
```

を Coordinator Policy オブジェクトへ移管。

AudioEngine 側から Publication Lifecycle を剥離。

---

# 2. RuntimeWorld Self-contained 化が未証明

## 2-1. 未達内容

Phase5 完了条件

```text
external semantic dependency 0
```

です。

しかし RuntimePublicationBridge は

```cpp
AudioEngine* engine_
```

を保持しています。 

これは RuntimeWorld の意味解釈が AudioEngine 外部状態へ依存している可能性を示します。

---

## 2-2. 該当箇所

```cpp
AudioEngine* engine_;
```



---

## 2-3. あるべき姿

```text
RuntimeWorld
 ├ topology
 ├ execution
 ├ overlap
 ├ retire
 └ publication
```

だけで意味が完結。

外部 mutable state を参照しない。

---

## 2-4. 改修方法

RuntimeWorld に必要情報を全格納。

AudioEngine 依存の意味判定を除去。

---

# 3. Observe Source 単一化が未証明

## 3-1. 未達内容

計画上の最終形は

```cpp
const RuntimeWorld* world =
    runtimeCoordinator.consume();
```

のみ。 

しかし計画書自体が

```text
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
TransitionState.active
RuntimeGraph.activeNode
```

を重複源として列挙しています。 

最新版コードでこれらが完全排除された証拠は確認できていません。

---

## 3-2. 該当箇所

監査対象：

```text
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
TransitionState
activeRuntimeDSP
fadingRuntimeDSP
```



---

## 3-3. あるべき姿

Audio Thread

```cpp
observeWorldHandle(...)
```

のみ。

---

## 3-4. 改修方法

grep対象

```text
activeRuntimeDSP
fadingRuntimeDSP
TransitionState
preparedCrossfade
```

Decision Source 使用を全排除。

Projection 限定へ格下げ。

---

# 4. Crossfade Authority Collapse が未証明

## 4-1. 未達内容

Phase6 完了条件

```text
transition branching authority 0
crossfade semantic leakage 0
```

です。

しかし計画書が問題源として

```text
preparedCrossfade
fade atomics
pending flags
executor fade state
```

を明示しています。 

最新版コードからこれらが完全に Projection 化された証拠は確認できません。

---

## 4-2. 該当箇所

監査対象

```text
preparedCrossfade
fadeProgress
fadeSamples
pendingFade
executorFadeState
```



---

## 4-3. あるべき姿

```text
RuntimeWorld.overlap
```

のみが Authority。

Executor の Crossfade 状態は

```text
ExecutorLocal
```

分類のみ。

---

## 4-4. 改修方法

Crossfade の分岐判断を RuntimeWorld.overlap へ統合。

Executor 状態は Diagnostic 化。

---

# 5. Legacy Runtime Semantic Removal 未達

## 5-1. 未達内容

Phase8 完了条件

```text
legacy runtime semantic 0
```

です。

しかしコード上には

```cpp
publishState(current,next,...)
```

形式が残っています。 

これは

```text
active runtime
next runtime
```

という Legacy Transition Semantic がまだ存在している可能性を示します。

---

## 5-2. 該当箇所

```cpp
publishState(current,
             next,
             policy,
             fadeTimeSec,
             active);
```



---

## 5-3. あるべき姿

```cpp
publish(RuntimeWorld&)
```

のみ。 

---

## 5-4. 改修方法

Publication API を

```cpp
publish(world)
```

へ統一。

current/next モデル廃止。

---

# 6. Runtime Meaning Source Collapse が未完

これが最重要です。

計画書の最終定義は

```text
Single Authoritative Observable Runtime
```

です。 

現状は

```text
RuntimeWorld
RuntimeGraph
activeRuntimeDSP
fadingRuntimeDSP
TransitionState
CrossfadeState
AudioEngine
```

のうち、

どれが最終 Authority でどれが Projection かがコードレベルで完全収束したとまでは確認できません。

---

# 優先順位

実運用で破綻しにくい Runtime という観点で優先順位を付けると、

| 優先度      | 項目                                      |
| -------- | --------------------------------------- |
| Critical | Observe Source 完全単一化                    |
| Critical | Crossfade/Transition Authority 排除       |
| High     | AudioEngine から Publication Authority 剥離 |
| High     | RuntimeWorld Self-contained 化           |
| Medium   | Legacy publishState(current,next) 廃止    |
| Medium   | RuntimeGraph の Projection 化徹底           |

特に、

```text
activeRuntimeDSP
fadingRuntimeDSP
TransitionState
preparedCrossfade
```

が AudioThread の分岐判断にまだ使われている場合、

Practical Stable ISR Bridge Runtime は未達です。

ここが現時点で最も重点的に監査すべき箇所です。



前回までに挙げた

* Observe Source 単一化未証明
* Crossfade Authority Collapse 未証明
* Publication Authority が AudioEngine に残存
* RuntimeWorld Self-contained 化未達
* Legacy publishState(current,next) 残存
* Runtime Meaning Source Collapse 未完

以外で、ConvoPeq.md から読み取れる未達成点を挙げます。

なお、以下は「実装されていない」ではなく、

> Practical Stable ISR Bridge Runtime の最終到達条件に対して未達または未証明

という観点です。

---

# 7. Publication API Zero-Call 未達

## 1. 未達内容

基本計画書では Phase2 の DoD が

```text
legacy publication API zero-call
```

です。 

しかし現実の Coordinator API は依然として

```cpp
publishState(
    current,
    next,
    policy,
    fadeTimeSec,
    active)
```

です。

これは RuntimeWorld publish ではなく、

Transition publish API です。

つまり Publication API の意味論がまだ RuntimeWorld 中心へ完全移行していません。

---

## 2. 該当箇所

```cpp
publishRuntimeStateNonRt(...)
{
    makeRuntimePublicationCoordinator().publishState(
        current,
        next,
        policy,
        fadeTimeSec,
        active);
}
```



および

```cpp
RuntimePublicationCoordinator::publishState(...)
```



---

## 3. あるべき姿

```cpp
publish(RuntimeWorld&& world)
```

のみ。

Transition 概念は Build 時に解決済み。

---

## 4. 改修方法

Coordinator の公開 API を

```cpp
publishWorld(...)
```

へ変更。

current/next/policy/fadeTime を API から排除。

---

# 8. RuntimeGeneration 単一性が未証明

## 1. 未達内容

計画書では

```text
唯一 authoritative な generation
```

が要求されています。 

しかし RuntimeWorld には

```cpp
generation
runtimeVersion
transitionId
generationSemantic
```

が共存しています。 

分類は付いていますが、

実際に runtimeVersion が branch に使われていないことは未証明です。

---

## 2. 該当箇所

```cpp
generation
runtimeVersion
transitionId
generationSemantic
```



---

## 3. あるべき姿

実行分岐可能な識別子は

```cpp
RuntimeGeneration
```

のみ。

---

## 4. 改修方法

Verifier追加。

grep監査対象

```text
runtimeVersion
transitionId
```

---

# 9. RuntimeSemanticSchema 完全一致が未証明

## 1. 未達内容

計画では

```text
schema と authority inventory が一致
```

が必要です。

しかし RuntimeWorld には

```cpp
EngineRuntime engine
RuntimeGraph graph
```

が残っています。 

これらが

```cpp
RuntimeSemanticSchema
```

に含まれていません。

---

## 2. 該当箇所

```cpp
EngineRuntime engine;
RuntimeGraph graph;
```



---

## 3. あるべき姿

どちらか。

### A

完全 Projection

```cpp
Diagnostic
Derived
```

限定

### B

Schemaへ編入

---

## 4. 改修方法

Authority Inventory Verifier 追加。

Schema外 branch source 検出。

---

# 10. Partial Publication Reject が未証明

## 1. 未達内容

計画書では

```text
Partial publication = 0
schema completeness reject
```

です。

しかし Coordinator publish 時に

```cpp
schema completeness
```

検査が見当たりません。

---

## 2. 該当箇所

Coordinator

```cpp
publishState(...)
```



---

## 3. あるべき姿

publish 前に

```cpp
RuntimeSemanticSchemaValidator
```

が

```cpp
complete
```

を保証。

---

## 4. 改修方法

Publication Reject Rule 追加。

例

```cpp
if (!schema.isComplete())
    reject;
```

---

# 11. PublicationEpoch ↔ RuntimeGeneration Contract 未証明

## 1. 未達内容

計画書は

```text
PublicationEpoch
RuntimeGeneration
```

の単調対応を要求しています。 

しかし確認できるのは

```cpp
PublicationSequenceId
generation
```

程度です。

Epoch 対応保証が確認できません。

---

## 2. 該当箇所

計画要求

```cpp
PublicationEpoch
```



実装確認できるもの

```cpp
publication.sequenceId
generation
```

---

## 3. あるべき姿

```cpp
epoch <= generation
```

または

```cpp
epoch -> generation
```

単調写像。

---

## 4. 改修方法

専用 verifier。

```cpp
validateEpochMapping()
```

追加。

---

# 12. Snapshot Non-Authority の完全証明不足

## 1. 未達内容

計画書は

```text
snapshot authority usage 0
```

を要求しています。

しかし RuntimeBuilder は

```cpp
sealedSnapshot
```

を受け取ります。 

これが projection のみなのか authority なのか保証が見えません。

---

## 2. 該当箇所

```cpp
buildRuntimePublishWorld(
 ...
 sealedSnapshot)
```



---

## 3. あるべき姿

Snapshot は

```cpp
diagnostic
cache
telemetry
```

のみ。

---

## 4. 改修方法

Verifier

```cpp
SnapshotAuthorityVerifier
```

追加。

---

# 13. Topology Authority Leakage 未証明

## 1. 未達内容

計画では

```text
topology authority は RuntimeWorld.topology のみ
```

です。

しかし RuntimeWorld には

```cpp
RuntimeGraph graph
```

が残っています。 

graph が execution order 判断に使われれば契約違反です。

---

## 2. 該当箇所

```cpp
RuntimeGraph graph;
```



---

## 3. あるべき姿

```cpp
RuntimeWorld.topology
```

のみが authority。

graph は projection。

---

## 4. 改修方法

禁止 verifier

```cpp
graph-based branching
```

追加。

---

# 14. RuntimeFieldDescriptor 完全被覆が未達

## 1. 未達内容

計画では

```text
全 runtime field が分類済み
```

が要求されています。

しかし RuntimeWorld の実フィールド数に対し

```cpp
kFieldDescriptors = 17
```

です。 

実際には

* metadata
* semanticHash
* projectionFreshness
* runtimeVersion
* transitionId
* graph
* engine

など多数存在します。

Descriptor が全被覆している保証がありません。

---

## 2. 該当箇所

```cpp
kFieldDescriptors
```



---

## 3. あるべき姿

```text
field count
descriptor count
```

一致。

---

## 4. 改修方法

静的検証追加。

```cpp
static_assert(...)
```

による inventory 完全一致検査。

---

# 実運用上の優先度

前回提示分を除いた追加事項だけで優先度を付けると、

| 優先度      | 項目                                      |
| -------- | --------------------------------------- |
| Critical | RuntimeGeneration単一性未証明                 |
| Critical | Topology Authority Leakage未証明           |
| High     | Publication API Zero-Call未達             |
| High     | Partial Publication Reject未証明           |
| High     | Snapshot Non-Authority未証明               |
| High     | Schema/Authority Inventory 不一致          |
| Medium   | PublicationEpoch ↔ RuntimeGeneration未証明 |
| Medium   | RuntimeFieldDescriptor 完全被覆未証明          |

この中で特に危険なのは、

**Topology Authority Leakage** と **RuntimeGeneration単一性** です。

これらは一見動作していても、長時間運用時に semantic drift を起こす典型的な残存経路です。現行コードは Schema と Verifier がかなり整備されていますが、「Verifierで禁止されていること」と「実際にコード上で不可能になっていること」の間にまだ監査余地が残っています。



ここでは、前回までに挙げた

* Observe Source 単一化
* Crossfade Authority
* Publication Authority
* RuntimeWorld Self-contained
* Legacy publishState
* Runtime Meaning Source Collapse
* Generation単一性
* Topology Authority Leakage
* Schema/Authority Inventory
* Partial Publication Reject
* Snapshot Non-Authority
* PublicationEpoch対応
* RuntimeFieldDescriptor完全被覆

以外の観点を挙げます。

最新版 ConvoPeq.md を見る限り、Practical Stable ISR Bridge Runtime の「実運用で破綻しにくい Runtime」という観点では、まだ以下の未達・未証明点があります。

---

# 15. RuntimeSemanticSchema と Plan Schema の乖離

## 1. 未達内容

v2.3計画では RuntimeSemanticSchema の authority 領域は

```cpp
GenerationSemantic
TopologySemantic
RoutingSemantic
ExecutionSemantic
PublicationSemantic
OverlapSemantic
RetireSemantic
```

を中核として定義されています。

しかし実装は

```cpp
TimingSemantic
LatencySemantic
SchedulingSemantic
ResourceSemantic
AffinitySemantic
AutomationSemantic
CoefficientSemantic
```

まで拡張されています。

問題は、

「これらが Authority なのか Projection なのか」

が計画書レベルで定義されていないことです。

---

## 2. 該当箇所

```cpp
struct RuntimeSemanticSchema
{
    ...
    TimingSemantic timing;
    LatencySemantic latency;
    SchedulingSemantic scheduling;
    ResourceSemantic resource;
    AffinitySemantic affinity;
    AutomationSemantic automation;
    CoefficientSemantic coefficient;
};
```

---

## 3. あるべき姿

各 Semantic が

```text
Authority
Derived
Diagnostic
ExecutorLocal
```

のどれかに固定分類されること。

---

## 4. 改修方法

SemanticCategory Inventory を作る。

全 Semantic に

```cpp
RuntimeAuthorityClass
```

を付与し verifier で監査する。

---

# 16. SchedulingSemantic と ExecutionSemantic の二重表現

## 1. 未達内容

ExecutionSemantic に

```cpp
transitionActive
crossfadeStartDelayBlocks
crossfadeDryHoldSamples
```

があります。

しかし SchedulingSemantic にも

```cpp
transitionActive
crossfadeStartDelayBlocks
crossfadeDryHoldSamples
```

があります。

これは意味的重複です。

Practical Stable ISR Runtime が最も嫌う

```text
semantic duplication
```

です。

---

## 2. 該当箇所

ExecutionSemantic

```cpp
transitionActive
crossfadeStartDelayBlocks
crossfadeDryHoldSamples
```

SchedulingSemantic

```cpp
transitionActive
crossfadeStartDelayBlocks
crossfadeDryHoldSamples
```

---

## 3. あるべき姿

Authority は1箇所。

例えば

```cpp
ExecutionSemantic
```

のみ。

SchedulingSemantic は削除または Derived。

---

## 4. 改修方法

SchedulingSemantic を Projection 化。

Verifier を追加。

```text
duplicate semantic source detector
```

---

# 17. ActivationEpoch の重複保持

## 1. 未達内容

ActivationEpoch が

```cpp
GenerationSemantic.activationEpoch
TimingSemantic.activationEpoch
```

に重複しています。

これは Generation Authority の重複です。

---

## 2. 該当箇所

GenerationSemantic

```cpp
activationEpoch
```



TimingSemantic

```cpp
activationEpoch
```



---

## 3. あるべき姿

ActivationEpoch の Authority は1箇所。

---

## 4. 改修方法

TimingSemantic の値を

```cpp
Derived
```

へ格下げ。

---

# 18. Semantic Hash Coverage 不完全

## 1. 未達内容

RuntimeSemanticHash が保持しているのは

```cpp
generation
topology
execution
routing
payload
publication
overlap
retire
```

です。

しかし Schema には

```cpp
timing
latency
scheduling
resource
affinity
automation
coefficient
```

があります。 

Hash対象外です。

つまり

```text
semantic drift
```

検知対象外の領域があります。

---

## 2. 該当箇所

```cpp
RuntimeSemanticHash
```

---

## 3. あるべき姿

Authority Semantic は全て Hash 対象。

---

## 4. 改修方法

Authority Semantic Inventory と Hash Coverage を一致させる。

---

# 19. Semantic Equivalence 判定が Schema 全体を見ていない

## 1. 未達内容

Semantic Equivalence は

```cpp
generation
topology
execution
routing
payload
publication
overlap
retire
```

のみ比較しています。 

しかし RuntimeSemanticSchema はもっと大きい。 

結果として

```text
schema drift
```

を見逃せます。

---

## 2. 該当箇所

```cpp
classifySemanticEquivalence(...)
```



---

## 3. あるべき姿

Authority Semantic 全比較。

---

## 4. 改修方法

Schema Inventory から自動生成する比較器へ変更。

---

# 20. Contract Registry 実装が未確認

## 1. 未達内容

計画書は

```text
ContractRegistry
VerifierRegistry
```

を唯一台帳と定義しています。

しかし ConvoPeq.md 内で確認できるのは

```cpp
kRequiredVerifierTable
```

です。 

ContractRegistry の実体が確認できません。

---

## 2. 該当箇所

```cpp
kRequiredVerifierTable
```



計画要求

```text
ContractRegistry
VerifierRegistry
```

---

## 3. あるべき姿

```cpp
ContractRegistry
VerifierRegistry
```

の二重台帳。

---

## 4. 改修方法

契約と verifier の依存関係を明示的に登録。

CI で整合検査。

---

# 21. Fail-Closed Publication が Engine Hook に依存

## 1. 未達内容

Coordinator は Publish 前に

```cpp
validatePublicationNonRt()
```

を呼びます。

しかし実体は

```cpp
engine_->runPublicationPrecheckNonRt()
```

です。 

Fail-Closed が Runtime 契約ではなく AudioEngine 実装へ依存しています。

---

## 2. 該当箇所

```cpp
validatePublicationNonRt
```

↓

```cpp
engine_->runPublicationPrecheckNonRt
```

---

## 3. あるべき姿

Coordinator 自身が契約検証を所有。

---

## 4. 改修方法

Publication Validator を独立コンポーネント化。

---

# 22. Runtime Semantic Lifecycle Verifier が未確認

## 1. 未達内容

状態遷移テストはあります。

```text
Draft
Publishing
Published
Retiring
Retired
Destroyed
```



しかしこれはユニットテストです。

実 Runtime がこの状態機械を強制している保証は見えません。

---

## 2. 該当箇所

```cpp
testRuntimeSemanticTransitionGraphValidation()
```



---

## 3. あるべき姿

Runtime 実体が

```cpp
LifecycleState
```

を保持。

Verifier が監査。

---

## 4. 改修方法

PublicationCoordinator に Lifecycle Tracking を追加。

---

# 実運用上の優先順位

今回追加分だけで優先順位を付けると、

| 優先度      | 項目                                          |
| -------- | ------------------------------------------- |
| Critical | ExecutionSemantic / SchedulingSemantic 二重表現 |
| Critical | ActivationEpoch 二重保持                        |
| High     | Semantic Hash Coverage 不完全                  |
| High     | Semantic Equivalence が Schema 全体を見ていない      |
| High     | Fail-Closed Publication が Engine Hook 依存    |
| Medium   | RuntimeSemanticSchema と Plan Schema の乖離     |
| Medium   | ContractRegistry 実装未確認                      |
| Medium   | Lifecycle Verifier 未確認                      |

この中で特に危険なのは **ExecutionSemantic と SchedulingSemantic の重複** および **ActivationEpoch の重複** です。これらは「動作は正しいが意味が分岐する」という Practical Stable ISR Runtime が最も警戒する種類の semantic drift 発生源になり得ます。
