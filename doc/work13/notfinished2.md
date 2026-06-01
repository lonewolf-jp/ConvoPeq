最新版 ConvoPeq.md を精査すると、Practical Stable ISR Bridge Runtime の未達項目はかなり絞られています。しかし残っているものは「最後に難しい部分」です。

以下、**計画書の到達条件に対して未達なものだけ**を整理します。

---

# 1. RuntimeWorld Self-contained 化未達（最重要）

v2.3 の最終目標は

```text
RuntimeWorld
=
唯一の authoritative semantic container
```

です。 

しかし現状は AudioEngine 側の atomic 群が依然として authority を持っています。

---

## 現在残っている authority 候補

### bypass

```cpp
eqBypassRequested
convBypassRequested
```

---

### processing order

```cpp
currentProcessingOrder
```

---

### gain semantic

```cpp
inputHeadroomDb
outputMakeupDb
convolverInputTrimDb
```

---

例えば

```cpp
applyDefaultsForCurrentMode()
```

は

```cpp
eqBypassRequested
convBypassRequested
currentProcessingOrder
```

を読んで

```cpp
inputHeadroomDb
outputMakeupDb
convolverInputTrimDb
```

を決定しています。 

つまり現在も

```text
AudioEngine atomic
↓
runtime semantic生成
```

が存在します。

これは

```text
RuntimeWorld
↓
projection
```

になっていません。

---

## 本来の完成形

本来は

```cpp
RuntimeWorld.graph.eqBypassed
RuntimeWorld.graph.convBypassed
RuntimeWorld.graph.processingOrder
```

から派生させるべきです。

---

# 2. Transition Semantic Leakage

計画では

```text
TransitionState
=
authority禁止
```

です。

しかし現状は AudioThread が transition を直接見ています。

---

## AudioThread

```cpp
runtimePublishView.transition.current
runtimePublishView.transition.next
```

で DSP を決定しています。 

---

## Runtime publish

```cpp
runtime.transition.current = current;
runtime.transition.next = next;
runtime.transition.active = active;
```

---

## Diagnostic だけではない

これは

```text
TransitionState
↓
DSP選択
↓
音が変わる
```

ため、

単なる diagnostic ではありません。

実質的に

```text
execution branching authority
```

です。

---

## v2.3 的には

本来

```cpp
RuntimeWorld.topology.hasFadingRuntime
```

のみが authority で、

transition は

```text
executor local
```

であるべきです。

---

# 3. Legacy Build Generation の残存

RuntimeBuildSnapshot に generation が残っています。

```cpp
struct RuntimeBuildSnapshot
{
    int generation = 0;
}
```

さらに

```cpp
snapshot.generation = generation;
```

も存在します。 

---

## 問題

RuntimeGraph 側にも generation がある。

```cpp
std::uint64_t generation = 0;
```



つまり

```text
Build generation
Runtime generation
```

の二重系統が存在します。

---

## 現時点の判定

危険度は中。

なぜなら検索範囲では

```cpp
snapshot.generation
```

で分岐している箇所はまだ確認できていません。

しかし

```text
authoritative generation 1系統
```

という契約にはまだ到達していません。 

---

# 4. Shadow Compare が契約レベル止まり

これは重要です。

---

## 現在あるもの

```text
ShadowCompareContractTests
```

---

## テスト内容

実際には

```cpp
classifySemanticEquivalence(...)
```

の単体テストです。 

---

## 無いもの

計画で要求している

```text
legacy runtime
↓
new runtime
↓
compare
↓
rollback
```

が確認できません。

---

つまり

```text
Shadow Compare Contract
```

はある。

しかし

```text
Shadow Runtime Operation
```

が見えない。

---

# 5. Crossfade Semantic の完全分離未達

Crossfade 状態が Runtime に混在しています。

---

現在

```cpp
runtime.latencyDelayOld
runtime.latencyDelayNew

runtime.dspCrossfadePending

runtime.dspCrossfadeUseDryAsOld

runtime.firstIrDryCrossfadePending

runtime.dspCrossfadeStartDelayBlocks

runtime.dspCrossfadeDryHoldSamples
```

が Runtime 構築時に注入されています。

---

しかし規約では

```text
Crossfade
=
executor local
```

です。 

---

つまり現在は

```text
Runtime Semantic
+
Execution Local State
```

がまだ完全分離されていません。

---

# 6. RuntimeWorld Ownership 完了未確認

v2.3 では

```text
world ownership verifier
```

が必須です。 

---

現在確認できるのは

```cpp
RuntimePublicationCoordinator
```

による publish / retire。 

---

しかし

```text
ownership model
ABA risk
epoch progression
reclaim latency
```

の実装確認まではできません。

---

したがって

```text
Phase5 Self-contained
```

は未完判定になります。

---

# 7. Legacy Mutable Runtime Semantic Removal 未完

計画の Phase8。 

---

現在も多数の mutable state が残っています。

例：

```cpp
eqBypassRequested
convBypassRequested

currentProcessingOrder

inputHeadroomDb
outputMakeupDb
convolverInputTrimDb

manualOversamplingFactor
oversamplingType
noiseShaperType
```

---

これらが

```text
Authoritative
Derived
Diagnostic
LegacyTemporary
```

へ完全分類された形跡は確認できません。

---

# 優先度順の残課題

## Critical

### A. RuntimeWorld Self-contained化

残存 authority

```text
eqBypassRequested
convBypassRequested
currentProcessingOrder
gain parameters
```

---

### B. Transition Authority Collapse

残存 authority

```text
transition.current
transition.next
transition.active
```

---

## High

### C. Crossfade executor-local化

残存 authority候補

```text
latencyDelayOld
latencyDelayNew
dspCrossfade*
```

---

### D. authoritative generation 完全単一化

残存

```text
RuntimeBuildSnapshot.generation
RuntimeGraph.generation
```

---

## Medium

### E. 実運用 Shadow Compare

現状

```text
contract test
```

のみ。

不足

```text
shadow runtime
mismatch telemetry
rollback trigger
```

---

# 現時点の到達率再評価

最新版ソースベースでは、

| フェーズ   | 状態  |
| ------ | --- |
| Phase0 | 95% |
| Phase1 | 90% |
| Phase2 | 95% |
| Phase3 | 90% |
| Phase4 | 90% |
| Phase5 | 70% |
| Phase6 | 60% |
| Phase7 | 85% |
| Phase8 | 60% |
| Phase9 | 30% |

総合すると、

**Practical Stable ISR Bridge Runtime 完成度は約 82〜87%** と評価できます。

残っているのは主に「RuntimeWorld外に残る authority の除去」と「Transition/Crossfade の executor-local 化」です。これらが解消されると、実質的に v2.3 の完成条件にかなり近づきます。



前回挙げた未達事項以外にも、最新版 ConvoPeq.md を見ると、Practical Stable ISR Bridge Runtime v2.x の DoD（Definition of Done）に照らして、さらに数個の未達候補があります。

ただし、ここから先は「未達が確定しているもの」と「未達の疑いが強いもの」を分けて扱う必要があります。

---

# 1. Authority Duplication Verifier が実質未達

v2.xでは

```text
authority duplication verifier
```

が必須です。 

しかし現在の RuntimeGraph 構築を見ると、

```cpp
graph.eqBypassed
graph.convBypassed
graph.inputHeadroomGain
graph.outputMakeupGain
graph.convolverInputTrimGain
```

へ値を投入する際、

```cpp
consumeAtomic(eqBypassRequested)
consumeAtomic(convBypassRequested)
consumeAtomic(inputHeadroomGain)
consumeAtomic(outputMakeupGain)
consumeAtomic(convolverInputTrimGain)
```

を直接読んでいます。 

つまり

```text
RuntimeWorld
    ↑
AudioEngine atomic
```

という構造がまだ残っています。

問題は、

これを検出する verifier が見当たらないことです。

---

## 本来必要

例えば

```text
RuntimeGraph.authority
=
唯一ソース
```

を検証する

```text
AuthorityDuplicationVerifier
```

が必要です。

現状は

```text
実装依存
```

になっています。

---

# 2. Publication Monotonicity Verifier 未確認

計画書では

```text
publication monotonicity verifier
```

が必須です。 

---

現状確認できるもの

```cpp
commitRuntimePublication(...)
```

では

```cpp
world.generation
world.publication.sequenceId
world.publication.epoch
```

を commit に渡しています。 

---

しかし

```text
sequenceId strictly monotonic
epoch rollback forbidden
generation regression forbidden
```

を保証する verifier が確認できません。

---

これは

```text
実装されている
≠
検証されている
```

です。

---

# 3. Publication Sequence Verifier 未確認

同様です。

計画では

```text
publication sequence verifier
```

必須。 

---

現状は

```cpp
world.publication.sequenceId
```

が存在します。 

しかし

```text
欠番
重複
巻き戻り
```

検出コードが確認できません。

---

つまり

```text
single source
```

には近づいたが

```text
sequence contract
```

までは到達確認できません。

---

# 4. World Ownership Verifier 実質未達

これはかなり重要です。

計画で必須。 

---

現在確認できるもの

```cpp
publish()
retire()
observeWorldHandle()
```

---

しかし

```text
owner count
reader count
epoch lifetime
ABA protection
```

を検査する verifier が見当たりません。

---

Practical Stable ISR では

```text
world ownership verifier pass
```

が DoD です。

ここは未達寄りです。

---

# 5. Runtime Topology Authority Verifier 未確認

計画では

```text
runtime topology authority verifier
```

必須。 

---

現在でも

```cpp
transition.current
transition.next
```

から topology が読めます。

---

つまり

```text
topology authority
RuntimeWorld
```

が保証されていません。

---

検証コードも確認できません。

---

# 6. Diagnostic-Only Boundary Verifier 未確認

規約では

```text
diagnostic
telemetry
visualization
```

は authority を持ってはならない。 

---

しかし現在

```cpp
logRuntimeTransitionEvent(...)
```

内部で

```cpp
transition.current
transition.next
transition.active
```

を参照しています。

---

ログ用途そのものは問題ありません。

問題は

```text
diagnostic-only verifier
```

が見当たらないことです。

---

つまり

将来誰かが

```cpp
if (diagState.xxx)
```

を入れても防げない。

---

# 7. Semantic Hash Equivalence Verifier 未達

これは見落とされやすいです。

---

現状

```cpp
RuntimeBuildFingerprint
```

があり

```cpp
dspParameterHash
```

も作られています。

---

しかし計画は

```text
semantic hash equivalence verifier
```

を要求しています。 

---

必要なのは

```text
hash一致
↓
semantic一致確認
```

ではなく

```text
hash不一致
↓
semantic不一致
```

を検査する運用です。

---

現在は

```text
fingerprint生成
```

までしか確認できません。

---

# 8. Rollback Governance 系ほぼ未実装

これは前回触れていませんでした。

規約では

```text
cooldown
escalation
repeated failure threshold
```

必須。 

---

さらに

```text
Soft
Medium
Hard
Emergency
```

の rollback level。 

---

しかし今回のソース断片からは

```text
rollback manager
rollback state machine
rollback hysteresis
```

が確認できません。

---

Shadow Compare が未完成である以上、

Rollback Governance も未達の可能性が高いです。

---

# 9. Soak Validation Infrastructure 未確認

Phase9です。 

---

計画要求

```text
IR reload storm
automation burst
sample-rate churn
suspend/resume churn
UI attach/detach
```



---

しかし確認できるのは主に unit test 群です。

長時間 Soak 用 infrastructure は確認できません。

---

# 10. Legacy Manifest Expiry System 未確認

Phase8の重要項目です。 

---

計画では

```text
temporary residue
```

を

```text
manifest管理
期限付き削除
```

する必要があります。

---

現在残っている

```cpp
eqBypassRequested
convBypassRequested
currentProcessingOrder
manualOversamplingFactor
```

などが

```text
LegacyTemporary
```

なのか

```text
Permanent
```

なのかを管理する manifest が確認できません。

---

# 追加で残っている未達項目の優先度

### Critical

* Authority Duplication Verifier 不在
* Runtime Topology Authority Verifier 不在
* World Ownership Verifier 不在
* Publication Monotonicity Verifier 不在
* Publication Sequence Verifier 不在

### High

* Semantic Hash Equivalence Verifier 不在
* Diagnostic-Only Boundary Verifier 不在
* Rollback Governance 不在

### Medium

* Soak Validation Infrastructure 不在
* Legacy Manifest Expiry System 不在

---

したがって、前回の

* RuntimeWorld Self-contained化
* Transition Authority Collapse
* Crossfade Executor Local化
* Generation単一化
* Shadow Runtime運用化

に加えて、

**「実装本体はかなり進んでいるが、Practical Stable ISR が要求する verifier／governance／operational hardening 層がまだ不足している」**

というのが最新版 ConvoPeq のもう一つの大きな未達領域です。特に verifier 群（authority duplication、publication monotonicity、world ownership）は、実装の正しさを継続的に保証するための中核であり、現状では到達条件を満たしたとはまだ言いにくい状態です。


前回まで挙げた項目以外で、さらにソースから読み取れる未達候補を精査すると、いくつかあります。

ただし今回は「Practical Stable ISR Bridge Runtime の契約文書（v2.2/v2.3）と現行コードを直接比較した場合」に限定して挙げます。

---

# 1. RuntimeStore が「RuntimeCoordinator consume()」完成形に到達していない

v3.1 完成像では Audio Thread は

```cpp
const RuntimeWorld* world = runtimeCoordinator.consume();
executor.process(world);
```

だけになることが要求されています。 

しかし現状の RuntimePublicationCoordinator は

```cpp
observePublishedWorld()
observeWorldHandle()
```

を静的ヘルパとして公開しています。 

つまり観測主体が

```text
RuntimeCoordinator
```

ではなく

```text
RuntimeStore
↓
observeWorldHandle()
```

です。

これは設計上の差ですが、

Practical Stable ISR Bridge Runtime の最終像である

```text
runtimeCoordinator.consume()
```

への収束は未完です。

---

# 2. RuntimeTopologyAuthoritySplit の完全解消未確認

v2.2/v2.3では危険構造として

```text
Runtime Topology Authority Split
```

が明示されています。

現状でも診断コード側で

```cpp
publishedWorld->topology.hasFadingRuntime
publishedWorld->engine.transition
```

を組み合わせて意味解釈しています。 

つまり

```text
topology.hasFadingRuntime
+
transition.next
```

の二系統が残っています。

もし topology が authority なら transition は不要、

transition が authority なら topology は不要です。

両方残っている時点で、

Topology Authority Split が完全に解消されたとは言い切れません。

---

# 3. Runtime Activity 重複の解消未確認

v3.1 では問題として

```text
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
TransitionState.active
RuntimeGraph.activeNode
```

の重複が挙げられています。 

しかし今回確認できるコードでも

```cpp
getActiveRuntimeDSP()
transition.current
transition.next
hasFadingRuntime
```

が共存しています。 

これらが内部的に統合済みかは確認できません。

したがって

```text
Runtime activity meaning source collapse
```

は未達疑いがあります。 

---

# 4. RuntimeSemanticSchema 完全分類の証跡不足

v2.2/v2.3では

全 runtime field を

```text
Authoritative
Derived
Diagnostic
ExecutorLocal
LegacyTemporary
```

へ分類することが完了条件です。

---

確かに

```cpp
AuthorityExhaustivenessVerifier
```

登録はあります。 

しかし確認できるのは

```text
Verifier名の存在
```

だけです。

---

一方で残存している

```cpp
manualOversamplingFactor
oversamplingType
noiseShaperType
eqBypassRequested
convBypassRequested
```

などが、

本当に inventory 化されている証跡は確認できません。

---

つまり

```text
authority inventory complete
```

はまだ未確認です。

---

# 5. Publication Atomicity Completion の完全達成未確認

v3.1 の Phase5 DoD は

```text
publish(RuntimeWorld*) のみ
```

です。 

---

現状 publish 経路は

```cpp
publishState(...)
↓
buildRuntimePublishWorld(...)
↓
publishAndSwap(...)
```

にかなり収束しています。

---

しかし同時に

```cpp
clearPublishedRuntimeSnapshotsNonRt()
```

が存在し、

```cpp
publishAndSwap(nullptr)
```

も行っています。 

---

つまり publish contract が

```text
publish(world)
```

だけではなく

```text
publish(nullptr)
```

も持っています。

これは shutdown 用途として合理的ですが、

契約的には

```text
Publication Atomicity Completion
```

の例外経路になります。

---

# 6. Runtime Identity 重複の完全解消未達

v3.1 では

```text
generation
runtimeVersion
transitionId
```

重複を問題視しています。 

---

現状確認できるもの

```cpp
snapshot.generation
```



```cpp
getVersion()
```



```cpp
runtimeUuid
```



---

つまり identity 系が

```text
generation
version
uuid
```

へ分散しています。

それぞれ用途は異なる可能性がありますが、

Practical Stable ISR が要求する

```text
authoritative generation singularization
```

はまだ証明できません。

---

# 7. Semantic First 運用の Nightly/Release 層が未確認

v2.1〜v2.3では

```text
PR
Nightly
Release
```

を分離した運用が要求されています。

---

現状確認できるのは

多数の verifier 登録です。 

しかし

```text
PR gate
Nightly gate
Release gate
```

の実装・CI設定は ConvoPeq.md からは確認できません。

---

これはコードではなく運用面の未達です。

---

# 8. Fail-Closed Governance の完全達成未確認

v2.3 の最終条件には

```text
verifier fail => build fail
policy mismatch => build fail
expiry mismatch => build fail
```

があります。

---

現状確認できるのは

```cpp
throw std::runtime_error(...)
```

型の契約テストです。 

しかし

```text
CI fail-closed
```

まで繋がっているかは確認できません。

---

# 現時点で追加できる未達候補の総括

前回までの指摘以外で、比較的有力なのは次の 8 項目です。

| 優先度    | 項目                                           |
| ------ | -------------------------------------------- |
| High   | RuntimeTopologyAuthoritySplit 完全解消未確認        |
| High   | Runtime Activity Meaning Source Collapse 未確認 |
| High   | Runtime Identity Singularization 未完          |
| Medium | RuntimeCoordinator consume 完成形未達             |
| Medium | RuntimeSemanticSchema 完全分類未証明                |
| Medium | Publication Atomicity Completion 例外経路残存      |
| Medium | Nightly/Release Governance 未確認               |
| Medium | Fail-Closed Governance 完全実装未確認               |

この中で特に重要なのは、

* RuntimeTopologyAuthoritySplit
* Runtime Activity Meaning Source Collapse
* Runtime Identity Singularization

の3つです。

これらは単なる verifier 不足ではなく、**「意味状態（semantic meaning）の単一化」という Practical Stable ISR Bridge Runtime の本質目標そのものに関わる残課題**です。現時点のコードを見る限り、まだ完全収束したと断言できる状態には達していません。


前回までに挙げた未達項目を除外して、さらに最新版 ConvoPeq.md を見ると、まだ指摘可能なものがあります。ただし、ここから先は「実装そのものの欠落」というより、「Practical Stable ISR Bridge Runtime が要求する統治契約に対して未証明・未収束」という分類になります。

## 1. RuntimePublicationCoordinator がまだ真の Runtime Authority になっていない

これは見落とされやすいです。

現状の Coordinator は

* publish
* retire
* backlog
* pressure

は管理しています。

しかし Coordinator 自身の API を見ると

```cpp
const void* RuntimePublicationCoordinator::getCurrent() const noexcept {
    return nullptr;
}
```

となっています。

つまり Coordinator 自体は

```text
authority owner
```

ではなく、

```text
RuntimeStore のラッパ
```

です。

さらに observe は

```cpp
observePublishedWorld()
observeWorldHandle()
```

として Store に委譲されています。

Practical Stable ISR の完成形では

```text
Coordinator
  ↓
consume()
  ↓
RuntimeWorld
```

になるべきですが、

現状は

```text
Coordinator
  ↓
RuntimeStore
  ↓
observe()
```

です。

したがって

**Publication Coordinator Centralization はまだ未完です。**

---

# 2. RuntimeStore Ownership Model が未証明

Practical Stable ISR では

```text
ownership transfer
ABA safety
reclaim ordering
```

が重要です。

実際、Verifier Registry には

```text
OwnershipTransferContractVerifier
ABAHazardVerifier
```

が登録されています。 

しかし RuntimeStore 実装側で

```text
epoch reclamation
hazard pointer
generation guard
```

に相当する実体が今回確認できません。

つまり

```text
Verifier名は存在
Ownership Model の実体確認は未完
```

です。

Practical Stable ISR の観点では

**Retire 安全性の形式証明が不足している状態**

です。

---

# 3. Publication Epoch と Runtime Generation の二重タイムライン

commit 時に

```cpp
world.generation
world.publication.sequenceId
world.publication.epoch
world.publication.mappedRuntimeGeneration
```

を同時に渡しています。 

つまり現在の RuntimeWorld には

少なくとも

```text
generation
publication sequence
publication epoch
mapped runtime generation
```

の4系統があります。

Practical Stable ISR が要求するのは

```text
authoritative timeline singularity
```

です。

現状は

```text
timeline が複数存在
```

しているため、

それぞれが

```text
Authoritative
Derived
Diagnostic
```

のどれなのかが明確に固定されていません。

---

# 4. RuntimeWorld Identity がまだ複数系統

RuntimeGraph を見ると

```cpp
runtimeUuid
fadingRuntimeUuid
transitionCurrentRuntimeUuid
transitionNextRuntimeUuid
generation
```

が存在します。 

Practical Stable ISR の完成形では

```text
Runtime Identity
```

は一意です。

しかし現状は

```text
runtime identity
transition identity
generation identity
```

が混在しています。

これは前回の generation 問題よりさらに広い話で、

**Identity Singularization が未完**

と言えます。

---

# 5. TransitionState が依然として Observe 対象になっている

Verifier Registry には

```text
ObserveForbiddenTypeVerifier
```

があります。 

さらに禁止型として

```text
TransitionState*
```

が登録されています。 

ところが実コードでは

```cpp
publishedWorld->engine.transition
```

を取得しています。 

これは診断用途とはいえ、

Practical Stable ISR の

```text
TransitionState observe 禁止
```

契約とは少し緊張関係があります。

少なくとも

```text
TransitionState が RuntimeWorld 外部へ露出している
```

という事実は残っています。

---

# 6. Deterministic Build の完全達成未証明

コードには

```cpp
DeterministicBuildVerifier
```

登録があります。

また

```cpp
RuntimeBuildFingerprint
```

もあります。

しかし BuildInput を見ると

```cpp
sampleRate
blockSize
oversamplingFactor
oversamplingType
noiseShaperType
```

のみです。 

一方、

実際の Runtime 構築経路では

```cpp
uiConvolverProcessor.captureBuildSnapshot()
```

など外部状態も関与しています。

したがって

```text
same semantic input
→ same RuntimeWorld
```

が本当に証明されているかは未確認です。

---

# 7. RuntimeBuilder がまだ AudioEngine に強く依存

Builder は

```cpp
class RuntimeBuilder
{
    AudioEngine& engine;
}
```

です。

Practical Stable ISR の理想像では

```text
Semantic Input
↓
RuntimeBuilder
↓
RuntimeWorld
```

です。

しかし現状は

```text
AudioEngine
↓
RuntimeBuilder
↓
RuntimeWorld
```

です。

つまり

**Builder が純粋な Semantic Builder になりきっていません。**

---

# 8. Semantic Dependency Graph の実運用確認不足

Work11 で

```text
SemanticDependencyGraphVerifier
```

が追加されています。

しかし確認できるのは

```text
Verifier 登録
テスト登録
```

です。

実際に RuntimeWorld 構築時の依存関係 DAG が

```text
循環禁止
hidden dependency 禁止
```

として強制されている箇所までは確認できません。

---

# 追加で残る未達候補の中で重要度が高いもの

優先度順に並べると、

### High

* Publication Coordinator Centralization 未完
* RuntimeStore Ownership Model 未証明
* Timeline Singularization 未完
* Runtime Identity Singularization 未完

### Medium

* TransitionState Observe Boundary 未収束
* Deterministic Build 完全証明不足
* RuntimeBuilder Pure Semantic Builder 化未完
* Semantic Dependency Graph 実運用未証明

---

ここまで含めると、現在の ConvoPeq は

```text
Publication/Retire Infrastructure
→ かなり完成

Verifier Framework
→ かなり完成

Semantic Authority Singularization
→ あと一歩

Identity / Timeline / Ownership
→ まだ収束途中
```

という状態です。

特に最後に残る本質的な課題は、

**「RuntimeWorld が唯一の意味の源であることを、identity・timeline・ownership の3軸で完全に証明できていない」**

点に集約できます。これは Practical Stable ISR Bridge Runtime の最終段階で解消されるべき領域です。
