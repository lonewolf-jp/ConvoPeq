最新版の ConvoPeq.md を前提に確認したところ、以前に議論されていた「Practical Stable ISR Bridge Runtime」はかなり進んでいます。ただし、「完全移行完了」ではなく、「実運用投入可能な段階に近づいているが、まだ移行残骸が残っている状態」という評価になります。

結論から言うと、

* Runtime Publication
* Immutable Runtime Snapshot
* Runtime Graph
* Epoch Retire
* Closure Validation
* Semantic Validation
* Runtime Governance

は既に実装済みです。

一方で、

* mutable runtimeの完全排除
* Runtime Worldの単一権威化
* Runtime Graphと旧AudioEngine状態の二重管理解消
* ISRガバナンスの実効性検証

はまだ未完了です。

---

# 総合評価

現在の達成度を推定すると

| 項目                            |   達成度 |
| ----------------------------- | ----: |
| Immutable Runtime Publication |   95% |
| Epoch Retire Runtime          |   90% |
| Runtime Graph化                |   90% |
| Publication Coordinator化      |   95% |
| Closure Validation            |   90% |
| Semantic Governance           |   85% |
| Mutable Runtime除去             |   70% |
| Single Source of Truth化       |   75% |
| Practical Stable Runtime完成度   | 85%前後 |

です。

---

# 実装済みと判断できるもの

## 1. RuntimePublicationCoordinator導入

存在確認できる。

* RuntimePublicationCoordinator.h
* ISRRuntimePublicationCoordinator.cpp
* ISRRuntimePublicationCoordinator.h

RuntimePublicationCoordinator は

```cpp
publishState(...)
```

で

```cpp
buildRuntimePublishWorld()
```

↓

```cpp
validatePublicationNonRt()
```

↓

```cpp
publishAndSwap()
```

↓

```cpp
retireRuntimePublishWorldNonRt()
```

という流れになっている。 

これは rev13 系で要求していた

「Build → Freeze → Publish → Retire」

そのものです。

---

## 2. RuntimePublishWorld

実装済み

確認できる内容

```cpp
world.isFrozen()
world.isSealedRecursively()
```

を publication 前に検査している。 

これは

Immutable Snapshot Runtime

の中核。

旧RuntimeState直公開ではない。

---

## 3. Closure Graph Validation

実装済み

ISRClosure.cpp に

```cpp
validateClosureGraph()
```

が存在。 

さらに

```cpp
externalMutableDependencies == 0
```

を要求している。

これは

「外部 mutable 依存禁止」

の実装。

かなり重要。

---

## 4. Semantic Schema Validation

実装済み

Publication前に

```cpp
world.schemaVersion
```

確認。 

さらに

```cpp
RuntimeState::validateDescriptorSet()
RuntimeGraph::validateDescriptorSet()
PublicationSemantic::validateDescriptorSet()
```

を実施。 

つまり

Semantic Schema Runtime

になっている。

---

## 5. Generation整合性

Publication前に

```cpp
runtimeGeneration
mappedRuntimeGeneration
generation
```

一致を強制。 

これは

ABA問題

Runtime混線

世代逆転

対策として正しい。

---

## 6. Epoch Retire Runtime

実装済み

確認できる。

```cpp
m_epochDomain.reclaimRetired();
```

```cpp
m_coordinator.reclaim(...)
```

が存在。 

つまり

RCU風

Epoch Based Reclamation

は既に導入済み。

---

## 7. Runtime Build Snapshot

実装済み

```cpp
RuntimeBuildSnapshot
```

```cpp
sealed
```

```cpp
RuntimeBuildFingerprint
```

が存在。 

さらに

```cpp
isRuntimeBuildSnapshotSealedAndCompatible()
```

まである。 

これは

Practical Runtime Fingerprint

そのもの。

---

# 良い意味で予想以上に進んでいる部分

## AuthorityClass

```cpp
Authoritative
Derived
Diagnostic
ExecutorLocal
LegacyTemporary
```

が存在。 

これは以前提案していた

「状態の権威レベル分類」

が実際にコード化されている。

かなり評価できる。

---

## Retire Enqueue Result

```cpp
Success
QueuePressure
QueueFull
Shutdown
```

が存在。 

つまり

Backpressure対応

が実装されている。

実運用では非常に重要。

---

# 残っている問題

ここからが重要。

---

# 問題1

Runtime Graph がまだ完全な唯一権威になっていない

AudioEngineには依然として

大量のatomic状態が残る。 

例

```cpp
eqBypassRequested
convBypassRequested

currentProcessingOrder

analyzerEnabled

ditherBitDepth
```

等。

これは

Runtime Graph

とは別の状態源。

---

結果

現在は

```text
RuntimeGraph
+
AudioEngine atomics
```

の二重権威状態。

---

これは

Practical Runtime

としては許容できるが、

理想的には

RuntimeWorldのみ

へ集約したい。

---

# 問題2

LegacyTemporary が存在

AuthorityClassに

```cpp
LegacyTemporary
```

がある。 

これは

移行残骸管理

用。

つまり作者自身が

「まだ移行完了していない」

と認識している。

---

# 問題3

BarrierOptimizer がダミー

```cpp
void optimizeBarriers()
{
    switch(...)
    {
        ...
    }
}
```

だけ。 

実質何もしていない。

---

現状

存在はするが

ISR Runtime成立には寄与していない。

---

# 問題4

Runtime World監査はあるが実効監査が弱い

現状

```cpp
validatePublicationNonRt()
```

はある。 

しかし

「mutableメンバが残っていない」

ことまでは検証していない。

---

つまり

構造は監査しているが

実体の不変性までは監査していない。

---

# 問題5

DSPCore がまだ巨大

DSPCoreには

```cpp
runtimeUuid
sampleRate
oversampling
outputFilter
adaptiveNoiseShaper
...
```

が大量に内包されている。 

これは Practical Runtime としては許容範囲だが、

今後

RuntimeNode分解

を行う余地がある。

---

# 最も重要な評価

今回の最新版を見る限り、

以前の

```text
Mutable Runtime
↓
RCU Runtime
↓
Immutable Snapshot Runtime
↓
Practical Stable ISR Bridge Runtime
```

という移行計画は、

実装レベルでは約85%程度まで到達しています。

特に

* RuntimePublicationCoordinator
* RuntimePublishWorld
* Closure Validation
* Semantic Schema
* Epoch Retire
* Build Snapshot
* Generation Governance

は実質完成レベルです。

一方で、実運用で最後に事故要因になりやすいのは、

1. AudioEngine atomics と RuntimeWorld の二重権威
2. LegacyTemporary の残存
3. RuntimeWorld外 mutable state の完全洗い出し
4. DSPCore内部の暗黙状態

です。

したがって現時点の監査結論は、

**Practical Stable ISR Bridge Runtime は「概ね成立している」が、「完全移行完了」と判定するにはまだ早い。現在は“実運用可能なβ最終段階”に相当する状態であり、残る作業はアーキテクチャ構築ではなく移行残骸の除去と権威の一本化である。**




最新版の ConvoPeq.md（2026-05-30版）を見ると、前回よりさらに ISR Runtime 化は進んでいます。しかし、Practical Stable ISR Bridge Runtime の完成条件を「RuntimeWorld が唯一の authoritative semantic container」で判定すると、まだ複数の未到達箇所があります。単なるコード品質ではなく、「構造的にまだ ISR Runtime に到達していない箇所」に絞って整理します。

---

# 1. RuntimeWorld が依然として自己完結していない

これは現在の最大の残課題です。

buildRuntimePublishWorld() を見ると、

```cpp
const auto runtimeReadView = readControlRuntimeView();
const auto* runtimeSnapshot = getRuntimeSnapshot(runtimeReadView);
```

から値を取得し、

```cpp
worldOwner->routing.processingOrder =
    runtimeSnapshot->processingOrder
```

を埋めています。 

つまり RuntimeWorld は

```text
RuntimeWorld
 ← RuntimeSnapshot
 ← ControlRuntimeView
 ← AudioEngine
```

に依存しています。

Practical Stable ISR Runtime の完成形では、

```text
RuntimeWorld
 ↓
Snapshot
 ↓
Diagnostics
```

でなければなりません。

現在は逆方向依存が残っています。

これは

```text
Snapshot → World
```

ではなく

```text
World ← Snapshot
```

になっている。

つまり RuntimeWorld がまだ完全な authority source になっていません。

---

# 2. RuntimeGraph が Derived になっている

RuntimeStateを見ると

```cpp
convo::RuntimeGraph graph {};
```

は

```cpp
AuthorityClass::Derived
```

になっています。

しかし実際には

```cpp
activeNode
fadingNode
runtimeUuid
fadingRuntimeUuid
```

は Audio Thread の実行を決定しています。

つまり現実には

```text
Execution Authority
```

を持っています。

これは分類上の矛盾です。

---

理想状態

```text
TopologySemantic
ExecutionSemantic
↓
RuntimeGraph生成
```

であるべきです。

現状

```text
RuntimeGraph
→ Execution決定
```

が残っています。

これは

RuntimeGraph がまだ半分 authority。

---

# 3. RuntimeWorld に EngineRuntime が残っている

現在

```cpp
convo::EngineRuntime engine {};
```

が RuntimeState 内にあります。

しかも

```cpp
engineState
↓
makeRuntimeGraphState(engineState)
```

という生成順です。 

つまり

```text
EngineRuntime
↓
RuntimeGraph
↓
RuntimeWorld
```

です。

本来の完成形は

```text
RuntimeSemanticSchema
↓
RuntimeWorld
↓
Projection
```

です。

EngineRuntime が RuntimeWorld の入力になっている限り、

旧 Runtime の影響が残っています。

---

# 4. RuntimeSemanticSchema が完全適用されていない

現状 Descriptor 検証はあります。

```cpp
RuntimeState::validateDescriptorSet()
```

```cpp
RuntimeGraph::validateDescriptorSet()
```

```cpp
PublicationSemantic::validateDescriptorSet()
```



しかし、

```cpp
engine
resource
automation
coefficient
affinity
```

は

Schema に含まれていません。 

つまり

```text
Schema外オブジェクト
```

がまだ RuntimeWorld に存在する。

これは計画書の

> schema 外の状態は projection か diagnostics に分類する

という条件を満たしていません。 

---

# 5. Publication Sequence が完全単一権威になっていない

現在

```cpp
publicationSequenceCounter_
```

で Sequence を発行しています。 

しかし RuntimeWorld 自身が

```text
publication sequence authority
```

になっていません。

つまり

```text
AudioEngine Counter
↓
RuntimeWorld
```

です。

完成形では

```text
PublicationCoordinator
↓
RuntimeWorld.publication
```

のみが権威です。

まだ AudioEngine に権威が残っています。

---

# 6. Generation が二重管理

現状

```cpp
runtimeGraphRevision
```

と

```cpp
generationSemantic.runtimeGeneration
```

が同時に存在します。 

しかも

```cpp
generationSemantic.runtimeGeneration
    = nextGraphGeneration;
```

で同期しています。

つまり

```text
GraphGeneration
↓ mirror
RuntimeGeneration
```

です。

これはまだ

```text
generation singularization
```

が完了していない状態です。

---

# 7. RuntimeWorld が Builder 専用ではない

最新版では

```cpp
RuntimePublishWorld::createForBuilder(
    RuntimePublishWorld::BuilderToken {}
)
```

が導入されています。 

これは非常に良い改善です。

しかし RuntimeState 自体は

```cpp
struct RuntimeState
```

として公開されています。

つまり理論上

```cpp
RuntimeState world;
```

生成ルートが残っていないかを全域監査する必要があります。

Builder以外から生成可能なら、

ISR Runtime の

```text
Construction Authority
```

が単一化されていません。

---

# 8. Shadow Compare 卒業条件が見当たらない

計画では

```text
Shadow Compare
↓
一致率確認
↓
Legacy Path削除
```

が必要です。

しかしコード断片からは

```cpp
RuntimeSemanticHash
```

は存在するものの、

```cpp
legacy path vs ISR path
```

の常時比較系が十分には確認できません。 

つまり

```text
移行完了を判定する仕組み
```

がまだ弱い可能性があります。

---

# 9. RuntimeWorld 外の Atomic Authority がまだ残存している可能性が高い

buildRuntimePublishWorld() 内で

```cpp
consumeAtomic(currentProcessingOrder)
```

を読んでいます。 

これは極めて重要な兆候です。

RuntimeWorld が唯一権威なら、

```cpp
RuntimeWorld
↓
Audio Thread
```

だけになるはずです。

しかし現状は

```cpp
Atomic currentProcessingOrder
↓
RuntimeWorld
```

です。

つまり

```text
Authority Migration
```

がまだ終わっていません。

---

# 10. RuntimeWorld が Semantic Container ではあるが Semantic Closure ではない

これは最も本質的な問題です。

現在の RuntimeWorld は

```text
GenerationSemantic
TopologySemantic
RoutingSemantic
ExecutionSemantic
PublicationSemantic
RetireSemantic
```

を持っています。 

しかし buildRuntimePublishWorld() を見ると、

値の多くは

```text
AudioEngine
RuntimeSnapshot
Atomics
DSPCore
```

から収集されています。 

つまり

```text
RuntimeWorld = Semantic Container
```

にはなっています。

しかし

```text
RuntimeWorld = Semantic Closure
```

にはなっていません。

Practical Stable ISR Runtime の最終到達点は

```text
RuntimeWorldだけ見れば
実行意味論が完全に決定できる
```

状態です。

現状はまだ

```text
AudioEngine
DSPCore
Atomic State
RuntimeSnapshot
```

を参照しないと意味論が閉じません。

---

# 現時点での到達度

前回は約85%と評価しましたが、最新版を構造面だけで再評価すると、

| 項目                           | 到達度 |
| ---------------------------- | --: |
| Publication Collapse         | 95% |
| Immutable Snapshot Runtime   | 95% |
| Epoch Retire                 | 90% |
| RuntimeWorld化                | 90% |
| Semantic Schema化             | 85% |
| Authority Singularization    | 75% |
| Observe Collapse             | 80% |
| Semantic Closure             | 70% |
| Single Authoritative Runtime | 75% |

です。

したがって、残っている本質的課題は「Publication」や「Retire」ではなく、

**RuntimeWorld が唯一の Semantic Closure になるまで、AudioEngine・DSPCore・Atomic群に残っている実質的な authority を RuntimeWorld に吸収し切れていないこと**

です。これが Practical Stable ISR Bridge Runtime 完全到達を妨げている最大要因です。




前回挙げた「Authority 一本化」「Semantic Closure」「RuntimeWorld 自己完結性」以外にも、最新版 ConvoPeq.md を Practical Stable ISR Runtime v3.1 の完成条件と照合すると、まだ監査対象になる箇所があります。

特に重要なのは、**現在のコードは RuntimeWorld を中心に構築されているが、RuntimeWorld を検証するための周辺統治機構がまだ完成しているとは言い切れない**点です。

---

# 11. Authority Inventory と RuntimeSemanticSchema の完全一致が未確認

計画書では

```cpp
RuntimeSemanticSchema
{
    GenerationSemantic
    TopologySemantic
    RoutingSemantic
    ExecutionSemantic
    PublicationSemantic
    OverlapSemantic
    RetireSemantic
}
```

のみが authority を持てる契約です。

しかし現状 RuntimeState には

```cpp
TimingSemantic
LatencySemantic
SchedulingSemantic
ResourceSemantic
AutomationSemantic
CoefficientSemantic
AffinitySemantic
```

が追加されています。 

問題は、

```text
これらが RuntimeSemanticSchema 拡張として
正式登録済みか
```

が確認できないことです。

---

Practical ISR Runtime の観点では、

```text
Authorityを持つ
↓
Schemaに存在する
↓
Verifierが存在する
```

が必要です。

もし

```cpp
TimingSemantic
LatencySemantic
SchedulingSemantic
```

が実質 authority を持っているのに、

Schema契約や verifier が追随していないなら、

それは

```text
Schema Drift
```

です。

これはまだ監査が必要です。 

---

# 12. RuntimeFieldDescriptor が RuntimeState 全項目を網羅していない可能性

RuntimeState 冒頭を見ると

```cpp
static constexpr std::array<RuntimeFieldDescriptor, 7>
```

しか見えません。 

しかし RuntimeState には

```text
20個以上
```

の semantic field が存在します。

もし実際に Descriptor が7項目しかないなら、

```text
Authority Inventory
≠
実フィールド
```

です。

---

計画書では

```text
schema と authority inventory が一致
```

が完了条件です。

そのため

```text
Descriptor完全網羅率
```

は詳細監査が必要です。

---

# 13. OverlapSemantic が真の唯一権威になっているか不明

v2.3 では

```text
overlap authority は RuntimeWorld.overlap のみ
```

です。

しかし ConvoPeq は歴史的に

```text
preparedCrossfade
fade atomics
TransitionState
active/fading DSP
```

など多数の overlap 情報源を持っていました。 

---

現在、

```cpp
OverlapSemantic overlap
```

は存在します。 

しかし

```text
旧 crossfade 情報源が
全て projection 化されたか
```

までは確認できません。

---

ここは非常に危険です。

Practical ISR Runtime が最後に壊れるのは、

大抵

```text
Crossfade
Overlap
Transition
```

です。

---

# 14. RuntimeGraph が Projection 化し切れていない可能性

計画では

```text
RuntimeGraph 等は authority ではなく projection
```

です。

しかし RuntimeState では

```cpp
RuntimeGraph graph;
```

が依然として保持されています。 

しかも

```cpp
AuthorityClass::Derived
```

扱いです。 

---

Derived は

```text
意味を持たない
```

ではありません。

Derived が execution 分岐に使われた瞬間、

authority に昇格します。

---

つまり監査対象は

```text
graph を参照した branch
```

です。

これが残っていると

```text
RuntimeWorld
+
RuntimeGraph
```

の二重権威になります。

---

# 15. Legacy Publication API の残存有無が不明

計画書は

```text
legacy publication API zero-call
```

を要求しています。

現在は

```cpp
commitRuntimePublication()
retireRuntimePublication()
```

経由になっています。 

これは良い状態です。

---

しかし重要なのは

```text
Coordinator以外から
runtimeGraphRevision
runtimeStore
runtime pointer
を書き換える経路
```

です。

---

つまり

```text
Publication Path Singularization
```

が本当に成立しているかは、

全 publish 経路の grep 監査が必要です。

---

# 16. Observe Path Collapse 完了が未証明

計画書では

```text
AudioThread observe source = RuntimeWorld only
```

です。

しかし ConvoPeq は長期間

```text
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
currentProcessingOrder
pendingRuntime
```

等を直接参照してきました。 

---

現状 RuntimeWorld はあります。

しかし

```text
Observe Verifier pass
```

がコード上で十分強制されているかは別問題です。

---

つまり

```text
RuntimeWorld以外を読む AudioThread コード
```

が1箇所でも残れば、

Observe Collapse は未完了です。

---

# 17. PublicationEpoch ↔ RuntimeGeneration の完全統治が未確認

v2.3 の重要契約は

```text
PublicationEpoch
↓
RuntimeGeneration
```

単調対応です。

---

現在、

```cpp
generation
publication
mappedRuntimeGeneration
```

は存在します。 

しかし

```text
Epoch が飛ぶ
Generation が飛ぶ
```

ケースの fail-closed 動作までは確認できません。

---

つまり

```text
Monotonic Mapping Verifier
```

の実装完成度監査が必要です。

---

# 18. LegacyTemporary 管理の完成度が不明

v3.1 はかなり厳しいです。

```text
LegacyTemporary は
manifest 登録必須
```

です。

さらに

```text
2 phase超存続禁止
```

です。 

---

しかしコード側では

```text
LegacyTemporary
```

分類の存在は確認できますが、

```text
manifest
expiry
owner
replacement_authority
```

まで CI 強制されているかは不明です。

---

これは運用面で未完成の可能性があります。

---

# 19. Shadow Compare が Release Gate に統合されているか不明

計画では

```text
semantic hash
generation
topology
overlap
retire ordering
```

比較を要求しています。

---

RuntimeSemanticHash はあります。

しかし

```text
Nightly
Release
Rollback Trigger
```

まで接続されているかは確認できません。

---

つまり

```text
検出できる
≠
防止できる
```

です。

---

# 20. Soak Governance が未確認

実は最終段階で最も重要です。

計画書の完成条件は

```text
long-run drift 不検出
```

です。

---

つまり必要なのは

```text
数時間〜数日
```

レベルの

* publication monotonicity
* retire backlog
* semantic hash
* visibility delay
* overlap drift

監視です。

---

コードを見る限り、

RuntimeWorld 構造そのものはかなり整っています。

しかし

```text
長期運用で壊れない
```

ことを保証する

```text
Operational Governance
```

まで完成しているかはまだ不明です。

---

## 現時点で追加発掘できる本質的未到達領域

重要度順に並べると、

1. RuntimeSemanticSchema と実フィールドの完全一致
2. RuntimeFieldDescriptor の完全網羅
3. Overlap/Crossfade authority の完全移管
4. RuntimeGraph の完全 projection 化
5. Publication 経路の完全単一化
6. Observe Path Collapse の実証
7. PublicationEpoch ↔ RuntimeGeneration の fail-closed 保証
8. LegacyTemporary manifest 運用完成
9. Shadow Compare の Release Gate 統合
10. Long-run Soak Governance 完成

です。

前回までに挙げた「Semantic Closure」「Authority Singularization」「RuntimeWorld 自己完結性」に加えると、現状の到達度は概ね 85〜90% 程度ですが、**残っている課題の多くは実装そのものではなく、『RuntimeWorld が唯一の意味源であることを継続的に証明する統治機構』に集中している**と評価できます。




前回までの指摘に加えて、最新版ソースを見ると、さらに「Practical Stable ISR Bridge Runtime の完成条件から見ると未到達」と評価できる箇所があります。

重要なのは、今回は「単なる改善余地」ではなく、**RuntimeWorld を唯一の意味源（Single Semantic Authority）とする構造がまだ崩れている箇所**に限定している点です。

---

# 21. RuntimeState の Descriptor と実体がまだ乖離している

最新版では Descriptor 数が 7→9 に増えています。

しかし RuntimeState 実体には、

* generationSemantic
* topology
* routing
* execution
* publication
* overlap
* retire
* timing
* latency
* scheduling
* resource
* affinity
* automation
* coefficient
* projectionFreshness
* semanticHash

など多数のフィールドがあります。 

一方 Descriptor 側は

```cpp
worldId
generation
graph
publication
metadata
retire
timing
latency
scheduling
```

程度しか登録されていません。 

つまり現状は

```text
RuntimeWorld実体
≠
Descriptor Inventory
```

です。

Practical ISR Runtime の完成形では

```text
Authority Field
↓
Descriptor
↓
Validator
```

が完全一致していなければなりません。

これはまだ未完成です。

---

# 22. validateDescriptorSet() が RuntimeWorld 全体を検証していない

現在の precheck は

```cpp
RuntimeState::validateDescriptorSet()
RuntimeGraph::validateDescriptorSet()
PublicationSemantic::validateDescriptorSet()
```

だけです。 

しかし RuntimeState に存在する

```cpp
OverlapSemantic
GenerationSemantic
RoutingSemantic
TimingSemantic
LatencySemantic
SchedulingSemantic
```

等について、

個別 descriptor validator が連鎖している形跡が見えません。

つまり

```text
Descriptor Validation
```

は存在するが

```text
Schema Closure Validation
```

には到達していません。

---

# 23. EngineRuntime が RuntimeWorld 内に残っている

RuntimeState 冒頭を見ると、

```cpp
convo::EngineRuntime engine {};
```

が残っています。 

しかも

```cpp
AuthorityClass::Derived
```

扱いです。 

これは非常に重要です。

Practical ISR Runtime の完成形では

```text
EngineRuntime
↓
RuntimeWorld
```

ではなく

```text
RuntimeWorld
↓
EngineProjection
```

であるべきです。

現在は

```text
EngineRuntime
→ RuntimeWorld
```

が残っています。

つまり RuntimeWorld がまだ完全な根源ではありません。

---

# 24. RuntimeGraph 依存が Validator に混入している

precheck を見ると、

```cpp
world.graph.activeNode
world.graph.fadingNode
```

を直接検査しています。 

さらに

```cpp
world.engine.transition.next
```

も参照しています。 

つまり Validator が

```text
Semantic
+
Projection
```

を混在して検証しています。

本来は

```text
Semantic Validation
↓
Projection Validation
```

で分離されるべきです。

---

# 25. RuntimePublishView が RuntimeGraph を露出している

RuntimePublishView を見ると、

```cpp
const RuntimeGraph* graph
```

を保持しています。 

さらに RuntimeReadView も

```cpp
const RuntimeGraph* graph
```

を保持しています。 

これは

```text
Observe Source
```

がまだ

```text
RuntimeWorld
```

ではなく

```text
RuntimeGraph
```

であることを意味します。

---

完成形は

```text
RuntimeWorldView
↓
Projection取得
```

です。

現在は

```text
RuntimeReadView
↓
RuntimeGraph直参照
```

になっています。

---

# 26. activeRuntimeDSPSlot / fadingRuntimeDSPSlot が残っている

これはかなり大きいです。

現状、

```cpp
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
```

が AudioEngine 内に存在します。 

さらに

```cpp
getActiveRuntimeDSP()
setActiveRuntimeDSP()
```

が存在します。 

これは

```text
RuntimeWorld
```

以外の実行権威です。

Practical ISR Runtime の完成形では、

```text
ExecutionSemantic.activeRuntime
```

のみが権威であるべきです。

---

# 27. commitNewDSP() が RuntimeWorld ではなく DSP ポインタを publish している

commitNewDSP を見ると、

```cpp
setActiveRuntimeDSP(newDSP);
```

を実行しています。 

さらに

```cpp
publishState(newDSP,...)
```

しています。 

つまり publish 単位が

```text
RuntimeWorld
```

ではなく

```text
DSPCore*
```

です。

これは旧 Runtime の発想がまだ残っています。

---

完成形では

```text
RuntimeWorld publish
↓
Execution projection update
```

です。

---

# 28. PublicationIntent Log がまだ RuntimeWorld 外に存在する

PublicationIntent Queue が残っています。 

```cpp
publicationLog.head
publicationLog.consumedTail
publicationLog.retiredHead
```

等です。 

これは Runtime Publication Coordinator の外側に

```text
別 publication state machine
```

が存在していることを意味します。

---

理想構造では

```text
PublicationCoordinator
```

だけが publication state を持ちます。

現在は二重化しています。

---

# 29. Runtime Publication Backlog が RuntimeWorld に入っていない

現在

```cpp
publicationBacklog_
rebuildBacklog_
publicationRejectCount_
```

等が AudioEngine atomic として存在します。

これは

```text
Operational Governance
```

情報です。

Practical ISR Runtime では、

最低限

```text
Publication Health
Retire Health
Visibility Health
```

は Runtime Governance Semantics に集約されるべきです。

現状は Engine 側に散在しています。

---

# 30. RuntimeWorld と DSP Handle Runtime が並列権威になっている

commitNewDSP では

```cpp
dspHandleRuntime_.activate(...)
dspHandleRuntime_.retire(...)
```

を実行しています。 

つまり

```text
RuntimeWorld
```

だけでなく

```text
DSPHandleRuntime
```

も実行状態を保持しています。

これは

```text
Execution Authority Split
```

です。

---

# 31. Crossfade Authority が RuntimeWorld に完全統合されていない

commitNewDSP 内で

```cpp
crossfadeAuthorityRuntime_.registerCrossfade(...)
```

があります。 

つまり

```text
OverlapSemantic
```

とは別に

```text
CrossfadeAuthorityRuntime
```

が存在しています。

これは典型的な

```text
二重権威
```

です。

---

# 32. Runtime Publication Coordinator が RuntimeGraphRevision を直接操作している

retireRuntimePublishWorldNonRt で

```cpp
runtimeGraphRevision = 0
```

が行われています。 

これは

```text
Publication Coordinator
```

が

```text
RuntimeGraph State
```

を直接変更していることを意味します。

理想的には

```text
PublicationSemantic
```

のみを扱うべきです。

---

# 総括

前回までの指摘を除外しても、最新版にはまだ以下の未到達領域があります。

1. Descriptor Inventory と RuntimeWorld 実体の不一致
2. Schema Closure Validation 不完全
3. EngineRuntime の残存
4. Validator の Projection 依存
5. RuntimeGraph 直観測経路
6. active/fading DSP slot の残存
7. DSPCore* publish 構造
8. PublicationIntent State Machine 残存
9. Governance 情報の Engine 側残存
10. DSPHandleRuntime 並列権威
11. CrossfadeAuthorityRuntime 並列権威
12. RuntimeGraphRevision 依存

これらはすべて共通しており、

**「Publication」「Retire」「Epoch」「Freeze」はほぼ完成しているが、Execution/Observe/Crossfade の権威がまだ RuntimeWorld に完全収束していない**

という一点に集約されます。

現状はおそらく **Practical Stable ISR Bridge Runtime の 88〜92% 程度**で、残作業の中心は RuntimeWorld 構築ではなく **権威の一本化（Authority Collapse）と Projection 分離の完遂**です。
