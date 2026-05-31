この設計書は前回指摘した「未達7項目」をかなり正確に取り込めています。

ただし、ConvoPeq の現状コードベースと Practical Stable ISR Runtime v3.1 / base_plan v2.3 の両方を踏まえると、

**「方向性は正しいが、このまま実装すると実際には未達成のまま残る箇所がかなりある」**

というのが結論です。

特に問題なのは、

* 設計思想レベル
* 実装統治レベル
* 検証レベル

は書かれているが、

**RuntimeWorld完全収束のための具体的な撤去対象**
が不足しています。

以下、重要度順に整理します。

---

# Critical-1

# RuntimeWorld定義が曖昧

本設計最大の問題です。

設計書では

```cpp
RuntimeWorld
```

を唯一authorityと書いていますが、

現状ConvoPeqには

* RuntimeState
* RuntimeGraph
* RuntimeBuildSnapshot
* PublicationIntent
* TransitionState
* CrossfadePreparedSnapshot

などが存在します。

設計書は

```text
RuntimeWorldを唯一authorityにする
```

とは書いていますが、

どれをRuntimeWorldに統合するか

が未定義です。

---

必要な追加設計

```cpp
RuntimeWorld
{
    RuntimeTopology
    RuntimeRouting
    RuntimeExecution
    RuntimePublication
    RuntimeOverlap
    RuntimeRetire
}
```

を定義する。

さらに

```text
RuntimeGraph
→ topology projection

RuntimeBuildSnapshot
→ build artifact

TransitionState
→ executor local

CrossfadePreparedSnapshot
→ executor local
```

へ降格することを明記する。

---

これがないと

RuntimeWorld導入後も

Authority二重化

が残ります。

---

# Critical-2

# Legacy Temporary除去計画がない

3.1では

```text
LegacyTemporary
```

が定義されています。

しかし設計書には

除去期限

がありません。

---

必要追加

各LegacyTemporaryに

```cpp
legacyId
owner
introducedPR
removalPR
expiryVersion
```

を持たせる。

期限超過で

CI Fail。

---

これをやらないと

LegacyTemporaryが永久化します。

これは過去のConvoPeqでも何度も起きています。

---

# Critical-3

# PublicationIntentの扱いが危険

設計書では

```cpp
PublicationIntent
は transport queue
```

となっています。

しかしこれは不十分。

---

追加必要

PublicationIntent禁止事項

```cpp
generation
overlap
routing
execution
```

を持ってはならない。

---

許可されるのは

```cpp
publishRequestId
targetWorldId
enqueueTime
```

程度。

---

さもないと

PublicationIntentが再びauthority化します。

---

# Critical-4

# ObservePathVerifier実装不能

設計書は

```cpp
ObservePathVerifier
```

を要求しています。

しかし静的解析だけでは無理です。

---

例えば

```cpp
graph->getNode()
```

がAudioThread内に残っても、

Verifierは検出できません。

---

必要追加

禁止リストを作る。

AudioThreadで

禁止型

```cpp
RuntimeGraph*
RuntimeBuildSnapshot*
PublicationIntent*
TransitionState*
```

を参照したらCI Fail。

---

これは設計書に必要です。

---

# Critical-5

# Overlap Authority設計不足

現状ConvoPeqは

Crossfade系がかなり多い。

そのため

```cpp
world.overlap
```

だけでは不十分です。

---

追加必要

OverlapSemantic定義。

最低限

```cpp
overlapMode
fadeShape
fadeDuration
overlapGeneration
```

を持つ。

---

ExecutorLocalは

```cpp
fadeProgress
sampleCounter
```

のみ。

---

これを分離しないと

Overlap Authority Leakage

が再発します。

---

# High-1

# RuntimeBuildSnapshotの位置付け不足

現設計では

BuildSnapshotが曖昧。

---

必要追加

BuildSnapshotは

```cpp
immutable
non-authoritative
pre-publish only
```

と定義。

publish後保持禁止。

---

残ると

Snapshot Authority

になります。

---

# High-2

# Projection Freshness契約が弱い

設計書に

```cpp
projection freshness
```

が少し出てきます。

しかし判定方法がない。

---

追加必要

```cpp
worldGeneration
projectionGeneration
```

比較。

---

許容差

```cpp
0
```

固定。

---

# High-3

# Retire Pressure Policyが危険

75/90/95%

固定値は危険。

環境依存。

---

必要追加

```cpp
retireQueueCapacity
```

依存で自動算出。

さらに

Evidenceから学習可能にする。

---

# High-4

# Starvation Contract不足

設計書は

```cpp
maxRetireDeferralEpochs
```

のみ。

---

不足。

さらに

```cpp
maxRetireWallClockMs
```

必要。

---

世代が進まない場合、

epochだけでは回収保証できません。

---

# High-5

# Semantic Hash設計不足

現設計は

```cpp
semantic hash
```

だけ。

---

必要追加

分割ハッシュ。

```cpp
topologyHash
routingHash
executionHash
publicationHash
overlapHash
retireHash
```

---

原因解析が容易になります。

---

# High-6

# Shadow Compare不足

現設計では

比較対象があるだけ。

---

必要追加

Mismatch Taxonomy。

```cpp
TopologyMismatch
ExecutionMismatch
PublicationMismatch
RetireMismatch
VisibilityMismatch
```

---

これがないと

Evidenceが役に立ちません。

---

# Medium-1

# Rollback設計がない

Practical Stable ISR Runtime v3.1で重要なのは

Fail Closed

だけではありません。

---

必要追加

```cpp
publish reject
↓
previous world continue
```

保証。

---

# Medium-2

# World Lifecycle設計不足

RuntimeWorldの寿命が未定義。

---

必要追加

状態機械。

```cpp
Building
Prepared
Published
Observed
Retiring
Retired
Destroyed
```

---

# Medium-3

# Verifier階層不足

現設計

```text
PR
Nightly
Release
```

だけ。

---

さらに

```text
Compile-time
Unit
Integration
Soak
```

階層化した方がよい。

---

# Medium-4

# Exit Criteria不足

現在

```text
Partial publication = 0
```

などしかない。

---

追加必要。

```text
Authority Leak = 0

Observe Leak = 0

Projection Freshness Violation = 0

Retire Starvation = 0

Semantic Drift = 0
```

---

# 現時点の総合評価

この設計書は、

前回指摘した

* Partial Publication
* Observe Collapse
* Overlap Authority
* Retire Pressure
* Starvation
* Semantic Hash
* Soak Governance

を概ね取り込めており、

**方向性としては80～85点程度**です。

ただしConvoPeqを本当に

> Practical Stable ISR Bridge Runtime 完全移行

へ持っていくには、

特に次の6項目が未記載です。

1. RuntimeWorldへの具体的統合作業
2. RuntimeGraph/BuildSnapshot/TransitionState降格計画
3. LegacyTemporary除去契約
4. Observe禁止型契約
5. RuntimeWorldライフサイクル定義
6. Rollback/Fail-Safe契約

この6項目を追加しない限り、実装後も「Authorityが複数残る Bridge Runtime」の状態から完全には脱出できない可能性が高いです。



あります。

むしろ前回までの指摘は「Runtime Semantic Governance層」の話が中心で、Practical Stable ISR Bridge Runtime を長期間運用して本当に事故を防ぐために必要な項目のうち、まだ設計書に入っていないものが残っています。

以下は優先度順です。

---

# Critical-6

# RuntimeWorld不変性(Immutable Contract)が未定義

現在の設計書は

```cpp
publish(world)
```

のみを定義しています。

しかし

```cpp
publish後にworldが変更可能
```

なら全て崩壊します。

---

必要追加

```cpp
Published RuntimeWorld
=
strict immutable
```

を明文化。

禁止事項

```cpp
world->routing = ...
world->overlap = ...
world->retire = ...
```

publish後変更禁止。

---

Verifier

```cpp
ImmutableWorldVerifier
```

追加。

---

実運用では非常に重要です。

現在の設計書はここが抜けています。

---

# Critical-7

# Generation Monotonicity契約不足

ISR Runtimeで最も危険なのは

```cpp
generation rollback
```

です。

---

必要追加

```cpp
generation(N+1) > generation(N)
```

絶対保証。

禁止

```cpp
N
↓
N+2
↓
N+1
```

---

Verifier

```cpp
GenerationMonotonicityVerifier
```

---

これがないとVisibility問題が発生します。

---

# Critical-8

# Publication Visibility契約不足

計画書v3.1でかなり重要な項目です。

現設計は

```cpp
publish(world)
```

だけ。

---

しかし必要なのは

```cpp
visible generation
```

の単調増加保証です。

---

必要追加

```cpp
visibleGeneration
```

は

```cpp
old <= new
```

のみ許可。

---

禁止

```cpp
g100
↓
g101
↓
g100
```

---

Verifier

```cpp
VisibilityMonotonicityVerifier
```

---

これはObserve Collapseと別物です。

---

# Critical-9

# RuntimeWorld自己完結性(Self-contained)不足

設計書では

```cpp
RuntimeWorld
```

を唯一authorityと書いています。

しかし

---

危険例

```cpp
world
  ↓
graphRegistry
  ↓
lookup
```

---

これでは

RuntimeWorld外依存

です。

---

必要追加

```cpp
worldのみで意味解釈可能
```

契約。

---

禁止

```cpp
global registry
singleton lookup
external authority table
```

---

これが無いとRuntimeWorld化は未完成です。

---

# Critical-10

# Publication Failure Taxonomy不足

現在

```cpp
reject
```

だけ。

---

必要追加

```cpp
SchemaFailure
AuthorityFailure
ProjectionFailure
GenerationFailure
RetireFailure
VisibilityFailure
```

分類。

---

実運用では

Reject理由が見えないと解析不能。

---

# High-7

# RuntimeWorld Versioning不足

将来変更時に破綻します。

---

必要追加

```cpp
schemaVersion
semanticVersion
```

保持。

---

Verifier

```cpp
VersionCompatibilityVerifier
```

---

長期運用では必須。

---

# High-8

# RuntimeWorld Migration契約不足

versionが増えると

```cpp
v1
↓
v2
```

移行が必要。

---

追加必要

```cpp
MigrationStrategy
```

定義。

---

将来の大型改修に効きます。

---

# High-9

# Build-Publish分離保証不足

設計書は

```cpp
Build
↓
Publish
```

と書いています。

しかし禁止事項がない。

---

必要追加

Build中に

```cpp
current world
```

参照禁止。

---

つまり

```cpp
Build Isolation Contract
```

が必要。

---

旧mutable runtimeへの逆流を防ぎます。

---

# High-10

# Runtime Leak Budget不足

長期運用では重要。

---

必要追加

Evidence項目

```cpp
liveWorldCount
retiredWorldCount
pendingRetireCount
```

---

閾値超過

↓

Fail。

---

Soakで効きます。

---

# High-11

# Publication Latency Budget不足

Practical Runtimeで非常に重要。

---

現在

```cpp
publish成功
```

しか見ていない。

---

必要追加

```cpp
build→publish latency
```

計測。

---

例

```cpp
P95
P99
Max
```

記録。

---

遅延ドリフト発見に必須。

---

# High-12

# Retire Latency Budget不足

現在

```cpp
retireされた
```

のみ。

---

必要追加

```cpp
publish→retire
```

時間。

---

長時間運用で重要。

---

# Medium-5

# Emergency Mode設計不足

設計書では

```cpp
Critical
↓
emergency drain
```

のみ。

---

必要追加

Emergency Runtime Mode

```cpp
publication stop
rebuild stop
retire priority
```

定義。

---

# Medium-6

# Backpressure Propagation不足

Retire Pressureがあっても

上流へ伝わらない。

---

必要追加

```cpp
RetirePressure
↓
BuildAdmission
↓
PublicationAdmission
```

伝播。

---

これがないとQueue崩壊。

---

# Medium-7

# Runtime Consistency Snapshot不足

診断用。

---

必要追加

```cpp
world consistency dump
```

生成。

---

障害解析が容易。

---

# Medium-8

# Determinism検証不足

ISR Runtimeは本質的に

```cpp
same input
→ same world
```

が重要。

---

追加Verifier

```cpp
DeterministicBuildVerifier
```

---

同入力で

semanticHash一致。

---

# Medium-9

# Authority Ownership監査不足

Authority Inventoryだけでは不足。

---

追加必要

各Authorityに

```cpp
owner
writer
reader
```

明示。

---

責務漏れを防ぐ。

---

# Medium-10

# CI Fail-Closed定義不足

設計書は

```cpp
fail closed
```

と言うだけ。

---

実際には

どのVerifierが

```cpp
Warning
Error
Fatal
```

か未定義。

---

Severity Tableが必要。

---

# 現時点で最も重要な追加項目

もし私がこの設計書に追加するなら、優先順位は次です。

1. Immutable RuntimeWorld Contract
2. Generation Monotonicity Contract
3. Visibility Monotonicity Contract
4. RuntimeWorld Self-contained Contract
5. Build Isolation Contract
6. Deterministic Build Contract
7. Runtime Leak Budget
8. Publication Latency Budget
9. Retire Latency Budget
10. Failure Taxonomy

この10項目は、これまでの議論で出てきた「Authority整理」や「Observe一本化」とは別系統ですが、実際の長期運用で発生する事故（世代逆行、可視性逆転、Build汚染、リーク蓄積、遅延ドリフト）を防ぐために必要な要素です。

特に **Generation Monotonicity・Visibility Monotonicity・Build Isolation** の3つは、現行設計書に欠落している中では最優先で追加すべき事項です。




まだあります。

ここから先は「Practical Stable ISR Runtime を成立させるための要件」ではなく、

**ConvoPeq のような実DSPアプリを数か月～数年保守し続けても Runtime が再劣化しないための統治・進化・運用設計**

の領域です。

前回までの指摘と重複しないものに限定すると、以下が残っています。

---

# Critical-11

# RuntimeWorld内部参照禁止契約

設計書は

```cpp
RuntimeWorld
```

を自己完結にしようとしています。

しかしさらに重要なのは

```cpp
RuntimeWorld
 ↓
shared_ptr<Node>
 ↓
外部所有
```

です。

---

危険例

```cpp
RuntimeWorld
{
    Node* node;
}
```

node が RuntimeWorld 外で管理される。

---

これでは

```cpp
RuntimeWorld only
```

が成立しません。

---

必要追加

```cpp
RuntimeWorld Reference Closure Contract
```

---

許可

```cpp
value
immutable snapshot
runtime-owned object
```

---

禁止

```cpp
external mutable object
singleton
global state
```

---

これは実装後に事故が起きやすい箇所です。

---

# Critical-12

# Hidden Authority Detector

Authority Inventoryを作っても、

人間やAIは新しいauthorityを作ります。

---

例

```cpp
cachedCurrentGeneration
```

```cpp
lastKnownOverlap
```

```cpp
pendingExecutionMode
```

---

こういうものが実質authorityになります。

---

必要追加

```cpp
HiddenAuthorityVerifier
```

---

条件

```cpp
branch source
decision source
routing source
```

になっている変数を検出。

---

これは長期的に極めて重要です。

---

# Critical-13

# RuntimeWorld Fork禁止契約

非常に見落とされやすい。

---

危険例

```cpp
RuntimeWorld A
 ↓
copy
 ↓
RuntimeWorld B
```

---

その後

```cpp
A publish
B modify
```

になる。

---

結果

semantic divergence

発生。

---

必要追加

```cpp
Forked RuntimeWorld Prohibited
```

---

許可

```cpp
BuildSnapshot
 ↓
Build
 ↓
New RuntimeWorld
```

のみ。

---

# Critical-14

# Publication Ordering Contract

現設計は

generation monotonic

のみ。

---

しかし

```cpp
publish request
```

の順序も必要。

---

例

```cpp
request 10
request 11
```

なのに

```cpp
11 publish
10 publish
```

は不可。

---

必要追加

```cpp
PublicationOrderingVerifier
```

---

# High-13

# RuntimeWorld Identity Contract

世界が変わったのか

中身だけ変わったのか

曖昧。

---

必要追加

```cpp
worldId
generation
semanticHash
```

の役割を明確化。

---

例

```cpp
worldId
=
Runtimeの個体

generation
=
公開世代

semanticHash
=
内容
```

---

混同禁止。

---

# High-14

# Semantic Compatibility Contract

将来の大型改修用。

---

必要追加

```cpp
isCompatible(worldA, worldB)
```

---

これが無いと

crossfadeやoverlapで

比較不能。

---

# High-15

# Runtime Admission Control

現設計は

Retire側だけ制御。

---

不足。

Build開始前に

```cpp
retire backlog
memory pressure
publish backlog
```

確認。

---

危険なら

```cpp
build reject
```

する。

---

つまり

```cpp
AdmissionControl
```

が必要。

---

# High-16

# Semantic Debt管理

LegacyTemporaryだけでは足りません。

---

必要追加

```cpp
SemanticDebtManifest
```

---

内容

```cpp
temporary authority
temporary projection
temporary compatibility path
```

---

期限付き管理。

---

AI実装では特に重要。

---

# High-17

# Runtime Memory Budget

現設計は

Retire Queue中心。

---

しかし

RuntimeWorld巨大化がある。

---

必要追加

```cpp
RuntimeMemoryBudget
```

---

監視

```cpp
world size
retained worlds
retire backlog memory
```

---

DSPでは重要。

---

# High-18

# Runtime Churn Budget

Practical Runtimeで重要。

---

例

```cpp
1秒間に100回 publish
```

---

動くが不安定。

---

必要追加

```cpp
publish/sec
build/sec
retire/sec
```

監視。

---

過剰ならThrottle。

---

# Medium-11

# Diagnostic Projection Contract

projectionは残る。

---

必要追加

分類。

```cpp
Authority Projection
Diagnostic Projection
UI Projection
```

---

混在禁止。

---

# Medium-12

# Evidence Schema Versioning

Evidence Exporterが既にある。

しかし

形式変更時に壊れる。

---

必要追加

```cpp
evidenceVersion
```

---

将来解析ツールを守る。

---

# Medium-13

# Runtime Replay Contract

障害解析用。

---

必要追加

```cpp
World Timeline
Publication Timeline
Retire Timeline
```

再生可能。

---

実運用で非常に有効。

---

# Medium-14

# Runtime Audit Trail

現在のEvidenceより一段上。

---

追加

```cpp
誰が
いつ
どのAuthorityを変更したか
```

追跡。

---

将来AI実装で効く。

---

# Medium-15

# Governance Drift Detector

非常に重要。

---

時間が経つと

Verifier自体が古くなる。

---

必要追加

```cpp
Authority Inventory
Schema
Verifier
```

の整合チェック。

---

つまり

```cpp
Governance validates Governance
```

です。

---

# 実際に最後まで残りやすい未記載項目 TOP10

もし私がAIに実装させる前に設計書へ追記するなら、

優先順位は次です。

1. RuntimeWorld Reference Closure Contract
2. Hidden Authority Detector
3. RuntimeWorld Fork禁止
4. Publication Ordering Contract
5. Admission Control
6. Runtime Memory Budget
7. Runtime Churn Budget
8. Semantic Debt Manifest
9. Runtime Replay Contract
10. Governance Drift Detector

この10項目は、前回までの「Authority統合」「Observe一本化」「Retire統治」とは別系統ですが、実際のConvoPeqのような継続開発プロジェクトでは後から再発するアーキテクチャ崩壊を防ぐために非常に有効です。

特に **Hidden Authority Detector** と **Governance Drift Detector** は、AI主導で長期間改修を続けるプロジェクトでは追加価値が大きい項目です。



ここまででかなり深いところまで掘れていますが、さらに「Practical Stable ISR Bridge Runtime を完成させた後に再び崩壊しないための設計」という観点で見ると、まだ残っています。

ただし、ここから先は優先度が一段下がります。

前回までの

* Authority統合
* Observe Collapse
* Publication Governance
* Retire Governance
* RuntimeWorld化
* Hidden Authority対策

とは別系統です。

---

# Architecture Regression Prevention が欠落

これは設計書全体で最も不足している運用設計です。

現在の設計は

```text
完成させる
```

までは書いてあります。

しかし

```text
完成後に壊さない
```

がありません。

---

# Critical-15

# Authority Regression Gate

現在の設計では

Authority Inventory

があります。

しかし

将来

```cpp
NewGlobalRuntimeState
```

みたいなものが追加されたら終わりです。

---

追加すべき契約

```cpp
Authority Registry
```

を唯一の登録場所にする。

---

新Authority追加時は

```cpp
owner
reason
removal plan
```

必須。

---

未登録Authority

↓

CI Fail

---

これは非常に有効です。

---

# Critical-16

# RuntimeWorld Expansion Gate

RuntimeWorld は時間と共に肥大化します。

---

危険例

```cpp
RuntimeWorld
{
    ...
    debugInfo
    telemetry
    uiState
    cache
}
```

---

結果

RuntimeWorld が

Authority Container

から

God Object

になります。

---

追加すべき事項

```cpp
RuntimeWorld Field Admission Rule
```

---

追加条件

```cpp
Authoritative Semantic
```

であること。

---

Diagnosticは不可。

---

# High-19

# Semantic Ownership Matrix

現在は

Authority Inventory

だけ。

---

不足。

各Authorityについて

```cpp
Writer
Reader
Publisher
Retirer
```

定義。

---

例

```cpp
OverlapSemantic

Writer:
Build

Reader:
Executor

Publisher:
PublicationCoordinator

Retirer:
RetireCoordinator
```

---

これが無いと責務が再拡散します。

---

# High-20

# Runtime Capability Model

非常に見落とされやすい。

---

今後

```cpp
oversampling
routing
multiband
midside
```

など追加される。

---

Capabilityが無いと

```cpp
if(flag)
```

地獄になります。

---

追加推奨

```cpp
RuntimeCapabilitySet
```

---

Worldが何をサポートするかを明示。

---

# High-21

# Semantic Feature Flag禁止

ISR Runtimeでは危険。

---

例

```cpp
enableNewOverlap
enableNewRouting
enableNewPublication
```

---

これが増えると

意味系統が二重化する。

---

追加契約

```cpp
Semantic Feature Flag Forbidden
```

---

許可されるのは

Diagnosticのみ。

---

# High-22

# RuntimeWorld Size Growth Monitoring

前回のMemory Budgetよりさらに重要。

---

監視

```cpp
sizeof(RuntimeWorld)
```

---

増加率。

---

例えば

```text
+20%
```

超過

↓

レビュー必須。

---

DSP系では重要。

---

# High-23

# Semantic Complexity Budget

将来の崩壊要因。

---

監視

```cpp
Authority Count
Semantic Count
Verifier Count
```

---

増加し続けたら

設計見直し。

---

---

# Medium-16

# RuntimeWorld Serialization Contract

今後

Replay

Evidence

Debug

で必要。

---

追加

```cpp
serialize(world)
deserialize(world)
```

定義。

---

同一性保証。

---

# Medium-17

# RuntimeWorld Diff Contract

Shadow Compare強化。

---

必要

```cpp
diff(worldA, worldB)
```

---

何が違うかを出せる。

---

Hashだけでは不足。

---

# Medium-18

# Semantic Tombstone Contract

削除済みAuthorityの管理。

---

例

```cpp
LegacyOverlapSemantic
```

削除。

---

しかし将来復活しがち。

---

追加

```cpp
Tombstone Registry
```

---

再導入時レビュー必須。

---

# Medium-19

# Runtime Evolution Log

長期保守向け。

---

各PRで

```cpp
Authority追加
Authority削除
Verifier追加
Verifier削除
```

記録。

---

アーキテクチャ履歴になる。

---

# Medium-20

# Governance Coverage Metrics

重要です。

---

現在

Verifier追加

だけ。

---

しかし

```text
Authority数
Verifier数
Coverage率
```

を計測していない。

---

結果

未監視領域が増える。

---

追加推奨

```cpp
GovernanceCoverage
```

---

# 実際に私なら追加する最後の重要項目

ここまで全部含めた上で、特に価値が高いのは次の5つです。

### 1. Authority Regression Gate

将来の再劣化防止。

---

### 2. RuntimeWorld Expansion Gate

God Object化防止。

---

### 3. Semantic Ownership Matrix

責務拡散防止。

---

### 4. Semantic Feature Flag禁止

意味系統の二重化防止。

---

### 5. Governance Coverage Metrics

Verifier漏れ検出。

---

この5項目は「今の問題を直す」ためではなく、

**Practical Stable ISR Bridge Runtime 完成後に、AI実装や将来機能追加によって再び Mutable Runtime 化していくことを防ぐための統治設計**

として追加価値があります。

ここまで含めると、設計対象は単なる Runtime 実装ではなく、

**「自己崩壊しにくい Runtime Governance System」**

のレベルになります。



ここまで来ると「漏れている技術項目」を探すより、

**Practical Stable ISR Bridge Runtime を完成させた後に、将来の変更・AI実装・機能追加によって再び破綻しないことを保証するための契約**

を探す段階になります。

その観点で見ると、まだ設計書に入っていない重要事項があります。

---

# Critical-17

# Semantic Transaction Contract

現在の設計は

```text
Build
↓
Publish
```

しか定義していません。

しかし RuntimeWorld の生成は実質的にトランザクションです。

---

危険例

```cpp
topology 完了
routing 完了
execution 失敗
```

↓

途中状態 world が存在

---

必要追加

```cpp
SemanticTransaction
```

概念。

---

状態

```cpp
Building
Validated
Committed
Published
```

のみ。

---

禁止

```cpp
PartiallyBuilt
PartiallyCommitted
```

の公開。

---

これは Partial Publication より一段上の契約です。

---

# Critical-18

# Publish Commit Point 定義不足

現在

```cpp
publish(world)
```

となっています。

しかし

「どの瞬間が commit か」

が未定義。

---

必要追加

```cpp
Publication Commit Point
```

定義。

---

例えば

```cpp
activeWorld.store(...)
```

成功時点。

---

Commit Point以降は

```cpp
rollback不可
mutation不可
```

を保証。

---

これが曖昧だと実装差異が発生します。

---

# Critical-19

# Authority Read Consistency Contract

Writer側だけでなく Reader側も定義が必要。

---

危険例

```cpp
generation = 10
routing = old
```

を読んでしまう。

---

必要追加

```cpp
Reader sees one world only
```

契約。

---

つまり

```cpp
read consistency
```

保証。

---

実質

Snapshot Consistencyです。

---

# Critical-20

# Retire Safety Contract

現在の設計は

```cpp
Retire Pressure
```

中心。

---

しかし重要なのは

```cpp
retire可能条件
```

です。

---

必要追加

```cpp
No Reader
No Executor Reference
No Pending Transition
```

を retire 前提条件にする。

---

これは UAF 防止の最後の砦です。

---

# High-24

# Semantic Invariant Registry

現在の verifier は個別。

---

不足。

全不変条件を一元管理する場所。

---

例

```cpp
Invariant:
GenerationMonotonic

Invariant:
SingleAuthority

Invariant:
ObserveSingleSource
```

---

Verifierは

Registryから生成。

---

設計と実装が乖離しにくくなる。

---

# High-25

# Runtime Contract Test Suite

現在は verifier 中心。

---

不足。

契約単位のテスト。

---

例

```cpp
PublicationContractTests
RetireContractTests
OverlapContractTests
```

---

将来の回帰検出に有効。

---

# High-26

# Semantic Change Impact Analysis

AI実装では特に重要。

---

例

```cpp
overlap semantic変更
```

↓

影響範囲

```cpp
publication
retire
executor
```

自動列挙。

---

追加推奨

```cpp
ImpactAnalysisManifest
```

---

# High-27

# Runtime Contract Coverage

前回の Governance Coverage の発展版。

---

測定対象

```cpp
Authority
Invariant
Verifier
Tests
Evidence
```

---

例えば

```text
Authority 15個
Verifier 12個
Coverage 80%
```

を可視化。

---

# High-28

# World Replacement Contract

非常に重要。

---

現在

```cpp
publish(newWorld)
```

のみ。

---

不足。

```cpp
oldWorld
↓
newWorld
```

置換時の契約。

---

定義すべきもの

```cpp
visibility
handover
retire start
```

---

Runtime の主機能です。

---

# Medium-21

# Runtime Dependency Graph

Authority間依存を明示。

---

例

```text
topology
 ↓
routing
 ↓
execution
 ↓
publication
```

---

循環依存禁止。

---

# Medium-22

# Semantic Blast Radius Budget

変更影響範囲を監視。

---

例えば

```cpp
routing変更
```

で

```cpp
20ファイル変更
```

必要なら危険。

---

設計劣化の兆候。

---

# Medium-23

# Runtime Contract Documentation Generator

実装から

```cpp
Authority
Invariant
Verifier
```

一覧を自動生成。

---

ドキュメント腐敗防止。

---

# Medium-24

# Retire Hazard Taxonomy

現在の Failure Taxonomy は Publish 側。

---

Retire側も必要。

---

例

```cpp
ReferenceLeak
EpochLeak
BacklogOverflow
RetireStarvation
```

---

原因分析が容易になる。

---

# Medium-25

# Runtime Governance Review Checklist

AI実装時向け。

---

PRごとに

```text
Authority増えたか

Verifier減ったか

Observe Path増えたか

Legacy増えたか

ProjectionがAuthority化したか
```

確認。

---

人間レビュー品質を上げる。

---

# 最後まで見落とされやすい重要項目 TOP5

ここまでの全議論を統合すると、まだ追加価値が高いのは次です。

### 1. Semantic Transaction Contract

RuntimeWorld生成をトランザクションとして定義。

---

### 2. Publication Commit Point

publishの確定点を厳密化。

---

### 3. Authority Read Consistency

Readerが常に一貫した world を観測する保証。

---

### 4. Retire Safety Contract

retire可能条件の形式化。

---

### 5. Semantic Invariant Registry

全契約を一元管理。

---

特に **Semantic Transaction Contract** と **Authority Read Consistency Contract** は、ここまでの議論でまだ直接触れられていない領域ですが、ISR Runtime の実装を最終的に安定化させるうえではかなり重要です。

実際には、ここまで列挙した内容をすべて実装する必要はありません。しかし「Practical Stable ISR Bridge Runtime 完全移行設計書」として完成度を高めるなら、これ以上追加する価値がある項目は主に **トランザクション性・一貫性・契約管理・回帰防止統治** の領域になります。




ここまでで「Practical Stable ISR Bridge Runtime の完成条件」に近いところまでは網羅できています。

そのため、これ以降は「まだ漏れている技術要件」ではなく、

**実際に ConvoPeq のようなリアルタイム DSP アプリで数年単位の保守運用を考えた場合に、設計書へ入れておくと後から効いてくる事項**

になります。

ただし、以下の中には実際にかなり重要なものがあります。

---

# Critical-21

# RT Safety Boundary Contract

これは意外と設計書にありません。

Practical Stable ISR Runtime の根本目的は

```text
RT thread を守る
```

です。

しかし現在の設計は

```text
Authority
Publication
Retire
```

中心です。

---

必要追加

各コンポーネントに

```cpp
RTSafe
NonRTOnly
Mixed
```

分類を持たせる。

---

禁止

```cpp
PublicationVerifier
↓
AudioThreadから呼ぶ
```

---

Verifier追加

```cpp
RTBoundaryVerifier
```

---

これは非常に価値があります。

---

# Critical-22

# Memory Ordering Contract

現行設計でほぼ触れられていません。

ISR Runtime の publish は本質的に

```cpp
atomic store
```

です。

---

必要追加

明示的定義

```cpp
publish
=
release

observe
=
acquire
```

---

禁止

```cpp
relaxed
```

利用。

---

これを定義しないと

CPU依存バグが残ります。

---

# Critical-23

# ABA Hazard Contract

Retire周りで重要。

---

危険例

```text
world A
↓
retire
↓
memory reuse
↓
new world A
```

---

参照側が区別できない。

---

必要追加

```cpp
worldId
generation
epoch
```

による識別。

---

Verifier

```cpp
ABAHazardVerifier
```

---

これは長期運用で発生しやすい。

---

# Critical-24

# Ownership Transfer Contract

Publish時に

```cpp
builder
↓
world
↓
publication
```

が発生する。

---

しかし

所有権移譲が曖昧。

---

必要追加

```cpp
ownership transfer matrix
```

---

例

```cpp
Build
→ Publication

Publication
→ Retire

Retire
→ Destroy
```

---

二重所有禁止。

---

# High-29

# Executor Local Purity Contract

前回

ExecutorLocal

分類を導入した。

---

しかし不足。

---

禁止

```cpp
ExecutorLocal
↓
authority write
```

---

つまり

ExecutorLocal は

```cpp
read-only semantic
```

でなければならない。

---

# High-30

# Projection Drift Contract

Projectionを許可するなら必要。

---

監視

```cpp
World
Projection
```

差分。

---

例えば

```cpp
routing projection
```

が古くなる。

---

Drift検出追加。

---

# High-31

# Runtime Destruction Contract

Retire後の最後の段階。

---

現在

```cpp
Retire
```

まではある。

---

不足

```cpp
Destroy
```

条件。

---

定義

```cpp
Retired
NoReader
NoReference
```

のみ破棄可能。

---

# High-32

# Semantic Dependency Freeze

実装中に起こる問題。

---

例

```cpp
routing
↓
publication
```

依存。

---

後で

```cpp
publication
↓
routing
```

追加。

---

循環依存化。

---

必要追加

```cpp
Dependency Freeze Manifest
```

---

# High-33

# Runtime State Explosion Guard

非常に実践的。

---

RuntimeWorldへ何でも入れ始める。

---

結果

```cpp
TopologyState
RoutingState
ExecutionState
...
```

増殖。

---

監視

```cpp
Semantic Count
```

---

閾値超過レビュー。

---

# Medium-26

# Evidence Sampling Policy

Evidence Exporterがあるなら必要。

---

全記録は危険。

---

定義

```cpp
critical
always

normal
sampled

debug
opt-in
```

---

# Medium-27

# Verifier Runtime Cost Budget

Verifier増殖問題。

---

監視

```cpp
Verifier Cost
```

---

PRで重くなりすぎるのを防ぐ。

---

# Medium-28

# Test Fixture Canonicalization

テスト品質向上。

---

共通 fixture

```cpp
minimal world
large world
overlap world
retire pressure world
```

定義。

---

# Medium-29

# Semantic Naming Contract

見落とされやすい。

---

禁止

```cpp
state
manager
helper
context
```

曖昧名。

---

許可

```cpp
OverlapSemantic
PublicationSemantic
RetireSemantic
```

---

Authority可視化に効く。

---

# Medium-30

# Runtime Review Checklist Automation

最後の運用系。

---

PRで自動確認。

```text
Authority増加

Verifier減少

Observe追加

Legacy追加

Projection追加
```

---

AI実装で特に有効。

---

# ここまで含めた場合の「本当に重要な残件」

ここまでの全議論を統合して、まだ設計書に無いが価値が高いものを5個だけ選ぶなら、

### 1. Memory Ordering Contract

実装バグ防止。

---

### 2. Ownership Transfer Contract

UAF防止。

---

### 3. ABA Hazard Contract

世代管理防止。

---

### 4. RT Safety Boundary Contract

DSP保護。

---

### 5. Executor Local Purity Contract

Authority逆流防止。

---

この5項目は、これまで主に議論してきた「意味論（Semantic Governance）」ではなく、

**実際のC++リアルタイム実装で事故を起こしやすい並行性・所有権・メモリモデル・RT安全性の契約**

です。

特に ConvoPeq の現状コードベースは ISR Runtime へ移行中であり、今後 AI に実装させるのであれば、

* Memory Ordering
* Ownership Transfer
* ABA Hazard
* RT Boundary

の4項目は設計書へ明文化しておく価値が高いと考えます。




ここまででかなり広範囲に網羅されています。

そのため、これ以上は「新しい機能要求」ではなく、

**ConvoPeq の現実のコードベースで AI が実装したときに発生しやすい抜け道・設計崩壊経路・統治不能化経路**

を発掘するフェーズになります。

その観点で見ると、まだいくつか重要な事項があります。

---

# Critical-25

# Semantic Alias Contract

非常に見落とされやすいです。

---

設計上は

```cpp
world.routing
```

だけが Authority。

しかし実装では

```cpp
cachedRouting
currentRouting
activeRouting
resolvedRouting
```

のような別名が生まれます。

---

結果

Authority Inventory上は1個でも

実質Authorityは4個になります。

---

追加すべき契約

```text
Semantic Alias Prohibited
```

---

許可されるのは

```cpp
const reference
read-only projection
```

のみ。

---

Verifier

```cpp
SemanticAliasVerifier
```

---

これは Hidden Authority Detector と似ていますが別物です。

Hidden Authority は「新しい意味源」。

Alias は「既存意味源の複製」。

---

# Critical-26

# Multi-Writer Prohibition

現在の設計は Authority を定義しています。

しかし

Writer数を制限していません。

---

危険例

```cpp
BuildCoordinator
```

も

```cpp
PublicationCoordinator
```

も

```cpp
routing
```

を書ける。

---

結果

責務競合。

---

追加契約

```text
One Authority
One Writer
```

---

複数Writer禁止。

---

Ownership Matrix と似ていますが、

こちらは強制契約です。

---

# Critical-27

# Semantic Rehydration Contract

将来発生します。

---

危険例

```cpp
world
↓
projection
↓
world再生成
```

---

つまり

```cpp
projection → authority
```

逆流。

---

追加契約

```text
Projection never rehydrates Authority
```

---

これは RuntimeWorld 設計では非常に重要です。

---

# Critical-28

# Semantic Clock Contract

Generationだけでは不足。

---

現実には

```cpp
publication
retire
visibility
```

が別速度で進む。

---

必要なのは

```cpp
semantic clocks
```

定義。

---

例

```cpp
generationClock
publicationClock
retireClock
```

---

単調増加保証。

---

長期運用で診断しやすくなる。

---

# High-34

# Contract Ownership Contract

意外な盲点。

---

現在

```cpp
Verifier
Invariant
Authority
```

はある。

---

しかし

誰が管理するか不明。

---

追加

各契約に

```cpp
owner
reviewer
```

を持つ。

---

AI実装時に有効。

---

# High-35

# Semantic Dead Code Detection

Authority削除後に残骸が残る。

---

例

```cpp
oldOverlapSemantic
```

参照なし。

---

しかし

将来復活する。

---

追加

```cpp
SemanticDeadCodeVerifier
```

---

# High-36

# RuntimeWorld Minimality Contract

非常に重要。

---

RuntimeWorldは

```cpp
必要最小限
```

であるべき。

---

追加条件

新Field追加時

```text
Authorityか？
```

を証明。

---

証明できなければ追加禁止。

---

Expansion Gateより強い契約。

---

# High-37

# Semantic Layer Separation

現在の設計は

```cpp
Topology
Routing
Execution
Publication
Retire
```

があります。

---

しかし層境界が曖昧。

---

追加

依存方向固定。

```text
Topology
 ↓
Routing
 ↓
Execution
 ↓
Publication
 ↓
Retire
```

---

逆方向依存禁止。

---

# High-38

# RuntimeWorld Replacement Atomicity

前回

Commit Point

は指摘しました。

---

さらに必要。

---

禁止

```cpp
old world retire
↓
new world publish
```

順序。

---

必須

```cpp
new world visible
↓
old world retire
```

---

Visibility Gap禁止。

---

# Medium-31

# Evidence Integrity Contract

Evidence Exporterがあるなら必要。

---

記録の途中欠落を禁止。

---

例

```cpp
publish event
```

だけあり

```cpp
retire event
```

が無い。

---

整合性検査を追加。

---

# Medium-32

# Verifier Dependency Contract

Verifier同士が依存し始める。

---

危険。

---

追加

```text
Verifier is side-effect free
```

---

Verifier間循環禁止。

---

# Medium-33

# Runtime Governance Bootstrap

新規参加AI向け。

---

現在の設計は巨大。

---

追加

```text
Authority
Invariant
Verifier
```

最小セットを自動生成。

---

AIの誤実装を減らす。

---

# Medium-34

# Semantic Saturation Detection

長期保守で重要。

---

監視

```text
Authority数
Verifier数
Contract数
```

---

増えすぎたら

設計再整理。

---

# Medium-35

# Runtime Archaeology Support

数年後向け。

---

各Authorityに

```cpp
introduced
modified
deprecated
removed
```

履歴保持。

---

設計意図の喪失を防ぐ。

---

# 現時点で最後まで残る重要項目 TOP5

ここまでの全指摘を統合し、まだ設計書に無い中で価値が高いものを挙げるなら、

1. **Semantic Alias Contract**

   * Authority複製の禁止

2. **Multi-Writer Prohibition**

   * 1 Authority = 1 Writer

3. **Semantic Rehydration Contract**

   * Projection → Authority 逆流禁止

4. **Semantic Layer Separation**

   * Topology→Routing→Execution→Publication→Retire の一方向依存

5. **RuntimeWorld Replacement Atomicity**

   * 新World可視化後に旧WorldをRetire

特に ConvoPeq の ISR Runtime 移行で実害が出やすいのは、

* Semantic Alias
* Multi-Writer
* Rehydration
* Replacement Atomicity

です。

これらはコードレビューで見落とされやすく、実装後数か月してから「なぜか Authority が増殖している」「Projection が意味源になっている」「古い World が復活した」といった形で問題化しやすい領域です。

正直な評価としては、ここまで列挙した内容のうち実装必須なのは全体の半分程度ですが、**Semantic Alias・Multi-Writer・Rehydration・Replacement Atomicity** の4項目は、Practical Stable ISR Bridge Runtime を長期維持するための契約として追加する価値があります。



ここまで来ると、一般的な ISR Runtime 設計レビューではまず出てこないレベルです。

ただし、ConvoPeq のように

* RuntimeWorld
* Publication
* Retire
* Overlap
* Semantic Governance
* AI実装主体

まで含むシステムとして考えると、まだいくつか本質的な抜けがあります。

これらは「機能追加」ではなく、

**Runtimeの意味論そのものが時間経過で腐敗する経路を封じる契約**

です。

---

# Critical-29

# Semantic Equivalence Contract

現在

```cpp
semanticHash
```

があります。

しかし

```cpp
hash一致
≠
意味一致
```

です。

---

例

```cpp
routing A
routing B
```

が

偶然同じhash。

---

逆もあります。

```cpp
hash違う
```

が

意味は同じ。

---

必要追加

```text
SemanticEquivalenceContract
```

---

定義

```cpp
Equivalent
Compatible
Different
```

の3段階。

---

Shadow Compareは

Hashではなく

Semantic Equivalenceで判定。

---

これは長期運用で重要です。

---

# Critical-30

# Semantic Canonical Form Contract

非常に重要。

---

同じ意味を

複数表現で保持し始める。

---

例

```cpp
fadeLengthSamples
```

と

```cpp
fadeLengthMs
```

を両方Authority化。

---

結果

同期崩壊。

---

追加契約

```text
Canonical Semantic Representation
```

---

各Semanticは

唯一表現のみ許可。

---

Projectionで変換。

---

# Critical-31

# Derived Semantic Non-Persistence Contract

Derivedと分類しても

保存し始める危険があります。

---

例

```cpp
routingCache
```

を

Worldに保存。

---

すると

DerivedがAuthority化。

---

追加契約

```text
Derived Semantic Never Persisted
```

---

Derivedは

毎回再計算可能であること。

---

# Critical-32

# Semantic Resurrection Prohibition

削除済みAuthority問題。

---

例

```cpp
OldOverlapSemantic
```

削除。

---

数か月後

AIが

```cpp
似た名前
```

で復活。

---

追加契約

```text
Semantic Resurrection Forbidden
```

---

Tombstone Registryと連携。

---

# High-39

# RuntimeWorld Normalization Contract

Build結果が

実装順序依存になる問題。

---

例

```cpp
node追加順
```

で

semanticHash変化。

---

必要

```text
Normalization
```

---

Publish前に

正規化。

---

Hash安定化。

---

# High-40

# Semantic Determinism Scope Contract

前回

DeterministicBuildVerifier

を提案しました。

---

さらに必要。

---

どこまで決定論か定義。

---

例

```cpp
Topology
Routing
Publication
```

は決定論。

---

しかし

```cpp
timestamp
```

は非決定論。

---

境界定義が必要。

---

# High-41

# RuntimeWorld Referential Transparency

かなり重要。

---

同じ入力

↓

同じWorld

を保証。

---

禁止

```cpp
global state
random
singleton mutable state
```

依存。

---

Build Isolationより一段強い。

---

# High-42

# Semantic Entropy Monitor

長期運用向け。

---

監視

```text
Authority増加率

Projection増加率

Verifier増加率
```

---

増え続けるなら

設計劣化。

---

# High-43

# Governance Bypass Detector

AI実装で重要。

---

危険例

```cpp
// verifier bypass
```

```cpp
if(debug)
```

```cpp
if(test)
```

---

で契約を回避。

---

追加

```text
GovernanceBypassVerifier
```

---

# Medium-36

# Semantic Compression Contract

意味論の重複管理。

---

例

```cpp
RoutingSemanticV1
RoutingSemanticV2
```

共存。

---

長期的に圧縮。

---

# Medium-37

# RuntimeWorld Structural Hash

現在

Semantic Hash中心。

---

追加

```cpp
StructuralHash
```

---

Topology構造専用。

---

原因解析しやすい。

---

# Medium-38

# Semantic Conflict Detector

複数Semantic間の矛盾。

---

例

```cpp
Overlap disabled
```

なのに

```cpp
Fade duration > 0
```

---

矛盾検出。

---

# Medium-39

# Authority Exhaustiveness Contract

非常に重要。

---

すべての

```cpp
if(...)
switch(...)
branch
```

の意味源が

Authority Inventoryに存在するか検証。

---

Hidden Authority Detectorより強い。

---

# Medium-40

# Runtime Governance Survivability

最終段階。

---

将来

```text
Verifier削除

Owner退職

AI変更
```

が起きても

維持できるか。

---

監視対象

```text
Documentation

Ownership

Coverage

Automation
```

---

# ここまで含めた場合に、本当に最後まで残る重要事項

私はここまでの全レビューを統合すると、

まだ追加価値が高いのは次の5項目です。

### 1. Semantic Canonical Form Contract

同じ意味の複数表現禁止。

---

### 2. Derived Semantic Non-Persistence Contract

DerivedのAuthority化防止。

---

### 3. RuntimeWorld Normalization Contract

Hash・比較安定化。

---

### 4. Governance Bypass Detector

AI実装による契約回避防止。

---

### 5. Authority Exhaustiveness Contract

全branch sourceの完全棚卸し。

---

特に **Authority Exhaustiveness Contract** は、これまでの議論で最も後になって出てくるにもかかわらず、実は非常に強力です。

なぜなら Practical Stable ISR Bridge Runtime の本質は、

```text
全ての意味決定源を把握できている
```

ことだからです。

もし設計書に追加するなら、

**Authority Exhaustiveness Contract + Semantic Canonical Form Contract**

の2つは、最終版に入れる価値があります。
