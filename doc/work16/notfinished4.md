最新版 ConvoPeq.md（2026-06-03版）を確認した限りでは、Practical Stable ISR Bridge Runtime の主要骨格はかなり実装されています。しかし、計画書 v2.3 / v3.1 の「完全移行条件」と照合すると、まだ未達成または達成を証明できない事項が複数残っています。

重要なのは、

> 「未達成」
>
> と
>
> 「達成を証明できない」

を分けて扱うことです。

前者は実際にコード上に残存している問題です。
後者は verifier や fail-closed 契約が不足しており、将来の回帰を防げない問題です。

---

# ① RuntimeWorld Self-contained 未達成

## 1. 未達成内容

計画書の完成条件は

> RuntimeWorld が唯一の authoritative semantic container

です。

しかし RuntimePublishWorld には依然として

```cpp
engine
graph
runtimeVersion
transitionId
```

が格納されています。 

さらに build 時に

```cpp
worldOwner->engine = engineState;
worldOwner->graph = graphState;
worldOwner->runtimeVersion = nextGraphGeneration;
worldOwner->transitionId = ...
```

が設定されています。 

これは RuntimeSemanticSchema と並行して legacy semantic が残っている状態です。

---

## 2. 該当箇所

RuntimeBuilder::buildRuntimePublishWorld()

```cpp
worldOwner->engine = engineState;
worldOwner->graph = graphState;
worldOwner->runtimeVersion = nextGraphGeneration;
worldOwner->transitionId = ...
```



---

## 3. あるべき姿

```cpp
RuntimeWorld
 └ RuntimeSemanticSchema
      ├ GenerationSemantic
      ├ TopologySemantic
      ├ RoutingSemantic
      ├ ExecutionSemantic
      ├ PublicationSemantic
      ├ OverlapSemantic
      └ RetireSemantic
```

以外は authority を持たない。

---

## 4. 改修方法

engine / graph を

```cpp
DerivedProjection
```

へ格下げ。

branch decision に利用禁止。

Verifier を追加。

```cpp
RuntimeWorldAuthorityProjectionTests
```

を強化して

```cpp
engine
graph
runtimeVersion
transitionId
```

参照を禁止する。

---

# ② RuntimeGraph Authority Collapse 未完

## 1. 未達成内容

計画書では

```text
RuntimeGraph = projection
```

が最終形です。

しかし build 時に

```cpp
graphState = makeRuntimeGraphState(...)
worldOwner->graph = graphState;
```

が依然残っています。 

---

## 2. 該当箇所

```cpp
computation.graphState
```

生成。 

```cpp
worldOwner->graph = graphState;
```

格納。 

---

## 3. あるべき姿

```cpp
TopologySemantic
```

のみ authority。

RuntimeGraph は UI / debug 用投影。

---

## 4. 改修方法

RuntimeGraph に

```cpp
[[deprecated("Projection only")]]
```

付与。

Verifier 追加。

```cpp
RuntimeGraphAuthorityVerifier
```

で

```cpp
decision
branch
publish
retire
```

への利用禁止。

---

# ③ Snapshot Non-Authority が未証明

## 1. 未達成内容

計画では

```text
snapshot = projection artifact only
```

です。

しかし現行ソースには

```cpp
GlobalSnapshot
SnapshotFactory
SnapshotAssembler
SnapshotCoordinator
```

が残っています。 

---

## 2. 該当箇所

ファイル群

```cpp
core/GlobalSnapshot.*
core/SnapshotFactory.*
core/SnapshotAssembler.*
core/SnapshotCoordinator.*
```



---

## 3. あるべき姿

Snapshot は

```cpp
Diagnostic
Telemetry
UI projection
```

のみ。

---

## 4. 改修方法

Authority inventory に

```cpp
Snapshot = NonAuthority
```

明記。

Verifier 追加。

```cpp
snapshot -> branch
snapshot -> publish
snapshot -> retire
```

検出で fail。

---

# ④ Observe Collapse 完了を証明できない

## 1. 未達成内容

計画書の完成条件

```text
Audio Thread observe = RuntimeWorld only
```

しかし ConvoPeq.md からは

AudioThread が本当に

```cpp
runtimeStore.observe()
```

だけを使うのか証明できません。

---

## 2. 該当箇所

Coordinator 側は

```cpp
consumePublishedWorld()
{
    return store.observe();
}
```

となっています。 

しかし AudioThread 側利用箇所の完全禁止証拠は確認できません。

---

## 3. あるべき姿

```cpp
AudioThread
  ↓
RuntimeStore.observe()
  ↓
RuntimeWorld
```

のみ。

---

## 4. 改修方法

Verifier を追加。

```cpp
NonAuthoritativeObserveVerifier
```

で

```cpp
activeRuntimeDSPSlot
fadingRuntimeDSPSlot
snapshot
runtimeGraph
```

参照を禁止。

---

# ⑤ Legacy Publication API 完全除去未達

## 1. 未達成内容

Coordinator にまだ

```cpp
publishState(...)
```

が存在します。 

コメントでは

```cpp
DEPRECATED
```

となっています。

しかし計画書の最終条件は

```cpp
publish(RuntimeWorld*)
only
```

です。

---

## 2. 該当箇所

```cpp
void publishState(...)
```



---

## 3. あるべき姿

```cpp
publishWorld(world)
```

のみ。

---

## 4. 改修方法

Sprint 完了後に

```cpp
publishState()
```

削除。

CI に

```cpp
legacy publication api zero-call
```

を追加。

---

# ⑥ Initial Atomic Fallback 残存

## 1. 未達成内容

現在

```cpp
allowInitialAtomicFallback
```

があります。 

つまり RuntimeWorld 不在時に

atomic 状態から意味を取得可能です。

これは計画書の

```text
observe source = RuntimeWorld only
```

と矛盾します。

---

## 2. 該当箇所

```cpp
allowInitialAtomicFallback
```

```cpp
allowTransitionFallback
allowRoutingAutomationFallback
allowAdaptiveBankIndexFallback
allowEqCoeffHashFallback
```



---

## 3. あるべき姿

RuntimeWorld が無いなら

```cpp
publish禁止
```

または

```cpp
bootstrap world
```

を使う。

---

## 4. 改修方法

Bootstrap RuntimeWorld を導入。

Atomic fallback を段階廃止。

---

# ⑦ Authority Inventory 完全実装不足

## 1. 未達成内容

計画では

```text
全 runtime field を

Authoritative
Derived
Diagnostic
ExecutorLocal
LegacyTemporary

へ分類
```

が必須です。

しかし ConvoPeq.md からは

全フィールド分類完了を確認できません。

---

## 2. 該当箇所

確認できるのは

```cpp
RuntimeFieldDescriptor
```

のみ。 

---

## 3. あるべき姿

全 Runtime フィールドに分類が存在。

---

## 4. 改修方法

機械可読 inventory を作成。

```cpp
authority_inventory.json
```

などで管理。

---

# ⑧ Fail-Closed Governance 未完成

## 1. 未達成内容

計画では

```text
verifier fail = build fail
```

が必須です。

しかし

以下の verifier の存在が確認できません。

```text
authority duplication verifier
observe path expansion verifier
non-authoritative observe detector
legacy expansion verifier
```



---

## 2. 該当箇所

テストは存在。 

しかし CI Gate は確認不能。

---

## 3. あるべき姿

PR

↓

Verifier

↓

Fail

↓

Build Stop

---

## 4. 改修方法

GitHub Actions に

```text
AuthorityVerifier
ObserveVerifier
SnapshotVerifier
PublicationVerifier
```

を mandatory 化。

---

# 優先順位

実運用リスク順に並べると

| 優先度      | 項目                             |
| -------- | ------------------------------ |
| Critical | Initial Atomic Fallback 残存     |
| Critical | Observe Collapse 未証明           |
| High     | RuntimeGraph Authority 残存      |
| High     | RuntimeWorld Self-contained 未完 |
| High     | Legacy Publication API 残存      |
| Medium   | Snapshot Non-Authority 未証明     |
| Medium   | Authority Inventory 不完全        |
| Medium   | Fail-Closed Governance 未完成     |

特に、

**Initial Atomic Fallback 残存**
**Observe Collapse 未証明**
**RuntimeGraph Authority 残存**

の3つは、Practical Stable ISR Bridge Runtime の「Single Authoritative Observable Runtime」という最終目標に直接抵触しており、現時点での最大の残課題と考えられます。



前回提示した

* RuntimeWorld Self-contained 未完
* RuntimeGraph Authority Collapse 未完
* Snapshot Non-Authority 未証明
* Observe Collapse 未証明
* Legacy Publication API 残存
* Initial Atomic Fallback 残存
* Authority Inventory 不完全
* Fail-Closed Governance 未完成

以外で、ConvoPeq.md から読み取れる追加の未達成事項を挙げます。

なお、ConvoPeq.md は巨大であり、現在参照できる断片ベースの評価です。そのため「未達成確定」と「未達成の可能性が高い」を区別しています。

---

# ⑨ Generation Identity Singularization 未完

## 1. 未達成内容

Practical Stable ISR Runtime の完成条件では

```text
Generation = single authoritative identity
```

です。

しかし現在も

```cpp
generation
worldId
publicationSequence
runtimeVersion
transitionId
```

が並存しています。

特に v3.1 が問題視していた

```text
generation
runtimeVersion
transitionId
```

の重複問題が完全解消された証拠がありません。 

---

## 2. 該当箇所

```cpp
struct RuntimePublicationIdentity
{
    uint64_t generation;
    uint64_t worldId;
    PublicationSequenceId publicationSequence;
};
```



さらに前回確認済み

```cpp
runtimeVersion
transitionId
```

も RuntimePublishWorld 側に残存。

---

## 3. あるべき姿

```cpp
GenerationSemantic.activationEpoch
GenerationSemantic.generationId
PublicationSemantic.sequence
```

のみ authority。

その他は diagnostics。

---

## 4. 改修方法

Identity Inventory を追加。

```cpp
IdentityAuthorityVerifier
```

を導入。

branch 条件に

```cpp
runtimeVersion
transitionId
worldId
```

が使われたら fail。

---

# ⑩ Publication Sequence Governance 未完成

## 1. 未達成内容

計画書では

```text
seq(n+1) > seq(n)
duplicate reject
out-of-order reject
```

が必須です。 

しかし現在見えるのは

```cpp
publicationSequenceCounter_
```

による採番だけです。 

---

## 2. 該当箇所

```cpp
identity.publicationSequence =
    fetchAddAtomic(...)
```



---

## 3. あるべき姿

```cpp
validateMonotonicSequence()
```

が publication 境界で必須。

---

## 4. 改修方法

Coordinator publish 時に

```cpp
lastCommittedSequence
```

との比較を強制。

逆転 publish を reject。

---

# ⑪ Mixed Semantic Source 禁止が未完成

## 1. 未達成内容

Audio Thread 側で

```cpp
runtimeWorld
```

以外に

```cpp
runtimeGraphRevision
```

を取得しています。 

これは runtime meaning source が複数存在する可能性を示します。

---

## 2. 該当箇所

```cpp
graphRevision =
    consumeAtomic(runtimeGraphRevision)
```



---

## 3. あるべき姿

Audio Thread が取得する semantic source は

```cpp
RuntimeWorld
```

のみ。

---

## 4. 改修方法

graphRevision を

```cpp
DiagnosticOnly
```

へ降格。

Audio callback での利用箇所を検査。

---

# ⑫ Executor-local Leakage 完全解消未確認

## 1. 未達成内容

v3.1 では

```text
transition
crossfade
executor detail
```

を executor-local に閉じ込めることが要求されています。 

しかし Audio Callback 内に

```cpp
currentFade_
```

が依然見えています。

---

## 2. 該当箇所

```cpp
makeRTExecutionFrame(
    ...
    currentFade_,
    ...
)
```



---

## 3. あるべき姿

crossfade 情報は

```cpp
ExecutionSemantic
```

として RuntimeWorld から導出。

Executor 内部状態は観測不可。

---

## 4. 改修方法

currentFade_ を

```cpp
ExecutorLocal
```

へ明示分類。

world 外へ漏れないことを verifier 化。

---

# ⑬ Runtime Topology Authority Split 未解消

## 1. 未達成内容

v2.3 が追加した重要課題です。 

```text
Runtime Topology Authority Split
```

---

## 2. 該当箇所

```cpp
RoutingSemantic.processingOrder
```

テストが存在。 

しかし

```cpp
RuntimeGraph
graphState
processingOrder
```

の authority 一元化が証明されていません。

---

## 3. あるべき姿

Topology authority は

```cpp
TopologySemantic
RoutingSemantic
```

のみ。

---

## 4. 改修方法

```cpp
RuntimeGraphAuthorityMismatch
```

検査を CI 強制。

---

# ⑭ Retire Ordering Contract 未完成

## 1. 未達成内容

Retire は

```text
retire single source
retire ordering stability
```

が必要です。

しかし現在

```cpp
DeferredRetireFallbackQueue
DeletionQueue
SnapshotRetireManager
```

が共存しています。 

複数 retire 経路の可能性があります。

---

## 2. 該当箇所

```text
DeferredRetireFallbackQueue
DeletionQueue
SnapshotRetireManager
```



---

## 3. あるべき姿

Retire authority は

```cpp
RuntimePublicationCoordinator
```

のみ。

---

## 4. 改修方法

Retire flow inventory 作成。

Coordinator 経由以外を fail。

---

# ⑮ Rebuild Admission Authority 未収束

## 1. 未達成内容

現在も

```cpp
queuedGeneration
committedGeneration
pendingIRChange
isLoadingIR
isIRFinalized
```

など複数状態を見て rebuild を決めています。 

これは rebuild authority が分散している可能性があります。

---

## 2. 該当箇所

```cpp
shouldDeferRebuild =
    outstandingRebuild
    || isLoadingIR()
    || pendingIRChange
    || !isIRFinalized()
```



---

## 3. あるべき姿

Rebuild admission は

```cpp
RebuildSemantic
```

または RuntimeWorld semantic。

---

## 4. 改修方法

AdmissionPolicy を semantic schema 化。

---

# ⑯ Snapshot Command Path 残存

## 1. 未達成内容

Snapshot が依然更新トリガとして使われています。

---

## 2. 該当箇所

```cpp
enqueueSnapshotCommand();
requestSnapshotForNoiseShaper();
```



---

## 3. あるべき姿

Snapshot は

```text
projection only
```

---

## 4. 改修方法

UI変更

↓

RebuildIntent

↓

RuntimeWorld Publish

↓

Snapshot再生成

へ方向を固定。

---

# ⑰ Soak Governance 未実装の可能性

## 1. 未達成内容

v3.1 は

```text
IR reload storm
automation storm
suspend/resume storm
4h soak
```

を要求しています。 

しかしテスト一覧に

```cpp
Soak
LongRun
Storm
```

系が見当たりません。 

---

## 2. 該当箇所

テスト一覧。

---

## 3. あるべき姿

Nightly に

```text
4h soak suite
```

を持つ。

---

## 4. 改修方法

CI レイヤを

* PR
* Nightly
* Release

へ分離。

---

# 重要度順

今回追加分だけで並べると、

| 優先度      | 項目                                     |
| -------- | -------------------------------------- |
| Critical | Generation Identity Singularization 未完 |
| Critical | Publication Sequence Governance 未完成    |
| Critical | Retire Ordering Contract 未完成           |
| High     | Mixed Semantic Source 残存               |
| High     | Runtime Topology Authority Split       |
| High     | Rebuild Admission Authority 分散         |
| Medium   | Executor-local Leakage                 |
| Medium   | Snapshot Command Path 残存               |
| Medium   | Soak Governance 未実装                    |

特に **⑨ Generation Identity**, **⑩ Publication Sequence**, **⑭ Retire Ordering** は、前回挙げた Observe/Authority 系の問題とは別系統であり、Practical Stable ISR Bridge Runtime の「generation・publication・retire の単一契約化」という完成条件そのものに直結するため、残課題として非常に重要です。
