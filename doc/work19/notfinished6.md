最新版 ConvoPeq.md を基準に、**Practical Stable ISR Bridge Runtime の本来目標（RuntimeWorld Authority / Publication Governance / Fail-Closed / DSP Lifetime Separation / AudioEngine非権威化）**とのギャップを整理すると、未達成事項は大きく分けて10項目です。

以下は優先度順です。

---

# ① Legacy Commit Path がまだ生存している

## 1. 未達成内容

本来は

```text
Rebuild
 ↓
Admission
 ↓
Executor
 ↓
Transition
 ↓
Retire
```

で完結するべきです。

しかし現在は

```text
processPendingCommit()
 ↓
applyRuntimeCommitFromIntent()
```

という旧モノリスが依然として主経路です。 

---

## 2. 該当箇所

### AudioEngine::processPendingCommit()

```cpp
applyRuntimeCommitFromIntent(dsp, gen, snap);
```



### AudioEngine::applyRuntimeCommitFromIntent()

550行級モノリス

```cpp
void AudioEngine::applyRuntimeCommitFromIntent(...)
```



---

## 3. あるべき姿

```text
AudioEngine
 ↓
Coordinator.submitPublishRequest()
 ↓
PublicationAdmission
 ↓
PublicationExecutor
 ↓
DSPTransition
```

---

## 4. 改修方法

* applyRuntimeCommitFromIntent() 完全削除
* processPendingCommit() 削除
* Coordinator.submitPublishRequest() に一本化

---

# ② Coordinator が Executor Authority になっていない

## 1. 未達成内容

Coordinator が存在するものの、

実際には

```cpp
coordinator.publishWorld(...)
```

の呼び出し先として使われているだけです。

Executor権威になっていません。

---

## 2. 該当箇所

```cpp
publishSmoothTransitionState
```

↓

```cpp
coordinator.publishWorld(...)
```



---

## 3. あるべき姿

```text
Coordinator
 ├ Admission
 ├ Executor
 ├ Transition
 └ Retire
```

---

## 4. 改修方法

Coordinator内部に

```cpp
submitPublishRequest()
```

を置き

publishWorld()を外部公開しない。

---

# ③ Crossfade Decision が完全に RuntimeWorld 化されていない

## 1. 未達成内容

evaluateFromWorlds() がある一方で

DSPCoreベースAPIが残っています。

---

## 2. 該当箇所

```cpp
evaluateOnly(
    AudioEngine&,
    DSPCore*,
    DSPCore*)
```

```cpp
evaluateAndRegister(
    AudioEngine&,
    DSPCore*,
    DSPCore*)
```



---

## 3. あるべき姿

```cpp
evaluateFromWorlds(
    oldWorld,
    newWorld)
```

のみ。

---

## 4. 改修方法

DSPCore API削除。

Crossfade判定入力を

```cpp
RuntimePublishWorld::dspProjection
```

へ統一。

---

# ④ DSPCore が Semantic Authority に残っている

## 1. 未達成内容

判定系がまだDSPCoreに依存しています。

---

## 2. 該当箇所

CrossfadeAuthority内部

```cpp
computeDecision(
    const DSPCore* oldDSP,
    const DSPCore* newDSP)
```



---

## 3. あるべき姿

```cpp
RuntimeWorld
↓
dspProjection
```

のみ。

DSPCoreはExecution Object。

---

## 4. 改修方法

RuntimeBuilderで

```cpp
dspProjection
```

を完全生成し、

以後DSPCore参照禁止。

---

# ⑤ DSPLifetimeManager が単なるラッパー

## 1. 未達成内容

現在は

```cpp
engine_.setActiveRuntimeDSP()
engine_.retireDSP()
engine_.getActiveRuntimeDSP()
```

呼び出ししかしていません。 

---

## 2. 該当箇所

```cpp
class DSPLifetimeManager
```



---

## 3. あるべき姿

```text
DSPLifetimeManager
 ↓
HandleRuntime
 ↓
EpochDomain
```

---

## 4. 改修方法

AudioEngine依存を排除。

Lifetime Authority を移譲。

---

# ⑥ AudioEngine が Runtime Authority を保持している

## 1. 未達成内容

activeDSP管理権限が依然AudioEngine側です。

---

## 2. 該当箇所

```cpp
getActiveRuntimeDSP()
setActiveRuntimeDSP()
exchangeFadingRuntimeDSP()
```



---

## 3. あるべき姿

```text
RuntimeStore
 ↓
RuntimeWorld
```

が唯一権威。

---

## 4. 改修方法

AudioEngineは利用者に降格。

State保持禁止。

---

# ⑦ Fading DSP 管理が旧Runtime構造

## 1. 未達成内容

Fadingスロット管理が残っています。

---

## 2. 該当箇所

```cpp
exchangeFadingRuntimeDSP(...)
```

```cpp
replaceFadingRuntimeDSPAndRetirePrevious
```



---

## 3. あるべき姿

```text
RuntimeWorld
 ├ activeHandle
 └ fadingHandle
```

---

## 4. 改修方法

DSPポインタ管理廃止。

Handle管理へ移行。

---

# ⑧ PublicationExecutor が実質薄い委譲

## 1. 未達成内容

Executorが存在しても

Coordinator.publishWorld()経由です。

---

## 2. 該当箇所

改修計画でも

```text
PR-3で置換予定
```

となっています。 

---

## 3. あるべき姿

```cpp
executor.publish(world)
```

↓

```cpp
store.publishAndSwap()
```

---

## 4. 改修方法

publishWorld()排除。

Executorが直接Storeを操作。

---

# ⑨ RuntimeWorld Immutability が未完成

## 1. 未達成内容

seal/freeze系が確認できません。

監査ログでも検出されています。 

---

## 2. 該当箇所

検索結果

```text
sealRecursively
freeze
isSealed
```

未整備。 

---

## 3. あるべき姿

```text
Build
 ↓
Seal
 ↓
Publish
 ↓
Immutable
```

---

## 4. 改修方法

PublicationExecutor publish前に

```cpp
world.sealRecursively()
```

導入。

---

# ⑩ RuntimeBuildSnapshot Authority が完全確立していない

## 1. 未達成内容

設計上は

```cpp
sealedSnapshot
```

がAuthorityですが、

実際にはDSPCore情報を読みに行く経路が残っています。

---

## 2. 該当箇所

改修計画にも

```text
sealedSnapshot authority 明確化
```

が未完として記載されています。

---

## 3. あるべき姿

```text
RuntimeBuildSnapshot
 ↓
RuntimeWorld
 ↓
Publish
```

DSPCoreは非権威。

---

## 4. 改修方法

World構築に必要なSemantic情報を

```cpp
RuntimeBuildSnapshot
```

へ完全投影。

DSPCoreから取得禁止。

---

# 監査結論

最新版 ConvoPeq.md の Practical Stable ISR Bridge Runtime 完成度を厳密に評価すると、

| 項目                       | 達成度 |
| ------------------------ | --- |
| RuntimeWorld Publication | 90% |
| Admission Governance     | 90% |
| Fail Closed              | 90% |
| RuntimeWorld Projection  | 80% |
| Crossfade Authority      | 75% |
| DSP Lifetime Separation  | 60% |
| AudioEngine 非権威化         | 55% |
| Legacy Commit Path 排除    | 40% |
| World Immutability       | 30% |

最大の未達成事項は、

**① applyRuntimeCommitFromIntent モノリスの残存**
**② AudioEngine が依然 Runtime Authority**
**③ DSPCore が Semantic Decision Source に残存**

の3つです。

この3点が解消されると、Practical Stable ISR Bridge Runtime の中核思想である

```text
RuntimeWorld Authority
↓
Admission
↓
Executor
↓
Transition
↓
Retire
```

への移行がほぼ完了したと評価できます。



前回挙げた

* Legacy Commit Path
* AudioEngine Authority残存
* DSPCore Semantic Authority残存
* CrossfadeAuthority未完
* DSPLifetimeManager未完
* Executor未完
* RuntimeWorld Immutability未完

以外にも、ConvoPeq.md を見ると **Practical Stable ISR Bridge Runtime の完成形からまだ距離がある箇所** が複数あります。

以下は「まだ前回挙げていないもの」を中心に整理します。

---

# ⑪ Publication Success 前に DSP Activation が発生し得る設計痕跡

## 1. 未達成内容

ISR Bridge Runtime の原則は

```text
Publish Success
 ↓
Activate
 ↓
Transition
 ↓
Retire
```

です。

しかし計画書・実装痕跡を見ると、

```cpp
oldDSP = lifetime.getActive();
lifetime.activate(req.newDSP);
```

の順序が Pipeline 側に存在する設計が残っています。

後続計画では修正されていますが、

この系統がコードベースに残っている可能性があります。

---

## 2. 該当箇所

PublicationPipeline execute案

```cpp
lifetime.activate(req.newDSP);
```

対して後続設計では

```cpp
executor.publish(...)
↓
transition.onPublishCompleted(...)
↓
activate
```

になっています。

---

## 3. あるべき姿

```text
Build
↓
Validate
↓
PublishAndSwap
↓
Success確認
↓
Activate
```

---

## 4. 改修方法

Activation実行箇所を

```cpp
DSPTransition::onPublishCompleted()
```

だけに限定する。

---

# ⑫ Warmup が Publication Pipeline の外にある

## 1. 未達成内容

Practical Stable ISR Bridge Runtimeでは

```text
Build
↓
Warmup
↓
Seal
↓
Publish
```

の責務分離が必要です。

しかし現状は

```text
applyRuntimeCommitFromIntent
↓
executeWarmup
```

由来の構造が残っています。 

---

## 2. 該当箇所

```text
warmup (RuntimeBuilder::executeWarmup)
```



---

## 3. あるべき姿

```text
WarmupExecutor
 ↓
RuntimeBuilder
 ↓
PublicationExecutor
```

---

## 4. 改修方法

Warmup専用コンポーネントへ移譲。

Publication系から切り離す。

---

# ⑬ Retire Epoch Authority が Coordinator に集約されていない

## 1. 未達成内容

Retire Epoch 更新がまだ独立Authorityになっていません。

計画上も

```text
advanceRetireEpoch
↓
Coordinator内蔵
```

が移行対象です。 

---

## 2. 該当箇所

```text
advanceRetireEpoch
```



---

## 3. あるべき姿

```text
PublicationExecutor
 ↓
RetireEpochAuthority
```

---

## 4. 改修方法

Epoch更新をCoordinator/Executor管理へ集約。

---

# ⑭ Latency Adjustment が Publication の副作用として残存

## 1. 未達成内容

レイテンシ更新がPublicationロジックに混在しています。

計画でも

```text
latency adjustment (~40行)
```

として独立抽出対象になっています。 

---

## 2. 該当箇所

```text
latency adjustment
```



---

## 3. あるべき姿

```text
Publication
 ↓
PublicationEvent
 ↓
LatencyService
```

---

## 4. 改修方法

Facade経由のイベント処理へ分離。

---

# ⑮ UI通知が Publication Pipeline に混在

## 1. 未達成内容

ISR Runtimeの権威経路に

```text
sendChangeMessage
triggerAsyncUpdate
enqueueLearningCommand
uiConvolverProcessor.setMixedPhaseState
```

が混在しています。 

---

## 2. 該当箇所

```text
enqueueLearningCommand
sendChangeMessage
triggerAsyncUpdate
```



---

## 3. あるべき姿

```text
PublicationCompletedEvent
 ↓
Observer
 ↓
UI
```

---

## 4. 改修方法

Coordinatorからイベント通知のみ行う。

---

# ⑯ Deferred Publication Queue が RuntimeStore Authority になっていない

## 1. 未達成内容

Deferred publish の保留管理が

まだCoordinator内部のローカル状態です。

```cpp
std::optional<PublishRequest> deferredRequest_;
bool hasDeferredRequest_;
```



---

## 2. 該当箇所

PublicationPipeline

```cpp
deferredRequest_
```



---

## 3. あるべき姿

```text
RuntimeStore
 ↓
PendingPublicationQueue
```

---

## 4. 改修方法

Store管理下へ移動。

---

# ⑰ PublicationRequest が DSPCore を直接保持

## 1. 未達成内容

PublishRequest が

```cpp
DSPCore* newDSP
```

を保持しています。 

ISR Runtime思想では

Semantic Authority は Snapshot です。

---

## 2. 該当箇所

```cpp
struct PublishRequest
{
    DSPCore* newDSP;
    RuntimeBuildSnapshot sealedSnapshot;
}
```



---

## 3. あるべき姿

```cpp
PublishRequest
{
    RuntimeBuildSnapshot sealedSnapshot;
    DSPHandle handle;
}
```

---

## 4. 改修方法

Publish判断ではDSPCore禁止。

Execution段階のみHandle→DSP取得。

---

# ⑱ RuntimeBuilder が DSPCore 依存のまま

## 1. 未達成内容

Builderが

```cpp
buildRuntimePublishWorld(
    currentDSP,
    oldDSP,
    ...
)
```

型です。

---

## 2. 該当箇所

```cpp
buildRuntimePublishWorld(
    req.newDSP,
    oldDSP,
    ...
)
```



---

## 3. あるべき姿

```cpp
buildRuntimePublishWorld(
    sealedSnapshot,
    transitionDecision)
```

---

## 4. 改修方法

Builder入力をSnapshot中心へ移行。

---

# ⑲ Crossfade Registration Authority が二重化

## 1. 未達成内容

CrossfadeAuthority が存在する一方、

DSPTransition側で直接

```cpp
engine_.crossfadeAuthorityRuntime_.registerCrossfade(...)
```

しています。 

---

## 2. 該当箇所

```cpp
registerCrossfade(...)
```

直接呼び出し。 

---

## 3. あるべき姿

```text
DSPTransition
 ↓
CrossfadeAuthority
 ↓
registerCrossfade
```

---

## 4. 改修方法

登録経路をCrossfadeAuthorityへ一本化。

---

# ⑳ Publish Entry Point が完全統一されていない

## 1. 未達成内容

計画には

```text
prepareToPlay()
timer path
rebuild path
```

の統合対象が明記されています。

これは裏返すと、

まだ複数入口が残っている可能性を示します。

---

## 2. 該当箇所

```text
prepareToPlay 経路
timer 経路
rebuild 経路
```

---

## 3. あるべき姿

```text
submitPublishRequest()
      ↑
 全 publish 経路
```

---

## 4. 改修方法

Coordinator以外から

```cpp
publishWorld()
```

を呼べないようにする。

---

# 追加監査結論

前回の10項目に加え、今回の10項目で特に重要なのは次の4つです。

1. Publish成功前Activationの排除（⑪）
2. PublishRequestがDSPCoreを権威として保持している点（⑰）
3. Crossfade登録Authorityの二重化（⑲）
4. Publish Entry Point未統一（⑳）

この4項目は単なるリファクタリングではなく、Practical Stable ISR Bridge Runtime の中核原則である

```text
Snapshot Authority
↓
Single Publication Entry
↓
Single Transition Authority
↓
Fail Closed
```

に直接関係する未達成事項です。これらが解消されると、残件は主に実装整理・責務分離レベルになります。
