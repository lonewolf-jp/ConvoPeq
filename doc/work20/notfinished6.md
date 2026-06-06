最新版 ConvoPeq.md（2026-06-06版）を前提に監査すると、Practical Stable ISR Bridge Runtime の中核部分（Admission → Build → Publish → Transition）はかなり整っています。

しかし、「Practical Stable ISR Bridge Runtime 完了」と呼ぶには、まだ複数の未達成項目があります。

以下は優先度順です。

---

# ① DSPLifetime がまだ AudioEngine Authority 配下

## 1. 未達成内容

現在の DSPLifetimeManager は独立した Runtime Authority ではなく、

```text
DSPLifetimeManager
  ↓
AudioEngine
```

というラッパーです。

Practical Stable ISR Bridge Runtime の最終形では

```text
RuntimePublicationOrchestrator
  ↓
DSPLifetimeManager
  ↓
RetireQueue
  ↓
DSPHandleRuntime
```

で完結し、

```cpp
AudioEngine::setActiveRuntimeDSP()
AudioEngine::retireDSP()
```

を直接呼ばない状態が望ましいです。

---

## 2. 該当ソース

DSPLifetimeManager



```cpp
engine_.setActiveRuntimeDSP(dsp);
```

```cpp
engine_.retireDSP(dsp);
```

```cpp
AudioEngine::DSPCore* getActive() const noexcept
{
    return engine_.getActiveRuntimeDSP();
}
```

またコメント自身が

```cpp
// ★Phase-A: AudioEngine のラッパーに過ぎない。
//   真の分離は Phase-B/C で行う。
```

と明示しています。 

---

## 3. あるべき姿

```text
AudioEngine
  ↓
Runtime APIs only

DSPLifetimeManager
  ↓
ActiveSlot
FadingSlot
RetireQueue
HandleRuntime
```

---

## 4. 改修方法

DSPLifetimeManager に

```cpp
activateHandle()
retireHandle()
exchangeFadingHandle()
```

を内蔵。

AudioEngine は

```cpp
setActiveRuntimeDSP()
retireDSP()
```

を Runtime 公開経路から使用しない。

---

# ② Execution Path がまだ DSPCore* ベース

## 1. 未達成内容

Practical Stable ISR Bridge Runtime の理想は

```text
Execution Path
=
DSPHandle
```

です。

しかし現在は

```cpp
DSPCore*
```

が多数残っています。

---

## 2. 該当ソース

RuntimePublicationOrchestrator



```cpp
auto* newDSPResolved =
    engine_.resolveDSPHandle(req.newDSP);

auto* oldDSP =
    engine_.getActiveRuntimeDSP();
```

DSPTransition



```cpp
onPublishCompleted(
    AudioEngine::DSPCore* newDSP,
    AudioEngine::DSPCore* oldDSP,
```

---

## 3. あるべき姿

```cpp
DSPHandle activeHandle;
DSPHandle fadingHandle;
```

のみ。

---

## 4. 改修方法

Transition 系 API を

```cpp
DSPCore*
↓
DSPHandle
```

へ変更。

DSPCore 解決は

```cpp
resolveDSPHandle()
```

の単一点のみ。

---

# ③ getActiveRuntimeDSP() Authority Leakage

## 1. 未達成内容

Practical Stable ISR Bridge Runtime の最重要原則の一つは

```text
Semantic Decision
=
RuntimeWorld
```

です。

しかし現状は

```cpp
getActiveRuntimeDSP()
```

取得経路が残っています。

---

## 2. 該当ソース

RuntimePublicationOrchestrator



```cpp
auto* oldDSP =
    engine_.getActiveRuntimeDSP();
```

releaseResources



```cpp
auto* const activeRaw =
    getActiveRuntimeDSP();
```

DSPLifetimeManager



```cpp
getActiveRuntimeDSP()
```

---

## 3. あるべき姿

Decision 系コードは

```cpp
RuntimePublishWorld
RuntimeState
dspProjection
```

のみ使用。

---

## 4. 改修方法

Execution 用のみ許可。

CI 監査追加。

```bash
grep getActiveRuntimeDSP
```

発見箇所をレビュー対象化。

---

# ④ Crossfade Runtime State が AudioEngine に残存

## 1. 未達成内容

Crossfade Authority は分離されたが、

Crossfade Execution State はまだ AudioEngine が保持しています。

---

## 2. 該当ソース

DSPTransition



```cpp
engine_.dspCrossfadeGain
```

```cpp
engine_.dspCrossfadePending
```

```cpp
engine_.queuedFadeTimeSec
```

```cpp
engine_.activeCrossfadeId_
```

---

## 3. あるべき姿

```text
CrossfadeRuntime
```

という Runtime Component が所有。

---

## 4. 改修方法

Crossfade 関連 atomics を

```cpp
CrossfadeRuntime
```

へ移動。

DSPTransition は

```cpp
crossfadeRuntime.start(...)
```

のみ呼ぶ。

---

# ⑤ RuntimeStore の完全 Authority 化未確認

## 1. 未達成内容

設計上は

```text
RuntimePublicationCoordinator
```

だけが Publish する想定。

しかしコード全体で保証されているか不明。

---

## 2. 該当ソース

監査対象。

RuntimeStore publish 呼び出し元全件確認が必要。

関連監査項目は過去計画にも存在。 

---

## 3. あるべき姿

```text
RuntimeStore::publish
```

呼び出し元

↓

```text
RuntimePublicationCoordinator
```

のみ。

---

## 4. 改修方法

friend 化または CI。

```bash
grep RuntimeStore.*publish
```

---

# ⑥ Shutdown Path が旧 Runtime 経路を保持

## 1. 未達成内容

Shutdown シーケンスがまだ

```cpp
getActiveRuntimeDSP()
retireDSP()
```

へ直接依存。

---

## 2. 該当ソース

releaseResources()



```cpp
activeToRelease
```

```cpp
fadingToRelease
```

```cpp
retireDSP(activeToRelease)
```

```cpp
retireDSP(fadingToRelease)
```

---

## 3. あるべき姿

```text
ShutdownRuntime
↓
DSPLifetimeManager
↓
Retire
```

---

## 4. 改修方法

Shutdown も

```cpp
DSPLifetimeManager
```

経由へ統一。

---

# ⑦ DSPTransition が巨大化している

## 1. 未達成内容

現在 DSPTransition が

```text
Activate
Crossfade Registration
Crossfade State
Atomic更新
Retire
```

全部持っています。

---

## 2. 該当ソース

DSPTransition

---

## 3. あるべき姿

```text
DSPTransition
  ↓
DSPLifetimeManager

CrossfadeRuntime

RetireRuntime
```

---

## 4. 改修方法

Transition を

```cpp
TransitionCoordinator
```

レベルまで縮小。

---

# ⑧ RuntimeWorld → DSPProjection Coverage が未完全

## 1. 未達成内容

CrossfadeAuthority が参照する Projection は

```cpp
irLoaded
structuralHash
oversamplingFactor
```

のみです。

しかし将来的に Crossfade 要否へ影響する項目が増えた場合、

Coverage 漏れが発生します。

---

## 2. 該当ソース

CrossfadeAuthority

```cpp
kEvaluateRelevantFieldNames
```

---

## 3. あるべき姿

Projection 定義と Crossfade 判定が自動整合。

---

## 4. 改修方法

Contract Test 強化。

```cpp
ProjectionFieldCoverageTest
```

追加。

---

# 総括

実運用で破綻しにくい Practical Stable ISR Bridge Runtime の観点では、未達成項目の重要度は次の順です。

| 優先度 | 項目                                          |
| --- | ------------------------------------------- |
| S   | DSPLifetimeManager がまだ AudioEngine ラッパー     |
| S   | getActiveRuntimeDSP() 経由の Authority Leakage |
| S   | Execution Path が DSPCore* 依存                |
| A   | Shutdown Path の旧 Runtime 経路                 |
| A   | Crossfade Runtime State が AudioEngine 所有    |
| A   | RuntimeStore Publish Authority の完全保証        |
| B   | DSPTransition の責務肥大化                        |
| B   | DSPProjection Coverage Contract 強化          |

この中で、**Practical Stable ISR Bridge Runtime の完成判定を妨げている本質的な未達成項目は上位3つ（DSPLifetime、DSPCore依存、Authority Leakage）**です。これらが解消されると、アーキテクチャ上はほぼ「Bridge Runtime 完了」と評価できる状態になります。



前回提示した未達成項目のうち、実際に「Practical Stable ISR Bridge Runtime 完了判定」に影響するものは主に以下です。

* S1: DSPLifetimeManager が AudioEngine ラッパー
* S2: DSPCore* 依存
* S3: getActiveRuntimeDSP() Authority Leakage
* A1: Shutdown Path の旧Runtime依存
* A2: Crossfade Runtime State の AudioEngine依存

ただし、ここで重要なのは、

> 「理論的に完全な分離」

ではなく、

> 「実運用で破綻しにくい」

です。

したがって、全面的な再設計ではなく、

**Authority分離を維持しつつ、実装規模を最小化する改修**

が望ましいです。

---

# S1. DSPLifetimeManager が AudioEngine ラッパー

---

## 現状

現在は実質的に

```text
RuntimePublicationOrchestrator
    ↓
DSPLifetimeManager
    ↓
AudioEngine
```

です。

DSPLifetimeManager の中身が

```cpp
engine_.setActiveRuntimeDSP(...)
engine_.retireDSP(...)
engine_.getActiveRuntimeDSP(...)
```

しかありません。

つまり

```text
Authority = AudioEngine
```

です。

---

## Practical Stable ISR としてのあるべき姿

必要なのは

```text
Authority = DSPLifetimeManager
Execution = AudioEngine
```

です。

---

## 推奨改修

### Step1

DSPLifetimeManager に状態を持たせる

追加：

```cpp
class DSPLifetimeManager
{
private:
    DSPHandle activeHandle_;
    DSPHandle fadingHandle_;
};
```

---

### Step2

activeDSP取得禁止

現在

```cpp
engine_.getActiveRuntimeDSP()
```

している箇所を

```cpp
lifetimeManager.getActiveHandle()
```

へ置換。

---

### Step3

AudioEngineは実行だけ

AudioEngineへは

```cpp
engine_.installActiveDSP(...)
```

のようなExecution APIだけ残す。

判断禁止。

---

## 完了判定

以下が成立。

```bash
grep getActiveRuntimeDSP RuntimePublicationOrchestrator.cpp
```

0件

---

# S2. DSPCore* 依存

---

## 現状

Orchestrator

DSPTransition

DSPLifetimeManager

が

```cpp
AudioEngine::DSPCore*
```

を扱っています。

---

## 問題

DSPCore が RuntimeWorld 外へ漏れる。

将来

```cpp
if (oldDSP->convolverRt())
```

みたいなコードが復活可能。

Authority逆流。

---

## Practical Stable ISR としてのあるべき姿

Runtime層は

```cpp
DSPHandle
```

のみ扱う。

---

## 推奨改修

### Step1

TransitionRequest変更

現在

```cpp
DSPCore* oldDSP;
DSPCore* newDSP;
```

↓

```cpp
DSPHandle oldHandle;
DSPHandle newHandle;
```

---

### Step2

DSPTransition変更

現在

```cpp
onPublishCompleted(
    DSPCore* oldDSP,
    DSPCore* newDSP)
```

↓

```cpp
onPublishCompleted(
    DSPHandle oldHandle,
    DSPHandle newHandle)
```

---

### Step3

DSP解決点を1箇所化

唯一許可

```cpp
resolveDSPHandle()
```

のみ。

---

## 完了判定

```bash
grep "DSPCore\*" runtime/
```

Execution層以外0件

---

# S3. getActiveRuntimeDSP Authority Leakage

---

これは最重要です。

---

## 現状

現在

```cpp
auto* oldDSP =
    engine_.getActiveRuntimeDSP();
```

があります。

Execution目的なら問題なし。

しかし監査不能。

---

## Practical Stable ISR としてのあるべき姿

Decisionコードでは

```cpp
RuntimeWorld
```

のみ。

---

## 推奨改修

### Rule1

取得禁止対象

```cpp
getActiveRuntimeDSP()
getFadingRuntimeDSP()
```

---

### Rule2

例外

許可：

```cpp
AudioEngine::processBlock()
```

のみ。

---

### Rule3

CI追加

```bash
grep -R "getActiveRuntimeDSP"
```

実施。

---

### Rule4

Semantic監査

発見時は

```cpp
if (...)
```

判定利用していないか確認。

---

## 完了判定

Decisionコードから完全消滅。

---

# A1. Shutdown Path

---

## 現状

releaseResources()

が

```cpp
activeDSP
fadingDSP
```

を直接取得。

---

## 問題

Runtime Authorityをバイパス。

---

## 推奨改修

Shutdown専用API

```cpp
lifetimeManager.shutdown();
```

追加。

---

内部

```cpp
retire(activeHandle_);
retire(fadingHandle_);
```

実施。

---

releaseResources()

は

```cpp
lifetimeManager.shutdown();
```

のみ。

---

## 完了判定

```bash
grep retireDSP AudioEngine.cpp
```

Shutdown以外0件

---

# A2. Crossfade Runtime State

---

## 現状

AudioEngine に

```cpp
dspCrossfadePending
dspCrossfadeGain
queuedFadeTimeSec
activeCrossfadeId_
```

があります。

---

## 問題

Crossfade AuthorityとExecutionが混在。

---

## 推奨改修

新規

```cpp
class CrossfadeRuntime
{
private:
    std::atomic<float> gain_;
    std::atomic<bool> pending_;
    ...
};
```

---

DSPTransition

現在

```cpp
engine_.dspCrossfadeGain
```

↓

```cpp
crossfadeRuntime_.gain()
```

---

AudioEngine

現在

```cpp
engine_.dspCrossfadePending
```

↓

```cpp
crossfadeRuntime.pending()
```

---

## 完了判定

AudioEngineから

```cpp
dspCrossfade*
```

系メンバ消滅。

---

# 実運用重視での推奨順序

理論上は全部やれますが、実運用で破綻しにくくするだけなら優先順位は以下です。

### Phase-A（必須）

1. getActiveRuntimeDSP()監査
2. DSPCore* Decision排除
3. Shutdown経路統一

---

### Phase-B（推奨）

4. DSPLifetimeManager Authority化
5. CrossfadeRuntime分離

---

### Phase-C（余裕があれば）

6. DSPTransition責務分割
7. RuntimeStore完全封鎖
8. Projection Coverage強化

実際には、Phase-A完了時点で Practical Stable ISR Bridge Runtime の安定性は大きく向上します。Phase-Bまで完了すると、現在の設計思想（Semantic World Authority + Runtime Publication Authority）と実装がほぼ一致する状態になります。



前回までに挙げた

* DSPLifetimeManager Authority化
* DSPCore*依存
* getActiveRuntimeDSP() Leakage
* Shutdown Path
* Crossfade Runtime State

以外にも、ConvoPeq.md を見ると、Practical Stable ISR Bridge Runtime の観点ではまだ未達成と考えられる項目があります。

ただし今回は、

> 「実運用で破綻しにくい Practical Stable ISR Bridge Runtime」

という基準で評価しています。

つまり「理論的にもっと綺麗にできる」ではなく、

「将来の改修でAuthority崩壊や経路逆流が発生し得る」

ものを抽出しています。

---

# ① Legacy Commit Path 完全消滅未達

## 1. 未達成内容

Practical Stable ISR Bridge Runtime では

```text
RuntimePublicationCoordinator
↓
Admission
↓
Build
↓
Publish
↓
Transition
```

のみが Publish 経路であるべきです。

しかし計画資料には

```cpp
applyRuntimeCommitFromIntent()
processPendingCommit()
PendingCommitData
```

削除対象として明記されています。 

これは逆に言うと、

過去監査時点では

```text
Legacy Commit Path
+
Bridge Runtime Path
```

が並存していたことを示しています。

---

## 2. 該当箇所

削除対象として明記

```cpp
AudioEngine::processPendingCommit()
AudioEngine::applyRuntimeCommitFromIntent()
PendingCommitData
pendingCommitFlag_
pendingCommit_
```



---

## 3. あるべき姿

```text
Publish Entry
=
RuntimePublicationCoordinatorのみ
```

---

## 4. 改修方法

CI追加

```bash
grep -R "applyRuntimeCommitFromIntent"
grep -R "processPendingCommit"
grep -R "PendingCommitData"
```

0件化。

Coordinator以外から Publish 不可。

---

# ② Coordinator が Facade 止まり

## 1. 未達成内容

Coordinator が存在していても、

実際の決定権が AudioEngine に残ると

Authority移行は未完了です。

監査資料でも

```text
Coordinator が Facade 止まり
```

が未達として挙がっています。 

---

## 2. 該当箇所

監査資料

```text
S-2 (⑰)
Coordinator が Facade 止まり
```



---

## 3. あるべき姿

```text
Coordinator
 ├ Admission
 ├ Build
 ├ Publish
 ├ Activate
 └ Retire
```

AudioEngineは実行体のみ。

---

## 4. 改修方法

Coordinatorに

```cpp
submitPublishRequest()
notifyTransitionComplete()
```

を集中。

AudioEngineから判断ロジックを除去。

---

# ③ Publication と DSP Lifetime の密結合

## 1. 未達成内容

Publish 成功後の DSP 遷移が

AudioEngine 側に埋め込まれていると

Runtime Authority が分裂します。

監査資料でも

```text
Publication と DSP Lifetime 密結合
```

が未達。 

---

## 2. 該当箇所

監査資料

```text
S-3 (⑱)
Publication と DSP Lifetime 密結合
```



---

## 3. あるべき姿

```text
PublicationExecutor
    ↓
PublishResult

DSPTransition
    ↓
DSPLifetimeManager
```

---

## 4. 改修方法

Publish 完了通知は

```cpp
PublicationResult
```

だけ返す。

DSP切替は別フェーズ化。

---

# ④ RuntimeWorld / DSPCore 二重モデル

## 1. 未達成内容

Semantic Authority が RuntimeWorld に移ったにもかかわらず、

別経路で DSPCore を読めると

Authority が二重化します。

監査資料でも

```text
RuntimeWorld/DSPCore 二重モデル
```

が未達。 

---

## 2. 該当箇所

監査資料

```text
A-3 (⑳)
RuntimeWorld/DSPCore 二重モデル
```



また DSPCore直読一覧

```cpp
convolverRt().isIRLoaded()
isIRFinalized()
getStructuralHash()
oversamplingFactor
sampleRate
```



---

## 3. あるべき姿

Semantic判断

```text
RuntimeWorld.dspProjection
```

のみ。

---

## 4. 改修方法

Projection追加。

```cpp
irLoaded
irFinalized
structuralHash
oversamplingFactor
sampleRate
```

を RuntimeWorld へ移管。

---

# ⑤ Observe Path 多重化

## 1. 未達成内容

状態取得経路が複数存在すると

RuntimeWorld が権威でなくなります。

監査資料でも

```text
Observe Path 多重化
```

が未達。 

---

## 2. 該当箇所

監査資料

```text
A-6 (⑧)
Observe Path 多重化
```



---

## 3. あるべき姿

```text
Read
 ↓
RuntimeReadHandle
 ↓
Projection
```

のみ。

---

## 4. 改修方法

監査ルール追加。

```bash
grep RuntimeWorld
grep getActiveRuntimeDSP
grep currentRuntimeState
```

読取経路を棚卸し。

---

# ⑥ Publish決定権がAudioEngine側

## 1. 未達成内容

Publish可否を AudioEngine が決めるなら

Coordinator は Authority ではありません。

監査資料で未達。 

---

## 2. 該当箇所

監査資料

```text
A-7 (㉕)
publish 決定権が AudioEngine 側
```



---

## 3. あるべき姿

```text
Admission
 ↓
Coordinator
 ↓
Accepted / Rejected
```

---

## 4. 改修方法

Publish判定APIを Coordinator 内へ集約。

AudioEngine 側の

```cpp
acceptsRuntimePublication()
```

系判定を削減。

---

# ⑦ Crossfade Authority 二重化

## 1. 未達成内容

CrossfadeAuthority が存在しても、

別の場所で Crossfade 判断を行うと Authority が二重になります。

監査資料で未達。 

---

## 2. 該当箇所

監査資料

```text
A-4 (⑲)
Crossfade Authority 二重化
```



旧設計資料にも

```cpp
computeCrossfadeContext
CrossfadeAuthority
```

の並存履歴あり。 

---

## 3. あるべき姿

```text
Crossfade Decision
=
CrossfadeAuthority only
```

---

## 4. 改修方法

CI追加。

```bash
grep CrossfadeContext
grep computeCrossfadeContext
```

CrossfadeAuthority以外での判断禁止。

---

# ⑧ Semantic/Execution 分離未完成

## 1. 未達成内容

Practical Stable ISR Bridge Runtime の核心です。

監査資料でも

```text
Semantic/Execution 分離未完成
```

が未達として挙がっています。 

---

## 2. 該当箇所

DSPCore直読箇所

```cpp
isIRLoaded()
isIRFinalized()
getStructuralHash()
oversamplingFactor
sampleRate
estimateRuntimeLatencyBaseRateSamples()
```



---

## 3. あるべき姿

```text
Semantic
 ↓
RuntimeWorld Projection

Execution
 ↓
DSPCore
```

完全分離。

---

## 4. 改修方法

Projectionを拡張。

DSPCoreからSemantic情報を剥離。

---

# ⑨ RuntimeWorld Construction Authority の完全封鎖未確認

## 1. 未達成内容

RuntimeBuilder 以外が RuntimeWorld を生成できるなら Authority が崩れます。

CI規約でも監査対象になっています。 

---

## 2. 該当箇所

CIルール

```bash
new RuntimePublishWorld
make_unique<RuntimePublishWorld>
```

検査。 

---

## 3. あるべき姿

```text
RuntimeWorld生成
=
RuntimeBuilderのみ
```

---

## 4. 改修方法

constructor非公開化。

Builderをfriend化。

---

# 総括

前回提示した項目を除外した場合、Practical Stable ISR Bridge Runtime の未達成事項として重要なのは次の9項目です。

| 優先度 | 項目                                           |
| --- | -------------------------------------------- |
| S   | Legacy Commit Path 残存                        |
| S   | Coordinator が Facade 止まり                     |
| S   | Publication と DSP Lifetime 密結合               |
| A   | RuntimeWorld / DSPCore 二重モデル                 |
| A   | Semantic / Execution 分離未完成                   |
| A   | Publish決定権が AudioEngine 側                    |
| A   | Crossfade Authority 二重化                      |
| A   | Observe Path 多重化                             |
| B   | RuntimeWorld Construction Authority の完全封鎖未確認 |

特に上位6項目は、単独の問題というより、

**「AudioEngine がまだ Runtime Orchestrator の主語として残っている」**

ことの派生問題です。ConvoPeq.md の監査資料でも同じ根本原因が指摘されており、Practical Stable ISR Bridge Runtime 完了のためには AudioEngine→Coordinator への主語移行が最終的な焦点になります。



まず前提として、今回挙げた9項目はすべて同じ方向を向いています。

Practical Stable ISR Bridge Runtime の本質は

```text
旧:
AudioEngine
 ├ Semantic
 ├ Runtime
 ├ DSP Lifetime
 ├ Publish
 └ Crossfade

↓

目標:
Runtime Authority Layer
 ├ Admission
 ├ RuntimeBuilder
 ├ RuntimePublicationCoordinator
 ├ DSPTransition
 ├ DSPLifetimeManager
 └ CrossfadeAuthority

AudioEngine
 └ Execution Only
```

です。

そのため、個別修正ではなく、

**「Authority封鎖」**
**「Decision経路封鎖」**
**「生成経路封鎖」**

の3軸で改修した方が安全です。

---

# ① Legacy Commit Path 残存

---

## 問題の本質

現在のCoordinator経路とは別に、

```cpp
processPendingCommit()
applyRuntimeCommitFromIntent()
```

が残っている場合、

```text
Publish経路A
Coordinator

Publish経路B
Legacy Commit
```

になります。

これは Runtime Authority が二重化します。

---

## 改修方針

### Phase1

全参照洗い出し

```bash
grep -R "applyRuntimeCommitFromIntent"
grep -R "processPendingCommit"
grep -R "PendingCommitData"
grep -R "pendingCommitFlag_"
grep -R "pendingCommit_"
```

---

### Phase2

呼び出し元特定

図にする

```text
caller
 ↓
processPendingCommit
 ↓
publish
```

---

### Phase3

Coordinatorへ移植

旧

```cpp
AudioEngine::processPendingCommit()
{
    ...
}
```

↓

```cpp
RuntimePublicationCoordinator::submit()
```

---

### Phase4

削除

```cpp
PendingCommitData
pendingCommit_
pendingCommitFlag_
```

削除

---

## 完了判定

```bash
grep -R "PendingCommit"
```

0件

---

# ② Coordinator が Facade 止まり

---

## 問題

Coordinator が単なる転送係。

例

```cpp
coordinator.submit()
{
    engine.process(...)
}
```

これでは

```text
Authority = Engine
```

です。

---

## 改修方針

Coordinator を状態機械化する。

---

### Coordinator所有物

```cpp
PublicationAdmission
PublicationExecutor
DSPTransition
```

---

### Coordinator責務

```cpp
submit()
accept()
reject()
publish()
activate()
retire()
```

---

### 禁止

```cpp
engine.determinePublish()
engine.determineCrossfade()
engine.determineActivation()
```

---

## 完了判定

Coordinator が Publish Sequence 全体を保持。

---

# ③ Publication と DSP Lifetime 密結合

---

## 問題

現在は

```text
Publish成功
 ↓
DSP切替
```

が同一関数。

障害切り分け不能。

---

## 改修方針

結果オブジェクト導入。

---

### 新規

```cpp
struct PublicationResult
{
    bool success;
    RuntimeHandle handle;
};
```

---

### Executor

```cpp
PublicationResult publish(...)
```

---

### Coordinator

```cpp
auto result = publish(...);

if (result.success)
{
    transition.activate(...);
}
```

---

## 完了判定

Publication層が DSP を知らない。

---

# ④ RuntimeWorld / DSPCore 二重モデル

---

## 問題

現在

```cpp
RuntimeWorld
```

と

```cpp
DSPCore
```

の両方から意味情報取得可能。

---

## 改修方針

Semantic Projection 完全化。

---

### 監査対象

```cpp
isIRLoaded()
isIRFinalized()
getStructuralHash()
oversamplingFactor()
sampleRate()
estimateRuntimeLatency...
```

---

### Projectionへ移行

```cpp
struct DSPProjection
{
    bool irLoaded;
    bool irFinalized;

    uint64_t structuralHash;

    int oversamplingFactor;

    double sampleRate;

    int latencySamples;
};
```

---

### Rule

Decision層

```cpp
DSPCore禁止
```

---

## 完了判定

Decisionコードから DSPCore読取0件。

---

# ⑤ Observe Path 多重化

---

## 問題

現在

```text
RuntimeWorld
RuntimeStore
DSPCore
AudioEngine
```

から読める。

---

## 改修方針

観測API統一。

---

### 新規規約

読取経路

```text
RuntimeReadHandle
 ↓
Projection
```

のみ

---

### CI

```bash
grep getActiveRuntimeDSP
grep currentRuntimeState
grep runtimeWorld
```

監査

---

## 完了判定

Observe Path が1本化。

---

# ⑥ Publish決定権が AudioEngine 側

---

## 問題

Coordinator がいても

```cpp
engine.canPublish()
```

ならAuthorityはEngine。

---

## 改修方針

Admission集約。

---

### Admission責務

```cpp
validate()
accept()
reject()
```

---

### Engine禁止

```cpp
canPublish()
acceptPublication()
rejectPublication()
```

系

---

### Coordinator

```cpp
if (!admission.accept(req))
    return;
```

---

## 完了判定

Publish判定が Admission のみ。

---

# ⑦ Crossfade Authority 二重化

---

## 問題

CrossfadeAuthority以外でも

```cpp
if (...)
{
    fade = true;
}
```

している可能性。

---

## 改修方針

Crossfade判定型導入。

---

### 新規

```cpp
enum class CrossfadeDecision
{
    Required,
    NotRequired
};
```

---

### Authority

```cpp
CrossfadeDecision evaluate(...)
```

のみ。

---

### 禁止

```cpp
needsCrossfade(...)
computeCrossfade(...)
```

複製実装。

---

### CI

```bash
grep -R "crossfade"
```

レビュー対象。

---

## 完了判定

Decision Source が1箇所。

---

# ⑧ Semantic / Execution 分離未完成

---

## 問題

DSPCoreを意味判定に使用。

---

## 改修方針

レイヤー規約導入。

---

### Semantic Layer

許可

```cpp
RuntimeWorld
Projection
SemanticSchema
```

---

### Execution Layer

許可

```cpp
DSPCore
AudioProcessor
Convolver
```

---

### CI

禁止パターン

```bash
grep DSPCore
```

対象ディレクトリ限定。

---

## 完了判定

Semantic層にDSPCore参照なし。

---

# ⑨ RuntimeWorld Construction Authority

---

## 問題

RuntimeBuilder以外で

```cpp
RuntimePublishWorld
```

生成可能。

---

## 改修方針

生成封鎖。

---

### RuntimePublishWorld

```cpp
private:
    RuntimePublishWorld(...)
```

---

### friend

```cpp
friend class RuntimeBuilder;
```

---

### Factory

```cpp
RuntimeBuilder::build(...)
```

のみ生成。

---

### CI

```bash
grep "new RuntimePublishWorld"
grep "make_unique<RuntimePublishWorld>"
```

0件。

---

# Practical Stable ISR Bridge Runtime 完了に向けた推奨実施順

実装効果が大きい順に並べると、

## P1（最優先）

1. Legacy Commit Path 削除
2. Publish決定権をAdmissionへ集約
3. Coordinator Authority化

---

## P2（高優先）

4. RuntimeWorld/DSPCore二重モデル解消
5. Semantic/Execution完全分離
6. Observe Path一本化

---

## P3（中優先）

7. PublicationとDSP Lifetime分離
8. Crossfade Authority一本化
9. RuntimeWorld生成封鎖

この順序で進めると、大規模リファクタリングを避けながら、Practical Stable ISR Bridge Runtime の設計目標に最も近づけます。特に P1 と P2 が完了すると、AudioEngine 主導の旧構造から Runtime Authority 主導構造への移行がほぼ完了します。
