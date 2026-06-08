最新版の ConvoPeq.md を前提に、「数学的な理想性」ではなく **実運用で破綻しにくい Practical Stable ISR Bridge Runtime** という観点で評価します。

結論から言うと、

> **Practical Stable ISR Bridge Runtime の骨格はかなり完成度が高い段階に到達している**
>
> しかし、
>
> **「完全なライフサイクル閉包」**
> **「強制停止時の最終回収保証」**
> **「世代再利用防止」**
> **「Deferred Publish の運用耐性」**
>
> の4領域はまだ実運用上の破綻余地が残っています。

---

# 現在達成できているもの

## 1. Reader保護の統一

従来の EpochDomainReaderGuard 系から、

* RCUReader
* RCUReaderGuard
* RuntimeReaderContext
* ObserveToken

へ統一されている。  

これは実運用上かなり重要です。

以前存在した

* Reader系統A
* Reader系統B

の二重管理は

* retire判定ズレ
* epoch advanceズレ

の原因になります。

現在は

```cpp
RCUReader
↓
IReaderEpochProvider
↓
ISRRetireRouter
↓
EpochDomain
```

という単一路線になっている。  

これは良い状態です。

---

# 2. Publish Authority の分離

現在は

```cpp
PublicationAdmission
↓
RuntimePublicationOrchestrator
↓
PublicationExecutor
↓
RuntimePublicationCoordinator
```

になっている。   

特に

```cpp
evaluate()
```

強制通過になっている。 

実運用では

「誰かが勝手に publish」

が最大事故要因なので、

Admission 強制化は非常に有効です。

---

# 3. Semantic Validation

以下の契約テストが存在する。

* RuntimePublicationCoordinatorTests
* ISRSemanticValidationTests
* RuntimeSemanticSchemaValidationTests
* PartialPublicationRejectTests
* RuntimeWorldAuthorityProjectionTests

など。 

さらに

```cpp
semanticComplete
publicationSequence
mappedRuntimeGeneration
executionTransitionPolicy
```

等の検証が実装されている。 

これは

「壊れた RuntimeWorld が publish される」

事故をかなり防げています。

---

# 4. Retire Intent Queue

現在

```cpp
RetireRuntime
```

が

```cpp
emitRetireIntentRT()
```

↓

```cpp
dequeuePendingRetireIntents()
```

という構造を持っている。 

これは

RT
→
NonRT

片方向伝達として正しい。

実運用上の思想としても良いです。

---

# 5. Shutdown FSM

ShutdownRuntime が存在する。 

段階も

```cpp
Running
AudioStopped
ObserverDrained
RetireClosed
EpochSettled
ReclaimComplete
ShutdownComplete
```

となっている。 

Practical Runtimeとして必要な

* Late Callback監視
* Post Stop Enqueue監視

もある。 

かなり実戦的です。

---

# しかし未達成な部分

---

# 問題1

## DSP Quarantine が実質フラグ管理のみ

現在の実装

```cpp
std::vector<std::atomic<bool>>
```

だけです。 

つまり

```cpp
quarantineHandle()
```

↓

```cpp
flag=true
```

しかやっていない。 

---

実運用で必要なのは

```text
Quarantine
 ↓
Reclaim禁止
 ↓
Generation禁止
 ↓
Publish禁止
 ↓
調査
 ↓
強制破棄
```

です。

現在は

単なる状態ビット。

---

## 推奨

Quarantineを

```cpp
slot
generation
reason
timestamp
```

保持する管理構造へ昇格。

最低でも

```cpp
isQuarantined(slot,generation)
```

が必要。

---

# 問題2

## generation 再利用防御が弱い

現在

```cpp
DSPHandle
slot
generation
```

へ進んでいる。 

これは正しい。

しかし検索結果上、

世代番号の wrap 対策が見えません。

---

実運用では

```cpp
slot=12
generation=17
```

のDSPが retire。

数日後

```cpp
slot=12
generation=17
```

再出現。

すると

古い参照が蘇る。

---

Practical Runtimeでは

最低でも

```cpp
uint64_t generation
```

か

```cpp
epoch stamp
```

を持たせたい。

---

# 問題3

## Deferred Publish が 1 件しか保持できない

現在

```cpp
std::optional<PublishRequest>
```

です。 

つまり

```cpp
Request A
Request B
Request C
```

が来ると

最終的に

```cpp
C
```

しか残らない。

---

これは設計としては理解できます。

ただし実運用で

IRロード連打

EQ変更連打

プリセット変更連打

が発生すると、

途中状態が消えます。

---

推奨

少なくとも

```cpp
latest-generation-wins
```

を明示契約化。

あるいは

```cpp
ring queue 4～8件
```

化。

---

# 問題4

## Publish成功後の Reclaim 完結保証が見えない

現状

```cpp
publish
↓
advanceRetireEpoch()
```

はある。 

しかし

実運用で重要なのは

```text
Publish
 ↓
RetireIntent
 ↓
Coordination
 ↓
Epoch
 ↓
Reclaim
 ↓
Memory Freed
```

が100%完了すること。

---

現在

```cpp
RetireLane
{
 RTIntent,
 Coordination,
 Epoch,
 Reclaim,
 Quarantine
}
```

はある。 

しかし

「全レーン収束確認」

が見えません。

---

## 推奨

Shutdown直前に

```cpp
assert(
 pendingIntent==0 &&
 coordination==0 &&
 reclaim==0
);
```

相当の監査を追加。

---

# 問題5

## RetireRuntime Queue Overflow

現在

```cpp
overflowCount_
droppedIntentCount_
```

は存在する。 

しかし

実運用で重要なのは

Overflow発生時の処理です。

---

例えば

```cpp
overflowCount > 0
```

になった瞬間

```cpp
retirePressurePublicationThrottleActive_
```

を強制有効化。

あるいは

```cpp
ShutdownSafeMode
```

へ移行。

---

現状は

計測のみで終わっているように見えます。 

---

# 問題6

## Shutdown FSM と PublicationAdmission の連携はあるが完全閉包は未確認

Admission側は

```cpp
isShutdownInProgress()
```

を見て reject。 

これは良い。

しかし

実運用では

```text
shutdown開始
↓
deferred publish残存
↓
transition完了
↓
再publish
```

が事故になります。

---

現在

```cpp
notifyTransitionComplete()
```

で

deferred再投入する実装がある。 

Shutdown中の再投入防止が明示的に見えません。

---

# 総合評価

Practical Stable ISR Bridge Runtime の完成度を数値化すると

| 項目                      | 達成度 |
| ----------------------- | --- |
| Reader統一                | 95% |
| Publication Authority   | 95% |
| Semantic Validation     | 95% |
| Crossfade制御             | 90% |
| Retire Intent           | 85% |
| Shutdown FSM            | 85% |
| Quarantine              | 60% |
| Generation Safety       | 70% |
| Final Reclaim Guarantee | 75% |
| Overflow Recovery       | 65% |

総合では

**約85〜90%完成**

です。

現在のコードは「設計実験段階」は既に脱しており、実運用可能なISR Runtimeの形になっています。残る大きな課題はアルゴリズムではなく、

* Quarantineの実体化
* Generation再利用防止
* Reclaim完結保証
* Overflow時の自己防衛
* Shutdown完全閉包

という「長時間運転時の破綻防止機構」です。これらを埋めると、Practical Stable ISR Bridge Runtimeとしてかなり完成度の高い状態になります。



ConvoPeq の最新版を見る限り、前回指摘した項目のうち、特に優先度が高いのは以下の順です。

1. Quarantine実体化
2. Shutdown完全閉包
3. Reclaim完結保証
4. Overflow自己防衛
5. Generation強化
6. Deferred Publish改善

理由は、実運用で実際にクラッシュやメモリリークへ直結するのは 1～4 だからです。

---

# 1. Quarantine実体化

現在は実質的に

```cpp
std::vector<std::atomic<bool>> quarantineFlags_;
```

によるフラグ管理です。 

また DSPHandleRuntime 側も

```cpp
void quarantine(DSPHandle handle)
{
    state = DSPState::Quarantined;
}
```

のみです。 

これは「隔離」ではなく「状態変更」です。

---

## 推奨構造

新規:

```cpp
enum class QuarantineReason
{
    GenerationMismatch,
    ResolveFailure,
    PublishViolation,
    CrossfadeViolation,
    ShutdownViolation,
    Unknown
};

struct QuarantineEntry
{
    uint32_t slot;
    uint32_t generation;

    QuarantineReason reason;

    uint64_t quarantineEpoch;
    uint64_t quarantineTimestampUs;

    bool reclaimAllowed;
};
```

---

## DSPQuarantineManager拡張

```cpp
class DSPQuarantineManager
{
private:
    std::unordered_map<uint32_t, QuarantineEntry> entries_;
};
```

---

## create()禁止

現在 create() は

```cpp
state == Reclaimed
```

だけ見ています。 

ここへ

```cpp
if (quarantineManager_.isQuarantined(slot))
    continue;
```

追加。

---

## resolve()強化

```cpp
if (quarantineManager_.isQuarantined(
        handle.slot,
        handle.generation))
{
    return {nullptr,false,false};
}
```

追加。

---

## 効果

これで

```text
異常DSP
 ↓
隔離
 ↓
再利用禁止
 ↓
管理者確認
 ↓
解放
```

になる。

実運用耐性はかなり上がります。

---

# 2. Shutdown完全閉包

ここが実は最重要です。

---

現在の危険

Shutdown開始後に

```text
Deferred Publish
Crossfade Completion
Transition Callback
```

が飛んでくる可能性があります。

---

## RuntimeState追加

```cpp
enum class RuntimeState
{
    Running,
    ShutdownRequested,
    ShutdownDraining,
    ShutdownClosed,
    ShutdownComplete
};
```

---

## PublicationAdmission

現在

```cpp
isShutdownInProgress()
```

判定があります。

ここを

```cpp
if (runtimeState >= ShutdownRequested)
    reject;
```

へ変更。

---

## Deferred Publish再投入禁止

現在

```cpp
notifyTransitionComplete()
```

系で再投入されるなら、

そこへ

```cpp
if (runtimeState != Running)
{
    deferredPublish_.reset();
    return;
}
```

を追加。

---

## Shutdown監査

Shutdown最終段階で

```cpp
struct RuntimeDrainAudit
{
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfade;
    uint64_t pendingDeletion;
};
```

作成。

---

終了条件

```cpp
audit.pendingPublication == 0
audit.pendingRetire == 0
audit.activeCrossfade == 0
audit.pendingDeletion == 0
```

全部成立。

成立しなければ

```cpp
ShutdownIncomplete
```

を記録。

---

# 3. Reclaim完結保証

RetireRuntimeEx は

```text
RTIntent
↓
Coordination
↓
Epoch
↓
Reclaim
```

を持っています。

しかし

「最後に解放されたか」

の保証が弱い。

---

## Reclaim Ticket導入

```cpp
struct ReclaimTicket
{
    uint32_t slot;
    uint32_t generation;

    bool retireCompleted;
    bool epochSettled;
    bool reclaimExecuted;
};
```

---

## retire時

```cpp
ticket.retireCompleted=true;
```

---

## epoch完了

```cpp
ticket.epochSettled=true;
```

---

## reclaim実行

```cpp
ticket.reclaimExecuted=true;
```

---

## Shutdown検査

```cpp
for(ticket)
{
    assert(ticket.retireCompleted);
    assert(ticket.epochSettled);
    assert(ticket.reclaimExecuted);
}
```

---

これで

```text
Retiredだけど解放されない
```

が見つかる。

---

# 4. Overflow自己防衛

現在は

```cpp
overflowCount_
droppedIntentCount_
```

中心です。

計測しかしていません。

---

## QueuePressureMode導入

```cpp
enum class RuntimePressureState
{
    Normal,
    Warning,
    Critical
};
```

---

## 閾値

例

```cpp
75%
90%
```

---

## Warning

```cpp
publish interval x2
```

---

## Critical

```cpp
new publish reject
crossfade reject
rebuild reject
```

---

## Recovery

```cpp
queue depth < 25%
```

で復帰。

---

これは DAW の長時間運転ではかなり効きます。

---

# 5. Generation強化

現在

```cpp
uint32_t generation
```

です。

---

## 推奨

単純に

```cpp
uint64_t generation;
```

へ変更。

---

さらに

```cpp
struct DSPHandle
{
    uint32_t slot;
    uint64_t generation;
};
```

へ変更。

---

## 理由

32bitでも理論上十分ですが、

Practical Runtimeでは

「考えなくてよい」

ことが重要です。

64bitなら事実上wrapを無視できます。

---

# 6. Deferred Publish改善

もし現在

```cpp
std::optional<PublishRequest>
```

で単一保持なら、

実運用向けには

---

## 推奨構造

```cpp
struct DeferredPublishSlot
{
    PublishRequest request;
    uint64_t publicationSequence;
};
```

---

```cpp
std::deque<DeferredPublishSlot>
```

最大4件。

---

## 溢れた場合

最古削除。

```cpp
A
B
C
D
E
```

↓

```cpp
B
C
D
E
```

---

## さらに推奨

同一ターゲットなら統合。

```cpp
IR load #1
IR load #2
IR load #3
```

↓

```cpp
IR load #3
```

だけ残す。

---

# 私なら実装順序をどうするか

実運用安定性だけを目的にするなら、

### Phase A（最優先）

* Quarantine実体化
* Shutdown完全閉包

---

### Phase B

* Reclaim完結保証
* Overflow自己防衛

---

### Phase C

* Generation 64bit化
* Deferred Publish多段化

です。

特に ConvoPeq の現状コードを見る限り、Generation の32→64bit変更よりも、Shutdown完全閉包とReclaim監査のほうが実運用上の効果ははるかに大きいです。現在のアーキテクチャで実際に長時間運転時の事故要因になりやすいのは、世代wrapではなく「Shutdown途中の残留タスク」と「回収未完了の見逃し」です。
