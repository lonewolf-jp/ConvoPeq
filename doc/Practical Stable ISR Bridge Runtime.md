Practical Stable ISR Bridge Runtime を一言で表現すると、

> 「RTスレッドは絶対に待たない。絶対に解放しない。絶対に判断しない。すべての危険操作はNonRTへ橋渡し(Bridge)し、状態遷移は観測可能で、停止時には完全排水(Drain)を保証する」

という構造です。

ConvoPeqは既にかなり近い位置まで来ていますが、理想形はさらに厳密です。

---

# 1. 全体アーキテクチャ

理想構造

```text
┌─────────────────────────────┐
│ UI Thread                   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Runtime Publication Layer   │
│                             │
│ Orchestrator                │
│ Coordinator                 │
│ Admission                   │
└────────────┬────────────────┘
             │
             ▼
══════════ ISR Bridge ══════════
             │
             ▼
┌─────────────────────────────┐
│ Audio Thread (RT)           │
│                             │
│ read only                   │
│ no lock                     │
│ no malloc                   │
│ no delete                   │
│ no decision                 │
└────────────┬────────────────┘
             │ retire intent
             ▼
┌─────────────────────────────┐
│ Retire Runtime              │
│ Intent Queue                │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Retire Coordinator          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Epoch Domain                │
│                             │
│ Retire Queue                │
│ Reader Tracking             │
│ Reclaim                     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Deferred Deletion           │
└─────────────────────────────┘
```

---

# 2. RTスレッドの責務

理想的なRTスレッドは

```text
読むだけ
```

です。

---

絶対禁止

```cpp
new
delete
malloc
free
mutex
condition_variable
future
shared_ptr destruction
filesystem
logging
```

---

許可

```cpp
atomic load
pointer swap
DSP process
```

のみ。

---

理想形

```cpp
processBlock()
{
    RuntimeWorld* world = currentWorld.load();

    world->dsp->process();

    if (retireNeeded)
        emitRetireIntent();
}
```

---

ここで重要なのは

RTは

```cpp
retire()
```

しないことです。

---

# 3. ISR Bridgeとは何か

ISR Bridgeは

```text
RT世界
↓
NonRT世界
```

の境界です。

---

理想構造

```text
RT
 ↓
RetireIntent
 ↓
Coordinator
 ↓
Epoch
 ↓
Delete
```

です。

---

RTは

```cpp
delete ptr;
```

しない。

---

代わりに

```cpp
emitRetireIntent(ptr);
```

だけ行う。

---

# 4. Publication Pipeline

理想構造

```text
Build
 ↓
Validate
 ↓
Admission
 ↓
Publish
 ↓
Observe
 ↓
Retire Old
```

---

ConvoPeqは既に

Admission

Coordinator

Orchestrator

を持っています。

しかし理想形ではさらに

```text
Publish Success
↓
Observe Complete
↓
Retire Authority Granted
```

になります。

---

# 5. Retire Pipeline

理想形

```text
Active
 ↓
PendingRetire
 ↓
RetireQueued
 ↓
EpochProtected
 ↓
ReclaimReady
 ↓
Deleted
```

です。

---

図

```text
Runtime
   │
   ▼
PendingRetire
   │
   ▼
RetireQueue
   │
   ▼
Epoch Waiting
   │
   ▼
Reclaim
   │
   ▼
Delete
```

---

重要なのは

```text
Delete
```

が最後であること。

---

# 6. EpochDomainの役割

EpochDomainは

RCUシステムの心臓です。

---

理想構造

```text
Reader A epoch=100

Reader B epoch=105

Reader C epoch=102

Current epoch=110

minReaderEpoch=100
```

---

Retire対象

```text
retireEpoch=101
```

なら

まだ削除禁止。

---

理由

```text
Reader A
```

が見ている可能性がある。

---

削除可能条件

```cpp
retireEpoch < minReaderEpoch
```

---

このルールが破れると

UAFになります。

---

# 7. Reader管理

理想構造

```text
register
enter
exit
release
```

が完全に対になる。

---

図

```text
register
   │
   ▼
enter
   │
   ▼
critical section
   │
   ▼
exit
```

---

異常系

```text
enter
↓
exit忘れ
```

↓

```text
epoch停止
```

↓

```text
reclaim停止
```

↓

```text
メモリ増殖
```

---

そのため監視が必要。

---

# 8. Runtime Publication Coordinator

理想的には

Coordinatorは

```text
唯一のAuthority
```

です。

---

禁止

```cpp
RuntimeAが直接Publish

RuntimeBが直接Retire
```

---

許可

```cpp
Coordinator.publish()
Coordinator.retire()
```

のみ。

---

図

```text
Builder
  │
  ▼
Coordinator
  │
 ├─ Publish
 └─ Retire
```

---

# 9. Runtime Health Monitor

理想形は

監視だけではない。

---

現在

```text
Detect
```

まで。

---

理想

```text
Detect
 ↓
Diagnose
 ↓
Recover
 ↓
Verify
```

---

例

```text
Retire Stall
```

↓

```text
Force Reclaim
```

↓

```text
Verify
```

---

例

```text
Publication Stall
```

↓

```text
Retry Publish
```

↓

```text
Verify
```

---

# 10. Overflow管理

実運用で最も重要

---

理想構造

```text
Intent Queue
       │
       ▼
 Queue Full
       │
       ▼
 Fallback Queue
       │
       ▼
 Deferred Queue
       │
       ▼
 Quarantine
```

---

ConvoPeqには

RetireIntent Queue

Overflow Counter

がありますが、

理想形は

```text
Overflowしても失われない
```

です。

---

# 11. Shutdown Pipeline

ここが最重要です。

---

理想構造

```text
Stop Accepting
        │
        ▼
Stop Publish
        │
        ▼
Stop Reader
        │
        ▼
Drain Intent
        │
        ▼
Drain Retire
        │
        ▼
Advance Epoch
        │
        ▼
Reclaim
        │
        ▼
Verify Empty
        │
        ▼
Shutdown Complete
```

---

図

```text
┌─────────────┐
│ Accept Work │
└──────┬──────┘
       ▼
┌─────────────┐
│ Stop Input  │
└──────┬──────┘
       ▼
┌─────────────┐
│ Drain Queue │
└──────┬──────┘
       ▼
┌─────────────┐
│ Reclaim     │
└──────┬──────┘
       ▼
┌─────────────┐
│ Verify      │
└──────┬──────┘
       ▼
┌─────────────┐
│ Exit        │
└─────────────┘
```

---

# 12. Practical Stable ISR Bridge Runtime の最終形

理想的なConvoPeqは以下の状態になります。

```text
                      ┌─────────────┐
                      │ UI          │
                      └──────┬──────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ Publication        │
                  │ Orchestrator       │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Coordinator        │
                  └─────────┬──────────┘
                            │
                publish     │     retire
                            │
                            ▼
══════════════════════════════════════════
               ISR Bridge
══════════════════════════════════════════
                            │
                            ▼
                  ┌────────────────────┐
                  │ Audio Thread       │
                  │ read only          │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Retire Intent      │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Epoch Domain       │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Reclaim            │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Delete             │
                  └────────────────────┘
```

この構造の本質は、

**RTスレッドを「実行主体」ではなく「観測主体」に限定し、すべての寿命管理・状態遷移・障害回復をNonRT側へ隔離すること**

です。

Practical Stable ISR Bridge Runtime の成熟度を測る指標としては、

1. RTで delete が発生しない
2. RTで lock が発生しない
3. Retire が必ず Epoch を通る
4. Shutdown が完全 Drain を保証する
5. Overflow がデータ喪失に直結しない
6. HealthMonitor が自己回復可能
7. Coordinator が唯一の Authority

の7条件をすべて満たしていることが、実運用で破綻しにくい到達点になります。



Practical Stable ISR Bridge Runtime を長期的に維持するためには、「個別のバグを潰す」よりも先に、**破ってはいけない設計不変条件（Architectural Invariants）** を明文化しておくことが重要です。

ConvoPeqの現状構造（RuntimeWorld / Snapshot / Crossfade / RCU / Publish Path）を前提にすると、今後の改修では最低でも以下を守るべきです。

---

# 1. RTスレッド不変条件

## ISR-RT-001

### RTスレッドは状態を決定しない

禁止：

```cpp
if (needCrossfade())
```

```cpp
if (shouldRetire())
```

```cpp
if (policy.requiresXXX())
```

RTは

```cpp
state = consumeAtomic(...)
execute(state);
```

のみ。

---

## ISR-RT-002

### RTスレッドはメモリ所有権を持たない

禁止：

```cpp
delete world;
```

```cpp
unique_ptr.reset();
```

```cpp
shared_ptr
```

---

## ISR-RT-003

### RTスレッドは動的確保禁止

禁止：

```cpp
new
delete
malloc
free
std::vector::push_back
std::string
```

---

## ISR-RT-004

### RTスレッドはログ出力禁止

禁止：

```cpp
DBG(...)
Logger::writeToLog(...)
```

---

# 2. RuntimeWorld不変条件

## ISR-WORLD-001

### RuntimeWorldはImmutable

Publish後禁止：

```cpp
world->xxx = ...
```

---

## ISR-WORLD-002

### RuntimeWorldは完全構築後にPublish

禁止：

```cpp
publish(world);

world->initSomething();
```

必ず

```cpp
build
validate
freeze
publish
```

---

## ISR-WORLD-003

### Publish後の修正経路を作らない

禁止：

```cpp
world->cache
```

```cpp
mutable
```

---

## ISR-WORLD-004

### RuntimeWorldは単一所有者

生成：

```cpp
WorldBuilder
```

公開：

```cpp
RuntimeStore
```

破棄：

```cpp
RetireManager
```

---

# 3. Snapshot不変条件

## ISR-SNAP-001

### Snapshotは読み取り専用

禁止：

```cpp
snapshot->modify()
```

---

## ISR-SNAP-002

### Snapshot生成元は1箇所

理想：

```cpp
SnapshotFactory
```

のみ。

---

## ISR-SNAP-003

### SnapshotとRuntimeWorldの差異を作らない

禁止：

```cpp
Snapshot独自状態
```

---

## ISR-SNAP-004

### Snapshotは寿命管理しない

禁止：

```cpp
snapshot.releaseWorld();
```

---

# 4. Crossfade不変条件

## ISR-XF-001

### Crossfade判定箇所は1箇所

禁止：

```cpp
CrossfadeAuthority
```

と

```cpp
HealthMonitor
```

の両方が判定

---

## ISR-XF-002

### CrossfadeはPure Function

理想：

```cpp
Decision evaluate(
    oldWorld,
    newWorld,
    policy);
```

---

## ISR-XF-003

### DSP状態を参照して判定しない

禁止：

```cpp
engine.currentValue
```

依存

---

## ISR-XF-004

### Crossfade DurationはPolicyのみ

禁止：

```cpp
duration=max(a,b,c,d)
```

---

# 5. Publish不変条件

## ISR-PUB-001

### Publish経路は1本

禁止：

```cpp
publishA()
publishB()
publishC()
```

---

## ISR-PUB-002

### Publish前Validation必須

順序固定：

```cpp
Build
↓
Validate
↓
Publish
```

---

## ISR-PUB-003

### Validatorは空実装禁止

禁止：

```cpp
return true;
```

---

## ISR-PUB-004

### Publish後のRollback禁止

Publish後は

```cpp
Retire
```

のみ。

---

# 6. Retire不変条件

## ISR-RET-001

### retire判定は1箇所

---

## ISR-RET-002

### retireとdeleteを分離

禁止：

```cpp
retire(world);
delete world;
```

同時実行

---

## ISR-RET-003

### RetireQueueのみが寿命を管理

禁止：

```cpp
Observerがdelete
```

---

## ISR-RET-004

### World破棄はDeferred

即時破棄禁止

---

# 7. Observer不変条件

## ISR-OBS-001

### Observerは副作用禁止

許可：

```cpp
metrics
logging
telemetry
```

禁止：

```cpp
publish
retire
crossfade変更
```

---

## ISR-OBS-002

### Observerは所有権禁止

禁止：

```cpp
shared_ptr<RuntimeWorld>
```

---

# 8. HealthMonitor不変条件

## ISR-HM-001

### Monitorは観測のみ

禁止：

```cpp
monitor→policy変更
```

---

## ISR-HM-002

### MonitorはWorld変更禁止

---

## ISR-HM-003

### MonitorはCrossfade変更禁止

---

# 9. Validation不変条件

## ISR-VAL-001

### Validation失敗時はPublish禁止

---

## ISR-VAL-002

### Validationは決定論的

禁止：

```cpp
現在CPU使用率で結果変化
```

---

## ISR-VAL-003

### Validationは副作用禁止

---

# 10. Atomic運用不変条件

## ISR-ATM-001

### atomic直接呼び出し禁止

ConvoPeqの方針通り

禁止：

```cpp
.load()
.store()
.exchange()
```

---

## ISR-ATM-002

### Wrapper経由のみ

```cpp
consumeAtomic()
publishAtomic()
```

---

## ISR-ATM-003

### MemoryOrder統一

各開発者が独自指定しない

---

# 11. Architectural Reviewで必ず確認する項目

PRレビュー時に以下を毎回確認する。

### 追加されたコードが

* RTでメモリ確保していないか
* RTでdeleteしていないか
* Publish経路を増やしていないか
* Crossfade判定箇所を増やしていないか
* RuntimeWorldを書き換えていないか
* Snapshotを書き換えていないか
* Observerに副作用がないか
* retire経路を増やしていないか
* Validatorを迂回していないか
* atomic直接呼び出ししていないか

---

# 最重要の3原則

実運用で破綻しにくい Practical Stable ISR Bridge Runtime を維持するうえで特に重要なのは次の3つです。

### 原則1

**RuntimeWorldは Publish 後完全Immutable**

---

### 原則2

**RTスレッドは実行のみ、判断しない**

---

### 原則3

**Publish / Crossfade / Retire の決定権はそれぞれ1箇所のみ**

この3原則が守られている限り、ConvoPeqは今後大規模改修を行っても構造的破綻を起こしにくい ISR Bridge Runtime を維持しやすくなります。



以下は単なる理想論ではなく、

**「RT安全性」**
**「長期保守性」**
**「将来機能追加耐性」**
**「障害時の局所化」**

を同時に満たすことを目的とした、

**Practical Stable ISR Bridge Runtime版 ConvoPeq 理想アーキテクチャ**

です。

---

# 全体思想

ConvoPeqは本質的には

```text
UI
 ↓
Intent
 ↓
Build
 ↓
Validate
 ↓
Publish
 ↓
Crossfade
 ↓
Retire
 ↓
Delete
```

だけです。

この流れ以外を作らないことが重要です。

---

# 全体構造

```text
┌─────────────────────┐
│ UI Thread           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Intent Layer        │
│ Parameter Changes   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Runtime Builder     │
│ WorldBuilder        │
└─────────┬───────────┘
          │ RuntimeWorld
          ▼
┌─────────────────────┐
│ Validation Layer    │
│ PublicationValidator│
└─────────┬───────────┘
          │ OK
          ▼
┌─────────────────────┐
│ Policy Engine       │
│  ├ Publication      │
│  ├ Crossfade        │
│  └ Retire           │
└─────────┬───────────┘
          │ Decision
          ▼
┌─────────────────────┐
│ Publication Layer   │
│ RuntimeStore        │
└─────────┬───────────┘
          │ Atomic Swap
          ▼
┌─────────────────────┐
│ Crossfade Runtime   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Retire Queue        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Deferred Deletion   │
└─────────────────────┘
```

---

# 1. Intent Layer

役割

```text
UI変更
Automation
Preset
MIDI
Host Automation
```

を

```text
Runtime Intent
```

へ変換する。

---

理想

```cpp
struct RuntimeIntent
{
    ParameterDelta delta;
};
```

ここでは

禁止

```cpp
Crossfade判定
World変更
DSP変更
```

---

# 2. Runtime Builder

責務

唯一の

```text
RuntimeWorld生成器
```

---

理想

```cpp
RuntimeWorldBuilder
```

のみが

```cpp
RuntimeWorld
```

を作る。

---

フロー

```text
Intent
 ↓
Build Context
 ↓
DspGraph
 ↓
ConvolutionConfig
 ↓
Projection
 ↓
RuntimeWorld
```

---

禁止

```cpp
publish
retire
crossfade
```

---

# 3. RuntimeWorld

ConvoPeqの中心

---

理想

```cpp
struct RuntimeWorld
{
    const DspGraph graph;

    const Routing routing;

    const Projection projection;

    const Metadata metadata;
};
```

---

重要

Publish後

```cpp
変更不可
```

---

つまり

```cpp
mutable
cache
lazy init
```

禁止

---

# 4. Validation Layer

Builderの次

---

理想

```cpp
RuntimePublicationValidator
```

---

実施項目

## Topology

```text
循環チェック
```

---

## Resource

```text
IR存在
メモリサイズ
```

---

## Semantic

```text
Channel一致
SampleRate整合
```

---

## Runtime

```text
Crossfade設定
Projection整合
```

---

結果

```cpp
ValidationResult
```

---

# 5. Policy Engine

ここが最重要

---

ConvoPeqで

決定権を持つ場所

---

```cpp
class RuntimePolicyEngine
{
};
```

---

内部

```cpp
PublicationPolicy
CrossfadePolicy
RetirePolicy
```

---

## Publication

```cpp
publishして良いか
```

---

## Crossfade

```cpp
Crossfade必要か
```

---

## Retire

```cpp
いつ破棄するか
```

---

重要

ここ以外で

```cpp
if (needCrossfade)
```

禁止

---

# 6. RuntimeStore

現在世界の唯一の保持者

---

理想

```cpp
class RuntimeStore
{
    std::atomic<RuntimeWorld*>
        activeWorld;
};
```

---

唯一許可される

Publish

```cpp
publishWorld()
```

---

フロー

```text
newWorld
 ↓
atomic swap
 ↓
oldWorld取得
 ↓
RetireQueue
```

---

# 7. Crossfade Runtime

責務

DSP実行のみ

---

入力

```cpp
CrossfadePlan
```

---

出力

```cpp
AudioBuffer
```

---

禁止

```cpp
Crossfade判定
```

---

つまり

```cpp
Decision
```

は持たない。

---

# 8. RT Thread

理想

```text
Read
 ↓
Execute
 ↓
Output
```

のみ。

---

完全図

```text
activeWorld
      ↓
read atomic
      ↓
process
      ↓
output
```

---

禁止

```cpp
build
validate
retire
delete
crossfade判定
```

---

# 9. Retire Manager

Publish後の旧World管理

---

理想

```cpp
RetireQueue
```

のみ。

---

フロー

```text
oldWorld
 ↓
RetireQueue
 ↓
RCU Safe Point
 ↓
DeletionQueue
```

---

# 10. Deferred Deletion

最後の責務

---

理想

```cpp
DeletionWorker
```

---

ここだけが

```cpp
delete RuntimeWorld
```

できる。

---

禁止

```cpp
RuntimeStore
Observer
Crossfade
```

からのdelete

---

# 11. Monitoring

重要

監視は監視だけ

---

理想

```cpp
RuntimeHealthMonitor
```

---

許可

```cpp
metrics
telemetry
statistics
```

---

禁止

```cpp
publish
retire
crossfade変更
```

---

# 理想フロー図（完成形）

```text
UI
 │
 ▼
Intent
 │
 ▼
RuntimeWorldBuilder
 │
 ▼
RuntimeWorld
 │
 ▼
PublicationValidator
 │
 ▼
RuntimePolicyEngine
 │
 ├─ PublicationDecision
 ├─ CrossfadeDecision
 └─ RetireDecision
 │
 ▼
RuntimeStore
 │
 ├─ activeWorld swap
 │
 └─ oldWorld
       │
       ▼
   RetireQueue
       │
       ▼
   DeferredDeletion
       │
       ▼
      delete
```

RT側は完全に分離され、

```text
Audio Thread
      │
      ▼
 RuntimeStore
      │
      ▼
 activeWorld
      │
      ▼
 process()
      │
      ▼
 output()
```

のみになります。

---

Practical Stable ISR Bridge Runtime の観点では、この構造の本質は次の一文に集約されます。

> **Build・Validate・Decide・Publish・Retire・Delete の責務を完全分離し、RTスレッドを「RuntimeWorldの消費者」に限定する。**

ConvoPeqが将来的にIR管理、マルチインスタンス、非同期ローダー、プリセットシステム、高度なクロスフェード戦略を追加しても、この責務分離を崩さなければ構造的な破綻は極めて起こりにくくなります。
