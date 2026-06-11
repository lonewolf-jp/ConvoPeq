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
