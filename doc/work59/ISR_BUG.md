ConvoPeq は Practical Stable ISR Bridge Runtime の到達度としてはかなり高く、特に

* Retire Authority の単一化（ISRRetireRouter）
* EpochDomain の隠蔽
* Crossfade Authority の単一化
* Reader診断
* Overflow可視化
* Shutdown Audit

は既に実装されています。 

しかし、提示された理想仕様と比較すると、まだいくつか「構造的に未達成」な箇所があります。

以下は優先度順です。

---

# ① Overflowしても失われない（未達成）

## 1. 未達成内容

Practical Stable ISR Bridge Runtime の理想条件では

> Overflow がデータ喪失に直結しない

が要求されています。

しかし ConvoPeq は現在、

* RetireIntent Queue
* Fallback Queue

の両方が満杯になると

**Intent を破棄します。**

---

## 2. 該当コード

RetireRuntime

```cpp
Fallback も満杯 → overflow としてカウント
droppedIntentCount_++
```



DeferredRetireFallbackQueue

```cpp
HardLimit超過→強制ドロップ（リーク許容）
```

```cpp
if (queue_.size() >= kFallbackHardLimit)
    return false;
```



---

## 3. あるべき姿

理想仕様では

```text
Intent Queue
 ↓
Fallback Queue
 ↓
Deferred Queue
 ↓
Quarantine
```

です。

つまり

```text
失う
```

ではなく

```text
隔離する
```

です。

---

## 4. 改修方法

HardLimit到達時

```cpp
return false;
```

ではなく

```cpp
quarantineRetire(entry);
```

へ移送。

新設：

```cpp
RetireQuarantineStore
```

保持：

```cpp
ptr
deleter
epoch
reason
timestamp
```

Shutdown Audit で

```cpp
quarantineResident
```

を必須確認。

---

# ② HealthMonitorが自己回復まで到達していない

## 1. 未達成内容

理想仕様

```text
Detect
↓
Diagnose
↓
Recover
↓
Verify
```



現在の RuntimeHealthMonitor は

ほぼ

```text
Detect
↓
Callback
```

です。

---

## 2. 該当コード

Reader Stuck

```cpp
detectStuckReaders()
```

↓

```cpp
m_callback(ev)
```

のみ。

Retire Stall

```cpp
retirePressureAdmissionStrict_=true
```

↓

PolicyEngine任せ

```cpp
回復は PolicyEngine に委譲
```



---

## 3. あるべき姿

例えば

```text
Reader Stuck
↓
Reader特定
↓
Admission停止
↓
Retire再試行
↓
Epoch前進
↓
Verify
```

まで。

---

## 4. 改修方法

追加：

```cpp
RecoverAction
```

```cpp
ForceReclaim
AdvanceEpoch
SuspendAdmission
DrainDeferredPublish
```

HealthMonitorは

```cpp
HealthEvent
↓
RecoveryPlan
↓
Executor
↓
Verification
```

構造にする。

---

# ③ Shutdown完全Drain保証が未達成

## 1. 未達成内容

理想仕様

```text
Verify Empty
↓
Shutdown Complete
```



現在は

```cpp
Drain incomplete
(observation only)
```

です。

---

## 2. 該当コード

```cpp
Drain incomplete
(observation only)
```



さらに

```cpp
quarantine residents remain
```

でも Shutdown 継続。

---

## 3. あるべき姿

Shutdown 完了条件

```text
pendingPublication = 0
pendingRetire = 0
activeReader = 0
deferredPublish = 0
quarantine = 0
```

---

## 4. 改修方法

追加：

```cpp
ShutdownVerificationResult
```

```cpp
Complete
Incomplete
Forced
```

ShutdownTrace に保存。

---

# ④ Reader Stuckは検出できるが解消権限がない

## 1. 未達成内容

Reader Stuck診断はかなり高度です。

```cpp
ownerTag
ownerThreadId
residencyTimeUs
```

まで持っています。

しかし

```text
誰が塞いでいるか
```

は分かるが

```text
どう解消するか
```

がありません。

---

## 2. 該当コード

```cpp
detectStuckReaders()
```



---

## 3. あるべき姿

```text
Reader Stuck
↓
RetireBlockerSnapshot
↓
RecoveryPolicy
↓
Verify
```

---

## 4. 改修方法

追加：

```cpp
RetireBlockerSnapshot
```

に

```cpp
threadName
ownerTag
readerAge
```

を保持。

HealthEvent発生時に Evidence 出力。

---

# ⑤ RuntimeWorld Immutable が100%ではない

## 1. 未達成内容

設計上は

```text
RuntimeWorld immutable
```

ですが、

実際には

```cpp
setHealthStateRef(...)
```

等の外部参照注入が Build 時に残っています。

また RuntimeWorld に対して

```cpp
resource
dspProjection
```

を段階的に埋めています。

---

## 2. あるべき姿

```text
Build
↓
Freeze
↓
Publish
```

のみ。

---

## 3. 改修方法

導入：

```cpp
FrozenRuntimeWorld
```

```cpp
validate()
freeze()
publish()
```

を強制。

Publish後は

```cpp
const RuntimeWorld
```

のみ扱う。

---

# ⑥ Retire Authority はほぼ達成だが完全ではない

## 1. 未達成内容

理想では

```text
Coordinator が唯一 Authority
```

です。

現在は

```cpp
DSPLifetimeManager
```

が

```cpp
router_->enqueueRetire(...)
```

を直接呼んでいます。

---

## 2. あるべき姿

```text
DSPLifetimeManager
↓
RetireCoordinator
↓
ISRRetireRouter
↓
EpochDomain
```

---

## 3. 改修方法

新設：

```cpp
RuntimeRetireCoordinator
```

責務

```cpp
retire admission
priority
overflow routing
shutdown gating
```

---

# ⑦ Shutdown Authority に Quarantine が含まれていない

## 1. 未達成内容

RuntimeDrainAudit は

```cpp
quarantineResident
```

を

```cpp
監査のみ
```

扱いにしています。

---

## 2. あるべき姿

理想仕様では

```text
Quarantine
↓
Delete
```

なので

Shutdown成功条件に含めるべきです。

---

## 3. 改修方法

変更：

```cpp
BlockingReason::QuarantineResident
```

を

```cpp
Audit only
```

から

```cpp
Shutdown blocker
```

へ昇格。

---

# 総合評価

Practical Stable ISR Bridge Runtime の観点で見ると現在の ConvoPeq はおおよそ

| 項目                     | 達成度     |
| ---------------------- | ------- |
| RTでlock禁止              | 98〜100% |
| RTでdelete禁止            | 100%    |
| Retire→Epoch経由         | 100%    |
| Crossfade Authority単一化 | 95〜100% |
| Reader Tracking        | 95%     |
| Publication Pipeline   | 90〜95%  |
| Shutdown Drain         | 85〜90%  |
| Overflow耐性             | 75〜80%  |
| Self Recovery          | 70〜80%  |
| Quarantine運用           | 60〜70%  |

です。

最も大きい未達成点は

**「Overflow時にまだデータ喪失が起こり得ること」**
と
**「HealthMonitorがRecover→Verifyまで責任を持っていないこと」**
です。これらは Practical Stable ISR Bridge Runtime の最終到達点との差として最も大きい部分です。
