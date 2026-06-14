Practical Stable ISR Bridge Runtime の場合、単純な

```text
異常検出
↓
HealthState=Critical
↓
Admission停止
```

では不十分です。

なぜなら ISR Runtime では、

```text
異常の種類
↓
システムへの影響
↓
許容可能な劣化動作
↓
最終停止
```

という段階制御が必要だからです。

現在の ConvoPeq は

```text
Diagnostic
↓
HealthState
↓
Admission Reject
```

しかありません。

私なら以下の Policy Layer を追加します。

---

# Practical Stable ISR Escalation Policy Layer

## Layer 0

Diagnostic

現在の

```cpp
WorldLifecycleAudit
RuntimeDrainAudit
RuntimeHealthMonitor
```

がここ。

まだ制御権を持たない。

例

```text
doubleRetireCount
routerPendingRetire
readerResidencyTime
overflowRate
worldConsistency
```

---

# Layer 1

Policy Evaluation

新設

```cpp
RuntimePolicyEngine
```

---

ここで

```cpp
Diagnostic Event
↓
Severity
↓
Persistence
↓
Blast Radius
```

を評価する。

---

## Severity

異常の重さ

例

```cpp
enum class Severity
{
    Info,
    Warning,
    Error,
    Critical
};
```

---

例

```text
overflowCount=1
```

↓

```text
Warning
```

---

```text
doubleRetireCount
```

↓

```text
Critical
```

---

## Persistence

継続時間

瞬間的異常と慢性異常を分離する。

例

```text
routerPendingRetire=100
```

だけでは昇格しない。

---

```text
routerPendingRetire>100
が
30秒継続
```

↓

昇格

---

## Blast Radius

影響範囲

---

例えば

```text
Deferred Publish TTL超過
```

は

```text
1 publishのみ
```

への影響。

---

しかし

```text
Reader Stuck
```

は

```text
Retire
↓
Router
↓
Shutdown
```

まで連鎖する。

---

影響範囲が大きい。

---

# Layer 2

Recovery Policy

ここが現在存在しない。

---

例えば

## Retire Pressure

```text
routerPendingRetire > 128
```

↓

```text
Health=Degraded
```

ではなく

まず

```text
Publish Rate Limit
```

発動

---

### RecoveryAction

```cpp
enum class RecoveryAction
{
    None,

    ThrottlePublication,

    RejectNewPublication,

    ForceRetireDrain,

    SuspendCrossfade,

    EmergencyShutdown
};
```

---

例

```text
overflowRate増加
```

↓

```text
ThrottlePublication
```

---

改善しない

↓

```text
RejectNewPublication
```

---

改善しない

↓

```text
Critical
```

---

# Layer 3

Authority Escalation

ここで初めて制御権を持つ。

---

## AuthorityClass

既存

```cpp
Diagnostic
Derived
Authoritative
```

を利用。

---

例えば

### RouterPendingRetire

```text
50
```

↓

```text
Diagnostic
```

---

```text
100
```

↓

```text
Policy Warning
```

---

```text
500
+
60秒継続
```

↓

```text
Authority
```

---

この時だけ

```cpp
Admission Reject
```

発動

---

# Layer 4

Fail-Safe

最後の手段

---

例

```text
Reader Stuck
+
Router Overflow
+
Reclaim停止
+
Shutdown不能
```

---

ここで初めて

```cpp
EmergencyShutdown
```

許可

---

# 異常分類テーブル

私ならこうする

| Event                           | Severity | Recovery            | Authority |
| ------------------------------- | -------- | ------------------- | --------- |
| overflowCount++                 | Warning  | なし                  | No        |
| overflowRate高                   | Error    | ThrottlePublication | No        |
| routerPendingRetire > threshold | Error    | RejectPublication   | 条件付き      |
| ReaderStuck                     | Critical | ForceRetireDrain    | Yes       |
| doubleRetire                    | Critical | 即HealthCritical     | Yes       |
| WorldConsistencyFailure         | Critical | 即HealthCritical     | Yes       |
| DeferredTTLExpired              | Warning  | Discard             | No        |
| CrossfadeTimeout                | Error    | SuspendCrossfade    | 条件付き      |
| CorrelationMismatch             | Critical | HealthCritical      | Yes       |

---

# Shutdown に対するポリシー

ここが最重要です。

Shutdown は

```text
Authority
```

とは分離する。

---

悪い例

```cpp
if (healthState == Critical)
    shutdownFailed();
```

---

良い例

```cpp
shutdownCompleted();
```

かつ

```cpp
ShutdownResult
{
    completed = true,
    healthState = Critical,
    blockingReason = ReaderStuck
}
```

---

つまり

```text
異常でも止まれる
```

を保証する。

---

# ConvoPeq に最も不足しているもの

実際には

```cpp
ISRAuthorityClass
```

も

```cpp
HealthMonitor
```

も既にあります。

不足しているのはその中間の

```cpp
RuntimePolicyEngine
```

です。

私は Practical Stable ISR Bridge Runtime の最終形として、

```text
Diagnostic
↓
Policy Engine
↓
Recovery Action
↓
Authority Escalation
↓
Fail Safe
```

の5層構造を推奨します。

現在の ConvoPeq はおおむね

```text
Diagnostic
↓
Authority
```

に近く、そのため「検出した異常をどう扱うか」の規則が各所に散在しています。まず RuntimePolicyEngine を導入して昇格規則を一元化するのが、最も効果の大きい改修になります。
