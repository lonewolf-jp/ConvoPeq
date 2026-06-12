ConvoPeq.md 全体を踏まえると、2026-06-12版は以前より大幅に改善されています。

特に、

* ISRRetireRouter による EpochDomain 隠蔽
* Reader Stuck 検出
* RuntimeHealthMonitor
* RuntimeDrainAudit
* WorldLifecycleAudit
* Crossfade Timeout 監視
* Graceful Drain
* HealthState による Admission 制御

は既に実装されています。  

そのため「ISR Runtime が成立していない」状態ではありません。

しかし、Practical Stable ISR Bridge Runtime（実運用で長期間破綻しない ISR Runtime）という観点では、まだいくつか未達成または不十分な箇所があります。

以下、重要度順に列挙します。

---

# 1. HealthMonitor が Shutdown 完了条件に統合されていない

## 未達成内容

HealthMonitor は異常を検出するだけで、

* Shutdown Runtime
* Publication Admission
* RuntimeDrainAudit

との閉ループ制御が不十分です。

つまり

「異常を見つける」

まではできるが

「異常を収束させる」

まではできていません。

---

## 該当箇所

RuntimeHealthMonitor

* Reader Stuck
* Retire Stall
* Crossfade Timeout

を検出。 

しかし shutdown 判定は RuntimeDrainAudit ベースです。

---

## あるべき姿

Practical Stable Runtime では

HealthMonitor
↓
HealthState
↓
Admission
↓
Recovery
↓
Drain

が閉ループになるべきです。

---

## 改修方法

RuntimeDrainAudit に

```cpp
ISRHealthState healthState;
```

を追加。

VerifyDrained 時に

```cpp
healthState != Critical
```

も監査条件へ追加。

---

# 2. WorldLifecycleAudit が Diagnostic のみ

## 未達成内容

WorldLifecycleAudit は

```cpp
Diagnostic 限定
Shutdown Authority にはしない
```

と明記されています。

しかし RuntimePublishWorld が ISR Runtime の実質的な真実源です。

---

## 該当箇所

WorldLifecycleAudit

```cpp
activeWorldCount_
publishedCount_
retiredCount_
```

のみ管理。

---

## 問題

以下を検出できません。

* Publish済みだが retire されない World
* retire されたが publish されていない World
* worldId 再利用
* publish-retire 対応崩壊

---

## あるべき姿

Shutdown 最終監査時に

```cpp
publishedCount == retiredCount + activeWorldCount
```

を保証。

---

## 改修方法

WorldLifecycleAudit に

```cpp
verifyConsistency()
```

追加。

VerifyDrained で必須監査。

---

# 3. Retire Overflow が回復制御へ接続されていない

## 未達成内容

Overflow は検出されています。

```cpp
m_overflowCount_
```



HealthMonitor でも Rate 監視があります。

しかし

Overflow
↓
Publication制限
↓
Rebuild抑止

の自動制御がありません。

---

## 該当箇所

ISRRetireRouter

```cpp
QueuePressure
```

返却。

---

## あるべき姿

Overflow 継続時

```cpp
Publication Freeze
```

へ遷移。

---

## 改修方法

HealthState == Critical の間

```cpp
submitPublishRequest()
```

を Deferred へ落とす。

---

# 4. Reader Leak 自動隔離がない

## 未達成内容

Reader Stuck は検出されます。

しかし

```cpp
detectStuckReaders()
```

で終わっています。

---

## 問題

実運用では

* UI Thread死亡
* Worker死亡
* Driver異常

が起きます。

検出だけでは復旧できません。

---

## あるべき姿

一定時間以上 Stuck の Reader は

```cpp
Zombie Reader
```

として隔離。

---

## 改修方法

ReaderSlot に

```cpp
state:
Active
Suspect
Zombie
```

を追加。

30秒以上停止した Reader を quarantine。

---

# 5. RuntimeDrainAudit が Reader 状態を持たない

## 未達成内容

RuntimeDrainAudit に

```cpp
activeReaderCount
stuckReaderCount
```

が存在しません。

---

## 問題

Shutdown 完了判定と

Reader 状態が分離しています。

---

## あるべき姿

Drain Audit が

```cpp
Readers
Retire
Publication
Crossfade
World
```

を網羅。

---

## 改修方法

```cpp
uint64_t activeReaders;
uint64_t stuckReaders;
```

追加。

---

# 6. Crossfade 完了保証が弱い

## 未達成内容

Crossfade Timeout は監視しています。

しかし

```cpp
crossfadeRuntime_.isPending()
```

が永続 true の場合、

検出のみです。

---

## 問題

Crossfade が壊れたら

古い DSP が retire されません。

---

## あるべき姿

Timeout → 強制完了。

---

## 改修方法

30秒超過時

```cpp
crossfadeRuntime_.complete();
```

実行。

その後

```cpp
DSPTransition::onTransitionComplete()
```

を起動。

---

# 7. RuntimeBuilder の Warmup Validation が非常に弱い

## 未達成内容

現在の Warmup 検証は

```cpp
isIRLoaded()
&& !isIRFinalized()
```

のみ。

---

## 問題

DSP Runtime の健全性をほぼ確認していません。

---

## あるべき姿

最低でも

* sampleRate
* blockSize
* convolver state
* EQ coefficient state
* DSPHandle state

を検査。

---

## 改修方法

```cpp
validateRuntimeIntegrity()
```

追加。

BuildResult 前に実行。

---

# 8. Shutdown が「待つ」だけで「収束」しない

## 未達成内容

Graceful Drain はあります。

しかし

```cpp
while(...)
{
    publishEpoch();
    tryReclaim();
}
```

です。

---

## 問題

収束しない原因を除去していません。

---

## あるべき姿

Shutdown 中に

* Deferred Publish 中止
* Crossfade 強制終了
* Reader 隔離
* Publication Admission 停止

を行う。

---

## 改修方法

ShutdownRuntime に

```cpp
EmergencyDrain
```

Phase を追加。

---

# 総評

2026-06-12版 ConvoPeq は、

* ISR Retire
* Publication
* Health Monitoring
* Lifecycle Audit
* Shutdown Audit

まではほぼ実装済みです。  

一方で Practical Stable ISR Bridge Runtime の最終段階として不足しているのは、

**「異常検出」ではなく「異常収束」**

です。

優先度順では、

1. Reader Leak 自動隔離
2. Crossfade Timeout 強制収束
3. RuntimeDrainAudit への Reader/Health 統合
4. WorldLifecycleAudit 整合性検証
5. Overflow → Admission 抑制の閉ループ化
6. Emergency Drain Phase

が残る主要課題です。これらが実装されると、実運用での長時間連続稼働に対する Practical Stable ISR Bridge Runtime の完成度はかなり高い水準に達します。
