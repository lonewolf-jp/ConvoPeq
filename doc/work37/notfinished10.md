最新版の ConvoPeq.md を確認した範囲では、以前より大幅に改善されています。しかし、「実運用で破綻しにくい Practical Stable ISR Bridge Runtime」という観点では、まだいくつかの構造的弱点が残っています。

重要なのは、今回見つかる問題の多くは「クラッシュするバグ」ではなく、

* 長時間運転
* 数万回以上の Publish/Retire
* Device切替
* Shutdown競合
* Reader異常滞留
* Deferred Publish連打

といった実運用条件で徐々に顕在化する種類です。

---

# ① HealthState が Shutdown Authority に組み込まれていない

## 未達成内容

HealthMonitor は存在するが、

```cpp
ISRHealthState
```

が

```cpp
RuntimeDrainAudit
```

の診断情報としてしか使われていない。 

またコメントで

```cpp
healthState は canShutdown 条件にしない
```

と明示されている。 

---

## 問題

例えば

* Publication Stall
* Reader Stuck
* Retire Age Critical
* Queue Overflow Critical

が発生していても

Drain条件だけ満たせば

Shutdown成功

になり得る。

つまり

「壊れた状態で正常終了扱い」

になる。

---

## あるべき姿

Shutdown完了判定は

```text
Drained
AND
HealthState != Critical
```

であるべき。

---

## 改修方法

Shutdown最終判定で

```cpp
audit.healthState
```

を確認。

```cpp
if (audit.healthState == ISRHealthState::Critical)
{
    emitFailure(...);
    return ShutdownResult::HealthCritical;
}
```

を追加。

---

# ② WorldLifecycleAudit が監査専用でリーク防止になっていない

## 該当箇所

WorldLifecycleAudit は

```cpp
activeWorldCount
publishedCount
retiredCount
```

を保持。 

DrainAuditでも参照される。 

しかし

```cpp
activeWorldCount
```

は Shutdown Authority に含まれていない。

---

## 問題

理論上

```text
published = 10000
retired = 9999
active = 1
```

は正常。

しかし

```text
published = 10000
retired = 9998
active = 2
```

でも Shutdown 完了可能性がある。

Worldリークを止められない。

---

## あるべき姿

```text
activeWorldCount <= 1
```

を保証。

---

## 改修方法

Shutdown時に

```cpp
if (audit.activeWorldCount > 1)
{
    failShutdown();
}
```

を追加。

---

# ③ Retire Queue Overflow が回復不能

## 該当箇所

ISRRetireRouter

```cpp
provider_->enqueueRetire(...)
```

失敗時

```cpp
tryReclaim()
```

を一度だけ実行。 

その後

```cpp
QueuePressure
```

を返す。

---

## 問題

Practical Stable Runtimeでは

Queue Full

は

一時的異常

であって

永続障害ではない。

しかし現状

```text
QueuePressure
↓
呼び出し元依存
↓
drop
```

になりうる。

実質的に

Retire Lost

が発生しうる。

---

## あるべき姿

Overflow時は

必ず

```text
DeferredRetireFallbackQueue
```

へ退避。

---

## 改修方法

QueuePressure返却前に

```cpp
fallbackQueue.push(...)
```

を実施。

成功時

```cpp
RetireEnqueueResult::Deferred
```

を返す。

---

# ④ Deferred Publish が単一スロット

## 該当箇所

```cpp
DeferredPublishSlot
```

1個だけ。 

さらに

```cpp
hasDeferred_
```

で管理。 

---

## 問題

高速連続変更で

```text
A
B
C
D
```

が来ると

A,B,C が消える。

現在は overwrite count を記録するだけ。 

---

## あるべき姿

少なくとも

```text
latest N requests
```

を保持。

---

## 改修方法

```cpp
SPSC queue<DeferredRequest,16>
```

化。

古いものから破棄。

---

# ⑤ Reader Stuck 判定が Epoch差のみ

## 該当箇所

```cpp
detectStuckReaders(10)
```



---

## 問題

Epochが進まない期間は正常でも

Stuck扱いになる可能性がある。

逆に

Epochだけ進んでいれば

長時間Reader占有を見逃す。

---

## あるべき姿

判定は

```text
epoch差
+
residency時間
```

両方。

---

## 改修方法

ReaderSlotへ

```cpp
enterTimestampUs
```

保持。

```cpp
epochGap > X
&&
residency > Y sec
```

で判定。

---

# ⑥ RuntimeHealthMonitor が「検出のみ」

## 該当箇所

HealthMonitor は

```cpp
Healthy
Degraded
Critical
```

へ遷移する。 

しかし

実際の

Admission
Publish
Shutdown

制御にはほぼ利用されていない。

---

## 問題

Critical検出後も

Publishを継続可能。

障害拡大を止められない。

---

## あるべき姿

Critical時は

Admission側で

```cpp
RejectedHealthCritical
```

を返す。

---

## 改修方法

PublicationAdmissionへ

```cpp
if (healthState == Critical)
    reject;
```

を追加。

---

# ⑦ WorldLifecycleAudit が診断ファイル依存

## 該当箇所

```cpp
emitSnapshot()
```

は

```cpp
world_lifecycle_audit.json
```

を書くだけ。 

---

## 問題

実運用では

ファイルを見ない。

異常検知が遅れる。

---

## あるべき姿

World異常を

HealthMonitorへ送る。

---

## 改修方法

例えば

```cpp
activeWorldCount > 1
```

で

```cpp
EVENT_WORLD_LEAK
```

発火。

---

# ⑧ Retire Epoch と Publish Epoch の完全整合保証がない

## 該当箇所

Retire

```cpp
currentEpoch()
```

取得。 

Publish成功後

```cpp
advanceRetireEpoch()
```

実行。 

---

## 問題

現在は

```text
Publish成功
↓
Transition
↓
Epoch Advance
```

だが、

失敗経路やShutdown競合で

Epoch進行とWorld遷移の因果関係を完全証明できない。

---

## あるべき姿

World単位で

```cpp
publishEpoch
retireEpoch
worldId
correlationId
```

を完全対応付け。

---

## 改修方法

WorldLifecycleRecordへ

```cpp
publicationSequence
```

を追加し、

Publish→Retireの因果追跡を強制。

---

# 優先度ランキング

実運用での危険度順に並べると、

| 優先度 | 項目                                       |
| --- | ---------------------------------------- |
| P0  | HealthState が Shutdown Authority に入っていない |
| P0  | Retire Queue Overflow の最終退避不足            |
| P0  | WorldLifecycleAudit がリーク検出権限を持たない        |
| P1  | HealthMonitor が Admission を止めない          |
| P1  | Reader Stuck 判定が Epoch差依存                |
| P1  | Deferred Publish 単一スロット                  |
| P2  | WorldLifecycleAudit がファイル監査のみ            |
| P2  | Publish Epoch と Retire Epoch の因果証跡不足     |

現状のコードは「ISR Runtime が成立していない」段階ではなく、かなり完成度が高い状態です。しかし Practical Stable ISR Bridge Runtime の最終形として見ると、上記の P0/P1 項目はまだ「監視はできるが強制力がない」という特徴があり、そこが残る主要な未達成点です。



最新版を確認すると、前回挙げた項目以外にも、Practical Stable ISR Bridge Runtime の観点で検討価値が高いものがまだあります。

ただし今回は、

* 「設計として未完成」
* 「実運用で長期的に事故要因になり得る」
* 「監査はあるが強制力が弱い」

というものに限定して挙げます。

---

# ⑨ World Consistency が Shutdown Authority になっていない

## 該当箇所

VerifyDrainedで

```cpp
audit.verifyWorldConsistency()
```

は実行されているが、

```cpp
diagLog(...)
emitEvidenceTickNonRt(...)
```

だけになっている。 

---

## 問題

例えば

```text
published=1000
retired=997
active=1
```

なら整合しているが、

```text
published=1000
retired=950
active=1
```

でも Shutdown は成立し得る。

World整合性異常が

診断情報

でしかない。

---

## あるべき姿

Shutdown Authority の最終条件に

```cpp
verifyWorldConsistency()==Consistent
```

を追加。

---

## 改修方法

```cpp
if (audit.verifyWorldConsistency()
        != RuntimeDrainAudit::ConsistencyState::Consistent)
{
    shutdownRuntime_.markFailed(...);
}
```

---

# ⑩ HealthState と Admission の粒度が粗い

これは最近改善されています。

最新版では

Critical → Reject

になっています。 

---

## 問題

現在

```text
Healthy
Degraded
Critical
```

の3段階しかない。

---

実運用では

```text
RetirePressure
ReaderPressure
CrossfadePressure
PublicationPressure
```

で対策が異なる。

---

## あるべき姿

HealthStateは

```text
状態
原因
```

を分離。

---

## 改修方法

例えば

```cpp
HealthCause
```

追加。

```cpp
RetireBacklog
ReaderStuck
CrossfadeTimeout
```

などを保持。

Admissionが原因別制御可能になる。

---

# ⑪ Deferred Publish の「滞留時間上限」が拒否条件に使われていない

## 該当箇所

監査情報として

```cpp
maxDeferredAgeMs
```

を保持。 

---

## 問題

現在は

観測のみ。

---

例えば

```text
Deferred Publish
↓
30秒放置
↓
Crossfade解除
↓
古い Publish 実行
```

が起きる。

---

## あるべき姿

Deferred Request に TTL を持たせる。

---

## 改修方法

実行前に

```cpp
if (age > limit)
{
    discard;
}
```

---

# ⑫ Router Pending Retire に Hard Limit が無い

## 該当箇所

DrainAudit は

```cpp
routerPendingRetire
```

を監視している。 

---

## 問題

監視だけ。

実際には

```text
Reader Stuck
↓
Retire進まない
↓
Router蓄積
↓
メモリ増加
```

が起き得る。

---

## あるべき姿

一定数超過で

```text
HealthState=Critical
```

遷移。

---

## 改修方法

HealthMonitorで

```cpp
routerPendingRetire > threshold
```

を Critical 条件へ。

---

# ⑬ CorrelationId が Publish→Retire 完全追跡に使われていない

## 該当箇所

CorrelationId 採番は存在。 

---

## 問題

採番できても

```text
Publish
↓
Activate
↓
Retire
↓
Reclaim
```

まで一本で追跡できる保証が見えない。

---

## あるべき姿

Lifecycle 全体で

```text
CID
PublicationSequence
WorldId
Epoch
```

を紐付ける。

---

## 改修方法

LifecycleAuditに

```cpp
CorrelationId
```

を記録。

---

# ⑭ Shutdown BlockingReason の優先順位が不十分

## 該当箇所

BlockingReason は

```cpp
PendingPublication
PendingRetire
ActiveCrossfade
DeferredPublish
...
```

の順。 

---

## 問題

複数異常時に

真の根本原因が隠れる。

例

```text
Reader Stuck
↓
Retire滞留
↓
PendingRetire増加
```

なら

PendingRetire

しか表示されない。

---

## あるべき姿

単一理由ではなく

```cpp
vector<BlockingReason>
```

的な収集。

---

## 改修方法

Primary/Secondary を保持。

---

# ⑮ HealthMonitor が「自動回復戦略」を持たない

## 該当箇所

RuntimeHealthMonitor は

```cpp
checkRetireStall();
checkPublicationStall();
diagnoseRetireStall();
```

を実行。 

---

## 問題

基本的に

```text
検出
↓
状態変更
```

まで。

---

Practical Stable Runtime では

```text
検出
↓
緩和
↓
回復
↓
最後にCritical
```

が望ましい。

---

## あるべき姿

例えば

```text
RetirePressure
↓
Publish Throttle
↓
改善しない
↓
Critical
```

の段階制御。

---

## 改修方法

HealthMonitorに

```cpp
RecoveryAction
```

層を追加。

---

# 私の評価

最新版のコードは、以前のような

* Epoch破綻
* Retire所有権崩壊
* Publish経路多重化
* Crossfade Authority違反

のような致命的構造問題はかなり解消されています。

現段階で残っているのは主に、

1. **診断情報が Authority になっていない**
2. **HealthMonitor が観測中心**
3. **異常時の自動回復戦略が弱い**
4. **長期運転時のメモリ圧力対策が弱い**
5. **ライフサイクル追跡の証跡がまだ十分ではない**

という「運用安定性の最終仕上げ」領域です。

Practical Stable ISR Bridge Runtime の完成度で言うと、私の評価では現在 **90〜95%程度** で、残りは「異常系ガバナンス」と「自己回復能力」の強化が中心になります。
