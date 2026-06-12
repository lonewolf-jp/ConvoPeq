ConvoPeq.md（2026-06-12版）を確認した限りでは、以前の監査で指摘した重大な ISR 破綻要因の多くは既に対処されています。

ただし、**「実運用で破綻しにくい Practical Stable ISR Bridge Runtime」**という基準で見ると、まだ未達成または不十分と思われる箇所が残っています。

以下は優先度順です。

---

# 1. Health Monitor が診断系に留まり、制御系へ十分反映されていない

## 1-1. 未達成内容

RuntimeHealthMonitor は

* Reader Stuck
* Publication Stall
* Retire Stall
* Overflow
* Reader Slot Exhaustion

を監視していますが、

実際に

* Publication Admission
* Rebuild Admission
* Runtime Builder
* Crossfade 開始

を止める権威になっている証拠が弱いです。

つまり

> 異常を発見する
>
> ↓
>
> ログを出す

まではあるが

> 異常を発見する
>
> ↓
>
> 新しい Publish を拒否する

まで到達していない可能性があります。

---

## 1-2. 該当箇所

RuntimeHealthMonitor



ISRHealthState



HealthState 更新



---

## 1-3. Practical Stable ISR Bridge Runtime のあるべき姿

HealthState が Critical になったら

* 新規 Publish 停止
* 新規 Rebuild 停止
* Crossfade 開始禁止

へ直結する。

---

## 1-4. 改修方法

PublicationAdmission に

```cpp
if (healthState == ISRHealthState::Critical)
    return Reject;
```

を追加。

Degraded では

* Publish頻度制限
* Build同時実行数制限

を行う。

---

# 2. Reader Slot 枯渇時の回復戦略が存在しない

## 2-1. 未達成内容

RCUReader は slot 取得失敗時に fail-closed になります。

これは正しいです。

しかし

* なぜ枯渇したか
* どう復旧するか

が無い。

---

## 2-2. 該当箇所

RCUReader

slot取得失敗



EpochDomain

reader上限



```cpp
kMaxReaders = 64
```

---

## 2-3. あるべき姿

Reader枯渇は

* stuck reader
* reader leak
* thread churn

の兆候。

HealthMonitor が検知したら

* Admission停止
* 強制診断ダンプ

へ進むべき。

---

## 2-4. 改修方法

HealthMonitor の

EVENT_READER_SLOT_USAGE

発火時に

WorldLifecycleAudit

Telemetry

EvidenceExporter

へ自動連携。

---

# 3. WorldLifecycleAudit が監査専用で、Shutdown Authority に参加していない

## 3-1. 未達成内容

WorldLifecycleAudit は

明確に

> Diagnostic 限定

とされています。



しかし Practical Runtime では

World 数の整合性は

Shutdown 完了判定の重要材料です。

---

## 3-2. 該当箇所

WorldLifecycleAudit

activeWorldCount



---

## 3-3. あるべき姿

Shutdown 判定時

```cpp
activeWorldCount == 0
```

を DrainAudit が確認する。

---

## 3-4. 改修方法

RuntimeDrainAudit に

```cpp
activeWorldCount
publishedCount
retiredCount
```

を取り込み、

整合性チェックを追加。

---

# 4. DSPLifetimeManager の retire が epoch を進めている

## 4-1. 未達成内容

retire のたびに

```cpp
publishEpoch()
```

しています。



---

## 4-2. 問題

Publish と Retire が同じ epoch source を共有する場合、

大量 retire 時に

epoch が異常加速する。

Practical Runtime では

epoch は

「世代境界」

であるべきで

「解放要求数」

で増えるべきではありません。

---

## 4-3. あるべき姿

epoch増加権限は

Publication Coordinator

のみ。

Retire は

```cpp
currentEpoch()
```

取得。

---

## 4-4. 改修方法

```cpp
const auto epoch = router_->currentEpoch();
```

へ変更。

publishEpoch を retire 経路から排除。

---

# 5. RuntimeBuilder が例外依存

## 5-1. 未達成内容

RuntimeBuilder は

```cpp
catch(...)
```

を使っています。



---

## 5-2. 問題

Practical Runtime の観点では

DSP Runtime Build の失敗理由は

診断可能であるべき。

catch(...) は

原因を消します。

---

## 5-3. あるべき姿

失敗分類

* OOM
* MKL
* Convolver Build
* Warmup

を保持。

---

## 5-4. 改修方法

BuildError を拡張。

```cpp
BuildError::MKLFailure
BuildError::ConvolverFailure
BuildError::PrepareFailure
```

などへ分解。

---

# 6. CrossfadeRuntime に実クロスフェード数の上限制御が見当たらない

## 6-1. 未達成内容

CrossfadeRuntime は

* Event Drop
* Timeout

を監視しています。



しかし

「同時Crossfade数制限」

が見当たりません。

---

## 6-2. 問題

UI連打

Preset連打

自動最適化

で

Crossfade連発が発生可能。

---

## 6-3. あるべき姿

```cpp
maxConcurrentCrossfades = 1
```

または

```cpp
2
```

固定。

---

## 6-4. 改修方法

CrossfadeAuthority 側で Admission。

---

# 7. EpochDomain が固定64 Reader

## 7-1. 未達成内容

```cpp
kMaxReaders = 64
```

固定。



---

## 7-2. 問題

将来

* UI
* Worker
* Learning
* Analyzer
* Debug

が増えると限界。

---

## 7-3. あるべき姿

Practical Runtime では

固定でもよいが

枯渇時に

* 誰が占有しているか
* 何秒保持しているか

が取得できること。

---

## 7-4. 改修方法

Reader Ownership Telemetry を常設。

HealthMonitor の

```cpp
readerIndex
readerEpoch
readerDepth
residencyTimeUs
```

を EvidenceExporter に流す。



---

# 総合評価

現状の ConvoPeq は以前と比較すると大幅に改善されており、

* Authority 分離
* ISRRetireRouter 導入
* RuntimeHealthMonitor
* WorldLifecycleAudit
* CrossfadeAuthority
* RCUReader fail-closed

は既に整備されています。

Practical Stable ISR Bridge Runtime 達成度を概算すると

| 項目                      | 達成度 |
| ----------------------- | --: |
| Authority分離             | 95% |
| Retire Pipeline         | 90% |
| Publication Runtime     | 90% |
| Reader Safety           | 90% |
| Shutdown Safety         | 80% |
| Health-based Admission  | 70% |
| Self-Healing / Recovery | 60% |
| Operational Diagnostics | 85% |

総合では **約85〜90%達成** と評価します。

残りの主要課題は、

**「診断できる」から「異常時に自動的にPublishを抑制できる」への移行**

です。これは Practical Stable ISR Bridge Runtime の最後の大きな未達成領域です。
