# ConvoPeq Practical Stable ISR Bridge Runtime 完全移行改修計画書

**Version:** Final
**前提:** 現行達成率 92〜95%
**方針:** 作り直し禁止・最小修正主義・実運用優先
**目標:** 「Practical Stable ISR Bridge Runtime」100%達成

---

# 0. エグゼクティブサマリー

調査結果から、ConvoPeq は既に以下を達成しています。

* RuntimeStore 単一路線化
* RuntimeWorld Semantic Schema 化
* RuntimePolicyEngine 実装
* RuntimePublicationCoordinator の権限集中
* ISRRetireRouter の責務分離
* SnapshotCoordinator の責務分離
* RuntimeHealthMonitor の Event 化
* SealedObject による freeze
* DSP 直読排除済み CrossfadeAuthority

これらは Practical Stable の中核要件です。

したがって、

```text
全面再設計
大規模リファクタリング
Authority再編
Runtime刷新
```

は不要です。

残る未達は、

```text
A. RuntimePublicationValidator 実体化
B. CrossfadeAuthority の Engine依存除去
```

の2項目です。

---

# 1. 完成目標アーキテクチャ

```text
RuntimeBuilder
      │
      ▼
RuntimePublishWorld
      │
      ▼
RuntimePublicationValidator
      │
      ▼
RuntimePolicyEngine
      │
      ▼
CrossfadeAuthority
      │
      ▼
RuntimePublicationCoordinator
      │
      ▼
RuntimeStore
      │
      ▼
Publish
      │
      ▼
DSPTransition
      │
      ▼
ISRRetireRouter
      │
      ▼
EpochDomain
```

---

# 2. Phase-0（即時修正）

## P0-1 Dormant Bug 除去

### 問題

RuntimeBuilder

```cpp
worldOwner->overlap.useDryAsOld = active;
```

概念的に誤り。

---

### 修正

```cpp
worldOwner->overlap.useDryAsOld =
    (policy == convo::TransitionPolicy::DryAsOld);
```

---

### 目的

将来

```cpp
setFirstIrDryPending(true)
```

が実装された際の休眠バグ顕在化を防止。

---

### リスク

極小

---

### 工数

5分

---

# 3. Phase-1 RuntimePublicationValidator 完成

## 優先度

最高

---

## 現状

以下3メソッドが Placeholder。

```cpp
validateTopology()
validateResources()
checkNoConflictingTransitions()
```



---

## P1-1 validateTopology 実装

### 実装項目

```cpp
if (world.generation > 0
    && world.topology.runtimeUuid == 0)
{
    return false;
}
```

```cpp
if (world.topology.hasFadingRuntime
    != (world.topology.fadingRuntimeUuid != 0))
{
    return false;
}
```

```cpp
if (world.topology.hasFadingRuntime
    != world.execution.transitionActive)
{
    return false;
}
```



---

## P1-2 validateResources 実装

### Oversampling

許容値

```text
1
2
4
8
16
```

以外拒否。



---

### Dither

許容値

```text
0
16
24
```

以外拒否。



---

### NoiseShaper

許容値

```text
0
1
2
```

以外拒否。



---

## P1-3 checkNoConflictingTransitions 実装

### SmoothOnly

```cpp
transitionActive
↓
fadeTimeSec > 0
```

必須

---

### DryAsOld

```cpp
transitionActive
&& useDryAsOld
&& fadeTimeSec > 0
```

必須

---

### HardReset

```cpp
transitionActive == false
```

または

```cpp
fadeTimeSec == 0
```

を保証。

---

## P1 完了条件

Placeholder がゼロ。

```cpp
return true; // placeholder
```

完全消滅。

---

# 4. Phase-2 CrossfadePolicy 抽出

## 優先度

高

---

## 現状問題

CrossfadeAuthority が

```cpp
engine.m_osFadeTimeSec
engine.m_irFadeTimeSec
engine.m_irLengthFadeTimeSec
engine.m_phaseFadeTimeSec
engine.m_directHeadFadeTimeSec
engine.m_nucFilterFadeTimeSec
engine.m_tailFadeTimeSec
```

を直接参照している。

---

## 理想

CrossfadeAuthority は

```text
Decision Authority
```

のみ。

Policy保有禁止。

---

## P2-1 CrossfadePolicy 追加

```cpp
struct CrossfadePolicy
{
    double osFadeTimeSec;
    double irFadeTimeSec;
    double irLengthFadeTimeSec;
    double phaseFadeTimeSec;
    double directHeadFadeTimeSec;
    double nucFilterFadeTimeSec;
    double tailFadeTimeSec;
};
```

---

## P2-2 evaluate変更

現在

```cpp
evaluate(engine,
         oldWorld,
         newWorld)
```

↓

改修後

```cpp
evaluate(oldWorld,
         newWorld,
         policy)
```



---

## P2-3 RuntimePublicationOrchestrator 修正

現在

```cpp
CrossfadeAuthority crossfade;
auto decision =
    crossfade.evaluate(engine_,
                       *oldWorld,
                       *worldOwner);
```



---

改修後

```cpp
CrossfadePolicy policy;
policy.xxx = consumeAtomic(...);

auto decision =
    crossfade.evaluate(
        *oldWorld,
        *worldOwner,
        policy);
```

---

## P2-4 Critical 判定除去

CrossfadeAuthority から

```cpp
getHealthStateRef()
```

依存を削除。

Health異常は既に

```text
RuntimeHealthMonitor
↓
RuntimePolicyEngine
```

が担当している。

---

## P2 完了条件

CrossfadeAuthority が

```cpp
AudioEngine&
```

を一切受け取らない。

---

# 5. Phase-3 テスト強化

## P3-1 Validator テスト

追加ケース

### Topology

```cpp
hasFadingRuntime=true
fadingRuntimeUuid=0
```

↓

Reject

---

### Resource

```cpp
oversamplingFactor=3
```

↓

Reject

---

### Transition

```cpp
HardReset
fadeTimeSec > 0
```

↓

Reject

---

## P3-2 CrossfadeAuthority テスト

### Policy差し替え

```cpp
policyA
```

↓

DecisionA

```cpp
policyB
```

↓

DecisionB

---

### Pure性確認

```cpp
同一world
同一policy
```

↓

常に同じDecision

---

# 6. 将来課題（今回は実施しない）

## 6-1 Monitor Verification 移管

現状

```text
RuntimeHealthMonitor
  └ Verification
```

↓

将来

```text
RuntimePolicyEngine
  └ Verification
```

優先度 LOW

---

## 6-2 Dead Code 整理

対象

```cpp
CrossfadeRuntime::setUseDryAsOld()
CrossfadeRuntime::setFirstIrDryPending()
```

呼び出し元ゼロ。

今回は削除しない。

---

# 7. 実施禁止事項

以下は費用対効果が悪く、実施禁止とする。

```text
RetireAuthority導入
Snapshot再設計
RuntimeStore刷新
Coordinator再設計
PolicyEngine再設計
CrossfadeAuthority削除
ISRPolicyEngine新規作成
```

これらは既に要件を満たしている。

---

# 8. 完了判定

以下を全て満たした時点で Practical Stable ISR Bridge Runtime 完了とする。

### 必須

* useDryAsOld Dormant Bug 除去
* Validator Placeholder 全廃
* CrossfadeAuthority Engine依存ゼロ
* Validator テスト追加
* CrossfadeAuthority テスト追加

### 任意

* Verification の PolicyEngine 移管
* Dead Code 整理

---

# 最終評価

この計画は「全面刷新計画」ではありません。

現在の ConvoPeq は既に Practical Stable ISR Bridge Runtime の骨格を完成しており、残る作業は

1. Dormant Bug 除去
2. RuntimePublicationValidator 実体化
3. CrossfadePolicy 抽出
4. テスト強化

の4項目のみです。

この計画完了時点で、Practical Stable ISR Bridge Runtime の達成率は実質 100% に到達すると評価します。
