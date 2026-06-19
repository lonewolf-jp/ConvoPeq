# original_plan.md 妥当性検証レポート

**検証日:** 2026-06-18
**検証対象:** `doc/work45/original_plan.md` — 「ConvoPeq Practical Stable ISR Bridge Runtime 完全移行改修計画書」
**検証方法:** Serena MCP (20+回のパターン検索・シンボル検索) + read_file (実ファイル全行読取) + graphify (依存関係確認)

---

## 検証結果サマリ

| 計画の主張 | 実コードでの確認 | 判定 |
|---|---|---|
| `useDryAsOld = active` が休眠バグ | ✅ `RuntimeBuilder.cpp L287` で確認 | **正確** |
| `validateTopology()` が Placeholder | ✅ `RuntimePublicationValidator.cpp L63-77` `return true;` | **正確** |
| `validateResources()` が Placeholder | ✅ `RuntimePublicationValidator.cpp L79-93` `return true;` | **正確** |
| `checkNoConflictingTransitions()` が Placeholder | ✅ `RuntimePublicationValidator.cpp L128-140` `return true;` | **正確** |
| CrossfadeAuthority が engine直読 | ✅ `CrossfadeAuthority.cpp L13-60` で `engine.m_irFadeTimeSec` 等7フィールド読み取り確認 | **正確** |
| 直読フィールドは7個 | ✅ `m_osFadeTimeSec, m_irFadeTimeSec, m_irLengthFadeTimeSec, m_phaseFadeTimeSec, m_directHeadFadeTimeSec, m_nucFilterFadeTimeSec, m_tailFadeTimeSec` 全7個確認 | **正確** |
| 加えて `getHealthStateRef()` 依存 | ✅ `CrossfadeAuthority.cpp L13-20` で確認 | **正確** |
| RuntimePolicyEngine は既存 | ✅ `RuntimePolicyEngine::evaluateAggregate()` 実装済み | **正確** |
| テストファイル存在 | ✅ `tests/PublicationValidatorIsolationTests.cpp` 存在 (7テスト) | **正確** |
| 実施禁止事項は妥当 | ✅ 全項目が既存実装で要件満たしていることを確認 | **正確** |

**総合評価: 計画の全主張はソースコードと整合しており、実施可能と判定。**

---

## 1. Phase-0: Dormant Bug 除去 — 検証

### 計画の主張

```cpp
// RuntimeBuilder.cpp
worldOwner->overlap.useDryAsOld = active;  // ← 概念的誤り
```

### 検証結果

**ファイル**: `RuntimeBuilder.cpp L287`
**コード**: ✅ 計画の記述と完全一致

```cpp
worldOwner->overlap.useDryAsOld = active;
```

前後のコンテキスト:
```cpp
worldOwner->execution.transitionActive = active;      // L285
worldOwner->execution.transitionPolicy = static_cast<int>(policy); // L286
worldOwner->overlap.useDryAsOld = active;              // L287 ← 対象
worldOwner->overlap.fadeTimeSec = fadeTimeSec;         // L288
```

### 計画の修正案

```cpp
worldOwner->overlap.useDryAsOld = (policy == convo::TransitionPolicy::DryAsOld);
```

**判定: 妥当。** `useDryAsOld` の論理的意味（DryAsOldポリシーが選択された場合のみtrue）と完全に一致する。

### 前回追跡調査の結論

- `setFirstIrDryPending(true)` — **呼び出し元ゼロ**のDEAD CODE
- `setUseDryAsOld(true)` — **呼び出し元ゼロ**のDEAD CODE
- 現在は `useDryAsOld=true` になっても後段で実害なし
- しかし**休眠バグ**として将来の顕在化リスクあり
- 1行修正で予防可能

**→ Phase-0 の実施は妥当。**

---

## 2. Phase-1: RuntimePublicationValidator 完成 — 検証

### 2-1. Placeholder の確認

#### validateTopology()

**ファイル**: `RuntimePublicationValidator.cpp L63-77`
**コード**:
```cpp
bool RuntimePublicationValidator::validateTopology(
    const RuntimePublishWorld& world) const
{
    [[maybe_unused]] const auto& routing = world.routing;
    // This is a placeholder for actual topology validation logic
    return true; // Placeholder
}
```

**判定: 計画の主張と完全一致。** `return true` のみ。コメントにも `Placeholder` と明記。

#### validateResources()

**ファイル**: `RuntimePublicationValidator.cpp L79-93`
**コード**:
```cpp
bool RuntimePublicationValidator::validateResources(
    const RuntimePublishWorld& world) const
{
    [[maybe_unused]] const auto& resource = world.resource;
    // This is a placeholder for actual resource validation logic
    return true; // Placeholder
}
```

**判定: 計画の主張と完全一致。** `return true` のみ。

#### checkNoConflictingTransitions()

**ファイル**: `RuntimePublicationValidator.cpp L128-140`
**コード**:
```cpp
bool RuntimePublicationValidator::checkNoConflictingTransitions(
    const RuntimePublishWorld& world) const
{
    [[maybe_unused]] const auto& exec = world.execution;
    [[maybe_unused]] const auto& overlap = world.overlap;
    // This is a placeholder for actual conflict detection logic
    return true; // Placeholder
}
```

**判定: 計画の主張と完全一致。**

### 2-2. 計画の実装案の妥当性検証

#### P1-1 validateTopology 実装案

```cpp
// 計画提案1: generation > 0 なら runtimeUuid != 0
if (world.generation > 0 && world.topology.runtimeUuid == 0)
    return false;
```

**検証**: `TopologySemantic` 構造体:
```cpp
struct TopologySemantic {
    std::uint64_t runtimeUuid = 0;
    std::uint64_t fadingRuntimeUuid = 0;
    bool hasFadingRuntime = false;
};
```

**判定: 妥当。** Bootstrap以外のworldがruntimeUuid=0は異常。

```cpp
// 計画提案2: hasFadingRuntime と fadingRuntimeUuid の整合性
if (world.topology.hasFadingRuntime != (world.topology.fadingRuntimeUuid != 0))
    return false;
```

**判定: 妥当。** 両者は論理的に一致すべき。

```cpp
// 計画提案3: hasFadingRuntime と transitionActive の整合性
if (world.topology.hasFadingRuntime != world.execution.transitionActive)
    return false;
```

**判定: 妥当。** 両者は同じ事実を異なるsemanticで表現している。

#### P1-2 validateResources 実装案

**Oversampling 検証**: 許容値 {1, 2, 4, 8, 16}

`ResourceSemantic`:
```cpp
struct ResourceSemantic {
    int oversamplingFactor = 1;
    int ditherBitDepth = 0;
    int noiseShaperType = 0;
};
```

**判定: 妥当。** Oversamplingは2のべき乗かつ1〜16が実装上許容される範囲。`isValidExecutionSemantic` パターンに準拠。

#### P1-3 checkNoConflictingTransitions 実装案

`TransitionPolicy` enum（`RuntimeTransition.h`）:
```cpp
enum class TransitionPolicy : uint8_t {
    SmoothOnly = 0,
    HardReset = 1,
    DryAsOld = 2
};
```

| 計画提案 | 検証 |
|----------|------|
| SmoothOnly + transitionActive → fadeTimeSec > 0 必須 | **妥当。** fadeなしのSmoothOnlyは矛盾 |
| DryAsOld + transitionActive → useDryAsOld=true + fadeTimeSec > 0 | **妥当。** DryAsOldの定義に合致 |
| HardReset → transitionActive=false または fadeTimeSec==0 | **妥当。** 即時切り替えの定義 |

**ただし注意**: 計画では `transitionPolicy` が `HardReset(1)` のとき `transitionActive=false` または `fadeTimeSec==0` としている。実際のOrchestratorでは、Builderが HardReset でビルド後、CrossfadeDecision に応じて SmoothOnly に上書きされる。Validatorは publish直前の最終状態を検証するため、このチェックは正しい。

**→ Phase-1 の全実装案は正確かつ実装可能。**

---

## 3. Phase-2: CrossfadePolicy 抽出 — 検証

### 3-1. evaluate() シグネチャ確認

**ファイル**: `CrossfadeAuthority.h L23-29`
```cpp
[[nodiscard]] Decision evaluate(
    const AudioEngine& engine,          // ← この依存を除去する
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld) noexcept;
```

**判定: 計画の主張と完全一致。**

### 3-2. Engine直読フィールドの完全確認

**ファイル**: `CrossfadeAuthority.cpp L13-60`

Serena MCP で全 `engine.*` 参照を抽出:

| # | 参照元 | 行 | 計画記載 | 一致 |
|---|---|---|---|---|
| 1 | `engine.getHealthStateRef()` | 13 | (記載) | ✅ |
| 2 | `engine.m_osFadeTimeSec` | 35 | ✅ `m_osFadeTimeSec` | ✅ |
| 3 | `engine.m_irFadeTimeSec` | 43 | ✅ `m_irFadeTimeSec` | ✅ |
| 4 | `engine.m_irLengthFadeTimeSec` | 50 | ✅ `m_irLengthFadeTimeSec` | ✅ |
| 5 | `engine.m_phaseFadeTimeSec` | 51 | ✅ `m_phaseFadeTimeSec` | ✅ |
| 6 | `engine.m_directHeadFadeTimeSec` | 52 | ✅ `m_directHeadFadeTimeSec` | ✅ |
| 7 | `engine.m_nucFilterFadeTimeSec` | 53 | ✅ `m_nucFilterFadeTimeSec` | ✅ |
| 8 | `engine.m_tailFadeTimeSec` | 54 | ✅ `m_tailFadeTimeSec` | ✅ |

**判定: 計画記載の7フィールド + getHealthStateRef() = 8依存、完全一致。**

### 3-3. 各フィールドの AudioEngine.h 定義確認

**ファイル**: `AudioEngine.h L1621-1627`
```cpp
std::atomic<double> m_irFadeTimeSec { 0.080 };
std::atomic<double> m_irLengthFadeTimeSec { 0.050 };
std::atomic<double> m_phaseFadeTimeSec { 0.060 };
std::atomic<double> m_directHeadFadeTimeSec { 0.010 };
std::atomic<double> m_nucFilterFadeTimeSec { 0.030 };
std::atomic<double> m_tailFadeTimeSec { 0.030 };
std::atomic<double> m_osFadeTimeSec { 0.030 };
```

**判定: 全フィールドが存在し、全て `std::atomic<double>` で `consumeAtomic` による安全な読み取りが可能。**

### 3-4. CrossfadePolicy 抽出の実装可能性

**計画提案**:
```cpp
struct CrossfadePolicy {
    double osFadeTimeSec;
    double irFadeTimeSec;
    double irLengthFadeTimeSec;
    double phaseFadeTimeSec;
    double directHeadFadeTimeSec;
    double nucFilterFadeTimeSec;
    double tailFadeTimeSec;
};
```

**呼び出し側の修正**（`RuntimePublicationOrchestrator.cpp L99-100`）:
現在:
```cpp
CrossfadeAuthority crossfade;
auto cfDecision = crossfade.evaluate(engine_, *oldWorld, *worldOwner);
```

修正後（計画案）:
```cpp
CrossfadePolicy policy;
policy.irFadeTimeSec = convo::consumeAtomic(engine_.m_irFadeTimeSec, ...);
// ... 同様に7フィールド
CrossfadeAuthority crossfade;
auto cfDecision = crossfade.evaluate(*oldWorld, *worldOwner, policy);
```

**判定: 実装可能。** `consumeAtomic` は既存の `AtomicAccess.h` で実装済み（`std::atomic_load_explicit` のラッパー）。呼び出し側で Policy を構築し、Authority に渡すパターンは既存の設計思想と整合する。

### 3-5. Critical判定除去

**計画主張**: CrossfadeAuthority 内の `engine.getHealthStateRef()` チェックは、`RuntimePolicyEngine` が既に MonitorState → RecoveryAction として処理しているため不要。

**検証**: 現状の `RuntimePolicyEngine::evaluateAggregate()` は MonitorState::Error（Critical時）に対して RecoveryAction::Critical を発行する。CrossfadeAuthority が自前で Critical チェックする必要はない。

**判定: 妥当。**

**→ Phase-2 の全実装案は正確かつ実装可能。**

---

## 4. Phase-3: テスト強化 — 検証

### 4-1. 既存テスト確認

**ファイル**: `tests/PublicationValidatorIsolationTests.cpp`

既存7テスト:
1. `ValidatePublication_SemanticConsistency_Success` — 正常系
2. `ValidatePublication_InvalidExecutionSemantic_Reject` — `crossfadeStartDelayBlocks < 0` でReject
3. `ValidatePublication_NegativeDryHoldSamples_Reject` — `crossfadeDryHoldSamples < 0` でReject
4. `ValidateSemanticConsistency_ActivationEpochDerived_Success` — Derived fieldテスト
5. `ValidateTopology_BasicTopology_Success` — `EXPECT_TRUE(isValid)`（placeholder確認のみ）
6. `ValidateResources_BasicResources_Success` — `EXPECT_TRUE(isValid)`（placeholder確認のみ）
7. `CheckNoConflictingTransitions_NoTransition_Success` — `EXPECT_TRUE(isValid)`（placeholder確認のみ）

**判定: テスト5-7は Placeholder をテストしているのみ。計画の提案する reject テスト（Topology矛盾、Resource範囲外、Transition矛盾）は全て新規必要。**

### 4-2. 計画提案のテストケース

| テスト | 計画記載 | 優先度 |
|--------|----------|--------|
| `hasFadingRuntime=true` + `fadingRuntimeUuid=0` → Reject | ✅ | HIGH |
| `oversamplingFactor=3` → Reject | ✅ | HIGH |
| `HardReset` + `fadeTimeSec>0` → Reject | ✅ | HIGH |
| CrossfadeAuthority: Policy差し替えでDecision変化 | ✅ | MEDIUM |
| CrossfadeAuthority: Pure性（同一入力→同一出力） | ✅ | MEDIUM |

**判定: 適切かつ十分なテストケース。**

---

## 5. 将来課題（実施しない項目）— 検証

### 5-1. Monitor Verification 移管

**計画主張**: 現状 `RuntimeHealthMonitor` が持つ Verification ロジックを `RuntimePolicyEngine` へ移管するか？ → 「優先度LOW、今回は実施しない」

**検証**: HealthMonitor.tick() 内の Verification コード（約50行）は既存実装として動作している。PolicyEngine は Verification の基盤（VerificationEntry, TrendSnapshot, RecoveryOutcome）を既に持っているため、移管は可能だが現状で問題なし。

**判定: 計画の判断は妥当。**

### 5-2. Dead Code 整理

**計画主張**: `setUseDryAsOld()` / `setFirstIrDryPending()` — 呼び出し元ゼロだが削除しない。

**検証**: 前回追跡調査で確認済み。CrossfadePolicy抽出時に再利用判断するのが合理的。

**判定: 計画の判断は妥当。**

---

## 6. 実施禁止事項 — 検証

### 6-1. RetireAuthority 導入

**計画主張**: 不要。ISRRetireRouter は Transport Layer であり判断分散ではない。

**検証**: `ISRRetireRouter.cpp` のコードは全て IEpochProvider への委譲のみ。判断ロジックは一切なし。

**判定: 禁止は妥当。**

### 6-2. Snapshot 再設計

**計画主張**: 不要。責務明確、IEpochProvider 委譲済み。

**検証**: `SnapshotCoordinator` は observe → switchImmediate → enqueueWithRetry の明確な責務境界。IEpochProvider 経由で retire 委譲済み。

**判定: 禁止は妥当。**

### 6-3. RuntimeStore 刷新

**計画主張**: 不要。単一路線化 + exchangeAtomic 完了済み。

**検証**: `RuntimeStore` は `exchangeAtomic(store_->current, next, memory_order_acq_rel)` の単一交換ポイント。設計は十分にシンプル。

**判定: 禁止は妥当。**

### 6-4. Coordinator 再設計

**計画主張**: 不要。適切なステートマシン + precheckPublish 実装済み。

**検証**: `ISRRuntimePublicationCoordinator` は 7状態（Bootstrapping→Ready→Publishing→Transitioning→Pressure→ShuttingDown→Faulted）のステートマシン + ClosureValidator + PayloadTierValidator 完備。

**判定: 禁止は妥当。**

### 6-5. PolicyEngine 再設計

**計画主張**: 不要。evaluateAggregate + Cooldown + Budget + Storm検出完備。

**検証**: `RuntimePolicyEngine` は evaluateAggregate(6種MonitorState)、evaluateEvent(10種PolicySource)、Cooldown制御、RecoveryBudget、EscalationTracker（Storm検出）完備。

**判定: 禁止は妥当。**

### 6-6. CrossfadeAuthority 削除

**計画主張**: 不要。むしろ残すべき。DSP直読排除済み。

**検証**: CrossfadeAuthority は world.dspProjection ベースの判断を行っており、削除すると判断ロジックが分散する。

**判定: 禁止は妥当。**

### 6-7. ISRPolicyEngine 新規作成

**計画主張**: 不要。RuntimePolicyEngine が既存。

**検証**: `RuntimePolicyEngine` クラスは存在し、実装済み。

**判定: 禁止は妥当。**

---

## 7. 変更影響範囲の評価

### Phase-0: 1行修正

| ファイル | 変更 | 影響 |
|----------|------|------|
| `RuntimeBuilder.cpp:287` | `useDryAsOld = active` → `useDryAsOld = (policy == DryAsOld)` | 最小。内部ロジック不変 |

### Phase-1: 約50〜100行追加

| ファイル | 変更 | 影響 |
|----------|------|------|
| `RuntimePublicationValidator.cpp` | 3メソッドの Placeholder を実装に置換 | 純関数追加、副作用なし。テスト容易 |
| `tests/PublicationValidatorIsolationTests.cpp` | Rejectテスト追加（約100行） | テスト追加のみ |

### Phase-2: 約30〜50行変更

| ファイル | 変更 | 影響 |
|----------|------|------|
| `CrossfadeAuthority.h` | `CrossfadePolicy` struct追加、evaluate()シグネチャ変更 | インターフェース変更、既存呼び出し側の修正必要 |
| `CrossfadeAuthority.cpp` | engine依存を policy パラメータで受け取るよう変更 | 内部ロジック不変 |
| `RuntimePublicationOrchestrator.cpp` | evaluate()呼び出し側で policy 生成 | Atomic読み取りの移動のみ |

**全フェーズ合計: 約80〜150行の変更。安全性は高い。**

---

## 8. 完了条件の達成検証

### 必須条件

| 条件 | 現状 | 計画実施後 |
|------|------|-----------|
| useDryAsOld Dormant Bug 除去 | ❌ 未対処 | ✅ Phase-0 |
| Validator Placeholder 全廃 | ❌ 3メソッドがPlaceholder | ✅ Phase-1 |
| CrossfadeAuthority Engine依存ゼロ | ❌ 8個のengine依存 | ✅ Phase-2 |
| Validator テスト追加 | ❌ Placeholderテストのみ | ✅ Phase-3 |
| CrossfadeAuthority テスト追加 | ❌ 未存在 | ✅ Phase-3 (予定) |

### 任意条件

| 条件 | 判定 | 備考 |
|------|------|------|
| Verification の PolicyEngine 移管 | 保留 | 優先度LOW、現状で問題なし |
| Dead Code 整理 | 保留 | CrossfadePolicy抽出時に判断 |

---

## 9. 総合判定

### 計画の正確性

| 評価項目 | 結果 |
|----------|------|
| 現状分析の正確性 | ✅ **極めて高い**。全主張が実コードと一致 |
| 修正案の妥当性 | ✅ **実装可能**。既存API・パターンに沿った設計 |
| 実施禁止の妥当性 | ✅ **正当**。全項目とも既存実装で要件充足 |
| 工数見積もりの妥当性 | ✅ **適切**。5分〜2日のレンジは現実的 |
| リスク評価の妥当性 | ✅ **適切**。各Phaseとも低リスク |

### コードベースとの整合性マトリクス

```
計画の主張                                   実コードでの確認
──────────────────────────────────────────────────────────
useDryAsOld = active (L287)              ✅ RuntimeBuilder.cpp L287
validateTopology → return true           ✅ RuntimePublicationValidator.cpp L63
validateResources → return true          ✅ RuntimePublicationValidator.cpp L79
checkNoConflictingTransitions → true     ✅ RuntimePublicationValidator.cpp L128
CrossfadeAuthority::evaluate(engine,...)  ✅ CrossfadeAuthority.h L26
engine.m_irFadeTimeSec 等 7フィールド     ✅ AudioEngine.h L1621-1627
engine.getHealthStateRef()               ✅ CrossfadeAuthority.cpp L13
RuntimePolicyEngine 実装済み              ✅ RuntimePolicyEngine.cpp L57 (evaluateAggregate)
テストファイル存在                        ✅ tests/PublicationValidatorIsolationTests.cpp
```

### 最終判定

**✅ original_plan.md は全主張が実ソースコードと整合しており、計画は実施可能である。**

```text
整合性スコア: 100% (全項目が実コードで確認済み)
リスク評価:    LOW (全Phaseとも小規模変更、副作用极少)
実装優先順位:  Phase-0(5分) → Phase-1(半日) → Phase-2(1-2日) → Phase-3(半日-1日)
```

唯一の注意点として、計画が **「CrossfadeAuthority の engine依存を除去」** としているが、実際には `CrossfadeAuthority.cpp` 内で `engine.getHealthStateRef()` という **8つ目の依存** が存在する。計画には Critical 判定除去（P2-4）として記載済みであり、問題はない。
