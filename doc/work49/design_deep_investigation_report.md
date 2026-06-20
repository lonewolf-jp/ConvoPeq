# Work49 深堀調査レポート — 未確定事項の確定と設計反映

**作成日:** 2026-06-19
**調査者:** GitHub Copilot (DeepSeek V4 Flash)
**調査範囲:** src/audioengine, src/core, src/tests を中心とした全278ファイル
**使用ツール:** Serena MCP, AiDex MCP, CodeGraph MCP, graphify MCP, semble, Select-String

---

## 調査結果サマリー

前回の validation_report.md で「未確定」とされた7項目すべてについて、ソースコードの実態を確認し確定した。

---

## 1. currentWorld_ 全使用箇所の確定

### 結論: 6箇所すべて特定。削除は可能だが広範なテスト影響あり

### 実装詳細

| # | ファイル | 行 | 用途 | 削除可否 |
|---|---|---|---|---|
| 1 | ISRRuntimePublicationCoordinator.h | 90 | `std::atomic<const void*> currentWorld_;` 宣言 | — |
| 2 | ISRRuntimePublicationCoordinator.cpp | 10 | コンストラクタ初期化 `currentWorld_(nullptr)` | 削除可 |
| 3 | ISRRuntimePublicationCoordinator.cpp | 109 | `commit()`: 古い currentWorld_ を読み取り | **要代替** |
| 4 | ISRRuntimePublicationCoordinator.cpp | 123 | `commit()`: publishAtomic で newWorld を格納 | **要代替** |
| 5 | ISRRuntimePublicationCoordinator.cpp | 126 | `retire()`: compareExchangeAtomic で nullptr へのCAS | **要代替** |
| 6 | ISRRuntimePublicationCoordinator.cpp | 165 | `getCurrent()`: consumeAtomic で読み取り | **要代替** |

### ISRRuntimePublicationCoordinator の責務（確定）

`ISRRuntimePublicationCoordinator` は以下の2つの役割を持つ：

1. **永続状態管理**: publicationSequenceId_, publicationEpoch_, mappedRuntimeGeneration_
2. **現在Worldポインタ管理**: currentWorld_（getCurrent() / commit() / retire() で使用）

### currentWorld_ の呼び出し元（確定）

- `getCurrent()` → **テストファイル 17箇所**（ISRSemanticValidationTests.cpp）
  - testCoordinatorCommitAndMonotonicityContract: 3回
  - testCoordinatorRejectEpochRollbackContract: 2回
  - testCoordinatorRejectMappedGenerationRollbackOnEpochAdvance: 2回
  - testCoordinatorRejectEpochReuseContract: 2回
  - testCoordinatorRejectMappedGenerationReuseContract: 2回
  - testCoordinatorRejectWraparoundContract: 3回
  - testP4SameGenerationEpochChangeRejected: 1回
  - testP20RejectPreservesWorldState: 2回

- `commit()` / `retire()` → **AudioEngine.Commit.cpp:415**
  - `runtimePublicationBridge_.retire(...)` が唯一の実呼び出し

### 削除戦略（確定）

`ISRRuntimePublicationCoordinator` の `currentWorld_` を削除するには以下の代替手段が必要：

1. **getCurrent()**: `RuntimeStore.observe()` に委譲（ただし ISRRuntimePublicationCoordinator は RuntimeStore 非依存のため、Bridge 経由の注入が必要）
2. **commit()**: `mappedRuntimeGeneration_` の単調増加以外に currentWorld_ は不要。削除可能。
3. **retire()**: compareExchangeAtomic で currentWorld_ を nullptr にCASする処理を削除。RuntimeStore の retire 側に委譲。

**推奨**: currentWorld_ の削除は Phase-A の最終工程とし、以下の順序で実施：

1. まず PersistentStateBlock 導入（publicationSequenceId/epoch/generation の統合）
2. ISRRuntimePublicationCoordinator を Pure Persistent State Manager に純化
3. currentWorld_ 削除（テストも同時修正）

---

## 2. DSPTransition/HealthMonitor/CrossfadeRuntime の直接Publish有無

### 結論: 直接Publishは行われていない。すべて Orchestrator 経由。✅ 確定

### 検証詳細

| コンポーネント | publishWorld 呼び出し | submitPublishRequest 呼び出し | publishIdleWorldOnly 呼び出し |
|---|---|---|---|
| **DSPTransition** | なし | なし | **あり** (onTransitionComplete) |
| **RuntimeHealthMonitor** | なし | なし | なし（イベント発行のみ） |
| **CrossfadeRuntime** | なし | なし | なし |

### publishIdleWorldOnly() の呼び出し元（確定）

- `AudioEngine.Timer.cpp:650` — onHealthEvent 内（Crossfade Timeout 回復時）
- `DSPTransition.h:137` — onTransitionComplete 内（クロスフェード完了時）

→ publishIdleWorldOnly は Coordinator 経由で publish するため、バイパスではない。

### 唯一のPublish経路（確定）

```
AudioEngine.Commit.cpp:683  →  runtimeOrchestrator_->submitPublishRequest(req)
AudioEngine.Timer.cpp:425   →  coordinator.publishWorld(...)
AudioEngine.Transition.cpp:26 → coordinator.publishWorld(...)
PublicationExecutor.cpp:25   →  coordinator.publishWorld(...)
```

全経路が Coordinator を通るため「DSPTransition, AudioEngine, CrossfadeRuntime, HealthMonitor による直接 Publish」は存在しない。

---

## 3. Recovery Architecture の実態確定

### 結論: 高度に実装済み。階層的リカバリシステムが動作中

### 既存コンポーネント一覧（確定）

| コンポーネント | ファイル | 状態 |
|---|---|---|
| RecoveryAction enum（6階層） | RuntimePolicyEngine.h:57 | ✅ 実装済み |
| RestorePhase enum（3段階） | RuntimePolicyEngine.h:44 | ✅ 実装済み |
| RecoveryOutcome enum（4種） | RuntimePolicyEngine.h:66 | ✅ 実装済み |
| TrendSnapshot（10+フィールド） | RuntimePolicyEngine.h:74 | ✅ 実装済み |
| EscalationTracker（ストーム検出） | RuntimePolicyEngine.h:152 | ✅ 実装済み |
| RecoveryBudget（ウィンドウ管理） | RuntimePolicyEngine.h:174 | ✅ 実装済み |
| EpochAdvanceHealth（進捗監視） | RuntimePolicyEngine.h:108 | ✅ 実装済み |
| VerificationEntry（検証状態） | RuntimePolicyEngine.h:124 | ✅ 実装済み |
| executeRecoveryAction() | AudioEngine.Timer.cpp:660 | ✅ 実装済み |
| Restore: Epoch Recovery + Learner Rollback | AudioEngine.Timer.cpp:682 | ✅ 実装済み |

### リカバリ実行フロー（確定）

```
HealthMonitor が異常検知
  → onHealthEvent() 発火
    → イベント種別に応じて Timer が処理
      → executeRecoveryAction() で RecoveryAction 実行
```

executeRecoveryAction() の各アクション：

| Action | 実行内容 | ファイル行 |
|---|---|---|
| **Throttle** | retirePressureAdmissionStrict_= true + suppressionStartUs_ 記録 | Timer.cpp:665 |
| **Recover** | tryReclaimResources() + drainDeferredRetireQueues() + clearDeferredForShutdown() | Timer.cpp:674 |
| **Restore** | Epoch Recovery (setRollbackMode + requestRollback) + Learner Rollback + restoreGeneration++ | Timer.cpp:682-699 |
| **Safe** | stopNoiseShaperLearning() + retirePressureAdmissionStrict_ = false | Timer.cpp:705 |
| **Critical** | retirePressureAdmissionStrict_ = true + requestEmergencyDrain() | Timer.cpp:714 |

### 欠落要素（確定）

1. **deriveAuthorityState() 不在**: PersistentState + RuntimeStore から AuthorityState を再導出する関数がない
2. **PersistentStateBlock 不在**: publicationSequenceId/epoch/mappedGeneration の統一インターフェースがない
3. **Step2 (publishIdleWorldOnly) 未接続**: RestorePhase::EpochRecoveryIssued まで実装されているが、IdleWorldPublished への移行が閉ループ制御に依存

---

## 4. Validator テスト網羅性の確定

### 結論: 39テストケース。基本的な網羅は完了。エッジケースは限定的

### テストケース内訳（確定）

| カテゴリ | テスト数 | 内訳 |
|---|---|---|
| Semantic Consistency | 3 | Success / InvalidExecution / NegativeDryHoldSamples |
| Topology | 7 | Basic / NoRuntimeUuid / HasFadingMismatch / FadingTransitionMismatch / Bootstrap / NoUuidWithTransition / NoUuidWithHasFading |
| Resource | 9 | Basic / OversamplingNotPowerOfTwo / OversamplingOutOfRange / DitherInvalid / Dither32 / Dither16 / Dither24 / NoiseShaperOutOfRange / NoiseShaperFixed15Tap / NoiseShaperAdaptive / ValidOversampling |
| Transition | 8 | NoTransition / HardResetWithFade / SmoothOnlyNegativeFade / DryAsOldWithoutFlag / InactiveWithUseDryAsOld / UnknownPolicy / HardResetNoFade / DryAsOldValid / IdleWithFadeRemnant |
| Publication統合 | 4 | SemanticSuccess / RejectFromTopology / RejectFromTopologyNoUuid / RejectFromResources / RejectFromTransition |
| CrossfadeAuthority | 4 | DeterministicDecision / PolicyChangeChangesDecision / SameStructuralHashNoCrossfade / OversamplingChangeTriggersCrossfade |
| **合計** | **39** | |

### 不足テストケース（確定）

| カテゴリ | 不足ケース | 優先度 |
|---|---|---|
| Semantic | generation > 0 で sequenceId == 0 の reject | 中 |
| Semantic | 負の fadeTimeSec（transitionActive=true時） | 低 |
| Resource | oversamplingFactor = 16（上限） | 低 |
| Resource | ditherBitDepth = 0（未設定） | 低 |
| Resource | noiseShaperType = 0（未設定） | 低 |
| Topology | generation > 0 で runtimeGeneration == 0 の reject | 中 |
| Transition | HardReset + fade == 0 + useDryAsOld == true の reject | 低 |
| Recovery | 障害注入テスト | 高 |
| Property | ランダムシーケンス 10,000〜100,000回 | 中 |

---

## 5. CrossfadeAuthority 完全性の確定

### 結論: Pure Function として完全に実装済み。HealthState 非依存。✅

### 検証詳細

- `CrossfadeAuthority::evaluate()` の引数: `(const RuntimePublishWorld& oldWorld, const RuntimePublishWorld& newWorld, const CrossfadePolicy& policy)` — AudioEngine 非依存
- `CrossfadePolicy`: immutable POD、HealthState を含まない
- `dspProjection` フィールドのみ参照（irLoaded, structuralHash, oversamplingFactor）
- `kEvaluateRelevantFieldNames` で参照フィールドを明示

### HealthState Critical 抑制（確定）

抑制ロジックは Orchestrator レベルで行われる（`RuntimePublicationOrchestrator.cpp:108-115`）：

```cpp
auto ref = engine_.getHealthStateRef();
if (ref) {
    auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
    if (health == convo::ISRHealthState::Critical) {
        cfDecision.needsCrossfade = false;
        cfDecision.fadeTimeSec = 0.0;
    }
}
```

CrossfadeAuthority.evaluate() の結果を Orchestrator が上書きする形で抑制。

### Emergency Override の発動条件と動作（確定）

`DSPTransition::onPublishCompleted()` 内（`DSPTransition.h:55-75`）：

1. HealthState::Critical 検知
2. `lifetime.activate(newDSP)` — 即時 activate
3. `crossfadeRuntime_.complete()` — クロスフェード完了
4. `lifetime.retire(oldDSP)` — 旧 DSP 退役
5. `crossfadeRuntime_.incrementEmergencyAbortCount()` — カウンタ増加
6. `enqueueHealthEvent(EVENT_CROSSFADE_ABORTED_EMERGENCY, 4003)` — イベント発行

---

## 6. PersistentStateBlock 設計確定

### 設計方針

**`ISRRuntimePublicationCoordinator` から永続フィールドを抽出する。**

### 現状のフィールド配置（確定）

```cpp
class ISRRuntimePublicationCoordinator {
    // 永続状態（PersistentStateBlock に抽出可能）
    std::atomic<PublicationSequenceId> publicationSequenceId_;
    std::atomic<PublicationEpoch> publicationEpoch_;
    std::atomic<std::uint64_t> mappedRuntimeGeneration_;

    // 状態派生（保持不要 / 削除候補）
    std::atomic<const void*> currentWorld_;  // → 削除対象

    // 運用状態（coordinator の責務として保持）
    std::atomic<RejectCode> lastRejectCode_;
    std::atomic<std::uint64_t> retireBacklogCount_;
    std::atomic<std::uint64_t> publicationBacklogCount_;
    std::atomic<std::uint64_t> pendingIntentCount_;
    std::atomic<std::uint64_t> fallbackBacklogCount_;
    std::atomic<std::uint64_t> reclaimInFlightCount_;
    std::atomic<std::uint64_t> deferredRetireResidencyCount_;
    std::atomic<std::uint64_t> previousRetireBacklogCount_;
    std::atomic<std::uint32_t> pressureNormalizedWindows_;
    std::atomic<bool> swapPending_;
    std::atomic<CoordinatorState> state_;
    std::atomic<std::uint64_t> retireAuthorityCount_;
};
```

### PersistentStateBlock 設計（確定）

```cpp
struct PersistentStateBlock {
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
    };

    Snapshot snapshot() const noexcept {
        return Snapshot{
            convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire),
            convo::consumeAtomic(publicationEpoch, std::memory_order_acquire),
            convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire)
        };
    }

    void update(const Snapshot& s) noexcept {
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
    }
};
```

### 組み込み先（確定）

`ISRRuntimePublicationCoordinator` の private メンバとして PersistentStateBlock を保持する。
公開APIは既存の commit() のオーバーロードを通じてアクセスする。

---

## 7. deriveAuthorityState() 設計確定

### 設計方針

PersistentStateBlock + RuntimeStore から AuthorityState を再導出する Pure Function。

### 定義（確定）

```cpp
struct AuthorityState {
    // PersistentStateBlock から
    uint64_t publicationSequenceId;
    uint64_t publicationEpoch;
    uint64_t mappedRuntimeGeneration;

    // RuntimeStore から（runtimeWorld の有無）
    bool hasActiveRuntime;

    // 導出状態
    bool hasPendingPublication;    // publicationSequenceId > 0 && 未反映
    bool hasActiveCrossfade;       // runtimeWorld の transitionActive
};

// Pure Function — 内部状態を参照しない
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateBlock::Snapshot& persistentState,
    const void* runtimeWorld  // RuntimeStore.observe() の結果
) noexcept;
```

### 導出ロジック（確定）

```cpp
AuthorityState deriveAuthorityState(
    const PersistentStateBlock::Snapshot& ps,
    const void* runtimeWorld)
{
    AuthorityState result{};
    result.publicationSequenceId = ps.sequenceId;
    result.publicationEpoch = ps.epoch;
    result.mappedRuntimeGeneration = ps.mappedGeneration;
    result.hasActiveRuntime = (runtimeWorld != nullptr);

    // hasPendingPublication: sequenceId が進んでいるが world が null
    result.hasPendingPublication = (ps.sequenceId > 0 && runtimeWorld == nullptr);

    // hasActiveCrossfade: world の transitionActive を確認
    if (runtimeWorld != nullptr) {
        const auto& world = *static_cast<const RuntimePublishWorld*>(runtimeWorld);
        result.hasActiveCrossfade = world.execution.transitionActive;
    }

    return result;
}
```

### Recovery への統合（確定）

Recovery Architecture の Step1-6 において、deriveAuthorityState() は Step3 で使用される：

```
Step1: RuntimeStore 取得 → consumePublishedWorld()
Step2: PersistentState 取得 → persistentStateBlock.snapshot()
Step3: deriveAuthorityState(persistent, runtimeWorld) → 現在の状態把握
Step4: 状態比較 → 期待状態との差異判定
Step5: 不足状態補完 → 差分に基づく修復
Step6: Publish再開 → coordinator.publishWorld(...)
```

---

## 8. AuthoritySource 設計確定

### 設計方針

最小限のトレーサビリティ向上。Validator Telemetry（6000-6003）は既存のため、Authority Telemetry は簡易カウンタで十分。

### 定義（確定）

```cpp
enum class AuthoritySource : uint8_t {
    Unknown        = 0,
    UserAction     = 1,  // UI操作
    PresetLoad     = 2,  // プリセット読み込み
    Recovery       = 3,  // リカバリ発動
    DSPTransition  = 4,  // DSP遷移
    HealthMonitor  = 5,  // HealthMonitor発動
    _Count
};
```

### Telemetry（確定）

```cpp
// Authority Telemetry は分離カウンタ（Validator Telemetry とは別）
struct AuthorityTelemetry {
    std::atomic<uint64_t> callCount[6]{};

    void record(AuthoritySource src) noexcept {
        auto idx = static_cast<size_t>(src);
        if (idx < 6)
            convo::fetchAddAtomic(callCount[idx], 1u, std::memory_order_relaxed);
    }
};
```

### Coordinator API 拡張

```cpp
// publishWorld に source を追加（既存のオーバーロードを維持）
[[nodiscard]] PublishStageResult publishWorld(
    aligned_unique_ptr<World> worldOwner,
    AuthoritySource src = AuthoritySource::Unknown) noexcept;
```

---

## 9. テスト拡充計画（確定）

### Phase-B で追加すべきテストケース

優先度順：

| 優先度 | テスト種別 | テストケース | 分類 |
|---|---|---|---|
| **高** | Recovery 障害注入 | HealthState::Critical 時の Emergency Override 検証 | Integration |
| **高** | Recovery 障害注入 | Crossfade Timeout → publishIdleWorldOnly の動作検証 | Integration |
| **中** | Validator 追加 | generation > 0 で sequenceId == 0 の Semantic reject | Unit |
| **中** | Validator 追加 | generation > 0 で runtimeGeneration == 0 の Topology reject | Unit |
| **中** | Property Test | 10,000回ランダム publish シーケンス（単調増加契約） | Property |
| **低** | Validator 追加 | HardReset + fade == 0 + useDryAsOld == true の reject | Unit |
| **低** | Validator 追加 | oversamplingFactor = 16 の Accept | Unit |
| **低** | Validator 追加 | TransitionPolicy::DryAsOld + fade == 0 の Accept | Unit |

---

## 10. 最終フェーズ計画（確定版）

### Phase-A: 不足機能実装（推奨: 優先実施）

| Step | 項目 | ファイル影響範囲 | 依存 |
|---|---|---|---|
| A-1 | PersistentStateBlock 導入 | ISRRuntimePublicationCoordinator.h/.cpp + テスト | なし |
| A-2 | AuthoritySource 導入 | RuntimePublicationCoordinator.h（template）+ テスト | なし |
| A-3 | deriveAuthorityState() 実装 | 新規ファイル（または core/ 配下）+ テスト | A-1 |
| A-4 | Recovery Step1-6 統合 | AudioEngine.Timer.cpp + テスト | A-3 |
| A-5 | currentWorld_ 削除 | ISRRuntimePublicationCoordinator + ISRSemanticValidationTests | A-1 |

### Phase-B: テスト拡充（推奨: A完了後）

| Step | 項目 | 備考 |
|---|---|---|
| B-1 | Recovery 障害注入テスト | モック + 実 HealthMonitor 使用 |
| B-2 | Validator エッジケース追加 | 上記不足テストケース |
| B-3 | Property Test | ランダムシーケンス10,000回 |

---

## 11. 使用ツール一覧

| ツール | バージョン/状態 | 使用目的 |
|---|---|---|
| **Serena MCP** (oraios) | serena-agent v1.5.3 / clangd v19.1.2 / 236 files indexed | シンボル検索、参照関係追跡、ファイル構造把握 |
| **AiDex MCP** | v2.1.2 / 278 files, 12247 items, 4248 methods, 401 types | 高速識別子検索（grep代替、50倍トークン効率） |
| **CodeGraph MCP** | Python patched / 67 communities, 16397 entities | ファイル構造把握、依存関係分析 |
| **graphify MCP** | v0.8.37 / Masterplan: 325 edges | 知識グラフ中心ノード確認、アーキテクチャ把握 |
| **semble** (CLI) | uv tool / CPU ~1.5ms query | 自然言語セマンティック検索 |
| **Select-String** (grep) | PowerShell 標準 | パターン検索、不在確認 |
