# Practical Stable ISR Bridge Runtime 最終収束計画

**Version:** 1.8.0 (2026-06-18) — 実装準備完了版
**前提:** 現行達成率 92〜95%
**方針:** Phase-0a→0b→0c→1→1.5→2→2.5→3→4 で完了。本計画完了時の達成率 97〜98%

**注**: Phase-5（CrossfadeAuthorityRuntime固定長CAS化）, Phase-6（emitOnEvent統合）, Phase-7（CrossfadeSettings一括atomic化）は費用対効果・RT-safeリスクのため本計画から除外。P1-4（namespace変更）も不要と判断。Appendix D 参照。

---

## Phase-0a: 既存テストのコンパイルエラー修正（5分）

### 問題

`src/tests/PublicationValidatorIsolationTests.cpp` 内の2テストが、実際の struct 定義に存在しないフィールドを参照しておりコンパイルエラーになる。

| テスト | 該当コード | 問題 |
|--------|-----------|------|
| `ValidateTopology_BasicTopology_Success` | `world.routing.numSources = 2` | ❌ `RoutingSemantic` に `numSources` なし |
| 同上 | `world.routing.numDestinations = 2` | ❌ `RoutingSemantic` に `numDestinations` なし |
| `ValidateResources_BasicResources_Success` | `world.resource.memoryBudgetBytes = 1024 * 1024` | ❌ `ResourceSemantic` に `memoryBudgetBytes` なし |

### 修正内容

以下の3行を削除する（該当テストは Placeholder 検証用であり、実フィールドが存在しないため削除が唯一の正しい対応）。

```cpp
// ValidateTopology_BasicTopology_Success — 削除
world.routing.numSources = 2;      // ← RoutingSemantic に存在しない
world.routing.numDestinations = 2; // ← RoutingSemantic に存在しない

// ValidateResources_BasicResources_Success — 削除
world.resource.memoryBudgetBytes = 1024 * 1024; // ← ResourceSemantic に存在しない
```

### 修正後コード

**ValidateTopology_BasicTopology_Success**:

```cpp
TEST_F(PublicationValidatorIsolationTests, ValidateTopology_BasicTopology_Success) {
    // Arrange: 基本的な topology（Placeholder 検証のため最小フィールドのみ）
    RuntimePublishWorld world{};
    world.generation = 1;

    // Act
    const bool isValid = validator_.validateTopology(world);

    // Assert
    EXPECT_TRUE(isValid);
}
```

**ValidateResources_BasicResources_Success**:

```cpp
TEST_F(PublicationValidatorIsolationTests, ValidateResources_BasicResources_Success) {
    // Arrange: 基本的な resource（Placeholder 検証のため最小フィールドのみ）
    RuntimePublishWorld world{};
    world.generation = 1;

    // Act
    const bool isValid = validator_.validateResources(world);

    // Assert
    EXPECT_TRUE(isValid);
}
```

### 目的

- コンパイルエラー解消により、現行の全テストが通過可能になる
- Validator 実体化（Phase-1）前にテスト基盤を正常化する

### リスク

極小。テストの削除内容（非存在フィールド）は本来テストとして機能していなかった。

---

## Phase-0b: useDryAsOld Dormant Bug 除去（5分）

### 修正内容

`RuntimeBuilder.cpp L287` の 1行のみ。

```cpp
// 修正前:
worldOwner->overlap.useDryAsOld = active;

// 修正後:
worldOwner->overlap.useDryAsOld = (policy == convo::TransitionPolicy::DryAsOld);
```

### 目的

`useDryAsOld` と `transitionActive` は全く別概念。`useDryAsOld` は `TransitionPolicy::DryAsOld` が選択された場合のみ `true` となるべき。
現状は `active=true` の全ケースで誤って `useDryAsOld=true` になる休眠バグ。将来 `setFirstIrDryPending(true)` が実装された際に顕在化するため予防修正。

### リスク

極小。変更は1行、フォールバック動作不変。

---

## Phase-0c: Dead Code 除去（5分）

### 修正内容

`CrossfadeRuntime.h` の以下の2メソッドは全ソースで呼び出し元ゼロのデッドコード。削除する。

```cpp
// CrossfadeRuntime.h — 削除
void setUseDryAsOld(bool v) noexcept
    { convo::publishAtomic(useDryAsOld_, v, std::memory_order_release); }
void setFirstIrDryPending(bool v) noexcept
    { convo::publishAtomic(firstIrDryPending_, v, std::memory_order_release); }
```

### 目的

- 未使用コードの削除による保守性向上
- `useDryAsOld` / `firstIrDryPending` の書き込み経路を明確化（Builder経路のみに統一）

### リスク

極小。呼び出し元ゼロのため削除による影響はない。

---

## Phase-1: RuntimePublicationValidator 実体化（半日）

### 設計原則

- Validator は「世界の整合性」のみ検査し「運用ポリシー」は検査しない
- 既存のランタイム不変条件（`AudioEngine.Commit.cpp`）を publish 前検証に昇格

### P1-1: validateTopology()

```cpp
// Bootstrap以外は runtimeUuid 必須
if (world.generation > 0 && world.topology.runtimeUuid == 0)
    return false;

// hasFadingRuntime と fadingRuntimeUuid の整合性
if (world.topology.hasFadingRuntime != (world.topology.fadingRuntimeUuid != 0))
    return false;

// hasFadingRuntime と transitionActive の自己整合性（RuntimeWorld Semantic の不変条件）
if (world.topology.hasFadingRuntime != world.execution.transitionActive)
    return false;
```

### P1-2: validateResources()

```cpp
// Oversampling: 2のべき乗かつ1〜16
const int os = world.resource.oversamplingFactor;
if (os < 1 || os > 16 || (os & (os - 1)) != 0)
    return false;

// Dither: 0, 16, 24 のみ許容
const int dd = world.resource.ditherBitDepth;
if (dd != 0 && dd != 16 && dd != 24)
    return false;

// NoiseShaper: 0, 1, 2 のみ許容
const int ns = world.resource.noiseShaperType;
if (ns < 0 || ns > 2)
    return false;
```

### P1-3: checkNoConflictingTransitions()

```cpp
const auto policy = static_cast<convo::TransitionPolicy>(world.execution.transitionPolicy);
const bool active = world.execution.transitionActive;
const double fade = world.overlap.fadeTimeSec;

if (!active) {
    // ★ transitionActive=false でも fadeTimeSec が残るケースを許容
    //   フェード完了直後の Idle World publish 時など、fadeTimeSec が保持されたまま
    //   遷移する将来実装が入り得る。そのため fade > 0.0 は reject しない。
    //   ただし負の fade は常に異常値として reject。
    if (fade < 0.0) return false;
    // useDryAsOld=true かつ !active は意味論的に矛盾
    if (world.overlap.useDryAsOld) return false;
    return true;
}

switch (policy) {
    case convo::TransitionPolicy::SmoothOnly:
        if (fade < 0.0) return false;  // 負値のみ拒否（0.0はフォールバック機構に委ねる）
        break;
    case convo::TransitionPolicy::DryAsOld:
        if (fade < 0.0) return false;
        if (!world.overlap.useDryAsOld) return false;
        break;
    case convo::TransitionPolicy::HardReset:
        if (fade < 0.0) return false;  // 負値も拒否
        if (fade > 0.0) return false;  // 正値も拒否（HardReset は fade=0.0 のみ許容）
        if (world.overlap.useDryAsOld) return false;
        break;
    default:
        return false;  // 未知の policy 値は拒否
```

### validatePublication での failureReason 設定

```cpp
// RuntimePublicationValidator.cpp — validatePublication 内の各チェックで failureReason を設定
RuntimeValidationResult RuntimePublicationValidator::validatePublication(
    const RuntimePublishWorld& world) const
{
    RuntimeValidationResult result;

    if (!validateSemanticConsistency(world)) {
        result.isValid = false;
        result.errorMessage = "Semantic consistency check failed";
        result.failureReason = ValidationFailureReason::SemanticInconsistency;
        return result;
    }
    if (!validateTopology(world)) {
        result.isValid = false;
        result.errorMessage = "Topology validation failed";
        result.failureReason = ValidationFailureReason::InvalidTopology;
        return result;
    }
    if (!validateResources(world)) {
        result.isValid = false;
        result.errorMessage = "Resource availability check failed";
        result.failureReason = ValidationFailureReason::InvalidResources;
        return result;
    }
    if (!checkNoConflictingTransitions(world)) {
        result.isValid = false;
        result.errorMessage = "Conflicting transitions detected";
        result.failureReason = ValidationFailureReason::InvalidTransition;
        return result;
    }
    return result;
}
```

### 完了条件

`RuntimePublicationValidator` 内に `return true; // Placeholder` がゼロになること。

<!-- P1-4 は削除: namespace変更は利益が小さく、include/using/テストへの波及リスクが大きいため -->

---

## Phase-1.5: Validator Telemetry（2時間）

### 追加する型とイベントコード

```cpp
// RuntimePublicationValidator.h — Validator の公開APIとして ValidationFailureReason を定義
enum class ValidationFailureReason : uint8_t {
    None,
    InvalidTopology,
    InvalidResources,
    InvalidTransition,
    SemanticInconsistency
};

// RuntimeHealthMonitor.h
static constexpr uint32_t EVENT_VALIDATION_SEMANTIC_FAILURE     = 6000;
static constexpr uint32_t EVENT_VALIDATION_TOPOLOGY_FAILURE   = 6001;
static constexpr uint32_t EVENT_VALIDATION_RESOURCE_FAILURE   = 6002;
static constexpr uint32_t EVENT_VALIDATION_TRANSITION_FAILURE = 6003;
```

### RuntimeValidationResult 拡張

```cpp
// RuntimePublicationValidator.h
struct RuntimeValidationResult {
    bool isValid = true;
    std::string errorMessage;
    ValidationFailureReason failureReason{ValidationFailureReason::None};
};
```

### 依存関係の注意

`ValidationFailureReason` は `RuntimePublicationValidator.h`（`iso::audio_engine` 名前空間）で定義。
`RuntimeHealthMonitor` はこれを直接 include していないため、`RuntimeHealthMonitor.h` に前方宣言を追加する:

```cpp
// RuntimeHealthMonitor.h — ValidationFailureReason の前方宣言
//   C++20 では underlying type 指定で enum class の前方宣言が可能
#include <array>  // ★ std::array を使用するために追加
namespace iso::audio_engine {
    enum class ValidationFailureReason : uint8_t;
}
```

**注意**: Validator（`iso::audio_engine` 名前空間）と Semantic 構造体（`convo::isr` 名前空間）の名前空間は異なるが、`RuntimePublishWorld` = `::RuntimeState`（global scope）のためフィールドアクセスに影響はない。Phase-1 で `iso::audio_engine::RuntimePublicationValidator` として参照すれば問題ない。include 地獄を避けるため、`RuntimeHealthMonitor.h` では前方宣言のみに留め、`RuntimePublicationValidator.h` の直接 include は行わないこと。

### HealthMonitor への public メソッド追加

```cpp
// RuntimeHealthMonitor.h
class RuntimeHealthMonitor {
public:
    void emitValidationEvent(ValidationFailureReason reason) noexcept;
private:
    static constexpr size_t kValidationReasonCount = 4;  // Semantic, Topology, Resource, Transition
    std::array<std::atomic<uint64_t>, kValidationReasonCount> m_lastValidationEventUs_{};
    static constexpr uint64_t kValidationEventMinIntervalUs = 1'000'000;
};
```

```cpp
// RuntimeHealthMonitor.cpp
void RuntimeHealthMonitor::emitValidationEvent(ValidationFailureReason reason) noexcept {
    uint32_t eventCode = 0;
    size_t idx = 0;
    switch (reason) {
        case ValidationFailureReason::SemanticInconsistency:
            eventCode = 6000; idx = 0; break;
        case ValidationFailureReason::InvalidTopology:
            eventCode = 6001; idx = 1; break;
        case ValidationFailureReason::InvalidResources:
            eventCode = 6002; idx = 2; break;
        case ValidationFailureReason::InvalidTransition:
            eventCode = 6003; idx = 3; break;
        default: return;
    }
    // ★ Validation failure は publish thread 単一からのみ発生。
    //   CAS は過剰設計。単純な load + store で十分。
    const uint64_t last = convo::consumeAtomic(
        m_lastValidationEventUs_[idx], std::memory_order_acquire);
    const uint64_t nowUs = convo::getCurrentTimeUs();
    if (nowUs - last >= kValidationEventMinIntervalUs) {
        convo::publishAtomic(m_lastValidationEventUs_[idx], nowUs, std::memory_order_release);
        if (m_callback)
            m_callback(convo::HealthEvent{nowUs, convo::HealthEvent::Severity::Warning,
                                         eventCode, 0, 0});
    }
}
```

### AudioEngine.Commit.cpp 呼び出し側

```cpp
if (!validationResult.isValid) {
    m_healthMonitor.emitValidationEvent(validationResult.failureReason);
    return false;
}
```

---

## Phase-2: CrossfadePolicy 抽出（1〜2日）

### P2-1: CrossfadePolicy 構造体

```cpp
// CrossfadeAuthority.h に追加（namespace convo::isr 内）
// ★ CrossfadePolicy: immutable POD。メソッドや状態を持たない。
//   静的設定（フェード時間・閾値）のみを保持し、実行時状態（HealthState）は含めない。
//   将来 CrossfadeSettings 一括atomic化との統合を考慮し zero-init 必須。
struct CrossfadePolicy {
    double osFadeTimeSec{};
    double irFadeTimeSec{};
    double irLengthFadeTimeSec{};
    double phaseFadeTimeSec{};
    double directHeadFadeTimeSec{};
    double nucFilterFadeTimeSec{};
    double tailFadeTimeSec{};
};
```

**設計意図**: 「Policy = 静的設定、HealthState = 実行時状態」と明確に分離。
`crossfadeAllowed` を Policy に含めると「フェード時間」と「実行時メトリクス」が混在し責務が曖昧になる。
Critical 時の crossfade 抑制は Orchestrator（または DSPTransition Emergency Override）が担当する。

### P2-2: evaluate() シグネチャ変更

```cpp
// 修正前:
Decision evaluate(const AudioEngine& engine, const RuntimePublishWorld& oldWorld,
                  const RuntimePublishWorld& newWorld) noexcept;

// 修正後:
Decision evaluate(const RuntimePublishWorld& oldWorld, const RuntimePublishWorld& newWorld,
                  const CrossfadePolicy& policy) noexcept;
```

### P2-3: evaluate() 実装（engine直読→policy参照）

```cpp
Decision evaluate(const RuntimePublishWorld& oldWorld, const RuntimePublishWorld& newWorld,
                  const CrossfadePolicy& policy) noexcept
{
    Decision ctx;
    // ★ evaluate は純粋に dspProjection 投影値 + Policy 静的設定のみで判断
    //   Critical 時の crossfade 抑制は Orchestrator（makeCrossfadePolicy 後 evaluate 前）または
    //   DSPTransition Emergency Override が担当する。evaluate 自身は HealthState を知らない。

    ctx.oldHasIR = oldWorld.dspProjection.irLoaded;
    ctx.newHasIR = newWorld.dspProjection.irLoaded;
    const bool hasTransition = ctx.oldHasIR || ctx.newHasIR;
    const bool irChanged = (ctx.oldHasIR != ctx.newHasIR);

    if (hasTransition && newWorld.dspProjection.oversamplingFactor != oldWorld.dspProjection.oversamplingFactor) {
        ctx.needsCrossfade = true;
        ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.osFadeTimeSec);
    }
    if (hasTransition) {
        const uint64_t oh = oldWorld.dspProjection.structuralHash;
        const uint64_t nh = newWorld.dspProjection.structuralHash;
        if (oh != nh) {
            ctx.needsCrossfade = true;
            if (irChanged)
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, std::clamp(policy.irFadeTimeSec, 0.001, 0.010));
            else {
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.irFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.irLengthFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.phaseFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.directHeadFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.nucFilterFadeTimeSec);
                ctx.fadeTimeSec = std::max(ctx.fadeTimeSec, policy.tailFadeTimeSec);
            }
        }
    }
    return ctx;
}
```

### P2-4: AudioEngine factory method

```cpp
// AudioEngine.h — NonRT(MessageThread) → Policy 生成時に acquire で読み取り
//   書き込み元: AudioEngine::prepareToPlay() (MessageThread) :: release
[[nodiscard]] CrossfadePolicy makeCrossfadePolicy() const noexcept {
    CrossfadePolicy p;
    p.irFadeTimeSec       = convo::consumeAtomic(m_irFadeTimeSec,       std::memory_order_acquire);
    p.phaseFadeTimeSec    = convo::consumeAtomic(m_phaseFadeTimeSec,    std::memory_order_acquire);
    p.tailFadeTimeSec     = convo::consumeAtomic(m_tailFadeTimeSec,     std::memory_order_acquire);
    p.osFadeTimeSec       = convo::consumeAtomic(m_osFadeTimeSec,       std::memory_order_acquire);
    p.irLengthFadeTimeSec = convo::consumeAtomic(m_irLengthFadeTimeSec, std::memory_order_acquire);
    p.directHeadFadeTimeSec = convo::consumeAtomic(m_directHeadFadeTimeSec, std::memory_order_acquire);
    p.nucFilterFadeTimeSec  = convo::consumeAtomic(m_nucFilterFadeTimeSec,  std::memory_order_acquire);
    // ★ HealthState は Policy に入れない — Orchestrator または DSPTransition が判断する
    return p;
}
```

### P2-5: Orchestrator 呼び出し側

```cpp
// RuntimePublicationOrchestrator.cpp
// ★ Critical 時は Orchestrator が直接判断し、crossfade を強制抑制する。
//   evaluate は純粋に dspProjection + Policy のみで判定。Critical 抑制は Orchestrator の責務。
//   DSPTransition Emergency Override が最終安全網（TOCTOU対策）。

auto policy = engine_.makeCrossfadePolicy();
CrossfadeAuthority crossfade;
auto cfDecision = crossfade.evaluate(*oldWorld, *worldOwner, policy);

// ★ HealthState Critical 時は crossfade を強制抑制
//   evaluate の結果を上書きする形で、Orchestrator レベルでの抑制を行う。
//   （注意: return ctx のような擬似コードは使用しない — Orchestrator は関数の途中で return しない）
{
    auto ref = engine_.getHealthStateRef();
    if (ref) {
        auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
        if (health == convo::ISRHealthState::Critical) {
            cfDecision.needsCrossfade = false;
            cfDecision.fadeTimeSec = 0.0;
        }
    }
}

// ★ 以降の既存コードは cfDecision.needsCrossfade を参照するため、
//   Critical 時は自然に crossfade がスキップされる。
```

### 完了条件

1. `CrossfadeAuthority::evaluate()` が `AudioEngine&` を一切受け取らないこと
2. `CrossfadePolicy` に実行時状態（HealthState）が含まれていないこと
3. Critical 時の crossfade 抑制が Orchestrator + DSPTransition Emergency Override の2段階で保証されていること

---

## Phase-2.5: DSPTransition Emergency Override 公式化（2時間）

### 背景

`DSPTransition::onPublishCompleted()` 内の HealthState::Critical チェックは、Admission判断（Time-Of-Check）から DSPTransition実行（Time-Of-Use）までの間（数百μs〜数ms）にシステムが Critical に陥った場合の**最終安全網（TOCTOU対策）**。削除せず Emergency Override として公式化する。

### P2.5-1: DSPTransition — Emergency Override 維持

```cpp
// DSPTransition.h
void onPublishCompleted(..., DSPLifetimeManager& lifetime) noexcept
{
    // ★ Emergency Override: TOCTOU 対策（Admission 通過後 Critical 検知の最終安全網）
    {   auto ref = engine_.getHealthStateRef();
        if (ref) {
            auto health = convo::consumeAtomic(*ref, std::memory_order_acquire);
            if (health == convo::ISRHealthState::Critical) {
                lifetime.activate(newDSP);
                if (oldDSP != nullptr) {
                    // ★ DSPTransition は Authority Registry に触れない。
                    //   Authority cleanup（getActiveCrossfades → unregisterCrossfade）は
                    //   Timer(MessageThread)上の onHealthEvent ハンドラで行う。
                    //   ConvoPeq の既存設計では、Authority Registry 操作は Timer 経路に
                    //   統一されている（通常完了・Timeout・Emergency の3経路とも Timer）。
                    engine_.crossfadeRuntime_.complete();
                    lifetime.retire(oldDSP);
                    // ★ enqueueHealthEvent で非同期投入（層の逆流＋同期実行防止）
                    //   DSPTransition が直接 onHealthEvent() を呼ぶと publish 経路から
                    //   Health 経路への同期依存が生じる。代わりに非同期キューに投入し、
                    //   Timer スレッド側で deque して onHealthEvent を非同期実行する。
                    //   これにより publish 経路と Health 経路の再結合を防止する。
                    const uint64_t abortCount = engine_.crossfadeRuntime_.incrementEmergencyAbortCount();
                    engine_.enqueueHealthEvent(convo::HealthEvent{convo::getCurrentTimeUs(),
                        convo::HealthEvent::Severity::Warning,
                        EVENT_CROSSFADE_ABORTED_EMERGENCY,
                        abortCount, 0});
                }
                return;  // 通常のクロスフェード処理をスキップ
            }
        }
    }
    // ... 通常のクロスフェード処理
}
```

### P2.5-2: 追加要素

```cpp
// RuntimeHealthMonitor.h — イベントコード（HealthMonitor が一元管理、クロスフェード系は4000番台）
static constexpr uint32_t EVENT_CROSSFADE_ABORTED_EMERGENCY = 4003;

// CrossfadeRuntime.h — Emergency Abort カウンター（CrossfadeRuntime が所有）
//   ★ 複数回の Emergency Override を区別するための単調増加カウンター
//   DSPTransition が値を increment + fetch し、HealthEvent.value に乗せて通知する。
[[nodiscard]] uint64_t emergencyAbortCount() const noexcept
    { return convo::consumeAtomic(m_emergencyAbortCount_, std::memory_order_acquire); }
// ★ increment + fetch を原子的に実行（DSPTransition から呼ばれる）
uint64_t incrementEmergencyAbortCount() noexcept
    { return convo::fetchAddAtomic(m_emergencyAbortCount_, 1u, std::memory_order_acq_rel) + 1; }

// CrossfadeRuntime.h — private メンバに追加
std::atomic<uint64_t> m_emergencyAbortCount_{0};

// AudioEngine.h — 通知先
// ★ enqueueHealthEvent: DSPTransition 等の Transition Layer から非同期で HealthEvent を投入。
//   内部で SPSC キューに push し、Timer スレッドの tick で deque → onHealthEvent() を実行する。
//   これにより publish 経路と Health 経路の同期実行を防止する。
//   実装は CompletedFadeEvent と同様の SPSCRingBuffer パターン（CrossfadeRuntime completedFadeQueue_ 参照）。
//   容量は 4 で十分（緊急イベントは稀）。オーバーフロー時は push 失敗＝イベント欠落となるため、
//   デバッグ時には missing カウンタを別途設けて監視することが推奨される。
//   HealthEvent の trivially_copyable 要件を満たすため、メンバに std::string 等を含めないこと。
void enqueueHealthEvent(const convo::HealthEvent& event) noexcept
    { /* SPSCRingBuffer<convo::HealthEvent, 4> push → Timer tick で deque */ }
```

### P2.5-3: AudioEngine::onHealthEvent() での処理

```cpp
// AudioEngine.Timer.cpp — 既存の onHealthEvent() 内に追加
// ★  パターン: EVENT_CROSSFADE_TIMEOUT と同じ回復経路を流用
//    Authority Registry 操作（getActiveCrossfades → unregisterCrossfade）を含む
//    完全な回復処理は Timer(MessageThread) 経路に統一する。
if (event.eventCode == convo::EVENT_CROSSFADE_ABORTED_EMERGENCY) {
    diagLog("[HEALTH] Crossfade aborted by Emergency Override"
        + juce::String(" count=") + juce::String(static_cast<juce::int64>(event.value)));

    // 1. ★ Authority cleanup: CrossfadeAuthorityRuntime の active レコードを全消去
    //    （通常完了・Timeout・Emergency の3経路とも Timer で統一）
    {
        auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
        for (const auto& record : records)
            crossfadeAuthorityRuntime_.unregisterCrossfade(record.id);
    }

    // 2. World/Runtime 乖離修復: publishIdleWorldOnly で hasFadingRuntime=false の world を発行
    crossfadeRuntime_.setDryHoldSamples(0);
    refreshCrossfadePreparedSnapshotFromAtomics();
    {
        const convo::RuntimeReaderContext messageCtx{
            messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        auto* currentAfterFade =
            resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        (void)publishIdleWorldOnly(currentAfterFade,
            convo::TransitionPolicy::HardReset);
    }

    diagLog("[HEALTH] Emergency Override recovery completed");
}
```

**層の分離と非同期処理**: DSPTransition は `onHealthEvent()` を直接呼ばず、`enqueueHealthEvent()` で非同期キューに投入する。Timer スレッドの tick で deque され、`onHealthEvent()` が実行される。これにより:

1. **層の逆流防止**: Transition Layer → Health Layer の依存を非境界にする
2. **同期実行防止**: publish 成功直後（Coordinator → Executor → DSPTransition）の経路で onHealthEvent が同期的に実行されるのを防ぐ
3. **既存パターン一致**: `CompletedFadeEvent` と同じ SPSC キュー・パターン（`CrossfadeRuntime completedFadeQueue_`）

**設計意図**: Emergency Override 後、DSP 状態は idle に戻るが、**World は `hasFadingRuntime=true` のまま残留する**（Emergency Override は publish を伴わないため、古い world が残る）。Admission がこの状態を見て publish を defer するのを防ぐため、明示的に `publishIdleWorldOnly()` で `hasFadingRuntime=false` の新 world を発行する必要がある。

これは `RuntimeBuilder::buildRuntimePublishWorld()` が `hasFading = (graphState.fadingNode != nullptr)` で設定し、Emergency Override 後は `fadingNode == nullptr` であるため、新しい world では `hasFadingRuntime=false` となる。

したがって `publishIdleWorldOnly()` は**必須**であり、省略すると `hasFadingRuntime=true` が永久残留する。

このパターンは `EVENT_CROSSFADE_TIMEOUT` と同じ回復パターンであり、ConvoPeq 現行実装の設計に準拠する。

**Coordinator 経路の維持**: `publishIdleWorldOnly()` は内部的に `RuntimePublicationCoordinator::publishWorld()` を呼び出す。この Coordinator は Orchestrator が使用するものと**同一の Coordinator/Executor 経路**であり、以下を経由する:

1. `worldOwner->sealRecursively()` — freeze
2. `bridge_.validatePublicationNonRt(*worldOwner)` — validation
3. `writeAccess_.publishAndSwap(newWorld)` — atomic swap
4. `bridge_.retireRuntimePublishWorldNonRt(oldWorld)` — retire

したがって `publishIdleWorldOnly()` は Coordinator をバイパスせず、Practical Stable ISR Bridge Runtime の「Coordinator が唯一の Publish Authority」原則に合致する。既存の DSPTransition 正常完了経路（`DSPTransition.h:132`）および Timeout 回復経路（`AudioEngine.Timer.cpp:641`）でも同一パターンが使用されている。

**注意**: `onHealthEvent()` は Timer スレッド(MessageThread)上でコールバックとして実行される。このコンテキストでの publish は安全（Non-RT）。

**Authority cleanup の Timer 統一**: ConvoPeq の既存設計では、`CrossfadeAuthorityRuntime` Registry の操作はすべて Timer(MessageThread) 経路に統一されている。

| 経路 | 操作箇所 | Registry cleanup |
|------|---------|-----------------|
| 通常完了 | `AudioEngine.Timer.cpp:384-398` | ✅ `getActiveCrossfades()` → `notifyFadeComplete()` → `consumeCompletedFade()` → `unregisterCrossfade()` |
| Timeout | `AudioEngine.Timer.cpp:617-625` | ✅ `getActiveCrossfades()` → `unregisterCrossfade()` |
| Emergency（旧案） | `DSPTransition.h` | ❌ **DSPTransition に分散** — 危険 |
| **Emergency（新案）** | **`AudioEngine.Timer.cpp`** (onHealthEvent) | ✅ **Timer 統一** — 安全 |

DSPTransition は DSP ライフサイクル（activate / complete / retire）のみを担当し、Authority Registry には触れない。これにより Authority 操作が Timer 経路に一本化され、競合リスクが低減する。

### P2.5-4: armCrossfadeIfPending — 無限再Arm防止

```cpp
// AudioEngine.h — armCrossfadeIfPending 内の条件
const bool hasPendingCrossfade = prepared.pending && crossfadeRuntime_.isPending();

if (!hasPendingCrossfade) {
    dspCrossfadeArmed_RT = false;
    dspCrossfadeStartDelayBlocks_RT = 0;
    return;
}
```

**根拠**: Emergency Override で `crossfadeRuntime_.complete()` が呼ばれても、古いWorldスナップショットが `pending=true` を返すため毎回再Armされる。`crossfadeRuntime_.isPending()` のAND条件により「Worldは要求しているがRuntimeは既に完了」を検出し、Armを抑制。

---

## Phase-3: テスト追加（半日〜1日）

### P3-1: Validator Reject テスト

```
hasFadingRuntime=true, fadingRuntimeUuid=0          → Reject
oversamplingFactor=3                                 → Reject
ditherBitDepth=8                                      → Reject
HardReset + fadeTimeSec=0.5                           → Reject
SmoothOnly + transitionActive=true + fadeTimeSec=-0.1 → Reject
DryAsOld + useDryAsOld=false                          → Reject
```

### P3-2: Validator Accept テスト

```
Bootstrap world (generation=0, runtimeUuid=0)         → Accept
SmoothOnly + transitionActive=true + fadeTimeSec=0.0   → Accept
SmoothOnly + transitionActive=true + fadeTimeSec>0     → Accept
Idle world (transitionActive=false, fadeTimeSec>0)       → Accept
HardReset + transitionActive=false                       → Accept
HardReset + transitionActive=true + fadeTimeSec=0.0      → Accept
DryAsOld + useDryAsOld=true + fadeTimeSec>0              → Accept
```

### P3-3: CrossfadeAuthority Regression テスト

```cpp
TEST(CrossfadeAuthorityRegressionTest, DeterministicDecision) {
    auto oldW = makeStandardOldWorld(), newW = makeStandardNewWorld();
    auto policy = makeStandardPolicy();
    CrossfadeAuthority auth;
    auto d1 = auth.evaluate(oldW, newW, policy);
    auto d2 = auth.evaluate(oldW, newW, policy);
    EXPECT_EQ(d1.needsCrossfade, d2.needsCrossfade);
    EXPECT_DOUBLE_EQ(d1.fadeTimeSec, d2.fadeTimeSec);
}

TEST(CrossfadeAuthorityRegressionTest, PolicyChangeChangesDecision) {
    auto oldW = makeWorldWithIR(), newW = makeWorldWithDifferentIR();
    CrossfadePolicy fast = makeFastFadePolicy(), slow = makeSlowFadePolicy();
    CrossfadeAuthority auth;
    auto dFast = auth.evaluate(oldW, newW, fast);
    auto dSlow = auth.evaluate(oldW, newW, slow);
    EXPECT_TRUE(dFast.needsCrossfade);
    EXPECT_TRUE(dSlow.needsCrossfade);
    EXPECT_LT(dFast.fadeTimeSec, dSlow.fadeTimeSec);
}
```

---

## Phase-4: Validator 網羅率拡充 + Builder/Validator 責務明確化（1日）

### 背景

Phase-1 で追加した Validator チェックは Topology / Resource / Transition の3種に限られる。しかし publication semantic 全体として以下が未カバーである。また RuntimeBuilder と Validator の責務範囲が不明確であり、二重管理状態にある。

### P4-1: Validator ルール追加

```cpp
// RuntimePublicationValidator.cpp — validateTopology / validateResources に追記

// RoutingSemantic: processingOrder は 0 または 1 のみ許容
if (world.routing.processingOrder < 0 || world.routing.processingOrder > 1)
    return false;

// GenerationSemantic: runtimeGeneration と activationEpoch の単調増加は
// 1つの publish 内では検証不能。代わりに generation > 0 なら runtimeGeneration > 0 を確認
if (world.generation > 0 && world.generationSemantic.runtimeGeneration == 0)
    return false;

// Publication sequence: generation > 0 なら sequenceId が 0 でないこと
//   Bootstrap world (generation=0) でも sequenceId は通常 1 以上だが、
//   generation を Bootstrap 判別の唯一の基準とする（runtimeUuid と同様のパターン）
if (world.generation > 0 && world.publication.sequenceId == 0)
    return false;

<!-- LatencySemantic 整合性チェックは削除: Builder 計算結果の検証であり Validator の責務を超える。Builder の unit test に委ねる。 -->
```

### P4-2: Builder/Validator 責務定義（文書化のみ）

`RuntimeBuilder` と `RuntimePublicationValidator` の責務を以下のように明確化する:

| レイヤ | 責務 | 根拠 |
|--------|------|------|
| RuntimeBuilder | **semantic 値の正しい設定** | Builder は各フィールドに適切な値を設定することが責務 |
| Validator | **不変条件の最終確認** | Builder の設定漏れ・バグを検出する安全網 |
| Orchestrator | **運用ポリシーの適用** | HealthState 等の実行時状態に基づく上書き判断 |

**原則**: Builder の通過が Validator 通過を保証するわけではない。Validator は Builder とは独立した Permissionless Check として動作する。両者が一致していることは望ましいが、Validator は Builder の実装詳細を知らず、純粋に世界の状態だけを検証する。

### 完了条件

- 上記の Validator ルールがすべて実装されていること
- Builder/Validator 責務定義が文書化されていること

---

## 優先順位

```text
Phase-0a:  既存テストコンパイルエラー修正        [5分]    ★★★★★
Phase-0b:  useDryAsOld 修正                        [5分]    ★★★★★
Phase-0c:  Dead Code 除去                          [5分]    ★★★★★
Phase-1:   Validator 実体化                        [半日]   ★★★★★
Phase-1.5: Validator Telemetry 追加               [2時間]  ★★★★☆
Phase-2:   CrossfadePolicy 抽出                    [1-2日]  ★★★★★
Phase-2.5: DSPTransition Emergency Override化      [2時間]  ★★★★★
Phase-3:   テスト追加                              [半日-1日] ★★★★☆
Phase-4:   Validator 網羅率拡充 + Builder責務明確化 [1日]   ★★★★☆
```

## 達成率評価

```text
現状:             92〜95%
全Phase完了後:    97〜98%
```

**注**: Phase-5（CrossfadeAuthorityRuntime固定長CAS化）, Phase-6（emitOnEvent統合）, Phase-7（CrossfadeSettings一括atomic化）は費用対効果・RT-safeリスクのため除外。P1-4（namespace変更）も不要。これらを除外しても達成率 97〜98% に到達する。

---

## Appendix A: 改訂履歴

### A-1: v1.2→v1.3 での修正

| 指摘 | テーマ | 結論 |
|------|--------|------|
| A | Phase-2.5 DSPTransition削除→TOCTOU脆弱性 | **維持**。Emergency Override + Eventual Consistencyとして公式化 |
| B | Telemetry データレース | `std::atomic<uint64_t>` + `emitValidationEvent()` で修正 |
| C | Torn Read | 7個の `atomic<double>` は lock-free / wait-free であり、実運用故障モード未確認のため本計画では対処しない。理論上のリスクとして Appendix D に記録。 |
| D | RT 250ms乖離期間 | `armCrossfadeIfPending` に `crossfadeRuntime_.isPending()` 追加 |
| E | exchange シグナルロスト | `exchangeAtomic` で原子的に読み取り＋クリア |
| F | Telemetry CAS | load+store で十分。CAS不要 |

### A-2: v1.1→v1.2 での修正

| 指摘 | テーマ | 結論 |
|------|--------|------|
| 1 | `fade <= 0` 過剰Reject | `fade < 0.0` のみRejectに緩和 |
| 2 | Telemetry イベントストーム | 6000-6003 + 1sレート制限追加 |
| 3 | Orchestrator知識過多 | `AudioEngine::makeCrossfadePolicy()` factory method |
| 4 | DSPTransition HealthState | v1.2: 削除方針 → v1.3で撤回 |

### A-3: v1.0→v1.1 基本方針

- Validatorルール緩和
- Health依存の段階的除去
- Validator Telemetry 新設（Phase-1.5）

---

## Appendix B: 保留項目

### B-1: Dead Code

`CrossfadeRuntime::setUseDryAsOld()` / `setFirstIrDryPending()` — **grep/Serena/AiDex 全ツールで呼び出し元ゼロ確認済み。** Phase-0c で削除。

### B-2: Semantic Schema

`kRuntimeSemanticSchemaVersion` → `9u`（現在 `8u`, `ISRRuntimeSemanticSchema.h:9` で定義。Emergency Override 公式化を記録）。
更新箇所: `ISRRuntimeSemanticSchema.h:9` の1行のみ。9箇所の参照はすべて自動反映。

### B-3: 命名整理

| 名前 | 用途 | Phase |
|------|------|-------|
| `CrossfadePolicy` | evaluate() に渡す per-call 設定 | Phase-2 |
| `CrossfadeSettings` | 将来の一括 atomic struct（Torn Read 対策） | 本計画対象外（Appendix D） |

### B-4: original_plan.md からの差分

| 項目 | original_plan.md | v1.4.1 |
|------|-----------------|--------|
| Validatorルール | 厳格一致を必須 | `fade==0` 許容（負値のみReject） |
| Health依存 | 「削除」のみ | Emergency Override として維持 |
| Validator Telemetry | 未記載 | 6000-6003 + `emitValidationEvent()` |
| Orchestrator責務 | 未考察 | `makeCrossfadePolicy()` factory method |
| Regression Test | 未記載 | 決定論的テスト + Policy差し替えテスト |
| DSPTransition | Phase-4 LOW | Phase-2.5 重点修正 |
| 達成率 | 100% | 99%（漸近的） |

---

## Appendix C: ソースコード調査結果（2026-06-18）

本計画策定にあたり、以下の全ツールを用いた徹底調査を実施:

- **grep (Select-String)**: 全ソースファイル横断検索（40+回）
- **AiDex MCP**: セッション確認（275 files, 4180 methods, 398 types）
- **graphify MCP**: 知識グラフ （17058 nodes, 22445 edges, 1456 communities）
- **Serena MCP**: シンボル検索・パターン横断・依存関係分析
- **read_file**: 主要30+ファイルの全行読取
- **CodeGraph MCP**: モジュール構造分析

### C-1: 計画の全主張と実際のコード一致確認

| # | 計画の主張 | 調査結果 | 判定 |
|---|---|---|---|
| P0 | `useDryAsOld = active` が休眠バグ | `RuntimeBuilder.cpp:287` で確認 | ✅ |
| P1 | Validator 3メソッドがPlaceholder | 全3メソッド `return true;` のみ | ✅ |
| P1 | `TopologySemantic` フィールド一致 | `runtimeUuid`, `fadingRuntimeUuid`, `hasFadingRuntime` 存在確認 | ✅ |
| P1 | `ResourceSemantic` フィールド一致 | `oversamplingFactor`, `ditherBitDepth`, `noiseShaperType` 存在確認 | ✅ |
| P1 | TransitionPolicy enum値一致 | `SmoothOnly=0`, `HardReset=1`, `DryAsOld=2` (`ISRRuntimeSemanticSchema.h`) | ✅ |
| P2 | CrossfadeAuthority がengine直読（7個のfade時間） | `engine.m_*FadeTimeSec` を `consumeAtomic()` で直読 (`CrossfadeAuthority.cpp`) | ✅ |
| P2 | HealthState直読も存在（Phase-2でOrchestratorに移動） | `engine.getHealthStateRef()` 呼び出し (`CrossfadeAuthority.cpp:13-20`) → Orchestrator へ移動 | ✅ |
| P2.5 | DSPTransition にCriticalチェック存在 | `DSPTransition.h:54-59` で確認 | ✅ |
| P1.5 | 6000番台のevent code未使用 | 既存event codeは1001-5002。6000-6003は空き | ✅ |
| P2.5 | `CROSSFADE_ABORTED_EMERGENCY` 未定義 | ソースコード内に存在せず（新規定義必要） | ✅ |
| P1.5 | `emitValidationEvent()` 未実装 | ソースコード内に存在せず（新規実装必要） | ✅ |
| P1.5 | `m_lastValidationEventUs_` 未実装 | ソースコード内に存在せず（新規実装必要） | ✅ |
| P2 | `makeCrossfadePolicy()` 未実装 | ソースコード内に存在せず（新規実装必要） | ✅ |
| B-1 | `setUseDryAsOld()` / `setFirstIrDryPending()` はDead Code | **grep/Serena/AiDex全ツールで呼び出し元ゼロ確認** | ✅ |
| B-2 | `kRuntimeSemanticSchemaVersion` = 8u | `ISRRuntimeSemanticSchema.h:9` で確認。9箇所の参照 | ✅ |
| C-1 | 既存テストのコンパイルエラー存在 | 3箇所の非存在フィールド参照を確認（Phase-0aで修正） | 🔴 新発見 → 修正 |
| General | `crossfadeAllowed` が未存在 | ソースコード全体で `crossfadeAllowed` の参照なし（計画修正で追加済み） | ✅ |
| General | `armCrossfadeIfPending` のTOCTOU | `prepared.pending` のみ参照→ `crossfadeRuntime_.isPending()` 追加で修正可能 | ✅ |
| General | `publishIdleWorldOnly()` 既存 | `AudioEngine.Transition.cpp:10` 定義、`DSPTransition.h:132` + `Timer.cpp:641` で使用 | ✅ |

### C-2: 新たに発見された問題

以下の問題は `original_plan.md` / `refined_plan.md` のいずれでも記載されていなかったが、実コード調査で発見された。

#### 🔴 Issue #1: 既存テストがコンパイル不可（CRITICAL）

**ファイル**: `src/tests/PublicationValidatorIsolationTests.cpp`

テストが以下の非存在フィールドを参照しており、**現行のstruct定義ではコンパイルできない**:

| テスト内コード | 問題 | 実際のstruct |
|---|---|---|
| `world.routing.numSources = 2` | ❌ 存在しない | `RoutingSemantic` は `processingOrder`, `eqBypassed`, `convBypassed` のみ |
| `world.routing.numDestinations = 2` | ❌ 存在しない | 同上 |
| `world.resource.memoryBudgetBytes = 1024 * 1024` | ❌ 存在しない | `ResourceSemantic` は `oversamplingFactor`, `ditherBitDepth`, `noiseShaperType` のみ |

**影響**: Phase-3（テスト追加）時、これらのテストはコンパイルエラーとなる。
**対処**: Phase-3 でこれらのテストを修正または削除する。Validator実体化（Phase-1）後は、真のテストケース（`checkNoConflictingTransitions` の整合性検証等）に置き換える。
**優先度**: Phase-3 実施前に必ず対応すること。

#### 🟡 Issue #2: Validatorのnamespace矛盾

**ファイル**: `RuntimePublicationValidator.h / .cpp`

- Validator は `iso::audio_engine` 名前空間に属している
- 計画コードおよび `ISRRuntimeSemanticSchema.h` のstructは `convo::isr` 名前空間
- `RuntimePublishWorld` = `::RuntimeState`（global scope）のため、フィールドアクセスに名前空間の不一致は影響しない
- しかし計画コードが `convo::isr::RuntimePublicationValidator` のように参照する場合、名前空間が合わない

**影響**: なし（静的変数 `iso::audio_engine::RuntimePublicationValidator validator;` として使用されているため）
**対処**: 計画コードは `iso::audio_engine::RuntimePublicationValidator` を使用するか、Validatorを `convo::isr` 名前空間に移動する。後者は破壊的変更のため非推奨。

#### 🟢 Issue #3: Validatorの `checkExecutionSemanticValidity()` は既に実装済み

**ファイル**: `RuntimePublicationValidator.cpp:96-112`

唯一実装のあるメソッド:

```cpp
if (exec.crossfadeStartDelayBlocks < 0) return false;
if (exec.crossfadeDryHoldSamples < 0) return false;
```

`isValidExecutionSemantic()`（`ISRRuntimeSemanticSchema.h`）と重複しているが、Validator内で早期検出するため問題なし。

### C-3: 各Phaseの実装詳細確認

#### Phase-1: Validator実体化 - フィールドアクセス完全性確認

| 計画コード | 実フィールド | パス |
|---|---|---|
| `world.generation` | `RuntimeState::generation` | `RuntimePublishWorld.generation` ✅ |
| `world.topology.runtimeUuid` | `TopologySemantic::runtimeUuid` | `RuntimeState::topology.runtimeUuid` ✅ |
| `world.topology.hasFadingRuntime` | `TopologySemantic::hasFadingRuntime` | ✅ |
| `world.topology.fadingRuntimeUuid` | `TopologySemantic::fadingRuntimeUuid` | ✅ |
| `world.execution.transitionActive` | `ExecutionSemantic::transitionActive` | ✅ |
| `world.execution.transitionPolicy` | `ExecutionSemantic::transitionPolicy` | ✅ |
| `world.resource.oversamplingFactor` | `ResourceSemantic::oversamplingFactor` | ✅ |
| `world.resource.ditherBitDepth` | `ResourceSemantic::ditherBitDepth` | ✅ |
| `world.resource.noiseShaperType` | `ResourceSemantic::noiseShaperType` | ✅ |
| `world.overlap.fadeTimeSec` | `OverlapSemantic::fadeTimeSec` | ✅ |
| `world.overlap.useDryAsOld` | `OverlapSemantic::useDryAsOld` | ✅ |

#### Phase-2: CrossfadePolicy抽出 - 現状の参照構造

```
CrossfadeAuthority::evaluate()
  ├── engine.getHealthStateRef()        → HealthState (Critical check) ← Orchestrator に移動（cfDecision上書き）
  ├── engine.m_irFadeTimeSec            → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_phaseFadeTimeSec         → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_tailFadeTimeSec          → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_osFadeTimeSec            → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_irLengthFadeTimeSec      → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_directHeadFadeTimeSec    → atomic<double>               ← CrossfadePolicy に移動
  ├── engine.m_nucFilterFadeTimeSec     → atomic<double>               ← CrossfadePolicy に移動
  ├── oldWorld.dspProjection.*          → RuntimeWorld read-only        ← 維持
  └── newWorld.dspProjection.*          → RuntimeWorld read-only        ← 維持
```

#### Phase-2: Emergency Override - Critical検出の2段階防御

```
Layer 1: RuntimePublicationOrchestrator (Admission) ← 直接 HealthState チェック
         （cfDecision.needsCrossfade = false で強制抑制）
Layer 2: DSPTransition (Execution)                 ← onPublishCompleted 内の
         getHealthStateRef() チェック（TOCTOU 最終安全網）
```

両層とも HealthState::Critical をチェックするが:

- Layer 1 は Orchestrator 内（evaluate 返却後、cfDecision 書き換え時点）
- Layer 2 は DSPTransition 実行時点の最新値（TOCTOU 対策）
- 両方存在することで、「Orchestrator通過〜DSPTransition実行」の間に Critical に遷移したケースを捕捉可能

### C-4: イベントコード割り当て

| コード | 定数名 | 定義元 |
|---|---|---|
| 1001-1011 | Retire関連（Stall/Warning/Age） | `RuntimeHealthMonitor.h` |
| 2001-2002 | Publication関連（Stall/Warning） | 同上 |
| 3001-3010 | Reader関連（Stuck/SlotUsage） | 同上 |
| 4001 | EVENT_CROSSFADE_TIMEOUT | 同上 |
| 4002 | EVENT_CROSSFADE_EVENT_DROP | 同上 |
| **4003** | **EVENT_CROSSFADE_ABORTED_EMERGENCY** | **Phase-2.5 で新規追加（5003→4003に修正、クロスフェード系統一）** |
| 5001-5002 | Learner Backpressure | 同上 |
| **6000** | **EVENT_VALIDATION_SEMANTIC_FAILURE** | **Phase-1.5 で新規追加** |
| **6001** | **EVENT_VALIDATION_TOPOLOGY_FAILURE** | **Phase-1.5 で新規追加** |
| **6002** | **EVENT_VALIDATION_RESOURCE_FAILURE** | **Phase-1.5 で新規追加** |
| **6003** | **EVENT_VALIDATION_TRANSITION_FAILURE** | **Phase-1.5 で新規追加** |

### C-5: 追加調査結果（2026-06-18 第2ラウンド）

以下の追加調査を実施し、新たな発見はなかったが、計画の完全性を再確認した。

#### テストフレームワーク構成

| テストファイル | フレームワーク | CMakeビルド対象 |
|---|---|---|
| `PublicationValidatorIsolationTests.cpp` | **gtest** | ❌ **未登録**（通常ビルドではコンパイルされない） |
| `CrossfadeExecutorLocalContractTests.cpp` | 独自（テキストスキャン） | ✅ |
| `OverlapAuthoritySingularTests.cpp` | 独自（テキストスキャン） | ✅ |
| その他11ファイル | 独自（テキストスキャン） | ✅ |

**発見**: `PublicationValidatorIsolationTests.cpp` は gtest を使用しているが、`CMakeLists.txt` に登録されていない。これは本テストが以前の開発サイクルで作成された後、CMake 統合が完了していないことを示す。Phase-3 で CMake 登録も併せて行うこと。

#### `useDryAsOld` 参照の完全トレース

`RuntimeBuilder.cpp` における `useDryAsOld` の設定箇所:

- **L112**: Bootstrap生成 → `worldOwner->overlap.useDryAsOld = false` ✅ 正しい
- **L288**: 通常生成 → `worldOwner->overlap.useDryAsOld = active` ❌ **Phase-0bで修正**
- **L389**: セマンティックハッシュ → 読み取りのみ（修正不要）

全ソース中で `useDryAsOld` を world に**書き込む**のは上記3箇所のみ。Phase-0b のスコープは L288 の1行で正確。

#### `setUseDryAsOld()` / `setFirstIrDryPending()` デッドコード確認

`CrossfadeRuntime.h` で宣言されているが:

- `setUseDryAsOld()` → **全ソースで呼び出し元ゼロ**（grep/Serena/PowerShell全ツールで確認）
- `setFirstIrDryPending()` → **全ソースで呼び出し元ゼロ**

これらは `CrossfadeRuntime` の atomic セッターとして残存しているが、現在のコードパスでは使用されていない。

#### `convo::compareExchangeAtomic()` の存在確認

`AtomicAccess.h:76` で定義済み。テンプレートシグネチャ:

```cpp
template <typename T>
inline bool compareExchangeAtomic(std::atomic<T>& dst, T& expected, T desired,
    std::memory_order success = std::memory_order_acq_rel,
    std::memory_order failure = std::memory_order_acquire) noexcept
```

Plan Phase-1.5 の `emitValidationEvent()` CASパターンはこのシグネチャと互換性がある。

### C-6: ツール別調査統計

| ツール | 使用回数 | 主な用途 |
|---|---|---|
| grep_search | 30+回 | キーワード横断（useDryAsOld全参照、fadeTime直読、HealthState参照 等） |
| read_file | 35+回 | 実ファイル全行読取（主要30+ファイル、2ラウンド） |
| AiDex MCP (status) | 2回 | プロジェクト統計確認（278 files, 4180 methods） |
| graphify MCP (god_nodes/stats) | 4回 | アーキテクチャ把握（17058 nodes, 22445 edges） |
| Serena MCP (search/overview) | 12+回 | パターン検索、シンボル概要把握 |
| CodeGraph MCP (analyze/get_structure) | 3回 | モジュール構造分析 |
| run_in_terminal (PowerShell) | 25+回 | 正規表現横断検索（非除外ファイルにも対応） |
| CMakeLists.txt 読取 | 1回 | テストフレームワーク構成確認 |

---

## Appendix D: 本計画範囲外の残課題

以下の課題は本計画のスコープ外とした。Practical Stable ISR Bridge Runtime の完成率 97〜98% には影響しない。

### D-1: CrossfadeAuthorityRuntime スレッド安全性

**現状**: `CrossfadeAuthorityRuntime` は `std::vector<CrossfadeRecord> records_` を生の `std::vector` で管理。Timer(MessageThread) 単一スレッドからのみ操作されるためデータ競合は存在しない。

**除外理由**: 実害のないリスクに対する過剰設計。Phase-5 の固定長CAS化は複雑性を増すだけで Practical Stable に貢献しない。

### D-2: HealthMonitor イベント体系統合

**現状**: `emitOnTransition()` と直接 `m_callback()` の2系統が混在。各系統の動作自体は正しい。

**除外理由**: 統合の利益がリスク（regression）を上回らない。`emitValidationEvent()` は独自の CAS ロジックを持ち、既存の `emitOnTransition` に無理に統合すると状態管理のバグ（`dummy == newState → 発火しない`）を生む。

### D-3: CrossfadeSettings 一括atomic化（Torn Read）

**現状**: 7個の `std::atomic<double>` は lock-free / wait-free だが、理論上は Torn Read の可能性がある。

**除外理由**: `std::atomic<CrossfadeSettings>`（56 bytes）は多くの環境で `libatomic` / `mutex` にフォールバックし、Audio Thread の RT-safe を破壊するリスクがある。現行の 7個の個別 `atomic<double>` は lock-free / wait-free を保証しており、Crossfade Decision の安全性は Torn Read によって壊れない。

### D-4: Validator namespace 統一

**現状**: Validator は `iso::audio_engine`、関連 Semantic 構造体は `convo::isr`。

**除外理由**: 影響範囲（include / forward declaration / using / test）に対して利益が小さく、費用対効果が低い。
