#include <gtest/gtest.h>
#include "RuntimePublicationValidator.h"
#include "ISRRuntimeSemanticSchema.h"
#include "RuntimeBuilder.h"

namespace iso::audio_engine {

/**
 * PublicationValidatorIsolationTests
 *
 * 検証対象：RuntimePublicationValidator の分離と純粋性
 *
 * Design Contract:
 * - Validator は AudioEngine に依存しない (pure validation logic)
 * - Validator は stateless (shared across threads)
 * - Validation は side-effect free
 */

class PublicationValidatorIsolationTests : public ::testing::Test {
protected:
    RuntimePublicationValidator validator_;
};

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_SemanticConsistency_Success) {
    // Arrange: 有効な semantic を持つ world を構築
    // Note: 実際のテストでは RuntimeBuilder を使用して world を生成するべき
    // ここでは簡略化のため、直接フィールドを設定

    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.generationSemantic.activationEpoch = 100;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    world.execution.transitionActive = false;
    world.execution.crossfadeStartDelayBlocks = 0;
    world.execution.crossfadeDryHoldSamples = 0;

    // Act
    const auto result = validator_.validatePublication(world);

    // Assert
    EXPECT_TRUE(result.isValid);
    EXPECT_TRUE(result.errorMessage.empty());
}

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_InvalidExecutionSemantic_Reject) {
    // Arrange: 無効な execution semantic (negative delay)
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.generationSemantic.activationEpoch = 100;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    world.execution.transitionActive = false;
    world.execution.crossfadeStartDelayBlocks = -1; // invalid
    world.execution.crossfadeDryHoldSamples = 0;

    // Act
    const auto result = validator_.validatePublication(world);

    // Assert
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.errorMessage, "Semantic consistency check failed");
}

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_NegativeDryHoldSamples_Reject) {
    // Arrange: 無効な execution semantic (negative dry hold samples)
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.generationSemantic.activationEpoch = 100;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    world.execution.transitionActive = false;
    world.execution.crossfadeStartDelayBlocks = 0;
    world.execution.crossfadeDryHoldSamples = -100; // invalid

    // Act
    const auto result = validator_.validatePublication(world);

    // Assert
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.errorMessage, "Semantic consistency check failed");
}

TEST_F(PublicationValidatorIsolationTests, ValidateSemanticConsistency_ActivationEpochDerived_Success) {
    // Arrange: activationEpoch は GenerationSemantic のみが authority
    // TimingSemantic.activationEpoch は derived field なので比較不要
    RuntimePublishWorld world{};
    world.generation = 1;
    world.generationSemantic.activationEpoch = 200;
    world.publication.sequenceId = 1;
    world.execution.transitionActive = false;

    // Act
    const bool isConsistent = validator_.validateSemanticConsistency(world);

    // Assert
    EXPECT_TRUE(isConsistent);
}

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_BasicTopology_Success) {
    // Arrange: 基本的な topology（Placeholder 検証のため最小フィールドのみ）
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;

    // Act
    const bool isValid = validator_.validateTopology(world);

    // Assert
    EXPECT_TRUE(isValid);
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_BasicResources_Success) {
    // Arrange: 基本的な resource（Placeholder 検証のため最小フィールドのみ）
    RuntimePublishWorld world{};
    world.generation = 1;

    // Act
    const bool isValid = validator_.validateResources(world);

    // Assert
    EXPECT_TRUE(isValid);
}

TEST_F(PublicationValidatorIsolationTests, CheckNoConflictingTransitions_NoTransition_Success) {
    // Arrange: transition なし
    RuntimePublishWorld world{};
    world.execution.transitionActive = false;
    world.overlap.fadeTimeSec = 0.0;

    // Act
    const bool hasNoConflict = validator_.checkNoConflictingTransitions(world);

    // Assert
    EXPECT_TRUE(hasNoConflict);
}

// ============================================================================
// ★ Phase-3: Validator Reject Tests
// ============================================================================

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_NoRuntimeUuid_Accept) {
    // ★ v8.3: runtimeUuid=0 は Bootstrap/Shutdown として許容。
    //   generation>0 でも runtimeUuid=0 は有効。
    //   代わりに transitionActive または fadingRuntimeUuid との矛盾をチェックする。
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;
    EXPECT_TRUE(validator_.validateTopology(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_IdentityCollision_Reject) {
    // ★ v8.3: Identity Invariant — fadingRuntimeUuid == runtimeUuid は自己同一性違反
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.topology.fadingRuntimeUuid = 100;  // 自己同一性違反
    EXPECT_FALSE(validator_.validateTopology(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_FadingWithoutUuid_Reject) {
    // ★ v8.3: runtimeUuid=0 で fadingRuntimeUuid!=0 は矛盾
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;       // Bootstrap/Shutdown
    world.topology.fadingRuntimeUuid = 200;  // 矛盾
    EXPECT_FALSE(validator_.validateTopology(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_OversamplingNotPowerOfTwo_Reject) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 3;  // 2のべき乗ではない
    EXPECT_FALSE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_OversamplingOutOfRange_Reject) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 32;  // 16超
    EXPECT_FALSE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_DitherInvalid_Reject) {
    RuntimePublishWorld world{};
    world.resource.ditherBitDepth = 8;  // 0,16,24,32 以外
    EXPECT_FALSE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_Dither32_Accept) {
    // ★ P1-1: dither=32 は kAdaptiveBitDepthValues の正規値
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 32;
    world.resource.noiseShaperType = 0;
    EXPECT_TRUE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_Dither16_Accept) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 16;
    world.resource.noiseShaperType = 0;
    EXPECT_TRUE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_Dither24_Accept) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 24;
    world.resource.noiseShaperType = 0;
    EXPECT_TRUE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_NoiseShaperOutOfRange_Reject) {
    RuntimePublishWorld world{};
    world.resource.noiseShaperType = 99;  // 0-3 以外
    EXPECT_FALSE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_NoiseShaperFixed15Tap_Accept) {
    // ★ P1-2: NoiseShaperType::Fixed15Tap(3) は正規ノイズシェーパータイプ
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 0;
    world.resource.noiseShaperType = 3;
    EXPECT_TRUE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_NoiseShaperAdaptive_Accept) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 0;
    world.resource.noiseShaperType = 2;
    EXPECT_TRUE(validator_.validateResources(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_HardResetWithFade_Reject) {
    // HardReset + fadeTimeSec > 0 は reject
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::HardReset);
    world.overlap.fadeTimeSec = 0.5;
    EXPECT_FALSE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_SmoothOnlyNegativeFade_Reject) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::SmoothOnly);
    world.overlap.fadeTimeSec = -0.1;
    EXPECT_FALSE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_DryAsOldWithoutFlag_Reject) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::DryAsOld);
    world.overlap.useDryAsOld = false;  // DryAsOld なのに useDryAsOld=false は矛盾
    world.overlap.fadeTimeSec = 0.005;
    EXPECT_FALSE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_InactiveWithUseDryAsOld_Reject) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = false;
    world.overlap.useDryAsOld = true;  // !active で useDryAsOld=true は矛盾
    EXPECT_FALSE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_UnknownPolicy_Reject) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = 99;  // 未知の policy
    EXPECT_FALSE(validator_.checkNoConflictingTransitions(world));
}

// ============================================================================
// ★ Phase-3: Validator Accept Tests
// ============================================================================

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_Bootstrap_Accept) {
    // Bootstrap world (generation=1 が実際の値) は runtimeUuid=0 でも accept
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;
    EXPECT_TRUE(validator_.validateTopology(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_NoUuidWithTransition_Reject) {
    // runtimeUuid=0 で transitionActive=true は矛盾 → reject
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;
    world.execution.transitionActive = true;
    EXPECT_FALSE(validator_.validateTopology(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_HardResetNoFade_Accept) {
    // HardReset + transitionActive=true + fadeTimeSec=0.0 は accept
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::HardReset);
    world.overlap.fadeTimeSec = 0.0;
    world.overlap.useDryAsOld = false;
    EXPECT_TRUE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_DryAsOldValid_Accept) {
    // DryAsOld + useDryAsOld=true + fadeTimeSec>0 は accept
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::DryAsOld);
    world.overlap.useDryAsOld = true;
    world.overlap.fadeTimeSec = 0.005;
    EXPECT_TRUE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, CheckTransition_IdleWithFadeRemnant_Accept) {
    // Idle world (transitionActive=false, fadeTimeSec>0) — フェード完了直後の残余値として許容
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = false;
    world.overlap.fadeTimeSec = 0.1;
    world.overlap.useDryAsOld = false;
    EXPECT_TRUE(validator_.checkNoConflictingTransitions(world));
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_ValidOversampling_Accept) {
    RuntimePublishWorld world{};
    world.resource.oversamplingFactor = 4;
    world.resource.ditherBitDepth = 24;
    world.resource.noiseShaperType = 1;
    EXPECT_TRUE(validator_.validateResources(world));
}

// ============================================================================
// ★ Phase-3: ValidatePublication 統合テスト
// ============================================================================

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_RejectFromTopology) {
    // ★ v8.3: Identity Invariant — fadingRuntimeUuid == runtimeUuid は自己同一性違反 → topology reject
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.topology.fadingRuntimeUuid = 100;  // 自己同一性違反
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    const auto result = validator_.validatePublication(world);
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.failureReason, iso::audio_engine::ValidationFailureReason::InvalidTopology);
}

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_RejectFromTopology_NoUuidWithTransition) {
    // runtimeUuid=0 で transitionActive=true は矛盾 → topology reject
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 0;
    world.execution.transitionActive = true;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    const auto result = validator_.validatePublication(world);
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.failureReason, iso::audio_engine::ValidationFailureReason::InvalidTopology);
}

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_RejectFromResources) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    world.resource.oversamplingFactor = 7;  // 2のべき乗ではない → reject
    const auto result = validator_.validatePublication(world);
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.failureReason, iso::audio_engine::ValidationFailureReason::InvalidResources);
}

TEST_F(PublicationValidatorIsolationTests, ValidatePublication_RejectFromTransition) {
    RuntimePublishWorld world{};
    world.generation = 1;
    world.topology.runtimeUuid = 100;
    world.generationSemantic.runtimeGeneration = 1;
    world.publication.sequenceId = 1;
    world.execution.transitionActive = true;
    world.execution.transitionPolicy = static_cast<int>(convo::TransitionPolicy::HardReset);
    world.overlap.fadeTimeSec = 0.5;  // HardReset なのに fade>0 → reject
    const auto result = validator_.validatePublication(world);
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.failureReason, iso::audio_engine::ValidationFailureReason::InvalidTransition);
}

// ============================================================================
// ★ Phase-3: CrossfadeAuthority Regression Tests
// ============================================================================

namespace {

// テスト用ヘルパー — 標準的な World と Policy を生成
[[nodiscard]] RuntimePublishWorld makeStandardOldWorld() noexcept {
    RuntimePublishWorld w{};
    w.generation = 1;
    w.topology.runtimeUuid = 100;
    w.dspProjection.irLoaded = true;
    w.dspProjection.structuralHash = 0xABCD;
    w.dspProjection.oversamplingFactor = 4;
    return w;
}

[[nodiscard]] RuntimePublishWorld makeStandardNewWorld() noexcept {
    RuntimePublishWorld w{};
    w.generation = 2;
    w.topology.runtimeUuid = 200;
    w.dspProjection.irLoaded = true;
    w.dspProjection.structuralHash = 0xABCD;  // 同じ IR
    w.dspProjection.oversamplingFactor = 4;
    return w;
}

[[nodiscard]] RuntimePublishWorld makeWorldWithIR() noexcept {
    RuntimePublishWorld w{};
    w.generation = 1;
    w.topology.runtimeUuid = 100;
    w.dspProjection.irLoaded = true;
    w.dspProjection.structuralHash = 0x1111;
    w.dspProjection.oversamplingFactor = 4;
    return w;
}

[[nodiscard]] RuntimePublishWorld makeWorldWithDifferentIR() noexcept {
    RuntimePublishWorld w{};
    w.generation = 2;
    w.topology.runtimeUuid = 200;
    w.dspProjection.irLoaded = true;
    w.dspProjection.structuralHash = 0x2222;  // 異なる IR
    w.dspProjection.oversamplingFactor = 4;
    return w;
}

[[nodiscard]] convo::isr::CrossfadePolicy makeFastFadePolicy() noexcept {
    convo::isr::CrossfadePolicy p;
    p.irFadeTimeSec = 0.002;
    p.phaseFadeTimeSec = 0.002;
    p.tailFadeTimeSec = 0.002;
    p.osFadeTimeSec = 0.002;
    p.irLengthFadeTimeSec = 0.002;
    p.directHeadFadeTimeSec = 0.002;
    p.nucFilterFadeTimeSec = 0.002;
    return p;
}

[[nodiscard]] convo::isr::CrossfadePolicy makeSlowFadePolicy() noexcept {
    convo::isr::CrossfadePolicy p;
    p.irFadeTimeSec = 0.080;
    p.phaseFadeTimeSec = 0.060;
    p.tailFadeTimeSec = 0.030;
    p.osFadeTimeSec = 0.030;
    p.irLengthFadeTimeSec = 0.050;
    p.directHeadFadeTimeSec = 0.010;
    p.nucFilterFadeTimeSec = 0.030;
    return p;
}

} // namespace

TEST(CrossfadeAuthorityRegressionTest, DeterministicDecision) {
    auto oldW = makeStandardOldWorld();
    auto newW = makeStandardNewWorld();
    auto policy = makeFastFadePolicy();
    convo::isr::CrossfadeAuthority auth;
    auto d1 = auth.evaluate(oldW, newW, policy);
    auto d2 = auth.evaluate(oldW, newW, policy);
    EXPECT_EQ(d1.needsCrossfade, d2.needsCrossfade);
    EXPECT_DOUBLE_EQ(d1.fadeTimeSec, d2.fadeTimeSec);
}

TEST(CrossfadeAuthorityRegressionTest, PolicyChangeChangesDecision) {
    auto oldW = makeWorldWithIR();
    auto newW = makeWorldWithDifferentIR();
    auto fast = makeFastFadePolicy();
    auto slow = makeSlowFadePolicy();
    convo::isr::CrossfadeAuthority auth;
    auto dFast = auth.evaluate(oldW, newW, fast);
    auto dSlow = auth.evaluate(oldW, newW, slow);
    EXPECT_TRUE(dFast.needsCrossfade);
    EXPECT_TRUE(dSlow.needsCrossfade);
    EXPECT_LT(dFast.fadeTimeSec, dSlow.fadeTimeSec);
}

TEST(CrossfadeAuthorityRegressionTest, SameStructuralHashNoCrossfade) {
    auto oldW = makeStandardOldWorld();
    auto newW = makeStandardNewWorld();  // 同じ structuralHash
    auto policy = makeSlowFadePolicy();
    convo::isr::CrossfadeAuthority auth;
    auto d = auth.evaluate(oldW, newW, policy);
    EXPECT_FALSE(d.needsCrossfade);  // IR が同じなので crossfade 不要
    EXPECT_DOUBLE_EQ(d.fadeTimeSec, 0.0);
}

TEST(CrossfadeAuthorityRegressionTest, OversamplingChangeTriggersCrossfade) {
    auto oldW = makeWorldWithIR();
    auto newW = makeWorldWithIR();
    newW.dspProjection.oversamplingFactor = 2;  // Oversampling 変更
    auto policy = makeFastFadePolicy();
    convo::isr::CrossfadeAuthority auth;
    auto d = auth.evaluate(oldW, newW, policy);
    EXPECT_TRUE(d.needsCrossfade);
    EXPECT_GT(d.fadeTimeSec, 0.0);
}

} // namespace iso::audio_engine
