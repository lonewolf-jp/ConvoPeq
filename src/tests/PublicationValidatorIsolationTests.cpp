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
    world.generationSemantic.activationEpoch = 100;
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
    world.generationSemantic.activationEpoch = 100;
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
    world.generationSemantic.activationEpoch = 100;
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
    world.execution.transitionActive = false;
    
    // Act
    const bool isConsistent = validator_.validateSemanticConsistency(world);
    
    // Assert
    EXPECT_TRUE(isConsistent);
}

TEST_F(PublicationValidatorIsolationTests, ValidateTopology_BasicTopology_Success) {
    // Arrange: 基本的な topology
    RuntimePublishWorld world{};
    world.generation = 1;
    world.routing.numSources = 2;
    world.routing.numDestinations = 2;
    
    // Act
    const bool isValid = validator_.validateTopology(world);
    
    // Assert
    EXPECT_TRUE(isValid);
}

TEST_F(PublicationValidatorIsolationTests, ValidateResources_BasicResources_Success) {
    // Arrange: 基本的な resource
    RuntimePublishWorld world{};
    world.generation = 1;
    world.resource.memoryBudgetBytes = 1024 * 1024; // 1MB
    
    // Act
    const bool isValid = validator_.validateResources(world);
    
    // Assert
    EXPECT_TRUE(isValid);
}

TEST_F(PublicationValidatorIsolationTests, CheckNoConflictingTransitions_NoTransition_Success) {
    // Arrange: transition なし
    RuntimePublishWorld world{};
    world.generation = 1;
    world.execution.transitionActive = false;
    world.overlap.fadeTimeSec = 0.0;
    
    // Act
    const bool hasNoConflict = validator_.checkNoConflictingTransitions(world);
    
    // Assert
    EXPECT_TRUE(hasNoConflict);
}

} // namespace iso::audio_engine
