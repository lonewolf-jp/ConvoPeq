// FFTBackendTests.cpp
// P1-1 Phase5: FFT Backend Concept + ProductionFft + TestFft テスト
//
// テスト内容:
//   1. FftStage enum の安全クランプ (ERRATA-V2023-2)
//   2. FftStatus 変換 (toFftStatus)
//   3. TestFft エラー注入 (正常系/異常系)
//   4. ProductionFft Plan 生成/破棄
//   5. FftBackendConcept 静的アサート
//   6. FFTExecutionContext nullptr ガード (PLAN-LT-9)
//
// ビルド: カスタム main() + bool testXxx() パターン

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "FFTBackend.h"
#include "FFTExecutionContext.h"

namespace {

using convo::FftStatus;
using convo::FftStage;
using convo::ProductionFft;
using convo::TestFft;
using convo::FFTExecutionContext;

//==============================================================================
// Test 1: FftStage enum safety clamp (ERRATA-V2023-2)
//==============================================================================
[[nodiscard]] bool testFftStageClamp()
{
    // Known valid stages must map correctly
    if (convo::toFftStage(1) != FftStage::IrForward) return false;
    if (convo::toFftStage(3) != FftStage::RuntimeForwardProcess) return false;
    if (convo::toFftStage(6) != FftStage::RuntimeInverseAdd) return false;
    if (convo::toFftStage(7) != FftStage::TailInverse) return false;

    // Diagnostic stage
    if (convo::toFftStage(99) != FftStage::Diagnostic) return false;

    // Unknown legacy stages must clamp to Diagnostic
    if (convo::toFftStage(0) != FftStage::Diagnostic) return false;
    if (convo::toFftStage(8) != FftStage::Diagnostic) return false;
    if (convo::toFftStage(-1) != FftStage::Diagnostic) return false;
    if (convo::toFftStage(100) != FftStage::Diagnostic) return false;

    // constexpr / noexcept compile check
    constexpr auto compiled = convo::toFftStage(1);
    (void)compiled;

    std::printf("[PASS] testFftStageClamp\n");
    return true;
}

//==============================================================================
// Test 2: FftStatus conversion from IppStatus
//==============================================================================
[[nodiscard]] bool testFftStatusConversion()
{
    if (convo::toFftStatus(ippStsNoErr) != FftStatus::Ok) return false;

    // Error statuses
    if (convo::toFftStatus(ippStsNullPtrErr) != FftStatus::InvalidArgument) return false;
    if (convo::toFftStatus(ippStsSizeErr) != FftStatus::InvalidArgument) return false;
    if (convo::toFftStatus(ippStsBadArgErr) != FftStatus::InvalidArgument) return false;
    if (convo::toFftStatus(ippStsMemAllocErr) != FftStatus::AllocationFailure) return false;

    // Unknown error → BackendError
    if (convo::toFftStatus(ippStsNoOperation) != FftStatus::BackendError) return false;

    // constexpr compile check
    constexpr auto compiled = convo::toFftStatus(ippStsNoErr);
    (void)compiled;

    std::printf("[PASS] testFftStatusConversion\n");
    return true;
}

//==============================================================================
// Test 3: TestFft error injection
//==============================================================================
[[nodiscard]] bool testTestFftErrorInjection()
{
    TestFft fft;
    TestFft::Plan plan = TestFft::createPlan(1024);

    // Normal: no error injected
    fft.setInjectError(false);
    if (fft.forwardRealToCCS(plan, nullptr, nullptr) != FftStatus::Ok) return false;
    if (fft.inverseCCSToR(plan, nullptr, nullptr) != FftStatus::Ok) return false;

    // Error injected
    fft.setInjectError(true);
    if (fft.forwardRealToCCS(plan, nullptr, nullptr) != FftStatus::BackendError) return false;
    if (fft.inverseCCSToR(plan, nullptr, nullptr) != FftStatus::BackendError) return false;

    // Plan isValid() must return true for TestFft
    if (!plan.isValid()) return false;

    std::printf("[PASS] testTestFftErrorInjection\n");
    return true;
}

//==============================================================================
// Test 4: FftBackendConcept static assertions
//==============================================================================
[[nodiscard]] bool testBackendConcept()
{
    // Compile-time checks (static_assert in header)
    // ProductionFft and TestFft must satisfy FftBackendConcept
    // If this compiles, the concept check passes.

    // Run-time: verify the concept check macro
    if constexpr (!convo::FftBackendConcept<ProductionFft>) return false;
    if constexpr (!convo::FftBackendConcept<TestFft>) return false;

    std::printf("[PASS] testBackendConcept\n");
    return true;
}

//==============================================================================
// Test 5: FFTExecutionContext nullptr guard (PLAN-LT-9)
//==============================================================================
[[nodiscard]] bool testExecutionContextNullGuard()
{
    // Default-constructed context has no Plan
    FFTExecutionContext ctx;

    // Must not crash — must return NotInitialized (PLAN-LT-9)
    if (ctx.processLayerFwd(nullptr, nullptr)
        != FftStatus::NotInitialized) return false;
    if (ctx.processLayerInv(nullptr, nullptr)
        != FftStatus::NotInitialized) return false;
    if (ctx.forwardRealToCCS(nullptr, nullptr) != FftStatus::NotInitialized) return false;
    if (ctx.inverseCCSToR(nullptr, nullptr) != FftStatus::NotInitialized) return false;

    // hasPlan() must return false
    if (ctx.hasPlan()) return false;
    if (ctx.isPlanValid()) return false;

    std::printf("[PASS] testExecutionContextNullGuard\n");
    return true;
}

//==============================================================================
// Test 6: FFTExecutionContext setPlan + rebind
//==============================================================================
[[nodiscard]] bool testExecutionContextSetPlan()
{
    // Use TestFft Plan for lightweight testing
    TestFft::Plan testPlan = TestFft::createPlan(512);
    // Plan for ProductionFft (null/invalid by default since no real Plan created)
    ProductionFft::Plan prodPlan{};  // invalid

    // Create context with reference — must accept
    FFTExecutionContext ctx(prodPlan);
    // plan_ is set to &prodPlan so hasPlan() returns true
    if (!ctx.hasPlan()) return false;
    // prodPlan default is invalid → isPlanValid() must be false
    if (ctx.isPlanValid()) return false;

    // setPlan with jassert guard — in test mode this would assert
    // For testing: create a context without plan, then set
    FFTExecutionContext emptyCtx;
    if (emptyCtx.hasPlan()) return false;

    // Note: setPlan() has jassert(plan_ == nullptr) which fires in Debug.
    // In Release build this test can run.
    // For now, verify that hasPlan/isPlanValid work correctly.

    std::printf("[PASS] testExecutionContextSetPlan\n");
    return true;
}

//==============================================================================
// Test 7: FFT-Stage contract: stable integer values
//==============================================================================
[[nodiscard]] bool testFftStageStableIntegers()
{
    // FFT-STAGE-1: FftStage must have stable integer values
    if (static_cast<int>(FftStage::Unknown) != 0) return false;
    if (static_cast<int>(FftStage::IrForward) != 1) return false;
    if (static_cast<int>(FftStage::IrInverse) != 2) return false;
    if (static_cast<int>(FftStage::RuntimeForwardProcess) != 3) return false;
    if (static_cast<int>(FftStage::RuntimeInverseProcess) != 4) return false;
    if (static_cast<int>(FftStage::RuntimeForwardAdd) != 5) return false;
    if (static_cast<int>(FftStage::RuntimeInverseAdd) != 6) return false;
    if (static_cast<int>(FftStage::TailInverse) != 7) return false;
    if (static_cast<int>(FftStage::Diagnostic) != 99) return false;

    std::printf("[PASS] testFftStageStableIntegers\n");
    return true;
}

//==============================================================================
// Test 8: ProductionFft output correctness (round trip + tone bin)
//   RES2-14: the suite only checked hasPlan/isPlanValid, so a broken transform
//   would pass. These checks compare against the input signal.
//==============================================================================
[[nodiscard]] bool testProductionFftOutputCorrectness()
{
    constexpr int N = 1024;
    constexpr int toneBin = 7;
    constexpr double pi = 3.14159265358979323846;

    ProductionFft::Plan plan = ProductionFft::createPlan(N);
    if (!plan.isValid())
    {
        std::printf("FAIL testProductionFftOutputCorrectness: createPlan(%d) returned an invalid plan\n", N);
        return false;
    }

    std::vector<double> input(static_cast<size_t>(N));
    std::vector<double> ccs(static_cast<size_t>(N) + 2);
    std::vector<double> output(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
    {
        input[static_cast<size_t>(i)] =
            0.25 + 0.5 * std::sin(2.0 * pi * static_cast<double>(toneBin) * static_cast<double>(i) / static_cast<double>(N));
    }

    ProductionFft fft;
    if (fft.forwardRealToCCS(plan, input.data(), ccs.data()) != FftStatus::Ok)
    {
        std::printf("FAIL testProductionFftOutputCorrectness: forwardRealToCCS failed\n");
        ProductionFft::destroyPlan(plan);
        return false;
    }
    if (fft.inverseCCSToR(plan, ccs.data(), output.data()) != FftStatus::Ok)
    {
        std::printf("FAIL testProductionFftOutputCorrectness: inverseCCSToR failed\n");
        ProductionFft::destroyPlan(plan);
        return false;
    }

    double maxRoundTripError = 0.0;
    for (int i = 0; i < N; ++i)
        maxRoundTripError = std::max(maxRoundTripError,
                                     std::fabs(output[static_cast<size_t>(i)] - input[static_cast<size_t>(i)]));

    // CCS packing: bin 0 (DC) at [0]; bin k >= 1 as (re, im) at [2k], [2k+1].
    const double dcBin = ccs[0];
    const double toneRe = ccs[2 * toneBin];
    const double toneIm = ccs[2 * toneBin + 1];
    const double toneMagnitude = std::sqrt(toneRe * toneRe + toneIm * toneIm);

    const double expectedDc = 0.25 * static_cast<double>(N);
    const double expectedTone = 0.5 * static_cast<double>(N) / 2.0;

    std::printf("[INFO] testProductionFftOutputCorrectness: roundTripMaxErr=%.3e dc=%.6f (exp %.6f) tone=%.6f (exp %.6f)\n",
                maxRoundTripError, dcBin, expectedDc, toneMagnitude, expectedTone);

    ProductionFft::destroyPlan(plan);

    if (maxRoundTripError > 1.0e-9)
    {
        std::printf("FAIL testProductionFftOutputCorrectness: round-trip error too large\n");
        return false;
    }
    if (std::fabs(dcBin - expectedDc) > 1.0e-6)
    {
        std::printf("FAIL testProductionFftOutputCorrectness: DC bin mismatch\n");
        return false;
    }
    if (std::fabs(toneMagnitude - expectedTone) > 1.0e-6)
    {
        std::printf("FAIL testProductionFftOutputCorrectness: tone bin magnitude mismatch\n");
        return false;
    }

    std::printf("[PASS] testProductionFftOutputCorrectness\n");
    return true;
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    bool allPassed = true;

    allPassed &= testFftStageClamp();
    allPassed &= testFftStatusConversion();
    allPassed &= testTestFftErrorInjection();
    allPassed &= testBackendConcept();
    allPassed &= testExecutionContextNullGuard();
    allPassed &= testExecutionContextSetPlan();
    allPassed &= testFftStageStableIntegers();
    allPassed &= testProductionFftOutputCorrectness();

    if (allPassed)
    {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return EXIT_SUCCESS;
    }
    else
    {
        std::printf("\n=== SOME TESTS FAILED ===\n");
        return EXIT_FAILURE;
    }
}
