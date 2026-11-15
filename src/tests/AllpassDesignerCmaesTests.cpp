// AllpassDesignerCmaesTests.cpp
// B-2: CMA-ES の早期終了条件（sigma 下限到達）の回帰テスト
//
// 背景:
//   CmaEsOptimizerDynamic::update() は sigma を [sigmaMin, sigmaMax] にクランプする。
//   旧 AllpassDesigner::designWithCMAES() の早期終了は
//     sigma < min(1e-4, sigmaMin * 0.1)
//   で、既定構成（sigmaMin=1e-6, 初期 sigma=0.3）では閾値 1e-7 となり到達不能だった。
//   つまり早期終了が無効化され、常に cmaesMaxGenerations まで回っていた。
//
// このテストの目的:
//   1. 設計が成功し、安定なセクションが得られること（品質の下限）
//   2. 同一シードで完全に再現すること（決定性）
//   3. 早期終了を有効にした前後で結果が変わらないことを、同じ入力・同じシードで
//      比較できるよう、設計結果を高精度で出力すること
//      （[SIG] 行を旧実装と新実装で diff する）
//
// ビルド: カスタム main() + bool testXxx() パターン

#include "AllpassDesigner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using convo::AllpassDesigner;
using convo::AllpassDesignerConfig;
using convo::DesignResult;
using convo::SecondOrderAllpass;

constexpr double kSampleRate = 48000.0;
constexpr int    kFreqPoints = 512;
constexpr int    kSections   = 8;

int gFailures = 0;

void report(const char* id, bool ok, const char* detail)
{
    std::printf("[%s] %s %s\n", ok ? "PASS" : "FAIL", id, detail);
    if (! ok)
        ++gFailures;
}

// 20 Hz〜20 kHz の対数等間隔グリッドと、一定群遅延の目標
std::vector<double> makeFreqGrid()
{
    std::vector<double> f(static_cast<size_t>(kFreqPoints));
    const double logMin = std::log(20.0);
    const double logMax = std::log(20000.0);
    for (int i = 0; i < kFreqPoints; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(kFreqPoints - 1);
        f[static_cast<size_t>(i)] = std::exp(logMin + (logMax - logMin) * t);
    }
    return f;
}

std::vector<double> makeFlatTarget()
{
    return std::vector<double>(static_cast<size_t>(kFreqPoints), 10.0);
}

struct RunOutcome
{
    DesignResult result = DesignResult::Failed;
    std::vector<SecondOrderAllpass> sections;
    double maxGroupDelayError = 0.0;
    double elapsedMs = 0.0;
    int generations = 0;      // 実際に回った世代数（shouldExit の呼出し回数から算出）
};

RunOutcome runDesign(const AllpassDesignerConfig& config,
                     const std::vector<double>& freq,
                     const std::vector<double>& target)
{
    RunOutcome out;
    AllpassDesigner designer;

    // shouldExit は「関数入口で 1 回 + 各世代の先頭で 1 回」呼ばれる。
    // 呼出し回数 - 1 が実際に実行された世代数になり、早期終了が効いたかを観測できる。
    int exitCalls = 0;
    const auto shouldExit = [&exitCalls]() { ++exitCalls; return false; };

    const auto t0 = std::chrono::steady_clock::now();
    out.result = designer.designWithCMAES(kSampleRate, freq, target, config,
                                          out.sections, shouldExit, nullptr);
    const auto t1 = std::chrono::steady_clock::now();
    out.elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.generations = exitCalls > 0 ? exitCalls - 1 : 0;

    if (out.result != DesignResult::Success)
        return out;

    // 全セクション合計の群遅延と目標の差（サンプル単位）
    double maxErr = 0.0;
    for (size_t i = 0; i < freq.size(); ++i)
    {
        const double omega = 2.0 * 3.14159265358979323846 * freq[i] / kSampleRate;
        double total = 0.0;
        for (const auto& s : out.sections)
            total += AllpassDesigner::sectionGroupDelayRhoTheta(s.rho, s.theta, omega, kSampleRate);
        maxErr = std::max(maxErr, std::fabs(total - target[i]));
    }
    out.maxGroupDelayError = maxErr;
    return out;
}

bool sectionsIdentical(const std::vector<SecondOrderAllpass>& a,
                       const std::vector<SecondOrderAllpass>& b)
{
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].rho != b[i].rho || a[i].theta != b[i].theta)
            return false;
    }
    return true;
}

void printSignature(const char* cfg, const std::vector<SecondOrderAllpass>& sections)
{
    std::printf("[SIG] cfg=%s", cfg);
    for (const auto& s : sections)
        std::printf(" %.17g %.17g", s.rho, s.theta);
    std::printf("\n");
}

//==============================================================================
// 1 構成を実行し、成功・安定性・決定性を確認して結果を出力する
//==============================================================================
void checkConfig(const char* cfg, const AllpassDesignerConfig& config, double maxAcceptableGdError, bool expectEarlyExit)
{
    const auto freq = makeFreqGrid();
    const auto target = makeFlatTarget();

    const RunOutcome first = runDesign(config, freq, target);

    char detail[512];
    std::snprintf(detail, sizeof(detail),
                  "cfg=%s result=%d sections=%zu gen=%d/%d maxGdErr=%.6f samples (limit %.3f) elapsed=%.1f ms",
                  cfg, static_cast<int>(first.result), first.sections.size(),
                  first.generations, config.cmaesMaxGenerations,
                  first.maxGroupDelayError, maxAcceptableGdError, first.elapsedMs);

    const bool ok = first.result == DesignResult::Success
                    && first.sections.size() == static_cast<size_t>(config.numSections)
                    && std::isfinite(first.maxGroupDelayError)
                    && first.maxGroupDelayError < maxAcceptableGdError;
    report("B-2-design", ok, detail);

    // 早期終了が実際に効いているか（旧実装では sigma 条件が到達不能で常に最大世代だった）
    const bool genOk = expectEarlyExit
        ? (first.generations > 0 && first.generations < config.cmaesMaxGenerations)
        : (first.generations == config.cmaesMaxGenerations);
    report("B-2-generations", genOk, detail);

    if (first.result != DesignResult::Success)
        return;

    bool allStable = true;
    for (const auto& s : first.sections)
        allStable = allStable && s.isStable();
    report("B-2-stable", allStable, detail);

    // 決定性: 同一シード・同一入力で完全一致すること
    const RunOutcome second = runDesign(config, freq, target);
    const bool deterministic = second.result == DesignResult::Success
                               && sectionsIdentical(first.sections, second.sections);
    report("B-2-deterministic", deterministic, detail);

    printSignature(cfg, first.sections);
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    // 既定構成（sigmaMin=1e-6, 初期 sigma=0.3）
    {
        AllpassDesignerConfig config;
        config.method = convo::OptimizationMethod::CMAES;
        config.numSections = kSections;
        config.freqPoints = kFreqPoints;
        config.cmaesMaxGenerations = 100;
        config.cmaesPopulationSize = 32;
        checkConfig("default", config, 22.0, true);
    }

    // sigma 下限が高い構成（旧コメントが想定していた sigmaMin=0.05）
    //   修正前: 閾値 min(1e-4, 0.005) = 1e-4 < 0.05 で到達不能 → 常に 100 世代
    //   修正後: sigma が 0.05 に張り付いた時点で終了
    {
        AllpassDesignerConfig config;
        config.method = convo::OptimizationMethod::CMAES;
        config.numSections = kSections;
        config.freqPoints = kFreqPoints;
        config.cmaesMaxGenerations = 100;
        config.cmaesPopulationSize = 32;
        config.cmaesParams.sigmaMin = 0.05;
        checkConfig("sigmaFloor005", config, 16.0, false);
    }

    if (gFailures == 0)
    {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return EXIT_SUCCESS;
    }

    std::printf("\n=== SOME TESTS FAILED (failures=%d) ===\n", gFailures);
    return EXIT_FAILURE;
}
