#include "AllpassDesigner.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>
#include <thread>
#include <chrono>
#include <random>

namespace convo {

namespace {

// デフォルトシードは実行時ランダム（std::random_device が利用不可の場合は時刻ベース）
inline uint64_t generateRandomSeed() noexcept
{
    try {
        std::random_device rd;
        // random_device のエントロピーと時刻を混合
        const auto now = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        return static_cast<uint64_t>(rd()) ^ (now << 11) ^ (now >> 17);
    } catch (...) {
        // random_device が例外を投げる環境では時刻のみ
        return static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
}

std::vector<double> buildFrequencyCandidates(double sampleRate)
{
    constexpr int kCandidateCount = 18;
    constexpr double kMinCandidateHz = 20.0;
    const double maxCandidateHz = std::max(kMinCandidateHz,
        std::min(0.45 * sampleRate, 0.499 * sampleRate));

    std::vector<double> candidates;
    candidates.reserve(kCandidateCount);

    if (maxCandidateHz <= kMinCandidateHz)
    {
        candidates.push_back(kMinCandidateHz);
        return candidates;
    }

    const double logMin = std::log(kMinCandidateHz);
    const double logMax = std::log(maxCandidateHz);
    for (int index = 0; index < kCandidateCount; ++index)
    {
        const double t = (kCandidateCount == 1)
            ? 0.0
            : static_cast<double>(index) / static_cast<double>(kCandidateCount - 1);
        candidates.push_back(std::exp(logMin + (logMax - logMin) * t));
    }

    return candidates;
}

double clampOptimizationFrequency(double sampleRate, double value) noexcept
{
    const double maxCandidateHz = std::max(20.0,
        std::min(0.45 * sampleRate, 0.499 * sampleRate));
    return std::clamp(value, 20.0, maxCandidateHz);
}

double makeRelativeFrequencyStep(double f0) noexcept
{
    return std::max(1.0e-3, std::abs(f0) * 1.0e-4);
}

double makeRelativeGainStep(double gain) noexcept
{
    return std::clamp(std::max(1.0e-6, std::abs(gain) * 1.0e-4), 1.0e-6, 5.0e-3);
}

inline double stableSigmoid01(double x) noexcept
{
    // |x| > 50 では exp がオーバーフロー/アンダーフローするためクランプ
    x = std::clamp(x, -50.0, 50.0);
    if (x >= 0.0)
    {
        const double expNegX = std::exp(-x);
        return 1.0 / (1.0 + expNegX);
    }

    const double expX = std::exp(x);
    return expX / (1.0 + expX);
}

inline uint64_t rotl64(uint64_t value, int count) noexcept
{
    return (value << count) | (value >> (64 - count));
}

inline uint64_t readLE64(const uint8_t* p) noexcept
{
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t readLE32(const uint8_t* p) noexcept
{
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t xxh64Round(uint64_t acc, uint64_t input) noexcept
{
    constexpr uint64_t kPrime1 = 11400714785074694791ull;
    constexpr uint64_t kPrime2 = 14029467366897019727ull;
    acc += input * kPrime2;
    acc = rotl64(acc, 31);
    acc *= kPrime1;
    return acc;
}

inline uint64_t xxh64MergeRound(uint64_t acc, uint64_t val) noexcept
{
    constexpr uint64_t kPrime1 = 11400714785074694791ull;
    constexpr uint64_t kPrime4 = 9650029242287828579ull;
    acc ^= xxh64Round(0, val);
    acc = acc * kPrime1 + kPrime4;
    return acc;
}

inline uint64_t xxh64Avalanche(uint64_t h) noexcept
{
    constexpr uint64_t kPrime2 = 14029467366897019727ull;
    constexpr uint64_t kPrime3 = 1609587929392839161ull;
    h ^= h >> 33;
    h *= kPrime2;
    h ^= h >> 29;
    h *= kPrime3;
    h ^= h >> 32;
    return h;
}

inline uint64_t xxh64Digest(const uint8_t* data, size_t len, uint64_t seed) noexcept
{
    constexpr uint64_t kPrime1 = 11400714785074694791ull;
    constexpr uint64_t kPrime2 = 14029467366897019727ull;
    constexpr uint64_t kPrime3 = 1609587929392839161ull;
    constexpr uint64_t kPrime4 = 9650029242287828579ull;
    constexpr uint64_t kPrime5 = 2870177450012600261ull;

    const uint8_t* p = data;
    const uint8_t* end = data + len;
    uint64_t h = 0;

    if (len >= 32)
    {
        uint64_t v1 = seed + kPrime1 + kPrime2;
        uint64_t v2 = seed + kPrime2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - kPrime1;

        const uint8_t* limit = end - 32;
        do
        {
            v1 = xxh64Round(v1, readLE64(p)); p += 8;
            v2 = xxh64Round(v2, readLE64(p)); p += 8;
            v3 = xxh64Round(v3, readLE64(p)); p += 8;
            v4 = xxh64Round(v4, readLE64(p)); p += 8;
        } while (p <= limit);

        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = xxh64MergeRound(h, v1);
        h = xxh64MergeRound(h, v2);
        h = xxh64MergeRound(h, v3);
        h = xxh64MergeRound(h, v4);
    }
    else
    {
        h = seed + kPrime5;
    }

    h += static_cast<uint64_t>(len);

    while ((p + 8) <= end)
    {
        const uint64_t k1 = xxh64Round(0, readLE64(p));
        h ^= k1;
        h = rotl64(h, 27) * kPrime1 + kPrime4;
        p += 8;
    }

    if ((p + 4) <= end)
    {
        h ^= static_cast<uint64_t>(readLE32(p)) * kPrime1;
        h = rotl64(h, 23) * kPrime2 + kPrime3;
        p += 4;
    }

    while (p < end)
    {
        h ^= static_cast<uint64_t>(*p) * kPrime5;
        h = rotl64(h, 11) * kPrime1;
        ++p;
    }

    return xxh64Avalanche(h);
}

} // namespace

//==============================================================================
// 静的ヘルパー：群遅延（(ρ, θ) 版）
//==============================================================================
double AllpassDesigner::sectionGroupDelayRhoTheta(double rho, double theta, double omega, double /*sampleRate*/) {
    const double rho2 = rho * rho;
    const double term_num = 1.0 - rho2;
    const double denom1 = 1.0 - 2.0 * rho * std::cos(omega - theta) + rho2;
    const double denom2 = 1.0 - 2.0 * rho * std::cos(omega + theta) + rho2;
    const double eps = 1e-12 * (1.0 + rho2);
    double tau = 0.0;
    if (denom1 > eps) tau += term_num / denom1;
    if (denom2 > eps) tau += term_num / denom2;
    return tau;
}

//==============================================================================
// 静的ヘルパー：群遅延（(f0, gain) 版、後方互換）
//==============================================================================
double AllpassDesigner::sectionGroupDelay(double f0, double gain, double omega, double sampleRate) {
    const double rho = std::clamp(std::abs(gain), 0.0, 0.995);
    const double theta = 2.0 * juce::MathConstants<double>::pi * f0 / sampleRate;
    return sectionGroupDelayRhoTheta(rho, theta, omega, sampleRate);
}

//==============================================================================
// 無制約変数 → 物理パラメータ変換
//==============================================================================
namespace {
inline double unconstrainedToRho(double x) {
    // 単調増加写像 R → (0, 0.98)。x=0 で ρ=0.49。
    // 旧実装は偶関数（rho(-x)==rho(x)）で二重縮退が生じていたため sigmoid に変更。
    return 0.98 * stableSigmoid01(x);
}

// θの最大値（Nyquist極配置回避）
constexpr double kThetaMax = 0.99 * juce::MathConstants<double>::pi;

inline double unconstrainedToTheta(double x) {
    // 単調増加写像 R → (0, kThetaMax)。x=0 で θ=0.495π。
    // 群遅延式は cos(ω-θ)+cos(ω+θ) の対称性から θ ∈ (0,π) で充足。
    // Nyquist 極配置を避けるため上限を π 未満に固定する。
    return kThetaMax * stableSigmoid01(x);
}
}

//==============================================================================
// designWithCMAES（修正版：cost関数は全セクション合計、パラメータ変換改善）
//==============================================================================
DesignResult AllpassDesigner::designWithCMAES(
    double sampleRate,
    const std::vector<double>& freq_hz,
    const std::vector<double>& target_group_delay_samples,
    const Config& config,
    std::vector<SecondOrderAllpass>& sections,
    const std::function<bool()>& shouldExit,
    std::function<void(float)> progressCallback)
{
    if (shouldExit && shouldExit()) return DesignResult::Cancelled;

    const int D = 2 * config.numSections;   // (x_rho, x_theta) のペア
    CmaEsOptimizerDynamic optimizer(D);
    optimizer.setParams(config.cmaesParams);
    optimizer.setSeed(config.cmaesSeed != 0 ? config.cmaesSeed : generateRandomSeed());
    if (config.cmaesInitialSigma > 0.0) {
        CmaEsOptimizerDynamic::Params p = config.cmaesParams;
        p.sigmaMin = std::min(p.sigmaMin, config.cmaesInitialSigma);
        p.sigmaMax = std::max(p.sigmaMax, config.cmaesInitialSigma);
        optimizer.setParams(p);
    }

    // 初期平均値（無制約空間）
    // ρ: sigmoid(0) = 0.49 を起点に全セクションで共通
    // θ: 20Hz〜20kHz を対数均等に各セクションへ分散配置
    //    （旧実装では全セクション θ=0 (DC集中) で探索効率が著しく低下していた）
    std::vector<double> initialMean(D, 0.0);
    {
        const double logMin = std::log(config.minFreqHz);
        const double logMax = std::log(config.maxFreqHz);
        for (int i = 0; i < config.numSections; ++i) {
            initialMean[2*i] = 0.0;  // unconstrainedToRho(0) = 0.49

            // θ = kThetaMax * sigmoid(x) なので x = logit(θ/kThetaMax)
            const double freqHz  = std::exp(logMin + (logMax - logMin) *
                                            (i + 0.5) / config.numSections);
            const double theta   = 2.0 * juce::MathConstants<double>::pi * freqHz / sampleRate;
            const double tNorm   = std::clamp(theta / kThetaMax, 1e-6, 1.0 - 1e-6);
            initialMean[2*i+1]   = std::log(tNorm / (1.0 - tNorm));  // logit
        }
    }
    optimizer.initFromParcor(initialMean.data());

    // cmaesInitialSigma を初期σに正しく反映
    // （旧実装では sigmaMin/Max の調整のみで initFromParcor の σ=0.12 に上書きされていた）
    if (config.cmaesInitialSigma > 0.0)
        optimizer.setSigma(config.cmaesInitialSigma);

    // 周波数重み（対数周波数均等に近い設計、低域重視しすぎない）
    std::vector<double, convo::MKLAllocator<double>> omega(freq_hz.size());
    std::vector<double, convo::MKLAllocator<double>> weight(freq_hz.size());
    std::vector<double, convo::MKLAllocator<double>> cosOmega(freq_hz.size());
    std::vector<double, convo::MKLAllocator<double>> sinOmega(freq_hz.size());
    double weightSum = 0.0;
    for (size_t i = 0; i < freq_hz.size(); ++i) {
        omega[i] = 2.0 * juce::MathConstants<double>::pi * freq_hz[i] / sampleRate;
        cosOmega[i] = std::cos(omega[i]);
        sinOmega[i] = std::sin(omega[i]);
        weight[i] = 1.0 / std::sqrt(freq_hz[i] + 1.0);
        if (freq_hz[i] >= 0.499 * sampleRate)
            weight[i] *= 0.1;
        weightSum += weight[i];
    }
    for (auto& w : weight) w /= weightSum;

    // 目的関数：全セクションの群遅延を合計してから誤差を計算
    auto costFunc = [&](const std::vector<double>& x) -> double {
        // 現在の候補から各セクションの (ρ, θ) を計算
        std::vector<double> rho_list(config.numSections);
        std::vector<double> theta_list(config.numSections);
        std::vector<double> cosTheta(config.numSections);
        std::vector<double> sinTheta(config.numSections);
        for (int s = 0; s < config.numSections; ++s) {
            rho_list[s]   = unconstrainedToRho(x[2*s]);
            theta_list[s] = unconstrainedToTheta(x[2*s+1]);
            cosTheta[s]   = std::cos(theta_list[s]);
            sinTheta[s]   = std::sin(theta_list[s]);
        }
        double weightedSquaredError = 0.0;
        for (size_t i = 0; i < freq_hz.size(); ++i) {
            double tau_sum = 0.0;
            const double cw = cosOmega[i];
            const double sw = sinOmega[i];
            for (int s = 0; s < config.numSections; ++s) {
                const double rho  = rho_list[s];
                const double rho2 = rho * rho;
                const double termNum  = 1.0 - rho2;
                const double cosMinus = cw * cosTheta[s] + sw * sinTheta[s];
                const double cosPlus  = cw * cosTheta[s] - sw * sinTheta[s];
                const double denom1   = 1.0 - 2.0 * rho * cosMinus + rho2;
                const double denom2   = 1.0 - 2.0 * rho * cosPlus  + rho2;
                const double eps      = 1e-12 * (1.0 + rho2);
                if (denom1 > eps) tau_sum += termNum / denom1;
                if (denom2 > eps) tau_sum += termNum / denom2;
            }
            const double diff = tau_sum - target_group_delay_samples[i];
            weightedSquaredError += weight[i] * diff * diff;
        }
        // weights が sum=1 に正規化済みなので weightedSquaredError は重み付き MSE そのもの
        // polePenalty（ρを小さくする方向へ働き最適化目標と相反）は除去
        // 振幅ペナルティ（response() が |H|=1 を保証するため常に≈0）は除去
        return std::sqrt(weightedSquaredError);
    };

    const int lambda = (config.cmaesPopulationSize > 0) ? config.cmaesPopulationSize : 4 * D;
    std::vector<std::vector<double>> population(lambda, std::vector<double>(D));
    std::vector<double> fitness(lambda);
    double bestFitness = std::numeric_limits<double>::max();
    std::vector<double> bestParams(D);
    double prevBestFitness = bestFitness;
    int stagnationCounter = 0;

    juce::Logger::writeToLog("CMA-ES optimization started with "
                             + juce::String(config.numSections)
                             + " sections, dim=" + juce::String(D)
                             + ", lambda=" + juce::String(lambda)
                             + ", maxGen=" + juce::String(config.cmaesMaxGenerations));

    for (int gen = 0; gen < config.cmaesMaxGenerations; ++gen) {
        if (shouldExit && shouldExit()) return DesignResult::Cancelled;

        optimizer.sample(population);
        for (int i = 0; i < lambda; ++i) {
            fitness[i] = costFunc(population[i]);
            if (fitness[i] < bestFitness) {
                bestFitness = fitness[i];
                bestParams = population[i];
            }
        }
        optimizer.update(population, fitness);

        // Non-RT worker thread側で協調的にCPUを譲り、再生スレッドへの干渉を抑える。
        if ((gen & 1) == 0)
            std::this_thread::yield();

        if ((gen % 10) == 0)
        {
            juce::Logger::writeToLog("CMA-ES gen " + juce::String(gen)
                                     + " bestFitness=" + juce::String(bestFitness)
                                     + " sigma=" + juce::String(optimizer.getSigma()));
        }

        // 進捗コールバック（Message Thread にマーシャリング）
        if (progressCallback) {
            float progress = 0.2f + 0.6f * static_cast<float>(gen) / juce::jmax(1, config.cmaesMaxGenerations);
            auto cb = progressCallback;
            if (auto* mm = juce::MessageManager::getInstanceWithoutCreating())
            {
                mm->callAsync([cb, progress]() {
                    cb(progress);
                });
            }
        }

        // 早期終了条件：sigma が十分小さい、十分収束、または改善停滞
        double currentSigma = optimizer.getSigma();
        // sigmaMin が 1e-4 を超える設定の場合（例: sigmaMin=0.05）、固定値 1e-4 では
        // 永遠に満たされない。設定された sigmaMin の 1/10 と 1e-4 の小さい方を閾値とする。
        if (currentSigma < std::min(1e-4, config.cmaesParams.sigmaMin * 0.1)) break;
        // 収束閾値: 重み付き RMSE 1.0サンプル（≈20μs @ 48kHz、5μs @ 192kHz）
        // 旧値 1e-3 は 0.001サンプル ≈ 5ns @ 192kHz で到達不能だったため修正
        if (bestFitness < 1.0) break;

        const double relImprovement = (prevBestFitness - bestFitness) / (prevBestFitness + 1e-12);
        const double absImprovement = prevBestFitness - bestFitness;
        if (gen > 20) {
            if (relImprovement < 1e-6 && absImprovement < 1e-2) {
                stagnationCounter++;
                if (stagnationCounter >= 6) break;
            } else {
                stagnationCounter = 0;
            }
        }
        prevBestFitness = bestFitness;
    }

    if (progressCallback)
    {
        auto cb = progressCallback;
        if (auto* mm = juce::MessageManager::getInstanceWithoutCreating())
        {
            mm->callAsync([cb]() {
                cb(0.9f);
            });
        }
    }

    juce::Logger::writeToLog("CMA-ES optimization finished. Best fitness="
                             + juce::String(bestFitness)
                             + ", sigma=" + juce::String(optimizer.getSigma()));

    sections.clear();
    if (!std::isfinite(bestFitness))
        return DesignResult::Failed;

    for (int s = 0; s < config.numSections; ++s) {
        SecondOrderAllpass section;
        section.rho = unconstrainedToRho(bestParams[2*s]);
        section.theta = unconstrainedToTheta(bestParams[2*s+1]);
        sections.push_back(section);
    }
    return DesignResult::Success;
}

//==============================================================================
// design（従来の Greedy+AdaGrad、分岐追加）
//==============================================================================
bool AllpassDesigner::design(double sampleRate,
                             const std::vector<double>& freq_hz,
                             const std::vector<double>& target_group_delay_samples,
                             const Config& config,
                             std::vector<SecondOrderAllpass>& sections,
                             const std::function<bool()>& shouldExit,
                             std::function<void(float)> progressCallback)
{
    if (config.method == OptimizationMethod::CMAES) {
        return designWithCMAES(sampleRate, freq_hz, target_group_delay_samples, config, sections, shouldExit, progressCallback) == DesignResult::Success;
    }

    if (shouldExit && shouldExit()) return false;

    // ========== 従来の Greedy+AdaGrad（内部で (ρ, θ) を使用） ==========
    std::vector<double, convo::MKLAllocator<double>> omega(freq_hz.size());
    for (size_t i = 0; i < freq_hz.size(); ++i)
        omega[i] = 2.0 * juce::MathConstants<double>::pi * freq_hz[i] / sampleRate;

    std::vector<double, convo::MKLAllocator<double>> residual(target_group_delay_samples.begin(),
                                                                target_group_delay_samples.end());
    sections.clear();
    sections.reserve(config.numSections);

    for (int sec = 0; sec < config.numSections; ++sec) {
        if (shouldExit && shouldExit()) return false;

        double best_f0 = 1000.0, best_gain = 0.5;
        if (!gridSearch2D(omega, residual, sampleRate, best_f0, best_gain))
            return false;
        adaptiveGradientDescent(omega, residual, sampleRate, best_f0, best_gain,
                                config.learningRate, config.maxIterations);

        SecondOrderAllpass section;
        section.rho = std::clamp(std::abs(best_gain), 0.0, 0.995);
        section.theta = 2.0 * juce::MathConstants<double>::pi * best_f0 / sampleRate;
        sections.push_back(section);

        // 残差更新（このセクションの群遅延を減算、負の群遅延はクリップしない）
        for (size_t i = 0; i < omega.size(); ++i) {
            residual[i] -= sectionGroupDelayRhoTheta(section.rho, section.theta, omega[i], sampleRate);
        }

        if (progressCallback) {
            float progress = 0.5f + 0.25f * static_cast<float>(sec + 1) / config.numSections;
            auto cb = progressCallback;
            if (auto* mm = juce::MessageManager::getInstanceWithoutCreating())
            {
                mm->callAsync([cb, progress]() {
                    cb(progress);
                });
            }
        }
    }
    return true;
}

//==============================================================================
// 従来の補助関数（実装維持、ただし内部で MKLAllocator を使用）
//==============================================================================
bool AllpassDesigner::gridSearch2D(const std::vector<double, convo::MKLAllocator<double>>& omega,
                                   const std::vector<double, convo::MKLAllocator<double>>& residual,
                                   double sampleRate,
                                   double& best_f0, double& best_gain) {
    const std::vector<double> f0_candidates = buildFrequencyCandidates(sampleRate);
    const std::vector<double> gain_candidates = { 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98 };

    double best_error = std::numeric_limits<double>::max();
    for (double f0 : f0_candidates) {
        for (double gain : gain_candidates) {
            double error = 0.0;
            for (size_t i = 0; i < omega.size(); ++i) {
                double tau = sectionGroupDelay(f0, gain, omega[i], sampleRate);
                double diff = tau - residual[i];
                error += diff * diff;
            }
            if (error < best_error) {
                best_error = error;
                best_f0 = f0;
                best_gain = gain;
            }
        }
    }
    return best_error < std::numeric_limits<double>::max();
}

bool AllpassDesigner::adaptiveGradientDescent(const std::vector<double, convo::MKLAllocator<double>>& omega,
                                              const std::vector<double, convo::MKLAllocator<double>>& residual,
                                              double sampleRate,
                                              double& f0, double& gain,
                                              double learningRate, int maxIterations) {
    double grad_f0_norm = 0.0, grad_gain_norm = 0.0;
    double prev_error = std::numeric_limits<double>::max();

    for (int iter = 0; iter < maxIterations; ++iter) {
        double error = 0.0;
        for (size_t i = 0; i < omega.size(); ++i) {
            double tau = sectionGroupDelay(f0, gain, omega[i], sampleRate);
            double diff = tau - residual[i];
            error += diff * diff;
        }
        if (error >= prev_error) break;
        prev_error = error;

        const double epsF0 = makeRelativeFrequencyStep(f0);
        const double epsGain = makeRelativeGainStep(gain);
        double err_f0_plus = 0.0, err_f0_minus = 0.0;
        double err_gain_plus = 0.0, err_gain_minus = 0.0;
        for (size_t i = 0; i < omega.size(); ++i) {
            err_f0_plus += std::pow(sectionGroupDelay(f0 + epsF0, gain, omega[i], sampleRate) - residual[i], 2);
            err_f0_minus += std::pow(sectionGroupDelay(f0 - epsF0, gain, omega[i], sampleRate) - residual[i], 2);
            err_gain_plus += std::pow(sectionGroupDelay(f0, gain + epsGain, omega[i], sampleRate) - residual[i], 2);
            err_gain_minus += std::pow(sectionGroupDelay(f0, gain - epsGain, omega[i], sampleRate) - residual[i], 2);
        }

        double grad_f0 = (err_f0_plus - err_f0_minus) / (2.0 * epsF0);
        double grad_gain = (err_gain_plus - err_gain_minus) / (2.0 * epsGain);
        grad_f0_norm += grad_f0 * grad_f0;
        grad_gain_norm += grad_gain * grad_gain;
        f0 -= learningRate * grad_f0 / (std::sqrt(grad_f0_norm) + 1e-8);
        gain -= learningRate * grad_gain / (std::sqrt(grad_gain_norm) + 1e-8);
        f0 = clampOptimizationFrequency(sampleRate, f0);
        gain = std::clamp(gain, 0.0, 0.995);
    }
    return true;
}

//==============================================================================
// computeResponse
//==============================================================================
std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>>
AllpassDesigner::computeResponse(const std::vector<SecondOrderAllpass>& sections,
                                 double sampleRate,
                                 const std::vector<double>& freq_hz)
{
    std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>> response(freq_hz.size(), 1.0);
    for (size_t i = 0; i < freq_hz.size(); ++i) {
        double omega = 2.0 * juce::MathConstants<double>::pi * freq_hz[i] / sampleRate;
        for (const auto& sec : sections) {
            response[i] *= sec.response(omega);
        }
    }
    return response;
}

//==============================================================================
// computeIRHash
//==============================================================================
uint64_t AllpassDesigner::computeIRHash(const juce::File& irFile, bool /*useMD5*/) {
    if (!irFile.existsAsFile())
        return 0;

    const auto sizeBefore = irFile.getSize();
    const auto mtimeBefore = irFile.getLastModificationTime().toMilliseconds();

    std::unique_ptr<juce::FileInputStream> stream(irFile.createInputStream());
    if (stream == nullptr)
        return 0;

    // xxHash64 でファイル全体を計算し、TOCTOU は before/after 検証で防止する。
    constexpr uint64_t kHashVersionSalt = 0x434f4e564f504551ull; // "CONVOPEQ"
    juce::HeapBlock<uint8_t> fileData;
    const size_t fileSize = static_cast<size_t>(juce::jmax<int64>(0, sizeBefore));
    if (fileSize > 0)
        fileData.malloc(fileSize);

    size_t writePos = 0;
    uint8_t tempBuffer[4096];
    for (;;)
    {
        const int bytesRead = stream->read(tempBuffer, static_cast<int>(sizeof(tempBuffer)));
        if (bytesRead <= 0)
            break;

        const size_t bytes = static_cast<size_t>(bytesRead);
        if (writePos + bytes > fileSize)
            return 0;
        if (bytes > 0)
            std::memcpy(fileData.getData() + writePos, tempBuffer, bytes);
        writePos += bytes;
    }

    if (stream->getStatus().failed())
        return 0;

    if (writePos != fileSize)
        return 0;

    const auto sizeAfter = irFile.getSize();
    const auto mtimeAfter = irFile.getLastModificationTime().toMilliseconds();
    if (sizeBefore != sizeAfter || mtimeBefore != mtimeAfter)
        return 0;

    const uint64_t hash = xxh64Digest(fileData.getData(), fileSize, kHashVersionSalt);
    return hash;
}

} // namespace convo
