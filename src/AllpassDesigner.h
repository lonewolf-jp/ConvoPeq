#pragma once

#include <JuceHeader.h>
#include <vector>
#include <complex>
#include <functional>
#include "AlignedAllocation.h"
#include "CmaEsOptimizerDynamic.h"

namespace convo {

//==============================================================================
/**
    2次全通過セクション（極半径ρ、極角度θで表現）
    Phase 2 修正版：正しい群遅延公式と (ρ, θ) パラメータ化を採用
*/
struct SecondOrderAllpass {
    double rho = 0.0;      // 極半径 (0 ≤ ρ < 1)
    double theta = 0.0;    // 極角度 (ラジアン)

    // 安定性判定（ρ < 1 であれば常に安定）
    bool isStable() const noexcept { return (rho >= 0.0 && rho < 1.0); }

    // 複素周波数応答 H(z) = (z⁻² - 2ρ cosθ z⁻¹ + ρ²) / (1 - 2ρ cosθ z⁻¹ + ρ² z⁻²)
    std::complex<double> response(double omega) const {
        const std::complex<double> z = std::exp(std::complex<double>(0.0, -omega));
        const double rho2 = rho * rho;
        const double a1 = -2.0 * rho * std::cos(theta);
        const double a2 = rho2;
        const std::complex<double> num = a2 + a1 * z + z * z;
        const std::complex<double> den = 1.0 + a1 * z + a2 * z * z;
        const double denMag = std::abs(den);
        if (denMag < 1e-12)
            return std::complex<double>(1.0, 0.0);

        auto h = num / den;
        const double mag = std::abs(h);
        if (mag > 1e-12)
            h /= mag;
        else
            h = std::complex<double>(1.0, 0.0);

        return h;
    }

    // 解析的群遅延（サンプル単位）
    // Phase 2 修正：2つの1次セクションの和として正しく計算
    double groupDelay(double omega) const {
        if (rho <= 0.0) return 0.0;
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
};

//==============================================================================
/** 最適化手法の列挙 */
enum class OptimizationMethod { GreedyAdaGrad, CMAES };

/** 設計結果のステータス */
enum class DesignResult {
    Success,
    Cancelled,
    Failed
};

//==============================================================================
/** AllpassDesigner 設定構造体 */
struct AllpassDesignerConfig {
    int numSections = 8;               // 2次セクション数（16次）
    int freqPoints = 512;              // 周波数サンプリング点数
    double minFreqHz = 20.0;
    double maxFreqHz = 20000.0;
    int maxIterations = 50;            // AdaGrad の最大反復回数
    double learningRate = 0.01;        // AdaGrad の初期学習率

    // Phase 3 追加メンバ
    OptimizationMethod method = OptimizationMethod::GreedyAdaGrad;
    CmaEsOptimizerDynamic::Params cmaesParams;
    int cmaesMaxGenerations = 100;
    int cmaesPopulationSize = 32;          // 0 → 自動 (4 * dim)
    double cmaesInitialSigma = 0.3;
    std::function<void(float)> progressCallback;

    AllpassDesignerConfig() {
        cmaesParams.sigmaMin = 1e-6;
        cmaesParams.sigmaMax = 2.0;
        cmaesParams.covRetentionTarget = 0.98;
        cmaesParams.covRetentionStep = 0.002;
    }
};

//==============================================================================
/**
    AllpassDesigner: 全通過フィルタの設計を行うクラス
*/
class AllpassDesigner {
public:
    using Config = AllpassDesignerConfig;

    // 従来の設計メソッド（Greedy+AdaGrad）
    bool design(double sampleRate,
                const std::vector<double>& freq_hz,
                const std::vector<double>& target_group_delay_samples,
                const Config& config,
                std::vector<SecondOrderAllpass>& sections,
                const std::function<bool()>& shouldExit = nullptr,
                std::function<void(float)> progressCallback = nullptr);

    // CMA-ES による設計メソッド
    DesignResult designWithCMAES(double sampleRate,
                                 const std::vector<double>& freq_hz,
                                 const std::vector<double>& target_group_delay_samples,
                                 const Config& config,
                                 std::vector<SecondOrderAllpass>& sections,
                                 const std::function<bool()>& shouldExit = nullptr,
                                 std::function<void(float)> progressCallback = nullptr);

    // 設計済みセクションを IR に適用する関数
    static juce::AudioBuffer<double> applyAllpassToIR(
        const juce::AudioBuffer<double>& linearIR,
        const std::vector<SecondOrderAllpass>& sections,
        double sampleRate,
        const std::vector<double>& freq_hz,
        int fftSize,
        const std::function<bool()>& shouldExit = nullptr,
        std::function<void(float)> progressCallback = nullptr);

    // ユーティリティ：IR ファイルのハッシュ計算（キャッシュキー用）
    static uint64_t computeIRHash(const juce::File& irFile, bool useMD5 = false);

    // 静的ヘルパー：群遅延計算（(ρ, θ) 版）
    static double sectionGroupDelayRhoTheta(double rho, double theta, double omega, double sampleRate);

    // 静的ヘルパー：群遅延計算（(f0, gain) 版、後方互換）
    static double sectionGroupDelay(double f0, double gain, double omega, double sampleRate);

    // 設計された全通過フィルタの複素周波数応答を計算
    static std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>>
        computeResponse(const std::vector<SecondOrderAllpass>& sections,
                        double sampleRate,
                        const std::vector<double>& freq_hz);

private:
    // 従来の補助関数（Greedy+AdaGrad 用、現状維持）
    static bool gridSearch2D(const std::vector<double, convo::MKLAllocator<double>>& omega,
                             const std::vector<double, convo::MKLAllocator<double>>& residual,
                             double sampleRate,
                             double& best_f0, double& best_gain);
    static bool adaptiveGradientDescent(const std::vector<double, convo::MKLAllocator<double>>& omega,
                                        const std::vector<double, convo::MKLAllocator<double>>& residual,
                                        double sampleRate,
                                        double& f0, double& gain,
                                        double learningRate, int maxIterations);
};

} // namespace convo
