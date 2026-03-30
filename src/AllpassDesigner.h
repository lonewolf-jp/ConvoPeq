#pragma once

#include <JuceHeader.h>
#include <vector>
#include <complex>
#include "AlignedAllocation.h"

namespace convo {

// 2次全通過セクションの係数
struct SecondOrderAllpass {
    double a1 = 0.0;  // -2ρ cosθ
    double a2 = 0.0;  // ρ²
    
    bool isStable() const noexcept { return std::abs(a2) < 1.0; }
    
    // 複素周波数応答
    std::complex<double> response(double omega) const {
        std::complex<double> z = std::exp(std::complex<double>(0.0, -omega));
        std::complex<double> num = a2 + a1 * z + z * z;
        std::complex<double> den = 1.0 + a1 * z + a2 * z * z;
        // ゼロ除算防止
        if (std::abs(den) < 1e-12) return std::complex<double>(1.0, 0.0);
        return num / den;
    }
    
    // 解析的群遅延（サンプル単位）
    double groupDelay(double omega) const {
        double cos_omega = std::cos(omega);
        double cos_2omega = std::cos(2.0 * omega);
        double denominator = 1.0 + a1 * cos_omega + a2 * cos_2omega;
        double numerator = 1.0 - a2 * a2;
        // デノーマル対策（ゼロ除算防止）
        if (std::abs(denominator) < 1e-12) return 0.0;
        return numerator / denominator;
    }
};

// 全通過フィルタ設計器
class AllpassDesigner {
public:
    struct Config {
        int numSections = 8;               // 2次セクション数（16次）
        int freqPoints = 512;              // 周波数サンプリング点数
        double minFreqHz = 20.0;
        double maxFreqHz = 20000.0;
        int maxIterations = 50;            // 勾配降下法の最大反復回数
        double learningRate = 0.01;        // 初期学習率
    };
    
    // 目標群遅延から全通過フィルタを設計
    bool design(double sampleRate,
                const std::vector<double>& freq_hz,
                const std::vector<double>& target_group_delay_samples,
                const Config& config,
                std::vector<SecondOrderAllpass>& sections);
    
    // 設計された全通過フィルタの複素周波数応答を計算
    static std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>> computeResponse(
        const std::vector<SecondOrderAllpass>& sections,
        double sampleRate,
        const std::vector<double>& freq_hz);
    
private:
    // 解析的な群遅延計算（パラメータから直接）
    static double sectionGroupDelay(double f0, double gain, double omega, double sampleRate);
    
    // グリッドサーチ（f0, gain の 2 次元、拡充版）
    static bool gridSearch2D(const std::vector<double, convo::MKLAllocator<double>>& omega,
                             const std::vector<double, convo::MKLAllocator<double>>& residual,
                             double sampleRate,
                             double& best_f0, double& best_gain);
    
    // 適応的勾配降下法（AdaGrad）
    static bool adaptiveGradientDescent(const std::vector<double, convo::MKLAllocator<double>>& omega,
                                        const std::vector<double, convo::MKLAllocator<double>>& residual,
                                        double sampleRate,
                                        double& f0, double& gain,
                                        double learningRate, int maxIterations);
};

} // namespace convo
