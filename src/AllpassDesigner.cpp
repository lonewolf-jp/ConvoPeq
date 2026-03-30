#include "AllpassDesigner.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace convo {

double AllpassDesigner::sectionGroupDelay(double f0, double gain, double omega, double sampleRate) {
    double omega0 = 2.0 * juce::MathConstants<double>::pi * f0 / sampleRate;
    double rho = gain;
    double rho2 = rho * rho;
    
    // 2次全通過セクションの群遅延公式
    // H(z) = (rho^2 - 2*rho*cos(omega0)*z^-1 + z^-2) / (1 - 2*rho*cos(omega0)*z^-1 + rho^2*z^-2)
    // a1 = -2*rho*cos(omega0), a2 = rho^2
    double a1 = -2.0 * rho * std::cos(omega0);
    double a2 = rho2;
    
    double cos_omega = std::cos(omega);
    double cos_2omega = std::cos(2.0 * omega);
    double denominator = 1.0 + a1 * cos_omega + a2 * cos_2omega;
    double numerator = 1.0 - a2 * a2;
    
    if (std::abs(denominator) < 1e-12) return 0.0;
    return numerator / denominator;
}

bool AllpassDesigner::gridSearch2D(const std::vector<double, convo::MKLAllocator<double>>& omega,
                                   const std::vector<double, convo::MKLAllocator<double>>& residual,
                                   double sampleRate,
                                   double& best_f0, double& best_gain) {
    // 拡充された候補点（低域・中域・高域をカバー）
    const std::vector<double> f0_candidates = {
        20.0, 40.0, 80.0, 120.0, 200.0,
        300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0,
        3000.0, 5000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0
    };
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
    const double eps = 1e-6;
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
        
        // 数値微分（中央差分）
        double err_f0_plus = 0.0, err_f0_minus = 0.0;
        double err_gain_plus = 0.0, err_gain_minus = 0.0;
        
        for (size_t i = 0; i < omega.size(); ++i) {
            double tau_f0_plus = sectionGroupDelay(f0 + eps, gain, omega[i], sampleRate);
            double diff_f0_plus = tau_f0_plus - residual[i];
            err_f0_plus += diff_f0_plus * diff_f0_plus;
            
            double tau_f0_minus = sectionGroupDelay(f0 - eps, gain, omega[i], sampleRate);
            double diff_f0_minus = tau_f0_minus - residual[i];
            err_f0_minus += diff_f0_minus * diff_f0_minus;
            
            double tau_gain_plus = sectionGroupDelay(f0, gain + eps, omega[i], sampleRate);
            double diff_gain_plus = tau_gain_plus - residual[i];
            err_gain_plus += diff_gain_plus * diff_gain_plus;
            
            double tau_gain_minus = sectionGroupDelay(f0, gain - eps, omega[i], sampleRate);
            double diff_gain_minus = tau_gain_minus - residual[i];
            err_gain_minus += diff_gain_minus * diff_gain_minus;
        }
        
        double grad_f0 = (err_f0_plus - err_f0_minus) / (2.0 * eps);
        double grad_gain = (err_gain_plus - err_gain_minus) / (2.0 * eps);
        
        // AdaGrad 更新
        grad_f0_norm += grad_f0 * grad_f0;
        grad_gain_norm += grad_gain * grad_gain;
        
        f0 -= learningRate * grad_f0 / (std::sqrt(grad_f0_norm) + 1e-8);
        gain -= learningRate * grad_gain / (std::sqrt(grad_gain_norm) + 1e-8);
        
        // クリップ
        f0 = std::clamp(f0, 20.0, 20000.0);
        gain = std::clamp(gain, 0.0, 0.99);
    }
    return true;
}

bool AllpassDesigner::design(double sampleRate,
                             const std::vector<double>& freq_hz,
                             const std::vector<double>& target_group_delay_samples,
                             const Config& config,
                             std::vector<SecondOrderAllpass>& sections) {
    // 角周波数に変換
    std::vector<double, convo::MKLAllocator<double>> omega(freq_hz.size());
    for (size_t i = 0; i < freq_hz.size(); ++i)
        omega[i] = 2.0 * juce::MathConstants<double>::pi * freq_hz[i] / sampleRate;
    
    std::vector<double, convo::MKLAllocator<double>> residual(target_group_delay_samples.begin(), target_group_delay_samples.end());
    sections.clear();
    sections.reserve(config.numSections);
    
    for (int sec = 0; sec < config.numSections; ++sec) {
        double best_f0 = 1000.0, best_gain = 0.5;
        if (!gridSearch2D(omega, residual, sampleRate, best_f0, best_gain))
            return false;
        
        adaptiveGradientDescent(omega, residual, sampleRate, best_f0, best_gain,
                                config.learningRate, config.maxIterations);
        
        // 係数計算
        double omega0 = 2.0 * juce::MathConstants<double>::pi * best_f0 / sampleRate;
        double a2 = best_gain * best_gain;
        double a1 = -2.0 * best_gain * std::cos(omega0);
        
        SecondOrderAllpass section;
        section.a1 = a1;
        section.a2 = a2;
        if (!section.isStable()) return false;
        sections.push_back(section);
        
        // 残差更新（このセクションの群遅延を減算）
        for (size_t i = 0; i < omega.size(); ++i) {
            residual[i] -= sectionGroupDelay(best_f0, best_gain, omega[i], sampleRate);
            if (residual[i] < 0.0) residual[i] = 0.0;
        }
    }
    return true;
}

std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>> AllpassDesigner::computeResponse(
    const std::vector<SecondOrderAllpass>& sections,
    double sampleRate,
    const std::vector<double>& freq_hz) {
    std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>> response(freq_hz.size(), std::complex<double>(1.0, 0.0));
    for (size_t i = 0; i < freq_hz.size(); ++i) {
        double omega = 2.0 * juce::MathConstants<double>::pi * freq_hz[i] / sampleRate;
        for (const auto& sec : sections) {
            response[i] *= sec.response(omega);
        }
    }
    return response;
}

} // namespace convo
