#include "AllpassDesigner.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace convo {

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
static inline double unconstrainedToRho(double x) {
    // 単調増加写像 R → (0, 0.98)。x=0 で ρ=0.49。
    // 旧実装は偶関数（rho(-x)==rho(x)）で二重縮退が生じていたため sigmoid に変更。
    return 0.98 / (1.0 + std::exp(-x));
}

static inline double unconstrainedToTheta(double x) {
    // 単調増加写像 R → (0, π)。x=0 で θ=π/2。
    // 群遅延式は cos(ω-θ)+cos(ω+θ) の対称性から θ ∈ (0,π) で充足。
    // 旧実装の ±π ハードクリップによるコスト関数不連続を解消。
    return juce::MathConstants<double>::pi / (1.0 + std::exp(-x));
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

            // θ = π * sigmoid(x) なので x = logit(θ/π)
            const double freqHz  = std::exp(logMin + (logMax - logMin) *
                                            (i + 0.5) / config.numSections);
            const double theta   = 2.0 * juce::MathConstants<double>::pi * freqHz / sampleRate;
            const double tNorm   = std::clamp(theta / juce::MathConstants<double>::pi, 1e-6, 1.0 - 1e-6);
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

        // 進捗コールバック
        if (progressCallback) {
            float progress = 0.2f + 0.6f * static_cast<float>(gen) / juce::jmax(1, config.cmaesMaxGenerations);
            progressCallback(progress);
        }

        // 早期終了条件：sigma が十分小さい、十分収束、または改善停滞
        double currentSigma = optimizer.getSigma();
        if (currentSigma < 1e-4) break;
        // 収束閾値: 重み付き RMSE 1.0サンプル（≈20μs @ 48kHz、5μs @ 192kHz）
        // 旧値 1e-3 は 0.001サンプル ≈ 5ns @ 192kHz で到達不能だったため修正
        if (bestFitness < 1.0) break;

        const double relImprovement = (prevBestFitness - bestFitness) / (prevBestFitness + 1e-12);
        const double absImprovement = prevBestFitness - bestFitness;
        if (gen > 20) {
            if (relImprovement < 1e-6 && absImprovement < 1e-2) {
                stagnationCounter++;
                if (stagnationCounter >= 15) break;
            } else {
                stagnationCounter = 0;
            }
        }
        prevBestFitness = bestFitness;
    }

    if (progressCallback)
        progressCallback(0.9f);

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
            progressCallback(0.5f + 0.25f * static_cast<float>(sec + 1) / config.numSections);
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
    const std::vector<double> f0_candidates = {
        20.0, 40.0, 80.0, 120.0, 200.0, 300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0,
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

        double err_f0_plus = 0.0, err_f0_minus = 0.0;
        double err_gain_plus = 0.0, err_gain_minus = 0.0;
        for (size_t i = 0; i < omega.size(); ++i) {
            err_f0_plus += std::pow(sectionGroupDelay(f0 + eps, gain, omega[i], sampleRate) - residual[i], 2);
            err_f0_minus += std::pow(sectionGroupDelay(f0 - eps, gain, omega[i], sampleRate) - residual[i], 2);
            err_gain_plus += std::pow(sectionGroupDelay(f0, gain + eps, omega[i], sampleRate) - residual[i], 2);
            err_gain_minus += std::pow(sectionGroupDelay(f0, gain - eps, omega[i], sampleRate) - residual[i], 2);
        }

        double grad_f0 = (err_f0_plus - err_f0_minus) / (2.0 * eps);
        double grad_gain = (err_gain_plus - err_gain_minus) / (2.0 * eps);
        grad_f0_norm += grad_f0 * grad_f0;
        grad_gain_norm += grad_gain * grad_gain;
        f0 -= learningRate * grad_f0 / (std::sqrt(grad_f0_norm) + 1e-8);
        gain -= learningRate * grad_gain / (std::sqrt(grad_gain_norm) + 1e-8);
        f0 = std::clamp(f0, 20.0, 20000.0);
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
// applyAllpassToIR  (Patch ①+⑤: MKL DFTI による正しい全通過変換実装)
//
// linearIR の各チャンネルを FFT し、設計済み全通過セクションの複素応答を
// Hermitian 対称性を保ちながら乗算したのち IFFT する。
// 振幅は必ず 1 に正規化 (resp /= mag) するため IR のレベルが保存される。
//==============================================================================
juce::AudioBuffer<double> AllpassDesigner::applyAllpassToIR(
    const juce::AudioBuffer<double>& linearIR,
    const std::vector<SecondOrderAllpass>& sections,
    double sampleRate,
    const std::vector<double>& /*freq_hz*/,
    int fftSize,
    const std::function<bool()>& shouldExit,
    std::function<void(float)> progressCallback)
{
    if (shouldExit && shouldExit()) return {};

    const int numChannels = linearIR.getNumChannels();
    const int irLen       = linearIR.getNumSamples();
    if (numChannels <= 0 || irLen <= 0 || sampleRate <= 0.0) return {};

    // FFT サイズを IR 長以上の次の 2 のべき乗に固定
    if (fftSize < irLen)
        fftSize = juce::nextPowerOfTwo(irLen);

    const int half        = fftSize / 2;
    const int complexSize = half + 1;

    // 1. 全通過フィルタの周波数応答を FFT ビン単位 (k = 0..N/2) で計算し
    //    振幅を 1 に強制正規化する (Patch ①核心)
    std::vector<std::complex<double>, convo::MKLAllocator<std::complex<double>>> allpassResp(complexSize);
    for (int k = 0; k < complexSize; ++k)
    {
        const double omega     = 2.0 * juce::MathConstants<double>::pi * k / fftSize;
        std::complex<double> resp(1.0, 0.0);
        for (const auto& sec : sections)
            resp *= sec.response(omega);
        const double mag = std::abs(resp);
        allpassResp[k] = (mag > 1e-12) ? (resp / mag) : std::complex<double>(1.0, 0.0);
    }
    // DC (k=0) と Nyquist (k=half) は実数のみ（位相は 0 か π のみ許容）
    allpassResp[0]    = std::complex<double>(1.0, 0.0);
    allpassResp[half] = std::complex<double>(
        std::real(allpassResp[half]) >= 0.0 ? 1.0 : -1.0, 0.0);

    // 2. MKL DFTI ディスクリプタ作成 (複素 in-place, BACKWARD_SCALE = 1/N)
    DFTI_DESCRIPTOR_HANDLE dfti = nullptr;
    const MKL_LONG len = static_cast<MKL_LONG>(fftSize);
    if (DftiCreateDescriptor(&dfti, DFTI_DOUBLE, DFTI_COMPLEX, 1, len) != DFTI_NO_ERROR)
        return {};
    if (DftiSetValue(dfti, DFTI_PLACEMENT, DFTI_INPLACE) != DFTI_NO_ERROR ||
        DftiSetValue(dfti, DFTI_BACKWARD_SCALE,
                     1.0 / static_cast<double>(fftSize)) != DFTI_NO_ERROR ||
        DftiCommitDescriptor(dfti) != DFTI_NO_ERROR)
    {
        DftiFreeDescriptor(&dfti);
        return {};
    }

    // 3. 作業バッファ: 複素数列 (fftSize 要素)
    convo::ScopedAlignedPtr<MKL_Complex16> spec(
        static_cast<MKL_Complex16*>(
            convo::aligned_malloc(static_cast<size_t>(fftSize) * sizeof(MKL_Complex16), 64)));
    if (!spec) { DftiFreeDescriptor(&dfti); return {}; }

    juce::AudioBuffer<double> result(numChannels, irLen);
    result.clear();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        if (shouldExit && shouldExit()) { DftiFreeDescriptor(&dfti); return {}; }

        // ゼロパディングしながら実部に IR をコピー
        std::memset(spec.get(), 0, static_cast<size_t>(fftSize) * sizeof(MKL_Complex16));
        const double* src = linearIR.getReadPointer(ch);
        for (int i = 0; i < irLen; ++i)
            spec.get()[i].real = src[i];

        // 順方向 FFT
        if (DftiComputeForward(dfti, spec.get()) != DFTI_NO_ERROR)
        { DftiFreeDescriptor(&dfti); return {}; }

        // 全通過応答を正の半スペクトル (k = 0..N/2) に乗算
        for (int k = 0; k <= half; ++k)
        {
            const std::complex<double> h(spec.get()[k].real, spec.get()[k].imag);
            const std::complex<double> out = h * allpassResp[k];
            spec.get()[k].real = out.real();
            spec.get()[k].imag = out.imag();
        }
        // Hermitian 対称性: 負の半スペクトル (k = N/2+1..N-1) を共役ミラーで更新
        for (int k = half + 1; k < fftSize; ++k)
        {
            const int mirror = fftSize - k;
            const std::complex<double> h(spec.get()[k].real, spec.get()[k].imag);
            const std::complex<double> out = h * std::conj(allpassResp[mirror]);
            spec.get()[k].real = out.real();
            spec.get()[k].imag = out.imag();
        }
        // DC と Nyquist の虚部を強制ゼロ（実 IR の Hermitian 条件）
        spec.get()[0].imag    = 0.0;
        spec.get()[half].imag = 0.0;

        // 逆方向 FFT (1/N スケーリング適用済み)
        if (DftiComputeBackward(dfti, spec.get()) != DFTI_NO_ERROR)
        { DftiFreeDescriptor(&dfti); return {}; }

        // 実部のみ irLen サンプル書き出し（デノーマル抑制付き）
        double* dst = result.getWritePointer(ch);
        for (int i = 0; i < irLen; ++i)
        {
            const double v = spec.get()[i].real;
            dst[i] = (std::abs(v) < 1.0e-18) ? 0.0 : v;
        }

        if (progressCallback)
            progressCallback(0.75f + 0.25f * static_cast<float>(ch + 1) / numChannels);
    }

    // Patch ⑤: 安全ガード - NaN/Inf チェック
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* p = result.getReadPointer(ch);
        for (int i = 0; i < irLen; ++i)
        {
            if (!std::isfinite(p[i]))
            {
                DBG("applyAllpassToIR: Safety guard triggered (NaN/Inf detected).");
                DftiFreeDescriptor(&dfti);
                return {};
            }
        }
    }

    double peak = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = result.getReadPointer(ch);
        for (int i = 0; i < irLen; ++i)
            peak = std::max(peak, std::abs(data[i]));
    }

    constexpr double kHeadroom = 0.708; // -3dB
    if (peak > kHeadroom)
    {
        const double gain = kHeadroom / peak;
        for (int ch = 0; ch < result.getNumChannels(); ++ch)
            result.applyGain(ch, 0, result.getNumSamples(), gain);
        DBG("applyAllpassToIR: peak reduced from " << peak << " to " << (peak * gain));
    }

    DftiFreeDescriptor(&dfti);
    return result;
}

//==============================================================================
// computeIRHash
//==============================================================================
uint64_t AllpassDesigner::computeIRHash(const juce::File& irFile, bool /*useMD5*/) {
    if (!irFile.existsAsFile()) return 0;

    uint64_t hash = static_cast<uint64_t>(irFile.getSize());
    hash ^= static_cast<uint64_t>(irFile.getLastModificationTime().toMilliseconds()) << 1;

    std::unique_ptr<juce::FileInputStream> stream(irFile.createInputStream());
    if (stream) {
        uint8_t buffer[1024];
        int bytesRead = stream->read(buffer, 1024);
        for (int i = 0; i < bytesRead; ++i) {
            hash = (hash * 31) + buffer[i];
        }
    }

    return hash;
}

} // namespace convo
