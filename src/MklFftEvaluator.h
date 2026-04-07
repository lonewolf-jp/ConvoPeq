#pragma once

#include <JuceHeader.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

// [v2.1] MKL DFTI を Intel IPP に換装。
// mkl_malloc / mkl_free は引き続きオーディオデータバッファ用に使用。
#include <mkl.h>   // mkl_malloc, mkl_free
#include <ipp.h>  // ippsFFTFwd_RToCCS_64f, IppsFFTSpec_R_64f

class MklFftEvaluator
{
public:
    static constexpr int kFftLength  = 4096;
    static constexpr int kSpectrumBins = (kFftLength / 2) + 1;  // = 2049
    static constexpr double kDefaultSampleRateHz = 48000.0;
    static constexpr int kBarkBandCount = 24;

    // [v2.1] MKL_Complex16 の代替。標準レイアウト構造体。
    // メモリ配置: { double real; double im; } = 16バイト。
    // IPP CCS 出力 [re0,im0,re1,im1,...] と同一レイアウトのため
    // reinterpret_cast による相互変換が安全。
    // .real / .imag のメンバ名を維持することで、呼び出し側 (NoiseShaperLearner) の
    // コード変更を MKL_Complex16 → MklFftEvaluator::CcsComplex の型名変更のみに限定する。
    struct CcsComplex
    {
        double real = 0.0;
        double imag = 0.0;
    };
    static_assert(sizeof(CcsComplex) == 2 * sizeof(double),
                  "CcsComplex must be exactly 2 doubles (16 bytes)");

    struct Result
    {
        double noisePower = 0.0;
        double spectralFlatnessPenalty = 0.0;
        double hfPenalty = 0.0;
        double timeDomainRms = 0.0;
        double compositeScore = 0.0;
    };

    MklFftEvaluator()
    {
        // オーディオデータバッファ (mkl_malloc: 64バイトアライン)
        inputLeft  = static_cast<double*>(mkl_malloc(sizeof(double) * kFftLength, 64));
        inputRight = static_cast<double*>(mkl_malloc(sizeof(double) * kFftLength, 64));

        // [v2.1] スペクトラムバッファ: CcsComplex 配列として確保
        // kSpectrumBins 個 × 2 double = kFftLength+2 doubles (IPP CCS 出力サイズと一致)
        spectrumLeft  = static_cast<CcsComplex*>(
            mkl_malloc(sizeof(CcsComplex) * kSpectrumBins, 64));
        spectrumRight = static_cast<CcsComplex*>(
            mkl_malloc(sizeof(CcsComplex) * kSpectrumBins, 64));

        // [v2.1] IPP FFT スペック初期化
        // kFftLength = 4096 = 2^12 → order = 12
        // IPP_FFT_NODIV_BY_ANY: forward に正規化なし (evaluate は forward のみ使用)
        // ippAlgHintFast: 速度優先ヒント
        constexpr int kOrder = 12; // log2(4096)
        static_assert((1 << kOrder) == kFftLength, "kOrder must be log2(kFftLength)");

        int sizeSpec = 0, sizeInit = 0, sizeWork = 0;
        const IppStatus getSt = ippsFFTGetSize_R_64f(
            kOrder, IPP_FFT_NODIV_BY_ANY, ippAlgHintFast,
            &sizeSpec, &sizeInit, &sizeWork);

        if (getSt == ippStsNoErr)
        {
            fftSpecBuf = ippsMalloc_8u(sizeSpec);

            Ipp8u* initBuf = (sizeInit > 0) ? ippsMalloc_8u(sizeInit) : nullptr;

            // [Bug 2 fix] ippsFFTInit_R_64f の戻り値と fftSpec の null を明示チェック。
            // 失敗時は fftSpec を nullptr のまま残し、evaluate()/computeFft() 冒頭の
            // ガードで早期リターンさせる。
            if (fftSpecBuf)
            {
                const IppStatus initSt = ippsFFTInit_R_64f(
                    &fftSpec, kOrder, IPP_FFT_NODIV_BY_ANY, ippAlgHintFast,
                    fftSpecBuf, initBuf);

                if (initSt != ippStsNoErr || fftSpec == nullptr)
                {
                    // 初期化失敗 → fftSpec を安全に nullptr にリセット
                    // fftSpecBuf はデストラクタで解放
                    fftSpec = nullptr;
                    DBG("MklFftEvaluator: ippsFFTInit_R_64f failed (status="
                        + juce::String(static_cast<int>(initSt)) + ")");
                }
            }
            if (initBuf) { ippsFree(initBuf); }

            // [Bug 3 fix] fftWorkBuf 確保失敗時のガード。
            // sizeWork > 0 かつ ippsMalloc が nullptr を返した場合、
            // nullptr を IPP に渡すと未定義動作になるため、fftSpec を無効化して
            // evaluate()/computeFft() 冒頭のガードで早期リターンさせる。
            if (sizeWork > 0)
            {
                fftWorkBuf = ippsMalloc_8u(sizeWork);
                if (!fftWorkBuf)
                {
                    fftSpec = nullptr;  // FFT 不使用状態に落とす
                    DBG("MklFftEvaluator: ippsMalloc_8u(sizeWork="
                        + juce::String(sizeWork) + ") failed");
                }
            }
        }

        configureForSampleRate(kDefaultSampleRateHz);
    }

    ~MklFftEvaluator()
    {
        // [v2.1] IPP FFT リソース解放
        if (fftSpecBuf)
        {
            ippsFree(fftSpecBuf);
            fftSpecBuf = nullptr;
            fftSpec    = nullptr;
        }
        if (fftWorkBuf)
        {
            ippsFree(fftWorkBuf);
            fftWorkBuf = nullptr;
        }

        if (inputLeft   != nullptr) mkl_free(inputLeft);
        if (inputRight  != nullptr) mkl_free(inputRight);
        if (spectrumLeft  != nullptr) mkl_free(spectrumLeft);
        if (spectrumRight != nullptr) mkl_free(spectrumRight);
    }

    void configureForSampleRate(double sampleRateHz) noexcept
    {
        const double safeSampleRateHz = std::max(1.0, sampleRateHz);
        const double nyquistHz = 0.5 * safeSampleRateHz;
        const double binWidthHz = nyquistHz / static_cast<double>(kSpectrumBins - 1);

        auto hzToBin = [binWidthHz](double hz) noexcept
        {
            if (binWidthHz <= 0.0)
                return 0;
            return std::clamp(static_cast<int>(std::lround(hz / binWidthHz)), 0, kSpectrumBins - 1);
        };

        configuredSampleRateHz = safeSampleRateHz;
        configuredWeightSum = 0.0;
        flatnessPenaltyWeight = 0.35;
        hfPenaltyWeight = std::clamp(0.20 * std::sqrt(48000.0 / safeSampleRateHz), 0.05, 0.20);

        double flatnessStartHz = std::min(12000.0, nyquistHz * 0.60);
        double flatnessEndHz = std::min(18000.0, nyquistHz * 0.82);
        if (flatnessEndHz <= flatnessStartHz + (binWidthHz * 8.0))
        {
            flatnessStartHz = nyquistHz * 0.50;
            flatnessEndHz = nyquistHz * 0.80;
        }

        flatnessStartBin = hzToBin(flatnessStartHz);
        flatnessEndBin = std::max(flatnessStartBin + 1, hzToBin(flatnessEndHz));

        double highBandStartHz = std::max(14000.0, nyquistHz * 0.60);
        if (highBandStartHz >= nyquistHz)
            highBandStartHz = nyquistHz * 0.60;

        double ultraHighStartHz = nyquistHz * 0.85;
        if (ultraHighStartHz <= highBandStartHz + (binWidthHz * 8.0))
            ultraHighStartHz = highBandStartHz + (binWidthHz * 8.0);

        highBandStartBin = hzToBin(highBandStartHz);
        ultraHighStartBin = std::max(highBandStartBin + 1, hzToBin(ultraHighStartHz));

        const int highBandBins = std::max(1, kSpectrumBins - highBandStartBin);
        const int ultraHighBins = std::max(1, kSpectrumBins - ultraHighStartBin);
        expectedUltraHighShare = static_cast<double>(ultraHighBins) / static_cast<double>(highBandBins);

        auto bandWeightForHz = [nyquistHz](double f) noexcept
        {
            if (f < 1.0) f = 1.0;

            const double f2 = f * f;
            const double h1 = -4.737338981378384e-24 * f2 * f2 * f2 + 2.043828333606125e-15 * f2 * f2 - 1.363894795463638e-7 * f2 + 1.0;
            const double h2 = 1.306612257402824e-19 * f2 * f2 * f - 2.118150887541247e-11 * f2 * f + 5.559488023498642e-4 * f;
            const double r_f = (1.246332637532143e-4 * f) / std::sqrt(h1 * h1 + h2 * h2);

            double w = r_f * r_f;

            if (f > 18000.0)
            {
                const double rollOff = std::pow(10.0, -12.0 * (f - 18000.0) / std::max(1000.0, nyquistHz - 18000.0) / 20.0);
                w *= rollOff * rollOff;
            }

            return std::max(1.0e-6, w);
        };

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const double frequencyHz = static_cast<double>(bin) * binWidthHz;
            weights[static_cast<size_t>(bin)] = bandWeightForHz(frequencyHz);
            configuredWeightSum += weights[static_cast<size_t>(bin)];
        }

        const double maxBark = freqToBark(nyquistHz);
        const double barkStep = std::max(1.0e-9, maxBark / static_cast<double>(kBarkBandCount));

        for (int band = 0; band < kBarkBandCount; ++band)
        {
            const double centerBark = (static_cast<double>(band) + 0.5) * barkStep;
            bandCenterBark[static_cast<size_t>(band)] = centerBark;
            bandCenterFreqHz[static_cast<size_t>(band)] = barkToFreq(centerBark);
        }

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const double frequencyHz = static_cast<double>(bin) * binWidthHz;
            const double bark = freqToBark(frequencyHz);
            const double athDb = computeAthSplDb(frequencyHz) - kReferenceSplDb + kCalibrationOffsetDb;

            freqHz[static_cast<size_t>(bin)] = frequencyHz;
            barkHz[static_cast<size_t>(bin)] = bark;
            athThresholdDb[static_cast<size_t>(bin)] = athDb;
            athThresholdPower[static_cast<size_t>(bin)] = dbToPower(athDb);
            neighborRangeBins[static_cast<size_t>(bin)] = computeNeighborRangeHz(frequencyHz, binWidthHz);

            int band = static_cast<int>(bark / barkStep);
            band = std::clamp(band, 0, kBarkBandCount - 1);
            binToBand[static_cast<size_t>(bin)] = band;
        }

        windowCorrection = 1.0;
    }

    Result evaluate(const double* errorLeft,
                    const double* errorRight,
                    const std::array<double, kSpectrumBins>* maskingThresholds = nullptr) noexcept
    {
        // [v2.1] IPP は完全シングルスレッド設計のため、
        // mkl_set_num_threads_local(1) の呼び出しは不要。

        // [Bug 2/3 fix] IPP 初期化失敗時の安全フォールバック。
        // fftSpec == nullptr は constructor でのエラー (OOM等) を示す。
        // クラッシュを防ぐため、ゼロ結果を返す。
        if (fftSpec == nullptr || inputLeft == nullptr || inputRight == nullptr
            || spectrumLeft == nullptr || spectrumRight == nullptr)
            return Result{};

        double sumSq = 0.0;
        for (int i = 0; i < kFftLength; ++i)
        {
            sumSq += 0.5 * (errorLeft[i] * errorLeft[i] + errorRight[i] * errorRight[i]);
        }
        const double timeRms = std::sqrt(sumSq / kFftLength);

        juce::FloatVectorOperations::copy(inputLeft,  errorLeft,  kFftLength);
        juce::FloatVectorOperations::copy(inputRight, errorRight, kFftLength);

        // [v2.1] Forward FFT: real → CCS
        // 出力は CcsComplex 配列に直接書き込む (reinterpret_cast 安全: 同一メモリレイアウト)
        ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
        ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);

        std::array<double, kSpectrumBins> averagePower {};

        double flatnessLogSum = 0.0;
        double flatnessPowerSum = 0.0;
        double highBandEnergy = 0.0;
        double ultraHighEnergy = 0.0;
        double peakEnergy = 0.0;
        double totalEnergy = 0.0;
        int flatnessBins = 0;

        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            // [v2.1] MKL_Complex16 の .real/.imag → CcsComplex の .real/.imag
            // CcsComplex は同名メンバを持つため、元のコードとの差分なし。
            const double magSqLeft  = spectrumLeft[bin].real  * spectrumLeft[bin].real
                                    + spectrumLeft[bin].imag  * spectrumLeft[bin].imag;
            const double magSqRight = spectrumRight[bin].real * spectrumRight[bin].real
                                    + spectrumRight[bin].imag * spectrumRight[bin].imag;
            const double averageMagSq = std::max(kMinPower, 0.5 * (magSqLeft + magSqRight) * windowCorrection);
            const double safeMagSq = averageMagSq + kMinPower;

            averagePower[static_cast<size_t>(bin)] = averageMagSq;
            totalEnergy += averageMagSq;

            if (bin >= flatnessStartBin && bin <= flatnessEndBin)
            {
                flatnessLogSum += std::log(safeMagSq);
                flatnessPowerSum += safeMagSq;
                ++flatnessBins;
            }

            if (bin >= highBandStartBin)
                highBandEnergy += averageMagSq;

            if (bin >= ultraHighStartBin)
                ultraHighEnergy += averageMagSq;

            if (bin > 0 && bin < kSpectrumBins - 1)
            {
                const double prevMagSq = 0.5 * (
                    (spectrumLeft[bin-1].real  * spectrumLeft[bin-1].real  + spectrumLeft[bin-1].imag  * spectrumLeft[bin-1].imag) +
                    (spectrumRight[bin-1].real * spectrumRight[bin-1].real + spectrumRight[bin-1].imag * spectrumRight[bin-1].imag));
                const double nextMagSq = 0.5 * (
                    (spectrumLeft[bin+1].real  * spectrumLeft[bin+1].real  + spectrumLeft[bin+1].imag  * spectrumLeft[bin+1].imag) +
                    (spectrumRight[bin+1].real * spectrumRight[bin+1].real + spectrumRight[bin+1].imag * spectrumRight[bin+1].imag));

                const double localAvg = 0.5 * (prevMagSq + nextMagSq) + kMinPower;
                if (averageMagSq > 6.0 * localAvg)
                    peakEnergy = std::max(peakEnergy, averageMagSq);
            }
        }

        MaskerBuffer tonalMaskers;
        std::array<bool, kSpectrumBins> tonalConsumed {};
        detectTonalMaskersFixed(averagePower, tonalMaskers, tonalConsumed);

        MaskerBuffer noiseMaskers;
        buildNoiseMaskersFixed(averagePower, tonalConsumed, noiseMaskers);

        MaskerBuffer allMaskers;
        for (int i = 0; i < tonalMaskers.size; ++i)
            allMaskers.push(tonalMaskers.data[static_cast<size_t>(i)]);
        for (int i = 0; i < noiseMaskers.size; ++i)
            allMaskers.push(noiseMaskers.data[static_cast<size_t>(i)]);

        std::array<double, kSpectrumBins> maskingEnergy {};
        computeMaskingEnergyStable(allMaskers, maskingEnergy);

        std::array<double, kSpectrumBins> thresholdDb {};
        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const size_t idx = static_cast<size_t>(bin);
            double threshold = std::max(powerToDb(maskingEnergy[idx]), athThresholdDb[idx]);
            if (maskingThresholds != nullptr)
                threshold = std::max(threshold, powerToDb(std::max((*maskingThresholds)[idx], kMinPower)));
            thresholdDb[idx] = threshold;
        }

        double psychoWeighted = 0.0;
        double psychoWeightSum = 0.0;
        for (int bin = 0; bin < kSpectrumBins; ++bin)
        {
            const size_t idx = static_cast<size_t>(bin);
            const double signalDb = powerToDb(averagePower[idx]);
            const double deltaDb = signalDb - thresholdDb[idx];
            const double jndWeight = computeJndWeight(computeJndDb(freqHz[idx]));
            const double effectiveDb = smoothCap(softplus(deltaDb), kEffectiveCapDb);
            const double effectivePower = std::max(0.0, dbToPower(effectiveDb) - 1.0);
            const double weight = weights[idx] * jndWeight;

            psychoWeighted += weight * effectivePower;
            psychoWeightSum += weight;
        }

        Result result;
        result.noisePower = (psychoWeightSum > kMinPower)
                          ? ((psychoWeighted / psychoWeightSum) * static_cast<double>(kFftLength))
                          : 0.0;

        if (flatnessBins > 0)
        {
            const double arithmeticMean = flatnessPowerSum / static_cast<double>(flatnessBins);
            const double geometricMean = std::exp(flatnessLogSum / static_cast<double>(flatnessBins));
            const double flatness = std::clamp(geometricMean / std::max(arithmeticMean, kMinPower), 0.0, 1.0);
            result.spectralFlatnessPenalty = 1.0 - flatness;
        }

        const double observedUltraHighShare = ultraHighEnergy / std::max(highBandEnergy + kMinPower, kMinPower);
        const double excessUltraHighShare = std::max(0.0, observedUltraHighShare - expectedUltraHighShare);
        result.hfPenalty = excessUltraHighShare / std::max(1.0 - expectedUltraHighShare, kMinPower);
        result.timeDomainRms = timeRms;

        const double tonalRatio = peakEnergy / (totalEnergy + kMinPower);
        const double tonalPenalty = std::max(0.0, tonalRatio - 0.05) * 10.0;

        result.compositeScore = result.noisePower
                              * (1.0
                                 + (flatnessPenaltyWeight * result.spectralFlatnessPenalty)
                                 + (hfPenaltyWeight * result.hfPenalty)
                                 + tonalPenalty);
        return result;
    }

    double computeMaskingThreshold(double energy, double freq) const noexcept
    {
        const double safeEnergy = std::max(energy, kMinPower);
        const double bark = freqToBark(freq);
        const double spreadDb = -12.0 - (0.6 * bark);
        const double spreadPower = safeEnergy * dbToPower(spreadDb);
        const double athDb = computeAthSplDb(freq) - kReferenceSplDb + kCalibrationOffsetDb;
        const double athPower = dbToPower(athDb);
        return std::max(athPower, spreadPower);
    }

    // [v2.1] 引数型を MKL_Complex16* → CcsComplex* に変更。
    // CcsComplex は .real/.imag メンバを持ち MKL_Complex16 と同一レイアウト。
    // 呼び出し側 (NoiseShaperLearner) の変更: 型名を CcsComplex に変更するのみ。
    void computeFft(const double* dataL, const double* dataR,
                    CcsComplex* outL, CcsComplex* outR) noexcept
    {
        // [Bug 2/3 fix] IPP 初期化失敗時のガード。出力をゼロクリアして返す。
        if (fftSpec == nullptr || inputLeft == nullptr || inputRight == nullptr)
        {
            if (outL) std::memset(outL, 0, sizeof(CcsComplex) * kSpectrumBins);
            if (outR) std::memset(outR, 0, sizeof(CcsComplex) * kSpectrumBins);
            return;
        }

        juce::FloatVectorOperations::copy(inputLeft,  dataL, kFftLength);
        juce::FloatVectorOperations::copy(inputRight, dataR, kFftLength);
        // CcsComplex は [double real, double im] の標準レイアウト構造体。
        // IPP CCS 出力 [re0,im0,...] と同一メモリ配置のため reinterpret_cast 安全。
        ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(outL), fftSpec, fftWorkBuf);
        ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(outR), fftSpec, fftWorkBuf);
    }

private:
    static constexpr double kMinPower = 1.0e-24;
    static constexpr double kReferenceSplDb = 90.0;
    static constexpr double kCalibrationOffsetDb = 0.0;

    static constexpr double kEffectiveCapDb = 20.0;
    static constexpr double kSoftplusK = 2.0;
    static constexpr double kJndMin = 0.5;
    static constexpr double kJndLowPeak = 1.0;
    static constexpr double kJndHighSlope = 0.2;
    static constexpr double kJndWeightConstant = 0.3;

    static constexpr double kTonalPeakThresholdDb = 7.0;
    static constexpr double kNoiseMaskerCorrectionBaseDb = -5.0;
    static constexpr double kTonalAbsorbRadiusBark = 0.5;
    static constexpr double kSpreadMaxDeltaBark = 8.0;
    static constexpr double kTonalityFromSfmA = -0.299;
    static constexpr double kTonalityFromSfmB = -0.43;
    static constexpr double kSpreadUpDbPerBark = -27.0;
    static constexpr double kSpreadDownDbPerBarkTonal = -24.0;
    static constexpr double kSpreadDownDbPerBarkNoise = -27.0;

    static constexpr int kMaxMaskers = 128;
    static constexpr int kMaxContributions = 256;

    enum MaskerType
    {
        Tonal = 0,
        Noise = 1
    };

    struct Masker
    {
        double energy = 0.0;
        double bark = 0.0;
        double levelDb = -300.0;
        int type = Noise;
        double tonality = 0.0;
    };

    struct MaskerBuffer
    {
        std::array<Masker, kMaxMaskers> data {};
        int size = 0;

        void clear() noexcept { size = 0; }

        void push(const Masker& masker) noexcept
        {
            if (size < kMaxMaskers)
                data[static_cast<size_t>(size++)] = masker;
        }
    };

    struct ContributionBuffer
    {
        std::array<double, kMaxContributions> db {};
        int size = 0;

        void clear() noexcept { size = 0; }

        void push(double valueDb) noexcept
        {
            if (size < kMaxContributions)
                db[static_cast<size_t>(size++)] = valueDb;
        }
    };

    static double powerToDb(double power) noexcept
    {
        return 10.0 * std::log10(std::max(power, kMinPower));
    }

    static double dbToPower(double db) noexcept
    {
        return std::pow(10.0, db / 10.0);
    }

    static double softplus(double x) noexcept
    {
        const double z = kSoftplusK * x;
        if (z > 50.0) return x;
        if (z < -50.0) return std::exp(z) / kSoftplusK;
        return std::log1p(std::exp(z)) / kSoftplusK;
    }

    static double smoothCap(double x, double cap) noexcept
    {
        const double safeCap = std::max(1.0e-6, cap);
        return safeCap * std::tanh(x / safeCap);
    }

    static double freqToBark(double frequencyHz) noexcept
    {
        const double f = std::max(0.0, frequencyHz);
        return 13.0 * std::atan(0.00076 * f) + 3.5 * std::atan(std::pow(f / 7500.0, 2.0));
    }

    static double barkToFreq(double bark) noexcept
    {
        const double z = std::max(0.0, bark);
        return 600.0 * std::sinh(z / 6.0);
    }

    static double computeAthSplDb(double frequencyHz) noexcept
    {
        const double fKHz = std::max(0.01, frequencyHz / 1000.0);
        const double f2 = fKHz * fKHz;
        const double term1 = 3.64 * std::pow(fKHz, -0.8);
        const double term2 = -6.5 * std::exp(-0.6 * std::pow(fKHz - 3.3, 2.0));
        const double term3 = 0.001 * f2 * f2;
        return term1 + term2 + term3;
    }

    static double computeJndDb(double frequencyHz) noexcept
    {
        const double f = std::max(0.0, frequencyHz / 1000.0);
        const double lowPeak = kJndLowPeak * std::exp(-0.5 * (f - 0.5) * (f - 0.5));
        const double highShape = kJndHighSlope * (f - 3.0) * (f - 3.0);
        return std::clamp(kJndMin + lowPeak + highShape, kJndMin, 3.0);
    }

    static double computeJndWeight(double jndDb) noexcept
    {
        return 1.0 / std::max(1.0e-6, jndDb + kJndWeightConstant);
    }

    static double computeTonalityFromSfm(double sfm) noexcept
    {
        const double safeSfm = std::max(sfm, 1.0e-12);
        const double tonality = kTonalityFromSfmA + (kTonalityFromSfmB * std::log10(safeSfm));
        return std::clamp(tonality, 0.0, 1.0);
    }

    static double spreadingFunctionAnnexD(double deltaBark, int maskerType) noexcept
    {
        if (deltaBark >= 0.0)
            return kSpreadUpDbPerBark * deltaBark;

        const double slope = (maskerType == Tonal) ? kSpreadDownDbPerBarkTonal : kSpreadDownDbPerBarkNoise;
        const double x = deltaBark + 0.474;
        const double nonLinear = 15.81 + 7.5 * x - 17.5 * std::sqrt(1.0 + (x * x));
        return nonLinear + (slope + 27.0) * std::abs(deltaBark);
    }

    static int computeNeighborRangeHz(double frequencyHz, double binWidthHz) noexcept
    {
        const double fKHz = std::max(0.0, frequencyHz / 1000.0);
        const double bandwidth = 25.0 + 75.0 * std::pow(1.0 + 1.4 * fKHz * fKHz, 0.69);
        const int range = static_cast<int>((bandwidth / std::max(1.0, binWidthHz)) * 0.5);
        return std::clamp(range, 1, 24);
    }

    double getBinWidth(int bin) const noexcept
    {
        if (bin <= 0)
            return freqHz[1] - freqHz[0];
        if (bin >= (kSpectrumBins - 1))
            return freqHz[kSpectrumBins - 1] - freqHz[kSpectrumBins - 2];
        return 0.5 * (freqHz[static_cast<size_t>(bin + 1)] - freqHz[static_cast<size_t>(bin - 1)]);
    }

    void detectTonalMaskersFixed(const std::array<double, kSpectrumBins>& power,
                                 MaskerBuffer& maskers,
                                 std::array<bool, kSpectrumBins>& consumed) const noexcept
    {
        maskers.clear();
        consumed.fill(false);

        for (int i = 3; i < kSpectrumBins - 3; ++i)
        {
            const int range = neighborRangeBins[static_cast<size_t>(i)];
            const double centerDb = powerToDb(power[static_cast<size_t>(i)]);
            bool isPeak = true;

            for (int k = 1; k <= range; ++k)
            {
                if ((i - k) >= 0)
                {
                    const double leftDelta = centerDb - powerToDb(power[static_cast<size_t>(i - k)]);
                    if (leftDelta < kTonalPeakThresholdDb) { isPeak = false; break; }
                }
                if ((i + k) < kSpectrumBins)
                {
                    const double rightDelta = centerDb - powerToDb(power[static_cast<size_t>(i + k)]);
                    if (rightDelta < kTonalPeakThresholdDb) { isPeak = false; break; }
                }
            }

            if (!isPeak) continue;

            const double centerBark = barkHz[static_cast<size_t>(i)];
            double sumEnergy = 0.0;
            double sumBarkWeighted = 0.0;

            const int start = std::max(0, i - 8);
            const int end = std::min(kSpectrumBins - 1, i + 8);
            for (int j = start; j <= end; ++j)
            {
                if (std::abs(barkHz[static_cast<size_t>(j)] - centerBark) > kTonalAbsorbRadiusBark)
                    continue;
                const double e = power[static_cast<size_t>(j)] * getBinWidth(j);
                sumEnergy += e;
                sumBarkWeighted += barkHz[static_cast<size_t>(j)] * e;
                consumed[static_cast<size_t>(j)] = true;
            }

            if (sumEnergy <= kMinPower) continue;

            Masker masker;
            masker.energy = sumEnergy;
            masker.bark = sumBarkWeighted / sumEnergy;
            masker.levelDb = powerToDb(sumEnergy);
            masker.type = Tonal;
            masker.tonality = 1.0;
            maskers.push(masker);
        }
    }

    double computeSfm(const std::array<double, kSpectrumBins>& power,
                      const std::array<bool, kSpectrumBins>& tonalConsumed,
                      int targetBand) const noexcept
    {
        double logSum = 0.0;
        double linearSum = 0.0;
        int count = 0;

        for (int i = 0; i < kSpectrumBins; ++i)
        {
            if (binToBand[static_cast<size_t>(i)] != targetBand || tonalConsumed[static_cast<size_t>(i)])
                continue;
            const double p = std::max(power[static_cast<size_t>(i)], 1.0e-15);
            logSum += std::log(p);
            linearSum += p;
            ++count;
        }

        if (count <= 0) return 1.0;

        const double geometric = std::exp(logSum / static_cast<double>(count));
        const double arithmetic = linearSum / static_cast<double>(count);
        return geometric / std::max(arithmetic, 1.0e-15);
    }

    void buildNoiseMaskersFixed(const std::array<double, kSpectrumBins>& power,
                                const std::array<bool, kSpectrumBins>& tonalConsumed,
                                MaskerBuffer& maskers) const noexcept
    {
        maskers.clear();

        for (int band = 0; band < kBarkBandCount; ++band)
        {
            double sumEnergy = 0.0;
            double sumBarkWeighted = 0.0;
            int count = 0;

            for (int i = 0; i < kSpectrumBins; ++i)
            {
                if (binToBand[static_cast<size_t>(i)] != band || tonalConsumed[static_cast<size_t>(i)])
                    continue;
                const double e = power[static_cast<size_t>(i)] * getBinWidth(i);
                sumEnergy += e;
                sumBarkWeighted += barkHz[static_cast<size_t>(i)] * e;
                ++count;
            }

            if (count <= 0 || sumEnergy <= kMinPower) continue;

            const double sfm = computeSfm(power, tonalConsumed, band);

            Masker masker;
            masker.energy = sumEnergy;
            masker.bark = sumBarkWeighted / sumEnergy;
            masker.levelDb = powerToDb(sumEnergy);
            masker.type = Noise;
            masker.tonality = computeTonalityFromSfm(sfm);
            maskers.push(masker);
        }
    }

    void computeMaskingEnergyStable(const MaskerBuffer& maskers,
                                    std::array<double, kSpectrumBins>& maskingEnergy) const noexcept
    {
        constexpr double kLogScale = 0.2302585093; // ln(10) / 10

        for (int i = 0; i < kSpectrumBins; ++i)
        {
            ContributionBuffer contributions;
            contributions.clear();

            double maxDb = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < maskers.size; ++j)
            {
                const Masker& masker = maskers.data[static_cast<size_t>(j)];
                const double deltaBark = barkHz[static_cast<size_t>(i)] - masker.bark;
                if (std::abs(deltaBark) > kSpreadMaxDeltaBark) continue;

                double levelDb = masker.levelDb;
                if (masker.type == Noise)
                    levelDb += kNoiseMaskerCorrectionBaseDb * (1.0 - masker.tonality);

                const double totalDb = levelDb + spreadingFunctionAnnexD(deltaBark, masker.type);
                contributions.push(totalDb);
                if (totalDb > maxDb) maxDb = totalDb;
            }

            if (contributions.size <= 0 || !std::isfinite(maxDb))
            {
                maskingEnergy[static_cast<size_t>(i)] = athThresholdPower[static_cast<size_t>(i)];
                continue;
            }

            double sum = 0.0;
            for (int k = 0; k < contributions.size; ++k)
            {
                const double valueDb = contributions.db[static_cast<size_t>(k)];
                sum += std::exp((valueDb - maxDb) * kLogScale);
            }

            const double totalPower = std::exp(maxDb * kLogScale) * sum;
            maskingEnergy[static_cast<size_t>(i)] = std::max(totalPower, athThresholdPower[static_cast<size_t>(i)]);
        }
    }

    // ── データバッファ (mkl_malloc 64バイトアライン) ──
    double*     inputLeft     = nullptr;  ///< FFT 入力 L ch (kFftLength doubles)
    double*     inputRight    = nullptr;  ///< FFT 入力 R ch (kFftLength doubles)
    CcsComplex* spectrumLeft  = nullptr;  ///< FFT 出力 L ch (kSpectrumBins CcsComplex)
    CcsComplex* spectrumRight = nullptr;  ///< FFT 出力 R ch (kSpectrumBins CcsComplex)

    // ── IPP FFT リソース ──
    IppsFFTSpec_R_64f* fftSpec    = nullptr; ///< IPP FFT スペック (fftSpecBuf 内を指す)
    Ipp8u*             fftSpecBuf = nullptr; ///< fftSpec のメモリオーナー
    Ipp8u*             fftWorkBuf = nullptr; ///< FFT スクラッチ (sizeWork==0 なら nullptr)

    // ── 分析パラメータ ──
    std::array<double, kSpectrumBins> weights {};
    std::array<double, kSpectrumBins> freqHz {};
    std::array<double, kSpectrumBins> barkHz {};
    std::array<int,    kSpectrumBins> binToBand {};
    std::array<int,    kSpectrumBins> neighborRangeBins {};
    std::array<double, kSpectrumBins> athThresholdDb {};
    std::array<double, kSpectrumBins> athThresholdPower {};
    std::array<double, kBarkBandCount> bandCenterBark {};
    std::array<double, kBarkBandCount> bandCenterFreqHz {};

    double configuredSampleRateHz = kDefaultSampleRateHz;
    double windowCorrection = 1.0;

    int flatnessStartBin  = 0;
    int flatnessEndBin    = kSpectrumBins - 1;
    int highBandStartBin  = 0;
    int ultraHighStartBin = kSpectrumBins - 1;

    double expectedUltraHighShare  = 0.0;
    double configuredWeightSum     = 1.0;
    double flatnessPenaltyWeight   = 0.35;
    double hfPenaltyWeight         = 0.20;
};
