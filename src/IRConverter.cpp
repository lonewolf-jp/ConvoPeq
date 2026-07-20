#include "IRConverter.h"
#include "IRDSP.h"
#include "IRAnalyzer.h"  // ★ v14.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <mkl.h>
#include <mkl_cblas.h>
#include "DiagnosticsConfig.h"

//==============================================================================
// ★ v14.0: 第1段 — Energy 補正（基本 scaleFactor + safetyMargin）
//==============================================================================
static double computeEnergyScale(const juce::AudioBuffer<double>& ir) noexcept
{
    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return 1.0;

    double maxChannelEnergy = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = ir.getReadPointer(ch);
        const double energy = cblas_ddot(numSamples, data, 1, data, 1);
        if (std::isfinite(energy) && energy > 1.0e-18)
            maxChannelEnergy = std::max(maxChannelEnergy, energy);
    }

    if (!(maxChannelEnergy > 1.0e-18) || !std::isfinite(maxChannelEnergy))
        return 1.0;

    constexpr double safetyMargin = 0.5011872336272722;  // -6dB
    return (1.0 / std::sqrt(maxChannelEnergy)) * safetyMargin;
}

//==============================================================================
// ★ v14.0: 第2段 — IR 解析（Peak/RMS + FFT）
//==============================================================================
struct IRAnalysisResult {
    double peakValue = 0.0;
    double rmsValue = 0.0;
    double frequencyPeakGain = 1.0;
};

static IRAnalysisResult analyzeIR(const juce::AudioBuffer<double>& ir, double currentScale) noexcept
{
    IRAnalysisResult result;
    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return result;

    double irPeak = 0.0;
    double irEnergySum = 0.0;

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = ir.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            const double value = data[i];
            irPeak = std::max(irPeak, std::abs(value));
            irEnergySum += value * value;
        }
    }

    result.peakValue = irPeak;
    const int totalSamples = numChannels * numSamples;
    result.rmsValue = (totalSamples > 0) ? std::sqrt(irEnergySum / static_cast<double>(totalSamples)) : 0.0;

    // FFT 解析（IRAnalyzer に委譲）
    result.frequencyPeakGain = IRAnalyzer::estimateMaxFrequencyResponseGain(ir);

    return result;
}

//==============================================================================
// ★ v14.0: 第3段 — 保護クランプ適用
//==============================================================================
static void applyClampProtection(IRConverter::ScaleFactorResult& result,
                                  double scale,
                                  const IRAnalysisResult& analysis,
                                  const juce::AudioBuffer<double>* currentIr,
                                  double currentScale,
                                  const juce::AudioBuffer<double>& ir) noexcept
{
    double peakAttenDb = 0.0, rmsAttenDb = 0.0, freqAttenDb = 0.0;

    // Peak クランプ
    constexpr double kMaxEffectivePeak = 0.5;
    if (analysis.peakValue * scale > kMaxEffectivePeak)
    {
        const double peakClamp = kMaxEffectivePeak / (analysis.peakValue * scale);
        result.scaleFactor *= peakClamp;
        scale *= peakClamp;  // ★ v14.0: 順序依存対応
        peakAttenDb = -20.0 * std::log10(peakClamp);
    }

    // RMS クランプ（Peak 適用後の scale で判定）
    constexpr double kMaxEffectiveRms = 0.25;
    if (analysis.rmsValue * scale > kMaxEffectiveRms)
    {
        const double rmsClamp = kMaxEffectiveRms / (analysis.rmsValue * scale);
        result.scaleFactor *= rmsClamp;
        rmsAttenDb = -20.0 * std::log10(rmsClamp);
    }

    // 周波数応答ピーククランプ
    constexpr double kMaxEffectiveFreqResponse = 1.41; // +3dB
    if (analysis.frequencyPeakGain > kMaxEffectiveFreqResponse)
    {
        const double freqClip = kMaxEffectiveFreqResponse / analysis.frequencyPeakGain;
        result.scaleFactor *= freqClip;
        freqAttenDb = -20.0 * std::log10(freqClip);
    }

    // additionalAttenuationDb = 追加減衰量（energy補正含まず）
    result.additionalAttenuationDb = static_cast<float>(peakAttenDb + rmsAttenDb + freqAttenDb);

    // ★ 既存: 現在の IR との比較による過大ジャンプ保護
    if (currentIr != nullptr && currentIr->getNumChannels() > 0 && currentIr->getNumSamples() > 0)
    {
        auto computePeakAndRmsWithScale = [](const juce::AudioBuffer<double>& buffer, double scale) -> std::pair<double, double>
        {
            const int channels = buffer.getNumChannels();
            const int samples = buffer.getNumSamples();
            if (channels <= 0 || samples <= 0)
                return { 0.0, 0.0 };

            double peak = 0.0;
            double energy = 0.0;
            for (int ch = 0; ch < channels; ++ch)
            {
                const double* data = buffer.getReadPointer(ch);
                for (int i = 0; i < samples; ++i)
                {
                    const double value = data[i] * scale;
                    peak = std::max(peak, std::abs(value));
                    energy += value * value;
                }
            }
            return { peak, std::sqrt(energy / static_cast<double>(channels * samples)) };
        };

        const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
        const auto [newPeak, newRms] = computePeakAndRmsWithScale(ir, result.scaleFactor);

        const bool excessivePeakJump = currentPeak > 1.0e-9 && newPeak > currentPeak * 4.0 && newPeak > 0.5;
        const bool excessiveRmsJump = currentRms > 1.0e-9 && newRms > currentRms * 4.0 && newRms > 0.25;

        if (excessivePeakJump || excessiveRmsJump)
        {
            double clampByPeak = std::numeric_limits<double>::infinity();
            double clampByRms = std::numeric_limits<double>::infinity();

            if (newPeak > 1.0e-12 && currentPeak > 1.0e-12)
                clampByPeak = (currentPeak * 4.0) / newPeak;
            if (newRms > 1.0e-12 && currentRms > 1.0e-12)
                clampByRms = (currentRms * 4.0) / newRms;

            const double clampRatio = std::min(clampByPeak, clampByRms);
            if (std::isfinite(clampRatio) && clampRatio > 0.0 && clampRatio < 1.0)
                result.scaleFactor *= clampRatio;
        }
    }
}

//==============================================================================
// computeScaleFactor — 3段階オーケストレーター（★ v14.0）
//==============================================================================
IRConverter::ScaleFactorResult IRConverter::computeScaleFactor(const juce::AudioBuffer<double>& ir,
                                                               const juce::AudioBuffer<double>* currentIr,
                                                               double currentScale) noexcept
{
    ScaleFactorResult result;

    // 第1段: Energy 補正
    double scale = computeEnergyScale(ir);
    if (scale <= 0.0 || !std::isfinite(scale))
        return result;

    result.scaleFactor = scale;
    result.hasScaleFactor = true;

    // 第2段: IR 解析（Peak/RMS/FFT）
    const auto analysis = analyzeIR(ir, scale);

    // 第3段: 保護クランプ
    applyClampProtection(result, scale, analysis, currentIr, currentScale, ir);

    return result;
}

bool IRConverter::loadAudioFile(const juce::File& file,
                                juce::AudioBuffer<double>& out,
                                double& sampleRateOut)
{
    if (!file.existsAsFile())
        return false;

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
    if (!reader)
        return false;

    const int64 n = reader->lengthInSamples;
    if (n <= 0 || n > static_cast<int64>(std::numeric_limits<int>::max()))
        return false;

    const int channels = static_cast<int>(reader->numChannels);
    if (channels <= 0)
        return false;

    juce::AudioBuffer<float> temp(channels, static_cast<int>(n));
    if (!reader->read(&temp, 0, static_cast<int>(n), 0, true, true))
        return false;

    out.setSize(channels, static_cast<int>(n));
    for (int ch = 0; ch < channels; ++ch)
    {
        const float* src = temp.getReadPointer(ch);
        double* dst = out.getWritePointer(ch);
        for (int i = 0; i < static_cast<int>(n); ++i)
            dst[i] = static_cast<double>(src[i]);
    }

    sampleRateOut = reader->sampleRate;
    return true;
}

std::unique_ptr<PreparedIRState> IRConverter::convertFile(const juce::File& irFile,
                                                          const ConvertConfig& config,
                                                          const std::function<bool()>& shouldCancel) const
{
    juce::AudioBuffer<double> ir;
    double sourceRate = 0.0;
    if (!loadAudioFile(irFile, ir, sourceRate))
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: loadAudioFile failed for "
            + irFile.getFullPathName());
        return nullptr;
    }

    if (shouldCancel && shouldCancel())
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: cancelled after load");
        return nullptr;
    }

    juce::AudioBuffer<double> converted = ir;
    double actualSampleRate = sourceRate;
    if (config.targetSampleRate > 0.0 && sourceRate > 0.0 && std::abs(sourceRate - config.targetSampleRate) > 1.0e-6)
    {
        converted = IRDSP::resampleIR(ir, sourceRate, config.targetSampleRate, shouldCancel);
        if (converted.getNumSamples() <= 0)
        {
            // ★ Workaround: r8brain resampling failed (e.g., 48000→192000 Hz).
            // Fall back to original IR. Report the IR at the target sample rate
            // so the convolver engine uses the correct processing rate.
            // The engine handles internal sample rate conversion.
            juce::Logger::writeToLog("[DIAG_IR] convertFile: resampleIR failed, "
                "falling back to original IR (srcSr=" + juce::String(sourceRate, 1)
                + " targetSr=" + juce::String(config.targetSampleRate, 1) + ")");
            converted = ir;
            actualSampleRate = config.targetSampleRate;
        }
        else
        {
            actualSampleRate = config.targetSampleRate;
        }
    }
    else
    {
        actualSampleRate = (config.targetSampleRate > 0.0) ? config.targetSampleRate : sourceRate;
    }

    if (shouldCancel && shouldCancel())
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: cancelled after resample");
        return nullptr;
    }

    const int fftSize = juce::jmax(32, config.fftSize);
    const int usableChannels = juce::jmax(1, converted.getNumChannels());
    const int samples = converted.getNumSamples();

    const int numPartitions = juce::jmax(1, (samples + fftSize - 1) / fftSize);
    const size_t totalSamples = static_cast<size_t>(numPartitions) * static_cast<size_t>(fftSize) * static_cast<size_t>(usableChannels);
    const size_t bytes = totalSamples * sizeof(double);

    double* data = static_cast<double*>(DIAG_MKL_MALLOC(bytes, 64));
    if (!data)
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: MKL_MALLOC failed bytes=" + juce::String(static_cast<int>(bytes)));
        return nullptr;
    }

    std::memset(data, 0, bytes);

    for (int ch = 0; ch < usableChannels; ++ch)
    {
        const double* src = converted.getReadPointer(ch);
        for (int i = 0; i < samples; ++i)
        {
            if ((i & 0xF) == 0 && shouldCancel && shouldCancel())
            {
                mkl_free(data);
                return nullptr;
            }

            const size_t idx = static_cast<size_t>(ch) * static_cast<size_t>(numPartitions) * static_cast<size_t>(fftSize)
                             + static_cast<size_t>(i);
            data[idx] = src[i];
        }
    }

    auto prepared = std::make_unique<PreparedIRState>();
    prepared->partitionData = data;
    prepared->partitionSizeBytes = bytes;
    prepared->numPartitions = numPartitions * usableChannels;
    prepared->fftSize = fftSize;
    prepared->numChannels = usableChannels;
    prepared->sampleRate = actualSampleRate;
    prepared->generationId = config.generationId;
    prepared->cacheKey = config.cacheKey;

    // 時間領域 IR を保持（UI 表示用）
    // converted はリサンプリング済みの加工済み IR
    prepared->timeDomainIR = std::make_unique<juce::AudioBuffer<double>>(std::move(converted));

    if (prepared->timeDomainIR && prepared->timeDomainIR->getNumSamples() > 0)
    {
        const auto scaleInfo = computeScaleFactor(*prepared->timeDomainIR);
        prepared->scaleFactor = scaleInfo.scaleFactor;
        prepared->hasScaleFactor = scaleInfo.hasScaleFactor;
        prepared->additionalAttenuationDb = scaleInfo.additionalAttenuationDb;

        // ★ v14.2: IRAnalyzer による周波数ピークゲイン推定
        //   scaledIR = timeDomainIR × scaleFactor を解析
        juce::AudioBuffer<double> scaledIR(*prepared->timeDomainIR);
        scaledIR.applyGain(prepared->scaleFactor);

        // Diagnostic: check if scaled IR has any data
        {
            double peak = 0.0;
            for (int ch = 0; ch < scaledIR.getNumChannels(); ++ch) {
                const double* d = scaledIR.getReadPointer(ch);
                for (int i = 0; i < std::min(100, scaledIR.getNumSamples()); ++i)
                    peak = std::max(peak, std::abs(d[i]));
            }
            juce::Logger::writeToLog("[DIAG_SCALE] scaleFactor=" + juce::String(prepared->scaleFactor, 6)
                + " scaledPeak=" + juce::String(peak, 8)
                + " timeDomainSamples=" + juce::String(prepared->timeDomainIR->getNumSamples()));
        }

        const double freqPeakLin = IRAnalyzer::estimateMaxFrequencyResponseGain(scaledIR);
        prepared->irFreqPeakGainDb = (freqPeakLin > 1e-18)
            ? static_cast<float>(20.0 * std::log10(freqPeakLin))
            : 0.0f;
        juce::Logger::writeToLog("[DIAG_IR_FREQ] freqPeakLin=" + juce::String(freqPeakLin, 8)
            + " irFreqPeakGainDb=" + juce::String(prepared->irFreqPeakGainDb, 2)
            + " scaleFactor=" + juce::String(prepared->scaleFactor, 6)
            + " sampleRate=" + juce::String(prepared->sampleRate, 1));
    }

    return prepared;
}

std::unique_ptr<PreparedIRState> IRConverter::convertToHighRes(const juce::File& irFile,
                                                               double sampleRate,
                                                               int nextFFTSize,
                                                               uint64_t generationId,
                                                               uint64_t cacheKey,
                                                               const std::function<bool()>& shouldCancel) const
{
    ConvertConfig cfg;
    cfg.fftSize = nextFFTSize;
    cfg.partitionSize = nextFFTSize;
    cfg.targetSampleRate = sampleRate;
    cfg.generationId = generationId;
    cfg.cacheKey = cacheKey;
    return convertFile(irFile, cfg, shouldCancel);
}

//==============================================================================
// ★ v14.0: 後方互換用デリゲート — IRAnalyzer に委譲
//==============================================================================
double IRConverter::estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir,
    double /*sampleRate*/) noexcept
{
    return IRAnalyzer::estimateMaxFrequencyResponseGain(ir);
}
