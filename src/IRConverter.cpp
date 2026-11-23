#include "IRConverter.h"
#include "IRDSP.h"
#include "IRAnalyzer.h"  // ★ v14.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

#include <mkl.h>
#include <mkl_cblas.h>
#include "DiagnosticsConfig.h"
#include "convolver/IRLoadAdmission.h"
#include "InputInspection.h"
#include "convolver/IRTrailingSilence.h"

namespace
{
// LoaderThread の kStreamChunk と同一。入力の float 一時バッファを固定長に抑える。
constexpr int kIRStreamChunk = 256 * 1024;

// B-01: 同時に確保する double/float バッファの合計を、既存の kMaxIRLoadBytes（1 GiB）で
// 制約する。bytesPerSample はその経路で同時に生きる double 本数（8 = 1 本、16 = 2 本、
// 24 = 3 本）。除算形で評価し、未検証値の積を作らない。
bool admitWorkingSet (int64_t channels, int64_t samples, int64_t bytesPerSample,
                      int64_t extraBytes, juce::String& why)
{
    if (channels < 1 || samples < 1 || bytesPerSample < 1 || extraBytes < 0)
        return false;

    const int64_t budget = convo::irload::kMaxIRLoadBytes;

    if (extraBytes > budget)
    {
        why = "IR too large: transient buffers exceed the load budget";
        return false;
    }

    const int64_t perSample = channels * bytesPerSample;
    if (perSample <= 0)
        return false;

    if (samples > (budget - extraBytes) / perSample)
    {
        why = "IR too large: working set (" + juce::String (channels * samples * bytesPerSample
                                                           + extraBytes)
              + " bytes) exceeds the load budget (" + juce::String (budget) + " bytes)";
        return false;
    }

    return true;
}
} // namespace

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
                                double& sampleRateOut,
                                juce::String* errorOut)
{
    const auto fail = [errorOut](const juce::String& why)
    {
        if (errorOut != nullptr)
            *errorOut = why;

        return false;
    };

    if (!file.existsAsFile())
        return fail ("IR file not found: " + file.getFileName());

    // ★ B-04: JUCE の WAV リーダーはメタデータチャンクの宣言長で確保するため、
    //   createReaderFor の前にチャンク境界を検査する。
    {
        juce::String riffReason;

        if (!convo::safeinput::inspectRiffChunkBounds (file, &riffReason))
            return fail (riffReason);
    }

    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader (fm.createReaderFor(file));
    if (reader == nullptr)
        return fail ("Unsupported or unreadable IR file: " + file.getFileName());

    // ★ B-01: 同期経路にも凍結済み admission を適用する（LoaderThread と同一契約）。
    //   narrow する前に unsigned / int64 の値で判定する（FC-INV-9 の順序）。
    const unsigned numChannels = reader->numChannels;
    const juce::int64 fileLength = reader->lengthInSamples;

    if (! convo::irload::admitChannelCountNonZero (numChannels))
        return fail (convo::irload::diagnosticChannelLimit (numChannels));

    if (! convo::irload::admitFileLengthNonZero (fileLength))
        return fail (convo::irload::diagnosticLengthLimit (fileLength));

    if (! convo::irload::admitFileLengthRepresentable (fileLength))
        return fail (convo::irload::diagnosticLengthLimit (fileLength));

    if (! convo::irload::admitChannelCount (numChannels))
        return fail (convo::irload::diagnosticChannelLimit (numChannels));

    if (! convo::irload::admitByteBudget (numChannels, fileLength))
        return fail (convo::irload::diagnosticByteLimit (numChannels, fileLength));

    // ★ C-06: 宣言長だけで無制限に確保させない。2^24 標本は 48 kHz で約 5.8 分で、
    //   実用的な IR より十分に長い。作業バッファを作る前に拒否する。
    static constexpr juce::int64 kMaxIrSamples = juce::int64{1} << 24;
    if (fileLength > kMaxIrSamples)
    {
        juce::Logger::writeToLog("[DIAG_IR] loadAudioFile: rejected length=" + juce::String(fileLength));
        return fail ("IR file is too long: " + file.getFileName());
    }

    // ★ B-01: 実行時に同時に確保する量（double IR 8·ch·N ＋ float chunk 4·ch·chunk）も
    //   既存の kMaxIRLoadBytes で制約する。新しい上限値は導入しない。
    const juce::int64 chunkSamples = std::min<juce::int64> (fileLength, kIRStreamChunk);
    {
        juce::String why;

        if (! admitWorkingSet (static_cast<int64_t> (numChannels), fileLength, 8,
                               static_cast<int64_t> (numChannels) * chunkSamples * 4, why))
            return fail (why);
    }

    const int channels = static_cast<int> (numChannels);
    const int totalSamples = static_cast<int> (fileLength);

    try
    {
        // ★ B-01: 256K サンプル単位で float→double 変換する（LoaderThread の
        //   kStreamChunk と同一）。float 一時バッファはチャンク分に固定され、
        //   fileLength に比例しない。
        juce::AudioBuffer<float> temp (channels, static_cast<int> (chunkSamples));
        out.setSize (channels, totalSamples);

        for (juce::int64 offset = 0; offset < fileLength; offset += kIRStreamChunk)
        {
            const int chunk = static_cast<int> (std::min<juce::int64> (kIRStreamChunk,
                                                                      fileLength - offset));

            if (! reader->read (&temp, 0, chunk, offset, true, true))
                return fail ("Failed to read audio data from IR file: " + file.getFileName());

            for (int ch = 0; ch < channels; ++ch)
            {
                const float* src = temp.getReadPointer (ch);
                double* dst = out.getWritePointer (ch);

                for (int i = 0; i < chunk; ++i)
                    dst[offset + i] = static_cast<double> (src[i]);
            }
        }
    }
    catch (const std::bad_alloc&)
    {
        return fail ("Not enough memory to load IR file: " + file.getFileName());
    }
    catch (const std::exception& e)
    {
        return fail ("Failed to load IR file: " + file.getFileName() + " (" + e.what() + ")");
    }


    sampleRateOut = reader->sampleRate;
    return true;
}

std::unique_ptr<PreparedIRState> IRConverter::convertFile(const juce::File& irFile,
                                                          const ConvertConfig& config,
                                                          const std::function<bool()>& shouldCancel,
                                                          juce::String* errorOut) const
{
    const auto fail = [errorOut](const juce::String& why)
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: " + why);

        if (errorOut != nullptr)
            *errorOut = why;

        return std::unique_ptr<PreparedIRState>();
    };

    juce::AudioBuffer<double> ir;
    double sourceRate = 0.0;
    juce::String loadError;

    try
    {
        if (! loadAudioFile (irFile, ir, sourceRate, &loadError))
            return fail (loadError.isNotEmpty()
                             ? loadError
                             : ("Failed to load IR file: " + irFile.getFileName()));
    }
    catch (const std::bad_alloc&)
    {
        return fail ("Not enough memory to load IR file: " + irFile.getFileName());
    }
    catch (const std::exception& e)
    {
        return fail ("Failed to load IR file: " + irFile.getFileName() + " (" + e.what() + ")");
    }

    if (shouldCancel && shouldCancel())
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: cancelled after load");
        return nullptr;
    }

    const int loadedChannels = ir.getNumChannels();
    const int rawSamples = ir.getNumSamples();

    if (loadedChannels <= 0 || rawSamples <= 0)
        return fail ("IR file contains no audio data: " + irFile.getFileName());

    // ★ B-01 / B-02: 末尾無音トリムを本適用（LoaderThread）と同じ規則で行う。
    //   トリム後の長さで FC-FORM-5 を評価し、resample するのもトリム後のバッファにする。
    //   従来は同期経路だけトリムせず、全体長で判定していたため、無音テールの長い IR で
    //   「プレビューは拒否、本適用は受理」という差が出ていた。
    {
        const int trimmed = convo::irtrim::computeTrimmedLength (
            ir.getReadPointer (0),
            loadedChannels > 1 ? ir.getReadPointer (1) : nullptr,
            rawSamples);
        if (trimmed > 0 && trimmed < rawSamples)
        {
            ir.setSize (loadedChannels, trimmed, true);
            juce::Logger::writeToLog ("[DIAG_IR] trailing silence trimmed: "
                                      + juce::String (rawSamples) + " -> " + juce::String (trimmed));
        }
    }

    const int loadedSamples = ir.getNumSamples();

    const bool needsResample = config.targetSampleRate > 0.0 && sourceRate > 0.0
                            && std::abs (sourceRate - config.targetSampleRate) > 1.0e-6;

    // ★ B-01: resample 出力長の上限（FC-FORM-5）を resample の前に評価する。
    //   この経路は末尾無音トリムを行わないため、評価対象は実際に resample する長さ
    //   そのもの（loadedSamples）になる（設計書 §7.4 / §7.6）。
    if (needsResample
        && ! convo::irload::admitResampleOutput (loadedSamples, sourceRate, config.targetSampleRate))
    {
        return fail (convo::irload::diagnosticResampleLimit (loadedSamples, sourceRate,
                                                             config.targetSampleRate));
    }

    juce::AudioBuffer<double> converted;
    double actualSampleRate = sourceRate;

    if (needsResample)
    {
        // 入力 8·ch·N と出力 8·ch·N′ が同時に生きるため、その合計を既存予算で制約する。
        const juce::int64 estimatedOut =
            static_cast<juce::int64> (std::ceil (static_cast<double> (loadedSamples)
                                                 * config.targetSampleRate / sourceRate)) + 2;
        {
            juce::String why;

            if (! admitWorkingSet (loadedChannels, estimatedOut, 8,
                                   static_cast<int64_t> (loadedChannels) * loadedSamples * 8, why))
                return fail (why);
        }

        try
        {
            converted = IRDSP::resampleIR (ir, sourceRate, config.targetSampleRate, shouldCancel);
        }
        catch (const std::bad_alloc&)
        {
            return fail ("Not enough memory to resample IR file: " + irFile.getFileName());
        }
        catch (const std::exception& e)
        {
            return fail ("Failed to resample IR file: " + irFile.getFileName() + " (" + e.what() + ")");
        }

        if (converted.getNumSamples() <= 0)
        {
            // ★ r8brain resample failed. IR data is at sourceRate, so label it
            //   accordingly. Previously this mislabeled as targetSampleRate,
            //   corrupting downstream frequency analysis.
            juce::Logger::writeToLog("[DIAG_IR] convertFile: resampleIR failed, "
                "falling back to original IR (srcSr=" + juce::String (sourceRate, 1)
                + " targetSr=" + juce::String (config.targetSampleRate, 1) + ")");
            converted = std::move (ir);
            actualSampleRate = sourceRate;
        }
        else
        {
            // ★ B-01: r8brain の丸め差があるため、実長でも上限を確認する。
            if (converted.getNumSamples() > convo::irload::kMaxIRResampleOutputSamples)
                return fail (convo::irload::diagnosticResampleLimit (converted.getNumSamples(),
                                                                     config.targetSampleRate,
                                                                     config.targetSampleRate));

            actualSampleRate = config.targetSampleRate;

            ir = juce::AudioBuffer<double>(); // 入力はもう不要（後段の同時確保量を減らす・noexcept）
        }
    }
    else
    {
        converted = std::move (ir);
        actualSampleRate = (config.targetSampleRate > 0.0) ? config.targetSampleRate : sourceRate;
    }

    if (shouldCancel && shouldCancel())
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: cancelled after resample");
        return nullptr;
    }

    const int fftSize = juce::jmax (32, config.fftSize);
    const int usableChannels = juce::jmax (1, converted.getNumChannels());
    const int samples = converted.getNumSamples();

    if (samples <= 0)
        return fail ("IR file contains no samples: " + irFile.getFileName());

    const int numPartitions = juce::jmax (1, (samples + fftSize - 1) / fftSize);
    const juce::uint64 totalSamples = static_cast<juce::uint64> (numPartitions)
                                    * static_cast<juce::uint64> (fftSize)
                                    * static_cast<juce::uint64> (usableChannels);
    const juce::uint64 bytes64 = totalSamples * static_cast<juce::uint64> (sizeof (double));

    if (bytes64 > static_cast<juce::uint64> (std::numeric_limits<size_t>::max()))
        return fail ("IR partition buffer size overflows: " + irFile.getFileName());

    const size_t bytes = static_cast<size_t> (bytes64);

    // ★ B-01: ピーク時に同時に生きるのは converted/IR(8·ch·N′)＋packed(8·ch·N′)＋
    //   解析用コピー(8·ch·N′)。その合計を既存予算で制約する。
    {
        juce::String why;

        if (! admitWorkingSet (usableChannels, samples,
                               static_cast<int64_t> (sizeof (double)) * 2,
                               static_cast<int64_t> (bytes), why))
            return fail (why);
    }

    double* data = static_cast<double*> (DIAG_MKL_MALLOC (bytes, 64));
    if (data == nullptr)
    {
        juce::Logger::writeToLog("[DIAG_IR] convertFile: MKL_MALLOC failed bytes=" + juce::String (static_cast<int> (bytes)));
        return fail ("Not enough memory for the IR partition buffer: " + irFile.getFileName());
    }

    std::memset(data, 0, bytes);

    for (int ch = 0; ch < usableChannels; ++ch)
    {
        const double* src = converted.getReadPointer (ch);
        for (int i = 0; i < samples; ++i)
        {
            if ((i & 0xF) == 0 && shouldCancel && shouldCancel())
            {
                DIAG_MKL_FREE (data, bytes);
                return nullptr;
            }

            const size_t idx = static_cast<size_t> (ch) * static_cast<size_t> (numPartitions) * static_cast<size_t> (fftSize)
                             + static_cast<size_t> (i);
            data[idx] = src[i];
        }
    }

    std::unique_ptr<PreparedIRState> prepared;
    bool partitionOwned = false;

    try
    {
        prepared = std::make_unique<PreparedIRState>();
        prepared->partitionData = data;
        partitionOwned = true; // 以降の解放は PreparedIRState が担う
        prepared->partitionSizeBytes = bytes;
        prepared->numPartitions = numPartitions * usableChannels;
        prepared->fftSize = fftSize;
        prepared->numChannels = usableChannels;
        prepared->sampleRate = actualSampleRate;
        prepared->generationId = config.generationId;
        prepared->cacheKey = config.cacheKey;

        // 時間領域 IR を保持（UI 表示用）
        // converted はリサンプリング済みの加工済み IR
        prepared->timeDomainIR = std::make_unique<juce::AudioBuffer<double>> (std::move (converted));

        if (prepared->timeDomainIR != nullptr && prepared->timeDomainIR->getNumSamples() > 0)
        {
            const auto scaleInfo = computeScaleFactor (*prepared->timeDomainIR);
            prepared->scaleFactor = scaleInfo.scaleFactor;
            prepared->hasScaleFactor = scaleInfo.hasScaleFactor;
            prepared->additionalAttenuationDb = scaleInfo.additionalAttenuationDb;

            // ★ v14.2: IRAnalyzer による周波数ピークゲイン推定
            //   scaledIR = timeDomainIR × scaleFactor を解析
            juce::AudioBuffer<double> scaledIR (*prepared->timeDomainIR);
            scaledIR.applyGain (prepared->scaleFactor);

            // Diagnostic: check if scaled IR has any data
            {
                double peak = 0.0;
                for (int ch = 0; ch < scaledIR.getNumChannels(); ++ch) {
                    const double* d = scaledIR.getReadPointer (ch);
                    for (int i = 0; i < std::min (100, scaledIR.getNumSamples()); ++i)
                        peak = std::max (peak, std::abs (d[i]));
                }
                juce::Logger::writeToLog("[DIAG_SCALE] scaleFactor=" + juce::String (prepared->scaleFactor, 6)
                    + " scaledPeak=" + juce::String (peak, 8)
                    + " timeDomainSamples=" + juce::String (prepared->timeDomainIR->getNumSamples()));
            }

            const double freqPeakLin = IRAnalyzer::estimateMaxFrequencyResponseGain (scaledIR);
            prepared->irFreqPeakGainDb = (freqPeakLin > 1e-18)
                ? static_cast<float> (20.0 * std::log10 (freqPeakLin))
                : 0.0f;
            juce::Logger::writeToLog("[DIAG_IR_FREQ] freqPeakLin=" + juce::String (freqPeakLin, 8)
                + " irFreqPeakGainDb=" + juce::String (prepared->irFreqPeakGainDb, 2)
                + " scaleFactor=" + juce::String (prepared->scaleFactor, 6)
                + " sampleRate=" + juce::String (prepared->sampleRate, 1));
        }
    }
    catch (const std::bad_alloc&)
    {
        if (! partitionOwned)
            DIAG_MKL_FREE (data, bytes);

        return fail ("Not enough memory to prepare the IR (partition/analysis): " + irFile.getFileName());
    }
    catch (const std::exception& e)
    {
        if (! partitionOwned)
            DIAG_MKL_FREE (data, bytes);

        return fail ("Failed to prepare the IR: " + irFile.getFileName() + " (" + e.what() + ")");
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
    double /*sampleRate - future use for frequency weighting*/) noexcept
{
    return IRAnalyzer::estimateMaxFrequencyResponseGain(ir);
}
