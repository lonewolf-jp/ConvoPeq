#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "convolver/IRLoadAdmission.h"
#include "convolver/IRTrimTestHooks.h"   // ★ WORK111: test/measurement-only hook
#include "AlignedAllocation.h"
#include <mkl.h>
#include <limits>   // ★ WORK112: std::numeric_limits (subnormal 判定)

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD)

ConvolverProcessor::LoaderThread::LoaderThread(ConvolverProcessor& p, const juce::File& f, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                                 float mixedF1, float mixedF2,
                                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn)
    : Thread("IRLoader"), owner(p), weakOwner(&p), file(f), sampleRate(sr), blockSize(bs), phaseMode(phase),
    mixedTransitionStartHz(mixedF1), mixedTransitionEndHz(mixedF2),
    buildSnapshot(buildSnapshotIn), isRebuild(false)
{}

ConvolverProcessor::LoaderThread::LoaderThread(ConvolverProcessor& p, const juce::AudioBuffer<double>& src, double srcSR, double sr, int bs, ConvolverProcessor::PhaseMode phase,
                                 float mixedF1, float mixedF2, double scale,
                                 const ConvolverProcessor::BuildSnapshot& buildSnapshotIn)
    : Thread("IRRebuilder"), owner(p), weakOwner(&p), sourceIR(src), sourceSampleRate(srcSR), sampleRate(sr), blockSize(bs), phaseMode(phase),
    mixedTransitionStartHz(mixedF1), mixedTransitionEndHz(mixedF2),
    buildSnapshot(buildSnapshotIn), isRebuild(true), scaleFactor(scale)
{}

ConvolverProcessor::LoaderThread::~LoaderThread()
{
    stopThread(500);

    auto* conv = std::exchange(stepResult.newConv, nullptr);
    owner.retireStereoConvolver(conv, 0);
}

void ConvolverProcessor::LoaderThread::run()
{
    // ★ B-6: Generation check is pending implementation in the load loop.
    //   See REPAIR_PLAN section B-6 for the complete pattern.

    if (auto* provider = owner.getRcuProvider(); provider != nullptr)
        provider->getAffinityManager().applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    juce::ScopedNoDenormals noDenormals;

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    struct FlagResetter {
        ConvolverProcessor& p;
        juce::WeakReference<ConvolverProcessor> weakP;
        const juce::Thread& t;
        bool success = false;
        ~FlagResetter() {
            if (!success) {  // ← 修正: threadShouldExit 条件を削除
                auto wp = weakP;
                const bool queued = juce::MessageManager::callAsync([wp] {
                    if (auto* o = wp.get()) {
                        convo::publishAtomic(o->isLoading, false, std::memory_order_release); // release: timer/UI の isLoading acquire と HB
                        convo::publishAtomic(o->isRebuilding, false, std::memory_order_release); // release: timer/load 経路 acquire と HB
                    }
                });

                if (!queued)
                {
                    if (auto* o = wp.get())
                    {
                        // callAsync 失敗 = MessageManager が利用できない状態（未初期化/終了中/Shutdown中）。
                        // atomic 状態のみ整合性を維持する（次回ロード時のフラグ競合防止）。
                        // UI コンポーネント状態は更新されない可能性があるが、これは仕様であり、
                        // Shutdown 完了後の再初期化で UI 状態はリセットされる。
                        convo::publishAtomic(o->isLoading, false, std::memory_order_release);
                        convo::publishAtomic(o->isRebuilding, false, std::memory_order_release);
                    }
                }
            }
        }
    } resetter { owner, weakOwner, *this };

    LoadResult result = performLoad(this);

    resetter.success = (result.success || result.finalizeQueued);

    owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);

    if (!result.success && result.errorMessage.isNotEmpty() && !threadShouldExit())
    {
        auto wp = weakOwner;
        const juce::String error = result.errorMessage;
        const bool queued = juce::MessageManager::callAsync([wp, error]()
        {
            if (auto* o = wp.get())
                o->handleLoadError(error);
        });

        if (!queued)
            juce::Logger::writeToLog("LoaderThread: callAsync failed in error path; dropping UI error dispatch");
    }
}

ConvolverProcessor::LoaderThread::LoadResult ConvolverProcessor::LoaderThread::performLoad(juce::Thread* thread)
{
    std::function<bool()> savedCheck = externalCancellationCheck;
    if (thread != nullptr)
    {
        externalCancellationCheck = [thread, saved = savedCheck]() -> bool {
            if (thread->threadShouldExit()) return true;
            return saved && saved();
        };
    }

    stepCurrentThread = thread;
    stepState = StepState::LoadIR;
    stepResult = LoadResult{};
    stepTrimmed.setSize(0, 0);
    stepFileHash = 0;

    try
    {
        while (true)
        {
            const bool terminal = stepOnce();
            if (terminal) break;
        }
    }
    catch (const std::bad_alloc&)
    {
        stepResult.errorMessage = "IR too large (Out of Memory)";
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }
    catch (const std::exception& e)
    {
        stepResult.errorMessage = "Error loading IR: " + juce::String(e.what());
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }
    catch (...)
    {
        stepResult.errorMessage = "Unknown error loading IR";
        juce::Logger::writeToLog("LoaderThread: " + stepResult.errorMessage);
    }

    externalCancellationCheck = std::move(savedCheck);
    stepCurrentThread = nullptr;
    return std::move(stepResult);
}

bool ConvolverProcessor::LoaderThread::buildConvolverFromTrimmed(LoadResult& result,
                                                                  const juce::AudioBuffer<double>& trimmed,
                                                                  double sr,
                                                                  int bs,
                                                                  juce::Thread* thread)
{
    if (trimmed.getNumChannels() == 0)
        return false;

    // ★ H-02 (H02-C3): energy-centroid 估算を廃止し、canonical helper（max-abs argmax・
    //   tie-break lowest index・非有限 skip）へ一本化。従来経路の NaN→centroid→
    //   static_cast<int>(floor(NaN+.5)) UB 変換は本置換で消滅。
    const int irPeakLatency = ConvolverProcessor::measureIrPeakLatencySamples(trimmed, result.targetLength);

    auto irL = convo::makeAlignedArray<double>(static_cast<size_t>(result.targetLength));
    auto irR = convo::makeAlignedArray<double>(static_cast<size_t>(result.targetLength));

    const double* srcL = trimmed.getReadPointer(0);
    const double* srcR = (trimmed.getNumChannels() > 1) ? trimmed.getReadPointer(1) : srcL;
    std::memcpy(irL.get(), srcL, result.targetLength * sizeof(double));
    std::memcpy(irR.get(), srcR, result.targetLength * sizeof(double));

    const int internalBlockSize = juce::nextPowerOfTwo(bs);

    if (owner.isVisualizationEnabled())
    {
        result.displayIR = trimmed;
        result.displayIR.applyGain(result.scaleFactor);
    }

    if (thread == nullptr)
        return initializeConvolverSynchronously(result,
                                                std::move(irL),
                                                std::move(irR),
                                                sr,
                                                irPeakLatency,
                                                internalBlockSize,
                                                bs);

    return queueFinalizeOnMessageThread(result,
                                        std::move(irL),
                                        std::move(irR),
                                        sr,
                                        irPeakLatency,
                                        internalBlockSize,
                                        bs);
}

bool ConvolverProcessor::LoaderThread::initializeConvolverSynchronously(LoadResult& result,
                                                                         convo::ScopedAlignedPtr<double> irL,
                                                                         convo::ScopedAlignedPtr<double> irR,
                                                                         double sr,
                                                                         int irPeakLatency,
                                                                         int internalBlockSize,
                                                                         int callBlockSize)
{
    auto newConv = convo::aligned_make_unique<StereoConvolver>();

    convo::FilterSpec spec;
    spec.sampleRate = sr;
    {
        spec.hcMode = static_cast<convo::HCMode>(buildSnapshot.nucHCMode);
        spec.lcMode = static_cast<convo::LCMode>(buildSnapshot.nucLCMode);
        spec.tailMode = juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                     static_cast<int>(TailMode::Bypass),
                                     buildSnapshot.tailMode);
        spec.tailEnabled = (spec.tailMode != static_cast<int>(TailMode::Bypass));
        spec.tailStartSeconds = static_cast<double>(buildSnapshot.tailStartSec);
        spec.tailStrength = static_cast<double>(buildSnapshot.tailStrength);
        spec.tailL1L2Multiplier = buildSnapshot.tailL1L2Multiplier;
    }

    if (newConv->init(irL.release(), irR.release(), result.targetLength, sr, irPeakLatency,
                             internalBlockSize, callBlockSize, result.scaleFactor,
                             owner.getExperimentalDirectHeadEnabled(),
                             &spec, &owner))
    {
        result.newConv = newConv.release();
        result.success = true;
        return true;
    }

    result.success = false;
    result.errorMessage = "Failed to initialize NUC engine (Memory allocation or MKL setup failed).";
    return false;
}

bool ConvolverProcessor::LoaderThread::queueFinalizeOnMessageThread(LoadResult& result,
                                                                     convo::ScopedAlignedPtr<double> irL,
                                                                     convo::ScopedAlignedPtr<double> irR,
                                                                     double sr,
                                                                     int irPeakLatency,
                                                                     int internalBlockSize,
                                                                     int callBlockSize)
{
    auto loadedIRRaw = new juce::AudioBuffer<double>(std::move(result.loadedIR));
    auto displayIRRaw = new juce::AudioBuffer<double>(std::move(result.displayIR));
    auto irLRaw = irL.release();
    auto irRRaw = irR.release();

    // ★ H-02: ラムダ内で unique_ptr / ScopedAlignedPtr にラップしているため、
    // weakOwner.get() が nullptr を返してもリソースは解放される。
    // 唯一の未解放経路は JUCE シャットダウン時の MessageManager キュー破棄だが、
    // これは正常シャットダウンで許容範囲の動作である。
    const bool queued = juce::MessageManager::callAsync([weakOwner = this->weakOwner,
                                     irLRaw,
                                     irRRaw,
                                     loadedIRRaw,
                                     displayIRRaw,
                                     length = result.targetLength,
                                     sr,
                                     peak = irPeakLatency,
                                     known = internalBlockSize,
                                     callQ = callBlockSize,
                                     isReb = isRebuild,
                                     file = file,
                                     buildSnapshot = this->buildSnapshot,
                                     scale = result.scaleFactor]()
    {
        convo::ScopedAlignedPtr<double> irLHolder(irLRaw);
        convo::ScopedAlignedPtr<double> irRHolder(irRRaw);
        std::unique_ptr<juce::AudioBuffer<double>> loadedIRHolder(loadedIRRaw);
        std::unique_ptr<juce::AudioBuffer<double>> displayIRHolder(displayIRRaw);

        if (auto* ownerPtr = weakOwner.get())
        {
            ownerPtr->finalizeNUCEngineOnMessageThread(std::move(irLHolder),
                                                       std::move(irRHolder),
                                                       length, sr, peak, known, callQ, isReb, file,
                                                       buildSnapshot,
                                                       scale, std::move(loadedIRHolder), std::move(displayIRHolder));
        }
    });

    if (!queued)
    {
        convo::aligned_free(irLRaw);
        convo::aligned_free(irRRaw);
        std::unique_ptr<juce::AudioBuffer<double>>{loadedIRRaw};  // RAII delete
        std::unique_ptr<juce::AudioBuffer<double>>{displayIRRaw}; // RAII delete

        juce::Logger::writeToLog("LoaderThread: callAsync failed, aborting IR load");
        result.errorMessage = "Internal message queue full, cannot complete IR load";

        owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);

        juce::MessageManager::callAsync([weakOwner = this->weakOwner, errorMsg = result.errorMessage]()
        {
            if (auto* ownerPtr = weakOwner.get())
                ownerPtr->handleLoadError(errorMsg);
        });

        return false;
    }

    result.finalizeQueued = true;
    return true;
}

void ConvolverProcessor::LoaderThread::runSynchronously()
{
    juce::ScopedNoDenormals noDenormals;
    LoadResult result = performLoad(nullptr);

    if (result.success)
    {
        auto* conv = std::exchange(result.newConv, nullptr);
        stepResult.newConv = nullptr;
        auto loadedIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.loadedIR));
        auto displayIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.displayIR));
        // ★ WORK105: 同期パスも engine build 時の processing quantum を刻印する。
        const int syncKnownBlock = juce::nextPowerOfTwo(std::max(blockSize, 1));
        owner.applyNewState(conv, std::move(loadedIR), result.loadedSR, result.targetLength, isRebuild, file,
                            result.scaleFactor, std::move(displayIR), syncKnownBlock, /*async=*/false);
    }
    else
    {
        if (result.newConv)
            owner.retireStereoConvolver(std::exchange(result.newConv, nullptr), 0);
    }
}

bool ConvolverProcessor::LoaderThread::stepOnce()
{
    switch (stepState)
    {
        case StepState::LoadIR:
            if (!doLoadIRStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Trim;
            return false;

        case StepState::Trim:
            if (!doTrimStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Transform;
            return false;

        case StepState::Transform:
            if (!doTransformStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Build;
            return false;

        case StepState::Build:
            if (!doBuildStep()) { stepState = StepState::Error; return true; }
            stepState = StepState::Done;
            return true;

        case StepState::Done:
        case StepState::Error:
            return true;
    }
    return true;
}

bool ConvolverProcessor::LoaderThread::doLoadIRStep()
{
    stepFileHash = 0;

    if (isRebuild)
    {
        stepResult.loadedIR = std::move(sourceIR);
        stepResult.loadedSR = sourceSampleRate;
        stepResult.scaleFactor = this->scaleFactor;
    }
    else
    {
        if (!file.existsAsFile())
        {
            stepResult.errorMessage = "IR file not found: " + file.getFullPathName();
            return false;
        }

        juce::AudioFormatManager formatManager;
        formatManager.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
        if (!reader)
        {
            stepResult.errorMessage = "Unsupported audio format or corrupted file: " + file.getFileName();
            return false;
        }

        // ★ WORK102 (big 1-8) FC-INV-9 admission ordering:
        //   FC-FORM-4 → 3 → 2 → 1 → [確保開始] → FC-FORM-6(hash) → … → FC-FORM-5(trim 後)。
        //   ここまで O(fileLength) / O(file bytes) の確保は 0 件。
        //   契約の正本: doc/work102/big18_failure_contract_arbitration_20260917.md（R2 凍結）。
        const int64 fileLength = reader->lengthInSamples;
        const unsigned rawChannels = reader->numChannels; // narrowing 前の unsigned で判定（IG §4.4）

        // FC-FORM-4: degenerate input bound
        if (!convo::irload::admitChannelCountNonZero(rawChannels))
        {
            stepResult.errorMessage = convo::irload::diagnosticChannelLimit(rawChannels);
            return false;
        }
        if (!convo::irload::admitFileLengthNonZero(fileLength))
        {
            stepResult.errorMessage = convo::irload::diagnosticLengthLimit(fileLength);
            return false;
        }

        // FC-FORM-3: INT32 representation bound（AudioBuffer::setSize(int) への narrowing 契約）
        if (!convo::irload::admitFileLengthRepresentable(fileLength))
        {
            stepResult.errorMessage = convo::irload::diagnosticLengthLimit(fileLength);
            return false;
        }

        // FC-FORM-2: channel bound（N ≤ 8 では SR-01B の「>2ch→先頭2」を維持）
        if (!convo::irload::admitChannelCount(rawChannels))
        {
            stepResult.errorMessage = convo::irload::diagnosticChannelLimit(rawChannels);
            return false;
        }

        // FC-FORM-1: byte bound（除算形。未検証値の乗算を行わない）
        if (!convo::irload::admitByteBudget(rawChannels, fileLength))
        {
            stepResult.errorMessage = convo::irload::diagnosticByteLimit(rawChannels, fileLength);
            return false;
        }

        const int numChannels = static_cast<int>(rawChannels); // 値域は上の FC-FORM-2/4 で保証

        // FC-FORM-6: hash は admission の後（FC-INV-1）。O(1) メモリのストリーミング実装。
        stepFileHash = convo::AllpassDesigner::computeIRHash(file);

        // ★ work92 B-5 (big 1-8 / R-新規C): ストリーミング読込。
        //   旧実装は fileLength 分を一括確保（ステレオ float ~16GB / double ~32GB の
        //   確保試行 → OOM）。チャンク読込に変更し、常時メモリはチャンク分のみ。
        //   G4 narrowing proof（PLAN v3 §3-B-5）:
        //   - offset は int64 のまま reader->read の startSampleInFile に渡す
        //     （juce_AudioFormatReader.h:282 — int64 契約・static_cast<int> 禁止）
        //   - ループ不変式: offset + chunk ≤ fileLength ≤ INT32_MAX（int64 算術）により
        //     copyFrom の int 位置 narrowing は値域証明済み
        //   - chunk ≤ kStreamChunk = 256*1024 で自明に int 域内
        constexpr int64 kStreamChunk = 256 * 1024;
        juce::AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(kStreamChunk));
        auto tempAligned = convo::makeAlignedArray<double>(static_cast<size_t>(kStreamChunk));
        if (!tempAligned)
        {
            stepResult.errorMessage = "Failed to allocate temporary buffer for IR loading.";
            return false;
        }

        stepResult.loadedIR.setSize(numChannels, static_cast<int>(fileLength));
        stepResult.loadedIR.clear();

        for (int64 offset = 0; offset < fileLength; offset += kStreamChunk)
        {
            if (externalCancellationCheck && externalCancellationCheck())
            {
                stepResult.errorMessage = "IR loading cancelled.";
                return false;
            }

            const int64 remaining = fileLength - offset; // ループ不変: remaining > 0
            const int chunk = static_cast<int>(std::min<int64>(kStreamChunk, remaining));
            jassert(offset + chunk <= 2147483647); // ★ G4: narrowing 前の belt-and-braces

            if (!reader->read(&tempFloatBuffer, 0, chunk, offset, true, true))
            {
                stepResult.errorMessage = "Failed to read audio data from file.";
                return false;
            }

            for (int ch = 0; ch < numChannels; ++ch)
            {
                const float* src = tempFloatBuffer.getReadPointer(ch);
                convo::input_transform::convertFloatToDoubleHighQuality(
                    src, tempAligned.get(), chunk);
                stepResult.loadedIR.copyFrom(ch, static_cast<int>(offset), tempAligned.get(), chunk);
            }
        }
        stepResult.loadedSR = reader->sampleRate;
    }

    return (stepResult.loadedIR.getNumSamples() > 0 && stepResult.loadedIR.getNumChannels() > 0);
}

// ★ WORK112 112-1: transform chain checkpoint（NonRT 観測のみ・値の変更なし）
static void printChainCheckpoint(const char* tag, const juce::AudioBuffer<double>& buf, double sr)
{
    if (buf.getNumSamples() <= 0 || buf.getNumChannels() <= 0 || sr <= 0.0)
    {
        std::fprintf(stderr, "[IR_CHAIN] %s empty\n", tag);
        return;
    }
    const int n = buf.getNumSamples();
    const double* p = buf.getReadPointer(0);
    double peak = 0.0, sum2 = 0.0, dc = 0.0;
    long long nonfinite = 0, subnormal = 0;
    const double subMin = std::numeric_limits<double>::min();
    for (int i = 0; i < n; ++i)
    {
        const double v = p[i];
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        const double a = std::abs(v);
        if (a > peak) peak = a;
        sum2 += v * v;
        dc += v;
        if (a > 0.0 && a < subMin) ++subnormal;
    }
    const double rms = std::sqrt(sum2 / static_cast<double>(std::max(1, n)));
    dc /= static_cast<double>(std::max(1, n));

    // 低域 band energy（1-pole LP の累積差分。0-20 / 20-50 / 50-100 / 100-200 Hz）
    const double fcs[4] = { 20.0, 50.0, 100.0, 200.0 };
    double al[4], y[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
    for (int k = 0; k < 4; ++k)
        al[k] = std::exp(-2.0 * 3.14159265358979323846 * fcs[k] / sr);
    for (int i = 0; i < n; ++i)
    {
        const double v = p[i];
        for (int k = 0; k < 4; ++k)
        {
            y[k] = (1.0 - al[k]) * v + al[k] * y[k];
            e[k] += y[k] * y[k];
        }
    }
    std::fprintf(stderr,
        "[IR_CHAIN] %s n=%d sr=%.0f peak=%.8e rms=%.8e dc=%.8e energy=%.8e "
        "nonfinite=%lld subnormal=%lld lf[0-20]=%.6e [20-50]=%.6e [50-100]=%.6e [100-200]=%.6e\n",
        tag, n, sr, peak, rms, dc, sum2, nonfinite, subnormal,
        e[0], e[1] - e[0], e[2] - e[1], e[3] - e[2]);
}

bool ConvolverProcessor::LoaderThread::doTrimStep()
{
    auto shouldStop = [this]() -> bool {
        return externalCancellationCheck && externalCancellationCheck();
    };

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    if (stepResult.loadedIR.getNumSamples() > 0)
    {
        const int numSamples = stepResult.loadedIR.getNumSamples();
        const int numChannels = stepResult.loadedIR.getNumChannels();
        const double threshold = 1.0e-15;

        int newLength = 0;
        if (numChannels > 0)
        {
            const double* ch0 = stepResult.loadedIR.getReadPointer(0);
            const double* ch1 = (numChannels > 1) ? stepResult.loadedIR.getReadPointer(1) : nullptr;

            const __m256d vThreshold = _mm256_set1_pd(threshold);
            const __m256d vSignMask = _mm256_set1_pd(-0.0);

            int i = numSamples;
            bool found = false;
            for (; i >= 4; i -= 4)
            {
                __m256d v0 = _mm256_loadu_pd(ch0 + i - 4);
                __m256d abs0 = _mm256_andnot_pd(vSignMask, v0);
                __m256d mask = _mm256_cmp_pd(abs0, vThreshold, _CMP_GT_OQ);
                if (ch1)
                {
                    __m256d v1 = _mm256_loadu_pd(ch1 + i - 4);
                    __m256d abs1 = _mm256_andnot_pd(vSignMask, v1);
                    mask = _mm256_or_pd(mask, _mm256_cmp_pd(abs1, vThreshold, _CMP_GT_OQ));
                }
                if (_mm256_testz_pd(mask, mask) == 0)
                {
                    for (int j = i - 1; j >= i - 4; --j)
                    {
                        if (std::abs(ch0[j]) > threshold || (ch1 && std::abs(ch1[j]) > threshold))
                        { newLength = j + 1; found = true; break; }
                    }
                    if (found) break;
                }
            }
            if (!found)
            {
                for (int j = i - 1; j >= 0; --j)
                {
                    if (std::abs(ch0[j]) > threshold || (ch1 && std::abs(ch1[j]) > threshold))
                    { newLength = j + 1; break; }
                }
            }
        }
        if (newLength < numSamples)
        {
            stepResult.loadedIR.setSize(numChannels, std::max(1, newLength), true);
            ConvolverProcessorInternal::shrinkToFit(stepResult.loadedIR);
        }
    }

    // ★ WORK102 FC-FORM-5: resample 出力長の admission。
    //   raw fileLength ではなく **trim 後** の長さに対して評価する（R2-D4）。
    //   raw に適用すると、長い無音テールを持つファイルを
    //   「UI は実効長で受理するが loader は拒否する」という回帰になる。
    //   この判定は resampler 構築（r8b getMaxOutLen の (int) 変換）より前でなければならない（FC-INV-10）。
    {
        const int64 trimmedLength = stepResult.loadedIR.getNumSamples();
        if (!convo::irload::admitResampleOutput(trimmedLength, stepResult.loadedSR, sampleRate))
        {
            stepResult.errorMessage = convo::irload::diagnosticResampleLimit(trimmedLength,
                                                                             stepResult.loadedSR,
                                                                             sampleRate);
            return false;
        }
    }

    if (stepResult.loadedSR > 0.0 && sampleRate > 0.0 &&
        std::abs(stepResult.loadedSR - sampleRate) > 1e-6)
    {
        const uint64_t myGen = owner.convolverStateGeneration.getCurrentGeneration();
        const r8b::EDSPFilterPhaseResponse r8bPhase =
            (owner.getResamplingPhaseMode() == ResamplingPhaseMode::Linear)
                ? r8b::fprLinearPhase : r8b::fprMinPhase;

        auto resampleOut = ConvolverProcessorInternal::resampleIR(
            stepResult.loadedIR, stepResult.loadedSR, sampleRate, r8bPhase,
            [&]() -> bool {
                return shouldStop() ||
                       !owner.convolverStateGeneration.isCurrentGeneration(myGen);
            });

        if (!owner.convolverStateGeneration.isCurrentGeneration(myGen))
            return false;

        switch (resampleOut.result)
        {
            case ConvolverProcessorInternal::ResampleResult::Success:
                stepResult.loadedIR = std::move(resampleOut.buffer);
                stepResult.loadedSR = sampleRate;
                break;
            case ConvolverProcessorInternal::ResampleResult::Cancelled:
                return false;
            case ConvolverProcessorInternal::ResampleResult::SilentIR:
                stepResult.errorMessage = "IR is silent (all samples near zero).";
                return false;
            case ConvolverProcessorInternal::ResampleResult::Error:
            default:
                stepResult.errorMessage = "Resampling failed (unknown error).";
                return false;
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    printChainCheckpoint("A_resampled", stepResult.loadedIR, stepResult.loadedSR);

    if (stepResult.loadedSR > 0.0 && stepResult.loadedIR.getNumSamples() > 0)
    {
        for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
        {
            convo::UltraHighRateDCBlocker dcBlocker;
            dcBlocker.init(stepResult.loadedSR, 1.0);
            double* data = stepResult.loadedIR.getWritePointer(ch);
            dcBlocker.process(data, stepResult.loadedIR.getNumSamples());
        }
    }

    printChainCheckpoint("B_dcblock", stepResult.loadedIR, stepResult.loadedSR);

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    if (stepResult.loadedIR.getNumSamples() > 0)
    {
        const int numSamples = stepResult.loadedIR.getNumSamples();
        for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
        {
            if (!ConvolverProcessorInternal::applyAsymmetricTukey(stepResult.loadedIR.getWritePointer(ch), numSamples))
            {
                stepResult.errorMessage = "Failed to allocate Tukey window buffer (Out of Memory).";
                return false;
            }
        }
    }

    printChainCheckpoint("C_tukey", stepResult.loadedIR, stepResult.loadedSR);

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    stepResult.targetLength = owner.computeTargetIRLength(stepResult.loadedSR, stepResult.loadedIR.getNumSamples());
    stepTrimmed.setSize(stepResult.loadedIR.getNumChannels(), stepResult.targetLength);
    stepTrimmed.clear();

    const int copySamples = std::min(stepResult.targetLength, stepResult.loadedIR.getNumSamples());
    constexpr int minFadeSamples = 256;
    constexpr double fadeRatio = 0.02;
    const int maxFadeSamples = juce::jmax(minFadeSamples, static_cast<int>(std::round(sampleRate * 0.080)));
    int fadeSamples = static_cast<int>(std::round(static_cast<double>(copySamples) * fadeRatio));
    fadeSamples = juce::jlimit(minFadeSamples, maxFadeSamples, fadeSamples);
    fadeSamples = juce::jmax(0, juce::jmin(fadeSamples, copySamples - 1));

    // ★ WORK111: 末尾 fade は production では常に適用。measurement-only hook が true のときのみ skip。
    //   NonRT (doTrimStep) 専用。RT は一切参照しない。
    const bool fadeDisabled =
        convo::trimtest::disableTailFadeForMeasurement().load(std::memory_order_relaxed);

    for (int ch = 0; ch < stepResult.loadedIR.getNumChannels(); ++ch)
    {
        stepTrimmed.copyFrom(ch, 0, stepResult.loadedIR, ch, 0, copySamples);
        if (fadeSamples > 0 && !fadeDisabled)
            stepTrimmed.applyGainRamp(ch, copySamples - fadeSamples, fadeSamples, 1.0, 0.0);
    }

    printChainCheckpoint("D_trimfade", stepTrimmed, stepResult.loadedSR);

    // ★ WORK111: 末尾 fade geometry / envelope / energy（NonRT trace のみ・出力は stderr）
    if (stepResult.loadedSR > 0.0 && copySamples > 0
        && stepResult.loadedIR.getNumChannels() > 0)
    {
        const int fadeStart = juce::jmax(0, copySamples - fadeSamples);
        const int fadeEnd   = copySamples;
        const uint64_t genDbg =
            static_cast<unsigned long long>(owner.convolverStateGeneration.getCurrentGeneration());

        std::fprintf(stderr,
            "[IR_TAIL_GEOM] gen=%llu loadedSr=%.0f loadedLen=%d targetLength=%d copySamples=%d "
            "fadeSamples=%d fadeStart=%d fadeEnd=%d fadeMs=%.4f fadeDisabled=%d\n",
            static_cast<unsigned long long>(genDbg), stepResult.loadedSR,
            stepResult.loadedIR.getNumSamples(), stepResult.targetLength,
            copySamples, fadeSamples, fadeStart, fadeEnd,
            1000.0 * static_cast<double>(fadeSamples) / stepResult.loadedSR,
            fadeDisabled ? 1 : 0);

        const auto* pre  = stepResult.loadedIR.getReadPointer(0); // Tukey 後 / fade 前
        const auto* post = stepTrimmed.getReadPointer(0);         // fade 後
        const int pts[8] = { fadeStart - 1024, fadeStart - 512, fadeStart,
                             fadeStart + 128, fadeStart + 256, fadeStart + 512,
                             fadeStart + 1024, fadeEnd - 1 };
        for (int t = 0; t < 8; ++t)
        {
            const int i = pts[t];
            if (i < 0 || i >= copySamples) continue;
            const double a = static_cast<double>(pre[i]);
            const double b = static_cast<double>(post[i]);
            std::fprintf(stderr,
                "[IR_TAIL_ENV] i=%d tMs=%.4f afterTukey=%.8e afterFade=%.8e gain=%.6f\n",
                i, 1000.0 * static_cast<double>(i) / stepResult.loadedSR, a, b,
                (std::abs(a) > 1.0e-30) ? b / a : 0.0);
        }

        // energy split（全帯域）+ 1-pole 100Hz LP 相当の低域 energy
        double eTot = 0.0, ePre = 0.0, eBand = 0.0;
        double lpTot = 0.0, lpPre = 0.0, lpBand = 0.0;
        const double aLp = std::exp(-2.0 * 3.14159265358979323846 * 100.0 / stepResult.loadedSR);
        double y = 0.0;
        for (int i = 0; i < copySamples; ++i)
        {
            const double v = static_cast<double>(pre[i]);
            const double s = v * v;
            eTot += s;
            if (i < fadeStart) ePre += s; else eBand += s;
            y = (1.0 - aLp) * v + aLp * y;
            const double sy = y * y;
            lpTot += sy;
            if (i < fadeStart) lpPre += sy; else lpBand += sy;
        }
        std::fprintf(stderr,
            "[IR_TAIL_ENERGY] gen=%llu total=%.8e preFade=%.8e fadeBand=%.8e bandFrac=%.6f "
            "lp100Total=%.8e lp100Pre=%.8e lp100Band=%.8e lp100BandFrac=%.6f\n",
            static_cast<unsigned long long>(genDbg), eTot, ePre, eBand,
            (eTot > 1.0e-30) ? eBand / eTot : 0.0,
            lpTot, lpPre, lpBand, (lpTot > 1.0e-30) ? lpBand / lpTot : 0.0);
    }

#if defined(__AVX2__)
    _mm256_zeroupper();
#endif
    return true;
}

bool ConvolverProcessor::LoaderThread::doTransformStep()
{
    auto shouldStop = [this]() -> bool {
        return externalCancellationCheck && externalCancellationCheck();
    };

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    auto validateBuffer = [](const juce::AudioBuffer<double>& buf) -> bool
    {
        if (buf.getNumSamples() <= 0 || buf.getNumChannels() <= 0) return false;
        double maxAbs = 0.0;
        for (int ch = 0; ch < buf.getNumChannels(); ++ch)
        {
            const double* ptr = buf.getReadPointer(ch);
            for (int i = 0; i < buf.getNumSamples(); ++i)
            {
                if (!std::isfinite(ptr[i])) return false;
                maxAbs = std::max(maxAbs, std::abs(ptr[i]));
            }
        }
        return maxAbs > 1.0e-12;
    };

    if (phaseMode == ConvolverProcessor::PhaseMode::Minimum ||
        phaseMode == ConvolverProcessor::PhaseMode::Mixed)
    {
        bool wasCancelled = false;
        auto minPhaseIR = ConvolverProcessorInternal::convertToMinimumPhase(stepTrimmed, shouldStop, &wasCancelled);
        if (wasCancelled) return false;

        if (validateBuffer(minPhaseIR))
        {
            if (phaseMode == ConvolverProcessor::PhaseMode::Minimum)
            {
                stepTrimmed = std::move(minPhaseIR);
            }
            else
            {
                bool mixedCancelled = false;
                auto progressCb = [this](float p) { owner.setLoadingProgress(p); };
                auto mixedIR = convertToMixedPhase(&owner, stepFileHash, stepTrimmed, minPhaseIR,
                                                   sampleRate,
                                                   static_cast<double>(mixedTransitionStartHz),
                                                   static_cast<double>(mixedTransitionEndHz),
                                                   32.0,  // tau (dummy, unused in DSP)
                                                   shouldStop, &mixedCancelled, progressCb);
                if (mixedCancelled) return false;
                if (validateBuffer(mixedIR))
                    stepTrimmed = std::move(mixedIR);
            }
        }
    }

    if (ConvolverProcessorInternal::checkCancellation(shouldStop, nullptr)) return false;

    {
        const IRState* currentState = owner.acquireIRState();
        auto currentIr = (currentState != nullptr) ? currentState->ir : nullptr;
        const double currentScale = convo::consumeAtomic(owner.currentIRScale, std::memory_order_acquire); // acquire: applyNewState の publishAtomic release と HB
        const auto scaleInfo = IRConverter::computeScaleFactor(stepTrimmed, currentIr, currentScale);
        owner.releaseIRState(currentState);

        stepResult.scaleFactor = scaleInfo.hasScaleFactor ? scaleInfo.scaleFactor : 1.0;
    }

    printChainCheckpoint("E_phase", stepTrimmed, sampleRate);
    std::fprintf(stderr, "[IR_CHAIN] F_scale scaleFactor=%.8f phaseMode=%d (scale は engine 内の周波数領域で乗算)\n",
                 stepResult.scaleFactor, static_cast<int>(phaseMode));

    return true;
}

bool ConvolverProcessor::LoaderThread::doBuildStep()
{
    return buildConvolverFromTrimmed(stepResult, stepTrimmed, sampleRate, blockSize, stepCurrentThread);
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD
