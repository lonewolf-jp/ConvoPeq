#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "CacheManager.h"
#include "ProgressiveUpgradeThread.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/ThreadAffinityManager.h"
#include "AlignedAllocation.h"
#include <mkl.h>

#include "audioengine/AtomicAccess.h"

// ★ work70: StereoConvolver::liveCount 静的定義
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint32_t> ConvolverProcessor::StereoConvolver::liveCount { 0 };
#endif

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE)

const ConvolverProcessor::IRState* ConvolverProcessor::acquireIRState() const noexcept
{
    return convo::consumeAtomic(currentIRState, std::memory_order_acquire); // acquire: updateIRState/releaseResources の exchangeAtomic acq_rel と HB
}

void ConvolverProcessor::releaseIRState(const IRState* /*state*/) const noexcept
{
    // IRState lifetime is managed by deferred retirement.
}

void ConvolverProcessor::updateIRState(const juce::AudioBuffer<double>& newIR, double newSR, float additionalAttenuationDb, float irFreqPeakGainDb)
{
    auto uniqueIR = std::make_unique<juce::AudioBuffer<double>>(newIR);

    auto newState = convo::aligned_make_unique<IRState>();
    newState->irOwner = std::move(uniqueIR);
    newState->ir = newState->irOwner.get();
    newState->sampleRate = newSR;
    newState->additionalAttenuationDb = additionalAttenuationDb;
    newState->irFreqPeakGainDb = irFreqPeakGainDb;
    if (auto* provider = getRcuProvider(); provider != nullptr)
        newState->generation = provider->snapshotRcuEpoch();
    else
        newState->generation = 1;
    std::atomic_thread_fence(std::memory_order_release); // release: 直後の currentIRState 交換公開前に newState 初期化を順序化

    auto* oldState = convo::exchangeAtomic(currentIRState, newState.release(), std::memory_order_acq_rel); // acq_rel: acquire で旧ポインタ観測と HB; release で acquireIRState/load 側と HB
    if (oldState != nullptr)
    {
        auto deleter = [](void* p)
        {
            auto* state = static_cast<IRState*>(p);
            state->~IRState();
            mkl_free(state);
        };

        if (auto* provider = getRcuProvider(); provider != nullptr)
            provider->enqueueDeferredDeleteNonRt(oldState, deleter);
        else
            deleter(oldState);
    }
}

void ConvolverProcessor::StereoConvolver::retireStereoConvolver(StereoConvolver* sc, AudioEngine* provider) noexcept
{
    if (!sc || convo::exchangeAtomic(sc->retired, true, std::memory_order_acq_rel)) // acq_rel: 二重 retire 防止の可視化を双方向で保証
        return;

    if (provider != nullptr)
    {
        provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver);
        return;
    }

    destroyStereoConvolver(sc);
}

void ConvolverProcessor::retireStereoConvolver(StereoConvolver* conv, uint64_t /*retireEpoch*/)
{
    StereoConvolver::retireStereoConvolver(conv, getRcuProvider());
}

// ────────────────────────────────────────────────────────────────
// Constructor
// ────────────────────────────────────────────────────────────────
ConvolverProcessor::ConvolverProcessor()
    : mixSmoother(1.0)
{
    irConverter = convo::aligned_make_unique<IRConverter>();

    cacheManager = convo::aligned_make_unique<CacheManager>();
    cacheManager->setSafeDeleteChecker([this](uint64_t key, int fftSize)
    {
        return this->isCacheEntrySafeToDelete(key, fftSize);
    });

    updateLatencyCache();
    debugCheckAtomicLockFree();
}

// ────────────────────────────────────────────────────────────────
// Destructor
// ────────────────────────────────────────────────────────────────
ConvolverProcessor::~ConvolverProcessor()
{
    stopUpgradeThread();
    stopTimer();
    forceCleanup();
    // スレッドを停止
    activeLoader.reset();

    // Destructor runs after AudioEngine destructor body.
    // Do not enqueue to global deferred queue here because final reclaim may have already happened.
    auto* oldConv = exchangeActiveEngine(nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 active engine 取得; release で null 公開
    // NOTE:
    // Destructor phase must not depend on rcuProvider/AudioEngine lifetime.
    // Force local retirement path to avoid touching potentially destroyed owner state.
    StereoConvolver::retireStereoConvolver(oldConv, nullptr);

    auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 IRState 取得; release で null 公開
    if (oldIrState != nullptr)
    {
        oldIrState->~IRState();
        mkl_free(oldIrState);
    }

    // Clean up latency snapshot pointer
    auto* oldSnap = convo::exchangeAtomic(cachedLatency, nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 snapshot 取得; release で null 公開
    std::unique_ptr<LatencySnapshot> owned{oldSnap}; // RAII delete

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
    }

    // Note: final deferred reclaim is owned by AudioEngine shutdown sequence.
}

// ────────────────────────────────────────────────────────────────
// Timer Callback
// ────────────────────────────────────────────────────────────────
void ConvolverProcessor::timerCallback()
{
    const int currentClampCount = convo::consumeAtomic(latencyClampCounter(), std::memory_order_acquire); // acquire: Runtime 側 fetchAddAtomic acq_rel と HB
    if (currentClampCount != lastReportedClampCount_)
    {
        juce::Logger::writeToLog("ConvolverProcessor: Latency clamp triggered (total: "
                                 + juce::String(currentClampCount) + " times)");
        lastReportedClampCount_ = currentClampCount;
    }

    // ── リングバッファオーバーフロー診断 (NUC ringOverflowCount の確認) ──
    {
        auto* conv = loadActiveEngine(std::memory_order_acquire); // acquire: exchangeActiveEngine acq_rel/release と HB
        if (conv != nullptr)
        {
            for (int ch = 0; ch < 2; ++ch)
            {
                if (conv->nucConvolvers[ch] != nullptr)
                {
                    const int ov = conv->nucConvolvers[ch]->getRingOverflowCount();
                    if (ov > 0)
                    {
                        juce::Logger::writeToLog("ConvolverProcessor: NUC ring overflow detected (ch="
                                                 + juce::String(ch) + ", count=" + juce::String(ov) + ")");
                        conv->nucConvolvers[ch]->resetRingOverflowCount();
                    }
                }
            }
        }
    }

    // ★ リングバッファオーバーフローによるリビルド要求を処理 (Audio Thread からは呼ばれない)
    if (convo::consumeAtomic(rebuildPendingAfterLoad, std::memory_order_acquire)) // acquire: LoadPipeline 側 publishAtomic/exchangeAtomic と HB
    {
        if (!convo::consumeAtomic(isLoading, std::memory_order_acquire) &&      // acquire: load/rebuild 側 publishAtomic release と HB
            !convo::consumeAtomic(isRebuilding, std::memory_order_acquire))      // acquire: load/rebuild 側 publishAtomic release と HB
        {
            juce::File irFile;
            {
                const juce::ScopedLock sl(irFileLock);
                irFile = currentIrFile;
            }
            if (irFile.existsAsFile())
            {
                convo::publishAtomic(rebuildPendingAfterLoad, false, std::memory_order_release); // release: timer/loader 側 acquire と HB
                loadImpulseResponse(irFile, false);
            }
        }
    }

    auto* provider = getRcuProvider();
    if (provider)
        provider->tryReclaimResources();

    cleanup();
}

// ────────────────────────────────────────────────────────────────
// prepareToPlay
// ────────────────────────────────────────────────────────────────
void ConvolverProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(*this);

    convo::publishAtomic(isPrepared, false, std::memory_order_release); // release: Runtime 側 isPrepared acquire と HB

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
        fftHandleSize = 0;
    }

    const bool rateChanged = (std::abs(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire) - sampleRate) > 1e-6); // acquire: 先行 publishAtomic release と HB
    const bool blockChanged = (convo::consumeAtomic(currentBufferSize, std::memory_order_acquire) != samplesPerBlock);           // acquire: 先行 publishAtomic release と HB

    convo::publishAtomic(currentBufferSize, samplesPerBlock, std::memory_order_release); // release: loader/runtime 側 acquire と HB
    convo::publishAtomic(currentSampleRate, sampleRate, std::memory_order_release);      // release: loader/runtime 側 acquire と HB

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(MAX_BLOCK_SIZE);
    spec.numChannels = 2;

    currentSpec = spec;

    auto* conv = loadActiveEngine(std::memory_order_acquire); // acquire: exchangeActiveEngine acq_rel/release と HB
    if (conv) {
        const int internalBlockSize = juce::nextPowerOfTwo(samplesPerBlock);

        if ((rateChanged || blockChanged) && conv->irDataLength > 0)
        {
            // ★ [M-1] 責務: この分岐は IR サンプル配列をリサンプリングせず再利用するため、
            // 生成された engine は未完成状態（IR 時間軸が processingRate 変化に対応していない）。
            // 呼び出し元（rebuildThreadLoop）はこの直後に必ず rebuildAllIRsSynchronous() を呼び、
            // ソース IR（IRState）から正しいリサンプリングを行って engine を完成させること。
            // この engine を単独で commit / publish してはならない。
            StereoConvolver* newConv = nullptr;
            try
            {
                auto newConvHolder = convo::aligned_make_unique<StereoConvolver>();
                newConv = newConvHolder.get();
                const BuildSnapshot buildSnapshot = captureBuildSnapshot();

                auto irL = convo::makeAlignedArray<double>(static_cast<size_t>(conv->irDataLength));
                auto irR = convo::makeAlignedArray<double>(static_cast<size_t>(conv->irDataLength));
                std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
                std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));

                convo::FilterSpec tailSpec;
                tailSpec.sampleRate = sampleRate;
                {
                    tailSpec.hcMode = static_cast<convo::HCMode>(buildSnapshot.nucHCMode);
                    tailSpec.lcMode = static_cast<convo::LCMode>(buildSnapshot.nucLCMode);
                    tailSpec.tailMode = juce::jlimit(static_cast<int>(TailMode::AirAbsorption),
                                                     static_cast<int>(TailMode::Bypass),
                                                     buildSnapshot.tailMode);
                    tailSpec.tailEnabled = (tailSpec.tailMode != static_cast<int>(TailMode::Bypass));
                    tailSpec.tailStartSeconds = static_cast<double>(buildSnapshot.tailStartSec);
                    tailSpec.tailStrength = static_cast<double>(buildSnapshot.tailStrength);
                    tailSpec.tailL1L2Multiplier = buildSnapshot.tailL1L2Multiplier;
                }

                if (newConv->init(irL.release(), irR.release(),
                                  conv->irDataLength, sampleRate, conv->irLatency, internalBlockSize, samplesPerBlock, conv->storedScale,
                                  getExperimentalDirectHeadEnabled(),
                                  &tailSpec, this))
                {
                    newConv = newConvHolder.release();
                    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->snapshotRcuEpoch() : 1;
                    auto* oldConv = exchangeActiveEngine(newConv, std::memory_order_acq_rel); // acq_rel: acquire で旧 engine 取得; release で新 engine 公開
                    if (oldConv)
                        retireStereoConvolver(oldConv, retireEpoch);
                    updateLatencyCache();
                }
                else
                {
                    juce::Logger::writeToLog("ConvolverProcessor::prepareToPlay: NUC re-init failed (MKL alloc?). Keeping existing engine.");
                }
            }
            catch (const std::bad_alloc&)
            {
                juce::Logger::writeToLog("ConvolverProcessor::prepareToPlay: NUC re-init failed (std::bad_alloc). Keeping existing engine.");
            }
        }
    }

    // DelayLine準備
    if (delayBufferCapacity < DELAY_BUFFER_SIZE)
    {
        auto newL = convo::makeAlignedArray<double>(static_cast<size_t>(DELAY_BUFFER_SIZE));
        auto newR = convo::makeAlignedArray<double>(static_cast<size_t>(DELAY_BUFFER_SIZE));
        delayBuffer[0] = std::move(newL);
        delayBuffer[1] = std::move(newR);
        delayBufferCapacity = DELAY_BUFFER_SIZE;
    }
    juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    // Dry/Wet/Smoothing/Oldバッファ確保 (まとめて処理)
    auto allocateIfNeeded = [](convo::ScopedAlignedPtr<double>* storage, int& capacity, const char* name) {
        if (capacity < MAX_BLOCK_SIZE)
        {
            auto newL = convo::makeAlignedArray<double>(static_cast<size_t>(MAX_BLOCK_SIZE));
            auto newR = convo::makeAlignedArray<double>(static_cast<size_t>(MAX_BLOCK_SIZE));
            storage[0] = std::move(newL);
            storage[1] = std::move(newR);
            capacity = MAX_BLOCK_SIZE;
        }
        return true;
    };

    if (!allocateIfNeeded(dryBufferStorage, dryBufferCapacity, "dry buffers")) return;
    if (!allocateIfNeeded(smoothingBufferStorage, smoothingBufferCapacity, "smoothing buffers")) return;
    if (!allocateIfNeeded(oldDryBufferStorage, oldDryBufferCapacity, "old dry buffers")) return;

    // Set buffer references
    double* dryChs[2] = { dryBufferStorage[0].get(), dryBufferStorage[1].get() };
    dryBuffer.setDataToReferTo(dryChs, 2, MAX_BLOCK_SIZE);
    dryBuffer.clear();

    double* smoothChs[2] = { smoothingBufferStorage[0].get(), smoothingBufferStorage[1].get() };
    smoothingBuffer.setDataToReferTo(smoothChs, 2, MAX_BLOCK_SIZE);
    smoothingBuffer.clear();

    double* oldDryChs[2] = { oldDryBufferStorage[0].get(), oldDryBufferStorage[1].get() };
    oldDryBuffer.setDataToReferTo(oldDryChs, 2, MAX_BLOCK_SIZE);
    oldDryBuffer.clear();

    // Wet buffers (simple array, not referable buffer)
    if (wetBufferCapacity < MAX_BLOCK_SIZE)
    {
        auto newL = convo::makeAlignedArray<double>(static_cast<size_t>(MAX_BLOCK_SIZE));
        auto newR = convo::makeAlignedArray<double>(static_cast<size_t>(MAX_BLOCK_SIZE));
        wetBufferStorage[0] = std::move(newL);
        wetBufferStorage[1] = std::move(newR);
        wetBufferCapacity = MAX_BLOCK_SIZE;
    }
    juce::FloatVectorOperations::clear(wetBufferStorage[0].get(), MAX_BLOCK_SIZE);

    // ★ M-05: delayFadeRamp バッファ確保 (prepareToPlayで事前確保)
    {
        const int neededFadeSamples = MAX_BLOCK_SIZE;
        if (delayFadeRampCapacity < neededFadeSamples)
        {
            delayFadeRampBuffer.reset(
                static_cast<double*>(convo::aligned_malloc(
                    static_cast<size_t>(neededFadeSamples) * sizeof(double), 64)));
            delayFadeRampCapacity = (delayFadeRampBuffer.get() != nullptr) ? neededFadeSamples : 0;
        }
    }
    juce::FloatVectorOperations::clear(wetBufferStorage[1].get(), MAX_BLOCK_SIZE);

    // スムージング時間の設定 (H3: pendingOverride が唯一の Source of Truth)
    {
        const juce::ScopedLock lock(pendingOverrideLock);
        mixSmoother.reset(sampleRate, static_cast<double>(pendingOverride.smoothingTimeSec));
        mixSmoother.setCurrentAndTargetValue(static_cast<double>(pendingOverride.mix));
    }

    // レイテンシー補正の初期化
    latencySmoother.reset(sampleRate, 0.1);
    crossfadeGain.reset(sampleRate, 0.02);
    crossfadeGain.setCurrentAndTargetValue(1.0);

    if (conv)
    {
        const int initialLatency = juce::jmin(conv->latency + conv->irLatency, MAX_TOTAL_DELAY);
        latencySmoother.setCurrentAndTargetValue(static_cast<double>(initialLatency));
    }
    else
    {
        latencySmoother.setCurrentAndTargetValue(0.0);
    }
    oldDelay = latencySmoother.getTargetValue();

    if (!deferredFreeThread)
    {
        ThreadAffinityManager* affinityMgr = nullptr;
        if (auto* provider = getRcuProvider(); provider != nullptr)
            affinityMgr = &provider->getAffinityManager();

        deferredFreeThread = convo::aligned_make_unique<convo::DeferredFreeThread>(rcuSwapper, affinityMgr);
    }

    // 世代カウンターをインクリメント (NonRT → RT 通知)
    convo::fetchAddAtomic(firstProcessCallGen, static_cast<uint64_t>(1), std::memory_order_acq_rel); // acq_rel: Runtime 側 acquire と HB

    convo::publishAtomic(isPrepared, true, std::memory_order_release); // release: Runtime 側 isPrepared acquire と HB
    updateLatencyCache();
    requestHostDisplayUpdate();
}

// ────────────────────────────────────────────────────────────────
// releaseResources
// ────────────────────────────────────────────────────────────────
void ConvolverProcessor::releaseResources()
{
    // Clean up thread-based loaders
    stopUpgradeThread();
    forceCleanup();
    activeLoader.reset();

    // バッファの解放
    delayBuffer[0].reset();
    delayBuffer[1].reset();
    delayBufferCapacity = 0;

    dryBufferStorage[0].reset();
    dryBufferStorage[1].reset();
    dryBufferCapacity = 0;

    oldDryBufferStorage[0].reset();
    oldDryBufferStorage[1].reset();
    oldDryBufferCapacity = 0;

    smoothingBufferStorage[0].reset();
    smoothingBufferStorage[1].reset();
    smoothingBufferCapacity = 0;

    cachedFFTBuffer.reset();
    cachedFFTBufferCapacity = 0;

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
        fftHandleSize = 0;
    }

    // Release active convolution engine
    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->snapshotRcuEpoch() : 1;
    auto* oldConv = exchangeActiveEngine(nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 engine 取得; release で null 公開
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);

    auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel); // acq_rel: acquire で旧 IRState 取得; release で null 公開
    if (oldIrState != nullptr)
    {
        oldIrState->~IRState();
        mkl_free(oldIrState);
    }

    if (deferredFreeThread)
        deferredFreeThread->shutdownAndDrain();
    deferredFreeThread.reset();

    while (auto* ptr = rcuSwapper.tryReclaim(std::numeric_limits<uint64_t>::max()))
        std::unique_ptr<convo::ConvolverState>{ptr}; // RAII delete

    convo::publishAtomic(isPrepared, false, std::memory_order_release); // release: Runtime 側 isPrepared acquire と HB
}

// ────────────────────────────────────────────────────────────────
// reset
// ────────────────────────────────────────────────────────────────
void ConvolverProcessor::reset()
{
    struct GlobalGuard {
        const ConvolverProcessor& cp;
        GlobalGuard(const ConvolverProcessor& cp_) : cp(cp_) { cp.enterGlobalReader(2); }
        ~GlobalGuard() { cp.exitGlobalReader(2); }
    } guard(*this);

    auto* conv = loadActiveEngine(std::memory_order_acquire); // acquire: exchangeActiveEngine acq_rel/release と HB
    if (conv)
        conv->reset();

    if (delayBuffer[0]) juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    if (delayBuffer[1]) juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    dryBuffer.clear();
    smoothingBuffer.clear();
    convo::fetchAddAtomic(mixSmootherResetPendingGen, static_cast<uint64_t>(1), std::memory_order_acq_rel); // acq_rel: Runtime 側 acquire と HB
    convo::publishAtomic(pendingLatencyValue, latencySmoother.getTargetValue(), std::memory_order_release);  // release: Runtime 側 pendingLatencyValue acquire と HB
    convo::fetchAddAtomic(latencyResetPendingGen, static_cast<uint64_t>(1), std::memory_order_acq_rel);      // acq_rel: Runtime 側 acquire と HB
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE
