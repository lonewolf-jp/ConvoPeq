#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "CacheManager.h"
#include "ProgressiveUpgradeThread.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochManager.h"
#include "core/RCUReader.h"
#include "core/ThreadAffinityManager.h"
#include "AlignedAllocation.h"
#include <mkl.h>

#include "audioengine/AtomicAccess.h"

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE)

const ConvolverProcessor::IRState* ConvolverProcessor::acquireIRState() const noexcept
{
    return convo::consumeAtomic(currentIRState, std::memory_order_acquire);
}

void ConvolverProcessor::releaseIRState(const IRState* /*state*/) const noexcept
{
    // IRState lifetime is managed by deferred retirement.
}

void ConvolverProcessor::updateIRState(const juce::AudioBuffer<double>& newIR, double newSR)
{
    auto uniqueIR = std::make_unique<juce::AudioBuffer<double>>(newIR);

    auto newState = convo::aligned_make_unique<IRState>();
    newState->irOwner = std::move(uniqueIR);
    newState->ir = newState->irOwner.get();
    newState->sampleRate = newSR;
    if (auto* provider = getRcuProvider(); provider != nullptr)
        newState->generation = provider->publishRcuEpoch();
    else
        newState->generation = 1;
    std::atomic_thread_fence(std::memory_order_release);

    auto* oldState = convo::exchangeAtomic(currentIRState, newState.release(), std::memory_order_acq_rel);
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
    if (!sc || convo::exchangeAtomic(sc->retired, true, std::memory_order_acq_rel))
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
    auto* oldConv = exchangeActiveEngine(nullptr, std::memory_order_acq_rel);
    StereoConvolver::retireStereoConvolver(oldConv, getRcuProvider());

    auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel);
    if (oldIrState != nullptr)
    {
        oldIrState->~IRState();
        mkl_free(oldIrState);
    }

    // Clean up latency snapshot pointer
    auto* oldSnap = convo::exchangeAtomic(cachedLatency, nullptr, std::memory_order_acq_rel);
    std::unique_ptr<LatencySnapshot> owned{oldSnap}; // RAII delete

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
    }

    // Note: Do NOT call g_deletionQueue.reclaimAllIgnoringEpoch() here.
    // AudioEngine destructor handles the final reclaim to avoid double-deletion.
}

// ────────────────────────────────────────────────────────────────
// Timer Callback
// ────────────────────────────────────────────────────────────────
void ConvolverProcessor::timerCallback()
{
    static int lastReportedClampCount = 0;
    const int currentClampCount = convo::consumeAtomic(g_totalLatencyClampCount, std::memory_order_acquire);
    if (currentClampCount != lastReportedClampCount)
    {
        juce::Logger::writeToLog("ConvolverProcessor: Latency clamp triggered (total: "
                                 + juce::String(currentClampCount) + " times)");
        lastReportedClampCount = currentClampCount;
    }

    // ★ リングバッファオーバーフローによるリビルド要求を処理 (Audio Thread からは呼ばれない)
    if (convo::consumeAtomic(rebuildPendingAfterLoad, std::memory_order_acquire))
    {
        if (!convo::consumeAtomic(isLoading, std::memory_order_acquire) &&
            !convo::consumeAtomic(isRebuilding, std::memory_order_acquire))
        {
            juce::File irFile;
            {
                const juce::ScopedLock sl(irFileLock);
                irFile = currentIrFile;
            }
            if (irFile.existsAsFile())
            {
                convo::publishAtomic(rebuildPendingAfterLoad, false, std::memory_order_release);
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

    convo::publishAtomic(isPrepared, false, std::memory_order_release);

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
        fftHandleSize = 0;
    }

    const bool rateChanged = (std::abs(convo::consumeAtomic(currentSampleRate) - sampleRate) > 1e-6);
    const bool blockChanged = (convo::consumeAtomic(currentBufferSize, std::memory_order_acquire) != samplesPerBlock);

    convo::publishAtomic(currentBufferSize, samplesPerBlock, std::memory_order_release);
    convo::publishAtomic(currentSampleRate, sampleRate, std::memory_order_release);

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(MAX_BLOCK_SIZE);
    spec.numChannels = 2;

    currentSpec = spec;

    auto* conv = loadActiveEngine(std::memory_order_acquire);
    if (conv) {
        const int internalBlockSize = juce::nextPowerOfTwo(samplesPerBlock);

        if ((rateChanged || blockChanged) && conv->irDataLength > 0)
        {
            StereoConvolver* newConv = nullptr;
            try
            {
                auto newConvHolder = convo::aligned_make_unique<StereoConvolver>();
                newConv = newConvHolder.get();

                auto irL = convo::makeAlignedArray<double>(static_cast<size_t>(conv->irDataLength));
                auto irR = convo::makeAlignedArray<double>(static_cast<size_t>(conv->irDataLength));
                std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
                std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));

                auto sizing = ConvolverProcessorInternal::computeMasteringSizing(internalBlockSize, conv->irDataLength);

                if (newConv->init(irL.release(), irR.release(),
                                  conv->irDataLength, sampleRate, conv->irLatency, sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, samplesPerBlock, conv->storedScale,
                                  convo::consumeAtomic(experimentalDirectHeadEnabled, std::memory_order_acquire),
                                  nullptr, this))
                {
                    newConv = newConvHolder.release();
                    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->publishRcuEpoch() : 1;
                    auto* oldConv = exchangeActiveEngine(newConv, std::memory_order_acq_rel);
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
    auto allocateIfNeeded = [this](convo::ScopedAlignedPtr<double>* storage, int& capacity, const char* name) {
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
    juce::FloatVectorOperations::clear(wetBufferStorage[1].get(), MAX_BLOCK_SIZE);

    // スムージング時間の設定
    mixSmoother.reset(sampleRate, static_cast<double>(convo::consumeAtomic(smoothingTimeSec)));
    mixSmoother.setCurrentAndTargetValue(static_cast<double>(convo::consumeAtomic(mixTarget)));

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

    // [F-01 fix] 世代カウンターをインクリメント (NonRT → RT 通知)
    firstProcessCallGen.fetch_add(1, std::memory_order_acq_rel);

    convo::publishAtomic(isPrepared, true, std::memory_order_release);
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
    const uint64_t retireEpoch = (getRcuProvider() != nullptr) ? getRcuProvider()->publishRcuEpoch() : 1;
    auto* oldConv = exchangeActiveEngine(nullptr, std::memory_order_acq_rel);
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);

    auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel);
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

    runtime.clear();

    convo::publishAtomic(isPrepared, false, std::memory_order_release);
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

    auto* conv = loadActiveEngine(std::memory_order_acquire);
    if (conv)
        conv->reset();

    if (delayBuffer[0]) juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    if (delayBuffer[1]) juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    dryBuffer.clear();
    smoothingBuffer.clear();
    mixSmootherResetPendingGen.fetch_add(1, std::memory_order_acq_rel);
    convo::publishAtomic(pendingLatencyValue, latencySmoother.getTargetValue());
    latencyResetPendingGen.fetch_add(1, std::memory_order_acq_rel);
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE
