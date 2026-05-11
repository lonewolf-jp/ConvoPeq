#include <JuceHeader.h>
#include "ConvolverProcessor.h"
#include "audioengine/AudioEngine.h"
#include "CacheManager.h"
#include "ProgressiveUpgradeThread.h"
#include "convolver/ConvolverProcessor.Internal.h"
#include "core/EpochManager.h"
#include "core/EBRQueue.h"
#include "core/RCUReader.h"
#include "core/ThreadAffinityManager.h"
#include "AlignedAllocation.h"
#include <mkl.h>

#if defined(CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE)

void ConvolverProcessor::overflowCallbackThunk(void* userData) noexcept
{
    auto* self = static_cast<ConvolverProcessor*>(userData);
    bool expected = false;
    self->overflowRequested.compare_exchange_strong(expected, true,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_relaxed);
}

const ConvolverProcessor::IRState* ConvolverProcessor::acquireIRState() const noexcept
{
    return currentIRState.load(std::memory_order_acquire);
}

void ConvolverProcessor::releaseIRState(const IRState* /*state*/) const noexcept
{
    // IRState lifetime is managed by deferred retirement.
}

void ConvolverProcessor::updateIRState(std::shared_ptr<juce::AudioBuffer<double>> newIR, double newSR)
{
    if (!newIR)
        return;

    void* mem = mkl_malloc(sizeof(IRState), 64);
    if (mem == nullptr)
    {
        jassertfalse;
        return;
    }

    auto* newState = new (mem) IRState();
    newState->irOwner = std::move(newIR);
    newState->ir = newState->irOwner.get();
    newState->sampleRate = newSR;
    newState->generation = rcuProvider ? rcuProvider->publishRcuEpoch() : 1;
    std::atomic_thread_fence(std::memory_order_release);

    auto* oldState = currentIRState.exchange(newState, std::memory_order_acq_rel);
    if (oldState != nullptr)
    {
        convo::retireObject(oldState, [](void* p)
        {
            auto* state = static_cast<IRState*>(p);
            state->~IRState();
            mkl_free(state);
        });
    }
}

void ConvolverProcessor::retireStereoConvolver(StereoConvolver* conv, uint64_t /*retireEpoch*/)
{
    StereoConvolver::retireStereoConvolver(conv);
}

// ────────────────────────────────────────────────────────────────
// Constructor
// ────────────────────────────────────────────────────────────────
ConvolverProcessor::ConvolverProcessor()
    : mixSmoother(1.0)
{
    irConverter = std::make_unique<IRConverter>();

    cacheManager = std::make_unique<CacheManager>();
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
    auto* oldConv = m_activeEngine.exchange(nullptr, std::memory_order_acq_rel);
    StereoConvolver::retireStereoConvolver(oldConv);

    auto* oldIrState = currentIRState.exchange(nullptr, std::memory_order_acq_rel);
    if (oldIrState != nullptr)
    {
        oldIrState->~IRState();
        mkl_free(oldIrState);
    }

    // Clean up latency snapshot pointer
    auto* oldSnap = cachedLatency.exchange(nullptr, std::memory_order_acq_rel);
    delete oldSnap;

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
    const int currentClampCount = g_totalLatencyClampCount.load(std::memory_order_relaxed);
    if (currentClampCount != lastReportedClampCount)
    {
        juce::Logger::writeToLog("ConvolverProcessor: Latency clamp triggered (total: "
                                 + juce::String(currentClampCount) + " times)");
        lastReportedClampCount = currentClampCount;
    }

    // ★ リングバッファオーバーフローによるリビルド要求を処理 (Audio Thread からは呼ばれない)
    if (rebuildPendingAfterLoad.load(std::memory_order_acquire))
    {
        if (!isLoading.load(std::memory_order_acquire) &&
            !isRebuilding.load(std::memory_order_acquire))
        {
            juce::File irFile;
            {
                const juce::ScopedLock sl(irFileLock);
                irFile = currentIrFile;
            }
            if (irFile.existsAsFile())
            {
                rebuildPendingAfterLoad.store(false, std::memory_order_release);
                loadImpulseResponse(irFile, false);
            }
        }
    }

    auto* provider = rcuProvider;
    if (provider) {
        convo::EBRQueue::instance().tryReclaim();
    }

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

    isPrepared.store(false, std::memory_order_release);

    if (fftHandle.get() != nullptr) {
        fftHandle.reset();
        fftHandleSize = 0;
    }

    const bool rateChanged = (std::abs(currentSampleRate.load() - sampleRate) > 1e-6);
    const bool blockChanged = (currentBufferSize.load(std::memory_order_relaxed) != samplesPerBlock);

    currentBufferSize.store(samplesPerBlock, std::memory_order_release);
    currentSampleRate.store(sampleRate, std::memory_order_release);

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(MAX_BLOCK_SIZE);
    spec.numChannels = 2;

    currentSpec = spec;

    auto* conv = m_activeEngine.load(std::memory_order_acquire);
    if (conv) {
        const int internalBlockSize = juce::nextPowerOfTwo(samplesPerBlock);

        if ((rateChanged || blockChanged) && conv->irDataLength > 0)
        {
            StereoConvolver* newConv = nullptr;
            try
            {
                void* mem = convo::aligned_malloc(sizeof(StereoConvolver), 64);
                new (mem) StereoConvolver();
                newConv = static_cast<StereoConvolver*>(mem);

                convo::ScopedAlignedPtr<double> irL(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
                convo::ScopedAlignedPtr<double> irR(static_cast<double*>(convo::aligned_malloc(conv->irDataLength * sizeof(double), 64)));
                std::memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double));
                std::memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double));

                auto sizing = ConvolverProcessorInternal::computeMasteringSizing(internalBlockSize, conv->irDataLength);

                if (newConv->init(irL.release(), irR.release(),
                                  conv->irDataLength, sampleRate, conv->irLatency, sizing.maxFFTSize, internalBlockSize, sizing.firstPartition, samplesPerBlock, conv->storedScale,
                                  experimentalDirectHeadEnabled.load(std::memory_order_acquire),
                                  nullptr, this))
                {
                    const uint64_t retireEpoch = rcuProvider ? rcuProvider->publishRcuEpoch() : 1;
                    auto* oldConv = m_activeEngine.exchange(newConv, std::memory_order_acq_rel);
                    if (oldConv)
                        retireStereoConvolver(oldConv, retireEpoch);
                    updateLatencyCache();
                }
                else
                {
                    juce::Logger::writeToLog("ConvolverProcessor::prepareToPlay: NUC re-init failed (MKL alloc?). Keeping existing engine.");
                    StereoConvolver::retireStereoConvolver(std::exchange(newConv, nullptr));
                }
            }
            catch (const std::bad_alloc&)
            {
                StereoConvolver::retireStereoConvolver(std::exchange(newConv, nullptr));
                juce::Logger::writeToLog("ConvolverProcessor::prepareToPlay: NUC re-init failed (std::bad_alloc). Keeping existing engine.");
            }
        }
    }

    // DelayLine準備
    if (delayBufferCapacity < DELAY_BUFFER_SIZE)
    {
        auto* newL = static_cast<double*>(convo::aligned_malloc(DELAY_BUFFER_SIZE * sizeof(double), 64));
        auto* newR = static_cast<double*>(convo::aligned_malloc(DELAY_BUFFER_SIZE * sizeof(double), 64));
        if (!newL || !newR)
        {
            if (newL) convo::aligned_free(newL);
            if (newR) convo::aligned_free(newR);
            lastError = "Failed to allocate delay buffers";
            isPrepared.store(false, std::memory_order_release);
            return;
        }
        delayBuffer[0].reset(newL);
        delayBuffer[1].reset(newR);
        delayBufferCapacity = DELAY_BUFFER_SIZE;
    }
    juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    // Dry/Wet/Smoothing/Oldバッファ確保 (まとめて処理)
    auto allocateIfNeeded = [this](convo::ScopedAlignedPtr<double>* storage, int& capacity, const char* name) {
        if (capacity < MAX_BLOCK_SIZE)
        {
            auto* newL = static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64));
            auto* newR = static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64));
            if (!newL || !newR)
            {
                if (newL) convo::aligned_free(newL);
                if (newR) convo::aligned_free(newR);
                lastError = juce::String("Failed to allocate ") + name;
                isPrepared.store(false, std::memory_order_release);
                return false;
            }
            storage[0].reset(newL);
            storage[1].reset(newR);
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
        auto* newL = static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64));
        auto* newR = static_cast<double*>(convo::aligned_malloc(MAX_BLOCK_SIZE * sizeof(double), 64));
        if (!newL || !newR)
        {
            if (newL) convo::aligned_free(newL);
            if (newR) convo::aligned_free(newR);
            lastError = "Failed to allocate wet buffers";
            isPrepared.store(false, std::memory_order_release);
            return;
        }
        wetBufferStorage[0].reset(newL);
        wetBufferStorage[1].reset(newR);
        wetBufferCapacity = MAX_BLOCK_SIZE;
    }
    juce::FloatVectorOperations::clear(wetBufferStorage[0].get(), MAX_BLOCK_SIZE);
    juce::FloatVectorOperations::clear(wetBufferStorage[1].get(), MAX_BLOCK_SIZE);

    // スムージング時間の設定
    mixSmoother.reset(sampleRate, static_cast<double>(smoothingTimeSec.load()));
    mixSmoother.setCurrentAndTargetValue(static_cast<double>(mixTarget.load()));
    (void)mixSmoother.getNextValue();

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
        if (rcuProvider != nullptr)
            affinityMgr = const_cast<ThreadAffinityManager*>(&rcuProvider->getAffinityManager());

        deferredFreeThread = std::make_unique<convo::DeferredFreeThread>(rcuSwapper, affinityMgr);
    }

    firstProcessCall.store(true, std::memory_order_release);

    isPrepared.store(true, std::memory_order_release);
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
    const uint64_t retireEpoch = rcuProvider ? rcuProvider->publishRcuEpoch() : 1;
    auto* oldConv = m_activeEngine.exchange(nullptr, std::memory_order_acq_rel);
    if (oldConv)
        retireStereoConvolver(oldConv, retireEpoch);

    auto* oldIrState = currentIRState.exchange(nullptr, std::memory_order_acq_rel);
    if (oldIrState != nullptr)
    {
        oldIrState->~IRState();
        mkl_free(oldIrState);
    }

    if (deferredFreeThread)
        deferredFreeThread->shutdownAndDrain();
    deferredFreeThread.reset();

    while (auto* ptr = rcuSwapper.tryReclaim(std::numeric_limits<uint64_t>::max()))
        delete ptr;

    runtime.clear();

    isPrepared.store(false, std::memory_order_release);
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

    auto* conv = m_activeEngine.load(std::memory_order_acquire);
    if (conv)
        conv->reset();

    if (delayBuffer[0]) juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
    if (delayBuffer[1]) juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
    delayWritePos = 0;

    dryBuffer.clear();
    smoothingBuffer.clear();
    mixSmootherResetPending.store(true, std::memory_order_release);
    pendingLatencyValue.store(latencySmoother.getTargetValue());
    latencyResetPending.store(true, std::memory_order_release);
}

#endif // CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE
