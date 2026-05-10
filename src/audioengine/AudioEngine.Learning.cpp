#include <JuceHeader.h>
#include "AudioEngine.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_START)

void AudioEngine::startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume)
{
    if (noiseShaperLearner == nullptr)
        return;

    pendingLearningMode.store(mode, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();

    if (noiseShaperType.load(std::memory_order_acquire) != NoiseShaperType::Adaptive9thOrder)
        setNoiseShaperType(NoiseShaperType::Adaptive9thOrder);

    const LearningCommand cmd {
        LearningCommand::Type::Start,
        resume,
        mode,
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] startNoiseShaperLearning: command queue overflow");
        return;
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_STOP)

void AudioEngine::stopNoiseShaperLearning()
{
    const LearningCommand cmd {
        LearningCommand::Type::Stop,
        false,
        pendingLearningMode.load(std::memory_order_acquire),
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] stopNoiseShaperLearning: command queue overflow");
    }

    if (noiseShaperLearner)
        noiseShaperLearner->stopLearning();
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_RESET)

void AudioEngine::resetLearningControlState() noexcept
{
    learningCommandWrite = 0;
    learningCommandRead = 0;
    learnerDispatchWrite = 0;
    learnerDispatchRead = 0;
    learnerDispatchOverflow.store(false, std::memory_order_release);
    lastFailedAction.store(LearnerDispatchAction {}, std::memory_order_release);
    learningRuntimeState = LearningRuntimeState::Idle;
    requestedLearningMode = pendingLearningMode.load(std::memory_order_acquire);
    requestedLearningResume = false;
    requestedLearningGeneration = pendingIRGeneration;
    currentIRGeneration = pendingIRGeneration;
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_DEFERRED)

void AudioEngine::processDeferredLearningActions()
{
    LearnerDispatchAction action;
    while (dequeueLearnerDispatch(action))
    {
        if (noiseShaperLearner == nullptr)
            continue;

        if (action.type == LearnerDispatchAction::Type::Stop)
        {
            noiseShaperLearner->stopLearning();
            continue;
        }

        noiseShaperLearner->setLearningMode(action.mode);
        noiseShaperLearner->startLearning(action.resume);
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_PROCESS)

void AudioEngine::processLearningCommands() noexcept
{
    if (learnerDispatchOverflow.load(std::memory_order_acquire))
    {
        const LearnerDispatchAction last = lastFailedAction.load(std::memory_order_acquire);
        if (enqueueLearnerDispatch(last))
            learnerDispatchOverflow.store(false, std::memory_order_release);
    }

    LearningCommand cmd;
    while (dequeueLearningCommand(cmd))
    {
        switch (cmd.type)
        {
            case LearningCommand::Type::Start:
            {
                requestedLearningMode = cmd.mode;
                requestedLearningResume = cmd.resume;
                requestedLearningGeneration = cmd.irGeneration;

                auto* dsp = currentDSP.load(std::memory_order_acquire);
                // irGeneration チェックを削除: DSP が有効かつ型が適切であれば即座に学習開始可能
                const bool dspReady = (dsp != nullptr)
                    && (dsp->noiseShaperType == NoiseShaperType::Adaptive9thOrder);

                if (!dspReady)
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    break;
                }

                if (learningRuntimeState == LearningRuntimeState::Running)
                {
                    const LearnerDispatchAction stopAction {
                        LearnerDispatchAction::Type::Stop,
                        false,
                        requestedLearningMode
                    };

                    if (!enqueueLearnerDispatch(stopAction))
                    {
                        DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                    }
                }

                const LearnerDispatchAction startAction {
                    LearnerDispatchAction::Type::Start,
                    requestedLearningResume,
                    requestedLearningMode
                };

                if (enqueueLearnerDispatch(startAction))
                    learningRuntimeState = LearningRuntimeState::Running;
                else
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    DBG("[AudioEngine] processLearningCommands: learner start queue overflow");
                }
                break;
            }

            case LearningCommand::Type::Stop:
            {
                requestedLearningResume = false;
                requestedLearningGeneration = currentIRGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                learningRuntimeState = LearningRuntimeState::Idle;
                break;
            }

            case LearningCommand::Type::IRChanged:
            {
                const bool shouldRestart = (learningRuntimeState != LearningRuntimeState::Idle);
                requestedLearningGeneration = cmd.irGeneration;

                const LearnerDispatchAction stopAction {
                    LearnerDispatchAction::Type::Stop,
                    false,
                    requestedLearningMode
                };

                if (!enqueueLearnerDispatch(stopAction))
                {
                    DBG("[AudioEngine] processLearningCommands: learner stop queue overflow");
                }

                if (shouldRestart)
                {
                    requestedLearningResume = false;
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                }
                else
                {
                    learningRuntimeState = LearningRuntimeState::Idle;
                }
                break;
            }

            case LearningCommand::Type::DSPReady:
            {
                currentIRGeneration = cmd.irGeneration;

                // irGeneration チェックを削除: WaitingForDSP 状態であれば遅延なく学習開始
                if (learningRuntimeState == LearningRuntimeState::WaitingForDSP)
                {
                    const LearnerDispatchAction startAction {
                        LearnerDispatchAction::Type::Start,
                        requestedLearningResume,
                        requestedLearningMode
                    };

                    if (enqueueLearnerDispatch(startAction))
                    {
                        learningRuntimeState = LearningRuntimeState::Running;
                    }
                    else
                    {
                        DBG("[AudioEngine] processLearningCommands: DSPReady learner start queue overflow");
                    }
                }
                break;
            }
        }
    }
}

#endif

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_HELPERS)

namespace
{
    static constexpr std::array<double, kAdaptiveNoiseShaperSampleRateBankCount> kAdaptiveSupportedSampleRatesHz_helpers
    {
        44100.0, 48000.0, 88200.0, 96000.0, 176400.0,
        192000.0, 352800.0, 384000.0, 705600.0, 768000.0
    };

    static constexpr std::array<double, kAdaptiveNoiseShaperOrder> kDefaultAdaptiveNoiseShaperCoeffs_helpers
    {
        0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
    };

    inline int clampAdaptiveBankIndex_helpers(int bankIndex) noexcept
    {
        if (bankIndex < 0)
            return 0;
        if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount)
            return kAdaptiveNoiseShaperSampleRateBankCount - 1;
        return bankIndex;
    }
}

void AudioEngine::setNoiseShaperLearningMode(NoiseShaperLearner::LearningMode mode)
{
    pendingLearningMode.store(mode, std::memory_order_release);
    selectAdaptiveCoeffBankForCurrentSettings();
    if (noiseShaperLearner)
        noiseShaperLearner->setLearningMode(mode);
}

bool AudioEngine::isNoiseShaperLearning() const
{
    return noiseShaperLearner && noiseShaperLearner->isRunning();
}

const NoiseShaperLearner::Progress& AudioEngine::getNoiseShaperLearningProgress() const
{
    jassert(noiseShaperLearner);
    return noiseShaperLearner->getProgress();
}

NoiseShaperLearner::Settings AudioEngine::getNoiseShaperLearnerSettings() const
{
    if (noiseShaperLearner)
        return noiseShaperLearner->getSettings();
    return {};
}

void AudioEngine::setNoiseShaperLearnerSettings(const NoiseShaperLearner::Settings& settings)
{
    if (noiseShaperLearner)
        noiseShaperLearner->setSettings(settings);
}

int AudioEngine::copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept
{
    return noiseShaperLearner ? noiseShaperLearner->copyBestScoreHistory(outScores, maxPoints) : 0;
}

const char* AudioEngine::getNoiseShaperLearningError() const noexcept
{
    if (noiseShaperLearner == nullptr)
        return nullptr;
    return noiseShaperLearner->getErrorMessage();
}

int AudioEngine::getAdaptiveSampleRateBankCount() noexcept
{
    return kAdaptiveNoiseShaperSampleRateBankCount;
}

double AudioEngine::getAdaptiveSampleRateBankHz(int bankIndex) noexcept
{
    return kAdaptiveSupportedSampleRatesHz_helpers[static_cast<size_t>(clampAdaptiveBankIndex_helpers(bankIndex))];
}

void AudioEngine::initialiseAdaptiveCoeffBanks() noexcept
{
    for (int srBank = 0; srBank < kAdaptiveNoiseShaperSampleRateBankCount; ++srBank)
    {
        double sr = getAdaptiveSampleRateBankHz(srBank);
        for (int bdIdx = 0; bdIdx < kAdaptiveBitDepthCount; ++bdIdx)
        {
            for (int modeIdx = 0; modeIdx < kLearningModeCount; ++modeIdx)
            {
                int bankIndex = (srBank * kAdaptiveBitDepthCount + bdIdx) * kLearningModeCount + modeIdx;
                auto& bank = adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
                bank.sampleRateHz = sr;

                for (int coeffIndex = 0; coeffIndex < kAdaptiveNoiseShaperOrder; ++coeffIndex)
                {
                    const double coefficient = kDefaultAdaptiveNoiseShaperCoeffs_helpers[static_cast<size_t>(coeffIndex)];
                    bank.coeffSetA.k[coeffIndex] = coefficient;
                    bank.coeffSetB.k[coeffIndex] = coefficient;
                }

                bank.activeIndex.store(0, std::memory_order_relaxed);
                bank.generation.store(1u, std::memory_order_relaxed);
                bank.writeLock.store(false, std::memory_order_relaxed);
            }
        }
    }
}

int AudioEngine::resolveAdaptiveCoeffBankIndex(double sampleRate) noexcept
{
    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::max();

    for (int bankIndex = 0; bankIndex < kAdaptiveNoiseShaperSampleRateBankCount; ++bankIndex)
    {
        const double distance = std::abs(sampleRate - getAdaptiveSampleRateBankHz(bankIndex));
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestIndex = bankIndex;
        }
    }

    return bestIndex;
}

int AudioEngine::getAdaptiveBitDepthIndex(int bitDepth) noexcept
{
    if (bitDepth <= 16) return 0;
    if (bitDepth <= 24) return 1;
    return 2;
}

int AudioEngine::getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, NoiseShaperLearner::LearningMode mode) noexcept
{
    const int srBank = resolveAdaptiveCoeffBankIndex(sampleRate);
    const int bdIdx  = getAdaptiveBitDepthIndex(bitDepth);
    const int modeIdx = static_cast<int>(mode);
    return (srBank * kAdaptiveBitDepthCount + bdIdx) * kLearningModeCount + modeIdx;
}

AudioEngine::AdaptiveCoeffBankSlot& AudioEngine::getAdaptiveCoeffBankForIndex(int bankIndex) noexcept
{
    if (bankIndex < 0) bankIndex = 0;
    if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount)
        bankIndex = kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount - 1;
    return adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
}

const AudioEngine::AdaptiveCoeffBankSlot& AudioEngine::getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept
{
    if (bankIndex < 0) bankIndex = 0;
    if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount)
        bankIndex = kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount - 1;
    return adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
}

void AudioEngine::selectAdaptiveCoeffBankForCurrentSettings() noexcept
{
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bd   = ditherBitDepth.load(std::memory_order_acquire);
    const auto mode = pendingLearningMode.load(std::memory_order_acquire);

    const int newBankIndex = getAdaptiveCoeffBankIndex(sr, bd, mode);

    if (newBankIndex != currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire))
    {
        currentAdaptiveCoeffBankIndex.store(newBankIndex, std::memory_order_release);

        if (noiseShaperLearner)
        {
            juce::MessageManager::callAsync([this, newBankIndex]() {
                if (auto* learner = noiseShaperLearner.get())
                    learner->onCoeffBankChanged(newBankIndex);
            });
        }
    }
}

void AudioEngine::getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(
        currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire));

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = bank.generation.load(std::memory_order_acquire);
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = bank.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            const int limit = std::min(kAdaptiveNoiseShaperOrder, maxCoefficients);
            for (int i = 0; i < limit; ++i)
                outCoeffs[i] = coeffSet->k[i];
            return;
        }
    }
}

void AudioEngine::setCurrentAdaptiveCoefficients(const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const int bankIndex = currentAdaptiveCoeffBankIndex.load(std::memory_order_acquire);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getCurrentAdaptiveCoefficients(stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::getAdaptiveCoefficientsForSampleRate(double sampleRate, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(resolveAdaptiveCoeffBankIndex(sampleRate));

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = bank.generation.load(std::memory_order_acquire);
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = bank.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            const int limit = std::min(kAdaptiveNoiseShaperOrder, maxCoefficients);
            for (int i = 0; i < limit; ++i)
                outCoeffs[i] = coeffSet->k[i];
            return;
        }
    }
}

void AudioEngine::setAdaptiveCoefficientsForSampleRate(double sampleRate, const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const int bankIndex = resolveAdaptiveCoeffBankIndex(sampleRate);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getAdaptiveCoefficientsForSampleRate(sampleRate, stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bank = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    const auto& slot = getAdaptiveCoeffBankForIndex(bank);

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = slot.generation.load(std::memory_order_acquire);
        const CoeffSet* active = AudioEngine::getActiveCoeffSet(slot);
        const uint32_t genAfter = slot.generation.load(std::memory_order_acquire);

        if (genBefore == genAfter)
        {
            if (active)
            {
                const int copyCount = std::min(maxCoefficients, kAdaptiveNoiseShaperOrder);
                std::memcpy(outCoeffs, active->k, static_cast<size_t>(copyCount) * sizeof(double));
            }
            else
            {
                std::memcpy(outCoeffs, kDefaultAdaptiveNoiseShaperCoeffs_helpers.data(),
                            sizeof(kDefaultAdaptiveNoiseShaperCoeffs_helpers));
            }
            return;
        }
    }
}

void AudioEngine::setAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, const double* coeffs, int numCoefficients)
{
    if (coeffs == nullptr || numCoefficients <= 0)
        return;

    if (isNoiseShaperLearning())
    {
        DBG_LOG("[AudioEngine] Coefficient update rejected during learning");
        return;
    }

    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bankIndex = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getAdaptiveCoefficientsForSampleRateAndBitDepth(sampleRate, bitDepth, stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    publishCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::setAdaptiveAutosaveCallback(std::function<void()> callback)
{
    const std::scoped_lock lock(adaptiveAutosaveCallbackMutex);
    adaptiveAutosaveCallback = std::move(callback);
}

void AudioEngine::requestAdaptiveAutosave()
{
    std::function<void()> callbackCopy;
    {
        const std::scoped_lock lock(adaptiveAutosaveCallbackMutex);
        callbackCopy = adaptiveAutosaveCallback;
    }

    if (callbackCopy)
        callbackCopy();
}

void AudioEngine::publishCoeffs(const double* coeffs)
{
    const double sr = currentSampleRate.load(std::memory_order_acquire);
    const int bd  = ditherBitDepth.load(std::memory_order_acquire);
    const auto mode = pendingLearningMode.load(std::memory_order_acquire);
    const int bank = getAdaptiveCoeffBankIndex(sr, bd, mode);

    publishCoeffsToBank(bank, coeffs);
}

void AudioEngine::publishCoeffsToBank(int bankIndex, const double* coeffs)
{
    if (coeffs == nullptr)
        return;

    jassert(juce::MessageManager::existsAndIsCurrentThread());

    auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);

    CoeffSetWriteLockGuard guard(bank);

    for (int retry = 0; retry < 100; ++retry)
    {
        if (guard.acquire())
            break;
        std::this_thread::yield();
    }

    if (!guard.isAcquired())
    {
        DBG_LOG("[AudioEngine] Failed to acquire coeff write lock (bank="
                + juce::String(bankIndex) + ")");
        return;
    }

    CoeffSet* inactive = getReservedInactiveCoeffSet(bank);

    for (int i = 0; i < kAdaptiveNoiseShaperOrder; ++i)
        inactive->k[i] = coeffs[i];

    guard.commit();
}

bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& outState) const noexcept
{
    const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(bank.stateMutex));
    outState = bank.state;
    return true;
}

void AudioEngine::setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& inState) noexcept
{
    auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    std::lock_guard<std::mutex> lock(bank.stateMutex);
    bank.state = inState;
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_LEARNING_HELPERS
