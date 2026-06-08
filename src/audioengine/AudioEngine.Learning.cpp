#include <JuceHeader.h>
#include "AudioEngine.h"
#include "core/RuntimeReaderContext.h"
#include "NoiseShaperLearner.h"

void AudioEngine::startNoiseShaperLearning(convo::NoiseShaperLearningMode mode, bool resume)
{
    convo::publishAtomic(pendingLearningMode, mode, std::memory_order_release); // release: selectAdaptiveCoeffBankForCurrentSettings/processLearningCommands acquire と HB
    selectAdaptiveCoeffBankForCurrentSettings();

    if (convo::consumeAtomic(noiseShaperType, std::memory_order_acquire) != NoiseShaperType::Adaptive9thOrder) // acquire: setNoiseShaperType の publishAtomic release と HB
        setNoiseShaperType(NoiseShaperType::Adaptive9thOrder);

    if (noiseShaperLearner == nullptr)
    {
        juce::Logger::writeToLog("[AudioEngine] startNoiseShaperLearning: learner unavailable after adaptive switch");
        return;
    }

    const LearningCommand cmd {
        LearningCommand::Type::Start,
        resume,
        mode,
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] startNoiseShaperLearning: command queue overflow");
        juce::Logger::writeToLog("[AudioEngine] startNoiseShaperLearning: command queue overflow");
        return;
    }

    juce::Logger::writeToLog("[AudioEngine] startNoiseShaperLearning: command queued mode="
                             + juce::String(static_cast<int>(mode))
                             + " resume=" + juce::String(static_cast<int>(resume)));
}


void AudioEngine::stopNoiseShaperLearning()
{
    juce::Logger::writeToLog("[AudioEngine] stopNoiseShaperLearning called");
    const LearningCommand cmd {
        LearningCommand::Type::Stop,
        false,
        convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire),
        pendingIRGeneration
    };

    if (!enqueueLearningCommand(cmd))
    {
        DBG("[AudioEngine] stopNoiseShaperLearning: command queue overflow");
    }

    if (noiseShaperLearner)
    {
        juce::Logger::writeToLog("[AudioEngine] stopNoiseShaperLearning: calling learner->stopLearning()");
        noiseShaperLearner->stopLearning();
    }

    convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
}


void AudioEngine::resetLearningControlState() noexcept
{
    learningCommandWrite = 0;
    learningCommandRead = 0;
    learnerDispatchWrite = 0;
    learnerDispatchRead = 0;
    convo::publishAtomic(learnerDispatchOverflow, false, std::memory_order_release); // release: processLearningCommands acquire と HB
    convo::publishAtomic(lastFailedAction, LearnerDispatchAction {}, std::memory_order_release); // release: processLearningCommands acquire と HB
    learningRuntimeState = LearningRuntimeState::Idle;
    requestedLearningMode = convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire); // acquire: publishAtomic release と HB
    requestedLearningResume = false;
    requestedLearningGeneration = pendingIRGeneration;
    currentIRGeneration = pendingIRGeneration;
    convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release); // release: audio thread consumeAtomic acquire と HB
}


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


void AudioEngine::processLearningCommands() noexcept
{
    if (convo::consumeAtomic(learnerDispatchOverflow, std::memory_order_acquire)) // acquire: publishAtomic release と HB
    {
        const LearnerDispatchAction last = convo::consumeAtomic(lastFailedAction, std::memory_order_acquire); // acquire: publishAtomic release と HB
        if (enqueueLearnerDispatch(last))
            convo::publishAtomic(learnerDispatchOverflow, false, std::memory_order_release); // release: next consumeAtomic acquire と HB
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

                const convo::RuntimeReaderContext messageCtx{ messageThreadRcuReader, convo::ObserveChannel::Message };
                const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
                auto* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
                // noiseShaperType 判定は AudioEngine の atomic 設定値を基準とする。
                // DSPCore の noiseShaperType フィールドは構築時のスナップショットであり、
                // setNoiseShaperType() 後に作成されなかった DSPCore では旧値が残る。
                // 実際のノイズシェイパー制御は AudioEngine の atomic で行われるため、
                // DSPCore のコピー値ではなく AudioEngine の設定値を参照する。
                const bool dspReady = (dsp != nullptr)
                    && (convo::consumeAtomic(noiseShaperType, std::memory_order_acquire)
                        == NoiseShaperType::Adaptive9thOrder);

                juce::Logger::writeToLog("[AudioEngine] processLearningCommands: Start state="
                    + juce::String(static_cast<int>(learningRuntimeState))
                    + " dspReady=" + juce::String(static_cast<int>(dspReady)));

                if (!dspReady)
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
                    break;
                }

                if (learningRuntimeState == LearningRuntimeState::Running)
                {
                    const LearnerDispatchAction stopAction {
                        LearnerDispatchAction::Type::Stop,
                        false,
                        requestedLearningMode
                    };

                    juce::Logger::writeToLog("[AudioEngine] processLearningCommands: enqueue Stop before Start (state was Running)");
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
                {
                    learningRuntimeState = LearningRuntimeState::Running;
                    convo::publishAtomic(adaptiveCaptureActiveRt, true, std::memory_order_release);
                    juce::Logger::writeToLog("[AudioEngine] processLearningCommands: Start dispatch enqueued ok");
                }
                else
                {
                    learningRuntimeState = LearningRuntimeState::WaitingForDSP;
                    convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
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
                convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release); // release: audio thread consumeAtomic acquire と HB
                break;
            }

            case LearningCommand::Type::IRChanged:
            {
                const bool shouldRestart = (learningRuntimeState != LearningRuntimeState::Idle);
                requestedLearningGeneration = cmd.irGeneration;

                juce::Logger::writeToLog("[AudioEngine] processLearningCommands: IRChanged state="
                    + juce::String(static_cast<int>(learningRuntimeState))
                    + " shouldRestart=" + juce::String(static_cast<int>(shouldRestart)));

                // When learner is actively Running, IRChanged during DSP replacement
                // (e.g. HardReset) is not an actual IR change. Don't stop the learner.
                if (learningRuntimeState == LearningRuntimeState::Running)
                {
                    currentIRGeneration = cmd.irGeneration;
                    juce::Logger::writeToLog("[AudioEngine] processLearningCommands: IRChanged suppressed (Running)");
                    break;
                }

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
                convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
                break;
            }

            case LearningCommand::Type::DSPReady:
            {
                currentIRGeneration = cmd.irGeneration;

                juce::Logger::writeToLog("[AudioEngine] processLearningCommands: DSPReady state="
                    + juce::String(static_cast<int>(learningRuntimeState)));

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
                        convo::publishAtomic(adaptiveCaptureActiveRt, true, std::memory_order_release);
                        juce::Logger::writeToLog("[AudioEngine] processLearningCommands: DSPReady -> enqueued Start dispatch");
                    }
                    else
                    {
                        convo::publishAtomic(adaptiveCaptureActiveRt, false, std::memory_order_release);
                        DBG("[AudioEngine] processLearningCommands: DSPReady learner start queue overflow");
                    }
                }
                break;
            }
        }
    }
}


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

void AudioEngine::setNoiseShaperLearningMode(convo::NoiseShaperLearningMode mode)
{
    convo::publishAtomic(pendingLearningMode, mode, std::memory_order_release); // release: worker consumeAtomic acquire と HB
    selectAdaptiveCoeffBankForCurrentSettings();
    if (noiseShaperLearner)
        noiseShaperLearner->setLearningMode(mode);
}

[[nodiscard]] bool AudioEngine::isNoiseShaperLearning() const
{
    return noiseShaperLearner && noiseShaperLearner->isRunning();
}

[[nodiscard]] const convo::NoiseShaperLearnerProgress& AudioEngine::getNoiseShaperLearningProgress() const
{
    jassert(noiseShaperLearner);
    return noiseShaperLearner->getProgress();
}

[[nodiscard]] convo::NoiseShaperLearnerSettings AudioEngine::getNoiseShaperLearnerSettings() const
{
    if (noiseShaperLearner)
        return noiseShaperLearner->getSettings();
    return {};
}

void AudioEngine::setNoiseShaperLearnerSettings(const convo::NoiseShaperLearnerSettings& settings)
{
    if (noiseShaperLearner)
        noiseShaperLearner->setSettings(settings);
}

[[nodiscard]] int AudioEngine::copyNoiseShaperLearningHistory(double* outScores, int maxPoints) const noexcept
{
    return noiseShaperLearner ? noiseShaperLearner->copyBestScoreHistory(outScores, maxPoints) : 0;
}

[[nodiscard]] const char* AudioEngine::getNoiseShaperLearningError() const noexcept
{
    if (noiseShaperLearner == nullptr)
        return nullptr;
    return noiseShaperLearner->getErrorMessage();
}

[[nodiscard]] int AudioEngine::getAdaptiveSampleRateBankCount() noexcept
{
    return kAdaptiveNoiseShaperSampleRateBankCount;
}

[[nodiscard]] double AudioEngine::getAdaptiveSampleRateBankHz(int bankIndex) noexcept
{
    return kAdaptiveSupportedSampleRatesHz_helpers[static_cast<size_t>(clampAdaptiveBankIndex_helpers(bankIndex))];
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

[[nodiscard]] int AudioEngine::getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, convo::NoiseShaperLearningMode mode) noexcept
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

[[nodiscard]] const AudioEngine::AdaptiveCoeffBankSlot& AudioEngine::getAdaptiveCoeffBankForIndex(int bankIndex) const noexcept
{
    if (bankIndex < 0) bankIndex = 0;
    if (bankIndex >= kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount)
        bankIndex = kAdaptiveNoiseShaperSampleRateBankCount * kAdaptiveBitDepthCount * kLearningModeCount - 1;
    return adaptiveCoeffBanks[static_cast<size_t>(bankIndex)];
}

void AudioEngine::selectAdaptiveCoeffBankForCurrentSettings() noexcept
{
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/setSampleRate publishAtomic release と HB
    const int bd   = convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire); // acquire: setDitherBitDepth publishAtomic release と HB
    const auto mode = convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire); // acquire: setNoiseShaperLearningMode publishAtomic release と HB

    const int newBankIndex = getAdaptiveCoeffBankIndex(sr, bd, mode);

    if (newBankIndex != convo::consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire)) // acquire: publishAtomic release と HB
    {
        convo::publishAtomic(currentAdaptiveCoeffBankIndex, newBankIndex, std::memory_order_release); // release: consumeAtomic acquire と HB

        if (noiseShaperLearner)
        {
            const juce::WeakReference<AudioEngine> weakEngine(this);
            juce::MessageManager::callAsync([weakEngine, newBankIndex]() {
                if (auto* engine = weakEngine.get())
                {
                    if (auto* learner = engine->noiseShaperLearner.get())
                        learner->onCoeffBankChanged(newBankIndex);
                }
            });
        }
    }
}

void AudioEngine::getCurrentAdaptiveCoefficients(double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(
        convo::consumeAtomic(currentAdaptiveCoeffBankIndex, std::memory_order_acquire)); // acquire: publishAtomic release と HB

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = convo::consumeAtomic(bank.generation, std::memory_order_acquire); // acquire: bank update publishAtomic release と HB
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = convo::consumeAtomic(bank.generation, std::memory_order_acquire); // acquire: publishAtomic release と HB

        if (genBefore == genAfter)
        {
            const int limit = std::min(kAdaptiveNoiseShaperOrder, maxCoefficients);
            for (int i = 0; i < limit; ++i)
                outCoeffs[i] = coeffSet->k[i];
            return;
        }
    }
}

void AudioEngine::getAdaptiveCoefficientsForSampleRate(double sampleRate, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto& bank = getAdaptiveCoeffBankForIndex(resolveAdaptiveCoeffBankIndex(sampleRate));

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = convo::consumeAtomic(bank.generation, std::memory_order_acquire); // acquire: bank update publishAtomic release と HB
        const auto* coeffSet = AudioEngine::getActiveCoeffSet(bank);
        const uint32_t genAfter = convo::consumeAtomic(bank.generation, std::memory_order_acquire); // acquire: publishAtomic release と HB

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

    storeLearnedCoeffsToBank(bankIndex, stagedCoefficients);
}

void AudioEngine::getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double* outCoeffs, int maxCoefficients) const noexcept
{
    if (outCoeffs == nullptr || maxCoefficients <= 0)
        return;

    const auto mode = convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire); // acquire: setNoiseShaperLearningMode publishAtomic release と HB
    const int bank = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    const auto& slot = getAdaptiveCoeffBankForIndex(bank);

    for (int retry = 0; retry < 3; ++retry)
    {
        const uint32_t genBefore = convo::consumeAtomic(slot.generation, std::memory_order_acquire); // acquire: bank update publishAtomic release と HB
        const CoeffSet* active = AudioEngine::getActiveCoeffSet(slot);
        const uint32_t genAfter = convo::consumeAtomic(slot.generation, std::memory_order_acquire); // acquire: publishAtomic release と HB

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

    const auto mode = convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire); // acquire: setNoiseShaperLearningMode publishAtomic release と HB
    const int bankIndex = getAdaptiveCoeffBankIndex(sampleRate, bitDepth, mode);
    double stagedCoefficients[kAdaptiveNoiseShaperOrder] = {};
    getAdaptiveCoefficientsForSampleRateAndBitDepth(sampleRate, bitDepth, stagedCoefficients, kAdaptiveNoiseShaperOrder);

    const int limit = std::min(kAdaptiveNoiseShaperOrder, numCoefficients);
    for (int i = 0; i < limit; ++i)
        stagedCoefficients[i] = coeffs[i];

    storeLearnedCoeffsToBank(bankIndex, stagedCoefficients);
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

void AudioEngine::storeLearnedCoeffs(const double* coeffs)
{
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire); // acquire: prepareToPlay/setSampleRate publishAtomic release と HB
    const int bd  = convo::consumeAtomic(ditherBitDepth, std::memory_order_acquire); // acquire: setDitherBitDepth publishAtomic release と HB
    const auto mode = convo::consumeAtomic(pendingLearningMode, std::memory_order_acquire); // acquire: setNoiseShaperLearningMode publishAtomic release と HB
    const int bank = getAdaptiveCoeffBankIndex(sr, bd, mode);

    storeLearnedCoeffsToBank(bank, coeffs);
}

void AudioEngine::storeLearnedCoeffsToBank(int bankIndex, const double* coeffs)
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

[[nodiscard]] bool AudioEngine::getAdaptiveNoiseShaperState(int bankIndex, convo::NoiseShaperLearnerState& outState) const noexcept
{
    const auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    outState = bank.state;
    return true;
}

void AudioEngine::setAdaptiveNoiseShaperState(int bankIndex, const convo::NoiseShaperLearnerState& inState) noexcept
{
    auto& bank = getAdaptiveCoeffBankForIndex(bankIndex);
    std::lock_guard<std::mutex> lock(bank.stateMutex);
    bank.state = inState;
}
