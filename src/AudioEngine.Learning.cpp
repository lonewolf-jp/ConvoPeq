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
