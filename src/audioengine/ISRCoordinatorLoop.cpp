#include <JuceHeader.h>
#include "ISRCoordinatorLoop.h"
#include "AudioEngine.h"

namespace convo::isr {

CoordinatorLoop::CoordinatorLoop(AudioEngine& engine) noexcept
    : juce::Thread("ConvoPeq.CoordinatorLoop"), engine_(engine)
{
}

CoordinatorLoop::~CoordinatorLoop()
{
    // juce::Thread asserts !isRunning() at destruction — guarantee exit first.
    stopLoop();
}

void CoordinatorLoop::startLoop() noexcept
{
    if (!isThreadRunning())
        startThread();
}

void CoordinatorLoop::stopLoop() noexcept
{
    signalThreadShouldExit();
    // Bounded join: the loop exits on the next tick once isShutdownInProgress().
    stopThread(2000);
}

void CoordinatorLoop::run()
{
    while (!threadShouldExit())
    {
        if (engine_.isShutdownInProgress())
            break;

        // Non-RT Coordinator phase (processIntent + overflow drain + deferred resubmit).
        engine_.runCoordinatorPhase();

        wait(kIntervalMs);
    }
}

} // namespace convo::isr
