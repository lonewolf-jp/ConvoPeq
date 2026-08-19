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

        // ★ E-1.9-B: Event-driven wake with 1ms fallback timeout.
        //   Blocks on drainCv_ until Q/E/T has pending entries (predicate:
        //   pendingRetireCount() != 0 || residentCountAtomic() != 0) or the
        //   1ms timeout expires. This preserves the existing 1ms polling cadence
        //   as a bounded fallback while enabling sub-1ms wake when entries arrive.
        //   Non-RT only (CoordinatorLoop is a juce::Thread, never RT).
        engine_.waitForDrainSignalOrTimeout(kIntervalMs);
    }
}

} // namespace convo::isr
