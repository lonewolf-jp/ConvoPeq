#pragma once

#include <JuceHeader.h>

class AudioEngine; // ★ global namespace — AudioEngine is ::AudioEngine (AudioEngine.h:119)

namespace convo::isr {

// ★ FUTURE-9: Dedicated Coordinator Worker (NonRT).
//   Replaces the MessageThread Timer's Scheduling Authority with a dedicated
//   juce::Thread so the Timer becomes observe-only (submitObserve). The worker
//   owns the periodic ISR coordinator cadence: processIntent + overflow drain
//   + deferred publish resubmit (see AudioEngine::runCoordinatorPhase).
//
//   Lifetime: started in AudioEngine::prepareToPlay (Init), joined in
//   stopRebuildThread/StopWorkers (ReleaseResources / CtorDtor) so no
//   publication work races teardown.
class CoordinatorLoop : public juce::Thread
{
public:
    explicit CoordinatorLoop(AudioEngine& engine) noexcept;
    ~CoordinatorLoop() override;

    void startLoop() noexcept;
    void stopLoop() noexcept;

    void run() override;

private:
    // ISR coordinator cadence (NonRT worker tick). juce::Thread::wait() is a
    // blocking sleep (not a spin), so idle polling is ~0% CPU.
    static constexpr int kIntervalMs = 1;

    AudioEngine& engine_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CoordinatorLoop)
};

} // namespace convo::isr
