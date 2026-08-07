#include "ISRShutdown.h"
#include "AudioEngine.h"

namespace convo {
namespace isr {
bool ShutdownRuntime::isShutdownInProgress() const noexcept { return false; }
}
}

void AudioEngine::debugAssertAudioThread() const {}
void AudioEngine::debugAssertNotAudioThread() const {}
