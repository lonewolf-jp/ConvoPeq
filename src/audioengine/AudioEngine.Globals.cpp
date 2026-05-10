#include <JuceHeader.h>
#include "AudioEngine.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_GLOBALS)

thread_local size_t AudioEngine::tls_readerSlot = SIZE_MAX;
thread_local size_t AudioEngine::DSPCore::tls_readerSlot = SIZE_MAX;

// グローバルインスタンスの定義
// (No global g_deletionQueue anymore)
std::atomic<uint64_t> g_currentEpoch{1};
std::atomic<bool> gShuttingDown{false};

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_GLOBALS)
