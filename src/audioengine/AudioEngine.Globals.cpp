#include <JuceHeader.h>
#include "AudioEngine.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_GLOBALS)

// グローバルインスタンスの定義
// (No global g_deletionQueue anymore)
std::atomic<uint64_t> g_currentEpoch{1};

#endif // defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_GLOBALS)
