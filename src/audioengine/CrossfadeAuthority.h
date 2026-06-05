#pragma once

#include "AudioEngine.h"

namespace convo::isr {

// CrossfadeAuthority: crossfade の decision と registration を統合する Authority。
// API は RuntimeWorld (RuntimeState) ベース。
// DSPCore からの情報取得は内部実装詳細として隠蔽する。
class CrossfadeAuthority {
public:
    struct Decision {
        bool needsCrossfade = false;
        bool oldHasIR = false;
        bool newHasIR = false;
        double fadeTimeSec = 0.0;
    };

    explicit CrossfadeAuthority() noexcept = default;

    // evaluateAndRegister: decision + registration を統合
    Decision evaluateAndRegister(AudioEngine& engine,
                                 AudioEngine::DSPCore* oldDSP,
                                 AudioEngine::DSPCore* newDSP,
                                 DSPHandle oldHandle,
                                 DSPHandle newHandle) noexcept;

    // evaluateOnly: decision のみ (registration 不要なケース用)
    Decision evaluateOnly(AudioEngine& engine,
                          AudioEngine::DSPCore* oldDSP,
                          AudioEngine::DSPCore* newDSP) noexcept;

    // evaluateFromWorlds: RuntimeWorld投影値ベースの判断 (PR-4)
    // oldWorld/newWorld の dspProjection を使用して判断する。
    // DSPCore 直読より優先して使用すること。
    [[nodiscard]] Decision evaluateFromWorlds(
        const AudioEngine& engine,
        const RuntimePublishWorld& oldWorld,
        const RuntimePublishWorld& newWorld) noexcept;

private:
    // computeDecision: 現在 computeCrossfadeContext ラムダにあるロジック
    Decision computeDecision(const AudioEngine& engine,
                             const AudioEngine::DSPCore* oldDSP,
                             const AudioEngine::DSPCore* newDSP) noexcept;

    // doRegister: crossfadeAuthorityRuntime_.registerCrossfade を内蔵
    void doRegister(AudioEngine& engine,
                    DSPHandle from, DSPHandle to) noexcept;
};

} // namespace convo::isr
