//==============================================================================
// AdmissionPackedStateTestAccess.h — D101-31-D-3: test-only Friend Test Access
//
// ShutdownRuntime::packedState_ への version 注入（6bit wrap regression test 用）。
// 本番 API は増やさない。Authority 境界を汚染しない。
// ISRShutdown.h 側の friend 宣言は #if defined(CONVOPEQ_UNIT_TESTS) でガードされ、
// Production ビルドではバイナリ無変更
// （AudioEngine.h の DeferredPublicationTestAccess と同一パターン）。
//
// ★ D101-31-D 制約: 本クラスは production authority（packedState_ の単一CAS word契約）
//   を崩さない。テストターゲット（AdmissionPackedStateTests）からのみ使用すること。
//==============================================================================
#pragma once

#include <atomic>
#include <cstdint>

#include "audioengine/ISRShutdown.h"
#include "audioengine/AtomicAccess.h"

namespace convo {
namespace isr {

struct AdmissionPackedStateTestAccess final
{
    static uint32_t load(const ShutdownRuntime& rt) noexcept
    {
        return convo::consumeAtomic(rt.packedState_, std::memory_order_acquire);
    }

    static void store(ShutdownRuntime& rt, uint32_t raw) noexcept
    {
        // テスト前提: シングルスレッド setup 時にのみ呼ぶ（production 契約外の直接書き込み）。
        convo::publishAtomic(rt.packedState_, raw, std::memory_order_release);
    }
};

}  // namespace isr
}  // namespace convo
