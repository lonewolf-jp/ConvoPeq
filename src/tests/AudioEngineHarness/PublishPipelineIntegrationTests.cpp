// PublishPipelineIntegrationTests.cpp
// B4 IntegrationTest: real AudioEngine publish pipeline (idle / rebuild / transition / teardown).
//
// 実 AudioEngine + CoordinatorLoop + rebuild thread + audio thread を起動し、
// B4 publish パイプライン (facade → OwnerChannel → IntentQueue → CoordinatorLoop →
// executePublish → RuntimeStore swap → receipt) を 4 シナリオで通す。
//
//   1. rebuild publish (#7): initialize() の Structural rebuild intent が
//      CoordinatorLoop 経由で store-swap されること（実スレッド自動検証）
//   2. idle publish (#2/#5/#6): commitRuntimePublication facade 直呼び出しで
//      enqueue → executePublish → store-swap + Transferred が成立すること
//   3. transition publish (#6): publishIdleWorldOnly(activeDSP, HardReset)
//      （rebuild で構築された active DSP に対して発行）
//   4. teardown publish (#4): releaseResources() が idle publish → receipt →
//      CoordinatorLoop join まで同期完了すること（デッドロックなし）
//
// ビルド: カスタム main() + bool testXxx() パターン（既存テストと同一）

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <chrono>
#include <thread>
#include <string>

#include "AudioEngineHarness.h"
#include "audioengine/RuntimeBuilder.h"

// Work91: soak シナリオ（SoakPublishIntegrationTests.cpp）の前方宣言
namespace convo_soak {
bool runSoakScenarios(bool full, const char* scenario);
}

// DeferredFlowIntegrationTests.cpp (ADR-C4 / design-D4)
int runDeferredFlowIntegrationTests();

// DeferredPublishViewStateMachineTests.cpp (design-D4 不変条件8 / 状態遷移表)
int runDeferredPublishViewStateMachineTests();

namespace {

bool waitUntil(double timeoutSec, const std::function<bool()>& pred)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeoutSec);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

// ── 1. rebuild publish (#7) ──
//   initialize() が投入する Structural rebuild intent は rebuild thread → Orchestrator →
//   facade → CoordinatorLoop → executePublish を経て store を swap する。
//   bootstrap (seq=1) より大きい sequenceId の world が観測されれば完了。
bool testRebuildPublishCompletes()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();
    const auto* world = e.observePublishedWorld();
    if (world == nullptr)
    {
        std::fprintf(stderr, "FAIL: no published world after initialize\n");
        return false;
    }
    const auto bootstrapSeq = world->publication.sequenceId;

    const bool rebuilt = waitUntil(20.0, [&] {
        const auto* w = e.observePublishedWorld();
        return w != nullptr && w->publication.sequenceId > bootstrapSeq;
    });
    if (!rebuilt)
    {
        std::fprintf(stderr, "FAIL: rebuild publish did not swap store within 20s (bootstrap seq=%llu)\n",
                     static_cast<unsigned long long>(bootstrapSeq));
        return false;
    }

    // audio thread が publish と並行稼働していたことを確認
    if (h.blocksProcessed() == 0)
    {
        std::fprintf(stderr, "FAIL: audio thread did not run during rebuild publish\n");
        return false;
    }
    return true;
}

// ── 2. idle publish (#2/#5/#6): facade 直呼び出し ──
//   null-DSP world を commitRuntimePublication に渡し、
//   Transferred + store-swap (seqId 一致) を検証する。
bool testIdlePublishViaFacade()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();

    convo::RuntimeBuilder builder(e);
    auto world = builder.buildRuntimePublishWorld(nullptr,
                                                  nullptr,
                                                  convo::TransitionPolicy::SmoothOnly,
                                                  0.0,
                                                  false);
    if (!world)
    {
        std::fprintf(stderr, "FAIL: could not build idle world\n");
        return false;
    }
    const auto seqId = world->publication.sequenceId;

    const auto result = e.commitRuntimePublication(std::move(world),
                                                   AudioEngine::RegistrationContext::none(),
                                                   convo::isr::DSPHandle::null());
    if (result.stage != convo::PublishStageResult::Success)
    {
        std::fprintf(stderr, "FAIL: idle publish not accepted (stage=%d)\n",
                     static_cast<int>(result.stage));
        return false;
    }
    if (result.ownership != AudioEngine::OwnershipDisposition::Transferred)
    {
        std::fprintf(stderr, "FAIL: idle publish ownership not Transferred (disp=%d)\n",
                     static_cast<int>(result.ownership));
        return false;
    }

    // CoordinatorLoop → executePublish → RuntimeStore swap を確認。
    // ★ work88 (X4-B §6.4 検証): publish パイプラインは FIFO のため、store が seqId 以上に到達
    //   すれば対象 publish の swap は成立（INV-X2-6 contiguous completion）。並行する rebuild
    //   publish がより新しい world で上書きする場合があるため、`== seqId` ではなく `>= seqId`
    //   で検証する（rebuild は正常なエンジン挙動 — 上書きは正しい最終状態）。
    const bool swapped = waitUntil(5.0, [&] {
        const auto* w = e.observePublishedWorld();
        return w != nullptr && w->publication.sequenceId >= seqId;
    });
    if (!swapped)
    {
        std::fprintf(stderr, "FAIL: idle publish did not swap store (seq=%llu)\n",
                     static_cast<unsigned long long>(seqId));
        return false;
    }
    return true;
}

// ── 3. transition publish (#6): publishIdleWorldOnly(activeDSP, HardReset) ──
//   rebuild で構築された active DSP を渡し、HardReset policy の world を
//   CoordinatorLoop 経由で publish する。
bool testTransitionPublish()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();

    // rebuild world の active DSP を取得（handle は production で active 化されないため
    // world 投影値 (RuntimeReadHandle 非依存) から直接解決する）
    convo::isr::DSPHandle activeHandle;
    const bool gotHandle = waitUntil(20.0, [&] {
        const auto* w = e.observePublishedWorld();
        return w != nullptr && w->engine.current != nullptr;
    });
    if (!gotHandle)
    {
        std::fprintf(stderr, "FAIL: no active DSP in world within 20s\n");
        return false;
    }

    // active DSP 解決: handle が有効なら resolve()、無効なら world の engine.current を直接使用
    AudioEngine::DSPCore* activeDSP = nullptr;
    activeHandle = e.dspHandleRuntime().getActiveRuntimeDSPHandle();
    if (!activeHandle.isNull())
    {
        const auto resolved = e.dspHandleRuntime().resolve(activeHandle);
        activeDSP = (resolved.valid) ? static_cast<AudioEngine::DSPCore*>(resolved.instance) : nullptr;
    }
    if (activeDSP == nullptr)
    {
        const auto* w = e.observePublishedWorld();
        activeDSP = static_cast<AudioEngine::DSPCore*>(w->engine.current);
    }
    if (activeDSP == nullptr)
    {
        std::fprintf(stderr, "FAIL: could not resolve active DSP\n");
        return false;
    }

    const auto beforeSeq = e.observePublishedWorld()->publication.sequenceId;
    const bool published = e.publishIdleWorldOnly(activeDSP, convo::TransitionPolicy::HardReset);
    if (!published)
    {
        std::fprintf(stderr, "FAIL: publishIdleWorldOnly returned false\n");
        return false;
    }

    const bool swapped = waitUntil(5.0, [&] {
        const auto* w = e.observePublishedWorld();
        return w != nullptr && w->publication.sequenceId > beforeSeq;
    });
    if (!swapped)
    {
        std::fprintf(stderr, "FAIL: transition publish did not swap store (before=%llu)\n",
                     static_cast<unsigned long long>(beforeSeq));
        return false;
    }
    return true;
}

// ── 4. teardown publish (#4): releaseResources() ──
//   stop() = audio thread join → releaseResources()。releaseResources は内部で
//   idle publish (#4) → waitForPublishReceipt → shutdownCoordinatorLoop join まで
//   同期実行する。デッドロック/ハングがあれば 15s 以内に戻らない。
bool testTeardownPublish()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    const auto t0 = std::chrono::steady_clock::now();
    h.stop();
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    if (elapsed > 15.0)
    {
        std::fprintf(stderr, "FAIL: teardown publish (releaseResources) took %.1fs\n", elapsed);
        return false;
    }
    return true;
}

// ── X2: Publish completion sequence monotonicity（dash §6.2 / INV-X2-5/6）──
//   連続 idle publish を facade 直呼び出しで行い、各 publish の completion が
//   publication sequence order と一致する（単調増加）ことを統合パイプライン
//   （commitRuntimePublication → OwnerChannel → IntentQueue → CoordinatorLoop →
//   executePublish → onPublishCommitted → receipt）で検証する。
//   contiguous completion 前提（PublishExecutor sole gateway + intentQueue_ FIFO）の
//   回帰検証。各 publish の store-swap（observePublishedWorld の seq 一致）を次の
//   publish 前に待つため、crossfade/deferred の影響を受けない。
bool testPublishCompletionMonotonicity()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();

    const auto* initial = e.observePublishedWorld();
    const auto baseSeq = (initial != nullptr) ? initial->publication.sequenceId : 0;

    convo::RuntimeBuilder builder(e);
    convo::isr::PublicationSequenceId lastObserved = baseSeq;

    constexpr int kPublishes = 8;
    for (int i = 0; i < kPublishes; ++i)
    {
        auto world = builder.buildRuntimePublishWorld(nullptr, nullptr,
                                                      convo::TransitionPolicy::SmoothOnly,
                                                      0.0, false);
        if (!world)
        {
            std::fprintf(stderr, "FAIL: X2: could not build idle world (i=%d)\n", i);
            return false;
        }
        const auto seqId = world->publication.sequenceId;
        // seqId 採番（publicationSequenceCounter_）は単調増加（INV-X2-1 の前提）
        if (seqId <= lastObserved)
        {
            std::fprintf(stderr, "FAIL: X2: seqId %llu not > lastObserved %llu\n",
                         static_cast<unsigned long long>(seqId),
                         static_cast<unsigned long long>(lastObserved));
            return false;
        }
        lastObserved = seqId;

        const auto result = e.commitRuntimePublication(std::move(world),
                                                       AudioEngine::RegistrationContext::none(),
                                                       convo::isr::DSPHandle::null());
        if (result.stage != convo::PublishStageResult::Success)
        {
            std::fprintf(stderr, "FAIL: X2: publish rejected (stage=%d)\n",
                         static_cast<int>(result.stage));
            return false;
        }

        // 各 publish の完了が seq order どおりに store-swap される
        //   （contiguous FIFO completion — INV-X2-6: completion order == publication order）
        // ★ work88 (X4-B §6.4 検証): 並行 rebuild publish が store を先へ進める場合があるため
        //   `>= seqId` で検証（FIFO なので seqId 到達 = 対象 publish の swap 成立）。
        const bool swapped = waitUntil(5.0, [&] {
            const auto* w = e.observePublishedWorld();
            return w != nullptr && w->publication.sequenceId >= seqId;
        });
        if (!swapped)
        {
            std::fprintf(stderr, "FAIL: X2: publish %d did not swap store (seq=%llu)\n",
                         i, static_cast<unsigned long long>(seqId));
            return false;
        }
    }

    std::printf("  [PASS] X2: publish completion monotonicity (contiguous FIFO, %d publishes)\n",
                kPublishes);
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    // Work91 §7-3: --soak で長時間（高負荷）シナリオ（S1/S2b/S3/S4/S5）を実行。
    // デフォルト（ctest 用）は下の 4 シナリオのみ = 短時間で green。
    if (argc > 1)
    {
        bool full = false;
        const char* scenario = "all";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a(argv[i]);
            if (a == "--soak")
                full = true;
            else if (a.rfind("--scenario=", 0) == 0)
                scenario = argv[i] + std::strlen("--scenario=");
        }
        return convo_soak::runSoakScenarios(full, scenario) ? 0 : 1;
    }

    if (!testRebuildPublishCompletes())
    {
        std::fprintf(stderr, "FAIL: testRebuildPublishCompletes\n");
        return 1;
    }

    if (!testIdlePublishViaFacade())
    {
        std::fprintf(stderr, "FAIL: testIdlePublishViaFacade\n");
        return 1;
    }

    if (!testTransitionPublish())
    {
        std::fprintf(stderr, "FAIL: testTransitionPublish\n");
        return 1;
    }

    if (!testTeardownPublish())
    {
        std::fprintf(stderr, "FAIL: testTeardownPublish\n");
        return 1;
    }

    if (!testPublishCompletionMonotonicity())
    {
        std::fprintf(stderr, "FAIL: testPublishCompletionMonotonicity\n");
        return 1;
    }

    if (runDeferredFlowIntegrationTests() != 0)
        return 1;

    if (runDeferredPublishViewStateMachineTests() != 0)
        return 1;

    std::printf("AudioEngineHarness: all publish pipeline tests PASS\n");
    return 0;
}
