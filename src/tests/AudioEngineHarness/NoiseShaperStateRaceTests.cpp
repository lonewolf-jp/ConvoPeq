//==============================================================================
// NoiseShaperStateRaceTests.cpp — BUG-1 の回帰テスト（専用ストレステスト）
//
// 目的:
//   AudioEngine::getAdaptiveNoiseShaperState() が書き手と同じ mutex を取ることを、
//   別スレッドで set / get を競合させて検証する。
//
// 検証方法:
//   書き手は世代番号 g を mean / bestCoefficients / covarianceUpperTriangle / iteration の
//   すべてに書き込む。読み手はコピーした State の全要素が同じ g であることを確認する。
//   ロックが無い場合、560 B のコピーが書き込みと重なって世代混在（torn read）になる。
//
// 実行:
//   AudioEngineHarness（環境変数 CONVOPEQ_NS_STATE_RACE_TEST=1）
//   通常のテストスイートには含めない（opt-in）。
//   実行時間は約 1.5 秒で固定する（書き手の反復回数ではなく期限で止める）。
//==============================================================================

#include <JuceHeader.h>

#include "AudioEngine.h"
#include "NoiseShaperLearnerTypes.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int runNoiseShaperStateRaceTests()
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    // 注意: AudioEngine のデストラクタは未 prepare のままでは rebuildThread の join で
    //   待ち続ける（shutdown 設計の既知事項）。本テストはプロセス終了で回収されるよう
    //   意図的に破棄しない。
    auto* engine = new AudioEngine();
    if (engine == nullptr)
        return 1;

    const int bank = AudioEngine::getAdaptiveCoeffBankIndex(
        48000.0, 24, convo::NoiseShaperLearningMode::Short);

    std::atomic<bool> writerDone { false };
    std::atomic<std::uint64_t> writes { 0 };
    std::atomic<std::uint64_t> torn { 0 };
    std::atomic<std::uint64_t> reads { 0 };

    std::thread writer([engine, bank, &writerDone, &writes]()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
        int g = 1;
        while (std::chrono::steady_clock::now() < deadline)
        {
            convo::NoiseShaperLearnerState s;
            for (int i = 0; i < 9; ++i)
            {
                s.mean[i] = static_cast<double> (g);
                s.bestCoefficients[i] = static_cast<double> (g);
            }
            for (int i = 0; i < 45; ++i)
                s.covarianceUpperTriangle[i] = static_cast<double> (g);
            s.iteration = g;
            engine->setAdaptiveNoiseShaperState(bank, s);
            ++g;
            writes.fetch_add(1, std::memory_order_relaxed);
        }
        writerDone.store(true, std::memory_order_release);
    });

    while (! writerDone.load(std::memory_order_acquire))
    {
        convo::NoiseShaperLearnerState out;
        (void) engine->getAdaptiveNoiseShaperState(bank, out);
        reads.fetch_add(1, std::memory_order_relaxed);

        const int g = out.iteration;
        bool consistent = true;
        for (int i = 0; i < 9; ++i)
            consistent = consistent && (out.mean[i] == static_cast<double> (g))
                                    && (out.bestCoefficients[i] == static_cast<double> (g));
        for (int i = 0; i < 45; ++i)
            consistent = consistent && (out.covarianceUpperTriangle[i] == static_cast<double> (g));

        if (! consistent)
            torn.fetch_add(1, std::memory_order_relaxed);
    }

    writer.join();

    const std::uint64_t tornCount = torn.load();
    std::printf("NS-RACE: reads=%llu writes=%llu torn=%llu\n",
                static_cast<unsigned long long> (reads.load()),
                static_cast<unsigned long long> (writes.load()),
                static_cast<unsigned long long> (tornCount));
    std::fflush(stdout);

    if (tornCount != 0)
    {
        std::fprintf(stderr, "NS-RACE: FAIL (%llu torn reads)\n",
                     static_cast<unsigned long long> (tornCount));
        std::fflush(stderr);
        return 1;
    }

    std::printf("NS-RACE: PASS (no torn read)\n");
    std::fflush(stdout);
    return 0;
}
