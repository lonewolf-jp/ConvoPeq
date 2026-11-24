//==============================================================================
// NoiseShaperRestoreSyncTests.cpp — B-07 の復旧要求と公開値の同期（専用テスト）
//
// 目的（Issue #278）:
//   1. 学習中の setState()（復元要求）が世代境界で適用されること。
//      すなわち、復元前に sampling した世代の結果が、復元後の状態を上書きしないこと。
//   2. 保存スナップショット（saveLearnedState）に別世代が混入しないこと。
//   3. 公開側（getState / progress）が復元と同時に読まれても世代混在しないこと。
//
// 検証方法:
//   - P1: 復元前の epoch で publishGenerationResult() を呼び、拒否されること
//         （係数と bestScore が復元値のまま）を確認する。
//   - P2: setState() を別スレッドで連続実行しながら saveLearnedState() を繰り返し、
//         XML の mean / 係数 / bestScore / phase / elapsed が単一世代であることを確認する。
//   - P3: 同じ競合下で getState() を繰り返し、返る State が単一世代であることを確認する。
//
// 世代の識別:
//   状態 A と B は全フィールドに固有値を書き込む（mean=m, bestCoefficients=c,
//   bestScore=s, phase=p, elapsed=e, iteration=i）。読み手は mean[0] から世代を決め、
//   他のフィールドが同じ世代かを照合する。
//
// 実行:
//   AudioEngineHarness（環境変数 CONVOPEQ_NS_RESTORE_SYNC_TEST=1）
//   通常のテストスイートには含めない（opt-in）。実行時間は約 1.2 秒で固定する。
//==============================================================================

#include <JuceHeader.h>

#include "AudioEngine.h"
#include "LockFreeRingBuffer.h"
#include "NoiseShaperLearner.h"
#include "NoiseShaperLearnerTypes.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

// ---- NoiseShaperRestoreSyncTestAccess（private の epoch ゲートへ開く定義） ----------
std::uint64_t NoiseShaperRestoreSyncTestAccess::epoch(const NoiseShaperLearner& learner) noexcept
{
    return convo::consumeAtomic(learner.optimizerStateEpoch, std::memory_order_acquire);
}

bool NoiseShaperRestoreSyncTestAccess::publish(NoiseShaperLearner& learner, const double* parcor,
                                               double score, int evaluatedCandidates,
                                               std::uint64_t expectedEpoch,
                                               double safetyMargin) noexcept
{
    return learner.publishGenerationResult(parcor, score, evaluatedCandidates, expectedEpoch,
                                           safetyMargin);
}

double NoiseShaperRestoreSyncTestAccess::bestCoefficient(const NoiseShaperLearner& learner,
                                                        int index) noexcept
{
    return convo::consumeAtomic(learner.bestCoefficients[static_cast<size_t> (index)],
                                std::memory_order_acquire);
}

double NoiseShaperRestoreSyncTestAccess::bestScore(const NoiseShaperLearner& learner) noexcept
{
    return convo::consumeAtomic(learner.progress.bestScore, std::memory_order_acquire);
}

int runNoiseShaperRestoreSyncTests()
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    // 注意: AudioEngine のデストラクタは未 prepare のままでは shutdown を待ち続ける
    //   （既知事項）。NoiseShaperStateRaceTests と同じくプロセス終了で回収する。
    auto* engine = new AudioEngine();
    auto* captureQueue = new LockFreeRingBuffer<AudioBlock, 4096>();
    auto* learner = new NoiseShaperLearner(*engine, *captureQueue);
    if (engine == nullptr || learner == nullptr)
        return 1;

    // ---- 世代 A / B -----------------------------------------------------------
    //   mean は CMA 空間なので ±50 まで許容（loadLearnedState のクランプと同条件）。
    convo::NoiseShaperLearnerState stateA;
    convo::NoiseShaperLearnerState stateB;

    for (int i = 0; i < 9; ++i)
    {
        stateA.mean[i] = 2.0;
        stateA.bestCoefficients[i] = 0.25;
        stateB.mean[i] = 6.0;
        stateB.bestCoefficients[i] = 0.75;
    }
    for (int i = 0; i < 45; ++i)
    {
        stateA.covarianceUpperTriangle[i] = 2.0;
        stateB.covarianceUpperTriangle[i] = 6.0;
    }
    stateA.iteration = 7;
    stateA.bestScore = 11.0;
    stateA.currentPhase = 1;
    stateA.elapsedPlaybackSeconds = 100.0;
    stateA.processCount = 70;
    stateA.totalGenerations = 70;

    stateB.iteration = 9;
    stateB.bestScore = 22.0;
    stateB.currentPhase = 2;
    stateB.elapsedPlaybackSeconds = 200.0;
    stateB.processCount = 90;
    stateB.totalGenerations = 90;

    auto expectedOf = [&] (double mean0) -> const convo::NoiseShaperLearnerState*
    {
        if (mean0 == 2.0) return &stateA;
        if (mean0 == 6.0) return &stateB;
        return nullptr;
    };

    // ---- P1: 復元前の epoch で確定しようとした世代は拒否される ---------------
    std::printf("NS-RESTORE: P1 stale-epoch publish must be rejected\n");
    std::fflush(stdout);

    learner->setState(stateA);
    const std::uint64_t epochBeforeRestore =
        NoiseShaperRestoreSyncTestAccess::epoch(*learner);

    learner->setState(stateB);   // 復元が epoch を進める

    double parcor[9] = {};
    for (int i = 0; i < 9; ++i)
        parcor[i] = 0.5;
    const bool stalePublished =
        NoiseShaperRestoreSyncTestAccess::publish(*learner, parcor, 33.0, 4,
                                                  epochBeforeRestore, 0.85);

    bool p1Ok = (stalePublished == false);
    if (NoiseShaperRestoreSyncTestAccess::bestCoefficient(*learner, 0) != 0.75
        || NoiseShaperRestoreSyncTestAccess::bestScore(*learner) != 22.0)
    {
        p1Ok = false;
    }
    std::printf("NS-RESTORE: P1 stalePublished=%d coeff0=%.3f bestScore=%.3f -> %s\n",
                stalePublished ? 1 : 0,
                NoiseShaperRestoreSyncTestAccess::bestCoefficient(*learner, 0),
                NoiseShaperRestoreSyncTestAccess::bestScore(*learner),
                p1Ok ? "PASS" : "FAIL");
    std::fflush(stdout);

    // ---- P2 / P3: 復元と保存・公開の競合 --------------------------------------
    const auto stateFile = juce::File::getSpecialLocation(juce::File::tempDirectory)
                               .getChildFile("convopeq_ns_restore_sync.xml");

    std::atomic<bool> writerDone { false };
    std::atomic<std::uint64_t> restores { 0 };

    std::thread writer([learner, &writerDone, &restores, &stateA, &stateB]()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1200);
        while (std::chrono::steady_clock::now() < deadline)
        {
            learner->setState(stateA);
            learner->setState(stateB);
            restores.fetch_add(2, std::memory_order_relaxed);
        }
        writerDone.store(true, std::memory_order_release);
    });

    std::uint64_t saves = 0;
    std::uint64_t mixedFiles = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t mixedSnapshots = 0;

    while (! writerDone.load(std::memory_order_acquire))
    {
        // P2: 保存ファイルは単一世代か
        if (learner->saveLearnedState(stateFile))
        {
            ++saves;
            const auto xml = juce::XmlDocument::parse(stateFile);
            bool consistent = (xml != nullptr);

            if (consistent)
            {
                const double mean0 = xml->getChildByName("CmaMean") != nullptr
                                         ? xml->getChildByName("CmaMean")->getDoubleAttribute("m0")
                                         : -1.0;
                const auto* expected = expectedOf(mean0);
                consistent = (expected != nullptr);

                if (consistent && xml->getChildByName("BestCoefficients") != nullptr)
                {
                    const auto* coeffs = xml->getChildByName("BestCoefficients");
                    for (int i = 0; i < 9 && consistent; ++i)
                    {
                        consistent = (coeffs->getDoubleAttribute("c" + juce::String(i))
                                      == expected->bestCoefficients[i]);
                    }
                    for (int i = 0; i < 9 && consistent; ++i)
                    {
                        consistent = (xml->getChildByName("CmaMean")->getDoubleAttribute("m" + juce::String(i))
                                      == expected->mean[i]);
                    }
                    consistent = consistent
                                 && (xml->getDoubleAttribute("bestScore") == expected->bestScore)
                                 && (xml->getIntAttribute("phase") == expected->currentPhase)
                                 && (xml->getDoubleAttribute("elapsedPlaybackSeconds")
                                     == expected->elapsedPlaybackSeconds);
                }
                else
                {
                    consistent = false;
                }
            }

            if (! consistent)
                ++mixedFiles;
        }

        // P3: 公開スナップショットは単一世代か
        convo::NoiseShaperLearnerState snapshot;
        learner->getState(snapshot);
        ++snapshots;

        const auto* expectedSnapshot = expectedOf(snapshot.mean[0]);
        bool snapshotConsistent = (expectedSnapshot != nullptr);
        if (snapshotConsistent)
        {
            for (int i = 0; i < 9 && snapshotConsistent; ++i)
                snapshotConsistent = (snapshot.bestCoefficients[i] == expectedSnapshot->bestCoefficients[i]);
            snapshotConsistent = snapshotConsistent
                                 && (snapshot.bestScore == expectedSnapshot->bestScore)
                                 && (snapshot.currentPhase == expectedSnapshot->currentPhase)
                                 && (snapshot.elapsedPlaybackSeconds == expectedSnapshot->elapsedPlaybackSeconds)
                                 && (snapshot.iteration == expectedSnapshot->iteration);
        }

        if (! snapshotConsistent)
            ++mixedSnapshots;
    }

    writer.join();
    stateFile.deleteFile();

    std::printf("NS-RESTORE: P2 saves=%llu mixedFiles=%llu restores=%llu\n",
                static_cast<unsigned long long> (saves),
                static_cast<unsigned long long> (mixedFiles),
                static_cast<unsigned long long> (restores.load()));
    std::printf("NS-RESTORE: P3 snapshots=%llu mixedSnapshots=%llu\n",
                static_cast<unsigned long long> (snapshots),
                static_cast<unsigned long long> (mixedSnapshots));
    std::fflush(stdout);

    // ループが実際に競合区間へ入ったことを確認する（0 回なら検出力が無い）。
    const bool exercised = (saves >= 20 && snapshots >= 20 && restores.load() >= 20);
    const bool p2Ok = exercised && mixedFiles == 0;
    const bool p3Ok = exercised && mixedSnapshots == 0;

    if (! p1Ok || ! p2Ok || ! p3Ok)
    {
        std::fprintf(stderr, "NS-RESTORE: FAIL (P1=%d P2=%d P3=%d exercised=%d)\n",
                     p1Ok ? 1 : 0, p2Ok ? 1 : 0, p3Ok ? 1 : 0, exercised ? 1 : 0);
        std::fflush(stderr);
        return 1;
    }

    std::printf("NS-RESTORE: PASS (single generation in file, snapshot and publish gate)\n");
    std::fflush(stdout);
    return 0;
}
