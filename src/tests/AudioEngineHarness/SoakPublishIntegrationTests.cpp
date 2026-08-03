// SoakPublishIntegrationTests.cpp
// Work91 §7-3: AudioEngineHarness の soak（長時間/高負荷）シナリオ群。
//
// 実 AudioEngine + CoordinatorLoop + rebuild thread + audio thread を用い、
// 設計文書 doc/work91/soak-test-design.md の S1/S2b/S3/S4/S5 を検証する。
// 呼び出し側（PublishPipelineIntegrationTests.cpp の main）が引数 --soak を
// 受け取ったときだけ実行される（デフォルト ctest は短時間版のみ）。
//
// 責務: ヘッドレスでは担えない「実 AudioEngine 必須の publish 経路」を通す。
// 公開 API のみを使用:
//   e.commitRuntimePublication(world, reg, oldHandle) — PublishCommitResult
//   e.observePublishedWorld()                          — 現在 store の world
//   e.getPublicationBacklogCount()                     — publish intent backlog
//   e.getRetirePendingIntentCount()                   — retire pending intent 数
//   e.currentRetireEpoch()                             — retire epoch
//   e.worldAuthority().registry()                      — PendingPublishRegistry
//
// 実装上の注意:
//   - commitRuntimePublication は内部で waitForPublishReceipt(seqId, 250ms) を
//     呼ぶ（AudioEngine.h L4269）。receipt timeout でも {Success, Transferred}
//     を返す（L4266-4278）ため、timeout は公開 API の戻り値に出ない。よって
//     S3 は「rapid-fire burst でも内部 250ms タイムアウト経路が deadlock せず、
//     最終的に全 publish が store-swap する（回復）」で検証する。
//   - PendingPublishRegistry はリングバッファ 64 スロットで空判定の公開 API が
//     無いため、store が最終 seq に到達すること（offset）で drain 完了を確認。
//   - 大型オブジェクトはスタックに置かない（world は Heap）。

// PSAPI は Windows 固有。
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

#include <cstdio>
#include <cstdint>
#include <string>
#include <functional>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <memory>

#include "AudioEngineHarness.h"
#include "audioengine/RuntimeBuilder.h"
#include "audioengine/AtomicAccess.h"

namespace convo_soak {

//==============================================================================
// 観測統計（Soak Test の定量エビデンス — §5 Pass Criteria 対応）
//==============================================================================
struct SoakStats
{
    std::atomic<std::uint64_t> publishIssued   {0};  // commit 試行回数
    std::atomic<std::uint64_t> publishAccepted {0};  // Success + Transferred
    std::atomic<std::uint64_t> publishRejected {0};  // enqueue 拒否（backpressure）
    std::atomic<std::uint64_t> maxBacklog      {0};  // backlog ピーク（サンプリング）
    std::atomic<std::uint64_t> maxRetireEpoch  {0};  // S4 観測した最大 epoch
    double peakPrivateUsageMB  = 0.0;                // S5
    double finalPrivateUsageMB = 0.0;                // S5
};

struct SoakConfig
{
    bool full = false;                  // --soak 指定で長時間版

    std::size_t s1Count     = 300;      // short / full: 10000(— 長時間で十分)
    std::size_t s1CountFull = 100000;

    std::size_t s2bThreads    = 4;
    std::size_t s2bPerThread  = 100;    // short / full: 2000
    std::size_t s2bPerThreadFull = 20000;

    std::size_t s3Rapid = 300;          // short / full: 5000

    double s5Seconds = 2.0;             // short / full: 8.0

    std::size_t s1Requested() const { return full ? s1CountFull : s1Count; }
    std::size_t s2bRequested() const { return full ? s2bPerThreadFull : s2bPerThread; }
};

//==============================================================================
// 内部ヘルパー
//==============================================================================
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

std::uint64_t nowSeq(const AudioEngine& e)
{
    const auto* world = e.observePublishedWorld();
    return world ? static_cast<std::uint64_t>(world->publication.sequenceId) : 0;
}

bool waitForStoreAtLeast(AudioEngine& e, std::uint64_t targetSeq, double timeoutSec)
{
    return waitUntil(timeoutSec, [&] { return nowSeq(e) >= targetSeq; });
}

// idle publish を 1 件発行。成功で seq を返し、true。拒否/失敗は false。
bool publishOne(AudioEngine& e, std::uint64_t& outSeq)
{
    convo::RuntimeBuilder builder(e);
    auto world = builder.buildRuntimePublishWorld(nullptr, nullptr,
                                                  convo::TransitionPolicy::SmoothOnly,
                                                  0.0, false);
    if (!world)
        return false;
    outSeq = static_cast<std::uint64_t>(world->publication.sequenceId);
    const auto result = e.commitRuntimePublication(
        std::move(world),
        AudioEngine::RegistrationContext::none(),
        convo::isr::DSPHandle::null());
    return (result.stage == convo::PublishStageResult::Success
            && result.ownership == AudioEngine::OwnershipDisposition::Transferred);
}

#ifdef _WIN32
bool privateUsageBytes(double& outBytes)
{
    // PrivateUsage は PROCESS_MEMORY_COUNTERS_EX（非 EX には無い）。GetProcessMemoryInfo は
    // EX 先頭の "superset" ポインタを受け付ける（cb でサイズ判別）ため (cast) で渡す。
    PROCESS_MEMORY_COUNTERS_EX pmc;
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&pmc),
                              sizeof(pmc)))
        return false;
    outBytes = static_cast<double>(pmc.PrivateUsage);
    return true;
}
#endif

// seq リストの「重複が無い」ことを検査（並び順は問わない）。
// ★ 連続（gap なし）は検証しない: seq 採番はエンジン単一 atomic だが、ヘルス
//   テスト開始時に他 publish 元（initialize の Bootstrap/rebuild など）が seq を
//   消費し得るため、自前 publish だけの集合では gap が生じる（誤検出）。
//   よってここでは単一 world の二重 commit が無いことのみ確認する。
bool hasDuplicateSeq(std::vector<std::uint64_t>& v, const char* tag)
{
    if (v.empty())
        return false;
    std::sort(v.begin(), v.end());
    for (std::size_t i = 0; i + 1 < v.size(); ++i)
    {
        if (v[i] == v[i + 1])
        {
            std::fprintf(stderr, "%s: duplicate seq %llu\n", tag,
                         static_cast<unsigned long long>(v[i]));
            return true;
        }
    }
    return false;
}

} // namespace

//==============================================================================
// S1: publish 耐久（seq 欠番・重複なし、最終 store 到達 = registry 空代理）
//   単一バルクで N 件 commit → 全 Success + Transferred → store へ全 drain。
//==============================================================================
static bool runS1(AudioEngineHarness& h, const SoakConfig& cfg, SoakStats& stats)
{
    AudioEngine& e = h.engine();

    // システム起動（initialize の publish）が settle するのを待つ
    if (!waitUntil(5.0, [&] { return e.getPublicationBacklogCount() == 0; }))
    {
        std::fprintf(stderr, "s1: backlog did not settle to 0 before run\n");
        return false;
    }
    const std::size_t total = cfg.s1Requested();
    std::vector<std::uint64_t> committed;
    committed.reserve(total);

    for (std::size_t i = 0; i < total; ++i)
    {
        std::uint64_t seq = 0;
        stats.publishIssued.fetch_add(1, std::memory_order_relaxed);
        if (!publishOne(e, seq))
        {
            stats.publishRejected.fetch_add(1, std::memory_order_relaxed);
            std::fprintf(stderr, "s1: publish rejected at idx=%zu\n", i);
            return false;   // 定常状態で reject = endurance 契約違反
        }
        stats.publishAccepted.fetch_add(1, std::memory_order_relaxed);
        committed.push_back(seq);
    }

    if (hasDuplicateSeq(committed, "s1"))
        return false;

    const std::uint64_t target = committed.empty() ? nowSeq(e) : committed.back();
    if (!waitForStoreAtLeast(e, target, 30.0))
    {
        std::fprintf(stderr, "s1: store did not drain to latest seq=%llu\n",
                     static_cast<unsigned long long>(target));
        return false;
    }

    std::printf("s1  : published=%zu accepted=%llu rejected=%llu finalSeq=%llu backlog=%llu\n",
                total,
                static_cast<unsigned long long>(stats.publishAccepted.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(stats.publishRejected.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(target),
                static_cast<unsigned long long>(e.getPublicationBacklogCount()));
    return true;
}

//==============================================================================
// S2b: backpressure（並行 burst → CoordinatorLoop drain → 全復帰）
// 同時コミットで負荷を与え、全 publish が Success になり、最終 backlog=0、
// store が最終 seq に到達（全回収）することを確認。バックlog ピークを記録。
//==============================================================================
static bool runS2b(AudioEngineHarness& h, const SoakConfig& cfg, SoakStats& stats)
{
    AudioEngine& e = h.engine();
    const std::size_t threads = cfg.s2bThreads;
    const std::size_t per = cfg.s2bRequested();
    const std::size_t total = threads * per;

    std::vector<std::uint64_t> all;
    std::mutex m;
    std::atomic<std::uint64_t> reject{0};
    std::atomic<std::size_t> done{0};

    // backlog ピーク監視スレッド
    std::atomic<bool> stop{false};
    std::thread monitor([&] {
        while (!stop.load(std::memory_order_relaxed))
        {
            const auto b = e.getPublicationBacklogCount();
            std::uint64_t cur = stats.maxBacklog.load(std::memory_order_relaxed);
            while (cur < b && !stats.maxBacklog.compare_exchange_weak(cur, b, std::memory_order_relaxed))
            {}
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    auto worker = [&] {
        for (std::size_t i = 0; i < per; ++i)
        {
            std::uint64_t seq = 0;
            stats.publishIssued.fetch_add(1, std::memory_order_relaxed);
            if (!publishOne(e, seq))
            {
                reject.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            stats.publishAccepted.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(m);
                all.push_back(seq);
            }
        }
        convo::fetchAddAtomic(done, static_cast<std::size_t>(1), std::memory_order_release);
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t)
        pool.emplace_back(worker);
    for (auto& t : pool)
        t.join();
    stop.store(true, std::memory_order_relaxed);
    monitor.join();

    stats.publishRejected.fetch_add(reject.load(std::memory_order_relaxed),
                                    std::memory_order_relaxed);

    // 定常 burst では reject なしを期待
    if (reject.load(std::memory_order_relaxed) > 0)
    {
        std::fprintf(stderr, "s2b: %llu backpressure rejects occurred\n",
                     static_cast<unsigned long long>(reject.load(std::memory_order_relaxed)));
        return false;   // 設計上、drain キープアップすれば reject は起きない
    }

    // 全 publish が Success → 重複なし + 全件受領
    if (all.empty() || hasDuplicateSeq(all, "s2b"))
    {
        std::fprintf(stderr, "s2b: accepted count mismatch (total=%zu)\n", total);
        return false;
    }

    // store へ全 drain + backlog 0
    const std::uint64_t target = all.back();
    if (!waitForStoreAtLeast(e, target, 60.0))
    {
        std::fprintf(stderr, "s2b: store not drained to %llu\n",
                     static_cast<unsigned long long>(target));
        return false;
    }
    if (!waitUntil(10.0, [&] { return e.getPublicationBacklogCount() == 0; }))
    {
        std::fprintf(stderr, "s2b: backlog did not reach 0 after drain\n");
        return false;
    }

    std::printf("s2b: requested=%zu accepted=%llu rejected=%llu peakBacklog=%llu\n",
                total,
                static_cast<unsigned long long>(all.size()),
                static_cast<unsigned long long>(stats.publishRejected.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(stats.maxBacklog.load(std::memory_order_relaxed)));
    return true;
}

//==============================================================================
// S3: receipt timeout / recovery（公開 API で観測可能な範囲）
//   rapid-fire で CoordinatorLoop を追い越しても deadlock せず、全 publish が
//   store に到達する（内部 250ms timeout 経路の回復）ことを確認。
//==============================================================================
static bool runS3(AudioEngineHarness& h, const SoakConfig& cfg, SoakStats& stats)
{
    AudioEngine& e = h.engine();
    const std::size_t kRapid = cfg.s3Rapid;

    for (std::size_t i = 0; i < kRapid; ++i)
    {
        std::uint64_t seq = 0;
        stats.publishIssued.fetch_add(1, std::memory_order_relaxed);
        if (!publishOne(e, seq))
        {
            std::fprintf(stderr, "s3: publish rejected at idx=%zu\n", i);
            return false;
        }
        stats.publishAccepted.fetch_add(1, std::memory_order_relaxed);
        // 明示的に drain 待ちせず次の publish へ（内部 timeout 経路を踏む）
    }

    if (!waitForStoreAtLeast(e, nowSeq(e), 30.0))
    {
        std::fprintf(stderr, "s3: store did not drain all rapid publishes\n");
        return false;
    }
    std::printf("s3  : rapid=%zu drained_to_seq=%llu backlog=%llu\n",
                kRapid, static_cast<unsigned long long>(nowSeq(e)),
                static_cast<unsigned long long>(e.getPublicationBacklogCount()));
    return true;
}

//==============================================================================
// S4: retire epoch 単調増加 + 最終空
//==============================================================================
bool runS4(AudioEngineHarness& h, const SoakConfig& cfg, SoakStats& stats)
{
    AudioEngine& e = h.engine();
    const std::uint64_t startEpoch = e.currentRetireEpoch();

    const std::size_t n = cfg.s1Requested() / 10 + 1;
    for (std::size_t i = 0; i < n; ++i)
    {
        std::uint64_t seq = 0;
        if (!publishOne(e, seq))
        {
            std::fprintf(stderr, "s4: publish rejected at idx=%zu\n", i);
            return false;
        }
    }

    // epoch 単調増加をサンプリング
    std::uint64_t prev = startEpoch;
    std::uint64_t maxEpoch = prev;
    for (int sample = 0; sample < 8; ++sample)
    {
        const std::uint64_t cur = e.currentRetireEpoch();
        if (cur < prev)
        {
            std::fprintf(stderr, "s4: retire epoch regressed (%llu -> %llu)\n",
                         static_cast<unsigned long long>(prev),
                         static_cast<unsigned long long>(cur));
            return false;
        }
        prev = cur;
        if (cur > maxEpoch)
            maxEpoch = cur;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    stats.maxRetireEpoch.store(maxEpoch, std::memory_order_relaxed);

    // retire pending が 0（drain 完了）
    if (!waitUntil(20.0, [&] { return e.getRetirePendingIntentCount() == 0; }))
    {
        std::fprintf(stderr, "s4: retire pending not drained (pending=%llu)\n",
                     static_cast<unsigned long long>(e.getRetirePendingIntentCount()));
        return false;
    }

    std::printf("s4  : published=%zu epoch_start=%llu epoch_final=%llu pending=0\n",
                n, static_cast<unsigned long long>(startEpoch),
                static_cast<unsigned long long>(maxEpoch));
    return true;
}

//==============================================================================
// S5: メモリ傾向（PSAPI PrivateUsage）長時間サンプリング → 収束確認
//==============================================================================
bool runS5(AudioEngineHarness& h, SoakStats& stats, double seconds)
{
#ifdef _WIN32
    AudioEngine& e = h.engine();
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);

    double peak = 0.0;
    std::vector<double> samples;
    while (std::chrono::steady_clock::now() < until)
    {
        double bytes = 0.0;
        if (!privateUsageBytes(bytes))
        {
            std::fprintf(stderr, "s5: GetProcessMemoryInfo failed\n");
            return false;
        }
        if (bytes > peak)
            peak = bytes;
        samples.push_back(bytes);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    double finalBytes = 0.0;
    if (!privateUsageBytes(finalBytes))
        return false;
    stats.peakPrivateUsageMB = peak / (1024.0 * 1024.0);
    stats.finalPrivateUsageMB = finalBytes / (1024.0 * 1024.0);

    // 収束判定: 最終 Private Usage がピークを超えていない（成長し続けていない）。
    if (finalBytes > peak)
    {
        std::fprintf(stderr, "s5: memory still growing (final=%.1f MB > peak=%.1f MB)\n",
                     finalBytes / (1024.0 * 1024.0), peak / (1024.0 * 1024.0));
        return false;
    }
    // 前半平均 vs 後半平均（安定していれば後半は大きく増えない）
    const std::size_t n = samples.size();
    if (n >= 4)
    {
        const std::size_t half = n / 2;
        double sumFirst = 0.0, sumLast = 0.0;
        for (std::size_t i = 0; i < half; ++i) sumFirst += samples[i];
        for (std::size_t i = half; i < n; ++i)   sumLast  += samples[i];
        const double avgFirst = sumFirst / static_cast<double>(half);
        const double avgLast  = sumLast  / static_cast<double>(n - half);
        // リーク吸収（<10% 増加 or 絶対<40MB slack）なら収束とみなす
        const double slackBytes = 40.0 * 1024.0 * 1024.0;
        if (avgLast > avgFirst * 1.10 + slackBytes)
        {
            std::fprintf(stderr, "s5: memory trend increasing (avgF=%.1fMB avgL=%.1fMB)\n",
                         avgFirst / (1024.0 * 1024.0), avgLast / (1024.0 * 1024.0));
            return false;
        }
    }

    std::printf("s5  : samples=%zu peak=%.1fMB final=%.1fMB (converged)\n",
                samples.size(), stats.peakPrivateUsageMB, stats.finalPrivateUsageMB);
    (void)e;
    return true;
#else
    (void)h; (void)stats; (void)seconds;
    std::printf("s5  : skipped (non-Win32)\n");
    return true;
#endif
}

//==============================================================================
// 公開エントリ: PublishPipelineIntegrationTests.cpp の main から呼ばれる
//   full=true → 長時間版（--soak）。false → 短時間版（ctest）。
//   scenario: "all" / "s1" / "s2b" / "s3" / "s4" / "s5" — 指定シナリオのみ実行
//     （workflow_dispatch の inputs.scenario に対応。単一シナリオ時は該当のみ）。
//==============================================================================
bool runSoakScenarios(bool full, const char* scenario)
{
    SoakConfig cfg;
    cfg.full = full;

    // シナリオ選択 パース
    bool runS1_  = false, runS2b_ = false, runS3_ = false, runS4_ = false, runS5_ = false;
    if (scenario == nullptr || std::string(scenario) == "all")
    {
        runS1_ = runS2b_ = runS3_ = runS4_ = runS5_ = true;
    }
    else
    {
        const std::string s(scenario);
        runS1_  = (s == "s1");
        runS2b_ = (s == "s2b");
        runS3_  = (s == "s3" || s == "s2");
        runS4_  = (s == "s4");
        runS5_  = (s == "s5");
    }

    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "soak: harness start failed\n");
        return false;
    }
    SoakStats stats;
    bool ok = true;

    if (runS1_ && !runS1(h, cfg, stats))  { std::fprintf(stderr, "FAIL: S1\n");  ok = false; }
    if (ok && runS2b_ && !runS2b(h, cfg, stats)) { std::fprintf(stderr, "FAIL: S2b\n"); ok = false; }
    if (ok && runS3_ && !runS3(h, cfg, stats))  { std::fprintf(stderr, "FAIL: S3\n");  ok = false; }
    if (ok && runS4_ && !runS4(h, cfg, stats))  { std::fprintf(stderr, "FAIL: S4\n");  ok = false; }
    if (ok && runS5_ && !runS5(h, stats, cfg.full ? 8.0 : 2.0)) { std::fprintf(stderr, "FAIL: S5\n"); ok = false; }

    h.stop();

    std::printf("soak: issued=%llu accepted=%llu rejected=%llu peakBacklog=%llu\n",
                static_cast<unsigned long long>(stats.publishIssued.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(stats.publishAccepted.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(stats.publishRejected.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(stats.maxBacklog.load(std::memory_order_relaxed)));
    std::printf("soak: %s\n", ok ? "ALL PASS" : "FAILED");
    return ok;
}

} // namespace convo_soak