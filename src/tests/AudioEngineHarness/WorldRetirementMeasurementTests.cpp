// WorldRetirementMeasurementTests.cpp
// ★ T1 (D99/D100): burst test harness — O_w（100ms sampler）と T_w（event-driven reference observer）の差
//   E_w = T_w - O_w を 3 条件（normal / burst / jitter）で測定する。
//
//   - Normal: 通常の publish → retire 反復（O_w と T_w の通常時差）
//   - Burst（本命）: sampler interval（100ms）より短い時間幅に retire を集中（T_w > O_w を意図的に生成できるか）
//   - Jitter: 負荷を変えて sampler の実測 gap / missed tick と E_w の関係を見る
//
//   M の判定: M = max(E_w) で終了しない（有限回の実測最大値は安全側上界ではない・D94/D95）。
//   ここでは O_w / T_w / E_w と sampling stats を記録し、measurement model 評価の入力とする。

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

#include "AudioEngineHarness.h"
#include "audioengine/RuntimeBuilder.h"
#include "audioengine/ISRWorldRetirementTelemetry.h"   // MeasurementSnapshot / MeasurementState

namespace {

// publish を 1 回発生させる（buildRuntimePublishWorld + commitRuntimePublication・facade 直呼び出し）。
bool publishOnce(AudioEngine& e, convo::RuntimeBuilder& builder)
{
    auto world = builder.buildRuntimePublishWorld(nullptr, nullptr,
                                                  convo::TransitionPolicy::SmoothOnly,
                                                  0.0, false);
    if (!world)
        return false;
    const auto result = e.commitRuntimePublication(std::move(world),
                                                   AudioEngine::RegistrationContext::none(),
                                                   convo::isr::DSPHandle::null());
    return result.stage == convo::PublishStageResult::Success;
}

// Closed snapshot（O_w を含む）を待つ（End 後に sampler が window を確定するまで）。
bool waitForClosed(const AudioEngine& e, int timeoutMs, convo::isr::MeasurementSnapshot& snap)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline)
    {
        snap = e.worldRetirementTelemetry().lastClosedSnapshot();
        if (snap.valid != 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// 3 条件共通の測定: Start → publish 反復 → release 待ち → End → Closed snapshot → O_w/T_w/E_w 記録・検証。
// ★ D100.5: sampler を 100ms 固定 cadence で独立バックグラウンドスレッドから駆動する。
//
//   M の判定: M = max(E_w) で終了しない（有限回の実測最大値は安全側上界ではない・D94/D95）。
//   ここでは O_w / T_w / E_w と sampling stats を記録し、measurement model 評価の入力とする。
//   publish/reclaim ループとは分離し、sampler が peak を捕捉しないタイミングで publish が走るようにする。
//   これにより O_w（sampled） < T_w（event-driven max）を意図的に生成できる。
bool runMeasurement(const char* condition, AudioEngineHarness& h,
                    int publishCount, int intervalMs, int samplerIntervalMs)
{
    AudioEngine& e = h.engine();
    convo::RuntimeBuilder builder(e);

    // Start request（Non-RT・sampler が次 tick で window 開始）
    e.requestWorldRetirementMeasurementStart();

    // ★ D100.5: sampler を独立スレッドで 100ms カデンスで駆動（publish ループと分離）。
    //   JUCE Timer はヘッドレスで動かないため、std::jthread で手動駆動。
    std::atomic<bool> stopSampler{false};
    std::jthread samplerThread([&e, samplerIntervalMs, &stopSampler]() {
        auto next = std::chrono::steady_clock::now();
        while (!stopSampler.load(std::memory_order_relaxed)) {
            next += std::chrono::milliseconds(samplerIntervalMs);
            std::this_thread::sleep_until(next);
            // ★ D100.5: sampler tick のみ — reclaim は publish ループで同 iteration 駆動（peak miss）
            e.driveWorldRetirementSamplerForMeasurement();
        }
    });

    // ★ sampler が Start を観測して window を開始するのを待つ（Running 遷移）
    const auto startDeadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(2000);
    while (e.worldRetirementTelemetry().measurementState()
               != convo::isr::MeasurementState::Running
        && std::chrono::steady_clock::now() < startDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (e.worldRetirementTelemetry().measurementState()
        != convo::isr::MeasurementState::Running)
    {
        stopSampler.store(true, std::memory_order_relaxed);
        std::fprintf(stderr, "FAIL(%s): measurement never started (state=%d)\n",
                     condition,
                     static_cast<int>(e.worldRetirementTelemetry().measurementState()));
        return false;
    }

    for (int i = 0; i < publishCount; ++i)
    {
        if (!publishOnce(e, builder))
        {
            stopSampler.store(true, std::memory_order_relaxed);
            std::fprintf(stderr, "FAIL(%s): publish %d failed\n", condition, i);
            return false;
        }
        // ★ D100.5: sampler は独立スレッドで駆動中 — publish ループでは駆動しない。
        //   publish + reclaim を同じ iteration で駆動 → acquire 直後の peak を sampler が miss する。
        e.driveWorldRetirementReclaimForMeasurement();
        if (intervalMs > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
    }

    // ★ 診断: publish 後の acquire / release 観測値
    std::printf("[%s] afterPublish acquireObserved=%llu referenceAcquire=%llu referenceRelease=%llu\n",
                condition,
                static_cast<unsigned long long>(e.worldRetirementTelemetry().acquireObserved()),
                static_cast<unsigned long long>(e.worldRetirementReference().referenceAcquireCount()),
                static_cast<unsigned long long>(e.worldRetirementReference().referenceReleaseCount()));

    // release イベント（deferred delete・epoch 安全到達後）が発生するのを待つ
    // ★ D100.4: epoch 進行 + tryReclaim は独立 sampler スレッドで駆動中だが、
    //   念のため追加で少数回駆動して type==World の terminal deleter（onRelease）を確実に発生させる。
    for (int i = 0; i < 16; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // End request（Non-RT・sampler が次 tick で window 確定）
    e.requestWorldRetirementMeasurementEnd();

    // ★: 独立 sampler スレッドが End を観測して window を確定するまで少し待つ
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stopSampler.store(true, std::memory_order_relaxed);
    samplerThread.join();

    // ★ 診断: End 後の state
    std::printf("[%s] state after End = %d\n", condition,
                static_cast<int>(e.worldRetirementTelemetry().measurementState()));

    convo::isr::MeasurementSnapshot snap;
    if (!waitForClosed(e, 3000, snap))
    {
        std::fprintf(stderr, "FAIL(%s): no closed snapshot\n", condition);
        return false;
    }

    // O_w（sampled windowMax）・T_w（reference max）・E_w = T_w - O_w
    const auto ow = snap.windowMax;
    const auto tw = e.worldRetirementReference().referenceMax();
    const auto ew = tw - ow;

    std::printf("[%s] O_w=%lld T_w=%lld E_w=%lld windowId=%llu "
                "sampleCount=%llu maxSamplingGapUs=%llu missedTickCount=%llu counterWrapped=%llu\n",
                condition,
                static_cast<long long>(ow),
                static_cast<long long>(tw),
                static_cast<long long>(ew),
                static_cast<unsigned long long>(snap.windowId),
                static_cast<unsigned long long>(snap.sampleCount),
                static_cast<unsigned long long>(snap.maxSamplingGapUs),
                static_cast<unsigned long long>(snap.missedTickCount),
                static_cast<unsigned long long>(snap.counterWrapped));

    // 検証: T_w >= O_w（reference は sampler 以上に観測する）。Burst では T_w > O_w を意図的に生成できるか確認。
    if (tw < ow)
    {
        std::fprintf(stderr, "FAIL(%s): T_w(%lld) < O_w(%lld)\n", condition, tw, ow);
        return false;
    }
    return true;
}

// ── 1. Normal: 通常の publish → retire 反復 ──
bool testNormalMeasurement()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;
    // Normal: sampler と publish が同等の cadence（100ms）→ O_w ≈ T_w
    return runMeasurement("normal", h, /*publishCount=*/8, /*intervalMs=*/150, /*samplerIntervalMs=*/100);
}

// ── 2. Burst（本命）: sampler interval（100ms）より短い時間幅に retire を集中 ──
//   ★ D100.5: intervalMs=150（sampler tick 100ms より長い）で publish+reclaim を
//     sampler tick の**前**に完結させる。sampler tick は peak（publish 直後の
//     outstanding=1）を miss する → O_w=0, T_w=1 → E_w=1 > 0。
//   intervalMs=0 では 20 回の publish が 1 tick より前に完結し sampler が peak を
//     捕捉して E_w=0 になってしまう（前回の問題）。
bool testBurstMeasurement()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;
    // interval 150ms で publish+reclaim → sampler tick（100ms カデンス）は peak を miss する
    // → O_w（sampled）が T_w（event-driven max）に達しない（E_w > 0）
    return runMeasurement("burst", h, /*publishCount=*/20, /*intervalMs=*/150, /*samplerIntervalMs=*/100);
}

// ── 3. Scheduler jitter: 負荷を変える（不規則な interval）──
bool testJitterMeasurement()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;

    AudioEngine& e = h.engine();
    convo::RuntimeBuilder builder(e);

    e.requestWorldRetirementMeasurementStart();

    // ★ T1 (D100): sampler を手動駆動（Start を観測して window 開始）
    e.driveWorldRetirementSamplerForMeasurement();
    if (e.worldRetirementTelemetry().measurementState() != convo::isr::MeasurementState::Running)
    {
        std::fprintf(stderr, "FAIL(jitter): measurement never started\n");
        return false;
    }

    // ★ D100.5: sampler を独立スレッドで 100ms カデンスで駆動（publish loop と分離）。
    std::atomic<bool> stopSampler{false};
    std::jthread samplerThread([&e, &stopSampler]() {
        auto next = std::chrono::steady_clock::now();
        while (!stopSampler.load(std::memory_order_relaxed)) {
            next += std::chrono::milliseconds(100);
            std::this_thread::sleep_until(next);
            // ★ D100.5: sampler tick のみ — reclaim は publish ループで同 iteration 駆動（peak miss）
            e.driveWorldRetirementSamplerForMeasurement();
        }
    });

    const int intervals[] = { 0, 80, 200, 0, 120, 300, 0, 60, 180, 0 };
    for (int it : intervals)
    {
        if (!publishOnce(e, builder))
        {
            stopSampler.store(true, std::memory_order_relaxed);
            return false;
        }
        // ★ D100.5: sampler は独立スレッドで駆動中 — publish ループでは駆動しない。
        if (it > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(it));
    }

    // ★ D100.4: 残りの retire を確実に release させる（audio thread の reader exit 待ち込み）。
    //   sampler スレッドが tryReclaim を駆動しているため、ここでは待機のみ。
    for (int i = 0; i < 16; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    e.requestWorldRetirementMeasurementEnd();

    // ★: 独立 sampler スレッドが End を観測して window を確定するまで待つ
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stopSampler.store(true, std::memory_order_relaxed);
    samplerThread.join();

    convo::isr::MeasurementSnapshot snap;
    if (!waitForClosed(e, 3000, snap))
    {
        std::fprintf(stderr, "FAIL(jitter): no closed snapshot\n");
        return false;
    }

    const auto ow = snap.windowMax;
    const auto tw = e.worldRetirementReference().referenceMax();
    const auto ew = tw - ow;

    std::printf("[jitter] O_w=%lld T_w=%lld E_w=%lld windowId=%llu "
                "sampleCount=%llu maxSamplingGapUs=%llu missedTickCount=%llu counterWrapped=%llu\n",
                static_cast<long long>(ow),
                static_cast<long long>(tw),
                static_cast<long long>(ew),
                static_cast<unsigned long long>(snap.windowId),
                static_cast<unsigned long long>(snap.sampleCount),
                static_cast<unsigned long long>(snap.maxSamplingGapUs),
                static_cast<unsigned long long>(snap.missedTickCount),
                static_cast<unsigned long long>(snap.counterWrapped));

    if (tw < ow)
    {
        std::fprintf(stderr, "FAIL(jitter): T_w(%lld) < O_w(%lld)\n", tw, ow);
        return false;
    }
    return true;
}

} // namespace

// エントリ: --measurement=normal|burst|jitter|all（AudioEngineHarness の main から呼ばれる）。
bool runWorldRetirementMeasurement(const char* condition)
{
    // ★ T1 (D100): JUCE Timer はヘッドレスで動かないため、MessageManager は起動しない。
    //   sampler は driveWorldRetirementSamplerForMeasurement() で手動駆動する。

    const std::string c(condition ? condition : "all");
    bool ok = false;
    if (c == "normal")
        ok = testNormalMeasurement();
    else if (c == "burst")
        ok = testBurstMeasurement();
    else if (c == "jitter")
        ok = testJitterMeasurement();
    else if (c == "all")
        ok = testNormalMeasurement() && testBurstMeasurement() && testJitterMeasurement();
    else
        std::fprintf(stderr, "unknown measurement condition: %s\n", condition);
    return ok;
}
