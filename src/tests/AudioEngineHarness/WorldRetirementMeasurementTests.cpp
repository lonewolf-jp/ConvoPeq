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

// ★ T1-MR/G2-1: 前方宣言（実装は G2-1 セクション）。measurement window 外で baseline を安定化する。
bool stabilizeMeasurementBaseline(AudioEngine& e, int stableMs = 400, int timeoutMs = 8000);
bool waitForWorldReclaimCount(AudioEngine& e, std::uint64_t target, int timeoutMs);

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

    // ★ T1-MR: measurement window 外で baseline を安定化（起動直後の未転送破壊を窓外へ排出）。
    if (!stabilizeMeasurementBaseline(e))
    {
        std::fprintf(stderr, "FAIL(%s): baseline never stabilized\n", condition);
        return false;
    }

    // ★ T1-MR raw evidence: window 開始前の baseline snapshot。
    const auto evA0 = e.worldRetirementTelemetry().acquireObserved();
    const auto evR0 = e.worldRetirementTelemetry().releaseObserved();
    const auto evM0 = e.worldRetirementTelemetry().observedOutstandingMax();
    const auto evWc0 = e.worldReclaimCountForMeasurement();
    const auto evRef0 = e.worldRetirementReference().referenceReleaseCount();
    std::printf("[%s] baseline A=%llu R=%llu observedOutstandingMax=%llu worldReclaimCount=%llu "
                "referenceRelease=%llu cadence=%dms publishes=%d interval=%dms\n",
                condition,
                static_cast<unsigned long long>(evA0),
                static_cast<unsigned long long>(evR0),
                static_cast<unsigned long long>(evM0),
                static_cast<unsigned long long>(evWc0),
                static_cast<unsigned long long>(evRef0),
                samplerIntervalMs, publishCount, intervalMs);

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

    // ★ T1-MR: sampler 停止後の残存転送を flush してから end snapshot を取る
    //   （背景回収と transfer の lag による dWc/dRel 不一致を防止）。
    if (!stabilizeMeasurementBaseline(e, /*stableMs=*/200, /*timeoutMs=*/4000))
    {
        std::fprintf(stderr, "FAIL(%s): end flush never stabilized\n", condition);
        return false;
    }

    // ★ T1-MR raw evidence: window 終了後 snapshot + acceptance checks。
    const auto evA1 = e.worldRetirementTelemetry().acquireObserved();
    const auto evR1 = e.worldRetirementTelemetry().releaseObserved();
    const auto evM1 = e.worldRetirementTelemetry().observedOutstandingMax();
    const auto evWc1 = e.worldReclaimCountForMeasurement();
    const auto evRef1 = e.worldRetirementReference().referenceReleaseCount();
    const auto dRel = evR1 - evR0;
    const auto dWc = evWc1 - evWc0;
    const auto dRef = evRef1 - evRef0;

    std::printf("[%s] evidence A_start=%llu R_start=%llu A_end=%llu R_end=%llu "
                "observedOutstandingMax_start=%llu end=%llu "
                "worldReclaimCount_start=%llu end=%llu referenceRelease_start=%llu end=%llu "
                "windowStartUs=%llu windowEndUs=%llu\n",
                condition,
                static_cast<unsigned long long>(evA0), static_cast<unsigned long long>(evR0),
                static_cast<unsigned long long>(evA1), static_cast<unsigned long long>(evR1),
                static_cast<unsigned long long>(evM0), static_cast<unsigned long long>(evM1),
                static_cast<unsigned long long>(evWc0), static_cast<unsigned long long>(evWc1),
                static_cast<unsigned long long>(evRef0), static_cast<unsigned long long>(evRef1),
                static_cast<unsigned long long>(snap.windowStartTimestampUs),
                static_cast<unsigned long long>(snap.windowEndTimestampUs));

    // acceptance: counter conservation / reference consistency / outstanding identity / wrap
    bool ok = true;
    if (dRel != dWc)
    {
        std::fprintf(stderr, "FAIL(%s): counter conservation dRelease(%llu) != dWorldReclaim(%llu)\n",
                     condition, static_cast<unsigned long long>(dRel),
                     static_cast<unsigned long long>(dWc));
        ok = false;
    }
    if (dRef != dWc)
    {
        std::fprintf(stderr, "FAIL(%s): reference consistency dReference(%llu) != dWorldReclaim(%llu)\n",
                     condition, static_cast<unsigned long long>(dRef),
                     static_cast<unsigned long long>(dWc));
        ok = false;
    }
    {
        const auto a = e.worldRetirementTelemetry().acquireObserved();
        const auto r = e.worldRetirementTelemetry().releaseObserved();
        const auto expect = static_cast<std::int64_t>(a) - static_cast<std::int64_t>(r);
        if (e.worldRetirementTelemetry().observedOutstandingEstimate() != expect)
        {
            std::fprintf(stderr, "FAIL(%s): outstanding identity mismatch\n", condition);
            ok = false;
        }
    }
    if (ew != tw - ow)
    {
        std::fprintf(stderr, "FAIL(%s): window identity E_w mismatch\n", condition);
        ok = false;
    }
    if (snap.counterWrapped != 0)
    {
        std::fprintf(stderr, "FAIL(%s): counterWrapped=%llu on valid run\n",
                     condition, static_cast<unsigned long long>(snap.counterWrapped));
        ok = false;
    }
    if (evM1 < evM0)
    {
        std::fprintf(stderr, "FAIL(%s): observedOutstandingMax decreased (%llu < %llu)\n",
                     condition, static_cast<unsigned long long>(evM1),
                     static_cast<unsigned long long>(evM0));
        ok = false;
    }
    return ok;
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

    // ★ T1-MR: measurement window 外で baseline を安定化。
    if (!stabilizeMeasurementBaseline(e))
    {
        std::fprintf(stderr, "FAIL(jitter): baseline never stabilized\n");
        return false;
    }
    const auto evA0 = e.worldRetirementTelemetry().acquireObserved();
    const auto evR0 = e.worldRetirementTelemetry().releaseObserved();
    const auto evM0 = e.worldRetirementTelemetry().observedOutstandingMax();
    const auto evWc0 = e.worldReclaimCountForMeasurement();
    const auto evRef0 = e.worldRetirementReference().referenceReleaseCount();
    std::printf("[jitter] baseline A=%llu R=%llu observedOutstandingMax=%llu worldReclaimCount=%llu "
                "referenceRelease=%llu cadence=100ms publishes=10 interval=irregular\n",
                static_cast<unsigned long long>(evA0),
                static_cast<unsigned long long>(evR0),
                static_cast<unsigned long long>(evM0),
                static_cast<unsigned long long>(evWc0),
                static_cast<unsigned long long>(evRef0));

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

    // ★ T1-MR: sampler 停止後の残存転送を flush してから end snapshot。
    if (!stabilizeMeasurementBaseline(e, /*stableMs=*/200, /*timeoutMs=*/4000))
    {
        std::fprintf(stderr, "FAIL(jitter): end flush never stabilized\n");
        return false;
    }
    const auto evA1 = e.worldRetirementTelemetry().acquireObserved();
    const auto evR1 = e.worldRetirementTelemetry().releaseObserved();
    const auto evM1 = e.worldRetirementTelemetry().observedOutstandingMax();
    const auto evWc1 = e.worldReclaimCountForMeasurement();
    const auto evRef1 = e.worldRetirementReference().referenceReleaseCount();
    const auto dRelJ = evR1 - evR0;
    const auto dWcJ = evWc1 - evWc0;
    const auto dRefJ = evRef1 - evRef0;

    std::printf("[jitter] evidence A_start=%llu R_start=%llu A_end=%llu R_end=%llu "
                "observedOutstandingMax_start=%llu end=%llu "
                "worldReclaimCount_start=%llu end=%llu referenceRelease_start=%llu end=%llu "
                "windowStartUs=%llu windowEndUs=%llu\n",
                static_cast<unsigned long long>(evA0), static_cast<unsigned long long>(evR0),
                static_cast<unsigned long long>(evA1), static_cast<unsigned long long>(evR1),
                static_cast<unsigned long long>(evM0), static_cast<unsigned long long>(evM1),
                static_cast<unsigned long long>(evWc0), static_cast<unsigned long long>(evWc1),
                static_cast<unsigned long long>(evRef0), static_cast<unsigned long long>(evRef1),
                static_cast<unsigned long long>(snap.windowStartTimestampUs),
                static_cast<unsigned long long>(snap.windowEndTimestampUs));

    bool okJ = true;
    if (dRelJ != dWcJ)
    {
        std::fprintf(stderr, "FAIL(jitter): counter conservation dRelease(%llu) != dWorldReclaim(%llu)\n",
                     static_cast<unsigned long long>(dRelJ), static_cast<unsigned long long>(dWcJ));
        okJ = false;
    }
    if (dRefJ != dWcJ)
    {
        std::fprintf(stderr, "FAIL(jitter): reference consistency dReference(%llu) != dWorldReclaim(%llu)\n",
                     static_cast<unsigned long long>(dRefJ), static_cast<unsigned long long>(dWcJ));
        okJ = false;
    }
    if (snap.counterWrapped != 0)
    {
        std::fprintf(stderr, "FAIL(jitter): counterWrapped=%llu on valid run\n",
                     static_cast<unsigned long long>(snap.counterWrapped));
        okJ = false;
    }
    if (evM1 < evM0)
    {
        std::fprintf(stderr, "FAIL(jitter): observedOutstandingMax decreased\n");
        okJ = false;
    }
    return okJ;
}

// ── G2-1: authoritative release observation path の動的検証 ──
//   destruction → worldReclaimCount → sampler delta-transfer → releaseObserved（production 同一 step）。
//   R1 double-count regression guard: ΔreleaseObserved == ΔworldReclaimCount == ΔreferenceReleaseCount。
//   2N になったら即 FAIL（reference observer が releaseObserved に寄与した証拠）。

// reclaim が目標数に達するまで駆動（RT reader の epoch 離脱待ち込み・背景 CoordinatorLoop も吸収）。
bool waitForWorldReclaimCount(AudioEngine& e, std::uint64_t target, int timeoutMs)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline)
    {
        e.driveWorldRetirementReclaimForMeasurement();
        if (e.worldReclaimCountForMeasurement() >= target)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return e.worldReclaimCountForMeasurement() >= target;
}

// 起動時の背景処理（Bootstrap / idle publish の回収・Structural rebuild intent の消化）を
// 測定窓の外に確定させる: reclaim+sampler をポンプし、主要カウンタが stableMs 間連続不変なら安定とみなす。
// 最後に sampler を 1 回余分に回し lastSampled cursor を現在世界破壊数へ同期する（C3 cursor semantics）。
bool stabilizeMeasurementBaseline(AudioEngine& e, int stableMs, int timeoutMs)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    int stableFor = 0;
    std::uint64_t prevRc = 0, prevA = 0, prevRel = 0;
    bool first = true;
    while (std::chrono::steady_clock::now() < deadline)
    {
        e.driveWorldRetirementReclaimForMeasurement();
        e.driveWorldRetirementSamplerForMeasurement();
        const auto rc = e.worldReclaimCountForMeasurement();
        const auto a = e.worldRetirementTelemetry().acquireObserved();
        const auto rel = e.worldRetirementTelemetry().releaseObserved();
        if (!first && rc == prevRc && a == prevA && rel == prevRel)
        {
            stableFor += 50;
            if (stableFor >= stableMs)
            {
                // cursor を現在値へ同期（以降の delta は新規 destruction のみを反映）
                e.driveWorldRetirementSamplerForMeasurement();
                return true;
            }
        }
        else
        {
            stableFor = 0;
        }
        first = false;
        prevRc = rc; prevA = a; prevRel = rel;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

bool runReleaseObservationCase(const char* label, int destructions)
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;
    AudioEngine& e = h.engine();
    convo::RuntimeBuilder builder(e);

    // ★ 背景処理を測定窓外へ排出し、cursor を同期してから baseline を取る。
    if (!stabilizeMeasurementBaseline(e))
    {
        std::fprintf(stderr, "FAIL(%s): baseline never stabilized\n", label);
        return false;
    }

    const auto reclaimBefore = e.worldReclaimCountForMeasurement();
    const auto acquireBefore = e.worldRetirementTelemetry().acquireObserved();
    const auto releaseBefore = e.worldRetirementTelemetry().releaseObserved();
    const auto refReleaseBefore = e.worldRetirementReference().referenceReleaseCount();

    for (int i = 0; i < destructions; ++i)
    {
        if (!publishOnce(e, builder))
        {
            std::fprintf(stderr, "FAIL(%s): publish %d failed\n", label, i);
            return false;
        }
        // terminal destruction を確定させる（epoch 進行 + tryReclaim・type==World deleter 実行）。
        if (!waitForWorldReclaimCount(e, reclaimBefore + static_cast<std::uint64_t>(i + 1), 5000))
        {
            std::fprintf(stderr, "FAIL(%s): worldReclaimCount did not reach %llu (actual=%llu)\n",
                         label,
                         static_cast<unsigned long long>(reclaimBefore + static_cast<std::uint64_t>(i + 1)),
                         static_cast<unsigned long long>(e.worldReclaimCountForMeasurement()));
            return false;
        }
        // ★ G2-1: production timerCallback と同一の delta-transfer を含む sampler step。
        e.driveWorldRetirementSamplerForMeasurement();
    }

    // 最終 flush: 残存 destruction を全て transfer してから after 値を取る。
    if (!stabilizeMeasurementBaseline(e, /*stableMs=*/300, /*timeoutMs=*/5000))
    {
        std::fprintf(stderr, "FAIL(%s): final flush never stabilized\n", label);
        return false;
    }

    const auto reclaimAfter = e.worldReclaimCountForMeasurement();
    const auto acquireAfter = e.worldRetirementTelemetry().acquireObserved();
    const auto releaseAfter = e.worldRetirementTelemetry().releaseObserved();
    const auto refReleaseAfter = e.worldRetirementReference().referenceReleaseCount();

    const auto dReclaim = reclaimAfter - reclaimBefore;
    const auto dAcquire = acquireAfter - acquireBefore;
    const auto dRelease = releaseAfter - releaseBefore;
    const auto dRef = refReleaseAfter - refReleaseBefore;

    std::printf("[%s] N=%d dAcquire=%llu dReclaim=%llu dReference=%llu dRelease=%llu outstanding=%lld\n",
                label, destructions,
                static_cast<unsigned long long>(dAcquire),
                static_cast<unsigned long long>(dReclaim),
                static_cast<unsigned long long>(dRef),
                static_cast<unsigned long long>(dRelease),
                static_cast<long long>(e.worldRetirementTelemetry().observedOutstandingEstimate()));

    // G2-1-A: destruction → worldReclaimCount が実際に増える
    if (dReclaim != static_cast<std::uint64_t>(destructions))
    {
        std::fprintf(stderr, "FAIL(%s): dReclaim(%llu) != N(%d)\n",
                     label, static_cast<unsigned long long>(dReclaim), destructions);
        return false;
    }
    // Step4 / R1 regression guard: releaseObserved は 1:1（2N → double-count 即 FAIL）
    if (dRelease != dReclaim)
    {
        std::fprintf(stderr, "FAIL(%s): dRelease(%llu) != dReclaim(%llu) — R1 double-count or missing transfer\n",
                     label, static_cast<unsigned long long>(dRelease), static_cast<unsigned long long>(dReclaim));
        return false;
    }
    // reference observer は同一 event 群を独立観測するが releaseObserved には加算されない
    if (dRef != dReclaim)
    {
        std::fprintf(stderr, "FAIL(%s): dReference(%llu) != dReclaim(%llu)\n",
                     label, static_cast<unsigned long long>(dRef), static_cast<unsigned long long>(dReclaim));
        return false;
    }
    // successful publish = exactly 1 acquire（背景 publish が無い穏静状態では dAcquire == N）。
    // 背景 Structural rebuild 等が割れた場合は dAcquire > N となるため、下限のみ hard assert。
    if (dAcquire < static_cast<std::uint64_t>(destructions))
    {
        std::fprintf(stderr, "FAIL(%s): dAcquire(%llu) < N(%d)\n",
                     label, static_cast<unsigned long long>(dAcquire), destructions);
        return false;
    }
    // Step5: observedOutstanding = A - R が動的に成立（符号付き恒等式の実測確認）
    const auto a = e.worldRetirementTelemetry().acquireObserved();
    const auto r = e.worldRetirementTelemetry().releaseObserved();
    const auto expectOutstanding = static_cast<std::int64_t>(a) - static_cast<std::int64_t>(r);
    if (e.worldRetirementTelemetry().observedOutstandingEstimate() != expectOutstanding)
    {
        std::fprintf(stderr, "FAIL(%s): outstanding estimate mismatch\n", label);
        return false;
    }
    return true;
}

// Step3: 1 destruction → releaseObserved +1
bool testReleaseObservationSingle()
{
    return runReleaseObservationCase("g2-single", /*destructions=*/1);
}

// Step5: N destruction → N releaseObserved（N=4）
bool testReleaseObservationMultiple()
{
    return runReleaseObservationCase("g2-multi", /*destructions=*/4);
}

// Step6: sampler 再駆動で二重転送しない（duplicate transfer / cursor 更新漏れ / 二重観測の検出）。
//   cursor 同期後は常に「累積転送量 == 累積破壊数」が成立する。これを全 step で検証する
//   （二重転送があれば transferred > destroyed として検出される）。
bool testNoDoubleTransfer()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;
    AudioEngine& e = h.engine();
    convo::RuntimeBuilder builder(e);

    if (!stabilizeMeasurementBaseline(e))
    {
        std::fprintf(stderr, "FAIL(no-double-transfer): baseline never stabilized\n");
        return false;
    }

    const auto releaseStart = e.worldRetirementTelemetry().releaseObserved();
    const auto reclaimStart = e.worldReclaimCountForMeasurement();

    // cursor 同済点での整合: transferred == destroyed
    auto checkConsistency = [&](const char* where) -> bool {
        const auto relDelta = e.worldRetirementTelemetry().releaseObserved() - releaseStart;
        const auto rcDelta = e.worldReclaimCountForMeasurement() - reclaimStart;
        if (relDelta != rcDelta)
        {
            std::fprintf(stderr, "FAIL(no-double-transfer) at %s: releaseDelta(%llu) != reclaimDelta(%llu)\n",
                         where,
                         static_cast<unsigned long long>(relDelta),
                         static_cast<unsigned long long>(rcDelta));
            return false;
        }
        return true;
    };

    for (int round = 0; round < 2; ++round)
    {
        if (!publishOnce(e, builder))
        {
            std::fprintf(stderr, "FAIL(no-double-transfer): publish round %d failed\n", round);
            return false;
        }
        if (!waitForWorldReclaimCount(e, reclaimStart + static_cast<std::uint64_t>(round + 1), 5000))
        {
            std::fprintf(stderr, "FAIL(no-double-transfer): reclaim round %d timed out\n", round);
            return false;
        }
        e.driveWorldRetirementSamplerForMeasurement();  // 新規 destruction 分を transfer
        if (!checkConsistency("after-first-sampler"))
            return false;

        // 直後の再駆動: 新規 destruction が無ければ transferred も増えない（二重転送の直接検出）。
        const auto relBeforeResample = e.worldRetirementTelemetry().releaseObserved();
        const auto rcBeforeResample = e.worldReclaimCountForMeasurement();
        e.driveWorldRetirementSamplerForMeasurement();
        const auto relAfterResample = e.worldRetirementTelemetry().releaseObserved();
        const auto rcAfterResample = e.worldReclaimCountForMeasurement();
        if (rcAfterResample == rcBeforeResample && relAfterResample != relBeforeResample)
        {
            std::fprintf(stderr, "FAIL(no-double-transfer): duplicate transfer detected (+%llu with no new destruction)\n",
                         static_cast<unsigned long long>(relAfterResample - relBeforeResample));
            return false;
        }
        if (!checkConsistency("after-resample"))
            return false;
    }

    // 最終 flush 後も整合が維持されること
    if (!stabilizeMeasurementBaseline(e, /*stableMs=*/300, /*timeoutMs=*/5000))
    {
        std::fprintf(stderr, "FAIL(no-double-transfer): final flush never stabilized\n");
        return false;
    }
    if (!checkConsistency("final"))
        return false;

    std::printf("[no-double-transfer] totalReleaseDelta=%llu totalReclaimDelta=%llu\n",
                static_cast<unsigned long long>(e.worldRetirementTelemetry().releaseObserved() - releaseStart),
                static_cast<unsigned long long>(e.worldReclaimCountForMeasurement() - reclaimStart));
    return true;
}

// ── G2-2: observedOutstandingMax（accumulated live max・15-field #4）の動的検証 ──
//   production と同一の measurement step（transfer + estimate/max + tag + samplerTick + reference 同期）が
//   harness から到達することを証明する。windowMax（window-local sampled max）とは別 semantic object として
//   個別に検証し、両者の値に関する cross-assertion は行わない（D91 基準 8）。
//
//   設計メモ（診断結果に基づく）: 本エンジンは NonRT publish の retired world を同期近傍で破壊するため
//   （diag: burst 後 wc が即時 +K）、outstanding の大きな peak は定常状態では観測不可。そこで本テストは
//   値の大きさではなく契約を検証する:
//     1. writer liveness  — 初回 step 前は max==0（唯一 writer は共有 step 内 updateObservedOutstandingMax）。
//                            初回 step 後 max >= 1 へ決定的に遷移する（生存 World により est >= 1 保証）。
//     2. 単調性契約       — publish/reclaim を交えた複数 step で max が不変または単調増加。
//     3. 減非反応         — drain 完了後（est 低減）の追加 step で max 不変。
//     4. windowTag 経路   — 非shutdown headless で Normal 分類（G2-2 同時修復）。
bool testObservedOutstandingMaxMonotonic()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
        return false;
    AudioEngine& e = h.engine();
    convo::RuntimeBuilder builder(e);

    // ヘッドレスでは JUCE Timer が動かず、harness も未駆動 → この時点で max は初期値 0 のはず。
    // （observedOutstandingMax_ の唯一 writer は共有 measurement step 内 updateObservedOutstandingMax）
    const auto m0 = e.worldRetirementTelemetry().observedOutstandingMax();

    // Step #1（初回）: transfer が起動以来の全破壊を計上し、estimate = A - R >= 1（生存 World 分）を
    //   max へ反映する。0 → >=1 の遷移は writer 実行の決定的証明（背景タイミングに依存しない）。
    e.driveWorldRetirementSamplerForMeasurement();
    const auto m1 = e.worldRetirementTelemetry().observedOutstandingMax();
    const auto est1 = e.worldRetirementTelemetry().observedOutstandingEstimate();

    std::printf("[max-monotonic] M0(before first step)=%lld -> M1=%lld est1=%lld\n",
                static_cast<long long>(m0), static_cast<long long>(m1),
                static_cast<long long>(est1));

    if (m0 != 0)
    {
        std::fprintf(stderr, "FAIL(max-monotonic): pre-step max expected 0 (actual=%lld)\n",
                     static_cast<long long>(m0));
        return false;
    }
    if (m1 <= m0)
    {
        std::fprintf(stderr, "FAIL(max-monotonic): writer did not execute on first step (M1=%lld)\n",
                     static_cast<long long>(m1));
        return false;
    }
    if (m1 < est1)
    {
        std::fprintf(stderr, "FAIL(max-monotonic): max below estimate after first step (M1=%lld est1=%lld)\n",
                     static_cast<long long>(m1), static_cast<long long>(est1));
        return false;
    }

    // 単調性サイクル: publish → reclaim 確定 → step を 4 回（各 step 後に非減少を検証）。
    const auto reclaimStart = e.worldReclaimCountForMeasurement();
    auto mPrev = m1;
    for (int i = 0; i < 4; ++i)
    {
        if (!publishOnce(e, builder))
        {
            std::fprintf(stderr, "FAIL(max-monotonic): cycle publish %d failed\n", i);
            return false;
        }
        if (!waitForWorldReclaimCount(e, reclaimStart + static_cast<std::uint64_t>(i + 1), 5000))
        {
            std::fprintf(stderr, "FAIL(max-monotonic): cycle reclaim %d timed out\n", i);
            return false;
        }
        e.driveWorldRetirementSamplerForMeasurement();
        const auto m = e.worldRetirementTelemetry().observedOutstandingMax();
        if (m < mPrev)
        {
            std::fprintf(stderr, "FAIL(max-monotonic): max decreased (cycle %d: %lld < %lld)\n",
                         i, static_cast<long long>(m), static_cast<long long>(mPrev));
            return false;
        }
        mPrev = m;
    }

    // 減非反応: 全て drain 済み（est 低減済み）の状態で追加 step → max は不変。
    const auto estDrained = e.worldRetirementTelemetry().observedOutstandingEstimate();
    e.driveWorldRetirementSamplerForMeasurement();
    const auto mFinal = e.worldRetirementTelemetry().observedOutstandingMax();

    std::printf("[max-monotonic] afterCycles=%lld estDrained=%lld MFinal=%lld tag=%d\n",
                static_cast<long long>(mPrev), static_cast<long long>(estDrained),
                static_cast<long long>(mFinal),
                static_cast<int>(e.worldRetirementTelemetry().windowTag()));

    if (mFinal != mPrev)
    {
        std::fprintf(stderr, "FAIL(max-monotonic): max changed on drained extra step (%lld != %lld)\n",
                     static_cast<long long>(mFinal), static_cast<long long>(mPrev));
        return false;
    }
    // windowTag 経路（G2-2 同時修復・診断分類）: 非shutdown の headless では Normal に分類される
    if (e.worldRetirementTelemetry().windowTag() != convo::isr::ObservationWindowTag::Normal)
    {
        std::fprintf(stderr, "FAIL(max-monotonic): windowTag not Normal (%d)\n",
                     static_cast<int>(e.worldRetirementTelemetry().windowTag()));
        return false;
    }
    return true;
}

} // namespace

// エントリ: --measurement=normal|burst|jitter|release|max|all（AudioEngineHarness の main から呼ばれる）。
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
    else if (c == "release")
        ok = testReleaseObservationSingle() && testReleaseObservationMultiple() && testNoDoubleTransfer();
    else if (c == "max")
        ok = testObservedOutstandingMaxMonotonic();
    else if (c == "all")
        ok = testNormalMeasurement() && testBurstMeasurement() && testJitterMeasurement()
             && testReleaseObservationSingle() && testReleaseObservationMultiple() && testNoDoubleTransfer()
             && testObservedOutstandingMaxMonotonic();
    else
        std::fprintf(stderr, "unknown measurement condition: %s\n", condition);
    return ok;
}
