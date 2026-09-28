// STG9ReclaimAccountingTests.cpp — STG-9-D1 / RC-1+R2 regression (T-01 〜 T-04).
//
// 対象欠陥（STG-9-D1）:
//   reclaimInFlightCount_ は deferred 通知ごとに +1 されるが、pending identity の
//   消滅経路（terminal drop / retry 置換）で旧 +1 が解放されず、counter が identity の
//   residency より長生きした。結果 ShutdownScheduler::isFullyDrained()
//   （Coordinator.cpp:542 の ==0 判定）が恒久 false となり waitForDrain が budget 満了し、
//   markShutdownComplete が CoordinatorState::Faulted を書く。
//
// 最終 accounting contract（RC-1 + R2 訂正）:
//   NEW deferred entry : reclaimNormal → onReclaimBegin(+1) + push entry
//   RETRY              : old entry 置換 → new deferred onReclaimBegin(+1) + push new entry
//                        + old entry の onReclaimEnd(-1)。差分 +0（R2 訂正）。
//                        retry を新しい logical obligation として扱わない。
//   SUCCESS            : reclaimNormal → onReclaimEnd(-1) + entry consume
//   TERMINAL DROP      : entry dropped + onReclaimEnd(-1)
//   目標: pending entry ⟷ outstanding reclaim counter unit の 1:1。
//
// Test contract:
//   MASTER（absolute）: drain 実行外の静止点では pending == counter が常に成立する。
//     各 entry は正確に 1 個の +1 を保持し（defer 由来）、消滅時は正確に 1 個の -1 と
//     対になるため。balanced pair（B1/E1・B2/E2）は drain 内部で閉じるため静止点に影響しない。
//   INV-1（baseline-relative）: test-induced pending delta と counter delta の 1:1。
//     符号付き差分で比較し、pre-existing stale entry の正当な同時解決を許容する。
//   INV-2（retry 置換）: retry のみの drain 前後で Δcounter == 0。
//     pending が変化しない drain では counter も変化しないこと。
//
// Stale entry についての注記:
//   停止済み engine の pending list には pre-existing entry が残り得る。その slot が
//   再利用（generation 更新）されている場合、test の state 遷移により stale entry の
//   isRetired guard が反転し、test の drain で同時解決（codrop）され得る。
//   codrop は会計上正しい（stale entry が保持していた +1 の解放）。
//   よって oracle は「絶対値ゼロへの復帰」ではなく以下で構成する:
//     (a) test-induced identity の presence / terminal-drop の直接検証
//     (b) 全静止点での pending == counter（MASTER）
//     (c) baseline に対する符号付き delta の 1:1（INV-1）
//     (d) pure-retry drain の Δcounter == 0（INV-2）
//
// 方針:
//   - 新 CTest 登録なし（既存 AudioEngineHarness exe のサブテスト。STG-8 と同じ形）。
//   - private 観測は既存 friend（DeferredPublicationTestAccess）のテスト専用 accessor のみ。
//     production ヘッダの可視性・production ロジックは変えない。
//   - 決定性のため h.start() → h.stop()（terminal teardown・CoordinatorLoop join 済み）で
//     実行する。単一スレッドのため reclaimNormal は必ず deferred を返す
//     （retireEpoch == currentEpoch >= minReaderEpoch が構造的に成立）。
//     バックグラウンド drain との競合窓を持たない。
//   - DSPCore 実体は作らない。test 用ダミーアドレスを registry に登録するだけで、
//     本テストが使う terminal sink（DSPHandleRuntime::quarantineSlot /
//     reclaimShutdownQuiescent）は instance を dereference しないため安全。
//     AudioEngine::quarantineSlot（= EBR destroy を enqueue する経路）は使わない。
//     理由: EBR 破壊権の取得は別 ownership 系であり、本テストの対象（pending entry の
//     lifecycle と counter の 1:1）には DSPHandleRuntime の state 遷移だけが関与する。
//     使うのは AudioEngine::quarantineSlot の Step 3 と同一の状態遷移である。

#include <cstdint>
#include <cstdio>

#include "AudioEngineHarness.h"
#include "DeferredPublicationTestAccess.h"

namespace {

// ---- 観測ヘルパ -------------------------------------------------------

static std::size_t stg9Pending(AudioEngine& e)
{
    return DeferredPublicationTestAccess::pendingReclaimCount(e);
}

static std::uint64_t stg9Counter(AudioEngine& e)
{
    return DeferredPublicationTestAccess::reclaimInFlightCount(e);
}

static bool stg9Present(AudioEngine& e, const convo::isr::DSPHandle& h)
{
    return DeferredPublicationTestAccess::pendingContains(e, h);
}

// MASTER（absolute）: 静止点では pending == counter。
static bool stg9Master(AudioEngine& e, const char* where)
{
    const auto p = stg9Pending(e);
    const auto c = stg9Counter(e);
    if (p != static_cast<std::size_t>(c))
    {
        std::fprintf(stderr,
            "FAIL: MASTER violated at %s (pending=%zu vs counter=%llu)\n",
            where, p, static_cast<unsigned long long>(c));
        return false;
    }
    return true;
}

// INV-1（baseline-relative）: 符号付き delta の 1:1。
static bool stg9Delta11(AudioEngine& e, std::size_t p0, std::uint64_t c0, const char* where)
{
    const auto p = stg9Pending(e);
    const auto c = stg9Counter(e);
    const auto dp = static_cast<std::int64_t>(p) - static_cast<std::int64_t>(p0);
    const auto dc = static_cast<std::int64_t>(c) - static_cast<std::int64_t>(c0);
    if (dp != dc)
    {
        std::fprintf(stderr,
            "FAIL: INV-1 delta 1:1 broken at %s (pending delta %+lld vs counter delta %+lld) "
            "[now p=%zu c=%llu baseline P0=%zu C0=%llu]\n",
            where, static_cast<long long>(dp), static_cast<long long>(dc),
            p, static_cast<unsigned long long>(c),
            p0, static_cast<unsigned long long>(c0));
        return false;
    }
    return true;
}

// 停止済み・完全に単スレッドな engine で deferred entry を 1 件作る。
//   requestReclaimHandle → reclaimNormal が deferred を返す（+1）→ pending entry 1 件。
//   呼び出し元は一意のダミーアドレスを渡すこと（registerDSPHandleForRuntime は
//   同一アドレスを同一 handle に解決するため、使い回すと別 entry にならない）。
static convo::isr::DSPHandle stg9DeferOne(AudioEngine& e, void* dspAddr)
{
    auto* dsp = static_cast<AudioEngine::DSPCore*>(dspAddr);
    const auto handle = e.registerDSPHandleForRuntime(dsp);
    if (handle.isNull())
    {
        std::fprintf(stderr, "FAIL: stg9DeferOne — handle registration failed\n");
        return convo::isr::DSPHandle::null();
    }
    e.requestReclaimHandle(handle);
    return handle;
}

//==============================================================================
// ★ T-01: terminal sink = quarantine（DSPHandleRuntime::quarantineSlot）
//   baseline(P0/C0) → defer → quarantine → drain。
//   test-induced identity の presence → terminal-drop を直接検証し、
//   全静止点で MASTER + INV-1 を検証する。
//   R1（drop の -1 なし）では MASTER が破れるため FAIL する。
//==============================================================================
static bool checkT01QuarantineDropReleasesCounter()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-9-D1 T-01] harness start failed\n");
        return false;
    }
    h.stop();                       // terminal teardown — 以降バックグラウンド drain なし
    AudioEngine& e = h.engine();

    const auto p0 = stg9Pending(e);
    const auto c0 = stg9Counter(e);
    if (!stg9Master(e, "T-01 baseline"))
        return false;

    static int dummyObject = 0;
    const auto handle = stg9DeferOne(e, &dummyObject);
    if (handle.isNull())
        return false;

    if (!stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-01 deferred identity not present after defer\n");
        return false;
    }
    if (!stg9Master(e, "T-01 after deferred") || !stg9Delta11(e, p0, c0, "T-01 after deferred"))
        return false;

    // terminal sink: quarantine lifecycle へ ownership を移管（state: Retired → Quarantined）
    DeferredPublicationTestAccess::handleRuntime(e).quarantineSlot(handle.slot);

    e.drainDeferredRetireQueues(true);

    // test-induced identity が terminal drop したこと（Quarantined のため list から消滅）
    if (stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-01 test identity was not terminal-dropped\n");
        return false;
    }
    if (!stg9Master(e, "T-01 after terminal drop + drain")
        || !stg9Delta11(e, p0, c0, "T-01 after terminal drop + drain"))
        return false;

    std::printf("STG9ReclaimAccountingTests: T-01 PASS (quarantine drop releases counter)\n");
    return true;
}

//==============================================================================
// ★ T-02: terminal sink = destroyQuarantineSlot（Quarantined → Reclaimed）
//   baseline(P0/C0) → defer → quarantineSlot → destroyQuarantineSlot → drain。
//   test-induced identity の terminal-drop を直接検証する。
//
//   本 sink は production の実経路である（Contract Audit §4.4 の T5/T6:
//   ReleaseResources.cpp:458 の shutdown quarantine cleanup、
//   AudioEngine.Commit.cpp:678 の quarantine 再評価 3 系統①）。
//   終端状態 Reclaimed は reclaimShutdownQuiescent（Coordinator.cpp:776）が作る状態と
//   同一であり、entry の drop 観測（isRetired == false → onReclaimEnd）は同一コード。
//
//   なぜ tryShutdownQuiescentReclaim を直接使わないか:
//   teardown 完了後の engine では Permit が構造的に stale となる（G19/T10 ABA 防止）。
//   実測: fresh proof{shutdown=1 epoch=7} に対し bound{shutdown=1 epoch=5} —
//   teardown が bind 後に epoch を 2 前進させるため、以後いかなる fresh proof も
//   bound identity と一致しない。これは設計どおりの動作であり production の欠陥ではない。
//   shutdown 経路自体の Reclaimed 遷移は既存 unit test testInvX3_4 が担保する。
//   R1（drop の -1 なし）では MASTER が破れるため FAIL する。
//==============================================================================
static bool checkT02DestroyDropReleasesCounter()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-9-D1 T-02] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    const auto p0 = stg9Pending(e);
    const auto c0 = stg9Counter(e);
    if (!stg9Master(e, "T-02 baseline"))
        return false;

    static int dummyObject = 0;
    const auto handle = stg9DeferOne(e, &dummyObject);
    if (handle.isNull())
        return false;

    if (!stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-02 deferred identity not present after defer\n");
        return false;
    }
    if (!stg9Master(e, "T-02 after deferred") || !stg9Delta11(e, p0, c0, "T-02 after deferred"))
        return false;

    // terminal sink: Quarantined → Reclaimed（production T5/T6 と同一の状態遷移）。
    //   destroyQuarantineSlot(slot, 0): generation 照合を skip し、active/fading/crossfade
    //   非関与を確認して DestroyPending 経由で Reclaimed + free-list 返却する。
    auto& rt = DeferredPublicationTestAccess::handleRuntime(e);
    rt.quarantineSlot(handle.slot);
    rt.destroyQuarantineSlot(handle.slot, 0);
    if (rt.isRetired(handle))
    {
        std::fprintf(stderr, "FAIL: T-02 test identity did not reach terminal state\n");
        return false;
    }

    e.drainDeferredRetireQueues(true);

    if (stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-02 test identity was not terminal-dropped\n");
        return false;
    }
    if (!stg9Master(e, "T-02 after destroy + drain")
        || !stg9Delta11(e, p0, c0, "T-02 after destroy + drain"))
        return false;

    std::printf("STG9ReclaimAccountingTests: T-02 PASS (destroy drop releases counter)\n");
    return true;
}

//==============================================================================
// ★ T-03（再定義）: baseline → test operation → test-induced activity →
//   terminal resolution → baseline 復帰。
//   - P0/C0/D0（isFullyDrained）を保存し、test-induced identity の終端後に
//     MASTER + INV-1 が成立することを検証する。
//   - P0 != 0 / C0 != 0 でも FAIL ではない。isFullyDrained() == true を絶対条件にしない。
//   - INV-2: retry のみの drain 前後で (pending, counter) が不変であること。
//   - drain-completion case: terminal resolution 後の追加 drain が fixed point であること。
//==============================================================================
static bool checkT03BaselineReturnOracle()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-9-D1 T-03] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    const auto p0 = stg9Pending(e);
    const auto c0 = stg9Counter(e);
    const bool d0 = e.isFullyDrained();
    if (!stg9Master(e, "T-03 baseline"))
        return false;

    static int dummyObject = 0;
    const auto handle = stg9DeferOne(e, &dummyObject);
    if (handle.isNull())
        return false;

    if (!stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-03 deferred identity not present after defer\n");
        return false;
    }
    if (!stg9Master(e, "T-03 after deferred") || !stg9Delta11(e, p0, c0, "T-03 after deferred"))
        return false;

    // INV-2: retry を 2 回回しても (pending, counter) が不変であること
    for (int i = 0; i < 2; ++i)
    {
        const auto pBefore = stg9Pending(e);
        const auto cBefore = stg9Counter(e);
        e.drainDeferredRetireQueues(true);
        if (stg9Pending(e) != pBefore || stg9Counter(e) != cBefore)
        {
            std::fprintf(stderr,
                "FAIL: T-03 INV-2 retry drain %d changed state "
                "(pending %zu -> %zu, counter %llu -> %llu)\n",
                i, pBefore, stg9Pending(e),
                static_cast<unsigned long long>(cBefore),
                static_cast<unsigned long long>(stg9Counter(e)));
            return false;
        }
        if (!stg9Present(e, handle))
        {
            std::fprintf(stderr, "FAIL: T-03 test identity lost during retry (step %d)\n", i);
            return false;
        }
        if (!stg9Master(e, "T-03 retry holds") || !stg9Delta11(e, p0, c0, "T-03 retry holds"))
            return false;
    }

    // terminal resolution
    DeferredPublicationTestAccess::handleRuntime(e).quarantineSlot(handle.slot);
    e.drainDeferredRetireQueues(true);

    if (stg9Present(e, handle))
    {
        std::fprintf(stderr, "FAIL: T-03 test identity was not terminal-dropped\n");
        return false;
    }
    if (!stg9Master(e, "T-03 after terminal resolution")
        || !stg9Delta11(e, p0, c0, "T-03 after terminal resolution"))
        return false;

    // drain-completion case: 追加 drain が fixed point（何も変化しない）
    {
        const auto pBefore = stg9Pending(e);
        const auto cBefore = stg9Counter(e);
        e.drainDeferredRetireQueues(true);
        if (stg9Pending(e) != pBefore || stg9Counter(e) != cBefore)
        {
            std::fprintf(stderr, "FAIL: T-03 drain is not a fixed point after resolution\n");
            return false;
        }
    }

    // drain 状態は test 前後で不変（test が drain 悪化を持ち込まない）
    if (e.isFullyDrained() != d0)
    {
        std::fprintf(stderr, "FAIL: T-03 isFullyDrained changed by test sequence\n");
        return false;
    }

    std::printf("STG9ReclaimAccountingTests: T-03 PASS (baseline return oracle)\n");
    return true;
}

//==============================================================================
// ★ T-04（最重要）: INV-1 + INV-2 property matrix。
//   defer / retry / quarantine / shutdown reclaim / drain の組合せで、
//   全静止点の MASTER（pending == counter）と INV-1（baseline-relative delta 1:1）、
//   test-induced identity の presence 追跡、pure-retry drain の INV-2 を検証する。
//   retry が新しい +1 として蓄積してはならない。
//
//   注: reclaimNormal の success パス（true 戻り）は単一スレッドの engine level では
//   決定論的に到達不能（retireEpoch == currentEpoch >= minReaderEpoch が構造的に成立し、
//   成功には Coordinator.cpp:698/699 の 2 連続 load 間での epoch 前進が必要）。
//   success 時の -1 は既存 unit test testInv3_1 / testInv3_2（TestEpochProvider による
//   決定論的 epoch 制御・counter 1→0 を assert）が担保する。
//   同様に tryShutdownQuiescentReclaim の teardown 後直接呼びは Permit が構造的に stale
//   となる（fresh proof{epoch=7} 対 bound{epoch=5} を実測 — G19/T10 ABA 防止の設計動作）。
//   よって shutdown-class 終端は destroyQuarantineSlot（Quarantined → Reclaimed）で代替する。
//   終端状態 Reclaimed と drop 観測は同一であり、production T5/T6 sink の実経路である。
//==============================================================================
static bool checkT04Inv1Inv2PropertyMatrix()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-9-D1 T-04] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    const auto p0 = stg9Pending(e);
    const auto c0 = stg9Counter(e);
    if (!stg9Master(e, "T-04 baseline"))
        return false;

    enum Op { kDefer = 0, kDrain = 1, kQuarantine = 2, kDestroy = 3 };
    // 必須 row 0: { defer, drain, drain, drain }（retry のみ。R1 の欠陥を初手で検出）
    // kDestroy = quarantineSlot + destroyQuarantineSlot（Quarantined → Reclaimed）。
    //   shutdown-class 終端（production T5/T6 sink と同一の状態遷移）。
    //   tryShutdownQuiescentReclaim の teardown 後直接呼びは Permit が構造的に stale となるため
    //   （T-02 コメント参照）、決定論的な本経路で代替する。drop 観測は同一コード。
    static const Op kMatrix[][5] = {
        { kDefer, kDrain, kDrain, kDrain },                   // retry のみ
        { kDefer, kDrain, kDrain, kQuarantine, kDrain },      // defer→retry→retry→terminal drop
        { kDefer, kDrain, kQuarantine, kDrain },              // defer→retry→quarantine→drain
        { kDefer, kDrain, kDestroy, kDrain },                 // defer→retry→shutdown-class→drain
        { kDefer, kDestroy, kDrain },                         // defer→shutdown-class→drain
        { kDefer, kDefer, kDrain, kQuarantine, kDrain },      // 複数 deferred + terminal drop
        { kDefer, kDefer, kDrain, kDestroy, kDrain },         // 複数 deferred + shutdown-class drop
        { kDefer, kDrain, kDefer, kDrain, kQuarantine },      // retry 後の追加 defer（closure で drop）
    };
    static const int kSteps[] = { 4, 5, 4, 4, 3, 5, 5, 5 };

    // 各 defer は一意のダミーアドレスを使う（同一アドレスは同一 handle に解決される）。
    static int dummyPool[32] = {};
    int nextObject = 0;

    for (std::size_t row = 0; row < sizeof(kMatrix) / sizeof(kMatrix[0]); ++row)
    {
        bool live[4] = { false, false, false, false };
        convo::isr::DSPHandle handles[4] = {};
        int nLive = 0;

        for (int step = 0; step < kSteps[row]; ++step)
        {
            switch (kMatrix[row][step])
            {
            case kDefer:
            {
                if (nextObject >= 32 || nLive >= 4)
                {
                    std::fprintf(stderr, "FAIL: T-04 row %zu step %d pool exhausted\n",
                        row, step);
                    return false;
                }
                const auto h2 = stg9DeferOne(e, &dummyPool[nextObject]);
                ++nextObject;
                if (h2.isNull())
                    return false;
                handles[nLive] = h2;
                live[nLive] = true;
                ++nLive;
                if (!stg9Present(e, h2))
                {
                    std::fprintf(stderr, "FAIL: T-04 row %zu step %d deferred identity absent\n",
                        row, step);
                    return false;
                }
                break;
            }
            case kDrain:
            {
                const auto pBefore = stg9Pending(e);
                const auto cBefore = stg9Counter(e);
                e.drainDeferredRetireQueues(true);
                // terminal 解決された test identity を live 追跡から外す。
                // drop / success は会計上同一（entry 消費 + counter -1）であり、
                // !isRetired でも presence でも両方で確認する。
                bool terminalResolved = false;
                for (int k = 0; k < nLive; ++k)
                {
                    if (!live[k])
                        continue;
                    if (stg9Present(e, handles[k]))
                        continue;   // 依然 present = retry 置換（Δ0）
                    live[k] = false;
                    terminalResolved = true;
                }
                // INV-2: terminal 解決を伴わない drain では (pending, counter) 不変
                if (!terminalResolved
                    && (stg9Pending(e) != pBefore || stg9Counter(e) != cBefore))
                {
                    std::fprintf(stderr,
                        "FAIL: T-04 row %zu step %d INV-2 pure-retry drain changed state "
                        "(pending %zu -> %zu, counter %llu -> %llu)\n",
                        row, step, pBefore, stg9Pending(e),
                        static_cast<unsigned long long>(cBefore),
                        static_cast<unsigned long long>(stg9Counter(e)));
                    return false;
                }
                break;
            }
            case kQuarantine:
                for (int k = 0; k < nLive; ++k)
                {
                    if (!live[k]) continue;
                    DeferredPublicationTestAccess::handleRuntime(e).quarantineSlot(handles[k].slot);
                }
                break;
            case kDestroy:
                for (int k = 0; k < nLive; ++k)
                {
                    if (!live[k]) continue;
                    auto& rt = DeferredPublicationTestAccess::handleRuntime(e);
                    rt.quarantineSlot(handles[k].slot);
                    rt.destroyQuarantineSlot(handles[k].slot, 0);
                    if (rt.isRetired(handles[k]))
                    {
                        std::fprintf(stderr,
                            "FAIL: T-04 row %zu step %d destroy did not terminalize\n",
                            row, step);
                        return false;
                    }
                }
                break;
            }

            // ★ MASTER + INV-1 を毎ステップで検査
            if (!stg9Master(e, "T-04 mid") || !stg9Delta11(e, p0, c0, "T-04 mid"))
            {
                std::fprintf(stderr, "  (row %zu step %d)\n", row, step);
                return false;
            }
        }

        // row closure: 残存 live を quarantine → drain で必ず閉じる
        for (int k = 0; k < nLive; ++k)
        {
            if (live[k])
                DeferredPublicationTestAccess::handleRuntime(e).quarantineSlot(handles[k].slot);
        }
        e.drainDeferredRetireQueues(true);

        for (int k = 0; k < nLive; ++k)
        {
            if (live[k] && stg9Present(e, handles[k]))
            {
                std::fprintf(stderr, "FAIL: T-04 row %zu closure leaked identity %d\n", row, k);
                return false;
            }
        }
        if (!stg9Master(e, "T-04 row closure") || !stg9Delta11(e, p0, c0, "T-04 row closure"))
        {
            std::fprintf(stderr, "  (row %zu)\n", row);
            return false;
        }
    }

    std::printf("STG9ReclaimAccountingTests: T-04 PASS (INV-1/INV-2 property matrix)\n");
    return true;
}

} // namespace

// main 側（PublishPipelineIntegrationTests.cpp）から呼ばれるエントリ。
// 新 CTest 登録なし（AudioEngineHarness 既存 exe のサブテスト）。
int runSTG9ReclaimAccountingTests()
{
    bool ok = true;
    if (!checkT01QuarantineDropReleasesCounter())
    {
        std::fprintf(stderr, "FAIL: checkT01QuarantineDropReleasesCounter\n");
        ok = false;
    }
    if (!checkT02DestroyDropReleasesCounter())
    {
        std::fprintf(stderr, "FAIL: checkT02DestroyDropReleasesCounter\n");
        ok = false;
    }
    if (!checkT03BaselineReturnOracle())
    {
        std::fprintf(stderr, "FAIL: checkT03BaselineReturnOracle\n");
        ok = false;
    }
    if (!checkT04Inv1Inv2PropertyMatrix())
    {
        std::fprintf(stderr, "FAIL: checkT04Inv1Inv2PropertyMatrix\n");
        ok = false;
    }
    if (ok)
        std::printf("STG9ReclaimAccountingTests: PASS (T-01/T-02/T-03/T-04)\n");
    return ok ? 0 : 1;
}
