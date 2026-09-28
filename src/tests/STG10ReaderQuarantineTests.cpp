// =============================================================================
// STG10ReaderQuarantineTests.cpp — STG-10-D1 (RC-1) 決定論的 regression test
//
// Defect（STG-10 Concrete Defect, Repair Contract Audit で PROVEN）:
//   EpochDomain::quarantineReader() が kQuarantinedFlag を立てた reader slot は
//   slot の封印ではない（registerReaderThread / reserveReaderThread は
//   quarantineFlags を読まず epoch==kInactiveEpoch だけで再割当する）。したがって
//   隔離済みの slot には次の audio block で新規 reader が入場しうる。
//   旧 getMinReaderEpoch() はフラグ単独で除外していたため、その live reader
//   (depth>0) が safe-epoch 計算から抜け、minReaderEpoch が真値より大きくなり、
//   その reader が参照中の entry が reclaim され得た（Release では無言）。
//
// RC-1: 除外条件に depth==0 を追加し、「非参加の隔離 reader のみを除外」にする。
//
// 方針:
//   ・単一スレッド・決定論的。race 再現は不要（S6 は生成経路によらず安全になる）。
//   ・Debug assert の発火を oracle にしない。oracle は getMinReaderEpoch() の
//     戻り値と「deleter が呼ばれたか」のみ。
//   ・production の内部変数名・private state に oracle を依存させない
//     （公開 API のみを使用）。
//   ・各 test に control を置く。「reader が出たら解放される」ことを確認する
//     ことで、test が「そもそも解放可能でない」ことで PASS していないことを示す。
//
// 既存 oracle の改変 = 0。既存 test の期待値変更 = 0。test hook / production 変更 = 0。
// =============================================================================

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#include "core/EpochDomain.h"

using convo::EpochDomain;

// ── Test observer: deleter の起動回数と test 単位の生存数を数える ──────────
//   生存数は test ごとに独立させる。test 間の連鎖汚染で failure の帰属が
//   曖昧になるのを防ぐため（1 test の leak が他 test を落とさない）。
namespace {

struct DeleterTracker
{
    std::atomic<int> invokeCount { 0 };
    std::atomic<int> aliveCount { 0 };
};

struct TestObject
{
    int id;
    DeleterTracker* tracker;

    TestObject(int i, DeleterTracker* t) : id(i), tracker(t) { ++tracker->aliveCount; }
    ~TestObject() { --tracker->aliveCount; }
};

void testDeleter(void* p) noexcept
{
    auto* obj = static_cast<TestObject*>(p);
    ++obj->tracker->invokeCount;
    delete obj;
}

// ── Helper: epoch を E まで進める（publishEpoch を E-1 回呼ぶ） ────────────
void advanceEpoch(EpochDomain& dom, std::uint64_t target)
{
    while (dom.currentEpoch() < target)
        (void) dom.publishEpoch();
}

// ── Helper: epoch をさらに N 回進める ───────────────────────────────────────
void bumpEpoch(EpochDomain& dom, int times)
{
    for (int i = 0; i < times; ++i)
        (void) dom.publishEpoch();
}

// ── Helper: retire entry を 1 件 enqueue する ───────────────────────────────
bool retireOne(EpochDomain& dom, DeleterTracker& tracker, int id, std::uint64_t epoch)
{
    auto* obj = new TestObject(id, &tracker);
    if (!dom.enqueueRetire(obj, &testDeleter, epoch))
    {
        delete obj;
        return false;
    }
    return true;
}

} // namespace

// =============================================================================
// T10-1: quarantined かつ参加中の reader は getMinReaderEpoch() から除外されない
//
// 手順（S6 を決定論的に生成する / race 不使用）:
//   1. epoch を進める
//   2. reader slot を取得
//   3. quarantineReader(i)            ← depth==0 なので即座隔離（flags=0x01）
//   4. enterReader(i)                 ← depth 0→1（= S6）
//   5. epoch をさらに進めて reader の epoch を stale にする
//   6. 過去 epoch（reader の epoch より新しく、currentEpoch より古い）で retire
//   7. getMinReaderEpoch() / tryReclaim()
//   8. control: reader 退出後は同一 entry が解放される
//
// 期待（RC-1 適用後）:
//   minReaderEpoch == reader の entry epoch（= 5）。したがって retireEpoch(7) は
//   isOlder(7, 5)==false で解放不可。deleter 未起動。
// 期待（RC-1 適用前・危険側）:
//   slot が除外され minReaderEpoch == currentEpoch（= 10）。isOlder(7,10)==true で
//   entry が解放され deleter が起動する。
// =============================================================================
static bool testT10_1_quarantinedLiveReaderIsNotExcluded()
{
    EpochDomain dom;
    DeleterTracker tracker;

    advanceEpoch(dom, 5);                                    // currentEpoch == 5
    if (dom.currentEpoch() != 5) return false;

    const int idx = dom.registerReaderThread("stg10-t10-1");
    if (idx < 0) return false;

    // 3. depth==0 なので即座隔離 succeeded になる
    if (!dom.quarantineReader(idx)) return false;
    if (dom.quarantinedReaderCount() != 1) return false;

    // 4. 隔離済みの slot へ入場 → S6 (flags=0x01 && depth>0)
    dom.enterReader(idx);
    if (dom.activeReaderCount() != 1) return false;
    if (dom.getReaderSlotDetail(idx).depth == 0) return false;

    // 5. reader は epoch 5 のまま、globalEpoch は 10 まで進む（滞留 reader を再現）
    bumpEpoch(dom, 5);
    if (dom.currentEpoch() != 10) return false;

    // 6. reader の epoch(5) より新しく、currentEpoch(10) より古い epoch で retire
    if (!retireOne(dom, tracker, 1, 7)) return false;
    if (dom.pendingRetireCount() != 1) return false;

    // 7. 判定: live reader が安全境界に寄与していること
    const std::uint64_t minEpoch = dom.getMinReaderEpoch();
    if (minEpoch != 5)
    {
        std::fprintf(stderr,
                     "T10-1: minReaderEpoch=%llu expected 5 (reader epoch)\n",
                     static_cast<unsigned long long>(minEpoch));
        return false;
    }

    dom.tryReclaim();

    // 8. premature reclaim されていないこと
    if (tracker.invokeCount != 0)
    {
        std::fprintf(stderr, "T10-1: premature reclaim (deleter called while reader active)\n");
        return false;
    }
    if (dom.pendingRetireCount() != 1) return false;
    if (dom.reclaimSuccessCount() != 0) return false;

    // control: reader が出れば同一 entry は解放される（= テストが「解放不能」で
    // PASS していないことの証明）
    dom.exitReader(idx);
    if (dom.activeReaderCount() != 0) return false;
    dom.tryReclaim();
    if (tracker.invokeCount != 1)
    {
        std::fprintf(stderr, "T10-1: control failed (entry not reclaimed after reader exit)\n");
        return false;
    }
    if (dom.pendingRetireCount() != 0) return false;
    if (tracker.aliveCount != 0) return false;

    return true;
}

// =============================================================================
// T10-2: quarantined slot の再利用は禁止しない（Q1/Q4）＋
//        再入場した live reader も安全境界から除外しない（RC-1）
//
// 手順（production の RT block と同じ形）:
//   1. epoch を進める
//   2. slot 取得 → enter → exit（epoch==kInactive, depth==0）
//   3. quarantineReader(i) → S5（flags=0x01 && epoch==kInactive）
//   4. reserveReaderThread(i) が true であること（＝slot は恒久封印ではない）
//   5. enterReader(i) → S6
//   6. epoch を進めて reader を stale にする
//   7. 過去 epoch で retire / tryReclaim → 解放されない
//   8. teardown: unquarantineAllReaders → count==0 / drainAll → 全解放
// =============================================================================
static bool testT10_2_quarantinedSlotIsReusableButLiveReaderProtected()
{
    EpochDomain dom;
    DeleterTracker tracker;

    advanceEpoch(dom, 5);

    const int idx = dom.registerReaderThread("stg10-t10-2");
    if (idx < 0) return false;

    // 2. RT block 1: enter → exit（epoch==kInactive になる）
    dom.enterReader(idx);
    dom.exitReader(idx);
    if (dom.activeReaderCount() != 0) return false;

    // 3. 即座隔離（S5）
    if (!dom.quarantineReader(idx)) return false;
    if (dom.quarantinedReaderCount() != 1) return false;

    // 4. ★ 契約: quarantine は slot の封印ではない。同一 slot を再取得できる。
    if (!dom.reserveReaderThread(idx))
    {
        std::fprintf(stderr, "T10-2: reserveReaderThread on quarantined slot returned false\n");
        return false;
    }

    // 5. RT block 2: 同一 slot へ再入場 → S6
    dom.enterReader(idx);
    if (dom.activeReaderCount() != 1) return false;

    // 6. reader を stale にする
    bumpEpoch(dom, 5);
    if (dom.currentEpoch() != 10) return false;

    // 7. 過去 epoch で retire。live reader がいる限り解放されない
    if (!retireOne(dom, tracker, 2, 7)) return false;
    const std::uint64_t minEpoch = dom.getMinReaderEpoch();
    if (minEpoch != 5)
    {
        std::fprintf(stderr,
                     "T10-2: minReaderEpoch=%llu expected 5 (reused slot's live reader)\n",
                     static_cast<unsigned long long>(minEpoch));
        return false;
    }
    dom.tryReclaim();
    if (tracker.invokeCount != 0)
    {
        std::fprintf(stderr, "T10-2: premature reclaim on reused quarantined slot\n");
        return false;
    }
    if (dom.pendingRetireCount() != 1) return false;

    // 8. teardown（shutdown 相当）
    dom.exitReader(idx);
    dom.unquarantineAllReaders();
    if (dom.quarantinedReaderCount() != 0) return false;
    dom.drainAll();
    if (tracker.invokeCount != 1) return false;
    if (dom.pendingRetireCount() != 0) return false;
    if (tracker.aliveCount != 0) return false;

    return true;
}

// =============================================================================
// T10-3: 正常 quarantine / exit / reclaim 経路の非回帰
//
// 期待（RC-1 適用前後で同一。kPendingQuarantineFlag は除外対象ではない）:
//   ・active 中は reclaim されない
//   ・quarantineReader は depth>0 なので false（deferred）を返す
//   ・pending 中は quarantinedReaderCount()==0
//   ・exit 後に pending→quarantined へ昇格し count==1
//   ・隔離により安全条件が成立し、deleter が呼ばれる
// =============================================================================
static bool testT10_3_deferredQuarantinePathStillUnblocksReclaim()
{
    EpochDomain dom;
    DeleterTracker tracker;

    advanceEpoch(dom, 5);

    const int idx = dom.registerReaderThread("stg10-t10-3");
    if (idx < 0) return false;

    // 参加中
    dom.enterReader(idx);
    if (dom.activeReaderCount() != 1) return false;

    bumpEpoch(dom, 5);
    if (dom.currentEpoch() != 10) return false;

    if (!retireOne(dom, tracker, 3, 7)) return false;

    // 参加中は保護される
    dom.tryReclaim();
    if (tracker.invokeCount != 0) return false;
    if (dom.pendingRetireCount() != 1) return false;

    // depth>0 なので deferred（false）を返す
    if (dom.quarantineReader(idx))
    {
        std::fprintf(stderr, "T10-3: quarantineReader returned true for an active reader\n");
        return false;
    }
    // pending は quarantined として計上されない
    if (dom.quarantinedReaderCount() != 0)
    {
        std::fprintf(stderr, "T10-3: pending quarantine must not be counted as quarantined\n");
        return false;
    }
    // pending 中も保護は維持（除外は kQuarantinedFlag のみが対象）
    dom.tryReclaim();
    if (tracker.invokeCount != 0) return false;
    if (dom.pendingRetireCount() != 1) return false;

    // 最終 exit で pending → quarantined へ昇格
    dom.exitReader(idx);
    if (dom.activeReaderCount() != 0) return false;
    if (dom.quarantinedReaderCount() != 1)
    {
        std::fprintf(stderr, "T10-3: quarantine not promoted on final exit\n");
        return false;
    }

    // 隔離により安全条件が成立し、reclaim が進む（= 機能を生かしている）
    dom.tryReclaim();
    if (tracker.invokeCount != 1)
    {
        std::fprintf(stderr, "T10-3: quarantined reader must unblock reclaim\n");
        return false;
    }
    if (dom.pendingRetireCount() != 0) return false;
    if (tracker.aliveCount != 0) return false;

    return true;
}

// =============================================================================
// T10-4: shutdown unquarantine / drain の非回帰
//
// 期待:
//   ・隔離（count==1）が成立する
//   ・activeReaderCount()==0（ReleaseResources.cpp:432 の shutdown 前提）
//   ・隔離により epoch-safe な entry は tryReclaim で解放される
//   ・unquarantineAllReaders() で count==0
//   ・drainAll() で残存 entry が全解放され pending==0・leak なし
// =============================================================================
static bool testT10_4_shutdownUnquarantineAndDrain()
{
    EpochDomain dom;
    DeleterTracker tracker;

    advanceEpoch(dom, 5);

    const int idx = dom.registerReaderThread("stg10-t10-4");
    if (idx < 0) return false;
    dom.enterReader(idx);
    dom.exitReader(idx);

    if (!dom.quarantineReader(idx)) return false;
    if (dom.quarantinedReaderCount() != 1) return false;

    // shutdown 前提: 全 reader が非参加
    if (dom.activeReaderCount() != 0)
    {
        std::fprintf(stderr, "T10-4: activeReaderCount must be 0 before shutdown drain\n");
        return false;
    }

    // 隔離により epoch-safe な entry は解放される
    if (!retireOne(dom, tracker, 4, 3)) return false;
    dom.tryReclaim();
    if (tracker.invokeCount != 1) return false;
    if (dom.pendingRetireCount() != 0) return false;

    // 非参加 reader しか居ないため、minReaderEpoch == currentEpoch
    bumpEpoch(dom, 5);
    if (dom.currentEpoch() != 10) return false;
    if (dom.getMinReaderEpoch() != 10) return false;

    // 現在の epoch で retire → isOlder(10,10)==false で tryReclaim では解放されない
    if (!retireOne(dom, tracker, 5, 10)) return false;
    dom.tryReclaim();
    if (tracker.invokeCount != 1)
    {
        std::fprintf(stderr, "T10-4: entry with epoch==minReaderEpoch must not be reclaimed early\n");
        return false;
    }
    if (dom.pendingRetireCount() != 1) return false;

    // shutdown: 全 quarantine を解放
    dom.unquarantineAllReaders();
    if (dom.quarantinedReaderCount() != 0)
    {
        std::fprintf(stderr, "T10-4: unquarantineAllReaders did not clear all quarantines\n");
        return false;
    }

    // shutdown: 全 entry を強制 drain
    dom.drainAll();
    if (tracker.invokeCount != 2) return false;
    if (dom.pendingRetireCount() != 0) return false;
    if (tracker.aliveCount != 0)
    {
        std::fprintf(stderr, "T10-4: leaked objects after drain (alive=%d)\n",
                     tracker.aliveCount.load());
        return false;
    }

    return true;
}

int main()
{
    int failures = 0;

    if (!testT10_1_quarantinedLiveReaderIsNotExcluded())
    {
        std::fprintf(stderr, "FAIL: T10-1 quarantined + live reader must not be excluded\n");
        ++failures;
    }
    else
    {
        std::printf("STG10ReaderQuarantineTests: T10-1 PASS (quarantined live reader contributes to minReaderEpoch)\n");
    }

    if (!testT10_2_quarantinedSlotIsReusableButLiveReaderProtected())
    {
        std::fprintf(stderr, "FAIL: T10-2 quarantined slot is reusable but its live reader is protected\n");
        ++failures;
    }
    else
    {
        std::printf("STG10ReaderQuarantineTests: T10-2 PASS (slot reuse allowed, live occupancy protected)\n");
    }

    if (!testT10_3_deferredQuarantinePathStillUnblocksReclaim())
    {
        std::fprintf(stderr, "FAIL: T10-3 deferred quarantine must still unblock reclaim\n");
        ++failures;
    }
    else
    {
        std::printf("STG10ReaderQuarantineTests: T10-3 PASS (pending->quarantined unblocks reclaim)\n");
    }

    if (!testT10_4_shutdownUnquarantineAndDrain())
    {
        std::fprintf(stderr, "FAIL: T10-4 shutdown unquarantine / drain\n");
        ++failures;
    }
    else
    {
        std::printf("STG10ReaderQuarantineTests: T10-4 PASS (shutdown unquarantine + drainAll)\n");
    }

    if (failures != 0)
    {
        std::fprintf(stderr, "STG10ReaderQuarantineTests: %d test(s) FAILED\n", failures);
        throw std::runtime_error("STG10ReaderQuarantineTests failed");
    }

    std::printf("STG10ReaderQuarantineTests: PASS (T10-1/T10-2/T10-3/T10-4)\n");
    return 0;
}
