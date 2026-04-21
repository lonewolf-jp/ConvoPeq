// src/SafeStateSwapper.h  ── v6.0 Final
// Phase 0: Multi-Reader 対応 Epoch-based RCU コア実装
//
// ■ 設計思想（理論的完全性）
//
//   [RCU 安全性の核心原理]
//   UAF（Use-After-Free）防止の絶対的保証は getMinReaderEpoch() による
//   厳密な判定ロジックに依存します。
//   条件:  「解放対象エントリのエポック < 現在アクティブな全 Reader の最小エポック」
//   が成立する場合にのみ解放を許可します。
//
//   [Quiescent State の定義]
//   Reader が exitReader() を呼び出し、自身のエポックを kIdleEpoch (UINT64_MAX) に
//   戻した状態を「非参加（静寂状態）」と見なします。
//   これにより「参加していないスレッド」と「古いエポックで停止中のスレッド」を
//   区別し、誤った解放ブロックを防ぎます。
//
//   [2-step bump の役割（補助的ヒューリスティック）]
//   swap() 時に 2 回エポックを進める処理の目的:
//     1. エポック値の単調増加を保証し、論理的な順序性を維持する。
//     2. 極端なタイミングでの Reader エポックと Writer エポックの乖離を吸収し、
//        reclaim 処理の効率性を高める補助的手段。
//   「2-step bump が安全を保証する」のではなく、
//   「minReaderEpoch 判定が本体、2-step bump はその補強」が正しい理解です。
//
// ■ スレッド安全性
//   - enterReader() / exitReader() / getState(): Audio Thread から呼び出し可能（Lock-free）
//   - swap() / tryReclaim() / getMinReaderEpoch(): 非 RT スレッドから呼び出す
//   - fallbackMutex は Audio Thread では一切使用しない
//
// ■ ODR 準拠
//   すべての関数をヘッダー内で inline 定義し、複数翻訳単位での二重定義を防止。
#pragma once

#include "ConvolverState.h"

#include <atomic>
#include <array>
#include <queue>
#include <mutex>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstddef>

class SafeStateSwapper
{
public:
    // リタイアバッファの最大エントリ数
    static constexpr size_t kMaxRetired = 256;

    // 同時参加できる Reader の最大数
    static constexpr int kMaxReaders = 8;

    // Reader が非参加（静寂状態）であることを示す特別な値
    // uint64_t の最大値を使うことで、最小値計算から自動的に除外される。
    static constexpr uint64_t kIdleEpoch = std::numeric_limits<uint64_t>::max();

    // -----------------------------------------------------------------------
    // コンストラクタ / デストラクタ
    // -----------------------------------------------------------------------
    SafeStateSwapper() noexcept : globalEpoch(0)
    {
        // 全 Reader を初期状態（非参加）に設定
        for (auto& e : readerEpochs)
            e.store(kIdleEpoch, std::memory_order_relaxed);
    }

    ~SafeStateSwapper() = default;

    SafeStateSwapper(const SafeStateSwapper&)            = delete;
    SafeStateSwapper& operator=(const SafeStateSwapper&) = delete;
    SafeStateSwapper(SafeStateSwapper&&)                 = delete;
    SafeStateSwapper& operator=(SafeStateSwapper&&)      = delete;

    // -----------------------------------------------------------------------
    // swap()  ── 非 RT スレッド（Message Thread / Rebuild Thread）から呼ぶ
    //
    // 古い状態を retired キューに積み、新しい状態を atomic にセットする。
    //
    // 2-step bump の意味:
    //   epoch1 = bump 前のエポック（旧データが有効だった時代を表す）
    //   retired エントリには epoch1 を記録し、reclaim 条件との整合性を確保する。
    //   newEpoch は単調増加の継続保証のために進める。
    //
    // @param newState  新しい状態（所有権を移譲。nullptr も可: 状態クリア用）
    // -----------------------------------------------------------------------
    void swap(ConvolverState* newState) noexcept
    {
        // 2-step bump（単調性確保 + 観測ズレの吸収）
        const uint64_t epoch1   = globalEpoch.fetch_add(1, std::memory_order_acq_rel);
        /* newEpoch = */ globalEpoch.fetch_add(1, std::memory_order_acq_rel);

        ConvolverState* oldState = activeState.exchange(newState, std::memory_order_acq_rel);
        if (oldState == nullptr) return;

        // リングバッファに積む
        size_t t    = tail.load(std::memory_order_relaxed);
        size_t next = (t + 1) % kMaxRetired;

        if (next == head.load(std::memory_order_acquire))
        {
            // バッファ溢れ: フォールバックキュー（非 RT パスなのでロック可）
            std::lock_guard<std::mutex> lock(fallbackMutex);
            fallbackQueue.push({oldState, epoch1});
            return;
        }

        retiredBuffer[t].state.store(oldState, std::memory_order_release);
        // epoch1 を記録: 「このデータが有効だった時代の直前エポック」
        retiredBuffer[t].epoch.store(epoch1,    std::memory_order_release);
        tail.store(next, std::memory_order_release);
    }

    // -----------------------------------------------------------------------
    // enterReader()  ── Audio Thread（RT）から呼ぶ  ✅ Lock-free
    //
    // 現在の globalEpoch を Reader のエポックとして登録し、参加を表明する。
    // この後 getState() で取得したポインタは、exitReader() までの間有効。
    //
    // @param readerIndex  0 ～ kMaxReaders-1 の固定インデックス
    // -----------------------------------------------------------------------
    void enterReader(int readerIndex) noexcept
    {
        if (readerIndex >= 0 && readerIndex < kMaxReaders)
        {
            const uint64_t currentEpoch = globalEpoch.load(std::memory_order_acquire);
            readerEpochs[readerIndex].store(currentEpoch, std::memory_order_release);
        }
    }

    // -----------------------------------------------------------------------
    // exitReader()  ── Audio Thread（RT）から呼ぶ  ✅ Lock-free
    //
    // kIdleEpoch に戻すことで「非参加（静寂状態）」を表明する。
    // この後、Deferred Free Thread は安全にオブジェクトを解放できる可能性がある。
    //
    // NOTE: exitReader() を呼ばずに Reader が停止した場合、そのインデックスの
    //       エポックは kIdleEpoch 以外のままとなり、reclaim がブロックされる。
    //       これは RCU の仕様（設計上の安全側フェール）。
    //
    // @param readerIndex  enterReader() に渡したのと同じインデックス
    // -----------------------------------------------------------------------
    void exitReader(int readerIndex) noexcept
    {
        if (readerIndex >= 0 && readerIndex < kMaxReaders)
        {
            readerEpochs[readerIndex].store(kIdleEpoch, std::memory_order_release);
        }
    }

    // -----------------------------------------------------------------------
    // getState()  ── Audio Thread（RT）から呼ぶ  ✅ Lock-free / Wait-free
    //
    // enterReader() と exitReader() の間で呼ぶ。
    // 返されたポインタは exitReader() まで有効が保証される。
    //
    // @return アクティブな ConvolverState へのポインタ（nullptr の場合あり）
    // -----------------------------------------------------------------------
    ConvolverState* getState() const noexcept
    {
        return activeState.load(std::memory_order_acquire);
    }

    // -----------------------------------------------------------------------
    // tryReclaim()  ── Deferred Free Thread / Message Thread から呼ぶ
    //
    // 解放可能なエントリを 1 件取り出して返す。
    // 安全性の保証:
    //   entryEpoch < minReaderEpoch が成立する場合のみ返す。
    //   これは「そのエントリを観測していた全 Reader がすでに退出済み」を意味する。
    //
    // @param minReaderEpoch  getMinReaderEpoch() の戻り値
    // @return 解放すべき ConvolverState*（なければ nullptr）
    // -----------------------------------------------------------------------
    ConvolverState* tryReclaim(uint64_t minReaderEpoch) noexcept
    {
        // フォールバックキューを先に確認（優先度付きキューなので最古エポックから）
        {
            std::lock_guard<std::mutex> lock(fallbackMutex);
            if (!fallbackQueue.empty())
            {
                const auto entry = fallbackQueue.top();
                if (entry.epoch < minReaderEpoch)
                {
                    if (entry.state != nullptr
                        && entry.state->snapshotRefCount.load(std::memory_order_relaxed) == 0)
                    {
                        fallbackQueue.pop();
                        return entry.state;
                    }
                }
            }
        }

        // リングバッファを確認
        const size_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return nullptr;

        // tail(acquire) により、それ以前の epoch/state の release store は可視
        const uint64_t entryEpoch = retiredBuffer[h].epoch.load(std::memory_order_acquire);
        // 注意: atomic_thread_fence(acquire) は不要。tail.load との同期で十分。
        if (isOlder(entryEpoch, minReaderEpoch))
        {
            ConvolverState* ptr = retiredBuffer[h].state.load(std::memory_order_acquire);
            if (ptr != nullptr
                && ptr->snapshotRefCount.load(std::memory_order_relaxed) == 0)
            {
                // state を nullptr にクリアしてから head を進める（二重取得防止）
                retiredBuffer[h].state.store(nullptr, std::memory_order_relaxed);
                head.store((h + 1) % kMaxRetired, std::memory_order_release);
                return ptr;
            }

            // 削除不可: head を進めて tail 側へ回転する
            head.store((h + 1) % kMaxRetired, std::memory_order_release);

            const size_t t = tail.load(std::memory_order_relaxed);
            const size_t nextTail = (t + 1) % kMaxRetired;
            if (nextTail == head.load(std::memory_order_acquire))
            {
                std::lock_guard<std::mutex> lock(fallbackMutex);
                fallbackQueue.push({ptr, entryEpoch});
            }
            else
            {
                retiredBuffer[t].state.store(ptr, std::memory_order_relaxed);
                retiredBuffer[t].epoch.store(entryEpoch, std::memory_order_relaxed);
                tail.store(nextTail, std::memory_order_release);
            }
            retiredBuffer[h].state.store(nullptr, std::memory_order_relaxed);
            return nullptr;
        }

        return nullptr;
    }

    // -----------------------------------------------------------------------
    // getMinReaderEpoch()  ── Deferred Free Thread / Message Thread から呼ぶ
    //
    // アクティブな全 Reader の中で最古のエポックを計算して返す。
    //
    // - kIdleEpoch のスロットは「非参加」として計算から除外。
    // - アクティブな Reader がいない場合 → 現在の globalEpoch を返す
    //   （すべての retired エントリが解放可能）。
    //
    // @return  tryReclaim() に渡すべき minReaderEpoch 値
    // -----------------------------------------------------------------------
    uint64_t getMinReaderEpoch() const noexcept
    {
        uint64_t minEpoch       = std::numeric_limits<uint64_t>::max();
        bool     hasActiveReader = false;

        for (int i = 0; i < kMaxReaders; ++i)
        {
            const uint64_t e = readerEpochs[i].load(std::memory_order_acquire);

            // kIdleEpoch = 非参加 → 最小値計算から除外
            if (e != kIdleEpoch)
            {
                hasActiveReader = true;
                if (e < minEpoch)
                    minEpoch = e;
            }
        }

        // アクティブな Reader が存在しない場合、すべての旧データは安全に解放可能
        if (!hasActiveReader)
            return globalEpoch.load(std::memory_order_acquire);

        return minEpoch;
    }

    // ラップアラウンドを考慮した「より古い」判定（RCU で必須）
    // AudioEngine の processDeferredReleases からも呼び出し可能
    static inline bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

private:
    // -----------------------------------------------------------------------
    // 内部データ構造
    // -----------------------------------------------------------------------

    // リタイアバッファの 1 エントリ
    struct RetiredEntry
    {
        std::atomic<ConvolverState*> state {nullptr};
        std::atomic<uint64_t>        epoch {0};
    };

    // フォールバックキューのエントリ（バッファ溢れ時に使用）
    // priority_queue を Min-Heap として使うため、epoch が大きいほど低優先度。
    struct FallbackEntry
    {
        ConvolverState* state;
        uint64_t        epoch;

        bool operator<(const FallbackEntry& o) const noexcept
        {
            // std::priority_queue はデフォルトで Max-Heap のため、
            // epoch が大きいものを low-priority（後回し）にするために逆転させる。
            return epoch > o.epoch;
        }
    };

    // アクティブな状態（Audio Thread が参照するポインタ）
    std::atomic<ConvolverState*> activeState {nullptr};

    // リタイアリングバッファ（固定容量、Lock-free）
    std::array<RetiredEntry, kMaxRetired> retiredBuffer {};
    std::atomic<size_t> head {0};
    std::atomic<size_t> tail {0};

    // グローバルエポックカウンタ（swap() ごとに 2 進む）
    std::atomic<uint64_t> globalEpoch {0};

    // Reader ごとのエポック追跡（kIdleEpoch = 非参加）
    std::array<std::atomic<uint64_t>, kMaxReaders> readerEpochs {};

    // フォールバックキュー（バッファ溢れ時、非 RT スレッドのみ使用）
    mutable std::mutex                         fallbackMutex;
    std::priority_queue<FallbackEntry>         fallbackQueue;
};
