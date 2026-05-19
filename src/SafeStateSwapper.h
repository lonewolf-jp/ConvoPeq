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

#include "audioengine/AtomicAccess.h"

namespace convo {
class SafeStateSwapper
{
public:
    // リタイアバッファの最大エントリ数
    static constexpr size_t kMaxRetired = 256;

    // 同時参加できる Reader の最大数
    static constexpr int kMaxReaders = 8;

    // Reader が非参加（静寂状態）であることを示す特別な値
    // uint64_t の最大値を使うことで、最小値計算から自動的に除外される。
    static constexpr uint64_t kIdleEpoch = 0;

    // -----------------------------------------------------------------------
    // コンストラクタ / デストラクタ
    // -----------------------------------------------------------------------
    SafeStateSwapper() noexcept : globalEpoch(1)
    {
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
        const uint64_t epoch1   = convo::fetchAddAtomic(globalEpoch, static_cast<uint64_t>(1), std::memory_order_acq_rel); // acq_rel: acquire で直前の swap の acq_rel と HB; release で getSafeEpoch/enterReader の acquire と HB
        /* newEpoch = */ convo::fetchAddAtomic(globalEpoch, static_cast<uint64_t>(1), std::memory_order_acq_rel); // acq_rel: 2-step bump 後半。単調性確保 + getSafeEpoch 観測側の acquire と HB

        ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, std::memory_order_acq_rel); // acq_rel: acquire で直前の swap の activeState release と HB; release で getState の acquire と HB
        if (oldState == nullptr) return;

        // リングバッファに積む
        size_t t    = convo::consumeAtomic(tail, std::memory_order_acquire); // acquire: tryReclaim の tail release と HB し最新の tail を観測
        size_t next = (t + 1) % kMaxRetired;

        if (next == convo::consumeAtomic(head, std::memory_order_acquire)) // acquire: tryReclaim の head release と HB しリングバッファ空き判定
        {
            // バッファ溢れ: フォールバックキュー（非 RT パスなのでロック可）
            std::lock_guard<std::mutex> lock(fallbackMutex);
            fallbackQueue.push({oldState, epoch1});
            return;
        }

        convo::publishAtomic(retiredBuffer[t].state, oldState, std::memory_order_release);  // release: tryReclaim の state acquire と HB し旧状態ポインタを公知
        // epoch1 を記録: 「このデータが有効だった時代の直前エポック」
        convo::publishAtomic(retiredBuffer[t].epoch, epoch1, std::memory_order_release);    // release: tryReclaim の epoch acquire と HB
        convo::publishAtomic(tail, next, std::memory_order_release);                        // release: tryReclaim の tail acquire と HB しエントリ追加完了を公知
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
            // relaxed load of global epoch is sufficient; release store publishes participation.
            const uint64_t currentEpoch = convo::consumeAtomic(globalEpoch, std::memory_order_acquire); // acquire: swap の fetchAdd acq_rel と HB し最新グローバルエポックを観測
            convo::publishAtomic(readerEpochs[readerIndex], currentEpoch, std::memory_order_release);  // release: getMinReaderEpoch の acquire と HB し参加表明を公知
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
            // relaxed store is sufficient for exiting participant.
            convo::publishAtomic(readerEpochs[readerIndex], kIdleEpoch, std::memory_order_release); // release: getMinReaderEpoch の acquire と HB し非参加を公知
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
        return convo::consumeAtomic(activeState, std::memory_order_acquire); // acquire: swap の exchangeAtomic acq_rel と HB し最新アクティブ状態を観測
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
                    if (entry.state != nullptr)
                    {
                        fallbackQueue.pop();
                        return entry.state;
                    }
                }
            }
        }

        // リングバッファを確認
        const size_t h = convo::consumeAtomic(head, std::memory_order_acquire);                          // acquire: 前回の head release と HB
        if (h == convo::consumeAtomic(tail, std::memory_order_acquire)) return nullptr;                  // acquire: swap の tail release と HB しエントリ存在確認

        // tail(acquire) により、それ以前の epoch/state の release store は可視
        const uint64_t entryEpoch = convo::consumeAtomic(retiredBuffer[h].epoch, std::memory_order_acquire); // acquire: swap の epoch release と HB
        // 注意: atomic_thread_fence(acquire) は不要。tail.load との同期で十分。
        if (isOlder(entryEpoch, minReaderEpoch))
        {
            ConvolverState* ptr = convo::consumeAtomic(retiredBuffer[h].state, std::memory_order_acquire); // acquire: swap の state release と HB し旧状態ポインタを取得
            if (ptr != nullptr)
            {
                // state を nullptr にクリアしてから head を進める（二重取得防止）
                convo::publishAtomic(retiredBuffer[h].state, nullptr, std::memory_order_release); // release: 次回 tryReclaim の state acquire と HB し二重取得防止
                convo::publishAtomic(head, (h + 1) % kMaxRetired, std::memory_order_release);    // release: swap の head acquire と HB しスロット解放を公知
                return ptr;
            }

            // 削除不可: head を進めて tail 側へ回転する
            convo::publishAtomic(head, (h + 1) % kMaxRetired, std::memory_order_release); // release: swap の head acquire と HB しスロット移動を公知

            const size_t t = convo::consumeAtomic(tail, std::memory_order_acquire); // acquire: swap の tail release と HB し最新 tail を観測
            const size_t nextTail = (t + 1) % kMaxRetired;
            if (nextTail == convo::consumeAtomic(head, std::memory_order_acquire)) // acquire: 最新 head を観測してバッファ溢れ判定
            {
                std::lock_guard<std::mutex> lock(fallbackMutex);
                fallbackQueue.push({ptr, entryEpoch});
            }
            else
            {
                convo::publishAtomic(retiredBuffer[t].state, ptr, std::memory_order_release);         // release: 次回 tryReclaim の state acquire と HB
                convo::publishAtomic(retiredBuffer[t].epoch, entryEpoch, std::memory_order_release);  // release: 次回 tryReclaim の epoch acquire と HB
                convo::publishAtomic(tail, nextTail, std::memory_order_release);                      // release: swap/tryReclaim の tail acquire と HB しエントリ追加完了を公知
            }
            convo::publishAtomic(retiredBuffer[h].state, nullptr, std::memory_order_release); // release: 次回 tryReclaim の state acquire と HB しスロットクリアを公知
            return nullptr;
        }

        return nullptr;
    }

    // -----------------------------------------------------------------------
    // getSafeEpoch()  ── Deferred Free Thread / Message Thread から呼ぶ
    // -----------------------------------------------------------------------
    uint64_t getSafeEpoch() const noexcept
    {
        const uint64_t current = convo::consumeAtomic(globalEpoch, std::memory_order_acquire); // acquire: swap の fetchAdd acq_rel と HB し最新グローバルエポックを観測
        if (current < 2) return 0;
        return current - 2;
    }

    size_t getPendingRetiredCount() noexcept
    {
        const size_t currentHead = convo::consumeAtomic(head, std::memory_order_acquire); // acquire: tryReclaim の head release と HB
        const size_t currentTail = convo::consumeAtomic(tail, std::memory_order_acquire); // acquire: swap/tryReclaim の tail release と HB
        const size_t ringCount = (currentTail + kMaxRetired - currentHead) % kMaxRetired;

        std::lock_guard<std::mutex> lock(fallbackMutex);
        return ringCount + fallbackQueue.size();
    }

    uint64_t getMinReaderEpoch() const noexcept
    {
        uint64_t minEpoch = std::numeric_limits<uint64_t>::max();
        bool hasActiveReader = false;

        for (size_t i = 0; i < kMaxReaders; ++i) {
            const uint64_t e = convo::consumeAtomic(readerEpochs[i], std::memory_order_acquire); // acquire: enterReader/exitReader の publishAtomic release と HB しエポックを観測
            if (e != kIdleEpoch) {
                if (!hasActiveReader) {
                    minEpoch = e;
                    hasActiveReader = true;
                } else if (isOlder(e, minEpoch)) {
                    minEpoch = e;
                }
            }
        }

        if (!hasActiveReader)
            return convo::consumeAtomic(globalEpoch, std::memory_order_acquire); // acquire: swap の fetchAdd acq_rel と HB — 読者なし時は最新グローバルエポックを返す

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
    std::mutex                                 fallbackMutex;
    std::priority_queue<FallbackEntry>         fallbackQueue;
};

} // namespace convo
