// src/DeferredFreeThread.h
// Phase 0: ConvolverState の遅延解放専用スレッド
//
// 設計思想:
//   - MKL メモリ解放（mkl_free）や DFTI ハンドル解放は時間がかかる場合があり、
//     Audio Thread 内で呼ぶことは絶対に禁止されている。
//   - DeferredFreeThread は SafeStateSwapper の retired キューを定期的に確認し、
//     安全に解放できるエントリを delete することで、Audio Thread の RT 性を守る。
//   - Audio Thread が参照中のオブジェクトを誤って解放しないよう、
//     Epoch-based RCU の判定（getMinReaderEpoch / tryReclaim）に完全に委ねる。
//
// ライフサイクル:
//   生成: ConvolverProcessor::prepareToPlay() で作成
//   停止: ConvolverProcessor::releaseResources() で stop() → デストラクタで join
//   デストラクタ: スレッド join 後、残った全エントリを強制解放（Audio Thread 停止後）
//
// スレッド安全性:
//   - run() ループは専用スレッドで実行（Audio Thread や Message Thread とは別）
//   - stop() は std::atomic<bool> への store のみ → RT-safe
#pragma once

#include "SafeStateSwapper.h"

#include <thread>
#include <atomic>
#include <limits>

class DeferredFreeThread
{
public:
    // -----------------------------------------------------------------------
    // コンストラクタ  ── Message Thread から呼ぶ
    //
    // @param swapper  解放対象の retired キューを持つ SafeStateSwapper への参照
    // -----------------------------------------------------------------------
    explicit DeferredFreeThread(SafeStateSwapper& swapper)
        : swapperRef(swapper), running(true)
    {
        thread = std::thread([this]() { run(); });
    }

    // -----------------------------------------------------------------------
    // デストラクタ  ── Message Thread から呼ぶ
    //
    // 1. running フラグを落として run ループを終了させる。
    // 2. スレッドの join を待つ（最大でポーリング周期 1 回分 = 数 ms）。
    // 3. スレッド停止後に残っているエントリを強制解放する。
    //    この時点で Audio Thread は停止済みのはずなので、UAF のリスクはない。
    // -----------------------------------------------------------------------
    ~DeferredFreeThread()
    {
        stop();

        if (thread.joinable())
            thread.join();

        // スレッド停止後の強制解放（Epoch を UINT64_MAX にすれば全エントリが対象）
        while (auto* ptr = swapperRef.tryReclaim(std::numeric_limits<uint64_t>::max()))
            delete ptr;
    }

    // コピー・ムーブ禁止
    DeferredFreeThread(const DeferredFreeThread&)            = delete;
    DeferredFreeThread& operator=(const DeferredFreeThread&) = delete;
    DeferredFreeThread(DeferredFreeThread&&)                 = delete;
    DeferredFreeThread& operator=(DeferredFreeThread&&)      = delete;

    // -----------------------------------------------------------------------
    // stop()  ── どのスレッドからも呼び出し可能
    //
    // running フラグを落とす。run() ループは次のポーリング周期で終了する。
    // join はデストラクタで行う。
    // -----------------------------------------------------------------------
    void stop() noexcept
    {
        running.store(false, std::memory_order_release);
    }

private:
    // -----------------------------------------------------------------------
    // run()  ── 専用スレッドで実行
    //
    // 1ループあたり最大 kMaxReclaimPerLoop 個まで解放してから yield する。
    // これにより IR 切り替え直後の素早い解放を維持しつつ、
    // 大量オブジェクト蓄積時の CPU スパイクを防ぐ。
    // [fix4 R5] sleep_for(1ms) → yield() に変更し応答性を向上
    // -----------------------------------------------------------------------
    static constexpr int kMaxReclaimPerLoop = 4;

    void run()
    {
        while (running.load(std::memory_order_acquire))
        {
            const uint64_t minEpoch = swapperRef.getMinReaderEpoch();
            int reclaimCount = 0;
            while (auto* ptr = swapperRef.tryReclaim(minEpoch))
            {
                delete ptr;
                if (++reclaimCount >= kMaxReclaimPerLoop) break;
            }
            if (reclaimCount == 0)
            {
                // 解放対象なし → yield して CPU 負荷を軽減（sleep より応答性良好）
                std::this_thread::yield();
            }
        }
    }

    SafeStateSwapper&     swapperRef;
    std::atomic<bool>     running;
    std::thread           thread;
};
