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
#include "core/ThreadAffinityManager.h"
#include "core/RetireBoundaryTelemetry.h"

#include <thread>
#include <atomic>
#include <chrono>
#include <limits>

#include <JuceHeader.h>

#include "audioengine/AtomicAccess.h"

namespace convo {

class DeferredFreeThread
{
public:
    // -----------------------------------------------------------------------
    // コンストラクタ  ── Message Thread から呼ぶ
    //
    // @param swapper  解放対象の retired キューを持つ SafeStateSwapper への参照
    // -----------------------------------------------------------------------
    explicit DeferredFreeThread(SafeStateSwapper& swapper, ThreadAffinityManager* affinityMgr = nullptr)
        : swapperRef(swapper), affinityManager(affinityMgr), running(true)
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
        shutdownAndDrain();
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
        convo::publishAtomic(running, false, std::memory_order_release); // release: run() の consumeAtomic acquire と HB しループ終了を公知
    }

    // -----------------------------------------------------------------------
    // shutdownAndDrain()  ── シャットダウン + 全強制解放
    //
    // 【安全契約】
    //   この関数を呼び出す時点で、Audio Thread が完全に停止していること。
    //   （通常は ConvolverProcessor::releaseResources() 経由で呼ばれる。
    //     releaseResources() は JUCE の AudioProcessor ライフサイクルにより
    //     Audio Thread 停止後に呼び出されることが保証されている。）
    //
    // 【二重呼び出し】
    //   この関数はデストラクタからも呼ばれる（二重呼び出し）。
    //   releaseResources() で先に呼ばれた場合、デストラクタ側の呼び出しは
    //   thread.joinable() == false により join をスキップし、
    //   drainAllRetired() は空キューを即時に完了するため安全。
    //   余計な状態変数を追加せず、joinable() 判定による冪等性に依存する。
    // -----------------------------------------------------------------------
    void shutdownAndDrain() noexcept
    {
        stop();

        if (thread.joinable())
            thread.join();

        drainAllRetired();
    }

    [[nodiscard]] convo::RetireBoundaryTelemetry snapshotBoundaryTelemetry() const noexcept
    {
        return convo::RetireBoundaryTelemetry {
            .pendingBacklog = static_cast<std::uint64_t>(swapperRef.getPendingRetiredCount()),
            .quarantineResidents = 0,
            .totalTransitions = 0,
            .boundaryActive = convo::consumeAtomic(running, std::memory_order_acquire)
        };
    }

private:
    // -----------------------------------------------------------------------
    // run()  ── 専用スレッドで実行
    //
    // 1ループあたり最大 kMaxReclaimPerLoop 個まで解放してから yield する。
    // これにより IR 切り替え直後の素早い解放を維持しつつ、
    // 大量オブジェクト蓄積時の CPU スパイクを防ぐ。
    // [Bug 2] アイドル時は短時間 sleep し、解放が進んだループのみ yield する
    // -----------------------------------------------------------------------
    static constexpr int kMaxReclaimPerLoop = 4;
    static constexpr size_t kPendingRetiredWarnThreshold = 64;

    // -----------------------------------------------------------------------
    // drainAllRetired()  ── 全 Retired エントリ強制解放（Shutdown 専用）
    //
    // 【前提条件】
    //   この関数を呼び出す時点で Audio Thread が完全に停止していること。
    //
    // 【備考】
    //   std::numeric_limits<uint64_t>::max() を tryReclaim に渡すことで
    //   エポック条件を無視した強制解放を行う。これは Audio Thread 停止後の
    //   クリーンアップ（releaseResources / デストラクタ）でのみ有効。
    //   通常の退役ループ（run()）では getMinReaderEpoch() を使用する。
    // -----------------------------------------------------------------------
    void drainAllRetired() noexcept
    {
        while (auto* ptr = swapperRef.tryReclaim(std::numeric_limits<uint64_t>::max()))
        {
            std::unique_ptr<convo::ConvolverState> owned{ptr}; // RAII delete
        }
    }

    void run()
    {
        if (affinityManager != nullptr)
            affinityManager->applyCurrentThreadPolicy(ThreadType::LightBackground);

        while (convo::consumeAtomic(running, std::memory_order_acquire)) // acquire: stop() の publishAtomic release と HB し最新の running 値を観測
        {
            const uint64_t minEpoch = swapperRef.getMinReaderEpoch();
            int reclaimCount = 0;
            while (auto* ptr = swapperRef.tryReclaim(minEpoch))
            {
                std::unique_ptr<convo::ConvolverState> owned{ptr}; // RAII delete
                if (++reclaimCount >= kMaxReclaimPerLoop) break;
            }
            if (reclaimCount == 0)
            {
                const size_t pendingRetired = swapperRef.getPendingRetiredCount();
                if (pendingRetired >= kPendingRetiredWarnThreshold)
                {
                    juce::Logger::writeToLog("[DIAG] DeferredFreeThread backlog pending="
                                             + juce::String(static_cast<juce::int64>(pendingRetired)));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            else
                std::this_thread::yield();
        }
    }

    SafeStateSwapper&     swapperRef;
    ThreadAffinityManager* affinityManager;
    std::atomic<bool>     running;
    std::thread           thread;
};

} // namespace convo
