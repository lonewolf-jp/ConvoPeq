// src/ConvolverState.h
// ★ [Architecture Debt] 軽量化版 — MKL/DFTI/作業バッファ管理を削除
// stateId / generationId / sampleRate のみを保持する軽量メタデータ構造。
// Epoch-based RCU の「保護対象オブジェクト」として SafeStateSwapper に渡す。
#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>      // uint64_t

#include "audioengine/AtomicAccess.h"

namespace convo {

// ---------------------------------------------------------------------------
// ConvolverState（軽量化版）
// ---------------------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4324)
struct ConvolverState
{
    // ★ [Architecture Debt] partitionData/overlapBuffer/inputBuffer/outputBuffer/fftHandle removed
    // (all were dead code — never read by Audio Thread)
    size_t   partitionSizeBytes = 0;
    int      numPartitions      = 0;
    int      fftSize            = 0;
    uint64_t generationId       = 0;
    double   sampleRate         = 0.0;

    // 冪等クリーンアップ用フラグ
    std::atomic<bool> cleanedUp {false};

    // スナップショット比較用の不変ID（UAF回避のためポインタ比較を避ける）
    uint64_t stateId = generateNewStateId();

private:

    static std::atomic<uint64_t> stateIdCounterStorage_;
    static std::atomic<uint64_t>& stateIdCounter() noexcept;
    static uint64_t generateNewStateId() noexcept;

    public:

    // -----------------------------------------------------------------------
    // デフォルトコンストラクタ（空の状態）
    // -----------------------------------------------------------------------
    ConvolverState() = default;

    // -----------------------------------------------------------------------
    // 軽量コンストラクタ — バッファ確保なし（すべてデッドコードのため除去）
    // -----------------------------------------------------------------------
    ConvolverState(int fSize, uint64_t genId, double sr)
        : fftSize(fSize),
          generationId(genId),
          sampleRate(sr)
    {
    }

    // -----------------------------------------------------------------------
    // デストラクタ（cleanup のみ）
    // -----------------------------------------------------------------------
    ~ConvolverState() { cleanup(); }

    // -----------------------------------------------------------------------
    // 冪等クリーンアップ（二重解放防止）
    // ★ [Architecture Debt] 以前の partitionData/overlapBuffer/inputBuffer/outputBuffer/fftHandle 解放は削除
    // -----------------------------------------------------------------------
    void cleanup() noexcept
    {
        if (convo::exchangeAtomic(cleanedUp, true, std::memory_order_acq_rel)) return;
    }

    // -----------------------------------------------------------------------
    // ムーブコンストラクタ
    // -----------------------------------------------------------------------
    ConvolverState(ConvolverState&& o) noexcept
    {
        partitionSizeBytes = o.partitionSizeBytes;
        numPartitions      = o.numPartitions;
        fftSize            = o.fftSize;
        generationId       = o.generationId;
        sampleRate         = o.sampleRate;
        stateId            = o.stateId;

        convo::publishAtomic(o.cleanedUp, true, std::memory_order_release);
        convo::publishAtomic(cleanedUp, false, std::memory_order_release);
    }

    // -----------------------------------------------------------------------
    // ムーブ代入演算子
    // -----------------------------------------------------------------------
    ConvolverState& operator=(ConvolverState&& o) noexcept
    {
        if (this != &o)
        {
            cleanup();

            partitionSizeBytes = o.partitionSizeBytes;
            numPartitions      = o.numPartitions;
            fftSize            = o.fftSize;
            generationId       = o.generationId;
            sampleRate         = o.sampleRate;
            stateId            = o.stateId;

            convo::publishAtomic(o.cleanedUp, true, std::memory_order_release);
            convo::publishAtomic(cleanedUp, false, std::memory_order_release);
        }
        return *this;
    }

    // コピー禁止
    ConvolverState(const ConvolverState&)            = delete;
    ConvolverState& operator=(const ConvolverState&) = delete;
};
#pragma warning(pop) // C4324 suppression scope end

} // namespace convo
