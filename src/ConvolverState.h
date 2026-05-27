// src/ConvolverState.h
// Phase 0: MKL リソース（FFT Descriptor, 作業バッファ）を RAII で管理するコンテナ
//
// 設計思想:
//   - MKL メモリ（mkl_malloc/mkl_free）と DFTI ハンドルを一括管理し、
//     例外安全・冪等なクリーンアップを保証する。
//   - cleanup() は何度呼んでも安全（atomic フラグで二重解放を防止）。
//   - Audio Thread は ConvolverState を読み取るだけで、生成・解放は行わない。
//
// メモリ規約:
//   - 全バッファは mkl_malloc(size, 64) で 64 byte アライン確保。
//   - new / std::vector は使用しない（コーディング規約準拠）。
//
// スレッド安全性:
//   - cleanup() 自体は Message Thread / Deferred Free Thread から呼ぶ想定。
//   - Audio Thread は cleanup() を呼ばない。
#pragma once

#include <atomic>
#include <cstring>      // memset
#include <stdexcept>
#include <cstddef>      // size_t
#include <cstdint>      // uint64_t

#include <mkl.h>
#include <mkl_dfti.h>

#include "AlignedAllocation.h"  // convo::aligned_malloc / aligned_free
#include "DftiHandle.h"

#include "audioengine/AtomicAccess.h"

namespace convo {

// ---------------------------------------------------------------------------
// ConvolverState
// 畳み込みエンジン 1 インスタンス分の MKL リソースをすべて保持する。
// Epoch-based RCU の「保護対象オブジェクト」として SafeStateSwapper に渡す。
// ---------------------------------------------------------------------------
#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
struct ConvolverState
{
    // --- FFT 作業バッファ（mkl_malloc 64-byte アライン）---
    // Audio Thread 側は helper 経由で取得して読み書きする。
    double* partitionData = nullptr;                // IR 周波数領域パーティション
    std::atomic<double*> overlapBuffer {nullptr};   // OLA オーバーラップ
    std::atomic<double*> inputBuffer   {nullptr};   // FFT 入力作業領域
    std::atomic<double*> outputBuffer  {nullptr};   // FFT 出力作業領域

    size_t   partitionSizeBytes = 0;
    int      numPartitions      = 0;
    int      fftSize            = 0;
    uint64_t generationId       = 0;
    double   sampleRate         = 0.0;

    // DFTI ハンドル（RAII 管理）
    ScopedDftiDescriptor fftHandle;

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
    // 完全初期化コンストラクタ
    //
    // @param data      IR パーティションデータ（mkl_malloc 済み、所有権を移譲）
    // @param dataSize  data のバイト数
    // @param nParts    パーティション数
    // @param fSize     FFT サイズ
    // @param genId     この状態が属する世代番号
    // @param sr        サンプルレート
    //
    // 例外安全: コンストラクタが throw した場合、確保済みリソースは cleanup() で解放。
    // -----------------------------------------------------------------------
    ConvolverState(double*  data,
                   size_t   dataSize,
                   int      nParts,
                   int      fSize,
                   uint64_t genId,
                   double   sr)
        : partitionSizeBytes(dataSize),
          numPartitions(nParts),
          fftSize(fSize),
          generationId(genId),
          sampleRate(sr)
    {
        jassert(data != nullptr);
        jassert(((uintptr_t)data % 64) == 0 && "ConvolverState: MKL Memory Alignment Failed!");

        partitionData = data;

        // 作業バッファ確保ヘルパー（失敗時は std::bad_alloc を throw）
        auto safeMalloc = [](size_t bytes) -> double*
        {
            auto* ptr = convo::makeAlignedArray<double>(bytes / sizeof(double)).release();
            if (!ptr) throw std::bad_alloc();
            return ptr;
        };

        // DFTI ハンドル生成
        if (DftiCreateDescriptor(fftHandle.put(), DFTI_DOUBLE, DFTI_REAL, 1,
                                 static_cast<MKL_LONG>(fftSize)) != DFTI_NO_ERROR)
        {
            throw std::runtime_error("ConvolverState: DftiCreateDescriptor failed");
        }

        // 各作業バッファを確保してゼロクリア
        {
            auto* ov = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            convo::publishAtomic(overlapBuffer, ov, std::memory_order_release); // release: getActiveCoeffSet の acquire と HB (初期化完了後の初回観測を保証)
            memset(ov, 0, static_cast<size_t>(fftSize) * sizeof(double));
        }
        {
            auto* ib = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            convo::publishAtomic(inputBuffer, ib, std::memory_order_release); // release: getActiveCoeffSet の acquire と HB (初期化完了後の初回観測を保証)
            memset(ib, 0, static_cast<size_t>(fftSize) * sizeof(double));
        }
        {
            auto* ob = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            convo::publishAtomic(outputBuffer, ob, std::memory_order_release); // release: getActiveCoeffSet の acquire と HB (初期化完了後の初回観測を保証)
            memset(ob, 0, static_cast<size_t>(fftSize) * sizeof(double));
        }
    }

    // -----------------------------------------------------------------------
    // デストラクタ
    // -----------------------------------------------------------------------
    ~ConvolverState() { cleanup(); }

    // -----------------------------------------------------------------------
    // 冪等クリーンアップ（二重解放防止）
    // Message Thread / DeferredFreeThread から呼ぶ。Audio Thread からは呼ばない。
    // -----------------------------------------------------------------------
    void cleanup() noexcept
    {
        // exchange が false を返した（＝まだ解放していない）場合のみ実行
        if (convo::exchangeAtomic(cleanedUp, true, std::memory_order_acq_rel)) return; // acq_rel: 冪等チェック — acquire で先行 cleanup の acq_rel と HB; release で 2 回目呼び出し元の acquire と HB

        // 各 atomic ポインタを nullptr に swap して古いポインタを解放
        if (auto* p = partitionData) { partitionData = nullptr; convo::aligned_free(p); }
        if (auto* p = convo::exchangeAtomic(overlapBuffer, nullptr, std::memory_order_acq_rel))  convo::aligned_free(p); // acq_rel: acquire で publishAtomic release と HB しポインタ取得; release で nullptr 観測者との HB
        if (auto* p = convo::exchangeAtomic(inputBuffer, nullptr, std::memory_order_acq_rel))    convo::aligned_free(p); // acq_rel: 同上
        if (auto* p = convo::exchangeAtomic(outputBuffer, nullptr, std::memory_order_acq_rel))   convo::aligned_free(p); // acq_rel: 同上

        // fftHandle は ScopedDftiDescriptor のデストラクタで自動解放される
        fftHandle.reset();
    }

    // -----------------------------------------------------------------------
    // ムーブコンストラクタ（所有権の移譲）
    // -----------------------------------------------------------------------
    ConvolverState(ConvolverState&& o) noexcept
    {
        partitionData = o.partitionData;
        o.partitionData = nullptr;
            convo::publishAtomic(overlapBuffer, convo::exchangeAtomic(o.overlapBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release); // exchange acq_rel: ムーブ元の publishAtomic release と HB; publish release: 新 owner の getActiveCoeffSet acquire と HB
            convo::publishAtomic(inputBuffer, convo::exchangeAtomic(o.inputBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release);     // 同上
            convo::publishAtomic(outputBuffer, convo::exchangeAtomic(o.outputBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release);   // 同上

        partitionSizeBytes = o.partitionSizeBytes;
        numPartitions      = o.numPartitions;
        fftSize            = o.fftSize;
        generationId       = o.generationId;
        sampleRate         = o.sampleRate;
        stateId            = o.stateId;
        fftHandle          = std::move(o.fftHandle);

        // ムーブ元は cleanedUp = true にして二重解放を防ぐ
            convo::publishAtomic(o.cleanedUp, true, std::memory_order_release);  // release: ムーブ元の cleanup() の exchangeAtomic acquire と HB し二重解放防止
            convo::publishAtomic(cleanedUp, false, std::memory_order_release);   // release: 新 owner の cleanup() の exchangeAtomic acquire と HB (未解放を表明)
    }

    // -----------------------------------------------------------------------
    // ムーブ代入演算子
    // -----------------------------------------------------------------------
    ConvolverState& operator=(ConvolverState&& o) noexcept
    {
        if (this != &o)
        {
            cleanup();

            partitionData = o.partitionData;
            o.partitionData = nullptr;
            convo::publishAtomic(overlapBuffer, convo::exchangeAtomic(o.overlapBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release); // exchange acq_rel: ムーブ元の publishAtomic release と HB; publish release: 新 owner の getActiveCoeffSet acquire と HB
            convo::publishAtomic(inputBuffer, convo::exchangeAtomic(o.inputBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release);     // 同上
            convo::publishAtomic(outputBuffer, convo::exchangeAtomic(o.outputBuffer, nullptr, std::memory_order_acq_rel), std::memory_order_release);   // 同上

            partitionSizeBytes = o.partitionSizeBytes;
            numPartitions      = o.numPartitions;
            fftSize            = o.fftSize;
            generationId       = o.generationId;
            sampleRate         = o.sampleRate;
            stateId            = o.stateId;
            fftHandle          = std::move(o.fftHandle);

            convo::publishAtomic(o.cleanedUp, true, std::memory_order_release);  // release: ムーブ元の cleanup() の exchangeAtomic acquire と HB し二重解放防止
            convo::publishAtomic(cleanedUp, false, std::memory_order_release);   // release: 新 owner の cleanup() の exchangeAtomic acquire と HB (未解放を表明)
        }
        return *this;
    }

    // コピー禁止（MKL リソースの二重解放を防ぐ）
    ConvolverState(const ConvolverState&)            = delete;
    ConvolverState& operator=(const ConvolverState&) = delete;
};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

} // namespace convo
