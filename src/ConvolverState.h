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

// ---------------------------------------------------------------------------
// DFTI_DESCRIPTOR_HANDLE の RAII ラッパー
// std::unique_ptr<void, ...> の代わりにクリーンな所有権管理を提供する。
// ---------------------------------------------------------------------------
class DftiHandleOwner
{
public:
    DftiHandleOwner() noexcept = default;

    explicit DftiHandleOwner(DFTI_DESCRIPTOR_HANDLE h) noexcept
        : handle(h) {}

    ~DftiHandleOwner() noexcept { reset(); }

    // ムーブのみ許可（コピー禁止）
    DftiHandleOwner(DftiHandleOwner&& o) noexcept
        : handle(o.handle) { o.handle = nullptr; }

    DftiHandleOwner& operator=(DftiHandleOwner&& o) noexcept
    {
        if (this != &o) { reset(); handle = o.handle; o.handle = nullptr; }
        return *this;
    }

    DftiHandleOwner(const DftiHandleOwner&)            = delete;
    DftiHandleOwner& operator=(const DftiHandleOwner&) = delete;

    // 新しいハンドルをセット（既存ハンドルは解放）
    void reset(DFTI_DESCRIPTOR_HANDLE h = nullptr) noexcept
    {
        if (handle) { DftiFreeDescriptor(&handle); handle = nullptr; }
        handle = h;
    }

    // ハンドルを所有権なしで取得（読み取り専用）
    DFTI_DESCRIPTOR_HANDLE get() const noexcept { return handle; }

    // 所有権を放棄してハンドルを返す（transfer ownership out）
    DFTI_DESCRIPTOR_HANDLE release() noexcept
    {
        auto h = handle; handle = nullptr; return h;
    }

    explicit operator bool() const noexcept { return handle != nullptr; }

private:
    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
};

// ---------------------------------------------------------------------------
// ConvolverState
// 畳み込みエンジン 1 インスタンス分の MKL リソースをすべて保持する。
// Epoch-based RCU の「保護対象オブジェクト」として SafeStateSwapper に渡す。
// ---------------------------------------------------------------------------
struct ConvolverState
{
    // --- FFT 作業バッファ（mkl_malloc 64-byte アライン）---
    // Audio Thread 側は .load() で取得して読み書きする。
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
    DftiHandleOwner fftHandle;

    // 冪等クリーンアップ用フラグ
    std::atomic<bool> cleanedUp {false};

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
            auto* ptr = static_cast<double*>(convo::aligned_malloc(bytes, 64));
            if (!ptr) throw std::bad_alloc();
            return ptr;
        };

        // DFTI ハンドル生成
        DFTI_DESCRIPTOR_HANDLE h = nullptr;
        if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1,
                                 static_cast<MKL_LONG>(fftSize)) != DFTI_NO_ERROR)
        {
            throw std::runtime_error("ConvolverState: DftiCreateDescriptor failed");
        }
        fftHandle.reset(h);

        // 各作業バッファを確保してゼロクリア
        {
            auto* ov = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            overlapBuffer.store(ov, std::memory_order_relaxed);
            memset(ov, 0, static_cast<size_t>(fftSize) * sizeof(double));
        }
        {
            auto* ib = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            inputBuffer.store(ib, std::memory_order_relaxed);
            memset(ib, 0, static_cast<size_t>(fftSize) * sizeof(double));
        }
        {
            auto* ob = safeMalloc(static_cast<size_t>(fftSize) * sizeof(double));
            outputBuffer.store(ob, std::memory_order_relaxed);
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
        if (cleanedUp.exchange(true, std::memory_order_acq_rel)) return;

        // 各 atomic ポインタを nullptr に swap して古いポインタを解放
        if (auto* p = partitionData) { partitionData = nullptr; convo::aligned_free(p); }
        if (auto* p = overlapBuffer.exchange(nullptr, std::memory_order_relaxed))  convo::aligned_free(p);
        if (auto* p = inputBuffer.exchange(nullptr, std::memory_order_relaxed))    convo::aligned_free(p);
        if (auto* p = outputBuffer.exchange(nullptr, std::memory_order_relaxed))   convo::aligned_free(p);

        // fftHandle は DftiHandleOwner のデストラクタで自動解放される
        fftHandle.reset();
    }

    // -----------------------------------------------------------------------
    // ムーブコンストラクタ（所有権の移譲）
    // -----------------------------------------------------------------------
    ConvolverState(ConvolverState&& o) noexcept
    {
        partitionData = o.partitionData;
        o.partitionData = nullptr;
        overlapBuffer.store(o.overlapBuffer.exchange(nullptr, std::memory_order_relaxed),
                            std::memory_order_relaxed);
        inputBuffer.store(o.inputBuffer.exchange(nullptr, std::memory_order_relaxed),
                          std::memory_order_relaxed);
        outputBuffer.store(o.outputBuffer.exchange(nullptr, std::memory_order_relaxed),
                           std::memory_order_relaxed);

        partitionSizeBytes = o.partitionSizeBytes;
        numPartitions      = o.numPartitions;
        fftSize            = o.fftSize;
        generationId       = o.generationId;
        sampleRate         = o.sampleRate;
        fftHandle          = std::move(o.fftHandle);

        // ムーブ元は cleanedUp = true にして二重解放を防ぐ
        o.cleanedUp.store(true, std::memory_order_relaxed);
        cleanedUp.store(false, std::memory_order_relaxed);
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
            overlapBuffer.store(o.overlapBuffer.exchange(nullptr, std::memory_order_relaxed),
                                std::memory_order_relaxed);
            inputBuffer.store(o.inputBuffer.exchange(nullptr, std::memory_order_relaxed),
                              std::memory_order_relaxed);
            outputBuffer.store(o.outputBuffer.exchange(nullptr, std::memory_order_relaxed),
                               std::memory_order_relaxed);

            partitionSizeBytes = o.partitionSizeBytes;
            numPartitions      = o.numPartitions;
            fftSize            = o.fftSize;
            generationId       = o.generationId;
            sampleRate         = o.sampleRate;
            fftHandle          = std::move(o.fftHandle);

            o.cleanedUp.store(true, std::memory_order_relaxed);
            cleanedUp.store(false, std::memory_order_relaxed);
        }
        return *this;
    }

    // コピー禁止（MKL リソースの二重解放を防ぐ）
    ConvolverState(const ConvolverState&)            = delete;
    ConvolverState& operator=(const ConvolverState&) = delete;
};
