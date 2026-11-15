#pragma once

#include <algorithm>
#include <memory>
#include <new>

#include <JuceHeader.h>
#include "audioengine/AtomicAccess.h"
#include "AlignedAllocation.h"

class AudioSegmentBuffer
{
public:
    static constexpr int kMaxSeconds = 5;
    static constexpr int kMaxSampleRate = 768000;
    static constexpr int kCapacity = kMaxSeconds * kMaxSampleRate;
    static constexpr size_t kAlignment = 64;  // SIMD 対応 alignment

    // ★ ISR: Memory Authority は Builder / MemoryPool が持つ
    //   v11: 暫定的に factory + unique_ptr
    //   将来: RuntimeBuilder::allocateSegmentBuffer() → MemoryPool から取得
    // ★ Contract: create() は NonRT（Builder/MessageThread）のみから呼び出し可能。
    //   RT（Audio Thread）内での呼び出しは禁止。
    // ★ Strong exception guarantee: success=fully initialized, failure=no allocation remains
    // ★ D162-2-F: allocator contract 一致修正。
    //   旧実装は _aligned_malloc（CRT）で確保し、所有者 ScopedAlignedPtr の dtor が
    //   aligned_free → mkl_free（DiagnosticsConfig.h CONVOPEQ_ALIGNED_FREE）で解放していた。
    //   この allocator mismatch により Debug CRT の _free_dbg が no-man's-land 破壊を検出し
    //   0xC0000005 が発生（D162-2-D0 PASS-B で確定）。
    //   修正: convo::aligned_malloc（→ DIAG_MKL_MALLOC/mkl_malloc）に統一し、
    //   所有者 ScopedAlignedPtr の aligned_free（→ mkl_free）と pair を一致させた。
    //   失敗時の _aligned_free も aligned_free（= mkl_free）に統一（no-op 安全: nullptr free）。
    [[nodiscard]] static std::unique_ptr<AudioSegmentBuffer> create()
    {
        // ★ B-1: aligned_malloc は失敗時に std::bad_alloc を送出する（nullptr を返さない）。
        //   旧実装は nullptr 判定（到達しないデッドコード）を持ち、2 本目の確保や
        //   new AudioSegmentBuffer() で例外が出ると 1 本目／両方が解放されずに漏れていた。
        //   ここでは確保関数を注入可能な 1 本の経路にまとめ、所有権を常に RAII に持たせる。
        return createWithAllocator([](size_t bytes, size_t alignment) {
            return convo::aligned_malloc(bytes, alignment);
        });
    }

    // ★ B-1: 例外経路を検証するための確保関数注入（テスト専用）。
    //   本番の呼出し元は create() のみ。テストは n 回目で std::bad_alloc を投げる
    //   確保関数を渡し、確保済みバッファが解放されることを確かめる。
    //   AudioSegmentBuffer を先に確保してからバッファを差し替えるため、
    //   確保関数が例外を投げた時点の所有者は必ず buf（unique_ptr）になる。
    //
    //   契約: allocate は失敗時に必ず例外を送出すること。nullptr を返してはならない。
    //   本番の convo::aligned_malloc は std::bad_alloc を送出するので create() は
    //   この契約を満たす。契約違反（nullptr 返し）で不完全なバッファを返さないよう、
    //   ここでも nullptr を検出したら std::bad_alloc にして fail-closed にする。
    template <typename AllocateFn>
    [[nodiscard]] static std::unique_ptr<AudioSegmentBuffer> createWithAllocator(AllocateFn&& allocate)
    {
        auto buf = std::unique_ptr<AudioSegmentBuffer>(new AudioSegmentBuffer());

        buf->leftSamples_.reset(allocateChecked(allocate));
        buf->rightSamples_.reset(allocateChecked(allocate));

        return buf;
    }

    // Rule of Five: コピー・ムーブ禁止（巨大バッファ保護）
    AudioSegmentBuffer(const AudioSegmentBuffer&) = delete;
    AudioSegmentBuffer& operator=(const AudioSegmentBuffer&) = delete;
    AudioSegmentBuffer(AudioSegmentBuffer&&) = delete;
    AudioSegmentBuffer& operator=(AudioSegmentBuffer&&) = delete;

    void clear() noexcept
    {
        // release: clear 後に pushBlock/copyLatest を実行するスレッドが
        //          acquire で 0 実再を観測できるよう HB を保証。
        convo::publishAtomic(writePosition, 0, std::memory_order_release);
        convo::publishAtomic(totalSamples, 0, std::memory_order_release);
    }

    void pushBlock(const double* left, const double* right, int numSamples) noexcept
    {
        if (left == nullptr || right == nullptr || numSamples <= 0)
            return;

        // ★ Bug C: 境界チェック（drop 方針）
        //   kCapacity を超える入力は契約違反。状態を変更せず return。
        if (numSamples > kCapacity)
        {
            jassert(numSamples <= kCapacity);
            return;
        }

        // acquire: 直前の clear/pushBlock の release と HB し、有効な writePosition を取得。
        const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
        int first = std::min(numSamples, kCapacity - currentWritePos);
        juce::FloatVectorOperations::copy(leftSamples_.get() + currentWritePos, left, first);
        juce::FloatVectorOperations::copy(rightSamples_.get() + currentWritePos, right, first);

        if (first < numSamples)
        {
            int second = numSamples - first;
            juce::FloatVectorOperations::copy(leftSamples_.get(), left + first, second);
            juce::FloatVectorOperations::copy(rightSamples_.get(), right + first, second);
            // release: 更新後の writePosition を読み取りスレッドに可視化。
            convo::publishAtomic(writePosition, second, std::memory_order_release);
        }
        else
        {
            int nextPos = currentWritePos + numSamples;
            if (nextPos >= kCapacity)
                nextPos = 0;
            // release: 次書き込み位置を読み取りスレッドに可視化。
            convo::publishAtomic(writePosition, nextPos, std::memory_order_release);
        }

        // acquire: clear/pushBlock の release と HB し、有効な totalSamples を取得。
        const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
        // release: 更新後の totalSamples を読み取りスレッドに可視化。
        convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
    }

    int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
    {
        if (outLeft == nullptr || outRight == nullptr || requestedSamples <= 0)
            return 0;

        // acquire: pushBlock の release と HB し、最新の writePosition/totalSamples を取得。
        // [work87 P1-0] Writerのrelease順序(writePosition→totalSamples)と一致させる
        const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
        const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);

        const int availableSamples = std::min(requestedSamples,
            currentTotal >= kCapacity ? kCapacity : currentTotal);
        const int start = (currentWritePos - availableSamples + kCapacity) % kCapacity;

        for (int i = 0; i < availableSamples; ++i)
        {
            const int sourceIndex = (start + i) % kCapacity;
            outLeft[i] = leftSamples_.get()[sourceIndex];
            outRight[i] = rightSamples_.get()[sourceIndex];
        }

        return availableSamples;
    }

    int getNumAvailableSamples() const noexcept
    {
        // acquire: pushBlock/clear の release と HB し、最新の totalSamples を取得。
        return convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    }

    ~AudioSegmentBuffer() = default;  // unique_ptr からの破棄を許可

private:
    AudioSegmentBuffer() = default;  // Factory create() のみ生成可能

    // ★ B-1: 確保関数の契約を強制する。nullptr を返したら std::bad_alloc にして
    //   不完全なバッファ（確保済みに見えるが中身が空）を返さない。
    template <typename AllocateFn>
    [[nodiscard]] static double* allocateChecked(AllocateFn&& allocate)
    {
        auto* p = static_cast<double*>(allocate(kCapacity * sizeof(double), kAlignment));
        if (p == nullptr)
            throw std::bad_alloc();
        return p;
    }

    convo::ScopedAlignedPtr<double> leftSamples_;   // heap, 64-byte aligned
    convo::ScopedAlignedPtr<double> rightSamples_;  // heap, 64-byte aligned
    std::atomic<int> writePosition { 0 };
    std::atomic<int> totalSamples { 0 };
};

// サイズ static_assert（ヒープ化確認）
static_assert(sizeof(AudioSegmentBuffer) < 1024,
    "AudioSegmentBuffer must be heap allocated — stack allocation prohibited");
