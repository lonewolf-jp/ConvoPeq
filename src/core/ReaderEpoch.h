//==============================================================================
// ReaderEpoch.h
// RCU Reader epoch 追跡（スレッドローカルスロット、overflow slot 戦略）
// v13.0 設計ロック準拠
//==============================================================================
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace convo {

class ReaderEpoch {
public:
    static constexpr size_t kMaxReaders = 512;   // スロット枯渇をほぼ防止
    static constexpr uint64_t kIdleEpoch = UINT64_MAX;

    static size_t getThreadSlot() noexcept;
    static void enter(size_t slot) noexcept;
    static void exit(size_t slot) noexcept;
    static uint64_t getMinActiveEpoch() noexcept;
    static uint64_t advanceGlobalEpoch() noexcept;
    static uint64_t getCurrentGlobalEpoch() noexcept;

private:
    ReaderEpoch() = delete;

    static std::atomic<uint64_t> s_readerEpochs[kMaxReaders];
    static std::atomic<uint64_t> s_globalEpoch;
};

class ReaderEpochGuard {
public:
    ReaderEpochGuard() noexcept : m_slot(ReaderEpoch::getThreadSlot()) {
        ReaderEpoch::enter(m_slot);
    }

    ~ReaderEpochGuard() noexcept {
        ReaderEpoch::exit(m_slot);
    }

    ReaderEpochGuard(const ReaderEpochGuard&) = delete;
    ReaderEpochGuard& operator=(const ReaderEpochGuard&) = delete;

private:
    size_t m_slot;
};

} // namespace convo
