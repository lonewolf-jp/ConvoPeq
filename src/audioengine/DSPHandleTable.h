//==============================================================================
// DSPHandleTable.h — FUTURE-6 (work88): DSPCore* → DSPHandle の O(1) 前方ハッシュテーブル
//
// 目的:
//   std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_ の置換。
//   - 固定容量オープンアドレッシング（再ハッシュなし・ヒープ確保なし）
//   - find/insert/erase O(1) expected（線形探索・負荷率 ≤0.5、容量 512 = 2x MAX_DSP_SLOTS）
//   - eraseByHandle は固定テーブルに対する O(n) 走査（上限 512。呼出し元は rollback パスのみ・
//     MAX_DSP_SLOTS=256 のため実質ボトルネックなし。将来 reverse map (slot → key) で O(1) 化可能）
//
// スレッド安全:
//   呼出し側（AudioEngine::runtimeDSPHandleMapMutex_）が全操作を保護する
//   （既存 unordered_map と同一契約。RT パスは lock なしで resolve を使用）。
//
// 移行: 旧構造（std::unordered_map）がコードベースから消えることを Phase 8 の受入条件とする。
//==============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include "ISRDSPHandle.h"  // DSPHandle

#ifdef _MSC_VER
#  pragma warning(push) // C4324 suppression scope begin: DSPHandle alignas(16) による意図的なパディングを許容
#  pragma warning(disable : 4324)
#endif

namespace convo {
namespace isr {

// DSPCore* → DSPHandle 前方マップ（open addressing, 固定容量）
class DSPHandleTable {
public:
    // 2の冪・2x MAX_DSP_SLOTS（負荷率 ≤0.5 で O(1) expected を維持）
    static constexpr std::uint32_t kCapacity = 512;
    static constexpr std::uint32_t kMask = kCapacity - 1;

    struct Entry {
        void* key = nullptr;
        DSPHandle value;
        bool occupied = false;
    };

    // 前方検索: key → DSPHandle（見つかれば out に設定して true）
    [[nodiscard]] bool find(void* key, DSPHandle& out) const noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            const auto& e = entries_[(idx + i) & kMask];
            if (!e.occupied)
                return false;  // 空 slot 到達 = 存在しない
            if (e.key == key) {
                out = e.value;
                return true;
            }
        }
        return false;
    }

    // 前方挿入（既存 key は value 更新）
    bool insert(void* key, const DSPHandle& value) noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            auto& e = entries_[(idx + i) & kMask];
            if (!e.occupied) {
                e.key = key;
                e.value = value;
                e.occupied = true;
                ++count_;
                return true;
            }
            if (e.key == key) {
                e.value = value;
                return true;
            }
        }
        return false;  // full（容量枯渇 — 通常到達しない）
    }

    // 前方削除（key → erase）
    bool erase(void* key) noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            auto& e = entries_[(idx + i) & kMask];
            if (!e.occupied)
                return false;
            if (e.key == key) {
                e.occupied = false;
                e.key = nullptr;
                e.value = DSPHandle{};
                --count_;
                return true;
            }
        }
        return false;
    }

    // 後方検索+削除: DSPHandle 一致エントリを探し、key を返して削除（O(n)・固定 512 上限）。
    //   DSPLifetimeManager::retireByHandle 用（value 一致 + key 取得 + erase を一括）。
    //   戻り値: 発見時は key を outKey に設定して true（エントリ削除済み）。
    [[nodiscard]] bool findAndEraseByHandle(const DSPHandle& handle, void*& outKey) noexcept
    {
        if (handle.isNull())
            return false;
        for (auto& e : entries_) {
            if (e.occupied && e.value == handle) {
                outKey = e.key;
                e.occupied = false;
                e.key = nullptr;
                e.value = DSPHandle{};
                --count_;
                return true;
            }
        }
        return false;
    }

    // 後方削除のみ（value 一致 → erase）。rollbackDSPHandleRegistration 用。
    [[nodiscard]] bool eraseByHandle(const DSPHandle& handle) noexcept
    {
        void* unused = nullptr;
        return findAndEraseByHandle(handle, unused);
    }

    [[nodiscard]] std::uint32_t size() const noexcept { return count_; }

private:
    static std::uint32_t hashKey(const void* key) noexcept
    {
        // ポインタの下位 4bit は alignas により 0 のことが多いため 4bit シフトで分散
        return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(key) >> 4);
    }

    std::array<Entry, kCapacity> entries_{};
    std::uint32_t count_ = 0;
};

} // namespace isr
} // namespace convo

#ifdef _MSC_VER
#  pragma warning(pop) // C4324 suppression scope end
#endif
