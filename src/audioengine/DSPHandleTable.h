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
        bool occupied = false;  // slot に内容（live または tombstone）がある
        bool deleted = false;   // ★ 監査指摘 (work88): tombstone（erase 済み。find は探索を継続）
    };

    // 前方検索: key → DSPHandle（見つかれば out に設定して true）。
    //   トゥームストーン（deleted）は飛ばして探索を継続し、真の空（!occupied）で終了。
    //   ★ 監査指摘 (work88): erase を tombstone 化しないと open addressing のクラスタ断絶が
    //     起き、同じバケットの後続エントリを find できなくなる（重複登録の原因）。
    [[nodiscard]] bool find(void* key, DSPHandle& out) const noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            const auto& e = entries_[(idx + i) & kMask];
            if (!e.occupied)
                return false;  // 真の空 slot 到達 = クラスタ終端 = 存在しない
            if (!e.deleted && e.key == key) {
                out = e.value;
                return true;
            }
        }
        return false;
    }

    // 前方挿入（既存 live key は value 更新。トゥームストーンは再利用）。
    bool insert(void* key, const DSPHandle& value) noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        std::int64_t firstAvail = -1;  // 最初の再利用可能 slot（tombstone または真の空）
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            auto& e = entries_[(idx + i) & kMask];  // insert は非constメンバのため const_cast 不要（LINT-AE-013）
            if (!e.occupied) {
                if (firstAvail < 0)
                    firstAvail = static_cast<std::int64_t>(i);
                break;  // クラスタ終端
            }
            if (e.deleted) {
                if (firstAvail < 0)
                    firstAvail = static_cast<std::int64_t>(i);
                continue;
            }
            if (e.key == key) {
                // 既存 live エントリの更新（同じ key は同一 DSP とみなす）
                e.value = value;
                return true;
            }
        }
        if (firstAvail < 0)
            return false;  // full（容量枯渇 — 通常到達しない）

        auto& slot = entries_[(idx + static_cast<std::uint32_t>(firstAvail)) & kMask];
        slot.key = key;
        slot.value = value;
        slot.occupied = true;
        slot.deleted = false;
        ++count_;
        return true;
    }

    // 前方削除（key → erase。トゥームストーン化する）。
    bool erase(void* key) noexcept
    {
        if (key == nullptr)
            return false;
        std::uint32_t idx = hashKey(key);
        for (std::uint32_t i = 0; i < kCapacity; ++i) {
            auto& e = entries_[(idx + i) & kMask];
            if (!e.occupied)
                return false;  // 真の空 = 存在しない
            if (!e.deleted && e.key == key) {
                e.key = nullptr;
                e.value = DSPHandle{};
                e.deleted = true;  // トゥームストーン（occupied は維持 → クラスタ断絶なし）
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
            if (e.occupied && !e.deleted && e.value == handle) {
                outKey = e.key;
                e.key = nullptr;
                e.value = DSPHandle{};
                e.deleted = true;  // トゥームストーン化（クラスタ断絶防止）
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
