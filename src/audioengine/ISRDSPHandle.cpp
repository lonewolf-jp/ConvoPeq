#include "ISRDSPHandle.h"
#include "AtomicAccess.h"

#include <cassert>
#include <fstream>

namespace convo {
namespace isr {

DSPHandleRuntime::DSPHandleRuntime()
{
    // Runtime初期化時に atomic<DSPHandle> のロックフリー性を一度だけ検証
    static const bool isLockFree = []{
        std::atomic<DSPHandle> test{ DSPHandle::null() };
        const bool ok = test.is_lock_free();
        // MSVC では 16バイト atomic の is_lock_free()/is_always_lock_free() が
        // false を返す（STL の保宅的判定）。実際は InterlockedCompareExchange128
        // (CMPXCHG16B) で lock-free に動作するため、MSVC ではアサートを回避する。
        // Clang/GCC x64 では alignas(16) により is_lock_free()==true が保証される。
        // see ISRDSPHandle.h:174-182 (ADR-005)。
#if defined(_MSC_VER)
        (void)ok;
#else
        assert(ok && "atomic<DSPHandle> must be lock-free on x64 for ISR Runtime");
#endif
        return ok;
    }();
    (void)isLockFree; // unused in Release

    for (size_t i = 0; i < MAX_DSP_SLOTS; ++i) {
        convo::publishAtomic(registry_[i].generation, 0u, std::memory_order_relaxed);
        registry_[i].instance = nullptr;
        convo::publishAtomic(registry_[i].state, DSPState::Reclaimed, std::memory_order_relaxed);
    }

    // ★ FUTURE-5 (work88): フリーリスト初期化（slot 1..255。slot 0 は null handle 表現のため除外）
    freeSize_ = 0;
    for (uint32_t slot = 1; slot < MAX_DSP_SLOTS; ++slot)
        freeSlots_[freeSize_++] = slot;
}

DSPHandleRuntime::~DSPHandleRuntime() = default;

DSPHandle DSPHandleRuntime::create(void* dspInstance)
{
    // ★ FUTURE-5 (work88): 線形スキャン → フリーリスト pop（O(1) 確保）
    std::lock_guard<std::mutex> lock(freeListMutex_);
    if (freeSize_ == 0) {
        assert(false && "DSP registry exhausted");
        return DSPHandle::null();
    }
    const uint32_t slot = freeSlots_[--freeSize_];
    auto& reg = registry_[slot];
    const auto gen = convo::consumeAtomic(reg.generation, std::memory_order_acquire) + 1u;
    reg.instance = dspInstance;
    convo::publishAtomic(reg.generation, gen, std::memory_order_release);
    convo::publishAtomic(reg.state, DSPState::Constructing, std::memory_order_release);
    return DSPHandle{ slot, gen };
}

ResolvedDSP DSPHandleRuntime::resolve(DSPHandle handle) const noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) {
        return { nullptr, false, false };
    }

    const auto& reg = registry_[handle.slot];
    const auto currentGen = convo::consumeAtomic(reg.generation, std::memory_order_acquire);
    if (currentGen != handle.generation) {
        return { nullptr, false, true };
    }

    const auto state = convo::consumeAtomic(reg.state, std::memory_order_acquire);
    if (state == DSPState::Reclaimed || state == DSPState::Quarantined) {
        return { nullptr, false, false };
    }

    return { reg.instance, true, false };
}

void DSPHandleRuntime::beginCrossfade(DSPHandle from, DSPHandle to, CrossfadeId id)
{
    assert(!from.isNull() && !to.isNull());
    convo::publishAtomic(registry_[from.slot].state, DSPState::CrossfadingOut, std::memory_order_release);
    convo::publishAtomic(registry_[to.slot].state, DSPState::CrossfadingIn, std::memory_order_release);

    // ★ 監査指摘 (work88): push_back と Timer 側の endCrossfade/isSlotInCrossfade 走査の競合を防ぐ
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    crossfadeRecords_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    convo::publishAtomic(fadingRuntimeDSPHandle_, from, std::memory_order_release);
}

void DSPHandleRuntime::activate(DSPHandle handle)
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) {
        return;
    }

    convo::publishAtomic(registry_[handle.slot].state, DSPState::Active, std::memory_order_release);
    convo::publishAtomic(activeRuntimeDSPHandle_, handle, std::memory_order_release);
    convo::publishAtomic(fadingRuntimeDSPHandle_, DSPHandle::null(), std::memory_order_release);
}

void DSPHandleRuntime::endCrossfade(CrossfadeId id)
{
    // ★ 監査指摘 (work88): beginCrossfade（CoordinatorLoop）との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    for (auto& record : crossfadeRecords_) {
        if (record.id != id || !record.active) {
            continue;
        }

        record.active = false;
        convo::publishAtomic(registry_[record.fromHandle.slot].state, DSPState::Retired, std::memory_order_release);
        convo::publishAtomic(registry_[record.toHandle.slot].state, DSPState::Active, std::memory_order_release);
        convo::publishAtomic(activeRuntimeDSPHandle_, record.toHandle, std::memory_order_release);
        convo::publishAtomic(fadingRuntimeDSPHandle_, DSPHandle::null(), std::memory_order_release);
        break;
    }
}

void DSPHandleRuntime::retire(DSPHandle handle)
{
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Retired, std::memory_order_release);
    }
}

void DSPHandleRuntime::reclaim(DSPHandle handle)
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS)
        return;
    std::lock_guard<std::mutex> lock(freeListMutex_);
    auto& reg = registry_[handle.slot];
    // ★ 監査指摘 (work88): stale handle 検出（generation 不一致なら slot は別世代で再利用中 —
    //   誤って新規 DSP の slot を Reclaimed 化しない）。FUTURE-5 generation タグの完全化。
    if (convo::consumeAtomic(reg.generation, std::memory_order_acquire) != handle.generation)
        return;
    // ★ 監査指摘 (work88): 二重 reclaim 防止（既に Reclaimed → free list へ重複 push せず、
    //   同一 slot の二重割当（double-allocation）を構造的に排除）。
    if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed)
        return;
    reg.instance = nullptr;
    convo::publishAtomic(reg.state, DSPState::Reclaimed, std::memory_order_release);
    // ★ FUTURE-5 (work88): Reclaimed 済み slot をフリーリストへ戻す（O(1) 再利用）
    if (handle.slot != 0 && freeSize_ < MAX_DSP_SLOTS)
        freeSlots_[freeSize_++] = handle.slot;
}

void DSPHandleRuntime::quarantine(DSPHandle handle)
{
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Quarantined, std::memory_order_release);
    }
}

bool DSPHandleRuntime::rollbackRegistration(DSPHandle handle) noexcept
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) return false;
    auto& reg = registry_[handle.slot];
    DSPState expected = DSPState::Constructing;
    // state のみ CAS（instance は不変）。create() が上書きするため不要。
    const bool ok = convo::compareExchangeAtomic(reg.state, expected, DSPState::Reclaimed,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire);
    // ★ FUTURE-5 (work88): ロールバック成功（Reclaimed 化）時は slot をフリーリストへ戻す
    if (ok) {
        std::lock_guard<std::mutex> lock(freeListMutex_);
        if (handle.slot != 0 && freeSize_ < MAX_DSP_SLOTS)
            freeSlots_[freeSize_++] = handle.slot;
    }
    return ok;
}

// ★ A-1.3: Slot 直接 quarantine — generation 一致を要求しない
void DSPHandleRuntime::quarantineSlot(uint32_t slot) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return;
    convo::publishAtomic(registry_[slot].state, DSPState::Quarantined,
                         std::memory_order_release);
}

// ★ A-1.5: slot が crossfade に関与しているか確認
bool DSPHandleRuntime::isSlotInCrossfade(uint32_t slot) const noexcept
{
    // ★ 監査指摘 (work88): beginCrossfade との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(crossfadeRecordsMutex_);
    for (const auto& record : crossfadeRecords_) {
        if (record.active &&
            (record.fromHandle.slot == slot || record.toHandle.slot == slot))
            return true;
    }
    return false;
}

// ★ A-1.4: shutdown専用解放（2段階: DestroyPending → Reclaimed）
void DSPHandleRuntime::destroyQuarantineSlot(
    uint32_t slot, uint64_t expectedGeneration) noexcept
{
    if (slot >= MAX_DSP_SLOTS)
        return;

    // generation 保護
    if (expectedGeneration != 0) {
        const auto currentGen = convo::consumeAtomic(
            registry_[slot].generation, std::memory_order_acquire);
        if (currentGen != expectedGeneration)
            return;
    }

    // state==Quarantined を表明
    const auto prevState = convo::consumeAtomic(
        registry_[slot].state, std::memory_order_acquire);
    assert(prevState == DSPState::Quarantined);
    if (prevState != DSPState::Quarantined)
        return;

    // Phase 1: 状態チェック — active/fading/crossfade に関与していないか
    const bool activeHandleMatch =
        (convo::consumeAtomic(activeRuntimeDSPHandle_, std::memory_order_acquire).slot == slot);
    const bool fadingHandleMatch =
        (convo::consumeAtomic(fadingRuntimeDSPHandle_, std::memory_order_acquire).slot == slot);
    const bool inCrossfade = isSlotInCrossfade(slot);

    if (activeHandleMatch || fadingHandleMatch || inCrossfade)
        return;

    // Phase 1: DestroyPending マーク（CAS で安全に遷移）
    auto expected = convo::consumeAtomic(
        registry_[slot].state, std::memory_order_acquire);
    while (expected == DSPState::Quarantined) {
        if (convo::compareExchangeAtomic(registry_[slot].state,
                                         expected, DSPState::DestroyPending,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
    }
    if (expected != DSPState::Quarantined)
        return;

    // Phase 2: instance 解放
    registry_[slot].instance = nullptr;
    convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed,
                         std::memory_order_release);
    // ★ 監査指摘 (work88): Reclaimed 化した quarantine slot を free list へ戻す（スロットリーク防止）。
    //   destroyQuarantineSlot は state==Quarantined を確認済みのため二重 push は起きない
    //   （reclaim() 側の state ガードとも整合）。
    std::lock_guard<std::mutex> lock(freeListMutex_);
    if (slot != 0 && freeSize_ < MAX_DSP_SLOTS)
        freeSlots_[freeSize_++] = slot;
}

DSPHandle DSPHandleRuntime::getActiveRuntimeDSPHandle() const noexcept
{
    return convo::consumeAtomic(activeRuntimeDSPHandle_, std::memory_order_acquire);
}

DSPHandle DSPHandleRuntime::getFadingRuntimeDSPHandle() const noexcept
{
    return convo::consumeAtomic(fadingRuntimeDSPHandle_, std::memory_order_acquire);
}

void DSPHandleRuntime::emitOwnershipTrace(const std::filesystem::path& outputPath) const
{
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        return;
    }

    file << "{\n  \"slots\": [\n";
    for (size_t i = 0; i < registry_.size(); ++i) {
        const auto state = convo::consumeAtomic(registry_[i].state, std::memory_order_acquire);
        file << "    { \"slot\": " << i << ", \"state\": " << static_cast<int>(state) << " }";
        if (i + 1u < registry_.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "  ]\n}\n";
}

CrossfadeAuthorityRuntime::CrossfadeAuthorityRuntime() = default;
CrossfadeAuthorityRuntime::~CrossfadeAuthorityRuntime() = default;

CrossfadeId CrossfadeAuthorityRuntime::registerCrossfade(DSPHandle from, DSPHandle to)
{
    const auto id = convo::fetchAddAtomic(nextId_, 1u, std::memory_order_acq_rel);
    // ★ 監査指摘 (work88): push_back による再確保と Timer 側の走査の競合を防ぐため lock。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    records_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    return id;
}

void CrossfadeAuthorityRuntime::unregisterCrossfade(CrossfadeId id)
{
    // ★ 監査指摘 (work88): register（CoordinatorLoop）との並行アクセスを防ぐため lock。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    for (auto& record : records_) {
        if (record.id == id) {
            record.active = false;
            break;
        }
    }
}

std::vector<CrossfadeRecord> CrossfadeAuthorityRuntime::getActiveCrossfades() const noexcept
{
    // ★ 監査指摘 (work88): register との並行アクセスを防ぐため lock（コピーは lock 内）。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    std::vector<CrossfadeRecord> result;
    for (const auto& record : records_) {
        if (record.active) {
            result.push_back(record);
        }
    }
    return result;
}

bool CrossfadeAuthorityRuntime::hasCrossfadeInvolving(DSPHandle handle) const noexcept
{
    // ★ 監査指摘 (work88): 同様に lock（現時点で呼出し元ゼロだが契約として保護）。
    std::lock_guard<std::mutex> lock(recordsMutex_);
    for (const auto& record : records_) {
        if (record.active && (record.fromHandle == handle || record.toHandle == handle)) {
            return true;
        }
    }
    return false;
}

} // namespace isr
} // namespace convo
