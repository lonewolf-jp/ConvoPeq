#include "ISRDSPHandle.h"
#include "AtomicAccess.h"

#include <cassert>
#include <fstream>

namespace convo {
namespace isr {

DSPHandleRuntime::DSPHandleRuntime()
{
    for (size_t i = 0; i < MAX_DSP_SLOTS; ++i) {
        convo::publishAtomic(registry_[i].generation, 0u, std::memory_order_relaxed);
        registry_[i].instance = nullptr;
        convo::publishAtomic(registry_[i].state, DSPState::Reclaimed, std::memory_order_relaxed);
    }
}

DSPHandleRuntime::~DSPHandleRuntime() = default;

DSPHandle DSPHandleRuntime::create(void* dspInstance)
{
    for (size_t slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
        auto& reg = registry_[slot];
        if (convo::consumeAtomic(reg.state, std::memory_order_acquire) == DSPState::Reclaimed) {
            const auto gen = convo::consumeAtomic(reg.generation, std::memory_order_acquire) + 1u;
            reg.instance = dspInstance;
            convo::publishAtomic(reg.generation, gen, std::memory_order_release);
            convo::publishAtomic(reg.state, DSPState::Constructing, std::memory_order_release);
            return DSPHandle{ static_cast<uint32_t>(slot), gen };
        }
    }

    assert(false && "DSP registry exhausted");
    return DSPHandle::null();
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

CrossfadeId DSPHandleRuntime::beginCrossfade(DSPHandle from, DSPHandle to)
{
    assert(!from.isNull() && !to.isNull());
    convo::publishAtomic(registry_[from.slot].state, DSPState::CrossfadingOut, std::memory_order_release);
    convo::publishAtomic(registry_[to.slot].state, DSPState::CrossfadingIn, std::memory_order_release);

    const auto id = convo::fetchAddAtomic(nextCrossfadeId_, 1u, std::memory_order_acq_rel);
    crossfadeRecords_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    convo::publishAtomic(fadingRuntimeDSPHandle_, from, std::memory_order_release);
    return id;
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
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        registry_[handle.slot].instance = nullptr;
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Reclaimed, std::memory_order_release);
    }
}

void DSPHandleRuntime::quarantine(DSPHandle handle)
{
    if (!handle.isNull() && handle.slot < MAX_DSP_SLOTS) {
        convo::publishAtomic(registry_[handle.slot].state, DSPState::Quarantined, std::memory_order_release);
    }
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
    records_.push_back(CrossfadeRecord{ id, from, to, 0u, true });
    return id;
}

void CrossfadeAuthorityRuntime::unregisterCrossfade(CrossfadeId id)
{
    for (auto& record : records_) {
        if (record.id == id) {
            record.active = false;
            break;
        }
    }
}

std::vector<CrossfadeRecord> CrossfadeAuthorityRuntime::getActiveCrossfades() const noexcept
{
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
    for (const auto& record : records_) {
        if (record.active && (record.fromHandle == handle || record.toHandle == handle)) {
            return true;
        }
    }
    return false;
}

} // namespace isr
} // namespace convo
