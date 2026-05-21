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
    convo::publishAtomic(fadingDSP_, from, std::memory_order_release);
    return id;
}

void DSPHandleRuntime::activate(DSPHandle handle)
{
    if (handle.isNull() || handle.slot >= MAX_DSP_SLOTS) {
        return;
    }

    convo::publishAtomic(registry_[handle.slot].state, DSPState::Active, std::memory_order_release);
    convo::publishAtomic(activeDSP_, handle, std::memory_order_release);
    convo::publishAtomic(fadingDSP_, DSPHandle::null(), std::memory_order_release);
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
        convo::publishAtomic(activeDSP_, record.toHandle, std::memory_order_release);
        convo::publishAtomic(fadingDSP_, DSPHandle::null(), std::memory_order_release);
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

DSPHandle DSPHandleRuntime::getActiveDSP() const noexcept
{
    return convo::consumeAtomic(activeDSP_, std::memory_order_acquire);
}

DSPHandle DSPHandleRuntime::getFadingDSP() const noexcept
{
    return convo::consumeAtomic(fadingDSP_, std::memory_order_acquire);
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
