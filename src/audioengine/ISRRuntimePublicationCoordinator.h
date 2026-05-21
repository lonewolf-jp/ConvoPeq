#pragma once
#include <atomic>
#include <memory>
#include <cstdint>
#include <vector>
#include <mutex>
#include "ISRClosure.h"
#include "ISRPayloadTier.h"
#include "ISRSealedObject.h"
#include "ISRRetire.h"
#include "ISRHB.h"
#include "ISRShutdown.h"

namespace convo::isr {

enum class PublishAuthority : uint8_t { Granted = 1 };
enum class RetireAuthority : uint8_t { Granted = 1 };
enum class ShutdownAuthority : uint8_t { Granted = 1 };

enum class RuntimeBoundary : uint8_t {
    RTWorld,
    NonRTWorld
};

class RuntimePublicationCoordinator {
public:
    RuntimePublicationCoordinator();
    bool precheckPublish(const PayloadClosureDescriptor& closure,
                         const TieredPayloadDescriptor& descriptor) noexcept;
    const char* lastRejectReason() const noexcept;
    void commit(PublishAuthority, RuntimeBoundary boundary, const void* newWorld, std::uint64_t version);
    void retire(RetireAuthority, RuntimeBoundary boundary, const void* oldWorld);
    const void* getCurrent() const noexcept;
    std::uint64_t getVersion() const noexcept;
private:
    enum class RejectCode : uint8_t {
        None = 0,
        InvalidClosure,
        InvalidPayloadTier
    };

    std::atomic<const void*> currentWorld_;
    std::atomic<std::uint64_t> version_;
    std::atomic<RejectCode> lastRejectCode_;
    std::mutex retireGuard_;
    std::vector<const void*> retiredWorlds_;
};

class MultiStagePublisher {
public:
    explicit MultiStagePublisher(RuntimeBoundary boundary = RuntimeBoundary::NonRTWorld) : boundary_(boundary) {}
    void publishTier(PayloadTier tier, const void* payload);
    [[nodiscard]] bool wasRejected() const noexcept { return rejected_; }

private:
    RuntimeBoundary boundary_;
    bool rejected_ = false;
};

class PublicationBuffer {
public:
    void enqueue(const void* world);
    void retireOld();

    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::vector<const void*> queued_;
    mutable std::mutex guard_;
};

} // namespace convo::isr
