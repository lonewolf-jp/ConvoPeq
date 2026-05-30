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
#include "ISRRuntimeSemanticSchema.h"

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
    enum class CoordinatorState : uint8_t {
        Bootstrapping = 0,
        Ready,
        Publishing,
        Transitioning,
        Pressure,
        ShuttingDown,
        Faulted
    };

    RuntimePublicationCoordinator();
    bool precheckPublish(const PayloadClosureDescriptor& closure,
                         const TieredPayloadDescriptor& descriptor) noexcept;
    const char* lastRejectReason() const noexcept;
    void commit(PublishAuthority, RuntimeBoundary boundary, const void* newWorld, std::uint64_t version);
    void commit(PublishAuthority,
                RuntimeBoundary boundary,
                const void* newWorld,
                std::uint64_t version,
                PublicationSequenceId sequenceId,
                PublicationEpoch epoch,
                std::uint64_t mappedGeneration);
    void retire(RetireAuthority, RuntimeBoundary boundary, const void* oldWorld);
    const void* getCurrent() const noexcept;
    std::uint64_t getVersion() const noexcept;
    void setRetireBacklogCount(std::uint64_t count) noexcept;
    void setPublicationBacklogCount(std::uint64_t count) noexcept;
    void setPendingIntentCount(std::uint64_t count) noexcept;
    void setFallbackBacklogCount(std::uint64_t count) noexcept;
    void setReclaimInFlightCount(std::uint64_t count) noexcept;
    void setDeferredRetireResidencyCount(std::uint64_t count) noexcept;
    void setSwapPending(bool pending) noexcept;
    [[nodiscard]] bool isSwapPending() const noexcept;
    [[nodiscard]] std::uint64_t getReclaimInFlightCount() const noexcept;
    [[nodiscard]] bool isFullyDrained() const noexcept;
    [[nodiscard]] CoordinatorState getState() const noexcept;
    void markTransitionStart() noexcept;
    void markTransitionCommitted() noexcept;
    void requestShutdown() noexcept;
    void markShutdownComplete() noexcept;
private:
    enum class RejectCode : uint8_t {
        None = 0,
        InvalidClosure,
        InvalidPayloadTier
    };

    std::atomic<const void*> currentWorld_;
    std::atomic<std::uint64_t> version_;
    std::atomic<PublicationSequenceId> publicationSequenceId_;
    std::atomic<PublicationEpoch> publicationEpoch_;
    std::atomic<std::uint64_t> mappedRuntimeGeneration_;
    std::atomic<RejectCode> lastRejectCode_;
    std::atomic<std::uint64_t> retireBacklogCount_;
    std::atomic<std::uint64_t> publicationBacklogCount_;
    std::atomic<std::uint64_t> pendingIntentCount_;
    std::atomic<std::uint64_t> fallbackBacklogCount_;
    std::atomic<std::uint64_t> reclaimInFlightCount_;
    std::atomic<std::uint64_t> deferredRetireResidencyCount_;
    std::atomic<std::uint64_t> previousRetireBacklogCount_;
    std::atomic<std::uint32_t> pressureNormalizedWindows_;
    std::atomic<bool> swapPending_;
    std::atomic<CoordinatorState> state_;
    std::mutex retireGuard_;
    std::vector<const void*> retiredWorlds_;
    static constexpr std::size_t kMaxRetiredWorldResidency = 4096;
    static constexpr std::uint64_t kPressureSlopeThreshold = 8;
    static constexpr std::uint32_t kPressureNormalizeWindows = 3;
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

    [[nodiscard]] std::size_t size() noexcept;

private:
    std::vector<const void*> queued_;
    std::mutex guard_;
};

} // namespace convo::isr
