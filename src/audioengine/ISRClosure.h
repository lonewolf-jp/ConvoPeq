#pragma once

#include <cstdint>
#include <vector>
#include <functional>

namespace convo {
namespace isr {

/**
 * ISR 10層 Architecture Layer 3: Payload Closure Descriptor
 * JUCE callback での publication graph の integrity を検証する。
 */

/**
 * closure 内の依存ノード
 */
struct ClosureNodeRef
{
    uint32_t nodeId;
    uint32_t payloadTier;  // PayloadTier enum value
    uint32_t kind = 0;
    uint32_t ownership = 0;
    uint32_t mutability = 0;
    uint32_t lifetime = 0;
    uint32_t hbDomain = 0;
    uint32_t authority = 0;
    uint32_t allocator = 0;
};

/**
 * publication graph の整合性を記述する closure
 */
struct PayloadClosureDescriptor
{
    uint32_t closureId;
    std::vector<ClosureNodeRef> nodes;
    std::vector<uint32_t> edges;  // (from, to) pairs
    uint32_t externalMutableDependencies = 0;
};

/**
 * closure graph validator
 */
class ClosureValidator
{
public:
    // closure の integrity をチェック（cycle, dangling ref, etc.）
    bool validateClosureGraph(const PayloadClosureDescriptor& closure) const noexcept;

    // closure を登録
    void registerClosure(const PayloadClosureDescriptor& closure);

    // 全 closure を検証
    bool validateAllClosures() const noexcept;

private:
    std::vector<PayloadClosureDescriptor> registeredClosures_;
};

}  // namespace isr
}  // namespace convo
