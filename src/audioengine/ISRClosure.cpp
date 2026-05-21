#include "ISRClosure.h"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace convo {
namespace isr {

bool ClosureValidator::validateClosureGraph(const PayloadClosureDescriptor& closure) const noexcept
{
    if (closure.closureId == 0) {
        return false;
    }

    if (closure.nodes.empty()) {
        return false;
    }

    if ((closure.edges.size() % 2u) != 0u) {
        return false;
    }

    std::unordered_set<uint32_t> nodeIds;
    nodeIds.reserve(closure.nodes.size());

    for (const auto& node : closure.nodes) {
        if (node.nodeId == 0u) {
            return false;
        }

        if (node.kind == 0u
            || node.ownership == 0u
            || node.mutability == 0u
            || node.lifetime == 0u
            || node.hbDomain == 0u
            || node.authority == 0u
            || node.allocator == 0u) {
            return false;
        }

        if (!nodeIds.insert(node.nodeId).second) {
            return false;
        }
    }

    if (closure.externalMutableDependencies != 0u) {
        return false;
    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> graph;
    graph.reserve(closure.nodes.size());

    for (std::size_t i = 0; i < closure.edges.size(); i += 2u) {
        const uint32_t from = closure.edges[i];
        const uint32_t to = closure.edges[i + 1u];

        if (nodeIds.find(from) == nodeIds.end() || nodeIds.find(to) == nodeIds.end()) {
            return false;
        }

        graph[from].push_back(to);
    }

    enum class Color : uint8_t { White = 0, Gray = 1, Black = 2 };
    std::unordered_map<uint32_t, Color> color;
    color.reserve(closure.nodes.size());
    for (const auto& id : nodeIds) {
        color.emplace(id, Color::White);
    }

    std::function<bool(uint32_t)> dfs = [&](uint32_t nodeId) -> bool {
        color[nodeId] = Color::Gray;
        const auto iter = graph.find(nodeId);
        if (iter != graph.end()) {
            for (const uint32_t next : iter->second) {
                const auto state = color[next];
                if (state == Color::Gray) {
                    return false;
                }
                if (state == Color::White && !dfs(next)) {
                    return false;
                }
            }
        }
        color[nodeId] = Color::Black;
        return true;
    };

    for (const auto& id : nodeIds) {
        if (color[id] == Color::White && !dfs(id)) {
            return false;
        }
    }

    return true;
}

void ClosureValidator::registerClosure(const PayloadClosureDescriptor& closure)
{
    if (validateClosureGraph(closure)) {
        registeredClosures_.push_back(closure);
    }
}

bool ClosureValidator::validateAllClosures() const noexcept
{
    for (const auto& closure : registeredClosures_) {
        if (!validateClosureGraph(closure)) {
            return false;
        }
    }
    return true;
}

}  // namespace isr
}  // namespace convo
