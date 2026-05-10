#pragma once

#include <cstdint>
#include <type_traits>

namespace convo {

enum class CommandType : std::uint8_t {
    UpdateParameters,
    ReplaceIR,
    ChangeSampleRate,
    ChangeOversampling,
    SuspendProcessing,
    ResumeProcessing,
    Shutdown
};

struct CommandMeta {
    std::uint64_t revision = 0;
    std::uint64_t issuedTick = 0;
    bool highPriority = false;
};

// T1 では queue 基盤のみを導入するため、payload は固定長の最小表現に留める。
// 詳細な build 入力は T2 以降で専用型へ分離する。
struct EngineCommand {
    CommandType type = CommandType::UpdateParameters;
    CommandMeta meta {};
    double sampleRate = 0.0;
    int blockSize = 0;
    int intValue = 0;
    int oversamplingFactor = 0;
    int oversamplingType = 0;
    int noiseShaperType = 0;
    std::uint64_t payloadHash = 0;
};

static_assert(std::is_trivially_copyable_v<CommandMeta>, "CommandMeta must be trivially copyable");
static_assert(std::is_trivially_copyable_v<EngineCommand>, "EngineCommand must be trivially copyable");

} // namespace convo
