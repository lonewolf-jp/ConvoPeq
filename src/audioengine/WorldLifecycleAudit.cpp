#include "WorldLifecycleAudit.h"
#include <fstream>
#include <filesystem>

namespace convo::isr {

void WorldLifecycleAudit::emitSnapshot() const noexcept
{
    const auto outputPath = std::filesystem::current_path() / "evidence" / "world_lifecycle_audit.json";
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        return;

    const uint64_t active = convo::consumeAtomic(activeWorldCount_, std::memory_order_acquire);
    const uint64_t published = convo::consumeAtomic(publishedCount_, std::memory_order_acquire);
    const uint64_t retired = convo::consumeAtomic(retiredCount_, std::memory_order_acquire);

    // RingBuffer から最新のレコードを取得
    constexpr size_t kSnapshotCount = 64;
    WorldLifecycleRecord records[kSnapshotCount];
    const size_t count = ringBuffer_.readLatest(records, kSnapshotCount);

    file << "{\n";
    file << "  \"schema\": \"world_lifecycle_audit_v1\",\n";
    file << "  \"activeWorldCount\": " << active << ",\n";
    file << "  \"publishedCount\": " << published << ",\n";
    file << "  \"retiredCount\": " << retired << ",\n";
    file << "  \"ringBufferSize\": " << ringBuffer_.size() << ",\n";
    file << "  \"ringBufferCapacity\": " << ringBuffer_.capacity() << ",\n";
    file << "  \"recentRecords\": [\n";

    for (size_t i = 0; i < count; ++i) {
        const auto& rec = records[i];
        file << "    {\n";
        file << "      \"worldId\": " << rec.worldId << ",\n";
        file << "      \"publishEpoch\": " << rec.publishEpoch << ",\n";
        file << "      \"retireEpoch\": " << rec.retireEpoch << ",\n";
        file << "      \"publishTimestampUs\": " << rec.publishTimestampUs << ",\n";
        file << "      \"retireTimestampUs\": " << rec.retireTimestampUs << ",\n";
        file << "      \"correlationIdShort\": " << rec.correlationId.shortValue() << "\"\n";
        file << "    }";
        if (i < count - 1)
            file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
}

} // namespace convo::isr
