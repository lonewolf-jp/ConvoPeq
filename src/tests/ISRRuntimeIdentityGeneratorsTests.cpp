#include <cstdint>
#include <stdexcept>

#include "audioengine/ISRRuntimeIdentityGenerators.h"
#include "audioengine/ISRRuntimeSemanticSchema.h"

namespace {

[[nodiscard]] bool testWorldIdMonotonic()
{
    convo::isr::RuntimeWorldIdGenerator gen;
    std::uint64_t prev = 0;
    for (int i = 0; i < 1024; ++i)
    {
        const auto v = gen.next();
        if (v <= prev)
            return false;
        prev = v;
    }
    return true;
}

[[nodiscard]] bool testGenerationMonotonic()
{
    convo::isr::RuntimeGenerationGenerator gen;
    std::uint64_t prev = 0;
    for (int i = 0; i < 1024; ++i)
    {
        const auto v = gen.next();
        if (v <= prev)
            return false;
        prev = v;
    }
    return true;
}

[[nodiscard]] bool testRuntimeMetadataDefaults()
{
    convo::isr::RuntimeMetadata meta;
    if (meta.schemaVersion != convo::isr::kRuntimeSemanticSchemaVersion)
        return false;
    if (meta.publicationSequence != 0)
        return false;
    return true;
}

} // namespace

int main()
{
    if (!testWorldIdMonotonic())
        throw std::runtime_error("RuntimeWorldIdGenerator monotonicity failed");

    if (!testGenerationMonotonic())
        throw std::runtime_error("RuntimeGenerationGenerator monotonicity failed");

    if (!testRuntimeMetadataDefaults())
        throw std::runtime_error("RuntimeMetadata defaults failed");

    return 0;
}
