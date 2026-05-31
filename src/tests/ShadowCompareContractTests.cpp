#include <stdexcept>

#include "audioengine/ISRRuntimeSemanticSchema.h"

namespace {

[[nodiscard]] bool testShadowCompareEquivalenceContract()
{
    convo::isr::RuntimeSemanticHash base {};
    base.generationSemanticHash = 10;
    base.topologyHash = 20;
    base.executionHash = 30;
    base.routingHash = 40;
    base.payloadHash = 50;
    base.publicationSemanticHash = 60;
    base.overlapSemanticHash = 70;
    base.retireSemanticHash = 80;

    auto equivalent = base;
    if (convo::isr::classifySemanticEquivalence(base, equivalent)
        != convo::isr::SemanticEquivalenceClass::Equivalent)
        return false;

    auto compatible = base;
    compatible.publicationSemanticHash += 1;
    if (convo::isr::classifySemanticEquivalence(base, compatible)
        != convo::isr::SemanticEquivalenceClass::Compatible)
        return false;

    auto different = base;
    different.executionHash += 1;
    if (convo::isr::classifySemanticEquivalence(base, different)
        != convo::isr::SemanticEquivalenceClass::Different)
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testShadowCompareEquivalenceContract())
        throw std::runtime_error("shadow compare contract failed");

    return 0;
}
