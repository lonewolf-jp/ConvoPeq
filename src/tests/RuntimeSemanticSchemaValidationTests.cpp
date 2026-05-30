#include <stdexcept>

#include "audioengine/ISRRuntimeSemanticSchema.h"

namespace {

[[nodiscard]] bool testRoutingProcessingOrderRange()
{
    convo::isr::RoutingSemantic validRouting {};
    validRouting.processingOrder = 1;
    if (!convo::isr::isValidRoutingSemantic(validRouting))
        return false;

    convo::isr::RoutingSemantic invalidRouting {};
    invalidRouting.processingOrder = 2;
    if (convo::isr::isValidRoutingSemantic(invalidRouting))
        return false;

    return true;
}

[[nodiscard]] bool testExecutionSemanticRangeAndNonNegativeFields()
{
    convo::isr::ExecutionSemantic validExecution {};
    validExecution.transitionPolicy = 2;
    validExecution.crossfadeStartDelayBlocks = 0;
    validExecution.crossfadeDryHoldSamples = 0;
    if (!convo::isr::isValidExecutionSemantic(validExecution))
        return false;

    convo::isr::ExecutionSemantic invalidPolicy {};
    invalidPolicy.transitionPolicy = 3;
    if (convo::isr::isValidExecutionSemantic(invalidPolicy))
        return false;

    convo::isr::ExecutionSemantic invalidDelay {};
    invalidDelay.transitionPolicy = 1;
    invalidDelay.crossfadeStartDelayBlocks = -1;
    if (convo::isr::isValidExecutionSemantic(invalidDelay))
        return false;

    convo::isr::ExecutionSemantic invalidHold {};
    invalidHold.transitionPolicy = 1;
    invalidHold.crossfadeDryHoldSamples = -1;
    if (convo::isr::isValidExecutionSemantic(invalidHold))
        return false;

    return true;
}

} // namespace

int main()
{
    if (!testRoutingProcessingOrderRange())
        throw std::runtime_error("routing semantic range validation failed");

    if (!testExecutionSemanticRangeAndNonNegativeFields())
        throw std::runtime_error("execution semantic validation failed");

    return 0;
}
