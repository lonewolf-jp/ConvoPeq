// STG11D12StateIntegerGuardTests.cpp - STG-11-D12-1 / D12-3 / D12-4 regression.
//
// Target defect (STG-11-D12-1):
//   AudioEngine::setDitherBitDepth stored its argument unconditionally. The
//   authoritative legal set is {0,16,24,32} (RuntimePublicationValidator.cpp
//   123-126, kAdaptiveBitDepthValues plus the Off sentinel), but neither the
//   setter nor AudioEngine.StateIO.cpp:100-101 checked the range, so a corrupted
//   or future-version session could store an arbitrary integer. The value is
//   written to the ditherBitDepth / m_currentDitherBitDepth atomics *before* any
//   publish decision, and captureBuildParameterSnapshot (RebuildDispatch.cpp:46)
//   re-reads it on every rebuild. validateResources then rejects the world,
//   RuntimePublicationCoordinator::publishWorld discards it and
//   PublicationExecutor::publishImpl performs no recovery. Because nothing
//   restores the value, the rebuild/publish path stays permanently rejected and
//   every later structural parameter change silently stops landing.
//
// Target defect (STG-11-D12-3):
//   DeviceSettings::updateBitDepthList injected the audio device's
//   getCurrentBitDepth() into the combo list and passed the maximum to
//   setDitherBitDepth. A device reporting more than 32 bits therefore produced a
//   value outside the authoritative set with no corrupted session involved, so
//   UI domain diverged from setter domain and validator domain.
//
// Target gap (STG-11-D12-4):
//   The only place that pinned {0,16,24,32} was
//   src/tests/PublicationValidatorIsolationTests.cpp, which CMakeLists.txt never
//   referenced, so it was never built. The contract was pinned by nothing that
//   runs. D12-4 pins it here against the real validator.
//
// Fix:
//   Reject out-of-set values at the setter boundary, keeping the previous value
//   (the D7 / D10 contract). No clamp: silently mapping 64 to 32 would make the
//   UI / setter / validator agreement accidental. Remove the device bit depth
//   injection from the UI list.
//
// Test contract:
//   D12-1-A  Invalid dither (8, -1, 64, 99) via setter and via session restore
//            leaves the authoritative value unchanged.
//   D12-1-B  Valid dither (0, 16, 24, 32) round-trips unchanged.
//   D12-1-C  After an invalid dither, a subsequent normal structural parameter
//            change still reaches a committed publication. This is the core
//            regression for the sticky-reject defect.
//   D12-3    The UI bit depth list is sourced from the authoritative set only and
//            keeps the existing Off / 16 / 24 / 32 selection semantics.
//   D12-4    validateResources accepts exactly {0,16,24,32} and rejects
//            {8,64,99} with ValidationFailureReason::InvalidResources.
//
// Negative controls:
//   Removing the setter guard makes D12-1-A and D12-1-C fail.
//   Restoring the device bit depth injection makes D12-3 fail.
//
// No new CTest registration: sub-test of AudioEngineHarness, STG-8/9/D3-D5/D7-D11
// shape. The fixture is owned by the harness (AudioEngine ~4MB must not live on
// TU stack).
// =============================================================================

#include <cstdio>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

#include "audioengine/AudioEngine.h"
#include "audioengine/RuntimePublicationValidator.h"
#include "AudioEngineHarness.h"

namespace {

// ── authoritative legal set (RuntimePublicationValidator.cpp:123-126) ──
constexpr int kLegalDither[]   = { 0, 16, 24, 32 };
constexpr int kIllegalDither[] = { 8, -1, 64, 99 };

[[nodiscard]] bool isLegalDither(int value) noexcept
{
    for (const int legal : kLegalDither)
        if (legal == value)
            return true;
    return false;
}

template <typename Predicate>
bool waitUntil(double timeoutSeconds, Predicate pred)
{
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(static_cast<long long>(timeoutSeconds * 1000.0));
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

// ── D12-1-A: invalid dither never reaches the authoritative value ──
bool checkD12T1InvalidDitherRejected()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12 T-1] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    bool ok = true;

    // Establish a known legal baseline through the setter.
    e.setDitherBitDepth(24);
    if (e.getDitherBitDepth() != 24)
    {
        std::fprintf(stderr, "D12-1-A: legal baseline 24 not accepted (got %d)\n",
                     e.getDitherBitDepth());
        ok = false;
    }

    // Path 1: the setter itself must reject.
    for (const int bad : kIllegalDither)
    {
        e.setDitherBitDepth(bad);
        if (e.getDitherBitDepth() != 24)
        {
            std::fprintf(stderr, "D12-1-A: setter accepted %d (value now %d)\n",
                         bad, e.getDitherBitDepth());
            ok = false;
        }
    }

    // Path 2: the session restore path must not be a way around the setter.
    const juce::ValueTree saved = e.getCurrentState();
    if (!saved.isValid())
    {
        std::fprintf(stderr, "D12-1-A: getCurrentState invalid\n");
        return false;
    }
    for (const int bad : kIllegalDither)
    {
        juce::ValueTree v = saved.createCopy();
        v.setProperty("ditherBitDepth", bad, nullptr);
        e.requestLoadState(v);
        if (e.getDitherBitDepth() != 24)
        {
            std::fprintf(stderr, "D12-1-A: session restore applied %d (value now %d)\n",
                         bad, e.getDitherBitDepth());
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12StateIntegerGuardTests: D12-1-A PASS (invalid dither rejected)\n");
    return ok;
}

// ── D12-1-B: valid dither round-trips ──
bool checkD12T2ValidRoundTrip()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12 T-2] harness start failed\n");
        return false;
    }
    h.stop();
    AudioEngine& e = h.engine();

    bool ok = true;
    for (const int legal : kLegalDither)
    {
        e.setDitherBitDepth(legal);
        if (e.getDitherBitDepth() != legal)
        {
            std::fprintf(stderr, "D12-1-B: setter rejected legal %d (value now %d)\n",
                         legal, e.getDitherBitDepth());
            ok = false;
            continue;
        }

        // Round-trip through the session serialisation.
        const juce::ValueTree saved = e.getCurrentState();
        e.setDitherBitDepth(legal == 0 ? 24 : (legal == 24 ? 16 : 24));
        e.requestLoadState(saved);
        if (e.getDitherBitDepth() != legal)
        {
            std::fprintf(stderr, "D12-1-B: round-trip lost legal %d (value now %d)\n",
                         legal, e.getDitherBitDepth());
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12StateIntegerGuardTests: D12-1-B PASS (legal dither round-trips)\n");
    return ok;
}

// ── D12-1-C: a later structural change still publishes after an invalid input ──
// This is the core regression. Before the setter guard, the invalid value stayed
// in the atomics, every subsequent rebuild produced a world that
// validateResources rejected, and the committed publication sequence stopped
// advancing for the rest of the engine lifetime.
bool checkD12T3PublishRecoversAfterInvalid()
{
    AudioEngineHarness h;
    if (!h.start(48000.0, 512))
    {
        std::fprintf(stderr, "[STG-11-D12 T-3] harness start failed\n");
        return false;
    }
    AudioEngine& e = h.engine();

    auto committedSeq = [&e] {
        return static_cast<unsigned long long>(e.getLastCommittedPublicationSequence());
    };

    // Let the startup publications settle so the baseline is stable.
    unsigned long long previous = committedSeq();
    const bool settled = waitUntil(20.0, [&] {
        const unsigned long long now = committedSeq();
        if (now == previous)
        {
            previous = now;
            return true;
        }
        previous = now;
        return false;
    });
    if (!settled)
    {
        std::fprintf(stderr, "D12-1-C: startup publications did not settle\n");
        return false;
    }

    // A legal value, to prove the pipeline is live before the invalid input.
    e.setDitherBitDepth(16);
    if (!waitUntil(20.0, [&] { return committedSeq() > previous; }))
    {
        std::fprintf(stderr, "D12-1-C: legal dither change did not publish (seq=%llu)\n",
                     committedSeq());
        return false;
    }
    previous = committedSeq();

    // Inject an invalid dither through the session restore path.
    juce::ValueTree saved = e.getCurrentState();
    saved.setProperty("ditherBitDepth", 8, nullptr);
    e.requestLoadState(saved);
    if (e.getDitherBitDepth() != 16)
    {
        std::fprintf(stderr, "D12-1-C: invalid dither was stored (value now %d)\n",
                     e.getDitherBitDepth());
        return false;
    }

    // A normal structural parameter change must still reach a committed world.
    e.setSaturationAmount(0.5f);
    if (!waitUntil(20.0, [&] { return committedSeq() > previous; }))
    {
        std::fprintf(stderr, "D12-1-C: publish stayed closed after an invalid dither (seq=%llu)\n",
                     committedSeq());
        return false;
    }

    // The published world must carry an in-set dither depth.
    const auto* world = e.observePublishedWorld();
    if (world == nullptr)
    {
        std::fprintf(stderr, "D12-1-C: no published world\n");
        return false;
    }
    if (!isLegalDither(world->resource.ditherBitDepth))
    {
        std::fprintf(stderr, "D12-1-C: published world carries ditherBitDepth=%d\n",
                     world->resource.ditherBitDepth);
        return false;
    }

    std::printf("STG11D12StateIntegerGuardTests: D12-1-C PASS (publish survives an invalid dither)\n");
    return true;
}

// ── D12-3: the UI bit depth list is sourced from the authoritative set ──
[[nodiscard]] std::string resolveRepoRelativePath(const char* relativePath)
{
    namespace fs = std::filesystem;
    const fs::path p(relativePath);

    for (const auto& prefix : { fs::path("."), fs::path(".."), fs::path("../.."), fs::path("../../..") })
    {
        const fs::path candidate = prefix / p;
        if (fs::exists(candidate))
            return candidate.string();
    }

    return {};
}

[[nodiscard]] std::string readAllText(const char* path)
{
    const std::string resolved = resolveRepoRelativePath(path);
    if (resolved.empty())
        return {};

    std::ifstream in(resolved, std::ios::in | std::ios::binary);
    if (!in)
        return {};

    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

// Strip comments so a source-text contract constrains code, not prose. The
// production comments in this area deliberately name the identifiers the repair
// removed, and a naive substring search would flag those comments instead.
[[nodiscard]] std::string stripComments(std::string text)
{
    std::string out;
    out.reserve(text.size());

    for (std::size_t i = 0; i < text.size();)
    {
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/')
        {
            while (i < text.size() && text[i] != '\n')
                ++i;
            continue;
        }
        if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < text.size() && !(text[i] == '*' && text[i + 1] == '/'))
                ++i;
            i = (i + 1 < text.size()) ? i + 2 : text.size();
            out.push_back(' ');
            continue;
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

bool checkD12T4UiBitDepthListStaysInSet()
{
    const std::string source = readAllText("src/DeviceSettings.cpp");
    if (source.empty())
    {
        std::fprintf(stderr, "D12-3: could not read src/DeviceSettings.cpp\n");
        return false;
    }

    // Isolate updateBitDepthList so an unrelated part of the file cannot satisfy
    // or break the assertions.
    const auto begin = source.find("void DeviceSettings::updateBitDepthList()");
    const auto end   = source.find("juce::File DeviceSettings::getSettingsFile()");
    if (begin == std::string::npos || end == std::string::npos || end <= begin)
    {
        std::fprintf(stderr, "D12-3: could not locate updateBitDepthList body\n");
        return false;
    }
    const std::string body = stripComments(source.substr(begin, end - begin));

    bool ok = true;

    // The device bit depth must not reach the list. That injection was the
    // out-of-set producer for devices reporting more than 32 bits.
    const char* forbidden[] = {
        "getCurrentBitDepth",
        "supportedBitDepths.add(current)",
    };
    for (const char* needle : forbidden)
    {
        if (body.find(needle) != std::string::npos)
        {
            std::fprintf(stderr, "D12-3: updateBitDepthList still contains '%s'\n", needle);
            ok = false;
        }
    }

    // The existing selection semantics must be preserved: the three literal
    // depths plus the Off entry.
    const char* required[] = {
        "supportedBitDepths.add(16)",
        "supportedBitDepths.add(24)",
        "supportedBitDepths.add(32)",
        "bitDepthComboBox.addItem(\"Off\", 999)",
    };
    for (const char* needle : required)
    {
        if (body.find(needle) == std::string::npos)
        {
            std::fprintf(stderr, "D12-3: updateBitDepthList lost '%s'\n", needle);
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12StateIntegerGuardTests: D12-3 PASS (UI list matches the authoritative set)\n");
    return ok;
}

// ── D12-4: the validator resource contract, executed against the real code ──
// RuntimeState deletes both its copy and move constructors, so the test-only
// factory hands back ownership instead of a value.
[[nodiscard]] std::unique_ptr<RuntimeState> makeMinimalValidWorld(int ditherBitDepth)
{
    auto world = RuntimeState::createForTest();
    world->assertMutable();

    world->generation = 1;
    world->topology.runtimeUuid = 100;
    world->generationSemantic.activationEpoch = 100;
    world->generationSemantic.runtimeGeneration = 1;
    world->publication.sequenceId = 1;
    world->execution.transitionActive = false;
    world->execution.transitionPolicy = 0;
    world->routing.processingOrder = 0;
    world->execution.crossfadeStartDelayBlocks = 0;
    world->execution.crossfadeDryHoldSamples = 0;
    world->overlap.fadeTimeSec = 0.0;
    world->overlap.useDryAsOld = false;
    world->resource.oversamplingFactor = 1;
    world->resource.noiseShaperType = 0;
    world->resource.ditherBitDepth = ditherBitDepth;
    return world;
}

bool checkD12T5ValidatorResourceContract()
{
    iso::audio_engine::RuntimePublicationValidator validator;

    bool ok = true;

    // Accepts the authoritative set. The baseline case also proves the rest of
    // the world satisfies every other validator stage, so a rejection below can
    // only come from ditherBitDepth.
    for (const int legal : kLegalDither)
    {
        auto world = makeMinimalValidWorld(legal);
        const auto result = validator.validatePublication(*world);
        if (!result.isValid)
        {
            std::fprintf(stderr, "D12-4: validator rejected legal dither %d (%s)\n",
                         legal, result.errorMessage.c_str());
            ok = false;
        }
    }

    // Rejects everything outside it, and attributes the rejection to resources.
    for (const int illegal : { 8, 64, 99 })
    {
        auto world = makeMinimalValidWorld(illegal);
        const auto result = validator.validatePublication(*world);
        if (result.isValid)
        {
            std::fprintf(stderr, "D12-4: validator accepted illegal dither %d\n", illegal);
            ok = false;
            continue;
        }
        if (result.failureReason != iso::audio_engine::ValidationFailureReason::InvalidResources)
        {
            std::fprintf(stderr, "D12-4: dither %d rejected with reason %d (expected InvalidResources=%d)\n",
                         illegal, static_cast<int>(result.failureReason),
                         static_cast<int>(iso::audio_engine::ValidationFailureReason::InvalidResources));
            ok = false;
        }
    }

    if (ok)
        std::printf("STG11D12StateIntegerGuardTests: D12-4 PASS (validator dither set is executable)\n");
    return ok;
}

} // namespace

int runSTG11D12StateIntegerGuardTests()
{
    bool ok = true;

    if (!checkD12T1InvalidDitherRejected())
    {
        std::fprintf(stderr, "FAIL: D12-1-A invalid dither rejected\n");
        ok = false;
    }
    if (!checkD12T2ValidRoundTrip())
    {
        std::fprintf(stderr, "FAIL: D12-1-B legal round-trip\n");
        ok = false;
    }
    if (!checkD12T3PublishRecoversAfterInvalid())
    {
        std::fprintf(stderr, "FAIL: D12-1-C publish survives invalid dither\n");
        ok = false;
    }
    if (!checkD12T4UiBitDepthListStaysInSet())
    {
        std::fprintf(stderr, "FAIL: D12-3 UI bit depth list stays in set\n");
        ok = false;
    }
    if (!checkD12T5ValidatorResourceContract())
    {
        std::fprintf(stderr, "FAIL: D12-4 validator resource contract\n");
        ok = false;
    }

    if (ok)
        std::printf("STG11D12StateIntegerGuardTests: PASS (D12-1-A/D12-1-B/D12-1-C/D12-3/D12-4)\n");
    return ok ? 0 : 1;
}
