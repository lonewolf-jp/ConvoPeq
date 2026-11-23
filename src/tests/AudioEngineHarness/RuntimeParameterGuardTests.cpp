// RuntimeParameterGuardTests.cpp — B-03（mixed f1/f2 の非有限値拒否）
//
// 破損セッションの NaN/Inf が pendingOverride へ入ると、mixed 位相の遷移周波数が
// 非有限になり、再構築後の IR が非有限になる。setter と setState 経路の両方で
// 「非有限は格納せず以前値を維持する」ことを確認する。

#include <JuceHeader.h>

#include "ConvolverProcessor.h"
#include "AudioEngineHarness.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

int gB03Failures = 0;

void expect (bool ok, const char* id, const juce::String& detail = {})
{
    std::printf ("[%s] %s%s\n",
                 ok ? "PASS" : "FAIL",
                 id,
                 detail.isNotEmpty() ? (" (" + detail + ")").toRawUTF8() : "");

    if (! ok)
        ++gB03Failures;
}

constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
constexpr float kInff = std::numeric_limits<float>::infinity();

} // namespace

int runRuntimeParameterGuardTests()
{
    gB03Failures = 0;

    {
        ConvolverProcessor conv;

        conv.setMixedTransitionStartHz (300.0f);
        conv.setMixedTransitionEndHz (800.0f);

        const float f1 = conv.getMixedTransitionStartHz();
        const float f2 = conv.getMixedTransitionEndHz();

        expect (std::isfinite (f1) && std::isfinite (f2), "B03-1 baseline is finite",
                "f1=" + juce::String (f1, 1) + " f2=" + juce::String (f2, 1));

        // B03-2: NaN は格納しない（両 setter）
        conv.setMixedTransitionStartHz (kNaNf);
        conv.setMixedTransitionEndHz (kNaNf);

        expect (conv.getMixedTransitionStartHz() == f1, "B03-2a NaN start is rejected",
                "got=" + juce::String (conv.getMixedTransitionStartHz(), 1));
        expect (conv.getMixedTransitionEndHz() == f2, "B03-2b NaN end is rejected",
                "got=" + juce::String (conv.getMixedTransitionEndHz(), 1));

        // B03-3: ±Inf も格納しない
        conv.setMixedTransitionStartHz (kInff);
        conv.setMixedTransitionEndHz (-kInff);

        expect (conv.getMixedTransitionStartHz() == f1, "B03-3a +Inf start is rejected");
        expect (conv.getMixedTransitionEndHz() == f2, "B03-3b -Inf end is rejected");

        // B03-4: 有効値は従来どおり通る（F1 100–400、F2 700–1300 の内側）
        conv.setMixedTransitionStartHz (250.0f);
        conv.setMixedTransitionEndHz (1200.0f);

        expect (std::abs (conv.getMixedTransitionStartHz() - 250.0f) < 1.0e-3f,
                "B03-4a valid start is accepted",
                "got=" + juce::String (conv.getMixedTransitionStartHz(), 1));
        expect (std::abs (conv.getMixedTransitionEndHz() - 1200.0f) < 1.0e-3f,
                "B03-4b valid end is accepted",
                "got=" + juce::String (conv.getMixedTransitionEndHz(), 1));

        // B03-5: setState（セッション復元の経路）に NaN を流しても壊れない
        const float beforeF1 = conv.getMixedTransitionStartHz();
        const float beforeF2 = conv.getMixedTransitionEndHz();

        juce::ValueTree saved ("Convolver");
        saved.setProperty ("mixedF1Hz", static_cast<double> (kNaNf), nullptr);
        saved.setProperty ("mixedF2Hz", static_cast<double> (kInff), nullptr);

        conv.setState (saved);

        expect (conv.getMixedTransitionStartHz() == beforeF1
                    && conv.getMixedTransitionEndHz() == beforeF2,
                "B03-5 setState with non-finite values keeps previous",
                "f1=" + juce::String (conv.getMixedTransitionStartHz(), 1)
                    + " f2=" + juce::String (conv.getMixedTransitionEndHz(), 1));
    }

    // ---- B-09: 適応係数の非有限値ゲート ----
    {
        AudioEngineHarness h;

        if (! h.start (48000.0, 512))
        {
            expect (false, "B09-0 harness start");
        }
        else
        {
            AudioEngine& e = h.engine();

            constexpr int kOrder = 9; // kAdaptiveNoiseShaperOrder
            const double seed[kOrder] = { 0.10, -0.20, 0.30, -0.40, 0.50,
                                          -0.60, 0.70, -0.80, 0.15 };

            e.setAdaptiveCoefficientsForSampleRate (48000.0, seed, kOrder);

            double before[kOrder] = {};
            e.getAdaptiveCoefficientsForSampleRate (48000.0, before, kOrder);

            expect (std::isfinite (before[3]), "B09-1 baseline coefficients are finite",
                    "c3=" + juce::String (before[3], 4));

            double bad[kOrder] = {};

            for (int i = 0; i < kOrder; ++i)
                bad[i] = before[i];

            bad[3] = std::numeric_limits<double>::quiet_NaN();
            bad[5] = std::numeric_limits<double>::infinity();

            e.setAdaptiveCoefficientsForSampleRate (48000.0, bad, kOrder);

            double after[kOrder] = {};
            e.getAdaptiveCoefficientsForSampleRate (48000.0, after, kOrder);

            bool unchanged = true;

            for (int i = 0; i < kOrder; ++i)
                if (after[i] != before[i])
                    unchanged = false;

            expect (unchanged, "B09-2 non-finite coefficients are rejected",
                    "c3=" + juce::String (after[3], 4) + " c5=" + juce::String (after[5], 4));

            h.stop();
        }
    }
    if (gB03Failures == 0)
        std::printf ("RuntimeParameterGuardTests: PASS (B-03 setters, B-09 coefficient gate)\n");

    return gB03Failures == 0 ? 0 : 1;
}
