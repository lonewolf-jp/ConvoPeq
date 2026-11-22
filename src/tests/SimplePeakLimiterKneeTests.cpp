//============================================================================
// SimplePeakLimiterKneeTests.cpp (#238)
//
// Pins the soft-knee gain curve of SimplePeakLimiter.
//
// The expected values in kGainCases are constants computed OUTSIDE this
// implementation: the curve was derived from the four boundary conditions
//   g(a)=1, g'(a)=0, g(b)=T/b, g'(b)=-T/b^2   (a = T*10^(-K/40), b = T*10^(K/40))
// by solving the cubic c0 + c1*p + c2*p^2 + c3*p^3 in the amplitude domain,
// then written here as literals. The implementation uses a Hermite basis and
// is never evaluated to build the expectations.
//
// The stateful envelope is kept out of the static-curve checks: every case
// uses a fresh limiter and a one-sample block, so the applied gain equals the
// desired gain (attack is immediate).
//============================================================================
#include <cmath>
#include <cstdio>
#include <limits>

#include "SimplePeakLimiter.h"

namespace
{
    constexpr double kT = 0.8413951287507587;          // -1.5 dBFS
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    double appliedGain (double peak, double kneeDb, double threshold = kT)
    {
        SimplePeakLimiter limiter;
        limiter.prepare (48000.0, 100.0);
        limiter.reset();

        double l = peak;
        limiter.processBlock (&l, nullptr, 1, threshold, kneeDb);
        return peak > 0.0 ? std::abs (l) / peak : 0.0;
    }

    bool near (double actual, double expected, double tol = 5.0e-13)
    {
        return std::abs (actual - expected) <= tol;
    }

    struct GainCase
    {
        const char* name;
        double peak;
        double kneeDb;
        double expected;
    };

    // 1 dB knee, centred on -1.5 dBFS: a = -2.0 dBFS, b = -1.0 dBFS.
    // 0.1 dB knee: a = 0.8365655829353869, b = 0.8462525558383925.
    const GainCase kGainCases[] = {
        { "1dB: knee start",            0.7943282225511487, 1.0, 1.0 },
        { "1dB: knee 25%",              0.8185588980321780, 1.0, 0.9960719682309618 },
        { "1dB: knee 50%",              0.8427895735132072, 1.0, 0.9848636549730552 },
        { "1dB: knee 75%",              0.8670202489942365, 1.0, 0.9672387333001025 },
        { "1dB: knee end",              0.8912509244752658, 1.0, 0.9440608762859234 },
        { "1dB: threshold",             0.8413951287507587, 1.0, 0.9856909440974633 },
        { "1dB: strong 1.4",            1.4,                1.0, 0.6009965205362563 },
        { "1dB: strong 2.0",            2.0,                1.0, 0.4206975643753793 },
        { "0.1dB: knee start",          0.8365655829353869, 0.1, 1.0 },
        { "0.1dB: knee middle",         0.8414090693868896, 0.1, 0.9985526870956250 },
        { "0.1dB: knee end",            0.8462525558383925, 0.1, 0.9942600739529567 },
    };

    bool checkStaticCurve()
    {
        bool ok = true;
        for (const auto& c : kGainCases)
        {
            const double g = appliedGain (c.peak, c.kneeDb);
            if (!near (g, c.expected))
            {
                std::printf ("FAIL %s: peak=%.17g gain=%.17g expected=%.17g diff=%.3e\n",
                             c.name, c.peak, g, c.expected, g - c.expected);
                ok = false;
            }
        }
        if (ok)
            std::printf ("[PASS] static soft-knee curve (%zu cases)\n",
                         sizeof (kGainCases) / sizeof (kGainCases[0]));
        return ok;
    }

    bool checkHardKneeWhenDisabled()
    {
        bool ok = true;
        const double knees[] = { 0.0, -1.0, kNaN };
        for (double knee : knees)
        {
            const double below = appliedGain (0.80, knee);
            const double atT = appliedGain (kT, knee);
            const double strong = appliedGain (1.4, knee);
            if (!near (below, 1.0) || !near (atT, 1.0) || !near (strong, kT / 1.4))
            {
                std::printf ("FAIL hard knee (knee=%g): below=%.17g atT=%.17g strong=%.17g\n",
                             knee, below, atT, strong);
                ok = false;
            }
        }
        if (ok)
            std::printf ("[PASS] knee <= 0 or NaN falls back to the hard knee\n");
        return ok;
    }

    bool checkBoundaryContinuity()
    {
        const double a = 0.7943282225511487;   // 1 dB knee start
        const double b = 0.8912509244752658;   // 1 dB knee end
        const double insideA = appliedGain (std::nextafter (a, b), 1.0);
        const double outsideA = appliedGain (std::nextafter (a, 0.0), 1.0);
        const double insideB = appliedGain (std::nextafter (b, a), 1.0);
        const double outsideB = appliedGain (std::nextafter (b, 2.0), 1.0);
        const double hardAtB = kT / b;

        bool ok = true;
        if (!near (insideA, 1.0, 1.0e-12) || !near (outsideA, 1.0, 1.0e-12))
        {
            std::printf ("FAIL knee start continuity: inside=%.17g outside=%.17g\n", insideA, outsideA);
            ok = false;
        }
        if (!near (insideB, hardAtB, 1.0e-12) || !near (outsideB, hardAtB, 1.0e-12))
        {
            std::printf ("FAIL knee end continuity: inside=%.17g outside=%.17g hard=%.17g\n",
                         insideB, outsideB, hardAtB);
            ok = false;
        }
        if (ok)
            std::printf ("[PASS] knee start/end are continuous (value)\n");
        return ok;
    }

    bool checkMonotonicAndBounded()
    {
        bool ok = true;
        const int n = 2000;
        double prevGain = 2.0;
        double prevOut = -1.0;
        for (int i = 0; i < n; ++i)
        {
            const double db = -12.0 + 20.0 * static_cast<double> (i) / (n - 1);
            const double p = std::pow (10.0, db / 20.0);
            const double g = appliedGain (p, 1.0);
            const double out = p * g;

            if (!std::isfinite (g) || g > 1.0 + 1.0e-12 || g < 0.0)
            {
                std::printf ("FAIL bounded: p=%.17g gain=%.17g\n", p, g);
                ok = false;
                break;
            }
            if (g > prevGain + 1.0e-12)
            {
                std::printf ("FAIL gain not monotone: p=%.17g gain=%.17g prev=%.17g\n", p, g, prevGain);
                ok = false;
                break;
            }
            if (out < prevOut - 1.0e-12)
            {
                std::printf ("FAIL output not monotone: p=%.17g out=%.17g prev=%.17g\n", p, out, prevOut);
                ok = false;
                break;
            }
            prevGain = g;
            prevOut = out;
        }
        if (ok)
            std::printf ("[PASS] curve is bounded by 1 and monotone (%d points)\n", n);
        return ok;
    }

    bool checkChannels()
    {
        SimplePeakLimiter limiter;
        limiter.prepare (48000.0, 100.0);
        limiter.reset();

        double l = 0.5;
        double r = 0.95;
        limiter.processBlock (&l, &r, 1, kT, 1.0);

        // 0.95 is above the 1 dB knee end (0.89125), so the hard ratio applies.
        const double expected = kT / 0.95;
        const double gainFromR = r / 0.95;
        const double gainFromL = l / 0.5;
        if (!near (gainFromL, expected) || !near (gainFromR, expected, 1.0e-15))
        {
            std::printf ("FAIL stereo: gainL=%.17g gainR=%.17g expected=%.17g\n",
                         gainFromL, gainFromR, expected);
            return false;
        }
        std::printf ("[PASS] stereo uses the louder channel and scales both (%.17g)\n", gainFromR);
        return true;
    }

    bool checkMonoNullRight()
    {
        SimplePeakLimiter limiter;
        limiter.prepare (48000.0, 100.0);
        limiter.reset();

        double l = 1.4;
        limiter.processBlock (&l, nullptr, 1, kT, 1.0);
        const double gain = l / 1.4;
        if (!near (gain, kT / 1.4))
        {
            std::printf ("FAIL mono (right=nullptr): gain=%.17g expected=%.17g\n", gain, kT / 1.4);
            return false;
        }
        std::printf ("[PASS] right=nullptr uses the left peak\n");
        return true;
    }

    bool checkInvalidThreshold()
    {
        bool ok = true;
        const double thresholds[] = { 0.0, -0.5, kNaN, kInf };
        for (double t : thresholds)
        {
            const double g = appliedGain (1.4, 1.0, t);
            if (!std::isfinite (g) || !near (g, 1.0))
            {
                std::printf ("FAIL invalid threshold %g: gain=%.17g\n", t, g);
                ok = false;
            }
        }
        if (ok)
            std::printf ("[PASS] invalid threshold passes through without NaN\n");
        return ok;
    }

    bool checkExtremeInputsAndKnee()
    {
        bool ok = true;
        const double extreme[] = { 1.0e-12, 1.0e-6, 1.0e6, 1.0e12 };
        for (double p : extreme)
        {
            const double g = appliedGain (p, 1.0);
            if (!std::isfinite (g) || g > 1.0 + 1.0e-12 || g < 0.0)
            {
                std::printf ("FAIL extreme input %.17g: gain=%.17g\n", p, g);
                ok = false;
            }
        }
        // A huge knee width must not produce non-finite bounds; it falls back
        // to the hard knee.
        const double hard = appliedGain (1.4, 1.0e9);
        if (!near (hard, kT / 1.4))
        {
            std::printf ("FAIL huge knee: gain=%.17g expected=%.17g\n", hard, kT / 1.4);
            ok = false;
        }
        if (ok)
            std::printf ("[PASS] extreme inputs and huge knee stay finite\n");
        return ok;
    }
    bool checkOutputCeiling()
    {
        // The limiter ceiling is T (-1.5 dBFS), below the hard clamp at
        // kOutputHeadroom (-1.0 dBFS), so the clamp stays a safety net.
        bool ok = true;
        const double b = 0.8912509244752658;
        for (int i = 0; i <= 1000; ++i)
        {
            const double p = 0.5 + 3.5 * static_cast<double> (i) / 1000.0;
            const double out = p * appliedGain (p, 1.0);
            if (out > kT + 1.0e-12)
            {
                std::printf ("FAIL ceiling: p=%.17g out=%.17g\n", p, out);
                ok = false;
                break;
            }
            if (p >= b && !near (out, kT, 1.0e-12))
            {
                std::printf ("FAIL ceiling above knee end: p=%.17g out=%.17g\n", p, out);
                ok = false;
                break;
            }
        }
        if (ok)
            std::printf ("[PASS] output ceiling is T (-1.5 dBFS) and the hard clamp is not reached\n");
        return ok;
    }

    bool checkSlopeContinuity()
    {
        const double a = 0.7943282225511487;
        const double b = 0.8912509244752658;
        const double h = 1.0e-7;
        const double rightAtA = (appliedGain (a + h, 1.0) - appliedGain (a, 1.0)) / h;
        const double leftAtB = (appliedGain (b, 1.0) - appliedGain (b - h, 1.0)) / h;
        const double rightAtB = (appliedGain (b + h, 1.0) - appliedGain (b, 1.0)) / h;
        const double hardSlope = -kT / (b * b);

        bool ok = true;
        if (std::abs (rightAtA) > 1.0e-6)
        {
            std::printf ("FAIL slope at knee start: %.3e (expected 0)\n", rightAtA);
            ok = false;
        }
        if (std::abs (leftAtB - hardSlope) > 1.0e-4)
        {
            std::printf ("FAIL slope at knee end inside: %.6f expected %.6f\n", leftAtB, hardSlope);
            ok = false;
        }
        if (std::abs (rightAtB - leftAtB) > 1.0e-4)
        {
            std::printf ("FAIL slope jump at knee end: inside %.6f outside %.6f\n", leftAtB, rightAtB);
            ok = false;
        }
        if (ok)
            std::printf ("[PASS] slope is continuous at both knee boundaries (%.6f)\n", hardSlope);
        return ok;
    }
} // namespace

int main()
{
    bool allPassed = true;
    allPassed &= checkStaticCurve();
    allPassed &= checkHardKneeWhenDisabled();
    allPassed &= checkBoundaryContinuity();
    allPassed &= checkSlopeContinuity();
    allPassed &= checkMonotonicAndBounded();
    allPassed &= checkOutputCeiling();
    allPassed &= checkChannels();
    allPassed &= checkMonoNullRight();
    allPassed &= checkInvalidThreshold();
    allPassed &= checkExtremeInputsAndKnee();

    if (allPassed)
    {
        std::printf ("SimplePeakLimiterKneeTests: PASS\n");
        return 0;
    }
    std::printf ("SimplePeakLimiterKneeTests: FAIL\n");
    return 1;
}
