// Knee curve probe for #238. Compiled twice: before (old header, kneeLinear)
// and after (new header, kneeDb). Not part of the product.
//
// The sampled peaks are the SAME in both builds (they are derived from a
// fixed 1 dB reference knee), so the two tables compare the same inputs. The
// point names are dB positions, not either implementation's knee bounds: the
// old knee spans threshold-0.054374 .. threshold, the new one is centred on
// the threshold. Each build prints its own bounds.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "SimplePeakLimiter.h"

namespace
{
    constexpr double T = 0.8413951287507587;
    constexpr double kRefKneeDb = 1.0;

    double gainAt (double peak, double kneeParam)
    {
        SimplePeakLimiter limiter;
        limiter.prepare (48000.0, 100.0);
        limiter.reset();
        double l = peak;
        double r = 0.0;
#if defined(OLD_API)
        limiter.processBlock (&l, &r, 1, T, kneeParam);
#else
        limiter.processBlock (&l, nullptr, 1, T, kneeParam);
#endif
        return peak > 0.0 ? std::abs (l) / peak : 0.0;
    }
}

int main (int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "table";
    const char* path = argc > 2 ? argv[2] : "knee.csv";
#if defined(OLD_API)
    const double knee = 0.108748;
    const char* api = "old-linear";
    const double boundA = T - knee * 0.5;
    const double boundB = T;
#else
    const double knee = 1.0;
    const char* api = "new-db";
    const double boundA = T * std::pow (10.0, -knee / 40.0);
    const double boundB = T * std::pow (10.0, knee / 40.0);
#endif

    std::FILE* f = std::fopen (path, "w");
    if (f == nullptr) { std::perror ("fopen"); return 2; }

    const double refA = T * std::pow (10.0, -kRefKneeDb / 40.0);
    const double refB = T * std::pow (10.0, kRefKneeDb / 40.0);
    const double refW = refB - refA;

    if (std::strcmp (mode, "table") == 0)
    {
        std::fprintf (f, "point,peak,gain,output\n");
        struct Point { const char* name; double p; };
        const Point points[] = {
            { "m2.00_eps",  refA * (1.0 - 1.0e-4) },
            { "m2.00",      refA },
            { "m1.75",      refA + 0.25 * refW },
            { "m1.50",      refA + 0.50 * refW },
            { "m1.25",      refA + 0.75 * refW },
            { "m1.00",      refB },
            { "m1.00_eps",  refB * (1.0 + 1.0e-4) },
            { "threshold",  T },
            { "p2.90",      1.4 },
            { "p6.02",      2.0 },
        };
        for (const auto& pt : points)
        {
            const double g = gainAt (pt.p, knee);
            std::fprintf (f, "%s,%.17g,%.17g,%.17g\n", pt.name, pt.p, g, pt.p * g);
        }
    }
    else
    {
        std::fprintf (f, "peak,gain,output\n");
        const int n = 400;
        for (int i = 0; i < n; ++i)
        {
            const double db = -6.0 + 6.0 * static_cast<double> (i) / (n - 1);
            const double p = std::pow (10.0, db / 20.0);
            const double g = gainAt (p, knee);
            std::fprintf (f, "%.17g,%.17g,%.17g\n", p, g, p * g);
        }
    }

    std::fclose (f);
    std::printf ("%s: wrote %s (param=%g, own knee %.9f .. %.9f)\n",
                 api, path, knee, boundA, boundB);
    return 0;
}
