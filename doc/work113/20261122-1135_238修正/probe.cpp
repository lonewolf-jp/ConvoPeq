// Knee curve probe for #238. Compiled twice: before (old header, kneeLinear)
// and after (new header, kneeDb). Not part of the product.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "SimplePeakLimiter.h"

namespace
{
    constexpr double T = 0.8413951287507587;

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
#else
    const double knee = 1.0;
    const char* api = "new-db";
#endif

    std::FILE* f = std::fopen (path, "w");
    if (f == nullptr) { std::perror ("fopen"); return 2; }

    const double a = T * std::pow (10.0, -knee / 40.0);
    const double b = T * std::pow (10.0, knee / 40.0);

    if (std::strcmp (mode, "table") == 0)
    {
        std::fprintf (f, "point,peak,gain,output\n");
        struct Point { const char* name; double p; };
        const Point points[] = {
            { "knee_start_minus_eps", a * (1.0 - 1.0e-4) },
            { "knee_start",           a },
            { "knee_25",              a + 0.25 * (b - a) },
            { "knee_50",              a + 0.50 * (b - a) },
            { "knee_75",              a + 0.75 * (b - a) },
            { "knee_end",             b },
            { "knee_end_plus_eps",    b * (1.0 + 1.0e-4) },
            { "threshold",            T },
            { "strong_1_4",           1.4 },
            { "strong_2_0",           2.0 },
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
    std::printf ("%s: wrote %s (knee=%g, a=%.9f, b=%.9f)\n", api, path, knee, a, b);
    return 0;
}
