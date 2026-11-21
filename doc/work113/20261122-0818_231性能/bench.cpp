// decimate stage benchmark for Issue #231 acceptance (4-tap case, N >= 20, median/IQR)
// Builds twice: MSVC AVX2 and MSVC SSE2 (scalar path), plus icx AVX2.
// The kernels are copied from src/CustomInputOversampler.cpp (lines noted below).

#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <windows.h>

#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

// src/CustomInputOversampler.cpp:45-58 (loadStride2)
#if HAS_AVX2
static inline __m256d loadStride2 (const double* ptr) noexcept
{
    __m128d v0 = _mm_loadu_pd (ptr - 6);
    __m128d v1 = _mm_loadu_pd (ptr - 4);
    __m128d v2 = _mm_loadu_pd (ptr - 2);
    __m128d v3 = _mm_loadu_pd (ptr);
    __m128d vLow  = _mm_unpacklo_pd (v3, v2);
    __m128d vHigh = _mm_unpacklo_pd (v1, v0);
    return _mm256_insertf128_pd (_mm256_castpd128_pd256 (vLow), vHigh, 1);
}

// src/CustomInputOversampler.cpp:216-287 (dotProductDecimateAvx2, convCount >= 8)
static double dotAvx2Wide (const double* __restrict history,
                           const double* __restrict coeffs,
                           int convCount) noexcept
{
    __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd(), acc5 = _mm256_setzero_pd();
    __m256d acc6 = _mm256_setzero_pd(), acc7 = _mm256_setzero_pd();

    int r = 0;
    const int unrollEnd = (convCount / 32) * 32;
    for (; r < unrollEnd; r += 32)
    {
        acc0 = _mm256_fmadd_pd (loadStride2 (history - (r << 1)),        _mm256_loadu_pd (coeffs + r),        acc0);
        acc1 = _mm256_fmadd_pd (loadStride2 (history - ((r +  4) << 1)), _mm256_loadu_pd (coeffs + r +  4), acc1);
        acc2 = _mm256_fmadd_pd (loadStride2 (history - ((r +  8) << 1)), _mm256_loadu_pd (coeffs + r +  8), acc2);
        acc3 = _mm256_fmadd_pd (loadStride2 (history - ((r + 12) << 1)), _mm256_loadu_pd (coeffs + r + 12), acc3);
        acc4 = _mm256_fmadd_pd (loadStride2 (history - ((r + 16) << 1)), _mm256_loadu_pd (coeffs + r + 16), acc4);
        acc5 = _mm256_fmadd_pd (loadStride2 (history - ((r + 20) << 1)), _mm256_loadu_pd (coeffs + r + 20), acc5);
        acc6 = _mm256_fmadd_pd (loadStride2 (history - ((r + 24) << 1)), _mm256_loadu_pd (coeffs + r + 24), acc6);
        acc7 = _mm256_fmadd_pd (loadStride2 (history - ((r + 28) << 1)), _mm256_loadu_pd (coeffs + r + 28), acc7);
    }

    const int simdEnd = (convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2 (history - (r << 1));
        __m256d vC = _mm256_loadu_pd (coeffs + r);
        acc0 = _mm256_fmadd_pd (vS, vC, acc0);
    }

    acc0 = _mm256_add_pd (acc0, acc1);
    acc2 = _mm256_add_pd (acc2, acc3);
    acc4 = _mm256_add_pd (acc4, acc5);
    acc6 = _mm256_add_pd (acc6, acc7);
    acc0 = _mm256_add_pd (acc0, acc2);
    acc4 = _mm256_add_pd (acc4, acc6);
    acc0 = _mm256_add_pd (acc0, acc4);

    __m128d vLo = _mm256_castpd256_pd128 (acc0);
    __m128d vHi = _mm256_extractf128_pd (acc0, 1);
    __m128d vSum = _mm_add_pd (vLo, vHi);
    vSum = _mm_hadd_pd (vSum, vSum);
    double result = _mm_cvtsd_f64 (vSum);

    for (; r < convCount; ++r)
        result += coeffs[r] * history[-(r << 1)];

    return result;
}

// src/CustomInputOversampler.cpp:674-696 (convCount 4-7 simple AVX2 branch)
static double dotAvx2Narrow (const double* __restrict history,
                             const double* __restrict coeffs,
                             int convCount) noexcept
{
    __m256d vAcc = _mm256_setzero_pd();
    int r = 0;
    const int simdEnd = (convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2 (history - (r << 1));
        __m256d vC = _mm256_loadu_pd (coeffs + r);
        vAcc = _mm256_fmadd_pd (vS, vC, vAcc);
    }
    __m128d vLo = _mm256_castpd256_pd128 (vAcc);
    __m128d vHi = _mm256_extractf128_pd (vAcc, 1);
    __m128d vSum = _mm_add_pd (vLo, vHi);
    vSum = _mm_hadd_pd (vSum, vSum);
    double acc = _mm_cvtsd_f64 (vSum);
    for (; r < convCount; ++r)
        acc += coeffs[r] * history[-(r << 1)];
    return acc;
}
#endif

static double dotScalar (const double* __restrict history,
                         const double* __restrict coeffs,
                         int convCount) noexcept
{
    double acc = 0.0;
    for (int r = 0; r < convCount; ++r)
        acc += coeffs[r] * history[-(r << 1)];
    return acc;
}

static double dotAny (const double* history, const double* coeffs, int convCount)
{
#if HAS_AVX2
    return convCount >= 8 ? dotAvx2Wide (history, coeffs, convCount)
                          : (convCount >= 4 ? dotAvx2Narrow (history, coeffs, convCount)
                                            : dotScalar (history, coeffs, convCount));
#else
    return dotScalar (history, coeffs, convCount);
#endif
}

static double nowMs()
{
    static LARGE_INTEGER freq { [] { LARGE_INTEGER f; QueryPerformanceFrequency (&f); return f; } () };
    LARGE_INTEGER c; QueryPerformanceCounter (&c);
    return static_cast<double> (c.QuadPart) * 1000.0 / static_cast<double> (freq.QuadPart);
}

int main (int argc, char** argv)
{
    const int outputs = argc > 1 ? std::atoi (argv[1]) : 1000000;
    const int reps    = argc > 2 ? std::atoi (argv[2]) : 25;
    const char* outPath = argc > 3 ? argv[3] : "bench.csv";
    const char* variant = argc > 4 ? argv[4] : "unknown";

    const int convCounts[] = { 4, 16, 32, 64 };
    std::FILE* f = std::fopen (outPath, "w");
    if (f == nullptr) { std::perror ("open"); return 2; }
    std::fprintf (f, "variant,convCount,rep,ms,nsPerOutput\n");

    volatile double sink = 0.0;

    for (int cc : convCounts)
    {
        const int historySize = 2 * outputs + 2 * cc + 256;
        std::vector<double> history (static_cast<size_t> (historySize));
        std::vector<double> coeffs (static_cast<size_t> (cc));
        for (int i = 0; i < historySize; ++i)
            history[static_cast<size_t> (i)] = std::sin (0.001 * i) * 0.5;
        for (int r = 0; r < cc; ++r)
            coeffs[static_cast<size_t> (r)] = 0.5 / (1.0 + r);

        // first element used by the loop must leave room for ptr[-6] and the stride
        const int start = 6 + 2 * cc + 8;

        // warm up
        dotAny (&history[static_cast<size_t> (start)], coeffs.data(), cc);

        std::vector<double> ms (static_cast<size_t> (reps));
        for (int rep = 0; rep < reps; ++rep)
        {
            const double t0 = nowMs();
            double acc = 0.0;
            int idx = start;
            for (int n = 0; n < outputs; ++n, idx += 2)
                acc += dotAny (&history[static_cast<size_t> (idx)], coeffs.data(), cc);
            const double t1 = nowMs();
            sink = sink + acc;
            ms[static_cast<size_t> (rep)] = t1 - t0;
        }

        for (int rep = 0; rep < reps; ++rep)
            std::fprintf (f, "%s,%d,%d,%.6f,%.6f\n", variant, cc, rep + 1,
                          ms[static_cast<size_t> (rep)],
                          ms[static_cast<size_t> (rep)] * 1.0e6 / static_cast<double> (outputs));
    }

    std::fclose (f);
    std::printf ("path=%s sink=%f\n", HAS_AVX2 ? "avx2" : "scalar", static_cast<double> (sink));
    return 0;
}
