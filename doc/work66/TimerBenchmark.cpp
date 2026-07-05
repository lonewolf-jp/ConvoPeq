// TimerBenchmark.cpp — QPC vs RDTSC vs std::chrono 比較
// ビルド: cl /O2 /EHsc /Fe:timer_benchmark.exe TimerBenchmark.cpp
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <intrin.h>

inline uint64_t getQpc() noexcept {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

inline uint64_t getRdtsc() noexcept {
    return __rdtsc();
}

inline uint64_t getChrono() noexcept {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

inline uint64_t getChronoUs() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count());
}

double calibrateTicksPerUs() {
    // QPC基準でticksPerUsを計算
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    const double qpcFreq = static_cast<double>(freq.QuadPart);

    const uint64_t qpcStart = getQpc();
    const uint64_t tscStart = getRdtsc();
    // busy wait ~1ms
    uint64_t now;
    do { now = getQpc(); } while (now - qpcStart < freq.QuadPart / 1000);
    const uint64_t tscEnd = getRdtsc();

    const double usElapsed = 1000.0;
    const double ticksElapsed = static_cast<double>(tscEnd - tscStart);
    return ticksElapsed / usElapsed;
}

template<typename Fn>
double benchmark(const char* name, Fn fn, int iterations) {
    // warmup
    volatile uint64_t sink = 0;
    for (int i = 0; i < 1000; i++) sink += fn();

    uint64_t qpcStart = getQpc();
    for (int i = 0; i < iterations; i++) sink += fn();
    uint64_t qpcEnd = getQpc();

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double nsPerCall = (static_cast<double>(qpcEnd - qpcStart) / freq.QuadPart * 1e9) / iterations;
    printf("%-30s %6d回 計%8.3f ms  1回あたり %6.1f ns\n",
           name, iterations, nsPerCall * iterations / 1e6, nsPerCall);
    return nsPerCall;
}

int main() {
    printf("=== Timer Benchmark ===\n\n");

    const int kIter = 1000000;

    double ticksPerUs = calibrateTicksPerUs();
    printf("TSC周波数: %.2f MHz (%.0f ticks/us)\n\n", ticksPerUs, ticksPerUs);

    volatile uint64_t sink = 0;

    benchmark("QueryPerformanceCounter", getQpc, kIter);
    benchmark("__rdtsc", getRdtsc, kIter);
    benchmark("std::chrono::steady_clock::now()", getChrono, kIter);
    benchmark("std::chrono::steady_clock::now() -> us", getChronoUs, kIter);

    printf("\n=== 実効コスト順位 ===\n");
    printf("QPC       = ベースライン\n");
    printf("RDTSC     = QPC比 推定1/5〜1/20\n");
    printf("chrono    = QPC比 推定1〜5倍\n");
    printf("chrono->us= chrono + 除算1回分\n");

    printf("\nsink=%llu (最適化防止)\n", (unsigned long long)sink);
    return 0;
}
