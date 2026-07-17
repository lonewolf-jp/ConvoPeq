#pragma once

#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE, _mm_getcsr, _mm_setcsr
#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE

namespace convo::cpu {

/// RAII ラッパー: コンストラクタで FTZ/DAZ を設定し、デストラクタで復元する。
/// ThreadPool ワーカー（std::async 等）で使用すること。
/// 専用スレッドや Realtime Audio Thread では使用しない。
class ScopedMXCSR final {
    unsigned int oldCsr;
public:
    ScopedMXCSR() noexcept : oldCsr(_mm_getcsr()) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    ~ScopedMXCSR() noexcept { _mm_setcsr(oldCsr); }
    ScopedMXCSR(const ScopedMXCSR&) = delete;
    ScopedMXCSR& operator=(const ScopedMXCSR&) = delete;
};

} // namespace convo::cpu
