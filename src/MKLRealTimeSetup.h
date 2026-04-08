#pragma once

#include <mkl.h>
#include <mutex>
#include <JuceHeader.h>

namespace MKLRealTime {

// MUST be called before any MKL DFTI usage.
// Safe to call multiple times (call_once ensures single execution).
inline void setup() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        // シングルスレッド固定
        mkl_set_num_threads(1);
        mkl_set_dynamic(0);
        
        // 環境変数でさらに強制（OpenMP 対策）
#ifdef _WIN32
        _putenv("MKL_NUM_THREADS=1");
        _putenv("OMP_NUM_THREADS=1");
#else
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("OMP_NUM_THREADS", "1", 1);
#endif
        
        // FTZ/DAZ（既存設定と重複可）
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        // vmlSetMode は MainApplication で既に呼ばれているのでここでは呼ばない
    });
}

// レイヤーごとにウォームアップ（Message Thread 限定）
// warmupFlag は Layer インスタンスごとに保持される。
// Layer が再生成される場合は warmupFlag も再初期化される。
template<typename Layer>
inline void warmupLayer(Layer& layer) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    
    std::call_once(layer.warmupFlag, [&]() {
        // [v2.1] IPP FFT warmup
        double* dummyIn = static_cast<double*>(mkl_malloc(layer.fftSize * sizeof(double), 64));
        double* dummyOut = static_cast<double*>(mkl_malloc((layer.fftSize + 2) * sizeof(double), 64));
        if (dummyIn && dummyOut) {
            ippsFFTFwd_RToCCS_64f(dummyIn, dummyOut, layer.fftSpec, layer.fftWorkBuf);
            ippsFFTInv_CCSToR_64f(dummyOut, dummyIn, layer.fftSpec, layer.fftWorkBuf);
        }
        mkl_free(dummyIn);
        mkl_free(dummyOut);
    });
}

} // namespace MKLRealTime
