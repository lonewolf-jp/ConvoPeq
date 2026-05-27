#include "MKLRealTimeSetup.h"

#include <mkl.h>
#include <immintrin.h>
#include <mutex>
#include <cstdlib>

namespace
{
    std::once_flag g_setupOnceFlag;
}

namespace MKLRealTime {

void setup() noexcept
{
    std::call_once(g_setupOnceFlag, []() noexcept {
        // 環境変数でさらに強制（OpenMP 対策）
#ifdef _WIN32
        _putenv_s("MKL_NUM_THREADS", "1");
        _putenv_s("OMP_NUM_THREADS", "1");
#else
        setenv("MKL_NUM_THREADS", "1", 1);
        setenv("OMP_NUM_THREADS", "1", 1);
#endif

        // シングルスレッド固定
        mkl_set_num_threads(1);
        mkl_set_dynamic(0);

        // FTZ/DAZ（既存設定と重複可）
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        // vmlSetMode は MainApplication で既に呼ばれているのでここでは呼ばない
    });
}

} // namespace MKLRealTime
