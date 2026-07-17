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
        // ★ [work74 FIX-01] 環境変数（_putenv_s / setenv）はプロセスグローバルな影響を与えるため削除。
        //   CMake では MKL_THREADING=sequential（シングルスレッドリンク）が指定されており、
        //   MKL が内部でスレッドを生成することはない。従って環境変数によるスレッド数固定は不要。
        //
        //   参考: bug-fix-plan.md FIX-01

        // シングルスレッド固定（スレッドローカル版: 他スレッド/プロセスに影響しない）
        // ★ FIX-01: mkl_set_dynamic(0) は削除。sequential MKL + local設定で十分。
        mkl_set_num_threads_local(1);

        // FTZ/DAZ（既存設定と重複可）
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        // vmlSetMode は MainApplication で既に呼ばれているのでここでは呼ばない
    });
}

} // namespace MKLRealTime
