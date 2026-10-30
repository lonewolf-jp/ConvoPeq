//==============================================================================
// 2026-10-30 (別変更単位 / 診断): 生存 AudioEngine インスタンス数の診断カウンタの検証
//
//   R-51 の修正で複数エンジン共存時の破棄保護は改善されたが、「製品が複数インスタンスを
//   許容するか」は未定義である。本カウンタの目的は**存在した事実の観測**であり、
//   生成の禁止・例外化・assert は行わない（Owner 判断 2026-10-30）。
//
//   ここでは次を確かめる。
//     (1) 1 個の生成・破棄でカウンタが ±1 して戻る
//     (2) 2 個同時に生存している間は +2 になる
//     (3) 生成・破棄を繰り返してもカウンタがずれない（対称性）
//
//   AudioEngineHarness は「AudioEngine をヒープに持つ RAII」なので、start() を呼ばずに
//   生成・破棄だけを行う（スレッドを起動しないため高速かつ副作用が小さい）。
//==============================================================================

#include "AudioEngineHarness.h"

#include <cstdio>

namespace {

bool expect(const char* name, int got, int want)
{
    if (got != want)
    {
        std::fprintf(stderr, "FAIL: ENGINE-COUNT %s (got=%d want=%d)\n", name, got, want);
        return false;
    }
    return true;
}

} // namespace

int runEngineInstanceCountTests()
{
    bool ok = true;

    const int base = AudioEngine::engineInstanceCountForTest();
    std::printf("[ENGINE-COUNT] baseline=%d\n", base);
    ok = expect("baseline is non-negative", base >= 0 ? 1 : 0, 1) && ok;

    // (1) 1 個の生成・破棄
    {
        AudioEngineHarness one;
        ok = expect("single alive", AudioEngine::engineInstanceCountForTest(), base + 1) && ok;
    }
    ok = expect("single released", AudioEngine::engineInstanceCountForTest(), base) && ok;

    // (2) 2 個同時
    {
        AudioEngineHarness a;
        AudioEngineHarness b;
        ok = expect("two alive", AudioEngine::engineInstanceCountForTest(), base + 2) && ok;
        std::printf("[ENGINE-COUNT] two instances alive = %d\n",
                    AudioEngine::engineInstanceCountForTest());
    }
    ok = expect("two released", AudioEngine::engineInstanceCountForTest(), base) && ok;

    // (3) 生成・破棄の反復（対称性）
    constexpr int kCycles = 3;
    for (int i = 0; i < kCycles; ++i)
    {
        AudioEngineHarness h;
        if (AudioEngine::engineInstanceCountForTest() != base + 1)
        {
            std::fprintf(stderr, "FAIL: ENGINE-COUNT cycle %d alive mismatch (got=%d want=%d)\n",
                         i, AudioEngine::engineInstanceCountForTest(), base + 1);
            ok = false;
        }
    }
    ok = expect("no drift after cycles", AudioEngine::engineInstanceCountForTest(), base) && ok;

    if (ok)
        std::printf("EngineInstanceCountTests: PASS (symmetric across %d cycles)\n", kCycles);
    return ok ? 0 : 1;
}
