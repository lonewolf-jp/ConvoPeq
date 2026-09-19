//==============================================================================
// IRRuntimeContractTests.cpp — WORK105 negative tests (JUCE/MKL 非依存・純粋関数)
// checkIRRuntimeContract() の判定表を検証する：
//   一致 → 許可 / rate 不一致 → RateMismatch（拒否）/ block 不一致 → BlockMismatch（拒否）
//   形状不明（旧経路）→ Unknown（許可＋呼び出し側loud log）。
// ビルドパスの統合的拒否は --buzz D 実験（AudioEngineHarness）で検証する。
//==============================================================================
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "../audioengine/IRRuntimeContract.h"

namespace {

int g_testsPassed = 0;
int g_testsFailed = 0;

void check(bool condition, const std::string& label)
{
    if (condition)
        ++g_testsPassed;
    else
        ++g_testsFailed, std::cerr << "[FAIL] " << label << "\n";
}

} // namespace

// ── 一致系（許可） ──
bool testMatch()
{
    bool ok = true;
    {
        const auto r = convo::checkIRRuntimeContract(384000.0, 2048, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::None || !r.ok() || r.refused()) ok = false;
    }
    {
        const auto r = convo::checkIRRuntimeContract(48000.0, 512, 48000.0, 512);
        if (r.violation != convo::IRRuntimeContractViolation::None || !r.ok()) ok = false;
    }
    if (ok) std::cerr << "[PASS] match → allow\n";
    else std::cerr << "[FAIL] match → allow\n";
    g_testsPassed += ok ? 1 : 0;
    g_testsFailed += ok ? 0 : 1;
    return ok;
}

// ── rate 不一致 → 拒否（WORK104 実測: 192k IR × 384k DSP） ──
bool testRateMismatchRefused()
{
    bool ok = true;
    {
        const auto r = convo::checkIRRuntimeContract(192000.0, 2048, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::RateMismatch || !r.refused()) ok = false;
    }
    {
        // 逆方向も拒否
        const auto r = convo::checkIRRuntimeContract(384000.0, 2048, 192000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::RateMismatch || !r.refused()) ok = false;
    }
    {
        // 微小差は許容（1e-9 相対）
        const auto r = convo::checkIRRuntimeContract(384000.0, 2048, 384000.0 * (1.0 + 1.0e-12), 2048);
        if (r.violation != convo::IRRuntimeContractViolation::None) ok = false;
    }
    {
        // 有意差は拒否（1%）
        const auto r = convo::checkIRRuntimeContract(48000.0, 512, 48000.0 * 1.01, 512);
        if (r.violation != convo::IRRuntimeContractViolation::RateMismatch) ok = false;
    }
    if (ok) std::cerr << "[PASS] rate mismatch → refuse\n";
    else std::cerr << "[FAIL] rate mismatch → refuse\n";
    g_testsPassed += ok ? 1 : 0;
    g_testsFailed += ok ? 0 : 1;
    return ok;
}

// ── block 不一致 → 拒否（WORK104 D実験: OS 2→4 で quantum 2048→4096） ──
bool testBlockMismatchRefused()
{
    bool ok = true;
    {
        const auto r = convo::checkIRRuntimeContract(384000.0, 2048, 384000.0, 4096);
        if (r.violation != convo::IRRuntimeContractViolation::BlockMismatch || !r.refused()) ok = false;
    }
    {
        const auto r = convo::checkIRRuntimeContract(384000.0, 1024, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::BlockMismatch || !r.refused()) ok = false;
    }
    if (ok) std::cerr << "[PASS] block mismatch → refuse\n";
    else std::cerr << "[FAIL] block mismatch → refuse\n";
    g_testsPassed += ok ? 1 : 0;
    g_testsFailed += ok ? 0 : 1;
    return ok;
}

// ── 形状不明（旧経路）→ 許可（loud log は呼び出し側責務） ──
bool testUnknownAllowed()
{
    bool ok = true;
    {
        const auto r = convo::checkIRRuntimeContract(192000.0, 0, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::UnknownSourceGeometry || !r.ok()) ok = false;
    }
    {
        const auto r = convo::checkIRRuntimeContract(0.0, 0, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::UnknownSourceGeometry || !r.ok()) ok = false;
    }
    {
        const auto r = convo::checkIRRuntimeContract(-1.0, 2048, 384000.0, 2048);
        if (r.violation != convo::IRRuntimeContractViolation::UnknownSourceGeometry || !r.ok()) ok = false;
    }
    if (ok) std::cerr << "[PASS] unknown source geometry → allow\n";
    else std::cerr << "[FAIL] unknown source geometry → allow\n";
    g_testsPassed += ok ? 1 : 0;
    g_testsFailed += ok ? 0 : 1;
    return ok;
}

// ── toString 網羅 ──
bool testToString()
{
    bool ok = true;
    ok = ok && (std::string(convo::toString(convo::IRRuntimeContractViolation::None)) == "None");
    ok = ok && (std::string(convo::toString(convo::IRRuntimeContractViolation::RateMismatch)) == "RateMismatch");
    ok = ok && (std::string(convo::toString(convo::IRRuntimeContractViolation::BlockMismatch)) == "BlockMismatch");
    ok = ok && (std::string(convo::toString(convo::IRRuntimeContractViolation::UnknownSourceGeometry)) == "UnknownSourceGeometry");
    if (ok) std::cerr << "[PASS] toString coverage\n";
    else std::cerr << "[FAIL] toString coverage\n";
    g_testsPassed += ok ? 1 : 0;
    g_testsFailed += ok ? 0 : 1;
    return ok;
}

int main()
{
    testMatch();
    testRateMismatchRefused();
    testBlockMismatchRefused();
    testUnknownAllowed();
    testToString();
    std::cerr << "IRRuntimeContractTests: passed=" << g_testsPassed
              << " failed=" << g_testsFailed << "\n";
    return g_testsFailed == 0 ? 0 : 1;
}
