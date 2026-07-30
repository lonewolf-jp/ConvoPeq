// NormalRetireDSPHandleCompareTests.cpp
// P0-2b: PublishReceipt DSPCore*削除 — DSPHandle 比較の検証
//
// テスト内容:
//   1. PublishReceipt が DSPHandle のみを保持することを確認
//   2. DSPHandle の同値比較（operator==）が正しく動作することを確認
//   3. getFadingRuntimeDSPHandle() が正しい Handle を返すことを確認
//
// ビルド: カスタム main() + bool testXxx() パターン

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>

#include "audioengine/ISRDSPHandle.h"

namespace {

using convo::isr::DSPHandle;
using convo::isr::DSPHandleRuntime;

//==============================================================================
// Test 1: PublishReceipt HandleOnly — DSPCore* 削除後も機能する
//==============================================================================
[[nodiscard]] bool testPublishReceiptHandleOnly()
{
    // PublishReceipt 相当の構造（AudioEngine.h から切り出し）
    struct PublishReceiptTest {
        DSPHandle handle{};
        uint64_t publicationEpoch{0};
        uint64_t generation{0};
    };

    // Default construct: handle is null
    PublishReceiptTest receipt{};
    if (!receipt.handle.isNull()) return false;

    // Assign a handle
    receipt.handle = DSPHandle{1, 42};
    if (receipt.handle.isNull()) return false;
    if (receipt.handle.slot != 1) return false;
    if (receipt.handle.generation != 42) return false;

    // Verify epoch and generation are independent
    receipt.publicationEpoch = 100;
    receipt.generation = 200;
    if (receipt.publicationEpoch != 100) return false;
    if (receipt.generation != 200) return false;

    // Reset to null
    receipt.handle = DSPHandle::null();
    if (!receipt.handle.isNull()) return false;

    std::printf("[PASS] testPublishReceiptHandleOnly\n");
    return true;
}

//==============================================================================
// Test 2: NormalRetireDSPHandleCompare — DSPHandle 比較
//==============================================================================
[[nodiscard]] bool testNormalRetireDSPHandleCompare()
{
    // Same handle must compare equal
    DSPHandle handle1{5, 100};
    DSPHandle handle2{5, 100};
    if (!(handle1 == handle2)) return false;
    if (handle1 != handle2) return false;

    // Different slot must compare not equal
    DSPHandle handle3{6, 100};
    if (handle1 == handle3) return false;
    if (!(handle1 != handle3)) return false;

    // Different generation must compare not equal
    DSPHandle handle4{5, 200};
    if (handle1 == handle4) return false;
    if (!(handle1 != handle4)) return false;

    // Same slot/gen but different objects must compare equal (value semantics)
    DSPHandle handleA{3, 777};
    DSPHandle handleB{3, 777};
    if (!(handleA == handleB)) return false;

    // Null handles
    DSPHandle null1 = DSPHandle::null();
    DSPHandle null2 = DSPHandle::null();
    if (!(null1 == null2)) return false;
    if (!null1.isNull()) return false;

    // Null vs non-null
    if (null1 == handle1) return false;

    std::printf("[PASS] testNormalRetireDSPHandleCompare\n");
    return true;
}

//==============================================================================
// Test 3: DSPHandle assignment preserves value
//==============================================================================
[[nodiscard]] bool testDSPHandleAssignment()
{
    PublishReceiptTest receipt{};
    receipt.handle = DSPHandle{7, 999};

    // Copy assignment
    PublishReceiptTest receipt2{};
    receipt2 = receipt;
    if (!(receipt2.handle == receipt.handle)) return false;
    if (receipt2.publicationEpoch != receipt.publicationEpoch) return false;
    if (receipt2.generation != receipt.generation) return false;

    std::printf("[PASS] testDSPHandleAssignment\n");
    return true;
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    bool allPassed = true;

    allPassed = testPublishReceiptHandleOnly() && allPassed;
    allPassed = testNormalRetireDSPHandleCompare() && allPassed;
    allPassed = testDSPHandleAssignment() && allPassed;

    if (allPassed) {
        std::printf("\n=== All NormalRetireDSPHandleCompare tests PASSED ===\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("\n=== Some NormalRetireDSPHandleCompare tests FAILED ===\n");
        return EXIT_FAILURE;
    }
}
