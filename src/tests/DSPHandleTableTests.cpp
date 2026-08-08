//==============================================================================
// DSPHandleTableTests.cpp — work88 (FUTURE-6 監査) HandleTable 単体テスト
//
// テスト対象: DSPHandleTable (src/audioengine/DSPHandleTable.h) — open addressing + tombstone
//
// ■ 測定項目（監査指摘: クラスタ断絶リグレッション防止）:
//   1. insert → find の基本（単一/複数エントリ）
//   2. erase 後に同じバケットの後続エントリが find できる（クラスタ断絶なし = tombstone 有効性）
//   3. erase → insert（tombstone 再利用）後に全エントリ find 可能
//   4. findAndEraseByHandle / eraseByHandle が key を返しつつ削除できる
//   5. 重複登録なし（find で既存 key を検出し、同一 key の二重 insert で size 不変）
//
// ■ ビルド:
//   CMakeLists.txt に以下を追加（既存テスト群と同じパターン）:
//     add_executable(DSPHandleTableTests
//         src/tests/DSPHandleTableTests.cpp
//     )
//     target_compile_features(DSPHandleTableTests PRIVATE cxx_std_20)
//     target_compile_options(DSPHandleTableTests PRIVATE /EHsc /utf-8)
//     target_include_directories(DSPHandleTableTests PRIVATE
//         ${CMAKE_CURRENT_SOURCE_DIR}
//         ${CMAKE_CURRENT_SOURCE_DIR}/src
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
//         ${CMAKE_CURRENT_SOURCE_DIR}/src/core
//     )
//     add_test(NAME DSPHandleTableTests COMMAND DSPHandleTableTests)
//
//==============================================================================

#include "audioengine/DSPHandleTable.h"  // テスト対象本体（ヘッダオンリー）

#include <cstdint>
#include <iostream>
#include <mutex>
#include <vector>

namespace {

int g_testCount = 0;
int g_failCount = 0;
std::mutex g_ioMutex;

void testPass(const char* name)
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << "  PASS: " << name << std::endl;
    ++g_testCount;
}

void testFail(const char* name, const char* detail = nullptr)
{
    std::lock_guard<std::mutex> lock(g_ioMutex);
    std::cout << "  FAIL: " << name;
    if (detail) std::cout << " -- " << detail;
    std::cout << std::endl;
    ++g_testCount;
    ++g_failCount;
}

void checkTrue(const char* name, bool condition)
{
    if (condition) testPass(name);
    else testFail(name, "condition was false");
}

// テスト用のダミーキー（実ポインタに見せかけた重複しないアドレス値）
inline void* makeKey(std::uintptr_t v) { return reinterpret_cast<void*>(v); }

// 1. 基本 insert/find
bool testBasicInsertFind()
{
    convo::isr::DSPHandleTable table;
    convo::isr::DSPHandle h1{1, 1};
    convo::isr::DSPHandle h2{2, 2};

    if (!table.insert(makeKey(0x1000), h1))
        return false;
    if (!table.insert(makeKey(0x2000), h2))
        return false;
    if (table.size() != 2)
        return false;

    convo::isr::DSPHandle out;
    if (!table.find(makeKey(0x1000), out) || out.slot != 1)
        return false;
    if (!table.find(makeKey(0x2000), out) || out.slot != 2)
        return false;
    if (table.find(makeKey(0x3000), out))
        return false;  // 存在しない key
    return true;
}

// 2. ★ クラスタ断絶リグレッション: 多数 insert → 一部 erase → 残り全部 find 可能
//   （erase を tombstone 化しないと、同じバケットの後続エントリが find 不能になる）
bool testEraseDoesNotBreakCluster()
{
    convo::isr::DSPHandleTable table;
    constexpr int kCount = 200;  // 512 容量に対し 200 挿入（負荷率 0.39 → 衝突発生）

    // 挿入（連続アドレス値 → ハッシュ衝突が発生しやすい）
    for (int i = 1; i <= kCount; ++i)
    {
        convo::isr::DSPHandle h{static_cast<std::uint32_t>(i), static_cast<std::uint64_t>(i)};
        if (!table.insert(makeKey(static_cast<std::uintptr_t>(0x1000 + i * 64)), h))
            return false;
    }
    if (table.size() != static_cast<std::uint32_t>(kCount))
        return false;

    // 半分を削除
    for (int i = 1; i <= kCount; i += 2)
    {
        if (!table.erase(makeKey(static_cast<std::uintptr_t>(0x1000 + i * 64))))
            return false;
    }
    if (table.size() != static_cast<std::uint32_t>(kCount / 2))
        return false;

    // 残り全部が find 可能（tombstone 化によりクラスタ断絶なし）
    for (int i = 2; i <= kCount; i += 2)
    {
        convo::isr::DSPHandle out;
        if (!table.find(makeKey(static_cast<std::uintptr_t>(0x1000 + i * 64)), out))
            return false;  // ★ クラスタ断絶バグがあればここで失敗
        if (out.slot != static_cast<std::uint32_t>(i))
            return false;
    }
    // 削除済みは find されない
    for (int i = 1; i <= kCount; i += 2)
    {
        convo::isr::DSPHandle out;
        if (table.find(makeKey(static_cast<std::uintptr_t>(0x1000 + i * 64)), out))
            return false;
    }
    return true;
}

// 3. erase → insert（tombstone 再利用）後に全エントリ find 可能
bool testTombstoneReuse()
{
    convo::isr::DSPHandleTable table;
    for (int i = 1; i <= 100; ++i)
    {
        convo::isr::DSPHandle h{static_cast<std::uint32_t>(i), static_cast<std::uint64_t>(i)};
        table.insert(makeKey(static_cast<std::uintptr_t>(0x5000 + i * 32)), h);
    }
    for (int i = 1; i <= 50; ++i)
        table.erase(makeKey(static_cast<std::uintptr_t>(0x5000 + i * 32)));

    // tombstone を再利用して新しい key を挿入
    for (int i = 1; i <= 50; ++i)
    {
        convo::isr::DSPHandle h{static_cast<std::uint32_t>(1000 + i), static_cast<std::uint64_t>(1000 + i)};
        if (!table.insert(makeKey(static_cast<std::uintptr_t>(0x9000 + i * 32)), h))
            return false;
    }
    if (table.size() != 100)
        return false;

    // 旧エントリ（51..100）と新エントリ（1001..1050）が全部 find 可能
    for (int i = 51; i <= 100; ++i)
    {
        convo::isr::DSPHandle out;
        if (!table.find(makeKey(static_cast<std::uintptr_t>(0x5000 + i * 32)), out))
            return false;
    }
    for (int i = 1; i <= 50; ++i)
    {
        convo::isr::DSPHandle out;
        if (!table.find(makeKey(static_cast<std::uintptr_t>(0x9000 + i * 32)), out))
            return false;
        if (out.slot != static_cast<std::uint32_t>(1000 + i))
            return false;
    }
    return true;
}

// 4. findAndEraseByHandle / eraseByHandle
bool testFindAndEraseByHandle()
{
    convo::isr::DSPHandleTable table;
    convo::isr::DSPHandle h1{7, 7};
    convo::isr::DSPHandle h2{8, 8};
    table.insert(makeKey(0xA000), h1);
    table.insert(makeKey(0xB000), h2);

    void* key = nullptr;
    if (!table.findAndEraseByHandle(h1, key))
        return false;
    if (key != makeKey(0xA000))
        return false;  // key が正しく返る
    if (table.size() != 1)
        return false;

    // 残りの h2 は find 可能（クラスタ断絶なし）
    convo::isr::DSPHandle out;
    if (!table.find(makeKey(0xB000), out))
        return false;

    if (!table.eraseByHandle(h2))
        return false;
    if (table.size() != 0)
        return false;
    if (table.find(makeKey(0xB000), out))
        return false;
    return true;
}

// 5. 同一 key の二重 insert で重複登録しない
bool testNoDuplicateOnReinsert()
{
    convo::isr::DSPHandleTable table;
    convo::isr::DSPHandle h{3, 3};
    table.insert(makeKey(0xCC00), h);
    // 同じ key を再 insert（値更新）→ size 不変
    convo::isr::DSPHandle h2{4, 4};
    table.insert(makeKey(0xCC00), h2);
    if (table.size() != 1)
        return false;
    convo::isr::DSPHandle out;
    if (!table.find(makeKey(0xCC00), out))
        return false;
    return out.slot == 4;  // 更新後の値が返る
}

} // anonymous namespace

//==============================================================================
// main
//==============================================================================
int main()
{
    std::cout << "DSPHandleTableTests" << std::endl;

    checkTrue("basic insert/find", testBasicInsertFind());
    checkTrue("erase does not break cluster (tombstone)", testEraseDoesNotBreakCluster());
    checkTrue("tombstone reuse", testTombstoneReuse());
    checkTrue("findAndEraseByHandle / eraseByHandle", testFindAndEraseByHandle());
    checkTrue("no duplicate on reinsert", testNoDuplicateOnReinsert());

    std::cout << "==========================================" << std::endl;
    std::cout << "Tests: " << g_testCount << ", Failures: " << g_failCount << std::endl;
    if (g_failCount == 0)
        std::cout << "ALL TESTS PASSED" << std::endl;
    else
        std::cout << "SOME TESTS FAILED" << std::endl;

    return (g_failCount == 0) ? 0 : 1;
}
