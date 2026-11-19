# ============================================================================
# ★ WORK113 (2026-11-18 Owner 承認): shutdown 基盤の ThreadSanitizer ハーネス。
#
#   JUCE / MKL / IPP に依存しない実ヘッダ（EpochDomain / DeferredDeletionQueue /
#   RetireQuarantineStore / ISRRetireLatency）を使い、publish gate・reader fence・
#   enqueue/reclaim 同時実行・takeForShutdown の受け渡し・capability 付き checked
#   drain・quarantine の同時 store/drain の 6 シナリオを回す。
#
#   単独 configure（ルートの ConvoPeq フルビルド不要・Linux clang 専用）:
#     cmake -S src/tests/tsan-substrate -B build-tsan -G Ninja \
#       -DCMAKE_CXX_COMPILER=clang++ \
#       -DCMAKE_BUILD_TYPE=Release \
#       -DCMAKE_CXX_FLAGS="-fsanitize=thread -O1 -g" \
#       -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
#     cmake --build build-tsan
#     ctest --test-dir build-tsan --output-on-failure
#
#   ルート CMakeLists からは CONVOPEQ_ENABLE_TSAN_SUBSTRATE=ON で
#   add_subdirectory される（Windows の通常ビルドには登録されない）。
#
#   TsanPositiveControl は陽性対照（意図的な data race を入れたビルド）。
#   ctest の対象外。TSan が実際に競合を検出できることを確認するために使う
#   （検出できなければ exit code が 0 のまま → CI 側で失敗扱いにする）。
# ============================================================================
cmake_minimum_required(VERSION 3.21)
project(TsanSubstrateRace LANGUAGES CXX)

set(REPO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../..")

find_package(Threads REQUIRED)

add_executable(TsanSubstrateRaceTests TsanSubstrateRaceTests.cpp)
target_include_directories(TsanSubstrateRaceTests PRIVATE ${REPO_SRC})
target_compile_definitions(TsanSubstrateRaceTests PRIVATE
    CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY=1
)
target_compile_features(TsanSubstrateRaceTests PRIVATE cxx_std_20)
target_link_libraries(TsanSubstrateRaceTests PRIVATE Threads::Threads)

add_executable(TsanPositiveControl TsanSubstrateRaceTests.cpp)
target_include_directories(TsanPositiveControl PRIVATE ${REPO_SRC})
target_compile_definitions(TsanPositiveControl PRIVATE
    CONVOPEQ_ENABLE_DRAIN_TEST_FACTORY=1
    TSAN_POSITIVE_CONTROL=1
)
target_compile_features(TsanPositiveControl PRIVATE cxx_std_20)
target_link_libraries(TsanPositiveControl PRIVATE Threads::Threads)

enable_testing()
add_test(NAME TsanSubstrateRace COMMAND TsanSubstrateRaceTests)
