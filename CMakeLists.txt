#============================================================================
# CMakeLists.txt  ── v0.6.10 (JUCE 8.0.12 / VS Code + MSVC + icx + Windows 11)
#
# ビルド環境:
#   - JUCE: 8.0.12
#   - Intel oneAPI (MKL/IPP)
#   - コンパイラ: MSVC 19.44+ (VS2022 17.11+) または Intel icx (oneAPI 2026.0)
#   - IDE: Visual Studio Code
#   - プラットフォーム: Windows 11 x64
#   - C++標準: C++20
#
# ビルド方法:
#   build.bat [Debug|Release]          # MSVC (デフォルト)
#   build.bat [Debug|Release] icx      # Intel icx
#   build.bat [Debug|Release] clean icx
#============================================================================

cmake_minimum_required(VERSION 3.22)

# CMP0091: MSVC_RUNTIME_LIBRARY プロパティを有効化
# CMake 3.9+ で導入。cmake_minimum_required(VERSION 3.22) により暗黙的に NEW だが、
# 明示指定により可読性・メンテナンス性を向上させる。
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

# Load project metadata from dedicated file
include(ProjectMetadata.cmake)

project(${APP_NAME}
    VERSION ${APP_VERSION_MAJOR}.${APP_VERSION_MINOR}.${APP_VERSION_PATCH}
    LANGUAGES CXX C)

# Tell CMake to generate the compile_commands.json file for clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#------------------------------------------------------------
# Clang-Tidy Integration (Build-time analysis)
#------------------------------------------------------------
option(CONVOPEQ_ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
find_program(CLANG_TIDY_EXECUTABLE clang-tidy)

# グローバル適用を明示的に無効化（JUCEモジュール巻き込み防止）
set(CMAKE_CXX_CLANG_TIDY "")

if(CONVOPEQ_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXECUTABLE)
    message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXECUTABLE}")
else()
    message(STATUS "clang-tidy build-time integration disabled.")
endif()

# CONVO_CI_BUILD 定義時は NUC_DEBUG_GUARDS も有効化（デバッグガードの CI 対応）
if(DEFINED ENV{CONVO_CI_BUILD})
    add_compile_definitions(NUC_DEBUG_GUARDS)
    message(STATUS "CI build detected: NUC_DEBUG_GUARDS enabled")
    # ★ [P2-2] CI ビルド時は clang-tidy を強制 ON（ローカルは OFF 維持）
    set(CONVOPEQ_ENABLE_CLANG_TIDY ON CACHE BOOL "Run clang-tidy during build" FORCE)
endif()

#------------------------------------------------------------
# Intel MKL / IPP Detection (must run BEFORE test target link logic)
# ★ Fix: CONVOPEQ_HAS_MKL / IPP_FOUND must be set before any target_link_libraries
#   references MKL::MKL / IPP::ippcore, because CMake evaluates if() immediately.
#   ConvoPeq target doesn't exist yet at this point, so linking is deferred
#   to the post-target section below.
#------------------------------------------------------------
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)

option(CONVOPEQ_REQUIRE_MKL "Require Intel MKL for MSVC builds (uses system aligned allocator when OFF)" ON)
set(CONVOPEQ_HAS_MKL OFF)

if(DEFINED ENV{MKLROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}/lib/cmake/mkl")
endif()

if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(CONVOPEQ_REQUIRE_MKL)
        find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
        set(CONVOPEQ_HAS_MKL ON)
        message(STATUS "MKL: found via CMake Config, will link MKL::MKL (CONVOPEQ_REQUIRE_MKL=ON)")
    else()
        find_package(MKL CONFIG QUIET COMPONENTS intel_lp64 sequential)
        if(MKL_FOUND)
            set(CONVOPEQ_HAS_MKL ON)
            message(STATUS "MKL: found (CONVOPEQ_REQUIRE_MKL=OFF but MKL present)")
        else()
            message(STATUS "MKL: not found, CONVOPEQ_REQUIRE_MKL=OFF → system aligned allocator (work72)")
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx: /Qmkl:sequential でコンパイラがMKLを自動リンク
    set(CONVOPEQ_HAS_MKL ON)
    # icx: Intel コンパイラランタイム (libircmt.lib / libmmd.lib) のパスを明示
    set(_INTEL_COMPILER_ROOT "")
    if(DEFINED ENV{ONEAPI_ROOT})
        set(_INTEL_COMPILER_ROOT "$ENV{ONEAPI_ROOT}/compiler/latest/lib")
    endif()
    if(NOT _INTEL_COMPILER_ROOT OR NOT EXISTS "${_INTEL_COMPILER_ROOT}")
        set(_INTEL_COMPILER_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    endif()
    # ★ Defer ConvoPeq link directories + /Qmkl to after target creation
    set(CONVOPEQ_INTEL_RT_LIB_DIR "${_INTEL_COMPILER_ROOT}")
    unset(_INTEL_COMPILER_ROOT)
endif()

#------------------------------------------------------------
# Intel IPP Detection (also moved before test targets)
#------------------------------------------------------------
set(IPP_LINK static)
set(IPP_THREADING sequential)
if(DEFINED ENV{IPPROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{IPPROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{IPPROOT}/lib/cmake/ipp")
endif()
find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)
if(IPP_FOUND)
    message(STATUS "Intel IPP found.")
else()
    message(STATUS "Intel IPP not found - skipping (CI build or no IPP SDK).")
endif()

#------------------------------------------------------------
# ISR Runtime Regression Tests (minimal CTest)
#------------------------------------------------------------
option(CONVOPEQ_ENABLE_ISR_TESTS "Enable minimal ISR runtime regression tests" ON)
option(CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS "Enable runtime diagnostic logging (XRUN/MEM/VERIFY/WORLD/etc.)" OFF)

if(CONVOPEQ_ENABLE_ISR_TESTS)
    enable_testing()

    file(WRITE "${CMAKE_BINARY_DIR}/DartConfiguration.tcl"
        "SET(CTEST_PROJECT_NAME \"${PROJECT_NAME}\")\n"
        "SET(CTEST_SOURCE_DIRECTORY \"${CMAKE_SOURCE_DIR}\")\n"
        "SET(CTEST_BINARY_DIRECTORY \"${CMAKE_BINARY_DIR}\")\n")

    add_executable(ISRRuntimeIdentityTests
        src/tests/ISRRuntimeIdentityGeneratorsTests.cpp
    )

    add_executable(RuntimePublicationCoordinatorTests
        src/tests/RuntimePublicationCoordinatorTests.cpp
    )

    # ★ Phase-2: Admission Policy unit test (pure Policy — evaluateDeferred() only)。
    #   PublicationAdmission.cpp → AudioEngine.h → JuceHeader.h は CONVOPEQ_HAS_MKL を
    #   継承するため、ISRSemanticValidationTests と同一パターンで mkl インクルードパスを
    #   追加 + MKL::MKL をリンク（DiagnosticsConfig.h の mkl.h 解決 + mkl_malloc シンボル）。
    add_executable(PublicationAdmissionTests
        src/tests/PublicationAdmissionTests.cpp
        src/audioengine/PublicationAdmission.cpp
        src/tests/PublicationAdmissionTestRuntimeStubs.cpp
    )
    target_include_directories(PublicationAdmissionTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/convolver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(PublicationAdmissionTests SYSTEM PRIVATE
        "$ENV{MKLROOT}/include"
        "$ENV{IPPROOT}/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
    )
    target_link_libraries(PublicationAdmissionTests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)
    target_compile_definitions(PublicationAdmissionTests PRIVATE _UNICODE UNICODE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    target_compile_features(PublicationAdmissionTests PRIVATE cxx_std_20)
    target_compile_options(PublicationAdmissionTests PRIVATE /utf-8)
    add_dependencies(PublicationAdmissionTests ConvoPeq)
    add_test(NAME PublicationAdmissionTests COMMAND PublicationAdmissionTests)

    # ISRSemanticValidationTests は ISR セマンティクス検証用。
    # RuntimePublicationCoordinator を直接使用するため .cpp を含める。
    # JUCE/MKL/r8brain の include path が必要（AudioEngine.h 経由の依存）。
    add_executable(ISRSemanticValidationTests
        src/tests/ISRSemanticValidationTests.cpp
        src/audioengine/ISRClosure.cpp
        src/audioengine/ISRPayloadTier.cpp
        src/audioengine/ISRRetireRouter.cpp
        src/audioengine/ISRRetire.cpp
        src/audioengine/ISRRetireRuntimeEx.cpp     # ★ A2: EpochControl (ctor referenced via LifetimeState)
        src/audioengine/ISRRuntimePublicationCoordinator.cpp
        src/audioengine/ISRDSPHandle.cpp
        src/audioengine/ISRDSPQuarantine.cpp
    )
    target_include_directories(ISRSemanticValidationTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/convolver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(ISRSemanticValidationTests SYSTEM PRIVATE
        "$ENV{MKLROOT}/include"
        "$ENV{IPPROOT}/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
    )
    target_link_libraries(ISRSemanticValidationTests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)
    # JuceHeader.h の生成を ISRSemanticValidationTests より先に行う
    add_dependencies(ISRSemanticValidationTests ConvoPeq)

    # ★ work88 (P2-2): INV-3 / INV-5 不変条件テスト（RuntimePublicationCoordinator 直接使用）
    #   ISRSemanticValidationTests と同一パターンで ISR*.cpp を同一ターゲットに同梱する。
    add_executable(invariant_INV3_INV5Tests
        src/tests/invariant_INV3_INV5.cpp
        src/audioengine/ISRClosure.cpp
        src/audioengine/ISRPayloadTier.cpp
        src/audioengine/ISRRetireRouter.cpp
        src/audioengine/ISRRetire.cpp
        src/audioengine/ISRRetireRuntimeEx.cpp     # ★ A2: EpochControl (ctor referenced via LifetimeState)
        src/audioengine/ISRRuntimePublicationCoordinator.cpp
        src/audioengine/ISRDSPHandle.cpp
        src/audioengine/ISRDSPQuarantine.cpp
    )
    target_include_directories(invariant_INV3_INV5Tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/convolver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(invariant_INV3_INV5Tests SYSTEM PRIVATE
        "$ENV{MKLROOT}/include"
        "$ENV{IPPROOT}/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
    )
    target_link_libraries(invariant_INV3_INV5Tests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)
    # JuceHeader.h の生成を invariant_INV3_INV5Tests より先に行う
    add_dependencies(invariant_INV3_INV5Tests ConvoPeq)

    add_executable(RetireGraceSemanticsTests
        src/tests/RetireGraceSemanticsTests.cpp
    )

    # ★ Work91: ISRSoakTests — ヘッドレスデータ構造耐久（publish を含まない）。
    #   publish 系は AudioEngineHarness 側に配置する原則（doc/work91/soak-test-design.md §2.2）。
    #   ISRSemanticValidationTests と同じ include/link パターン（RuntimePublicationCoordinator
    #   を直接使用するため .cpp を含める）。
    add_executable(ISRSoakTests
        src/tests/ISRSoakTests.cpp
        src/audioengine/ISRClosure.cpp
        src/audioengine/ISRPayloadTier.cpp
        src/audioengine/ISRRetireRouter.cpp
        src/audioengine/ISRRetire.cpp
        src/audioengine/ISRRetireRuntimeEx.cpp     # ★ A2: EpochControl (ctor referenced via LifetimeState)
        src/audioengine/ISRRuntimePublicationCoordinator.cpp
        src/audioengine/ISRDSPHandle.cpp
        src/audioengine/ISRDSPQuarantine.cpp
    )
    target_include_directories(ISRSoakTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/convolver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(ISRSoakTests SYSTEM PRIVATE
        "$ENV{MKLROOT}/include"
        "$ENV{IPPROOT}/include"
        ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
    )
    target_link_libraries(ISRSoakTests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)
    add_dependencies(ISRSoakTests ConvoPeq)
    target_compile_features(ISRSoakTests PRIVATE cxx_std_20)
    target_compile_options(ISRSoakTests PRIVATE /EHsc /utf-8)

    add_test(NAME ISRSoakTests COMMAND ISRSoakTests)

    # ★ B2: OwnerChannel unit tests (OwnerChannel.h is JUCE-independent; tests sole-
    #   ownership transfer, single-take, key isolation, no-overwrite, no-leak under SPSC).
    add_executable(OwnerChannelTests
        src/tests/OwnerChannelTests.cpp
    )
    target_compile_features(OwnerChannelTests PRIVATE cxx_std_20)
    target_compile_options(OwnerChannelTests PRIVATE /EHsc /utf-8)
    target_include_directories(OwnerChannelTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    add_test(NAME OwnerChannel COMMAND OwnerChannelTests)

    # P0-2b: PublishReceipt DSPCore*削除 — DSPHandle 比較の検証
    # DSPHandleRuntime::getFadingRuntimeDSPHandle() の動作検証のため
    # ISRDSPHandle.cpp を同一ターゲットでコンパイルする。
    add_executable(NormalRetireDSPHandleCompareTests
        src/tests/NormalRetireDSPHandleCompareTests.cpp
        src/audioengine/ISRDSPHandle.cpp
    )

    add_executable(RuntimeSemanticSchemaValidationTests
        src/tests/RuntimeSemanticSchemaValidationTests.cpp
    )

    add_executable(ObservePathSingleSourceTests
        src/tests/ObservePathSingleSourceTests.cpp
    )

    add_executable(OverlapAuthoritySingularTests
        src/tests/OverlapAuthoritySingularTests.cpp
    )

    add_executable(ShadowCompareContractTests
        src/tests/ShadowCompareContractTests.cpp
    )

    add_executable(CrossfadeExecutorLocalContractTests
        src/tests/CrossfadeExecutorLocalContractTests.cpp
    )

    add_executable(RuntimeWorldAuthorityProjectionTests
        src/tests/RuntimeWorldAuthorityProjectionTests.cpp
    )

    add_executable(PartialPublicationRejectTests
        src/tests/PartialPublicationRejectTests.cpp
    )

    add_executable(RebuildAdmissionRegressionTests
        src/tests/RebuildAdmissionRegressionTests.cpp
    )

    add_executable(BuildInputSemanticContractTests
        src/tests/BuildInputSemanticContractTests.cpp
    )

    # ★ Phase3/5: Priority Integration Tests
    # ★ Bug#5: DeferredDeletionQueue reclaim 自動実測テスト
    #   MPMC epoch逆転測定・同時実行ストレステスト・基本機能検証
    add_executable(DeferredDeletionQueueReclaimTests
        src/tests/DeferredDeletionQueueReclaimTests.cpp
    )
    add_test(NAME DeferredDeletionQueueReclaimTests
        COMMAND DeferredDeletionQueueReclaimTests)

    # ★ work88 (FUTURE-10 前提 0): MpscBoundedRing 単体テスト（Vyukov bounded MPSC）
    #   1/N producers・queue full・FIFO・producer hole・cross-type FIFO を検証
    add_executable(MpscBoundedRingTests
        src/tests/MpscBoundedRingTests.cpp
    )
    # ★ work88 (P2-3): producer-hole 再現テスト（CONVO_TESTING フック）を有効化。
    #   本番ターゲットには定義しない（ヘッダのフックは CONVO_TESTING 時のみコンパイルされる）。
    target_compile_definitions(MpscBoundedRingTests PRIVATE CONVO_TESTING=1)
    add_test(NAME MpscBoundedRingTests
        COMMAND MpscBoundedRingTests)

    # ★ work88 (FUTURE-6 監査): DSPHandleTable 単体テスト（open addressing + tombstone）
    #   erase 後のクラスタ断絶リグレッション防止（find が後続エントリを探し続けること）
    add_executable(DSPHandleTableTests
        src/tests/DSPHandleTableTests.cpp
    )
    add_test(NAME DSPHandleTableTests
        COMMAND DSPHandleTableTests)

    add_executable(PriorityIntegrationTests
        src/tests/PriorityIntegrationTests.cpp
        src/audioengine/ISRRetire.cpp              # RetireRuntime
        src/audioengine/ISRRetireRuntimeEx.cpp     # A2: EpochControl (ctor referenced via LifetimeState)
    )
    target_compile_options(PriorityIntegrationTests PRIVATE /EHsc)
    # ★ A2: ISRRetireRuntimeEx.cpp pulls JuceHeader.h (via DspNumericPolicy.h) → add the JUCE include
    #   path + juce module link, mirroring ISRSemanticValidationTests (L95-110).
    target_include_directories(PriorityIntegrationTests PRIVATE
        "${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode"
        "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules"
    )
    target_link_libraries(PriorityIntegrationTests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)
    # ★ A2: ISRRetireRuntimeEx.cpp pulls JuceHeader.h (via DspNumericPolicy.h) → needs the JUCE include
    #   path + juce module link, mirroring ISRSemanticValidationTests. ConvoPeq dependency generates JuceHeader.h.
    target_include_directories(PriorityIntegrationTests PRIVATE
        "${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode"
        "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules"
    )
    add_dependencies(PriorityIntegrationTests ConvoPeq)
    target_link_libraries(PriorityIntegrationTests PRIVATE juce::juce_core juce::juce_gui_extra juce::juce_gui_basics r8brain)

    # ★ v14.0 Phase 8: Auto Gain Staging 契約テスト
    #   AutoGainPlanner::plan() の4パターン × Auto On/Off を検証（リファレンス実装）。
    #   JUCE/MKL に依存しない純粋関数テスト。
    add_executable(GainStagingContractTests
        src/tests/GainStagingContractTests.cpp
    )
    target_include_directories(GainStagingContractTests PRIVATE ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    add_test(NAME GainStagingContractTests COMMAND GainStagingContractTests)

    # ★ v14.0 Phase 8: EQ 応答計算の数学的契約テスト
    #   getMagnitudeSquared + svfToDisplayBiquad の数学的正当性を検証。
    #   JUCE/MKL に依存しない純粋数学テスト。
    add_executable(EQProcessorMaxGainTests
        src/tests/EQProcessorMaxGainTests.cpp
    )
    target_include_directories(EQProcessorMaxGainTests PRIVATE ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    add_test(NAME EQProcessorMaxGainTests COMMAND EQProcessorMaxGainTests)

    # ★ v14.47: 3層リファクタリング 単体テスト
    #   PeakEstimator / UpperBoundEstimator / AnalysisMerge / EQResponseSampler
    #   JUCE/MKL に依存しない純粋数学テスト。
    add_executable(EQAnalysisUnitTests
        src/tests/EQAnalysisUnitTests.cpp
    )
    target_include_directories(EQAnalysisUnitTests PRIVATE ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    add_test(NAME EQAnalysisUnitTests COMMAND EQAnalysisUnitTests)

    # ★ P1-1 Phase5: FFT Backend Concept + ProductionFft + TestFft テスト
    add_executable(FFTBackendTests
        src/tests/FFTBackendTests.cpp
        src/FFTBackend.cpp
        src/FFTExecutionContext.cpp
    )
    target_include_directories(FFTBackendTests PRIVATE ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/src)
    # ★ Guard IPP link: IPP::ippcore/IPP::ipps only exist if find_package(IPP) succeeded.
    #   CI runners without Intel oneAPI will have IPP_FOUND=FALSE → skip linking gracefully.
    if(IPP_FOUND)
        target_link_libraries(FFTBackendTests PRIVATE IPP::ippcore IPP::ipps)
    endif()
    target_compile_features(FFTBackendTests PRIVATE cxx_std_20)
    target_compile_options(FFTBackendTests PRIVATE /EHsc)
    add_test(NAME FFTBackendTests COMMAND FFTBackendTests)

    # ★ Week2: boundExcessDb 分布測定ベンチマーク
    #   JUCE/MKL に依存しない純粋数学ベンチマーク。
    add_executable(EQBoundExcessBenchmark
        src/tests/EQBoundExcessBenchmark.cpp
    )
    target_include_directories(EQBoundExcessBenchmark PRIVATE ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    add_test(NAME EQBoundExcessBenchmark COMMAND EQBoundExcessBenchmark --quick)

    if(CONVOPEQ_HAS_MKL AND MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_link_libraries(RuntimePublicationCoordinatorTests PRIVATE MKL::MKL)
        target_link_libraries(ISRSemanticValidationTests PRIVATE MKL::MKL)
        target_link_libraries(invariant_INV3_INV5Tests PRIVATE MKL::MKL)
        target_link_libraries(PartialPublicationRejectTests PRIVATE MKL::MKL)
        # ★ ASan-CMAKE-4 修正（2026-07-31）: FFTBackend.cpp が AlignedAllocation.h
        #   (mkl_malloc/mkl_free) を使用するため MKL::MKL をリンク（include も propagate される）
        target_link_libraries(FFTBackendTests PRIVATE MKL::MKL)
        # ★ Work91: ISRSoakTests も AlignedAllocation.h (mkl_malloc/mkl_free) を使用するため
        #   MKL::MKL をリンク（RuntimeState::createForBuilder が aligned_make_unique を呼ぶ）。
        target_link_libraries(ISRSoakTests PRIVATE MKL::MKL)
        # ★ Phase-2: PublicationAdmission.cpp → AudioEngine.h → JuceHeader.h に
        #   CONVOPEQ_HAS_MKL 定義を継承するため、mkl.h + mkl シンボルを解決する。
        #   ISRSemanticValidationTests と同一パターン。
        target_link_libraries(PublicationAdmissionTests PRIVATE MKL::MKL)
        # ★ DiagnosticsConfig.h は JUCE_DSP_USE_INTEL_MKL 定義에 따라 mkl.h 를 include する。
        #   ConvoPeq の PRIVATE 정의は 테스트 타깃에 상속되지 않으므로、
        #   여기서 직접 정의 + MKLROOT/include를 명시적으로 추가して mkl.h 解決을 보장する。
        target_compile_definitions(ISRSemanticValidationTests PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
        target_compile_definitions(invariant_INV3_INV5Tests PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
        target_compile_definitions(ISRSoakTests PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
        target_compile_definitions(PublicationAdmissionTests PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
        target_include_directories(ISRSemanticValidationTests SYSTEM PRIVATE "$ENV{MKLROOT}/include")
        target_include_directories(invariant_INV3_INV5Tests SYSTEM PRIVATE "$ENV{MKLROOT}/include")
        target_include_directories(ISRSoakTests SYSTEM PRIVATE "$ENV{MKLROOT}/include")
        target_include_directories(PublicationAdmissionTests SYSTEM PRIVATE "$ENV{MKLROOT}/include")
    endif()

    target_compile_features(ISRRuntimeIdentityTests PRIVATE cxx_std_20)
    target_compile_features(RuntimePublicationCoordinatorTests PRIVATE cxx_std_20)
    target_compile_features(ISRSemanticValidationTests PRIVATE cxx_std_20)
    target_compile_features(invariant_INV3_INV5Tests PRIVATE cxx_std_20)
    target_compile_features(RetireGraceSemanticsTests PRIVATE cxx_std_20)
    target_compile_features(NormalRetireDSPHandleCompareTests PRIVATE cxx_std_20)
    target_compile_features(RuntimeSemanticSchemaValidationTests PRIVATE cxx_std_20)
    target_compile_features(ObservePathSingleSourceTests PRIVATE cxx_std_20)
    target_compile_features(OverlapAuthoritySingularTests PRIVATE cxx_std_20)
    target_compile_features(ShadowCompareContractTests PRIVATE cxx_std_20)
    target_compile_features(CrossfadeExecutorLocalContractTests PRIVATE cxx_std_20)
    target_compile_features(RuntimeWorldAuthorityProjectionTests PRIVATE cxx_std_20)
    target_compile_features(PartialPublicationRejectTests PRIVATE cxx_std_20)
    target_compile_features(GainStagingContractTests PRIVATE cxx_std_20)
    target_compile_features(EQProcessorMaxGainTests PRIVATE cxx_std_20)
    target_compile_features(EQAnalysisUnitTests PRIVATE cxx_std_20)
    target_compile_features(EQBoundExcessBenchmark PRIVATE cxx_std_20)
    target_compile_features(RebuildAdmissionRegressionTests PRIVATE cxx_std_20)
    target_compile_features(BuildInputSemanticContractTests PRIVATE cxx_std_20)
    target_compile_features(DeferredDeletionQueueReclaimTests PRIVATE cxx_std_20)
    target_compile_features(PriorityIntegrationTests PRIVATE cxx_std_20)
    target_include_directories(ISRRuntimeIdentityTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
    )
    target_include_directories(RuntimePublicationCoordinatorTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(ISRSemanticValidationTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(RetireGraceSemanticsTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_include_directories(NormalRetireDSPHandleCompareTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(RuntimeSemanticSchemaValidationTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(ObservePathSingleSourceTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(OverlapAuthoritySingularTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(ShadowCompareContractTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(CrossfadeExecutorLocalContractTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(RuntimeWorldAuthorityProjectionTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(PartialPublicationRejectTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(RebuildAdmissionRegressionTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(BuildInputSemanticContractTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(DeferredDeletionQueueReclaimTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_compile_features(MpscBoundedRingTests PRIVATE cxx_std_20)
    target_compile_options(MpscBoundedRingTests PRIVATE /EHsc /utf-8)
    target_include_directories(MpscBoundedRingTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_compile_features(DSPHandleTableTests PRIVATE cxx_std_20)
    target_compile_options(DSPHandleTableTests PRIVATE /EHsc /utf-8)
    target_include_directories(DSPHandleTableTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    )
    target_include_directories(PriorityIntegrationTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    )
    add_test(NAME ISRRuntimeIdentityGenerators COMMAND ISRRuntimeIdentityTests)
    add_test(NAME RuntimePublicationCoordinatorRejects COMMAND RuntimePublicationCoordinatorTests)
    add_test(NAME ISRSemanticValidationRejects COMMAND ISRSemanticValidationTests)
    add_test(NAME InvariantINV3INV5 COMMAND invariant_INV3_INV5Tests)
    add_test(NAME RetireGraceSemantics COMMAND RetireGraceSemanticsTests)
    add_test(NAME NormalRetireDSPHandleCompare COMMAND NormalRetireDSPHandleCompareTests)
    add_test(NAME RuntimeSemanticSchemaValidation COMMAND RuntimeSemanticSchemaValidationTests)
    add_test(NAME ObservePathSingleSource COMMAND ObservePathSingleSourceTests)
    add_test(NAME OverlapAuthoritySingular COMMAND OverlapAuthoritySingularTests)
    add_test(NAME ShadowCompareContract COMMAND ShadowCompareContractTests)
    add_test(NAME CrossfadeExecutorLocalContract COMMAND CrossfadeExecutorLocalContractTests)
    add_test(NAME RuntimeWorldAuthorityProjectionContract COMMAND RuntimeWorldAuthorityProjectionTests)
    add_test(NAME PartialPublicationReject COMMAND PartialPublicationRejectTests)
    add_test(NAME RebuildAdmissionRegression COMMAND RebuildAdmissionRegressionTests)
    # CI環境（CONVO_CI_BUILD定義時）は音声デバイスがないためスキップ
    if(NOT DEFINED ENV{CONVO_CI_BUILD})
        add_test(NAME HeadlessAudioPathVerification COMMAND powershell -NoProfile -ExecutionPolicy Bypass -File ${CMAKE_CURRENT_SOURCE_DIR}/.github/scripts/cli-smoke-test.ps1 -KillExisting -RequireAudioCallbacks)
    endif()
    add_test(NAME BuildInputSemanticContract COMMAND BuildInputSemanticContractTests)
    add_test(NAME PriorityIntegration COMMAND PriorityIntegrationTests)

    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ISRRuntimeIdentityTests PRIVATE /utf-8)
        target_compile_options(RuntimePublicationCoordinatorTests PRIVATE /utf-8)
        target_compile_options(ISRSemanticValidationTests PRIVATE /utf-8)
        target_compile_options(invariant_INV3_INV5Tests PRIVATE /utf-8)
        target_compile_options(RetireGraceSemanticsTests PRIVATE /utf-8)
        target_compile_options(NormalRetireDSPHandleCompareTests PRIVATE /utf-8)
        target_compile_options(RuntimeSemanticSchemaValidationTests PRIVATE /utf-8)
        target_compile_options(ObservePathSingleSourceTests PRIVATE /utf-8)
        target_compile_options(OverlapAuthoritySingularTests PRIVATE /utf-8)
        target_compile_options(ShadowCompareContractTests PRIVATE /utf-8)
        target_compile_options(CrossfadeExecutorLocalContractTests PRIVATE /utf-8)
        target_compile_options(RuntimeWorldAuthorityProjectionTests PRIVATE /utf-8)
        target_compile_options(PartialPublicationRejectTests PRIVATE /utf-8)
        target_compile_options(RebuildAdmissionRegressionTests PRIVATE /utf-8)
        target_compile_options(BuildInputSemanticContractTests PRIVATE /utf-8)
        target_compile_options(PriorityIntegrationTests PRIVATE /utf-8)
        target_compile_options(ISRSoakTests PRIVATE /utf-8)
    endif()

    # テストターゲットに Windows プラットフォーム定義を追加（Debug ビルドでも標準ヘッダーが正しく include されるよう確保）
    if(WIN32)
        foreach(tgt IN ITEMS ISRRuntimeIdentityTests RuntimePublicationCoordinatorTests
                     ISRSemanticValidationTests invariant_INV3_INV5Tests
                     RetireGraceSemanticsTests
                     NormalRetireDSPHandleCompareTests
                     RuntimeSemanticSchemaValidationTests ObservePathSingleSourceTests
                     OverlapAuthoritySingularTests ShadowCompareContractTests
                     CrossfadeExecutorLocalContractTests RuntimeWorldAuthorityProjectionTests
                     PartialPublicationRejectTests RebuildAdmissionRegressionTests
                     BuildInputSemanticContractTests PriorityIntegrationTests ISRSoakTests)
            target_compile_definitions(${tgt} PRIVATE
                _UNICODE UNICODE NOMINMAX _CRT_SECURE_NO_WARNINGS
            )
        endforeach()
    endif()

    # icx: テスト実行ファイルで LTCG(/Qipo) を無効化
    # icx(clang-cl) が LTCG 有効時に生成する .obj は lld-link が認識できない形式になり、
    # LNK1107 "ファイルが無効" エラーが発生する。テストはコンパイルのみ正常でリンク時に失敗する。
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set_target_properties(ISRRuntimeIdentityTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(RuntimePublicationCoordinatorTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(ISRSemanticValidationTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(invariant_INV3_INV5Tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(RetireGraceSemanticsTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(NormalRetireDSPHandleCompareTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(RuntimeSemanticSchemaValidationTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(ObservePathSingleSourceTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(OverlapAuthoritySingularTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(ShadowCompareContractTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(CrossfadeExecutorLocalContractTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(RuntimeWorldAuthorityProjectionTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(PartialPublicationRejectTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(RebuildAdmissionRegressionTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
        set_target_properties(BuildInputSemanticContractTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()

    # テストターゲットに /EHsc を追加（例外処理を有効化）
    # AlignedAllocation.h の throw/try-catch 使用箇所、および MSVC <chrono> の
    # 例外パスで C4530 (アンワインド未有効) が発生するため全コンパイラに適用
    target_compile_options(ISRRuntimeIdentityTests PRIVATE /EHsc)
    target_compile_options(RuntimePublicationCoordinatorTests PRIVATE /EHsc)
    target_compile_options(ISRSemanticValidationTests PRIVATE /EHsc)
    target_compile_options(invariant_INV3_INV5Tests PRIVATE /EHsc)
    # ISRSemanticValidationTests は juce_core.cpp をコンパイルするため、
    # コマンドライン NOMINMAX=1 と JUCE の #define NOMINMAX（空）の衝突警告を抑制する。
    # （ConvoPeq 本体と同様の対応。CMakeLists.txt 参照）
    # ★ ASan-CMAKE-4 修正（2026-07-31）: これらの -Wno-* は Clang 系（icx）専用であり、
    #   MSVC cl では D8021（無効なコマンドライン引数）になる。Clang/icx 系のみに限定する。
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
        target_compile_options(ISRSemanticValidationTests PRIVATE
            -Wno-unused-command-line-argument
            -Wno-macro-redefined
        )
    endif()
    target_compile_options(RetireGraceSemanticsTests PRIVATE /EHsc)
    target_compile_options(NormalRetireDSPHandleCompareTests PRIVATE /EHsc)
    target_compile_options(RuntimeSemanticSchemaValidationTests PRIVATE /EHsc)
    target_compile_options(ObservePathSingleSourceTests PRIVATE /EHsc)
    target_compile_options(OverlapAuthoritySingularTests PRIVATE /EHsc)
    target_compile_options(ShadowCompareContractTests PRIVATE /EHsc)
    target_compile_options(CrossfadeExecutorLocalContractTests PRIVATE /EHsc)
    target_compile_options(RuntimeWorldAuthorityProjectionTests PRIVATE /EHsc)
    target_compile_options(PartialPublicationRejectTests PRIVATE /EHsc)
    target_compile_options(RebuildAdmissionRegressionTests PRIVATE /EHsc)
    target_compile_options(BuildInputSemanticContractTests PRIVATE /EHsc)
    # BuildInputSemanticContractTests: 大規模ソース読み取りでスタックオーバーフローを防ぐため8MBに拡大
    # B5: MSVC と IntelLLVM の両方で適用（clang-cl 互換）
    if(MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
        target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
    endif()

    # icx: MKL を使用するテストターゲットに /Qmkl:sequential を追加
    # （MSVC は target_link_libraries で MKL::MKL をリンク、icx はコンパイルオプションで
    #  リンク指示を .obj に埋め込む。ConvoPeq 本体は post-target MKL config セクションで設定済み）
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(RuntimePublicationCoordinatorTests PRIVATE /Qmkl:sequential)
        target_compile_options(PartialPublicationRejectTests PRIVATE /Qmkl:sequential)
    endif()
endif()

# バージョン文字列の分解 (必要に応じて使用)
# string(REPLACE "." ";" ConvoPeq_VERSION_LIST "${PROJECT_VERSION}")
# list(GET ConvoPeq_VERSION_LIST 0 ConvoPeq_VERSION_MAJOR)
# list(GET ConvoPeq_VERSION_LIST 1 ConvoPeq_VERSION_MINOR)
# list(GET ConvoPeq_VERSION_LIST 2 ConvoPeq_VERSION_PATCH)

#------------------------------------------------------------
# C++標準設定
#------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------
# x64 固定（Visual Studio / Ninja Multi-Config）
#------------------------------------------------------------
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(FATAL_ERROR "ConvoPeq must be configured as x64. Current target is x86.")
    endif()
endif()

#------------------------------------------------------------
# Windows RC Compiler Fix for Ninja
#------------------------------------------------------------
if(WIN32 AND CMAKE_GENERATOR MATCHES "Ninja")
    # MSVC環境が未初期化だと windows.h / C++標準ヘッダーが見つからない
    if(MSVC AND (NOT DEFINED ENV{VCToolsInstallDir} OR NOT DEFINED ENV{WindowsSdkDir}))
        message(FATAL_ERROR
            "MSVC/Windows SDK environment is not initialized.\n"
            "Please build via build.bat or run vcvarsall.bat x64 before CMake."
        )
    endif()

    if(DEFINED ENV{INCLUDE})
        # rc.exe 向けに INCLUDE を /I へ展開
        string(REPLACE ";" "|" _include_raw "$ENV{INCLUDE}")
        string(REPLACE "|" ";" INCLUDE_LIST "${_include_raw}")
        foreach(p IN LISTS INCLUDE_LIST)
            if(NOT p STREQUAL "")
                string(APPEND CMAKE_RC_FLAGS " /I\"${p}\"")
            endif()
        endforeach()
    endif()
endif()

#------------------------------------------------------------
# icx(Clang-cl) 전역 경고 억제
# NOMINMAX 재정의(-Wmacro-redefined) 및 UNUSED-COMMAND-LINE-ARGUMENT
# 경고는 JUCE 모듈(native header)과 충돌하므로 전역적으로 억제
#------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    add_compile_options(
        -Wno-macro-redefined
        -Wno-unused-command-line-argument
    )
endif()

#------------------------------------------------------------
# JUCEを追加（JUCE 8.0以降はextras/Build/CMake/JUCEModuleSupport.cmake不要）
# プロジェクトルートに JUCE/ ディレクトリが必要
#------------------------------------------------------------
add_subdirectory(JUCE)

#------------------------------------------------------------
# r8brain-free-src ライブラリの追加
#------------------------------------------------------------
add_library(r8brain INTERFACE)

# Mark r8brain includes as SYSTEM to suppress clang-tidy warnings.
target_include_directories(r8brain SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src")

target_compile_features(r8brain INTERFACE cxx_std_17)
target_compile_definitions(r8brain INTERFACE
    R8B_FASTTIMING=1          # 高速モード（推奨）
    R8B_EXTFILTERS=0
)

#------------------------------------------------------------
# GUIアプリケーション定義
#------------------------------------------------------------
juce_add_gui_app(ConvoPeq
    PRODUCT_NAME "${APP_NAME}"
    VERSION ${PROJECT_VERSION}
    COMPANY_NAME "${APP_COMPANY}"
    BUNDLE_ID "${APP_BUNDLE_ID}"
    ICON_BIG "resources/icon.png"
)

# ★ B13: MT-NUPC Measurement (Phase 1 — コンソールアプリ)
if(CONVOPEQ_ENABLE_ISR_TESTS)
    juce_add_console_app(MTNUPCMeasurement)
    target_sources(MTNUPCMeasurement PRIVATE
        src/tests/MT-NUPC-Measurement.cpp
        src/MKLNonUniformConvolver.cpp
        # P1-1: FFT Backend Concept + ProductionFft + FFTExecutionContext
        src/FFTBackend.cpp
        src/FFTExecutionContext.cpp
    )
    target_include_directories(MTNUPCMeasurement PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/convolver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
    )
    target_include_directories(MTNUPCMeasurement SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
    )
    target_compile_definitions(MTNUPCMeasurement PRIVATE
        JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
        $<$<BOOL:${CONVOPEQ_HAS_MKL}>:JUCE_DSP_USE_INTEL_MKL=1>
        JUCE_USE_SIMD=1
    )
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        if(CONVOPEQ_HAS_MKL)
            target_link_libraries(MTNUPCMeasurement PRIVATE MKL::MKL)
        else()
            message(STATUS "MTNUPCMeasurement: CONVOPEQ_HAS_MKL=OFF → skipping MKL link (system aligned allocator)")
        endif()
        target_compile_options(MTNUPCMeasurement PRIVATE /arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(MTNUPCMeasurement PRIVATE /QxCORE-AVX2 /Qmkl:sequential
            -Wno-macro-redefined
            -Wno-unused-command-line-argument
        )
    endif()
    target_link_libraries(MTNUPCMeasurement PRIVATE r8brain)
    if(MSVC)
        target_compile_options(MTNUPCMeasurement PRIVATE /utf-8 /EHsc)
    endif()
    target_compile_definitions(MTNUPCMeasurement PRIVATE
        _UNICODE UNICODE NOMINMAX _CRT_SECURE_NO_WARNINGS
    )
    add_test(NAME MTNUPCMeasurement COMMAND MTNUPCMeasurement)
    juce_generate_juce_header(MTNUPCMeasurement)
    # ★ JUCE モジュールをリンク
    target_link_libraries(MTNUPCMeasurement PRIVATE
        juce::juce_core juce::juce_dsp juce::juce_audio_basics
        juce::juce_audio_utils juce::juce_events
    )
    # ★ IPP include (リンクは IPP 設定セクションで追加)
    if(DEFINED ENV{IPPROOT})
        target_include_directories(MTNUPCMeasurement SYSTEM PRIVATE "$ENV{IPPROOT}/include")
    endif()
endif()

#------------------------------------------------------------
# ソースファイル
#------------------------------------------------------------
set(CONVOPEQ_ALL_SOURCES
    src/MainApplication.cpp
    src/MainWindow.cpp
    src/audioengine/AudioEngine.RebuildDispatch.cpp
    src/audioengine/AudioEngine.Commit.cpp
    src/audioengine/AudioEngine.CtorDtor.cpp
    src/audioengine/AudioEngine.Globals.cpp
    src/audioengine/AudioEngine.Learning.cpp
    src/audioengine/AudioEngine.Init.cpp
    src/audioengine/AudioEngine.Cache.cpp
    src/audioengine/AudioEngine.Snapshot.cpp
    src/audioengine/AudioEngine.Fifo.cpp
    src/audioengine/AudioEngine.Timer.cpp
    src/audioengine/AudioEngine.Mmcss.cpp
    src/audioengine/AudioEngine.Transition.cpp
    src/audioengine/AudioEngine.Parameters.cpp
    src/audioengine/AudioEngine.EQResponse.cpp
    src/audioengine/AudioEngine.UIEvents.cpp
    src/audioengine/AudioEngine.StateIO.cpp
    src/audioengine/AudioEngine.Threading.cpp
    src/audioengine/AudioEngine.Publication.cpp
    src/audioengine/AudioEngine.Reader.cpp
    src/audioengine/AudioEngine.Retire.cpp
    src/audioengine/ISRLifecycle.cpp
        src/audioengine/ISRRTExecution.cpp
            src/audioengine/ISRDSPHandle.cpp
                src/audioengine/ISRClosure.cpp
                src/audioengine/ISRPayloadTier.cpp
                src/audioengine/ISRHB.cpp
                src/audioengine/ISRRetire.cpp
                src/audioengine/ISRRetireRouter.cpp
                src/audioengine/ISRShutdown.cpp
                src/audioengine/ISRRuntimePublicationCoordinator.cpp
                src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp
                src/audioengine/RuntimePublicationValidator.cpp
                src/audioengine/CrossfadeAuthority.cpp
                src/audioengine/PublicationAdmission.cpp
                src/audioengine/PublicationExecutor.cpp
                src/audioengine/RuntimePublicationOrchestrator.cpp
                src/audioengine/DSPLifetimeManager.cpp
                src/audioengine/ISRCoordinatorLoop.cpp  # ★ FUTURE-9: Dedicated Coordinator Worker
                src/audioengine/TelemetryRecorder.cpp
                src/audioengine/ISRDSPQuarantine.cpp
                src/audioengine/ISRClosureGraphWalker.cpp
                src/audioengine/ISRDebugRuntime.cpp
                src/audioengine/ISRRetireRuntimeEx.cpp
                src/audioengine/FrozenRuntimeWorld.cpp          # ★ Phase4
                src/audioengine/ISREvidenceExporter.cpp
                src/audioengine/RuntimeHealthMonitor.cpp
                src/audioengine/RuntimePolicyEngine.cpp    # ★ work37 Phase 0
                src/audioengine/WorldLifecycleAudit.cpp
                src/audioengine/AudioEngine.Processing.Latency.cpp
    src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp
    src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
    src/audioengine/AudioEngine.Processing.AudioBlock.cpp
    src/audioengine/AudioEngine.Processing.BlockDouble.cpp
    src/audioengine/AudioEngine.Processing.Snapshot.cpp
    src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp
    src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp
    src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp
    src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
    src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp
    src/CustomInputOversampler.cpp
    src/TruePeakDetector.cpp
    src/LoudnessMeter.cpp
    src/NoiseShaperLearnerTypes.h
    src/audioengine/RuntimeBuildTypes.h
    src/audioengine/RuntimeBuilder.h
    src/audioengine/RuntimeBuilder.cpp
    src/audioengine/AutoGainPlanner.cpp  # ★ v14.0
    src/audioengine/AudioEngineProcessor.cpp
    src/PsychoacousticDither.cpp
    src/MKLRealTimeSetup.cpp
    # Convolver split implementation (v1)
    src/convolver/ConvolverProcessor.Internal.h
    src/convolver/ConvolverProcessor.Lifecycle.cpp
    src/convolver/ConvolverProcessor.Rebuild.cpp
    src/convolver/ConvolverProcessor.LoaderThread.cpp
    src/convolver/ConvolverProcessor.MixedPhase.cpp
    src/convolver/ConvolverProcessor.ResampleAndFallback.cpp
    src/convolver/ConvolverProcessor.LoadPipeline.cpp
    src/convolver/ConvolverProcessor.Runtime.cpp
    src/convolver/ConvolverProcessor.StateAndUI.cpp
    src/ConvolverState.cpp
    # Legacy monolithic (kept for backward compat, functions guarded by #if)
    src/MKLNonUniformConvolver.cpp
    # P1-1: FFT Backend Concept + ProductionFft + TestFft
    src/FFTBackend.cpp
    src/FFTExecutionContext.cpp
    src/eqprocessor/EQProcessor.Core.cpp
    src/eqprocessor/EQProcessor.Parameters.cpp
    src/eqprocessor/EQProcessor.Coefficients.cpp
    src/eqprocessor/EQProcessor.Processing.cpp
    src/eqprocessor/EQProcessor.ProcessingCache.cpp
    src/eqprocessor/PeakEstimator.cpp
    src/eqprocessor/UpperBoundEstimator.cpp
    src/eqprocessor/EQResponseSampler.cpp
    src/eqprocessor/BandHelper.cpp
    src/EQEditProcessor.cpp
    src/ConvolverControlPanel.cpp
    src/EQControlPanel.cpp
    src/SpectrumAnalyzerComponent.cpp
    src/DeviceSettings.cpp
    src/NoiseShaperLearningComponent.cpp
    src/OutputFilter.cpp
    src/NoiseShaperLearner.cpp
    src/AllpassDesigner.cpp
    src/CmaEsOptimizerDynamic.cpp
    src/AsioBlacklist.h
    src/CustomInputOversampler.h
    src/audioengine/AudioEngineProcessor.h
    src/PsychoacousticDither.h
    src/MKLNonUniformConvolver.h
    src/OutputFilter.h
    src/AllpassDesigner.h
    src/MixedPhaseOptimizationComponent.cpp
    src/IRConverter.cpp
    src/IRAnalyzer.cpp  # ★ v14.0
    src/ProgressiveUpgradeThread.cpp
    src/MixedPhasePersistentCache.cpp
    src/CacheManager.cpp
    src/ConvolverSettingsComponent.cpp
    src/IRDSP.cpp
    # Phase 1: RCU Snapshot Foundation (v13.0)
    src/core/EQParameters.h
    src/core/GlobalSnapshot.h
    src/core/GlobalSnapshot.cpp
    src/core/SnapshotParams.h
    src/core/SnapshotFactory.h
    src/core/SnapshotFactory.cpp
    src/core/SnapshotAssembler.h
    src/core/SnapshotAssembler.cpp
    src/core/EpochDomain.h
    src/core/RCUReader.h
    src/core/SnapshotCoordinator.h
    src/core/SnapshotCoordinator.cpp
    src/core/DeletionQueue.cpp
    # Phase 2: Types
    src/core/Types.h
    # Phase 3: Debounced snapshot worker
    src/core/CommandBuffer.h
    src/core/ThreadAffinityManager.h
    src/core/WorkerThread.h
    src/core/WorkerThread.cpp
    # ★ [P0-1] AVX2 ランタイムチェック
    src/CpuFeatureCheck.cpp
)

target_sources(ConvoPeq PRIVATE ${CONVOPEQ_ALL_SOURCES})

#------------------------------------------------------------
# リンクとコンパイル定義
#------------------------------------------------------------
target_link_libraries(ConvoPeq PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_devices
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_gui_extra
    juce::juce_core
    r8brain
)

# Mark JUCE includes as SYSTEM to suppress clang-tidy warnings.
target_include_directories(ConvoPeq SYSTEM PRIVATE
    # This makes all headers under the JUCE modules directory treated as system headers.
    "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules"
)

target_compile_definitions(ConvoPeq PRIVATE
    CONVOPEQ_STANDALONE_ONLY=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI=1
    $<$<BOOL:${CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS}>:CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1>
    # CONVOPEQ_DIAG_SAMPLE_MASK は DiagnosticsConfig.h で自動設定 (Release:0xFF, Debug:0x3F)。
    # CMake 変数 CONVOPEQ_DIAG_SAMPLE_MASK が明示的に指定された場合のみ上書きする。
    $<$<BOOL:${CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS}>:$<$<BOOL:${CONVOPEQ_DIAG_SAMPLE_MASK}>:CONVOPEQ_DIAG_SAMPLE_MASK=${CONVOPEQ_DIAG_SAMPLE_MASK}>>
    JUCE_DONT_DEFINE_MIN_MAX_MACROS=1
    JUCE_APPLICATION_NAME_STRING="${APP_NAME}"
    JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
    # Windows/ASIO/SIMD settings preserved for functionality
    JUCE_USE_SSE_INTRINSICS=1
    JUCE_USE_SIMD=1
    $<$<PLATFORM_ID:Windows>:
        _UNICODE
        UNICODE
        NOMINMAX
        JUCE_ASIO=1
        _CRT_SECURE_NO_WARNINGS
    >
)

target_include_directories(ConvoPeq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
)

#------------------------------------------------------------
# Intel MKL Configuration (post-target: linking + include paths)
# ★ Detection was moved above (before test target definitions) so CONVOPEQ_HAS_MKL
#   is available for guard clauses in the ISR test block.
#------------------------------------------------------------
# MKLROOT env var search path addition (for find_package fallback)
if(DEFINED ENV{MKLROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}/lib/cmake/mkl")
endif()

# Add MKL include directory as a SYSTEM path to suppress clang-tidy warnings.
if(DEFINED ENV{MKLROOT})
    target_include_directories(ConvoPeq SYSTEM PRIVATE "$ENV{MKLROOT}/include")
    if(TARGET MTNUPCMeasurement)
        target_include_directories(MTNUPCMeasurement SYSTEM PRIVATE "$ENV{MKLROOT}/include")
    endif()
    # ★ ASan-CMAKE-4 修正（2026-07-31）: FFTBackendTests は src/FFTBackend.cpp を含み、
    #   同ファイルが AlignedAllocation.h → mkl.h を include するため MKL include が必要。
    if(TARGET FFTBackendTests)
        target_include_directories(FFTBackendTests SYSTEM PRIVATE "$ENV{MKLROOT}/include")
    endif()
endif()

# Link MKL to ConvoPeq main target (detection already set CONVOPEQ_HAS_MKL above).
# ★ Note: find_package already called in the detection block above. Linking is deferred to here
#   because ConvoPeq target wasn't defined during detection.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(CONVOPEQ_HAS_MKL)
        target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE /Qmkl:sequential)
    if(DEFINED ENV{ONEAPI_ROOT})
        set(_INTEL_COMPILER_ROOT "$ENV{ONEAPI_ROOT}/compiler/latest/lib")
    else()
        set(_INTEL_COMPILER_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    endif()
    if(EXISTS "${_INTEL_COMPILER_ROOT}")
        target_link_directories(ConvoPeq PRIVATE "${_INTEL_COMPILER_ROOT}")
        if(TARGET MTNUPCMeasurement)
            target_link_directories(MTNUPCMeasurement PRIVATE "${_INTEL_COMPILER_ROOT}")
        endif()
        message(STATUS "icx: Intel compiler runtime lib dir = ${_INTEL_COMPILER_ROOT}")
    else()
        message(WARNING "icx: Intel compiler runtime lib dir not found. Link may fail with LNK1104 (libircmt.lib).")
    endif()
    unset(_INTEL_COMPILER_ROOT)
endif()

# ★ work72: JUCE_DSP_USE_INTEL_MKL は MKL が実際に利用可能な場合のみ定義
if(CONVOPEQ_HAS_MKL)
    if(TARGET ConvoPeq)
        # ★ 2026-08-08: icx (IntelLLVM) では /Qmkl:sequential によりコンパイラが自動リンクする。
        #   find_package(MKL) は icx 分岐で呼ばれないため MKL::MKL ターゲットが存在しない。
        #   MSVC (cl) の場合のみ MKL::MKL target_link_libraries を実行する。
        if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
            target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
        endif()
    endif()
    message(STATUS "JUCE_DSP_USE_INTEL_MKL=1 → MKL allocator (mkl_malloc/mkl_free)")
    target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
else()
    message(STATUS "JUCE_DSP_USE_INTEL_MKL undefined → system aligned allocator (work72)")
endif()

#------------------------------------------------------------
# Intel IPP Configuration
#------------------------------------------------------------
if(DEFINED ENV{IPPROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{IPPROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{IPPROOT}/lib/cmake/ipp")
    target_include_directories(ConvoPeq SYSTEM PRIVATE "$ENV{IPPROOT}/include")
    if(TARGET MTNUPCMeasurement)
        target_include_directories(MTNUPCMeasurement SYSTEM PRIVATE "$ENV{IPPROOT}/include")
    endif()
endif()

if(IPP_FOUND)
    message(STATUS "Intel IPP found.")
    target_link_libraries(ConvoPeq PRIVATE IPP::ippcore IPP::ipps)
    if(TARGET MTNUPCMeasurement)
        target_link_libraries(MTNUPCMeasurement PRIVATE IPP::ippcore IPP::ipps)
    endif()
else()
    message(STATUS "Intel IPP not found - skipping (CI build or no IPP SDK).")
endif()

#------------------------------------------------------------
# JUCEヘッダー自動生成 (JuceHeader.h)
#------------------------------------------------------------
juce_generate_juce_header(ConvoPeq)

#------------------------------------------------------------
# PGO (Profile-Guided Optimization) Support
# MSVC専用・Ninja Multi-Config対応・Audio thread制約完全遵守
# プロファイル収集は build\ConvoPeq_artefacts\Release\ConvoPeq.exe 実行時を前提
# .pgd も .pgc と同じ artefacts/Release フォルダ内に配置（ユーザーの指摘対応）
#------------------------------------------------------------
option(CONVOPEQ_PGO_INSTRUMENT "PGO instrumentation build (1st pass: /GENPROFILE)" OFF)
option(CONVOPEQ_PGO_USE        "PGO optimized build (2nd pass: /USEPROFILE)" OFF)
set(CONVOPEQ_PGO_PGD "${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/Release/ConvoPeq.pgd" CACHE FILEPATH "PGO .pgd file path (USE phase)")

# B3: icx で PGO が要求された場合は警告
if(CONVOPEQ_PGO_INSTRUMENT AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    message(WARNING "[PGO] PGO is currently supported only for MSVC. "
                    "icx users: /Qprof-gen and /Qprof-use are available but not yet integrated.")
endif()

if(CONVOPEQ_PGO_INSTRUMENT)
    message(STATUS "[PGO] Instrumentation mode enabled (/GENPROFILE) ── .pgc/.pgdはartefacts/Release内")
    target_compile_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/GL>           # Whole Program Optimization
    )
    target_link_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/LTCG /GENPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
    )
elseif(CONVOPEQ_PGO_USE)
    message(STATUS "[PGO] Optimization mode enabled (/USEPROFILE with ${CONVOPEQ_PGO_PGD})")
    target_compile_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/GL>
    )
    target_link_options(ConvoPeq PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/LTCG /USEPROFILE:PGD=${CONVOPEQ_PGO_PGD}>
    )
else()
    # 通常のRelease（LTCGのみ）
    if("Release" IN_LIST CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "[PGO] Normal Release build with LTCG")
        target_compile_options(ConvoPeq PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/GL>
        )
        target_link_options(ConvoPeq PRIVATE
            $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/LTCG>
        )
    endif()
endif()

#------------------------------------------------------------
# プラットフォーム固有設定
#------------------------------------------------------------
if(WIN32)
    # Windows: COM初期化ライブラリ (WASAPI/ASIO等で必要)
    target_link_libraries(ConvoPeq PRIVATE
        ole32 avrt
    )

    # Visual Studio / icx 用設定
    set(CONVOPEQ_ARTEFACTS_ROOT "${CMAKE_BINARY_DIR}/ConvoPeq_artefacts")
    set_target_properties(ConvoPeq PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        RUNTIME_OUTPUT_DIRECTORY "${CONVOPEQ_ARTEFACTS_ROOT}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CONVOPEQ_ARTEFACTS_ROOT}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CONVOPEQ_ARTEFACTS_ROOT}/Release"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CONVOPEQ_ARTEFACTS_ROOT}/RelWithDebInfo"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CONVOPEQ_ARTEFACTS_ROOT}/MinSizeRel"
    )

    # 著作権情報の設定 (Windows リソース)
    set_target_properties(ConvoPeq PROPERTIES
        JUCE_COMPANY_COPYRIGHT "${APP_COPYRIGHT}"
    )

endif()

#------------------------------------------------------------
# MSVC固有設定
#------------------------------------------------------------
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE
        /utf-8      # ソースコードと実行時の文字セットをUTF-8に設定
        /W4         # 警告レベル4 (高)
        # B6: /wd4100 /wd4189 は JUCE/r8brain 対策。自前コードの警告も隠す可能性あり。
        # 将来的に JUCE ライブラリターゲットと自前コードターゲットを分割して分離を検討。
        /wd4100     # C4100: 参照されないパラメーター (JUCE内部の警告を抑制)
        /wd4189     # C4189: ローカル変数が初期化されましたが、参照されていません (r8brain対策)
        /MP1        # マルチプロセッサコンパイル有効化 (1コアに制限してメモリ使用量を最小化)
        /EHsc       # 標準C++例外処理のみ有効化 (SEH無効 - 構造化例外処理は使用しない)
        /Zm400      # C1060緩和: コンパイラ内部ヒープ上限を400%に設定
        /bigobj     # 大規模オブジェクトファイル対応: .objファイルサイズ制限を解放
    )

    target_compile_definitions(ConvoPeq PRIVATE JUCE_DISABLE_ACCESSIBILITY=1)
    target_compile_definitions(ConvoPeq PRIVATE JUCE_WIN_PER_MONITOR_DPI_AWARE=0)

    # グローバルReleaseフラグ: /GL と /arch:AVX2 は含めない（ConvoPeq ターゲットに個別適用済み）
    # /GL → INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE で自動追加
    # /arch:AVX2 → target_compile_options(ConvoPeq PRIVATE /arch:AVX2) で追加
    set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi /utf-8")
    set(CMAKE_C_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi /utf-8")

    # Debugフラグを明示設定（CMakeデフォルトに /utf-8 /bigobj /Zm400 を追加）
    # ランタイムライブラリ (/MDd /MTd) は JUCE の juce_add_gui_app が制御するため設定しない
    set(CMAKE_CXX_FLAGS_DEBUG "/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc")
    set(CMAKE_C_FLAGS_DEBUG   "/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc")
    # MSVCリンカーフラグは target_link_options で設定（他コンパイラに漏洩しない）
    target_link_options(ConvoPeq PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/DEBUG /LTCG /OPT:REF /OPT:ICF /OPT:LBR>
    )

    # AVX2を全コンフィグで有効化
    target_compile_options(ConvoPeq PRIVATE /arch:AVX2)

    # ReleaseビルドでLTCGを有効化
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

#    # ASanフラグが環境変数やデフォルト設定から混入する場合があるため、強力に削除
#    foreach(flag_var
#        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE)
#        string(REGEX REPLACE "[/-][Ff][Ss]anitize=address" "" ${flag_var} "${${flag_var}}")
#    endforeach()

#    # Visual Studioプロジェクト設定でASanを強制的に無効化
#    set_target_properties(ConvoPeq PROPERTIES
#        VS_GLOBAL_EnableAsan "false"
#    )
    # Debugビルドでインクリメンタルリンクを無効化
    target_link_options(ConvoPeq PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/INCREMENTAL:NO>
    )

    # スタティックCRTリンク（DLL依存排除・ASan除外）
    # Debug→/MTd、Release/RelWithDebInfo/MinSizeRel→/MT
    # 注意: /MTd は /fsanitize=address と非互換のため、Debugタスクから ASan を除去すること
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # ============================================================
    # IntelLLVM (icx) 固有設定
    # ============================================================
    target_compile_options(ConvoPeq PRIVATE
        -Wall -Wextra
        -Wno-unused-parameter
        /EHsc
        /utf-8
        -Wno-unknown-argument
    )

    target_compile_definitions(ConvoPeq PRIVATE JUCE_DISABLE_ACCESSIBILITY=1)
    target_compile_definitions(ConvoPeq PRIVATE JUCE_WIN_PER_MONITOR_DPI_AWARE=0)

    # Release 最適化フラグ: Intel 第4世代Core(Haswell, 2013〜)以降に極限最適化
    # /O3: ループ自動ベクトル化・アグレッシブインライン展開
    # /QxCORE-AVX2: Haswell以降のAVX2+FMA必須、Intel専用コード生成
    # /fp:fast: 高速浮動小数点（icx デフォルト）。明示的なデノーマル対策により
    #   微小信号の消失リスクを排除しつつ、最高の演算性能を実現。
    # /Qipo: プログラム全体最適化(=LTO, リンク時に -fuse-ld=lld 自動追加)
    # /Gy: 関数レベルリンク
    # /Zi: PDBデバッグ情報
    # 注: /fp:precise + /Qimf-arch-consistency:true は icx 2026.0 でメモリ枯渇
    #     (LLVM ERROR: out of memory) を引き起こすため使用しない。
    #     /Oi /Ot は /O3 指定時のデフォルトと同一のため明示指定不要。
    #     /Qdiag-disable は ICX で非サポート（Intel移行ガイド）。
    #     xilink.exe は icx では非推奨（Intel移行ガイド）。
    # 注: 下記の string(REGEX REPLACE "/GL|-GL") は直後の set() で上書きされるため削除済
    # icx Release: CMake規定の-MDを/MTで上書き（静的CRT+静的MKLリンク）
    # ASan 有効時は動的CRT（/MD）が必要なため、/MT を付与しない。
    target_compile_options(ConvoPeq PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${ENABLE_ASAN}>>>:/MT>
    )
    # /Qipo は CMAKE_CXX_FLAGS_RELEASE からは除去し、ConvoPeq ターゲットのみに適用。
    # 注: string(REGEX REPLACE "/GL|-GL") は直後の set() で上書きされるため不要（除去済）
    set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
    set(CMAKE_C_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
    # ConvoPeq ターゲットのみ LTCG(/Qipo) を有効化
    # ASan 有効時は Qipo を無効化（ASan と LTO は非互換）
    target_compile_options(ConvoPeq PRIVATE
        $<$<AND:$<CONFIG:Release>,$<NOT:$<BOOL:${ENABLE_ASAN}>>>:/Qipo>
    )

    # icx リンカーフラグ: /Qipo はリンク時に自動で -fuse-ld=lld を追加
    # /DEBUG, /OPT:REF, /OPT:ICF は target_link_options で設定（MSVCに漏洩しない）
    # ASan 有効時は Qipo 無効
    target_link_options(ConvoPeq PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<CONFIG:Release>,$<NOT:$<BOOL:${ENABLE_ASAN}>>>:/DEBUG /Qipo>
    )

    # AVX2 フラグ（全コンフィグで有効化）
    target_compile_options(ConvoPeq PRIVATE /QxCORE-AVX2)

    # icx -mvzeroupper: AVX→legacy SSE 境界に vzeroupper を自動挿入
    # CXX 翻訳単位のみに限定（Cコンパイラには不要）
    target_compile_options(ConvoPeq PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mvzeroupper>
    )

    # JUCE の NOMINMAX 再定義警告・未使用コマンドライン引数警告を抑制
    target_compile_options(ConvoPeq PRIVATE
        -Wno-unused-command-line-argument
        -Wno-macro-redefined
    )

    # icx では /Qipo でIPOを明示指定するため、INTERPROCEDURAL_OPTIMIZATION は不要
    # （CMakeのINTERPROCEDURAL_OPTIMIZATIONはMSVCで/GLを追加するが、icxでは警告になるため）

    # icx の Debug フラグは MSVC と同一形式で動作
    # /Od(最適化無効), /Zi(PDB) は CMake デフォルトで設定されるため特別な分岐不要

    # B2: icx Debug にも明示的に /MT を指定（デフォルトと同一だが明示性向上）
    # ASan 有効時は動的CRT（/MDd）が必要なため、/MT を付与しない。
    target_compile_options(ConvoPeq PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:${ENABLE_ASAN}>>>:/MT>
    )
    # icx Windows のデフォルトは /MT（静的CRTリンク）で追加設定不要
    # Intel公式ドキュメント(2025.2)で Default=/MT を確認済
endif()

#------------------------------------------------------------
# AddressSanitizer (ASan) オプション（ターゲット定義後でなければならない）
#------------------------------------------------------------
# 重要: MSVC ASan は動的CRT（/MDd）との組み合わせでのみ機能する。
# 静的CRT（/MTd）は LNK2038 エラーを引き起こす。
# 従って ENABLE_ASAN=ON 時は CRT を DLL リンクに切り替える。
option(ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)
if(ENABLE_ASAN)
    # ASan と PGO は同時有効化不可
    if(CONVOPEQ_PGO_INSTRUMENT OR CONVOPEQ_PGO_USE)
        message(FATAL_ERROR "ENABLE_ASAN cannot be combined with PGO. "
                            "Disable PGO before enabling ASan.")
    endif()

    # ASan 時は LTCG / IPO / Qipo を無効化
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)
    # ASan 時は /RTC1 を除去（/RTC1 と ASan は非互換）
    # /RTC1- は MSVC cl では /RTC1 として解釈されるため、Debug 構成の
    # CMAKE_CXX_FLAGS_DEBUG 由来の /RTC1 のみ除去対象とする
    # （RelWithDebInfo は /O2 のため /RTC1 と共存不能、/RTC1 自体を持たない）。
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE $<$<CONFIG:Debug>:/RTC1-> /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
        # MSVC ASan requires dynamic CRT (/MDd) — override static CRT
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
        # icx でも動的 CRT に切り替え（ENABLE_ASAN 時のみ）
        # 注意: 通常時は静的CRT（/MT）のまま。ASan は動的CRT（/MD）が必須。
        # 既存の target_compile_options に /MT が残っていないことを確認すること。
        # set_property だけでは不十分で、compile_options 側の /MT が優先される場合がある。
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

#------------------------------------------------------------
# ENABLE_TSAN: ThreadSanitizer (Clang only, MSVC未対応)
#------------------------------------------------------------
option(ENABLE_TSAN "Enable ThreadSanitizer (Clang only)" OFF)

if(ENABLE_TSAN AND ENABLE_ASAN)
    message(FATAL_ERROR "ENABLE_TSAN and ENABLE_ASAN are mutually exclusive."
                        "Enable only one sanitizer at a time.")
endif()

if(ENABLE_TSAN AND MSVC)
    message(FATAL_ERROR "ENABLE_TSAN requires Clang (MSVC not supported)."
                        "Use Clang on Linux/WSL or enable ASan instead.")
endif()

if(ENABLE_TSAN)
    # TSan は動的 CRT が必須（静的 CRT とは非互換）
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    target_compile_options(ConvoPeq PRIVATE -fsanitize=thread)
    target_link_options(ConvoPeq PRIVATE -fsanitize=thread)

    # TSan と LTCG/IPO は非互換
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
    set_property(TARGET ConvoPeq PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

    message(STATUS "ENABLE_TSAN=ON: ThreadSanitizer enabled (Clang only)")
endif()

#------------------------------------------------------------
# GCC/Clangコンパイラ設定 (Windows/MSVC環境では未使用だが参考として残す)
#------------------------------------------------------------
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(ConvoPeq PRIVATE
#         # 警告
#         -Wall -Wextra -Wpedantic
#
#         # Release最適化
#         $<$<CONFIG:Release>:-O3>            # 最大最適化
#         $<$<CONFIG:Release>:-march=native>  # CPUネイティブ命令
#
#         # Debug情報
#         $<$<CONFIG:Debug>:-g -O0>
#     )
# endif()

#------------------------------------------------------------
# インストール設定（オプション）
#------------------------------------------------------------
install(TARGETS ConvoPeq
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin
    RESOURCE DESTINATION bin
)

if(MSVC)
    # PDBファイル（デバッグ情報）のインストール
    # Releaseビルドでもクラッシュ時の解析に役立ちます
    install(FILES $<TARGET_PDB_FILE:ConvoPeq>
        DESTINATION bin
        OPTIONAL
    )
endif()

#------------------------------------------------------------
# RC1109対策: icx+Ninja環境でcmcldeps.exeがRCコンパイラを
# 正しく実行できない問題。Windows SDK インクルードパスを
# CMAKE_RC_FLAGS に明示設定してrc.exeがwindows.hを見つけられるようにする。
# 注: INCLUDE展開ロジックは上部「Windows RC Compiler Fix for Ninja」と重複するが、
# こちらは cmcldeps 無効化とダミーディレクトリ作成が目的であり、
# INCLUDE展開は上部ブロックが Ninja 専用、こちらは全ジェネレータ対応の補完として残す。
#------------------------------------------------------------
if(WIN32 AND CMAKE_GENERATOR MATCHES "Ninja" AND (NOT MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    # icx+Ninja: cmcldeps による RC 実行を無効化（RC1109回避）
    # IntelLLVM（icx）は MSVC=TRUE となるため、NOT MSVC だけでは不足
    set(CMAKE_NINJA_CMCLDEPS_RC "")

    foreach(config Release Debug RelWithDebInfo MinSizeRel)
        file(MAKE_DIRECTORY
            "${CMAKE_BINARY_DIR}/CMakeFiles/ConvoPeq_rc_lib.dir/${config}/ConvoPeq_artefacts/JuceLibraryCode")
    endforeach()
endif()

#------------------------------------------------------------
# Clang-Tidy Integration (Build-time analysis)
#------------------------------------------------------------
# src/*.cpp のみに clang-tidy を適用（JUCE配下は対象外）
if(CONVOPEQ_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXECUTABLE)
    file(GLOB_RECURSE CONVOPEQ_APP_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    )

    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CLANG_TIDY_CMD
            "${CLANG_TIDY_EXECUTABLE}"
            "-p=${CMAKE_BINARY_DIR}"
            "--extra-arg-before=--driver-mode=cl"
            "--extra-arg=/EHsc"
            "--extra-arg=-fexceptions"
            "--extra-arg=-D_HAS_EXCEPTIONS=1"
            "--header-filter=.*/src/.*"
        )
    else()
        set(CLANG_TIDY_CMD
            "${CLANG_TIDY_EXECUTABLE}"
            "-p=${CMAKE_BINARY_DIR}"
            "--extra-arg=/EHsc"
            "--extra-arg=-fexceptions"
            "--extra-arg=-D_HAS_EXCEPTIONS=1"
            "--header-filter=.*/src/.*"
        )
    endif()

    set_source_files_properties(${CONVOPEQ_APP_SOURCES}
        PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_CMD}"
    )
endif()

# ★ icx/LLVM: 深いテンプレート展開時に OOM (clang frontend signal) が発生する。
#   -j2 でビルド並列度制限 (build.bat) により回避。
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set_source_files_properties(src/audioengine/AudioEngine.Snapshot.cpp
        PROPERTIES COMPILE_FLAGS "/O1"
    )
    set_source_files_properties(src/audioengine/AudioEngine.Learning.cpp
        PROPERTIES COMPILE_FLAGS "/O1"
    )
    set_source_files_properties(src/audioengine/AudioEngine.RebuildDispatch.cpp
        PROPERTIES COMPILE_FLAGS "/O1"
    )
    set_source_files_properties(src/audioengine/AudioEngine.Transition.cpp
        PROPERTIES COMPILE_FLAGS "/O1"
    )
    set_source_files_properties(src/MainApplication.cpp
        PROPERTIES COMPILE_FLAGS "/O1"
    )
endif()
# B4: アプリ本体がテストに依存する逆依存を解消。テスト集約ターゲットが存在する場合はそちらに付け替える
# add_dependencies(ConvoPeq GainStagingContractTests EQProcessorMaxGainTests)

#------------------------------------------------------------
# B4 IntegrationTest Harness: real AudioEngine (no GUI)
#------------------------------------------------------------
# B4 残課題 #1: 実 AudioEngine を起動し publish パイプライン
#   (commitRuntimePublication facade → OwnerChannel → IntentQueue → CoordinatorLoop →
#    executePublish → RuntimeStore swap → onPublishCommitted → receipt) を実スレッドで通す
#   統合テスト。GUI (MainApplication/MainWindow) を除外し、AudioEngine 本体ソースを
#   コンパイルするスタンドアロン exe とする。
if(CONVOPEQ_ENABLE_ISR_TESTS)
    set(CONVOPEQ_HARNESS_EXCLUDED_SOURCES
        src/MainApplication.cpp
        src/MainWindow.cpp
    )
    set(CONVOPEQ_HARNESS_SOURCES)
    foreach(_src ${CONVOPEQ_ALL_SOURCES})
        list(FIND CONVOPEQ_HARNESS_EXCLUDED_SOURCES "${_src}" _excluded_idx)
        if(_excluded_idx EQUAL -1)
            list(APPEND CONVOPEQ_HARNESS_SOURCES "${_src}")
        endif()
    endforeach()

    add_executable(AudioEngineHarness
        src/tests/AudioEngineHarness/AudioEngineHarness.cpp
        src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
        src/tests/AudioEngineHarness/SoakPublishIntegrationTests.cpp
        src/tests/AudioEngineHarness/DeferredFlowIntegrationTests.cpp
        src/tests/AudioEngineHarness/DeferredPublishViewStateMachineTests.cpp
        ${CONVOPEQ_HARNESS_SOURCES}
    )
    target_compile_features(AudioEngineHarness PRIVATE cxx_std_20)
    target_compile_options(AudioEngineHarness PRIVATE /EHsc /utf-8)
    # ★ AVX2: DSPCoreDouble.cpp の fastTanhV256 が __AVX2__/__FMA__ ガード付きのため必須
    #   （ConvoPeq ターゲットと同一フラグ: MSVC → /arch:AVX2, IntelLLVM → /QxCORE-AVX2）
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(AudioEngineHarness PRIVATE /arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(AudioEngineHarness PRIVATE /QxCORE-AVX2)
    endif()
    target_compile_definitions(AudioEngineHarness PRIVATE
        CONVOPEQ_STANDALONE_ONLY=1
        CONVOPEQ_UNIT_TESTS=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME=1
        CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI=1
        $<$<BOOL:${CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS}>:CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1>
        JUCE_DONT_DEFINE_MIN_MAX_MACROS=1
        JUCE_APPLICATION_NAME_STRING="${APP_NAME}"
        JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
        JUCE_USE_SSE_INTRINSICS=1
        JUCE_USE_SIMD=1
        $<$<PLATFORM_ID:Windows>:
            _UNICODE
            UNICODE
            NOMINMAX
            JUCE_ASIO=1
            _CRT_SECURE_NO_WARNINGS
        >
    )
    target_include_directories(AudioEngineHarness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/eqprocessor
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
    )
    target_include_directories(AudioEngineHarness SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules")
    if(DEFINED ENV{MKLROOT})
        target_include_directories(AudioEngineHarness SYSTEM PRIVATE "$ENV{MKLROOT}/include")
    endif()
    if(DEFINED ENV{IPPROOT})
        target_include_directories(AudioEngineHarness SYSTEM PRIVATE "$ENV{IPPROOT}/include")
    endif()
    target_link_libraries(AudioEngineHarness PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_devices
        juce::juce_audio_basics
        juce::juce_dsp
        juce::juce_gui_extra
        juce::juce_gui_basics
        juce::juce_core
        r8brain
    )
    # ★ ASan-CMAKE-4 fix: Guard MKL/IPP links — only link when actually found.
    #   CI runners without Intel oneAPI have CONVOPEQ_HAS_MKL=OFF → skip linking gracefully.
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        if(CONVOPEQ_HAS_MKL)
            target_link_libraries(AudioEngineHarness PRIVATE MKL::MKL)
        else()
            message(STATUS "AudioEngineHarness: CONVOPEQ_HAS_MKL=OFF → skipping MKL link (system aligned allocator)")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(AudioEngineHarness PRIVATE /Qmkl:sequential)
    endif()
    # ★ IPP: ProductionFft::createPlan / NoiseShaperLearner が ippsMalloc_8u・ippsFFT* を参照
    #   （ConvoPeq 本体と同一: 930行 find_package(IPP) → IPP::ippcore IPP::ipps）
    if(IPP_FOUND)
        target_link_libraries(AudioEngineHarness PRIVATE IPP::ippcore IPP::ipps)
    endif()
    # JuceHeader.h 生成順序: ConvoPeq (juce_add_gui_app) が生成するため依存させる
    add_dependencies(AudioEngineHarness ConvoPeq)
    add_test(NAME AudioEngineHarness COMMAND AudioEngineHarness)
endif()

#------------------------------------------------------------
# ENABLE_ASAN: テスト/ハーネス ターゲットにも ASan を適用
#   ConvoPeq 本体は既存の ENABLE_ASAN ブロックで処理済み（動的 CRT /MDd に切替）。
#   テスト/ハーネスは CMake デフォルトの動的 CRT（/MDd）のため /fsanitize=address
#   をそのまま付加できる。/RTC1 は ASan と非互換のため除去する。
#------------------------------------------------------------
if(ENABLE_ASAN)
    set(CONVOPEQ_ASAN_TEST_TARGETS
        ISRRuntimeIdentityTests RuntimePublicationCoordinatorTests
        ISRSemanticValidationTests RetireGraceSemanticsTests
        OwnerChannelTests NormalRetireDSPHandleCompareTests
        RuntimeSemanticSchemaValidationTests ObservePathSingleSourceTests
        OverlapAuthoritySingularTests ShadowCompareContractTests
        CrossfadeExecutorLocalContractTests RuntimeWorldAuthorityProjectionTests
        PartialPublicationRejectTests RebuildAdmissionRegressionTests
        BuildInputSemanticContractTests DeferredDeletionQueueReclaimTests
        MpscBoundedRingTests DSPHandleTableTests
        PriorityIntegrationTests GainStagingContractTests EQProcessorMaxGainTests
        EQAnalysisUnitTests FFTBackendTests EQBoundExcessBenchmark
        AudioEngineHarness ISRSoakTests)
    foreach(tgt IN LISTS CONVOPEQ_ASAN_TEST_TARGETS)
        if(TARGET ${tgt})
            if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
                # /RTC1- は MSVC cl では /RTC1 として解釈される（D8016）。Debug 構成の
                # CMAKE_CXX_FLAGS_DEBUG 由来の /RTC1 のみ除去対象とする。
                # /utf-8: コメントの日本語が Shift-JIS 誤解釈でパース崩れを起こすため
                # （RelWithDebInfo の CMAKE_CXX_FLAGS には /utf-8 が無い）。
                target_compile_options(${tgt} PRIVATE $<$<CONFIG:Debug>:/RTC1-> /utf-8 /fsanitize=address)
                target_link_options(${tgt} PRIVATE /fsanitize=address)
                # MSVC ASan は CRT と一致したランタイム DLL を要求する:
                #   /MDd (Debug)          -> clang_rt.asan_dbg_dynamic-*.dll
                #   /MD  (RelWithDebInfo) -> clang_rt.asan_dynamic-*.dll
                # 展開は release 版 DLL で統一しているため、ASan 実行は RelWithDebInfo
                # (/MD) を標準とする。Debug をビルドする場合も CRT を一致させる。
                set_property(TARGET ${tgt} PROPERTY MSVC_RUNTIME_LIBRARY_DEBUG "MultiThreadedDebugDLL")
                set_property(TARGET ${tgt} PROPERTY MSVC_RUNTIME_LIBRARY_RELWITHDEBINFO "MultiThreadedDLL")
                set_property(TARGET ${tgt} PROPERTY MSVC_RUNTIME_LIBRARY_RELEASE "MultiThreadedDLL")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
                target_compile_options(${tgt} PRIVATE -fsanitize=address)
                target_link_options(${tgt} PRIVATE -fsanitize=address)
            endif()
        endif()
    endforeach()
endif()
