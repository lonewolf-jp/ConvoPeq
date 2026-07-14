#============================================================================
# CMakeLists.txt  ── v0.5.0 (JUCE 8.0.12 / VS Code + MSVC + icx + Windows 11)
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

    add_executable(ISRSemanticValidationTests
        src/tests/ISRSemanticValidationTests.cpp
        src/audioengine/ISRClosure.cpp
        src/audioengine/ISRPayloadTier.cpp
        src/audioengine/ISRRetireRouter.cpp
        src/audioengine/ISRRetire.cpp               # ★ Phase5: coordinator が RetireRuntime を参照
        src/audioengine/ISRRuntimePublicationCoordinator.cpp
    )

    add_executable(RetireGraceSemanticsTests
        src/tests/RetireGraceSemanticsTests.cpp
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
    add_executable(PriorityIntegrationTests
        src/tests/PriorityIntegrationTests.cpp
        src/audioengine/ISRRetire.cpp              # RetireRuntime
    )

    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_link_libraries(RuntimePublicationCoordinatorTests PRIVATE MKL::MKL)
        target_link_libraries(PartialPublicationRejectTests PRIVATE MKL::MKL)
    endif()

    target_compile_features(ISRRuntimeIdentityTests PRIVATE cxx_std_20)
    target_compile_features(RuntimePublicationCoordinatorTests PRIVATE cxx_std_20)
    target_compile_features(ISRSemanticValidationTests PRIVATE cxx_std_20)
    target_compile_features(RetireGraceSemanticsTests PRIVATE cxx_std_20)
    target_compile_features(RuntimeSemanticSchemaValidationTests PRIVATE cxx_std_20)
    target_compile_features(ObservePathSingleSourceTests PRIVATE cxx_std_20)
    target_compile_features(OverlapAuthoritySingularTests PRIVATE cxx_std_20)
    target_compile_features(ShadowCompareContractTests PRIVATE cxx_std_20)
    target_compile_features(CrossfadeExecutorLocalContractTests PRIVATE cxx_std_20)
    target_compile_features(RuntimeWorldAuthorityProjectionTests PRIVATE cxx_std_20)
    target_compile_features(PartialPublicationRejectTests PRIVATE cxx_std_20)
    target_compile_features(RebuildAdmissionRegressionTests PRIVATE cxx_std_20)
    target_compile_features(BuildInputSemanticContractTests PRIVATE cxx_std_20)
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
    )
    target_include_directories(RetireGraceSemanticsTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_BINARY_DIR}/ConvoPeq_artefacts/JuceLibraryCode
        ${CMAKE_CURRENT_SOURCE_DIR}/JUCE/modules
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
    target_include_directories(PriorityIntegrationTests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audioengine"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/core"
    )
    add_test(NAME ISRRuntimeIdentityGenerators COMMAND ISRRuntimeIdentityTests)
    add_test(NAME RuntimePublicationCoordinatorRejects COMMAND RuntimePublicationCoordinatorTests)
    add_test(NAME ISRSemanticValidationRejects COMMAND ISRSemanticValidationTests)
    add_test(NAME RetireGraceSemantics COMMAND RetireGraceSemanticsTests)
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
        target_compile_options(RetireGraceSemanticsTests PRIVATE /utf-8)
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
    endif()

    # テストターゲットに Windows プラットフォーム定義を追加（Debug ビルドでも標準ヘッダーが正しく include されるよう確保）
    if(WIN32)
        foreach(tgt IN ITEMS ISRRuntimeIdentityTests RuntimePublicationCoordinatorTests
                     ISRSemanticValidationTests RetireGraceSemanticsTests
                     RuntimeSemanticSchemaValidationTests ObservePathSingleSourceTests
                     OverlapAuthoritySingularTests ShadowCompareContractTests
                     CrossfadeExecutorLocalContractTests RuntimeWorldAuthorityProjectionTests
                     PartialPublicationRejectTests RebuildAdmissionRegressionTests
                     BuildInputSemanticContractTests PriorityIntegrationTests)
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
        set_target_properties(RetireGraceSemanticsTests PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
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
    target_compile_options(RetireGraceSemanticsTests PRIVATE /EHsc)
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
    if(MSVC)
        target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
        target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
    endif()

    # icx: MKL を使用するテストターゲットに /Qmkl:sequential を追加
    # （MSVC は target_link_libraries で MKL::MKL をリンク、icx はコンパイルオプションで
    #  リンク指示を .obj に埋め込む。ConvoPeq 本体は CMakeLists.txt 上部で既に設定済み）
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
        JUCE_DSP_USE_INTEL_MKL=1
        JUCE_USE_SIMD=1
    )
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_link_libraries(MTNUPCMeasurement PRIVATE MKL::MKL)
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
target_sources(ConvoPeq PRIVATE
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
                src/audioengine/RuntimePublicationValidator.cpp
                src/audioengine/CrossfadeAuthority.cpp
                src/audioengine/PublicationAdmission.cpp
                src/audioengine/PublicationExecutor.cpp
                src/audioengine/RuntimePublicationOrchestrator.cpp
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
    src/eqprocessor/EQProcessor.Core.cpp
    src/eqprocessor/EQProcessor.Parameters.cpp
    src/eqprocessor/EQProcessor.Coefficients.cpp
    src/eqprocessor/EQProcessor.Processing.cpp
    src/eqprocessor/EQProcessor.ProcessingCache.cpp
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
# Intel MKL Configuration
#------------------------------------------------------------
# MKLROOT環境変数が設定されている場合、検索パスに追加して検出率を向上させる
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
endif()

# スタティックリンクとシーケンシャル実行を指定してDLL依存を排除
# MKL_THREADING=sequential: OpenMPなどのスレッドライブラリに依存しない、シングルスレッド版のMKLをリンクします。
#                           リアルタイムオーディオ処理の安定性確保に不可欠です。
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)

# MKL リンク方式はコンパイラに応じて分岐
# MSVC: CMake Config 経由（従来方式）
# icx: /Qmkl:sequential でコンパイラが自動リンク
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
    target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx: /Qmkl:sequential でコンパイラがMKLを自動リンク
    # Windows では static link がデフォルト、/MT と一貫性あり
    # コンパイル時オプションとして指定し、リンカ指令をオブジェクトファイルに埋め込む
    target_compile_options(ConvoPeq PRIVATE /Qmkl:sequential)
    # icx: Intel コンパイラランタイム (libircmt.lib / libmmd.lib) のパスを明示
    # icx がリンカとして呼ばれる際に Intel ランタイムを見つけられるようにする。
    # 典型的なパス: compiler\2026.0\lib\ (oneAPI 2026.0)
    # ONEAPI_ROOT は oneAPI setvars.bat で設定される環境変数
    set(_INTEL_COMPILER_ROOT "")
    if(DEFINED ENV{ONEAPI_ROOT})
        set(_INTEL_COMPILER_ROOT "$ENV{ONEAPI_ROOT}/compiler/latest/lib")
    endif()
    if(NOT _INTEL_COMPILER_ROOT OR NOT EXISTS "${_INTEL_COMPILER_ROOT}")
        # フォールバック: ProgramFiles(x86) から検出
        set(_INTEL_COMPILER_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib")
    endif()
    if(EXISTS "${_INTEL_COMPILER_ROOT}")
        target_link_directories(ConvoPeq PRIVATE "${_INTEL_COMPILER_ROOT}")
        message(STATUS "icx: Intel compiler runtime lib dir = ${_INTEL_COMPILER_ROOT}")
    else()
        message(WARNING "icx: Intel compiler runtime lib dir not found. Link may fail with LNK1104 (libircmt.lib).")
    endif()
    unset(_INTEL_COMPILER_ROOT)
endif()

message(STATUS "Intel MKL found. Enabling JUCE_DSP_USE_INTEL_MKL=1.")
target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=1)

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

set(IPP_LINK static)
set(IPP_THREADING sequential)
find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)
if(IPP_FOUND)
    message(STATUS "Intel IPP found.")
    target_link_libraries(ConvoPeq PRIVATE IPP::ippcore IPP::ipps)
    if(TARGET MTNUPCMeasurement)
        target_link_libraries(MTNUPCMeasurement PRIVATE IPP::ippcore IPP::ipps)
    endif()
else()
    message(STATUS "Intel IPP not found - skipping (CI build or no IPP SDK).")
endif()
# 注意: R8B_IPP=1 は使用しない。
# r8brain の CDSPRealFFT.h が期待する IPP API (IppsFFTSpec_R_64f 等) は
# Intel IPP 2022.3 以降のバージョンと非互換のため、r8brain には内蔵 FFT を使用する。
# ConvoPeq 本体は IPP::ippcore / IPP::ipps を MKL 補完用にリンクする。

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

    # Visual Studio用設定
    set(CONVOPEQ_ARTEFACTS_ROOT "${CMAKE_SOURCE_DIR}/build/ConvoPeq_artefacts")
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
    set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi")
    set(CMAKE_C_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi")

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
    target_compile_options(ConvoPeq PRIVATE
        $<$<CONFIG:Release>:/MT>
    )
    # /Qipo は CMAKE_CXX_FLAGS_RELEASE からは除去し、ConvoPeq ターゲットのみに適用。
    # 注: string(REGEX REPLACE "/GL|-GL") は直後の set() で上書きされるため不要（除去済）
    set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi")
    set(CMAKE_C_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi")
    # ConvoPeq ターゲットのみ LTCG(/Qipo) を有効化
    target_compile_options(ConvoPeq PRIVATE
        $<$<CONFIG:Release>:/Qipo>
    )

    # icx リンカーフラグ: /Qipo はリンク時に自動で -fuse-ld=lld を追加
    # /DEBUG, /OPT:REF, /OPT:ICF は target_link_options で設定（MSVCに漏洩しない）
    target_link_options(ConvoPeq PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:IntelLLVM>,$<CONFIG:Release>>:/DEBUG /Qipo>
    )

    # AVX2 フラグ（全コンフィグで有効化）
    target_compile_options(ConvoPeq PRIVATE /QxCORE-AVX2)

    # JUCE の NOMINMAX 再定義警告・未使用コマンドライン引数警告を抑制
    target_compile_options(ConvoPeq PRIVATE
        -Wno-unused-command-line-argument
        -Wno-macro-redefined
    )

    # icx では /Qipo でIPOを明示指定するため、INTERPROCEDURAL_OPTIMIZATION は不要
    # （CMakeのINTERPROCEDURAL_OPTIMIZATIONはMSVCで/GLを追加するが、icxでは警告になるため）

    # icx の Debug フラグは MSVC と同一形式で動作
    # /Od(最適化無効), /Zi(PDB) は CMake デフォルトで設定されるため特別な分岐不要

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
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
        # MSVC ASan requires dynamic CRT (/MDd) — override static CRT
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
    endif()
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
if(WIN32 AND CMAKE_GENERATOR MATCHES "Ninja" AND NOT MSVC)
    # icx+Ninja: cmcldeps による RC 実行を無効化（RC1109回避）
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
            "${CLANG_TIDY_EXECUTABLE};
             -p=${CMAKE_BINARY_DIR};
             --extra-arg-before=--driver-mode=cl;
             --extra-arg=/EHsc;
             --extra-arg=-fexceptions;
             --extra-arg=-D_HAS_EXCEPTIONS=1;
             --header-filter=.*/src/.*"
        )
    else()
        set(CLANG_TIDY_CMD
            "${CLANG_TIDY_EXECUTABLE};
             -p=${CMAKE_BINARY_DIR};
             --extra-arg=/EHsc;
             --extra-arg=-fexceptions;
             --extra-arg=-D_HAS_EXCEPTIONS=1;
             --header-filter=.*/src/.*"
        )
    endif()

    set_source_files_properties(${CONVOPEQ_APP_SOURCES}
        PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_CMD}"
    )
endif()
