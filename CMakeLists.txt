#============================================================================
# CMakeLists.txt  ── v0.5.0 (JUCE 8.0.12 / VS Code + MSVC + Windows 11)
#
# ビルド環境:
#   - JUCE: 8.0.12
#   - Intel oneAPI
#   - コンパイラ: MSVC 19.44.35222.0 (Visual Studio 2022 17.11以降)
#   - IDE: Visual Studio Code
#   - プラットフォーム: Windows 11
#   - C++標準: C++20
#
# ビルド方法:
#   mkdir build && cd build
#   cmake .. -G "Visual Studio 17 2022" -A x64
#   cmake --build . --config Release
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
# Windows RC Compiler Fix for Ninja
#------------------------------------------------------------
if(WIN32 AND CMAKE_GENERATOR STREQUAL "Ninja")
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

#------------------------------------------------------------
# ソースファイル
#------------------------------------------------------------
target_sources(ConvoPeq PRIVATE
    src/MainApplication.cpp
    src/MainWindow.cpp
    src/AudioEngine.cpp
    src/CustomInputOversampler.cpp
    src/AudioEngineProcessor.cpp
    src/ConvolverProcessor.cpp
    src/MKLNonUniformConvolver.cpp
    src/EQProcessor.cpp
    src/ConvolverControlPanel.cpp
    src/EQControlPanel.cpp
    src/SpectrumAnalyzerComponent.cpp
    src/DeviceSettings.cpp
    src/NoiseShaperLearningComponent.cpp
    src/OutputFilter.cpp
    src/NoiseShaperLearner.cpp
    src/AsioBlacklist.h
    src/CustomInputOversampler.h
    src/AudioEngineProcessor.h
    src/PsychoacousticDither.h
    src/MKLNonUniformConvolver.h
    src/OutputFilter.h
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
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
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
endif()

# スタティックリンクとシーケンシャル実行を指定してDLL依存を排除
# MKL_THREADING=sequential: OpenMPなどのスレッドライブラリに依存しない、シングルスレッド版のMKLをリンクします。
#                           リアルタイムオーディオ処理の安定性確保に不可欠です。
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)

# その上で、LP64（64bit）と Sequential（単一スレッド）のみを要求
find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
message(STATUS "Intel MKL found. Enabling JUCE_DSP_USE_INTEL_MKL=1.")
target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=1)

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
            $<$<CXX_COMPILER_ID:MSVC>:/GL>
        )
        target_link_options(ConvoPeq PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/LTCG>
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
    set_target_properties(ConvoPeq PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )

    # 著作権情報の設定 (Windows リソース)
    set_target_properties(ConvoPeq PROPERTIES
        JUCE_COMPANY_COPYRIGHT "${APP_COPYRIGHT}"
    )

endif()

#------------------------------------------------------------
# MSVC固有設定
#------------------------------------------------------------
if(MSVC)
    target_compile_options(ConvoPeq PRIVATE
        /utf-8      # ソースコードと実行時の文字セットをUTF-8に設定
        /W4         # 警告レベル4 (高)
        /wd4100     # C4100: 参照されないパラメーター (JUCE内部の警告を抑制)
        /wd4189     # C4189: ローカル変数が初期化されましたが、参照されていません (r8brain対策)
        /MP1        # マルチプロセッサコンパイル有効化 (1コアに制限してメモリ使用量を最小化)
        /EHsc       # 標準C++例外処理のみ有効化 (SEH無効 - 構造化例外処理は使用しない)
    )

    target_compile_definitions(ConvoPeq PRIVATE JUCE_DISABLE_ACCESSIBILITY=1)
    target_compile_definitions(ConvoPeq PRIVATE JUCE_WIN_PER_MONITOR_DPI_AWARE=0)

    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF /OPT:LBR")

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
        $<$<CONFIG:Debug>:/INCREMENTAL:NO>
    )
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
# Clang-Tidy Integration (Build-time analysis)
#------------------------------------------------------------
# src/*.cpp のみに clang-tidy を適用（JUCE配下は対象外）
if(CONVOPEQ_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXECUTABLE)
    file(GLOB_RECURSE CONVOPEQ_APP_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    )

    set(CLANG_TIDY_CMD
        "${CLANG_TIDY_EXECUTABLE};
         -p=${CMAKE_BINARY_DIR};
         --extra-arg-before=--driver-mode=cl;
         --extra-arg=/EHsc;
         --extra-arg=-fexceptions;
         --extra-arg=-D_HAS_EXCEPTIONS=1;
         --header-filter=.*/src/.*"
    )

    set_source_files_properties(${CONVOPEQ_APP_SOURCES}
        PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_CMD}"
    )
endif()
