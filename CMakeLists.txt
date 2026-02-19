#============================================================================
# CMakeLists.txt  ── v0.1 (JUCE 8.0.12 / VS Code + MSVC + Windows 11)
#
# ビルド環境:
#   - JUCE: 8.0.12
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
    LANGUAGES CXX)

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
# MSVC最適化オプション
#------------------------------------------------------------
if(MSVC)
    add_compile_options(
        /utf-8      # ソースコードと実行時の文字セットをUTF-8に設定
        /W4         # 警告レベル4 (高)
        /wd4100     # C4100: 参照されないパラメーター (JUCE内部の警告を抑制)
        /wd4189     # C4189: ローカル変数が初期化されましたが、参照されていません (r8brain対策)
        /MP         # マルチプロセッサコンパイル有効化
        /EHsc       # 標準C++例外処理のみ有効化 (SEH無効 - 構造化例外処理は使用しない)
    )

    # グローバルな安定性設定 (JUCEモジュール全体に適用)
    add_compile_definitions(JUCE_DISABLE_ACCESSIBILITY=1)
    add_compile_definitions(JUCE_WIN_PER_MONITOR_DPI_AWARE=0)
    # JUCE_DIRECT2D=1 はJUCE 8のデフォルトのため、明示的な設定は不要

    # Release最適化
    # /O2:        最大速度最適化
    # /Ob3:       積極的なインライン展開
    # /GL:        プログラム全体の最適化 (LTCG)
    # /arch:AVX2: AVX2命令セット使用 (モダンCPU向け)
    # /fp:fast:   高速浮動小数点演算 (厳密性より速度優先)
    # /Gw:        グローバルデータの最適化 (バイナリサイズ削減)
    # /Gy:        関数レベルリンク有効化 (未使用関数の削除を促進)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy")

    # リンカー最適化
    # /LTCG:      リンク時コード生成
    # /OPT:REF:   参照されていない関数とデータを削除
    # /OPT:ICF:   同一のCOMDATを統合 (コードサイズ削減)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG /OPT:REF /OPT:ICF /OPT:LBR")
endif()

#------------------------------------------------------------
# JUCEを追加
# プロジェクトルートに JUCE/ ディレクトリが必要
#------------------------------------------------------------
add_subdirectory(JUCE)

#------------------------------------------------------------
# FFTConvolver ライブラリの追加
#------------------------------------------------------------
add_library(FFTConvolver STATIC
    FFTConvolver/FFTConvolver.cpp
    FFTConvolver/AudioFFT.cpp
    FFTConvolver/Utilities.cpp
)

target_include_directories(FFTConvolver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/FFTConvolver
)

# ✅ double精度を有効化
target_compile_definitions(FFTConvolver PUBLIC
    FFTCONVOLVER_USE_DOUBLE
)

target_compile_features(FFTConvolver PUBLIC cxx_std_17)

#------------------------------------------------------------
# r8brain-free-src ライブラリの追加
#------------------------------------------------------------
add_library(r8brain INTERFACE)

target_include_directories(r8brain INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/r8brain-free-src
)

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
    src/ConvolverProcessor.cpp
    src/EQProcessor.cpp
    src/ConvolverControlPanel.cpp
    src/EQControlPanel.cpp
    src/SpectrumAnalyzerComponent.cpp
    src/DeviceSettings.cpp
    src/AsioBlacklist.h
    src/PsychoacousticDither.h
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
    FFTConvolver
    r8brain
)

target_compile_definitions(ConvoPeq PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DONT_DEFINE_MIN_MAX_MACROS=1
    JUCE_APPLICATION_NAME_STRING="${APP_NAME}"
    JUCE_APPLICATION_VERSION_STRING="${PROJECT_VERSION}"
    JUCE_STANDALONE_APPLICATION=1
    FFTCONVOLVER_USE_DOUBLE=1
    # Windows/ASIO/SIMD settings preserved for functionality
    JUCE_USE_SSE_INTRINSICS=1
    JUCE_USE_SIMD=1
    $<$<PLATFORM_ID:Windows>:
        _UNICODE
        UNICODE
        NOMINMAX
        JUCE_ASIO=1
    >
)

# ReleaseでLTCG有効化（MSVC用）
if(MSVC)
    target_compile_options(ConvoPeq PRIVATE /GL)
    target_link_options(ConvoPeq PRIVATE /LTCG)
endif()

#------------------------------------------------------------
# Intel MKL Configuration
#------------------------------------------------------------
# MKLROOT環境変数が設定されている場合、検索パスに追加して検出率を向上させる
if(DEFINED ENV{MKLROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MKLROOT}/lib/cmake/mkl")
endif()

# スタティックリンクとシーケンシャル実行を指定してDLL依存を排除
set(MKL_LINK static)
set(MKL_THREADING sequential)

find_package(MKL CONFIG QUIET)

if(MKL_FOUND)
    message(STATUS "Intel MKL found. Enabling JUCE_DSP_USE_INTEL_MKL=1.")
    target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
    target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=1)
else()
    message(STATUS "Intel MKL not found. Using JUCE internal FFT (JUCE_DSP_USE_INTEL_MKL=0).")
    target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=0)
endif()

#------------------------------------------------------------
# JUCEヘッダー自動生成 (JuceHeader.h)
#------------------------------------------------------------
juce_generate_juce_header(ConvoPeq)

#------------------------------------------------------------
# プラットフォーム固有設定
#------------------------------------------------------------
if(WIN32)
    # Windows: COM初期化ライブラリ (WASAPI/ASIO等で必要)
    target_link_libraries(ConvoPeq PRIVATE
        ole32
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
# MSVCコンパイラ設定
#------------------------------------------------------------
if(MSVC)
    # ASanフラグが環境変数やデフォルト設定から混入する場合があるため、強力に削除
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE)
        string(REGEX REPLACE "[/-][Ff][Ss]anitize=address" "" ${flag_var} "${${flag_var}}")
    endforeach()

    # Visual Studioプロジェクト設定でASanを強制的に無効化
    set_target_properties(ConvoPeq PROPERTIES
        VS_GLOBAL_EnableAsan "false"
    )

    target_link_options(ConvoPeq PRIVATE
        # Debugビルドでインクリメンタルリンクを無効化
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
