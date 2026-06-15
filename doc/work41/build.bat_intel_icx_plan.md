# build.bat Intel icx/icpx コンパイラ対応 改修計画

## 1. 概要

**目的**: `build.bat` のデフォルトビルドは現状どおり MSVC (`cl`) とし、コマンドラインオプションで Intel oneAPI DPC++/C++ Compiler (`icx`/`icpx`) を選択可能にする。

**関連ファイル**:

| ファイル | 改修対象 | 備考 |
|---|---|---|
| `build.bat` | ✅ 要改修 | コンパイラ選択ロジック・CMake引数の分岐 |
| `CMakeLists.txt` | ✅ 要改修 | MSVC固有フラグのガード + IntelLLVM用フラグ追加 |
| `CMakePresets.json` | △ 推奨 | icx用プリセット追加 |
| `.vscode/tasks.json` | △ 推奨 | icx用タスク追加 |
| `BUILD_GUIDE_WINDOWS.md` | ✅ 要改修 | icxビルド手順を追記 |
| `src/**/*.{cpp,h,hpp}` | △ 一部要確認 | MSVC固有コードの互換性確保 |

---

## 2. 現状分析

### 2.1 build.bat の現在のフロー

```
引数パース (Debug/Release/clean/nopause/pgo-gen/pgo-use)
  ↓
JUCE存在チェック
  ↓
Clean (optional)
  ↓
▼ MSVC環境セットアップ (vcvarsall.bat x64) ─── 必須・唯一の選択肢
  ↓
▼ Intel oneAPI環境セットアップ (setvars.bat intel64) ─── MKLリンクのため必須
  ↓
CMake Configure (-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl)
  ↓
CMake Build (--config %BUILD_CONFIG%)
  ↓
成果物チェック
```

### 2.2 Intel icx/icpx コンパイラの特徴

#### icx と icpx の本質的な違い

| 観点 | `icx` | `icpx` |
|---|---|---|
| **言語** | C 専用ドライバ（← **従来の常識だがWindowsでは異なる**） | C++ 専用ドライバ |
| **Windows実体** | `icx.exe` — **C/C++両方をコンパイル可能**（MSVCの`cl.exe`と同様） | `icpx.exe` — GNU-styleドライバ |
| **Windows推奨度** | ✅ **Intel推奨** | ❌ 非推奨（GNU形式フラグを期待するがCMakeとの相性が悪い） |
| **フラグ形式** | MSVC互換（`/Qx`, `/fp`, `/MT` 等） | GCC互換（`-march`, `-ffp`, `-static` 等） |
| **Linuxでの役割** | Cコンパイラ（`gcc`相当） | C++コンパイラ（`g++`相当） |

**重要なポイント**: Linux では `icx` = C専用、`icpx` = C++専用 ですが、**Windows では `icx.exe` 1つでC/C++の両方のコンパイルが可能**です。Intel公式ドキュメントの「Different C++ Compilers and Drivers」テーブルでも、Windows上のC++コンパイルには `icx.exe` が記載されています。

ConvoPeq は `CMakeLists.txt` で `LANGUAGES CXX C` と宣言しており、C++が主言語です。Windows上では以下の設定がIntel推奨:

```cmake
-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
```

（C用もC++用も同じ `icx.exe` バイナリ。CMakeが適切に言語を区別してドライバを呼び出す）

#### Windows上の icx の特徴

| 項目 | 値 |
|---|---|
| **CMake Compiler ID** | `IntelLLVM`（C/C++両方） |
| **ベース** | Clang/LLVM (Clang 22 front-end for 2026.0) |
| **MSVC互換性** | 高い（`_MSC_VER` 定義、`/Qx`/`/fp`/`/MT` 等のMSVC形式フラグ対応） |
| **対応ジェネレータ** | Ninja のみ |
| **最小CMakeバージョン** | 3.21（PCHなし）/ 3.23（PCHあり） |

### 2.3 CMakeLists.txt のMSVC依存箇所

| 行範囲 | 内容 | 備考 |
|---|---|---|
| L423-L440 | `if(MSVC)` 内: `/utf-8 /W4 /wd4100 /wd4189 /MP1 /EHsc /Zm400 /bigobj` | icxでは不要・別フラグ |
| L444-L448 | `CMAKE_CXX_FLAGS_RELEASE` に `/O2 /Ob2 /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi` | icx用の別フラグ必要 |
| L452 | `target_compile_options(ConvoPeq PRIVATE /arch:AVX2)` | icxは `-march=core-avx2` |
| L488 | `MSVC_RUNTIME_LIBRARY` (static CRT) | icxでは `-static-libstdc++` |
| L498-504 | PGO: `$<CXX_COMPILER_ID:MSVC>` でガード | icx: `-ipo -flto` / 別PGO方式 |
| L96-101 | ASan: `if(MSVC)` でガード | icx用ASanフラグ追加 |
| L670-675 | Clang-tidy: `--driver-mode=cl` | icxでは不要 |
| テスト全体 | `if(MSVC)` で `/utf-8` | icxでは不要 |

### 2.4 ソースコードのMSVC固有パターン

| パターン | 使用ファイル | icx対応 |
|---|---|---|
| `__assume(expr)` | `CustomInputOversampler.cpp` L218-L219, L615 | icx Windowsモードでは `__assume` をサポート（MSVC互換）。ただしClangモードでは `__builtin_assume`。**確認要** |
| `#pragma warning(push/pop/disable)` | `CommandBuffer.h`, `LockFreeRingBuffer.h`, `ConvolverProcessor.h`, `ConvolverState.h` | icx WindowsモードではMSVC互換pragmaをサポート。**ほぼ問題なし** |
| `#pragma comment(lib, "winmm.lib")` | `MainApplication.cpp` L36 | icxでは `#pragma comment(lib)` が未サポート。リンカに直接 `winmm.lib` を渡す必要あり |
| `_MM_SET_FLUSH_ZERO_MODE` / `_MM_SET_DENORMALS_ZERO_MODE` | `MainApplication.cpp`, `MKLRealTimeSetup.cpp`, `NoiseShaperLearner.cpp`, `WorkerThread.cpp` | icx Windowsモードでは `<xmmintrin.h>` / `<pmmintrin.h>` 経由で利用可能。**問題なし** |
| `vmlSetMode(VML_FTZDAZ_ON)` | `MainApplication.cpp` | MKL VML関数。icxでも利用可能。**問題なし** |
| `_putenv_s` | `MKLRealTimeSetup.cpp` | MSVC CRT関数。icxでも利用可能。**問題なし** |
| `_aligned_malloc` / `_aligned_free` | なし（`mkl_malloc`/`mkl_free`を使用） | **問題なし** |
| `__declspec(...)` | アプリコードでは使用なし（r8brain DLLコードのみ） | icx Windowsモードでサポート。**問題なし** |

---

## 3. 改修計画（全4フェーズ）

### フェーズ1: build.bat の改修

#### 3.1.1 引数パースに `icx` 追加（`icpx` はオプション維持）

**基本方針**: Windows では `icx.exe` 1つでC/C++両方をカバーするため、`CMAKE_C_COMPILER` と `CMAKE_CXX_COMPILER` の両方に `icx` を指定する（Intel推奨）。
`icpx` はWindowsではGNUスタイルドライバのため実験的サポートとする。

```bat
REM --- 新規変数 ---
set "COMPILER_MODE=msvc"   "icx"のみ正式対応、"icpx"は実験的

for %%A in (%*) do (
    ...
    if /i "%%~A"=="icx"   set "COMPILER_MODE=icx"
    if /i "%%~A"=="icpx"  set "COMPILER_MODE=icpx"
)
```

#### 3.1.2 コンパイラ環境セットアップの分岐

```bat
if "!COMPILER_MODE!"=="msvc" (
    REM 現行のMSVC環境セットアップ (vcvarsall.bat)
    ...
) else (
    REM Intel icx環境の場合: vcvarsall.bat は不要（icxが自動検出）
    REM ただし oneAPI setvars.bat は icx の発見と MKL のために必要
)
```

**重要な考慮点**:

- **Intel icx on Windows では MSVC の vcvarsall.bat は不要**。icx コンパイラは必要な Windows SDK パスを自力で解決する（または setvars.bat 経由で設定される）。
- **ただし setvars.bat intel64 は icx の検出と MKL 環境変数 (MKLROOT) のために必要**。
- PGO (`pgo-gen` / `pgo-use`) は MSVC モード限定とする（icx では未対応、将来拡張）。

#### 3.1.3 CMake 引数の分岐

```bat
if "!COMPILER_MODE!"=="msvc" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
        %CMAKE_PGO_FLAGS%
) else if "!COMPILER_MODE!"=="icx" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
    REM icx では PGO 未対応（将来拡張）
) else if "!COMPILER_MODE!"=="icpx" (
    REM icpx (GNU-style driver): Windowsでは実験的サポート
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
)
```

> **補足**: ConvoPeq は C++ が主言語だが、**Windows では `icx.exe` 1つでC/C++両方をカバーする**（MSVCの`cl.exe`と同様）。したがって `CMAKE_C_COMPILER=icx` / `CMAKE_CXX_COMPILER=icx` の両方に同じ `icx` を指定するのがIntel推奨。`icpx` はWindowsではGNUスタイルドライバであり、Linuxと異なり明示的に使う必要はない。

#### 3.1.4 成果物パスとビルド検証の調整

icx モードではビルド成果物の拡張子・パスが同じであるため、現行のチェックロジックがそのまま使える。

#### 3.1.5 使用例（改修後）

```cmd
REM デフォルト: MSVC (従来通り)
build.bat Release

REM Intel icx推奨: Windowsではicx.exeがC/C++両方のコンパイラ
build.bat Release icx

REM Intel icpx (GNU-style driver) - Windowsでは実験的サポート
build.bat Debug icpx

REM clean + icx
build.bat Release clean icx
```

---

### フェーズ2: CMakeLists.txt の改修

#### 3.2.1 MSVC 固有フラグのガード強化

```cmake
# --- 現状 ---
if(MSVC)
    target_compile_options(ConvoPeq PRIVATE
        /utf-8
        /W4
        ...
    )

# --- 改修後 ---
if(MSVC)
    target_compile_options(ConvoPeq PRIVATE
        /utf-8
        /W4
        /wd4100
        /wd4189
        /EHsc
        /Zm400
        /bigobj
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE
        -Wall -Wextra
        -Wno-unused-parameter
        -fexceptions
        -finput-charset=utf-8
    )
endif()
```

#### 3.2.2 Release フラグの分岐

```cmake
# --- 現状 ---
set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi")

# --- 改修案 ---
# CMakeLists.txt 冒頭でコンパイラ種別に応じて分岐
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi")
    set(CMAKE_C_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=core-avx2 -fp-model=fast -ipo -ffunction-sections -fdata-sections -g")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG -march=core-avx2 -fp-model=fast -ipo -ffunction-sections -fdata-sections -g")
endif()
```

#### 3.2.3 AVX2 フラグの分岐

```cmake
# --- 現状 ---
target_compile_options(ConvoPeq PRIVATE /arch:AVX2)

# --- 改修案 ---
if(MSVC)
    target_compile_options(ConvoPeq PRIVATE /arch:AVX2)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ConvoPeq PRIVATE -march=core-avx2)
endif()
```

#### 3.2.4 リンカーフラグの分岐

```cmake
# --- 現状 ---
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /OPT:REF /OPT:ICF /OPT:LBR")

# --- 改修案 ---
if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /LTCG /OPT:REF /OPT:ICF /OPT:LBR")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -g -flto")
endif()
```

#### 3.2.5 Static CRT リンクの分岐

```cmake
# --- 現状 ---
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# --- 改修案 ---
if(MSVC)
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx では CMAKE_MSVC_RUNTIME_LIBRARY は機能しない
    # 静的リンクには以下を追加
    target_link_options(ConvoPeq PRIVATE -static-libgcc -static-libstdc++)
endif()
```

#### 3.2.6 PGO のガード

```cmake
# --- 現状 ---
$<$<CXX_COMPILER_ID:MSVC>:/GL>

# --- 改修案 ---
$<$<CXX_COMPILER_ID:MSVC>:/GL>
$<$<CXX_COMPILER_ID:IntelLLVM>:-ipo>
```

PGO の `/GENPROFILE` / `/USEPROFILE` は icx では異なる方式（`-prof-gen` / `-prof-use`）となる。
**提案**: フェーズ2では icx PGO は実装せず、`pgo-gen`/`pgo-use` は MSVC 限定とし、`build.bat` で icx + pgo の組み合わせをエラーにする。

#### 3.2.7 テストのMSVCガード

現状のテスト向け `/utf-8` フラグは `if(MSVC)` で既にガードされているため改修不要。

#### 3.2.8 Debug リンカーフラグ

```cmake
# --- 現状 ---
target_link_options(ConvoPeq PRIVATE
    $<$<CONFIG:Debug>:/INCREMENTAL:NO>)

# --- 改修案（MSVCガード追加）---
target_link_options(ConvoPeq PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/INCREMENTAL:NO>)
```

#### 3.2.9 Clang-tidy ドライバーモード

clang-tidy は icx でも使用可能だが `--driver-mode=cl` は MSVC モード専用。

```cmake
# --- 現状 ---
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE};
     -p=${CMAKE_BINARY_DIR};
     --extra-arg-before=--driver-mode=cl;
     ...")

# --- 改修案 ---
if(MSVC)
    set(CLANG_TIDY_CMD
        "${CLANG_TIDY_EXECUTABLE};
         -p=${CMAKE_BINARY_DIR};
         --extra-arg-before=--driver-mode=cl;
         ...")
else()
    set(CLANG_TIDY_CMD
        "${CLANG_TIDY_EXECUTABLE};
         -p=${CMAKE_BINARY_DIR};
         ...")
endif()
```

#### 3.2.10 ASan のガード

```cmake
# ASan は MSVC / icx で別フラグ
option(ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)
if(ENABLE_ASAN)
    if(MSVC)
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
    endif()
endif()
```

---

### フェーズ3: ソースコードの互換性確保

#### 3.3.1 `#pragma comment(lib)` の置き換え

`MainApplication.cpp` L36:

```cpp
// 現状 (MSVC only):
#pragma comment(lib, "winmm.lib")

// 改修案: CMakeLists.txt で winmm.lib をリンク
// CMakeLists.txt:
//   target_link_libraries(ConvoPeq PRIVATE ... winmm ...)
//
// ソースコードから #pragma comment(lib) を削除し、
// CMakeLists.txt の WIN32 ブロックに winmm を追加
```

`CMakeLists.txt` の既存の WIN32 リンク行を確認:

```cmake
if(WIN32)
    target_link_libraries(ConvoPeq PRIVATE ole32 avrt)
    # → winmm を追加
    target_link_libraries(ConvoPeq PRIVATE ole32 avrt winmm)
endif()
```

#### 3.3.2 `#pragma warning` のガード

icx (Clangベース) では `#pragma warning` がサポートされない可能性がある。
Clang スタイルの `#pragma clang diagnostic` と併記するか、条件付きで切り替える。

```cpp
// 現行パターン（CommandBuffer.h / LockFreeRingBuffer.h 等）:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
// ... alignas 使用コード ...
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// 改修案 1: MSVC 限定のまま（icx Windowsモードは #pragma warning をサポートするため）
// → icx の Windows モード (icx) は MSVC互換pragmaをサポートするため、
//    _MSC_VER が定義されている場合はそのまま動作する。
//    __INTEL_LLVM_COMPILER でも _MSC_VER が定義されることを確認する。

// 改修案 2（安全策）:
#if defined(_MSC_VER) && !defined(__INTEL_LLVM_COMPILER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
```

**推奨**: icx Windowsモードでは `_MSC_VER` が定義されるため、現行コードはそのまま動作する。改修不要だが、icx の `__INTEL_LLVM_COMPILER` と `_MSC_VER` の関係を実機で確認すること。

#### 3.3.3 `__assume` のガード

`CustomInputOversampler.cpp`:

```cpp
// 現状:
__assume(convCount >= 8);
__assume(convCount >= 0);

// 改修案（安全策）:
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
    __assume(convCount >= 8);
    __assume(convCount >= 0);
#else
    // clang/gcc equivalent or no-op
    if (!(convCount >= 8)) { /* hint */ }
#endif
```

**推奨**: icx Windowsモードでは `__assume` をサポートするため、現行コードで動作する。実機確認後に判断。

---

### フェーズ4: ドキュメント・タスク更新

#### 3.4.1 BUILD_GUIDE_WINDOWS.md 更新

```markdown
## 5. Intel icx コンパイラでのビルド

```cmd
REM Intel icx推奨: Windowsではicx.exeがC/C++両方のコンパイラ
build.bat Release icx

REM Intel icpx (GNU-style driver) - Windowsでは実験的サポート
build.bat Release icpx
```

## 6. 手動ビルド（icx 相当）

> **注**: ConvoPeq はC++プロジェクトだが、**Windowsの `icx.exe` はC/C++両方に対応している**。
> Linuxのように `icx`（C用）と `icpx`（C++用）を区別する必要はなく、両方に `icx` を指定するのがIntel推奨。

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
cmake --build build --config Release
```

```

#### 3.4.2 CMakePresets.json 更新 (推奨)

```json
{
    "name": "icx-x64",
    "displayName": "Intel icx x64",
    "generator": "Ninja Multi-Config",
    "binaryDir": "${sourceDir}/build-icx",
    "cacheVariables": {
        "CMAKE_C_COMPILER": "icx",
        "CMAKE_CXX_COMPILER": "icx"
    }
}
```

#### 3.4.3 .vscode/tasks.json 更新 (推奨)

```json
{
    "label": "Release (icx)",
    "type": "shell",
    "command": "call \"C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat\" intel64 && cmake -S \"${workspaceFolder}\" -B \"${workspaceFolder}\\build-icx\" -G \"Ninja Multi-Config\" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx && cmake --build \"${workspaceFolder}\\build-icx\" --config Release",
    "group": "build",
    "problemMatcher": "$msCompile"
}
```

---

## 4. リスクと注意点（Web調査結果反映）

### 4.1 JUCE の IntelLLVM 互換性

JUCE 8.0.12 は IntelLLVM (icx) での公式テスト実績が不明。以下の点に注意:

- JUCE の CMake モジュール (`JUCE/extras/Build/CMake/`) 内で `if(MSVC)` 判定が多用されている可能性
- icx は Windows 上で `_MSC_VER` を定義するため（[StackOverflow 確認済](https://stackoverflow.com/questions/77012074)）、多くの MSVC パスを通過する
- ただし MSVC と完全互換ではないため、JUCE の特定機能でコンパイルエラーが出る可能性あり
- **対策**: まず `build.bat Release icx clean` で試験ビルドを行い、エラー箇所を特定する
- **参考情報**: Melatonin の Pamplejuce テンプレートで Intel IPP と JUCE の CMake 統合事例あり（[JUCE Forum](https://forum.juce.com/t/article-using-ipp-with-juce-cmake/54850)）。ただし Intel コンパイラそのものの検証ではない

### 4.2 icx の Windows ドライバモードとプリプロセッサマクロ（Intel公式情報に基づく）

**出典**: [Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference 2025.2 - Additional Predefined Macros](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/additional-predefined-macros.html)

Intel公式ドキュメント（2025.2, Clang 22ベース。2026.0も同一のマクロ体系）に基づくプリプロセッサマクロ情報:

| マクロ | プラットフォーム | Intel公式説明 | icx (Windows) |
|---|---|---|---|
| `_MSC_VER` | Windows | "The Visual C++ version being used." | ✅ **定義される** |
| `_MSC_FULL_VER` | Windows | "The Visual C++ version being used." | ✅ 定義される |
| `_MSC_EXTENSIONS` | Windows | "Defined when Microsoft extensions are enabled." | ✅ 定義される |
| `__INTEL_LLVM_COMPILER` | Linux Windows | "Version in VVVVMMUU format. e.g. 20230100. Recognized by CMake." | ✅ **公式に定義される** |
| `__VERSION__` | Linux Windows | "Compiler version string." | ✅ 定義される |
| `_WIN64` | Windows | "Defined as 1." | ✅ 定義される |
| `__AVX2__` | Linux Windows | "/QxCORE-AVX2 or higher" | ✅ `/QxCORE-AVX2`指定時 |
| `_MT` | Windows | "Defined as 1 when /MD[d] or /MT[d] specified" | ✅ 指定に応じて定義 |
| `__llvm__` | — | **Intelの公式マクロ一覧に記載なし** | ❌ **定義されない可能性が高い** |

**重要ポイント**:

1. **`_MSC_VER`**: Intel公式で icx on Windows が `_MSC_VER` を定義することを確認。現状の `#ifdef _MSC_VER` ガードは icx でも通過する。
2. **`__INTEL_LLVM_COMPILER`**: Intel公式で定義が保証されている。2026.0でも継続して定義される。2025の一部VS統合モードで未定義になったという報告は、特定のVisual Studioツールセット統合環境下での現象であり、**コマンドラインの `icx` では常に定義される**。
3. **`__llvm__`**: Intel公式のマクロ一覧に記載がない。icxはClangベースだが、`__llvm__` は定義されない可能性が高い。したがって、**`__llvm__` を使ってicxを検出するのは信頼できない**。
4. **推奨検出方法**: CMakeLists.txt内では `CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"` を使用。ソースコード内では、真のMSVCのみを対象としたい場合は `#if defined(_MSC_VER) && !defined(__INTEL_LLVM_COMPILER)` を使用する。icxの検出には `#ifdef __INTEL_LLVM_COMPILER` を使用する（Intel公式保証）。
5. **`__assume`**: icx (Windowsモード) は MSVC 互換として `__assume` をサポートする（Clangベースでありながら MSVC 互換のビルトインを提供）。

| モード | ドライバ | フラグ形式 | `_MSC_VER` | `__INTEL_LLVM_COMPILER` |
|---|---|---|---|---|
| Windows | `icx` | `/Qx`, `/fp` | ✅ Intel公式保証 | ✅ Intel公式保証 |
| GNU | `icpx` | `-march`, `-ffp` | ❌ | ✅ Intel公式保証 |

**推奨**: `icx` (Windowsモード) を優先。`icpx` は実験的サポートとする。
ソースコード内のコンパイラ判定は `__INTEL_LLVM_COMPILER` をベースにする。

### 4.3 ソースコードのプリプロセッサガード戦略（Intel公式情報に基づく）

Intel公式ドキュメント [Additional Predefined Macros](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/additional-predefined-macros.html) に基づき、以下の戦略を推奨:

**基本方針**: `_MSC_VER` は icx でも定義される。真のMSVCとicxを区別するには `__INTEL_LLVM_COMPILER` の有無を使用する（Intel公式で定義が保証されているため）。

| 用途 | 推奨ガード | 根拠 |
|---|---|---|
| MSVC警告抑制(`#pragma warning`) | `#ifdef _MSC_VER` → そのままOK | icxでも解釈される |
| MSVC固有機能(`__assume`) | ガードなしでOK → icx Windowsモードでサポート | icx公式互換 |
| MSVC限定リンカ指令(`#pragma comment(lib)`) | CMakeの `target_link_libraries` に移管 | icxで未サポートの可能性 |
| **真のMSVCのみ**通過させたい | `#if defined(_MSC_VER) && !defined(__INTEL_LLVM_COMPILER)` | Intel公式保証の識別子 |
| **icx (IntelLLVM) のみ**通過させたい | `#ifdef __INTEL_LLVM_COMPILER` | Intel公式保証 |
| icxまたはMSVCの両方 | `#ifdef _MSC_VER` | icxもこれを定義 |

**`__llvm__` は使用しないこと**: Intel公式のマクロ一覧に記載がなく、icxで定義される保証がない。

### 4.4 PGO 非対応

icx にも PGO 機能は存在する (`-prof-gen` / `-prof-use`) が、MSVC と方式・フラグが異なる。フェーズ2では icx PGO は実装せず、MSVC 限定とする。

### 4.5 静的CRTリンク（Intel公式情報に基づく）

**出典**: [Intel® oneAPI DPC++/C++ Compiler Developer Guide - `/MT` option](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/mt.html)

Intel公式ドキュメントに基づく重要な事実:

| 項目 | Intel公式の記述 |
|---|---|
| `/MT` | "Tells the linker to search for unresolved references in a multithreaded, **static** runtime library." |
| **デフォルト値** | **`/MT`**（静的CRTリンクがデフォルト） |
| 補足 | `/MD` を指定すると動的リンクになる。`-fsycl` は自動で `/MD` を設定する |

**結論**: **Intel icx (Windowsモード) はデフォルトで静的CRTリンク (`/MT`) を使用する**。これは現在の MSVC ビルド（`MSVC_RUNTIME_LIBRARY` で明示的に `/MT` を指定）と同じ動作である。

したがって:

- icx 向けに特別な静的CRTリンクの設定は**不要**
- `MSVC_RUNTIME_LIBRARY` プロパティは MSVC 固有だが、icx がデフォルトで `/MT` を使用するため問題にならない
- `-fsycl` (SYCL/DPC++モード) を使用する場合のみ `/MD` が強制されるが、本プロジェクトは SYCL を使用しないため問題なし
- **実機確認推奨だが、理論上は追加設定不要**

> **注意**: 上記は `icx` (Windowsモード/MSVC互換ドライバ) の場合。`icpx` (GNUモード) では動作が異なる可能性がある。

### 4.6 ビルドディレクトリの分離

icx と MSVC は同じ build ディレクトリを共有できない（CMakeCache.txt のコンパイラ設定が競合する）。
**推奨**: icx モードでは自動的に `build-icx` ディレクトリを使用する。

---

## 5. 実装優先順位

```
P0 [必須] build.bat: 引数パース + CMake分岐
P0 [必須] CMakeLists.txt: MSVCガード + IntelLLVMフラグ追加
P1 [推奨] MainApplication.cpp: #pragma comment(lib) → CMake移管
P1 [推奨] BUILD_GUIDE_WINDOWS.md 更新
P2 [任意] CMakePresets.json: icxプリセット追加
P2 [任意] .vscode/tasks.json: icxタスク追加
P3 [確認] ソースコード互換性確認（__assume / #pragma warning）
```

---

## 6. 検証手順

1. **MSVC リグレッションテスト**: 改修後も MSVC ビルドが正常に動作することを確認

   ```cmd
   build.bat Release clean
   build.bat Debug clean
   ```

2. **icx 試験ビルド**:

   ```cmd
   build.bat Release icx clean
   ```

3. **実行確認**: icx ビルドのバイナリが正常起動し、オーディオ処理が動作することを確認

4. **JUCE ヘッダー互換性**: JUCE 8.0.12 の JuceLibraryCode 生成に IntelLLVM 特有の問題がないか確認

5. **MKL リンク確認**: icx ビルドで MKL が正しくリンクされていることを確認

   ```cmd
   dumpbin /imports build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe | findstr mkl
   ```

6. **ASan 試験 (Optional)**:

   ```cmd
   build.bat Debug icx clean   # + ASan オプション未実装のため手動で cmake に -DENABLE_ASAN=ON を追加
   ```
