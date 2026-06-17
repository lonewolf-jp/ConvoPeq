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
| **ベース** | Clang/LLVM (Clang 22 front-end, 2026.0リリースノートで確認) |
| **MSVC互換性** | 高い（`_MSC_VER` 定義、`/Qx`/`/fp`/`/MT` 等のMSVC形式フラグ対応） |
| **対応ジェネレータ** | Ninja Multi-Config（推奨）、Visual Studio ジェネレータも対応可能（VS2022/2025用IDE拡張あり） |
| **最小CMakeバージョン** | 3.15（基本）/ 3.21（PCHなし）/ 3.23（PCHあり） — 現行の 3.22 で問題なし |

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
| `__assume(expr)` | `CustomInputOversampler.cpp` L218-L219, L615 | icx Windowsモードでは `__assume` をサポート（MSVC互換）。Intel公式の互換pragma一覧で確認済。**変更不要** |
| `#pragma warning(push/pop/disable)` | `CommandBuffer.h`, `LockFreeRingBuffer.h`, `ConvolverProcessor.h`, `ConvolverState.h` | icx WindowsモードではMSVC互換pragmaとして `warning` をサポート。Intel公式の互換pragma一覧で確認済。**変更不要** |
| `#pragma comment(lib, "winmm.lib")` | `MainApplication.cpp` L36 | icx WindowsモードではMSVC互換pragmaとして `comment` をサポート。Intel公式の互換pragma一覧で確認済。**変更不要**（icpx(GNU)のみ注意） |
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

**ビルドディレクトリの分離（必須）**: MSVC と icx では CMake キャッシュが互換性を持たない。同一ディレクトリでコンパイラを切り替えると「ヘテロジニアスな破壊」（一部オブジェクトが MSVC、一部が icx でコンパイルされた状態でのリンク）が発生するリスクがある。そのため、**コンパイラごとにビルドディレクトリを強制分離する**。

```bat
REM --- 新規変数 ---
set "COMPILER_MODE=msvc"   "icx"のみ正式対応、"icpx"は実験的
set "BUILD_ROOT=build"     REM デフォルト: MSVC用

for %%A in (%*) do (
    ...
    if /i "%%~A"=="icx"   set "COMPILER_MODE=icx"
    if /i "%%~A"=="icpx"  set "COMPILER_MODE=icpx"
)

REM コンパイラに応じてビルドディレクトリを分離（CMakeキャッシュ衝突防止）
if "!COMPILER_MODE!"=="msvc" (
    set "BUILD_ROOT=build"
) else (
    set "BUILD_ROOT=build-icx"
)
set "BUILD_DIR=%BUILD_ROOT%"
```

**PGO と icx の排他チェック**: icx モードでは PGO は未対応のため、組み合わせ指定時にエラーにする。

```bat
if not "!COMPILER_MODE!"=="msvc" (
    if not "!PGO_MODE!"=="normal" (
        echo [ERROR] PGO is only supported with MSVC compiler.
        echo [ERROR] icx/icpx PGO support is planned for a future phase.
        call :maybe_pause
        exit /b 1
    )
)
```

#### 3.1.2 コンパイラ環境セットアップの分岐

```bat
if "!COMPILER_MODE!"=="msvc" (
    REM 現行のMSVC環境セットアップ (vcvarsall.bat)
    ...
    REM setvars.bat は icx/icpx に先立ってロード不要だが、
    REM MKLリンクのためには必要（後段で共通ロード）
) else (
    REM Intel icx環境の場合: vcvarsall.bat は不要（icxが自動検出）
    REM oneAPI setvars.bat は以下2点のために必要:
    REM   1) icx/icpx コンパイラの検出と環境変数設定
    REM   2) MKL 環境変数 (MKLROOT) の設定
)
```

**重要な考慮点**:

- **Intel icx on Windows では MSVC の vcvarsall.bat は不要**。icx コンパイラは必要な Windows SDK パスを自力で解決する（または setvars.bat 経由で設定される）。
- **ただし setvars.bat intel64 は icx の検出と MKL 環境変数 (MKLROOT) のために必要**。MSVC モードとは異なり vcvarsall.bat をスキップしても、setvars.bat は必ず実行する。
- PGO (`pgo-gen` / `pgo-use`) は MSVC モード限定とする（icx では未対応、将来拡張）。3.1.1 の排他チェックで未然防止する。

**コンパイラ混在検出の防御的チェック（追加）**:
MSVC で構築済みの `build` ディレクトリが存在する状態で icx モードを実行すると、`build-icx` ディレクトリは別なので問題ない。しかし念のため、既存の `CMakeCache.txt` に前回と異なるコンパイラ ID が記録されていないかを確認するチェックを CMake configure 前に追加するのは有用である。ただし `build.bat` では実装が複雑になりすぎるため、以下の対応とする：

```bat
REM コンパイラ変更時の安全策:
REM build-icx ディレクトリが既に存在し、かつ MSVC のキャッシュを含む場合は警告
if "!COMPILER_MODE!"=="icx" (
    if exist "%BUILD_DIR%\CMakeCache.txt" (
        findstr /C:"CMAKE_CXX_COMPILER:FILEPATH=cl" "%BUILD_DIR%\CMakeCache.txt" >nul 2>&1
        if not errorlevel 1 (
            echo [WARN] Existing MSVC cache detected in %BUILD_DIR%.
            echo [WARN] Recommend using 'clean' flag to avoid cache conflicts.
            echo [WARN] Proceeding anyway (CMake will auto-regenerate).
        )
    )
)
```

#### 3.1.3 CMake 引数の分岐

```bat
REM BUILD_DIR は 3.1.1 でコンパイラに応じて分離済み
REM   MSVC → build,  icx/icpx → build-icx

if "!COMPILER_MODE!"=="msvc" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
        %CMAKE_PGO_FLAGS%
) else if "!COMPILER_MODE!"=="icx" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" ^
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
    REM icx では PGO 未対応（3.1.1 の排他チェック済み）
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
# ターゲット: Intel 第4世代Core(Haswell, 2013〜)以降。AVX2をベースラインとする
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi")
    set(CMAKE_C_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy /Zi")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Intel icx 極限最適化: Haswell(CORE-AVX2, 2013〜)以降をターゲット
    # /O3: 浮動小数点演算ループに最適（Fusion/Block-Unroll-and-Jam有効）
    # /QxCORE-AVX2: Haswell〜のAVX2命令セット
    # /fp:fast: 最大演算速度
    # /Qipo: プログラム全体最適化（=LTO, リンク時に -fuse-ld=lld 自動追加）
    # /Gy: 関数レベルリンク
    # /Zi: PDBデバッグ情報
    set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Qipo /Gy /Zi")
    set(CMAKE_C_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Qipo /Gy /Zi")
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
    # icx Windows: -march はLinux専用。Windowsでは /QxCORE-AVX2 または /arch:AVX2 を使用
    target_compile_options(ConvoPeq PRIVATE /QxCORE-AVX2)
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
    # /Qipo はリンク時に自動で -fuse-ld=lld を追加
    # /OPT:REF, /OPT:ICF: 未参照関数・重複COMDATの削除
    # /LBR: リンカーベーシックブロック再配置（MSVC専用、icxでは不要）
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /Qipo /OPT:REF /OPT:ICF")
endif()
```

#### 3.2.5 Static CRT リンクの分岐

```cmake
# --- 現状 ---
set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# --- 改修案 ---
if(MSVC)
    # CMAKE_MSVC_RUNTIME_LIBRARY を明示設定（デフォルト依存による環境差異を防止）
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx Windows のデフォルトは /MT（静的CRTリンク）
    # Intel公式ドキュメント(2025.2)で Default=/MT を確認済
    # https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/mt.html
    # 特別なフラグは不要。ただしリンク時に MSVC の libcmt.lib が
    # 自動選択されるよう、MSVC ツールチェーンが利用可能であること（setvars.bat 経由）
endif()
```

> **補足**: `CMAKE_MSVC_RUNTIME_LIBRARY` 変数は CMake 3.15 で導入されたポリシー制御用変数で、`MSVC_RUNTIME_LIBRARY` ターゲットプロパティのデフォルト値を設定する。これを明示することで、環境変数やツールチェーンファイルからの暗黙的な設定に依存せず、**MSVC 間で CRT リンクの一貫性を確保**できる。IntelLLVM ではこの変数が効かないが、icx のデフォルト `/MT` により静的 CRT リンクが自動適用されるため問題ない。

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

#### 3.2.11 最適化戦略：Intel 第4世代 Core（Haswell, 2013〜）以降に極限最適化

**ターゲット定義**: Intel Core 第4世代 (Haswell) 以降（2013年以降発売の全Intel CPU）。Haswell で導入された以下の機能をベースラインとする：

| 機能 | Haswell以降の対応 |
|---|---|
| AVX2 (256bit SIMD) | ✅ Haswell (2013) 以降すべて |
| FMA (Fused Multiply-Add) | ✅ Haswell (2013) 以降すべて |
| BMI1/BMI2 (ビット操作拡張) | ✅ Haswell (2013) 以降すべて |
| MOVBE | ✅ Haswell (2013) 以降すべて |

**icx 最適化フラグの根拠**:

| フラグ | Intel公式の説明 | 適用理由 |
|---|---|---|
| `/O3` | "O2 optimizations + aggressive loop transformations: Fusion, Block-Unroll-and-Jam, collapsing IF" | オーディオ処理は浮動小数点ループが支配的。`/O3` は Intel 公式が「浮動小数点演算を多用するループ処理」に推奨。 |
| `/QxCORE-AVX2` | "May generate AVX2, AVX, SSE4.2... for Intel processors" | Haswell以降のAVX2命令セットを最大活用。CPUチェックにより非対応CPUでの実行を防止。 |
| `/fp:fast` | "Enables more aggressive optimizations for floating-point" | オーディオ処理では完全なIEEE準拠より速度優先が許容される。 |
| `/Qipo` | "Enables whole program link time optimization" | 全翻訳単位を横断したIPOにより、関数のインライン展開・デッドコード除去を最大化。 |
| `/Gy` | "Enables function-level linking" | COMDATによる未参照関数のリンク時除去を可能に。 |
| `/Qmkl:sequential` | "Tells the compiler to link using the sequential libraries in Intel MKL" | MKL をシングルスレッド（ジッターなし）で自動リンク。CPUチェックにより AVX2 最適化カーネルが自動選択される。static link がデフォルト。 |

**MSVC 最適化フラグ**（現状維持、`/O2` がMSVCの最大一般最適化レベル）:

| フラグ | 説明 |
|---|---|
| `/O2 /Ob2` | 最大速度最適化 + 自動インライン展開 |
| `/arch:AVX2` | Haswell以降のAVX2命令セット（CPUチェックなし） |
| `/fp:fast` | 高速浮動小数点演算 |
| `/GL /LTCG` | プログラム全体最適化 + リンク時コード生成 |
| `/Gw /Gy` | グローバルデータ最適化 + 関数レベルリンク |

**注意**: icx の `/QxCORE-AVX2` は実行時に Intel CPU チェックを行い、非対応CPUでは致命的エラーとなる。これは意図的な設計（Intel CPU上での最大性能を引き出すための制約）。AMD CPU での実行が必要な場合は `/arch:AVX2` に変更すること。

#### 3.2.12 MKL リンク戦略：`/Qmkl:sequential` の導入（icx 専用）

**Intel公式ドキュメント(2025.2)の記述**: `/Qmkl:sequential` は icx コンパイラに MKL の sequential（シングルスレッド）ライブラリを自動リンクさせるオプション。Windows では static link がデフォルト。<https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/mkl.html>

**提案の妥当性評価**: この提案は以下の理由から **有用かつ採用すべき**。

| 観点 | 現状（MSVC方式） | `/Qmkl:sequential` 採用後（icx方式） |
|---|---|---|
| MKLリンク | `find_package(MKL)` → `target_link_libraries(PRIVATE MKL::MKL)` | コンパイラが自動でリンク。CMakeのMKL Config不要 |
| スレッド制御 | `MKL_THREADING=sequential` で明示指定 | `:sequential` 指定でシングルスレッド固定（ジッターなし） |
| アーキテクチャ最適化 | 別途 `arch:AVX2` 指定が必要 | `/QxCORE-AVX2` と連動して AVX2 最適化カーネルが自動選択 |
| CMake複雑度 | MKLROOT検出 + find_package + コンポーネント指定が必要 | コンパイルオプション1行で完了 |

**改修案（CMakeLists.txt）**:

```cmake
# --- 現状（全コンパイラ共通） ---
find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
target_link_libraries(ConvoPeq PRIVATE MKL::MKL)

# --- 改修案 ---
if(MSVC)
    # MSVC: 従来の CMake Config 経由の MKL リンク
    find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
    target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # icx: /Qmkl:sequential でコンパイラが自動リンク
    # Windows では static link がデフォルト（/MT と一貫性あり）
    # 注意: /Qmkl:sequential はコンパイル時オプションであり、
    # リンカ指令をオブジェクトファイルに埋め込む
    target_compile_options(ConvoPeq PRIVATE /Qmkl:sequential)
endif()
```

**注意点**:

1. **MKL インクルードパス**: `/Qmkl` はリンクのみを処理する。MKL ヘッダー（`mkl.h` 等）のインクルードパスは従来通り `MKLROOT` 経由で設定する。`setvars.bat` が `MKLROOT` を設定するため、現行の `target_include_directories(ConvoPeq SYSTEM PRIVATE "$ENV{MKLROOT}/include")` は変更不要。

2. **IPP との併用**: `/Qmkl:sequential` は MKL 専用。IPP（`find_package(IPP)`）は MSVC/icx 共通で別途管理。

3. **MSVC 非互換**: `/Qmkl:sequential` は icx 専用。MSVC では従来の `find_package(MKL)` を維持。

4. **CPU チェック**: `/Qx` 系オプションと同様、`/Qmkl` も Intel CPU チェックが行われる。AMD CPU では MKL リンクが行われない可能性がある。

---

### フェーズ3: ソースコードの互換性確保

#### 3.3.1 `#pragma comment(lib)` の互換性

`MainApplication.cpp` L36:

```cpp
#pragma comment(lib, "winmm.lib")
```

**Intel公式ドキュメント(2025.2)による確認**: `#pragma comment` は Intel icx の「Pragmas Compatible with the Microsoft* Compiler」一覧に含まれている。<https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/pragmas-compatible-with-other-compilers.html>

したがって、**icx Windows モードでは `#pragma comment(lib, "winmm.lib")` はそのまま動作する**。MSVCからの移行に特別な対応は不要。

ただし、将来的な移植性やicpx(GNUモード)対応を考慮する場合は、CMakeLists.txtのWIN32ブロックでの明示リンクに移行することも選択肢として有効：

```cmake
if(WIN32)
    target_link_libraries(ConvoPeq PRIVATE ole32 avrt)
    # → winmm を追加（#pragma comment(lib) と同等）
    target_link_libraries(ConvoPeq PRIVATE ole32 avrt winmm)
endif()
```

#### 3.3.2 `#pragma warning` の互換性

icx (Clangベース) では `#pragma warning` がサポートされない可能性があると懸念されていたが、**Intel公式ドキュメント(2025.2)により MSVC 互換pragmaとして正式にサポートされている**ことが確認された。

出典: <https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/pragmas-compatible-with-other-compilers.html>
「Pragmas Compatible with the Microsoft* Compiler」一覧に `warning` が記載。

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
```

**結論**: icx Windowsモードでは `_MSC_VER` が定義され、かつ `#pragma warning` がMSVC互換としてサポートされるため、**現行コードは一切の変更不要でそのまま動作する**。

#### 3.3.3 `__assume` の互換性

`CustomInputOversampler.cpp` で使用されている `__assume`:

```cpp
__assume(convCount >= 8);
__assume(convCount >= 0);
```

**Intel公式ドキュメント(2025.2)による確認**: icx (Windowsモード) は MSVC 互換モードで動作し、`__assume` を含む MSVC 互換のビルトインをサポートする。Intel公式のマクロ一覧においても `_MSC_VER` が定義されることが確認されており、`__assume` は MSVC 互換の一部として機能する。

**結論**: icx Windowsモードでは `__assume` はそのまま動作する。改修不要。`icpx` (GNUモード) でのみ `__builtin_assume` への置き換えが必要となるが、本計画では icpx は実験的サポートのため、必要な場合のみ対応する。

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

JUCE 8.0.12 は IntelLLVM (icx) での公式テスト実績が不明だが、JUCE コミュニティでは Intel コンパイラ（ICC Classic → icx）での Windows ビルド実績がある（JUCE Forum 2019年以前から報告あり）。以下の点に注意:

- JUCE の CMake モジュール (`JUCE/extras/Build/CMake/`) 内で `if(MSVC)` 判定が多用されている可能性
- icx は Windows 上で `_MSC_VER` を定義するため（[StackOverflow 確認済](https://stackoverflow.com/questions/77012074)）、多くの MSVC パスを通過する
- icx は LLVM/Clang 22 ベースであり、MSVC 互換性が極めて高い。旧来の ICC Classic より互換性が向上している（新しいLLVMベースの icx は Clang のドロップイン代替としても動作）
- ただし MSVC と完全互換ではないため、JUCE の特定機能でコンパイルエラーが出る可能性あり（特に Objective-C ラッパー等の macOS 固有コード）
- **Windows に限定すれば、icx での JUCE ビルドは高い確率で成功すると期待される**

### 4.2 SIMD (AVX/AVX2) — Intel 第4世代Core(Haswell)以降に極限最適化

**ターゲット**: Intel Core 第4世代 (Haswell, 2013〜) 以降。Haswell で導入された AVX2 をベースラインとする。

`__m128` / `__m256` 等の SIMD 組み込み関数を使用している箇所（`CustomInputOversampler.cpp` 等）では、icx の解釈が MSVC と微妙に異なる可能性がある。

- **`__assume()`**: Intel 公式ドキュメント(2025.2)の互換pragma一覧で確認済。icx Windows モードでは MSVC 互換として `__assume` をサポートする。
- **AVX2 組み込み関数**: `<immintrin.h>` / `<xmmintrin.h>` / `<pmmintrin.h>` 等のSIMDヘッダーは icx Windows モードで利用可能。`_MM_SET_FLUSH_ZERO_MODE` / `_MM_SET_DENORMALS_ZERO_MODE` も icx Windows モードでサポート。
- **フラグ**: Windows icx では `-march=core-avx2` は使用不可（Linux専用）。代わりに `/QxCORE-AVX2`（Haswell〜向けIntel最適化、CPUチェックあり）を使用する。`/QxHost`（ホストCPUの最大命令セットを自動検出）に切り替えるオプションを後から追加できる余地を残すこと。

### 4.4 icx の Windows ドライバモードとプリプロセッサマクロ（Intel公式情報に基づく）

**出典**: [Intel® oneAPI DPC++/C++ Compiler Developer Guide and Reference 2025.2 - Additional Predefined Macros](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/additional-predefined-macros.html)

Intel公式ドキュメント（2025.2, Clang 22ベース。2026.0も同一のマクロ体系）に基づくプリプロセッサマクロ情報:

| マクロ | プラットフォーム | Intel公式説明 | icx (Windows) |
|---|---|---|---|
| `_MSC_VER` | Windows | "The Visual C++ version being used." | ✅ **定義される** |
| `_MSC_FULL_VER` | Windows | "The Visual C++ version being used." | ✅ 定義される |
| `_MSC_EXTENSIONS` | Windows | "Defined when Microsoft extensions are enabled." | ✅ 定義される |
| `__INTEL_LLVM_COMPILER` | Linux Windows | "Version in VVVVMMUU format. e.g. 20230100. Recognized by CMake." | ✅ **公式に定義される** |
| `__VERSION__` | Linux | "Compiler version string." | ❌ **Windowsでは未定義**（Intel公式テーブルにWindows記載なし） |
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

### 4.5 ソースコードのプリプロセッサガード戦略（Intel公式情報に基づく）

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

### 4.6 PGO 非対応

icx にも PGO 機能は存在するが、MSVC と方式・フラグが異なる。本フェーズでは icx PGO は実装せず、MSVC 限定とする。`build.bat` では icx + `pgo-gen`/`pgo-use` の組み合わせを検出したらエラーにする（3.1.1 排他チェック）。

icx の PGO が MSVC と異なる点：

- icx は `-prof-gen` / `-prof-use` フラグを使用（MSVC の `/GENPROFILE` / `/USEPROFILE` とは別）
- プロファイルデータの形式も異なる（.profraw / .profdata vs .pgd / .pgc）
- LLVM ベースの `llvm-profdata` ツールが必要

### 4.7 静的CRTリンク（Intel公式情報に基づく）

**出典**: [Intel® oneAPI DPC++/C++ Compiler Developer Guide 2025.2 - `/MT` option](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/mt.html)

Intel公式ドキュメント 2025.2 に基づく確定情報:

| 項目 | Intel公式の記述 |
|---|---|
| `/MT` | "Tells the linker to search for unresolved references in a multithreaded, **static** runtime library." |
| **デフォルト値** | **`/MT`**（静的CRTリンクがデフォルト） |
| 補足 | `/MD` を指定すると動的リンクになる。`-fsycl` は自動で `/MD` を設定する |

**結論**: **Intel icx (Windowsモード) はデフォルトで静的CRTリンク (`/MT`) を使用する**。これは現在の MSVC ビルド（`MSVC_RUNTIME_LIBRARY` で明示的に `/MT` を指定）と同じ動作である。**追加設定は不要**。

### 4.8 ビルドディレクトリの分離

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

### 3.2.13 Debug ビルドの icx 対応（参考情報）

icx の Debug モードフラグ（Intel公式ドキュメント 2025.2 で確認済）:

| フラグ | icxでの対応 | 説明 |
|---|---|---|
| `/Od` | ✅ 対応（Windows専用） | 全最適化を無効化。MSVCと同一のフラグ形式 |
| `/Zi` | ✅ 対応 | PDBファイルにデバッグ情報を生成。`/O2` 以上を明示指定しない場合は `/Od` がデフォルトになる |
| `/Z7` | ✅ 対応 | デバッグ情報を .obj ファイルに埋め込み（PDBなし） |
| `/DEBUG` | ✅ 対応（リンカオプション） | リンカがPDBを生成 |

icx Debug モードの推奨 CMake 設定:

```cmake
# icx の Debug フラグは MSVC と同一で動作するため、特別な分岐は不要
# CMake の CMAKE_CXX_FLAGS_DEBUG にはデフォルトで /Od が設定される
```

**注意**: `/Zi` を指定すると `/O2` が暗黙的に無効化され `/Od` がデフォルトになる。明示的に `/O2`（Release）または `/O3`（icx Release）を指定した場合のみ最適化が有効になる。

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

3. **icx Debug 試験ビルド**:

   ```cmd
   build.bat Debug icx clean
   ```

4. **実行確認**: icx ビルドのバイナリが正常起動し、オーディオ処理が動作することを確認

5. **JUCE ヘッダー互換性**: JUCE 8.0.12 の JuceLibraryCode 生成に IntelLLVM 特有の問題がないか確認

6. **MKL リンク確認**: icx ビルドで MKL が正しくリンクされていることを確認

   ```cmd
   dumpbin /imports build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe | findstr mkl
   ```

7. **ASan 試験 (Optional)**:

   ```cmd
   build.bat Debug icx clean   # + ASan オプション未実装のため手動で cmake に -DENABLE_ASAN=ON を追加
   ```
