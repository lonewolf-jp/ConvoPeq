# ConvoPeq ビルドシステム・DSP論理バグ監査報告 【検証済み版】

対象: `file8831314928463026568.md`（build.bat / CMakeLists.txt / AllpassDesigner / AlignedAllocation）
監査方法: 全指摘項目の実コード直接検証（2016-07-24時点の最新コード）
検証ツール: serena MCP server, AiDex MCP server, 実コード直接読み取り

---

## 🚨 Bug #1: `build.bat` の引数解析ロジックによる変数破壊
**状態: ✅ 確認済みの真正バグ**

**該当箇所（build.bat 206-211行目）:**
```bat
for %%A in (%*) do (
    set "arg=%%~A"
    if "!arg:~0,2!"=="-D" (
        set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! !arg!=ON"
    )
)
```

**検証結果:**
- Windows `cmd.exe` では `=` がトークンの区切り文字として扱われることは **公知の事実**。
- `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` を実行すると、`%*` から `for` への受け渡し時点で `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` と `OFF` の **2つの別引数に分割**される。
- ループ内で:
  - 1回目: `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` → `-D` マッチ → `=ON` 強制付与
  - 2回目: `OFF` → `-D` にマッチせず、キーワードにもマッチせず → **サイレント破棄**
- **結果: `-Dxxx=OFF` や `-Dxxx=任意値` は指定不可能。常に `=ON` が強制される。**

**報告の修正案の問題点:**
```bat
cmake -B build -S . -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %*
```
- `%*` には最初の引数（`Release`）も含まれる → cmake に `-DCMAKE_BUILD_TYPE=Release Release -D...` と渡される。最初の `Release` がpositional argumentとしてcmakeに無視されるため実害は軽微だが、**ダーティな手法**。
- **推奨修正: `shift` 後に `%*` を使うか、明示的に `%*` から最初の引数を除外する処理を入れる。**

**優先度: 高** — 開発者が `OFF` や任意の CMake 変数値を指定したい場合に確実に動作しない。

---

## 🚨 Bug #2: `CMakeLists.txt` の Clang-Tidy 引数パース失敗
**状態: ✅ 確認済みの真正バグ**

**該当箇所（CMakeLists.txt 1067-1074行目）:**
```cmake
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE};
     -p=${CMAKE_BINARY_DIR};
     --extra-arg-before=--driver-mode=cl;
     --extra-arg=/EHsc;
     --extra-arg=-fexceptions;
     --extra-arg=-D_HAS_EXCEPTIONS=1;
     --header-filter=.*/src/.*"
)
```

**検証結果:**
- ダブルクォーテーションで囲まれた文字列内に **改行と5文字のインデントスペース** が含まれている。
- CMakeは `"..."` 内のセミコロンをリスト区切りとして解釈するが、**各要素の先頭に改行＋スペースが含まれる**。
- 例えば `-p=${CMAKE_BINARY_DIR}` の代わりに `     -p=<binary_dir>`（先頭に5スペース）がClang-Tidyに渡される。
- **結果: Clang-Tidy が不正な引数としてエラーを吐く、または完全に無視される。**

**報告の修正案は正しい:**
```cmake
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE}"
    "-p=${CMAKE_BINARY_DIR}"
    "--extra-arg-before=--driver-mode=cl"
    # ...
)
```

**優先度: 中** — Clang-Tidy機能はデフォルトOFF。ONにした際に確実に動作しないため、誰かが有効化しようとした時に混乱を招く。

---

## 🚨 Bug #3: `AllpassDesigner.cpp` の無意味な周波数クランプ（論理バグ）
**状態: ✅ 確認済みの論理バグ（デッドコード）**

**該当箇所（AllpassDesigner.cpp 33行目）:**
```cpp
const double maxCandidateHz = std::max(kMinCandidateHz,
    std::min(0.45 * sampleRate, 0.499 * sampleRate));
```

**検証結果:**
- `0.45 * sampleRate` は正の `sampleRate` において **常に** `0.499 * sampleRate` よりも小さい。
- 従って `std::min(0.45 * sampleRate, 0.499 * sampleRate)` の結果は **常に `0.45 * sampleRate`**。
- `0.499` の制約は **完全なデッドコード**。決して使われることはない。
- 同じパターンが `clampOptimizationFrequency()`（60行目）にも存在。

**意図されたコード（報告の修正案は正しい）:**
```cpp
const double maxCandidateHz = std::min(20000.0, 0.499 * sampleRate);
```

**優先度: 低** — デッドコードであり、`0.45 * sampleRate` のみが使用される。`44.1kHz` 時 `0.45*44100=19845Hz`、`192kHz` 時 `0.45*192000=86400Hz`。後者は可聴域を超えるが、AllpassDesignerのCMA-ES最適化の周波数上限として機能面の実害はない（高い周波数でも群遅延の近似には影響が少ない）。ただしコード品質上の問題として修正推奨。

---

## 🚨 Bug #4: Intel icx 環境におけるテストターゲットのリンク失敗 (LNK1104)
**状態: ⚠ 部分的に確認（環境依存）**

**該当箇所（CMakeLists.txt 614-619行目）:**
```cmake
if(EXISTS "${_INTEL_COMPILER_ROOT}")
    target_link_directories(ConvoPeq PRIVATE "${_INTEL_COMPILER_ROOT}")
    ...
endif()
unset(_INTEL_COMPILER_ROOT)
```

**検証結果:**
- `_INTEL_COMPILER_ROOT` は `ConvoPeq` ターゲットにのみ適用され、**適用直後に `unset()` で削除されている**。
- `MTNUPCMeasurement` を含む全テストターゲットはこのリンクパスの恩恵を受けられない。
- ただし `/Qmkl:sequential` を使用するテストターゲットはコンパイラ指令が.objに埋め込まれるため、**MKLそのものは自動リンクされる。**
- `libircmt.lib`（Intel C++ ランタイム）がデフォルトパスにない環境では **LNK1104 が現実的に発生しうる**。
- **条件付きで確認**: oneAPI 2026.0 のデフォルトインストールでは `compiler/latest/lib` がパスに入るため問題にならない場合が多いが、カスタムインストールや環境変数の欠落時には発生する。

**報告の修正案は妥当（ただし防御的すぎる場合あり）:**
```cmake
if(TARGET MTNUPCMeasurement)
    target_link_directories(MTNUPCMeasurement PRIVATE "${_INTEL_COMPILER_ROOT}")
endif()
```

**優先度: 低** — 特定の環境でのみ発生。テスト前に `LNK1104` が出た時点で気づけるため、本番バイナリに影響なし。

---

## ⚠️ Bug A: `makeAlignedArray` の未初期化メモリ (DSPハザード)
**状態: ✅ 確認済み（設計上のリスク）**

**該当箇所（AlignedAllocation.h 125-131行目）:**
```cpp
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray(size_t count) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "Aligned array only supports trivially destructible types");
    T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    if (!ptr) throw std::bad_alloc();
    return ScopedAlignedArray<T>(ptr);
}
```

**検証結果:**
- `aligned_malloc()` → `mkl_malloc()` は **calloc ではない**。メモリは未初期化。
- `trivially_destructible` の static_assert により、POD型のみ許容されるが、初期化は行われない。
- 一方、`makeAlignedArray_nothrow()` も同様に未初期化。

**実コードでの初期化状況を確認:**
- `NoiseShaperLearner.cpp`: `candidatePopulation` は `optimizer.sample()` で上書きされるため未初期化でも問題なし。`segmentBuffer` は `clear()` 経由で使用。
- `TruePeakDetector.cpp`: `upsampleBuffer` は使用前に `FloatVectorOperations::copy()` で埋められる。
- `CmaEsOptimizer`: `mean` と `covariance` は `initFromParcor()` または `resetIdentityCovariance()` で初期化される。
- **調査範囲では、未初期化メモリがそのままDSPに使用されるケースは確認できなかった。**

**報告の修正案の問題点:**
- `_aligned_malloc` を使用しているが、現状コードは `convo::aligned_malloc` → `mkl_malloc` / `DIAG_MKL_MALLOC` を使用。**修正案は MKL 診断機能をバイパスしてしまう。**
- `if constexpr` は C++17 必須だが、プロジェクトは C++20 のため問題なし。
- **推奨修正**: `convo::aligned_malloc` を使い、`std::memset(ptr, 0, count * sizeof(T))` でゼロクリアする。

**優先度: 低（予防的）** — 現状の呼び出し側は全て適切に初期化している。ただし将来の新規コードで未初期化バッファがDSPに渡されるリスクを予防するため、ゼロクリアを追加するのは妥当。

---

## ⚠️ Bug B: IntelLLVM (icx) + AddressSanitizer のランタイム競合
**状態: ✅ 確認済みの真正バグ**

**該当箇所（CMakeLists.txt 791-800行目）:**
```cmake
if(ENABLE_ASAN)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
        # MSVC ASan requires dynamic CRT (/MDd)
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")     # ← 動的CRT
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)  # ← ASan有効化のみ
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
        # ★ CRT切り替えが欠落！
    endif()
endif()
```

**加えて icx ビルドは静的 CRT を強制している（776-779行目）:**
```cmake
target_compile_options(ConvoPeq PRIVATE
    $<$<CONFIG:Debug>:/MT>     # ← 静的CRT
)
```

**検証結果:**
- MSVCブランチは `MultiThreaded$<$<CONFIG:Debug>:Debug>DLL`（動的CRT: `/MD` or `/MDd`）に切り替えている。
- icxブランチは CRT 切り替えが **完全に欠落**。静的CRT（`/MT` or `/MTd`）のまま ASan を有効化する。
- Windows上のASanは動的CRT（`/MDd`）を要求する。静的CRTとの組み合わせでは **LNK2038（ランタイムライブラリの不一致）** またはリンク時の未解決シンボルエラーが発生する。
- **結果: icx + ENABLE_ASAN=ON ではビルドが確実に失敗する。**

**報告の修正案:**
```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
```

**優先度: 中** — ASan使用時（Debug専用）にのみ影響。Releaseビルドには影響なし。

---

## 検証結果サマリー

| Bug# | ステータス | 深刻度 | 実害 |
|------|-----------|--------|------|
| #1 build.bat `=` 分割 | ✅ 確認 | **高** | `-Dxxx=OFF/任意値` が不可能 |
| #2 Clang-Tidy 引数 | ✅ 確認 | 中 | 機能デフォルトOFFだが、ON時は無効 |
| #3 AllpassDesigner デッドコード | ✅ 確認 | 低 | 機能面の実害なし、コード品質 |
| #4 icx テストリンク | ⚠ 環境依存 | 低 | 特定icx環境でLNK1104 |
| A makeAlignedArray 未初期化 | ✅ 確認（設計リスク） | **低（予防的）** | 現状の呼出側は全て適切に初期化済み |
| B icx + ASan CRT競合 | ✅ 確認 | 中 | icx + ASan でビルド失敗確実 |

### 本物のバグ（要修正）
1. **Bug#1**: `build.bat` の `-D` 引数処理 — `=` が区切り文字になる問題。**優先度: 高。**
2. **Bug#2**: CMakeLists.txt Clang-Tidy 引数 — 先頭スペース入りの不正フォーマット。**優先度: 中。**
3. **Bug#B**: icx ASan CRT切り替え欠落 — 静的CRTとASan非互換。**優先度: 中。**

### 報告の修正案の問題点
- **Bug#1** の修正案: `%*` に最初の引数が含まれる。`shift` 後に `%*` を使うか、`%*` から最初の引数を除外すべき。
- **Bug A** の修正案: `_aligned_malloc` を使用しているが、現状コードは `convo::aligned_malloc`（MKL経由）を使用。`mkl_malloc` を使い続ける必要がある。また `DIAG_MKL_MALLOC` の診断機能をバイパスするため、修正時は `convo::aligned_malloc` + `std::memset` とすべき。

### 元報告で正しかった指摘（高品質）
- Bug#1 の `=` 分割問題は Windows バッチスクリプティングの **発見が難しいコーナーケース** を正確に捉えている。
- Bug#3 のデッドコード発見は **静的解析の良い事例**。
- Bug#B のASan CRT競合は icx 環境での **コンパイラ間の微妙な差異** を見逃していない。
