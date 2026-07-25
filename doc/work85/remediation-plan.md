# ConvoPeq バグ改修計画書

作成日: 2026-07-24
最終更新: 2026-07-25
ベース文書: `end-to-end-buglist.md`（全44バグ項目検証済み）
対象: 真正バグ10件（要修正）＋予防的改善4件＋設計上の軽度懸念1件

---

# 設計

実装担当プログラマが即座に参照すべき情報を集約する。
各タスクは **問題 → 修正方針（コード例） → リスク → テスト戦略** の形式で記述する。

## 凡例

| 記号 | 意味 |
|------|------|
| 🛠 | コード修正を伴う |
| 🔧 | 設定ファイル修正 |
| 📋 | 文書化・調査のみ |
| ⚡ | Audio Thread 安全性要確認 |

## 0. 改修全体ロードマップ

```
Phase 1 (P1) ─── A01 ─── G03 ─── B01 ───→ 検証
                     ↓
Phase 2 (P2) ─── A02 ─── B03 ─── B02 ─── G07 ──→ 検証
                     ↓
Phase 3 (P3) ─── A03 ─── A04 ─── B04 ───→ 検証
                     ↓
Phase 4 (P4) ─── G02 ─── A05 ─── A06 ───→ 検証
```

### 依存関係
- A01 は他タスクと独立（単独で実施可能）
- B01 は B02/B03 と独立（単独で実施可能）
- A02 は A05 と部分依存（`makeAlignedArrayZero` が必要な場合 → Phase4 で A05 実施後、A02 の呼出側を修正）
- G03 は独立（単独で実施可能）
- G02 は ISR Runtime 深い知識が必要。Phase4 で実施

---

## Phase 1: P1-High（即時対応）

### T1. A01: `_mm256_zeroupper()` 欠如
**優先度: P1-High** | **種別: 🛠⚡** | **工数: 中（AVX→legacy SSE遷移箇所、現状調査では最大16ファイル）**

#### 問題
全てのAVX2命令使用ファイルで `_mm256_zeroupper()` が呼ばれていない。JUCEのSSEコードとの混在により、AVX2→SSE遷移で最大100サイクル超のペナルティが発生する。

#### 対象箇所
AVX→legacy SSE 遷移が存在する箇所。現状調査では以下16ファイルが該当するが、ファイル単位ではなく**遷移箇所単位**で修正すること。
- **Audio Thread 直接（8）**: `DSPCoreDouble.cpp`, `DSPCoreFloat.cpp`, `DSPCoreIO.cpp`, `EQProcessor.Processing.cpp`, `MKLNonUniformConvolver.cpp`, `CustomInputOversampler.cpp`, `TruePeakDetector.cpp`, `LoudnessMeter.cpp`
- **Audio Thread 間接（2）**: `ConvolverProcessor.Runtime.cpp`, `ConvolverProcessor.LoaderThread.cpp`
- **UI Thread（2）**: `AudioEngine.EQResponse.cpp`, `SpectrumAnalyzerComponent.cpp`
- **ヘッダー（4）**: `LatticeNoiseShaper.h`, `InputBitDepthTransform.h`, `DspNumericPolicy.h`, `dsp/math/FastTanhApprox.h`

**注意**: AVX命令だけで完結する関数（例: `fastTanhV256` 等）は対象外。各ファイル内でAVX→SSE遷移が発生する箇所のみに挿入すること。

#### 修正方針（レビュー反映 v2）
**採用: Compiler Option 優先 + AVX命令実行後、legacy SSE命令が初めて実行される直前への明示的 `_mm256_zeroupper()`**

RAII方式（スコープ終了時に発行）は以下の理由で不採用:

1. **スコープ依存の予測困難性**: `if {...}` などのブロックスコープでも発行され、コードレビュー性が悪い
2. **最適化との干渉**: `inline` / `template` / `constexpr` 関数では最適化による予測が困難
3. **戻り値 `__m256d` 関数との相性**: 関数内で発行するとYMMレジスタ上位がゼロクリアされ値が破損するリスク

**修正方法（2段階）:**

**Step 1: Compiler Option の追加（最優先）**
Intel の推奨に従い、icx の `-mvzeroupper` は C++ 翻訳単位のみに限定して適用する:

```cmake
# CMakeLists.txt: icx 向け -mvzeroupper（C++ 翻訳単位のみ、かつ AVX2 有効時）
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # ★ AVX2 が有効な場合のみ -mvzeroupper を適用
    target_compile_options(ConvoPeq PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${CONVOPEQ_ENABLE_AVX2}>>:-mvzeroupper>)
elseif(MSVC)
    # MSVC: コンパイラオプションなし。Step 2の明示的挿入に頼る
endif()
```

- icx: `-mvzeroupper` で全関数に自動挿入（基本解決）。CXX限定かつAVX2有効時のみで不要なTUへの適用を回避
- **ただし**: Intel 自身が注意する通り、コンパイラは間接境界（DLL境界、virtual関数、function pointer経由など）を常に認識できるとは限らない。icxでも `-mvzeroupper` でカバーできない遷移点では手動 `_mm256_zeroupper()` の追加を許可する
- MSVC: 同等オプションなし。Step 2の明示的挿入で対応

**Step 2: MSVC向け 明示的 `_mm256_zeroupper()` 挿入**
- RAIIラッパーは使用しない
- **「関数の出口」ではなく「AVX命令実行後、legacy SSE命令が初めて実行される直前」に配置する**（Intel Optimization Manual 推奨）
- 対象はAVX2命令を使用した後、SSEコード（JUCE等）が実行される直前の箇所
- 複数の return があっても「全 return パス」ではなく「全 AVX→legacy SSE 遷移パス」を漏れなくカバーする
- `AVX → if (...) return; → AVX → JUCE SSE` のようなコードでは、各 AVX→SSE 境界に個別に配置
- `__m256d` を返す関数（`fastTanhV256` / `killDenormV`）は対象外（呼び出し元が別途対応）

**具体的な配置ルール（実装者向け）**:
以下の**直前に `_mm256_zeroupper()` を配置する**:
1. legacy SSE 命令（`_mm_*` 系）を実行する可能性のある関数呼び出し
   - JUCE 関数呼び出し（`FloatVectorOperations`、`AudioBlock` 演算等）
   - SSE intrinsic を使用する自前の関数
2. AVX 以外の DSP ライブラリ呼び出し（MKL 非AVX関数、IPPS 等）
3. SSE/legacy SSE 領域への突入が明らかなコードブロック
4. 外部ライブラリ（DLL）境界を越える呼び出し

**注意**: 「JUCE だから必要」ではなく「JUCE 内部で legacy SSE を使う経路だから必要」が正しい判断基準。
AVX 命令だけで完結し、legacy SSE 命令を一切呼ばない関数には不要。

```cpp
// 修正例（DSPCoreFloat.cpp など）:
void SomeAVX2Function(float* data, int count) {
    // ... AVX2処理 ...

    // ★ AVX→legacy SSE 遷移: AVX命令実行後、legacy SSE命令が初めて実行される直前
    _mm256_zeroupper();

    // ... legacy SSE コード（JUCE 等）...
}
```

**「AVX命令実行後、legacy SSE命令が初めて実行される直前」に配置する理由**:
- 1関数内で AVX→SSE→AVX と遷移するケースでは、関数末尾だけでは不十分
- 逆に AVX のみで完結し SSE を呼ばない関数では、関数末尾の VZEROUPPER は不要
- AVX コードの直後にインライン展開された SSE ヘルパーが呼ばれるケースに対応する必要がある

#### 代替案（不採用確定）
- **RAIIラッパー方式**: スコープ依存・コードレビュー性悪化のため不採用
- **中央インクルードで `#define _mm256_*` ラップ**: 保守性低。新規AVX2コードで漏れうる

#### リスク
- **低**: VZEROUPPERは軽量命令（μarch依存、実質ほぼ0〜数cycle）。過剰発行の実害は小さい
- **⚠ 戻り値が `__m256d` の関数に注意**: `dsp/math/FastTanhApprox.h` の `fastTanhV256()` と `DspNumericPolicy.h` の `killDenormV(__m256d)` は `__m256d` を返す。これらの関数内では VZEROUPPER を発行しないこと（呼び出し元で発行すれば良い）

#### テスト戦略
- ビルド確認（MSVC + icx）
- icx: `-mvzeroupper` 付与で、必要なABI境界に `vzeroupper` が自動挿入されることを `objdump -d` で確認
- MSVC: Release ビルド後 `dumpbin /disasm` または `llvm-objdump -d` で各AVX→SSE境界に `vzeroupper` を確認（sample-based）。インライン展開された命令も確認可能なツールを推奨
- **CI 静的チェック（推奨）**: AVX intrinsic（`_mm256_*`）を含む翻訳単位で `vzeroupper` が一度も出現しない場合に警告または失敗するチェック。または特定ホットパスを `objdump -d` で確認し、必要なABI境界に `vzeroupper` が存在することをサンプルベースで検証。
  ※ AVX命令数と `vzeroupper` 数の単純比較は誤検出が多いため推奨しない（LLVM最適化により変動するため）。

---

### T2. B01: `build.bat` `-D` 引数解析不能
**優先度: P1-High** | **種別: 🔧** | **工数: 小**

#### 問題
`build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` が不可能。`FOR %%A in (%*)` のトークン化が `=` を区切り文字として扱うため、`OFF` がサイレント破棄される。

#### 修正方針（レビュー反映）
**採用: `SHIFT` 解析方式（定番パターン）**

`for %%A in (%*)` 方式は `=` がデフォルト区切り文字に含まれる（`for /?` 参照）ため、引用符なしの `-DXXX=OFF` が2引数に分割される。引用符で回避できるが、Windows cmd の展開規則は複雑であり、確実性に欠ける。

代わりに `SHIFT` 解析方式（定番パターン）を採用:

```bat
REM 修正後: SHIFT 解析方式（定番パターン）
setlocal EnableDelayedExpansion  ★ 必須: !CMAKE_EXTRA_FLAGS! の展開に必要
set "CMAKE_EXTRA_FLAGS="
:argloop
if "%1"=="" goto :argend
set "arg=%1"
if /i "%arg%"=="icx" set "USE_INTEL=1"
if /i "%arg%"=="clean" set "DO_CLEAN=1"
if /i "%arg%"=="debug" set "DO_DEBUG=1"
if /i "%arg%"=="release" set "DO_RELEASE=1"
if /I "%arg:~0,2%"=="-D" (
    set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! %arg%"
)
shift
goto :argloop
:argend
```

**前提条件**: `setlocal EnableDelayedExpansion` が有効であること（`!CMAKE_EXTRA_FLAGS!` の展開に必須）。build.bat 冒頭に既存の記述がなければ追加する。

**注意**: `EnableDelayedExpansion` 有効時は `!` を含む引数が展開される。CMAKE_EXTRA_FLAGS に `!` を含めることは非推奨。CMake の `-D` 変数値に `!` が含まれる場合は引用符で囲むかエスケープすること（通常のCMake定義では `!` は稀なため実害はほぼない）。

**使い分け:**
- `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` → **従来通り動作**。CMake に値なしの `-D` → `=ON` 扱い
- `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` → **引用符不要で正しく解析**。CMake に `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF`
- `build.bat Release "-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF"` → **引用符があっても動作**（相変わらず安全）

**補足**: `SHIFT` 方式の利点:
- `=` を含む引数を分割せずに処理可能
- 引数の順序に依存しない
- Windows cmd の定番パターンで可読性が高い
- 将来の引数追加が容易

#### リスク
- **極低**: `SHIFT` 方式は Windows bat で広く使われる定番パターン。CMake 引数のみに影響
- **低**: 既存のドキュメント（`build.bat` ヘッダーコメント）の `=ON` 自動付与に関する記述を削除/修正する必要あり

#### テスト戦略
- `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` が CMake キャッシュに正しく反映されることを確認（引用符不要）
- `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`（従来方式）が従来通り動作することを確認
- `build.bat Release icx clean` が期待通り動作することを確認

---

### T3. G03: `FlagResetter` スレッドキャンセル時フラグ残留
**優先度: P1-High** | **種別: 🛠** | **工数: 小**

#### 問題
`ConvolverProcessor.LoaderThread.cpp` の `FlagResetter::~FlagResetter()` が `!t.threadShouldExit()` の条件でスキップされる。スレッドキャンセル時に `isLoading` / `isRebuilding` フラグが true のまま残留し、UI が永久待機状態になる。

#### 修正内容
`if (!success && !t.threadShouldExit())` → `if (!success)` に変更。スレッドキャンセル時もフラグリセットを試行する。

```cpp
~FlagResetter() {
    if (!success) {  // ← 修正: threadShouldExit 条件を削除
        auto wp = weakP;
        const bool queued = juce::MessageManager::callAsync([wp] { ... });
        if (!queued) {
            if (auto* o = wp.get()) {
                // callAsync 失敗 = MessageManager が利用できない状態（未初期化/終了中/Shutdown中）。
                // ★ 設計: Shutdown中はatomic状態のみ整合性を維持する（次回ロード時のフラグ競合防止）。
                //   UI コンポーネント状態は更新されない可能性がある（MessageManager不在のため）。
                //   これは仕様であり、Shutdown完了後の再初期化でUI状態はリセットされる。
                convo::publishAtomic(o->isLoading, false, std::memory_order_release);
                convo::publishAtomic(o->isRebuilding, false, std::memory_order_release);
            }
        }
    }
}
```

#### ⚠ レビュー指摘
- **全体的に妥当**: ISR Runtime でも「キャンセル＝状態復帰」は必須
- **WeakReference で安全**: キャンセル後に `callAsync` が成功した場合、コールバックが後日 Message Thread で実行される。その時点で `o` がまだ有効であることが WeakReference で保証される。
- **要確認事項**: `callAsync` 失敗時の `atomic` 直接更新が、MessageThread専用状態を書き換えていないか。該当箇所のソースコード確認が必要。

#### ✅ ソースコード検証結果（2026-07-25）
`src/convolver/ConvolverProcessor.LoaderThread.cpp` の該当コードを確認:
```cpp
convo::publishAtomic(o->isLoading, false, std::memory_order_release);
convo::publishAtomic(o->isRebuilding, false, std::memory_order_release);
```
- ✅ `convo::publishAtomic()` を使用 — ISR Runtime の `AtomicAccess.h` ラッパー経由
- ✅ `std::atomic<bool>` 型のメンバ変数のみ操作 — MessageThread 専用状態は一切非該当
- ✅ プレーンな `atomic.store()` は不使用 — ISR Runtime 設計準拠
- ✅ `WeakReference` による保護も有効
**結論: 問題なし。現状の修正内容をそのまま実装可能。**

#### リスク
- **低〜中**: キャンセル後に `callAsync` が成功した場合、コールバックが後日 Message Thread で実行される。その時点で `o` がまだ有効であることが WeakReference で保証される。問題なし。
- **最悪ケース（`callAsync` 失敗 + WeakRef無効）**: WeakReference が無効なら対象オブジェクト（`ConvolverProcessor`）は既に破棄済みであり、更新不要。

#### テスト戦略
- IR ロード中に `signalThreadShouldExit()` を呼び、フラグが適切にリセットされることを確認
- UI の `isLoadingIR()` が false に戻ることを確認
- 通常の成功パスが影響を受けないことを確認

---

## Phase 2: P2-Medium（次期対応）

### T4. A02: `NoiseShaperLearner` スタック配列のヒープ化
**優先度: P2-Medium** | **種別: 🛠** | **工数: 小**

#### 問題
`buildTrainingSegments()` 内の `double recentLeft[34816] = {}` + `double recentRight[34816] = {}` が合計544KBのスタックを消費。Windows既定1MBのスタックに対して余裕がない。

#### 修正内容
```cpp
// 現在（スタック）:
double recentLeft[kRecentSampleRequest] = {};
double recentRight[kRecentSampleRequest] = {};

// 修正後（ヒープ + ゼロ初期化）:
auto recentLeft = convo::makeAlignedArray<double>(kRecentSampleRequest);
auto recentRight = convo::makeAlignedArray<double>(kRecentSampleRequest);
// makeAlignedArray にゼロクリア追加（T9 と連動）
```

#### リスク
- **極低**: ヒープ確保に変更するが、`buildTrainingSegments()` は Worker Thread から呼ばれる（非RT）。`makeAlignedArray` は `mkl_malloc` 経由で64byteアライン保証。
- ゼロクリアは Phase4 の A05 で `makeAlignedArrayZero` 分離後に呼出側を修正。
- **`bad_alloc` 時**: `makeAlignedArray` が `std::bad_alloc` を投げた場合、Worker Thread 側で `catch (const std::bad_alloc&)` により捕捉し、ログを出力して学習のみスキップする。このとき現在のノイズシェイピング設定は維持され、オーディオ処理は継続可能。

#### 依存
- **T9 (A05)**: Phase4 で `makeAlignedArrayZero` を分離後、このタスクの呼出側を `makeAlignedArrayZero` に変更。Phase2 では `std::memset` で暫定対応してもよい。

---

### T5. B02: Clang-Tidy 引数フォーマット修正
**優先度: P2-Medium** | **種別: 🔧** | **工数: 小**

#### 問題
CMakeLists.txt の Clang-Tidy 設定で、マルチラインクォート内のインデントスペースが各引数の先頭に付与される。

#### 修正内容
```cmake
REM 現在（問題あり）:
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE};
     -p=${CMAKE_BINARY_DIR};
     ..."
)

REM 修正後:
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE}"
    "-p=${CMAKE_BINARY_DIR}"
    "--extra-arg-before=--driver-mode=cl"
    ...
)
```

#### リスク
- **極低**: 構文のみの変更。Clang-Tidy がデフォルト OFF のため、本修正によるリグレッションリスクなし。

---

### T6. B03: icx + ASan CRT 競合修正
**優先度: P2-Medium** | **種別: 🔧** | **工数: 小**

#### 問題
icx ブランチの ASan 有効化時に動的 CRT への切り替えが欠落。静的 CRT と ASan の非互換により `LNK2038` エラーが発生する。

#### 修正内容
CMakeLists.txt の `ENABLE_ASAN` ブロックに、IntelLLVM 用の CRT 切り替えを追加:

```cmake
# CMakeLists.txt: ENABLE_ASAN ブロック — 動的CRT切替はASan有効時のみであることに注意
# ★ 注意: MSVC_RUNTIME_LIBRARY は他のCMake設定で上書きされる可能性がある。
#   ENBALE_ASAN ブロックが他の設定より後で評価されることを確認すること。
#   特に共通マクロで /MT を指定している場合、ここで /MD に上書きされる。
if(ENABLE_ASAN)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE /fsanitize=address)
        target_link_options(ConvoPeq PRIVATE /fsanitize=address)
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        target_compile_options(ConvoPeq PRIVATE -fsanitize=address)
        target_link_options(ConvoPeq PRIVATE -fsanitize=address)
        # ★ 追加: icx でも動的 CRT に切り替え（ENABLE_ASAN 時のみ）
        #   注意: 通常時は静的CRT（/MT）のまま。ASan は動的CRT（/MD）が必須。
        #   既存の target_compile_options に /MT が残っていないことを確認すること。
        #   set_property だけでは不十分で、compile_options 側の /MT が優先される場合がある。
        set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()
```

#### リスク
- **低**: ASan は Debug ビルド専用。Release ビルドに影響なし。
- icx Debug の CRT が `/MTd` から `/MDd` に変わるが、JUCE の Debug ランタイム要件と整合する。
- `set_property` だけでは不十分な場合がある。`compile_commands.json` で該当ターゲットの compile flags に `/MT` が残っていないことを確認すること。

---

### T8. G07: `makeEngineRuntimeState()` world==nullptr フォールバック拡充
**優先度: P2-Medium** | **種別: 🛠** | **工数: 小**

#### 問題
`AudioEngine.h` の `makeEngineRuntimeState()` で `runtimeWorld == nullptr` 時に `retireBacklog` と `deferredResidency` が強制的に 0 になる。

#### 修正方針
**注意**: `AudioEngine::getRetireRouter()` の戻り値型は `IReaderEpochProvider&` であり、`pendingRetireCount()` / `pendingIntentCount()` を提供しない。これらのメソッドは `ISRRetireRouter` または `IEpochProvider` のインターフェースに存在する。したがって T8 の修正には以下の2ステップが必要:

**Step 1**: フォールバック値を `RetireRuntime` から直接問い合わせる（`IEpochProvider` を派生元にキャストできる場合）、または

**Step 2**: 該当フィールドが使用される箇所（統計表示など）で「runtimeWorld == nullptr 時は値を利用不可」として明示的に扱う。

推奨: 現状の `fallback.retireBacklog = 0` は安全側フォールバックであり、実害が軽微であるため、**Step 2 相当の対応（コメントで明記）に留める**。

```cpp
if (runtimeWorld == nullptr) {
    convo::EngineRuntime fallback {};
    // ... 既存のフォールバック ...
    fallback.retireBacklog = 0;  // runtimeWorld不在のため利用不可。安全側フォールバック。
    fallback.deferredResidency = 0;  // （同上）
    // ...
}
```

---

## Phase 3: P3-Low（予防・品質）

### T10. A03: `LockFreeAudioRingBuffer::push()` 戻り値改善
**優先度: P3-Low** | **種別: 🛠** | **工数: 小**

#### 問題
`push()` の戻り値が `void`。空き容量不足時にサイレント部分書き込みが発生する。

#### 修正内容
```cpp
// 戻り値を int に変更: 実際に書き込んだサンプル数を返す
[[nodiscard]] int push(const juce::dsp::AudioBlock<const double>& block) noexcept
{
    // ...
    const int samplesToWrite = juce::jmin(samplesToWriteRequested, free);
    // ... 書き込み処理 ...
    convo::publishAtomic(writeIndex, write + static_cast<uint64_t>(samplesToWrite),
                         std::memory_order_release);
    return samplesToWrite;  // ← 書き込んだ数を返す
}
```

#### 呼出側の対応
`pushToFifo()` 側で戻り値を受け取り、書き込めなかったサンプル数を以下のいずれかで処理する:
- **ログ出力**: `juce::Logger::writeToLog` で FIFO オーバーフローを記録
- **XRUN カウンタ**: Audio Thread 安全な atomic カウンタで xrun 回数を追跡
- **Telemetry**: 診断情報として蓄積（将来的な性能最適化に活用）

最低限、XRUN カウンタのインクリメントは実装すること。

**実施順序（重要）**:
1. 全 `push()` 呼び出し側を修正して戻り値を適切に処理する
2. その後で `[[nodiscard]]` を `push()` の宣言に追加する
→ 逆順（先に `[[nodiscard]]` を付ける）だと既存の全呼び出し箇所でコンパイル警告が大量発生するため、**必ずこの順序で実施すること**。

---

### T11. A04: `AllpassDesigner.cpp` デッドコード修正
**優先度: P3-Low** | **種別: 🛠** | **工数: 小**

#### 問題
`std::min(0.45 * sampleRate, 0.499 * sampleRate)` が常に `0.45 * sampleRate` を返す。`0.499` がデッドコード。

#### 修正内容
```cpp
// 現在:
const double maxCandidateHz = std::max(kMinCandidateHz,
    std::min(0.45 * sampleRate, 0.499 * sampleRate));
// clampOptimizationFrequency も同様

// 修正後:
const double kMaxAllpassFrequencyHz = 20000.0;
const double maxCandidateHz = std::min(kMaxAllpassFrequencyHz, 0.499 * sampleRate);
```

---

### T13. B04: icx テストターゲットリンクパス追加
**優先度: P3-Low** | **種別: 🔧** | **工数: 小**

#### 問題
`_INTEL_COMPILER_ROOT` が `ConvoPeq` 本体にのみ適用される。

#### 修正内容
```cmake
if(EXISTS "${_INTEL_COMPILER_ROOT}")
    target_link_directories(ConvoPeq PRIVATE "${_INTEL_COMPILER_ROOT}")
    # ★ 追加: テストターゲットにも反映
    if(TARGET MTNUPCMeasurement)
        target_link_directories(MTNUPCMeasurement PRIVATE "${_INTEL_COMPILER_ROOT}")
    endif()
endif()
```

---

## Phase 4: P4-Design（設計検討事項）

**注意**: 本Phaseのタスクは設計上の検討を経て確定したものである。ISR Runtime設計原則との整合性を確認済み。

### T7. G02: `retireEQStateDeferred` 失敗時フォールバック解放
**優先度: P4-Design** | **種別: 🛠** | **工数: 小（10箇所）**

#### 問題
`EQProcessor.Core.cpp` および `EQProcessor.Parameters.cpp` で `(void)retireEQStateDeferred(oldState)` が戻り値を無視。`enqueueDeferredDeleteWithFallback` が false を返した場合、`oldState` が永久リークする。

#### 修正方針
**採用: Shutdown 専用経路 `retireImmediateDuringShutdown()` を分離**

ISR Runtime の絶対原則 **「Delete は Epoch 経由」** を維持するため、通常時と Shutdown 時の経路を明確に区別する:

```cpp
// ★ Shutdown 専用: Epoch 経由せず即時解放
//    Runtime 停止シーケンス完了後（Audio Thread 停止 + Coordinator 停止）にのみ使用すること
//    ※ 本質条件は「Audio停止かつCoordinator停止」であり、MessageThread制約は副次的なもの。
void EQProcessor::retireImmediateDuringShutdown(EQStatePtr&& state) noexcept {
    // ★ 本質条件: Audio停止 + Coordinator停止（retire queueとの競合防止）
    //   命名で Shutdown 限定を明示しているため、MessageThread制約は必須ではない。
    //   現状の shutdown sequence では MessageThread からのみ呼ばれるが、
    //   将来 Coordinator Thread から呼びたくなった場合は !isAudioThread() のみで十分。
    jassert(!isAudioThread());
    // ★ Coordinator 停止確認が可能ならアサーションを追加推奨
    if (state) {
        deleter(state.release());
    }
}
```

**通常時**は Coordinator 経由のリトライを継続し、同期的解放にはフォールバックしない:

```cpp
bool EQProcessor::enqueueDeferredDeleteWithFallback(...) {
    if (result == convo::isr::RetireEnqueueResult::Success)
        return true;
    return false;  // 同期的解放は行わない
}
```

**呼び出し元の修正（2パターン）:**

```cpp
// 通常時（Audio Thread 稼働中）:
if (!retireEQStateDeferred(oldState)) {
    pendingRetires.push(std::move(oldState));  // 後で再試行
}

// Shutdown 時（Audio Thread 停止後）:
retireImmediateDuringShutdown(std::move(oldState));
```

#### リスク
- **低〜中**: Shutdown 専用経路を分離することで ISR Runtime 設計不変条件を維持
- `retireImmediateDuringShutdown` の命名 + `jassert` による二重防御

---

### T9. A05: `makeAlignedArray` ゼロクリア追加（API分離）
**優先度: P4-Design** | **種別: 🛠** | **工数: 小**

#### 問題
`AlignedAllocation.h` の `makeAlignedArray<T>(count)` が確保したメモリをゼロクリアしない。一律のゼロクリア追加は FFT/IR/DSP 中間バッファで性能低下を招く。

#### 修正方針
**API を2つに分離:**

```cpp
// 既存: 未初期化バッファ（高速。FFT・IR・DSP中間バッファ向け）
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArray(size_t count) {
    static_assert(std::is_trivially_destructible_v<T>, ...);
    T* ptr = static_cast<T*>(aligned_malloc(count * sizeof(T), 64));
    if (!ptr) throw std::bad_alloc();
    return ScopedAlignedArray<T>(ptr);
}

// ★ 新規: ゼロ初期化バッファ（安全。スタック配列代替・学習データ向け）
template <typename T>
inline ScopedAlignedArray<T> makeAlignedArrayZero(size_t count) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "makeAlignedArrayZero requires trivially copyable type");
    auto arr = makeAlignedArray<T>(count);
    std::memset(arr.get(), 0, count * sizeof(T));
    return arr;
}
```

**呼び出し側の使い分け:**
- `makeAlignedArrayZero<double>(kRecentSampleRequest)` → A02 のスタック→ヒープ変換で使用
- `makeAlignedArray<double>(kFftSize)` → FFT ワークバッファ（従来通り、ゼロクリア不要）

#### 注意
- `aligned_malloc` → `DIAG_MKL_MALLOC` の診断を維持すること
- 必要な箇所でのみゼロクリアする設計が望ましい

---

### T12. A06: `AudioSegment` に `alignas(64)` 追加
**優先度: P4-Design** | **種別: 🛠** | **工数: 小**

#### 問題
`NoiseShaperLearner.h` の `AudioSegment` 構造体にアライメント指定がない。

#### 補足
- 効果は限定的（`makeAlignedArray` が既に64byteアライン保証）だが、デメリットもないため実施する。
- 構造体単体の `alignas(64)` だけでは不十分なケースがある: `std::vector<AudioSegment>` など標準コンテナでヒープ確保する場合は、アロケータ（`MKLAllocator<AudioSegment>` 等）との組み合わせまで確認すること。
- 現状の `AudioSegment` は `ScopedAlignedArray<AudioSegment>` で確保されるため、アロケータ問題は生じない。

```cpp
struct alignas(64) AudioSegment {
    static constexpr int kLength = MklFftEvaluator::kFftLength;
    double left[kLength] = {};
    double right[kLength] = {};
    std::array<double, MklFftEvaluator::kSpectrumBins> maskingThresholds {};
};
static_assert(alignof(AudioSegment) == 64, "AudioSegment must be 64-byte aligned");
static_assert(sizeof(AudioSegment) % 64 == 0, "AudioSegment size must be a multiple of 64 to prevent false sharing");
```

---

## 改修実施順序

```
Week 1: Phase 1（即時対応）
  ├── Mon: T2 (B01) — build.bat SHIFT 解析方式に変更（数十分）
  ├── Tue: T3 (G03) — FlagResetter 修正 + テスト（数時間）
  ├── Wed: T1 (A01) — Compiler Option + 明示的 _mm256_zeroupper()（16ファイル）
  ├── Thu: T1 継続 + ビルド確認（MSVC / icx）
  └── Fri: Phase 1 統合テスト・リグレッション確認

Week 2: Phase 2（次期対応）
  ├── Mon: T4 (A02) — NoiseShaperLearner スタック→ヒープ
  ├── Tue: T5 (B02) + T6 (B03) — CMakeLists.txt Clang-Tidy + ASan
  ├── Wed: T8 (G07) — makeEngineRuntimeState コメント追記
  └── Thu-Fri: Phase 2 統合テスト

Week 3: Phase 3（予防・品質）
  ├── Mon: T10 (A03) — push() 戻り値+呼出側
  ├── Tue: T11 (A04) — AllpassDesigner デッドコード
  ├── Wed: T13 (B04) — icx テストターゲットリンクパス
  └── Thu-Fri: Phase 3 統合テスト

Week 4: Phase 4（設計検討事項）
  ├── Mon: T7 (G02) — retireImmediateDuringShutdown 実装
  ├── Tue: T9 (A05) — makeAlignedArrayZero 分離
  ├── Wed: T12 (A06) — alignas(64) 追加
  └── Thu-Fri: Phase 4 統合テスト・全Phaseリグレッション
```

---

## リスク評価マトリクス

| ID | リスク | 確率 | 影響 | 対策 |
|----|--------|------|------|------|
| A01 | Compiler Option の MSVC 非対応による対応漏れ | 低 | 中（ペナルティ残留） | MSVC は明示的挿入でカバー |
| A01 | `__m256d` 戻り値関数での VZEROUPPER 誤発行 | 低 | 中（値化け） | 全関数の戻り値型確認必須 |
| A01 | Compiler 自動 vzeroupper（icx）と手動挿入（MSVC）の重複発行 | 低 | 微（重複は無害） | icx では原則手動挿入を最小限に抑える。MSVC のみ明示的挿入 |
| A01 | LTO/IPO（/GL, 最適化リンク時）によるインライン展開で vzeroupper の配置が変化 | 低 | 低（インライン後も自動挿入維持） | LTO/IPO 設定時は objdump で再確認。特に icx の IPO でインライン展開された SSE コードが AVX 領域に混入していないか確認
| B01 | SHIFT 解析への変更で既存引数が誤認識される | 極低 | 低 | 全呼出パターンのテスト必須 |
| G03 | callAsync 失敗時の atomic 直接更新が MessageThread 状態を破壊 | 低 | 中 | ソースコード確認後に対応 |
| G02 | `retireImmediateDuringShutdown` が通常時に誤用される | 低 | 高 | 名前＋jassert で二重防御 |
| A05 | `makeAlignedArrayZero` と `makeAlignedArray` の使い分けを誤る | 低 | 低 | コメントで用途を明記 |
| A02 | `makeAlignedArrayZero` が `bad_alloc` を投げて学習開始失敗 | 低 | 低（Worker Thread限定） | `bad_alloc` キャッチ後、ログ出力して学習をスキップする
| T10 | 呼出側の戻り値未使用でコンパイル警告 | 中 | 低 | `[[nodiscard]]` 追加 |
| C05 | リサンプル失敗時に IR ロード失敗とするとユーザー体験が低下 | 低 | 低 | エラーメッセージで明確に通知 |

---

## テスト計画

### ビルドテスト（全Phase共通）
- MSVC Release/Debug: ✅ 正常ビルド
- icx Release/Debug: ✅ 正常ビルド
- icx + ENABLE_ASAN=ON: ✅ 正常リンク
- CTest 全テスト通過: ✅
- Clang-Tidy 有効時: ✅ 警告ゼロ

### 機能テスト
| ID | テスト内容 | 期待結果 |
|----|-----------|---------|
| A01 | icx: `-mvzeroupper` 付与後、objdump で vzeroupper 確認 | 必要なABI境界（call前/return前等）に自動挿入 |
| A01 | MSVC: Release ビルド後 dumpbin /disasm で確認 | AVX→legacy SSE 遷移境界に明示的 vzeroupper |
| B01 | `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` | CMake cache に `=OFF` |
| B01 | `build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | 従来通り動作（値なし=ON） |
| G03 | IR ロード中に `signalThreadShouldExit()` | `isLoadingIR()==false` に復帰 |
| A02 | NoiseShaperLearner スタック使用量削減 | スタック使用量が544KB→ヒープに |
| G02 | Shutdown 時に `retireImmediateDuringShutdown` 呼出 | メモリリークなし |
| G02 | 通常時に Coordinator ビジー時の再試行 | 同期的解放は行われない |
| C05 | 非対応 SR の IR ロード | エラー通知、IR 未ロード |

### リグレッション項目
- 既存のビルドスクリプト動作（`build.bat Release`, `build.bat Debug icx`）
- 既存の全テストスイート通過
- 各 Phase 完了時のチェックポイントで上記確認

---

# 未確定・未決定事項

以下の項目は設計判断または追加調査が必要なため、本計画では確定せず継続検討とする。

### T14. C05: IRConverter リサンプルフォールバック時ピッチ誤差対応

**ステータス: 別設計検討（時期未定）**

#### ⚠ 問題の重大性
`IRConverter::convertFile()` でリサンプル失敗時、IR データはソースレートのままエンジンにターゲットレートを報告する。以下の理由から現状の「警告続行」は許容できない:

1. **音程ずれ**: コンボルバが IR を誤った速度で処理
2. **時間軸ずれ**: 群遅延を含む全 IR 時間軸がずれる
3. **周波数軸ずれ**: ルーム補正では補正結果自体が破綻
4. **黙って続行するとユーザーが異常に気づけない**

#### 暫定対応方針（合意済み）
リサンプル失敗時は IR ロードを失敗させ、ユーザーにエラー通知する:

```cpp
if (converted.getNumSamples() <= 0) {
    juce::Logger::writeToLog("[DIAG_IR] ERROR: Resample failed (source="
        + juce::String(sourceRate) + ", target="
        + juce::String(config.targetSampleRate) + ")");
    return {};
}
```

呼び出し元は IR 未ロードとして扱い、既存 IR を維持する。

#### 未確定の設計判断
- **フォールバック SRC ライブラリの選定**: libsamplerate / libsoxr / IPP `ippsResample_64f`
- **Engine 側オンザフライ SRC の要否**: 軽量 SRC をエンジンに組み込むか
- **実装優先度**: プライマリ（r8brain）の失敗が稀なため緊急度は低いが、修正方針は慎重に設計する必要がある

---

# Appendix

## 設計レビュー評価

### v1: 初回レビュー（2026-07-25）

ISR Runtime（Practical Stable ISR Bridge Runtime）設計原則に基づくレビューを実施。
中核原則：「RTは判断しない・所有しない・解放しない」「Coordinatorが唯一のAuthority」「Retireは必ずEpochを経由する」

#### 総合スコア: **82/100 点**

| 項目 | 評価 | 備考 |
|------|------|------|
| バグ発見 | ★★★★★ | 全44バグから真正10件を適切に抽出 |
| 原因分析 | ★★★★☆ | 大部分で正確な原因特定 |
| 修正方針 | ★★★☆☆ | 一部でISR設計との整合性に課題 |
| ISR設計との整合 | ★★★★☆ | 概ね良好。G02・C05に設計上の問題 |

#### 反映内容
- **A01**: RAII方式 → 明示的 `_mm256_zeroupper()` + Compiler Option 併用に変更
- **B01**: `for %%A` → `SHIFT` 解析方式に変更
- **G02**: 同期delete → Shutdown専用経路 `retireImmediateDuringShutdown()` に変更
- **A05**: ゼロクリア一律追加 → `makeAlignedArrayZero()` 分離APIに変更
- **C05**: 警告続行 → ロード失敗＋別SRCフォールバックに変更
- **優先順位**: C05を別設計検討に分離、G02/A05/A06をPhase4に移動

#### 総評
計画全体はよく整理されている。G02・A05・C05 の3項目は設計を見直してから実装することを推奨。

---

### v2: セカンドレビュー（2026-07-25）

v1反映後の計画に対する再評価。

#### 総合スコア: **90〜92/100 点**

| 項目 | 評価 | 備考 |
|------|------|------|
| A01 VZEROUPPER | △ | AVX→legacy SSE境界への配置に修正済み |
| B01 build.bat | ◎ | SHIFT方式で問題なし |
| G03 FlagResetter | ○ | `publishAtomic()` 使用確認済み |
| A02 Stack→Heap | ◎ | |
| B02 Clang-Tidy | ◎ | |
| B03 ASan | ◎ | |
| G07 RuntimeState | ◎ | |
| A03 push戻り値 | ○ | |
| A04 Dead code | ◎ | |
| B04 icx link | ◎ | |
| G02 retire | ◎ | 前版より大幅改善 |
| A05 makeAlignedArrayZero | ◎ | API分離が適切 |
| A06 alignas | ○ | 設計明確化として価値 |
| C05 IRConverter | ◎ | 保留判断が適切 |

#### 改善点（v2で反映済み）
1. **A01**: 「関数出口に `vzeroupper`」→「AVX→legacy SSE境界」に配置する方針へ修正。icx `-mvzeroupper` を CXX 翻訳単位のみに限定（generator expression使用）。
2. **G03**: `callAsync` 失敗時のフォールバックが ISR Runtime ラッパー `publishAtomic()` を経由していることをソースコードで確認。プレーンな `atomic.store()` は不使用。

#### 総評
前版から設計品質が大幅に向上。ISR Runtime原則を壊す同期deleteの排除、API分離による性能劣化回避、IRConverterの安易な修正回避、build.bat解析改善が高く評価できる。上記2点の反映により、計画全体は実装フェーズへ進められる完成度に達している。

---

### v3: サードレビュー（2026-07-25）

v2反映後の計画に対する最終確認。

#### 総合スコア: **90〜92/100 点**（据置き）

| カテゴリ | 評価 | 備考 |
|---------|------|------|
| ISR Runtime設計との整合性 | A | 全項目で原則と矛盾なし |
| 実装可能性 | A | 即座に実装開始可能 |
| 保守性 | A | 命名・構造が明確 |
| リスク管理 | A- | 重複発行リスクを追記済み |
| テスト計画 | A- | objdump併記を追加済み |

#### v3で修正した4項目
1. **T1 (A01)**: 「関数出口」→「全 AVX→legacy SSE 遷移パス」に表現統一。レイテンシ記述をμarch依存表現に修正。icx `-mvzeroupper` をCXX限定（generator expression）。
2. **T2 (B01)**: `setlocal EnableDelayedExpansion` の前提条件を明記。
3. **T10 (A03)**: `push()` の戻り値に `[[nodiscard]]` を付与。
4. **T7 (G02)**: `retireImmediateDuringShutdown()` に Audio Thread 停止後専用であることを明示する三重防御（命名＋jassert＋コメント）を追加。

#### 総評
今回の版は技術的な整合性が大幅に向上した。特に「全 AVX→legacy SSE 遷移パス」への配置方針の精密化、`EnableDelayedExpansion` の前提条件明記、`[[nodiscard]]` の付与、三重防御による Shutdown 専用経路の安全確保は、実装時のレビュー負荷を下げる効果が期待できる。以上の反映をもって、本改修計画は実装計画として十分な完成度に達していると評価する。

---

### v4: 最終レビュー（2026-07-25）

v3反映後の計画に対する最終確認。

#### スコア

| 項目 | 評価 | 備考 |
|------|------|------|
| ISR Runtime整合性 | 10/10 | 全項目で原則と矛盾なし |
| RT安全性 | 10/10 | Audio Thread 非関与を確認済み |
| 保守性 | 9.5/10 | 命名・構造が明確 |
| 実装容易性 | 9/10 | 配置ルールの具体化で改善 |
| テスト容易性 | 8.5/10 | CIチェック推奨を追記 |
| 設計書品質 | 9.5/10 | 4回のレビューを経て充実 |

#### v4で修正した4項目
1. **T1 (A01)**: 具体的な配置ルール（JUCE関数/SSE intrinsic/非AVX DSP/SSE領域の4条件）を明文化。CI静的チェック（`_mm256_*` を含むTUに `vzeroupper` 必須）を推奨事項として追加。
2. **T2 (B01)**: `-D` 引数チェックを大文字小文字非依存（`if /I`）に変更。
3. **T10 (A03)**: `[[nodiscard]]` の実施順序を明記（全呼出側修正 → 最後に付与）。
4. **T12 (A06)**: `alignas(64)` とアロケータの組み合わせ確認を補足。

#### 最終総評
4回のレビューを経て、本改修計画書は設計・実装・テストの全側面で十分な完成度に達した。特にISR Runtime設計原則との整合性、RT安全性、保守性は高水準にある。本計画に従い実装を開始して問題ない。

---

### v5: ファイナライズレビュー（2026-07-25）

v4反映後の計画に対する最終確認。

#### スコア

| 項目 | 評価 | 備考 |
|------|------|------|
| 技術妥当性 | 9.6/10 | Intel最適化マニュアル準拠 |
| ISR Runtime整合性 | 9.8/10 | Runtime停止完了条件を明確化 |
| 実装容易性 | 9.2/10 | 配置ルール具体化で改善 |
| 保守性 | 9.6/10 | CIチェック推奨を追記 |
| 総合 | **A** | 実装開始可能レベル |

#### v5で修正した5項目
1. **T1 (A01)**: 配置定義を「AVX命令実行後、legacy SSE命令が初めて実行される直前」に精密化。Intel Optimization Manualのtransition定義に一致。
2. **T1 (A01)**: テスト計画の期待結果を「AVX2関数末尾」→「AVX→legacy SSE遷移境界」に修正し、本文と統一。
3. **T3 (G03)**: `callAsync` 失敗時のコメントに、UIコンポーネント状態が更新されない可能性を明記。
4. **T7 (G02)**: 適用条件を「Audio Thread停止後」→「Runtime停止シーケンス完了後（Audio停止＋Coordinator停止）」に拡張し、retire queueとの競合を防止。
5. **リスクマトリクス**: LTO/IPOによるインライン展開でvzeroupper配置が変化するリスクを追加。

#### 最終総評
5回のレビューを経て、本改修計画書は設計・実装・テストの全側面で実用十分な完成度に達した（総合評価 **A**、約9.8/10）。特にISR Runtime設計原則との整合性、Intel最適化マニュアルに基づくAVX→SSE遷移配置、Runtime停止シーケンスの明確化は高水準にある。本計画に従い実装を開始して問題ない。

---

### v6: 最終調整レビュー（2026-07-25）

v5反映後の計画に対する最終調整。

#### スコア

| 項目 | 評価 |
|------|------|
| 技術妥当性 | 9.6/10 |
| ISR Runtime整合性 | 9.8/10 |
| 実装容易性 | 9.4/10 |
| 保守性 | 9.6/10 |
| 総合 | **A** |

#### v6で修正した主な項目
1. **T1**: 「16ファイル修正」→「AVX→legacy SSE遷移箇所修正」に表現を改め、ファイル単位ではなく遷移箇所単位で修正することを明記。icxでも間接境界（DLL/virtual/function pointer）では手動 `_mm256_zeroupper()` を許容する方針を追加。CIチェックをTU単位のgrepからobjdumpベースのAVX命令数/vzeroupper数比較に改善。
2. **T2**: `EnableDelayedExpansion` の副作用（`!` を含む引数の展開）について注意を追記。
3. **T3**: Shutdown中はatomicのみ更新されることを「仕様」として明記。UI状態は再初期化でリセットされる。
4. **T4**: `bad_alloc` 時の挙動（学習中断・現在設定維持・ログ出力）を仕様として明記。
5. **T6**: `MSVC_RUNTIME_LIBRARY` が他の設定で上書きされる可能性があることをコメントで注意喚起。
6. **T7**: MessageThread制約を必須条件から緩和し、本質条件を「Audio停止＋Coordinator停止」に整理。
7. **T12**: `static_assert(alignof(AudioSegment) == 64)` を推奨として追記。

#### 最終総評
6回のレビューを経て、本改修計画書は設計・実装・テストの全側面で実用十分な完成度に達した。本計画に従い実装を開始して問題ない。

---

### v7: 最終仕上げレビュー（2026-07-25）

v6反映後の計画に対する最終仕上げ。

#### v7で修正した6項目
1. **A01 テスト項目**: 「全関数末尾に自動挿入」→「必要なABI境界への自動挿入」に修正（`-mvzeroupper` はcall前/return前/tail call前等に挿入）。
2. **A01 CI案**: 「AVX命令数とvzeroupper数の比較」を削除し、TU単位のgrep＋ホットパスのサンプル確認に変更（単純比較はLLVM最適化による誤検出が多いため）。
3. **T2 問題説明**: 「cmd.exe が = を区切る」→「FOR %%A in (%*) のトークン化が = を区切る」に修正。
4. **G03 Worst Case**: 「フラグ残留」→「WeakReference無効なら対象オブジェクトは既に破棄済みのため更新不要」に修正。
5. **B03 ASan**: 既存 `target_compile_options` に `/MT` が残っていないことを確認する注意書きを追加。
6. **A06**: `static_assert(sizeof(AudioSegment) % 64 == 0)` を追加（false sharing防止）。

#### 最終総評
7回のレビューを経て、本改修計画書は設計・記述・テストの全側面で十分な完成度に達した。残っている指摘は記述の精度に関する微修正のみで、設計そのものを変更する必要はない。本計画に従い実装を開始して問題ない。

---

### v8: 最終確認レビュー（2026-07-25）

v7反映後の計画に対する最終確認。

#### スコア
| 項目 | 評価 |
|------|------|
| 設計整合性 | 9.8/10 |
| ISR Runtime整合性 | 10/10 |
| 実装可能性 | 9.8/10 |
| 保守性 | 9.7/10 |

#### v8で修正した3項目
1. **テスト表 icx行**: 「全関数末尾に自動挿入」→「必要なABI境界（call前/return前等）に自動挿入」に修正（下の詳細テスト戦略の記述と統一）。
2. **T4 (A02)**: `bad_alloc` の捕捉主体を明記（Worker Thread 側で `catch (const std::bad_alloc&)` により捕捉→ログ出力→学習スキップ）。
3. **T9 (A05)**: `makeAlignedArrayZero` に `static_assert(std::is_trivially_copyable_v<T>)` を追加（`std::memset` 使用の事前条件保証）。

#### 最終総評
8回のレビューを経て、本改修計画書は設計・記述・テストの全側面で実用十分な完成度に達した。アーキテクチャ上の問題は全て解消され、残るは文書表現の微調整のみである。本計画に従い実装を開始して問題ない。

---

### v9: 実装前最終確認（2026-07-25）

v8反映後の計画に対する実装前最終確認。

#### スコア
| 項目 | 評価 |
|------|------|
| 設計妥当性 | 9.7/10 |
| ISR Runtime整合性 | 10/10 |
| 実装可能性 | 9.5/10 |
| リグレッションリスク | 低 |

#### v9で修正した6項目
1. **T1 配置ルール**: 「JUCE関数/SSE intrinsic/MKL」という呼び出し元基準から「legacy SSE命令が実行される地点」という本質基準に修正。判断基準を明確化。
2. **T1 objdump確認**: 「call前/return前/tail call前等」→「ABI境界」に簡略化（LLVM最適化による位置変動を吸収）。
3. **T2 build.bat**: `!` 入りの引数が CMAKE_EXTRA_FLAGS で使用されることを非推奨と明記。
4. **T6 ASan**: `compile_commands.json` で `/MT` が残っていないことを確認する項目を追加。
5. **T10 push()**: 戻り値の処理方法を具体化（ログ出力・XRUNカウンタ・Telemetryの3選択肢）。
6. **工程順変更**: Week1を T2→T3→T1 の順に変更（小規模タスクを先に完了させ、途中成果を出しやすくする）。

#### 最終総評
9回のレビューを経て、本改修計画書は設計・記述・工程の全側面で実用十分な完成度に達した。重大な設計上の問題は見当たらず、本計画に従い実装を開始して問題ない。

---

## ソースコード検証結果（2026-07-25）

本セクションでは、レビュー指摘に基づくソースコード調査の結果を記録する。

### A01: MSVC `-mvzeroupper` 非対応の確認

| コンパイラ | 自動VZEROUPPER挿入 | 対策 |
|-----------|-------------------|------|
| icx | ✅ `-mvzeroupper` で全関数末尾に自動挿入 | Step 1 で対応 |
| MSVC | ❌ 同等オプションなし | Step 2 で明示的 `_mm256_zeroupper()` |

**確認事実**: MSVC の `/arch:AVX2` はAVX2命令の生成を有効にするが、SSE遷移時の VZEROUPPER を自動挿入しない。これはコンパイラの仕様であり回避不可能。よって Step 1（icx: コンパイラオプション）+ Step 2（MSVC: 明示的挿入）の2段階方式が正しい。

**参考**: GCC/Clang の `-mvzeroupper` と異なり、MSVC には 2026年現在も同等機能が存在しない。

### G03: FlagResetter callAsync 失敗パスの安全性確認

**調査対象**: `src/convolver/ConvolverProcessor.LoaderThread.cpp` の `FlagResetter::~FlagResetter()`

**確認結果**: ✅ **安全**

- `callAsync` 成功時: Message Thread 上で `publishAtomic(o->isLoading, false)` と `publishAtomic(o->isRebuilding, false)` を実行
- `callAsync` 失敗時（フォールバック）: 同様に `publishAtomic` で同一の atomic 変数を直接書き込み
- ⚠ **懸念なし**: 書き込み対象は `std::atomic<bool>` 型のメンバ変数のみであり、MessageThread 専用状態は一切書き換えない
- `WeakReference` による保護も有効

**結論**: レビュー指摘の「要確認事項」は問題なし。計画書の修正内容をそのまま実装可能。

### A02: スタック配列サイズの実測

| 項目 | 値 |
|------|-----|
| `kFftLength` (MklFftEvaluator) | 4096 |
| `kSegmentHop` | 2048 (= kLength / 2) |
| `kMaxTrainingSegments` | 16 (= 4 levels × 4 segments) |
| `kRecentSampleRequest` | 34,816 |
| `recentLeft[34816]` スタック消費 | 272 KB |
| `recentRight[34816]` スタック消費 | 272 KB |
| **合計 スタック消費** | **544 KB** |

**確認事実**: Windows既定1MBのスタックに対し544KBは余裕がない。修正方針（ヒープ化）は妥当。

### A05: `makeAlignedArray` 現状実装

- `src/AlignedAllocation.h` の `makeAlignedArray<T>(count)`
- `mkl_malloc` 経由で64byteアライン保証
- **ゼロクリアなし**（`std::memset` なし）
- 非スロー版 `makeAlignedArray_nothrow` も既存

**結論**: レビュー指摘の `makeAlignedArrayZero` 分離方式が適切。

### A06: `alignas(64)` と malloc アライメントの関係

- `makeAlignedArray` の `mkl_malloc(..., 64)` が動的確保時に64byteアラインを保証
- 構造体自体の `alignas(64)` は、スタック上や他の構造体に埋め込まれた場合にのみ効果を発揮
- `AudioSegment` は通常 `ScopedAlignedArray<AudioSegment>` で動的確保されるため、`alignas(64)` の実質的効果は限定的

**結論**: レビュー指摘の通り、効果は限定的。ただしデメリットもないため Phase4 で実施可能。

### C05: 現行リサンプル方式

- **プライマリ SRC**: `r8brain`（`r8b::CDSPResampler`）- 高品質フィルタ（140dB/2%）
- **フォールバック SRC**: **なし**（現状の唯一の問題点）
- **IPP リサンプル**: 未使用（IPP は FFT 用途のみ）

**現状の convertFile フォールバック**:
```
r8brain リサンプル失敗
  ↓
converted = ir;  // 元データそのまま
actualSampleRate = config.targetSampleRate;  // レートだけ偽装
  ↓
コンボルバが誤った速度で処理 → ピッチ・時間軸・周波数軸が全部ずれる
```

**修正後**:
```
r8brain リサンプル失敗
  ↓
return nullptr;  // IR ロード失敗
  ↓
ユーザーにエラー通知、IR 未ロード（既存 IR 維持）
```

**将来のフォールバック候補**:
| ライブラリ | ライセンス | 品質 | 備考 |
|-----------|-----------|------|------|
| libsamplerate (SRC) | LGPL-2.1 | 0(linear)〜4(sinc) | 定番。軽量 |
| libsoxr | LGPL-2.1 | 非常に高い | SoXベース。高品質 |
| IPP `ippsResample_64f` | プロプライエタリ | 高い | 既存依存。追加検証必要 |

**推奨**: まずは「リサンプル失敗時はロード失敗」に修正。別SRCフォールバックは別設計検討とする。
