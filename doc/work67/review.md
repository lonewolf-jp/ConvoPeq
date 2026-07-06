# ConvoPeq ソースコード総合監査 — プロ仕様オーディオソフトウェアとして適切でない実装状態

前回のセッションで指摘したビルドフラグバグ（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`常時1化）と`processDouble()`内の無条件ファイルI/O（`diagStartCapture`/`diagWriteCapture`）は、**現在のファイルでは修正済み**であることを確認しました（`DiagnosticsConfig.h`は`#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 0`に修正、ANS_SWITCHも`juce::Logger::writeToLog()`直接呼び出しから`DiagEvent`リングバッファpushへ置き換え済み）。RTホットパス（`Add`/`Get`/`processDouble`/`process`各関数）を実際に追跡した限り、新規の割り当て・ロック・同期I/Oは検出されませんでした。

その前提で、今回は全体を通読し、新たに以下の問題を確認しました。

---

## 1. ビルド構成に起因するリスク

### 1-1. AVX2/FMA3必須化にランタイムCPU機能検出が存在しない（確定・重大）

`CMakeLists.txt`:
```cmake
# AVX2を全コンフィグで有効化
target_compile_options(ConvoPeq PRIVATE /arch:AVX2)
...
# /QxCORE-AVX2: Haswell以降のAVX2+FMA必須、Intel専用コード生成
set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi")
```

`/arch:AVX2`（MSVC）・`/QxCORE-AVX2`（icx）が**ターゲット全体・全コンフィグに無条件適用**されています。ソース全体を検索しましたが、`IsProcessorFeaturePresent`・`__cpuid`等によるランタイムCPU機能検出は**どこにも存在しません**。

- Haswell以前のIntel CPU、Excavator/Zen以前のAMD CPUでは、実行開始直後または最初にAVX2/FMA命令に到達した時点で`STATUS_ILLEGAL_INSTRUCTION`（SIGILL相当）で**即クラッシュ**します。エラーダイアログすら出ません。
- 対応: 起動時に`IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE)`相当のチェックを行い、非対応CPUでは分かりやすいエラーメッセージを出す（最低限）。理想的にはSSE2ベースラインビルド＋実行時ディスパッチ（関数ポインタ／IFUNC相当）でAVX2コードパスを選択する構成にする。

### 1-2. `/fp:fast` と `std::isnan()` によるNaNガードの併用（確定・重大）

`LatticeNoiseShaper.h`（アダプティブノイズシェイパの係数クランプ、RTスレッドから毎バンク切替時に呼ばれる）:

```cpp
static inline double clampCoeff(double value) noexcept
{
    constexpr double kLimit = 0.85;
    if (std::isnan(value))   // ← /fp:fast (icxのLLVM fast-math) 下では
        return 0.0;          //    最適化で消去されるリスクがある
    ...
}
```

呼び出し経路: `DSPCore::processDouble()`/`processFloat()`（30780行目付近／31743行目付近）→ `adaptiveNoiseShaper.applyMatchedCoefficients()` → `clampCoeff()`。

一方で同じコードベースの他の箇所（`DSPCoreDouble.cpp`のNaN/Infスクラブ処理等）では、まさにこの問題を回避するために**ビットパターン判定の自前関数**（`isFiniteNoLibm`）を使っています：

```cpp
inline bool isFiniteNoLibm(double x) noexcept
{
    union { double d; uint64_t u; } v { x };
    return ((v.u >> 52) & 0x7FFu) != 0x7FFu;
}
```

つまり**同一プロジェクト内でNaN検出方式が統一されていません**。icxビルド（Clang/LLVM系）の`/fp:fast`は`nnan`系のfast-math属性を伴うことがあり、`std::isnan()`が定数畳み込みで`false`に最適化される可能性があります。これが起きた場合、係数破損時のフォールバック（0.0への丸め込み）が働かず、ノイズシェイパの反射係数がNaNのまま伝播 → フィルタ発振・爆音（DC/ノイズバースト）につながり得ます。ノイズシェイパの安定性を守る「最後の砦」がコンパイラフラグ次第で無効化されうるという点で、プロ仕様としては看過できません。
- 対応: `clampCoeff`系のNaN判定を`isFiniteNoLibm`と同様のビットパターン判定に統一する、または当該関数のみ`#pragma float_control(precise, on)`（MSVC）/ `__attribute__((optnone))`相当で保護する。

---

## 2. RT設計上の懸念

### 2-1. Double precision処理の無条件強制と、事実上デッドコード化したfloatパス

`MainWindow.cpp`:
```cpp
audioProcessorPlayer.setDoublePrecisionProcessing(true);
audioProcessorPlayer.setProcessor(audioEngineProcessor.get());
```

これにより`AudioEngineProcessor::processBlock(float&, ...)`（`AudioEngine::getNextAudioBlock`）は**実運用では呼ばれず**、`processBlock(double&, ...)`（`AudioEngine::processBlockDouble` → `DSPCore::processDouble()`）が唯一の実行経路になります。ASIOはfloat32でI/Oするため、JUCEが毎コールバックfloat→double→floatの変換を行い、EQ/畳み込み/オーバーサンプリング/ディザ全段がdouble精度で常時実行されます。

構造的な問題点:
- **`AudioEngine.Processing.AudioBlock.cpp`（float版）と`AudioEngine.Processing.BlockDouble.cpp`（double版）が、XRUN検出・クロスフェード分岐・テレメトリまで含めてほぼ同一内容（各700行前後）で二重実装**されています。実運用で通るのはdouble版のみのため、float版は事実上「テストされないコード」のまま保守対象であり続けます。
- これは仮説ではなく実証済みのリスクです。前回セッションで見つかった`diagStartCapture`/`diagWriteCapture`の無条件ファイルI/Oバグは**double版にのみ**存在し、float版には存在しませんでした。「本番で通る経路にだけ手を入れた結果、使われない方の経路と乖離する」という典型的な二重実装の実害が既に一度発生しています。
- 対応: (a) double精度が本当に必要か再検討し、不要ならfalseにしてfloat単一パスに一本化する。(b) 本当に必要なら、float版を削除するか、共通のテンプレート実装（`template<typename SampleType>`）に統合して二重保守を解消する。

### 2-2. 常時稼働の全バッファNaN/Infスクラビング

`DSPCore::processDouble()`では、コンボルバ処理の前後で毎ブロック・全サンプルに対しAVX2マスク演算によるNaN/Inf走査＋ゼロ化を実施しています。これは「異常時の保険」としては妥当ですが、**常時・全サンプルに対して恒久的なCPU税として組み込まれている**点が気になります。プロ仕様DSPの一般的な設計では、NaN/Infガードは（a）パラメータ変更・IR切替などの不連続点、または（b）フィードバック構造を持つ段（IIR、ソフトクリップの`prevSampleInOut`等）の出力にのみ局所的に配置し、無条件経路（入力信号がそもそも有限であることが保証された区間）には置かないのが一般的です。全パスに毎回配置されているのは、根本的な数値安定性が解析的に保証されていないことの裏返しである可能性があり、コードスメルとして指摘します。

---

## 3. 例外・エラーハンドリング設計

### 3-1. 例外を投げるアロケータに型レベルでのRT安全性の区別がない

`AlignedAllocation.h`の`aligned_malloc` / `aligned_make_unique` / `makeAlignedArray` はすべて失敗時に`std::bad_alloc`を送出する汎用関数です。「RTスレッドから呼んではいけない」という制約は`makeAlignedCopy`にのみ日本語コメントで書かれているだけで、関数名・型システムでの強制はありません。

現状、RTホットパス（`Add`/`Get`/`process`/`processDouble`等）を実際に検索した限りではこれらの呼び出しは見つからず、**現時点でRTスレッド内から例外が送出される具体的な経路は確認できませんでした**。ただし、この設計のままでは将来の変更で誰かが「バッファが足りないので`makeAlignedArray`で拡張しよう」とRTパスに書いてしまっても、コンパイル時・レビュー時に検出する仕組みがありません。RTオーディオコールバック内で例外が送出されると、スタック巻き戻し自体が非決定的時間を要するうえ、多くのホスト実装では未捕捉例外がプロセスクラッシュに直結します。
- 対応: 非スロー版（`nothrow`でnullptr返却）を`_rt`サフィックス等で明示的に分離し、RTファイル群では非スロー版のみを`#include`できるようにする、あるいはclang-tidyのカスタムチェックでRT関数（`noexcept`かつ特定ファイル）からの`aligned_malloc`系呼び出しを検出する。

### 3-2. `jassert`のみに依存した重要不変条件（Release版では無防備）

`MKLNonUniformConvolver::Add()`:
```cpp
jassert(consumed <= numSamples);
```
`jassert`はNDEBUG（Release）で無効化されるマクロです。ユーザーの手元に届くのはReleaseビルドであり、この不変条件が破れた場合（将来のリファクタでオフセット計算にバグが混入した場合など）、Release版では**何のチェックもなく未定義動作（バッファオーバーラン等）に直結**します。同様のパターンが畳み込みエンジンの他の不変条件チェックにも見られます。プロ仕様のRTコードでは、Debug用`jassert`と対になる形で、Release版でも軽量な範囲チェック＋安全側フォールバック（clampして継続、または該当レイヤーをスキップ）を併置するのが望ましい設計です。

---

## 4. 開発プロセス上の所見

- 既に確認した「診断フラグが常に1に強制される」バグと「RTスレッド内の無条件ファイルI/O」は、いずれも`work52`/`work60`等のタグが付いた**調査用デバッグコードがそのままRTパスに残存**したことに起因していました。これは一度きりの偶発ミスではなく、構造的に起こりやすいパターンだと考えられます。
- `clang-tidy`統合は`option(CONVOPEQ_ENABLE_CLANG_TIDY ... OFF)`でデフォルト無効、かつ`// NOLINT(rt-logger)`という独自チェック名が使われていることから、ロガー呼び出しを検出するカスタムルールは存在するものの、(a) デフォルトでビルド時に実行されない、(b) `fopen_s`/`fwrite`のような生ファイルI/Oまではおそらくカバーしていない、という2点で網羅性に穴があります。
- 対応: `CONVOPEQ_ENABLE_CLANG_TIDY`をCIパイプラインで強制ON にする。`rt-logger`チェックの対象を「ロガー呼び出し」から「`noexcept`かつRTディレクトリ内関数からの、ファイルI/O・ヒープ確保・mutexロックを含む全呼び出し」に拡張する。

### 4-1. 軽微: 未使用になったdiagLog関数の残存

`DSPCoreDouble.cpp`の`diagLog()`は`[[maybe_unused]]`が付与されており、実際に呼び出し箇所は現在ゼロです（ANS_SWITCHの修正で不要になった）。機能上の実害はありませんが、単純なデッドコードなので削除を推奨します。

---

## 5. 確認できた良好な実装（公平のため付記）

- `EQProcessor::calcSVFCoeffs`は、サンプルレート不正値・Q/周波数/ゲインのクランプ、`std::pow`/`std::tan`を含む計算がMessage Thread専用であることの明記など、RTとMessage Threadの責務分離が明確です。
- IR読み込み時のリサンプルは、固定マージンのハードコードではなくr8brainの`getMaxOutLen()`を正しく使用しており、出力バッファ不足のリスクはありません。
- `MKLNonUniformConvolver::Add`/`Get`は実際に確認した範囲でヒープ確保・ロック・ログ出力が一切なく、AVX2アライメントチェック付きの丁寧な実装でした。
- 前回指摘した2件（ビルドフラグバグ／RTスレッド内ファイルI/O）は適切に修正されており、ANS_SWITCHはリングバッファ方式に統一されています。