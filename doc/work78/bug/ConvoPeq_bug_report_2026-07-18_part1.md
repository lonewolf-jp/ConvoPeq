# ConvoPeq ソースコード調査報告書（Part 1）

- 対象: `ConvoPeq.md`（2026-07-18 19:07:05 生成、76,677行 / 265ファイル）
- 対象コミット相当: GitHub `lonewolf-jp/ConvoPeq` main（添付Markdown時点）
- 調査者: Claude（Anthropic）
- 調査方針: 静的読解 + クロスリファレンス検証（推測のみでの指摘は行わず、書込み側と読出し側を突き合わせる等、実コードで裏取りできたものだけを「確定バグ」として計上）

---

## 0. サマリ

| # | 重大度 | ファイル | 概要 |
|---|--------|----------|------|
| 1 | **Low**（診断機能限定・音声出力に影響なし） | `AudioEngine.Processing.AudioBlock.cpp`<br>`AudioEngine.Processing.BlockDouble.cpp` | `CallbackTimingHistory` リングバッファの書込みインデックス計算にオフバイワン。全書込みが本来の位置から1つずれる。 |
| 2 | **Medium**（コーディング規約違反 + noexcept関数からのstd::terminateリスク） | `IRAnalyzer.cpp` | MKL関連バッファに`std::make_unique<double[]>`を使用（規約違反）。加えて`noexcept`関数内でbad_allocを投げ得る割当を行っており、OOM時にプロセス全体が`std::terminate()`する経路が存在。 |

いずれも **Audio Thread のリアルタイム処理そのものには影響しません**（#1は診断ビルドのみ、#2はIRロード時の非RTスレッド処理）。ただし#2は明確な規約逸脱であり、将来同種のコピペが増えると危険度が上がるため早めの修正を推奨します。

このほか、疑わしく見えて実際には問題なしと確認できた箇所（誤検知の回避）を第2部に列挙しています。他のAIに引き継ぐ際は、まずこちらの「確定バグ」と「検証済みで問題なし」のリストをご参照いただくと、重複調査を避けられます。

---

## 1. 【Low】CallbackTimingHistory リングバッファのオフバイワン

### 該当ファイル・行

- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`（`AudioEngine::getNextAudioBlock` 内、ローカル719-720行目）
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`（同型処理、ローカル674-675行目）
- 読出し側: `src/audioengine/AudioEngine.Timer.cpp`（ローカル1342-1355行目付近）

いずれも `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` の診断ビルドでのみ有効になるコードです（デフォルトOFF）。

### 現在のコード（書込み側、2箇所とも同一パターン）

```cpp
    // ★ B: CallbackTimingHistory リングバッファ書込
    {
        const uint64_t endUs = convo::getCurrentTimeUs();
        const uint64_t processTime = callbackTelemetry.startUs > 0
            ? (endUs > callbackTelemetry.startUs ? endUs - callbackTelemetry.startUs : 0)
            : 0;
        const uint64_t wc = convo::fetchAddAtomic(
            rtLocalState_.callbackTimingWriteCount,
            uint64_t{1}, std::memory_order_relaxed);
        const size_t idx = static_cast<size_t>(
            (wc - 1) % RTLocalState::kCallbackTimingSlots);
        auto& entry = rtLocalState_.callbackTimingHistory[idx];
        entry.callbackIndex = thisCallbackIndex;
        entry.processTimeUs = processTime;
        entry.driftUs = rtLocalState_.lastCallbackDriftUs.load(
            std::memory_order_relaxed);
        entry.cpu = rtLocalState_.lastCallbackProcessor.load(
            std::memory_order_relaxed);
```

読出し側（`AudioEngine.Timer.cpp`）:

```cpp
            const uint64_t wc = rtLocalState_.callbackTimingWriteCount.load(
                std::memory_order_relaxed);
            if (wc != rtAuxMutable_.lastCbHistDumpedWriteCount)
            {
                rtAuxMutable_.lastCbHistDumpedWriteCount = wc;
                if (s_xRunPopCount > 0)
                {
                    const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
                const uint64_t start = (wc > RTLocalState::kCallbackTimingSlots)
                    ? wc - RTLocalState::kCallbackTimingSlots : 0;
                for (uint64_t i = start; i < wc; ++i)
                {
                    const size_t idx = i % RTLocalState::kCallbackTimingSlots;
                    const auto& e = rtLocalState_.callbackTimingHistory[idx];
```

（`kCallbackTimingSlots = 32`、`AudioEngine.h` にて定義）

### 問題点

`convo::fetchAddAtomic` は `std::atomic::fetch_add` と同じく **加算前（pre-increment）の値**を返します。つまり1回目の呼び出しで `wc = 0`、2回目で `wc = 1`……という**インクリメント前=0始まりのインデックスそのもの**が返ります。

一方、読出し側は「`callbackTimingWriteCount` の現在値 `wc`」を「これまでの書込み回数」として扱い、`i = start..wc-1` の範囲を `i % 32` でスキャンします。この読出し側のロジックは、書込み側が **`wc % 32`（インデックスそのもの）** を使っている前提で設計されています。

しかし実際の書込み側コードは `(wc - 1) % 32` と、さらに `-1` した値を使っています。この結果：

- 全書込みが本来のスロットより1つ前にずれて格納される。
- 特に **診断有効化後の最初の呼び出し**（`wc = 0`）では `(0 - 1)` が `uint64_t` の下限からラップアラウンドして `UINT64_MAX % 32 = 31` となり、スロット31に書き込まれます。これは読出し側が「リングバッファがまだ32回未満しか埋まっていない」間（`wc <= 32`）は範囲外として読まないスロットのため、**起動直後32コールバック分のCB_HISTダンプから最初のエントリが欠落**します。
- 起動後32回転目以降は「ズレたまま」全スロットを一巡するため、実害は「表示される順序・対応関係が1個ずれる」程度に収まりますが、`entry.sequence`（0初期値）を使った有効判定と組み合わさっており、意図した「直近32件」とは異なる中身を表示し得ます。

**音声処理そのものへの影響はありません**（このブロックは`std::atomic`のfetch_addとインデックス計算のみで、audio thread規約への違反もありません）。影響は診断ビルドでの `CB_HIST` ダンプ内容が不正確になる点に限られます。

### 検証根拠

- 書込み側: `fetchAddAtomic` の実装（`AtomicAccess.h`）が `std::atomic_fetch_add_explicit` の戻り値（pre-increment値）をそのまま返すことを確認。
- 読出し側: `AudioEngine.Timer.cpp` の `idx = i % kCallbackTimingSlots`（`-1`なし）というロジックと突き合わせ、書込み側にのみ余分な `-1` があることを確認。
- 同一パターンが `AudioEngine.Processing.AudioBlock.cpp` と `AudioEngine.Processing.BlockDouble.cpp` の双方に存在することを`grep`で確認済み（コピペ由来と推測されます）。

### 修正パッチ

`(wc - 1) % RTLocalState::kCallbackTimingSlots` → `wc % RTLocalState::kCallbackTimingSlots` に変更するだけです。2箇所とも同一修正。

```diff
--- a/src/audioengine/AudioEngine.Processing.AudioBlock.cpp
+++ b/src/audioengine/AudioEngine.Processing.AudioBlock.cpp
@@ -707,10 +707,10 @@
     // ★ B: CallbackTimingHistory リングバッファ書込
     {
         const uint64_t endUs = convo::getCurrentTimeUs();
         const uint64_t processTime = callbackTelemetry.startUs > 0
             ? (endUs > callbackTelemetry.startUs ? endUs - callbackTelemetry.startUs : 0)
             : 0;
         const uint64_t wc = convo::fetchAddAtomic(
             rtLocalState_.callbackTimingWriteCount,
             uint64_t{1}, std::memory_order_relaxed);
         const size_t idx = static_cast<size_t>(
-            (wc - 1) % RTLocalState::kCallbackTimingSlots);
+            wc % RTLocalState::kCallbackTimingSlots);
         auto& entry = rtLocalState_.callbackTimingHistory[idx];
         entry.callbackIndex = thisCallbackIndex;
         entry.processTimeUs = processTime;
```

```diff
--- a/src/audioengine/AudioEngine.Processing.BlockDouble.cpp
+++ b/src/audioengine/AudioEngine.Processing.BlockDouble.cpp
@@ -662,10 +662,10 @@
     // ★ B: CallbackTimingHistory リングバッファ書込
     {
         const uint64_t endUs = convo::getCurrentTimeUs();
         const uint64_t processTime = callbackTelemetry.startUs > 0
             ? (endUs > callbackTelemetry.startUs ? endUs - callbackTelemetry.startUs : 0)
             : 0;
         const uint64_t wc = convo::fetchAddAtomic(
             rtLocalState_.callbackTimingWriteCount,
             uint64_t{1}, std::memory_order_relaxed);
         const size_t idx = static_cast<size_t>(
-            (wc - 1) % RTLocalState::kCallbackTimingSlots);
+            wc % RTLocalState::kCallbackTimingSlots);
         auto& entry = rtLocalState_.callbackTimingHistory[idx];
         entry.callbackIndex = thisCallbackIndex;
         entry.processTimeUs = processTime;
```

読出し側（`AudioEngine.Timer.cpp`）は修正不要です（`i % kCallbackTimingSlots` のままで正しい）。

---

## 2. 【Medium】IRAnalyzer.cpp: MKL関連バッファへの `std::make_unique` 使用（規約違反）+ noexcept関数からの例外送出リスク

### 該当ファイル

`src/IRAnalyzer.h` / `src/IRAnalyzer.cpp`（今回の7/18版で新規追加されたファイル。7/12版のプロジェクトファイルには存在しません）

### 問題点（2つ複合）

**(a) コーディング規約違反**

本プロジェクトの規約:

> Audio thread以外で、かつINTEL oneAPI MKL(oneMKL)使用箇所では、new、std::vector、std::make_uniqueを使用せず、かわりにmkl_malloc / mkl_free、_aligned_malloc(64)、std::pmr + custom allocatorを使用してください。

`IRAnalyzer.cpp` は `#include <mkl_dfti.h>` して `DftiCreateDescriptor`/`DftiComputeForward` 等のMKL DFTI APIを直接呼び出しているにもかかわらず、FFT用の作業バッファ4箇所すべてで `std::make_unique<double[]>`（`operator new[]`経由、非64byteアライメント）を使用しています。

該当箇所（現状のコード、`IRAnalyzer.cpp` ローカル行番号）:

```cpp
    29:    auto tukeyWindow = std::make_unique<double[]>(static_cast<size_t>(fftSize));
    69:        auto in = std::make_unique<double[]>(static_cast<size_t>(fftSize));
    87:        auto out = std::make_unique<double[]>(static_cast<size_t>(fftSize));
   124:            auto mags = std::make_unique<double[]>(static_cast<size_t>(numBins + 1));
```

他の全ファイル（`TruePeakDetector.cpp`、`CacheManager.cpp` など）は一貫して `convo::makeAlignedArray<double>(...)`（`AlignedAllocation.h` で定義された64byteアライメント・`mkl_malloc`/`mkl_free`ベースのRAIIラッパー）を使用しており、本ファイルだけが規約から外れています。

**(b) `noexcept` 関数からの `std::bad_alloc` 送出リスク**

`estimateMaxFrequencyResponseGain` は `IRAnalyzer.h` で `noexcept` 宣言されていますが、`std::make_unique<double[]>` はメモリ確保失敗時に `std::bad_alloc` を送出します。`noexcept` 関数内で例外が送出されると `std::terminate()` が呼ばれ、**アプリケーション全体が即死**します。

この関数自身は他の全ての異常系（`numSamples<=0`、`fftSize<2`、`windowMean<1e-18` 等）に対して「安全なデフォルト値 `1.0` を返す」という一貫したフェイルセーフ設計になっているため、メモリ確保失敗だけが `std::terminate()` という非対称な挙動になっている点は設計意図とも矛盾します。

参考: 同様に `convo::makeAlignedArray` を使う `TruePeakDetector::prepare()` は `noexcept` を付けていません（`makeAlignedArray`が`bad_alloc`を投げ得ることを踏まえた一貫した設計）。`IRAnalyzer` はこの点でプロジェクト内の他の実装パターンとも異なります。

### 検証根拠

- `grep`で `IRAnalyzer.cpp`/`.h` 全体を確認し、MKL API呼び出し（`DftiCreateDescriptor`等）と `std::make_unique<double[]>` が同一関数内に同居していることを直接確認。
- `AlignedAllocation.h` の `makeAlignedArray<T>()` 実装を確認し、`aligned_malloc`（64byte, `mkl_malloc`経由）を使うプロジェクト標準のドロップイン代替であることを確認。
- `estimateMaxFrequencyResponseGain` の呼び出し元（`IRConverter.cpp` の `analyzeIR()`）を確認し、IRロード時の非RTスレッド処理であることを確認（Audio Thread規約そのものへの抵触ではなく、あくまで「MKL関連バッファのアロケータ規約」違反）。

### 注意点（パッチ作成時に気づいた副次的な罠）

`std::make_unique<double[]>(n)` は**値初期化**されるため、確保直後は全要素 `0.0` です。元コードの `in` バッファは「後半をゼロパディングする」ことをこの暗黙のゼロ初期化に依存しています（コメント「残りはゼロパディング（デフォルト値0.0）」はあるが、実際にゼロを書き込むコードはありません）。

一方 `mkl_malloc`（および `convo::makeAlignedArray` はその薄いラッパー）は**ゼロ初期化を行いません**。単純に `make_unique` → `makeAlignedArray` に置換するだけだと、ゼロパディング領域が未初期化メモリになり、FFT結果が不定になる新規バグを生みます。下記パッチでは明示的なゼロ埋めループを追加してこれを回避しています。この「`make_unique<T[]>`との暗黙のゼロ初期化差」は、他ファイルで同種の置換を行う際にも注意が必要です。

### 修正パッチ（unified diff）

```diff
--- a/src/IRAnalyzer.cpp
+++ b/src/IRAnalyzer.cpp
@@ -1,5 +1,6 @@
 #include "IRAnalyzer.h"
 #include "DftiHandle.h"
+#include "AlignedAllocation.h"
 #include <algorithm>
 #include <numeric>
 #include <cmath>
@@ -10,6 +11,7 @@
 double estimateMaxFrequencyResponseGain(
     const juce::AudioBuffer<double>& ir) noexcept
 {
+  try {
     const int numSamples = ir.getNumSamples();
     const int numChannels = ir.getNumChannels();
     if (numSamples <= 0 || numChannels <= 0)
@@ -26,7 +28,11 @@
     const double pi = juce::MathConstants<double>::pi;
     const double taperLen = kTukeyAlpha * static_cast<double>(fftSize - 1) * 0.5;
 
-    auto tukeyWindow = std::make_unique<double[]>(static_cast<size_t>(fftSize));
+    // ★ FIX: std::make_unique<double[]> は operator new[] を使用するため、
+    //   Audio Thread以外でもMKL関連バッファにnew/make_uniqueを使用しない
+    //   というプロジェクト規約に反する。convo::makeAlignedArray に置換し、
+    //   64byteアライメント + mkl_malloc/mkl_free 経由の確保・解放に統一する。
+    auto tukeyWindow = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));
     for (int i = 0; i < fftSize; ++i)
     {
         const double t = static_cast<double>(i);
@@ -66,10 +72,12 @@
         const double* src = ir.getReadPointer(ch);
 
         // 入力バッファ: 窓適用 + ゼロパディング
-        auto in = std::make_unique<double[]>(static_cast<size_t>(fftSize));
+        auto in = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));
         for (int i = 0; i < copyLen; ++i)
             in[i] = src[i] * tukeyWindow[i];
         // 残りはゼロパディング（デフォルト値 0.0）
+        for (int i = copyLen; i < fftSize; ++i)
+            in[i] = 0.0;
 
         // DFTI 記述子作成（実数→複素: 出力長 fftSize/2 + 1）
         convo::ScopedDftiDescriptor dfti;
@@ -84,7 +92,7 @@
             continue;
 
         // 出力: 複素数値 (実部, 虚部) のインターリーブ、長さ fftSize
-        auto out = std::make_unique<double[]>(static_cast<size_t>(fftSize));
+        auto out = convo::makeAlignedArray<double>(static_cast<size_t>(fftSize));
         if (DftiComputeForward(dfti.handle, in.get(), out.get()) != DFTI_NO_ERROR)
             continue;
 
@@ -121,7 +129,7 @@
         //   但し単一正弦波近似であるため、複雑スペクトルでは限界あり
         {
             // CCS で振幅配列を作り直す（簡易版: DC と Nyquist 除く中間 bin のみ）
-            auto mags = std::make_unique<double[]>(static_cast<size_t>(numBins + 1));
+            auto mags = convo::makeAlignedArray<double>(static_cast<size_t>(numBins + 1));
             mags[0] = std::abs(out[0]);
             for (int b = 1; b < numBins; ++b)
             {
@@ -156,7 +164,12 @@
     maxMagnitude /= windowMean;
 
     return (maxMagnitude > 1e-18) ? maxMagnitude : 1.0;
+  } catch (...) {
+    // ★ FIX: makeAlignedArray/ScopedDftiDescriptor は理論上 std::bad_alloc を
+    //   投げ得るため、noexcept 関数からの意図しない std::terminate() を防止し、
+    //   本関数の既存の「異常時は安全側 1.0 を返す」契約を維持する。
+    return 1.0;
+  }
 }
 
 } // namespace IRAnalyzer
```

`IRAnalyzer.h` 側の変更は不要です（シグネチャ・`noexcept`はそのまま維持できます。関数内部で全例外を捕捉するため）。

---

## 3. 第2部について

以下は分量の都合上、別ファイルにまとめます（続けて出力します）:

- 検証したが問題なしと判断した箇所（誤検知の記録。二重調査防止のため）
- 未調査・調査が浅い領域のマップ（265ファイル中、重点的に読んだファイルと、まだ読めていないファイルの一覧）
- 次の調査候補
