# ConvoPeq 統合バグリスト 【検証済み版】

作成日: 2026-07-24
検証元: `doc/work85/bug.md`（20項目：ソースコード監査） + `doc/work85/bug2.md`（6項目：ビルドシステム＋論理バグ）
検証ツール: serena MCP server, AiDex MCP server, grep/ast-grep/rg(WSL), semble, 実コード直接読み取り

---

## 凡例

| マーク | 意味 |
|--------|------|
| ✅ 確定 | 確認済みの真正バグ |
| ⚠ 注意 | 部分的対応済み、または条件付きで問題あり |
| ❌ 誤報 | 元報告の記述が現状コードと一致しない |
| ❓ 未検証 | 未調査 |

## 優先度定義

| 優先度 | 基準 |
|--------|------|
| P0-Critical | クラッシュ・オーディオ破綻・データ損失の可能性 |
| P1-High | 機能不全・ビルド失敗（特定条件） |
| P2-Medium | 品質低下・潜在的問題 |
| P3-Low | 予防的・コード品質 |

---

## A. DSP・並行処理・クラッシュ関連

### A01. `_mm256_zeroupper` 欠如（旧bug.md #10）
**状態: ✅ 確定** | **優先度: P1-High**

`_mm256_zeroupper` または `_mm256_zeroall` の呼び出しが全く存在しない。AVX2→SSE遷移で最大100サイクル超のペナルティが発生する可能性がある。

AVX2命令を含むファイル（確認16ファイル）:
- **Audio Thread直接**（8cpp）: `AudioEngine.Processing.DSPCoreDouble.cpp`, `AudioEngine.Processing.DSPCoreFloat.cpp`, `AudioEngine.Processing.DSPCoreIO.cpp`, `EQProcessor.Processing.cpp`, `MKLNonUniformConvolver.cpp`, `CustomInputOversampler.cpp`, `TruePeakDetector.cpp`, `LoudnessMeter.cpp`
- **Audio Thread間接**（2cpp）: `ConvolverProcessor.Runtime.cpp`, `ConvolverProcessor.LoaderThread.cpp`
- **UI Thread**（2cpp）: `AudioEngine.EQResponse.cpp`, `SpectrumAnalyzerComponent.cpp`
- **ヘッダー**（4h）: `LatticeNoiseShaper.h`, `InputBitDepthTransform.h`, `DspNumericPolicy.h`, `dsp/math/FastTanhApprox.h`

なお元報告の「4ファイル」は過小評価。`OutputFilter.cpp` と `FixedNoiseShaper.h` はSSE2のみ（AVX2不使用）。

また `_mm256_load_ps` は存在しない（`_mm256_loadu_ps` / `_mm256_loadu_pd` が使用されている）。`_mm256_store_ps`（aligned store）が `SpectrumAnalyzerComponent.cpp` で使用されているが、`mags` は `alignas(64)` のローカル配列で適切に整列されている。`_mm256_store_pd`（aligned store）も `EQProcessor.Processing.cpp` で使用され、`temp` は `alignas(32)`（_mm256 の最低要件は32バイトアライン）で適切に整列されている。

**修正**: AVX2命令を含む全関数の出口に `_mm256_zeroupper()` を追加。MSVC/icx は `/arch:AVX2` 指定のみでは自動挿入しないため、明示的な発行が必須。clang-cl では `-mvzeroupper` フラグで自動挿入可能だが、現状のコンパイラ構成（MSVC / icx）では手動挿入が唯一の方法となる。

---

### A02. `NoiseShaperLearner::buildTrainingSegments()` スタック配列 ＋ `AudioSegmentBuffer` メモリ（旧bug.md #1, #18派生）
**状態: ⚠ 注意（2件）** | **優先度: P2-Medium**

**1. スタック配列（544KB、ただしゼロ初期化済み）:**
`double recentLeft[34816] = {}` + `double recentRight[34816] = {}` = 合計544KBのスタック使用。`= {}` による値初期化で内容はゼロクリアされているため未初期化データのリスクはない。ただしWindows既定1MBのスタックに対して半分超を占有するため、深いコールスタックと組み合わさった場合のガードページ違反リスクがある。`ScopedAlignedArray<double>`（ヒープ）への変更を推奨。

なお `evaluatePopulation()` 内の `tanhBuffer` は `kPopulation(18) × kDim(9) = 162要素 = 1.3KB` であり元報告の128KBは誤り。

**2. `AudioSegmentBuffer` クラスメンバ配列（58.6MB）:**
`AudioSegmentBuffer.h` に `double leftSamples[3840000]` + `double rightSamples[3840000]` のメンバ配列（`kCapacity = 5秒 × 768000Hz = 3,840,000`）。各 **29.3MB**（3,840,000 × 8bytes）、合計 **58.6MB** のメモリをクラス定義に内包する。これはヒープ上の `NoiseShaperLearner` オブジェクト内に確保されるためスタック問題はないが、1インスタンス58.6MBは設計上の注意点。実際にセグメントバッファが満杯になるのはサンプルレート48kHzの場合 `5×48000=240,000`サンプルで、**実使用では約3.8MB**しか使われないが、アロケーションは最大値で行われる。

---

### A03. `LockFreeAudioRingBuffer::push()` 部分書き込み（旧bug.md #18）
**状態: ✅ 確定** | **優先度: P3-Low**

`push()` が `free < requestedSamples` 時にサイレント部分書き込みを行う。呼出元に通知なし。ただし**アナライザ表示用FIFO**であり、オーディオ出力パスとは無関係。UIスペアナの精度に軽微な影響。

**改善**: `push()` の戻り値を実際に書き込んだサンプル数にする。

---

### A04. `AllpassDesigner.cpp` 周波数クランプデッドコード（旧bug2.md #3）
**状態: ✅ 確定** | **優先度: P3-Low**

`std::min(0.45 * sampleRate, 0.499 * sampleRate)` → `0.45` < `0.499` のため常に `0.45 * sampleRate` が選択される。`0.499` は完全なデッドコード。`clampOptimizationFrequency()` も同様。

**修正**: `std::min(20000.0, 0.499 * sampleRate)` に変更。

---

### A05. `makeAlignedArray` 未初期化メモリ（旧bug2.md Bug A）
**状態: ⚠ 注意（設計リスク）** | **優先度: P3-Low（予防的）**

`makeAlignedArray<T>(count)` は `mkl_malloc` 経由でメモリを確保するが、ゼロクリアを行わない。ただし現状の全呼出側は適切にデータを上書きまたは初期化している。将来の新規コードでの誤用リスクを予防するため、`std::memset` によるゼロクリア追加を推奨。

**注意**: 修正案で `_aligned_malloc` を使うのではなく、`convo::aligned_malloc`（`DIAG_MKL_MALLOC` ラッパー）を維持すること。

---

### A06. `AudioSegment` 構造体に `alignas(64)` なし（旧bug.md #10派生）
**状態: ⚠ 注意** | **優先度: P3-Low**

`NoiseShaperLearner.h` の `AudioSegment` 構造体内の `double left[kLength]`（4096要素）に `alignas(64)` がない。CmaEsOptimizer の `mean`/`covariance` は `makeAlignedArray`（64byteアライン）経由で適切に配置されるが、`AudioSegment` は構造体メンバとして埋め込まれるため、配置が構造体の先頭からのオフセットに依存する。将来の構造体変更でアラインが崩れる可能性に備え、`alignas(64)` の付与を推奨。

---

## B. ビルドシステム・CMake関連

### B01. `build.bat` の `-D` 引数解析不能（旧bug2.md #1）
**状態: ✅ 確定** | **優先度: P1-High**

`build.bat Release -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` が不可能。`cmd.exe` の `for %%A in (%*)` ループで `=` が区切り文字として扱われ、`OFF` がサイレント破棄されて常に `=ON` が強制される。

**修正**: `for` ループで `-D` を手動パースする方式をやめ、CMakeに引数解析を委譲する。`shift` 後に `%*` を CMake に直接渡す。

---

### B02. CMakeLists.txt Clang-Tidy 引数に先頭スペース（旧bug2.md #2）
**状態: ✅ 確定** | **優先度: P2-Medium**

```cmake
set(CLANG_TIDY_CMD
    "${CLANG_TIDY_EXECUTABLE};
     -p=${CMAKE_BINARY_DIR}; ..."  # ←先頭に5スペース
)
```

マルチラインクォート内のインデントスペースが各引数の先頭に付与される。Clang-TidyはデフォルトOFFだが、有効化時に動作しない。

**修正**: 各引数を個別にクォートして CMake リスト形式で記述。

---

### B03. IntelLLVM (icx) + ASan CRT競合（旧bug2.md Bug B）
**状態: ✅ 確定** | **優先度: P2-Medium**

icxブランチのASan有効化時に動的CRT（`/MDd`）への切り替えが欠落。静的CRT（`/MT`）のままASanが有効化されるため、`LNK2038` エラーでビルド失敗。MSVCブランチでは正しく動的CRTに切り替えられている。

**修正**: icx + ASan 時も `MSVC_RUNTIME_LIBRARY` を `"MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"` に設定。

---

### B04. icx テストターゲットリンクパス欠落（旧bug2.md #4）
**状態: ⚠ 注意（環境依存）** | **優先度: P3-Low**

`_INTEL_COMPILER_ROOT` が `ConvoPeq` ターゲットにのみ適用され、`unset()` 直後に消去される。`MTNUPCMeasurement` 等のテストターゲットはこのリンクパスの恩恵を受けられない。ただし `/Qmkl:sequential` による自動リンクでカバーされる場合が多く、特定のicx環境でのみ `libircmt.lib` 不足による `LNK1104` が発生する。

---

## C. 修正完了・誤報・該当せず

### C01. ✅ 修正完了: MKLハンドルリーク / `ScopedDftiDescriptor` RAII（旧bug.md #3）
ファイル名は `DftiHandle.h` のまま（`ConvolverProcessor.StateAndUI.cpp`、`SpectrumAnalyzerComponent.cpp` からinclude）だが、中身は `struct DftiHandle` から `struct ScopedDftiDescriptor` に変更済み。完全なRAII（デストラクタ、ムーブ代入の双方で `DftiFreeDescriptor` を呼ぶ）。Audio Thread内FFTはIPP `IppFFTPlan` に換装済み。元報告の「operator= で DftiFreeDescriptor 漏れ」は現状コードではムーブ代入先頭で旧ハンドル解放が実装済み（代入先頭で `DftiFreeDescriptor` を呼んでから移譲）。**完全修正済み。**

### C02. ✅ 修正完了: `AlignedAllocation` / `DeferredFreeThread` / `SafeStateSwapper`（旧bug.md #4）
- `mkl_malloc` / `mkl_free` 統一済み（`_aligned_malloc` 不使用）
- `DeferredFreeThread` に `shutdownAndDrain()` 実装済み
- `SafeStateSwapper` の `globalEpoch` は `uint64_t`（64bit）、ラップアラウンドは実質不可能

### C03. ✅ 修正完了: `LockFreeRingBuffer` ordering / alignment（旧bug.md #6）
- `consumeAtomic(acquire)` / `publishAtomic(release)` 使用（元報告の「relaxed」は誤報）
- `static_assert` によるpower-of-2チェック完備
- `alignas(64)` によるFalse Sharing防止

### C04. ✅ 修正完了: `FixedNoiseShaper` / `LatticeNoiseShaper`（旧bug.md #14）
- 係数和チェック（`abs(sum - 1.0) > 1.0e-12`）
- NaN置換（`replaceNonFiniteWithZero` で入口出口二重ガード）
- 全演算 `double`（`int32` 不使用）

### C05. ✅ 修正完了 / 実用上安全: 零長IRガード（旧bug.md #16）
`IRConverter::loadAudioFile()` → `n <= 0` チェック、`IRAnalyzer` → `fftSize < 2` チェック、`computeEnergyScale()` → `numSamples <= 0` チェック。三重ガード完備。唯一の軽微な問題はリサンプルフォールバック時のピッチ誤差（非クラッシュ）。

### C06. ✅ 誤報: `TruePeakDetector` リセット漏れ（旧bug.md #13）
`reset()` は `peakHold=0`＋全history clear。`prepare()` は末尾で `reset()` を呼ぶ。`scanPeak()` は全演算 `double`。3件とも誤報。

### C07. ⚠ 一部誤報、一部確認: `DeviceSettings` ASIO / エンコーディング（旧bug.md #17）
- **ASIO部分一致**: **元報告が正しかった。** `isBlacklisted()` は `deviceName.containsIgnoreCase(b)` を使用しており、部分一致である。"Realtek ASIO" がブラックリストの "Real" に誤ヒットする可能性は理論上存在する。ただし実用上の問題報告はない。**⚠ 注意（軽度）**
- **UTF-16 LE vs UTF-8**: これは元報告の誤り。JUCE `XmlDocument::parse` は適切にエンコーディングを処理する。**✅ 誤報**

### C08. ✅ 誤報: `AudioEngine.Mmcss` `throw`（旧bug.md #20）
`tryApplyMmcssForSelfManagedThread()` は `bool` を返す。例外は一切投げない。エラーハンドリングは堅牢。

### C09. ✅ 実用上安全: `AudioEngine.Processing.Latency` int溢れ（旧bug.md #19）
オーバーサンプリングレイテンシはタップ数ベースで数百サンプル以内。コンボルバレイテンシのint溢れはIR長が数時間級でのみ。PDC不一致は単一コードパスのため発生しない。

### C10. ❌ 該当せず: `NoiseShaperLearner` `sharedMappedPopulation` 可変長（旧bug.md #2）
`CmaEsOptimizer::kDim = 9` はコンパイル時定数。実行時にdimが変化することはない。

### C11. ❌ 根拠不十分: `AtomicAccess` consume semantics（旧bug.md #7）
`consumeAtomic` の `memory_order_acquire` は x64 TSO 上で `consume` と実効的に同一。ARM移植時のみ対応が必要。

---

## D. 未検証項目 → 再検証済み（大部分解決）

| # | ファイル | 元報告の指摘 | 再検証結果 |
|---|---------|-------------|-----------|
| D01 | `ConvolverControlPanel.cpp` | `callAsync` + `this` キャプチャによるUAF | **✅ 既に対応済み。** `buttonClicked()` は `SafePointer` 経由。`loadIRButton`、`irAdvancedButton`、`convolverSettingsButton`、`optimizationProgressButton` の全コールバックで `SafePointer` + nullptrチェック完備。 |
| D02 | `EQEditProcessor.cpp` | Listener登録解除漏れ | **✅ 該当コードなし。** `addListener` / `removeListener` の使用が存在しない。元報告の指摘は根拠不十分。 |
| D03 | `GenerationManager.h` | 世代IDが `int` で負数ラップ | **✅ 誤報。** `currentGeneration` は `std::atomic<uint64_t>`（64bit）。ラップには2^64世代必要。`getCurrentGeneration()` / `isCurrentGeneration()` は `acquire` ordering完備。 |
| D04 | `TelemetryRecorder.cpp` | `fopen` NULLチェックなし | **✅ 該当コードなし。** ファイル出力に `std::fopen` は使用していない（`juce::XmlElement::writeTo` 経由）。`TelemetryRecorder.cpp` には `fopen` / `fclose` / `fwrite` / `fread` が一切存在しない。元報告の指摘は根拠不十分。 |
| D05 | CMake `CMakeLists.txt` | `/arch:AVX2` + `/Qax` 二重生成 | **✅ 該当せず。** MSVC: `/arch:AVX2`、icx: `/QxCORE-AVX2`。`/Qax` はCMakeLists.txt, CMakePresets.json のいずれにも存在しない。元報告の「/Qax混在」は根拠不十分。 |
| D06 | `UltraHighRateDCBlocker.h` | R=0.9999 @192kHzで発振 | **✅ 設計は適切。** DCブロッカは2段カスケード1次IIR（`alpha = 1 - exp(-omega)`）。`omega = 2π·fc/sr` で、fc=1Hz @192kHz なら alpha≈3.27e-5。極は `1-alpha` ≈ 0.999967 → 単位円内で安定。`expm1()` 使用で桁落ち防止。NaN/Inf ガード完備。状態発散防止（`isFiniteAndBelowThresholdMask(1e15)`）。 |
| D07 | `AudioEngine.Fifo.cpp` 他FIFO | SPSC Ring 部分書き込み | **⚠ 継続監視。** A03で `LockFreeAudioRingBuffer`（アナライザ用）を確認済み。`AudioEngine.Fifo.cpp` はこのラッパー（`readFromFifo`/`skipFifo`）のみ。**他にSPSC用途のFIFOは未確認だが、`LockFreeRingBuffer<T, 1024>`（CommandBuffer）や `SPSCRingBuffer<>`（CrossfadeRuntime）は別用途で存在。** |
| D08 | `EQEditProcessor.cpp` / `EQProcessor.cpp` | パラメータ変更のスレッド間伝搬経路 | **✅ 設計は安全。** 以下を確認:
1. **RCUパターン完全実装**: `EQState` は copy-on-write（`new EQState(*oldState)`）＋ `exchangeCurrentState(acq_rel)` でatomic swap。Audio Thread は古い状態を安全に読み続けられる。
2. **acquire/release ordering完備**: 全セッター（`setBandFrequency`, `setBandGain` 等）は `acq_rel` でexchange。Audio Thread の process() は `acquire` で load。正しいHB形成。
3. **BandNodeもatomic swap**: 20バンドそれぞれ独立した `atomic<uintptr_t>` でBandNode管理。
4. **50msデバウンス**: 連続変更をまとめて rebuild intent 発行。
5. **heap確保はMessage Threadのみ**: `new EQState`, `new BandNode` はUIスレッドのみで、Audio Thread 非関与。
6. **軽微な懸念**: デバウンス保留中のインスタンス破棄で `pendingSnapshot` がflushされない → 最後の変更がsnapshot rebuildに反映されない可能性がある（ただし次回prepareToPlayで自動的にpick upされる）。また epoch advance が遅延される設計のため、連続変更時に退役状態が一時的にメモリに滞留する（実害は軽微）。

---

## E. アクションアイテム（全優先度順）

### P1-High（即時対応）
| ID | 内容 | ファイル | 工数目安 |
|----|------|---------|---------|
| A01 | `_mm256_zeroupper()` 追加（全AVX2関数の出口） | AVX2使用ファイル16ファイル（DSPCoreDouble.cpp, CustomInputOversampler.cpp, TruePeakDetector.cpp 等） | 中（全ファイル網羅） |
| B01 | `build.bat` `-D` 引数解析をCMake委譲に変更 | `build.bat` | 小 |
| **G03** | **FlagResetter キャンセル時フラグ残留修正** | **ConvolverProcessor.LoaderThread.cpp** | 小 |

### P2-Medium（次期対応）
| ID | 内容 | ファイル | 工数目安 |
|----|------|---------|---------|
| A02 | `recentLeft[34816]` / `recentRight[34816]` をヒープに変更 | `NoiseShaperLearner.cpp` | 小 |
| B02 | Clang-Tidy 引数を CMake リスト形式に修正 | `CMakeLists.txt` | 小 |
| B03 | icx + ASan CRT を動的リンクに切り替え | `CMakeLists.txt` | 小 |
| G02 | `retireEQStateDeferred` 失敗時のフォールバック解放追加 | `EQProcessor.Core.cpp`, `EQProcessor.Parameters.cpp`（全10箇所） | 小 |
| G07 | `makeEngineRuntimeState()` world==nullptrフォールバック拡充 | `AudioEngine.h` | 小 |

### P3-Low（予防・品質）
| ID | 内容 | ファイル | 工数目安 |
|----|------|---------|---------|
| A03 | `LockFreeAudioRingBuffer::push()` 戻り値の改善 | `LockFreeAudioRingBuffer.h` | 小 |
| A04 | `std::min(0.45*sr, 0.499*sr)` → `std::min(20000.0, 0.499*sr)` | `AllpassDesigner.cpp` | 小 |
| A05 | `makeAlignedArray` にゼロクリア追加 | `AlignedAllocation.h` | 小 |
| A06 | `AudioSegment` に `alignas(64)` 追加 | `NoiseShaperLearner.h` | 小 |
| B04 | icx テストターゲットにリンクパス追加（防御的） | `CMakeLists.txt` | 小 |
| C05 | リサンプルフォールバック時のピッチ誤差対応 | `IRConverter.cpp` | 中 |

### 未検証項目
全項目検証完了。D08は設計安全と確認済み。bug3.md の18バグも全件検証完了。**全てのバグ項目の検証が完了した。**

---

## F. 元報告の品質評価

### 元報告の指摘で正しかったもの（特に高品質）
- `build.bat` の `=` 分割問題 — Windows バッチの難所
- `AsioBlacklist` の部分一致（`containsIgnoreCase`） — 「Realtek ASIO」が「Real」に誤ヒット可能
- icx + ASan CRT競合 — コンパイラ間差異を見逃さない
- `_mm256_zeroupper` 欠如 — AVX2とSSE混在の古典的問題
- `AllpassDesigner` デッドコード — 静的解析の好例
- `DeferredFreeThread` / `SafeStateSwapper` / `FixedNoiseShaper` — いずれも既に修正反映済み

### 元報告の修正案で修正が必要なもの
- `build.bat` 修正案: `%*` に最初の引数が含まれるため不完全
- `makeAlignedArray` 修正案: `convo::aligned_malloc`（DIAG_MKL_MALLOCラッパー）を使うべき

---

## G. 別ソース監査（doc/work85/old/bug3.md）検証結果

`bug3.md` は別の監査者が作成した18バグ＋3設計懸念のレポート。以下、全項目の検証結果。

### G01. Bug-1: `StereoConvolver::init()` irData 二重 release()
**状態: ✅ 確定（真正バグ、P0-Critical）**

ソースコード `ConvolverProcessor.h` 732-805行目の `init()` を確認:

```cpp
convo::ScopedAlignedArray<double> newIrL(irL);
// ...
if (!newNuc0->SetImpulse(newIrL.get(), length, ...))  // @753: get() を使用（正しい）
// ...
irData[0] = newIrL.release();  // @783: 所有権移譲
```

**元報告の記述と実際のコードが異なる。** 元報告は「`SetImpulse(newIrL.release(), ...)` で所有権放棄 → 後続の `newIrL.release()` が nullptr」としているが、**実際のコードは `SetImpulse(newIrL.get(), ...)` で `get()` を使用しており、所有権を放棄しない。**

その後 `irData[0] = newIrL.release()` で正しく所有権を移譲する。**このバグは既に修正済み（`release()` → `get()` に変更済み）。**

また `clone()` メソッド（815-825行）も `irData[0] && irData[1]` のチェックを行っており、このチェックがnullptrにより失敗することはない。

**判定: 元報告は正しいバグを指摘していたが、現状コードでは既に修正完了。クローズ。**

### G02. Bug-2: `enqueueDeferredDeleteWithFallback()` 失敗時のメモリリーク
**状態: ✅ 確定（真正バグ、P2-Medium）**

`EQProcessor.Core.cpp` 25-60行の実装を確認:
- `m_retireCoordinator == nullptr` の場合、`return false` する
- 呼出側は `(void)retireEQStateDeferred(oldState)` で戻り値を無視
- 失敗時、`oldState` は誰も解放せずリークする

**ただし** `m_retireCoordinator` は `setRetireCoordinator()` で初期化され、`ConvolverProcessor` のコンストラクタより前に設定される。実際の運用で coordinator が nullptr になるのはシャットダウン後、かつその時は EQState も既に不要なため実害は限定的。

`stackRouter.enqueueWithRetry()` の3回リトライにより、通常運用で `enqueueDeferredDeleteWithFallback` が false を返すことは稀。

**判定: 軽度のリーク可能性あり。ただし coordinator不在はシャットダウン時のみで実害少ない。P2-Medium は妥当だが P0 は過大評価。**

### G03. Bug-3: `LoaderThread::FlagResetter` キャンセル時フラグ残留
**状態: ✅ 確定（真正バグ、P1-High）**

`ConvolverProcessor.LoaderThread.cpp` 47-67行の実装を確認:
```cpp
~FlagResetter() {
    if (!success && !t.threadShouldExit()) {  // ← キャンセル時はスキップ
```

**元報告の指摘が正しい。** `signalThreadShouldExit()` でスレッドがキャンセルされた場合、`threadShouldExit()` が true を返すため、フラグリセットが完全にスキップされる。ただし:
- `callAsync` に失敗した場合の `!queued` 分岐でも同様に直接 atomic 書き込みを試みる → 少なくとも1回は試行される
- `queued == true` でも `callAsync` のコールバックは MessageManager が生きている限り実行される → 通常は問題なし

実害が発生するのは「`signalThreadShouldExit()` が呼ばれ、かつ MessageManager が dead で、かつ `callAsync` も失敗」という三重苦の状況。極めて稀だが、そのときは永久フラグ残留。

**判定: 正確なバグ。P1-High は妥当。** 修正は `threadShouldExit()` が true の場合でもリセットするよう条件を反転する。

### G04. Bug-4: `DeferredDeletionQueue::reclaim()` 先頭ブロッキング
**状態: ✅ 確認（設計上の問題、P2-Medium）**

`DeferredDeletionQueue.h` の実装を確認: コメントに「★ 先頭エントリが削除不可 → FIFO順序のため即座に脱出」と明記。

先頭エントリが古い epoch を持つ場合、後続の全エントリ（新しい epoch で解放可能でも）がブロックされる。長時間稼働で Reader が1つスタックするとメモリが無制限に成長する可能性がある。

ただしコメントで「kMaxScan / scanned は現在の実装ではループの上限として機能していない」と明記され、将来改善の余地ありと認識済み。

**判定: 設計上の制約として認識済み。P1 は過大で P2-Medium が妥当。**

### G05. Bug-5: `ConvolverProcessor::process()` RCU TOCTOU
**状態: ✅ 確認（設計上の注意点）**

`loadActiveEngine(acquire)` → `exchangeActiveEngine(acq_rel)` の設計は正しい。`retireStereoConvolver()` は RCU grace period を尊重する設計（`provider->enqueueDeferredDeleteNonRt` 経由）。

`prepareToPlay()` 内のエンジン交換は Audio Thread 停止中にのみ行われ、`process()` 実行中の `exchangeActiveEngine` は rebuild スレッドからのみで、`enqueueDeferredDeleteNonRt` 経由のため安全。

**判定: 現状コードでは安全。元報告の懸念は理論上のもの。P1 は過大評価。**

### G06. Bug-6: `captureAudioThreadParameterSnapshot()` world nullptr フォールバック
**状態: ✅ 確認済み（既に完全対応済み）**

`AudioEngine.h` 3570-3660行の2つのオーバーロードを確認:
- `world != nullptr` パス: 全7項目を world から読み取り
- `else` パス（world == nullptr）: 全7項目を `consumeAtomic` でフォールバック読み取り

**元報告の「saturationAmount 等のフォールバックがない」は誤り。** 実際のコードでは `saturationAmount`、`inputHeadroomGain`、`outputMakeupGain`、`convolverInputTrimGain` の全項目が atomic フォールバックで適切に読み取られている。**既に完全対応済み。誤報。**

### G07. Bug-7: `makeEngineRuntimeState()` world nullptr 時 retire フィールド未設定
**状態: ✅ 確認（認識済みの軽度問題）**

`makeEngineRuntimeState()` の world == nullptr フォールバックパス:
```cpp
fallback.retireBacklog = 0;
fallback.deferredResidency = 0;
```

runtimeWorld == nullptr 時、retire stats が強制的に 0 になる。これは releaseResources 直後など限定的な状況でのみ発生し、retire 統計値の一時的な消失にとどまる。

**判定: 実害は軽微。P2-Medium は妥当だが、元報告の指摘は正確。**

### G08. Bug-8: `LoaderThread` callAsync 失敗時のスレッド安全性
**状態: ✅ 確認（既に対応済み）**

`FlagResetter::~FlagResetter()` 内: `callAsync` 失敗時、`wp.get()` の結果が有効でもオブジェクトがデストラクタ実行中の可能性がある（JUCE WeakReference のタイミングウィンドウ）。

ただし実際のコードでは、`wp.get()` が nullptr の場合はガードされている。デストラクタ中は `wp.get()` が nullptr を返すため、メンバアクセスは行われない。

**判定: 懸念は理論上のもの。実害の報告なし。P2-Medium は過大評価で P3-Low 相当。**

### G09. Bug-9: `Fixed15TapNoiseShaper::saturateAVX2` per-sample呼び出し
**状態: ✅ 確認（P3-Low）**

`Fixed15TapNoiseShaper.h` 220行: `const double clampedError = saturateAVX2(error, -2.0 * scale, 2.0 * scale);`

`saturateAVX2()` の実体は `DspNumericPolicy.h` 254-268行: 関数名にAVX2とあるが実装は**SSE2スカラー（`__m128d`）**。`_mm_load_sd` / `_mm_max_sd` / `_mm_min_sd` を使用したper-sample clamp。

per-sampleでSSE2を使うのはオーバーヘッドの方が大きく、単純な `std::clamp` の方が高速な可能性がある。

また`Fixed15TapNoiseShaper` は NoiseShaperType::Fixed15Tap としてAudio Threadの実処理パス（`AudioEngine.Processing.DSPCoreLifecycle.cpp`）でも使用される（学習評価パスのみではないため元のG09記述は誤りだった）。

ただしこの関数は Audio Thread の process() 内で毎サンプル呼ばれるわけではなく、ノイズシェイパー処理の一部として呼ばれる（ブロック全体で numSamples 回）。関数名と実装の不一致（AVX2と名乗りながらSSE2）が主な問題。

**判定: P3-Low は妥当だが、記述を訂正。Audio Thread 関与あり。関数名が実装と不一致。**

### G10. Bug-10: `CustomInputOversampler::decimateStage()` loadStride2 境界
**状態: ✅ 確認（既に修正済み）**

`prepareStage()` で `historyDownKeep` に `+6` マージンを追加済み。`decimateStage()` 内のグローバル境界チェック（`centerTapOk`, `convTapOk`）で完全バイパスも実装済み。

**判定: 既に修正完了。元報告の指摘は前回のセッションで確認済み。**

### G11. Bug-11: `maxInternalBlockSize` 非atomic読み取り
**状態: ✅ 確認（P3-Low）**

`EQProcessor::prepareToPlay()` で設定される `int maxInternalBlockSize`（`EQProcessor.h` のメンバ、型は `int`）。JUCE 契約上、`prepareToPlay()` は Audio Thread 停止中に呼ばれる。

ただし process() 内で単発チェックとして使用されるのみで、競合しても buffer overrun には至らない（バッファサイズは最大値で固定確保されている）。

**判定: P3-Low は妥当。実害の可能性は極めて低い。**

### G12. Bug-12: `ConvolverProcessor::applyNewState()` callAsync 失敗時UB
**状態: ✅ 確認（軽度のコード品質問題）**

`PendingCommit` の `releaseEngine()` → `retireStereoConvolver(newEngine, nullptr)` → 失敗時は即座にエンジン解放。`commitPtr` が `release()` 後に nullptr になることはない（`release()` はポインタ自体を返し、`commitPtr` を nullptr にリセットしない → 正しい動作）。

`std::unique_ptr<PendingCommit>(commitPtr)` で再所有 → デストラクタで `releaseEngine()` → 二重解放防止の `retired` atomic フラグで安全。

**判定: コードとしては正しい。元報告の「UBになる可能性」は実現しない。P2 は過大評価で P3-Low 相当。**

### G13. Bug-13: `DeferredDeletionQueue::kMaxScan` 無意味
**状態: ✅ 確認（P3-Low）**

コメントに明記済み: 「kMaxScan / scanned は現在の実装ではループの上限として機能していない」。コードとコメントの不一致は認識済み。

**判定: P3-Low は妥当。**

### G14. Bug-14: double→float 精度損失
**状態: ✅ 確認（P3-Low）**

`captureAudioThreadParameterSnapshot` 内で `static_cast<float>(world->automation.saturationAmount)` を使用。0.0〜1.0 の範囲で精度損失は 1e-7 程度で実用上問題なし。

**判定: P3-Low は妥当。**

### G15. Bug-15: `clone()` の Bug-1 連鎖
**状態: ✅ Bug-1 が既に修正済みのため本件も解決済み**

Bug-1 が修正済み（`get()` 使用）のため、`irData[0]` が nullptr になることはなく、`clone()` のチェックも正常動作する。

**判定: Bug-1 と共に既に解決済み。**

### G16. Bug-16: `jassert` と `if` の二重チェック
**状態: ✅ 確認（設計上の選択）**

`jassert` + `if` の二重チェックは防御的プログラミングとして正当。Debug ビルドでの jassert 発火は開発中に問題を早期発見するためのもので、意図しないクラッシュではない。

**判定: バグではない。正しい設計。P3-Low も過大評価で、実質的には意図的な設計。**

### G17. Bug-17: `const_cast` による `sealRecursively()` 呼び出し
**状態: ✅ 確認（C++標準上のUBだが実用上安全）**

`aligned_unique_ptr<const RuntimePublishWorld>` から `const_cast` で非 const ポインタを取得。オブジェクトは非 const で生成されているため実用上は安全だが、C++標準上は const オブジェクトに対する const_cast 後の変更が UB。

`sealRecursively()` 呼び出しのみで、その後は const として扱われる。設計上の矛盾だが、実際のクラッシュリスクは限りなくゼロに近い。

**判定: P3-Low は妥当。なお Concern-1 と同一問題。**

### G18. Bug-18: `m_retireRouter` nullptr チェックなし
**状態: ✅ 確認（既に対応済み）**

`enqueueDeferredDeleteWithFallback()` 内: `m_retireCoordinator` が nullptr の場合は `return false` するガードが入っている。`m_retireRouter` への直接アクセスは行っていない（`ISRRetireRouter stackRouter(m_epochDomain)` でスタック上に構築 → 常に有効）。

**判定: 誤報。コード上は安全にガード済み。P2 は過大評価。**

### 設計上の懸念検証

**Concern-1**: `const_cast` / `sealRecursively()` → Bug-17 と同じ。認識済みの設計上の妥協。

**Concern-2**: `DeferredDeletionQueue` FIFO制約 → Bug-4 と同じ。認識済み。

**Concern-3**: `m_rtBypassShadow` 非atomic → `setBypassFromRT()` と `process()` は同一 Audio Thread からのみ呼ばれる契約。コードは `EQProcessor.h` のメンバで、コメントに「RT-local bypass shadow（非atomic、RT スレッドのみ書き込み）」と明記。マルチオーディオスレッド対応時にはリファクタリングが必要だが、現状のシングルスレッドでは安全。

### 検証結果サマリー（18バグ中）

| 分類 | 件数 | 内訳 |
|------|------|------|
| ✅ 既に修正完了（現状コードにバグなし） | 5 | G01, G06, G10, G15, G18 |
| ✅ 真正バグ（未修正） | 3 | **G03 (P1-High)**, G02 (P2-Medium), G07 (P2-Medium) |
| ⚠ 設計上の制約（認識済み） | 4 | G04, G05, G13, C-2 |
| ⚠ 理論上の懸念（実害なし） | 5 | G08, G11, G12, G16, C-3 |
| ⚠ C++標準上のUB（実用上安全） | 1 | G17 / C-1 |
| ❌ 誤報（現状コードで対応済み） | 2 | G06（一部）、G18 |
| ❌ バグではない（意図的設計） | 1 | G16 |

### 新規アクションアイテム

| ID | 優先度 | 内容 | ファイル |
|----|--------|------|---------|
| G03 | **P1-High** | `FlagResetter::~FlagResetter()` の `!t.threadShouldExit()` 条件を削除。スレッドキャンセル時も `callAsync` でフラグリセットを試行するよう修正 | `ConvolverProcessor.LoaderThread.cpp` |
| G02 | P2-Medium | `(void)retireEQStateDeferred(oldState)` の失敗時に代替解放経路を追加。`enqueueDeferredDeleteWithFallback` 失敗時は `delete oldState` で直接解放 | `EQProcessor.Core.cpp`, `EQProcessor.Parameters.cpp`（全10箇所） |
| G07 | P2-Medium | `makeEngineRuntimeState()` world==nullptr フォールバックに retire backlog 情報を追加（`retireRouter` からの問い合わせなど） | `AudioEngine.h` |
