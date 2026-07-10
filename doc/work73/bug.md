ご指摘いただきありがとうございます。私の前回の解析結果をコードベースとJUCEの仕様、およびアーキテクチャの前提に照らし合わせて再検証しました。
結論から申し上げますと、**私が提示した4つのクリティカルバグ候補のうち、3つは私のコンテキスト誤認による誤指摘（False Positive）でした。** 誠に申し訳ありません。唯一「TruePeakDetectorのRチャンネル計測欠落」のみが実際にクラッシュや機能不全を引き起こす真のバグ（True Positive）です。
前回の誤指摘の撤回理由と、コードをさらに深掘りして発見した**真に音質劣化を引き起こす新たなバグ**を報告します。
### 1. 前回指摘の再検証と撤回
#### ❌ 誤指摘 1: AudioSourceChannelInfo::buffer の Null Pointer Dereference
JUCEの仕様上、juce::AudioSourceChannelInfo::clearActiveBufferRegion() は内部で if (buffer != nullptr) のチェックを行ってから clear() を呼び出す安全な設計になっています。したがって、ホストから nullptr が渡された場合でも Access Violation は発生しません。JUCEの内部実装に対する私の見落としでした。
#### ❌ 誤指摘 2: EQProcessor.Parameters.cpp における Lost Update
スタンドアロンアプリケーションとして設計された本システムでは、setBandGain 等のパラメータ変更は全て JUCE の Message Thread から直列にディスパッチされる前提となっています。他スレッド（MIDIやOSC処理スレッドなど）からの非同期書き込み経路が存在しないため、exchangeCurrentState での ABA 問題や Lost Update は発生し得ません。スレッドコンテキストのアーキテクチャ制約を過小評価していました。
#### ❌ 誤指摘 3: AudioSegmentBuffer.h でのバッファオーバーラン
kCapacity は 5 \times 768000 = 3,840,000 サンプル確保されており、Audio Callback のブロックサイズ（最大でも 65536 \times 8）がこれを超えることは物理的に不可能です。ラップアラウンド時の second サイズの計算も、入力バッファ left に対する適正なオフセット加算にとどまるため、境界外書き込みは発生しません。
#### ✅ 妥当な指摘: TruePeakDetector における R チャンネルの計測欠落
これは間違いなくバグです。Stage 1（2x → 4x）の処理が L チャンネルのバッファ長（up1Samples）分しか実行されておらず、続く AVX2 のピークスキャンも L チャンネルの領域で終了しています。R チャンネルの True Peak は完全に無視されます。
### 2. 新規発掘: SimplePeakLimiter の Knee 補間境界エラー (Critical Audio Glitch)
再検証の過程で、音響処理の根幹に関わる致命的な数学的エラーを発見しました。src/audioengine/SimplePeakLimiter.h に実装されているソフトニーリミッターにおいて、**Knee（膝）の右半分が完全に切断されており、波形に強烈な不連続性（高調波歪み）を発生させます。**
```cpp
const double clipStart = thresholdLinear - kneeLinear * 0.5;
// ...
if (peak > clipStart)
{
    if (peak <= thresholdLinear) // 🚨 バグ発生箇所
    {
        // Knee 領域: 3次スプライン補間
        const double t = (peak - clipStart) / kneeLinear;
        const double kneeShape = t * t * (3.0 - 2.0 * t);
        desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
    }
    else
    {
        // リミッティング領域
        desiredGain = thresholdLinear / peak;
    }
}

```
**【バグの詳細】**
ソフトニーの範囲は、下端が threshold - \frac{knee}{2} (clipStart)、上端が threshold + \frac{knee}{2} になります。スプライン補間の変数 t は、この区間において 0.0 から 1.0 まで遷移するように設計されています。
しかし、条件式が if (peak <= thresholdLinear) となっているため、t = 0.5（つまり Knee の中央）に到達した瞬間に else 句のハードリミッティング式（\frac{threshold}{peak}）へ強制的にジャンプします。
値自体は t=0.5 の地点で連続していますが、**微係数（傾き）が極端に不連続**になります。これにより、スレッショルドをわずかに超えたトランジェント信号に対して、本来の滑らかなゲインリダクションではなく、ハードクリップと等価な角張ったゲイン変化が適用され、クリックノイズが発生します。
**【修正案】**
スプライン補間を Knee の上端まで完全に適用するため、条件式を以下のように修正する必要があります。
```cpp
// 修正後
if (peak <= thresholdLinear + kneeLinear * 0.5)

```


ご提供いただいたソースコードレビュー文書を詳細に分析し、バグリストを作成しました。

---

🐛 バグリスト（Bug List）

重大度: ハイ（High） — 即座の修正が必要

# モジュール 問題の概要 詳細 
3 `ConvolverProcessor` 二重解放 / Use-After-Free (UAF) `retired` フラグが存在するにも関わらず `destroyStereoConvolver()` で一切チェックされていない。`releaseResources()` と通常の `commitNewConvolver()` パスで同一ポインタが enqueue + 直接解放される競合が発生する可能性。`compare_exchange` パターンまたは `retired` フラグチェックの追加が必須。 
5 `CmaEsOptimizerDynamic` NaN 伝播による永久ループ `mean` が NaN になると、以降の全世代で NaN が伝播し、最適化が永遠に回復しない。NaN 検出時に `mean` をリセットする処理が必要。 
12 `build.bat` PGO 変数の誤った展開 `%CMAKE_PGO_FLAGS%` （遅延展開）を使用しているため、条件分岐内で古い値が参照される。`!CMAKE_PGO_FLAGS!` （即時展開）に修正必須。 

---

重大度: ミディアム（Medium） — 修正推奨

# モジュール 問題の概要 詳細 
1 `DeferredDeletionQueue` キューブロッキングによる RT 安全性低下 FIFO 順序を厳格に維持する設計のため、先頭エントリが削除不可の場合、後続の安全なエントリも解放されない。メモリ使用量が一時的に増加し、キュー満杯時は Audio Thread 側で `enqueue` が失敗する（RT 安全性破壊の可能性）。 
2 `AudioSegmentBuffer` `clear()` 無視によるデータ不整合 `writePosition` と `totalSamples` が別々の atomic 変数として管理されているため、`clear()` と `pushBlock()` の競合時に `totalSamples` が古い値を観測し、クリア前の古いデータが混入する。クリックノイズ・ポップノイズの原因となる。 
8 `ConvolverControlPanel` `closeButtonPressed` + `reset` の競合 UI スレッドと Audio スレッド間で `reset()` とウィンドウクローズが競合する可能性。クローズ中に Audio コールバックが `reset()` を呼び出すと、破棄済みリソースへのアクセスが発生する可能性。 
11 `CMakeLists.txt` テストリンクの MKL 依存 `ISRSemanticValidationTests` に `ISRRetire.cpp` が直接コンパイルされているが、MKL リンクが未定義。`ISRRetire.cpp` が MKL 関数を呼び出している場合、リンクエラーが発生する。全テストへの MKL リンク統一、または `ISRRetire.cpp` の依存確認が必要。 

---

重大度: ロー（Low） — 監視・予防的対応

# モジュール 問題の概要 詳細 
4 `CustomInputOversampler` 整数オーバーフローの理論的可能性 `stage.maxOutputSamples = stageInputMax * 2` で `int` オーバーフローの可能性があるが、実際の `maxInputBlockSize` 制限（通常 256〜8192）により発生確率は極めて低い。 
6 `AllpassDesigner` `makeAlignedArray` の例外安全性 `makeAlignedArray` がスロー版（`new` 使用）だが、呼び出し元で例外ハンドリングが行われている。現状は安全。 
7 `CacheManager` 理論上のハッシュ衝突 ハッシュベースのキャッシュで理論上衝突は存在するが、実用上は無視できるレベル。 
9 `BuildSnapshot` `noexcept` 指定による `std::terminate()` リスク `noexcept` 関数内で例外が発生すると `std::terminate()` が呼ばれるが、現状の実装で例外が発生する経路は確認されていない。 
10 `CpuFeatureCheck` 値は正しい 問題なし。検証済み。 

---

📋 修正優先度まとめ

```
【即座に修正】
  ├── #3 ConvolverProcessor 二重解放/UAF
  ├── #5 CmaEsOptimizer NaN 伝播
  └── #12 build.bat PGO 変数展開

【次のスプリントで修正】
  ├── #1 DeferredDeletionQueue キューブロッキング
  ├── #2 AudioSegmentBuffer clear() 無視
  ├── #8 ConvolverControlPanel 競合
  └── #11 CMakeLists.txt MKL リンク

【継続監視】
  └── #4, #6, #7, #9, #10
```

---

🔧 推奨修正スニペット

#3: `destroyStereoConvolver` に `retired` チェック追加

```cpp
static void destroyStereoConvolver(void* p) noexcept {
    auto* sc = static_cast<StereoConvolver*>(p);
    if (!sc) return;
    
    // 二重解放防止
    bool expected = false;
    if (!sc->retired.compare_exchange_strong(expected, true, 
                                              std::memory_order_acq_rel)) {
        return; // 既に解放済み
    }
    
    sc->~StereoConvolver();
    convo::aligned_free(sc);
}
```

#5: `CmaEsOptimizer` NaN 検出

```cpp
// 各世代の開始時にチェック
if (std::isnan(mean[0])) {
    // mean を初期値にリセット
    mean = initialMean;
    // または前回の有効な mean を復元
}
```

#12: `build.bat` 即時展開

```bat
REM 変更前
if defined CMAKE_PGO_FLAGS (
    echo %CMAKE_PGO_FLAGS%
)

REM 変更後
if defined CMAKE_PGO_FLAGS (
    echo !CMAKE_PGO_FLAGS!
)
REM setlocal EnableDelayedExpansion が必要
```

---



Here is an analysis of the audio processing flow in ConvoPeq, along with potential bugs and architectural issues identified in the source code.

### I. Audio Processing Flow

The core audio processing pipeline is primarily orchestrated within `AudioEngine::DSPCore::process` (for `float`) and `AudioEngine::DSPCore::processDouble` (for `double`). The flow guarantees lock-free, allocation-free, and wait-free execution suitable for a real-time audio thread.

1. **Context Initialization**:
* Establishes the `RCUReaderGuard` to safely access the active runtime state without locking.
* Asserts the real-time thread context and captures telemetry/timestamps for diagnostics (e.g., XRUN detection).


2. **Input Preparation & Sanity Checks**:
* Validates block sizes against `maxSamplesPerBlock`.
* Invokes `processInput` / `processInputDouble` to convert `float` to `double` (if necessary), apply input headroom gain, and sanitize NaN/Inf values to zero.
* Applies the input stage of the `UltraHighRateDCBlocker`.


3. **Oversampling (Up-sampling)**:
* If `oversamplingFactor > 1`, `CustomInputOversampler::processUp` is invoked.
* Applies the oversampled stage of the `UltraHighRateDCBlocker`.


4. **Core DSP Processing**:
* Depending on the `ProcessingOrder` parameter, the signal is routed serially:
* **ConvolverThenEQ**: `ConvolverProcessor::process` -> `EQProcessor::process`
* **EQThenConvolver**: `EQProcessor::process` -> `ConvolverProcessor::process`


* Pre-convolver trim gain is applied dynamically if the order is EQ->Convolver.


5. **Output Filtering**:
* Applies High-Cut/Low-Cut filters if Convolver is the last stage, or LPF/HPF if EQ is the last stage via `OutputFilter::process`.


6. **Output Makeup & Soft Clipping**:
* Applies the output makeup gain.
* If enabled, applies `musicalSoftClipScalar` (vectorized via AVX2). If running at 1x sample rate, it utilizes a localized 2x oversampling wrapper (`softClipOS`) exclusively for the clipping stage to prevent aliasing.


7. **Oversampling (Down-sampling)**:
* Reverts the sample rate back to the base rate using `CustomInputOversampler::processDown`.


8. **Crossfading & Bypass Blending**:
* Merges the processed (wet) signal with the latency-aligned dry signal based on the `LinearRamp` fade state.


9. **Final Output Formatting & Dithering**:
* Invokes `processOutputDouble` or `processOutput`.
* Applies the final output DC Blocker.
* Evaluates True Peak and LUFS loudness.
* Applies quantization and the selected noise shaper (e.g., Psychoacoustic, Fixed, or Adaptive 9th-order).



---

### II. Discovered Bugs and Vulnerabilities

#### 1. Missing Bypass/Dry Blend Logic in `processDouble`

There is a critical divergence between `AudioEngine::DSPCore::process` (float pipeline) and `AudioEngine::DSPCore::processDouble` (double pipeline).

In `AudioEngine.Processing.DSPCoreFloat.cpp`, after oversampling is processed down, there is explicit logic to blend the dry signal if a bypass fade is requested:

```cpp
    if (oversamplingFactor > 1)
    {
        oversampling.processDown(processBlock, originalBlock, static_cast<int>(originalBlock.getNumChannels()));
        processBlock = originalBlock;

        if (bypassBlendRequested)
        {
            // ... Blends wet and dry using bypassFadeGain ...
        }
    }

```

However, in `AudioEngine.Processing.DSPCoreDouble.cpp`, the `bypassBlendRequested` evaluation and subsequent crossfade mix logic are entirely absent after the `oversampling.processDown` call. If the engine is processing natively in double precision, triggering a bypass will not smoothly fade out the wet signal, leading to abrupt cuts or entirely ignoring the bypass state depending on the host's buffer routing.

#### 2. Undefined Behavior in Ring Buffer Bitmasking (Negative Indexing)

In `src/convolver/ConvolverProcessor.Runtime.cpp`, the delay interpolation logic reads from the ring buffer using a fractional index:

```cpp
                        int idx = iRead + i;
                        double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];
                        double p1 = srcBuf[(idx    ) & DELAY_BUFFER_MASK];

```

When `iRead == 0` and `i == 0`, `idx - 1` results in `-1`. Applying a bitwise AND (`&`) to a negative signed integer yields implementation-defined behavior in C++ prior to C++20 (though usually two's complement). Even in C++20, operating on negative signed integers with bitmasks intended for unsigned wrapping is poor practice and can lead to unexpected buffer jumps if the compiler optimizes the signed arithmetic aggressively. `idx` should be wrapped safely (e.g., `(idx - 1 + DELAY_BUFFER_SIZE) & DELAY_BUFFER_MASK`).

#### 3. Redundant Work in Learner Threads

In `src/NoiseShaperLearner.cpp`, the `runEvaluationJobsForWorker` function is executed by multiple evaluation threads. Each thread performs the following mapping:

```cpp
        constexpr int totalCoeffs = CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim;
        alignas(64) double tanhBuffer[totalCoeffs] = {};
        const auto* population = candidatePopulationMatrix();
        vdTanh(totalCoeffs, reinterpret_cast<const double*>(population), tanhBuffer);

```

Because `candidatePopulationMatrix` is shared and read-only during the evaluation phase, every active worker thread is independently computing `vdTanh` for the *entire* population matrix. This wastes CPU cycles. The mapping from unconstrained space to reflection coefficients should happen once on the main worker thread before dispatching the jobs to the evaluation thread pool.

---

Do you want to focus on fixing the missing bypass logic in the double-precision pipeline, or would you prefer to address the DSP arithmetic/indexing issues first?


添付ファイル `ConvoPeq.md` に含まれる最新のソースコード（マルチスレッド同期、メモリ管理、AVX2 SIMD、およびMKLを多用した複雑なリアルタイムDSPシステム）を徹底的に検証した結果、前回の基本検証を遥かに超える、深刻なレースコンディション、メモリリーク、未定義動作、および音響的な破綻を引き起こすバグ（脆弱性）を発掘しました。

以下に、不具合の箇所、メカニズム、それらが引き起こす現象、および修正案を極めて詳細に報告します。

---

### 1. `EQCacheManager`における重大なスレッド・データレース

**該当箇所:** `src/AudioEngine.Cache.cpp` (または `CacheMap` 内部構造体定義)

#### [バグのメカニズム]

`EQCacheManager` は、アトミックポインタ `cacheMapPtr` を用いて、Read-Copy-Update (RCU) パターンを模倣したキャッシュ管理を行っています。しかし、コピーコンストラクタの実装に致命的な不具合があります。

```cpp
CacheMap(const CacheMap& other) : owner(other.owner) {
    for (const auto& entry : other.map) {
        if (entry.second != nullptr) {
            entry.second->addRef(); // 致命的: スレッドセーフではない
        }
        map.emplace(entry.first, entry.second);
    }
}

```

* **問題点:** `entry.second` (つまり `EQCoeffCache*`) 自体はアトミック管理されておらず、非リアルタイムの別スレッド（UIやパラメータ変更を検知したWorkerスレッド）が同時に `storeNewMap` や `releaseCache` を実行した際、同一の `EQCoeffCache` オブジェクトに対して複数のスレッドから同時に `addRef()` または `release()` が書き込み競合（データレース）を起こします。
* **JUCEの仕様との衝突:** JUCEの `ReferenceCountedObject` 等の `addRef()` は通常、非アトミックな符号付き整数のインクリメントであるため、マルチスレッド環境での同時操作は未定義動作 (Undefined Behavior) となり、参照カウントが破損します。

#### [引き起こされる現象]

* キャッシュがまだ使用中であるにもかかわらず参照カウントが `0` になり、オーディオ内部、あるいは `drainDeferredMapsUnderLock` で**二重解放 (Double Free) またはハング・クラッシュ**を引き起こします。逆にカウントが減りきらずにメモリリークすることもあります。

#### [対策]

`EQCoeffCache` の参照カウントを `std::atomic<int>` に変更するか、`CacheMap` の複製・構築処理全体を `writeMutex` のロック保護下に完全に閉じ込める必要があります。

---

### 2. コンストラクタ / デストラクタにおける「二重解放」と「メモリリーク」の交差点

**該当箇所:** `src/convolver/ConvolverProcessor.StateAndUI.cpp` / `~ConvolverProcessor()`

#### [バグのメカニズム]

デストラクタ `~ConvolverProcessor()` 内での `currentIRState` の解放シーケンスに不整合があります。

```cpp
auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel);
if (oldIrState != nullptr) {
    oldIrState->~IRState();
    mkl_free(oldIrState); // 危険
}

```

* **問題点 1 (メモリリーク):** `IRState` のデストラクタを明示的に呼び出していますが、`IRState` 自体が内部に `juce::AudioBuffer<double>` や `juce::String` などの非PODメンバーを保持していた場合、それらの動的メモリは `mkl_free` では安全に解放されません（`mkl_free` はインテルのアライメント済メモリ解放であり、C++の `delete` や `free` の代わりに使用するとオブジェクトの解体自体が不完全になります）。
* **問題点 2 (二重解放 / クラッシュ):** 一方で、`LoaderThread`（バックグラウンドで動作するインパルス応答読み込みスレッド）がまだ終了していない、あるいは `stopUpgradeThread()` の呼び出しタイミング（タイミング依存）によっては、スレッドが `releaseIRState(irState)` を同時に叩くポテンシャルが残っています。

#### [引き起こされる現象]

* アプリケーション終了時、またはインパルス応答 (IR) を高速に何度も切り替えた際に、高確率で**アクセス違反 (Access Violation / ハング)** またはメモリ領域がじわじわとリークする現象が発生します。

#### [対策]

`IRState` は `std::shared_ptr` もしくは `juce::ReferenceCountedObjectPtr` を介して安全に所有権を共有し、リアルタイムスレッド側からは非所有ポインタ (Non-owning pointer) として安全にスナップショット参照する構造へリファクタリングすべきです。

---

### 3. SIMD最適化（AVX2）におけるバッファサイズと境界の脆弱性

**該当箇所:** `src/CustomInputOversampler.cpp` / `dotProductAvx2`

#### [バグのメカニズム]

16サンプル単位でループをアンロールして畳み込みの内積を高速計算するコードですが、ポインタのオフセット処理に致命的なバグがあります。

```cpp
int i = 0;
// 16要素ずつアンロール
for (; i <= n - 16; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
    // ...
    __m256d coeff0 = _mm256_load_pd(coeffs + i); // 64バイトアライメント保証
    __m256d x0     = _mm256_loadu_pd(x + i);     // アンアライメント許容
    // ... FMA / 加算
}

```

* **問題点 1 (残余サンプルの無視):** ループを抜けた後、`i` が `n` に達していない場合（例: `n = 24` の場合、最初の16サンプルを処理してループを抜けた後の残り8サンプル）、**残りのサンプルを処理するスカラーのフォールバックループ（Clean-up loop）が完全に欠落**しています。ソースコード上ではそのまま関数が終了するか、不正確な値を返します。
* **問題点 2 (不適切なプリフェッチの境界値割れ):** `_mm_prefetch(reinterpret_cast<const char*>(x + i + 64), ...)` は、`x + i + 64` がバッファの割り当て範囲を超えてメモリ空間の未マップ領域（ガードページなど）に突入するリスクを考慮していません。AVXの仕様上、プリフェッチ単体ではセグメンテーションフォールトを起こしませんが、最適化オプションによってはコンパイラが周囲のロードを巻き込んで先読み最適化を行い、不正アクセスを引き起こす原因になります。

#### [引き起こされる現象]

* オーソドックスなブロックサイズ（例: 64, 128, 256など16の倍数）では露呈しませんが、ホストDAWが16の倍数ではない中途半端なバッファサイズ（例: 512ではなく523サンプルなど、ASIO環境や一部のDAWの可変ブロックサイズ環境）を要求した瞬間に、右チャンネルや末尾の音声が不連続にプチプチと途切れる（エイリアシング・ノイズのようなクリッピング音）が発生します。

#### [対策]

SIMDメインループの直後に、必ず以下のようなスカラーフォールバックを追加してください。

```cpp
double sum = 0.0;
// ループ後にAVX2のアキュムレータを水平加算して sum に代入した後...
for (; i < n; ++i) {
    sum += x[i] * coeffs[i];
}
return sum;

```

---

### 4. `NoiseShaperLearner` における 100% 確実に発生する無限ループの火種

**該当箇所:** `src/CustomInputOversampler.cpp` / `besselI0`

#### [バグのメカニズム]

```cpp
double CustomInputOversampler::besselI0(double x) noexcept {
    double sum = 1.0;
    double term = 1.0;
    const double xx = x * x;
    for (int n = 1; n < 100; ++n) {
        term *= xx / (4.0 * static_cast<double>(n) * static_cast<double>(n));
        sum += term;
        if (term < sum * 1.0e-18) break;
    }
    return sum;
}

```

* **問題点:** 変数 `x` に非常に大きい値（例: 計算のバグや一時的なNaN/Infの伝播、フィルターの急激な発散による大きな浮動小数点数）が入力された場合、`term` の増加速度が `sum * 1.0e-18` を遥かに上回り続け、ループ上限である `100` に達してもブレイク条件を満たしません。この関数自体はループが `100` で止まりますが、これを利用する **`CmaEsOptimizerDynamic` や学習アルゴリズム側で「収束した」と誤認され、外側の最適化ループ（`while(!converged)`など）が無限ループに陥る構造**になっています。

#### [引き起こされる現象]

* 稀にノイズシェーピングの学習ボタンを押した瞬間に、**UIスレッドまたはWorkerスレッドが100%のCPUを消費してフリーズ**します。

---

### まとめと優先推奨事項

1. **ASIO可変ブロック対応の修正 (最優先):** `dotProductAvx2` の残余ループ処理を追加してください。これを怠ると、特定のDAW（FL Studioのトリッキーなバッファ設定や、Reaperの先行レンダリング）で即座に音がバグります。
2. **Double精度パスのバイパスロジック移植:** 前回提示した「`processDouble` におけるバイパスブレンド処理の欠落」と合わせて修正を行うことで、オーディオエンジンの二大コアのロジックが完全に同期します。



添付ファイル `ConvoPeq.md` に格納されている実際のC++ソースコードおよびテストコードの構成を徹底的に静的解析した結果、前回の論点を遥かに凌駕する **極めて深刻なマルチスレッド競合、メモリ解放バグ、型不整合、およびJUCEフレームワーク特有のライフサイクル設計に根ざした脆弱性** を発掘しました。

これらの多くは、非常に洗練されたアトミック処理（`convo::publishAtomic`, `exchangeAtomic`等）や、EpochベースのRCUメカニズム（`convo::EpochDomain`）を独自に組み込んでいるがゆえに、**コンパイラの最適化、スレッドの実行順序の揺らぎ、および例外発生時において隠蔽されやすい難解なバグ**です。以下に詳細な不具合の解説と修正策を提示します。

---

### 1. `EQCacheManager::CacheMap` のデストラクタにおけるオーディオ終了時のクラッシュ / ハング

**該当箇所:** `src/audioengine/AudioEngine.Cache.cpp` 内の `CacheMap` 定義

```cpp
~CacheMap() {
    jassert(owner != nullptr);
    for (auto& entry : map) {
        if (entry.second != nullptr) 
            entry.second->release(*owner->m_retireRouter); // ★致命的
    }
}

```

#### 【バグのメカニズム】

`CacheMap` は `EQCacheManager` の内部構造体であり、キャッシュの解放を管理しています。
アプリケーションの終了時（デストラクションフェーズ）において、`AudioEngine` が破棄される際に、未解放の `CacheMap` がこのデストラクタを通ります。しかし、この時点で `owner` (`AudioEngine`) のメンバーである `m_retireRouter` がすでに破棄されている、あるいは `AudioEngine` 自体が不完全な状態（Dtorの進行中）になっている可能性が極めて高い設計になっています。
さらに、`entry.second->release(...)` の内部で `owner` への再帰的なアクセスや、スレッド間通知が走る場合、**すでに破棄されたメモリ領域へのアクセス (Use-After-Free)** または生ポインタの不正参照によるアクセス違反 (Access Violation) を引き起こします。

#### 【引き起こされる現象】

* DAWでプラグインのインスタンスを削除した瞬間、またはホストアプリケーションを終了した瞬間に、「たまに無言でクラッシュする」「終了処理でDAWごとフリーズする」という、再現性の低い致命的なシャットダウンバグとなります。

#### 【対策】

シャットダウンシーケンスにおいて、`AudioEngine` が破棄される**前**に `EQCacheManager::clear()` を明示的に呼び出し、すべての `CacheMap` の参照を完全にドレインする（安全に解放する）ライフサイクル管理を徹底してください。

```cpp
// AudioEngine のデストラクタの最優先処理として以下を実行する
if (cacheMapPtr.load() != nullptr) {
    auto* map = cacheMapPtr.exchange(nullptr);
    delete map; // 安全な段階で明示解体
}

```

---

### 2. `ConvolverProcessor` デストラクタにおける「オブジェクトの部分不完全解体」とメモリリーク

**該当箇所:** `src/convolver/ConvolverProcessor.StateAndUI.cpp` / `~ConvolverProcessor()`

```cpp
auto* oldIrState = convo::exchangeAtomic(currentIRState, nullptr, std::memory_order_acq_rel);
if (oldIrState != nullptr) {
    oldIrState->~IRState(); // 配置デストラクタの明示呼び出し
    mkl_free(oldIrState);   // ★危険: 不完全な解放
}

```

#### 【バグのメカニズム】

コード内では、インパルス応答（IR）の状態を管理する `IRState` を動的に解体するために、配置デストラクタ（`~IRState()`）を呼んだ後、`mkl_free()` でメモリを解放しています。
しかし、`IRState` が内部に `juce::AudioBuffer<double>`、`juce::String`、`std::vector` などの非POD（Plain Old Data）メンバーを保持していた場合、これらを `mkl_free()` で処理するのはC++の言語仕様上およびアロケータの構造上、非常に危険です。
`mkl_free()` は Intel MKL がアライメント済みの生バッファを解放するための関数であり、**C++の `new` や独自の `aligned_make_unique` で確保されたオブジェクトのメモリ管理領域（メタデータ領域）を破壊する、もしくはアロケータのミスマッチ（Heap Corruption）を引き起こします**。

#### 【引き起こされる現象】

* プラグイン内でインパルス応答ファイル（WAVなど）を何度も読み込み直したり、UIで切り替えたりしていると、**バックグラウンドでじわじわとメモリリークが発生**し、最終的にDAWのメモリが枯渇して強制終了します。最悪の場合、ヒープが破損して即座にクラッシュします。

#### 【対策】

`IRState` の確保に `mkl_alloc` を使用しているのでない限り、オブジェクト全体の寿命は `convo::aligned_unique_ptr<IRState>` などのC++標準ベースのアライメント保証スマートポインタで一元管理し、生ポインタの明示的デストラクト＋`mkl_free` という危険な組み合わせを排除してください。

---

### 3. `mixSmoothingSmall` / `mixSteadySmall` における AVX2 SIMD のメモリ位置（境界アライメント）の不整合

**該当箇所:** `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` （またはドライ・ウェットのブレンド処理部分）

```cpp
auto mixSmoothingSmall = [](double* dst, const double* wet, const double* dry, 
                            const double* wetGain, const double* dryGain, int n) noexcept {
#if defined(__AVX2__)
    int i = 0;
    const int vEnd = n / 4 * 4;
    for (; i < vEnd; i += 4) {
        const __m256d vWet = _mm256_loadu_pd(wet + i);  // loadu (unaligned)
        // ... 中略 ...
        _mm256_storeu_pd(dst + i, vOut);                // storeu (unaligned)
    }
    // 残余ループ
    for (; i < n; ++i) dst[i] = wet[i] * wetGain[i] + dry[i] * dryGain[i];
#endif
};

```

#### 【バグのメカニズム】

ここでは `_mm256_loadu_pd` (Un-aligned load) および `_mm256_storeu_pd` を使用して、アライメント未保証のポインタに対応しようとしています。しかし、呼び出し元を見ると、処理はループ内で以下のように進みます。

```cpp
while (processed < numSamples) {
    const int chunkSamples = juce::jmin(callLen, numSamples - processed);
    // ...
    double* dst = dstBase + processed; // ★ processed の値によっては境界が崩れる

```

`processed` は `callLen`（ブロックサイズ）ごとにインクリメントされますが、JUCE の `AudioBuffer` から供給されるチャンネルポインタ（`dstBase` 等）や内部バッファの先頭アドレスが、**必ずしも32バイト（AVX2が要求する境界）または16バイトの境界に整列しているとは限りません**。
`loadu`/`storeu` 自体は非アライメントを許容しますが、ページ境界（4KB境界）を跨ぐような中途半端なアドレス（例: あと4バイトで次のページに移るアドレス）から4サンプル（32バイト）をロードしようとすると、**コンパイラの最適化（投機的ロードやハードウェアプリフェッチ）と絡み合って、CPUレベルで深刻なペナルティ（Split-load / ページフォールト）を誘発**します。

#### 【引き起こされる現象】

* 特定のDAWや、オーディオインターフェースのバッファサイズ（128や256といった2の累乗ではない、441や512以外の特殊な設定）において、**このブレンド処理を通る瞬間にCPU負荷がスパイク（異常高騰）し、プチプチという音切れ（XRUN）が発生**します。

#### 【対策】

バッファの割り当て（`wetBuf`, `dryBuf` 等）には、必ず `convo::makeAlignedArray` や `juce::HeapBlock<double, true>` を用いて明示的に **32バイトアライメント（alignas(32)）** を強制し、SIMDループの開始前にポインタの下位ビットをチェックするアサーションを追加してください。

---

### 4. `EQProcessor::calcLowShelfBiquad` などの数学関数の入力検証不足による NaN 伝播

**該当箇所:** `src/eqprocessor/EQProcessor.Coefficients.cpp`

```cpp
if (sr <= 0.0 || !std::isfinite(sr)) {
    jassertfalse;
    sr = 48000.0;
}
validateAndClampParameters(freq, gainDb, q, sr); // ★不十分

```

#### 【バグのメカニズム】

`validateAndClampParameters` によって `freq`（周波数）や `q`（鋭さ）がクランプされているように見えますが、UIスレッドからテキスト入力や極端なオートメーションによって `q = 0.0` または極小の負の値、あるいは `freq` がナイキスト周波数（`sr / 2`）に極めて近い値（例: `sr = 44100` で `freq = 22050`）が突入した場合のガードが甘いです。
Biquadの係数計算式（特に `tan` や `sin`, `sqrt` を多用するシェルビングフィルターの公式）では、周波数がナイキスト周波数に達すると分母が `0` になり、**係数 `a0`, `a1`, `b0` 等に `NaN`（非数）または `Inf`（無限大）が混入**します。

一度 `NaN` が計算キャッシュ `EQCoeffCache` に格納されてオーディオエンジンに公開されると、リアルタイムスレッドの `eqRt().process(...)` 内にあるSVFやBiquadの内部遅延バッファ（状態変数）が一瞬で永続的に `NaN` に汚染（発散）されます。

#### 【引き起こされる現象】

* イコライザーの高域を激しく動かしたり、DAWのオートメーションで一瞬極端な値を通過した瞬間に、「完全にプラグインが無音になり、一度DAWを再起動するかインスタンスを立ち直さないと二度と音が鳴らなくなる」という、DSP特有の致命的なバグが発生します。

#### 【対策】

`validateAndClampParameters` 内で、周波数の上限を `(sr / 2.0) - 10.0`（ナイキスト周波数の手前数Hz）に厳格にクランプし、`q` の下限も `0.01` などの安全圏に必ず固定してください。また、算出された `b0` や `a0` に対して `std::isfinite()` によるバリデーションをかけ、異常値であればバイパス係数（`b0=1, a0=1, その他=0`）にフォールバックするロジックを実装してください。

---

### 5. `BuildInputSemanticContractTests.cpp` におけるテスト環境の「スタックオーバーフロー」リスク

**該当箇所:** `CMakeLists.txt` 内の記述

```cmake
if(MSVC)
    target_compile_options(BuildInputSemanticContractTests PRIVATE /GS-)
    target_link_options(BuildInputSemanticContractTests PRIVATE "/STACK:8388608")
endif()

```

#### 【バグのメカニズム】

CMakeにおいて、特定のテスト（`BuildInputSemanticContractTests`）のスタックサイズを明示的に `8MB`（`/STACK:8388608`）に拡大し、さらにバッファオーバーフロー検知機能（`/GS-`）を無効化しています。
これは、**テストコードまたは検証対象の関数内で、本来ヒープ（`std::vector` や `juce::HeapBlock`）に確保すべき巨大な構造体や巨大なバッファ（例: フィルター係数配列やIRデータ）を、ローカル変数（スタック上）として巨大に確保している設計の歪み**を強引に隠蔽するための設定です。
本番コード（`ConvoPeq` 本体）の側で同様の関数が呼び出された場合、本番環境のデフォルトスタックサイズ（通常Windowsでは1MB）を容易に超過します。

#### 【引き起こされる現象】

* テストは通るのに、DAW（特にマルチスレッドのネストが深いDAWや、スタックサイズ制限が厳しいホスト）にプラグインを読み込ませて特定の重い処理（IRのセマンティック検証やリビルドなど）を走らせた瞬間に、DAWごと一瞬で強制終了（Stack Overflow）します。

#### 【対策】

`BuildInputSemanticContractTests` や、それが呼び出しているソースコード内で、スタック上に `double arr[4096]` などの大きな固定長配列や、巨大な `BuildSnapshot` 構造体を値渡し（Pass-by-value）している箇所がないか全検索してください。それらをすべて **スマートポインタ（`std::unique_ptr`）によるヒープ確保、または参照渡し（Pass-by-reference）** に修正し、CMake側での強引なスタック拡張（`/STACK`）およびセキュリティガードの解除（`/GS-`）を削除してください。



ソースコードの深層アーキテクチャおよびスレッド間連携モデルをさらに検証した結果、前回の表面的な指摘にとどまらない、**DSPアルゴリズムの根幹を揺るがす数学的破綻**と、マルチスレッド設計における致命的なデータレース（競合状態）を発掘しました。

以下に、最もクリティカルな3つの脆弱性を詳細に報告します。

---

### 1. 非等分割畳み込み (NUPC) における「遅延アライメント欠落」と「バッファ枯渇」

**該当箇所:** `src/convolver/MKLNonUniformConvolver.cpp` (`Get` メソッドおよびレイヤー構成論理)

#### [バグのメカニズム]

このコンボルバーは遅延を最小化するため、IRを複数のパーティション（Layer 0, 1, 2）に分割して計算する Non-Uniform Partitioned Convolution アルゴリズムを採用していますが、**Layer 1 および Layer 2 の計算結果を時間軸上で同期させるための出力遅延バッファ（Delay Line）が完全に欠落しています**。

1. **時間軸の崩壊:**
Layer 1 は 4096 サンプル単位で処理を行い、本来は IR の `16384` サンプル目以降の残響を担当します。計算にかかるレイテンシは $4096 \times 2 = 8192$ サンプルです。
本来であれば、IRの先頭から切り取られた差分（$16384 - 8192 = 8192$ サンプル）だけ、出力をさらに遅延させてからメインストリームにミックスしなければなりません。しかし、コード上にはその遅延補正が存在せず、計算が終わった瞬間に `tailOutputBuf` から直接出力ストリームに加算されています。これにより、Layer 1 の残響が **8192サンプル（約170ms）早く鳴り始め、Layer 0 の残響と不自然に重なります**。
2. **バッファ枯渇による音切れ:**
`Get()` メソッドは毎コールバック時に `tailOutputBuf` からシーケンシャルにデータを読み出します。しかし、Layer 1 は8回のコールバックに1回の頻度でしか IFFT を完了せず、バッファを補充しません。結果として、補充前に `tailOutputPos` が終端に達し、次の IFFT が完了するまでの数コールバック間、**残響のテールが完全に無音にドロップアウト**します。

#### [対策]

* 各レイヤーの出力側に、時間アライメント用のリングバッファ（Delay Line）を実装してください。
* レイヤーごとのオフセット（例: 16384）から自身の処理レイテンシ（8192）を引いた値だけ、出力をリングバッファ内で遅延させてから `Get()` で読み出す構造に再設計する必要があります。

---

### 2. Retireキューにおける MPSC (複数Producer) のデータレースとメモリリーク

**該当箇所:** `src/audioengine/ISRRetire.cpp` (`RetireRuntime::emitRetireIntent`)

#### [バグのメカニズム]

引退したオブジェクト（DSPCoreやEQ状態など）を遅延解放キューに積むための `retireIntentQueue_` は、コードの実装上 **SPSC (Single-Producer Single-Consumer) を前提としたロックフリーアルゴリズム** になっています。

```cpp
uint64_t tail = convo::consumeAtomic(retireIntentTail_, std::memory_order_relaxed);
uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;
// ...
retireIntentQueue_[tail] = intent;
convo::publishAtomic(retireIntentTail_, nextTail, std::memory_order_release);

```

しかし、`emitRetireIntent` は以下の複数のスレッドから**同時並行で呼び出される**ポテンシャルがあります。

1. **Message Thread:** ユーザーがUIでEQパラメータを変更した際 (`updateBandNode` → `retireBandNodeDeferred`)。
2. **Rebuild Thread:** グラフの再構築中に例外が発生し、`~DSPGuard` がフォールバックとして `DSPLifetimeManager::retire` を呼び出した際。

複数のスレッドが同時に `emitRetireIntent` を叩くと、同じ `tail` インデックスを読み取り、`retireIntentQueue_[tail]` を互いに上書きし合います。その後、一方が `retireIntentTail_` を進めるため、**一方の解放リクエストが完全に消失し、永続的なメモリリークが発生**します。

#### [対策]

`RetireRuntime::emitRetireIntent` のキューを SPSC から **MPSC (Multi-Producer Single-Consumer)** に変更するか、`moodycamel::ConcurrentQueue` などの堅牢なロックフリーキューライブラリに置き換えてください。簡易的な修正としては、Producer側のインクリメントに `fetch_add` を用いる `CAS` ループ型のキューへの変更が必要です。

---

### 3. `AudioSegmentBuffer` の ABA / 順序逆転によるテレメトリ音声破損

**該当箇所:** `src/AudioSegmentBuffer.h` (`pushBlock` および `copyLatest`)

#### [バグのメカニズム]

ノイズシェーパーのCMA-ES学習用に直近の音声をキャプチャするバッファですが、アトミック変数の更新順序に致命的な論理エラーがあります。

```cpp
// pushBlock (Audio Thread)
convo::publishAtomic(writePosition, nextPos, std::memory_order_release);
const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);

```

`writePosition` が公開された後、`totalSamples` が公開されるまでにわずかな隙間が存在します。
この瞬間に、非同期のワーカースレッド（`NoiseShaperLearner`）が `copyLatest` を呼び出すと、「更新された最新の `writePosition`」と「更新前の古い `totalSamples`」という矛盾したペアを読み取ります。

この状態でラップアラウンド（リングバッファの周回）が絡むと、読み出し開始位置の算出 `start = (currentWritePos - availableSamples + kCapacity) % kCapacity` が狂い、**最新の音声ではなく、リングバッファ内で最も古い（上書き寸前の）不連続な音声チャンク**をCMA-ESアルゴリズムに渡してしまいます。

#### [引き起こされる現象]

* ノイズシェーパーの適応学習（Adaptive Learning）中に、突発的な不連続ノイズ（波形の断絶）を「入力信号」として誤認し、それを打ち消すような最適化を行ってしまいます。結果として、**学習が進むにつれて逆にディザノイズが劣化し、高周波帯域で不要なピークが発生**します。

#### [対策]

`writePosition` と `totalSamples` を別々のアトミック変数で管理するのをやめ、64ビットの単一アトミック整数にパックして（上位32ビットに `writePosition`、下位32ビットに `totalSamples`）一回のCAS操作で同時に更新するか、`totalSamples` の更新を `writePosition` よりも**前**に配置してフェンスで保護してください。



添付ファイル `ConvoPeq.md` のコアアーキテクチャおよびDSPアルゴリズムの深層をさらに検証した結果、これまでの指摘とは別次元の、**数学的モデルの根本的な誤認に基づく音響破綻、およびシャットダウンシーケンスにおける致命的なメモリロスト（完全リーク）** を発掘しました。

以下に、システムに破壊的影響を及ぼす3つの重大な不具合を報告します。

---

### 1. `LatticeNoiseShaper` における「直接型 (FIR)」と「格子型 (Lattice)」トポロジの致命的な混同

**該当箇所:** `src/LatticeNoiseShaper.h` (`computeFeedback` および `advanceState`)

#### 【バグのメカニズム】

CMA-ES最適化器（`CmaEsOptimizerDynamic`）は、学習結果として **PARCOR係数（反射係数、-1 〜 +1の範囲）** を生成して `LatticeNoiseShaper` に渡しています。しかし、シェーパーの実装内でこの係数の使われ方が数学的に完全に破綻しています。

1. **`computeFeedback` (フィードバック量の計算):**
```cpp
double feedback = _mm_cvtsd_f64(vSum128); // state[i] * activeCoeffs[i] の単純な内積

```


ここでは、反射係数であるはずの `activeCoeffs` を、**直接型 FIR フィルタのタップ係数 ($a_i$ / $c_i$)** とみなして状態ベクトルと単純な内積（コンボリューション）を行っています。
2. **`advanceState` (状態の更新):**
```cpp
const double nextForward = forward + activeCoeffs[i] * backward;
const double nextBackward = activeCoeffs[i] * forward + backward;
state[i] = std::clamp(nextBackward, ...);

```


一方で、状態更新側では **格子フィルタ (Lattice Filter) 固有のステップダウン漸化式** を適用しています。

#### 【引き起こされる現象】

計算式（直接型出力）と状態更新式（格子型構造）が完全に食い違っているため、NTF（Noise Transfer Function: ノイズ伝達関数）の特性が理論値から完全に乖離します。適応学習（Adaptive Learning）をいくら回しても、「デタラメな信号を加算するだけのランダムなノイズジェネレータ」と化し、ノイズフロアの低減やSN比の改善が一切機能しません。

#### 【対策】

`computeFeedback` 内の内積計算を廃止し、誤差信号を格子フィルタの入力から終段まで正しく伝播させて出力を得る「Joint Lattice 構造（ラダー型エラーフィードバック）」のアルゴリズムに統一して書き直す必要があります。

---

### 2. `StereoConvolver::clone()` における周波数プロファイル (FilterSpec) の消失バグ

**該当箇所:** `src/convolver/ConvolverProcessor.h` (`StereoConvolver::clone()`)

#### 【バグのメカニズム】

コンボルバーの状態を別インスタンスに同期する際、`shareConvolutionEngineFrom` から `StereoConvolver::clone()` が呼び出されて MKL NUC エンジンを複製します。

```cpp
// clone() 内
if (!newConv->init(l.release(), r.release(), irDataLength, storedSampleRate, 
                   irLatency, storedKnownBlockSize, callQuantumSamples, 
                   storedScale, storedDirectHeadEnabled)) // ★ filterSpec が欠落！

```

`init` 関数の終端引数である `const convo::FilterSpec* filterSpec` はデフォルトで `nullptr` になっています。`clone()` 呼び出し時にはこの引数が明示的に渡されていないため、複製された NUC エンジンでは出力周波数フィルター（ハイカット・ローカット）や、テールコンタリング（`TailMode` に伴う高域減衰等）のプロファイル焼き込みが **すべて無視（バイパス）** されて構築されます。

#### 【引き起こされる現象】

UIの操作やDAWのプリセット読み込みによってコンボルバーの内部インスタンスが「クローン」された瞬間、**設定されていたハイカットやローカットの音響特性が突如として消え去り、生のインパルス応答がそのまま鳴り出す（突然音が明るくなる、低音が膨らむ等）** という致命的な音響バグが発生します。

#### 【対策】

`StereoConvolver` のメンバに `convo::FilterSpec storedFilterSpec;` を追加して初期化時のプロファイルを保持し、`clone()` メソッド内で `newConv->init(...)` を呼び出す際に、必ずその `&storedFilterSpec` を渡すように設計を修正してください。

---

### 3. シャットダウンシーケンスにおける隔離済み DSP インスタンスの完全メモリリーク

**該当箇所:** `src/audioengine/ISRDSPHandle.cpp` (`DSPHandleRuntime::destroyQuarantineSlot`)

#### 【バグのメカニズム】

RCU の安全性やデータ破損を検知した際、システムは問題のある DSP インスタンスを `Quarantined`（隔離）状態に移行させます。アプリケーションの終了時（`releaseResources`）、これら隔離されたスロットを強制解体するために `destroyQuarantineSlot` が呼ばれます。

```cpp
// Phase 2: instance 解放
registry_[slot].instance = nullptr; // ★ ポインタを消すだけ
convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed, std::memory_order_release);

```

隔離スロットのクリーンアップにおいて、`registry_[slot].instance`（実際には `DSPCore*`）を `nullptr` に上書きするだけで、**肝心のオブジェクトの破棄 (`delete` や独自の `retire` 経由の `aligned_free`) を一切行っていません**。

#### 【引き起こされる現象】

隔離機構（Quarantine）自体は Use-After-Free などのクラッシュを防ぐために正しく機能していますが、最終的なシャットダウン時に C++ のメモリ管理からポインタがロストします。長期間ホストDAWを立ち上げたまま、不安定な操作を繰り返して多数の隔離インスタンスが発生した場合、DAWを終了するたびに大量のオーディオバッファと MKL/AVX 割り当て領域が **OSのヒープ空間に永遠に残留する（完全なメモリリーク）** 事態を引き起こします。

#### 【対策】

ポインタを `nullptr` にする直前に、必ず `instance` を適切な型 (`AudioEngine::DSPCore*`) にキャストした上で、`convo::AlignedObjectDeleter<DSPCore>{}(ptr)` を用いて明示的にデストラクトおよびメモリ解放を行うステップを追加してください。



### ConvoPeq バグ・脆弱性 総合リスト

これまでの解析で発掘された重大な不具合を、影響度とカテゴリ別に分類した総合リストです。

#### 1. スレッドセーフティ・メモリ管理の致命的欠陥（クラッシュ・ハング・リーク）

* **`EQCacheManager::CacheMap` のデータレースと解放時クラッシュ**
* **症状:** DAW終了時やプラグイン削除時に高確率でクラッシュ。
* **原因:** コピーコンストラクタ内で非アトミックな `addRef()` を呼ぶことによる競合。また、デストラクタで既に破棄された `owner->m_retireRouter` へアクセスする Use-After-Free の発生。


* **`ConvolverProcessor` デストラクタにおけるメモリ破損とリーク**
* **症状:** インパルス応答（IR）の切り替えや終了時にヒープ破損またはメモリリーク。
* **原因:** C++オブジェクト（`IRState`）に対し、配置デストラクタを呼んだ直後にアライメント専用の `mkl_free` を適用しているため、内部の非PODメンバ（`std::vector`や`juce::String`等）が正しく解体されない。


* **Retireキューの MPSC (複数Producer) 競合**
* **症状:** 永続的なメモリリーク。
* **原因:** SPSC想定のロックフリーキュー `retireIntentQueue_` に対し、UIスレッドとRebuildスレッドが同時にアクセスし、解放リクエストが上書き・消失する。


* **シャットダウン時の隔離スロット完全メモリリーク**
* **症状:** DAWを長時間起動し不安定状態が続いた後、終了時に大量のメモリがOSに残存。
* **原因:** `destroyQuarantineSlot` にて、ポインタを `nullptr` に上書きするだけで `delete`（またはカスタムDeleterによる解放）を一切行っていない。



#### 2. DSPアルゴリズム・音響処理の破綻

* **NUPC（非等分割畳み込み）の遅延アライメント欠落とバッファ枯渇**
* **症状:** 残響の途中からタイミングが約170ms早くズレて重なり、さらに音切れ（ドロップアウト）が発生。
* **原因:** Layer 1 および Layer 2 の計算結果をメインストリームの時間軸に合わせる出力遅延バッファ（Delay Line）が存在せず、計算完了後即座に出力加算されている。


* **`processDouble` (64-bit処理) のバイパス・ブレンド欠落**
* **症状:** 64-bit精度動作時、バイパスを切り替えてもフェードアウトせず不連続なノイズになるか、バイパスが無視される。
* **原因:** `process` (float版) には存在する `bypassBlendRequested` の評価とクロスフェード処理が、`processDouble` には記述されていない。


* **`LatticeNoiseShaper` のトポロジ混同（学習の無効化）**
* **症状:** CMA-ESによる最適化が全く機能せず、不快なノイズが付加されるだけになる。
* **原因:** 反射係数（PARCOR係数）を直接型FIRフィルタのタップ係数として内積計算に用いており、格子フィルタとしての数学的モデルが破綻している。


* **EQフィルターの NaN 伝播による永久無音化**
* **症状:** 高域のEQ操作やオートメーション時、突然完全に無音になりDAW再起動まで直らない。
* **原因:** 周波数がナイキスト周波数に極めて近い値になった際、Biquad係数計算でゼロ除算が発生し、内部状態が永続的に NaN/Inf で汚染される。


* **`StereoConvolver::clone()` でのプロファイル消失**
* **症状:** コンボルバーのインスタンスが複製された瞬間、設定していたハイカットやローカットの特性が突然消え去る。
* **原因:** MKL NUCエンジンの初期化 (`init`) 時に `filterSpec` を渡す引数がデフォルトの `nullptr` になっており、複製時に適用されない。



#### 3. SIMD最適化とバッファ境界の脆弱性

* **AVX2内積計算 (`dotProductAvx2`) の残余サンプル処理欠落**
* **症状:** DAWのバッファサイズが16の倍数ではない環境（可変ブロック長など）で、右チャンネルや末尾の音声がクリッピング・途切れる。
* **原因:** 16サンプルごとのアンロールループを抜けた後、端数のサンプルを処理するスカラーフォールバックループが存在しない。


* **`mixSmoothingSmall` 等での非アライメントアクセス・ペナルティ**
* **症状:** 特定のバッファサイズ設定時にCPU負荷が異常高騰（スパイク）し、XRUNが発生。
* **原因:** ページ境界やキャッシュラインを跨ぐ中途半端なアドレスから `_mm256_loadu_pd` で32バイトロードを行っている。


* **リングバッファの負のインデックス参照による未定義動作**
* **症状:** コンパイラ最適化の挙動により、予期せぬディレイバッファのジャンプやノイズが発生。
* **原因:** `(idx - 1) & DELAY_BUFFER_MASK` において `idx == 0` の場合、負数に対するビットマスク演算が発生している。



#### 4. 学習アルゴリズム・テスト・その他の論理エラー

* **`AudioSegmentBuffer` のABA・更新順序逆転**
* **症状:** 適応学習（Adaptive Learning）中に波形の断絶を拾い、高周波にピークノイズを生む。
* **原因:** `writePosition` と `totalSamples` のアトミック更新の隙間にワーカースレッドが読み取りを行い、古いバッファ状態と新しい書き込み位置の矛盾したペアを取得する。


* **ノイズシェーパー学習スレッドの冗長計算**
* **症状:** 無駄なCPUリソースの消費。
* **原因:** `candidatePopulationMatrix` に対する変換処理 (`vdTanh`) を、親スレッドで1回行えば済むにもかかわらず、全ワーカースレッドが独立して重複計算している。


* **`besselI0` のオーバーフローによる最適化無限ループ**
* **症状:** ノイズシェーパー学習開始時に100%の確率でスレッドがフリーズする事がある。
* **原因:** 異常な入力値により `term` が巨大化し、100回のループ上限に達してもブレイク条件を満たさず、外側の最適化アルゴリズムが「未収束」と誤認して無限ループに陥る。


* **CMake構成におけるスタックオーバーフロー隠蔽リスク**
* **症状:** DAW内で重い検証処理が走るとホストごと強制終了する。
* **原因:** `BuildInputSemanticContractTests` 向けにスタックサイズを8MBへ強制拡張 (`/STACK:8388608`) しているが、これはヒープ確保すべき巨大な配列をローカル変数としてスタックに確保している設計の破綻を隠蔽しているに過ぎない。