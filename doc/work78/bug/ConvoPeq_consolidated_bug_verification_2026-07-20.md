# ConvoPeq 統合バグリスト 検証レポート

> **検証日**: 2026-07-20
> **検証対象**: `ConvoPeq_consolidated_bug_list_2026-07-18.md`（48件）
> **検証方法**: ソースコード読解 + grep/ripgrep + コンテキスト分析
> **結論**: 48件中 **26件が確認（有効）**, **10件が修正済み**, **7件が無効**, **5件が一部確認/要追加調査**

---

## サマリ

| 検証結果 | 件数 | 内訳 |
|----------|------|------|
| **確認（有効）** | 26 | C-01, C-02, C-03, C-07, H-04, H-11, M-01, M-02, M-03, M-05, M-08, M-10, M-11, M-12, M-13, M-14, L-01, L-02, L-03, L-04, I-03, I-06, I-07, I-08, I-09, I-10, I-11 |
| **修正済み** | 10 | C-06, H-06, H-07, H-08, H-09, H-10, C-08 (要 revision), M-09, I-02 (一部), I-01 (一部) |
| **無効** | 7 | C-04, H-01, H-02, H-05, C-05 (実害なし), M-09, I-02 (DSPHandle) |
| **要追加調査** | 5 | C-05, H-03, I-01, I-02, I-04/I-05 |

---

## Critical（8件）

### C-01: `CustomInputOversampler.cpp` — プリフェッチがガードページを越える
**検証結果: ✅ 確認（有効）**

ソース確認:
- Line 174: `_mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);`
- ループ: `for (; i <= n - 16; i += 16)` → 最大 `i = convCount - 16`
- プリフェッチ範囲: `x + convCount + 48`
- バッファ: `xWindow = history + idx - (convCount - 1)`、`capacity = historyUpKeep + maxInputSamples + 16`
- `idx = capacity - 1` のとき `x` から使える要素は `convCount` のみ
- **結果: プリフェッチがバッファ末尾から48要素超過。guard page接触でXRUN発生リスクあり。**

### C-02: `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` — 量子化オーバーフロー
**検証結果: ✅ 確認（有効）**

ソース確認（Fixed15TapNoiseShaper.h lines 314-333）:
```cpp
const double maxV = 1.0 - (1.0 / invScale);  // 16bit: 0.99997
if (v > maxV) v = maxV;                       // clamp
v += (u1 + u2 - 1.0) * scale;                // dither: max +1/32768
return q * scale;                              // 0.99997 + 0.00003 = 1.0 → 32768
```
- **結果: クランプ→ディザ順序で `maxV + scale = 1.0` となり、int16で32768にオーバーフロー。破壊音発生。**

### C-03: `Fixed15TapNoiseShaper.h` — プリセット線形補間で不安定化
**検証結果: ✅ 確認（有効）**

ソース確認（lines 96-102）:
```cpp
for (int i = 0; i < ORDER; ++i) {
    interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];  // 格子係数の線形補間
}
```
- 補間後 `isStable()` チェックなし
- `kErrorStateThreshold = 1e6` で発散検知 → `needsReset` で無音化
- **結果: 中間レートで極が単位円外に出るリスク。エラー検知までに最大1ブロックのNaN/無音。**

### C-04: `MKLNonUniformConvolver.cpp` — Directパス memset 境界超過
**検証結果: ❌ 無効**

ソース確認（line 762）:
```cpp
m_directTapCount = (enableDirectHead ? std::min(irLen, std::min(directPart, kMaxDirectTaps)) : 0);
```
- `m_directTapCount` は常に `irLen` 以下
- Line 802: `memset(impulseForFft.get(), 0, m_directTapCount * sizeof(double))` は安全
- **結果: `std::min(irLen, ...)` により境界超過は発生しない。バグ報告は誤り。**

### C-05: `SafeStateSwapper.h` — 2-step bump 同時実行によるUse-After-Free
**検証結果: ⚠️ 一部確認（要追加調査）**

ソース確認:
- `swap()` の呼び出し元: `ConvolverProcessor.StateAndUI.cpp:1019` のみ（1箇所）
- `fallbackMutex` は `fallbackQueue` のみ保護（line 112）
- `epoch` bump と `exchangeAtomic` は mutex 外
- **EQProcessor は `swap()` を直接呼ばない**（ISRRetireRouter 経由）
- **結果: 現在は単一Writer（ConvolverProcessor）のみ。競合は未確認だが、設計上の脆弱性は残存。**

### C-06: `IRAnalyzer.cpp` — FFT binループ範囲外読み出し
**検証結果: ✅ 修正済み**

ソース確認（lines 119-130）:
```cpp
for (int bin = 0; bin <= numBins; ++bin)  // <= で正しい
{
    if (bin == 0)        { re = out[0]; im = 0.0; }
    else if (bin == numBins) { re = out[1]; im = 0.0; }  // Nyquist は out[1]
    else                 { re = out[2 * bin]; im = out[2 * bin + 1]; }
}
```
- **結果: `bin <= numBins` + CCSフォーマットの正しいアクセス。修正済み。**

### C-07: `IRConverter.cpp` — 過大ジャンプ保護が常に無が常に無効
**検証結果: ✅ 確認（有効）**

ソース確認（lines 148-149）:
```cpp
const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*currentIr, result.scaleFactor);  // ← *currentIr が正しい
```
- **結果: 両行で `*currentIr` を使用。2行目は `*ir`（新IR）を使うべき。ジャンプ保護が無効。**

### C-08: `AutoGainPlanner.cpp` — QSurge係数が常に上限張り付き
**検証結果: ⚠️ 要revision（コード変更済み）**

ソース確認:
- 現行コードに `peakingSurge` / `0.15` 定数なし
- `kConvFirstInputCeiling` は削除（header: "固定 Ceiling 廃止"）
- 現行のマージン: `kMarginEqFirst=1.5`, `kMarginConvFirst=1.0`, `kMarginInterStage=1.0`
- **結果: コードリファクタリングで問題が変化。旧バグ描述は該当しないが、マージン設計は要継続監視。**

---

## High（11件）

### H-01: `LockFreeAudioRingBuffer.h` — チャンネル拡張競合
**検証結果: ❌ 無効**

ソース確認（lines 48-93）:
- `push()` で channel 0 と channel 1 を両方書いてから `writeIndex` を更新
- `channelsToWrite == 1 && numChannels > 1` のとき mono→stereo コピー
- **writeIndex は両チャンネル書き込み後に1回だけ更新**
- **結果: 読み取り側は writeIndex 更新後に新しいデータを見る。L/R位相差は発生しない。**

### H-02: `CustomInputOversampler.cpp` — サイレンス最適化でDCリーク
**検証結果: ❌ 無効**

ソース確認（lines 569-584）:
```cpp
if (inputSilent && historySilent)
{
    juce::FloatVectorOperations::clear(output, outSamples);
    juce::FloatVectorOperations::clear(history, keep);
    return;  // ← ここで早期リターン
}
```
- サイレンスパスは早期リターンするため、`memmove` は実行されない
- 次回の呼出で `copy(history + keep, input, inputSamples)` で上書き
- **結果: サイレンスパスで `keep` 以降のデータが残っても、使用されない。無害。**

### H-03: `LatticeNoiseShaper.h` — ブロック末尾での状態クランプ遅延
**検証結果: ⚠️ 一部確認（要追加調査）**

- `clampStateSIMD` をブロック終了時に1回のみ呼出
- `kStateLimit=1e12` は大きいが、`kOrder=9` の不安定係数でブロック内発散の可能性
- **結果: 設計上は正しいが、kStateLimit の妥当性は要評価。**

### H-04: `OutputFilter.cpp` — HPFのナイキストチェック欠落
**検証結果: ✅ 確認（有効）**

ソース確認（lines 46-62）:
```cpp
BiquadCoeff OutputFilter::makeHPF(double fc, double Q, double fs) noexcept
{
    if (fc <= 0.0 || Q <= 0.0 || fs <= 0.0)  // ← 上限チェックなし
        return makeIdentity();
    const double w0 = 2.0 * pi * fc / fs;
    // fc=0.4999*fs で w0≈pi, sin(w0)≈0, alpha≈0 → b0≈0, a1≈2
}
```
- `makeLPF` には `fc >= nyq` チェックがあるが `makeHPF` にはない
- **結果: fc がナイキストに近い場合、極が単位円上に乗り発振リスク。**

### H-05: `UltraHighRateDCBlocker.h` — 超高レートでの精度消失
**検証結果: ❌ 無効**

ソース確認:
- `alpha = -std::expm1(-omega)` で桁落ち防止済み
- `killDenormal` は `1e-20` 未満を0に（仕様通り）
- 768kHz/fc=20Hz で `alpha≈1.6e-4` → 52bit double で加算可能
- **結果: `std::expm1()` 使用で桁落ちは回避。Kahan補償は不要。**

### H-06: `DSPCoreDouble.cpp` — softClipのprevSample保存バグ
**検証結果: ✅ 修正済み**

ソース確認（lines 208-220）:
```cpp
// AVX2 パス:
const double nextPrev = data[i + 3]; // store前に元の入力値を退避
_mm256_storeu_pd(data + i, result);
prevScalar = nextPrev;

// スカラーパス:
prevScalar = inputVal; // 修正: 処理前の生入力値を保存
```
- **結果: AVX2パスはstore前に読み取り、スカラーパスは `inputVal` を保存。両方修正済み。**

### H-07: `EQProcessor.Processing.cpp` — `calculateRMS` のSSE除算で0除算
**検証結果: ✅ 修正済み**

ソース確認（line 23）:
```cpp
if (data == nullptr || numSamples <= 0)
    return 0.0;  // ← 早期リターンあり
```
- **結果: `numSamples <= 0` で早期リターン。0除算は発生しない。**

### H-08: `AutoGainPlanner.cpp` — Conv→EQ 時に透過でも -6dB 強制
**検証結果: ✅ 修正済み**

ソース確認（lines 70-76）:
```cpp
// Conv→PEQ: 固定Ceiling廃止、マージンのみで保護
inputDb = -(std::max(0.0f, convBoost - kMarginConvFirst)
          + std::max(0.0f, eqBoost - kMarginInterStage)
          + qMargin);
```
- `kConvFirstInputCeiling` は削除
- **結果: 固定Ceilingなし。マージンベースの動的保護に変更済み。**

### H-09: `AutoGainPlanner.cpp` — Conv only で -6dB 天井なし
**検証結果: ✅ 修正済み**

- 同上。固定Ceiling廃止済み。

### H-10: `DSPCoreDouble.cpp` — `convolverInputTrimGain` が Conv→EQ で完全無視
**検証結果: ✅ 修正済み**

ソース確認（lines 436-446）:
```cpp
if (!state.convBypassed)
{
    if (state.convolverInputTrimGain != 1.0)
    {
        // trim適用 — 処理順序チェックなし
    }
    convolverRt().process(processBlock);
}
```
- **結果: `order` チェックなし。Conv→EQ でも trim が適用される。修正済み。**

### H-11: `DeviceSettings.cpp` — タイマー5Hzが編集中を上書き
**検証結果: ✅ 確認（有効）**

ソース確認（grep結果）:
- `updateGainStagingDisplay` で `getText()` と `engine.getInputHeadroomDb()` の比較
- 差異があれば `setText(dontSendNotification)` で上書き
- `hasKeyboardFocus()` チェックなし
- **結果: ユーザータイピング中にタイマーで入力が上書きされる。**

---

## Medium（14件）

### M-01: `IRAnalyzer.cpp` — MKLバッファに `std::make_unique` 使用
**検証結果: ✅ 確認（有効）**

ソース確認（lines 63-79）:
```cpp
double estimateMaxFrequencyResponseGain(...) noexcept  // ← noexcept
{
    // ...
    auto tukeyWindow = std::make_unique<double[]>(...);  // ← bad_alloc で std::terminate
    auto mags = std::make_unique<double[]>(...);
}
```
- **結果: `noexcept` 関数内で `std::make_unique` → OOM時に `std::terminate`。規約違反。**

### M-02: RT Capability/Allocator Firewall が未接続
**検証結果: ✅ 確認（有効）**

grep結果:
- `auditPublishAttempt` / `onAllocAttempt` の定義は存在
- 呼出し箇所: **ゼロ**
- **結果: 安全網が未接続。RT違反が検知できない。**

### M-03: LifecycleToken が発行されるだけで検証されていない
**検証結果: ✅ 確認（有効）**

ソース確認（lines 48-115）:
```cpp
void LifecycleIsolationRuntime::leavePrepare(LifecycleToken token) {
    // token 引数を使用せず、phase_ のみチェック
}
void LifecycleIsolationRuntime::leaveAudioCallback(LifecycleToken token) {
    // token 引数を使用せず、phase_ のみチェック
}
```
- **結果: `leave*` 関数で `token.epochId` が未検証。設計上の安全機構が未実装。**

### M-04: SVF `tan` 発散ガード欠落
**検証結果: ⚠️ 一部確認（要追加調査）**

ソース確認:
- `g = std::tan(pi * freq / sr)` の後に `g` の上限チェックなし
- `freq` は `nyquist * 0.95` でクランプだが `g` は制限なし
- **結果: `g` の上限ガード欠落。高域で係数急変の可能性。**

### M-05: 大ブロック無音化
**検証結果: ✅ 確認（有効）**

ソース確認（lines 289-291）:
```cpp
if (numSamples > dsp->maxSamplesPerBlock)
{
    buffer.clear();
    return;  // ← 1ブロック無音
}
```
- **結果: チャンク分割せず即無音。可変ブロックホストで不定期無音。**

### M-06: MKL DFTIスケーリングの二重適用
**検証結果: ⚠️ 要追加調査**

- MKL/IPP パス混在時のスケーリング確認が必要
- **結果: 判定保留（追加調査必要）**

### M-07: `IRConverter.cpp` `size_t` オーバーフロー
**検証結果: ⚠️ 要追加調査**

- `bytes = numChannels * numSamples * sizeof(double)` の32bit int計算確認が必要
- **結果: 判定保留（追加調査必要）**

### M-08: キャッシュハッシュ衝突で誤ったIR再利用
**検証結果: ✅ 確認（有効）**

ソース確認:
- `StateKey` のハッシュ: ファイルパス + サンプルレート + fftSize + phaseMode + partitionSize
- `mtime` なし
- **結果: IRファイル上書き時にキャッシュヒットで古いIRを使用する。**

### M-09: MMCSSハンドルリーク
**検証結果: ❌ 無効**

ソース確認:
- `thread_local HANDLE t_mmcssHandle = nullptr;`
- `thread_local bool t_mmcssTried = false;` — 再登録防止
- `AvRevertMmThreadCharacteristics` で正しく開放
- **結果: `thread_local` + `t_mmcssTried` ガードで再登録は防止。リークなし。**

### M-10: `fc_hc` / `fc_lp` サンプルレート分岐が粗い
**検証結果: ✅ 確認（有効）**

ソース確認（lines 79-80）:
```cpp
const double fc_hc = (sampleRate <= 48000.0) ? 19000.0 : 22000.0;
const double fc_lp = (sampleRate <= 48000.0) ? 19000.0 : 24000.0;
```
- **結果: 2分岐のみ。88.2k→19kと不連続切替。音質劣化の可能性。**

### M-11: `computeEstimatedMaxGainDb` LPF/HPFを常にブースト扱い
**検証結果: ✅ 確認（有効）**

- LPF/HPF を `gainBoosting=true` で処理
- Q=0.707のButterworthはブーストなし
- **結果: AutoGain過剰ヘッドルーム。**

### M-12: `computeEstimatedMaxGainDb` totalGainDbの二重カウント
**検証結果: ✅ 確認（有効）**

- Master Gain を `maxLinearGain` に乗算 → makeupと合算で二重補正
- **結果: Master Gain+6dBでEQピーク無くても-6dBヘッドルーム。**

### M-13: 適応サンプリングのスキップ条件が逆
**検証結果: ✅ 確認（有効）**

- `if (range > freq*0.5) continue` で低Q広帯域をスキップ
- **結果: シェルフEQの推定誤差。**

### M-14: `BuildAnalysis` 封印失敗時のサイレントフォールバック
**検証結果: ✅ 確認（有効）**

- `sealBuildAnalysis` が `BuildAnalysis{}` を返す場合、呼び出し側が未チェック
- **結果: 解析失敗時にAutoGainはフラットと誤認。**

---

## Low（4件）

### L-01: `CallbackTimingHistory` リングバッファのオフバイワン
**検証結果: ✅ 確認（有効）**

- `fetchAddAtomic` は pre-increment 値を返す → `(wc-1)%32` で1つずれる
- **結果: 診断ビルドのみ。音声処理に影響なし。**

### L-02: `processBand` と `processBandStereo` の異常値ハンドリング不一致
**検証結果: ✅ 確認（有効）**

- スカラー版は `|output|>=1e15` で無音化、SIMD版は有限巨大値のみクランプ
- **結果: チャンネルモード次第で挙動不一致。**

### L-03: `ISRRetireRouter`: null deleter のサイレント成功扱い
**検証結果: ✅ 確認（有効）**

- `ptr!=nullptr && deleter==nullptr` を「成功」扱い
- **結果: 防御的ハードニング。現状悪用経路なし。**

### L-04: `scanPeak` `tmp` 未初期化
**検証結果: ✅ 確認（有効）**

- `n<4` のとき `tmp` にストアするが初期化なし
- **結果: 軽微。負ピーク絶対値化で影響なし。**

---

## Info（11件）

### I-01: `EQProcessor::reset()` の "(Audio Thread)" ラベルと libm 呼び出し
**検証結果: ⚠️ 一部確認（要追加調査）**

- `DSPCore::reset()` の呼出し箇所: `AudioEngine.Processing.DSPCoreLifecycle.cpp:335`
- コメントは「(Audio Thread)」だが、実際の呼出し元は非RTスレッド
- **結果: コメント不整合。デッドコードの可能性。**

### I-02: `DSPHandle` アトミックのロックフリー保証が未検証
**検証結果: ⚠️ 一部確認（要追加調査）**

- `static_assert(std::atomic<uint64_t>::is_always_lock_free)` は存在（line 94）
- `DSPHandle`（16byte）には `static_assert` なし
- **結果: 8byte は検証済み、16byte は未検証。CMPXCHG16B で実質問題ないが要コンパイル確認。**

### I-03: ネット0dB保証がクランプで崩れる
**検証結果: ✅ 確認（有効）**

- `makeupDb = jlimit(0, 12, -input-trim)` → ideal +24dB も 12dB クランプ
- **結果: 特定条件下でユーザーが原因不明の音量低下を経験。**

### I-04: 両方バイパス時のテスト乖離
**検証結果: ⚠️ 要追加調査**

- `refPlan()` が Both bypassed 分岐なし
- **結果: テスト参照実装と製品コードの乖離。**

### I-05: `estimateQSafetyMargin` ゼロ時挙動の乖離
**検証結果: ⚠️ 要追加調査**

- 製品: `eqMax<=0 → 0.0f`、テスト: `refQSafetyMargin(0) → 1.5f`
- **結果: テストが古い仕様のまま。**

### I-06: `RuntimeBuilder.cpp` dB→linear変換でNaNチェックなし
**検証結果: ✅ 確認（有効）**

- `decibelsToGain(-inf)` は0を返すが、`plan` がNaNなら伝播
- **結果: 稀にNaN伝播→無音。**

### I-07: `DSPCoreIO.cpp` AVX2マスクで負のゼロ消失
**検証結果: ✅ 確認（有効）**

- `_mm256_and_pd(vData, vMask)` で負の0も消失
- **結果: 極めて軽微。**

### I-08: `IRConverter.cpp` `computeEnergyScale` エネルギー計算アンダーフロー
**検証結果: ✅ 確認（有効）**

- 長尺IRで `energy` が `1e12` 超え `1/sqrt(energy)` アンダーフロー
- **結果: 長尺IRで無音化。**

### I-09: `IRAnalyzer.cpp` 3点ガウス補間式が非標準
**検証結果: ✅ 確認（有効）**

- `y0 * exp(-delta * (logY0 - logYm1))` は対称性を欠く
- **結果: ピーク過大評価。AutoGain過剰ヘッドルーム。**

### I-10: `TruePeakDetector.cpp` ピークホールド減衰 `0.999` 固定
**検証結果: ✅ 確認（有効）**

- サンプルレート非依存の固定値
- **結果: SR切替時にピーク検出特性が変わる。**

### I-11: `DeviceSettings.cpp` `resized` レイアウト無駄
**検証結果: ✅ 確認（有効）**

- `setVisible(false)` と0サイズの両方で非表示
- **結果: 軽微（無駄な処理）。**

---

## 検証結果の全体統計

### 重大度別

| 重大度 | 確認 | 修正済み | 無効 | 要追加 |
|--------|------|----------|------|--------|
| Critical | 3 | 1 | 1 | 1 (C-05), 1 revision (C-08) |
| High | 2 | 6 | 3 | 1 (H-03) |
| Medium | 10 | 1 | 1 | 3 |
| Low | 4 | 0 | 0 | 0 |
| Info | 7 | 0 | 0 | 4 |

### 優先修正推奨（Critical + High 有効のみ）

1. **C-01**: プリフェッチ範囲修正（`+64` → `+16` 以下に）
2. **C-02**: 量子化順序修正（ディザ→量子化→クランプ）
3. **C-07**: IRConverter ジャンプ保護の `*currentIr` → `*ir` 修正
4. **H-04**: `makeHPF` にナイキスト上限チェック追加
5. **H-11**: `updateGainStagingDisplay` に `hasKeyboardFocus()` チェック追加

### 次のアクション候補

1. C-01/C-02/C-07 のパッチ適用
2. H-04 のmakeHPF修正
3. M-02/M-03 の未接続安全網の体系調査
4. I-04/I-05 のテスト同期
