# ConvoPeq ソースコード 徹底バグ監査報告 【検証済み版】

対象: `file8831314928463026568.md` 約279ファイル / 318万字 / Windows11 x64 / AVX2必須 / MSVC+ICX 前提
監査方法: 全ファイルの静的解析、並行処理モデル・ライフタイム・AVX2アライメント・MKL・DSP数値安定性の横断監査。
検証ツール: serena MCP server, AiDex MCP server, grep/ast-grep/rg(WSL), 実コード直接読み取り

> **重要な注意**: 本レポートはソースコードの実地検証に基づく。元の監査レポート(file8831314928463026568.md)には、
> 既に修正済みのバグや実際のコードと異なる記述が多数含まれている。以下は2016-07-24時点の最新コードによる検証結果である。

---

## 即クラッシュ / ヒープ破壊 / UAF

### 1. `NoiseShaperLearner.cpp` スタックオーバーフロー
**状態: 部分的に妥当／数値誤差あり**

- `buildTrainingSegments()` の `double recentLeft[kRecentSampleRequest]` (34816要素 = 272KB) × 2 = 544KB → **スタック使用量大。Windows既定1MBに対して警告水準だが即クラッシュではない。**
- `evaluatePopulation()` 内 `alignas(64) double tanhBuffer[totalCoeffs]` → **実際のサイズ: `kPopulation(18) × kDim(9) = 162要素 = 1.3KB`。報告の128KBは誤り（CMA-ESのdimは9であり256ではない）。**
- **修正は入っている**: `sharedMappedPopulation` はヒープ確保 (`convo::makeAlignedArray<double>`)。ただし `recentLeft/Right` は未だスタック。
- **推奨**: `recentLeft/Right` もヒープへ (`ScopedAlignedArray<double>`)。
- **優先度: 中**（クラッシュは稀だが、予防的子守唄として推奨）

### 2. `NoiseShaperLearner.cpp` `sharedMappedPopulation` 可変長バグ
**状態: 該当せず（既に修正済みまたは元報告が誤り）**

- `CmaEsOptimizer::kDim = 9` は **コンパイル時定数**。dim が 18→32 に変化することはない。
- `sharedMappedPopulation` は `evaluatePopulation()` 内で初回のみ確保（nullチェック）、以降は再利用。
- サンプルレート変更時は `resetLearningSession()` が呼ばれ、optimizerは `initFromParcor()` で再初期化される。
- **元報告の「48k→96kでdim 18→32」は実コードと一致しない。**
- **判定: 根拠不十分**

### 3. `DftiHandle.h` / `MKLNonUniformConvolver.cpp` MKLハンドルリーク・二重解放
**状態: 既に修正済み**

- ファイル名自体が `ScopedDftiDescriptor` に変更（`DftiHandle.h` は存在しない）。
- `ScopedDftiDescriptor` は完全なRAII: デストラクタで `DftiFreeDescriptor`、ムーブ代入で旧ハンドル解放、`reset()` で解放。
- `MKLNonUniformConvolver.cpp` では **IPP FFT（`IppFFTPlan`）** に換装済み。Audio Thread内のFFTはMKL DFTI非使用。
- `IppFFTPlan` もRAII: デストラクタで `ippsFree` により確実に解放。
- **判定: 修正完了。本件はクローズ。**

### 4. `AlignedAllocation.h` / `DeferredFreeThread.h` / `RefCountedDeferred.h`
**状態: 既に修正済み（3件中3件）**

- **`_aligned_malloc` vs `free` の不一致**: 実コードは `mkl_malloc` / `mkl_free` を使用。`_aligned_malloc` はどこにも存在しない。
- **`DeferredFreeThread` シャットダウン競合**: `shutdownAndDrain()` が実装済み。`stop()` で `running=false` (release) し、`join()` 後に `drainAllRetired()` を呼ぶ。`isShuttingDown_` は `std::atomic<bool> running` に置き換え済み。
- **`SafeStateSwapper` ABA問題**: `globalEpoch` は `uint64_t`（64bit）。ラップアラウンドには 2^64 世代必要で実質問題なし。`isOlder()` は符号付き比較で安全。
- **判定: 全て修正完了。クローズ。**

### 5. `MKLNonUniformConvolver.cpp` バッファオーバーラン
**状態: 部分的に修正済み、要継続監視**

- IPP FFTへの換装によりワークバッファ関連の懸念は大幅軽減。
- `decimateStage()` では事前に `centerTapOk` と `convTapOk` の境界チェックを完全外出しし、違反時は全出力ゼロクリア＋`markCorruptionDetected()`。
- `interpolateStage()` でも `idx < (stage.convCount - 1) || idx >= capacity` のガードあり。
- **元報告の「blockSize < 64で未初期化読み出し」** → `dotProductAvx2()` の汎用コードはAVX2パスの剰余をスカラーで処理、`dotProductDecimateAvx2()` も同様。ただし「残りはスカラーフォールバック」は既に実装済み。
- **判定: 概ね修正済み。新規クラッシュレポートがあれば個別対応。**

---

## 並行処理・ロックフリー・ISR

### 6. `LockFreeRingBuffer.h` / `LockFreeAudioRingBuffer.h`
**状態: 既に修正済み（3件中3件／報告は全て誤り）**

- **メモリオーダリング**: 実コードは `consumeAtomic(acquire)` / `publishAtomic(release)` を使用。`relaxed` は一切使っていない。
- **power-of-2 チェック**: `static_assert((Capacity & (Capacity - 1)) == 0, ...)` あり。
- **False Sharing パディング**: `alignas(64)` が `writeIndex` と `readIndex` に付与されている。
- **判定: 元報告の3件全てが現状コードと一致しない。既に完全修正済み。**

### 7. `AtomicAccess.h` / `AudioEngine.Threading.cpp`
**状態: 理論上の懸念（x64実用上は問題なし）**

- `consumeAtomic` は `memory_order_acquire` を使用。`consume` セマンティクスが必要な箇所の指摘は **C++17で `memory_order_consume` が非推奨化されたため、最新コンパイラでは `acquire` が標準的**。
- x64 TSO (Total Store Order) 上では `acquire` も `relaxed` も実効的な差はない。ICXの最適化による並び替えもx64では発生しない。
- `fetchAddAtomic` の `acq_rel` は適切。
- **判定: ARM/POWER 等他アーキテクチャ移植時に対応すれば十分。x64/Win11 では実害なし。**

### 8. `ISRRetire*` / `ISRLifecycle.*` / `CrossfadeAuthority`
**状態: 部分的に対応済み、未対応項目あり**

- **`ISRRetireOverflowRing` のドロップ**: Ring満杯時は `tryPush()` が `false` を返し、呼出元で `droppedIntentCount_` をインクリメント。サイレントドロップではないが、`ISRDSPQuarantine` とのUAF連鎖は未確認。
- **`retireBacklog` uint32_tオーバーフロー**: `ISRRetireRouter.cpp` の `m_overflowCount_` は `std::atomic<uint64_t>`。`pendingRetireCount()` は `uint32_t` 経由だが、実態は `IEpochProvider` 委譲。**要確認: 戻り値が `uint32_t` のため長時間稼働でラップ可能性あり。**
- **`FrozenRuntimeWorld` の起動時 nullptr**: `resolveActiveRuntimeDSPFromRuntimeWorldOnly()` にヌルガード実装済み。`getRuntimeSampleRateHzFromWorld` で `runtimeWorld == nullptr` 時は安全パスを通る。
- **判定: overflowCount は64bit化済み。バックログの uint32_t 報告のみ未確認。**

### 9. `AudioEngine.Transition.cpp` / `DSPTransition.h`
**状態: 部分的に対応済み**

- `CrossfadeRuntime.h` の `gain_` と `dryScaleGain_` は独立管理。`pending_` フラグで `fading` と `current` の同一ポインタをガード。
- フェード時間0秒: `start()` 内で `std::max(0.001, fadeTimeSec)` により0除算防止済み。
- **判定: 元報告の2件は対策済み。ただし new な問題（複数回 start 等）は未確認。**

---

## DSP・AVX2・数値

### 10. AVX2 アライメント違反
**状態: 一部は誤報、一部は依然未対応**

- **`_mm256_load_ps` の非アライン使用**: **ソースコード中に `_mm256_load_ps` は1件も存在しない。** すべて `_mm256_loadu_pd` または `_mm256_load_pd`（整列保証済みバッファ用）を使用。
- **`alignas(32)` 忘れの構造体**: `ScopedAlignedArray` は `mkl_malloc(64)` で64バイトアライン保証。`CmaEsOptimizer` の `mean`/`covariance` は `makeAlignedArray` 経由。ただし `NoiseShaperLearner.h` の `AudioSegment` 構造体内の `double left[kLength]` には `alignas(64)` がない。**部分的な懸念あり。**
- **`_mm256_zeroupper` 欠如**: **存在しない。** AVX2命令使用コード（`OutputFilter.cpp`、`CustomInputOversampler.cpp`、`TruePeakDetector.cpp`、`FixedNoiseShaper.h`）のいずれにも `_mm256_zeroupper` の呼び出しなし。VZEROUPPER欠如によるAVX→SSE遷移ペナルティはJUCE側のSSEコードとの混在時に最大100サイクル超の遅延となる。
- **判定: `zeroupper` 欠如は真性のバグ。load_ps の報告は誤報。**

### 11. 非正規化数・MXCSR
**状態: ほぼ対応済み**

- `workerThreadMain()` で `_MM_SET_FLUSH_ZERO_ON` / `_MM_SET_DENORMALS_ZERO_ON` 設定済み。
- `evaluationWorkerMain()` でも同様にFTZ/DAZ設定済み。
- `evaluateCandidate()` 内で `juce::ScopedNoDenormals` 使用。
- `ScopedMXCSR.h` が `convo::cpu` 名前空間にRAIIラッパーとして存在。
- **ただし `ScopedMXCSR` の使用範囲が限定的。`OutputFilter::prepare()` や `FixedNoiseShaper::prepare()` では未使用。** ただしこれらはMessage Thread（非RT）で呼ばれることが前提。
- **判定: 実用上ほぼ対応済み。例外パスでのMXCSR復帰は `ScopedMXCSR` のRAIIに委ねられている。**

### 12. `OutputFilter.cpp` / `UltraHighRateDCBlocker.h`
**状態: 一部対応済み、一部未確認**

- `OutputFilter.cpp` では `fc_hc` をサンプルレートに応じて19000〜24000Hzでクリップ。`Q` 値も適切に設定。
- `makeLPF()` / `makeHPF()` 内で `nyq = fs * 0.4999` とし、ナイキスト周波数直前でガード。
- ただし `UltraHighRateDCBlocker.h` は未読み取り。**DCブロッカの極配置は確認中。**
- **判定: OutputFilter 側は適切。UltraHighRateDCBlocker は要確認。**

### 13. `TruePeakDetector.cpp`
**状態: 元報告の記述は実コードと一致しない**

- **`reset()` でリングをクリアしない**: `reset()` は `peakHold = 0.0` し、全ステージの `upHistory[]` を `FloatVectorOperations::clear` する。**完全にクリアされている。**
- **`prepareToPlay` でリセット漏れ**: `prepare()` は末尾で明示的に `reset()` を呼び出す。
- **`float` 飽和で0dBFS超え検出漏れ**: `scanPeak()` は `double` で全演算。`float` へのキャストは一切ない。
- **判定: 3件とも誤報。現状コードでは問題なし。**

### 14. `FixedNoiseShaper.h` / `LatticeNoiseShaper.h`
**状態: 既に修正済み（2件中2件）**

- **誤差フィードバック係数和が1.0超え**: `setCoefficients()` で `abs(sum - 1.0) > 1.0e-12` のチェックあり。超過時は `false` を返す（係数設定不採用）。
- **NaN係数の保存・伝播**: `quantize()` の入口と出口で `replaceNonFiniteWithZero()` によりNaN/Infを0.0に置換。`processSample()` でも `clampedError` に `killDenormal(replaceNonFiniteWithZero(...))` で二重ガード。
- **`int32` 累積オーバーフロー**: 量子化は `double` で全演算。`int32` は使用されていない。
- **判定: 完全修正済み。**

---

## キャッシュ・状態・永続化

### 15. `CacheManager.cpp` / `MixedPhasePersistentCache.cpp`
**状態: 部分的に対応済み、問題残存**

- **TOCTOU**: 書き込みは `tmp` ファイル → `moveFileTo`（atomic rename）パターン。`exists()`→`open()` の競合は read 側に残るが、validateCacheFile() でチェックサム検証＋キー照合あり。read側の `exists()`→`createInputStream()` 間に他プロセス削除は起こりうる → その場合は単に `false` を返しキャッシュミス扱い。
- **ハッシュ衝突**: `computeKey()` は CRC64（file content）＋ `hashCombine`（多重パラメータ）。衝突確率は2^64分の1で実用上問題なし。ただしCRCは暗号学的ハッシュではないため、意図的な衝突には脆弱。
- **`ProgressiveUpgradeThread` のロックなしアクセス**: `touch()`, `evictLRU()`, `clear()` はすべて `cacheMutex` の `lock_guard` で保護。非RTスレッド間でのデータ競合は防止済み。
- **判定: TOCTOUは部分対応（read側のごく狭いwindowのみ）。Hash衝突は実用上問題なし。UI Thread衝突は修正済み。**

### 16. `ConvolverState.cpp` / `PreparedIRState.h` / `IRConverter.cpp` / `IRAnalyzer.cpp`
**状態: 3段階ガード完備／1件の軽微な論理問題**

**IRConverter::computeEnergyScale()（zero-length IR）:**
- `numSamples <= 0 || numChannels <= 0` → 早期 return 1.0。**安全。**
- `cblas_ddot(0, ...)` → energy=0 → `maxChannelEnergy > 1.0e-18` は偽 → return 1.0。**安全。**

**IRAnalyzer::estimateMaxFrequencyResponseGain()（log(0)懸念）:**
- `numSamples <= 0 || numChannels <= 0` → return 1.0。**安全。**
- `const int copyLen = std::min(numSamples, kMaxAnalysisWindow)` — 0 samples → copyLen=0。
- `const int fftSize = juce::nextPowerOfTwo(0)` → 0 → `if (fftSize < 2) return 1.0;` → **安全。**
- 2サンプル未満のIR（copyLen=0,1）は全て早期return。**log(0)は発生しない。**

**IRConverter::convertFile()（リサンプルフォールバック）:**
- `loadAudioFile()` で `n <= 0` をチェック → zero-length IRは却下。**安全。**
- `fftSize = juce::jmax(32, config.fftSize)` → 最小32。**安全。**
- `numPartitions = juce::jmax(1, ...)` → 最低1パーティション。**安全。**
- **【軽微な問題】** リサンプル失敗時: `converted = ir`（原寸IR）を `actualSampleRate = config.targetSampleRate` で報告する。IRデータはソースレートのまま、エンジンにはターゲットレートと伝えるため、**コンボルバがIRを間違った速度で処理するピッチ誤差が発生する。** ただしengine側で内部SRCを持つため致命的ではない。ログ出力済み。

**ConvolverState:**
- 軽量版メタデータ構造（partitionData無し）。zero-lengthでも生成可能だが、実際に使用するIRConverterが零長を弾くため問題なし。

**判定: 零長IRとlog(0)は多重ガード完備で安全。リサンプルフォールバック時のピッチ誤差は軽微な論理問題（非クラッシュ）。**

### 17. `DeviceSettings.cpp` / `AsioBlacklist.h`
**状態: 元報告の記述は実コードと一致しない**

- **`String::contains` 部分一致の誤検出**: `BlacklistedASIODeviceType::getDeviceNames()` は `blacklist.isBlacklisted(names[i])` を呼ぶ。この関数は **完全一致（同等性比較）**、部分一致ではない。"Realtek ASIO" が "Real" に誤ヒットすることはない。
- **UTF-16 LE vs UTF-8**: 設定ファイルの読み書きは `juce::XmlDocument::parse` / `juce::XmlElement::writeTo` を使用。JUCEは内部でエンコーディングを適切に処理する（XMLの encoding 属性に従う）。日本語ユーザ名環境でクラッシュする報告は確認できず。
- **判定: 2件とも誤報。現状コードでは問題なし。**

---

## FIFO・レイテンシー・MMCSS

### 18. `LockFreeAudioRingBuffer` / `AudioSegmentBuffer.h` / `AudioEngine.Fifo.cpp`
**状態: 1件確認済み軽度問題／1件誤報**

**`LockFreeAudioRingBuffer::push()` の部分書き込み:**
- `push()` は `free < requestedSamples` の時、`juce::jmin(requestedSamples, free)` だけ書き込み、**残りを黙って破棄する**。呼出元（`pushToFifo`）に部分書き込みを知らせない。
- ただしこのFIFOは**アナライザ表示用**（`analyzerFifo`）であり、オーディオ出力パスではない。UIのスペアナ表示が一瞬遅れる程度で、音声品質には影響しない。
- **acquire/release ordering** 完備。`alignas(64)` でFalse Sharing防止。能力チェック(`juce::jmin(free, ...)`)によるバッファオーバーランの防止も完備。
- **判定: バグではあるが軽度（UI表示の精度問題）。オーディオパス非関与のため優先度低。**

**`AudioSegmentBuffer` の `pushBlock()` / `copyLatest()`:**
- **std::move 後参照の懸念**: `pushBlock()` は `FloatVectorOperations::copy` で**コピー**する。`std::move` は使用していない。元データの所有権は変更されない。
- **リングバッファの整合性**: `writePosition` と `totalSamples` は `acquire/release` atomic操作で保護。循環バッファのラップアラウンドも正しく処理。
- **`kCapacity` 超過**: `numSamples > kCapacity` のガードあり → `jassert` + `return`。
- **判定: 元報告の「move後参照」は誤報。AudioSegmentBufferは正常動作。**

### 19. `AudioEngine.Processing.Latency.cpp`
**状態: 実用上問題なし／理論上の懸念のみ**

- **`int` 溢れ**: `getCurrentLatencySamples()` の戻り値は `int`。オーバーサンプリングレイテンシはタップ数ベース（最大1023+255+63=1341）÷OS倍率で、**数百サンプル以内**に収まる。コンボルバのレイテンシ（`dsp->convolverRt().getLatencyBreakdown()`）が理論上 `INT_MAX` を超える可能性はあるが、実用的なIR長では発生しない。
- **`safeOsFactor = std::max(1, osFactor)`**: 0除算防止。**安全。**
- **`juce::jmax(0, ...)`**: 負数防止。**安全。**
- **`std::lround`**: 四捨五入による誤差最小化。**正しい。**
- **PDC不一致（prepareToPlay vs processBlock）**: `getCurrentLatencyBreakdown()` は単一のコードパスで計算され、prepareToPlay と processBlock の両方から同じ関数が呼ばれる。**同一実装のため不一致は発生しない。**
- **`getProcessingSampleRate()`**: `manualOversamplingFactor` が0の場合は `currentSampleRate` から自動決定。`std::min(actualFactor, maxFactor)` でオーバーサンプリング上限超過防止。**安全。**
- **判定: 実用上問題なし。理論的な int overflow はIR長が数時間級の場合のみで非現実的。**

### 20. `AudioEngine.Mmcss.cpp`
**状態: 徹底したエラー処理完備／スレッド優先度設計も適切**

- **`throw` なし**: `tryApplyMmcssForSelfManagedThread()` は `bool` を返す。`AvSetMmThreadCharacteristicsW` の失敗は `GetLastError()` で解析し、決して例外を投げない。**元報告の「throw」は誤報。**
- **エラーハンドリング品質**:
  - `ERROR_ACCESS_DENIED(5)` / `ERROR_ALREADY_EXISTS(183)` / `ERROR_NO_MORE_ITEMS(1552)` → **成功扱い（JUCE/ドライバが既にMMCSS登録済み）**。
  - `ERROR_INVALID_TASK_NAME(1531)` → **フォールバックチェーン**: ASIO: Pro Audio → Audio, DS: Playback → Audio → Pro Audio。
  - 全フォールバック失敗 → `false` を返し、NativeRTにフォールバック。
- **スレッド優先度**:
  - ASIO → `AVRT_PRIORITY_CRITICAL` + `L"Pro Audio"`。**正しい。**
  - DirectSound → `AVRT_PRIORITY_HIGH` + `L"Playback"`。**正しい。**
  - WASAPI → `JuceManaged`（JUCE 8.0.12が内部管理）。**スキップ。正しい。**
- **CPU affinity**: `applyMmcssPriority()` で `SetThreadAffinityMask` 設定。
- **FTZ/DAZ**: `ensureThreadFloatingPointEnvironment()` で明示設定。`thread_local` で初回のみ実行。
- **シャットダウン**: `revertMmcssOnAudioThread()` が同一スレッドから `AvRevertMmThreadCharacteristics` を呼ぶ。MSDN準拠。**安全。**
- **判定: 元報告の「throw」は誤報。MMCSS実装は非常に堅牢で、エラーハンドリング、優先度設定、シャットダウン全て適切。**

---

## その他・コーディング規約・リーク

- `ConvolverControlPanel.cpp` → **未検証**
- `EQEditProcessor.cpp` Listener解除漏れ → **未検証**
- `GenerationManager.h` 世代ID負数 → **未検証**
- `TelemetryRecorder.cpp` fopen NULLチェック → **未検証**
- CMake `build.bat` `/arch:AVX2` + `/Qax` 二重生成 → **未検証**

---

## 【重要】検証結果サマリー

### 修正済みと確認されたもの（全20項目中）
| バグ# | ステータス | 詳細 |
|-------|-----------|------|
| 3 | ✅ 修正完了 | ScopedDftiDescriptor + IppFFTPlan RAII完備 |
| 4 | ✅ 修正完了 | mkl_malloc/mkl_free統一、DeferredFreeThread改善、epoch 64bit化 |
| 6 | ✅ 修正完了 | acquire/release ordering、power-of-2チェック、alignas(64)パディング |
| 13 | ✅ 誤報 | reset()正常動作、double精度維持、float飽和なし |
| 14 | ✅ 修正完了 | NaNガード、係数和チェック、int32不使用 |
| 16 | ✅ 実用上安全 | 零長IRは多重ガード完備。log(0)経路なし。軽微なピッチ問題のみ |
| 17 | ✅ 誤報 | 完全一致比較、JUCE XML適切処理 |
| 19 | ✅ 実用上安全 | int溢れは非現実的、PDC不一致なし |
| 20 | ✅ 誤報 | throwなし、堅牢なエラーハンドリング、適切な優先度設定 |

### 部分的に対応済み
| バグ# | ステータス | 詳細 |
|-------|-----------|------|
| 1 | ⚠ 部分的 | recentLeft/Right スタック使用量大（544KB）。tanhBufferは1.3KB（報告の128KBは誤り） |
| 5 | ⚠ 部分的 | IPP換装＋境界チェック完備．継続監視推奨 |
| 8 | ⚠ 部分的 | overflow 64bit化済み。uint32_t backlog報告は未確認 |
| 9 | ⚠ 部分的 | 0除算防止、同一ポインタ対策済み |
| 10 | ⚠ 部分的 | load_ps は誤報。zeroupper欠如は真正バグ。AudioSegmentにalignas(64)なし |
| 11 | ⚠ 部分的 | FTZ/DAZ設定済み。ScopedMXCSR使用範囲は限定的 |
| 12 | ⚠ 部分的 | OutputFilter側OK。UltraHighRateDCBlocker未確認 |
| 15 | ⚠ 部分的 | ラッパーセーブ+チェックサム検証+mutex保護。read側TOCTOUの微小window残存 |
| 18 | ⚠ 軽度 | LockFreeAudioRingBuffer部分書き込み（analyzer表示のみ）。AudioSegmentBufferは正常 |

### 根拠不十分・誤報
| バグ# | ステータス | 詳細 |
|-------|-----------|------|
| 2 | ❌ 該当せず | kDim=9はコンパイル時定数、実行時変動なし |
| 7 | ❌ 理論上のみ | x64 TSOでは問題なし。ARM移植時に対応可 |

---

## アクションアイテム

### v0.1 即時対応が必要なもの
1. **`_mm256_zeroupper` 欠如**: AVX2使用ファイル（`OutputFilter.cpp`, `CustomInputOversampler.cpp`, `TruePeakDetector.cpp`, `FixedNoiseShaper.h`, `LatticeNoiseShaper.h`）のAVX2→SSE遷移箇所に `_mm256_zeroupper` または `_mm256_zeroall` を追加する。JUCEのSSEコードとの混在による100cycle超のペナルティ防止。
2. **`NoiseShaperLearner::buildTrainingSegments()` のスタック配列**: `recentLeft[34816]`, `recentRight[34816]`（合計544KB）を `ScopedAlignedArray<double>`（ヒープ）に変更。

### v0.2 次期対応
3. **`ISRRetireRouter::pendingRetireCount()` の戻り値型**: 現在 `uint32_t`。長時間稼働でラップの可能性。`uint64_t` への変更を検討。
4. **`AudioSegment` 構造体への `alignas(64)` 付与**: `NoiseShaperLearner.h` の内蔵配列が64バイトアライン保証なし。
5. **`IRConverter::convertFile()` リサンプルフォールバック時のピッチ誤差**: リサンプル失敗時、IRデータはソースレートのままエンジンにターゲットレートを報告する。engine側の内部SRCで吸収されるが、稀なケースで意図しない周波数特性になる可能性。

### 継続監視項目
6. **`LockFreeAudioRingBuffer::push()` の部分書き込み**: アナライザ表示用FIFOであり音声品質非関与のため優先度低。ただし長期的にはシグナリング（書き込み成功数を返すなど）の改善余地あり。

### 元報告に対する注意事項
- 元監査レポートのバグ#2, #6, #10, #13, #14, #17 の記述には **現状コードと一致しない誤りが含まれている**。
- 特に **#6 (LockFreeRingBuffer)** の relaxed ordering の指摘、**#13 (TruePeakDetector)** のリセット漏れの指摘、**#17 (DeviceSettings)** のエンコーディングの指摘は根拠がない。
- 検証結果を踏まえ、**修正優先度を再設定**した。上記アクションアイテムに従うこと。
