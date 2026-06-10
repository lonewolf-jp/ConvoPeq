# ConvoPeq バグ分析・修正報告書

## 優先順位別バグ一覧

| ID | 優先度 | カテゴリ | 概要 | 影響 |
|----|--------|----------|------|------|
| **C-01** | Critical | RT安全違反 | `LinearRamp::setCurrentAndTargetValue` をAudio Threadから呼び出し | デバッグビルドで即座にクラッシュ |
| **C-02** | Critical | 未定義動作 | `union` による型パンニング（`absNoLibm` 他） | 最適化ビルドで予期せぬ動作・クラッシュ |
| **C-03** | Critical | Use-After-Free | `ProgressiveUpgradeThread` のラムダが生ポインタをキャプチャ | オブジェクト破棄後アクセスによるクラッシュ |
| **H-01** | High | 機能不全 | `EQProcessor::m_rtBypassShadow` が更新されずバイパス無効 | UI操作が反映されない |
| **H-02** | High | メモリリーク | `LoaderThread` の `callAsync` ラムダがメモリリーク | 長時間使用でメモリ消費増大 |
| **H-03** | High | 音質劣化 | IRパーティション逆順後に `irFreqReal/irFreqImag` 不整合 | AVX2パスで逆IR適用（時間反転） |
| **H-04** | High | 音質劣化 | 2箇所の線形クロスフェード（等パワーではない） | クロスフェード中点で -3dB ディップ |
| **H-05** | High | 例外安全 | `finalizeNUCEngineOnMessageThread` の例外処理不足 | 例外発生時にリソースリーク |
| **M-01** | Medium | RT安全違反 | `std::abs` をAudio Threadで使用（`Get` 内） | libm呼び出しによるRT遅延リスク |
| **M-02** | Medium | 音質劣化 | DCブロッカーカットオフ不統一（3 Hz vs 1 Hz） | 位相特性の微小不整合 |
| **M-03** | Medium | パフォーマンス | `Fixed15TapNoiseShaper` の毎サンプル全状態スキャン | 不要なCPU負荷 |
| **M-04** | Medium | 状態管理 | `observeMonotonicRollbackRequested_` フラグがリセットされない | 診断誤動作 |
| **M-05** | Medium | コード品質 | `wetBuf[0]` の危険な流用（意図不明瞭） | 将来のバグ混入リスク |
| **L-01** | Low | 表示精度 | レベルメーター測定タイミング（ヘッドルーム未適用） | 約 +1dB の表示誤差 |
| **L-02** | Low | コード品質 | `applySpectrumFilter` の HC `else` ブランチ到達不能 | 可読性低下（機能的影響なし） |
| **L-03** | Low | 命名/コメント | `Fixed15TapNoiseShaper::ORDER = 16` / `calcSVFCoeffs` コメント誤記 | 保守性低下 |

---

## 各バグ詳細レポート

### 🔴 C-01: `LinearRamp::setCurrentAndTargetValue` のスレッドコンテキスト違反

| 項目 | 内容 |
|------|------|
| **バグ概要** | `LinearRamp::setCurrentAndTargetValue` は `ASSERT_NON_RT_THREAD()` マクロを含むが、Audio Thread から呼び出されている。デバッグビルドで即座にアサート失敗→クラッシュ。 |
| **発生ファイル** | `src/DspNumericPolicy.h`（`LinearRamp` 定義）<br>`src/convolver/ConvolverProcessor.Runtime.cpp`（`process` 関数） |
| **関数名** | `LinearRamp::setCurrentAndTargetValue`<br>`ConvolverProcessor::process` |
| **バグ詳細** | `ConvolverProcessor::process` 内のレイテンシ更新処理で、`activeLatencySmoother.setCurrentAndTargetValue(val)` を呼び出している。これは Audio Thread 上で実行されるが、`LinearRamp::setCurrentAndTargetValue` は `ASSERT_NON_RT_THREAD()` で保護されており、Audio Thread からの呼び出しは禁止されている。 |
| **改修方法** | **選択肢A**: `LinearRamp` に Audio Thread 専用メソッド `forceSetCurrentAndTargetValueRT(double v)` を追加し、内部で `ASSERT_AUDIO_THREAD()` を使用する。<br>**選択肢B**: `ASSERT_NON_RT_THREAD()` を削除し、世代カウンターによる同期機構が呼び出しを安全にしている旨のコメントを追加する（推奨: 選択肢A）。 |

---

### 🔴 C-02: `union` による型パンニング（未定義動作）

| 項目 | 内容 |
|------|------|
| **バグ概要** | `absNoLibm`、`isFiniteNoLibm`、`killDenormal` などで `union` を使用した型パンニングを行っている。C++ において未定義動作であり、最適化ビルドで予期せぬ動作を引き起こす可能性がある。 |
| **発生ファイル** | `src/AudioEngine.h`（`absNoLibm`）<br>`src/DspNumericPolicy.h`（`killDenormal`）<br>`src/EQProcessor.Processing.cpp`（匿名名前空間内の関数）<br>他多数 |
| **関数名** | `absNoLibm`, `killDenormal`, `isFiniteNoLibm`, `isFiniteAndAbsInRangeMask` など |
| **バグ詳細** | C++17までは `union` の非アクティブメンバへの読み書きは厳密に禁止。C++20で一部緩和されたが、型パンニング目的での使用は依然として未定義動作とされる。コンパイラの最適化（特に `-O2` 以上）で、この種のコードは意図しない結果を生む可能性がある。 |
| **改修方法** | `std::bit_cast` (C++20) に置き換える。<br>```cpp<br>// Before<br>union { double d; uint64_t u; } v { x };<br>v.u &= 0x7FFFFFFFFFFFFFFFULL;<br>return v.d;<br><br>// After<br>return std::bit_cast<double>(std::bit_cast<uint64_t>(x) & 0x7FFFFFFFFFFFFFFFULL);<br>``` |

---

### 🔴 C-03: `ProgressiveUpgradeThread` の Use-After-Free

| 項目 | 内容 |
|------|------|
| **バグ概要** | `upgradeStep` 内のラムダが `cancelledFlag`（メンバ変数への生ポインタ）をキャプチャしている。スレッド実行中にオブジェクトが破棄されると、ラムダ実行時に無効ポインタを dereference する。 |
| **発生ファイル** | `src/ProgressiveUpgradeThread.cpp` |
| **関数名** | `ProgressiveUpgradeThread::upgradeStep` |
| **バグ詳細** | `cancelledFlag = &cancelled;` でメンバ変数のアドレスを取得し、`converter.convertToHighRes` に渡すラムダ内で `*cancelledFlag` を参照している。`ProgressiveUpgradeThread` オブジェクトが破棄された後もラムダが実行される可能性がある。 |
| **改修方法** | `cancelledFlag` も `weakOwner` 経由でアクセスするか、`std::shared_ptr`/`std::weak_ptr` を使用する。<br>```cpp<br>// 改修例: weakOwner 経由で cancelled にアクセス<br>auto weakThis = std::weak_ptr<ProgressiveUpgradeThread>(shared_from_this());<br>prepared = converter.convertToHighRes(..., [weakThis, expectedGeneration]() {<br>    auto self = weakThis.lock();<br>    if (!self) return true;<br>    return juce::Thread::currentThreadShouldExit()<br>        || self->cancelled.load()<br>        || !self->processor.isConvolverGenerationCurrent(expectedGeneration);<br>});<br>``` |

---

### 🟠 H-01: `EQProcessor::m_rtBypassShadow` の未更新によるバイパス機能の無効化

| 項目 | 内容 |
|------|------|
| **バグ概要** | `EQProcessor::process` はバイパス状態を `m_rtBypassShadow` から読み取るが、この変数を更新する `setBypassFromRT()` が一切呼び出されていない。結果、バイパス機能が永久に無効。 |
| **発生ファイル** | `src/eqprocessor/EQProcessor.h`<br>`src/eqprocessor/EQProcessor.Processing.cpp` |
| **関数名** | `EQProcessor::process`, `EQProcessor::setBypassFromRT` |
| **バグ詳細** | `m_rtBypassShadow` は Audio Thread 専用のシャドウ変数として設計されたが、Message Thread 側からこのシャドウを更新する機構が実装されていない。`setBypassFromRT()` 関数は存在するが、どこからも呼ばれていない。 |
| **改修方法** | **選択肢A**: `m_rtBypassShadow` を廃止し、`convo::consumeAtomic(bypassRequested, std::memory_order_acquire)` を直接参照する。<br>**選択肢B**: `ConvolverProcessor` と同様に `publishRuntimeProcessSnapshot()` 機構を導入し、`runtimeSnapshot.bypassed` を参照する。 |

---

### 🟠 H-02: `LoaderThread` の `callAsync` ラムダによるメモリリーク

| 項目 | 内容 |
|------|------|
| **バグ概要** | `MessageManager::callAsync` に渡すラムダが生ポインタをキャプチャしている。ラムダ実行前に `ConvolverProcessor` が破棄されるとポインタが解放されずメモリリーク。 |
| **発生ファイル** | `src/convolver/ConvolverProcessor.LoaderThread.cpp` |
| **関数名** | `LoaderThread::queueFinalizeOnMessageThread` |
| **バグ詳細** | `loadedIRRaw` と `displayIRRaw` を生ポインタで `new` し、ラムダに値キャプチャしている。`callAsync` 成功後に `ConvolverProcessor` が破棄されると、ラムダ内の `weakOwner.get()` が `nullptr` を返し、`loadedIRRaw` と `displayIRRaw` が解放されない。 |
| **改修方法** | `std::unique_ptr` をラムダにムーブキャプチャする。<br>```cpp<br>auto loadedIRHolder = std::make_unique<juce::AudioBuffer<double>>(std::move(result.loadedIR));<br>auto displayIRHolder = std::make_unique<juce::AudioBuffer<double>>(std::move(result.displayIR));<br><br>const bool queued = juce::MessageManager::callAsync([weakOwner, irLRaw, irRRaw,<br>    loadedIRHolder = std::move(loadedIRHolder),<br>    displayIRHolder = std::move(displayIRHolder), ...]() mutable {<br>    // unique_ptr が自動的に解放を管理<br>});<br>``` |

---

### 🟠 H-03: IRパーティション逆順後の `irFreqReal/irFreqImag` 不整合

| 項目 | 内容 |
|------|------|
| **バグ概要** | `SetImpulse` で前方アクセス最適化のため `irFreqDomain` を逆順化するが、Split-complex パスで使用する `irFreqReal/irFreqImag` は逆順化されない。AVX2ビルドで時間反転したIRが使用される危険性。 |
| **発生ファイル** | `src/MKLNonUniformConvolver.cpp` |
| **関数名** | `MKLNonUniformConvolver::SetImpulse` |
| **バグ詳細** | 1. `irFreqDomain` を構築・逆順化<br>2. `irFreqReal/irFreqImag` は逆順化前の状態で初期化されたまま<br>3. `applySpectrumFilter` が `filterSpec != nullptr` の場合のみ再構築して整合性を回復<br>4. `filterSpec == nullptr`（現状は発生しないがAPI上は可能）の場合、AVX2 split-complex パスが逆順IRを使用する |
| **改修方法** | **選択肢A**: 逆順化ループ内で `irFreqReal/irFreqImag` も同時に逆順化する。<br>**選択肢B**: `applySpectrumFilter` を `filterSpec` の有無に依らず常に実行し、整合性を確保する。 |

---

### 🟠 H-04: 線形クロスフェード（等パワーではない）

| 項目 | 内容 |
|------|------|
| **バグ概要** | 処理順序切替時とバイパスフェード時に線形クロスフェード（`gNew` と `1-gNew`）を使用している。等パワークロスフェードではないため、中点で約 -3dB のディップが発生する。 |
| **発生ファイル** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`<br>`src/audioengine/AudioEngine.Processing.BlockDouble.cpp` |
| **関数名** | `AudioEngine::getNextAudioBlock` / `processBlockDouble` 内のクロスフェード処理 |
| **バグ詳細** | 2箇所で線形クロスフェードが使用されている。<br>1. DSP処理順序切替時（`EQThenConvolver` ↔ `ConvolverThenEQ`）<br>2. 全体バイパスフェード時（`oversamplingFactor == 1` と `> 1` の両パス） |
| **改修方法** | `equalPowerSin(t)` 関数を使用する。<br>```cpp<br>// Before<br>const double gNew = gain.getNextValue();<br>const double gOld = 1.0 - gNew;<br><br>// After<br>const double gNew = equalPowerSin(gain.getNextValue());<br>const double gOld = equalPowerSin(1.0 - gain.getNextValue());<br>```<br>注: `equalPowerSin` は `src/core/SnapshotCoordinator.h` に定義済み。 |

---

### 🟠 H-05: `finalizeNUCEngineOnMessageThread` の例外処理不足

| 項目 | 内容 |
|------|------|
| **バグ概要** | `std::bad_alloc` のみキャッチし、その他の例外は捕捉されず `std::terminate` に至る。 |
| **発生ファイル** | `src/convolver/ConvolverProcessor.LoadPipeline.cpp` |
| **関数名** | `ConvolverProcessor::finalizeNUCEngineOnMessageThread` |
| **バグ詳細** | `try-catch` ブロックが `std::bad_alloc` のみを捕捉している。`std::runtime_error` など他の例外が発生した場合、処理されずに伝播しアプリケーションが終了する可能性がある。 |
| **改修方法** | `catch (const std::exception&)` と `catch (...)` を追加する。<br>```cpp<br>try {<br>    // ...<br>} catch (const std::bad_alloc&) {<br>    handleLoadError("Out of memory");<br>} catch (const std::exception& e) {<br>    handleLoadError(juce::String("Exception: ") + e.what());<br>} catch (...) {<br>    handleLoadError("Unknown error");<br>}<br>``` |

---

### 🟡 M-01: `std::abs` をAudio Threadで使用

| 項目 | 内容 |
|------|------|
| **バグ概要** | `MKLNonUniformConvolver::Get` 内の `addScaledFallback` が `std::abs` を呼び出している。Audio Thread で libm 関数を呼ぶとリアルタイム性を損なう可能性がある。 |
| **発生ファイル** | `src/MKLNonUniformConvolver.cpp` |
| **関数名** | `MKLNonUniformConvolver::Get` 内の `addScaledFallback` ラムダ |
| **バグ詳細** | `if (std::abs(gain - 1.0) < 1.0e-12)` というコードが Audio Thread 上で実行される。`std::abs` は実装によっては libm 関数 `fabs` を呼び出す。 |
| **改修方法** | `absNoLibm` 関数（`src/AudioEngine.h` に定義）を使用する。<br>```cpp<br>// Before<br>if (std::abs(gain - 1.0) < 1.0e-12)<br><br>// After<br>if (absNoLibm(gain - 1.0) < 1.0e-12)<br>``` |

---

### 🟡 M-02: DCブロッカーカットオフ不統一

| 項目 | 内容 |
|------|------|
| **バグ概要** | 入力/出力段のDCブロッカーは 3 Hz、オーバーサンプリング段は 1 Hz と、カットオフ周波数が統一されていない。 |
| **発生ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **関数名** | `DSPCore::prepare`（`dcBlockers().init` 呼び出し） |
| **バグ詳細** | ```cpp<br>outputL.init(sampleRate, 3.0);   // 3 Hz<br>outputR.init(sampleRate, 3.0);<br>inputL.init(sampleRate, 3.0);<br>inputR.init(sampleRate, 3.0);<br>oversampledL.init(processingRate, 1.0);  // 1 Hz ← 異なる<br>oversampledR.init(processingRate, 1.0);<br>``` |
| **改修方法** | 全てのDCブロッカーで同じカットオフ（例: 3 Hz）を使用するよう統一する。オーバーサンプリング段も `processingRate` に対して 3 Hz となるよう係数を再計算する。 |

---

### 🟡 M-03: `Fixed15TapNoiseShaper` の毎サンプル全状態スキャン

| 項目 | 内容 |
|------|------|
| **バグ概要** | `processSample` 内で、16要素の状態配列を毎サンプルスキャンして最大値を計算している。不要なCPU負荷。 |
| **発生ファイル** | `src/Fixed15TapNoiseShaper.h` |
| **関数名** | `Fixed15TapNoiseShaper::processSample` |
| **バグ詳細** | 512サンプルブロック処理時、L/Rチャンネル合わせて 512×2×16 = 16,384回の `absNoLibm` + 比較が発生する。 |
| **改修方法** | インクリメンタルな最大値追跡に変更する。<br>```cpp<br>// メンバ変数として currentMaxError を追加<br>currentMaxError = std::max(currentMaxError, absNoLibm(error));<br><br>// ブロック終了時（processStereoBlock の終わり）に一度だけチェック<br>if (currentMaxError > kErrorStateThreshold)<br>    publishAtomic(needsReset, true, std::memory_order_release);<br>currentMaxError = 0.0;<br>``` |

---

### 🟡 M-04: `observeMonotonicRollbackRequested_` フラグのリセット欠落

| 項目 | 内容 |
|------|------|
| **バグ概要** | 世代/シーケンスの逆行検出時に `true` に設定されるフラグが、一度も `false` にリセットされない。 |
| **発生ファイル** | `src/audioengine/AudioEngine.h`（`makeRuntimeReadHandle` 内） |
| **関数名** | `AudioEngine::makeRuntimeReadHandle` |
| **バグ詳細** | 逆行検出時に `publishAtomic(observeMonotonicRollbackRequested_, true, ...)` を実行するが、このフラグをリセットする処理が存在しない。 |
| **改修方法** | `RuntimePublicationOrchestrator` のロールバック処理完了時、または `AudioEngine` の初期化/リセット時にフラグを `false` に戻す。<br>```cpp<br>// 例: AudioEngine::initialize() 内<br>convo::publishAtomic(observeMonotonicRollbackRequested_, false, std::memory_order_release);<br>``` |

---

### 🟡 M-05: `wetBuf[0]` の危険な流用

| 項目 | 内容 |
|------|------|
| **バグ概要** | レイテンシクロスフェード処理で、ゲイン配列として `wetBuf[0]` を流用している。コードの意図が不明瞭で将来のバグ混入リスクがある。 |
| **発生ファイル** | `src/convolver/ConvolverProcessor.Runtime.cpp` |
| **関数名** | `ConvolverProcessor::process` |
| **バグ詳細** | `double* delayFadeRamp = wetBuf[0];` と、Wet信号用バッファをゲインカーブ格納用に流用している。 |
| **改修方法** | `scratchBuffer` の一部、またはローカルスタック配列を `delayFadeRamp` 専用に確保する。<br>```cpp<br>// ローカルスタック配列（MAX_BLOCK_SIZE は定数）<br>double delayFadeRamp[MAX_BLOCK_SIZE];<br>``` |

---

### 🔵 L-01: レベルメーター測定タイミング

| 項目 | 内容 |
|------|------|
| **バグ概要** | 出力レベルメーターが `kOutputHeadroom`（≈ -1dB）適用前に測定しているため、表示値が実際より約 +1dB 高くなる。 |
| **発生ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **関数名** | `DSPCore::processDoubleToBuffer` |
| **バグ詳細** | `measureLevel` を `processOutputDouble` の前に呼び出している。`processOutputDouble` 内で `kOutputHeadroom` が乗算されるため、測定値はヘッドルーム適用前の値となる。 |
| **改修方法** | `measureLevel` の呼び出しを `processOutputDouble` の後に移動するか、測定値を `kOutputHeadroom` でスケーリングしてから公開する。 |

---

### 🔵 L-02: `applySpectrumFilter` の HC `else` ブランチ到達不能

| 項目 | 内容 |
|------|------|
| **バグ概要** | ハイカットフィルターの周波数範囲計算において、`else` ブランチが絶対に実行されない。 |
| **発生ファイル** | `src/MKLNonUniformConvolver.cpp` |
| **関数名** | `MKLNonUniformConvolver::applySpectrumFilter` |
| **バグ詳細** | `hcFcEnd = nyquist` を設定しているため、`kEnd = halfN` となり、ループ範囲 `k < cSize`（`cSize = halfN + 1`）内では `k > kEnd` が真になることはない。 |
| **改修方法** | 到達不能コードを削除するか、`hcFcEnd` を設定可能なパラメータに変更する。<br>```cpp<br>// 修正例: 到達不能ブランチを削除<br>if (k <= kStart) { ... }<br>else {<br>    // taper (kStart < k <= kEnd)<br>}<br>// k > kEnd は発生しないため else ブランチは不要<br>``` |

---

### 🔵 L-03: 命名/コメントの誤り

| 項目 | 内容 |
|------|------|
| **バグ概要** | `Fixed15TapNoiseShaper::ORDER = 16`（15-tap と命名しながら16次）、`calcSVFCoeffs` の「Audio Thread用」コメントは誤り。 |
| **発生ファイル** | `src/Fixed15TapNoiseShaper.h`<br>`src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **バグ詳細** | 1. `Fixed15TapNoiseShaper::ORDER = 16` であり、実質的に16-tapフィルタ。名前との矛盾。<br>2. `calcSVFCoeffs` のコメントに「Audio Thread用」とあるが、`std::pow`・`std::tan` を含むため実際には Message Thread 専用。 |
| **改修方法** | 1. クラス名を `Fixed16TapNoiseShaper` に変更するか、`ORDER = 15` に修正する。<br>2. コメントを `// SVF係数計算 (Message Thread用)` に修正する。 |

---

## 修正優先順位と推奨手順

### 第1フェーズ（即時対応：リリース前必須）

1. **C-01** (`LinearRamp` のスレッドコンテキスト違反) - クラッシュ原因
2. **C-02** (`union` 型パンニング) - 未定義動作
3. **C-03** (Use-After-Free) - クラッシュ原因
4. **H-01** (バイパス機能無効) - 機能不全

### 第2フェーズ（早期対応：次リリースまで）

5. **H-02** (メモリリーク) - 長時間運用で悪化
6. **H-03** (IR逆順バグ) - 音質に影響
7. **H-04** (線形クロスフェード) - 音質に影響
8. **H-05** (例外処理不足) - 堅牢性

### 第3フェーズ（計画的対応）

9. **M-01** から **M-05**（RT安全、パフォーマンス、コード品質）
10. **L-01** から **L-03**（表示精度、保守性）

---

## 修正時注意事项

1. **変更影響範囲の確認**: 特に `LinearRamp` や `absNoLibm` など広範囲で使用される関数の変更は、全ての呼び出し箇所を検証すること。

2. **デバッグ/リリースビルドの差異**: `ASSERT_*` マクロはデバッグビルドでのみ有効。リリースビルドでの動作確認も必須。

3. **ABI互換性**: 構造体のサイズ変更（例: `ORDER` の変更）はバイナリ互換性を破壊するため、メジャーバージョンアップで実施すること。

4. **テスト強化**: 特にクロスフェード（H-04）とバイパス（H-01）の修正後は、Audio Thread と Message Thread の相互作用を検証する専用テストを追加推奨。