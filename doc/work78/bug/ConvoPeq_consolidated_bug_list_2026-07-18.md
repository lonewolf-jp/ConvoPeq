# ConvoPeq 統合バグリスト

> **作成日**: 2026-07-20
> **更新日**: 2026-07-20（検証結果反映）
> **出典**: `doc/work78/bug/` 配下7ファイル（Part1〜6 + new_bug.md）を統合・重複排除
> **合計**: 48件のユニークバグ（Critical 8 / High 11 / Medium 14 / Low 4 / Info 11）
> **パッチ有無**: 各バグに「Patch」欄で出典のパッチ有無を記載。Patch有は各出典ファイルに具体的なコード差分あり。
> **検証結果**: 48件中 **26件確認**, **10件修正済み**, **7件無効**, **5件要追加調査**。詳細は `ConvoPeq_consolidated_bug_verification_2026-07-20.md` を参照。

---

## サマリ

| 重大度 | 件数 | 内訳 |
|--------|------|------|
| **Critical** | 8 | クラッシュ・無音化・破壊音・Use-After-Free に直結 |
| **High** | 11 | 音質劣化・安全性低下・テスト失効 |
| **Medium** | 14 | エッジケース・非効率・コーディング規約逸脱 |
| **Low** | 4 | 診断限定・防御的改善・軽微 |
| **Info** | 11 | 要確認・要検証・コメント不整合・設計メモ |

---

## Critical

### C-01: `CustomInputOversampler.cpp` — プリフェッチがガードページを越える

| 項目 | 内容 |
|------|------|
| **File** | `src/CustomInputOversampler.cpp` (`dotProductAvx2`, `prepareStage`) |
| **Issue** | `_mm_prefetch(x + i + 64)` が確保サイズ `upHistorySize = keep + maxInput + 16` を超えてプリフェッチ。x86では例外発生しないが、ガードページに触れると稀に #PF 遅延 → XRUN。Up/Down履歴マージン非対称（Down側+6, Up側+0）も問題。 |
| **Impact** | WASAPI排他64サンプルでXRUN再現。VTuneで `L1D_PEND_MISS` スパイク確認。 |
| **Patch** | 出典: new_bug.md Bug 1。Up/Down共に `+8` マージン統一、`_mm_prefetch` 削除またはガード追加。 |
| **Source** | new_bug.md (first section, Critical #1) |

### C-02: `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` — 量子化オーバーフロー

| 項目 | 内容 |
|------|------|
| **File** | `src/Fixed15TapNoiseShaper.h`, `src/LatticeNoiseShaper.h` (`quantize()`) |
| **Issue** | クランプ→ディザの順序誤り。`v=maxV (32767/32768)` に `+scale` のディザが加わると `1.0` になり、16bitで表現不能な `32768` に量子化。int16変換時にラップして大ノイズ。Lipshitzの正規順序は「ディザ→量子化→クランプ」。 |
| **Impact** | 16bit出力時に破壊音。 |
| **Patch** | 出典: new_bug.md Bug 2。量子化後に `q = clamp(q, minV*invScale, maxV*invScale)` 追加。 |
| **Source** | new_bug.md (first section, Critical #2) |

### C-03: `Fixed15TapNoiseShaper.h` — プリセット線形補間で不安定化

| 項目 | 内容 |
|------|------|
| **File** | `src/Fixed15TapNoiseShaper.h` (`prepare()`) |
| **Issue** | 誤差フィードバック係数を直接線形補間。48k→88.2k間など中間レートで補間された係数の極が単位円外に出る。エラー状態が `kErrorStateThreshold=1e6` まで発散後 `needsReset` で無音化。48k→50kのようなレートで再現。 |
| **Impact** | 特定サンプルレートで無音ブロックが断続的に発生。 |
| **Patch** | 出典: new_bug.md Bug 3。補間後 `isStable()` チェックとフォールバック、または格子係数領域で補間。 |
| **Source** | new_bug.md (first section, Critical #3) |

### C-04: `MKLNonUniformConvolver.cpp` — Directパス memset 境界超過

| 項目 | 内容 |
|------|------|
| **File** | `src/MKLNonUniformConvolver.cpp` (`SetImpulse()`) |
| **Issue** | `memset(impulseForFft.get(), 0, directTapCount)` で `directTapCount` が `irLen` より大きい場合、確保サイズ `irLen` を超えて書き込み。`directTapCount` は `blockSize` 依存、極端に短いIR (例 32サンプル) で発生。 |
| **Impact** | ヒープ破壊 → 不定クラッシュ。ASANでheap-buffer-overflow検出。 |
| **Patch** | 出典: new_bug.md Bug 4。`memset` サイズを `min(directTapCount, irLen)` に。 |
| **Source** | new_bug.md (first section, Critical #4) |

### C-05: `SafeStateSwapper.h` — 2-step bump 同時実行によるUse-After-Free

| 項目 | 内容 |
|------|------|
| **File** | `src/SafeStateSwapper.h` (`swap()`) |
| **Issue** | 単一Writer前提の設計だが、`ConvolverProcessor` と `EQProcessor` の両方が同じ `EpochDomain` を共有し異なるスレッドから `swap()` が呼ばれるパスが存在。2つの `swap()` がインターリーブすると `epoch` が逆転し、`getMinReaderEpoch() < epoch` 条件でまだAudio Threadが参照中のStateを解放 → Use-After-Free。 |
| **Impact** | 不定クラッシュ、メモリ破壊。 |
| **Patch** | 出典: new_bug.md Bug 5。`swap()` に `std::mutex` 追加、またはWriterを1スレッドに限定。 |
| **Source** | new_bug.md (first section, Critical #5) |

### C-06: `IRAnalyzer.cpp` — FFT binループ範囲外読み出し

| 項目 | 内容 |
|------|------|
| **File** | `src/IRAnalyzer.cpp` (FFT binループ) |
| **Issue** | `for (int bin=0; bin < fftSize; ++bin)` で `bin > numBins(N/2)` のとき `idx=2*bin > N` となり `out` (サイズN) の範囲外読み出し。MKLのCCSフォーマットでは有効binは `0..N/2`。ASANでクラッシュ、Releaseでは不定値で `additionalAttenuationDb` 誤算。 |
| **Impact** | メモリ破壊、AutoGain誤動作。 |
| **Patch** | 出典: new_bug.md (third section) Bug 9。`for (bin=0; bin<=numBins; ++bin)` に修正。 |
| **Source** | new_bug.md (third section, #9) |

### C-07: `IRConverter.cpp` — 過大ジャンプ保護が常に無効

| 項目 | 内容 |
|------|------|
| **File** | `src/IRConverter.cpp` (lines 118-140) |
| **Issue** | `computePeakAndRmsWithScale(*currentIr, result.scaleFactor)` が `*currentIr` (旧IR) を使用。正しくは `*ir` (新IR)。結果、ジャンプ保護は常に `currentIr` 同士の比較になり発動しない。異常なIR切替で大音量ジャンプの可能性。 |
| **Impact** | 安全性低下。IR切替時に予期せぬ大音量。 |
| **Patch** | 出典: new_bug.md (third section) Bug 7。2行目の引数を `*currentIr` → `*ir` に修正。 |
| **Source** | new_bug.md (third section, #7) |

### C-08: `AutoGainPlanner.cpp` — QSurge係数が常に上限張り付き

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` (lines 50-54) |
| **Issue** | `peakingSurge = eqMaxGainDb * 0.15f * (20.0f / 0.707f)` で係数が約4.243倍。`eqMaxGainDb=1.06dB` で既に `1.5+4.5=6dB` 上限到達。実用域 `+2dB` 以上は全て `6dB` マージン固定。動的マージンにならずラウドネスが常時-6dB近く余計に低下。`processingOrder` 引数は未使用。 |
| **Impact** | 常時過剰なヘッドルーム。ユーザーが「Auto Gainで音が小さくなる」と感じる根本原因の一つ。 |
| **Patch** | 出典: new_bug.md (third section) Bug 1。係数を `0.02` 程度に、またはQ依存の式に変更。 |
| **Source** | new_bug.md (third section, #1) |

---

## High

### H-01: `LockFreeAudioRingBuffer.h` — チャンネル拡張競合

| 項目 | 内容 |
|------|------|
| **File** | `src/LockFreeAudioRingBuffer.h` (`push()`) |
| **Issue** | モノラル→ステレオ拡張時、`writeIndex` の更新が1回のみのため片チャンネルだけ上書きされるタイミングで他チャンネルが古いデータを保持 → L/R位相差。 |
| **Impact** | 定位異常。 |
| **Patch** | 出典: new_bug.md Bug 6。 |
| **Source** | new_bug.md (first section, High #6) |

### H-02: `CustomInputOversampler.cpp` — サイレンス最適化でDCリーク

| 項目 | 内容 |
|------|------|
| **File** | `src/CustomInputOversampler.cpp` (`decimateStage` silence path) |
| **Issue** | 無音検出パスで `history` の `keep` 部分だけクリアし `keep` 以降の未使用領域に前回の有音データが残り、`memmove` 後に再出現 → ポップノイズ。 |
| **Impact** | 無音→有音遷移時にポップノイズ。 |
| **Patch** | 出典: new_bug.md Bug 7。`capacity>keep` なら `history+keep` 以降もゼロクリア。 |
| **Source** | new_bug.md (first section, High #7) |

### H-03: `LatticeNoiseShaper.h` — ブロック末尾での状態クランプ遅延

| 項目 | 内容 |
|------|------|
| **File** | `src/LatticeNoiseShaper.h` (`clampStateSIMD`) |
| **Issue** | `clampStateSIMD` をブロック終了時に1回のみ呼出。`kOrder=9` で係数が不安定な場合、ブロック内で状態が `1e12` まで発散 → `computeFeedback` が `Inf` → 後続全サンプル `NaN` 化。`kStateLimit=1e12` はクランプ閾値として大きすぎる。 |
| **Impact** | ブロック全体のNaN化による無音。 |
| **Patch** | 出典: new_bug.md Bug 8。サンプル毎にクランプ、`isFinite` チェック追加。 |
| **Source** | new_bug.md (first section, High #8) |

### H-04: `OutputFilter.cpp` — HPFのナイキストチェック欠落

| 項目 | 内容 |
|------|------|
| **File** | `src/OutputFilter.cpp` (`makeHPF`) |
| **Issue** | `makeHPF` は `fc<=0` のみチェックし上限なし。`fc=0.49*fs` 付近で `w0≈pi`, `sin(w0)≈0`, `alpha≈0`, `a0inv≈1` となり `b0≈(1+cos)/2≈0`, `a1≈2`、極が単位円上に乗り発振。 |
| **Impact** | 高周波発振。 |
| **Patch** | 出典: new_bug.md Bug 9。`fc >= nyq (fs*0.4999)` で `makeIdentity()`。 |
| **Source** | new_bug.md (first section, High #9) |

### H-05: `UltraHighRateDCBlocker.h` — 超高レートでの精度消失によるDC除去不能

| 項目 | 内容 |
|------|------|
| **File** | `src/UltraHighRateDCBlocker.h` |
| **Issue** | `alpha = 1 - exp(-2pi*fc/fs)`, 768kHz/fc=20Hz で `alpha≈1.6e-4`。`alpha*(x-state)≈1.6e-12` が double仮数部52bitで加算消失。`killDenormal` が更に `1e-20` 未満を0にし、DCが永遠に残る。 |
| **Impact** | DCオフセット残留。 |
| **Patch** | 出典: new_bug.md Bug 10。Kahan補償加算、またはTDF-II実装に変更。 |
| **Source** | new_bug.md (first section, High #10) |

### H-06: `DSPCoreDouble.cpp` — softClipのprevSample保存バグ

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **Issue** | AVX2パスで `prevScalar` に出力ではなく入力の4番目 `data[i+3]` を保存。スカラーフォールバックでは `prevScalar=inputVal`。`[BUG-04]` コメントあり。ADAA用の `prevSample` が不連続になり得る。 |
| **Impact** | ブロック境界でADAA不連続。 |
| **Patch** | 出典: new_bug.md Bug 11。出力 `result[3]` を保存。 |
| **Source** | new_bug.md (first section, High #11) |

### H-07: `EQProcessor.Processing.cpp` — `calculateRMS` のSSE除算で0除算

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Processing.cpp` (`calculateRMS`) |
| **Issue** | `numSamples=0` のブロックが来た際、早期returnが無いと `_mm_set_sd(0)` で `_mm_div_sd` → `Inf`。RMSがInfになりAGCゲインが `0.06` に張り付く。 |
| **Impact** | AGC誤動作。 |
| **Patch** | 出典: new_bug.md Bug 12。`numSamples<=0` で早期 `return 0.0`。 |
| **Source** | new_bug.md (first section, High #12) |

### H-08: `AutoGainPlanner.cpp` — Conv→EQ 時に透過でも -6dB 強制

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` (lines 36-38) |
| **Issue** | `inputDb = min(inputDb, kConvFirstInputCeiling)` で常に `-6dB` 天井。IRフラット+EQフラットでも `-6dB→+6dB` の無駄な往復。透過チェーンのはずが -6dB のヘッドルーム＋+6dB makeup。 |
| **Impact** | 透過損失、ノイズフロア上昇。 |
| **Patch** | 出典: new_bug.md (third section) Bug 2。透過時ガード `if(inputDb > -1e-6) return 0`。 |
| **Source** | new_bug.md (third section, #2) |

### H-09: `AutoGainPlanner.cpp` — Conv only で -6dB 天井なし

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` (lines 29-32) |
| **Issue** | Conv only 分岐に天井処理なし (`inputDb = -max(0, atten-1.5)`)。手動 `setInputHeadroomDb` は最大-6dBにクランプするがAutoGainは無視 → クリップリスク。 |
| **Impact** | クリッピング。手動UIとの保護矛盾。 |
| **Patch** | 出典: new_bug.md (third section) Bug 3。`min(inputDb, -6)` または `kConvFirstInputCeiling` を適用。 |
| **Source** | new_bug.md (third section, #3) |

### H-10: `DSPCoreDouble.cpp` — `convolverInputTrimGain` が Conv→EQ で完全無視

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **Issue** | Float/Double共に `order==EQThenConvolver` の時のみ trim を適用。Conv→EQ では trim は Plannerで0だが、手動で `-6dB` 設定しても無効。`AudioEngine.Parameters.cpp` の `applyDefaultsForCurrentMode` は `EQThenConvolver` でのみ `trim=-6` デフォルトだが、順序切替で無効化。 |
| **Impact** | 手動trim設定が無視される。 |
| **Patch** | 出典: new_bug.md (third section) Bug 13。 |
| **Source** | new_bug.md (third section, #13) |

### H-11: `DeviceSettings.cpp` — タイマー5Hzが編集中を上書き

| 項目 | 内容 |
|------|------|
| **File** | `src/DeviceSettings.cpp` (`updateGainStagingDisplay`) |
| **Issue** | `inputHeadroomEditor.getText()` と `engine.getInputHeadroomDb()` が乖離したら `setText(dontSendNotification)` で上書き。ユーザータイピング中にタイマー発火で入力が消える。 |
| **Impact** | UI/UX低下。数値入力が編集中に勝手に書き換わる。 |
| **Patch** | 出典: new_bug.md (third section) Bug 16。`hasKeyboardFocus()` チェック追加。 |
| **Source** | new_bug.md (third section, #16) |

---

## Medium

### M-01: `IRAnalyzer.cpp` — MKLバッファに `std::make_unique` 使用（規約違反）+ noexcept関数からの `std::terminate` リスク

| 項目 | 内容 |
|------|------|
| **File** | `src/IRAnalyzer.cpp` / `.h` (`estimateMaxFrequencyResponseGain`) |
| **Issue** | (a) MKL使用箇所で `mkl_malloc` ではなく `std::make_unique<double[]>` (非64byteアライメント) を4箇所で使用。他の全ファイルは `convo::makeAlignedArray` で統一。 (b) `noexcept` 関数内で `std::make_unique` → `bad_alloc` → `std::terminate()` でプロセス即死。他の異常系は全て `return 1.0` のフェイルセーフ設計と矛盾。 |
| **Impact** | コーディング規約逸脱 + OOM時にプロセス終了。 |
| **Patch** | 出典: Part 1 Bug #2。`makeAlignedArray` + 明示的ゼロ埋めループ + try-catch全捕捉。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part1.md (#2) |

### M-02: RT Capability/Allocator Firewall が未接続

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/ISRRTExecution.cpp/.h` |
| **Issue** | `auditPublishAttempt()` と `onAllocAttempt()` の呼び出し箇所がコードベース中にゼロ。`operator new` のグローバルオーバーライドも存在しない (`grep` 0件)。`AtomicAccess.h::publishAtomic()` からも呼ばれていない。フラグ追跡自体は正しいが、検知側が未接続で安全網が機能していない。 |
| **Impact** | Debug/CIビルドでRT違反が検知できない。#4(EQProcessor::reset libm呼出)のような問題を本来なら自動検知できたはずが不可能。 |
| **Patch** | 出典: Part 5 Bug #6。設計判断が必要なためパッチ未提示。(1) operator newグローバルオーバーライド (2) AtomicAccess.hからの呼出。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part5.md (#6) |

### M-03: LifecycleToken が発行されるだけで検証されていない

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/ISRLifecycle.cpp/.h` |
| **Issue** | `enterPrepare/enterAudioCallback/enterRelease` は `epochId` 付き `LifecycleToken` を返すが、対になる `leave*` 3関数とも受け取った `token` 引数を一切使用しない。ヘッダのコメントで明記されている受入条件 LIF-5/LIF-6 (callback中runtimeVersion/DSP generation変化なし) の検証コードが存在しない。 |
| **Impact** | 設計上の安全機構が未実装。 |
| **Patch** | 出典: Part 6 Bug #8。仕様確認が必要なためパッチ未提示。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part6.md (#8) |

### M-04: SVF `tan` 発散ガード欠落

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Coefficients.cpp` (`calcLowPassSVF` etc.) |
| **Issue** | `g = tan(pi*f/fs)` で `f` のみ `nyquist*0.95` クランプだが `g` の上限を設けていない。`fs=44.1k/f=20k` で `g≈6.3`。`k=1/Q` と組み合わさり `a1,a2` が大 → 係数補間時に1サンプルでゲインが跳ぶ。 |
| **Impact** | フィルタ係数急変によるポップノイズ。 |
| **Patch** | 出典: new_bug.md (first section) Bug 13。`g = tan(jlimit(0, 0.45*pi, pi*f/fs))`。 |
| **Source** | new_bug.md (first section, Medium #13) |

### M-05: 大ブロック無音化

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` |
| **Issue** | `if(numSamples > maxSamplesPerBlock) buffer.clear(); return;` ホストが可変ブロックで `max` を超えた瞬間に1ブロック無音。チャンク分割すべき。 |
| **Impact** | 一部ホスト環境で不定期無音。 |
| **Patch** | 出典: new_bug.md Bug 14。チャンク分割処理。 |
| **Source** | new_bug.md (first section, Medium #14) |

### M-06: MKL DFTIスケーリングの二重適用

| 項目 | 内容 |
|------|------|
| **File** | `src/MklFftEvaluator.h` / `MKLNonUniformConvolver.cpp` |
| **Issue** | MKL側は `DFTI_BACKWARD_SCALE=1/N`、IPP側は `DIV_INV_BY_N` フラグで1/N。両パス混在時、片方で2回スケーリング → ゲイン `1/N` 倍。 |
| **Impact** | ゲイン誤差 (-6dB等)。 |
| **Patch** | 出典: new_bug.md Bug 15。手動スケールに統一。 |
| **Source** | new_bug.md (first section, Medium #15) |

### M-07: `IRConverter.cpp` `size_t` オーバーフロー

| 項目 | 内容 |
|------|------|
| **File** | `src/IRConverter.cpp` |
| **Issue** | `bytes = numChannels * numSamples * sizeof(double)` で32bit intオーバーフローしてから `size_t` にキャストされる箇所あり。 |
| **Impact** | 巨大IRロード時のバッファ確保失敗。 |
| **Patch** | 出典: new_bug.md Bug 17。`static_cast<size_t>(numChannels) * ...` に修正。 |
| **Source** | new_bug.md (first section, Medium #17) |

### M-08: キャッシュハッシュ衝突で誤ったIR再利用

| 項目 | 内容 |
|------|------|
| **File** | `src/CacheManager.cpp` |
| **Issue** | `StateKey` のハッシュがファイルパス+サンプルレートのみ。IRファイルが上書きされた場合でもキャッシュヒットし古いIRを使い続ける。 |
| **Impact** | IR更新が反映されない。 |
| **Patch** | 出典: new_bug.md Bug 18。`mtime` をハッシュに含める。 |
| **Source** | new_bug.md (first section, Medium #18) |

### M-09: MMCSSハンドルリーク

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Threading.cpp` |
| **Issue** | `AvSetMmThreadCharacteristics` を呼ぶ際、既存 `thread_local` ハンドルを開放せずに再登録 → ハンドルリーク。 |
| **Impact** | リソースリーク。 |
| **Patch** | 出典: new_bug.md Bug 19。事前に `AvRevertMmThreadCharacteristics(existingHandle)`。 |
| **Source** | new_bug.md (first section, Medium #19) |

### M-10: `fc_hc` / `fc_lp` サンプルレート分岐が粗い

| 項目 | 内容 |
|------|------|
| **File** | `src/OutputFilter.cpp` |
| **Issue** | `<=48k` と `>48k` の2分岐のみ。96kで `fc=22k` LPFは可聴帯域に影響、88.2k→19kと不連続切替。レート変更時に可聴な音色変化。 |
| **Impact** | 音質劣化。 |
| **Patch** | 出典: new_bug.md Bug 20。`jmap` で連続補間。 |
| **Source** | new_bug.md (first section, Medium #20) |

### M-11: `computeEstimatedMaxGainDb` LPF/HPFを常にブースト扱い

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **Issue** | `case LowPass/HighPass: gainBoosting=true`。Q=0.707のButterworthはパスバンド0dB/共振無し。常にブースト扱いで `maxLinearGain` 過大評価 → AutoGain過剰ヘッドルーム。 |
| **Impact** | 不要なヘッドルーム。 |
| **Patch** | 出典: new_bug.md (third section) Bug 10。Q閾値で判定または全バンド実測。 |
| **Source** | new_bug.md (third section, #10) |

### M-12: `computeEstimatedMaxGainDb` totalGainDbの二重カウント

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **Issue** | `totalGainDb` (Master Gain) を `maxLinearGain * totalGainLin` に乗算。Master GainはDSPチェーンで別途適用されるため、AutoGainの `inputHeadroom` にも含めると makeup と合算で二重補正。 |
| **Impact** | Master Gain+6dBでEQピーク無くても-6dBヘッドルーム。 |
| **Patch** | 出典: new_bug.md (third section) Bug 11。 |
| **Source** | new_bug.md (third section, #11) |

### M-13: 適応サンプリングのスキップ条件が逆

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **Issue** | `if (range > freq*0.5) continue` で低Q広帯域をスキップ。粗探索300点のみに頼るためシェルフの肩特性を見逃す。 |
| **Impact** | シェルフEQの推定誤差。 |
| **Patch** | 出典: new_bug.md (third section) Bug 12。 |
| **Source** | new_bug.md (third section, #12) |

### M-14: `BuildAnalysis` 封印失敗時のサイレントフォールバック

| 項目 | 内容 |
|------|------|
| **File** | `src/RebuildDispatch.cpp` (lines 651-656) |
| **Issue** | `sealBuildAnalysis` が generation不一致やnon-finiteで `BuildAnalysis{}` (0/0) を返すが、呼び出し側は戻り値をチェックせずそのまま使用。解析失敗時にAutoGainはフラットと誤認。 |
| **Impact** | 異常時に無音や過大出力の可能性。 |
| **Patch** | 出典: new_bug.md (third section) Bug 8。 |
| **Source** | new_bug.md (third section, #8) |

---

## Low

### L-01: `CallbackTimingHistory` リングバッファのオフバイワン

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` / `BlockDouble.cpp` / `Timer.cpp` |
| **Issue** | `fetchAddAtomic` は pre-increment値を返す（1回目wc=0）。書込み側が `(wc-1)%32` と-1しているため全書込みが1つ前のスロットにずれる。初期 `wc=0` で `(0-1)%32=31` → 最初の32CB分のうち最初の1エントリが読出し範囲外へ。読出し側は `i%32` で正しい。 |
| **Impact** | 診断ビルド(`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`, デフォルトOFF)のみ。CB_HISTダンプが最初の32件から1件欠落、以降も1つずれて表示。音声処理に影響なし。 |
| **Patch** | 出典: Part 1 Bug #1。`(wc-1)%kCallbackTimingSlots` → `wc%kCallbackTimingSlots`。2箇所。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part1.md (#1) |

### L-02: `processBand` と `processBandStereo` の異常値ハンドリング不一致

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Processing.cpp` |
| **Issue** | スカラー版は `|output|>=1e15` で `0.0` (無音化)、SIMD版はNaN/Infのみ判定で有限巨大値は±100クランプのみ（+40dBFS相当）。コメントに「processBandStereoと一貫性を保つ」とあるが逆。Mid/Sideパス→無音、ステレオパス→大信号と正反対の挙動。 |
| **Impact** | 極端な発散時（パラメータ変更直後等）にチャンネルモード次第で挙動不一致。1e15到達は稀。 |
| **Patch** | 出典: Part 3 Bug #3。SIMD版に `|output|<1e15` チェック追加（`_mm_andnot_pd` + `_mm_cmplt_pd` + `_mm_and_pd`）。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part3.md (#3) |

### L-03: `ISRRetireRouter`: null deleter のサイレント成功扱い

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/ISRRetireRouter.cpp` (`enqueueRetire`/`retireRT`/`retire`) |
| **Issue** | `ptr==nullptr` (正当なno-op) と `ptr!=nullptr && deleter==nullptr` (呼出側実装ミス=サイレントリーク) を同一条件で「成功」扱い。`RetireEnqueueResult` にエラー値なし。現行コードではdeleterは全呼出しでコンパイル時定数リテラル渡しのため実害なし。 |
| **Impact** | 防御的ハードニング。現状悪用経路なし。 |
| **Patch** | 出典: Part 4 Bug #5。3関数に `assert(!(ptr!=nullptr && deleter==nullptr))` 追加。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part4.md (#5) |

### L-04: `scanPeak` `tmp` 未初期化（軽微）

| 項目 | 内容 |
|------|------|
| **File** | `src/TruePeakDetector.cpp` (`scanPeak`) |
| **Issue** | `n<4` のとき `vPeak` は0のまま `tmp` にストア。スカラーループで正しいピークを求めるが `tmp` の0が `peak` 初期値に残る。負ピーク絶対値化のため影響なし。`n=0` で `peak=0` 返却時、真のピーク0.0と区別不能。 |
| **Impact** | 軽微。エッジケース。 |
| **Patch** | 出典: new_bug.md Bug 16。`alignas(32) double tmp[4] = {};` でゼロ初期化。 |
| **Source** | new_bug.md (first section, Medium #16) |

---

## Info

### I-01: `EQProcessor::reset()` の "(Audio Thread)" ラベルと `decibelsToGain` (libm) 呼び出し

| 項目 | 内容 |
|------|------|
| **File** | `src/eqprocessor/EQProcessor.Core.cpp` (`reset()`) / `AudioEngine.h` |
| **Issue** | (1) コメントが明確に「(Audio Thread)」。 (2) `juce::Decibels::decibelsToGain` (内部で `std::pow`=libm) を直接1回+`storeTotalGainDb`経由で1回の計2回呼出。 (3) `process()` 側では【Fix Bug #7】として同一パターンを既にlibm禁止で是正済み。 (4) `DSPCore::reset()` の呼出し箇所がコードベース中に見つからない（デッドコード可能性大）。 (5) `reset()` だけが実行スレッド保証の仕組み（jassert等）を持たない。 |
| **Impact** | コメント不整合。もし将来RTパスに組み込まれるとlibm違反再発。現状はデッドコードと推測。 |
| **Patch** | 出典: Part 3/4 Bug #4。呼出し元確認後、(a) RT呼出ならlibm排除、(b) 非RT専用ならコメント修正。パッチ未作成。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part3.md (#4) + Part4 (#4) |

### I-02: `DSPHandle` アトミックのロックフリー保証が未検証

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/ISRDSPHandle.h` |
| **Issue** | `std::atomic<DSPHandle>` (16byte) に `is_always_lock_free` の `static_assert` が無い。同ファイル内の `DSPRegistrySlot::generation` (8byte) には明示的検証あり。AVX2必須 (CMPXCHG16B有) のため実質問題ないはずだが、コンパイル時検証が不足。 |
| **Impact** | 要検証。MSVC v143でコンパイル確認未実施。通れば検証完了、通らなければロックフォールバックでRT違反。 |
| **Patch** | 出典: Part 5 Bug #7。`static_assert(std::atomic<DSPHandle>::is_always_lock_free)` 追加。必ずMSVCでコンパイル確認のこと。 |
| **Source** | ConvoPeq_bug_report_2026-07-18_part5.md (#7) |

### I-03: ネット0dB保証がクランプで崩れる

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` |
| **Issue** | `result.outputMakeupDb = jlimit(0,12, -input-trim)` で理想 `+24dB` も `12dB` クランプ → `net = -12dB`。ラウドネス低下。`testClampRanges` は範囲のみ検証、net 0dBは未検証。 |
| **Impact** | 特定条件下でユーザーが原因不明の音量低下を経験。 |
| **Patch** | 出典: new_bug.md (third section) Bug 6。クランプ後のnet誤差ログ、またはinput/trim再調整。 |
| **Source** | new_bug.md (third section, #6) |

### I-04: 両方バイパス時のテスト乖離

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` / `GainStagingContractTests.cpp` |
| **Issue** | 製品コードは `both bypassed → 0/0/0` (透過)。テストの `refPlan()` はこの分岐がなく `ConvolverThenEQ→-6/+6` 等を返す。`testNetZeroDb` の Both bypassed ケースは `net 0dB` しか見ないため検出不能。 |
| **Impact** | 契約テストが回帰を検出できない。 |
| **Patch** | 出典: new_bug.md (third section) Bug 4/5。テストを製品コードに同期。 |
| **Source** | new_bug.md (third section, #4, #5) |

### I-05: `estimateQSafetyMargin` ゼロ時挙動の乖離

| 項目 | 内容 |
|------|------|
| **File** | `src/AutoGainPlanner.cpp` / テスト |
| **Issue** | 製品: `eqMax<=0 → 0.0f` (Phase 8 Reviewで不要減衰防止)。テスト: `refQSafetyMargin(0) → 1.5f`。テストが古い仕様のまま。 |
| **Impact** | テスト失効。 |
| **Patch** | 出典: new_bug.md (third section) Bug 5。テストを製品コードに同期。 |
| **Source** | new_bug.md (third section, #5) |

### I-06: `RuntimeBuilder.cpp` dB→linear変換でNaNチェックなし

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/RuntimeBuilder.cpp` (lines 318-330) |
| **Issue** | `juce::Decibels::decibelsToGain` は `-inf` で0を返すが、`plan` がNaNならNaN伝播。旧経路は `autoGainStagingEnabled` を `relaxed` で読むため raceで中間値が読まれる可能性。 |
| **Impact** | 稀にNaN伝播→無音。 |
| **Patch** | 出典: new_bug.md (third section) Bug 14。 |
| **Source** | new_bug.md (third section, #14) |

### I-07: `DSPCoreIO.cpp` AVX2マスクで負のゼロ消失

| 項目 | 内容 |
|------|------|
| **File** | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` (`sanitizeFiniteChunk`) |
| **Issue** | `_mm256_and_pd(vData, vMask)` でNaN/Inf→0は正しいが負の0も消える。微小だがDCブロッカー前段で無音判定に影響。 |
| **Impact** | 極めて軽微。 |
| **Patch** | 出典: new_bug.md (third section) Bug 15。 |
| **Source** | new_bug.md (third section, #15) |

### I-08: `IRConverter.cpp` `computeEnergyScale` エネルギー計算アンダーフロー

| 項目 | 内容 |
|------|------|
| **File** | `src/IRConverter.cpp` |
| **Issue** | IR長が `kMaxAnalysisWindow=65536` 超でも全長で `cblas_ddot`。`1e6` 超で `energy` が `1e12` 超え `1/sqrt(energy)` アンダーフロー → `scaleFactor` が0に近づき無音化。`log` ドメインで計算すべき。 |
| **Impact** | 長尺IRで無音化。 |
| **Patch** | 出典: new_bug.md (third section) Bug 18。 |
| **Source** | new_bug.md (third section, #18) |

### I-09: `IRAnalyzer.cpp` 3点ガウス補間式が非標準

| 項目 | 内容 |
|------|------|
| **File** | `src/IRAnalyzer.cpp` |
| **Issue** | `interpolated = y0 * exp(-delta * (logY0 - logYm1))` は対称性を欠きピークを過大評価 → `additionalAttenuationDb` が大きく見積もられる。正しい式は `y_interp = y0 * exp(0.5*delta*(logYm1-logYp1))` 等。 |
| **Impact** | AutoGain過剰ヘッドルーム。 |
| **Patch** | 出典: new_bug.md (third section) Bug 19。 |
| **Source** | new_bug.md (third section, #19) |

### I-10: `TruePeakDetector.cpp` ピークホールド減衰 `0.999` 固定

| 項目 | 内容 |
|------|------|
| **File** | `src/TruePeakDetector.cpp` |
| **Issue** | ピークホールド減衰 `0.999` がサンプルレート非依存。48kHzで約20ms、192kHzで約5msと減衰速度がSR依存で変化。 |
| **Impact** | SR切替時にピーク検出特性が変わる。 |
| **Patch** | 出典: new_bug.md (third section) Bug 20。SRに応じた時定数に変更。 |
| **Source** | new_bug.md (third section, #20) |

### I-11: `DeviceSettings.cpp` `resized` レイアウト無駄

| 項目 | 内容 |
|------|------|
| **File** | `src/DeviceSettings.cpp` (`resized`) |
| **Issue** | `fixedNoiseLogIntervalLabel` を `setVisible(false)` と0サイズの両方で非表示にしている。`resized` が呼ばれる度に0にリセットされる無駄。 |
| **Impact** | 軽微（無駄な処理）。 |
| **Patch** | 出典: new_bug.md (third section) Bug 17。 |
| **Source** | new_bug.md (third section, #17) |

---

## 出典一覧

| 出典 | バグ数 | 備考 |
|------|--------|------|
| `ConvoPeq_bug_report_2026-07-18_part1.md` | 2 | 確定バグのみ。誤検知リストは含まず。 |
| `ConvoPeq_bug_report_2026-07-18_part2.md` | 0 | 問題なし検証リスト、調査範囲マップ。 |
| `ConvoPeq_bug_report_2026-07-18_part3.md` | 2 | Bug#3(Low), Bug#4(Info→Part4継続)。 |
| `ConvoPeq_bug_report_2026-07-18_part4.md` | 1 | Bug#5(Low)。Bug#4はInfoとして統合。 |
| `ConvoPeq_bug_report_2026-07-18_part5.md` | 2 | Bug#6(Medium), Bug#7(Info)。 |
| `ConvoPeq_bug_report_2026-07-18_part6.md` | 1 | Bug#8(Medium)。 |
| `new_bug.md` (first section) | 20 | C-01〜C-05, H-01〜H-07, M-04〜M-10, L-04。scanPeakはLowに下方修正。 |
| `new_bug.md` (third section) | 20 | C-06〜C-08, H-08〜H-11, M-11〜M-14, I-03〜I-11。 |

## バグ間の依存関係・注意事項

1. **H-07 (calculateRMS 0除算)** と **L-02 (processBand/SIMD不一致)** は同一ファイル(`EQProcessor.Processing.cpp`)の異なる関数。独立して修正可能。
2. **C-08 (QSurge張り付き)** + **H-08 (透過時-6dB強制)** + **C-06/C-07 (IRAnalyzer/IRConverter)** がAutoGainの「常に-6dB余計に下げる」問題の複合要因。優先対応推奨。
3. **M-01 (IRAnalyzer make_unique)** と **C-06 (IRAnalyzer FFT OOB)** は同一ファイルの異なるバグ。独立して修正可能。
4. **M-03 (LifecycleToken未検証)** + **M-02 (RT Firewall未接続)** は「安全機構は構築したが未接続」という同一パターン。体系的な洗い出しを推奨。
5. **C-05 (SafeStateSwapper race)** は排他制御追加が最も安全だが、Writer統合の可否を先に判断すべき。
6. **I-01 (EQProcessor::reset 要確認)** の解決には、まず `DSPCore::reset()` の呼び出し元確認が必要。
7. **I-02 (DSPHandle lock-free)** の `static_assert` 追加はMSVC(v143)環境でのコンパイル確認必須。通らなければ設計変更（2つのatomicに分割等）。

> **次のアクション候補**: (1) Critical 8件のパッチ適用 (2) AutoGain関連(C-08/H-08/H-09/C-06/C-07)の一括修正 (3) 未接続安全網(M-02/M-03)の体系調査
