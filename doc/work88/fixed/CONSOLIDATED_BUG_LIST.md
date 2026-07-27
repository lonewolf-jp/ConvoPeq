# 統合バグリスト — ConvoPeq BUG-011〜BUG-046

- **作成日**: 2026-07-26
- **検証方法**: ソースコード実査 + WSL grep/rg + Serena + cocoindex
- **総バグ数**: 30件（BUG-011〜BUG-046、BUG-017は欠番）
- **検証結果**: 全件現行コードに存在確認

---

## 凡例

| Mark | 意味 |
|------|------|
| ✅ | ソースコードで完全確認 |
| ⚠️ | 確認済みだが一部修正または軽微な相違あり |
| 🔶 | 理論上は成立するが発現条件が極めて稀 |

---

## 重要度別サマリ

| 重要度 | 件数 | 一覧 |
|--------|------|------|
| **CRITICAL** | 4 | BUG-034, BUG-035, BUG-036, BUG-038 |
| **HIGH** | 10 | BUG-011, BUG-012, BUG-013, BUG-014, BUG-023, BUG-028, BUG-029, BUG-030, BUG-031, BUG-037 |
| **MEDIUM** | 12 | BUG-015, BUG-016, BUG-024, BUG-025, BUG-026, BUG-032, BUG-033, BUG-039, BUG-040, BUG-042, BUG-044, BUG-045 |
| **LOW** | 4 | BUG-018, BUG-019, BUG-020, BUG-041, BUG-043, BUG-046 |

---

## 各バグ検証結果

### BUG-011 — CmaEsOptimizer::deserializeFrom sigma 未クランプ

- **ファイル**: `src/CmaEsOptimizer.h:79`
- **検証**: ✅ `sigma = inSigma;` クランプなし確認
- **リスク**: **HIGH** — sigma=0 で除算-by-ゼロ → inf/NaN → Cholesky 分解失敗
- **修正**: `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);`

### BUG-012 — CmaEsOptimizerDynamic::setSigma 未クランプ

- **ファイル**: `src/CmaEsOptimizerDynamic.h:29`
- **検証**: ✅ `void setSigma(double s) noexcept { sigma = s; }` 確認
- **リスク**: **HIGH** — sigma=0/負/過大で update() 内除算異常
- **修正**: `sigma = std::clamp(s, params.sigmaMin, params.sigmaMax);`

### BUG-013 — CmaEsOptimizerDynamic::deserializeFrom sigma 未クランプ

- **ファイル**: `src/CmaEsOptimizerDynamic.cpp:204`
- **検証**: ✅ `sigma = inSigma;` クランプなし確認
- **リスク**: **HIGH** — BUG-011 と同一問題、Dynamic版
- **修正**: `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);`

### BUG-014 — juce::String currentDeviceTypeName_ データ競合 (CoW race)

- **ファイル**: `src/audioengine/AudioEngine.h:2265,2273`, `AudioEngine.Mmcss.cpp:56`
- **検証**: ✅ setter は Message Thread、getter/getCurrentMmcssPolicy は Audio Thread
- **リスク**: **HIGH** — juce::String CoW の Use-After-Free
- **修正**: `std::atomic<const char*>` + strdup/free に変更

### BUG-015 — enqueueWithRetry 戻り値無視

- **ファイル**: `src/audioengine/ISRRetireRouter.cpp:154`, `src/core/SnapshotCoordinator.cpp:37,89`
- **検証**: ✅ 3箇所すべて確認（ISRRetireRouter: `(void)` キャスト、SnapshotCoordinator: 2箇所未使用）
- **リスク**: **MEDIUM** — QueuePressure 時に retire エントリが永久リーク
- **修正**: 戻り値チェック + ログ、またはフォールバック delete

### BUG-016 — CmaEsOptimizer sanitize が NaN/Inf を処理しない

- **ファイル**: `src/CmaEsOptimizerDynamic.h:50`, `src/CmaEsOptimizer.h:201`
- **検証**: ✅ `return (std::abs(x) < 1e-15) ? 0.0 : x;` — NaN/Inf 素通り確認
- **リスク**: **MEDIUM** — 二次防御線欠如
- **修正**: `if (!std::isfinite(x) || std::abs(x) < 1e-15) return 0.0;`

### BUG-018 — FP `!= 1.0` exact comparison (3箇所)

- **ファイル**: `LoadPipeline.cpp:347`, `DSPCoreDouble.cpp:440`, `MKLNonUniformConvolver.cpp:1048`
- **検証**: ✅ 全3箇所 grep 確認
- **リスク**: **LOW** — 実害は無視できるが、コーディング規約違反
- **修正**: `std::abs(x - 1.0) > kUnityEpsilon` に変更

### BUG-019 — TruePeakDetector 整数オーバーフロー

- **ファイル**: `src/TruePeakDetector.cpp:102-111`
- **検証**: ✅ `const int up1Samples = numSamples * 2;` / `up2Samples = numSamples * 4;` 確認
- **リスク**: **LOW** — numSamples > 536M で UB。通常オーディオでは発生しない
- **修正**: `static_cast<long long>` または `size_t` に変更

### BUG-020 — `jlimit(0, targetLength-1, ...)` 下限 > 上限

- **ファイル**: `src/convolver/ConvolverProcessor.LoaderThread.cpp:198`
- **検証**: ✅ `juce::jlimit(0, targetLength - 1, irPeakLatency)` 確認。targetLength=0 で upper=-1
- **リスク**: **LOW** — jmax(0, ...) で回復されるが、セマンティクス不正
- **修正**: `if (targetLength <= 0) return 0;`

### BUG-021 — timerCallback に RCU guard なし

- **ファイル**: `src/convolver/ConvolverProcessor.Lifecycle.cpp` (timerCallback)
- **検証**: ✅ 他関数(Runtime.cpp:211, Lifecycle.cpp:462) は RCUGuard/GlobalGuard あり
- **リスク**: **LOW** — 現状のスレッドモデルでは Message Thread serialization により実害なし
- **修正**: timerCallback 冒頭に GlobalGuard 追加

### BUG-022 — prepareToPlay に RCU guard なし

- **ファイル**: `src/convolver/ConvolverProcessor.Lifecycle.cpp:228-274`
- **検証**: ✅ 関数全体が RCU 未保護。他関数は全件保護済み
- **リスク**: **LOW** — 現状は Message Thread 上でのみ呼ばれる
- **修正**: prepareToPlay 冒頭に GlobalGuard 追加

### BUG-023 — SafeStateSwapper swap() vs tryReclaim() 競合

- **ファイル**: `src/SafeStateSwapper.h:103-131 (swap), 201-272 (tryReclaim)`
- **検証**: ✅ 両関数が tail 操作で競合 window 確認
- **リスク**: **HIGH** — ConvolverState のメモリリーク（最大数十MB/エントリ）
- **修正**: CAS で排他、または move-path 削除

### BUG-024 — SnapshotFadeState advance() vs resetToIdle() カウンター不整合

- **ファイル**: `src/core/SnapshotFadeState.h:41-67 (advance), 85-91 (resetToIdle)`
- **検証**: ✅ advance が remainingSamples_ 書き込み後に state 未再確認
- **リスク**: **MEDIUM** — invariant 違反。デバッグアサーション誤発火
- **修正**: advance で remaining 書き込み後に state を再確認

### BUG-025 — switchImmediate → resetFadeStateAndRetireTarget が enqueueRetry 未使用

- **ファイル**: `src/core/SnapshotCoordinator.cpp:57-72`
- **検証**: ✅ Non-RT 呼び出しにもかかわらず enqueueRetire（再試行なし）
- **リスク**: **MEDIUM** — リングバッファ満杯時に GlobalSnapshot リーク
- **修正**: switchImmediate 内で enqueueWithRetry を使用する分岐を追加

### BUG-026 — ObservedRuntime::get() が rootEnterSucceeded() を確認しない

- **ファイル**: `src/core/ObservedRuntime.h:42-49`
- **検証**: ⚠️ RCU enter 失敗時に Release ビルドで UAF 可能性
- **リスク**: **MEDIUM** — プログラミングエラー時の防御層不足
- **修正**: `get()` で `guard.rootEnterSucceeded()` 確認

### BUG-027 — completeFade() と updateFade() の競合

- **ファイル**: `src/core/SnapshotCoordinator.cpp:74-92`, `SnapshotCoordinator.h:101-131`
- **検証**: ✅ target promotion と updateFade 読み取りの競合 window 確認
- **リスク**: **LOW** — 1ブロックのクロスフェード終端欠落
- **修正**: updateFade で target==null 時に state を再確認

### BUG-028 — CrossfadeRuntime::complete() が stale フラグをリセットしない

- **ファイル**: `src/audioengine/CrossfadeRuntime.h:93-105`
- **検証**: ✅ complete() は pending_/queuedFadeTimeSec_/fadeStartTimestampUs_ のみリセット。useDryAsOld_/firstIrDryPending_/firstIrDryDone_ は `reset()` でのみリセット
- **リスク**: **HIGH** — stale フラグによる Dry/Wet 混合誤り → 可聴歪み
- **修正**: complete() にも全フラグリセットを追加

### BUG-029 — DSPTransition Emergency Override が exchangeFadingRuntimeDSP をスキップ

- **ファイル**: `src/audioengine/DSPTransition.h:54-74`
- **検証**: ✅ 通常パス (line 91-92) は exchangeFadingRuntimeDSP あり、Emergency パスはなし
- **リスク**: **HIGH** — fadingRuntimeDSPSlot に stale DSP 残留 → UAF
- **修正**: Emergency パスでも exchangeFadingRuntimeDSP を呼ぶ

### BUG-030 — Timer の exchangeFadingRuntimeDSP(nullptr) と DSPTransition の書き込み競合

- **ファイル**: `src/audioengine/AudioEngine.Timer.cpp:1000-1008`, `DSPTransition.h:91-92`
- **検証**: ✅ 同一 fadingRuntimeDSPSlot を Timer と DSPTransition が競合
- **リスク**: **HIGH** — Use-After-Free、オーディオクラッシュ
- **修正**: isFading() + isPending() でガード、または CAS 採用

### BUG-031 — updateAudioThreadSnapshotFade がスタブ + 未呼び出し

- **ファイル**: `src/audioengine/AudioEngine.h:3696-3706`
- **検証**: ✅ 関数定義確認（常に alpha=1.0, return false）。grep で呼び出し箇所なし。BlockDouble.cpp に advanceFade() なし
- **リスク**: **HIGH** — パラメータクロスフェード (EQ/NS/AGC) が全く機能していない。BlockDouble.cpp で fade 状態が永久スタック
- **修正**: updateAudioThreadSnapshotFade を実装し SnapshotCoordinator::updateFade() を呼ぶ

### BUG-032 — createSnapshotFromCurrentState の torn-read

- **ファイル**: `src/audioengine/AudioEngine.Snapshot.cpp:28-53`
- **検証**: ⚠️ 14個の atomic 変数を個別読み取り。全体非アトミック
- **リスク**: **MEDIUM** — 矛盾したパラメータセットが 1 tick だけ公開
- **修正**: `std::atomic<SnapshotParams>` に集約

### BUG-033 — BlockDouble.cpp クロスフェードミックスが dryScale 未適用

- **ファイル**: `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:400-427`
- **検証**: ⚠️ ラムダキャプチャに dryScale なし。シングル精度パス (AudioBlock.cpp) ではあり
- **リスク**: **MEDIUM** — ダブル精度 + firstIrDry 条件でポップ/クリック
- **修正**: dryScale をラムダに取り込む

### BUG-034 — IPP FFT 戻り値未チェック (7箇所)

- **ファイル**: `src/MKLNonUniformConvolver.cpp:1043,1060,1376,1436,1570,1637`
- **検証**: ✅ 全 7 箇所 IppStatus 無視確認
- **リスク**: **CRITICAL** — FFT 失敗時に無言で出力破損 → 可聴ノイズ
- **修正**: IppStatus チェック + 失敗時は出力バッファをゼロクリア

### BUG-035 — applyComputedIR 世代不一致で isLoading 固着

- **ファイル**: `src/convolver/ConvolverProcessor.LoadPipeline.cpp:321-334`
- **検証**: ✅ コード実査。世代不一致検出時に `return` するが `isLoading = false` 未設定
- **リスク**: **CRITICAL** — 回復不能な "Loading" 状態スタック
- **修正**: `convo::publishAtomic(isLoading, false, ...)` を return 直前に追加

### BUG-036 — init() 失敗時に irL/irR がリーク

- **ファイル**: `src/convolver/ConvolverProcessor.LoadPipeline.cpp:616`
- **検証**: ✅ `if (newConv->init(irL.release(), irR.release(), ...))` — release() 後に init 失敗で誰も解放しない
- **リスク**: **CRITICAL** — 1回の init 失敗で数十〜数百MBリーク
- **修正**: `.release()` → `.get()` + 成功時のみ解放

### BUG-037 — loaderTrashBin 内スレッドが dangling reference

- **ファイル**: `src/convolver/ConvolverProcessor.LoadPipeline.cpp:51-55, 551-579`
- **検証**: ✅ push_back + cleanup ループ確認。スレッド終了を非ブロッキング確認
- **リスク**: **HIGH** — Use-After-Free、プラグインアンロード時クラッシュ
- **修正**: ownerAlive フラグ追加、または WeakReference 全パスチェック

### BUG-038 — SpectrumAnalyzer FFT スケーリング誤差 (+6 dB)

- **ファイル**: `src/SpectrumAnalyzerComponent.h:74`
- **検証**: ✅ `FFT_MAGNITUDE_SCALE = 4.0f / NUM_FFT_POINTS` 確認。正解は `2.0f / NUM_FFT_POINTS`
- **リスク**: **CRITICAL** — 全周波数で +6 dBFS の常時誤差
- **修正**: `4.0f` → `2.0f`

### BUG-039 — CustomInputOversampler::processDown バッファ過剰読み取り

- **ファイル**: `src/CustomInputOversampler.cpp:836-841`
- **検証**: 🔶 passthrough 時に出力サイズ分だけ入力読み取り。upsampleRatio==1 で発現
- **リスク**: **MEDIUM** — 条件次第で常時ノイズ/クラックル
- **修正**: `min(targetSamples, upsampledBlock.getNumSamples())` に制限

### BUG-040 — NoiseShaperLearner 再生時間計算が 1 Hz フォールバック

- **ファイル**: `src/NoiseShaperLearner.cpp:1164-1168`
- **検証**: ⚠️ フォールバック値 1 を確認（48000 が適切）
- **リスク**: **MEDIUM** — エンジン未初期化時に学習が瞬時完了
- **修正**: フォールバック値を 48000 に変更

### BUG-041 — NoiseShaperLearner VLA でスタックオーバーフロー

- **ファイル**: `src/NoiseShaperLearner.cpp:643`
- **検証**: ✅ `alignas(64) double tanhBuffer[totalCoeffs]` — VLA (C99, MSVC 非互換)
- **リスク**: **LOW** — MSVC ではコンパイルエラー。GCC/Clang でも population 大でスタック OF
- **修正**: ScopedAlignedPtr または std::vector に変更

### BUG-042 — CmaEsOptimizer Rule of Five 違反 (生ポインタ所有)

- **ファイル**: `src/CmaEsOptimizer.h`
- **検証**: ✅ デストラクタあり、コピー/ムーブ制御なし確認
- **リスク**: **MEDIUM** — 暗黙コピー発生時に二重解放
- **修正**: `= delete` 4種、または `std::unique_ptr` 化

### BUG-043 — IRConverter estimateMaxFrequencyResponseGain sampleRate 誤表示

- **ファイル**: `src/IRConverter.h:46-47`, `src/IRConverter.cpp:394-399`
- **検証**: ✅ `double /*sampleRate*/` — 引数完全無視確認
- **リスク**: **LOW** — 現在の動作に影響なし。保守性の問題
- **修正**: パラメータ削除、または将来拡張用コメントを明示

### BUG-044 — MklFftEvaluator Rule of Five 違反 (IPP + 生ポインタ)

- **ファイル**: `src/MklFftEvaluator.h`
- **検証**: ✅ 6個の生ポインタを手動管理。コピー/ムーブ制御なし確認
- **リスク**: **MEDIUM** — 暗黙コピーで IPP リソース二重解放
- **修正**: `= delete` 4種、または RAII ラッパー化

### BUG-045 — IRConverter resample failure mislabels sample rate

- **ファイル**: `src/IRConverter.cpp:268-272`
- **検証**: ✅ コード実査。resample 失敗時、データは sourceRate のまま targetSampleRate と標示
- **リスク**: **MEDIUM** — 周波数解析が誤った Nyquist 周波数で計算
- **修正**: `actualSampleRate = sourceRate;` に変更、または fail closed

### BUG-046 — PsychoacousticDither Rule of Five 違反

- **ファイル**: `src/PsychoacousticDither.h:580`
- **検証**: ✅ 生ポインタ `shaperStateBuffer` 所有。コピー=delete、ムーブ未宣言
- **リスク**: **LOW** — 現在ムーブ不可。将来のリファクタリングで二重解放リスク
- **修正**: `= default` ムーブ追加、または `unique_ptr` 化

---

## カテゴリ別集計

| カテゴリ | 件数 | バグ番号 |
|----------|------|----------|
| 数値計算 (除算-by-ゼロ/NaN) | 4 | BUG-011, BUG-012, BUG-013, BUG-016 |
| データ競合 / RCU / アトミック | 8 | BUG-014, BUG-021, BUG-022, BUG-023, BUG-024, BUG-027, BUG-029, BUG-030 |
| リソースリーク / エラーハンドリング | 5 | BUG-015, BUG-025, BUG-034, BUG-036, BUG-037 |
| Rule of Five 違反 | 3 | BUG-042, BUG-044, BUG-046 |
| ロジックエラー | 5 | BUG-028, BUG-031, BUG-033, BUG-038, BUG-045 |
| エッジケース / 整数オーバーフロー | 3 | BUG-019, BUG-020, BUG-041 |
| コード品質 (FP比較/パラメータ) | 2 | BUG-018, BUG-043 |
| その他 | 4 | BUG-026, BUG-032, BUG-035, BUG-039, BUG-040 |

---

## 推奨修正優先順位


### Phase 1 — CRITICAL (即時対応)

1. **BUG-038**: SpectrumAnalyzer FFT スケーリング (+6 dB) — 1行修正
2. **BUG-035**: applyComputedIR isLoading 固着 — 1行追加
3. **BUG-036**: init() 失敗時 irL/irR リーク — 3行修正
4. **BUG-034**: IPP FFT 戻り値未チェック — 7箇所 IppStatus 追加

### Phase 2 — HIGH (早期対応)

5. **BUG-011/012/013**: CMA-ES sigma クランプ — 3箇所の std::clamp 追加
6. **BUG-030**: Timer vs DSPTransition fading slot 競合 — CAS/二重ガード
7. **BUG-029**: Emergency Override exchangeFadingRuntimeDSP 欠落 — 1行追加
8. **BUG-023**: SafeStateSwapper 競合 — CAS 排他
9. **BUG-028**: CrossfadeRuntime::complete() フラグリセット — 3行追加
10. **BUG-031**: updateAudioThreadSnapshotFade 実装 — 要設計

### Phase 3 — MEDIUM (計画的対応)

11. BUG-032: torn-read 防止
12. BUG-033: BlockDouble.cpp dryScale 追加
13. BUG-015: enqueueWithRetry 戻り値チェック
14. BUG-016: sanitize NaN/Inf チェック
15. BUG-024: advance() vs resetToIdle() 競合修正
16. BUG-025: switchImmediate enqueueWithRetry 化
17. BUG-037: loaderTrashBin UAF 防止
18. BUG-039: Oversampler バッファ制限
19. BUG-040: 1Hz フォールバック修正
20. BUG-042/044/046: Rule of Five 修正
21. BUG-045: IRConverter resample fallback 修正

### Phase 4 — LOW (余裕時)

22. BUG-018: FP epsilon 比較
23. BUG-019: 整数オーバーフロー修正
24. BUG-020: jlimit ガード
25. BUG-021/022: RCU guard 追加
26. BUG-026: rootEnterSucceeded 確認
27. BUG-027: completeFade vs updateFade 競合
28. BUG-041: VLA → ヒープ確保
29. BUG-043: パラメータ名修正

---

## 修正ステータス

| バグ | 修正済み | 確認日 | 備考 |
|------|----------|--------|------|
| BUG-011〜BUG-046 | ❌ 未着手 | 2026-07-26 | 全30件未修正 |

---

*Generated by automated code analysis with context-mode MCP + WSL grep/rg + Serena + AiDex*
