# INTEGRATED_BUG_LIST — 未確認 mini bugs 統合リスト

- **作成日**: 2026-09-09
- **統合元**: 本フォルダ内の個別バグドキュメント 35 件（BUG-011〜BUG-046・欠番 017 は既存）
- **共通発見日**: 2026-07-26（全件）
- **修正状態**: 全件 **未修正（unchecked）** — 2026-09-09 時点で個別の修正記録は本リスト作成時点で未確認
- **注意**: 本リストは統合サマリであり、詳細（コード引用・再現手順・修正案全文）は各個別ファイルを参照すること。ファイル名は本リストの「元ファイル」列に記載。

---

## 1. サマリ（リスク別）

| リスク | 件数 | BUG 番号 |
|---|---|---|
| HIGH | 4 | 011, 012, 013, 014 |
| MEDIUM | 2 | 015, 016 |
| LOW / Medium / Low（明記あり） | 5 | 018, 019, 020, 045, 046 |
| 明記なし（詳細ファイル参照） | 24 | 021〜041, 042, 043, 044 |

> BUG-021 以降の多くはヘッダ形式が異なりリスク欄を持たない。内容から見た実質リスクは §2 のカテゴリ分類と個別記載を参照（特に BUG-014 系の競合・UAF は HIGH 相当の可能性があるが、原文にリスク明記がないためここでは分類しない）。

---

## 2. カテゴリ別統合リスト

### 2-1. 数値計算系（8 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 011 | `CmaEsOptimizer::deserializeFrom` が sigma をクランプしない（除算-by-ゼロ） | CmaEsOptimizer.h:75-80 / DeviceSettings.cpp:942 | HIGH | BUG-011_cmaes_deserialize_sigma_no_clamp.md |
| 012 | `CmaEsOptimizerDynamic::setSigma` が sigma をクランプしない | CmaEsOptimizerDynamic.h | HIGH | BUG-012_cmaes_setsigma_no_clamp.md |
| 013 | `CmaEsOptimizerDynamic::deserializeFrom` が sigma をクランプしない（除算-by-ゼロ） | CmaEsOptimizerDynamic.h / AllpassDesigner.cpp | HIGH | BUG-013_cmaes_dynamic_deserialize_sigma_no_clamp.md |
| 016 | sanitize 関数が NaN/Infinity を処理しない | CmaEsOptimizer / CmaEsOptimizerDynamic | MEDIUM | BUG-016_sanitize_no_nan_inf_check.md |
| 018 | 浮動小数点 `!= 1.0` の等価比較 | 詳細参照 | LOW（設計原則違反） | BUG-018-FP-eq-comparison.md |
| 019 | TruePeakDetector バッファオフセット計算の整数オーバーフローリスク | TruePeakDetector | LOW（発現時は UB） | BUG-019-TPD-int-overflow.md |
| 038 | SpectrumAnalyzer FFT マグニチュードスケーリング誤差（全周波数 +6 dB） | SpectrumAnalyzerComponent.cpp:456-501 | 明記なし（UI 表示品質） | BUG-038-spectrum-analyzer-6dB-error.md |
| 041 | `NoiseShaperLearner::evaluatePopulation` の VLA によるスタック破壊 | NoiseShaperLearner.cpp:643 | 明記なし（発現時スタック破壊） | BUG-041-nsl-vla-stack-overflow.md |

**共通テーマ**: sigma クランプ欠如（011/012/013 は同一 root cause の 3 変体 — まとめて修正可能）。NaN/Inf 伝播（016）。数値演算のエッジケース（018/019）。VLA は MSVC 非標準拡張でスタック破壊リスク（041）。

### 2-2. データ競合 / スレッドセーフ系（8 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 014 | `juce::String currentDeviceTypeName_` の CoW データ競合（MessageThread 書込 vs AudioThread 読取 → UAF） | AudioEngine.h:2278 / AudioEngine.Mmcss.cpp:54-64 | HIGH | BUG-014_juce_string_cow_data_race.md |
| 021 | timerCallback が RCU reader なしで engine にアクセス | ConvolverProcessor.Lifecycle.cpp:150-169 | 明記なし | BUG-021-timer-no-rcu-guard.md |
| 022 | prepareToPlay が RCU reader なしで engine データにアクセス | ConvolverProcessor.Lifecycle.cpp:228-274 | 明記なし | BUG-022-prepareToPlay-no-rcu-guard.md |
| 023 | SafeStateSwapper tryReclaim() move-path と swap() のリングバッファ競合 | SafeStateSwapper.h:103-131, 201-272 | 明記なし | BUG-023-SafeStateSwapper-race.md |
| 024 | SnapshotFadeState advance()（Audio Thread）vs resetToIdle()（Timer）のカウンター不整合 | core/SnapshotFadeState.h:41-67, 85-91 | 明記なし | BUG-024-SnapshotFadeState-race.md |
| 027 | completeFade()（Timer）と updateFade()（Audio Thread）の競合による 1 ブロックのクロスフェード欠落 | core/SnapshotCoordinator.cpp:74-92 / .h:101-131 | 明記なし | BUG-027-completeFade-updateFade-race.md |
| 030 | Timer の exchangeFadingRuntimeDSP(nullptr) と DSPTransition の書き込み競合 | AudioEngine.Timer.cpp:1000-1008 / DSPTransition.h:91-92 | 明記なし | BUG-030-Timer-exchangeFadingDSP-vs-DSPTransition-race.md |
| 037 | loaderTrashBin 内スレッドが ConvolverProcessor 破棄後に dangling reference を保持 | ConvolverProcessor.LoadPipeline.cpp:51-55, 551-579 | 明記なし（UAF） | BUG-037-loader-thread-trash-bin-UAF.md |

**共通テーマ**: Timer/MessageThread（NonRT）と Audio Thread の共有データ保護欠如。BUG-014 と 037 は UAF 潜在で HIGH 相当。021/022 は RCU reader 契約違反。023/024/027/030 は非 atomic な複合操作の interleave。

### 2-3. リーク / 所有権系（6 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 015 | enqueueWithRetry の戻り値無視による QueuePressure サイレントドロップ | ISRRetireRouter.cpp / SnapshotCoordinator.cpp | MEDIUM | BUG-015_enqueuewithretry_return_ignored.md |
| 025 | switchImmediate 経由の resetFadeStateAndRetireTarget で enqueueRetry 未使用によるリーク | core/SnapshotCoordinator.cpp:68 / .h:83 | 明記なし | BUG-025-switchImmediate-enqueueRetry-leak.md |
| 028 | CrossfadeRuntime::complete() が useDryAsOld_ / firstIrDryPending_ をリセットしない（stale flag 固着） | audioengine/ISR/CrossfadeRuntime.h:93-98 | 明記なし | BUG-028-CrossfadeRuntime-complete-leaks-stale-flags.md |
| 035 | applyComputedIR() の世代不一致で isLoading が true に固着 | ConvolverProcessor.LoadPipeline.cpp:329-334 | 明記なし | BUG-035-isLoading-stuck-on-generation-mismatch.md |
| 036 | finalizeNUCEngineOnMessageThread で init() 失敗時に irL/irR がリーク | ConvolverProcessor.LoadPipeline.cpp:616-618 | 明記なし | BUG-036-irL-irR-release-leak-on-init-failure.md |
| 039 | CustomInputOversampler::processDown — passthrough 時の入力バッファ overread | CustomInputOversampler.cpp:836-841 | 明記なし | BUG-039-oversampler-buffer-overread.md |

**共通テーマ**: 戻り値無視・早期 return 時の状態後始ond漏れ。025 は 015 と同一の enqueueRetry 契約問題。035 は状態固着による機能停止。

### 2-4. Rule of Five / 生ポインタ所有系（3 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 042 | CmaEsOptimizer Rule of Five 違反 — 生ポインタ所有による二重解放リスク | CmaEsOptimizer.h | Medium（将来リファクタで発症） | BUG-042_cmaes_rule_of_five.md |
| 044 | MklFftEvaluator Rule of Five 違反 — IPP リソース + 生ポインタ所有 | MklFftEvaluator.h（830 行） | Medium | BUG-044_mklfftevaluator_rule_of_five.md |
| 046 | PsychoacousticDither Rule of Five 違反 — 生所有ポインタ `shaperStateBuffer` | PsychoacousticDither.h:55-587 | Low（ムーブ有効化時に double-free） | BUG-046_psychoacousticdither_rule_of_five.md |

**共通テーマ**: `makeAlignedArray().release()` → 生ポインタ + aligned_free のパターンでコピー/ムーブ制御が未定義。現行使用では発症しないが、リファクタリング時の地雷。

### 2-5. クロスフェード / Snapshot 意味論系（4 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 026 | ObservedRuntime::get() が rootEnterSucceeded() を確認しない | core/ObservedRuntime.h:42-49 / RCUReader.h:36-83 | 明記なし | BUG-026-ObservedRuntime-rootEnterSucceeded-gap.md |
| 029 | DSPTransition Emergency Override パスが exchangeFadingRuntimeDSP を呼ばない | DSPTransition.h:54-74 | 明記なし | BUG-029-DSPTransition-emergency-override-skips-exchangeFadingDSP.md |
| 031 | updateAudioThreadSnapshotFade() がスタブ — SnapshotCoordinator の alpha が未使用 | AudioEngine.h:3696-3706 / AudioBlock.cpp:475 | 明記なし | BUG-031-updateAudioThreadSnapshotFade-stub.md |
| 033 | BlockDouble.cpp のクロスフェードミックスが dryScale を適用しない | BlockDouble.cpp:400-427 / AudioBlock.cpp:432-451 | 明記なし | BUG-033-double-path-dryScale-missing.md |

**共通テーマ**: snapshot fade / crossfade の alpha 適用経路の不完全実装（スタブ・未接続・パス間不整合）。音響的な微細な出力差を生む系。

### 2-6. サンプルレート / IR 変換系（3 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 040 | NoiseShaperLearner 再生時間計算が 1 Hz にフォールバックし学習が異常終了 | NoiseShaperLearner.cpp:1164-1168 | 明記なし | BUG-040-nsl-playbacktime-1hz-fallback.md |
| 043 | IRConverter::estimateMaxFrequencyResponseGain の sampleRate パラメータが実質未使用 | IRConverter.h:46-47 / .cpp:394-399 | Low（API 意図と乖離） | BUG-043_irconverter_samplerate_mislabel.md |
| 045 | IRConverter::convertFile — resample 失敗時に sample rate を誤ラベルし周波数解析が破綻 | IRConverter.cpp:258-281 | Medium | BUG-045_irconverter_resample_failure_samplerate_mislabel.md |

**共通テーマ**: サンプルレートのメタデータとデータ実体の不一致。045 はデータ破壊系として Medium。

### 2-7. エラーハンドリング / その他（3 件）

| BUG | タイトル | 対象 | リスク | 元ファイル |
|---|---|---|---|---|
| 020 | `juce::jlimit` 下限 > 上限時の未定義動作 | LoaderThread.cpp:198 | LOW | BUG-020-jlimit-edge.md |
| 032 | createSnapshotFromCurrentState で個別 atomic 読み取り間の torn-read | AudioEngine.Snapshot.cpp:28-53 | 明記なし | BUG-032-snapshot-param-torn-read.md |
| 034 | IPP FFT 関数の戻り値未チェック — 無言でガページデータ伝搬 | MKLNonUniformConvolver.cpp（7 箇所） | 明記なし | BUG-034-IPP-FFT-return-code-unchecked.md |

---

## 3. 修正優先度の提案（実運用リスク順）

> 本リストは未確認（unchecked）バグの統合であり、優先度は各原文のリスク記載と影響Path からの推定。修正着手時は D159 通常開発ゲート（scope 確定 → invariant 影響確認）を通すこと。

### Tier 1 — HIGH（早めの対処推奨）

1. **BUG-011/012/013** — sigma クランプ欠如 3 変体。同一修正方針（deserializeFrom / setSigma で clamp）で一括修正可能。除算-by-ゼロ → NaN 伝播 → 最適化機能停止。
2. **BUG-014** — juce::String CoW UAF。Audio Thread クラッシュの可能性。atomic 化またはセッション開始前設定保証。
3. **BUG-037** — loaderTrashBin UAF。プロセッサ破棄タイミングで dangling reference。

### Tier 2 — 競合・リーク（条件付き発症）

4. **BUG-015/025** — enqueueRetry 戻り値契約（同一 root cause・一括修正候補）。
5. **BUG-021/022** — RCU reader 契約違反（timerCallback / prepareToPlay）。
6. **BUG-023/024/027/030** — Audio Thread vs Timer の非 atomic 複合操作。
7. **BUG-035/036** — 早期 return 時の状態固着 / init 失敗時のリーク。

### Tier 3 — 数値・データ品質

8. **BUG-016** — NaN/Inf sanitize。
9. **BUG-040/043/045** — サンプルレート誤ラベル系（045 は Medium）。
10. **BUG-034** — IPP 戻り値チェック（7 箇所）。
11. **BUG-041** — VLA 廃止（std::array/vector へ）。

### Tier 4 — 将来リスク・コード品質

12. **BUG-042/044/046** — Rule of Five（リファクタリング前に `= delete` 追加が最小修正）。
13. **BUG-018/019/020** — FP 等価比較 / int overflow / jlimit edge。
14. **BUG-026/028/029/031/032/033/038/039** — 意味論不完全実装・表示品質系。

---

## 4. 関連メモ

- **BUG-017 は欠番**（本フォルダに存在しない）。BUG-001〜010 は別管理（mini_bugs_checked 等）と推定される。
- BUG-011〜013・015・016 系は `CmaEsOptimizer` / `SnapshotCoordinator` / `ISRRetireRouter` に集中しており、**2026-07-26 以降の大規模改修（D135〜D179 系列）で対象コードが変更されている可能性がある**。修正着手前には必ず現行 HEAD で該当箇所の実在性を再確認すること（inventory stale 化の教訓 — D163/D174 参照）。
- 特に ISR 関連（015/025/026/028）は **D101〜D179 の authority 改修で既に構造が変わっている可能性が高い**。Closed boundary（T3c / Retire / Publish authority）に接触する修正は D159 ゲート必須。
- 音響パス系（027/031/033/038/039）は機能品質の問題であり、Practical Stable ISR の safety 境界には直接触れない。

---

## 5. 個別ファイル索引（アルファベット順）

| 元ファイル | BUG | カテゴリ |
|---|---|---|
| BUG-011_cmaes_deserialize_sigma_no_clamp.md | 011 | 数値 / クランプ |
| BUG-012_cmaes_setsigma_no_clamp.md | 012 | 数値 / クランプ |
| BUG-013_cmaes_dynamic_deserialize_sigma_no_clamp.md | 013 | 数値 / クランプ |
| BUG-014_juce_string_cow_data_race.md | 014 | 競合 / UAF |
| BUG-015_enqueuewithretry_return_ignored.md | 015 | リーク / 契約 |
| BUG-016_sanitize_no_nan_inf_check.md | 016 | 数値 / NaN-Inf |
| BUG-018-FP-eq-comparison.md | 018 | 数値 / FP |
| BUG-019-TPD-int-overflow.md | 019 | 整数 / overflow |
| BUG-020-jlimit-edge.md | 020 | エッジケース |
| BUG-021-timer-no-rcu-guard.md | 021 | 競合 / RCU |
| BUG-022-prepareToPlay-no-rcu-guard.md | 022 | 競合 / RCU |
| BUG-023-SafeStateSwapper-race.md | 023 | 競合 / リングバッファ |
| BUG-024-SnapshotFadeState-race.md | 024 | 競合 / 状態 |
| BUG-025-switchImmediate-enqueueRetry-leak.md | 025 | リーク / 契約 |
| BUG-026-ObservedRuntime-rootEnterSucceeded-gap.md | 026 | 意味論 / RCU |
| BUG-027-completeFade-updateFade-race.md | 027 | 競合 / fade |
| BUG-028-CrossfadeRuntime-complete-leaks-stale-flags.md | 028 | 意味論 / stale flag |
| BUG-029-DSPTransition-emergency-override-skips-exchangeFadingDSP.md | 029 | 意味論 / transition |
| BUG-030-Timer-exchangeFadingDSP-vs-DSPTransition-race.md | 030 | 競合 / transition |
| BUG-031-updateAudioThreadSnapshotFade-stub.md | 031 | 意味論 / スタブ |
| BUG-032-snapshot-param-torn-read.md | 032 | 競合 / torn-read |
| BUG-033-double-path-dryScale-missing.md | 033 | 意味論 / dryScale |
| BUG-034-IPP-FFT-return-code-unchecked.md | 034 | エラーハンドリング / IPP |
| BUG-035-isLoading-stuck-on-generation-mismatch.md | 035 | 状態固着 |
| BUG-036-irL-irR-release-leak-on-init-failure.md | 036 | リーク |
| BUG-037-loader-thread-trash-bin-UAF.md | 037 | 競合 / UAF |
| BUG-038-spectrum-analyzer-6dB-error.md | 038 | 表示品質 |
| BUG-039-oversampler-buffer-overread.md | 039 | バッファ overread |
| BUG-040-nsl-playbacktime-1hz-fallback.md | 040 | サンプルレート |
| BUG-041-nsl-vla-stack-overflow.md | 041 | 数値 / VLA |
| BUG-042_cmaes_rule_of_five.md | 042 | Rule of Five |
| BUG-043_irconverter_samplerate_mislabel.md | 043 | サンプルレート / API |
| BUG-044_mklfftevaluator_rule_of_five.md | 044 | Rule of Five |
| BUG-045_irconverter_resample_failure_samplerate_mislabel.md | 045 | サンプルレート / データ破壊 |
| BUG-046_psychoacousticdither_rule_of_five.md | 046 | Rule of Five |
