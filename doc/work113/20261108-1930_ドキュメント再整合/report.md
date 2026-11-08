# 4 ドキュメントの実装整合・第 2 次（README / ARCHITECTURE / SOUND_PROCESSING / BUILD_GUIDE）

- 実施: 2026-11-08 18:10〜
- 対象: `main` = `810115ba`
- 目的: ルートの 4 ドキュメントを現在のソースコードに再度突き合わせ、1 回目で残った記述ずれを修正する。英語で記述する。`README.md` の "New in v0.6.11" 節は対象外。

## 1. 方法

1. 数値の総当たり監査を 2 種類用意した。`(N.N KB)`・`N lines` の主張を同一行のファイル名から解決して実測と比較するものと、バッククォートされた識別子 439 個をソース全体と照合するもの。
2. 読み取り専用の調査エージェント 2 本で、`ARCHITECTURE.md` §6〜§14 と `BUILD_GUIDE_WINDOWS.md` §5〜§20、および `SOUND_PROCESSING.md` の数値・構造・Dither・§8・§9・§11 を再監査した。
3. 監査の指摘 30 件は、すべて該当コードを読んで裏取りしてから適用した（そのまま採用した指摘はない）。
4. README は自分で数値・記述を確認した（前回の監査対象外の箇所を含む）。

## 2. README.md（10 件）

| 箇所 | 修正 |
|---|---|
| Aspect 表 | 行末の閉じパイプ欠落を修正 |
| Phase modes | `f1`/`f2`/`tau` → `mixedTransitionStartHz`（既定 200 Hz）/ `mixedTransitionEndHz`（既定 1 kHz）/ `tau` |
| Mixed の向き | 低域＝線形・高域＝最小位相 → **低域＝最小位相・高域＝線形位相**（`ConvolverProcessor.MixedPhase.cpp` で `wMinimum=1.0` が低域） |
| `MKLNonUniformConvolver` | 「legacy」→ 現行 NUC エンジン（`nucConvolvers[2]`） |
| Soft clip | `prevSampleInOut` を「フィードバック」と書いていたが、実際は書き込みのみ。局所 2× ハーフバンドの記述に置換 |
| 自動保存 | 実際の保存項目（shaper 種別、スレッド優先度、学習係数バンク等）を追記 |
| プリセット | EQ/Convolver 子状態（位相、`mixedF1Hz`/`mixedF2Hz`、IR 長、自動/手動）を明記し、OS 倍率はデバイス設定だと注記 |
| Quick Build | `app` オプションを追加し、`pgo-use` 行の桁ずれを修正 |

## 3. ARCHITECTURE.md（18 件）

| 箇所 | 修正 |
|---|---|
| §3.1 / §3.3 | `MKLNonUniformConvolver.cpp` を現行エンジンと明記（「legacy / backward compat」を削除） |
| §3.2 | `AudioEngine.h` を 5,176 → **5,777 行**に修正（サイズ 315.6 KB は実測どおり） |
| §4.1 | チェーン図の Output Filter を「最終段で分岐」→「conv 出力に HC/LC、EQ 出力に HPF/LPF」 |
| §4.2 | `AudioEngineProcessor::processBlock(double)` → `AudioEngine::processBlockDouble()`、`RuntimeWorldAuthority::consumeWorldHandle(readToken)`、`DSPCore::processDouble`、OutputFilter の 2 段呼び出し、processOutput の実順（DC → dither → メータ → リミッタ → 固定遅延）に修正 |
| §5.3 | TransitionPolicy は `RuntimePublicationOrchestrator` が選択し、`CrossfadeAuthority` は要否とフェード時間のみ |
| §6.2 | EQ サチュレーションは SSE2（`fastTanhV128Output`）で、AVX2 はゲインランプ/ステレオ側 |
| §6.3 | IR ロードは `LoaderThread`（juce::Thread）、Mixed の向き、現行エンジン名 |
| §7 | スレッド表を再構成。IR Loader / Rebuild / Worker を分離し、**CoordinatorLoop（1 ms）** を追加、Timer からアナライザ処理を削除（`SpectrumAnalyzerComponent` は自前タイマ 60/15/5 Hz）、DeferredFree は ConvolverState のみ |
| §7 | 遅延削除を `ISRRetireRouter` 経由（DSP/EQ/BandNode）と `DeferredFreeThread`（ConvolverState）に分離 |
| §9 | フェード時間は該当定数の **最大値**、IR 有無変化は 1–10 ms にクランプ。ミックスループは 9 引数シグネチャに修正し、`LinearRamp` は線形と明記。表に `m_irFadeTimeSec`（80 ms）を追加 |
| §9 | crossfade 時間の「typically 30–80 ms」→ 10–80 ms |
| §13 | `CONVO_CI_BUILD` は `HeadlessAudioPathVerification` を **スキップ**する（従来は「有効化」と誤記、CMakeLists の `if(NOT DEFINED ENV{CONVO_CI_BUILD})` と一致） |

## 4. SOUND_PROCESSING.md（20 件）

| 箇所 | 修正 |
|---|---|
| §1.2 | スタンドアロンでは float 経路がスタブで、`setDoublePrecisionProcessing(true)` により double 経路のみ実行されることを明記 |
| §5.2 | fastTanh の識別子を実装（`NumA`…`clipThreshold`）に修正し、スカラー版は閾値で分岐することを明記 |
| §6 共通 | 64 バイト整列は係数表のみ。Fixed 4th は AVX2 経路がなく SSE2 量子化のみ。診断は音声スレッドで加算 |
| §6.1 | VSL 不在・リング枯渇時のフォールバックは **xorshift64\***（`fallbackUniform`）。リング補充は timer（メッセージ）スレッド。整形誤差は `scale = 2⁻¹⁵` として −60 dBFS |
| §6.4 | 係数ハンドオフは `CoeffSet` 二重バッファ + 世代カウンタ（`LearnedState` はファイル保存専用）。CMA-ES はコーディネータ 1 本 + 評価ワーカー最大 6 本。格子は \|k\|≤0.85 クランプで安定 |
| §8.1/8.2 | LUFS 集計は未実装（リングに消費者なし）。TruePeak は 63/31 タップの 2 段。`peakHold` は atomic ではない |
| §9.4/9.5/9.6 | ロウドネス行を実態に修正。事前確保バッファを実名（`alignedL/R`、`dryBypassBufferDoubleL/R`、EQ の `scratchBuffer` 他）に修正。`RefCountedDeferred` は未使用と注記 |
| §12 | `EpochDomain.h` 32.2 → 35.6 KB |

## 5. BUILD_GUIDE_WINDOWS.md（3 件）

| 箇所 | 修正 |
|---|---|
| §6.2 | icx 用の build preset は存在しない（`debug`/`release` は `vs2026-x64` 束縛）ため、`cmake --build build-icx --config Release` に修正 |
| 付録 A | `AudioEngine.*.cpp` 32 → **31**、`ConvolverProcessor.*.cpp` 9 → **8 TU** |
| §16 E) | `mklink /J` の重複を統合（シンボリックリンクは `/D`） |

## 6. 検証

- 数値監査（KB・行数）を全 4 ドキュメントで再実行し、不一致 0 件。
- 識別子監査（439 個）で未発見は 7 件のみで、いずれも外部 API（`DftiMalloc`、`AudioIODeviceCallback`）か `Class::member` の修飾による誤検出。
- 表の列数（ヘッダーとの一致）とコードフェンスの偶数性を全 4 ドキュメントで確認（malformed 0）。
- `find-nise-kanji` で偽漢字なし、`0.6.10` の残存 0 件。
- 監査で挙がった 30 件は、`PsychoacousticDither.h`、`FixedNoiseShaper.h`、`LatticeNoiseShaper.h`、`TruePeakDetector.{h,cpp}`、`LoudnessMeter.h`、`ISRCoordinatorLoop.h`、`CrossfadeAuthority.{h,cpp}`、`DspNumericPolicy.h`、`CMakeLists.txt`、`CMakePresets.json` を読んで裏取りした。

## 7. 確認できなかった事項

- オーバーサンプラ各段がループ単位で AVX2 化されているか（関数の存在と全体構造は確認済み）。
- `ConvolverBuilder` が FFT Plan の唯一の生成者であることの網羅証明。
- Dither の RNG リングが実運用で枯渇する頻度（測定していない）。
- AMD CPU での icx バイナリ実行可否（環境がない）。

## 8. 変更ファイル

| 種別 | ファイル |
|---|---|
| 文書 | `README.md`、`ARCHITECTURE.md`、`SOUND_PROCESSING.md`、`BUILD_GUIDE_WINDOWS.md` |
| 報告 | 本ファイル |
