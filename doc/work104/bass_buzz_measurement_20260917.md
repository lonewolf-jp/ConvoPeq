# WORK104 — CLI自動計測拡張と低音ジジジ原因の実行時確定

- **作成日**: 2026-09-17
- **種別**: Implementation + Measurement（CLI自動テストモードの拡張＋自動計測による調査）
- **前工程**: WORK103（`doc/work103/`）、WORK103-R2
- **結論**:  buzzの主因は**レート不整合のコンボルバーが出すガベージ**（実行時確定）＋**常時作動のリミッター**（実行時確定）。B2も実行時再現。C（超音波）は測定法確定後に再測要。

```text
判定: D(新規・主因) RUNTIME-PROVEN / A(常時化) RUNTIME-PROVEN / B2 RUNTIME-PROVEN / C RETEST-REQUIRED
```

---

## 1. CLI拡張の内容（production無変更・test-only）

### 1.1 変更ファイル

| ファイル | 変更 | 内容 |
|---|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **新規** | 測定本体。`int runBassBuzzMeasurement(int argc, char* argv[])` |
| `src/tests/AudioEngineHarness/AudioEngineHarness.h/.cpp` | +tap seam | `HarnessTapFn`（入力充填前／処理後フック）＋`setTap/clearTap`。production無変更 |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +5行 | `--buzz`／`--buzz-*`のdispatch |
| `CMakeLists.txt` | +1行 | `AudioEngineHarness` exeにTU追加 |

### 1.2 使い方

```text
AudioEngineHarness.exe --buzz [--buzz-quick] [--buzz-sr=192000] [--buzz-block=1024]
  [--buzz-ir=<path>] [--buzz-out=<csv>] [--buzz-dur=<sec>] [--buzz-rigcheck[=bare|ir|eq]]
```

- 信号: sine50／sine40／sine100（0dBFS）／kick（55Hz減衰＋3kHzクリック）／multi（50/100/1k/10k/16k等振幅）。
- 設定行列: C0 bypass-all／C1 conv-only／C2 eq-only／C3 conv+eq（default.xml等価）／C4 makeup-6dB／C5 softclip-off／C8 shaper-psycho／C6 phase-AsIs。
- 指標: in/outPeak・RMS・crest・flatTop（|x|≥0.89＝HardClamp代理）・limitZone（|x|≥0.8414＝Limiter代理）・THD（最小二乗正弦fit残差）・ultra比（24k-96k／20Hz-20k、juce::dsp::FFT）・DC・drift（ブロックピーク前後比較＝遷移検出）。
- ゲート: 自己診断（C0 bypass透過性 peak≈in×0.891・THD<-80dB）不合格で即FAIL／world-publish追跡（commit-seq前進要求、外れ値はSKIP行化）。
- 特殊実験: D（OS factor 2→4、IR再ロードなし）、B2（+9dB@50Hz合成ブーストIRのreadback）。

## 2. 実行時確定した事実（192kHz／1024／OS×2／Mixed／impulse.wav）

### 2.1 主因D：コンボルバー出力そのものがガベージ（RUNTIME-PROVEN）

C0がcleanなのに対し、**conv活性なC1／C3／C4はbit一致の同一ガベージ**（world seq 10→12→13→15すべてcommit済み、drift±0.2dB＝定常）：

| row | outPeak | THD | ultra | 設計値 |
|---|---|---|---|---|
| C0 sine50 | 0.6577 | -150.9dB | -138.7dB（floor） | bypass相当・clean |
| C1 sine50 | **0.0474** | **-16.5dB** | **-56.4dB（+82dB浮上）** | ~0.45・clean |
| C3 sine50 | **0.0474** | **-16.5dB** | **-56.4dB** | ~0.7・limiter微接触 |
| C1 kick | 0.0397 | — | -50.3dB | — |
| C1 multi | 0.0817 | — | -69.4dB | — |

- 設計出力（~0.45）の1/10のレベル、-16.5dBの高調波歪、+82dBの超音波ハッシュ。いかなる線形畳み込みでも説明不能。
- ログ裏付け：`transferIRStateFrom: ch=2 len=31457 sr=192000.0`（5回）。DSP処理レート384kに対し**ホストレート192kのIRが循環**。
  31457 = 8253×4（48k→192k）から末尾trimmed。UI convolverが`PrepareToPlay.cpp:325`でホストレートprepareされるため、そのLoaderThreadは常に192kでIRをbuildする。
- D実験（OS 2→4、IR再ロードなし）：outPeak 0.0474→**0.0312**、他指標も連動変化。同一IRのままruntime quantumだけ変えるとガベージの性格が変わる＝**レート／ブロック不整合に無防備（無警告・no-guard）**。
- 知覚対応：歪＋超音波ハッシュは入力振幅に追従するため、大音量の低音でのみジジジとして顕在化。PEQのみ／bypassがcleanなことと整合（convが経路に入る条件でのみ発症）。

### 2.2 主因Aの常時化：リミッターはbypassでも0dBFSに噛む（RUNTIME-PROVEN）

- `--buzz-rigcheck`（bypass、0dBFS sine50）：outPeak **0.8414に完全一致**（kPLThreshold = 0.8413951287507587）、THD -56.2dB。
- 機序：dither headroom 0.891 > limiter threshold 0.8414 のため、マスター済み音源の全ピークが常時リミッティングされる。lookaheadなし・即時アタック・100msリリース。
- Conv+EQ時は低域+5.19dBが加算され食い込みが深化（C3のlimカウンタ等で定量化可能）。低域ほど可聴な歪（文献）＝「低音でジジジ」。

### 2.3 B2の実行時再現（RUNTIME-PROVEN）

- +9dB@50Hz合成ブーストIR（RBJ peakingのインパルス応答、peak正規化0.95、float32 WAV自作ライタで保存）をロード→`getIrFreqPeakGainDb()` readback **0.00dB**。
- PlannerはconvBoost=0のまま。ブースト型IR（room-null補償の定番）でheadroom不足→確定的クリップの潜伏欠陥。

### 2.4 Planner実測値＝手計算と一致（レベルモデル検証）

- `[AUTO_GAIN_PLAN] eqMaxGainDb=4.99 eqMaxQ=0.7070 irFreqPeakGainDb=0.00 → input -4.99dB / makeup +4.99dB`。WORK103-R2のRBJ厳密値（+5.19dB@50Hz等）と整合。

### 2.5 C（超音波）は再測要

- 自作FFT指標の配置バグ（インターリーブ充填＝零詰め2倍と等価で鏡像を生成）を発見・修正。修正後はfloor -138dB級で正常動作。
- 旧測定のultra値（+0.1dB等）はすべて無効。C1の-56.4dBは修正後の正規測定（ガベージ由来の実ハッシュ）。

## 3. 因果チェーン（確定版）

1. UI convolverがホストレート（192k）でIRをbuild（`PrepareToPlay.cpp:325`）。［コード確定＋transferログ］
2. DSP world（384k）がそのIRStateをtransferで受領。DSP側再構築が追いつかない／M-1再利用されると誤レートIRのまま発音。（transferログ5回＋C1ガベージ＋D-shift）
3. 誤レート／誤ブロックのパーティション畳み込みが歪＋超音波ハッシュを出力（0.0474／-16.5dB／+82dB）。［実行時測定］
4. 低音の大振幅がこのハッシュを振幅変調 → ジジジとして可聴。（知覚対応：C0 clean／conv活性のみ発症）
5. さらに出力安全鎖（limiter 0.8414＋clamp 0.891）が0dBFS級の全ピークに常時接触し、低域ブーストで深化。（rigcheck 0.8414完全一致）
6. B2（感度0のconvBoost報告）がブースト型IRのheadroomを奪い、5を悪化させる latent 経路。

## 4. 修正案（優先順・ファイル指定）

- **F1（最優先）**: `AudioEngine.Processing.PrepareToPlay.cpp:325` — UI convolverをprocessing rateでprepareする（またはLoaderThreadの目標レートをDSP処理レートに一本化）。IRStateにレートを刻印し、消費側で照合する。
- **F2**: `MKLNonUniformConvolver::SetImpulse`／`StereoConvolver::init` — `knownBlockSize`とruntime quantumの不一致をhard-guard（M-04式のdeterministic containment＋loud log）。無警告ガベージの根絶。
- **F3**: `RuntimeBuilder::build` のtransfer後に`rebuildAllIRsSynchronous`（DSP自身のレートで）を必須化し、完了までcommitを保留（`validateWarmup`の実効化）。OS factor変更時はIR再構築を必須化（Dの無警告遷移を塞ぐ）。
- **F4**: リミッターにlookahead（1〜4sample）＋天井運用の見直し（bypass時0dBFS接触の解消）。短期回避：運用でoutputMakeupを下げる／-1.5dBFS以下で使う。
- **F5**: B2 — `PendingCommit`にirFreqPeakGainDbを追加し`updateIRState`へ伝搬（NUC経路）。`IRAnalyzer`は窓掛け前ピークサーチへ。
- **F6**: 短期回避（ユーザー向け）：convolver使用時はdevice 48kHz＋OS×1〜×2でレート差を縮小、makeupを-3〜-6dB、softClipは現状維持。

## 5. 方法論・副産物・残課題

- 自作FFTバグ（§2.5）は本タスク内で発見・修正・再検証済み。教訓としてrow出力＋CSVに`driftDb`（遷移検出）を追加済み。
- 測定実行は`evidence/*.json`にtelemetry副作用を残すため、実行後に`git checkout -- evidence/`で復元した（本報告時点clean）。再実行時は無視してよい。
- LSP診断は生成JuiceHeader欠如下では大量の誤検出を出す（`waitUntil`ラムダ等、既存正常コードにも誤検出）。実ビルドを正とした。
- `--buzz-out` CSVが生成されない事例あり（stderr行には全row出力済みのためデータ欠落なし）。軽微rig不具合として残置。
- 残課題：(a)C0 bypassの0.6577（-3.6dB）の exact 内訳は未確定（clean線形のため結論に影響なし）、(b)フル行列のC2／C5／C6／C8、(c)PEQ-only試聴対応、(d)Dr.Memory動的検査（build/にReleaseあり）。
- 文献：asj-freshはリンク集のため直接収穫なし。代わりにtonalux（lookahead数学・intersample peak数dB）、tonestack（THD可聴閾値の3レベル分析枠組み）、Audioholics／TOYO（低域歪・IMD和差周波数）で裏付け。AES／音響学会誌は本件の直接証拠としては未使用。

## 6. ツール使用記録（本タスク分）

- headroom proxy（transport）＋context-mode（batch/execute系・並列3〜4）＋rtk常時。serena MCPはHTTP自営（:8123）で使用実績あり（本タスクではAiDex／rgを主用）。
- WSL: rg／ast-grep 0.44／fdfind 10.3／ag 2.2／fzf／sed／awk。cocoindex ccc、graphify（61029ノード）、semble、AiDex（query／note）。
- cppcheck（C++指定・機能欠陥0）、clang-tidy（対象TU・警告0）、tgrep（System32実在・search実行）、Obscura（fetch実行）、DDGS＋trafilatura（tonalux 6819字・tonestack 17696字抽出）、context7（JUCE resolve＋query、寄与なし）、firecrawl（web＋developer検索）、brave（低域歪・THD閾値）。
- Crawl4AIはimport確認のみ（単頁抽出に過剰）。Dr.MemoryはASIO実機再生を要し未実行（バイナリはbuild/Releaseに存在）。
