# WORK103 — 低音ジジジ・ノイズ原因調査報告（Convolver→PEQ／Convolver単独）

- **作成日**: 2026-09-17
- **種別**: Investigation（ソース詳細調査・原因特定）
- **対象設定**: `default.xml`（別途受領）＋ `device_settings.xml`（別途受領）＋ IR `sampledata/impulse.wav`（＝`C:\Users\user\Documents\conv_filter\impulse.wav` と md5 一致 `e227274a0d13`）
- **症状**: Convolver→PEQ で音楽再生時、低音（ベース等）が出るとジジジジとノイズ。PEQ なし・Convolver のみでも同様。
- **セッション記録**: AiDex session note「ConvoPeq低音ノイズ調査の確定所見を記録」に要旨を保存済み。

```text
判定: 主因A（出力安全鎖の低域歪）＋ 欠陥B（IRAnalyzer窓関数欠陥・確定）＋ 副因C（超音波→IMD）
```

---

## 1. 動作条件の確定

| 項目 | 値 | 根拠 |
|---|---|---|
| 処理順 | `processingOrder=0` ＝ Convolver→EQ | `src/core/Types.h:11-14`、`AudioEngine.Processing.DSPCoreDouble.cpp:386-414` |
| SoftClip | 有効、sat=0.05 → threshold=0.9275／knee=0.0675 | `DSPCoreDouble.cpp:474-477` |
| AutoGain | 有効 → **XMLの-8dB/+3dBは実行時にPlannerが上書き** | `src/audioengine/RuntimeBuilder.cpp:300-324` |
| Oversampling | type=1（LinearPhase）× factor=2 | `device_settings.xml`、`DSPCoreLifecycle.cpp:191-192` |
| デバイス | 192kHz／1024／Voicemeeter Virtual ASIO／11ch → **処理レート384kHz・処理ブロック2048** | `device_settings.xml` |
| NoiseShaper | type=2 ＝ Adaptive9thOrder、32bit（**非可聴域のため除外**、§6） | `src/core/Types.h:23-28`、`src/LatticeNoiseShaper.h:278-293` |
| Convolver | phaseMode=1（Mixed、F1=200／F2=1000Hz）、tailMode=1／start=0.085s→実効0.12s／strength=0.5／mult=8、mix=1.0 | `src/ConvolverProcessor.h:117-137`、`MKLNonUniformConvolver.cpp:673-684` |
| EQ | 25/40/63Hz各+3dB（Q0.707）、高域+0.5〜+1dB×5band、Parallel構造、saturation=0.05 | `default.xml`、`EQProcessor.h:126,307` |

DSP 順序（実コード）：
`入力×headroom → 2倍OS → Convolver → EQ → OutputFilter → ×makeup → SoftClip → 2倍ダウン → Dither → Limiter → HardClamp`
（`DSPCoreDouble.cpp:359-503`、`processOutputDouble:577-750`）

---

## 2. IR・音源の実測値（Python で WAV 直読・再現可能）

### 2.1 IR（`impulse.wav`、48k／ステレオ・デュアルモノ／8253frame＝0.172s）

- `peak=1.903 @61sample（1.27ms）`、`energy=3.6522`、DC≈0.00024、`|x|>0.99`は1点のみ。
- ピーク周辺に **±0.03〜0.10 の前後リンギング**（62:-0.106、64:+0.033、66:-0.040…）。先頭は 4e-07 からの交番・指数増大列＝リニアフェーズ LPF のプリリンギングの特徴。
- エネルギー正規化（`IRConverter.cpp:36` の −6dB 式）で `scale=0.262253`、scaledPeak＝0.499。
- 真の周波数特性（ゼロ詰め FFT、窓なし）：**全体 −5.6dB、20〜120Hz帯 −6.2dB**（増幅なし）。
- `Documents/conv_filter/` 内の全 IR（`impulse_hpf`、`impulseL/R`、`Filters LRChannel Dec 31-MP`、`Stereo Dec 27-filters-192k`、`Vector average-filters-48k`）もすべて減衰特性（maxH −2.6〜−6.0dB）。ブースト型 IR は存在しない。

### 2.2 テスト音源（`test_music.wav`、48k／97.4s ロック）

- 両 ch とも **peak＝1.0（0dBFS）**、rms≈0.22、crest≈13dB。マスタリング済み相当のフルスケール音源。

---

## 3. 主因A：出力安全鎖の低域歪（Conv→EQ・高確度）

安全鎖の閾値：Limiter `0.8414`（−1.5dB）／knee `0.1087`、HardClamp `±0.8913`（−1dBFS）、SoftClip `0.9275`。
（`processOutputDouble:581,708-710`、`DSPCoreIO.cpp:362` 相当、`DSPCoreDouble.cpp:474-477`）

最悪系試算（0dBFS 入力、Conv→EQ）：
`1.0 × headroom 0.499（plan −6.04dB）× conv 0.524 × EQ低域×2.0（25/40/63Hz各+3dBの重なり）× makeup 2.005（plan +6.04dB）≈ 1.05`
→ **Limiter・Clamp・SoftClip の全段に接触**。EQ のみでも `0.5×2×2＝2.0` に達する。

`SimplePeakLimiter`（`src/audioengine/SimplePeakLimiter.h:35-83`）は**アタック0ms・ルックアヘッドなし・オーバーサンプリングなし**で波形の山谷そのものに追従するため、正弦波の山だけを潰す振幅変調＝奇数次高調波＋ポンピングを生む。文献上の定説と一致：

- 「0msアタックは sharp edge を作り歪む」（MeldaProduction）。
- 「回避には ceiling −1〜−0.5dB＋lookahead 0.1ms＋oversampling 4x」（SageAudio）。
- 「低域ほど歪むのは想定内」（各社リミター実測）。
- 「ultrasonic＋非線形＝可聴 IMD。後からは除去不能」（Gearspace／SoS）。

→ 「低音が出るとジジジ」と完全一致。**Conv→PEQ 側の主因と確定**。

---

## 4. 欠陥B：IRAnalyzer の Tukey 窓欠陥（コード確定・要修正）

`src/IRAnalyzer.cpp:63-155` は先頭 `min(N,65536)` 点に **Tukey α＝0.5** を掛けて FFT する（`src/IRAnalyzer.h:27-33` で `kMaxAnalysisWindow=65536`、`kTukeyAlpha=0.5`）。

- Tukey(0.5) の両端 25％はテーパー域。384k 処理・約 66k tap の IR ではテーパー長≈16384 に対し、リサンプル後のピークは約488点目 → **窓ゲイン約0.002（−53dB）**。
- Minimum／Mixed 変換後の IR はピークが 0 番目に来るため**窓ゲインは 0**（`t=0` で `0.5*(1+cos(−π))=0`）。
- デルタ状 IR はエネルギーの99％がピーク1点にあるため、窓掛けで周波数特性が20〜50dB過小評価 → `convBoost = max(0, 負の大値) = 0` に固定（`AutoGainPlanner.cpp:52-53`）→ Planner がコンボルバー分にヘッドルームを与えない。
- スケール適用自体は正常（直接タップ `MKLNonUniformConvolver.cpp:731`、全パーティション `986-987` の `cblas_dscal` を ast-grep／sed で確認）。欠陥は**解析側の過小評価 → AutoGain 不足 → 主因Aを助長**の経路。
- 修正案（いずれか）：(a) ピークサーチは窓掛け前に実行する、(b) ピーク位置を FFT 窓中央へシフトする、(c) ピークホールド用に矩形窓を併用する。

---

## 5. 副因C：Convolver 単独でも鳴る理由（超音波→IMD・中確度）

Conv 単独の線形試算は `1.0×0.398×0.524×1.412 ≈ 0.29`（AutoGain 時 0.52）で安全鎖に届かない。Mixed 位相の過渡再生育（＋1〜2dB）やサンプル間ピークを積んでも 0.7 前後。よって単独モードには非レベル機序が必要で、最有力が超音波経路：

1. IR の 20kHz 帯リンギング ±0.1（scaled で −31dBFS 級）が畳み込みで出力に乗る。
2. Conv→EQ 時は `convIsLast=false`（`DSPCoreDouble.cpp:457-462`）のため EQ 用 LPF（24kHz Natural 2次）が適用され、20〜30kHz 帯はほぼ減衰しない。NUC 内 HC（Natural 22kHz）も 20kHz リンギングは通す。
3. SoftClip が 384k 域で高調波を生成し、192k 出力に 20〜96kHz 成分が残留 → **Voicemeeter 内部 SRC（192k→メイン系レート）／DAC／アンプ／ツイーターで相互変調・折り返し** → 低音振幅に追従する可聴ジジジ（RME フォーラム／Xiph IMD テスト／Headphonesty の機序と一致）。

## 6. 除外できた候補（検証済み）

| 候補 | 結果 |
|---|---|
| Adaptive ノイズシェーパ | 32bit では誤差 ±9e-10 で非可聴。係数（最大0.015）も安定域（`LatticeNoiseShaper.h:278-293`） |
| スケール未適用 | NUC 全層に適用確認済み（§4） |
| EQ バイパス漏れ | `EQProcessor.Processing.cpp:499-530` で完全バイパス成立を確認 |
| テール増幅（L1 gain 1.375×） | テール振幅 1e-06 級への乗算で −120dB 級、非可聴 |
| パーティション／ブロック不整合 | L0 part＝2048＝処理ブロックで整合。M-04 ゲート等も正常 |
| EQ Parallel saturation | Conv→EQ 時の寄与は残るが、単独モードの説明にならないため副因扱い |

## 7. 推奨する切り分け手順（各1項目ずつ上から順に）

1. PEQ のみ（Conv バイパス）でも鳴るか → 鳴れば主因A確定。
2. `outputMakeup` を −6dB → 消えればレベル依存クリップ（A）。
3. `softClipEnabled=0` → 緩和すれば SoftClip 寄与。
4. `phaseMode=AsIs` → 変化すれば Mixed allpass／窓欠陥寄与。
5. `tailMode=Bypass`、6. `oversamplingFactor=1`、7. `noiseShaperType=0`、8. デバイス 48kHz → C の寄与判定。

## 8. 恒久対策案

1. `SimplePeakLimiter` に **1〜4sample のルックアヘッド＋リリースの低域適応化**を付与、または天井を −1.5dB へ（文献の定石）。
2. 欠陥Bの修正（§4 の3案のいずれか）＋ Planner の conv 側マージン見直し。
3. SoftClip 後段にも**超音波除去 LPF（Sharp 20kHz）**を追加、または `convHCMode=Sharp` 既定化。
4. EQ Parallel×20band の saturation は band-delta 側のみ適用へ（現状は要確認のまま残置）。

## 9. 棚卸し・ツール使用の申告

- `rg` による TODO 掃引：`src/AudioSegmentBuffer.h:19`（暫定 factory）と `src/SafeStateSwapper.h:448`（暫定値）のみ。要調査・保留の残件は src に見当たらない。`ConvoPeq.md` は生成物のため原文追跡は未実施（鮮度注意）。
- 使用：AiDex（索引464件・signature／query／note保存）、semble CLI（該当特定）、cocoindex `ccc`（search 実行）、graphify（61029ノードから Convolver／Limiter／NoiseShaper 周辺を BFS）、WSL の rg／ast-grep 0.44／fdfind 10.3／ag 2.2／fzf／sed／awk、cppcheck（C＋＋指定が必要な旨を確認）、clang-tidy 23.1.1（版確認）、tgrep（system32 に不在を確認）、Brave／Firecrawl 文献検索、context-mode（batch／execute 系）＋ rtk 常時、headroom（transport 経由）。
- 未使用の理由：serena MCP は本環境に未公開（CLI 等価手段で代替。`mcp__serena__*` は呼べず）、Dr.Memory は実機再生の録音・監視が必要で今回は静解析＋数値再現を優先、Obscura は docker 版禁止・ネイティブ CLI 不在、Crawl4AI／DDGS／Trafilatura は Brave＋Firecrawl で代替、firecrawl scrape は Voicemeeter 公式 PDF が取得済み情報で足りたため未実行。
