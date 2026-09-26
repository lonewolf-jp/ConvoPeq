# P1-5-IR-P2 — Step 5-B / P3-0: Attribution Protocol Freeze（read-only source audit）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-0）
- **性質**: **完全 read-only**。source / CMake / production / settle / sleep = **0 変更**。
  追加実測（A3 2・3 回目、P2 probe 再実行、`--buzz-os`、settle 変更）= **0**。
- **目的**: `+1.4759 dB` が DSP chain のどの段階で生成されるかの切り分け**設計**のため、
  最新 source に基づく boundary の確定と hypothesis 候補の限定を行う（原因の断定はしない）。

---

## P3-0-1 State Freeze（PASS）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e3` |
| `kP15FullMatrix` | `true`（P3-0 では維持） |
| A3 binary SHA-256[:16] | `b2c39a9a0e3b1fe3` |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `BOOL=OFF` |
| production source / CMake / JUCE / default / calibration diff | **0** |
| settle / sleep diff | **0** |
| measurement semantics diff | **0** |

```text
A3 = REPRODUCED / A4 = PASS / P3 = NOT STARTED / H-A = CLOSED / H-B = OPEN
1.4759 dB = REPRODUCED MEASUREMENT ANOMALY / mechanism = UNKNOWN
```

---

## P3-0-2 gainDb 生成点からの逆 tracing（確定済み境界）

```text
input amplitude            v = amp * sin(phase)          makeTap :161
   ↓                       amp = 10^(ampDb/20)           runCase :420
AudioEngine output         512-sample blocks @ base SR   makeTap :192-194
   ↓
Convolver / oversampling   DSPCore::processBlockDouble   DSPCoreDouble :318-
   ↓
capture output             cap.outL（12 blocks = 6144）   makeTap :194
   ↓
dftMag                     Hann 単一ビン DFT              :200-214
   ↓
dftDb = 20*log10(dftMag)                                  :459
   ↓
gainDb = dftDb − ampDb                                    :460
```

---

## P3-0-3 `dftMag` は原因候補から除外できるか

| 確認項目 | 結果 | 根拠 |
| --- | --- | --- |
| 1. 入力波形を変更していないか | **変更しない** | 引数は `const std::vector<float>& x`（:200）。読み取りのみ（`x[from+i]`）で書き込みなし |
| 2. 解析式そのものから差が発生する余地 | **余地なし（level 完全線形）** | `mag = 2*sqrt(re²+im²)/wsum` は入力振幅に厳密比例。線形 chain なら `dftDb` は `20log10(k)` だけ移動し、`gainDb = dftDb − ampDb` で `ampDb = 20log10(amp)` と厳密に相殺 → **線形 chain では gainDb は level 非依存** |
| 3. historical と A3 で同一解析コードか | **同一** | A3 = historical ソース + 1 token（`kP15FullMatrix`）。`p15ir` 18 行が bit-identical |

```text
measurement formula = NOT PRIMARY DIFFERENTIATOR
```

- 副次的な帰結（source-level の数学的性質）: **観測された Δgain ≠ 0 は、
  2 行の間で chain が level 線形でないこと（または 2 行の内部状態が異なること）を意味する**。
  これは measurement 側ではなく chain 側の性質である。

---

## P3-0-4 `gainDb` の入力側 tracing

| 項目 | source 実読値 | 両行での扱い |
| --- | --- | --- |
| A. amplitude generation | `amp = std::pow(10.0, ampDb / 20.0)`（:420） | 両行で `ampDb` のみ相違（−20 / −6）。`amp` と `ampDb` は厳密整合 |
| B. sine generation | `v = c.amp * sin(c.phase)`／`phase += 2π·freq/kSr`（:161-163）・`freq = 1000.0` | 両行同一。位相は `configureCapture` で 0 に reset（:131） |
| C. capture window | `kWarmBlocks=4 + kCapBlocks=12` blocks（:425）・解析窓 `kAnalysisN=4096`（末尾・:439） | 両行同一 |
| D. `ampDb` subtraction | `gainDb = dftDb − ampDb`（:460） | 両行同一（`ampDb` は −20 / −6） |

```text
input definition = NOT A DIFFERENTIATOR（H2 → CONTRADICTED）
```

- 周波数基準 `kSr = 48000.0` は base SR と一致（engine は `h.start(48000, 512)`）。

---

## P3-0-5 Convolver / oversampling boundary（本命調査対象）

### ① OS=1 は本当に unity-rate path か

| 確認項目 | source 実読結果 |
| --- | --- |
| `processingRate` の決定 | `processingRate = newSampleRate * static_cast<double>(oversamplingFactor)`（`DSPCoreLifecycle.cpp:191`・`:263`）。os=1 → 48000（ログの `processingRate=192000` 等と整合） |
| oversampler が bypass されるか | **bypass される**。`if (oversamplingFactor > 1) { processBlock = oversampling.processUp(...); }`（`DSPCoreDouble.cpp:359-361`）および `if (oversamplingFactor > 1) { oversampling.processDown(...); }`（`:539-541`）。**os=1 では processUp/processDown は呼ばれない** |
| 1-stage processing が残るか | **OS=1 固有の分岐が存在する**。`if (state.softClipEnabled)` 内で `oversamplingFactor > 1` なら upsampled 領域の softClip、**else（= OS=1）なら「局所2倍OS: processUp → SoftClip → processDown」**（`:491-514`）を実行 |
| gain normalization が存在するか | `processUp`/`processDown` が呼ばれないため、その内部正規化も os=1 では適用されない。**「OS1 だから gain 1.0」という推論は source 上では成立しない**（ガードにより分岐が異なる） |

- **重要（sc0 との関係）**: 参照行の `gainDb_sc0` は `softClipEnabled = false`（`runPair` が `runPair → runCase(..., false, ...)`）。
  sc0 では softClip ブロック自体が**実行されない**ため、上記の局所 2×OS 経路は sc0 では通らない。
  sc1 のみが局所 2×OS 経路を通る。

### ② IR `F_scale` の生成と適用

**生成**（`IRConverter::computeScaleFactor` :175-196）:

```text
第1段: scale = computeEnergyScale(ir)
       = (1 / sqrt(maxChannelEnergy)) * 0.5011872336272722   (safetyMargin = −6 dB, :36)
第2段: analysis = analyzeIR(ir, scale)     → peakValue / rmsValue / frequencyPeakGain
第3段: applyClampProtection(result, scale, analysis, ...)
       - peakClamp  if peakValue*scale > kMaxEffectivePeak = 0.5      (:94-100)
       - rmsClamp   if rmsValue*scale  > kMaxEffectiveRms  = 0.25     (:104-109)
       - freqClip   if frequencyPeakGain > kMaxEffectiveFreqResponse = 1.41 (+3 dB)  (:113-118)
```

**適用**（`IRConverter::convertFile` :346）: `scaledIR.applyGain(prepared->scaleFactor);`

**その後さらに gain が掛かる経路**:

| 経路 | 値 | level 依存性 |
| --- | --- | --- |
| `state.inputHeadroomGain`（`processInputDouble` :334） | `configureChain` が `setInputHeadroomDb(0)` | 静的（level 非依存） |
| `state.convolverInputTrimGain`（`scaleBlockFallback` :451） | `setConvolverInputTrimDb(0)` | 静的 |
| `state.outputMakeupGain`（`scaleBlockFallback` :480） | `setOutputMakeupDb(0)` | 静的 |
| `LoudnessMeter::processBlock`（:713） | 引数は `const double*`（`LoudnessMeter.h:37`） | **音声を変更しない（meter のみ）** |
| **`SimplePeakLimiter::processBlock`（:722）** | `kPLThreshold = 0.8413951287507587`・`kPLKnee = 0.108748` | **level 依存・無条件適用** |

- `F_scale` は**両参照行で完全同一**（0.50013092 / 0.50026187）であり、`[L0_WRITE] slot=11 peak=0.500000` も両行一致。
- **注意（指示どおり）**: 「F_scale が一致している ≠ IR gain mechanism が正常」。一致は
  「Δ の differentiator ではない」ことのみを示す。

### ③ `n=0` の意味

```cpp
const int nStages = (p.os == 1) ? 0 : ((p.os == 2) ? 1 : ((p.os == 4) ? 2 : 3));   // runPair :669
```

- **`n` = oversampling stage 数 = log2(os)**。`n=0` ⇔ `os=1`（0 段 = OS なし）。
  推測ではなく source の定義（:669。同型の定義が :490・:600 にもある）。
- `n` は phase / partition / variant ではない。

### ④ 追加発見: `SimplePeakLimiter` の metric blind spot と未 reset 状態

| 発見 | 内容 | 根拠 |
| --- | --- | --- |
| **knee 開始点が metric 閾値より低い** | `clipStart = kPLThreshold − kPLKnee*0.5 = 0.8413951287507587 − 0.054374 = 0.7870210…`。gain reduction は peak > 0.7870 で始まるが、vehicle の `limitingEngaged` は `>= kLimThreshold = 0.8413951287507587` を数える | `SimplePeakLimiter.h:41,56-70`・`DSPCoreDouble.cpp:720-722`・`P1PolyphaseGainCharacterization.cpp:442` |
| **`limitingEngaged=0` は limiter 非動作を証明しない** | 0.7870〜0.8414 の knee 帯で gain reduction が起きても metric は 0 のまま | 同上 |
| **`DSPCore::reset()` に `peakLimiter.reset()` が無い** | reset 対象は `convolverState` / `eqState` / `dcBlockers` / `dither` / `fixedNoiseShaper` / `adaptiveNoiseShaper` / `oversampling` / `outputFilter` / `activeAdaptiveCoeff*` / バッファ / `ramps()` / `histories()` のみ。**`peakLimiter` は含まれない** | `DSPCoreLifecycle.cpp:381-406` |
| **`prepare()` も envelope を reset しない** | `prepare()` は `releaseCoeff` のみ設定。`reset()` が `envelope = 1.0` を設定する（呼ばれていない） | `SimplePeakLimiter.h:19-24`・`:27-30` |
| **帰結** | limiter の `envelope` は **DSPCore インスタンスの寿命中、ケース・ブロックを跨いで持続する**（同一 DSPCore が維持される限り） | 同上 |

---

## P3-0-6 `os8 → os1` transition の source trace

| 段階 | source 実読 | 備考 |
| --- | --- | --- |
| OS 変更の入口 | `AudioEngine::setOversamplingFactor(int factor)`（`Parameters.cpp:538`） | `0/1/2/4/8` のみ受理。`manualOversamplingFactor` と `m_currentOversamplingFactor` を更新 |
| 変化時のみ実行 | `if (consumeAtomic(manualOversamplingFactor) != newFactor)`（:547） | 同一値なら**何もしない**（= am-20→am-6 では再 prepare なし） |
| UI convolver の形状追従 | `reprepareUiConvolverForProcessingGeometry(...)`（:554） | コメント（:551-556）: 「追従なしでは以降の IR ロードが旧 processing 形状で build され、DSP world（新形状）との mismatch → **publish 拒否の連鎖**になる」= **既知の hazard が source に明記されている** |
| intent 1 | `submitRebuildIntent(Structural, EnqueueSnapshotCommand, Snapshot, Replaceable)`（:557） | bulk restore 中でも投入される |
| intent 2（抑制条件） | `if (!m_isRestoringState && sr > 0.0) submitRebuildIntent(Structural, RequestRebuildKindEntry, Structural, Replaceable)`（:559-565） | `configureChain` は `beginBulkParameterRestore()`〜`endBulkParameterRestore(true)` で囲むため、**この行は抑制**される |
| bulk restore | `beginBulkParameterRestore()` → `m_isRestoringState = true`（:203-206）／`endBulkParameterRestore(true)` → `false` + `RequestRebuildKindEntry` Structural intent（:208-221） | 結果として **OS 変更 1 回につき Structural intent が 2 本**投入される |
| convolver reprepare | `convolverState->prepare(owner, processingRate, processingBlockSize)`（`DSPCoreLifecycle.cpp:196`・`:267`） | processingRate = sr × os |
| IR state transfer | `[CONV_IR] transferIRStateFrom`（`ConvolverProcessor.LoaderThread.cpp`） | 参照ログでは placeholder world で `no IR data to transfer`、IR world で `IR transferred ch=2 len=1 sr=48000.0 block=1024 gen=N` |
| publication | `[PUBLISH] seq=N gen=N` | |
| capture | `runCase` の capture | |

**transition topology（両ログ一致・A4 で確定済み）**:

```text
p15eq id=b9_am0 os=8   (L132, 両ログ)
        ↓  setOversamplingFactor(1) → structural intent ×2
p15ir id=g0_os1_am-20 os=1   (L235, 両ログ)  ← 遷移直後の初ケース
        ↓
p15ir id=g0_os1_am-6  os=1   (L356, 両ログ)  ← OS 変更なし（setOversamplingFactor は no-op）
```

- **「os8 → os1 遷移が 1.4759 dB を発生させた」とは結論しない**（A4 の規律を維持）。
  本項で確定したのは transition の source 経路と topology の一致まで。

---

## P3-0-7 既存 `--buzz` 結果との照合（candidate differentiator の一覧化のみ）

| Vehicle | SR | OS | IR | settle | Δgain |
| --- | ---: | -: | --- | --- | ---: |
| historical / A3 `--p1-char` | 48k | 1 | g0 / 48k（`resampled=no`） | legacy vehicle（`waitBacklogZero` + `waitWorldPublished` + `sleepPump(800)`） | **+1.4759 dB** |
| P2 probe（Step 4） | 192k | 1 | g0 / 192k（`resampled=yes`） | 150 s quiet + `waitIrFinalized` + waits | **≈ +0.00090 dB** |

- **SR または settle が原因とは結論しない。** 本表は differentiator の列挙のみ。
- 両 vehicle で共通なのは「OS=1・softClip off・g0 IR」であり、相違は **base SR（48k/192k）**、
  **settle 構造（800 ms / 150 s + finalize wait）**、**IR geometry（irLen 48000 / 96000）**、
  **sc 構成（sc0+sc1 2 パス / sc0 のみ）**、**解析定義（`dftMag`+`gainDb` / `[PROBE]` outPeak）**。

---

## P3-0-8 attribution hypothesis（候補の限定・順位付けなし）

| H | 内容 | 分類 | source 根拠 |
| --- | --- | --- | --- |
| **H1** | measurement / DFT normalization | **CONTRADICTED** | `dftMag` は読み取り専用（`const&`）・式は level 完全線形・`gainDb = dftDb − ampDb` が level を厳密相殺・historical/A3 同一コード（P3-0-3） |
| **H2** | input amplitude definition | **CONTRADICTED** | `amp = 10^(ampDb/20)` が `ampDb` 減算と厳密整合・正弦位相基準は `kSr`=base SR・両行同一（P3-0-4） |
| **H3** | IR `F_scale` / IR preparation | **CONTRADICTED**（Δ の differentiator として） | F_scale が両行で完全同一（0.50013092 / 0.50026187）・L0 peak も両行 0.500000。**両行で同一の量は行間差を生成できない**（P3-0-5②） |
| **H4** | OS1 processing gain | **CONTRADICTED**（Δ の differentiator として） | 両行とも os=1。`oversamplingFactor > 1` ガードは両行同一で `processUp`/`processDown` は両行とも未実行（P3-0-5①） |
| **H5** | OS8→OS1 state transition | **SUPPORTED** | am-20 の sc0 が遷移直後の初ケース（L132→L235 両ログ一致）・`setOversamplingFactor` が geometry mismatch hazard を明記し structural intent を 2 本投入・UI convolver 形状追従が必須とコメント（P3-0-6） |
| **H6** | convolver output scaling | **SUPPORTED** | `SimplePeakLimiter` が**無条件**で毎サンプルに乗算（level 依存）・knee 開始 0.7870 が metric 閾値 0.8414 より低い（blind spot）・`DSPCore::reset()` に `peakLimiter.reset()` が無く envelope がケース間で持続（P3-0-5④） |
| **H7** | stale RuntimeWorld / publication timing | **SUPPORTED** | `runCase` が `waitWorldPublished` の**戻り値を無視**（:417）・settle は `sleepPump(800)` 固定・timeout でも capture へ進む経路が存在（P3-0-6） |
| **H8** | SR-dependent processing path | **CONTRADICTED**（Δ）／**SUPPORTED**（vehicle 間） | Δ については両行 base SR 48000 で同一。vehicle 間では 48k vs 192k の差が実在（P3-0-7） |
| **H9** | legacy p1-char vehicle-specific behavior | **SUPPORTED** | vehicle 固有構造: sc0/sc1 の 2 パス＋その間の IR 再ロード、os=1 かつ softClip 有効時の局所 2×OS、publish timeout 無視、800 ms 固定 settle（P3-0-5①/⑤・P3-0-6） |

- **順位付けは行わない**（指示どおり）。SUPPORTED は「source 上に level 依存／遷移依存の機構が
  実在し、候補として成立する」ことのみを意味し、原因であることは意味しない。

---

## P3-0 終了条件（boundary 表）

| Boundary | Source path | Evidence | Status |
| --- | --- | --- | --- |
| Input amplitude | `runCase` :420 `amp=10^(ampDb/20)` → `makeTap` :161 `v=c.amp*sin(phase)` | A4 source | **SUPPORTED**（Δ には非寄与） |
| DFT | `dftMag` :200-214 → `dftDb` :459 → `gainDb=dftDb−ampDb` :460 | A4 source | **SUPPORTED**（level 線形・NOT differentiator） |
| OS selection | `setOversamplingFactor` :538-567／`processingRate = sr*os` DSPCoreLifecycle :191,:263 | P3 source | **SUPPORTED** |
| OS1 processing | `if (oversamplingFactor > 1)` DSPCoreDouble :359,:539（skip）／softClip 局所2×OS :501-514 | P3 source | **SUPPORTED**（OS=1 固有分岐が実在） |
| IR `F_scale` | `computeScaleFactor` :175-196／`computeEnergyScale` :17-38／clamps :94-118 | P3 source | **SUPPORTED**（両行同一 → Δ には非寄与） |
| IR application | `scaledIR.applyGain(prepared->scaleFactor)` IRConverter :346 | P3 source | **SUPPORTED** |
| Convolver output gain | 静的 gain（headroom/trim/makeup = 0 dB）＋ `SimplePeakLimiter` :722（無条件・level 依存） | P3 source | **SUPPORTED**（limiter = level 依存候補・metric blind spot あり） |
| OS8→OS1 transition | `setOversamplingFactor` :547-565／`endBulkParameterRestore` :208-221／geometry hazard :551-556 | P3 source | **SUPPORTED**（intent 2 本・hazard 明記） |
| Publication | `waitWorldPublished` :86-103（戻り値は :417 で無視） | P3 source | **SUPPORTED**（settle 非保証経路が実在） |
| SR dependence | p1-char 48k vs P2 probe 192k（P3-0-7） | P3 / P2 evidence | **SUPPORTED**（vehicle 間）／**CONTRADICTED**（Δ） |

## P3-0 判定

```text
P3-0 = PASS
1.4759 dB = REPRODUCED
mechanism = UNKNOWN
```

- source-first の boundary 確定と hypothesis 候補の限定（H1〜H9 の 3 値分類）を完了。
- **H1/H2/H3/H4 は Δ の differentiator として source 上否定**され、**H5/H6/H7/H9 が候補として残存**。
  H6 については metric blind spot（`limitingEngaged` の閾値 0.8414 > knee 開始 0.7870）と
  `peakLimiter` の未 reset 状態という、従来未記録の source 事実を新規に確定した。
- **P3-1（最小 attribution experiment）は設計段階であり、本 P3-0 では実測を追加していない。**

## P3-0 禁止事項の遵守

```text
production source 変更 0 / P1PolyphaseGainCharacterization 変更 0 / kP15FullMatrix 変更 0
runCase 変更 0 / settle 変更 0 / sleepPump 変更 0 / 新 CLI flag 0
IR compensation 0 / gain compensation 0 / normalization 変更 0
P2 probe 再実行 0 / A3 2nd-3rd run 0 / H-B 修正 0 / polyphase correction ON 0
```

- `1.4759 dB` を「bug」とは呼称しない。正確な表現は
  **REPRODUCED MEASUREMENT ANOMALY / mechanism UNKNOWN**。
