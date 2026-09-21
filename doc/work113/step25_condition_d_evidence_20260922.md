# Step 2.5 条件D エビデンスパッケージ（2026-09-22・read-only 監査）

- **工程**: Phase 0 freeze 確定（`phase0_characterization_report_20260922.md` §6）→ **本 doc（条件D 判断材料）** → Phase 1 eligibility
- **前提 BUILD-ID**: HEAD `88c6f00`（親 `85aa13b9`）/ production `src/**` の HEAD 差分 **0** / `CMakeLists.txt` HEAD 同一 / `production_flag: undefined/off` / `shadow_candidate: 0` / snapshot `ConvoPeq.md`（header 2026-09-22 06:20:11・`--check` FRESH・NEWER_SRC_COUNT=0）
- **本 doc は production 変更を一切含まない**: 案E（`centerValue *= 2.0`）・CMake flag 定義・default ON・calibration・F-3/F-4・O-20 は HOLD 継続
- **表記規約**: 各項を **SOURCE**（読み取ったコード/保存された実測の位置）・**OBSERVATION**（観測された数値・構造）・**INFERENCE**（そこから導かれる解釈・未検証）・**CONTRACT IMPACT**（条件D 判断への影響）に分離する。判断材料としての強度は OBSERVATION > INFERENCE。

---

## D-1 production 利得経路 census（up/down round trip の位置確定）

### D-1.1 SOURCE — double core の信号鎖（`oversamplingFactor > 1` の既定経路）

| # | 処理 | 位置 | 利得 | factor 依存 |
|---|------|------|------|------------|
| 1 | input headroom | `AudioEngine.Processing.DSPCoreDouble.cpp:334`（`state.inputHeadroomGain`） | −12〜0 dB（自動は −18〜0）・既定 0 or −6 dB | なし |
| 2 | **up（B-1 の第1出力点）** | `:361` `oversampling.processUp()` | **base 0.75^N / cand 1.0** | **あり（ここが B-1）** |
| 3 | DC blocker（OS レート） | `:371-375`（`dc.oversampledL/R`） | DC/超低域のみ | なし |
| 4 | Convolver（+ input trim `:446-453`・+ outputFilter HC/LC） | `:386-475` | 静的 | なし |
| 5 | EQ（`processingOrder` 依存） | `:386-475` | 静的 | なし |
| 6 | output makeup | `:477-481`（`state.outputMakeupGain`） | 0〜+12 dB・既定 0/+10/+12 dB | なし |
| 7 | SoftClip | `:483-515`（OS>1 は up ドメイン・OS==1 は local 2x OS `softClipOS` `:505-513`・31/90 単段） | 非線形 | 「どちらの支路か」のみ |
| 8 | bypass dry/wet blend | `:517-537` / `:544-559`（dry は `:351-355` で **pre-up 採取**） | fade | なし |
| 9 | **down（B-1 の第2出力点）** | `:539-542` `oversampling.processDown()` | **base 0.75^N / cand 1.0** | **あり** |
| 10 | finite guard → TruePeak/LUFS → peakLimiter（θ=0.8413951287507587・knee=0.108748）→ hard clamp ±`kOutputHeadroom`（0.8912509381337456 = −1.0 dBFS）→ fixed latency delay | `:684-751` | 静的 | なし |

- 参考（単一ソース）: float 経路も同一構造（`AudioEngine.Processing.DSPCoreFloat.cpp:235/263/382/405-419`）。
- 参考（local SoftClip OS の構築）: `AudioEngine.Processing.DSPCoreLifecycle.cpp:188/261` `softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock)`。
- 参考（factor 決定）: `OversamplingPolicy.h:42-49, 79-82`（Auto は `maxAllowedFactor`）→ `DSPCoreLifecycle.cpp:112-130`（`oversamplingFactor = 1 << factorLog2`）。既定値: `AudioEngine.h:2615` `manualOversamplingFactor { 0 }`（Auto）・`DeviceSettings.cpp:1217/1287`（永続既定も 0 = Auto）。

### D-1.2 OBSERVATION

- 利得を持つ段は 1・2・4(trim)・6・7・9・10 の 7 箇所。**`oversamplingFactor` に依存する利得は 2（up）と 9（down）のみ**で、他は factor 非依存の静的値。
- makeup（#6）は **up と down の内側**にあり、down の後段は factor 非依存の kOutputHeadroom と limiter/hard clamp のみ。
- **local SoftClip OS（OS==1 かつ SoftClip 有効）も同じ 31/90 単段 design** であるため、この経路にも base の 0.75 loss（1 段 = −2.4988 dB）が乗る。
- bypass の dry は `:351-355` で **pre-up**（input headroom 後・up 前）に採取される。

### D-1.3 INFERENCE

- base の出力は factor を 2 倍にするごとに **−2.4988 dB** 下がる（線形 0.75^N / dB −2.498775・−4.997549・−7.496324 for N=1/2/3）。この差は factor 非依存の静的 gain では打ち消せない（D-2/D-4 で補償の不在を確認）。
- **Auto 既定（`manualOversamplingFactor = 0`）は `maxAllowedFactor` を返すため、≤96 kHz のホストで 8x（3 stage）→ 既定構成が最大の −7.4963 dB** になる。2x 明示で −2.4988 dB、1x で 0 dB（OS 経路そのものが skip される）。
- bypass 切替時は dry（pre-up）へ fade するため、base では **up/down の loss 分が切替段差に寄与**する（headroom/makeup 由来の段差は別途、既存挙動として残る）。
- engine では up 直後に DC blocker（#3）が入るが、これは DC/超低域を対象とする素子であり、**passband の flat gain 差（+2.4988N dB）を打ち消すものではない**（blocker の corner は本監査で実測していない）。

### D-1.4 CONTRACT IMPACT

- 「up/down 以外に factor 依存 gain が存在しない」＝ **打ち消し相手が構造的に存在しない** → 二重補償リスクは D-2/D-4 で「なし」と確定できる。
- 一方で **level は確実に上がる**（既定 +7.4963 dB）。案E採用時の音量差の扱い（release note・calibration・F-3/F-4）は Phase 1 完了後のユーザー GO 事項として切り出す。

---

## D-2 二重補償 census

### D-2.1 SOURCE — 検索手順（production `src/**`・tests/harness 除外）

- トークン検索: `*2.0` / `*= 2.0` / `/ 2.0`、`0.75` / `0.5625` / `0.421875` / `2.4988` / `6.0206` / `4.0/3.0` / `1.3333` / `1.41421356` / `0.70710678`、`headroom` / `makeup` / `trim` / `normalize` / `compensat` / `unity` / `autoGain` / `loudness`
- 全文読解: `AutoGainPlanner.h`（125 行）・`AutoGainPlanner.cpp`（111 行）・`AudioEngine.Parameters.cpp:285-347`・`RuntimeBuilder.cpp:295-340`・`AudioEngine.Processing.DSPCoreDouble.cpp:300-560, 700-762`

### D-2.2 OBSERVATION

- **×2/÷2 の利得出現は `src/CustomInputOversampler.cpp:557`（`convValue *= 2.0`）の 1 箇所のみ**（＝ B-1 そのもの）。同ファイルの他の `2.0` 出現は `:313` の窓関数定数（`kPi * 0.5 * t`）のみ。
- 他の `*2.0` 系は `DSPCoreFloat.cpp:104`（crossfade ramp の `gainStep * 2.0f`）・`PrepareToPlay.cpp:199`（delay 上限 `safeSampleRate * 2.0`）で、いずれも利得補償ではない。
- `0.75` の出現は `AudioEngine.Retire.cpp:26`（fade scale clamp 0.75〜1.50）・`RuntimeHealthMonitor.h:86`（health threshold 0.75）・UI alpha（`EQControlPanel.cpp:493`）のみ。**0.75^N を補償するコードは存在しない**。
- `6.0206` / `±6.0206` / `1.3333` / `4/3` 系の定数は production に **存在しない**。
- **利得ではない ×2 トークン（誤読防止のための明示）**: `CustomInputOversampler.cpp:433`（`stageInputMax *= 2` = 段別ブロック長の計算）、`TruePeakDetector.cpp:49`（同型のサイズ計算）、`DSPCoreFloat.cpp:104`（crossfade ramp の `gainStep * 2.0f`）、`ConvolverControlPanel.cpp:778/782`（UI タイマー `intervalSec *= 2.0`）、`LoudnessMeter.cpp:170/172`（biquad 係数）、`OutputFilter.cpp:31-62`・`EQProcessor.Coefficients.cpp:172-206`・`EQResponseSampler.cpp:13/46`（biquad/評価係数の `2.0 * pi * f / fs`）、`MKLNonUniformConvolver.cpp:364/422/451`（Nyquist と cos テーパ）、`SimplePeakLimiter.h:41`（knee 中点）、`IRAnalyzer.cpp:78`（taper 長）、`SpectrumAnalyzerComponent.cpp:777`（Nyquist）、`PrepareToPlay.cpp:199`（delay 上限）。**いずれも利得補償ではない。**
- staging の実装（`AutoGainPlanner.cpp:52-95`・`AutoGainPlanner.h:51-55`）:
  - `inputDb = −max(0, eqBoost − margin) − qMargin`（EQ first）/ `−max(0, convBoost − margin)`（Conv first）等
  - `trimDb = −max(0, convBoost − kMarginInterStage)`
  - `rawMakeupDb = −clampedInput − clampedTrim` → `jlimit(0, 12)`
  - **入力 DTO は `eqMaxGainDb` / `eqMaxQ` / `irFreqPeakGainDb` の 3 値のみ**（`PlannerInput`）で、**出力レベル実測は入力に無い**。呼び出し元は `RuntimeBuilder.cpp:303-337` の 1 箇所。
- 既定 staging（`applyDefaultsForCurrentMode`・`AudioEngine.Parameters.cpp:298-347`）: conv bypass + EQ active → `0/0/0`、EQThenConvolver かつ両者 active → `0/+10/−6`、その他 → `−6/+12/0`。**いずれも factor 非依存**。

### D-2.3 INFERENCE / CONTRACT IMPACT

- **INFERENCE**: 案E の +2.4988N dB を打ち消す既存補償は存在しない → **二重補償は成立しない**。逆に「案E の level 上昇が自動 gain に吸収される」こともない（自動 gain は EQ/IR の feed-forward のみで、出力実測を持たない）。
- **CONTRACT IMPACT**: 条件D の「既存補償との二重補償にならないか」＝ **NO（二重補償なし）**。残る論点は「+2.4988N dB（既定 +7.4963 dB）の level 変更をどう扱うか（calibration）」に移る。

---

## D-3 案E（`centerValue *= 2.0`）仮想影響マトリクス（production 変更なし）

| 軸 | base | cand（案E） | 差分 | 根拠（OBSERVATION の出所） |
|----|------|-------------|------|---------------------------|
| DC 利得（oversampler 単体） | 0.75^N 厳密 | 1.0 厳密 | **+2.4988N dB** | P0-A（±1e-6・IIR3/LP3 × r=2/4/8） |
| 50 Hz / 1 kHz / 10k / 48k / 86.4k / 94.08k | −2.4988N dB | 0 dB | **+2.4988N dB 一様** | P0-B/C'（cand unity u50/u1k=0.0000・差分 maxDev ≤0.0002 dB、devTerm = 20N·log10(4/3)） |
| passband ripple [0.005, 0.30] | — | 0.0000〜0.0010 dB | 変化なし | P0-C（gate ≤0.05） |
| stopband / alias | 相対 −(A−10) dB 以下 | 同左（6 design PASS） | 相対不変・絶対は +2.4988N dB 一様 | P0-D（stopMax: −137.1/−107.0/−87.0/−157.1/−137.0/−117.0 dB） |
| up ドメイン image（f̂≤0.30） | **−9.542425 dB（= 1/3）** | ≤ −46.2 dB | **改善** | P0-E E-1 / E-1c（min margin 36.8 dB） |
| output レート image | −223.3 / −217.7 / −184.3 dB | 同値（≤0.01 dB 差） | 変化なし | P0-E E-2（f̂ 0.05/0.10/0.20・S1） |
| impulse 位置 / 推定 latency | argmax ∈ [floor(D), floor(D)+1]・base==cand bitwise | 同左 | **変化なし** | P0-F（4 経路：S1 15 / 511 255 / IIR3 290.25 / LP3 582.25） |
| group delay | 対称 up/down（`Latency.cpp:6-8` static_assert）・推定式不変 | 同左 | 変化なし | SOURCE + P0-F |
| SoftClip 動作点 | clip 入力は up ドメイン（makeup 後・clip 前） | **+2.4988N dB** 駆動増（閾値固定: 既定 sat=0.1 で θ_clip=0.905・knee 0.085・asym 0.01） | 動作点移動（小） | P0-I I-b/c（15 stimuli: events 119,768→121,267 = +1.25%・R_s=1.013・\|y\|max=0.9・hard clamp 0 件・tanhClamp 18,200 件） |
| bypass 切替 | dry=pre-up へ fade → 段差に up/down loss が寄与 | 段差の OS 分（−2.4988N dB）が消える | headroom/makeup 由来の段差は残存（既存挙動） | SOURCE `:351-355` / `:517-559` |
| EQ / Convolver | linear・乗算的 | 同左 | 変化なし（非線形段でのみ相互作用） | SOURCE `:386-475` |
| 最終 limiter | θ=0.8413951287507587（kOutputHeadroom −0.5 dB）・knee=0.108748・clamp ±0.8912509381337456 | 同左 | 閾値超の material では上昇分の一部が limiter に吸収され limiting 量が増加。閾値未満では **+2.4988N dB がそのまま出る** | SOURCE `:715-749` |
| float 経路 | 構造同一（`DSPCoreFloat.cpp:235/263/382/405-419`）・同一 10395 Padé | 同左 | 変化なし（量子化差のみ） | P0-G（RT maxAbs 2.58e-08 / SC 2.96e-08 ≤5e-7） |

**CONTRACT IMPACT**: 案E の影響は (a) 一様 gain（level）、(b) up ドメイン image の改善、(c) 非線形段（SoftClip / limiter）の動作点移動 —— の 3 つに閉じる。timing・latency・ripple・相対 stopband 特性は不変。**可聴影響の主張はしない**（up ドメイン image は linear 経路では down 後に −184 dB 以下まで落ちる = E-2 OBSERVATION。SoftClip 併用時の intermodulation は INFERENCE に留める）。

---

## D-4 既存補償 inventory（利得管理機構の全量）

| 機構 | 位置 | 値・範囲 | 既定 | factor 依存 | 役割 |
|------|------|----------|------|------------|------|
| input headroom gain | `DSPCoreDouble.cpp:334` / `Parameters.cpp:225-249` / 自動は `AutoGainPlanner` | 手動 −12〜0 dB・自動 −18〜0 dB | 0 or −6 dB | なし | clip 余裕 |
| convolver input trim | `DSPCoreDouble.cpp:446-453` / `Parameters.cpp:270-297` | −12〜0 dB | 0 or −6 dB | なし | 段間余裕 |
| output makeup | `DSPCoreDouble.cpp:477-481` / `Parameters.cpp:250-267` | 0〜+12 dB | 0 / +10 / +12 dB | なし | ネット 0 dB 整合（planner: `makeup = −input − trim`） |
| DC blocker（OS レート） | `DSPCoreDouble.cpp:371-375` | — | 有効 | なし | DC 除去 |
| SoftClip | `DSPCoreDouble.cpp:483-515`（本体 `AudioEngine.Processing.DSPCoreDouble.cpp` 内 `softClipBlockAVX2`） | θ_clip=0.95−0.45·sat・knee=0.05+0.35·sat・asym=0.10·sat | enabled=true / sat=0.1 | なし | 非線形保護 |
| peak limiter + hard clamp | `DSPCoreDouble.cpp:715-749` | θ=0.8413951287507587・knee=0.108748・clamp ±0.8912509381337456（−1.0 dBFS） | 有効 | なし | 出力保護 |
| **補償ではない（明示）** | `Retire.cpp:26`（0.75〜1.50 clamp）・`LoudnessMeter.cpp:195`（1/√2 Q）・`PsychoacousticDither.h:219`（NS 係数）・UI alpha 0.75 | — | — | なし | 0.75^N と無関係 |
| **該当なし（探索結果）** | ×0.5 / ÷2 / 6.0206 dB / ±6.0206 / (4/3) 系 | — | — | — | production に存在しない |

**CONTRACT IMPACT**: 「既存補償 inventory」に up/down 由来 gain を打ち消す機構は 1 つも無い。補償候補として残るのは「level 変更を意図的に受け入れる（release note）」か「別途 calibration を設計する（F-3/F-4・HOLD）」の二択のみ。

---

## D-5 リスク・矛盾リスト

1. **[高] level 変更**: 既定（Auto=8x）で **+7.4963 dB**、2x で +2.4988 dB。ユーザー可聴の音量差。→ 緩和は release note / calibration（F-3/F-4 は HOLD のため Phase 1 では扱わない）。
2. **[中] SoftClip 動作点移動**: clip 入力が +2.4988N dB 上がり、固定閾値に対する駆動が増える（OBSERVATION: events +1.25%・R_s 1.013）。→ Phase 1 の flag ON 実測で R12-8 の再測定対象に含める。
3. **[中] limiter 到達量の増加**: hot material では上昇分の一部が limiter に吸収され、level 差が圧縮されて聞こえる代わりに limiting 量が増える。→ Phase 1 で limiter 前後（θ 到達率）を記録。
4. **[低] up ドメイン image（−9.54 dB = 1/3 振幅）**: linear 経路では down 後に −184 dB 以下へ減衰（E-2 OBSERVATION）。可聴影響の主張は不可。SoftClip 併用時の intermodulation 寄与は **INFERENCE**。
5. **[低] 既存の独立要因**: mode 依存の既定 staging（`0/+10/−6`・`−6/+12/0`）と bypass 段差（dry は makeup を通らない）は案E と独立に存在する。**「案E で level が揃う」と誤記しないこと**（揃うのは OS 由来分のみ）。
6. **[構造] 矛盾は検出されず**: 二重補償の相手が存在しないため、案E と既存補償の衝突は構造的に起こらない。唯一の未解明点は「なぜ既定 Auto が最大 loss（8x/3 stage）を選ぶ設計なのか」で、これは Phase 1 で flag OFF のまま実測記録を取る。
7. **[境界] 測定構成の限界**: Phase 0 の数値は **flag off / shadow_candidate 0**（REF-FIDELITY により production↔shadow bitwise 等価）での測定であり、production 動作点（flag ON）での再測定は Phase 1（R12-8）。聴感評価（listening）は未実施。

---

## D-6 Phase 1 eligibility 判定

**判定: ELIGIBLE（条件付き）**

根拠（すべて D-1〜D-5 の OBSERVATION に紐付け）:

1. **二重補償なし**（D-2/D-4）: factor 依存の補償は production に 0 件、自動 gain は EQ/IR の feed-forward のみ（出力実測なし）。
2. **影響範囲が閉じている**（D-3）: level 一様 +2.4988N dB・timing/latency/ripple/相対 stopband 不変・image は改善・非線形段の動作点移動は測定済み（+1.25% events）。
3. **Phase 0 全 gate PASS（54/54・exit 0）**＋ REF-FIDELITY 31/31（production↔shadow bitwise）で、変更の数学的契約が固定されている。
4. **変更対象が局所**: `src/CustomInputOversampler.cpp` の center 位相 1 行（`convValue *= 2.0` と対称な位置）＋ flag 経路のみで、既定 OFF を維持したまま実装できる（R15-2/R18-1）。

Phase 1 で満たすべき条件（判定の付帯条件）:

- flag は **default OFF を維持** · CMake token は Phase 1 の範囲内（R15-2/R18-1）· production 既定値を Phase 0 freeze の attestation から変えない。
- Phase 1 は **flag ON ビルドでの production 動作点再測定**（R12-8）を含む: level（DC/50 Hz/1 kHz/passband）・SoftClip 動作点（events/R_s）・limiter 到達率・latency 不変（P0-F 相当）。
- level 変更の扱い（release note・calibration・F-3/F-4・default ON 判断）は **Phase 1 完了後・ユーザー GO の対象**（本 doc では判断しない）。

**これは案E採用の確定ではない。** 採用は R15-4 条件D + ユーザー最終 GO の後。本 doc は判断材料の提示のみで、production・CMake・flag・default・calibration のいずれも変更していない。

---

## 付録: 本監査の限界（未検証項目）

- 聴感評価（listening test）は未実施。
- AudioEngine 全体（真の signal chain・実ホスト動作）での level 実測は未実施 — 本監査は **oversampler 単体の実測**（P0-A/C'/E/F/G/I）と **コード経路の読解**（D-1/D-2 SOURCE）で構成される。
- DC blocker の corner 周波数は未実測（D-1.3 の「passband gain 差を打ち消さない」は構造からの INFERENCE）。
- 実測はすべて `production_flag: undefined/off` / `shadow_candidate: 0` の構成。
- working tree の外部変更（`BassBuzzMeasurement.cpp` +636 行・`PublishPipelineIntegrationTests.cpp` +8 行・`src/tools/check_layout_offsets.py` 削除）は本監査と無関係（R18-3 記録済み）。

---

## C2 pre-commit invariant check（2026-09-22・C2 直前に実測）

| 検査 | コマンド | 期待 | 実測 |
|------|----------|------|------|
| production src 差分 | `git diff --name-only -- src` | 0（production） | **production 0** — 変更は `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp`（C2 evidence）・`BassBuzzMeasurement.cpp` / `PublishPipelineIntegrationTests.cpp`（外部変更）・`src/tools/check_layout_offsets.py`（外部削除）のみ |
| CMakeLists.txt 差分 | `git diff --name-only -- CMakeLists.txt` | 0 | **0**（出力空） |
| flag token（CMakeLists.txt） | `git grep -n CONVOPEQ_CORRECT_POLYPHASE_GAIN -- CMakeLists.txt` | 0 | **0 件** |
| flag token（production src） | `git grep -n … -- src/audioengine src/*.cpp src/*.h`（tests/tools 除外） | 0 | **0 件** |
| flag token（repo 全体・参考） | `git grep -n CONVOPEQ_CORRECT_POLYPHASE_GAIN -- CMakeLists.txt src` | 0（R18-1 の不変条件は **CMakeLists.txt 限定**） | **3 件 = 検出器/コメントのみ・定義なし**: `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h:39`（コメント）・`src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp:107`（`#ifdef` 検出 — 定義されていたら「UNEXPECTED — Phase 0 契約違反: R15-2」を出力）・`src/tools/build_identity_gate.py:448`（CMakeLists.txt 内 token 検出器 — 検出時は「DEFINED — Phase 0 契約違反: 測定無効」） |

**判定**: R18-1 の commit 不変条件（**flag token が CMakeLists.txt に現れた場合＝違反**）は**成立**。test/tooling 側の 3 件は R15-2 を**強制する検出機構**であり token を定義しない（C1 commit `85aa13b9` 時点から同一の状態で、C1 gate でも同じ scope で PASS 済み）。production 側 0 件・CMake 側 0 件・production src 差分 0・CMakeLists 差分 0 → **C2 続行可**。

### C2 staged 集合（限定）

```text
M  src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp   ← Phase 0 evidence（+668 行）
A  doc/work113/phase0_characterization_report_20260922.md        ← freeze 記録 §6（SHA-256 attestation）
A  doc/work113/step25_condition_d_evidence_20260922.md           ← 条件D D-1〜D-6
```

**C2 に含めない**: `src/CustomInputOversampler.*` / `src/audioengine/**` / `CMakeLists.txt` / `build_identity_gate.py` / `output_sourcecode_markdown.py` / `ConvoPeq.md` / `BassBuzzMeasurement.cpp` / `PublishPipelineIntegrationTests.cpp` / `src/tools/check_layout_offsets.py` / その他 O-item。
