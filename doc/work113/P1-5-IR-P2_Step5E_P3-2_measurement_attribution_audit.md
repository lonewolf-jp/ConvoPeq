# P1-5-IR-P2 — Step 5-E / P3-2: Measurement Attribution Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-2）
- **性質**: **完全 read-only**。再実行・build・source変更・logger/CLI追加なし。
  調査対象は現行 source の読みと既存 evidence（A3 ログ・Forward ログ）の読みのみ。
- **目的**: `+1.4759 dB` の原因を決めることではなく、
  **測定器としての P1CHAR を信頼してよいか**を先に確定する。
  すなわち測定系だけで Δ が説明できるか（M1）、測定は妥当か（M2/M3/M4）を分類する。
- **前提**: P3-1-C=STOP / P3-1C-R1=COMPLETE(C4) / Reverse=NOT EXECUTED /
  P3-1-D=NOT STARTED / H-B=OPEN・対象外。

---

## 1. State Freeze

```text
HEAD                         1e9e63e3（1e9e63e34bed7adb9342ebc81259ded9689fc48a）
kP15FullMatrix               true
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
A3 binary / backup           b2c39a9a0e3b1fe3 / b2c39a9a0e3b1fe3
production / CMake / JUCE / settle-sleep / measurement-semantics  0 diff
```

禁止事項（すべて遵守・0件）:
`Forward/Reverse再実行・p15hrnl skip・settle/sleep変更・logger追加・
crash dump設定変更・limiter reset追加・CLI追加・production変更・test source変更・build`。

---

## 2. gainDb source trace（P3-2-A 本幹）

`runCase`（P1PolyphaseGainCharacterization.cpp:404-464）の確定経路：

```text
ampDb（引数）
 → amp = 10^(ampDb/20)                                   :420
 → configureCapture(cap, Sine, 1000.0, amp, false)       :421（c.amp=amp・phase=0 reset）
 → tap が v = c.amp·sin(phase) を input buffer に書く    :161（phase+=2π·1000/48000）
 → engine_->getNextAudioBlock(info)                      AudioEngineHarness.cpp:107
 → capture（output tap・下記§5）
 → warm 4 + cap 12 blocks 待機（20 s 打切り付き）        :424-432
 → from = total−4096, n = kAnalysisN = 4096             :439
 → dftDb = 20·log10(dftMag+1e-300)                       :459
 → gainDb = dftDb − ampDb                                :460（これだけ。補正・正規化の追加なし）
```

定数（:34-47）: `kSr=48000.0 / kBlock=512 / kWarmBlocks=4 / kCapBlocks=12 /
kAnalysisN=4096 / kLimThreshold=0.8413951287507587 / kHardClamp=0.8912509381337456`。

## 3. ampDb source trace（A1 結論）

- `ampDb` は `runPair` の `PairCfg` 由来の**指定値**（p15ir では `adb` loop変数そのもの：:746→:749）。
- generator への対応は厳密：`amp=10^(ampDb/20)`（:420）と tap の `v=c.amp·sin`（:161）の間に
  分岐・条件・丸め（float化を除く相対誤差〜1e-7・レベル非依存）なし。
- captured input amplitude は**存在しない**：`runPair`→`runCase` は常に
  `captureInput=false`（:666/:685）で呼び、`c.inL` は空のまま。
  よって `ampDb` は「記録された入力レベル＝発生させたレベル」であり、測定された入力ではない。
- `-20` 行の amp=`10^(−20/20)=0.1`、`-6` 行の amp=`10^(−6/20)≈0.5011872`。両行とも同式。

## 4. dftMag normalization audit（A2/A4 結論）

`dftMag`（:200-214）の確定仕様：

```text
window      Hann w=0.5·(1−cos(2πi/(n−1)))（:205-206）
window sum  wsum=Σw（:211。Hann wsum≈n/2=2048）
size        n=kAnalysisN=4096（両行同一）
single-bin  相関型単一成分（FFT bin ではない）。対象 freq=1000.0・kSr=48000（両行同一）
×2          mag=2·|X|/max(wsum,1e-30)（:213。coherent gain=1 の正規化）
input norm  なし（x を読むだけ。:208）
output norm なし（振幅依存の分岐なし。全演算は x に対して線形）
```

A4（両行の同一性・by construction＋ログ確認 `os=1 n=0` 両行）:
測定窓（末尾4096）・相関周波数（1000 Hz）・window（Hann）・capture block 数（warm4+cap12、
解析は先頭12block中の blocks 4–11＝末尾4096サンプル）・`dftDb−ampDb` 式は完全同一。
差は `p.ampDb` のみ（P3-1-B §1-1-10 と一致）。

## 5. capture tap audit（P3-2-B 結論）

### 5-1. harness tap（AudioEngineHarness.cpp:99-109）

```text
tapCopy(buffer, true)    ← 入力tap（信号生成）  :105
engine_->getNextAudioBlock(info)                 :107
tapCopy(buffer, false)   ← 出力tap（capture元） :109
```

出力 tap は `getNextAudioBlock` の**後**＝DSP chain 全段の下流。

### 5-2. DSP chain 実順序（source確定）

`processBlockDouble`（DSPCoreDouble.cpp）→ `processOutputDouble`（:589-761）：

```text
processInputDouble（headroom）          :334
→ os>1 のみ processUp                   :361（os=1 で skip）
→ convolver（ConvolverThenEQ）          :386-454（p15ir: convBypass=false）
→ trim                                  :446-451（0 dB）
→ EQ（全band disabled＝identity）       :371-375相当（両行同一）
→ softClip/saturation                   :483-（sc0 は block 全体 skip。sat は同 block 内 :486 のみ）
→ os>1 のみ processDown                 :541（os=1 で skip）
→ processOutputDouble                   :572
   → DC blocker                         :613
   → dither/noiseShaper（kOutputHeadroom 径）:656-675（ditherBitDepth=32 のため applyDither=true）
   → NaN/Inf scrub                      :677-705
   → truePeak / loudness（meter のみ）   :710/:713
   → SimplePeakLimiter（無条件）         :722（kPLThreshold=0.8413951287507587・kPLKnee=0.108748）
   → hard clamp ±kOutputHeadroom        :724-749
   → fixed latency delay                :751
   → buffer へ copy                     :756-758 → harness output tap
```

### 5-3. P1CHAR tap と processOutput tap の対応（再確認）

- P1CHAR の `outL` は harness output tap（§5-1）の `capBlocks=12` 先頭分であり、
  `processOutputDouble` の copy 先 buffer（:756-758）と同一信号である。
- 既存監査の「capture は limiter downstream」は本追跡で再確認された。
  さらに hard clamp・dither・delay の**下流**であることも確定した。
- sc1 のみ局所2倍OS経路（:503-513 `processUp→SoftClip→processDown`）を通る（P3-0 §P3-0-5①と一致）。

## 6. limiter/clip field producer audit（P3-2-D 結論）

### 6-1. SimplePeakLimiter の確定仕様（SimplePeakLimiter.h:35-83）

```text
clipStart = threshold − knee/2 = 0.8413951287507587 − 0.054374 = 0.7870211…
knee帯（0.7870〜0.8414）: smoothstep t²(3−2t) 補間（:61-63）
>threshold: threshold/peak（:68）
envelope: attack即時（desiredGain<envelope → 代入 :73-74）、release時定数100 ms
         （prepare(newSampleRate,100.0)・DSPCoreLifecycle.cpp:228,292）
適用: 無条件に dataL/R[i] *= envelope（:79-81）
accessor getCurrentEnvelope() の呼び出しは全sourceで0件（P3-1-A確定・維持）
```

### 6-2. P1CHAR 各fieldのproducer（CaseOut :339-349・runCase :440-443・runPair :690-703）

| field | producer | 意味（確定） |
| --- | --- | --- |
| `gainDb_sc0/sc1` | `dftDb−ampDb`（:460） | post-chain 出力の1 kHz成分レベル−入力レベル |
| `limitingEngaged` | `countAtOrAboveWindow(outL,…,kLimThreshold)`（:442） | **capture済み出力**が0.8414以上だったサンプル数。limiter内部状態ではない |
| `hardClamp` | `countAtOrAboveWindow(outL,…,kHardClamp)`（:441） | capture済み出力が0.89125以上だったサンプル数 |
| `clipEngagement` | `diffStats(yOn,yOff)` の差分数（:691-695） | **sc1 capture と sc0 capture の差分サンプル数**（\|a−b\|>0）。limiter engagement counter ではない |
| `clipEngMax` | 同 `max\|a−b\|` | sc1−sc0 の最大絶対差 |
| `limiterInPeakDb` | `e.getOutputLevel()`（:443。p15ir行には非出力） | UI用 output level meter（AudioBlock.cpp:417/474更新の走行meter）の読値。case-gated精密peakではない |

### 6-3. 両方向の早計の禁止（確定）

- `limitingEngaged=0` は limiter 非動作を証明しない：
  出力側カウンタであり、knee帯（0.7870〜0.8414）の gain reduction や
  release残留（領域1でも `envelope<1.0` が乗算され続ける：P3-1-A §2-1）は
  カウントされない。peak が limiter で削られて0.8414未満で出力された場合も0のまま。
- `clipEngagement=4096` は limiter 原因を意味しない：
  sc0/sc1 キャプチャ差分であり、sc1 の局所2倍OS再サンプリング（§5-3）だけでも
  全サンプルに微差が生じる。`clipEngMax`（0.024784 / 0.163382）は sc1−sc0 差の大きさである。
- よって現段階では limiter の肯定・否定いずれも行わない。

## 7. crossfade/runtime-state audit（P3-2-E 結論）

### 7-1. 確定機構（source）

- `getNextAudioBlock` は world 切替時に **等電力 crossfade（旧core出力＋新core出力の latency-aligned mix）**
  を行う（AudioBlock.cpp:286 snapshot・:349-371 arm・:384-460 mix・`equalPowerSin` :19-24）。
  fade時間は world overlap パラメータ（`fadeTimeSec` 系・AudioEngine.h:2369/2422-2427、Commit.cpp:234）。
- `runCase` は `waitWorldPublished(seqBefore,30000)` の戻り値を無視する（:417）。
  よって capture が (a) stale world、(b) crossfade混合中、(c) 新world安定後のいずれで
  処理されたかは測定ログだけでは確定できない（PUBLISH/REBUILD_TELEMETRY は本vehicle非出力・P3-1-A確定）。
- `RuntimeBuilder` は successful build ごとに新 `DSPCore` を生成する（:469）。
  rebuild完了時は envelope を含め新規状態に戻る（事実維持・P3-1-C §7）。
- crossfade混合時は**旧core（前ケースの履歴を持つ）の出力が現captureに混入しうる**。
  これは source 上 reachable であり、limiter envelope 持続の結論とは独立の事実である
  （「envelopeが持続している」とは結論しない。指示どおり）。

### 7-2. convolver history の付帯観測（source＋定数のînferred fact・候補扱い）

- IR長 48000 samples（1 s）に対し1ケースは16 blocks＝8192 samples（§2）。
  解析窓（blocks 4–11）はいずれの行でも IR tail の過渡内に位置する（両行共通手順）。
- LTI chain を仮定しても、前ケースの tail は**加算的履歴**として現ケースに残る：
  `out(−20)=0.1·H＋T(p15eq系)`、`out(−6)=0.501·H＋T(−20系)`。
  tail は現ampに比例しないため、LTI のままでも `gainDb` は行間で一致しない**可能性がある**。
- ただし rebuild 完了時に新coreの convolver history が cleared／transferred のいずれかは
  本 audit の範囲では未確定（`transferIRStateFrom` は IR データ転送・history扱いは別）。
  よって本項は M3 の候補として記録し、結論化しない。

## 8. existing evidence reconciliation

- Forward g0_os1 2行は A3 と全field一致（`−14.5035 / −13.0276 / Δ=+1.4759`）。
  既知値の再現は測定器の再現性として確定してよい。
- `p15ir` 18行の `gainDb` は A3–Forward で18/18一致。
  os4/os8 am-6 の `clipEngagement` count 微差は metric-level jitter（gainDb不変）。
- `clipEngagement=4096`（両行）は sc0/sc1差分であり、sc1局所2倍OS経路（§5-3）と整合。
- 後段crash（R1/C4）は `p15ir` evidence に触れない。H-Bマーカーは不在のまま。

### 数値の再掲（測定式からの検算・新規計測ではない）

```text
am−20: dftDb = −14.5035＋(−20) = −34.5035 dB → mag 0.018829
am−6 : dftDb = −13.0276＋(−6)  = −19.0276 dB → mag 0.111846
出力比 = 5.940 ≠ 線形期待 10^(14/20) = 5.012（比 1.1855× = +1.4759 dB と整合）
```

すなわち線形期待からの乖離は evidence 上確定している。問題はその帰属のみである。

## 9. 振幅依存性8候補の切分け（P3-2-C 結論・推測による選択なし）

| # | 候補 | 判定 | 根拠 |
| --- | --- | --- | --- |
| [1] | 入力振幅の記録誤り | **排除** | `amp=10^(ampDb/20)` 厳密・tap直結（§3）。float化誤差は相対〜1e-7でレベル非依存 |
| [2] | 出力DFT正規化のレベル依存 | **排除** | `dftMag` は入力に厳密線形・振幅分岐なし（§4） |
| [3] | capture窓の相違 | **排除** | 同一code path・同一const（§4・A4） |
| [4] | transient/warm-up混入 | **単独原因として排除・相互作用は保留** | 両行同一手順・LTIスケーリング下では単独でΔを生まない。ただし§7-2のtail加算は別候補としてM3に计上 |
| [5] | DSP chain非線形 | **確定も排除もしない** | sc0はsoftClip/saturation skip・EQ identity（source確定）。残る無条件非線形はlimiter・hard clamp・ditherのみで、いずれも本行での作用は非観測（§6-3） |
| [6] | limiter/clamp/saturationの実作用 | **確定も排除もしない** | カウンタ0は非動作の証明にならない（blind spot・§6-3）。`limiterInPeakDb` はp15ir非出力・走行meter由来 |
| [7] | crossfade/runtime state相違 | **排除できない** | publish無視（:417）＋等電力mix（§7-1）＋新core生成タイミング依存。両行の履歴位置は異なる（os8遷移直後 vs −20直後） |
| [8] | その他（dither・丸め等） | **differentiatorとして排除** | 両行同一手順・common-mode・微小（dB換算で無視可能量） |

## 10. M1/M2/M3/M4 classification

| 分類 | 判定 | 理由 |
| --- | --- | --- |
| **M1** Measurement artifact | **REJECTED** | [1][2][3]排除・[4]単独排除（§9）。測定式・窓・正規化だけではΔは説明できない |
| **M2** Valid／非線形path確定 | **NOT ESTABLISHED** | sc0の非線形stage作用がいずれも非観測（§6-3・§9[5][6]）。振幅依存と履歴位置が交絡しており、level単独への帰属は不可 |
| **M3** Valid／runtime-state candidate | **ADOPTED** | 測定器は妥当（M1棄却）。一方で stale/crossfade-mix capture（§7-1）・convolver tail加算（§7-2）・新core timing をいずれも排除できない |
| **M4** Inconclusive | — | M3が成立するため不採用 |

## 11. next-gate recommendation

- M3 の帰結として、次段では**短時間・限定 vehicle による runtime attribution**を設計する
  （70分 full-matrix への Reverse 再投入は R1/C4 の交絡リスクにより推奨しない）。
- 限定 vehicle の設計（対象：world/crossfade/history の分離）は次gateの判断事項であり、
  本 audit では設計・実装に進まない。
- P3-1-A の X-1/X-2/X-3 はいずれも選択しない（P3-1-C §7遵守）。
  特に X-3（limiter envelope accessor/log）は不要を維持する。
  `reverse-order evidence = 0` の現状では limiter 計装は問いを直接解決しない。

## 12. prohibited actions（遵守記録）

```text
source変更 0 / test source変更 0 / 新規logger 0 / 新規CLI 0 / 再build 0
Forward再実行 0 / Reverse再実行 0 / H-B再実行 0
settle/sleep変更 0 / crash dump設定変更 0 / limiter reset追加 0
```

STOP 条件に触れる事項は発生しなかった（本 audit は read-only で完結）。
H-B との因果関係の主張は行っていない（H-B は対象外を維持）。

---

## 最重要の解釈制約（P3-1-C §7・再掲・維持）

```text
RuntimeBuilder は successful build ごとに新 DSPCore を生成する。
したがって rebuild 完了時には SimplePeakLimiter の envelope は
新規状態に戻る。

runCase の waitWorldPublished() 戻り値は無視されるため、
ケース境界で rebuild が完了したかは今回の測定ログだけでは確定できない。

したがって reverse-order の結果だけから
「limiter envelope のケース間持続」を肯定/否定してはならない。
```

本 P3-2 でも limiter の肯定・否定は行っていない（§6-3・§9[5][6]）。
