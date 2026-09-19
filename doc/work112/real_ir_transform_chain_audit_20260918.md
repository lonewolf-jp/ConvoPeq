# WORK112（112-0〜112-3）— Real IR Transform Chain / LF Residual Origin Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（112-0〜112-3 実施、指示によりここで一旦停止）
- **前工程**: WORK111（`doc/work111/real_ir_tail_fade_impact_20260918.md`）
- **正本**: `ConvoPeq.md` 2026-09-18 21:46:55 / 5,260,149 bytes（112-0 完了）
- **制約遵守**: production algorithm / resampler / Tukey パラメータ / phase / scale policy / limiter /
  L0・FDL / OS 補償 / IR 長補正 の**いずれも未変更**。RT に分岐・確保・ログなし。
  追加は NonRT trace（`[IR_CHAIN]`）と test-only 解析（`[LF_RESIDUAL]`）のみ。

```text
LF RESIDUAL ORIGIN（112-3 時点の暫定）

Input:        clean（sine50_IN residualRms=3.53e-09, res/fund=-154.0 dB, nonfinite=0）
IR chain:     A→E で非有限 0 / subnormal 0 / 低域 band energy ほぼ不変（§2）
              末尾 fade は WORK111 で除外済み
              増分の出所なし
Output:       fundamental は 50Hz に正しく出る（h1=5.278e-3）
              一方 residual は **高調波ではない**（h2..h5 ≤ 1.8e-6 = fund の -69 dB 以下）
              residual は 100–2000 Hz の広帯域（sine50: band 100-200=1.23e2, 200-2k=1.16e2）
              residualPeak = 1.73e-3 = fundamental peak(5.28e-3) のわずか **-9.7 dB**
              subnormal = 0（→ denormal 仮説 棄却）

判定（暫定）:
    Case A（Transform 起因）  … 不支持（chain は清浄・低域 energy も不変）
    Case B（Phase 起因）      … 棄却（phaseMode=0 = AsIs。phase transform は実行されていない）
    Case C（Engine 起因）     … 有力
    Case D（Measurement 起因）… ほぼ否定（residual は float 量子化の ~1e6 倍。§4-3）
    ※ C と D の最終分離には 112-5（delta/silence IR + reference FIR の null test）が必要

Confidence:
    「residual は高調波ではなく非調和な広帯域成分」… PROVEN
    「denormal ではない」                          … PROVEN（subnormal=0）
    「phase 起因ではない」                        … PROVEN（AsIs）
    「Transform 起因ではない」                    … STRONGLY SUPPORTED
    「Engine 起因」                               … OPEN（要 112-5）
```

---

## 1. 112-0 対象関数の確認と double→float 境界

| 段 | 関数 | 型 |
|---|---|---|
| resample | `ConvolverProcessorInternal::resampleIR` / `IRDSP::resampleIR` | double |
| 長さ | `owner.computeTargetIRLength` | int（targetLength=192000） |
| trim | `LoaderThread::doTrimStep` | `juce::AudioBuffer<double>` |
| DC | `convo::UltraHighRateDCBlocker` | double |
| window | `applyAsymmetricTukey` | double |
| fade | `stepTrimmed.applyGainRamp` | double |
| phase | `convertToMinimumPhase` / `convertToMixedPhase`（**AsIs では未実行**） | double |
| scale | `IRConverter::computeScaleFactor` → `stepResult.scaleFactor` | double（スカラー） |
| engine 入力 | `buildConvolverFromTrimmed` → `MKLNonUniformConvolver::SetImpulse(const double*, ...)` | **double** |
| scale 適用 | `MKLNonUniformConvolver::SetImpulse` 内 `cblas_dscal(scale, irFreqDomain)`（周波数領域） | double |
| 出力 | `MKLNonUniformConvolver::Get(double* output, ...)` | double → host で float、harness capture は float |

- **double→float はエンジン出力以降（host バッファ／harness capture）**。IR は SetImpulse まで double のまま。

## 2. 112-1 checkpoint（実 IR / 同一 generation）

```text
[IR_CHAIN] A_resampled n=62914 sr=384000 peak=9.80902780e-01 rms=1.13128024e-02 dc=1.39149764e-04 energy=8.05170220e+00 nonfinite=0 subnormal=0 lf[0-20]=8.508513e-03 [20-50]=6.896089e-03 [50-100]=1.164867e-02 [100-200]=2.954676e-02
[IR_CHAIN] B_dcblock  n=62914 sr=384000 peak=9.80754818e-01 rms=1.13119425e-02 dc=2.25287649e-07 energy=8.05047821e+00 nonfinite=0 subnormal=0 lf[0-20]=7.577165e-03 [20-50]=6.874675e-03 [50-100]=1.164361e-02 [100-200]=2.954419e-02
[IR_CHAIN] C_tukey    n=62914 sr=384000 peak=9.80754818e-01 rms=1.13118274e-02 dc=7.86257730e-06 energy=8.05031433e+00 nonfinite=0 subnormal=0 lf[0-20]=7.559482e-03 [20-50]=6.880379e-03 [50-100]=1.164911e-02 [100-200]=2.954244e-02
[IR_CHAIN] D_trimfade n=192000 sr=384000 peak=9.80754818e-01 rms=6.47523903e-03 dc=2.57687237e-06 energy=8.05031433e+00 nonfinite=0 subnormal=0 lf[0-20]=7.559620e-03 [20-50]=6.880243e-03 [50-100]=1.164911e-02 [100-200]=2.954244e-02
[IR_CHAIN] E_phase   n=192000 sr=384000 peak=9.80754818e-01 rms=6.47523903e-03 dc=2.57687237e-06 energy=8.05031433e+00 nonfinite=0 subnormal=0 lf[0-20]=7.559620e-03 [20-50]=6.880243e-03 [50-100]=1.164911e-02 [100-200]=2.954244e-02
[IR_CHAIN] F_scale scaleFactor=0.03002679 phaseMode=0
```

| checkpoint | n | peak | rms | dc | energy | nonfinite | subnormal |
|---|---|---|---|---|---|---|---|
| A resampled | 62914 | 0.98090278 | 1.13128e-2 | 1.3915e-4 | 8.05170220 | 0 | 0 |
| B dcblock | 62914 | 0.98075482 | 1.13119e-2 | **2.2529e-7** | 8.05047821 | 0 | 0 |
| C tukey | 62914 | 0.98075482 | 1.13118e-2 | 7.8626e-6 | 8.05031433 | 0 | 0 |
| D trim/fade | **192000** | 0.98075482 | 6.4752e-3 | 2.5769e-6 | 8.05031433 | 0 | 0 |
| E phase | 192000 | 同上（**AsIs のため無変換**） | | | | 0 | 0 |
| F scale | — | `scaleFactor=0.03002679`（2 回目 build は 0.02527018） | | | | | |

**要点**
- DC block は DC を 1.39e-4 → 2.25e-7（**-56 dB**）に低減。energy への影響は 0.015%。
- Tukey は energy を 8.050478 → 8.050314（**-0.002%**）、低域 band もほぼ不変。
  → Tukey は実 IR の主要部に触れていない（peak が index≈488 と早いため）。
- D で n=192000 になるのは `computeTargetIRLength` による **ゼロパディング**。energy 不変＝追加分は無音。
- E = D（`phaseMode=0` = **AsIs**）→ **phase transform は実行されていない**。
- **A→E の全段で nonfinite=0 / subnormal=0、低域 band energy は単調微減のみ**。
  → **Transform 側で低域の非線形残差は生成されていない**。

## 3. 112-2 harmonic の絶対振幅（最重要データ）

`[LF_RESIDUAL]`（flat-top HFT95 windowed DFT, n=131072, 解析レート 192000）

| | sine50_IN | sine50_OUT | sine40_IN | sine40_OUT |
|---|---|---|---|---|
| fundamental（h1） | 2.5000e-01 | **5.2780e-03** | 2.5000e-01 | **2.3140e-03** |
| h2 | 1.83e-06 | 1.79e-06 | 3.89e-06 | 2.83e-06 |
| h3 | 1.69e-06 | 4.88e-08 | 2.84e-06 | 6.90e-07 |
| h4 | 1.40e-06 | 3.74e-08 | 1.28e-06 | 1.47e-06 |
| h5 | 1.07e-06 | 1.23e-07 | 9.64e-07 | 4.85e-07 |
| dc | 5.49e-13 | -2.99e-07 | 2.86e-13 | 3.24e-06 |
| fundRms（LS） | 1.7678e-01 | 3.7326e-03 | 1.7678e-01 | 1.6333e-03 |
| residualRms | 3.53e-09 | **5.5635e-04** | 3.73e-09 | **2.2545e-03** |
| residualPeak | 7.70e-09 | 1.7311e-03 | 7.58e-09 | 7.0205e-03 |
| res/fund | -154.0 dB | **-16.53 dB** | -153.5 dB | **+2.80 dB** |
| subnormal | 0 | **0** | 0 | **0** |
| nonzero | 131072 | 131072 | 131072 | 131072 |
| nonfinite | 0 | 0 | 0 | 0 |

**要点**
- 入力は **-154 dB の清浄な正弦**（解析系の妥当性確認を兼ねる）。
- 出力の **h2..h5 は fundamental の -69 dB 以下**（sine50: h2=1.79e-6 vs h1=5.28e-3）。
  → **「THD」として観測されていた残差は高調波歪ではない**。
- にもかかわらず residualRms は fundamental に対し **-16.5 dB（sine50）/ +2.8 dB（sine40）**。
  sine40 では **残差が基本波を上回る**。

## 4. 112-3 residual の分類

### 4-1 residual band energy（Hann/65536、解析レート 192000）

| band | sine50_IN | sine50_OUT | sine40_IN | sine40_OUT |
|---|---|---|---|---|
| 0–20 Hz | 8.1e-17 | 1.58e-02 | 2.2e-17 | 3.28e-01 |
| 20–50 Hz | 4.6e-16 | 1.48e-01 | 1.9e-16 | 7.64e-02 |
| 50–100 Hz | 1.2e-16 | 2.06e-02 | 1.0e-19 | 4.19e-02 |
| **100–200 Hz** | 2.7e-11 | **1.227e+02** | 3.8e-11 | **1.834e+03** |
| **200–2k Hz** | 2.6e-10 | **1.162e+02** | 2.4e-10 | **2.046e+03** |
| 2k–20k Hz | 9.7e-09 | 1.10e+01 | 1.1e-08 | 1.90e+02 |

- residual は **100 Hz–2 kHz の広帯域**が支配的（低域 0–100 Hz は相対的に小さい）。
- 入力の同帯域は 1e-11〜1e-10（＝数値ノイズ床）。出力は **12〜13 桁大きい**。

### 4-2 residual の上位ピーク（非調和）

```text
sine50_OUT resPeaks: 137.7Hz:9.02e+00 237.3Hz:5.23e+00 134.8Hz:4.98e+00 140.6Hz:4.06e+00 325.2Hz:3.82e+00
sine40_OUT resPeaks: 146.5Hz:3.23e+01 149.4Hz:2.63e+01 228.5Hz:2.10e+01 225.6Hz:1.71e+01 334.0Hz:1.42e+01
```

- 支配成分は **≈135–150 Hz / ≈225–240 Hz / ≈325–335 Hz** の群。
- **駆動周波数の整数倍ではない**（50→100/150/200 でも 40→80/120/160 でもない）。
  sine50 で 137.7、sine40 で 146.5 と **入力周波数に依存して移動**する。
- → **固定 IR の線形畳み込み（定常状態）では生成不可能な成分**。
  （純正弦 × 固定 IR の出力は同じ周波数の正弦のみ。）

### 4-3 denormal / float 量子化の除外

- **subnormal = 0**（出力 131072 点すべてで |x| ≥ `float` の最小正規値）。
  → **denormal 仮説（112-6）は棄却**。
- residualPeak 1.73e-3 は、signal（fundamental 5.28e-3）に対する `float` 量子化（相対 ~1e-7、
  絶対 ~5e-10）の **約 1e6 倍**。→ **capture 側の float 丸めでは説明できない**。

## 5. 判定ゲート（暫定）

| Case | 判定 | 根拠 |
|---|---|---|
| **A: Transform 起因** | **不支持（STRONGLY SUPPORTED）** | A→E で nonfinite=0 / subnormal=0 / 低域 energy 不変。増分なし（§2） |
| **B: Phase 起因** | **棄却（PROVEN）** | `phaseMode=0`（AsIs）。`convertToMinimumPhase`/`MixedPhase` は実行されていない（§1/§2） |
| **C: Engine 起因** | **有力（OPEN）** | 入力清浄・chain 清浄・float 量子化の 1e6 倍・非調和広帯域（§3/§4） |
| **D: Measurement 起因** | **ほぼ否定（OPEN）** | 入力対照が -154 dB、解析妥当。residual は float 量子化を大きく超過（§4-3） |

- 112-4（1 段ずつ A/B）は不要な段が多い（DC/Tukey/phase/scale はいずれも §2 で残差を生まない）。
- **C と D の最終分離には 112-5 の null test が必要**：
  - 同一 `G`（engine input）に対し、test-only の単純 FIR/参照畳み込みと ConvoPeq engine 出力を比較し、
    `null RMS / null dB` を取る。
  - 併せて **silence IR（全零）で出力が厳密ゼロか**、**delta IR で出力が入力の遅延コピーか**を確認すれば、
    engine のベースライン数値挙動と capture 由来を一撃で分離できる。

## 6. 未確定事項（次工程）

1. **Case C/D の分離**（112-5 null test、silence/delta IR の応答）。
2. **非調和 135–150 Hz 群の発生機構**：入力周波数に依存して移動することから、
   固定周期（ブロック／リング／スケジューラ）との変調・折返しが候補。候補周波数：
   base block 1024@192k = 187.5 Hz、engine block 2048@384k = 187.5 Hz、ring 16384@384k = 23.4 Hz。
   135–150 Hz はこれらの整数倍ではないため、要特定。
3. resample 比 7.6241（`8252 → 62914`）の出力長式（112-7、本 WORK では未着手）。

## 7. 変更したファイル（test-only / NonRT trace のみ）

| ファイル | 変更 |
|---|---|
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | `[IR_CHAIN]` checkpoint A〜F（NonRT 観測のみ）。値の変更なし |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `printLfResidual()`（LS 基本波除去 + flat-top DFT h1..h5 + residual FFT band/上位ピーク + subnormal 計数）と probe 経路での `_IN`/`_OUT` 出力。test-only |

- production algorithm / resampler / Tukey / phase / scale / limiter / L0・FDL / IR 長 は**未変更**。
- RT（Add/Get）は**未変更**。`IRRuntimeContract` 未変更。

## 8. 指示

112-0〜112-3 を完了したため、指示どおり**ここで一旦停止**します。
次段（112-5 null test、silence/delta IR）に進むか、112-4 の A/B（DC/Tukey/phase/scale）を
形式的に実施するかの判断を仰ぎます。
