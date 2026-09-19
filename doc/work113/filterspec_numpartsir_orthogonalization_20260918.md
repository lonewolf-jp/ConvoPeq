# WORK113-7C — FilterSpec / numPartsIR 直交化監査

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（standalone NUC + `NUPCTestAccess` 読み取り専用 getter）
- **前工程**: `doc/work113/l0_internal_seam_7b_20260918.md`（7B）
- **制約遵守**: production fix / FilterSpec 修正 / FFT 修正 / FDL 修正 / block-size sweep / WORK114 は**未着手**。
  FilterSpec・FFT・FDL・partition・block size は**未変更**。

```text
# WORK113-7C — FilterSpec / numPartsIR 直交化（判定: C-1）

2×2 のうち **Case C（FilterSpec=nullptr + numPartsIR=32）は public API で構成不能**と判明:
  HCMode / LCMode に Disabled 値が存在しない（OrthogonalFilter.h: Sharp=0/Natural=1/Soft=2、LCMode: Natural=0/Soft=1）。
  FilterSpec=nullptr が唯一のフィルタ無効化手段だが、nullptr は tailMode=1/tailEnabled=true を強制し L0 が小さいまま。
  → 3 点（A / B / D）で直交化を実施。

Case  Filter     irLen   numPartsIR  numParts  irFreqNZ  irFreqPeak  irFreqEnergy  E 残留      sideband(137.5Hz)
A     none       192000      3           4         1        1.0         2.049e+03     3.558e-09   9.79e-18   ← 正常
B     Natural      6144      3           4         1        1.0         9.14375e+02   4.840e-02   4.787e-02  ← 再現
D     Natural     65536     32          32         1        1.0         9.14375e+02   4.840e-02   4.787e-02  ← 再現

判定:  A = 正常 / B = 異常（側帯波）/ D = 異常（側帯波）
       B と D は報告指標すべてで **一致**（残留 4.84000448e-02、sideband 4.78659316e-02 / 2.77119832e-02 / 2.02510135e-02、
       irFreqEnergy 914.375、D_blockAvgRms 1.09910243e-04）
       → **numPartsIR = 3 vs 32 の効果は無い（差ゼロ）**

→ ★ C-1: **FilterSpec dependency を局所化**（numPartsIR 依存は棄却）

first-order finding:
  FilterSpec ON により再現 / FilterSpec OFF では再現せず。
  （「FilterSpec が犯人」とはまだ断定しない。次段で irFreq の期待値照合を行う。）

Confidence:
  numPartsIR 非依存            … PROVEN（B ≡ D が全指標で一致）
  FilterSpec ON/OFF 依存       … PROVEN（A 3.56e-09 vs B/D 4.84e-02）
  FilterSpec 内部の発生箇所    … OPEN
```

---

## 1. 各ケースの実測

```text
[7C] A_none_np3   useSpec=0 irLen=192000 numParts=4  numPartsIR=3  irFreqNZ=1 irFreqTopP=2  irFreqPeak=1.000000e+00 irFreqEnergy=2.049000e+03
[7C] B_spec_np3   useSpec=1 irLen=6144   numParts=4  numPartsIR=3  irFreqNZ=1 irFreqTopP=2  irFreqPeak=1.000000e+00 irFreqEnergy=9.143750e+02
[7C] D_spec_np32  useSpec=1 irLen=65536  numParts=32 numPartsIR=32 irFreqNZ=1 irFreqTopP=31 irFreqPeak=1.000000e+00 irFreqEnergy=9.143750e+02
```

| Case | `E` residualRms | res/fund | sideband 137.5 | 237.5 | 325.0 | `D_blockAvgRms` | errIn/errOut（7B seam） |
|---|---|---|---|---|---|---|---|
| A | 3.558e-09 | -153.9 dB | 9.79e-18 | 9.64e-18 | 7.46e-18 | 4.41e-18 | 0 / ≤3e-16（7B） |
| B | **4.840e-02** | **-6.36 dB** | **4.787e-02** | **2.771e-02** | **2.025e-02** | 1.099e-04 | — |
| D | **4.840e-02** | **-6.36 dB** | **4.787e-02** | **2.771e-02** | **2.025e-02** | 1.099e-04 | — |

- **A**：`E−R` の residualRms = 5.68e-17（機械ノイズ）。側帯波は存在しない。
- **B / D**：`E` の residualRms = 4.840e-02、resPeaks = 134.8 / 140.6 / 240.2 / 234.4 / 322.3 Hz（bin 5.859 Hz）。
  `E−R` は `E` と同一（A と同じく R が清浄なため）。
- B と D は **`numPartsIR` が 3 と 32 で全く同じ数値**（差は `irFreqTopP` の 2 vs 31 のみ）。

## 2. `irFreq` partition 分布（7C-5）

- 3 ケースすべてで **`irFreqNZ = 1`**（非零 partition は 1 個のみ）。
  → filter は時間領域 delta を他 partition へ拡散させない。
- `irFreqPeak = 1.000000e+00` は不変。
- **`irFreqEnergy` のみ変化**：A 2.049e+03 → B/D 9.14375e+02（比 0.446、−3.5 dB）。
  → filter は非零 partition の**周波数応答の大きさを整形**している（静止・LTI のゲイン変化）。

## 3. 判定と解釈

| 判定分岐 | 条件 | 該当 |
|---|---|---|
| C-1 FilterSpec 単独 | A 正常 / B 側帯波 / C 正常 / D 側帯波 | **該当（C は構成不能、B/D で確認）** |
| C-2 numPartsIR 単独 | A 正常 / B 正常 / C 側帯波 / D 側帯波 | 棄却（B が既に異常） |
| C-3 interaction | A 正常 / B 正常 / C 正常 / D 側帯波 | 棄却（D ≡ B） |
| C-4 どれでもない | — | 棄却 |

- **numPartsIR（L0 topology）は無関係**であることが、B と D の完全一致で確定。
- 側帯波は **FilterSpec を有効にしたときだけ**発生する。

### 重要な未解決点（次段）

`irFreq` は **静止**（`irFreqNZ=1`, `irFreqPeak=1.0`, energy の一様スケール）であるのに、
FilterSpec ON で側帯波が出る。静止な LTI 応答では側帯波は生成できないため、
**FilterSpec 由来の周波数応答整形が CCS パッキング/エルミート対称性と整合していない**疑いが強い。

具体的な検証手順（次段）:
1. `irFreqReal[p]/irFreqImag[p]`（p = 非零 partition）を全 bin について保存し、
   **harness 側で独立に計算した「意図された LC(18Hz HPF) → HC0/HC1(LR4 LPF) の周波数応答 × delta のスペクトル」**
   と bin ごとに比較する（振幅・位相・DC/Nyquist bin の特別扱い）。
2. 特に **k=0（DC）と k=complexSize-1（Nyquist）** の扱い、
   および実数信号の CCS 表現で必要な共役対称性が保たれているかを確認する。
3. 保たれていない場合、その非対称が「各 block の FFT フレーム」との積で
   時間変動（= 側帯波）を生む経路を特定する。

## 4. 変更したファイル（test-only・読み取り専用のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7c`、`runNucStandalone(..., useSpec, irLenArg)`、`[7C]` irFreq partition 統計 |
| （7B で追加済）`src/tests/NUPCTestAccess.h` | 読み取り専用 getter 群（setter なし） |

- FilterSpec / FFT / FDL / partition / block size / resampler / scale / limiter / softClip は**未変更**。
  production RT への追加なし。

## 5. 保留

- §3 の `irFreq` 期待値照合（次段）。
- 113-7 block-size sweep / WORK114（resample 比 7.6241）は引き続き未着手。
