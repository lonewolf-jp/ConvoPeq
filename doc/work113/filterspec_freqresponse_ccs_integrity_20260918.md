# WORK113-7D — FilterSpec Frequency-Response / CCS Integrity Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（`NUPCTestAccess` getter + Python(NumPy) 独立照合）
- **前工程**: `doc/work113/filterspec_numpartsir_orthogonalization_20260918.md`（7C）
- **制約遵守**: FilterSpec / FFT / FDL / partition / block size / resampler / scale / limiter は**未変更**。
  113-7 block sweep・WORK114 は未着手。

```text
# WORK113-7D — 判定: Case D1（generation は正常 / use 側が問題）

Baseline（7C-B 再現）:
  sr=384000 / P=2048 / IR=delta@488 / irLen=6144 / FilterSpec=Natural,Nat / tailEnabled=false / scale=1.0
  → [7C] useSpec=1 numPartsIR=3 irFreqNZ=1 irFreqTopP=2 irFreqPeak=1.0 irFreqEnergy=9.143750e+02
    [7D] dump=irfreq_7d_B.csv topP=2 partSize=2048 fftSize=4096 complexSize=2049

独立期待値（NumPy・production の係数生成を再利用せず仕様から再計算）:
  HC Natural: kS=round(22000*4096/384000)=235, kE=min(2048, Nyquist=2048)=2048
              g[k]=1 (k<=235) / 0.5*(1+cos(pi*(k-235)/1813)) (235<k<=2048)
  LC Natural: kEnd=round(8*4096/384000)=0, kStart=round(18*4096/384000)=0 → g[0]=0 のみ
  X_delta[k]=exp(-j*2*pi*k*488/4096)
  expected[k] = X_delta[k] * g[k]

actual vs expected:
  DC (k=0) imaginary          = 0.0        → PASS
  Nyquist (k=2048) imaginary  = 0.0        → PASS
  conjugate symmetry          = 実数FIRのCCS表現として整合（half-spectrum が expected と一致）→ PASS
  maxAbsDiff                  = 6.918e-07  （rel 6.9e-7、|act|max 1.000001 vs |exp|max 1.000000）
  rmsDiff                     = 3.439e-07
  corr_coef                   = 1.00000000
  full-bin error              → PASS（振幅・位相とも期待値と一致、bin 置換/符号反転なし）
  irFreq finite               → PASS（nonfinite=0）

追加検査（OLS 長の妥当性）:
  フィルタ後カーネル h = IFFT(expected) を 2P=4096 で評価
    h[0]=5.571e-01, h[1]=2.593e-01, h[2]=-2.49e-02, h[5]=-1.91e-03, h[10]=-1.52e-04 …
  ±P/2 窓内 energy = 0.999727（窓外 2.7e-4）
  → **カーネルは十分短く、OLS の有効半仮定は破れていない**（「カーネルが P を超えて wrap」仮説は棄却）

独立 LTI 出力の sideband:
  expected カーネルでの線形畳み込み（sine50/40）→ 137.5/237.5/325 は生成されない（LTI のため）
  NUC output sideband → 有（-6.36 dB）

★ 判定: Case D1
   irFreq の生成・表現（FilterSpec → CCS → irFreq）は**数学的に正常**。
   側帯波は **irFreq を convolution 側で使用する段（FDL×irFreq / interleave / IFFT / OLS 組立）**で発生。

判別手がかり（7E へ引き継ぎ）:
  A（FilterSpec 無し）: |irFreq| = 1（純位相ランプ）        → NUC 出力 機械ε で clean
  B/D（FilterSpec 有り）: |irFreq| = g（振幅テーパ、g[0]=0） → NUC 出力 -6.36 dB で sideband
  → 差は「|irFreq| がフラットでないこと」のみ。
    振幅整形されたスペクトルを扱う経路（複素積算・interleave・IFFT・OLS 組立）を 7E で監査する。
```

---

## 7D-8 最終判定表

| 検査 | 結果 |
|---|---|
| `irFreq` finite | **PASS** |
| DC imaginary | **PASS**（0.0） |
| Nyquist imaginary | **PASS**（0.0） |
| conjugate symmetry | **PASS**（CCS half-spectrum が expected と一致） |
| magnitude vs expected | **PASS**（max 6.9e-7） |
| phase vs expected | **PASS**（corr 1.00000000） |
| full-bin error | **6.918e-07 max / 3.439e-07 rms** |
| フィルタ後カーネル長（OLS 妥当性） | **PASS**（±P/2 窓内 99.973%） |
| independent LTI output sideband | **無**（LTI のため原理的に出ない） |
| NUC output sideband | **有**（-6.36 dB） |

### どこまで正常か

```text
FilterSpec generation        … PASS（仕様から独立再計算した g[k] と一致）
        ↓
irFreq representation (CCS)  … PASS（DC/Nyquist imag=0、corr=1.0、bin/符号異常なし）
        ↓
FDL × irFreq                 … ← ここ以降を 7E で監査（未測定）
        ↓
interleave / IFFT            … 未測定
        ↓
NUC output                   … sideband 有（-6.36 dB）
```

## 7D-2 に記録した独立計算の仕様（production 係数生成を再利用せず）

| 項目 | 値 |
|---|---|
| FFT | N = fftSize = 4096（partSize=2048 の 2P） |
| complexSize | 2049（CCS half-spectrum、k=0..N/2） |
| bin→Hz | k × fs/N = k × 93.75 Hz |
| HC (`HCMode::Natural`) | fc_start=22 kHz（fs>48k）、fc_end=Nyquist。gain=raised-cosine 0.5(1+cos(πx)), x=(k−235)/1813 |
| LC (`LCMode::Natural`) | fc_end=8 Hz, fc_start=18 Hz。N=4096 では round() により両者 bin 0 → **g[0]=0 のみ** |
| delta | 物理 partition 0 内 index 488 → X_delta[k]=exp(−j2πk·488/4096) |
| 格納 p と物理 partition の対応 | 逆順化により 物理 p0 → 格納 slot `numPartsIR−1`（本例 topP=2） |

## 次の作業（7D の結論より）

**WORK113-7E — irFreq / FDL complex-multiply interpretation audit**
対象（この順に）:
1. `accumulateSplitComplex(fdl[linStart+p], irFreq[p])` の split-complex 期待と実装の照合
2. `interleaveComplex` → `accumBuf` の並び（CCS 復元）
3. `processLayerInv`（IFFT）後の `fftOutBuf[P:2P]` と、harness 側で
   `IFFT(FFT(frame) × irFreq)` を独立計算した値の比較
4. 「|irFreq| が非フラット」のときだけ差が出るかを 1 点で直接確認
   （例: g を低域のみ 0.5 倍した合成 irFreq を作り、同じ残差が出るか）

> 注: 7C のコメント「`irFreqNZ=1` → partition を拡散していない」は threshold 依存のため、
> 本 7D では **全 bin の実値**を保存して判定した（拡散していないことを全 bin で確認）。

## 変更したファイル（test-only・読み取り専用のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7d`（Case B の irFreq 全 bin を `irfreq_7d_B.csv` へダンプ） |
| （7B 追加済）`src/tests/NUPCTestAccess.h` | 読み取り専用 getter 群 |

- FilterSpec / FFT / FDL / partition / block size / resampler / scale / limiter / softClip は**未変更**。
  production RT への追加なし。

## 保留
- 113-7E（上記）、113-7 block sweep、WORK114（resample 比 7.6241）。
