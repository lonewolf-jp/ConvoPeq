# WORK113-7A — L0 Temporal Continuity Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（standalone NUC + block 同期平均。内部 seam は未実装）
- **前工程**: `doc/work113/nuc_standalone_isolation_20260918.md`（C1 PASS / C2 FAIL）
- **正本**: `ConvoPeq.md` 5,276,742 B / 2026-09-18 22:33:38（HEAD `0654e7b5`）
- **制約遵守**: algorithm / index / ringWrite / FFT / partition / block size の**変更なし**。
  production logging なし。RT buffer allocation / RT file I/O なし。

```text
# WORK113-7A — L0 Temporal Continuity Audit（到達点）

手法: AudioEngine 非経由の standalone NUC（delta IR, 384k, block 2048, L0 のみ）に対し、
      D[n] = E[n] - R[n]（R[n] = x[n-488]）を算出し、
      **block 同期平均** D_avg[phase] = mean_b D[b*P + phase]（P=2048, b=0..749）を取った。
      入力周波数が block 周期と非整数周期（50Hz: 4/15 cycle/block）なので、
      正弦成分は多数 block 平均で減衰し、block 周期に**同期した**成分だけが残る。

1) DC（定数入力 1.0）— 境界の時間連続性
   E_blockAvgRms = 6.665e-04（低域カットで DC はほぼ除去）
   D_blockAvgRms = 9.993e-01（D ≈ -1.0 = 0 - R）
   D_avg[P-1] = -9.99333333e-01 , D_avg[0] = -9.99333333e-01
   **jump = 0.00000000e+00**（完全に 0）
   → block 境界に時間領域の段差は**存在しない**（入力組立・ringWrite とも）

2) sine50 / sine40 — 側帯波の再確認と block 同期成分
   sine50: D_blockAvgRms = 1.09910243e-04 / residualRms = 4.840e-02  → 比 **0.23%**
           D_avg[P-1]=-1.099098e-04, D_avg[0]=-1.099103e-04, **jump=-4.10e-10**
           D_avg[0..7] は -1.099103e-04 でほぼ完全に平坦
           E_blockAvgRms = 1.000e-04（E の基本波 1.42e-01 に対し 0.07%）
           sidebands: 137.5Hz:4.787e-02 / 237.5:2.771e-02 / 325.0:2.025e-02（E, E-R とも）
   sine40: D_blockAvgRms = 9.599e-05 / jump=-7.35e-10、sidebands 147.5:4.564e-02 等（同様）

判定:
   A  Input assembly（時間領域境界）  … **異常なし**（DC で jump=0、正弦でも平坦）
   D  IFFT/ringWrite（時間領域境界）  … **異常なし**（同上）
   B  Forward FFT / 周波数領域        … **未測定（内部 seam 未実装）**
   C  FDL/index（時間写像）           … **未測定（内部 seam 未実装）**

   → 「per-block の時間領域段差（comb）」仮説は **不支持**。
      block rate 187.5 Hz の側帯波は、**block 同期した時間波形を持たない**形で存在する。

Confidence:
   DC 境界に段差なし                       … PROVEN（jump = 0.00000000e+00）
   block 同期成分は residual の 0.23% のみ  … PROVEN
   B/C の関与                              … OPEN（内部 seam が必要）
```

---

## 1. 実施内容

`--buzz --nuc6=<freq>`（sine）と `--buzz --nuc7a`（DC）で standalone NUC を駆動し、
`E`, `R`, `D=E−R` を出力、`[NUC7A]` で block 同期平均を記録した。

```text
[NUC6]  sine50 sr=384000 block=2048 irLen=192000 tap=488 blockRate=187.5Hz fIn=50.0 dcMode=0
[NUC7A] sine50 blocks=750 D_blockAvgRms=1.09910243e-04 D_avg[last]=-1.09909840e-04 D_avg[0]=-1.09910251e-04 jump=-4.10402815e-10
[NUC7A] sine50 D_avg[0..7]= -1.099103e-04 ×8
[NUC7A] sine50 D_avg[P-4..P-1]= -1.099103e-04 -1.099103e-04 -1.099099e-04 -1.099098e-04  E_blockAvgRms=1.00003009e-04
[NUC6_SIDEBAND] sine50 n=1 : 137.5Hz E=4.78659316e-02 R=1.21809299e-06 D=4.78658267e-02 | 237.5Hz E=2.77119832e-02 R=3.42367045e-08 D=2.77119808e-02
[NUC6_SIDEBAND] sine50 n=2 : 325.0Hz E=2.02510135e-02 R=3.92982247e-08 D=2.02510100e-02 | 425.0Hz E=1.54861207e-02 R=7.56948386e-09 D=1.54861202e-02

[NUC7A] dc7a blocks=750 D_blockAvgRms=9.99333333e-01 D_avg[last]=-9.99333333e-01 D_avg[0]=-9.99333333e-01 jump=0.00000000e+00
[NUC7A] dc7a D_avg[0..7]= -9.993333e-01 ×8
[NUC7A] dc7a D_avg[P-4..P-1]= -9.993333e-01 ×4  E_blockAvgRms=6.66521158e-04
```

## 2. 解釈

### 2-1. DC 入力（最重要の対照）
- 定数入力に対し `D_avg` は −0.9993 で**完全に平坦**、`jump = 0.00000000e+00`（厳密に 0）。
- これは「出力が block ごとに段差・欠落・ゼロ埋めを起こしていない」ことの**直接証明**。
- `E_blockAvgRms = 6.7e-04` は低域カット（`LCMode::Natural`, fc≈18 Hz）による DC 除去の残り。
- → **入力組立（prevInputBuf/inputAccBuf/fftTimeBuf）と ringWrite は時間連続**。

### 2-2. 正弦入力
- `D_avg` は平坦、`jump ≈ 4e-10`（DC の 2e-10 と同水準）。
- block 同期成分 `D_blockAvgRms = 1.10e-04` は residual（`4.84e-02`）の **0.23%**。
- 一方で側帯波（137.5/237.5/325.0 Hz）は residual の**支配成分**として健在。
- → **側帯波は「block 境界の時間段差」からは生じていない。**
  周波数軸に `n·187.5 ± f_in` が立つのに、時間軸に block 同期成分がほぼ無い。

### 2-3. 注意（推定器の限界）
- 50 Hz は block 周期に対し 4/15 cycle/block（15 block で位相が厳密に繰り返す）。
  したがって block 同期平均は 50 Hz 成分を**完全には消さない**（残差 −1.10e-4 の主因はこれ）。
- それでも振幅は residual の 0.23% であり、「境界段差」を支持するには小さすぎる。

## 3. 判定（7A-6 の表）

| 境界 | 異常あり | 異常なし |
|---|---|---|
| A Input assembly | — | **✓（時間領域）** |
| B Forward FFT | 未測定（seam 必要） | — |
| C FDL / index | 未測定（seam 必要） | — |
| D IFFT / ringWrite | — | **✓（時間領域）** |

- **側帯波が「その段の出力に初めて出現したか」**：時間領域では A/D を通過しても comb は現れない。
  周波数領域（B/C）での出現有無は**未測定**。
- したがって現時点の最有力は **C1-B（Forward FFT / 周波数領域の位相回転・スケーリング）
  または C1-C（FDL/index の時間写像）** であり、いずれも内部 seam を必要とする。

## 4. 次段（内部 seam の設計・未実装）

7A-2〜7A-5 を実施するには、standalone NUC から以下を NonRT で観測する seam が必要
（production RT へは追加しない。standalone 専用の読み出し API を test-only で用意する案）：

1. **B**: `fftTimeBuf` と `currentFDLSlot` をブロックごとにコピーし、harness 側の独立 FFT と比較（maxAbsDiff/rmsDiff/phaseDiff）。
2. **C**: `fdlIndex`, `mirrorIndex`, `linStart`, `index(p)` をブロックごとに記録（CSV）し、wrap（30→31→0→1）を跨ぐ論理履歴の連続性を確認。
3. **C/7A-4**: delta では `numPartsIR=32` だが非零 partition は 1（slot=31）。積算は 1 項に縮退するため、
   `FDL[linStart+31]` と `irFreq[31]` の積そのものを観測すれば足りる。
4. **D**: `fftOutBuf[0:P]` と `fftOutBuf[P:2P]` を記録し、`ringWrite` が後半を渡しているか、
   overlap-save の有効半が正しいかを確認（delta では理想出力 `x[n-488]` が既知）。

> これらは NUC の private メンバを読む必要があるため、**standalone 専用の
> NonRT 読み出し API（test-only）** を追加する設計が妥当。
> production の RT 経路・アルゴリズムには一切触れない。

## 5. 変更したファイル（test-only のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7a`（DC 入力）, block 同期平均 `[NUC7A]`、`runNucStandalone` の DC 対応 |

- production / NUC algorithm / partition / FDL / block size / resampler / scale / limiter / softClip /
  smoothing / wet-dry / OS 補償 は**未変更**。RT 追加なし。

## 6. 保留

- 7A-2〜7A-5（内部 seam による B/C/D の直接観測）。
- 113-7 block-size sweep と WORK114（resample 比 7.6241）は引き続き未着手。
