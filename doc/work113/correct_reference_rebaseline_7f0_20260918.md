# WORK113-7F-0 — Correct Reference Re-baseline

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（測定側 reference の修正のみ。production 変更なし）
- **前工程**: `doc/work113/fdl_irfreq_ifft_interpretation_20260918.md`（7E）
- **制約遵守**: production / NUC / FilterSpec / FFT / OLS / ringWrite は**一切変更なし**。getter と CSV のみ。

```text
# WORK113-7F-0 — 判定: Case R0（NUC は reference と一致。真の LTI は clean）

条件: sr=384000 / P=2048 / FFT=4096 / IR=delta@488 / irLen=2048 / numPartsIR=1 / numParts=1 /
      FilterSpec=Natural,Nat / tail=OFF / scale=1 / 入力 sine50・sine40（4 s）

A. E（NUC Get 出力） vs R'（独立 OLS reference, h=IFFT(irFreq) を Python で構成）
   best lag = **-2048**, corr = **0.9999999999985**  → **E == R'（2048 sample の遅延差のみ）**

B. R' の sideband（独立に生成した reference 自身）
   f=50: 50Hz=1.4226e-01 | 187.5-50=4.7866e-02 | 187.5+50=2.7712e-02 | 375-50=2.0251e-02 | 375+50=1.5486e-02
   → **R' にも sideband が存在し、E の値と 5 桁まで一致**

C. 真の LTI 畳み込み（scipy.signal.fftconvolve(x, h)）— 対照
   f=50: 50Hz=**2.5090e-01** | 137.5Hz=**1.2231e-06** | 237.5Hz=**3.3740e-08**
   → **真の LTI は clean であり、基本波振幅も正しい（0.2509）**

D. OLS reference vs 真の畳み込み
   rmsDiff=9.455e-02, corr=0.883432  → **OLS 実装は真の畳み込みを再現していない**

E. カーネル h の形状
   |h[0..5]| = 2.44e-4（一定）, argmax|h| = 488
   energy: first 300 = 1.788e-05 / rest = 4.4645e-01（= 全体 4.4647e-01）
   → h は「488 の delta」ではなく **全 4096 sample にわたる ~2.44e-4 の floor を持つ**（HC/LC フィルタの効果）

★ 結論（Case R0）
   NUC は reference と**完全一致**（corr 1.0）。真の LTI 畳み込みは **clean**（sideband なし、振幅正常）。
   したがって 137.5/237.5/325 Hz は **NUC 実装の異常ではなく**、
   「**partition の spectrum を周波数領域でフィルタした結果、partition カーネルが P を超え、
   2P-OLS の valid-half 仮定が破れる**」ことで生じる **block-rate 歪み**である。
   → 7F-1 以降（ringWrite/Get 監査）は**中止**（指示どおり）。
```

---

## 1. これまでとの整合

| 観測 | 説明 |
|---|---|
| Case A（FilterSpec 無し）が clean | カーネルが delta（長さ≈1）で P 以内 → OLS 正常 |
| Case B/D（FilterSpec 有り）で sideband | HC/LC によりカーネルが全 4096 sample に広がる → OLS valid-half 破れ |
| numPartsIR=3 と 32 が同一 | 機構は「カーネル長 vs P」であり partition 数に依存しない |
| 周波数が n·187.5 ± f_in | 破れが**ブロック周期**で起きるため（187.5 Hz = 384000/2048） |
| delta IR でも実 IR でも同じ周波数 | ブロック率は IR に依存しない |
| zero-input で residual ゼロ | 入力に比例する歪み（ゼロ入力ならゼロ） |

## 2. 定量的な影響

- 基本波: 真値 2.509e-01 に対し NUC は 1.4226e-01（**-4.9 dB、0.567×**）→ フィルタ通過後の低域が過小。
- 側帯波: 137.5 Hz に基本波の 34%（= -9.4 dB）、237.5 Hz に 19%（-14 dB）。
  → **低域（50/40 Hz）再生時に、ブロック率で変調された粗いノイズ**として聴感上「ジジジ」に対応し得る。

## 3. 発生箇所（確定）

```text
FilterSpec（HC/LC の周波数ゲイン）
        ↓  applySpectrumFilter: irFreqReal/Imag に等価な実ゲインを乗算
irFreq（partition の 2P スペクトル）
        ↓  ← ★ ここで「partition のカーネル」が P を超える（floor が全 frame に広がる）
2P-OLS（frame=[prev|cur] → FFT × irFreq → IFFT → [P:2P] を valid とみなす）
        ↓  circular wrap が valid-half に混入（ブロック周期）
NUC 出力（= 独立 reference と corr 1.0）
```

- 7D: `irFreq` の生成は数学的に正常（corr 1.0）→ 生成側は無罪。
- 7E: FDL→accum→interleave→IFFT は数値一致 → 複素積/IFFT も無罪。
- 7F-0: **NUC == reference、真の LTI は clean** → 欠陥は
  **「partition 単位に周波数領域フィルタを適用したまま 2P-OLS で valid-half を切り出す」というアルゴリズム**
  自体にある（カーネル長 ≤ P の前提が、フィルタ適用により崩れる）。

## 4. 次段（修正はまだしない）

現時点で考えられる方向（**設計判断は次工程**。本 WORK では実装しない）:
- 出力フィルタの適用を **partition ではなく IR 全体（時間領域）** で行う、または
- フィルタ適用時の **FFT 長を 4P に拡張**してカーネル長 P を保証する、または
- HC/LC を **時間領域 biquad としてリング出力後**に適用する（`OutputFilter::process` の本来の設計）。

> 注: 7C の `irFreqNZ` は threshold 依存であったため、7D 以降は全 bin 実値で判定した。
> 7E の「厳密一致」表現は、interleave のみ 0.000e+00、他は rel ~1e-6 / max ~6e-7 の
> 「倍精度数値誤差範囲で一致」である（結論は変わらない）。

## 5. WORK113 仮説の再分類（指示に従い実施）

| # | 仮説 | 判定 |
|---|---|---|
| 1 | NUC 実装（FFT/FDL/複素積/IFFT）の異常 | **棄却**（7E: 数値一致、7F-0: corr 1.0） |
| 2 | FilterSpec の生成/CCS 表現の異常 | **棄却**（7D: corr 1.0、DC/Nyquist=0） |
| 3 | OLS/ringWrite/Get の実装異常 | **棄却**（NUC == reference） |
| 4 | **partition 単位の周波数領域フィルタ + 2P-OLS のカーネル長前提** | **★ 主因（新規・有力）** |
| 5 | 真の LTI 応答そのもの | **clean**（sideband なし） |

## 6. 変更したファイル（test-only・読み取り専用のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7f0`、`runNucStandalone` に E の CSV ダンプ（`7f0_E_<tag>.csv`）を追加 |
| （7B/7E で追加済）`src/tests/NUPCTestAccess.h` | 読み取り専用 getter 群 |

- production / NUC / FilterSpec / FFT / OLS / ringWrite は**未変更**。RT 追加なし。

## 7. 保留
- §4 の実装方向の設計判断（次工程）。113-7 block sweep、WORK114（resample 比 7.6241）。
