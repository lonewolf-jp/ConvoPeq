# WORK113-8 — Filter Application Architecture / Contract Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / 設計契約監査（production 変更 0）
- **前工程**: `doc/work113/correct_reference_rebaseline_7f0_20260918.md`（7F-0）
- **制約遵守**: 実装・修正は一切なし。113-7 block sweep / WORK114 は引き続き保留。

---

## A. 現行データフロー（convolver 側）

```text
IR（trim 後）
  ↓
partition ごとに FFT（2P フレーム, 0 詰め）
  ↓
applySpectrumFilter(FilterSpec)      ← ★ HC/LC を irFreqReal/Imag に直接乗算（周波数領域）
  ↓
irFreq（SoA, complexSize=2049）
  ↓  （partition 逆順化）
FDL × irFreq  → accum → interleave → IFFT → fftOutBuf[P:2P]
  ↓
2P-OLS（frame=[prev|cur], valid=[P:2P]）
  ↓
ringWrite → ringRead → Get  →（L1/L2 tail 加算）→ ConvolverProcessor 出力
```

## B. OutputFilter データフロー（AudioEngine 側・実在）

```text
ConvolverProcessor 出力（EQ と合成後）
  ↓
AudioEngine::outputFilter.process(block, convIsLast, state.convHCMode, state.convLCMode, state.eqLPFMode)
  ↓
convIsLast == true :  LC(HPF 18Hz) → HC0 → HC1 (LR4 LPF fc=22kHz@384k)   ←「① コンボルバー最終段」
convIsLast == false:  HPF(20Hz 固定) → LP0 → LP1 (fc=24kHz@384k)          ←「② EQ 最終段」
  ↓
outputMakeupGain → host output
```

- `outputFilter` は **AudioEngine のメンバ**（`AudioEngine.h:968`）。
- 呼び出し: `AudioEngine.Processing.DSPCoreDouble.cpp:460` / `DSPCoreFloat.cpp:360`（同一構造）。
- `convIsLast = convActive && (!eqActive || order == EQThenConvolver)`（`:458`）。

## C. FilterSpec ownership（誰が HC/LC を供給するか）

```text
UI: ConvolverControlPanel → engine.setConvHCFilterMode() / setConvLCFilterMode()
StateIO: state "convHCFilterMode" / "convLCFilterMode"
        ↓
AudioEngine atomics: convHCFilterMode / convLCFilterMode
        ├─(1)→ buildSnapshot.nucHCMode / nucLCMode
        │      → convo::FilterSpec.hcMode / lcMode      [LoaderThread.cpp:213-214, LoadPipeline.cpp:658-659, Lifecycle.cpp:303-304]
        │      → MKLNonUniformConvolver::applySpectrumFilter()   ← IR の周波数領域に焼き込み
        └─(2)→ state.convHCMode / convLCMode (AudioEngine.h:3992-3993)
               → OutputFilter::process(convIsLast=true)          ← 出力に時間領域 biquad で適用
```

→ **同じ HC/LC 設定が 2 経路に供給されている**（`convIsLast=true` では両方が動作）。

## D. 数学 reference（7F-0 の結果を 3 分離）

| ref | 定義 | 実測 50 Hz 振幅 | sideband |
|---|---|---|---|
| **R0** | raw convolution（フィルタ無し delta） | （Case A で clean, 0.25） | 無 |
| **R1** | true convolution × intended H（`h=IFFT(irFreq)` の線形畳み込み） | **0.2509** | **無**（137.5Hz=1.22e-6, 237.5Hz=3.37e-8） |
| **R2** | current partition-OLS × H（＝ NUC 出力と corr 1.0） | **0.14226** | **有**（137.5Hz=4.79e-2, 237.5Hz=2.77e-2） |

- 関係: `R2 == NUC`（corr 0.9999999999985, lag −2048）、`R1 clean`、`R1 != R2`（corr 0.883, 振幅比 0.5738）。

### 113-8-2: 50 Hz の理論ゲイン（production 係数生成を再利用せず独立計算）

| 量 | 値 |
|---|---|
| `H_LC(50Hz)`（2次 Butterworth HPF 18 Hz） | **0.991706** |
| `H_HC(50Hz)`（LR4 LPF 22 kHz, 2 段） | **1.000000** |
| `H_total(50Hz)` | **0.991706** |
| intended 50 Hz 振幅 = 0.25 × H_total | **0.247927** |
| 実測 R1（true convolution） | 0.2509（≈ intended ✓、delta の 0.95 スケール/測定誤差内） |
| 実測 R2（OLS/NUC） | 0.14226 → **intended の 0.5738×** |

→ **現在の OLS 出力は「意図したフィルタ特性そのもの」ではない**（別の演算になっている）。

## E. 結論

### E-1. FilterSpec の canonical な適用位置

**B（convolution kernel への焼き込み）ではなく C（コンボルバー出力の時間領域フィルタ）が canonical。**
根拠：
- `OutputFilter::process` の設計コメントが「convIsLast=true = ① コンボルバー最終段。処理順 LC→HC0→HC1」と明示。
- AudioEngine が `convIsLast` に応じて ①/② を切り替える構造そのものが「最終段で 1 回だけ適用する」設計。
- `FilterSpec` 側のコメントも「NUC は SetImpulse() 内で SoA に**周波数ゲイン**を直接適用する（Audio Thread の追加コストはゼロ）」と、**最適化のための代替実装**として書かれている。

### E-2. 現在の `applySpectrumFilter` は仕様上必要か

**不要であり、有害。**
- 望ましい特性は OutputFilter（LR4 LPF 22 kHz / 2次 HPF 18 Hz）が担う。
- 現在の NUC 実装は**同じ特性ではない別の曲線**：
  - HC: **raised-cosine ゲイン 1→0（22 kHz→Nyquist）**（線形位相、Nyquist で厳密 0）。LR4 biquad とは一致しない。
  - LC: `kEnd=round(8·4096/384000)=0`, `kStart=round(18·4096/384000)=0` → **DC bin のみ 0**（18 Hz HPF ではない）。
- さらに **partition の kernel 長が P を超える**ため 2P-OLS の valid-half 前提を破り、**block-rate 歪み**（n·187.5 ± f_in の側帯波、基本波 −4.9 dB）を生む（7F-0）。

### E-3. OutputFilter との関係

- **`convIsLast == true`（conv が最終段）: 二重適用**。同じ HC/LC 設定が IR（周波数領域・非等価な曲線）と出力（時間領域 biquad）の両方に適用される。
- **`convIsLast == false`（EQ が最終段）: NUC 側のみ**。OutputFilter は ② 分岐（HPF 20 Hz + LP 24 kHz）を適用し、LC/HC は適用しない。
  - 本 WORK113 の probe は `order=ConvolverThenEQ` かつ EQ 有効 → **convIsLast=false**。
    したがって **113-6〜7F0 の側帯波は NUC の `applySpectrumFilter` 単独で発生**しており、
    OutputFilter は関与していない（7C の FilterSpec ON/OFF 依存と整合）。
  - 一方、実製品で「conv が最終段」の構成では **二重適用**になる。

### 113-8-4 の確認項目

| # | 項目 | 結果 |
|---|---|---|
| 1 | OutputFilter::process が convolver 出力へ適用されているか | **適用されている**（convIsLast 経由） |
| 2 | convIsLast の供給元 | `convActive && (!eqActive \|\| order==EQThenConvolver)`（DSPCoreDouble:458） |
| 3 | applySpectrumFilter も同じ特性か | 同じ**設定**だが**別の曲線**（非等価） |
| 4 | 二重適用か | `convIsLast=true` で **二重**（異なる 2 曲線の直列） |
| 5 | HC/LC 設定の同一性 | 同一 atomics 由来。ただし実装が異なる |
| 6 | bypass 時 | **未検証**（要追加読解） |
| 7 | mode 変更時の reset/crossfade | **未検証**（`OutputFilter::reset` は `prepare` から呼ばれる。モード変更時の呼出は未確認） |
| 8 | stereo 状態独立性 | OutputFilter: `lcState[2] / hcState[2][2] / lpState[2][2] / hpfState[2]` でチャンネル独立 ✓。NUC: `nucConvolvers[0/1]` の 2 インスタンス ✓ |

### 113-8-5. 修正案 3 つの設計比較（**winner は決めない**）

| 案 | 概要 | 監査ポイント |
|---|---|---|
| A | IR 全体を時間領域でフィルタしてから partition 化 | IR 全体で 1 回だけ適用 → OLS 前提は「partition 化前の IR 長」で決まる。フィルタ後 IR が長くなる分だけ numPartsIR が増える（コスト増）。**contract 上は最も素直** |
| B | partition 用 FFT 長を拡張して安全化 | **数学条件（下記）を満たす必要があり、4P で必ず解消とは言えない**。メモリ・latency・既存 NUC 構造への影響大 |
| C | convolution 後に `OutputFilter` | 既存 API・canonical 設計と一致。ただし現行 `applySpectrumFilter` を外す前提が必要（二重適用の解消）。EQ 最終段との順序整合も要確認 |

### B の数学条件（導出）

```text
partition 長            = P
フィルタ IR 長（有効長） = L_f
partition にフィルタを適用した kernel 長 L_p ≤ P + L_f − 1
2P-OLS の valid-half [P,2P) が厳密に正しい条件:
      L_p ≤ P        ⇔   L_f ≤ 1
→ 非自明なフィルタ（L_f > 1）は、この構成では原理的に valid-half 前提を破る。

FFT 長を F（フレーム長 N=F, valid 長 = N/2）に拡張する場合の条件:
      N/2 ≥ L_p  かつ  N/2 ≥ block(=P)   →   N ≥ 2·max(P, P + L_f − 1) = 2(P + L_f − 1)
つまり L_f が tens〜hundreds samples なら N=8192（=4P）で足り得る。
ただし 7F-0 で観測した h の「全フレーム一定の floor（2.44e-4）」が
**減衰しない DC 起因のオフセット**である場合、有限の N では解消しない。
→ 「4P なら必ず解決」とは現時点では言えない。
```

## 成果物チェック（指示の A〜E）

- [x] A. 現行データフロー
- [x] B. OutputFilter データフロー
- [x] C. FilterSpec ownership（同じ設定が 2 経路へ供給）
- [x] D. 数学 reference R0/R1/R2（数値付き）
- [x] E. 結論 3 点（canonical 位置 / applySpectrumFilter の要否 / OutputFilter との関係）

## 保留（変更なし）
- 修正方針の設計（別 WORK）。113-7 block sweep。WORK114（resample 比 7.6241）。
- 未検証: OutputFilter の bypass 時挙動、mode 変更時 reset、`applySpectrumFilter` を外した場合の
  実製品（conv 最終段）での二重適用解消の確認。
