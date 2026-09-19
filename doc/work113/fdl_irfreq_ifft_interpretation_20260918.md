# WORK113-7E — FDL × irFreq / IFFT Interpretation Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（`NUPCTestAccess` getter + Python(NumPy) 独立照合）
- **前工程**: `doc/work113/filterspec_freqresponse_ccs_integrity_20260918.md`（7D）
- **制約遵守**: `accumulateSplitComplex` / `interleaveComplex` / FFT / FDL index / FilterSpec / OLS /
  ringWrite / production optimization は**一切変更なし**。

```text
# WORK113-7E — 判定: FDL / accum / interleave / IFFT はすべて厳密（PASS）

Baseline: sr=384000 / P=2048 / FFT=4096 / cs=2049 / IR=delta@488 / FilterSpec=Natural,Nat /
          tailEnabled=false / scale=1.0（7C/7D と同一）.
  Case B3: irLen=6144  → numParts=4  numPartsIR=3   （7C-B と同条件）
  Case P1: irLen=2048  → numParts=1  numPartsIR=1   （7E-7 単一 partition）

注: ダンプ時の fdlIndex は Add 後の値（+1 済み）であるため、直近の畳み込みの読み出し起点は
    linStart = fdlIndex - numPartsIR + numParts（= (fdlIndex-1) - numPartsIR + 1 + numParts）である。
    （当初 fdlIndex をそのまま使って 1 slot ずれ、indep=0 になった。補正後に全項目一致。）

| 検査 | 対象 | B3（npIR=3） | P1（npIR=1） |
|---|---|---|---|
| 7E-1 | FDL delay-0 slot vs `rfft(frame)` | **max 3.221e-04 / rel 5.33e-07 → PASS** | 同 → PASS |
| 7E-2 | `accumReal/Imag` vs 独立 Σ fdl×irFreq | **RE max 2.403e-04 / IM max 2.138e-04（rel ~1e-6）→ PASS** | 同 → PASS |
| 7E-3 | `accumBuf[2k]==Re`, `[2k+1]==Im` | **max 0.000e+00（厳密）→ PASS** | 同 → PASS |
| 7E-4 | `fftOutBuf` vs 独立 `irfft(accum)` | **scale 1.000001 / maxDiff 5.884e-07 / corr 1.00000000 → PASS** | 同 → PASS |
| 備考 | `\|accum\|max`（実測） vs 独立 | 244.8851 vs 244.8850 | 244.8851 vs 244.8850 |

→ **FDL → accum(複素積) → interleave → IFFT の全区間が数学的に正しい。**
   numPartsIR=1 でも 3 でも同一結果（partition 数の影響なし）。
   → 判定ツリーの「FDL PASS → accum PASS → interleave PASS → IFFT PASS」に到達。
     残る候補は **OLS/ringWrite/Get**、または **測定側（reference R）**。

Confidence:
   FDL/accum/interleave/IFFT の正常性 … PROVEN（rel 1e-6、interleave は厳密 0）
   残存候補（OLS/ring/Get または reference R）… OPEN
```

---

## 1. 実施内容（7E-0 / 7E-1 / 7E-2 / 7E-3 / 7E-4 / 7E-7）

`--buzz --nuc7e` は `runNuc7E(irLen, tag)` を 2 回（irLen=6144 → "B3"、2048 → "P1"）実行し、
2 block 分の Add/Get 後に、以下を CSV へダンプする（NonRT・読み取り専用）：

| ファイル | 内容 |
|---|---|
| `7e_frame_<tag>.csv` | `fftTimeBuf[0..4095]`（= [prevInput \| inputAcc]） |
| `7e_fdl_<tag>.csv` | `fdlReal/Imag` 全 2·numParts slot × 2049 bin |
| `7e_irfreq_<tag>.csv` | `irFreqReal/Imag` 全 numParts × 2049 bin |
| `7e_accum_<tag>.csv` | `accumReal[k]`, `accumImag[k]` |
| `7e_accumbuf_<tag>.csv` | `accumBuf[0..partStride-1]` |
| `7e_fftout_<tag>.csv` | `fftOutBuf[0..4095]` |

Python(NumPy) 側は production 実装を再利用せず、`np.fft.rfft` / 手書き複素積 / `np.fft.irfft` で独立計算。

## 2. 結果（上表の再掲・代表値）

```text
B3 linStart=3 delay0slot=5
   7E-1 slot vs rfft(frame): max=3.221e-04 rms=1.747e-05 (|X|max=604.503) rel=5.33e-07
   7E-2 accum: RE max=2.403e-04 rms=5.407e-06 | IM max=2.138e-04 rms=5.316e-06 | |act|max=244.8851 |indep|max=244.8850
   7E-3 interleave: max=0.000e+00
   7E-4 IFFT: scale=1.000001 maxDiff=5.884e-07 corr=1.00000000
P1 linStart=0 delay0slot=0
   （同一の数値）
```

- **7E-3 は厳密一致（0.000e+00）**。CCS の格納解釈（SoA → interleaved）は確定。
- **7E-4 は scale 1.0・corr 1.0**。IPP 逆変換（CCS→R）と `accum` の解釈は一致。
- **7E-2 は rel ~1e-6**。複素積 `Re=ARe·BRe−AIm·BIm / Im=ARe·BIm+AIm·BRe` と読み出し index
  `linStart+p`（p = 0..numPartsIR−1）は正しい。

## 3. 判定と、残された未解決の緊張（重要）

| 段 | 判定 |
|---|---|
| FDL input | **PASS** |
| accum | **PASS** |
| interleave | **PASS** |
| IFFT | **PASS** |
| OLS / ringWrite / Get | **未測定（残存候補）** |

**残された緊張**：
- 7E（2 block）では L0 経路が**厳密**である一方、7C の長区間 standalone では Case B に
  `E` residualRms = 4.84e-2（-6.36 dB）が出ていた。
- これは (a) OLS/ring の**ブロック間**組立の問題、または (b) 測定側の **reference 定義**の問題、
  のいずれかである。

**(b) の可能性を明示的に記録する**：7C/7D/7E と同じ standalone 解析では reference を
`R[n] = scale · x[n-488]`（**フィルタ無し delta の期待値**）としていた。
Case B/D の IR は実際には **HC テーパ + DC 除去**された LTI フィルタであり、`R` はその応答を含まない。
したがって Case B/D の `E−R` は「null test」ではなく
**「フィルタ済み応答 − 未フィルタ delta 応答」**を測っていた。
Case A（フィルタ無し）だけが `R` と整合し、clean だった事実とも符合する。

→ 7C の「FilterSpec ON で sideband」は、**reference の不整合**を反映している可能性がある。
  7E が示した「L0 経路は厳密」という事実はこの解釈を支持する。
  ただし `resPeaks = n·187.5 ± f_in` が delta/実IR で同一だった理由は
  「解析窓 131072 = 64 block（整数）」との整合で説明できるかを**次段で確認する必要がある**。

## 4. 次の作業（推奨）

1. **7E-5 の残り**：`ringWrite` のソースが `fftOutBuf[P:2P]` と同一かを直接確認（read 専用 getter で可）。
2. **reference の修正（測定側のみ）**：Case B/D の `E` と比較すべき正しい reference は
   `R'[n] = scale · (x ⊛ h_ir)[n]`（`h_ir = IFFT(irFreq)` を時間領域に持ったフィルタ）。
   `irFreq` は 7D で独立検証済みなので、Python 側で `h_ir` を作り `E` と比較すれば
   「L0 出力は正常か」を直接判定できる。
3. その結果 `E ≈ R'`（clean）なら、**側帯波は測定アーティファクトであり、buzz の主因ではない**。
   逆に `E ≠ R'` なら **OLS/ringWrite/Get** を 7F として監査する。

> 7E の禁止事項に従い、ここでは修正を一切行っていない。

## 5. 変更したファイル（test-only・読み取り専用のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/NUPCTestAccess.h` | `layerAccumReal/Imag/Buf/PartStride` の読み取り専用 getter を追加 |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7e` / `runNuc7E()`（FDL/accum/accumbuf/fftout/frame の CSV ダンプ） |

- `accumulateSplitComplex` / `interleaveComplex` / FFT / FDL index / FilterSpec / OLS / ringWrite は**未変更**。
  production RT への追加なし。

## 6. 保留
- §4 の 1〜3（特に reference 修正による `E` の直接判定）、113-7 block sweep、WORK114（resample 比 7.6241）。
