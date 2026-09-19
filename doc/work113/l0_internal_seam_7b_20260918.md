# WORK113-7B — L0 Frequency/FDL Internal Seam Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（`NUPCTestAccess` へ読み取り専用 getter 追加 + standalone snapshot）
- **前工程**: `doc/work113/l0_temporal_continuity_20260918.md`（7A）, `doc/work113/nuc_standalone_isolation_20260918.md`（113-6 C1 PASS）
- **制約遵守**: FFT algorithm / FDL index / partition / `Add`/`Get` の意味 / ringWrite / block size /
  resampler / scale / limiter / softClip の**変更なし**。RT logging・RT allocation・RT file I/O なし。

```text
# WORK113-7B — L0 Frequency/FDL Internal Seam Audit（到達点）

Seam: NUPCTestAccess（friend・読み取り専用 getter のみ）に
      fftTimeBuf / fftOutBuf / fdlReal / fdlImag / irFreqReal / irFreqImag /
      fdlIndex / fdlMask / complexSize / fftSize / inputPos / baseFdlIdxSaved を追加。
      standalone harness（--buzz --nuc7b）から NonRT でスナップショット。production RT へは追加なし。

構成: sr=384000, block=P=2048, IR=delta(384k idx488), scale=1.0, L0 のみ, 40 block。
      ★ この試験では filterSpec=nullptr（出力周波数フィルタ無効）→ L0 IR は純 delta。
      numParts=4, numPartsIR=3（trace 実測）

結果（40 block すべて）:
  errIn  = 0.000e+00        … fftTimeBuf[P:2P] == 投入した x block（ビット厳密一致）
  errOut <= 3.053e-16       … fftOutBuf[P:2P] == frame を tap だけ遅延させた理論値（機械ε）
  SUMMARY errInMax=0.000e+00 errOutMax=3.053e-16

  fdlIndex 列: 1,2,3,0,1,2,3,0,... （mask=3, 4 block 周期で wrap、40 block で 10 回通過）
  mirror     = fdlIndex + numParts       （常に整合）
  linStart   = fdlIndex - numPartsIR + 1 + numParts （常に整合）
  → wrap（3→0）を跨いでも index 対応・論理履歴は連続。

判定（7B-6 表 / この構成に限る）:
  A Input assembly       … 異常なし（errIn = 0）
  B Forward FFT          … 異常なし（errOut が機械ε＝FFT→積算→IFFT が厳密）
  C FDL/index/mirror     … 異常なし（index 整合・wrap 連続）
  D IFFT / valid-half    … 異常なし（fftOutBuf[P:2P] が理論値と一致）

  137.5 / 237.5 / 325.0 Hz … この構成では **出現しない**（errOut ≤ 3e-16）

★ 重要な含意:
  7B の構成（filterSpec=nullptr）では L0 immediate path は**機械精度で正常**。
  一方 WORK113-6 では同じ L0 で側帯波が -6 dB で出ていた。
  両者の構成差は次の 2 点のみ:
      (i)  FilterSpec の有無（出力周波数フィルタ hcMode/lcMode）
      (ii) numPartsIR = 3（7B） vs 32（WORK113-6）
  → 側帯波は **コア OLA 機構（A/B/C/D）ではなく、FilterSpec 依存経路または numPartsIR 依存経路**
     から入る可能性が高い。

Confidence:
   no-filter 構成での A/B/C/D 正常性 … PROVEN（errIn=0, errOut≤3.05e-16）
   FDL index/wrap/linStart の整合     … PROVEN
   側帯波の発生源                      … OPEN（FilterSpec 依存経路 / numPartsIR=32）
```

---

## 1. 7B-1〜7B-5 の実施内容と結果

| 手順 | 測定 | 結果 |
|---|---|---|
| 7B-1 Forward FFT 入力 | `fftTimeBuf[P:2P]` と投入 block の差 | **errIn = 0.000e+00**（40 block 全て） |
| 7B-2 current FDL slot | `fdlIndex` / `mirror` / `linStart` | 全て整合（§2） |
| 7B-3 index / mirror / linStart | wrap 列 | `1,2,3,0,1,2,3,0,...`（mask=3）を 10 回通過、連続 |
| 7B-4 delta 1 項積算 | `fftOutBuf[P:2P]` vs `frame[tap 遅延]` | **errOut ≤ 3.053e-16**（機械ε） |
| 7B-5 IFFT / valid half | `fftOutBuf[P:2P]` が有効半か | **YES**（理論値と一致） |

```text
[7B] s50 b= 0 fdlIndex= 1 mask=3 mirror=5 linStart=3 numParts=4 numPartsIR=3 errIn=0.000e+00 errOut=1.665e-16
[7B] s50 b= 3 fdlIndex= 0 mask=3 mirror=4 linStart=2 numParts=4 numPartsIR=3 errIn=0.000e+00 errOut=2.220e-16
...
[7B] s50 SUMMARY errInMax=0.000e+00 errOutMax=3.053e-16
```

- `errOut` の理論値は「delta IR（tap 488, scale 1.0, フィルタ無し）」に対する
  `fftOutBuf[P+i] == fftTimeBuf[P+i-488]`。40 block すべてで機械ε以内。
- したがって **overlap-save の組立・FFT・FDL 読出・積算・IFFT・有効半選択**は、この構成で完全に正常。

## 2. FDL index 対応（7B-3）

`mask = numParts - 1 = 3`、`fdlIndex = (fdlIndex + 1) & 3` が Add ごとに進む。
`mirror = fdlIndex + numParts`、`linStart = fdlIndex - numPartsIR + 1 + numParts` は
全 40 block で上式どおりで、wrap（3→0）を跨いでも論理的な遅延履歴は連続。
→ **7A の結論（時間領域に段差なし）と整合**。

## 3. 未解決の焦点（次段）

WORK113-6（側帯波あり, -6 dB）と 7B（側帯波なし, ≤3e-16）の構成差は次の 2 点のみ。

1. **FilterSpec**: 113-6 は `{sampleRate=384000, tailEnabled=false, tailMode=2, hcMode/lcMode=Natural}`、
   7B は `nullptr`（出力フィルタ無効）。`HCMode`/`LCMode`（`OutputFilter.h`）は
   `SetImpulse` 内で `irFreq` に適用される。
2. **numPartsIR**: 113-6 = **32**（tailEnabled=false → `l0Len = l0MaxLen = 32*2048`）、7B = **3**。

### 次段の最小手順（内部 seam は既に用意済み）

1. 7B の harness に **WORK113-6 と同一の FilterSpec** を渡し、
   `fftOutBuf[P:2P]` を「harness 側で `irFreqReal/Imag`（accessor で取得）から再計算した値」と比較する。
   → フィルタの応答を知らなくても、NUC 内部の FFT×積算×IFFT が正しいかを直接判定できる。
2. 同じく `numPartsIR=32`（tailEnabled=false）で errIn/errOut を再測し、
   **errOut が機械εから外れる最初の block** を特定する。
3. 外れた場合、`linStart + p` の index 列と、非零でない `irFreq[p]`（フィルタ適用後に
   微小値を持つ partition）を突き合わせる。
   → フィルタ適用により **非零 partition が増える** 場合、
     「sum される項数」と「index 対応」が側帯波の発生条件になり得る。

> 注意: 7B では L0 の非零 IR partition は 1 個（delta）である前提だったが、
> フィルタ適用後は複数 partition に微小成分が生じる可能性がある。
> その場合 `numPartsIR` の増加が効く理由を説明できる。

## 4. 変更したファイル（test-only・読み取り専用のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/NUPCTestAccess.h` | 読み取り専用 getter 追加（fftTimeBuf/fftOutBuf/fdlReal/fdlImag/irFreqReal/irFreqImag/fdlIndex/fdlMask/complexSize/fftSize/inputPos/baseFdlIdxSaved）。setter なし |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc7b` と `runNuc7B()`（standalone snapshot） |

- FFT algorithm / FDL index / partition / ringWrite / block size / resampler / scale / limiter /
  softClip は**未変更**。production RT への追加なし。

## 5. 保留

- 上記 §3 の次段（FilterSpec 同一条件 + `irFreq` からの再計算比較）。
- 113-7 block-size sweep / WORK114（resample 比 7.6241）は引き続き未着手。
