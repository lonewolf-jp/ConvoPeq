# BUG-034: IPP FFT 関数の戻り値が未チェック — 無言でガページデータが伝搬

## 発見日
2026-07-26

## ファイル
`src/MKLNonUniformConvolver.cpp:1043, 1060, 1376, 1436, 1570, 1637, 1750`

## 問題
`ippsFFTFwd_RToCCS_64f()` および `ippsFFTInv_CCSToR_64f()` の戻り値
（`IppStatus`）が全くチェックされていない。

```cpp
// line 1043 (IR周波数変換)
ippsFFTFwd_RToCCS_64f(irL + p * partSize, tempFreq, partSize, pFFTSpec);
// 戻り値 ippStsNoErr 以外 → tempFreq にゴミデータ！
memcpy(irFreqDomain, tempFreq, fftWorkSize * sizeof(double));

// line 1570 (オーディオ処理)
ippsFFTInv_CCSToR_64f(accumBuf, fftOutBuf, fftSize, pFFTSpec);
// 戻り値 ippStsNoErr 以外 → fftOutBuf にゴミデータ → 出力リングバッファへ書き込み！
```

### 影響
FFT が失敗する原因（`fftSpec == nullptr`、メモリ破損、アライメント違反）:
- **line 1043**: IR ロード時に IR 周波数領域データが破損 → 誤ったインパルス応答が永久に保持される
- **line 1060**: IR 反転時 — 同上
- **line 1376**: オーディオ処理中の FDL 変換 → 1ブロックの出力破損
- **line 1436/1570**: FFT→IFFT チェーン → accumBuf のゴミが出力リングバッファに書き込まれる → **可聴なノイズ/クラックル**
- **line 1637/1750**: 同様

### 根本原因
全 7 箇所の IPP FFT 呼び出しで戻り値が無視されている。`pFFTSpec` が
何らかの理由で無効になっても、エラーは検出されず処理が続行される。

### リスク評価
- **重大度**: CRITICAL — エラー時に無言で出力破損（可聴ノイズ）
- **発生頻度**: 低（通常は FFT は成功するが、メモリ圧縮時や不整合時に発生しうる）
- **検出性**: 難（通常ノイズと区別がつかない）

### 修正方針
全 FFT 呼び出しで `IppStatus` をチェックし、失敗時は出力バッファを
ゼロクリアする:
```cpp
IppStatus status = ippsFFTFwd_RToCCS_64f(..., pFFTSpec);
if (status != ippStsNoErr) {
    memset(tempFreq, 0, fftWorkSize * sizeof(double));
}
```
