# Gardner Null Test 実施手順書

- **日付**: 2026-06-25
- **目的**: MKLNonUniformConvolver (NUC) の出力と理想逐次FFT畳み込みの差分を定量評価する
- **背景**: 静的解析ではレイヤ間時間整合性の誤差（Gardner理論）を確定できなかった。実測による決着が必要。

---

## 1. テスト概要

NUC の出力と、オーバーラップ無しの逐次FFT畳み込み（リファレンス）の出力を比較し、差分レベルを dBFS で評価する。

### 判定基準

| 差分レベル | 判定 | 意味 |
|-----------|------|------|
| < -90 dBFS | ✅ PASS | NUCは正確。DSPコアは無罪 |
| -70〜-90 dBFS | ⚠️ WARN | ほぼ正常。浮動小数点加算順序差の範囲 |
| -50〜-70 dBFS | 🔍 INVESTIGATE | Gardner誤差/分散MAC近似誤差の可能性 |
| > -50 dBFS | ❌ FAIL | 構造的誤差。レイヤ間時間整合性の問題を示唆 |

---

## 2. テスト方法A: 手動テスト（即日実施可能）

### 手順

1. **任意のIRファイルを用意**（例: 86msルーム補正IR、またはテスト用短IR）
2. **ConvoPeq を起動**し、該当IRをロード
3. **20-200Hz ログスイープ音源**を再生
4. **NUC通過後の出力を録音**（DAW等でキャプチャ）
5. **同じIRでリファレンス畳み込み**:
   - IRと同じ長さのFFTを使い、ゼロパディングで線形畳み込み
   - ツール例: `sox`, `ffmpeg`, `python`, Matlab, 等
6. **両者の差分をFFT分析**

### 必要なツール

```
# Windows版 sox でのリファレンス畳み込み例:
sox input.wav ir.wav reference_output.wav

# 差分FFT分析 (Python):
python -c "
import numpy as np
import scipy.io.wavfile as wav

rate_ref, ref = wav.read('reference_output.wav')
rate_nuc, nuc = wav.read('nuc_output.wav')
# アライメント (NUCはl0Partサンプルのレイテンシ)
align = 64  # l0Part
diff = nuc[align:] - ref[:len(nuc)-align]
rms_ref = np.sqrt(np.mean(ref**2))
rms_diff = np.sqrt(np.mean(diff**2))
db = 20 * np.log10(rms_diff / rms_ref)
print(f'RMS Error: {db:.1f} dBFS')
"
```

---

## 3. テスト方法B: 専用テストアプリ（高精度）

以下のC++コードを独立したテストアプリとしてビルドする。
JUCE非依存でMKL+IPPのみを使用。

### 3.1 設計方針

```
main()
  ├── generateTestIR()       // 疑似ルーム補正IR (2000 samples)
  ├── generateTestInput()    // 20-200Hz ログスイープ + インパルス
  ├── processNUC()           // MKLNonUniformConvolver で畳み込み
  ├── processReference()     // IPP FFT overlap-add でリファレンス
  └── computeError()         // RMS差 in dBFS
```

### 3.2 実装上の注意

- `MKLNonUniformConvolver.cpp` は `JuceHeader.h` に依存
- 回避策1: テスト専用の簡略 `JuceHeader.h` スタブを用意
- 回避策2: 既存の `ConvoPeq.exe` をヘッドレスモードで呼び出し、I/Oをファイル経由で行う
- **推奨**: 回避策2（既存バイナリのヘッドレス利用が最小工数）

### 3.3 スケルトンコード（参考）

```cpp
// リファレンス畳み込み (IPP FFT overlap-add)
void referenceConvolve(const double* ir, int irLen,
                       const double* input, int inputLen,
                       double* output) {
    int fftSize = 1;
    while (fftSize < irLen * 2) fftSize <<= 1;
    int order = 0;
    for (int t = fftSize; t > 1; t >>= 1) ++order;

    IppsFFTSpec_R_64f* spec = nullptr;
    int sizeSpec, sizeInit, sizeWork;
    ippsFFTGetSize_R_64f(order, IPP_FFT_DIV_INV_BY_N,
                         ippAlgHintFast, &sizeSpec, &sizeInit, &sizeWork);
    Ipp8u* specBuf = ippsMalloc_8u(sizeSpec);
    Ipp8u* initBuf = sizeInit > 0 ? ippsMalloc_8u(sizeInit) : nullptr;
    ippsFFTInit_R_64f(&spec, order, IPP_FFT_DIV_INV_BY_N,
                      ippAlgHintFast, specBuf, initBuf);
    if (initBuf) ippsFree(initBuf);
    Ipp8u* workBuf = sizeWork > 0 ? ippsMalloc_8u(sizeWork) : nullptr;

    std::vector<double> irPad(fftSize, 0.0);
    std::vector<double> blkPad(fftSize, 0.0);
    std::vector<double> freqIR(fftSize + 2, 0.0);
    std::vector<double> freqBlk(fftSize + 2, 0.0);
    std::vector<double> temp(fftSize, 0.0);
    std::vector<double> accum(inputLen + irLen, 0.0);

    memcpy(irPad.data(), ir, irLen * sizeof(double));
    ippsFFTFwd_RToCCS_64f(irPad.data(), freqIR.data(), spec, workBuf);
    int cplxSize = fftSize / 2 + 1;
    int blockSize = fftSize - irLen + 1;

    for (int pos = 0; pos < inputLen; pos += blockSize) {
        int copyLen = std::min(blockSize, inputLen - pos);
        memset(blkPad.data(), 0, fftSize * sizeof(double));
        memcpy(blkPad.data(), input + pos, copyLen * sizeof(double));
        ippsFFTFwd_RToCCS_64f(blkPad.data(), freqBlk.data(), spec, workBuf);
        for (int k = 0; k < cplxSize; ++k) {
            double re = freqBlk[2*k], im = freqBlk[2*k+1];
            freqBlk[2*k]   = re * freqIR[2*k] - im * freqIR[2*k+1];
            freqBlk[2*k+1] = re * freqIR[2*k+1] + im * freqIR[2*k];
        }
        memset(temp.data(), 0, fftSize * sizeof(double));
        ippsFFTInv_CCSToR_64f(freqBlk.data(), temp.data(), spec, workBuf);
        for (int i = 0; i < fftSize && pos + i < inputLen + irLen; ++i)
            accum[pos + i] += temp[i];
    }
    memcpy(output, accum.data(), (inputLen + irLen - 1) * sizeof(double));

    if (workBuf) ippsFree(workBuf);
    ippsFree(specBuf);
}
```

### 3.4 誤差計算

```cpp
double computeRMSErrorDB(const double* ref, const double* test, int len) {
    double sumRefSq = 0.0, sumDiffSq = 0.0;
    for (int i = 0; i < len; ++i) {
        sumRefSq += ref[i] * ref[i];
        double d = test[i] - ref[i];
        sumDiffSq += d * d;
    }
    if (sumRefSq < 1e-30) return -200.0;
    return 20.0 * log10(sqrt(sumDiffSq / len) / sqrt(sumRefSq / len));
}
```

---

## 4. テストパラメータ

| パラメータ | 値 | 備考 |
|-----------|-----|------|
| IR長 | 2000 samples (41.7ms @48kHz) | l0Part=64, l0Len=2048未満 → L0のみで処理される |
| IR長 (長) | 5000 samples (104.2ms @48kHz) | L0(2048) + L1(2952) の2層構成になる |
| ブロックサイズ | 64 samples | 標準的なオーディオコールバック |
| 入力信号 | インパルス + 20-200Hzログスイープ | 過渡応答 + 周波数応答の両方を評価 |
| サンプルレート | 48000 Hz | |

### 重要: 2つのIR長でテストする理由

- **IR長 2000**: l0Len(=jlimit(64, 2048, 4080)=2048)未満 → L0単層で動作。Gardner誤差は発生しない。
  - このテストで差分大なら、NUCのDSPコア自体に問題
- **IR長 5000**: L0(2048) + L1(2952) の2層構成。Gardner誤差が存在する場合に差分が増大する。
  - 2000との差分差が Gardner 誤差の指標

---

## 5. 期待される結果

| シナリオ | 期待値 | 判断 |
|---------|-------|------|
| IR=2000 (L0のみ) | < -90 dBFS | NUCのDSPコアは正確 |
| IR=5000 (L0+L1) | < -90 dBFS | Gardner誤差は無視できるレベル ✅ |
| IR=5000 (L0+L1) | -50〜-90 dBFS | Gardner誤差が存在する可能性。要追加解析 |
| IR=5000 (L0+L1) | > -50 dBFS | 構造的レイヤ間誤差。実装修正が必要 |

---

## 6. テスト実施後のアクション

### 結果が PASS (< -90 dBFS) の場合

- Gardner仮説は棄却
- NUC DSPコアは健全
- 原因の調査対象を以下に移行:
  1. ドライバ/ホスト要因（ASIOバッファ問題）
  2. 実機でのIR特有の数値的問題
  3. ConvoPeq の他の信号処理モジュール（EQ、ノイズシェイパー等）

### 結果が FAIL (> -50 dBFS) の場合

- Gardnerレイヤ間時間整合性の問題が確定
- 以下の修正を検討:
  1. Get() で L1/L2 出力に追加遅延を入れる
  2. リングバッファに L1 出力を遅延書き込みする機構
  3. tailOutputBuf の読み出し開始位置を l0Len だけずらす
