# ConvoPeq メモリ占有調査報告書（改訂版 v2）— 2.33GB 原因分析

---

**作成日**: 2026-07-08 (v2: NUC メモリ前提を全面修正)
**計測条件**: 192kHz SR, ×2 OS (384kHz), **IR長 25906 samples (135ms)**, 1024ブロック, ステレオ
**前版からの重大な修正**: NUC 1基あたり 333MB → **31MB** に下方修正

---

## 0. 重要な訂正

前回報告書の **NUC 1基 333MB** という値は、**5秒超の長尺IR** を前提としたものであり、**135ms IR では全く異なる**。

| 項目 | 前回（誤） | 今回（正） |
| :--- | :--- | :--- |
| NUC 1基あたり | 333 MB | **~31 MB** |
| StereoConvolver 1組 | 666 MB | **~62 MB** |
| 三重保持 合計 | 2,000 MB | **~186 MB (2.33GBの8%)** |

**結論**: NUC / StereoConvolver / AoS は 2.33GB の主原因ではない。

---

## 1. NUC メモリの正確な再計算

### パラメータ

- OS後 IR長: 51812 samples @384kHz (25906×2)
- L0 partSize: 2048, numParts: 32 → 65536 samples カバー (IR 完収)
- L1: 不要 (L0のみで完結)

### NUC 1基あたりのバッファ群（AoSスクラッチ化後）

| バッファ | 容量 |
| :--- | :--- |
| irFreqDomain (scratch) | 32 KB |
| fdlBuf (scratch) | 66 KB |
| irFreqReal / irFreqImag (SoA) | 1,049 KB |
| fdlReal / fdlImag (SoA) | 2,098 KB |
| fftTimeBuf / fftOutBuf | 66 KB |
| prevInputBuf / inputAccBuf | 33 KB |
| accumBuf / accumReal / accumImag | 66 KB |
| **NUC 1基 合計** | **~31 MB** |

**StereoConvolver (L+R): ~62 MB**
**三重保持しても: ~186 MB (2.33GBのわずか8%)**

---

## 2. 2.33GB の真の原因候補

NUC が 186MB しか消費しないなら、**残り ~2,144 MB は別の要因**。

### DSPCore 世代滞留説（最有力）

ログでは DSPCORE_PREPARE が **8回** 実行。各世代の DSPCore が解放されず滞留した場合:

| 項目 | 1世代 | 8世代累積 |
| :--- | :--- | :--- |
| EQ scratch/channel/msWork | ~80 MB | ~640 MB |
| DSPCore alignedL/R/dryBypass | ~17 MB | ~136 MB |
| NUC | ~31 MB | ~248 MB |
| NoiseShaper | ~61 MB | ~61 MB |
| **小計** | **~189 MB** | **~1,085 MB** |

### internalMaxBlock=524288 の連鎖

SAFE_MAX_BLOCK_SIZE=65536 × MAX_OS_FACTOR=8 = 524288。
このサイズのバッファが各 DSPCore 内に 15〜20 本生成される:

- 15本 × 524288 × 8 = 63 MB/世代
- 8世代滞留: ~500 MB

### NoiseShaper 学習バッファ

`bufferedSamples=3,840,000`: ~31 MB (mono) / ~62 MB (stereo)

---

## 3. 次に取るべき唯一のアクション

コード調査の結果、**既存のインスタンスカウント機構は存在しない** ことを確認。

**`liveCount` を追加し、どのオブジェクトが何個生存しているか実測する** ことが、2.33GB の正体を特定する唯一の方法。

### 追加推奨カウンタ

```cpp
// src/MKLNonUniformConvolver.h
static std::atomic<int> liveCount;

// src/ConvolverProcessor.h (StereoConvolver)
static std::atomic<int> liveCount;

// src/audioengine/AudioEngine.h (DSPCore)
static std::atomic<int> liveCount;
```

### 期待される診断結果

| 実測値 | 診断 |
| :--- | :--- |
| NUC=2, DSPCore=1, StereoConv=1 | 正常。NUC以外が主原因 → 524288バッファ群を調査 |
| NUC=6, DSPCore=3, StereoConv=3 | 三重保持（前回仮説）。しかし186MBしか説明できず |
| NUC=2, DSPCore=8, StereoConv=1 | DSPCore 世代滞留が主因。EQ/alignバッファが累積 |

---

## 4. 結論

1. **AoS スクラッチ化は正常動作。NUC 1基は約 31MB（5秒IR前提の333MBではない）**
2. **NUC / StereoConvolver は 2.33GB の主原因ではない**
3. **最も疑わしいのは DSPCore 世代滞留による EQ/align バッファの累積**
4. **唯一の確定方法は liveCount インスタンスカウンタの実装と実測**
