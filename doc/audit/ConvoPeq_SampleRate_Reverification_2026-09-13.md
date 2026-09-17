# ConvoPeq サンプルレート対応 厳密再検証報告

- **日付**: 2026-09-13（再検証ラウンド）
- **対象**: SR-01〜SR-05 の妥当性を、実効コード経路の追跡と文献で再判定
- **先行**: `doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md`

---

## 1. 判定サマリ（再検証後）

| ID | 初回 | **再検証後** | 理由 |
|----|------|--------------|------|
| SR-01 | High | **確認（High）** | 3s IR の無言打ち切りは複数経路で実在。OS 経由でも発生 |
| SR-02 | High | **確認（High）** | L0 が秒換算で縮む。192kHz/bs=256 から既に潰れる |
| SR-03 | Medium | **格上げ（High）** | 整合バッファ 2s 上限と、Convolver が申告し得る 2.7s+ の乖離 → クロスフェード破壊の可能性 |
| SR-04 | Medium | **確認（Medium）** | スライダー上限は常に 3.0s。hardMax と非同期 |
| SR-05 | Low | **維持（Low）** | 設計意図の可能性大。ただし 768kHz で HC がほぼ無意味 |

---

## 2. SR-01 — MAX_IR_LATENCY 不足（**確認・High**）

### 2.1 定数と上限

```250:257:src/ConvolverProcessor.h
static constexpr int MAX_IR_LATENCY = 2097152; // 2^21 (3.0s @ 384kHz = ~1.15M samples をカバー)
static constexpr int MAX_BLOCK_SIZE = 524288;
static constexpr int MAX_TOTAL_DELAY = MAX_IR_LATENCY + MAX_BLOCK_SIZE;
static constexpr int DELAY_BUFFER_SIZE = 4194304; // 2^22
```

```223:223:src/ConvolverProcessor.h
static constexpr float IR_LENGTH_MAX_SEC = 3.0f;
```

```1119:1119:src/audioengine/AudioEngine.h
static constexpr double SAFE_MAX_SAMPLE_RATE = 768000.0;
```

コメントは「3.0s @ 384kHz」まで。`SAFE_MAX=768kHz` と矛盾。

### 2.2 処理レート = デバイス SR × OS で判定すべき

OversamplingPolicy は処理レート上限 768 kHz。**768 kHz に到達する経路は 768 kHz デバイスだけではない。**

| デバイス SR | OS | 処理レート | hardMax = MAX_IR/proc | 3s 可能 |
|-------------|-----|------------|----------------------|---------|
| 44.1 k | 8 | 352.8 k | 5.94 s | ○ |
| 48 k | 8 | 384 k | 5.46 s | ○ |
| **88.2 k** | **8** | **705.6 k** | **2.97 s** | **×** |
| **96 k** | **8** | **768 k** | **2.73 s** | **×** |
| 176.4 k | 4 | 705.6 k | 2.97 s | × |
| **192 k** | **4** | **768 k** | **2.73 s** | **×** |
| 384 k | 2 | 768 k | 2.73 s | × |
| 705.6 k | 1 | 705.6 k | 2.97 s | × |
| **768 k** | **1** | **768 k** | **2.73 s** | **×** |

**96 kHz × 8× OS という通常の高音質設定でも 3.0 s IR が溢れる。**

### 2.3 経路別のクランプ整合

| 経路 | クランプ | 768 kHz 処理時 |
|------|----------|----------------|
| `setTargetIRLength` | `jlimit(MIN, hardMax=2.731s, t)` | **正しい** |
| `applyAutoDetectedIRLength` | 同上 hardMax | **正しい** |
| `analyzeImpulseResponseFile` | `hardMaxSec` + `exceedsHardLimit` 拒否 | **正しい**（ファイルロードは拒否） |
| **`applySnapshot`** | **`jlimit(MIN, IR_LENGTH_MAX_SEC=3.0, t)`** | **hardMax を使わない** |
| **`computeTargetIRLength`** | **`min(sr*sec, MAX_IR_LATENCY)`** | **無言キャップ** |
| **UI スライダー** | **`setRange(MIN, 3.0)`** | **hardMax と非同期** |

```162:164:src/convolver/ConvolverProcessor.StateAndUI.cpp
pendingOverride.targetIRLengthSec = juce::jlimit(IR_LENGTH_MIN_SEC,
                                                 IR_LENGTH_MAX_SEC,   // ← 3.0 固定
                                                 snapshot.targetIRLengthSec);
```

```945:949:src/convolver/ConvolverProcessor.StateAndUI.cpp
static constexpr int kMaxIRCap = MAX_IR_LATENCY;
int target = static_cast<int>(sampleRate * targetIRTimeSec);
target = (std::min)(target, kMaxIRCap);   // ← 無言
```

### 2.4 実害シナリオ

1. 48 kHz で 3.0 s IR をロードし状態保存
2. 768 kHz デバイス（または 96 kHz×8 OS）へ切替
3. `applySnapshot` が `targetIRLengthSec=3.0` を復元
4. `computeTargetIRLength` が 2,304,000 → 2,097,152 に **0.269 s を無言で打ち切り**
5. ユーザーへの警告なし。UI は 3.0 s のまま表示され得る

ファイルから直接ロードする経路は `exceedsHardLimit` で拒否されるが、**スナップショット復元・スライダー経路は警告なし。**

### 2.5 判定

**確認（High）**  
`MAX_IR_LATENCY` を `2^22=4,194,304`（768 kHz で 5.46 s）へ拡張するか、`applySnapshot`/UI を hardMax に揃え、`computeTargetIRLength` のキャップをユーザー可視化する。

---

## 3. SR-02 — tailStart と l0MaxLen（**確認・High**）

### 3.1 実装

```738:745:src/MKLNonUniformConvolver.cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l0MaxLen = kL0MaxParts * l0Part;  // 32 * l0Part
const int l0LenByTailStart = static_cast<int>(std::llround(tailStartSec * sampleRateForTail));
const int l0LenTarget = juce::jlimit(l0Part, l0MaxLen, l0LenByTailStart);
const int l0Len = std::min(irLen, tailEnabled ? l0LenTarget : l0MaxLen);
```

`kL0MaxParts=32` は**パーティション数**上限。秒換算すると SR に反比例する。

### 3.2 処理レート別の実効 L0（tailStart デフォルト 0.085 s）

| 処理レート | bs | l0Part | l0MaxLen | 0.085 s 要求 | 実効 L0 | 意図比 |
|------------|-----|--------|----------|--------------|---------|--------|
| 48 k | 256 | 256 | 8,192 | 4,080 | 4,080 | 100% |
| 96 k | 256 | 256 | 8,192 | 8,160 | 8,160 | 100% |
| **192 k** | **256** | 256 | 8,192 | **16,320** | **8,192** | **50%** |
| 192 k | 512 | 512 | 16,384 | 16,320 | 16,320 | 100% |
| **384 k** | **512** | 512 | 16,384 | **32,640** | **16,384** | **50%** |
| 384 k | 1024 | 1024 | 32,768 | 32,640 | 32,640 | 100% |
| **768 k** | **256** | 256 | 8,192 | **65,280** | **8,192** | **12.5%** |
| **768 k** | **512** | 512 | 16,384 | **65,280** | **16,384** | **25%** |
| **768 k** | **1024** | 1024 | 32,768 | **65,280** | **32,768** | **50%** |
| 768 k | 2048 | 2048 | 65,536 | 65,280 | 65,280 | 100% |

**低レイテンシ用途の小さい bs ほど、高 SR で L0 が短くなる。**

### 3.3 DSP 影響

- L0 は低遅延・高解像度層。L1 は `l0Part × 8` の大パーティション
- L0 が 85 ms から 10–40 ms に縮むと、**早期反射が L1 の分散 MAC に乗る**
- tailMode=AirAbsorption の L1/L2 ゲイン設計は「L0 が tailStart まで持つ」前提
- Gardner 型 NUC の文献では、先頭パーティションは最小ホップで早期反射を担う設計が標準

### 3.4 判定

**確認（High）**  
`l0MaxLen` を秒ベース（`ceil(tailStartMaxSec * sr / l0Part) * l0Part`、2 冪切り上げ）に変更するか、`kL0MaxParts` を SR 依存に拡大。

---

## 4. SR-03 — 整合バッファと申告遅延の乖離（**格上げ・High**）

### 4.1 定数

```2115:2116:src/audioengine/AudioEngine.h
static constexpr int kMaxLatencySamples = 1536000; // 最大2秒@768kHz対応
static constexpr int MAX_LATENCY_ALIGN_SAMPLES = 96000 * 2; // 2秒@48kHz
```

### 4.2 整合バッファ確保

```193:196:src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp
// 最大遅延（2秒上限・kMaxLatencySamples制限）
const int maxDelay = std::min(kMaxLatencySamples, static_cast<int>(safeSampleRate * 2.0));
const int requiredLatencyBufSize = maxDelay + bufferSize + 2;
```

768 kHz デバイスでは `min(1536000, 1536000) = 1,536,000`（2.00 s）。

### 4.3 Convolver が申告し得る遅延

```266:274:src/convolver/ConvolverProcessor.Runtime.cpp
const int algorithmLatency = conv->storedDirectHeadEnabled ? 0 : juce::jmax(0, conv->latency);
const int irPeakLatency = juce::jmin(rawIrPeakLatency, MAX_IR_LATENCY);
int totalLatency = static_cast<int>(std::min<int64_t>(calculatedLatency64, MAX_TOTAL_DELAY));
```

- `irPeakLatency` 上限 = `MAX_IR_LATENCY` = 2,097,152（**2.73 s @768 kHz**）
- `MAX_TOTAL_DELAY` = 2,621,440（**3.41 s @768 kHz**）
- H-02 未修正ならエネルギー重心がさらに押し上げ

### 4.4 PDC と整合バッファの乖離

| 項目 | 768 kHz での上限 |
|------|------------------|
| 整合バッファ `maxDelay` | **2.00 s** |
| Convolver `irPeakLatency` | **2.73 s** |
| Convolver `MAX_TOTAL_DELAY` | **3.41 s** |

`runLatencyAlignedCrossfadeMixLoop` は `wrapIdx(writePos - delay, bufferSize)` で読み出す。

```4232:4234:src/audioengine/AudioEngine.h
auto wrapIdx = [](int idx, int sz) { while (idx < 0) idx += sz; while (idx >= sz) idx -= sz; return idx; };
const int readOld = wrapIdx(writePos - delayOld, bufferSize);
```

`delayOld/delayNew` が `bufferSize` を超えると **リングが一周し、古い位置のサンプルを読んでしまう**（整合破壊）。  
`jassert(delayOld >= 0)` は負値のみ検査し、**上限超過は検出しない**。

### 4.5 判定

**格上げ（High）**  
`kMaxLatencySamples` を `MAX_TOTAL_DELAY` 以上（768 kHz で 3.5 s 級）に揃えるか、`latencyDelayOld/New` をバッファサイズで clamp し、超過を診断カウンタに入れる。

---

## 5. SR-04 — UI スライダー上限（**確認・Medium**）

```87:87:src/ConvolverControlPanel.cpp
irLengthSlider.setRange(IR_LENGTH_MIN_SEC, IR_LENGTH_MAX_SEC, 0.1);  // 0.5–3.0
```

```1309:1311:src/ConvolverControlPanel.cpp
const double irLengthSliderMax = std::max(IR_LENGTH_MAX_SEC, ...getTargetIRLength());
irLengthSlider.setRange(IR_LENGTH_MIN_SEC, irLengthSliderMax, 0.1);
```

- 上限は常に **3.0 s 以上**
- `hardMaxSec` は preview のみで、スライダー range には反映されない
- 768 kHz 処理でユーザーが 3.0 s を選んでも、内部は 2.73 s に落ちる（SR-01）

---

## 6. SR-05 — HC フィルタの 48 kHz 二値分岐（**維持・Low**）

```341:342:src/MKLNonUniformConvolver.cpp
const double hcFcStart = (fs <= 48000.0) ? 18000.0 : 22000.0;
const double hcFcEnd   = nyquist;
```

| SR | Nyquist | HC 開始 | 遷移帯 |
|----|---------|---------|--------|
| 44.1 k | 22.05 k | 18 k | 4.05 k |
| 96 k | 48 k | 22 k | 26 k |
| **768 k** | **384 k** | **22 k** | **362 k** |

R-1（OutputFilter fc）は音響設計値として確定済み。NUC 側も同系統。  
768 kHz では「超音波の緩やかな減衰」になり、**可聴帯域への影響はほぼない**。実害は小さいが、高 SR で HC を「有効」と表示する UI は誤解を招く。

---

## 7. 文献照合

### 7.1 Overlap-Save（Wikipedia / Rabiner & Gold）

- ホップ L、フィルタ長 M に対し FFT 長 N ≥ L+M−1（本実装は N=2L, M=L で適合）
- 出力は各ブロックの後半を連結 → **配列上 y[n] は n に置かれる**（H-01 と整合）
- 高 SR では M がサンプル数で増えるため、パーティション分割の必要性が上がる（本実装の NUC 設計は妥当）

### 7.2 高 SR 畳み込みの慣行

- 処理レート 768 kHz では 3 s IR = 2.3 M サンプル。2^21 上限は **384 kHz 時代の設計**
- 業界の高 SR プラグインは IR 長を**秒でクランプ**し、サンプル上限を SR から導出するのが一般的
- 本実装の `hardMaxSec = MAX_IR_LATENCY/sr` は正しい思想。**適用漏れが SR-01**

---

## 8. 優先修正（再検証後）

1. **SR-01** `MAX_IR_LATENCY` を `2^22` へ。`applySnapshot` を hardMax クランプに変更。`computeTargetIRLength` のキャップをログ/警告
2. **SR-03** `kMaxLatencySamples` を `MAX_TOTAL_DELAY` と一致させ、`latencyDelay*` にバッファ上限 clamp
3. **SR-02** `l0MaxLen` を秒ベース計算へ
4. **SR-04** スライダー上限を SR 依存 hardMax に
5. **H-02** 真のピークへ（SR-03 の悪化要因）

---

## 9. 推奨テスト

| テスト | 期待 |
|--------|------|
| 48 kHz で 3.0 s IR → 768 kHz へ切替 | 打ち切り警告、または完全保持 |
| 96 kHz × 8 OS で 3.0 s IR | 同上（**処理レート 768 kHz**） |
| 768 kHz / bs=512 / tailStart=0.085 | L0 長 = 85 ms 相当 |
| 768 kHz で 2.5 s IR（重心 irPeak 大） | クロスフェード整合バッファ超過なし |
| setLatencySamples | 2.73 s IR + 重心が 2.0 s 上限に当たらない |

---

## 10. 結論

- **SR-01 は 768 kHz デバイスだけでなく、96 kHz×8 OS など処理レート 705.6/768 kHz に到達する全経路で発生する。**
- **ファイルロードは hardMax で拒否されるが、スナップショット復元と computeTargetIRLength は無言キャップ。**
- **SR-02 は 192 kHz / bs=256 から既に顕在。** 低レイテンシ×高 SR で L0 が半減する。
- **SR-03 は再検証で格上げ。** 整合バッファ 2 s と Convolver 申告 2.7–3.4 s の乖離は、クロスフェード時のリング一周による整合破壊に直結し得る。

以上。
