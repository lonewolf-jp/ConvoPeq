# ConvoPeq サンプルレート対応（44.1 kHz–768 kHz）検証報告

- **日付**: 2026-09-13
- **前提**: 内部処理レート 44.1 kHz–768 kHz 対応が必要
- **先行**: `doc/audit/ConvoPeq_Bug_Verification_2026-09-13.md`

---

## 0. 結論サマリ

| ID | 重大度 | 内容 | 影響レート |
|----|--------|------|------------|
| **SR-01** | **High** | `MAX_IR_LATENCY` が 768 kHz の 3 s IR に不足 | 705.6 / 768 kHz |
| **SR-02** | **High** | `tailStart` が `l0MaxLen` で潰れ L0/L1/L2 分割が歪む | 384 kHz 超（bs 依存） |
| **SR-03** | **Medium** | AudioEngine `kMaxLatencySamples` と Convolver `MAX_TOTAL_DELAY` の不整合 | 768 kHz |
| **SR-04** | **Medium** | UI IR スライダーが hardMax を無視して 3.0 s を許容 | 768 kHz |
| **SR-05** | **Low** | HC フィルタ閾値が 48 kHz 二値のみ。超高域で遷移帯が巨大 | 192 kHz 超 |
| **SR-06** | **Info** | OversamplingPolicy / prepare ガードは妥当 | 全域 |
| **SR-07** | **Info** | 768 kHz + 長 IR のメモリは L2 SoA で ~100 MB/ch 級 | 768 kHz |

既知バグ H-02（エネルギー重心）は **サンプルレートに比例して誤差が拡大**する。

---

## 1. SR-01 — MAX_IR_LATENCY が 768 kHz で不足（**High**）

### 1.1 定数

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

### 1.2 数値

| SR | 3 s IR [samples] | MAX_IR_LATENCY | hardMax [s] | 3 s 可能 |
|----|------------------|----------------|-------------|----------|
| 384 kHz | 1,152,000 | 2,097,152 | 5.46 | ○ |
| 705.6 kHz | 2,116,800 | 2,097,152 | 2.98 | **×** |
| **768 kHz** | **2,304,000** | 2,097,152 | **2.731** | **×** |

コメントは「3.0s @ 384kHz」まで想定。`SAFE_MAX_SAMPLE_RATE=768000` と矛盾。

### 1.3 実害

- `computeTargetIRLength` は `min(sr*sec, MAX_IR_LATENCY)` で **2.731 s に打ち切り**（`StateAndUI.cpp:945-949`）
- `getMaximumAllowedIRLengthSecForSampleRate` は 2.731 s を返す（UI preview の hardMax）
- **IR_LENGTH_MAX_SEC=3.0 のスライダーは 768 kHz でも 3.0 s を許容**（SR-04）
- ユーザーが 3.0 s 指定すると **約 0.27 s のテールが無言で失われる**

### 1.4 修正案

- `MAX_IR_LATENCY` を `2^22 = 4,194,304`（768 kHz で 5.46 s）へ拡張、または
- `SAFE_MAX` に合わせて `IR_LENGTH_MAX_SEC` を SR 依存 `min(3.0, MAX_IR_LATENCY/sr)` にし、スライダー上限を動的化

---

## 2. SR-02 — tailStart と l0MaxLen のレート非依存キャップ（**High**）

### 2.1 実装

```738:745:src/MKLNonUniformConvolver.cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l0MaxLen = kL0MaxParts * l0Part;  // kL0MaxParts = 32
const int l0LenByTailStart = static_cast<int>(std::llround(tailStartSec * sampleRateForTail));
const int l0LenTarget = juce::jlimit(l0Part, l0MaxLen, l0LenByTailStart);
const int l0Len = std::min(irLen, tailEnabled ? l0LenTarget : l0MaxLen);
```

`tailStartSec` デフォルト 0.085 s。`l0MaxLen` は **パーティション数×partSize** で、秒換算すると SR に反比例して短くなる。

### 2.2 768 kHz での実効 L0 長

| blockSize | l0Part | l0MaxLen | 0.085 s 要求 | 実効 L0 | 意図比 |
|-----------|--------|----------|--------------|---------|--------|
| 256 | 256 | 8,192 | 65,280 | **8,192** | **12.5%** |
| 512 | 512 | 16,384 | 65,280 | **16,384** | **25%** |
| 1024 | 1,024 | 32,768 | 65,280 | **32,768** | **50%** |
| 2048 | 2,048 | 65,536 | 65,280 | 65,280 | 100% |
| 4096 | 4,096 | 131,072 | 65,280 | 65,280 | 100% |

48 kHz では `0.085*48000=4080` が l0Max（例: 32768）未満のため問題なし。  
**高レート＋小さいブロックで L0 が極端に短くなり、早期反射が L1（大パーティション）へ落ちる。**

### 2.3 DSP 影響

- L0 = 低遅延・高解像度層。L1 は `l0Part*8` パーティション
- L0 が 10 ms 相当に潰れると、**早期反射の時間解像度が落ち、L1 の分散計算遅延に乗りやすい**
- tailMode=AirAbsorption の L1/L2 ゲイン設計は「L0 が tailStart まで持つ」前提

### 2.4 修正案

- `kL0MaxParts` を SR 依存に拡大（例: `max(32, ceil(tailStartMaxSec * sr / l0Part))`）、または
- `l0MaxLen` を秒ベースで計算し partSize の倍数に切り上げ

---

## 3. SR-03 — AudioEngine と Convolver のレイテンシ上限不整合（**Medium**）

| 定数 | 値 | 768 kHz での秒 |
|------|-----|----------------|
| `AudioEngine::kMaxLatencySamples` | 1,536,000 | **2.00 s** |
| `ConvolverProcessor::MAX_TOTAL_DELAY` | 2,621,440 | **3.41 s** |
| `ConvolverProcessor::MAX_IR_LATENCY` | 2,097,152 | 2.73 s |
| `DELAY_BUFFER_SIZE` | 4,194,304 | 5.46 s |

```2115:2116:src/audioengine/AudioEngine.h
static constexpr int kMaxLatencySamples = 1536000; // 最大2秒@768kHz対応
static constexpr int MAX_LATENCY_ALIGN_SAMPLES = 96000 * 2; // 2秒@48kHz
```

Convolver 側が 2.73 s の IR 遅延を申告し得るのに、Engine 側 PDC 整合は 2.0 s 上限。  
**768 kHz で長 IR + 大きな irPeak（H-02 未修正だと重心が大きい）で PDC が切れる可能性。**

---

## 4. SR-04 — UI IR 長スライダーが hardMax を無視（**Medium**）

```87:87:src/ConvolverControlPanel.cpp
irLengthSlider.setRange(ConvolverProcessor::IR_LENGTH_MIN_SEC, ConvolverProcessor::IR_LENGTH_MAX_SEC, 0.1);
```

```1309:1311:src/ConvolverControlPanel.cpp
const double irLengthSliderMax = std::max(IR_LENGTH_MAX_SEC, ...getTargetIRLength());
irLengthSlider.setRange(IR_LENGTH_MIN_SEC, irLengthSliderMax, 0.1);
```

- 上限は常に `IR_LENGTH_MAX_SEC=3.0`（以上）
- `hardMaxSec` は preview で検査されるが、**スライダー自体は 768 kHz でも 3.0 s を選択可能**
- `setTargetIRLength` は `getMaximumAllowedIRLengthSec` でクランプするため実データは 2.73 s に落ちるが、**UI 表示と実効が乖離**

---

## 5. SR-05 — HC スペクトラムフィルタの 48 kHz 二値分岐（**Low**）

```341:342:src/MKLNonUniformConvolver.cpp
const double hcFcStart = (fs <= 48000.0) ? 18000.0 : 22000.0;
const double hcFcEnd   = nyquist;
```

```80:80:src/OutputFilter.cpp
const double fc_hc = (sampleRate <= 48000.0) ? 19000.0 : 22000.0;
```

| SR | Nyquist | HC 開始 | 遷移帯幅 |
|----|---------|---------|----------|
| 44.1 kHz | 22.05 kHz | 18 kHz | 4.05 kHz |
| 48 kHz | 24 kHz | 18 kHz | 6 kHz |
| 96 kHz | 48 kHz | 22 kHz | 26 kHz |
| 192 kHz | 96 kHz | 22 kHz | 74 kHz |
| **768 kHz** | **384 kHz** | **22 kHz** | **362 kHz** |

- 44.1 kHz では遷移が非常に急（設計意図どおりか要聴感確認）
- 768 kHz では「22 kHz から緩やかに Nyquist まで」になり、**超音波域のカット形状が SR で大きく変わる**
- R-1（OutputFilter fc）は音響設計値として確定済みだが、**NUC 側 applySpectrumFilter の 18k/22k は 384 kHz Nyquist を前提に再検討の余地**

---

## 6. SR-06 — Oversampling / prepare ガード（**妥当**）

```42:48:src/audioengine/OversamplingPolicy.h
// sr ≤ 96k → OS max 8; ≤192k → 4; ≤384k → 2; ≤768k → 1; >768k → 不可
```

- 処理レート上限 `kMaxInternalRate = 768000`
- 768 kHz デバイスでは OS=1。44.1 kHz では最大 8× → 352.8 kHz
- `prepareToPlay` は `SAFE_MAX_SAMPLE_RATE` 超で 48 kHz フォールバック（`PrepareToPlay.cpp:127-129`）

**オーバーサンプリング経由での 768 kHz 超は発生しない。**

---

## 7. SR-07 — 768 kHz 長 IR のメモリ（**Info**）

`MAX_IR_LATENCY` キャップ後 irLen=2,097,152、bs=1024 と仮定:

| 項目 | 見積 |
|------|------|
| irData ステレオ | ~33.6 MB |
| L0 SoA | ~1.6 MB |
| L1 SoA | ~25 MB |
| **L2 SoA** | **~100 MB/ch** |
| Convolver delay 2ch | ~67 MB |
| r8brain 44.1k→768k 3s | 出力 ~18 MB/ch（一時） |

L2 が `partSize=65536, complexSize=65537, numParts~32` で 100 MB 級。  
ワークステーションでは許容だが、32 bit プロセスや低メモリ環境では失敗し得る。`SetImpulse` の OOM パスは存在する。

---

## 8. 既知バグとの相互作用（レート依存の悪化）

| 既知 ID | レート依存の悪化 |
|---------|------------------|
| **H-02** エネルギー重心 | 重心 [samples] ∝ SR。768 kHz では 48 kHz の 16 倍。2 s IR で **~3–5 秒相当の PDC 誤差**になり得る |
| **H-01** dry 遅延 | `partSize` は SR 非依存だが、`irPeak`（重心）が SR 比例で増える |
| **M-01** デノーマル | 高 SR では IFFT テールがより深くデノーマル域へ入る |

---

## 9. 優先修正（レート対応）

1. **SR-01** `MAX_IR_LATENCY` を 768 kHz×3 s 以上（`2^22` 推奨）へ。`DELAY_BUFFER_SIZE` も再検証
2. **SR-02** `l0MaxLen` を秒ベース計算に変更（`ceil(tailStartMax * sr / l0Part) * l0Part` かつ 2 冪）
3. **SR-03/SR-04** Engine `kMaxLatencySamples` と IR スライダー上限を SR 依存へ統一
4. **H-02** 真のピークへ（高 SR で最重要）
5. **SR-05** HC の遷移帯を Nyquist 比で設計し直す（任意）

---

## 10. 推奨テスト

| テスト | 期待 |
|--------|------|
| 768 kHz で 3.0 s IR ロード | 打ち切り警告 or 完全ロード（どちらか一貫） |
| 768 kHz / bs=512 で tailStart=0.085 | L0 長が 85 ms 相当 |
| 768 kHz で setLatencySamples | 2.73 s IR + 重心 irPeak が 2.0 s 上限に当たらない |
| 44.1 kHz → 768 kHz IR リサンプル | r8brain 成功・メモリ・所要時間 |
| 768 kHz NUC Null Test | M1 −90 dB 以下を維持 |

---

## 11. 結論

- **705.6 / 768 kHz では `MAX_IR_LATENCY` が 3 s IR に足りず、無言のテール打ち切り**（SR-01）
- **高 SR + 小ブロックで L0 が tailStart に届かず、NUC 分割が設計から乖離**（SR-02）
- Oversampling 上限と prepare ガードは 768 kHz まで整合
- 既知 H-02 は高 SR で PDC 誤差が線形に悪化するため、**レート対応と同時に修正すべき**

以上。
