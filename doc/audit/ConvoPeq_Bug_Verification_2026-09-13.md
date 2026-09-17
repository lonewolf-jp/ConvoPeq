# ConvoPeq バグ検証報告書（深層調査版）

- **日付**: 2026-09-13
- **対象**: 初回監査のバグ候補について、ソース詳細・シミュレーション・文献により妥当性を確定
- **先行**: `doc/audit/ConvoPeq_IR_Audio_Path_Audit_2026-09-13.md`

---

## 1. 検証結果サマリ

| ID | 初回判定 | **検証後** | 備考 |
|----|----------|------------|------|
| C-01 | Critical | **棄却（当初主張）** / 限定残存 | L1/L2時計乖離は非成立。非典型的bsでwet穴＋direct消失のみ |
| H-01 | High | **確認** | dry 遅延と wet OLS の時間整合不良 |
| H-02 | High | **確認** | エネルギー重心をピークとして使用 |
| H-03 | High | **部分確認（設計）** | loadIR→deferred Structural rebuild→NUC |
| M-01 | Medium | **確認（弱）** | L1/L2 IFFT前デノーマル欠落 |
| M-02 | Medium | **確認** | NaN スクラビングが FDL を放置 |
| M-03 | Medium | **確認** | Direct Head の latency/フィルタ不整合 |
| M-04 | Medium | **確認** | スムージングバッファ不足で早期 return |
| R-1/R-2/R-3 | — | **変更不要確定** | 前回確定を維持 |

---

## 2. C-01 — 非2冪ブロック / Get 時計（**当初主張は棄却**）

### 2.1 コード

```cpp
// MKLNonUniformConvolver.cpp:1691-1762
const std::uint64_t t0 = m_outputSamplesProcessed;
const int got = ringRead(output, numSamples);
delayLineReadAdd(l, output, numSamples, t0, gain);
m_outputSamplesProcessed += static_cast<std::uint64_t>(got);
```

```cpp
// StereoConvolver::process (Runtime.cpp:1174-1183)
const int got = nucConvolvers[channel]->Get(out, numSamples);
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
```

### 2.2 シミュレーション（partSize = nextPow2(bs)）

| bs | partSize | warmup後の underflow | 判定 |
|----|----------|----------------------|------|
| 64 / 128 / 256 / 512 / 1024 / 2048 | 同値 | 0/100 | OK |
| **480** | 512 | **0/100**（初回のみ） | 定常は安全 |
| 240 / 768 / 960 | 256/1024/1024 | 0/100 | OK |
| 100 | 128 | 3/100 | 限定残存 |
| 300 | 512 | 4/100 | 限定残存 |
| 441 | 512 | 5/100 | 限定残存 |
| 882 | 1024 | 5/100 | 限定残存 |

### 2.3 機序の再解析

1. **L1/L2 時計乖離は発生しない**  
   wipe + `clock += got` により未コミット区間は次 Get で再読。読み専用 delay line のため再読は安全。恒久的な重複・スキップは起きない。

2. **bs=480, partSize=512** では leftover が蓄積し、process しない callback でも `avail >= 480`。定常 underflow なし。

3. **残存リスク（bs=100/300/441/882 等）**  
   - wet のゼロ詰め区間（可聴ドロップ）  
   - Direct path: pending クリア後 wipe でテール永久消失  
   - L1/L2 自体は整合維持

### 2.4 判定

- **当初の Critical「L1/L2 ストリーム時計乖離」は棄却。**
- **一般的ホストバッファ（2冪、480, 240, 768, 960）では問題なし。**
- 非典型的サイズでの wet 穴・direct 消失は **Medium（条件付き）**。

---

## 3. H-01 — dry 遅延 vs wet OLS（**確認**）

### 3.1 コード

| 項目 | 場所 | 値 |
|------|------|-----|
| `m_latency` | `MKLNonUniformConvolver.cpp:1055` | `L0.partSize` |
| host PDC | `getLatencyBreakdown` → `setLatencySamples` | `algorithmLatency + irPeakLatency` |
| dry 遅延 | `Runtime.cpp:266-286, 551-566` | 同上 |
| bypass | `Runtime.cpp:142-146` | 同上 |
| wet | OLS 後半 ring → Get | **出力 n に y[n]=(h*x)[n]** |

### 3.2 DSP 理論

- Overlap-Save は入力 `[n0, n0+B)` に対し IFFT 後半で **同一時間指標** の `y[n0..n0+B)` を出す。配列上 wet に追加遅延なし（M1 Null Test -309 dB）。
- dry を `partSize + irPeak` だけ遅らせると

```
out[n] = gW*(h*x)[n] + gD*x[n - Npart - Npeak]
```

  となり、mix 中間で **コムフィルタ／遅延エコー**。
- `irPeak` 分の dry 遅延は「wet 直接音ピークへの合わせ」として正当化し得る。
- **`partSize` 分の dry 遅延は wet 側に対応が無いため不要。**

### 3.3 含意

- mix=1.0: dry 不使用 → 影響なし
- mix=0.0: dry のみ遅延 → 無加工相当なのに遅延
- bypass: 同式 → トグルで位相ジャンプの可能性

### 3.4 判定

**確認（High）**  
`m_latency = 0` にするか、dry 遅延から `algorithmLatency` を外し `irPeak` のみにする。  
mix=0.5 dry/wet null test を追加すべき。

---

## 4. H-02 — irPeakLatency がエネルギー重心（**確認**）

### 4.1 実装

`LoaderThread.cpp:149-208`:

- `ENERGY_THRESHOLD = 0.999` で累積エネルギー cutoff
- `[0..cutoff]` の **エネルギー重心** を採用
- チャネル間 max、`floor(+0.5)`

コメント・フィールド名は「ピーク位置」（`ConvolverProcessor.h:720`）。

### 4.2 数値見積り（48 kHz）

| IR 形状 | 真の max-abs | 重心（実装式） | 誤差 |
|---------|--------------|----------------|------|
| exp 減衰 τ=10 ms | 0 | 238 | +238 |
| τ=100 ms | 0 | 2,383 | +2,383（約50 ms） |
| τ=500 ms | 0 | 11,893 | +11,893 |
| 直達@50 + τ=200 ms | 50 | **4,806** | +4,756（約100 ms） |

長いリバーブ IR では **PDC・bypass・dry 遅延が 100 ms 超**。

### 4.3 経路不整合

| 経路 | 測定 |
|------|------|
| LoaderThread → NUC `irLatency` | エネルギー重心 |
| `applyComputedIR` UI 表示 | **真の max-abs**（`LoadPipeline.cpp:500-514`） |

### 4.4 判定

**確認（High）**  
真のピークへ統一。重心を使うなら PDC に使わない。

---

## 5. H-03 — loadIR と NUC 差替え（**部分確認・設計**）

### 5.1 経路

```
UI Load IR
  → requestConvolverPreset → loadIR
  → applyComputedIR（IRState/UI/RCU のみ。SetImpulse なし）
  → changeListenerCallback → convolverParamsChanged
  → submitRebuildIntent(Structural) ※初回 +200 ms defer
  → rebuildAllIRsSynchronous → LoaderThread → SetImpulse
```

ProgressiveUpgrade 最終も `applyComputedIR` のみ。

### 5.2 判定

**バグというより設計上の遅延差替え。**  
rebuild 完了まで旧 IR / 無音ウィンドウ。`partitionData` はライブ DSP ではデッド。

---

## 6. M 系列

| ID | 判定 | 根拠 |
|----|------|------|
| M-01 | 確認（弱） | L0 は pre-IFFT `killDenormalV`、L1/L2 はなし。Release は FTZ依存で no-op |
| M-02 | 確認 | 出力 NaN のみ 0 化、FDL 残留で永続ミュート |
| M-03 | 確認 | direct 32tap、latency=0 申告、HC/LC 不適用 |
| M-04 | 確認 | `Runtime.cpp:593-594` capacity 不足で return |
| M-05 | 確認（既知） | SR 変更時の未リサンプル IR（コメント [M-1]） |

---

## 7. 非バグ確定

| 項目 | 根拠 |
|------|------|
| IPP CCS | 公式: 標準複素 N/2+1 = N+2 double。`deinterleaveComplex` と一致 |
| IPP スケール | `IPP_FFT_DIV_INV_BY_N` |
| B13 Policy R | oPE=0 / M1 -309 dB（work57） |
| R-1/R-2/R-3 | 変更不要確定 |

---

## 8. 優先修正（検証後）

1. **H-02** 真のピークへ統一（PDC 影響最大）
2. **H-01** dry 遅延から `algorithmLatency` を外す / mix=0.5 null test
3. **M-02** NaN 時に FDL クリア
4. **M-01** L1/L2 にも pre-IFFT デノーマル
5. **C-01 残存** 非典型的 bs の wet 穴（必要なら内部正規化）
6. **H-03** 初回 200 ms + rebuild 窓の UI 表示

---

## 9. 推奨追加テスト

| テスト | 期待 |
|--------|------|
| 2s 減衰 IR で irLatency vs max-abs | 一致（H-02） |
| impulse + mix=0.5 交差相関 | 相対遅延 0（H-01） |
| bs=300 短 IR | wet 周期ゼロなし（C-01） |
| UI Load IR 直後の engine | rebuild 前は旧/null（H-03 仕様） |

---

## 10. 結論

- **最優先は H-02（PDC 過大）と H-01（dry/wet コム）。**
- **C-01 は当初 Critical を棄却。** 一般バッファは安全。
- **H-03 は deferred rebuild 設計。** stale 窓の文書化が課題。
- コア畳み込み（CCS・スケール・OLS・B13）は引き続き妥当。

以上。
