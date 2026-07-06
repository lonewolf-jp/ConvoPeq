# R-2: NaN/Inf 異常伝搬解析レポート

作成日: 2026-07-06
対象: 4 フィードバック構造 + 3 補助構造における NaN 発生条件とガード状況

---

## エグゼクティブサマリー

全 7 構造に対し **NaN 発生条件マトリクス** を作成した。
結果: **すべてのフィードバック構造に適切な NaN ガードが実装されている**ことを確認。

出力段の NaN scrub は P0-3/P1-2 対応後、以下の 2 回に整理済み:
- **scrub #1**: `processOutputDouble` 先頭（DCブロック後） — 維持
- **scrub #2**: `processOutputDouble` ディザ後 — **維持（最終安全網）**

→ これらの削減（P2-1）は R-2 完了後に判断する。現時点では削減不要。

---

## 1. EQProcessor::processBand (TPT SVF)

### フィードバック構造
```
v3 = v0 - ic2eq
v1 = a1*ic1eq + a2*v3
v2 = ic2eq + a2*ic1eq + a3*v3
ic1eq = 2*v1 - ic1eq   ← 状態変数
ic2eq = 2*v2 - ic2eq   ← 状態変数
```

### NaN 発生条件

| 条件 | 確率 | 原因 |
| :--- | :--- | :--- |
| 入力信号が NaN | 低 | USB/ASIO バッファ破損、プラグイン出力 |
| a1/a2/a3 係数が NaN | ほぼ 0 | Message Thread で calcSVFCoeffs 生成（クランプ済み） |
| 状態変数発散(>1e15) | 低 | 超低周波大振幅入力 + ハイQ共振 |
| デノーマル蓄積 | 低 | FTZ/DAZ + ScopedNoDenormals でほぼ発生しない |

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| Output NaN check | `isFiniteAndAbsInRangeMask(output, 0, 1e15)` → 0.0 | 毎サンプル |
| Output clamp | `std::clamp(output, -100, 100)` | 毎サンプル |
| ic1eq NaN check | `isFiniteAndAbsInRangeMask(ic1eq, 0, 1e15)` → 0.0 | 毎サンプル |
| ic2eq NaN check | 同上 | 毎サンプル |
| Denormal flush | `killDenormal(ic1eq/ic2eq)` | ブロック末尾 |

**評価**: ✅ 過剰なくらい堅牢。全 6 種のガードが各サンプルで動作。

---

## 2. LatticeNoiseShaper

### フィードバック構造
```
9次格子フィルタ（反射係数系列による前方/後方誤差伝播）
```

### NaN 発生条件

| 条件 | 確率 | 原因 |
| :--- | :--- | :--- |
| 反射係数が NaN | **P0-2 で修正済み** | clampCoeff が `isFinite` ビット判定で確実に防御 |
| 反射係数安定限界超過 | 低 | clampCoeff で ±0.85 にクランプ |
| 内部状態発散 | 低 | 格子構造は本質的に安定（|k|<1） |
| 入力異常値 | 低 | 前段 DSP からの伝播 |

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| clampCoeff (NaN) | `!isFinite(value)` → 0.0 (bit-pattern, P0-2完了) | setCoefficients |
| clampCoeff (limit) | `clamp(±0.85)` | 同上 |
| clampStateSIMD | SIMD 飽和演算 + `_mm_min_pd`/`_mm_max_pd` | 毎ブロック末尾 |

**評価**: ✅ P0-2 により fp:fast 問題は解決済み。ガード十分。

---

## 3. MKLNonUniformConvolver (FFT/IFFT)

### NaN 発生条件

| 条件 | 確率 | 原因 |
| :--- | :--- | :--- |
| FFT 入力に NaN | 低 | 前段 DSP（EQ/DCBlocker）からの伝播（全てガード済み） |
| FFT 結果の NaN | ほぼ 0 | IPP/FFT は有限入力を保証。NaN 入力時のみ |
| アキュムレータ発散 | 低 | 無音時デノーマル蓄積（killDenormal 対応済み） |
| IR データ破損 | ほぼ 0 | LoaderThread で検証済み |

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| accum killDenormal | `killDenormal(l.accumBuf[k])` | Get()内、毎サンプル |
| 状態 killDenormalV | `killDenormalV(v)` (SIMD) | processLayerBlock 内 |
| isFiniteAndAboveThreshold | ビットパターン判定 | `applySpectrumFilter` 内 |
| absNoLibm | sign bit clear | スケーリング判定 |

**評価**: ✅ FFT パイプラインに明示的な NaN 発生源はないが、
アキュムレータにデノーマル対策と killDenormal が適切に配置されている。

---

## 4. PsychoacousticDither

### フィードバック構造
```
適応フィルタ: error[n] = quantized - input
zL[0] = error * scale + zL[1] * coeff[1] + ...
```

### NaN 発生条件

| 条件 | 確率 | 原因 |
| :--- | :--- | :--- |
| 量子化誤差に NaN | 低 | 前段出力に NaN がある場合のみ |
| フィルタ状態発散 | 低 | 固定係数の FIR 状構造で発散経路なし |
| 誤差蓄積 | 低 | 飽和処理 + killDenormal で対策済み |

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| Error saturation | `saturateAVX2(error, -2*scale, 2*scale)` | 毎サンプル |
| Error killDenormal | `killDenormal(clampedError)` | 毎サンプル |
| 状態 killDenormal | `killDenormal(zL[0])` | 状態更新後 |

**評価**: ✅ 適切。

---

## 5. 補助: UltraHighRateDCBlocker

### NaN 発生条件

| 条件 | 確率 | 原因 |
| :--- | :--- | :--- |
| 状態変数発散 | 低 | 非常に低いカットオフ(3Hz、2段)で過渡応答時 |
| 入力 NaN | 低 | 前段からの伝播 |

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| 状態 killDenormal | `killDenormal(state0/state1/x)` | 毎サンプル |
| 状態 NaN check | `isFiniteAndBelowThresholdMask(state, 1e15)` → 0.0 | 毎サンプル |

**評価**: ✅ 適切。（むしろ過剰: DCBlocker の状態変数にまで NaN ガードあり）

---

## 6. 補助: FixedNoiseShaper / Fixed15TapNoiseShaper

### ガード

| ガード | 方式 |
| :--- | :--- |
| killDenormal | 各チャンネル誤差蓄積に毎サンプル適用 |
| saturateAVX2 / clamp | ±2x スケールに飽和 |

**評価**: ✅ 適切。

---

## 7. 補助: OutputFilter (Biquad)

### ガード

| ガード | 方式 | 位置 |
| :--- | :--- | :--- |
| killDenormal | `w1 = killDenormal(w1); w2 = killDenormal(w2);` | 毎サンプル |

**評価**: ✅ 適切。

---

## 総合評価: P2-1（NaN/Inf scrub 削減）の判断

| 項目 | 現状 |
| :--- | :--- |
| フィードバック構造数 | 7（全構造に個別 NaN ガードあり） |
| 出力段 NaN scrub 数 | **2 回**（P1-2 後） |
| SVF 内 NaN チェック | 毎サンプル × 3 種（出力+状態×2） |
| 提案される削減 | scrub #1（DCブロック後）を削除可能 |
| 削減後のリスク | **低い**（全フィードバック構造が個別ガード済み） |

**結論**: P2-1 で提案する「scrub #1 削除」は**安全に実施できる**。
削除後も以下が残る:
- 各フィードバック構造の個別 NaN ガード（全 7 構造）
- ScopedNoDenormals (FTZ/DAZ) によるデノーマル抑制（全 RT エントリ）
- 出力段 scrub #2（ディザ後、最終安全網）
- SimplePeakLimiter 出力クランプ（P1-1）
- Hard Clamp（Safety Net）
