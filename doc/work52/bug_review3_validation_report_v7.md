# bug_review3.md 検証レポート v7（総合確定版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md`
- **参考**: `doc/gain_staging_analysis_2026-06-21.md`
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI

---

## 0. バグ内容の整理

| 条件 | 結果 |
|:----|:----:|
| Conv→Peq + Adaptive9th | **発生** |
| Conv→Peq + Fixed4Tap | **発生** |
| Conv→Peq + Fixed15Tap | **発生** |
| **PEQ-only + 全NS** | **発生せず** |
| トリガー条件 | **低音が大きい時**（全体音量が小さくても発生） |

---

## 1. 調査の変遷と確定事項

### 1.1 各バージョンの結論

| v | 中心仮説 | 最終評価 |
|:-:|---------|:--------:|
| 1-3 | ゲイン構造→NS過大入力 | ❌ ゲイン分析で否定 |
| 4 | post-NS ハードクランプ主犯 | ❌ ユーザー評価で否定 |
| 5 | NS入力過大→内部状態不安定化 | ❌ ゲイン分析で否定 |
| 6 | IR波形特性 + SoftClip相互作用 | ⚠️ 「レベル同等」でNS説は弱体化 |
| **7** | **SoftClip+NS相互作用（Convolver波形変形起因）** | **✅ 最有力** |

### 1.2 確定した事実

1. **IR Scale Factor** による自動正規化（-6dB）が存在する
2. **inputHeadroom -6dB** との相殺により、SoftClip直前の信号レベルは PEQ-only と Conv→Peq で **同等**
3. **+12dB outputMakeupGain** は NoiseShaper より前にある（コード確定）
4. **kOutputHeadroom (0.891)** は NoiseShaper 内部で動作基準値として使用される
5. **post-NS clamp** は全モード共通の最終安全柵であり、主犯ではない
6. **SoftClip AVX2 パスは memoryless waveshaper**（P3でmidVec削除済み）
7. **SoftClip スカラーフォールバックは履歴付き**（`avg = 0.5*(prev+input)`）
8. **全 NoiseShaper で発生する**（Adaptive9th固有ではない）

---

## 2. ソースコードから見た SoftClip 実態

### 2.1 AVX2 パス（メイン処理）— memoryless waveshaper

`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` lines 142-197:

```cpp
// [P3] midVec事前平均化ブロックを完全削除
// x はそのまま後続のSoftClip（fastTanh近似）へ流れる。

__m256d absX = _mm256_andnot_pd(vSignMask, x);
// ↓ 純粋な waveshaper：瞬時値のみで処理
__m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);
// ... TanhApprox + knee処理 ...
result = _mm256_blendv_pd(x, result, needClip);
_mm256_storeu_pd(data + i, result);
```

**→ 標準ブロックサイズ（4の倍数）では過去サンプルを使用しない純粋な waveshaper。**

### 2.2 スカラーフォールバック — 履歴付き

```cpp
// lines 200-215（numSamples % 4 ≠ 0 の場合のみ実行）
const double mid = (prevScalar + inputVal) * 0.5;
// ↑ 1サンプル履歴 + 現在値の平均
```

**→ 非標準ブロックサイズでのみ履歴が影響する。**

### 2.3 DSPCoreFloat.cpp の SoftClip — 常に履歴付き

```cpp
// AudioEngine.Processing.DSPCoreFloat.cpp lines 115-127
const double avg = 0.5 * (x + prevSample);
prevSample = x;
data[i] = musicalSoftClipScalar(avg, threshold, knee, asymmetry);
```

**→ Float パスでは全サンプルで履歴付き処理。** Float パスが使用される条件は要確認。

---

## 3. 相互作用メカニズム（最有力仮説）

```
低音入力
  ↓
Convolver（IRとの畳み込み）
  ↓  波形が変形：群遅延・残響・位相変化・crest factor変化
EQ（低域ブースト）
  ↓  +12dB makeup（+6dB正味）
SoftClip
  ↓  波形の非対称性 → 偶数次高調波・変調成分が増加
  ↓  （AVX2:瞬時値 / Float:履歴付きでさらに非対称性増大）
OS Downsampler
  ↓  高調波の一部が除去されるが、低次成分は残る
DC Blocker（2nd-order IIR）
  ↓  低域の過渡応答がIIR状態に影響
NoiseShaper（全方式共通）
  ↓  変形された信号＋IIRの過渡応答を処理 → 「ジジジジ」
出力
```

### 「低音だけ」の説明

| 要因 | 説明 |
|:----|------|
| 低音の高エネルギー | 同振幅でも低域はサンプル間の相関が高く、FIR/IIRフィルタの状態変数に蓄積されやすい |
| Convolverの低域テール | IRの低域成分が残響状に伸び、SoftClipで非対称にクリップされる |
| DC Blockerの過渡応答 | 低域の急峻な変化 → IIRフィルタの状態が変動 → NoiseShaperへ |
| NSの低域感度 | 全NSとも低域の状態変数変動に敏感（Latticeの`kLatticeStateLimit=2.0`、Fixedの`channelErrors`、IIRのstate） |

---

## 4. 現時点の確率評価

| 仮説 | 確率 |
|:----|:---:|
| **Convolver→SoftClip→NS相互作用** | **45%** |
| Convolver由来の超低域成分 | 25% |
| SoftClip + Float path 履歴効果 | 15% |
| NS入力レベル問題 | 5% |
| post-NS clamp | 3% |
| Adaptive9th固有バグ | 1% |
| Partition glitch | 1% |

---

## 5. 未確定事項・要調査事項

| # | 項目 | 状態 | 確定方法 |
|:-:|:----|:----:|:---------|
| U1 | SoftClip完全OFF時の症状 | 未確認 | `softClipEnabled = false` でテスト |
| U2 | NS完全OFF時の症状 | 未確認 | `applyDither = false` 相当でテスト |
| U3 | outputMakeup 0dB時の症状 | 未確認 | `newOutputMakeupDb = 0.0f` でテスト |
| U4 | OS=1x固定時の症状 | 未確認 | `manualOversamplingFactor = 1` でテスト |
| U5 | Float/Double パス分岐条件 | 未確認 | どの条件でFloatパスが使われるか |
| U6 | Float path の SoftClip 履歴効果 | 未確認 | Floatパス使用時の症状変化 |
| U7 | IR長さと症状の相関 | 未確認 | 短IR vs 長IR の比較 |
| U8 | post-NS clamp 発動率 | 未測定 | clampヒット回数のカウンタ挿入 |
| U9 | DC Blocker の IIR 状態変動 | 未確認 | DC Blocker前後の信号比較 |

---

## 6. 推奨テスト手順

### テスト1: SoftClip完全OFF（最重要）

```cpp
// DSPCoreDouble.cpp line 442
if (state.softClipEnabled)  →  if (false)
```

**消える → SoftClip + NS 相互作用が主因で確定。**

### テスト2: NoiseShaper OFF

`applyDither = false` 相当の状態にする。

**消える → NSが原因経路の一部。残る → NS以前の経路が原因。**

### テスト3: Output Makeup = 0dB

```cpp
// AudioEngine.Parameters.cpp line 326
newOutputMakeupDb = 12.0f → 0.0f;
```

**SoftClipの動作点確認。消える → ゲイン構造が間接関与。**

### テスト4: OS=1x固定

```cpp
manualOversamplingFactor = 1;
```

**消える → Downsampler または OS-rate SoftClip が関与。**

### テスト5: 短IR vs 長IRの比較

数十msのIRと数秒のリバーブIRで比較。IR長で症状が変わるか確認。

---

## 7. 結論

```
確定: +12dB makeup は NoiseShaper より前に存在
確定: IR Scale Factor による自動正規化あり
確定: 全 NoiseShaper で発生（Adaptive9th固有ではない）
確定: 低音特化性（全体音量が小さくても発生）

推定: SoftClip + NoiseShaper 相互作用が最有力
       Convolverで変形した低域波形 → SoftClipで非対称歪み
       → NoiseShaperで「ジジジジ」に変換

未確定: SoftClip OFF で症状が消えるか
未確定: NS OFF で症状が消えるか
未確定: post-NS clamp の実際の発動率
```

**テスト1（SoftClip完全OFF）が最も高い診断価値を持つ。**
