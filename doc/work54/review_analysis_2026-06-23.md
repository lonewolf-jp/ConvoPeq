# 外部レビュー検証結果

**日付**: 2026-06-23
**対象**: ConvoPeq コードベース
**検証者**: AI Assistant (Serena MCP, CodeGraph MCP, AiDex MCP, grep/semble 併用)

---

## 検証サマリ

| バグID | レビュー主張 | 判定 | 重要度 |
|--------|------------|------|--------|
| Bug 1 | NUPC 因果律制約違反による無音ギャップ | **Architecture Misunderstanding** | — |
| Bug 2 | LatticeNoiseShaper FIR格子更新式破壊 | **設計逸脱の疑惑。致命的バグは未証明** | — |

---

## Bug 1: NUPC 因果律制約違反 → **Architecture Misunderstanding**

### レビュアーの主張

> L1 レイヤーは partSize = l1Part = l0Part *8 = 4096。
> 因果律制約により L1 の最小オフセットは 2*l1Part - l0Part = 7680 サンプル。
> しかし SetImpulse() では tailStartSec=0.085s (48kHzで4080サンプル) がそのまま l0Len (=l1Offset) に適用される。
> 4080 < 7680 で制約違反 → 4080〜7679サンプルの音が欠落し、8192サンプル目で異常な遅延エコーが発生。

### 検証結果: 誤り

レビュアーの処理モデル（「入力が l1Part 貯まるのを待つ → さらに l1Part の時間をかけて分散積算」）は、**この実装のアーキテクチャと一致しません**。

#### 実際のNUPC処理モデル

本実装の `MKLNonUniformConvolver` は **3層非均一パーティション畳み込み + FDL (Frequency-Delay Line)** 方式です。

**L0 (Immediate レイヤー)**:

- `isImmediate = true`
- `partSize` ごとに全パーティション即時処理
- 出力をリングバッファに直接書き込み

**L1/L2 (Non-Immediate レイヤー)**:

- `isImmediate = false`
- `partSize` 蓄積後、即座に Forward FFT → FDL に格納
- 複素乗算積算を **毎コールバック分散実行**（`partsPerCallback` ずつ）
- 全パーティション完了時 IFFT → `tailOutputBuf` にコピー
- `Get()` で `ringRead(L0) + add(L1.tail) + add(L2.tail)` で合成

**分散積算は並行処理**: 新しい入力サンプルを蓄積しながら、前フレームの積算を進める。
したがって L1 に「処理時間」は**発生しません**。

```cpp
// MKLNonUniformConvolver.cpp:Add() - L1/L2 分散積算
if (!l.isImmediate && l.distributing)
{
    const int endPart = std::min(l.nextPart + l.partsPerCallback, l.numPartsIR);
    // ... 現在の Add() コール内で積算を進める ...
    l.nextPart = endPart;
    if (l.nextPart >= l.numPartsIR) {
        ippsFFTInv_CCSToR_64f(...);  // IFFT → tailOutputBuf ready
    }
}
```

**FDL 構造**: FDL には過去全フレームの周波数領域表現が保持されており、線形化アクセス（`linStart = baseFdlIdx - numPartsIR + 1 + numParts`）で任意の範囲の畳み込みを計算可能。

#### 主要パラメータの実測検証

```cpp
// MKLNonUniformConvolver.cpp:SetImpulse() L610-630
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));  // 1024 @ 48kHz
const int l1Part = l0Part * tailL1L2Mult;                           // 8192
const int l2Part = l1Part * tailL1L2Mult;                           // 65536

const int l0Len = std::min(irLen, tailEnabled ? l0LenTarget : l0MaxLen);
const int l1Offset = l0Len;  // 正しい
```

**`partsPerCallback` の計算**:

```cpp
const int blocksPerPart = (l.partSize + max(blockSize, 1) - 1) / max(blockSize, 1);
l.partsPerCallback = max(1, (l.numPartsIR + blocksPerPart - 1) / blocksPerPart);
```

| シナリオ | numPartsIR | blocksPerPart | partsPerCallback | 完了サイクル |
|---------|-----------|---------------|-----------------|-------------|
| IR 200ms | 1 | 8 | 1 | 即時（1 Add()） |
| IR 2s | 12 | 8 | 2 | 6 Add() = 6144 samples |

L1 出力は新しい入力フレームと同じタイミングで `tailOutputBuf` をリフレッシュし続けるため、**定常状態ではタイミングずれは収束します**。

#### レビュアー誤りの原因（結論）

レビュアーは以下の点で**アルゴリズム理解を誤っている**:

1. **NUPC の FDL 方式を理解していない**: 単純な "待つ→処理する" モデルを仮定
2. **分散積算の並行性を無視**: 各コールバックで `Add()` が複数の処理（新規入力蓄積 + 前フレーム積算継続）を実行する設計を見落とし
3. **partition offset ≠ algorithmic latency**: FDL 方式ではパーティションオフセットとアルゴリズムレイテンシは独立

**提案された修正（`minL1Offset` クランプ）は不要。** 適用すると l0Len が不必要に拡大され、L0 の処理負荷が増大するだけの悪影響がある。

---

## Bug 2: LatticeNoiseShaper FIR格子更新式破壊 → **「設計逸脱の疑惑」までは成立。「致命的バグ」は未証明**

### レビュアーの主張

> `advanceState()` が `state[i]` に `nextBackward`（自段の後方反射波）を保存している。
> 正しい FIR 格子では前段の後方反射波を保存すべき。
> 現在のコードは極 z=1 を持つ不安定な IIR 構造にフィルタを破壊している。

### 検証結果

#### ✅ 文献比較: CMSIS-DSP (ARM公式) と Ne10 の両方と比較

**ARM CMSIS-DSP** (`arm_fir_lattice_f32.c`) — ARM 公式 DSP ライブラリ、最も権威ある実装:

アルゴリズム定義:

```
f0[n] = g0[n] = x[n]
fm[n] = fm-1[n] + km * gm-1[n-1]  for m = 1,...,M
gm[n] = km * fm-1[n] + gm-1[n-1]  for m = 1,...,M
y[n] = fM[n]
```

State配列（公式ドキュメント）: `{g0[n], g1[n], g2[n], ..., gM-1[n]}`

CMSIS-DSP のスカラーパス（非ループアンローリング時）の処理:

```c
/* g0(n-1) を state[0] から読み出し */
gcurr0 = *px;
/* f1(n) = f0(n) + K1 * g0(n-1) */
fnext0 = (gcurr0 * (*pk)) + fcurr0;
/* g1(n) = f0(n) * K1 + g0(n-1) */
gnext0 = (fcurr0 * (*pk++)) + gcurr0;
/* ★ state[0] に g0[n] (= x[n]) を明示的に保存 */
*px++ = fcurr0;

/* 次段: state[1] を読み出し */
gcurr0 = *px;    /* = g1[n-1] */
/* ★ state[1] に g1[n] を保存 */
*px++ = gnext0;
```

**CMSIS-DSP の state 保存パターン（決定的）**:

- state[i] には **g_i[n]** を保存
- 次サンプルでは state[i] = **g_i[n-1]** として使用
- 各段の backward = **g_i[n-1]**（正しい）

**ConvoPeq 現行の state 保存パターン**:

- state[i] には **g_{i+1}[n]** を保存（1段ずれ）
- 次サンプルでは state[i] = **g_{i+1}[n-1]** として使用
- 各段の backward = **g_{i+1}[n-1]**（1段ずれ）

**Ne10** (Project Ne10, ARM 非公式) も CMSIS-DSP と同様のパターン。

**→ 「state[i] = nextBackward;」というパターン自体は CMSIS-DSP / Ne10 と共通だが、CMSIS-DSP は state[0] に g0[n] を明示的に保存している。ConvoPeq は state[0] に g1[n] を保存しており、CMSIS-DSP と比較して 1 段ずれている。**

#### kOrder=1 での具体的な数値差

**CMSIS-DSP (正しい FIR 格子)**:

```
n=0: state[0] = e[0] (= g0[0])
n=1: backward = state[0] = e[0] (= g0[0])
     feedback(1) = k0 * e[0]
```

**ConvoPeq 現行**:

```
n=0: state[0] = k0*e[0] (= g1[0])
n=1: backward = state[0] = k0*e[0] (= g1[0])
     feedback(1) = k0 * (k0*e[0]) = k0² * e[0]
```

→ ConvoPeq の feedback には **k0² が乗算される**（標準は k0）。
k0=0.5 の場合、feedback は 0.25*e[0]（標準: 0.5*e[0]）と半分になる。

**kOrder=2 での影響**: feedback に k0·k1 の積項が出現。標準 FIR 格子の feedback（k0, k1 が独立な線形和）とは構造が異なる。

#### ✅ 教科書的 FIR 格子との相違は存在

| | state[i] に保存される値 | 次サンプルで使われる値 |
|---|---|---|
| 標準 FIR 格子（教科書） | g_i[n] | g_i[n-1] |
| Ne10 / ConvoPeq 現行 | g_{i+1}[n] | g_{i+1}[n-1] |
| 提案修正（旧コード） | g_i[n] | g_i[n-1] |

→ `computeFeedback()` は `Σ k_i * state[i]` を計算する。state[i] の意味（g_i か g_{i+1} か）が異なれば、feedback の値も異なる。

#### ⚠️ 線形 Z 領域解析は成立しない — clamp による非線形性

**重要な注意**: 本実装の `advanceState()` には `std::clamp(nextBackward, -2.0, 2.0)` が存在する。この clamp により状態遷移は**非線形**となり、Z 変換（線形時不変システムの解析手法）は**厳密には成立しない**。

レポート前版で行った以下の解析:

```
state_i[z] = k_i·f_i(z) / (1 - z⁻¹)
```

これは `clamp` を無視した**線形近似**であり、以下の理由で結論の根拠として不十分:

1. `clamp(±2.0)` により状態値が飽和 → 高ゲイン時の動作は線形モデルと乖離
2. 非線形系では「極」の概念が厳密には定義できない
3. 実際の動作は係数値と入力振幅に依存し、線形解析の予測と異なる可能性がある

したがって「IIR vs FIR」の断定は**現状の資料だけではできない**。

#### ✅ 最も重要な発見: CMA-ES 学習器との整合性

`NoiseShaperLearner::evaluateCandidateMapped()`（`NoiseShaperLearner.cpp` L1253）の実装:

1. `EvaluationContext` 内の `LatticeNoiseShaper shaper` で**実際のコード**を実行
2. `context.shaper.processStereoBlock()` で音声処理
3. 誤差を FFT 評価 → CMA-ES がスコア最小化

**→ CMA-ES の最適化対象 = 現在の advanceState() 実装（clamp 込み）**

| シナリオ | 影響 |
|---------|------|
| `advanceState()` を標準 FIR 格子に変更 | 最適化済み係数が全失効、NTF が変化、再学習必須 |
| 変更 + CMA-ES 再学習 | 新構造用の係数に再最適化可能 |
| 現状維持 | 非標準構造だが CMA-ES 込みで一貫した最適化が成立 |

### 総合評価

| 論点 | 判定 |
|------|------|
| 教科書的 FIR 格子との相違 | ✅ 存在（state[i] に g_{i+1}[n] を保存） |
| Ne10 との一致 | ✅ 同一パターン |
| 即修正が必要な致命的バグ | ❌ **証明されていない** |
| CMA-ES との整合性 | ✅ 現状で一貫（再学習不要） |
| NTF 測定の価値 | ✅ **非常に高い** — 判断の決め手 |

**現時点で最も可能性の高い構図:**

```
レビュー原文:  致命的バグと断定
一次検証:     理論逸脱を発見
本検証:       Ne10一致 + CMA-ES整合性を確認 → 独自実装＋学習器込みで整合している可能性
```

**推奨される判断:**

1. **コード変更には進まない** — NTF 実測比較なしではリスクが大きい
2. まず**現行実装の NTF を測定**
3. 次に**提案修正版の NTF を測定**（同一係数、clamp あり/なし両方）
4. さらに **CMA-ES 再学習後の NTF を測定**
5. 上記3条件を比較して初めて変更判断が可能

---

## 付録: CMSIS-DSP 準拠修正の影響評価

`advanceState()` を ARM CMSIS-DSP 準拠（`state[i] = clamp(b_prev, ...)` = g_i[n] を保存）に変更した場合の影響分析。

### 修正内容の対比

| | 現在 | CMSIS-DSP 準拠 |
|---|---|---|
| state[i] 保存値 | g_{i+1}[n] (= nextBackward) | g_i[n] (= b_prev) |
| feedback | Σ k_i · g_{i+1}[n-1] | Σ k_i · g_i[n-1] |
| kOrder=1 feedback(1) | k₀² · e[0] | k₀ · e[0] |

### 影響1: NTF が変化する（確定、再学習必須）

係数が同じでも NTF が変わる。過去の学習セッションで得られた係数は現在の構造用であり、修正適用と同時に無効化される。

```
変更前の最適化対象:  現在の advanceState()
変更後の最適化対象:  CMSIS-DSP 準拠 advanceState()
→ 再学習なしでは係数と構造の不一致が生じる
```

### 影響2: `isStable()` の意味が整合する（改善）

| | 現在 | CMSIS-DSP 準拠後 |
|---|---|---|
| isStable 条件 | \|k_i\| < 1（FIR 用） | \|k_i\| < 1（FIR 用） |
| 実際の構造 | IIR 積分器 | FIR 格子 |
| 理論的一致 | ❌ 乖離あり | ✅ 完全一致 |
| 発散ガード | clamp(±2.0) が必要 | clamp(±1e12) で十分 |

→ `advanceState` 内の `kLatticeStateLimit = 2.0` は緩和または撤廃可能。

### 影響3: CMA-ES の収束性が改善する可能性（推測）

現在の IIR 積分器構造:

- clamp による非線形性で目的関数の landscape が複雑
- 高ゲイン時の振る舞いが線形モデルと乖離

CMSIS-DSP 準拠の純粋 FIR:

- 線形時不変 → 目的関数が滑らか
- CMA-ES がより安定して収束する可能性

### 影響4: 低域ノイズシェーピング特性（未知）

現在の IIR 構造は z=1 の極により低域で無限大の DC ゲインを持つ。CMSIS-DSP 準拠の FIR は有限の DC ゲイン。低域のノイズシェーピング能力が低下する可能性があるが、実際の可聴帯域での差は CMA-ES 再学習後の NTF 測定で確認が必要。

### 結論

| 観点 | リスク/影響 |
|------|-----------|
| NTF 変化 | ✅ 確定。再学習必須 |
| 学習係数の失効 | ✅ 確定。再学習必須 |
| isStable 整合性 | ✅ 改善。理論と実装が一致 |
| CMA-ES 収束性 | ⚠️ 改善する可能性。未確認 |
| 低域ノイズ特性 | ⚠️ 再学習後 NTF 測定が必要 |
| 発散リスク | ✅ 低減。FIR 格子は理論的に安定 |

**CMSIS-DSP 準拠＋CMA-ES 再学習のセットで適用すべき。片方だけの適用は危険。**

## アーキテクチャ評価の検証（正常部分）

レビュアーの以下の評価は**すべて正当**です:

1. **IR リバース並び替え (Mirror Write)**: ✅
   - `SetImpulse()` 内で `irFreqDomain` のパーティションを逆順に並び替え
   - `Add()` 内で `linStart + p` によるメモリ順次アクセス（キャッシュフレンドリ）
   - CPU プリフェッチ (`_mm_prefetch`) と組み合わせて高効率

2. **Intel IPP FFT 換装**: ✅
   - MKL DFTI → IPP への換装でオーディオスレッドのスレッドプール問題を回避
   - `IppFFTPlanCache` による FFT スペックのキャッシュ＆再使用
   - ワークバッファ事前確保で Audio Thread 内メモリ確保ゼロ

3. **SSE2/FMA ステレオ Biquad パッキング**: ✅
   - `OutputFilter.cpp` で L/R を `__m128d` にパック
   - 3段カスケード DF-II Transposed を FMA 同時演算

---

## 付録: 検証に使用したツール

| ツール | 使用目的 | 結果 |
|--------|---------|------|
| AiDex MCP | シンボル検索（MKLNonUniformConvolver, LatticeNoiseShaper） | 16件/9件ヒット |
| Serena MCP | `advanceState`/`computeFeedback`/`processSample` のボディ取得・依存関係 | 正確な関数シグネチャ取得 |
| grep/Select-String | SetImpulse/Add/processLayerBlock の行特定 | 全処理フローの追跡 |
| CodeGraph MCP | アーキテクチャ全体図 | 対象外のため未使用 |
| semble CLI | 追加検証 | 情報量十分のため省略 |
| インターネット文献 | 格子フィルタ理論（Haykin, Wefers, Gardner） | 理論的裏付け確認済み |
