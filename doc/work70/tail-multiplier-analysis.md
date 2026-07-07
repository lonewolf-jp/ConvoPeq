# tailL1L2Multiplier メモリ増幅分析

**調査日**: 2026-07-07
**根拠コード**: `src/MKLNonUniformConvolver.cpp` L514-617

---

## 1. multiplier の決定ロジック

3つの経路があり、モードによって下限が強制される:

```cpp
// L514: デフォルト
tailL1L2Mult = jlimit(2, 16, filterSpec->tailL1L2Multiplier)    → 既定値 8

// L534: tailMode==0 (Air Absorption)
tailL1L2Mult = jlimit(2, 16, max(tailL1L2Mult, 6))              → 最小 6

// L544: tailMode==1 (Layer Tail Contouring)  ← デフォルト
tailL1L2Mult = jlimit(2, 16, max(tailL1L2Mult, 12))             → 最小 12
```

**デフォルト動作では multiplier ≧ 12 が強制される。**

## 2. レイヤーサイズ計算

```cpp
l0Part = nextPowerOfTwo(max(blockSize, 64))
l1Part = l0Part * tailL1L2Mult
l2Part = l1Part * tailL1L2Mult = l0Part * tailL1L2Mult²
```

## 3. メモリ試算（blockSize = 1024 の場合）

### 3.1 L2 small buffers の内訳

multiplier=12 (tailMode=1のデフォルト):

| バッファ | 要素数 | サイズ |
|---------|--------|-------|
| fftTimeBuf | 294,912 | 2.25 MB |
| fftOutBuf | 294,912 | 2.25 MB |
| prevInputBuf | 147,456 | 1.13 MB |
| inputAccBuf | 147,456 | 1.13 MB |
| accumBuf | 294,920 | 2.25 MB |
| accumReal | 147,457 | 1.13 MB |
| accumImag | 147,457 | 1.13 MB |
| tailOutputBuf | 147,456 | 1.13 MB |
| **合計** | | **12.4 MB** |

### 3.2 multiplier 別比較

| mult | l2Part | small buffers | 384kHz/3sIRでのL2利用率 |
|------|--------|---------------|------------------------|
| **4** | 16,384 | 1.4 MB | 多数のパーティションに分割 |
| **6** | 36,864 | 3.1 MB | 5 parts @3s / 効率的 |
| **8** | 65,536 | 5.5 MB | 1 part @3s / やや無駄 |
| **10** | 102,400 | 8.6 MB | 0-1 part / 無駄 |
| **12** | **147,456** | **12.4 MB** | **0 part / 完全無駄** |
| **14** | 200,704 | 16.8 MB | 0 part / 大損 |
| **16** | 262,144 | 22.0 MB | 致命的 |

### 3.3 重要な発見

**mult=12 の場合、3秒IR @384kHz では L2 に割り当てられるデータがゼロになる**:
```
L0: 32 * 1024 = 32,768  samples  (先頭85ms)
L1: 64 * 12,288 = 786,432 samples  (残り2.05s)
L2: 残り 0 samples → L2part=0
```

→ **12.4 MB の L2 small buffers が完全に無駄になっている。**

IRが5.46s（`MAX_IR_LATENCY`上限）の場合も、L2はわずか2-3パーティションにしかならず、12.4MBのsmall buffersは依然として過大。

## 4. 改善案

### 案A: tailMode=1 の最小 multiplier を 12→8 に緩和する（即効・低リスク）

```cpp
// 変更前 (L544):
tailL1L2Mult = juce::jlimit(2, 16, std::max(tailL1L2Mult, 12));
// 変更後:
tailL1L2Mult = juce::jlimit(2, 16, std::max(tailL1L2Mult, 8));
```

**効果**: small buffers 12.4 MB → **5.5 MB**（56%削減）
**副作用**: L2のパーティション数が増加する（3s IR時: 0→1, 5s IR時: 2→6）。
**CPU影響**: IFFT回数がやや増えるが、partsPerCallback 分散によりスパイクは抑制される。
**音響影響**: L2の partSize が65,536サンプル（@384kHzで170ms）→ テールのレイテンシが短くなる方向。良好。

### 案B: 非対称 multiplier を採用する（中程度の工数）

```cpp
// L1は大きめに、L2は控えめに
l1Part = l0Part * 12;    // 即時性を維持
l2Part = l1Part * 6;     // テールは小さめで効率的
```

または:

```cpp
l1Part = l0Part * 8;
l2Part = l1Part * 8;     // 8倍で統一（Garcia 2002 の推奨値に近い）
```

**効果**: L2 small buffers 5.5 MB。L1のカバレッジが減少するためL2パーティションが増える。
**注意**: `FilterSpec` 構造体に非対称用のフィールド追加が必要。

### 案C: 4層設計に拡張する（大工数・次フェーズ推奨）

```
L0: partSize = 1024,   maxParts=32  (即時, 先頭32KB)
L1: partSize = 4096,   maxParts=32  (分散, 32KB-160KB)
L2: partSize = 16384,  maxParts=64  (分散, 160KB-1.2MB)
L3: partSize = 65536,  maxParts=∞   (テール, 1.2MB以降)
```

**効果**: 各層のpartSizeが穏やかに増加 → L2/L3のsmall buffersが適正化。
**副作用**: 4層目のdispatchロジック追加、コード変更量が多い。

### 案D: multiplier の下限を状況に応じて動的に選択する（最もスマート）

```cpp
// IRが短い場合は大きなmultiplier、長い場合は小さなmultiplier
const double irLenSec = static_cast<double>(irLen) / sampleRate;
if (irLenSec > 3.0)
    tailL1L2Mult = std::max(tailL1L2Mult, 8);  // 長IR → 小さめ
else
    tailL1L2Mult = std::max(tailL1L2Mult, 12); // 短IR → 大きめでも問題なし
```

## 5. 推奨

| 優先度 | 案 | 効果 | 工数 | リスク |
|--------|-----|------|------|--------|
| 🥇 | **案A: 最小12→8** | 56%削減 | 1行 | 低（ただし要性能確認） |
| 🥈 | **案D: 動的選択** | 状況最適化 | 3行 | 低 |
| 🥉 | 案B: 非対称 | バランス | 小 | 中 |
| 次期 | 案C: 4層 | 理想解 | 大 | 中 |

## 6. "2.5GBとの差分" 再考

Plan.mdの§6では「AoS除去後も1GBに届かない場合の計測方法」が示されている。
本分析から、**mult=12によるL2 small buffers 12.4MB/ch の無駄**も積み上がる可能性がある:

| 要素 | ステレオ | active/fading二重 |
|------|---------|-----------------|
| L2 small 12.4MB | ×2ch = 24.8MB | ×2 = 49.6MB |

これは2.5GBの約2%にすぎず、主要因ではない。AoS除去＋mult調整で total の大部分が説明可能になる。
