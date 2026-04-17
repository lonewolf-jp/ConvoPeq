# partition配列 完全逆解析

## 1. 全体構造: Non-Uniform Partitioned Convolution (NUC)

ConvoPeq の IR処理は **Intel MKL ベースの3層非統一分割畳み込み**を実装しています。

```
┌─────────────────────────────────────────────────────┐
│ IR (Impulse Response)                               │
│ Total length: irLen samples                         │
└─────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    ┌────────┐      ┌────────┐       ┌────────┐
    │ Layer 0│      │ Layer 1│       │ Layer 2│
    │(Head)  │      │(Middle)│       │(Tail)  │
    └────────┘      └────────┘       └────────┘
      Direct          FFT L1           FFT L2
      Form
```

---

## 2. 各層の詳細仕様

### Layer 0: Direct Form (HEAD PARTITION)

**定義箇所**: `MKLNonUniformConvolver.cpp:534`

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
```

| 項目 | 式 | 説明 |
|------|-----|------|
| **パーティションサイズ** | `l0Part = nextPowerOfTwo(max(blockSize, 64))` | 最小64, ブロックサイズに2の累乗調整 |
| **最大IRカバー長** | `l0Len = min(irLen, kL0MaxParts * l0Part)` | L0は最大32パーティション |
| **処理方式** | Direct FIR タップ演算 | メモリ効率重視（FFTなし） |
| **latency** | `l0Part - 1` サンプル | 最初のIRサンプルが出力に現れるまで |
| **headPartitionSize** | **`l0Part`** | これが **実質的な headPartition** |

**具体例** (blockSize=512):

- `l0Part = nextPowerOfTwo(512) = 512`
- `headPartitionSize = 512`
- `headLatency = 511 samples`

---

### Layer 1: FFT Middle

**定義箇所**: `MKLNonUniformConvolver.cpp:538`

```cpp
const int l1Part = l0Part * 8;
```

| 項目 | 式 |
|------|-----|
| **パーティションサイズ** | `l1Part = l0Part * 8` |
| **最大IRカバー長** | `l1Len = min(irLen - l0Len, kL1MaxParts * l1Part)` |
| **処理方式** | Overlap-Add FFT |

**例**: blockSize=512 の場合

- `l1Part = 512 * 8 = 4096`
- `numL1Partitions = ceil(remainingIR / 4096)`

---

### Layer 2: FFT Tail

**定義箇所**: `MKLNonUniformConvolver.cpp:539`

```cpp
const int l2Part = l1Part * 8 = l0Part * 64;
```

| 項目 | 式 |
|------|-----|
| **パーティションサイズ** | `l2Part = l0Part * 64` |
| **IR カバー** | 残り全部 (`l2Len = irLen - l0Len - l1Len`) |
| **処理方式** | Overlap-Add FFT |

**例**: blockSize=512 の場合

- `l2Part = 512 * 64 = 32768`

---

## 3. headPartition の正体

```cpp
// ✔ これが headPartition
int headPartitionSize = juce::nextPowerOfTwo(std::max(blockSize, 64));
```

**重要**:

- `headPartitionSize` ≠ FFTサイズ / 2
- `headPartitionSize` = **最初のパーティションのサイズ** (L0層)
- L0層は Direct FIR で処理される（FFTではない）

---

## 4. 実コードでのデータレイアウト

### メモリ配置（ConvolverState内）

```
partitionData (mkl_malloc, 64-byte aligned)
├─ L0 IR周波数領域: numL0Parts × fftSize × sizeof(double)
├─ L1 IR周波数領域: numL1Parts × fftSize × sizeof(double)
└─ L2 IR周波数領域: numL2Parts × fftSize × sizeof(double)
```

**計算例** (blockSize=512, irLen=48000 @ 48kHz):

```
l0Part = 512
l0Len = min(48000, 32 * 512) = 16384 samples

l1Part = 4096
l1Len = min(31616, 32 * 4096) = 31616 samples

l2Part = 32768
l2Len = 0 samples

numL0Partitions = ceil(16384 / 512) = 32
numL1Partitions = ceil(31616 / 4096) = 8

partitionSizeBytes = (32 + 8) * fftSize * sizeof(double)
                   = 40 * fftSize * 8 bytes
```

---

## 5. partition計算ロジック (IRConverter.cpp:183)

```cpp
// FFTサイズ決定（config.fftSize が入力）
const int fftSize = juce::jmax(32, config.fftSize);

// partition数計算 (ceil division)
const int numPartitions = juce::jmax(1, (samples + fftSize - 1) / fftSize);

// メモリ確保
const size_t totalSamples = (size_t)numPartitions * fftSize * usableChannels;
const size_t bytes = totalSamples * sizeof(double);

double* data = mkl_malloc(bytes, 64);  // ← 64-byte aligned
```

**この計算は L1/L2 FFTパーティション用**。L0は別途Direct Formで計算。

---

## 6. 実装の FFT Plan 生成

**ConvolverState.cpp:157-162**:

```cpp
DFTI_DESCRIPTOR_HANDLE h = nullptr;
if (DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1,
                         static_cast<MKL_LONG>(fftSize)) != DFTI_NO_ERROR)
{
    throw std::runtime_error("ConvolverState: DftiCreateDescriptor failed");
}
fftHandle.reset(h);
```

**重要**: FFT計画は **単一サイズ** (`fftSize`) で生成される。

- L0はFFTなし（Direct Form）
- L1/L2は同じ FFTサイズで複数パーティション処理

---

## 7. setFixedLatency への統合

```cpp
// ✔ headPartition を遅延補正に使用
int headPartitionSize = juce::nextPowerOfTwo(std::max(blockSize, 64));
convolver.setFixedLatency(headPartitionSize - 1);  // or headPartitionSize

// → これが正しい（L0パーティション遅延 = headPartitionSize - 1）
```

---

## 8. avoidables チェックリスト

✅ **headPartitionSize = `nextPowerOfTwo(max(blockSize, 64))`**
✅ **これは L0パーティションサイズ（最初のレイヤー）**
✅ **L0は Direct FIR（FFTなし）**
✅ **L1/L2は FFT パーティション**
✅ **partitionData は全層を1つの MKL確保ブロックで管理**
✅ **各層は progressively larger partitions**

---

## 9. まとめ表

| 項目 | L0 Head | L1 Middle | L2 Tail |
|------|---------|-----------|---------|
| **パーティションサイズ** | `nextPowerOfTwo(max(blockSize, 64))` | L0Part × 8 | L0Part × 64 |
| **処理方式** | Direct FIR | FFT Overlap-Add | FFT Overlap-Add |
| **最大カバー** | kL0MaxParts × L0Part (32K) | kL1MaxParts × L1Part | 残り全部 |
| **遅延** | L0Part - 1 | L1Part / 2 | L2Part / 2 |
| **FFT Plan** | なし | DftiCreateDescriptor(fftSize) | 同じ FFT Plan 共有 |

---

## 10. コード特定（行番号付き）

| 関数/場所 | ファイル | 行 | 内容 |
|-----------|---------|------|------|
| `buildLayers()` | MKLNonUniformConvolver.cpp | 486-580 | 全層構築 |
| L0パーティション | MKLNonUniformConvolver.cpp | 486-490 | `l0Part = nextPowerOfTwo(...)` |
| L1パーティション | MKLNonUniformConvolver.cpp | 534-538 | `l1Part = l0Part * 8` |
| L2パーティション | MKLNonUniformConvolver.cpp | 539 | `l2Part = l1Part * 8` |
| partition計算 | IRConverter.cpp | 183 | `numPartitions = (samples + fftSize - 1) / fftSize` |
| メモリ確保 | IRConverter.cpp | 189-198 | `mkl_malloc(bytes, 64)` |
| FFT Plan生成 | ConvolverState.h | 157-162 | `DftiCreateDescriptor()` |

---

**結論**: `headPartitionSize` は **`nextPowerOfTwo(max(blockSize, 64))`** で、これが最初のDirect Form レイヤーのパーティションサイズであり、全3層構造の根幹となる値です。
