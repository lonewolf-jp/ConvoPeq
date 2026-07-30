# BUG-041: NoiseShaperLearner::evaluatePopulation — VLA（可変長配列）によるスタック破壊

## 発見日
2026-07-26

## ファイル
`src/NoiseShaperLearner.cpp:643`

## 問題
`evaluatePopulation()` 内で可変長配列（VLA）をスタックに確保している:

```cpp
alignas(64) double tanhBuffer[totalCoeffs] = {};
```

ここで `totalCoeffs = CmaEsOptimizer::kPopulation * CmaEsOptimizer::kDim`。

### 想定サイズ
- `kPopulation` と `kDim` の値次第だが、CMA-ES の典型的な設定:
  - `kPopulation`: 4 + floor(3 * log(kDim)) 程度（例: kDim=64 → population≈16）
  - `kDim` = `LatticeNoiseShaper::kOrder`（通常 16〜64）
  - `totalCoeffs` = 16 × 64 = 1024 double → 約 8 KB
- ただし、実装によっては population が大きく（例: 200+）、kDim も大きい場合:
  - `totalCoeffs` = 200 × 64 = 12800 double → 約 102 KB

### 問題点
1. **VLA は C++ 標準外**（C99 の機能）であり、MSVC は VLA をサポートしていない
2. MSVC では `constexpr` でない配列サイズはコンパイルエラー
3. 仮にコンパイルが通った場合でも、スタックサイズは通常 1MB 以下（Windows のデフォルト）であり、大きな population ではスタックオーバーフローによるクラッシュ

### 実際の動作
MSVC ではコンパイルエラーになるはずだが、回避策（`/Ze` または `__pragma`）がある可能性。少なくとも移植性の問題がある。

### 修正方針
`tanhBuffer` をスタックではなくヒープに確保する。`sharedMappedPopulation` と同様に `ScopedAlignedPtr` を使用するか、`std::vector<double>` に `alignas` を適用する。
