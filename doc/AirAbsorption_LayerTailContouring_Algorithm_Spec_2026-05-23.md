# Air Absorption / Layer Tail Contouring 計算アルゴリズム仕様（ConvoPeq）

作成日: 2026-05-23
対象実装: `src/MKLNonUniformConvolver.cpp`（`SetImpulse()` / `Get()`）

---

## 1. 目的

今後の改修時に、`Air Absorption` と `Layer Tail Contouring` の計算ロジックを
**コードと同じ粒度**で参照できるよう、アルゴリズムを文書化する。

本書は実装の SoT ではなく、実装追従の仕様メモである。
SoT/責務境界は `doc/work/ISR_AirTail_統合設計_2026-05-22.md` を正本とする。

---

## 2. 入力パラメータと境界

`FilterSpec` から受ける主要パラメータ（`SetImpulse()` 冒頭）:

- `tailMode`（0=Air Absorption, 1=Layer Tail Contouring, 2=Bypass）
- `tailEnabled`
- `tailStartSeconds`
- `tailStrength`
- `tailL1L2Multiplier`
- `sampleRate`

実装での初期クランプ:

- `tailMode = clamp([0, 2])`
- `tailStartSec = clamp([0.01, 0.80])`
- `userTailStrength = clamp([0.0, 2.0])`
- `tailL1L2Mult = clamp([2, 16])`

補助値:

- `strength01 = clamp([0.0, 1.0], userTailStrength * 0.5)`

---

## 3. モード別アルゴリズム

### 3.1 Bypass / tail 無効

判定:

- `tailEnabled == false` または `tailMode == 2`

結果:

- `tailStrength = 0`
- `layer1Gain = 0`
- `layer2Gain = 0`

この場合、`Get()` で L1/L2 からの加算寄与は実質ゼロになる。

---

### 3.2 Air Absorption（`tailMode == 0`）

#### (A) モード内再クランプ（Air専用下限）

- `tailStartSec = clamp([0.01, 0.80], max(tailStartSec, 0.055))`
- `tailL1L2Mult = clamp([2, 16], max(tailL1L2Mult, 6))`
- `tailStrength = clamp([0.0, 2.0], userTailStrength)`

#### (B) 層別ゲイン（Layer）

- $g_1 = clamp([0,2],\ tailStrength \cdot (0.95 - 0.25\cdot strength01))$
- $g_2 = clamp([0,2],\ tailStrength \cdot (0.80 - 0.45\cdot strength01))$

実装格納:

- `m_tailLayerGain[1] = g1`
- `m_tailLayerGain[2] = g2`

#### (C) 高域減衰（HF damping）を L1/L2 の周波数領域IRへ焼き込み

追加で以下を計算:

- `startNorm = clamp([0.65, 1.55], tailStartSec / 0.085)`
- `dampingBase = (0.35 + 1.10 * strength01) * startNorm`

レイヤーごと係数:

- `layerWeight = 1.0`（L1）/ `1.6`（L2）
- `dampingCoeff = dampingBase * layerWeight`

周波数ビンごとのチルト:

$$
fNorm = \frac{k}{\max(1, complexSize-1)}
$$

$$
hfTilt(k) = \exp(-dampingCoeff\cdot fNorm^2)
$$

`hfTilt` を interleaved complex の実部/虚部両方へ乗算し、
その後 `deinterleaveComplex(...)` で SoA バッファ（`irFreqReal/Imag`）を再生成する。

---

### 3.3 Layer Tail Contouring（`tailMode == 1`）

#### (A) モード内再クランプ（Layer専用下限）

- `tailStartSec = clamp([0.01, 0.80], max(tailStartSec, 0.12))`
- `tailStrength = clamp([0.0, 2.0], max(tailStrength, 1.25))`
- `tailL1L2Mult = clamp([2, 16], max(tailL1L2Mult, 12))`

#### (B) 層別ゲイン

- $g_1 = clamp([0,2],\ tailStrength \cdot (1.05 + 0.20\cdot strength01))$
- $g_2 = clamp([0,2],\ tailStrength \cdot (0.82 + 0.12\cdot strength01))$

実装意図は「L1 主軸、L2 補助」の輪郭形成。

---

## 4. パーティション構成への反映

`SetImpulse()` で次を決定:

- `l0Part = nextPowerOfTwo(max(blockSize, 64))`
- `l1Part = l0Part * tailL1L2Mult`
- `l2Part = l1Part * tailL1L2Mult`

L0 の長さ:

- `l0LenByTailStart = round(tailStartSec * sampleRate)`
- `l0LenTarget = clamp([l0Part, l0MaxLen], l0LenByTailStart)`
- `l0Len = min(irLen, tailEnabled ? l0LenTarget : l0MaxLen)`

L1/L2:

- `tailEnabled == false` の場合は `l1Len = l2Len = 0`
- `tailEnabled == true` の場合は残りIRを L1→L2 へ割り当て

---

## 5. Audio Thread 側の合成（`Get()`）

`Get(output, n)` の流れ:

1. L0（即時層）のリングバッファを `ringRead`
2. 直達（direct head）が有効なら `m_directOutBuf` を加算
3. L1/L2 の `tailOutputBuf` を層別ゲインで加算

L1/L2 加算式（レイヤー `li`）:

$$
output[i] \mathrel{+}= tailOutputBuf[i] \cdot m\_tailLayerGain[li]
$$

ここで `m_tailLayerGain` は build 時に確定済みで、Audio Thread 内で再計算しない。

---

## 6. 実装上の不変条件（改修時チェックリスト）

1. `SetImpulse()` 以外で `m_tailLayerGain[]` を更新しない
2. `Get()` 内で mode 分岐計算を追加しない（read-only 合成のみ）
3. Air の HF damping を入れる場合は `irFreqDomain` と SoA の整合更新をセットで行う
4. `tailEnabled` 判定はクランプ済み `tailMode` に整合させる
5. Audio Thread で alloc / lock / log / FFT再初期化を行わない

---

## 7. 参照実装箇所

- `src/MKLNonUniformConvolver.cpp`
  - `SetImpulse(...)`: モード別導出、パーティション決定、Air HF damping
  - `Get(...)`: L0 + direct + L1/L2層別ゲイン合成
- `src/MKLNonUniformConvolver.h`
  - `FilterSpec`
  - `m_tailLayerGain[kNumLayers]`

---

## 8. 変更時の推奨手順

1. 本書 3章（モード別式）と 4章（パーティション）を先に更新
2. 実装を更新
3. `Strict Atomic Dot-Call Scan` 実施
4. Debug Build 実施
5. `doc/work/ISR_AirTail_統合設計_2026-05-22.md` との整合を再確認
