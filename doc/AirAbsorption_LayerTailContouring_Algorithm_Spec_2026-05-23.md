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

- `tailEnabled == false`（`tailMode == 2` を含む）

`tailEnabled` は `tailMode != 2` と `FilterSpec::tailEnabled` の両方が真のときだけ真になる。

結果:

- `tailStartSec` は 0.12 秒の下限へ再クランプ
- `tailStrength = 1.0`
- `layer1Gain = 1.0`
- `layer2Gain = 1.0`
- `m_tailEnabled = true`（読み出し側の層ゲインもすべて 1.0）

Bypass は層を無効化する処理ではない。L0 + L1 + L2 で IR 全体を覆い、すべての層を等倍で
加算する。L0 の目標長も tail 有効時と同じ 0.12 秒下限を使う。したがって
「`Get()` で L1/L2 からの加算寄与は実質ゼロになる」という記述は誤りである。

---

### 3.2 Air Absorption（`tailMode == 0`）

#### (A) モード内再クランプ（Air専用下限）

- `tailStartSec = clamp([0.01, 0.80], max(tailStartSec, 0.055))`
- `tailL1L2Mult = clamp([2, 16], max(tailL1L2Mult, 6))`
- `tailStrength = clamp([0.0, 2.0], userTailStrength)`

#### (B) 時間位置ゲイン

- $g_1 = clamp([0,2],\ tailStrength \cdot (0.95 - 0.25\cdot strength01))$
- $g_2 = clamp([0,2],\ tailStrength \cdot (0.80 - 0.45\cdot strength01))$

実装格納:

- `m_tailLayerGain[0..2] = 1.0`（`Get()` ではゲインを掛けない）
- $g_1$ / $g_2$ は `SetImpulse()` で IR の各サンプルへ掛ける（§4.1）

B-12 以前は `m_tailLayerGain[1] = g1` / `m_tailLayerGain[2] = g2` とし、`Get()` で層ごとに
掛けていた。層の被覆区間は block サイズで変わるため、同じ IR でも block によって実効ゲインが
変わっていた。

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

#### (B) 時間位置ゲイン

- $g_1 = clamp([0,2],\ tailStrength \cdot (1.05 + 0.20\cdot strength01))$
- $g_2 = clamp([0,2],\ tailStrength \cdot (0.82 + 0.12\cdot strength01))$

実装意図は「L1 主軸、L2 補助」の輪郭形成。適用位置は §4.1 の時間位置ゲインで決まり、層の
被覆区間とは独立である。

---

## 4. パーティション構成への反映

`SetImpulse()` で次を決定:

- `l0Part = nextPowerOfTwo(max(blockSize, 64))`
- `l1Mult` は 1 から始めて、`2 * l1Mult <= tailL1L2Mult`、`2 * l1Mult <= 4`、
  `2 * l1Mult * l0Part <= max(l0Part, l0LenByTailStart)` の 3 条件を満たす間 2 倍した値
- `l1Part = l0Part * l1Mult`
- `l2Part = l1Part * tailL1L2Mult`

`l1Mult` は層の構成を決める値であり、§4.1 の `refL1Mult` とは別の計算値である。`refL1Mult` は
テール整形ゲインの時間境界を決めるための参照値で、`l0Part` に依存しない。

L0 の長さ:

- `l0LenByTailStart = round(tailStartSec * sampleRate)`
- `l0CoverageAligned = ceil(l0LenByTailStart / l0Part) * l0Part`（I5 の整除条件を満たすための切り上げ）
- `l0LenTarget = clamp([l0Part, l0MaxLen], l0CoverageAligned)`
- `l0Len = min(irLen, l0LenTarget)`

`l0LenTarget` が `l0Part` の倍数へ切り上げられる点が、§4.1 の `b1` との差の原因である。

L1/L2:

- 残り IR を L1→L2 へ割り当てる（Bypass でも同じ。§3.1）
- `l1Len = max(0, min(irLen - l0Len, kL1MaxParts * l1Part))`
- `l2Len = max(0, irLen - l0Len - l1Len)`

### 4.1 テール整形ゲインの適用位置（B-12）

IR 上のサンプル位置 $t$ に対して、次のゲインを掛ける。

- $t < b_1$ : 1.0
- $b_1 \le t < b_2$ : $g_1$
- $b_2 \le t$ : $g_2$

境界:

- $b_1 = \mathrm{jlimit}(1, irLen, \mathrm{llround}(tailStartSec \cdot sampleRate))$
  - `tailStartSec` は §3.2(A) / §3.3(A) のモード内再クランプ後の値
  - `sampleRate` は `FilterSpec::sampleRate`
- $b_2 = b_1 + kL1MaxParts \cdot kRefL0Part \cdot refL1Mult$
  - $kRefL0Part = 256$、$kL1MaxParts = 64$
  - $refL1Mult$ は 1 から始めて $2 \cdot refL1Mult \cdot 256 \le \max(256, b_1)$ を満たす間
    2 倍し、4 と `tailL1L2Mult` で頭打ちにした値
  - $b_2$ の項だけが block 256 の L1 被覆長（$64 \times 1024$ サンプル）に対応する

適用先:

- FFT 経路: `impulseForFft[i] = impulse[i] * gain(i)`
- direct head: `m_directIRRev[i] = impulse[irIndex] * scale * gain(irIndex)`

$b_1$ は時間指定から直接求めるため、パーティション格子（`l0Part`）に整列しない。旧実装は
L0 長を `l0Part` へ切り上げていたので、block 256 でも旧実装とは $b_1$ / $b_2$ が最大
`l0Part - 1` サンプルずれる。48 kHz・mode 1 では 128 サンプル（2.67 ms）、全 IR エネルギー差
+0.024 dB である。

この差は `tailStartSeconds`（Tail 開始目安・秒）の指定を優先する判断として許容する。
block 256 との完全一致は要求しない。

### 4.2 block 不変性の保証範囲（B-12）

- **tailMode 1（既定）と tailMode 2（Bypass）**: ゲインは IR のサンプルへ掛かるため、実効 IR は
  分割に依存しない。2 の冪の block 64〜524288 の 14 種類で実測し、最大差 4.4e-16（倍精度の
  丸め誤差）を確認した。direct head を有効にした場合も、direct 経路と FFT 経路の合計は連続する
  （最大 3.9e-16）。
- ただし 14 種類は全整数値ではない。また `MKLNonUniformConvolver::kMaxSupportedBlockSize`
  （32768）を超える block は、`SetImpulse()` が性能保証外として 1 回警告する。幾何は健全に
  保たれるが、性能は保証されない。したがって「すべての block サイズで保証する」とは書かない。
- **tailMode 0（Air Absorption）**: §3.2(C) の HF 減衰が層の `complexSize` に依存するため、
  実効 IR は block に依存したままである。実測では block 64〜524288 の実効 IR の最大差が
  4.7e-2、16 kHz 帯の偏りが block 1024 で −3.19 dB、block 524288 で 0 dB だった。B-12 は
  この機構に触れていない。別課題として扱う。

### 4.3 音響的な位置づけ（未判定）

$g_1$ / $g_2$ の値と $b_1$ / $b_2$ の位置は、RT60・吸音率・聴感目標のいずれからも導出されて
いない。したがって「音響的に正しいゲイン分布」であることは判定していない。B-12 が保証するのは
「§3.2(B) / §3.3(B) の式を IR の時間位置へ適用し、tailMode 1/2 で block 依存を除去した」こと
までである。

実測した音響指標（RT60 1〜3 s の減衰ノイズ IR、48 kHz、tailStrength 1.0）:

- 振幅特性の変化: 1/3 オクターブ平滑で最大 +2.17 dB、RMS 1.71 dB。帯域による偏りは ±0.15 dB
  で、音色を作るフィルタではなくテール全体のレベル変化として働く
- 全 IR エネルギー: RT60 2 s で +1.69 dB
- T30: RT60 1 s で +0.1 %、2 s で −1.0 %、3 s で −7.7 %
- $b_1$ の段差: IR ピークの 5〜9 %。$b_2$ の段差: 0.2 % 未満

旧実装との比較値には 2 種類ある。証拠レベルが違うので混同しないこと。

| 数値 | 証拠レベル | 出典 |
| --- | --- | --- |
| 3.15 dB / 1.66 dB / +0.024 dB | 復元した旧ジオメトリとの比較計算値。修正前バイナリとの実測ではない | 本節と §4.1 |
| 2.41 dB / 3.15 dB / 0.83 dB | 修正前後の実測値。2.41 は block 256 基準、残り 2 つは生 IR 基準 | `doc/work113/20261110-1130_未修正修正/report.md` §3.3 |

復元値は、変更されていない幾何コード（§4 と旧 `Get()` の層ゲイン適用）から旧被覆区間を
求めて計算した。旧バイナリは残っていない。復元値と §3.3 の実測値は整合する。

---

## 5. Audio Thread 側の合成（`Get()`）

`Get(output, n)` の流れ:

1. L0（即時層）のリングバッファを `ringRead`
2. 直達（direct head）が有効なら `m_directOutBuf` を加算
3. L1/L2 の `tailOutputBuf` を加算

L1/L2 加算式（レイヤー `li`）:

$$
output[i] \mathrel{+}= tailOutputBuf[i] \cdot m\_tailLayerGain[li]
$$

B-12 後は `m_tailLayerGain[li]` がすべて 1.0 である。テール整形のゲインは `SetImpulse()` で
IR のサンプルへ掛けてあるため、Audio Thread 側では何も掛けない（§4.1）。`m_tailLayerGain` は
build 時に確定済みで、Audio Thread 内で再計算しない点は変わらない。

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
  - `SetImpulse(...)`: モード別導出、パーティション決定、時間位置ゲインの IR への適用、Air HF damping
  - `Get(...)`: L0 + direct + L1/L2 合成（層ゲインは常に 1.0）
- `src/MKLNonUniformConvolver.h`
  - `FilterSpec`
  - `m_tailLayerGain[kNumLayers]`

---

## 8. 変更時の推奨手順

1. 本書 3章（モード別式）・4章（パーティション）・5章（合成）を先に更新
2. 実装を更新
3. `Strict Atomic Dot-Call Scan` 実施
4. Debug Build 実施
5. `doc/work/ISR_AirTail_統合設計_2026-05-22.md` との整合を再確認
6. `RELEASE_NOTES.md` の `Unreleased` に挙動変更を記載（利用者に見える変化がある場合）

B-12 では 1 と 2 を同一 PR で行った。以降も 1 と 2 は同一の変更単位として扱う。
