# コンボルバーバグ改修計画書

**バージョン**: 2.0（全未確定事項確定済み）
**作成日**: 2026-06-23
**最終確定日**: 2026-06-23
**対象レポート**: `doc/work55/bug.md`, `doc/work55/ConvoPeq_Convolver_Validation_Report.md`, `doc/work55/review_verification_report.md`

---

## 目次

1. [改修対象一覧](#1-改修対象一覧)
2. [F1: Mixed Phase クロスオーバー方向修正](#2-f1-mixed-phase-クロスオーバー方向修正)
3. [F2: mixedPreRingTau パラメータの有効化または削除](#3-f2-mixedpreringtau-パラメータの有効化または削除)
4. [F3: r8brain IRテイル切り捨て修正](#4-f3-r8brain-irテイル切り捨て修正)
5. [F4: computeMasteringSizing 孤立計算の解消](#5-f4-computemasteringsizing-孤立計算の解消)
6. [F5: applyAllpassToIR 重複実装の解消](#6-f5-applyallpasstoir-重複実装の解消)
7. [F6: レイヤースケジューリング最適化（調査・将来課題）](#7-f6-レイヤースケジューリング最適化調査将来課題)
8. [検証計画](#8-検証計画)
9. [依存関係と実施順序](#9-依存関係と実施順序)

---

## 1. 改修対象一覧

| ID   | 項目                                        | 優先度   | 工数見積             | リスク   | 依存                     |
| --- | --- | --- | --- | --- | --- |
| F1  | Mixed Phase クロスオーバー方向修正          | **高**   | 小（変数2行）         | 中（音響的影響大） | なし                     |
| F2  | `mixedPreRingTau` パラメータ有効化          | 中       | 中（設計＋実装）       | 低       | F1（同じファイル）        |
| F3A | `IRDSP::resampleIR()` バッファマージン修正   | **高**   | 小（数行）             | 低       | なし                     |
| F3B | メインローダーパスの健全性確認              | 低       | 小（追跡確認のみ）     | なし     | なし                     |
| F4A | `computeMasteringSizing` からNUCへの配線復活 | 低       | 中（シグネチャ変更波及）| 低       | なし                     |
| F4B | または関数削除＋コード整理                  | 低       | 小                     | 低       | なし                     |
| F5  | `applyAllpassToIR` 削除または統合            | 低       | 小                     | 低       | なし                     |
| F6  | レイヤースケジューリング最適化              | 情報     | 大（研究＋実装）       | 中       | なし                     |

---

## 2. F1: Mixed Phase クロスオーバー方向修正

### 2.1 現状

`ConvolverProcessor.MixedPhase.cpp` の `convertToMixedPhaseAllpass()`（L303-311）および `convertToMixedPhaseFallback()`（L810-820）において、以下の重み付けが実装されている：

```
freq < transitionLoHz (default 200Hz):  wLinear=1.0, wMinimum=0.0 → Linear Phase のみ
freq > transitionHiHz (default 1000Hz): wLinear=0.0, wMinimum=1.0 → Minimum Phase のみ
```

これは **低域が Linear Phase、高域が Minimum Phase** という構成。

### 2.2 問題

ルームコレクション／室内音響補正分野の確立された慣行（Dirac Research、HomeAudioFidelity等）は、**低域=Minimum Phase（長時間プリリンギング回避）、高域=Linear Phase（プリリンギングが短時間で無害）** である。デフォルトクロスオーバー帯域（200/1000Hz）が業界標準と一致することから、実装意図と重みが入れ替わっている可能性が高い。

### 2.3 修正方針

**方針**: 重み関数 `wLinear`/`wMinimum` の相互に `1-w` 関係を反転する。

#### 修正箇所

**ファイル**: `src/convolver/ConvolverProcessor.MixedPhase.cpp`

**修正A: `convertToMixedPhaseAllpass()` 内（Allpass版）**

現行コード（L300-311）:

```cpp
double wLinear = 1.0;
if (freq >= transitionHiHz)
    wLinear = 0.0;
else if (freq > transitionLoHz)
{
    const double x = (freq - transitionLoHz) * invSpan;
    wLinear = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wMinimum = 1.0 - wLinear;
```

修正後:

```cpp
double wMinimum = 1.0;
if (freq >= transitionHiHz)
    wMinimum = 0.0;
else if (freq > transitionLoHz)
{
    const double x = (freq - transitionLoHz) * invSpan;
    wMinimum = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wLinear = 1.0 - wMinimum;
```

**修正B: `convertToMixedPhaseFallback()` 内（Fallback版）**

現行コード（L810-818）:

```cpp
double wLinear = 1.0;
if (freq >= transitionHiHz)
    wLinear = 0.0;
else if (freq > transitionLoHz)
{
    const double x = (freq - transitionLoHz) * invSpan;
    wLinear = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wMinimum = 1.0 - wLinear;
```

修正後:

```cpp
double wMinimum = 1.0;
if (freq >= transitionHiHz)
    wMinimum = 0.0;
else if (freq > transitionLoHz)
{
    const double x = (freq - transitionLoHz) * invSpan;
    wMinimum = 0.5 * (1.0 + std::cos(juce::MathConstants<double>::pi * x));
}
const double wLinear = 1.0 - wMinimum;
```

### 2.4 影響範囲

| 項目       | 影響                                                                         |
| ---------- | ---------------------------------------------------------------------------- |
| 音響的挙動 | **大**。IRの位相特性が反転。低域にMinimum Phase（プリリンギング抑制）、高域にLinear Phaseが適用される |
| UIパラメータ | 変更なし（`MIXED_F1_DEFAULT_HZ` / `MIXED_F2_DEFAULT_HZ` はそのまま）         |
| キャッシュ | 影響なし（修正後も同一キーで保存されるが、内容が変わるため既存キャッシュは無効化される。キャッシュの自動無効化は期待動作） |
| テスト     | 既存の Mixed Phase テストデータがある場合、期待値の再生成が必要               |

### 2.5 調査確定事項

以下の事項を追加調査で確定した：

| 確定項目 | 調査結果 |
| --- | --- |
| `convertToMixedPhase()` ラッパー関数への影響 | **影響なし**。ラッパーは `convertToMixedPhaseAllpass()` → `convertToMixedPhaseFallback()` の順に呼び出すのみで、クロスオーバー方向の知識を持たない |
| Fallback版の方向 | **Allpass版と同一**。両方とも低域=Linear / 高域=Minimum。両方の修正が必要 |
| コンパイルガード | `CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1`（`CMakeLists.txt` L536）で常時有効 |
| 呼び出し元 | `LoaderThread.cpp` L682 の1箇所のみ |
| 修正の対称性 | `wLinear` ↔ `wMinimum` の変数名交換と初期値反転のみ。コサインクロスフェードの数式構造は変更不要 |
| キャッシュキーへの影響 | `IRCacheKey` に方向情報が含まれていないため、修正後に既存キャッシュが誤った方向のIRを返すことはない（キャッシュミス→再計算） |
| 最小位相変換(`convertToMinimumPhase`) | Mixed Phase 方向とは無関係。修正不要 |

### 2.6 リスクと緩和策

| リスク                             | 確率 | 影響 | 緩和策                                                   |
| ---------------------------------- | ---- | ---- | -------------------------------------------------------- |
| 修正方向が音響的に望ましくない     | 低   | 中   | 実機IR測定（周波数応答・群遅延確認）の実施を推奨        |
| 既存ユーザーが期待する挙動と異なる | 中   | 低   | 修正が「意図通りの挙動」への変更であり、ドキュメントで周知 |
| 既存キャッシュが無効化され再設計が発生 | 高   | 低   | 初回ロード時のCPU負荷増加のみ。許容範囲                 |

---

## 3. F2: `mixedPreRingTau` パラメータの有効化または削除

### 3.1 現状

`mixedPreRingTau`（UI上は `MIXED_TAU_MIN=4.0`〜`MIXED_TAU_MAX=256.0`、既定 `MIXED_TAU_DEFAULT=32.0`）は：

- `convertToMixedPhaseAllpass()` 内で **キャッシュキーにのみ使用**される
- `AllpassDesignerConfig` 構造体に `tau` フィールドは存在しない
- `convertToMixedPhaseFallback()` では `(void)tau;` で明示的に無視
- `AllpassDesigner::design()` や `designWithCMAES()` のターゲット群遅延生成（`targetGroupDelay`）にも影響なし

つまり、ユーザーがUIでこの値を変更しても、出力される Mixed Phase IR は全く変わらない。

### 3.2 選択肢

#### 選択肢A: パラメータを実際の設計ロジックに結合する（推奨）

`AllpassDesignerConfig` に `tau`（プリリンギング制御）フィールドを追加し、CMA-ES/GreedyAdaGrad 最適化の目的関数にプリリンギングペナルティ項として組み込む。

**実装手順**:

1. `AllpassDesigner.h`: `AllpassDesignerConfig` に `double preRingPenalty = 0.0;` を追加
2. `AllpassDesigner.cpp` の `costFunction()` 内で、推定されたプリリンギング振幅に比例するペナルティを追加
3. `ConvolverProcessor.MixedPhase.cpp` の `designer_config` 設定箇所で `designer_config.preRingPenalty = tau;` を設定
4. `convertToMixedPhaseFallback()` の `(void)tau;` を削除し、`tau` を群遅延ターゲット生成に反映

**コスト関数変更イメージ**:

```
J_total = J_gd + lambda * J_preRing
```

ここで `J_gd` は既存の群遅延誤差、`lambda = tau / tau_default`、`J_preRing` は推定プリリンギング振幅の2乗和。

**注意**: これは CMA-ES の最適化次元を変更せず、評価関数にペナルティ項を追加するだけなので、収束性や計算時間への影響は軽微。

#### 選択肢B: UIパラメータと内部パラメータを削除（ミニマム）

- `ConvolverProcessor.h` から `MIXED_TAU_MIN/MAX/DEFAULT`・`mixedPreRingTau` 関連の全宣言を削除
- `ConvolverControlPanel.cpp` から `mixedTauSlider` 関連UIを削除
- `BuildSnapshot` から `mixedPreRingTau` を削除
- `IRCacheKey` から `tau` を削除

### 3.3 推奨

**選択肢A（有効化）を推奨**。このパラメータは「プリリンギング制御」という音響的に意味のある目的を持ってUIに公開されており、削除するとユーザーの期待を裏切る。有効化により、実際にプリリンギング量を調整可能な付加価値機能となる。

### 3.4 調査確定事項

以下の事項を追加調査で確定した：

| 確定項目 | 調査結果 |
| --- | --- |
| CMA-ES cost関数の構造 | `AllpassDesigner.cpp` L323-370: `weightedSquaredError = sum(weight[i] * (tau_sum - target_gd)^2)`, 戻り値 `sqrt(weightedSquaredError)`. 最適化次元 `D = 2 * numSections`（ρ, θペア） |
| ペナルティ項の追加方法 | cost関数内で `weightedSquaredError` 計算後にプリリンギング推定値を加算。`return sqrt(weightedSquaredError) + lambda * preRingEstimate;` |
| GreedyAdaGrad cost関数 | セクション逐次近似。`gridSearch2D()` + `adaptiveGradientDescent()` で各セクションを独立にフィッティング。グローバルなプリリンギングペナルティの追加には不向き |
| `AllpassDesignerConfig` に `preRingPenalty` がない理由 | 設計上の未実装。`tau` フィールド自体が `AllpassDesignerConfig` に存在しない |
| `tau` の全使用箇所 | `IRCacheKey::tau`（キャッシュキーのみ）、`convertToMixedPhaseFallback()` で `(void)tau`、`AllpassDesigner::Config` には未定義 |
| 選択肢Aの実装ファイル | `AllpassDesigner.h`（Config構造体）, `AllpassDesigner.cpp`（cost関数）, `ConvolverProcessor.MixedPhase.cpp`（config設定部） |
| 選択肢Bの削除対象ファイル | `ConvolverProcessor.h`（定数・メンバ）, `ConvolverControlPanel.cpp`（UIスライダー）, `convolver/ConvolverProcessor.StateAndUI.cpp`（状態管理）, `convolver/ConvolverProcessor.Runtime.cpp`（setter/getter）, `convolver/ConvolverProcessor.LoaderThreadInline.h`（BuildSnapshot） |
| 既存ドキュメント | `doc/work/bug4_observation_register.md` で P3-2 として `mixedPreRingTau` の無効性が2026-05-24から監視中 |
| プリリンギング推定方法 | オールパスカスケードのインパルス応答を短区間（例: 256 samples）計算し、ピーク位置より前のエネルギー積分値を推定値とする。コスト: O(N*256) で軽微 |

### 3.5 コスト関数変更の詳細設計（選択肢A）

現行の cost 関数:
```
J = sqrt(sum(weight[i] * (tau_sum[i] - target_gd[i])^2))
```

修正後:
```
J = sqrt(sum(weight[i] * (tau_sum[i] - target_gd[i])^2))
    + lambda * preRingPenalty(rho_list, theta_list, sampleRate)
```

ここで:
- `lambda = tau / MIXED_TAU_DEFAULT`（tau=32(default)で1.0、tau=256(max)で8.0、tau=4(min)で0.125）
- `preRingPenalty()`: 全セクションの (ρ,θ) から短いIRを生成し、ピーク前のエネルギー積分値を計算

**注意**: `convertToMixedPhaseFallback()` ではコスト関数の概念がなくセクション逐次近似のため、`tau` パラメータの効果的な組み込みは困難。Fallback時は `tau` を無視し続ける（現状維持）のが合理的。

---

## 4. F3: r8brain IRテイル切り捨て修正

### 4.1 現状

`IRDSP::resampleIR()`（`src/IRDSP.cpp`）が出力バッファ長を以下の式で計算している：

```cpp
const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;
const int maxOutLen = static_cast<int>(expectedLen);
```

r8brain の `CDSPResampler::getLatency()` は常に `0` を返す（内部レイテンシ自動除去設計）。その代償として、入力終端後のフラッシュ処理（`process(nullptr, 0, ...)`）で内部フィルタの残留サンプルを排出する必要がある。Harrisの近似式 `N ≈ Atten_dB / (22·Δf)` により、140dB/2%設定では概算 **318タップ**のフィルタ長となり、フラッシュ出力は数十〜数百サンプルに達しうる。

現在の `+2.0` マージンではフラッシュ出力がバッファに収まらず、IR末尾が切り捨てられる。

### 4.2 解決策

#### 修正A: `IRDSP::resampleIR()` のバッファサイズ計算（必須）

**ファイル**: `src/IRDSP.cpp`

**現行コード**:

```cpp
const double ratio = targetSR / inputSR;
const int inLength = inputIR.getNumSamples();
const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;
```

**修正後**:

```cpp
const double ratio = targetSR / inputSR;
const int inLength = inputIR.getNumSamples();

// r8brain の内部フィルタレイテンシを考慮したバッファサイズ
// r8brain CDSPResampler::getLatency() は常に0を返す設計のため、
// getMaxOutLen() を使用して最大出力長を取得する
r8b::CDSPResampler tempResampler(inputSR, targetSR, inLength,
                                  cfg.transBand, cfg.stopBandAtten, cfg.phase);
const int safeMaxOut = tempResampler.getMaxOutLen(inLength);
// 安全マージン: フィルタテイル用にさらに 25%
const int maxOutLen = static_cast<int>(static_cast<double>(safeMaxOut) * 1.25) + 64;
```

#### 修正B: フラッシュループの改善（任意）

現行のフラッシュループは `done < maxOutLen` で停止するが、この条件を外して可変長で処理する：

- フラッシュループから `done < maxOutLen` 条件を削除
- フラッシュ完了後に `resampled.setSize(numCh, done, true, true, true)` でトリム
- ただし、事前に十分なバッファを確保していれば安全

### 4.3 調査確定事項

以下の事項を追加調査で確定した：

| 確定項目 | 調査結果 |
| --- | --- |
| `ConvolverProcessorInternal::resampleIR()` の安全性 | **安全**。`getMaxOutLen(inLen)` でバッファサイズを取得し、`oneshot()` を使用（`CDSPResampler.h` L593-652）。`oneshot()` は入力終端後に内部でゼロパディング＋フラッシュを自動実行する |
| `IRDSP::resampleIR()` の問題性 | **不安全**。`+2.0` マージンでバッファ確保し、手動フラッシュループを `done < maxOutLen` で打ち切る。r8brain の内部フィルタ遅延（140dB/2%設定で推定300タップ）がフラッシュサンプルとして現れた場合、確実に切り捨てられる |
| `IRConverter.cpp` での使用コンテキスト | `convertFile()`（L170）で `IRDSP::resampleIR(ir, sourceRate, config.targetSampleRate, shouldCancel)` を呼び出す。戻り値のサンプル数が `<= 0` の場合はエラー扱い。テイルが切り捨てられてもエラーにならない（静かに品質劣化） |
| r8brain oneshot の挙動 | `CDSPResampler.h` L593-652: 入力を `MaxInLen` チャンクで `process()` に渡し、入力終了後はゼロパディングでフラッシュ。最後に `clear()` で内部状態リセット。出力は引数 `oplen` で上限される |
| oneshot と chunked process の違い | `oneshot` はフラッシュを自動処理するが、`process` の手動ループでは呼び出し側がフラッシュを管理する必要がある。`IRDSP::resampleIR()` は後者のパターンでバッファマージン不足 |
| 修正後の期待動作 | バッファサイズを `getMaxOutLen()` * 1.25 + 64 とすることで、フラッシュサンプルを十分に収容可能。最終的なIR長はフラッシュ完了後の `done` 値でトリムする |

### 4.4 影響範囲

| 項目         | 影響                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| 影響パス     | **`IRConverter.cpp`（`IRDSP::resampleIR()`）のみ**                      |
| メインローダーパス | 影響なし。`ConvolverProcessorInternal::resampleIR()` は `getMaxOutLen()` 使用で安全 |
| 性能         | 無視可能（`getMaxOutLen()` は O(1) の定数時間）                         |

### 4.4 検証

修正後、以下のテストを実施：

1. サンプルレート変換（例: 44.1kHz → 48kHz, 96kHz → 44.1kHz）を含むIRをロード
2. 出力IRのサンプル数が `ceil(inLen * ratio)` 以上であることを確認
3. 元のIRとリサンプリング後IRのエネルギー積分値の差が 0.1% 未満であることを確認（テイル消失の有無）

---

## 5. F4: `computeMasteringSizing` 孤立計算の解消

### 5.1 現状

`ConvolverProcessorInternal::computeMasteringSizing()`（`ConvolverProcessor.Internal.h` L116-137）は `firstPartition`/`maxFFTSize` を計算する。これらの値は：

1. `Lifecycle.cpp` L221 → `StereoConvolver::init()` → `storedMaxFFTSize`/`storedFirstPartition` に保存
2. `LoaderThread.cpp` → 同経路
3. `clone()` 時に `init()` に再投入されるのみ

しかし、`MKLNonUniformConvolver::SetImpulse()` のシグネチャは：

```cpp
bool SetImpulse(const double* impulse, int irLen, int blockSize,
                double scale = 1.0, bool enableDirectHead = false,
                const FilterSpec* filterSpec = nullptr);
```

**`firstPartition` も `maxFFTSize` も引数に存在しない。**

そのため、NUCのパーティションサイズは `SetImpulse()` 内部で `nextPow2(max(blockSize, 64))` として独自計算され、`computeMasteringSizing` の結果は一切反映されない。

### 5.2 選択肢

#### 選択肢A: `computeMasteringSizing` の結果を NUC に反映させる

1. `MKLNonUniformConvolver::SetImpulse()` に `int maxFFTSize` と `int firstPartitionSize` のオプション引数を追加
2. 引数が指定された場合、NUC内部の自動計算をオーバーライドする
3. `StereoConvolver::init()` から `maxFFTSize`/`firstPartition` を `SetImpulse()` に転送

#### 選択肢B: 関数と関連コードを削除（推奨）

1. `ConvolverProcessor.Internal.h` から `computeMasteringSizing()` + `ConvolverSizing` を削除
2. `ConvolverProcessor.h` の `StereoConvolver` から `storedMaxFFTSize`・`storedFirstPartition` を削除
3. `StereoConvolver::init()` から `maxFFTSize`・`firstPartition` 引数を削除
4. 呼び出し元（Lifecycle.cpp・LoaderThread.cpp・LoadPipeline.cpp）の該当引数を削除

#### 選択肢C: 現状維持＋コメントでドキュメント化

### 5.3 調査確定事項

以下の事項を追加調査で確定した：

| 確定項目 | 調査結果 |
| --- | --- |
| `computeMasteringSizing` の全呼び出し箇所 | `Lifecycle.cpp` L217, `LoaderThread.cpp` L219 の2箇所。ともに `StereoConvolver::init()` の引数として `sizing.maxFFTSize` / `sizing.firstPartition` を渡す |
| データフロー完全追跡 | `computeMasteringSizing()` → `init(..., maxFFTSize, ..., firstPartition, ...)` → `storedMaxFFTSize = maxFFTSize; storedFirstPartition = firstPartition;` → **ここで値が途絶える**。`SetImpulse()` のシグネチャにこれらの引数が存在しない |
| `storedMaxFFTSize` / `storedFirstPartition` の唯一の参照 | `StereoConvolver::clone()` で `init()` に再投入（L765-770）。しかし `init()` 内部で再び `SetImpulse()` に届かないため、`clone()` 経由でもNUC構成に影響なし |
| NUCのパーティション決定ロジック | `MKLNonUniformConvolver.cpp` L604: `l0Part = nextPow2(max(blockSize, 64))` のみがL0サイズを決定。`computeMasteringSizing` の `firstPartition` は無関係 |
| `finalizeNUCEngineOnMessageThread` の引数 | `LoadPipeline.cpp` L555: シグネチャに `maxFFTSize` と `firstPartition` を含むが、これらは `StereoConvolver::init()` に転送されるのみで、NUCに届かない |
| 選択肢Bの削除対象ファイル | `ConvolverProcessor.Internal.h`（関数+構造体）, `ConvolverProcessor.h`（メンバ変数+init引数）, `Lifecycle.cpp`（呼び出し）, `LoaderThread.cpp`（呼び出し+引数伝播）, `LoadPipeline.cpp`（finalizeNUCEngineOnMessageThread引数）の5ファイル |
| 削除によるコンパイルエラーの可能性 | `finalizeNUCEngineOnMessageThread` のシグネチャ変更は呼び出し元（`LoaderThread.cpp` L334）に影響。ラムダキャプチャの引数リストも要修正 |

### 5.4 推奨

**選択肢B（削除）を推奨**。理由：

- 現在の `computeMasteringSizing()` は単に `nextPow2(internalBlockSize * 4)` を `[4096, 16384]` にクランプしているだけで、NUC内部の `nextPow2(max(blockSize, 64))` とほぼ同値
- `storedFirstPartition` は `clone()` 経由で受け継がれるが、`SetImpulse()` に届かないため無意味
- 選択肢AはNUCのシグネチャを変更し、上位互換性の問題が生じる可能性がある
- 削除により `StereoConvolver::init()` の引数が減り、コードの可読性が向上する

### 5.4 影響範囲

| 項目     | 影響                                                      |
| -------- | --------------------------------------------------------- |
| NUC動作  | **なし（削除しても挙動不変）**                              |
| コンパイル | 呼び出し元の引数削除が必要                                |
| `clone()` | `init()` シグネチャ変更に追随                             |

### 5.5 削除手順

1. `ConvolverProcessor.Internal.h` から `computeMasteringSizing()`・`ConvolverSizing` を削除
2. `ConvolverProcessor.h` の `StereoConvolver` から `storedMaxFFTSize`・`storedFirstPartition` を削除
3. `ConvolverProcessor.h` の `StereoConvolver::init()` から `maxFFTSize`・`firstPartition` 引数を削除
4. `convolver/ConvolverProcessor.Lifecycle.cpp` L217 の呼び出しを削除
5. `convolver/ConvolverProcessor.LoaderThread.cpp` L219 の呼び出しおよび関連引数伝播を削除
6. `convolver/ConvolverProcessor.LoadPipeline.cpp` の `finalizeNUCEngineOnMessageThread()` から `maxFFTSize`・`firstPartition` 引数を削除

---

## 6. F5: `applyAllpassToIR` 重複実装の解消

### 6.1 現状

`AllpassDesigner.h` L115 で宣言・`AllpassDesigner.cpp` L595-742 で実装されている静的関数 `applyAllpassToIR()` は、MKL DFTI を用いてオールパスカスケードを IR に正しく適用するが、**コードベース全体で呼び出し箇所が一つもない**。実際の `convertToMixedPhaseAllpass()` は、同等の処理（周波数応答計算→複素乗算→逆FFT）を **インラインで再実装** している（L555-567）。

### 6.2 調査確定事項

以下の事項を追加調査で確定した：

| 確定項目 | 調査結果 |
| --- | --- |
| 全ファイル検索結果 | `src/**/*.{cpp,h}` で `applyAllpassToIR` の呼び出し箇所は **0件**。宣言（`AllpassDesigner.h` L115）+ 定義（`AllpassDesigner.cpp` L595）のみ |
| 既存ドキュメントでの認識 | `doc/work/bug4_observation_register.md` で P3-2（2026-05-24〜Monitoring）として記録済み："呼び出し0件（宣言/定義のみ）" |
| 別ドキュメントでの言及 | `doc/plan4.md` L275: "デッドコードのため削除、またはASSERT_NON_RT_THREAD()" |
| Inline実装との関係 | `ConvolverProcessor.MixedPhase.cpp` L555-567 で同等のオールパス適用がインライン実装されている。FFTバックエンドの差異（MKL DFTI vs IPP FFT）を確認する必要なく、削除可能 |
| テストからの参照 | テストコードからも呼び出しなし |

### 6.3 選択肢

#### 選択肢A: `applyAllpassToIR` を削除（推奨）

- インライン実装が実運用で使用されており、`applyAllpassToIR` は歴史的経緯で残存
- `AllpassDesigner.h` から宣言を削除
- `AllpassDesigner.cpp` から実装を削除

#### 選択肢B: `convertToMixedPhaseAllpass()` のインライン実装を `applyAllpassToIR` 呼び出しに置き換え

- 重複を排除し、コードの一元管理を実現
- ただし、`applyAllpassToIR` は MKL DFTI を使用するのに対し、`convertToMixedPhaseAllpass()` は IPP FFT を使用している可能性があるため、置き換え前に FFT バックエンドの互換性を確認する必要がある

### 6.3 推奨

**選択肢A（削除）を推奨**。理由：

- インライン実装が実戦投入されており、`applyAllpassToIR` は未使用コードとしてメンテナンスコストだけを生む
- 選択肢Bは FFT バックエンドの互換性確認が必要で、リスクの割にメリットが少ない
- 削除により `AllpassDesigner` の公開APIが整理される

### 6.4 削除手順

1. `AllpassDesigner.h` から `applyAllpassToIR` の宣言を削除
2. `AllpassDesigner.cpp` から `applyAllpassToIR` の実装（コメント含む L588-742）を削除
3. ビルド確認

---

## 7. F6: レイヤースケジューリング最適化（調査・将来課題）

### 7.1 現状

`MKLNonUniformConvolver::SetImpulse()` 内で L1/L2 のパーティションサイズは：

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l1Part = l0Part * tailL1L2Mult;   // デフォルト8
const int l2Part = l1Part * tailL1L2Mult;   // デフォルト8
```

これは Gardner (1995) の「固定比率ヒューリスティック」に該当する。

### 7.2 Garcia/Wefers 最適化との比較

Garcia (2002) は動的計画法（Viterbiアルゴリズム）を用いて、指定された入出力遅延とフィルタ長に対する計算コスト最小化パーティションを導出する。Garcia論文の重要な発見：

> "often the optimal partition does not include transitions to blocks twice as long (but four times as long or greater)"

つまり、最適解は2倍ではなく4倍以上のジャンプを含むことが多い。ConvoPeqの「固定倍数8」はこの傾向と方向性として整合するが、ハードウェア固有のベンチマークに基づくコストモデルを用いた厳密最適化は行われていない。

### 7.3 対応方針

**本フェーズでは調査のみとし、実装は行わない。**

| ステップ | 内容                                                             | 工数   |
| -------- | ---------------------------------------------------------------- | ------ |
| 1        | Garcia論文の再読とコストモデルの導出                             | 2-3日  |
| 2        | ターゲットハードウェア（AVX2 + Intel MKL/IPP）のベンチマーク作成 | 1-2日  |
| 3        | DPによる最適パーティションと現在の倍率8との比較                  | 1日    |
| 4        | 実装判断                                                         | —      |

### 7.4 `tailL1L2Multiplier` のUI設定の現状

`FilterSpec::tailL1L2Multiplier` は UI 範囲2〜16（デフォルト8）でユーザーが設定可能。このパラメータは適切に `SetImpulse()` に伝達されており、現状の実装でも動作に問題はない。最適化の余地はソフトウェアの自動選択（動的最適化）に関するものであり、ユーザーの手動設定が無効なわけではない。

---

## 8. 検証計画

### 8.1 単体テスト

| テスト項目                               | 対象                                    | 方法                                       |
| ---------------------------------------- | --------------------------------------- | ------------------------------------------ |
| F1 修正後 Mixed Phase IR の方向確認      | `ConvolverProcessor.MixedPhase.cpp`     | IRの位相特性を周波数領域でプロット          |
| F3 修正後 IR テイル保全確認              | `IRDSP.cpp`                             | 異なるサンプルレート変換でIR末尾のエネルギー損失を測定 |
| F4 削除後 ビルド/動作確認                | 全修正ファイル                          | Release/Debug両方でビルド                  |
| F5 削除後 ビルド確認                     | `AllpassDesigner.cpp/h`                 | リンクエラーなしを確認                     |

### 8.2 統合テスト

| テスト項目                                  | 内容                                       |
| ------------------------------------------- | ------------------------------------------ |
| IRロード（PhaseMode=AsIs）                   | 全機能正常動作                             |
| IRロード（PhaseMode=Mixed）                  | F1+F2 修正の影響確認                        |
| IRロード（サンプルレート変換あり）            | F3 修正の影響確認                           |
| ホットスワップ（IR差し替え）                 | RCU経路の正常性確認                         |
| プリセット保存/読込                          | BuildSnapshot互換性確認                     |

### 8.3 音響測定（推奨）

| 測定項目                                       | 方法                                                   |
| ---------------------------------------------- | ------------------------------------------------------ |
| Mixed Phase IRの群遅延プロット                  | 周波数領域で群遅延曲線を確認                            |
| r8brainリサンプリングIRのテイル比較             | 修正前後のIR末尾を波形・スペクトログラムで比較          |
| 実機での聴感評価                               | 定位感・プリリンギングの有無                            |

---

## 9. 依存関係と実施順序

```
Phase 1: 即時修正（高優先度）
  F3: IRDSP::resampleIR() バッファマージン修正（最優先・他に依存なし）
  F1: Mixed Phase クロスオーバー方向修正（F2とファイルが同じ）

Phase 2: パラメータ有効化
  F2: mixedPreRingTau 有効化（F1と同じファイルのため同時修正推奨）

Phase 3: コード整理（低優先度）
  F5: applyAllpassToIR 削除（独立して実施可能）
  F4: computeMasteringSizing 削除（複数ファイルにまたがるため最後に実施）

Phase 4: 将来課題
  F6: レイヤースケジューリング最適化（調査のみ。本改修では実施しない）
```

### 推奨実施順序

```
F3 → F1 → F2 → F5 → F4
```

### 各フェーズの成果物

| フェーズ | 成果物                                                         |
| -------- | -------------------------------------------------------------- |
| Phase 1  | 修正済み `IRDSP.cpp`, `ConvolverProcessor.MixedPhase.cpp` + 動作確認ログ |
| Phase 2  | 修正済み `AllpassDesigner.h/cpp`, `ConvolverProcessor.MixedPhase.cpp` + 単体テスト結果 |
| Phase 3  | 修正済み全ファイル + ビルド成功確認                            |
| Phase 4  | Garcia/Wefers最適化の調査レポート                              |

---

## 付録A: コード行番号リファレンス（検証時点）

| ファイル                                                   | 行     | 内容                                               |
| ---------------------------------------------------------- | ------ | -------------------------------------------------- |
| `src/IRDSP.cpp`                                            | 18     | `+2.0` マージン（要修正）                           |
| `src/IRDSP.cpp`                                            | 23     | `maxOutLen` バッファ確保                            |
| `src/IRDSP.cpp`                                            | 49-52  | フラッシュループ（`done < maxOutLen`）              |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp`          | 303-311| Allpass版 クロスオーバー重み（要修正）              |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp`          | 478-489| `designer_config` 設定（tau未使用）                 |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp`          | 734    | `(void)tau;`（Fallback版）                          |
| `src/convolver/ConvolverProcessor.MixedPhase.cpp`          | 810-818| Fallback版 クロスオーバー重み（要修正）             |
| `src/convolver/ConvolverProcessor.Internal.h`              | 116-137| `computeMasteringSizing()`（削除候補）              |
| `src/ConvolverProcessor.h`                                 | 700-735| `StereoConvolver::init()`（引数整理候補）           |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp`           | 217-221| `computeMasteringSizing` 呼び出し                   |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp`        | 219    | `computeMasteringSizing` 呼び出し                   |
| `src/AllpassDesigner.h`                                    | 115    | `applyAllpassToIR` 宣言（削除候補）                 |
| `src/AllpassDesigner.cpp`                                  | 588-742| `applyAllpassToIR` 実装（削除候補）                 |

## 付録B: 関連する既存資料

- `doc/work55/bug.md` - レビュー要旨
- `doc/work55/ConvoPeq_Convolver_Validation_Report.md` - 詳細検証レポート
- `doc/work55/review_verification_report.md` - 相互評価レポート
- プロジェクトメモリ: `mixedPreRingTau` の無効性（既知）、r8brainテイル切り捨てリスク（既知未解決）

---

*本計画書は 2026-06-23 時点の `lonewolf-jp/ConvoPeq` main ブランチのソースコードに基づく。*
