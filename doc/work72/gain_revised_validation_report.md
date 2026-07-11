# `doc/work72/gain_revised.md` 妥当性検証レポート（第2次）

検証日: 2026-07-11
検証対象: `doc/work72/gain_revised.md`（418行、改訂版）
検証方法: コードベース実装確認 + 音響工学文献調査 + 数式検証

---

## 0. 検証サマリー

| 重大度 | 項目 | 数 |
|--------|------|----|
| 🔴 重大（設計欠陥） | 実装すると意図通り動作しない、または誤動作する | 4 |
| 🟡 中程度（要修正） | 実装可能だが仕様記述に誤りがある | 5 |
| 🟢 軽微/確認済み | 妥当であることを確認、または軽微な改善推奨 | 8 |

---

## 1. 🔴 重大な問題（設計欠陥）

### 1.1 Conv→PEQモードで `convolverInputTrimGain` が適用されない

**問題の核心。** 改訂版の§3.5.1で Conv→PEQ モードの計算式は:
```
input = -max(0, irResidual - 1.5)
trim  = -max(0, eqMax - 2.0) - input
makeup = -input - trim
```

しかし、実際のDSP処理チェーン（`AudioEngine.Processing.DSPCoreDouble.cpp:429-457`）を確認すると:

```cpp
if (state.order == ProcessingOrder::ConvolverThenEQ)
{
    if (!state.convBypassed)
        convolverRt().process(processBlock);      // コンボルバー処理
    if (!state.eqBypassed)
        eqRt().process(processBlock, ...);         // EQ処理
    // ★ convolverInputTrimGain は適用されない！
}
else  // EQThenConvolver
{
    if (!state.eqBypassed)
        eqRt().process(processBlock, ...);         // EQ処理
    if (!state.convBypassed)
    {
        if (state.convolverInputTrimGain != 1.0)
            scaleBlockFallback(..., state.convolverInputTrimGain);  // ★ ここでのみ適用
        convolverRt().process(processBlock);       // コンボルバー処理
    }
}
```

`convolverInputTrimGain` は `EQThenConvolver` パス（lines 483-490）でのみ適用され、`ConvolverThenEQ` パス（lines 429-457）では**一切適用されない**。

**影響**: Conv→PEQ モードで `trim` に計算した値はAudio Threadで無視される。式の意図は「コンボルバー後にEQブーストを保護するための段間トリム」だが、`convolverInputTrimGain` は「コンボルバー入力前」のゲインであり、Conv→PEQではコンボルバーが最初なので適用箇所がない。

**具体例での影響**（irResidual=6dB, eqMax=9dB）:
- 計算値: input=-4.5dB, trim=-2.5dB, makeup=+7.0dB
- 実際の信号経過: 入力-4.5dB → コンボルバー+6dB → EQ+9dB → メイクアップ+7.0dB = **+17.5dB**
- 意図した経過: 入力-4.5dB → コンボルバー+6dB → トリム-2.5dB → EQ+9dB → メイクアップ+7.0dB = **+15.0dB**
- 差分: +2.5dB の過剰ゲイン → リミッターが意図より多く動作する

**修正案**: 以下のいずれかを採用する必要がある:
- **(A)** Conv→PEQ モードの式を変更し、EQ保護分をinputに折りたたむ:
  `input = -max(0, irResidual - 1.5) - max(0, eqMax - 2.0)`、`trim = 0`、`makeup = -input`
- **(B)** DSPコードにConv→PEQパス用の段間ゲインステージを新設する（コード変更量大）
- **(C)** `convolverInputTrimGain` をEQ前にも適用するようDSPコードを修正する

推奨は **(A)**。式がシンプルで、DSPコードの変更が不要。

### 1.2 `z = exp(-jω)` の符号が既存コードと逆

改訂版§3.1.1で `z = exp(-jω)` を使用すると記載。しかし既存の `getMagnitudeSquared`（`EQProcessor.Coefficients.cpp:325-330`）は:
```cpp
const double w = 2.0 * pi * freq / sampleRate;
std::complex<double> z(std::cos(w), std::sin(w));  // z = exp(+jω)
```
**`z = exp(+jω)`（正の指数）を使用している。** これは標準的なDSPの慣行（Julius O. Smith III, *Introduction to Digital Filters*, CCRMA Stanford）でもある。

`z = exp(-jω)` を使うと:
- マグニチュード応答 `|H(z)|` は同じ（OK）
- 位相応答は符号反転する（NG — M/Sデコードで複素応答をカスケード積算する場合、位相が重要）

**修正**: `z = exp(+jω)` を使用し、既存コードとの整合を保つ。マグニチュードのみが必要な場合はどちらでも同じ結果だが、新しい関数が既存の `getMagnitudeSquared` と一貫しているべき。

### 1.3 Q閾値の数学的誤記: "1.414 (1/sqrt(2))"

改訂版§3.3.1項5:
> Peakingで`gain > 0`かつ`Q > 1.414`の場合、`gain * 0.15 * (Q / 1.414)`を加算

ここで 1.414 = √2 だが、括弧内の "(1/sqrt(2))" は 1/√2 ≈ 0.707 のことである。**1.414 ≠ 1/√2**。

**文献調査結果**:
- Q = 1/√2 ≈ 0.707 は **Butterworth Q**（2次フィルタで最大平坦特性を与えるQ値）。Wikipedia "Q factor" および "Butterworth filter" で確認。
- Q = √2 ≈ 1.414 は Butterworth Qの2倍であり、理論的な特別な閾値ではない。

**修正案**:
- 閾値を Q > 0.707 (Butterworth Q) にするのが理論的に正当
- または Q > 1.414 を実用的ヒューリスティックとして維持するが、"(1/sqrt(2))" の注記を削除し "√2" と正しく記載

### 1.4 Tukey窓の-60dBサイドローブ抑制は非現実的

改訂版§3.1.3のテスト基準:
> Tukey窓適用後のスペクトルリーケージが-60dB以下に抑制されることを確認

**文献調査結果**（Wikipedia "Window function"、MATLAB Signal Processing Toolbox）:
- Tukey窓 α=0.1（10%テーパー）の第1サイドローブ: **約-15〜-18 dB**
- Hann窓（α=1.0）でも第1サイドローブ: 約-31 dB
- -60 dB を達成するには Blackman窓（-58 dB）または Blackman-Harris窓（-92 dB）が必要

ただし、65536点FFTの場合、周波数分解能が非常に高く（48kHzで0.73 Hz/bin）、メインローブから離れた周波数でのリークは-60 dB以下になる可能性はある。しかし「第1サイドローブが-60 dB以下」という基準では非現実的。

**修正案**: テスト基準を以下のいずれかに変更:
- 「第1サイドローブが-15 dB以下に抑制されること」（現実的）
- または窓関数を Blackman 窓に変更して-58 dBを狙う（ただしIRのエネルギーが減衰する副作用）
- または「メインローブピークから10倍周波数bin離れた位置で-60 dB以下」（65536点FFTなら達成可能）

---

## 2. 🟡 中程度の問題（要修正）

### 2.1 IRロード完了位置の記載誤り: `AudioEngine.Cache.cpp`

改訂版§4（修正ファイル一覧）:
> `src/audioengine/AudioEngine.Cache.cpp` | IRロード完了時の`recomputeAutoGainStaging()`呼び出し追加

**実態**: `AudioEngine.Cache.cpp`（156行）は `EQCacheManager` のみを含み、IRロード完了とは無関係。

実際のIRロード完了フロー:
1. `ConvolverProcessor::applyComputedIR()` — `src/convolver/ConvolverProcessor.LoadPipeline.cpp:308`
2. → `postCoalescedChangeNotification()` — 同ファイル:500
3. → `AudioEngine::convolverParamsChanged()` — `src/audioengine/AudioEngine.UIEvents.cpp:36-195`

**修正**: `recomputeAutoGainStaging()` の呼び出し追加先を `AudioEngine.UIEvents.cpp`（`convolverParamsChanged` 内）に修正。

### 2.2 `setInputHeadroomDb` のクランプ範囲がルーティング依存

改訂版§3.5.1:
> クランプ：`trim`は[-12, 0]dB、`makeup`は[0, 12]dB（既存setter内で実施）

`setOutputMakeupDb` と `setConvolverInputTrimDb` のクランプ範囲は固定（それぞれ [0,12], [-12,0]）で正しい。しかし `setInputHeadroomDb`（`AudioEngine.Parameters.cpp:224-242`）は:

```cpp
const bool convIsFirst = !convBypassed && (order == ConvolverThenEQ || eqBypassed);
const float maxDb = convIsFirst ? -6.0f : 0.0f;
float clampedDb = juce::jlimit(-12.0f, maxDb, db);
```

上限が**動的**: Conv-first（デフォルトモード）では -6 dB、EQ-first または Conv-bypass では 0 dB。

**影響**: Conv→PEQ モード（デフォルト）で `recomputeAutoGainStaging` が input = 0 dB を計算した場合、setterが-6 dBにクランプする。Auto ON の計算結果が意図せず制限される可能性がある。

**修正案**: 計画書にルーティング依存のクランプ動作を明記し、`recomputeAutoGainStaging` 側でクランプ後の値を再読み込みして `makeup` を調整するロジックを追加することを推奨。

### 2.3 `setProcessingOrder` の呼び出し順序の不整合

改訂版§3.5.2では、3つのsetterの末尾で `recomputeAutoGainStaging()` を呼ぶと記載。しかし実際の呼び出し順序:

| Setter | 呼び出し順序 |
|--------|-------------|
| `setEqBypassRequested` | publish → `applyDefaultsForCurrentMode()` → `submitRebuildIntent()` → `sendChangeMessage()` |
| `setConvolverBypassRequested` | publish → `applyDefaultsForCurrentMode()` → `submitRebuildIntent()` → `sendChangeMessage()` |
| `setProcessingOrder` | publish → **`submitRebuildIntent()`** → **`applyDefaultsForCurrentMode()`** |

`setProcessingOrder` だけ順序が逆（`submitRebuildIntent` → `applyDefaultsForCurrentMode`）。しかも `applyDefaultsForCurrentMode()` 内部でも `submitRebuildIntent` が呼ばれる（line 339）ため、**`setProcessingOrder` は2回の rebuild intent を発行する**。

`recomputeAutoGainStaging()` を追加する位置によっては、3回の rebuild intent が発生する。

**修正案**: `setProcessingOrder` 内の `submitRebuildIntent`（line 273）を削除するか、`recomputeAutoGainStaging()` の呼び出しを `applyDefaultsForCurrentMode()` の直前（bypass setterと同じ位置）に統一する。

### 2.4 Q Surge Marginの0.15係数に理論的根拠がない

**文献調査結果**:
- 高Qピーキングフィルタで過渡ピークが定常マグニチュードを超える現象は**実在する**（Wikipedia "Q factor"の過渡応答オーバーシュート参照）
- しかし、オーバーシュート量とQの関係は**指数関数** `e^{-πζ/√(1-ζ²)}`（ζ = 1/(2Q)）であり、線形ではない
- 0.15という係数の**出版物での根拠は見つからない**。カスタムヒューリスティックと思われる
- 例: Q=4, gain=12dB → margin = 12 × 0.15 × (4/1.414) ≈ 5.1 dB。これは過渡オーバーシュート理論値（Q=4で約80% = 約2.2 dB）より大きく、保守的

**評価**: 係数自体は保守的で安全側だが、理論的根拠がないことを文書に明記すべき。また、オーバーシュートは入力信号の性質に依存するため、最悪ケース（ステップ入力等）を想定した検証がテストフェーズで必要。

### 2.5 予測型静的マージン方式は業界に先行例がない

**文献調査結果**:
- FabFilter、iZotope、Waves等のプロフェッショナルEQプラグインで、EQカーブから最大ゲインを事前計算して入力ヘッドルームを自動調整する製品は**見つからなかった**
- iZotope Ozone等の「Auto Gain」機能はRMSベース（反応型）であり、予測型（事前計算型）ではない
- 本方式は**ノベルティ**であり、概念的には健全だが、実績による検証が必要

**評価**: 設計方針として妥当だが、Phase 7のテストで特に慎重な検証が必要。特に「予測値が実際のピーク挙動とどれくらい一致するか」の定量的評価を追加すべき。

---

## 3. 🟢 確認済み・妥当な項目

### 3.1 Biquad伝達関数の評価方法 ✅
`H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)` は標準的な方法（W3C Audio EQ Cookbook、Smith, CCRMA Stanford）。既存コード（`EQProcessor.Coefficients.cpp:332-342`）も同じアプローチを使用。

### 3.2 周波数スキャン300点 ✅
300点（対数スケール、20Hz〜Nyquist）は約30点/オクターブを与え、最大ゲイン検出用途として十分。MATLAB `freqz` のデフォルト512点より少ないが、対数スケールでの300点は最大値探索には十分。AVX2化しなくても1ms未満で完了する見込み。

### 3.3 L2（エネルギー）正規化の妥当性 ✅
文献調査で確認: リバーブIRのような指数減衰信号では、L2正規化がL1正規化より優れている。L1は長いテールの小さな値の累積に支配され、初期反射音の振幅を過度に減衰させる。L2は二乗により高振幅部の重みが大きく、初期反射音のS/N比が良い。改訂版のL2採用は正しい。

### 3.4 6 dB IR安全マージン ✅
正式な標準（AES, ITU）はないが、リバーブIRのピーク対RMS比が12〜30 dBに達することを考慮すると、6 dB（2倍の振幅マージン）は妥当な工学慣行。`safetyMargin = 0.5011872336272722` = `10^(-6/20)` = 正確に-6 dB。

### 3.5 -1.0 dBFS出力ヘッドルーム ✅
`kOutputHeadroom = 0.8912509381337456` ≈ -1.0 dBFS。全主要ストリーミングプラットフォーム（Spotify, Apple Music, YouTube）とEBU R128放送基準が-1 dBTPを要求。完全に整合。

### 3.6 4xオーバーサンプリング（768kHz） ✅
ITU-R BS.1770-4がTrue Peak検出に4xオーバーサンプリングを規定。192kHz → 768kHz（4x）は正確に規格に一致。コードベースの `TruePeakDetector` も4x OS（2段×2x）で実装済み。

### 3.7 `m_isRestoringState` のRAII保証 ✅
`requestLoadState()`（`AudioEngine.StateIO.cpp:32-163`）はRAIIガード `RestoreStateGuard` を使用し、例外発生時でも確実に `m_isRestoringState = false` に戻る。改訂版の「リストア完了後に1回再計算」の方針は有効。

ただし `beginBulkParameterRestore()` / `endBulkParameterRestore()` はRAIIではなく、呼び出し元が `endBulkParameterRestore()` を忘れるとフラグが `true` のまま残留する。この経路を使用する場合は要注意。

### 3.8 `EQProcessor::computeMaxGainDb()` のconst実装可能性 ✅
`getEQState()` は既に `const`（`EQProcessor.h:348-349`）。`EQState` へのアクセスは `std::atomic` ロードのみで `mutable` 不要。`calcBiquadCoeffs` と `getMagnitudeSquared` は `static` であり、`const` メソッドから呼び出し可能。`EQEditProcessor` は `EQProcessor` をpublic継承しているため、追加した `computeMaxGainDb()` に直接アクセス可能。

---

## 4. 4モード計算式の数値検証

改訂版§3.5.1の4モード計算式を数値例で検証する。

### 4.1 PEQ only（Conv bypass, EQ active）
式: `input = -max(0, eqMax - 3.0)`、`trim = 0`、`makeup = -input`

| eqMax | input | trim | makeup | net | EQ後ピーク | 備考 |
|-------|-------|------|--------|-----|-----------|------|
| 0 dB | 0 | 0 | 0 | 0 | 0 | ヘッドルーム不要 ✅ |
| 3 dB | 0 | 0 | 0 | 0 | +3 | マージン内 ✅ |
| 6 dB | -3 | 0 | +3 | 0 | +3 | 保護あり ✅ |
| 12 dB | -9 | 0 | +9 | 0 | +3 | 保護あり ✅ |

検証: `input + makeup = 0`（ネット0dB）。EQ前のヘッドルーム = `eqMax - 3 dB`。✅ 妥当。

### 4.2 Conv only（EQ bypass, Conv active）
式: `input = -max(0, irResidual - 1.5)`、`trim = 0`、`makeup = -input`

| irResidual | input | trim | makeup | net | Conv後ピーク | 備考 |
|------------|-------|------|--------|-----|-------------|------|
| 0 dB | 0 | 0 | 0 | 0 | 0 | ✅ |
| 3 dB | -1.5 | 0 | +1.5 | 0 | +1.5 | ✅ |
| 6 dB | -4.5 | 0 | +4.5 | 0 | +1.5 | ✅ |

検証: ✅ 妥当。

### 4.3 Conv→PEQ（Conv first, both active）⚠️
式: `input = -max(0, irResidual - 1.5)`、`trim = -max(0, eqMax - 2.0) - input`、`makeup = -input - trim`

**trimは適用されない（§1.1参照）。実際の信号経路での検証:**

| irResid | eqMax | input | trim(計算) | trim(実際) | makeup | 実ネット | 意図ネット | 差 |
|---------|-------|-------|-----------|-----------|--------|---------|-----------|-----|
| 6 | 3 | -4.5 | -1-(-4.5)=+3.5→**0** | 0 | +4.5 | 0 | 0 | 0 ✅ |
| 6 | 9 | -4.5 | -7-(-4.5)=-2.5 | **0** | +7.0 | +2.5 | 0 | **+2.5** ❌ |
| 3 | 9 | -1.5 | -7-(-1.5)=-5.5 | **0** | +7.0 | +5.5 | 0 | **+5.5** ❌ |
| 0 | 9 | 0 | -7-0=-7 | **0** | +7.0 | +7.0 | 0 | **+7.0** ❌ |

**結論**: Conv→PEQモードで `eqMax > 2.0 + (irResidual - 1.5)` の場合、trimがクランプされて意図したゲインバランスが崩れる。特にEQブーストが大きい場合にリミッターが過剰動作する。

### 4.4 PEQ→Conv（EQ first, both active）✅
式: `input = -max(0, eqMax - 3.0)`、`trim = -max(0, irResidual - 2.0) - input`、`makeup = -input - trim`

**trimは適用される（EQThenConvolverパス）:**

| eqMax | irResid | input | trim | makeup | net | EQ後 | Conv後 | 最終 | 備考 |
|-------|---------|-------|------|--------|-----|------|--------|------|------|
| 9 | 6 | -6 | -4-(-6)=+2→**0** | +6 | 0 | +3 | +9 | +15 | trimクランプ ⚠️ |
| 3 | 6 | 0 | -4-0=-4 | +4 | 0 | +3 | +9 | +13 | ✅ |
| 9 | 3 | -6 | -1-(-6)=+5→**0** | +6 | 0 | +3 | +6 | +12 | trimクランプ ⚠️ |

PEQ→Convでもtrimが正にクランプされるケースがある。ただし、この場合は「入力で十分に保護しているので段間トリム不要」として扱われるため、ネット0dBは保たれる。ただしConv後ピークが意図より高くなる場合がある（eqMax=9, irResid=6 のケースで最終+15dB、これは-1dBFSリミッターで大幅に圧縮される）。

**補足**: PEQ→Convモードのtrimクランプは `trim = -max(0, X) - input` の設計上、inputが大きい（負が大きい）ほどtrimが正に傾く。これは「入力ですでに保護済み」を意味するので論理的には正しいが、最終ピークが高くなる問題は入力信号自体のピークレベルに依存する。

---

## 5. モード別計算式の修正提案

### 5.1 Conv→PEQモードの修正（§1.1の対策）

現行式（trim不適用）:
```
input = -max(0, irResidual - 1.5)
trim  = -max(0, eqMax - 2.0) - input    ← 適用されない
makeup = -input - trim
```

修正案(A)（trimを廃止し、inputに統合）:
```
input = -max(0, irResidual - 1.5) - max(0, eqMax - 2.0)
trim  = 0
makeup = -input
```

検証:
| irResid | eqMax | input | trim | makeup | net | 備考 |
|---------|-------|-------|------|--------|-----|------|
| 6 | 9 | -4.5-7=-11.5 | 0 | +11.5 | 0 | makeup上限12以内 ✅ |
| 6 | 3 | -4.5-0=-4.5 | 0 | +4.5 | 0 | ✅ |
| 0 | 9 | 0-7=-7 | 0 | +7 | 0 | ✅ |
| 6 | 15 | -4.5-13=-17.5→**-12** | 0 | +12 | 0 | inputクランプ ⚠️ |

inputが-12dBクランプされる極端ケースではmakeup+12でも補填不足（-5.5dBのネット減衰）。ただし安全側に倒れるため許容可能。PEQ only モードも同じ構造なので一貫性がある。

### 5.2 PEQ→Convモードの微調整（任意）

現行式でも概ね妥当だが、trimクランプ時の挙動を明記:
```
input = -max(0, eqMax - 3.0)
trim  = clamp(-12, 0, -max(0, irResidual - 2.0) - input)
makeup = clamp(0, 12, -input - trim)
```

trimが0にクランプされた場合、makeup = -input となり、Conv保護がスキップされる。これは「入力保護で十分」という判断であり、IR risk < 2dB の場合は許容可能。

---

## 6. その他の確認事項

### 6.1 `convolverInputTrimGain` の適用条件
`AudioEngine.Processing.DSPCoreDouble.cpp:483`: `if (state.convolverInputTrimGain != 1.0)` のガードがある。trim = 0 dB（gain = 1.0）の場合はスケール処理をスキップするため、パフォーマンスへの影響なし。

### 6.2 setterの二重atomic更新
3つのsetter（`setInputHeadroomDb`等）はいずれも dB値、linear gain値、mirror値の3つのatomicを更新する（`AudioEngine.Parameters.cpp:237-239, 256-258, 284-286`）。`recomputeAutoGainStaging()` がsetterを呼ぶ場合、3つのatomicすべてが正しく更新される。✅

### 6.3 `applyDefaultsForCurrentMode()` の直接atomic更新
`applyDefaultsForCurrentMode()`（`AudioEngine.Parameters.cpp:329-338`）はsetterを経由せず、直接 `publishAtomic` を呼ぶ。`recomputeAutoGainStaging()` がsetterを呼ぶ場合、`applyDefaultsForCurrentMode()` の値は上書きされる。これは改訂版の意図通り。✅

### 6.4 `EQCoeffsBiquad` の a0 正規化
`EQCoeffsBiquad`（`EQProcessor.h:101-106`）は `a0` を1.0以外の値も取りうる。既存の `getMagnitudeSquared` は `z^2` 倍して分子分母を z^2 の多項式として評価するため、a0 ≠ 1 でも正しく動作する。新しい複素応答関数でも同じアプローチを使用する必要がある。✅

---

## 7. 総合評価

### 7.1 改訂版の改善点（初版からの）
初版の7点の不整合はすべて修正されている:
1. ✅ 32bit float に修正
2. ✅ 予測型静的マージン方式の明確化
3. ✅ L2正規化への統一
4. ✅ Q Surge Margin への改称
5. ✅ 周波数スキャン点数の統一（300点）
6. ✅ ファイルパスの修正（`src/DspNumericPolicy.h`）
7. ✅ スケジュールの現実化（23人日）

### 7.2 新たに発見された問題
改訂版の深入り検証で以下の新問題が発見された:

| # | 問題 | 重大度 | 修正必要性 |
|---|------|--------|-----------|
| 1 | Conv→PEQでtrim未適用 | 🔴 重大 | 実装前に式を修正必須 |
| 2 | z = exp(-jω) 符号逆転 | 🔴 重大 | exp(+jω) に修正必須 |
| 3 | Q閾値の数学的誤記 | 🔴 重大 | 0.707 または √2 に修正必須 |
| 4 | Tukey窓-60dB非現実的 | 🔴 重大 | テスト基準を修正必須 |
| 5 | IRロード完了位置誤記 | 🟡 中 | UIEvents.cpp に修正 |
| 6 | inputクランプがルーティング依存 | 🟡 中 | 文書に明記必須 |
| 7 | setProcessingOrder の二重rebuild intent | 🟡 中 | 呼び出し位置の明示必須 |
| 8 | 0.15係数の理論的根拠不足 | 🟡 中 | 文書に「ヒューリスティック」と明記 |
| 9 | 予測型方式の業界先行例なし | 🟡 中 | テストの慎重化 |

### 7.3 結論

改訂版は初版の問題を大幅に改善しているが、**DSP処理チェーンの実装詳細との照合が不十分**であり、特に Conv→PEQ モードの `convolverInputTrimGain` 未適用は実装すると意図通り動作しない重大な設計欠陥である。この問題は式の修正のみで対応可能（§5.1の修正案A）。

また、`z = exp(-jω)` の符号、Q閾値の数学的誤記、Tukey窓のテスト基準も実装前に修正すべき項目である。

これら4つの重大問題を修正すれば、本計画は実行可能で技術的に妥当な設計書となる。

---

## 8. 引用文献・参考資料

1. Julius O. Smith III, *Introduction to Digital Filters with Audio Applications*, CCRMA, Stanford University — https://ccrma.stanford.edu/~jos/filters/
2. W3C Audio EQ Cookbook (Robert Bristow-Johnson) — https://www.w3.org/TR/audio-eq-cookbook/
3. Wikipedia — Digital biquad filter — https://en.wikipedia.org/wiki/Digital_biquad_filter
4. Wikipedia — Q factor — https://en.wikipedia.org/wiki/Q_factor
5. Wikipedia — Butterworth filter — https://en.wikipedia.org/wiki/Butterworth_filter
6. Wikipedia — Window function (Tukey window) — https://en.wikipedia.org/wiki/Window_function
7. Wikipedia — EBU R 128 — https://en.wikipedia.org/wiki/EBU_R128
8. ITU-R BS.1770-4 — Algorithms to measure audio programme loudness and true-peak audio level — https://www.itu.int/rec/R-REC-BS.1770
9. EBU R 128 — Loudness normalisation and permitted maximum level — https://tech.ebu.ch/publications/r128/
10. Wikipedia — Loudness war (streaming platform targets) — https://en.wikipedia.org/wiki/Loudness_war
