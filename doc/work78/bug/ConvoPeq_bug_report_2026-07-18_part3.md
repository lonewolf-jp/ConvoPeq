# ConvoPeq ソースコード調査報告書（Part 3）

Part 1/2 の続き。今回は最優先項目としていた `src/eqprocessor/` 一式（`EQProcessor.Processing.cpp`, `EQProcessor.Core.cpp`, `EQProcessor.ProcessingCache.cpp`）を精読しました。

---

## 0. 今回のサマリ

| # | 重大度 | ファイル | 概要 |
|---|--------|----------|------|
| 3 | **Low**（極端な異常値時のみ発現するエッジケース） | `EQProcessor.Processing.cpp` | `processBand`（スカラー）と `processBandStereo`（SIMD）で、`\|output\|≥1e15` という極端な有限値の扱いが異なる。スカラー版は無音化(0.0)、SIMD版はNaN/Infのみ判定するため±100クランプ止まり（+40dBFS相当の非常に大きな値）になり得る。コメントには「processBandStereoと一貫性を保つ」とあるが実際には不一致。 |
| 4 | **要確認**（確定バグではないが内部矛盾あり） | `EQProcessor.Core.cpp` | `EQProcessor::reset()` はコメント上 "(Audio Thread)" だが、`juce::Decibels::decibelsToGain()`（内部で`std::pow`=libm呼び出し）を直接2回（直接呼び出し1回+`storeTotalGainDb()`経由で1回）実行している。これは`process()`側で「【Fix Bug #7】」として明示的に排除済みの禁止パターンと同一。Audio Threadから実際に呼ばれているかの確証は得られなかったため「確定バグ」ではなく「要確認」として報告する。 |

---

## 1. 【Low】processBand と processBandStereo の異常値ハンドリング不一致

### 該当ファイル・行

`src/eqprocessor/EQProcessor.Processing.cpp`

- スカラー版 `processBand`（ローカル112-170行目）
- SIMD版 `processBandStereo`（ローカル175-255行目）

両者はTPT SVF（Zavalishin方式、Cytomic/Andrew Simper型の実装）の同一アルゴリズムをスカラー/SSE2の2通りで実装したものです。**フィルタ係数計算・状態更新式自体は数式的に完全に正しいことを確認済みです**（`ic1eq`/`ic2eq`の更新式、FMA変換、L/Rパッキングのlower/upper対応関係を含め、参照文献 Zavalishin "Art of VA Filter Design" のTPT SVF方程式と1対1で対応することを検算しました）。問題はフィルタ本体ではなく、その後段の異常値セーフガードのみです。

### 現在のコード

スカラー版（`processBand`、抜粋）:
```cpp
            // NaN/Infチェックとクランプを追加 (processBandStereoと一貫性を保つ)
            if (!isFiniteAndAbsInRangeMask(output, 0.0, 1.0e15))
                output = 0.0;

            // 出力もクランプして発散を防ぐ
            data[n] = std::clamp(output, -100.0, 100.0);
```

`isFiniteAndAbsInRangeMask(output, 0.0, 1.0e15)` は「有限」かつ「`0.0 <= |output| < 1.0e15`」を判定します（下限0.0は実質無条件のため、事実上「有限かつ`|output|<1e15`」のチェックです）。該当しなければ**0.0に無音化**した上で、さらに`std::clamp(-100,100)`を適用します。

SIMD版（`processBandStereo`、抜粋）:
```cpp
            // NaN/Infチェック (isfinite): (x - x) は xがInf/NaNの時NaNになる
            const __m128d diff = _mm_sub_pd(output, output);
            const __m128d mask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
            output = _mm_and_pd(output, mask);

            // クランプ (-100, +100) で発散防止
            output = _mm_min_pd(_mm_max_pd(output, cLow), cHigh);
```

SIMD版は **NaN/Infのみ** を判定して無音化し、`|output|`の大きさそのものは見ていません。`output`が有限だが`1e15`以上（例: フィルタが不安定化して`5e15`のような値を出した場合）でも通過し、直後の`_mm_min_pd/_mm_max_pd`によって**±100にクランプされるだけ**になります。

### 影響

通常運用（正しい係数・通常のオーディオ信号）では`output`が1e15に達することはまずありません。しかし本コードが`ic1eq`/`ic2eq`の発散防止チェックを別途持っていること自体、極端な係数やパラメータ変更直後の過渡応答等で状態が発散し得ることを前提にした設計です。そのような発散が実際に起きた場合:

- Mid/Side/Monoパスを通る `processBand` → 出力は **0.0（無音）**
- ステレオパスを通る `processBandStereo` → 出力は **±100.0**（フルスケールの100倍 = +40dBFS相当）

同一の異常事態に対して、チャンネルモード次第で「無音」と「非常に大きな信号」という正反対の挙動になります。後段に必ずリミッター等があれば実害は限定的ですが、`processBand`自身のコメントが「processBandStereoと一貫性を保つ」と明記している以上、これは設計意図と実装の乖離です。

### 検証根拠

- 両関数のNaN/Infチェック部分を並べて比較し、`isFiniteAndAbsInRangeMask`（3引数、大きさチェックあり）と`_mm_cmpeq_pd(diff,0)`のみ（大きさチェックなし）という非対称性を確認。
- `processBand`のコメント「(processBandStereoと一貫性を保つ)」という明示的な設計意図を確認。
- 具体的な数値例（`output=500.0`等）で追跡し、1e15未満の領域では両者とも最終的に±100クランプで同じ結果になること、乖離が生じるのは`|output|>=1e15`という極端域に限られることを確認済み（誤検知を避けるための裏取り）。

### 修正パッチ

SIMD版に同じ大きさチェックを追加し、スカラー版と同じ「1e15以上は無音化」に統一します。

```diff
--- a/src/eqprocessor/EQProcessor.Processing.cpp
+++ b/src/eqprocessor/EQProcessor.Processing.cpp
@@ -193,6 +193,12 @@
         const __m128d two = _mm_set1_pd(2.0);
         const __m128d cHigh = _mm_set1_pd(100.0);
         const __m128d cLow  = _mm_set1_pd(-100.0);
+        // ★ FIX: processBand（スカラー版）と同じ「|output|>=1e15 も無効値として0扱い」
+        //   にするための閾値。従来はNaN/Infのみ判定しており、有限だが超巨大な値が
+        //   そのまま±100クランプに流れ込み、スカラー版（0.0）と異なる非常に大きな
+        //   （+40dBFS相当）出力になり得た。
+        const __m128d cHugeAbs = _mm_set1_pd(1.0e15);
+        const __m128d cAbsMask = _mm_set1_pd(-0.0);
 
         [[maybe_unused]] constexpr double DENORMAL_THRESHOLD = convo::numeric_policy::kDenormThresholdAudioState;
 
@@ -233,8 +239,12 @@
             }
 
             // NaN/Infチェック (isfinite): (x - x) は xがInf/NaNの時NaNになる
+            // + |output|<1e15 チェック（processBandスカラー版と同一のセーフティ条件に統一）
             const __m128d diff = _mm_sub_pd(output, output);
-            const __m128d mask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
+            const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
+            const __m128d absOutput = _mm_andnot_pd(cAbsMask, output);
+            const __m128d rangeMask = _mm_cmplt_pd(absOutput, cHugeAbs);
+            const __m128d mask = _mm_and_pd(finiteMask, rangeMask);
             output = _mm_and_pd(output, mask);
 
             // クランプ (-100, +100) で発散防止
```

追加コストはループ内で`_mm_andnot_pd`/`_mm_cmplt_pd`/`_mm_and_pd`が各1命令増える程度で、AVX2/FMA環境下では無視できるレベルです。

---

## 2. 【要確認】EQProcessor::reset() の "(Audio Thread)" ラベルと decibelsToGain (libm) 呼び出し

### 該当ファイル・行

`src/eqprocessor/EQProcessor.Core.cpp`、`EQProcessor::reset()`（ローカル237-274行目付近）

### 現在のコード

```cpp
//============================================================================
// フィルタ状態リセット (Audio Thread)
//============================================================================
void EQProcessor::reset()
{
    // フィルタ状態をリセット (memsetで高速化)
    std::memset(filterState.data(), 0, sizeof(filterState));

    convo::publishAtomic(agcCurrentGain, 1.0, std::memory_order_release);
    convo::publishAtomic(agcEnvInput, 0.0, std::memory_order_release);
    convo::publishAtomic(agcEnvOutput, 0.0, std::memory_order_release);

    auto state = loadCurrentState(std::memory_order_acquire);
    if (state)
    {
        smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
        storeTotalGainDb(state->totalGainDb);
    }
    ...
```

### 問題点

1. 関数直上のコメントは明確に **「(Audio Thread)」** と記載しています。
2. `juce::Decibels::decibelsToGain<double>()` は内部で `std::pow()` を呼びます。これはコーディング規約で明示的に禁止されているlibm呼び出しです。
3. 同一クラスの `EQProcessor::process()` には、まさにこの関数について「**【Fix Bug #7】Audio Thread内でのlibm呼び出し禁止。juce::Decibels::decibelsToGain()は内部でstd::pow()(libm)を呼ぶためAudio Thread内での使用は規約違反である**」という明示的な修正コメントがあり、dB→linear変換をMessage Thread側の`storeTotalGainDb()`に集約し、Audio Threadは`totalGainTarget`（`std::atomic<double>`）を読むだけにする設計へ既に是正されています。
4. ところが`reset()`はこの是正パターンに従わず、`decibelsToGain`を**直接呼び出す**上に、その次の行で`storeTotalGainDb()`（内部でも`decibelsToGain`を呼ぶヘルパー）を**さらに呼んで同じ値を二重計算**しています。

### 実際にAudio Threadから呼ばれるかの検証状況

`AudioEngine::DSPCore::reset()` → `eqState->resetForRuntime()` → `EQProcessor::reset()` という呼び出し連鎖は確認できました。しかし `DSPCore::reset()` 自体の呼び出し元は、今回精読した Audio Thread 本体（`AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp`）内には見つからず、他の全ファイルを横断検索しても具体的な呼び出し箇所を特定できませんでした。

一方で、同じ`EQProcessor.Core.cpp`内の類似関数（`syncStateFrom`は`jassert(isThisTheMessageThread())`で保護、`syncGlobalStateFrom`はコメントで「Worker Threadからも安全」と明記）と比較すると、`reset()`だけが「Audio Thread」を名乗りながら実行スレッドを保証する仕組み（jassert等）を一切持たない点が際立ちます。

**このため「確定バグ」ではなく「要確認」として報告します。** 以下のいずれかだと考えられます:

- (a) 実際にAudio Threadから呼ばれる経路が本当に存在し、`process()`で既に修正済みの規約違反が`reset()`に再発している（＝真のバグ）。
- (b) `reset()`は実際には非RTスレッド（DSPCore構築時のウォームアップ処理等）からしか呼ばれておらず、コメントの「(Audio Thread)」は不正確・古い記述である（＝ドキュメント上の問題）。

いずれであってもコメントと実装のどちらか一方に不整合があるため、修正または確認を推奨します。

### 推奨対応

**(a) の場合（Audio Threadから呼ばれる）:** `decibelsToGain`の直接呼び出しと`storeTotalGainDb()`呼び出しの両方をAudio Thread側から除去する必要があります。`storeTotalGainDb()`自体もMessage Thread専用ヘルパーであるため、単純な呼び出し順序の入れ替えでは根本解決になりません。`reset()`側では既に`prepareToPlay()`等で発行済みのはずの`totalGainTarget`アトミック値を読むだけに留める設計変更が必要です（`process()`と同じ設計思想）。

**(b) の場合（実際は非RT専用）:** コメントを実態に合わせて修正するだけで問題ありません。ついでに、直接呼び出し+`storeTotalGainDb()`経由の二重計算になっている非効率（同じdB値から同じlinear値を2回計算している）を解消することを推奨します。

どちらの対応が適切か判断するため、`DSPCore::reset()`の実際の呼び出し元（おそらく`RuntimeBuilder.cpp`や`ProgressiveUpgradeThread.cpp`等、まだ精読できていないファイル群）の確認を次の調査候補とします。

---

## 3. 調査範囲の更新

`src/eqprocessor/` 一式の状況:

- `EQProcessor.Processing.cpp`（1259行）: **精読完了**。`process()`（2オーバーロード）, `processAGC`, `processBand`, `processBandStereo`, `applyGainRamp_AVX2` 等、Audio Thread最重要パスを一通り確認。
- `EQProcessor.Core.cpp`（860行）: **精読完了**。コンストラクタ/デストラクタ, `resetToDefaults`, `reset`, `releaseResources`, `syncStateFrom`, `syncGlobalStateFrom`, EBR経由の削除処理を確認。
- `EQProcessor.ProcessingCache.cpp`（89行）: **精読完了**。`computeParamsHash`, `createCoeffCache`（非RT確認済み）。
- `EQProcessor.Coefficients.cpp`: 前回セッションで一部確認済み（libm呼び出しがMessage Thread専用であることのみ）。係数計算式自体（周波数ワーピング、Q変換等）の数式的検証は未実施。
- `EQProcessor.Parameters.cpp`: 前回セッションで確認済み（RCUパターン、UIスレッド専用）。
- `EQProcessor.h`: 断片的にしか見ていません（インライン関数・メンバ変数宣言の全体像は未確認）。

---

## 4. 次のステップ（提案）

引き続き「つづけて」で以下の優先順で継続できます:

1. `EQProcessor.Coefficients.cpp` の係数計算式そのものの数式検証（未実施）
2. `RuntimeBuilder.cpp` / `ProgressiveUpgradeThread.cpp` — 上記2節の`DSPCore::reset()`呼び出し元特定
3. `ISR*` 系（RCU publish/retireの本体、約40ファイル）
4. `DSPCoreFloat.cpp` / `DSPCoreIO.cpp` / `DSPCoreToBuffer.cpp`
5. `ConvolverProcessor.*` 一式
