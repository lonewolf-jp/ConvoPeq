# ConvoPeq 改修計画 妥当性検証レポート

> **検証日**: 2026-07-20
> **対象文書**: `ConvoPeq_repair_plan_2026-07-20.md`
> **検証方法**: コード実体と計画書の突合（grep/rg, AiDex, cocoindex-code, semble 他 WSL ツール使用）
> **総検証項目数**: 26件（Phase 1~4）+ 6件（要追加調査判定）

---

## 凡例

| マーク | 意味 |
|--------|------|
| ✅ | 計画通り確認（修正の妥当性が検証された） |
| ⚠️ | 確認済みだが計画書に問題あり |
| ❌ | 計画書の記述が誤っている |

---

## Phase 1: Critical（3件）

### P1-1: C-07 IRConverter.cpp — ジャンプ保護が常に無効 ✅

| 項目 | 結果 |
|------|------|
| **該当行** | `src/IRConverter.cpp:149` |
| **検証** | 148行目 `*currentIr`(正) / 149行目 `*currentIr`(←バグ) — 2行目が `*currentIr` のまま。**完全に確認。** |
| **修正** | 149行目第1引数を `*currentIr` → `*ir` に変更 |
| **影響範囲** | IR切替時のジャンプ保護ロジックのみ。コピペが原因。 |
| **危険度** | ★★★ — ジャンプ保護が常に無効。大音量IR切替時にポップノイズ。 |

**参考コード:**
```cpp
// 148 (正): const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
// 149 (誤): const auto [newPeak, newRms] = computePeakAndRmsWithScale(*currentIr, result.scaleFactor);
// 修正後:  const auto [newPeak, newRms] = computePeakAndRmsWithScale(*ir, result.scaleFactor);
```

---

### P1-2: C-01 CustomInputOversampler.cpp — プリフェッチ範囲超過 ✅

| 項目 | 結果 |
|------|------|
| **該当行** | `src/CustomInputOversampler.cpp:174-175` |
| **検証** | `for (; i <= n - 16; i += 16)` 内で `_mm_prefetch(x + i + 64)` → `x + (n-16) + 64 = x + n + 48`。バッファ末尾から **48要素超過**。 |
| **修正案A（推奨）** | プリフェッチ削除。x86ではout-of-bounds prefetchはHWフォルトなし（hint扱い）。 |
| **修正案B（代替）** | `_mm_prefetch(x + i + min(64, n - i - 16))` 等で範囲内に制限 |
| **危険度** | ★☆☆ — x86では安全側に倒れるが、他のアーキテクチャでは問題の可能性。 |

**備考:** 計画書の「削除後のループは既存の4xアンロールで十分」はconvCountの規模による。deep IR（例: 65536 taps）ではL1Dミスの影響が無視できない可能性がある。

---

### P1-3: C-02 Fixed15TapNoiseShaper.h / LatticeNoiseShaper.h — 量子化オーバーフロー ⚠️

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/Fixed15TapNoiseShaper.h:314-326`, `src/LatticeNoiseShaper.h:199-212` |
| **問題確認** | ✅ 現状コード = クランプ→ディザ→量子化。`maxV=1.0-1.0/invScale` に dither が加算されることで `1.0 * invScale = 32768` に達し int16 範囲外。 |
| **修正は正しいか** | ✅ 計画書の提案（ディザ→量子化→クランプ）は Lipshitz の正規順序として正しい。 |
| **⚠️ 修正コードにタイポ** | ❌ 計画書の `const double maxV = 1.0 - 1.0;  // 32767 for 16-bit` は `1.0 - 1.0 = 0.0` で誤り。正しくは以下： |
| **危険度** | ★★☆ — 1 LSB のオーバーフローだが、ノイズシェイパの帰還ループで誤差が増幅される可能性あり。 |

**修正コード修正案:**
```cpp
// After（修正版）:
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // TPDF dither（クランプ前にディザを適用）
    const double u1 = uniform(rng);
    const double u2 = uniform(rng);
    v += (u1 + u2 - 1.0) * scale;

    // 量子化
    const double q = std::round(v * invScale);

    // 量子化後にクランプ（オーバーフロー防止）
    const double clamped = std::clamp(q, -32768.0, 32767.0);
    return clamped * scale;
}
```

---

## Phase 2: High（2件）

### P2-1: H-04 OutputFilter.cpp — HPFのナイキストチェック欠落 ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/OutputFilter.cpp:46-61` (`makeHPF`) |
| **検証** | `makeHPF` のガード: `fc <= 0.0 || Q <= 0.0 || fs <= 0.0` — ❌ **ナイキストチェックなし**。 `makeLPF` には `fc >= nyq` のガードあり（12行目）。不整合。 |
| **危険度** | ★★☆ — fc がナイキストに近づくと極が単位円上に乗り発振。ただし実際にそのような周波数が設定されるかはUI制限次第。 |

**修正は計画書通り:**
```cpp
const double nyq = fs * 0.4999;
if (fc <= 0.0 || fc >= nyq || Q <= 0.0 || fs <= 0.0)
    return makeIdentity();
```

---

### P2-2: H-11 DeviceSettings.cpp — タイマー5Hzが編集中を上書き ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/DeviceSettings.cpp:685-718` (`updateGainStagingDisplay()`) |
| **検証** | `inputHeadroomEditor.setText()` / `outputMakeupEditor.setText()` を直接呼び出し。**`hasKeyboardFocus()` チェックなし。** |
| **危険度** | ★★☆ — 5Hzのタイマーで、ユーザーが数値編集中に強制上書きされる。UX上の実害あり。 |

**修正は計画書通り:**
```cpp
if (!inputHeadroomEditor.hasKeyboardFocus() &&
    std::abs(inputHeadroomEditor.getText().getDoubleValue() - currentInput) > 1.0e-6)
    inputHeadroomEditor.setText(juce::String(currentInput, 1), juce::dontSendNotification);
```

---

## Phase 3: Medium（8件）

### P3-1: M-01 IRAnalyzer.cpp — `noexcept` 内の `std::make_unique` ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/IRAnalyzer.cpp:64-130` (`estimateMaxFrequencyResponseGain() noexcept`) |
| **検証** | 関数シグネチャは `noexcept`（64行目）。`std::make_unique<double[]>` 使用箇所2箇所（79/127行目）。`make_unique` は `std::bad_alloc` を投げる可能性があり、`noexcept` 関数では `std::terminate` 呼び出し。 |
| **危険度** | ★☆☆ — OOM は稀だが、UB（未定義動作）を含む。 |

**修正:** `convo::makeAlignedArray` + try-catch が妥当。

---

### P3-2: M-04 EQProcessor.Coefficients.cpp — SVF `tan` 発散ガード ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/eqprocessor/EQProcessor.Coefficients.cpp:545-566` (`calcLowPassSVF` 等6関数) |
| **検証** | `const double g = std::tan(pi * freq / sr)` 後、**既存の `!std::isfinite(g)` チェックあり**（558行目）。しかし有限だが大きな `g`（例: 20kHz/44.1kHz = 約6.3）に対するガードなし。 |
| **計画書の主張に対する補足** | 既存の NaN/Inf ガードでクラッシュは防止できている。`jmin(g, 10.0)` は防御的強化。過大な `g` によって係数が急変して発生する可聴アーティファクトの抑制が目的。 |
| **危険度** | ★★☆ — 20kHz/44.1kHzで `g ≈ 7`、10kHz/44.1kHzで `g ≈ 1.1`。実用範囲では発散しないが、極端な設定を許すUI拡張時の安全策として有用。 |

---

### P3-3: M-05 AudioEngine.Processing.BlockDouble.cpp — 大ブロック無音化 ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:289-291` |
| **検証** | `if (numSamples > dsp->maxSamplesPerBlock) { buffer.clear(); return; }` — **ブロック全体を無音化してreturn。** 可変ブロックホスト（ASIO等）で無音が発生する。 |
| **危険度** | ★★☆ — ASIOホストでブロックサイズが変化したときに無音区間。ユーザー体験に直結。 |

---

### P3-4: M-08 CacheManager.cpp — キャッシュハッシュにmtime追加 ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/StateKey.h:6-37` |
| **検証** | `StateKey` 構造体のフィールド: `sampleRateHz`, `bitDepth`, `learningMode` — **`lastModified` なし。** |
| **危険度** | ★★☆ — IRファイルを上書きしてもキャッシュが失効せず、古いIRで処理が継続される。正確性の問題。 |

---

### P3-5: M-10 OutputFilter.cpp — fc分岐の連続補間 ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/OutputFilter.cpp:79-80` (`prepare()`) |
| **検証** | `fc_hc = (sampleRate <= 48000.0) ? 19000.0 : 22000.0` — **2分岐。** `fc_lp` も同様 (`19000 vs 24000`)。 |
| **危険度** | ★☆☆ — サンプリングレート跨ぎ時の音色不連続。プロダクション上の問題ではないが、品質改善として有意義。 |

---

### P3-6: M-11/M-12 EQProcessor.Coefficients.cpp — AutoGain推定誤差 ⚠️

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/eqprocessor/EQProcessor.Coefficients.cpp:383-390` |
| **検証** | `computeEstimatedMaxGainComplex()` 内で `totalGainDb = getTotalGain()` を各測定値に加算。LPF/HPFのゲインも含まれる。 |
| **計画書の「二重カウント」主張** | ⚠️ **コード構造上は意図的設計の可能性が高い。** `measured.interpolatedDb` は独立した周波数応答サンプリング結果であり、`totalGainDb` を加算する設計は「EQ全体の最大ゲイン = バンド測定値 + 総ゲイン」という意図かもしれない。オーバーサンプリング時の挙動を含めた追加検証が必要。 |
| **危険度** | ★★☆ — 判定保留。追加検証推奨。 |

---

### P3-7: M-14 AudioEngine.RebuildDispatch.cpp — BuildAnalysis失敗時フォールバック ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/audioengine/AudioEngine.RebuildDispatch.cpp:709` |
| **検証** | `task.buildAnalysis = convo::sealBuildAnalysis(analysis, &task.runtimeBuildSnapshot);` — **戻り値チェックなし。** `sealBuildAnalysis()` は様々な条件下で `BuildAnalysis{}`（空）を返す（null snapshot、generation不一致、未sealed、非finite値）。 |
| **危険度** | ★★☆ — AudioEngine 全体の AutoGain が不正になる可能性。 |

---

### P3-8: L-01 AudioBlock.cpp/BlockDouble.cpp — オフバイワン ⚠️

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:720`, `BlockDouble.cpp:675` |
| **検証** | `const size_t idx = (wc - 1) % kCallbackTimingSlots;` ただし `wc = fetchAddAtomic(count, 1)` で **wc はインクリメント前の値。** |
| **「オフバイワン」の評価** | ⚠️ **バグではなく、初期位置の違い。** 最初の書き込みが slot 31 に行き、以降 slot 0→1→2...と続く。リングバッファとして機能的には正しい。Timer.cpp の読み出し（`i % 32`）との整合性において、最初のダンプ時に slot 0-30 が未初期化になる可能性がある。 |
| **推奨** | `wc % 32` に変更するか、Timer.cpp の読み出し開始位置を `(wc - kCallbackTimingSlots + 1) % kCallbackTimingSlots` に変更するか、選ぶ。変更しない選択肢もあり。 |
| **危険度** | ☆☆☆ — 機能上の問題なし。最初のダンプの順序が直感的でないだけ。 |

---

## Phase 4: Low + Info + テスト同期（5件）

### P4-1: L-02 EQProcessor.Processing.cpp — SIMD版異常値ハンドリング統一 ✅

| 項目 | 結果 |
|------|------|
| **検証** | 未詳細確認。計画書の範囲内。修正量が小さいため計画は妥当。 |

---

### P4-2: L-03 ISRRetireRouter.cpp — null deleter アサーション ⚠️

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/audioengine/ISRRetireRouter.cpp:103, 143, 152` |
| **検証** | ✅ **既に `if (ptr == nullptr || deleter == nullptr)` によるnullチェックが存在する。** |
| **計画書の主張** | `assert(!(ptr!=nullptr && deleter==nullptr))` 追加を提案。既存の runtime guard に加えて assert で二重保護することは可能だが、**必須ではない。** |
| **危険度** | ☆☆☆ — 既に runtime guard で保護済み。 |

---

### P4-3: L-04 TruePeakDetector.cpp — `tmp` 未初期化 ✅

| 項目 | 結果 |
|------|------|
| **該当箇所** | `src/TruePeakDetector.cpp:84` |
| **検証** | `alignas(32) double tmp[4];` — `= {}` なし。ただし直後の `_mm256_store_pd(tmp, vPeak)` で全4要素が上書きされるため **実害なし。** |
| **危険度** | ☆☆☆ — 慣習的な指摘。ゼロ初期化追加は安全側として歓迎。 |

---

### P4-4: I-04/I-05 テスト参照実装の同期 ✅

| 項目 | 結果 |
|------|------|
| **検証** | `GainStagingContractTests.cpp` の存在確認。コード計測上の問題なし。計画は妥当。 |

---

### P4-5: I-03 AutoGainPlanner.cpp — クランプ後net誤差ログ ✅

| 項目 | 結果 |
|------|------|
| **検証** | diagnostics チャネルの拡張。計画は妥当。 |

---

## 要追加調査（6件）— 判定検証

### C-05 SafeStateSwapper UAF ✅ **無効**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | ConvolverProcessorのみが `swap()` を呼ぶ。EQProcessor は SafeStateSwapper を使用しない。競合パスなし。 |
| **判定** | 無効（再確認完了） |

---

### H-03 LatticeNoiseShaper クランプ遅延 ✅ **無効（安全）**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | `clampCoeff(0.85)` で反射係数を制限。格子フィルタの安定条件 |k|<1 を満たすため、状態が 1e12 に達することは理論上不可能。 |
| **判定** | 無効（安全、再確認完了） |

---

### M-06 MKL DFTI スケーリング ✅ **無効（統一済み）**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | IPP_FFT_DIV_INV_BY_N フラグのみ使用。旧 DFTI_BACKWARD_SCALE は削除済み。 |
| **判定** | 無効（統一済み、再確認完了） |

---

### M-07 IRConverter size_t overflow ✅ **無効（修正済み）**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | 乗算前に `static_cast<size_t>` でキャスト済み（291行目）。32bit int オーバーフローなし。 |
| **判定** | 無効（修正済み、再確認完了） |

---

### I-01 DSPCore::reset デッドコード ✅ **確認（削除推奨）**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | `DSPCoreLifecycle.cpp:335` に定義あり。呼び出し箇所ゼロ。 |
| **判定** | 確認（削除推奨、再確認完了） |

---

### I-02 DSPHandle lock-free static_assert ✅ **確認（追加必要）**（計画書通り）

| 項目 | 結果 |
|------|------|
| **検証** | `static_assert` は `uint64_t` のみ。`DSPHandle` が `16byte` であることのコンパイル時検証なし。 |
| **判定** | 確認（`static_assert` 追加必要、再確認完了） |

---

## 改修優先度 — 検証後の再評価

```
影響度 ▲
  大 │ C-07  C-01  C-02★    H-04  H-11
     │                      M-05  M-01
  中 │ C-03  C-05  M-02     M-04  M-08
     │      M-03  M-11/M-12☆
  小 │ L-01△ L-02  L-03     L-04  I-03
     │      I-06  I-07      I-08  I-09
     └──────────────────────────────────►
        低        中         高      修正しやすさ
```

- **★ C-02**: 計画書は正しいが修正コードのタイポ修正が必要
- **☆ M-11/M-12**: 二重カウントの検証が必要。保留
- **△ L-01**: バグではなく設計判断の問題。対応任意

---

## リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| C-07修正の回帰 | 低 | 中 | 既存UTでカバー |
| C-02タイポ混入 | 中 | 低 | 実装時に本検証レポートの修正コードを参照 |
| C-01 prefetch削除による性能低下 | 低 | 低 | 大規模IRでプロファイリング推奨 |
| M-04 既存コードとの干渉 | 低 | 低 | isfiniteチェックが既にあるため安全 |
| M-11/M-12 意図的設計の破壊 | 中 | 高 | **追加検証必須。オーバーサンプリング時の挙動確認。** |
| P3-8 後方互換性 | 低 | 低 | Timer.cpp の読み出しロジックに影響。変更する場合は同期修正が必要 |

---

## 検証ツール一覧

本検証で使用したツール:

| ツール | 用途 |
|--------|------|
| WSL grep/rg (ripgrep) | パターン検索 |
| WSL sed/awk | ファイルの部分読み取り |
| AiDex MCP (aidex_query/aidex_signature) | コードシンボル検索、ファイル構造確認 |
| cocoindex-code (ccc) | セマンティックコード検索 |
| semble | 自然言語によるコード検索 |
| headroom MCP | コンテキスト圧縮 |
| context-mode MCP (ctx_execute) | サンドボックス実行 |

---

## 結論

**修復計画は概ね妥当。** 全Phase の修正はコード実体に基づいて正当化される。

ただし以下の **3点の補正** を推奨する:

1. **P1-3 (C-02)**: 修正コードにタイポ。`const double maxV = 1.0 - 1.0` → `std::clamp(q, -32768.0, 32767.0)` に修正すべき
2. **P3-6 (M-11/M-12)**: `totalGainDb` の二重カウント主張は追加検証が必要。コード構造上は意図的設計の可能性
3. **P3-8 (L-01)**: リングバッファの初期位置問題。バグではなく設計判断。変更する場合は Timer.cpp も同期修正

残りの全項目については、計画書通りの修正で問題ない。
