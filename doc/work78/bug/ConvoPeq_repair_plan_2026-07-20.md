# ConvoPeq 改修計画書

> **策定日**: 2026-07-20
> **根拠**: `ConvoPeq_consolidated_bug_verification_2026-07-20.md`（検証結果）
> **対象**: 検証結果が「確認（有効）」の26件 + 要追加調査5件のうち改修推奨分
> **方針**: 重大度順・ファイル単位でバッチ改修。各フェーズでビルド＋テスト検証を実施。

---

## フェーズ划分

| フェーズ | 内容 | バグ数 | 目安工数 |
|----------|------|--------|----------|
| **Phase 1** | Critical 確認件の即時修正 | 3件 | 半日 |
| **High 優先** | 音質・安全性に直結する High | 2件 | 半日 |
| **Phase 2** | 残り High + Medium 主要 | 8件 | 1日 |
| **Phase 3** | Low + Info + 要追加調査 | 10件 | 1日 |
| **Phase 4** | テスト同期・仕様確認 | 3件 | 半日 |

---

## Phase 1: Critical 即時修正（3件）

### P1-1: C-07 `IRConverter.cpp` — ジャンプ保護が常に無効

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/IRConverter.cpp` |
| **該当行** | 149行目 |
| **問題** | `computePeakAndRmsWithScale(*currentIr, result.scaleFactor)` の2行目が `*currentIr`（旧IR）を使用 |
| **修正** | 2行目の第1引数を `*currentIr` → `*ir` に変更 |
| **影響範囲** | IR切替時のジャンプ保護ロジックのみ。他の処理に影響なし。 |
| **検証** | 1. ビルド成功確認 2. IR切替時にジャンプ保護が発動するかUT 3. 異常IRで大音量ジャンプが抑制されるか手動検証 |

**修正コード案:**
```cpp
// Before:
const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*currentIr, result.scaleFactor);

// After:
const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*ir, result.scaleFactor);
```

### P1-2: C-01 `CustomInputOversampler.cpp` — プリフェッチ範囲超過

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CustomInputOversampler.cpp` |
| **該当行** | 174行目 |
| **問題** | `_mm_prefetch(x + i + 64)` がバッファ末尾から48要素超過 |
| **修正案A（推奨）** | プリフェッチを削除。x86ではプリフェッチ未ヒットでも例外なし。安全側に倒す。 |
| **修正案B** | プリフェッチ先を `x + i + convCount` に変更（バッファ範囲内に収める） |
| **影響範囲** | パフォーマンスに若干の影響（L1Dミス増）。XRUNリスクを優先。 |
| **検証** | 1. ビルド成功 2. WASAPI排他64サンプルでXRUN再現テスト 3. VTuneでL1D_PEND_MISS確認 |

**修正コード案（案A）:**
```cpp
// Before:
_mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
_mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);

// After: プリフェッチ削除（xWindowバッファが制約内で安全なため不要）
// 削除後のループは既存の4xアンロールで十分な隠蔽効果あり
```

### P1-3: C-02 `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` — 量子化オーバーフロー

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/Fixed15TapNoiseShaper.h`, `src/LatticeNoiseShaper.h` |
| **該当箇所** | `quantize()` 関数（両ファイル同一パターン） |
| **問題** | クランプ→ディザ→量子化の順序で `maxV + scale = 1.0` → int16で32768にオーバーフロー |
| **修正** | Lipshitzの正規順序「ディザ→量子化→クランプ」に変更。量子化後にクランプ追加。 |
| **影響範囲** | 16bit出力時の量子化ノイズ特性。聴感上の改善（破壊音消失）。 |
| **検証** | 1. ビルド成功 2. 16bit出力で破壊音が発生しないか確認 3. 量子化ノイズ特性の比較測定 |

**修正コード案（Fixed15TapNoiseShaper.h）:**
```cpp
// Before:
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);
    if (v < minV) v = minV;
    else if (v > maxV) v = maxV;
    // TPDF dither
    const double u1 = nextUniform(rng);
    const double u2 = nextUniform(rng);
    v += (u1 + u2 - 1.0) * scale;
    const double q = std::round(v * invScale);
    return q * scale;
}

// After:
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // TPDF dither（クランプ前にディザを適用）
    const double u1 = nextUniform(rng);
    const double u2 = nextUniform(rng);
    v += (u1 + u2 - 1.0) * scale;

    // 量子化
    const double q = std::round(v * invScale);

    // 量子化後にクランプ（オーバーフロー防止）
    // int16範囲: -32768 .. 32767
    const double clamped = std::clamp(q, -32768.0, 32767.0);
    return clamped * scale;
}
```

**修正コード案（LatticeNoiseShaper.h）:** 同様のパターンで適用。

---

## Phase 1 ビルド＋テスト検証

```bash
# Debug Build + Test
cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Debug
cd build && ctest -C Debug --output-on-failure
```

**検証項目:**
- [ ] ビルドエラーなし
- [ ] 既存テスト全パス
- [ ] C-07: IR切替ジャンプ保護の発動確認
- [ ] C-01: WASAPI排他64SPでXRUNなし
- [ ] C-02: 16bit出力で破壊音なし

---

## Phase 2: High 優先修正（2件）

### P2-1: H-04 `OutputFilter.cpp` — HPFのナイキストチェック欠落

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/OutputFilter.cpp` |
| **該当箇所** | `makeHPF()` 関数 |
| **問題** | `fc <= 0` のみチェック。fc がナイキストに近い場合、極が単位円上に乗り発振 |
| **修正** | `makeLPF` と同じ `fc >= nyq` チェックを追加。超過時は `makeIdentity()` を返す。 |
| **検証** | 1. ビルド成功 2. fc=nyquist*0.99 で発振しないことの確認 3. 既存のHPF使用箇所に影響なし |

**修正コード案:**
```cpp
BiquadCoeff OutputFilter::makeHPF(double fc, double Q, double fs) noexcept
{
    const double nyq = fs * 0.4999;
    if (fc <= 0.0 || fc >= nyq || Q <= 0.0 || fs <= 0.0)
        return makeIdentity();
    // ... 既存コード
}
```

### P2-2: H-11 `DeviceSettings.cpp` — タイマー5Hzが編集中を上書き

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/DeviceSettings.cpp` |
| **該当箇所** | `updateGainStagingDisplay()` |
| **問題** | `hasKeyboardFocus()` チェックなしで入力が上書きされる |
| **修正** | エディタにキーボードフォーカスがある場合はスキップ |
| **検証** | 1. ビルド成功 2. 数値入力中にタイマーで上書きされないことを確認 |

**修正コード案:**
```cpp
// Before:
if (inputHeadroomEditor.getText().getIntValue() != ...)
    inputHeadroomEditor.setText(..., dontSendNotification);

// After:
if (!inputHeadroomEditor.hasKeyboardFocus() &&
    inputHeadroomEditor.getText().getIntValue() != ...)
    inputHeadroomEditor.setText(..., dontSendNotification);
```

---

## Phase 3: Medium 主要修正（8件）

### P3-1: M-01 `IRAnalyzer.cpp` — `noexcept` 内の `std::make_unique`

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/IRAnalyzer.cpp` |
| **該当箇所** | `estimateMaxFrequencyResponseGain()` |
| **修正** | `std::make_unique` → `convo::makeAlignedArray` + try-catch |
| **検証** | ビルド成功 + OOM時のフェイルセーフ動作確認 |

### P3-2: M-04 `EQProcessor.Coefficients.cpp` — SVF `tan` 発散ガード

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **該当箇所** | `calcLowPassSVF()` 等のSVF係数計算 |
| **修正** | `g = tan(pi*freq/sr)` 後に `g = jmin(g, 10.0)` のような上限クランプ追加 |
| **検証** | 高域（20kHz/44.1kHz）で係数が急変しないことの確認 |

### P3-3: M-05 `AudioEngine.Processing.BlockDouble.cpp` — 大ブロック無音化

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` |
| **該当箇所** | `numSamples > maxSamplesPerBlock` チェック |
| **修正** | `buffer.clear()` の代わりにブロックを分割処理 |
| **検証** | 可変ブロックホスト（ASIO等）で無音が発生しないことの確認 |

### P3-4: M-08 `CacheManager.cpp` — キャッシュハッシュにmtime追加

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CacheManager.cpp`, `src/StateKey.h` |
| **該当箇所** | `StateKey` のハッシュ計算 |
| **修正** | `StateKey` に `lastModified` フィールド追加 + ハッシュに含める |
| **検証** | IRファイル上書き後にキャッシュが失効することの確認 |

### P3-5: M-10 `OutputFilter.cpp` — fc 分岐の連続補間

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/OutputFilter.cpp` |
| **該当箇所** | `prepare()` 内の `fc_hc` / `fc_lp` 設定 |
| **修正** | 2分岐を `jmap` で連続補間に変更 |
| **検証** | 88.2kHz/96kHz/192kHz で音色が連続的に変化することの確認 |

### P3-6: M-11/M-12 `EQProcessor.Coefficients.cpp` — AutoGain推定誤差 ⚠️ 判定保留

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/eqprocessor/EQProcessor.Coefficients.cpp` |
| **該当箇所** | `computeEstimatedMaxGainDb()` |
| **状態** | **判定保留** — 検証レポートで「コード構造上は意図的設計の可能性が高い」と指摘。オーバーサンプリング時の挙動を含めた追加検証が必要。 |
| **修正（仮）** | LPF/HPFのgainBoosting判定をQ閾値で条件付きに。totalGainDbの二重カウント除去。 |
| **検証** | AutoGainのヘッドルームが適切に計算されることの確認。**特にオーバーサンプリング時・Master Gain使用時の挙動を確認。** |
| **リスク** | 意図的な設計を破壊する可能性あり。修正前にコード意図の確認必須。 |

### P3-7: M-14 `AudioEngine.RebuildDispatch.cpp` — BuildAnalysis失敗時のフォールバック

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/audioengine/AudioEngine.RebuildDispatch.cpp` |
| **該当箇所** | `sealBuildAnalysis` 戻り値の使用箇所 |
| **修正** | 戻り値が `BuildAnalysis{}` の場合、フェイルセーフ処理（既存のデフォルト値使用）を明示 |
| **検証** | 生成失敗時にAutoGainが誤動作しないことの確認 |

### P3-8: L-01 `AudioEngine.Processing.AudioBlock.cpp` — オフバイワン ⚠️ 設計判断

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `BlockDouble.cpp` |
| **該当箇所** | `CallbackTimingHistory` リングバッファ |
| **状態** | **設計判断** — 検証レポートで「バグではなく初期位置の違い」と指摘。`wc` はインクリメント前の値を返すため、最初の書き込みが slot 31 に行く。機能的には正しい。 |
| **修正（任意）** | `(wc-1)%32` → `wc%32` に変更する場合、Timer.cpp の読み出しロジックも同期修正が必要。変更しない選択肢もあり。 |
| **検証** | CB_HISTダンプの最初のエントリが正しいことの確認（修正する場合のみ） |
| **リスク** | Timer.cpp との同期不備でダンプ順序が不正になる可能性 |

---

## Phase 3 ビルド＋テスト検証

```bash
cmake --build build --config Debug
cd build && ctest -C Debug --output-on-failure
```

---

## Phase 4: Low + Info + テスト同期（5件）

### P4-1: L-02 `EQProcessor.Processing.cpp` — SIMD版異常値ハンドリング統一

- SIMD版に `|output|<1e15` チェック追加

### P4-2: L-03 `ISRRetireRouter.cpp` — null deleter アサーション

- 3関数に `assert(!(ptr!=nullptr && deleter==nullptr))` 追加

### P4-3: L-04 `TruePeakDetector.cpp` — `tmp` 未初期化

- `alignas(32) double tmp[4] = {};` でゼロ初期化

### P4-4: I-04/I-05 テスト参照実装の同期

- `GainStagingContractTests.cpp` の `refPlan()` / `refQSafetyMargin()` を製品コードに同期

### P4-5: I-03 `AutoGainPlanner.cpp` — クランプ後のnet誤差ログ

- `clamped` フラグと実効net値をdiagnosticsに追加

---

## 要追加調査（判定確定）2026-07-20 調査完了

| ID | 内容 | 調査結果 | 判定 |
|----|------|----------|------|
| C-05 | SafeStateSwapper UAF | swap() 呼出し元は ConvolverProcessor のみ。EQProcessor は SafeStateSwapper を使用しない。EpochDomain は共有だが retirement/reclaim 専用。競合パスなし。 | 無効 |
| H-03 | LatticeNoiseShaper クランプ遅延 | kStateLimit=1e12 は clampCoeff(0.85) と組み合わせて安全。格子フィルタの反射係数が k0.85 に制限されているため、状態が 1e12 に達する前に収束。 | 無効（安全） |
| M-06 | MKL DFTI スケーリング | IPP_FFT_DIV_INV_BY_N フラグのみ使用。旧 DFTI_BACKWARD_SCALE は削除済み。スケーリング統一済み。 | 無効（統一済み） |
| M-07 | IRConverter size_t overflow | 乗算前に static_cast size_t にキャスト済み（line 291）。32bit int オーバーフローなし。 | 無効（修正済み） |
| I-01 | DSPCore::reset デッドコード | 定義は DSPCoreLifecycle.cpp:335 にあるが、呼出し箇所がゼロ。ヘッダコメントでのみ参照。デッドコード確認。 | 確認（削除推奨） |
| I-02 | DSPHandle lock-free | std::atomic DSPHandle は16byte。static_assert は uint64_t のみ。CMPXCHG16B で実質 lock-freeだが、コンパイル時検証が不足。 | 確認（static_assert 追加必要） |

---

## 改修優先度マトリクス

```
影響度 ▲
  大 │ C-07  C-01  C-02     H-04  H-11
     │                      M-05  M-01
  中 │ C-03  C-05  M-02     M-04  M-08
     │      M-03  M-11/M-12
  小 │ L-01  L-02  L-03     L-04  I-03
     │      I-06  I-07      I-08  I-09
     └──────────────────────────────────►
        低        中         高      修正しやすさ
```

---

## リスク管理

| リスク | 対策 |
|--------|------|
| 修正による回帰 | 各フェーズでビルド+既存テスト実行 |
| C-02量子化修正によるノイズ特性変化 | 修正前後のA/B比較測定 |
| M-05ブロック分割によるレイテンシ影響 | 最大分割回数を制限（例: max 4分割） |
| M-08mtime追加によるパフォーマンス影響 | mtime取得はファイルロード時のみ |

---

## 進捗管理

| フェーズ | 状態 | 開始日 | 完了日 | 担当 |
|----------|------|--------|--------|------|
| Phase 1 | 未着手 | | | |
| Phase 2 | 未着手 | | | |
| Phase 3 | 未着手 | | | |
| Phase 4 | 未着手 | | | |
| 要追加調査 | **完了** | 2026-07-20 | 2026-07-20 | AI |

---

## 調査結果サマリー（2026-07-20 追加）

### 6件の未確定事項 → 全件確定

| 結果 | 件数 | 内容 |
|------|------|------|
| **無効（調査結果）** | 4件 | C-05, H-03, M-06, M-07 |
| **確認（要対応）** | 2件 | I-01（デッドコード削除）, I-02（static_assert追加） |

### 主要な調査発見

1. **C-05（SafeStateSwapper UAF）**: EQProcessor は SafeStateSwapper を使用しない。ConvolverProcessor のみが swap() を呼ぶ。**競合は存在しない。**

2. **H-03（kStateLimit=1e12）**: clampCoeff が反射係数を 0.85 に制限。格子フィルタの安定条件 |k|1 を満たすため、状態は収束。1e12 への到達は理論上不可能。

3. **M-06（MKL DFTI）**: IPP_FFT_DIV_INV_BY_N のみ使用。旧 DFTI_BACKWARD_SCALE は完全に削除。スケーリング統一済み。

4. **M-07（size_t overflow）**: 乗算前に static_cast<size_t> でキャスト済み。オーバーフローなし。

5. **I-01（DSPCore::reset）**: 定義はあるが呼出しゼロ。デッドコード。削除推奨。

6. **I-02（DSPHandle lock-free）**: 16byte atomic に static_assert なし。CMPXCHG16B で実質 lock-freeだが、検証不足。

---

## 参考資料

- `ConvoPeq_consolidated_bug_list_2026-07-18.md` — 統合バグリスト（48件）
- `ConvoPeq_consolidated_bug_verification_2026-07-20.md` — 検証レポート
- 各 Part レポート（Part1〜6 + new_bug.md）— 出典元
