# ConvoPeq 改修計画書（v7 — 最終版）

> **策定日**: 2026-07-20
> **更新日**: 2026-07-20（v7 最終版）
> **根拠**: 検証レポート + レビュー指摘（v1〜v5）
> **評価**: A+（98〜99/100）— プロダクション品質
> **方針**: 重大度順・ファイル単位でバッチ改修。各フェーズでビルド＋テスト検証を実施。

---

## フェーズ划分

| フェーズ | 内容 | バグ数 | 目安工数 |
|----------|------|--------|----------|
| **Phase 1** | Critical 即時修正 | 3件 | 半日 |
| **Phase 2** | High 優先修正 | 2件 | 半日 |
| **Phase 3** | Medium 主要修正 | 4件 | 1日 |
| **Phase 4** | Low + Info + テスト同期 | 5件 | 半日 |
| **Phase 5** | 回帰テスト拡充 | — | 1.5日 |
| **保留** | 追加検証が必要 | 3件 | — |
| **分離** | 別設計書に分離 | 1件 | — |

---

## 影響範囲分類一覧

| ID | 修正内容 | RT分類 | ISR影響 | ABI/Snapshot |
|----|----------|--------|---------|--------------|
| C-07 | IRConverter ジャンプ保護 | Non-RT（IR解析時） | なし | なし |
| C-01 | CustomInputOversampler プリフェッチ | RT-safe（Audio Thread内） | なし | なし |
| C-02 | NoiseShaper 量子化クランプ | RT-safe（Audio Thread内） | なし | なし |
| H-04 | OutputFilter HPF ナイキスト | Non-RT（prepare時） | なし | なし |
| H-11 | DeviceSettings タイマー | UI（Message Thread） | なし | なし |
| M-04 | SVF tan freq制限 | Non-RT（Coefficients計算時） | なし | なし |
| M-08 | CacheManager ハッシュ拡張 | Non-RT（ファイルロード時） | なし | なし |
| M-01 | IRAnalyzer make_unique | Non-RT（IR解析時） | なし | なし |
| L-03 | ISRRetireRouter アサーション | Non-RT | なし | なし |

---

## Phase 1: Critical 即時修正（3件）

### P1-1: C-07 `IRConverter.cpp` — ジャンプ保護が常に無効

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/IRConverter.cpp` |
| **該当行** | 149行目 |
| **修正** | 2行目の第1引数を `*currentIr` → `*ir` に変更 |
| **RT分類** | Non-RT（IR解析時） |

修正コード:
```cpp
// Before:
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*currentIr, result.scaleFactor);
// After:
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*ir, result.scaleFactor);
```

### P1-2: C-01 `CustomInputOversampler.cpp` — プリフェッチ範囲超過

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CustomInputOversampler.cpp` |
| **該当行** | 174行目 |
| **RT分類** | RT-safe（Audio Thread内） |

**境界変数の根拠:**
現行コードでは `n == convCount` であり、この前提では境界チェックは妥当。
- 関数シグネチャ（line 159）: `dotProductAvx2(const double* x, const double* coeffs, int n)`
- 呼出し元（line 496）: `dotProductAvx2(xWindow, stage.convCoeffsReversed.get(), stage.convCount)`
- aligned allocation / padding / sentinel の有無は現行コードに確認なし

**補足:** 今回の修正は安全性修正。性能最適化（prefetch距離のチューニング）とは別問題。

修正コード:
```cpp
if (i + 64 < n) {
    _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
}
```

### P1-3: C-02 `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` — 量子化オーバーフロー

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/Fixed15TapNoiseShaper.h`, `src/LatticeNoiseShaper.h` |
| **修正** | **順序変更は行わない**。量子化後にint16クランプを追加。 |
| **RT分類** | RT-safe（Audio Thread内） |

修正コード:
```cpp
const double q = std::round(v * invScale);
const double clamped = std::clamp(q, -32768.0, 32767.0);
return clamped * scale;
```

---

## Phase 1 ビルド＋テスト検証

```bash
cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Debug
cmake --build build --config Release
cd build && ctest -C Debug --output-on-failure
cd build && ctest -C Release --output-on-failure
```

---

## Phase 2: High 優先修正（2件）

### P2-1: H-04 `OutputFilter.cpp` — HPFのナイキストチェック欠落

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/OutputFilter.cpp` |
| **RT分類** | Non-RT（prepare時） |

修正コード:
```cpp
const double nyq = fs * 0.4999;
if (fc <= 0.0 || fc >= nyq || Q <= 0.0 || fs <= 0.0)
    return makeIdentity();
```

### P2-2: H-11 `DeviceSettings.cpp` — タイマー5Hzが編集中を上書き

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/DeviceSettings.cpp` |
| **RT分類** | UI（Message Thread） |
| **補足** | JUCE の `hasKeyboardFocus()` は IME 変換中の状態と異なる場合があるが、現状実装として十分。将来タスク: JUCE が IME 編集状態を取得可能か調査する。 |

修正コード:
```cpp
if (!inputHeadroomEditor.hasKeyboardFocus() &&
    std::abs(inputHeadroomEditor.getText().getDoubleValue() - currentInput) > 1.0e-6)
    inputHeadroomEditor.setText(juce::String(currentInput, 1), juce::dontSendNotification);
```

---

## Phase 3: Medium 主要修正（4件）

### P3-1: M-04 SVF `tan` 発散ガード + Nyquist Guard 定数統一

| 項目 | 内容 |
|------|------|
| **配置先** | `src/DspNumericPolicy.h`（既存DSP共通ヘッダ） |
| **RT分類** | Non-RT（Coefficients計算時） |

**既存定数との関係:**
- `DSP_MAX_FREQ_NYQUIST_RATIO = 0.95`（EQProcessor.h）— EQ周波数制限用（maxFreq = nyquist x 0.95）
- `kBiquadNyquistGuardRatio = 0.4999`（新規）— Biquadフィルタ安定性用（sin(w0) ≈ 0 回避）
- **目的が異なるため別定数が適切。名前で区別。**

追加定数:
```cpp
// DspNumericPolicy.h に追加
// Biquadフィルタのナイキスト近傍ガード用（sin(w0) ≈ 0 回避）
// DSP_MAX_FREQ_NYQUIST_RATIO (=0.95) はEQ周波数制限用で目的が異なる
static constexpr double kBiquadNyquistGuardRatio = 0.4999;
```

適用先:
- `src/OutputFilter.cpp:27` — `0.4999` → `kBiquadNyquistGuardRatio`
- `src/AllpassDesigner.cpp:35,62,315` — `0.499` → `kBiquadNyquistGuardRatio`
- `src/eqprocessor/EQProcessor.Coefficients.cpp` — freq制限に `kBiquadNyquistGuardRatio` 使用

### P3-2: M-08 キャッシュハッシュに path+size+mtime 追加

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/CacheManager.cpp`, `src/StateKey.h` |
| **RT分類** | Non-RT（ファイルロード時） |
| **将来改善候補** | ネットワークドライブ/コピーで mtime が維持されるケース対応。将来的に先頭数KBのハッシュまたは軽量CRC追加を検討。現段階では path+size+mtime で十分。 |

### P3-3: M-01 `noexcept` 内の `std::make_unique`（優先度ダウン）

| 項目 | 内容 |
|------|------|
| **状態** | **優先度ダウン** — 非RT関数。 |
| **RT分類** | Non-RT（IR解析時） |

### P3-4: L-03 null deleter アサーション

| 項目 | 内容 |
|------|------|
| **修正** | `assert(!(ptr!=nullptr && deleter==nullptr))` 追加 |
| **RT分類** | Non-RT |

---

## Phase 4: Low + Info + テスト同期（5件）

- **P4-1**: L-02 SIMD版異常値ハンドリング統一（RT-safe）
- **P4-2**: L-04 tmp ゼロ初期化（防御的プログラミング、RT-safe）
- **P4-3**: I-01 DSPCore::reset を `[[deprecated("次リリース削除予定")]]` 化（Non-RT）
- **P4-4**: I-02 `runtimeStore` 等の `std::atomic<DSPHandle>` メンバーに対して起動時 `assert(handle.is_lock_free())` を検証（主目的）。`static_assert(sizeof(DSPHandle)==16)` は補助条件（CMPXCHG16B前提の確認用）。
- **P4-5**: I-03 クランプ後のnet誤差ログ（Non-RT）

**I-02 補足:** `static_assert(is_always_lock_free)` はコンパイラ依存。起動時 `assert(is_lock_free())` が本質的な保証。`sizeof==16` は補助条件。

---

## Phase 5: 回帰テスト拡充（判定基準付き）

| テスト名 | 内容 | 判定基準 | テスト条件 | 優先度 |
|----------|------|----------|------------|--------|
| **Null Test** | 入力0→出力0 | `RMS(output) <= numericalNoiseFloor`（double: -220 dBFS相当、Bypass時: `1e-15`） | AutoGain OFF | 高 |
| **IR切替AB比較** | IR切替前後の出力が連続 | TruePeak < -60dB | — | 高 |
| **Peak Detector Validation** | TruePeakDetectorが仕様通り動作 | 既存UTパス + 追加ケース | — | 高 |
| **LUFS一致** | 出力LUFSが入力LUFSと整合 | AutoGain OFF: `abs(diff) < 0.2 dB` / AutoGain ON: `abs(diff) < 0.5 dB` | AutoGain ON/OFF分離 | 中 |
| **Release Build検証** | Debug/Release 両方で CTest 実行 | 全テストパス | Debug + Release | 高 |
| **Runtime Publish Stress Test** | 10000回連続IR切替 + 10000回EQ変更 | publish reject = 0, retire queue overflow = 0, deferred delete overflow = 0, leaked Runtime = 0, XRUN = 0, Crash = 0, Peak暴走なし | ISR設計の安定性確認（Epoch/Retire/Deferred Delete/Crossfade 検証） | 高 |

**検証環境:**
- Debug Build + CTest
- Release Build + CTest
- AVX2 パス + SSE パス（SIMD 両方）

---

## 保留（追加検証が必要 — 3件）

### R-1: M-10 OutputFilter fc 分岐
設計意図の可能性。確認なしで触るべきではない。

### R-2: M-14 BuildAnalysis失敗時のISR設計
ISR 設計で BuildAnalysis 失敗→Publish Reject の可能性。責務境界崩壊リスク。

### R-3: M-11/M-12 AutoGain推定誤差
意図的設計の可能性。追加検証必須。

---

## 分離（別設計書に分離 — 1件）

### S-1: M-05 大ブロック無音化 — アーキテクチャ変更
Convolver, Delay, Crossfade, Oversampling, Runtime Snapshot, Automation, Peak Meter, True Peak, Latency を含む設計変更。

---

## 要追加調査（判定確定）

| ID | 内容 | 判定 |
|----|------|------|
| C-05 | SafeStateSwapper UAF | 無効（競合パスなし） |
| H-03 | LatticeNoiseShaper クランプ遅延 | 無効（安全） |
| M-06 | MKL DFTI スケーリング | 無効（統一済み） |
| M-07 | IRConverter size_t overflow | 無効（修正済み） |
| I-01 | DSPCore::reset デッドコード | 確認（deprecated推奨、次リリース削除予定） |
| I-02 | DSPHandle lock-free | 確認（sizeof + 起動時assert） |

---

## リスク管理

| リスク | 確率 | 対策 |
|--------|------|------|
| C-07修正の回帰 | 低 | 既存UT + 回帰テスト |
| C-01ガードの性能影響 | 低 | 大規模IRでプロファイリング |
| C-02クランプのノイズ特性変化 | 低 | A/B比較測定 |

---

## 進捗管理

| フェーズ | 状態 | 開始日 | 完了日 | 担当 |
|----------|------|--------|--------|------|
| Phase 1 | 未着手 | | | |
| Phase 2 | 未着手 | | | |
| Phase 3 | 未着手 | | | |
| Phase 4 | 未着手 | | | |
| Phase 5 | 未着手 | | | |
| 保留 | 調査中 | | | |
| 分離 | 設計中 | | | |

---

## 参考資料

- `ConvoPeq_consolidated_bug_list_2026-07-18.md`
- `ConvoPeq_consolidated_bug_verification_2026-07-20.md`
- `ConvoPeq_repair_plan_validation_2026-07-20.md`
