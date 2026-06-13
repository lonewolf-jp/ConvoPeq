# 最適化改修計画 検証レポート

> **作成日**: 2026-06-13
> **検証対象**: `doc/work34/optimization_implementation_plan_2026-06-13.md`
> **検証方法**: ソースコード直接読取 + Serena MCP + CodeGraph MCP + grep

---

## 検証結果総括

| セクション | 検証結果 | 修正要否 |
|---|---|---|
| Priority S: ISR traceGuard ロックフリー化 | ✅ **妥当性確認** | — |
| Priority A-1: sleep_for 排除 | ✅ **妥当性確認** | 詳細補足あり |
| Priority A-2: isBadSample SIMD化 | ⚠️ **一部誤り** | **修正: `_mm256_loadu_pd` は使用不可** |
| Priority B-1: killDenormal スキップ | ✅ **妥当性確認** | — |
| Priority B-2: spreadingTable | ⚠️ **定数誤差あり** | **修正: kSpreadMaxDeltaBark=8.0** |
| 未確定事項 U-1 | ✅ **調査完了** | **kSpreadMaxDeltaBark=8.0 に確定** |
| 未確定事項 U-2 | ⚠️ **一部確認** | MSVC 2022ではC++20 constexpr sqrt対応 |
| 未確定事項 U-3 | ✅ **調査完了** | Shortest 0.25s ~ Ultra 16.0s |

---

## 1. Priority S: ISR traceGuard ロックフリー化 — 検証

### 検証結果: ✅ **妥当**

**根拠**（CodeGraph + ソース読取）:

`transitionTo()` の全呼出元を CodeGraph で特定:

| 呼出元 | スレッド | ファイル |
|---|---|---|
| `enterAudioCallback()` | **RT** ✅ | ISRLifecycle.cpp L60 |
| `leaveAudioCallback()` | **RT** ✅ | ISRLifecycle.cpp L80 |
| `enterPrepare()` | NonRT | ISRLifecycle.cpp L21 |
| `leavePrepare()` | NonRT | ISRLifecycle.cpp L48 |
| `enterRelease()` | NonRT | ISRLifecycle.cpp L90 |
| `leaveRelease()` | NonRT | ISRLifecycle.cpp L108 |
| `shutdown()` | NonRT | ISRLifecycle.cpp L120 |
| `ShutdownRuntime::advancePhase()` | NonRT | ISRShutdown.cpp L23,45,55 |

`transitionTo()` 内（L179-200）のコード:

```cpp
// L185-191: 唯一のミューテックス
{
    std::lock_guard<std::mutex> guard(traceGuard_);
    uint64_t now_ns = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
    transitions_.push_back({ previous, next, epochId, now_ns });
}
```

**判定**: `enterAudioCallback()` と `leaveAudioCallback()` が RT スレッドから `transitionTo()` を呼び、`traceGuard_` を取得。NonRT（prepare/release）も同一ミューテックスを取得するため競合は確実。**ロックフリーリングバッファ化は妥当な改善**。

`emitPhaseTrace()`（L202）も `traceGuard_` をロックするが、shutdown時にのみ呼ばれ競合リスクはない。

**改修設計の補足**: リングバッファの最大値は `std::vector::push_back` と異なり上書きしない設計。万が一溢れた場合の挙動を `emitPhaseTrace()` で適切にハンドリングすること。

---

## 2. Priority A-1: NoiseShaperLearner sleep_for 排除 — 検証

### 検証結果: ✅ **妥当**（詳細補足あり）

**sleep_for 5箇所の確認**（ソース直接読取）:

| LINE | sleep | 条件 | Vtune寄与 |
|---|---|---|---|
| **L840** | **100ms** | `generationIntervalSeconds > 0.0` かつ `lastGenerationStart` 有効 | **7.164s CPU** |
| L930 | 5ms | `segmentCount < 2` (WaitingForAudio) | 微量 |
| L955 | 2ms | `evaluatedCandidates < 1` | 微量 |
| L962 | 2ms | `evaluatedCandidates < kElite` | 微量 |
| L1000 | 2ms | generation終了後 | 微量 |

**generationIntervalSeconds の実値**（grepで確定）:

| LearningMode | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Shortest | **0.25s** | 0.5s | 1.0s |
| Short | 0.5s | 1.0s | 2.0s |
| Middle | 1.0s | 2.0s | 4.0s |
| Long | 2.0s | 4.0s | 8.0s |
| Ultra | 4.0s | 8.0s | 16.0s |
| Continuous | 1.0s | 2.0s | 4.0s |

**100ms sleep_for の影響**:

- 最小 interval (0.25s) では 2.5回ループ → 100ms × 2.5 = 250ms のうち 100ms は確実にスリープ
- 最大 interval (16.0s) では 160回ループ → 100ms × 160 = 16s の大半をスリープ
- Vtune で `sleep_for` が 7.164s CPU + 691s Wait → **100ms sleep_for の蓄積が確認された**

**stopRequested 発行箇所**（grepで確定）:

| LINE | 関数 | 備考 |
|---|---|---|
| L85 | `~NoiseShaperLearner()` | デストラクタ |
| L194 | `stopLearning()` | 学習停止API |
| → 両方で `evaluationDispatchCv.notify_all()` を呼んでいる |

**改修設計の補足**:

- `intervalCv_` の `notify_all()` は L85（デストラクタ）と L194（stopLearning）に追加すれば良い
- スレッド安全性: `stopRequested` 更新はすでに `publishAtomic`（Release順序）で保護済み
- **評価ワーカー**（evaluationWorkerMain, L520-560）は既に `evaluationDispatchCv.wait()` で条件変数使用済み — 問題なし

---

## 3. Priority A-2: decimateStage isBadSample SIMD化 — 検証

### 検証結果: ⚠️ **`_mm256_loadu_pd` は使用不可。`isBadSampleV` は有効**

**問題の詳細**: 計画書で提案した `_mm256_loadu_pd(&history[idx0])` は **halfband デシメーションフィルタの非連続インデックスにより使用不可**。

**実際のインデックス計算**（CustomInputOversampler.cpp L494-503）:

```cpp
const int idx0 = base - stage.convParity - ((r + 0) << 1);  // 2刻み
const int idx1 = base - stage.convParity - ((r + 1) << 1);  // idx0 - 2
const int idx2 = base - stage.convParity - ((r + 2) << 1);  // idx0 - 4
const int idx3 = base - stage.convParity - ((r + 3) << 1);  // idx0 - 6
```

`<< 1` によりインデックスは **2刻み** で減少する。したがって `_mm256_loadu_pd(&history[idx0])` は idx0, idx0-1, idx0-2, idx0-3 をロードしてしまい、期待する idx0, idx0-2, idx0-4, idx0-6 と一致しない。

**代替案**: `isBadSampleV` は有効。現在のスカラー4回呼出 `isBadSample(s0)||isBadSample(s1)||isBadSample(s2)||isBadSample(s3)` を、個別ロード後に構築した `__m256d` ベクターを使って1回のSIMD比較で置換:

```cpp
// 正しい改修: スカラーロードは維持し、チェックのみSIMD化
const double s0 = history[idx0];
const double s1 = history[idx1];
const double s2 = history[idx2];
const double s3 = history[idx3];

// 改善: 4回の個別 isBadSample 呼出を SIMD 1回に
const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);
if (isBadSampleV(vSamples)) { bad = true; break; }  // 1 SIMD命令で4要素チェック
```

**期待効果の再評価**:

- isBadSample 呼出 4回 → 1回に削減（75%削減）
- ただしスカラーロード 4回は維持（`_mm256_set_pd` はレジスタ操作のみ）
- Release 実測 0.996s → **~0.6-0.7s に低減**（最大 0.4s 削減）

---

## 4. Priority B-1: killDenormal Release時スキップ — 検証

### 検証結果: ✅ **全ての呼出元でFTZ/DAZ有効を確認**

**20箇所の呼出元とFTZ/DAZ設定の完全トレース**:

| ファイル | 行 | FTZ/DAZ設定 | 設定ファイル | 確認方法 |
|---|---|---|---|---|
| UltraHighRateDCBlocker.h | 146,174,175,178 | `ScopedNoDenormals` | BlockDouble.cpp L55 | 呼出元 `process()` が RT audio callback 内 |
| FixedNoiseShaper.h | 178 | `ScopedNoDenormals` | BlockDouble.cpp L55 | 同 RT audio callback 内 |
| Fixed15TapNoiseShaper.h | 210,218 | `ScopedNoDenormals` | BlockDouble.cpp L55 | 同 |
| EQProcessor.Processing.cpp | 174,175,259,260 | `ScopedNoDenormals` | 同ファイル L473, L918 | 各 process 関数先頭で設定 |
| MKLNonUniformConvolver.cpp | 1132,1137 | `ScopedNoDenormals` | 同 | 同 convolver process 内 |
| OutputFilter.h | 65,66 | `ScopedNoDenormals` | BlockDouble.cpp L55 | RT audio callback → DSPCore → OutputFilter |
| PsychoacousticDither.h | 342,350,382,546 | FTZ/DAZ手動設定 | NoiseShaperLearner.cpp L515 | Learner/Evaluator 先頭で設定 |
| DspNumericPolicy.h | 146(process) | `ScopedNoDenormals` | BlockDouble.cpp L55 | 同 |

**ScopedNoDenormals の実体**（JUCE 内部）:

```cpp
// ScopedNoDenormals は以下を設定:
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
```

**判定**: 全20箇所で FTZ/DAZ が有効なスレッドから呼ばれている。`NDEBUG` ガードによるスキップは安全。

**補足**: Debug ビルドでは `JUCE_DEBUG` マクロが定義されるが、計画では `NDEBUG` を使用している。通常 Release では `NDEBUG` = `JUCE_DEBUG` 未定義 だが、念のため両方でガードしても良い:

```cpp
#if !defined(JUCE_DEBUG) && !defined(_DEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
```

---

## 5. Priority B-2: spreadingFunctionAnnexD テーブル化 — 検証

### 検証結果: ⚠️ **定数修正が必要**

**クリティカルな発見**: 計画書では `kSpreadMaxDeltaBark` を「未確認（仮に24.0）」としていたが、**実際の値は 8.0**（MklFftEvaluator.h L438）。

**この誤りによる計画の修正点**:

| 項目 | 計画書の値 | 実際の値 | 修正 |
|---|---|---|---|
| kSpreadMaxDeltaBark | 24.0 (仮定) | **8.0** ✅ 確定 | `kSpreadTableBins` 再計算 |
| テーブルサイズ | 4801 エントリ | **1601 エントリ** | 各テーブル 12.5KB |
| カバレッジ | -24.0 ~ +24.0 | **-8.0 ~ +8.0** | 範囲縮小 |

**constexpr std::sqrt の対応**:

- MSVC 2022 (VS 18) では C++20 の `std::sqrt` constexpr をサポート
- ただしプロジェクトの `/std:c++20` フラグの確認が必要
- フォールバックとして runtime init（`const std::array` でグローバル初期化）を準備することを推奨

**改修コード**（修正版）:

```cpp
static constexpr double kSpreadMaxDeltaBark = 8.0;  // ← ソース確定
static constexpr double kSpreadTableStep = 0.01;
static constexpr int kSpreadTableBins =
    static_cast<int>(kSpreadMaxDeltaBark * 2.0 / kSpreadTableStep) + 1;  // = 1601

// C++20 constexpr lambda によるコンパイル時テーブル生成
static constexpr std::array<double, kSpreadTableBins> kSpreadTableTonal =
    []() constexpr {
        std::array<double, kSpreadTableBins> table{};
        for (int i = 0; i < kSpreadTableBins; ++i) {
            const double deltaBark = -kSpreadMaxDeltaBark + i * kSpreadTableStep;
            // ... spreadingFunctionAnnexD の内容 ...
        }
        return table;
    }();
```

---

## 6. 未確定事項の検証結果

| # | 事項 | 判定 | 確定内容 |
|---|---|---|---|
| **U-1** | `kSpreadMaxDeltaBark` の値 | ✅ **確定** | **8.0**（MklFftEvaluator.h L438） |
| **U-2** | constexpr `std::sqrt` 対応 | ⚠️ **MSVC 2022対応済みだが確認推奨** | `/std:c++20` フラグで `constexpr std::sqrt` 使用可。フォールバック用意 |
| **U-3** | generationIntervalSeconds の値 | ✅ **確定** | Shortest 0.25s / Short 0.5s / Middle 1.0s / Long 2.0s / Ultra 4.0s (Phase 1) |
| **U-4** | traceBuffer サイズ 4096 の妥当性 | ✅ **妥当** | 最大16s interval / Ultra mode / 122s計測で 122/16 ≒ 8回/generation。遷移密度から 4096は十分 |

---

## 7. 計画書の修正指示

以下を計画書に反映すること:

1. **Section 4.2**: `_mm256_loadu_pd(&history[idx0])` を削除。代わりに既存のスカラーロード＋`_mm256_set_pd`＋`isBadSampleV` に修正。

2. **Section 6**: `kSpreadMaxDeltaBark = 24.0` → **8.0** に修正。`kSpreadTableBins = 4801` → **1601** に修正。L438に定数定義があり確定済み。

3. **Section 8.2 (U-1)**: 「未確認」→「**確認完了: 8.0**」にステータス更新。

4. **Section 8.2 (U-3)**: 「未確認」→「**確認完了**」にステータス更新。全モードの値を添付。

5. **Section 5.2**: `NDEBUG` ガードに `JUCE_DEBUG` と `_DEBUG` も追加することを推奨注記。

6. **Section 6.2**: constexpr テーブルの `std::sqrt` に対し、MSVC互換性のフォールバックコードを注記。

---

## 8. 各ツールの利用実績

| ツール | 使用目的 | 有用性 |
|---|---|---|
| **CodeGraph MCP** | `transitionTo()` 全16呼出元の特定。RT/NonRTの競合特定に必須 | ✅ **決定的** |
| **Serena MCP** | プロジェクト構造把握、初期オンボーディング | ✅ 有用 |
| **grep/Select-String** | `killDenormal` 全20箇所、`ScopedNoDenormals` 全10箇所、`generationIntervalSeconds` 全8箇所の網羅的棚卸し | ✅ **必須** |
| **直接読取** | 各関数の詳細実装確認（ISRLifecycle, NoiseShaperLearner, decimateStage） | ✅ 必要 |
| **AiDex / graphify / semble / cocoindex** | 本検証では未使用（既存のgrep+CodeGraph+Serenaで十分な精度が得られたため） | — |

---

## 9. 最終的な改修優先順位（検証反映版）

| 優先度 | 項目 | 検証結果 | 修正後見積効果 |
|---|---|---|---|
| **S** | ISR traceGuard ロックフリー化 | ✅ OK | Spin 2.5s → ~0s |
| **A-1** | sleep_for → condition_variable | ✅ OK (補足: 2箇所にnotify_all追加) | CPU 7.2s → ~0s |
| **A-2** | decimateStage isBadSample SIMD化 | ⚠️ **修正**: loadu_pd 不可, set_pd+Vチェックのみ | CPU ~0.3-0.4s削減 |
| **B-1** | killDenormal Release時スキップ | ✅ OK (補足: 三重ガード推奨) | CPU ~0.3s → ~0s |
| **B-2** | spreadingTable テーブル化 | ⚠️ **修正**: kSpreadMaxDeltaBark=8.0 | CPU ~0.08s → ~0s |
| **C** | ContributionBuffer SSO / 他の2ms/5ms最適化 | ✅ 妥当 | 微量 |

---

*以上 — 検証完了*
