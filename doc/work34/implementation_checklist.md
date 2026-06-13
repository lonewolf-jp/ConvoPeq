# 最適化実装チェックリスト

> **作成日**: 2026-06-13
> **ベース**: doc/work34/optimization_implementation_plan_2026-06-13.md

---

## Phase 1: 即効性＋低リスク

### [S] ISR traceGuard ロックフリー化

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| S-1 | `ISRLifecycle.h`: `traceGuard_` + `vector` → `array<PhaseTransition,4096>` + `atomic<size_t>` | ISRLifecycle.h | � 完了 | ✅ |
| S-2 | `ISRLifecycle.cpp`: `transitionTo()` 内のロック→fetch_add リングバッファ書込 | ISRLifecycle.cpp | 🟢 完了 | ✅ |
| S-3 | `ISRLifecycle.cpp`: `emitPhaseTrace()` の読出しをリングバッファ対応 | ISRLifecycle.cpp | 🟢 完了 | ✅ |
| S-4 | `ISRLifecycle.h`: `traceGuard_` + `vector` 削除、`<vector>` include削除 | ISRLifecycle.h | 🟢 完了 | ✅ |
| S-5 | Debugビルド確認（既存警告のみで新規エラーゼロ） | — | 🟢 完了 | ✅ |

### [A-1] NoiseShaperLearner sleep_for 排除

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| A1-1 | `NoiseShaperLearner.h`: `intervalMutex_` + `intervalCv_` 追加 | NoiseShaperLearner.h | 🟢 完了 | ✅ |
| A1-2 | `NoiseShaperLearner.cpp`: sleep_for(100ms) → intervalCv_.wait_until | NoiseShaperLearner.cpp | 🟢 完了 | ✅ |
| A1-3 | デストラクタに intervalCv_.notify_all() 追加 | NoiseShaperLearner.cpp | 🟢 完了 | ✅ |
| A1-4 | stopLearning() に intervalCv_.notify_all() 追加 | NoiseShaperLearner.cpp | 🟢 完了 | ✅ |
| A1-5 | Debugビルド確認（新規エラーゼロ） | — | 🟢 完了 | ✅ |

### [B-1] killDenormal Release時スキップ

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| B1-1 | `killDenormal(double)` に三重ガード追加 | DspNumericPolicy.h | 🟢 完了 | ✅ |
| B1-2 | `killDenormal(float)` に三重ガード追加 | DspNumericPolicy.h | 🟢 完了 | ✅ |
| B1-3 | `killDenormalV(__m256d)` に三重ガード追加 | DspNumericPolicy.h | 🟢 完了 | ✅ |
| B1-4 | `killDenormalV(__m128d)` に三重ガード追加 | DspNumericPolicy.h | 🟢 完了 | ✅ |
| B1-5 | Debug + Release両ビルド確認（新規エラーゼロ） | — | 🟢 完了 | ✅ |

---

## Phase 2: 中程度の工数

### [A-2] decimateStage isBadSample SIMD化

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| A2-1 | `CustomInputOversampler.cpp`: `isBadSampleV(__m256d)` 関数追加 | CustomInputOversampler.cpp | � 完了 | ✅ |
| A2-2 | AVX2パス: isBadSample×4 → isBadSampleV×1 + _mm256_set_pd に置換 | CustomInputOversampler.cpp | 🟢 完了 | ✅ |
| A2-3 | Debugビルド確認（新規エラーゼロ） | — | 🟢 完了 | ✅ |

### [B-2] spreadingFunctionAnnexD テーブル化

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| B2-1 | 定数定義再配置（table前に移動）+ テーブルパラメータ追加 | MklFftEvaluator.h | 🟢 完了 | ✅ |
| B2-2 | `kSpreadTableTonal` + `kSpreadTableNoise` (inline static const) 追加 | MklFftEvaluator.h | 🟢 完了 | ✅ |
| B2-3 | spreadingFunctionAnnexD をテーブルルックアップ化 | MklFftEvaluator.h | 🟢 完了 | ✅ |
| B2-4 | Debugビルド確認（新規エラーゼロ） | — | 🟢 完了 | ✅ |

---

## Phase 3: 二次的最適化

### [C-1] ContributionBuffer SSO最適化

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| C1-1 | ContributionBuffer: 最初32エントリをスタック→超過時のみヒープ | MklFftEvaluator.h | 🔴 未着手 | |

### [C-2] decimateStage partial-SIMDパス

| # | タスク | ファイル | 状況 | 検証 |
|---|---|---|---|---|
| C2-1 | AVX2 bad検出後、該当4要素のみ再計算するパス追加 | CustomInputOversampler.cpp | 🔴 未着手 | |

---

## 凡例

- 🔴 未着手
- 🟡 作業中
- 🟢 完了
- ✅ 検証済み
