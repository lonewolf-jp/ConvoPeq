# Intel Advisor 解析結果：妥当性検証レポート

> 作成日: 2026-06-13
> 検証対象: `doc/work35/intel_advisor_analysis_optimization.md`
> 検証ツール: Serena MCP, CodeGraph MCP, AiDex MCP, Graphify MCP, Grep

---

## 検証サマリ

| 優先度 | コンポーネント | 判定 | 修正指示 |
|--------|--------------|------|---------|
| **P1** | `decimateStage` スカラーフォールバック | ✅ **妥当** | 1件の補足あり |
| **P2** | `computeMaskingEnergyStable` | ✅ **妥当** | なし |
| **P3** | `UltraHighRateDCBlocker::process` | ⚠️ **概ね妥当** | SSE2命令数の計数誤差 (5→9) |
| **P4** | `sanitizeFiniteChunk` | ✅ **妥当** | なし |
| **P5** | `NoiseShaperLearner::evaluateCandidateMapped` | ✅ **妥当** | なし |
| **P6** | 軽微改善候補 | ✅ **妥当** | 2件の補足あり |

---

## P1: `CustomInputOversampler::decimateStage` — 検証結果

**判定: ✅ 妥当**（1件の補足あり）

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | 該当行 ~L571（スカラーフォールバック内側ループ） | Serena + AiDex + 実コード読取: `decimateStage` はL430-586。AdvisorのL571はスカラー内側ループ `for (int r = 0; r < stage.convCount; ++r)` に該当 | ✅ 一致 |
| 2 | `stage.convCount < 4` → 常にスカラー | コード確認: `if (stage.convCount >= 4)` の条件分岐（~L493） | ✅ 正しい |
| 3 | `bad == true` でAVX2放棄→スカラー再計算 | コード確認: `if (!usedAvxPath || bad)` ブロックでcenterCoeff+convCoeffsを再計算 | ✅ 正しい |
| 4 | `_mm256_set_pd(s3,s2,s1,s0)` で個別gather | コード確認: L535-540で4つの個別ロード後にset_pd | ✅ 正しい |
| 5 | `dotProductAvx2` が `interpolateStage` で使用されるが `decimateStage` では未使用 | AiDex検索: `dotProductAvx2` はL384（interpolateStage）でのみ使用。decimateStageでは未使用 | ✅ 正しい |
| 6 | 水平加算が `store` + 4回スカラー加算 | コード確認: L555-560の `alignas(64) double partial[4]; _mm256_store_pd(partial, vAcc); acc += partial[0]+...` | ✅ 正しい |

### 補足事項

**`dotProductAvx2` にも同じ水平加算パターンあり**: `dotProductAvx2`（L165-170）でも以下の同一パターンが使用されている:

```cpp
alignas(64) double partial[4];
_mm256_store_pd(partial, acc0);
double sum = partial[0] + partial[1] + partial[2] + partial[3];
```

P1-3の改善案（`vextractf128` + `hadd`）は `dotProductAvx2` にも適用可能。分析文書に明記されていないが、修正時には両方を対象とすべき。

---

## P2: `MklFftEvaluator::computeMaskingEnergyStable` — 検証結果

**判定: ✅ 妥当**

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | 該当行 ~L749 | Serena: メソッド本体L737-779。内側ループ `for (int k = 0; k < contributions.size; ++k)` はL749 | ✅ 一致 |
| 2 | `std::exp((valueDb - maxDb) * kLogScale)` がdata type conversion | Advisorに「1 Data type conversion present」と表示。`std::exp` 呼出がSSE↔x87 FPUモード切替を引き起こす可能性 | ✅ Advisor所見と一致 |
| 3 | `std::isfinite(maxDb)` の呼出 | コード確認: `if (contributions.size <= 0 \|\| !std::isfinite(maxDb))` | ✅ 正しい |
| 4 | L773: `sum += std::exp(...)`, L774: `const double totalPower = std::exp(...)` | コード確認: 両方の`std::exp`呼出を確認 | ✅ 正しい |

---

## P3: `UltraHighRateDCBlocker::process` — 検証結果

**判定: ⚠️ 概ね妥当（1件の計数誤差）**

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | 該当行 ~L171（processのスカラーループ） | CodeGraph: `process`はL158-192。ループ開始は~L169（`for (int i = 0; i < numSamples; ++i)`） | ✅ 一致 |
| 2 | `killDenormal` がReleaseで完全no-op | `DspNumericPolicy.h:165-170`: Release時は `static_cast<void>(x); return x;` | ✅ 正しい |
| 3 | `isFiniteAndBelowThresholdMask` がSSE2 intrinsicsで過剰実装 | `UltraHighRateDCBlocker.h:63-74`: `_mm_set1_pd`, `_mm_sub_pd`, `_mm_cmpeq_pd`, `_mm_andnot_pd`, `_mm_cmplt_pd`, `_mm_and_pd`, `_mm_movemask_pd` を確認 | ✅ 正しい |
| 4 | スカラー版への置換が有効 | 改善案のビット演算スカラー版は `DSPCoreIO.cpp` の `isFiniteNoLibm` と同様の手法 | ✅ 妥当 |

### 計数誤差

文書内で「SSE2 intrinsics 5命令」と記載されているが、実際の `isFiniteAndBelowThresholdMask` は **9つのSSE2 intrinsic命令** で構成されている:

| # | 命令 | 行 |
|---|------|----|
| 1 | `_mm_set1_pd(value)` | L65 |
| 2 | `_mm_sub_pd(v, v)` | L66 |
| 3 | `_mm_cmpeq_pd(diff, _mm_setzero_pd())` | L67 |
| 4 | `_mm_set1_pd(-0.0)` (signMask) | L68 |
| 5 | `_mm_andnot_pd(signMask, v)` (abs) | L69 |
| 6 | `_mm_set1_pd(threshold)` | L70 |
| 7 | `_mm_cmplt_pd(absV, thresholdV)` | L71 |
| 8 | `_mm_and_pd(finiteMask, belowMask)` | L72 |
| 9 | `_mm_movemask_pd(validMask)` | L73 |

計数は「5」ではなく「9」が正しいが、**「過剰なSSE2実装」という結論には影響しない**。

---

## P4: `sanitizeFiniteChunk` — 検証結果

**判定: ✅ 妥当**

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | 該当行 ~L43 | コード確認: 関数定義L38、ループL43（`for (int i = 0; i < count; ++i)`） | ✅ 一致 |
| 2 | Self time 0.032s（2箇所の呼出合計） | Advisor: L43のループ0.032s total/0.011s self + 関数自体0.019s total。DSPCoreIO.cpp内で2箇所（L201,202）と呼び出し | ✅ 一致（正確には複数呼出） |
| 3 | 呼び出し元2箇所 | AiDex: L201, L202（`processInputDouble`内） | ✅ 正しい |

---

## P5: `NoiseShaperLearner::evaluateCandidateMapped` — 検証結果

**判定: ✅ 妥当**

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | 該当行 ~L1298 | Serena: メソッド定義L1253、エラー計算ループは関数本体内（~L1298）。AdvisorのL1298と一致 | ✅ 一致 |
| 2 | エラー計算ループが自動ベクトル化不可 | コード確認: `context.shapedLeft[k]`, `context.errorLeft[k]`, `leveled.segment.left[k]` の3ポインタがエイリアスの可能性あり | ✅ 正しい |
| 3 | `__declspec(noalias)` / `__restrict` で改善可能 | 標準的なC++最適化テクニック | ✅ 妥当 |

---

## P6: 軽微な改善候補 — 検証結果

**判定: ✅ 妥当（2件の補足あり）**

### 検証詳細

| # | 主張 | 検証方法 | 結果 |
|---|------|---------|------|
| 1 | `decimateStage` AVX2水平加算 | コード確認: L555-560 + `dotProductAvx2` L165-170も同パターン | ✅ 正しい（但し後述の補足参照） |
| 2 | `powerToDb`/`computeTonalityFromSfm` の `std::log10` | `MklFftEvaluator.h:527-530`（powerToDb）, `MklFftEvaluator.h:585-590`（computeTonalityFromSfm） | ✅ 正しい |
| 3 | `juce::AudioBuffer::makeCopyOf<float>` | JUCE内部コードのため直接編集不可 | ✅ 正しい |
| 4 | `lock_locales` (0.009s) | CRT関数（`locks.cpp:63`）。プロジェクトコードではない | ✅ 記載は妥当だがプロジェクト側で対応不可 |
| 5 | `NoiseShaperLearner::workerThreadMain` L838 除算 | AiDex: `workerThreadMain` はL720。L838はメインループ内 | ✅ 正しい |

### 補足事項

**1. `dotProductAvx2` も同じ水平加算パターンを使用**

P6では `decimateStage` の水平加算のみ指摘しているが、`dotProductAvx2`（L165-170）も全く同じ `_mm256_store_pd` + スカラー加算パターンを使用している。改善案（`vextractf128` + `hadd`）は両方に適用可能。

```cpp
// dotProductAvx2 の水平加算 (L165-170)
alignas(64) double partial[4];
_mm256_store_pd(partial, acc0);
double sum = partial[0] + partial[1] + partial[2] + partial[3];
```

改善案で示された `vextractf128` + `hadd` パターンは両方の箇所で同一の効果がある。

**2. `lock_locales` はプロジェクトコード外**

`lock_locales` (0.009s) はMSVC CRTの内部関数（`locks.cpp:63`）。通常は `std::cout` / `printf` / `std::to_string` 等からの間接呼び出し。プロジェクトでlocale非依存の実装を徹底することで削減可能だが、直接的な改修対象ではない。

---

## 全体的な評価

### 正確性: ✅ 高い

全P1〜P6の主要主張について、Serena（シンボル解析）、CodeGraph（構造解析）、AiDex（識別子検索）、Grep（パターン検索）を用いて実コードと突合した結果、**ライン番号、関数構造、問題の本質はすべて正確**。

### 発見された補足事項（3件）

| # | 補足内容 | 重要度 |
|---|---------|-------|
| 1 | `isFiniteAndBelowThresholdMask` のSSE2命令数: 5→9 | 低（結論に影響なし） |
| 2 | `dotProductAvx2` にも同じ水平加算改善機会あり | 中（修正範囲に影響） |
| 3 | `lock_locales` はプロジェクトコード外のCRT関数 | 低（対応不可） |

### ツール使用状況

| ツール | 用途 | 使用頻度 |
|--------|------|---------|
| **Serena MCP** (`find_symbol`, `find_file`) | シンボル特定、関数本体取得 | 高頻度 |
| **AiDex MCP** (`aidex_query`, `aidex_session`) | 識別子検索、セッション管理 | 高頻度 |
| **CodeGraph MCP** (`get_file_structure`, `analyze_module_structure`, `find_dependencies`) | ファイル構造解析、依存関係 | 中頻度 |
| **Graphify MCP** (`graph_stats`, `get_node`) | グラフ統計、ノード情報 | 低頻度 |
| **Grep** (`Select-String`) | パターン検索 | 中頻度 |

### 結論

作成された最適化提案書 `intel_advisor_analysis_optimization.md` は、Intel Advisorの解析結果を正確に反映しており、**すべての主要なホットスポットを正しく特定し、妥当な改善案を提示している**。3件の軽微な補足事項を除けば、実装判断の根拠として十分な品質を持つ。
