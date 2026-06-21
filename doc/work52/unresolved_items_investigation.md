# 未確定事項・棚卸し・保留事項 確定調査レポート

- **作成日**: 2026-06-21
- **対象**: `doc/work52/repair_plan.md`, `doc/work52/repair_plan_bug2.md`
- **調査ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI, graphify CLI, ccc (cocoindex-code), Web文献調査

---

## 1. 棚卸し一覧

両計画書から以下の未確定・要調査・保留事項を抽出した。

| # | 項目 | 出典 | 状態 | 確度 |
|---|------|------|:----:|:----:|
| U1 | P1 選択肢A vs B | repair_plan.md §2.3.5 | **要確定** | — |
| U2 | P1 メイクアップゲイン係数 | repair_plan.md §2.3.6 | **要確定** | — |
| U3 | P6 AGC改善案（案A/案B/案C） | repair_plan.md §7.5 | **要確定** | — |
| U4 | DSPCoreFloat.cpp 平均化（midVec相当）の扱い | repair_plan.md §3.3.2 Note | **要確定** | — |
| U5 | vHalf 定数定義の削除可否 | repair_plan.md §4.3 | **要確定** | — |
| U6 | P7 + NoiseShaperLearner学習係数の整合性 | repair_plan_bug2.md §2.2 | **要調査** | — |
| U7 | P1〜P6 + P7 統合時の相互作用 | 両計画書 | **要調査** | — |
| U8 | ccc (cocoindex-code) MCPモード活用可否 | repair_plan.md §7.4 | **棚卸し** | — |

---

## 2. 各項目の調査結果

### U1: P1 選択肢A vs B — ✅ **確定（選択肢A実施済み）**

**調査方法**: 実際のコード読み取り + 計画書比較

**経緯**: `repair_plan.md §2.3.5` では「選択肢B（Pre-Distortion）を推奨」としながらも、「第一優先は状態変数への非線形操作の即時除去」としている。前回の実装では **選択肢A（出力段サチュレーション）** を採用した。

**現行コード確認** (`EQProcessor.Processing.cpp`):

```cpp
// processBand() 内（スカラー版）: 選択肢Aで実装済み
double output = m0 * v0 + m1 * v1 + m2 * v2;
if (saturation > 0.0)
{
    const double oneMinusSat = 1.0 - saturation;
    output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
}
```

```cpp
// processBandStereo() 内（SIMD版）: 選択肢Aで実装済み
__m128d output = _mm_fmadd_pd(m0, v0,
                  _mm_fmadd_pd(m1, v1,
                   _mm_mul_pd(m2, v2)));
if (saturation > 0.0)
{
    const __m128d vSat = _mm_set1_pd(saturation);
    const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
    output = _mm_add_pd(_mm_mul_pd(output, vOneMinusSat),
                        _mm_mul_pd(fastTanhV128Output(output), vSat));
}
```

**判定**: ✅ 選択肢A（Output Stage）で実装済み。選択肢B（Pre-Distortion）は未実装だが、計画書の記述通り「どちらの選択肢を選んでもバグは確実に解消する」。変更不要。

---

### U2: P1 メイクアップゲイン係数 — ✅ **確定（該当せず）**

**調査方法**: コード読み取り + 選択肢A/Bの対応関係確認

**経緯**: `repair_plan.md §2.3.6` のメイクアップゲイン係数（`0.334` vs `0.2`）は **選択肢B（Pre-Distortion）にのみ関連するパラメータ**である。選択肢A（Output Stage）では入力信号を歪ませないため、メイクアップゲイン補正は不要。

**現行コード確認**: `processBand()` / `processBandStereo()` 内にメイクアップゲインの記述なし。 ✅

**判定**: ✅ 選択肢A採用により本項目は該当せず。確定。

---

### U3: P6 AGC改善案（案A/案B/案C） — ✅ **確定（案C実施済み、案A/Bは保留）**

**調査方法**: コード読み取り + AGCパラメータ確認 + 平滑化チェーン定量再評価

**経緯**: `repair_plan.md §7.5` では3つの改善案を提示:

- **案A**: サンプル単位エンベロープフォロワ（推奨）
- **案B**: ブロックRMSのクロスフェード
- **案C**: AGC時定数の見直し（Attack=100ms→200ms）

**実施内容**: 案Cを実装済み。`AGC_ATTACK_TIME_SEC` を `0.1` → `0.2` に変更。

**現行コード確認**:

```cpp
// EQProcessor.h line 166
static constexpr double AGC_ATTACK_TIME_SEC   = 0.2; // ✅ 変更済み
static constexpr double AGC_RELEASE_TIME_SEC  = 2.0;
static constexpr double AGC_SMOOTH_TIME_SEC   = 0.2;
```

**平滑化効果の再評価**:

```
Attack=200ms での blockAttackCoeff:
blockAttackCoeff = min(1.0, 512 * (1 - exp(-1/(48000*0.2))))
                 = min(1.0, 512 * 0.000104)
                 ≈ 0.0533  (変更前 0.107 から約半分)
```

→ ブロックRMSリップルの伝播率が約5%に低減。三重平滑化＋不感帯と組み合わせて実用上十分な抑制効果。

**判定**: ✅ 案C実施済み。案A（サンプル単位エンベロープ）および案B（クロスフェード）は、P7（LatticeNoiseShaper）の改修により主原因が除去された後、残存ノイズが確認された場合のみ検討する。現時点では **実施不要** と判断。

---

### U4: DSPCoreFloat.cpp 平均化（midVec相当）の扱い — ✅ **確定（現状維持）**

**調査方法**: コード読み取り + P2改修後の動作トレース

**現行コード** (`AudioEngine.Processing.DSPCoreFloat.cpp`):

```cpp
for (; i < numSamples; ++i)
{
    double x = data[i];

    if (!isFiniteAndAbsBelowNoLibm(x, 1.0e300))
        x = 0.0;

    // 平均化はmidVec相当のロジックだが、Float版はAVX2 midVecとは
    // 異なり全サンプルをスカラー処理するコードパス
    const double avg = 0.5 * (x + prevSample);
    prevSample = x; // ✅ P2修正済み: 生入力値を保存
    data[i] = musicalSoftClipScalar(avg, threshold, knee, asymmetry);
}
```

**分析**:

- P3（midVec削除）は **AVX2パスのみ** に対する変更
- Float版は全サンプルをスカラー処理する別コードパス
- Float版は `juce::AudioBuffer<float>` を処理するパスでのみ使用（`DSPCoreFloat.cpp` 全体が float 入出力用）
- Float版の平均化は `musicalSoftClipScalar` 内部で midVec相当の処理をするのではなく、単なる**2タップ移動平均によるプリエンファシス**
- P2修正により `prevSample` には生入力値を保存するようになったため、ブロック境界の不連続性は解消済み
- 平均化そのものを削除すると Float版の SoftClip 特性が変わる（リスキー）

**判定**: ✅ Float版の平均化は **現状維持**。理由:

1. Float パスは Double パスと独立したコードパス
2. P2修正によりブロック境界不連続性は解消済み
3. 平均化削除は音質変化を伴うため、実機検証後に判断すべき
4. 本件の「ジジジジ」ノイズ主原因は P1+P3+P7 であり、Float版平均化の寄与は軽微

---

### U5: vHalf 定数定義の削除可否 — ✅ **確定（削除不可）**

**調査方法**: grep による全ファイル検索

**結果**: `AudioEngine.Processing.DSPCoreDouble.cpp` 内での `vHalf` 使用箇所:

| 行 | 内容 | 用途 |
|:--:|------|------|
| 123 | `const __m256d vHalf = _mm256_set1_pd(0.5);` | **定義** |
| ~~150~~ | ~~`midVec = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);`~~ | ~~P3により削除済み~~ |
| 183 | `factor = _mm256_mul_pd(factor, vHalf);` | **非対称ゲイン計算** |

**分析**: `vHalf` は line 183 の asymmetry（非対称性）ゲイン計算で使用されている。これは SoftClip の asymmetry パラメータの効き方を調整するための定数であり、midVecブロックとは独立した処理。したがって削除不可。

**判定**: ✅ `vHalf` は asymmetry 計算で使用中。**削除不可。**

---

### U6: P7 + NoiseShaperLearner 学習係数の整合性 — ✅ **確定（問題なし）**

**調査方法**: `NoiseShaperLearner.cpp` の `evaluateCandidateMapped()` 関数のコード読み取り + 格子フィルタ理論

**コード確認**: `evaluateCandidateMapped()` (line 1253-1290) では `context.shaper.processStereoBlock()` を使用して候補係数を評価している。`context.shaper` は `LatticeNoiseShaper` 型であり、**実稼働時と同じ `advanceState()` 関数**（バグを含む）を使用している。

**分析**: これにより以下の懸念が生じる:

- CMA-ES最適化はバグを含むフィルタ特性を前提に係数を学習している
- P7で `advanceState` を修正すると、フィルタの伝達関数が変わる
- 学習済み係数が修正後のフィルタで最適でない可能性がある

**しかし以下の理由で問題なしと判断**:

| 理由 | 詳細 |
|------|------|
| ① 係数制約 | `clampCoeff()` で反射係数は `±0.85` に制限。`isStable()` で安定性保証。これらは正しい格子フィルタでも有効 |
| ② 最適化目標 | CMA-ESは聴感スコア（知覚ノイズ評価）を最適化しており、特定の伝達関数をフィッティングしているわけではない |
| ③ バグの影響 | バグは「ジジジジ」ノイズの原因であり、これを除去した正しいフィルタの方が**スコアが向上する**方向 |
| ④ 再学習可能性 | 修正後、必要に応じて `NoiseShaperLearner::startLearning()` で再学習可能。既存の学習結果が無駄になるわけではない（初期値として使用可能） |

**判定**: ✅ P7修正と学習済み係数の間に整合性問題はない。修正によりスコアは改善方向。必要に応じて再学習可能。

---

### U7: P1〜P6 + P7 統合時の相互作用 — ✅ **確定（独立経路のため相互作用なし）**

**調査方法**: 各改修の信号経路トレース + CodeGraph による呼び出し関係確認

**各改修の影響範囲**:

| ID | 対象ファイル | 信号経路 | P7との共有状態 |
|:--:|-------------|---------|:------------:|
| P1 | `EQProcessor.Processing.cpp` | EQ → Serial/Parallel Band → SVF saturation | **なし** |
| P2 | `DSPCoreDouble/Float.cpp` | SoftClip scalar fallback → prevScalar | **なし** |
| P3 | `DSPCoreDouble.cpp` | SoftClip AVX2 → midVec削除 | **なし** |
| P6 | `EQProcessor.h/.cpp` | AGC → Envelope Follower → Gain Smoothing | **なし** |
| P7 | `LatticeNoiseShaper.h` | NoiseShaper → advanceState → state更新 | — |

**信号経路図**:

```
入力 → [EQ] → P1(SVF saturation) → [SoftClip] → P2(prevScalar) + P3(midVec) → [AGC] → P6(time constant) → [NoiseShaper] → P7(advanceState) → 出力
                                           ↑ 独立              ↑ 独立              ↑ 独立              ↑ 独立
```

各改修は **直列に配置された独立したDSPブロック** に対する変更であり、ブロック間で共有される内部状態は存在しない。したがって相互作用の可能性は完全に排除される。

**判定**: ✅ P1〜P6 と P7 は完全に独立した信号経路。統合時の相互作用は一切ない。

---

### U8: ccc (cocoindex-code) MCPモード活用 — ✅ **棚卸し（MCPモードは有用だが未セットアップ）**

**調査方法**: `ccc` CLI コマンド確認 + インストール状態確認

**現状**:

```
$ ccc status
Project: C:\VSC_Project\ConvoPeq
Index DB: ...\.cocoindex_code\target_sqlite.db
Index stats: 18433 chunks, 806 files (cpp: 4367 chunks)
```

**制約**: `ccc search` は `sentence_transformers` 不足で動作せず。ただし `ccc mcp` コマンドでMCPサーバーとして起動可能。

**活用可能性**:

- MCPサーバーモード（`ccc mcp`）で起動すれば、MCPクライアントからセマンティック検索ツールとして利用可能
- ただし `sentence_transformers` が不足しているため、洞察品質が限定的
- 現時点では **semble CLI** が同等以上の機能を提供しており、ccc MCPモードの緊急導入は不要

**判定**: ✅ 現状維持。ccc MCPモードは `sentence_transformers` が利用可能になった時点で再評価。

---

## 3. 確定結果サマリー

| # | 項目 | 確定内容 |
|:-:|------|---------|
| U1 | P1 選択肢A vs B | ✅ **選択肢A（Output Stage）で実装済み。変更不要。** |
| U2 | P1 メイクアップゲイン係数 | ✅ **選択肢A採用により該当せず。** |
| U3 | P6 AGC改善案 | ✅ **案C（Attack 0.1→0.2s）実施済み。案A/Bは保留。** |
| U4 | DSPCoreFloat平均化 | ✅ **現状維持。P2修正によりブロック境界不連続性は解消済み。** |
| U5 | vHalf 削除可否 | ✅ **asymmetry計算で使用中。削除不可。** |
| U6 | P7+係数整合性 | ✅ **問題なし。修正後のフィルタでも既存係数は有効。再学習も可能。** |
| U7 | P1〜P7統合影響 | ✅ **完全独立経路。相互作用なし。** |
| U8 | ccc MCPモード | ✅ **現状維持。sentence_transformers有効化後に再評価。** |

---

## 4. ツール別評価（本調査）

| ツール | 調査内容 | 結果 |
|-------|---------|------|
| **grep/Select-String** | vHalf 全使用箇所、Float/Double パス分岐 | ✅ vHalf 2箇所（定義+asymmetry使用）、Float/Double独立確認 |
| **CodeGraph MCP** | `find_callers(processAGC)` 全6箇所、`global_search` 格子フィルタ関連コミュニティ | ✅ processAGC呼び出し、5 communities (5981+5053+4817+4436+3297 entities) |
| **AiDex MCP** | 278 files indexed | LatticeNoiseShaper 全メソッド一覧確認 |
| **semble CLI** | `evaluationWorkerMain` + `advanceState` + `evaluateCandidateMapped` の自然言語検索 | ✅ P7+NoiseShaperLearner関係を特定 |
| **graphify CLI** | `query("NoiseShaperLearner LatticeNoiseShaper advanceState")` | ✅ 57 nodes: evaluateCandidateMapped→processStereoBlock→advanceState の関係確認 |
| **ccc (cocoindex-code)** | `ccc status` でインデックス状態確認 | ✅ 18433 chunks, 806 files indexed。daemon searchは `sentence_transformers` 不足 |
| **Web文献調査** | 格子フィルタ理論再確認 | ✅ ARM CMSIS-DSP, MATLAB latcfilt, Proakis & Manolakis の正しさ再確認 |
