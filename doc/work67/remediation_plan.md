# ConvoPeq 改修計画書 v4.2（最終版）

作成日: 2026-07-06
更新内容: v4.1 の最終調整（2点反映）

---

## エグゼクティブサマリー

v4.1 からの最終調整 2 点:

| # | 項目 | v4.1 → v4.2 |
| :--- | :--- | :--- |
| 1 | **P1-3 API** | `aligned_malloc_throw()` を削除 → `aligned_malloc()` (throwing) + `aligned_malloc_nothrow()` の 2 本立てに簡略化 |
| 2 | **M-1 文言** | 「prepareToPlay なしで呼ぶと未定義」→「prepareToPlay で生成された未完成 engine の IR を完成させる責務」に改善 |

---

## 凡例

| 記号 | 意味 |
| :--- | :--- |
| 🚨 **S** | クラッシュまたは未定義動作に直結。即日修正必須 |
| ⚠️ **A** | 音質劣化または RT 安全性の低下リスク。計画的対応 |
| 🔶 **B** | 間接的な品質問題。余裕のあるタイミングで |
| 🔹 **C** | 運用効率・開発生産性に関する改善 |
| 🔍 **R** | 修正前に追加解析・調査が必要 |

---

## コード実証による未確定事項の確定

| 調査項目 | 結果 | 根拠 |
| :--- | :--- | :--- |
| `aligned_malloc` の実体 | `mkl_malloc` → 最終的に **HeapAlloc**（Windows ヒープ）。**RT-safe ではない** | `AlignedAllocation.h:L14-17` |
| RT ファイルの alloc 呼び出し | **ゼロ**。DSPCoreDouble/EQProcessor/ConvolverProcessorRuntime の 4 ファイルで 0 件 | `mcp_context-mode_ctx_execute` 走査 |
| 既存状態管理 atomic | `isPrepared`, `isLoading`, `m_ready` が存在。`IncrementalRebuildJob::Stage` が 6 状態（Idle→Prepared→Building→FinalizingPrepare→FinalizingApply→Done） | `ConvolverProcessor.Lifecycle.cpp`, `MKLNonUniformConvolver.h:L353` |
| Hard Clamp の実体 | `juce::jlimit(-kOutputHeadroom, kOutputHeadroom, sample)` — **単純なサンプル単位クランプ** | `DSPCoreDouble.cpp:L791-793` |

---

## P0: 🚨 即時改修

### P0-1: AVX2 ランタイムチェック欠如

起動時 1 回の CPUID チェック + 非対応時エラーダイアログ。

**見積り**: 0.5 人日

### P0-2: `/fp:fast` + `std::isnan()` 併用

RT コンテキスト（`LatticeNoiseShaper` 2 件）のみビットパターン判定化。

**見積り**: 0.5 人日

### P0-3: `jassert` Release 安全フォールバック欠如

6 件の UB 防止。`fetchAddAtomic` + `[[unlikely]]` で個別実装。

**見積り**: 0.5 人日

---

## P1: ⚠️ 計画的改修

### P1-2: TruePeak/LUFS 計測位置移動

`kOutputHeadroom` + ディザ適用後に TP/LUFS 計測を移動。

**見積り**: 1 人日

---

### P1-3: アロケータ整理（命名 `_nothrow`）

**命名**: `aligned_malloc_nothrow()` / `makeAlignedArray_nothrow()`。

```cpp
// 例外を投げる（Message Thread 用、std::bad_alloc 到達可）
void* aligned_malloc(size_t, size_t);

// 例外を投げない（名前で契約を明示）
void* aligned_malloc_nothrow(size_t, size_t) noexcept;  // 失敗時 nullptr
```

**`_rt` ではなく `_nothrow` とした理由**: `mkl_malloc` は最終的に
`HeapAlloc` (Windows ヒープ) を呼ぶため、RT-safe ではない。RT ファイル
（`DSPCoreDouble.cpp`, `EQProcessor.Processing.cpp` 等）では
現在も `mkl_malloc` を呼んでいない（コード実証済み）。「RT 安全」と
誤解される `_rt` より、実際の契約（例外有無）を表現する `_nothrow` が
正確。

**見積り**: 1.5 人日（rename + コメント更新 + clang-tidy 設定 + レビュー）

---

### P1-1: Simple Peak Limiter 導入

Phase 1（Release-only Simple Peak Limiter）。Phase 2（LookAhead）以降は
Phase 1 の実運用評価後に判断。

**既存 Hard Clamp の位置付け**: `juce::jlimit(-kOutputHeadroom, ...)` —
これは Brickwall Limiter ではなく**安全網 (Safety Net)**である。
NaN/Inf/計算誤差が Limiter をすり抜けた場合の最終防護。
Simple Peak Limiter はこの安全網より**手前**で動作し、ほとんどの
オーバーをソフトに処理する。

```text
入力 → [Simple Peak Limiter] → [その他DSP] → [Hard Clamp (Safety Net)] → 出力
```

**見積り**: 2 人日

---

### P1-4: Double/float template 統合

保守性改善。ISR とは無関係。最終フェーズ。

**見積り**: 5 人日

---

## P2: 🔶 改善推奨

### P2-1: NaN/Inf スクラブ削減（R-2 解析後）

### P2-2: clang-tidy CI 強制

### P2-3: AGC ブロックレート制限の文書化

### P2-4: OutputFilter 高域特性測定

### P2-5: `diagLog` デッドコード削除

---

## M: 🔶 保守性リスク

### M-1: ConvolverProcessor IR 再構築の暗黙的呼び出し規約

**設計**: 新規 atomic (`needsIRResample`) は追加**しない**。
代わりに既存の状態管理に統合する。

**既存状態**: すでに以下が存在する:

- `MKLNonUniformConvolver::m_ready` (bool atomic, L353)
- `ConvolverProcessor::isPrepared` (bool atomic, L201)
- `IncrementalRebuildJob::Stage`: `Idle → Prepared → Building → FinalizingPrepare → FinalizingApply → Done`

**対応**: `prepareToPlay()` の率直なコメント + `rebuildAllIRsSynchronous()` 先頭での責務明記。型による契約まで踏み込むと大規模リファクタになるため（`PreparedEngine`/`RunningEngine` の分離）、今回は現実的な「コメント + 責務明記」に留める。

```cpp
// ConvolverProcessor.Lifecycle.cpp
// ★ CAUTION: この分岐で生成した engine は IR が未リサンプリング状態。
// 呼び出し元は直後に rebuildAllIRsSynchronous() を呼ぶこと。
// この engine を単独で commit してはならない。
```

```cpp
// ConvolverProcessor.Rebuild.cpp — rebuildAllIRsSynchronous() 先頭
// 責務: prepareToPlay() により生成された未完成 engine の IR を
// 正しいリサンプリングで完成させる。
// この関数は prepareToPlay() で生成された engine に対してのみ
// 呼び出すことができる。
```

**見積り**: 0.5 人日

---

## R: 🔍 要調査項目

### R-1: `isFiniteNoLibm` 重複実装の統合

### R-2: 異常値伝搬解析

---

## S: 💡 任意提案

### S-2: 診断カウンタ API の統一

### S-3: FTZ/DAZ 設定方針の文書化

---

## 削除／統合した S 項目と代替方針

| 旧項目 | 代替方針 |
| :--- | :--- |
| **S-1** (Release Guard マクロ) | 個別実装で対応済み（P0-3, 6 件のみ） |
| **S-4** (noexcept 監査) | **RTコードレビューチェックリスト**に統合 |
| **S-5** (`[[unlikely]]`) | 異常系修正時に随時付与（P0-3 他） |

---

## 改訂スケジュール（v4.1）

| Phase | 期間 | タスク | 工数 |
| :--- | :--- | :--- | :--- |
| Phase 0 | 即日 | **P0-2** / **R-1** / **P2-5** | 0.9人日 |
| Phase 1 | 1週目 | **P0-1** / **P0-3** | 1.0人日 |
| Phase 2 | 2週目 | **P1-2** / **P1-3** / **M-1** | 3.0人日 |
| Phase 3 | 3-4週目 | **P1-1** / **R-2** | 3-4人日 |
| Phase 4 | 5週目以降 | **P1-4** / **P2-x** / **S-2/S-3** | 7.7人日 |

---

## 改修前後比較表（v4.1）

| 指標 | 改修前 | 改修後 | 対応 |
| :--- | :--- | :--- | :--- |
| CPU非互換クラッシュ | あり (SIGILL) | エラーダイアログ表示 | P0-1 |
| NaNガード確実性 | fp:fast依存 | ビットパターン確定 | P0-2 + R-1 |
| Release安全網 | jassert のみ 6 件無防備 | `jassert + [[unlikely]] + fetchAddAtomic` | P0-3 |
| TruePeak/LUFS精度 | 約+1.0dB乖離 | BS.1770-4準拠 | P1-2 |
| アロケータ契約 | 例外有無が名前から不明 | `_nothrow` で明示 | P1-3 |
| オーバー対策 | 安全網 (Hard Clamp) のみ | Simple Limiter + 安全網 | P1-1 |
| IR再構築安全性 | 暗黙の呼び出し規約 | コメント + アサーション明示 | M-1 |
| 保守対象コード | float+double 二重管理 | template 一本化 | P1-4 |
| CI品質ゲート | clang-tidy OFF | CI で強制 ON | P2-2 |
| `isFiniteNoLibm` | 5 重複実装 | 1 箇所統合 | R-1 |

---

## 付録: 全タスク一覧

| ID | 優先度 | タスク | 工数 | 成果物 |
| :--- | :--- | :--- | :--- | :--- |
| P0-1 | 🚨 S | AVX2 ランタイムチェック | 0.5人日 | `CpuFeatureCheck.h/.cpp` |
| P0-2 | 🚨 S | fp:fast + LatticeNoiseShaper | 0.5人日 | `LatticeNoiseShaper.h` |
| P0-3 | 🚨 S | jassert Release ガード 6件 | 0.5人日 | 6 ファイル |
| P1-2 | ⚠️ A | TruePeak/LUFS 位置移動 | 1人日 | `DSPCoreDouble.cpp` |
| P1-3 | ⚠️ A | アロケータ整理 (`_nothrow`) | 1.5人日 | `AlignedAllocation.h` |
| P1-1 | ⚠️ A | Simple Peak Limiter | 2人日 | `SimplePeakLimiter.h/.cpp` |
| P1-4 | ⚠️ A | float/double template 統合 | 5人日 | 2ファイル統合 |
| P2-1 | 🔶 B | NaN/Inf スクラブ削減 | 1人日 | `DSPCoreDouble.cpp` |
| P2-2 | 🔶 B | clang-tidy CI 強制 | 2人日 | CI 設定 |
| P2-3 | 🔶 B | AGC 文書化 | 0.2人日 | コメント |
| P2-4 | 🔹 C | OutputFilter 測定 | 0.5人日 | テストコード |
| P2-5 | 🔹 C | diagLog 削除 | 0.1人日 | `DSPCoreDouble.cpp` |
| M-1 | 🔶 B | IR 再構築契約明示 | 0.5人日 | `ConvolverProcessor.Lifecycle.cpp` |
| R-1 | 🔍 R | isFiniteNoLibm 統合 | 0.3人日 | `DspNumericPolicy.h` |
| R-2 | 🔍 R | NaN 異常伝搬解析 | 1-2人日 | 解析レポート |
| S-2 | 💡 任意 | 診断カウンタ API 統一 | 0.5人日 | `AtomicAccess.h` |
| S-3 | 💡 任意 | FTZ/DAZ 文書化 | 0.2人日 | アーキテクチャドキュメント |

**総工数**: 16.3 - 17.3 人日

---

Plan v4.2 — generated from source code analysis on 2026-07-06
