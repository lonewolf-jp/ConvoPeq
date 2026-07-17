# ConvoPeq バグ改修計画書

**作成日**: 2026-07-17 | **最終更新**: 2026-07-17
**ベース文書**: `ConvoPeq_bug_report_2026-07-17.md`
**リポジトリ**: <https://github.com/lonewolf-jp/ConvoPeq/tree/main>

---

# Phase 2 改修設計 — 未解決項目

このセクションは Phase 1（既に実装済み）で未着手となった3項目の改修設計を定義する。
いずれも ISR Runtime の Authority Singularization を完成させるための Architecture Improvement である。

---

## R-1: `RefCountedDeferred` の Retire Router 統合（抽象 Interface 経由）

### 現状

```cpp
// src/RefCountedDeferred.h
void release(convo::IEpochProvider& provider) {
    if (fetchSub(refCount, 1, acq_rel) == 1) {
        fence(acquire);
        if (!provider.enqueueRetire(this, deleter, provider.currentEpoch())) {
            if (!isAudioThread()) {
                provider.tryReclaim();
                (void)provider.enqueueRetire(this, deleter, provider.currentEpoch());
            }
        }
    }
}
```

**問題**: `IEpochProvider` を直接使用。Router 非経由のため Authority 分散、QueuePressure 通知が届かない。

### 修正方針

`RefCountedDeferred` は `ISRRetireRouter`（具象型）に直接依存せず、抽象 Interface `IRetireRouter` を介する。これにより Core Utility 層が AudioEngine 層に依存する Layering 違反を防止する。

```cpp
// ★ 新規: IRetireRouter — 最小限の抽象 IF
//   「Retire してください」だけを表現する。リトライ戦略・epoch・削除分類は
//   すべて Router 内部に隠蔽され、呼び出し元（RefCountedDeferred）はそれらを知らない。
//   QueuePressure が発生した場合、Router が RuntimeHealthMonitor へ通知する責務を持つ。
class IRetireRouter {
public:
    virtual ~IRetireRouter() = default;
    // RT-safe: 単発 enqueue、リトライなし。戻り値で成否を呼び出し元に伝える。
    // QueueFull 時は呼び出し元（AudioEngine）が後続処理を判断する。
    // NonRT の retire() と異なり、RT はリトライ不可のため bool を返す。
    virtual bool retireRT(
        void* ptr, void (*deleter)(void*)) noexcept = 0;
    // NonRT: リトライ込みの retire。Router 内部で tryReclaim + 再試行を行い、
    // 最終失敗時に QueuePressure を RuntimeHealthMonitor へ通知する。
    // 呼び出し元はリトライ回数・方針を一切知らない。戻り値不要のため void。
    virtual void retire(
        void* ptr, void (*deleter)(void*)) noexcept = 0;
};
```

`ISRRetireRouter` がこの IF を継承:
```cpp
class ISRRetireRouter : public IRetireRouter {
    // retire() の内部で currentEpoch() を呼び、DeletionEntryType::Generic を指定し、
    // enqueueWithRetry 相当のリトライ + QueuePressure 通知を行う。
    // retireRT() は単発 enqueue（RT-safe 用、リトライなし）。
};
```

`RefCountedDeferred::release()` は抽象 IF 経由:
```cpp
// ★ R-1: 抽象 IRetireRouter 経由（最小 IF + 全隠蔽）
void release(convo::IRetireRouter& router) {
    if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
        std::atomic_thread_fence(std::memory_order_acquire);
        (void)router.retire(
            static_cast<T*>(this),
            [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); });
    }
}
```

### 3種類の release 責務

| メソッド | 対象 | リトライ | 備考 |
|---------|------|:--------:|------|
| `releaseRT(IRetireRouter&)` | Audio Thread | なし | Router::retireRT 経由。戻り値 bool で QueueFull を検出。呼び出し元が後続処理を判断 |
| `releaseNonRT(IRetireRouter&)` | Non-RT Thread（RT禁止） | あり（Router::retire） | 通常の退役。RT からは呼ばないこと。リトライ・QueuePressure 通知は Router 内部で完結 |
| `releaseDirect()` | デストラクタ（Shutdown 専用） | なし | Shutdown 時の最終手段。即時 delete。事前条件: Retire Queue 停止済み、Epoch 終了済み、他スレッド不在 |

| 影響ファイル | 変更内容 |
|-------------|---------|
| `src/core/IRetireRouter.h`（新規） | 抽象 Interface 定義 |
| `src/audioengine/ISRRetireRouter.h` | `IRetireRouter` を継承 |
| `src/RefCountedDeferred.h` | `release()` 引数型変更 + `releaseRT()` 追加 |
| 全呼び出し元 | `IEpochProvider&` → `IRetireRouter&` に変更 |

**工数**: 設計レビュー1h + 実装2-3h + テスト1h

---

## R-2: `retireDSP()` の整理

### 現状

`AudioEngine.h:3928` に `retireDSP()` が inline 定義されているが、呼び出し元が存在しない（rg で確認済み）。

### 修正方針

削除。DSP の退役は `DSPLifetimeManager` 経由に一本化済み。

```cpp
// retireDSP は呼び出し元不在につき削除。
// DSP の退役は DSPLifetimeManager 経由に一本化。
// [[deprecated("Use DSPLifetimeManager::retire() instead")]]
// inline void retireDSP(DSPCore*) = delete;
```

| 影響ファイル | 変更内容 |
|-------------|---------|
| `src/audioengine/AudioEngine.h` | `retireDSP()` 宣言・実装を削除 |

### 削除前確認事項

`retireDSP` の参照は以下を含めて全リポジトリを検索し、ゼロであることを確認してから削除する:

- 直接呼び出し: `retireDSP(`
- 関数ポインタ: `&AudioEngine::retireDSP`
- メンバ関数ポインタ: `&decltype(...)::retireDSP`, `decltype(&AudioEngine::retireDSP)`
- `std::bind` / `std::function`: `std::bind(&AudioEngine::retireDSP, ...)`
- テンプレート推論: テンプレート引数としての参照
- `static_cast` / `reinterpret_cast` 経由のポインタ取得
- コメント・ドキュメント内の言及
- スクリプト（`.github/scripts/` 等）・テストコード内の文字列

**加えて**: `grep` / `rg` による文字列検索に加え、**clangd Call Hierarchy** または **clang-query** を用いたシンボル参照解析も実施し、関数ポインタ経由・テンプレート推論経由の参照を網羅する。

```bash
# 検索コマンド例
grep -rn "retireDSP" src/ --include="*.cpp" --include="*.h"
# 上記に加え
rg "retireDSP" src/ -g "*.cpp" -g "*.h"
```

**リスク**: いずれかが存在する場合は削除せず、参照元の修正を先行させる。
**工数**: 10分（検索込）

---

## R-3: DSPCoreDouble TanhApprox の共通 Utility 統合

### 現状

`DSPCoreDouble.cpp` に独自の匿名名前空間 `TanhApprox`（10395 係数）が存在。AVX2 版の係数も直接参照。

### 修正方針

匿名名前空間を削除し、`SoftClipPadéPolicy`（既に `FastTanhApprox.h` に定義済み）を使用する。

```cpp
// DSPCoreDouble.cpp 変更後:
#include "dsp/math/FastTanhApprox.h"
// 匿名名前空間 TanhApprox を削除
// スカラー: convo::dsp::fastTanh<convo::dsp::SoftClipPadéPolicy>(x)
// AVX2: convo::dsp::fastTanhV256<convo::dsp::SoftClipPadéPolicy>(x)
//   → Policy は係数・閾値のみ。SIMD は FastTanhApprox 側が Policy::Coefficients を読む
```

> **設計**: Policy は係数と閾値のみを保持するデータ構造。`compute()` のようなアルゴリズムは持たない。SIMD 演算は `FastTanhApprox` 側の `fastTanhV128<Policy>()` / `fastTanhV256<Policy>()` が `Policy::Coefficients` を読み込んで実行する。これにより AVX512 / NEON / SVE 追加時も Policy の変更は一切不要になる。
>
> **Policy 要件**: Policy は以下の `static constexpr` メンバを提供しなければならない（命名規則: PascalCase で統一）。
> - `ClipThreshold` — クリップ閾値（例: 4.5）
> - `NumA`, `NumB`, `NumC` — 分子多項式係数
> - `DenA`, `DenB`, `DenC` — 分母多項式係数
>
> これらの存在は `static_assert` または C++20 `requires` でコンパイル時に検証することを推奨する。
>
> `fastTanh<Policy>()` は `Policy::NumA` 等を直接参照する。`SoftClipPadéPolicy` は内部で係数を完結して持つため、`DefaultFastTanhCoefficients` の固定参照は行わない。係数を Policy 直下に持つことで、AVX2 / AVX512 / NEON / SVE の全 SIMD 実装から同一パターンでアクセスできる。

`FastTanhApprox.h` の `fastTanhV256<Policy>()` に `SoftClipPadéPolicy` 係数対応を追加（Policy テンプレート版）:

```cpp
// DSPCoreDouble 向け AVX2 版 — SoftClipPadéPolicy の係数を Policy::Coefficients から取得
template<class Policy = DefaultFastTanhPolicy>
[[nodiscard]] inline __m256d fastTanhV256(__m256d x) noexcept {
    const auto vClipHigh = _mm256_set1_pd(Policy::ClipThreshold);
    const auto vClipLow  = _mm256_set1_pd(-Policy::ClipThreshold);
    const auto xClamped = _mm256_min_pd(_mm256_max_pd(x, vClipLow), vClipHigh);
    const auto x2 = _mm256_mul_pd(xClamped, xClamped);
    // ★ Policy が直接保持する static constexpr 係数を参照
    //   （Coefficients 構造体のネストを避け、constexpr 最適化を促進）
    const auto vNumA = _mm256_set1_pd(Policy::NumA);
    const auto vNumB = _mm256_set1_pd(Policy::NumB);
    const auto vNumC = _mm256_set1_pd(Policy::NumC);
    const auto vDenA = _mm256_set1_pd(Policy::DenA);
    const auto vDenB = _mm256_set1_pd(Policy::DenB);
    const auto vDenC = _mm256_set1_pd(Policy::DenC);
    const auto num = _mm256_mul_pd(xClamped, _mm256_add_pd(vNumA,
        _mm256_mul_pd(x2, _mm256_add_pd(vNumB, _mm256_mul_pd(x2, vNumC)))));
    const auto den = _mm256_add_pd(vDenA,
        _mm256_mul_pd(x2, _mm256_add_pd(vDenB, _mm256_mul_pd(x2, _mm256_add_pd(vDenC, x2)))));
    return _mm256_div_pd(num, den);
}
```

| 影響ファイル | 変更内容 |
|-------------|---------|
| `src/dsp/math/FastTanhApprox.h` | `SoftClipPadéPolicy` に係数・閾値定義追加。`fastTanhV256()` に AVX2 版追加（Policy はデータのみ、SIMD は FastTanhApprox 側） |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 匿名名前空間削除 + 共通 Utility `fastTanh<SoftClipPadéPolicy>()` / `fastTanhV256<SoftClipPadéPolicy>()` 参照切替 |

**工数**: 1.25時間（FastTanhApprox.h AVX2追加 15分 + DSPCoreDouble切替 30分 + 数値検証 30分）

---

## Phase 2 優先順位

```
優先順位:
  1. R-3: DSPCoreDouble TanhApprox 統合（影響範囲限定、単一コンポーネント）
  2. R-2: retireDSP 削除（Dead code、影響リスク小）
  3. R-1: RefCountedDeferred Router 統合（Interface 変更、影響範囲最大のため最後）
```

### 実装条件

- [ ] R-3 完了後、Bug 3 の THD/Transfer Curve 測定で新旧一致確認
- [ ] R-1 完了後、EQ/Convolver のリグレッションテスト
- [ ] R-2 完了後、全リポジトリで `retireDSP` の参照がゼロであることを確認

---

# Part 1: 実装済み設計仕様（Appendix）

以下の項目は **Phase A / Phase B（Phase 1）として実装済み**。

---

## 1. 実装済み項目一覧

| # | 項目 | ファイル | 状態 |
|---|------|---------|:----:|
| P1-A | Bug 1: delayLineBuf メモリリーク | `MKLNonUniformConvolver.h/.cpp` | ✅ 実装済み |
| P1-C | Bug 7: SetImpulse() ブレース漏れ | `MKLNonUniformConvolver.cpp` | ✅ 実装済み |
| P3-B | Bug 6: delayLineReadAdd std::abs | `MKLNonUniformConvolver.cpp` | ✅ 実装済み |
| P2-B | Bug 4: ProgressiveUpgradeThread FTZ/DAZ | `ProgressiveUpgradeThread.cpp` + `IRDSP.cpp` + `ScopedMXCSR.h` | ✅ 実装済み |
| P2-A | Bug 3: FastTanhApprox 共通 Utility 化 | `FastTanhApprox.h` + `EQProcessor.Processing.cpp` | ✅ 実装済み |
| P1-B | Bug 2 Phase 1: enqueueWithRetry 集約 | `ISRRetireRouter.h/.cpp` + `DSPLifetimeManager.h` + `AudioEngine.h` | ✅ 実装済み |
| F-01 | MKL mkl_set_dynamic(0) 削除 | `MKLRealTimeSetup.cpp` | ✅ 実装済み |

---

## 2. ファイル変更一覧（実装記録）

| # | ファイル | 変更内容 | フェーズ |
|---|---------|---------|---------|
| 1 | `src/MKLNonUniformConvolver.h` | `LayerAllocSizes::delayLineBuf` 追加 | P1-A |
| 2 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()`: delayLineBuf DIAG_MKL_MALLOC 化 | P1-A |
| 3 | `src/MKLNonUniformConvolver.cpp` | `freeAll()`: freeTracked(delayLineBuf) 追加 | P1-A |
| 4 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()`: if(!l.isImmediate) ブレース修正 | P1-C |
| 5 | `src/MKLNonUniformConvolver.cpp` | delayLineReadAdd: std::abs→absNoLibm + [[nodiscard]] constexpr | P3-B |
| 6 | `src/audioengine/ISRRetireRouter.h/cpp` | enqueueWithRetry() 追加（リトライAuthority集約） | P1-B |
| 7 | `src/audioengine/DSPLifetimeManager.h` | retire(): enqueueWithRetry 委譲 | P1-B |
| 8 | `src/audioengine/AudioEngine.h` | enqueueDeferredDeleteNonRtWithResult: enqueueWithRetry 委譲 | P1-B |
| 9 | `src/eqprocessor/EQProcessor.Core.cpp` | stackRouter.enqueueWithRetry 使用 | P1-B |
| 10 | `src/ProgressiveUpgradeThread.cpp` | run() 冒頭で FTZ/DAZ 設定 | P2-B |
| 11 | `src/IRDSP.cpp` | std::async ラムダ内 ScopedMXCSR | P2-B |
| 12 | `src/core/ScopedMXCSR.h`（新規） | convo::cpu::ScopedMXCSR RAII クラス | P2-B |
| 13 | `src/dsp/math/FastTanhApprox.h`（新規） | Policy テンプレート版 fastTanh / fastTanhV128 | P2-A |
| 14 | `src/eqprocessor/EQProcessor.Processing.cpp` | fastTanh → 共通 Utility 参照切替 | P2-A |
| 15 | `src/MKLRealTimeSetup.cpp` | mkl_set_dynamic(0) 削除 | F-01 |
| — | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | TanhApprox 統合保留 | R-3 |

---

## 3. テスト計画（実装確認記録）

### 3.1 検証結果サマリ

| ツール | 用途 | 結果 |
|--------|------|:----:|
| rg/sed/awk | freeTracked/allocSizes ペア検証 | ✅ allocSizes 15 = freeTracked 15 完全一致 |
| ast-grep | DIAG_MKL_MALLOC 網羅率 | ✅ 永続レイヤーバッファ全15件カバー |
| Python | FastTanhApprox 数値検証 | ✅ 27/9公式と10395公式の両方を確認 |
| AiDex | コードインデックス検索 | ✅ 変更箇所の索引確認 |

### 3.2 残テスト項目

- Bug 3 音響評価（THD/Transfer Curve/dy/dx/d²y/dx²） — 実機試験
- Bug 2 Overflow Test + Publish Latency (P50/P95/P99) — 実機試験
- MXCSR 確認（_mm_getcsr()） — 1回のみ
- Shutdown シナリオ試験 — 結合テスト

---

# Part 2: 未確定事項・保留事項

## 2.1 実測が必要な項目

| 項目 | 目的 | 状態 |
|------|------|:----:|
| `overflowCount()` 実測 | Bug 2 の実運用発生確認 | ⏳ 未実測 |
| QueueDepth 時系列推移 | Bug 5 の epoch 逆転検出 | ⏳ 未実測 |
| Publish Latency (P50/P95/P99) | Admission 制御の副作用評価 | ⏳ 未実測 |
| `_mm_getcsr()` 確認 | Bug 4 std::async MXCSR 継承状況 | ⏳ 未確認 |

## 2.2 コンパイラ依存（調査済み）

| 項目 | 確定判断 |
|------|:--------:|
| `std::bit_cast` constexpr | ✅ MSVC 2022 で安全。`[[nodiscard]] constexpr inline` 採用済み |
| `std::async` MXCSR 継承 | ✅ 継承を前提にしない（実装依存のため毎回明示設定） |

## 2.3 要追加調査（調査済み）

| 調査項目 | 確定判断 |
|---------|:--------:|
| `ConvolverProcessor.LoaderThread` retire 経路 | ✅ enqueueDeferredDeleteNonRt 経由で epoch 確保済み |

---

# Part 3: 設計改訂履歴・リスク評価・付録

## 3.1 設計改訂履歴

| 回 | 主な変更点 |
|:--:|-----------|
| 1-13 | 設計レビュー・計画書作成 |
| 14 | 実装フェーズ開始（全16項目実装） |
| 15-16 | 実装検証・バグ修正 |
| 17 | Phase 2 設計追加。文書3部構成に再編（実装済み→Appendix） |

## 3.2 リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|:----:|:----:|------|
| R-1 引数変更の影響範囲 | 中 | 中 | 全呼び出し元の洗い出し必須 |
| R-3 数値一致 | 低 | 低 | 新旧 Transfer Curve 比較で確認 |

## 3.3 追加 FIX 項目

| FIX | 項目 | 状態 |
|-----|------|:----:|
| FIX-01 | MKL `mkl_set_dynamic(0)` 削除 | ✅ 実装済み |
| FIX-02 | LoudnessMeter BS.1770 動的係数計算 | ✅ 既に実装済み: `updateCoefficients()` が RBJ Cookbook で計算 |
| FIX-03 | Denormal Architectural Invariant 文書化 | 📝 文書化作業のみ |

## 3.4 Authority Singularization 監査レポート（概要）

全19項目を Authority 観点から再評価。16件安全確認、1件（`ConvolverProcessor.LoaderThread` retire 経路）は調査完了し安全確認済み。

## 3.5 検証結果の概要

全7バグについて実コード検証完了。全件が正確な指摘であることを確認。誤検知5項目もすべて「問題なし」確認済み。
