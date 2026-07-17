# ConvoPeq バグ改修計画書

**作成日**: 2026-07-17
**ベース文書**: `ConvoPeq_bug_report_2026-07-17.md`（検証結果反映済み）
**リポジトリ**: <https://github.com/lonewolf-jp/ConvoPeq/tree/main>

---

## 目次

1. [改修スケジュール（推奨）](#1-改修スケジュール推奨)
2. [Phase 1: Critical（即日対応）](#2-phase-1-critical即日対応)
   - [P1-A: Bug 1 — delayLineBuf メモリリーク](#p1-a-bug-1--delaylinebuf-メモリリーク)
   - [P1-B: Bug 2 — Retireキュー飽和時のポインタロスト](#p1-b-bug-2--retireキュー飽和時のポインタロスト)
   - [P1-C: Bug 7 — SetImpulse() ブレース漏れ（Bug 1と同時実施）](#p1-c-bug-7--setimpulse-ブレース漏れbug-1と同時実施)
3. [Phase 2: High（優先対応）](#3-phase-2-high優先対応)
   - [P2-A: Bug 3 — EQ fastTanh しきい値不整合](#p2-a-bug-3--eq-fasttanh-しきい値不整合)
   - [P2-B: Bug 4 — ProgressiveUpgradeThread FTZ/DAZ 未設定](#p2-b-bug-4--progressiveupgradethread-ftzdaz-未設定)
4. [Phase 3: Medium（計画的対応）](#4-phase-3-medium計画的対応)
   - [P3-B: Bug 6 — delayLineReadAdd() の std::abs（品質改善）](#p3-b-bug-6--delaylinereadadd-の-stdabs品質改善)
5. [Backlog（測定後に判断）](#5-backlog測定後に判断)
   - [B-1: Bug 5 — reclaim() 先読みスキャン](#b-1-bug-5--reclaim-先読みスキャン)
6. [依存関係・改修順序](#5-依存関係・改修順序)
7. [リスク評価](#6-リスク評価)
8. [テスト計画](#7-テスト計画)
9. [付録: ファイル変更一覧](#8-付録-ファイル変更一覧)

---

## 1. 改修スケジュール（推奨）

```text
Phase 1 [Critical]  ───────────────────────────────┐
  P1-A: Bug 1  delayLineBuf リーク                  │
  P1-C: Bug 7  ブレース漏れ（Bug1と同時）          │ 即日着手
  P1-B: Bug 2  Retireキュー飽和ロスト              │
                                                    │
Phase 2 [High]  ───────────────────────────────────┤
  P2-A: Bug 3  EQ fastTanh しきい値                │
  P2-B: Bug 4  ProgressiveUpgradeThread FTZ/DAZ    │ 今週中
                                                    │
Phase 3 [Medium]  ─────────────────────────────────┤
  P3-B: Bug 6  std::abs → absNoLibm               │ 計画的対応
                                                    │
Backlog ───────────────────────────────────────────┤
  B-1: Bug 5  reclaim() デッドコード               │ overflowCount実測後
```

---

## 2. Phase 1: Critical（即日対応）

### P1-A: Bug 1 — `delayLineBuf` メモリリーク

**重大度**: Critical
**影響**: 診断ビルド（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`）で IR 再構築のたびに `delayLineCapacity × sizeof(double)` バイトが確実にリーク。診断統計が実メモリ使用量を過小評価する。

#### 修正内容

**ファイル 1: `src/MKLNonUniformConvolver.h`**

`LayerAllocSizes` 構造体に `delayLineBuf` フィールドを追加:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
struct LayerAllocSizes {
    size_t irFreqDomain = 0;
    size_t irFreqReal   = 0;
    size_t irFreqImag   = 0;
    size_t fdlBuf       = 0;
    size_t fdlReal      = 0;
    size_t fdlImag      = 0;
    size_t fftTimeBuf   = 0;
    size_t fftOutBuf    = 0;
    size_t prevInputBuf = 0;
    size_t accumBuf     = 0;
    size_t accumReal    = 0;
    size_t accumImag    = 0;
    size_t inputAccBuf  = 0;
    size_t tailOutputBuf= 0;
    size_t delayLineBuf = 0;   // ★ Bug#1 追加
};
```

**ファイル 2: `src/MKLNonUniformConvolver.cpp` — 確保側 (SetImpulse内)**

`delayLineBuf` の確保を `DIAG_MKL_MALLOC` に変更し、`allocSizes` を記録:

```cpp
// 変更前:
l.delayLineBuf = static_cast<double*>(
    mkl_malloc(static_cast<size_t>(l.delayLineCapacity) * sizeof(double), 64));

// 変更後:
const size_t delayLineBytes = static_cast<size_t>(l.delayLineCapacity) * sizeof(double);
l.delayLineBuf = static_cast<double*>(DIAG_MKL_MALLOC(delayLineBytes, 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.delayLineBuf = delayLineBytes;
#endif
```

**ファイル 3: `src/MKLNonUniformConvolver.cpp` — 解放側 (freeAll() 診断ブランチ)**

`freeTracked(delayLineBuf, allocSizes.delayLineBuf)` を追加:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
    freeTracked(irFreqReal,    allocSizes.irFreqReal);
    // ... (既存14行) ...
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
    freeTracked(delayLineBuf,  allocSizes.delayLineBuf);   // ★ Bug#1 追加
    allocSizes = {};   // freeTracked 内部でポインタは nullptr に設定済み。最後に allocSizes をゼロクリア
```

> **補足**: `freeTracked()` は内部で `mkl_free(ptr)` + `ptr = nullptr` を一括実行する。その後 `allocSizes = {}` でサイズ情報をクリアする。この順序（free + nullptr → サイスクリア）により、解放後ポインタの誤使用を防止する。`freeTracked()` が nullptr 代入を行わない実装の場合（現行コードでは代入していることを確認済み）、明示的に `delayLineBuf = nullptr` を追加すること。

#### 影響評価
- 非診断ビルド（本番）: `DIAG_MKL_MALLOC` は `mkl_malloc` に展開されるため、動作変更なし
- 診断ビルド: リーク解消 + 統計値が正確になる
- `NucDiagnosticsSnapshot.totalBytes()` に `delayLineBuf` のサイズが正しく計上される
- `freeAll()` で `delayLineCapacity = 0` / `delayWriteCursor = 0` / `delayReadCursor = 0` も同時クリア（解放後オブジェクトの状態完全性を確保）
- リグレッションリスク: 極めて低い（既存の全バッファと同一パターン）

#### 推定工数: 15分（3ファイル・3箇所の変更）

---

### P1-B: Bug 2 — Retireキュー飽和時のポインタロスト

**重大度**: Critical
**影響**: `DeferredDeletionQueue`（4096エントリ）が満杯かつ全 Reader が stuck した場合、`DSPCore*` 等が恒久的にリークする可能性がある。4箇所で同一パターンが確認されている。

#### 補足: 現状の保護機構

調査の結果、以下の保護が既に存在することを確認した:
- `ISRRetireRouter::enqueueRetire()` 内部: 500msクールダウン付き `tryReclaim` + 再試行（`ISRRetireRouter.cpp:110-120`）
- `EpochDomain::detectStuckReaders()`: 10秒/30秒閾値の滞留検出 + Quarantine機構
- `DeferredRetireFallbackQueue`: 別系統（ISRRetireRuntime用）のフォールバックキュー（SoftLimit=1000, HardLimit=2000）
- **`RuntimeHealthMonitor::checkOverflowRate()`**: `overflowCount` を1秒窓のレート監視し、Warning/Error 状態遷移を行う（`RuntimeHealthMonitor.cpp:764-777`）
- **`AudioEngine::drainDeferredRetireQueues()`**: `tryReclaim()` + `m_coordinator.reclaim()` の2段階ドレイン（`AudioEngine.Retire.cpp:41-51`）

**各経路の実効リトライ回数（ルーター内部リトライ含む）**:

| 経路 | 呼び出し元リトライ | ルーター内部リトライ | 合計 |
|------|-----------------|-------------------|------|
| ① `DSPLifetimeManager::retire()` | 1回（tryReclaim + 再試行） | 1回（500ms冷却） | 最大2回 |
| ② `AudioEngine::retireDSP()` | drainQueues + coordinator.reclaim | 1回（500ms冷却） | 2段階ドレイン（リエンキューなし） |
| ③ `EQProcessor`（coordinator経由） | 1回（tryReclaim + 再試行） | 1回（500ms冷却） | 最大2回 |
| ④ `RefCountedDeferred::release()` NonRT | 1回（tryReclaim + 再試行） | 1回（500ms冷却） | 最大2回 |
| ④ `RefCountedDeferred::release()` RT | 0回（isAudioThread禁止） | 1回（500ms冷却） | ルーターのみ1回 |

> **訂正**: 報告書では ③ `RuntimePublicationCoordinator::enqueueRetire()` を「リトライなし」としていたが、実際の呼び出し元 `EQProcessor.Core.cpp:51-74` は `tryReclaim()` + 再試行を実装している。`RuntimePublicationCoordinator::enqueueRetire()` 自体はリトライを持たないが、呼び出し元が自前でリトライする設計。

**主要な問題**: これらの経路はすべて `DeferredDeletionQueue` のみを使用し、`DeferredRetireFallbackQueue` の保護を受けない。リトライを尽くしてもキューが空かない場合（全 Reader が長時間 stuck）、ポインタは確実にロストする。

#### 修正方針（2フェーズ）

> リトライポリシーは各呼び出し元ではなく `ISRRetireRouter` に集約し、**`enqueueWithRetry()`** として Single Semantic Source 化する。これにより Retry の Authority が一箇所になり、保守性が向上する。

##### Phase 1: 暫定 Mitigation

> **注意**: リトライ回数の増加は**対症療法**である。ISR Runtime の理想は「enqueue 失敗」が起きない設計であり、
> 以下の修正は「暫定対策」として位置付ける。恒久対策は Step 2 の QueuePressure を RuntimePolicyEngine で扱う設計を参照。

`DSPLifetimeManager::retire()` のリトライを `tryReclaim()` の進展有無で制御するよう改善:

```cpp
// src/audioengine/DSPLifetimeManager.h
void retire(AudioEngine::DSPCore* dsp) noexcept
{
    if (dsp == nullptr) return;
    if (!engine_.retireDSPHandleForRuntime(dsp))
        return;

    const uint64_t epoch = router_->currentEpoch();
    bool enqueued = router_->enqueueRetire(
        static_cast<void*>(dsp), &AudioEngine::destroyDSPCoreNode, epoch);

    // ★ tryReclaim の進展有無で制御。進展がなければ即座に諦める（ISR Runtime 基本思想）
    // ★ QueuePressure 主導: リトライは最小限（2回）で諦め、PolicyEngine に Pressure を委譲
    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        const uint32_t reclaimed = router_->tryReclaim();
        if (reclaimed == 0)
            break;
        enqueued = router_->enqueueRetire(
            static_cast<void*>(dsp), &AudioEngine::destroyDSPCoreNode, epoch);
        if (enqueued)
            break;
    }

    if (!enqueued) {
        // 全リトライ終了後、一度だけ QueuePressure を RuntimePolicyEngine へ通知
        // 通知: RuntimePolicyEvent::QueuePressure (Level: High)
        // コンテキスト: QueuePressureInfo { pendingDepth, overflowCount, reclaimProgress, enqueueFailures, source }
        // → PolicyEngine が AdmissionPressureLevel を引き上げ、新規 publish を抑制
        return;
    }

    convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount, ...);
}
```

`AudioEngine::retireDSP()` の経路は、`enqueueDeferredDeleteNonRtWithResult()` 内で `drainDeferredRetireQueues(false)` を呼んだ後にリエンキューを試行するよう修正:

```cpp
// src/audioengine/AudioEngine.h — enqueueDeferredDeleteNonRtWithResult()
    if (m_retireRouter->enqueueRetire(...) == Success)
        return Success;

    // [P0-5] enqueue failure -> best-effort drain + **retry enqueue**
    drainDeferredRetireQueues(false);
    if (m_retireRouter->enqueueRetire(...) == Success)
        return Success;

    // QueuePressure を RuntimePolicyEngine へ通知
    // → PolicyEngine が Admission PressureLevel 引き上げ等を判断
    const std::uint64_t retireDepth = ...;
    return RetireEnqueueResult::QueuePressure;
```

##### Phase 2: 恒久的 Architecture Improvement（ISR Runtime整合）

- **`DeferredDeletionQueue` の容量拡大**: `kQueueSize`（現状4096）を必要に応じて増加
- **RT-safe オーバーフローリング**: `RefCountedDeferred::release()` の RT パスでも安全に退避できる SPSC オーバーフローリングの導入（`ISRRetireOverflowRing` と同様の設計）
- **QueuePressure → RuntimeHealth → PolicyEngine**: enqueue 失敗を RuntimePolicyEngine へ直接通知せず、既存の `RuntimeHealthMonitor` 経由で伝播する。`RuntimeHealthSnapshot` に retire 関連メトリクス（pendingDepth, overflowCount, reclaimProgress, enqueueFailures）を追加し、HealthMonitor が Pressure 閾値超過を検出した場合に PolicyEngine へ `RuntimePolicyEvent::AdmissionPressure (Level: High)` を発行する。これにより PolicyEngine は Runtime 内部の Queue 実装詳細を知らずに Admission 制御ができる。
- **実測による確認**: `overflowCount()` の値をまず確認。非ゼロなら本バグが実運用で発生している確定的証拠

#### 影響評価
- Phase 1（暫定 Mitigation）: Non-RT確定パスでのみ実施。リグレッションリスク低
- Phase 1（`enqueueDeferredDeleteNonRtWithResult` リエンキュー追加）: `drainDeferredRetireQueues` 後に再度 enqueue を試みるようになる。既存の `QueuePressure` 戻り値に影響なし
- Phase 2（恒久的 Architecture Improvement）: 要設計レビュー。別チケットとして管理推奨

#### 推定工数
- Phase 1（暫定 Mitigation）: 30分（2ファイル・2箇所の変更）
- Phase 2（Architecture Improvement）: 設計により変動（見積もり不可）

---

### P1-C: Bug 7 — `SetImpulse()` ブレース漏れ（Bug 1 と同時実施）

> **Bug 1 と同時実施**: 同じファイル（`MKLNonUniformConvolver.cpp`）・同じ関数（`SetImpulse()`）内の修正であり、1回のレビューでまとめて対応可能。静的解析でも推奨されるパターン。

**重大度**: Low（実害なし確認済み）
**影響**: `if (!l.isImmediate)` にブレースがないため、L0（immediate）で `allocSizes.tailOutputBuf` に誤値が書き込まれる。`freeTracked()` の nullptr チェックで保護されるため実害はない。

#### 修正内容

**ファイル: `src/MKLNonUniformConvolver.cpp`** — `SetImpulse()` 内

```cpp
// 変更前:
        if (!l.isImmediate)
            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);
#endif

// 変更後:
        if (!l.isImmediate)
        {
            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);
#endif
        }
```

#### 影響評価
- 動作変更なし（実害なし確認済み）
- コードの意図が明確になる
- リグレッションリスク: ゼロ

#### 推定工数: 5分（Bug 1 と同時実施で追加コストほぼゼロ）

---

## 3. Phase 2: High（優先対応）

### P2-A: Bug 3 — EQ `fastTanh` しきい値不整合

**重大度**: High
**影響**: サチュレーション有効時、`|x| ∈ (3, 4.5)` で出力が `> 1.0` になるオーバーシュート。スカラー/SIMD経路間で不整合。`x=4.5` で不連続点。

#### 修正方針

**推奨: 専用ヘッダ `src/dsp/math/FastTanhApprox.h` に抽出（共通Utility化のみ）**

> **ISR Runtime 準拠（Single Semantic Source）**: 飽和特性の数値実装を一箇所に集約し、DSPCoreDouble（SoftClip）・EQProcessor（Saturation）・SIMD版・スカラー版のすべてが同一の係数を参照する。専用ヘッダとすることで `DSPMath` 名前空間の肥大化を防ぎ、近似関数ごとに独立したユーティリティとして管理できる。
>
> **⚠️ 注意**: 本 Bug は「Utility 化」のみをスコープとする。**Padé 近似の係数変更（3次/2次→5次/6次）は DSP 仕様変更であり、別チケットとして分離する**。現行の `27/9` 係数はそのまま共通 Utility に移行し、クリップ閾値パラメータ化のみ行う。係数変更は十分な音響評価（THD/Transfer Curve/Preset 比較）を経て別途実施する。

`TanhApprox` の係数（共通ユーティリティに集約。クリップ閾値は呼び出し側が指定）:
```
FastTanhCoefficients { ... }  // 現行 27/9 係数をそのまま移行
```

係数値は現行コードのまま移行し、変更しない。

**新規ファイル: `src/dsp/math/FastTanhApprox.h`**（専用ユーティリティ）

```cpp
// src/dsp/math/FastTanhApprox.h（新規）
#pragma once
#include <immintrin.h>

// ★ DSP Utility 名前空間。将来の FastExpApprox/FastAtanApprox も同一階層に配置
namespace convo::dsp {
    // ★ 係数は detail 名前空間に隠蔽。API を汚染しない
    namespace detail {
        struct FastTanhCoefficients {
            static constexpr double NumA = 10395.0;
            static constexpr double NumB = 1260.0;
            static constexpr double NumC = 21.0;
            static constexpr double DenA = 10395.0;
            static constexpr double DenB = 4725.0;
            static constexpr double DenC = 210.0;
        };
    }

    // Scalar — 近似式とクリップ閾値は分離。閾値は呼び出し側が指定
    inline double fastTanh(double x, double clipThreshold = 4.5) noexcept
    {
        if (x >= clipThreshold) return 1.0;
        if (x <= -clipThreshold) return -1.0;
        const double x2 = x * x;
        const double num = x * (detail::FastTanhCoefficients::NumA
                      + x2 * (detail::FastTanhCoefficients::NumB
                      + x2 * detail::FastTanhCoefficients::NumC));
        const double den = detail::FastTanhCoefficients::DenA
                  + x2 * (detail::FastTanhCoefficients::DenB
                  + x2 * (detail::FastTanhCoefficients::DenC + x2));
        return num / den;
    }

    // SSE2（YAGNI のため AVX2 版は DSPCore 側で必要になった時点で追加）
    inline __m128d fastTanhV128(__m128d x, double clipThreshold = 4.5) noexcept { /* 同一係数を使用 */ }
}
} // namespace convo::dsp
```

**変更が必要なファイル**:

1. **`src/eqprocessor/EQProcessor.Processing.cpp`** — `fastTanhScalarOutput()` / `fastTanhV128Output()` を `convo::dsp::fastTanh()` / `fastTanhV128()` に置換
2. **`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`** — 匿名名前空間 `TanhApprox` の定義を削除し `convo::dsp::fastTanh()` を参照

```cpp
// 変更後: fastTanhScalarOutput
inline double fastTanhScalarOutput(double x) noexcept
{
    constexpr double kClipThreshold = 4.5;
    if (x >= kClipThreshold) return 1.0;
    if (x <= -kClipThreshold) return -1.0;
    const double x2 = x * x;
    // TanhApprox (5th/6th Padé): converges to ~0.99927 at x=4.5
    const double num = x * (10395.0 + x2 * (1260.0 + x2 * 21.0));
    const double den = 10395.0 + x2 * (4725.0 + x2 * (210.0 + x2));
    return num / den;
}
```

```cpp
// 変更後: fastTanhV128Output
inline __m128d fastTanhV128Output(__m128d x) noexcept
{
    constexpr double kClipThreshold = 4.5;
    const __m128d vClipHigh  = _mm_set1_pd(kClipThreshold);
    const __m128d vClipLow   = _mm_set1_pd(-kClipThreshold);
    const __m128d vNumA      = _mm_set1_pd(10395.0);
    const __m128d vNumB      = _mm_set1_pd(1260.0);
    const __m128d vNumC      = _mm_set1_pd(21.0);
    const __m128d vDenA      = _mm_set1_pd(10395.0);
    const __m128d vDenB      = _mm_set1_pd(4725.0);
    const __m128d vDenC      = _mm_set1_pd(210.0);

    const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vClipLow), vClipHigh);
    const __m128d x2 = _mm_mul_pd(xClamped, xClamped);
    const __m128d num = _mm_mul_pd(xClamped, _mm_add_pd(vNumA,
        _mm_mul_pd(x2, _mm_add_pd(vNumB, _mm_mul_pd(x2, vNumC)))));
    const __m128d den = _mm_add_pd(vDenA,
        _mm_mul_pd(x2, _mm_add_pd(vDenB, _mm_mul_pd(x2, _mm_add_pd(vDenC, x2)))));
    return _mm_div_pd(num, den);
}
```

#### 代替案（不採用）: 閾値を3.0に戻す
- 利点: 変更が `kClipThreshold` の値1行のみで最小
- 欠点: Low Shelf +12dB ブースト時に開発者が意図した「過剰クリップ抑制」が後退する
- **上記の理由で不採用。本修正（TanhApprox差し替え）を推奨**

#### 影響評価
- 数値誤差: 従来の `x(27+x²)/(27+9x²)` と `TanhApprox` は `x=3` で一致（ともに 1.0）。`x→0` で一致（ともに `x` に漸近）。中間域で微小差
- サウンド特性: 高次近似により飽和特性がより滑らかになる。意図しないオーバーシュートが解消される
- パフォーマンス: 追加の乗算2回・加算2回・除算は変わらず。実測上の影響は無視できる
- リグレッションリスク: 中。数値差によるEQ特性の微変化が生じ得る。**テイスティングのみでは不足**。以下の測定が必要:
  - THD / THD+N: 新旧の飽和特性差を定量的に評価
  - Transfer Curve: -6〜+6 を 0.01刻みで測定し最大誤差を確認
  - Frequency Response: 飽和特性が周波数特性に与える影響を確認

#### 推定工数: 1時間（実装30分 + テイスティング30分 + 測定30分）

**SIMD版 確認**: EQProcessor は SSE2（`__m128d`）とスカラーの2経路のみ。`__m256d`（AVX2）版は YAGNI のため実装せず、DSPCore 側で必要になった時点で追加する。専用ヘッダには Scalar + SSE2 のみ含め、DSPCoreDouble（SoftClip用AVX2）は従来通り独自実装を維持する。

**Semantic Coupling への注意**: SoftClip と EQ Saturation で同一 Utility を共有すると、将来の独立チューニングが困難になる。そのため、共通化は「今は係数が一致している」という一時的な状態に基づくものとし、以下のいずれかの設計を採用すること:
- **案A（推奨）**: `FastTanhApprox` を係数テンプレート化し、`FastTanhApprox<Params>` で SoftClip と EQ が独立した係数インスタンスを持てるようにする
- **案B（簡易）**: `FastTanhApproxPolicy` 構造体を定義し、呼び出し元がポリシーを注入する。デフォルト値は現行係数とし、将来の変更に備える

---

### P2-B: Bug 4 — `ProgressiveUpgradeThread` FTZ/DAZ 未設定

**重大度**: High
**影響**: バックグラウンドIRリサンプリング（r8brain FIR処理）でデノーマルペナルティによる速度低下 → ウォームアップ時間延長 → 2エンジン共存時間増加 → メモリ削減効果の相殺

#### 修正内容

**ファイル 1: `src/ProgressiveUpgradeThread.cpp`**

インクルード追加 + `run()` 冒頭で FTZ/DAZ 設定:

```cpp
// インクルード追加
#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE

void ProgressiveUpgradeThread::run()
{
    // ★ Bug#4: 他の全バックグラウンドスレッドと同様に FTZ/DAZ を有効化
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    if (affinityManager != nullptr)
        affinityManager->applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    setPriority(Priority::low);
    // ... 以下既存コード ...
}
```

**ファイル 2（必須）: `src/IRDSP.cpp`** — `std::async` ラムダ内で FTZ/DAZ 設定

> ⚠️ **重要**: `std::async(std::launch::async, ...)` で生成される子スレッドの MXCSR（FTZ/DAZ）設定が親スレッドから継承されるかどうかは **MSVC STL / Windows ThreadPool / CRT の実装依存**であり、将来のバージョンで変更される可能性もある。したがって継承の有無を断定せず、**安全側として各ワーカースレッドで明示的に FTZ/DAZ を設定する（二重保護）** 設計とする。`IRDSP::resampleIR()` はチャンネル並列処理に `std::async` を使用しており、このラムダ内部で実際の r8brain リサンプリング（デノーマル典型処理）が行われるため、**`IRDSP.cpp` の非同期ラムダ内でも明示的に FTZ/DAZ を設定する**。

```cpp
// src/IRDSP.cpp — resampleIR() 内 std::async ラムダ
    for (int ch = 0; ch < numCh; ++ch) {
        futures.emplace_back(std::async(std::launch::async, [&, ch]() {
            // ★ Bug#4: 実装依存のため安全側として各ワーカースレッドで明示設定
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
            try {
                auto resampler = std::make_unique<r8b::CDSPResampler>(...);
                // ... 以下既存コード ...
```

ヘッダインクルード追加:
```cpp
#include <xmmintrin.h>   // _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h>   // _MM_SET_DENORMALS_ZERO_MODE
```

#### 影響評価
- デノーマルペナルティが解消される（特に r8brain FIR フィルタ処理で顕著）
- FTZ/DAZ有効化による精度低下: IRの減衰テール処理では問題にならない（-300dB以下は切り捨てても可聴域に影響なし）
- **設定箇所**: `ProgressiveUpgradeThread::run()`（スレッド開始時）+ `IRDSP.cpp` の `std::async` ラムダ内（ワーカースレッド）。`IRConverter::convertFile()` は上記スレッド上で実行されるため二重設定不要
- リグレッションリスク: 極めて低い（全バックグラウンドスレッドで既に実施済みの対策）

#### MXCSR 設定（専用Threadは設定のみ、async側はRAII）

**専用スレッド（`ProgressiveUpgradeThread`）**: スレッド開始時に FTZ/DAZ を設定し、終了までそのまま維持する（RAII 保存＋復元は不要。専用スレッド専有のため復元コストが無駄）。

```cpp
void ProgressiveUpgradeThread::run() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    // ... （スレッド終了まで設定維持）
}
```

**`std::async` ワーカー（`IRDSP.cpp`）**: ThreadPool 実装が変わる可能性があるため、RAII で保存・復元する。

`ScopedMXCSR` クラスは Runtime Utility（Thread 管理の責務）として独立ヘッダ `src/runtime/ScopedMXCSR.h` に定義する。

```cpp
class ScopedMXCSR {
    unsigned int oldCsr;
public:
    ScopedMXCSR() noexcept : oldCsr(_mm_getcsr()) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    ~ScopedMXCSR() noexcept { _mm_setcsr(oldCsr); }
};
```

**方針まとめ**:
- 専用スレッド（ProgressiveUpgradeThread）: RAII不要。設定のみ。
- ThreadPool 利用（std::async）: RAII推奨。`ScopedMXCSR` で保存＋復元。

#### 推定工数: 15分（2ファイル・2箇所の追加：ProgressiveUpgradeThread + IRDSP）

---

## 4. Phase 3: Medium（計画的対応）

### P3-A: Bug 5 — `reclaim()` 先読みスキャンデッドコード（Backlog）

**重大度**: Medium（ただし**現時点では着手非推奨**）
**影響**: `kMaxScan=1024` の先読みスキャンが到達不能。MPMC enqueue による epoch 逆転が発生した場合、先頭エントリが reclaim 不可だと、後続の reclaim 可能なエントリも回収されず、キュー飽和（Bug 2）を誘発しやすくなる。

#### 現状の判断

**この Bug は現在 Backlog 扱いとする。Phase 3 から降格。**

理由:
1. 当該コードが「デッドコード」なのか「将来の scan 実装途中」なのか、現時点では判断できない
2. ISR Runtime において Retire Queue は極めて重要であり、コード整理目的だけで削除するのは危険
3. 先読みロジックが存在しなくても機能上の正しさ（回収漏れがないこと）は維持されている
4. 以下の実測データを取得した上で、初めて着手可否を判断すべき:
   - `overflowCount()` の実測値
   - `QueueDepth`（`pendingRetireCount`）の時系列推移
   - Epoch 分布（キュー内の epoch 値の逆転有無）

#### 着手条件

- [ ] `overflowCount() > 0` が確認された（＝キュー飽和が実際に発生している）
- [ ] キュー内 epoch 逆転が観測された
- [ ] Bug 2 の恒久対応（PolicyEngine 連携）が完了している

上記すべてが満たされた場合のみ、以下の修正を検討する。

#### 修正方針（参考）

**選択肢 A: epoch 逆転が起こらないことを確認 → デッドコード削除（単純化）**

`DeferredDeletionQueue` の全 enqueue が単一スレッドからのみ行われる場合（Serialized enqueue）、epoch 逆転は起こらない。この場合は `kMaxScan` 関連コードを削除し、意図と実装を一致させる。

```cpp
            } else {
                // ★ 先頭が削除不可 → 即座に脱出（FIFO順序保証）
                break;
            }
```

**選択肢 B: epoch 逆転が起こり得る → 先読みロジックの再実装（本格対応）**

先読みで「reclaim 可能だが非先頭」のエントリを発見した場合、dequeue は動かさずに scanPos だけ進めて回収する。ただし **FIFO 順序保証**を崩さない設計が必要（別途リーダブルな設計文書が必須）。

#### 推定工数
- 選択肢A: 10分
- 選択肢B: 設計により異なる（半日〜2日）
- **着手時期**: Bug 2 の恒久対応完了後、実測データを確認してから判断

---

### P3-B: Bug 6 — `delayLineReadAdd()` の `std::abs`（品質改善）

**重大度**: Low（品質改善）
**影響**: `absNoLibm` が同一ファイル内に定義されているにもかかわらず、`delayLineReadAdd()` で `std::abs` を使用。Audio Thread 内での libm 呼び出しのリスクを排除するため統一する。

#### 修正内容

**ファイル: `src/MKLNonUniformConvolver.cpp`** — `delayLineReadAdd()` 内

```cpp
// 変更前:
if (std::abs(gain - 1.0) < 1.0e-12)

// 変更後（2箇所）:
if (absNoLibm(gain - 1.0) < 1.0e-12)
```

#### 影響評価
- MSVC デフォルト設定では `std::abs(double)` は単一の AND 命令にインライン化されるため実害はない
- `/fp:strict` などインライン化が阻害される設定でのみ差異が生じる
- コーディング規約の一貫性を保つ意味での品質改善として修正する
- RT 違反が実際に発生するケースは極めて限定的
- リグレッションリスク: ほぼゼロ（値の等価性は完全に同一）

#### 推定工数: 5分（1ファイル・2箇所）

> **補足**: 現状の `absNoLibm` は `inline double absNoLibm(double x) noexcept`。純粋なビット演算であり、**`[[nodiscard]] inline double absNoLibm(double x) noexcept` への強化を推奨**（`constexpr` は MSVC の `std::bit_cast` 対応状況によってはコンパイルエラーとなる可能性があるため、今回は追加せず。コンパイラ要件を確認後、別途追加可）。

---

## 5. 依存関係・改修順序

```
Bug 1 ─── 独立 ──── (即時着手可)
Bug 2 ─── 独立 ──── (即時着手可)
Bug 3 ─── 独立 ──── (DSPCoreDouble.cpp の TanhApprox を参照)
Bug 4 ─── 独立 ──── (即時着手可)
Bug 5 ─── Bug 2 の補完 ──── (Bug 2 の overflowCount 実測後に判断)
Bug 6 ─── 独立 ──── (他の変更とコンフリクトしない)
Bug 7 ─── 独立 ──── (即時着手可)
```

**推奨着手順**:
1. **Phase A: DSP 修正を先行安定化**
   - Bug 1 + Bug 7（同一ファイル `MKLNonUniformConvolver.cpp`、まとめて1回のレビュー）
   - Bug 6（コード規約統一。同一ファイル内で Bug 1/7 と同時に実施）
   - Bug 4（`ProgressiveUpgradeThread.cpp` + `IRDSP.cpp` + `ScopedMXCSR.h`）— 独立・低コスト・即時恩恵
   - Bug 3（`EQProcessor.Processing.cpp` + `src/dsp/math/FastTanhApprox.h`新規）— 専用ユーティリティ抽出

2. **音響評価**（Bug 3 の THD/Transfer Curve/dy/dx/d²y/dx² 確認）

3. **Phase B: Runtime 修正**（Bug 2 は Runtime 全体に影響するため DSP 安定後）
   - Bug 2 Phase 1（暫定 Mitigation: `DSPLifetimeManager.h` + `AudioEngine.h`）
   - 長時間試験（overflowCount / QueueDepth / Publish Latency 確認）
   - Bug 2 Phase 2（Architecture Improvement: PolicyEngine 連携を含む）

4. **Bug 5**（`DeferredDeletionQueue.h`）— **Backlog。overflowCount 実測後に判断**

5. すべて完了後に統合テスト

---

## 6. リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| Bug 1 パッチのリグレッション | 低 | 低 | DIAG_MKL_MALLOC は非診断時 mkl_malloc に展開されるため |
| Bug 2 リトライ強化による無限ループ | 極低 | 高 | tryReclaim の進展有無で終了判定。進展なし=0件で即 break |
| Bug 3 TanhApprox による音質変化 | 中 | 低 | テイスティング必須。オーバーシュート除去の恩恵が上回る |
| Bug 4 FTZ/DAZ による精度低下 | 低 | 低 | IR減衰テール処理では問題なし |
| Bug 5 先読み削除による回収効率低下 | 低 | 低 | 現状と同じ挙動を維持（デッドコードの確認のみ） |
| Bug 6 std::abs → absNoLibm の値誤差 | 極低 | 低 | ビット単位で同一結果 |
| Bug 7 ブレース追加 | ゼロ | ゼロ | 動作変更なし |

---

## 7. テスト計画

### 8.1 単体テスト（ビルド確認）

| Bug | 確認内容 | 関連テストファイル |
|-----|---------|------------------|
| 1 | 診断ビルドで delayLineBuf 統計が正しく計上される | `RetireGraceSemanticsTests.cpp` |
| 1 | 非診断ビルドでリグレッションなし | 全 `ctest` スイート |
| 2 | retire → reclaim → メモリ解放が正しく行われる | `RetireGraceSemanticsTests.cpp`, `RuntimePublicationCoordinatorTests.cpp` |
| 2 | overflowCount モニタリング | `RuntimeHealthMonitor.cpp`（単体テスト） |
| 3 | fastTanh の入出力が期待範囲内 | `EQProcessorMaxGainTests.cpp` + テイスティング |
| 4 | ProgressiveUpgradeThread 起動 | 結合テスト（`Debug Build + Test`） |
| 5 | reclaim 動作（デッドコード削除後） | `RetireGraceSemanticsTests.cpp` |
| 6 | delayLineReadAdd の加算結果 | 結合テスト |
| 7 | 診断ビルドで allocSizes 整合性 | 結合テスト（診断ビルド） |

**テスト実行コマンド**:
```bash
# 全テスト実行（Debug）
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"

# 特定テストのみ実行
ctest -C Debug -R "RetireGrace|RuntimePublicationCoordinator"

# 診断ビルド用テスト
cmake -S . -B build_diag -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1
cmake --build build_diag --config Debug
cd build_diag && ctest -C Debug --output-on-failure
```

### 8.2 結合テスト

1. **Debug Build 正常性確認**: `Debug Build + Test` タスクを実行し全テストパス
2. **Release Build 正常性確認**: `Release` タスクを実行
3. **診断ビルド**: `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` でビルドし、`NucDiagnosticsSnapshot` の統計値が正しいことを確認

### 8.3 テイスティング + 測定（Bug 3 専用）

- **テイスティング**: Bug 3 修正後、以下の条件でEQ出力を聴取確認:
  - Low Shelf +12dB ブースト + サチュレーション有効
  - 高周波数帯でのサチュレーション特性
  - ブロックサイズ差異による経路分岐（スカラー vs SIMD）での音質一致確認
- **THD / THD+N 測定**: 新旧の飽和特性差を定量的に評価。0dBFS サイン波入力時の THD を比較
- **Transfer Curve 測定**: 入出力特性を -6〜+6 の範囲で 0.01刻みで測定し、新旧の最大誤差を確認
- **dy/dx 微分分析**: Transfer Curve の数値微分（`(y_{i+1} - y_i) / (x_{i+1} - x_i)`）を計算。不連続点がある場合、微分値にスパイクとして現れる。新旧の微分曲線を比較し、不連続の有無を確認
- **SIMD/Scalar ULP 比較**: SIMD経路（`fastTanhV128`）とスカラー経路（`fastTanh`）の出力を全入力範囲で比較し、**max ULP < 2** を確認。ビット一致ではなく ULP 差で評価することで、実用的な一致を保証
- **Monotonicity（単調増加性）確認**: 全区間で `x[i+1] > x[i] → f(x[i+1]) >= f(x[i])` を検証。Padé近似は極稀に局所的な逆転を起こす可能性があるため、Transfer Curve 取得と同時に自動検証する
- **dy/dx 最小勾配確認**: Transfer Curve の数値微分を計算し、`min(dy/dx) > 0`（全区間で正の勾配）を確認。局所逆転がある場合、微分値が負の区間として検出される
- **d²y/dx²（二階微分）評価**: 数値二階微分を計算し、急変点（C2 不連続に近い挙動）の有無を確認。Padé近似の倍音構造に影響する可能性があるため、品質保証として自動評価する
- **NaN / Inf / Denormal 入力試験**: 不正な入力（NaN、±Inf、Denormal 値）を与えた場合の出力を確認。`fastTanh` がクラッシュせず正常範囲（`[-1, 1]`）の値を返すことを確認
- **Frequency Response 測定**: 飽和特性が周波数特性に与える影響を 20Hz〜20kHz で確認

### 8.4 Overflow Test（Bug 2 専用）

Queue を 4096 エントリまで埋めた状態で Reader を停止し、以下の系列を確認:

```text
QueueFull到達 → overflowCount増加 → Recovery（Reader再開）→ 全エントリ回収 → Leakなし
```

**確認項目**:
- `overflowCount()` が正しくインクリメントされること
- `pendingRetireCount()` が Recovery 後に 0 に戻ること
- プロセスメモリに恒久的な増加がないこと（Leak 検出）
- **Publish Latency（P50/P95/P99）**: Queue 圧迫時に「Publish 開始 → Runtime 切替完了」までの時間を計測。**P50（中央値）/ P95 / P99（99パーセンタイル）** を取得し、Admission 制御の副作用を評価する
- **tryReclaim 進展確認**: `tryReclaim()` の戻り値（解放件数）が正しく返ること、および進展がない場合（戻り値=0）に早期 break されることを確認

### 8.5 Bug 1 繰り返し耐久試験

`SetImpulse()` → `freeAll()` のサイクルを **10 万回** 繰り返し、メモリ使用量が一定であることを確認（MKL allocator の断片化まで検出可能）:

```bash
# 診断ビルドで実行
build_diag/ConvoPeq_Standalone --test-repeat-setimpulse 1000
```

**確認項目**:
- `NucDiagnosticsSnapshot` の値が各サイクルで一貫していること
- プロセスメモリ（private bytes）が一定であること（リークなし）
- `lostFreeCount` が 0 であること

### 8.6 MXCSR 確認（Bug 4 専用）

`std::async` で生成された子スレッドの MXCSR レジスタを `_mm_getcsr()` で確認:

```cpp
// 確認コード例
#include <xmmintrin.h>
auto csr = _mm_getcsr();
bool ftz = (csr & _MM_FLUSH_ZERO_ON) != 0;
bool daz = (csr & _MM_DENORMALS_ZERO_ON) != 0;
```

- Thread 開始直後に FTZ/DAZ が期待通り設定されていることを確認
- 1回確認すれば以降のテストは不要

### 8.6 QueuePressure → Admission 抑制 → 回復テスト（Bug 2 Phase 2 設計検証）

以下のサイクルを自動テストで検証:

```text
QueuePressure通知 (RuntimePolicyEvent::QueuePressure)
↓
AdmissionPressureLevel 引き上げ (PolicyEngine)
↓
新規 Publish 抑制
↓
Queue 減少 (pendingRetireCount が低下)
↓
Pressure 解除 → 正常運用復帰
```

**確認項目**:
- QueuePressure 通知後に `AdmissionPressureLevel` が適切に引き上げられること
- Publish 抑制中に新規 publish 要求が正しくブロック／キューイングされること
- Queue 減少後に Pressure が自動解除されること
- 一連のサイクル中にデータ喪失（DSPCore ロスト等）がないこと

### 8.7 実機検証

- 長時間動作（30分以上）でメモリリークがないことをプロセスモニタで確認
- `overflowCount()` の値が非ゼロからゼロになったことを確認（Bug 2）

---

## 8. 付録: ファイル変更一覧

| # | ファイル | 変更内容 | フェーズ | 推定工数 |
|---|---------|---------|---------|---------|
| 1 | `src/MKLNonUniformConvolver.h` | `LayerAllocSizes::delayLineBuf` 追加 | P1-A | 5分 |
| 2 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()` 内: `delayLineBuf` 確保を `DIAG_MKL_MALLOC` に変更 + `allocSizes` 記録 | P1-A | 5分 |
| 3 | `src/MKLNonUniformConvolver.cpp` | `freeAll()` 診断ブランチ: `freeTracked(delayLineBuf, ...)` 追加 | P1-A | 5分 |
| 4 | `src/audioengine/DSPLifetimeManager.h` | `retire()`: tryReclaim 進展ベースのリトライに改善 + QueuePressure通知 | P1-B | 10分 |
| 5 | `src/audioengine/AudioEngine.h` | `enqueueDeferredDeleteNonRtWithResult()`: drain後リエンキュー追加 | P1-B | 10分 |
| 6 | `src/eqprocessor/EQProcessor.Processing.cpp` | `fastTanhScalarOutput()` / `fastTanhV128Output()`: 5次/6次 Padéに差し替え | P2-A | 30分 |
| 7 | `src/ProgressiveUpgradeThread.cpp` | `run()` 冒頭で FTZ/DAZ 設定 | P2-B | 5分 |
| 8 | **`src/IRDSP.cpp`**（⚠️ 必須） | `std::async` ラムダ内で FTZ/DAZ 設定（実装依存のため安全側で明示設定） | P2-B | 10分 |

| 10 | `src/DeferredDeletionQueue.h` | `reclaim()` デッドコード削除（または先読み再実装） | P3-A | 10分〜2日 |
| 11 | `src/MKLNonUniformConvolver.cpp` | `delayLineReadAdd()`: `std::abs` → `absNoLibm`（2箇所） | P3-B | 5分 |
| 11 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()`: `if (!l.isImmediate)` にブレース追加（Bug 1と同時実施） | **P1-C** | 5分 |

**合計推定工数**: Phase 1〜3 で約 3〜5 時間

| Bug | 工数見積（実装） | 備考 |
|-----|:--------------:|------|
| Bug1 | 10分 | 3ファイル・3箇所、既存パターンに従うのみ |
| Bug7 | 3分 | Bug1と同時に実施（同一ファイル） |
| Bug6 | 2分 | 2箇所の置換 + constexpr 追加 |
| Bug4 | 15分 | ProgressiveUpgradeThread + IRDSP.async |
| Bug3 | 1〜2時間 | 専用ヘッダ抽出 + 既存2ファイルの参照切替 + テスト |
| Bug2 | 半日 | コード変更より設計レビューの比重が大きい |
| Bug5 | 保留 | Backlog。overflowCount 実測後に判断 |

---

> **注意**: 本計画書は静的コード解析に基づく。`overflowCount()` の実測値、および各修正後の実機テスト結果に応じて、優先順位・アプローチを適宜調整すること。

---

## 9. 追加FIX項目（ISR Runtime 設計レビューに基づく）

### FIX-01: MKL Thread Local 設定の完全化

**ISR Runtime 設計上の位置付け**: グローバル MKL 状態（`mkl_set_dynamic`）は、ISR Bridge の「副作用の局所化」原則に反する。

#### 現状

| 設定 | スコープ | 現状 | 評価 |
|------|---------|------|------|
| `mkl_set_num_threads_local(1)` | スレッドローカル | ✅ 設定済み | 問題なし（ISR準拠） |
| `mkl_set_dynamic(0)` | プロセスグローバル | ⚠️ 残存 | ISR非準拠。要評価 |

`MKLRealTimeSetup.cpp` のコメント:
```cpp
// mkl_set_dynamic(0) はプロセスグローバル設定。必要性を評価した上で保持・削除を判断する。
mkl_set_dynamic(0);
```

#### 評価結果

**結論: `mkl_set_dynamic(0)` は不要。削除推奨。**

理由:
1. CMake ビルドでは `MKL_THREADING=sequential`（シングルスレッドリンク）が指定されている
2. `mkl_set_num_threads_local(1)` でスレッド数は既に 1 に固定されている
3. `mkl_set_dynamic(0)` の役割は「MKL が実行時にスレッド数を変更するのを禁止する」ことだが、 sequential threading では MKL はスレッドを生成しない
4. プロセスグローバル設定は ISR Runtime の「副作用局所化」原則に反する

#### 対応

```cpp
// 変更前 (MKLRealTimeSetup.cpp):
mkl_set_num_threads_local(1);
mkl_set_dynamic(0);  // ← プロセスグローバル

// 変更後:
mkl_set_num_threads_local(1);
// mkl_set_dynamic(0) は削除: sequential MKL では不要。スレッド数は local 設定で十分。
```

**リスク評価**: 極めて低い。sequential MKL では `mkl_set_dynamic` の有無で MKL の動作は変わらない。

**推定工数**: 5分

---

### FIX-02: LoudnessMeter BS.1770 Annex B サンプルレート対応

**ISR Runtime 設計上の位置付け**: オーディオ品質に関わるパラメータは RuntimeWorld のサンプルレートに追従すべき。

#### 調査結果

**✅ 既に実装済み。追加対応不要。**

`LoudnessMeter::updateCoefficients(double fs)` は RBJ Audio EQ Cookbook 公式を使用したサンプルレート依存係数計算を完全実装している:

```cpp
void LoudnessMeter::updateCoefficients(double fs)
{
    // Stage 2: RLB filter (High-pass, fc=38Hz, Q=0.50)
    //   → RBJ Cookbook 高域通過公式で fs 依存計算
    const double w0 = 2.0 * M_PI * 38.0 / fs;
    // ...

    // Stage 1: Pre-filter (High-shelf, f₀=1500Hz, Q=1/√2, G=+4dB)
    //   → RBJ Cookbook 高域シェルフ公式で fs 依存計算
    constexpr double kPreFreq = 1500.0;
    const double w0 = 2.0 * M_PI * kPreFreq / fs;
    // ...
}
```

`preFilterCoeffs[5]` および `rlbFilterCoeffs[5]` はインスタンスメンバであり、`static constexpr kPreBiquad[5]` / `kRlbBiquad[5]`（48kHz固定値）はデフォルト初期値としてのみ使用される。

**確認された対応サンプルレート**:
- 44.1k / 48k / 88.2k / 96k / 176.4k / 192k — すべて Cookbook 公式で計算可能
- テスト推奨: `libebur128` との比較（44.1k〜192k の6条件）

**残課題**: テストによる数値検証のみ。
```bash
# 推奨テスト手順
# 1. LibEbur128参照実装で各SRの係数を出力
# 2. LoudnessMeter::updateCoefficients() の計算結果と比較
# 3. 誤差 < 1e-12 を確認
```

**推定工数**: テスト含めて 2〜3時間（ユーザー指摘の通り、6条件の検証が必要）

---

### FIX-03: Denormal Architectural Invariant の文書化

**ISR Runtime 設計上の位置付け**: RT スレッドの性能保証（determinism）の前提条件として、denormal 回避を Architectural Invariant に昇格すべき。

#### 現状

- `doc/design-guidelines/denormal-handling.md` 作成済み（2026-07-17）
- プロセス全体 FTZ/DAZ 設定済み（`MKLRealTimeSetup.cpp`）
- `killDenormalV` / `killDenormal` / `sanitizeAndLimit` 等のガード実装済み
- `ScopedNoDenormals` が各所に存在（冗長だが無害）

#### 対応

`doc/Practical Stable ISR Bridge Runtime.md` または新しい Architectural Invariant 文書に以下を追記:

```markdown
## Architectural Invariant: Denormal Avoidance

**適用範囲**: 全 DSPCore パス（Audio Thread / バックグラウンド処理スレッド）

**不変条件**:
- DSP State must never enter denormal region.
- すべての IIR フィルタ状態は、1サンプル処理後に明示的 denormal 除去（killDenormal）を行うこと。
- float 型 IIR では特に積極的な killDenormal を適用すること。
- FTZ/DAZ ハードウェア設定に依存する場合も、コード上の明示的ガードを併置すること（移植性保証のため）。

**違反時の影響**:
- RT スレッドで denormal 演算が発生 → 数十〜数百サイクルのレイテンシペナルティ → 音飛び（xrun）
- バックグラウンドスレッドで denormal 演算 → ウォームアップ時間延長 → 2エンジン共存時間増加 → メモリ増大

**検証方法**:
- CI で `_MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON` および `_MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON` の確認
- 新規 IIR フィルタ追加時のコードレビューで killDenormal の存在を確認
```

**推定工数**: 30分（文書化のみ）

---

## 10. Appendix A: Authority Singularization 監査レポート

### 背景

ISR Runtime 設計では、単に「バグがあるか」ではなく「**Authority が一元化されているか**」が重要である。以下の監査は、`ConvoPeq_bug_report_2026-07-17.md` で「誤検知として除外」または「未精査」とされた項目について、Authority Singuralization の観点から再評価したものである。

### 監査基準

各コンポーネントについて以下を確認:
1. **Authority**: 操作（publish / retire / crossfade）の権限が一箇所に集約されているか
2. **Ownership**: RuntimeWorld 単位で所有されているか（共有されていないか）
3. **Immutability**: RCU 保護対象として immutable か、mutable runtime state か

### 監査結果

#### A-1: `TruePeakDetector`（誤検知#2関連）

| 観点 | 結果 |
|------|------|
| Authority | ✅ `AudioEngine.h:961` で `DSPCore` のメンバ（インスタンス所有） |
| Ownership | ✅ per-World（DSPCore は RuntimeWorld に紐付く） |
| Immutability | ✅ `prepare()` (Message Thread) / `processBlock()` (RT) の分離、共有なし |
| **判定** | **安全。追加改修不要。** |

#### A-2: `ConvolverState`（未精査ファイル関連）

| 観点 | 結果 |
|------|------|
| Authority | ✅ `SafeStateSwapper` が唯一のスワップ Authority |
| Ownership | ✅ RCU 保護対象オブジェクトとして EpochDomain が管理 |
| Immutability | ✅ **Immutable State**（生成後に変更なし） |
| **判定** | **安全。atomic pack 不要。** |

#### A-3: `SafeStateSwapper`

| 観点 | 結果 |
|------|------|
| Authority | ✅ 状態スワップの唯一の Authority |
| Ownership | ✅ Message Thread 所有、RT は読むのみ |
| Immutability | ✅ スワップ対象オブジェクト（ConvolverState 等）は immutable |
| **判定** | **安全。追加改修不要。** |

#### A-4: `IRAnalyzer`（未精査ファイル関連）

| 観点 | 結果 |
|------|------|
| Authority | ✅ UI Thread 専用。RT からアクセスなし |
| Ownership | ✅ UI コンポーネント所有 |
| 数値ガード | ✅ `log()` / `sqrt()` / 除算 のガード確認済み |
| **判定** | **安全。追加改修不要。** |

#### A-5: `NoiseShaperLearner`（未精査ファイル関連）

| 観点 | 結果 |
|------|------|
| Authority | ✅ 学習スレッド専用。RT からアクセスなし |
| 二重停止 | ✅ `std::atomic<bool>` + `stop_token` の二重停止 — 一般的な設計として問題なし |
| **判定** | **安全。追加改修不要。** |

#### A-6: `OutputFilter` / `AllpassDesigner` / `LatticeNoiseShaper` / `FixedNoiseShaper`（未精査ファイル）

| 観点 | 結果 |
|------|------|
| Authority | 全て UI/Message Thread 専用、または DSPCore 内部で完結 |
| RT アクセス | なし |
| **判定** | **安全。追加改修不要。** |

#### A-7: `LoudnessMeter`（未精査ファイル）

| 観点 | 結果 |
|------|------|
| Authority | ✅ `DSPCore` のメンバ（AudioEngine.h:960） |
| Ownership | ✅ per-World |
| BS.1770 準拠 | ✅ `updateCoefficients()` で動的係数計算済み（FIX-02） |
| **判定** | **安全。追加改修不要（テストのみ推奨）。** |

#### A-8: `ConvolverProcessor.*` 群（未精査ファイル、十数ファイル）

| 観点 | 結果 |
|------|------|
| Authority | ⚠️ **部分的不明**。LoaderThread / MixedPhase 等の Authority が DSPLifetimeManager 経由か要確認 |
| Ownership | ✅ per-World（StereoConvolver として DSPCore 内で所有） |
| RT アクセス | ✅ `Get()` / `Add()` のみ |
| **判定** | **要追加調査**: LoaderThread の retire 経路が DSPLifetimeManager 経由か確認が必要 |

#### A-9: `NoiseShaperLearner.cpp`（CMA-ES, 1300行超）

| 観点 | 結果 |
|------|------|
| Authority | ✅ 学習スレッド専用。RT アクセスなし |
| 学習状態 | ✅ 学習完了後に係数を atomic publish |
| **判定** | **安全。追加改修不要。** |

### 総合判定

| カテゴリ | 件数 | 判定 |
|---------|:----:|------|
| 安全（追加改修不要） | **16** | 全 Authority 条件を充足 |
| 軽微な補足推奨 | **2** | LoudnessMeter テスト、TruePeakDetector のWorld所有権明示 |
| 要追加調査 | **1** | ConvolverProcessor.LoaderThread の retire 経路 |
| 計 | **19** | |

> **注意**: 本監査は「誤検知」「未精査」とされた全項目を Authority 観点から再評価した結果、**「不要」と断定できるもの**と**「Authority未確認のため要追加調査」**を明確に区分した。特に `ConvolverProcessor.LoaderThread` の retire 経路については、`DSPLifetimeManager` 経由であることの確認が不足している。ここだけは追加調査を推奨する。

---

## 11. 総合改修項目一覧

### Phase 0: 追加FIX（本レビューで確定）

| # | 項目 | ファイル | 対応 | 工数 |
|---|------|---------|------|:----:|
| F-01 | MKL `mkl_set_dynamic(0)` 削除 | `src/MKLRealTimeSetup.cpp` | プロセスグローバル設定の除去 | 5分 |
| F-02 | LoudnessMeter BS.1770 検証 | `src/LoudnessMeter.cpp` | 6条件のテスト（実装済みの確認） | 2〜3時間 |
| F-03 | Denormal Architectural Invariant | `doc/design-guidelines/` | ISR Runtime 設計書に不変条件追記 | 30分 |

### Phase 1-4: 既存7バグ（変更なし）

Bug 1〜7 の改修項目は前版から変更なし。全12ファイル。

### 要追加調査

| 項目 | 内容 | 優先度 |
|------|------|--------|
| `ConvolverProcessor.LoaderThread` retire 経路 | DSPLifetimeManager 経由の確認 | Medium |

### 総工数（全フェーズ）

| フェーズ | 工数 |
|---------|:----:|
| Phase 0: 追加FIX | 2.5〜3.5時間 |
| Phase 1: Critical | 45分 |
| Phase 2: High | 1.5時間 |
| Phase 3: Medium | 15分 |
| Phase 4: Low | 5分 |
| **合計** | **約 4.5〜7 時間** |
