# ConvoPeq バグ改修計画書

**作成日**: 2026-07-17 | **最終更新**: 2026-07-17 (10回のレビュー反映済み)
**ベース文書**: `ConvoPeq_bug_report_2026-07-17.md`（検証結果反映済み）
**リポジトリ**: <https://github.com/lonewolf-jp/ConvoPeq/tree/main>

---

# Part 1: 設計仕様（実装者向け）

このセクションにはプログラマが実装に必要な全情報を記載する。各バグの修正は**独立して実施可能**であり、記載された順序に従うこと。

---

## 1. 実装優先順位

```
Phase A: DSP安定化（Bug1/7/6/4/3）
  Bug1 + Bug7 → Bug6（同上ファイル）→ Bug4 → Bug3
  ↓ 音響評価
Phase B: Runtime修正（Bug2）
  Phase1 (暫定Mitigation) → 長時間試験 → Phase2 (Architecture Improvement)
  ↓
Bug5: Backlog（overflowCount実測後に判断）
```

### 推奨着手順

1. **Bug 1 + Bug 7**（`MKLNonUniformConvolver.cpp/h` — 同一ファイル・同時レビュー）
2. **Bug 6**（同上ファイル · コード規約統一。Bug 1/7 と同時実施）
3. **Bug 4**（`ProgressiveUpgradeThread.cpp` + `IRDSP.cpp` + `ScopedMXCSR.h` — 独立・低コスト）
4. **Bug 3**（`EQProcessor.Processing.cpp` + `src/dsp/math/FastTanhApprox.h` — ユーティリティ抽出）
5. **音響評価**（Bug 3 の THD/Transfer Curve/dy/dx/d²y/dx² 確認）
6. **Bug 2 Phase 1**（`DSPLifetimeManager.h` + `AudioEngine.h` — 暫定 Mitigation）
7. **長時間試験**（overflowCount / QueueDepth / Publish Latency 確認）
8. **Bug 2 Phase 2**（Architecture Improvement — PolicyEngine 連携）
9. **Bug 5**（Backlog — overflowCount 実測後に判断）

---

## 2. 各バグの実装仕様

### 2.1 Bug 1 — `delayLineBuf` メモリリーク（Critical）

| 項目 | 内容 |
|------|------|
| 影響 | 診断ビルドで IR 再構築のたびに `delayLineCapacity×8` バイトがリーク |
| 原因 | `LayerAllocSizes` に delayLineBuf フィールドなし。解放側診断ブランチに `freeTracked` なし |

#### 変更① `src/MKLNonUniformConvolver.h` — LayerAllocSizes

```cpp
struct LayerAllocSizes {
    // ...既存14フィールド...
    size_t tailOutputBuf = 0;
    size_t delayLineBuf = 0;   // ★ 追加（Bug#1）
};
```

#### 変更② `src/MKLNonUniformConvolver.cpp` — SetImpulse() 確保側

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

#### 変更③ `src/MKLNonUniformConvolver.cpp` — freeAll() 診断ブランチ

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    freeTracked(/* ...既存14行... */);
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
    freeTracked(delayLineBuf,  allocSizes.delayLineBuf);   // ★ 追加
    allocSizes = {};
    // ★ カーソルも同時クリア（解放後オブジェクトの状態完全性）
    delayLineCapacity = 0;  delayWriteCursor = 0;  delayReadCursor = 0;
```

> **`freeTracked()` の動作**: 内部で `mkl_free(ptr)` + `ptr = nullptr` を一括実行後、`allocSizes = {}` でサイズ情報をクリア。この順序（解放+nullptr → サイスクリア）で誤使用を防止。

> **`Layer::reset()` 不存在確認**: `Layer` 構造体に `reset()` メソッドは存在しない。cursor クリアは `freeAll()` 内で行うのが適切。\n>\n> **コーディングルール**: 以後追加される MKL バッファは `mkl_malloc` ではなく **`DIAG_MKL_MALLOC` を使用すること**。`freeAll()` での `freeTracked()` 登録も忘れず行う。これにより診断ビルドの統計漏れを防止する。

| 工数 | リグレッションリスク |
|:----:|:------------------:|
| 10分 | 極低（既存パターンに完全一致） |

---

### 2.2 Bug 7 — `SetImpulse()` ブレース漏れ（Low · Bug 1 と同時実施）

| 項目 | 内容 |
|------|------|
| 影響 | L0 で `allocSizes.tailOutputBuf` に誤値（実害なし） |
| 修正 | `if (!l.isImmediate)` にブレース追加し診断ブロックをガード内に移動 |

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

| 工数 | リスク |
|:----:|:------:|
| 3分（Bug1と同時） | ゼロ（動作変更なし） |

---

### 2.3 Bug 6 — `delayLineReadAdd()` の `std::abs`（品質改善 · Low）

| 項目 | 内容 |
|------|------|
| 影響 | `absNoLibm` が同一ファイルに定義されているにも関わらず `std::abs` を使用 |
| 修正 | `std::abs(gain - 1.0)` → `absNoLibm(gain - 1.0)`（2箇所） |
| 補足 | `absNoLibm` に **`[[nodiscard]] constexpr inline`** を追加（MSVC 2022 / C++20 で確認済み） |

```cpp
// 変更前: inline double absNoLibm(double x) noexcept
// 変更後: [[nodiscard]] inline double absNoLibm(double x) noexcept
```

| 工数 | リスク |
|:----:|:------:|
| 2分 | ほぼゼロ |

---

### 2.4 Bug 4 — `ProgressiveUpgradeThread` FTZ/DAZ 未設定（High）

| 項目 | 内容 |
|------|------|
| 影響 | バックグラウンド IR リサンプリングでデノーマルペナルティ → ウォームアップ延長 → メモリ増大 |
| 原因 | `ProgressiveUpgradeThread::run()` と `IRDSP.cpp` の `std::async` ラムダに FTZ/DAZ 設定なし |

#### 変更① `src/ProgressiveUpgradeThread.cpp`

```cpp
#include <xmmintrin.h>
#include <pmmintrin.h>

void ProgressiveUpgradeThread::run()
{
    // ★ 専用スレッド: 設定のみ（RAII 保存＋復元は不要）
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    // ... 以下既存コード ...
}
```

#### 変更② `src/IRDSP.cpp` — `std::async` ラムダ内

```cpp
    futures.emplace_back(std::async(std::launch::async, [&, ch]() {
        // ★ std::async ワーカー: ThreadPool 実装依存のため RAII（ScopedMXCSR）で保存＋復元
        ScopedMXCSR mxcsr;
        // ... 以下既存コード ...
    }));
```

#### 新規ファイル `src/core/ScopedMXCSR.h`（CPU実行環境設定 Utility）

```cpp
#pragma once
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace convo::cpu {
class ScopedMXCSR final {
    unsigned int oldCsr;
public:
    ScopedMXCSR() noexcept : oldCsr(_mm_getcsr()) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    ~ScopedMXCSR() noexcept { _mm_setcsr(oldCsr); }
    ScopedMXCSR(const ScopedMXCSR&) = delete;
    ScopedMXCSR& operator=(const ScopedMXCSR&) = delete;
};
} // namespace convo::cpu
```

**方針**:
- 専用スレッド（ProgressiveUpgradeThread）: RAII 不要。設定のみ。
- ThreadPool 利用（std::async）: RAII 推奨。`ScopedMXCSR` で保存＋復元。
- **Realtime Audio Thread では MXCSR を書き換えないこと**（Worker Thread / Offline Render Thread を含む全 Audio スレッド）。ScopedMXCSR は IR リサンプリング等のバックグラウンド処理専用。

| 工数 | リスク |
|:----:|:------:|
| 15分 | 極低（全バックグラウンドスレッドで既に実施済み） |

---

### 2.5 Bug 3 — EQ `fastTanh` Utility 化（High）

| 項目 | 内容 |
|------|------|
| 影響 | DSPCoreDouble（SoftClip）と EQProcessor（Saturation）で独立した Tanh 実装 → 保守性低下 |
| スコープ | **Utility 化のみ。係数変更（3次→5次 Padé）は別チケット。** |
| 現行係数 | 現行 `27/9` 係数をそのまま移行。変更しない。 |

#### 新規ファイル `src/dsp/math/FastTanhApprox.h`

```cpp
#pragma once
#include <immintrin.h>

namespace convo::dsp {
    namespace detail {
        // ★ 将来ポリシー追加時に備え、デフォルト係数であることを名前に明示
        struct DefaultFastTanhCoefficients {
            static constexpr double NumA = /* 現行値 */;
            static constexpr double NumB = /* 現行値 */;
            static constexpr double NumC = /* 現行値 */;
            static constexpr double DenA = /* 現行値 */;
            static constexpr double DenB = /* 現行値 */;
            static constexpr double DenC = /* 現行値 */;
        };
    }

    // ★ Policy テンプレート化: 呼び出し側が係数と閾値をポリシーとして注入
    //   SoftClipPolicy, EQSaturationPolicy 等を定義し、Future 拡張に備える。
    template<class Policy = DefaultFastTanhPolicy>
    inline double fastTanh(double x) noexcept
    {
        if (x >= Policy::clipThreshold) return 1.0;
        if (x <= -Policy::clipThreshold) return -1.0;
        const double x2 = x * x;
        using C = typename Policy::Coefficients;
        const double num = x * (C::NumA + x2 * (C::NumB + x2 * C::NumC));
        const double den = C::DenA + x2 * (C::DenB + x2 * (C::DenC + x2));
        return num / den;
    }

    // SSE2（YAGNI: AVX2 版は DSPCore 側で必要になった時点で追加）
    template<class Policy = DefaultFastTanhPolicy>
    inline __m128d fastTanhV128(__m128d x) noexcept { /* 同上 */ }
}
```

#### 変更が必要なファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/eqprocessor/EQProcessor.Processing.cpp` | `fastTanhScalarOutput()` / `fastTanhV128Output()` → `convo::dsp::fastTanh()` / `fastTanhV128()` |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 匿名名前空間 `TanhApprox` → `convo::dsp::fastTanh()` 参照に置換 |

#### Semantic Coupling への注意

SoftClip と EQ Saturation の独立チューニングに備え、以下を検討:
- **案A（推奨）**: 係数テンプレート化 `FastTanhApprox<Params>`
- **案B（簡易）**: `FastTanhApproxPolicy` 構造体で呼び出し元がポリシー注入

| 工数 | リスク |
|:----:|:------:|
| 1〜2時間 | 中（数値変化に伴う音響評価が必要） |

---

### 2.6 Bug 2 — Retire キュー飽和時のポインタロスト（Critical）

#### 現状の保護機構（既存）

- `ISRRetireRouter::enqueueRetire()` 内部: 500ms クールダウン付き `tryReclaim` + 再試行
- `EpochDomain::detectStuckReaders()`: 10秒/30秒閾値の滞留検出 + Quarantine
- `DeferredRetireFallbackQueue`: 別系統フォールバック（ISRRetireRuntime 用）
- `RuntimeHealthMonitor::checkOverflowRate()`: 1秒窓のレート監視
- `AudioEngine::drainDeferredRetireQueues()`: `tryReclaim()` + `coordinator.reclaim()` 2段階

**問題**: 全経路が `DeferredDeletionQueue` のみを使用し、リトライ尽くしても空かない場合ポインタは確実にロストする。

#### Phase 1: 暫定 Mitigation

> **リトライロジックの Authority を `ISRRetireRouter::enqueueWithRetry()` に完全集約する**。DSPLifetimeManager / AudioEngine / EQProcessor の各呼び出し元はリトライループを持たず、Router に委譲する。Phase 2 ではこの集約を前提に QueuePressure → RuntimeHealth → PolicyEngine の経路を完成させる。

**新規メソッド: `src/audioengine/ISRRetireRouter.h/cpp` — `enqueueWithRetry()`**

```cpp
// ★ リトライロジックの Authority: Router に集約。呼び出し元はリトライループ不要。
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                                       DeletionEntryType type) noexcept
{
    auto result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success) return result;

    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        // ★ tryReclaim も Router 内部で完結。呼び出し元は内部実装を意識しない。
        const uint32_t reclaimed = tryReclaim();
        if (reclaimed == 0) break;
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success) return result;
    }

    // ★ QueuePressure 通知責務: Router 内部で RuntimeHealthMonitor へ直接通知する
    //   （呼び出し側は Router が通知することを前提に、戻り値だけで判断する）。
    //   例: runtimeHealth_->notifyQueuePressure(QueuePressureInfo{...});
    //   これにより QueuePressure の通知 Authority も Router に一本化される。
    return RetireEnqueueResult::QueuePressure;
}
```

**変更① `src/audioengine/DSPLifetimeManager.h` — `retire()`（リトライループ削除）**

```cpp
    // ★ リトライは Router に委譲。呼び出し元は enqueueWithRetry を1回呼ぶだけ。
    router_->enqueueWithRetry(static_cast<void*>(dsp), &AudioEngine::destroyDSPCoreNode, epoch,
                              DeletionEntryType::Generic);
    convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount, ...);
```

**変更② `src/audioengine/AudioEngine.h` — `enqueueDeferredDeleteNonRtWithResult()`（リトライループ削除）**

```cpp
    // ★ Router の enqueueWithRetry に委譲。リトライループ不要。
    auto result = m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, DeletionEntryType::Generic);
    if (result == RetireEnqueueResult::Success)
        runtimePublicationBridge_.setRetireBacklogCount(...);
    return result;
```

#### Phase 2: 恒久的 Architecture Improvement（設計方針のみ）

- **QueuePressure → RuntimeHealth → PolicyEngine**: 直接通知せず HealthMonitor 経由
- **新規キュー追加しない**: Overflow Ring 追加は 3 系統化
- **Queue Full → Admission Stop**: Publish 抑制で対処

| Phase | 工数 | リスク |
|-------|:----:|:------:|
| Phase 1（暫定） | 30分 | 低（NonRT のみ） |
| Phase 2（恒久） | 設計による | 要設計レビュー |

---

### 2.7 Bug 5 — `reclaim()` 先読みスキャンデッドコード（Backlog）

**現時点では着手しない。** 以下の実測データ取得後に判断:

- [ ] `overflowCount() > 0` の確認（キュー飽和が実際に発生しているか）
- [ ] キュー内 epoch 逆転の観測
- [ ] Bug 2 の恒久対応完了

---

## 3. ファイル変更一覧

| # | ファイル | 変更内容 | フェーズ | 工数 |
|---|---------|---------|---------|:----:|
| 1 | `src/MKLNonUniformConvolver.h` | `LayerAllocSizes::delayLineBuf` 追加 | P1-A | 5分 |
| 2 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()`: delayLineBuf 確保を `DIAG_MKL_MALLOC` に | P1-A | 5分 |
| 3 | `src/MKLNonUniformConvolver.cpp` | `freeAll()`: `freeTracked(delayLineBuf,...)` 追加 | P1-A | 5分 |
| 4 | `src/MKLNonUniformConvolver.cpp` | `SetImpulse()`: `if (!l.isImmediate)` ブレース修正 | P1-C | 3分 |
| 5 | `src/MKLNonUniformConvolver.cpp` | `delayLineReadAdd()`: `std::abs` → `absNoLibm`（2箇所） | P3-B | 2分 |
| 6 | `src/audioengine/ISRRetireRouter.h/cpp`（変更） | `enqueueWithRetry()` 追加（retryロジック集約 + QueuePressure通知） | P1-B | 15分 |
| 7 | `src/audioengine/DSPLifetimeManager.h` | `retire()`: enqueueWithRetry 委譲（リトライループ削除） | P1-B | 5分 |
| 8 | `src/audioengine/AudioEngine.h` | `enqueueDeferredDeleteNonRtWithResult()`: enqueueWithRetry 委譲（リトライループ削除） | P1-B | 5分 |
| 9 | `src/ProgressiveUpgradeThread.cpp` | `run()` 冒頭で FTZ/DAZ 設定 | P2-B | 5分 |
| 10 | `src/IRDSP.cpp`（必須） | `std::async` ラムダ内で FTZ/DAZ 設定 | P2-B | 10分 |
| 11 | `src/core/ScopedMXCSR.h`（新規） | RAII MXCSR ユーティリティ（CPU実行環境設定） | P2-B | 5分 |
| 12 | `src/dsp/math/FastTanhApprox.h`（新規） | Tanh 共通 Utility | P2-A | 15分 |
| 13 | `src/eqprocessor/EQProcessor.Processing.cpp` | fastTanh → convo::dsp::fastTanh 参照切替 | P2-A | 15分 |
| 14 | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | TanhApprox → convo::dsp::fastTanh 参照切替 | P2-A | 10分 |

**合計**: 14 ファイル / 約 3.5〜5.5 時間

---

## 4. テスト計画（実装者向け）

### 4.1 ビルド確認

```bash
# 全テスト（Debug）
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"
# 特定テスト
ctest -C Debug -R "RetireGrace|RuntimePublicationCoordinator"
# 診断ビルド
cmake -S . -B build_diag -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1
cmake --build build_diag --config Debug
cd build_diag && ctest -C Debug --output-on-failure
```

### 4.2 Bug 1 耐久試験

`SetImpulse()` → `freeAll()` を **10 万回** 繰り返しメモリ一定を確認:

```bash
build_diag/ConvoPeq_Standalone --test-repeat-setimpulse 100000
```

### 4.3 Bug 2 Overflow Test

Queue 4096 エントリ充填 → Reader 停止 → QueueFull → Recovery の完全シナリオ。
計測: overflowCount, pendingRetireCount, Publish Latency (P50/P95/P99), reclaimProgress

**耐久試験**: Recovery → 再充填 → Recovery のサイクルを複数回（例: 10回）繰り返し、Epoch 系バグが累積しないことを確認。さらに各 Recovery 後に **Publish を挟むシナリオ**（Recovery → Publish → Recovery → Publish → ...）も追加し、Runtime の状態遷移を含めて検証する。

**Shutdown シナリオ試験**: QueuePressure 発生状態から Plugin Close → Shutdown → Retire Drain の一連の流れを確認:

**Shutdown + Epoch 変動試験**: Shutdown 中に Reader が解放され Epoch が変化するケース:
```text
QueuePressure
↓
Shutdown 開始
↓
Shutdown 途中で Reader 解放 → Epoch 変化
↓
Retire 再開（変化後の Epoch で reclaim 可能に）
↓
Drain 完了
```
このシナリオは ISR Runtime の Epoch ベース reclamation が Shutdown 中でも正しく機能することを確認する。
```text
QueuePressure 発生
↓
Plugin Close 要求
↓
Shutdown FSM: StopAccepting → StopAudio → DrainIntent → DrainRetire
↓
EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete
↓
全 Retire エントリが drain され、データ喪失がないこと
```
このシナリオは Shutdown 系 regressions を防止する。

### 4.4 Bug 3 音響評価

- THD / THD+N | Transfer Curve (-6〜+6 / 0.01刻み) | dy/dx / d²y/dx²
- SIMD/Scalar ULP (max < 2) | Monotonicity | NaN/Inf/Denormal 入力試験 | Frequency Response

### 4.5 Bug 4 MXCSR 確認

`_mm_getcsr()` で子スレッドの FTZ/DAZ 状態を確認（1回のみ）。

### 4.6 Bug 2 Phase 2 設計検証

QueuePressure → AdmissionPressure 引上げ → Publish 抑制 → Queue 減少 → Pressure 解除の全サイクル。

---

# Part 2: 未確定事項・保留事項

## 2.1 実測が必要な項目

| 項目 | 目的 | 判定基準 | 状態 |
|------|------|---------|:----:|
| `overflowCount()` 実測 | Bug 2 の実運用発生確認 | 非ゼロなら確定的証拠 | ⏳ 未実測 |
| QueueDepth 時系列推移 | Bug 5 の epoch 逆転検出 | 時系列で観測 | ⏳ 未実測 |
| Publish Latency (P50/P95/P99) | Admission 制御の副作用評価 | ベースライン対比 | ⏳ 未実測 |
| `_mm_getcsr()` 確認 | Bug 4 std::async MXCSR 継承状況 | 1回のみ | ⏳ 未確認 |

> **注意**: 上記はすべて**実機で計測が必要**な項目であり、コード解析のみでは確定できない。実装 Phase 完了後に計測し、計画書の優先順位・方針を適宜調整すること。

## 2.2 コンパイラ依存の確認項目（調査結果反映済み）

| 項目 | 懸念 | 調査結果 | 確定判断 |
|------|------|---------|:--------:|
| `std::bit_cast` の constexpr | MSVC で constexpr 未対応の可能性 | C++20 標準で constexpr 対応。MSVC 2019 16.10+（ConvoPeq は MSVC 2022）で完全サポート。`double`↔`uint64_t` は制約条件（非ポインタ・非共用体）に合致。 | ✅ 対応済み。**`[[nodiscard]] constexpr inline double absNoLibm(double x) noexcept` に更新可**。ただし現在のサポート対象コンパイラ（MSVC 2022 / C++20）を前提とする。将来コンパイラ要件変更時には再確認が必要。 |
| `std::async` MXCSR 継承 | MSVC STL/CRT/ThreadPool 実装依存 | Windows x64 では新規スレッドは独立した MXCSR を持つが、「std::async が継承しない」ことは C++ 標準で保証されていない（スレッドプール実装は処理系依存）。したがって**継承を前提にせず、毎回明示設定する**設計が適切。 | ✅ 確定。継承を前提にしないため毎回明示設定。 |
| `constexpr inline` の absNoLibm | ビット演算の constexpr 可否 | `std::bit_cast` が constexpr 対応のため、`absNoLibm` も constexpr 化可能。ただしコンパイルテストでの確認を推奨。 | ✅ 安全に追加可能。コンパイルテストで確認後、`constexpr` 追加。 |

## 2.3 要追加調査（調査結果反映済み）

| 調査項目 | 事前の懸念 | 調査結果 | 確定判断 |
|---------|-----------|---------|:--------:|
| `ConvolverProcessor.LoaderThread` retire 経路 | DSPLifetimeManager 経路の確認不足 | `LoaderThread` → `owner.retireStereoConvolver()` → `StereoConvolver::retireStereoConvolver()` → `AudioEngine::enqueueDeferredDeleteNonRt()` → `m_retireRouter->enqueueRetire()` → `ISRRetireRouter` → `EpochDomain` → `DeferredDeletionQueue` の経路を確認。内部で `markRetireEpoch()` を呼び epoch も正しく設定される。 | ✅ **経路確認完了。DSPLifetimeManager は経由しないが、適切な epoch 経由の retire パスが確保されている。追加調査不要。** |

---

# Part 3: Appendix

## 3.1 設計改訂履歴

| 回 | 主な変更点 |
|:--:|-----------|
| 1 | 初版作成（バグレポート検証に基づく7バグ計画） |
| 2 | Bug2 PolicyEngine 言及追加、Bug3 係数訂正、Bug5 Backlog 化 |
| 3 | Bug2 進展ベース終了条件、Bug3 共通ユーティリティ化、テスト計画拡充 |
| 4 | Bug2 1回+QueuePressure 単純化、Bug3 CLIP_THRESHOLD パラメータ化、ULP 比較追加 |
| 5 | Bug3 `FastTanhApprox.h` 専用ヘッダ化、Bug4 IRConverter 削除、Monotonicity 追加 |
| 6 | Bug2 進展ベース確定、Bug3 `convo::dsp` namespace、Bug4 MXCSR RAII 明確化 |
| 7 | Bug2 リトライ上限 kMaxRetry=8、Bug3 係数 detail 隠蔽、`[[nodiscard]]` 追加 |
| 8 | Bug2 kMaxRetry=2、QueuePressureInfo 拡充、d²y/dx² テスト |
| 9 | Bug2 QueuePressure→Health 経由、Bug3 Utility/係数変更分離、専用Thread RAII 不要 |
| 10 | Bug2 `enqueueWithRetry()` 集約、Semantic Coupling 対策、10万回耐久試験 |
| — | **文書3部構成に再編**（設計/未確定事項/Appendix） |

## 3.2 リスク評価

| リスク | 確率 | 影響 | 対策 |
|--------|:----:|:----:|------|
| Bug1 パッチのリグレッション | 低 | 低 | DIAG_MKL_MALLOC は非診断時 mkl_malloc に展開 |
| Bug2 リトライでの長時間ループ | 極低 | 高 | tryReclaim 進展なしで即 break + kMaxRetry=2 |
| Bug3 音質変化 | 中 | 低 | テイスティング + THD/Transfer Curve 測定必須 |
| Bug4 FTZ/DAZ 精度低下 | 低 | 低 | IR 減衰テールでは問題なし |
| Bug5 デッドコード削除 | 低 | 低 | 現状と同じ挙動を維持 |
| Bug6 absNoLibm 値誤差 | 極低 | 低 | ビット単位で同一結果 |
| Bug7 ブレース追加 | ゼロ | ゼロ | 動作変更なし |

## 3.3 追加 FIX 項目（ISR Runtime 設計レビュー）

### FIX-01: MKL `mkl_set_dynamic(0)` 削除

`MKL_THREADING=sequential` + `mkl_set_num_threads_local(1)` では不要。

### FIX-02: LoudnessMeter BS.1770 サンプルレート対応（✅ 既に実装済み）

`LoudnessMeter::updateCoefficients(double fs)` は RBJ Cookbook 公式で動的計算済み。6条件のテストのみ推奨。

### FIX-03: Denormal Architectural Invariant 文書化

ISR Runtime 設計書へ "DSP State must never enter denormal region" の不変条件を追記。

## 3.4 Authority Singularization 監査レポート（概要）

「誤検知」「未精査」とされた全19項目を Authority 観点から再評価:

| カテゴリ | 件数 | 判定 |
|---------|:----:|------|
| 安全（追加改修不要） | 16 | 全 Authority 条件を充足 |
| 軽微な補足推奨 | 2 | LoudnessMeter テスト、TruePeakDetector World所有権明示 |
| 要追加調査 | 1 | `ConvolverProcessor.LoaderThread` retire 経路 |
| **計** | **19** | |

詳細は旧版 `ConvoPeq_fix_plan_2026-07-17_old.md` の Appendix A を参照。

## 3.5 検証結果の概要

`ConvoPeq_bug_report_2026-07-17.md` に記載された全7バグについて、実コード検証の結果、全件が正確な指摘であることを確認。誤検知とされた5項目もすべて「問題なし」を確認済み。
