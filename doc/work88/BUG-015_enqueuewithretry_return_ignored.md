# BUG-015: enqueueWithRetry の戻り値無視による QueuePressure サイレントドロップ

- **発見日**: 2026-07-26
- **カテゴリ**: リソースリーク / エラーハンドリング
- **関連**: ISRRetireRouter.cpp, SnapshotCoordinator.cpp
- **リスク**: MEDIUM
- **修正**: 未

## 概要

`enqueueWithRetry()` は `RetireEnqueueResult` を返し、`QueuePressure` を返すことで
「リトライ上限到達」を通知する設計になっている。しかし、いくつかの NonRT 呼び出し元で
戻り値が無視されており、QueuePressure 状態がサイレントにドロップされる。

これにより、Retire Queue が枯渇した場合、オブジェクトが永久に解放されない
メモリリークが発生する可能性がある。

## 該当箇所

### 1. ISRRetireRouter::retire（NonRT）

**`ISRRetireRouter.cpp:148-155`**

```cpp
// ★ R-1: IRetireRouter::retire — リトライ込み（NonRT）
void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return;
    (void)enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), DeletionEntryType::Generic);
    // ↑ (void) キャストで QueuePressure がサイレントドロップ
}
```

### 2. SnapshotCoordinator::startFade（NonRT Timer）

**`SnapshotCoordinator.cpp:33-38`**

```cpp
GlobalSnapshot* oldTarget = m_slots.exchangeTarget(target, std::memory_order_acq_rel);
if (oldTarget) {
    const uint64_t retireEpoch = m_epochProvider->currentEpoch();
    // [work37 Phase 1.2] enqueueWithRetry を使用（startFade は NonRT Timer からのみ）
    enqueueWithRetry(*m_epochProvider, oldTarget, snapshotDeleter, retireEpoch);
    // ↑ 戻り値無視
}
```

### 3. SnapshotCoordinator::completeFade（NonRT Timer）

**`SnapshotCoordinator.cpp:86-89`**

```cpp
GlobalSnapshot* old = m_slots.exchangeCurrent(target, std::memory_order_acq_rel);
if (old)
    // [work37 Phase 1.2] enqueueWithRetry を使用（completeFade は NonRT）
    enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);
    // ↑ 戻り値無視
```

## 問題の詳細

### enqueueWithRetry の仕様

**`ISRRetireRouter.cpp:158-183`**

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(...) noexcept
{
    // 1. 通常の enqueue を試行
    auto result = enqueueRetire(...);
    if (result == RetireEnqueueResult::Success)
        return result;

    // 2. 追加リトライ: tryReclaim → enqueue（最大 2 回）
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();
        result = enqueueRetire(...);
        if (result == RetireEnqueueResult::Success)
            return result;
        if (result != RetireEnqueueResult::QueuePressure)
            break;
    }

    // 3. 全リトライ失敗 → QueuePressure
    return RetireEnqueueResult::QueuePressure;
}
```

`QueuePressure` は「3回試行しても Retire Queue が満杯」を意味する。
この状態では、渡されたポインタは **永久に解放されない**。

### 比較：正しく戻り値をチェックしている箇所

**`EQProcessor.Core.cpp:61-62`** — 正しい実装

```cpp
result = stackRouter.enqueueWithRetry(ptr, deleter, retireEpoch, DeletionEntryType::Generic);
return result == convo::isr::RetireEnqueueResult::Success;
```

**`ISRRuntimePublicationCoordinator.cpp:147-149`** — 正しい実装

```cpp
const auto result = router.enqueueWithRetry(...);
if (result != RetireEnqueueResult::Success)
    return result;
```

## 影響

- **ISRRetireRouter::retire**: NonRT スレッドからの退役リクエストがサイレントに失敗
- **SnapshotCoordinator**: 旧スナップショットが解放されず、メモリリーク
- 通常時: Retire Queue は空きがあるため問題は顕在化しない
- 負荷時: 大量の退役リクエストで Queue が溢れ、漏洩が活性化

## 修正案

### オプション 1: 戻り値をチェックし、失敗時にログ

```cpp
auto result = enqueueWithRetry(...);
if (result != RetireEnqueueResult::Success) {
    // RuntimeHealthMonitor へ通知
    // または直接 delete（NonRT スレッドなので安全）
}
```

### オプション 2: enqueueWithRetry 自体を void 化し、内部でフォールバック

```cpp
void enqueueWithRetry(...) noexcept {
    auto result = enqueueRetire(...);
    if (result == RetireEnqueueResult::Success) return;
    // リトライ...
    if (result != RetireEnqueueResult::Success) {
        // QueuePressure → 直接 delete（NonRT スレッドなので安全）
        deleter(ptr);
    }
}
```

## 補足

BUG-010（retireEQStateDeferred/retireBandNodeDeferred）とは異なり、
ここでは `enqueueWithRetry` 自体の戻り値を無視している。
`enqueueWithRetry` は既に内部でリトライを行っているため、
QueuePressure は「本当にどうしようもない」状態である。
