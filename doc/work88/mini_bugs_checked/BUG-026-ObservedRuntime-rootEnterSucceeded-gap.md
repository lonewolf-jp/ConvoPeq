# BUG-026: ObservedRuntime::get() が rootEnterSucceeded() を確認しない

## 発見日
2026-07-26

## ファイル
`src/core/ObservedRuntime.h:42-49 (get)`
`src/core/RCUReader.h:36-83 (enter)`

## 問題
`ObservedRuntime::get()` は Release ビルドにおいて、RCU の reader 参加が実際に成功したかどうかを
確認せずに ptr を返す。RCU 保護なしで返されたポインタは、Deferred Free Thread により
いつでも解放される可能性がある。

### RCUReader::enter() の失敗経路

`RCUReader::enter()` には 2 つのサイレント失敗経路がある:

**経路1 — スレッド競合（RCUReader.h:51-60）:**
```cpp
// CAS 失敗 = 別スレッドが既にこの RCUReader を所有
if (!convo::compareExchangeAtomic(ownerThreadToken,
                                  expectedOwner, threadToken,
                                  std::memory_order_acq_rel,
                                  std::memory_order_acquire)
    && expectedOwner != threadToken)
{
    convo::fetchSubAtomic(nestingDepth, 1, std::memory_order_acq_rel);
    return;  // ← エポック保護なしで return！
}
```

**経路2 — スロット不足（RCUReader.h:62-82）:**
```cpp
const int tid = acquireThreadSlot();
if (tid >= 0) {
    epochProvider->enterReader(tid);
    rootEnterSucceeded_ = true;
} else {
    rootEnterSucceeded_ = false;  // ← 失敗を記録
    convo::fetchSubAtomic(nestingDepth, 1, ...);
    ...  // return（エポック保護なし）
}
```

### ObservedRuntime::get() の現状

```cpp
const GlobalSnapshot* get() const noexcept
{
#ifndef NDEBUG
    if (ownerThreadId != std::this_thread::get_id())
        return nullptr;  // Debug のみチェック
#endif
    return ptr;  // Release: 常に ptr を返す（RCU 保護状態を無視）
}
```

`guard` メンバ（`RCUReaderGuard`）は存在するが、`enter()` の成功を確認していない。
`RCUReader::rootEnterSucceeded()`（P1-A で追加された防御層）を呼び出していない。

### リスク
- Release ビルドで enter() が失敗した場合、get() が返す ptr は RCU 保護されていない
- Deferred Free Thread がポインタの指す先を解放する可能性がある（Use-After-Free）
- 実際のところ、正しいスレッドからのみ ObservedRuntime が作成される前提で動いているため、
  現状のコードではこのパスは通らない — しかし防御層としてのチェックが欠落している

### リスク評価
- **重大度**: MEDIUM — プログラミングエラー（間違ったスレッドからの呼び出し）時の防御層不足
- **発生頻度**: 低（現在のコードベースでは正しいスレッドからのみ呼ばれる）
- **影響**: Release ビルドでの Use-After-Free（クラッシュまたは silent メモリ破壊）

### 修正方針
`ObservedRuntime::get()` で `guard` 経由で `rootEnterSucceeded()` を確認する:
```cpp
const GlobalSnapshot* get() const noexcept
{
    if (!guard.rootEnterSucceeded())
        return nullptr;
    return ptr;
}
```
