# メモリ占有調査のためのインストルメンテーション改修案 v39（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v38 からの変更**: 1 点の設計修正 + 2 点の改善。

---

## 0. v38 の問題点と v39 での修正方針

| # | v38 の問題 | v39 の修正 |
|:--|:----------|:----------|
| 1 | `freeTracked(size==0)` が `lostFreeCount` と `zeroAllocSizeCount` の両方を増やす → 両者の意味論が混在。`lostFreeCount` は「解放サイズが不明なケース」、`zeroAllocSizeCount` は「allocSizes 保存漏れ」と分離すべき | **`freeTracked(size==0)` は `zeroAllocSizeCount` のみ増加。`lostFreeCount` は `diagMklFree(size==0)` のみで増加（本来の意味に限定）** |
| 2 | `diagLogNonRt()` に RT 禁止の注釈なし | **コメント追加: "Never call from audio callback (RT)"** |
| 3 | `kFirstRuntimeDiagSeq` がない（`fetch_add(1) + 1` の `+1` の意味が暗黙的） | **`kFirstRuntimeDiagSeq = 1` を追加** |

---

## 1. Patch A: DiagnosticsConfig.h — lostFreeCount と zeroAllocSizeCount の役割分離

### A-1. freeTracked — zeroAllocSizeCount のみ加算（★ 設計修正）

```cpp
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
        {
            DIAG_MKL_FREE(p, size);
        }
        else
        {
            // ★ v39: size==0 は allocSizes 保存漏れ → zeroAllocSizeCount のみ増加。
            //   lostFreeCount は変更しない（diagMklFree(size==0) のみが増加）。
            //   これにより「解放サイズ不明」と「診断コードの不整合」が分離可能。
            mklStats().zeroAllocSizeCount.fetch_add(1, std::memory_order_relaxed);
            mkl_free(p);
        }
        p = nullptr;
    }
}
```

### A-2. diagMklFree — lostFreeCount の増加（本来の意味に限定）

`diagMklFree` は既に size==0 の場合に `lostFreeCount` を増加させる。変更なし。

```cpp
inline void diagMklFree(void* ptr, size_t size, ...) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);
        if (size > 0)
        {
            mklStats().allocatedBytes.fetch_sub(...);
            mklStats().totalFreedBytes.fetch_add(...);
        }
        else
        {
            mklStats().lostFreeCount.fetch_add(1, ...);  // ★ これのみが lostFreeCount を増やす
            DBG("[DIAG] diagMklFree size=0 at ...");
        }
    }
}
```

### A-3. diagLogNonRt — RT 禁止コメント追加

```cpp
/// ★ v39: 非 RT スレッドからの診断ログ出力。
///   Never call from audio callback (RT).
///   現在は juce::Logger::writeToLog を直接呼ぶ。
///   将来 Worker Thread 対応が必要な場合、この関数のみ修正すればよい。
inline void diagLogNonRt(const juce::String& message) noexcept
{
    juce::Logger::writeToLog(message);
}
```

---

## 2. Patch B: MKLNonUniformConvolver — kFirstRuntimeDiagSeq

### B-1. MKLNonUniformConvolver.h

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;
    static std::atomic<uint64_t> globalDiagSeq;

    /// デストラクタ等、SetImpulse 以外の経路で使用される seq 値。
    static constexpr uint64_t kReservedDiagSeq = 0;
    /// ★ v39: SetImpulse の最初の seq 値。fetch_add(1) に加算して 1 開始とする。
    static constexpr uint64_t kFirstRuntimeDiagSeq = 1;

    // ...
};
```

### B-2. SetImpulse — kFirstRuntimeDiagSeq 使用

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed)
                           + kFirstRuntimeDiagSeq;
    // ...
#endif
    // ...
}
```

---

## 3. 全ログフォーマット（v39 確定版、v38 から変更なし）

### IR_RELEASE / IR_LOAD / IR_LAYOUT / MEM_SNAP（v38 と同一）

---

## 4. v38 からの改善点一覧

| # | v38（問題） | v39（修正） |
|:--|:----------|:-----------|
| 1 | `freeTracked(size==0)` が `lostFreeCount` と `zeroAllocSizeCount` の両方を増加 → 両者の意味論が混在。「解放サイズ不明(lostFreeCount)」と「診断コード不整合(zeroAllocSizeCount)」が区別不可 | **`freeTracked(size==0)` は `zeroAllocSizeCount` のみ増加。`lostFreeCount` は `diagMklFree(size==0)` のみで増加（役割完全分離）** |
| 2 | `diagLogNonRt()` に RT 禁止の注釈なし | **"Never call from audio callback (RT)" コメント追加** |
| 3 | `fetch_add(1)+1` の `+1` の意味が暗黙的 | **`kFirstRuntimeDiagSeq = 1` 定数化。意味が明示的に** |
