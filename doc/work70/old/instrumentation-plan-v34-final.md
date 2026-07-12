# メモリ占有調査のためのインストルメンテーション改修案 v34（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v33 からの変更**: ソースコード調査で確定した 7 項目を反映。v33 は設計変更なし。

---

## 0. ソースコード調査で確定した未確定事項

| # | 調査項目 | 結果 | 設計への影響 |
|:--|:--------|:-----|:------------|
| 1 | `Layer m_layers[3]` 未使用要素の初期化状態 | **全 pointer = nullptr, allocSizes = 0（値初期化）** → `freeAll()` ループ安全 | 変更不要 |
| 2 | `DiagnosticsConfig.h` インクルード | **`<cstdint>`, `<atomic>` 既存。`<mutex>`, `<unordered_map>` は不要（v24 で削除済み）** | 変更不要 |
| 3 | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の使用パターン | **全既存コードで `#if`（`#ifdef` ではない）** → `#if` を統一使用 | 仕様書で `#if` と明記 |
| 4 | NUC コンストラクタの liveCount | **現状は `liveCount` 未追加** → ctor で `fetch_add(1)` が必要。dtor は `fetch_sub(1)` + `jassert(old>0)` | ctor/dtor 変更 |
| 5 | SetImpulse エラー経路の allocSizes 整合性 | **エラー経路でも `releaseAllLayers()` 呼び出しあり。`allocSizes` は各確保直後に保存・未確保は 0 のため安全** | 変更不要（確認済み） |
| 6 | mkl_malloc 戻り値キャスト | **`static_cast<double*>(mkl_malloc(...))` → `static_cast<double*>(DIAG_MKL_MALLOC(...))` で同様に動作** | 変更不要 |
| 7 | `DiagnosticsConfig.h` の `jassert`/`DBG` 依存 | **JUCE 未インクルード状態で使用される可能性 → `<cassert>` の `assert()` に変更推奨** | **`jassert` → `assert` に変更** |

---

## 1. 調査結果を反映した修正箇所

### 1-1. NUC コンストラクタ/デストラクタ（★ liveCount 追加）

```cpp
MKLNonUniformConvolver::MKLNonUniformConvolver()
{
    mkl_set_num_threads(1);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
#endif
    releaseAllLayers();
}
```

### 1-2. DiagnosticsConfig.h — jassert を assert に変更（★ DiagnosticsConfig.h は JUCE 非依存に）

```cpp
// DiagnosticsConfig.h 内の freeTracked と addIfAlive:
#include <cassert>

template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        assert(size != 0);  // ★ jassert ではなく標準 assert を使用
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}

static uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* name) noexcept
{
    if (ptr)
    {
        if (allocSize == 0)
        {
            // ★ DBG ではなく標準出力（または #if で分岐）
            //   実際の実装では fprintf や OutputDebugString を使用
        }
        return allocSize;
    }
    return 0;
}
```

**補足**: `DBG()` は JUCE マクロ。`DiagnosticsConfig.h` を JUCE 非依存にするため、`addIfAlive` 内の `DBG` もプリプロセッサ分岐で代替する:
```cpp
#if defined(JUCE_DEBUG) && JUCE_DEBUG
    DBG("[DIAG] addIfAlive: " << name << " size=0");
#else
    // Debug ビルドの標準出力（無視しても診断価値に影響なし）
#endif
```

---

## 2. 全ログフォーマット（v34 確定版、v33 から変更なし）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only: sizeof tracked entries, not actual heap) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 3. 出力例（v34 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 4. 実装手順（v34 確定版）

| 手順 | ファイル | 変更内容 | 行数目安 |
|:----|:--------|:--------|:---------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 | +2行 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked(assert使用) + addIfAlive + updateAtomicMaximum64 + DIAG_MKL_* マクロ + computeOtherPrivate | ~75行 |
| 3 | `src/MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot(拡張版) + liveCount(static) + globalDiagSeq(static) + getDiagnostics | ~30行 |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor liveCount.fetch_add(1) + dtor fetch_sub(1)/jassert + globalDiagSeq定義 + 全 mkl_malloc→DIAG_MKL_MALLOC(28) + allocSizes保存 + freeAll freeTracked + releaseAllLayers freeTracked + 無名名前空間 logIrRelease + IR_RELEASE/IR_LOAD/IR_LAYOUTログ | ~165行 |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` | ~5行 |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `m_trackedPendingEntries_` + `trackedRatio()` | ~15行 |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | enqueueRetire/tryReclaim での m_pendingRetireBytes_ 更新 | ~15行 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ | ~25行 |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore liveCount + 確保量ログ | ~15行 |

**合計: 9 ファイル変更 / 約 347 行追加**
**変更不要**: `ConvolverProcessor.h`（6 引数のまま動作）

---

## 5. v33 からの改善点一覧

| # | 調査結果 | 設計への反映 |
|:--|:--------|:------------|
| 1 | `Layer m_layers[3]` 未使用要素は値初期化済み | 確認のみ（変更不要） |
| 2 | `DiagnosticsConfig.h` に `<mutex>`/`<unordered_map>` 不要 | 確認のみ（変更不要） |
| 3 | `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 統一 | 仕様書で `#if` 使用を明記 |
| 4 | NUC ctor/dtor に liveCount 操作未実装 | **ctor/dtor に `fetch_add(1)` / `fetch_sub(1)` + `jassert(old>0)` 追加** |
| 5 | SetImpulse エラー経路の allocSizes 整合性問題なし | 確認のみ（変更不要） |
| 6 | `static_cast<double*>(mkl_malloc(...))` ↔ `static_cast<double*>(DIAG_MKL_MALLOC(...))` 互換 | 確認のみ（変更不要） |
| 7 | `DiagnosticsConfig.h` の `jassert`/`DBG` が JUCE 非依存を阻害 | **`jassert` → `assert`（標準 `<cassert>`）に変更。`DBG` は `#if` 分岐で代替** |
