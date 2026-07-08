# メモリ占有調査のためのインストルメンテーション改修案 v23（最終版）— 実装確認完了版

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v22 からの変更**: ソースコード調査結果を反映し、実装上の不確定要素を全て確定。

---

## 0. ソースコード調査で確認した事実

### 調査結果一覧

| # | 調査項目 | 結果 | 設計への影響 |
|:--|:--------|:-----|:------------|
| 1 | SetImpulse 呼び出し元 | `ConvolverProcessor.h` L717/L718 の 2 箇所。各 6 引数で呼び出し | `diagnosticGeneration=-1` 既定値で互換性維持 |
| 2 | RuntimeState.generation | `RuntimeState` の `uint64_t generation` が利用可能（L169） | 呼び出し元から `runtimeWorld->generation` を取得可 |
| 3 | releaseAllLayers early return | なし。全呼び出しで前処理なしに解放実行 | IR_RELEASE は常に出力可能 |
| 4 | CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS | CMakeLists.txt L57: 既定値 `OFF` | 調査時は `cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` が必要 |
| 5 | MKLNonUniformConvolver ctor/dtor | 両方とも存在（L302/L308） | ctor で `liveCount.fetch_add(1)`、dtor で `fetch_sub(1)` + `jassert` |
| 6 | SetImpulse シグネチャ | 6 引数（L149） | 7 引数目 `diagnosticGeneration=-1` を追加以外変更不要 |
| 7 | getProcessMemoryInfo() | `DiagnosticsConfig.h` L59 に存在 | 追加コード不要、`#include` のみで使用可能 |
| 8 | MEM_SNAP ログ | 未実装（今回新規追加） | v23 のフォーマットで新規実装 |

---

## 1. 確定した変更仕様

### 1-1. DiagnosticsConfig.h — 追加コード

```cpp
// 既存の getProcessMemoryInfo() / updateAtomicMaximum64 をそのまま使用。
// 以下を追加:

namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    std::atomic<uint32_t> lostFreeCount  { 0 };
};

inline MklAllocStats& mklStats() noexcept { ... }
inline void* diagMklMalloc(size_t size, int alignment) noexcept { ... }
inline void diagMklFree(void* ptr, size_t size,
                         const char* file, int line, const char* func) noexcept { ... }
// count のみの accessor
[[nodiscard]] inline uint32_t lostFreeCount() noexcept { ... }
inline void resetDiagnostics() noexcept { ... }

} // namespace convo::diag

// マクロ
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
  #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))
  #ifdef _MSC_VER
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE_NAME__, __LINE__, __func__)
  #else
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE__, __LINE__, __func__)
  #endif
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif

// freeTracked — Layer 専用（ptr と size が 1 セット）
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p) { jassert(size != 0); DIAG_MKL_FREE(p, size); p = nullptr; }
}

// freeTrackedSize — NUC レベル専用（動的サイズ計算、jassert なし）
template<typename T>
inline void freeTrackedSize(T*& p, size_t size) noexcept
{
    if (p) { if (size > 0) DIAG_MKL_FREE(p, size); else mkl_free(p); p = nullptr; }
}

// addIfAlive — ポインタ生存確認 + size==0 警告
static uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* name) noexcept
{
    if (ptr) { if (allocSize == 0) DBG("[DIAG] addIfAlive: " << name << " size=0"); return allocSize; }
    return 0;
}
```

### 1-2. MKLNonUniformConvolver.h — 変更

```cpp
struct LayerAllocSizes { /* 14 フィールド */ };

struct Layer {
    // ... 既存メンバ ...
    void freeAll() noexcept;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    LayerAllocSizes allocSizes;
#endif
};

struct NucDiagnosticsSnapshot {
    uint64_t layerBufs[3] = { 0, 0, 0 };
    uint64_t directBytes  = 0;
    uint64_t ringBytes    = 0;
    int      numActiveLayers = 0;
    bool     isReady         = false;
};

class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;

    bool SetImpulse(const double* impulse, int irLen, int blockSize,
                    double scale = 1.0,
                    bool enableDirectHead = false,
                    const FilterSpec* filterSpec = nullptr,
                    int diagnosticGeneration = -1);  // ★ 追加以外変更なし

    [[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept;
    // ... 既存メンバ ...
};
```

### 1-3. MKLNonUniformConvolver.cpp — 変更

**コンストラクタ/デストラクタ**:

```cpp
MKLNonUniformConvolver::MKLNonUniformConvolver()
{
    mkl_set_num_threads(1);
    liveCount.fetch_add(1, std::memory_order_relaxed);
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    releaseAllLayers();
}
```

**releaseAllLayers() — IR_RELEASE**:

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 既存の解放ロジック（freeTracked/freeTrackedSize 使用）...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);
    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)this,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB));
#endif
}
```

**SetImpulse — IR_LOAD + IR_LAYOUT**:

```cpp
bool MKLNonUniformConvolver::SetImpulse(..., int diagnosticGeneration)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;
    releaseAllLayers();

    // ... 既存の確保 + プリコンピュートループ ...
    //   全 mkl_malloc → DIAG_MKL_MALLOC
    //   全 mkl_free → DIAG_MKL_FREE
    //   allocSizes は各確保直後に保存

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;

    // IR_LOAD
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d gen=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u",
        (void*)this, irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        diagnosticGeneration,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT（getDiagnostics の Layer 走査を再利用できないため
    // 別途集計。診断用途のため二重走査を許容）
    const auto snap = getDiagnostics();
    // ... 種別ごとに集計（addIfAlive で 14 バッファ × アクティブ Layer）...
    // Total を算出
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB "
        "Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB "
        "Total=%.0fMB", ...));
#endif

    return true;
}
```

### 1-4. ConvolverProcessor.h — SetImpulse 呼び出し（既定値で動作）

既存コードは `diagnosticGeneration` 既定値 `-1` により**変更不要**。
Generation を渡したい場合は:

```cpp
// 例: nuc0->SetImpulse(ir, len, blockSize, scale, directHead, &filterSpec, diagnosticGeneration);
```

### 1-5. AudioEngine.Timer.cpp — MEM_SNAP

```cpp
// publishWorld() 成功直後（RuntimeState::generation が利用可能）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const auto nucStats = /* MKLNonUniformConvolver の統計 */;
    const auto retireStats = /* ISRRetireRouter の統計 */;
    const auto osMem = getProcessMemoryInfo();
    // ... MEM_SNAP フォーマットで出力 ...
#endif
```

---

## 2. 全ログフォーマット（確定版）

### IR_RELEASE

```
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD

```
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d gen=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT

```
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB
```

### MEM_SNAP

```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

---

## 3. 実装手順

| 手順 | ファイル | 変更内容 | 行数目安 |
|:----|:--------|:--------|:---------|
| 1 | `DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked/freeTrackedSize + addIfAlive + updateAtomicMaximum64 + マクロ | ~70行 |
| 2 | `MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot + liveCount + getDiagnostics + SetImpulse 引数追加 | ~15行 |
| 3 | `MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + 全 mkl_malloc→DIAG_MKL_MALLOC + allocSizes 保存 + freeAll DIAG_MKL_FREE + releaseAllLayers DIAG_MKL_FREE + IR_RELEASE + IR_LOAD + IR_LAYOUT | ~150行 |
| 4 | `DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` | ~5行 |
| 5 | `audioengine/ISRRetireRouter.h` | m_pendingRetireBytes_ + trackedRatio() | ~10行 |
| 6 | `audioengine/ISRRetireRouter.cpp` | enqueueRetire objectBytes オーバーロード | ~15行 |
| 7 | `audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ | ~25行 |
| 8 | `audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore liveCount + 確保量ログ | ~15行 |

### 事前準備

```bash
cmake -S . -B build -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1
cmake --build build --config Debug
```
