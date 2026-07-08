# メモリ占有調査のためのインストルメンテーション改修案 v25（最終版）— pendingRetireBytes インターフェース確定

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v24 からの変更**: ソースコード調査で発見した未確定事項 1 点を確定。

---

## 0. ソースコード調査で発見した未確定事項

### 調査: IRetireProvider に pendingRetireBytes がない

`src/core/IRetireProvider.h`:

```cpp
class IRetireProvider {
    virtual bool enqueueRetire(...) noexcept = 0;
    virtual void tryReclaim() noexcept = 0;
    virtual uint32_t pendingRetireCount() const noexcept = 0;  // ★ ある
    virtual void drainAll() noexcept = 0;
    // ★ pendingRetireBytes() がない！
};
```

`pendingRetireCount()` はあるのに `pendingRetireBytes()` がない。
ISRRetireRouter に `m_pendingRetireBytes_` だけ追加しても、インターフェース経由ではアクセスできない。

**修正**: `IRetireProvider` に `pendingRetireBytes()` 仮想関数を追加する。

---

## 1. IRetireProvider.h — pendingRetireBytes 追加

```cpp
class IRetireProvider
{
public:
    virtual ~IRetireProvider() = default;

    virtual bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept = 0;
    virtual void tryReclaim() noexcept = 0;
    virtual uint32_t pendingRetireCount() const noexcept = 0;
    virtual void drainAll() noexcept = 0;

    /// ★ v25: 退役キュー滞留バイト数（概算）。
    ///   既定値 0 を返す実装との互換性を維持。
    [[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept { return 0; }
};
```

**既定値 `return 0`** により、この仮想関数をオーバーライドしない既存実装との互換性を維持。

---

## 2. 影響を受けるファイル

| ファイル | 変更内容 |
|:--------|:--------|
| `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 |
| `src/audioengine/ISRRetireRouter.h` | `uint64_t pendingRetireBytes() const noexcept override` + `m_pendingRetireBytes_` |
| `src/audioengine/ISRRetireRouter.cpp` | `enqueueRetire` で `m_pendingRetireBytes_` 更新、`tryReclaim` で減算 |

---

## 3. 最終変更ファイル一覧（v25 確定版）

| # | ファイル | 変更内容 | 行数目安 |
|:--|:--------|:--------|:---------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 | +2行 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked + addIfAlive + updateAtomicMaximum64 + マクロ + computeOtherPrivate | ~75行 |
| 3 | `src/MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot (拡張版) + liveCount + getDiagnostics | ~20行 |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + 全 mkl_malloc→DIAG_MKL_MALLOC + allocSizes 保存 + freeAll/releaseAllLayers freeTracked + IR_RELEASE + IR_LOAD + IR_LAYOUT | ~150行 |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` | ~5行 |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `m_trackedRetireEntries_` + `trackedRatio()` | ~15行 |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | `enqueueRetire` objectBytes オーバーロード + `tryReclaim` デクリメント | ~20行 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ | ~25行 |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore liveCount + 確保量ログ | ~15行 |

**合計: 9 ファイル / 約 327 行**

---

## 4. 全ログフォーマット（v25 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

---

## 5. 出力例（v25 最終版）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB
```
