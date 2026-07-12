# メモリ占有調査のためのインストルメンテーション改修案 v31（最終版）— 全未確定事項確定

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v30 からの変更**: ソースコード調査で確定した未確定事項 4 項目を反映。

---

## 0. ソースコード調査で確定した未確定事項

### 調査結果一覧

| # | 調査項目 | 結果 | 設計への影響 |
|:--|:--------|:-----|:------------|
| 1 | `freeAll()` 内 mkl_free → `freeTracked()` にすべきか？ | **はい。v24 の一本化設計により、`freeAll()` 内でも `freeTracked()` を使用する。`allocSizes` からサイズを取得する（freeAll 内でサイズ再計算不要）** | freeAll の解放コードを修正 |
| 2 | `globalDiagSeq` 静的初期化は安全か？ | **`std::atomic<uint64_t>{0}` は constexpr コンストラクタを持つ。静的初期化安全** | 変更不要 |
| 3 | SetImpulse 呼び出し元の正確な数と引数 | **`ConvolverProcessor.h` L717/L718 の 2 箇所。各 6 引数（filterSpec まで）。v30 の「API変更なし」方針が正しい** | 変更不要（確認済み） |
| 4 | 一時バッファの完全リスト | **6 種: `impulseForFft`, `tempTime`, `tempFreq`, `swapSoA`, `gainReal`, `reusableGain`（全て SetImpulse/applySpectrumFilter 内で解放）** | 明示的に除外リスト化 |

---

## 1. 確定した全変更方針

### 1-1. DIAG_MKL_MALLOC 対象（永続バッファ、allocSizes 保存あり）

| グループ | バッファ | 確保場所 | 解放場所 | サイズ保存 |
|:---------|:--------|:--------|:--------|:----------|
| Layer 15種 | irFreqDomain / irFreqReal / irFreqImag / fdlBuf / fdlReal / fdlImag / fftTimeBuf / fftOutBuf / prevInputBuf / accumBuf / accumReal / accumImag / inputAccBuf / tailOutputBuf | SetImpulse レイヤーループ | **`freeAll()` → `freeTracked(ptr, allocSizes.XXX)`** | **各 DIAG_MKL_MALLOC 直後** |
| NUC 4種 | m_directIRRev / m_directHistory / m_directWindow / m_directOutBuf | SetImpulse 先頭 | **`releaseAllLayers()` → `freeTracked(ptr, size)`** | malloc 時に size 計算 |
| NUC 1種 | m_ringBuf | SetImpulse 末尾 | **`releaseAllLayers()` → `freeTracked(ptr, size)`** | m_ringSize から計算 |

### 1-2. DIAG_MKL_MALLOC 対象外（一時バッファ、`mkl_malloc`/`mkl_free` のまま）

| バッファ | 変数型 | 生存範囲 |
|:---------|:-------|:---------|
| `impulseForFft` | `ScopedAlignedPtr<double>` | SetImpulse() 内 |
| `tempTime` / `tempFreq` | raw `double*` | SetImpulse レイヤーループ内 |
| `swapSoA` | raw `double*` | SetImpulse レイヤーループ内（条件付き） |
| `gainReal` | `ScopedAlignedPtr<double>` | applySpectrumFilter() レイヤーループ内 |
| `reusableGain` | `ScopedAlignedPtr<double>` | applySpectrumFilter() 内（ループ間で再利用） |

### 1-3. freeAll() の最終コード（★ v31 確認）

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v31: freeTracked を使用（allocSizes からサイズ取得）
    freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
    freeTracked(irFreqReal,    allocSizes.irFreqReal);
    freeTracked(irFreqImag,    allocSizes.irFreqImag);
    freeTracked(fdlBuf,        allocSizes.fdlBuf);
    freeTracked(fdlReal,       allocSizes.fdlReal);
    freeTracked(fdlImag,       allocSizes.fdlImag);
    freeTracked(fftTimeBuf,    allocSizes.fftTimeBuf);
    freeTracked(fftOutBuf,     allocSizes.fftOutBuf);
    freeTracked(prevInputBuf,  allocSizes.prevInputBuf);
    freeTracked(accumBuf,      allocSizes.accumBuf);
    freeTracked(accumReal,     allocSizes.accumReal);
    freeTracked(accumImag,     allocSizes.accumImag);
    freeTracked(inputAccBuf,   allocSizes.inputAccBuf);
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
#else
    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    // ... 既存の 14 個の mkl_free（変更なし）...
#endif

    // ... 既存の状態リセット ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    allocSizes = {};
#endif
}
```

### 1-4. releaseAllLayers() の最終コード

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();  // ★ freeAll 内で freeTracked
    m_numActiveLayers = 0;
    m_latency         = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ NUC レベルバッファも freeTracked
    freeTracked(m_ringBuf, static_cast<size_t>(m_ringSize) * sizeof(double));
    freeTracked(m_directIRRev, static_cast<size_t>(m_directTapCount) * sizeof(double));
    freeTracked(m_directHistory, static_cast<size_t>(m_directHistLen) * sizeof(double));
    freeTracked(m_directWindow, static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double));
    freeTracked(m_directOutBuf, static_cast<size_t>(m_directMaxBlock) * sizeof(double));
#else
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }
#endif

    // ... 既存の状態リセット ...
}
```

### 1-5. SetImpulse — allocSizes 保存パターン

各 `DIAG_MKL_MALLOC` の直後に size を保存:

```cpp
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif
```

### 1-6. globalDiagSeq の初期化

```cpp
// MKLNonUniformConvolver.cpp:
std::atomic<uint64_t> MKLNonUniformConvolver::globalDiagSeq { 0 };
// std::atomic の初期化は constexpr で静的初期化安全
```

---

## 2. 全ログフォーマット（v31 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

---

## 3. 実装手順（最終版）

| 手順 | ファイル | 変更内容 | 行数目安 |
|:----|:--------|:--------|:---------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 | +2行 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked + addIfAlive + updateAtomicMaximum64 + DIAG_MKL_* マクロ + computeOtherPrivate | ~75行 |
| 3 | `src/MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot(拡張版) + liveCount + globalDiagSeq + getDiagnostics + logIrRelease friend宣言 | ~25行 |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + globalDiagSeq定義 + 全28箇所 mkl_malloc→DIAG_MKL_MALLOC + allocSizes保存(15箇所) + freeAll freeTracked + releaseAllLayers freeTracked + logIrRelease + IR_RELEASE/IR_LOAD/IR_LAYOUTログ | ~160行 |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` | ~5行 |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `m_trackedRetireEntries_` + `trackedRatio()` | ~15行 |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | enqueueRetire/tryReclaim で m_pendingRetireBytes_ 更新 | ~15行 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ（NUC/Retire/OS 統計） | ~25行 |
| 9 | `src/audioengine/DSPCoreLifecycle.cpp` | DSPCore liveCount + 確保量ログ | ~15行 |
| 10 | `src/ConvolverProcessor.h` | **変更不要**（既定値のまま 6 引数で動作） | 0行 |

**合計: 9 ファイル変更 / 約 337 行追加**

---

## 4. 出力例（v31 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
```
