# ConvoPeq バグ調査報告書

**調査日**: 2026-07-17
**調査対象**: `ConvoPeq.md`（連結ソース, 75,561行 / 261ファイル）を展開し個別ファイルとして静的解析
**リポジトリ**: https://github.com/lonewolf-jp/ConvoPeq/tree/main
**方針**: 疑わしい箇所は必ず実装・呼び出し元・関連ヘッダを追跡し、実害の有無をコード上の証拠で確認してから報告（推測ベースの指摘は排除）。誤検知と判断した候補も末尾に記録する。

---

## サマリ

| # | 重大度 | 概要 | ファイル |
|---|--------|------|----------|
| 8 | **Critical** | `~NoiseShaperLearner()` がワーカースレッド群を join せずに戻り、後から破棄されるメンバ（CMA-ES用バッファ）へのUse-After-Freeを誘発しうる | `NoiseShaperLearner.h/cpp` |
| 1 | **Critical** | `Layer::delayLineBuf` が診断ビルドで解放されない（真性メモリリーク） | `MKLNonUniformConvolver.cpp/h` |
| 2 | **Critical** | Retireキュー枯渇時に `DSPCore*` 等が救済されずロストする（系統的リーク） | `DSPLifetimeManager.h`, `AudioEngine.h`, `ISRRuntimePublicationCoordinator.cpp`, `RefCountedDeferred.h` |
| 3 | **High** | EQのサチュレーション用 `fastTanh` がしきい値変更後も旧係数のままで飽和特性が破綻 | `EQProcessor.Processing.cpp` |
| 4 | **High** | `ProgressiveUpgradeThread`（高FFTサイズ再構築の背景スレッド）だけ FTZ/DAZ 未設定 | `ProgressiveUpgradeThread.cpp`, `IRConverter.cpp` |
| 5 | **Medium** | `DeferredDeletionQueue::reclaim()` の先読みスキャン処理が到達不能（デッドコード） | `DeferredDeletionQueue.h` |
| 6 | **Low/Medium** | `delayLineReadAdd()`（Audio Thread専用）で `std::abs` を使用し、同ファイル内の `absNoLibm` 規約と不整合 | `MKLNonUniformConvolver.cpp` |
| 7 | **Low(実害なし)** | `SetImpulse()` でブレース漏れにより L0 の `allocSizes.tailOutputBuf` に誤った値が入る | `MKLNonUniformConvolver.cpp` |

---

## Bug 1 [Critical] `Layer::delayLineBuf` が `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` ビルドで解放されない

### 根拠

`MKLNonUniformConvolver.h` の `LayerAllocSizes` は診断ビルド用にレイヤー内バッファのサイズを保持する構造体ですが、`delayLineBuf`（L1/L2遅延補償リングバッファ、B13で追加）のフィールドが存在しません。

```cpp
// src/MKLNonUniformConvolver.h  (L71-87)
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
    // ← delayLineBuf 用のフィールドが存在しない
};
```

確保側 (`SetImpulse`) は `delayLineBuf` を **生の `mkl_malloc`** で確保しており、`DIAG_MKL_MALLOC`（診断トラッキング付きアロケータ）も通していません。

```cpp
// src/MKLNonUniformConvolver.cpp (L1095-1120)
        l.fdlIndex       = 0;
        l.inputPos       = 0;
        l.nextPart       = 0;
        l.tailOutputPos  = 0;
        l.baseFdlIdxSaved = 0;
        l.distributing   = false;
        l.outputDelaySamples = 0;

        // ★ B13: 遅延補償リングバッファ設定 (L1/L2)
        if (prevLayerTotalSamples > 0) {
            l.outputDelaySamples = prevLayerTotalSamples;
            l.delayLineCapacity = ((prevLayerTotalSamples + l.partSize + m_maxBlockSize + 15) / 16) * 16;
            l.delayLineBuf = static_cast<double*>(
                mkl_malloc(static_cast<size_t>(l.delayLineCapacity) * sizeof(double), 64));
            if (l.delayLineBuf == nullptr) {
                releaseAllLayers();
                return false;
            }
            juce::FloatVectorOperations::clear(l.delayLineBuf, l.delayLineCapacity);
            jassert(l.outputDelaySamples > 0);
        }

        ++m_numActiveLayers;

        // ★ B13: 先行レイヤーの IR 総長を累積 (次レイヤーの outputDelaySamples 用)
        prevLayerTotalSamples += cfgs[li].len;
```

そして解放側 `Layer::freeAll()` の **診断ビルド分岐 (`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`)** には `delayLineBuf` に対する `freeTracked()` 呼び出しが一切なく、**非診断ビルド分岐 (`#else`) にのみ** `mkl_free(delayLineBuf)` が存在します。

```cpp
// src/MKLNonUniformConvolver.cpp (L318-368)
//==============================================================================
// Layer::freeAll
//==============================================================================
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // [v2.2] FFT plan はサイズ単位の共有キャッシュ管理。
    // レイヤー側は所有権のみ解放し、スペック実体はキャッシュ側で保持する。
    fftPlanOwner.reset();
    fftSpec = nullptr;
    if (fftWorkBuf)
    {
        ippsFree(fftWorkBuf);
        fftWorkBuf = nullptr;
    }
    descriptorCommitted = false;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ work70: freeTracked を使用（allocSizes からサイズ取得）
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
    allocSizes = {};
    // ← delayLineBuf の freeTracked 呼び出しが存在しない
#else
    if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
    if (irFreqReal)    { mkl_free(irFreqReal);    irFreqReal    = nullptr; }
    if (irFreqImag)    { mkl_free(irFreqImag);    irFreqImag    = nullptr; }
    if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
    if (fdlReal)       { mkl_free(fdlReal);       fdlReal       = nullptr; }
    if (fdlImag)       { mkl_free(fdlImag);       fdlImag       = nullptr; }
    if (fftTimeBuf)    { mkl_free(fftTimeBuf);     fftTimeBuf    = nullptr; }
    if (fftOutBuf)     { mkl_free(fftOutBuf);      fftOutBuf     = nullptr; }
    if (prevInputBuf)  { mkl_free(prevInputBuf);   prevInputBuf  = nullptr; }
    if (accumBuf)      { mkl_free(accumBuf);       accumBuf      = nullptr; }
    if (accumReal)     { mkl_free(accumReal);      accumReal     = nullptr; }
    if (accumImag)     { mkl_free(accumImag);      accumImag     = nullptr; }
    if (inputAccBuf)   { mkl_free(inputAccBuf);    inputAccBuf   = nullptr; }
    if (tailOutputBuf) { mkl_free(tailOutputBuf);  tailOutputBuf = nullptr; }
    if (delayLineBuf)  { mkl_free(delayLineBuf);   delayLineBuf  = nullptr; }  // ← #else側にしかない
#endif
```

`freeAll()` は `releaseAllLayers()`（`SetImpulse()` 冒頭・各種早期returnパス、デストラクタ経由）から呼ばれるため、**診断ビルドでIRを再ロードするたび（サンプルレート変更・オーバーサンプリング変更・IR差し替え等）に、そのときの `delayLineCapacity × sizeof(double)` バイト分の `delayLineBuf` が確実にリークします**。L1/L2の `delayLineCapacity` は `prevLayerTotalSamples`（先行レイヤーIR総長）に比例するため、長いIR・高いオーバーサンプリング倍率ほどリーク量も増大します。

さらに実害以外にも、`DiagnosticsConfig.h` の `freeTracked` は「`allocSizes` に登録されているのにポインタが nullptr でない」ケースを `zeroAllocSizeCount` で検出する安全網を備えていますが（L185-206）、本バグは **`freeTracked` 自体が一度も呼ばれない** ため、この安全網にも掛かりません。つまり診断ビルドの既存メモリ計測（`getDiagnostics()` / MEM_SNAP等）は `delayLineBuf` 分のメモリを一切カウントしておらず、現在進めているメモリ削減調査の実測値からもこの分だけ漏れている可能性があります。

### 影響
- 診断ビルド（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1`、まさに音飛び調査・メモリ調査で使うビルド）でIR再構築のたびに確実にメモリリーク。
- 診断統計 (`NucDiagnosticsSnapshot` 等) が実メモリ使用量を過小評価する。

### 修正パッチ

**(a) `src/MKLNonUniformConvolver.h`** — `LayerAllocSizes` に `delayLineBuf` を追加

```diff
@@ src/MKLNonUniformConvolver.h (L67-88) @@
 //==============================================================================
 // ★ work70: LayerAllocSizes — レイヤーの全 MKL バッファサイズ
 //   SetImpulse() で確保時に計算・保存し、freeAll() で DIAG_MKL_FREE に渡す。
 //==============================================================================
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
+    size_t delayLineBuf = 0;   // ★ Bug#1 修正: B13 遅延補償リングバッファのサイズ追跡を追加
 };
 
 /// NUC インスタンス単位の診断スナップショット（グローバル統計は含まない）。
 struct NucDiagnosticsSnapshot {
```

**(b) `src/MKLNonUniformConvolver.cpp`** — 確保側を `DIAG_MKL_MALLOC` に統一し `allocSizes` を記録

```diff
@@ src/MKLNonUniformConvolver.cpp (L1095-1120) @@
         l.fdlIndex       = 0;
         l.inputPos       = 0;
         l.nextPart       = 0;
         l.tailOutputPos  = 0;
         l.baseFdlIdxSaved = 0;
         l.distributing   = false;
         l.outputDelaySamples = 0;
 
         // ★ B13: 遅延補償リングバッファ設定 (L1/L2)
         if (prevLayerTotalSamples > 0) {
             l.outputDelaySamples = prevLayerTotalSamples;
             l.delayLineCapacity = ((prevLayerTotalSamples + l.partSize + m_maxBlockSize + 15) / 16) * 16;
-            l.delayLineBuf = static_cast<double*>(
-                mkl_malloc(static_cast<size_t>(l.delayLineCapacity) * sizeof(double), 64));
+            const size_t delayLineBytes = static_cast<size_t>(l.delayLineCapacity) * sizeof(double);
+            l.delayLineBuf = static_cast<double*>(DIAG_MKL_MALLOC(delayLineBytes, 64));
             if (l.delayLineBuf == nullptr) {
                 releaseAllLayers();
                 return false;
             }
+#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
+            l.allocSizes.delayLineBuf = delayLineBytes;   // ★ Bug#1 修正: サイズを記録し freeAll() で回収可能にする
+#endif
             juce::FloatVectorOperations::clear(l.delayLineBuf, l.delayLineCapacity);
             jassert(l.outputDelaySamples > 0);
         }
 
         ++m_numActiveLayers;
 
         // ★ B13: 先行レイヤーの IR 総長を累積 (次レイヤーの outputDelaySamples 用)
         prevLayerTotalSamples += cfgs[li].len;
```

> 備考: `DIAG_MKL_MALLOC(size, align)` は `DiagnosticsConfig.h` で診断ON/OFF双方に対して自己完結的に定義されているため（OFF時は単純に `mkl_malloc` に展開）、この置き換えは非診断ビルドの挙動を変えません。

**(c) `src/MKLNonUniformConvolver.cpp`** — `freeAll()` の診断ブランチに解放を追加

```diff
@@ src/MKLNonUniformConvolver.cpp (L318-368) @@
 void MKLNonUniformConvolver::Layer::freeAll() noexcept
 {
     // [v2.2] FFT plan はサイズ単位の共有キャッシュ管理。
     // レイヤー側は所有権のみ解放し、スペック実体はキャッシュ側で保持する。
     fftPlanOwner.reset();
     fftSpec = nullptr;
     if (fftWorkBuf)
     {
         ippsFree(fftWorkBuf);
         fftWorkBuf = nullptr;
     }
     descriptorCommitted = false;
 
 #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
     // ★ work70: freeTracked を使用（allocSizes からサイズ取得）
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
+    freeTracked(delayLineBuf,  allocSizes.delayLineBuf);   // ★ Bug#1 修正
     allocSizes = {};
 #else
     if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
     if (irFreqReal)    { mkl_free(irFreqReal);    irFreqReal    = nullptr; }
     if (irFreqImag)    { mkl_free(irFreqImag);    irFreqImag    = nullptr; }
     if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
     if (fdlReal)       { mkl_free(fdlReal);       fdlReal       = nullptr; }
     if (fdlImag)       { mkl_free(fdlImag);       fdlImag       = nullptr; }
     if (fftTimeBuf)    { mkl_free(fftTimeBuf);     fftTimeBuf    = nullptr; }
     if (fftOutBuf)     { mkl_free(fftOutBuf);      fftOutBuf     = nullptr; }
     if (prevInputBuf)  { mkl_free(prevInputBuf);   prevInputBuf  = nullptr; }
     if (accumBuf)      { mkl_free(accumBuf);       accumBuf      = nullptr; }
     if (accumReal)     { mkl_free(accumReal);      accumReal     = nullptr; }
     if (accumImag)     { mkl_free(accumImag);      accumImag     = nullptr; }
     if (inputAccBuf)   { mkl_free(inputAccBuf);    inputAccBuf   = nullptr; }
     if (tailOutputBuf) { mkl_free(tailOutputBuf);  tailOutputBuf = nullptr; }
     if (delayLineBuf)  { mkl_free(delayLineBuf);   delayLineBuf  = nullptr; }
 #endif
```

---

## Bug 2 [Critical] Retireキュー枯渇時に `DSPCore*` 等が救済されずロストする（系統的パターン）

### 根拠

`DeferredDeletionQueue`（固定長 `kQueueSize = 4096` のロックフリーMPMCリングバッファ）が満杯の場合、少なくとも **4箇所**で「1回だけ `tryReclaim()` してリトライし、それでも失敗したら **ポインタを何もせず捨てて return**」という同一パターンが実装されています。コメントは一貫して「HealthMonitor の overflowCount 監視に委ねるベストエフォート」としていますが、**監視カウンタを増やすだけでは対象オブジェクトのメモリは一切解放されず、リークしたまま**です。

**① `DSPLifetimeManager::retire()`**

```cpp
// src/audioengine/DSPLifetimeManager.h (L37-71)
    // Authority: DSPLifetimeManager (Lifecycle Authority)
    // Retire pipeline: DSPLifetimeManager → ISRRetireRouter → EpochDomain
    // [work37 Phase 1.1] enqueueRetire の戻り値をチェックし、失敗時に tryReclaim + 再試行
    void retire(AudioEngine::DSPCore* dsp) noexcept
    {
        if (dsp == nullptr) return;
        // 1. Release DSP handle (must happen before enqueue)
        if (!engine_.retireDSPHandleForRuntime(dsp))
            return;

        // 2. Route through ISRRetireRouter → EpochDomain
        // ★ S-1: publishEpoch() → currentEpoch() に変更。retire が epoch を進めない。
        const uint64_t epoch = router_->currentEpoch();
        if (!router_->enqueueRetire(static_cast<void*>(dsp),
                                    &AudioEngine::destroyDSPCoreNode,
                                    epoch)) {
            // ★ work37: 初回失敗 → tryReclaim で backlog 消化後に再試行
            router_->tryReclaim();
            if (!router_->enqueueRetire(static_cast<void*>(dsp),
                                        &AudioEngine::destroyDSPCoreNode,
                                        epoch)) {
                // 再試行失敗は HealthMonitor overflowCount 監視に委ねる（ベストエフォート）
                return;   // ← dsp は二度と参照されず解放もされない
            }
        }

        convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount,
                              static_cast<std::uint64_t>(1),
                              std::memory_order_acq_rel);

        // ★ work70 P1-c: 最新の retire 対象世代を記録（MEM_SNAP の retiringGeneration 用）
        const uint64_t committedGen = convo::consumeAtomic(
            engine_.lastCommittedRuntimeGeneration_, std::memory_order_acquire);
        convo::publishAtomic(currentRetiringGeneration_, committedGen, std::memory_order_release);
    }
```

`DSPLifetimeManager` クラス冒頭のコメント（「Publication 完了後に NonRT で非同期的に呼ばれる」）より、この `retire()` は **Non-RTスレッドから**呼ばれることが確認できるため、RT安全性を犠牲にせず、より強いリトライ（後述の修正案）を入れる余地があります。

**② `AudioEngine::retireDSP()` / `enqueueDeferredDeleteNonRtWithResult()`**（①とは別系統の並行パス）

```cpp
// src/audioengine/AudioEngine.h (L3901-3954)
inline convo::isr::RetireEnqueueResult enqueueDeferredDeleteNonRtWithResult(void* ptr, void (*deleter)(void*)) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return convo::isr::RetireEnqueueResult::Success;

    if (isShutdownInProgress())
        return convo::isr::RetireEnqueueResult::Shutdown;

    const uint64_t epoch = markRetireEpoch();

    // [P0-5] 単一回試行 + drop. Router経由.
    if (m_retireRouter->enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic) == convo::isr::RetireEnqueueResult::Success)
    {
        runtimePublicationBridge_.setRetireBacklogCount(
            static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount()));
        return convo::isr::RetireEnqueueResult::Success;
    }

    // [P0-5] enqueue failure -> best-effort drain + telemetry.
    drainDeferredRetireQueues(false);
    const std::uint64_t retireDepth = static_cast<std::uint64_t>(m_retireRouter->pendingRetireCount());
    convo::publishAtomic(retireQueueDepth_, retireDepth, std::memory_order_release);
    runtimePublicationBridge_.setRetireBacklogCount(retireDepth);
    return convo::isr::RetireEnqueueResult::QueuePressure;   // ← ptr は再enqueueされずに戻る
}

inline void retireDSP(DSPCore* dsp) noexcept
{
    if (dsp == nullptr)
        return;

    // 退役の唯一の入口。
    // ここでは「公開済みハンドルの解放」と「実体の deferred delete 予約」をまとめて行い、
    // active runtime slot / fading runtime slot など複数の非所有スロットからの回収責務を集約する。
    if (!retireDSPHandleForRuntime(dsp))
        return;

    convo::fetchAddAtomic(rtAuxMutable_.runtimeRetireCount,
                         static_cast<std::uint64_t>(1),
                         std::memory_order_acq_rel);
    switch (enqueueDeferredDeleteNonRtWithResult(dsp, &AudioEngine::destroyDSPCoreNode))
    {
        case convo::isr::RetireEnqueueResult::Success:
        case convo::isr::RetireEnqueueResult::QueuePressure:
            return;                                          // ← QueuePressure でも dsp は破棄されない
        case convo::isr::RetireEnqueueResult::QueueFull:
            convo::fetchAddAtomic(rtAuxMutable_.debugRebuildDispatchRuntimeQueueFullCount,
                                  static_cast<std::uint64_t>(1),
                                  std::memory_order_acq_rel);
            return;                                          // ← 同上
        case convo::isr::RetireEnqueueResult::Shutdown:
            return;
    }
}
```

この経路はコメントにある通り「`enqueue` 失敗時に **1回だけ** `drainDeferredRetireQueues()` を呼ぶがリエンキューはしない」ため、①よりも取りこぼしやすい実装です。

**③ `RuntimePublicationCoordinator::enqueueRetire()`**

```cpp
// src/audioengine/ISRRuntimePublicationCoordinator.cpp (L132-152)
RetireEnqueueResult RuntimePublicationCoordinator::enqueueRetire(RetireAuthority,
                                                                   ISRRetireRouter& router,
                                                                   void* ptr,
                                                                   void (*deleter)(void*),
                                                                   std::uint64_t epoch) noexcept
{
    convo::fetchAddAtomic(retireAuthorityCount_,
                          static_cast<std::uint64_t>(1),
                          std::memory_order_acq_rel);

    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    if (router.enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic) != RetireEnqueueResult::Success)
        return RetireEnqueueResult::QueueFull;      // ← リトライなしで即座に失敗を返す。ptr は呼び出し元任せ

    const auto backlog = convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire) + 1u;
    setRetireBacklogCount(backlog);

    return RetireEnqueueResult::Success;
}
```

**④ `RefCountedDeferred<T>::release(IEpochProvider&)`**（EQ/Convolverの参照カウント付き状態など、複数の型で使われる汎用テンプレート）

```cpp
// src/RefCountedDeferred.h (L32-53)
    // [work37 Phase 1.3] enqueueRetire 戻り値をチェック。RT から呼ばれ得るため、
    //   canBlock() (Non-RT) の場合のみ tryReclaim 再試行を行う。
    //   RT からの失敗は HealthMonitor overflowCount 監視に委ねる。
    void release(convo::IEpochProvider& provider) {
        if (convo::fetchSubAtomic(refCount, 1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            if (!provider.enqueueRetire(
                    static_cast<T*>(this),
                    [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                    provider.currentEpoch())) {
                // canBlock() が false (RT) なら tryReclaim 禁止
                if (!convo::numeric_policy::isAudioThread()) {
                    provider.tryReclaim();
                    (void)provider.enqueueRetire(
                        static_cast<T*>(this),
                        [](void* p) { std::default_delete<T>{}(static_cast<T*>(p)); },
                        provider.currentEpoch());
                }
                // 再試行失敗は HealthMonitor overflowCount 監視に委ねる
            }
        }
    }
```

このテンプレートは **Audio Threadから呼ばれる可能性がある**ことがコメントと `isAudioThread()` 分岐から明確です。Audio Threadから呼ばれて enqueue に失敗した場合、`tryReclaim()`（内部で `deleter()` を同期実行しうる = RT安全性違反の恐れ）を避けるため一切のリトライをせず即座に諦めており、これは設計上妥当な判断です。しかし結果として **参照カウントが0になったオブジェクトが二度と解放されない**（`refCount` は既に0を経由しているため以後 `tryAddRef()` もできず、誰からも再 `release()` されない = 完全にロスト）という点は他の3箇所と同じです。

### 実害の検証

`overflowCount()` は `AudioEngine.Threading.cpp` / `AudioEngine.Timer.cpp` で実際に監視対象として読み出されており、この値が increment されている場合は取りこぼしが実際に発生していることを意味します（このカウンタ自体は「発生を検知する」ためのものであり、「発生を防ぐ」ものではない点に注意）。

```
src/audioengine/AudioEngine.Timer.cpp:913:  const uint64_t overflow = m_retireRouter ? m_retireRouter->overflowCount() : 0;
```

`DeferredDeletionQueue` の容量は固定 4096 エントリで、アプリ全体の複数サブシステム（DSPCore退役、EQ/Convolverの参照カウント付き状態、その他 `RefCountedDeferred` 系オブジェクト全て）が **同一インスタンスを共有**しています。`detectStuckReaders()` のような「Readerが長時間 stuck する」ケースを想定したコードが `EpochDomain` に存在すること自体、reclaim が長時間進まない状況が設計上想定されていることの傍証です。そのような状況下でリトライを行うEQ/短命オブジェクトの高頻度な生成・破棄（パラメータの連続変更、IRの連続切り替え等）が重なると、本パターンにより `DSPCore`（EQ状態・Convolverランタイム参照・オーバーサンプラースクラッチ領域等を保持する大きめのオブジェクト）が回収不能になる可能性があります。

### 影響
長時間・高頻度の操作（EQの連続的な調整、IRの連続切り替え、再構築の輻輳等）でRetireキューが一時的に飽和すると、その瞬間に retire しようとしていたオブジェクトが **恒久的にリーク**します。ユーザーが体感する「長時間使用しているとメモリが徐々に増える」系の症状の有力な説明候補です。

### 修正方針（アーキテクチャ判断が必要なため、方向性を提示）

Non-RT確定である①③は「1回リトライ」ではなく **有界のリトライループ**（例: 数回 `tryReclaim()` を挟みつつ再試行、それでも失敗した場合のみ最終手段としてログ＋telemetryに残す）に強化できます。

```diff
@@ src/audioengine/DSPLifetimeManager.h (L37-61) @@
     // Authority: DSPLifetimeManager (Lifecycle Authority)
     // Retire pipeline: DSPLifetimeManager → ISRRetireRouter → EpochDomain
     // [work37 Phase 1.1] enqueueRetire の戻り値をチェックし、失敗時に tryReclaim + 再試行
     void retire(AudioEngine::DSPCore* dsp) noexcept
     {
         if (dsp == nullptr) return;
         // 1. Release DSP handle (must happen before enqueue)
         if (!engine_.retireDSPHandleForRuntime(dsp))
             return;
 
         // 2. Route through ISRRetireRouter → EpochDomain
         // ★ S-1: publishEpoch() → currentEpoch() に変更。retire が epoch を進めない。
         const uint64_t epoch = router_->currentEpoch();
-        if (!router_->enqueueRetire(static_cast<void*>(dsp),
-                                    &AudioEngine::destroyDSPCoreNode,
-                                    epoch)) {
-            // ★ work37: 初回失敗 → tryReclaim で backlog 消化後に再試行
-            router_->tryReclaim();
-            if (!router_->enqueueRetire(static_cast<void*>(dsp),
-                                        &AudioEngine::destroyDSPCoreNode,
-                                        epoch)) {
-                // 再試行失敗は HealthMonitor overflowCount 監視に委ねる（ベストエフォート）
-                return;
-            }
-        }
+        bool enqueued = router_->enqueueRetire(static_cast<void*>(dsp),
+                                               &AudioEngine::destroyDSPCoreNode,
+                                               epoch);
+        // ★ Bug#2 修正: Non-RT確定パスのため、1回に限らず有界回数リトライする。
+        //   本当に埋まっている場合のみ最終的に諦め、少なくとも「一時的な瞬間飽和」による
+        //   ロストは防ぐ。kMaxRetireRetries は運用実績を見て調整可能。
+        constexpr int kMaxRetireRetries = 8;
+        for (int attempt = 0; !enqueued && attempt < kMaxRetireRetries; ++attempt) {
+            router_->tryReclaim();
+            enqueued = router_->enqueueRetire(static_cast<void*>(dsp),
+                                              &AudioEngine::destroyDSPCoreNode,
+                                              epoch);
+        }
+        if (!enqueued) {
+            // ★ 最終手段: それでも空かない場合のみ HealthMonitor overflowCount 監視に委ねる。
+            //   将来的には「オーバーフロー専用の二次キュー」に退避させ、
+            //   低優先度スレッドが後で吸収する設計への置き換えを推奨。
+            return;
+        }
 
         convo::fetchAddAtomic(engine_.rtAuxMutable_.runtimeRetireCount,
                               static_cast<std::uint64_t>(1),
                               std::memory_order_acq_rel);
```

④（`RefCountedDeferred::release`）はAudio Threadから呼ばれ得るため、同じ「ループでリトライ」は使えません。ここは根本的に **キュー自体を涸れにくくする**（`kQueueSize` の拡大、またはRT側は溢れた分を専用の小容量ロックフリー・オーバーフロースタックへ退避させ、非RTの `DeferredFreeThread` が定期的にそちらも回収する）方向の対策が必要です。まずは `overflowCount()` を実運用で監視し、実際にインクリメントされているかどうかを確認することを推奨します（0であれば理論上のリスクに留まりますが、非0であれば確実にリークが発生しています）。

---

## Bug 3 [High] EQサチュレーション用 `fastTanh` がしきい値変更後も旧係数のままで飽和特性が破綻

### 根拠

`EQProcessor.Processing.cpp` のスカラー版・ベクトル版 `fastTanh` は「Padé近似 `x(27+x²)/(27+9x²)` はちょうど `x=3` で厳密に `1.0` になる」という代数的性質を利用したクリッパーです（`3·(27+9)/(27+81) = 108/108 = 1`）。

```cpp
// src/eqprocessor/EQProcessor.Processing.cpp (L86-112)
    // fastTanh（出力用）— クリップ閾値を 4.5 に引き上げ
    // SVF出力信号（特に Low Shelf +12dB ブースト時）は容易に ±3.0 を超えるため、
    // 状態変数用の fastTanh（閾値3.0）では通常のオーディオ信号が頻繁にクリップする。
    // SoftClip用 TanhApprox（閾値4.5）に合わせることで、自然な飽和特性を維持する。
    inline double fastTanhScalarOutput(double x) noexcept
    {
        constexpr double kClipThreshold = 4.5;
        if (x >= kClipThreshold) return 1.0;
        if (x <= -kClipThreshold) return -1.0;
        const double x2 = x * x;
        return x * (27.0 + x2) / (27.0 + 9.0 * x2);
    }

    inline __m128d fastTanhV128Output(__m128d x) noexcept
    {
        constexpr double kClipThreshold = 4.5;
        const __m128d vClipHigh = _mm_set1_pd(kClipThreshold);
        const __m128d vClipLow  = _mm_set1_pd(-kClipThreshold);
        const __m128d vNine = _mm_set1_pd(9.0);
        const __m128d vTwentySeven = _mm_set1_pd(27.0);

        const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vClipLow), vClipHigh);
        const __m128d x2 = _mm_mul_pd(xClamped, xClamped);
        const __m128d num = _mm_mul_pd(xClamped, _mm_add_pd(vTwentySeven, x2));
        const __m128d den = _mm_add_pd(vTwentySeven, _mm_mul_pd(vNine, x2));
        return _mm_div_pd(num, den);
    }
```

コメントより、開発意図は「状態変数用（旧: 閾値3.0）だと Low Shelf +12dB 時に頻繁にクリップするため、閾値を SoftClip 側（4.5）に合わせた」というものです。しかし **`kClipThreshold` の値だけを 3.0→4.5 に変更し、分子分母の係数 `27.0` / `9.0` は据え置き**になっています。この係数は `x=3` で厳密に1になるよう設計されたものであり、`x=4.5` に対して再設計されたものではありません。実際に計算すると、`x=3` を超えた範囲で関数値は **1.0を超えて単調増加**します（境界で不連続にはならず、`x→∞` で `x/9` に漸近し発散します）。

| x | f(x) = x(27+x²)/(27+9x²) |
|---|---|
| 3.0 | 1.00000（設計上の基準点） |
| 3.5 | 1.00091 |
| 4.0 | 1.00585 |
| 4.499999... | ≈1.01613（`kClipThreshold` 直前の極限） |
| 4.5 | スカラー版: `x>=4.5` 分岐で **厳密に 1.0**（≈1.01613 から不連続に落ちる）<br>ベクトル版: クランプ後に同じ式を評価するため **≈1.01613 のまま**（1.0を超える） |

つまり：
1. **スカラー版**は `x∈(3, 4.5)` の範囲で `|出力|>1.0` を返し得る（本来のサチュレータの意味に反し、ブースト方向に振れる）上、`x=4.5` ちょうどで関数値が `1.01613→1.0` へ**不連続にジャンプ**します（サチュレーションカーブに小さいがはっきりした段差が生じる）。
2. **ベクトル版 (`fastTanhV128Output`)** はスカラー版のような早期return（`x>=4.5 → 1.0`）を持たず、単に `x` を `[-4.5, 4.5]` にクランプしてから同じ式を評価するため、`|x|>=4.5` の入力全てで **常に ≈1.01613倍**（スカラー版の1.0とは異なる値）を返します。

呼び出し元はブロック内サンプル数が4の倍数かどうかでSIMD経路とスカラー経路のどちらを通るかが変わるため（`processBand` 対 `processBandStereo`、あるいは同一関数内のAVX2ループ+残余スカラーループ）、**全く同じ入力サンプルでも処理経路によって出力が変わり得る**という、決定論的でない挙動になっています。

```cpp
// src/eqprocessor/EQProcessor.Processing.cpp (L153-159) 呼び出し例（スカラー, processBand内）
            double output = m0 * v0 + m1 * v1 + m2 * v2;

            if (saturation > 0.0)
            {
                const double oneMinusSat = 1.0 - saturation;
                output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
            }
```

```cpp
// src/eqprocessor/EQProcessor.Processing.cpp (L237-247) 呼び出し例（ベクトル, processBandStereo内）
            // FMA: m0*v0 + m1*v1 + m2*v2
            __m128d output = _mm_fmadd_pd(m0, v0,
                              _mm_fmadd_pd(m1, v1,
                               _mm_mul_pd(m2, v2)));

            if (saturation > 0.0)
            {
                const __m128d vSat = _mm_set1_pd(saturation);
                const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
                output = _mm_add_pd(_mm_mul_pd(output, vOneMinusSat),
                                    _mm_mul_pd(fastTanhV128Output(output), vSat));
            }
```

コメント自体が「Low Shelf +12dB ブースト時は容易に±3.0を超える」と明言しているため、この不具合は極端な入力でのみ発生する理論上のコーナーケースではなく、**サチュレーション有効時の通常運用（ブースト系EQ設定）で恒常的に踏みうる**範囲です。

### 影響
- `nonlinearSaturation > 0` のとき、EQ出力がわずかに1.0を超えるオーバーシュートを起こし得る（後段のクランプ `[-100,100]` には掛からない程度の小さな超過だが、意図した「サチュレータ」としての役割を外れる）。
- スカラー経路とSIMD経路で異なる値を出す＝ブロックサイズや実行環境（AVX2有無、コンパイラの自動ベクトル化状況）によって微妙に音が変わり得る、再現性のないバグ。
- `x=4.5` 境目でスカラー経路にのみ不連続点があり、非常に小さいながら理論上クリック様の歪みの原因になり得る。

### 修正パッチ

もっとも簡単で安全な修正は、**係数をそのままに、早期リターンの閾値を「関数が本来1.0に収束する3.0」に戻す**ことです（`kClipThreshold` の名前が示す通り、この関数はそもそも3.0でしか正しく機能しません）。「Low Shelf +12dBで頻繁にクリップする」問題を解決したいのであれば、閾値の数値変更ではなく、**4.5で1.0に収束する係数の別カーブ**（例えば `DSPCoreDouble.cpp` の `TanhApprox`（7次/7次、閾値4.5で校正済み）と同じ係数セットに差し替える）が必要です。

```diff
@@ src/eqprocessor/EQProcessor.Processing.cpp (L86-112) @@
     // fastTanh（出力用）— クリップ閾値を 4.5 に引き上げ
     // SVF出力信号（特に Low Shelf +12dB ブースト時）は容易に ±3.0 を超えるため、
     // 状態変数用の fastTanh（閾値3.0）では通常のオーディオ信号が頻繁にクリップする。
     // SoftClip用 TanhApprox（閾値4.5）に合わせることで、自然な飽和特性を維持する。
+    //
+    // ★ Bug#3 修正: 27.0/9.0 の係数は x=3 でのみ厳密に 1.0 に収束するよう設計された
+    //   Padé近似であり、kClipThreshold だけを 4.5 に変更しても x∈(3,4.5) で |出力|>1.0 の
+    //   オーバーシュートと、スカラー/SIMD間の不整合（早期return の有無）が生じる。
+    //   恒久対応としては AudioEngine.Processing.DSPCoreDouble.cpp の
+    //   TanhApprox（NUM_A/B/C, DEN_A/B/C, 閾値4.5で校正済み）へ統一することを推奨。
+    //   ここでは最小修正として、係数と整合するオリジナルの閾値 3.0 に戻す。
     inline double fastTanhScalarOutput(double x) noexcept
     {
-        constexpr double kClipThreshold = 4.5;
+        constexpr double kClipThreshold = 3.0;   // 27.0/9.0 係数が 1.0 に収束する点
         if (x >= kClipThreshold) return 1.0;
         if (x <= -kClipThreshold) return -1.0;
         const double x2 = x * x;
         return x * (27.0 + x2) / (27.0 + 9.0 * x2);
     }
 
     inline __m128d fastTanhV128Output(__m128d x) noexcept
     {
-        constexpr double kClipThreshold = 4.5;
+        constexpr double kClipThreshold = 3.0;   // スカラー版と揃える
         const __m128d vClipHigh = _mm_set1_pd(kClipThreshold);
         const __m128d vClipLow  = _mm_set1_pd(-kClipThreshold);
         const __m128d vNine = _mm_set1_pd(9.0);
         const __m128d vTwentySeven = _mm_set1_pd(27.0);
 
         const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vClipLow), vClipHigh);
         const __m128d x2 = _mm_mul_pd(xClamped, xClamped);
         const __m128d num = _mm_mul_pd(xClamped, _mm_add_pd(vTwentySeven, x2));
         const __m128d den = _mm_add_pd(vTwentySeven, _mm_mul_pd(vNine, x2));
         return _mm_div_pd(num, den);
     }
```

> **設計判断が必要な点**: 上記パッチは「オーバーシュートと不整合を消す」ことを優先し、`kClipThreshold` を係数と整合する 3.0 に戻す最小修正です。ただし、これは開発者が意図した「Low Shelf +12dBブースト時の過剰クリップ抑制」という当初の目的を後退させます。その目的を維持したまま安全に4.5まで伸ばすには、DSPCoreDouble.cpp の高次 `TanhApprox`（4.5で校正済み）に差し替えるほうが適切です。どちらの方針を取るかはサウンドデザイン上の意図次第のため、最終判断を委ねます。

---

## Bug 4 [High] `ProgressiveUpgradeThread` だけ FTZ/DAZ（デノーマル保護）が未設定

### 根拠

本プロジェクトでは重い浮動小数点処理を行うほぼ全てのスレッドで `_MM_SET_FLUSH_ZERO_MODE` / `_MM_SET_DENORMALS_ZERO_MODE`（もしくは `juce::ScopedNoDenormals`）を明示的に設定しています。

```
src/MainApplication.cpp:142-146        （メッセージスレッド起点）
src/core/WorkerThread.cpp:62-63
src/NoiseShaperLearner.cpp:526-530, 737-742（学習スレッド）
src/convolver/ConvolverProcessor.LoaderThread.cpp:41-46, 358
src/convolver/ConvolverProcessor.MixedPhase.cpp:151-152, 734-735
src/audioengine/AudioEngine.RebuildDispatch.cpp:743-745
src/MKLRealTimeSetup.cpp:31-33（Audio Thread起点）
src/audioengine/AudioEngine.Processing.BlockDouble.cpp:91（ScopedNoDenormals）
src/audioengine/AudioEngine.Processing.AudioBlock.cpp:89（ScopedNoDenormals）
src/eqprocessor/EQProcessor.Processing.cpp:481, 1036（ScopedNoDenormals）
```

しかし、**`ProgressiveUpgradeThread::run()`**（メモリ関連メモにある「60〜70秒ウォームアップ窓」を生成している、高FFTサイズへの背景再構築スレッド）にはFTZ/DAZ設定が一切ありません。

```cpp
// src/ProgressiveUpgradeThread.cpp (L73-92)
void ProgressiveUpgradeThread::run()
{
    if (affinityManager != nullptr)
        affinityManager->applyCurrentThreadPolicy(ThreadType::HeavyBackground);

    // JUCE のクロスプラットフォーム優先度設定
    setPriority(Priority::low);

    if (checkAndCancel())
        return;

    for (int step : upgradeSteps)
    {
        if (!isGenerationValid())
            return;

        if (!upgradeStep(step))
            return;
    }
}
```

このスレッドが呼び出す `upgradeStep()` → `IRConverter::convertToHighRes()` → `IRConverter::convertFile()` の中でも、FTZ/DAZ設定は行われていません。

```cpp
// src/IRConverter.cpp (L236-257) convertFile() 冒頭（FTZ/DAZ設定なし）
std::unique_ptr<PreparedIRState> IRConverter::convertFile(const juce::File& irFile,
                                                          const ConvertConfig& config,
                                                          const std::function<bool()>& shouldCancel) const
{
    juce::AudioBuffer<double> ir;
    double sourceRate = 0.0;
    if (!loadAudioFile(irFile, ir, sourceRate))
        return nullptr;

    if (shouldCancel && shouldCancel())
        return nullptr;

    juce::AudioBuffer<double> converted = ir;
    if (config.targetSampleRate > 0.0 && sourceRate > 0.0 && std::abs(sourceRate - config.targetSampleRate) > 1.0e-6)
    {
        converted = IRDSP::resampleIR(ir, sourceRate, config.targetSampleRate, shouldCancel);
        if (converted.getNumSamples() <= 0)
            return nullptr;
    }
```

`IRDSP::resampleIR()` はIR（インパルスレスポンス、すなわち指数的に減衰していくテール）に対するFIRベースのリサンプリング処理であり、**減衰テールはデノーマル数の典型的な発生源**です。このスレッドでFTZ/DAZが有効化されていない場合、x86でデノーマル演算に入るたびに数十〜数百倍のレイテンシペナルティが発生し得ます。

### 影響
ユーザーのメモリ調査メモに記載されている「`ProgressiveUpgradeThread` の 60〜70秒ウォームアップ窓での2エンジン共存」が、まさにこのスレッドの実行時間に直結しています。デノーマル起因の速度低下が発生すると、この待機時間がさらに伸び、**2エンジンが同時にメモリ上に存在する時間そのものが延びる**ため、現在取り組んでいるメモリ削減の効果を相殺する方向に働きます。加えて、単純に「IR切り替え後、高品質エンジンへの昇格が完了するまでの体感時間」が悪化する副作用もあります。

### 修正パッチ

コードベース内の他のバックグラウンドスレッドと同じパターンに揃えるのが最小修正です。

```diff
@@ src/ProgressiveUpgradeThread.cpp (L1-15) @@
 #include "ProgressiveUpgradeThread.h"
 
 #include "ConvolverProcessor.h"
 #include "IRConverter.h"
 #include "CacheManager.h"
 #include "PreparedIRState.h"
 #include "core/ThreadAffinityManager.h"
 
 #include <cmath>
 
 #include "audioengine/AtomicAccess.h"
+#include <xmmintrin.h>  // _MM_SET_FLUSH_ZERO_MODE
+#include <pmmintrin.h>  // _MM_SET_DENORMALS_ZERO_MODE
```

```diff
@@ src/ProgressiveUpgradeThread.cpp (L73-92) @@
 void ProgressiveUpgradeThread::run()
 {
+    // ★ Bug#4 修正: 他の全バックグラウンドスレッド（WorkerThread, NoiseShaperLearner,
+    //   ConvolverProcessor.LoaderThread 等）と同様に、スレッド開始時点で FTZ/DAZ を
+    //   有効化する。convertToHighRes() 内の IRDSP::resampleIR() は減衰する IR テールを
+    //   処理するため、デノーマル発生によるレイテンシ悪化＝ウォームアップ窓の延長を招きやすい。
+    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
+    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
+
     if (affinityManager != nullptr)
         affinityManager->applyCurrentThreadPolicy(ThreadType::HeavyBackground);
 
     // JUCE のクロスプラットフォームの優先度設定
     setPriority(Priority::low);
 
     if (checkAndCancel())
         return;
 
     for (int step : upgradeSteps)
     {
         if (!isGenerationValid())
             return;
 
         if (!upgradeStep(step))
             return;
     }
 }
```

> `MainApplication.cpp` 等では `vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)` もあわせて呼ばれています。`convertFile()` 内でMKL VML関数（`vdExp`等）を使用している場合は、そちらもこのスレッド起点で呼ぶ必要がないか確認してください（本調査ではVML関数の使用は確認できませんでしたが、`IRDSP::resampleIR` の内部実装までは追い切れていません）。

---

## Bug 5 [Medium] `DeferredDeletionQueue::reclaim()` の先読みスキャンが到達不能

### 根拠

```cpp
// src/DeferredDeletionQueue.h (L107-167)
    // Message Thread / Timer から呼ばれる。
    uint32_t reclaim(uint64_t minReaderEpoch) {
        constexpr int kMaxScan = 1024;
        uint32_t deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
        uint32_t scanPos = deqPos;
        int scanned = 0;
        uint32_t reclaimed = 0;

        while (scanned < kMaxScan) {
            auto& seq_atom = sequences[scanPos & kMask];
            const uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire);
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(scanPos + 1);

            if (diff != 0) {
                break; // Empty
            }

            auto& entry = ringBuffer[scanPos & kMask];
            bool canDelete = false;

            if (isOlder(entry.epoch, minReaderEpoch)) {
                canDelete = true;
            }

            // FIFO を維持するため、現在の dequeue 先頭と一致した時だけ削除する。
            if (canDelete && scanPos == deqPos) {
                if (convo::compareExchangeAtomic(dequeuePos,
                                                 deqPos,
                                                 static_cast<uint32_t>(deqPos + 1),
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
                    if (entry.deleter && entry.ptr) {
                        entry.deleter(entry.ptr);
                    }
                    ++reclaimed;
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    entry.type = DeletionEntryType::Generic;
                    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release);

                    ++deqPos;        // [BUG-03] dequeuePos の新値 (deqPos+1) に追従
                    scanPos = deqPos;
                    scanned = 0;
                } else {
                    deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
                    scanPos = deqPos;
                    scanned = 0;
                }
            } else {
                // ★ 最適化: 先頭エントリが削除不可の場合、後続も削除不可（FIFO順序）のため即座に脱出
                if (!canDelete)
                    break;
                if (scanPos - deqPos > static_cast<uint32_t>(kMaxScan)) {
                    scanPos = deqPos;
                } else {
                    ++scanPos;
                }
                ++scanned;
            }
        }
        return reclaimed;
    }
```

このループの `if`／`else` を辿ると、**`scanPos` と `deqPos` は常に等しい状態でしか `else` 節に到達しません**。理由:

- ループ先頭で `scanPos == deqPos`（初期化時、および毎回のCAS成功／失敗パスの直後で必ず `scanPos = deqPos` に再同期される）。
- `else` 節に入るのは `canDelete && scanPos == deqPos` が偽の場合。`scanPos == deqPos` は常に真なので、`else` に入るのは常に `canDelete == false` の場合のみ。
- その `else` 節内の最初の行が `if (!canDelete) break;` であるため、**`else` に入った時点で必ず即座に `break` します**。

結果として、`else` 節後半にある「`scanPos - deqPos > kMaxScan` ならリセット、そうでなければ `++scanPos`」という**先読みスキャン用のロジックと `kMaxScan=1024` のスキャン予算は実行されることがない、到達不能コード**になっています。

現状の実効的な動作は「先頭（`deqPos` が指すエントリ）が reclaim 可能な間は次々回収し続け、reclaim 不可能な先頭に当たった瞬間に即終了」であり、これは**キュー内のエポックが挿入順に単調非減少である**という前提が常に成り立つ場合のみ正しい早期終了です。しかし本キューは **MPMC**（複数プロデューサ・複数コンシューマ）であり、`enqueue()` はロックなしで複数スレッドから同時に呼ばれます。異なるスレッドが独立に取得した epoch を積む場合、スレッドスケジューリング次第では「わずかに新しいepochのエントリが先にキューへ入り、わずかに古いepochのエントリが後から入る」という**キュー内での逆転が理論上起こり得ます**（少なくとも、それが起こらないという保証はこのファイル単体のコードからは読み取れません）。もしこれが実際に起こると、先頭に居座る「一時的にreclaim不可な新しめのエントリ」の背後に、本来ならとっくにreclaim可能な古いエントリが並んでいても、`kMaxScan` の先読み予算があるにもかかわらず**一切スキップして回収することができず**、先頭のエントリが十分古くなるまでキュー全体の回収が足止めされます。

### 影響
- 現状、機能的に「間違った」回収漏れ（誤ってpermanentにリークする）は起こしません（先頭が結局は古くなれば通常どおり回収されるため）が、**キュー内でepochの逆転が発生する運用下では、reclaim の遅延・キューの見かけ上の飽和**（≒ Bug#2 の `QueuePressure`/`QueueFull` を誘発しやすくする）につながり得ます。
- `kMaxScan = 1024` という定数と、それに紐づくコメント・スキャンカウンタの存在が示す設計意図（＝ある程度先読みしてでも回収を進める）と、実際の挙動（先読みは一切行われない）が食い違っており、将来のメンテナンス時に誤解を招きます。

### 修正方針

epochの逆転が実運用で起こり得るかどうか（＝本当に複数プロデューサが同時にこのキューへ enqueue するか）を切り分けた上で、
- **起こり得ない**ことが確認できるなら、到達不能な `else` 節後半とスキャンカウンタ関連のコードを削除し、意図と実装を一致させる（単純化）。
- **起こり得る**なら、「先頭が reclaim 不可でも即 break せず、`deqPos` 自体は動かさずに `scanPos` だけ先読みし、"reclaim 可能だが `scanPos != deqPos`" のエントリを見つけたら **元の位置に留めたまま**（FIFO順序を壊さないよう、後で改めて先頭から辿り直す設計に作り替える）先読みロジックを実際に機能させる必要があります。この場合、単純な「先読みで見つけたら即回収」は導入すると FIFO 順序保証が崩れる（`entry.epoch` 以外の順序依存処理がある場合に影響）ため、設計の再検討が必要です。

```diff
@@ src/DeferredDeletionQueue.h (L152-165) @@
             } else {
-                // ★ 最適化: 先頭エントリが削除不可の場合、後続も削除不可（FIFO順序）のため即座に脱出
-                if (!canDelete)
-                    break;
-                if (scanPos - deqPos > static_cast<uint32_t>(kMaxScan)) {
-                    scanPos = deqPos;
-                } else {
-                    ++scanPos;
-                }
-                ++scanned;
+                // ★ Bug#5: 現状 scanPos は常に deqPos と一致した状態でしかこの else へ
+                //   到達しないため（!canDelete の場合のみ到達）、以下は事実上
+                //   "先頭が reclaim 不可なら即終了" と等価。
+                //   MPMC enqueue によりキュー内で epoch の挿入順逆転が起こらないことが
+                //   保証できるのであれば、この break のみで正しく、
+                //   kMaxScan による先読みは不要（コメント・変数を削除して簡素化すべき）。
+                //   逆転が起こり得るなら、scanPos だけを先に進めて "reclaim 可能だが
+                //   非先頭" のエントリを検出する先読みロジックを別途、FIFO 順序を
+                //   崩さない形で実装し直す必要がある。
+                break;
             }
```

> 上記パッチは「現状の実効動作を変えずに、デッドコードである旨を明記して簡素化する」保守的な対応です。逆転が実運用で起こり得るかどうかはこのファイル単体では判定できないため、**Bug#2 の `overflowCount()` 監視結果と合わせて、実測で切り分けることを推奨**します。

---

## Bug 6 [Low/Medium] `delayLineReadAdd()`（Audio Thread専用）で `std::abs` を使用

### 根拠

`MKLNonUniformConvolver.cpp` は「libm呼び出しを avoid するため」明示的に `absNoLibm()` をファイル内に定義しており（ビット演算で符号ビットを落とすだけの実装）、実際に **`Get()`** 自身（`// Get ─ Audio Thread` とコメントされた、Audio Thread専用関数）内の `addScaledFallback` ラムダではこの `absNoLibm` を使っています。

```cpp
// src/MKLNonUniformConvolver.cpp (L40-43)
// absNoLibm — 標準ライブラリ abs を経由せずビット操作で |x| を求める (RT-safe)
inline double absNoLibm(double x) noexcept
{
    ...
}
```

```cpp
// src/MKLNonUniformConvolver.cpp (L1640-1690) Get() 内
//==============================================================================
// Get  ─ Audio Thread
//==============================================================================
int MKLNonUniformConvolver::Get(double* output, int numSamples)
{
    ...
    const int got = ringRead(output, numSamples);

    auto addFallback = [](int n, double* dst, const double* src) noexcept
    {
        ...
    };

    [[maybe_unused]] auto addScaledFallback = [&addFallback](int n, double* dst, const double* src, double gain) noexcept
    {
        if (absNoLibm(gain - 1.0) < 1.0e-12)
        {
            addFallback(n, dst, src);
            return;
        }
        for (int i = 0; i < n; ++i)
            dst[i] += src[i] * gain;
    };
```

ところが、同じ `Get()` から直接呼ばれる **`delayLineReadAdd()`**（L1/L2 遅延補償リングバッファの読み出し、これもAudio Thread専用）では `std::abs` を使っています。

```cpp
// src/MKLNonUniformConvolver.cpp (L1738-1777)

//==============================================================================
// ★ B13: delayLineReadAdd — 遅延補償リングバッファ読み出し + 加算 (Get)
//==============================================================================
void MKLNonUniformConvolver::delayLineReadAdd(Layer& l, double* dst, int numSamples, double gain) noexcept
{
    if (l.delayLineBuf == nullptr || l.delayLineCapacity <= 0 || dst == nullptr)
        return;

    // ★ readCursor = max(readCursor, writeCursor - outputDelaySamples)
    const uint64_t maxRead = (l.delayWriteCursor >= static_cast<uint64_t>(l.outputDelaySamples))
        ? (l.delayWriteCursor - static_cast<uint64_t>(l.outputDelaySamples))
        : 0;
    const uint64_t actualReadStart = std::max(l.delayReadCursor, maxRead);

    // ★ Writer がまだ outputDelaySamples 分先に進んでいない → スキップ
    if (actualReadStart + static_cast<uint64_t>(numSamples) > l.delayWriteCursor)
        return;

    // ★ リングバッファ読み出し
    const size_t readOffset = static_cast<size_t>(actualReadStart % static_cast<uint64_t>(l.delayLineCapacity));
    const int first = std::min(numSamples, l.delayLineCapacity - static_cast<int>(readOffset));
    if (first > 0) {
        const double* src = l.delayLineBuf + readOffset;
        if (std::abs(gain - 1.0) < 1.0e-12)
            for (int i = 0; i < first; ++i) dst[i] += src[i];
        else
            for (int i = 0; i < first; ++i) dst[i] += src[i] * gain;
    }
    if (first < numSamples) {
        const double* src = l.delayLineBuf;
        const int second = numSamples - first;
        if (std::abs(gain - 1.0) < 1.0e-12)
            for (int i = 0; i < second; ++i) dst[first + i] += src[i];
        else
            for (int i = 0; i < second; ++i) dst[first + i] += src[i] * gain;
    }

    l.delayReadCursor = actualReadStart + static_cast<uint64_t>(numSamples);
}
```

コーディング規約の「Audio Thread内でlibm呼び出しとなる関数を使わない」という要件に対し、`std::abs(double)` はMSVC/x64 + 通常の最適化設定下では単一のAND命令にインライン化されるのが一般的ですが、それを**保証する記述はこのファイルにもコーディング規約にもなく**、少なくとも同一ファイル内・同一呼び出しチェーン内（`Get()` → `delayLineReadAdd()`）で「片方は `absNoLibm`、片方は `std::abs`」という不整合があるのは、意図せぬ実装漏れである可能性が高いです。

### 影響
- 実害はビルド環境・最適化設定に依存（多くの場合インライン化され実害なし）。
- ただし規約上は違反であり、将来的に `/fp:strict` 等へビルドオプションが変わった場合や、Debugビルド以外の何らかの理由でインライン化が阻害された場合に、想定外のlibm呼び出しがAudio Thread上で発生するリスクを残します。

### 修正パッチ

```diff
@@ src/MKLNonUniformConvolver.cpp (L1756-1774) @@
     // ★ リングバッファ読み出し
     const size_t readOffset = static_cast<size_t>(actualReadStart % static_cast<uint64_t>(l.delayLineCapacity));
     const int first = std::min(numSamples, l.delayLineCapacity - static_cast<int>(readOffset));
     if (first > 0) {
         const double* src = l.delayLineBuf + readOffset;
-        if (std::abs(gain - 1.0) < 1.0e-12)
+        if (absNoLibm(gain - 1.0) < 1.0e-12)
             for (int i = 0; i < first; ++i) dst[i] += src[i];
         else
             for (int i = 0; i < first; ++i) dst[i] += src[i] * gain;
     }
     if (first < numSamples) {
         const double* src = l.delayLineBuf;
         const int second = numSamples - first;
-        if (std::abs(gain - 1.0) < 1.0e-12)
+        if (absNoLibm(gain - 1.0) < 1.0e-12)
             for (int i = 0; i < second; ++i) dst[first + i] += src[i];
         else
             for (int i = 0; i < second; ++i) dst[first + i] += src[i] * gain;
     }
```

---

## Bug 7 [Low・実害なし確認済み] `SetImpulse()` のブレース漏れで `allocSizes.tailOutputBuf` に誤値

### 根拠

```cpp
// src/MKLNonUniformConvolver.cpp (該当箇所、SetImpulse 内)
        if (!l.isImmediate)
            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);
#endif
```

`if (!l.isImmediate)` にブレースがなく、直後の1文（`l.tailOutputBuf = ...`）にしか及びません。続く `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ブロックは `isImmediate` の真偽に関わらず無条件に実行されるため、L0（immediateレイヤー、`tailOutputBuf` を使わない）でも `allocSizes.tailOutputBuf` に非ゼロのサイズが書き込まれます。

**実害の検証**: `freeTracked()` および `addIfAlive()`（`DiagnosticsConfig.h`）はいずれも最初に `if (p)` でポインタのnullチェックを行うため、L0では `tailOutputBuf` 自体がnullptrのまま残り、`allocSizes.tailOutputBuf` の値が使われることはありません。診断統計・解放処理のいずれにも影響しないことを確認済みです。

### 修正パッチ（コードの意図を明確にするための整合性修正、実害はない）

```diff
@@ src/MKLNonUniformConvolver.cpp (SetImpulse内、tailOutputBuf確保箇所) @@
-        if (!l.isImmediate)
-            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
-#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
-        l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);
-#endif
+        if (!l.isImmediate)
+        {
+            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
+#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
+            l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);   // ★ Bug#7: isImmediateガード内に移動
+#endif
+        }
```

> 該当箇所の正確な行番号はビルド構成によって前後する可能性があるため、`grep -n "l.tailOutputBuf = static_cast<double\*>(DIAG_MKL_MALLOC" src/MKLNonUniformConvolver.cpp` で最新行を確認の上、適用してください。

---

## Bug 8 [Critical] `~NoiseShaperLearner()` がワーカースレッドをjoinせずに戻り、後から破棄されるバッファへのUse-After-Freeを誘発しうる

### 根拠

`NoiseShaperLearner` はCMA-ESによるノイズシェーパー係数の自己学習を行うクラスで、メインの学習スレッド (`workerThread`, `std::jthread`) と、並列候補評価用の補助スレッド群 (`evaluationWorkers[]`、各要素が独自の `std::jthread` を保持) を持ちます。

デストラクタは各スレッドに対して「停止をリクエストする」ことは行いますが、**実際にスレッドが停止するまで `join()` して待つ処理が一切ありません**。

```cpp
// src/NoiseShaperLearner.cpp (L80-99)
NoiseShaperLearner::~NoiseShaperLearner()
{
    // Transition:
    // Running/Starting -> Stopping
    convo::publishAtomic(workerState, WorkerState::Stopping, std::memory_order_release);
    convo::publishAtomic(stopRequested, true, std::memory_order_release);
    {
        const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
        evaluationWorkersShouldExit = true;
    }
    evaluationDispatchCv.notify_all();
    intervalCv_.notify_all();

    if (workerThread.joinable())
        workerThread.request_stop();

    // Transition:
    // Stopping -> Idle
    convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
}
```

`workerThread.request_stop()` は `stop_token` に停止要求を伝えるだけで、スレッド関数の実際の終了を待ちません。`evaluationWorkers[]` の各補助スレッドに至っては `request_stop()` すら呼ばれておらず（`evaluationWorkersShouldExit` フラグと `notify_all()` のみに依存）、この関数はいずれのスレッドも「停止した」ことを確認せずに `return` します。

一方、`NoiseShaperLearner.h` のメンバ宣言順序は次の通りです（抜粋、宣言順）:

```cpp
// src/NoiseShaperLearner.h (L238-286、宣言順)
    AudioEngine& engine;
    LockFreeRingBuffer<AudioBlock, 4096>& captureQueue;
    convo::RCUReader rcuReader;

    std::jthread workerThread;                              // ← L242
    std::atomic<WorkerState> workerState { WorkerState::Idle };
    ...(省略)...
    AudioSegmentBuffer segmentBuffer;
    CmaEsOptimizer optimizer;
    std::array<EvaluationWorkerSlot, kMaxParallelEvaluators> evaluationWorkers {};  // ← L268
                                                              //   (EvaluationWorkerSlot は内部に std::jthread thread を保持)
    int activeEvaluationWorkerCount = 1;
    int activeAuxEvaluationWorkerCount = 0;
    ...(省略)...
    convo::ScopedAlignedPtr<double> candidatePopulation;     // ← L281
    convo::ScopedAlignedPtr<double> candidateFitness;        // ← L282
    ...
    convo::ScopedAlignedPtr<double> sharedMappedPopulation;  // ← L285
```

C++はメンバを**宣言順の逆順**で破棄します。そのため実際の破棄順序は：

`sharedMappedPopulation` → `candidateFitness` → `candidatePopulation` →（中略）→ `evaluationWorkers`（各要素の `jthread` はここで初めて自動 `join` される）→（中略）→ `workerThread`（ここで初めて自動 `join` される）

すなわち、**`candidatePopulation` / `candidateFitness` / `sharedMappedPopulation`（CMA-ESの個体群・適応度バッファ）は、`workerThread` と `evaluationWorkers[]` のどちらか一方でも `join` される前に、ヒープメモリごと解放されます**。

これらのバッファは実際にワーカースレッドから直接読み書きされています：

```
src/NoiseShaperLearner.cpp:605-657 付近  runEvaluationJobsForWorker() / evaluationWorkerMain()
                                         → candidatePopulationMatrix(), sharedMappedPopulation を読み取り

src/NoiseShaperLearner.cpp:810,818,940,961,969,971
                                         → workerThreadMain() の CMA-ES ループ (optimizer.sample/update 相当)
                                           → candidatePopulationMatrix(), candidateFitnessData() を読み書き
```

`candidatePopulationMatrix()` / `candidateFitnessData()` はいずれも `ScopedAlignedPtr` の生ポインタをそのまま返す薄いラッパです：

```cpp
// src/NoiseShaperLearner.h (L116-133)
    double (*candidatePopulationMatrix() noexcept)[CmaEsOptimizer::kDim]
    {
        return reinterpret_cast<double (*)[CmaEsOptimizer::kDim]>(candidatePopulation.get());
    }
    ...
    double* candidateFitnessData() noexcept
    {
        return candidateFitness.get();
    }
```

**再現条件**: `NoiseShaperLearner` インスタンスが「学習が実行中（`workerThread` および/または `evaluationWorkers[]` がCMA-ESの1世代分の計算を実行中）」のタイミングで破棄される場合（アプリ終了時に学習セッションが走ったままだった場合、あるいは所有元が明示的な `stopLearning()`+`join()` を行わずにオブジェクトを解放した場合）に発生します。`stopLearning()` 自身も同じ「joinしない」パターンであるため（下記参照）、外部からの正しい停止手順を踏んでも回避できません。

### 補強証拠1: `stopLearning()` も同じ欠陥を持つ

```cpp
// src/NoiseShaperLearner.cpp (L190-210)
void NoiseShaperLearner::stopLearning()
{
    convo::publishAtomic(workerState, WorkerState::Stopping, std::memory_order_release);
    convo::publishAtomic(stopRequested, true, std::memory_order_release);
    intervalCv_.notify_all();

    stopEvaluationWorkers();   // ★ こちらは内部で各補助スレッドを正しく join している

    if (workerThread.joinable())
        workerThread.request_stop();   // ★ しかし主スレッドは request_stop のみで join なし

    evaluationDispatchCv.notify_all();
    convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
}
```

`stopEvaluationWorkers()` 自体は補助スレッドを正しく `join()` しています（下記）が、**主スレッド `workerThread` は `stopLearning()` の中でも一度も `join()` されません**。

```cpp
// src/NoiseShaperLearner.cpp (L489-505)
void NoiseShaperLearner::stopEvaluationWorkers() noexcept
{
    ...
    for (int workerIndex = 0; workerIndex < activeAuxEvaluationWorkerCount; ++workerIndex)
    {
        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
        if (slot.thread.joinable())
        {
            slot.thread.request_stop();
            slot.thread.join();      // ★ 補助スレッドはここで正しく join されている
        }
    }
}
```

### 補強証拠2: `startLearning()` は再利用前に明示的な `join()` が必要であることを示している

```cpp
// src/NoiseShaperLearner.cpp (startLearning() 冒頭付近)
    if (isRunning() || workerThread.joinable())
    {
        stopLearning();
        if (workerThread.joinable())
            workerThread.join();   // ★ ここでは明示的に join している
    }
```

`startLearning()` 自身が「`stopLearning()` だけでは不十分で、再利用前に明示的な `join()` が必要」であることを実装で示しています。デストラクタにはこの最後の一手が欠けています。

### 補強証拠3: 同一パターンを正しく実装している他クラスとの比較

コードベース内で `std::jthread`/`std::thread` をメンバに持つクラスは他に3つありますが、いずれも「デストラクタ本体の中で明示的に `join()` してから戻る」という正しいパターンを実装しています（`AudioEngine::~AudioEngine()` は `stopRebuildThread()` 経由で `rebuildThread.join()`、`shutdownWorkerThread()` 経由で内部の `WorkerThread` を停止／`WorkerThread::~WorkerThread()` は `stop()` 経由で `thread.join()`／`DeferredFreeThread::~DeferredFreeThread()` は `shutdownAndDrain()` 経由で `thread.join()`）。`NoiseShaperLearner` だけがこの確立されたパターンから外れています。

### 影響
学習セッションが進行中のままアプリ終了・機能無効化・エンジン破棄等で `NoiseShaperLearner` が破棄されると、CMA-ES用バッファへのUse-After-Freeが発生しうる、クラッシュ・ヒープ破壊クラスの重大な不具合です。発生頻度はタイミング依存（スレッドがちょうど計算中かどうか）のため、毎回再現するとは限らず、負荷や実行環境によって断続的に発生する「たまに落ちる」系の不具合として現れる可能性があります。

### 修正パッチ

デストラクタ本体で、メンバ破棄が始まる前に全スレッドを確実に停止・joinさせます（`stopEvaluationWorkers()` の補助スレッドjoinパターンと、`startLearning()` の `workerThread.join()` パターンを、デストラクタ自身にも適用する形です）。

```diff
@@ src/NoiseShaperLearner.cpp (L80-99) @@
 NoiseShaperLearner::~NoiseShaperLearner()
 {
     // Transition:
     // Running/Starting -> Stopping
     convo::publishAtomic(workerState, WorkerState::Stopping, std::memory_order_release);
     convo::publishAtomic(stopRequested, true, std::memory_order_release);
     {
         const std::scoped_lock<std::mutex> lock(evaluationDispatchMutex);
         evaluationWorkersShouldExit = true;
     }
     evaluationDispatchCv.notify_all();
     intervalCv_.notify_all();
 
-    if (workerThread.joinable())
-        workerThread.request_stop();
+    // ★ Bug#8 修正: メンバ破棄（candidatePopulation 等の解放）が始まる前に、
+    //   全スレッドの終了を実際に待つ。request_stop() のみでは、
+    //   workerThread / evaluationWorkers[] がまだ CMA-ES バッファへアクセス
+    //   している最中にメンバが破棄され、Use-After-Free を引き起こす。
+    //   （stopEvaluationWorkers() が補助スレッドに対して行っているのと同じ
+    //     join パターンを、主スレッドと補助スレッドの両方に適用する）
+    for (int workerIndex = 0; workerIndex < activeAuxEvaluationWorkerCount; ++workerIndex)
+    {
+        auto& slot = evaluationWorkers[static_cast<size_t>(workerIndex)];
+        if (slot.thread.joinable())
+        {
+            slot.thread.request_stop();
+            slot.thread.join();
+        }
+    }
+
+    if (workerThread.joinable())
+    {
+        workerThread.request_stop();
+        workerThread.join();
+    }
 
     // Transition:
     // Stopping -> Idle
     convo::publishAtomic(workerState, WorkerState::Idle, std::memory_order_release);
 }
```

> **補足**: `workerThreadMain()` / `evaluationWorkerMain()` はいずれも `stop_token`/`evaluationWorkersShouldExit` を短いポーリング間隔（数十〜数百ms オーダー）でチェックしていることを確認済みのため、上記の `join()` 追加によってデストラクタが無期限にブロックするリスクは低いと判断しています。ただし `evaluatePopulation()` 内の1世代分の評価処理そのものが長時間かかる場合、その処理が区切りに達するまで `join()` は待つことになります。

> **恒久対応（任意）**: 上記はデストラクタ本体を直す最小修正です。より構造的な対策として、`workerThread` および `evaluationWorkers` をクラスの**最後**に宣言し直す（＝スレッドが参照する全データより後に構築・先に破棄されるようにする、という「スレッドは最後に宣言する」というC++の定石）方法もあります。ただしこの場合、コンストラクタの初期化順（メンバ初期化リストの順序は宣言順に従う必要がある）や、他のメンバとの依存関係を再確認する必要があるため、まずは上記の最小修正を適用し、リファクタリングは別途ご判断ください。

---

## 誤検知として除外した候補（検証の記録）

指摘前に実装を確認し、**問題なし**と判断したものです（推測での指摘を避けるため記録します）。

1. **`MKLNonUniformConvolver::Layer` の AoS/SoA 二重保持** — メモリに記載の既知課題ですが、現在のヘッダでは `irFreqDomain`/`fdlBuf` は既にスクラッチ用の小サイズ（`partStride`/`2*partStride`）に修正済みであることを確認しました。追加の指摘なし。
2. **`processOutput()` のNaN/Infサニタイズが Float 経路にだけ無い疑い** — 当初 `DSPCoreFloat.cpp` の `process()` に最終スクラブが無いように見えましたが、共有ヘルパー `AudioEngine::DSPCore::processOutput()`（`DSPCoreIO.cpp`）に同等のAVX2 NaN/Infスクラブが存在し、Float経路もこれを呼び出していることを確認しました。Double経路（`processDoubleToBuffer`）は同等のロジックを自前でインライン実装しています。両経路とも保護は同等です。
3. **`pushAdaptiveCaptureBlocks` が複数ファイルに存在** — `DSPCoreDouble.cpp` と `DSPCoreIO.cpp` の双方に匿名名前空間内の別定義が存在しますが、匿名名前空間は翻訳単位ごとに内部リンケージのため、リンクエラーにはなりません（単なるコード重複であり、機能上の不具合ではありません）。
4. **`softClipBlockAVX2` の `prevSample`/`softClipPrevSample` が未使用に見える件** — Float版のコメント「状態更新のみ（ADAA用にフィールド残す）」より、将来のADAA（Antiderivative Anti-Aliasing）実装のために意図的に保持されているフィールドであることが確認できました。現状は未使用ですが、開発者の設計意図が明記されているため指摘対象から除外しました。
5. **`softClipBlockAVX2` の Double版とFloat版で実装が大きく異なる件（AVX2 vs スカラーのみ、NaNガードの有無）** — 呼び出し元の直前後を追跡した結果、いずれの経路でも最終的に `processOutput`/`processDoubleToBuffer` 内の無条件NaN/Infスクラブを通過するため、実害としての違いは確認できませんでした。
6. **`ConvolverProcessor::process()` の遅延補正クロスフェードが線形補間である件** — メモリに記載の「クロスフェードは等電力補間を使うべき」という原則との整合性を懸念しましたが、これは新旧2エンジンの出力（無相関）を混ぜる場面の話であり、`ConvolverProcessor.Runtime.cpp` L381-534 の対象は「同一ディレイラインの新旧読み出し位置」（強く相関した信号）のクロスフェードです。相関信号の再ターゲットには線形補間が正しい選択であり、実際に Wet/Dry ミックス側（同ファイル L588-589, 659-660）では `equalPowerSin()` による等電力補間が別途正しく使われていることを確認しました。バグではありません。
7. **`OutputFilter.cpp` のBiquad係数符号（RBJ Audio EQ Cookbook との整合性）** — LPF/HPFの係数導出式とDirect Form II Transposedの実装を照合し、正規化された `a1'`/`a2'` の符号がCookbookの差分方程式と一致することを確認しました。バグではありません。
8. **`PsychoacousticDither.h` のRTスレッドから `vdRngUniform`（MKL VSL, RT禁止操作）が呼ばれる可能性** — リングバッファ方式を採用しており、Audio Thread側の消費関数 `popUniformFromRing()` はリング枯渇時に `vdRngUniform` を呼ばず、MKLを一切使わないXorshiftベースの `fallbackUniform()` にフォールバックすることを確認しました。`vdRngUniform` は `refillRandomRingNonRt()`（名前の通りWorkerスレッド専用）からのみ呼ばれています。バグではありません。
9. **`LatticeNoiseShaper.h` の状態クランプが2箇所で異なる上限値（`advanceState`は±2.0、`clampStateSIMD`は±1e12）を使っている件** — 一見不整合に見えますが、`std::clamp`（`advanceState`側）はNaNを通過させてしまう一方、`clampStateSIMD`が使う `_mm256_min_pd`/`_mm256_max_pd` はIntel SIMDの仕様上NaNを第2オペランド（クランプ境界値）に置き換える副次効果があり、ブロック単位でのNaN回収の役割を果たしています。意図的な多層防御として機能しており、バグではありません。
10. **`std::jthread`/`std::thread` をメンバに持つ他クラス（`AudioEngine`, `core::WorkerThread`, `DeferredFreeThread`）に Bug#8 と同種の破棄順序問題がないか** — 全て確認し、いずれも「デストラクタ本体内で明示的に `join()` を完了させてから戻る」という正しいパターンを実装していることを確認しました（詳細はBug#8参照）。`NoiseShaperLearner` のみがこのパターンから外れています。

---

## 調査範囲についての注記（2026-07-17 第2回調査で更新）

**追加で精査したファイル（第2回）**: `NoiseShaperLearner.cpp/h`（CMA-ES本体, 全体）, `ConvolverProcessor.Runtime.cpp`（クロスフェード・Wet/Dryミックス・レイテンシ補正）, `OutputFilter.cpp`（Biquad係数・SIMDステレオ処理）, `PsychoacousticDither.h`（RTスレッド安全性・RNGリングバッファ）, `LatticeNoiseShaper.h`（格子フィルタ状態クランプ）, `core/WorkerThread.h/cpp`, `AudioEngine.CtorDtor.cpp`（デストラクタ順序の横断検証）。

**累計で精査したファイル**:
`MKLNonUniformConvolver.cpp/h`, `AudioEngine.Processing.DSPCoreDouble/Float/IO/BlockDouble/AudioBlock/DSPCoreLifecycle.cpp`, `EQProcessor.Processing.cpp`, `CustomInputOversampler.cpp`, `DspNumericPolicy.h`, `Fixed15TapNoiseShaper.h`, `DiagnosticsConfig.h`, `core/EpochDomain.h`, `DeferredDeletionQueue.h`, `DeferredFreeThread.h`, `ISRRetireRouter.cpp/h`, `DSPLifetimeManager.h`, `AudioEngine.h`（retire関連 + デストラクタ）, `ISRRuntimePublicationCoordinator.cpp`, `RefCountedDeferred.h`, `ProgressiveUpgradeThread.cpp`, `IRConverter.cpp`, `CacheManager.cpp`, `NoiseShaperLearner.cpp/h`, `ConvolverProcessor.Runtime.cpp`, `OutputFilter.cpp`, `PsychoacousticDither.h`, `LatticeNoiseShaper.h`, `core/WorkerThread.h/cpp`, `AudioEngine.CtorDtor.cpp`。

以下は261ファイル中、**まだ未精査**です: `AllpassDesigner.cpp/h`（Mixed-Phase設計、非RT）, `TruePeakDetector.cpp`, `LoudnessMeter.cpp`（RTメータリング）, `FixedNoiseShaper.h`（RTノイズシェーパー, Fixed15Tap以外の実装）, `ConvolverProcessor.StateAndUI.cpp`（状態マシン, 1035行）, `ConvolverProcessor.MixedPhase.cpp`（868行）, その他 `ConvolverProcessor.*.cpp` 群, `SafeStateSwapper.h`（RCUスワップ機構本体）, `SnapshotFactory.cpp`, `SnapshotCoordinator.h`, `CmaEsOptimizer` 本体の数値アルゴリズム部分（`NoiseShaperLearner.cpp` からの呼び出しは確認済みだが、CMA-ES自体の数学的実装は未精査）, UI関連一式（`MainWindow.cpp`, `DeviceSettings.cpp`, `ConvolverControlPanel.cpp` 等）。「つづけて」とご指示いただければ、`SafeStateSwapper.h`（RCU中核）と `ConvolverProcessor.StateAndUI.cpp`（状態マシン）から着手します。
