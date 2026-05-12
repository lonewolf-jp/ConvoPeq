# Phase 0: 完全メンバ分類台帳

> 目的：Immutable Runtime Graph + Per-thread DSP State 移行の起点として、
> すべての DSP 関連メンバを4分類に完全割り当てし、移行時の判断基準とする。
>
> 分類記号（terms.md 2節・2.5節に基づく必須コメント）:
>
> - `IMMUTABLE_RUNTIME` : RuntimeGraph に移行。publish 後変更禁止。
> - `DSP_THREAD_STATE`  : DSPExecutionState に移行。audio thread confined。
> - `WORKER_ONLY`       : Builder/Loader スレッド専有。RT 非関与。
> - `UI_ONLY`           : UI/message thread 専有。RT 非関与。
> - `ENGINE_CONTROL`    : AudioEngine 本体が保持する制御状態（非DSP）。

---

## 1. DSPCore（src/audioengine/AudioEngine.h）

### 1-1. DSPCore 直下フィールド

| フィールド | 型 | 分類 | 移行先 | 備考 |
|---|---|---|---|---|
| `convolver` | `ConvolverProcessor` | DSP_THREAD_STATE | DSPExecutionState.convolverState | §1-3参照 |
| `eq` | `EQProcessor` | DSP_THREAD_STATE | DSPExecutionState.eqState | §1-4参照 |
| `dcBlockerState` | `DCBlockerRuntimeState*` | DSP_THREAD_STATE | DSPExecutionState.dcState | §1-2参照 |
| `convolverState` | `ConvolverRuntimeState*` | DSP_THREAD_STATE | DSPExecutionState（adapter） | prepare/bindを統合 |
| `eqState` | `EQRuntimeState*` | DSP_THREAD_STATE | DSPExecutionState（adapter） | prepare/bindを統合 |
| `rampState` | `RampRuntimeState*` | DSP_THREAD_STATE | DSPExecutionState.rampState | §1-5参照 |
| `historyState` | `HistoryRuntimeState*` | DSP_THREAD_STATE | DSPExecutionState.historyState | §1-6参照 |
| `dither` | `PsychoacousticDither` | DSP_THREAD_STATE | DSPExecutionState.ditherState | **rngProducerThread要注意→§6** |
| `fixedNoiseShaper` | `FixedNoiseShaper` | DSP_THREAD_STATE | DSPExecutionState.fixedNsState | §1-7参照 |
| `fixed15TapNoiseShaper` | `Fixed15TapNoiseShaper` | DSP_THREAD_STATE | DSPExecutionState.fixed15NsState | §1-8参照 |
| `adaptiveNoiseShaper` | `LatticeNoiseShaper` | DSP_THREAD_STATE | DSPExecutionState.adaptiveNsState | coeffsはAdaptiveCoeffSnapshot→§5 |
| `outputFilter` | `OutputFilter` | DSP_THREAD_STATE | DSPExecutionState.outputFilterState | §1-9参照 |
| `oversampling` | `CustomInputOversampler` | DSP_THREAD_STATE | DSPExecutionState.oversamplingState | taps/coeffs部のみIMMUTABLE→§1-10 |
| `oversamplingFactor` | `size_t` | IMMUTABLE_RUNTIME | RuntimeGraph.oversamplingFactor | prepare時確定 |
| `activeOversamplingType` | `OversamplingType` | IMMUTABLE_RUNTIME | RuntimeGraph.oversamplingType | prepare時確定 |
| `ditherBitDepth` | `int` | IMMUTABLE_RUNTIME | RuntimeGraph.ditherBitDepth | prepare時確定 |
| `noiseShaperType` | `NoiseShaperType` | IMMUTABLE_RUNTIME | RuntimeGraph.noiseShaperType | prepare時確定 |
| `activeAdaptiveCoeffGeneration` | `uint32_t` | DSP_THREAD_STATE | DSPExecutionState（コピー値） | Adaptive更新追跡 |
| `activeAdaptiveCoeffBankIndex` | `int` | DSP_THREAD_STATE | DSPExecutionState（コピー値） | Adaptive更新追跡 |
| `currentCaptureSessionId` | `uint64_t` | IMMUTABLE_RUNTIME | RuntimeGraph.captureSessionId | build時確定 |
| `runtimeUuid` | `uint64_t` | IMMUTABLE_RUNTIME | RuntimeGraph.runtimeUuid | build時確定 |
| `sampleRate` | `double` | IMMUTABLE_RUNTIME | RuntimeGraph.sampleRate | prepare時確定 |
| `alignedL` / `alignedR` | `double*` (mkl_malloc) | DSP_THREAD_STATE | DSPExecutionState.scratch | 64byte align scratch |
| `alignedCapacity` | `size_t` | DSP_THREAD_STATE | DSPExecutionState（scratch管理） | |
| `maxSamplesPerBlock` | `int` | IMMUTABLE_RUNTIME | RuntimeGraph.maxSamplesPerBlock | |
| `maxInternalBlockSize` | `int` | IMMUTABLE_RUNTIME | RuntimeGraph.maxInternalBlockSize | |
| `ownerEngine` | `AudioEngine*` | ENGINE_CONTROL | 廃止（graph→stateの参照で代替） | |
| `dryBypassBufferDoubleL/R` | `double*` | DSP_THREAD_STATE | DSPExecutionState.dryBypassBuffer | crossfade dry path |
| `dryBypassCapacityDouble` | `int` | DSP_THREAD_STATE | DSPExecutionState（buffer管理） | |

### 1-2. DCBlockerRuntimeState（DSPCore内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `outputL`, `outputR` (UltraHighRateDCBlocker) | DSP_THREAD_STATE | DSPExecutionState.dcBlocker.outputPair |
| `inputL`, `inputR` (UltraHighRateDCBlocker) | DSP_THREAD_STATE | DSPExecutionState.dcBlocker.inputPair |
| `oversampledL`, `oversampledR` (UltraHighRateDCBlocker) | DSP_THREAD_STATE | DSPExecutionState.dcBlocker.oversampledPair |

**UltraHighRateDCBlocker内部：**

| フィールド | 分類 | 備考 |
|---|---|---|
| `m_state` (double) | DSP_THREAD_STATE | DC追跡一次遅延値 |
| `m_alpha` (double) | IMMUTABLE_RUNTIME | サンプルレート依存係数、prepare時確定 |

### 1-3. ConvolverRuntimeState（DSPCore内ネスト）

bindアダプタ。移行後は不要（DSPExecutionState.convolverState として統合）。

### 1-4. EQRuntimeState（DSPCore内ネスト）

bindアダプタ。移行後は不要（DSPExecutionState.eqState として統合）。

### 1-5. RampRuntimeState（DSPCore内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `fadeInSamplesLeft` (int) | DSP_THREAD_STATE | DSPExecutionState.ramp.fadeInSamplesLeft |
| `bypassFadeGainDouble` (double) | DSP_THREAD_STATE | DSPExecutionState.ramp.bypassFadeGain |
| `bypassedDouble` (bool) | DSP_THREAD_STATE | DSPExecutionState.ramp.bypassed |

### 1-6. HistoryRuntimeState（DSPCore内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `fixedLatencyBufferL` (double*) | DSP_THREAD_STATE | DSPExecutionState.history.fixedLatBufL |
| `fixedLatencyBufferR` (double*) | DSP_THREAD_STATE | DSPExecutionState.history.fixedLatBufR |
| `fixedLatencyBufferSize` (int) | DSP_THREAD_STATE | DSPExecutionState.history（buffer管理） |
| `fixedLatencyWritePos` (int) | DSP_THREAD_STATE | DSPExecutionState.history.writePos |
| `fixedLatencySamples` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.fixedLatencySamples |
| `softClipPrevSample` (double) | DSP_THREAD_STATE | DSPExecutionState.history.softClipPrev |

### 1-7. FixedNoiseShaper（convo::FixedNoiseShaper）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `coeffs[ORDER]` (double) | IMMUTABLE_RUNTIME | RuntimeGraph.fixedNsCoeffs（プリセット選択） |
| `errors[MAX_CH][ORDER]` (double) | DSP_THREAD_STATE | DSPExecutionState.fixedNsState.errors |
| `writePos` (int) | DSP_THREAD_STATE | DSPExecutionState.fixedNsState.writePos |
| `rngState` (Xoshiro256State) | DSP_THREAD_STATE | DSPExecutionState.fixedNsState.rngState |
| `currentBitDepth` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.ditherBitDepth と同期 |
| `scale`, `invScale` (double) | IMMUTABLE_RUNTIME | ditherBitDepthから導出 |
| `diagSumSqL/R`, `diagPeakAbs` 等 | DSP_THREAD_STATE | DSPExecutionState（diagnostics buffer） |
| `errorBufferL/R`, `errorWritePos` | DSP_THREAD_STATE | DSPExecutionState（diagnostics buffer） |
| `needsReset` (atomic) | DSP_THREAD_STATE | DSPExecutionState（resetフラグ） |

### 1-8. Fixed15TapNoiseShaper（convo::Fixed15TapNoiseShaper）

FixedNoiseShaper と同構造。移行先はそれぞれ `DSPExecutionState.fixed15NsState`。

### 1-9. OutputFilter（convo::OutputFilter）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `hcCoeff`, `lcCoeff`, `hpfCoeff`, `lpCoeff` (BiquadCoeff) | IMMUTABLE_RUNTIME | RuntimeGraph.outputFilterCoeffs |
| `hcState`, `lcState`, `hpfState`, `lpState` (BiquadState) | DSP_THREAD_STATE | DSPExecutionState.outputFilterState |

**BiquadState内部：**

| フィールド | 分類 |
|---|---|
| `w1`, `w2` (double) | DSP_THREAD_STATE |

### 1-10. CustomInputOversampler（DSP_THREAD_STATE + IMMUTABLE_RUNTIME 分離）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `upsampleRatio` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.oversamplingFactor と同一 |
| `activePreset` (Preset enum) | IMMUTABLE_RUNTIME | RuntimeGraph.oversamplingType と対応 |
| `numStages` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.osNumStages |
| `maxInputBlockSize`, `maxUpsampledBlockSize` | IMMUTABLE_RUNTIME | RuntimeGraph.osMaxBlockSizes |
| `stages[n].taps`, `centerTap`, `centerParity` 等の係数群 | IMMUTABLE_RUNTIME | RuntimeGraph.osStageCoeffs[n]（FIRタップ） |
| `stages[n].convCoeffs`, `convCoeffsReversed` | IMMUTABLE_RUNTIME | RuntimeGraph.osStageCoeffs[n] |
| `stages[n].upHistory[upHistorySize]` | DSP_THREAD_STATE | DSPExecutionState.osState.stages[n].upHistory |
| `stages[n].downHistory[downHistorySize]` | DSP_THREAD_STATE | DSPExecutionState.osState.stages[n].downHistory |
| `stages[n].centerDelayInput` | DSP_THREAD_STATE | DSPExecutionState.osState.stages[n].centerDelay |
| `workA`, `workB` (work buffers) | DSP_THREAD_STATE | DSPExecutionState.osState.workBuffers |
| `workCapacity`, `blockChannels` | DSP_THREAD_STATE | DSPExecutionState.osState（buffer管理） |
| `corruptionDetected` (atomic) | DSP_THREAD_STATE | DSPExecutionState.osState |

### 1-11. ProcessingState（DSPCore内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `eqBypassed` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.eqBypassed |
| `convBypassed` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.convBypassed |
| `order` (ProcessingOrder) | IMMUTABLE_RUNTIME | RuntimeGraph.processingOrder |
| `analyzerSource` (AnalyzerSource) | IMMUTABLE_RUNTIME | RuntimeGraph.analyzerSource |
| `analyzerEnabled` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.analyzerEnabled |
| `softClipEnabled` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.softClipEnabled |
| `saturationAmount` (float) | IMMUTABLE_RUNTIME | RuntimeGraph.saturationAmount |
| `inputHeadroomGain` (double) | IMMUTABLE_RUNTIME | RuntimeGraph.inputHeadroomGain |
| `outputMakeupGain` (double) | IMMUTABLE_RUNTIME | RuntimeGraph.outputMakeupGain |
| `convolverInputTrimGain` (double) | IMMUTABLE_RUNTIME | RuntimeGraph.convInputTrimGain |
| `convHCMode`, `convLCMode`, `eqLPFMode` | IMMUTABLE_RUNTIME | RuntimeGraph.filterModes |
| `adaptiveCoeffBankIndex` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.adaptiveCoeffBankIndex |
| `adaptiveCoeffSet` (const CoeffSet*) | IMMUTABLE_RUNTIME | RuntimeGraph.adaptiveCoeffSnapshot（raw pointer、graph寿命と同期） |
| `adaptiveCoeffGeneration` (uint32_t) | IMMUTABLE_RUNTIME | RuntimeGraph.adaptiveCoeffGeneration |
| `adaptiveCaptureSampleRateHz` (double) | IMMUTABLE_RUNTIME | RuntimeGraph に含める |
| `adaptiveCaptureBitDepth` (int) | IMMUTABLE_RUNTIME | RuntimeGraph に含める |
| `captureSessionId` (uint64_t) | IMMUTABLE_RUNTIME | RuntimeGraph.captureSessionId |
| `adaptiveCaptureQueue` (void* / queue ptr) | DSP_THREAD_STATE | DSPExecutionState.captureQueueRef（nullチェックして使用） |

---

## 2. ConvolverProcessor（src/ConvolverProcessor.h）

### 2-1. StereoConvolver（ConvolverProcessor内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `irData` (double*) | IMMUTABLE_RUNTIME | IRBank.irData（shared immutable） |
| `nucConvolvers` (NUC convolver array) | DSP_THREAD_STATE | DSPExecutionState.conv.nucStates（FDL partition history） |
| `irDataLength` (int) | IMMUTABLE_RUNTIME | IRBank.length |
| `latency` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.convolverAlgorithmLatency |
| `irLatency` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.irPeakLatency |
| `callQuantumSamples` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.callQuantum |
| `prewarmedMaxSamples` (int) | IMMUTABLE_RUNTIME | RuntimeGraph |
| `storedSampleRate` (double) | IMMUTABLE_RUNTIME | RuntimeGraph.sampleRate |
| `storedMaxFFTSize` (int) | IMMUTABLE_RUNTIME | IRBank / partition layout |
| `storedKnownBlockSize` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.maxSamplesPerBlock |
| `storedFirstPartition` (int) | IMMUTABLE_RUNTIME | IRBank.firstPartition |
| `storedScale` (double) | IMMUTABLE_RUNTIME | IRBank.scale |
| `storedDirectHeadEnabled` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.directHeadEnabled |
| `retired` (atomic<bool>) | WORKER_ONLY | reclaim時のフラグ |

### 2-2. ConvolverProcessor 直下フィールド（process経路に関与するもの）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `m_activeEngine` (atomic<StereoConvolver*>) | DSP_THREAD_STATE | RCU read → RuntimeGraph.irBank 参照へ移行 |
| `delayBuffer[2]` / `delayWritePos` | DSP_THREAD_STATE | DSPExecutionState.conv.bypassDelayBuf |
| `delayBufferCapacity` | DSP_THREAD_STATE | DSPExecutionState.conv（buffer管理） |
| `crossfadeGain` (LinearRamp) | DSP_THREAD_STATE | DSPExecutionState.conv.crossfadeGain |
| `mixTarget` / `mixSmoother` | DSP_THREAD_STATE | DSPExecutionState.conv.mixSmoother |
| `bypassed` (atomic<bool>) | DSP_THREAD_STATE | RuntimeGraph.convBypassed（audio thread read） |
| `dryBuffer` / `dryBufferStorage` | DSP_THREAD_STATE | DSPExecutionState.conv.dryBuf |
| `dryBufferCapacity` | DSP_THREAD_STATE | DSPExecutionState.conv（buffer管理） |
| `oldDryBuffer` / `oldDryBufferStorage` | DSP_THREAD_STATE | DSPExecutionState.conv.oldDryBuf |
| `wetBufferStorage` / `wetBufferCapacity` | DSP_THREAD_STATE | DSPExecutionState.conv.wetBuf |
| `smoothingBuffer` / `smoothingBufferStorage` | DSP_THREAD_STATE | DSPExecutionState.conv.smoothingBuf |
| `latencySmoother` | DSP_THREAD_STATE | DSPExecutionState.conv.latencySmoother |
| `oldDelay` (int) | DSP_THREAD_STATE | DSPExecutionState.conv.oldDelay |
| `pendingLatencyValue` (atomic<double>) | ENGINE_CONTROL | AudioEngine側で atomic 読み取り |
| `latencyResetPending` (atomic<bool>) | ENGINE_CONTROL | AudioEngine側制御 |
| `latencyChangePending` / `lastReportedLatency` | ENGINE_CONTROL | AudioEngine/UI 側 |

### 2-3. ConvolverProcessor 直下フィールド（WORKER_ONLY / UI_ONLY）

| フィールド | 分類 |
|---|---|
| `cachedFFTBuffer` / `cachedFFTBufferCapacity` | WORKER_ONLY（visualization FFT） |
| `cachedLinearMagsBuffer` / `cachedSmoothedMagsBuffer` | WORKER_ONLY（visualization） |
| `fftHandle` / `fftHandleSize` | WORKER_ONLY（visualization FFT plan） |
| `irName` / `irLength` | UI_ONLY |
| `irWaveform` / `irMagnitudeSpectrum` / `irSpectrumSampleRate` | UI_ONLY（表示用） |
| `visualizationDataLock` | UI_ONLY |
| `irCache` / `cacheMutex` / `cacheManager` | WORKER_ONLY（Loaderスレッド） |
| `currentIrFile` / `irFileLock` | WORKER_ONLY |
| `upgradeThread` / `progressiveUpgrade*` | WORKER_ONLY |
| `isLoading` / `isRebuilding` (atomic) | ENGINE_CONTROL |
| `irFinalized` (atomic) | ENGINE_CONTROL |
| `rebuildJob` / `activeLoader` / `loaderTrashBin` | WORKER_ONLY |
| `currentSampleRate` / `isPrepared` | ENGINE_CONTROL |
| `rcuProvider` / `rcuSwapper` | ENGINE_CONTROL（RCU移行対象） |
| `convolverState` / `convolverStateGeneration` | ENGINE_CONTROL（RCU publish state） |
| `deferredFreeThread` | WORKER_ONLY |

---

## 3. EQProcessor（src/eqprocessor/EQProcessor.h）

### 3-1. EQCoeffCache（EQProcessor内ネスト）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `coeffs[]` (EQCoeffsBiquad / EQCoeffsSVF) | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank.bandCoeffs |
| `bandActive[]` / `channelModes[]` | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank.bandActive / modes |
| `filterStructure` | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank.filterStructure |
| `paramsHash` / `generation` | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank（cache key） |
| `sampleRate` / `maxBlockSize` | IMMUTABLE_RUNTIME | RuntimeGraph に同期 |
| `parallelInputBuffer` | DSP_THREAD_STATE | DSPExecutionState.eq.parallelInputBuf **← 要分離** |
| `parallelWorkBuffer` | DSP_THREAD_STATE | DSPExecutionState.eq.parallelWorkBuf **← 要分離** |
| `parallelAccumBuffer` | DSP_THREAD_STATE | DSPExecutionState.eq.parallelAccumBuf **← 要分離** |
| `parallelBufferSize` | DSP_THREAD_STATE | DSPExecutionState.eq（buffer管理） |

### 3-2. EQProcessor 直下フィールド（DSP_THREAD_STATE）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `filterState` (SVF/biquad z状態配列) | DSP_THREAD_STATE | DSPExecutionState.eq.filterState |
| `agcCurrentGain` (double) | DSP_THREAD_STATE | DSPExecutionState.eq.agcGain |
| `agcEnvInput` / `agcEnvOutput` (double) | DSP_THREAD_STATE | DSPExecutionState.eq.agcEnv |
| `smoothTotalGain` (double) | DSP_THREAD_STATE | DSPExecutionState.eq.smoothTotalGain |
| `bypassFadeGain` (double) | DSP_THREAD_STATE | DSPExecutionState.eq.bypassFadeGain |
| `bypassed` (bool) | DSP_THREAD_STATE | RuntimeGraph読み取りに移行 |
| `bypassRequested` (atomic) | ENGINE_CONTROL | AudioEngine 側で管理 |
| `scratchBuffer` / `scratchCapacity` | DSP_THREAD_STATE | DSPExecutionState.eq.scratchBuf |
| `dryBypassBuffer` / `dryBypassCapacity` | DSP_THREAD_STATE | DSPExecutionState.eq.dryBypassBuf |
| `parallelInputBuffer` / `parallelWorkBuffer` / `parallelAccumBuffer` | DSP_THREAD_STATE | DSPExecutionState.eq.parallelBufs |
| `structureOldOutBuffer` / `structureNewOutBuffer` | DSP_THREAD_STATE | DSPExecutionState.eq.structureXfadeBufs |
| `parallelBufferCapacity` / `structureXfadeBufferCapacity` | DSP_THREAD_STATE | DSPExecutionState.eq（buffer管理） |

### 3-3. EQProcessor 直下フィールド（IMMUTABLE_RUNTIME相当）

| フィールド | 分類 | 備考 |
|---|---|---|
| `currentStateRaw` (EQState) | IMMUTABLE_RUNTIME | 係数導出のためのパラメータスナップショット |
| `bandNodes` (BandNode[]: coeffs+active+mode) | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank に統合 |
| `activeBandNodes` (int) | IMMUTABLE_RUNTIME | RuntimeGraph.eqCoeffBank.numActiveBands |
| `currentSampleRate` / `maxInternalBlockSize` | IMMUTABLE_RUNTIME | RuntimeGraph に同期 |
| `nonlinearSaturation` (bool) | IMMUTABLE_RUNTIME | RuntimeGraph.eqSaturation |
| `requestedStructure` / `activeStructure` | IMMUTABLE_RUNTIME + DSP_THREAD_STATE | 切替xfade中はDSP側、完了後はgraph確定 |

### 3-4. EQProcessor AGC係数テーブル（prepare時計算）

| フィールド | 分類 | 移行先 |
|---|---|---|
| `agcAttackCoeff` / `agcReleaseCoeff` / `agcSmoothCoeff` | IMMUTABLE_RUNTIME | RuntimeGraph.eqAgcCoeffs |
| `agcAttackCoeffTable[]` / `agcReleaseCoeffTable[]` / `agcSmoothCoeffTable[]` | IMMUTABLE_RUNTIME | RuntimeGraph.eqAgcCoeffTables（SR毎） |
| `agcCoeffTableCapacity` | IMMUTABLE_RUNTIME | RuntimeGraph |
| `totalGainDbTarget` / `totalGainTarget` | IMMUTABLE_RUNTIME | RuntimeGraph.eqTotalGain |
| `m_pendingAGCChange` (atomic) | ENGINE_CONTROL | AudioEngine 側で管理 |
| `agcResetRequest` (atomic) | ENGINE_CONTROL | AudioEngine 側で管理 |
| `bandResetMask` (atomic) | ENGINE_CONTROL | AudioEngine 側で管理 |
| `agcEnabled` | IMMUTABLE_RUNTIME | RuntimeGraph.eqAgcEnabled |

---

## 4. AudioEngine 本体（DSP制御関連フィールド）

### 4-1. Crossfade / Fading 制御（ENGINE_CONTROL → 移行後縮小）

| フィールド | 分類 | 移行方針 |
|---|---|---|
| `currentDSP` (raw ptr) | ENGINE_CONTROL | activeDSPに統合済み → RuntimeGraph atomic ptr に |
| `activeDSP` (raw ptr) | ENGINE_CONTROL | RuntimeGraph atomic ptr に |
| `fadingOutDSP` (atomic ptr) | ENGINE_CONTROL | Phase 6後は「fadingExecutionState」のみに縮小 |
| `queuedOldDSP` (atomic ptr) | ENGINE_CONTROL | Phase 6後は廃止（queueing不要化） |
| `dspCrossfadeGain` (LinearRamp) | DSP_THREAD_STATE | DSPExecutionState.crossfade.gainRamp |
| `dspCrossfadeDryScaleGain` (LinearRamp) | DSP_THREAD_STATE | DSPExecutionState.crossfade.dryScaleRamp |
| `dspCrossfadeFloatBuffer` / `dspCrossfadeDoubleBuffer` | DSP_THREAD_STATE | DSPExecutionState.crossfade.mixBuf |
| `latencyBufOldL/R` / `latencyBufNewL/R` | DSP_THREAD_STATE | DSPExecutionState.latencyAlign.bufs |
| `latencyWritePos` | DSP_THREAD_STATE | DSPExecutionState.latencyAlign.writePos |
| `latencyDelayOld_RT` / `latencyDelayNew_RT` | DSP_THREAD_STATE | DSPExecutionState.latencyAlign.delaySamples |
| `m_fadeFloatBuffer` / `m_fadeDoubleBuffer` | DSP_THREAD_STATE | DSPExecutionState.crossfade（予備バッファ） |
| `dspCrossfadeUseDryAsOld` (atomic) | ENGINE_CONTROL | Phase 6後に廃止方向 |
| `dspCrossfadePending` (atomic) | ENGINE_CONTROL | Phase 6後に廃止方向 |
| `firstIrDryCrossfadePending` (atomic) | ENGINE_CONTROL | Phase 6後に廃止方向 |

### 4-2. RuntimePublish / EngineRuntime（移行対象）

| フィールド | 移行フェーズ | 方針 |
|---|---|---|
| `runtimePublishState` (atomic) | Phase 7 廃止対象 | engineRuntimeState への一本化後に削除 |
| `runtimePublishRevision` (atomic) | Phase 7 廃止対象 | 同上 |
| `engineRuntimeState` (atomic) | Phase 2→7 で中心化 | 最終は RuntimeGraph atomic ptr に |
| `engineRuntimeRevision` (atomic) | Phase 7 後に廃止 | UUID + generation で代替 |

### 4-3. thread_local（terms.md 3.3節違反・移行必須）

| 箇所 | ファイル:行 | 移行方針 |
|---|---|---|
| `static thread_local convo::RCUReader tls_rcuReader` | EQProcessor.Processing.cpp:15 | 明示的 RCUReader を引数で渡すか DSPExecutionState に組み込む |
| `DSPCore::tls_readerSlot` | AudioEngine.h（static） | EpochManager への明示登録 API に移行 |

---

## 5. AdaptiveCoeffBank / NoiseShaperLearner（特殊移行対象）

| フィールド/構造体 | 分類 | 移行方針 |
|---|---|---|
| `adaptiveCoeffBanks[10×bitdepth×mode]` | ENGINE_CONTROL（バンク管理） | AudioEngine が継続保持 |
| `AdaptiveCoeffBankSlot.coeffSetA/B` (CoeffSet) | IMMUTABLE_RUNTIME | publish時にRuntimeGraphのadaptiveCoeffSnapshot に raw const ptr でスナップショット |
| `AdaptiveCoeffBankSlot.activeIndex` / `generation` | ENGINE_CONTROL | publish時確定、RuntimeGraph.adaptiveCoeffGenerationに反映 |
| `AdaptiveCoeffBankSlot.stateMutex` / `writeLock` | WORKER_ONLY | NoiseShaperLearnerのwrite側のみ使用 |
| `LatticeNoiseShaper.coeffs` | DSP_THREAD_STATE | RuntimeGraph.adaptiveCoeffSnapshot から audio thread で snapshot 読み込み |
| `LatticeNoiseShaper.states[ORDER][CH]` | DSP_THREAD_STATE | DSPExecutionState.adaptiveNsState.states |
| `LatticeNoiseShaper.rngState` | DSP_THREAD_STATE | DSPExecutionState.adaptiveNsState.rngState |

**設計注記**：`LatticeNoiseShaper.coeffs` は rebuild 無しで NoiseShaperLearner から更新されるため、
RuntimeGraph には「最後に採用された coeffs のスナップショット」のみを入れる。
Audio thread 側は DSPExecutionState に保持した係数コピーを使用し、
world 切替は 「coeffs の atomic publish → DSPExecutionState.adaptiveNsState.coeffs 更新」の軽量パスで行う。

---

## 6. PsychoacousticDither の特殊問題（terms.md 3.1節・H-2問題）

`PsychoacousticDither` は内部に `rngProducerThread`（MKLでRNGリングを補充するバックグラウンドスレッド）を持つ。

**問題**：DSPExecutionState に PsychoacousticDither をそのまま入れると、
「per-instance にスレッド付きオブジェクトが存在する」状態になる。

**移行方針**：

1. `rngProducerThread` を PsychoacousticDither から分離し、AudioEngine が1本のワーカーとして保持する（WORKER_ONLY）。
2. RNGリング（`rngRing`, `rngReadPos`, `rngWritePos`）を共有リングとして AudioEngine に切り出す。
3. DSPExecutionState には shaperStateBuffer（歴史値）と rngRing への参照のみを持たせる。
4. `rngRing` への書き込みはワーカー、読み取りは audio thread のみとし、SPSC 設計を維持する。

**現行コードの DeferredDeletionQueue との関係**：
`DeferredDeletionQueue.enqueue()` はロックフリー MPMC で音声スレッドから呼び出し可能（B-1問題の解決に活用可能）。

---

## 7. 分類確認ゲート（実装前必須チェック）

実装前に各追加メンバについて以下を確認すること（terms.md 6節）：

- [ ] mutable state が IMMUTABLE_RUNTIME 構造体に混入していないか
- [ ] process() 経路に alloc / lock / shared_ptr ownership 変更がないか
- [ ] crossfade が「ownership 移動」ではなく「state 複製実行」になっているか
- [ ] thread_local 依存が残っていないか
- [ ] prepare/reset に hidden allocation 再発がないか
- [ ] audio thread が `retireDSP` / `delete` / `shared_ptr::reset` を呼んでいないか
- [ ] 新規メンバに `// IMMUTABLE_RUNTIME` 等の分類コメントが付与されているか
