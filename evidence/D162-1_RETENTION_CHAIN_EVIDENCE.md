# D162-1 — Retention Chain Evidence（read-only 監査・production source 変更 0）

- Date: 2026-09-01
- Authority: 現行 ConvoPeq.md（Generated 2026-09-01 21:47:45 / FRESH 再生成確認済み）
- Scope: D116-OP-2 TRIGGERED 実測（evidence/D116_OP2_soak.log / D116_OP2_memory_soak.csv）を起点とする rebuild generation 系メモリ滞留の原因監査
- Source changes: **0**（`git status src/` の既存差分は CR-α / ND 世代のもので D162-1 追加 0）

## E-1. 実測値（診断 OFF ビルド・Release）

| 項目 | 値 | 出所 |
| --- | --- | --- |
| private growth | 82.1 MB (@4s) → 7,740.6 MB (@426s) = **+7,658.5 MB / 422 s** | evidence/D116_OP2_memory_soak.csv |
| working set | 7,403.7 MB まで並行追従（= 実 resident 滞留・swap 幻影ではない） | 同上 |
| generation rate | ≈7 s / generation → 422 s ≒ **59 generations** | soak log CONV_REBUILD ×59 |
| **滞留率** | 7,658.5 MB / 59 gen ≈ **129.8 MB / generation** | 計算 |
| publish 数 | **10 / 59**（gen = 6,9,12,…,33 の every-3rd パターン、間隔 ≈36 s = crossfade 窓） | [PUBLISH] 行 |
| rebuild tasks | 59 全件が `enqueuePublicationIntentForRuntimeCommit` を通過（handle 登録済み） | Phase 1 追跡 |
| Shutdown audit | routerPendingRetire=1 / deferred=0 / quarantine=0 / oldestAgeMs=361325 | soak log 末尾 |
| DEFERRED/Rejected/enqueueDeferred/discard 痕跡 | **0 行**（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF のため telemetry 非出力。build/CMakeCache.txt:BOOL=OFF 確認） | soak log + CMakeCache |

## E-2. Generation lifecycle chain（Phase 1）

59 gen 全体の経路。ソース行番号は ConvoPeq.md 同世代の現行コード。

1. build: `RuntimeBuilder.cpp:425` `aligned_make_unique<DSPCore>()` → `:437` `result.runtime = runtime.release()`
2. commit 委譲: `AudioEngine.RebuildDispatch.cpp:1358-1390` — `dspToCommit = dspGuard.ptr; dspGuard.ptr = nullptr;` → `:1387 enqueuePublicationIntentForRuntimeCommit(...)`
3. handle 登録: `AudioEngine.Commit.cpp:804 registerDSPHandleForRuntime(newDSP)`（:787-870 全文確認。**submit 後に DSP 処分コードなし**）
4. submit: `Commit.cpp` → `runtimeOrchestrator_->submitPublishRequest(req)`
5. Admission: `PublicationAdmission.cpp:10-60` — Shutdown / StaleGeneration / NotFinalized / Pressure / FadingActive(→Deferred) / Accepted
6. publish: `PublicationExecutor.cpp:55-95` `[PUBLISH]`（gen=6,9,…,33 の 10 件）
7. retire: 正規 published-DSP retire → `DSPLifetimeManager.cpp:40-160` → RetireRouter → reclaim → `destroyDSPCoreNode`（Threading.cpp:17-25）

### 非対称性（本監査の構造的中心）

| 経路 | DSP 処分 | 根拠 |
| --- | --- | --- |
| publish 成功 | retire pipeline 経由で回収 | 正規経路 |
| **Rejected\***（StaleGeneration / NotFinalized / Pressure / Shutdown / PublishFailure） | **処分なし** | `RuntimePublicationOrchestrator.cpp:380-436` 全 case = telemetry + obligation 系 API のみ |
| **Deferred → discard** | **処分なし** | `:685-691 view->discard()` = reason 記録 + finishView のみ |
| Deferred → overwrite | 旧 slot DSP のみ retire | `:441-534` enqueueDeferred の overwrite 時（:461-468・D132 INV-DEFERRED-2）— **非 discard 系で唯一の処分点** |
| DSPGuard 未登録 dtor | direct destroy | `AudioEngine.RebuildDispatch.cpp:932-964`（work70-FIX fallback） |
| warmup retry / obsolete | guard dtor が救済（リークなし） | `:1208-1221` isObsolete→continue、`:1264-1330` Schedule→continue のいずれも ptr 設定済みで dtor 処理 |

→ Rejected / discard 経路に乗った generation の DSPCore は handle 登録されたまま誰も retire しない構造。ただし soak では diagnostics OFF のため **49 件の非 publish generation がどの経路（Deferred→overwrite 消化 or discard or Rejected）を通ったか観測不能**。

## E-3. targetLength チェーン（Phase 3）

- rebuild 経路の `loadedSR = sourceSampleRate = 192000`（`ConvolverProcessor.LoaderThread.cpp:428`）
- `computeTargetIRLength(192000, 76800)` = `192000 × pendingOverride.targetIRLengthSec`（StateAndUI.cpp:942-957）、cap `MAX_IR_LATENCY = 2,097,152`（ConvolverProcessor.h:198）は非拘束
- CONV_STATUS `irLen=192000` → `targetIRLengthSec = 1.0` 確定 → **NUC 入力長 = 192,000 samples**（LoadPipeline.cpp:619→690→799）
- transferIRStateFrom の `len=76800` はソース IR 実体コピー。NUC へは trim 後 192000 が渡る（二重性の解明済み）

## E-4. NUC フットプリント計算（part0 = 2048, tailStart = 4,080, mult = 8）

| Layer | 有効長 | part | numParts(=nextPow2) | FFT | complex | irFreq×2 / fdl×2 / fdlBuf | 合計 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 4,080 | 2,048 | 4 | 4,096 | 2,049 | — | 0.25 MB |
| L1 | 187,920 | 16,384 | 16 | 32,768 | 16,385 | — | 12.50 MB |
| L2 | 0 | — | — | — | — | — | 0 |
| **計** | | | | | | | **12.8 MB / ch → ×2ch = 25.5 MB / DSPCore** |

構造推定 per retained DSPCore: NUC 25.5 + irData(double 192000×2ch) 2.9 + EQ(384k/8192) 1-2 + oversampling 1-3 + misc <1 ≈ **33 MB**

## E-5. 残差

実測 129.8 MB/gen に対し構造推定 ≈33 MB/gen → **~97 MB/gen が DSPCore 単体では説明不能**。
説明候補（Phase 3 分類）:
(a) NUC 実サイズが推定上回る（MKL DFTI descriptor/work buffer が diag allocator 追跡外）
(b) generation-scoped 非 DSPCore 割当（EQ/OS 大型バッファ・RuntimePublishWorld・engine 側 IR AudioBuffer 群・AUTO_GAIN）
(c) retained DSPCore 複数化（H1 経路で複数/世代生存）
(d) JUCE/allocator arena 滞留（WS 追従からは優先度低）
→ 判別は Phase 4 診断ビルド実測に委譲（[MEM_SNAP] の NUC alloc MB / Other MB / DC・SC live が直接 discriminate する）。

## E-6. 既存診断器 inventory（Phase 4 入力・RUNTIME_DIAGNOSTICS=ON で出力）

- `[MEM_SNAP]`（Timer.cpp:1024-1110・timerCallback 毎）: NUC live/alloc/peak/tA/tF/lostFree/zeroAlloc、**DC live / SC live**、RetireRouter pend/trBytes/tracked/ovf/rec/**retiringGen**、Priv/WS、**Other = Priv − NUC − retire**、TRK（**active DSP のみ** total/OS/EQ/AL/LT）
- `[MEM]`（1 s 周期）: Private/WS/Pagefile/PageFaults+Delta
- `[D117_RETIRE] dsp=%p retired/enqueue/epoch`（DSPLifetimeManager.cpp:46/63）/ `[D117_DESTROY]`（Threading.cpp:17-25）= DSP pointer 単位 retire/destroy trace
- 常時出力: `[PUBLISH]` `[CONV_STATUS]` `[CONV_REBUILD]` `[CONV_IR]` `[DSPCORE_PREPARE]` `[EQ_PREPARE]`
- Shutdown: collectDrainAudit（Threading.cpp:74-110）= routerPendingRetire/quarantineResident/activeWorldCount/pendingRetire

### gap（診断ビルドでも未計測・将来最小追加候補）

1. `runtimeDSPHandleMap_` size（AudioEngine.h:4307-4346・size accessor/log なし）
2. generation ごとの admission 分類 counter（Accepted/Deferred/Rejected\* 累積 — [PUBLISH] は成功のみ）
3. discard reason 累積（slot の lastDiscardReason は最終値のみ）
4. retire 時点の per-DSP TrackedMemoryStatistics（TRK は active DSP のみサンプリング）
5. obsolete generation count（guard dtor fallback が silent）

## E-7. Soak 実行条件再現

`--cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 60 --cli-intent-burst-interval-ms 6000 --cli-exit-ms 420000`（Release・CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF）。Phase 4 は同一条件 + `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` で再実行する設計。
