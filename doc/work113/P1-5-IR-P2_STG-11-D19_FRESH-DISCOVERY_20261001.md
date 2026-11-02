# STG-11-D19 Fresh Discovery — Publication Failure / Fallback / User-Visible State Consistency

- Document: `doc/work113/P1-5-IR-P2_STG-11-D19_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D19**
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`
  - baseline Generated `2026-10-01 19:25:35`
  - SHA-256 `80E3E9459D4B8308DE771F2D4FC13BE4E0FD34C01B1F37C4CD7A9B6D30E22CEF`
  - size `5,907,375 B` / `NEWER_SRC_COUNT = 0` = **FRESH**
  - `ConvoPeq(20261001-095757).md` と同一ファイル（coder 側 `ConvoPeq.md`）
- 参照方法: 本 fresh discovery は `ConvoPeq.md` を authority baseline として固定した上で、
  結論の根拠は**実ソースファイル**を直接読んで導出した（md の行番号・旧監査資料・記憶は未使用）。
- commit / push: **未実施**

---

## 0. 判定

```
STG-11-D19
NO-GO — no concrete defect
```

Owner の 7 条件を**すべて**満たす concrete production bug は、本 STG の failure boundary
走査では確認できなかった。

ただしOwner の指示どおり「単に問題なしとしない」。以下を記録する。

- §2: 実際に追跡した failure boundary と、その上で確認した state transition
- §3: concrete harm まで到達しなかった **確定した observation 7 件**（証拠付き）
- §4: 既に D1〜D18 と重複しないことの確認

---

## 1. 追跡したデータフロー

```text
Parameter / UI setter
      ↓  submitRebuildIntent → requestRebuild（message thread, sr>0 && bs>0 で即 dispatch）
Build（RebuildThread / DSPCore 構築）
      ↓  enqueuePublicationIntentForRuntimeCommit
Validation（PublicationAdmission::evaluate）
      ↓  trySubmitImpl（build → crossfade 判定 → 再 build → publish）
Publish（commitRuntimePublication / enqueueRuntimePublicationFireAndForget）
      ↓
Failure 境界（RejectedShutdown / RejectedStaleGeneration / RejectedNotFinalized /
             RejectedPressure / RejectedPublishFailure / DeferredFadingActive）
      ↓
disposition（submitPublishRequest の switch）
      ↓
User-visible / RT-visible state（learningRuntimeState / adaptiveCaptureActiveRt /
             RebuildReason bits / telemetry progress-health / rebuildRequestGeneration）
```

---

## 2. failure boundary と確認できた state transition

### 2.1 Admission 判定の disposition（`RuntimePublicationOrchestrator.cpp:361-471`）

`submitPublishRequest` の switch は `PublicationAdmission::Decision` 全 7 種を網羅。
`default:` 相当の抜けなし。

| Decision | disposition | 確認 |
| --- | --- | --- |
| `Accepted` | return（`trySubmitImpl` 内で publish 済み） | OK |
| `DeferredFadingActive` | `enqueueDeferred(req)` → return | OK（retry であって再 enqueue ではない） |
| `RejectedStaleGeneration` | `onRejected(0)` + `recordFailure(Admission, StaleGeneration)` + `resolveIfRecovery(StaleSuperseded)` + **`retireRegisteredDSP`** | OK。DSP が orphan 化しない |
| `RejectedNotFinalized` | `onRejected(0)` + `recordFailure(Admission, ValidationFailed)` + `postRecoveryFailureSignal`（recovery 時）+ **`retireRegisteredDSP`** | OK。build 失敗 3 経路（build#1 / crossfade rebuild / admission 拒否）で `markTransientFailure` が 1 回だけ呼ばれる設計（A/B/D 統一） |
| `RejectedPressure` | `onRejected(0)` + `recordFailure(Admission, QueuePressure)` + `resolveIfRecovery(Retry)` + `rearmRecoveryRetry`（recovery 時）+ `postRecoveryFailureSignal` + **`retireRegisteredDSP`** | OK。obligation は Live のまま（retry 可能）、本 DSPCore のみ retire |
| `RejectedShutdown` | `onRejected(0)` + `recordFailure(Shutdown, ShutdownRejected)` + `resolveIfRecovery(ShutdownDiscarded)` + **`retireRegisteredDSP`** | OK |
| `RejectedPublishFailure` | `onRejected(0)` + `recordFailure(Execution, PublishFailed)`。retire は**しない** | OK（`trySubmitImpl:294-295` が `destroyRolledBackDSP` で既に物理破棄済み・handle は rollback=Reclaimed。INV-D162-3 により二重破壊禁止） |

**D19-1（failure 後の rollback / retention）は、全 decision で破棄・隔離・retry のいずれかが
決定的に行われている。pending state が success 扱いになる抜けは無い。**

### 2.2 build 失敗（`trySubmitImpl:177-194` / `:245-263`）

```text
worldOwner = buildRuntimePublishWorld(...)      // nullptr
  → lifetime_.retire(newDSPResolved)           // 新 DSP を authority retire
  → stateOwner_.onExecutorFailed(correlationId)
  → recordProgress(Built) + recordFailure(Execution, PublishFailed, "trySubmit:build")
  → return RejectedNotFinalized
```

`RejectedNotFinalized` は §2.1 で `retireRegisteredDSP` まで通るので、
`newDSPResolved` 方面的 disposition は二重にならない（`retire` は冪等）。
crossfade 再 build 失敗（`:245-263`）も同型。

**「build success ≠ publish success」境界（Owner D19-2）は、build 失敗時に
`stateOwner_.onPublished()` / `PublishStage::Published` を呼ばない形で維持されている。**

### 2.3 publish 失敗（`trySubmitImpl:283-319`）

```text
result = executor_.publish(...)                 // PublishResult::PublishFailed
  → lifetime_.destroyRolledBackDSP(newDSPResolved)   // 物理破棄（handle は Reclaimed）
  → stateOwner_.onExecutorFailed(correlationId)
  → recordFailure(Execution, PublishFailed, "trySubmit:publish")
  → recordProgress(PublishStage::Published)          // ← §3 O-3 で記録
  → shutdown 中 ? RejectedShutdown : RejectedPublishFailure
```

`publishImpl`（`PublicationExecutor.cpp:69-78`）は `isCommitted(stage)==false` のとき
**自前で破棄せず** `OwnershipDisposition::CallerDestroy` のみ通知し、caller が
`destroyRolledBackDSP` する。所有権規律（Allocated → Transferred → Committed → Completed）は
破綻していない。

### 2.4 receipt timeout（`AudioEngine.h:4992-5001`）

`commitRuntimePublication` は `isCommitted(stage) && seqId != 0` のときだけ
`waitForPublishReceipt(seqId, 250ms)` する。timeout しても戻り値の stage は
`Success` のままで、caller は失敗扱いしない。これは
「timeout ≠ publish failure（所有権は enqueue 時点で Transferred）」という
明示契約で、rollback すると double ownership / double publish になる。
**意図された semantics であり defect ではない。**

### 2.5 snapshot 作成の抑制（`AudioEngine.Snapshot.cpp:14-173`）

| 早期 return | 先行して消費された flag | 判定 |
| --- | --- | --- |
| `:69` EQ cache unavailable | `m_pendingAGCChange` は **set** 済み（消費前） | 持ち越される。OK |
| `:114` `promoteToStructural` | `m_pendingIRChange`（`:103` の exchange） | ヘッダ契約どおり「構造 intent が先行発行済み」の level marker。writer は `Timer.cpp:959` / `UIEvents.cpp:177` の 2 箇所のみで、両者とも直前に `submitRebuildIntent(Structural)` を発行。OK |
| `:148` `createImpl` == nullptr | `m_pendingNSChange` / `m_pendingAGCChange`（`:118/:123` の exchange）、`m_pendingIRChange`（`:103`） | `createImpl == nullptr` は「新 snapshot が観測中 snapshot と同一」の no-op 抑制。AGC/NS 係数は次回 `getCurrentAdaptiveCoefficients` で再読される。fade 選択のみ失われるが、適用すべき変更が存在しないため実害なし |

### 2.6 IR load / rebuild の早期 return（`ConvolverProcessor.LoadPipeline.cpp`）

```text
:41-44  !isRebuild && !irFile.existsAsFile()  → return false     （状態変更 BEFORE）
:71-79  isRebuild && IRState 不正            → isLoading=false, isRebuilding=false → return false
:34-38  missingState                          → isRebuilding=false → return false
```

いずれも `isLoading = true` / `irFinalized = false`（`:46-47`）より**前**に_return する。
load 失敗で `irLoaded=true && irFinalized=false` が恒久化する経路は無い。

### 2.7 finalize-aware defer の解放（`AudioEngine.Timer.cpp:979-1055`）

```text
finalizeReady = (!irLoaded || irFinalized) && !irLoading && !structuralDeferred
             && !pendingIrChange && !outstandingRebuild
if (finalizeReady || timedOut) { clearRebuildReason(DeferredFinalizeAware); submitRebuildIntent(MustExecute); }
```

`kFinalizeDeferMaxDurationMs = 2000` の**強制 dispatch timeout** があるため、
IR が finalize しない場合でも defer は必ず解放される。
**恒久的な deferral（以後の rebuild が永久に-block）は構造的に起きない。**
（`finalizeReady` の `!outstandingRebuild` 条件は §3 O-1 で述べる通り恒 false だが、
timeout 経路が救っている。）

### 2.8 `submitRebuildIntent` は defer reason をゲートにしない（`AudioEngine.RebuildDispatch.cpp:343-376`）

```text
kind == Structural && isMessageThread && sr > 0 && bs > 0
    → requestRebuild(sr, bs, ...)   // 即 dispatch
    → return
```

`hasRebuildReason(DeferredFinalizeAware)` は `submitRebuildIntent` 内で
**参照されない**（:set されるのみ `:365` / `:449`、かつ `:365` は sr/bs 不在時の別 defer 理由）。
したがって setter が `DeferredFinalizeAware` を立てても、
`:397` で先に出した 1 本目の `submitRebuildIntent` は**遅延されない**。
パラメータ適用は即時に requestRebuild される。

---

## 3. concrete harm まで到達しなかった observation（確定）

いずれも**実在する不整合**だが、Owner の判定条件 5
（state corruption / user-data loss / wrong runtime / wrong audio / stale state /
subsequent operation failure の実発生）へ到達しなかったため **defect としない**。

### O-1 `lastCommittedRebuildGeneration` が一度も書かれない

- 宣言: `AudioEngine.h:2749` `std::atomic<int> lastCommittedRebuildGeneration { 0 }; // commit 完了済み世代`
- 読取 5 箇所: `Parameters.cpp:408` / `:479`、`Timer.cpp:1002`、`RebuildDispatch.cpp:178`、`RuntimeHealthMonitor.cpp:1146`
- **書込み: `src/` 内に 0 箇所。** `setCommittedRebuildGenRef()` は宣言のみ（`RuntimeHealthMonitor.h:194`）で呼び出し 0。

帰結:
- `outstandingRebuild = (rebuildRequestGeneration > 0)` が**恒 true**。
- `setDitherBitDepth`（`:409-422`）と `setNoiseShaperType`（`:480-493`）は
  `shouldDeferRebuild` が常に true → 常に `DeferredFinalizeAware` を立て、
  else 枝の `submitRebuildIntent(RequestRebuildKindEntry)` は到達しない。
- Timer の `finalizeReady` も恒 false → 解放は 2000 ms timeout のみ（§2.7）。
- `RebuildDispatch.cpp:197` の `if (!rebuildOutstanding) valid = false;` は dead。
  ただし `:220` が毎回 `valid = true` を無条件に書くため実害なし。

なぜ defect としないか: 影響は「`setDitherBitDepth` / `setNoiseShaperType` のたびに
約 2 秒後に**冗長な 2 本目の `requestRebuild`** が追加される」ことまでが確定。
§2.8 より 1 本目は即時 dispatch されるため**パラメータ適用自体は遅延しない**。

**追加調査（2026-10-01 実施）で 2 本目の実挙動まで確定した。**
- Timer の強制 dispatch（`Timer.cpp:1053`）は `RebuildTelemetryPolicy::MustExecute` を渡す。
  `requestRebuild:491-493` で `forceMustExecute=true` → `collapsePolicy = MustExecute`。
  merge は `Replaceable` のときだけ（`:212`）なので、**2 本目は merge による取りこぼしを受けない**
  （`:271` の `sameAsPendingWouldMerge && shouldApplyLatestWinsMerge` が false）。
  → 必ず full rebuild が走る。
- full rebuild は新規 DSPCore を publish する。dither / noiseShaperType のみの変更では
  `CrossfadeAuthority::evaluate` の判定分岐（`oversamplingFactor` 差 /
  `structuralHash` 差 / `irLoaded` 差 / 各 fade 時間フィールド）がいずれも成立しないため
  `needsCrossfade = false` → `DSPTransition::onPublishCompleted` の
  `else if (oldDSP != nullptr && oldDSP != newDSP)` 枝に入り、
  `crossfadeRuntime_.complete(); lifetime_.retire(oldDSP);`（= crossfade 無しの hard swap）。
- しかし 2 本目は**同一パラメータからの再構築**なので、swap 前後の出力は同一。
  音色の変化・クリック・不連続は生じない。

→ 確定した実害は **CPU 浪費（不要な full DSP rebuild + publish を各設定変更後に 1 回追加）** のみ。
wrong audio / state corruption / stale state / subsequent operation failure の
いずれにも該当しないため **Owner の条件 5 を満たさず defect としない**。
失われた `finalizeReady` fast path も、2000 ms timeout が代替するため
「以後の操作が失敗する」ことには当たらない。

### O-2 `OrchestratorForwardProgress::executedCount` が増やされない

- 宣言: `RuntimePublicationState.h:51`
- `onPublished/onRetired/onReclaimed/onSubmitted/onBuilt/onValidated/onRejected/onExecutorFailed`
  は存在するが **executed に対応する `onExecuted()` が無い**。
- `detectStuckStage()`（`:69-77`）の `validatedCount > executedCount → StuckStage::Executor`
  が、初回 validation 成功以後**恒 true** になる。
- 消費は `ISREvidenceExporter.cpp:212`（evidence JSON 出力）のみ。

なぜ defect としないか: 診断出力専用。`ISRHealthState`・admission・policy の
いずれの制御経路にも入っていない。

### O-3 publish 失敗分岐が `PublishStage::Published` を進捗リングに記録

- `RuntimePublicationOrchestrator.cpp:301-303`（`if (result != PublishResult::Success)` の中）
- build 失敗 2 経路（`:186` / `:254`）は「到達した最終段階」を記録しているのに対し、
  publish 失敗経路だけ**成功経路（`:327-329`）と同一の `Published`** を記録する。

なぜ defect としないか:
- 進捗リングの消費は `TelemetryRecorder.cpp:108 readLatest(...)` → evidence snapshot のみ。
- publication stall 検出（`RuntimeHealthMonitor::checkPublicationStall:331-374`）は
  `getPendingIntentCount()` / `hasDeferredRequest()` / `getPublicationBacklogCount()` と
  `isPublicationStalled()` を使い、**進捗リングの stage を使っていない**。
  → stall 検出が偽陽性に化けるroutes は無い。
- `stateOwner_.onPublished()`（`publishedCount`）は publish 成功時のみ呼ばれており、
  カウンタも偽勢わない。

### O-4 publish 成否に関係なく `DSPReady` が enqueue される

- `AudioEngine.Commit.cpp:878` `runtimeOrchestrator_->submitPublishRequest(req);`
  は **void 戻り値**（`RuntimePublicationOrchestrator.h` / `.cpp:361`）。
  したがって caller は decision を区別できない。
- `:880-890` で**無条件に** `LearningCommand::Type::DSPReady` を enqueue。
- handler（`AudioEngine.Learning.cpp:248-276`）は `Start` handler（`:134-136` が
  `dsp != nullptr && noiseShaperType == Adaptive` を検証する）と異なり、
  **ready 検証を一切行わない**。`WaitingForDSP → Running` と
  `adaptiveCaptureActiveRt = true`（`AudioEngine.h:4276` で
  `adaptiveCaptureQueue` が非 null になる）を無条件に成立させる。

なぜ defect としないか: `WaitingForDSP` への遷移点は
`Learning.cpp:144`（`Start` 時に active DSP が無い／noiseShaperType が非 Adaptive）と
`:238` のみ。`:144` の場合 publish 失敗時も active DSP が無いので
`buildAudioThreadProcessingState` に到達せず capture は発生しない。
`:238` は既に `WaitingForDSP` の状態を維持するだけで、実害の経路Opened。
`currentIRGeneration` は書かれるだけで**読まれない**（`src/` 内に reader 0）。
→  demonstrable な wrong audio / state corruption が見えない。

### O-5 Pending-Structural-Deployment 監視が dead

- `RuntimeHealthMonitor::checkPendingStructuralDeployment:1139-1176` は
  `m_requestedRebuildGenRef_` / `m_committedRebuildGenRef_` を要求するが、
  `setRequestedRebuildGenRef` / `setCommittedRebuildGenRef` の**呼び出しは 0 箇所**
  → `:1141` で常に早期 return。event 5004 は発行されない。
- 5004 は `AudioEngine.Timer.cpp` の health handler にも存在しない（1801/1819/1841/1856/1874/1963 のみ）。

### O-6 `ConvolverProcessor::rebuildPendingAfterLoad` に writer が無い

- 宣言: `ConvolverProcessor.h:1160`
- ヘッダコメント（`:1161-1163`）は「Audio Thread から store(true)、`process()` 先頭で
  exchange(false) して `rebuildPendingAfterLoad` に転送」と記載するが、
  **`src/` 内に RT からの writer も転送処理も存在しない**。
  実在する書込みは `LoadPipeline.cpp:885`（`callAsync` 失敗時の復元）のみ。
- 消費側 `ConvolverProcessor.Lifecycle.cpp:206-222` は
  `consumeAtomic(rebuildPendingAfterLoad)` で**先に消費**してから
  `!isLoading && !isRebuilding && irFile.existsAsFile()` を判定するため、
  仮に writer が存在しても false 時に要求が失われる構造。
- NUC ring overflow の検出（`Lifecycle.cpp:184-203`）は counter を reset するだけで
  recovery を要求しない。

なぜ defect としないか: 現状 writer が無いため要求は発生せず「失われる request」も存在しない。
「意図された機構が未配線」は **欠落機能** であり、Owner の条件 3
（現行 source に具体的な state transition がある）/ 条件 5（実害）を満たさない。
なお recovery を「rebuild で回復すべきか」の設計判断は Owner の
「設計論で defect 判定しない」線上にあり、本 STG の対象外。

### O-7 rebuild 世代ガードの実効

`RebuildDispatch.cpp:196-229` の pending-intent merge は
`shouldApplyLatestWinsMerge` で `elapsed <= latestWinsWindowTicks`
（Structural なら `getRebuildDebounceMs()`）に限定される。
O-1 で `!rebuildOutstanding` の reset が dead でも、
`pending` は `:220-228` で毎回上書きされ window 判定は機能するため、
merge 過剰による rebuild 取りこぼしは無い。

---

## 4. D1〜D18 との境界 / 非重複の確認

- D18（crossfade lifecycle）は再掘り返していない。本 STG は
  publication failure / fallback / user-visible state の領域のみを走査。
- O-1〜O-7 はすべて `lastCommittedRebuildGeneration` / telemetry /
  learning / convolver flag / health monitor の領域であり、
  D1〜D11 の production 変更対象
  （`EQProcessor.*`, `AudioEngine.StateIO.cpp`, `AudioEngine.Timer.cpp`,
  `AudioEngine.Mmcss.cpp`, `AudioEngine.h`, `SnapshotCoordinator.cpp`,
  `ISRDSPHandle.cpp`, `DSPTransition.h`, `ConvolverProcessor.Runtime.cpp`）
  とは別 theme。ただし D18 commit 後に O-1 の production 修正を行う場合は
  `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.Parameters.cpp` が対象となり、
  D1〜D11 の未 commit 変更と同一 theme（rebuild dispatch）でなければならぬため、
  commit 境界の扱いを Owner と合意してから着手する。

---

## 5. 使用ツール

- authority baseline 固定: `output_sourcecode_markdown.py --check`（SHA-256 / size /
  Generated / NEWER_SRC_COUNT / FRESH）
- ソース走査: WSL `rg`（rtk prefix）、`sed`、`read`
- 静的解析: cppcheck 2.22（`--project`, warning/performance/portability, c++17）、
  clang-tidy 23.1.1（`clang-analyzer-*,bugprone-*,performance-*,misc-*`）— D19 では
  production 変更がないため再実行不要（前工程 D18 で同一 TU を実行済み、hunk 指摘 0）
- production 変更: **なし**（NO-GO のため build / CTest / ASan / ConvoPeq.md 再生成は不要）

---

## 6. 未確定事項（保持）

1. ~~**O-1 の完全証明**~~ → **2026-10-01 解決済み**（O-1 の「なぜ defect としないか」に記載）。
   2 本目の rebuild は `MustExecute` で merge されずに必ず走り、crossfade は成立せず
   同一パラメータからの hard swap となるため、音色変化は生じない。実害は CPU 浪費のみ。
2. **O-4 の到達条件**: `LearningRuntimeState::WaitingForDSP` に active DSP が存在する経路が
   他に無いか（他スレッドからの state 遷移）を確認していない。現時点で確認した
   2 つの遷移点（`:144` / `:238`）はいずれも demonstrable な害を生じない。
3. **O-6 の設計意図**: ring overflow recovery を rebuild で行うべきかは設計判断。
   Owner の「設計論で defect 判定しない」線上にあり defect にはしない。
4. O-2 / O-3 / O-5 は診断専用。診断値の信頼性改善は production bug fix ではないため、
   Owner の指示に従い**横道に入れていない**。
5. **D19 で production 変更は無い**ため、build / CTest / ASan / `ConvoPeq.md` 再生成は
   実施していない（authority baseline は 19:25:35 の生成時点で FRESH 持续）。