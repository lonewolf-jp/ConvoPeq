# P1-5-IR-P0 — Protocol Reconciliation (read-only audit report)

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 0
- **基準 source**: `ConvoPeq.md` Generated 2026-09-22 06:20:11 と同等の現行 working tree(HEAD = 1e9e63e3・production/CMake/default/calibration/commit/harness 変更 0)
- **方法**: source-first の追跡(関数名・行番号は現行 working tree のもの)+ 既存 evidence(tmp/ ログ群・`tmp/p1_5_hr_step5a_evidence.json`)との突合。推測の統合は禁止し、確定できない点は UNKNOWN とする。

---

## 1. 現行 IR state machine(実コード裏付け)

```text
IR Load
  ↓
Prepare (LoaderThread / cache+RCU convert / progressive FFT upgrade)
  ↓
Finalize (NUC descriptor commit → engine/state swap → irFinalized=true)
  ↓
Rebuild (deferred 200ms or immediate structural intent → RT world 再構築)
  ↓
Validate (prepare 時 payload validation + runtime world dspReady)
  ↓
Publish (commit sequence 前進)
  ↓
Active Runtime (convolverRt() が新 IR で実行)
```

### 1.1 IR Load — 2 系統の入口が存在する(両方とも UI 側 `uiConvolverProcessor` に対する NonRT 操作)

| 系統 | 入口 | 呼び出し元 | 戻り値の意味 |
| --- | --- | --- | --- |
| A: LoaderThread 系 | `ConvolverProcessor::loadImpulseResponse(irFile, optimizeForRealTime)` — `src/convolver/ConvolverProcessor.LoadPipeline.cpp:17` | harness(`ensureTestIr` = `P1PolyphaseGainCharacterization.cpp:394`)・`ConvolverProcessor.Lifecycle.cpp:219`(自動 IR ロード)・テスト一式 | `true` = LoaderThread 起動受付(rebuild skip 時も `true`)。**読み込み完了は意味しない** |
| B: RCU cache 経路 | `AudioEngine::requestConvolverPreset(irFile)` = `uiConvolverProcessor.loadIR(irFile)` — `AudioEngine.Parameters.cpp:198` → `LoadPipeline.cpp:230` | `MainWindow.cpp:796`(--cli-ir・200ms 遅延)、UI ファイルロード | 生成 ID bump + cache/convert + `applyComputedIR` を Message Thread 上で完結 |

系統 A の開始時に行われること(`LoadPipeline.cpp:46-47`):
```cpp
convo::publishAtomic(isLoading, true, ...);
convo::publishAtomic(irFinalized, false, ...);   // ← 系統 A のみの reset point
```

### 1.2 Prepare

- 系統 A: `LoaderThread`(`src/convolver/ConvolverProcessor.LoaderThread.cpp`)がバックグラウンドでファイル読込・リサンプルを行い、**`[IR_TAIL_GEOM] gen=%llu loadedSr=%.0f loadedLen=%d targetLength=%d copySamples=%d`**(同ファイル :759)をログ出力。IR は target length = processingRate × targetSec へ加工される(irLen はロード構成依存 — 既存 doc §5 記録と一致)。
- 系統 B: `loadIR` 内で `CacheManager::computeKey(file, fft, sr, phase, partition)` → targetFFT(既定 4096)キャッシュ → 無ければ低解像度 512 FFT 変換 → `applyComputedIR` → `startProgressiveUpgrade`(512→4096 のバックグラウンド FFT 昇格、中間 publish は `suppressIntermediateMixedPhasePublish` で抑止 — `AudioEngine.UIEvents.cpp:55-71`)。

### 1.3 Finalize(= UI 側 ConvolverProcessor 上の engine/state 差し替えと irFinalized=true)

- 系統 A: `LoaderThread` → `finalizeNUCEngineOnMessageThread`(`LoadPipeline.cpp:635`)→ `StereoConvolver::init` 成功 → `applyNewState(..., async)` → `executePendingCommit`(`LoadPipeline.cpp:796`):
  - Phase 1: `updateIRState(loadedIR, sampleRate, 0, 0, knownBlockSize)`(non-rebuild のみ) + `currentIRScale` publish
  - Phase 3: `switchEngineOnMessageThread(newEngine)` — UI 側 active engine を排他的交換
  - Phase 4: `publishAtomic(irLength...)`・`publishAtomic(currentSampleRate...)` → **`publishAtomic(irFinalized, true)`**(:835)→ `refreshLatency()` → `isLoading/isRebuilding=false`
- 系統 B: `applyComputedIR`(`LoadPipeline.cpp:383`) — SR 一致検証・scaleFactor 適用・振幅 validation → `updateIRState` → `updateConvolverState(new ConvolverState(fftSize, generationId, sampleRate))` → **`publishAtomic(irFinalized, true)`**(:579)。
- **irFinalized=true は UI 側 processor の状態**であり、この時点では RT world はまだ旧構成である(§1.4)。

### 1.4 Rebuild 発火条件(`AudioEngine.UIEvents.cpp:55-190` convolverParamsChanged)

1. `suppressIntermediateMixedPhasePublish` が立っている間は中間 publish を抑制(Suppressed telemetry・MixedPhaseIntermediate)。
2. `needsStructuralRebuild` の判定条件:
   - `uiHasIr = uiConvolverProcessor.isIRLoaded()` と `committedHasIr = lastCommittedConvolverHasIr_` の不一致(:91)
   - または構造ハッシュ不一致: `uiConvolverProcessor.getStructuralHash()` ≠ `lastCommittedConvolverStructuralHash_`(:93-97)
3. hash dedup: 直前に発行した同一ハッシュは Structural rebuild を再発火しない(`lastIssuedConvolverStructuralHash_`、:108-130)。
4. **deferred 発火条件(:132-156)**: `needsStructuralRebuild && uiHasIr && !committedHasIr`(DSP が IR をまだ持たない=初回 IR)かつ `(now − getLastPreparedIRApplyTicks()) < 200ms` の場合 → `RebuildReason::DeferredStructural` を立て、`deferredStructuralRebuildDueTicks_ = lastPreparedIRApplyTicks + 200ms` に設定して当該通知は rebuild を発行しない。
   - **注意**: この遅延値は **200ms**(`ticksPerSecond / 5`)。--cli-rebuild の 500ms とは別物(§1.7)。
5. 即時発火: `submitRebuildIntent(RebuildKind::Structural)`(:162)+ `IRChanged` learning command + `pendingIRGeneration++`(:174-189)。

`AudioEngine.Timer::timerCallback`(`src/audioengine/AudioEngine.Timer.cpp:770-810`): `RebuildReason::DeferredStructural` 保持中に `nowTicks >= dueTicks` になった時点で、**`uiConvolverProcessor.isIRLoaded()` を再確認してから** `submitRebuildIntent(Structural, DeferredStructuralRebuildRequested)` を発行し、`pendingIRGeneration++`・`setIRChangeFlag()`・`IRChanged` command を投入。

### 1.6 Runtime への IR transfer と publish

- rebuild intent を受けた worker が runtime world を再構築する。RT world の convolver は `convolverRt()`(UI processor とは別インスタンス)であり、IR の実体は `transferIRStateFrom(source)`(`ConvolverProcessor.h:1269`、コメント: 「Must be called before rebuildAllIRsSynchronous() on this instance」)で UI → RT へコピーされ、`rebuildAllIRsSynchronous()` が RT 側 partition を組み直す。
- `RuntimeBuilder.cpp:240-252` は world 生成時に `dspProjection.irLoaded / irFinalized / structuralHash / oversamplingFactor / sampleRate / baseLatencySamples` を stamp する(sealedSnapshot があればそれを、無ければ `current->convolverRt()` の実測値)。
- world の publish は commit sequence を前進させる。**Active Runtime はこの world を受けて初めて新 IR で実行される**。したがって UI 側 `irFinalized=true` と「Active Runtime が新 IR を持つ」の間には **structural rebuild(+ publish)の分だけ必ず遅延**がある(初回 IR 且つ prepared-IR-apply 直後はさらに 200ms の deferral が挟まる)。バックグラウンドでの MKL partition 書き込みは `[L0_WRITE] gen ir=... / geom part=... irLen=...`(`MKLNonUniformConvolver.cpp:1068-1087`)として観測できる。

### 1.7 CLI 経路(--cli-ir / --cli-rebuild)

- `--cli-ir <path>`(`MainWindow.cpp:777-818`): **200ms 遅延**で `audioEngine.requestConvolverPreset(irFile)`(= RCU 系統 B)を発火し、その直後に `[CLI_IR] isIRLoaded/irLen` を記録。オプション `--cli-ir-reload-count`(--既定 0)・`--cli-ir-reload-interval-ms`(既定 300)で再ロード storm を追加。
- `--cli-rebuild`(`MainWindow.cpp:1029-1052`): **delayMs = 500** で `[CLI_REBUILD] isIRLoaded/irLen` をログした後 `audioEngine.requestStructuredRebuildIntent(RebuildKind::Structural)` を強制発行(= `AudioEngine.h:1486` → `submitRebuildIntent(kind, RequestRebuildKindEntry, Structural, Replaceable)`)。telemetry suppression を迂回する既存経路。
- `--cli-exit-ms` は IR/rebuild 使用時の最小値が 3000ms に底上げされる(:1059-1073)。

### 1.8 存在しない遷移(明記)

- **「Prepare完了」を表す独立 state は存在しない**(UI 可視用の `isLoading` と `lastPreparedIRApplyTicks` のみ。prepared 専用の state accessor は無い)。
- **RT 側への IR 転送完了を直接待つ accessor は存在しない**(UI 側 `irFinalized`・RT world の `dspProjection` フラグ・RT convolver の `getIRLength()>0` が間接指標のみ。BassBuzz WORK104-R2 コメントがこれを明言)。
- `irFinalized` の **generation token 付き読み取りは存在しない**(単一 atomic<bool>)。
- load 失敗時の自動 retry や IR 再ロードの自動連鎖は `rebuildPendingAfterLoad` のみ(executePendingCommit :841-850)で、IR ファイル自体の自動再試行は存在しない。

---

## 2. 既知 anomaly の再検証(過去記録 → 現行 source 照合)

出典: `doc/work113/p1_5_adoption_characterization_20260922.md` §5(:53-58)。

| anomaly | 過去記録 | 現行 source 上の対応 | 分類 |
| --- | --- | --- | --- |
| **1.48 dB mismatch** | 同一 IR g0・os=1・sc0 で amp=−20 → −14.5035 dB、amp=−6 → −13.0276 dB(線形域で 1.48 dB 差) | §2.1 参照。測定 row 間の差であり、IR レベル鎖の特定 2 点ではない。1-run-lag と同じ整定不足で説明可能(INFERRED) | **OBSERVED(値は記録済み)/ 機因は INFERRED・IR 内部チェーンは UNKNOWN** |
| **1-run lag** | 行 N の sc1 == 行 N+1 の sc0 | harness の待機が UI 側 `isIRFinalized` のみで、(a) deferred 200ms structural rebuild と (b) OS rate 変更再準備の完了を待たない(runCase の settle = waitBacklogZero + waitWorldPublished(直前 seq) + sleepPump(800))。さらに `[IR_TAIL_GEOM]` の通り IR はロード構成ごとに再加工(irLen=rate×1s)で cold 時は分単位(BassBuzz WORK104-R2 記載) | **OBSERVED(パターン)/ 発生点は INFERRED: measurement sampling 遅延が主因** |
| **isIRFinalized stale** | 前回ロード状態で true を返し得る | §3 の表の通り: **RCU 系統(loadIR→applyComputedIR)には load 開始時の irFinalized=false reset が存在しない**(`LoadPipeline.cpp:230-318` に reset 経路なし)。`loadImpulseResponse` 系統のみ :47 で reset | **OBSERVED(source 上の実在する非対称)** |

- `ir` listen 素材(Step 5-A)では同 anomaly は**再現していない**(ΔRMS=+7.4967/+7.4962 の純粋利得): ensureTestIr が 300 秒 budget の poll を行い、runCase の waitBacklogZero + waitWorldPublished + sleepPump(800) が整定を挟んだため → **NOT REPRODUCED(当該条件下)** と併記する。
- `[IR_TAIL_GEOM]`/`[L0_WRITE]` により **IR の実効スカラー利得はロード構成(SR/OS/target length)に依存**する(doc §5 既記録)→ 「テスト IR の nominal 0/+3/+6 dB は絶対値不成立」= **OBSERVED(ログ裏付けあり)**。1.48 dB をこの構成依存と直接同一視することはまだ evidence がない(UNKNOWN)。

---

## 3. 1-run-lag の定義の固定

5 候補の分離:

| 候補 | 現行 source 上の評価 | 分類 |
| --- | --- | --- |
| IR generation 遅延 | LoaderThread/convertFile/ProgressiveUpgrade 自体の所要(cold 時分単位)。遅延自体は存在するが「run 間 1 遅れ」を説明するのは測定タイミング側 | 存在する(INFERRED・副次) |
| publish sequence 遅延 | structural rebuild → world publish の commit seq 前進は、param 変更 world と IR world で **2 回以上**発生し得る。harness は「seqBefore 以降の任意の world publish 1 回」を待つだけで IR world を特異的に待てない | **INFERRED: 認識不足ではなく待機対象の誤指定に相当** |
| **measurement sampling 遅延** | runCase の settle(`waitBacklogZero`+`waitWorldPublished`(直前 seq)+`sleepPump(800)`)は **IR-structural world の publish を保証しない**(ensureTestIr は UI 側 isIRFinalized のみ待つ)。遅延した IR world は「次の run の sc0」で初めて反映される | **OBSERVED に最も整合する発生点(INFERRED・主因候補)** |
| `isIRFinalized` state visibility 遅延 | RCU 系統では reset が無く前回状態の true が残る(§3)→ 待機判定そのものが早期成立する | **OBSERVED(source 上の実在条件)・副次** |
| CLI timing の問題 | --cli-ir は 200ms 遅延後に requestConvolverPreset、--cli-rebuild は 500ms 強制 intent。deferred 200ms(UIEvents)と別系列 | 存在するが本 anomaly の主因とは特定しない(UNKNOWN) |

**結論(単一の確定)**: 1-run-lag は **測定プロトコルの整定不足(3 番目)** を主因候補とし、RCU 系統の visibility 欠落(4 番目)がこれを増幅する 2 因構成とする。単一原因への還元はしない(推測統合禁止)。

---

## 4. 1.48 dB mismatch の比較対象の定義

点列の区分(ユーザー指示の 5 層):

| 層 | 現行 source 上の観測点 | ログ/Accessor |
| --- | --- | --- |
| input IR level | WAV 自体 — **未ログ** | UNKNOWN(未計測) |
| prepared IR level | `applyComputedIR` の payload(scaleFactor適用後)・`[IR_TAIL_GEOM]`/`[L0_WRITE]` のみ部分観測 | **UNKNOWN(未計測)** |
| RuntimeWorld に入った IR level | `transferIRStateFrom` + `rebuildAllIRsSynchronous` 経路 — **未計測** | UNKNOWN |
| actual convolver output level | 実行時の DSP 出力 — 直接の利得ログは無い | UNKNOWN(間接のみ) |
| measurement output level | `runCase` の DFT `gainDb`(dftDb − ampDb)— **p1_5 doc §5 の値はこれ** | **OBSERVED** |

**確定**: 「1.48 dB」= **measurement output level の 2 測定行間差**(同一 IR g0・os=1・sc0 で amp=−20 と amp=−6 の行間差)。IR 内部チェーン(input→prepared→RuntimeWorld)は**未計測であり UNKNOWN**。したがって 1.48 dB を「IR 自体の利得誤差」と特定することは**現時点では不能**(測定不成立の記録どおり invalid)。

---

## 5. `isIRFinalized` stale の実体(writer/reader 表)

| 項目 | 実装位置 | 説明 |
| --- | --- | --- |
| **state 本体** | `irFinalized` atomic\<bool\>(単一・generation token 無し) | `ConvoProcessor.h:444`(`consumeAtomic` acquire) |
| **writer (false)** | `loadImpulseResponse`(`LoadPipeline.cpp:47`) | 系統 A の load 開始時のみ |
| **writer (true)** | `applyComputedIR`(:579) / `executePendingCommit`(:835) | engine/state swap 完了後・Message Thread |
| **writer (rollback)** | `handleLoadError`(:593) — `irFinalized = isIRLoaded()` | ロード失敗時: 前回 IR が存在すれば true のまま(stale-true の第二発生条件) |
| **reset point** | **系統 A(loadImpulseResponse)のみ**:47。**系統 B(loadIR→applyComputedIR)には reset が存在しない** | 前回ロードの true が convert/cache 区間中も残る = stale の実体 |
| **reader (NonRT)** | `isIRFinalized()` — UI/timer/tests/MixedPhaseOptimizationComponent:65/BassBuzz settle | UI 状態としての finalized |
| **reader (Runtime 側)** | `convolverRt().isIRFinalized()`(`RuntimeBuilder.cpp:248` で stamp) | RT world は別インスタンスの atomic を持ち、**UI 側 true とは別の状態** |
| **publish point** | engine swap(executePendingCommit)/state publish(updateConvolverState)直後。`lastPreparedIRApplyTicks`(:584)が整定判定の基準時刻 | |
| **lifetime** | processor の生存期間で単一フラグ(生成 token 無し)。generation は `convolverStateGeneration` が管理するが irFinalized とは独立 | |
| **thread** | writer: Message Thread(applyComputedIR/executePendingCommit)+LoaderThread(経由の queue finalize)。reader: 任意(UI/timer/harness pump/RuntimeBuilder) | |

**stale-true の成立条件(source から一意)**: (a) RCU 系統での再ロード中に前回の true が残る、(b) `handleLoadError` の rollback で前回 IR の true が残る、(c) UI 側 true と RT world の実際の IR 適用の乖離(構造 rebuild 未完了の間)。3 条件は独立に存在する。

---

## 6. PASS 判定

```text
[x] IR lifecycle を現行 source から一意に説明できる(§1 — 2系統入口・1つの finalize commit・deferred 200ms structural rebuild・RT world 経由の transfer)
[x] 1.48 dB mismatch の比較対象が定義できる(measurement output level の amp=−20 行 vs amp=−6 行。IR 内部チェーンは UNKNOWN)
[x] 1-run-lag の発生点が定義できる(measurement sampling 整定不足=主因候補 INFERRED・RCU reset 欠落=実在条件 OBSERVED)
[x] isIRFinalized stale の writer/reader/lifetime が追跡できる(§5 の表)
[x] 未確定部分が UNKNOWN として明示されている(IR 内部チェーン・1.48 dB の機因帰属・coloration)
```

**P1-5-IR-P0 = PASS**。本レポート作成にあたり production source / CMake / default / calibration / commit / harness の変更は行っていない(読み取り監査のみ)。1.48 dB mismatch を「補正すべき gain error」と仮定した補正・fix は一切実施していない。

次工程はユーザー指示の順序(`P1-5-IR-P1 measurement protocol` 以降)で開始する。
