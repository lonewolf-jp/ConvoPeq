# A-6 / A-12 事前調査メモ (2026-05-11)

## 目的

- A-6: `currentDSP` / `fadingOutDSP` / `queuedOldDSP` / `retireDSP` の三重寿命管理リスクを縮小する。
- A-12: `DSPCore` 内の mutable state を段階的に分離し、クロスフェード時の状態汚染リスクを下げる。

## 現状観測 (コード根拠)

### 1) 所有権・寿命管理

- `DSPCore` 本体は `AudioEngine` 内部に定義され、mutable 処理器 (`convolver`, `eq`, `dcBlocker`, `oversampling`, `adaptiveNoiseShaper`) を内包。
  - 参照: `src/audioengine/AudioEngine.h` (`struct DSPCore`)
- ランタイムポインタは以下で分散管理。
  - `currentDSP` (RT読取)
  - `activeDSP` (Message Thread 所有)
  - `fadingOutDSP` (クロスフェード旧系)
  - `queuedOldDSP` (フェード待機)
  - 参照: `src/audioengine/AudioEngine.h`
- 解放は `retireDSP()` 経由で `retireObject` に委譲。
  - 参照: `src/audioengine/AudioEngine.h`
- `commitNewDSP()` で `currentDSP.store(newDSP)` と `activeDSP = newDSP` の更新後、分岐ごとに `fadingOutDSP.exchange` / `queuedOldDSP.exchange` / `retireDSP` が実行される。
  - 参照: `src/audioengine/AudioEngine.Commit.cpp`

### 2) mutable state 汚染

- `DSPCore::prepare()` で `convolver.prepareToPlay`, `oversampling.prepare`, `dcBlocker.init` など mutable state 初期化。
  - 参照: `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`
- Audio Thread 側 `process` 経路で mutable state が毎ブロック更新される。
  - `oversampling.processUp/processDown`
  - `convolver.process`
  - `eq.process`
  - `dcBlocker.processStereo`, `osDCBlocker.process`
  - `adaptiveNoiseShaper.applyMatchedCoefficients`
  - 参照: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`, `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`

## A-6 観点の主リスク

- ownership の事実が単一構造体ではなく複数ポインタに分散している。
- クロスフェード分岐で `fadingOutDSP` と `queuedOldDSP` の入替え順序に依存するため、将来改修時に二重 retire / 取り残しが混入しやすい。
- `RuntimePublishState` は可観測性を改善しているが、寿命管理の唯一の真実にはなっていない。

## A-12 観点の主リスク

- `DSPCore` が immutable 設定と mutable 処理状態を同居させている。
- `DSPCore` をクロスフェード対象として扱う現行設計では、切替時に mutable 履歴(Convolver, EQ, DC blocker, oversampling)の境界が曖昧。
- warmup 追加で軽減しているが、根本的な分離には未到達。

## 最小侵襲の実装ステップ案

### Phase 1 (低リスク)

1. `RuntimeHandle` 的な薄い管理構造を導入し、`current/fading/queued` の更新APIを1箇所に集約。
2. `commitNewDSP` で直接 `exchange/store` せず、集約API経由に限定。
3. retire 対象決定を「旧ハンドル単位」にまとめ、`retireDSP` 呼出し箇所を削減。

### Phase 2 (中リスク)

1. `DSPCore` から mutable 実行状態を `PerAudioThreadDSPState` に分離する足場を追加。
2. まず `dcBlocker` / `osDCBlocker` / ramp系の軽量状態を分離し、`convolver/eq` は後段。
3. クロスフェードに参加するのは immutable graph + state bundle の組にする。

### Phase 3 (高リスク)

1. `EngineRuntime` 単一 atomic 公開へ移行。
2. `currentDSP/fadingOutDSP/queuedOldDSP` の直接運用を廃止。

## 次の着手候補 (実装順)

1. `AudioEngine.Commit.cpp` のポインタ更新・retire を薄い helper へ集約 (Phase 1-1,1-2)。
2. 集約後に `retireDSP` 呼出し箇所を棚卸しし、二重 retire 防止アサーションを追加。
3. その後、mutable state 分離の第一段として `dcBlocker` 系の state 抽出に入る。

## 進捗スナップショット (2026-05-11 追記)

### A-6 Phase 1 で実装済み

1. `commitNewDSP` の主要寿命遷移を helper 化し、分岐の更新規約を集約。
2. self-retire ガードと runtime UUID 診断ログを導入。
3. `active/fading/queued` 重複監視を helper 単位で導入。
4. 関数境界監視を `commitNewDSP` の `entry / afterPublish / beforeSendChangeMessage` に追加。
5. 非RT経路として Timer queued 昇格、`releaseResources`、デストラクタにも重複監視を追加。
6. RT経路（fade 完了 retire）にはログ無しの `validateDistinctRuntimeSlotsRT` を追加し、Audio Thread 制約を維持。

### A-6 Phase 1 の残タスク

1. 監視追加済み経路を基準に、`retireDSP` 呼出し面を再棚卸しして「呼出し起点 -> retire 単位」の対応表を固定する。
2. helper の責務境界を維持したまま、次段 (`EngineRuntime` 単一 publish) へ移行する最小差分案を整理する。

### A-6 追加改善 (2026-05-11)

1. `timerCallback` の queued old DSP 昇格（`queuedOldDSP -> fadingOutDSP`）で `RuntimePublishState` の再公開を追加。
2. これにより、非RT昇格時の publish スナップショットと atomic 状態の乖離ウィンドウを縮小した。
3. `prepareToPlay` で `dspCrossfadePending=false` を設定した直後に `RuntimePublishState` を再公開し、再初期化時の pending フラグ差分を publish 側へ同期するようにした。

### A-6 retire 呼び出し対応表 (2026-05-11)

1. 起点: `prepareCommit`
1. 条件: shutdown 中または deferred commit queue 追加前に shutdown 検知
1. retire 単位: `newDSP`

1. 起点: `executeCommit`
1. 条件: queue 取り出し後に shutdown 検知
1. retire 単位: `staging.newDSP`

1. 起点: `commitNewDSP`
1. 条件: generation 不一致 / 非finalized publish拒否 / warmup失敗
1. retire 単位: `newDSP`

1. 起点: `replaceQueuedOldDSPAndRetirePrevious`
1. 条件: `queuedOldDSP.exchange` で旧値あり（同一ポインタを除く）
1. retire 単位: `prev (旧 queued)`

1. 起点: `replaceFadingOutDSPAndRetirePrevious`
1. 条件: `fadingOutDSP.exchange` で旧値あり（同一ポインタを除く）
1. retire 単位: `prev (旧 fading)`

1. 起点: `retireRuntimeImmediately`
1. 条件: active/current/published-current と不一致の runtime 即時退役
1. retire 単位: `dsp`

1. 起点: `finalizeCrossfadeMixPath` / `cleanupCrossfadeDirectPath`
1. 条件: fade 完了で `fadingOutDSP.exchange(nullptr)` が成功
1. retire 単位: `done (完了済み fading)`

1. 起点: `requestRebuild`
1. 条件: pending task 置換で孤立した旧 `currentToRelease`
1. retire 単位: `currentToRelease`

1. 起点: `rebuildThreadLoop::DSPGuard`
1. 条件: commit へ受け渡されずスコープ終了
1. retire 単位: `dspGuard.ptr`

1. 起点: `releaseResources`
1. 条件: active/fading/queued/pending task/deferred commit の一括クリア
1. retire 単位: `activeToRelease`, `fadingToRelease`, `queuedToRelease`, `pendingCurrentToRelease`, `staging.newDSP`, `staging.oldDSP`

1. 起点: `~AudioEngine`
1. 条件: release 未実行異常系を含む最終クリア
1. retire 単位: `activeToRelease`, `fadingToRelease`, `queuedToRelease`, `pendingTask.currentDSP`, `staging.newDSP`, `staging.oldDSP`

### A-6 Phase 2 最小差分案 (`EngineRuntime` 単一 publish)

1. 目的: audio thread が参照する可変状態を `currentDSP/fadingOutDSP/queuedOldDSP + crossfade flags` の分散 atomic から、単一 `EngineRuntime` スナップショットへ集約する。
1. 方針: 既存 `RuntimePublishState` を置換せず、段階的に `EngineRuntime` を追加して読取優先度を切り替える。

#### ステップ P2-1 (型追加)

1. `EngineRuntime` 構造体を追加する。
1. 必須フィールド: `current`, `fading`, `queuedOld`, `transition`, `latencyDelayOld/New`, `dspCrossfadePending`, `dspCrossfadeUseDryAsOld`, `dspCrossfadeStartDelayBlocks`, `queuedFadeTimeSec`, `queuedNextFadeTimeSec`, `revision`。
1. 配置候補: `src/audioengine/RuntimeTransition.h`（`RuntimePublishState` 近傍）。

#### ステップ P2-2 (公開 API 追加)

1. `std::atomic<EngineRuntime*> engineRuntimeState` を `AudioEngine` に追加する。
1. `publishEngineRuntimeState(const EngineRuntime&)` と `getEngineRuntimeState()` を追加する。
1. 退役は既存 EBR (`convo::retireObject`) を流用し、`RuntimePublishState` と同様の publish/reclaim パターンを使う。

#### ステップ P2-3 (書込側の二重更新)

1. `commitNewDSP` の publish 点で `RuntimePublishState` と `EngineRuntime` を同時更新する。
1. `timerCallback` の queued 昇格点、`prepareToPlay` の crossfade reset 点、`releaseResources` の clear 点で `EngineRuntime` 再公開を追加する。
1. この段階では既存 atomic (`fadingOutDSP`, `dspCrossfadePending` など) を残し、挙動差分を出さない。

#### ステップ P2-4 (読取側の切替)

1. Audio thread の読取優先度を `EngineRuntime` 優先へ変更する。
1. 対象: `getNextAudioBlock`, `processBlockDouble`, `processWithSnapshot` の current/fading/pending/useDryAsOld 取得。
1. フォールバックとして既存 atomic 読取を残し、`EngineRuntime == nullptr` 時のみ使用する。

#### ステップ P2-5 (旧 atomic の縮退)

1. 非RT診断用途を除き `fadingOutDSP/queuedOldDSP/dspCrossfade*` の直接参照を削減する。
1. `validateDistinctRuntimeSlotsRT` は `EngineRuntime` スロット比較版へ移行する。

#### Phase 2 DoD

1. Audio thread の crossfade 判定は `EngineRuntime` 由来で動作し、既存 atomic はフォールバック扱いである。
1. `commit/timer/prepare/release` の主要更新点で `EngineRuntime` が再公開される。
1. `get_errors` がグリーンで Debug ビルド成功。

#### Phase 2 実施結果 (P2-1 / P2-2)

1. `RuntimeTransition.h` に `EngineRuntime` 構造体を追加した（`current/fading/queued/transition/latency/crossfade flags/revision`）。
2. `AudioEngine` に `engineRuntimeState` / `engineRuntimeRevision` atomic と `getEngineRuntimeState()` を追加した。
3. `publishEngineRuntimeState()` を追加し、既存 `publishRuntimePublishState()` から dual-write で `EngineRuntime` も再公開するようにした。
4. これにより、読取側切替（P2-4）前でも write 側の snapshot 生成基盤を先行整備した。

#### Phase 2 実施結果 (P2-4 first-cut, 再整理)

1. 読取側の crossfade 判定で `EngineRuntime` 優先、`RuntimePublishState` フォールバックを適用した。
2. 対象は `AudioBlock` / `BlockDouble` / `commitNewDSP` の dedup 判定 / `timerCallback` の queued 昇格ゲート。
3. `prepareToPlay` の placeholder 判定も `EngineRuntime` 優先に切替し、publish世代の整合性を合わせた。
4. timer/transition の診断ログ読取（revision/UUID 系）も `EngineRuntime` 優先へ切替し、監視系の読取経路を処理系と揃えた。

#### Phase 2 実施結果 (P2-5 audit, 再整理)

1. `audioengine/*.cpp` の `runtimePublish->...` 直接参照を棚卸しし、読取経路をヘルパー経由に統一した。
2. `prepareToPlay` と `timer` は `EngineRuntime` 優先ヘルパーへ置換済み。
3. 現在の `runtimePublish->...` 参照は `AudioEngine.h` 内のフォールバック層（互換目的）に限定される。

### A-6 Phase 2 の移行境界・完了条件

- `EngineRuntime-first resolver (`engineRuntime*` 系) は EngineRuntime のみ参照し、cpp 側はこれのみ使うのが最終目標。
- fallback/compatibility resolver (`runtime*` 系) は旧APIで、EngineRuntime未移行経路や互換維持のために残す。cpp 側からの直接利用がゼロになった時点で削除可能。
- AudioEngine.h の該当箇所に詳細コメントを追加し、移行境界・完了条件を明記。
- すべての cpp 主要経路（AudioEngine.Commit.cpp, AudioEngine.Processing.AudioBlock.cpp, AudioEngine.Processing.BlockDouble.cpp, AudioEngine.Timer.cpp, AudioEngine.Processing.PrepareToPlay.cpp）でエラーゼロを確認。

### DoD（Definition of Done）

- EngineRuntime-first resolver のみで cpp 側の全読取が完結し、fallback層が不要になった時点で Phase 2 完了。
- 以降は fallback/compatibility resolver を削除し、EngineRuntime 単一 publish/read 体制へ移行可能。
- 本ドキュメントおよび AudioEngine.h のコメントで移行境界・完了条件が明示されていること。

### 備考

- 本節は A-6 Phase 2 の「EngineRuntime移行境界・完了条件」明記のための追記。
- 以降の設計・実装・レビュー時は本節および AudioEngine.h コメントを参照し、移行状況・残件を即時把握できるようにする。

### A-6 Phase 2 の実施結果

1. `RuntimeTransition.h` に `EngineRuntime` 構造体を追加した（`current/fading/queued/transition/latency/crossfade flags/revision`）。
2. `AudioEngine` に `engineRuntimeState` / `engineRuntimeRevision` atomic と `getEngineRuntimeState()` を追加した。
3. `publishEngineRuntimeState()` を追加し、既存 `publishRuntimePublishState()` から dual-write で `EngineRuntime` も再公開するようにした。
4. これにより、読取側切替（P2-4）前でも write 側の snapshot 生成基盤を先行整備した。
5. `get_errors` は対象ファイルでエラーなし、Debug ビルド成功を確認した。

#### Phase 2 実施結果 (P2-4 first-cut)

1. 読取側の crossfade 判定で `EngineRuntime` 優先、`RuntimePublishState` フォールバックを適用した。
2. 対象は `AudioBlock` / `BlockDouble` / `commitNewDSP` の dedup 判定 / `timerCallback` の queued 昇格ゲート。
3. `prepareToPlay` の placeholder 判定も `EngineRuntime` 優先に切替し、publish世代の整合性を合わせた。
4. timer/transition の診断ログ読取（revision/UUID 系）も `EngineRuntime` 優先へ切替し、監視系の読取経路を処理系と揃えた。

#### Phase 2 実施結果 (P2-5 audit)

1. `audioengine/*.cpp` の `runtimePublish->...` 直接参照を棚卸しし、読取経路をヘルパー経由に統一した。
2. `prepareToPlay` と `timer` は `EngineRuntime` 優先ヘルパーへ置換済み。
3. 現在の `runtimePublish->...` 参照は `AudioEngine.h` 内のフォールバック層（互換目的）に限定される。

### A-12 着手境界

1. A-6 Phase 1 は「監視強化の区切り」まで到達。
2. 次の実装開始点は `dcBlocker` / `osDCBlocker` の mutable state 抽出。

## A-12 第一段の実装メモ (2026-05-11 追記)

1. `DSPCore` に `DCBlockerRuntimeState` sidecar を導入し、`dcBlocker` / `inputDCBlocker` / `osDCBlocker` の mutable state を `DSPCore` 主要メンバから分離。
2. `prepare/reset` は sidecar 初期化 API (`init/reset`) 経由へ移行。
3. `processInput/processOutput/processDouble` を含む I/O・OS経路の参照を sidecar 経由へ置換。
4. Debug ビルドでリンク完了を確認。

## A-12 第二段の実装メモ (2026-05-11 追記)

1. `DSPCore` に `RampRuntimeState` sidecar を導入し、`fadeInSamplesLeft` / `bypassFadeGainDouble` / `bypassedDouble` を分離。
2. `prepare/reset` と `processFloat/processDouble` の ramp 参照を sidecar 経由へ置換。
3. rebuild 経路の fade-in 初期化を `newDSP->ramps().fadeInSamplesLeft` へ移行。
4. Debug ビルドでリンク完了を確認。

## A-12 第三段の実装メモ (2026-05-11 追記)

1. `DSPCore` に `HistoryRuntimeState` sidecar を導入し、`fixedLatencyBufferSize` / `fixedLatencyWritePos` / `fixedLatencySamples` / `softClipPrevSample` を分離。
2. `setFixedLatencySamples` / `reset` / `applyFixedLatencyDelay` の履歴参照を sidecar 経由へ置換。
3. float/double の soft-clip 履歴更新を `history.softClipPrevSample` 経由へ置換。
4. Debug ビルドでリンク完了を確認。

## A-12 第四段の実装メモ (2026-05-11 追記)

1. `HistoryRuntimeState` に固定レイテンシ用バッファ実体（`fixedLatencyBufferL/R`）を移管し、所有境界を sidecar へ集約。
2. `setFixedLatencySamples` / `reset` / `applyFixedLatencyDelay` のバッファ参照を `history.fixedLatencyBufferL/R` へ置換。
3. Debug ビルドでリンク完了を確認。

## A-12 第五段の実装メモ (2026-05-11 追記)

1. sidecar の初期化責務を型メソッドへ集約。
2. `RampRuntimeState::prepare/resetForRuntime` を追加し、prepare/reset 側の初期化ロジックを移管。
3. `HistoryRuntimeState::configureFixedLatencySamples/resetForRuntime/clearSoftClipHistory` を追加し、固定レイテンシと履歴クリア責務を移管。
4. `DSPCoreLifecycle` 側は sidecar API 呼び出し中心へ簡素化。
5. Debug ビルドでリンク完了を確認。

### A-12 次ステップ

1. sidecar 化済み state の初期化責務（prepare/reset/rebuild）を型単位で明文化し、将来の再混在を防ぐ。
2. `convolver/eq` の mutable state 分離は別フェーズで実施（大規模変更扱い）。

更新:

1. sidecar 初期化責務の型メソッド集約まで完了。
2. 次は `rebuild` と `prepare/reset` の責務境界をチェックリスト化して、A-12 の軽量フェーズ完了条件を確定する。

## A-12 軽量フェーズ責務境界チェックリスト (2026-05-11 追記)

### 1) prepare の責務

1. sidecar の `prepare` 相当初期化は `DSPCore` 側で直接 field 代入しない。
2. `RampRuntimeState` は `prepare(sampleRate)` で初期化する。
3. `HistoryRuntimeState` は `configureFixedLatencySamples(...)` で固定レイテンシ初期化する。
4. `DCBlockerRuntimeState` は `init(sampleRate, processingRate)` で初期化する。

### 2) reset の責務

1. sidecar のランタイムクリアは `resetForRuntime()` 系メソッドへ集約する。
2. `DSPCore::reset()` では sidecar の内部 field を直接 clear/reset しない。
3. soft-clip 履歴クリアは `HistoryRuntimeState` 側責務 (`clearSoftClipHistory`) とする。

### 3) rebuild の責務

1. rebuild 経路で mutable な初期値を書き込む場合は sidecar accessor (`ramps()/histories()/dcBlockers()`) 経由に限定する。
2. `newDSP` への初期化代入で `DSPCore` 直下 field を新規追加しない。
3. fade-in 開始値は `newDSP->ramps().fadeInSamplesLeft` 経由で設定する。

### 4) 完了判定 (軽量フェーズ)

1. `dc/ramp/history` の mutable state が sidecar 経由でのみ更新されること。
2. `prepare/reset/rebuild` のいずれでも sidecar 内部 field への直アクセス追加がないこと。
3. Debug ビルドと `get_errors` がグリーンであること。

### 5) 次フェーズへの受け渡し

1. 本チェックリストを満たした時点を A-12 軽量フェーズ完了とする。
2. 次フェーズは `convolver/eq` mutable state 分離を大規模変更として別タスク化する。

## A-12 次フェーズ (convolver/eq 分離) タスク分割案

### Task C1: 依存境界の可視化

1. `DSPCore` から `convolver` / `eq` へ渡している mutable 参照一覧を抽出。
2. `prepare/reset/process` ごとに read/write 境界を表に整理。
3. DoD: 境界表に「RT更新」「非RT更新」「再構築時更新」の区分がある。

### Task C1 実施結果 (2026-05-11)

1. RT更新 / convolver
1. 代表経路: `DSPCore::process` float/double
1. 主な呼び出し: `convolver.process(...)`
1. 備考: オーディオ処理本体。`ProcessingOrder` 分岐に依存。

1. RT更新 / eq
1. 代表経路: `DSPCore::process` float/double
1. 主な呼び出し: `eq.setBypass(...)`, `eq.process(...)`
1. 備考: snapshot/fallback EQ params 経由の呼び出しを含む。

1. RT更新 / convolver
1. 代表経路: `RuntimeBuilder::warmup`
1. 主な呼び出し: `runtime.convolver.process(...)`
1. 備考: 無音ウォームアップ処理。commit 前の準RT系。

1. 非RT更新 / convolver
1. 代表経路: `DSPCore::prepare/reset`
1. 主な呼び出し: `convolver.setRcuProvider(...)`, `convolver.prepareToPlay(...)`, `convolver.reset()`
1. 備考: lifecycle 起点。

1. 非RT更新 / eq
1. 代表経路: `DSPCore::prepare/reset`
1. 主な呼び出し: `eq.prepareToPlay(...)`, `eq.reset()`
1. 備考: lifecycle 起点。

1. 非RT更新 / convolver
1. 代表経路: `AudioEngine::timerCallback`
1. 主な呼び出し: `dsp->convolver.cleanup()`
1. 備考: Message Thread の定期 cleanup。

1. 非RT更新 / eq
1. 代表経路: `AudioEngine::timerCallback`
1. 主な呼び出し: `dsp->eq.cleanup()`
1. 備考: Message Thread の定期 cleanup。

1. 再構築時更新 / convolver
1. 代表経路: `AudioEngine::rebuildThreadLoop`
1. 主な呼び出し: `shareConvolutionEngineFrom(...)`, `rebuildAllIRsSynchronous(...)`, `refreshLatency()`
1. 備考: rebuild スレッドでの mutable 更新。

1. 再構築時更新 / convolver
1. 代表経路: `RuntimeBuilder::applyBuildSnapshot/validate/warmup`
1. 主な呼び出し: `applyBuildSnapshot(...)`, `isIRLoaded/isIRFinalized` 参照, `process(...)`
1. 備考: 新 runtime 構築中の更新/検証。

1. 再構築時更新 / eq
1. 代表経路: `AudioEngine::rebuildThreadLoop`
1. 主な呼び出し: 直接 mutable 更新なし
1. 備考: 現状は `prepare` 後に RT パスで利用される設計。

#### C1 参照根拠

1. RT更新 (convolver/eq): `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`, `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
1. 非RT更新 (prepare/reset/timer): `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`, `src/audioengine/AudioEngine.Timer.cpp`
1. 再構築時更新: `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/RuntimeBuilder.cpp`

#### C1 DoD 判定

1. RT更新 / 非RT更新 / 再構築時更新の3区分を満たした。
2. 代表経路と主呼び出しを記載した。
3. 次タスク C2/C3 で必要な分離対象（convolver は更新点多数、eq は主に RT 側）を特定した。

### Task C2: Convolver state の分離足場

1. `ConvolverRuntimeState` の器だけ先行追加（実ロジック移行は最小）。
2. `DSPCore` からの直接 field 参照を accessor 経由へ置換。
3. DoD: 既存挙動を変えずにコンパイル・Debugビルド成功。

#### C2 実施結果 (first-cut)

1. `DSPCore` に `ConvolverRuntimeState` を追加し、`convolverRt()` accessor を導入した。
2. `DSPCoreLifecycle` / `DSPCoreFloat` / `DSPCoreDouble` の `convolver` 直参照を `convolverRt()` 経由へ置換した。
3. `get_errors` は対象ファイルでエラーなし、Debug ビルド成功を確認した。
4. 段階2として `AudioEngine.Commit.cpp` / `AudioEngine.RebuildDispatch.cpp` / `RuntimeBuilder.cpp` の `newDSP/activeDSP/runtime` 経路を `convolverRt()` 経由へ置換した。
5. `AudioEngine.UIEvents.cpp` / `AudioEngine.Timer.cpp` / `AudioEngine.Processing.Latency.cpp` の補助経路も `convolverRt()` 経由へ統一した。
6. C2 完了: 主要経路（DSPCore処理・Lifecycle・Commit・RebuildDispatch・RuntimeBuilder）と補助経路（UIEvents・Timer・Latency）で `convolver` 直参照の accessor 化を完了。

### Task C3: EQ state の分離足場

1. `EQRuntimeState` の器を追加。
2. EQ 関連の mutable 参照を accessor 経由へ置換。
3. DoD: 既存挙動を変えずにコンパイル・Debugビルド成功。

#### C3 実施結果

1. `DSPCore` に `EQRuntimeState` を追加し、`eqRt()` accessor を導入した。
2. `DSPCoreLifecycle` / `DSPCoreFloat` / `DSPCoreDouble` の EQ 参照を `eqRt()` 経由へ置換した。
3. 補助経路として `AudioEngine.Timer.cpp` の `dsp->eq.cleanup()` も `dsp->eqRt().cleanup()` へ置換した。
4. `src/audioengine/**` で `eq` 直参照を再検索し、対象パターンの残件なしを確認した。
5. `get_errors` は対象ファイルでエラーなし、Debug ビルド成功を確認した。

### Task C4: 初期化責務の集約

1. C2/C3 で追加した sidecar の `prepare/reset` 系 API へ初期化責務を集約。
2. `DSPCoreLifecycle` のロジックを API 呼び出し中心へ簡素化。
3. DoD: 直接 field 初期化が再混在していない。

#### C4 実施結果

1. `ConvolverRuntimeState` に `prepare/resetForRuntime/cleanupForRuntime` を追加し、convolver 初期化・リセット・cleanup の責務を sidecar API 化した。
2. `EQRuntimeState` に `prepare/resetForRuntime/cleanupForRuntime` を追加し、EQ 初期化・リセット・cleanup の責務を sidecar API 化した。
3. `DSPCoreLifecycle` の convolver/EQ 初期化・リセットを sidecar API 呼び出しへ置換し、直接メンバ呼び出しを整理した。
4. `AudioEngine.Timer` の定期 cleanup も sidecar API 経由へ統一した。
5. `get_errors` は対象ファイルでエラーなし、Debug ビルド成功を確認した。

### Task C5: 移行完了判定

1. `prepare/reset/rebuild` 経路で convolver/eq mutable state が sidecar 経由でのみ更新される。
2. `get_errors` がグリーンで Debug ビルド成功。
3. 影響範囲を `doc/bug.md` と本ファイルへ反映。

#### C5 実施結果

1. `prepare/reset/rebuild` 対象として `DSPCoreLifecycle` / `RebuildDispatch` / `RuntimeBuilder` / `PrepareToPlay` を再確認し、convolver/eq mutable 参照は sidecar 経由のみであることを確認した。
2. `PrepareToPlay` の placeholder 初期化に残っていた `placeholderDSP->convolver` 直参照2箇所を `placeholderDSP->convolverRt()` へ置換し、残件を解消した。
3. 対象ファイルの `get_errors` はグリーン、Debug ビルド成功を再確認した。
4. 影響範囲として `doc/bug.md` と本ファイルへ反映した。

更新:

1. ramp 系の sidecar 化は完了。
2. 次は軽量履歴状態（例: `softClipPrevSample`, 固定レイテンシ write index）を候補に段階移行する。

再更新:

1. 軽量履歴状態 sidecar 化も完了。
2. 固定レイテンシ用バッファ実体の sidecar 化も完了。
