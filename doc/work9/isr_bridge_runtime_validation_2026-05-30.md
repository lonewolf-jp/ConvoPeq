# Practical Stable ISR Bridge Runtime 妥当性検証レポート（2026-05-30）

## 1. 目的

提示された評価・改修計画について、ConvoPeq 現行ソースを対象に「実運用で破綻しにくいか」の観点で妥当性を検証した。

---

## 2. 検証方法

- CodeGraph による構造探索・シンボル確認
- `grep` による関連実装の横断検索
- 主要ソースの直接精読

### 主な確認ファイル

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- `src/audioengine/RuntimeGraph.h`
- `src/audioengine/ISRRuntimeSemanticSchema.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/audioengine/ISRRetire.h`
- `src/audioengine/ISRRetire.cpp`
- `src/audioengine/ISRRetireRuntimeEx.h`
- `src/audioengine/ISRRetireRuntimeEx.cpp`
- `src/audioengine/ISRRTExecution.h`
- `src/audioengine/ISRRTExecution.cpp`
- `src/core/RuntimePublicationCoordinator.h`
- `src/core/RuntimeStore.h`
- `src/SafeStateSwapper.h`

補足:

- `ConvoPeq.md` はワークスペース内で検出できなかった。
- `AllpassDesigner` は `src/AllpassDesigner.*` および `src/convolver/*` 側で確認できた（ISR中核ではない）。

---

## 3. 総合判定（要約）

提示された評価は**大筋で妥当**。

- 設計思想は高水準（publication precheck / freeze/seal / HB意識）
- ただし、実行時の単一権威化（single semantic authority）と観測経路一本化（observe collapse）は未完
- 特に `RuntimeGraph` への実行状態混在と retire intent queue の可観測性不足は、長時間運用の破綻リスク要因になりうる

---

## 4. 主張ごとの妥当性判定

## 4.1 RuntimeWorld が唯一 Authority になっていない

### 判定（4.1）

妥当（強）

根拠:

- `AudioEngine::RuntimeReadView` が `graph` と `snapshot` を同時保持
- `readAudioRuntimeView()` 後に `getRuntimeGraph()` と `getRuntimeSnapshot()` を個別利用
- `AudioCallbackAuthorityView` でも `runtimeGraph` / `snapshot` / `preparedCrossfade` を同時に持つ

結論:

- observe source は 1 本化されておらず、RuntimeWorld 単一観測は未達。

## 4.2 Snapshot authority usage が残っている

### 判定（4.2）

妥当（強）

根拠:

- `makeRuntimeGraphState()` 内で snapshot から `eqBypassed`, `convBypassed`, `softClipEnabled`, `saturationAmount` 等を転写
- `buildRuntimePublishWorld()` でも `runtimeSnapshot` 由来で `routing.processingOrder` 等を設定

結論:

- snapshot は builder input を超えて意味依存に残っている。

## 4.3 RuntimeGraph に実行状態が混在

### 判定（4.3）

妥当（強）

根拠:

- `RuntimeGraph` に `activeNode`, `fadingNode` が存在
- さらに `dspCrossfadePending`, `dspCrossfadeUseDryAsOld`, `queuedFadeTimeSec`, `latencyDelayOld/New` など実行状態寄り項目を保持

結論:

- 「純粋 immutable processing description」への分離は未完。

## 4.4 Retire queue overflow ガバナンス不足

### 判定（4.4）

妥当（中〜強）

根拠:

- `ISRRetire::RETIRE_INTENT_QUEUE_SIZE = 256`
- `emitRetireIntent()` は full 時に無言 return（drop/overflow count なし）
- 一方で `AudioEngine` 側には retire saturation 系指標（`retireSaturation*`）が存在

結論:

- システム全体に緩和策はあるが、ISRRetire の局所経路は observability が不足。

## 4.5 Transition が authority semantic に混在

### 判定（4.5）

妥当（中〜強）

根拠:

- `ISRRuntimeSemanticSchema::ExecutionSemantic` に transition/crossfade 項目
- `buildRuntimePublishWorld()` で transition を world へ反映
- `commitNewDSP()` で `publishState(... TransitionPolicy ...)` による遷移を publish

結論:

- transition executor-local 化は未完。

## 4.6 Generation authority が多重

### 判定（4.6）

部分妥当

根拠:

- `generation`, `runtimeVersion`, `generationSemantic.runtimeGeneration`, `publication.sequenceId`, `epoch`, `mappedRuntimeGeneration` が共存
- ただし `runPublicationPrecheckNonRt()` で相互整合を厳格検査

結論:

- 無秩序ではないが、語彙整理（authoritative vs diagnostic mirror）が必要。

## 4.7 RuntimeWorldAuthority 不在

### 判定（4.7）

妥当（強）

根拠:

- `RuntimeWorldAuthority` / `observeWorld` は未検出
- 現状は `RuntimePublicationCoordinator + RuntimeStore + Bridge` の組み合わせ

---

## 5. 「思ったより堅い」既存実装

- `runPublicationPrecheckNonRt()` が強い
  - schemaVersion
  - descriptor set validate
  - generation/sequence/epoch monotonic 整合
  - `isFrozen()` / `isSealedRecursively()`
- publish/observe の memory order 設計が明示的
- `RuntimeStore` の publish swap / observe acquire は最小要件を満たす

---

## 6. 実運用観点での優先改修（推奨順）

1. **Observe Path Collapse**
   - RT observe を `const RuntimeWorld*` 1本に固定
   - `getRuntimeSnapshot/getRuntimeGraph` の個別経路を段階廃止

2. **RuntimeGraph 純化**
   - `activeNode/fadingNode/crossfade*` を `RuntimeExecutionState` へ分離
   - graph は topology + immutable params のみ

3. **Transition executor-local 化**
   - fade progress / ramp / dryAsOld 進捗を RT local state に限定

4. **Retire governance hardened**
   - `ISRRetire` に overflow/drop counter 追加
   - saturation と retire-intent queue を接続して可観測化

5. **Generation semantic 整理**
   - authoritative generation を 1 つに明示
   - その他は diagnostic mirror として命名・責務分離

---

## 7. 追加メモ

- `SafeStateSwapper` は `src/core` ではなく `src/SafeStateSwapper.h` に存在（Convolver 系の RCU 実装）
- `RTAllocatorFirewall` は実装上グローバルフラグで RT context を持つ構成であり、厳密な thread-local 化ではない

---

## 8. 最終結論

提示された「現在 65〜75% 到達、残課題は単一 authority 化・observe collapse・transition local化・retire governance」の方向性は、現行コード実体と整合している。

従って、提示改修計画は**妥当**。特に先行実施すべきは次の 2 点。

- Observe Path Collapse
- RuntimeGraph と RuntimeExecutionState の分離

この 2 点で、長時間運用時の partial visibility / semantic drift / retire race のリスクを大きく下げられる。
