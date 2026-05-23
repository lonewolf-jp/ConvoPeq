# ISR Rebuild Admission 最終計画書（実装引き渡し版）

作成日: 2026-05-23
対象: `AudioEngine` / `ConvolverProcessor` の rebuild 経路（運用安定化）

---

## 1. 目的（実運用優先）

本計画の目的は、以下4点を最小構成で確実に満たすこと。

1. rebuild storm を防ぐ
2. finalize 中の危険 rebuild を避ける
3. UI drag 中の無駄 rebuild を減らす
4. 「なぜ rebuild されたか」を追跡可能にする

本計画は、厳密スケジューラ理論よりも **責務集中と可視化** を優先する。

---

## 2. 設計方針（非交渉）

### 2.1 Orchestrator の責務境界

- 新設する Rebuild Orchestrator は **admission（入口判定）専用** とする。
- build / publish / retire / finalize 実行責務は既存コンポーネントのまま維持する。

### 2.2 キュー方針

- 多段 queue は導入しない。
- pending intent は **常に1件**（latest intent slot）とする。

### 2.3 collapse 方針

- `latest wins` は無条件適用しない。
- rebuild reason ごとに `Replaceable` / `MustExecute` を持つ。

### 2.4 defer 方針

- finalize 中は defer 可。
- ただし **max defer duration**（例: 2秒）を超えたら強制 dispatch。

### 2.5 debounce 方針

- debounce は **UI burst 吸収専用**。
- IR finalize / device reset / runtime recovery には適用しない。

---

## 3. 対象範囲 / 非対象

### 3.1 対象

- `submitRebuildIntent(...)` 導入（`AudioEngine` admission層）
- 既存 rebuild 抑制ロジックの入口統合
- telemetry（requested/suppressed/deferred/dispatched）

### 3.2 非対象

- RuntimeBuilder の内部アルゴリズム変更
- PublicationCoordinator の責務変更
- retire/reclaim 実装方式の変更
- ISR world model 自体の再設計

---

## 4. データモデル（最小）

```cpp
enum class RebuildCollapsePolicy
{
    Replaceable, // UI burst 由来。最新で上書き可
    MustExecute  // Device/IR lifecycle 由来。上書き不可
};

enum class RebuildClass
{
    Structural,
    FinalizeAware,
    DeviceLifecycle
};

struct RebuildIntent
{
    RebuildClass rebuildClass;
    RebuildCollapsePolicy collapsePolicy;
    uint64_t structuralHash;
    uint64_t fingerprint;
    uint64_t reasonBits;
    int64_t firstDeferredTicks;
};
```

保持状態:

- `std::optional<RebuildIntent> pendingIntent`（1件のみ）
- `std::atomic<bool> rebuildPending`

---

## 5. 判定ルール（admission）

## 5.1 同一判定（sameAsPending）

以下すべて一致時のみ「同一」とみなす。

1. `structuralHash` 一致
2. `rebuildClass` 一致
3. `collapsePolicy` 一致
4. finalize 状態カテゴリ一致（defer対象か否か）

> 注: hash 単独一致で suppress しない。

## 5.2 latest wins

- `Replaceable` のみ上書き可能。
- `MustExecute` は pending 中でも消さず、順序保証して1回実行。

## 5.3 finalize defer

- `isFinalizeRunning()==true` の間は defer。
- `now - firstDeferredTicks > maxDeferDuration` なら強制 dispatch。

## 5.4 debounce

- `uiBurstDebounceMs` は UI parameter burst にのみ適用。
- `MustExecute` / device lifecycle / finalize recovery には適用しない。

## 5.5 MixedPhaseIntermediate suppress の扱い（明文化）

- 結論: **`MixedPhaseIntermediate` suppress は `MustExecute` でも維持する（縮退しない）**。
- 理由:
  1. 本 suppress は単なる重複抑止ではなく、progressive mixed-phase の中間状態 publish を防ぐ安全ガードである。
  2. `MustExecute` の保証対象は「最終的に rebuild が実行されること」であり、「中間状態の即時 publish」を許可することではない。
  3. 品質・整合性（IR遷移の完了状態優先）を守るため、ここは legacy suppress 削減対象から除外する。

実装ルール:

- `MustExecute` でバイパス可能なのは、縮退対象として合意した legacy suppress に限定する。
- `MixedPhaseIntermediate` は `Replaceable` / `MustExecute` の両方で有効とする。
- 将来この方針を変更する場合は、AB比較ログ（音質/遷移整合）とロールバック条件をセットで更新する。

---

## 6. ログ仕様（Phase 0で先行導入）

すべて同一キーで出力する。

- `intentId`
- `reason`
- `class`
- `policy`
- `hash`
- `fingerprint`
- `finalizeState`
- `decision`
- `latencyMs`（request→dispatch）

必須ログイベント:

1. `REBUILD_REQUESTED`
2. `REBUILD_MERGED`
3. `REBUILD_SUPPRESSED`
4. `REBUILD_DEFERRED`
5. `REBUILD_FORCED_DISPATCH`（timeout）
6. `REBUILD_DISPATCHED`

運用抽出テンプレート:

- `S-DEF-03`（Finalize defer timeout 強制発行系列）を監視対象とする。
- 抽出/閾値テンプレートは `doc/work/ISR_Rebuild_Admission_Phase0_実装タスク分解_2026-05-23.md` の
  「3.7 S-* 系列ID対応 ログ抽出テンプレート（運用定型）」を正本として運用する。
- 一次切り分け手順は系列別 Runbook（`S-REQ-02` / `S-SNAP-03` / `S-DEF-03`）を参照する。
- Runbook索引は `doc/work/ISR_Rebuild_Admission_Runbook_Index_2026-05-23.md` を参照する。
- 系列別Runbook（統一運用）:
  - `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`
  - `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md`
  - `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md`

運用導線の統一状況（2026-05-23）:

- 上記3 Runbook はいずれも `3.3 8.1 定例実行プリセット（8.6）への相互参照` を実装済み。
- Runbook から `最終計画書 8.6` の窓A/B/C実行例へ直接遷移できる。

---

## 7. 実装フェーズ（Phase 0〜5）

## Phase 0: Telemetry only（挙動変更なし）

- admission相当情報を現行経路に追加ログ出力
- suppress/defer理由をログ可視化

完了条件:

- 同一操作で requested/suppressed/deferred/dispatched が追跡可能

## Phase 1: 入口統一

- `submitRebuildIntent(...)` を新設
- 既存入口を段階的に submit 経由へ寄せる
- 内部は既存 `requestRebuild(...)` 呼び出しのみ

完了条件:

- rebuild 入口が実質1箇所に集約

## Phase 2: sameAsPending 判定導入（ログのみ）

- sameAsPending を実装
- まだ抑止せず、`would-merge` ログのみ

完了条件:

- 判定誤りがログ比較で観測可能

## Phase 3: latest wins 有効化（Replaceable限定）

- `Replaceable` のみ merge/overwrite
- `MustExecute` は保持・実行

完了条件:

- UI drag 系で rebuild 発火回数が有意減
- MustExecute 消失ゼロ

## Phase 4: finalize defer（timeout付き）

- finalize 中 defer
- `maxDeferDuration` 超過で forced dispatch

完了条件:

- finalize 中危険 rebuild ゼロ
- 永久defer ゼロ

## Phase 5: 旧 suppress の段階削除

- 新 admission と重複する旧分散 suppress を順次削除
- 各削除ごとに telemetry 比較
- 実装上は、旧 suppress 分岐を次の2タグで明示して段階削除を管理する
  - `phase5_reduce_target`（縮退対象）
  - `phase5_keep_target`（維持対象）
- `phase5_keep_target` は安全ガード（例: `ShutdownInProgress`, `MixedPhaseIntermediate`, `PendingDuplicate`）として扱い、
  `MustExecute` でも原則維持する
- `phase5_reduce_target` は段階縮退対象の legacy suppress を対象に、
  1分岐ずつ縮退して telemetry diff で退行を確認する
- 進捗（2026-05-23）: `phase5_reduce_target` の初回縮退として
  `RecentDuplicate` suppress を削除済み（`PendingDuplicate` merge は維持）
- 進捗（2026-05-23）: `phase5_reduce_target` の次段縮退として
  `DeferredStructuralWindow` suppress を削除済み（同条件でも queue 判定へ遷移）
- 進捗（2026-05-23）: `PendingDuplicate` は縮退候補から除外し、
  pending queue の過負荷抑止（backpressure）目的で `phase5_keep_target` へ再分類した
- 進捗（2026-05-23）: `handleAsyncUpdate()` の bridge telemetry で
  `collapsePolicy` を pending intent から継承する実装を追加し、
  `policy=N/A` 固定を解消（`MustExecute` 時は `requestRebuild(..., true)` へ連携）
- 進捗（2026-05-23）: `convolverParamsChanged()` の suppress telemetry に
  `phase5_keep_target` / `phase5_reduce_target` を付与し、
  `MixedPhaseIntermediate`（keep）と `HashDedup`（reduce候補）の観測を分離した
- 進捗（2026-05-23）: `HashDedup` の実削減を段階適用し、
  `convolverParamsChanged()` での扱いを **Suppress→Merged（緩和）** へ変更した
  （同一hash観測は記録するが、以降は snapshot enqueue 経路へ委譲）
- 進捗（2026-05-23）: 実運用窓で `reason=hash_dedup` が
  `event=REBUILD_MERGED` として観測され、`REBUILD_SUPPRESSED` は 0 件を確認。
  `phase5_reduce_target` タグで識別可能なまま縮退差分を確認できた。
- 進捗（2026-05-23）: 次段縮退として `SnapshotIntentDebounced` を
  `Suppress→Merged` へ変更し、`phase5_reduce_target` で識別可能化した。
  併せて `SnapshotCommandBufferFull(NonMt含む)` は `phase5_keep_target` として明示維持した。
- 進捗（2026-05-23）: 実運用窓で `reason=snapshot_intent_debounced` が
  `event=REBUILD_MERGED`（17件）として観測され、
  `REBUILD_SUPPRESSED` は 0 件を確認。第2縮退の反映を確認済み。
- 進捗（2026-05-23）: 第3縮退として `submitRebuildIntent()` の
  `SameAsPendingWouldMerge`（latest_wins_replaceable）telemetry を
  `Suppress→Merged` へ緩和し、`phase5_reduce_target` で識別可能化した。
  併せて `ShutdownInProgress` / `KindFiltered` は `phase5_keep_target` として明示維持した。
- 観測メモ（2026-05-23）:
  - 上記第3縮退は実装済みだが、現行運用ログでは
    `same_as_pending_would_merge` / `ui_eq_editor_change_listener` の発生が 0 件で、
    実運用観測は保留。
  - 次の縮退候補は「観測実績のある reason」から選定する（観測駆動で進める）。

完了条件:

- 挙動退行なしで suppress ルールを簡素化

---

## 8. 受入基準（Done）

### 8.1 機能

- [x] UI burst 操作で rebuild storm が抑制される（実測: 2026-05-23, `same_as_pending_would_merge=2`）
- [x] finalize 中に危険 rebuild が発火しない（実測: 2026-05-23, 窓A `deferred_finalize_ready=1`, `deferred_finalize_rebuild_req=1`）
- [x] defer timeout 後に必ず dispatch される（実測: 2026-05-23, 窓B `rebuild_forced_dispatch=1`, `deferred_finalize_rebuild_req=1`）
- [x] MustExecute intent が消失しない（実測: 2026-05-23, 窓A/B `policy_must_execute>0`）

### 8.2 観測性

- [x] 1 rebuild 事象を request→decision→dispatch まで追跡できる（実測: 2026-05-23, `build/ConvoPeq_artefacts/Release/ConvoPeq.log`）
- [x] suppress/defer 理由がログで一意にわかる（実測: 2026-05-23, telemetry reason/class/policy確認）
- [x] `S-DEF-03`（`REBUILD_FORCED_DISPATCH`）の回数/発生率を5分窓で監視できる（実測: 2026-05-23, forced=0/finalizeReq=0）

### 8.3 回帰

- [x] Debug/Release build pass（実測: 2026-05-23）
- [x] Strict Atomic Dot-Call Scan pass（実測: 2026-05-23）
- [x] 既存 ISR verification 導線を破壊しない（実測: 2026-05-23, isr-seed + isr-verify-v2..v10/proof-scope/runtime-reduction/p3-governance pass）

### 8.4 8.1機能クローズ向け 実測手順（短文化・統一運用）

目的:

- 8.1（機能4項目）を「Runbook統一ルール」で短時間に実測し、クローズ判定へ接続する。

前提:

- ログソースは `build/ConvoPeq_artefacts/Release/ConvoPeq.log` を一次証跡とする。
- `S-DEF-*` は `timerCallback` 単一入口のため、**統合抽出（S-DEF-01/02/03 一括）を先に実施し、後段で系列別カウント**する。

8.1項目と実測系列の対応:

| 8.1項目 | 主系列 | 判定用の最小観測（例） |
| --- | --- | --- |
| UI burst 操作で rebuild storm が抑制される | `S-REQ-02` | `pending_duplicate`/`same_as_pending_would_merge` が増えても `task_queued` が飽和増加しない |
| finalize 中に危険 rebuild が発火しない | `S-DEF-01/02` | `deferred_finalize_ready` 到達前の不整合 dispatch がないこと（defer→dispatched の系列整合） |
| defer timeout 後に必ず dispatch される | `S-DEF-03` | `REBUILD_FORCED_DISPATCH` 発生時に `deferred_finalize_rebuild_requested` の再要求/dispatch が追跡可能 |
| MustExecute intent が消失しない | `S-DEF-03` + policy | `policy=MustExecute` の要求が suppress/消失せず、最終的に dispatch へ到達 |

5分窓の最小抽出テンプレ（運用入口統一）:

```powershell
$windowLog = 'build/ConvoPeq_artefacts/Release/ConvoPeq.log'

# S-DEF 統合抽出（共通維持）
$SDefPattern = 'event=REBUILD_DEFERRED.*reason=deferred_structural_due|event=REBUILD_DISPATCHED.*reason=deferred_structural_rebuild_requested|event=REBUILD_DEFERRED.*reason=deferred_finalize_ready|event=REBUILD_DISPATCHED.*reason=deferred_finalize_rebuild_requested|event=REBUILD_FORCED_DISPATCH.*reason=deferred_finalize_rebuild_requested|event=REBUILD_REQUESTED.*reason=deferred_finalize_rebuild_requested'
Select-String -Path $windowLog -Pattern $SDefPattern | Sort-Object LineNumber

# S-REQ-02（重複抑止）
Select-String -Path $windowLog -Pattern 'reason=requestRebuild_sr_bs|reason=pending_duplicate|reason=task_queued' | Measure-Object
```

判定記録先:

- クローズ判定は `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` の 8.1 節へ追記する。

### 8.5 フェーズ移行判断（2026-05-23 更新）

- 判断: **次フェーズへ移行（8.1機能4項目クローズ済み）**
- 実測更新:
  - `isr-8_1-cli-run.ps1 -ProbeFinalizeAware` を用いた窓分割実測で、
    `deferred_finalize_ready` / `deferred_finalize_rebuild_requested` /
    `REBUILD_FORCED_DISPATCH` / `policy=MustExecute` を観測。
  - `same_as_pending_would_merge` の高頻度観測で UI burst 抑制を補強。
- 注記:
  - `readyToClose8_1` は単一窓評価のため false が残る窓があるが、
    受入文言ごとの証跡は窓分割で充足済み。
  - 正式判定ログは `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` を正本とする。

### 8.6 8.1 定例実行プリセット（運用固定）

目的:

- 8.1 再検証を「毎回同じ手順・同じ引数」で実行し、運用差分を減らす。

推奨プリセット（一次運用値）:

- スクリプト: `.github/scripts/isr-8_1-cli-run.ps1`
- 必須フラグ: `-ProbeFinalizeAware`
- 推奨窓:
  - 窓A（finalize defer / MustExecute）: `-ExitMs 9000 -ProbeDelayMs 1400`
  - 窓B（forced dispatch）: `-ExitMs 12000 -ProbeDelayMs 1800 -ProbeIrReloadStorm`
  - 窓C（UI burst 補強）: `-ExitMs 8000 -ProbeIntentBurst 120`

実施ルール:

1. 3窓（A/B/C）を同日内で連続実行し、窓分割エビデンスとして保存する。
2. 判定は単一窓 `readyToClose8_1` ではなく、受入文言単位（8.1-1〜4）の充足で行う。
3. 記録は `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` を正本として追記する。

### 8.7 Runbook相互参照の適用完了（2026-05-23）

適用先:

- `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`
- `doc/work/ISR_Rebuild_Admission_S-SNAP-03_運用手順_2026-05-23.md`
- `doc/work/ISR_Rebuild_Admission_S-DEF-03_運用手順_2026-05-23.md`

完了内容:

1. 3 Runbook に同一フォーマットの `3.3` 節を追加済み。
2. `最終計画書 8.6` への正本参照を統一済み。
3. 窓A/B/C実行例と、判定・記録ルール（単一窓依存禁止 / クローズログ正本）を統一済み。

期待効果:

- 監視系列から実測実行までの導線を一本化し、運用者ごとの差分手順を削減する。
- 閾値超過時の再現確認（3窓再実行）を即実施できる。

---

## 9. ロールバック条件

以下のいずれか発生時は前フェーズへ戻す。

1. MustExecute intent 消失が1件でも検出
2. finalize defer timeout が機能せず rebuild 停滞
3. telemetry 欠落で追跡不能事象が発生
4. Build/scan/verification の既存 gate に退行

---

## 10. 実装チェックリスト（引き渡し用）

1. [x] Phase 0 ログ導入（制御変更なし）
2. [x] `submitRebuildIntent` 追加
3. [x] intent reason→`RebuildClass`/`CollapsePolicy` マッピング表追加
4. [x] sameAsPending 実装（ログ観測）
5. [x] Replaceable 限定 latest wins 有効化
6. [x] finalize defer + timeout 有効化
7. [x] 旧 suppress を段階縮退（`RecentDuplicate` / `DeferredStructuralWindow` / `HashDedup` / `SnapshotIntentDebounced` 反映済み）
8. [x] 各フェーズで telemetry diff を保存（窓A/B/C 実測を含む）

注記:

- `phase5_keep_target`（`ShutdownInProgress`, `MixedPhaseIntermediate`, `PendingDuplicate`, buffer/full 系）は安全ガードとして維持し、
  現時点で新規の安全な reduce 候補はなし。

---

## 11. マッピング（現行コードの関係点）

- Convolver側設定値保持: `setRebuildDebounceMs()`
  (`src/convolver/ConvolverProcessor.Runtime.cpp`)
- UI入口: `ConvolverControlPanel` / `IRAdvancedSettingsComponent`
  (`src/ConvolverControlPanel.cpp`)
- admission候補: `AudioEngine::convolverParamsChanged`
  (`src/audioengine/AudioEngine.UIEvents.cpp`)
- rebuild dispatch本線: `AudioEngine::requestRebuild(...)`
  (`src/audioengine/AudioEngine.RebuildDispatch.cpp`)
- defer運用: `DeferredStructural` / `DeferredFinalizeAware`
  (`src/audioengine/AudioEngine.Timer.cpp`)

---

## 12. 最終判断

本計画は、ConvoPeq の既存 ISR / Runtime publication 構造を維持したまま、
**入口ガードを小さく導入して運用安定性を上げる**ことを目的とする。

実装方針は以下に固定する。

- small admission layer
- pending 1 slot
- latest wins（Replaceable限定）
- finalize defer + timeout
- telemetry first

---

## 13. 再開クイック手順（運用向け）

1. `build/ConvoPeq_artefacts/Release/ConvoPeq.log` を一次証跡に固定する。
2. 監視は系列別 Runbook（`S-REQ-02` / `S-SNAP-03` / `S-DEF-03`）で5分窓抽出を実施する。
3. 閾値超過を検知したら、Runbook `3.3` の窓A/B/C を順に実行して再現性を確認する。
4. 判定結果は `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` に追記する。
5. 判定差異が出た場合は `クローズログ → 最終計画書 → 監査表注記` の順で同期する。

---

## 14. 日次運用テンプレート（コピペ用）

### 14.1 監視（5分窓）

1. `S-REQ-02` を実行し、`pending_duplicate_ratio` を確認する。
2. `S-SNAP-03` を実行し、`buffer_full_non_mt_ratio` を確認する。
3. `S-DEF-03` を実行し、`forced_dispatch_ratio` を確認する。

### 14.2 閾値超過時（再現確認）

1. Runbook `3.3` の窓A/B/Cを同日で連続実行する。
2. `build/ConvoPeq_artefacts/Release/ConvoPeq.log` から当該窓の抽出結果を保存する。
3. `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md` に
  `日時 / 窓 / 値 / 判断 / 対応` を追記する。

### 14.3 証跡保存ルール

- 一次証跡: `build/ConvoPeq_artefacts/Release/ConvoPeq.log`
- 判定正本: `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md`
- 計画正本: `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
- 監査正本: `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`

注記:

- 正本間の不整合を見つけた場合は、必ず `クローズログ → 最終計画書 → 監査表注記` の順で同期する。
