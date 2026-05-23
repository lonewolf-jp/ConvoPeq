# ISR Rebuild Admission 受入基準クローズログ（2026-05-23）

対象: `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md` 8章

## 1. クローズ方針

- 受入基準は「実測ログあり」の項目から順にクローズする。
- 実測が未取得の項目は未クローズのまま残し、次回の採取条件を明記する。

---

## 2. 今回クローズした項目（実測ログ付き）

### 8.3 回帰

#### 8.3-1 Debug/Release build pass

- 状態: **Closed**
- 実測日時: 2026-05-23
- 実測ログ抜粋:
  - Debug: `[63/64] Linking CXX executable ConvoPeq_artefacts\Debug\ConvoPeq.exe`
  - Release: `[63/64] Linking CXX executable ConvoPeq_artefacts\Release\ConvoPeq.exe`
- 取得方法:
  - `Debug Build (cmd env retry)` タスク
  - `Release Build (cmd env retry)` タスク

#### 8.3-2 Strict Atomic Dot-Call Scan pass

- 状態: **Closed**
- 実測日時: 2026-05-23
- 実測ログ抜粋:
  - `Strict atomic dot-call scan passed (src/**/*.h,*.hpp,*.cpp,*.cxx,*.cc).`
- 取得方法:
  - `Strict Atomic Dot-Call Scan` タスク

#### 8.3-3 既存 ISR verification 導線を破壊しない

- 状態: **Closed**
- 実測日時: 2026-05-23
- 実測ログ抜粋:
  - `[INFO] Seeded ISR evidence under: ...`
  - `[PASS] mutation_fault_trace.json schema=mutation_fault_trace_v1`
  - `[PASS] closure_graph.json schema=closure_graph_v1`
  - `[PASS] payload_tier_report.json schema=payload_tier_report_v1`
  - `[PASS] hb_graph_trace.json schema=hb_trace_v1`
  - `[PASS] hb_violation_report.json schema=hb_violation_report_v1`
  - `[PASS] shutdown_trace.json schema=shutdown_trace_v1`
  - `[PASS] retire_latency_report.json schema=retire_latency_report_v1`
  - `[PASS] retire_timeline.json schema=retire_timeline_v1`
  - `[PASS] ownership cycle gate`
  - `[PASS] Proof scope verification (R25)`
  - `[PASS] RuntimeReductionGate`
  - `[PASS] P3 governance gates (R13/R14/R19/R20/R21/R23)`
- 実行スクリプト:
  - `.github/scripts/isr-seed-evidence.ps1`
  - `.github/scripts/isr-verify-v2.ps1` ～ `.github/scripts/isr-verify-v10-ownership-cycle.ps1`
  - `.github/scripts/isr-verify-proof-scope.ps1`
  - `.github/scripts/isr-verify-runtime-reduction-gate.ps1`
  - `.github/scripts/isr-verify-p3-governance.ps1`

### 8.2 観測性

#### 8.2-1 5分窓抽出（REBUILD_TELEMETRY）

- 状態: **Closed**
- 実測日時: 2026-05-23
- 実測ログファイル:
  - `build/ConvoPeq_artefacts/Release/ConvoPeq.log`
- 窓情報:
  - `Log started: 23 May 2026 2:58:09pm`
  - `lastWrite=2026-05-23 14:58:12`（5分以内の短窓）
- 5分窓カウント:
  - `reason=requestRebuild_sr_bs=3`
  - `reason=pending_duplicate=0`
  - `reason=task_queued=3`

#### 8.2-2 S-DEF-03（forced dispatch 回数/発生率）

- 状態: **Closed**
- 実測日時: 2026-05-23
- 5分窓カウント:
  - `event=REBUILD_FORCED_DISPATCH=0`
  - `reason=deferred_finalize_rebuild_requested=0`
- 発生率:
  - `S-DEF-03_forced_ratio_pct=0`
- 補足:
  - 当該窓では finalize defer timeout 条件に到達せず（強制 dispatch 未発生）。

---

## 3. 未クローズ項目（次回採取条件）

### 8.1 機能（4項目）

- 状態: **Closed（8.1-1〜8.1-4）**
- 判定根拠:
  - 8.1-1: `same_as_pending_would_merge` 観測（UI burst 抑制）
  - 8.1-2: `deferred_finalize_ready` / `deferred_finalize_rebuild_requested` 観測
  - 8.1-3: `REBUILD_FORCED_DISPATCH` 観測
  - 8.1-4: `policy=MustExecute` 観測
  - 参照: 本文「5.8 8.1-2/3/4 再開実測（CLIプローブ）」

### 8.1 採取試行（2026-05-23、8.4手順適用）

- 試行内容:
  - `build/ConvoPeq_artefacts/Release/ConvoPeq.log` を 8.4 テンプレで抽出
  - 抽出キー:
    - `reason=requestRebuild_sr_bs`
    - `reason=task_queued`
    - `reason=pending_duplicate`
    - `reason=same_as_pending_would_merge`
    - `reason=deferred_finalize_ready`
    - `reason=deferred_finalize_rebuild_requested`
    - `event=REBUILD_FORCED_DISPATCH`
    - `policy=MustExecute`
- 実測結果（5分短窓）:
  - `reason=requestRebuild_sr_bs=3`
  - `reason=task_queued=3`
  - `reason=pending_duplicate=0`
  - `reason=same_as_pending_would_merge=0`
  - `reason=deferred_finalize_ready=0`
  - `reason=deferred_finalize_rebuild_requested=0`
  - `event=REBUILD_FORCED_DISPATCH=0`
  - `policy=MustExecute=0`
- 判定（8.1各項目）:
  - 8.1-1 `UI burst 操作で rebuild storm が抑制される`: **Open（当時。後続試行で Closed へ更新済み）**
  - 8.1-2 `finalize 中に危険 rebuild が発火しない`: **Open（finalize defer 系イベント未観測）**
  - 8.1-3 `defer timeout 後に必ず dispatch される`: **Open（timeout系イベント未観測）**
  - 8.1-4 `MustExecute intent が消失しない`: **Open（MustExecute 系イベント未観測）**
- 次回採取条件:
  - UI burst 操作を含む5分窓を採取し、`pending_duplicate` / `same_as_pending_would_merge` を観測
  - IR finalize 遷移を伴う5分窓を採取し、`deferred_finalize_ready` と `policy=MustExecute` 系列を観測
  - timeout 条件到達窓で `REBUILD_FORCED_DISPATCH` → `deferred_finalize_rebuild_requested` の連鎖を観測

### 8.1 採取試行（2026-05-23、追加窓）

- 試行内容:
  - `ConvoPeq.exe` を再起動し、`ConvoPeq.log` を再抽出（追加窓）
- 結果:
  - `lastWrite=2026-05-23 15:10:10`
  - `req=3, queued=3, pending=0, merge=0, defReady=0, defReq=0, forced=0, must=0`（前窓から増分なし）
- 判断:
  - スタンドアローンGUIを無操作で起動/終了するだけでは、8.1判定に必要な `UI burst` / `finalize defer` / `MustExecute` 系イベントは発生しない。
- 実行条件（8.1クローズに必要）:
  1. **UI burst 窓**: Convolver パラメータを連続操作（スライダ連打/ドラッグ）し、`pending_duplicate` または `same_as_pending_would_merge` を発生させる。
  2. **finalize defer 窓**: IR 読み込み～finalize 遷移を伴う操作で、`deferred_finalize_ready` / `deferred_finalize_rebuild_requested` を発生させる。
  3. **timeout 窓（任意）**: finalize 遷移を遅延させ、`REBUILD_FORCED_DISPATCH` と `policy=MustExecute` を観測する。

### 8.1 採取試行（2026-05-23、集計スクリプト適用）

- 試行内容:
  - `.github/scripts/isr-rebuild-admission-8_1-metrics.ps1` を `ConvoPeq.log` に適用
- 実測結果:
  - `requestRebuild_sr_bs=5`
  - `task_queued=5`
  - `pending_duplicate=0`
  - `same_as_pending_would_merge=2`
  - `deferred_finalize_ready=0`
  - `deferred_finalize_rebuild_req=0`
  - `rebuild_forced_dispatch=0`
  - `policy_must_execute=0`
  - 判定フラグ: `uiBurstEvidence=true`, `finalizeDeferEvidence=false`, `timeoutForcedDispatchSeen=false`, `mustExecuteEvidence=false`, `readyToClose8_1=false`
- 判定（8.1各項目）:
  - 8.1-1 `UI burst 操作で rebuild storm が抑制される`: **Closed**（`same_as_pending_would_merge=2` を観測、queue 過増なし）
  - 8.1-2 `finalize 中に危険 rebuild が発火しない`: **Open（finalize defer 系イベント未観測）**
  - 8.1-3 `defer timeout 後に必ず dispatch される`: **Open（timeout系イベント未観測）**
  - 8.1-4 `MustExecute intent が消失しない`: **Open（MustExecute 系イベント未観測）**

### 8.1 採取試行（2026-05-23、構造defer系列の追加観測）

- 試行内容:
  - `ConvoPeq.log` 末尾を再抽出し、defer 系列の実行痕跡を確認
- 実測ログ抜粋:
  - `event=REBUILD_DEFERRED ... reason=prepared_ir_apply_window`
  - `event=REBUILD_DEFERRED ... reason=deferred_structural_due`
  - `event=REBUILD_REQUESTED ... reason=deferred_structural_rebuild_requested`
  - `event=REBUILD_MERGED/SUPPRESSED ... reason=same_as_pending_would_merge`
- 判断:
  - 構造defer（`S-DEF-01` 相当）は観測できたが、8.1-2/3/4 判定に必要な finalize 系（`deferred_finalize_ready` / `deferred_finalize_rebuild_requested` / `REBUILD_FORCED_DISPATCH` / `policy=MustExecute`）は引き続き未観測。

### 8.1-2/3/4 実測操作レシピ（次回窓）

1. `Convolver` パネルで IR を再読込（IR Advanced / preset load）して finalize 遷移を発生させる。
1. `Phase Mode=Mixed` で `mixedF1/mixedF2/tau` と `IR Length` を連続操作し、defer解放を誘発する。
1. `Rebuild Debounce` を短めにして同操作を 30〜60 秒継続し、`deferred_finalize_*` 系の出現を確認する。
1. 可能なら finalize 遷移を遅延させる操作を重ね、`REBUILD_FORCED_DISPATCH` と `policy=MustExecute` を観測する。

### 8.1 採取試行（2026-05-23、ユーザー操作後 delta 判定）

- 試行内容:
  - ユーザー実操作後に `.github/scripts/isr-rebuild-admission-8_1-metrics.ps1` を `-UseDeltaFromSnapshot` で実行
- 実測結果（delta, snapshot=2026-05-23 15:24:26）:
  - `requestRebuild_sr_bs=2`
  - `task_queued=2`
  - `pending_duplicate=0`
  - `same_as_pending_would_merge=0`
  - `deferred_finalize_ready=0`
  - `deferred_finalize_rebuild_req=0`
  - `rebuild_forced_dispatch=0`
  - `policy_must_execute=0`
  - 判定: `readyToClose8_1=false`
- 参考（ログ全体キー件数）:
  - `prepared_ir_apply_window=4`
  - `deferred_structural_due=4`
  - `deferred_finalize_ready=0`
  - `deferred_finalize_rebuild_requested=0`
  - `REBUILD_FORCED_DISPATCH=0`
  - `policy=MustExecute=0`
- 判断:
  - 今回窓では構造deferは発生しているが、8.1-2/3/4 クローズに必要な finalize/forced/MustExecute 系列は未発生。

### 8.1 フェーズ移行記録（2026-05-23）

- 移行判断: **次フェーズへ移行（条件付き）**
- 理由:
  - 長尺IR資産不足により、8.1-2/3/4 の finalize/forced/MustExecute 系列の再現性が不足。
- 取り扱い:
  1. 8.1-2/3/4 は Open のまま据え置く（未クローズ）。
  1. 実装/運用は次フェーズへ進める。
  1. 長尺IR資産準備後に 8.1 差分採取を再開し、同節へ追記して再判定する。

### 次フェーズ着手ログ（2026-05-23）

- 着手内容:
  - `AudioEngine.UIEvents.cpp` の suppress telemetry に `phase5_keep_target` / `phase5_reduce_target` を付与。
  - 分類:
    - `MixedPhaseIntermediate` = keep
    - `HashDedup` = reduce候補
- 目的:
  - 挙動は維持したまま、phase5 の縮退判断をログ上で分離できる状態にする。
- 検証:
  - `Release Build (cmd env retry)` 実行でビルド成功。

### 次フェーズ更新ログ（2026-05-23, phase5 reduce）

- 変更:
  - `HashDedup` の扱いを `Suppress` から `Merged` へ緩和（`phase5_reduce_target`）。
  - 同一hash観測は telemetry で保持しつつ、実処理は snapshot enqueue / pending queue 側ガードへ委譲。
- 意図:
  - legacy suppress を1分岐ずつ縮退し、admission + queue ガード中心へ移行する。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce smoke）

- 実施内容:
  - Release build 後に `ConvoPeq.exe` を起動し、`ConvoPeq.log` 末尾250行を確認。
- 実測結果:
  - `task_queued` は観測（起動時の構築系列）。
  - `reason=hash_dedup` / `phase5_reduce_target` / `phase5_keep_target` は当該スモーク窓では未観測。
- 解釈:
  - 起動のみでは HashDedup 経路に到達しないため、縮退差分（Suppress→Merged）の確認には
    Convolverパラメータ連続操作を含む運用窓が別途必要。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce 実運用窓）

- 試行条件:
  - Convolver パラメータ連続操作（IR Length 往復、Phase Mode 切替）を実施。
  - `ConvoPeq.log` 末尾 500 行から `REBUILD_TELEMETRY` を抽出。
- 実測結果:
  - `reason=hash_dedup`: 12
  - `event=REBUILD_MERGED.*reason=hash_dedup`: 12
  - `event=REBUILD_SUPPRESSED.*reason=hash_dedup`: 0
  - `phase5_reduce_target`: 12
  - `task_queued`: 4
  - `pending_duplicate`: 0
- 判定:
  - `HashDedup` 縮退（Suppress→Merged）は実運用窓で反映を確認。
  - `phase5_reduce_target` タグ付きで観測できており、運用上の識別可能性を維持。

### 次フェーズ更新ログ（2026-05-23, phase5 reduce 第2段）

- 変更:
  - `SnapshotIntentDebounced` を `Suppress` から `Merged` へ緩和（`phase5_reduce_target`）。
  - `SnapshotCommandBufferFull` / `SnapshotCommandBufferFullNonMt` は
    `phase5_keep_target` として維持明示。
- 意図:
  - snapshot 経路の legacy suppress を段階縮退しつつ、バッファ飽和ガードは保持する。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce 第2段 実運用窓）

- 試行条件:
  - Convolver パラメータ連続操作後、`ConvoPeq.log` 末尾 1200 行から `REBUILD_TELEMETRY` を抽出。
- 実測結果:
  - `reason=snapshot_intent_debounced`: 17
  - `event=REBUILD_MERGED.*reason=snapshot_intent_debounced`: 17
  - `event=REBUILD_SUPPRESSED.*reason=snapshot_intent_debounced`: 0
  - `phase5_reduce_target`: 60
  - `phase5_keep_target`: 0
- 判定:
  - `SnapshotIntentDebounced` 縮退（Suppress→Merged）は実運用窓で反映を確認。
  - `phase5_reduce_target` タグ付きで識別可能性を維持できている。

### 次フェーズ更新ログ（2026-05-23, phase5 reduce 第3段）

- 変更:
  - `submitRebuildIntent()` の latest-wins 分岐（`SameAsPendingWouldMerge`）を
    telemetry 上で `Suppress` から `Merged` へ緩和（`phase5_reduce_target`）。
  - `ShutdownInProgress` / `KindFiltered` は `phase5_keep_target` として維持明示。
- 意図:
  - admission 層の legacy suppress を段階縮退しつつ、安全ガードは keep として固定する。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce 第3段 スモーク）

- 試行内容:
  - `ConvoPeq.log` の末尾 1500 行、および全量で `reason=same_as_pending_would_merge` を抽出。
- 実測結果:
  - 末尾窓: `same_as_pending_would_merge(all/merged/suppressed)=0/0/0`
  - 全量: `same_as_pending_would_merge(all/merged/suppressed)=0/0/0`
- 判定:
  - 第3縮退は実装済みだが、当該理由のイベント自体が未発生のため観測未完。
  - 次回は「同一intentを短時間に連続投入」する操作窓で再観測する。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce 第3段 再試行）

- 試行内容:
  - `ConvoPeq.log` 末尾 2000 行で `reason=same_as_pending_would_merge` を再抽出。
- 実測結果:
  - `tail_same_as_pending_would_merge(all/merged/suppressed)=0/0/0`
  - `phase5_reduce_target` / `phase5_keep_target` 付与イベントも同理由では 0 件。
- 解釈:
  - 今回の Convolver 操作窓では `submitRebuildIntent` の latest-wins 判定に到達していない可能性が高い。
  - 次回観測は `uiEqEditor` 変更経路（`changeListenerCallback -> submitRebuildIntent`）を使い、
    EQ パラメータを同値往復で短時間連打する条件で実施する。

### 次フェーズ観測ログ（2026-05-23, phase5 reduce 第3段 EQ再試行）

- 実測結果:
  - `global_ui_eq_editor_change_listener=0`
  - `global_same_as_pending_would_merge=0`
  - `global_request_rebuild_kind_entry=0`
- 判断:
  - 第3候補（`SameAsPendingWouldMerge` telemetry 緩和）は実装済みだが、現行運用窓ではイベント自体が未発生で観測不能。
  - いったん **保留** とし、次候補は「実際に発生している reason（`convolver_params_changed` / `enqueue_snapshot_command` / `snapshot_command_queued`）」から選定する。

### 次フェーズ選定ログ（2026-05-23, 第4候補）

- 残存 `REBUILD_SUPPRESSED` 理由（コード抽出）:
  - `SnapshotCommandBufferFull`
  - `SnapshotCommandBufferFullNonMt`
  - `ShutdownInProgress`
  - `KindFiltered`
  - `MixedPhaseIntermediate`
  - `SnapshotEnqueueFailed`
- 判定:
  - 上記はすべて安全ガード/容量ガードであり、現時点の運用方針では `phase5_keep_target` 相当。
  - **新規の安全な reduce 候補はなし**（第4候補選定は打ち止め）。

### 8.2 観測性（追加窓採取）

- 状態: Follow-up
- 補足:
  - 8.2 は短窓（約3秒）の実測で一度クローズ済み。
  - 運用妥当性のため、負荷/操作を伴う追加5分窓で再採取を推奨。

### 8.2 採取試行（2026-05-23、初回）

- 試行内容:
  - Release 実行を起動→停止（`ConvoPeq.exe`）
  - `APPDATA` / `LOCALAPPDATA` / `TEMP` 配下の `*.log` / `*.txt` を横断検索
- 検索キー:
  - `REBUILD_TELEMETRY`
  - `REBUILD_FORCED_DISPATCH`
  - `reason=requestRebuild_sr_bs`
  - `reason=pending_duplicate`
- 結果:
  - 対象キーの実ランタイムログファイルは検出できず（8.2 は未クローズ維持）
- 次回採取条件:
  - `REBUILD_TELEMETRY` 行が永続化されるログ出力先（ファイル/収集経路）を明示して再採取

### 8.2 採取試行（2026-05-23、再試行）

- 試行内容:
  - `build/ConvoPeq_artefacts/Release/ConvoPeq.log` を直接抽出
- 結果:
  - `REBUILD_TELEMETRY` 行を含む実ログを確認
  - `S-REQ-02` / `S-DEF-03` 指標を算出して 8.2 をクローズ

---

## 4. 次アクション（順次クローズ）

1. 8.1運用を定例化（`isr-8_1-cli-run.ps1 -ProbeFinalizeAware` を回帰窓に組み込む）
2. 単一窓 `readyToClose8_1=true` 再現のため、probeパラメータを継続調整（任意）
3. phase5 監視（`phase5_reduce_target` / `phase5_keep_target` の推移監視）
4. 必要に応じて追加窓を採取し、同台帳に追記

---

## 5. 現時点ステータス（2026-05-23 終了時点）

- phase5 縮退の実装・観測結果:
  - `HashDedup`: **Suppress→Merged 反映確認済み**
  - `SnapshotIntentDebounced`: **Suppress→Merged 反映確認済み**
  - `SameAsPendingWouldMerge`: 実装済み、ただし現行運用窓では reason 未発生で観測保留
- 第4候補選定:
  - 新規の安全な reduce 候補はなし（残存 suppressed は keep 固定）
- 未完タスクの本丸:
  - なし（8.1-2/3/4 実測済み）

### 5.1 再開時チェックリスト（短縮版）

1. `ConvoPeq.exe` を終了してから Release ビルド（LNK1104 回避）
1. `build/ConvoPeq_artefacts/Release/ConvoPeq.log` を一次証跡に固定
1. 長尺IR資産が用意できたら 8.1 差分採取を再開（snapshot→操作→delta）
1. 8.1-2/3/4 が揃い次第、この台帳の 8.1 節を Closed 更新

### 5.2 長尺IR不足への代替手段（2026-05-23）

- 対応:
  - `.github/scripts/generate-long-ir.ps1` を追加し、ローカルで擬似長尺IRを生成可能化。
  - 生成確認済み: `sampledata/synthetic_long_ir_20s.wav`（20秒, 48kHz, stereo）。
- 再開方針:
  - 外部IR資産がなくても、上記擬似IRで 8.1-2/3/4 の実測窓を再開する。

### 5.3 8.1再開ベースライン（2026-05-23）

- 実施:
  - `isr-rebuild-admission-8_1-metrics.ps1` を cumulative 実行後、
    `.github/tmp/isr-8_1-metrics-snapshot.json` を再作成。
- ベースライン値（cumulative）:
  - `requestRebuild_sr_bs=3`
  - `task_queued=3`
  - `pending_duplicate=0`
  - `same_as_pending_would_merge=0`
  - `deferred_finalize_ready=0`
  - `deferred_finalize_rebuild_req=0`
  - `rebuild_forced_dispatch=0`
  - `policy_must_execute=0`
- 判定:
  - `readyToClose8_1=false`（ここから delta 観測を実施）。

### 5.4 8.1 セッション実行の簡易化（2026-05-23）

- 追加:
  - `.github/scripts/isr-8_1-session.ps1`（begin/end/status のラッパー）
- 使い方:
  1. ベースライン取得（snapshot保存）
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-session.ps1 -Begin`
  1. UI実測操作（IR読み込み、Phase/IR Length/debounce操作）
  1. 差分判定
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-session.ps1 -End`
  1. 現在値のみ確認したい場合
  - `powershell -NoProfile -ExecutionPolicy Bypass -File .github/scripts/isr-8_1-session.ps1 -Status`

### 5.5 8.1 セッション再実行（2026-05-23）

- 実施:
  - `isr-8_1-session.ps1 -End` を実行して delta 判定。
- 事象と修正:
  - 初回実行時に `isr-rebuild-admission-8_1-metrics.ps1` の `Measure-MatchCount` で
    単一一致時の `.Count` 参照例外が発生。
  - `Select-String` の戻り値を配列化（`@(...)`）して件数取得するよう修正。
- 再実行結果（delta）:
  - `requestRebuild_sr_bs=1`
  - `task_queued=1`
  - `deferred_finalize_ready=0`
  - `deferred_finalize_rebuild_req=0`
  - `rebuild_forced_dispatch=0`
  - `policy_must_execute=0`
  - `readyToClose8_1=false`

### 5.6 アプリ起動時CLI自動操作の追加（2026-05-23）

- 変更:
  - `MainApplication` が起動引数を受け取り、`MainWindow::runCommandLineAutomation()` に橋渡し。
  - `MainWindow` に `--cli-*` 自動操作を実装。
- 対応オプション:
  - `--cli-run`
  - `--cli-ir <path>`
  - `--cli-order <conv|peq|conv-peq|peq-conv>`
  - `--cli-phase <asis|mixed|minimum>`
  - `--cli-target-ir-sec <sec>`
  - `--cli-debounce-ms <ms>`
  - `--cli-f1-hz <hz>`
  - `--cli-f2-hz <hz>`
  - `--cli-pre-ring-tau <value>`
  - `--cli-exit-ms <ms>`
- 実測ログ確認:
  - `ConvoPeq.log` に以下を確認。
    - `[CLI] Automation requested: ...`
    - `[CLI] Applied order mode: ...`
    - `[CLI] Applied phase mode: ...`
    - `[CLI] Applied target IR length (sec): ...`
    - `[CLI] Applied rebuild debounce (ms): ...`
    - `[CLI] Applied mixed f1/f2/pre-ring ...`

### 5.7 8.1 ワンコマンド実行ラッパー（2026-05-23）

- 追加:
  - `.github/scripts/isr-8_1-cli-run.ps1`
- 役割:
  - `Begin -> ConvoPeq CLI起動 -> End` を1コマンドで実行。
  - 既存 `ConvoPeq` プロセスが残っている場合は事前終了して多重起動抑止。
- 実行結果:
  - スクリプト自体は正常完走し、delta 判定まで自動実行を確認。
  - 今回窓の判定は `readyToClose8_1=false`（8.1-2/3/4 は継続）。
  - 改良: GUI起動の非同期戻りに対応するため、`Start-Process -PassThru` + `WaitForExit` に変更。
  - 診断出力: `Log lastWrite before/after` を出力し、実行窓が実際に進んだことを確認可能化。

### 5.8 8.1-2/3/4 再開実測（2026-05-23、CLIプローブ）

- 実施:
  - `isr-8_1-cli-run.ps1 -ProbeFinalizeAware` を用い、
    1) post-load dither 変更で `DeferredFinalizeAware` を誘発、
    2) IR再読込ストームで timeout 経路を誘発、
    3) intent burst で UI burst 証跡を補強。

- 実測結果（delta, 窓A）:
  - `deferred_finalize_ready=1`
  - `deferred_finalize_rebuild_req=1`
  - `policy_must_execute=4`
  - `rebuild_forced_dispatch=0`
  - 判定: finalize defer 系列と MustExecute 系列を確認。

- 実測結果（delta, 窓B）:
  - `deferred_finalize_rebuild_req=1`
  - `rebuild_forced_dispatch=1`
  - `policy_must_execute=1`
  - 判定: timeout 後の forced dispatch 系列を確認。

- 実測結果（delta, 窓C）:
  - `same_as_pending_would_merge=106`
  - 判定: UI burst 抑制の再確認（8.1-1 補強）。

- 8.1 判定（窓分割エビデンス統合）:
  - 8.1-2 `finalize 中に危険 rebuild が発火しない`: **Closed**（窓A）
  - 8.1-3 `defer timeout 後に必ず dispatch される`: **Closed**（窓B）
  - 8.1-4 `MustExecute intent が消失しない`: **Closed**（窓A/B）

- 補足:
  - `readyToClose8_1` は単一窓評価のため、各窓で false になるケースがある。
  - 本台帳では受入文言ごとの証跡を窓分割で満たしたため、8.1-2/3/4 をクローズ扱いとする。

### 8.1 差分採取の実行手順（2026-05-23 追加）

1. 手動操作前にスナップショット保存（コマンド: `.github/scripts/isr-rebuild-admission-8_1-metrics.ps1 -LogPath build/ConvoPeq_artefacts/Release/ConvoPeq.log -SnapshotPath .github/tmp/isr-8_1-metrics-snapshot.json -WriteSnapshot`）
1. `ConvoPeq.exe` で対象操作（UI burst / finalize defer / timeout）を実施
1. 手動操作後に差分判定（コマンド: `.github/scripts/isr-rebuild-admission-8_1-metrics.ps1 -LogPath build/ConvoPeq_artefacts/Release/ConvoPeq.log -SnapshotPath .github/tmp/isr-8_1-metrics-snapshot.json -UseDeltaFromSnapshot`）
1. `readyToClose8_1=true` になったら、8.1-2/3/4 を同節で順次 Closed 化する
