# TODO implementation (work7 bridge runtime)

## 目的

`doc/work7/implementation_priorities_bridge_runtime.md` と `doc/work7/ai_governance_v1_1_review.md`、`doc/work7/practical_stable_isr_bridge_runtime_migration_review_2026-05-30.md` をもとに、Practical Stable ISR Bridge Runtime を実装・文書化・検証するための作業台帳。

## 実装タスク

### M1: Observe Path Collapse

- [x] T1-1 `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` の Audio callback 観測ソースを単一 authority view に収束させる
- [x] T1-2 `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` の Audio callback 観測ソースを単一 authority view に収束させる
- [x] T1-3 `src/audioengine/AudioEngine.Processing.Snapshot.cpp` の `processWithSnapshot` 観測面を統一し、callback 配下の追加 authority 再取得を排除する

### M2: Snapshot Semantic Unification

- [x] T2-1 `doc/work7/` に semantic source table を作成し、active / fading / crossfade の取得元を固定する
- [x] T2-2 crossfade prepared snapshot の責務を再定義し、参照側を単一路線に統一する

### M3: Publication Invariant Centralization

- [x] T3-1 `AudioEngine.Commit.cpp` / `AudioEngine.Timer.cpp` / `AudioEngine.Processing.PrepareToPlay.cpp` / `AudioEngine.Processing.ReleaseResources.cpp` の publish 前後 invariant チェックを共通化する
- [x] T3-2 publish 失敗時の fail-safe 方針と診断タグを統一する

### M4: Retire Governance Validation

- [x] T4-1 `src/audioengine/AudioEngine.Threading.cpp` と運用文書に retire 閾値の運用基準を追記する
- [x] T4-2 retire / fallback queue depth の監視観点と超過時アクションを文書化する
- [x] T4-3 Retire telemetry baseline としてメトリクス名・周期・閾値・収束条件を固定する

### M5: RuntimeGraph → RuntimeWorld 依存整理

- [x] T5-1 runtime graph / DSP resolve 経路の依存分離スパイク文書を作成し、段階移行案を整理する

### M6: Contract/Verifier 可視化

- [x] T6-1 非RT側の validator 群を束ねる Contract/Verifier 相当レイヤーの命名と追跡窓口を整備する

## テスト / 検証タスク

### 静的検証

- [x] 変更対象ファイルの診断を全件クリアにする
- [x] Audio callback 配下で `readAudioRuntimeView(...)` / `getRuntimeSnapshot(...)` / `consumeCrossfadePreparedSnapshot(...)` の再取得が残っていないことを確認する
- [x] `RuntimeGraph` / `RuntimeSnapshot` / `CrossfadePreparedSnapshot` への直接参照残存を確認する
- [x] publish 入口が Commit / Timer / PrepareToPlay / ReleaseResources に収束していることを確認する
- [x] retire telemetry 名称と運用基準が文書と実装で一致していることを確認する

### ビルド / スキャン

- [x] Debug ビルドを成功させる
- [x] Release ビルドを成功させる
- [x] strict atomic dot-call scan を成功させる
- [x] list compliance scan を実行し、失敗件数 0 を確認する
- [x] RT 安全制約に反する追加がないことを再確認する

### 運用ゲート

- [x] Long Run IR Switch Test を実行し、30分以上の連続 IR 切替で破綻がないことを確認する
- [x] Publish Burst Test を実行し、100回以上の連続 publish と回復性を確認する
- [x] Retire Queue Saturation Test を実行し、飽和後の回復を確認する
- [x] Crossfade Stress Test を実行し、click / pop / state inconsistency / semantic divergence が出ないことを確認する

### 設計図書網羅マップ

- [x] Observe Path Collapse 実装差分を反映する
- [x] Snapshot Semantic Unification 実装差分を反映する
- [x] Publication Invariant Centralization 実装差分を反映する
- [x] Retire Governance 運用基準文書を反映する
- [x] Retire Telemetry Baseline 文書を反映する
- [x] 依存整理スパイク文書を反映する
- [x] Contract/Verifier 可視化を反映する
- [x] Long Run / Publish Burst / Retire Saturation / Crossfade Stress の試験記録を反映する

### 完了条件

- [x] 上記の実装・文書・検証タスクがすべて完了する
- [x] 主要ビルドと検証スキャンが Green になる
- [x] 変更内容が `doc/work7` の設計図書と整合している
