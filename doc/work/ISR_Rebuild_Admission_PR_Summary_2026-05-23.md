# ISR Rebuild Admission PR Summary（2026-05-23）

## 1. 目的

`requestRebuild` 周辺の legacy suppress を段階縮退しつつ、
`MustExecute` の最終実行保証と運用可観測性（telemetry / Runbook）を維持する。

---

## 2. このセッションでの実装確定点

### 2.1 `src/audioengine/AudioEngine.RebuildDispatch.cpp`

- `submitRebuildIntent(...)` を中心に admission 入口を統一し、
  `Requested/Merged/Suppressed/Deferred/Dispatched` を記録。
- `requestRebuild(double,int,bool bypassLegacySuppression)` へ拡張。
- Phase5 段階縮退:
  - 削除済み: `RecentDuplicate` suppress
  - 削除済み: `DeferredStructuralWindow` suppress
- 維持対象（KEEP）:
  - `ShutdownInProgress`
  - `MixedPhaseIntermediate`
  - `PendingDuplicate`（pending queue の backpressure 保護）
- `PHASE5-KEEP/REDUCE` ログと `phase5_keep_target/phase5_reduce_target` タグで
  分岐意図を可視化。

---

## 3. 文書同期（運用・計画）

- `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
  - Phase5 の進捗・最終分類を反映（PendingDuplicate は KEEP）。
- `doc/work/ISR_Rebuild_Admission_Phase0_実装タスク分解_2026-05-23.md`
  - `requestRebuild` の Reason/Event マッピングを現行コードへ同期。
  - `S-REQ-02` テンプレートを `pending_duplicate` 中心へ更新。
- `doc/work/ISR_Rebuild_Admission_S-REQ-02_運用手順_2026-05-23.md`
  - KPI/抽出式/記録欄を `pending_duplicate_ratio` 基準へ更新。
- `doc/work/ISR_Rebuild_Admission_Runbook_Index_2026-05-23.md`
  - `S-REQ-02` 説明を `pending_duplicate` 前提へ更新。

---

## 4. 検証結果（本セッション実行）

- `Strict Atomic Dot-Call Scan`: PASS
- Debug build (`Debug Build (cmd env retry)`): PASS
- Release build (`Release Build (cmd env retry)`): PASS
- 変更対象ファイルの診断（`get_errors`）: 問題なし

---

## 5. リスク評価（短評）

- `RecentDuplicate` / `DeferredStructuralWindow` 削除後も、
  `PendingDuplicate` merge を維持したため、同一 pending の過剰置換リスクを抑制。
- `MixedPhaseIntermediate` を KEEP 維持しているため、
  progressive mixed-phase 中間 publish の安全性を担保。

---

## 6. 備考

作業ツリーには Admission 以外の変更も存在するため、PR作成時は
本サマリー記載の対象ファイルに限定して差分確認すること。

---

## 7. 正本参照（レビュー時の固定導線）

- 受入/フェーズ基準（計画正本）:
  - `doc/work/ISR_Rebuild_Admission_最終計画書_2026-05-23.md`
- 受入クローズ判定（判定正本）:
  - `doc/work/ISR_Rebuild_Admission_受入基準クローズログ_2026-05-23.md`
- R11〜R25監査（監査正本）:
  - `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`

運用ルール: 判定差異が見つかった場合は、`クローズログ → 最終計画書 → 監査表注記` の順で同期する。
