# Practical Stable ISR Bridge Runtime 実装TODO

対象設計図書:

- `doc/work8/practical_stable_isr_bridge_runtime_migration_plan.md`
- `doc/work8/practical_stable_isr_bridge_runtime_ai_governance.md`
- `doc/work8/practical_stable_isr_bridge_runtime_audit.md`

## 0. 事前調査（統治規約 第1条）

- [x] `RuntimeWorld` 定義・呼出元・呼出先・派生利用箇所を列挙
- [x] `RuntimeState` 定義・呼出元・呼出先・派生利用箇所を列挙
- [x] `RuntimePublishWorld` 定義・呼出元・呼出先・派生利用箇所を列挙
- [x] `RuntimePublicationCoordinator` 定義・呼出元・呼出先・派生利用箇所を列挙
- [x] `RuntimeGraph` 定義・呼出元・呼出先・派生利用箇所を列挙
- [x] `AudioEngine` の publication/read/retire経路を列挙
- [x] `SnapshotCoordinator` の observe経路を列挙
- [x] `RetireRuntimeEx` の rollback/retire経路を列挙
- [x] `observeCurrentRuntime` 利用箇所を列挙
- [x] `getActiveRuntimeDSP` 利用箇所を列挙
- [x] `activeRuntimeDSPSlot` 利用箇所を列挙
- [x] `fadingRuntimeDSPSlot` 利用箇所を列挙

## 1. フェーズ1: 監視と境界の明確化

- [x] `RuntimeState/RuntimePublishWorld` を「Phase1暫定 authoritative」としてコード上で明示
- [x] `activeRuntimeDSPSlot/fadingRuntimeDSPSlot` を migration-only 用途へ限定
- [x] `observeCurrentRuntime()` を runtime read 主経路として文言・呼出経路で明示
- [x] `getActiveRuntimeDSP()` の用途を分類（publish前作業/retire/診断/互換）
- [x] world と slot の差分診断ログが追跡可能であることを確認

## 2. フェーズ2: RuntimeWorld 単一 Authority 化

- [x] `RuntimeWorld` 必須メンバー（generation/graph/publication.sequenceId/retire/visibility）を固定
- [x] authoritative runtime semantic を `RuntimeWorld` へ集約
- [x] read path の既定値を `RuntimePublicationCoordinator::observePublishedWorld()` に寄せる
- [x] `prepareToPlay/releaseResources/commitNewDSP` の slot参照の world化可能部分を置換
- [x] `RuntimeWorld` 以外への authoritative runtime semantic 保持を除去
- [x] `visibility` が許可要素のみ保持（metadata/policy state/violation status）

## 3. フェーズ3: Runtime Semantic Schema 機械可読化

- [x] `RuntimeFieldDescriptor` 相当の機械可読スキーマ定義を追加
- [x] 各 field に `AuthorityClass/Ownership/Mutability/Visibility/Lifetime` を付与
- [x] `RuntimeWorld/RuntimeGraph/PublicationSemantic` をスキーマ検証対象化
- [x] schema drift 検出をローカル検査（必要ならCI連携可能形式）で実施
- [x] `kRuntimeSemanticSchemaVersion` 更新条件を文書化
- [x] schema逸脱テストを少なくとも1本追加

## 4. フェーズ4: PublicationSequenceId 導入補強

- [x] `publication.sequenceId` を RuntimeWorld の常時構成要素として維持
- [x] sequence generator の単一性を確認（重複発番器の不在）
- [x] publish 直前の単調増分を確認・不足があれば補強
- [x] retire/evidence/log で同一 sequence を参照可能にする
- [x] sequence 欠損 publish を fail-close で拒否
- [x] sequence 欠損/重複シナリオ検証を追加

## 5. フェーズ5: Visibility Monotonicity 防御化

- [x] observe側 local state（`lastSeenGeneration/lastSeenSequenceId`）を追加
- [x] observe local state を Authority から分離（RuntimeWorld 非格納）
- [x] generation backward を異常として検出し telemetry/diagnostic に記録
- [x] sequence backward を異常、gap を telemetry（必要時 escalation）として処理
- [x] 異常時段階動作（telemetry -> quarantine/fail-close -> rollback候補）を接続
- [x] monotonic violation シナリオ検証を追加

## 6. フェーズ6: Legacy Runtime Semantic 除去

- [x] 用語（active/current/visible/latest）を整理しコードコメントへ反映
- [x] `observeCurrentRuntime()` 以外の現在値APIを責務分類（deprecated/migration-only/diagnostic-only）
- [x] 新規コードから legacy API 参照を禁止
- [x] legacy参照を read path から切り離し（互換・診断用途へ限定）

## 7. フェーズ7: Shadow Compare の fail-safe 接続

- [x] `RuntimeSemanticHash/TopologyHash/PublicationHash/Generation` 比較結果を整理
- [x] 旧observe経路と新observe経路の比較を実装
- [x] mismatch/violation閾値を定義
- [x] mismatch時に policy/evidence へ接続（Phase7では直接publish停止しない）
- [x] mismatchシナリオ再現テストを追加

## 8. フェーズ8: Retire Governance 強化

- [x] retire pressure 指標（queue/fallback/quarantine/latency）を明確化
- [x] high/low watermark と saturation enter/exit を明示
- [x] `RetireRuntimeEx` の lane/lifecycle/rollback state 整理
- [x] `maxRetireDeferralEpochs` 相当統治の必要性判断・実装
- [x] `maxObservedRetireDeferralEpochs` 記録を保証
- [x] 長時間運転相当で回復検証

## 9. フェーズ9: SoftRollback 最小実装

- [x] 異常検知条件を最小2種に限定（monotonic violation/publication mismatch）
- [x] 最後に正常 publish された RuntimeWorld を1件保持
- [x] 異常時に直前 world を再publishする最小 rollback を実装
- [x] rollback失敗時の安全側挙動を固定
- [x] `RetireRuntimeEx::requestRollback()` を実運用経路へ配線
- [x] 人工異常シナリオで rollback 成功/失敗の両系統を検証

## 10. 統治規約セルフチェック

- [x] Authority Owner が `RuntimePublicationCoordinator` のみであることを確認
- [x] RuntimeWorld の post-publish in-place mutation が無いことを確認
- [x] RuntimeWorld 肥大化禁止項目（Telemetry/Diagnostic等）非格納を確認
- [x] RuntimeGraph に禁止semantic（generation/sequence等）非格納を確認
- [x] Authority Mirror 新設なしを確認
- [x] 新規Thread/計画外Lock/将来フェーズ機能先行導入なしを確認

## 11. ビルド・テスト・最終検証

- [x] `Debug Build (cmd env retry)` 成功
- [x] `Strict Atomic Dot-Call Scan` 成功
- [x] 追加/更新テストを含む検証一式を実行し全Green
- [x] Observe/Publication/Retire の3経路で generation と sequenceId 追跡可能を確認
- [x] 変更箇所一覧・Authority経路確認・新旧参照比較・ビルド結果を最終報告に反映
