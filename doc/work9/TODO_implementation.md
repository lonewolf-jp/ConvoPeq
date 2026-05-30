# ISR Bridge Runtime 実装TODO（rev13準拠）

作成日: 2026-05-30
参照:

- `doc/work9/isr_bridge_runtime_implementation_tasks_phase_file_2026-05-30_rev13.md`
- `doc/work9/isr_bridge_runtime_ai_governance_v1_7_2026-05-30.md`
- `doc/work9/isr_bridge_runtime_validation_2026-05-30.md`

## 進捗ログ

- 2026-05-30: 漏れ監査レポート（LA-01〜LA-05）に基づく改修漏れを収束。
  - 変更: `AudioEngine.h`（RT snapshot 保持除去 / RuntimeWorld 生成封鎖 / 監査メトリクス追加）
  - 変更: `AudioEngine.Commit.cpp`（publish/retire メトリクス更新、Grace/Free 条件明示）
  - 変更: `ISRRetireRuntimeEx.h`（`isGracePeriodCompleted` / `canTransitionRetirePendingToFree` 追加）
  - 変更: `SpectrumAnalyzerComponent.cpp`（snapshot 取得アクセサ統一）
  - 追加: `src/tests/RetireGraceSemanticsTests.cpp`
  - 検証: Debug build 成功 / CTest 4件 pass / Strict Atomic Dot-Call Scan pass
- 2026-05-30: `RuntimeGraph` から crossfade/latency 実行進捗フィールドを削除し、Execution 側へ寄せる分離を前進。
  - 削除: `dspCrossfadePending`, `dspCrossfadeUseDryAsOld`, `firstIrDryCrossfadePending`, `queuedFadeTimeSec`, `dspCrossfadeStartDelayBlocks`, `dspCrossfadeDryHoldSamples`, `dryScaleTarget`, `latencyDelayOld`, `latencyDelayNew`, `latencyResetPending`
  - 検証: Debug build 成功 / Release build 成功 / Strict Atomic Dot-Call Scan pass
- 2026-05-30: 最小 CTest を追加し、採番・メタデータ初期値の回帰検証を自動化。
  - 追加: `src/tests/ISRRuntimeIdentityGeneratorsTests.cpp`
  - 検証: `ISRRuntimeIdentityGenerators` テスト pass (1/1)
- 2026-05-30: `RuntimePublicationCoordinator` の拒否契約回帰テストを追加。
  - 追加: `src/tests/RuntimePublicationCoordinatorTests.cpp`
  - 検証: `RuntimePublicationCoordinatorRejects` テスト pass (1/1)
  - 検証内容: 再publish拒否、rollback拒否（101→100）、publicationSequence 単調増加と拒否時不変
- 2026-05-30: RT 経路の snapshot 依存を停止し、authority 判定源を RuntimeWorld 側へ収束。
  - 変更: `AudioEngine.Processing.AudioBlock.cpp` / `AudioEngine.Processing.BlockDouble.cpp` / `AudioEngine.h`
  - 検証: Debug build 成功 / CTest 3件 pass / Strict Atomic Dot-Call Scan pass
- 2026-05-30: ISR semantic validation 回帰テストを追加。
  - 追加: `src/tests/ISRSemanticValidationTests.cpp`
  - 検証: `ISRSemanticValidationRejects` テスト pass (1/1)
  - 検証内容: invalid closure reject / invalid payload tier reject

## 0. 事前調査・監査証跡

- [x] Serena で `RuntimeWorld/Authority/Store/Retire/Snapshot/Builder` の参照一覧を取得
- [x] CodeGraph で direct/indirect caller・callee を取得
- [x] 影響範囲一覧（grep + Serena + CodeGraph の和集合）を作成
- [x] 影響範囲内の主要シンボルごとに「修正/修正不要」を記録
- [x] publish/observe/retire/generation/worldId/publicationSequence の data-flow を列挙

## 1. Authority 一元化（publish/observe/retire）

- [x] publish 経路を `RuntimeWorldAuthority` 相当の単一路へ収束（bypass除去）
- [x] `RuntimePublicationCoordinator::publish(...)` 直接利用を禁止・置換
- [x] retire 登録/解放要求確定を retire coordinator 管理下へ統一
- [x] 外部公開 `retireWorld()` を非公開化（存在時）
- [x] 同一 world 再 publish を拒否する検証を実装

## 2. RuntimeWorld モデル/不変条件

- [x] `RuntimeWorld` の必須要素（worldId/generation/graph/executionDescriptor/metadata）を確認・不足補完
- [x] `RuntimeMetadata` の `schemaVersion/publicationSequence` を必須化
- [x] `RuntimeWorld` から snapshot 参照/保持を除去
- [x] world 内 `mutable`/post-publish 更新経路を除去
- [x] constructor を封鎖し Builder 以外から生成不可にする
- [x] Freeze 後 `shared_ptr<const RuntimeWorld>` を強制

## 3. RuntimeGraph / ExecutionDescriptor / Snapshot 境界

- [x] `RuntimeGraph` から実行進捗状態（active/fading/crossfade進捗等）を分離
- [x] `ExecutionDescriptor` に progress/live counter/ownership が混入しないよう是正
- [x] snapshot は builder input 専用に制限し runtime authority 判定源から除外
- [x] RT から Snapshot 到達経路を 0 にする

## 4. Identity / Generation / Sequence 規約

- [x] `RuntimeWorldIdGenerator` を唯一発番主体に統一
- [x] `RuntimeGenerationGenerator` を唯一採番主体に統一
- [x] `publicationSequence` を global monotonic + commit直前確定に統一
- [x] validation 失敗時に publicationSequence 未確定を担保
- [x] rollback 検出 fail-closed を authority で担保

## 5. Observe API ガード

- [x] `observeWorldHandle()` に `[[nodiscard]]` を付与
- [x] observe 戻り値の長期保持禁止（メンバ保存/キャッシュ/async capture）を静的検査で担保
- [x] RT observe は `const RuntimeWorld*` のみに制約
- [x] RT path で `shared_ptr` copy/reset/move を排除

## 6. Retire / GracePeriod / Free

- [x] GracePeriod 判定を single-reader 最適化規約へ適合
- [x] RetirePending->Free 条件（grace完了 + pending正当取得 + authoritative ownership放棄）を明示
- [x] free責務（確定）と destroy条件（refcount 0）を分離
- [x] retire queue で Grace 条件を満たした world のみ Free 遷移に制約
- [x] retire intent queue overflow/drop の可観測メトリクスを追加

## 7. 監査メトリクス

- [x] pending/retire/published/observed の世代・件数・年齢メトリクスを実装/検証
- [x] `publishedWorldCount` / `retiredWorldCount` を累積定義で担保
- [x] `retirePressure` / `retireDepth` を監査可能にする

## 8. テスト（新規 + 更新 + 回帰）

- [x] Generation rollback 拒否（100→101→100, 100→101→102→101）
- [x] 再 publish 拒否
- [x] invalid graph/routing/execution descriptor 拒否
- [x] WorldId monotonic / Generation monotonic / publicationSequence monotonic
- [x] validation 失敗時 sequence 未確定
- [x] RuntimeWorld immutable（直接/間接）
- [x] RT->Snapshot 到達不可
- [x] observe 長期保持禁止
- [x] RetireCoordinator 以外から free 判定不可
- [x] GracePeriod 判定（single-reader 前提）

## 9. 実装後監査

- [x] grep監査（publish/retire/snapshot/shared_ptr/atomic/free系）を実施
- [x] 否定検索（legacy経路）を実施
- [x] Serena/CodeGraph 再監査で旧経路残存なしを確認
- [x] rev13 受け入れゲート 1〜28 の適合証跡（ファイル/行/シンボル）を作成
- [x] rev13 本文適合表（図/責務/不変条件/禁止事項）を作成

## 10. 完了条件

- [x] 上記タスクがすべてチェック済み
- [x] Debug/Release ビルド成功
- [x] 関連テスト全件 Green
