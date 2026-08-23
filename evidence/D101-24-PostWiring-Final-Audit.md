# D101-24 Post-Wiring Final Audit — Production Wiring 完了確認

Date: 2026-08-24
Baseline: ConvoPeq.md 2026-08-23 23:29:45 生成版
Status: D101-24 RetryScheduler Production Wiring 完了として確定

## 1. Baseline 正当性

- 規約: 以後 ConvoPeq の検討・検証は毎回 `ConvoPeq.md` 最新ソース基準
- 今回基準: 2026-08-23 23:29:45 生成版を `ConvoPeq.md` として扱い、D101-24 実装はこの版の `AudioEngine/RebuildDispatch/CtorDtor` 境界と整合
- `git log / git diff --stat / git status` で wiring 差分のみ存在、他 branch への逸脱なし
- `git diff --check` 0

## 2. Wiring 4 Steps 適合性（現行ソース基準）

| Step | 実装 | 検証 | 結果 |
|------|------|------|------|
| Step1 ownership | `AudioEngine.h:106` forward-decl `RetryScheduler` + `2675` `unique_ptr<RetryScheduler> retryScheduler_` | rg/fdfind/coco/graphify/semble一致。non-owning back-ptr維持 | PASS |
| Step2 DispatchFn | `AudioEngine.CtorDtor.cpp:98` `make_unique<RetryScheduler>([this](req){submitRebuildIntent(...)})` 単一配線。`RetryScheduler.h/.cpp` は DispatchFn 単経路化（engine_ 削除、RETRY_SCHEDULER_WITH_AUDIO_ENGINE 削除） | sg `RetryScheduler` 2 hitsのみ、coco `ccc grep` 1本一致、semble top RetryScheduler.h | PASS |
| Step3 single producer | `AudioEngine.RebuildDispatch.cpp:1178` `retryScheduler_->schedule(req,0ms)` exactly 1。caller-side `BuildErrorPolicy::classifyBuildError` → `RetryDisposition!=NoRetry` 判定、4-field `RetryScheduleRequest` のみ渡す | rg `retryScheduler_->schedule` prod 1 / test 0 本番混入なし、Gate A PASS | PASS |
| Step4 shutdown | `AudioEngine.CtorDtor.cpp:123-127` `if(retryScheduler_) retryScheduler_->shutdown(); → shutdownCoordinatorLoop(); → stopRebuildThread();` 明示順序、member destruction 依存なし、idempotent | rg `retryScheduler_->shutdown` / `shutdownCoordinatorLoop` / `stopRebuildThread` 順序固定、Gate D PASS | PASS |

## 3. Wiring-Specific Gates（D101-24 Step5 定義）

- Gate A producer uniqueness: `retryScheduler_->schedule` production exactly 1 — PASS
- Gate B semantic contamination: `BuildError/RetryDisposition/RecoveryGeneration/epoch/RuntimeWorld/PublicationAdmission/RecoveryEpisode/supersession/obligation` in RetryScheduler.h/.cpp 0 — PASS
- Gate C dispatch boundary: `submitRebuildIntent/requestRebuild/rebuildRequestGeneration` in RetryScheduler.h/.cpp 0、DispatchFn → submitRebuildIntent → requestRebuild → rebuildRequestGeneration++ 単一境界 — PASS
- Gate D shutdown order: `RetryScheduler::shutdown() → Coordinator停止 → RebuildThread停止` — PASS

全ツール横断:

- WSL: rg/ag/fdfind/fzf/sed/awk — 上記 Gates 0/1 hits 完全一致
- ast-grep `sg run -p 'RetryScheduler' / 'retryScheduler_' / 'BuildError'` — 2/1/0 hits で rg と一致
- serena `.serena/project.yml` language_servers 正常
- cocoindex `ccc status` 133k chunks / `ccc grep RetryScheduler` hits=RetryScheduler.h/.cpp/CtorDtor/AudioEngine.h/RebuildDispatch のみ / `BuildErrorPolicy` 集約確認
- graphify `query RetryScheduler` 1 node / `query BuildError` Not found pre-wiring と整合
- semble `search RetryScheduler` top RetryScheduler.h / `search BuildError` policy 側のみ
- AiDex `.aidex` index 26M 維持、代替 rg で contamination 0 確認

## 4. ビルド/テスト Gate

- Debug Build (Ninja Multi-Config, MSVC 19.51 + oneAPI MKL): 12/12 target 成功（前回 Step1 では 513 コマンドフルビルド、今回は増分 12 で収束）
- Debug CTest: 37/37 PASS（RetrySchedulerTests 0.72s 含む、Total 34.99s）
- Release は未実行だが wiring は Debug と同一起源、追加差分なし

## 5. 要調査・棚卸し・保留事項の確定

| 項目 | 従前状態 | 今回確定 |
|------|----------|----------|
| RetryScheduler が BuildError/RetryDisposition を知るべきか | 未確定（D-5-2 Step2 で policy 非侵入方針） | 知らないまま確定。caller-side `classifyBuildError` を RebuildDispatch で実行、scheduler は 4-field のみ |
| PublicationAdmission/RuntimeWorld/RecoveryEpisode を scheduler が触るべきか | 保留（D101-23 Gate 3） | 触れないで確定。DispatchFn → submitRebuildIntent 境界を維持 |
| ownership を AudioEngine 以外（Coordinator/Orchestrator）に置く案 | 未確定 | AudioEngine unique_ptr で確定。Coordinator は Retire/Publication、Orchestrator は Publish semantic と分離 |
| backoff/queue capacity 変更 | 保留 | 本 Step では変更しないで確定（D101-24 今回やらないこと 10項目維持） |
| D14/D15 obligation/backpressure への影響 | 要調査 | 変更なしで確定。RetryScheduler wiring を理由に recovery semantics を変更しない |
| ConvoPeq.md の正本 | 規約 | 2026-08-23 23:29:45 版を正本として確定、以後毎回この運用を継続 |

未確定・保留は上記で全て確定。文献検索（BuildError/RetryScheduler 概念）は現行 `BuildErrorPolicy.h` / `RuntimeBuilder.h` の 8値分類・3値 disposition が既に executable contract として固定されており、外部文献追加の必要なし。

## 6. 次段計画

- 今回で D101-24 は close。本 wiring を基盤に今後必要になれば W1-W6（ownership/warmup→schedule/request integrity/no leakage/shutdown suppression/idempotence）の formal wiring tests を追加可能だが、現行 37/37 PASS で即時追加は不要
- 次回検討時も必ず ConvoPeq.md 最新版（次回生成時）を再取得して基準とする
- Release CTest は次回フルゲート時に実施

## 結論

D101-24 Production Wiring は 4 Gates 全 PASS、Debug Build/CTest 37/37、semantic contamination 0、git diff 正常で完了扱いとして確定する。
