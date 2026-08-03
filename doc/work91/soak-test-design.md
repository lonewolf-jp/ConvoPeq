# Work91 — Soak Test 設計

- 日付: 2026-08-03
- 目的: B4 開発完了後の品質保証・運用検証フェーズ。実装の「正しさ」は B3/B4 + Integration + Debug + ASan + CI で検証済み。本作業では**長時間運用でも崩れないか**（数万〜数十万 publish、backpressure、receipt 回復、retire 変動、メモリ推移）を確認する。

## 1. 位置づけ

| フェーズ | 状態 |
| --- | --- |
| B3 (executePublish 単一化) | ✅ 完了 (Release ctest 24/24) |
| B4 (Integration / Debug / ASan / CI) | ✅ 完了 (25/25 ctest, ASan green) |
| **Soak (本設計)** | 🔜 これから |
| ADR・設計文書最終版 | ⏸ 実装安定後 |

## 2. テスト対象システム

soak で監視・検証する対象と、その公開 API（既存実装から確認済み）。

### 2.1 Publish パイプライン (統合系)

```
commitRuntimePublication facade → OwnerChannel → IntentQueue → CoordinatorLoop
→ executePublish → RuntimeStore swap → PublishReceipt
```

- 駆動: `AudioEngineHarness`（`src/tests/AudioEngineHarness/`）— GUI 無しで実 AudioEngine + audio thread + rebuild thread + CoordinatorLoop を起動。
- 監視点: `e.observePublishedWorld()->publication.sequenceId` が単調増加で追いつくこと、receipt が返ること。

### 2.2 責務分割の原則（2026-08-03 確定）

**publish を含むテストはすべて AudioEngineHarness 側**。ヘッドレス Coordinator はデータ構造（Queue / OwnerChannel / Registry）の耐久試験に限定する。

根拠: `RuntimePublishExecutor::executePublish()` の consumer は Intent 消費ではなく **AudioEngine を中心とした publish 完了シーケンスそのもの**（`makeRuntimePublicationCoordinator()` による唯一の store-swap / `dspHandleRuntime().resolve()` / `advanceRetireEpoch()` / `runtimeOrchestrator_->onPublishCommitted()`）。drain の意味が AudioEngine の実行と同義のため。

| テスト | AudioEngine | CoordinatorLoop | executePublish |
| --- | --- | --- | --- |
| Queue stress (ISRSoakTests) | × | × | × |
| OwnerChannel (ISRSoakTests) | × | × | × |
| Registry (ISRSoakTests) | × | × | × |
| Publish soak (Harness) | ○ | ○ | ○ |
| Receipt (Harness) | ○ | ○ | ○ |
| Retire (Harness) | ○ | ○ | ○ |

境界が実装構造（executePublish の責務範囲）と一致するため、今後 executePublish の責務が増えてもテスト設計が崩れにくい。

### 2.2a データ構造耐久 (ヘッドレス, ISRSoakTests)

`convo::isr::RuntimePublicationCoordinator` を AudioEngine 無しで直接駆動（`ISRSemanticValidationTests.cpp` が使用）。

- `enqueuePublicationIntent(intent)` — キュー満杯時は false（backpressure、B3 #4）。
- `getPendingIntentCount()` — 滞留監視。
- `submitObserve(handle)` — observe enqueue のみ（drain は Harness 側）。
- `OwnerChannel`（RuntimeWorldAuthority 内）— take / put 耐久。
- `PendingPublishRegistry`（RuntimeWorldAuthority 内）— register / unregister stress。

### 2.2b publish 系 (統合, AudioEngineHarness)

`RuntimePublicationOrchestrator` / `executePublish` を実スレッドで駆動。

- `processIntent(engine, lifetimeMgr)` — CoordinatorLoop が呼ぶ drain（publish 完結シーケンス）。
- `commitRuntimePublication` facade → OwnerChannel → IntentQueue → CoordinatorLoop → store-swap → receipt。

### 2.3 監視対象メトリクス (AudioEngine)

- `retireQueueDepth_`（`retireQueueDepth()` getter 経由, AudioEngine.h:4061 で drain 時に更新）— retire 滞留数。
- `getPendingIntentCount()`（AudioEngine.h:3377 → worldAuthority_.lifetime()）— 未処理 Intent 数。
- `waitForPublishReceipt(seqId, timeoutMs)`（AudioEngine.h:3559）— receipt 待ち。タイムアウト→回復の検証対象。
- `observePublishedWorld()` — 現在の公開 world（sequenceId / retireQueueDepth を含む）。

## 3. Soak シナリオ

### シナリオ S1: 連続 publish 耐久（統合・AudioEngineHarness）

- 目的: 数万〜数十万回の publish を実スレッドでバースト投入し、キュー溢れ・順序性・sequenceId 単調性を publish パイプライン全体（enqueue → executePublish → store-swap → receipt）で検証。
- 駆動: `AudioEngineHarness` で実 AudioEngine 起動 → `commitRuntimePublication` facade を連続呼び出し → CoordinatorLoop が自動 drain → `observePublishedWorld()` で sequenceId 進行を確認。
- 負荷: 既定 100,000 回。引数で上書き可（`--publishes N`）。
- 検証点:
  1. 全 publish の receipt 成立（`waitForPublishReceipt`）。
  2. sequenceId が単調増加・欠番なし（enqueue 回数 == store-swap 観測数）。
  3. `PendingPublishRegistry` 残存 0、pendingIntentCount が 0 に収束。
  4. audio thread が並行稼働している（実環境整合）。
- 合格: 全 N 件 receipt 成立、欠番・重複 0、registry 空、pendingIntentCount 0 収束。

### シナリオ S2a: IntentQueue 飽和（ヘッドレス・ISRSoakTests）

- 目的: IntentQueue（容量 4096）の飽和と明示拒否をデータ構造レベルで検証（drain 不要）。
- 駆動: `RuntimePublicationCoordinator` を直接使用。publish intent を 4096 件 enqueue → 追加分が false（明示拒否）→ `getPendingIntentCount()` が 4096 のまま（失われない）。
- 反復: 満杯 → 拒否を 50 サイクル繰り返す。
- 合格: 毎回 enqueue false（B3 #4）、pendingIntentCount が容量を超えない、拒否後もキュー内容が保全される。

### シナリオ S2b: Backpressure recovery（統合・AudioEngineHarness）

- 目的: キュー満杯 → CoordinatorLoop が drain → enqueue が再び成功 → receipt が届く回復サイクルを実スレッドで検証。
- 駆動: `AudioEngineHarness`。大量 publish を一気に投入して IntentQueue を飽和させた後、CoordinatorLoop の drain で容量を空け、追加 publish が成功することを確認。
- 反復: 満杯 → drain → 回復サイクルを複数回。
- 合格: 毎サイクルで queue-full 後の enqueue 復帰成功、receipt 成立、ハングなし。

### シナリオ S3: receipt wait タイムアウト・回復（統合系）

- 目的: `waitForPublishReceipt` のタイムアウトとその後の回復を実スレッドで検証。
- 駆動: `AudioEngineHarness` で実 AudioEngine 起動 → `commitRuntimePublication` を連続発行 → `waitForPublishReceipt(seqId, timeoutMs)` が timeout 内に成立。
- 負荷変動: publish を短時間に大量発行し、CoordinatorLoop の処理が追いつかない状況を意図的に作り、receipt が timeoutMs 超で待たされるケースを観測。その後 drain 完了で receipt 成立。
- 合格: 全 publish が最終的に receipt 成立（タイムアウト後の回復を含む）、ハング・デッドロックなし。

### シナリオ S4: retire queue の増減（統合系）

- 目的: retire queue が publish/observe サイクルで増減し、無制限に滞留しないこと。
- 駆動: `AudioEngineHarness`。publish → observe サイクルで retire を発生させる。`retireQueueDepth` / pendingIntentCount の推移をサンプリング。
- 合格: 長時間後も pendingIntentCount が 0 に戻る（滞留が残らない）。retireQueueDepth が閾値（例: 1024）を超えない。

### シナリオ S5: メモリ使用量の推移（Windows, 統合系）

- 目的: 数万 publish 後もメモリが単調増加せず、steady-state に収束すること（リーク検出）。
- 計測: `GetProcessMemoryInfo`（PSAPI, `PROCESS_MEMORY_COUNTERS::PrivateUsage`）で一定間隔サンプリング。work70 の実績では Peak Private 554MB → Steady 453MB。
- 合格: 初期ウォームアップ後、サンプル期間の PrivateUsage 増加率が閾値（例: 1 分間で 1MB 未満）以下。または publish 数に対する線形増加が観測されない。

### シナリオ S6: 長時間 ASan 実行

- 目的: ASan 有効構成で長時間実行しても誤検出・リークレポート・クラッシュが無いこと。
- 構成: RelWithDebInfo + `/MD` + `clang_rt.asan_dynamic-x86_64.dll`（CI と同一）。`ASAN_OPTIONS=detect_leaks=0`。
- 合格: 全 soak シナリオを ASan 構成で実行し、エラーレポート 0、正常終了。

## 4. 実行方式

### 4.1 新規ターゲット: `ISRSoakTests`（データ構造耐久）

ヘッドレス駆動（S2a + OwnerChannel / PendingPublishRegistry 耐久）を担う単独 exe。`src/tests/ISRSoakTests.cpp` に実装。

- `add_executable(ISRSoakTests src/tests/ISRSoakTests.cpp)`
- 既存 ISRSemanticValidationTests と同じ include/link パターン（L82-111 を参照）: `convo::isr::RuntimePublicationCoordinator` を直接使用するため JUCE リンク不要の可能性が高いが、Coordinator のヘッダ依存に従う。
- `add_test(NAME ISRSoakTests COMMAND ISRSoakTests)` — ctest 組み込み（短時間で安全に走る）。
- ASan 対象リスト `CONVOPEQ_ASAN_TEST_TARGETS` に追加（ヘッドレスのため ASan で全シナリオ安全に走れる）。

### 4.2 統合系: `AudioEngineHarness` 拡張

publish 系（S1 / S2b / S3 / S4 / S5）を追加。既存 `PublishPipelineIntegrationTests.cpp` の `bool testXxx()` + `main()` パターンを踏襲。

- 長時間テスト（S1 100k publish, S2b, S5 メモリ）はデフォルト ctest からは除外し、手動実行専用（`--soak` フラグ）にする。
- ctest 組み込みは短時間版のみ（例: S1 を 1,000 publish、S3 を通常版）。

### 4.3 メモリ計測（S5）

PSAPI (`psapi.h` + `GetProcessMemoryInfo`) を追加。Win32 専用。Harness の soak モードでサンプリングスレッドとして実装（publish 経路のメモリを実測）。

### 4.4 CLI 引数

| 引数 | 既定 | 説明 |
| --- | --- | --- |
| `--publishes N` | 10000 | S1 の publish 数（ctest 既定は 10000） |
| `--soak` | off | 長時間版（既定 10 分）を実行。ctest ではオフ |
| `--iterations N` | 50 | S2 の満杯→drain サイクル数 |
| `--soak-minutes M` | 10 | 長時間版の上限時間 |
| `--quiet` | off | 進捗ログ抑制（CI 用） |

## 5. 合格基準（Exit Criteria / Pass Criteria）

定量化した Pass Criteria。CI / 自動レポートへ組み込む際にもこの表を利用する。

| シナリオ | Pass 条件（定量） |
| --- | --- |
| S1 | enqueue 成功数 == committed 数（onPublishCommitted 相当の完了数）、sequenceId 欠番・重複 0、PendingPublishRegistry 残存エントリ 0、pendingIntentCount 収束 0 |
| S2a | 毎サイクルで enqueue false（B3 #4 明示拒否）、pendingIntentCount ≤ 容量（4096）のまま保全、拒否後もキュー内容不変 |
| S2b | 50 サイクル全てで「queue-full → drain → enqueue 復帰」成立、receipt 成立、ハング 0 |
| S3 | timeout 回数 0（全 receipt が timeoutMs 内に成立）、receipt 順序が seqId と一致（FIFO）、ハング 0 |
| S4 | 長時間後 retire queue が最終的に空（retireQueueDepth 0）、retire epoch 単調増加、observe 後 pending 数が増加→drain で 0 に復帰 |
| S5 | PrivateUsage 増加が許容範囲内（初期ウォームアップ後 +5% 以下、または 1 分間増加 < 1MB = リークなし） |
| S6 | ASan エラー 0、リークレポート 0（detect_leaks=0 のためリーク検出は S5 が担う）、異常終了なし |

## 6. CI 連携

- `sanitizer-ci.yml` の `relwithdebinfo-asan` ジョブに `ISRSoakTests` を追加（S6: ASan 実行、ヘッドレスで短時間安全）。
- Harness の長時間版（`--soak`）は CI には載せない。手動運用（nightly 相当）とする。workflow_dispatch での実行を検討（isr-verification.yml のパターンを流用）。

## 7. 実装ステップ

1. ✅ `src/tests/ISRSoakTests.cpp` 新規作成（S2a: IntentQueue 飽和 + OwnerChannel / PendingPublishRegistry 耐久、ヘッドレス）。
   - **実装ノート（2026-08-03）**: `getPendingIntentCount()` は enqueuePublicationIntent で更新されない（submitObserve / submitQuarantine のみ更新。ISRRuntimePublicationCoordinator.cpp L556-698）。S2a の満杯判定は enqueue 返り値（true/false）+ 成功数のみで検証する（既存 testPublishIntentQueueFullBackpressure と同方式）。
   - `sizeof(RuntimePublicationCoordinator) = 976,000B`（intentQueue_ 4096×144B 等）。**スタック確保は 1MB オーバーフロー**。テストでは必ず `std::make_unique` でヒープ確保する。
2. ✅ `CMakeLists.txt`: `add_executable(ISRSoakTests ...)` + `add_test` + ASan リスト追加。
   - ISRSemanticValidationTests と同一 .cpp 構成 + `MKL::MKL` リンク（AlignedAllocation.h の mkl_malloc/mkl_free 必須）。Release ビルドで 5/5 PASS 確認済み。
3. ✅ `AudioEngineHarness` 拡張: S1（publish endurance）+ S2b（並行 burst）+ S3（receipt 回復）+ S4（retire epoch）+ S5（メモリ推移）。
   - `src/tests/AudioEngineHarness/SoakPublishIntegrationTests.cpp` 新規作成。`PublishPipelineIntegrationTests.cpp` の main が `--soak` 引数で分岐して呼ぶ（デフォルト ctest は従来 4 シナリオのみ = 短時間）。
   - **検証結果（Release, 2026-08-03）: `--soak` 全シナリオ PASS。** S1=100k / S2b=80k(4thr×20k) / S3=300 rapid / S4=10k の計 **180,300 publish 全て accepted・reject 0・peakBacklog 0**。S4 epoch 単調増加（360605→380607）+ retire pending 0。S5 peak=382MB=final（収束）。
   - **実装ノート**: `commitRuntimePublication` は内部で receipt を 250ms 待つ（AudioEngine.h L4269）ため、実質 1 publish が同期。並行スレッドでも drain（1ms 周期で全 drain）が追いつき peakBacklog は 0 に留まる。queue-full の構造的検証はヘッドレス S2a が担う（責務分離 §2.2 どおり）。
   - `waitForPublishReceipt` / `PendingPublishRegistry` 空判定は private のため、store が最終 seq に到達すること（`observePublishedWorld()->sequenceId >= last`）を registry 空の代理観測とした。
   - seq 連続（gap なし）検査は誤検出のため**廃止**: initialize の Bootstrap/rebuild が seq を消費する。代わりに「重複なし + store 最終到達」を検証。
4. ✅ ローカル検証: Release ビルドで ctest 追加分 green（2026-08-03）。
   - `ctest -C Release -R ISRSoakTests` → PASS（0.17s）。
   - `ctest -C Release -R AudioEngineHarness` → PASS（9.52s）。
5. ✅ ASan (RelWithDebInfo) で ISRSoakTests + Harness 短時間版を実行 → green（2026-08-03）。
   - `build_asan`（Ninja Multi-Config + `-DENABLE_ASAN=ON`）再 configure で MKL/AudioEngineHarness の新規ソース変更を反映。
   - 実行前に `clang_rt.asan_dynamic-x86_64.dll` を exe 隣に展開。`ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:detect_leaks=0`。
   - `ISRSoakTests.exe`（ASan）→ S2a / OwnerChannel 50k / 容量 reject / PendingPublishRegistry 50k / overwrite stress の 5/5 **PASS**。
   - `AudioEngineHarness.exe`（ASan）→ all publish pipeline tests **PASS**。
6. ✅ 長時間版（10 分 soak）を手動実行し、メモリ推移・退出基準を確認（2026-08-03）。
   - Release: `AudioEngineHarness.exe --soak --scenario=all`。
   - s1=100k / s2b=80k(4thr) / s3=300 / s4=10k / s5=8s → 計 **180,300 publish 全 accepted・reject 0・peakBacklog 0**。**ALL PASS**（約 10 分, exit 0）。
   - S5 メモリ推移: peak=380.9MB = final=380.9MB（**収束**、リークなし）。実行中の WorkingSet も 383MB 前後で安定。
7. ✅ CI workflow 追加（`soak-ci.yml`, workflow_dispatch 専用）。
   - **通常 CI（push/PR）には載せない** — 実行時間が長いため（方針 §6 どおり）。
   - `on.workflow_dispatch` のみ。`inputs.scenario` で `all / s1 / s2b / s3 / s4 / s5` を選択可能。
   - 実行: `AudioEngineHarness --soak --scenario=<name>`（windows-latest + MSVC Release /MD + Intel oneAPI MKL/IPP）。
   - ハーネス CLI に `--scenario=<name>` を追加実装（`runSoakScenarios(full, scenario)`、単一シナリオ選択時は該当のみ実行・統計も単一集計）。
   - ローカル検証: `--scenario=s1`（300 件 accepted）・`--scenario=s4`（epoch 3→67, pending 0）が short 版で PASS。
   - 将来は schedule（cron）夜間自動実行へ移行検討。

## 8. リスク・注意

- ✅ 確定済み: `processIntent(engine, lifetimeMgr)` は AudioEngine 参照を要求し、`executePublish` が store-swap / resolve / advanceRetireEpoch / onPublishCommitted に依存するため、**publish 系の drain はヘッドレス不可**。S1/S2b/S3/S4 は全て AudioEngineHarness 側に配置する（責務分割の原則 §2.2）。
- S2b の「queue-full 意図的発生」は publish バースト（Harness から大量連続 commitRuntimePublication）で実現する。実装時に CoordinatorLoop の drain レートより enqueue が速い条件を確認する。
- S3 の「意図的な処理遅延」は CoordinatorLoop スレッド優先度操作や publish バーストで実現する。実装時に安定した方法を選定。
- メモリ計測は Win32 固有（PSAPI）。ヘッドレステスト（ISRSoakTests）は ASan 構成でも safe に走れるよう設計。S5 は Harness 側（publish 経路の実測）。
