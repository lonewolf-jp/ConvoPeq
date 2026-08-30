# D135-8/9 Gate C — CTest Audit (Debug + Release)

Date: 2026-08-30
Scope: post-Gate-B working tree, unchanged. **Test execution only — 0 production/test source changes, 0 fixes applied.**
Wrapper: `tools/gatec_ctest.bat` (vcvarsall x64 + oneAPI setvars → `ctest -C <cfg> --output-on-failure`; **all 40 registered tests** — the 2 standard-exclusion tests were also run and classified separately).
ctest 4.4.3. Logs: `evidence/D135-8-9_GATE_C_DEBUG_CTEST_LOG.txt` / `evidence/D135-8-9_GATE_C_RELEASE_CTEST_LOG.txt`.

## Verdict: **Gate C = FAIL（即停止）**

判定表適用: Debug 40/40 PASS、Release で **D135-8/9 関連テスト `AudioEngineHarness` が FAIL**（決定論的・3/3 再現）。
→ 「D135-8/9関連テストFAIL = Gate C FAIL、即停止」行を適用。原因修正は実施していない。

## Results

| Config | Result | Detail |
| --- | --- | --- |
| Debug | **40/40 PASS**（exit 0, 0 failed, 0 not run） | 標準除外対象の #33/#37 も含め全PASS。所要: 最大 AudioEngineHarness 17.99s |
| Release | **39/40 PASS, 1 FAIL**（ctest exit 8） | FAIL: **#40 AudioEngineHarness**（60.81 sec） |

Release PASS の中には D135-8/9 関連テストを含む: `RuntimeHealthMonitorTierTests`(0.05s), `RetrySchedulerTests`(0.60s),
`DeferredDeletionQueueReclaimTests`(3.07s), `RuntimePublicationCoordinatorRejects`, `RebuildAdmissionRegression`,
`ShutdownRetireIntentDrain`, `PublicationAdmissionTests` — 全て PASS。

## 失敗記録（Gate C 指示項目: テスト名・exit code・ログ・失敗箇所）

- **失敗テスト**: ctest Test #40 `AudioEngineHarness` → 内部 unit test **`testDeferredBacklogDrainsCompletely`**
  （`src/tests/AudioEngineHarness/DeferredFlowIntegrationTests.cpp:122`）
- **exit code**: ctest経由 = **8**（ハーネス内部の複数サブチェック失敗を反映する戻り値）。
  直接実行（`build/Release/AudioEngineHarness.exe`）= **1** ×3回（`runDeferredFlowIntegrationTests` が最初の失敗で即 return 1）。
- **失敗箇所**: `DeferredFlowIntegrationTests.cpp:138`
  `std::fprintf(stderr, "FAIL: cycle %d did not defer\n", cycle);` — **`FAIL: cycle 1 did not defer`**
- **失敗様相**: `testDeferredBacklogDrainsCompletely` の 2サイクルのうち **cycle 0 は成功**（defer 観測 → fade 解除 → backlog 完全排出）。
  **cycle 1 は `setFadingRuntimePresent(true)` + `requestRebuild(Structural)` 後、`orch.hasDeferredRequest()` が 45 秒以内に一度も true にならずタイムアウト。**
  すなわち cycle 1 の publish が一切 Deferred にならず、enqueueDeferred が呼ばれていない。
- **再現性**: Release 直接実行 **3/3 同一失敗（決定論的、フレークではない）**。Debug は PASS。
- **ログ**: `evidence/D135-8-9_GATE_C_AudioEngineHarness_release_direct.log`（直接実行キャプチャ）+
  Release CTest ログ（ctest は GUI サブシステム exe のため失敗時出力をキャプチャできず — 96行で終端）。

## 切り分けで確認済みの事実（読み取り専用）

1. **kMax=2 跨ぎサイクル累積説は不成立**: generation は `generation = ++rebuildRequestGeneration`
   （RebuildDispatch.cpp:655）でリビルド要求ごとに新規採番 → cycle 1 の最初の defer は
   `sameObligation=false` → count リセットで通過する（Orch.cpp:470-503 の discard は発火しない）。
2. **admission の hasFading 分岐**: PublicationAdmission.cpp:51-58
   `testFadingRuntimePresent() || hasFadingRuntimeInWorld(...)` → `DeferredFadingActive`。
   テストフックは atomic release-store（DeferredPublicationTestAccess.h:36）/ acquire-read（AudioEngine.h:3678-3681）
   で可視性は確保済み。かつ cycle 0 は Release でも defer に成功 → フック自体は Release で有効。
3. **requestRebuild には重複抑制ガードが存在**: RebuildDispatch.cpp:623-660
   （`hasPendingTask` + `sameAsPending` → `blockedAsDuplicate` → タスク棄却・generation 不採番）。
   cycle 1 でこれが発火した場合「タスクが走らない → defer しない」という症状と整合するが、
   発火条件（hasPendingTask 残存）は cycle 0 完了後には成立していないはずで、未確定。
4. **Diagnostic ビルド未実施**: `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` のため cycle 1 の
   submit/admission 経路の実トレースは不可能（Gate C はテスト実行のみで、診断ビルドは行っていない）。

## 未確定事項・棚卸し（次フェーズへの引き継ぎ）

- **[要調査] cycle 1 の defer 不発生の根本原因**: 候補は (a) requestRebuild 重複抑制によるタスク棄却、
  (b) admission が Deferred 以外（Ready / RejectedPressure など）を返した、(c) RebuildThread 側の滞留。
  `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` ビルド + `[D135] re-defer` / `[DIAG] requestRebuild` ログ取得で
  特定可能。**Gate C では未実施（修正・再構成は行っていない）。**
- **[未確定] ベースライン比較**: HEAD a65ace1（D135 作業前）の Release AudioEngineHarness が PASS するかは未検証
  （別 worktree ビルドが必要。Gate C の制約上、実施していない）。Debug は D135 作業ツリーで PASS している点に注意。
- **[観測] ctest exit 8 vs 直接実行 exit 1**: ctest 実行ではハーネス内部でより多くのサブチェックが失敗したことを
  示唆するが、ハーネスが GUI サブシステムのため ctest 側に出力が残らない。直接実行の完全出力取得方法の確立が課題。
- **[既知] 標準 CTest 手順は Debug のみ**: `tools/run-ctest-full.bat` は `-C Debug` のみで、Release CTest は
  標準手順に含まれない。Release の `AudioEngineHarness` が過去に PASS した記録は evidence/ に確認できていない。

## Static regression（実行後 working tree）— **3/3 PASS**

1. **production/test source 変更なし**: `git status --porcelain src/` = Gate B 時と同一の 18 ファイル
   （audioengine 16 + Main/MainWindow + DeferredFlowIntegrationTests.cpp）。Gate C 実行中の新規変更は 0。
2. **Gate A 修正の維持**: `RuntimePublicationOrchestrator.cpp:580` = `convo::publishAtomic(deferredClearRequested_, true, std::memory_order_release);` — 維持確認。
3. **raw atomic access の復活なし**: `deferredClearRequested_.(store|load|exchange|compare_exchange|fetch_*)` = 0件
   （ラッチは :580 publishAtomic / :591 exchangeAtomic のみ）。

## Artifacts

- `evidence/D135-8-9_GATE_C_DEBUG_CTEST_LOG.txt`（全40テスト個別結果付き）
- `evidence/D135-8-9_GATE_C_RELEASE_CTEST_LOG.txt`
- `evidence/D135-8-9_GATE_C_AudioEngineHarness_release_direct.log`（Release ハーネス直接実行キャプチャ）
- `tools/gatec_ctest.bat`（CTest 実行ラッパーのみ。ソース/テストは不変）
