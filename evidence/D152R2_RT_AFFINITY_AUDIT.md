# RT Affinity Audit — RecoveryLifecycleWord (W) のスレッド接触監査（read-only）

**Date:** 2026-08-31 (+09:00)
**Type:** read-only 実装後ソース監査。**Production source: 0 変更 / Test source: 0 変更 / CMake: 0。**
**目的:** D154-F1 の安全性評価が依存する前提「W access = {CoordinatorLoop, RebuildThread}、ISR/audio thread 非接触」を、実装後の現行ソースで再検証する。
**Baseline:** D154 検証時と同一の worktree（ソース変更 0）。

---

## 0. 判定（先出し）: **PASS — W の触达者は全て NonRT。ISR / audio callback / RT パスからの接触 0 件**

---

## 1. W アクセス点の全列挙（実装後ソース・実測）

`RecoveryLifecycleWord` への load / CAS は **ISRRuntimePublicationCoordinator.{h,cpp} 内のみ**に存在（外部からの直接アクセス 0 — メンバ `lifecycle` は coordinator 内 private 構造）:

| # | 位置 | 関数 | 操作 |
|---|---|---|---|
| A1 | cpp:902/936/951/958/972 | `submitRecoveryRequest`（coalesce + tryInsert 呼び出し元） | load + CAS（coalesce identity CAS） |
| A2 | cpp:1081-1105 | `postRecoveryFailureSignal`（T1） | load + CAS（pending++） |
| A3 | cpp:1121-1150 | `adjudicateRecoveryFailureSignals`（T2） | load + CAS（drain+apply / terminalize 経由 resolve） |
| A4 | cpp:1031-1062 + h:512/520 | `resolveRecoveryObligation` → `table.resolve`（T3 / Completion Authority） | load + CAS（Live→terminal） |
| A5 | cpp:1415-1430 | `casDelivery`（T6/T7 attach primitive） | load + CAS（delivery） |
| A6 | h:437/465/482 | `table.findByKey` / `table.tryInsert` | load + CAS（Live 公開） |
| A7 | cpp:1164/1196/1215/1225 | `peekLifecycleForTest` / table resolve 内部 | load（test-only / A4 内部） |
| A8 | cpp:1463 | `redriveDeferredRecoveryObligations` のスキャン | load |

## 2. 各入口の thread affinity（実測 — enclose 関数 + スレッド起動点で確認）

| 入口 | 呼び出し点 | enclose 関数 | スレッド | 根拠 |
|---|---|---|---|---|
| E1 | RebuildDispatch.cpp:1006/1033/1091/1115（postSignal） | `AudioEngine::rebuildThreadLoop`（:824） | **RebuildThread** | 専用スレッド本体 |
| E2 | Threading.cpp:272（adjudicate）/ 278（redrive）/ 263（processIntent） | `AudioEngine::runCoordinatorPhase`（:258） | **CoordinatorLoop** | `CoordinatorLoop` = `juce::Thread("ConvoPeq.CoordinatorLoop")`（ISRCoordinatorLoop.cpp:8）・run() から runCoordinatorPhase 呼び出し（:39）・ソース明記「Non-RT only (CoordinatorLoop is a juce::Thread, never RT)」 |
| E3 | ProcessIntent.cpp:140/169（submitRecoveryIntent）→ AudioEngine.h:4486（submitRecoveryRequest） | `QuarantineIntentHandler::handle` → `submitRecoveryIntent` | **CoordinatorLoop** | E2 の processIntent phase 内で実行（Threading.cpp:263） |
| E4 | Commit.cpp:782（enqueuePublicationIntentForRuntimeCommit → Orchestrator::submitPublishRequest → trySubmitImpl:311/320 / onPublishCommitted:354 → postSignal/resolve） | `AudioEngine::enqueuePublicationIntentForRuntimeCommit` | **RebuildThread**（Route A） | 呼び出し元は RebuildDispatch.cpp:1051/1131/1337 = rebuildThreadLoop 内 |
| E5 | Orchestrator.cpp:371/401（Route C） | `submitPublishRequest` | **CoordinatorLoop**（receipt 配送 = processIntent 経路）+ RebuildThread（processDeferredAdmission 経由） | RebuildDispatch.cpp:912-919 / Threading.cpp:240/299 の記載と一致 |
| E6 | RebuildDispatch.cpp:810（discardRecoveryRequestsOnShutdown → resolve ShutdownDiscarded） | `AudioEngine::stopRebuildThread` | **join 後の単一スレッド**（shutdown close） | Threading.cpp:249 shutdownCoordinatorLoop 順序 |

## 3. Audio / RT パスの非接触確認（実測 0 件）

- `AudioEngineProcessor::processBlock` → `getNextAudioBlock` → RT 処理系ファイル（`AudioEngine.Processing.AudioBlock.cpp` / `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` / `DSPCoreIO.cpp` / `BlockDouble.cpp` / `DSPCoreLifecycle.cpp` / `AudioEngineProcessor.cpp` / `AudioEngine.Reader.cpp`）を grep:
  - `postRecoveryFailureSignal` / `adjudicateRecoveryFailureSignals` / `resolveRecoveryObligation` / `submitRecoveryRequest` / `recoveryAdmissions_` → **0 件**（HIT_COUNT=0）
  - `recovery` / `RuntimeIntentCoordinator` / `runtimePublicationBridge` を含めても、実コード参照は `DSPCoreLifecycle.cpp:90-96` の `setRetireCoordinator`（**prepareToPlay でのポインタ配線のみ・非 RT セットアップ**）だけ
- `EQProcessor.h:466 m_retireCoordinator` / ConvolverProcessor 同様: ポインタ保持のみ、**deref 呼び出し 0 件**（`retireCoordinator_->` / `retireCoordinator_.` の検索 = 0）— RT callback から coordinator のいかなるメソッドも呼ばれない
- ISR callback / Timer 経路: `submitRecoveryIntent` 等の recovery API 呼び出しは AudioEngine.Timer.cpp に 0 件（旧 timerCallback cadence は CoordinatorLoop に移設済み — Threading.cpp:260 の記載と整合）

## 4. D154-F1 前提の維持判定

```text
W access
  ├─ CoordinatorLoop  (= juce::Thread "ConvoPeq.CoordinatorLoop" — 非RT)
  ├─ RebuildThread    (= rebuildThreadLoop 専用スレッド — 非RT)
  └─ shutdown close   (= join 後の単一スレッド — 非RT)

ISR / audio thread (processBlock → getNextAudioBlock → DSPCore*)
  └─ access 0 件（実測）
```

**D154-F1 の「lock-free ではないが NonRT-only なので許容（D150 §12 既存承認の範囲内）」という結論は、仮定ではなく現行 topology の実測に基づいて維持される。**

**変質リスクの注記**: 将来 `setRetireCoordinator` で配線されたポインタを RT callback 内から deref する変更（eq/convolver からの直接 resolve/postSignal 等）が入った時点で、本前提は崩れる。その場合は D152 §2 の `_InterlockedCompareExchange128` wrapper 案（真の lock-free）への backend 切替が必須（D152-R2 §5 に明記済み）。

---

## 5. 監査手順の記録

- 全 W アクセス点列挙: grep `lifecycle.`（ISRRuntimePublicationCoordinator.{h,cpp}）→ 20 サイト（上記 A1-A8 に整理）
- 外部呼び出し元列挙: grep recovery API 7 種を src 全域（production のみ、tests/coordinator 内部を除外）→ E1-E6 に集約
- enclose 関数特定: awk による直前関数シグネチャ追跡（RebuildDispatch:824 rebuildThreadLoop / Threading:258 runCoordinatorPhase / Commit:782 / Orchestrator:40,343,357）
- スレッド実体: ISRCoordinatorLoop.cpp（juce::Thread 名 "ConvoPeq.CoordinatorLoop"・Non-RT 明記）/ RebuildDispatch.cpp:824
- RT パス非接触: 8 RT 系ファイル grep → recovery API 0 件・retireCoordinator deref 0 件

**判定: RT affinity audit = PASS。T3c close candidate。**
