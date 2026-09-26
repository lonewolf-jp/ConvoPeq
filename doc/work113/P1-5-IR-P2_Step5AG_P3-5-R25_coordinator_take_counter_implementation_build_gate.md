# P1-5-IR-P2 — Step 5-AG / P3-5-R25: Coordinator Take Counter Implementation / Build Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R25）
- **判定**: **R25-B**。Coordinator take 境界そのものは一意に特定したが、
  当該境界が複数経路を混在させるため、main-site-only の 1-writer 実装は承認しない。
  実装・build・run なし。本 Step で停止する。
- **方法**: 最新ソース再トレース＋R21/R22 先例対比＋R23 実測裏付け。read-only 監査。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（git rev-parse 一致）
kP15FullMatrix = true（P1PolyphaseGainCharacterization.cpp:43・1件一意）
CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（CMakeLists.txt:133 option既定OFF・cache実値OFF維持）
R16 take＋R19 B2＋R22 commit＋R10 accessor＋T6/T9＋R23 delta観測 保持
R24 design（Step5AF・read-only・source無変更）
ConvoPeq.md fresh（2026-09-23 21:32:56・5,521,528 B・最新src編集 20:40 より後）
working tree = R23 vehicle（AudioEngine.h +78・RebuildDispatch.cpp +14・
  P1PolyphaseGainCharacterization.cpp +594・いずれも承認範囲内・§2）
coordinatorTakeCount_／getCoordinatorTakeCount 出現数 = 0（未実装・§7）
```

- R24文書だけではなく最新ソースで所有権移動境界を再確認した（§3–§5）。
- R23 vehicle を revert していない。production 差分は R16/R19/R22 の counter 群のみ。

## 2. 既存 counter topology（R23 vehicle・live source 再確認）

```text
rebuildTakeCount_              宣言 AudioEngine.h:5052／getter :2072／writer RebuildDispatch.cpp:911
rebuildBuildResultCount_       宣言 AudioEngine.h:5055／getter :2078／writer RebuildDispatch.cpp:1235
rebuildCommitEnqueueCount_     宣言 AudioEngine.h:5059／getter :2084／writer RebuildDispatch.cpp:1420（main-siteのみ）
emitQDelta                     req／que／dup／take／bld／cmt／drp／seq／blo（P1PolyphaseGainCharacterization.cpp:472-486）
```

- writer は各1・順序 take→B2→commit→sequence。R18／R21／R22 監査内容と一致。
- R22 の main-site-only は「共有関数内ではなく caller 側（:1420）に置く」ことで達成した
  （R21 §7決定・recovery :1085／:1165 を含めない）。R25 も同形を狙ったが不成立（§5）。

## 3. Source trace（最新ソース基準・R24 §3 の再確認＋精密化）

```text
main-site commit enqueue（RebuildDispatch.cpp:1420・cmt++）
        ↓
enqueuePublicationIntentForRuntimeCommit（Commit.cpp:800）
  │ currentBuildSnapshot_ 更新（Mutex・:817）
  │ handle 事前登録（registerDSPHandleForRuntime・:822）
  │ newDSP==nullptr 時はここで return（:808・submitPublishRequest 未到達）
        ↓（同期直接呼出し・中間queueなし・:878）
runtimeOrchestrator_->submitPublishRequest(req)（Orchestrator.cpp:358）
        ↓
trySubmitImpl（Orchestrator.cpp:41）
  │ admission_.evaluate（:47・Accepted以外は return・DeferredはdeferredSlot_へ・:375-460）
  │ Accepted → world build → PublicationExecutor::publish → publishImpl
        ↓
commitRuntimePublication → enqueueRuntimePublicationFireAndForget（AudioEngine.h:4854-4894）
  │ ownerChannel_.enqueue（world所有権のproducer側移譲・:4854）
  │ runtimePublicationBridge_.enqueuePublicationIntent(intent)（:4873・唯一のproduction caller）
  │   └─> intentQueue_.push（MPSC共通queue・fire-and-forget・所有権移譲済み）
        ↓（非同期・CoordinatorLoop 1ms tick）
RuntimeIntentCoordinator::processIntent（ProcessIntent.cpp:10）
  │ while (intentQueue_.pop(commonIntent))（:47・★★ Coordinator take境界・唯一のconsumer）
  │ type分岐でresidency減算（Publish→publicationIntentResidencyCount_--・:55-56）
  │ DispatchTable → PublishIntentHandler → PublishExecutor::executePublish
        ↓
ownerChannel_.take（RuntimePublishExecutor.h:31・world所有権のconsumer側取得）
authority.publish → commit → sequence bump（Commit.cpp:402 lastCommittedPublicationSequence_）
onPublishCommitted → notifyPublishReceipt（receipt配送）
```

- R24 の「internal queue」は intentQueue_（＋ownerChannel_）に確定。
- 「Coordinator intent take」は `intentQueue_.pop` 成功（Publish 型）に確定（一意）。
- 問題は専用性である（§4–§5）。

## 4. 混在の直接証拠（3点・いずれも live source）

### (a) 三経路集約の自己文書化（ISRRuntimePublicationCoordinator.h:784-786）

```cpp
//   全 3 enqueue 経路（通常 rebuild / Recovery publish / deferred 再 enqueue）がここに
//   集約されるため、本 counter は単一箇所で reservation され二重計上されない（§6.5）。
```

- `enqueuePublicationIntent` が Publish intent の単一 funnel であり、
  source 自身が通常 rebuild・Recovery publish・deferred 再 enqueue の
  3起源を明記している。consumer 側の pop も同一 funnel の出口であり、
  起源別の consume 地点は存在しない。

### (b) 起源消去（AudioEngine.h:4872＋Orchestrator.cpp:320-321）

```cpp
intent.payload.publish.recoveryObligationId = 0;   // ★ D105-R5-8: Route B (non-recovery) ⇒ no obligation
```

- fire-and-forget 経路では recoveryObligationId を常に 0 で固定する。
  trySubmitImpl は req.recoveryObligationId を publishImpl へ渡さない
  （`executor_.publish(engine_, frozen, req.newDSP, oldHandle)`）ため、
  enqueue 時点で既に起源情報は失われている。
- recovery obligation の解決は trySubmitImpl 内で同期実行される
  （Route A completion authority・Orchestrator.cpp:321
  `resolveRecoveryObligation(req.recoveryObligationId, Published)`）。
  したがって pop 時点の intent payload に main-site／recovery の区別は残らない。
- `boundary` は NonRTWorld 固定（:4870）、`decision` snapshot に起源 field なし、
  generation による対応付けは per-task linkage 禁止（R24 §7遵守・R15既知制限）に抵触するため不可。

### (c) R23 実測裏付け（Step5AE §10）

```text
gap→pair2-pre： seq+3 > cmt+1 ＝ main-site 以外の publish 経路（recovery／idle 等）の存在と整合
```

- 混在は仮説ではなく、R23 vehicle で実際に観測された non-main-site publish により裏付けられる。
  pop counter はこれらを数える（cmt に対応する coord がない take が発生する）。

## 5. R25 §5 絶対条件の判定（FAIL → R25-B）

```text
要求： coordinatorTakeCount_ declaration=1／getter=1／writer=1
      ＋ recovery enqueue／other publication path／other Coordinator task を数えない
結果： writer 候補は processIntent pop（:47）に一意に定まるが、
      Publish 型フィルタを掛けても通常 rebuild＋Recovery publish＋deferred再enqueue を混在計数する。
      main-site-only フィルタは code 上存在しない（§4b）。
      → 「main-site counterだからCoordinator側でもmain-siteだけ」は code 確認により否定された。
      → R25 §5 の絶対条件を満たす 1-writer は存在しない。
```

- R21/R22 との決定的差異：cmt は caller 側配置（:1420）で main-site-only を達成できた。
  coord 側に対応する caller 側配置は存在しない（consumer は単一 pop loop のみ）。
- producer 側代替案の検討（いずれも棄却・参考記録）：
  - `submitPublishRequest` 入口＋`recoveryObligationId==0` フィルタ
    ＝ main-site-only は code 確認可能だが、cmt（:1420）との間に queue がなく同期直結のため
    coordΔ==cmtΔ が恒等的に成立し、C-1／C-2 分離（R26目的）が不能。計装として vacuous。
  - `trySubmitImpl` の admission Accepted 後＋同フィルタ
    ＝ R24 の概念A（Coordinator到達）ではなく概念B（Admission受理）を数えることになり、
    A/B 混同（R24 §4 禁止）に抵触する。
- 起源 plumbing（intent への obligation 伝搬）は D105 completion authority・HANDLER-1・
  transport trivially-copyable 制約に触れる非最小変更であり、R25 §3 の「追加するものは以下だけ」に違反する。
  本 Step では設計しない（R26 側の改訂設計事項として §8 に再開条件のみ記録する）。

## 6. Build Gate（R25-B のため未実施・正当記録）

```text
BUILD_EXIT = （未実行）
理由： R25-B（実装非承認）のため、§3–§6 の production／test-only 変更は一切行っていない。
Full-target MKL include-path 問題との分離記録も発生しない（harness build 自体を実行していない）。
```

- R19/R22 の分離記録（harness target を authoritative gate とする）は継承有効のまま温存する。
- 禁止事項（R25 §9）の遵守：F run／R run／R23再測定／30s measurement／
  coordinator原因判定／admission原因判定／retry/defer計装／drop reason計装／P3-1-D／
  buzz原因帰属のいずれも実行していない。

## 7. Source reconciliation（R25-B 用・現状 vehicle の再確認）

```text
coordinatorTakeCount_ declaration = 0（AudioEngine.h に存在しない）
getCoordinatorTakeCount() = 0（同上）
writer = 0（ProcessIntent.cpp:47 付近に fetchAddAtomic 追加なし）
既存 counter（take／bld／cmt）の declaration／getter／writer 各1・順序不変（§2）
emitQDelta に coord field なし（req／que／dup／take／bld／cmt／drp／seq／blo のまま）
production差分 = R23 vehicle のまま（§1）。R25 による差分 0。
```

- writer 位置の順序確認（`ownership transfer → coord++ → next processing`）は、
  実装が存在しないため対象なし。境界自体の順序は §3 に記録した
  （`intentQueue_.pop → residency減算 → dispatch → ownerChannel_.take → publish → seq bump`）。

## 8. R25 Gate

```text
R25-A： 非該当（§5 FAIL のため IMPLEMENTED＋BUILD PASS に進めない）。
R25-B： ADOPTED
  Coordinator take 境界は source 上一意に確定した
  （processIntent の intentQueue_.pop・ProcessIntent.cpp:47・Publish型）。
  しかし当該境界は通常 rebuild／Recovery publish／deferred再enqueue を混在させ
  （ISRRuntimePublicationCoordinator.h:785-786 の自己文書化）、
  intent payload に起源 marker は残らず（AudioEngine.h:4872・Orchestrator.cpp:321）、
  generation linkage も禁止のため、main-site-only の 1-writer は構成不能。
  加えて R23 §10 で non-main-site publish の実在が裏付け済み。
  → 実装承認せず。build せず。R26（5軸測定）は本設計のままでは実行不能のため block とする。
R25-C： 非該当（既存 counter との意味重複・writer 数・ownership 境界に矛盾なし。
  矛盾ではなく「専用性の不存在」という設計事実の確定であるため B とする）。
```

- R26 再開条件（設計改訂事項・本 Step では着手しない）：
  起源 marker の最小 plumbing 可否、または window-level での recovery 不在の検証可能化。
  いずれも別 gate の設計監査を要する。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。
- source は R23 vehicle のまま保持する。revert なし。
