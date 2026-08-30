# D129-2 — Unified Repair Implementation Contract Review

**Date:** 2026-08-29
**性質:** read-only 実装契約レビュー。**production source 変更 0**
**最優先問:** 「DeferredFadingActive になった build 済み DSPCore の唯一の ownership holder は誰か」— **解決（line-level）**

---

## 1. 最優先回答 — DeferredFadingActive DSPCore の ownership holder（line-level）

### 実コードの所有権遷移（rebuild path の publish）

```text
rebuild thread: build DSPCore → world 構築
  ↓ trySubmit → executor_.publish → commitRuntimePublication facade
  ↓ registerDSPHandleForRuntime(regCtx.dsp)          ← runtimeDSPHandleMap_ に登録
  ↓   rollbackHandle = handle（ScopeExit で rollback 予定）
  ↓ enqueue 成功
  ↓ rollbackHandle = null()                           AudioEngine.h:4689
  ↓   【rollback 無効化 — 登録は永続化】               ★ 所有権の durable 点
  → PublishIntent enqueue → CoordinatorLoop
```

### 分岐別の DSPCore 帰趨

| admission decision | world（RuntimeState） | DSPCore |
| --- | --- | --- |
| **Accepted** → executePublish | ownerChannel take → authority.publish（swap）→ oldWorld は bridge retire | tail → onPublishCompleted → 正常 lifecycle |
| **DeferredFadingActive** → enqueueDeferred(req) | **deferredSlot_.request が world を所有**（DeferredPublishSlot、RuntimePublicationOrchestrator.h:32-38） | **runtimeDSPHandleMap_ 登録が永続**（retirable な durable identity） |
| **Discard**（TTL 30s / Generation / Sequence — PublicationAdmission.cpp:70-93） | finishView → slot reset → **world 破棄** | **world は DSPCore を non-owning 参照のため破棄されない → map 登録のまま永久滞留** |

**回答（line-level 確定）**:
1. **DSPCore の唯一の durable retirable identity holder = `runtimeDSPHandleMap_` の registration**（facade enqueue 成功時に rollback 無効化済み — AudioEngine.h:4689 `rollbackHandle = DSPHandle::null()`）。map erase（DELETE-1）が唯一の ownership 終端。
2. **world（RuntimeState）の所有者 = deferredSlot_ の PublishRequest**。ただし world → DSPCore は **non-owning 参照**（graph.activeNode は "visibility only, no ownership" — RuntimeGraph.h:15）。
3. **現行 discard 経路の欠陥（line-level）**: `DeferredPublishView::discard`（RuntimePublicationOrchestrator.cpp:570-576）→ `finishView()` → slot reset → **world は破棄されるが DSPCore の retire は行われない** → INV-REBUILD-EXEC-2 違反（ownership holder は存在するが終端経路が未実装）。

### 契約追加（D129 実装項目に確定）

**INV-DEFERRED-1**: deferred slot の discard（TTL/Stale/Generation）時に、slot に紐づく未 publish DSPCore を `retireDSPHandleForRuntime(dsp)` で retire すること（map erase → EBR enqueue → destroy。未 publish につき RT reader なしで epoch 即時成立）。
実装位置: `DeferredPublishView::discard` または `finishView`（discard 時のみ）— request.stateOwner_ の world から topology.activeDSP を取得して retire。**None（Ready）では実行しない**。

## 2. M1 契約（G1 確認）

- publish **成功後のみ** activate（tail 内 / onPublishCompleted）。publish 前・rollback 後は呼ばない（facade の ScopeExit が rollback を完結させてから tail に到達するため構造的に分離済み）
- emergency path も同一 ownership invariant（activate は publish 成功後の不変操作）
- `newDSP == nullptr`（idle family）では activate しない

## 3. M2 overlap — dual retire の順序固定（G2/G5 再確認・修正）

**採用**: CAS + explicit retire chain（exchange 復活は不採用 — emergency 経路の `exchangeFadingRuntimeDSP` は現行のまま独立維持）。

```text
1. C publish（commit）
2. claim(B) 失敗（slot = A 占有）
3. slot 占有者 A を CAS で取得（上書き前に ownership 確定 — D129-2 指示）
4. retire(A) — 中断 crossfade の fading DSP
5. retire(B) — 置換された current
6. A→B record unregister + B→C は通常遷移
7. C remains active
8. fading slot == nullptr
```

**禁止**: `fadingRuntimeDSPSlot` を先に上書きしてから A を探索する方式（D129-2 指示どおり — 占有者確定を先に行う）。

## 4. handle map / EBR の順序分離（ユーザー指摘の修正）

D129-1 の「{A,B,C} → dual retire → {C}」表現は**最終状態のみ**を示したもの。line-level の順序契約:

```text
logical ownership removed   = runtimeDSPHandleMap_ erase        （DELETE-1、retire 時）
  ↓ registry state          = DSPHandle → Retired              （ISRDSPHandle.cpp:125）
  ↓ retire enqueued         = enqueueWithRetry(destroyDSPCoreNode)（DELETE-3 enqueue）
  ↓ epoch safe              = drain が minReaderEpoch 通過を確認
  ↓ slot reclaim            = registry → Reclaimed + freelist push
  ↓ physical destruction    = destroyDSPCoreNode（~DSPCore → aligned_free）
```

I4 契約（Publish → Fade → Retire → Grace(EBR) → Reclaim）と完全整合 — 短絡なし。D129-1 の表は「map の最終状態」として読み替える。

## 5. timeout は recovery path（G6 補完）

```text
normal:    RT 1→0 → completion signal → Timer → ownership resolution → retire
exception: fade age > deadline（getFadeAgeUs）→ timeout sweep → 同一 ownership resolution routine → retire
```

- timeout は completion event を発行**しない**（同一 ownership resolution routine への直接入口）
- completion と timeout の競合 → terminal retire exactly-once（**INV-XFADE-COMP-2**）— 冪等 retire（map erase 済み no-op）で担保
- queue drop（容量32）時: timeout net が唯一の recovery（D129-1 確定）

## 6. Invariants（D129 実装契約・最終版）

| ID | invariant |
| --- | --- |
| INV-XFADE-COMP-1 | completion signal は ownership identity を生成・変更せず、durable state（slot + receipt）から対象を解決するのみ |
| INV-XFADE-COMP-2 | completion/timeout 競合時も terminal ownership transition は exactly-once |
| INV-XFADE-OVERLAP-1 | claim failure 時、slot 占有者と新 oldDSP の ownership を双方終端 |
| INV-XFADE-OVERLAP-2 | retire **前に** slot 占有者を取得して ownership 確定（上書き先行禁止） |
| INV-XFADE-OVERLAP-3 | retire 後 fading slot は必ず null |
| INV-XFADE-TIMEOUT-1 | timeout は completion と独立した recovery path（event 発行なし） |
| INV-REBUILD-EXEC-1 | build 済み DSPCore は publish / obsolete-destroy / discard-retire のいずれかで必ず終端 |
| INV-REBUILD-EXEC-2 | deferred / obsolete DSPCore には ownership holder（map registration）が存在し、終端経路（retire）が実装される |
| INV-REBUILD-WAKE-1 | RetryReady wake は pending task がなければ build しない |
| INV-DEFERRED-1 | deferred slot discard 時、未 publish DSPCore を retire する |
| INV-LIFECYCLE-1 | publish → transition → retire → destroy が単一 lifecycle chain（map erase → enqueue → EBR → destroy の分離維持） |

## 7. M3 / M4 / M5 / M6

- **M3**: D127 実装（15,371 RetryReady wake → stale build 0）を**無変更で採用**。拡張しない
- **M4**: **保留**（D127 指示継続）— M3 の効果確認が先行
- **M5**: Observe 撤去（D127-D 実装済み設計）— retire authority から完全除外
- **M6**: shutdown 診断 — SHUTDOWN_BEGIN / LOGGER_DETACH / mainWindow.reset() / SHUTDOWN_END の順序で 139 crash phase を特定（D127-E 実装を再適用）

## 8. 診断項目（D129 実装時・macro-gated）

```text
[D129_XFADE_START]      xfadeId / fromHandle / toHandle / fromDSP / toDSP
[D129_XFADE_COMPLETE]   xfadeId(0=signal) / fadingHandle / fadingDSP
[D129_XFADE_RETIRE]     xfadeId / handle / DSP / retireResult
[D129_XFADE_TIMEOUT]    xfadeId / ageUs / handle / DSP
[D129_ADMISSION]        seq / generation / decision / deferReason
[D129_TASK_EXEC]        intentId / taskGen / buildResult / publishResult / obsolete / destroy
[D129_TASK_WAKE]        （D126 実装済み・M3-A 版に修正済み）
```

identity-free completion でも **DSPCore pointer + handle + generation + sequence** の組合せで lifecycle 全程を対合可能。

## 9. GO/NO-GO（8項目）

| # | 条件 | 判定 |
| --- | --- | --- |
| 1 | M1 activate が publish-success-only | **PASS**（tail 内・rollback 分離構造確認） |
| 2 | M2 completion が RT 1→0 edge 起点 | **PASS**（LinearRamp remaining 収束 — 方式A） |
| 3 | completion event が pure signal | **PASS**（identity は slot CAS + receipt — D129-1 採用案） |
| 4 | overlap dual-retire が ownership-safe | **PASS**（両者新 world から到達不能・冪等・EBR保護 — 条件: retire 前 slot 取得） |
| 5 | completion/timeout exactly-once | **PASS**（冪等 retire + INV-XFADE-COMP-2） |
| 6 | deferred DSPCore ownership holder 明確 | **PASS** — map registration（durable retirable identity）+ discard 時 retire 追加を契約化（INV-DEFERRED-1） |
| 7 | M3 無変更再利用 | **PASS** |
| 8 | production semantics 変更なし（レビュー段階） | **PASS** |

**D129-2: PASS（8/8）→ D129-3（実装パッチ作成）を GO。**

## D129-3 実装パッチの構成（確定）

| 項目 | ファイル | 内容 |
| --- | --- | --- |
| M1 | DSPTransition.h | activate 公開（normal/emergency）+ same-DSP ガード（D127-A 実装の再適用） |
| M2a | AudioBlock.cpp | RT ramp 完了エッジ → notifyRampComplete（純シグナル） |
| M2b | CrossfadeRuntime.h | notifyRampComplete（SPSC push、id=0） |
| M2c | AudioEngine.Timer.cpp | consume → active records 解決 → endCrossfade → **slot 取得 → dual retire**（claim-fail 分岐含む）→ retirePublishedDSP + timeout net |
| M3 | AudioEngine.RebuildDispatch.cpp | wake reason 分離 + stale task ガード（D127-B 実装の再適用） |
| M5 | ProcessIntent.cpp | Observe retire 誘発撤去（D127-D 実装の再適用） |
| M6 | MainApplication.cpp / MainWindow.cpp | shutdown 診断（D127-E 実装の再適用） |
| 新規 | RuntimePublicationOrchestrator.cpp または DeferredPublishView | **discard 時 DSPCore retire**（INV-DEFERRED-1） |
| 診断 | 各所 | D129 trace 7 種（macro-gated） |

**検証 Gate**: Compile（両config）→ CTest 40/40 → 3-burst（TASK_WAKE/TRANS/RETIRE/DESTROY 対合）→ 6/10-publish（DC bounded）→ burst×10 → ≥180s long-run（slope≈0）→ shutdown ×8（139 頻度 + crash phase 特定）

## 生成物

- 本ファイル（`evidence/D129_2_UNIFIED_CONTRACT_REVIEW.md`）
- production source 変更: **0**
