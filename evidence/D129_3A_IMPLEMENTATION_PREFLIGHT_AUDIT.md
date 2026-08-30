# D129-3A — Implementation Preflight Audit

**Date:** 2026-08-29
**性質:** read-only 実装前監査。**production source 変更 0**
**目的:** D129-3（6 Patch 実装）の実装可能性を、残る 3 つの未確定点の line-level 証明により固定する
**判定:** **PASS（5/5 Gate）→ D129-3 実装 GO**

---

## Gate A — timeout authority の構造確認: **PASS**

### 実コード（Timer.cpp:1674-1720 — 第3の CAS サイト）

```cpp
if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
{
    // 1. fading slot CAS → current（DSPCore*）取得
    if (CAS(fadingRuntimeDSPSlot, current → nullptr))
    {
        DSPLifetimeManager lifetime(*this);                    // ★ 生成されるが未使用
        const auto fadingHandle = getFadingRuntimeDSPHandle(); // ← endCrossfade 済みで null
        if (!fadingHandle.isNull())                            // ← 常に skip
            runtimePublicationBridge_.submitObserve(...);      // ← retire 誘発せず
    }   // ★ current はここで廃棄 — retire されない（第3の漏出サイト）
    // 2. 全 active crossfade record を unregister
    // 3. crossfadeRuntime_.complete()
    // 4. publishIdleWorldOnly(currentAfterFade, HardReset)
}
```

**確定事項**:
1. **第3の漏出サイト確定**: CAS で取得した `current`（DSPCore\*）が retire されず廃棄。fading handle は activate/endCrossfade の遷移で null になるため submitObserve も不発
2. **ISR-HM 準拠構造は既存**: HealthMonitor（RuntimeHealthMonitor.cpp:548/555）は `HealthEvent`（シグナル）の発行のみ — **retire を実行しない**。実際の ownership transition は Timer（message thread = NonRT lifecycle authority）が実行する構造 → **ISR-HM-001/003 と整合**
3. **修正設計（M5 + Gate A）**: ハンドラ内の `submitObserve` ブロックを **`retirePublishedDSP(current, lifetime)` に置換** — HealthMonitor はシグナル発行のみ維持し、ownership transition は既存 NonRT lifecycle authority（Timer 上の retirePublishedDSP）が担う。`DSPLifetimeManager lifetime` は既に同関数内で生成可能（他サイトと同一パターン）

**HealthMonitor を retire authority にしない構造**: ✅ line-level 確定（シグナル → Timer 実行の分離は現行のまま）

## Gate B — Deferred discard の DSPCore identity: **PASS**

### identity chain（line-level）

```text
DeferredPublishSlot.request                        RuntimePublicationOrchestrator.h:32-38
  ↓ PublishRequest.newDSP（DSPHandle、generation 付き）   PublicationAdmission.h:20
  ↓ resolve(newDSP) → DSPCore*                        ISRDSPHandle.cpp:61-75（generation 検証）
  ↓ runtimeDSPHandleMap_ lookup                       AudioEngine.h:4309
  ↓ retireDSPHandleForRuntime(dsp) → DELETE-1/2/3
```

### ownership の実コード証明

| 項目 | 事実 |
| --- | --- |
| DSPCore の map 登録 | rebuild path は `RegistrationContext::alreadyRegistered(existingHandle)`（PublicationExecutor.cpp:56）— **DSPCore は build 時に create 済みで map 登録永続**（facade の rollback は enqueue 成功時に無効化 — AudioEngine.h:4689） |
| world の所有者 | deferredSlot_.request.stateOwner（RuntimeState）— **DSPCore を non-owning 参照**（graph.activeNode は visibility only） |
| discard 時の world | finishView → slot reset → **world は破棄されるが DSPCore は独立して生存** |
| discard 時の DSPCore | map 登録のまま・retire されず → **永久滞留（INV-REBUILD-EXEC-2 違反を実コードで確定）** |
| identity の一意性 | `request.newDSP` は当該 build の create() が返した generation 付き handle — **1:1 対応・他 DSPCore と混同不可能** |

**結論**: `discard()` 時に `resolve(request.newDSP).instance` で DSPCore を取得し `retireDSPHandleForRuntime` → EBR destroy で終端できる。**INV-DEFERRED-1 の実装は identity 証明済みとして可能**。

### ownership lifetime の注意（consume との対称性）

- `consume()`: `request` を move-out → finishView（slot reset）→ request は呼び出し側で生存
- `discard()`: request は slot 内で破棄されるが、**DSPCore は world と別の allocation**（map 登録が identity）のため、retire は discard の**前でも後でも**可能（推奨: discard 内・finishView 前に実行して確実性向上）
- **deferred 中の_DSPCore が map から消えることはない**（erase は retire のみ）— discard 時 lookup 必ず成功

## Gate C — 共通 ownership-terminalization routine: **PASS（設計確定）**

```text
terminalizeFadingDSP() [NonRT・Timer thread]:
  1. CAS fadingRuntimeDSPSlot: current → nullptr（取得失敗なら即 return）
  2. retirePublishedDSP(current, lifetimeMgr)
       identity = current（DSPCore*・一次）
       receipt 交叉検証（resolve(receipt.handle).instance == current）
       mismatch → receipt 破棄 + retire(current, 0)（quarantine/fatal 禁止 — D122-B）
  ↓ DELETE-1: retireDSPHandleForRuntime（map erase + slot Retired）
  ↓ DELETE-2: requestReclaimHandle（epoch 安全 → reclaim / 不安全 → pendingReclaimHandles_）
  ↓ DELETE-3: enqueueWithRetry(destroyDSPCoreNode) → drain → physical destroy
```

**呼び出し元（3 経路が同一 routine に収束）**:
1. fadeCompleted ブロック（Timer:958-975 — 通常 completion）
2. EVENT_CROSSFADE_TIMEOUT ハンドラ（Timer:1690-1700 — timeout recovery）
3. Timer:1102 ブロック（!isFading パス）

**効果**: completion / timeout / queue drop / overlap / double retire が **同一 terminalization primitive** で処理される。INV-XFADE-COMP-2（exactly-once）は冪等 retire（map erase 済み no-op）で担保。**timeout は completion event を発行しない**（直接 routine 入口 — D129-2 契約どおり）。

**SnapshotCoordinator の `advanceFade()` は本修正に混入しない**（独立機構 — 維持確認済み）。

## Gate D — A→B→C dual-retire の接続点: **PASS**

接続点（実コード位置）:

1. **DSPTransition.h claim-fail 分岐**（現行: `if (!claimed) { lifetime.retire(oldDSP); }`）に追加:
   ```text
   slot 占有者（旧 fading DSP）を CAS 取得 → retire（両方）
   + 当該 crossfade record を unregister
   ```
   順序（D129-2 契約）: **retire 前に slot occupant を CAS 取得**（上書き先行禁止）→ retire(A) → retire(B) → slot == nullptr
2. **完了経路**（Timer consume → endCrossfade → slot CAS → retirePublishedDSP）は M2 実装時に接続

**短絡禁止の確認**: 両 retire とも `DSPLifetimeManager::retire`（map erase → EBR enqueue）経由 — **直接 delete / aligned_free なし**（Practical 原則: Retire と Delete の分離維持）。

## Gate E — trace 契約: **PASS（確定）**

```text
[D129_XFADE_START]      xfadeId / fromHandle / toHandle / fromDSP / toDSP（DSPTransition・macro-gated）
[D129_XFADE_COMPLETE]   xfadeId(0=signal) / fadingHandle / fadingDSP（Timer consume）
[D129_XFADE_RETIRE]     xfadeId / handle / DSP / retireResult（retirePublishedDSP）
[D129_XFADE_TIMEOUT]    xfadeId / ageUs / handle / DSP（timeout sweep）
[D129_ADMISSION]        seq / generation / decision / deferReason（macro-gated 1点）
[D129_TASK_EXEC]        intentId / taskGen / buildResult / publishResult / obsolete / destroy
[D129_TASK_WAKE]        （D126 実装済み）
```

DSPCore\* + handle + generation + sequence の組合せで identity-free completion でも全程対合可能。

## D129-3 実装順序（Preflight PASS に基づく確定版）

| Patch | 内容 | 依存 |
| --- | --- | --- |
| 1 | M3（D127-B 実装の再適用・無変更） | なし |
| 2 | M1（D127-A 実装の再適用・無変更） | なし |
| 3 | M2: RT edge + SPSC signal + Timer consume + **terminalizeFadingDSP 統一routine**（通常 completion / overlap dual-retire / claim-fail）| Patch 1, 2 |
| 4 | M5: timeout ハンドラの submitObserve → terminalizeFadingDSP 置換 + Observe 撤去 | Patch 3 |
| 5 | Deferred discard retire（INV-DEFERRED-1） | Gate B 証明済み |
| 6 | M6 shutdown 診断 | なし |

## 生成物

- 本ファイル（`evidence/D129_3A_IMPLEMENTATION_PREFLIGHT_AUDIT.md`）
- production source 変更: **0**
