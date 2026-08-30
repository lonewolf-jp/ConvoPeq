# D131 — DeferredFadingActive / Deferred Slot Ownership Repair Design Audit

**Date:** 2026-08-29
**性質:** read-only 設計監査。**production source 変更 0**
**Frozen HEAD:** `a65ace1` + 診断 trace 群
**判定:** **PASS（G1-G6）— D132 実装スコープ確定（M1+M2+M3+M5+M6+INV-DEFERRED-2/3 の単一パッチ）**

---

## D131-G1 — DeferredPublishSlot overwrite の実所有権（line-level 確定）

### `enqueueDeferred` の overwrite 実装（RuntimePublicationOrchestrator.cpp:460-476）

```cpp
deferredSlot_ = DeferredPublishSlot{
    .request = req,          // ← const& 受け取り（move しない）
    ...
};
```

**orphan 生成点は 2 箇所**（line-level）:

| # | 生成点 | 機構 |
| --- | --- | --- |
| ① | `submitPublishRequest` return 時 — `enqueueDeferred` は **const& 受け取り（move しない）** → 呼び出し側の `req.stateOwner_`（world unique_ptr）が return で破棄され、**未 publish DSPCore が world と共に取り残される** | world 破棄 ≠ DSPCore 破棄（non-owning 参照） |
| ② | slot 上書き時 — `deferredSlot_ = 新slot` の move-assign で**旧 slot の request（前回の world）が破棄** → 同様に DSPCore orphan | 同上 |

**DSPCore の durable identity**: `runtimeDSPHandleMap_` の registration（facade enqueue 成功時に rollback 無効化 — AudioEngine.h:4689）。**map erase が唯一の ownership 終端**（INV-REBUILD-EXEC-2）。

**D129-3B 実測との整合**: 6 burst → 6 DeferredFadingActive → 各 return 時 orphan ①（+slot 上書き時 ②）→ DC live 5 ≈ baseline + orphans ✓ 完全一致。

## D131-G2 — INV-DEFERRED-2 の契約確定

**採用順序（実コード上最も安全な形）**:

```text
new request 到着（DeferredFadingActive）
  ↓ 【新 request が same-DSP republish でないことを確認】
  ↓ 【旧 request が存在する場合（hasDeferred_ == true）】:
      oldDSP = resolve(old request.newDSP)      ← generation-safe resolve
      retireDSPHandleForRuntime(oldDSP)          ← DELETE-1（map erase + EBR enqueue）
  ↓ 旧 request/world release（slot 上書きまたは破棄）
  ↓ 新 request install
```

**順序の判定**: 旧 DSPCore retire → world release → 新 install の順。逆順（world release → retire）は world 破棄後に resolve できなくなるリスク（DSPCore は map から取得可能だが、generation 検証のため handle が必要 — handle は request が保持）のため、**retire を先に実行**する方が安全。

**注意**: retire の対象は「旧 request の newDSP」= 「deferred されていた未 publish DSPCore」。**当該 DSPCore は新 request の world から到達不能**（新 request は新 build の新 DSP を参照）→ retire は ownership-safe。

## D131-G3 — consume / discard / overwrite 三経路分離（G3）

| 経路 | old/new DSP | world | terminal action | ownership |
| --- | --- | --- | --- | --- |
| **consume**（fade 完了後） | newDSP は publish candidate | consumer（submitPublishRequest）へ移動 | publish lifecycle → tail → transition | **fade 完了まで slot が DSPCore を保持**（ ownership 移譲なし） |
| **discard**（TTL/stale） | unpublished | 破棄 | **DSPCore retire 必須**（現行欠落 → INV-DEFERRED-1） | finishView 前に retire |
| **overwrite**（新 DeferredFadingActive） | 旧: unpublished / 新: 新 build | 旧破棄 → 新 install | **旧 DSPCore retire 必須**（現行欠落 → INV-DEFERRED-2） | slot 上書き前に retire |

**consume 後の guard ownership**: consume は `request` を move-out して finishView → **slot は空になり DSPCore の ownership は publish lifecycle へ移譲** ✓（guard は残留しない）。

## D131-G4 — M2 × INV-DEFERRED-2 の double-retire 検証（G4 最重要）

| DSPCore | M2 retire 対象 | INV-DEFERRED-2 retire 対象 | 二重 retire の可能性 |
| --- | --- | --- | --- |
| fading slot 占有者（旧 current） | **✅ 対象**（slot CAS 取得） | ✗（deferred slot の newDSP は別個体） | **なし** |
| deferred slot の newDSP | ✗（fading slot ではない） | **✅ 対象**（overwrite/discard 時） | **なし** |

**同一 DSPCore が両経路の対象になるケース**: 「deferred の newDSP が fade completion によって fading になった」場合 — 発生条件: fade 完了 → slot clear → 次の publish（DeferredFadingActive）→ newDSP が deferred slot へ → **次の fade** が開始 → その fade の fading DSP は前回の newDSP… 

**検証結果**: 各 fade の fading DSP は「前回 publish 済みの current DSP」であり、deferred の newDSP は「新 build の未 publish DSP」— **常に異なる個体**。二重 retire は構造的に発生しない。**冪等 retire（map erase 済み no-op）も保険として成立**。

**結論: exactly-once が構造的に保証される（pointer-based idempotence + 対象分離）**。

## D131-G5 — `setIRChangeFlag()` の責務監査（G5）

全 3 caller（D124 確定）の意味論:

| caller | 起因 | flag の意味論 | 評価 |
| --- | --- | --- | --- |
| UIEvents.cpp:177 | UI/外部 IR 変更 | **semantic な変更通知**（IR が変わった → 構造 rebuild が必要） | 正当 |
| Timer.cpp:800 | deferred Structural rebuild **発行時の伴奏** | rebuild を起こした後の伴奏 flag | **責務過剰** — rebuild は既に発行済み、flag は二重通知 |
| DSPTransition.h:123 | rebuild 自身の crossfade | **crossfade 後処理としての IR 変更通知** | **責務過剰** — rebuild 起因の crossfade は IR 変更を意味しない（新 DSP は既存 IR を transfer） |

**D129-3B の `1 publish → crossfade pending → subsequent rebuild → DeferredFadingActive → overwrite → orphan` は setIRChangeFlag の責務過剰で説明可能**:
- DSPTransition の crossfade 分岐が flag を設定 → promote → sr_bs rebuild が発火 → 新 DSP + DeferredFadingActive → orphan
- **flag の本来の意味（「IR が変わった」）からすると、rebuild 起因 crossfade で flag を立てる理由がない** — UI 側 IR 変更（UIEvents）と Timer deferred release（rebuild 伴奏）のみが正当

**ただし**: Timer:800 の伴奏 flag は「deferred rebuild の伴奏」として既存設計の可能性がある（単独ではループを作らない — D124 確定）。**DSPTransition:123 のみがループ入口**であり、D132 の修正対象は :123 に限定される。

## D131-G6 — M2 completion による DeferredFadingActive 解消（G6）

**line-level 証明**:

```text
fade completion（M2 consume block）:
  endCrossfade(id) → from→Retired / to→Active / activeHandle=to / fading=null
  terminalizeFadingDSP() → fading slot CAS clear → retirePublishedDSP(旧DSP)
```

- **fading slot** = null → `claimFadingRuntimeDSP` が次回 publish で成功可能
- **fadingRuntimeDSPHandle_** = null → `hasFadingRuntimeInWorld` は world 投影を見るため、**idle world 再 publish（fadingNode=null）が必要**

**重要な注意（D129 実装の追加項目として確定）**: published world は immutable のため、fade 完了後も `hasFadingRuntimeInWorld` が true を返し続ける可能性がある（current world の fadingNode が残存）。→ **M2 consume block 内で terminalizeFadingDSP の後に、fadingNode=null の idle world 再 publish（既存 `publishIdleWorldOnly` を利用）を追加する必要がある** — これが `hasFadingRuntimeInWorld` を false に戻し、次の publish を Accepted に戻す。

**証明**:
```text
M2 consume: endCrossfade → terminalizeFadingDSP → retire → destroy（EBR）
  ↓ publishIdleWorldOnly(current, HardReset)   ← fadingNode=null の idle world
  ↓ evaluate: hasFadingRuntimeInWorld → false（新 world に fading なし）
  → Accepted ✓
```

**結論**: M2 は「leak repair ではなく、DeferredFadingActive を解除する state-machine repair」という位置付け**が成立する** — ただし idle world 再 publish の併設が必須。

## D131-G7 — 修正案比較（G7 最終）

| 軸 | 案 A: M2 + INV-DEFERRED-2 | 案 B: 案 A + setIRChangeFlag 責務修正（:123 撤去） | 案 C: INV-DEFERRED-2 のみ |
| --- | --- | --- | --- |
| orphan prevention | ✅（overwrite/discard 時 retire） | ✅ | ⚠️ 部分的（slot overwrite 時のみ） |
| crossfade liveness | ⚠️ fade 完了するが promote ループが残る（sr_bs rebuild が 4s 毎に来続ける → 1 publish/4s で正常化するが余分な rebuild が継続） | ✅（ループ入口遮断） | ✗（fade pending 継続 → deferral ループ残存） |
| rebuild storm prevention | ⚠️ 部分的（rebuild 自体は継続・publish は正常化） | ✅ | ✗ |
| ownership safety | ✅ | ✅ | ⚠️ |
| RT safety | ✅ | ✅ | ✅ |
| semantic change magnitude | 中（M2 が大） | **やや大（:123 撤去を含むが行削除のみ）** | **小** |

**結論: 案 B を採用** — 理由:
1. 案 A では promote ループ（setIRChangeFlag → sr_bs rebuild）が残り、**各 cycle で新 DSPCore が build され DeferredFadingActive → INV-DEFERRED-2 retire が常時発火**（負荷・ログ・race surface の増大）
2. 案 B の `DSPTransition.h:123` 撤去は **1 行削除**であり、D124 で確定した「rebuild 起因 crossfade で flag を立てる理由がない」（新 DSP は既存 IR を transfer するため IR 変更通知の意味がない）と整合
3. INV-DEFERRED-2 は案 B でも必須（Timer:800/UIEvents:177 の正当な flag 設定時に overwrite が発生し得るため）

## D129-3B 実測との整合確認

| D129-3B 観測 | 案 B 適用後 |
| --- | --- |
| 6 burst → 1 publish / 4 orphan | 6 burst → 6 publish（idle world 再 publish で fade 解消後 Accepted）→ **orphan 0** |
| 1 crossfade → 永久 pending | completion 駆動で fading 解消 → 次の publish Accepted |
| RetryReady wake 450Hz storm | fade 解消 + deferred 解消後は storm 消失（deferral 理由が fade だったため） |

## GO/NO-GO（G1-G6）

| Gate | 判定 |
| --- | --- |
| G1 overwrite ownership 追跡 | **PASS**（orphan 生成点 2 箇所 line-level 確定） |
| G2 INV-DEFERRED-2 契約 | **PASS**（retire 先行順序 + generation-safe resolve + 冪等性） |
| G3 三経路分離 | **PASS**（consume は非terminal・discard/overwrite は terminal の分離表） |
| G4 double-retire 不在 | **PASS**（対象分離 + 冪等保険 — 構造的に exactly-once） |
| G5 setIRChangeFlag 責務 | **PASS**（:123 のみループ入口・責務過剰を D129-3B 実測で説明） |
| G6 M2 による loop 停止 | **PASS**（idle world 再 publish の併設を契約に追加） |

**D131: PASS（G1-G6）** → **D129-2 契約を案 B で更新し、D132（M1+M2+M3+M5+M6+INV-DEFERRED-2/3 統合実装）へ GO。**

## 生成物

- 本ファイル（`evidence/D131_DEFERRED_OWNERSHIP_REPAIR_AUDIT.md`）
- production source 変更: **0**
