# D162-2-I2-0 — Repair Preflight（read-only・production/test/build 0 変更）

```text
Date:     2026-09-05
Type:     I2-0 implementation preflight（read-only）
Baseline: ConvoPeq.md Generated 2026-09-04 23:38:10・R2 binary Sep 4 23:09
Input:    D162-2-I1-D-R0（GO）+ 指示 §I2-0-1〜I2-0-4
判定:     **I2-0 = PASS → I2-1（A 案 1 箇所実装）GO**
```

---

## I2-0-1. destroy 対象 pointer の同一性（実装形確定）

### ownership chain（実コード・再確認）

```text
PrepareToPlay.cpp:241  placeholderDSP = aligned_make_unique<DSPCore>()   ← owner = unique_ptr
:264                   setActiveRuntimeDSP(placeholderDSP.release())      ← owner なしに遷移
:277-280               commitRuntimePublication(needsRegistration(getActiveRuntimeDSP()))
                         → AudioEngine.h:4613 tryAdmit fail → {Failed, CallerDestroy}
                         → ignoreUnused(pubResult2)                       ← 義務が消滅（defect）
```

**pointer identity の確定**: `:264` の `release()` 直後、`getActiveRuntimeDSP()` は
直前の `publishAtomic(activeRuntimeDSPSlot, value, release)`（h:2236）が返す値で、
同一 Message Thread 上の release-store → acquire-load であるため **release した
placeholder と必ず同一**（先行書き込みの値であり、この間に第三者による
`setActiveRuntimeDSP` 書き換えは prepareToPlay 契約（JUCE: audio thread 停止中）で
存在しない。`setActiveRuntimeDSP` の caller は CtorDtor/PrepareToPlay/
ReleaseResources の 3 箇所のみで、いずれも Message Thread・本位置と排他的）。

ただし指示どおり、**identity を間接経由（再取得）ではなく release 値の明示保持で固定**する:

```cpp
DSPCore* placeholderRaw = placeholderDSP.release();
setActiveRuntimeDSP(placeholderRaw);
// ... commitRuntimePublication(..., needsRegistration(placeholderRaw), null)
// 失敗時: destroyRolledBackDSP(placeholderRaw) — 同一 pointer を直接渡す
```

これで ownership が一本道になり、`:277` での `getActiveRuntimeDSP()` 再取得を廃止できる。

### DSPLifetimeManager の公開範囲（実装可否）

- `DSPLifetimeManager.h:19`: `void destroyRolledBackDSP(void* dsp) noexcept;` — **public**。
- 実装（DSPLifetimeManager.cpp:149-157）: `destroyDSPCoreNode(dsp)` 直呼び + retiring
  generation counter ++。null guard あり。
- include 経路: PrepareToPlay.cpp は `RuntimePublicationOrchestrator.h` を include し、
  同 header は `DSPLifetimeManager.h` を include 済み（Orchestrator.h:13）→
  **追加 include 不要**で `DSPLifetimeManager` をスタック構築可能（CtorDtor.cpp:178 /
  ReleaseResources.cpp:135/535 と同型の既存パターン）。
- `AudioEngine::destroyDSPCoreNode` 自体は public static（h:4804）だが、既存契約では
  破壊は DSPLifetimeManager 経由に統一（Orchestrator 前例）→ I2 もこれに従う。

## I2-0-2. CallerDestroy 全失敗点の再検証（最新 source）

`enqueueRuntimePublicationFireAndForget`（h:4596-4704）の全失敗 return:

| # | 行 | 条件 | ownership | regCtx.dsp の登録状態 |
| --- | --- | --- | --- | --- |
| 1 | :4613-4614 | **tryAdmit fail** | `CallerDestroy` | **登録前（:4631 に未到達）→ 未登録** |
| 2 | :4636-4637 | register 失敗（rollbackHandle null） | `None`（既定値） | 未登録（jassertfalse・理論上のみ） |
| 3 | :4646-4647 | world == nullptr | `None`（既定値） | **登録済み（:4633 通過後）→ ScopeExit で rollback 済み** |
| 4 | :4650-4651 | seqId == 0 | `None`（既定値） | 同上（登録済み→rollback 済み） |
| 5 | :4663-4668 | OwnerChannel enqueue fail | `CallerDestroy` | 同上（rollback 済み） |
| 6 | :4682-4692 | ISR intent queue full | `CallerDestroy` | 同上（rollback 済み） |

**確定**:

1. `CallerDestroy` を返す 4 失敗点のうち、**DSP が未登録のまま帰るのは (1) tryAdmit
   失敗のみ**。(3)(4)(5)(6) は `ScopeExit guard`（h:4627-4630）が rollback
   （`rollbackRegistration`: Constructing→Reclaimed CAS + map erase・ISRDSPHandle.cpp:157-173）
   を完了してから帰るため、DSP は Reclaimed 状態の「rollback 完了・未公開」個体であり、
   CallerDestroy 義務の履行手段は同じ `destroyRolledBackDSP`。
2. **(2)(3)(4) は `ownership = None` を返す**ため、`CallerDestroy` 分岐に入っても
   誤爆しない（None は破壊義務なし。ただし (3)(4) は rollback 済み個体が caller に
   残る = work70 時点で「commitRuntimePublication 側では破壊しない」契約のもと
   caller 側対応を前提とした経路。prepareToPlay は world==nullptr / seqId==0 になり得ない
   （buildRuntimePublishWorld 成功後・sequenceId は reserve 済み）ため実質不発。
   この理論上の残余リスクは I2 scope 外として記録（I3 候補・None 返り値の caller 契約明確化）。
3. **I2-1 の実装は「CallerDestroy を受け取ったら破壊」で 4 失敗点を一括履行**できる。
   (5)(6) が prepareToPlay 経路で発生した場合も rollback 完了済みのため
   `destroyRolledBackDSP` が正しい履行手段（Orchestrator:291-292 と同型）であり、
   二重破壊は発生しない（rollback は state CAS のみで instance を触らない・
   destroyRolledBackDSP は EBR を通らない単発破壊）。

## I2-0-3. pubResult1 / pubResult2 の ownership 差（固定）

### pubResult2（:277・placeholder・本修復対象）

- 新規生成 DSP → **map 未登録** → needsRegistration → tryAdmit 失敗 → 登録処理に到達せず
  → **未登録のまま CallerDestroy** → 誰も回収できない → orphan（実測 6/6）。

### pubResult1（:155・idle #2・既存 DSP）

- `hasAnyRuntime`（world 解決で current/fading が実在）のときのみ発行。
  対象は **world に公開済み = map 登録済み DSP**（world publish 時に registration 完了）。
- needsRegistration を渡しても `registerDSPHandleForRuntime` は **idempotent**
  （h:4327-4329: `find` で既存 handle を返す・新規 slot を消費しない）。
- tryAdmit 失敗時: rollbackHandle = **既存 handle** → ScopeExit で
  `rollbackRegistration` を試みるが、CAS は **Constructing→Reclaimed のみ**成功するため
  **Active 状態の既存 DSP では失敗 → 登録温存**（ISRDSPHandle.cpp:161-165）。
- つまり pubResult1 が CallerDestroy を返しても、DSP は world に公開されたまま
  正規 authority（EBR/retire chain）で回収可能 — **破壊してはならない**（破壊すると
  world の dangling current になる = UAF 潜在）。**pubResult1 に destroy 分岐を入れる
  ことは現行契約では誤り**。

### 「pubResult1 に check を入れない理由」（明文化）

1. pubResult1 の対象 DSP は world 公開済み・registration 温存が保証され、
   既存 retire authority の管理下にある。CallerDestroy が返っても「ownership は
   world 側が引き続き保持する」意味に変化し、caller による破壊は
   **world dangling current（UAF）を生む誤実装**になる。
2. CallerDestroy の契約（h:3635「rollback 完了、呼び出し元が物理解放すべき」）は、
   **rollback が成功して Reclaimed 化した DSPCore** を想定しており、CAS 失敗で
   rollback が不成立（registration 温存）の場合は契約の前提自体が崩れるため、
   ownership フラグ単独では区別できない。実装的には「**CallerDestroy かつ
   当該 caller が唯一の owner である（release() で放棄した新規 placeholder）場合のみ
   destroy**」が安全側で、その条件を満たすのは pubResult2 のみ。
3. pubResult1 が失敗しても leak は発生しない（上記 1-2 の構造的保証・R0 §2.2 再確認済み）。
   したがって最小修復単位 = pubResult2 のみが正しい。

## I2-0-4. activeRuntimeDSPSlot の後始末 + テストインフラ

### slot 後始末

- slot は非所有 topology mirror（h:2245 comment）だが、**dangling pointer を残すと
  将来の address-reuse 時に誤 lookup の潜在（D162-2-C §4.3 / D0-5-A の教訓）**。
- 失敗時: `destroyRolledBackDSP(placeholderRaw)` の後 `setActiveRuntimeDSP(nullptr)`
  で mirror を明示クリア（destroy 前後どちらでもよいが、destroy → null の順が
  読みやすい）。成功時は現行どおり slot に placeholder を残す（起動時 placeholder は
  A2 実測どおり slot 残置・publish 成功後も現行挙動）— **成功パスの挙動変更はしない**。

### テストインフラ（既存調査結果）

- テスト target: `AudioEngineHarness`（CMakeLists.txt:1861-1867）・runner は
  `PublishPipelineIntegrationTests.cpp` の main（デフォルトモードで
  testRebuildPublishCompletes → testIdlePublishViaFacade → ... → runDeferredFlow →
  runDeferredPublishView を直列実行）。
- 既存に `OwnershipDisposition::Transferred` 検証（testIdlePublishViaFacade:143）があり、
  **CallerDestroy 系の直接テストは存在しない** → 新規 1 テスト関数の追加が最小。
- admission Closed をテストで起こす方法: `h.start()`（initialize + prepareToPlay）後、
  `h.engine().releaseResources()` を呼べば closeAdmission 実行 → 同一 engine 上で
  `aligned_make_unique<DSPCore>` + `commitRuntimePublication(needsRegistration)` を
  直呼びすれば tryAdmit 失敗経路を決定論的に再現できる（D profile の実機条件と同一）。
- T-I2-1/2/3 の対応:
  - T-I2-1（CallerDestroy → 破壊確認）: 上記再現 + 破壊確認は
    `[DSP_DESTROY_FOOTPRINT]`/`[DSP_FOOTPRINT_RELEASED]` は logger 前提のため harness では
    JUCE logger capture が必要 → 実用上は「破壊後 slot==nullptr + lookupDSPHandle null +
    MEM 安定」で代理検証。logger capture は JuceLoggerCapture 相当を harness 内で
    実装可能だが I2 では最小構成（slot/lookup 突合）とする。
  - T-I2-2（成功時 destroy されない）: 既存 testIdlePublishViaFacade が
    Transferred を検証済み → 追加不要（T-I2-2 は既存で充足）。
  - T-I2-3（registered DSP failure で二重 destroy なし）: pubResult1 経路の直接テストは
    harness からは組みにくい（world 公開済み DSP の failure を起こすには admission
    close 後に needsRegistration(公開済み DSP) を直呼び → rollback CAS 失敗で登録温存を
    検証可能）。最小実装として「CallerDestroy 後も lookupDSPHandleForRuntime が
    非 null（登録温存）+ slot 突合」を 1 ケースに含める。

## GO / NO-GO

**GO** — 以下が確定した:

1. pointer identity: release 保持形で一本道にできる（再取得不要）。
2. DSPLifetimeManager::destroyRolledBackDSP は public・include 追加不要・既存前例どおり。
3. CallerDestroy 4 失敗点のうち未登録帰還は tryAdmit のみ・それ以外は rollback 済み
   （同一履行手段 destroyRolledBackDSP で一括）・None 返り値 2 点は理論上のみで不発。
4. slot 後始末: 失敗時 destroy → setActiveRuntimeDSP(nullptr)。成功パス不変。
5. pubResult1 を触らない理由が構造的に確定（CAS 失敗 → 登録温存・破壊は UAF 誘発）。
6. テスト: T-I2-1/3 を harness に 1 関数追加・T-I2-2 は既存で充足。
