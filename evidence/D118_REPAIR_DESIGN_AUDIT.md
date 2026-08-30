# D118 — Fade Completion Retire Path Repair Design Audit

**Date:** 2026-08-29
**性質:** 設計監査のみ — **production source 変更 0 ファイル**（D117 の観測専用 trace 2 ファイルは macro-gated のまま保持）
**凍結:** commit 凍結・Phase-II 凍結・production fix 凍結 継続
**Frozen HEAD:** `a65ace1df9b2012a12fa6b656a91f33ad4da9000` + D117 観測トレース（`DSPLifetimeManager.cpp`, `AudioEngine.Threading.cpp` — `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ゲート、production binary への影響ゼロ）

---

## CAUSE-C1 — fade completion ordering defect: **CONFIRMED（静的コードで再証明）**

```text
publish N
  ↓ DSPTransition::onPublishCompleted（CoordinatorLoop / publish 経路）
oldDSP = publish N-1 の current
  ↓ claimFadingRuntimeDSP(oldDSP)           AudioEngine.h:2200  — CAS nullptr→oldDSP 成功
  ↓ storeReceipt(fadingHandle, epoch)       AudioEngine.h:1161  — crossfade 開始時に handle+epoch 保存
  ↓ beginCrossfade(oldHandle, newHandle)    ISRDSPHandle.cpp:82 — fadingRuntimeDSPHandle_ = oldHandle
  ↓ （crossfade 実行 ~158ms）
fade completion（Timer, message thread）
  ↓ endCrossfade(ev.id)                     Timer.cpp:954 / ISRDSPHandle.cpp:103
  ↓   fadingRuntimeDSPHandle_ = null        ISRDSPHandle.cpp:117  ← ここで identity が消滅
  ↓ CAS fading slot (DSPCore*) 成功          Timer.cpp:961-965
  ↓ getFadingRuntimeDSPHandle()             Timer.cpp:970 → **null**
  ↓ submitObserve() 非実行                   Timer.cpp:972（if (!isNull) が不成立）
  ↓ retireByHandle() 非実行                  （D117 トレース: 0 発火）
  ↓ destroyDSPCoreNode() 非実行              （D117 トレース: 0 発火）
  ⇒ DSPCore + StereoConvolver + NUC×2 (~150MB) が publish 毎に滞留
```

補助確定事項: `retirePublishedDSP`（receipt ベースの保険経路）は呼び出し元ゼロ（D118-7 で DEAD-API-1 CLOSED）。

---

## D118-2 — Candidate 比較

### DESIGN-A — handle 読み出しを `endCrossfade()` 前へ移動: **FAIL**

```text
fadingHandle = getFadingRuntimeDSPHandle()   ← 先読み
endCrossfade()
CAS fading slot
submitObserve(fadingHandle)
```

**却下理由 — identity race（D118-5 I-1 不成立）**:
- `fadingRuntimeDSPHandle_` は Timer だけでなく **publish 経路（CoordinatorLoop 上の DSPTransition）** からも書かれる: `activate(newDSP)` が null 化（ISRDSPHandle.cpp:96-99）、次の publish の `beginCrossfade` が **次の crossfade の handle で上書き**（:90）
- よって fade completion 時点で読んだ値が「CAS した DSPCore の handle」である保証が**一般に成立しない**。重複 publish 時は **現行 active DSP の handle を誤 retire する**危険
- 「先に読む」ことと「それが唯一の retire identity である」ことは同義ではない（ユーザー指摘のとおり）

### DESIGN-B — `retirePublishedDSP()` を正規経路として接続: **PASS（2つの必須修正付き）**

構造: fade completion の CAS 成功後、`retirePublishedDSP(current, lifetimeMgr)` を呼ぶ。

**本質的な強み**:
- **identity は CAS で得た `DSPCore*` から直接取る**（`retireDSPHandleForRuntime(dsp)` が runtimeDSPHandleMap_ を DSPCore* で引く — AudioEngine.h:4309）。volatile な fading handle atomic に依存しない
- retire が **無条件に発火**する 3 経路を持つ: Normal（receipt 一致）/ Fallback（receipt 未 ready → `retire(current, 0)`）/ Emergency（不一致 → `retire(current, 0)`）— **receipt が失われても retire 自体は失われない**（失われるのは epoch metadata 伝搬のみ）
- receipt は crossfade **開始時**に保存されるため、endCrossfade による null 化と競合しない

**必須修正 1 — identity check の付け替え（D118-5）**:
現行 `retirePublishedDSP` は `currentHandle(=getFadingRuntimeDSPHandle()) == receipt.handle` 比較（Timer.cpp:1888-1895）だが、現行シーケンスでは endCrossfade 済みで**必ず null ≠ receipt.handle → 常に Emergency 経路**になり、さらに mismatch 時の `submitQuarantine(receipt.handle, PublishViolation)` が**正当な旧 handle を quarantine する**（retire と二重扱い・reclaim 遷移を妨害し得る）。
→ 修正案: 一致判定を **`resolve(receipt.handle).instance == current`**（generation-safe、ISRDSPHandle.cpp:61-75 — generation 不一致は stale として検出）に置換。quarantine submission は真の不正一致の場合のみに限定。

**必須修正 2 — 呼び出し位置**: fadeCompleted ブロックの CAS 成功直後（Timer.cpp:961-965 の置き換え）に接続し、同一ブロックの submitObserve レガシーを retire authority から外す（authority singularization）。

### DESIGN-C — A + B 二重経路: **REJECT**

同一 DSPHandle の二重 retire（submitObserve 経由と retirePublishedDSP 経由）が可能になり、I4 authority singularization の思想に反する。**retire authority は 1 つ**に限定（原則: RT は retire を実行せず Intent/Timer 経由で Coordinator/DSPLifetimeManager へ）。

---

## D118-3 — DELETE-1/2/3 順序（DESIGN-B 適用時）

| 段階 | 内容 | 実装位置 |
| --- | --- | --- |
| **DELETE-1**（retire authority acquisition） | `retireDSPHandleForRuntime(dsp)`: runtimeDSPHandleMap_ から erase + `dspHandleRuntime_.retire(handle)`（slot → Retired 状態） | AudioEngine.h:4309 |
| **DELETE-2**（epoch/reclaim authorization） | `requestReclaimHandle`: epoch 安全なら `requestReclaim`（retire→waitReaders→reclaim、**slot 状態遷移のみ**）、不安全なら `pendingReclaimHandles_` 保留 → drainDeferredRetireQueues で再試行 | AudioEngine.h:4353 |
| **DELETE-3**（physical destruction） | `enqueueWithRetry(destroyDSPCoreNode)` — retire router の drain が epoch 通過後に実行。失敗時は RetireQuarantineStore へ移送（directDelete 禁忌） | DSPLifetimeManager.cpp:49/96 |

契約整合: REPAIR_PLAN.md の確立済み原則「requestReclaim は物理削除を行わない。物理削除は DSPLifetimeManager 経由で別途」及び I4 の ownership chain（publish → oldWorld → retire → epoch grace → destroy）と **完全に整合**。「handle を Retired にした」と「物理 delete してよい」は分離されており、DESIGN-B はこの分離を壊さない（fade completion は DELETE-1 の起点を提供するだけ）。

---

## D118-4 — `endCrossfade()` の責務分離

`fadingRuntimeDSPHandle_` の null 化は **「crossfade 状態の終了」** を表す（`isSlotInCrossfade` / crossfade records 走査 / 次の beginCrossfade の前提など他の消費者が存在）。**「retire identity の消失」まで含意してはならない**。

→ 結論: null 化を遅延させる（Candidate A' 系）は他消費者の意味論を変えるため**不採用**。retire identity は handle atomic から切り離し、**(a) CAS 済み DSPCore*（一次）+ (b) crossfade 開始時に保存した receipt（epoch metadata）** で表現する（DESIGN-B）。

---

## D118-5 — identity 一致証明（DESIGN-B 適用時）

| 項目 | 証明 |
| --- | --- |
| I-1: CAS success → current == retire 対象 | `claimFadingRuntimeDSP(oldDSP)` で slot に入った DSPCore* が fade completion の CAS でそのまま取出される（両者とも message thread / Timer が単一 writer）。DESIGN-B は **ポインタ直接**なので成立 |
| I-2: generation 一致 | `resolve(receipt.handle)` は generation 不一致を stale 検出（ISRDSPHandle.cpp:67-70）— 誤 retire 防止の検証に使用（必須修正 1） |
| I-3: slot 再利用の混入 | `create()` は reclaim 時に generation を bump（ISRDSPHandle.cpp:52-59）— 再利用 slot は stale として判別可能。**stale handle を誤 retire しない**ことは resolve ベース検証で担保 |

---

## D118-6 — Receipt single-slot semantics（実コード確認）

`storeReceipt`（AudioEngine.h:1161-1169）: `receiptReady_` が既に true の場合 **Release では assert 消滅・無音に破棄**。
- Crossfade 重複時: 後発の storeReceipt は棄却 → receipt は先発のまま。DESIGN-B ではこれでも retire は fallback 経由で発火するため **漏出しない**（epoch 伝搬のみ失われる — HW-1 の metadata 目的のみの損失）
- `claimFadingRuntimeDSP` CAS 失敗時（DSPTransition.h:100-103）は直接 `lifetime.retire(oldDSP)` — 既にこの経路は正しく動作している（重複 publish 時の即時 retire）
- 複数 crossfade 同時進行は現行 Timer が `jassert(records.size() == 1)` で単一前提を表明（Timer.cpp:947）。設計としては crossfade の直列化を前提とし、将来の重複許可は receipt の queue 化（Phase 設計変更）として**別契約**に分離

---

## D118-7 — `retirePublishedDSP` call graph: **DEAD-API-1 CLOSED**

`rg 'retirePublishedDSP'` 全ソース + ConvoPeq.md:
- definition: 1（Timer.cpp:1868）
- declaration: 1（AudioEngine.h:1174）
- direct callers: **0**
- indirect references（関数ポインタ・メンバ参照・文字列）: **0**
- コメント内の契約記述のみ（ISRRuntimePublicationCoordinator.h:197「Coordinator Loop で取り出して retirePublishedDSP を実行する」等）— **コメントと実装の不一致**であり、ObserveIntentHandler は実際には `retireByHandle` を呼ぶ（ProcessIntent.cpp:107）

## D118-8 — World lifetime 分離（観測のみ・未修正）

`[WORLD] RetireQueue=1` の滞留は DSPCore lifetime とは別管理（I4: World は DSP handle と別の retire entry）。DSPCore destroy 修正後（D119+）に `WorldRetireQueue / pending World / RuntimeState live` の変化を再測定するまで**保留**。本次 ~150MB/publish の主体は DSPCore 側（MEM_SNAP の DC/SC/NUC live 相関で確定済み）。

## D118-9 — test 21 SEGFAULT: **D118-BLOCKER-1（分離済み）**

診断ビルド全体実行時に 1 回のみ・単独実行 ×2 は PASS・非決定的。D117 の原因認定は MEM_SNAP + retire/destroy trace 0 発火 + 静的コード解析で成立しており、test 21 に依存しない。diagnostics instrumentation 由来か既存 race かは未特定（修正前に特定必須ではない）。

## 追加発見（DESIGN-B 実装時に整理が必要な契約曖昧性）

**Observe intent の epoch フィルタ（ObserveIntentHandler / drainObserveDeferred）**: `intent.epoch < currentEpoch → return（retire せず破棄）`（ProcessIntent.cpp:100-101）。publish が進むと observe intent は stale として**retire されずに捨てられる**意味論。OBSERVE-1 契約（「Timer → submitObserve → Coordinator が retirePublishedDSP を起動」）と実装の乖離の一部であり、Observe を retire authority として残すなら **stale intent でも retireByHandle（冪等・二重安全）を実行するべき**かの契約決定が必要。DESIGN-B 採用時は Observe を retire authority から外すため本件は保留可能（記録のみ）。

---

## D118-10 — GO 判定サマリ

| Gate | DESIGN-A | DESIGN-B（修正1・2付き） |
| --- | --- | --- |
| Authority（retire authority 1つ） | ✓（ただし identity 欠陥） | ✓（Observe を authority から外す） |
| Identity | **FAIL**（I-1 不成立・重複 publish で誤 retire） | ✓（ポインタ一次 + resolve 検証） |
| Generation | △ | ✓（stale 検出） |
| Epoch | ✓（observe intent が epoch 伝搬） | ✓（receipt epoch 伝搬・fallback は runtime epoch） |
| RT safety | ✓ | ✓（message thread / NonRT のみ・delete は router drain） |
| Queue（intent loss 契約） | △（epoch filter で喪失） | ✓（retire は loss-free: normal/fallback/emergency の全経路で発火） |
| Duplicate | ✓ | ✓（retireByHandle/retire は map erase 済みで冪等スキップ） |
| Receipt | — | ✓（single-slot 棄却時も fallback retire 発火） |
| Crossfade | **FAIL**（from/to identity 崩壊） | ✓ |
| Shutdown | ✓ | ✓（shutdown reclaim とは別経路のまま） |
| World | ✓（触れない） | ✓（触れない） |
| Phase-II | ✓ 導入せず | ✓ 導入せず |
| I4 DELETE-1/2/3 | ✓ | ✓ |

```text
D118

CAUSE-C1  fade completion ordering defect — CONFIRMED
DESIGN-A  read fading handle before endCrossfade — FAIL (identity race)
DESIGN-B  receipt-based pointer retire (retirePublishedDSP 接続) — PASS
          必須修正1: identity check を resolve(receipt.handle).instance == current に置換
          必須修正2: fadeCompleted CAS 直後に接続し submitObserve 経路を authority から外す
          実装時注意: mismatch 時 submitQuarantine の扱い（現行シーケンスでは常時 mismatch になるため）
DESIGN-C  dual retire path — REJECT (authority singularization 違反)
AUTHORITY selected retire authority = Timer fade-completion → DSPLifetimeManager::retire(DSPCore*) （ポインタ一次）
DELETE    DELETE-1 = retire（map erase + slot Retired）
          DELETE-2 = epoch/reclaim authorization（requestReclaimHandle / pendingReclaimHandles_ 再試行）
          DELETE-3 = physical destruction（enqueueWithRetry → drain → destroyDSPCoreNode）
IDENTITY  DSPCore* ↔ DSPHandle = CAS ポインタ一次 + resolve generation 検証で成立（A は不成立）
RECEIPT   single-slot safe = 重複時 storeReceipt 棄却 → fallback retire で漏出なし（epoch 伝搬のみ損失）
WORLD     independent / unresolved（D119+ で再測定）
TEST-21   separate blocker（D118-BLOCKER-1）
PHASE-II  unchanged（RecoveryEpisode 等は導入しない）
I4        DELETE-1/2/3 + authority contract 満たす（B のみ）
IMPLEMENTATION  0 files changed（production）
```

## 推奨

**D119（実装設計→実装）は DESIGN-B を基に**, 上記必須修正 1・2 と mismatch 時 quarantine 扱いの明確化を含む実装仕様を作成してから着手すること。D117 観測トレースは D119 の実装検証（destroy 発火確認）に再利用するため保持。

## 生成物

- 本ファイル（`evidence/D118_REPAIR_DESIGN_AUDIT.md`）
- コード変更: **0**（production）／ D117 trace 2 ファイルは macro-gated のまま維持
