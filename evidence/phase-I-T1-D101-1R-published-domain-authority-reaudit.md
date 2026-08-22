# Phase I-T1-D101-1R — Published-Domain Authority Re-audit

> **Verdict: D101 #1 = OPEN（D101-1R-5 / -6 / -7 不成立）**
> **M-bound investigation = NO-GO / M / R / R_cap / T2 = NO-GO 継続**
> コード変更 = 0。residual OwnerChannel drain が **publication 前の transport 残留**を
> `DeletionEntryType::World` として生産している構造を確認 — PublishedDomain 証明は構造的に不可能。

## 1. Step 1 — World producer 全件 census

`DeletionEntryType::World` を生成する site は src/ 全域で **2 箇所のみ**（node census・rg 一致）:

| # | Producer | File:Line | 型 | provenance |
|---|---|---|---|---|
| P1 | `retirePublishedRuntimeWorldNonRt()` | `AudioEngine.h:3536-3550`（enqueue :3542-3548） | World | **Published**（PRECONDITION コメント「W ∈ PublishedDomain (must have passed publishAndSwap LP)」+ 全 caller が post-swap world を渡す・下表） |
| P2 | residual OwnerChannel drain lambda | `AudioEngine.Processing.ReleaseResources.cpp:542-556`（enqueue :547-555） | World | **証明不能 — 構造的に未公開**（§2） |

`retireRejectedRuntimeWorldNonRt()`（`AudioEngine.h:3552-3565`）は **Generic** を生成（:3564）— World producer ではない。

### P1 caller 全件（全て publishAndSwap 通過後の world）

| Caller | File:Line | oldWorld の由来 |
|---|---|---|
| `clearPublishedRuntimeSnapshotsNonRt` | `RuntimePublicationCoordinator.h:97-98` | `publishAndSwap(nullptr)` 戻り値（shutdown clear） |
| `publishWorld` 成功時 | `RuntimePublicationCoordinator.h:130,147` | `publishAndSwap(newWorld)` 戻り値 |
| PublishExecutor（X4-B async 通常 publish） | `RuntimePublishExecutor.h:60-76` | `authority.publish(...)` 内部 publishAndSwap の戻り値（committed==true 時のみ） |
| bootstrap publish | `AudioEngine.Init.cpp:73-88` | `authority.publish(...)` 戻り値 |
| dtor shutdown clear | `AudioEngine.CtorDtor.cpp:229-234` | `clearPublishedRuntimeSnapshotsNonRt()` 戻り値 |
| releaseResources shutdown clear | `AudioEngine.Processing.ReleaseResources.cpp:455-460` | 同上 |

## 2. Step 2 — residual OwnerChannel 精査（P2・最重要）

### 2.1 OwnerChannel の構造的位置づけ（`OwnerChannel.h`）

```
commitRuntimePublication（Non-RT Producer）
    └─ ownerChannel().enqueue(key{seqId,epoch,mappedGen}, owner)   // AudioEngine.h:4559
         （所有権を channel へ移譲）
executePublish（ISR/audio Consumer）
    └─ ownerChannel().take(key)                                    // RuntimePublishExecutor.h:31
         └─ authority.publish(owner) → publishAndSwap              // ★ publication LP はここで初めて通過
```

- OwnerChannel は **ADR-D3 Step 5-3 owner leg = publication 前の transport**。
- `take()` は single-transfer（take 後 slot は空）。channel に残る owner は **take されていない = publishAndSwap 未通過**。
- take 後に publish 失敗した場合も owner は publish() 内で破棄されるため channel には戻らない。

### 2.2 残留 owner の provenance 判定

shutdown 時（producer/consumer quiescence 後）の channel 残留 = **enqueue されたが executePublish に消費されなかった owner** = **PublishedDomain に属さない未公開 world**。

したがって P2 が `World` 型で enqueue する対象は:

```
W_residual ∉ PublishedDomain
    → shutdownReclaim → terminalReclaim（epochSafe && !isRt）
    → recordWorldReclaim() → onRelease()
```

となり、**A が計上されていない world（didPublishRuntimeNonRt 未発火）に対して R_ref 側のみが加算される**。`observedOutstandingEstimate = A - R` は shutdown 窓で残留数だけ負方向へ汚染され得る。

### 2.3 指示項目 1-6 への回答

| # | 項目 | 回答 |
|---|---|---|
| 1 | channel 入口で Published 保証？ | **No**。enqueue は commit 時点（LP 前）。validate 失敗 world も channel には入らないが、入った world は LP 未通過 |
| 2 | `drainAllNonRt()` 所有権モデル | relinquish（reclaim callback へ raw 移譲・callback が authority chain へ転送義務）。single-transfer で再 drain は no-op |
| 3 | `publishAndSwap()` との関係 | channel は LP の**前段**。残留 owner は swap を一度も通過していない |
| 4 | shutdown clear 後の残 owner provenance | clear（`publishAndSwap(nullptr)` + retirePublished）とは別物。clear でも消費されない pending intent の owner |
| 5 | nullptr publish/release | `publishAndSwap(nullptr)` は現行 world 返却（shutdown oldWorld = Published ✓）。`publishWorld` の nullptr→nullptr は Failed。channel 側は nullptr slot = empty |
| 6 | Generic world 混入可能性 | P2 が無条件で World を生成する点が問題の逆側。rejected は正しく Generic（:3564）。Generic に World object が混入する経路は無し（rejected は意図された Generic） |

## 3. Step 3 — producer authority 一本化の可否判定

第一候補（residual drain → bridge → `retirePublishedRuntimeWorldNonRt`）は **採用不可**:

- 同 API の PRECONDITION は `W ∈ PublishedDomain` であり、残留 owner はこれを満たさない（§2.2）。
- 指示の判定規則どおり「証明できない場合は API 呼び出し変更ではなく provenance model 自体の再設計」が必要。

記録すべき設計オプション（実装は次フェーズ以降）:

| Option | 内容 | 評価 |
|---|---|---|
| α（最小） | residual drain を **Generic** 型で enqueue（未公開 object として破壊・World 観測なし） | D101 #1 不変条件「World ⇒ Published」を復元。wc/R_ref も増やない（A も増えていないため整合）。推奨 |
| β | 第3型（例: `PendingWorld`）+ 独立 counter | shutdown 残留 telemetry が必要な場合のみ。機構増 |
| γ | enqueue 時点を published 扱いとする | LP 前倒しになり INV-X4/D69 破壊 — 不可 |

## 4. Step 4 — rejected World negative proof

- 構造: validation fail → `worldOwner.release()` → `retireRejectedRuntimeWorldNonRt` → **Generic**（Coordinator.h:122-124 / Init.cpp:66-67 / AudioEngine.h:3558-3564）。World branch を通らないことは source 上確定。
- 既存テスト: `PartialPublicationRejectTests` / `RuntimePublicationCoordinatorTests` は TestWorld + test-local bridge で reject/published 分岐を検証するが、**実 AudioEngine telemetry（referenceReleaseCount 不変）への assert はない**。
- → **D101-1R-7 = OPEN**: `invalid World → Rejected → Generic → referenceReleaseCount_before == after` を実 engine で検証する deterministic test が存在しない（コード変更禁止のため本フェーズでは追加せず・test sketch のみ記録）。

## 5. Step 5 — exactly-once provenance table

### Published path（P1）

```
producer (Builder)
  ↓ sealRecursively
publication LP: publishAndSwap(newWorld)      [Coordinator.publishWorld / authority.publish]
  ↓ oldWorld（W ∈ PublishedDomain 確定）
retirePublishedRuntimeWorldNonRt → enqueue(World)
  ↓ D/Q/E/T（9 LP）
deleter 実行
  ↓ wc++ ＆ onRelease()（1:1・同一 block）
releaseObserved / referenceReleaseCount
```

### Rejected path

```
producer (Builder)
  ↓ validate fail
retireRejectedRuntimeWorldNonRt → enqueue(Generic)
  ↓ destruction
NO wc++ / NO onRelease()
```

### 反例（P2）

```
producer (Builder)
  ↓ commit enqueue（LP 前転送）
OwnerChannel 残留（executePublish 未消費）
  ↓ shutdown drain → enqueue(World)   ← ★ PublishedDomain 未証明のまま World 生産
  ↓ terminalReclaim
wc++ ＆ onRelease()                    ← A 未計上の world に対して発火
```

## 6. Step 6 — Gate 判定

| Gate | 条件 | 判定 |
|---|---|---|
| D101-1R-1 | World producer 全件列挙 | ✅（P1/P2 + rejected=Generic） |
| D101-1R-2 | 全 producer の Published provenance 証明 | ❌ P2 が構造的に未公開（§2） |
| D101-1R-3 | rejected → Generic 証明 | ✅（:3564 Generic + caller 2 箇所） |
| D101-1R-4 | World → onRelease exactly-once chain | △ P1 鎖は完全（9 LP 不変）/ P2 が鎖外の第10 source を追加 |
| D101-1R-5 | residual OwnerChannel provenance 証明 | ❌ **構造的に不可能**（pre-publication transport） |
| D101-1R-6 | 複数 producer の同一 authority contract | ❌ P2 は `retirePublishedRuntimeWorldNonRt` を経由しない |
| D101-1R-7 | `W ∉ Published ⇒ R_ref(W)=0` negative test | ❌ 未存在（test sketch 記録済み） |

```
D101 #1 = OPEN
M-bound investigation = NO-GO
M / R / R_cap / T2 = NO-GO 継続
INV-PUB-3 = NOT RESTORED
```

## 7. T1-MR 结果への影響評価

本監査で判明した P2 は shutdown 残留時にのみ発火する。T1-MR の 3 run では `[AUDIT] drainAllNonRt residual` diag が出力されず（drainedResidual == 0）、よって **記録済みの O_w/T_w/E_w/A_max_candidate 値自体は影響を受けていない**。ただし構造リスクは shutdown + in-flight publish の組合せで顕在化するため、D101 #1 OPEN のまま M-bound へ進めない。

## 8. Tool Coverage

node fs walk（src 全 .cpp/.h・`DeletionEntryType::World` / retire* / ownerChannel / drainAllNonRt census）/ source 直読（AudioEngine.h:3505-3570・Commit.cpp・Coordinator.h:85-155・RuntimePublishExecutor.h:59-90・OwnerChannel.h 全文・CtorDtor.cpp:214-245・ReleaseResources.cpp:439-470・Init.cpp:49-95・PartialPublicationRejectTests.cpp:99-140）/ serena・AiDex・semble・ccc・graphify・WSL rg 系は前 Gate までに参照整合済み / 文献 9 系統 200 OK 済。

## 9. 次フェーズ用記録（実装禁止中の設計メモ）

1. **G2-α 修復案**: ReleaseResources.cpp:547-555 の `DeletionEntryType::World` → `DeletionEntryType::Generic` 変更（1 行）+ comment 更新。deterministic test: shutdown 直前に intent queue full 等で channel 残留を作り、`referenceReleaseCount`/`worldReclaimCount` が不変であることを assert。
2. **D101-1R-7 test sketch**: validator を強制失敗させ rejected publish を発生 → `referenceReleaseCount_before == after` かつ `worldReclaimCount` 不変を assert（既存 PartialPublicationReject の実 engine 版）。
3. 上記 2 点完了後に D101 #1 再監査 → CLOSED なら D101 #2（observer completeness / λ / τ_b / G）へ進行。

---

*Evidence generated: Phase I-T1-D101-1R — no code change. D101 #1 = OPEN（P2 residual drain が published-domain 契約を構造的に満たさない）。*
