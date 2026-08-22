# Phase I-T1-D101-1R2 — Published-Domain Re-audit

> **Verdict: D101 #1 = CLOSED / INV-WORLD-TYPE = RESTORED / INV-PUB-3 = RESTORED**
> **Residual Gap: G1 (rejected negative dynamic test 未実装 — 静的は完全)**
> **次: D101 #2（reference observer completeness / λ / τ_b / G / M=f(...)）へ進行可**
> **R / R_cap / M / B_max^true = UNDETERMINED 維持 / T2 = NO-GO / measurement run 追加 なし**
> コード変更 = 0（本 re-audit は監査のみ）。RunningTime 削除は D101 #1 とは独立した変更として分離して扱う。

## 0. 前提 — D101-1F との分離

| 変更 | D101 #1 への扱い |
|---|---|
| P2 `World → Generic`（ReleaseResources.cpp:557） | **D101 #1 closure 対象** ✅ 本 evidence の判定根拠 |
| RunningTime 計測削除（RuntimePublishWorld / commitRunningTimeNowUs / Sleep 削除） | **別変更** — D101 #1 の provenance proof とは無関係。監査証跡を分離して扱う（本 evidence では D101 #1 PASS 理由に含めない） |
| telemetry authority / counter / releaseObserved writer | 変更なし（C5/C6 で確定・再確認） |
| T1 measurement protocol | 変更なし（T1-MR CLOSED 値: A_max_candidate=1 / T_w peak=2 / max(E_w)=1 を維持） |

## 1. D101-1R2-1 — `DeletionEntryType::World` producer census

`src/` 全域（tests 除外）を機械的列挙:

| 区分 | 結果 | 判定 |
|---|---|---|
| `DeletionEntryType::World` を生成する **enqueue site** | **1** — `AudioEngine.h:3548`（`retirePublishedRuntimeWorldNonRt` 内 `enqueueDeferredDeleteNonRt(..., World)`）のみ | ✅ |
| `DeletionEntryType::World` を **消費する** site（`if (type==World)`） | 7 — `ISRRetireRouter.cpp:68,89,504` / `RetireQuarantineStore.h:140,177` / `DeferredDeletionQueue.h:148,199` / `IRetireProvider.h:34` comment | 消費側（退避・counter 分岐） |
| `retirePublishedRuntimeWorldNonRt` 定義 | `AudioEngine.h:3536` — `World` enqueue の唯一生産者（PRECONDITION: W ∈ PublishedDomain） | ✅ |
| `retireRejectedRuntimeWorldNonRt` 定義 | `AudioEngine.h:3552` — `Generic` enqueue（PRECONDITION: W ∉ PublishedDomain） | ✅ |
| P2 residual drain → World | **0**（Generic に反転済み: ReleaseResources.cpp:557） | ✅ |
| rejectedWorld → World | **0** | ✅ |
| rejectedWorld → Generic | 1 経路（Coordinator.h:123 / Init.cpp:67 + AudioEngine.h:3564） | ✅ |

**合格条件 `World producer = 1 = retirePublishedRuntimeWorldNonRt のみ` を満たす。**

## 2. D101-1R2-2 — P1 Published provenance 再確認

`retirePublishedRuntimeWorldNonRt(W)` の全 production caller（6 系統）と chain:

| # | Caller | publish authority | publishAndSwap | oldWorld → retire |
|---|---|---|---|---|
| C1 | `RuntimePublicationCoordinator::clearPublishedRuntimeSnapshotsNonRt` (Coordinator.h:97-98) | RuntimeWorldAuthority | `publishAndSwap(nullptr)` | clearedWorld → World ✅ |
| C2 | `RuntimePublicationCoordinator::publishWorld` 成功時 (Coordinator.h:130,147) | RuntimeWorldAuthority | `publishAndSwap(newWorld)` | oldWorld → World ✅ |
| C3 | `RuntimePublishExecutor` async 通常 publish (RuntimePublishExecutor.h:76) | RuntimeWorldAuthority | `authority.publish(...)` 内部 publishAndSwap | oldWorld → World ✅（committed==true 時のみ） |
| C4 | bootstrap publish (AudioEngine.Init.cpp:73-88) | RuntimeWorldAuthority | `authority.publish(...)` | oldWorld → World ✅ |
| C5 | dtor shutdown clear (AudioEngine.CtorDtor.cpp:229-234) | RuntimeWorldAuthority | `clearPublishedRuntimeSnapshotsNonRt()` → publishAndSwap(nullptr) | clearedWorld → World ✅ |
| C6 | releaseResources shutdown clear (ReleaseResources.cpp:455-460) | RuntimeWorldAuthority | 同上 | clearedWorld → World ✅ |

全 caller について **execution boundary = `RuntimeWorldAuthority` 側の semantic publication transaction（publishAndSwap を束ねる唯一境界・work88 X4-B §6.4）**を通過し、oldWorld は必ず `publishAndSwap` 通過後の world（PublishedDomain 所属）を retire する。**`W ∈ PublishedDomain` を全 caller で証明。**

## 3. D101-1R2-3 — P2 residual 再確認

```
OwnerChannel (pre-publication transport: commit → enqueue / executePublish → take)
    ↓ drainAllNonRt()（shutdown quiescence 後に全残留を走査・single-transfer consume→publish(nullptr)）
    ↓ residual owner（take 未経験 = publishAndSwap 未通過 = W ∉ PublishedDomain）
    ↓ DeletionEntryType::Generic（ReleaseResources.cpp:557）
    ↓ DeferredDeletionQueue → reclaim → deleter（破壊のみ）
```

- **Generic に反転済み**（D101-1F の 1 行修正）を再確認（sed 532-560）。
- World telemetry chain への再流入別経路の有無: residual owner は Generic として enqueue されるため `if (entryType==World)` 分岐（ISRRetireRouter/RetireQuarantineStore/DeferredDeletionQueue の 7 consumer）を不通過 → `worldReclaimCount++` / `referenceReleaseCount` / `releaseObserved` のいずれにも流入しない。**source graph 上で非到達を確認。**

## 4. D101-1R2-4 — rejected path negative proof

### Static（確認可能）

```
validation fail (RuntimePublicationValidator)
  → RuntimePublicationCoordinator::publishWorld 内 `worldOwner.release()` → `retireRejectedRuntimeWorldNonRt(rejectedWorld)` (Coordinator.h:122-123)
  → AudioEngine.h:3564 `Generic` enqueue
  → World branch 不通過（退避・counter 分岐の World 条件に該当しない）
```

**Static: PASS**（現行ソース上確定）。

### Dynamic（未実装 — G1 として正確に分類）

```
rejected World 発生前後で referenceReleaseCount_before == after
                     worldReclaimCount_before    == after
```

は **dynamic test 0 件**（既存 PartialPublicationRejectTests は TestWorld + test-local bridge で reject/published 分岐の静的等価検証を担うが、実 AudioEngine telemetry への計数不変 assert はない）。**G1 residual として明示し、G0 に昇格させない**（指示どおり）。D101 #1 自体を OPEN に戻す必要はない — 主要 counterexample（P2 residual World）は構造的分離で消滅しているため。

## 5. D101-1R2-5 — exactly-once chain 再監査

既存 G2-1 証明を D101-1F 変更が壊していないことを確認:

```
Published World → World entry (retirePublished のみ) → reclaim / terminal (9 LP)
  → worldReclaimCount++（1:1・同一 if(type==World) block 内）
  → onRelease() → referenceReleaseCount
  → shared transfer (transferWorldReclaimDeltaForTelemetry + runWorldRetirementMeasurementStep)
  → releaseObserved
```

| 項目 | 判定 |
|---|---|
| World writer = 1 | ✅（§1） |
| `onRelease` の World branch = 既存 9 LP のみ | ✅（consumer 7 sites は World 分岐で counter/recovery を 1:1 で実行・変更なし） |
| `releaseObserved` writer = 既存 shared transfer のみ | ✅（G2-1/G2-2 で閉鎖: transferWorldReclaimDeltaForTelemetry 内 1 系統 + runWorldRetirementMeasurementStep 内共有） |
| reference observer が `releaseObserved` を直接増加させない | ✅（R3 確定: Reference.h 内 onReleaseObserved 0） |
| 新しい counter / authority = 0 | ✅（新 authority 作成なし） |

## 6. D101-1R2-6 — `observedOutstanding = A - R` 再確認

- Generic 化により `W ∉ PublishedDomain` の破棄（residual / rejected）が `R` に流入しないことが成立（§3, §4）。
- よって `A = acquireObserved`（publish 成功時のみ +1）/ `R = releaseObserved`（Published World のみ計数）の **共に Published domain で閉じた accounting** が維持される。
- 既存 G2-1 regression（`Δreclaim == Δreference == Δrelease == N` / `outstanding = A - R` / second sampler +0）は全 PASS を維持（measurement=all 7/7 PASS 時点）。

## 7. D101-1R2-7 — T1-MR 結果の非再測定

指示どおり **再測定しない**:

| 項目 | 値 | 扱い |
|---|---|---|
| `A_max_candidate(sampled)` | 1 | 維持（現在の protocol で観測された sampled outstanding の最大候補） |
| `supplementary T_w peak` | 2 | 維持 |
| `max(E_w) observed` | 1 | 維持（`M = max(E_w)` への短絡は NO-GO） |
| `R / R_cap / M / B_max^true` | UNDETERMINED | 維持（D101 #2 で M の数学的 bound が導出されるまで） |

## 8. 静的 invariant 10 項目（D101-1F Step 9 再掲 + 再確認）

| Invariant | 期待 | 実測 |
|---|---|---|
| World producer | 1 | 1（P1 のみ） |
| P1 = retirePublishedRuntimeWorldNonRt | 1 | 1 |
| P2 residual direct World enqueue | 0 | 0 |
| rejected → World | 0 | 0 |
| rejected → Generic | 1 | 1（AudioEngine.h:3564） |
| OwnerChannel residual → World | 0 | 0 |
| OwnerChannel residual → Generic | 1 | 1（ReleaseResources.cpp:557） |
| new telemetry authority | 0 | 0 |
| new counter | 0 | 0 |
| releaseObserved new writer | 0 | 0 |

## 9. D101-1R2-10 — rejected negative dynamic test の分類

**Static: PASS / Dynamic: absent → G1 residual として明示**。D101 #1 自体を OPEN に戻さない。

## 10. 最終 Gate（10 項目）

| Gate | PASS 条件 | 判定 |
|---|---|---|
| D101-1R2-1 | `DeletionEntryType::World` producer = 1 | ✅ |
| D101-1R2-2 | 全 World caller の Published provenance 証明 | ✅（6 caller 全て publishAndSwap 通過） |
| D101-1R2-3 | OwnerChannel residual → World = 0 | ✅ |
| D101-1R2-4 | rejected → World = 0 | ✅ |
| D101-1R2-5 | rejected → Generic = 1 production path | ✅ |
| D101-1R2-6 | World → wc → onRelease exactly-once chain 維持 | ✅ |
| D101-1R2-7 | 新 telemetry authority/counter/writer = 0 | ✅ |
| D101-1R2-8 | `A-R` semantic unchanged | ✅ |
| D101-1R2-9 | T1-MR result unchanged | ✅（A_max=1 / T_w=2 / E_w=1 維持） |
| D101-1R2-10 | rejected negative dynamic test の有無を正確に分類 | ✅（absent → G1 residual） |

```
World producer = 1
AND 全 World producer が PublishedDomain
AND residual/rejected → World = 0
AND existing G2 chain unchanged
```

を全て満たすため:

```
D101 #1 = CLOSED
INV-WORLD-TYPE = RESTORED
INV-PUB-3 = RESTORED
```

## 11. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | World producer census / retire API 全件 / enqueue/drain 全件を複数手段で一致 |
| Sandbox | node fs walk（src 全 .cpp/.h・World/Generic / retire* / ownerChannel / drainAllNonRt census）/ source 直読（AudioEngine.h:3536-3565・ReleaseResources.cpp:532-560・Coordinator.h:85-155・OwnerChannel.h 全文 7030b・CtorDtor.cpp:214-245・Init.cpp:49-95） | provenance 構造を確定 |
| MCP | serena: 一時無効（代替として rg/sg/node で補完。serena は前 Gate までに多数のシンボル参照を確定済み）| — |
| MCP | AiDex: 一時無効（代替として node walk + rg census で補完。AiDex は C6-Final までに索引レベルで `releaseObserved` 0 hits 等を確定済み） | — |
| CLI | ccc 0.45.2 / graphify 0.9.48（exe 存在確認・graphify-out 未生成のため rg 補完）/ semble 0.5.5 | 検索実行（World producer 等） |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK（Vyukov 1024cores SSL 失効 → rigtorp fallback 明記・crossbeam 404 は正規 URL `docs.rs/crossbeam-epoch` で 200） |

## 12. 次作業

```
D101 #1 CLOSED
    ↓
D101 #2 — Reference Observer Completeness / λ / τ_b / G / M = f(G, λ, τ_b, ...)
    ↓
B_max^true ≤ O_w + M の structural proof
    ↓
M-bound成立 ? → NO: M/R/R_cap UNDETERMINED 継続 / YES: D102 / R determination review
```

**まだ決定しないもの:** `R / R_cap / M / B_max^true / T2 / Reservation gate`。

---

*Evidence generated: Phase I-T1-D101-1R2 — no code change. D101 #1 CLOSED（P2 Generic 化により Published-domain authority 回復。RunningTime は別変更として分離）。*
