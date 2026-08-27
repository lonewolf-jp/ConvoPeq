# D102-C2-5-D8-2-D — DSPLifetimeManager Caller Ownership Audit

- **実施日**: 2026-08-26
- **基準版**: ConvoPeq.md `Generated: 2026-08-26 20:06:47`（ローカルソース突合済み）
- **作業種別**: **read-only audit**（production source 変更 **0** / test 変更 **0** / contract 変更 **0**）
- **総合判定**: **PASS** — production 変更不要。ownership violation 0 件。

---

## D1. `DSPLifetimeManager::retire()` ownership trace

`DSPLifetimeManager.cpp:33-63`:

```text
dsp == nullptr → return                                    … 対象なし
engine_.retireDSPHandleForRuntime(dsp) == false → return   … 未登録/二重 retire 防止ゲート
router_ == nullptr → return                                 … 移送先不在（ptr は呼出し元の責務で保持継続）
epoch = publicationEpoch > 0 ? それ : router_->currentEpoch()
result = router_->enqueueWithRetry(dsp, &destroyDSPCoreNode, epoch, Generic)
ignoreUnused(result)                                        … C4/D8 参照
fetchAddAtomic(currentRetiringGeneration_, 1)               … dsp への以後のアクセスなし
```

- `retireDSPHandleForRuntime` は `runtimeDSPHandleMap_.erase(dsp)` を先に実行（`AudioEngine.h:4296`）。
  → **enqueueWithRetry より前に caller 側レジストリから ptr が除去される** = 再取得経路の遮断。
- enqueueWithRetry 後、caller は `dsp` を一切再利用しない（generation カウンタ更新のみ）。
- 二重 retire: 同一 ptr の 2 回目呼出しは map 不在で `false` → early return。構造的に不可能。

**判定: PASS**

---

## D2. `retireByHandle()` ownership trace

`DSPLifetimeManager.cpp:65-107`:

```text
handle.isNull() → return
lock(runtimeDSPHandleMapMutex_)
  findAndEraseByHandle(handle, rawKey) == false → toDelete = nullptr → return
     （既に erase 済み = 二重 retire・二重 delete の入口自体が存在しない）
unlock
dspHandleRuntime_.retire(handle)          … slot 状態 Active→Retired（物理削除なし）
requestReclaimHandle(handle)              … epoch 安全なら reclaimNormal → Reclared 化（物理削除なし）
enqueueWithRetry(toDelete, &destroyDSPCoreNode, ...)   … 物理削除の唯一の委譲点
ignoreUnused(result)
```

**重要確認 — `requestReclaimHandle` は物理削除を行わない:**
- `RuntimeIntentCoordinator::reclaimNormal`（`ISRRuntimePublicationCoordinator.cpp:677-706`）Step 3:
  ```cpp
  //    Reclaimed 状態への遷移のみ（物理削除は retire path の enqueueWithRetry が担当）
  handleRuntime.reclaim(handle);
  ```
- `DSPHandleRuntime::reclaim`（`ISRDSPHandle.cpp:129-148`）: `reg.instance = nullptr` + Reclaimed 化 +
  slot フリーリスト返却のみ。deleter 呼出しなし。stale handle / 二重 reclaim はガード済み。

→ **handle ライフサイクル（retire→reclaim）と物理破棄（destroyDSPCoreNode）の責務分離が明示的に実装されており、
reclaim 実行後も enqueueWithRetry の deleter が唯一の物理削除点として機能する。矛盾なし。**

**判定: PASS**

---

## D3. handle-map lifetime trace

| 時点 | runtimeDSPHandleMap_ | registry_ (DSPHandleRuntime) | 物理 DSPCore |
|---|---|---|---|
| activate/register | insert(dsp, handle) | Constructing→Active | alive |
| retire()/retireDSPHandleForRuntime | **erase(dsp)** | Retired → (epoch 安全なら) Reclaimed | alive（authority 内） |
| retireByHandle | **findAndEraseByHandle** | 同上 | alive（authority 内） |
| enqueueWithRetry deleter 実行時 | 不在（再解決不能） | Reclaimed（resolve 拒否） | destroyed |

- map erase は enqueueWithRetry **前** → deleter 実行後に resolve で ptr を再取得する経路が存在しない。
- resolve() は Quarantined/Reclaimed 状態を拒否（`ISRDSPHandle.h` resolve 契約）→ EBR 保護と整合。
- slot 回収（freelist push）と物理破棄は独立しており、slot 再利用が物理破棄を待つ必要がない
  （instance=nullptr 済みのため新世代が旧 ptr を参照しない）。

**判定: PASS** — erase timing と物理破棄の責務に矛盾なし。

---

## D4. `AudioEngine::destroyDSPCoreNode` caller audit

実装: `AudioEngine.Threading.cpp:17-22` — `~DSPCore()` + `aligned_free`。

全 production caller（rg 全走査）:

| # | Caller | 対象 population | 重複可能性 |
|---|---|---|---|
| 1 | `DSPLifetimeManager::retire()` deleter（cpp:50） | 登録済み・retire 完了 DSP | authority chain 内で exactly-once（B-2 T8 実証） |
| 2 | `DSPLifetimeManager::retireByHandle()` deleter（cpp:97） | 同上 | 同上 |
| 3 | `destroyRolledBackDSP`（cpp:123）← `RuntimePublicationOrchestrator.cpp:285` | publish **失敗** → Handle rollback済(Reclaimed)、map 不在 | #1/#2 と排他（下記） |
| 4 | `DSPGuard::~DSPGuard`（`RebuildDispatch.cpp:908-911`） | rebuild-obsolete（未 publish・未登録） | `retireDSPHandleForRuntime==false` 時のみ direct destroy |
| 5 | warmup failure 直接破棄（`RebuildDispatch.cpp:979 / 1052`） | 未コミット recovery DSP | 直後に `dspGuard.ptr = nullptr` |

**#3 の排他根拠**（`RuntimePublicationOrchestrator.cpp:279-285` コメント明記）:
publish 失敗時は commitRuntimePublication の ScopeExit が Handle を Reclaimed 化済み →
`retireDSPHandleForRuntime` は false を返す → lifetime_.retire() 経路に入らないため
`destroyRolledBackDSP()` での直接破棄が唯一の回収点。

**#4/#5 の排他根拠**: 未コミット DSP は registerDSPHandleForRuntime を経由しない
（DIAG ビルドでは `jassert(lookupDSPHandleForRuntime(ptr).isNull())` で表明 — RebuildDispatch:900-907）。

**判定: PASS** — 5 caller は互いに素な population を担当し、重複到達構造がない。

---

## D5. double-delete audit

```text
destroyDSPCoreNode 実行点の partition:
  ├─ authority chain (D/Q/E/T deleter)   … ptr は map から erase 済み → rollback/guard 経路から到達不可
  ├─ rollback path (#3)                   … map 不在が前提 → retire 経路の enqueue 前提と矛盾しない
  └─ uncommitted path (#4/#5)             … 未登録 → authority chain に一度も入っていない

同一 ptr が 2 系統に属する条件:
  「map 在籍かつ authority 投入済み」で rollback/guard に到達 → 存在しない
  （rollback は publish 失敗=未 commit=未 enqueue、guard は commit 前 ptr を保持後即 null 化）
```

補足観察（violation ではなく hardening メモ）:
`DSPGuard` dtor の `retireDSPHandleForRuntime()==true` 分岐（RebuildDispatch:908）は、
true 時に enqueue も direct destroy も行わない。ただしこの分岐は
「コミット済み ptr は commit 前に guard から null 化される」構造不変条件により到達不能であり、
DIAG jassert が逸脱を検知する。将来 guard の扱いを変える際は本不変条件の維持が必要。

**判定: PASS**

---

## D6. double-quarantine audit

`retire()` / `retireByHandle()` の enqueueWithRetry 以降の全文:

```cpp
juce::ignoreUnused(result);
convo::fetchAddAtomic(currentRetiringGeneration_, 1, ...);
// 関数終了
```

- 追加の `quarantineRetire` / `emergencyQuarantine` / `terminalReclaim` / `shutdownReclaim` 呼出し: **0 件**
- コメント（cpp:53-57, 100-101）が「二重移送（double-quarantine → double-free）を避けるため追加処置なし」と契約を明記
- 他の Router API caller 走査でも SnapshotCoordinator（C3）以外に caller 側追加 quarantine なし

**判定: PASS**

---

## D7. shutdown overlap audit

1. **DSPLifetimeManager に destructor 定義なし**（ヘッダ確認）— 暗黙 dtor のみ、再退避なし。
2. **shutdown 時の retire 経路**: `AudioEngine.CtorDtor.cpp:176-192`（dtor）および
   `ReleaseResources.cpp:135,330-337` とも `lifetimeMgr.retire(...)` を使用。
   単一引数 ctor は `router_(engine_.m_retireRouter.get())` を設定（cpp:4-8）するため、
   shutdown 時も正式な `enqueueWithRetry` → D→Q→E→T 経路で完結。
3. **shutdownReclaim との重複なし**: `isShutdownInProgress()` → `shutdownReclaim` ブランチは
   `AudioEngine::enqueueDeferredDeleteNonRtWithResult`（AudioEngine.h:4212-4222）にのみ存在し、
   DSPLifetimeManager はこの API を通らない。両者は入口 API が異なり、同一 ptr が両方に進むことはない。
4. **drainDeferredRetireQueues(true)**（ReleaseResources:341）は `pendingReclaimHandles_`
   （slot 状態の保留再試行）のみを扱う（`AudioEngine.Retire.cpp:71-91`）。物理削除の再試行ではない。

**判定: PASS** — shutdown と通常 retire の重複経路なし、再退避なし。

---

## D8. `ignoreUnused(result)` contract audit

前提（D8-2-C C1/C2 確定）: `enqueueWithRetry` は
Success→D / QueuePressure→Q or E / TerminalReclaim→T-or-immediate-delete のいずれかで
**Router 内に ownership を残して return する**。無所有権 return 経路は存在しない。

したがって:

```text
enqueueWithRetry()
        │
        ├─ Success          → D owns ..................... caller ownership = 0 ✅
        ├─ QueuePressure    → Q or E owns ............... caller ownership = 0 ✅
        └─ TerminalReclaim  → T owns / immediate delete .. caller ownership = 0 ✅
                              ↓
                    ignoreUnused(result) は orphan を意味しない
```

- `QueueFull`: 現行 production 到達不能（C1/C2）
- `Shutdown`: enqueueWithRetry の disposition ではない（C6）
- result を quarantineRetire() への置換・追加対応は不要かつ有害（double-quarantine リスク）

**判定: PASS** — `ignoreUnused(result)` は現行契約下で正しい。修正対象外を維持。

---

## Final Verdict

### 4 条件チェック

| 条件 | PASS 基準 | 判定 |
|---|---|---|
| transfer 後の再利用 | caller が DSP ptr を再利用しない | ✅ PASS（D1/D2 — generation 更新のみ） |
| double quarantine | ignoreUnused 後の別 retire/quarantine なし | ✅ PASS（D6 — 呼出し 0 件） |
| double delete | destroyDSPCoreNode が複数 authority から呼ばれない | ✅ PASS（D4/D5 — 5 caller が互いに素） |
| handle ownership | map erase と物理破棄の責務が矛盾しない | ✅ PASS（D3 — reclaim は状態遷移のみ、物理削除は deleter 専任） |

### 判定: **PASS**

- production source 変更: **不要**
- violation: **0 件**
- 観察事項（non-blocking）: DSPGuard dtor true 分岐の理論 leak は構造不変条件 + DIAG jassert で封止済み
  （将来 guard 扱い変更時の注意点として本報告に記録）

### D8-2 進行状況

```text
D8-2-B-2  Configurable Provider + T1/T2/T3/T4/T7/T8      PASS
D8-2-C     Disposition Contract Audit                     PASS 9/9
D8-2-D     DSPLifetimeManager Caller Ownership Audit      PASS ★ 本監査
              ↓
        D2-1 Bounded Terminal Design Gate（design-only）へ進行可能
```
