# D170-1 — Preflight Source Audit（pointer retire 廃止の安全性実証）

```text
D170-1 — Preflight source audit for D170-2 minimal implementation
Date:     2026-09-07
Type:     read-only source audit（production source 変更 0）
Contract: evidence/D169/D169_1R_REPAIR_CONTRACT.md（RC-D169-1-1〜5）
Method:   src 全走査（Node script）+ 対象 region 直接読取
Verdict:  **GO — pointer-value retire の完全廃止は安全**
```

---

## Q1. 各 pointer slot に対応する authoritative handle は存在するか

### activeRuntimeDSPSlot

- writer は **prepareToPlay の placeholder 作成のみ**（`PrepareToPlay.cpp:270` setActiveRuntimeDSP / `:301` rollback 時 null）。
- placeholder は `commitRuntimePublication(std::move(worldOwner), RegistrationContext::needsRegistration(placeholderRaw), DSPHandle::null())`（PrepareToPlay.cpp:283-285）で
  registration → `dspHandleRuntime_.activate(newHandle)`（`DSPTransition.h:71/104`）→
  `activeRuntimeDSPHandle_` 公開（`ISRDSPHandle.cpp:100`）。
- **結論: pointer slot の DSP は必ず handle registry に在り、handle 系で disposition 可能。**

### fadingRuntimeDSPSlot

- fading は publish commit の crossfade 時に `beginCrossfade(oldHandle, newHandle)`（`DSPTransition.h:119`）
  → `fadingRuntimeDSPHandle_ = from`（`ISRDSPHandle.cpp:90`）で handle 側も追跡される。
- terminal は `getFadingRuntimeDSPHandle()`（ReleaseResources.cpp:467）を handle 系入力に使用済み。
- **結論: handle 系でカバー済み。**

### pendingNewToRelease

- **src 全走査で代入箇所 0 件**（宣言 ReleaseResources.cpp:153 のみ）。
  常に nullptr → `:356 if (pendingNewToRelease)` は恒常不発の死分岐。
- **結論: 廃止は無影響。**

### pendingCurrentToRelease（pendingTask.currentDSP）

- `RebuildTask::currentDSP`（AudioEngine.h:2766）の非 null writer は **src 全体に 0 件**
  （唯一の値設定は `RebuildDispatch.cpp:602 task.currentDSP = nullptr`）。
  残 13 site は null 代入 / copy（`task = pendingTask`, `pendingTask = task`）／読み取りのみ。
- CtorDtor.cpp:172 の「worker 側の未コミット生成物」コメントは歴史的経緯の名残で、
  現行 source では DSPCore 生成が task に載らないため死変数。
- **結論: 廃止は無影響。**

## Q2. terminal 時点で handle が既に retire 済みになり得るか / Q3. MISS は idempotent か

- `dspHandleRuntime_.retire(handle)` は `registry_[slot].state → Retired` の無条件 publish（`ISRDSPHandle.cpp:122-127`）＝冪等。
- `resolveDSPHandle` は Retired を valid（instance あり）で返す（`ISRDSPHandle.cpp:73-78` — Reclaimed/Quarantined のみ nullptr）→
  **Retired 後も V-D-b authority retire が到達可能**。
- `DSPLifetimeManager::retireByHandle` は `findAndEraseByHandle` MISS 時に log のみの no-op
  （`DSPLifetimeManager.cpp:95-103`）＝**正常な冪等 MISS**。
- `DSPLifetimeManager::retire(void*)`（pointer 系の実体）→ `retireDSPHandleForRuntime(DSPCore*)` は
  `runtimeDSPHandleMap_.find(dsp)` — **raw pointer key lookup が defect 本体**（AudioEngine.h:4394）。
- **結論: handle 系は全経路冪等。二重 retire は構造的に不可（INV-D162-3）。**

## Q4. pointer 側を削除すると未処理 DSP が残るか

残存 disposition 経路の網羅確認:

| 経路 | 位置 | 対象 |
| --- | --- | --- |
| rebuild publish old-handle retire intent | CoordinatorLoop executePublish tail（DSPLifetimeManager::retire(oldDSP)） | in-flight rebuild の旧 DSP |
| graceful drain + OverflowRing 再注入 | ReleaseResources.cpp:265-348 | retire queue 残 |
| handle retire + quiescent reclaim | ReleaseResources.cpp:466-505 | active/fading final |
| V-D-b authority retire | ReleaseResources.cpp:556-576 | resolve 成功 DSP（EBR→destroy） |
| deferred slot clear (S3) | ReleaseResources.cpp:587-588 | deferred publish slot |
| pendingReclaimHandles 再試行 | drainDeferredRetireQueues | epoch 未回収 handle |
| PR2 quarantine drain | ReleaseResources.cpp:419-460 | quarantine slot 全件 |
| dtor E-2 + CtorDtor drain | CtorDtor.cpp:204-214 | shutdown 残 active/fading handle |
| waitForDrain + D5/D8 | ReleaseResources.cpp:597 / dtor body | EBR 残 destroy |

pointer-value retire（:352-359）が唯一 owner であるケースは **0 件**
（Q1 の通り全 pointer slot DSP は registration 済み＝handle 系対象）。

## 追加確定事項

- `lifetimeForShutdown`（:157）は pointer retire 専用のローカル — 廃止後は生成ごと削除可能。
- capture 変数 `activeToRelease` / `fadingToRelease` は slot clear（:172-188）の
  観測・validateDistinctRuntimeSlots 用途として維持し `juce::ignoreUnused` する（dtor E-2 前例準拠）。
- `pendingNewToRelease` / `pendingCurrentToRelease` / `pendingTask` consume（:191-198）は
  **hasPendingTask/publishRetryReady の clear 副作用が実意味**のため維持（RC-D169-1-2・RC-D169-1-4 準拠）。

## Verdict

**GO。** D170-2 の最小修復は「ReleaseResources.cpp:352-359 の 4 連 pointer retire と
lifetimeForShutdown の削除」のみで成立し、RC-D169-1-5 の禁止修復は一切不要。
