# D169-1R — Repair Contract Approval（RC-D169-1-1〜5 契約固定）

```text
D169-1R — In-flight Rebuild × Terminal Shutdown Race Repair Contract
Date:        2026-09-07
Type:        read-only / contract freeze（production source 変更 0）
Prior audit: evidence/D169/D169_1_INFLIGHT_REBUILD_TERMINAL_RACE_AUDIT.md（Case A — Defect confirmed）
Scope:       D170 実装（race repair implementation）への拘束力ある契約
Status:      **APPROVED — 契約固定**
```

---

## 0. 契約の根拠

D169-1 が source-level で確定した defect:

> terminal `releaseResources()` が同一 registered DSPCore に対し
> **handle-based retire（system A・epoch 保護・正）** と
> **pointer-value retire（system B・`activeRuntimeDSPSlot` 等の legacy slot 値を
> `retireDSPHandleForRuntime(DSPCore*)` の raw-pointer map key に使用）** の
> 二重 destroy authority を併存する。rebuild publish が placeholder を handle 経路で
> retire/destroy した後も pointer slot は free 済み address を保持し続け
> （rebuild publish は pointer slot を更新しない）、後続 rebuild の新 DSP が
> address 再利用で map に再登録されると、terminal の pointer-value retire が
> **生存 DSP を誤 lookup し** destroy する（probe8/9 実測: `[D117_RETIRE] 69F6E080
> retired=1`（map 再命中）→ 二重 `[D117_DESTROY]` → 0xC0000005）。

本 defect の本質は「pointer-value が ownership authority として残っていること」であり、
修復の第一原則は **pointer retire を安全化するのではなく、destroy authority から除去する**。

---

## RC-D169-1-1 — Single Destroy Authority

> registered DSPCore の terminal retirement は **DSPHandle identity による
> handle-based retirement のみ**を使用する。

したがって terminal `releaseResources()`（`AudioEngine.Processing.ReleaseResources.cpp`）
における次の pointer-value retire は**廃止対象**:

```text
activeToRelease          ← getActiveRuntimeDSP()（:172・legacy placeholder slot）
fadingToRelease          ← fadingRuntimeDSPSlot CAS（:178-186）
pendingNewToRelease      ← 宣言のみ・非 null writer 不在（D170-1 preflight 実測）
pendingCurrentToRelease  ← pendingTask.currentDSP（:194・非 null writer 不在）
    ↓
lifetimeForShutdown.retire(rawPointer)          ← :352-359 廃止
```

現行 source で既に存在する handle 経路が単一 authority として機能する:

```text
:466-467  dspHandleRuntime_.getActiveRuntimeDSPHandle() / getFadingRuntimeDSPHandle()
:487-489  resolveDSPHandle(handle)（generation 検証付き）
:490-505  dspHandleRuntime_.retire(handle) + tryShutdownQuiescentReclaim(handle)
:556-576  V-D-b: resolve 値を DSPLifetimeManager::retire() で authority 経由 dispose
          （map erase + registry Retired + requestReclaim + EBR enqueue → destroyDSPCoreNode）
```

## RC-D169-1-2 — Pointer Slot は Ownership Source ではない

> `activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` は **legacy placeholder observation**
> に限定し、**retire / destroy / ownership resolution の入力に使用しない**。

- これは D162-2-E (E-2) が dtor（`CtorDtor.cpp:190-214`）に適用済みの原則を
  `releaseResources()` に横展開するものである。E-2 コメントは本 race を
  「address reuse 時に生存 DSP の map entry を誤 lookup して二重破壊」として
  明示的に予見している。
- slot 値自体は topology convenience state であり、ownership authority に昇格させない
  （`AudioEngine.h:2226-2231` / `:2264-2267` の既存コメントと整合）。
- capture 変数は観測用として維持し、`juce::ignoreUnused` で抑制する（dtor E-2 前例と同一）。

## RC-D169-1-3 — Handle path の既存 lifetime protocol を変更しない

> D170 は次の既存経路を変更しない。

```text
DSPHandle
 → resolveDSPHandle() / getActive(Fading)RuntimeDSPHandle()
 → dspHandleRuntime_.retire(handle)（registry Retired 遷移・冪等）
 → tryShutdownQuiescentReclaim()（ShutdownQuiescent Permit による reclaim）
 → DSPLifetimeManager::retire/retireByHandle
 → runtimeDSPHandleMap_ identity removal
 → requestReclaimHandle（epoch 安全確認 + pendingReclaimHandles 保留）
 → ISRRetireRouter::enqueueWithRetry（EBR・epoch 安全後に destroy）
 → destroyDSPCoreNode
```

`retireByHandle(handle)` は handle identity で `findAndEraseByHandle()` する
（`DSPLifetimeManager.cpp:79-135`）。map MISS は「既に disposition 済み」を意味する
正常な冪等 no-op である（D170-1 preflight で実証・後述）。

## RC-D169-1-4 — terminal/reconfigure semantics は D167 のまま

> D170 は D167 の reconfigure/terminal 分離を変更しない。

```text
reconfigure → admission Open → transient release only（reconfigure pass）
terminal    → existing shutdown pipeline（releaseResources terminal pass）
            → handle-based DSP retirement（RC-D169-1-1）
```

## RC-D169-1-5 — 禁止する修復

D170 で以下を行わない:

- generation tag を pointer に追加
- pointer slot に fake handle を持たせる
- `retire(void*)` の内部で generation 判定を追加
- address reuse 検出による防御的 workaround
- `sleep` / delay / 500ms stabilization
- rebuild を terminal 前に強制停止するだけの workaround
- `runtimeDSPHandleMap_` の lifetime semantics 変更
- EBR protocol の変更
- shutdown FSM の変更

---

## 1. 契約と既存 invariant の整合

| 契約 | 既存規範 | 整合 |
| --- | --- | --- |
| RC-D169-1-1 | INV-D162-1/3（registered DSP 処分は DSPLifetimeManager handle 経由）・D162-2-A §5 使用規則「registered DSPCore を手放す全経路 → DSPLifetimeManager::retire()」 | ✓ |
| RC-D169-1-2 | D162-2-E (E-2) dtor 前例・`AudioEngine.h:2264-2267` placeholder 専用スロットの規定 | ✓ |
| RC-D169-1-3 | D162-2-B (C′) retireDSPHandleForRuntime 契約注記（AudioEngine.h:4332-4348）・work88 Phase 3 requestReclaim 一本化 | ✓ |
| RC-D169-1-4 | D167 DS-F2（reconfigure pass / terminal pass 分離・D168 で全軸検証済み） | ✓ |

## 2. D170-1 preflight で確定した事実（契約の前提条件）

1. **pointer slot の writer は prepareToPlay のみ**（`PrepareToPlay.cpp:270/301`）。
   bootstrap placeholder は `commitRuntimePublication(RegistrationContext::needsRegistration)` で
   registration・activate され、その handle は `activeRuntimeDSPHandle_` に公開される
   （`DSPTransition.h:71/104` → `ISRDSPHandle.cpp:93-102`）。すなわち pointer slot の
   DSP は **常に handle 系で disposition 可能**である。
2. **`pendingNewToRelease` は宣言のみで非 null になり得ない**（src 全走査: 代入 0 件）。
3. **`pendingTask.currentDSP` は非 null になり得ない**（`RebuildDispatch.cpp:602` の
   `task.currentDSP = nullptr` 初期化のみで、非 null writer は src 全走査で 0 件。
   残 13 site はすべて null 代入・copy・読み取り）。CtorDtor.cpp:172 の
   「worker 側の未コミット生成物」コメントは歴史的経緯の名残であり現行 source では死変数。
4. **handle registry の active/fading handle writer は publish commit のみ**
   （`DSPTransition.h:71/104` activate・`:119` beginCrossfade / `Timer.cpp:954` endCrossfade）。
   terminal 時点で handle が Retired 済みでも `resolve()` は Retired を valid と返し、
   `dspHandleRuntime_.retire()` は冪等、map MISS は no-op — **全冪等で二重 retire 不可**。
5. pointer 側廃止後の未処理 DSP リスク: なし（§1 の handle 経路 + rebuild pipeline の
   old-handle retire intent + graceful drain / pendingReclaimHandles / PR2 quarantine drain /
   waitForDrain / dtor E-2 / D5-D8 drain が全 disposition を網羅）。

## 3. D170 実装範囲（契約に基づく）

**対象ファイル: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` のみ。**

```text
BEFORE                                         AFTER
terminal releaseResources                      terminal releaseResources
    ├─ pointer slot → retire(void*)  ← REMOVE      └─ handle → resolveDSPHandle
    │     (:151-157 decl/:172/:178/:194 capture)       → dspHandleRuntime_.retire
    │                                                  → tryShutdownQuiescentReclaim
    └─ handle path（:466-505）← KEEP                   → V-D-b authority retire
       V-D-b（:556-576）← KEEP                      （SINGLE AUTHORITY）
```

capture 変数・slot clear（`setActiveRuntimeDSP(nullptr)` / fading CAS /
pendingTask consume）は観測・slot 衛生として**維持**する（RC-D169-1-2）。

## 4. D170 検証条件（契約に基づく受入基準）

```text
D170-1  source audit（本契約 §2 の前提実証）
D170-2  minimal implementation（上記範囲のみ・RC-D169-1-5 禁止事項非適用）
D170-3  Debug CTest 全 PASS
D170-4  Release CTest 全 PASS
D170-5  reconfigure → prepare → rebuild in-flight → immediate terminal
        （D169-1 probe8/9 直接再現条件）で exit 0 / dump 0 / TV=0
D170-6  address reuse stress（rebuild churn による allocator 再利用下で
        D169-1 signature 非再現）
D170-7  repeated restart（×6 全 exit 0x0）
D170-8  DS/WA regression（long-run で D168 基準維持・TV=0）
D170-9  trace ownership accounting: [D117_RETIRE]/[D117_DESTROY]/
        [DSP_FOOTPRINT_RELEASED] を DSP identity 単位で突合し
        **同一 DSP に destroy 2 回が存在しない**ことを実証
```

---

## 5. 変更範囲

production 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary 0（本契約文書のみ新規作成）。

## 6. 後続

- D170（race repair implementation・本契約に拘束）
- D169-2（duplicate-prepare collapse abort）は **D170 完了後に独立 track**（修復原理が異なる:
  physical lifetime/ownership ではなく prepare transaction / lifecycle FSM）。
