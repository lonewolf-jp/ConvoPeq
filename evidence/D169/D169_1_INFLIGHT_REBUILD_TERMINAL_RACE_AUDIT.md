# D169-1 — In-flight Rebuild × Terminal Shutdown Lifetime / Ownership Race Audit

```text
Task:   D169-1 — In-flight Rebuild × Terminal Shutdown Lifetime / Ownership Race Audit
Date:   2026-09-06
Type:   read-only architectural / source audit（production 変更 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary rebuild 0）
Source: 現行 production source を直接読取（ConvoPeq.md snapshot 2026-09-05 12:02:18 は source authority として不使用）
Trigger: D167 §8-1 で実測した 0xC0000005（probe8/probe9・harness testD167ReconfigureKeepsAdmissionOperational step-f）
Verdict: **Case A — Defect confirmed（source-level・object lifetime / ownership chain と terminal/rebuild interleaving を成立証明）**
```

---

## 0. 判定サマリ

> **Case A。** terminal `releaseResources()` は同一 DSPCore object に対して
> **handle-based retire（epoch 保護・正しい）** と **pointer-value retire（legacy
> `activeRuntimeDSPSlot`・epoch 保護なし）** の **二重 destroy authority** を併存して使う。
> 後者は D162-2-E (E-2) が **dtor 側から「address reuse 時に生存 DSP を誤 lookup し二重破壊する」
> として廃止済み**の pattern であり、releaseResources 側にのみ残存している。
> D167 の reconfigure pass が「placeholder を handle 経路で retire/destroy した後も
> pointer slot にその address が残る」前条件を実運用で成立可能にしたため、
> terminal の pointer-value retire が reuse 後 address の生存 DSP を destroy し、
> handle 経路の destroy と二重化して 0xC0000005 に至る。

---

## 1. destroy authority の全 caller（GO 条件 [8]）

DSPCore の物理破壊は `AudioEngine::destroyDSPCoreNode`（`AudioEngine.Threading.cpp:22` に
`[D117_DESTROY]` log）に収束する。これを取得する権限経路は 2 系統：

| 系統 | 入口 | 台帳操作 | epoch 保護 | 帰結 |
| --- | --- | --- | --- | --- |
| **A. handle-based（正しい）** | `DSPLifetimeManager::retireByHandle(handle)` / `retire(void*)` 経由の `retireDSPHandleForRuntime` | `runtimeDSPHandleMap_` 解除 + `dspHandleRuntime_.retire(handle)` + `requestReclaimHandle` | **あり**（`enqueueWithRetry` → epoch 安全後に destroy） | 単一 destroy |
| **B. pointer-value（legacy）** | `DSPLifetimeManager::retire(void* dsp)` → `retireDSPHandleForRuntime(DSPCore*)` の **map.find(by pointer)** | pointer 値で map を引く | **map 命中時に限り** epoch 保護 | **address reuse で別 object を引く** |

`retireDSPHandleForRuntime(DSPCore* dsp)`（`AudioEngine.h:4388`）は
`runtimeDSPHandleMap_.find(dsp, handle)` — **raw pointer を key とする map lookup** を行う。
この lookup が system B の本質であり、E-2 が dtor から除去した対象と同一。

---

## 2. terminal releaseResources の ownership chain（GO 条件 [3][4]）

`AudioEngine.Processing.ReleaseResources.cpp`（terminal pass・D167 で無変更部）:

```text
:172  activeToRelease = getActiveRuntimeDSP();      ← system B 入力（legacy pointer slot）
:174  setActiveRuntimeDSP(nullptr);                 ← slot は以後 null（capture 済み値は保持）
...
:201  shutdownCoordinatorLoop();                    ← CoordinatorLoop join
:202  stopRebuildThread();                          ← rebuild thread join
...
:352  if (activeToRelease) lifetimeForShutdown.retire(activeToRelease);   ← ★ system B destroy 発火
...
:444  const auto activeHandle = dspHandleRuntime_.getActiveRuntimeDSPHandle();  ← system A 入力（handle）
:465  activeDSPToDestroy = resolveDSPHandle(activeHandle);
:468  if (!activeHandle.isNull()) { dspHandleRuntime_.retire(activeHandle); tryShutdownQuiescentReclaim(...); }
:534  lifetimeMgrForFinalDSP.retire(activeDSPToDestroy);   ← ★ system A destroy 発火
```

**同一 terminal pass 内で pointer-value retire（:352）と handle-based retire（:534）が
両方走る。** 通常（placeholder が生存・map に在る）は :352 が placeholder を、:534 が
world active（handle）を扱い、対象が別 object なので無害。危険は **placeholder が既に
handle 経路で retire/destroy 済み** のときに発生する（§4）。

対して **dtor（`AudioEngine.CtorDtor.cpp`）は E-2 で pointer-value retire を廃止済み**:

```text
CtorDtor.cpp:151  activeToRelease = getActiveRuntimeDSP();   ← 観測専用（retire に使わない）
CtorDtor.cpp:202  // activeToRelease / fadingToRelease は handle-based retire で全てカバーされるため
                  //   pointer-value retirement は廃止
CtorDtor.cpp:208  lifetimeMgr.retireByHandle(activeHandleAtDtor);   ← system A のみ
```

E-2 コメント（`CtorDtor.cpp:186-196`）が本 race を明示的に予見している:

> 「activeRuntimeDSPSlot / fadingRuntimeDSPSlot は placeholder 専用レガシースロット…
>  bootstrap placeholder 破壊後は dangling ポインタを保持し続ける。この値を
>  retireDSPHandleForRuntime（map.find by pointer value）に渡すと、**address reuse 時に
>  生存 DSP の map entry を誤 lookup して二重破壊を引き起こし得る**（D162-2-C §4.3 / D0-5-A）。」

**releaseResources 側にはこの E-2 修正が適用されていない**（:172/:352 が pointer-value retire のまま）。

---

## 3. pointer slot の寿命（GO 条件 [2]）

`activeRuntimeDSPSlot`（`AudioEngine.h:2228`）の writer は **prepareToPlay の placeholder 作成のみ**:

```text
PrepareToPlay.cpp:270  setActiveRuntimeDSP(placeholderRaw);   ← placeholder 作成時
PrepareToPlay.cpp:301  setActiveRuntimeDSP(nullptr);           ← CallerDestroy rollback のみ
```

`hasPublishedRuntimeDSP()` のコメント（`AudioEngine.h:2265`）が明記する通り
**「activeRuntimeDSPSlot は placeholder 専用のレガシースロットで通常動作（runtime world
公開後）では null」**。すなわち **rebuild publish は pointer slot を一切更新しない**
（rebuild は handle / world 側のみ操作）。

→ placeholder が rebuild publish の handle 経路で retire/destroy されても、
`activeRuntimeDSPSlot` は **その free 済み address を保持し続ける**（clear されない）。
terminal :172 はその stale address を `activeToRelease` に capture する。

---

## 4. interleaving 完全再現（GO 条件 [5]・R3）

D167 probe8/probe9 の `[D117_*]` 実測 trace と source を対応させた成立順序:

```text
rebuild thread (in-flight task, SR 変更 re-prepare が dispatch)
   │  build 完了 → CoordinatorLoop executePublish tail (RuntimePublishExecutor.h:98)
   │    onPublishCompleted → DSPLifetimeManager::retire(oldDSP=placeholder 69F6E080)
   │      → retireDSPHandleForRuntime: map.erase(69F6E080) + handle Retired + enqueue destroy
   ▼
[D117_RETIRE] 69F6E080 retired=1 epoch=8        (probe log :3407)
[D117_DESTROY] 69F6E080                          (probe log :3427  ← EBR により物理破壊)
[DSP_FOOTPRINT_RELEASED] 69F6E080 remaining=0   (probe log :3475  ← free 完了)
   │
   │  ※ activeRuntimeDSPSlot は 69F6E080 のまま（rebuild は pointer slot を触らない・§3）
   ▼
terminal releaseResources()（h.stop()）
   :172 activeToRelease = getActiveRuntimeDSP() = 69F6E080（stale）
   :352 lifetimeForShutdown.retire(69F6E080)
          → retireDSPHandleForRuntime: map.find(69F6E080)
   ▼
[D117_RETIRE] 69F6E080 retired=1 epoch=12       (probe log :3530  ← ★ map 再命中 = address reuse)
[D162-2G2_VD_RETIRE] 69F6E080 target=active-final (probe log :3556)
[D117_RETIRE] 69F6E080 retired=0                (probe log :3557  ← 二重 retire 拒否)
[D117_DESTROY] 69F6E080                          (probe log :3558  ← ★ 二重物理破壊 → 0xC0000005)
```

:3530 の `retired=1`（map 再命中）は、free 済み 69F6E080 address が **後続 rebuild の
新 DSP に再利用され map に再登録された** ことを意味する（E-2 の「address reuse」条項）。
terminal は pointer 値 69F6E080 を **生存 DSP の handle として誤 lookup** し、
handle 経路（:534）が destroy する **同一 object を pointer 経路（:352）でも destroy** →
二重破壊。

---

## 5. R1〜R5 判定

### R1 — 実際の double-destroy か → **Yes（C パターン）**
単なる「同一ポインタ 2 回 delete」ではなく、
```text
A: rebuild publish（handle 経路）が placeholder 69F6E080 を retire → EBR destroy（:3427）
B: terminal が pointer slot の stale address 69F6E080 を capture（:172）
   → その address が後続新 DSP に reuse され map 再登録
C: terminal pointer-value retire（:352）が reuse 後 object を destroy（:3558）
   ＝ handle 経路（:534）が destroy する object と同一 → 二重破壊
```
E-2 が dtor で予見・排除した pattern と逐語一致。

### R2 — ownership が二重化する地点 → **terminal pass 内 :352 と :534**
```text
Build local ownership（rebuild thread）
   ↓ publish tail onPublishCompleted
activeRuntimeDSPHandle_（handle・world active）  ← system A の入力
   ↓
RuntimeWorld / publication
   ↓
retire / shutdown
   ↓
★ activeRuntimeDSPSlot（legacy pointer・placeholder 専用）  ← system B の入力
```
**同一 DSP object に対し、terminal pass が handle 権限（:444/:534）と pointer 権限
（:172/:352）の両方を持つ。** dtor は E-2 で pointer 権限を捨てたが releaseResources は捨てていない。

### R3 — shutdown と rebuild の linearization → **window あり**
- 「shutdown が rebuild の不在を判断する時点」= :201/:202 の join 完了後。
- 「Builder が自分の DSP を有効と判断する時点」= publish tail の `enqueueWithRetry` 済み・
  EBR destroy 未達の間、および **reuse 後新 DSP が map に再登録された時点**。
- join 完了後も **EBR destroy と address reuse / 再登録は CoordinatorLoop 停止前の
  pending queue・drain 経路で進行** するため、:172 capture と :352 retire の間に
  placeholder の destroy→reuse→再登録が挟まり得る。window は source 上成立。

### R4 — generation は lifetime protection か → **No（logical のみ）**
`isRebuildObsolete(gen)`（`AudioEngine.h:2598`）は `rebuildRequestGeneration` の
**int 比較**で stale **request** を弾くだけで、**physical DSP handle / pointer の lifetime
は一切保護しない**。pointer slot は generation tag を持たず、`retireDSPHandleForRuntime`
の map lookup は raw pointer 一致のみ。よって stale physical handle を防げていない
（指示の懸念どおり）。

### R5 — Epoch/retire がこの DSP に効いているか → **system B は protocol を迂回**
handle 経路（system A）は retire → epoch → reclaim → destroy の一本路を通る。
pointer 経路（system B）は **pointer→handle 再解決を map lookup で行う**ため、
reuse 後は「別 object の epoch 安全な destroy」ではなく「生存 object を epoch 文脈外から
選んで destroy」になる。Practical Stable ISR Bridge の
ownership → retire → epoch → reclaim → delete 一本路から **pointer-value retire が逸脱**。

---

## 6. 成立条件（再現の十分条件）

1. placeholder が **handle 経路で retire/destroy される** rebuild が 1 回以上 publish 済み
   （= reconfigure 後に rebuild が走る。D167 が可能にした）。
2. その **address が後続 DSP に reuse** され map 再登録される（allocator 挙動・高確率）。
3. terminal が :172 で stale pointer を capture し :352 で pointer-value retire する。
4. handle 経路 :534 が同一（reuse 後）object を destroy する。

D165 以前は (1) が device switch 後に発生しなかった（admission Closed で rebuild 停止）ため
顕在化せず、D167 の修復が (1) を成立可能にしたことで latent defect が露出した。

---

## 7. repair contract 方向（提示のみ・実装しない — GO 条件 [10]）

**原則: destroy authority を handle-based 一本路に収束させ、pointer-value retire を
terminal から除去する（E-2 を releaseResources へ横展開）。**

- 最小案: `releaseResources()` の :172 capture / :352 retire（`activeToRelease` /
  `fadingToRelease` / `pendingNewToRelease` / `pendingCurrentToRelease` の pointer-value
  retire）を、dtor と同一の **handle-based retire（`retireByHandle` / :444-:534 の
  existing authority）** に統合し、pointer slot は観測専用化する。
- 収束先: 「registered DSP を手放す全経路 → DSPLifetimeManager 経由・handle identity」
  （`AudioEngine.h:4343` INV-D162-1/3 と整合）。
- 検証条件: reconfigure→rebuild→即 terminal の interleaving で
  `D117_DESTROY` が DSP あたり 1 回・`remaining=0`・0xC0000005 非発生。

**D170 の実装契約は本 audit 承認後に確定**（指示どおり本 audit 中に決めない）。

---

## 8. GO 条件照合

```text
[1] 現行 source を直接解析                          ✅（ConvoPeq.md 不使用）
[2] rebuild-thread lifetime chain 完全追跡          ✅ §3（pointer slot は rebuild 非更新）
[3] terminal lifetime chain 完全追跡                ✅ §2（:172/:352 と :444/:534 の併存）
[4] DSP object ownership graph                      ✅ §5-R2
[5] shutdown/rebuild interleaving 最低 1 本完全再現  ✅ §4（probe8/9 trace 対応）
[6] generation と physical lifetime の関係           ✅ §5-R4（logical のみ・physical 未保護）
[7] Epoch/retire/reclaim の関与                      ✅ §5-R5（system B が一本路を逸脱）
[8] destroy authority 全 caller                     ✅ §1（system A/B）
[9] 0xC0000005 成立点を source-level で特定          ✅ §2 :352×:534 二重 destroy
[10] repair contract は audit 後にのみ提示            ✅ §7（方向のみ）
[11] production/test/CMake/build 変更 = 0            ✅（本監査 0）
```

→ **Case A — Defect confirmed。**
