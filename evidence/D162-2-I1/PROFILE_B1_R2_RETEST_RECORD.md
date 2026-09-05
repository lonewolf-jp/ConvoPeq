# D162-2-I1-R1/R2 — A′ Implementation Preflight + Implementation + B-1 Re-test Record

```text
Date:     2026-09-04
R1:       A′ Implementation Preflight（read-only）→ **6/6 PASS → 実装 GO**
R2:       A′ Implementation（production 変更 1 ファイル 1 箇所）→ Build 3 config EXIT=0 /
          CTest Debug 40/40 + Release 39/40（pre-existing のみ）→ **B-1 再試験 PASS**
Baseline: ConvoPeq.md 20:28:50 → 修復後再生成 `Generated: 2026-09-04 23:14:29`
```

---

## 1. R1 Preflight（6 条件判定・6/6 PASS）

| # | 条件 | 実コード確認 |
| --- | --- | --- |
| 1 | A′ insertion point 一意 | ReleaseResources.cpp 内の `clearDeferredForShutdown` 呼出は :359（EmergencyDrain 条件付き）のみ。新規位置（V-D retire block 直後）は一意 |
| 2 | RebuildThread join 済み | `stopRebuildThread()` :202（join 完了）→ VerifyDrained :441 → V-D :543/:552 → A′。全て post-join |
| 3 | world clear 後 | `requestShutdownClearNonRt` :503 / `clearPublishedRuntimeSnapshotsNonRt` :504 / `retirePublishedRuntimeWorldNonRt` :508 → V-D block → A′ |
| 4 | V-D retire と競合しない | A′ は V-D block の**後**。deferred DSP は V-D 対象と別個体。重複時も map erase guard で no-op |
| 5 | shutdown caller と冪等 | 入口 gate `consumeAtomic(hasDeferred_)`（Orchestrator :594）— 先に clear 済みなら 2 回目は body 全体 skip。S3 block は `deferredSlot_.has_value()` + resolve nullptr（Reclaimed/erase 済み）で no-op。:634-635 で `hasDeferred_=false` |
| 6 | EBR enqueue → 既存 final drain | A′ → waitForDrain(2000,2) :563 → ~AudioEngine D5 graceful drain + D8 drainAll → E-3 assert（G4 実績どおり） |

`[PUBLISH] seq=7 gen=7` タグ意味論: **DEFER 維持**（本判断に不要）。

## 2. R2 Implementation（diff）

`AudioEngine.Processing.ReleaseResources.cpp` **1 箇所**のみ（V-D retire block 直後・
jassert(SHUTDOWN-ORDER) の前）:

```cpp
    // ★ D162-2-I1 修復（R0 Candidate A′）: shutdown 時 deferred slot の無条件 terminal
    //   disposition。S3（clearDeferredForShutdown）の既存呼出経路は EmergencyDrain
    //   （isEmergencyDrainRequested 条件付き）/ Timer midrun（C2/C3/C4 trigger 条件）/
    //    C1 fallback のいずれも条件付き trigger のため、短時間 shutdown では slot 残留
    //    DSP が無処分になる（Profile B-1 実測: residual=1・149MB）。
    //    本位置は (i) RebuildThread join 後（単一 writer 契約成立）・(ii) world clear 後
    //   （world topology 参照解消済み）であり、slot 保持 DSP を authority（EBR）経由で
    //    処分する。slot 空の場合は no-op（EmergencyDrain で先に clear された場合も冪等）。
    //    EBR entry は下記 waitForDrain と ~AudioEngine D5/D8 drain で消化（INV-D162-8 準拠）。
    if (runtimeOrchestrator_)
        runtimeOrchestrator_->clearDeferredForShutdown();
```

- 触れていないもの: test / CMake / Orchestrator（S3 実装本体は G1 のまま） / EBR / V-D / E-3 / その他 shutdown 経路。

## 3. Build / CTest

| Gate | 結果 |
| --- | --- |
| Build（Debug / Release / RWDI） | 全 **EXIT=0** |
| CTest Debug | **40/40 PASS**（39.09s） |
| CTest Release | **39/40** — AudioEngineHarness 0xC0000374 のみ（pre-existing class）→ 新規失敗 0 |

## 4. B-1 再試験（修復後）— **PASS**

| 項目 | 実測 | 合格 |
| --- | --- | --- |
| exit / new dump | **0x00000000** / **0**（CrashDumps 10 不変） | ✓ |
| shutdown zone | **clean**（:3778 SHUTDOWN_BEGIN → :3913 reset completed → :3915 LOGGER_DETACH/END） | ✓ |
| IR load / rebuild | 1 回 / 3 build（gen 6/7/8） | — |
| deferred CREATE | **68**（gen8 retention churn） | — |
| **E-4 accounting** | **CREATE 68 = CONSUME 67 + OVERWRITE 0 + CLEAR 1 + DISCARD 0**（B-1 修復前は 69 ≠ 68） | **✓ 完全収支** |
| **CLEAR_SHUTDOWN_DISPOSITION** | **1** | ✓ |
| **retired(gen8, shutdown-clear origin)** | **= 1** ← **G1-G4 通じて初めての S3 standalone retired=1 実測** | ✓ |
| EBR enqueue | Success（epoch 17） | ✓ |
| destroy(gen8) / remaining | **1 / remaining=0** | ✓ |
| destroy 合計 / release | **4 / 4**（gen5 placeholder・gen6・gen7・gen8 = 1:1） | ✓ |
| residual | **0**（B-1 修復前は 1） | ✓ |
| generation 1:1 | gens 5-8 連続・重複 0 | ✓ |
| direct destroy（registered） | 0 | ✓ |
| stale map | MISS 2 / HIT 1（midrun 正常 destroyのみ・禁止 HIT 0） | ✓ |
| E-3 / INV-D162-8 | 0 | ✓ |
| Signature A / B / C | 0 / 0 / 0 | ✓ |
| XRUN | 1 件（Callback 0.65ms・Pressure=0・**shutdown window 0**） | ✓ |
| memory | DC live=2 / Priv=593MB 台 | ✓ |

### 4.1 S3 standalone retired=1 実測（G1-G4 未観測事項の解消）

```text
[D162-2E_DEFERRED] event=CLEAR gen=8 dsp=0000025EBA320080
[D162-2E_DEFERRED] event=CLEAR_SHUTDOWN_DISPOSITION gen=8 dsp=0000025EBA320080
[D162-2B_RETIRE]   dsp=0000025EBA320080 gen=8 origin=shutdown-clear
[D117_RETIRE]      dsp=0000025EBA320080 retired=1            ← ★ G1-G4 通じて初観測
[D117_RETIRE]      dsp=0000025EBA320080 enqueue=0 epoch=17   （EBR Success）
[D117_DESTROY]     dsp=0000025EBA320080
[DSP_FOOTPRINT_RELEASED] dsp=0000025EBA320080 remaining=0
```

A′（無条件呼出）により、slot 保持 DSP が shutdown 時に S3 経由で **authority retire され、
EBR → destroy まで閉じる**ことが実証された。B-1 の leak（149MB 無処分）は消滅。

### 4.2 V-D 観測（ B-1 と同型・安全）

- active-final（gen7 相当・0000025EACDE5080）: retired=1 → EBR → destroy → remaining=0 ✓
- fading-final（0000025E9C7D7080）: midrun destroy 済みの dangling resolve → **V-D-b no-op（retired=0）** ✓（B-1 と同一の防御パターン）
- 最終 2 destroy は ~AudioEngine dtor body 内（enter :3841 → destroy :3849/:3871）— INV-D162-8 適合

## 5. 判定

**R1 = PASS（6/6）→ R2 実装完了 → B-1 再試験 = PASS。**

| ユーザー指定判定項目 | 実測 |
| --- | --- |
| residual = 0 | ✓ |
| CREATE == terminal disposition | ✓（68 = 67+0+1+0） |
| CLEAR_SHUTDOWN = 1 | ✓ |
| retired(gen8) = 1 | ✓ |
| EBR enqueue > 0 | ✓ |
| destroy(gen8) = 1 | ✓ |
| FOOTPRINT_RELEASED remaining = 0 | ✓ |
| E-3 = 0 | ✓ |
| exit = 0 / new dump = 0 / zone = clean | ✓ |

**次工程**: B-2〜B-6 再開の GO（ユーザー判定）→ Profile B completion → C → D → I1 final。
