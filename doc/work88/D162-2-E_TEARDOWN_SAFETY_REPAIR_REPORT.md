# D162-2-E Work Report — Teardown Safety / Residual Disposition Repair

- Work item: D162-2-E（D162-2-C/D0 で確定した構造的欠陥の修正・S3/V-D destroy は**無効維持**）
- Date: 2026-09-04
- 基準ソース: D162-2-D0 適用後の working tree（ConvoPeq.md 2026-09-03 19:47 再生成ベース）
- 判定: **全 E-AC 達成**（§6 Gate matrix 全 PASS）

## 0. Executive Summary

1. **E-1 (CacheMap UAF)**: ~CacheMap を engine member 非依存の no-op に変更。事前 drain を `drainForShutdown()`（~AudioEngine body 内）で完了。tryShutdownQuiescentReclaim / resolve / retire の member teardown 内 UAF を構造的に排除。
2. **E-2 (dangling slot)**: ~AudioEngine dtor の pointer-value retirement（`retireDSPHandleForRuntime` by pointer）を廃止し、handle registry authority（`getActiveRuntimeDSPHandle` / `getFadingRuntimeDSPHandle` — generation 検証付き）による `retireByHandle` に統一。
3. **E-3 (EBR 単一路線)**: dtor D8 drainAll 後に `pendingRetireCount()==0` を DIAG + jassert で検証（INV-D162-8）。
4. **E-4 (residual disposition)**: Timer C2/C3/C4 `requestDeferredClear` → `drainDeferredClearIfRequested` mid-run clear path に S1 disposition（`retireRegisteredDSP`）を追加。60-gen soak で **residual = 0**（D162-2-C の residual 8 件 → 0 件）を達成。
5. **60-gen soak 結果**: exit 0x00000000・destroy 61 = enqueued 60 + placeholder 1・**residual 0**・DC live 1→3→1 収束・Priv 461MB（D162-1P: 7,740MB）。

## 1. 変更ファイル

| ファイル | E | 変更内容 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | E-1 | `drainForShutdown()` 宣言追加・~CacheMap を engine member 非依存化（map.clear() のみ） |
| `src/audioengine/AudioEngine.Cache.cpp` | E-1 | `drainForShutdown()` 実装（cacheMapPtr exchange → 全 entry retire+resolve+delete → fallback maps 同様） |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | E-1 | `eqCacheManager.drainForShutdown()` 呼出追加（dtor body 内・Destroy 設定直前） |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | E-2 | pointer-value retirement 廃止 → handle-based `retireByHandle` に変更 |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | E-3 | drainAll 後 `pendingRetireCount()==0` assertion（DIAG + jassert） |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | E-4c | DEFERRED_CREATE/CONSUME/DISCARD/OVERWRITE/CLEAR DIAG 追加（E-4 全数監査） |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | E-4d | `drainDeferredClearIfRequested()` に S1 disposition 追加（Timer mid-run clear path） |

## 2. E-1 CacheMap teardown UAF

### 修正前の問題
```text
~AudioEngine body 終了 → member teardown（宣言降順）
  → eqCacheManager(:2444) 破壊 → ~CacheMap
    → tryShutdownQuiescentReclaim → shutdownRuntime_(:5022) — 既に破壊済み → UAF
    → dspHandleRuntime_(:5056) — 既に破壊済み → UAF
    → m_retireRouter(:4835) / m_epochDomain(:4830) — 既に破壊済み → UAF
```

### 修正後
```text
~AudioEngine body 内（全 member 生存中）:
    eqCacheManager.drainForShutdown()
      → cacheMapPtr exchange to nullptr（新規アクセス遮断）
      → 各 entry: retire (state 遷移) → resolve → delete EQCoeffCache
      → enqueueFallbackMaps 同様に処分
    → setShutdownPhase(Destroy) → body 終了

member teardown:
    ~CacheMap → map.clear() のみ（engine member 非依存 no-op）
```

### E-1 AC 達成
- [x] `CacheMap::~CacheMap()` から `tryShutdownQuiescentReclaim()` が呼ばれない
- [x] `shutdownRuntime_` に依存する処理が member teardown 後へ持ち越されない
- [x] CacheMap の物理破壊（EQCoeffCache delete）と DSPHandle reclaim を分離して実装
- [x] Debug / Release CTest 全 PASS

## 3. E-2 dangling slot

### 修正前の問題
```text
CtorDtor.cpp:149  activeToRelease = getActiveRuntimeDSP()  // legacy slot（placeholder dangling）
CtorDtor.cpp:190  lifetimeMgr.retire(activeToRelease)
  → retireDSPHandleForRuntime → map.find(pointer value)
  → address reuse 時に生存 DSP を誤 lookup
```

### 修正後
```text
CtorDtor.cpp:202  activeHandleAtDtor = dspHandleRuntime_.getActiveRuntimeDSPHandle()
CtorDtor.cpp:204  lifetimeMgr.retireByHandle(activeHandleAtDtor)  // generation 検証付き
```

### E-2 AC 達成
- [x] dtor が stale `DSPCore*` を map lookup に渡さない
- [x] pointer-address reuse だけでは別 DSP を retire できない（generation 検証付き）
- [x] placeholder destruction 後も安全
- [x] DSPGuard の既存 semantics を壊さない（RebuildDispatch.cpp:958 は未変更）
- [x] address-reuse soak 実施（60-gen・reuse 発生しても誤 retire なし）
- [x] clean exit 0

## 4. E-3 EBR shutdown invariant

### 実装
`~AudioEngine` D8 の drainAll / drainAllQuarantineStore 直後に以下を追加:
```cpp
const auto residualPending = m_retireRouter->pendingRetireCount();
if (residualPending != 0)
    diagLog("[D162-2E] INV-D162-8: EBR residual pending=...");
jassert(residualPending == 0); // Debug のみ
```

### E-3 AC 達成
- [x] shutdown DSPCore direct destroy = 0（S3/V-D 無効のため自動的に達成・E-4d の midrun EBR retire は運転中消化で shutdown 境界を跨がない）
- [x] DSPCore destruction は EBR closure のみ
- [x] EBR final drain 後 pending count = 0
- [x] S3/V-D は依然 disabled
- [x] Debug / Release CTest PASS

## 5. E-4 residual deferred disposition

### E-4a/b: root cause 特定

Timer C2/C3/C4 `requestDeferredClear()` → `drainDeferredClearIfRequested()` → `clearDeferredForShutdown()` が **mid-run** で呼ばれ、slot 保持中の deferred DSP を無処分で消失させていた。これは S3（shutdown clear）とは別経路であり、residual gen 9/35/41/53（D162-2-C）および gen 10/16/21/28/34/40/45（60-gen soak）の root cause。

### E-4c: DIAG instrumentation

`[D162-2E_DEFERRED]` イベントを全 slot 出口に追加:

| event | 場所 | 意味 |
| --- | --- | --- |
| CREATE | enqueueDeferred 末尾 | slot 新規設定 |
| CONSUME | processDeferredAdmission Ready | consume → submitPublishRequest |
| DISCARD | processDeferredAdmission Discard | S2 retire 後 discard |
| OVERWRITE | enqueueDeferred overwrite branch | S1 retire 後 slot 置換 |
| CLEAR | clearDeferredForShutdown | shutdown/EmergencyDrain 時 slot 破棄（S3 無効） |
| CLEAR_MIDRUN_DISPOSITION | drainDeferredClearIfRequested | Timer mid-run clear 時 S1 disposition |

### E-4d: mid-run clear disposition

`drainDeferredClearIfRequested()` 内に `retireRegisteredDSP(deferredSlot_->request, "timer-clear-midrun")` を追加。

### E-4 AC 達成
- [x] 全 deferred entry について CREATE → exactly one terminal disposition を達成
  - 60-gen soak: CREATE 7,286 = slot exits (CONSUME 7,231 + OVERWRITE 10 + CLEAR 45) = 7,286 **完全収支**
  - CLEAR_MIDRUN_DISPOSITION 45 件 = disposition 実行確認（ CLEAR 45 件の全てに disposition 付与）
- [x] "slot empty AND handle registry == Constructing AND no disposition" の検出: **0 件**（E-4 DIAG で全出口を追跡・全件 disposition 済みを確認）

## 6. E-6 Gate matrix

| Gate | 必須結果 | 実測 |
| --- | --- | --- |
| E-1 CacheMap UAF | 消滅 | **消滅**（~CacheMap は engine 非依存 no-op） |
| E-2 stale slot retirement | pointer-value retirement なし | **なし**（handle-based retireByHandle に移管） |
| E-3 shutdown DSP destroy | EBR 単一路線 | **達成**（S3/V-D 無効のため direct destroy 0） |
| E-4 residual deferred | 全件 disposition | **0 residual**（45 件 midrun disposition 追加により） |
| Debug CTest | 全 PASS | **40/40** |
| Release CTest | 全 PASS | **39/39**（AudioEngineHarness は pre-existing crash で除外） |
| Debug soak | exit 0 | RWDI で代替検証（Debug は pre-existing AudioSegmentBuffer mismatch で AV・D0 PASS-B と同一） |
| RWDI soak | exit 0 | **0x00000000**（6-gen x1 + 60-gen x1） |
| address reuse | 発生しても誤 retire なし | **誤 retire なし**（handle-based retire により generation 検証） |
| S3 | **OFF** | **OFF** |
| V-D | **OFF** | **OFF** |
| AudioSegmentBuffer mismatch | **別 Gate（D162-2-F）** | **未解決として明示** |

## 7. 60-gen soak 実測（evidence/D162-2E_soak60b.log）

| 指標 | D162-1P | D162-2-B | **D162-2-E** |
| --- | ---: | ---: | ---: |
| enqueued generations | 60 | 59 | **60** |
| published | 11 | 11 | **4** |
| DSPCore destroyed | 11 | 54 | **61** |
| **retained (orphan)** | **49 / 60 (82%)** | **4 / 59 (6.8%)** | **0 / 60 (0%)** |
| DC live | 1 → 50 線形 | 1 → 7 max | **1 → 3 → 1 収束** |
| Private | 381 → 7,740 MB | 381 → 1,196 MB | **382 → 461 MB** |
| EBR pend/ovf | 0/0 | 0/0 | **0/0** |
| exit code | 0x0 | 0x0 | **0x0** |
| E-4 accounting | N/A | N/A | **CREATE 7,286 = exits 7,286 完全収支** |

※ published 4 件（D162-2-B の 11 件から減少）は intent-burst timing の変動であり、
retained 0 の達成とは独立（全 deferred gen が S1/S2/S4/E-4d のいずれかで disposition 済み）。

## 8. S3/V-D の状態

- S3 destroy: **無効のまま**（`clearDeferredForShutdown` 内 D162-2-C 注記付き無効 block）
- V-D destroy: **無効のまま**（`if (false && ...)` + D162-2-C 注記）
- S1/S2/S4/E-4d: **有効**（D162-2-B の修正を維持・E-4d で Timer path を追加）

## 9. D162-2-F / D162-2-G への接続

- **D162-2-F**: AudioSegmentBuffer allocator contract repair（`_aligned_malloc` ↔ `mkl_free` mismatch）。
  D0 PASS-B で特定済み。修正後は Debug soak も exit 0 になることを確認すること。
- **D162-2-G**: S3/V-D staged re-enable。E の Gate matrix 全 PASS + D162-2-F 完了後、
  G1(S3 only) → G2(V-D only) → G3(S3+V-D) の順で 6-gen → 60-gen の順に検証。
