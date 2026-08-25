# D102-C2-5-D7 — Patch Specification / Test-First Audit（read-only, production変更0）

- **実施日**: 2026-08-26 02:15 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **目的**: D6で確定したAPI境界を実装可能なpatch specificationとtest specificationへ変換する。未確定部分を無理に実装案へ落とさず、9 sinkを実コード上のsink単位で再固定し、Cacheを最優先でpatch仕様を固定、ConvolverのPattern A終端追跡、resetFadeのRTゲート分離、Terminal bounded化の非混入、test-first仕様を確定する
- **判定**: **CONDITIONAL PASS** — 9 sinkの実コード再固定、Group A/B/C分離、Cache/Snapshot/DSPのclosureは具体的patchに固定。Convolverの終端追跡で上位伝播先のownership-bearing containerが未確定のためConditional、resetFadeはRTゲートとして分離し別gateへ

---

## D7-1. 9件をSink単位で再固定

| # | Source location | Function | Ptr | Deleter | Epoch source | Current return type | QueueFull propagation | Post-return owner |
|---|---|---|---|---|---|---|---|
| 1 | `src/audioengine/AudioEngine.Cache.cpp:16` | `EQCacheManager::tryEnqueueDeferredMap(CacheMap* map)` | `map` | `delete CacheMap` | `markRetireEpoch()` via `owner` | `bool` (`return true` 固定) | `WithResult==QueueFull → bool==true`（現行誤） | **なし**（`map` は param、return後 lost） |
| 2 | `src/audioengine/AudioEngine.Cache.cpp:41` | `EQCacheManager::storeNewMap(CacheMap* newMap)` 内 `old` | `old` (`exchangeAtomic`取得) | 同上 | 同上 | `void`（`enqueueDeferredDeleteNonRt` 戻り無視） | `QueueFull → bool true` | **なし** |
| 3 | `src/convolver/ConvolverProcessor.Lifecycle.cpp:57` | `ConvolverProcessor::updateIRState` 内 `oldState` | `oldState` | `deleter` | `provider->currentEpoch()` | `void`（`bool` 無視） | 同上 | **なし** |
| 4 | `src/convolver/ConvolverProcessor.Lifecycle.cpp:70` | `ConvolverProcessor` 内 `sc` | `sc` (`StereoConvolver*`) | `destroyStereoConvolver` | 同上 | `void` | 同上 | **なし** |
| 5 | `src/core/SnapshotCoordinator.cpp:57` | `SnapshotCoordinator::startFade` | `oldTarget` (`exchangeTarget`取得) | `SnapshotFactory::destroy` | `m_epochProvider->currentEpoch()` | `RetireEnqueueResult`（`if (!result) quarantine` あり） | `QueueFull → false → quarantineRetireSink` | **B**（quarantine） |
| 6 | `src/core/SnapshotCoordinator.cpp:114` | `SnapshotCoordinator::completeFade` | `old` (`exchangeCurrent`取得) | 同上 | `m_epochProvider->publishEpoch()` | 同上 | 同上 | **B** |
| 7 | `src/audioengine/DSPLifetimeManager.cpp:49` | `DSPLifetimeManager::retire(dsp, epoch)` | `dsp` (`DSPCore*`) | `destroyDSPCoreNode` | `router_->currentEpoch()` | `RetireEnqueueResult`（`ignoreUnused`） | `QueueFull` でも `ignoreUnused` | **なし**（D） |
| 8 | `src/audioengine/DSPLifetimeManager.cpp:96` | `DSPLifetimeManager::retire` (2経路, 共通 helper `retire(dsp,0)` → `retire(dsp,epoch)`) | 同上 | 同上 | 同上 | 同上 | 同上 | **なし** |
| 9 | `src/core/SnapshotCoordinator.h:60` / `cpp:81` | `SnapshotCoordinator::resetFadeStateAndRetireTarget` | `target` (`exchangeTarget(nullptr)`取得) | 同上 | `m_epochProvider->publishEpoch()` | **直接 `enqueueRetire`**（`enqueueWithRetry` ではない） `void` | `QueuePressure/QueueFull` でも無視 | **なし** |

**DSPの2経路が同一helperを通るか:** `DSPLifetimeManager::retire(void* dsp) noexcept { retire(dsp,0); }` (`cpp:28`) — 2件は同一 `retire(dsp, publicationEpoch)` 共通修正として扱える（個別patchではなく1つの共通修正）。

---

## D7-2. Patchを3群に分離

### Group A — Wrapper Semantics（`AudioEngine::enqueueDeferredDeleteNonRt`）

- **対象:** `src/audioengine/AudioEngine.h:4198` `bool` wrapper
- **現行:** `return result != Shutdown` → `QueueFull` を `true` と誤認
- **仕様:** `Success/QueuePressure/TerminalReclaim → true` / `QueueFull/Shutdown → false`（D3で固定）
- **追加設計なし** — D7-2では意味論のみ固定

### Group B — Non-RT Local Closure

- **対象候補:** `Cache ×2` / `Snapshot startFade/completeFade` / `DSPLifetimeManager ×2`（計 6件、resetFade除く）
- **条件:** `QueueFull → 既存 ownership-bearing container / quarantine authority → その container を誰が drain するか → shutdown 時` まで確認
- **Cache/Convolverへ `quarantineRetireSink` をexportしない** — D6-1で `SnapshotCoordinator` private と確定したため、新API追加はしない

### Group C — Convolver Pattern A（`WithResult`化）

- **対象:** `Convolver 2` （`Lifecycle.cpp:57,70`）
- **D7-4で終端追跡** — 上位 caller が `QueueFull` を無視するなら一段上へ移しただけ（D4 orphan再発）

---

## D7-3. Cacheは最優先でPatch仕様を固定

### 現行 Loop（`AudioEngine.Cache.cpp` / `AudioEngine.h:2142`）

```text
tryEnqueueDeferredMap(map)
  ↓ enqueueDeferredDeleteNonRt(map,deleter) → WithResult → enqueueWithRetry → Q/E/T
  ↓ return true 固定
drainDeferredMapsUnderLock()
  ↓ for each fallbackMaps: if (!tryEnqueueDeferredMap(*it)) *out++ = *it;
```

- `enqueueFallbackMaps` (`AudioEngine.h:2142` `vector<CacheMap*>`) は **既に QueueFull時の退避と再試行の閉ループ**として設計されている
- **現行 `tryEnqueueDeferredMap` が `return true` 固定**のため、`QueueFull → bool false` になっても `true` を返す → fallbackから除去され **ownership closureしない**

### D7 Patch仕様（Cache）

```text
tryEnqueueDeferredMap(map)
  ↓ enqueueDeferredDeleteNonRtWithResult(map, ...) → RetireEnqueueResult
  ↓ true: Success/QueuePressure/TerminalReclaim → caller owns no longer → fallbackから除去
  ↓ false: QueueFull/Shutdown → map remains in enqueueFallbackMaps → later drain (B)
```

- `tryEnqueueDeferredMap` を `bool` 正しく返すように修正（`return result == Success || result == QueuePressure || result == TerminalReclaim`）
- `storeNewMap` の `old` も同様に `enqueueFallbackMaps.push_back(old)` へ（現行は `enqueueDeferredDeleteNonRt` 戻り無視）
- **D3 wrapper修正だけではCache closureにならない** — Cache側の return propagation を patch specificationとして明示

---

## D7-4. Convolverは「Pattern Aで閉じる」と決め打ちしない

### 上位追跡

```text
Convolver lifecycle caller (Lifecycle.cpp:57)
  ↓ provider->enqueueDeferredDeleteNonRt(oldState, deleter) // void
  ↓ 呼び出し元: ConvolverProcessor::updateIRState 等
  ↓ その戻り値を変更可能か: 現行 void のため不可
  ↓ その上位に ownership-bearing object/containerがあるか: ConvolverProcessor 自体が oldState を local に持つが function return 後は lost
  ↓ 既存 quarantine authorityへ到達可能か: provider は IRetireProvider で quarantine API なし
```

- `WithResult → return QueueFull` だけして上位 caller が `result` を無視するなら、D4と同じorphanを一段上へ移しただけ
- **Result propagationの終端まで追跡して初めてPattern A成立** — 現行 Convolver の上位は `void` のため **CONDITIONAL**（上位での `B` 追加が必須だが、上位の `AudioEngine` quarantine authority への委譲可否が未確定）

---

## D7-5. resetFadeは別Gateに分離（D7-R）

- **現行:** `resetFadeStateAndRetireTarget() noexcept` (`SnapshotCoordinator.h:138`) は `h:150` コメント「RT(updateFade) から呼ばれ得るため除外」と明記。現行 `m_epochProvider->enqueueRetire` 直接（`cpp:81`）で `enqueueWithRetry` ではない
- **`enqueueWithRetry` は `tryReclaim() / Q/E/T mutex/allocation` を含むため Non-RT限定** — RTから `quarantineRetireSink` を直接呼ぶと RT境界違反
- **D7では resetFadeを既存9件の通常patchに含めない** — **D7-R: RT Ownership Closure Sub-Gate**として独立
- **確認対象:** `RT → enqueueRetire(D only) → QueuePressure/QueueFull → どこにptrが残るか → NonRT coordinatorがどう再取得するか` — RTでownershipを失わずNonRT側へ搬送する既存mechanismの確認。存在しないなら **resetFadeはNO-GOのまま保留**

---

## D7-6. Terminal Bounded化を混ぜない

- `TerminalReclaimAuthority` (`std::vector` growable) / `kMaxLogicalRecoveryObligations` / `Terminal capacity` / `Emergency capacity` / `P-4` は **変更対象から除外**
- 現行 P-4 前提（growableで常に受領）を維持し、**orphan closureとbounded Terminalは別変更単位**として維持

---

## D7-7. Test-First仕様

| Test | 対象 | 必須検証 |
|---|---|---|
| D7-T1 | bool wrapper | `QueueFull → false` |
| D7-T2 | bool wrapper | `Shutdown → false` |
| D7-T3 | bool wrapper | `Success → true` |
| D7-T4 | Cache | `QueueFull → fallbackMaps保持` |
| D7-T5 | Cache | `retry成功 → fallbackMapsから1回だけ除去` |
| D7-T6 | Snapshot startFade | `QueueFull → quarantine` |
| D7-T7 | Snapshot completeFade | `QueueFull → quarantine` |
| D7-T8 | DSP | `QueueFull → quarantine` |
| D7-T9 | DSP | `quarantine後のdouble-retireなし` |
| D7-T10 | Convolver | `QueueFullが上位終端まで伝播` |
| D7-T11 | Convolver | `上位でownership保持` |
| D7-T12 | resetFade | `RTからNonRT-only APIを呼ばない` |
| D7-T13 | Shutdown | `caller-retainを維持` |
| D7-T14 | transferred | `Success/QueuePressure/TerminalReclaimでownership移転` |
| D7-T15 | conservation | `orphan / double ownership / double delete = 0` |

---

## D7-8. Patch Specification 形式

```text
D102-C2-5-D7
├─ Patch P1: AudioEngine.h
│  └─ bool wrapper semantics (return !=Shutdown && !=QueueFull)
│     現行: return result != Shutdown
│     変更理由: QueueFullもcaller retains
│     ownership before: Success/QueuePressure/TerminalReclaim/QueueFull == true
│     ownership after: QueueFull==false, Shutdown==false, 転送3値==true
│     QueueFull path: false → caller retains
│     Shutdown path: false
│     RT/NonRT: NonRT
│     必要test: D7-T1, T2, T3, T14
│     rollback: 1行差し戻し
│
├─ Patch P2: AudioEngine.Cache.cpp (Cache)
│  └─ fallback ownership closure (tryEnqueueDeferredMap/storeNewMap)
│     現行: return true 固定 / bool 無視
│     変更: WithResultを直接見て bool 正しく返す + QueueFull時に enqueueFallbackMaps へ退避
│     ownership before: QueueFullでも fallbackから除去 → orphan
│     ownership after: QueueFull→ fallback保持 → later drainで transfer
│     QueueFull path: enqueueFallbackMaps
│     Shutdown path: 同上（shutdownDiscardとして別途）
│     RT/NonRT: NonRT
│     必要test: D7-T4, T5, T15
│     rollback: 2関数差し戻し
│
├─ Patch P3: SnapshotCoordinator.cpp (startFade/completeFade)
│  └─ result closure (既に B として正しいが D7-1で再確認: 現行既に if (!result) quarantine あり)
│     現行: if (!result) quarantineRetireSink → B (既に閉じている)
│     変更: なし（再確認のみ）または DSP と同様に QueueFull時の quarantine を明示
│     ownership before/after: Bで閉じている
│     QueueFull path: quarantine
│     Shutdown path: quarantine or shutdownDiscard
│     RT/NonRT: NonRT Timerからのみ
│     必要test: D7-T6, T7
│     rollback: なし
│
├─ Patch P4: DSPLifetimeManager.cpp (retire 2経路)
│  └─ QueueFull ownership closure (ignoreUnused → quarantine)
│     現行: ignoreUnused(result) → D
│     変更: if (result == QueueFull) quarantineRetire または上位へ QueueFull 伝播
│     現行共通helper retire(dsp) → retire(dsp,0) で1修正で2件閉じる
│     ownership before: D orphan
│     ownership after: B (quarantine) or C (propagation)
│     QueueFull path: quarantine
│     Shutdown path: shutdownDiscard
│     RT/NonRT: NonRT
│     必要test: D7-T8, T9
│     rollback: 1 helper差し戻し
│
├─ Patch P5: ConvolverProcessor.Lifecycle.cpp (oldState/sc)
│  └─ result propagation — conditional (D7-4)
│     現行: void 無視 → D
│     変更候補: WithResult化 → QueueFull を上位へ伝播、上位で B
│     現行上位は void のため終端まで追跡が必要、CONDITIONAL
│     ownership before: D
│     ownership after: C (上位伝播後に B)
│     QueueFull path: 上位へ QueueFull
│     Shutdown path: 上位へ Shutdown
│     RT/NonRT: NonRT
│     必要test: D7-T10, T11
│     rollback: 2箇所 + 上位変更
│
└─ Gate P6: resetFadeStateAndRetireTarget (SnapshotCoordinator.h:60)
   └─ RT ownership closure — separate gate (D7-R)
      現行: direct enqueueRetire, RTから呼ばれ得るため WithRetry除外
      変更: なし（RT境界維持）。NonRT側の既存 mechanism で搬送可能か別途確認
      RT/NonRT: RT
      必要test: D7-T12
      rollback: なし
```

---

## D7 PASS条件 チェックリスト

```text
[x] 9 orphanの実コード上のsinkを再確定（上記 9件、DSPは共通helperで1修正で2件）
[x] 共通修正可能なsinkを統合（DSP 2→1、Cache 2は共通 tryEnqueue、Snapshot 2は同型 quarantine）
[x] Cache closureを具体的patchに固定（P2: tryEnqueue を bool 正しく + fallback）
[x] Snapshot closureを具体的patchに固定（P3: 既に B、再確認のみ）
[x] DSP closureを具体的patchに固定（P4: ignoreUnused → quarantine）
[x] Convolverはresult propagationの終端まで追跡（P5: 上位伝播先が void のため Conditional）
[x] resetFadeをRT gateとして分離（P6: D7-R）
[x] RTからenqueueWithRetry/quarantineを呼ばない（resetFadeはRTのまま）
[x] D3 wrapper semanticsを変更しない（P1のみ D3で固定した意味論を実装）
[x] Terminal bounded化を混ぜない（P-4 維持）
[x] 各patchにtestを対応付け（D7-T1〜T15）
[x] double ownership / double delete / disappearanceを禁止（T15）
[x] I4 D15.2 conservationを維持（QueueFullを right-hand に追加せず B/C へ再計上）
[x] production source変更 = 0
```

### D7 判定

**CONDITIONAL PASS** — 実装対象がほぼ閉じ、各patchの変更境界とtestが一意に固定されたが、**Convolverの上位伝播先 ownership-bearing containerが未確定**のためConditional。Convolverの終端追跡で上位の `AudioEngine` quarantine authority への到達可能性が確定すれば PASS

### D7後の実装順序（指示どおり）

```text
D8-1 → Wrapper + Cache (P1+P2) — QueueFull → false → fallback → retry の最初の完全な閉ループを証明
D8-2 → Snapshot + DSP (P3+P4)
D8-3 → Convolver (P5)
D8-4 → resetFade（別gate P6）
```

*本監査は read-only であり、production source 変更 0、Terminal bounded化なし、D14/D15 改訂なしで実施された。*
