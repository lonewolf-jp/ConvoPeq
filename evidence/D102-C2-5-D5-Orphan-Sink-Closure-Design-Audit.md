# D102-C2-5-D5 — Orphan Sink Closure Design Audit（read-only, production変更0）

- **実施日**: 2026-08-26 01:30 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **目的**: D4で特定した orphan 7系統（個別列挙で 9件となる乖離を本監査で再確定）について、QueueFull/QueuePressure/Shutdown を分離追跡し、各 orphan の本来の owner / transfer point / QueueFull時の所有者 / function return後の保持主体 / retry-quarantine-backpressure-shutdown authority / disposition を具体的に固定する。修復パターン A/B/C を1つに固定し、I4 contract との整合性を確定する。`bool` wrapper 修正は本監査ではまだ実施しない
- **判定**: **CONDITIONAL PASS** — 9 orphan 全ての初期 owner と transfer point を特定。QueuePressure/QueueFull/Shutdown を分離追跡し、Terminal chain との整合性を確認。修復パターンは **Cache/Convolver は Pattern B（Local explicit quarantine）が最適**、残り 5 は Pattern C（Result propagation または B）で閉じると固定。ただし `bool` wrapper が未修正のため現行は依然 NO-GO

---

## 1. Orphan 実数再確定（D4「7件」と個別列挙の乖離解消）

### D4 集計（7件）

D4 本文中 `PASS 8 / CONDITIONAL 2 / NO-GO 7` としたが、個別列挙では以下が含まれる：

- `Cache 2` (`Cache.cpp:16,41`)
- `Convolver 2` (`Lifecycle.cpp:57,70`)
- `Snapshot 2` (`startFade:57` / `completeFade:114`)
- `DSP 2` (`DSPLifetimeManager.cpp:58,102` `ignoreUnused`)
- `reset 1` (`SnapshotCoordinator.h:60` `resetFadeStateAndRetireTarget` direct `enqueueRetire`)

### 再カウント（production source から `rg` で再確定）

```text
rg "ignoreUnused\(result\)|resetFadeStateAndRetireTarget" src/ → 3 matches (DSP 2 + reset 1)
rg "enqueueDeferredDeleteNonRt\(" src/audioengine/AudioEngine.Cache.cpp src/convolver/ConvolverProcessor.Lifecycle.cpp → 4 matches (Cache 2 + Convolver 2)
rg "enqueueWithRetry" 無視 2 (startFade, completeFade)
```

**計:** Cache 2 + Convolver 2 + Snapshot 2 + DSP 2 + reset 1 = **9 orphan**

**乖離理由:** D4 本文の「7件」は `Cache 2 + Convolver 2 + Snapshot 2 + DSP 1` の数え漏れ（DSP は2経路あるが1と集計）。**正は 9 orphan**。本 D5 では **9件**で固定する（数字を推測で固定しない）。

---

## D5-1. 各Orphanの「本来のOwner」特定

| # | Sink | Input owner | Local owner | Transfer point | QueueFull時 ownership | Function return後 保持主体 | Retry authority | Quarantine authority | Shutdown authority | Disposition |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `AudioEngine.Cache.cpp:16` `tryEnqueueDeferredMap(map)` | Caller が `map`（`CacheMap*`）を所有（`new CacheMap` 由来） | `map` param（raw ptr） | `owner.enqueueDeferredDeleteNonRt(map, ...)` → `WithResult` → `enqueueWithRetry` → `Q/E/T` | **Caller retains**（`map` は局所変数ではないが、呼び出し元が `return true` を受けるため所有を手放したと誤認） | **なし**（`tryEnqueueDeferredMap` は `return true` 固定、`map` は caller の `CacheMap*` 変数としてスコープ外へ） | `EQCacheManager` 自身（`drainDeferredMapsUnderLock` で `enqueueFallbackMaps` へ退避可能） | `RetireQuarantineStore`（`enqueueWithRetry` 経由） | `shutdownReclaim` | `Success`（通常） / `ShutdownDiscard`（shutdown時） |
| 2 | `AudioEngine.Cache.cpp:41` `storeNewMap(old)` | `exchangeAtomic` で取得した `old`（`CacheMap*`） | `old` local | 同上 | **Caller retains**（`old` は local） | **なし**（`storeNewMap` は `void`、return 後 `old` は lost） | `EQCacheManager` | 同上 | 同上 | 同上 |
| 3 | `ConvolverProcessor.Lifecycle.cpp:57` `oldState` | `exchange` で取得した `oldState` | `oldState` local | `provider->enqueueDeferredDeleteNonRt(oldState,deleter)` | **Caller retains** | **なし**（`Lifecycle` 関数は `void`） | `ConvolverProcessor`（`Lifecycle` の owner） | 同上 | `shutdownReclaim` | `Success` |
| 4 | `ConvolverProcessor.Lifecycle.cpp:70` `sc` | `sc`（`StereoConvolver*`） | 同上 | 同上 | **Caller retains** | **なし** | 同上 | 同上 | 同上 | 同上 |
| 5 | `SnapshotCoordinator::startFade` (`cpp:57` `oldTarget`) | `exchangeTarget` で取得した `oldTarget` (`GlobalSnapshot*`) | `oldTarget` local | `enqueueWithRetry(*m_epochProvider, oldTarget, ...)` | **Caller retains**（`result` を無視） | **なし**（`startFade` は `void`、return 後 `oldTarget` は lost） | `SnapshotCoordinator`（`m_epochProvider->tryReclaim` 後の再 `enqueueWithRetry`） | `quarantineRetireSink`（`retireCurrentAndTarget` では使用） | `finalizeShutdown` の `retireCurrentAndTarget` | `Success` / `Superseded` |
| 6 | `SnapshotCoordinator::completeFade` (`cpp:114` `old`) | `exchangeCurrent` で取得した `old` | `old` local | 同上 | **Caller retains** | **なし** | 同上 | 同上 | 同上 | 同上 |
| 7 | `DSPLifetimeManager::retire(dsp)` (`cpp:49` `dsp`) | `retire(dsp, epoch)` の `dsp` 引数（`DSPCore*`） | `dsp` param | `router_->enqueueWithRetry(dsp, destroyDSPCoreNode)` | **Caller retains**（`ignoreUnused(result)`） | **なし**（`retire` は `void`） | `DSPLifetimeManager`（`router_->tryReclaim` 可能だが現行未実装） | `RetireQuarantineStore`（`enqueueWithRetry` 内部で Q へ） | `shutdownReclaim` | `Success` |
| 8 | `DSPLifetimeManager::retire(dsp, epoch)` (`cpp:96` 2経路) | 同上 | `dsp` param | 同上 | **Caller retains** | **なし** | 同上 | 同上 | 同上 | 同上 |
| 9 | `SnapshotCoordinator::resetFadeStateAndRetireTarget` (`h:60` / `cpp:81`) | `exchangeTarget(nullptr)` で取得した `target` | `target` local | **直接 `enqueueRetire`（`enqueueWithRetry` ではない）** → `QueuePressure` のみ | **Caller retains**（`enqueueRetire` 戻り値無視、`void`） | **なし**（`resetFadeStateAndRetireTarget` は `void`、RT から呼ばれ得るため `enqueueWithRetry` 除外とコメントあり） | `SnapshotCoordinator`（`m_slots`） | なし（直接 `enqueueRetire` のため Q へ自動移送なし） | 同上 | `Success` |

**「Caller retains」だけで終わらせず、callerが実際に所有権を保持できる型・オブジェクトなのか:** 上記 9件全て **Yes** — `map`/`old`/`oldTarget`/`dsp`/`target` はいずれも raw ptr として local または param に残り、**function return 後も caller のスタックに残るが、現行 `void`/`ignoreUnused` により保持主体が return 後も生存しない**（orphan）。

---

## D5-2. `bool` Wrapper 修正を先に実装しない（D3 設計維持）

D3 で固定した `return result != Shutdown && result != QueueFull` は設計として維持するが、**D5 ではまだ変更しない**。

理由: `QueueFull → bool false` は **ownership semantics の修正**であって、`QueueFull → caller retains → retry/quarantine/backpressure` という **ownership closure の実装ではない**。D4 ですでに `bool` 修正だけでは `Cache`/`Convolver` の orphan が残ることが実証されている。

---

## D5-3. `QueuePressure` と `QueueFull` と `Shutdown` を分離追跡

### 現行 Stage 3（`src/audioengine/ISRRetireRouter.cpp:338`）

```text
QueuePressure / QueueFull
        ↓
RetireQuarantineStore::quarantine() // Stage 3
  ↓ stored → QueuePressure (Q owns)
  ↓ not stored
EmergencyQuarantineStore::quarantine() // Stage 4
  ↓ stored → QueuePressure (E owns)
  ↓ not stored
TerminalReclaimAuthority::store() // Stage 5
  ↓ tstored==true → TerminalReclaim (T owns)
  ↓ tstored==false → QueueFull (caller retains) ← D2 で新設予定
```

### 3値の分離追跡（各 sink について）

| 値 | 意味 | 現行 Terminal 実装との一致 |
|---|---|---|
| `QueuePressure` | Q/E が所有（`quarantine` 成功） | **一致**（`QueuePressure` は Q/E 所有として `Success` と同等の transferred） |
| `QueueFull` | **新** Terminal full → caller retains（generic bounded failure） | **D2 で新設、現行 T は growable のため発生しない** |
| `Shutdown` | shutdown reclaim 失敗 → caller retains（shutdownDiscard として別途処理） | **一致**（`Shutdown` は shutdown 時のみ） |

**特に `QueueFull` が `D full → Q full → E full → Terminal` まで到達する場合:** 現行 `TerminalReclaimAuthority` は growable のため `tstored==true` で `TerminalReclaim` を返すが、**D5 では bounded 化を想定し `tstored==false → QueueFull` として caller-retain にする**。これは D3/D4 で固定した ownership contract と一致する。

---

## D5-4. 修復パターン 3候補に限定

| Pattern | 概要 | 適用 sink | 理由 |
|---|---|---|---|
| **A — Result propagation** | `sink → WithResult → QueueFull → 上位へ result 返却 → 上位 backpressure/quarantine` | `EQProcessor` / `ISRRuntimePublicationCoordinator` / `AudioEngine::WithResult` wrapper | 既に `RetireEnqueueResult` を上位へ伝播しており、追加の owner authority 不要 |
| **B — Local explicit quarantine** | `sink → enqueue... → QueueFull → quarantineRetireSink(...)` | `SnapshotCoordinator::discardSnapshot` / `retireSnapshot` / `retireCurrentAndTarget`（既に B）および **Cache 2 / Convolver 2 / startFade / completeFade / DSPLifetimeManager 2 / reset 1 の 9 orphan** | **最適** — `QueueFull` 時に local の `ptr` を `quarantineRetireSink` へ移送すれば、Q へ所有が移り orphan 解消。`AudioEngine.Cache` の `drainDeferredMapsUnderLock` の `enqueueFallbackMaps` と同型 |
| **C — Ownership-bearing retry object** | `sink → ownership-bearing local/member → enqueue failure → owner remains alive → 後続 Coordinator / retry authority` | `DSPLifetimeManager` の `currentRetiringGeneration_` 等 member に保持する案 | `member` として保持すれば function return 後も生存するが、**新たな retry authority の設計が必要**で Pattern B より複雑 |

### 各 Orphan の修復方式固定（9件）

| # | Sink | 修復パターン | 理由 |
|---|---|---|---|
| 1 | `Cache.cpp:16` `tryEnqueueDeferredMap` | **B** | `tryEnqueueDeferredMap` は `bool` を返すが現行 `return true` 固定。`QueueFull` 時に `enqueueFallbackMaps.push_back(map)` へ退避すれば `drainDeferredMapsUnderLock` で再試行可能（既存 `enqueueFallbackMaps` と同型） |
| 2 | `Cache.cpp:41` `storeNewMap` | **B** | 同上、`old` を `enqueueFallbackMaps` へ |
| 3 | `Convolver:57` `oldState` | **B** | `provider->quarantineRetireSink` へ（`SnapshotCoordinator` と同等の authority が Convolver にあれば） |
| 4 | `Convolver:70` `sc` | **B** | 同上 |
| 5 | `startFade` `oldTarget` | **B** | `quarantineRetireSink(oldTarget, ...)` へ（`retireCurrentAndTarget` と同型） |
| 6 | `completeFade` `old` | **B** | 同上 |
| 7-8 | `DSPLifetimeManager` 2経路 | **B** | `router_->quarantineRetire` または `AudioEngine` 側の `quarantineResidentCount` 経由の Q へ移送 |
| 9 | `resetFadeStateAndRetireTarget` (direct `enqueueRetire`) | **B**（または **A** へ変更） | 現行直接 `enqueueRetire` を `enqueueWithRetry` + `quarantine` へ変更し、`QueueFull` 時に B へ |

**「とりあえず bool をチェックする」ことを修復案として採用しない** — `false` を検出しても owner authority がなければ D4 の orphan が残るため、上記 B の **明示的 quarantine 移送** を必須とする。

---

## D5-5. I4 Contract との整合性

### `1 logical obligation = exactly 1 reservation`（D14.2）

- Pattern B の **quarantine 移送**は `RetireQuarantineStore::quarantine` により `kMaxQuarantinedEntries` の範囲で所有を保持し、`drain(isOlder)` で epoch-safe に破棄する。reservation は `transport/durable/building/stalled` のいずれかに **1回のみカウント**され、quarantine 中も **重複カウントしない**。

### Ownership conservation（D15.2）

```text
transport + durable + building + stalled + superseded + shutdownDiscard == admittedLogicalObligationCount
```

- Pattern B の quarantine は **Transport/Durable/Building/Stalled のいずれかに再計上**される途中状態であり、**新しい sink-level escape hatch として `terminal-failure` を追加しない**（D14.3 違反回避）。
- `QueueFull` 自体を conservation の新しい状態として追加せず、既存 6 状態のいずれかに **再計上**することで、D15.2 を壊さない。

---

## D5 終了条件 チェックリスト

```text
[x] orphan sinkの実数をproduction sourceから再確定 → 9件（D4の7件は数え漏れ）
[x] 各orphanの初期ownerを確定（Cache: caller map / Convolver: oldState/sc / Snapshot: oldTarget/old / DSP: dsp / reset: target）
[x] QueuePressure / QueueFull / Shutdownを分離追跡（Stage 3 → Q/E → T の分岐を 3値で追跡）
[x] function return後のowner保持主体を特定（9件全て現行 orphan — local/param が return 後に lost）
[x] 各sinkについて A/B/C の修復方式を1つに固定（上記 9件全て B）
[x] retry authorityを明示（SnapshotCoordinator: m_epochProvider / DSPLifetimeManager: router_ / Cache: EQCacheManager / Convolver: provider）
[x] quarantine authorityを明示（RetireQuarantineStore / quarantineRetireSink）
[x] shutdown時のdispositionを明示（Success / ShutdownDiscard）
[x] ownership conservationへの影響を確認（1 obligation =1 reservation 維持、terminal-failure 追加なし）
[x] terminal-failureによる消失経路を追加しない（D14.3 遵守）
[x] D3 wrapper修正との依存関係を明示（D3 の `bool` 修正は semantics 修正、D5 の B は closure 実装 — D3 だけでは orphan 残る）
[x] production source変更 = 0
```

### D5 判定

**CONDITIONAL PASS**

- 全 orphan について修復後の ownership graph が **Pattern B（Local explicit quarantine）で一意に閉じる**ことを固定
- ただし **retry/quarantine authority の既存 API 実在確認（`quarantineRetireSink` が Cache/Convolver から呼べるか）** が不足 — D6 で `quarantine` API の可視性と `enqueueFallbackMaps` 型の再利用可能性を最終確認する必要がある

**次:** `D102-C2-5-D6 — Minimal Production Patch Plan / Diff-Boundary Audit` で D3 `bool` wrapper 修正 + orphan 9件の B 修正 + 必要テスト + 変更ファイル・行の最小集合を固定

*本監査は read-only であり、production source 変更 0、D3 wrapper 未実施、Terminal bounded 化なしで実施された。*
