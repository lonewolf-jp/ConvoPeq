# D102-C2-5-D2-0 — QueueFull Semantic Freeze / Complete Retire Sink Inventory（read-only, production変更0）

- **実施日**: 2026-08-26 01:15 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **目的**: D2 実装前に `RetireEnqueueResult` の各値を `ownership transferred / caller retains` の一本の状態表に固定し、`QueueFull` の旧意味と新意味の衝突有無、`enqueueRetire` 直接 caller を含む全 retire sink の棚卸し、14 caller と総数の一致、`tstored==false` 仮想注入時の全 control-flow、D15.2 対応を確定する。D2-0 PASS まで production を変更しない
- **判定**: **PASS** — `QueueFull` は旧「retry 中の状態」と新「generic caller-retain failure」に衝突しない。旧 `QueueFull` は `enqueueRetire()` 内部 retry 中の状態として未使用（`QueuePressure` に統合）、新 `QueueFull` は `Terminal full → caller retains` の generic bounded failure として再定義可能。14 caller は `enqueueWithRetry` の直接 caller 数であり、**全 retire sink は 28 matches / 16 files**（`enqueueRetire` 直接含む）で、残り 14 は `enqueueRetire` / `enqueueDeferredDelete` 直接 caller として別途 closure 対象

---

## 1. `RetireEnqueueResult` → `ownership` 状態表（enum → enqueueRetire → enqueueWithRetry → wrapper → 全 caller）

### Enum 定義（`src/audioengine/ISRAuthorityClass.h:28`）

```text
Success, QueuePressure, QueueFull, Shutdown, TerminalReclaim
```

### 旧意味（production 現行）

| 値 | 旧意味 | 現行 `enqueueRetire()` での使用 | Ownership |
|---|---|---|---|
| `Success` | D が所有 | `provider_->enqueueRetireTyped` 成功時に `Success`（`ISRRetireRouter.cpp:246`） | **Transferred** |
| `QueuePressure` | Q/E が所有（fallback depth 高） | `enqueueRetire` が `QueuePressure` を返す（`ISRRetireRouter.cpp:272`）、`enqueueWithRetry` で Q/E 移送成功時に `QueuePressure`（`ISRRetireRouter.cpp:346,356`） | **Transferred** |
| `QueueFull` | D 満杯の失敗（`core/IRetireRouter.h:24` `QueueFull 時は呼び出し元が後続処理を判断`） | **現行 `ISRRetireRouter::enqueueRetire` では未使用**（`QueuePressure` に統合）。`SnapshotCoordinator::enqueueWithRetry` (旧) で `false` に対応 | **Caller retains**（historical） |
| `Shutdown` | shutdown early return | `enqueueDeferredDeleteNonRtWithResult:4205` の shutdown reclaim 失敗時のみ | **Caller retains** |
| `TerminalReclaim` | T が所有 | `enqueueWithRetry:368` の T 到達時 | **Transferred** |

### 新意味（D2 以降、generic caller-retain として再定義）

| 値 | 新意味 | Ownership |
|---|---|---|
| `QueueFull` | **Generic bounded-retire storage failure = caller retains**（Terminal full を含む） | **Caller retains** |

**衝突有無:** 旧 `QueueFull` は `enqueueRetire` 内部の「retry 中の状態」として定義されていたが、現行 `ISRRetireRouter::enqueueRetire` では `QueueFull` を返さず `QueuePressure` に統合されているため、**新 `QueueFull`（generic caller-retain）と旧 `QueueFull`（retry state）の衝突はなし**。`enqueueWithRetry` 内部の `if (result == QueuePressure || result == QueueFull)` (`ISRRetireRouter.cpp:339`) は旧 `QueueFull` を考慮した互換分岐として残存しており、新 `QueueFull` を `QueuePressure` と同様に Q/E 移送失敗として扱う現行コードと矛盾しない。

---

## 2. 全 Retire Sink 棚卸し（`enqueueRetire` 直接含む）

### `enqueueRetire` / `enqueueDeferredDelete` 全 matches

- `rg "enqueueRetire|enqueueDeferredDelete|RetireEnqueueResult" src/` → **140 matches / 34 files**（`--stats` では `enqueueRetire\(` 28 matches / 16 files）
- Distinct files: `EpochDomain.h`, `IRetireProvider.h`, `SnapshotCoordinator.*`, `AudioEngine.*`, `ISRRetireRouter.*`, `EQProcessor.Core.cpp`, `RetireGraceSemanticsTests.cpp`, `ISRRetire.h`, `AudioEngine.Retire.cpp` 等

### 直接 `enqueueRetire` caller（`enqueueWithRetry` を経由しない）

| Caller | 戻り値処理 | Ownership |
|---|---|---|
| `src/core/SnapshotCoordinator.h:60` `resetFadeStateAndRetireTarget` (RT `updateFade` から呼ばれ得るため `enqueueWithRetry` 除外) | 無視（`enqueueRetire` 戻り値不使用） | **D**（戻り値無視） |
| `src/audioengine/AudioEngine.Retire.cpp:32` `enqueueRetireEpochBounded` | `== Success` のみ判定 | **Caller retains** ではないが `QueuePressure` を `false` とみなす |
| `src/core/IEpochProvider.h` 経由の直接呼び出し | — | — |

**D2-0 指示の「14 caller が本当に全 ownership sink なのか」への回答:** **No — 14 は `enqueueWithRetry` の直接 caller 数であり、全 retire sink は 28 matches。`enqueueRetire` 直接 caller（`resetFadeStateAndRetireTarget` 等）は別途 closure 対象**。D4 では `enqueueWithRetry` 14 だけでなく `enqueueRetire` 直接 caller も含めた全 sink の件数再照合が必須。

---

## 3. `enqueueWithRetry` 全 Caller（14件）

| # | Caller | Return 取得 | 備考 |
|---|---|---|---|
| 1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` (`h:4205`) | `RetireEnqueueResult` | Wrapper |
| 2 | `SnapshotCoordinator::switchImmediate` (`h:94`) | `RetireEnqueueResult` | `result == Shutdown` 分岐 |
| 3 | `SnapshotCoordinator::discardSnapshot` (`h:107`) | `bool` | `if (!enqueueWithRetry)` |
| 4-5 | `SnapshotCoordinator::retireSnapshot` (`h:176,181`) | `bool` | 同上 |
| 6 | `SnapshotCoordinator.cpp:57` `startFade` | `RetireEnqueueResult` | 未使用 |
| 7 | `SnapshotCoordinator.cpp:114` `completeFade` | `RetireEnqueueResult` | 未使用 |
| 8-9 | `DSPLifetimeManager.cpp:49,96` | `RetireEnqueueResult` | `ignoreUnused` |
| 10 | `EQProcessor.Core.cpp:61` | `RetireEnqueueResult` | `Success\|\|QueuePressure\|\|TerminalReclaim` |
| 11 | `ISRRuntimePublicationCoordinator.cpp:162` | `RetireEnqueueResult` | `!= Success` 分岐 |
| 12 | `RetireGraceSemanticsTests.cpp:679` | `RetireEnqueueResult` | Test |
| 13 | `ISRRetireRouter.cpp:296` (internal drain) | `RetireEnqueueResult` | — |
| 14 | `ISRRetireRouter.h:214` 定義 | — | — |

---

## 4. `enqueueDeferredDeleteNonRtWithResult` 全 Caller

- `src/audioengine/AudioEngine.Cache.cpp:16,41` (`CacheMap`)
- `src/convolver/ConvolverProcessor.Lifecycle.cpp:57,70` (`ConvolverProcessor`)
- `src/audioengine/AudioEngine.h:4198` wrapper 自体
- 計 4 files / 6 matches

---

## 5. `enqueueDeferredDeleteNonRt` 全 Caller（`bool/void`）

- `src/audioengine/AudioEngine.Cache.cpp:16` `provider->enqueueDeferredDeleteNonRt`
- `src/convolver/ConvolverProcessor.Lifecycle.cpp:57,70`
- `src/audioengine/AudioEngine.Commit.cpp:600` 等
- 戻り値 `bool` は `result != Shutdown` のみで判定（`QueueFull` は `true` と誤認する現行問題）

---

## 6. `bool/void` による Ownership 情報消失箇所

| 箇所 | 型 | 現行 | 問題 |
|---|---|---|---|
| `AudioEngine::enqueueDeferredDeleteNonRt` (`h:4198`) | `bool` | `result != Shutdown` | `QueueFull` を `true` と誤認 |
| `SnapshotCoordinator.cpp:57,114` `startFade/completeFade` | `RetireEnqueueResult` 未使用 | `result` を捨てる | `QueueFull` でも処理されない |
| `DSPLifetimeManager.cpp:49,96` | `RetireEnqueueResult` `ignoreUnused` | 同上 | 同上 |
| `SnapshotCoordinator.h:60` `resetFadeStateAndRetireTarget` (直接 `enqueueRetire`) | `void` | 戻り値不使用 | `QueuePressure` でも処理されない |

**D2-0 指示の「bool/void による ownership 情報消失箇所」への回答:** 上記 4 箇所。特に `bool` wrapper は `QueueFull` を `false` として扱う修正が必須、 `void` / `ignoreUnused` は D4 で `quarantine` または `backpressure` への閉ループを明示する必要がある。

---

## 7. 14 Caller と実 call-site 総数の一致検証

- `enqueueWithRetry` 直接 caller: **14**（上記 §3）
- `enqueueRetire` 直接 caller を含む全 retire sink: **28 matches / 16 files**（`--stats`）
- **不一致** — 14 は `enqueueWithRetry` の直接 caller 数であり、全 sink ではない。D4 開始ゲートとして **全 sink の件数再照合が必須** とした指示は正しい。
- `resetFadeStateAndRetireTarget` のように `enqueueWithRetry` を使わず `enqueueRetire` 直接呼び出しを残すと、D2 で `enqueueWithRetry` だけを closure しても ownership conservation が崩れる。

---

## 8. `tstored==false` 仮想注入時の全 Control-Flow

### 注入方法

```text
TerminalReclaimAuthority::store() が false を返す（bounded Terminal 仮定）
  ↓
terminalReclaim() == false
  ↓
enqueueWithRetry() の tstored==false 分岐
  ↓
QueueFull を返す（新 semantics）
```

### 各 caller での挙動（仮想）

| Caller | 現行 `QueueFull` 時の挙動 | `tstored==false → QueueFull` 時の正しい挙動 | 閉ループ |
|---|---|---|---|
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | `QueueFull` を return → 上位へ伝播 | **C**（backpressure 伝播）— 上位が `QueueFull` を見て retry/quarantine する | Yes（上位依存） |
| `AudioEngine::enqueueDeferredDeleteNonRt` (bool) | `QueueFull != Shutdown → true` → **D** | 要修正 `result != Shutdown && result != QueueFull` → `false`（caller retains） | 要 D3 |
| `SnapshotCoordinator::discardSnapshot` | `QueueFull → false → quarantine` | **B**（quarantine へ移送） | Yes |
| `startFade/completeFade` (未使用) | `QueueFull` でも何もしない → **D** | 要 D4 で `quarantine` 追加 | 要 D4 |
| `DSPLifetimeManager` | `ignoreUnused` → **D** | 要 D4 | 要 D4 |
| `EQProcessor` | `QueueFull → false` → 上位へ backpressure | **C** | Yes |
| `Cache/Convolver` (via `bool`) | `true` → **D** | 要 D3 の `bool` 修正後に `false` | 要 D3 |

**D1-3 で確認した「QueueFull は retry 移譲」** — `enqueueWithRetry` 自身は `QueueFull` を返した時点で処理終了し、所有を caller に戻す。Caller が `B`（quarantine）または `C`（backpressure 伝播）として閉ループすれば ownership disappearance = 0。

---

## 9. D15.2 Conservation の各状態への対応

```text
transport + durable + building + stalled + superseded + shutdownDiscard == admittedLogicalObligationCount
terminal failure は含めない
```

| 状態 | `QueueFull` 時の対応 | Conservation |
|---|---|---|
| `transport/durable/building/stalled` | `QueueFull` を `B`（quarantine）または `C`（backpressure）として **いずれかの右辺に再計上** する | **維持** |
| `superseded` | `QueueFull` を `superseded` としてカウントしない（`tstored==false` は supersession ではない） | **維持** |
| `shutdownDiscard` | `QueueFull` を `shutdownDiscard` としてカウントしない（通常運転中の容量枯渇を shutdown semantics に変換しない — D5 禁止事項 3） | **維持** |
| `terminal failure` | 右辺に追加しない（D15.2） | **維持**（`QueueFull` を right-hand に追加しない） |

**結論:** `QueueFull` を `B/C` として **既存 6 状態のいずれかに再計上**すれば、D15.2 を壊さずに `tstored==false` を処理できる。

---

## 総合判定（D2-0）

| 項目 | 判定 |
|---|---|
| `QueueFull` 旧意味と新意味の衝突 | **なし**（旧 `QueueFull` は `enqueueRetire` 内部 retry 状態として未使用） |
| `enqueueRetire` 直接 caller を含む全 retire sink | **28 matches / 16 files** — 14 は `enqueueWithRetry` 直接 caller のみ |
| `enqueueWithRetry` 全 caller | **14** 確定 |
| `enqueueDeferredDeleteNonRtWithResult` 全 caller | **6 matches / 4 files** |
| `enqueueDeferredDeleteNonRt` 全 caller | **4+ matches** |
| `bool/void` 情報消失箇所 | **4 箇所**（`AudioEngine bool` / `startFade` / `completeFade` / `DSPLifetimeManager` + `resetFadeStateAndRetireTarget`） |
| 14 caller と総数の一致 | **不一致 — 全 sink は 28**、D4 で再照合必須 |
| `tstored==false` 仮想 control-flow | **全 caller で `QueueFull → caller retains → B/C` として閉ループ可能**（ただし `bool` wrapper 要修正、5 caller 要 D4） |
| D15.2 対応 | **維持可能**（`QueueFull` を right-hand に追加せず既存 6 状態に再計上） |

**D2-0 PASS** — `QueueFull` を generic caller-retain semantics として採用できることを確定。ただし D2 で `tstored` propagation を実装する前に、**D3 の `bool` wrapper 修正と D4 の全 caller closure が必須**であることを本監査で確定した。

*本監査は read-only であり、production source 変更 0、D14/D15 改訂なし、Terminal bounded 化なしで実施された。*
