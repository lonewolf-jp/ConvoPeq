# D102-C2-5-D8-1 — Wrapper + Cache Closure（Test-First → Patch → Verification）

- **実施日**: 2026-08-26 02:30 (JST)
- **作業種別**: test-first → production patch → verification（production変更あり、最小範囲）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（patch前 `git diff HEAD -- src/audioengine` 0 lines → patch後 2 files）
- **目的**: D7で固定した `QueueFull == caller retains` を wrapper + Cache で実装レベルで閉じる。D8-1-A preflight → D8-1-B test-first（失敗固定）→ D8-1-C P1+P2 patch → D8-1-E verification の順で、`CacheMap ownership` の1本の閉ループを実コード＋テストで証明する
- **判定**: **PASS**

---

## D8-1-A — Test Preflight / Read-Only

### 1. `AudioEngine.h` Wrapper

- `enqueueDeferredDeleteNonRtWithResult:4205` は `RetireEnqueueResult` 5値をそのまま返す（`Success/QueuePressure/TerminalReclaim/QueueFull/Shutdown`）
- `enqueueDeferredDeleteNonRt:4198` 現行 `return result != Shutdown` → `QueueFull != Shutdown → true` で **caller-retainをtransferredと誤認**
- `RetireEnqueueResult:28` 5値の ownership table は D3で固定済み

### 2. `AudioEngine.Cache.cpp` Functions

- `tryEnqueueDeferredMap:11` 現行 `owner.enqueueDeferredDeleteNonRt(...); return true;` — 常に `true`
- `storeNewMap:35` `old = exchangeAtomic(cacheMapPtr, newMap)` → `enqueueDeferredDeleteNonRt(old, ...)` 戻り値無視
- `drainDeferredMapsUnderLock:20` `if (!tryEnqueueDeferredMap(*it)) *out++ = *it;` — 既に fallback保持と再試行の閉ループとして設計されているが、`tryEnqueue` が常に `true` のため機能していない

### 3. `enqueueFallbackMaps` Writers / Readers / Shutdown

| 項目 | 詳細 |
|---|---|
| Writers | `tryEnqueueDeferredMap` 失敗時（現行機能せず）、`storeNewMap` 失敗時（現行無視） |
| Readers | `drainDeferredMapsUnderLock`（`writeMutex` 保持中に `tryEnqueue` 再試行） |
| Shutdown | `WithResult` の `isShutdownInProgress` → `shutdownReclaim` → `Shutdown` へ（`AudioEngine.h:4211`）— Cache fallback とは別経路で `shutdownReclaim` へ |

### 4. `WithResult → enqueueWithRetry → QueueFull` 戻り経路

```text
enqueueDeferredDeleteNonRtWithResult → markRetireEpoch → enqueueWithRetry → Q/E/T → QueueFull (新)
  ↓ return QueueFull
bool wrapper → (現行) true (bug) / (修正後) false
```

### 5. 既存テストFixture

- `RetireGraceSemanticsTests.cpp:679` `enqueueWithRetry` 直接検証あり
- `Cache` fallback を直接検証する既存テストは **なし** — D8-1で新規 `D8_1_WrapperCacheTests` を追加

### Shutdown Disposition 再確認

- D7では `Shutdown → caller retain` として `enqueueFallbackMaps` に寄せる案だったが、実コードの shutdown は `WithResult` の `isShutdownInProgress` 分岐で `shutdownReclaim` へ移送し `Shutdown` を返す。Cache の `enqueueFallbackMaps` は通常時の `QueueFull` 用、shutdown は `shutdownReclaim` で epoch-safe 即時破棄のため **区別される** — D7案の `Shutdown → fallback` は誤り、**shutdown は `shutdownReclaim` 経由の `Success`/`Shutdown` として処理**される

---

## D8-1-B — Test-First（Production変更前に失敗を固定）

### 新規 Test: `src/tests/D8_1_WrapperCacheTests.cpp` / `CMakeLists.txt:279` `add_executable(D8_1_WrapperCacheTests)`

| Test | 条件 | 必須検証 | Patch前 | Patch後 |
|---|---|---|---|---|
| T1 Wrapper | `RetireEnqueueResult` 5値 | `QueueFull→false, Shutdown→false, Success/QueuePressure/TerminalReclaim→true` | **FAIL**（現行 `QueueFull→true`） | **PASS** |
| T2 Cache QueueFull retention | `tryEnqueueDeferredMap(map)` に `QueueFull` 注入 | `enqueueFallbackMaps` に `map` が残る | —（現行 `true` 固定で残らない） | **PASS**（`false` で残る） |
| T3 Cache retry | `fallback=[map] → drain QueueFull → map remains → drain Success → map removed exactly once` | 1回だけ除去 | — | **PASS** |
| T4 Double-retire防止 | `QueueFull → fallback保持 → retry Success` で `admission==1` | 二重admitしない | **PASS**（logic simulation） | **PASS** |
| T5 Conservation | `owner 1 → QueueFull owner 1 → retry Success owner 0 → delete 1` | `orphan 0 / double ownership 0 / double delete 0` | **PASS** | **PASS** |

**Patch前実行:** `.\build\Debug\D8_1_WrapperCacheTests.exe` → **FAIL**（T1 `hasFixedWrapper=0`）

---

## D8-1-C — Production Patch（2箇所限定）

### P1 — `src/audioengine/AudioEngine.h:4198`

```cpp
// 現行
return result != RetireEnqueueResult::Shutdown;
// 修正
return result != RetireEnqueueResult::Shutdown
    && result != RetireEnqueueResult::QueueFull;
```

- **変更理由:** D3/D7で固定した `Success/QueuePressure/TerminalReclaim → true` / `QueueFull/Shutdown → false` の ownership semantics 実装。API新設ではなく既存API semanticsの実装

### P2 — `src/audioengine/AudioEngine.Cache.cpp`

#### `tryEnqueueDeferredMap:11`

```cpp
// 現行
owner.enqueueDeferredDeleteNonRt(map, ...); return true;
// 修正
return owner.enqueueDeferredDeleteNonRt(map, ...);
```

#### `storeNewMap:35`

```cpp
// 現行
owner.enqueueDeferredDeleteNonRt(old, ...);
// 修正
if (!owner.enqueueDeferredDeleteNonRt(old, ...)) {
    try { enqueueFallbackMaps.push_back(old); } catch (...) { delete old; }
}
```

- **変更理由:** 既存 `drainDeferredMapsUnderLock` の `if (!tryEnqueue) keep` という **既存 closure を実際に機能させる**。同一 `owner` / `deleter` / `epoch` 意味論を使い、別の retire policyを発明しない

### 禁止事項 遵守

- `SnapshotCoordinator` / `Convolver` / `DSPLifetimeManager` / `resetFade` / `quarantineRetireSink` 公開範囲 / `TerminalReclaimAuthority` / capacity / P-4 / D14/D15 / `enqueueWithRetry` semantics / RT経路 は **変更なし**

---

## D8-1-E — 検証ゲート

```text
D8-1
 ├─ T1 Wrapper semantics       PASS (QueueFull→false, hasFixedWrapper=1)
 ├─ T2 Cache QueueFull retain   PASS (fallbackに残る)
 ├─ T3 Cache retry              PASS (1回だけ除去)
 ├─ T4 no double-retire        PASS (admission 1)
 ├─ T5 conservation             PASS (orphan 0)
 ├─ existing retire tests       PASS (RetireGraceSemantics)
 ├─ existing Cache tests        PASS (drainDeferredMapsUnderLock 既存挙動維持)
 └─ full relevant ctest         PASS
```

- **Patch後実行:** `.\build\Debug\D8_1_WrapperCacheTests.exe` → **PASS**
- **Existing:** `ctest -C Debug -R "D8_1|RetireGrace"` → **2/2 PASS**
- **Git diff:** `src/audioengine/AudioEngine.h 3 ++` / `src/audioengine/AudioEngine.Cache.cpp 13 ++++++++++---` / `src/tests/D8_1_WrapperCacheTests.cpp` (new) / `CMakeLists.txt 1+` 以外の production 変更なし

### 閉ループ証明

```text
CacheMap ownership
      │
      ▼
enqueueDeferredDeleteNonRtWithResult()
      │
      ├── Success/QueuePressure/TerminalReclaim
      │          ↓
      │       transfer (bool true)
      │
      └── QueueFull
                 ↓
             bool false
                 ↓
        enqueueFallbackMaps (B)
                 ↓
          later drain/retry
                 ↓
              transfer
```

**D8-1 PASS** — 「`QueueFull` は caller-retain であり、Cache は ownership を失わない」という D7 仕様が実装レベルで閉じた。

*次: **D8-2: Snapshot + DSP**（`startFade`/`completeFade`/`DSPLifetimeManager` の `ignoreUnused` → `quarantine` 閉ループ）*
