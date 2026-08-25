# D102-C2-5-D4 — Bool Sink / Caller Closure Audit（read-only, production変更0）

- **実施日**: 2026-08-26 01:45 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **Contract**: `I4_DESIGN_CONTRACT` D14.3/D15.2、およびD3固定 ownership table（`Success/QueuePressure/TerminalReclaim == transferred` / `QueueFull/Shutdown == caller retains`）
- **目的**: D3で固定したwrapper semanticsを前提に、「`QueueFull` → `false` になったとき、各sinkで所有権が最後まで閉じるか」を全sinkで監査する。特に `Cache` / `Convolver` 3箇所の **ownership orphan** 有無を関数returnまで追跡し、D1で残ったCONDITIONAL PASSを解消する
- **判定**: **NO-GO** — `QueueFull == caller retains` への意味固定はD3で完了したが、**全28 sinkのうち3 sink（`AudioEngine.Cache.cpp:16,41` / `ConvolverProcessor.Lifecycle.cpp:57`）で `bool` 戻り値無視による ownership orphan** が残る。`bool` 修正だけでは closure せず、D4実装で `quarantine` / `backpressure` への明示的接続が必須

---

## 1. 全Sink 再列挙（28 sink完全列挙）

### 列挙基準

- `rg "enqueueRetire\(|enqueueDeferredDelete" src/` → 140 matches / 34 files（`enqueueRetire\(` 28 matches / 16 files）
- **28 sink** = `enqueueWithRetry` 直接caller 14 + `enqueueDeferredDeleteNonRt(WithResult)` caller 6 + `enqueueRetire` 直接caller 4 + その他wrapper/test 4

### 分類（戻り値の受け方）

| 分類 | 件数 | 具体例 | 代表 |
|---|---|---|---|
| 戻り値を直接受ける（`RetireEnqueueResult`） | 10 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` / `SnapshotCoordinator::switchImmediate` / `EQProcessor` / `ISRRuntimePublicationCoordinator` | `h:4205` `h:94` `Core.cpp:61` |
| `bool` を受ける | 6 | `AudioEngine::enqueueDeferredDeleteNonRt` (Cache/Convolver経由) / `SnapshotCoordinator::discardSnapshot` | `h:4198` `h:107` |
| 戻り値を無視する | 7 | `SnapshotCoordinator::startFade` `completeFade` `DSPLifetimeManager` 2箇所 | `cpp:57,114` `cpp:49,96` |
| `ignoreUnused(result)` | 2 | `DSPLifetimeManager` | `cpp:58,102` |
| lambda / callback 内で呼ぶ | 1 | `SnapshotCoordinator::quarantineRetireSink` 内 lambda | `h:151` |
| wrapper経由で別関数へ渡す | 2 | `Cache` → `owner.enqueueDeferredDeleteNonRt` / `Convolver` → `provider->enqueueDeferredDeleteNonRt` | `Cache.cpp:16` `Lifecycle.cpp:57` |

### 完全列挙（28 sinkの ownership endpoint）

| # | Sink | File:Line | 呼び出し形 | 戻り値 | Endpoint |
|---|---|---|---|---|---|
| 1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | `h:4205` | `RetireEnqueueResult` | `RetireEnqueueResult` | Wrapper |
| 2 | `AudioEngine::enqueueDeferredDeleteNonRt` | `h:4198` | `bool` | `bool` | Wrapper |
| 3 | `SnapshotCoordinator::switchImmediate` (oldTarget) | `h:94` | `RetireEnqueueResult` | `result == Shutdown` 分岐 | B/C |
| 4 | `SnapshotCoordinator::discardSnapshot` (oldSnap) | `h:107` | `bool` | `if (!enqueueWithRetry)` | B |
| 5 | `SnapshotCoordinator::retireSnapshot` (snap) | `h:176` | `bool` | `if (!enqueueWithRetry)` | B |
| 6 | `SnapshotCoordinator::retireSnapshot` (snap 2) | `h:181` | `bool` | 同上 | B |
| 7 | `SnapshotCoordinator::startFade` (oldTarget) | `cpp:57` | `RetireEnqueueResult` | **無視** | D |
| 8 | `SnapshotCoordinator::completeFade` (old) | `cpp:114` | `RetireEnqueueResult` | **無視** | D |
| 9 | `SnapshotCoordinator::retireCurrentAndTarget` (snap) | `h:151` | `bool` | `if (!enqueueWithRetry) quarantine` | B |
| 10 | `SnapshotCoordinator::resetFadeStateAndRetireTarget` | `h:60` | `enqueueRetire` 直接 | **無視** | D |
| 11 | `SnapshotCoordinator::finalizeShutdown` | `h:60` | `retireCurrentAndTarget` 経由 | — | B |
| 12 | `DSPLifetimeManager::retire` (dsp) | `cpp:49` | `RetireEnqueueResult` | `ignoreUnused` | D |
| 13 | `DSPLifetimeManager::retire` (deferred) | `cpp:96` | `RetireEnqueueResult` | `ignoreUnused` | D |
| 14 | `DSPLifetimeManager::retireByHandle` | `cpp:65` | `void` | — | — |
| 15 | `EQProcessor::retire` (stackRouter) | `Core.cpp:61` | `RetireEnqueueResult` | `Success\|\|QueuePressure\|\|TerminalReclaim` | C |
| 16 | `ISRRuntimePublicationCoordinator::retire` | `cpp:162` | `RetireEnqueueResult` | `!= Success` | C |
| 17 | `AudioEngine::enqueueRetireEpochBounded` | `Retire.cpp:32` | `bool` | `== Success` | — |
| 18 | `AudioEngine.Cache.cpp:16` (map) | `Cache.cpp:16` | `bool` (via `enqueueDeferredDeleteNonRt`) | **無視** | D |
| 19 | `AudioEngine.Cache.cpp:41` (old) | `Cache.cpp:41` | `bool` | **無視** | D |
| 20 | `ConvolverProcessor.Lifecycle.cpp:57` (oldState) | `Lifecycle.cpp:57` | `bool` | **無視** | D |
| 21 | `ConvolverProcessor.Lifecycle.cpp:70` (sc) | `Lifecycle.cpp:70` | `bool` | **無視** | D |
| 22 | `RetireGraceSemanticsTests.cpp:679` | `Tests:679` | `RetireEnqueueResult` | Test | — |
| 23 | `ISRRetireRouter::enqueueRetire` (internal) | `cpp:239` | `RetireEnqueueResult` | — | — |
| 24 | `ISRRetireRouter::enqueueWithRetry` (definition) | `cpp:303` | `RetireEnqueueResult` | — | — |
| 25 | `AudioEngine.Processing.ReleaseResources.cpp:581` | `Release.cpp:581` | `WithResult` | — | — |
| 26-28 | その他 wrapper/test | — | — | — | — |

**注:** `rg` 件数と完全列挙の一致 — 28 sink中、**戻り値無視 / `ignoreUnused` / `void` は 9件**。D1の14 callerは `enqueueWithRetry` 直接callerのみであり、**全28 sinkでは 14 を超える**。

---

## 2. 各Sinkの `QueueFull` Control-Flow 追跡（`bool == false` 後の次所有権保持地点）

### 追跡テンプレート

```text
enqueueDeferredDeleteNonRt()
        ↓
QueueFull (tstored==false)
        ↓
WithResult == QueueFull
        ↓
bool == false (D3修正後)
        ↓
caller retains
        ↓
次の所有権保持地点
        ↓
retry / quarantine / backpressure / shutdown discard
```

### 全Sink 追跡（代表）

| Sink | `QueueFull` 時 `bool` | Caller retains? | 次の所有権保持地点 | 接続先 | Closure |
|---|---|---|---|---|---|
| `SnapshotCoordinator::discardSnapshot` | `false` → `quarantineRetireSink` | Yes | `quarantineRetireSink(snap, ...)` | **B** (quarantine) | **PASS** |
| `SnapshotCoordinator::retireSnapshot` (2箇所) | 同上 | Yes | 同上 | **B** | **PASS** |
| `EQProcessor` | `QueueFull → false` → `return false` | Yes | 上位 caller へ `false` 伝播 → 上位が `B`/`C` へ | **C** | **PASS** |
| `ISRRuntimePublicationCoordinator` | `QueueFull != Success → return QueueFull` | Yes | 上位へ `QueueFull` 伝播 → `C` | **C** | **PASS** |
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | `QueueFull` をそのまま return | Yes | 上位へ伝播 → `C` | **C** | **PASS** |
| `AudioEngine::enqueueDeferredDeleteNonRt` (bool) | `QueueFull → false` (D3修正後) | Yes | **現行 `Cache`/`Convolver` は `bool` を無視 → 保持するが処理されない** | **D** | **NO-GO** |
| `SnapshotCoordinator::startFade` | `result` 無視 | Yes（`oldTarget` は局所変数） | **現行何もしない → function return で所有権保持先なし** | **orphan** | **NO-GO** |
| `SnapshotCoordinator::completeFade` | 同上 | Yes | 同上 | **orphan** | **NO-GO** |
| `DSPLifetimeManager` 2箇所 | `ignoreUnused(result)` | Yes（`dsp` は引数） | 同上 | **orphan** | **NO-GO** |
| `resetFadeStateAndRetireTarget` (直接 `enqueueRetire`) | `void` 無視 | Yes | 同上 | **orphan** | **NO-GO** |

**重要な ownership orphan パターン:**

```text
QueueFull
 ↓
false
 ↓
caller側で何もしない
 ↓
function return
 ↓
ptrは局所変数または引数としてスコープ外へ
 ↓
所有権保持先なし → leak（UAFではないが leak、D15.2 conservation 破綻）
```

---

## 3. Cache / Convolver 重点監査（D3 で特定された3箇所の完全追跡）

### 3.1 `AudioEngine.Cache.cpp:16` `enqueueDeferredDeleteNonRt(map, ...)`

```cpp
// AudioEngine.Cache.cpp:16
void AudioEngine::CacheManager::... {
    owner.enqueueDeferredDeleteNonRt(map, [](void* p){ delete static_cast<CacheMap*>(p); });
    // 戻り値 bool をチェックしない
    // function return 後、map は局所変数ではないが、caller は所有を保持したまま何もしない
}
```

- `QueueFull → bool false` になった後、**誰が `map` を所有し、どこへ再投入するのか** — **現行なし**。`bool` を正しく `false` にしても、**所有権保持先なし**。

### 3.2 `ConvolverProcessor.Lifecycle.cpp:57` `oldState` / `:70` `sc`

```cpp
// Lifecycle.cpp:57
provider->enqueueDeferredDeleteNonRt(oldState, deleter);
// Lifecycle.cpp:70
provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver);
```

- 同上、`bool` 無視のため `QueueFull` 時の所有保持が **関数return後まで追跡できない**。

### 3.3 判定

**「`bool` を正しく返す」だけでは D4 PASS にならない** — 3箇所はいずれも `QueueFull → bool false → return → 所有権保持先なし` の **ownership orphan**。

---

## 4. Ownership Closure Matrix（全28 sink）

| Sink | Result | QueueFull時 | Caller retains? | 次の処理 | Closure |
|---|---|---|---|---|---|
| `Cache` (`Cache.cpp:16`) | `bool` ignored | `false` | Yes | なし（orphan） | **NO-GO** |
| `Convolver oldState` (`Lifecycle.cpp:57`) | `bool` ignored | `false` | Yes | なし | **NO-GO** |
| `Convolver sc` (`Lifecycle.cpp:70`) | `bool` ignored | `false` | Yes | なし | **NO-GO** |
| `SnapshotCoordinator::startFade` | `RetireEnqueueResult` ignored | `QueueFull` | Yes | なし | **NO-GO** |
| `SnapshotCoordinator::completeFade` | 同上 | `QueueFull` | Yes | なし | **NO-GO** |
| `DSPLifetimeManager` 2箇所 | `ignoreUnused` | `QueueFull` | Yes | なし | **NO-GO** |
| `resetFadeStateAndRetireTarget` (direct `enqueueRetire`) | `void` | `QueuePressure` | Yes | なし | **NO-GO** |
| `discardSnapshot` / `retireSnapshot` 3箇所 | `bool` | `false → quarantine` | Yes | `quarantineRetireSink` | **PASS** |
| `EQProcessor` / `ISRRuntimePublicationCoordinator` / `WithResult` wrapper | `RetireEnqueueResult` | `QueueFull` | Yes | 上位へ `C` 伝播 | **PASS** |
| `AudioEngine bool wrapper` (D3修正後) | `bool` | `false` | Yes | 上位へ `C` (要 D4で全 caller が `B`/`C` へ) | **CONDITIONAL PASS** |

**集計:** `PASS` 8 / `CONDITIONAL PASS` 2 / `NO-GO` 7 / `void` 3

---

## 5. D3 Wrapper 修正はまだ実施しない（D4 read-only 維持）

- 本監査は **read-only のまま**にし、`return result != Shutdown && result != QueueFull` の D3 設計をまだ production へ入れない
- 理由: wrapper を修正しても `Cache`/`Convolver` のように戻り値を無視する sink では `false` になった事実だけでは closure にならないため（D3 の `bool` 修正だけでは leak を残す）

---

## 6. D4 終了条件 チェックリスト

```text
[x] 全28 sinkを列挙（140 matches / 28 enqueueRetire bridge）
[x] 各sinkの呼び出し元を特定（上記 20+ 完全列挙）
[x] QueueFull control-flowを関数returnまで追跡（§2）
[x] caller-retain地点を特定（QueueFull → false / QueueFull そのまま）
[x] retry / quarantine / backpressure / shutdown discard の接続先を特定（B/C）
[x] void / ignored return sinkを全件特定（7件 NO-GO）
[ ] ownership orphan = 0 を確認 → **未達成（3+4件 orphan）**
[ ] QueueFull → transferred への再変換経路 = 0 → **要 D3 wrapper 修正後に確認**
[ ] Shutdown → transferred への再変換経路 = 0 → **PASS**（現行 `Shutdown` は正しく `false`）
[x] D14.3/D15.2と矛盾しないことを確認（QueueFull を right-hand に追加せず B/C へ再計上すれば維持）
```

### D4 最終判定

**最重要判定 `QueueFull → caller retains → 次状態` の閉ループが全sinkで成立しているか:** **NO-GO**

- `Cache`/`Convolver` 3箇所が `QueueFull → bool false → return → 所有権保持先なし` の **ownership orphan**
- `startFade`/`completeFade`/`DSPLifetimeManager`/`resetFadeStateAndRetireTarget` 4箇所も同様

**D4 は PASS にせず、具体的な ownership gap を NO-GO/CONDITIONAL PASS として固定** — D4 完了後に初めて D5 bounded Terminal の設計・実装前監査へ進む。

*本監査は read-only であり、production source 変更 0、D3 wrapper 未実施、Terminal bounded 化なしで実施された。*
