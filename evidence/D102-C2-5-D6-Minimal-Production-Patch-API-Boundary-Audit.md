# D102-C2-5-D6 — Minimal Production Patch / API Boundary Audit（read-only, production変更0）

- **実施日**: 2026-08-26 02:00 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **Contract**: `I4` D14.2/D14.3/D15.2 + D3 ownership table (`Success/QueuePressure/TerminalReclaim == transferred` / `QueueFull/Shutdown == caller retains`) + D5 Pattern B
- **目的**: D5 で Pattern B（Local explicit quarantine）が最適とされた 9 orphan について、**既存 API だけで最小修正が確定するか**を監査する。`quarantineRetireSink` の実在・可視性、Cache の `enqueueFallbackMaps` の ownership 保持、Convolver の `provider` quarantine API、RT 境界、新 retire path 有無、テスト境界を確定し、D7 Patch Specification 前の最終 API 境界を固定する
- **判定**: **CONDITIONAL PASS** — `quarantineRetireSink` は `SnapshotCoordinator` 内部専用（Cache/Convolver から直接利用不可）のため D5 前提は未確定。ただし **Cache は既存 `enqueueFallbackMaps` で Pattern B が成立**、Convolver/DSPLifetimeManager/RT `resetFade` は別 authority への委譲または `WithResult` への変更が必要。9 orphan の最小修正は **共通 wrapper 1件 + 個別 B 5件** で確定可能

---

## D6-1. `quarantineRetireSink` の実在性確認（D5 前提の検証）

### 定義（`src/core/SnapshotCoordinator.cpp:21`）

```cpp
void SnapshotCoordinator::quarantineRetireSink(void* ptr, void (*deleter)(void*),
                                               uint64_t epoch, const char* reason) noexcept
{
    if (m_retireSink) m_retireSink->quarantineRetire(ptr, deleter, epoch, DeletionEntryType::Generic,
                                                     reason, 0, 0);
    else { /* m_retireSink == nullptr → leak のみ・UAF なし */ }
}
```

| 項目 | 確認内容 | 結果 |
|---|---|---|
| 定義 | `SnapshotCoordinator.cpp:21` | **存在** |
| 宣言 | `src/core/SnapshotCoordinator.h:145` `void quarantineRetireSink(...) noexcept;` | **存在** |
| public/private | `SnapshotCoordinator` の `private`（`m_retireSink` と共に private セクション） | **private** |
| 呼び出し可能な所有者 | `SnapshotCoordinator` 内部のみ（`switchImmediate:97` `discardSnapshot:108` `retireCurrentAndTarget:177,182` `startFade:60` `completeFade:117`） | **SnapshotCoordinator 内部専用** |
| 引数型 | `void* ptr, void(*deleter)(void*), uint64_t epoch, const char* reason` | **確認** |
| RT/NonRT | `quarantineRetire` は `RetireQuarantineStore::quarantine`（mutex + allocation-free array）内部で `NonRT` のみ（`ISRRetireRouter::quarantineRetire` も `NonRT`） | **NonRT** |
| Cacheから直接利用可能か | `AudioEngine::EQCacheManager` は `SnapshotCoordinator` の member ではないため **直接利用不可** | **No** |
| Convolverから直接利用可能か | `ConvolverProcessor` は `SnapshotCoordinator` と無関係、**直接利用不可** | **No** |
| SnapshotCoordinator内部専用APIか | **Yes** | **未確定前提は誤り** |

**結論:** D5 の「`Cache / Convolver → provider->quarantineRetireSink(...)`」前提は **実コードで未確定**。`quarantineRetireSink` は `SnapshotCoordinator` 内部専用 API であり、Cache/Convolver からそのまま利用できない。

---

## D6-2. Cache の既存 Fallback 経路を完全確認

### `AudioEngine.Cache.cpp` 全経路

```text
tryEnqueueDeferredMap(CacheMap* map) // h:Cache.cpp:16
  ↓ owner.enqueueDeferredDeleteNonRt(map, deleter) // bool wrapper
  ↓ return true 固定（現行 QueueFull でも true）

storeNewMap(CacheMap* newMap) // Cache.cpp:41
  ↓ old = exchangeAtomic(cacheMapPtr, newMap)
  ↓ owner.enqueueDeferredDeleteNonRt(old, deleter) // bool 無視

drainDeferredMapsUnderLock() // Cache.cpp: drain
  ↓ for each fallbackMaps: if (!tryEnqueueDeferredMap(*it)) *out++ = *it;
  ↓ erase(out, end) // 成功したものだけ除去、失敗は保持

enqueueFallbackMaps // AudioEngine.h:2142 std::vector<CacheMap*> enqueueFallbackMaps;
```

| 項目 | 確認内容 |
|---|---|
| `CacheMap*` → `QueueFull` → `enqueueFallbackMaps` | **可能** — `tryEnqueueDeferredMap` が `false` を返せば `drainDeferredMapsUnderLock` の `if (!tryEnqueue...)` で `enqueueFallbackMaps` に残る |
| 誰が保持する？ | `EQCacheManager::enqueueFallbackMaps`（`std::vector<CacheMap*>`）— **ownership-bearing container** |
| いつ再試行？ | `drainDeferredMapsUnderLock()` で `writeMutex` 取得中に再 `tryEnqueueDeferredMap` |
| 成功時にどこで ownership transfer？ | `tryEnqueueDeferredMap` が `true`（`Success/QueuePressure/TerminalReclaim`）で `enqueueFallbackMaps` から除去 |
| shutdown時は？ | `shutdownReclaim` 経由で `TerminalReclaimAuthority` へ（`AudioEngine.h:4198` shutdown early return） |
| 単に「既存vectorがある」だけか？ | **No** — `enqueueFallbackMaps` は **既に QueueFull 時の退避と再試行の閉ループ**として設計されている。`storeNewMap` → `enqueueDeferredDeleteNonRt` → `drainDeferredMapsUnderLock` の経路で **B (Local explicit quarantine) として成立** |

**D6-2 判定: Cache fallback ownership は `enqueueFallbackMaps` で成立 — Pattern B で閉じる**

---

## D6-3. Convolver については「B」が本当に可能か再検証

### `ConvolverProcessor.Lifecycle.cpp:57,70`

```cpp
provider->enqueueDeferredDeleteNonRt(oldState, deleter);
provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver);
```

| 項目 | 確認内容 |
|---|---|
| `ConvolverProcessor` の `provider` 実型 | `IRetireProvider` / `IEpochProvider`（`ConvolverProcessor.h` で `IRetireProvider* provider`） |
| `quarantine` API | `provider` に `quarantineRetireSink` は **存在しない**（`IRetireProvider` は `enqueueRetire` / `tryReclaim` のみ） |
| そのAPIの ownership contract | `enqueueRetire` は `Success/QueuePressure` のみ（`QueueFull` は D2-0 で新設予定） |
| `QueueFull` 時の再帰/再quarantine | `provider` に quarantine API がないため **再quarantine 不可** |
| shutdown時の処理 | `shutdownReclaim` 経由 |

**結論:** D5 の「`Convolver → provider->quarantineRetireSink`」は **不可**。`provider` に quarantine API が存在しない。

**代替（新API追加を即決しない）:**

| Pattern | 概要 | Convolver への適合 | 最小変更 |
|---|---|---|---|
| **A — Result propagation** | `sink → WithResult → QueueFull → 上位へ result 返却 → 上位 backpressure/quarantine` | Convolver の `Lifecycle` 関数が `RetireEnqueueResult` を上位へ返せれば可能 | `enqueueDeferredDeleteNonRtWithResult` への変更 + 上位での `B` 処理 |
| **既存 `AudioEngine` quarantine authority への委譲** | `Convolver` が `AudioEngine` の `RetireQuarantineStore` へ委譲 | `ConvolverProcessor` が `AudioEngine` の `m_retireRouter` へアクセス可能か要確認 | 要 `AudioEngine` 経由の委託 API 確認 |
| **C — Ownership-bearing member** | `oldState` を `ConvolverProcessor` の member に保持し、後続 Coordinator で retry | 新たな member と retry authority が必要 | 複雑、Pattern A/B より大 |

**D6-3 判定: Convolver の Pattern B（直接 `quarantineRetireSink`）は不可。Pattern A（`WithResult` への変更 + 上位での B）が既存 architecture に最小変更で適合**

---

## D6-4. 9 Orphan を「実際に変更可能な最小単位」に分解

| Sink | 現行API | QueueFull保持者 | 最小修正 | 新API必要? | RT影響 | Closure |
|---|---|---|---|---|---|---|
| `Cache:16` `tryEnqueueDeferredMap(map)` | `bool` wrapper (`return true` 固定) | `map`（param） | `tryEnqueueDeferredMap` を `bool` 正しく返す + `drainDeferredMapsUnderLock` で B | No（既存 `enqueueFallbackMaps` 再利用） | **B** |
| `Cache:41` `storeNewMap(old)` | `bool` wrapper 無視 | `old`（local） | 同上、`old` を `enqueueFallbackMaps` へ退避 | No | **B** |
| `Convolver:57` `oldState` | `bool` wrapper 無視 | `oldState`（local） | `enqueueDeferredDeleteNonRtWithResult` へ変更 + 上位で `B` | No（`WithResult` 既存） | **C**（上位伝播後に B） |
| `Convolver:70` `sc` | 同上 | `sc`（local） | 同上 | No | **C** |
| `startFade` `oldTarget` | `RetireEnqueueResult` 無視 | `oldTarget`（local） | `if (!result) quarantineRetireSink` 追加 | No（既存 `quarantineRetireSink` 再利用） | **B** |
| `completeFade` `old` | 同上 | `old`（local） | 同上 | No | **B** |
| `DSP:49` `retire(dsp)` | `RetireEnqueueResult` `ignoreUnused` | `dsp`（param） | `if (result == QueueFull) quarantine` 追加 | No（`DSPLifetimeManager` は `router_->quarantineRetire` を呼べる） | **B** |
| `DSP:96` `retire(dsp, epoch)` | 同上 | `dsp` | 同上 | No | **B** |
| `resetFadeStateAndRetireTarget` `target` (direct `enqueueRetire`) | `void` 無視 | `target`（local） | `enqueueWithRetry` + `quarantine` へ変更（ただし RT 境界要確認） | No（既存 `enqueueWithRetry` 再利用） | **B**（D6-6 で RT 要確認） |

**「1 sink = 1修正」とは限らない:** `Cache` 2件は共通 `tryEnqueueDeferredMap` の `bool` 修正 + `enqueueFallbackMaps` で同時 closure。`Snapshot` 2件は同型 `quarantine` 追加で共通化可能。

---

## D6-5. D3 Wrapper 修正の正確な位置を固定

### 固定済み設計

```cpp
return result != RetireEnqueueResult::Shutdown
    && result != RetireEnqueueResult::QueueFull;
```

### 分離

| 分類 | 該当 sink | 説明 |
|---|---|---|
| **D3-only** | `EQProcessor` / `ISRRuntimePublicationCoordinator` / `WithResult` wrapper 自体 | `QueueFull` を `RetireEnqueueResult` として正しく返すだけで閉じる（既に `C`） |
| **D3 + local quarantine** | `Cache` 2 / `Snapshot` 2 / `DSP` 2 / `resetFade` | D3 の `bool` 修正後に **追加で local `quarantine` への退避**が必要 |
| **D3 + result propagation** | `Convolver` 2 | D3 後に `WithResult` への変更 + 上位での `B` が必要 |

---

## D6-6. RT境界を必ず確認（`resetFadeStateAndRetireTarget`）

### `resetFadeStateAndRetireTarget` の caller context

```text
resetFadeStateAndRetireTarget() // h:138
  ↓ コメント: "RT(updateFade) から呼ばれ得るため除外" (h:150)
  ↓ 現行直接 enqueueRetire (NonRT の tryReclaim を含まない lock-free path)
  ↓ 呼び出し元: SnapshotCoordinator::finalizeShutdown / SnapshotCoordinator::updateFade 等
```

| 項目 | 確認内容 |
|---|---|
| Caller context | `updateFade` は **RT Audio Thread** から呼ばれ得る |
| RT / NonRT | **RT** caller 存在 |
| QueueFull時に quarantine可能か | **No** — `quarantineRetireSink` は `RetireQuarantineStore::quarantine`（mutex）内部で **NonRT のみ**。RT から `quarantineRetireSink` を直接呼ぶと RT 境界違反 |
| 原則 | **RTから `quarantineRetireSink` を直接呼ぶ案は採用しない** |

**代替:** `resetFadeStateAndRetireTarget` は RT から呼ばれ得るため、**`enqueueWithRetry` への変更は不可**。現行 `enqueueRetire`（lock-free D のみ）のまま、**QueueFull 時は `tryReclaim` を含む NonRT 側の `drain` に期待するベストエフォート**とし、RT 側で ownership を `target` として保持する member への退避（Pattern C）を検討。

---

## D6-7. 新しい Retire Path を増やさないことを確認

| 修正 | 既存 authority 再利用か | 新規 path か |
|---|---|---|
| `Cache` 2件 `enqueueFallbackMaps` | **既存** `EQCacheManager::enqueueFallbackMaps` | — |
| `Convolver` 2件 `WithResult` 化 | **既存** `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | — |
| `Snapshot` 2件 `quarantineRetireSink` | **既存** `SnapshotCoordinator::quarantineRetireSink` | — |
| `DSP` 2件 `quarantine` | **既存** `DSPLifetimeManager::router_->quarantineRetire` | — |
| `resetFade` | **既存** `enqueueRetire` のまま（RT 境界維持） | 新規追加なし |

**全修正は既存 retire authority の再利用** — 新しい retire authority / queue / quarantine path の追加はなし。Practical Stable ISR Bridge Runtime の「retire経路を増やさない」原則に合致。

---

## D6-8. テスト境界を先に固定

| Test | 条件 | 期待 |
|---|---|---|
| `T-D3-1` QueueFull → bool false | `enqueueWithRetry: QueueFull` → `WithResult: QueueFull` → `bool: false` | `false` |
| `T-D6-1` Cache QueueFull → ownership retained | `Cache.cpp:16` `map` / `:41` `old` | `bool == false` かつ `enqueueFallbackMaps` に残留 |
| `T-D6-2` Convolver QueueFull → ownership retained | `Lifecycle.cpp:57` `oldState` / `:70` `sc` | `WithResult == QueueFull` かつ `sc` が上位で保持 |
| `T-D6-3` Snapshot startFade QueueFull → retained | `cpp:57` `oldTarget` | `quarantineRetireSink` へ移送 |
| `T-D6-4` completeFade QueueFull → retained | `cpp:114` `old` | 同上 |
| `T-D6-5` DSP QueueFull → retained | `cpp:49,96` `dsp` | `quarantine` へ |
| `T-D6-6` resetFade QueuePressure/QueueFull → retained | `h:60` `target` (RT) | `enqueueRetire` の戻りを保持（RT 境界維持） |
| `T-D6-7` Shutdown → caller retains | `shutdownReclaim` | `Shutdown` |
| `T-D6-8` Success/QueuePressure/TerminalReclaim → transferred | 通常時 | `true` |

**追加境界:**

```text
no double quarantine — Q/E/T への二重移送なし（DSPLifetimeManager コメントの double-quarantine 回避）
no double delete — deleter は epoch-safe かつ NonRT のみ 1回実行
no ownership disappearance — QueueFull/Shutdown 時に ptr がいずれの container にも属さない状態を作らない
```

---

## D6 終了条件 チェックリスト

```text
[x] quarantineRetireSink の実在・可視性を確認 → SnapshotCoordinator private、Cache/Convolver から直接利用不可
[x] Cache fallback ownershipを確認 → enqueueFallbackMaps で Pattern B 成立
[x] Convolver provider APIを確認 → IRetireProvider に quarantine API なし、Pattern A（WithResult化）が最小
[x] Snapshot APIの再利用可否を確認 → startFade/completeFade は quarantineRetireSink 再利用可、resetFade は RT のため不可
[x] DSP APIの再利用可否を確認 → router_->quarantineRetire で Pattern B 可
[x] resetFadeのRT/NonRT callerを確認 → RT caller 存在、quarantine 直接呼び出し不可
[x] 9 orphan全ての最小修正を確定（上記 D6-4 表）
[x] D3-only / D3+B / D3+A を分類（D6-5）
[x] 新規retire authorityを増やさない（D6-7 既存再利用のみ）
[x] RTでquarantine/ownership operationを追加しない（D6-6 resetFade は RT のまま）
[x] 各修正に対応するテストを確定（D6-8 T-D3-1〜T-D6-8）
[x] I4 D14.3/D15.2と矛盾なし（QueueFull を right-hand に追加せず B/C へ再計上）
[x] production source変更 = 0
```

### D6 判定

**CONDITIONAL PASS**

- **PASS:** 実在する既存 API だけで 9 orphan のうち **Cache 2 / Snapshot 2 / DSP 2 の 6件** は最小修正（D3 `bool` 1件 + 各 local `quarantine` 5件）で確定
- **CONDITIONAL:** `Convolver` 2件 と `resetFade` 1件 は **API 追加なしでは完全な Pattern B が不可** — Convolver は `WithResult` 化 + 上位での B が必要だが、上位の `AudioEngine` quarantine authority への委譲可否が未確定。`resetFade` は RT 境界のため `quarantine` ではなく Pattern C（ownership-bearing member）または現行 `enqueueRetire` のままベストエフォートとする必要がある。両件について D7 で patch 境界を最終確定

*本監査は read-only であり、production source 変更 0、Terminal bounded 化なし、D14/D15 改訂なしで実施された。*
