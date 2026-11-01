# A2 → A3-a 最新ソース照合資料（2026-11-01）

- 状態: **照合資料のみ。製品コードは変更していない**（実装は Owner HOLD）
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示「最新ソースと照合し、assert・CAS・deferred mark・再試行の関係を具体的に示す」

## 1. 現行コードの実測（行番号つき）

### 1.1 `destroyQuarantineSlot`（`src/audioengine/ISRDSPHandle.cpp:512-585`）

| 段 | 行 | 実コード（要旨） |
|---|---|---|
| 0 | 515-516 | `if (slot >= MAX_DSP_SLOTS) return false;`（`MAX_DSP_SLOTS = 256`、`ISRDSPHandle.h:197`） |
| 1 | 519-524 | generation 照合（`expectedGeneration != 0` のときのみ。呼び出しは全て `0` なので**実質スキップ**） |
| 2 | **527-531** | `prevState = load(state);`<br>`assert(prevState == DSPState::Quarantined);` ← **本件の assert**<br>`if (prevState != Quarantined) return false;` |
| 3 | 533-541 | active / fading / crossfade 照合 → 該当すれば `false` |
| 4 | **551-559** | `closePinsForDestroy(slot, gen);`<br>`if (pinnedCount(slot) != 0) { deferredQuarantineDestroy_ += 1; return false; }` ← **deferred 計上** |
| 5 | **562-572** | `expected = load(state);`<br>`while (expected == Quarantined) { if (CAS(state, expected, DestroyPending)) break; }`<br>`if (expected != Quarantined) return false;` ← **CAS が唯一の遷移所有者** |
| 6 | 574-584 | `instance = nullptr; state = Reclaimed;` → `freeListMutex_` 下でフリーリストへ push |

**観察**: 段 2 の assert は **段 5 の CAS より前**にあり、しかも「他者が先に遷移させた場合」を
`false` で返す契約（段 2 の `if`）と**矛盾**している。これが誤検出の原因。

### 1.2 `retryDeferredQuarantineDestroy`（`src/audioengine/ISRDSPHandle.cpp:590-614`）

```cpp
if (convo::consumeAtomic(deferredQuarantineDestroy_, acquire) == 0)
    return 0;                                   // ← グローバルカウンタがゲート
for (uint32_t slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
    if (state(slot) != Quarantined) continue;
    if (instance(slot) == nullptr) continue;
    if (destroyQuarantineSlot(slot, 0)) ++destroyed;      // ← 全 Quarantined を対象
    else if (pinnedCount(slot) != 0) ++stillDeferred;
}
publishAtomic(deferredQuarantineDestroy_, stillDeferred, release);
```

**観察**: 対象は「Quarantined な全スロット」であり、**commit 経路が今まさに処理中のスロットも含む**。
これが二重試行（assert 誤検出）の発生源。

### 1.3 呼び出し元（製品コード・全 5 箇所）

| 呼び出し元 | 行 | 呼ぶ関数 | 契約 |
|---|---|---|---|
| commit 経路（quarantine 再評価） | `AudioEngine.Commit.cpp:682` | `destroyQuarantineSlot(qslot, 0)` | `false` → `continue`（レーン/フラグを**触らない**） |
| shutdown 経路 | `AudioEngine.Processing.ReleaseResources.cpp:468` | `destroyQuarantineSlot(slot, 0)` | 同上 |
| shutdown の再試行 | `AudioEngine.Processing.ReleaseResources.cpp:450` | `retryDeferredQuarantineDestroy()` | 戻り値をログのみ |
| reclaim tick | `AudioEngine.Retire.cpp:55`（`tryReclaimResources`） | `retryDeferredQuarantineDestroy()` | 戻り値は破棄 |
| drain | `AudioEngine.Retire.cpp:89`（`drainDeferredRetireQueues`） | `retryDeferredQuarantineDestroy()` | 戻り値は破棄 |

### 1.4 カウンタの**ゲート**としての使われ方（互換の要点）

`deferredQuarantineDestroyCount()` は診断値ではなく**早期 return の条件**である。

- `AudioEngine.Retire.cpp:44`: `quarantineEmpty` → `tryReclaimResources` が**何もしない**
- `AudioEngine.Retire.cpp:72`: `drainDeferredRetireQueues` の**抑制条件**

**したがって A3-a でマーク方式に変えても、この関数は「保留中のスロット数」を返し続ける必要がある。**
（0 を返すと tick が早期 return し、deferred スロットが永久に回収されない＝リーク）

## 2. A2 の具体的な変更点（照合結果）

### 2.1 変更前（現行 `:527-531` と `:562-572`）

```cpp
// 段 2（現行）
const auto prevState = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
assert(prevState == DSPState::Quarantined);
if (prevState != DSPState::Quarantined)
    return false;

// 段 5（現行）
auto expected = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
while (expected == DSPState::Quarantined) {
    if (convo::compareExchangeAtomic(registry_[slot].state, expected, DSPState::DestroyPending,
                                     std::memory_order_acq_rel, std::memory_order_acquire))
        break;
}
if (expected != DSPState::Quarantined)
    return false;
```

### 2.2 変更後（A2）

```cpp
// 段 2（A2）: 観測が Quarantined でなければ「競合の正常系」として false。
//             ここでは assert しない（他者が先に回収した正当なケースのため）。
const auto observed = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
if (observed != DSPState::Quarantined)
    return false;

// 段 3・4 は現行のまま（active/fading/crossfade、pin gate）

// 段 5（A2）: CAS が唯一の遷移所有者。結果に基づいて検査する。
auto expected = observed;
while (!convo::compareExchangeAtomic(registry_[slot].state, expected, DSPState::DestroyPending,
                                     std::memory_order_acq_rel, std::memory_order_acquire))
{
    if (expected != DSPState::Quarantined)
        return false;   // 他者が先に遷移させた（正常）。自分は何も遷移させていない
    // expected == Quarantined のまま CAS 失敗 = 状態が Quarantined へ逆行した（真の異常）
    assert(!"DSPState regressed to Quarantined without a transition owner");
    return false;
}
assert(expected == DSPState::Quarantined);   // ★ 述語を保持（CAS 成功時の不変条件）
```

### 2.3 assert の述語・位置・CAS との関係（Owner 要求）

| 項目 | 現行 | A2 | 判定 |
|---|---|---|---|
| 述語 `prevState == Quarantined` | 残る | **残る**（CAS 成功時の不変条件として） | 削除していない |
| 位置 | CAS の**前**（観測値） | CAS の**結果**（成功時／失敗時） | 遷移の所有者側へ移動 |
| CAS との関係 | assert と CAS が**別の観測**を見ている | assert は CAS の結果と**同じ観測**を見る | 一貫 |
| 真の異常の検出 | できない（正常な競合と区別不能） | **できる**（`Quarantined` への逆行） | 判別力が上がる |
| 誤検出 | する（Debug の abort 原因） | しない | 問題を隠さず解消 |

## 3. A3-a の具体的な変更点（照合結果）

### 3.1 変更前（現行 `:551-559` と `:590-614`）

```cpp
// 段 4（現行）
if (pinnedCount(slot) != 0) {
    convo::fetchAddAtomic(deferredQuarantineDestroy_, std::uint32_t{1}, std::memory_order_acq_rel);
    return false;
}
// retry（現行）: グローバルカウンタ != 0 なら全 Quarantined を走査
```

### 3.2 変更後（A3-a）

```cpp
// 新設: スロット単位のマーク（MAX_DSP_SLOTS = 256 → uint64 × 4）
std::atomic<std::uint64_t> deferredMask_[4];
inline void markDeferred(uint32_t s) noexcept {
    fetchOrAtomic(deferredMask_[s >> 6], uint64_t{1} << (s & 63), std::memory_order_acq_rel);
}
inline void clearDeferred(uint32_t s) noexcept {
    fetchAndAtomic(deferredMask_[s >> 6], ~(uint64_t{1} << (s & 63)), std::memory_order_acq_rel);
}
inline uint32_t deferredCount() const noexcept {   // ← 既存 API の戻り値
    uint32_t n = 0; for (int i = 0; i < 4; ++i) n += popcount(load(deferredMask_[i]));
    return n;
}

// 段 4（A3-a）
if (pinnedCount(slot) != 0) {
    markDeferred(slot);          // ★ グローバル +1 ではなく、このスロットをマーク
    return false;
}

// retry（A3-a）: マークされたスロットだけを対象にする
if (deferredCount() == 0) return 0;
for (uint32_t slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
    if (!isMarked(slot)) continue;
    if (state(slot) != Quarantined) { clearDeferred(slot); continue; }  // 他者が終端済み
    if (destroyQuarantineSlot(slot, 0)) { clearDeferred(slot); ++destroyed; }
    else if (pinnedCount(slot) == 0) clearDeferred(slot);               // 不要になった
    else ++stillDeferred;
}
// 既存 API の契約維持: 「保留中のスロット数」
//   → deferredQuarantineDestroyCount() は deferredCount() を返す
```

### 3.3 関係の整理（Owner 要求「assert・CAS・deferred mark・再試行の関係」）

| 要素 | 責務 | 変更 |
|---|---|---|
| **assert** | 遷移の不変条件（自分の遷移が Quarantined から始まったこと）と真の異常検出 | A2 で位置と述語を CAS 側へ |
| **CAS** | `Quarantined → DestroyPending` の**唯一の遷移所有者**。二重解放を構造的に防ぐ | 変更なし（これが正しさの源） |
| **deferred mark** | 「pin が残って破棄できなかった」スロットの記録。**グローバル → スロット単位** | A3-a |
| **再試行（tick）** | マークされたスロットだけを再試行。commit 経路との重複を構造的に減らす | A3-a |
| **ゲート `deferredQuarantineDestroyCount()`** | tick の早期 return 判定（`Retire.cpp:44`・`:72`） | **契約維持**（保留中のスロット数を返す） |

### 3.4 適用順序と、それぞれが閉じる範囲

```
A2 適用後:  assert の誤検出が消える（競合は false で正常終了）。
            ただし commit 経路と tick の「不要な競合」は残る（頻度が下がるだけ）。
A3-a 適用後: tick が commit 経路の処理中スロットに触れなくなる（重複の発生源を除去）。
            assert の誤検出は構造的にも起きにくくなる。
```

**両方適用して初めて「誤検出しない」かつ「不要な競合を生まない」状態になる。**
逆順（A3-a → A2）では、A3-a だけでは commit 経路と tick の競合が残り、assert の誤検出が残る。

## 4. 二重解放と誤状態処理の分離検証（Owner 要求）

| 検証 | 何を証明するか | 方法 | 対象コード |
|---|---|---|---|
| **V1** 二重解放が起きない | `freeSlots_` への同一スロット・同一世代の push が 1 回以下 | push をスロット×世代でカウントする計装（テストビルドのみ）＋ 2 スレッド同時破棄の競合テスト | `:574-584` |
| **V2** 呼び出し側が誤った状態を扱わない | `false` を受けた呼び出し元が lane/flag を触らない | 競合を注入し、`false` 時に `lifetime().reclaim` / `quarantineManager_.reclaimSlot` が呼ばれないことを確認 | `Commit.cpp:682-685`、`ReleaseResources.cpp:468-470` |
| **V3** Debug/Release の整合 | 同一の遷移表で最終状態が一致 | 同じ競合パターンを両構成で実行し、`state`／フリーリスト／世代を比較 | 全体 |

**V1 と V2 は独立**であり、一方の成功をもう一方の根拠にしない。

## 5. 実装前に残る確認事項

1. `MAX_DSP_SLOTS = 256` に対し `deferredMask_[4]` で足りる（`256 / 64 = 4`）。**実測済み**。
2. `deferredQuarantineDestroyCount()` の戻り値型は `std::uint32_t`（`ISRDSPHandle.h:299`）。
   `deferredCount()` の `popcount` 合計で一致する。**互換維持可**。
3. マーク漏れ（Quarantined かつ instance 非 null だが未マーク）の検出方法は実装時に確定。
4. Debug 実行での assert 発火頻度は未計測（A2 適用後の V3 で 0 回を確認）。

## 6. この資料で言えること

- A2 と A3-a の変更点は、**現行コードの行番号と契約に一致**させて具体化した。
- assert は削除せず、述語を CAS 成功時の不変条件として保持し、真の異常検出を追加する。
- `deferredQuarantineDestroyCount()` の**ゲート契約**を維持する（これを壊すとリークする）。
- 実装は Owner の HOLD を維持し、**コードは変更していない**。
