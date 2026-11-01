# A2 → A3-a 実装記録（2026-11-01）

- ブランチ: `fix/b2-diag-sink-20261101`（B-2 と同一ブランチ。**変更単位は分けて記録**する）
- 基準: `main` = `6bb7c8d5`
- 設計根拠: `A2-A3A-ASSERT-DESIGN-20261101.md` / `A2-A3A-SOURCE-RECONCILIATION-20261101.md`

## 1. A2 — assert の位置と述語

### 1.1 変更前（`ISRDSPHandle.cpp` の `destroyQuarantineSlot`）

```cpp
const auto prevState = convo::consumeAtomic(registry_[slot].state, acquire);
assert(prevState == DSPState::Quarantined);     // ← CAS 前の観測で assert
if (prevState != DSPState::Quarantined) return false;
...
auto expected = convo::consumeAtomic(registry_[slot].state, acquire);
while (expected == Quarantined) {
    if (compareExchangeAtomic(state, expected, DestroyPending)) break;
}
if (expected != Quarantined) return false;
```

問題: assert は「CAS 前の観測」、遷移の所有権は「CAS」。他者が先に `DestroyPending` へ遷移させた
**正常な競合**でも assert が発火し、Debug で abort していた。

### 1.2 変更後

```cpp
const auto observed = convo::consumeAtomic(registry_[slot].state, acquire);
if (observed != DSPState::Quarantined)
    return false;                                // 競合の正常系（assert しない）

auto expected = observed;
while (!compareExchangeAtomic(state, expected, DestroyPending, acq_rel, acquire))
{
    if (expected != DSPState::Quarantined)
        return false;                            // 他者が遷移済み（正常）
    assert(!"DSPState regressed to Quarantined without a transition owner");
    return false;                                // 真の異常（旧実装では検出不能）
}
assert(expected == DSPState::Quarantined);       // CAS 成功時の不変条件
```

| 項目 | 変更 |
|---|---|
| 述語 `Quarantined` | **残す**（削除していない） |
| 位置 | CAS の**前**（観測値）→ CAS の**結果**（成功時・失敗時） |
| 判別力 | 「他者の遷移」と「状態の逆行」を区別できるようになった |
| 誤検出 | 正常な競合では assert しない |

## 2. A3-a — スロット単位の deferred マーク

### 2.1 変更前

```cpp
if (pinnedCount(slot) != 0) {
    fetchAddAtomic(deferredQuarantineDestroy_, 1, acq_rel);   // グローバル +1
    return false;
}
// retry: グローバルカウンタ != 0 なら「Quarantined な全スロット」を走査
```

問題: retry が commit 経路で処理中のスロットにも触れる（二重試行の発生源）。

### 2.2 変更後

```cpp
// ヘッダ
static constexpr std::uint32_t kDeferredMaskWords = MAX_DSP_SLOTS / 64;   // 256/64 = 4
std::array<std::atomic<std::uint64_t>, kDeferredMaskWords> deferredQuarantineMask_{};

void markQuarantineDeferred(std::uint32_t slot) noexcept;
void clearQuarantineDeferred(std::uint32_t slot) noexcept;
[[nodiscard]] bool isQuarantineDeferred(std::uint32_t slot) const noexcept;
[[nodiscard]] std::uint32_t deferredCount() const noexcept;

// pin 残留
if (pinnedCount(slot) != 0) {
    markQuarantineDeferred(slot);
    return false;
}

// retry: マークされたスロットのみ
if (deferredCount() == 0) return 0;
for (slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
    if (!isQuarantineDeferred(slot)) continue;
    if (state != Quarantined) { clearQuarantineDeferred(slot); continue; }
    if (instance == nullptr)  { clearQuarantineDeferred(slot); continue; }
    if (destroyQuarantineSlot(slot, 0)) { ++destroyed; continue; }  // 成功時に mark 解除
    if (pinnedCount(slot) == 0) clearQuarantineDeferred(slot);
}
```

### 2.3 既存契約の維持（重要）

`deferredQuarantineDestroyCount()` は**診断値ではなく早期 return のゲート**である
（`AudioEngine.Retire.cpp:44` の `quarantineEmpty`、`:72` の抑制条件）。
A3-a 後も「保留中のスロット数」を返すよう、**マスクの popcount** を返す実装にした。
0 を返すと tick が早期 return して deferred スロットが永久に回収されない（リーク）。

| 項目 | 維持 |
|---|---|
| `MAX_DSP_SLOTS = 256` | 維持（マスク 4 word で充足） |
| 戻り値の型 `std::uint32_t` | 維持 |
| 意味（保留中のスロット数） | 維持 |
| CAS による遷移所有権 | 維持（A2 で検査のみ追加） |
| `freeSlots_` への push 経路 | 維持（成功時のみ） |

## 3. 検証計画との対応

| 検証 | 内容 | 結果 |
|---|---|---|
| V1 二重解放 | 同一スロット・同一世代の free が 1 回以下 | **PASS**（Release / Debug とも） |
| V2 誤状態処理 | `false` 時に lane/flag を触らない | **PASS**（deferred 時は状態不変） |
| V3 Debug/Release 整合 | 同一遷移表で最終状態が一致 | Debug / Release の両方で同一結果 |

### 3.1 専用テスト（`A2A3aQuarantineTests.cpp`、opt-in `A2A3A_TEST=1`）

| 検査 | 内容 | 結果 |
|---|---|---|
| create | ハンドルが得られる | PASS |
| pin 取得 | `tryPin` が有効な pin を返す | PASS |
| pin 残留時の破棄 | `destroyQuarantineSlot` が false（deferred） | PASS |
| スロット単位マーク | `isQuarantineDeferred(slot)` が true | PASS |
| ゲート契約 | `deferredQuarantineDestroyCount() == 1` | PASS |
| pin 解放後の再試行 | `retryDeferredQuarantineDestroy() == 1` | PASS |
| マーク解除 | 回収後に `isQuarantineDeferred` が false | PASS |
| カウント復帰 | `deferredQuarantineDestroyCount() == 0` | PASS |
| 二重解放の抑止 | 2 回目の retry が 0 | PASS |
| 回収済みスロット | `destroyQuarantineSlot` が false | PASS |
| 未マークの扱い | 未マークの Quarantined は retry の対象外（カウントも 0 のまま） | PASS |
| 競合（2 スレッド） | 同時破棄で勝者が正確に 1 人 | PASS |
| 残留確認 | 競合後も deferred 残留 0 | PASS |

実行結果: `[A2A3A] RESULT failures=0`（Release / Debug 診断 ON の両方）。

## 4. 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/audioengine/ISRDSPHandle.h` | マスクメンバと 4 つのアクセサ宣言（`<array>` / `<atomic>` / `<cstdint>` 追加） |
| `src/audioengine/ISRDSPHandle.cpp` | A2（assert 位置・述語・真の異常検出）、A3-a（mark/clear/count と retry の対象限定） |
