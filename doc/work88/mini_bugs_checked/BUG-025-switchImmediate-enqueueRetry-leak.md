# BUG-025: switchImmediate 経由の resetFadeStateAndRetireTarget で enqueueRetry 未使用によるリーク

## 発見日
2026-07-26

## ファイル
`src/core/SnapshotCoordinator.cpp:68 (resetFadeStateAndRetireTarget)`
`src/core/SnapshotCoordinator.h:83 (switchImmediate → resetFadeStateAndRetireTarget 呼び出し)`

## 問題
`switchImmediate()` は明らかに Non-RT（Timer/Message Thread）から呼ばれる関数であるにもかかわらず、
内部で呼ぶ `resetFadeStateAndRetireTarget()` が `enqueueWithRetry` ではなく素の `enqueueRetire` を使用している。

### コード

`SnapshotCoordinator.cpp:57-72`:
```cpp
void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
{
    ...
    GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
    if (target)
    {
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);  // ← 再試行なし
    }
    m_fade.resetToIdle();
}
```

`SnapshotCoordinator.h:77-91`:
```cpp
void switchImmediate(GlobalSnapshot* newSnap) noexcept {
    ...
    resetFadeStateAndRetireTarget();  // ← Non-RT から呼ぶが enqueueRetry 不使用

    GlobalSnapshot* oldSnap = m_slots.exchangeCurrent(newSnap, std::memory_order_release);
    if (oldSnap) {
        uint64_t newEpoch = m_epochProvider->publishEpoch();
        enqueueWithRetry(*m_epochProvider, oldSnap, snapshotDeleter, newEpoch);  // ← こちらは retry あり
    }
}
```

### 問題

`enqueueWithRetry` は以下の処理を行う:
1. `enqueueRetire()` を試行
2. 失敗したら `tryReclaim()` を呼んでリングバッファを空ける
3. 再度 `enqueueRetire()` を試行

`resetFadeStateAndRetireTarget()` は 1 のみを行い、2 と 3 をスキップする。
これは `updateFade()`（Audio RT Thread）から呼ばれる場合には正しい（`tryReclaim` はブロッキングのため RT で呼べない）。

しかし `switchImmediate()` から呼ばれる場合には Non-RT スレッド上であり、
`tryReclaim` の呼び出しが可能かつ望ましい。

**リークシナリオ:**
1. リングバッファが満杯（kMaxRetired エントリ全て使用中、reclaim 待ち多数）
2. `switchImmediate()` が呼ばれる
3. `resetFadeStateAndRetireTarget()` → `enqueueRetire()` が **false を返す**（バッファ満杯）
4. target ポインタがリーク — `SnapshotFactory::destroy()` が永遠に呼ばれない
5. 以降、この GlobalSnapshot が指すメモリ（数十KB〜数MB）が解放されない

### 影響範囲

`switchImmediate()` の呼び出し元:
- `startFade()` — target==null または fadeSamples<=0 の場合（SnapshotCoordinator.cpp:16, 24）
- `createSnapshotFromCurrentState()` — switchImmediate 直接呼び出し（Snapshot.Snapshot.cpp:149, 158）
- `SnapshotCoordinator::~SnapshotCoordinator()` — 間接的に（最終手段の switchImmediate）

### リスク評価
- **重大度**: MEDIUM-HIGH — リングバッファ満杯時にリークが発生する
- **発生頻度**: 低（連続した高速スナップショット切り替え＋reclaim 遅延時にのみ発生）
- **リーク量**: GlobalSnapshot 1つあたり数十KB〜数MB

### 修正方針
`resetFadeStateAndRetireTarget()` をリファクタリングし、RT セーフなパスと Non-RT なパスを分離する。
または `switchImmediate()` 内で resetFadeStateAndRetireTarget の代わりに enqueueWithRetry を使った直接 exchangeTarget + retire を行う。
