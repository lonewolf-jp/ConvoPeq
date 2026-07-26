# BUG-030: Timer の exchangeFadingRuntimeDSP(nullptr) と DSPTransition の書き込み競合

## 発見日
2026-07-26

## ファイル
`src/audioengine/AudioEngine.Timer.cpp:1000-1008`
`src/audioengine/DSPTransition.h:91-92`

## 問題
Timer コールバック（line 1000-1008）は以下を実行する:
```cpp
if (!m_coordinator.isFading())
{
    auto* const doneRaw2 = exchangeFadingRuntimeDSP(nullptr);
    if (auto* done = ...) {
        DSPLifetimeManager lifetimeMgr(*this);
        lifetimeMgr.retire(done);
    }
}
```

DSPTransition（line 91-92）は以下を実行する:
```cpp
auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP);
```

これらは同じ `fadingRuntimeDSPSlot` を操作するが、タイミング調整は行われていない。

### 競合シナリオ（Use-After-Free）

```
Timer (JUCE Message Thread)          DSPTransition.onPublishCompleted()
│                                    │
├─ m_coordinator.isFading() → false  │
│                                    ├─ exchangeFadingRuntimeDSP(oldDSP)
│                                    │   → slot = oldDSP
│                                    ├─ crossfadeRuntime_.start()
│                                    ├─ SnapshotCoordinator に fade を開始させる
│                                    │   （まだ isFading() = false の可能性）
│                                    │
├─ exchangeFadingRuntimeDSP(nullptr)  │
│   → oldDSP を取得！                 │
├─ lifetimeMgr.retire(oldDSP)        │
│   → ★ フェード中の DSP を retire！  │
│                                    │
│  [Audio Thread]                     │
│  RuntimeWorld が oldDSP を参照     │
│  oldDSP の process() を呼ぶ        │
│  → ★ Use-After-Free ★             │
```

### 根本原因

`isFading()` のチェックは `SnapshotCoordinator` のフェード状態を見ており、
`DSPTransition` がクロスフェードを開始してから `SnapshotCoordinator` が
`startFade()` を呼び出すまでの間にウィンドウが存在する。

また `isFading()`（line 657, 1000）の読み取りと `exchangeFadingRuntimeDSP(nullptr)`
（line 1002）の間はアトミックではない — この間に DSPTransition が
`fadingRuntimeDSPSlot` に値を書き込める。

### リスク評価
- **重大度**: HIGH — Use-After-Free／オーディオクラッシュ
- **発生頻度**: 低〜中（Timer 周期と遷移完了のタイミングに依存）
- **検出性**: 難（通常の使用では再現困難、稀にしか発現しない）

### 修正方針
以下のいずれかの対策が必要:
1. Timer の fading slot クリアを `isFading()` + `isPending()` の両方でガードする
2. `exchangeFadingRuntimeDSP(nullptr)` を CAS で行い、DSPTransition の書き込みと競合しないことを確認する
3. `isFading()` または `crossfadeRuntime_.isPending()` が true の場合は
   `exchangeFadingRuntimeDSP(nullptr)` をスキップする
