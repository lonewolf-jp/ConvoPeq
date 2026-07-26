# BUG-031: updateAudioThreadSnapshotFade() がスタブ — SnapshotCoordinator の alpha が未使用

## 発見日
2026-07-26

## ファイル
`src/audioengine/AudioEngine.h:3696-3706`
`src/audioengine/AudioEngine.Processing.AudioBlock.cpp:475`
`src/audioengine/AudioEngine.Processing.BlockDouble.cpp`

## 問題
`updateAudioThreadSnapshotFade()` は以下のハードコードされたスタブである:

```cpp
inline bool updateAudioThreadSnapshotFade(int numSamples,
                                          float& snapshotAlpha,
                                          const convo::GlobalSnapshot*& snapshotFrom,
                                          const convo::GlobalSnapshot*& snapshotTo) noexcept
{
    juce::ignoreUnused(numSamples);
    snapshotAlpha = 1.0f;       // ★ 常に完全フェードイン
    snapshotFrom = nullptr;     // ★ source なし
    snapshotTo = nullptr;       // ★ target なし
    return false;               // ★ 「フェード非アクティブ」
}
```

さらにこの関数はコードベースのどこからも呼ばれていない。
また `SnapshotCoordinator::updateFade()`（SnapshotCoordinator.h:101）も
Audio 処理パスから呼ばれていない。

### 影響
`SnapshotCoordinator::startFade(target, fadeSamples)` は Timer スレッドから
正しく呼ばれている（Snapshot.cpp:145）。しかし Audio スレッドは alpha を
読まないため、以下の問題が発生する:

1. **クロスフェードが効かない**: `startFade` で fadeSamples=512 のフェードを
   開始しても、Audio は常に alpha=1.0 で処理する → 新パラメータが即時適用される
2. **CompleteFade が遅延する**: Audio スレッドで `advanceFade()` が
   remainingSamples を減算するため、N サンプル後に `tryCompleteFade()` が成功する。
   その間、Audio は alpha=1.0 で新パラメータを使い続けるが、current スナップショットは
   まだ古いまま。completeFade で current が更新されても alpha=1.0 のため変化なし。
3. **全体として**: SnapshotCoordinator の fade メカニズムが実質的に No-Op になっている。
   意図されたパラメータクロスフェード（EQ/NS/AGC の段階的適用）は機能していない。

### さらに深刻な問題（BlockDouble.cpp）

BlockDouble.cpp には `advanceFade()` の呼び出し自体が存在しない。
ダブル精度パスでは remainingSamples が永遠に減算されず:
- `tryCompleteFade()` が永遠に成功しない
- フェードが FadingIn 状態でスタックする
- `startFade()` の呼び出しが `isFading() == true` により抑制される可能性がある
- 以降のスナップショット更新がブロックされる

### リスク評価
- **重大度**: HIGH — パラメータクロスフェード（EQ/ノイズシェイパー/AGC）が機能していない
- **BlockDouble.cpp の欠落**: HIGH — フェード状態が永久に FadingIn でスタック
- **発生頻度**: 常時（シングル精度）、常時（ダブル精度でスナップショット更新が止まる）

### 修正方針
1. `updateAudioThreadSnapshotFade()` を実装し `SnapshotCoordinator::updateFade()` を呼ぶ
2. その alpha/snapshotFrom/snapshotTo を DSP 処理パスで使用する
3. BlockDouble.cpp に `advanceFade()` を追加する
