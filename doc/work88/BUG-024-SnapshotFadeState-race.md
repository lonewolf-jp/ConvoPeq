# BUG-024: SnapshotFadeState advance() vs resetToIdle() 競合によるカウンター不整合

## 発見日
2026-07-26

## ファイル
`src/core/SnapshotFadeState.h:41-67 (advance), 85-91 (resetToIdle)`

## 問題
`advance()`（Audio Thread）と `resetToIdle()`（Timer Thread）の間に以下の競合が存在する。

### 競合シナリオ

`advance()` (Audio Thread):
```
line 43: if (state() != FadingIn) return;   // ← FadingIn
line 46: remaining = remainingCount();        // ← 123 (正の値)
line 50: newRemaining = remaining - numSamples;
line 59: remainingSamples_.store(newRemaining, release);  // ★ 書き込みA
line 60: total = totalCount();                // ★ 読み取りB
line 61: if (total > 0) { alpha update }
```

`resetToIdle()` (Timer Thread, from completeFade/resetFadeStateAndRetireTarget):
```
line 88: state_.store(Idle, release);
line 89: alpha_.store(1.0, release);
line 90: totalSamples_.store(0, release);
line 91: remainingSamples_.store(0, release); // ★ ゼロクリア
```

競合のタイミング:
```
advance():  line 59 → remainingSamples_ に newRemaining を release store (書き込みA)
resetToIdle(): line 88-91 → state=Idle, alpha=1.0, total=0, remaining=0 (ゼロクリア)
advance():  line 60 → totalCount() が 0 を読む (読み取りB) → alpha 更新スキップ
```

**結果:**
- `state_` = Idle
- `remainingSamples_` = newRemaining (≠ 0) — advance の書き込みA が resetToIdle のゼロクリアを上書き
- `totalSamples_` = 0
- Invariant 違反: remainingSamples_ > 0 かつ totalSamples_ == 0

### 影響
- フェードは Idle 状態なので音声処理に直接影響はない（advance は state != FadingIn で早期 return）
- しかし次回 `start()` が呼ばれるまで不整合が持続する
- デバッグアサーション `assert(remainingSamples_ <= totalSamples_)` が追加された場合に誤発火する
- `tryComplete()` が state != FadingIn により早期 return するため、カウンターが永遠に修復されない

### リスク評価
- **重大度**: MEDIUM — 直接的なクラッシュやメモリ破壊は起こさないが、内部不変条件が破壊される
- **発生頻度**: 低（正確なタイミング依存、resetToIdle と advance が同時に実行される必要あり）
- **検出性**: デバッグビルドのアサーションなしでは検出困難

### 修正方針
`advance()` で remainingSamples_ を書き込んだ後に state の再確認を行う:
```cpp
void advance(int numSamples) noexcept
{
    if (state() != FadeState::FadingIn)
        return;
    const int remaining = remainingCount();
    if (remaining <= 0)
        return;
    const int newRemaining = remaining - numSamples;
    if (newRemaining <= 0)
    {
        publishAtomic(remainingSamples_, 0, release);
        return;
    }
    // ★ state を再確認：resetToIdle() により Idle になっていたら書き込みを破棄
    if (state() != FadeState::FadingIn)
        return;
    publishAtomic(remainingSamples_, newRemaining, release);
    ...
}
```
