# BUG-027: completeFade() と updateFade() の競合による 1 ブロックのクロスフェード欠落

## 発見日
2026-07-26

## ファイル
`src/core/SnapshotCoordinator.cpp:74-92 (completeFade)`
`src/core/SnapshotCoordinator.h:101-131 (updateFade)`

## 問題
`completeFade()`（Timer Thread）と `updateFade()`（Audio Thread）の間に競合が存在する。

`completeFade()`:
```
line 76: exchangeTarget(nullptr)      // ★ target slot = nullptr
line 86: exchangeCurrent(target)      // ★ current = target (promoted)
line 91: m_fade.resetToIdle()         // ★ state = Idle
```

`updateFade()` は以下の順で atomic 変数を読む:
```
line 106: state = m_fade.state()      // ★ 読み取り #1
line 117: outCurrent = loadCurrent()  // ★ 読み取り #2
line 118: outTarget = loadTarget()    // ★ 読み取り #3
```

### 競合シナリオ

```
Timer (completeFade)                 Audio (updateFade)
│                                    │
├─ line 76: target = nullptr         │
├─ line 86: current = T (promote)    │
│                                    ├─ #1: state = FadingIn（まだ Idle ではない）
│                                    ├─ #2: outCurrent = T（昇格後の current）
│                                    ├─ #3: outTarget = nullptr（クリア済み）
│                                    ├─ outTarget == nullptr 分岐に入る
│                                    ├─ resetFadeStateAndRetireTarget()
│                                    │   → exchangeTarget(nullptr) → 既に null
│                                    │   → m_fade.resetToIdle() → state = Idle
│                                    │
├─ line 91: resetToIdle()            │
│   （state は既に Idle、冪等）        │
```

**結果:** この Audio ブロックでは、`outCurrent=T, outTarget=nullptr` が出力される。
つまり、本来このブロックで行われるべき **target→current の最終クロスフェード段階が 1 ブロック欠落**する。
Audio は唐突に T へ切り替わる。

### 影響
- クロスフェードの最終 1 ブロック（通常 64〜512 サンプル）が失われる
- フェードの完了タイミングが 1 ブロック分早まり、理論上の α=1.0 の代わりに α≈0.98〜0.99 相当で打ち切られる
- 聴感上のクリックは稀だが、クロスフェードの解析的な完全性が損なわれる
- 次回のフェード開始には影響なし

### リスク評価
- **重大度**: LOW — 1 ブロックのクロスフェード終端の欠落。聴感上の影響はごく稀
- **発生頻度**: 中〜高 — timer が 30〜50ms 周期、audio が 1〜10ms 周期のため、競合は頻繁に発生しうる
- **検出性**: 難 — 1 ブロックだけの α 誤差のため通常は検出不可能

### 修正方針
`updateFade()` で target が null の場合、state を再確認してから
`resetFadeStateAndRetireTarget()` を呼ぶ:
```cpp
if (outTarget == nullptr)
{
    if (m_fade.state() != FadeState::FadingIn)
        return false;  // ★ 別スレッドが既に fade を完了させた
    resetFadeStateAndRetireTarget();
    ...
}
```
