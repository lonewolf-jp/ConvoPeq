# BUG-033: BlockDouble.cpp のクロスフェードミックスが dryScale を適用しない

## 発見日
2026-07-26

## ファイル
`src/audioengine/AudioEngine.Processing.BlockDouble.cpp:400-427`
`src/audioengine/AudioEngine.Processing.AudioBlock.cpp:432-451`

## 問題
シングル精度パス（AudioBlock.cpp:442）は `useDryAsOld` フラグを確認し、
Dry 信号のスケーリングを適用する:
```cpp
const double dryScale = useDryAsOld
    ? crossfadeRuntime_.getDryScaleGain().getNextValue()
    : 1.0;
```

ダブル精度パス（BlockDouble.cpp:400-427）はこの処理が欠落している:
```cpp
// BlockDouble.cpp のラムダキャプチャ:
[&oldSignal, &fadingSignal, ... /* dryScale なし */](int ch, int s) {
    auto& oldRef = oldSignal.getWritePointer(ch)[s];
    auto& fadeRef = fadingSignal.getWritePointer(ch)[s];
    const double g = crossfadeGain;
    oldRef = oldRef * (1.0 - g) + fadeRef * g;
    fadeRef = 0.0;
};
```

コード内コメントでは「double版では useDryAsOld=false のため dryScale=1.0 固定」と
主張しているが、これは誤りである。

`armCrossfadeIfPending()`（AudioEngine.h:3708）はダブル精度パスからも
呼ばれており（BlockDouble.cpp:348）、以下の条件で `useDryAsOld = true` を設定する:
```cpp
if (firstLoadDryPending)
    useDryAsOld = true;
```

また BlockDouble.cpp:325-327 でも防御的に設定される:
```cpp
if (fading == dsp)
    useDryAsOld = true;  // 自己クロスフェード時のドライ保存
```

### 影響
- `firstIrDryCrossfadePending` が有効な場合、ダブル精度パスでは
  ドライ信号がスケーリングされずにそのまま出力される
- 意図された Dry/Wet クロスフェードが正しく機能しない
- ポップ/クリックが発生する可能性がある

### リスク評価
- **重大度**: MEDIUM — 初期 IR ロード時のクロスフェードでダブル精度が誤動作
- **発生頻度**: 低（ダブル精度 + firstIrDry 条件の重なり）
- **影響範囲**: 可聴なポップ/クリック

### 修正方針
シングル精度パスと同様に `useDryAsOld` と `dryScale` をラムダに取り込む:
```cpp
const double dryScale = useDryAsOld
    ? crossfadeRuntime_.getDryScaleGain().getNextValue()
    : 1.0;
```
