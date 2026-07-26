# BUG-038: SpectrumAnalyzer FFT マグニチュードスケーリング誤差 — 全周波数で +6 dB

## 発見日
2026-07-26

## ファイル
`src/SpectrumAnalyzerComponent.h:74`
`src/SpectrumAnalyzerComponent.cpp:456-501`

## 問題
FFT マグニチュードのスケーリングファクターが誤っている:

```cpp
static constexpr float FFT_MAGNITUDE_SCALE = 4.0f / NUM_FFT_POINTS;  // = 4/4096
```

複素 FFT（`DFTI_COMPLEX`）に実信号を入力した場合:
- フルスケール正弦波 (A=1.0) → |X[k]| = A * N / 2 = 2048
- 適用されるスケール: 2048 * (4/4096) = 2.0
- `gainToDecibels(2.0)` = +6.02 dBFS

正しいスケールは振幅保存の場合 `2.0 / NUM_FFT_POINTS`:
- 2048 * (2/4096) = 1.0 → 0 dBFS

### 影響
- スペクトラムアナライザの表示が全周波数で **+6 dB 過大**
- 0 dBFS 正弦波が +6 dBFS と表示される
- マスタリング/ラウドネス計測で誤った値

### リスク評価
- **重大度**: CRITICAL — 計測値の常時誤差
- **発生頻度**: 常時
- **検出性**: 他の DAW のスペアナとの比較で容易に発見可能

### 修正方針
`FFT_MAGNITUDE_SCALE` を `2.0f / NUM_FFT_POINTS` に変更する。
