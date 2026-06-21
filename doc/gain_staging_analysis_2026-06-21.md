# ゲインステージング解析レポート

**日付**: 2026-06-21
**分析対象**: ConvoPeq ソースコード（Audio Settings / DeviceSettings, AudioEngine 処理経路）
**関連ファイル**: `src/DeviceSettings.cpp`, `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.Processing.DSPCore*.cpp`, `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`, `src/IRConverter.cpp`, `src/ConvolverControlPanel.cpp`

---

## 1. Audio Settings 画面のゲイン入力ラベル調査

### 1.1 ラベル更新ロジック

`DeviceSettings::updateGainStagingDisplay()`（`src/DeviceSettings.cpp:584`）が 5Hz タイマーで動作し、モード切替に追従して以下のラベルを更新する：

| UI要素 | 初期テキスト | 更新後テキスト（例） |
|---|---|---|
| `inputHeadroomLabel` | `"Input Headroom:"` | `"Input Headroom (-12.0..-6.0 dB):"` |
| `outputMakeupLabel` | `"Output Makeup:"` | `"Output Makeup (0.0..12.0 dB):"` |

### 1.2 モード別分岐の正しさ

各モードで正しく分岐していることを確認済み。

| orderModeBox 表示 | eqBypassed | convBypassed | order | modeText | inputMaxDb |
|---|---|---|---|---|---|
| `Conv` | true | false | — | `"Conv only"` | -6.0 |
| `Peq` | false | true | — | `"PEQ only"` | 0.0 |
| `Conv->Peq` | false | false | ConvolverThenEQ | `"Conv -> PEQ"` | -6.0 |
| `Peq->Conv` | false | false | EQThenConvolver | `"PEQ -> Conv"` | 0.0 |

**判定**: ラベル更新ロジックは正しく、モード切替に追従して正常に変更される。

### 1.3 Output Makeup ラベル

`makeupMinDb = 0.0f`, `makeupMaxDb = 12.0f` はモード分岐内で一切変更されず、全モードで常に `"Output Makeup (0.0..12.0 dB):"` を表示する。Output Makeup は全処理チェーンの最終段に適用されるゲインであり、処理モードに依存しないため、これは設計として妥当。

### 1.4 補足: Input Headroom エディターの制約不一致

`inputHeadroomEditor.onTextChange` の丸め制限は常に `-12.0..0.0` である。Conv/Conv->Peq モードではラベルが上限 -6.0 を表示する一方、エディターは -3.0 のような範囲外の値も受け付ける。

```cpp
// src/DeviceSettings.cpp:368-374
inputHeadroomEditor.onTextChange = [this] {
    double val = inputHeadroomEditor.getText().getDoubleValue();
    if (val < -12.0) val = -12.0;
    if (val > 0.0) val = 0.0;          // ← 常に 0.0、モード別の上限なし
    audioEngine.setInputHeadroomDb(static_cast<float>(val));
};
```

一方 `AudioEngine::setInputHeadroomDb()`（Parameters.cpp:224）内部では正しくモード別に丸められるため、エンジン側は保護されている。

---

## 2. Conv->Peq 時のゲインステージング関係

### 2.1 信号経路図

```
入力信号 (float, 例: 0 dBFS = 1.0)
  │
  ▼
[1] float→double変換 (gain = 1.0)
  │
  ▼
[2] Input Headroom Gain (-6.0 dB, ×0.5012)
  │  ← Conv系では上限 -6.0 dB に制限
  │
  ▼
[3] DC Block (入力)
  │
  ▼
[4] ↑OS (オーバーサンプリング、任意)
  │
  ▼
[5] Convolver 処理
  │   IR Scale Factor (自動計算、≈ 0.5012)
  │   convolverInputTrimGain は Conv->Peq では適用されない
  │
  ▼
[6] EQ 処理
  │
  ▼
[7] Output Filter (HC/LC)
  │
  ▼
[8] ↓OS (ダウンサンプリング、任意)
  │
  ▼
[9] Output Makeup Gain (+12.0 dB, ×3.981)
  │  ← 全モード共通 0.0..12.0 dB
  │
  ▼
[10] Soft Clip (デフォルト有効, threshold ≈ 0.905 @sat=0.1)
  │
  ▼
[11] DC Block (出力)
  │
  ▼
[12] Noise Shaper / Dither (任意)
  │   kOutputHeadroom = 0.8912509 (-1.0 dB) 乗算
  │
  ▼
[13] ±0.891 ハードクリップ + double→float変換
  │
  ▼
出力信号 (float, max ≈ -1.0 dBFS)
```

### 2.2 3つのゲインの定義

#### ① Input Headroom Gain (`inputHeadroomGain`)

- **定義箇所**: `AudioEngine.h:1752`
- **初期値**: `0.5011872336272722`（-6.0 dB）
- **適用**: `DSPCoreIO.cpp:247` (`processInput`) — 入力信号に対して乗算
- **範囲**: Conv系: -12.0 ～ -6.0 dB / PEQ系: -12.0 ～ 0.0 dB
- **役割**: Convolver がチェーン先頭の場合、過大入力による畳み込み後のクリップを防止

#### ② IR Scale Factor (`scaleFactor`)

- **定義箇所**: `IRConverter.cpp:11` (`computeScaleFactor`)
- **適用**: IRファイル読み込み時に1回計算され、`LoadPipeline.cpp:324` で IR データに乗算
- **計算式**:

  ```
  makeup = 1.0 / sqrt(maxChannelEnergy)
  scaleFactor = makeup × 0.5011872336272722  (safetyMargin = -6 dB)
  → Peak/RMS 超過時は絶対値クランプでさらに抑制
  ```

- **役割**: IRファイルのエネルギーを正規化し、どのような IR でも出力クリップを防止

#### ③ Output Makeup Gain (`outputMakeupGain`)

- **定義箇所**: `AudioEngine.h:1754`
- **初期値**: `3.981071705534972`（+12.0 dB）
- **適用**: `DSPCoreFloat.cpp:248-253` — 全処理の最終段で乗算（Soft Clip の直前）
- **範囲**: 0.0 ～ 12.0 dB（全モード共通）
- **役割**: Input Headroom で落とした -6 dB と畳み込みによるエネルギー損失を補償

### 2.3 モード別デフォルト値

`applyDefaultsForCurrentMode()`（`AudioEngine.Parameters.cpp:288-318`）:

| モード | Input Headroom | Output Makeup | Conv Trim | 備考 |
|---|---|---|---|---|
| **Conv->Peq** | **-6.0 dB** | **+12.0 dB** | 0.0 dB（非表示） | Convが先頭 → 入力保護 |
| Peq->Conv | 0.0 dB | +10.0 dB | -6.0 dB（表示） | EQが先頭 → 入力保護不要 |
| Conv only | -6.0 dB | +12.0 dB | 0.0 dB（非表示） | |
| PEQ only | 0.0 dB | 0.0 dB | 0.0 dB（非表示） | |

### 2.4 Conv Trim の表示条件

`ConvolverControlPanel::updateTrimSlider()`（`ConvolverControlPanel.cpp:907`）:

```cpp
const bool shouldShowTrim = !eqBypassed && !convBypassed && isEqThenConv;
```

Conv Trim スライダーは **Peq->Conv (EQThenConvolver) 時のみ表示**される。Conv->Peq 時は非表示。

---

## 3. デフォルト設定で 0 dB を超える可能性

### 3.1 各段階の最大信号レベル（入力 0 dBFS 時）

| 段階 | ゲイン | 信号レベル(線形値) | 0dB超過? |
|---|---|---|---|
| ① 入力 float (0 dBFS) | 1.0 | 1.0 | — |
| ② Input Headroom Gain | ×0.5012 | 0.5012 (-6.0 dBFS) | ❌ |
| ③ Convolver (典型IR) | IR scaleFactor ≈ 0.5012 | 0.126～0.501 (-6～-18 dBFS) | △ IR次第 |
| ④ EQ (最大ブースト想定) | 最大 +12 dB | 最大 0.5～2.0 (+6 dBFS) | **◯ 可** |
| ⑤ Output Makeup Gain | **×3.981 (+12 dB)** | **最大 2.0～8.0 (+6～+18 dBFS)** | **✅ 確実** |
| ⑥ Soft Clip (default ON) | threshold ≈ 0.905 | 0.905 に圧縮 | ここで抑制 |
| ⑦ kOutputHeadroom | ×0.891 (-1.0 dB) | 0.806 (-1.87 dBFS) | ❌ |
| ⑧ 出力ハードクリップ | ±0.891 リミット | 最大 0.891 (-1.0 dBFS) | ❌ |

### 3.2 0dB 超過が発生する箇所

#### ✅ ⑤ Output Makeup Gain 適用後（最も深刻）

`DSPCoreFloat.cpp:248-253`

```cpp
for (size_t ch = 0; ch < processBlock.getNumChannels(); ++ch) {
    double* ptr = processBlock.getChannelPointer(ch);
    scaleBlockFallback(ptr, (int)processBlock.getNumSamples(), state.outputMakeupGain);
}
```

+12.0 dB（×3.981）の固定ゲインにより、**入力が -12 dBFS より大きい場合、必ず 0 dBFS を超過**する。

#### △ ③ Convolver 出力

IR Scale Factor がエネルギー正規化を行うため典型的には入力以下だが、IRの共振特性によって建設的干渉が発生し、入力を上回る可能性がある。

#### ◯ ④ EQ 出力

EQ の周波数ブーストにより、畳み込み後の信号がさらに持ち上げられる。

### 3.3 最終出力の保護機構（二重防御）

| 防御 | 条件 | しきい値 | 効果 |
|---|---|---|---|
| Soft Clip | デフォルト有効 (`softClipEnabled=true`) | ≈ 0.905 (-0.87 dB) @sat=0.1 | ピークをなめらかに圧縮 |
| 出力ハードクリップ | 常時有効 (`processOutput`) | ±0.891 (-1.0 dBFS) | 絶対最大値を保証 |

**`processOutput`（`DSPCoreIO.cpp:498-505`）**:

```cpp
constexpr double kOutputHeadroom = 0.8912509381337456;  // ≈ -1.0 dB
// ...
dstL[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]));
```

Soft Clip が無効でも、このハードクリップにより最終出力が ±0.891 (-1.0 dBFS) を超えることはない。

### 3.4 結論

| 箇所 | 0dB超過 | 補足 |
|---|---|---|
| 内部（Output Makeup 後） | ✅ **ほぼ確実に超過** | +12 dB ゲインにより +6～+18 dBFS に到達 |
| 最終 float 出力 | ❌ **されない** | -1.0 dBFS ハードクリップ + Soft Clip で二重防御 |
| Soft Clip OFF 時 | ❌ **されない**（ただし歪み増加） | -1.0 dBFS ハードリミッターが最終防衛線 |

---

## 4. 関連ソースコード一覧

| ファイル | 該当行 | 内容 |
|---|---|---|
| `src/DeviceSettings.cpp` | 584-637 | `updateGainStagingDisplay()` ラベル更新 |
| `src/DeviceSettings.cpp` | 508-525 | `timerCallback()` 5Hz定期実行 |
| `src/audioengine/AudioEngine.Parameters.cpp` | 224-247 | `setInputHeadroomDb()` モード別上限 |
| `src/audioengine/AudioEngine.Parameters.cpp` | 288-318 | `applyDefaultsForCurrentMode()` デフォルト値 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 240-260 | Output Makeup + Soft Clip 適用 |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | 342-505 | `processOutput()` 出力保護 |
| `src/IRConverter.cpp` | 11-60 | `computeScaleFactor()` IR正規化 |
| `src/ConvolverControlPanel.cpp` | 907-920 | `updateTrimSlider()` Conv Trim表示条件 |
| `src/MainWindow.cpp` | 1236-1256 | `orderModeBoxChanged()` モード切替 |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | 198-265 | `processInput()` 入力処理 |
