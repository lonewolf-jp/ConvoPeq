# LatticeNoiseShaper state convention 最終報告書 v5

**日付**: 2026-06-23
**状態**: 全条件テスト完了（10SR x 3bit = 30条件）。デフォルト係数更新 + Debugログ追加完了。

---

## 実施済み変更

**デフォルト係数を Pattern B 向けに更新（2ファイル）**:

- src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:16
- src/audioengine/AudioEngine.Learning.cpp:286

旧値: {0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07}
新値: {-0.003796, -0.006752, 0.008418, -0.010546, 0.004716, -0.007624, -0.020750, -0.002049, -0.003632}

## 全条件テスト結果（全10サンプルレート × 3ビット深度 = 30条件）

### 新デフォルト (Pattern B最適化): 全30条件 STABLE

| SR     | 16bit DC | 16bit RMS | 24bit DC | 24bit RMS | 32bit DC | 32bit RMS | SNR(24) |
|--------|:--------:|:---------:|:--------:|:---------:|:--------:|:---------:|:-------:|
| 44.1k  | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.5dB |
| 48k    | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.5dB |
| 88.2k  | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.1dB |
| 96k    | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 100.7dB |
| 176.4k | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.2dB |
| 192k   | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.3dB |
| 352.8k | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 100.9dB |
| 384k   | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.5dB |
| 705.6k | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.1dB |
| 768k   | 3.2e-7   | 1.6e-5    | 1.2e-9   | 6.1e-8    | 4.9e-12  | 2.4e-10   | 101.4dB |

### 旧デフォルト (Pattern A最適化, Pattern Bで使用): 全30条件 DRIFT

全10SR x 3bit で Peak=1.0 (Full Scale), DC=0.01〜0.33, RMS~1.0。

### ロバストネステスト（+/-30%摂動100回）

新デフォルト: 0/100 失敗（全試行安定、平均 SNR=101.0dB）
旧デフォルト: 平均 DC=0.18、平均 SNR=-43dB（破綻）

## 実機ログ機構

adaptiveCoeffSet変更時のDebugログ追加（2026-06-23実装完了）:
- ✅ `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:425` — `juce::Logger::writeToLog("[AudioEngine] DSPCoreIO::processInput: adaptiveCoeffSet switch bank=... gen=...")`
- ✅ `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:725` — `juce::Logger::writeToLog("[AudioEngine] DSPCoreDouble::processDoubleToBuffer: adaptiveCoeffSet switch bank=... gen=...")`
- ログ内容: bankIndex + generation番号（係数切り替わり時に1行出力）
- 既存の `[AudioEngine]` プレフィックス統一パターンに準拠

## 評価サマリ

| 項目 | 判定 |
|------|------|
| Pattern B維持 | 妥当 |
| P7ロールバック見送り | 妥当 |
| デフォルト係数更新（2ファイル） | ✅ 完了 |
| Debugログ追加（DSPCoreIO/Double） | ✅ 完了 |
| 全10SR(44.1k〜768k)での安定性 | 確認済み (30条件OK) |
| 問題は「解決済み」と断定 | 時期尚早（実機確認必要） |

## データ保存先

- doc/work54/data/all_10sr_test.csv (30条件x2の生データ)
- doc/work54/data/ntf_comparison.csv
- doc/work54/data/bitdepth_comparison.csv
- doc/work54/data/learning_log_summary.csv
- tools/analysis/compare_noiseshaper_patterns.py
- %APPDATA%\ConvoPeq\noise_shaper_learn.xml (267KB)
