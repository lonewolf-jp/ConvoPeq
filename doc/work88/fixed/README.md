# ConvoPeq バグ発見レポート — 成果サマリ (work88)

## 探索範囲
探索日: 2026-07-26

### カバーしたファイル
- SpectrumAnalyzerComponent (.h/.cpp) — 新規探索
- CustomInputOversampler (.h/.cpp) — 新規探索
- NoiseShaperLearner (.h/.cpp) — 詳細分析
- DeferredDeletionQueue (.h) — レビュー
- RuntimePublicationCoordinator (.h) — レビュー
- AudioEngine.Init.cpp — レビュー
- AudioEngine.CtorDtor.cpp — レビュー
- EQProcessor (.h/.cpp) — 概要＋Processing.cpp LSP診断確認
- PeakEstimator (.h/.cpp) — 詳細分析
- No tests directory found
- DspNumericPolicy.h — killDenormalV 定義確認

## 発見バグ: 24件

### BUG-018〜BUG-037 (20 bugs)
探索日: 2026-07-26 (前半セッション)
- *詳細は doc/work88/BUG-018-*.md 〜 BUG-037-*.md を参照*

### BUG-038〜BUG-041 (4 bugs, 今回追加)
| ID | ファイル | タイトル | 重大度 |
|----|---------|---------|--------|
| BUG-038 | SpectrumAnalyzerComponent.h:74 | FFT マグニチュード +6 dB スケーリング誤差 | **CRITICAL** |
| BUG-039 | CustomInputOversampler.cpp:836-841 | processDown パススルー時バッファオーバーリード | **HIGH** |
| BUG-040 | NoiseShaperLearner.cpp:1164-1168 | 再生時間 1Hz フォールバックで学習異常終了 | MEDIUM |
| BUG-041 | NoiseShaperLearner.cpp:643 | 可変長配列（VLA）スタック破壊リスク | **HIGH** |

## 発見しきれなかった領域
- テストコード（tests/ ディレクトリなし — プロジェクトに存在しない）
- IRConverter.cpp（JuceHeaderのインクルードパスエラーのみ確認、論理バグは未探索）
- 各種 ISR ファイル（isr/ISRRuntimePublicationCoordinator 等 — 極めて堅牢に見えた）
- EQProcessor.Processing.cpp（長大につき部分読了のみ）

## 重要な所見
1. **SpectrumAnalyzer +6 dB** は最も影響の大きい発見（全周波数で誤表示）
2. **Oversampler buffer over-read** は条件次第で常時ノイズ
3. **VLA スタック破壊** は MSVC でのコンパイル互換性問題
4. **AudioEngine.h の const-correctness 警告** (enterReader on const object) は SafeStateSwapper.h:141 が const 非対応の可能性
5. ほとんどのコードは異常系ガード（bad sample検出、nullチェック、範囲チェック）が充実しており、品質水準は高い
