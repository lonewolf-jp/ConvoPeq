# Memory Allocation Audit (MKL-path scoped)

更新日: 2026-03-16

## 監査スコープ

- 対象: `src/` 配下の自前コード
- 非対象: `JUCE/` と `r8brain-free-src/`（編集禁止・サードパーティ）
- 判定基準: 「MKL API を直接呼ぶ処理経路」または「MKLバックエンド周辺の一時確保」

## 反映済み（置換完了）

- `src/ConvolverProcessor.cpp`
  - FFTスペクトラム平滑化の一時バッファを `std::vector<float>` から `mkl_malloc + ScopedAlignedPtr<float>` へ置換
  - IR有効長推定の一時バッファを `std::vector<double>` から `mkl_malloc + ScopedAlignedPtr<double>` へ置換
- `src/AudioEngine.cpp`
  - `timerCallback()` の削除退避バッファを `std::vector<DSPCore*>` から `ScopedAlignedPtr<DSPCore*>` へ置換
- `src/ConvolverProcessor.cpp`
  - `cleanup()` の `toRelease` を `ScopedAlignedPtr<StereoConvolver*>` へ置換
  - `forceCleanup()` の `stereoConvolversToDelete` を `ScopedAlignedPtr<std::pair<StereoConvolver*, uint32>>` へ置換

## 残存 std::vector（維持理由付き）

### A. API/可視化データ（MKL演算バッファではないため維持）

- `src/ConvolverProcessor.h`
  - `getIRWaveform()`, `getIRMagnitudeSpectrum()` の返却型
  - `irWaveform`, `irMagnitudeSpectrum` の保持
- 理由:
  - UI層へ可変長データを返す公開APIであり、`std::vector` が自然
  - MKL API 入出力バッファではない

### B. ゴミ箱本体コンテナ（可変長所有管理のため維持）

- `src/AudioEngine.h`: `trashBin`
- `src/ConvolverProcessor.h`: `trashBin`
- `src/EQProcessor.h`: `bandNodeTrashBin`, `bandNodeTrashBinPending`, `stateTrashBin`, `stateTrashBinPending`
- 理由:
  - これらは「寿命管理コンテナ本体」。push/erase が必要な可変長管理構造
  - Audio Thread 直下で確保していない（Message/Timer 側）
  - 一時退避バッファは置換済みで、リアルタイム影響を低減済み

### C. UI状態管理（MKL無関係）

- `src/EQControlPanel.h`: `controlMap`
- `src/SpectrumAnalyzerComponent.h`: `individualCurvePathsL`, `individualCurvePathsR`
- 理由:
  - UIレンダリング／コントロール対応表のため、MKLパスではない

## 結論

- 「MKL経路限定」方針での置換対象は対応済み。
- 残存 `std::vector` は、公開API・可視化・可変長ライフサイクル管理など、`mkl_malloc + ScopedAlignedPtr` へ一律置換する合理性が低い箇所のみ。
- 今後の運用ルール:
  1. MKL API を直接呼ぶ新規コードでは `std::vector` を使わない。
  2. Audio Thread でメモリ確保を行わない。
  3. 可視化/API返却用途は必要最小限で `std::vector` 維持可。
