更新日: 2026-03-22

## [MKLNonUniformConvolver/ConvolverProcessor系] メモリアロケーション監査（2026-03-22更新）

- **アロケーション方式**
  - すべての大規模バッファ（IR, FFT, 作業領域等）は`convo::aligned_malloc`（64byteアライメント）＋`ScopedAlignedPtr`（RAII）で確保・解放。
  - JUCEの`AudioBuffer<double>`は内部バッファとして一時的に使うが、MKL/AVX経路で使うデータは必ずaligned_mallocで再確保し、アライメントを保証。
  - `std::vector`や`new`/`delete`は**一切使用せず**、MKL/AVX経路はすべて明示的なアライメント確保。
  - FFT/DFTバッファ、IRバッファ、ウィンドウ関数バッファ、変換用一時バッファ等、すべて64byteアライメント。

- **確保タイミング・スレッド**
  - すべてのメモリ確保は**Audio Thread外**（LoaderThread/Message Thread/prepareToPlay等）でのみ実施。
  - Audio Thread（getNextAudioBlock等）では**一切のメモリ確保・解放なし**。
  - LoaderThread/prepareToPlay/releaseResources等でバッファの確保・解放を集中管理。

- **RAII・例外安全性**
  - すべてのバッファは`ScopedAlignedPtr`でRAII管理。例外・早期return時もリークなし。
  - StereoConvolver等のエンジン本体もaligned_malloc＋placement new＋addRef/releaseで厳密管理。
  - try-catchで`std::bad_alloc`等の例外を捕捉し、失敗時は必ずrelease/解放を実施。

- **解放タイミング**
  - releaseResources/デストラクタ/prepareToPlay等で、すべてのバッファ・エンジンを明示的に解放。
  - activeConvolution, fadingOutConvolution, trashBin等も参照カウント・ロックで安全に解放。

- **アライメント保証**
  - convo::aligned_mallocは常に64byteアライメント。AVX2/AVX512/MKLの要件を完全充足。
  - JUCEのAudioBuffer→MKL経路は必ずaligned_mallocで再確保し、アライメント違反を防止。

- **バッファ種別と用途**
  - IRバッファ（L/R）, FFTバッファ, 窓関数バッファ, 作業領域, Delay/Dry/Wet/Oldバッファ, crossfade/smoothingバッファ等、すべて用途ごとに明示的に確保・解放。
  - 一時バッファもすべてScopedAlignedPtrでRAII管理。

- **禁止事項遵守**
  - Audio Thread内でのメモリ確保・libm呼び出し・同期/通信・例外等は**一切なし**。
  - JUCE/・r8brain-free-src/配下の編集・独自アロケータ導入もなし。

- **バグ・修正履歴**
  - 過去にstd::vectorやJUCE AudioBufferのアライメント不足によるクラッシュ・パフォーマンス劣化があり、すべてaligned_malloc＋RAIIに統一済み。
  - 例外発生時のリーク・Use-After-Freeもtry-catch＋releaseで完全防止。

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
