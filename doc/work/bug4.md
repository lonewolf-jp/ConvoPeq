# bug4 修正優先順位付きアクションプラン

以下、これまでのレビューと検証を踏まえた有効なバグ・問題点の改修計画書を提示します。

---

## 現在の状況メモ

- 項目 1 は修正済み。
- 項目 2 は設計改善候補として観察継続。
- 項目 5 は有効な設計リスクとして継続管理。
- 項目 3, 4, 6, 7, 8, 9, 10, 11 は現コードで不成立または低優先として整理済み。

## 凡例

- **重要度**: Critical / High / Medium / Low
- **対象**: ファイル名（おおよその行番号）
- **問題**: バグまたは設計上のリスク
- **改修方針**: 具体的な修正案
- **工数**: 推定作業時間（目安）

---

## 🔴 Critical（至急対応）

### 1. Audio Thread 内での MKL BLAS (`cblas_dscal`) 呼び出し

> ステータス: ✅ 修正済み（2026-05-24）

- **対象**: `src/InputBitDepthTransform.h` (L100-120), `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` (L150-170)
- **問題**: Audio Thread から `cblas_dscal` を呼び出している。MKL BLAS は初回呼び出し時に内部メモリ確保やロック初期化を行う可能性があり、Audio Thread のリアルタイム制約に違反する。
- **改修**: MKL BLAS 依存を排除し、AVX2 SIMD またはスカラーループで代替。`applyHighQuality64BitTransform` 内の `cblas_dscal` を以下のような実装に置き換える。

  ```cpp
  const __m256d vGain = _mm256_set1_pd(gain);
  int i = 0;
  const int vEnd = numSamples / 4 * 4;
  for (; i < vEnd; i += 4) {
      __m256d v = _mm256_loadu_pd(data + i);
      _mm256_storeu_pd(data + i, _mm256_mul_pd(v, vGain));
  }
  for (; i < numSamples; ++i) data[i] *= gain;
  ```

- **工数**: 0.5日

---

### 2. `RuntimePublishWorld` の生ポインタ観測に RCU ガードがない

- **対象**: `src/audioengine/AudioEngine.h` (`getRuntimePublishView`), `src/core/RuntimeStore.h` (`observe`)
- **問題**: `RuntimeStore::observe()` は生ポインタを返し、呼び出し側で `EpochDomainReaderGuard` を使っていないため、参照中の `RuntimeState` が解放される可能性がある（UAF リスク）。
- **改修**: `ObservedRuntime` のような RAII ガードを導入する。

  ```cpp
  struct ObservedRuntimeWorld {
      EpochDomainReaderGuard guard;
      const RuntimePublishWorld* ptr;
  };
  ObservedRuntimeWorld observeRuntimeWorld() const;
  ```

  または、`getRuntimePublishView()` 内で一時的に RCU reader を入退場させる。
- **工数**: 1日

---

### 3. `SafeStateSwapper::tryReclaim` の複数スレッド呼び出しリスク

> ステータス: ⏳ 監視中（現時点では不成立）

- **対象**: `src/SafeStateSwapper.h` (L120-180)
- **問題**: `tryReclaim` は `head` の読み取り、エントリ取得、`head` 更新を複数ステップで行っており、現在は単一スレッド (`DeferredFreeThread`) のみから呼ばれているが、コメント上は複数スレッド呼び出しを想定している。将来複数スレッドから呼ばれた場合、同一エントリの二重解放（double-free）のリスクがある。
- **改修**:
  - 呼び出しスレッドを明確に単一に制限し、`jassert` を追加する。
  - または `compare_exchange_strong` ループで `head` 更新をアトミック化する。
  - コメントと実装の整合性を取る。
- **工数**: 0.5日

---

## 🟠 High（早めの対応推奨）

### 4. `AllpassDesigner::applyAllpassToIR` の未使用パラメータと安全ガード不備

- **対象**: `src/AllpassDesigner.cpp` (L305-420)
- **問題**:
  - `freq_hz` パラメータが全く使用されていない（設計ミス）。
  - NaN/Inf 検出後、空のバッファを返すが、呼び出し元が適切にハンドリングしていない可能性。
  - ピークリダクションが常に約 -3dB 固定で、過減衰の可能性。
- **改修**:
  - `freq_hz` を削除するか、実際に使用する処理を追加。
  - ピークリダクションを RMS ベースの適応的ゲインに変更。
  - 戻り値の空チェックを呼び出し側に追加（または assert）。
- **工数**: 1日

---

### 5. `DSP lifecycle` の ownership 分散（アーキテクチャリスク）

- **対象**: `src/audioengine/AudioEngine.h` (`activeDSP`, `fadingOutDSP`, `pendingTask.currentDSP`, `retireDSP`)
- **問題**: `DSPCore` の寿命管理が複数の経路 (`activeDSP`, `fadingOutDSP`, `RuntimeState`, `pendingTask`) に分散しており、`retireDSP` 経由の解放と RCU ベースの解放が混在。現状は慎重な制御で動いているが、将来的な変更で UAF を引き起こしやすい。
- **改修**:
  - `DSPCore` の所有権を一箇所に集約する（例: `std::unique_ptr` で管理し、公開は生ポインタ＋RCU ガードのみ）。
  - または `RefCountedDeferred` を継承して参照カウント方式に統一する。
  - 短期対応としては、`retireDSP` の呼び出し箇所を体系的にドキュメント化し、コードレビューで厳格に管理する。
- **工数**: 3～5日（リファクタリング規模による）

---

### 6. `NoiseShaperLearner` の `candidatePopulation` 同期不足

- **対象**: `src/NoiseShaperLearner.cpp` (L600-650), `src/NoiseShaperLearner.h` (L140-150)
- **問題**: `candidatePopulationMatrix()` が返すバッファは `optimizer.sample()` で書き換えられると同時に、評価ワーカーから読み取られる。適切な同期機構がなく、データ競合の可能性がある。
- **改修**:
  - ダブルバッファリングを導入（`sample` 用と `evaluate` 用を分け、`swap` で切り替え）。
  - または `std::atomic` でフラグを立てて排他制御。
- **工数**: 1日

---

### 7. `LinearRamp::skip` の浮動小数点誤差・ゼロステップバグ

- **対象**: `src/DspNumericPolicy.h` (`convo::LinearRamp::skip`)
- **問題**: `step` がゼロの場合に `current` が更新されず、`remaining` だけが減って誤動作する。また浮動小数点誤差による累積も考慮されていない。
- **改修**:

  ```cpp
  void skip(int numSamples) noexcept {
      if (numSamples <= 0 || remaining <= 0) return;
      if (step == 0.0) { remaining = 0; return; }  // 追加
      if (numSamples >= remaining) {
          current = target;
          remaining = 0;
          return;
      }
      current += step * static_cast<double>(numSamples);
      remaining -= numSamples;
  }
  ```

- **工数**: 0.5日

---

## 🟡 Medium（余裕を持って対応）

### 8. `ConvolverProcessor::prepareToPlay` の例外安全性

- **対象**: `src/convolver/ConvolverProcessor.Lifecycle.cpp` (`prepareToPlay`)
- **問題**: `convo::makeAlignedArray` は `std::bad_alloc` を投げうるが、JUCE ホスト環境では例外がキャッチされない場合クラッシュする。
- **改修**: `std::nothrow` 版のアロケータを使用するか、`try-catch` で囲み失敗時に安全にロールバックする。
- **工数**: 0.5日

---

### 9. 未初期化変数 `dspCrossfadeArmed_RT` の可能性

> ステータス: ❌ 不成立

- **対象**: `src/audioengine/AudioEngine.h` (`dspCrossfadeArmed_RT`), `src/audioengine/AudioEngine.Init.cpp` (`initialize`)
- **問題**: `dspCrossfadeArmed_RT` は `initialize()` で `false` に設定されるが、`initialize()` が呼ばれる前に `armCrossfadeIfPending()` が実行される経路がある可能性（厳密にはコンストラクタと初期化順序に依存）。
- **改修**: メンバ変数宣言時に初期化（`bool dspCrossfadeArmed_RT = false;`）する。
- **工数**: 0.25日

---

### 10. `MKLNonUniformConvolver` の `applySpectrumFilter` メモリ断片化

> ステータス: ✅ 実装確認済み（バッファ再利用あり）

- **対象**: `src/MKLNonUniformConvolver.cpp` (`applySpectrumFilter`)
- **問題**: ループ内で `gain` バッファを毎回確保・解放しており、メモリ断片化のリスクがある（実害は稀）。
- **改修**: `gain` をループ外で確保し再利用する。
- **工数**: 0.25日

---

### 11. コピペによるコメント誤り

- **対象**: `src/audioengine/AudioEngine.h` (L2500 付近), `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` (L100)
- **問題**: メモリオーダーに関するコメントが実際の変数名と一致していない箇所がある。
- **改修**: コードレビューでコメントを修正。
- **工数**: 0.25日

---

## 🔵 Low / Info（任意対応）

### 12. 未使用変数の再確認

> ステータス: ℹ️ 追加削除対象なし（確認済み）

- `src/AllpassDesigner.cpp` の `freq_hz` は実装内で使用されているか、意図的に無視している箇所があるため、現時点での削除対象ではない
- `src/audioengine/AudioEngine.Learning.cpp` の `dspHasIrForRebuild` は現行実装で使用されている
- `src/eqprocessor/EQProcessor.Core.cpp` の `maxBandsWarningShown` も現行実装で使用されている

### 13. `std::llround` の可読性向上

- Message Thread での使用のため問題はないが、`static_cast<int>(std::round(...))` に統一してもよい。

---

## 優先順位と依存関係

```text
優先度1 (Critical) → 項目1,2,3
       ↓
優先度2 (High)   → 項目4,5,6,7
       ↓
優先度3 (Medium) → 項目8,9,10,11
       ↓
優先度4 (Low)    → 項目12,13
```

**依存関係**:

- 項目5（DSP lifecycle）は大規模リファクタリングのため、他の修正と並行して進めるか、後回しにする。
- 項目2（RuntimePublishWorld）は項目5と関連する可能性があるが、独立して修正可能。
- 項目4（AllpassDesigner）は機能に影響しないため後回し可。

---

## 推奨実施順序

1. **項目1** (`cblas_dscal` 置換) — Audio Thread の安定性に直結
2. **項目2** (RuntimePublishWorld ガード) — UAF リスク除去
3. **項目3** (SafeStateSwapper スレッド制限) — 将来のバグ予防
4. **項目7** (LinearRamp::skip) — 簡単に修正可能
5. **項目9** (未初期化変数) — 簡単に修正可能
6. **項目6** (NoiseShaperLearner 同期) — 学習機能の信頼性向上
7. **項目8,10,11** — 余裕があれば
8. **項目5** (DSP lifecycle) — 設計レビュー後に計画的に実施

---

以上が有効なバグの改修計画です。特に **項目1** と **項目2** は早急な対応を推奨します。
