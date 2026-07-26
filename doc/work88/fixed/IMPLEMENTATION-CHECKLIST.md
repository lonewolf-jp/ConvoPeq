# 実装チェックリスト — work88 改修計画 v6

**作成日**: 2026-07-26
**進捗状況**: ✅ Phase 1-3 完了 / 🟡 Phase 4 未着手

---

## ✅ Phase 1: 確定着手可能（全5件完了）

| # | BUG-ID | ファイル | 変更内容 | 状態 |
|---|--------|---------|---------|------|
| 1 | BUG12 | `ConvolverProcessor.h:268-269` | enterStateReader/exitStateReader → rcuSwapper委譲 | ✅ **完了** |
| 2 | BUG13 | `SafeStateSwapper.h:106-130` | retire epoch1→epoch2, コメント更新 | ✅ **完了** |
| 3 | BUG11 | `AudioEngine.h:1996-2030` | NonOwningPtr→std::atomic<DSPCore*>, アクセサ置換 | ✅ **完了** |
| 4 | BUG4 | `AudioBlock.cpp:605` | xRunBuffer.push()戻り値チェック追加 | ✅ **完了** |
| 5 | BUG10 | `DeferredDeletionQueue.h:80,120,172` | intptr_t→int32_tモジュラ減算 | ✅ **完了** |

## ✅ Phase 2: レビュー承認済み実装案（全3件完了）

| # | BUG-ID | ファイル | 変更内容 | 状態 |
|---|--------|---------|---------|------|
| 6 | BUG-001/002/003 | `DSPCoreIO.cpp:512-530` | zeroupper/truePeakDetector/loudnessMeter/peakLimiter追加 | ✅ **完了** |
| 7 | BUG17 | `AudioEngine.Retire.cpp:136` | /1000→/1'000'000 | ✅ **完了** |
| 8 | BUG16 | `ISRRetire.cpp:56` | duration_cast<microseconds>追加 | ✅ **完了** |

## ✅ Phase 3: カウンタ追加（1件完了）

| # | BUG-ID | ファイル | 変更内容 | 状態 |
|---|--------|---------|---------|------|
| 9 | BUG-010 | `EQProcessor.h`, `EQProcessor.Core.cpp`, `EQProcessor.Parameters.cpp`, `EQProcessor.Coefficients.cpp` | m_retireDropCount追加, 17箇所の(void)キャスト→カウンタ | ✅ **完了** |

## 🟡 Phase 4: 後回し可（未着手）

### C1. `_mm256_zeroupper()` 追加（8ファイル）

| # | ファイル | 状態 | 備考 |
|---|---------|------|------|
| 1 | `CustomInputOversampler.cpp` | ⬜ 未着手 | isBadSampleV, loadStride2, FIRカーネル関数末尾 |
| 2 | `ConvolverProcessor.Runtime.cpp` | ⬜ 未着手 | #if defined(__AVX2__) ブロック末尾 |
| 3 | `MKLNonUniformConvolver.cpp` | ⬜ 未着手 | AVX2使用FFT/conv関数末尾 |
| 4 | `SpectrumAnalyzerComponent.cpp` | ⬜ 未着手 | スペクトル計算AVX2ブロック末尾 |
| 5 | `EQProcessor.Processing.cpp` | ⬜ 未着手 | applyGainRamp_AVX2関数末尾 |
| 6 | `AudioEngine.EQResponse.cpp` | ⬜ 未着手 | __m256d使用各ブロック末尾 |
| 7 | `ConvolverProcessor.LoaderThread.cpp` | ⬜ 未着手 | Loader処理AVX2ブロック末尾 |
| 8 | `DSPCoreIO.cpp` | ✅ **完了** | BUG-001/002/003と同時修正済み |

**テンプレート**:
```cpp
#if defined(__AVX2__)
    _mm256_zeroupper();
#endif
```

### C2. `static_cast<size_t>` 追加（24箇所）

| # | ファイル | 箇所数 | 状態 |
|---|---------|--------|------|
| 1 | `MKLNonUniformConvolver.cpp` | 18 | ⬜ 未着手 |
| 2 | `ConvolverProcessor.Runtime.cpp` | 2 (388,390) | ⬜ 未着手 |
| 3 | `ConvolverProcessor.Lifecycle.cpp` | 2 (248,249) | ⬜ 未着手 |
| 4 | `ConvolverProcessor.h` | 2 (821,822) | ⬜ 未着手 |

**テンプレート**:
```cpp
// 基本:
memcpy(dst, src, static_cast<size_t>(n) * sizeof(double));
// 複合式（掛け算の前にキャスト）:
memcpy(dst, src, static_cast<size_t>(complexSize) * 2 * sizeof(double));
```

---

## 実装サマリ

| Phase | 完了 | 未着手 | 合計 |
|-------|------|--------|------|
| Phase 1 (A) | 5 | 0 | 5 |
| Phase 2 (B) | 3 | 0 | 3 |
| Phase 3 (BUG-010) | 1 | 0 | 1 |
| Phase 4 (C1) | 0 | 7 | 7 |
| Phase 4 (C2) | 0 | 4 | 4 |
| **合計** | **9** | **11** | **20** |
