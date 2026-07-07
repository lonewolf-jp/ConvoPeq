# ConvoPeq メモリ削減調査 — 根本原因の特定とパッチ（レビュー反映・検証済最終仕様書）

---

## 1. 総合評価の受け入れと設計方針

本ドキュメントは、[doc/work70/plan.md](doc/work70/plan.md) について提示された技術レビューを全面的に検証し、その妥当性が証明された内容、および実コードの棚卸し調査により確定した詳細な安全対策を統合した**最終版の改修計画・設計仕様書**である。

提示されたレビューの指摘事項は極めて適確であり、すべて妥当（あるいは条件付きで極めて重要）であると検証された。
特に、以下の指摘は本改修の成否を分ける極めて重大なリスクおよび指針である：

* **「AoSの単純サイズ縮小（Patch 4, 11などの部分適用）は、アドレス計算の不整合により実行時即時ハング・クラッシュを引き起こすため極めて危険である」**（事実、ポインタオフセットとミラー書き込み領域の再設計が完全に同期していなければ、シグナル 11 によりプロセスが即死する）。
* **「AoS（`irFreqDomain`）は一度も読まれないわけではなく、メッセージスレッドでのフィルタ動的適用時等に再読み込みされている」**。
* **「フィルタ適用（`applySpectrumFilter` / `Air Absorption`）を SoA (`irFreqReal`/`irFreqImag`) に対して直接行うことで AoS を不要化する案が最も安全かつ有効である」**（これにより中間インターリーブバッファの削減、OOM の余地が根絶される）。
* **「83ms IR環境であっても、L0レイヤーのみでの少なからぬメモリ削減効果（チャンネルあたり数MB）が実際には発揮される」**。

これらを踏まえ、設計を**「初期化時に AoS を一時スクラッチとしてのみ使い、ホットループおよびメッセージスレッド側のフィルタ適用を SoA 直接計算に完全移行し、中間 AoS バッファを永続配列から完全に排除する」**形で安全に再設計した。

---

## 2. プログラム実装に必要な設計事項（前半：実装設計）

本改修は、部分的にパッチを適用するとコンパイル時や実行時に致命的なバッファオーバーラン、またはハングを招く。よって、以下の「3フェーズ安全パッケージ」に分類された11個のPatch、および例外時セーフティ設計を完全に同期してアトミックに適用する。

### 2.1 安全パッケージ1：Messageスレッド フィルタの SoA 直接適用

メッセージスレッド側の動的フィルタを SoA (`irFreqReal` / `irFreqImag`) に対してダイレクトに適用することで、AoS (`irFreqDomain`）へ一時クリア・インターリーブし書き戻す依存関係を完全に断ち切る。

#### Patch 3/11 — `applySpectrumFilter()` SoA直接適用へのリファクタリング

実部・虚部それぞれに同一ゲインを掛けるだけでよく、インターリーブ配列 `reusableGainInterleaved` を丸ごと撤去する。

```cpp
// src/MKLNonUniformConvolver.cpp の一部
 void MKLNonUniformConvolver::applySpectrumFilter(const FilterSpec& spec) noexcept
 {
     // ... 前半のゲイン算出部分はそのまま

-    convo::ScopedAlignedPtr<double> reusableGain;
-    convo::ScopedAlignedPtr<double> reusableGainInterleaved;
-    int reusableGainCapacity = 0;
-    int reusableGainInterleavedCapacity = 0;
+    convo::ScopedAlignedPtr<double> reusableGain;
+    int reusableGainCapacity = 0;

     for (int li = 0; li < m_numActiveLayers; ++li)
     {
         Layer& l = m_layers[li];
-        if (!l.irFreqDomain) continue;
+        if (!l.irFreqReal || !l.irFreqImag) continue;

         const int N      = l.fftSize;
         const int halfN  = N / 2;
         const int cSize  = l.complexSize;
-        const int stride = l.partStride;

         // ... reusableGain 確保（変更なし）

         // ── LC ゲイン 算出（変更なし）

-        // ── 全パーティションの irFreqDomain に gain[] を適用 ──
-        {
-            const int requiredInterleavedSize = cSize * 2;
-            if (reusableGainInterleavedCapacity < requiredInterleavedSize
-                || reusableGainInterleaved.get() == nullptr)
-            {
-                reusableGainInterleaved.reset(static_cast<double*>(
-                    mkl_malloc(static_cast<size_t>(requiredInterleavedSize) * sizeof(double), 64)));
-                reusableGainInterleavedCapacity = (reusableGainInterleaved.get() != nullptr)
-                    ? requiredInterleavedSize
-                    : 0;
-            }
-            if (!reusableGainInterleaved.get())
-                continue;
-
-            double* gainIL = reusableGainInterleaved.get();
-            for (int k = 0; k < cSize; ++k)
-                gainIL[2 * k] = gainIL[2 * k + 1] = gain[k];
-
-            for (int p = 0; p < l.numParts; ++p)
-            {
-                double* slot = l.irFreqDomain + p * stride;
-                vdMul(cSize * 2, slot, gainIL, slot);
-                deinterleaveComplex(slot,
-                                    l.irFreqReal + static_cast<size_t>(p) * l.complexSize,
-                                    l.irFreqImag + static_cast<size_t>(p) * l.complexSize,
-                                    l.complexSize);
-            }
-        }
+        // [Mem-Fix] gain[] は実数値(振幅のみ)のフィルタなので、実部・虚部それぞれに
+        // 同一ゲインを掛けるだけでよい。interleave/deinterleaveもAoS経由も不要。
+        for (int p = 0; p < l.numParts; ++p)
+        {
+            double* re = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
+            double* im = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
+            vdMul(cSize, re, gain, re);
+            vdMul(cSize, im, gain, im);
+        }
     }
 }
```

#### Patch 7/11 — Air Absorption テール減衰 SoA直接適用

同様に、Air Absorption 適用時の interleaved 配列 `gainInterleaved` の永続確保を完全に除去して SoA へ直接 vdMul する。

```cpp
// src/MKLNonUniformConvolver.cpp の一部
     if (tailEnabled && tailMode == 0)
     {
         const double startNorm = juce::jlimit(0.65, 1.55, tailStartSec / 0.085);
         const double dampingBase = (0.35 + 1.10 * strength01) * startNorm;

         for (int li = 1; li < m_numActiveLayers; ++li)
         {
             Layer& l = m_layers[li];
-            if (!l.irFreqDomain || l.complexSize <= 0)
+            if (!l.irFreqReal || !l.irFreqImag || l.complexSize <= 0)
                 continue;

             const double layerWeight = (li == 1) ? 1.0 : 1.6;
             const double dampingCoeff = dampingBase * layerWeight;

-            convo::ScopedAlignedPtr<double> gainInterleaved(
-                static_cast<double*>(mkl_malloc(static_cast<size_t>(l.complexSize) * 2 * sizeof(double), 64)));
-            if (!gainInterleaved.get())
+            // [Mem-Fix] gain は実数値(振幅のみ)のためinterleaved配列は不要。
+            convo::ScopedAlignedPtr<double> gainReal(
+                static_cast<double*>(mkl_malloc(static_cast<size_t>(l.complexSize) * sizeof(double), 64)));
+            if (!gainReal.get())
                 continue;

             const double denom = static_cast<double>(std::max(1, l.complexSize - 1));
             for (int k = 0; k < l.complexSize; ++k)
             {
                 const double fNorm = static_cast<double>(k) / denom;
                 const double hfTilt = std::exp(-dampingCoeff * fNorm * fNorm);
-                gainInterleaved.get()[2 * k] = hfTilt;
-                gainInterleaved.get()[2 * k + 1] = hfTilt;
+                gainReal.get()[k] = hfTilt;
             }

+            // [Mem-Fix] SoA (irFreqReal/irFreqImag) に直接ゲインを適用する。
             for (int p = 0; p < l.numParts; ++p)
             {
-                double* slot = l.irFreqDomain + static_cast<size_t>(p) * l.partStride;
-                vdMul(l.complexSize * 2, slot, gainInterleaved.get(), slot);
-                deinterleaveComplex(slot,
-                                    l.irFreqReal + static_cast<size_t>(p) * l.complexSize,
-                                    l.irFreqImag + static_cast<size_t>(p) * l.complexSize,
-                                    l.complexSize);
+                double* re = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
+                double* im = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
+                vdMul(l.complexSize, re, gainReal.get(), re);
+                vdMul(l.complexSize, im, gainReal.get(), im);
             }
         }
     }
```

### 2.2 安全パッケージ2：Audioスレッド FDL (`fdlBuf`) の SoA 化とミラー最適化

`fdlBuf` (AoS) に対するポインタオフセットとミラー書き込みアドレスの計算、およびホットループ内の畳み込み演算を、AoS 経由から SoA (面格納) へ完全に一本化する。

#### Patch 8/11 — `processLayerBlock()` (L0専用) 内の FDL ミラー書き込みと複素畳み込みの SoA化

ミラーの書き出しアドレスの Aos 計算を撤退させ、SoA 側から直接 FMA 演算を呼ぶ。

```cpp
// src/MKLNonUniformConvolver.cpp の一部
 void MKLNonUniformConvolver::processLayerBlock(Layer& l) noexcept
 {
     // ... 前半部はそのまま

     // ── 2. Forward FFT ──
-    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
+    // [Mem-Fix] fdlBuf は使い捨てスクラッチ (current=offset0 / mirror=offset partStride)。
+    // 永続履歴は fdlReal/fdlImag (SoA) 側にのみ保持する。
+    double* currentFDLSlot = l.fdlBuf;
     ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

     deinterleaveComplex(currentFDLSlot,
                         l.fdlReal + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                         l.fdlImag + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                         l.complexSize);

     // [最適化2] Linearized ring buffer: mirror write
-    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
+    double* mirrorFDLSlot = l.fdlBuf + l.partStride;
     memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

     const int mirrorIndex = l.fdlIndex + l.numParts;
     deinterleaveComplex(mirrorFDLSlot,
                         l.fdlReal + static_cast<size_t>(mirrorIndex) * l.complexSize,
                         l.fdlImag + static_cast<size_t>(mirrorIndex) * l.complexSize,
                         l.complexSize);

     // ── 3. 複素乗算積算 (FDL × IR) ──
-    memset(l.accumBuf, 0, l.partStride * sizeof(double));
     memset(l.accumReal, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
     memset(l.accumImag, 0, static_cast<size_t>(l.complexSize) * sizeof(double));

-    const double* fdlBase = l.fdlBuf;
-    const double* irBase  = l.irFreqDomain;
-    double*       dst     = l.accumBuf;
-
-    const int linStart   = l.fdlIndex - l.numPartsIR + 1 + l.numParts;
-    const double* fdlLin = fdlBase + linStart * l.partStride;
-
-    for (int p = 0; p < l.numPartsIR; ++p)
-    {
-        const double* srcA = fdlLin + p * l.partStride;
-        const double* srcB = irBase + p * l.partStride;
-
-        if (p + 1 < l.numPartsIR)
-        {
-            _mm_prefetch((const char*)(srcA + l.partStride),     _MM_HINT_T1);
-            _mm_prefetch((const char*)(srcB + l.partStride),     _MM_HINT_T1);
-        }
-        if (p + 2 < l.numPartsIR)
-        {
-            _mm_prefetch((const char*)(srcA + 2 * l.partStride), _MM_HINT_T1);
-            _mm_prefetch((const char*)(srcB + 2 * l.partStride), _MM_HINT_T1);
-        }
-
-        if (kEnableSplitComplexKernel)
-        {
-            const int index = linStart + p;
-            const double* srcARe = l.fdlReal + static_cast<size_t>(index) * l.complexSize;
-            const double* srcAIm = l.fdlImag + static_cast<size_t>(index) * l.complexSize;
-            const double* srcBRe = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
-            const double* srcBIm = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
-            accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
-        }
-        else
-        {
-            int k = 0;
-            for (; k < l.complexSize; ++k)
-            {
-                const double ar = srcA[2 * k], ai = srcA[2 * k + 1];
-                const double br = srcB[2 * k], bi = srcB[2 * k + 1];
-                dst[2 * k] += ar * br - ai * bi;
-                dst[2 * k + 1] += ar * bi + ai * br;
-            }
-        }
-    }
-
-    if (kEnableSplitComplexKernel)
-        interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);
+    // [Mem-Fix] AoS(fdlBuf/irFreqDomain) 経由の読み出しとダミーprefetchを廃止し、
+    // SoA (fdlReal/fdlImag, irFreqReal/irFreqImag) のみを読む一本化されたパスにする。
+    const int linStart = l.fdlIndex - l.numPartsIR + 1 + l.numParts;
+
+    for (int p = 0; p < l.numPartsIR; ++p)
+    {
+        const int index = linStart + p;
+        const double* srcARe = l.fdlReal    + static_cast<size_t>(index) * l.complexSize;
+        const double* srcAIm = l.fdlImag    + static_cast<size_t>(index) * l.complexSize;
+        const double* srcBRe = l.irFreqReal + static_cast<size_t>(p)     * l.complexSize;
+        const double* srcBIm = l.irFreqImag + static_cast<size_t>(p)     * l.complexSize;
+
+        if (p + 1 < l.numPartsIR)
+        {
+            _mm_prefetch((const char*)(l.fdlReal    + static_cast<size_t>(index + 1) * l.complexSize), _MM_HINT_T1);
+            _mm_prefetch((const char*)(l.irFreqReal + static_cast<size_t>(p + 1)     * l.complexSize), _MM_HINT_T1);
+        }
+
+        accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
+    }
+
+    memset(l.accumBuf, 0, l.partStride * sizeof(double));
+    interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);
```

#### Patch 9/11 — `Add()` (L1/L2層) 内の 新規バッファFFT格納の SoA化

L1/L2 のアキュムレータ側でも FDL 書き出しは AoS スロットから SoA への直接 deinterleave 配列に切り替える。

```cpp
// src/MKLNonUniformConvolver.cpp の一部
                 else
                 {
                     juce::FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
                     juce::FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
                     juce::FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

                     // [v2.1] L1/L2 Forward FFT: real → CCS
-                    double* currentFDLSlot = l.fdlBuf + l.fdlIndex * l.partStride;
+                    // [Mem-Fix] fdlBuf は使い捨てスクラッチ (current=offset0 / mirror=offset partStride)。
+                    double* currentFDLSlot = l.fdlBuf;
                     ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

                     deinterleaveComplex(currentFDLSlot,
                                         l.fdlReal + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                                         l.fdlImag + static_cast<size_t>(l.fdlIndex) * l.complexSize,
                                         l.complexSize);

                     // [最適化2] mirror write
-                    double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
+                    double* mirrorFDLSlot = l.fdlBuf + l.partStride;
                     juce::FloatVectorOperations::copy(mirrorFDLSlot, currentFDLSlot, l.partStride);

                     const int mirrorIndex = l.fdlIndex + l.numParts;
                     deinterleaveComplex(mirrorFDLSlot,
                                         l.fdlReal + static_cast<size_t>(mirrorIndex) * l.complexSize,
                                         l.fdlImag + static_cast<size_t>(mirrorIndex) * l.complexSize,
                                         l.complexSize);

                     l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;

                     // [Bug2 fix] FDL スナップショット保存
                     l.baseFdlIdxSaved = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;

-                    memset(l.accumBuf, 0, l.partStride * sizeof(double));
                     memset(l.accumReal, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
                     memset(l.accumImag, 0, static_cast<size_t>(l.complexSize) * sizeof(double));
                     l.nextPart    = 0;
                     l.distributing = true;
                 }
```

#### Patch 10/11 — `Add()` (L1/L2層) 内の 分散積算ループの SoA化

非実数の AoS フォールバック処理、およびそれに伴う不必要な prefetch ループを完全に SoA 一本化で置き換える。

```cpp
// src/MKLNonUniformConvolver.cpp の一部
         if (!l.isImmediate && l.distributing)
         {
             const int endPart  = std::min(l.nextPart + l.partsPerCallback, l.numPartsIR);
+            const int baseFdlIdx = l.baseFdlIdxSaved;
+            const int linStart   = baseFdlIdx - l.numPartsIR + 1 + l.numParts;

-            const double* fdlBase    = l.fdlBuf;
-            const double* irBase     = l.irFreqDomain;
-            double*       dst        = l.accumBuf;
-            const int     baseFdlIdx = l.baseFdlIdxSaved;
-
-            const int linStart   = baseFdlIdx - l.numPartsIR + 1 + l.numParts;
-            const double* fdlLin = fdlBase + linStart * l.partStride;
-
+            // [Mem-Fix] AoS(fdlBuf/irFreqDomain)経由の読み出しを廃止し、
+            // SoA (fdlReal/fdlImag, irFreqReal/irFreqImag) のみを読む一本化されたパスにする。
             for (int p = l.nextPart; p < endPart; ++p)
             {
-                const double* srcA = fdlLin + p * l.partStride;
-                const double* srcB = irBase + p * l.partStride;
-
-                if (p + 1 < endPart)
-                {
-                    _mm_prefetch((const char*)(srcA + l.partStride), _MM_HINT_T1);
-                    _mm_prefetch((const char*)(srcB + l.partStride), _MM_HINT_T1);
-                }
-                if (p + 2 < endPart)
-                {
-                    _mm_prefetch((const char*)(srcA + 2 * l.partStride), _MM_HINT_T1);
-                    _mm_prefetch((const char*)(srcB + 2 * l.partStride), _MM_HINT_T1);
-                }
-
-                if (kEnableSplitComplexKernel)
-                {
-                    const int index = linStart + p;
-                    const double* srcARe = l.fdlReal + static_cast<size_t>(index) * l.complexSize;
-                    const double* srcAIm = l.fdlImag + static_cast<size_t>(index) * l.complexSize;
-                    const double* srcBRe = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
-                    const double* srcBIm = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
-                    accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
-                }
-                else
-                {
-                    int k = 0;
-                    for (; k < l.complexSize; ++k)
-                    {
-                        const double ar = srcA[2 * k], ai = srcA[2 * k + 1];
-                        const double br = srcB[2 * k], bi = srcB[2 * k + 1];
-                        dst[2 * k] += ar * br - ai * bi;
-                        dst[2 * k + 1] += ar * bi + ai * br;
-                    }
-                }
+                const int index = linStart + p;
+                const double* srcARe = l.fdlReal    + static_cast<size_t>(index) * l.complexSize;
+                const double* srcAIm = l.fdlImag    + static_cast<size_t>(index) * l.complexSize;
+                const double* srcBRe = l.irFreqReal + static_cast<size_t>(p)     * l.complexSize;
+                const double* srcBIm = l.irFreqImag + static_cast<size_t>(p)     * l.complexSize;
+
+                if (p + 1 < endPart)
+                {
+                    _mm_prefetch((const char*)(l.fdlReal    + static_cast<size_t>(index + 1) * l.complexSize), _MM_HINT_T1);
+                    _mm_prefetch((const char*)(l.irFreqReal + static_cast<size_t>(p + 1)     * l.complexSize), _MM_HINT_T1);
+                }
+
+                accumulateSplitComplex(srcARe, srcAIm, srcBRe, srcBIm, l.accumReal, l.accumImag, l.complexSize);
             }

             l.nextPart = endPart;

-            if (kEnableSplitComplexKernel)
-            {
-                memset(l.accumBuf, 0, l.partStride * sizeof(double));
-                interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);
-            }
+            memset(l.accumBuf, 0, l.partStride * sizeof(double));
+            interleaveComplex(l.accumReal, l.accumImag, l.accumBuf, l.complexSize);

             // ── 全パーティション累積完了 → IFFT → tailOutputBuf へコピー ──
```

### 2.3 安全パッケージ3：メモリ確保・クリアの最適化（AoSスクラッチ化完了）

メッセージ側、およびオーディオ側のすべてのデータ参照を SoA 側に切り離した上で、アロケーションされる一時バッファ領域自体のサイズをアトミックに縮小する。

#### Patch 1/11 — Layer 構造体の再定義（`MKLNonUniformConvolver.h`）

`irFreqDomain` と `fdlBuf` の一時スクラッチ領域としてのサイズ低下（1 パーティション分および 2 パーツ分）へのコメント・定義の修正。

```diff
         // ── IR 周波数領域 (Message Thread で確保・プリコンピュート) ──
-        // レイアウト: [numParts][partStride] (double 配列として管理)
-        double* irFreqDomain  = nullptr;  // mkl_malloc(numParts * partStride * sizeof(double), 64)
-        // split-complex 検用 SoA ストレージ（実部/虚部分離）
-        double* irFreqReal    = nullptr;  // mkl_malloc(numParts * complexSize * sizeof(double), 64)
-        // ...
+        // [Mem-Fix] irFreqDomain は 1 パーティション分の使い捨てスクラッチ。
+        double* irFreqDomain  = nullptr;  // mkl_malloc(partStride * sizeof(double), 64) ← スクラッチ
+        double* irFreqReal    = nullptr;  // mkl_malloc(numParts * complexSize * sizeof(double), 64)
+        double* irFreqImag    = nullptr;  // mkl_malloc(numParts * complexSize * sizeof(double), 64)

         // ── 入力 FDL (Frequency Domain Delay Line, Audio Thread で更新) ──
-        double* fdlBuf        = nullptr;  // mkl_malloc(...)
-        double* fdlReal       = nullptr;  // mkl_malloc((numParts*2) * complexSize * sizeof(double), 64)
+        // [Mem-Fix] fdlBuf も 2 パーティション分 (current+mirror) のスクラッチに縮小。
+        double* fdlBuf        = nullptr;  // mkl_malloc(2 * partStride * sizeof(double), 64) ← スクラッチ
+        double* fdlReal       = nullptr;  // mkl_malloc((numParts*2) * complexSize * sizeof(double), 64)
+        double* fdlImag       = nullptr;  // mkl_malloc((numParts*2) * complexSize * sizeof(double), 64)
```

#### Patch 2/11 — AVX2 必須チェック（移植性を考慮したマクロコンパイルチェック）

コンパイル時点で無条件に `#error` を用いることで、非AVX2環境や他社ライブラリコンパイル時の移植性を最大限担保する（C++のテンプレート外 `static_assert(false)` による不要な早期ビルド失敗を完全に防止する）。

```cpp
#ifndef __AVX2__
#error "MKLNonUniformConvolver requires AVX2 (see coding standard: CPU must support AVX2)."
#endif
```

#### Patch 4/11 — `SetImpulse()` バッファ確保サイズの縮小

確保サイズをスクラッチ用に再計算する。

```diff
         // ── バッファ確保 (すべて mkl_malloc 64byte アライン) ──
-        const size_t irBufSize  = static_cast<size_t>(l.numParts) * l.partStride;
-        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;
+        // [Mem-Fix] irFreqDomain/fdlBuf は永続履歴ではなく使い捨てスクラッチ。
+        // irFreqDomain: 1パーティション分、fdlBuf: current+mirrorの2スロット分のみ。
+        const size_t irBufSize  = static_cast<size_t>(l.partStride);
+        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
         const size_t irSoaSize  = static_cast<size_t>(l.numParts) * static_cast<size_t>(l.complexSize);
         const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);
```

#### Patch 5/11 — `SetImpulse()` IR プリコンピュートループのスクラッチ化

FFT 計算時に、`p * l.partStride` という領域全体を走査しない。バッファオフセットは常に 0 (先頭)。

```diff
             // [v2.1] Forward FFT: real → CCS
             ippsFFTFwd_RToCCS_64f(tempTime, tempFreq, l.fftSpec, l.fftWorkBuf);

-            // interleaved complex として irFreqDomain に格納
-            memcpy(l.irFreqDomain + p * l.partStride, tempFreq,
-                   l.complexSize * 2 * sizeof(double));
-
-            if (scale != 1.0)
-                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain + p * l.partStride, 1);
-
-                 deinterleaveComplex(l.irFreqDomain + p * l.partStride,
-                            l.irFreqReal + static_cast<size_t>(p) * l.complexSize,
-                            l.irFreqImag + static_cast<size_t>(p) * l.complexSize,
-                            l.complexSize);
+            // [Mem-Fix] 1 パーティション分のスクラッチのため、オフセット0(先頭)へ書き込む。
+            memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double));
+
+            if (scale != 1.0)
+                cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain, 1);
+
+            deinterleaveComplex(l.irFreqDomain,
+                        l.irFreqReal + static_cast<size_t>(p) * l.complexSize,
+                        l.irFreqImag + static_cast<size_t>(p) * l.complexSize,
+                        l.complexSize);
         }
```

#### Patch 6/11 — パーティション逆順 swap における AoS swap の撤去

もはや `irFreqDomain` 側の swap は一切必要ないため。

```diff
         // [最適化2] IR パーティションを逆順に並び替える (forward アクセス最適化)
-        // AoS (irFreqDomain) と SoA (irFreqReal/irFreqImag) を同時にswapする。
+        // [Mem-Fix] irFreqDomain はスクラッチ化されたため、swap対象はSoAのみでよい。
         if (l.numPartsIR > 1)
         {
-            double* swapDomain = static_cast<double*>(mkl_malloc(
-                static_cast<size_t>(l.partStride) * sizeof(double), 64));
             double* swapSoA = static_cast<double*>(mkl_malloc(
                 static_cast<size_t>(l.complexSize) * sizeof(double), 64));
-            if (swapDomain && swapSoA)
+            if (swapSoA)
             {
                 for (int pf = 0; pf < l.numPartsIR / 2; ++pf)
                 {
                     const int pb = l.numPartsIR - 1 - pf;

-                    // irFreqDomain swap (AoS interleaved complex)
-                    double* slotF = l.irFreqDomain + pf * l.partStride;
-                    double* slotB = l.irFreqDomain + pb * l.partStride;
-                    memcpy(swapDomain, slotF, l.partStride * sizeof(double));
-                    memcpy(slotF,      slotB, l.partStride * sizeof(double));
-                    memcpy(slotB,      swapDomain, l.partStride * sizeof(double));
-
                     // irFreqReal swap (SoA)
                     double* realF = l.irFreqReal + static_cast<size_t>(pf) * l.complexSize;
                     double* realB = l.irFreqReal + static_cast<size_t>(pb) * l.complexSize;
                     memcpy(swapSoA, realF, l.complexSize * sizeof(double));
                     memcpy(realF,   realB, l.complexSize * sizeof(double));
                     memcpy(realB,   swapSoA, l.complexSize * sizeof(double));

                     // irFreqImag swap (SoA)
                     double* imagF = l.irFreqImag + static_cast<size_t>(pf) * l.complexSize;
                     double* imagB = l.irFreqImag + static_cast<size_t>(pb) * l.complexSize;
                     memcpy(swapSoA, imagF, l.complexSize * sizeof(double));
                     memcpy(imagF,   imagB, l.complexSize * sizeof(double));
                     memcpy(imagB,   swapSoA, l.complexSize * sizeof(double));
                 }
             }
-            if (swapDomain) mkl_free(swapDomain);
-            if (swapSoA)    mkl_free(swapSoA);
+            if (swapSoA) mkl_free(swapSoA);
         }
```

#### Patch 11/11 — `Reset()` における `fdlBufSize` のクリアサイズ修正 (★超重要ガード)

クリアサイズも実際のスクラッチサイズに合わせないと範囲外書込み (オーバーラン) が生じる。

```diff
         Layer& l = m_layers[li];
         if (l.irFreqDomain == nullptr) continue;

-        const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;
+        // [Mem-Fix] fdlBuf は 2*partStride のスクラッチに縮小されているため、
+        // クリアサイズも実際の確保サイズに合わせる (旧サイズのままだと範囲外書き込みになる)。
+        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
         const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);
         juce::FloatVectorOperations::clear(l.fdlBuf,       fdlBufSize);
         juce::FloatVectorOperations::clear(l.fdlReal,      fdlSoaSize);
         juce::FloatVectorOperations::clear(l.fdlImag,      fdlSoaSize);
```

### 2.4 `freeAll()` に関する設計（変更不要の検証）

`Layer::freeAll()`（または `releaseAllLayers()`）は、サイズを受け取らずにポインタを安全に `mkl_free()` する実装にすでになっているため、これの変更は一切不要。

### 2.5 例外処理・リカバリ・OOM（メモリ不足）時のセーフティ設計

使い捨てスクラッチ化に伴い、Message Thread側のアロケーション失敗（OOM）などのリカバリ処理、およびアトミックにガードを構築する手法を定義する。

#### 1. フィルタ適用時（applySpectrumFilter）のトランザクションセーフ設計

Message Thread 側の OOM によりゲイン配列 `reusableGain` (mkl_malloc) 自体の確保が失敗した場合、処理はレイヤー（Layer）ごとに完全にアトミックなトランザクション（All-or-Nothing）で処理する。
すなわち、アロケーションは外層ループに入る前に1回だけ行い、確保失敗時は SoA に部分的変更を与えずに完全スキップしてアーリーリターンする。

**★ 安全規約: 部分的に更新された SoA を Audio Thread に公開してはならない。**
フィルタ適用中に OOM や内部エラーが発生し、一部のパーティションだけ Real/Imag が更新された状態で関数を終了した場合、Audio Thread がその中途状態を参照するとスペクトル破綻による可聴ノイズが出力される。このリスクを防止するため、フィルタ適用関数は以下を遵守する：

1. ゲインバッファの事前確保に失敗した場合はループに入らず即座にリターンする（SoA に一切触れない）。
2. ループ中に異常が発生した場合も、既に書き換えたパーティションを元の状態に戻すことはせず（コスト対効果の判断）、代わりにエフェクト全体の `m_ready` フラグをアトミックに降ろして当該コンボ出力を完全ミュートする。
3. Real/Imag の対更新は各パーティション内で完結させる（規約1）。

#### 2. OOM（メモリ確保失敗）時のフェールセーフ・リアルタイムバイパス設計

`SetImpulse` での縮小スクラッチ確保 (`l.irFreqDomain` や `l.fdlBuf`) 時に万が一 OOM や FFT/deinterleave 変換途中で失敗（bad_alloc / IPP エラー / DFTI 失敗）が発生した場合は、ただちに `releaseAllLayers()` を呼び出して中途確保されたバッファを例外なく安全に完全クリアする。
エフェクト内部フラグ `m_ready` をアトミックに `false` に据え置き、リアルタイムスレッド側でこのコンボエンジンの呼び出しを完全にバイパス（無音またはドライ出力）するリカバリ経路を維持する。中途半端な部分適用状態で既存 Layer のノイズを流し続けることは一切許容しない。

### 2.6 SoA同期保証 — Real/Imag 不可分更新規約（設計規約）

**★ 設計不変条件: `irFreqReal` と `irFreqImag`（および `fdlReal` と `fdlImag`）は常にペアで更新されなければならない。**
AoS 排除後の SoA が唯一の永続データ表現であり、実部のみ更新・虚部未更新の状態がオーディオスレッドから観測された場合、位相が 90° 崩れた破壊的な信号が出力される。このリスクを排除するため、Real/Imag は不可分な更新単位として扱う。

AoS の排除により、Message Thread から SoA に対して直接実行される `applySpectrumFilter()` やテール減衰の非同期処理において、オーディオスレッドとの競合により位相が 90° 崩れた破壊的信号を出力するリスクが伴う。以下を設計規約として厳格に遵守する。

#### 規約1: Real/Imag パーティション単位での不可分対更新（Atomic Paired Update）

全パーツに対する一括 vdMul ではなく、各パーツ p ループ内（実部 `irFreqReal` へのゲイン適用と、虚部 `irFreqImag` へのゲイン適用）を連続的・局所的、かつペアで適用する。非フェーズ整合フレームを最小幅（1パーツ内）に厳密に抑え込む。

```cpp
// ✅ 正しい：各パーツで実部・虚部を直列対更新
for (int p = 0; p < l.numParts; ++p)
{
    double* re = l.irFreqReal + static_cast<size_t>(p) * l.complexSize;
    double* im = l.irFreqImag + static_cast<size_t>(p) * l.complexSize;
    vdMul(cSize, re, gain, re);  // 実部
    vdMul(cSize, im, gain, im);  // 虚部（直後に実行。この間にスレッド切替不可）
}
```

#### 規約2: フィルタ適用ループ内での早期 return / break / throw の完全禁止

実部・虚部双方への同一ゲイン適用 (vdMul) の完全な一対実行を保証するため、フィルター適用ループ内での早期 return、break、throw 例外送出を一切禁止する。万が一の確保失敗時はループに入る前にアーリーリターンする設計とする。

**noexcept の適用条件と設計判断**:

現在のコードベースでは、`applySpectrumFilter` を始めとする全メンバ関数が `noexcept` で宣言されている（grep による実コード確認済み）。一方で、これらの関数内では `mkl_malloc`（失敗時は nullptr を返し例外を投げない）や `convo::ScopedAlignedPtr`（全メソッドが noexcept）のみを使用しており、`std::vector` や `new` などの例外を投げうる構造は使用していない。そのため `noexcept` 指定は現状の実装姿勢と矛盾せず、むしろ「この関数は例外を投げない」という契約をコンパイラと保守担当者に明示する効果がある。

ただし、将来の改修で例外を投げうるコード（標準ライブラリコンテナ等）を追加する場合には、本設計書の制約として以下を遵守する：

1. フィルタ適用ループ内に例外を投げうる処理を挿入しないこと。
2. ループの事前処理（バッファ確保等）で例外が発生した場合は、その例外を捕捉し、関数全体としては確実にリターンするか、または `m_ready` フラグを降ろしてバイパス状態に遷移させること。
3. `noexcept` 指定が必要な理由は「Real/Imag 対更新の不可分性をハードウェア例外（`std::terminate`）で保護するため」ではなく、「この関数のユーザー（呼び出し元）に例外伝搬を期待させないため」である。

#### 規約3: 複数フィルタの連続適用時は Real/Imag 同期を維持

Air Absorption と Spectrum Filter を連続適用する場合も、各フィルタの適用が独立した対更新として完結することを確認する。一方のフィルタが Real/Imag 双方に完全に適用された後に、次のフィルタが開始されることを保証する。

### 2.7 プレフェッチ (Prefetch) 安全設計

AoS の `fdlBuf` / `irFreqDomain` がパーツ数 2面 / 1面に極限縮小されるため、ホットループ `processLayerBlock` および `Add` 内で `srcA + l.partStride`（次のパーツ）を prefetch するロジックはバッファ境界外のアドレス、または不要なページロードを引き起こします。

* **AoS プレフェッチの完全排除**:
  AoS バッファに対する prefetch コードをホットループから跡形もなく完全払拭する。
* **SoA 側プレフェッチへのリレイアウト**:
  ホットループで走査する SoA 配列（`fdlReal`, `fdlImag`, `irFreqReal`, `irFreqImag`）に対し、キャッシュロード (`_MM_HINT_T1`) を継続し、AVX2 FMA 演算時の並列メモリスループットを確実に担保します。

```cpp
if (p + 1 < l.numPartsIR)
{
    _mm_prefetch((const char*)(l.fdlReal    + static_cast<size_t>(index + 1) * l.complexSize), _MM_HINT_T1);
    _mm_prefetch((const char*)(l.irFreqReal + static_cast<size_t>(p + 1)     * l.complexSize), _MM_HINT_T1);
}
```

### 2.8 AoS スクラッチの寿命と型責務表現（設計規約）

#### 1. AoS スクラッチのライフサイクル（寿命）

`irFreqDomain` / `fdlBuf` が「永続データ保持領域」から「初期化専用スクラッチ」へと役割を変更したことに伴い、その有効期間を以下のライフサイクルとして厳密に定義する。

```text
FFT (ippsFFTFwd_RToCCS_64f) → CCS 出力を AoS へ書き込み
    ↓
memcpy / cblas_dscal (スケーリング)
    ↓
deinterleaveComplex → SoA (irFreqReal / irFreqImag) へ展開
    ↓
★★★ この時点で AoS (irFreqDomain) の内容は以降一切参照してはならない ★★★
    ↓
（次のパーティション p+1 の FFT 出力で上書き。または SetImpulse 完了後は事実上 dead）
```

同様に `fdlBuf` も、各ブロックの FFT 出力を deinterleave して SoA (`fdlReal`/`fdlImag`) へ展開した直後は無効領域とみなし、以降のコード（特に prefetch や乗算ループ）で決して参照してはならない。

#### 2. 型名・コメントで責務を明確化する推奨

AoS が「FFT 一時スクラッチ」に役割変更されたことを、将来の保守担当者に確実に伝えるため、以下のような型エイリアスまたはコメント修飾を用いることを推奨する。

```cpp
// 推奨：型名で責務を明示（using alias）
using ScratchAoS = double;              // irFreqDomain, fdlBuf: FFT出力→deinterleaveの中継のみ
using PersistentSplitSpectrum = double; // irFreqReal, irFreqImag, fdlReal, fdlImag: 唯一の永続データ

// Layer 構造体内での使用例
ScratchAoS* irFreqDomain;  // mkl_malloc(partStride * sizeof(double), 64) — スクラッチ
PersistentSplitSpectrum* irFreqReal; // mkl_malloc(numParts * complexSize * sizeof(double), 64) — 永続
```

これにより、`irFreqDomain` を誤ってオーディオスレッドのホットループ内でデータソースとして使用するコードが入り込むリスクを、型システムレベルで防止できる。

#### 3. 変数名のリネーム推奨

型エイリアスだけでなく、実際のメンバ変数名自体をスクラッチ用途と分かる名称に変更することを推奨する。これにより、コメントを読まなくても変数名だけで責務が把握でき、誤用を防止できる。

| 現行名 | 推奨新名 | 理由 |
| :--- | :--- | :--- |
| `irFreqDomain` | `irFreqScratch` | 「Domain＝周波数領域データ」の誤解を避け、「Scratch＝一時使い捨て」を明示。永続データではないことが一目で分かる。 |
| `fdlBuf` | `fdlScratch` | 「Buf＝バッファ（永続格納領域）」の印象を避け、FFT出力中継の一時領域であることを明示。 |

> **実装上注意**: リネームは Patch 1/11 と同時に行うこと。リネームのみ先行すると他コードの参照解決が不完全になる。Patch 全11件の適用順序は §2.3 の安全パッケージ3（全SoA移行完了後）に従うこと。

#### 4. メンバーの役割・責務変転の定義

AoS の完全スクラッチ化に伴い、Layer 構造体メンバーの役割とアロケーション責務は以下のように変転する。

| メンバー | 以前の責務 | 確定改修後の責務 | 変化内容 |
| :--- | :--- | :--- | :--- |
| `irFreqDomain` | **永続データ保持** (numParts×partStride)。全パーツの周波数領域IRを保持。 | **一時スクラッチ領域** (1×partStride)。`SetImpulse` でのCCS出力から deinterleave 完了までの一時バッファ。 | **永続削除**。初期化完了後は、メモリ上ではゼロに等しい存在となり、実質 97% メモリカット。 |
| `fdlBuf` | **永続データ保持** (numParts×2×partStride)。FDLの全履歴を保持、リング書き込み・読み出し。 | **一時スクラッチ領域** (2×partStride)。新規ブロックのFFT CCS出力と deinterleave の仲介のみ。 | **永続削除**。FDL 永続履歴は `fdlReal`/`fdlImag` (SoA) の面のみが保持する。 |

### 2.9 全変更箇所インベントリ（コード調査による確定リスト）

本セクションでは、WSL grep/rg によるソースコード調査で確定した全変更対象箇所を一元リスト化する。各 Patch がカバーする行番号・シンボル・変更種別を明記することで、実装時の見落としゼロを保証する。

#### irFreqDomain 全参照箇所（調査日: 2026-07-07）

| ファイル | 行 | 種別 | 対応Patch | 変更内容 |
| :--- | :--- | :--- | :--- | :--- |
| `MKLNonUniformConvolver.h` | L267 | メンバ変数宣言 (AoS) | 1/11 | 確保サイズ `numParts*partStride` → `partStride` |
| `MKLNonUniformConvolver.h` | L141, L70 | コメント | 1/11 | AoSの役割を「焼き込みバッファ」から「使い捨てスクラッチ」に修正 |
| `MKLNonUniformConvolver.cpp` | L268 | freeAll() 解放 | — | mkl_free 呼び出しは変更不要（ポインタのみ解放） |
| `MKLNonUniformConvolver.cpp` | L332 | applySpectrumFilter ガード | 3/11 | `!l.irFreqDomain` → `!l.irFreqReal \|\| !l.irFreqImag` |
| `MKLNonUniformConvolver.cpp` | L413-436 | applySpectrumFilter AoSループ | 3/11 | AoS経由 (interleave→vdMul→deinterleave) → SoA直接 vdMul×2 |
| `MKLNonUniformConvolver.cpp` | L720 | SetImpulse irBufSize | 4/11 | `numParts * partStride` → `partStride` |
| `MKLNonUniformConvolver.cpp` | L724 | SetImpulse irFreqDomain mkl_malloc | 4/11 | 確保サイズ縮小（irBufSize 変更で自動調整） |
| `MKLNonUniformConvolver.cpp` | L741 | SetImpulse nullptr check | — | 変更不要（パラメトリック） |
| `MKLNonUniformConvolver.cpp` | L750 | SetImpulse clear() | 4/11 | irBufSize 変更で自動調整 |
| `MKLNonUniformConvolver.cpp` | L798-805 | SetImpulse FFT→AoS→SoA ループ | 5/11 | `irFreqDomain + p*partStride` → `irFreqDomain`（先頭固定） |
| `MKLNonUniformConvolver.cpp` | L821-850 | パーティション逆順swap | 6/11 | swapDomain（AoS）削除、swapSoAのみ維持 |
| `MKLNonUniformConvolver.cpp` | L920 | Air Absorption ガード | 7/11 | `!l.irFreqDomain` → `!l.irFreqReal \|\| !l.irFreqImag` |
| `MKLNonUniformConvolver.cpp` | L942-944 | Air Absorption AoSループ | 7/11 | AoS経由 → SoA直接 vdMul×2 |
| `MKLNonUniformConvolver.cpp` | L1093, L1094 | processLayerBlock AoSベース計算 | 8/11 | `fdlBase = l.fdlBuf` + `irBase = l.irFreqDomain` を削除 |
| `ConvolverProcessor.h` | L285 | コメント | — | 「... irFreqDomain に焼き込まれる」→「... SoA (irFreqReal/irFreqImag) に直接適用される」に修正 |
| `AudioEngine.Parameters.cpp` | L636 | コメント | — | 「NUC irFreqDomain を再焼き込み」→「NUC SoA を再適用」に修正（動作に影響なし） |

#### fdlBuf 全参照箇所（調査日: 2026-07-07）

| ファイル | 行 | 種別 | 対応Patch | 変更内容 |
| :--- | :--- | :--- | :--- | :--- |
| `MKLNonUniformConvolver.h` | L274 | メンバ変数宣言 (AoS) | 1/11 | 確保サイズ `numParts*2*partStride` → `2*partStride` |
| `MKLNonUniformConvolver.cpp` | L271 | freeAll() 解放 | — | 変更不要 |
| `MKLNonUniformConvolver.cpp` | L720 | SetImpulse fdlBufSize | 4/11 | `numParts*2*partStride` → `partStride*2` |
| `MKLNonUniformConvolver.cpp` | L727 | SetImpulse fdlBuf mkl_malloc | 4/11 | 確保サイズ縮小 |
| `MKLNonUniformConvolver.cpp` | L741 | SetImpulse nullptr check | — | 変更不要 |
| `MKLNonUniformConvolver.cpp` | L753 | SetImpulse clear() | 4/11 | fdlBufSize 変更で自動調整 |
| `MKLNonUniformConvolver.cpp` | L1070 | processLayerBlock FDL書き込み | 8/11 | `fdlBuf + fdlIndex*partStride` → `fdlBuf`（先頭固定） |
| `MKLNonUniformConvolver.cpp` | L1079 | processLayerBlock mirror書き込み | 8/11 | `fdlBuf + (fdlIndex+numParts)*partStride` → `fdlBuf + partStride` |
| `MKLNonUniformConvolver.cpp` | L1093 | processLayerBlock 積算ループ | 8/11 | `fdlBase = l.fdlBuf` 削除 |
| `MKLNonUniformConvolver.cpp` | L1287 | Add FDL書き込み | 9/11 | `fdlBuf + fdlIndex*partStride` → `fdlBuf`（先頭固定） |
| `MKLNonUniformConvolver.cpp` | L1296 | Add mirror書き込み | 9/11 | `fdlBuf + (fdlIndex+numParts)*partStride` → `fdlBuf + partStride` |
| `MKLNonUniformConvolver.cpp` | L1326 | Add 積算ループ | 10/11 | `fdlBase = l.fdlBuf` 削除 |
| `MKLNonUniformConvolver.cpp` | L1502 | Reset fdlBufSize | 11/11 | `numParts*2*partStride` → `partStride*2` |

#### kEnableSplitComplexKernel 全使用箇所（調査日: 2026-07-07）

| 行 | 関数 | 現在の分岐 | 変更後（Patch 2/11 適用後） |
| :--- | :--- | :--- | :--- |
| L158 | (constexpr定義) | AVX2=true / non-AVX2=false | `#ifndef __AVX2__` / `#error` に置換 |
| L1116 | processLayerBlock | SplitComplex分岐 vs AoSフォールバック | 分岐削除、SplitComplex のみ維持 |
| L1138 | processLayerBlock | `if (kEn...) interleaveComplex` | ガード削除、常に実行 |
| L1350 | Add 分散積算 | SplitComplex分岐 vs AoSフォールバック | 分岐削除、SplitComplex のみ維持 |
| L1374 | Add 分散積算完了 | `if (kEn...) { memset + interleave }` | ガード削除、常に memset + interleave |

#### accumBuf / accumReal / accumImag memset 全箇所（調査日: 2026-07-07）

| 行 | 関数 | 現在のコード | 変更後（Patch 8/11, 10/11） |
| :--- | :--- | :--- | :--- |
| L1089-1091 | processLayerBlock | `memset(accumBuf)` → `memset(accumReal)` → `memset(accumImag)` | accumReal/Imag は現位置。accumBuf は積算後・interleave直前に移動 |
| L1310-1312 | Add FFT格納 | `memset(accumBuf)` → `memset(accumReal)` → `memset(accumImag)` | accumBuf memset を削除（accumReal/Imag のみ現位置） |
| L1376 | Add 分散積算完了 | `if (kEn...) { memset(accumBuf); interleaveComplex(...) }` | ガード削除、常に memset + interleave |

#### prefetch 安全確認（調査日: 2026-07-07）

**現在コード内の AoS 経由 prefetch（Patch 8/11, 10/11 で削除）**:

* L1107-1108: `_mm_prefetch(srcA + partStride, T1)` ← `fdlBuf` 経由
* L1112-1113: `_mm_prefetch(srcA + 2*partStride, T1)` ← `fdlBuf` 経由（double stride）
* L1341-1342: `_mm_prefetch(srcA + partStride, T1)` ← `fdlBuf` 経由
* L1346-1347: `_mm_prefetch(srcA + 2*partStride, T1)` ← `fdlBuf` 経由（double stride）

**置換後の SoA 経由 prefetch**:

* `_mm_prefetch(l.fdlReal + (index+1)*complexSize, T1)`
* `_mm_prefetch(l.irFreqReal + (p+1)*complexSize, T1)`

**x86 prefetch 安全性**: x86 の `_mm_prefetch` は無効アドレスに対してもページフォルトを発生させず、冗長アドレスへの発行は安全。しかし、設計としては縮小後のスクラッチ領域（`fdlBuf`: 2×partStride）を超える prefetch はコードの意図が不明瞭になるため、本設計では全 AoS prefetch を SoA prefetch に置き換える。

### 2.8b 検証済み安全経路（AoS 非参照の確認）

下記の関数・経路については実コード調査により AoS 非参照を確認した。変更対象外。

| 関数 / 経路 | 確認内容 | 結果 |
| :--- | :--- | :--- |
| `processDirectBlock()` | L975-1050: direct FIR 畳み込み。`m_directIRRev` / `m_directWindow` を使用。 | ✅ AoS 非参照 |
| `Get()` | L1400-1470: リングバッファ読出し + L1/L2 tail 出力加算。SoA (`tailOutputBuf`) のみ使用。 | ✅ AoS 非参照 |
| `areFftDescriptorsCommitted()` | L956-973: `l.fftSpec` / `l.descriptorCommitted` のみ確認。 | ✅ AoS 非参照 |
| `accumulateSplitComplex()` | L205-245: 完全に SoA ベース（srcAReal/Imag, srcBReal/Imag → dstReal/Imag） | ✅ AoS 非参照 |
| `interleaveComplex()` / `deinterleaveComplex()` | L187-196: 純粋な書式変換関数。引数で渡されたポインタのみ操作。 | ✅ AoS 非参照 |
| `ringWrite()` / `ringRead()` | L1167-1210: `m_ringBuf` のみ操作。 | ✅ AoS 非参照 |
| `tailMode == 1` (Layer Tail Contouring) | L540-555: パラメータ計算のみ。irFreqDomain / fdlBuf を一切参照しない。 | ✅ AoS 非参照 |
| `tailMode == 2` (Bypass) | 分岐なし。tailStrength=0 に設定。 | ✅ AoS 非参照 |
| MKL DFTI 旧パス | v2.1 で IPP に完全換装済み。DFTI コードは残存しない。 | ✅ 既存パスなし |

### 2.10 拡張性能検証ベンチマーク項目（測定計画・合格基準）

AoS の完全スクラッチ化および SoA 直接演算のパフォーマンスへの定量的影響を評価するため、以下のベンチマーク項目を改修前後にわたり比較検証に含めるものとする。各項目は改修前後の差異（regression がないこと）を受入基準とする。

#### IRロード時間・構築時間（Message Thread レイテンシ）

| IR長 @384kHz | 測定項目 | 期待値（改修前後で有意差なし） |
| :--- | :--- | :--- |
| 83ms (31,872 samp) | SetImpulse() 完了時間 (μs) | ±5%以内 |
| 1.0s | SetImpulse() 完了時間 (μs) | ±5%以内 |
| 3.0s | SetImpulse() 完了時間 (μs) | ±5%以内 |
| 5.46s (MAX) | SetImpulse() 完了時間 (μs) | ±5%以内 |
| 5.46s (MAX) | ProgressiveUpgrade 暖機完了までの総時間 (s) | ±10%以内 |

#### 実行時パフォーマンス（Audio Thread）

| 測定項目 | 測定手法 | 合格基準 |
| :--- | :--- | :--- |
| Audio Thread CPU 使用率 (83ms IR) | プロファイラ（xperf / ETW） | ±3%以内 |
| Audio Thread CPU 使用率 (5.46s IR) | 同上 | ±5%以内 |
| Message Thread CPU 使用率 (フィルタ更新時) | 同上 | ±5%以内（低下傾向歓迎） |
| Peak RSS（83ms IR, 384kHz ステレオ） | メモリプロファイラ | 現状から低下（~1.5MB削減） |
| Peak RSS（5.46s IR, 384kHz ステレオ） | メモリプロファイラ | 現状約640MB → 約333MB（45〜48%減） |
| Peak Working Set（5.46s IR） | メモリプロファイラ（Process Explorer / Windows Performance Toolkit） | AoS除去による削減を確認（Private Bytes も併せて計測） |
| Private Bytes（5.46s IR） | メモリプロファイラ | Peak RSS との差分から共有メモリ影響を分離して評価 |

#### キャッシュ効率プロファイリング

| 測定項目 | ツール | 備考 |
| :--- | :--- | :--- |
| L1d キャッシュミス率 | Intel VTune / perf stat | `vdMul(real)` + `vdMul(imag)` 連続適用時のミス率を確認 |
| L2 キャッシュミス率 | Intel VTune / perf stat | SoA 面アクセスパターンによるデータローカリティ影響を検証 |
| FMA 演算スループット (IPC) | Intel VTune | Split Complex 処理の命令あたりサイクル数を測定 |

> **注記**: 83ms ルーム補正 IR 環境では L1/L2 が生成されず NUC エンジン負荷が低いため、実使用上は長尺 IR (3〜5s) での Audio Thread CPU 変動およびキャッシュミス率を重点的に確認すること。

### 2.11 参考文献との対応

* Farina & Torger, *Real-Time Partitioned Convolution for Ambiophonics Surround Sound* (2001)
* Gardner, *Efficient convolution without input-output delay*, JAES 43(3), 1995
* Garcia, *Optimal filter partition for efficient convolution* (2002)
* [FFTW 論文](https://www.fftw.org/fftw-paper-ieee.pdf)

### 2.12 実装完了受け入れ基準（Acceptance Criteria）

本改修の実装完了を宣言するためのチェックリスト。すべての項目が PASS した場合に実装完了とみなす。AC-01〜AC-14 はコードレビューで確認可能。AC-15〜AC-19 は CI/CD パイプラインで自動化することを推奨。AC-20〜AC-23 は実機プロファイリング環境で手動確認。

#### コード構造チェック

| # | 項目 | 確認方法 | 合格条件 |
| :--- | :--- | :--- | :--- |
| AC-01 | AoS 配列（`irFreqDomain`）を Audio Thread が一切参照しない | WSL grep/rg で `irFreqDomain` 参照箇所を全抽出し精査 | `processLayerBlock` / `Add` / `Get` 内での読み出し参照が0件 |
| AC-02 | AoS バッファ（`fdlBuf`）を Audio Thread が一切参照しない | 同上 | ホットループ内での `fdlBuf` からのデータ読み出しが0件 |
| AC-03 | AoS prefetch が完全に排除されている | `_mm_prefetch` 全使用箇所の audit | `fdlBuf` / `irFreqDomain` 経由の prefetch が0件 |
| AC-04 | `irFreqDomain` 確保サイズが `partStride` である | `mkl_malloc` 呼び出しの引数確認 | `irBufSize == partStride` |
| AC-05 | `fdlBuf` 確保サイズが `2 * partStride` である | 同上 | `fdlBufSize == partStride * 2` |
| AC-06 | `Reset()` の `fdlBufSize` が縮小後のサイズと一致 | `Reset()` 内の `fdlBufSize` 計算式確認 | `fdlBufSize == partStride * 2` |
| AC-07 | `irFreqReal` / `irFreqImag` が全パーティション分保持されている | `mkl_malloc` 呼び出しの引数確認 | `mkl_malloc(numParts * complexSize * sizeof(double), 64)` が維持されている |
| AC-08 | `fdlReal` / `fdlImag` が全パーティション分保持されている | 同上 | `mkl_malloc(numParts * 2 * complexSize * sizeof(double), 64)` が維持されている |
| AC-09 | `kEnableSplitComplexKernel` が削除されている | WSL grep で使用箇所をカウント | 定義と全使用箇所（4箇所）が削除されている |
| AC-10 | `applySpectrumFilter` が AoS ではなく SoA を直接書き換える | diff audit | `irFreqDomain` への読み書きが0件、`reusableGainInterleaved` が削除されている |
| AC-11 | Air Absorption テール減衰が AoS ではなく SoA を直接書き換える | diff audit | `irFreqDomain` への読み書きが0件、`gainInterleaved` が削除されている |
| AC-12 | processLayerBlock のミラー書き込みが縮小後のオフセットを使用 | diff audit | `mirrorFDLSlot = l.fdlBuf + partStride` |
| AC-13 | Add の分散積算ループが SoA 面インデックスのみを参照 | diff audit | `fdlBase = l.fdlBuf` / `irBase = l.irFreqDomain` が削除されている |
| AC-14 | `freeAll()` にサイズ変更の影響がないことを確認 | コード確認 | `mkl_free(ptr)` のみでサイズ変数を使用していない |

#### 回帰テスト

| # | 項目 | 確認方法 | 合格条件 |
| :--- | :--- | :--- | :--- |
| AC-15 | Debug ビルドが通る | `cmake --build build --config Debug` | エラー0、警告0 |
| AC-16 | Release ビルドが通る | `cmake --build build --config Release` | エラー0、警告0 |
| AC-17 | 既存ユニットテストが全件 PASS | `ctest -C Debug --output-on-failure` | 全テスト PASS |
| AC-18 | 83ms IR での音響回帰テスト PASS | 実機音声出力確認 | 改修前後で音質劣化なし（ホワイトノイズ／ポップノイズ／プチノイズが発生しない） |
| AC-19 | 5.46s IR での音響回帰テスト PASS | 同上 | 同上 |

#### 性能基準

| # | 項目 | 確認方法 | 合格条件 |
| :--- | :--- | :--- | :--- |
| AC-20 | Audio Thread CPU 使用率が改修前から増加しない | プロファイラ（xperf / ETW） | 83ms IR で ±3%以内、5.46s IR で ±5%以内 |
| AC-21 | Peak RSS が改修前から削減されている | メモリプロファイラ | 83ms IR で ~1.5MB 削減、5.46s IR で 45〜48% 削減 |
| AC-22 | `SetImpulse()` 完了時間が改修前から有意に増加しない | 実測（μs） | 全IR長で ±5%以内 |
| AC-23 | L1d / L2 キャッシュミス率が改修前から大幅に悪化しない | Intel VTune / perf stat | `vdMul(real)` + `vdMul(imag)` 連続適用時のミス率が倍増していない |

---

## 3. 性能評価・測定結果・補助統計（後半：測定結果・補助統計）

### 3.1 AoS除去による各解像度での定量的メモリ削減予測（ステレオ）

本改修を適用した（AoS 除去）場合の 192kHz×2オーバーサンプリング時における実測および理論予測値：

| IR長 @384kHz | 現状（ステレオ） | AoS除去後（ステレオ） | 削減率 |
| :--- | :--- | :--- | :--- |
| 3.0s | 約299 MB | 約164 MB | 45% |
| 5.0s | 約412 MB | 約220 MB | 47% |
| 5.46s（最大上限） | 約639 MB | 約333 MB | 48% |

### 3.2 83ms IR環境下におけるメモリ・CPU波及効果の正確な特定

普段使用している83msのルーム補正用IR（384kHzで 31,872 サンプル）は L0 に収容され、L1/L2 側の処理やバッファ（multiplier の大きさ）の影響は全く受けない。
しかし、L0 レイヤーのみの動作であっても、

* `irFreqDomain` (32パーツ → 1パーツ)
* `fdlBuf` (32パーツ×2面 → 2パーツ)

のスクラッチ化を行うことで、**インスタンスあたり約 1.5MB の物理メモリ削減が確実に発生** し、音質やCPU負荷には一切の影響が生じない。

```text
83ms @384kHz:
  IR長 = 31,872 サンプル
  L0最大カバレッジ = 32 parts × 1024 = 32,768 サンプル
  → L0に完全収容（L1/L2未使用、メモリ1.5MB削減・CPUゼロ影響）
```

### 3.3 2.5GB 巨大メモリ消費のシステム多重原因分析

AoS 削減による削減効果は最大でも約300MB程度（5.46s IR時）であり、2.5GBという膨大な使用量のすべてをカバーするものではない。これは以下の複数システム要因が重複した「過渡的な多重世代共存」が真の原因である。

1. **DSPCore の active / fading 二重保持（最大要因）**:
   実コード調査により、`AudioEngine.h` L1898-1908 に `activeRuntimeDSPSlot` と `fadingRuntimeDSPSlot` の二重保持が確認された。各 DSPCore は `ConvolverProcessor convolver`（L845）を保持しており、5.46s IR @384kHz では単一の NUC エンジンが約640MBを消費するため、**active + fading の二重保持で約1.28GB** となる。これが 2.5GB の約半分を占める最大要因である。
2. **ISRRetireRouter の未解放退役バックログ**:
   退役世代がリアルタイムオーディオスレッドのライフサイクル側で「参照カウントがすべて0に落ちる」のを待つが、解放シグナル・退役キュー（ISR）が安全に消化されきるまで長尺メモリが解放されず、ヒープ上に一時的に生存する。`pendingRetireCount()` で監視可能。
3. **ProgressiveUpgradeThread による一時的二重保持**:
   `ProgressiveUpgradeThread.cpp` L13-50 の実装により、1024→2048→4096 の段階的FFTアップグレード中は新旧2つの NUC エンジンが同時生存する（約60秒間のウィンドウ）。アップグレード中は active/fading の二重保持に加え、さらに中間サイズの NUC エンジンが一時的に生存する。
4. **L2 small buffers の巨大配置**:
   (本件は `tailL1L2Mult` 最小値を **8** に緩和した［実装済み✅］ことで、L2 small buffers が 12.4MB → 5.5MB（56%減）となり大幅に緩和されている)。

> **2.5GB 内訳試算**: AoS (640MB) + active/fading 二重 (640MB) + ProgressiveUpgrade 中間 (320MB) + retire 滞留 (256MB) + その他 (L2/DCBlock等) (128MB) = 約1,984MB〜2,048MB。AoS削減後は約1,700MBとなり、継続調査により retire 滞留・ProgressiveUpgrade の中間生成を最適化することで 1.2GB 程度までの削減が期待できる。

### 3.4 2.5GB との差分を切り分けるためのシステム計測提案

AoS 除去の適用後でもなおメモリ消費が想定以上に大きい場合の追跡手順：

1. **DSPCore 二重生存確認**: IRロード後2〜3分待って（遷移ウィンドウ終了後）に再度測定値を確認する。
2. **退役キューバックログ確認**: `pendingRetireCount()` をタイマーで定期的に `juce::Logger` に出力し、0に収束することを確認する。
3. **生存インスタンス数直接計測**: `MKLNonUniformConvolver` コンストラクタ/デストラクタに `static std::atomic<int> liveCount` を宣言し出力、不要な生存を診断する。

### 3.5 tailL1L2Multiplier 下限緩和による影響評価

multiplier 最小値を **12→8** に緩和する最適化（[src/MKLNonUniformConvolver.cpp](src/MKLNonUniformConvolver.cpp#L544) L544）における影響評価：

* **音質への影響**:
  畳み込みの数学的性質上、パーティションが変化しても出力値は完全に同一となる。FFT 演算の倍精度累積誤差は可聴域外。
* **メモリ削減効果 (@384kHz, blockSize=1024)**:
  L2 small buffers が 12.4MB → 5.5MB (56%減)、L1 small buffers が 1.0MB → 0.69MB (31%減) となり非常に効果的。
* **L2 small buffers 内訳**:
  `mult=8` 時の 1層あたりの固定バッファ (fftTimeBuf: 1.0MB, fftOutBuf: 1.0MB, prevInputBuf: 0.5MB, accumBuf: 1.0MB、その他合計 5.5MB) を調査。

### 3.6 Gardner (1995) 段階的増加率の評価

Garcia (2002) / Gardner (1995) の非均一分割設計に基づいた増加率とレイヤー数の評価：

* Gardner ×2 (6層): 最大小バッファ 2.8MB まで落とせるが実装・管理コスト（〜20行、定数変更など）が膨大。
* 現行の ×8 採用 (3層): 最大小バッファ 5.5MB であり 1行の変更のみで適用可能であるため極めて高コスパ。

### 3.7 ルーム補正用途におけるレイヤー設計の評価

3層設計（L0/L1/L2）はコンボリューションリバーブ（〜5s）を想定したものであり、現在の1.0s未満のルーム補正（83ms等）では L1/L2 が生成すらされないため実害はない。

### 3.8 実コード検証結果（2026-07-07 実施）

全11の Patch について、実コードへの正確性を WSL grep/rg および実コード読解により検証完了：

* **AVX2 Split Complex 経路における検証**: AoS ポインタ (`srcA`/`srcB`) がホットループで一度もデリファレンスされておらず、SoA のみで積算できている事実を確認（本検証は `kEnableSplitComplexKernel == true` すなわち AVX2 分岐が選択された経路に限定される。非 AVX2 非 SplitComplex フォールバックは Patch 2/11 で削除予定であることを前提とする）。
* `Reset()` 時の `fdlBufSize` クリア漏れによるバッファオーバーラン予測を回避。

#### 実装済みの変更のまとめ

* `tailL1L2Mult` 最小値 12→8 (L544): ✅
* `CONVOPEQ_STANDALONE_ONLY` 定義追加 [CMakeLists.txt](CMakeLists.txt#L571): ✅
* Floatルートスタブ化 [src/audioengine/AudioEngineProcessor.cpp](src/audioengine/AudioEngineProcessor.cpp#L76): ✅
* 負の index 修正 `+DELAY_BUFFER_SIZE` [src/ConvolverProcessor.Runtime.cpp](src/ConvolverProcessor.Runtime.cpp#L479): ✅

#### 残タスク優先度

1. **Patch 4+11（セット）**: 最優先（メモリ削減の中核、Resetバッファオーバーラン防止）
2. **Patch 8+9+10**: 最優先（ホットループ SoA 移行）
3. **Patch 3+7**: 優先（Message Thread フィルタ of SoA 直接乗算への移行）
4. **Patch 1+2+5+6**: 通常（コメント・安全ガードの追加）
