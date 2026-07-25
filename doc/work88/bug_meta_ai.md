添付の結合Markdownから279ファイル、約81k行を展開し、AST的ヒューリスティックと手動レビューで全量解析しました。

**手法:**
- `aligned_malloc / mkl_malloc / Dfti` ライフサイクルの追跡
- `_mm256_load/store` アライメント契約の検証
- Vyukov MPMC / SPSCリング、RCU、epoch比較のメモリオーダ検証
- `new / delete / mkl_free` の早期return漏れ検出
- Audio Thread (RT) での `mutex / alloc / throw` 禁止違反検出

結果、クラッシュに繋がるものから潜在的なものまで計18件を検出しました。

### 重大度 Critical / High

| # | ファイル:行 | 症状 | 影響 | 修正案 |
|---|---|---|---|---|
| **C-1** | `src/InputBitDepthTransform.h:114-115` <br> ` _mm256_store_pd(dst + i, ...)` | `convertFloatToDoubleHighQuality` が aligned store `_mm256_store_pd` を使用。関数の契約に「dstは64byte aligned」の明記が無い。現在の呼び出し元 (`AudioEngine.Processing.DSPCoreIO.cpp:216` `alignedL.get()`、`LoaderThread: tempAligned`) は偶然 aligned だが、将来の呼び出しで非alignedな `double*` が渡ると #GP 例外で即落ち。 | プロセス全体クラッシュ | `storeu` に変更するか、契約を `[[expects]]` で明示: <br> ` _mm256_storeu_pd(dst+i, ...)` |
| **C-2** | `src/AudioSegmentBuffer.h:50-80` | `pushBlock()` は `leftSamples/rightSamples` へのコピー後に `writePosition` を release。`copyLatest()` は `writePosition` を acquire 後に読み取るが、リングラップ時の2ndチャンク書き込み中にリーダーが `start=0` を読むと、書き込み途中の領域を読み取る。C++メモリモデル上は non-atomic配列へのデータ競合。 | 歪み、torn read、最悪クラックルノイズ | SeqLock化: `totalSamples` と `writePosition` を単一の64bitシーケンスにまとめるか、ダブルバッファにする。簡易修正はリーダー側で2回読み取り検証。 |
| **C-3** | `src/MKLNonUniformConvolver.cpp:1042,1059` | `ippsFFTFwd_RToCCS_64f / ippsFFTInv_CCSToR_64f` の戻り値 `IppStatus` を無視。IPP初期化失敗や不正な `fftSpec` 時に `irFreqDomain` にゴミが残る。 | 無音またはNaN爆発 | `if(status != ippStsNoErr){ releaseAllLayers(); return false; }` |
| **C-4** | `src/LockFreeRingBuffer.h:63-69` `size()` | `writeIndex` と `readIndex` を別々に acquire。間に Producer が進むと `w-r` が Capacity を超えた値を返し、`push()` の満杯判定をすり抜ける。SPSCなので実害は稀だが、`getAvailableSamples()` が負や巨大値を返す。 | キュー破壊、誤ったドロップ/上書き | スナップショットは best-effort と文書化するか、 `writeIndex` を1回読んだ後に `readIndex` を読む順序を固定。 |
| **H-1** | `src/core/RCUReader.h:enter()` | `ownerThreadToken` に `cachedThreadHash()` を使用。ハッシュ衝突時に2スレッドが同一オーナーと誤認し、`activeThreadId` を共有、`epochProvider` の slotを二重登録。 | epochが永遠に進まず `DeferredDeletionQueue` が詰まりメモリリーク → 最終的に reclaim不能 | `thread_local uint64_t` で単調増加IDを採番。ハッシュは診断用に留める。 |
| **H-2** | `src/audioengine/ISRRetire.cpp:20-75` | `emitRetireIntent` は本来 RT安全を謳うが、輻輳時に `std::lock_guard<std::mutex> fallbackMutex_` を取得。コメントにも `Finding 9: RTではない` とあるが、将来 Audio Thread から呼ばれると優先度逆転でオーディオドロップ。 | RT違反、クリック | RTパスは mutex無しの overflowRing のみに退避、non-RT側で fallbackへ移動する2段階化。 |
| **H-3** | `src/convolver/ConvolverProcessor.LoaderThread.cpp:467` | `tempFloatBuffer(numChannels, fileLength)` で `fileLength` は `MAX 2,147,483,647` まで許可。ステレオで8GB確保試行、`std::bad_alloc` は捕捉されるが、JUCEの `AudioBuffer` 内部は `new` 失敗時に例外ではなくアサーションになるビルドもある。 | OOMクラッシュ | ストリーミング読み込み: 256kブロック毎に `convertFloatToDoubleHighQuality` へ流す。 |
| **H-4** | `src/MKLNonUniformConvolver.cpp:775,926-988` | `l.fftSize * sizeof(double)` が `int * size_t`。`fftSize` は `int` だが、極端なIR (例 5秒@768kHz=3.8M, partSize 262k → fftSize 524k) でも収まるが、将来 `kL0MaxParts` 拡張で int溢れ。`allocSizes` も同様。 | ヒープ破壊 | `static_cast<size_t>(l.fftSize) * sizeof(double)` に統一。 |

### Medium

**M-1  AVX→SSE遷移ペナルティ**
`TruePeakDetector.cpp:scanPeak()` `EQProcessor.Processing.cpp:calculateRMS()` 等で `_mm256_store_pd` 後に `_mm256_zeroupper()` 無し。AVX使用後にSSEのJUCE関数を呼ぶと Skylake以降で ~70cycle ペナルティ。
修正: 関数末尾で `_mm256_zeroupper()`。

**M-2  `CacheManager.cpp:203,241` volatile sink**
最適化抑止に `volatile uint8_t sink` を使用。MSVCでは volatileがメモリバリアにならず、かつC++20では非推奨。`std::atomic_thread_fence` または `DoNotOptimize()` (benchmarkライブラリパターン) を使用。

**M-3  `SnapshotFactory.cpp:hashCombineFloat`**
`-0.0f` と `0.0f` は同一視するが、NaNのペイロード違いは別ハッシュ。`areSnapshotsEquivalent` は epsilon比較で同値と判定するが、ハッシュ不一致で常に新規 `GlobalSnapshot` を `new` し、RCUに流す無駄な生成。
修正: NaNは `0x7FC00000` に正規化してからハッシュ。

**M-4  `DeferredDeletionQueue.h:120` kMaxScan デッドコード**
コメントにもある通り `scanPos==deqPos` で即breakするため `scanned < kMaxScan` は機能しない。将来「先頭以外が先にreclaimable」な拡張をした際に無限ループ化。
修正: 現状維持なら `kMaxScan` 削除、将来拡張ならリング先頭からの線形スキャンを実装。

**M-5  `AlignedAllocation.h:20-22` 例外のRT伝播**
`aligned_malloc` が `throw bad_alloc`。`makeAlignedArray` は Audio Thread から呼ばれることは無い想定だが、`MKLAllocator` が `std::vector` 経由で RTに使われると例外がRTを破壊。`_nothrow` 版をRTでは強制するラッパーを用意。

**M-6  `MKLNonUniformConvolver.cpp:1674-1684` アライメント判定の競合**
`aligned` フラグは関数入口で一度だけ計算。`dst` と `src` が同じアライメントでも `i` が4進む毎に32byte境界を跨ぐため、ベースが32byte alignedなら常にaligned。現状は安全だが、`partStride` が奇数の時に `accumBuf` が 32byte境界を外す可能性。`accumBuf` は64byte確保だが `partStride` が4の倍数でないと崩れる。`partStride` を常に8の倍数に丸めるべき。

### Low / 品質

- **L-1** `src/SpectrumAnalyzerComponent.cpp:475` は一見 `_mm256_store_ps` の不正に見えるが `alignas(64) float mags[8]` で安全。ただしループ内に `alignas` を置くとMSVCで毎回スタックを64byteアラインし直すため無駄。ループ外に移動推奨。
- **L-2** `src/ConvolverProcessor.h:919` `cachedLatency { new LatencySnapshot() }` はデストラクタで `exchangeAtomic(nullptr)` + `unique_ptr` で解放しておりリークは無いが、例外安全ではない。`aligned_unique_ptr` に移行推奨。
- **L-3** `src/core/GlobalSnapshot.h` コピー禁止だがムーブも禁止のため `SnapshotFactory::create` が `new` しか選べない。`std::make_unique` 経由のNRVO阻害。
- **L-4** `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:206` BUG-04 は修正済みだが、スカラーフォールバック側で `prevSampleInOut` に生入力を保存する意図がコメントと実装で逆だったのが修正されている。同様のパターンが他にも無いか要監査。

### 推奨対応順序

1.  **C-1, C-3 を即時修正** – クラッシュ/無音に直結。
2.  **C-2, H-1, H-2** – 長時間稼働で顕在化するRCU/キューの詰まり。診断ビルドで `ISRRuntimePublicationCoordinator` の `pendingIntentCount` を監視。
3.  Mediumはパフォーマンス/将来の拡張性。`_mm256_zeroupper` と `storeu` 化は機械的に置換可能。

必要なら、各バグに対するパッチ差分 (`*.patch` 形式) や、clang-tidy / AddressSanitizer / ThreadSanitizer を Windows11 + ICX向けに有効化する `CMakePresets.json` も作成できます。