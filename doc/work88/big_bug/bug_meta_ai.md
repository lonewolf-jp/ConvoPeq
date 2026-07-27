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



# ConvoPeq 改修計画書 - バグレポート基準

> 前回検出 62件 (Critical 12 / High 18 / Medium 24 / Low 8) に対する是正計画
> 対象: Windows11 x64専用, AVX2必須, MSVC 19.44+/icx 2026.0, JUCE 8.0.12, MKL/IPP
> 期間目安: 7週間 / 3名体制想定

## 0. 改修の基本方針

**RTセーフティを最上位不変条件とする**
1.  **オーディオスレッド禁止物リストをコードで強制**: `new/malloc/free, mutex/lock, File I/O, QPC/getCurrentTimeUs, mkl_malloc, std::abort` を `AudioEngine.Processing.*` , `DSPCore*`, `MKLNonUniformConvolver::process*` で `static_assert` / `clang-tidy` カスタムチェックでビルドエラー化
2.  **AVX2は `loadu/storeu` をデフォルトに**: `load_ps/store_ps` は `alignas(32)` が証明できる箇所のみ許可。JUCEの `AudioBuffer` 由来は全て `u` 版。
3.  **MPMCキューはライブラリに寄せる**: 自前Vyukov実装は `uint64_t` 化で延命しつつ、最終的に `moodycamel::ConcurrentQueue` (B23コメントの意図通り) へ移行。
4.  **ゼロ除算は型で防ぐ**: `SampleRate` を `NonZeroDouble` ラッパー型にし、0が入らないことを型システムで保証。

## 1. フェーズ計画

### Phase 0: 緊急止血 - ビルドゲートとクラッシュ潰し (Week 1) - P0

| # | 対象バグ | 作業 | ファイル | 工数 |
|---|---:|---|---|---|
| 0-1 | #1, #12 | **RT違反ビルドゲート導入**: `src/audioengine/RTAssert.h` を新設。`CONVOPEQ_RT_SAFE` マクロ内で `new` を `=delete` したアロケータを上書き。`clang-tidy` に `bugprone-rt-alloc` カスタムチェック追加。CIで `CONVOPEQ_ENABLE_CLANG_TIDY=ON` を強制 | `CMakeLists.txt`, `DiagnosticsConfig.h`, 全 `Processing.*` | 1d |
| 0-2 | #1 | `DSPCoreLifecycle` の `new` を `prepareToPlay` 事前確保に移動。`DCBlockerRuntimeState` 等を `std::unique_ptr` ではなく `AlignedStorage` + placement new で `prepare()` 時に構築 | `AudioEngine.Processing.DSPCoreLifecycle.cpp`, `DSPLifetimeManager.h` | 1d |
| 0-3 | #2 | `LockFreeAudioRingBuffer::push` 修正。`if (channelsToWrite==1)` 時の2ch固定複製を `for (int ch=1; ch<numChannels; ++ch)` に | `LockFreeAudioRingBuffer.h:60` | 0.5d |
| 0-4 | #11 | AVX即時修正: 全 `store_ps/load_ps` を `storeu_ps/loadu_ps` に置換。`rgrep _mm256_(load|store)_ps` で機械置換後、パフォーマンス計測 | `SpectrumAnalyzerComponent.cpp:475`, `CustomInputOversampler.cpp`, `MKLNonUniformConvolver.cpp` | 0.5d |
| 0-5 | #23 | `sampleRate` ゼロガード一括挿入。`sanitizeSampleRate(double sr) -> double { return sr>1000 ? sr : 48000; }` を `DspNumericPolicy.h` に追加し、全除算前に適用 | `AllpassDesigner.cpp:230` 他6箇所, `UltraHighRateDCBlocker.h:98`, `IRDSP.cpp` | 1d |
| 0-6 | 雑 | `src/nul` ファイル削除, `graphify-out/` を `.gitignore` 追加。Windows予約名対策 | - | 0.2d |

**DoD**: `build.bat Release icx` が警告0、RTスレッドでの `new` 検出テストがCIで落ちること。48h連続再生でクラッシュ0。

### Phase 1: 並行性とライフサイクルの核を直す (Week 2-3) - P0/P1

| # | 対象 | 作業詳細 |
|---|---|---|
| 1-1 | **DeferredDeletionQueue** #3, #4 | **a)** シーケンスを `uint32_t`→`uint64_t` に。`kQueueSize=4096` は維持。`diff` 計算を `int64_t` に。<br>**b)** `reclaim()` の先頭ブロッキング解消: `kMaxScan=1024` を活かし、非FIFOスキャンを許容する `reclaimAny()` を新設。FIFO順序が必要な `Generic` 型のみ旧挙動、DSP/スナップショットは順序緩和。<br>**c)** `alignas(64)` は維持しつつ、`enqueuePos/dequeuePos` を `atomic<uint64_t>` に。 |
| 1-2 | **LockFreeRingBuffer** #10, #20 | `size()` を `sizeApprox()` にリネーム。`clear()` は `prepare()` からのみ呼べるように `assert(isPrepared==false)`。`pushWithWriter` に `noexcept` 強制と `static_assert(std::is_nothrow_invocable_v<Writer>)`。 |
| 1-3 | **ISRRetireRouter** #5 | RTパス分離: `retireRT()` は時間取得も `tryReclaim()` もせず `enqueue()` 失敗時は即 `QueuePressure` を返すのみ。新設 `retireBackground()` がノンRTで `getCurrentTimeUs()` + `tryReclaim()`。`m_lastForcedReclaimTimeUs_` はノンRT専用に。 |
| 1-4 | **ISRLifecycle** #6 | `std::abort()` を `RuntimeHealthMonitor::recordHostChaosViolation()` + `return fallbackToken` に。オーディオスレッドでは絶対に `abort` しない。Host Chaos時はバイパス処理で無音を返す。 |
| 1-5 | **MKLハンドル** #9, #21 | `DftiHandle` に統一。`ConvolverProcessor.h` の直接 `DftiCommitDescriptor` 呼び出しを削除。`IppFFTPlanCache` を Meyers Singleton化: `static auto& instance() { static IppFFTPlanCache c; return c; }` で初期化順序問題解消。 |

**検証**: 
- `DeferredDeletionQueueReclaimTests.cpp` にラップアラウンドテスト (2^32+1000回 enqueue/dequeue) 追加。
- `PriorityIntegrationTests` で MPMC 8スレッド x 1M enqueue/dequeue ストレステスト、TSanで実行。
- 72時間再生 + ランダム `prepareToPlay/releaseResources` 呼び出し (Host Chaos Simulator)

### Phase 2: DSP数値安定性とリソース管理 (Week 4-5) - P1

| # | 対象 | 作業 |
|---|---:|---|
| 2-1 | CustomInputOversampler #7 | `prepare()` で `historySize < 6` なら `jassertfalse` + 早期リターンではなく `historySize = max(historySize, 6)` に丸め。`loadStride2` 入口で `JUCE_ASSERT(ptr != nullptr && isAligned)`。単体テストで `ratio=1` 時の境界読み取りを ASanで検出。 |
| 2-2 | MKLNonUniformConvolver #21 | 一時バッファを `struct MklBuffer { double* p=nullptr; ~MklBuffer(){ if(p) mkl_free(p);} }` でRAII化。`SetImpulse` 内の `tempTime/tempFreq` も `std::unique_ptr<double, MklDeleter>` に。 |
| 2-3 | OutputFilter / TruePeak / DCBlocker #24, #25 | 全フィルタ係数算出後に `isFiniteNoLibm()` チェック。Inf/NaNなら係数を直前値にフォールバック + `RuntimeHealthMonitor::recordDSPAnomaly()`。デノーマル対策: `flushDenormal()` を `process` 入口で呼ぶか、係数に `kDenormThreshold` 加算。 |
| 2-4 | EQ / CmaEs #36, #37 | `EQEditProcessor::gainToLinear` を `double` 中間計算 + `juce::jlimit(1e-6, 1e6, value)` でクランプ。CMA-ESは `best_f0` が NaNならループ脱出。 |
| 2-5 | CacheManager #22, #40 | `createDirectory()` の戻り値チェック + リトライ3回。`deleteFile()` 失敗時は `moveToTrash` ではなく `replaceFile` パターンに。キャッシュ書き込みは `write temp -> flush -> atomic rename` でTOCTOU防止。パスは `getChildFile` の前に `..` 除去サニタイズ。 |

**検証**: `EQProcessorMaxGainTests`, `EQAnalysisUnitTests` に極端値 (gain 60dB, Q 100, sampleRate 0, 192kHz) を追加。`MklFftEvaluator` で 1e-6 以下のデノーマル入力でCPU時間計測。

### Phase 3: UI/デバイス/ビルド基盤 (Week 6) - P2

- **DeviceSettings / AsioBlacklist** #38: 完全一致ブラックリストに変更。正規表現 `^ASIO$` ではなく `equalsIgnoreCase`。`ASIO4ALL` 誤ブロック解消。単体テスト追加。
- **CpuFeatureCheck** #39: `__cpuid` + `XGETBV` で OSXSAVE + AVX2 有効確認。`IsProcessorFeaturePresent` 削除。
- **CacheManager / ProgressiveUpgradeThread**: バックグラウンドスレッドの `catch(...)` で半完成キャッシュを削除する `ScopedCacheTransaction` 導入。
- **ビルド**: `CMakeLists.txt` で `CONVOPEQ_ENABLE_CLANG_TIDY` をデフォルトON、ローカルでも `NUC_DEBUG_GUARDS` 有効化。`build.bat` に `set -e` 的エラー停止。`clang-tidy` チェックに `concurrency-mt-unsafe`, `bugprone-exception-escape`, `performance-noexcept-move` 追加。
- **計測**: `TelemetryRecorder` の `liveCount` を `relaxed` から `acquire/release` に。診断用とはいえ可視性が必要。

### Phase 4: 移行と長期対策 (Week 7) - P2/P3

1.  **moodycamel移行 PoC**: `DeferredDeletionQueue` を `ConcurrentQueue<DeletionEntry>` に置換したブランチを作成。ベンチマークで RTバジェット (平均 < 50us) を満たすか計測。満たせば本移行。
2.  **型システム強化**: `SampleRate` 型、`NonNullPtr<T>` 型導入でゼロ除算・ヌルポインタをコンパイル時排除。
3.  **ドキュメント**: `docs/RT_Safety_Guidelines.md` 作成。禁止APIリスト、AVXアライメント規約、MPMCキューの正しい使い方を明記。PRテンプレートにチェックリスト追加。
4.  **ファジング**: `IRConverter` に libFuzzer で破損WAV投入、クラッシュ0を確認。

## 2. 検証計画詳細

| 層 | 手法 | 対象バグ |
|---|---|---|
| **静的** | `clang-tidy`, `MSVC /analyze:WX`, `cppcheck --enable=all` をCIゲート化 | RT違反, 未初期化, リーク |
| **動的** | ASan (ヒープ外), UBSan (ゼロ除算/オーバーフロー), TSan (データ競合) を `Debug` で nightly実行 | #3, #4, #7, #22 |
| **オーディオ** | 1) 48k/96k/192kHzで 24h正弦波再生 + ランダム `prepare/release` (Host Chaos) 2) `JACK` での XRUNカウンタ0 3) `Intel VTune` で `processBlock` < 30% CPU | #1, #5, #6, #24 |
| **単体** | 既存 21テスト + 新規 15テスト (ラップアラウンド, 境界読み, 0サンプルレート, モノラル>ステレオ) | 全Critical |
| **実機** | RME Babyface Pro FS + ASIO, mac無し, Windows11 22H2/23H2, `LatencyMon` で DPC < 500us | #38, #39 |

## 3. リリースとロールバック

- **Canary**: Phase1完了時点で内部β (v14.5-canary)。テレメトリで `reclaimFailCount / queueFullCount` を監視。
- **段階リリース**: Phase2完了で v14.6-beta (50%ユーザ)。クラッシュレポートが前バージョン比 -80% ならGA。
- **ロールバック条件**: XRUN +20% または `liveCount` リーク検出で即ロールバック。`CacheManager` は旧フォーマットも読めるように後方互換維持。

## 4. 成果物と工数

- **成果物**: 修正済みソース + `RT_Safety_Guidelines.md` + `clang-tidy` カスタムチェック + 新単体テスト15 + 検証レポート
- **工数目安**: P0 3d, P1 8d, P2 7d, P3 4d = 22人日。3名で7週間、レビュー含めバッファ2週間。

---

**次のアクション**: Phase0-1の `DeferredDeletionQueue` `uint64_t` 化と `LockFreeAudioRingBuffer` 修正の差分パッチから着手します。パッチ生成をご希望ですか？