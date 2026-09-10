# ConvoPeq 修正済み統合バグリスト（CORRECT_INTEGRATED_BUG_LIST）

- **作成日**: 2026-09-09
- **検証基準**: 現行ソースコード（HEAD 54ba7b40・ConvoPeq.md Generated: 2026-09-09 01:32:32・NEWER_SRC_COUNT=0）を直接照合して全項目を再判定
- **元リスト**: `doc/work92/INTEGRATED_BUG_LIST.md`（3 ソース統合版）の全記載を、実コードで「妥当（バグとして残存）/ 修正済み / バグではない」に再分類
- **検証方法**: ctx_batch_execute + rtk(WSL) rg + grep/sed による全サイト実測（本リスト各行の「実測」欄に根拠コードを明記）

---

## 0. 検証の総括

```text
元リスト 56 BUG 番号 + big_bug 固有 35 項目を現行ソースで全件再判定:

✅ 修正済み確定（実コードで修正確認）  : 62 項目
🔴 バグとして残存（修正必要）          : 11 項目
🟡 バグではない / 軽微・設計判断        : 12 項目（根拠併記）
⚠️  潜伏・デッドコード経路（現状無害）  : 5 項目
```

> **最重要訂正**: 元リストで「OPEN（未修正）」としていた **BUG-011/012/013（sigma クランプ欠如 3 変体）は全て修正済み**だった。
> - BUG-011: `CmaEsOptimizer.h:84` で `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax)` 実装確認
> - BUG-012: `CmaEsOptimizerDynamic.h:29` で `setSigma` が `std::clamp` 実装確認
> - BUG-013: `CmaEsOptimizerDynamic.h:37` は宣言だが、実装 (.cpp) 側でクランプされている（動作は R-18 修正と同時に SANITIZE 済みの値が流入）
> - BUG-016 も同時修正済み: `CmaEsOptimizer.h:206-209` / `CmaEsOptimizerDynamic.h:50` の `sanitize()` が `(!std::isfinite(x) || std::abs(x) < 1e-15) ? 0.0 : x` を実装

---

## 1. 重大度別 整理（バグとして残存するもの）

### 🔴 P0 — Critical / High（ユーザーデータ消失・クラッシュ・RT 違反）

| # | BUG / ID | 内容 | 実測根拠（現行ソース） | 重大度 |
|---|---|---|---|---|
| 1 | **big 1-1** | `nucHCMode` / `nucLCMode` が `getState()` / `setState()` の ValueTree 永続化から欠落 | `ConvolverProcessor.StateAndUI.cpp` の `getState()`（:202-250）の setProperty 全 20 箇所に nucHCMode/nucLCMode が存在しない（:205-245 実測）。`setState()`（:289-）にも読込なし。ハッシュ計算（:55-56, :865）と snapshot 同期（:142-143, :194-199）には含まれるため、UI 変更→セッション保存→再読込で **2 設定のみがサイレントにデフォルトへ戻る**（ユーザーデータ消失） | **P0** |
| 2 | **big 1-3** | `_mm256_store_pd`（アライメント要求命令）が契約なしの `dst` に使用 | `InputBitDepthTransform.h:114-115` で `_mm256_store_pd(dst + i, ...)` が残存。:60/:81 は `_mm256_storeu_pd` に修正済みだが **114-115 のみ未修正**。`MKLNonUniformConvolver.cpp:1407, 1668` も `_mm256_store_pd` 残存（:178-179, :1670 は storeu 併用）。非アラインド `dst` が渡ると #GP で即クラッシュ | **P0** |
| 3 | **big 1-6 (= BUG-034)** | IPP FFT `ippsFFTFwd_RToCCS_64f` の戻り値を無視 | `MklFftEvaluator.h:270-271, 425-426` で戻り値をキャプチャせず呼び出し（実測: sed で確認）。IPP 初期化失敗時にゴミデータが下流へ伝播し NaN/無音爆発の起点 | **P0** |

### 🟠 P1 — High（RT 契約・メモリ安全・データ整合）

| # | BUG / ID | 内容 | 実測根拠 | 重大度 |
|---|---|---|---|---|
| 4 | **big 1-7** | `emitRetireIntentRT` が輻輳時 `std::mutex` を取得（RT 違反の芽） | `ISRRetire.cpp:94-104` — 実装は `emitRetireIntent()` 素通しで、輻輳時 :44/:135/:265 の `std::lock_guard<std::mutex> lock(fallbackMutex_)` に到達。**ただし**: コメント「Finding 9: 呼び出し元は全て非 RT スレッドであることを確認済み」が実在し、現状 RT 到達経路なし。`将来 Audio Thread から呼ぶ場合は mutex を使わない別実装を用意すること` と明記された **既知の契約付き設計**。RT 契約 violation ではなく「RT で呼んでも安全な API 名の誤解リスク」に降格し **P1（リネーム推奨）** | P1 |
| 5 | **big 1-8** | LoaderThread が `MAX_FILE_LENGTH = INT32_MAX` まで一括確保（OOM） | `ConvolverProcessor.LoaderThread.cpp` に MAX_FILE_LENGTH ガードはあるがストリーミング化は未実装（R-新規C と同根）。2GB 超 IR で最大 ~16GB 確保試行 | P1 |
| 6 | **big 1-9** | `l.fftSize * sizeof(double)` が `int * size_t`（将来的 int 溢れ） | `MKLNonUniformConvolver.cpp` の fftSize は依然 `int`（int64_t 化は未実施）。現行の fftSize 範囲では発症しないため **P1→将来リスク** | P1 |
| 7 | **big 1-10** | `m_pendingIRChange` の公開前クリア（IR 変更要求消失） | `AudioEngine.Snapshot.cpp:95` で `exchangeAtomic(m_pendingIRChange, false, acq_rel)` が残存（実測）。スナップショット等価判定通過時に IR 変更要求が永久消失 | P1 |
| 8 | **big 2-6** | RCUReader の `cachedThreadHash` ハッシュ衝突で reader 二重登録の可能性 | `RCUReader.h:47` の `currentThreadToken()` が `std::hash<std::thread::id>` 由来（ThreadHash.h）。単調 ID 採番への置換は未実施。衝突時 epoch が進まず reclaim 停止の潜在 | P1 |
| 9 | **big 2-9** | タイミング計算の `uint64` 減算 underflow | `AudioEngine.Processing.AudioBlock.cpp` の `nowUs - cbStartUs` 系が saturating 化されていない（実測: :624, :630, :664 系パターン残存）。診断表示の誤値 | P1（表示系） |
| 10 | **big 2-10** | `NoiseShaperType` enum キャストに範囲チェックなし | `AudioEngine.StateIO.cpp:89-90` で `(NoiseShaperType)(int)state.getProperty(...)` が依然無検証（実測）。:120 の `hasIntRange("noiseShaperType", 0, 3)` は別関数（hasIntRange は AudioEngine.Parameters.cpp 側）で StateIO では未使用 | P1 |
| 11 | ~~BUG-065 残存~~ **解消済み（work92 B-7a・2026-09-10）** | rt シャドウ直接書込 6 行（両サイト × 3 行）を削除 | `EQProcessor.Core.cpp` の reset()/prepareToPlay() から `rtDeferredBandResetMask.store(0)`・`rtSeenBandResetSerial = 0`・`rtSeenAgcResetSerial = 0` を削除（work92 B-7a 実装・CTest 40/40 PASS）。serial は先行 fetchAdd で前進済みのため Audio Thread が shadow を自己更新 | **解消済み** |

### 🟡 P2 — Medium（コード品質・保守リスク）

| # | BUG / ID | 内容 | 実測根拠 | 重大度 |
|---|---|---|---|---|
| 12 | **big 2-1** | 非 ASCII 識別子 `SoftClipPadéPolicy`（U+00E9） | `dsp/math/FastTanhApprox.h:63` に残存。`DSPCoreDouble.cpp:127, 191` が使用。cppcheck クラッシュの実績あり | P2 |
| 13 | **big 2-2** | 入力側 DC ブロッカー後の NaN/Inf スクラブ欠如 | `DSPCoreIO.cpp` の `sanitizeFiniteChunk` は :231-232, :282-283（入力前）のみ。DC ブロッカー後の追加なし | P2 |
| 14 | **big 2-3** | テストの矛盾条件（`delta < 1e-6` と `if (delta > 1e-6)`） | `EQProcessorMaxGainTests.cpp:355-358` に**矛盾条件がそのまま残存**（実測: :355 ループ条件 `< 1e-6`、:357 内側 `> 1e-6` — 同時に真になり得ず logBound は常に 0.0）。テストは finite チェックのみで実質何も検証していない | P2 |
| 15 | **big 2-4** | CacheManager strict-aliasing 違反 | `CacheManager.cpp:267` の `reinterpret_cast<const double*>` 残存 | P2 |
| 16 | **big 2-5** | `LockFreeRingBuffer::size()` の read 順序未固定 | `LockFreeRingBuffer.h:76-81` が readIndex→writeIndex 分離読み取りのまま | P2 |
| 17 | **big 3-6** | SnapshotFactory の NaN 等価誤判定（`NaN > epsilon` は false） | `SnapshotFactory.cpp` に `std::isnan` チェック未導入（実測: grep 0 件） | P2 |
| 18 | **big 3-9/3-10 + R-9** | `/fp:fast` と `/QxCORE-AVX2` が `CMAKE_CXX_FLAGS_RELEASE` で全ターゲットに適用 | `CMakeLists.txt:1519-1520`（MSVC /fp:fast）・:1606（icx /fp:fast + /QxCORE-AVX2）が残存。icx は意図的（:1585-1587 コメント: LLVM OOM 回避で /O2・AVX2+fp:fast で性能維持）。**AMD 実行は想定外である旨の文書判断が必要** | P2（文書判断） |

---

## 2. 修正済み確定（実コード照合・62 項目）

> 元リストの FIXED 判定を実コードで再確認。いずれも修正コードが現行 HEAD に存在。

### 2-1. mini bugs 系（BUG-011〜046 のうち 25 件）

| BUG | 修正の実測根拠 |
|---|---|
| **011** | `CmaEsOptimizer.h:84` — `sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax)`（deserializeFrom 内） |
| **012** | `CmaEsOptimizerDynamic.h:29` — `setSigma` が `std::clamp` 実装 |
| **013** | BUG-012 と同様の SANITIZE 経路で NaN/Inf が 0 に置換され除算-by-ゼロは構造的に到達不能に |
| **014** | `AudioEngine.h:2520-2531` — `juce::String currentDeviceTypeName_` は完全削除（grep 全 src 0 件）。`std::atomic<MmcssPolicy>` + static_assert（trivially copyable / lock-free）に置換。`Mmcss.cpp:55` は 1 byte atomic acquire load のみ |
| **015** | `SnapshotCoordinator.cpp:57, 114` / `ISRRetireRouter.cpp:296` — `const auto result = enqueueWithRetry(...)` で戻り値受取。`SnapshotCoordinator.cpp:16` に `★ BUG-015/027` 退避移送コメント |
| **016** | `CmaEsOptimizer.h:206-209` / `CmaEsOptimizerDynamic.h:50` — `sanitize()` が `(!std::isfinite(x) || std::abs(x) < 1e-15) ? 0.0 : x` |
| **018** | `!= 1.0` FP 等価比較パターンが LoadPipeline/DSPCoreDouble/MKLNonUniformConvolver で消滅（big_bug R-26 再確認） |
| **019** | `TruePeakDetector.cpp:102-103` — `static_cast<size_t>(numSamples) * 2/4` |
| **021** | `ConvolverProcessor.Lifecycle.cpp:144-151` — timerCallback に GlobalGuard パターン追加 |
| **022** | `Lifecycle.cpp:211-217` — prepareToPlay に GlobalGuard 追加 |
| **024** | `SnapshotFadeState.h:39-72` — `fadeGeneration_` ABA generation（:40 publish・:50 保存・:67-72 再確認）+ :144 宣言 |
| **026** | `ObservedRuntime.h:48-50` — `if (!guard.rootEnterSucceeded()) return nullptr;` が実装（★ C-7 コメント・Release でも有効） |
| **028** | `CrossfadeRuntime.h` — `dryScaleTarget_/startDelayBlocks_/dryHoldSamples_` の stale target 解消（:107, :134, :138 で publishAtomic リセット） |
| **029** | `DSPTransition.h:75-77` — Emergency Override が `exchangeFadingRuntimeDSP(oldDSP)` を呼び prevRaw を retire（BUG-029 意図的設計・work89 N-1 再確認） |
| **031** | `AudioEngine.h` — `updateAudioThreadSnapshotFade` は [DELETED] 2026-07-28 で完全削除 |
| **033** | `BlockDouble.cpp:420-427` — `★ BUG-033/C-1` コメント付きで `dryScale = useDryAsOld ? crossfadeRuntime_.getDryScaleGain().getNextValue() : 1.0` 適用 |
| **035** | `ConvolverProcessor.LoadPipeline.cpp:324-336` — `ApplyComputedIRLoadingGuard` RAII クラス（:350 世代 mismatch は guard 範囲外で return） |
| **036** | `LoadPipeline.cpp:640-648`（R-23）— `.get()` → init() 成功時のみ `.release()` |
| **037** | `StateAndUI.cpp:969-985` — `forceCleanup()` が `loader->stopThread(500)`（R-37） |
| **038** | `SpectrumAnalyzerComponent.h:74` — `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` が正値（レポートが旧版基準の誤報） |
| **039** | `CustomInputOversampler.cpp:793` — `std::min(targetSamples, upsampledBlock.getNumSamples())` |
| **041** | `NoiseShaperLearner.cpp:645-657` — VLA 消滅・`convo::makeAlignedArray<double>` + `vdTanh` に置換 |
| **042** | `CmaEsOptimizer.h:43-46` — copy/move 4 種 `= delete` |
| **045** | `IRConverter.cpp:267-273` — resample 失敗時 `converted = ir; actualSampleRate = sourceRate;`（★ コメントで旧誤ラベルを明記） |
| **046** | `PsychoacousticDither.h:102-105` — copy/move 4 種 `= delete` |

### 2-2. work89 系（BUG-047〜065 の 19 件・全件修正確認）

work89 §15（2026-08-12 再検証）+ 本検証での抽查で全件維持を確認。代表根拠:

| BUG | 修正の実測根拠（本検証の抽查） |
|---|---|
| 047 | `EQProcessor.h:265` / `ProcessingCache.cpp:24, :65` — `computeParamsHash(params, sampleRate, maxBlockSize)` 3 引数化 |
| 052 | `RuntimePublicationOrchestrator.h` — `DeferredPublishView`（move-only・finishView 委譲）パターン |
| 053 | `AudioEngine.Learning.cpp:55-63` — 直接 `stopLearning()` 削除 |
| 056/057/058/062 | `RuntimeHealthMonitor` — Normal 復帰パス・`EVENT_OVERFLOW_RATE_*` 専用コード（h:60-61）・`m_prevWorldConsistencyState_`（h:372）・uint64 版復帰 |
| 060 | `ISRRetireRuntimeEx.cpp:218` — `fetchSubAtomic` 単一アトミック + previous==0 回復 |
| 061 | `ISRDSPQuarantine.cpp` — `std::mutex` 全アクセス保護 |
| 063 | `EpochDomain.h:74, 523-525, 571` — `ownerThreadId`（std::atomic<uint64_t>）ガード |
| 065 | `EQProcessor.Core.cpp:281, :792` — `fetchAddAtomic(agcResetSerial, 1)` increment 化（★ BUG-065 コメント付き）。**rtSeen 直接書込は work92 B-7a（2026-09-10）で全廃済み** |
| 048-051, 054, 055, 059, 064 | work89 §15.2 の行番号表で全件確認済み（本検証で BUG-049 の CAS 遷移 :255-316・BUG-050 の epoch 先行 store :107-115 も再確認） |

### 2-3. big_bug 系のうち修正・解消済み

| ID | 判定と根拠 |
|---|---|
| big 1-4（fastTanh 複製） | **部分解消** — `DSPCoreDouble.cpp` は `convo::dsp::fastTanh<SoftClipPadéPolicy>` を使用（正規実装）。ただし `DSPCoreFloat.cpp`/`DSPCoreIO.cpp` のローカル複製は要確認（本検査では Float 側のローカル fastTanh 定義は grep で非検出 → 複製は解消済みの可能性が高い。ただし :146 系の呼び出し形跡は旧行番号のため確定には至らず）→ **P2 として残す**（§1 には計上せず・次回 Float/IO の呼び出し確認推奨） |
| big 1-2（coordinatorDeferredRing_ / lastResortQueue_） | **設計判断確定済み（バグ扱い解除）** — `ISRRuntimePublicationCoordinator.h:1013-1016` に構造は残存するが、big_bug §10-3-1 で「producer は実装しない（X6 設計方針と整合）・値初期化 `{}` のみ安全に実施可」と方針確定済み。lastResortQueue_ 未初期化の UB の芽は `{}` で除去可能なため **P2（保守）に降格** |
| big 2-7（atomic\<DSPHandle\> 検証不足） | `ISRDSPHandle.h:107, :215` に `static_assert(std::atomic<DSPHandle>::is_always_lock_free)` が **2 箇所実装済み**（実測）→ **修正済みに転換**（big_bug の旧指摘 :186 ガードの記述は現行と不整合） |
| big 2-8（cleanup 強制削除） | **バグ扱い解除** — `forceCleanup()`（:969-985）が stopThread(500) を担い、cleanup() は破棄時の安全回収と役割分担確定（big_bug §10-3-3） |
| R-新規E | 修正済み（監査記録のみ） |
| R-9 | CMake グローバル上書きは残存 → §1 P2 の 3-9/3-10 に統合計上 |

---

## 3. バグではないもの（根拠つき・12 項目）

| # | BUG / ID | 元リストの主張 | **バグではない根拠（現行ソース実測）** |
|---|---|---|---|
| 1 | BUG-038 | SpectrumAnalyzer +6dB 表示誤差 | `SpectrumAnalyzerComponent.h:74` は `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` と正しい値（実数 FFT の正規化）。元レポートは古いバージョン基準で、big_bug R-22 で誤報確定済み → **実コードも正当** |
| 2 | BUG-031 | updateAudioThreadSnapshotFade スタブ | 2026-07-28 に Dead Code として完全削除済み（[DELETED] コメント）。スタブ自体が存在しない → **解消済み（バグ不存在）** |
| 3 | BUG-043 | estimateMaxFrequencyResponseGain の sampleRate 引数未使用 | `IRConverter.cpp:394-399` — 引数は `/*sampleRate - future use for frequency weighting*/` と**意図的な将来予約**として文書化済み。API 意図の明示であり誤実装ではない |
| 4 | BUG-027 / BUG-031 系（completeFade/updateFade 競合） | updateFade が completeFade と競合 | `SnapshotCoordinator.h:118` — `★ [DELETED] 2026-07-28: updateFade は Dead Code のため削除（呼び出し元なし）`。**競合相手の関数が存在しない**ため競合として成立しない |
| 5 | BUG-025 | switchImmediate の enqueueRetry 未使用リーク | `SnapshotCoordinator.cpp:57, :114` 両サイトとも `const auto result = enqueueWithRetry(...)` で結果受取＋`quarantineRetireSink` 退避（:117 `"completeFade:queueFull"`）実装済み → BUG-015 修正と同時に解消 |
| 6 | BUG-044 | MklFftEvaluator Rule of Five 違反 | **解消済み確定（2026-09-10 再確認）**: `MklFftEvaluator.h:138-141` に copy/move ctor/assign の `= delete` 4 連が実在（RECONCILIATION 2026-09-09 §4 でも確認済み）。旧 grep 非検出は検証ツールのエンコーディング起因と判断 |
| 7 | big 3-1 | AudioSegmentBuffer リングラップ競合 | `NoiseShaperLearner.cpp` のみで使用される SPSC（単一 producer/consumer）で、`pushBlock` → `copyLatest` の実測呼び出しパターンでは競合ウィンドウが構造的に到達困難。C++ メモリモデル上は UB の芽だが実害なし（big_bug §3-1 の検証詳細どおり）→ **設計許容** |
| 8 | big 3-3 | AlignedAllocation 例外 RT 伝播 | `aligned_malloc_nothrow()`（:32）が提供済みで、RT パス（Processing 系）に確保が存在しないことを big_bug §10-3-4 が全数確認済み → **契約充足** |
| 9 | big 3-4 | MKLNonUniformConvolver アライメント判定 | dst/src はともに `mkl_malloc(…, 64)` で 64-byte アライン保証 → aligned フラグは常に true で実害なし（big_bug §3-4 検証詳細どおり） |
| 10 | big 3-8 | cachedLatency 例外安全性 | コピーが `= delete` されており、`new LatencySnapshot()` の bad_alloc は NonRT 公開前のみ → **安全確認済み** |
| 11 | big 2-8 | cleanup() 強制削除未実装 | 役割分担は意図どおり（cleanup=安全回収 / forceCleanup=強制停止）。`forceCleanup()` が `stopThread(500)` を実装（§2-3）→ **設計どおり** |
| 12 | BUG-040 | NSL 再生時間 1Hz フォールバック | `NoiseShaperLearner.cpp:1174-1178` — `playbackSampleRateHz = (session.sampleRateHz > 0) ? session.sampleRateHz : ((block.sampleRateHz > 0) ? block.sampleRateHz : 48000)` と **3 段フォールバックが実装済み**（session → block → 48000）。1Hz にフォールバックする経路は存在しない → **誤報（旧コード基準）** |

---

## 4. 潜伏・デッドコード経路（現状無害・5 項目）

| # | ID | 内容 | 現状と根拠 |
|---|---|---|---|
| 1 | R-新規A | `IncrementalRebuildJob::reset()` の pendingConv 直接破棄（NUC リーク） | `Rebuild.cpp:110-135` に `~StereoConvolver() + aligned_free` パターン残存（実測）。ただし rebuildJob は呼び出し元が `setUseIncrementalRebuild`（:282）等のデッドコード経路のみで、現行では必ず nullptr → **未発火** |
| 2 | R-新規B | Incremental rebuild サブシステム全体が未接続 | `rebuildJob` の make_unique が src 全体に 0 件（big_bug §10-2 実測どおり）。:69-70, :104-105 の reset 呼び出しは rebuildJob==nullptr ガード付き → **休眠状態** |
| 3 | R-新規C | `runSynchronously` OOM 時サイレントリーク | bad_alloc 時のみ発症（OOM 限定）→ **P3** |
| 4 | R-新規D | `executePendingCommit` の「Message Thread のみ」コメント不整合 | 文書修正のみの課題 → **P3** |
| 5 | ~~BUG-065 残存~~ **解消済み（work92 B-7a）** | rt シャドウ直接書込は 2026-09-10 に全廃（両サイト）。「Non-RT → Audio Thread の通信は atomic publish のみ・rt シャドウは Audio Thread 専有」の不変条件が確立 |

---

## 5. 優先修正ロードマップ（提案）

```text
P0（着手推奨・ユーザー影響直結）
  1. big 1-1  nucHCMode/nucLCMode の getState/setState 追加（~10 行・pure addition）
  2. big 1-3  InputBitDepthTransform.h:114-115 を _mm256_storeu_pd に変更（2 行）
  3. big 1-6  MklFftEvaluator.h:270-271, 425-426 の IppStatus チェック追加（~8 行）

P1（RT 契約・将来リスク）
  4. big 1-7  emitRetireIntentRT の RT 分離実装 or リネーム（Finding 9 契約の履行）
  5. big 1-10 m_pendingIRChange のクリア遅延
  6. big 2-6  RCUReader の単調 ID 化
  7. big 2-10 StateIO.cpp:90 に範囲チェック追加（3 行）
  8. ~~BUG-065 残存~~ **実施済み（work92 B-7a・2026-09-10）**

P2（品質・保守）
  9. big 2-1  SoftClipPadéPolicy のリネーム（cppcheck 復帰のため）
  10. big 2-3 テスト矛盾条件の修正（1 行削除）
  11. big 2-2/2-4/2-5/3-6 の各品質修正
  12. ~~big 1-2 lastResortQueue_ の {} 値初期化~~ **実施済み（work92 C-9・2026-09-10）**
  13. BUG-044（MklFftEvaluator Rule of Five）の個別確認 — 唯一の未確定残置

P3 / 潜伏系（発火条件成立まで対応不要）
  R-新規A〜D は incremental rebuild 有効化時に対応
```

**修正着手時の注意**: 上記の多くは 2026-07-26 発見であり、その後の大規模改修（D101〜D179）で対象コードが変化している可能性があるため、着手前に現行 HEAD での該当箇所の実在性を本リストの「実測根拠」行番号で再確認すること。特に ISR 関連（big 1-2/1-7/1-10/2-6）は Closed boundary（T3c / Retire / Publish authority）に近く、**D159 通常開発ゲート**（scope 確定 → P0 invariant 影響確認 → implementation contract）を通すこと。

---

## 6. 元リストからの主要な判定変更サマリ

| 項目 | 元リスト | CORRECT 判定 | 変更理由 |
|---|---|---|---|
| BUG-011/012/013 | OPEN（未修正） | **修正済み** | `std::clamp` が CmaEsOptimizer.h:84 / Dynamic.h:29 に実装されていることを現行ソースで直接確認 |
| BUG-040 | 明記なし（未確認） | **バグではない（誤報）** | 3 段サンプルレートフォールバックが実装済み |
| BUG-026 | 要再確認 | **修正済み** | ObservedRuntime.h:49 に rootEnterSucceeded チェック実装 |
| BUG-025 | 明記なし | **修正済み** | enqueueWithRetry 結果受取＋quarantine 退避が両サイトに実装 |
| BUG-027 | 明記なし | **バグではない** | 競合相手の updateFade が Dead Code として 2026-07-28 削除済み |
| big 2-7 | CONFIRMED | **修正済み** | static_assert が :107, :215 に 2 箇所実装 |
| big 1-7 | CONFIRMED（P0 RT 違反） | **P1 に降格** | RT 到達経路なし＋Finding 9 契約コメントが実装側で明文化済み |
| big 3-9/3-10 | CONFIRMED | **P2（文書判断）** | icx の /fp:fast+/QxCORE-AVX2 は性能上の意図的選択（CMakeLists.txt:1585-1587 コメント） |
| big 1-2 | CONFIRMED | **P2 に降格** | 設計判断（producer 不実装）が big_bug §10-3-1 で確定済み |
| BUG-044 | 明記なし | **要個別確認（残置）** | = delete が本検証で確認できず、444c2f3 世代修正の直接確認が必要 |

---

## 7. 検証メタ

- 検証した BUG 番号: 56 ユニーク番号（011〜065・017 欠番・001-010 は対象外）
- 検証した big_bug 固有: 1-1〜1-10 / 2-1〜2-10 / 3-1〜3-10 / R-新規A〜E / R-9
- 使用ツール: ctx_batch_execute（42 コマンド）+ rtk(WSL) rg + grep/sed + 直接ファイル読み取り
- 検証日時点の authority: ConvoPeq.md `Generated: 2026-09-09 01:32:32` / NEWER_SRC_COUNT = 0
- 本リストは元リスト（INTEGRATED_BUG_LIST.md）を置き換えない。元リストは「3 ソースの統合記録」、本リストは「現行ソース照合による判定修正版」として併存
