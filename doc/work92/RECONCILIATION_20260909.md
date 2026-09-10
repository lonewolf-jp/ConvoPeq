# work92 計画 v3 未確定事項の確定報告書（RECONCILIATION）

- **作成日**: 2026-09-09
- **検証基準**: 現行 HEAD **40f4229e**（2026-09-09 02:01:49）/ ConvoPeq.md `Generated: 2026-09-09 01:32:32` / **NEWER_SRC_COUNT=0**（src・CMakeLists・build.bat はスナップより新しくない＝全実測はスナップ世代と一致）
- **対象**: `doc/work92/PLAN.md` v3（gate 完了版）の全項目 + `CORRECT_INTEGRATED_BUG_LIST.md` 残存 11 項目の HEAD 再照合
- **方法**: ctx_batch_execute / ctx_execute + WSL(rg/grep/sed) + Windows find/grep による全サイト実測。Read/Edit は本報告書作成のみ。

## 0. 総括

| 判定 | 件数 | 内訳 |
|---|---|---|
| ✅ 計画どおり実装可能（設計確定） | 15 項目 | A-1, A-2, B-1, B-2, B-3, B-4, B-5, B-6, B-7a, B-7b, B-8, C-1〜C-7, C-9 |
| ⚠️ 計画の行番号ズレあり → 本報告書で訂正 | 3 項目 | B-8（:624/:630/:664 → **:632/:638**）、C-7（CacheManager :203/:241 → **:203/:243 で実質正**）、B-6（MKLNonUniformConvolver.h:330 は正・cpp 側 :843/:847 → **:841/:845/:893/:894/:906/:907 等**） |
| ✅ HEAD 進行による判定変化なし | — | 54ba7b40 → 40f4229e の src 差分は diagFootprintCaptured plain-bool 化・NOLINT 追加・コメントのみ。全バグ判定は不変 |
| 📌 新発見（設計影響あり） | 2 件 | ①C-8 の icx /QxCORE-AVX2 は `CMAKE_CXX_FLAGS_RELEASE`（:1606/:1607）に加え **target 固有指定 :1622 にも存在**（global 除去のみでは不完全・:1622 も除去要）。②B-4 の token は DspNumericPolicy.h にも漏出（:44, :120）— PLAN §7 静的監査パターン `src/core/` では拾えない |

## 1. Phase A 項目の確定

### A-1. big 1-1 nucHCMode/nucLCMode 永続化 — ✅ 確定（実装可能）
- **実測**: `ConvolverProcessor.StateAndUI.cpp` の `getState()`(:202-) setProperty は :205〜:245 で **nucHCMode/nucLCMode は 1 箇所も存在しない**（grep 0 件）。`setState()`(:289-) の :344-362 も読込なし。
- 一方 runtime 同期系は存在: Snapshot.cpp:55-56（ハッシュ）・:142-143（snapshot）・:194-199（jlimit）・StateAndUI.cpp:816-838（`setNUCFilterModes` — changed 判定 + `postCoalescedChangeNotification()` 内蔵）。
- **設計確定**: PLAN v3 どおり。`setState()` への追加は `setNUCFilterModes()` 1 呼び出しで coalesce 動作込み・jlimit 正規化は Snapshot.cpp:194-199 側が担当するため StateAndUI 側に重複クランプ不要を再確認。
- 追加根拠: `getState()` は `setNUCFilterModes` に対応する getter 経路（`pendingOverride` 経由 snapshot）が既にあり、A-1 は ValueTree ↔ snapshot の往復追加のみで authority 不変。

### A-2. big 1-3 store_pd 統一 — ✅ 確定（行番号訂正あり）
- **実測**:
  - `InputBitDepthTransform.h`: :60/:81 は `_mm256_storeu_pd`（修正済み）・**:114-115 のみ `_mm256_store_pd` 残存**（契約なし dst）— PLAN 記載どおり。
  - `MKLNonUniformConvolver.cpp:1405-1408`: `_mm256_load_pd` + `_mm256_store_pd` とも `l.accumBuf`（mkl_malloc 64-byte、:906-907 系の確保）→ **契約済み・修正不要**を確認。
  - `MKLNonUniformConvolver.cpp:1668`: **aligned/unaligned を ptr 下位 5 ビットで実測分岐**（:1660-1662 の `aligned` フラグ）→ store_pd は `aligned==true` のときのみ実行 → **契約済み・修正不要**。
- **結論**: 修正は `InputBitDepthTransform.h:114-115` の 2 行のみで確定。MKL 側 2 箇所は安全設計が既に成立している（PLAN v1 の「修正不要」判断を本実測で再確定）。

### A-3. big 1-6 IPP FFT 戻り値無視 — ✅ 確定（fftFailed 方式）
- **実測**:
  - `MklFftEvaluator.h:40-47` `struct Result` — **fftFailed フィールドは未実装**（5 double のみ）。実装対象のまま。
  - `:248-256` evaluate() 冒頭 guard — `fftSpec == nullptr || ...` で `return Result{}`。**既存 failure semantics = 「FFT 利用不可 → ゼロ結果」が確立済み**（[Bug 2/3 fix] コメント実在）。
  - `:270-271` — `ippsFFTFwd_RToCCS_64f` 戻り値無視のまま。status キャプチャはゼロ。
  - `:410` `void computeFft(...)` — **void 版の :425-426 も status 無視**。呼び出し元は NoiseShaperLearner.cpp:1346 の 1 箇所のみ（evaluate）。
  - BUG-044（copy/move delete）: `:138-141` に **`= delete` 4 連が実在** → 解消済みを再確認。
- **結論**: PLAN v3 の fftFailed 追加（additive）+ status キャプチャ + zero-both + 初回 DBG は現行コード構造にそのまま実装可能。void 版は「status チェック + メンバ失敗記録 + memset ゼロ」で同一方針。G2 結論（zero-both が唯一整合）と矛盾なし。

## 2. Phase B 項目の確定

### B-1. emitRetireIntentRT リネーム — ✅ 確定（影響範囲 1 呼び出し元のみ）
- **実測**: `emitRetireIntentRT` は宣言 `ISRRetire.h:59`・定義 `ISRRetire.cpp:94`・呼び出し `AudioEngine.Commit.cpp:485` の **3 箇所のみ**（コメント 1 件 ISRRuntimePublicationCoordinator.cpp:132）。
- **結論**: リネーム影響は実質 1 呼び出し元。機械的置換で Low リスク確定。

### B-2. m_pendingIRChange — ✅ 確定（コード変更なし・V-B2-1 完了済み）
- **実測**: writer は `AudioEngine.h:1522-1524`（setIRChangeFlag 定義）への呼び出し **2 箇所のみ**（Timer.cpp:800 / UIEvents.cpp:177）。両者とも直前のブロックで `submitRebuildIntent(Structural)`（Timer :795・UIEvents :162）が先行し、**flag → intent の順序は submit 先行で不変**。
- consumer は Parameters.cpp:381/:452（ consume 判定）・Timer.cpp:849（観測）・Snapshot.cpp:95（**唯一の acknowledge 点** `exchangeAtomic(false, acq_rel)`）。
- **結論**: 元バグ指摘（big 1-10）は現行順序で不成立 — PLAN §2 の判定を HEAD 進行後も維持。実装は **プロトコルを文書化するコメント追加のみ**（Snapshot.cpp:95 に「exchange は acknowledge であり、submit は必ず先行済み」という契約を固定）。コード動作変更ゼロ。

### B-3. StateIO enum 範囲チェック — ✅ 確定（防御層ゼロを確認）
- **実測**: `AudioEngine.StateIO.cpp:90` `setNoiseShaperType((NoiseShaperType)(int)state.getProperty(...))` — キャスト時に範囲チェックなし。
- `setNoiseShaperType` 実装（AudioEngine.Parameters.cpp:405-）は **jlimit/clamp なし**で `noiseShaperType` に直接 publish（比較・分岐はあるが値域正規化はゼロ）。**:120 の setOversamplingType も同型の無検証キャスト**（PLAN が scope 外とした隣接箇所・同じ pattern）。
- **結論**: PLAN どおり :90 に範囲ガード（~6 行）追加で確定。**設計判断 1 点**: OversamplingType(:120) は同種の無検証キャストのため、同一 commit で同様の範囲ガードを入れるかどうかを実装時に判定（推奨: 同時修正・ただし PLAN 変更は「B-3 の拡張オプション」として記録のみ・scope 変更はユーザー gate）。

### B-4. thread token 単調 ID 化 — ✅ 確定（漏出点 1 件追加・静的監査パターン訂正）
- **実測**:
  - `ThreadHash.h:9-16` `cachedThreadHash()` — `std::hash<std::thread::id>` thread_local キャッシュのまま（置換対象の現状どおり）。
  - token 消費点: `RCUReader.h:47/:127/:150-152`（`currentThreadToken()`）・`EpochDomain.h:75`（ownerThreadId の publish 源）。
  - **PLAN にない消費点を 1 件発見**: `src/DspNumericPolicy.h:44`（`detail::currentThreadTag()`）・`:120`（`isAudioThread()`）が `convo::cachedThreadHash()` を直接呼ぶ。これらは RCU/Epoch の token と**同一の ID 系を共有**しており、B-4 で ThreadHash を置換する場合、**DspNumericPolicy 側は置換対象か・ハッシュ継続かを決める必要がある**。
- **設計確定（v3.1 提案）**: `acquireUniqueThreadId()` を ThreadHash.h に追加し、RCUReader/EpochDomain の ownerThreadId 系を新 ID に統一。**DspNumericPolicy の tag は「スレッド一意性が目的・衝突許容の役割タグ」なので cachedThreadHash のまま維持**（isAudioThread の slot tag 照合は登録時の値と比較するだけのため単調 ID でもハッシュでも動くが、既存動作の bit 変化を避けるため維持）。置換対象は `src/core/` の token 系のみ。
- **PLAN §7 静的監査コマンド訂正**: `rg -n "currentThreadToken\|cachedThreadHash\|acquireUniqueThreadId" src/core/` → **`src/` 全域**に拡張（DspNumericPolicy.h が src/ 直下のため）。判定基準も「RCU/Epoch 系 token のみ acquireUniqueThreadId・DspNumericPolicy 系は cachedThreadHash 維持」に修正。

### B-5. LoaderThread ストリーミング読込 — ✅ 確定（JUCE 署名を本 HEAD JUCE で確認）
- **実測**:
  - `ConvolverProcessor.LoaderThread.cpp:448-485` — 現行は一括確保: `tempFloatBuffer(numChannels, static_cast<int>(fileLength))` + `tempAligned(fileLength)` + `reader->read(&tempFloatBuffer, 0, static_cast<int>(fileLength), 0, ...)`。ストリーミング化は未実装（対象のまま）。
  - **JUCE 署名確認（本リポ JUCE/modules/juce_audio_formats/format/juce_AudioFormatReader.h）**: `:282` `virtual bool read(... int64 startSampleInFile, int numSamples)`・`:319` `readSamples(..., int startOffsetInDestBuffer, int64 startSampleInFile, ...)` — **startSampleInFile は int64** を本ツリーで実確認（PLAN G4 の前提確定）。
  - `MAX_FILE_LENGTH = 2147483647`（:450）は samples 限定（:454 メッセージ確認）。
- **結論**: PLAN v3 のチャンク設計（int64 offset を reader にそのまま渡す・copyFrom の int 位置は維持）で確定。`kStreamChunk` ループへの jassert 追加も既存ヘッダ文脈でそのまま可能。

### B-6. fftSize int64 化 — ✅ 確定（影響点を列挙・行番号訂正）
- **実測**: `MKLNonUniformConvolver.h:330` `int fftSize = 0;`（Layer 構造体）・:335 `complexSize = fftSize / 2 + 1`（int）。
- **cpp 側の消費点（置換影響リスト）**: :297（初期化 0）、:355（`const int N = l.fftSize`）、:778-781（設定）、:792/:797（plan・log）、:841/:845（`DIAG_MKL_MALLOC(l.fftSize * sizeof(double), 64)` — **int × size_t の積がここ**）、:843/:847（allocSizes 記録）、:893/:894（clear）、:906/:907（tempTime/tempFreq mkl_malloc + `(l.fftSize + 2)`）。
- **結論**: `fftSize` → `int64_t` 変更で、:355 の `const int N`・juce::String 変換・mkl_malloc 呼び出しの整合を同時修正する必要。PLAN 記載の「:843/:847/:853 系」より範囲が広い → **上記 15 箇所を修正チェックリストとして B-6 実装時に全数消す**（コンパイルエラー駆動で抜けなしにできる）。

### B-7a. reset/prepareToPlay の rt シャドウ直接書込削除 — ✅ 確定
- **実測**: `EQProcessor.Core.cpp` reset() 側 **:282-284**（`rtDeferredBandResetMask.store(0, relaxed)` / `rtSeenBandResetSerial = 0` / `rtSeenAgcResetSerial = 0`）・prepareToPlay() 側 **:793-795**（同 3 行）。先行 `fetchAddAtomic(agcResetSerial, 1, acq_rel)` は :281・:792 に実在。
- consumer（Audio Thread）: `EQProcessor.Processing.cpp:585-603` — agcSerial/bandSerial とも `!= rtSeen` で shadow 自己更新 → Non-RT 事前書込は不要（PLAN の根拠どおり）。
- **結論**: 3 行 × 2 箇所の削除のみで確定。削除対象行は本実測行番号（:282-284・:793-795）。

### B-7b. bandResetPacked serial protocol — ✅ 確定（G5 全容を HEAD で再実測）
- **実測**:
  - writer 3 系統: `EQProcessor.h:531-551` `requestBandReset()`（CAS ループ: serial+1・mask OR・acq_rel/acquire）・`Core.cpp:280` reset()（`publishAtomic(bandResetPacked, 0, release)`）・`Core.cpp:791` prepareToPlay()（同）。
  - reader: `Core.cpp:590/:650`（syncStateFrom/syncGlobalStateFrom — consume のみ）・`Processing.cpp:595-601` と `:1079-1085`（Audio Thread consumer — serial 進行検知 → rtSeen 更新 → `rtDeferredBandResetMask.fetch_or(mask)`）。
  - **consumer は serial-advance+mask=0 を fetch_or(0)=no-op で吸収** → B-7b で consumer 無変更の前提が HEAD で成立。
  - `EQProcessor.h:515-529` に `bandResetSerialFromPacked` / `bandResetMaskFromPacked` / `makeBandResetPacked` ヘルパー実在 → PLAN の CAS 置換コードは既存ヘルパーで書ける。
  - 呼び出し元: Parameters.cpp:89/:169/:192（per-band）・`requestAllBandReset`（:558 = 0xFFFFFFFF）。
- **結論**: PLAN v3 の置換コード（consume→CAS ループ serial+1/mask=0）で確定。G5 の pre-existing clobber 競合（並行 requestBandReset の mask 消失）も本実測で writer 構造から再確認。

### B-8. saturating subtraction — ✅ 確定（行番号訂正）
- **実測**: 減算 + uint32_t キャストは **4 箇所のみ**:
  - `AudioEngine.Processing.AudioBlock.cpp:632`（`nowUs - cbStartUs`）・**:638**（`cbStartUs - cbPrevEndUs` — PLAN の :630 はズレ・実際は 638）
  - `AudioEngine.Processing.BlockDouble.cpp:589`・**:594**
- いずれも `cbStartUs != kNeverStartedUs` / `cbPrevEndUs > 0` のガードありだが、逆転時の wrap は未防衛。
- **結論**: 修正対象は上記 4 箇所（PLAN の :624/:630/:664 → 訂正 :632/:638）。saturating 減算ユーティリティの導入位置は BlockDouble/AudioBlock 共通で使える場所（convo 名前空間系）が適切。

## 3. Phase C 項目の確定

| 項目 | HEAD 実測 | 判定 |
|---|---|---|
| **C-1** SoftClipPadéPolicy | `FastTanhApprox.h:63` 実在・使用 `DSPCoreDouble.cpp:127/:191`（2 箇所） | ✅ 確定（リネーム 3 箇所） |
| **C-2** テスト矛盾条件 | `src/tests/EQProcessorMaxGainTests.cpp:355-358` — `for (delta = 1e-15; delta < 1e-6; ...)` 内 `if (delta > 1e-6)` が同時に真になり得ず logBound 常に 0.0 | ✅ 確定（意図は「切り捨て条件と同じ挙動の検証」だが分岐不成立・修正時に `delta >= 1e-8` など実効的な比較へ要改訂） |
| **C-3** DC 後 sanitize | `DSPCoreIO.cpp` の sanitizeFiniteChunk は入力前のみ（実測: 既存 guard は :231-232/:282-283 系） | ✅ 確定 |
| **C-4** CacheManager aliasing | `CacheManager.cpp:267` `reinterpret_cast<const double*>(tdStart)` 残存 | ✅ 確定 |
| **C-5** RingBuffer size() | `LockFreeRingBuffer.h:76-81` — `w = consume(writeIndex, acquire)` → `r = consume(readIndex, acquire)` の順で `w - r` 返却（read 後 read で producer が w を進めると r が古く負になり得る）。**ただし関数直下のコメント「スレッドセーフではない・停止後のみ呼ぶ」が実在** → 本質は best-effort ユーティリティ | ✅ 確定（writeIndex 先読み方式への変更は既存契約コメントと整合・AC は「負値なし」） |
| **C-6** SnapshotFactory NaN | `areSnapshotsEquivalent`（:46-）は `std::abs(...) > 1e-9/1e-12` 比較のみ・**std::isnan 0 件** → NaN 同士は `NaN > eps == false` で等価と誤判定 | ✅ 確定 |
| **C-7** volatile sink / alignas | `CacheManager.cpp:203`・**:243**（PLAN :241 は微ズレ）に volatile sink warm-up 2 箇所・`SpectrumAnalyzerComponent.cpp:474` に `alignas(64) float mags[8]` がループ内相当 | ✅ 確定（行番号 :203/:243 に訂正） |
| **C-8** compiler matrix | **icx branch（:1590-1630）**: `CMAKE_CXX_FLAGS_RELEASE`（:1606/:1607）に `/QxCORE-AVX2 /fp:fast`。**加えて :1622 `target_compile_options(ConvoPeq PRIVATE /QxCORE-AVX2)`（全コンフィグ）が target 固有に存在**。MSVC branch: `:1519-1520` に `/fp:fast`（/arch:AVX2 は :1542 target 固有・global には無し = PLAN の (1) は :1519-1520 の fp:fast 除去のみで正）。IntelLLVM 判定ガードは :79/:93/:637/:849/:897/:954/:962/:1085 に既存 | ⚠️ **修正設計を 1 点更新**: icx の (2) global 除去に加え **:1622 の target 固有 /QxCORE-AVX2（無条件・全 config）を config-gated に置換**する必要（現行は全 config で AVX2 指定 → 除去漏れがあると Debug も AVX2 要求のまま）。/QxCORE-AVX2 target 固有行（:1093 MTNUPCMeasurement・:1892 AudioEngineHarness）は test/measurement ツールであり scope 外維持 |
| **C-9** lastResortQueue_ 値初期化 | `ISRRuntimePublicationCoordinator.h:1016` — `RetireOverflowEntry lastResortQueue_[4096];` **無初期化**。要素は `RetireIntent intent + uint64 overflowTimestampUs + uint16 reinjectRetryCount`（ISRRetireOverflowRing.h:42-48）。Coordinator 使用箇所（.cpp:386/:402/:405）は `lastResortCount_` 越アクセスなしで論理的に未初期化領域を読まない構造だが、Debug ビルド CDCD パターン観測リスクが元指摘 | ✅ 確定（`RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity] {};`（値初期化）で解消・文言修正どおり Low） |

## 4. 棚卸し（CORRECT_INTEGRATED_BUG_LIST 残存 11 項目）の HEAD 再判定

| # | 項目 | HEAD 40f4229e 再判定 |
|---|---|---|
| 1 | big 1-1 nuc 永続化 | **残存確認**（A-1 で実測）→ Phase A-1 として実装 |
| 2 | big 1-3 store_pd | **残存確認**（:114-115 のみ）→ Phase A-2 |
| 3 | big 1-6 FFT status | **残存確認**（A-3 で実測）→ Phase A-3 |
| 4 | big 1-7 emitRetireIntentRT | **残存**（P1 リネーム）→ Phase B-1 |
| 5 | big 1-8 LoaderThread | **残存**（B-5 で実測: 一括確保のまま）→ Phase B-5 |
| 6 | big 1-9 fftSize | **残存**（B-6 で実測: int のまま）→ Phase B-6 |
| 7 | big 1-10 m_pendingIRChange | **不成立→コメント文書化のみ**（B-2 で実測） |
| 8 | big 2-6 thread token | **残存**（B-4 で実測: hash 由来のまま）→ Phase B-4 |
| 9 | big 2-9 timing underflow | **残存**（B-8 で実測: 4 箇所）→ Phase B-8 |
| 10 | big 2-10 NoiseShaperType | **残存**（B-3 で実測: ガードなし）→ Phase B-3 |
| 11 | BUG-065 rtSeenAgcResetSerial | **残存（潜伏）**（B-7a で実測: :284/:795 直書込のまま）→ Phase B-7a が削除対象 |

- **BUG-044**: `MklFftEvaluator.h:138-141` `= delete` 4 連実在 → **解消済みを再確認**。
- **OPEN 3（BUG-011/012/013 sigma）**: `CmaEsOptimizer.h:85` `std::clamp(inSigma, params.sigmaMin, params.sigmaMax)` 実在 → **修正済み判定を維持**（台帳どおり OPEN 記載が誤り・CORRECT 版の訂正は正）。
- **Phase C 対象**（C-1〜C-7・C-9 = 残存 11 のうち P2 群）: 上表 §3 のとおり全件実装可能。

## 5. PLAN v3.1 への反映事項（確定済み訂正リスト）

1. **B-8**: 対象行を `AudioBlock.cpp:632/:638` + `BlockDouble.cpp:589/:594` に訂正（旧 :624/:630/:664）。
2. **B-6**: cpp 側影響リストを 15 箇所で明記（:297/:355/:778-781/:792/:797/:841/:843/:845/:847/:893/:894/:906/:907）。コンパイルエラー駆動チェックリスト化。
3. **B-4**: 静的監査パターンを `src/` 全域に拡張 + **DspNumericPolicy.h:44/:120 は cachedThreadHash 維持**の設計判断を明記（RCU/Epoch token 系のみ acquireUniqueThreadId へ統一）。
4. **B-3**: OversamplingType(:120) が同型の無検証キャストであることを記録（同時修正は B-3 拡張オプションとしてユーザー gate）。
5. **C-8**: icx branch には **:1622 の target 固有 `/QxCORE-AVX2`（全 config 無条件）** も存在 → 「global 除去」設計に「:1622 の config-gated 化」を追加。MSVC 側は :1519-1520 の `/fp:fast` 除去のみで正（/arch:AVX2 は :1542 target 固有）。
6. **C-7**: 行番号 :203/:243 に訂正。
7. **B-7a/B-7b**: 削除行を :282-284・:793-795 に確定（旧 :284 付近/:795 付近より精緻化）。
8. **B-2**: Snapshot.cpp:95 に契約コメント 1 件（実装タスクの全内容）。
9. **全項目共通**: 対象ファイルの実測行番号は本報告書 §1〜§4 を authority とする（ConvoPeq.md 01:32:32 = HEAD 40f4229e と一致）。

## 6. 実装着手の確定順序（PLAN §3 どおり・確定版）

```text
B-3 (+: OversamplingType 判定) → B-8 (4 箇所訂正版) → B-1 (1 呼び出し元) → B-6 (15 箇所リスト)
   ↓
B-4 (DspNumericPolicy 維持を含む設計確定版)
   ↓
B-7a (:282-284/:793-795 削除)
   ↓
B-2 (Snapshot.cpp:95 コメントのみ)
   ↓
B-7b (CAS 統一・consumer 無変更)
   ↓
B-5 (int64 reader offset・G4 proof 済)
   ↓
A-1 → A-2 → A-3 (Phase A は B 系より実害が大きいため A-1 からでも可 — PLAN では Phase A 先頭)
   ↓
C-1 → C-2 → C-4 → C-5 → C-6 → C-7 → C-9 → C-3 → C-8 (最後・ビルド設定変更)
```

> **注意**: PLAN v3 の Phase A（A-1/A-2/A-3 = P0）は実害順位が最上位のため、ユーザー推奨の「B-3 単純項目から順に gate を閉じる」進行であっても A 系は Phase B 完了後必ず着手する。順序の逆転（B 先行）は gate 閉鎖の遅れを伴わない範囲で許容（A/B 系はファイル依存なし・gate は独立）。

## 7. 検証メタ

- 検証日: 2026-09-09 / HEAD 40f4229e（54ba7b40 との src 差分 = diag flag plain bool 化 + NOLINT + コメントのみ → バグ判定不変）
- 使用ツール: ctx_batch_execute（26 コマンド超）+ ctx_execute（shell・WSL rg）+ Windows grep/find/sed — 全実測は ConvoPeq.md `Generated: 2026-09-09 01:32:32` 世代
- 本報告書は PLAN.md v3 を置き換えない。**v3 の設計判断は全項目維持**。本報告書は「行番号・影響範囲・設計補完」の確定層。
- 残る未確定事項: **ゼロ**（B-3 の OversamplingType 同時修正可否のみユーザー gate として残置）
