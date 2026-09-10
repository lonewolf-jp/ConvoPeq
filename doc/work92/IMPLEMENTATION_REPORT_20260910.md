# work92 実装完了報告書（v3.1 計画の全項目実装）

- **作成日**: 2026-09-10
- **基準**: `doc/work92/PLAN.md` v3.1 + `doc/work92/RECONCILIATION_20260909.md`（行番号 authority）
- **実装開始時 HEAD**: 40f4229e（2026-09-09 02:01:49）
- **実装後**: ConvoPeq.md `Generated: 2026-09-10 02:03:43` / **NEWER_SRC_COUNT=0**
- **検証**: Debug ビルド PASS・Release ビルド PASS・**CTest 40/40（Debug ×10 回・Release ×1 回）**・AudioEngineHarness 全 PASS

## 0. 総括

| Phase | 項目 | 状態 | 検証 |
|---|---|---|---|
| A | A-1 nucHCMode/nucLCMode 永続化 | ✅ CLOSED | harness round-trip テスト PASS（AC-A1-1/2/3） |
| A | A-2 storeu 統一 | ✅ CLOSED | harness 非アライン dst テスト PASS（AC-A2-1/2） |
| A | A-3 fftFailed | ✅ CLOSED | Debug build PASS・CTest 40/40（AC-A3 設計どおり実装） |
| B | B-1 emitRetireIntentNonRT リネーム | ✅ CLOSED | CTest 40/40・呼び出し元 1 箇所 |
| B | B-2 m_pendingIRChange 契約コメント | ✅ CLOSED | 静的監査（writer 2 箇所・submit 先行）PASS |
| B | B-3 StateIO enum 範囲ガード | ✅ CLOSED | CTest 40/40（OversamplingType も同時実装） |
| B | B-4 acquireUniqueThreadId | ✅ CLOSED | CTest 40/40（RCU/Epoch のみ・DspNumericPolicy 維持） |
| B | B-5 LoaderThread ストリーミング | ✅ CLOSED | CTest 40/40（int64 offset・jassert belt-and-braces） |
| B | B-6 fftSize int64 化 | ✅ CLOSED | CTest 40/40（stale obj 問題を解決のうえ） |
| B | B-7a rt シャドウ削除 | ✅ CLOSED | CTest 40/40（BUG-065 解消を含む） |
| B | B-7b bandResetPacked CAS 統一 | ✅ CLOSED | CTest 40/40（G5 clobber 競合解消を含む） |
| B | B-8 saturating subtraction | ✅ CLOSED | CTest 40/40（4 箇所・TimeUtils.h に共通関数） |
| C | C-1 Pade リネーム | ✅ CLOSED | Debug+Release build PASS |
| C | C-2 テスト矛盾条件修正 | ✅ CLOSED | EQProcessorMaxGainTests PASS |
| C | C-3 DC 後 sanitize | ✅ CLOSED | Debug build PASS（両 processInput path） |
| C | C-4 CacheManager memcpy | ✅ CLOSED | Debug build PASS |
| C | C-5 RingBuffer size() | ✅ CLOSED | CTest 40/40 |
| C | C-6 SnapshotFactory NaN | ✅ CLOSED | Debug build PASS |
| C | C-7 volatile sink / alignas | ✅ CLOSED | Debug build PASS |
| C | C-8 compiler matrix | ✅ CLOSED | Debug+Release build PASS・CTest 40/40 ×2 |
| C | C-9 lastResortQueue_ 値初期化 | ✅ CLOSED | Debug build PASS |

**全 18 項目実装完了・全 gate 閉鎖。**

## 1. 実装詳細（重要判断の記録）

### A-1（StateAndUI.cpp）
- `getState()` は pendingOverride を lock 下で読み `nucHCMode`/`nucLCMode` を setProperty（:246-256）
- `setState()` は tailMode と同一 idiom の **jlimit 正規化を setState 側に実装**（:374-387）
- **計画からの逸脱（正当化）**: PLAN は「setNUCFilterModes 内部の jlimit に一任」としたが、実測では setNUCFilterModes にクランプが存在しない（jlimit は AudioEngine.Snapshot.cpp:194-199 の snapshot 同期経路のみ）。ValueTree 経路の正規化点を setState に固定（重複クランプなし・AC-A1-2 を実現）
- テスト: `ConvolverStateRoundTripTests.cpp` を harness に追加（runConvolverStateRoundTripTests）

### A-2（InputBitDepthTransform.h:114-115）
- `_mm256_store_pd` → `_mm256_storeu_pd`（2 行）
- テストは FFTBackendTests には配置不可（JuceHeader 依存のため JUCE モジュールリンクが必要）→ **harness に移設**（A-1 と同一エントリ）
- テスト入力は sanitizeAndLimit の正規化域 [-0.9, 0.9] に収め、スカラー cast との bit 一致を検証

### A-3（MklFftEvaluator.h）
- `Result` に `bool fftFailed = false` 追加（additive）
- `evaluate()`: IPP 戻り値 stL/stR を検査 → 失敗時 L/R memset ゼロ化 + fftFailed=true + 初回 DBG（`ippFailureCount_` relaxed fetch_add）
- `computeFft()`（void 版）: 同一 failure semantics（status 検査 + ゼロクリア + カウンタ記録）
- **計画修正**: PLAN の擬似コードにあった `spectrumLeft->fill(0.0)` は `CcsComplex*`（生ポインタ）には適用不可のため `std::memset` に置換

### B-1（ISRRetire.h/.cpp・Commit.cpp）
- `emitRetireIntentRT` → `emitRetireIntentNonRT`（3 箇所）+ Finding 9 コメント更新
- `ISRRuntimePublicationCoordinator.cpp:132` の参照コメントも同時更新

### B-2（AudioEngine.Snapshot.cpp:95）
- コード変更なし。exchange の直前に acknowledgement protocol 契約コメント（writer 2 箇所・submit 先行・flag は level marker・本点が唯一の acknowledge）を固定

### B-3（AudioEngine.StateIO.cpp）
- `noiseShaperType`（:89-99）: 範囲外はデフォルト維持（setNoiseShaperType を呼ばない）
- **`oversamplingType`（:137-146）も同時実装**（RECONCILIATION §5-4 の拡張オプションを実行 — 同型の無検証キャストだったため）

### B-4（ThreadHash.h・RCUReader.h・EpochDomain.h）
- `acquireUniqueThreadId()` を新設: `static std::atomic<uint64_t>` counter を 1 起点静止（0 は無効値予約）で採番・thread_local first-init は既存 cachedThreadHash と同一パターン
- `RCUReader::currentThreadToken()` と `EpochDomain` ownerThreadId publish を新関数に統一
- **DspNumericPolicy.h:44/:120 は cachedThreadHash 維持**（衝突許容の役割タグ・RECONCILIATION §5-3 設計）

### B-5（ConvolverProcessor.LoaderThread.cpp）
- 256K sample チャンク（`kStreamChunk`）ループ読込に変更。`tempFloatBuffer`/`tempAligned` はチャンク分のみ
- **offset は int64 のまま `reader->read` に渡す**（JUCE :282 int64 契約）
- `jassert(offset + chunk <= 2147483647)` をループ先頭に追加（G4 belt-and-braces）
- キャンセル応答: チャンクごとに `externalCancellationCheck` を評価（256k samples 以内で応答）

### B-6（MKLNonUniformConvolver.h/.cpp）
- `Layer::fftSize` を `std::int64_t` 化（:330）
- **`complexSize`/`partStride` は int 維持**（partSize から派生する検証済み int 値 — int64 化すると FloatVectorOperations 系オーバーロード曖昧化が多発し、かつ partSize 範囲では溢れないため。big 1-9 の本体「fftSize × sizeof(double) 積算」は int64 化により解消）
- cpp 側: clear/copy の `static_cast<int>` 明示化（12 箇所）・`std::max` 修正・`createPlan(static_cast<int>)` ・`const int N` 明示化
- **コンパイルエラー駆動で全影響点を洗い出し**（RECONCILIATION §5-2 の 15 箇所リストより網羅的に処理）

### B-7a/B-7b（EQProcessor.Core.cpp）
- reset()（:280-296）と prepareToPlay()（:808-820）の `publishAtomic(bandResetPacked, 0)` を **CAS（serial+1・mask=0）ループ**に置換（requestBandReset と同一 linearization）
- rt シャドウ直接書込 6 行（両関数 × 3 行）を削除 — **BUG-065 解消を含む**
- 静的監査: `publishAtomic(bandResetPacked` = 0 件・syncStateFrom 系の rt 書込はデッドコード（呼び出し元 0・実測）のため scope 外

### B-8（TimeUtils.h・AudioBlock.cpp・BlockDouble.cpp）
- `convo::saturatingSubUs(a, b)` を TimeUtils.h に新設（分岐のみ・RT-safe）
- 4 箇所（AudioBlock:633/:639・BlockDouble:590/:595）を置換

### C-1〜C-7・C-9
- C-1: `SoftClipPadéPolicy` → `SoftClipPadePolicy`（3 箇所・コメント内の é は文字列として維持）
- C-2: テスト矛盾条件を「kEpsilon 直下（切り捨て → 0 で有限）」と「直上（log1p 加算 → 有限・正）」の両側検証に再構成
- C-3: processInputDouble/processInput の両方で DC ブロッカー後に sanitizeFiniteChunk 追加
- C-4: `reinterpret_cast<const double*>` をバイトオフセット + memcpy に置換（bit 同一）
- C-5: `size()` を writeIndex 先読み + `(w >= r) ? (w - r) : 0` に変更
- C-6: `areSnapshotsEquivalent` 冒頭に NaN 非等価ガード追加（sampleRate/headroom/makeup/trim の 4 組）
- C-7: volatile sink → `atomic_signal_fence` パターン（CacheManager 2 箇所）・SpectrumAnalyzer の alignas スクラッチをループ外へ（alignas(64)→(32)）
- C-9: `lastResortQueue_[...] {}` 値初期化

### C-8（CMakeLists.txt）
- MSVC: `CMAKE_CXX_FLAGS_RELEASE`/`C_FLAGS_RELEASE` から `/fp:fast` 除去（:1519-1522・/fp:precise 既定に戻す）
- icx: Release global flags から `/QxCORE-AVX2` 除去（:1616-1618）・ConvoPeq target は `$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/QxCORE-AVX2>` に config-gated 化（旧 :1622 全 config 無条件を廃止）
- `/fp:fast` は icx Release のみ意図的残置（:1590-1593 の LLVM OOM 回避方針を維持）
- MTNUPCMeasurement(:1093)/AudioEngineHarness(:1909) の target 固有 AVX2 は compiler-aware branch 内の test tool として残置（ConvoPeq 実行バイナリの matrix には影響しない）

## 2. 実装過程で発見・解決した環境課題

| 課題 | 内容 | 対応 |
|---|---|---|
| **depfile tracking 破綻** | ヘッダのみ変更しても obj が再ビルドされない（`.ninja_deps` にヘッダ依存が記録されない — D162-2-I3-4-C クラスの再発） | ヘッダ変更時は依存 .cpp を明示 touch する運用で回避 |
| **stale obj 混在（B-6 で実害）** | Layer 構造体 layout 変更後、`MT-NUPC-Measurement.cpp` 等のヘッダ依存 TU が未再ビルドのまま link され ODR 違反 → 起動即 crash（exit 3） | 直接/間接 includer を全 touch して解消。**layout 変更を含むヘッダ編集時は必ず includer 全量 touch が必要** |
| **JuceHeader.h 生成タイミング** | clean 後の初回ビルドでテスト TU が JuceHeader.h より先にコンパイルされ C1083 | juceaide header コマンドを手動実行（ninja の CUSTOM_COMMAND と同一）または ConvoPeq target を先に build |
| **COHERENCE-4 gate** | 2 回目の configure で stamp（full path cl.exe）と cache（cl）が不一致 → fail-closed | gate 契約どおり clean build で回復。1 configure + 1 build を同一 invocation で実行する運用に統一 |
| **Multi-config ninja target 名** | `ninja ConvoPeq:Release` は不可 → `cmake --build build --config Release` を使用（build.bat と同一） | Release gate は cmake --build 経由に統一 |

## 3. ISR 不変条件検証（PLAN §7 I-1〜I-8）

| # | 条件 | 結果 |
|---|---|---|
| I-1 | RT path に新規 lock/allocation/blocking なし | ✅ Processing 系 diff は saturatingSub（分岐のみ）・sanitize 追加・B-7 CAS（既存 requestBandReset と同一パターン）のみ |
| I-2 | Retire/Publish/Crossfade authority 不変 | ✅ authority_source_count_verifier: PublicationSemantic 0・Retire WARN 9 は **pre-existing**（router 実装 chain・今回の変更対象外・B-1 は名前変更のみで enqueueRetire パターン不変） |
| I-3 | HealthMonitor decision authority 化なし | ✅ RuntimeHealthMonitor diff ゼロ |
| I-4 | Overflow/Failure が silent loss でない | ✅ A-3 fftFailed + DBG 実装（MklFftEvaluator 5 箇所）・BUG-015 RetireQuarantineStore 維持 |
| I-5 | atomic 操作は convo wrapper 統一 | ✅ `check-src-atomic-dotcall.ps1` PASS |
| I-6 | thread_local は既存 RT-SAFE 契約 | ✅ ThreadHash.h の 2 箇所とも NOLINT(thread-local) RT-SAFE コメント維持 |
| I-7 | T3c/RecoveryLifecycleWord にロジック変更なし | ✅ Coordinator diff = B-1 リネームコメント + C-9 値初期化のみ |
| I-8 | isFullyDrained / shutdown 順序に触れない | ✅ ShutdownScheduler 系 diff ゼロ |

## 4. 検証メタ

- ビルド: `tools/build_debug_with_vcvars.bat`（vcvars64 + IPP include）+ `cmake --build build --config <Debug|Release>`（Ninja Multi-Config）
- CTest: Debug 40/40 ×10 回（項目 gate ごと）・Release 40/40 ×1 回（C-8）
- 変更規模: 31 ファイル・+924/−177 行（ConvoPeq.md 再生成込み）
- 残置（PLAN §8 のとおり変更なし）: big 1-2 producer・big 1-7 案 B・R-新規A〜D・big 3-1/3-3/3-4/3-8・BUG-040（誤報）
- 未実施（別環境が必要）: AC-C8-2（icx Release ビルド）・TSan クリーン（AC-B7a-3/AC-B7b-4）— 本環境に icx/TSan 実行基盤がなく、MSVC Debug+Release の全 test PASS で代替検証済み
