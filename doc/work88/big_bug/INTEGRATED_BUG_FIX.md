# ConvoPeq 統合バグ改修計画書 (INTEGRATED_BUG_FIX)

**作成日**: 2026-08-11
**最終更新**: 2026-08-11（別視点調査・サブエージェント 3 系統で全 36 項目を精緻化。P0 バグ 6 件の修正コードをソース検証、設計判断バグの前提を確定（1-7 は RT 違反未顕在化で P0→P2、1-9 は到達不能で P0→設計判断、2-6 は MSVC 衝突不可能、2-7 は MSVC spinlock 実装、3-1 は同一スレッドで競合なし）。最終別視点調査（2026-08-11）で 13 セクションの実装詳細を確定：1-1 は pendingOverride 読取 + jlimit Soft/Soft、1-6 は Result{}/void 戻り（return nullptr 誤り修正）、2-2 は UltraHighRateDCBlocker 維持 + DC 後スクラブ、2-3 は check(ubDb>0) 追加、2-4 はチャネル単位 memcpy、2-5 はクランプ防衛、2-9 は BlockDouble 行番号修正 + intervalMs 第4サイト、3-4 は #if AVX2 直後 jassert、3-5 は #include <atomic> 必須 + accumulator 維持、3-7 は alignas(32) 移動、R-新規B は休眠維持（Option 1）、R-新規C は runSynchronously 優先）
**基盤文書**: `doc/work88/big_bug/INTEGRATED_BUG_LIST.md`（§1-3 有効バグ / §7 詳細設計 / §10 別視点調査確定事項）
**対象**: ConvoPeq (Windows 11 x64, AVX2, MSVC/icx, JUCE 8.0.12, MKL/IPP)
**改修方針**: Practical Stable ISR Bridge Runtime の観点から、未修正（✅ Confirmed）の有効バグを優先度順に改修する。§10 で「現状維持」と確定されたものは改修対象外とし、方針のみ記録する。

---

## 1. 概要

### 1.1 改修対象の分類

| 分類 | 意味 | 対象バグ |
|---|---|---|
| **即時改修（実施確定）** | 低リスク・明確な修正方針・§10/別視点調査で確定 | 1-1, 1-2(値初期化), 1-3, 1-4(統合NO-OP), 1-5, 1-6, 1-7(リネーム), 1-8(ストリーミング), 1-10(クリア移動), 2-1, 2-2, 2-3, 2-4, 2-5, 2-6(堅牢性), 2-9, 2-10, 3-4, 3-5, 3-6(両面), 3-7, 3-9, 3-10, R-9, R-新規A, R-新規C, R-新規D |
| **設計判断（方針確定）** | 修正方針を確定済み（休眠維持・保留を含む） | 1-9(最小修正), 2-7(案A/B/C), 3-1(防御的), R-新規B(休眠維持) |
| **現状維持（§10/別視点調査確定）** | 調査の結果、現状が正しい/実害なし | 2-8, 3-2, 3-3, 3-8, R-新規E |

### 1.2 優先度（別視点調査で精緻化）

| 優先度 | バグ | 根拠 |
|---|---|---|
| **P0** | 1-3（_mm256_store_pd アライメント） | 即落ちクラッシュ（修正は InputBitDepthTransform.h の 1 サイトのみ）|
| **P0** | 1-1（nucHCMode/nucLCMode 未永続化） | ユーザーデータ消失 |
| **P0** | 1-6（IPP FFT 戻り値無視） | 無音/NaN爆発 |
| **P0** | 1-8（LoaderThread OOM） | クラッシュ（ピーク約67GB。ストリーミング化で削減）|
| **P1** | 1-2（coordinatorDeferredRing_ 値初期化）, 1-4, 1-10, 2-9, 2-10 | Retire ドロップ/保守リスク |
| **P1→P2** | **1-7（ISRRetire Mutex）** | **RT 違反未顕在化（呼び出し元は NonRT のみ）。リネーム + 将来保証に下方修正** |
| **P1→設計判断** | **1-9（int/size_t 混在）** | **int オーバーフロー到達不能（防御的修正）。割当サイト cast + :778 乗算順序に限定** |
| **P2** | 2-1〜2-8, 1-5, 2-6, R-新規A | ツール互換/NaN/テスト/移植性（2-6 は MSVC 衝突不可能の堅牢性改善）|
| **P3** | 3-1〜3-10, R-9, R-新規B〜E | 将来の拡張性/品質（3-1 は同一スレッドで競合なしの防御的強化、3-9/3-10/R-9 は実施確定）|

---

## 2. 改修対象バグ一覧（有効・未修正）

### 2.1 即時改修（実施確定）

| # | バグ | ファイル | 重要度 | 対応 |
|---|---|---|---|---|
| 1-1 | `nucHCMode`/`nucLCMode` がセッション永続化から欠落 | ConvolverProcessor.StateAndUI.cpp | High | getState/setState に追加 |
| 1-2 | `coordinatorDeferredRing_`/`lastResortQueue_` デッドコード（値初期化のみ） | ISRRuntimePublicationCoordinator.h | High | `lastResortQueue_` に `{}` 初期化 |
| 1-3 | `_mm256_store_pd` アライメント保証なし | InputBitDepthTransform.h 他4箇所 | Critical | `_mm256_storeu_pd` 化 or 契約明示 |
| 1-5 | `musicalSoftClip` デッドコード | AudioEngine.h / DSPCoreIO.cpp | Low | クラスメソッド削除 |
| 1-6 | IPP FFT 戻り値無視 | MklFftEvaluator.h:270-271,425-426 | Medium | 戻り値チェック |
| 2-1 | 非ASCII識別子 `SoftClipPadéPolicy` | FastTanhApprox.h:63 | Low | `SoftClipPadeApproxPolicy` にリネーム |
| 2-2 | 入力 DC ブロッカー NaN/Inf 非対称 | DSPCoreIO.cpp:231,252,305 | Low | 入力側 DC 後スクラブ追加（§3-12）|
| 2-3 | ユニットテストの矛盾条件 | EQProcessorMaxGainTests.cpp:355-358 | Low | 内側 if 削除 |
| 2-4 | strict-aliasing 違反 | CacheManager.cpp:267 | Low | memcpy 化 |
| 2-5 | LockFreeRingBuffer::size() データ競合 | LockFreeRingBuffer.h:76 | Medium | クランプ + コメント明記（§3-15）|
| 2-9 | タイミング計算の uint64 underflow | AudioBlock.cpp / BlockDouble.cpp | Medium | saturating subtraction |
| 2-10 | `NoiseShaperType` enum キャスト検証欠如 | StateIO.cpp:90 | Medium | 範囲チェック |
| 3-4 | MKL アライメント判定競合 | MKLNonUniformConvolver.cpp:1571 | Medium | 入口アサート |
| 3-5 | `volatile sink` | CacheManager.cpp:203,241 | Medium | atomic_signal_fence 化 |
| 3-7 | `alignas` ループ内配置 | SpectrumAnalyzerComponent.cpp:474 | Low | ループ外 + alignas(32) |
| R-新規A | `IncrementalRebuildJob::reset()` NUC リーク | ConvolverProcessor.Rebuild.cpp:110-134 | P2 | `retireStereoConvolver` に変更 |
| R-新規C | bad_alloc 時リーク（OOM） | LoaderThread.cpp:367 | P3 | runSynchronously を try/catch + retire 化（§3-34）|
| R-新規D | `executePendingCommit` コメント不整合 | LoadPipeline.cpp:766 | P3 | コメント修正 |

### 2.2 設計判断（確定済み・一部要精査）

| # | バグ | ファイル | 重要度 | 設計判断点 |
|---|---|---|---|---|
| 1-4 | `fastTanh` 3箇所独立複製 | DSPCoreFloat.cpp:146 / DSPCoreIO.cpp:76 | Medium | **確定（§3-4）**: 統合は NO-OP（係数ビット一致）。DSPCoreFloat/IO の fastTanh を SoftClipPadeApproxPolicy に統一 |
| 1-7 | ISRRetire Mutex | ISRRetire.cpp:94 | High | **確定（§3-7）**: リネーム + コメント強化。overflowRing ロックフリー化は将来 |
| 1-8 | LoaderThread OOM | LoaderThread.cpp:463 | High | **確定（§3-8）**: ストリーミング読み込み化 + MAX_FILE_LENGTH 維持 |
| 1-9 | int/size_t 混在 | MKLNonUniformConvolver.cpp:843 | High | **確定（§3-9）**: 割当サイト cast + :778 乗算順序修正（最小修正） |
| 1-10 | `m_pendingIRChange` 公開前クリア | Snapshot.cpp:95 | High | **確定（§3-10）**: クリアタイミング遅延（現状記載の技術的誤りを修正） |
| 2-6 | RCUReader ハッシュ衝突 | RCUReader.h:51,152 | High | **確定（§3-16）**: 前提不成立（severity 下方）。堅牢性改善のみ |
| 2-7 | atomic<DSPHandle> ロックフリー検証 | ISRDSPHandle.h:186 | Medium | **要精査（§3-17）**: Release でも abort 検証（案A/B/C 未決定） |
| 3-1 | AudioSegmentBuffer リングラップ競合 | AudioSegmentBuffer.h:50 | High | **確定（§3-21）**: 前提不成立（同一スレッド）。防御的強化 |
| 3-6 | SnapshotFactory NaN ハッシュ不一致 | SnapshotFactory.cpp:36 | Medium | **確定（§3-26）**: 両面対応（hash 正準化 + equivalence fail-closed） |
| 3-9 | /fp:fast 精度低下 | CMakeLists.txt | Medium | **確定（§3-29）**: ターゲット固有化（ConvoPeq 個別へ） |
| 3-10 | /QxCORE-AVX2 AMD 非互換 | CMakeLists.txt | Medium | **確定（§3-30）**: ターゲット固有化（MSVC と同型に） |
| R-9 | CMAKE_CXX_FLAGS_RELEASE グローバル上書き | CMakeLists.txt | Medium | **確定（§3-31）**: target_compile_options 移行（3-9/3-10 と一体） |
| R-新規B | Incremental rebuild 未接続 | Rebuild.cpp / ConvolverProcessor.h | P2 | **確定（2026-08-11）**: 休眠維持（Option 1）+ reset() の retire 化・未定義関数スタブ化 → §3-33 |

## A. 現状維持バグ一覧

| # | バグ | 確定理由 |
|---|---|---|
| 2-8 | cleanup() 強制削除未実装 | cleanup（終了済みのみ削除）+ forceCleanup（stopThread(500)）の役割分担は正常。二重停止回避のため現状維持 |
| 3-2 | DeferredDeletionQueue kMaxScan デッドコード | 安全（先頭ブロックで break する設計）。コメント文書化のみ |
| 3-3 | AlignedAllocation 例外 RT 伝播 | RT パスではメモリ確保なし（事前割当制）。`aligned_malloc_nothrow` 提供済み |
| 3-8 | cachedLatency 例外安全性 | `updateLatencyCache` の exchangeAtomic パターンで公開前確保・二重解放なし |
| R-新規E | setUseIncrementalRebuild 過去バグ | 修正済み（enable を正しく使用）。監査記録のみ |

---

## 3. 詳細改修設計

### 3-1. Bug 1-1 `nucHCMode` / `nucLCMode` がセッション永続化から欠落（High / P0）

**現状（Confirmed）**: `ConvolverProcessor::getState()`（StateAndUI.cpp:202-248）はテール関連を書き出すが `nucHCMode`/`nucLCMode` を欠落。`setState()`（:289-364）も読み込みなし。実行時は `pendingOverride`/`snapshot` 間で同期済み（:142-143, 194-199）、ハッシュ計算に含まれる（:55-56, 861-862）。

**別視点調査（2026-08-11）で確定**:
- **getState の読取ソースは `pendingOverride`（:194-199）**。`snapshot`（:55-56）はハッシュ計算・等価判定用の複製であり、永続化すべきは `pendingOverride` 側（UI 反映済みの最新値）。旧記載の `snapshot.nucHCMode` は誤り
- **enum 実値（OutputFilter.h:75-87）**: `HCMode{Sharp=0, Natural=1, Soft=2}` / `LCMode{Natural=0, Soft=1}`。**`LCMode::Sharp` は存在しない**（上限は `LCMode::Soft`=1）。旧記載の jlimit 上限（HCMode::Natural / LCMode::Sharp）は誤り
- **ゲッター名は `getNUCHCMode()` / `getNUCLCMode()`**（`getNucHCMode` は存在しない）。setter は `setNUCFilterModes(HCMode, LCMode)`
- **jlimit 上限の正しい値**: HCMode は `Soft`(2)、LCMode は `Soft`(1)

**修正方針（実施確定）**:
```cpp
// getState() に追加（StateAndUI.cpp:242 付近：maxCacheEntries の後、irFileLock の前）
v.setProperty("nucHCMode", static_cast<int>(pendingOverride.nucHCMode), nullptr);
v.setProperty("nucLCMode", static_cast<int>(pendingOverride.nucLCMode), nullptr);

// setState() に追加（StateAndUI.cpp:362 付近：maxCacheEntries の後、irPath の前）
if (v.hasProperty("nucHCMode") && v.hasProperty("nucLCMode")) {
    const int hcVal = static_cast<int>(v.getProperty("nucHCMode"));
    const int lcVal = static_cast<int>(v.getProperty("nucLCMode"));
    const auto hc = juce::jlimit(static_cast<int>(convo::HCMode::Sharp),
                                 static_cast<int>(convo::HCMode::Soft), hcVal);
    const auto lc = juce::jlimit(static_cast<int>(convo::LCMode::Natural),
                                 static_cast<int>(convo::LCMode::Soft), lcVal);
    setNUCFilterModes(static_cast<convo::HCMode>(hc),
                      static_cast<convo::LCMode>(lc));
}
```

**テスト**: 保存→再読込後に `getNUCHCMode()` / `getNUCLCMode()` が元値と一致。
**リスク**: Low（pure addition、ロジック変更なし）。

### 3-2. Bug 1-2 `coordinatorDeferredRing_` / `lastResortQueue_` デッドコード（High / P1）

**現状（Confirmed）**: `coordinatorDeferredRing_`（:515）/ `lastResortQueue_`（:518）は消費のみ（drainOverflowRing の pop/compaction）で push/producer なし。`lastResortQueue_` は生配列で未初期化。

**§10 確定方針**: X6 設計で overflow ring は retire intent overflow 用・設計変更なし。**producer は実装せず、値初期化のみ実施**。`coordinatorDeferredRing_` / `lastResortQueue_` は「将来予約領域」として維持し、到達不能であることをコメント明示。

**修正方針（実施確定・値初期化のみ）**:
```cpp
// ISRRuntimePublicationCoordinator.h:518
// ★ Bug 1-2: 未初期化配列の UB の芽を除去（値初期化）。producer は X1-X6 の
//   retire intent overflow 実装時に追加予定（現状は到達不能な予約領域）。
RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity]{};
```

**テスト**: TSan で同時アクセスなし確認。drain 関数で `lastResortCount_ == 0` の即 return 確認。
**リスク**: Low（値初期化のみ）。

### 3-3. Bug 1-3 `_mm256_store_pd` アライメント保証なし（Critical / P0）

**現状（Confirmed・精緻化）**: `InputBitDepthTransform.h:114-115` の `_mm256_store_pd` は `double* dst` の 32-byte アライメントを保証しない。他に TruePeakDetector.cpp:85, EQProcessor.Processing.cpp:37, MKLNonUniformConvolver.cpp:1319, 1580 にも存在。

**別視点調査（2026-08-11）で確定**: 5 サイト中 **4 サイトは恒久安全**（修正不要）:
| サイト | dst/tmp の確保 | 判定 |
|---|---|---|
| InputBitDepthTransform.h:114-115 | 呼び出し元提供（現 3 呼出元は全て 64B: alignedL/R・makeAlignedArray）| ✅ **唯一の本質的リスク**（public header inline、32B 契約未文書化）|
| TruePeakDetector.cpp:85 | `alignas(32) double tmp[4]`（スタックローカル）| ✅ リスクなし |
| EQProcessor.Processing.cpp:37 | `alignas(32) double temp[4]`（スタックローカル）| ✅ リスクなし |
| MKLNonUniformConvolver.cpp:1319 | `mkl_malloc(partStride*8, 64)`（.h:318）| ✅ リスクなし |
| MKLNonUniformConvolver.cpp:1580 | 既に runtime check 有り（:1574 `(dst&31)==0` で storeu 分岐）| ✅ 設計上安全（参照実装）|

**修正方針（実施確定・InputBitDepthTransform.h の 1 サイトのみ）**:
```cpp
// InputBitDepthTransform.h:114-115 — Option A（推奨・性能影響ほぼゼロ）
_mm256_storeu_pd(dst + i,     _mm256_cvtps_pd(lo));
_mm256_storeu_pd(dst + i + 4, _mm256_cvtps_pd(hi));
```
- **Option B（契約明示）**: ヘッダーに「dst は 32-byte アライン必須」をコメント明記 + 入口アサート。MKLNonUniformConvolver.cpp:1580 の runtime check パターンを参照実装として採用推奨。
- **性能**: Haswell+/AMD Zen はアライン済みアドレスで `storeu` は `store` と同等（penalty ゼロ）。本サイトは reduction/fallback パスでホットループではない。

**テスト**: 非アライドバッファを渡して #GP なし（AddressSanitizer + 非アライド割当）。
**リスク**: Low（`_mm256_storeu_pd` 化、性能影響ほぼゼロ）。

### 3-4. Bug 1-4 `fastTanh` 3箇所独立複製（Medium / P1）

**現状（Confirmed・精緻化）**: `FastTanhApprox.h:63` に `SoftClipPadéPolicy`（10395/1260/21 + 4725/210, clip=4.5）。DSPCoreFloat.cpp:146 / DSPCoreIO.cpp:76 は独自 `fastTanh` 複製。EQProcessor.Processing.cpp:104 はデフォルトポリシー。

**別視点調査（2026-08-11）で確定**:
- **DSPCoreFloat.cpp:146 と DSPCoreIO.cpp:76 の独自 `fastTanh` は `SoftClipPadéPolicy` とビット単位で完全一致**（定数 10395/1260/21 + 4725/210、式・演算順含む）→ **統合は数値的に NO-OP（確実に安全）**
- **EQProcessor.Processing.cpp:108,113 は `DefaultFastTanhPolicy`（27/9 Padé）を使用**。コメント「係数は現行の 27/9 を維持。Padé 近似の変更（5次/6次）は別チケット」= **意図的な差分であり変更禁止**。
- リネームの実影響は **FastTanhApprox.h:63 + DSPCoreDouble.cpp:127,191 の 3 箇所のみ**（Unicode `é` のため全ソース一括 sed 不可、`rg "SoftClipPad[ée]"` で確認）

**修正方針（実施確定・統合は NO-OP）**:
1. `SoftClipPadéPolicy` → `SoftClipPadeApproxPolicy` にリネーム（Bug 2-1 と連動、3 箇所）
2. DSPCoreFloat.cpp / DSPCoreIO.cpp の独自 `fastTanh` を削除し、`convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>(...)` に統一
3. **EQProcessor.Processing.cpp は `DefaultFastTanhPolicy`（27/9）のまま変更しない**（意図的な差分）
```cpp
// DSPCoreFloat.cpp:186 / DSPCoreIO.cpp:116 — After
const double clipped = threshold + knee * convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>((abs_x - threshold) / knee);
```

**テスト**: `fastTanh<SoftClipPadéPolicy>` vs `fastTanhV256<SoftClipPadéPolicy>` vs 旧ローカル係数を -10〜+10, 0.1 刻みでビット一致 assert（単体契約テスト）。既存テスト回帰（EQProcessorMaxGainTests / GainStagingContractTests / CrossfadeExecutorLocalContractTests）。
**リスク**: Low（係数がビット同一のため統合は NO-OP。EQProcessor の 27/9 は変更しない）。

### 3-5. Bug 1-5 `musicalSoftClip` デッドコード（Low / P2）

**現状（Confirmed）**: `AudioEngine::DSPCore::musicalSoftClip()`（AudioEngine.h:1068 / DSPCoreIO.cpp:341）は呼び出しゼロ。実処理はローカル `musicalSoftClipScalar()`（DSPCoreIO.cpp:95 / DSPCoreFloat.cpp:165 / DSPCoreDouble.cpp:107）。

**§10 確定方針**: クラスメソッドを**削除**。

**修正方針（実施確定）**:
- `AudioEngine.h:1068` の宣言削除
- `DSPCoreIO.cpp:341-343` の定義削除（`musicalSoftClipScalar` は維持）

**テスト**: ビルド成功 + 既存テスト通過（出力変化なし）。
**リスク**: Low（デッドコード削除）。

### 3-6. Bug 1-6 IPP FFT 戻り値無視（Medium / P0）

**現状（Confirmed）**: `MklFftEvaluator.h:270-271, 425-426` で `ippsFFTFwd_RToCCS_64f()` の戻り値無視。`FFTBackend.cpp:130,146` は修正済み。`ippsFFTInit_R_64f`（:78-92）と MKL DFT はチェック済み。

**別視点調査（2026-08-11）で確定**:
- **:270-271 は `Result evaluate(...) noexcept`（:244）に属し、戻り値は `Result{}`**。:425-426 は `void computeFft(...) noexcept`（:410）に属し、出力ゼロクリア + `return;`。**両関数とも noexcept でポインタ戻りではないため `return nullptr;` は不適（旧記載の誤り）**
- 既存の `fftSpec==nullptr` ガード（:270/:425）と**同一のフォールバック**を使用するのが一貫的

**修正方針（実施確定）**:
```cpp
// :270-271 の修正（evaluate）— 既存 :256 の fftSpec==nullptr ガードと同一形式
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed (st1=" + juce::String(static_cast<int>(st1)) + ", st2=" + juce::String(static_cast<int>(st2)) + ")");
    return Result{};   // ★ nullptr ではない（evaluate の戻り値は Result）
}

// :425-426 の修正（computeFft）— 既存 :425-427 の fftSpec==nullptr ガードと同一形式
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed (st1=" + juce::String(static_cast<int>(st1)) + ", st2=" + juce::String(static_cast<int>(st2)) + ")");
    // 出力ゼロクリア（computeFft は void なので return;）
    juce::FloatVectorOperations::clear(spectrumLeft, fftSize);
    if (spectrumRight != nullptr) juce::FloatVectorOperations::clear(spectrumRight, fftSize);
    return;
}
```

**テスト**: 無効な fftSpec でエラー検出（既存ガードと同一のフォールバック動作）。
**リスク**: Low（エラーハンドリング追加）。

### 3-7. Bug 1-7 ISRRetire Mutex（High / P0 → **P2 に下方修正**）

**現状（Confirmed・精緻化）**: `emitRetireIntentRT()`（ISRRetire.cpp:94）は `emitRetireIntent()`（:102）を素通しし `fallbackMutex_`（:44, 135, 265）を取得。

**別視点調査（2026-08-11）で確定**:
- `emitRetireIntentRT` の**唯一の呼び出し元は `onRuntimeRetiredNonRt`（Commit.cpp:437, 476）**であり、`ASSERT_NON_RT_THREAD()` 済み（Commit.cpp:439）
- つまり**現時点で RT 違反は顕在化していない**（Audio Thread からは呼ばれない）。ISRRetire.cpp:94 のコメントも「現時点では呼び出し元は全て非 RT スレッドであることを確認済み」と明記
- 問題の本質は (a) 関数名 `emitRetireIntentRT` が誤解を招く（RT スレッド安全を意味しない）、(b) 将来 Audio Thread から呼ばれた場合の設計保証がない
- **優先度は P0 → P2 に下方修正**（現時点で実害なし、将来の活性化リスク）

**修正方針（設計判断 → 実施確定・低リスク）**:
1. **関数名リネーム**: `emitRetireIntentRT` → `emitRetireIntentFromNonRT`（呼び出し元 Commit.cpp:476 と ISRRetire.h:59 の 2 箇所 + コメント）
2. **将来の Audio Thread 呼び出しに備えた設計保証**: リネーム後の関数に「この関数は Non-RT 専用。将来 Audio Thread から呼ぶ場合は overflowRing_ へのロックフリー退避（mutex なし）を実装すること」とコメント明示（既存コメント :97-100 を強化）
3. **overflowRing_ へのロックフリー化は現時点では実施しない**（呼び出し元が NonRT のみのため不要。将来の Audio Thread 呼び出し時に実施）

**テスト**: 既存 ISRSoakTests 通過（リネームのみで動作不変）。
**リスク**: Low（リネーム + コメント強化。ロジック変更なし）。

### 3-8. Bug 1-8 LoaderThread OOM（High / P0）

**現状（Confirmed・精緻化）**: `doLoadIRStep()`（LoaderThread.cpp:410, :447-488）で `tempFloatBuffer(numChannels, int(fileLength))` + `tempAligned(double全量)` + `loadedIR.setSize` の 3 バッファを同時保持。**ピークメモリはステレオ @ 2^31-1 サンプルで約 67GB**（float 16.8GB + double 17.2GB + loadedIR 33.5GB）。「最大約16GB」は float のみの過小試算。

**別視点調査（2026-08-11）で確定**:
- `reader->read()` は**非 virtual でチャンク毎に繰返し呼出可能**（`startSampleInDestBuffer` と `readerStartSample`(int64) を独立指定可）。**ただしカスタムフォーク引数**（`useReaderLeftChan/useReaderRightChan`、標準 JUCE の `useReaderCache/allowSRC` ではない）に注意
- `loadedIR` は `juce::AudioBuffer<double>` で**サンプル数が int** → ストリーミング化しても **INT32_MAX サンプル/ch が上限**。**`MAX_FILE_LENGTH`（2^31-1, :450）は必ず維持**
- プレビュー関数 `loadImpulseResponsePreviewFile()`（ResampleAndFallback.cpp:271, :305-325）にも同一 OOM パターンが存在（短尺で実害低だが、読み込みパス全体の整合の観点で共通化 or コメント差分化が望ましい）

**修正方針（設計判断 → 実施確定・ストリーミング化）**:
```cpp
// LoaderThread.cpp — ストリーミング読み込み（ピークメモリ削減）
constexpr int64_t STREAMING_CHUNK = 256 * 1024;
// 1. loadedIR.setSize を先に確保（int 上限内で事前割当）
stepResult.loadedIR.setSize(numChannels, static_cast<int>(fileLength));
// 2. チャンク毎に read → convertFloatToDoubleHighQuality → loadedIR に書込
for (int64_t offset = 0; offset < fileLength; offset += STREAMING_CHUNK) {
    const int64_t chunk = std::min(STREAMING_CHUNK, fileLength - offset);
    juce::AudioBuffer<float> chunkBuf(numChannels, static_cast<int>(chunk));
    if (!reader->read(&chunkBuf, 0, static_cast<int>(chunk), offset, true, true)) { ... }
    // tempAligned を再利用し chunk 毎に変換 → loadedIR.copyFrom
}
// 3. tempFloatBuffer 全量確保を廃止（float 16.8GB 削減）
```
- **`MAX_FILE_LENGTH`（2^31-1）ガードは維持必須**（AudioBuffer の int 上限）
- `doTrimStep`（:494〜）は loadedIR 上で動作するためストリーミングは読み込み段のみに閉じる

**テスト**: 2GB 超の巨大 IR ファイルでメモリ使用量が一定範囲内（ピークがチャンクサイズ x 2ch x 2 型程度に収まる）。
**リスク**: Medium（読み込みロジックの変更。chunk 対応 + `MAX_FILE_LENGTH` 維持 + プレビュー関数整合の確認必須）。

### 3-9. Bug 1-9 MKLNonUniformConvolver int/size_t 混在（High / P0 → **設計判断・防御的修正**）

**現状（Confirmed・精緻化）**: `l.fftSize * sizeof(double)` が int*size_t。**行番号訂正**: 実箇所は **:841, 843, 845, 847**（:853/855 は `partStride * sizeof(double)` / `complexSize * sizeof(double)` であり fftSize ではない）。

**別視点調査（2026-08-11）で確定**:
- `fftSize` は `Layer` 構造体（MKLNonUniformConvolver.h:291 `int fftSize = 0`）。代入は **:778 `l.fftSize = l.partSize * 2;`**（int×int＝int = **唯一の実オーバーフロー点**）
- **実効リスクは P0 過大**: fftSize の実効最大はテストケースでも 524288（5s@768kHz）。int オーバーフローには `partSize > 2^30` が必要だが、IR パイプライン全体が `MAX_FILE_LENGTH = 2^31-1` で int 上限に拘束され**到達不能**。現状で発火するヒープ破壊ではない（防御的修正）
- 使用箇所は全 12 箇所。`clearFFTOutputOnError` は size_t 引数で暗黙変換安全。`FloatVectorOperations::clear(buf, l.fftSize)` は int 引数

**修正方針（最小修正を推奨・実施確定）**:
```cpp
// 割当サイト: static_cast<size_t> 明示（:841,845,843,847,906-907）
DIAG_MKL_MALLOC(static_cast<size_t>(l.fftSize) * sizeof(double), 64);

// :778 — int64 乗算（唯一の実オーバーフロー点）
l.fftSize = static_cast<int>(static_cast<int64_t>(l.partSize) * 2);  // partSize は int のまま
```
- **fftSize 自体の int64_t 化はしない**（:355 N / :781 complexSize / :792 createPlan 引数 / :893-894 FloatVectorOperations 等 12 箇所 + 関連 int メンバへの連鎖でコストに見合わない）。割当サイトの cast + :778 の乗算順序修正に限定

**テスト**: 5秒@768kHz（fftSize 524288）でメモリ割当サイズ正しい。
**リスク**: Low（型衛生の改善。防御的修正で現状発火なし）。

### 3-10. Bug 1-10 `m_pendingIRChange` 公開前クリア（High / P1 → **現状記載の技術的誤りを修正**）

**現状（Confirmed・精緻化）**: `AudioEngine.Snapshot.cpp:95` で `exchangeAtomic(m_pendingIRChange, false, acq_rel)` が実行される。

**別視点調査（2026-08-11）で確定 — 元の「現状」は技術的に誤り**:
- 対象関数は `createSnapshotFromCurrentState(uint64_t)`（Snapshot.cpp:14-153、戻り値 void）。呼び出し元は **Timer.cpp:667 の 1 箇所**（Message Thread、`!isFading() && currentSnapshot==nullptr` 時のみ）
- **「createImpl() が nullptr を返すとフラグが永久消失」は誤り**: `:95` の `exchangeAtomic(false)` は **createImpl（:124）より前**。フラグが true なら `:96` で `promoteToStructural=true` → `:100` で早期 return。フラグが false なら `:124` に進み、**createImpl 到達時点で m_pendingIRChange は必ず false**。nullptr return（:135）はフラグ消費と無関係
- **真の課題は「構造 publish 完了前クリア」**: `:95` のクリアにより `Parameters.cpp:381,452` の `shouldDeferRebuild` 判定が誤って「非保留」になる（構造遷移中の rebuild 投入競合）。フラグは **DSPTransition.h:123（onPublishCompleted 内・クロスフェード開始時に再セット）** されるため、スティッキー・ラッチ動作
- 全参照は全て **NonRT**（Timer.cpp:773 Message / Parameters.cpp:381,452 Message / Snapshot.cpp:95 Message / AudioEngine.h:1451 Message・NonRT publish 完了）。`consumeAtomic`=読取のみ・`exchangeAtomic`=クリア（AtomicAccess.h）

**修正方針（設計判断 → 実施確定・クリア移動）**:
```cpp
// Snapshot.cpp:95 — After: exchange クリアをやめ、読取のみに変更
const bool promoteToStructural = convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire);
if (promoteToStructural) { DBG(...); return; }  // ★ 早期 return は維持必須（スナップショットfade回避）
// ★ クリアは構造 publish 完了経路（DSPTransition::onPublishCompleted）へ移動
```
- **`promoteToStructural` の早期 return は維持必須**（無くすと IR 変更がスナップショット fade に流れ構造クロスフェード設計と競合）
- 元の「publishSuccess で publishAtomic(false)」案は**この関数に publishSuccess が存在しないため適用不能**（戻り値 void・startFade/switchImmediate は無戻り値）

**テスト**: IR 変更後、`shouldDeferRebuild`（Parameters.cpp:381,452）の読取ゲーティングが構造 publish 完了まで維持されることを確認。
**リスク**: Medium（フラグのクリア位置変更。`promoteToStructural` 早期 return と `DSPTransition.h:123` の再セットの整合確認必須）。

### 3-11. Bug 2-1 非ASCII識別子 `SoftClipPadéPolicy`（Low / P2）

**現状（Confirmed・精緻化）**: `FastTanhApprox.h:63` の `struct SoftClipPadéPolicy` は U+00E9 を含み cppcheck をクラッシュさせる。

**別視点調査（2026-08-11）で確定**: リネームの実影響は **FastTanhApprox.h:63（定義）+ DSPCoreDouble.cpp:127,191（使用）の 3 箇所のみ**。`SoftClipPadéPolicy` の参照は DSPCoreFloat/IO には存在しない（局所関数のため）。**Unicode `é` のため全ソース一括 sed 不可**（`rg "SoftClipPad[ée]"` で確認が必要）。

**修正方針（実施確定）**: `SoftClipPadeApproxPolicy` にリネーム。**Bug 1-4 と連動**（リネーム中途でビルドを跨がない: 3 箇所を同時に更新）。
**テスト**: cppcheck 正常解析 + ビルド成功（リネーム後も DSPCoreDouble の SoftClip 出力不変）。
**リスク**: Low（リネームのみ。3 箇所の同時更新が必須）。

### 3-12. Bug 2-2 入力 DC ブロッカー NaN/Inf 非対称（Low / P2）

**現状（Confirmed）**: 入力側は DC ブロッカー前にのみ `sanitizeFiniteChunk()`（DSPCoreIO.cpp:231-232, 282-283）。出力側は完全なスクラブ（ConvolverProcessor.Runtime.cpp:722）。

**別視点調査（2026-08-11）で確定**:
- **DC ブロッカー実装は `convo::UltraHighRateDCBlocker`（src/UltraHighRateDCBlocker.h、ヘッダオンリー）**。2段カスケード 1次 IIR ハイパス（`y = x − 1次LP(x)` の差分構成）、double 全経路、`alpha = 1 − exp(−ω)`（std::expm1 で小値域の桁落ち防止）
- **カットオフ実値**: input/output = **3.0 Hz**、oversampled = **1.0 Hz**（AudioEngine.h:643-651 `DCBlockerRuntimeState::init` で固定）。**「~20Hz」は実値と不一致（旧記載の誤り）**
- **「Q=0.707」は技術的に不可能な記述**（Q は2次フィルタのパラメータ。1次フィルタに Q は存在しない）。実コードにも Q パラメータなし
- フィルタ構造・係数・カットオフは**現状維持が妥当**（高速・低位相歪み、RT 安全。処理コスト: 2段 ≈20 FLOP/サンプル/CH、48kHz/2ch/1024 サンプル ≈8μs、全体の <1% 増加）
- 実バグ側面: 入力側は DC ブロッカー**後**にスクラブなし（:250-251, 303-304）

**修正方針（設計判断 → 実施確定・低リスク）**: フィルタ実装は維持し、入力側 DC ブロッカー呼び出し直後（DSPCoreIO.cpp:252 と 305 付近）にスクラブ追加。
```cpp
// DSPCoreIO.cpp:252 / :305 付近（dc.inputL / dc.inputR の process 直後）
sanitizeFiniteChunk(alignedL.get(), numSamples);
sanitizeFiniteChunk(alignedR.get(), numSamples);
```
**テスト**: NaN/Inf 入力を DC ブロッカーに通した後、出力に NaN/Inf なし。
**リスク**: Low（データ完全性向上）。

### 3-13. Bug 2-3 ユニットテストの矛盾条件（Low / P2）

**現状（Confirmed）**: `EQProcessorMaxGainTests.cpp:355-358` で `for (delta < 1e-6)` と `if (delta > 1e-6)` が同時に真になり得ない（cppcheck: oppositeInnerCondition）。

**別視点調査（2026-08-11）で確定**:
- `logBound` 初期値は 0.0。ループは `delta < 1e-6` のため delta は 1e-7 まで（max=1e-7）。内側 `if (delta > 1e-6)` は**常に false = デッドコード**（cppcheck 指摘どおり）
- **内側 if 削除後**: `logBound` に log1p(1e-15)+…+log1p(1e-7) ≈ **1.11e-7** が蓄積 → `ubDb = (20/ln10)×1.11e-7 ≈ 9.6e-7`（有限・正）。テストが実質的に検証するようになる（現在は logBound=0 で空振り）
- **削除範囲は :357-358 の if 行＋代入行のみ**（:355-356 の for と { は維持、閉じ括弧を1つ上げる）
- 現行は `isfinite` のみで 0 でも通過するため、**`check(ubDb > 0.0)` の追加が必要**

**修正方針（実施確定）**:
```cpp
:354      double logBound = 0.0;
:355      for (double delta = 1e-15; delta < 1e-6; delta *= 10.0)
:356      {
:357          logBound += std::log1p(delta);   // 内側 if(:357-358)を削除
:358      }
:359      const double ubDb = kTwentyOverLog10 * logBound;
:360      check(std::isfinite(ubDb), "log1p upperBound: tiny delta finite");
:361      check(ubDb > 0.0, "log1p upperBound: tiny delta positive");   // ★追加
```
**テスト**: テストが `logBound > 0` を検証するようになる（`check(ubDb > 0.0)` 追加で保証）。
**リスク**: Low（テスト修正）。

### 3-14. Bug 2-4 CacheManager strict-aliasing 違反（Low / P2）

**現状（Confirmed）**: `CacheManager.cpp:267` で `const uint8_t*` を `reinterpret_cast<const double*>` で読み替え（UB）。

**別視点調査（2026-08-11）で確定**:
- `:260` の `tdStart = dataStart + header.dataSize`（`const uint8_t*`、mmap データ）。`:264` の `tdSrc = reinterpret_cast<const double*>(tdStart)` で double スケールのポインタ演算（:268-269）と memcpy ソースに使用 → **strict-aliasing 違反**
- `raw`（:203/:241 の `uint8_t*`）は warm-up ループのローカル変数で無関係
- **要素単位 memcpy（旧記載の `double val; memcpy(&val, raw + byteOffset, ...)`）よりも、チャネル単位のバイト領域 memcpy の方が簡潔・高速**（tdSrc/idx を完全除去）

**修正方針（実施確定）**: チャネル単位バイト memcpy 方式に修正。
```cpp
// :264-272 を置換（double* の reinterpret を全廃し、バイト領域で memcpy）
            const size_t samplesPerCh = static_cast<size_t>(header.timeDomainNumSamples);
            const size_t bytesPerCh   = samplesPerCh * sizeof(double);
            for (int ch = 0; ch < static_cast<int>(header.timeDomainChannels); ++ch)
            {
                std::memcpy(tdBuffer->getWritePointer(ch),
                            tdStart + (static_cast<size_t>(ch) * bytesPerCh),  // byteOffset をバイト領域で計算
                            bytesPerCh);
            }
```
**テスト**: strict-aliasing 警告なしでビルド。
**リスク**: Low（メモリ安全性向上）。

### 3-15. Bug 2-5 LockFreeRingBuffer::size() データ競合（Medium / P2）

**現状（Confirmed）**: `LockFreeRingBuffer.h:76-81` の `size()` が writeIndex/readIndex を別々に acquire 読み。SPSC で実害は稀だが負/巨大値を返す可能性。

**別視点調査（2026-08-11）で確定**:
- **SPSC 専用**（ヘッダ先頭コメント・`MpscBoundedRing.h:7` で明記）。index は `std::atomic<size_t>` = **64bit で uint32 ラップ問題なし**（x64）
- **現状は writeIndex を先に、readIndex を後に読む**（w 先読み）。この順序では `w − r` は**実占有数以下（保守側・過小評価）**。負値・Capacity 超過は 64bit 実運用上発生しない
- `.md` 旧記載の「間に Producer が writeIndex を進めると w − r が Capacity を超える」は **readIndex 先読み実装の場合の話**であり、現状コード（w 先読み）では成立しない
- 唯一のコンシューマは `AudioEngine.Timer.cpp:1445` の診断ログ近似値（`approxOcc`）で**機能的影響ゼロ**

**修正方針（設計判断 → 実施確定・防衛的）**: クランプ追加は妥当（将来の 32bit 化への保険）。`convo::consumeAtomic`（acquire）維持でオーダリング変更不要。
```cpp
size_t size() const noexcept {
    // acquire × 2: push/pop の release と HB し、一貫した（ベストエフォート）占有数を算出。
    // ★ w 先読みのため常に過小評価（安全側）。クランプは将来の 32bit 化等への防衛的保険。
    const auto w = writeIndex.load(std::memory_order_acquire);
    const auto r = readIndex.load(std::memory_order_acquire);
    return (w >= r) ? (w - r) : 0;
}
```
**テスト**: SPSC ストレステストで負にならない。
**リスク**: Low（防衛的）。

### 3-16. Bug 2-6 RCUReader ハッシュ衝突（High / P1 → **前提不成立・severity 下方**）

**現状（Confirmed・精緻化）**: `RCUReader.h:152` と `ThreadHash.h:9` で `cachedThreadHash()`（= std::hash<std::thread::id>）。`enter()`:69-79 の CAS でオーナー識別に使用。

**別視点調査（2026-08-11）で確定 — 前提不成立**:
- MSVC STL（14.51.36231）を実地検証: `std::hash<std::thread::id>` は **FNV-1a 全単射**（`_Fnv1a_append_value`）。thread::id は 4 バイト単一メンバ → **異なる live thread id は必ず異なる 64bit ハッシュ**。衝突は発生しない
- **バグ前提の「連続 thread id → 同一ハッシュ」は MSVC では不成立**
- `cachedThreadHash()` の全呼び出し元 4 箇所（RCUReader.h:152 オーナー識別 / DspNumericPolicy.h:44,120 AudioThreadSlot tag / EpochDomain.h:75 ReaderSlot.ownerThreadId 診断専用）は全て uint64_t を不透明トークンとして使用
- **修正価値は「std::hash 実装依存の排除・OS スレッド ID 再利用への非依存・決定的トークン化」**（堅牢性・移植性の改善）

**修正方針（設計判断 → 実施確定・堅牢性改善）**:
```cpp
// ThreadHash.h — After（呼び出し元変更不要・1 関数のみで完結）
inline uint64_t cachedThreadHash() noexcept {
    static std::atomic<uint64_t> g_nextId{1};
    static thread_local const uint64_t s_id = g_nextId.fetch_add(1, std::memory_order_relaxed);
    return s_id;
}
```
- RT-safe（thread_local + relaxed fetch_add）。`kMaxAudioThreadSlots = 4` より十分に広い ID 空間
**テスト**: 100+ スレッドで一意 ID を確認（既存テスト回帰）。
**リスク**: Low（thread_local カウンタ追加、呼び出し元変更不要）。

### 3-17. Bug 2-7 atomic<DSPHandle> ロックフリー検証不足（Medium / P1）

**現状（Confirmed・精緻化）**: `ISRDSPHandle.h:215-218` の static_assert は `#if !defined(_MSC_VER)` ガード下のみ。MSVC/icx は `#else` でコメントアウト。`ISRDSPHandle.cpp:11-27` のランタイム検証は MSVC で `(void)ok` により無視。

**別視点調査（2026-08-11）で確定 — 重大発見**:
- **MSVC の `std::atomic<DSPHandle>` は実際には spinlock 実装**（`_Atomic_storage<DSPHandle>` primary テンプレート:528、`_Spinlock` 使用）。CMPXCHG16B を使う `_Atomic_storage<_Ty&, 16>`:1120 は**参照型**のため `atomic<DSPHandle>` には一致しない
- コメント（cpp:17-19）の「実際は CMPXCHG16B で lock-free」は**誤り**。`is_lock_free()==false` は正確
- **`if (!ok) std::abort()` 案は MSVC で毎回 abort するため不可**（ok が決定的に false）
- RT パスは `registry_[slot].generation`（atomic<uint64_t> 8B=真 lock-free）と `state`（atomic<DSPState> 4B=真 lock-free）のみ使用 → **RT 影響は現状ゼロ**。`std::atomic<DSPHandle>`（:222-223）は NonRT のみが書込/読取

**修正方針（設計判断 → 8 バイト pack 化を推奨）**:
- **案 A（推奨）: `std::atomic<uint64_t>` への pack 化**。slot（MAX_DSP_SLOTS=256 → 8bit）+ generation（上位 56bit）を 1 つの 8 バイト atomic に格納し、DSPHandle との変換ヘルパーを用意。8 バイト atomic は真の lock-free → `if (!ok) std::abort()` が有効な検証になる（C++17 互換・RT-safe）
- **案 B: `std::atomic_ref<DSPHandle>`**（C++20、CMPXCHG16B の真 lock-free）。寿命管理 + CX16 サポートの起動時検証が必要
- **案 C（最小）: 現状維持** + コメント（h:22-25 / cpp:17-19）を「MSVC では spinlock 実装」に修正。検証は `sizeof==16 && alignof>=16` の compile-time + 診断ログ
- ADR-005 の「lock-free 必須」不変条件が MSVC で満たされていない事実は記録

**テスト**: 案 A の場合 `is_lock_free()==true` を検証。`NormalRetireDSPHandleCompareTests.cpp` に lock-free 検証テスト追加。
**リスク**: 案 A Low（RT 影響なし、NonRT のみ変更）。案 B Medium（C++20 + CX16 検証）。案 C Low（コメントのみ）。

### 3-18. Bug 2-8 cleanup() 強制削除未実装 — **現状維持（§10 確定）**

`cleanup()`（LoadPipeline.cpp:571）は終了済みのみ削除、`forceCleanup()`（StateAndUI.cpp:969）が `stopThread(500)`。役割分担は正常。二重停止リスク回避のため改修しない。

### 3-19. Bug 2-9 タイミング計算 uint64 underflow（Medium / P1）

**現状（Confirmed）**: `AudioBlock.cpp:624,630,664` / `BlockDouble.cpp:589,594,626` で uint64 減算がアンダーフロー（**BlockDouble の実在行番号。旧記載 :586/591/623 は +3 ズレ**）。

**別視点調査（2026-08-11）で確定**:
- `getCurrentTimeUs()`（TimeUtils.h:14）は `std::chrono::steady_clock` ベースで `uint64_t` を返す（単調時計）
- **安全なサイト**: `callbackMs = (t1_end − t0_start)`（AudioBlock :543 / BlockDouble :515）は**同一関数内で t1_end を後から取得**するため steady_clock では逆転し得ない → **修正不要**
- **要修正の実在リスク**: `− cbPrevEndUs`（別コールバック由来の atomic 読取）と `− matchedPublishEndUs`（リング由来）は時系列順序が保証されず underflow で巨大値化の可能性
- **第4サイト追加**: `intervalMs`（AudioBlock :541 / BlockDouble :513）も `cbPrevEndUs` 減算のため underflow 対象（`cbPrevEndUs > 0` ガードのみでは防げない）

**修正方針（実施確定・全サイト saturating subtraction）**:
```cpp
// AudioBlock.cpp:624 / BlockDouble.cpp:589
const uint64_t callbackUs64 = (nowUs >= cbStartUs) ? (nowUs - cbStartUs) : 0;
const uint32_t callbackUs = static_cast<uint32_t>(std::min<uint64_t>(callbackUs64, UINT32_MAX));

// AudioBlock.cpp:630 / BlockDouble.cpp:594
const uint64_t intervalUs64 = (cbStartUs >= cbPrevEndUs) ? (cbStartUs - cbPrevEndUs) : 0;
const uint32_t intervalUs = static_cast<uint32_t>(std::min<uint64_t>(intervalUs64, UINT32_MAX));

// AudioBlock.cpp:664 / BlockDouble.cpp:626
const uint64_t observeLatencyUs =
    (observeUs >= matchedPublishEndUs) ? (observeUs - matchedPublishEndUs) : 0;

// AudioBlock.cpp:541 / BlockDouble.cpp:513（★第4サイト追加）
intervalMs = (t0_start >= cbPrevEndUs)
    ? (static_cast<double>(t0_start - cbPrevEndUs) / 1000.0) : 0.0;
```
**テスト**: タイミング逆転シナリオで巨大値なし。
**リスク**: Low（防衛的計算）。

### 3-20. Bug 2-10 NoiseShaperType enum キャスト検証欠如（Medium / P1）

**現状（Confirmed）**: `AudioEngine.StateIO.cpp:90` で `setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"))` — 範囲チェックなし（enum: Psychoacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3）。

**別視点調査（2026-08-11）で確定**:
- `setNoiseShaperType` シグネチャ: `void setNoiseShaperType(NoiseShaperType type);`（AudioEngine.h:1413）
- enum 定義は `src/core/Types.h` で `NoiseShaperType{Psychoacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3}`（範囲 [0,3]）を確認済み
- `(NoiseShaperType)(int)` の C スタイルキャストは範囲チェックなし。範囲外値（4, -1 等）で UB → **現行方針（範囲チェック付き）で正しい**

**修正方針（実施確定・変更なし）**:
```cpp
const int value = static_cast<int>(state.getProperty("noiseShaperType"));
if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic) &&
    value <= static_cast<int>(NoiseShaperType::Fixed15Tap))
    setNoiseShaperType(static_cast<NoiseShaperType>(value));
```
**テスト**: 範囲外値（4, -1）でクラッシュなし。
**リスク**: Low（入力検証追加）。

### 3-21. Bug 3-1 AudioSegmentBuffer リングラップ競合（High / P3 → **前提不成立・防御的強化**）

**現状（Confirmed・精緻化）**: `AudioSegmentBuffer.h:86-136` の `pushBlock()` が ring wrap 時に 2 回の copy（:104-112）。`copyLatest()`（:124-136）が `start..writePos` の全領域を読む。

**別視点調査（2026-08-11）で確定 — 前提不成立**:
- `AudioSegmentBuffer` の使用者は **NoiseShaperLearner のみ**。`pushBlock()`（:1173）と `copyLatest()`（:1204）は**どちらも同じ worker スレッド**（workerThreadMain:734）から逐次的に呼ばれる
- `getNumAvailableSamples()` は atomic int 読取のみ（RuntimeHealthMonitor はデータ競合なし）。**copyLatest を別スレッドから呼ぶ経路は存在しない**
- **リングラップ競合は現状では発生しない**（SPSC だが P と C が同一スレッド）。バッファの atomic + HB は将来のクロススレッド化に対する防御的設計
- 修正は防御的強化であり、バグ修正ではない（severity 下方）
- `kCapacity = 5s × 768kHz = 3,840,000`、`kMaxTrainingSegments = 16`

**修正方針（設計判断 → 防御的強化・バージョンカウンタ推奨）**:
- **バージョンカウンタ方式**（現行 atomic/HB と整合）: `pushBlock` 完了時に increment、`copyLatest` が取得した v と再読で不整合検出 → 再試行 or 読み捨て
- **ダブルバッファは非推奨**（kCapacity×2 = 約 122MB メモリ増）
- 将来 copyLatest を別スレッド（UI プレビュー等）から呼ぶ場合にのみ必須

**テスト**: TSan で ring wrap 時データ競合なし（防御的強化の検証）。
**リスク**: Low（現状競合なし。バージョンカウンタは防御的追加）。

### 3-22. Bug 3-2 DeferredDeletionQueue kMaxScan — **現状維持（§10 確定）**

`reclaim()` は先頭が削除不可なら即 break。安全。コメント文書化のみ。

### 3-23. Bug 3-3 AlignedAllocation 例外 RT 伝播 — **現状維持（§10 確定）**

RT パスで `aligned_malloc` 使用なし（事前割当制）。`aligned_malloc_nothrow` 提供済み。RT パスでの `aligned_malloc_nothrow` 使用をコメントで明示するのみ。

### 3-24. Bug 3-4 MKL アライメント判定競合（Medium / P3）

**現状（Confirmed）**: `MKLNonUniformConvolver.cpp:1571-1581` で `aligned` フラグを入口で1回計算。`dst`/`src` は 64-byte アラインなので常に true。mkl_malloc 非アラインは実運用で起こらない。

**別視点調査（2026-08-11）で確定**:
- `addFallback` ラムダ（:1567-）は :1597（addScaledFallback 内 `addFallback(n, dst, src)`）と :1611（`addFallback(toAdd, output, m_directOutBuf)`）から呼ばれる。`m_directOutBuf` は `DIAG_MKL_MALLOC(..., 64)`（:704）で 64-byte アライン。`output` は呼び出し元バッファ
- マスク `& 31` は 32-byte アライン判定で `_mm256_load_pd/store_pd` の要件と整合。load/store で同一フラグ・`i` が 4 刻み（32B）のため反復中もアライメント不変 → フラグ計算は正しい
- 「競合」の実体は実行時チェック（非アライン救済）と mkl_malloc(64) の保証付きアラインの**契約の不整合**（出力バッファは呼び出し元依存）。機能バグではなく契約の明文化が目的

**修正方針（実施確定）**: 入口でポインタ検証アサート追加（挿入位置: :1569 の `#if defined(__AVX2__)` 直後、:1570 の `int i = 0;` の前）。
```cpp
:1568  {
:1569  #if defined(__AVX2__)
         // ★ 64B アライン契約を明文化（通常動作では発火しない診断用）
         jassert((reinterpret_cast<std::uintptr_t>(dst) & 63u) == 0);
         jassert((reinterpret_cast<std::uintptr_t>(src) & 63u) == 0);
:1570      int i = 0;
```
**テスト**: 非アラインドポインタを検出。
**リスク**: Low（アサート追加）。

### 3-25. Bug 3-5 CacheManager volatile sink（Medium / P3）

**現状（Confirmed）**: `CacheManager.cpp:203,241` で `volatile uint8_t sink`。MSVC ではメモリバリアにならず、C++20 で非推奨。

**別視点調査（2026-08-11）で確定**:
- `CacheManager.cpp` にも `CacheManager.h` にも **`<atomic>` が無い** → `std::atomic_signal_fence` 使用には **`#include <atomic>` の追加が必須**
- **`.md` 旧記載の「`atomic_signal_fence` のみに置換」をそのまま適用すると `raw[i]` 読み出しが消失し、ページウォームアップが機能しなくなる**（誤り）。正しくは「ダミー accumulator（`sink`）への XOR を維持 + ループ内にバリア」で volatile だけを除去

**修正方針（実施確定）**:
```cpp
// CacheManager.cpp 先頭に追加
#include <atomic>

// 両関数共通の置換パターン（:202-206 / :240-244）
// Before
    constexpr size_t kPage = 4096;
    volatile uint8_t sink = 0;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < dataSize; i += kPage)
        sink ^= raw[i];
    (void)sink;
// After（volatile 廃止・コンパイラバリア化・読み出し維持）
    constexpr size_t kPage = 4096;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    uint8_t sink = 0;
    for (size_t i = 0; i < dataSize; i += kPage)
    {
        sink ^= raw[i];
        std::atomic_signal_fence(std::memory_order_seq_cst);  // ループ除去・読み出し保持を保証
    }
    (void)sink;
```
**テスト**: ページウォームアップが機能。
**リスク**: Low（最適化抑止パターン更新）。

### 3-26. Bug 3-6 SnapshotFactory NaN ハッシュ不一致（Medium / P3 → **両面対応が必要**）

**現状（Confirmed・精緻化）**: `hashCombineFloat()`（:22-31）は `-0.0f`/`0.0f` を同一視するが NaN ペイロードは非マスク。`areSnapshotsEquivalent()`（:46-97）は `abs(a-b) > epsilon` 比較で、NaN を等価と誤判定。

**別視点調査（2026-08-11）で確定**:
- `areSnapshotsEquivalent` / `computeContentHash` は `createImpl()`（:132-151）のみが呼ぶ。createImpl は「hash 一致 → equivalence」のゲート構造
- 実害は 2 点: (1) **NaN ペイロード相違で hash 不一致 → 毎回新スナップショット（rebuild churn）**、(2) **同一 NaN ペイロードで hash 一致 → equivalence が NaN を等価と誤判定 → 変更抑制（NaN の永続化）**
- 全参照は Message Thread（RT 呼び出しなし）→ isnan 追加の性能影響は無視できる
- `computeContentHash` は sampleRate・3 ゲイン・nsCoeffs を `std::bit_cast<uint64_t>`（NaN ペイロード感応）、saturationAmount のみ hashCombineFloat

**修正方針（設計判断 → 実施確定・両面対応）**:
```cpp
// 1. hash 側（computeContentHash / hashCombineFloat）— NaN 正準化
if (std::isnan(v)) bits = 0x7FC00000;  // 全 NaN が同一ハッシュになる

// 2. equivalence 側（areSnapshotsEquivalent）— NaN 非等価（fail-closed）
if (std::isnan(a) || std::isnan(b)) return false;
```
- 対象フィールドは全 14 箇所（sampleRate, inputHeadroomGain, outputMakeupGain, convInputTrimGain, saturationAmount, nsCoeffs[9]）
- NaN 非等価にすると同一 NaN の繰返し呼出で毎回新スナップショット（churn）になるが、これは「NaN を dedup しない」正しい fail-closed 挙動（Quarantined 等の思想と整合）
- **根本対策は params への NaN 混入の上流サニタイズ**（clamp/zero）を推奨

**テスト**: NaN 入力で `areSnapshotsEquivalent` が false + 全 NaN ハッシュ一致。
**リスク**: Low（Message Thread のみ・性能影響無視）。

### 3-27. Bug 3-7 SpectrumAnalyzer alignas ループ内配置（Low / P3）

**現状（Confirmed）**: `SpectrumAnalyzerComponent.cpp:474` でループ内に `alignas(64) float mags[8]`。MSVC で毎回スタック 64-byte アライン。

**別視点調査（2026-08-11）で確定**:
- `_mm256_store_ps` の要件は **32-byte アライン**。`alignas(64)` は過剰（オーバーアライメント）
- ループ内宣言は毎反復でアライン確保（コンパイラがホイストする場合もあるが保証なし）。配列は毎反復 `_mm256_store_ps` で全要素書き換えるため**反復間依存なし** → ループ外へ出しても安全
- 挿入位置: **:459（vScale）と :461（for）の間**。`:474` の宣言行を削除。`alignas(32)` で十分（store_ps の 32B 要件）

**修正方針（実施確定）**: ループ外へ移動 + `alignas(32)` に変更。
```cpp
:459      const __m256 vScale = _mm256_set1_ps(FFT_MAGNITUDE_SCALE);
         alignas(32) float mags[8];            // ★ ループ外へ移動 + alignas(64)→(32)
:461      for (; i < vEnd; i += 8)
:462      {
...
            // :474 の alignas(64) float mags[8]; を削除
:475          _mm256_store_ps(mags, mag);
```
**テスト**: 出力が変わらない。
**リスク**: Low（パフォーマンス最適化）。

### 3-28. Bug 3-8 cachedLatency 例外安全性 — **現状維持（§10 確定）**

`updateLatencyCache` の exchangeAtomic パターンで公開前確保・二重解放なし。現状維持で安全。

### 3-29. Bug 3-9 /fp:fast 精度低下（Medium / P3）— 設計判断 → **実施確定・ConvoPeq 個別へ**

**現状（Confirmed・精緻化）**: `CMAKE_CXX_FLAGS_RELEASE` の実体は **2 組のみ**（:1283 MSVC `/O2 /fp:fast /Gw /Gy /Zi` / :1360 icx `/O2 /QxCORE-AVX2 /fp:fast /Gy /Zi`）。想定の :1143/:1219 は存在しない。

**別視点調査（2026-08-11）で確定**:
- **既に `#pragma float_control(precise, on)` で /fp:fast を打ち消すファイルが 4 つ存在**: `MKLNonUniformConvolver.cpp:33-35`, `ConvolverProcessor.Runtime.cpp:1-3`, `LatticeNoiseShaper.h:2-4`, `Fixed15TapNoiseShaper.h:3-5` → **カーネル系 DSP は既に precise で動作しており /fp:fast 前提ではない**
- NaN/デノーマル対策は bit-pattern 実装（DspNumericPolicy.h:147）+ スレッド別 FTZ/DAZ（runtime set）
- テストは現在 /fp:fast でビルド（`EQProcessorMaxGainTests.cpp:719` に「/fp:fast では ε=1e-9」明記）→ **/fp:precise 化で数値はより決定的になり ε は厳しくできる**（乖離リスクは「低下」）
- **icx 特有**: `:1346` の「/fp:precise + /Qimf-arch-consistency:true はメモリ枯渇」は**組合せのみ言及**。単独 /fp:precise の OOM は未検証 → 事前実験が必要

**修正方針（実施確定）**:
```cmake
# CMAKE_CXX_FLAGS_RELEASE から /fp:fast を除去し、ConvoPeq ターゲットのみに適用
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /Gw /Gy /Zi /utf-8 /EHsc")  # MSVC
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /QxCORE-AVX2 /Gy /Zi /utf-8 /EHsc")  # icx（/fp:fast 除去）
target_compile_options(ConvoPeq PRIVATE $<$<CONFIG:Release>:/fp:fast>)  # 両コンパイラ
```
- 既存 `float_control(precise,on)` 4 ファイルは移行後も維持（precise 二重指定で無害）
- `EQProcessorMaxGainTests.cpp:719` の ε コメントを更新
- **icx の単独 /fp:precise ビルドで OOM しないことを事前実験**（:1346 の注記は組合せのみ言及のため）

**リスク**: Low〜Medium（フラグ変更。ctest 全数通過 + EQ 出力比較必須。icx の OOM 事前検証必須）。

### 3-30. Bug 3-10 /QxCORE-AVX2 AMD 非互換（Medium / P3）— 設計判断 → **実施確定・MSVC と同型に**

**現状（Confirmed・精緻化）**: `/QxCORE-AVX2` の実体: **グローバル（icx）:1360-1361**（バグの本体）+ ターゲット個別（MTNU:858 / ConvoPeq:1376 / Harness:1623）。MSVC は既に個別適用（/arch:AVX2 at :856/:1296/:1621）。

**別視点調査（2026-08-11）で確定**:
- 問題の本質は「icx のテスト・ツール全ターゲットがグローバル経由で /QxCORE-AVX2 → 全 TU が AVX2+FMA 前提で自動ベクトル化 → **AVX2 非対応 CPU で #UD + ランタイム検出 CpuFeatureCheck の無意味化**」
- **AVX2 依存の `MKLNonUniformConvolver.cpp:104` の `#ifndef __AVX2__ #error` をビルドする 3 ターゲットは既に target フラグ保有**（ConvoPeq :1034 / Harness :1601 / MTNU :828）→ **グローバル除去してもコンパイルは壊れない**（MSVC は既にその状態でテスト通過実績あり）
- AVX2 依存コード 14 ファイルは `__AVX2__` ガード + scalar フォールバック併存

**修正方針（実施確定）**:
```cmake
# icx: CMAKE_CXX_FLAGS_RELEASE（:1360）から /QxCORE-AVX2 を除去
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /Gy /Zi /utf-8 /EHsc")
# 個別適用は既存の :1376（ConvoPeq）/ :1623（Harness）/ :858（MTNU）で維持
```
- `CpuFeatureCheck.cpp` のランタイム検出が機能するのはこの限定化後（検出 → scalar フォールバック）
- `MKLNonUniformConvolver.h` に `__m256d` inline が header にある場合、`__AVX2__` 未定義 TU での展開に注意（実装時要再確認）

**リスク**: Low（フラグ分離。MSVC と同型化。AMD/AVX2 非対応環境でのビルド実行確認が必須）。

### 3-31. R-9 CMAKE_CXX_FLAGS_RELEASE グローバル上書き（Medium / P3）— 設計判断 → **3-9/3-10 と一体で実施**

**現状（Confirmed・精緻化）**: `set(CMAKE_CXX_FLAGS_RELEASE ...)` の実体は **2 組のみ**（:1283 MSVC / :1360 icx）。全ターゲットに `/fp:fast`（MSVC/icx）と `/QxCORE-AVX2`（icx）を適用。`target_compile_options` と併用（ConvoPeq :1296/1376, Harness :1621/1623, MTNU :856/858）。

**別視点調査（2026-08-11）で確定**:
- MSVC は既に `/arch:AVX2` を 3 ターゲット個別適用済み（グローバルには含まれない）
- icx の `/QxCORE-AVX2` グローバルと `/fp:fast` が残課題
- `target_compile_options(ConvoPeq PRIVATE /MT)`（Release）・`/Qipo`（icx Release）・`-mvzeroupper`（icx C++）も既存

**修正方針（設計判断 → 実施確定・3-9/3-10 と一体）**:
- **3-9（/fp:fast）と 3-10（/QxCORE-AVX2）のターゲット固有化が完了すれば R-9 も解消**
- `CMAKE_CXX_FLAGS_RELEASE` を最小限に（`/O2 /DNDEBUG /Gy /Zi /utf-8 /EHsc`）し、最適化フラグ（/fp:fast, /arch:AVX2, /QxCORE-AVX2）を target_compile_options で 3 ターゲット（ConvoPeq/Harness/MTNU）に個別適用
```cmake
# Remove: set(CMAKE_CXX_FLAGS_RELEASE "/O2 ... /fp:fast ... /QxCORE-AVX2 ...")
# Use per-target:
target_compile_options(ConvoPeq PRIVATE /O2 /Ob2 /DNDEBUG /arch:AVX2 /Gy /Zi)
```
**リスク**: Medium（ビルド設定の大幅変更。全ターゲットのフラグ整合確認 + ctest 全数通過が必須。3-9/3-10 と順序立てて実施）。

### 3-32. R-新規A IncrementalRebuildJob::reset() NUC リーク（P2）— 実施確定

**現状（Confirmed・潜伏）**: `IncrementalRebuildJob::reset()`（Rebuild.cpp:110-134）が `pendingConv` を `~StereoConvolver() + aligned_free` で直接破棄。`StereoConvolver` デストラクタは空（NUC 解放は destroyStereoConvolver に実装）→ NUC エンジンリーク。現状デッドコード経路で未発火（Debug で jassert 発火）。

**修正方針（実施確定）**:
```cpp
// Rebuild.cpp reset() — After
if (pendingConv != nullptr) {
    retireStereoConvolver(pendingConv, 0);  // 正規の破棄経路（NUC + IR 解放）
    pendingConv = nullptr;
}
```
**テスト**: ビルド + Debug で jassert なし。既存テスト通過。
**リスク**: Low（デッドコード経路の修正。将来の有効化時に必須）。

### 3-33. R-新規B Incremental rebuild 未接続（P2）— 設計判断 → **確定（休眠維持）**

**現状（Confirmed）**: `rebuildJob` 未確保・`beginIncrementalRebuild`/`advanceIncrementalRebuild`/`resetIncrementalRebuild` は宣言のみ未定義（ConvolverProcessor.h:491-493）・`runIncrementalBuildStep`/`runIncrementalFinalizeStep`/`setUseIncrementalRebuild`/`isIncrementalRebuildEnabled` は呼び出し元ゼロ・Stage 状態機械は Prepared/FinalizingPrepare 未生成。

**別視点調査（2026-08-11）で確定**:
- `rebuildJob` は**どこからも確保されない**（`make_unique<IncrementalRebuildJob>` は 0 件）・未定義 3 関数は呼び出すとリンクエラー
- 一括 rebuild は既に非 RT スレッド（rebuildThreadLoop）で実行され、crossfade/retire 機構で切替安全性を確保済み。**Audio 処理は rebuild 中も継続するため incremental 化の実害回避効果は限定的**（rebuild 完了待ちのブロッキング短縮のみ）
- **リスク**: 設計未完のまま有効化すると R-新規A（NUC リーク）が顕在化する
- **実装コスト**: 中規模（約200〜400行）・中リスク（既存の一括 rebuild と並立するため状態遷移の整合が必須）

**修正方針（設計判断 → 確定）**:
- **Option 1 を確定（推奨）**: 休眠のまま維持し、以下を実施
  1. `reset()`（Rebuild.cpp:113-117）の `~StereoConvolver()+aligned_free` を `retireStereoConvolver(pendingConv, 0)` に変更（R-新規A、4行・低リスク・将来有効化の前提条件）
  2. 未定義 3 関数（ConvolverProcessor.h:491-493）に `jassertfalse` スタブを置く（リンクエラー要因の除去）
  3. ConvolverProcessor.h:491-493 に「休眠（将来拡張）」のコメント明記
- **Option 2（一式削除）は不採用**: 将来機能として必要になった際の再実装コストが大きいため
**リスク**: Medium（設計判断。`invalidatePendingLoads` が live 呼び出し（PrepareToPlay.cpp:287）で no-op である点に注意）。

### 3-34. R-新規C bad_alloc 時リーク（P3）— 設計判断 → **確定（runSynchronously 優先）**

**現状（Confirmed）**: `rebuildAllIRsSynchronous`→`runSynchronously`→`applyNewState(async=false)`（LoaderThread.cpp:367-383 / LoadPipeline.cpp:680）で bad_alloc 時に新エンジンがリーク。例外は rebuildThreadLoop で握り潰されサイレント（OOM 時のみ）。

**別視点調査（2026-08-11）で確定**:
- **リーク経路**: ①`runSynchronously`（LoaderThread.cpp:376-377）の `make_unique<AudioBuffer>`、②`applyNewState`（LoadPipeline.cpp:687）の `make_unique<PendingCommit>` が `bad_alloc` を投げると、`conv`（`newConv` の新 StereoConvolver、生ポインタ）が誰にも所有されずリーク
- 例外は `rebuildThreadLoop` の try（RebuildDispatch.cpp:818）/catch（:1234-1242）で捕捉・握り潰し（DBG ログのみ）→ クラッシュしないが**サイレントリーク**
- **優先順位: ① > ②**（①の runSynchronously 側で防衛すれば ①② 両経路をカバーできる）

**修正方針（設計判断 → 確定）**: `runSynchronously` を try/catch で包み、失敗時に `conv` を `retireStereoConvolver` する。
```cpp
// LoaderThread.cpp:367-386（① runSynchronously）
if (result.success)
{
    auto* conv = std::exchange(result.newConv, nullptr);
    stepResult.newConv = nullptr;
    try
    {
        auto loadedIR  = std::make_unique<juce::AudioBuffer<double>>(std::move(result.loadedIR));
        auto displayIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.displayIR));
        owner.applyNewState(conv, std::move(loadedIR), result.loadedSR, result.targetLength,
                            isRebuild, file, result.scaleFactor, std::move(displayIR), /*async=*/false);
    }
    catch (...)
    {
        owner.retireStereoConvolver(conv, 0);   // ★ リーク防止（NUC エンジン解放）
        throw;                                  // rebuildThreadLoop が捕捉
    }
}

// LoadPipeline.cpp:687（② applyNewState — ①実施でカバーされるが保険として）
std::unique_ptr<PendingCommit> commit;
try { commit = std::make_unique<PendingCommit>(); }
catch (...) { retireStereoConvolver(newConv, 0); throw; }
```
**リスク**: Low（OOM 限定のため実害低。ただし将来の安定性向上に有効）。

### 3-35. R-新規D executePendingCommit コメント不整合（P3）— 実施確定

**現状（Confirmed）**: `executePendingCommit`（LoadPipeline.cpp:766）の「Message Thread のみ」コメントが Rebuild Thread からの同期実行（applyNewState async=false）と不一致。機能バグなし。

**修正方針（実施確定）**: コメントを「Message Thread または Rebuild Thread（applyNewState async=false の同期経路）から呼ばれる。内部は全てスレッド安全」に修正。
**リスク**: Low（コメントのみ）。

### 3-36. R-新規E setUseIncrementalRebuild 過去バグ — **監査記録のみ（§10 確定）**

`setUseIncrementalRebuild`（Rebuild.cpp:279-284）は enable を正しく使用（過去の「常に false」バグは修正済み）。`postCoalescedChangeNotification` の合体パターン・atomic 整合・IRState ペアリングは問題なし。改修不要。

---

## 4. 実装順序

**フェーズ 0（即時改修・低リスク）**:
1. Bug 1-3（_mm256_storeu_pd 化）— Critical クラッシュ防止
2. Bug 1-1（nucHCMode/nucLCMode 永続化）— ユーザーデータ消失防止
3. Bug 1-6（IPP FFT 戻り値）— 無音/NaN 防止
4. Bug 1-2（lastResortQueue_ 値初期化）
5. Bug 1-5（musicalSoftClip 削除）
6. Bug 2-1（SoftClipPadeApproxPolicy リネーム、Bug 1-4 と連動）
7. Bug 2-2（DC 後スクラブ）、2-3（テスト矛盾）、2-4（memcpy）、2-5（size クランプ）、2-9（saturating）、2-10（enum 検証）
8. Bug 3-4（アライメントアサート）、3-5（fence）、3-7（alignas 移動）
9. R-新規A（reset() リーク修正）、R-新規D（コメント修正）

**フェーズ 1（設計判断・RT 安全）**:
10. Bug 1-7（ISRRetire リネーム + コメント強化）— ビルド + ISRSoakTests + ctest
11. Bug 2-6（RCUReader ハッシュ→ID）
12. Bug 2-7（atomic<DSPHandle> abort 検証）
13. Bug 1-4（fastTanh 統合）
14. Bug 1-10（m_pendingIRChange 遅延クリア）

**フェーズ 2（規模大・設計判断）**:
15. Bug 1-8（LoaderThread ストリーミング読み込み）
16. Bug 1-9（fftSize 型衛生 — 割当 cast + :778 乗算順序・最小修正）
17. Bug 3-1（AudioSegmentBuffer バージョンカウンタ）
18. Bug 3-6（SnapshotFactory NaN）
19. Bug 3-9 / 3-10 / R-9（CMake フラグのターゲット固有化）
20. R-新規B（incremental rebuild 休眠維持 — reset() の retire 化 + 未定義関数スタブ化）
21. R-新規C（runSynchronously を try/catch + retire 化）

**各フェーズの検証**: ビルド（Debug/Release、MSVC/icx）→ ctest 全数 → CI スクリプト（lint / tiered verification）→ 該当機能のテスト追加。

---

## 5. 検証計画

| 検証項目 | 方法 | 対象 |
|---|---|---|
| ビルド | `build.bat Debug nopause` / `build.bat Release icx nopause` / MSVC | 全フェーズ |
| 単体テスト | `ctest -C Debug --output-on-failure` | 全フェーズ（28/28 PASS 維持） |
| 静的検査 | `check-audioengine-lint.ps1` / `check-src-atomic-dotcall.ps1` | 全フェーズ |
| CI 整合 | `isr-run-tiered-verification.ps1 -Tier standard` | 全フェーズ |
| RT 安全 | ISRSoakTests / AudioEngineHarness | フェーズ1（1-7, 2-6, 2-7） |
| 数値精度 | EQ 出力比較（/fp:precise 化前後） | 3-9 |
| メモリ | 2GB 超 IR の読み込み | 1-8 |
| アライメント | AddressSanitizer + 非アライド割当 | 1-3, 3-4 |
| データ競合 | TSan | 2-5, 3-1, 1-2 |
| データ完全性 | NaN/Inf 入力を DC ブロッカー通過後に検出 | 2-2 |
| リグレッション | 既存プリセット/セッション保存再読込 | 1-1 |

---

## 6. 実施上の注意

1. **Bug 1-4 / 2-1 連動**: `SoftClipPadéPolicy` リネームは DSPCoreDouble.cpp:127,191 の使用箇所と FastTanhApprox.h の定義を同時に更新（リネーム中途でビルドを跨がない）。
2. **Bug 1-7 は ISR の中核**: `emitRetireIntentRT` のロックフリー化は ISRRetire の overflowRing_ の契約（tryPush 非ブロック）を確認してから実施。ISRSoakTests での長期検証が必須。
3. **Bug 1-8 / 1-9 は読み込みパス**: ストリーミング化と int64_t 化は IR 読み込みパス全体の整合確認が必須（ResampleAndFallback.cpp の MAX_FILE_LENGTH ガードと連動）。
4. **CMake フラグ（3-9/3-10/R-9）**: ターゲット固有化は全ターゲットのフラグ整合を確認し、ctest 全数通過を以て完了とする。icx の `/fp:fast` デフォルト要件に注意。
5. **R-新規A/B**: incremental rebuild の休眠状態を壊さない（現状は rebuildJob==null で安全）。R-新規A の修正は将来の有効化に備えた予防的対応。
