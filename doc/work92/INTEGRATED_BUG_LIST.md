# ConvoPeq 統合バグリスト（work92 統合版）

- **作成日**: 2026-09-09
- **統合元**: 以下 3 リストを 1 つに統合し、重複を解消した
  1. `doc/work88/big_bug/INTEGRATED_BUG_LIST.md`（2026-07-30 作成・2026-08-11 更新・重症〜低症・検証済み）
  2. `doc/work88/mini_bugs_unchecked/INTEGRATED_BUG_LIST.md`（2026-09-09 作成・mini bugs 35 件の統合・**ステータス記載が古い**）
  3. `doc/work89/INTEGRATED-BUG-LIST.md`（2026-08-08 作成・2026-08-12 再検証・BUG-047〜065 全 19 件修正済み確認）
- **BUG 番号体系**: BUG-001〜046（mini bugs 系・2026-07-26 発見）+ BUG-047〜065（work89 系・2026-07-26 発見）+ big_bug 固有番号（1-1〜1-10 / 2-1〜2-10 / 3-1〜3-10 / R-新規A〜E / 9-1〜9-56 系の設計課題）

---

## 0. 最重要: ステータスの正（どの記載を信じるか）

3 ソースの間で **同一 BUG 番号のステータス記載が矛盾**している。統合の結論は以下のとおり:

> ### ✅ 判定の正: `big_bug` の R-17〜R-38 検証記録（2026-07-30〜08-11）と `work89` の §15 再検証（2026-08-12）
>
> mini_bugs_unchecked（2026-09-09 統合時点）は「**修正: 未**」の古いヘッダを保持しているが、
> その多くは **big_bug R-17〜R-38 で ✅ Fixed 判定が実コード照合により確定済み**（commit 444c2f3・2026-08-05 に一括導入）。
> **mini 個別ファイルのヘッダは未更新なだけであり、実態は修正済み**。

### 0-1. ステータス矛盾の解消表（26 重複番号）

| BUG | mini_unchecked 記載 | big_bug 判定（実コード照合） | **統合判定** |
|---|---|---|---|
| 011 | 修正: 未 | （R 記録なし） | **未修正（OPEN）** — sigma クランプ 3 変体の 1 つ |
| 012 | 修正: 未 | （R 記録なし） | **未修正（OPEN）** |
| 013 | 修正: 未 | （R 記録なし） | **未修正（OPEN）** |
| 014 | 修正: 未 | R-21 ✅ Fixed（atomic<MmcssPolicy> 化） | **修正済み** |
| 015 | 修正: 未 | R-19 ✅ Fixed（戻り値チェック実装） | **修正済み** |
| 016 | 修正: 未 | R-18 ✅ Fixed（isfinite ガード追加） | **修正済み** |
| 018 | （明記なし） | R-26 ✅ Fixed（`!= 1.0` パターン消滅） | **修正済み** |
| 019 | （明記なし） | R-25 ✅ Fixed（static_cast\<size_t\> 化） | **修正済み** |
| 021 | （明記なし） | R-27 ✅ Fixed（GlobalGuard 追加） | **修正済み** |
| 022 | （明記なし） | R-28 ✅ Fixed（GlobalGuard 追加） | **修正済み** |
| 024 | （明記なし） | R-36 ✅ Fixed（fadeGeneration_ 再確認） | **修正済み** |
| 026 | （明記なし） | （R 記録なし） | **要再確認（OPEN 候補）** — big_bug §8 で「調査済み」とのみ記載 |
| 028 | （明記なし） | R-33 ✅ Fixed（stale flag リセット） | **修正済み** |
| 029 | （明記なし） | R-20 ✅ Fixed（exchangeFadingRuntimeDSP 呼出追加） | **修正済み**（work89 §15.3 N-1 で意図的設計と再確認） |
| 031 | （明記なし） | R-29 ✅ Fixed（スタブ関数削除） | **修正済み** |
| 033 | （明記なし） | R-30 ✅ Fixed（dryScale 適用追加） | **修正済み** |
| 034 | （明記なし） | R-35 ⚠️（当該ファイルに実呼び出しなし → BUG 1-6 = MklFftEvaluator.h へ参照） | **修正済み相当**（BUG 1-6 は P0 残課題のまま・下記 §2 参照） |
| 035 | （明記なし） | R-24 ✅ Fixed（RAII ガード化） | **修正済み** |
| 036 | （明記なし） | R-23 ✅ Fixed（init 成功時のみ release） | **修正済み** |
| 037 | （明記なし） | R-37 ✅ Fixed（forceCleanup stopThread(500)） | **修正済み** |
| 038 | （明記なし） | R-22 ✅ Fixed（レポート自体が古いバージョン基準） | **修正済み（誤報）** |
| 039 | （明記なし） | R-17 ✅ Fixed（std::min ガード追加） | **修正済み** |
| 041 | （明記なし） | R-34 ✅ Fixed（VLA 消滅） | **修正済み** |
| 042 | （明記なし） | R-31 ✅ Fixed（`= delete` 4 種追加） | **修正済み** |
| 045 | （明記なし） | R-32 ✅ Fixed（sourceRate 維持） | **修正済み** |
| 046 | （明記なし） | R-38 ✅ Fixed（`= delete` 4 種追加） | **修正済み** |

### 0-2. 番号の一意性

- **BUG-001〜010**: 3 ソースいずれにも収録なし（別管理と推定）
- **BUG-017**: 欠番（該当ファイルなし）
- **BUG-047 の重複ファイル**（`BUG-047-EQCoeffCache-ハッシュにsampleRate不足.md`）: work89 §4 で同一バグの重複と判定済み → 相互リンク化運用
- 統合後の **ユニーク BUG 番号 = 55**（011〜046 の 35 + 047〜065 の 19 + BUG-001 は番号採番のみ）

---

## 1. 統合バグ一覧（現行ステータスつき・全 55 件）

> ステータス: **OPEN** = 未修正確認 / **FIXED** = 実コード照合で修正確認済み / **CONFIRMED** = 実在確認・設計判断待ち / **REJECTED** = 誤報・非問題 / **DORMANT** = 潜伏（デッドコード経路で未発火）
> 詳細は各ソースファイル参照（「詳細」列）。

### 1-1. HIGH / Critical（実害・クラッシュ・データ消失系）

| BUG | タイトル | ステータス | 詳細 |
|---|---|---|---|
| 011 | CmaEsOptimizer::deserializeFrom sigma クランプなし（除算-by-ゼロ） | **OPEN** | mini BUG-011 |
| 012 | CmaEsOptimizerDynamic::setSigma クランプなし | **OPEN** | mini BUG-012 |
| 013 | CmaEsOptimizerDynamic::deserializeFrom クランプなし（除算-by-ゼロ） | **OPEN** | mini BUG-013 |
| 014 | juce::String CoW データ競合（UAF 潜在） | **FIXED**（R-21） | mini BUG-014 / big 1-7 参照 |
| 047 | EQCoeffCache ハッシュに sampleRate 不足 | **FIXED**（444c2f3） | work89 BUG-047 |
| big 1-1 | nucHCMode/nucLCMode がセッション永続化から欠落 | **CONFIRMED** | big_bug §1-1 |
| big 1-2 | coordinatorDeferredRing_/lastResortQueue_ デッドコード | **CONFIRMED**（方針確定: 値初期化のみ・producer 実装せず） | big_bug §1-2 / §10-3-1 |
| big 1-3 | `_mm256_store_pd` アライメント保証なし（#GP クラッシュ） | **CONFIRMED**（P0） | big_bug §1-3 |
| big 1-7 | ISRRetire.cpp Mutex 使用（RT 違反リスク） | **CONFIRMED**（P0） | big_bug §1-7 |
| big 1-8 | LoaderThread OOM リスク（16GB 確保試行） | **CONFIRMED**（P0） | big_bug §1-8 |
| big 1-9 | MKLNonUniformConvolver int/size_t 混在 | **CONFIRMED**（P0） | big_bug §1-9 |
| big 1-10 | m_pendingIRChange 公開前クリア（IR 変更要求消失） | **CONFIRMED**（P1） | big_bug §1-10 |

### 1-2. MEDIUM（機能劣化・リーク・競合系）

| BUG | タイトル | ステータス | 詳細 |
|---|---|---|---|
| 015 | enqueueWithRetry 戻り値無視によるサイレントドロップ | **FIXED**（R-19） | mini BUG-015 |
| 016 | sanitize が NaN/Inf を処理しない | **FIXED**（R-18） | mini BUG-016 |
| 021 | timerCallback が RCU reader なしで engine にアクセス | **FIXED**（R-27・GlobalGuard 追加） | mini BUG-021 |
| 022 | prepareToPlay が RCU reader なしで engine データにアクセス | **FIXED**（R-28・GlobalGuard 追加） | mini BUG-022 |
| 024 | SnapshotFadeState advance vs resetToIdle 競合 | **FIXED**（R-36） | mini BUG-024 |
| 028 | CrossfadeRuntime::complete が stale flag を残す | **FIXED**（R-33） | mini BUG-028 |
| 029 | DSPTransition Emergency Override が exchangeFadingRuntimeDSP を呼ばない | **FIXED**（R-20・意図的設計） | mini BUG-029 / work89 N-1 |
| 031 | updateAudioThreadSnapshotFade() スタブ | **FIXED**（R-29・スタブ関数削除） | mini BUG-031 |
| 035 | applyComputedIR 世代不一致で isLoading 固着 | **FIXED**（R-24） | mini BUG-035 |
| 036 | init() 失敗時に irL/irR リーク | **FIXED**（R-23） | mini BUG-036 |
| 037 | loaderTrashBin dangling reference（UAF） | **FIXED**（R-37） | mini BUG-037 |
| 039 | Oversampler passthrough 時 buffer overread | **FIXED**（R-17） | mini BUG-039 |
| 045 | resample 失敗時のサンプルレート誤ラベル | **FIXED**（R-32） | mini BUG-045 |
| 048 | detectStuckReaders が最初の一致で break | **FIXED**（3 パス評価） | work89 BUG-048 |
| 049 | quarantineFlags 並行 store 競合 | **FIXED**（CAS 化） | work89 BUG-049 |
| 050 | enterReader の HB 順序（epoch store が depth++ の後） | **FIXED** | work89 BUG-050 |
| 052 | consumeDeferredRequest の non-atomic アクセス | **FIXED**（DeferredPublishView 再設計） | work89 BUG-052 |
| 053 | stopLearning() 二重呼び出し | **FIXED** | work89 BUG-053 |
| 054 | onPublishCompleted の crossfade handle 不一致 | **FIXED** | work89 BUG-054 |
| 056 | crossfade MonitorState 永久 Error 貼り付き | **FIXED**（Normal 復帰パス） | work89 BUG-056 |
| 058 | checkWorldConsistency の state 誤流用 | **FIXED**（専用 state 新設） | work89 BUG-058 |
| 059 | reset() の MonitorState リセット不整合 | **FIXED**（全 Normal 化） | work89 BUG-059 |
| 060 | quarantineResidentCount TOCTOU underflow | **FIXED**（fetchSub 単一アトミック化） | work89 BUG-060 |
| 061 | DSPQuarantineManager vector データ競合 | **FIXED**（mutex 保護） | work89 BUG-061 |
| 062 | uint64 版 retire age 復帰イベント欠如 | **FIXED** | work89 BUG-062 |
| 064 | Float/Double 出力パスの clamp/delay 順序不一致 | **FIXED**（clamp→delay 統一） | work89 BUG-064 |
| 065 | EQProcessor::reset() が AGC リセットを発火しない | **FIXED**（serial increment 化）+ **残存リスク 1 件**（rtSeenAgcResetSerial 直接書込・D-1 案 A 確定・実装は別作業） | work89 BUG-065 / §7-§10 |
| big 1-4 | fastTanh 3 箇所独立複製（係数乖離リスク） | **CONFIRMED**（P1） | big_bug §1-4 |
| big 1-6 | IPP FFT 戻り値無視（MklFftEvaluator.h:270-271, 425-426） | **CONFIRMED**（P0・未修正） | big_bug §1-6 / BUG-034 参照 |
| big 2-5 | LockFreeRingBuffer::size() データ競合 | **CONFIRMED** | big_bug §2-5 |
| big 2-6 | RCUReader::enter() ハッシュ衝突リスク | **CONFIRMED**（P1） | big_bug §2-6 |
| big 2-7 | atomic\<DSPHandle\> ロックフリー検証不足（Release） | **CONFIRMED**（P1） | big_bug §2-7 |
| big 2-9 | タイミング計算 uint64 underflow | **CONFIRMED**（P1） | big_bug §2-9 |
| big 2-10 | NoiseShaperType enum キャスト検証欠如 | **CONFIRMED**（P1） | big_bug §2-10 |

### 1-3. LOW / 品質・将来リスク系

| BUG | タイトル | ステータス | 詳細 |
|---|---|---|---|
| 018 | FP `!= 1.0` 等価比較 | **FIXED**（R-26） | mini BUG-018 |
| 019 | TruePeakDetector int オーバーフロー | **FIXED**（R-25） | mini BUG-019 |
| 020 | jlimit 下限 > 上限 UB | （明記なし・LOW） | mini BUG-020 |
| 023 | SafeStateSwapper tryReclaim と swap の競合 | （明記なし） | mini BUG-023 |
| 025 | switchImmediate 経由の enqueueRetry 未使用リーク | （明記なし・015 と同根） | mini BUG-025 |
| 026 | ObservedRuntime::get() が rootEnterSucceeded を確認しない | （明記なし・要再確認） | mini BUG-026 |
| 027 | completeFade と updateFade の競合 | （明記なし） | mini BUG-027 |
| 030 | Timer exchangeFadingDSP(nullptr) と DSPTransition 競合 | （明記なし） | mini BUG-030 |
| 032 | createSnapshotFromCurrentState torn-read | （明記なし） | mini BUG-032 |
| 033 | BlockDouble クロスフェード dryScale 未適用 | **FIXED**（R-30） | mini BUG-033 |
| 034 | IPP FFT 戻り値未チェック（7 箇所） | **FIXED 相当**（R-35: 実態は BUG 1-6 に集約） | mini BUG-034 |
| 038 | SpectrumAnalyzer +6dB スケーリング誤差 | **FIXED**（R-22・誤報） | mini BUG-038 |
| 040 | NSL 再生時間 1Hz フォールバック | （明記なし） | mini BUG-040 |
| 041 | NSL VLA スタック破壊 | **FIXED**（R-34） | mini BUG-041 |
| 042 | CmaEsOptimizer Rule of Five 違反 | **FIXED**（R-31） | mini BUG-042 |
| 043 | IRConverter sampleRate パラメータ未使用 | （明記なし・Low） | mini BUG-043 |
| 044 | MklFftEvaluator Rule of Five 違反 | （明記なし・Medium） | mini BUG-044 |
| 046 | PsychoacousticDither Rule of Five 違反 | **FIXED**（R-38） | mini BUG-046 |
| 051 | exchangeFadingRuntimeDSP sentinel デッドコード | **FIXED** | work89 BUG-051 |
| 055 | runPublicationPrecheckNonRt 到達不能 else if | **FIXED** | work89 BUG-055 |
| 057 | checkOverflowRate の eventCode 誤用 | **FIXED**（専用コード新設） | work89 BUG-057 |
| 063 | EpochDomain ownerTag data race | **FIXED**（ownerThreadId ガード） | work89 BUG-063 |
| big 1-5 | musicalSoftClip デッドコード | **CONFIRMED**（削除方針確定） | big_bug §1-5 / §10-3-2 |
| big 2-1 | 非 ASCII 識別子 SoftClipPadéPolicy | **CONFIRMED**（P2） | big_bug §2-1 |
| big 2-2 | 入力側 DC ブロッカー NaN/Inf 非対称 | **CONFIRMED**（P2） | big_bug §2-2 |
| big 2-3 | ユニットテスト矛盾条件 | **CONFIRMED**（P2） | big_bug §2-3 |
| big 2-4 | CacheManager strict-aliasing 違反 | **CONFIRMED**（P2） | big_bug §2-4 |
| big 2-8 | cleanup() 強制削除未実装 | **CONFIRMED**（役割分担正常・対応不要確定） | big_bug §2-8 / §10-3-3 |
| big 3-1 | AudioSegmentBuffer リングラップ競合 | **CONFIRMED**（SPSC 緩和） | big_bug §3-1 |
| big 3-2 | DeferredDeletionQueue kMaxScan デッドコード | **CONFIRMED**（安全） | big_bug §3-2 |
| big 3-3 | AlignedAllocation 例外 RT 伝播 | **CONFIRMED**（RT パス確保なし確定） | big_bug §3-3 / §10-3-4 |
| big 3-4 | MKLNonUniformConvolver アライメント判定 | **CONFIRMED**（実害なし） | big_bug §3-4 |
| big 3-5 | CacheManager volatile sink | **CONFIRMED** | big_bug §3-5 |
| big 3-6 | SnapshotFactory NaN ハッシュ不一致 | **CONFIRMED** | big_bug §3-6 |
| big 3-7 | SpectrumAnalyzer alignas ループ内 | **CONFIRMED** | big_bug §3-7 |
| big 3-8 | cachedLatency 例外安全性 | **CONFIRMED**（安全確認済み） | big_bug §3-8 / §10-3-5 |
| big 3-9 | CMakeLists /fp:fast | **CONFIRMED**（icx では要件・現状維持許容） | big_bug §3-9 / §10-3-6 |
| big 3-10 | CMakeLists /QxCORE-AVX2（AMD 非互換） | **CONFIRMED**（icx のみ・target 移行推奨） | big_bug §3-10 / §10-3-6 |

### 1-4. big_bug 固有: 別視点調査で確定した潜伏バグ（R-新規系）

| ID | タイトル | ステータス | 詳細 |
|---|---|---|---|
| R-新規A | IncrementalRebuildJob::reset() が NUC エンジンをリーク（潜伏・デッドコード経路） | **CONFIRMED**（DORMANT） | big_bug §10-2-1 |
| R-新規B | Incremental rebuild サブシステム全体が未接続（宣言のみ未定義含む） | **CONFIRMED**（DORMANT） | big_bug §10-2-1 |
| R-新規C | rebuildAllIRsSynchronous の OOM 時サイレントリーク | **CONFIRMED**（Low） | big_bug §10-2-1 |
| R-新規D | executePendingCommit コメント不整合 | **CONFIRMED**（コメント修正のみ） | big_bug §10-2-1 |
| R-新規E | setUseIncrementalRebuild 過去バグ | **FIXED**（監査記録のみ） | big_bug §10-2-1 |

### 1-5. 誤報・非問題（REJECTED 参照記録）

以下は big_bug §4 で ❌ Rejected 判定（Markdown 破損・既存防御実装・物理的不可能等）。詳細は big_bug §4 の表を参照:
R-1（dryScaledL 構文エラー=破損）・R-2（resample 無限ループ）・R-3（AVX2 チェック）・R-4（bypass 曖昧）・R-5（NoiseShaper 未防御）・R-6（EQ 係数未検証）・R-7（MessageBox 破損）・R-8（pragma pop）・R-10（CMA-ES 型矛盾=別クラス誤認）・R-11（RTTraceRelay 未結線）・R-12（DSPQuarantineManager 未使用）・R-13（advancePhase デッドコード）・R-14（IR resample キャンセル）・R-15（audioCallbackActiveCount overflow）

---

## 2. 未修正（OPEN）バグの残置一覧 — 次回修正候補

統合の結果、**実コード照合で未修正と確定しているのは以下のみ**:

| 優先 | BUG | 内容 | 修正概要 |
|---|---|---|---|
| **P0** | 011/012/013 | sigma クランプ欠如 3 変体 | deserializeFrom/setSigma で `std::clamp(inSigma, params.sigmaMin, params.sigmaMax)`。一括修正可能 |
| **P0** | big 1-6 | IPP FFT 戻り値無視（MklFftEvaluator.h:270-271, 425-426） | IppStatus チェック + エラー時 nullptr return |
| **P0** | big 1-1 | nucHCMode/nucLCMode 未永続化 | getState/setState に setProperty/読込追加 |
| **P0** | big 1-3 | `_mm256_store_pd` アライメント | `_mm256_storeu_pd` 変更（InputBitDepthTransform.h:114-115 ほか検証） |
| **P0** | big 1-7 | ISRRetire mutex RT 違反 | RT パス 2 段階化 |
| **P0** | big 1-8 | LoaderThread OOM | ストリーミング読み込み |
| **P0** | big 1-9 | int/size_t 混在 | fftSize を int64_t 化 |
| P1 | big 1-10 | m_pendingIRChange 公開前クリア | クリア遅延 |
| P1 | big 2-6 | RCUReader ハッシュ衝突 | thread_local 単調 ID |
| P1 | big 2-7 | atomic\<DSPHandle\> 検証不足 | Release でも runtime abort |
| P1 | big 2-9 | uint64 underflow | saturating subtraction |
| P1 | big 2-10 | enum キャスト検証欠如 | 範囲チェック追加 |
| P1 | 020/023/025/026/027/030/032/040/043/044 | mini bugs 明記なし系（10 件） | 修正着手前に現行 HEAD で該当箇所の実在性を再確認（下記 §4 参照） |
| P2 | big 2-1〜2-5, 2-8 / big 3-1〜3-10 | 品質・将来リスク系 | big_bug §5 の P2/P3 分類どおり |
| P2 | R-新規A/B/C/D | 潜伏系（デッドコード経路） | big_bug §10-2-1 の修正方針どおり |
| — | BUG-065 残存 | rtSeenAgcResetSerial 直接書込（data race 潜在） | work89 §7 案 A 確定済み・実装は別作業（D-1） |

---

## 3. work89 系の特記事項（BUG-047〜065）

- **全 19 件修正済み**（単一コミット 444c2f3・2026-08-05・34 ファイル +1569/−519 で一括導入）
- 唯一の残存リスク: **BUG-065** の `rtSeenAgcResetSerial = 0` 直接書込（data race 潜在・ただし `EQProcessor::reset()` はデッドコード確定のため現状未実行）。改修案 A（rt シャドウ書込全廃 + bandResetPacked serial 前進クリア）は work89 §7 で設計確定済み。実装は別作業
- 関連するデッドコード確定（work89 §8-§12）: `DSPCore::reset()` / `EQProcessor::reset()` / `syncStateFrom()` / `syncGlobalStateFrom()` / `syncBandNodeFrom()` / `ConvolverProcessor::syncStateFrom()` — 呼び出し元ゼロを全数確認・verifier 監視下で維持
- **N-1 教訓**: DSPTransition.h:63-65 の `exchangeFadingRuntimeDSP` は work88 BUG-029 修正の意図的設計（displacement セマンティクス）であり、CAS-only 化に置換してはならない

---

## 4. 統合時の注意事項（バグ修正着手前の必須確認）

1. **番号の重複は解消済みだが、mini_unchecked 個別ファイルのヘッダ（修正: 未）は未更新のまま残っている**。修正着手時は本リスト §0-1 の統合判定を正とし、個別ファイルのヘッダ更新を同時に行うこと
2. **2026-07-26 発見のバグの多くは大規模改修（D101〜D179 系列）で対象コードが変更済み**。特に ISR 関連（015/021/022/025/026/028/030/032）は authority 改修で構造が変わっている可能性が高く、**現行 HEAD での再確認なしに修正コードを書かないこと**（inventory stale 化の教訓 — D163/D174 参照）
3. Closed boundary（T3c / RecoveryLifecycleWord / Retire / Publish authority）に接触する修正は **D159 通常開発ゲート**（scope 確定 → P0 invariant 影響確認 → implementation contract）を通すこと
4. **BUG-026（ObservedRuntime rootEnterSucceeded）** は 3 ソース間でステータスが最も不確実（big_bug §8「調査済み」記載のみで Fixed/OPEN 判定が確定していない）。修正着手前に個別再監査を推奨
5. mini_unchecked の元個別ファイルは `doc/work88/mini_bugs_unchecked/BUG-XXX_*.md` に残存（本リスト §1-2〜1-3 の「詳細」列は元ファイル名に対応）

---

## 5. 統合元への対応付け（トレーサビリティ）

| BUG 番号帯 | 元ソース | 参照セクション |
|---|---|---|
| 001〜010 | （3 ソースに収録なし・別管理） | — |
| 011〜046 | doc/work88/mini_bugs_unchecked/（個別 35 ファイル + 同フォルダ INTEGRATED_BUG_LIST.md） | mini カテゴリ分類 |
| 014〜046 のステータス確定 | doc/work88/big_bug/INTEGRATED_BUG_LIST.md §4（R-17〜R-38）+ §1〜§3 | big_bug 各節 |
| 047〜065 | doc/work89/INTEGRATED-BUG-LIST.md（§1〜§16） | work89 各節 |
| BUG-050 の big_bug 側参照 | big_bug §10-3-1（work89 8.2 引用） | — |
| BUG-065 の big_bug 側参照 | big_bug §10-4（EQProcessor::reset デッドコード確定） | — |
| big_bug 固有（1-1〜3-10 / R-新規 / 9-x 設計課題） | doc/work88/big_bug/INTEGRATED_BUG_LIST.md §1〜§3 / §10-2-1 / §9 | — |

> **9-x 系（REPAIR_PLAN2-dash レビュー記録）について**: big_bug §9 の 9-1〜9-56 はバグではなく**設計課題・レビュー記録**（X1〜X6 / P2-1〜P2-4 の ISR 設計レビュー）であり、本統合リストの「バグ」定義からは除外する。ただし 9-1（submitRecoveryRequest push 失敗）は修正済み、9-2/9-5 は P3 降格、9-3/9-7 は撤回済みなど結論は big_bug §9 に確定記録済み。また X1〜X6 の設計はその後 D105〜D179 系列で実装・検証済み（pendingRecoveryAdmission_ / RecoveryLifecycleWord 等）。
