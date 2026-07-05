# オーディオスレッドリアルタイム性 完全調査レポート — 新規発見・見落とし・潜在リスク

**調査日**: 2026-07-05
**調査範囲**: `src/audioengine/`, `src/eqprocessor/`, `src/convolver/`, `src/core/`, `src/MKLNonUniformConvolver.*`, `src/*NoiseShaper*.h`, `src/OutputFilter.*`, `src/LoudnessMeter.*`, `src/TruePeakDetector.*`, `src/CustomInputOversampler.*`, `src/InputBitDepthTransform.h`, `src/DiagnosticsConfig.h`
**使用ツール**: context-mode MCP (JavaScript sandbox), RTK(WSL版), cocoindex-code(ccc.exe), semble, graphify, AiDex MCP, PowerShell Select-String, WSL grep/awk
**ベース文書**: `doc/work66/AudioThread_RealtimeBlockers_Analysis.md`
**既出改修計画**: `doc/work66/RT_Improvement_Plan.md`

---

## 凡例

| タグ | 意味 |
|------|------|
| ✅ | 確認済み・問題なし |
| ⚠️ | 条件付きリスクあり |
| ❌ | 明確な問題 |
| 🔥 | 直ちに対応が必要（P0相当） |

---

## 1. 既存分析文書の検証結果サマリー（前回からの引用）

| # | カテゴリ | 文書の正確性 | 修正後重要度 |
|---|---------|------------|------------|
| 1 | MMCSS system calls | ✅ 正確 | CRITICAL 維持 |
| 2 | high_resolution_clock::now() | ⚠️ 過大評価（kTraceBufferSize=4096により起動時限定）| HIGH→MEDIUM |
| 3 | RTCapabilityFirewall::enter() | ❌ publishAtomic回数誤り（2回→1回）| HIGH→MEDIUM |
| 4 | std::hash<thread::id> | ⚠️ 範囲拡大（DspNumericPolicy.h見落とし）| MEDIUM 維持 |
| 5 | atomic密度 | ⚠️ 数値過大（20+→9-16回）| MEDIUM 維持 |
| 6 | diagLog()定義 | ❌ DSPCoreFloat.cppガード欠如 | **LOW-MEDIUM→MEDIUM** |
| 7 | getCurrentTimeUs() | ⚠️ 無条件呼出4回見落とし | LOW 維持 |
| 8 | RT-safe確認 | ✅ 正確 | INFO 維持 |

---

## 2. 🔥【新規P0-Critical】— 直ちに修正すべきバグ

### 2-1. DSPCoreFloat.cpp diagLog() ガード欠如（再掲・最優先）

**既に改善計画に記載済みだが、発見された中で唯一の確定バグ。**

- **発見ツール**: rtk(grep)
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:9-13`
- **問題**: `diagLog()` 定義が `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガードされていないため、全ビルドで `juce::Logger::writeToLog()`（ミューテックス＋ファイルI/O）がコンパイルされる
- **リスク**: 誤ってホットパスから呼ばれた場合、音飛び確実。`[[maybe_unused]]` で警告も抑制されている
- **対比**: 同パターンの `DSPCoreDouble.cpp:12-16` は正しくガードあり

---

## 3. ⚠️【新規P1-High】— 確認・改善が必要な問題

### 3-1. RTLocalState 構造体の false sharing リスク

- **発見ツール**: AiDex MCP (ファイル構造分析), PowerShell Select-String
- **ファイル**: `src/audioengine/AudioEngine.h:1472-1516`
- **問題**: `RTLocalState` 構造体に `alignas(64)` が一切ない

```
struct RTLocalState {                    // ← キャッシュライン境界未指定
    std::atomic<uint64_t> audioCallbackEpochCounter;  // ← 音声スレッド書込
    std::atomic<uint64_t> audioSampleCursorCounter;   // ← 音声スレッド書込
    std::atomic<uint32_t> audioCallbackActiveCount;   // ← 音声スレッド書込
    std::atomic<uint64_t> audioThreadRetireEnqueueDropped;
    std::atomic<uint64_t> lastCallbackEndTicks;       // ← 音声スレッド書込
    std::atomic<uint64_t> xrunSequenceCounter;        // ← 音声スレッド書込
    std::atomic<uint64_t> lastActivatedGeneration;    // ← 音声スレッド書込
    // ... (さらに ~25 のフィールド + 配列) ...
    std::atomic<uint64_t> publishTimingWriteCount;    // ← Messageスレッド書込 ⚠️
    PublishTimingEntry publishTimingHistory[16];
    std::atomic<uint64_t> lastCallbackEntryUs;
    std::atomic<int64_t>  lastCallbackDriftUs;
    std::atomic<uint32_t> lastCallbackProcessor;
    std::atomic<uint64_t> cpuMigrationCount;
    std::atomic<uint64_t> lastCallbackPublicationSeq;
    CallbackTimingEntry callbackTimingHistory[32];    // ← 音声スレッド書込
    std::atomic<uint64_t> callbackTimingWriteCount;   // ← 音声スレッド書込
    uint64_t expectedCallbackIntervalUs { 0 };
    uint32_t cachedThreadId { 0 };
};  // ← alignas(64) なし
```

- **false sharing影響**: `publishTimingWriteCount` が Message Thread から書き込まれると、同一キャッシュライン上の `lastCallbackEndTicks`（音声スレッド読取）とコンフリクト。キャッシュラインのバウンスが発生する。
- **推定影響コスト**: 2つのコア間でキャッシュラインがバウンスするたびに ~1000-3000ns のペナルティ
- **対処**: `alignas(128)` で構造体全体をキャッシュライン境界に配置する。ただし、struct 内部のメンバ変数のグループ分けは別途検討が必要。

### 3-2. AudioEngine クラスのアトミック変数 false sharing（部分的緩和済みだが不十分）

- **発見ツール**: PowerShell Select-String, 直接読取
- **ファイル**: `src/audioengine/AudioEngine.h:2055-2130`
- **問題**: 一部のアトミック変数に `alignas(64)` が付与されているが、class全体のレイアウトはコンパイラ任せ

**alignas(64) あり**:
- `pendingLearningMode`, `adaptiveCaptureActiveRt`, `globalCaptureSessionId` (line 2084-2086)
- `learningCommandWrite/Read`, `learnerDispatchWrite/Read` (line 2096-2099)
- `saturationAmount` (line 2127)

**alignas(64) なし（かつ音声スレッドから書込/読取が発生するもの）**:
- `useMmcssPriority`, `mmcssApplied_` (line 2111, 2113) — 音声スレッド書込
- `mmcssShutdownRequested` (line ~2120) — Messageスレッド書込
- `manualOversamplingFactor` (line ~2130)
- `softClipEnabled` (line ~2110)
- `currentSampleRate` (line ~2105) — 音声スレッド読取

→ `mmcssShutdownRequested` と `mmcssApplied_` が同一キャッシュライン上にある可能性がある。一方は Message スレッド書込、もう一方は音声スレッド書込。

### 3-3. MKLNonUniformConvolver::IppFFTPlanCache::getOrCreate() が `std::lock_guard<std::mutex>` + 動的確保を含む

- **発見ツール**: rtk(grep), 直接読取
- **ファイル**: `src/MKLNonUniformConvolver.cpp:87-99`
- **問題**: `getOrCreate()` は `std::lock_guard<std::mutex>` で保護されており、初回呼出時は `make_unique<IppFFTPlan>()` + `ippsMalloc_8u()` の動的確保を行う
- **現状**: `prepare()` パス（Message Thread）からのみ呼ばれるため、RTパスからは呼ばれない
- **リスク**: コメントで「Audio Thread からの FFT 再初期化禁止」と明記されている。しかし、`ConvolverProcessor` のコールバックパス (`process()`) 経由で何らかの理由で `getOrCreate()` に到達した場合、**ミューテックスロック＋メモリ確保** が音声スレッド上で発生する
- **緩和策**: `getOrCreate()` の RTパスからの呼出が不可能であることを CI スキャンで検証するアサーションの追加を推奨

### 3-4. MKLNonUniformConvolver.cpp:697 の `juce::Logger::writeToLog()`（prepare()パス）

- **発見ツール**: rtk(grep), 直接読取
- **ファイル**: `src/MKLNonUniformConvolver.cpp:697`, `704-708`
- **問題**: prepare()パスのエラーハンドリングで `juce::Logger::writeToLog()` を直接呼んでいる。同じファイルの `~MKLNonUniformConvolver()` でも呼ばれている可能性がある
- **現状**: 非RTスレッドからの呼出で問題はないが、もし将来 RT パスから `prepare()` 相当コードが呼ばれた場合、ファイルI/O＋ミューテックスが音声スレッド上で発生する
- **推奨**: `Logger::writeToLog()` を `diagLog()` に置き換え、`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードと統一する

---

## 4. ⚠️【新規P2-Medium】— 計画的に対応すべき問題

### 4-1. `GetCurrentProcessorNumber()` システムコール（診断ビルド時）

- **発見ツール**: rtk(grep), 直接読取
- **ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` (複数行), `BlockDouble.cpp` (複数行)
- **問題**: 診断ビルド時、`GetCurrentProcessorNumber()`（Win32 API → カーネル遷移）を毎コールバック複数回呼ぶ
- **呼出例**: AudioBlock.cpp: `cpu = static_cast<uint32_t>(::GetCurrentProcessorNumber())` — これは ring-3からring-0への遷移が発生する可能性がある
- **推定コスト**: 数百ns〜数µs（状況依存）
- **診断ビルドのみ**: ✅ `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内でガードされている
- **補足**: `GetCurrentProcessorNumber()` は Windows 8+ では高速化されているが、それでもユーザー空間のみで完結するわけではない

### 4-2. `AudioCallbackRuntimeScope` コンストラクタ内のアトミック操作連鎖

- **発見ツール**: 直接読取
- **ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:58-76`
- **問題**: `AudioCallbackRuntimeScope` コンストラクタが以下を連続実行:
  1. `lifecycleRuntime_.enterAudioCallback()` → `transitionTo()` → `consumeAtomic` + `publishAtomic` + `fetchAddAtomic`
  2. `rtCapabilityFirewall_.enter()` → `get_id()` + `publishAtomic`
  3. `RTAllocatorFirewall::markRTContext(true)` → `publishAtomic`
  4. `fetchAddAtomic(audioCallbackActiveCount, acq_rel)`
- **影響**: コールバック開始直後に4つのアトミックRMW操作が連続。cache coherence traffic が集中する
- **スコープ終了時**: 同様に4つのアトミック操作が逆順で実行される
- **推定コスト**: 〜200-400ns合計

### 4-3. `juce::ScopedNoDenormals` オブジェクト構築（毎コールバック）

- **発見ツール**: 直接読取
- **ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:77`, `BlockDouble.cpp:78`, `EQProcessor.Processing.cpp:477`
- **問題**: `juce::ScopedNoDenormals` はコンストラクタで MXCSR レジスタの FTZ/DAZ ビットを設定、デストラクタで復元する。MXCSR は浮動小数点ユニットの制御レジスタであり、この操作は CPU の MSR 書込を伴う（ただし、同一スレッド内では一度設定すれば変わらないため、無駄が多い）
- **コスト**: MSR書込は比較的軽量だが、パイプラインへの影響（シリアライゼーション）がある
- **改善案**: スレッドローカルで FTZ/DAZ の現在状態を追跡し、変更が必要な場合のみ MXCSR を更新する
- **参考**: `EQProcessor` も独自に `ScopedNoDenormals` を構築しており、DSPCore と EQ で二重設定が発生している

### 4-4. `juce::dsp::AudioBlock<double>` スタックオブジェクト構築

- **発見ツール**: 直接読取
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:384-385`, `DSPCoreFloat.cpp`, `DSPCoreIO.cpp`
- **問題**: 毎コールバックで以下のコードが実行される:
  ```cpp
  double* channels[2] = { alignedL.get(), alignedR.get() };
  juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples);
  ```
  - Releaseビルドではポインタをラップするだけ
  - Debugビルドでは内部で範囲チェックアサーションが実行される可能性がある
  - `AudioBlock<double>` のテンプレート展開はコンパイル時に解決されるが、コンストラクタ呼出自体は毎回発生する
- **影響**: 無視できるレベルだが、診断ビルド時には注意

### 4-5. `ASSERT_AUDIO_THREAD()` 内の `isAudioThread()` が `hash<thread::id>` を毎回実行（Debugビルド）

- **発見ツール**: 直接読取
- **ファイル**: `src/DspNumericPolicy.h:158`, `146-155`
- **問題**: `ASSERT_AUDIO_THREAD()` は `isAudioThread()` を呼ぶ。`isAudioThread()` は `currentThreadTag()` (= `hash<thread::id>`) を呼び、スロット配列を線形走査する
- **現状**: Debugビルドのみ。リリースビルドでは `jassert` は消去される
- **改善案**: `ASSERT_AUDIO_THREAD()` 内でも `thread_local` キャッシュを活用する

### 4-6. `convo::isr::RTAllocatorFirewall::markRTContext()` がグローバル atomic 書込

- **発見ツール**: 直接読取
- **ファイル**: `src/audioengine/ISRRTExecution.cpp`, `AudioBlock.cpp:67-68`, `BlockDouble.cpp:~`
- **問題**: `markRTContext(true/false)` がグローバルな `std::atomic<bool>` への `publishAtomic(release)` を毎コールバック行う。この変数は Commit スレッドから `consumeAtomic(acquire)` で読まれる可能性がある
- **影響**: コア間で false sharing が発生する可能性。グローバル変数（`s_sharedRtContextFlag`）のため、全コアのキャッシュラインを汚染する

### 4-7. `DSPCoreIO.cpp` の `musicalSoftClip()` 実装

- **発見ツール**: 直接読取
- **ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
- **問題**: `musicalSoftClip()` が `musicalSoftClipScalar()` に委譲する。この関数が libm 呼出（`std::pow` など）を含んでいる場合、RTブロッカーになる
- **要確認**: `musicalSoftClipScalar` の実装を確認する必要がある（後続調査推奨）

---

## 5. ✅【確認済み問題なし】— 元文書の「RT-safe」主張の検証結果

### 5-1. 全ライブラリ呼出のRT安全性

| モジュール | 結果 | 確認方法 |
|-----------|------|---------|
| `LockFreeRingBuffer` push/pop | ✅ SPSCロックフリー | 直接読取 |
| `LockFreeAudioRingBuffer` | ✅ 同様 | 直接読取 |
| `CrossfadeRuntime` | ✅ アトミックのみ | 直接読取 |
| `RampBase` / Smoothers | ✅ アトミック＋数値演算のみ | 直接読取 |
| `MKLNonUniformConvolver::Add()` | ✅ メモリ確保・ロックなし | 直接読取 |
| `MKLNonUniformConvolver::Get()` | ✅ メモリ確保・ロックなし | 直接読取 |
| `EQProcessor::process()` | ✅ RCU読取＋pre-alloc buffers | 直接読取 |
| `ConvolverProcessor::process()` | ✅ RCU読取＋pre-alloc buffers | 直接読取 |
| `OutputFilter::process` | ✅ 係数準備済み | 直接読取 |
| `LoudnessMeter::processBlock` | ✅ AVX2内積のみ | 直接読取 |
| `TruePeakDetector::processBlock` | ✅ 事前割当バッファのみ | 直接読取 |
| `PsychoacousticDither::process` | ✅ 事前計算テーブルのみ | 直接読取 |
| `FixedNoiseShaper` | ✅ 事前割当バッファのみ | 直接読取 |
| `LatticeNoiseShaper` | ✅ 事前割当バッファのみ | 直接読取 |
| `UltraHighRateDCBlocker` | ✅ 一次IIRフィルタのみ | 直接読取 |
| `CustomInputOversampler::processUp/Down` | ✅ 事前割当ワークバッファ＋SIMD | 直接読取 |
| `InputBitDepthTransform` (convertFloatToDoubleHighQuality) | ✅ AVX2/SIMDのみ、CPU内蔵 | 直接読取 |

### 5-2. 全ミューテックス使用箇所の確認

| ファイル | Mutex名 | スレッド | 問題 |
|---------|---------|---------|------|
| `ISRLifecycle.cpp` | `nonRtGuard_` | ✅ Message/Worker | 非RTスレッド保護 |
| `PrepareToPlay.cpp:79` | `rebuildMutex` | ✅ Message Thread | `ASSERT_NON_RT_THREAD()` 確認 |
| `ReleaseResources.cpp:115` | `rebuildMutex` | ✅ Message Thread | `ASSERT_NON_RT_THREAD()` 確認 |
| `MKLNonUniformConvolver.cpp:89` | `IppFFTPlanCache::getMutex()` | ✅ prepare()のみ | コメントで確認（非RTスレッド） |
| `AudioEngine.Commit.cpp` | — | ✅ Commit Thread | `ASSERT_NON_RT_THREAD()` 確認 |
| `AudioEngine.Parameters.cpp` | — | ✅ Message Thread | `ASSERT_NON_RT_THREAD()` 確認 |
| `AudioEngine.Threading.cpp` | — | ✅ Message Thread | `ASSERT_NON_RT_THREAD()` 確認 |
| `ISRRetireRuntimeEx.cpp:185` | — | ✅ Retire Thread | `ASSERT_NON_RT_THREAD()` 確認 |

### 5-3. 全 `diagLog()` 呼出元の確認

| ファイル | 呼出有無 | スレッド | 安全 |
|---------|---------|---------|------|
| `AudioEngine.Commit.cpp` | ✅ 複数 | Commit | ✅ `ASSERT_NON_RT_THREAD()` |
| `AudioEngine.CtorDtor.cpp` | ✅ 複数 | Message | ✅ コンストラクタ/デストラクタ |
| `AudioEngine.Init.cpp` | ✅ 複数 | Message | ✅ 初期化 |
| `AudioEngine.Parameters.cpp` | ✅ 複数 | Message | ✅ `ASSERT_NON_RT_THREAD()` |
| `AudioEngine.Processing.DSPCoreLifecycle.cpp` | ✅ 多数 | Message | ✅ prepareToPlay |
| `AudioEngine.Processing.PrepareToPlay.cpp` | ✅ 複数 | Message | ✅ prepareToPlay |
| `AudioEngine.Processing.ReleaseResources.cpp` | ✅ 複数 | Message | ✅ releaseResources |
| `AudioEngine.Processing.DSPCoreFloat.cpp` | ❌ **定義のみ** | — | ⚠️ **ガード欠如** |
| `AudioEngine.Processing.DSPCoreDouble.cpp` | ✅ 定義 | — | ✅ ガードあり |

---

## 6. 📊 全発見事項の重要度マトリクス

### P0（= 即時修正）

| # | 発見 | 影響 | 難易度 | ツール |
|---|------|------|--------|-------|
| 1 | DSPCoreFloat.cpp diagLog() ガード欠如 | 音飛びリスク | 低（1行）| rtk(grep), 直接読取 |

### P1（= 優先対応）

| # | 発見 | 影響 | 難易度 | ツール |
|---|------|------|--------|-------|
| 2 | RTLocalState false sharing | キャッシュバウンス µs級 | 中（alignas追加）| PowerShell, 直接読取 |
| 3 | AudioEngine atomic false sharing | 同上 | 中 | PowerShell, 直接読取 |
| 4 | getOrCreate lock+alloc | 非RTだが警戒 | 低（CIアサーション） | rtk(grep), 直接読取 |
| 5 | MKLNonUniformConvolver Logger | prepareパスのLogger | 低 | rtk(grep), 直接読取 |
| 6 | hash<thread::id> + isAudioThread | Debugでのオーバーヘッド | 低 | 直接読取 |
| 7 | 無条件 getCurrentTimeUs() 5回 | 小 | 中 | 検証レポート参照 |

### P2（= 計画的対応）

| # | 発見 | 影響 | 難易度 | ツール |
|---|------|------|--------|-------|
| 8 | CallbackRuntimeScope atomic連鎖 | 200-400ns/コールバック | 中 | 直接読取 |
| 9 | ScopedNoDenormals 重複構築 | 小 | 中（thread_local最適化）| 直接読取 |
| 10 | GetCurrentProcessorNumber syscall | 診断時数百ns | 低 | 直接読取 |
| 11 | markRTContext グローバルatomic | キャッシュ汚染 | 中 | 直接読取 |
| 12 | AudioBlock stack構築（Debug） | 小 | 低 | 直接読取 |
| 13 | musicalSoftClip 実装調査 | 未確定 | 調査必要 | 直接読取 |

### 既存文書から引き継ぎ（改善計画済み）

| # | 発見 | 改修計画でのタスク |
|---|------|------------------|
| 14 | MMCSS syscall初回 | フェーズ2検討 |
| 15 | kTraceBufferSize branch回収 | フェーズ2タスク9 |
| 16 | Diagnostics sampling改善 | フェーズ2タスク9 |
| 17 | Firewall release時省略 | フェーズ2タスク12 |
| 18 | CallbackTelemetryScope条件化 | フェーズ1タスク6 |
| 19 | XRUN検出サンプリング化 | フェーズ2タスク10 |
| 20 | thread_local キャッシュ | フェーズ1タスク2-5 |

---

## 7. 🔧 使用ツールの有効性評価

| ツール | 使い方 | 有効性 | 今回の発見数 | 特記事項 |
|-------|--------|--------|------------|---------|
| **rtk (WSL版)** | `wsl bash -c '... ~/.local/bin/rtk grep ...'` | ✅ 最重要 | 15+ | 出力圧縮でトークン節約。全調査の基盤 |
| **PowerShell Select-String** | `Select-String -Path $f -Pattern "..."` | ✅ 必須 | 10+ | ファイルリスト+パターンでWSLより安定 |
| **cocoindex-code (ccc.exe)** | `ccc grep "pattern"` / `ccc search "query"` | ✅ 構造型grep | 5+ | 言語認識パターンマッチ。`--context`非対応 |
| **semble (semble.exe)** | `semble search "query" --max-snippet-lines N` | ✅ 意味検索 | 4 | 自然言語クエリで予期せぬ関連を発見 |
| **AiDex MCP** | `aidex_tree`, `aidex_signature`, `aidex_query` | ✅ 構造把握 | 8+ | 全288ファイルの一覧＋シグネチャ把握に最適 |
| **graphify (graphify.exe)** | `graphify path "A" "B"` | ⚠️ 準備不足 | 0 | `graph.json` 未生成で今回は使用不可 |
| **Serena MCP** | `serena_*` tools | ❌ アクセス不可 | 0 | マニュアル読み取りに失敗 |
| **context-mode MCP** | `ctx_execute(javascript, code)` | ✅ データ解析 | 3 | 大規模スキャンに有効。ファイルパス注意 |
| **WSL grep (直接)** | `grep -n "pat" file` | ✅ 基本 | 5+ | 単純検索には安定。 | は注意 |

---

## 8. 🎯 総合結論

### 8-1. コードベースの評価

ConvoPeq の音声スレッドRT設計は**非常に堅牢**であり、基本的な設計思想（ロックフリー、事前割当、RCU、atomic publish/consume）は完全に守られている。以下の理由から、**商用クラスのリアルタイムオーディオプラグインとして十分な品質**であると言える:

- 全DSPカーネルにロックなし
- 全アロケーションは prepareToPlay() で事前完了
- RT-safeでない操作は `ASSERT_NON_RT_THREAD()` で明示的に保護
- 音声スレッド境界は `ASSERT_AUDIO_THREAD()` + `ScopedThreadRole` + `FirewallToken` の3重防御

### 8-2. 発見された問題の性質

発見された問題のほとんどは「すでに設計上は安全だが、細部の最適化や将来のリグレッション防止」に関するものである。唯一の例外が **DSPCoreFloat.cppのdiagLog()ガード欠如（P0）** であり、これは修正必須。

### 8-3. 改修優先順位（更新版）

```
即時（P0）:  DSPCoreFloat.cpp diagLog() ガード    ← 本日中
────────────────────────────────────────────
優先（P1）:  RTLocalState alignas(64)             ← 1-2日
             AudioEngine.h false sharing 対策      ← 1日
             thread_local キャッシュ導入           ← 0.5日（改善計画済み）
             CallbackTelemetryScope 条件化         ← 1日（改善計画済み）
             MKL logger 統一                       ← 0.5日
             XRUN サンプリング                     ← 1日（改善計画済み）
────────────────────────────────────────────
計画的（P2）: その他8項目                           ← 3-5日
```
