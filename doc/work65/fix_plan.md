# work65: XRUN原因特定レビュー指摘対応 改修計画書 v2

**作成日**: 2026-07-05
**最終更新日**: 2026-07-05 (v2: レビューフィードバック反映)
**対象レビュー**: ConvoPeq × Voicemeeter Virtual ASIO XRUN多発 解析結果
**対応方針**: RTスレッド上のブロッキング要因（同期ファイルI/O・同期Logger書込み）をロックフリーリングバッファ経由または削除で除去する

---

## 全体サマリー

| # | 改修項目 | 根拠 | 影響ファイル数 | 優先度 |
|---|---------|------|--------------|--------|
| 1 | work52調査用キャプチャコード削除 | 無条件同期ファイルI/O（未ガード） | 1 | **CRITICAL** |
| 2 | eqDiagBuffer 双定義バグ修正 | DSPCoreDouble.cppのeqDiagBufferがnullptrでEQ_TIME消失 | 2 | **CRITICAL** |
| 3 | ANS_SWITCH → LockFreeRingBuffer (double) | RTスレッドで同期Logger書込み | 2 | HIGH |
| 4 | ANS_SWITCH → LockFreeRingBuffer (float) | floatパスも将来のために統一 | 1 | LOW |
| 5 | [[maybe_unused]] diagLog維持 | 実害なし、将来再利用可能性 | 0 | — |

---

## 事前確認: 呼び出し経路

```
Double path (active, setDoublePrecisionProcessing=true):
  processBlock(double&) → processBlockDouble() → DSPCore::processDouble()
    ├→ processInputDouble()                       [DSPCoreIO.cpp]  ← RT-safe
    ├→ convolver処理 + diagCapture削除対象          [DSPCoreDouble.cpp]  ← Fix 1
    ├→ EQ処理 + logEqTime → eqDiagBuffer(Double)   [DSPCoreDouble.cpp]  ← ★ NULL BUG
    └→ processOutputDouble()                      [DSPCoreDouble.cpp]
         └→ ANS_SWITCH writeToLog → ★ Fix 3

Float path (inactive, would need setDoublePrecisionProcessing=false):
  processBlock(float&) → getNextAudioBlock() → DSPCoreFloat::process()
    ├→ processInput()                             [DSPCoreIO.cpp]
    ├→ convolver/EQ処理
    └→ processOutput()                            [DSPCoreIO.cpp]
         └→ ANS_SWITCH writeToLog → ★ Fix 4
```

---

## RTスレッド ブロッキング要因 完全監査

全オーディオ処理ファイル（BlockDouble.cpp, DSPCoreDouble.cpp, DSPCoreIO.cpp, DSPCoreFloat.cpp, AudioBlock.cpp）を対象に、RTスレッド上で実行されうるブロッキング操作を網羅調査した。

### 凡例

| マーク | 意味 |
|--------|------|
| ✅ 問題なし | RTセーフな操作（atomic, lock-free, pre-allocated） |
| ⚠️ 条件付き安全 | 実装上問題ないが注意を要する |
| ❌ **修正対象** | 本work65で修正するRT違反 |
| 🟡 デッドコード | 現設定では実行されないパス |

### 監査結果

#### カテゴリ1: writeToLog / Logger / DBG 直接呼び出し

| ファイル | 行 | 内容 | ガード | 判定 |
|---------|----|------|--------|------|
| `DSPCoreDouble.cpp` | L43-44 | `diagLog()` 関数内の `DBG` + `writeToLog` | `CONVOPEQ_ENABLE_RUNTIME_DIAG` | 🟡 現在は誰も呼ばないデッドコード |
| `DSPCoreDouble.cpp` | L808-809 | ANS_SWITCH: `DBG` + `writeToLog` | `CONVOPEQ_ENABLE_RUNTIME_DIAG` | ❌ **Fix 2: LockFreeRingBuffer経由に変更** |
| `DSPCoreFloat.cpp` | L11-12 | `diagLog()` 関数内の `DBG` + `writeToLog` | `CONVOPEQ_ENABLE_RUNTIME_DIAG` | 🟡 floatパス（デッドコード） |
| `DSPCoreIO.cpp` | L438-439 | ANS_SWITCH: `DBG` + `writeToLog` | `CONVOPEQ_ENABLE_RUNTIME_DIAG` | 🟡 floatパス（デッドコード）→ ❌ **Fix 3: 削除** |

#### カテゴリ2: mutex / lock / 同期プリミティブ

| ファイル | 結果 |
|---------|------|
| BlockDouble.cpp | ✅ `std::mutex` / `lock_guard` / `unique_lock` なし |
| DSPCoreDouble.cpp | ✅ すべて `convo::consumeAtomic` / `publishAtomic` / `fetchAddAtomic` のみ |
| DSPCoreIO.cpp | ✅ 同 |
| DSPCoreFloat.cpp | ✅ 同 |
| AudioBlock.cpp | ✅ 同 |

結論: **RTスレッド上の同期プリミティブはゼロ。**

#### カテゴリ3: ファイルI/O

| ファイル | 行 | 内容 | ガード | 判定 |
|---------|----|------|--------|------|
| `DSPCoreDouble.cpp` | L101 | `fopen_s("C:\\TEMP\\conv_output_l.raw", "wb")` | **なし** | ❌ **Fix 1: 削除** |
| `DSPCoreDouble.cpp` | L132 | `fwrite(&hdr, 64, 1, ...)` | **なし** | ❌ **Fix 1: 削除** |
| `DSPCoreDouble.cpp` | L140 | `fwrite(dataL, sizeof(double), n, ...)` | **なし** | ❌ **Fix 1: 削除** |
| `DSPCoreDouble.cpp` | L143 | `fclose(g_diagCaptureFile)` | **なし** | ❌ **Fix 1: 削除** |
| `DSPCoreDouble.cpp` | L129 | `GetSystemTimePreciseAsFileTime` | **なし** | ❌ **Fix 1: 削除** |
| `DSPCoreDouble.cpp` | L114,119 | `std::getenv` (×2) | **なし** | ❌ **Fix 1: 削除** |
| その他全ファイル | — | `FILE*` / `fopen` / `fwrite` / `fclose` / `fprintf` | — | ✅ なし |

#### カテゴリ4: 動的メモリ確保（new / malloc / free / allocator）

| ファイル | 行 | 内容 | 判定 |
|---------|----|------|------|
| 全RTファイル | — | `new` / `malloc` / `calloc` / `realloc` / `free` / `delete` | ✅ なし |
| JUCE dsp内部 | — | `oversampling.processUp()` / `processDown()` | ⚠️ prepareToPlayで事前確保されていれば安全 |
| EQProcessor.Processing.cpp | — | ヒープ確保なし | ✅ |
| ConvolverProcessor.Runtime.cpp | — | ヒープ確保なし | ✅ |

**特記事項**: `pushAdaptiveCaptureBlocks()` は `LockFreeRingBuffer` 経由（RT-safe）。`pushToFifo()` は `LockFreeAudioRingBuffer` 経由（RT-safe）。

#### カテゴリ5: 仮想関数 / 動的ディスパッチ

| 呼び出し元 | 対象 | 判定 |
|-----------|------|------|
| `BlockDouble.cpp` L360,403 | `dsp->processDouble()` | ✅ DSPCore::processDouble（仮想だが実体確定、vtable lookupのみ） |
| `BlockDouble.cpp` L309,357 | `fading->processDouble*()` | ✅ 同 |
| `DSPCoreDouble.cpp` L523,585 | `convolverRt().process()` | ✅ 事前準備済み、仮想だが実体確定 |
| `DSPCoreDouble.cpp` L537-573 | `eqRt().process()` | ✅ 同 |
| `DSPCoreDouble.cpp` L494,634,642,670 | `oversampling.processUp/Down` | ✅ JUCE dsp（prepareToPlayで事前準備） |
| `DSPCoreDouble.cpp` L601 | `outputFilter.process()` | ✅ JUCE dsp |
| 各種 `.processStereoBlock()` / `.processBlock()` | dither/noiseShaper/loudness/peak | ✅ すべてテンプレート/インライン、仮想なし |

結論: **RTスレッド上の仮想関数呼び出しはすべて事前準備済みのオブジェクトに対する確定済みディスパッチ。動的メモリ確保やブロッキングを伴わない。**

#### カテゴリ6: その他OS呼び出し / 同期操作

| ファイル | 内容 | 判定 |
|---------|------|------|
| 全RTファイル | `Sleep` / `WaitFor` / `CreateThread` / `CreateMutex` / `SetEvent` / `MessageBox` | ✅ なし |
| DSPCoreDouble.cpp | `ScopedNoDenormals` / `ScopedThreadRole` / `ASSERT_AUDIO_THREAD` | ✅ スタックローカル、副作用なし |

### 総括表

| # | カテゴリ | 修正対象 | デッドコード | 条件付き | 問題なし |
|---|---------|---------|------------|---------|---------|
| 1 | writeToLog/Logger/DBG | **2件** (Fix 2, 3) | 2件 | 0 | 0 |
| 2 | mutex/lock | 0 | 0 | 0 | **全ファイル** |
| 3 | ファイルI/O | **6件** (Fix 1) | 0 | 0 | 0 |
| 4 | 動的メモリ確保 | 0 | 0 | 2件 | 全RTファイル |
| 5 | 仮想関数 | 0 | 0 | 0 | **全RTファイル** |
| 6 | OS呼び出し | 0 | 0 | 0 | **全RTファイル** |

**修正対象は合計8件、すべて本work65で対応する。** 上記は現時点のソースコード静的解析で確認できたRT違反の一覧である。JUCE内部・MKL内部・Convolver Runtime内部など外部ライブラリの動作は静的解析の対象外であり、それら内部のブロッキング可能性については本監査では評価していない。

---

## Fix 1: work52キャプチャコード削除（原因A対応）

### 削除対象コード

`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` から以下を削除:

| 削除対象 | 行範囲 | 内容 |
|---------|--------|------|
| `#include <cstdio>` | L7 | `fopen_s` / `fwrite` / `fclose` 用 |
| `#include <cstdint>` | L8 | `uint32_t`等（他で使われていなければ） |
| `struct CaptureHeader` | L11-L28 | 64バイトキャプチャヘッダ |
| `static_assert(sizeof(CaptureHeader)==64)` | L29 | ヘッダサイズ検証 |
| `static FILE* g_diagCaptureFile` | L32 | ファイルポインタ |
| `static int g_diagCaptureRemaining` | L33 | 残りキャプチャサンプル数 |
| `static constexpr int kDiagCaptureMaxSamples` | L34 | 最大キャプチャ長（10秒） |
| `void diagEnableToneInjection()` | L36 | 前方宣言（機能削除済み） |
| `void diagStartCapture()` | L99-L131 | ファイルオープン＋ヘッダ書込み＋fwrite |
| `void diagWriteCapture(...)` | L137-L144 | サンプル書込み＋fclose |
| `diagStartCapture()` 呼び出し | L525-L528 | Conv→EQ 分岐内 |
| `diagStartCapture()` 呼び出し | L588-L589 | EQ→Conv 分岐内 |

### 削除後の影響

- `C:\TEMP\conv_output_l.raw` へのファイル出力が完全に停止
- RTスレッド上の `fopen_s` / `fwrite` / `fclose` / `GetSystemTimePreciseAsFileTime` / `std::getenv` がすべて除去される
- 環境変数 `CONVOPEQ_TAIL_BYPASS`, `CONVOPEQ_DIRECT_HEAD`, `CONVOPEQ_CAPTURE_INPUT` → 関連コードも削除される
- 使用していたファイルがなくなるため、リンク時に avrt.lib 以外の追加ライブラリは不要（確認済み）

### リスク

- **ゼロ**: このコードは work52 調査用の一時的コードであり、本機能に一切寄与しない。

---

## Fix 2: `eqDiagBuffer` 双定義バグ修正 🐛

### 現状の問題

`eqDiagBuffer` が**2つの異なる翻訳単位（Anonymous namespace in TU）で別々のstatic変数として宣言**されている:

| TU | 変数 | 初期化 |
|----|------|--------|
| `DSPCoreFloat.cpp` | `eqDiagBuffer` (anonymous ns) | ✅ `setEqDiagBuffer()` で `&diagBuffer` に設定 |
| `DSPCoreDouble.cpp` | `eqDiagBuffer` (anonymous ns) | ❌ **未初期化（nullptr）** |

`setEqDiagBuffer()` の実体は `DSPCoreFloat.cpp` にある。この関数内で代入する `eqDiagBuffer` は**関数と同じ翻訳単位の匿名名前空間変数**のみを指す。よって `DSPCoreDouble.cpp` の `eqDiagBuffer` は常に `nullptr` のまま。

**結果**: `DSPCoreDouble.cpp::logEqTime()` 内の `if (eqDiagBuffer == nullptr) return;` が常に成立し、EQ_TIME診断イベントが**唯一のアクティブパス（double）から常に消失**している。無害（クラッシュしない）だが診断が完全に機能していない。

### 修正方針

`eqDiagBuffer` を `extern` グローバル変数に変更し、全TUで同じ実体を参照する:

1. `AudioEngine.h`（または `DiagnosticsConfig.h`）に `extern` 宣言を追加
2. `DSPCoreFloat.cpp` で実体を定義
3. `DSPCoreDouble.cpp` + 将来追加する `DSPCoreIO.cpp` の重複宣言を削除
4. `setEqDiagBuffer` がただ1つのグローバル変数を設定するよう変更

```cpp
// AudioEngine.h に追加:
extern LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>* g_eqDiagBuffer;
extern DiagPerTickCounter* g_eqTickPushed;
extern DiagPerTickCounter* g_eqTickDropped;
extern std::atomic<uint64_t>* g_eqTotalPushed;
```

**注意**: `DSPCoreDouble.cpp` の `logEqTime()` 内で `eqDiagBuffer` → `g_eqDiagBuffer` に変数名を変更する必要がある。

---

## Fix 3: ANS_SWITCH → LockFreeRingBuffer (doubleパス)

### 現状

`DSPCoreDouble.cpp::processOutputDouble()` 内 L795-L809:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansStartUs = convo::getCurrentTimeUs();
#endif
        adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if (ansElapsedUs > 10)
        {
            juce::String alog("[ANS_SWITCH] us=");
            alog += juce::String(static_cast<int64_t>(ansElapsedUs));
            DBG(alog); // NOLINT(rt-logger)
            juce::Logger::writeToLog(alog); // NOLINT(rt-logger)   ← ★同期I/O
        }
#endif
```

発火条件: `ansElapsedUs > 10`（バンク切替が10μs超えた場合）

### 3-a: AudioEngine.h — DiagCategory拡張

```cpp
// ★ work65: ANS_SWITCH bank switch timing
struct AnsSwitchData {
    uint64_t elapsedUs;
};

enum class DiagCategory : uint8_t {
    ...
    CallbackArrival = 8,
    AnsSwitchTime = 9,
    Count                     // 10
};

// DiagEvent union に追加: 既存最大サイズ88を超えない
```

### 3-b: DSPCoreDouble.cpp — 置き換え後

変更ポイント:
1. ✅ **`eventIndex` に `this->currentCallbackSeq` を使用**（DSPCoreメンバ。`processOutputDouble()` からアクセス可能）
2. ✅ **閾値`ansElapsedUs > 10` 撤廃** → `CONVOPEQ_DIAG_SAMPLE_MASK` でサンプリング
3. ✅ **出力先**: Fix 2 で修正する `g_eqDiagBuffer`（== `AudioEngine::diagBuffer`）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansStartUs = convo::getCurrentTimeUs();
#endif
        adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
        adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
        activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
        activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if ((currentCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0 && g_eqDiagBuffer != nullptr)
        {
            DiagEvent event{};
            event.category = DiagCategory::AnsSwitchTime;
            event.eventIndex = currentCallbackSeq;
            event.data.ansSwitchTime.elapsedUs = ansElapsedUs;
            g_eqDiagBuffer->push(event);  // RT-safe: drop on full
        }
#endif
```

### 3-c: AudioEngine.Timer.cpp — formatDiagEvent追加

```cpp
case DiagCategory::AnsSwitchTime:
    return diagPrefix(gen) + " [ANS_SWITCH] cbIdx="
        + juce::String(static_cast<juce::int64>(event.eventIndex))
        + " us=" + juce::String(static_cast<juce::int64>(event.data.ansSwitchTime.elapsedUs));

static_assert(static_cast<int>(DiagCategory::Count) == 10,
    "DiagCategory enum changed: update formatDiagEvent() accordingly");
```

### 消費側: 変更不要

既存の `timerCallback()` → `diagBuffer.pop()` → `formatDiagEvent()` → `diagLog()` → `asyncSink()` パイプラインがそのままANS_SWITCHを消費。`MaxDrainPerTick=16`で十分。

---

## Fix 4: ANS_SWITCH → LockFreeRingBuffer (floatパス)

### 変更前

`DSPCoreIO.cpp::processOutput()` L435-L439:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if (ansElapsedUs > 10)
        {
            juce::String alog("[ANS_SWITCH] us=");
            alog += juce::String(static_cast<int64_t>(ansElapsedUs));
            DBG(alog); // NOLINT(rt-logger)
            juce::Logger::writeToLog(alog); // NOLINT(rt-logger)
        }
#endif
```

### 変更方針

「現在デッドコードだから削除」ではなく、**doubleパスと同じRingBuffer経由に統一**する。理由:
- 将来 `setDoublePrecisionProcessing(false)` に戻した場合に備える
- コードベースの一貫性（両パスで診断方式が統一されている）
- `processOutput()` は `DSPCore` メンバ関数であり `this->currentCallbackSeq` にアクセス可能

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if ((currentCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0 && g_eqDiagBuffer != nullptr)
        {
            DiagEvent event{};
            event.category = DiagCategory::AnsSwitchTime;
            event.eventIndex = currentCallbackSeq;
            event.data.ansSwitchTime.elapsedUs = ansElapsedUs;
            g_eqDiagBuffer->push(event); // RT-safe: drop on full
        }
#endif
```

---

## Fix 5: `[[maybe_unused]] diagLog` 関数 — 維持

`DSPCoreDouble.cpp` L41 と `DSPCoreFloat.cpp` L9 の `[[maybe_unused]] diagLog()` は削除しない。
- 実害なし（`[[maybe_unused]]` で警告抑制済み）
- 将来の診断用途で再利用される可能性あり
- 本work65の修正後は誰も呼ばない無害なデッドコードとして残る

```cpp
// formatDiagEvent() switch に追加:
case DiagCategory::AnsSwitchTime:
    return diagPrefix(gen) + " [ANS_SWITCH] cbIdx=" + juce::String(static_cast<juce::int64>(event.eventIndex))
        + " us=" + juce::String(static_cast<juce::int64>(event.data.ansSwitchTime.elapsedUs));

// static_assert 更新:
static_assert(static_cast<int>(DiagCategory::Count) == 10,
    "DiagCategory enum changed: update formatDiagEvent() switch accordingly");
```

### 消費側（既存）: 変更不要

Timerスレッドの `timerCallback()` → `diagBuffer.pop()` + `formatDiagEvent()` + `diagLog()` → `asyncSink()` の既存パイプラインがそのまま ANS_SWITCH イベントを消費する。`MaxDrainPerTick`（16件/tick）の制限内で動作。書き込み頻度はバンク切替時のみ（通常時は0）であり、バッチ書き込みへの変更は不要。

---

---

## 変更ファイル一覧

| ファイル | 変更内容 | 行数変化 |
|---------|---------|---------|
| `src/audioengine/AudioEngine.h` | DiagCategory::AnsSwitchTime / AnsSwitchData / extern g_eqDiagBuffer / static_assert | +15行 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | キャプチャコード削除 / eqDiagBuffer→g_eqDiagBuffer / ANS_SWITCH DiagEvent化 | -80行 / +15行 |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | eqDiagBuffer→g_eqDiagBuffer / setEqDiagBuffer修正 | 0行 |
| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | ANS_SWITCH writeToLog→DiagEvent push / g_eqDiagBuffer参照 | -5行 / +12行 |
| `src/audioengine/AudioEngine.Timer.cpp` | formatDiagEvent AnsSwitchTime / static_assert更新 | +4行 |

**正味削減行数**: 約40行削減

---

## ビルド確認項目

- [ ] `cmake --build build --config Debug` が通ること
- [ ] `cmake --build build --config Release` が通ること
- [ ] `check-audioengine-lint.ps1` がパスすること
- [ ] `check-src-atomic-dotcall.ps1` がパスすること
- [ ] work21 EpochDomain CI Gate がパスすること
- [ ] CLI Smoke Test が通ること

---

---

## 不確定事項の確定報告

本work65の調査過程で、計画書v2時点での未確定事項5項目を徹底調査した。以下、各項目の確定結果を報告する。

---

### 【確定】1. eqDiagBuffer双定義バグ 🐛 — Fix 2で修正

**調査方法**: DSPCoreFloat.cpp (TU1) と DSPCoreDouble.cpp (TU2) の両方で `eqDiagBuffer` 変数宣言を確認。

**確定内容**: 完全に確認。`setEqDiagBuffer()` の実体は `DSPCoreFloat.cpp` にあり、関数内で代入する `eqDiagBuffer` は同一翻訳単位の匿名名前空間変数のみを指す（C++標準のODR則による）。`DSPCoreDouble.cpp` の同名変数は完全に独立した別実体であり、常に `nullptr`。この結果、`logEqTime()` 内の `if (eqDiagBuffer == nullptr) return;` が常に成立し、EQ_TIME診断イベントがdoubleパスから永久に消失している。

**影響**: 診断機能の一部が完全に機能していないが、アプリケーションの動作には無害（クラッシュなし）。**修正はFix 2でexternグローバル変数化により行う**。

---

### 【確定】2. Convolver Runtime RT安全性 ✅

**調査方法**: `ConvolverProcessor.Runtime.cpp` の全行を走査し、mutex/lock/fopen/fwrite/malloc/new/Sleep/WaitFor等のブロッキング操作を網羅検索。加えて `ConvolverProcessor.Lifecycle.cpp` との責務分離を確認。

**確定内容**: ✅ **RTスレッド上にブロッキング操作は一切存在しない。**
- `process()` 関数: `RCUReaderGuard`（非ブロッキング）、事前確保済みバッファ、`convo::consumeAtomic` のみ
- MKL FFT呼出し（`DftiComputeForward/Backward`）: すべて `ConvolverProcessor.MixedPhase.cpp` と `ResampleAndFallback.cpp` にあり、これらはIRプリプロセス（非RT）のみ。RTパスでは使用しない
- `mkl_malloc/free`: `Lifecycle.cpp` のみ（prepareToPlay/releaseResources＝非RT）
- `juce::dsp::Convolution`（JUCE内蔵）: 使用していない。ConvoPeqは独自の `ConvolverProcessor` を使用

**結論**: ツール（rg/grep）による静的解析の範囲で **ブロッキング要因ゼロを確認**。MKL内部のFFT演算はCPU boundであり、システムコールを伴わない。

---

### 【確定】3. JUCE dsp Oversampling RT安全性 ⚠️（注意事項あり）

**調査方法**: `JUCE/modules/juce_dsp/processors/juce_Oversampling.h/.cpp` の `processSamplesUp/Down` および `initProcessing` を確認。

**確定内容**:
- `initProcessing()`: ヒープ確保あり（`setSize`, `resize`）→ **prepareToPlayで事前呼出し必須**
- `processSamplesUp()` / `processSamplesDown()`: いずれも `noexcept`、事前確保済みバッファのみを使用、**ランタイムのヒープ確保なし**
- JUCE dsp全体でランタイムに `new`/`malloc` を行うのはフィルタ設計（`FilterDesign.cpp`）のみで、これはprepareToPlay経路
- `juce::dsp::Convolution` に `SpinLock` があるが、ConvoPeqは使用していない（無関係）

**結論**: ✅ `prepareToPlay` で `initProcessing()` が正しく呼ばれていれば、RTスレッドでの `processUp/Down` は安全。この前提は `DSPCoreDouble.cpp` の buildAudioThreadProcessingState → DSPCore::prepare 経路で満たされている。

---

### 【確定】4. DiagnosticsConfig.h と CMake の不一致（保留 → work66）

**調査方法**: CMakeLists.txt L55とL578、DiagnosticsConfig.h L19-21 の両方を確認し、動作をトレース。

**確定内容**: 以下が完全に確認された。
- CMake: `option(CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS OFF)` — デフォルト **OFF**
- CMake generator expression: `$<$<BOOL:${CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS}>:CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1>` — **ONの時のみ** `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` をコンパイラに渡す
- OFFの場合 → コンパイラには未定義 → `DiagnosticsConfig.h` の `#ifndef` が発火 → `#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 1` → **常に1**

**結果**: CMakeでOFFに設定しても、Release/Debug問わず常に診断コードがコンパイル・実行される。コメントの「デフォルト0」とも矛盾。

**対応**: 本work65では修正しない。work66以降で以下のいずれかを実施:
- 案A: `DiagnosticsConfig.h` の `#define` を `0` に変更（最もシンプル）
- 案B: CMakeの generator expression を `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=$<BOOL:${CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS}>` に変更し、常に定義値を渡す

---

### 【確定】5. pushAdaptiveCaptureBlocks / pushToFifo RT安全性 ✅

**調査方法**: DSPCoreIO.cpp L118-L169 の `pushAdaptiveCaptureBlocks` 実装と `LockFreeAudioRingBuffer` を確認。

**確定内容**:
- `pushAdaptiveCaptureBlocks`: `LockFreeRingBuffer<AudioBlock, 4096>::pushWithWriter()` 経由。SPSC lock-freeリングバッファ。満杯時は静かにドロップ。**完全にRT-safe** ✅
- `pushToFifo`: `LockFreeAudioRingBuffer::push()` 経由。同様にlock-free。**完全にRT-safe** ✅

---

### 【確定】6. NOLINT(rt-logger) 全4箇所の内訳

**調査方法**: `src/` 全体から `NOLINT(rt-` を網羅検索し、各出現箇所をコンテキスト付きで確認。

**確定内容**:

| # | ファイル | 行 | 内容 | 状態 |
|---|---------|----|------|------|
| 1 | `DSPCoreDouble.cpp` | L43-44 | `diagLog()` 内の `DBG`+`writeToLog` | 🟡 誰も呼ばないデッドコード → 維持 |
| 2 | **`DSPCoreDouble.cpp`** | **L808-809** | **ANS_SWITCHの `DBG`+`writeToLog`** | **❌ Fix 3: RingBuffer化** |
| 3 | `DSPCoreFloat.cpp` | L11-12 | `diagLog()` 内の `DBG`+`writeToLog` | 🟡 floatパス（デッドコード）→ 維持 |
| 4 | **`DSPCoreIO.cpp`** | **L438-439** | **ANS_SWITCHの `DBG`+`writeToLog`** | **❌ Fix 4: RingBuffer化** |

---

### 【未確定のまま残る事項】XRUNとANS_SWITCHの時系列相関

本work65の全修正後にETW再測定を行うことで、「ANS切替が発生したコールバック番号」と「XRUNが発生したコールバック番号」の対応検証が可能になる。本work65で `eventIndex = currentCallbackSeq` を導入するのはこのためである。
