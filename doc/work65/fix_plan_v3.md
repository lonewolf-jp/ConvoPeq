# work65: XRUN原因特定レビュー指摘対応 改修計画書 v3

**作成日**: 2026-07-05
**最終更新日**: 2026-07-05 (v3: Fix2設計変更 / ANS_SWITCH統計追加 / DiagnosticsConfig前倒し)
**対象レビュー**: ConvoPeq × Voicemeeter Virtual ASIO XRUN多発 解析結果
**対応方針**:
- RTスレッド上のブロッキング要因（同期ファイルI/O・同期Logger書込み）をロックフリーリングバッファ経由に変更
- 診断フラグのビルド設定をCMakeと一致させ、Releaseビルドで診断コードが確実に無効化されるようにする

---

## 全体サマリー

| # | 改修項目 | 根拠 | 影響ファイル数 | 優先度 |
|---|---------|------|--------------|--------|
| 1 | work52調査用キャプチャコード削除 | 無条件同期ファイルI/O（未ガード） | 1 | **CRITICAL** |
| 2 | `eqDiagBuffer` 双定義 → 単一TU集約 | DSPCoreDouble.cppのeqDiagBufferがnullptrでEQ_TIME消失 | **新規: +1** | **CRITICAL** |
| 3 | ANS_SWITCH → LockFreeRingBuffer (double) | RTスレッドで同期Logger書込み | 2 | HIGH |
| 4 | ANS_SWITCH → LockFreeRingBuffer (float) | floatパスも将来のために統一 | 1 | LOW |
| 5 | **DiagnosticsConfig.h デフォルト0化** | CMake OFFが無視されるバグ修正（再発防止） | 1 | **CRITICAL** |
| 6 | `[[maybe_unused]] diagLog` 維持 | 実害なし | 0 | — |

---

## RTスレッド ブロッキング要因 監査サマリー

事前監査結果の詳細は v1/v2 を参照。本v3では以下の修正を行う:

| # | カテゴリ | 修正対象 | 内訳 |
|---|---------|---------|------|
| 1 | writeToLog/Logger/DBG 直接呼出し | **2件** | ANS_SWITCH (double), ANS_SWITCH (float) |
| 2 | ファイルI/O | **6件** | fopen_s, fwrite×2, fclose, GetSystemTimePreciseAsFileTime, getenv×2 |
| 3 | 診断ビルド設定 | **1件** | DiagnosticsConfig.h デフォルト値 |
| 4 | eqDiagBuffer バグ | **1件** | 双定義によるnullptr問題 |

---

## Fix 1: work52キャプチャコード削除

`DSPCoreDouble.cpp` から以下を全削除:

| 削除対象 | 内容 |
|---------|------|
| `#include <cstdio>` / `#include <cstdint>` | fopen/fwrite/fclose用 |
| `struct CaptureHeader` + static_assert | 64バイトヘッダ |
| `g_diagCaptureFile` / `g_diagCaptureRemaining` / `kDiagCaptureMaxSamples` | キャプチャstatic変数 |
| `diagEnableToneInjection()` | テストトーン注入（機能削除済み） |
| `diagStartCapture()` L99-L131 | fopen_s + fwrite + GetSystemTimePreciseAsFileTime + getenv×2 |
| `diagWriteCapture()` L137-L144 | fwrite + fclose |
| L525-L528 / L588-L589 の呼び出し | Conv→EQ / EQ→Conv 両方 |

**リスク**: ゼロ。調査用一時コード。

---

## Fix 2: `eqDiagBuffer` — 単一翻訳単位への集約（extern不使用）

### 問題

`eqDiagBuffer` 関連の4変数が2つの翻訳単位で別々のstatic変数として宣言されている:

| TU | 初期化 | 状態 |
|----|--------|------|
| `DSPCoreFloat.cpp` | ✅ `setEqDiagBuffer()` が自TUの変数を設定 | 正常 |
| `DSPCoreDouble.cpp` | ❌ 未初期化（nullptr） | **EQ_TIME常に消失** |

`setEqDiagBuffer()` の実体は `DSPCoreFloat.cpp` にあり、その関数内で代入する変数は同一TUのもののみ。

### 修正方針（extern不使用）

**新規ファイル `src/audioengine/AudioEngine.Diagnostics.cpp` を作成し、変数＋関数を1TUに集約する。**

```
新設: AudioEngine.Diagnostics.cpp (anonymous namespace)
  ├── eqDiagBuffer / eqTickPushed / eqTickDropped / eqTotalPushed   ← 変数
  ├── setEqDiagBuffer(...)                                            ← setter（既存宣言を流用）
  └── logEqTime(...)                                                  ← 集約（重複削除）

削除: DSPCoreFloat.cpp から eqDiagBuffer変数 + logEqTime + setEqDiagBuffer定義
削除: DSPCoreDouble.cpp から eqDiagBuffer変数 + logEqTime + setEqDiagBuffer参照コメント
追加: CMakeLists.txt に AudioEngine.Diagnostics.cpp のエントリ
```

**既存の宣言（AudioEngine.h）はそのまま流用**:

```cpp
// AudioEngine.h L480（既存）:
void setEqDiagBuffer(
    LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>& db,
    DiagPerTickCounter& tickPushed, DiagPerTickCounter& tickDropped,
    std::atomic<uint64_t>& totalPushed) noexcept;
```

**新規追加: AudioEngine.h に logEqTime の宣言を追加**:

```cpp
// ★ [work65] 単一TUに集約（AudioEngine.Diagnostics.cpp）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
void logEqTime(uint64_t eqStartUs, int numSamples, int numChannels,
               const convo::EQParameters* eqParams,
               convo::ProcessingOrder order,
               double sampleRate,
               uint64_t callbackSeq, uint32_t cpu);
#endif
```

### 新ファイル: AudioEngine.Diagnostics.cpp

```cpp
#include <JuceHeader.h>
#include "AudioEngine.h"
#include "DiagnosticsConfig.h"

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
    LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>* eqDiagBuffer = nullptr;
    DiagPerTickCounter* eqTickPushed = nullptr;
    DiagPerTickCounter* eqTickDropped = nullptr;
    std::atomic<uint64_t>* eqTotalPushed = nullptr;
}
#endif

void setEqDiagBuffer(
    LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>& db,
    DiagPerTickCounter& tickPushed, DiagPerTickCounter& tickDropped,
    std::atomic<uint64_t>& totalPushed) noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    eqDiagBuffer = &db;
    eqTickPushed = &tickPushed;
    eqTickDropped = &tickDropped;
    eqTotalPushed = &totalPushed;
#else
    juce::ignoreUnused(db, tickPushed, tickDropped, totalPushed);
#endif
}

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
void logEqTime(uint64_t eqStartUs, int numSamples, int /*numChannels*/,
               const convo::EQParameters* eqParams,
               convo::ProcessingOrder order,
               double sampleRate,
               uint64_t callbackSeq, uint32_t cpu)
{
    const uint64_t eqElapsedUs = convo::getCurrentTimeUs() - eqStartUs;
    if (sampleRate <= 0.0 || numSamples <= 0 || eqDiagBuffer == nullptr) return;
    if ((callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) != 0) return;

    int activeBands = 0;
    if (eqParams != nullptr) {
        for (int i = 0; i < 20; ++i)
            if (eqParams->bands[i].enabled
                && std::abs(eqParams->bands[i].gain) > 0.01f)
                ++activeBands;
    }
    const double expectedUs = static_cast<double>(numSamples) / sampleRate * 1e6;
    const uint32_t budgetPermille = (expectedUs > 0.0)
        ? static_cast<uint32_t>((static_cast<double>(eqElapsedUs) / expectedUs) * 1000.0)
        : 0;

    DiagEvent event{};
    event.category = DiagCategory::EqTime;
    event.eventIndex = callbackSeq;
    event.data.eqTime.cpu = cpu;
    event.data.eqTime.us = eqElapsedUs;
    event.data.eqTime.activeBands = static_cast<uint8_t>(activeBands);
    event.data.eqTime.order = static_cast<uint8_t>(order);
    event.data.eqTime.budgetPercent = budgetPermille;
    if (eqDiagBuffer->push(event))
    {
        eqTickPushed->value.fetch_add(1, std::memory_order_relaxed);
        eqTotalPushed->fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        eqTickDropped->value.fetch_add(1, std::memory_order_relaxed);
    }
}
#endif
```

### DSPCoreFloat.cpp から削除するもの

- `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内の anonymous namespace ブロック全体（L34-L39: eqDiagBuffer/eqTickPushed/eqTickDropped/eqTotalPushed）
- `setEqDiagBuffer()` 関数定義全体（L45-L57）
- `logEqTime()` 関数定義全体（L69-L102）

### DSPCoreDouble.cpp から削除するもの

- `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内の anonymous namespace ブロック全体（L49-L54: eqDiagBuffer/eqTickPushed/eqTickDropped/eqTotalPushed）
- `logEqTime()` 関数定義全体（L57-L96）
- コメント `// setEqDiagBuffer の実体は DSPCoreFloat.cpp にあり（ODR回避）`（不要になる）

### CMakeLists.txt に追加

CMakeLists.txt の既存オーディオエンジンファイルリストに `src/audioengine/AudioEngine.Diagnostics.cpp` を追加（任意の適切な位置、例: `AudioEngine.Cache.cpp` の隣）。

---

## Fix 3: ANS_SWITCH → LockFreeRingBuffer (doubleパス)

### 置き換え前

`DSPCoreDouble.cpp::processOutputDouble()` L795-L809:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansStartUs = convo::getCurrentTimeUs();
#endif
        adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
        ...
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if (ansElapsedUs > 10)
        {
            juce::String alog("[ANS_SWITCH] us=");
            ...
            DBG(alog);            // NOLINT(rt-logger)
            juce::Logger::writeToLog(alog); // NOLINT(rt-logger)
        }
#endif
```

### 3-a: AudioEngine.h — DiagCategory拡張

```cpp
// ★ [work65] ANS_SWITCH bank switch timing
struct AnsSwitchData {
    uint64_t elapsedUs;
};

enum class DiagCategory : uint8_t {
    ...
    CallbackArrival = 8,
    AnsSwitchTime = 9,             // ★ 追加
    Count                           // 10
};

// DiagEvent union に ansSwitchTime を追加（既存最大サイズ88を超えない）
```

`static_assert(sizeof(DiagEvent) == kDiagEventSizeMax)` は変更不要（`AnsSwitchData` が最大メンバを超えないため）。
`static_assert(static_cast<int>(DiagCategory::Count) == 10)` を Timer.cpp 側で更新。

### 3-b: DSPCoreDouble.cpp — 置き換え後

変更ポイント:
- `this->currentCallbackSeq` を `eventIndex` に使用
- 閾値 `ansElapsedUs > 10` 撤廃 → `CONVOPEQ_DIAG_SAMPLE_MASK` でサンプリング
- 出力先: `eqDiagBuffer`（Fix2で単一TU化済み）
- **push成功/失敗を既存統計カウンタと同様に更新**

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
        if ((currentCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0 && eqDiagBuffer != nullptr)
        {
            DiagEvent event{};
            event.category = DiagCategory::AnsSwitchTime;
            event.eventIndex = currentCallbackSeq;
            event.data.ansSwitchTime.elapsedUs = ansElapsedUs;
            if (eqDiagBuffer->push(event))
            {
                // ★ work65: AnsSwitchTime push成功 → pushedカウンタ（既存統計と同様）
                //   カウンタ変数は eqTickPushed/eqTickDropped と共用
                eqTickPushed->value.fetch_add(1, std::memory_order_relaxed);
                eqTotalPushed->fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                eqTickDropped->value.fetch_add(1, std::memory_order_relaxed);
            }
        }
#endif
```

**注意**: `eqDiagBuffer`, `eqTickPushed`, `eqTickDropped`, `eqTotalPushed` は Fix2 で単一TU化され `namespace` 内で共有されているため、**DSPCoreDouble.cpp の anonymous namespace 内には存在しない**。アクセスには宣言が必要だが、Fix2 でこれらの変数は `AudioEngine.Diagnostics.cpp` の anonymous namespace 内にあるため、**DSPCoreDouble.cpp からは直接アクセスできない**。

**解決策**: 以下のいずれか:
- **案A**: ANS_SWITCH の統計カウンタを独立した別変数として DSPCoreDouble.cpp の anonymous namespace 内に追加する（Fix2 の変数と混ぜない）
- **案B**: `setEqDiagBuffer` と同様に、ANS_SWITCH 用の統計変数へのポインタを `AudioEngine.Diagnostics.cpp` 内で管理し、setter経由で設定する

**採用: 案A**（最もシンプル。既存の EQ_TIME 統計と ANS_SWITCH 統計を分離する）

```cpp
// DSPCoreDouble.cpp anonymous namespace 内:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
static DiagPerTickCounter s_ansTickPushed{};
static DiagPerTickCounter s_ansTickDropped{};
static std::atomic<uint64_t> s_ansTotalPushed{0};
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

`timerCallback()` → `diagBuffer.pop()` → `formatDiagEvent()` → `diagLog()` → `asyncSink()` パイプラインがそのままANS_SWITCHを消費。

---

## Fix 4: ANS_SWITCH → LockFreeRingBuffer (floatパス)

`DSPCoreIO.cpp::processOutput()` 内の同じコードパターンを Fix 3 と同一方法で置き換える。

floatパス（`DSPCoreIO.cpp`）の `processOutput()` は `DSPCore` メンバ関数のため `this->currentCallbackSeq` にアクセス可能。

統計カウンタは案Aに従い `DSPCoreIO.cpp` の anonymous namespace 内に独立して追加:

```cpp
// DSPCoreIO.cpp anonymous namespace 内:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
static DiagPerTickCounter s_ansIOTickPushed{};
static DiagPerTickCounter s_ansIOTickDropped{};
static std::atomic<uint64_t> s_ansIOTotalPushed{0};
#endif
```

---

## Fix 5: DiagnosticsConfig.h デフォルト0化（再発防止）

### 修正前

```cpp
// DiagnosticsConfig.h L19-21:
#ifndef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 1   // ← バグ: コメントには「デフォルト0」とある
#endif
```

### 修正後

```cpp
#ifndef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 0   // ← コメントと一致
#endif
```

### 変更後の動作

| ビルド設定 | コンパイラ定義 | `#ifndef` 判定 | 結果 |
|-----------|--------------|---------------|------|
| `cmake ..` (CMake OFF) | 未定義 | 成立 → `#define 0` | **診断OFF** ✅ |
| `cmake .. -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` | 不成立（既定義） | **診断ON** ✅ |
| `cmake .. -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` | 未定義 | 成立 → `#define 0` | **診断OFF** ✅ |

### この修正をwork65に含める理由

- このバグにより **診断コードがCMake設定に関わらず常時有効** になる
- Fix3/Fix4でANS_SWITCHの同期writeToLogを除去しても、**将来の改修で新たな `DBG`/`writeToLog` が書かれた場合に常時有効のまま**になり、同じ問題が再発する
- 修正量は **わずか1文字**（`1` → `0`）であり、work66へ回す理由がない
- 設計者判断により本work65に含める

---

## Fix 6: `[[maybe_unused]] diagLog` 維持

`DSPCoreDouble.cpp` L41 と `DSPCoreFloat.cpp` L9 の `[[maybe_unused]] diagLog()` は削除しない。
- 実害なし
- 将来の診断用途で再利用される可能性

---

## 変更ファイル一覧

| # | ファイル | 変更内容 | 行数変化 |
|---|---------|---------|---------|
| 1 | **新規** `src/audioengine/AudioEngine.Diagnostics.cpp` | eqDiagBuffer変数 + setEqDiagBuffer + logEqTime 集約 | **+60行** |
| 2 | `src/audioengine/AudioEngine.h` | AnsSwitchData / DiagCategory::AnsSwitchTime / DiagEvent union / logEqTime宣言 | +12行 |
| 3 | `CMakeLists.txt` | AudioEngine.Diagnostics.cpp エントリ追加 | +1行 |
| 4 | `src/DiagnosticsConfig.h` | `#define ... 1` → `0`（1文字） | 0行 |
| 5 | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | キャプチャコード全削除 / eqDiagBuffer重複削除 / logEqTime重複削除 / ANS_SWITCH DiagEvent化 | -110行 / +25行 |
| 6 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | eqDiagBuffer重複削除 / logEqTime重複削除 / setEqDiagBuffer移動 | -50行 |
| 7 | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | ANS_SWITCH writeToLog→DiagEvent push | -5行 / +18行 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | formatDiagEvent AnsSwitchTime / static_assert更新 | +4行 |

**正味削減行数**: 約40行削減（新規ファイル分を差し引いても減少）

---

## ビルド確認項目

- [ ] `cmake --build build --config Debug` が通ること
- [ ] `cmake --build build --config Release` が通ること（診断コードがコンパイルされないことを確認）
- [ ] `cmake --build build --config Debug -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` が通ること（明示的ONで診断コードがコンパイルされることを確認）
- [ ] `check-audioengine-lint.ps1` がパスすること
- [ ] `check-src-atomic-dotcall.ps1` がパスすること
- [ ] work21 EpochDomain CI Gate がパスすること
- [ ] CLI Smoke Test が通ること

---

## 確定調査結果（v2からの継続）

以下の未確定事項はv2での調査により確定済み。詳細は v2 版を参照。

| 項目 | 確定結果 |
|------|---------|
| eqDiagBuffer双定義 | 🐛 完全確認（TU分離によるnullptr） → Fix2で修正 |
| Convolver Runtime | ✅ ブロッキングゼロ |
| JUCE dsp Oversampling | ✅ prepareToPlayで事前確保されていればRT-safe |
| pushAdaptiveCaptureBlocks | ✅ LockFreeRingBuffer RT-safe |
| NOLINT(rt-logger) 全4箇所 | 内訳確定（2件削除・2件維持） |

### 唯一の残未確定

ANS_SWITCH + 旧writeToLog と XRUN の時系列相関。本work65で `eventIndex = currentCallbackSeq` を導入することで修正後のETW再測定が可能になる。
