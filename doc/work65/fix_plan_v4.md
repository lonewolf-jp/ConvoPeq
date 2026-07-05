# work65: XRUN原因特定レビュー指摘対応 改修計画書 v4

**作成日**: 2026-07-05
**最終更新日**: 2026-07-05 (v4: Fix2 extern方式 / Fix3 統計カウンタ不要 / Fix5 前倒し確定)
**対象レビュー**: ConvoPeq × Voicemeeter Virtual ASIO XRUN多発 解析結果
**対応方針**:
- RTスレッド上のブロッキング要因（同期ファイルI/O・同期Logger書込み）をロックフリーリングバッファ経由に変更
- 診断フラグのビルド設定をCMakeと一致させる（再発防止）

---

## 全体サマリー

| # | 改修項目 | 根拠 | 影響行数 | 優先度 |
|---|---------|------|---------|--------|
| 1 | work52キャプチャコード削除 | 同期fopen/fwrite/fclose（未ガード） | -90行 | **CRITICAL** |
| 2 | `eqDiagBuffer` extern 一元化 | 双定義によるDoubleパスEQ_TIME常時消失 | **±0行** | **CRITICAL** |
| 3 | ANS_SWITCH → LockFreeRingBuffer (double) | RTスレッドで同期writeToLog | +8行 | HIGH |
| 4 | ANS_SWITCH → LockFreeRingBuffer (float) | floatパスも統一 | +8行 | LOW |
| 5 | **`DiagnosticsConfig.h` デフォルト0化** | CMake OFFが無視されるバグ | **1文字** | **CRITICAL** |

---

## Fix 1: work52キャプチャコード削除

`DSPCoreDouble.cpp` から以下を全削除。変更なし（v1〜v3から継続）。

**リスク**: ゼロ。調査用一時コード。

---

## Fix 2: `eqDiagBuffer` — extern参照に一元化（最小修正）

### 問題

`eqDiagBuffer` 関連4変数が**2つの翻訳単位で別々のstatic変数**として宣言されている:

| TU | 宣言場所 | 設定元 | 状態 |
|----|---------|--------|------|
| `DSPCoreFloat.cpp` | anonymous ns内（内部リンケージ） | `setEqDiagBuffer()` が自TU変数を設定 | ✅ 正常動作 |
| `DSPCoreDouble.cpp` | anonymous ns内（内部リンケージ） | **誰も設定しない** | ❌ `nullptr` |

`setEqDiagBuffer()` の実体は `DSPCoreFloat.cpp` にあり、代入先は**自TUの変数**のみ。よって `DSPCoreDouble.cpp` の `logEqTime()` → `eqDiagBuffer->push(event)` は常に `if (eqDiagBuffer == nullptr) return;` で早期returnし、EQ_TIMEイベントがdoubleパスから永久消失する。

### 修正方針（extern 一元化 — 最小修正）

**AudioEngine.h** に extern 宣言を追加し、全TUから同じ実体を参照する:

```cpp
// AudioEngine.h（setEqDiagBuffer 宣言の近く、L470付近）:
// ★ [work65] EQ_TIME診断バッファ — 実体は DSPCoreFloat.cpp（external linkage）
extern LockFreeRingBuffer<DiagEvent, DiagRuntimeLimits::BufferCapacity>* eqDiagBuffer;
extern DiagPerTickCounter* eqTickPushed;
extern DiagPerTickCounter* eqTickDropped;
extern std::atomic<uint64_t>* eqTotalPushed;
```

**なぜ AudioEngine.h か**: `DSPCoreIO.cpp` は `DiagnosticsConfig.h` を include していないが、`AudioEngine.h` はすべての該当ファイルから include されている。また `AudioEngine.h` は既に `setEqDiagBuffer()` / `DiagEvent` / `LockFreeRingBuffer` などを宣言しており、責務が一致する。

**DSPCoreFloat.cpp** — 変数のリンケージを internal → external に変更。anonymous namespace を早期クローズし、eqDiagBuffer 変数群を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内で file scope に出す:

```cpp
// 修正前（anonymous ns → internal linkage）:
namespace {
    // ...diagLog()...
    // ...inline helpers...
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    namespace {
        LockFreeRingBuffer<...>* eqDiagBuffer = nullptr;  // internal linkage
        ...
    }
#endif
} // close outer anonymous ns

// 修正後（anonymous ns 早期クローズ + #if 内で file scope 宣言）:
namespace {
    // ...diagLog()...
    // ...inline helpers...
} // close outer anonymous ns EARLY

// ★ [work65] external linkage — AudioEngine.h で extern 宣言
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
LockFreeRingBuffer<...>* eqDiagBuffer = nullptr;
DiagPerTickCounter* eqTickPushed = nullptr;
DiagPerTickCounter* eqTickDropped = nullptr;
std::atomic<uint64_t>* eqTotalPushed = nullptr;
#endif

// setEqDiagBuffer と logEqTime の位置はそのまま（setEqDiagBuffer は常時コンパイル）
```

**なぜ `#if` ガード内に残すか**: 診断OFF時（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS==0`）は `logEqTime()` 自体がコンパイルされないため、`eqDiagBuffer` 変数を常時定義する意味がない。Releaseビルドで不要なグローバル変数を増やさないため、実体定義は `#if` ガード内に残す。`extern` 宣言も常時（`AudioEngine.h`）で問題ない。実体は `DSPCoreFloat.cpp` に1つだけ存在するためリンク時に解決可能であり、2TU目以降の `#include` で同一実体を正しく参照できる。

**DSPCoreDouble.cpp** — 重複変数宣言ブロックを削除 + 古い ODR コメントを削除:

```cpp
// 削除するブロック（#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 内）:
// namespace {
//     LockFreeRingBuffer<...>* eqDiagBuffer = nullptr;  ← 削除
//     ...
// }
// // setEqDiagBuffer の実体は DSPCoreFloat.cpp にあり（ODR回避） ← このコメントも削除
```

**注意**: `#include <cstdint>` はキャプチャコード削除後も残す（`uint64_t`, `uint32_t`, `uint8_t` が残存コードで使用）。

`logEqTime()` の定義は**両ファイルに残す**（inline相当の小さな関数であり、外部関数化する必要はない）。修正後、DSPCoreDouble.cpp の `logEqTime()` は `AudioEngine.h` の `extern` 宣言を通じて DSPCoreFloat.cpp の変数を正しく参照する。

### 影響

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| TU間の変数共有 | ❌ 独立した2実体（Double側nullptr） | ✅ 単一実体をextern共有 |
| 新規ファイル | — | なし |
| CMakeLists.txt 変更 | — | なし |
| `AudioEngine.h` 変更 | extern宣言4行追加（責務範囲内） | **+4行** |
| 総変更行数 | — | **宣言移動のみ、±0行** |

**extern宣言追加時の推奨コメント**:

```cpp
// ★ [work65] Shared runtime diagnostics state — external linkage.
// Must be shared across DSPCoreFloat.cpp, DSPCoreDouble.cpp, and
// DSPCoreIO.cpp. DO NOT move into an anonymous namespace in any
// of those translation units.
```

このコメントにより、将来の改修者が「これはexternで共有すべき変数」と認識できるようになり、anonymous namespace に再び閉じ込める誤った修正を防止できる。

---

## Fix 3: ANS_SWITCH → LockFreeRingBuffer (doubleパス)

### 置き換え前

`DSPCoreDouble.cpp::processOutputDouble()` 内:

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansStartUs = convo::getCurrentTimeUs();
#endif
        adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
        // ...
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        const uint64_t ansElapsedUs = convo::getCurrentTimeUs() - ansStartUs;
        if (ansElapsedUs > 10)                          // ← 閾値でデータ消失
        {
            juce::String alog("[ANS_SWITCH] us=");
            // ...
            DBG(alog);                                    // NOLINT(rt-logger)
            juce::Logger::writeToLog(alog);                // NOLINT(rt-logger)  ← 同期I/O
        }
#endif
```

### 3-a: AudioEngine.h — DiagCategory拡張

```cpp
struct AnsSwitchData {
    uint64_t elapsedUs;           // バンク切替時間 (μs)
};

enum class DiagCategory : uint8_t {
    ...
    CallbackArrival = 8,
    AnsSwitchTime = 9,            // ★ 追加
    Count                          // 10
};

// DiagEvent union に ansSwitchTime 追加（既存最大88byteを超えない）
```

`kDiagEventSizeMax == 88` の static_assert は変更不要。`DiagCategory::Count` の static_assert は `10` に更新。

### 3-b: DSPCoreDouble.cpp — 置き換え後

変更ポイント:
- ✅ `eventIndex = this->currentCallbackSeq`（既存診断イベントと同じ粒度）
- ✅ 閾値 `>10` 撤廃 → `CONVOPEQ_DIAG_SAMPLE_MASK` でサンプリング
- ✅ **統計カウンタは追加しない** — `diagBuffer`（リング単位）のpush成功/失敗は永続的統計として既存の `rtAuxMutable_.diagTickPushed/Popped/Dropped` で管理される。ANS_SWITCH 個別のカウンタは不要
- ✅ 出力先: `eqDiagBuffer`（Fix2で extern 一元化済み）

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
            [[maybe_unused]] const bool pushed = eqDiagBuffer->push(event);  // RT-safe: drop on full
            juce::ignoreUnused(pushed); // ANS_SWITCH統計はリング単位で管理。個別カウンタ不要。
        }
#endif
```

### 3-c: AudioEngine.h — recordCallbackArrival と同様のinline helper追加（任意）

現状の `recordCallbackArrival` のようなinline helperを追加してもよいが、コード量が小さいため `processOutputDouble()` 内に直接記述する。

### 3-d: AudioEngine.Timer.cpp — formatDiagEvent追加

```cpp
case DiagCategory::AnsSwitchTime:
    return diagPrefix(gen) + " [ANS_SWITCH] cbIdx="
        + juce::String(static_cast<juce::int64>(event.eventIndex))
        + " us=" + juce::String(static_cast<juce::int64>(event.data.ansSwitchTime.elapsedUs));

static_assert(static_cast<int>(DiagCategory::Count) == 10,
    "DiagCategory enum changed: update formatDiagEvent() accordingly");
```

### 消費側: 変更不要

`timerCallback()` → `diagBuffer.pop()` → `formatDiagEvent()` → `diagLog()` → `asyncSink()` パイプラインがそのままANS_SWITCHを消費。`MaxDrainPerTick=16`で十分。

---

## Fix 4: ANS_SWITCH → LockFreeRingBuffer (floatパス)

`DSPCoreIO.cpp::processOutput()` 内の同一コードを Fix 3-b と同様に置き換える。統計カウンタも追加しない。`this->currentCallbackSeq` は `DSPCore` メンバとしてアクセス可能。

---

## Fix 5: DiagnosticsConfig.h デフォルト0化（再発防止）

### 修正前

```cpp
#ifndef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 1   // ← CMake OFF時に強制的にON
#endif
```

### 修正後

```cpp
#ifndef CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
#define CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS 0   // ← CMakeと一致。OFFならOFF
#endif
```

### 変更後の動作マトリクス

| ビルド設定 | コンパイラ定義 | `#ifndef` | 結果 |
|-----------|--------------|-----------|------|
| `cmake ..` (default OFF) | 未定義 | 成立→`#define 0` | **診断OFF** ✅ |
| `cmake .. -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` | `=1` | 不成立（既定義） | **診断ON** ✅ |
| Debugビルド（CMake OFF） | 未定義 | →`#define 0` | 診断OFF ⚠️ 必要な場合は-Dで明示的ON |

**注意**: これまで DiagnosticsConfig.h の fallback により Debug ビルドでも常に診断ONだった。本修正後は Debug ビルドでも CMake 設定（デフォルトOFF）が尊重される。診断が必要な場合は `cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` を明示する。

---

## 変更ファイル一覧

| # | ファイル | 変更内容 | 行数変化 |
|---|---------|---------|---------|
| 1 | `src/DiagnosticsConfig.h` | `#define ... 1` → `0` | **+0行**（1文字変更） |
| &nbsp; | `src/audioengine/AudioEngine.h` | extern宣言4行追加（setEqDiagBuffer近く） | **+4行** |
| 2 | `src/audioengine/AudioEngine.h` | AnsSwitchData / DiagCategory::AnsSwitchTime / DiagEvent union / static_assert | +10行 |
| 3 | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | キャプチャコード削除 / eqDiagBuffer重複宣言削除 / ANS_SWITCH DiagEvent化 | **-80行** / +10行 |
| 4 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | eqDiagBuffer変数の anonymous ns 解除（external linkage化） | **±0行** |
| 5 | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | ANS_SWITCH writeToLog→DiagEvent push | -5行 / +8行 |
| 6 | `src/audioengine/AudioEngine.Timer.cpp` | formatDiagEvent AnsSwitchTime / static_assert更新 | +4行 |

**新規ファイル**: なし
**CMakeLists.txt 変更**: なし
**正味削減行数**: 約60行削減

---

### extern宣言の可視性確認

| ファイル | `AudioEngine.h` include | `DiagnosticsConfig.h` include | extern可視性 |
|---------|----------------------|-----------------------------|-------------|
| `DSPCoreDouble.cpp` | ✅ L3 | ✅ L4 | ✅ `AudioEngine.h` 経由 |
| `DSPCoreFloat.cpp` | ✅ L3 | ✅ L4 | ✅ `AudioEngine.h` 経由 |
| `DSPCoreIO.cpp` | ✅ L4 | ❌ **未include** | ✅ `AudioEngine.h` 経由（こちらが決め手） |

`DSPCoreIO.cpp` は `DiagnosticsConfig.h` を include していないため、もし extern を `DiagnosticsConfig.h` に置くと Fix 4（DSPCoreIO.cpp の ANS_SWITCH）がコンパイルエラーになる。extern を `AudioEngine.h` に置くことで全ファイルで確実に可視となる。

### 【推奨】`eqDiagBuffer` 変数名のリネーム

Fix3/Fix4 により `eqDiagBuffer` には EQ_TIME と AnsSwitchTime の両イベントが流れる。`eq`（EQ専用の意）という名前に実態が合わなくなる。`runtimeDiagBuffer` や `rtDiagBuffer` へのリネームを推奨する。本work65の必須タスクではないが、変数名と実態の不一致はコード理解の妨げになるため、近いワークでの実施を強く推奨する。

---

## ビルド確認項目

- [ ] `cmake --build build --config Debug` が通ること（診断OFF=デフォルト）
- [ ] `cmake --build build --config Release` が通ること
- [ ] `cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON -B build` + `cmake --build build --config Debug` が通ること（明示的ON）
- [ ] `check-audioengine-lint.ps1` がパスすること
- [ ] `check-src-atomic-dotcall.ps1` がパスすること
- [ ] work21 EpochDomain CI Gate がパスすること

---

## 残余調査の確定報告（2026-07-05 全項目確定）

work65策定過程で発生した未確定事項・懸念事項をソースコード静的解析により全件調査した。

### 【✅ 確定】1. `#include <cstdint>` は維持

`DSPCoreDouble.cpp` の `#include <cstdint>` は Fix1 のキャプチャコードと共に削除しない。残存する `logEqTime()` や `isFiniteNoLibm()` 等が `uint64_t`, `uint32_t`, `uint8_t` を使用しているため。`JuceHeader.h` や `AudioEngine.h` からの推移的includeに依存せず、明示的に維持する。

### 【✅ 確定】2. ODRコメントの削除

DSPCoreDouble.cpp L55 の `// setEqDiagBuffer の実体は DSPCoreFloat.cpp にあり（ODR回避）` は Fix2 で不要になる。extern 宣言により変数が一元化されるため、コメントも削除する。

### 【✅ 確定】3. extern 宣言は無条件（#if ガード不要）

AudioEngine.h に追加する extern 宣言は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガードしない。理由:
- `DiagEvent` / `LockFreeRingBuffer` / `DiagPerTickCounter` の型定義は AudioEngine.h に常時存在（条件付きではない）
- DSPCoreFloat.cpp の変数実体も常時定義（`#if` ガード外、OFF時は nullptr）
- 診断OFF時は `logEqTime()` の `if (eqDiagBuffer == nullptr) return;` で早期return

### 【✅ 確定】4. `convDiagBuffer` は単一TU — 双定義問題なし

`ConvolverProcessor.Runtime.cpp` のみに定義されており、`setConvDiagBuffer` も同一TU内。`eqDiagBuffer` と同様の双定義バグは存在しない。

### 【✅ 確定】5. DSPCoreFloat/Double 間に他に重複変数なし

anonymous namespace ブロックを全数調査した結果、`eqDiagBuffer` 以外に2TU間で重複する変数は存在しない。インライン補助関数（`isFiniteNoLibm`, `scaleBlockFallback` 等）は意図的なコード重複（internal linkage + inline により安全）。

### 【✅ 確定】6. DSPCoreIO.cpp の include 経路は正常

DSPCoreIO.cpp は `DiagnosticsConfig.h` を include していないが、extern 宣言を `AudioEngine.h` に置くことで問題ない。`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` が DSPCoreIO.cpp で未定義の場合、`#if` ブロックはプリプロセス時にスキップされ、コンパイルエラーにならない。

### 【✅ 確定】7. `convolverRt()` / `eqRt()` は RT-safe

両関数とも単純な `return convolverState->ref()` / `return eqState->ref()` の inline getter。`jassert` は Release ビルドで no-op。ヒープ確保・mutex・I/O なし。

### 【⚠️ 残留意事項】DiagnosticsConfig 修正後の Debug ビルド運用

Fix5 後は Debug ビルドでも診断コードがデフォルト OFF になる。診断が必要な開発者は `cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` を明示すること。この運用変更を README または開発メモに記載推奨。

### extern 宣言を AudioEngine.h に置く理由

v4初期案では `DiagnosticsConfig.h` への extern 追加を想定していたが、`DSPCoreIO.cpp` が `DiagnosticsConfig.h` を include していない（`AudioEngine.h` のみ）ためコンパイルエラーになる。`AudioEngine.h` は全3ファイルから include されており、かつ既に `setEqDiagBuffer()` / `DiagEvent` / `LockFreeRingBuffer` 等の実行時診断APIが宣言されているため、責務としても適切。

### Fix2: extern 一元化を選んだ理由（v3の新規TU案から変更）

| 観点 | extern 一元化（v4） | 新規TU集約（v3） |
|------|-------------------|-----------------|
| 変更範囲 | 3ファイル（既存のみ） | 4ファイル+新規1 |
| CMake修正 | 不要 | 必要 |
| API追加 | なし | `logEqTime` 宣言追加 |
| 設計負債 | externが数行増える | 翻訳単位増加 |
| バグ修正範囲 | バグと同程度 | バグより大きい |

**判断**: 修正対象のバグ（2TUで変数が分離）に対して、新しい翻訳単位を作るのはオーバー。extern で済ませる方が変更範囲・レビュー容易性・保守性のバランスが良い。

### Fix3: 統計カウンタを追加しない理由（v3の案Aから変更）

ANS_SWITCH イベントは `eqDiagBuffer`（== `AudioEngine::diagBuffer`）に push される。このリングバッファのpush成功/失敗は `rtAuxMutable_.diagTickPopped/Dropped` でリング単位で管理されている。ANS_SWITCH 個別のカウンタを追加すると、Timer側の統計（`diagBuffer.pop()` 数）とずれる可能性がある。「リング単位で管理するべき」という設計判断による。

### Fix5: work65 に前倒しした理由

CMake設定を無視して常時診断ONになるバグは、Fix3/Fix4 でANS_SWITCHを修正しても**同じパターンの再発**（将来の改修で新たな `DBG`/`writeToLog` が書かれた場合に常時有効のまま）を許す。修正量が1文字で副作用も限定されることから、work66に回さず本work65で対処する。

---

## 追加確認: コードベース全体の網羅的調査結果（2026-07-05）

### 【✅ 確定】`eqDiagBuffer` 以外の双定義バグは存在しない

`src/` 全体の `static` / `namespace` 変数、`*DiagBuffer` ポインタ、`set*DiagBuffer` パターンを全件調査:

| 変数 | 宣言TU | 件数 | 状態 |
|------|--------|------|------|
| `eqDiagBuffer` | DSPCoreFloat.cpp + DSPCoreDouble.cpp | **2TU** | ❌ **Fix2で修正** |
| `convDiagBuffer` | ConvolverProcessor.Runtime.cpp（単一） | 1TU | ✅ 問題なし |
| `diagBuffer` (AudioEngineメンバ) | AudioEngine.h（単一メンバ） | 1箇所 | ✅ 問題なし |
| `s_logBuffer`, `s_droppedLogs` | AudioEngine.Timer.cpp（単一） | 1TU | ✅ 問題なし |
| 各種 `inline` 補助関数 | 各cpp anonymous ns内 | 各TU独立 | ✅ 意図的重複（internal linkage） |

**結論**: 今回のwork65で修正する `eqDiagBuffer` 以外に、TU間で重複して初期化されない変数は存在しない。

### 【✅ 確定】`setConvDiagBuffer` は単一定義

- **宣言**: AudioEngine.h L474（1箇所のみ）
- **定義 + 実装**: ConvolverProcessor.Runtime.cpp L66（1TUのみ）
- `eqDiagBuffer` と異なり `setConvDiagBuffer` と変数実体が同一TU内に存在するため、双定義問題は発生していない。

### 【✅ 確定】`RUNTIME_DIAG_LOG` マクロはデッドコード

`DiagnosticsConfig.h` L31-35 で定義されているが、**コードベース全体で1回も使用されていない**。work65では対象外とし、必要に応じてwork66以降で削除を検討。

### 【✅ 確定】`NOLINT` 全9箇所の内訳

| # | ファイル | 種類 | 行 | 内容 | work65影響 |
|---|---------|------|----|------|-----------|
| 1 | DSPCoreDouble.cpp | rt-logger | L43 | `diagLog()`内DBG+writeToLog | 🟡 維持（dead code） |
| 2 | DSPCoreDouble.cpp | rt-logger | L44 | 同上 | 🟡 維持（dead code） |
| 3 | **DSPCoreDouble.cpp** | **rt-logger** | **L808** | **ANS_SWITCH: DBG** | **❌ Fix3: RingBuffer化** |
| 4 | **DSPCoreDouble.cpp** | **rt-logger** | **L809** | **ANS_SWITCH: writeToLog** | **❌ Fix3: RingBuffer化** |
| 5 | DSPCoreFloat.cpp | rt-logger | L11 | `diagLog()`内DBG+writeToLog | 🟡 維持（dead code） |
| 6 | DSPCoreFloat.cpp | rt-logger | L12 | 同上 | 🟡 維持（dead code） |
| 7 | **DSPCoreIO.cpp** | **rt-logger** | **L438** | **ANS_SWITCH: DBG** | **❌ Fix4: RingBuffer化** |
| 8 | **DSPCoreIO.cpp** | **rt-logger** | **L439** | **ANS_SWITCH: writeToLog** | **❌ Fix4: RingBuffer化** |
| 9-12 | 他ファイル | atomic-dot-call | 4箇所 | counter/timestamp操作 | ✅ 非RT、対象外 |
| 13 | DiagnosticsConfig.h | atomic-dot-call | L87 | CASループ | ✅ 非RT、対象外 |

**結論**: 全 `NOLINT` の所在と分類が確定。RT違反（rt-logger）は今回のFix3/Fix4で全件対応。atomic-dot-callは非RTで問題なし。

### 【未確定】ANS_SWITCH + XRUN の時系列相関

本work65の全修正後にETW再測定を行う。`eventIndex = currentCallbackSeq` を用いて、ログ出力とXRUNイベントの突き合わせが可能となる。
