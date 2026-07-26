# バグ修正改修計画書 — ConvoPeq work88

**作成日**: 2026-07-26
**対象**: 検証レポート v2（18件のバグ報告確定版）
**分類**: コード品質 / メモリ安全性 / データ競合 / パフォーマンス

---

## 0. 改修方針

### 優先順位決定基準

1. **CRITICAL**: 未定義動作（UB）、use-after-free、データ破壊を引き起こすもの
2. **HIGH**: 音質劣化、メモリリーク、機能喪失に直結するもの
3. **MEDIUM**: パフォーマンス低下、計測機能欠落、ARM64移植性問題
4. **LOW**: コーディング規約の一貫性、理論上のリスク

### 改修順序の依存関係

```
BUG12 (SafeStateSwapper no-op)
  └→ BUG13 (epoch bump before swap) — BUG12修正後に顕在化
      └→ 両方修正後、SafeStateSwapper RCUが正しく機能

BUG-001 (Float peak limiter)
BUG-002 (Float loudness meter)     ← 同一ファイル (DSPCoreIO.cpp)、同時修正推奨
BUG-003 (Float true peak detector)
BUG-004 (Float zeroupper)

BUG-009 (zeroupper 9ファイル)       ← BUG-004の拡張、同一パターンの系統的修正

BUG-005〜008 (static_cast<size_t>)  ← 24箇所の機械的修正、並行作業可能
```

---

## 1. 🔴 CRITICAL — 最優先（3件）

### 1.1 BUG12: SafeStateSwapper enterStateReader/exitStateReader No-Op

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — Use-after-free |
| **ファイル** | `ConvolverProcessor.h:268-269` |
| **修正難易度** | ★☆☆☆☆（超軽微） |
| **影響範囲** | `isCacheEntrySafeToDelete()`, `createSnapshotFromCurrentState()` |
| **テスト有無** | `ConvolverProcessorSwapTests.cpp` 等で間接カバー有 |
| **回帰リスク** | LOW — SafeStateSwapperへの委譲のみ |
| **見積り** | 15分（コード変更）＋30分（レビュー＋テスト） |

**修正内容**:
```cpp
// 変更前（ConvolverProcessor.h:268-269）
void enterStateReader(int /*readerIndex*/) const noexcept {}
void exitStateReader(int /*readerIndex*/) const noexcept {}

// 変更後
void enterStateReader(int readerIndex) const noexcept
{
    rcuSwapper.enterReader(readerIndex);
}
void exitStateReader(int readerIndex) const noexcept
{
    rcuSwapper.exitReader(readerIndex);
}
```

**依存**: この修正単体で問題を解決する。ただし、この修正により BUG13 が顕在化するため、BUG13とセットで修正すること。

---

### 1.2 BUG11: activeRuntimeDSPSlot / fadingRuntimeDSPSlot Non-Atomic Data Race

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — C++未定義動作（データ競合） |
| **ファイル** | `AudioEngine.h:1996-1999`, `AtomicAccess.h:31-33` |
| **修正難易度** | ★★☆☆☆（単純だが影響範囲の確認が必要） |
| **影響範囲** | `setActiveRuntimeDSP()`, `exchangeFadingRuntimeDSP()`, `getActiveRuntimeDSP()` |
| **テスト有無** | Latency関連テストで間接カバー |
| **回帰リスク** | LOW〜MEDIUM — アトミック化による性能影響は無視できる |
| **見積り** | 30分（コード変更）＋1h（全呼び出し元確認＋テスト） |

**修正内容**:
```cpp
// AudioEngine.h
// 変更前:
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };

// 変更後:
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
std::atomic<DSPCore*> fadingRuntimeDSPSlot{nullptr};

// アクセサも修正
inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
{
    return fadingRuntimeDSPSlot.exchange(value, std::memory_order_acq_rel);
}
inline DSPCore* getActiveRuntimeDSP() const noexcept
{
    return activeRuntimeDSPSlot.load(std::memory_order_acquire);
}
inline void setActiveRuntimeDSP(DSPCore* value) noexcept
{
    activeRuntimeDSPSlot.store(value, std::memory_order_release);
}
```

**注意点**:
- `NonOwningPtr` のセンチネルパターン（`~0` = retiring）と `std::atomic` の互換性確認
- `DSPTransition.h:93-94` の `reinterpret_cast<uintptr_t>(prevRaw) == ~static_cast<uintptr_t>(0)` チェックは `std::atomic` で返ったポインタでも同様に動作する

---

### 1.3 BUG4: xRunBuffer ACTIVATE イベント無条件プッシュ

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — イベント喪失によるRuntimeWorld追跡不能 |
| **ファイル** | `AudioBlock.cpp:605`, `BlockDouble.cpp:572` |
| **修正難易度** | ★☆☆☆☆（極軽微） |
| **影響範囲** | ACTIVATE検出パスのみ |
| **テスト有無** | XRun RingBufferテストで間接カバー |
| **回帰リスク** | LOW — 隣接コードと同一パターン |
| **見積り** | 10分（2行修正）＋20分（テスト＋確認） |

**修正内容**:
```cpp
// 変更前（AudioBlock.cpp:605, BlockDouble.cpp:572）:
xRunBuffer.push(ev);

// 変更後:
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

---

## 2. 🔴 HIGH — 高優先度（5件）

### 2.1 BUG13: SafeStateSwapper Epoch Bump Before Pointer Swap

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — BUG12修正後にUAF顕在化 |
| **ファイル** | `SafeStateSwapper.h:106-109` |
| **修正難易度** | ★★☆☆☆（順序入れ替え＋エポック記録修正） |
| **影響範囲** | 全 `SafeStateSwapper::swap()` 呼び出し元 |
| **テスト有無** | SafeStateSwapperテストあり |
| **回帰リスク** | MEDIUM — RCUのタイミング依存。Swapperテストがキャッチするはず |
| **見積り** | 1h（コード変更＋テスト＋レビュー） |
| **依存** | BUG12とセットで修正。BUG12単体で適用するとこの問題が顕在化する |

**修正内容**:
```cpp
// SafeStateSwapper.h:106-109 — 変更前
const uint64_t epoch1 = convo::fetchAddAtomic(globalEpoch, 1, acq_rel);   // bump #1
/* newEpoch = */ convo::fetchAddAtomic(globalEpoch, 1, acq_rel);           // bump #2
ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, acq_rel); // SWAP

// 変更後: SWAP → BUMP の順序に
ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, acq_rel); // SWAP first
const uint64_t epoch2 = convo::fetchAddAtomic(globalEpoch, 1, acq_rel);            // bump after
if (oldState != nullptr)
    retireEntry(oldState, epoch2);  // entryEpoch = epoch2
```

---

### 2.2 BUG-001: Float処理パスにPeak Limiterが未実装

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 音質劣化（ハードクリップ歪み） |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:506-514` |
| **修正難易度** | ★★★☆☆（メンバ追加の設計判断が必要） |
| **影響範囲** | Float処理パスの出力音質 |
| **テスト有無** | なし（現状のバグのため） |
| **回帰リスク** | MEDIUM — `peakLimiter` がAudioEngineのメンバであることを確認要 |
| **見積り** | 2h（設計確認＋実装＋テスト） |
| **依存** | なし（独立して修正可能） |

**修正内容**:
1. `DSPCoreIO.cpp` の `processOutput()` 内、ハードクリップ前に `peakLimiter.processBlock()` を追加
2. `DSPCoreIO` が AudioEngine の `peakLimiter` メンバにアクセスできることを確認
3. 必要なら `DSPCoreIO` に `peakLimiter` 参照を追加

```cpp
// DSPCoreIO.cpp processOutput() — NaN/Inf Scrub の後、Hard Clamp の前
constexpr double kPLThreshold = 0.8413951287507587;  // -1.5dBFS
constexpr double kPLKnee = 0.108748;
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
```

**事前確認事項**: AudioEngineの `peakLimiter` メンバが DSPCoreIO の `processOutput()` から可視であること。（`DSPCoreDouble.cpp:710` と同一パターンでアクセス可能なはず）

---

### 2.3 BUG17: overflowDurationMs 単位不一致（5ms→5秒）

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 慢性OVF検出が1000倍早く発動 |
| **ファイル** | `AudioEngine.Retire.cpp:134-137` |
| **修正難易度** | ★☆☆☆☆（1行の定数変更） |
| **影響範囲** | 慢性オーバーフロー検出のタイミング |
| **テスト有無** | Retire関連テストあり |
| **回帰リスク** | LOW — 閾値を意図値に変更するのみ |
| **見積り** | 15分（修正＋テスト） |

**修正内容**:
```cpp
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;  // → マイクロ秒！
chronicByDuration = (overflowDurationMs > 5000);  // 5ms で発動

// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;  // → ミリ秒
chronicByDuration = (overflowDurationMs > 5000);  // 5000ms = 5秒（意図通り）
```

---

### 2.4 BUG-010: retireEQStateDeferred/retireBandNodeDeferred 戻り値無視

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — MKLメモリリーク（17箇所） |
| **ファイル** | `EQProcessor.Core.cpp`, `EQProcessor.Parameters.cpp`, `EQProcessor.Coefficients.cpp` |
| **修正難易度** | ★★☆☆☆（設計判断: void化 vs 戻り値チェック） |
| **影響範囲** | EQパラメータ更新パス全体 |
| **テスト有無** | EQ関連テストあり |
| **回帰リスク** | LOW〜MEDIUM（オプション依存） |
| **見積り** | 1h（設計判断＋実装＋テスト） |

**推奨修正**: オプション1（void化＋内部フォールバック）

```cpp
// EQProcessor.Core.cpp
void EQProcessor::retireEQStateDeferred(EQState* state) noexcept
{
    if (!state) return;
    const uint64_t epoch = m_epochDomain.currentEpoch();
    if (!enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch))
    {
        // 退役失敗時のフォールバック: 直接解放
        deleteEQStatePtr(state);
    }
}
// 呼び出し元の (void) キャストを削除
retireEQStateDeferred(oldState);  // (void) 不要に
```

---

### 2.5 BUG10: DeferredDeletionQueue uint32_t ラップアラウンド

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — カウンタラップでキュー誤動作 |
| **ファイル** | `DeferredDeletionQueue.h:80,120,172` |
| **修正難易度** | ★☆☆☆☆（3行の型変更） |
| **影響範囲** | 全DeferredDeletionQueue利用パス |
| **テスト有無** | `DeferredDeletionQueueReclaimTests.cpp` あり |
| **回帰リスク** | LOW — 減算のセマンティクスが正しくなる方向 |
| **見積り** | 30分（3行修正＋テスト） |

**修正内容**:
```cpp
// DeferredDeletionQueue.h:80,120,172
// 変更前:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 変更後:
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

---

## 3. 🟡 MEDIUM — 中優先度（6件）

### 3.1 BUG-009: 9ファイルで `_mm256_zeroupper()` 欠落

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — AVX→SSE遷移ペナルティ |
| **影響ファイル** | 9ファイル |
| **修正難易度** | ★☆☆☆☆（9ファイル×1行挿入） |
| **影響範囲** | AVX2使用ホットパスの後続スカラー/SSEコード |
| **テスト有無** | なし（性能問題） |
| **回帰リスク** | LOW — 数サイクルのオーバーヘッドのみ |
| **見積り** | 1h（9ファイルの該当箇所特定＋挿入＋確認） |
| **備考** | BUG-004（DSPCoreIO）と重複するが、BUG-004は別件として独立 |

**修正対象ファイル一覧**:

| # | ファイル | AVX2使用箇所 | zeroupper挿入位置 |
|---|---------|-------------|-------------------|
| 1 | `CustomInputOversampler.cpp` | `isBadSampleV`, `loadStride2`, FIRカーネル | 各AVX2ブロック末尾 |
| 2 | `ConvolverProcessor.Runtime.cpp` | `#if defined(__AVX2__)` ブロック ×3 | 各 `#endif` 直前 |
| 3 | `MKLNonUniformConvolver.cpp` | FFT/conv処理 | AVX2使用関数末尾 |
| 4 | `SpectrumAnalyzerComponent.cpp` | スペクトル計算 | AVX2ブロック末尾 |
| 5 | `EQProcessor.Processing.cpp` | `applyGainRamp_AVX2` | 関数末尾 |
| 6 | `AudioEngine.EQResponse.cpp` | `__m256d` 多用箇所 | 各AVX2ブロック末尾 |
| 7 | `AudioEngine.Processing.DSPCoreIO.cpp` | NaN/Inf Scrub後 | BUG-004参照 |
| 8 | `ConvolverProcessor.LoaderThread.cpp` | Loader処理 | AVX2ブロック末尾 |
| 9 | `DSPCoreDouble.cpp` | ✅ 既にあり (742行) | — |

```cpp
// 各AVX2ブロックのスカラー遷移直前に追加:
_mm256_zeroupper();
```

---

### 3.2 BUG-004: DSPCoreIO.cpp の `_mm256_zeroupper()` 欠落

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:470-505` |
| **修正難易度** | ★☆☆☆☆ |
| **影響範囲** | Float処理パスのAVX→SSE遷移 |
| **見積り** | 15分 |
| **備考** | BUG-009の一部でもあるが独立管理。先に修正してよい |

**修正内容**: NaN/Inf Scrubブロックの後、`applyFixedLatencyDelay()` の前に `_mm256_zeroupper();` を追加。

---

### 3.3 BUG9: EQProcessor シャドウ relaxed-only データ競合

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — ARM64で顕在化リスク |
| **ファイル** | `EQProcessor.Core.cpp:592-596,655-659`, `EQProcessor.Processing.cpp` |
| **修正難易度** | ★★☆☆☆（5変数のstore/load ordering変更） |
| **影響範囲** | EQパラメータ伝搬パス |
| **テスト有無** | なし（タイミング依存） |
| **回帰リスク** | LOW — ordering強化のみ、セマンティクス不変 |
| **見積り** | 1h（全変数のordering変更＋コードレビュー） |

**修正内容**:
```cpp
// NonRT書き込み側（EQProcessor.Core.cpp）: relaxed → release
rtBypassedShadow.store(syncedBypassed, std::memory_order_release);

// RT読み取り側（EQProcessor.Processing.cpp）: relaxed → acquire
bool effectiveBypass = rtBypassedShadow.load(std::memory_order_acquire);
```

---

### 3.4 BUG-002: Float処理パスでLoudness Meter未実行

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — LUFSメーター不正確 |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:375` 周辺 |
| **修正難易度** | ★★☆☆☆ |
| **影響範囲** | FloatパスのLUFS測定 |
| **見積り** | 30分 |
| **備考** | BUG-003と同時修正推奨 |

**修正内容**: DC Blocker適用後、`loudnessMeter.processBlock(dataL, dataR, numSamples);` を追加。

---

### 3.5 BUG-003: Float処理パスでTruePeak Detector未実行

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — TruePeak測定不正確 |
| **ファイル** | `AudioEngine.Processing.DSPCoreIO.cpp:376-378` 周辺 |
| **修正難易度** | ★★☆☆☆ |
| **影響範囲** | FloatパスのTruePeak測定 |
| **見積り** | 30分 |
| **備考** | BUG-002と同時修正推奨 |

**修正内容**: DC Blocker適用後、`truePeakDetector.processBlock(dataL, dataR, numSamples);` を追加。

---

### 3.6 BUG16: Retire Overflow Timestamp 単位不一致（ナノ秒保存・μs読取）

| 項目 | 内容 |
|------|------|
| **リスク** | HIGHだが影響はデッドコードのため実害なし |
| **ファイル** | `ISRRetire.cpp:55-57`, `ISRRuntimePublicationCoordinator.cpp:279-284` |
| **修正難易度** | ★☆☆☆☆（1行の単位統一） |
| **影響範囲** | `overflowAgeWarnCallback_` コードパス |
| **見積り** | 20分 |

**修正内容**:
```cpp
// ISRRetire.cpp:56 — Producer を microseconds に修正
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
```

---

## 4. 🟢 LOW — 低優先度（1件、4サブカテゴリ）

### 4.1 BUG-005〜008: `static_cast<size_t>` 欠落（24箇所）

| 項目 | 内容 |
|------|------|
| **リスク** | LOW — MSVCでは実質安全、規約上の問題 |
| **対象ファイル** | 4ファイル、24箇所 |
| **修正難易度** | ★☆☆☆☆（機械的修正） |
| **影響範囲** | なし（移植性保証） |
| **回帰リスク** | LOW — キャスト明示化のみ |
| **見積り** | 1h（24箇所の機械的修正＋確認） |

**修正パターン**:
```cpp
// 変更前:
memset(dst, 0, n * sizeof(double));
memcpy(buf + wPos, src, samplesFirst * sizeof(double));

// 変更後:
memset(dst, 0, static_cast<size_t>(n) * sizeof(double));
memcpy(buf + wPos, src, static_cast<size_t>(samplesFirst) * sizeof(double));
```

**修正対象一覧（24箇所）**:

| ファイル | 行 | コードパターン |
|---------|-----|--------------|
| `MKLNonUniformConvolver.cpp` | 1030 | `memset(tempTime, 0, l.fftSize * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1037 | `memcpy(tempTime, irSrc + copyStart, copyLen * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1045 | `memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1080-1089 | `memcpy(swapSoA, realF, l.complexSize * sizeof(double))` ×6 |
| `MKLNonUniformConvolver.cpp` | 1156 | `memcpy(scratch, src, scratchSize * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1384 | `memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1417 | `memset(l.accumBuf, 0, l.partStride * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1488 | `memset(dst, 0, n * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1501 | `memset(dst + toRead, 0, (n - toRead) * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1538 | `memcpy(l.inputAccBuf + l.inputPos, input + consumed, toFill * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1540 | `memset(l.inputAccBuf + l.inputPos, 0, toFill * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1628 | `memset(l.accumBuf, 0, l.partStride * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1637 | `memcpy(l.tailOutputBuf, l.fftOutBuf + l.partSize, l.partSize * sizeof(double))` |
| `ConvolverProcessor.Runtime.cpp` | 388 | `memcpy(buf + wPos, src, samplesFirst * sizeof(double))` |
| `ConvolverProcessor.Runtime.cpp` | 390 | `memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double))` |
| `ConvolverProcessor.Lifecycle.cpp` | 248 | `memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double))` |
| `ConvolverProcessor.Lifecycle.cpp` | 249 | `memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double))` |
| `ConvolverProcessor.h` | 821 | `memcpy(l.get(), irData[0], irDataLength * sizeof(double))` |
| `ConvolverProcessor.h` | 822 | `memcpy(r.get(), irData[1], irDataLength * sizeof(double))` |

---

## 5. 改修スケジュール（推奨）

### Phase 1 — クリティカル修正（即日）

| 順 | BUG-ID | 工数 | 担当 |
|----|--------|------|------|
| 1 | BUG12 | 45min | 安全クリティカル、最も影響大 |
| 2 | BUG13 | 1h | BUG12とセットで修正 |
| 3 | BUG4 | 30min | 軽微、即効性あり |
| 4 | BUG11 | 1.5h | アトミック化、テスト確認含む |

**Phase 1 総工数**: 3.75h
**Phase 1 リリース判断**: 全修正完了後、CTestフルスイート + SafeStateSwapperテスト通過を条件とする

---

### Phase 2 — 高優先度修正（週内）

| 順 | BUG-ID | 工数 | 備考 |
|----|--------|------|------|
| 5 | BUG-001 | 2h | Floatパス音質修正。設計確認含む |
| 6 | BUG-010 | 1h | 戻り値破棄の是正。void化＋フォールバック |
| 7 | BUG-001/002/003 | 1h | Floatパス一括修正（同一ファイル） |
| 8 | BUG17 | 15min | 1行修正 |
| 9 | BUG10 | 30min | 3行修正 |

**Phase 2 総工数**: 4.75h
**Phase 2 リリース判断**: 全修正完了後、CTest + EQ関連テスト + Float/Double比較テスト通過

---

### Phase 3 — 中優先度修正（余裕のある時）

| 順 | BUG-ID | 工数 | 備考 |
|----|--------|------|------|
| 10 | BUG-009 | 1h | 9ファイルのzeroupper追加 |
| 11 | BUG-004 | 15min | BUG-009の一部だが独立管理 |
| 12 | BUG9 | 1h | relaxed→release/acquire |
| 13 | BUG16 | 20min | 単位統一 |

**Phase 3 総工数**: 2.5h

---

### Phase 4 — 低優先度（リファクタリング時に）

| 順 | BUG-ID | 工数 | 備考 |
|----|--------|------|------|
| 14 | BUG-005〜008 | 1h | 機械的修正。自動化可能 |

**Phase 4 総工数**: 1h

---

## 6. 総合見積り

| Phase | 工数 | リスク低減効果 |
|-------|------|---------------|
| Phase 1（CRITICAL） | 3.75h | 未定義動作・UAF・イベント喪失を排除 |
| Phase 2（HIGH） | 4.75h | 音質劣化・メモリリーク・OVF誤検出を修正 |
| Phase 3（MEDIUM） | 2.5h | AVX-SSEペナルティ・移植性問題を改善 |
| Phase 4（LOW） | 1h | コーディング規約の一貫性を確保 |
| **合計** | **12h** | |

---

## 7. リスク評価

| リスク | 影響 | 確率 | 対策 |
|--------|------|------|------|
| BUG12+BUG13 修正後のRCU動作不整合 | 高い | 中 | SafeStateSwapper単体テストを強化 |
| BUG11 アトミック化後のセンチネル互換性 | 中 | 低 | `DSPTransition.h` のパターンチェック |
| BUG-001 peakLimiterメンバ可視性不足 | 中 | 低 | 事前にAudioEngine.hのinclude関係を確認 |
| BUG-010 即時解放によるAudio Thread競合 | 中 | 低 | EpochDomainによる安全確認後、解放を実行 |
| BUG-009 zeroupper追加によるペナルティ悪化 | 低 | 極低 | `VZEROUPPER` は数サイクルのみ |
| 回帰テストの不足 | 中 | 中 | 各Phaseリリース前にCTestフルスイート必須 |

---

## 8. テスト計画

| Phase | テスト項目 | 期待結果 |
|-------|-----------|---------|
| Phase 1 | CTest全スイート | Pass |
| Phase 1 | SafeStateSwapper単体テスト | Pass（BUG12/13修正後） |
| Phase 1 | XRun RingBufferテスト | Pass（BUG4修正後） |
| Phase 2 | Float/Double出力比較テスト | FloatとDoubleの出力が一致（ピークリミッター追加後） |
| Phase 2 | EQパラメータ変更テスト | メモリリーク0（BUG-010修正後） |
| Phase 2 | 慢性OVF検出テスト | 5秒のタイムアウトで正しく発動（BUG17修正後） |
| Phase 2 | DeferredDeletionQueueテスト | ラップアラウンド後も正しい満杯/空検出（BUG10修正後） |
| Phase 3 | AVX-SSE遷移ベンチマーク | 小バッファ(32)でのレイテンシ改善確認 |
| Phase 3 | EQパラメータ伝搬テスト | ARM64エミュレーションで値の一貫性確認 |
| Phase 4 | 静的解析（cast検出） | `static_cast<size_t>` 欠落ゼロ |

---

## 9. 補足: オプションの検討

### BUG-001/002/003: Float/Doubleパスの共通化

現在の設計では DSPCoreDouble.cpp と DSPCoreIO.cpp で同様の処理が重複している。
中期的には `processOutput()` の共通ベースクラスまたはユーティリティ関数への
抽出を検討してもよいが、**本改修計画では最小修正を優先**する。

### BUG-010: フォールバック戦略

オプション1（void化＋内部フォールバック）を推奨する。
オプション2（戻り値チェック＋再試行）は複雑性が増す割にメリットが少ないため非推奨。
オプション3（ドロップカウンター追加）は監視のみでリークを防止しないため二次策。

### BUG-005〜008: 自動修正の可能性

sed または Python スクリプトによる機械的修正が可能。
ただし `l.complexSize * 2 * sizeof(double)` のような複合式は
`static_cast<size_t>(l.complexSize) * 2 * sizeof(double)` のように
正しいキャスト位置を判断する必要があるため、完全自動化には注意が必要。
