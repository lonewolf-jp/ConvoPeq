# バグ修正改修計画書 v3 — ConvoPeq work88（設計者向け確定版）

**作成日**: 2026-07-26
**分類**: 設計指示 / 未確定事項 / Appendix

---

## 目次

1. **設計** — 実装作業に必要な情報のみ（即時着手可能）
2. **未確定事項** — 追加設計検討・確認が必要な項目
3. **Appendix** — 参考情報（分類表・スケジュール・調査方法）

---

# 1. 設計 — 実装着手可能な修正指示

> このセクションには、**今すぐ実装を開始できる確定済みの修正**のみを記載する。
> 各項目にはファイルパス・行番号・修正コード・注意点を含む。

---

## 1-A: CRITICAL — 最優先

### A1. BUG12: ConvolverProcessor.h enterStateReader/exitStateReader No-Op

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — SafeStateSwapper RCU無効によるUse-After-Free |
| **ファイル** | `src/ConvolverProcessor.h:268-269` |
| **修正量** | 4行追加（2箇所の空実装をSafeStateSwapper委譲に置換） |
| **依存** | なし（単独修正可能） |
| **テスト** | SafeStateSwapper単体テスト + CTest |

**修正内容**:
```cpp
// ConvolverProcessor.h:268-269
// 変更前:
void enterStateReader(int /*readerIndex*/) const noexcept {}
void exitStateReader(int /*readerIndex*/) const noexcept {}

// 変更後:
void enterStateReader(int readerIndex) const noexcept
{
    rcuSwapper.enterReader(readerIndex);
}
void exitStateReader(int readerIndex) const noexcept
{
    rcuSwapper.exitReader(readerIndex);
}
```

**注意点**:
- `ConvolverProcessor` が `SafeStateSwapper` メンバ `rcuSwapper` を持つことを確認すること（`ConvolverProcessor.h` 内のメンバ宣言を参照）
- BUG13（swap順序問題）は本修正とは独立して存在する。BUG12単体で適用しても問題ないが、BUG13の修正設計完了後に両者を組み合わせることを推奨

---

### A2. BUG11: AudioEngine.h activeRuntimeDSPSlot / fadingRuntimeDSPSlot のアトミック化

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — Non-Atomicポインタのデータ競合（C++ UB） |
| **ファイル** | `src/audioengine/AudioEngine.h:1996-1999`, `src/audioengine/AtomicAccess.h`（関連） |
| **修正量** | 型変更 + アクセサ修正（〜15行） |
| **確認事項** | `DSPTransition.h:93` のセンチネル判定との互換性確認済み |

**修正内容**:

```cpp
// AudioEngine.h:1996-1999
// 変更前:
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };

// 変更後:
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
std::atomic<DSPCore*> fadingRuntimeDSPSlot{nullptr};
```

```cpp
// AudioEngine.h:2001-2017 アクセサ
// 変更前:
inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
{
    DSPCore* previous = fadingRuntimeDSPSlot.get();
    fadingRuntimeDSPSlot.operator=(value);
    return previous;
}
[[nodiscard]] inline DSPCore* getActiveRuntimeDSP() const noexcept
{
    return activeRuntimeDSPSlot.get();
}
inline void setActiveRuntimeDSP(DSPCore* value) noexcept
{
    activeRuntimeDSPSlot = value;
}

// 変更後:
inline DSPCore* exchangeFadingRuntimeDSP(DSPCore* value) noexcept
{
    return fadingRuntimeDSPSlot.exchange(value, std::memory_order_acq_rel);
}
[[nodiscard]] inline DSPCore* getActiveRuntimeDSP() const noexcept
{
    return activeRuntimeDSPSlot.load(std::memory_order_acquire);
}
inline void setActiveRuntimeDSP(DSPCore* value) noexcept
{
    activeRuntimeDSPSlot.store(value, std::memory_order_release);
}
```

**センチネル互換性（DSPTransition.h:93）**:
```cpp
// DSPTransition.h:93 — 現行コードのセンチネル判定（変更不要）
// prevRaw は getActiveRuntimeDSP() の戻り値（DSPCore*）になる
if (auto* prev = (reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0)))
    ? nullptr : reinterpret_cast<DSPCore*>(prevRaw))
// std::atomic<DSPCore*>::load() が DSPCore* を返すため、このコードはそのまま動作する
```

---

### A3. BUG4: xRunBuffer ACTIVATE イベントの戻り値チェック追加

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL — キュー満杯時にACTIVATEイベントが通知なく消失 |
| **ファイル** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:605`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:572` |
| **修正量** | 2ファイル×2行追加（隣接コードと同一パターン） |

**修正内容**:
```cpp
// AudioBlock.cpp:605 および BlockDouble.cpp:572
// 変更前:
xRunBuffer.push(ev);

// 変更後:
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

**注意点**: 隣接するXRUNイベントのプッシュ（AudioBlock.cpp:579, BlockDouble.cpp:546）と同一パターン。コピー＆アダプトで問題ない。

---

## 1-B: HIGH — 高優先度

### B1. BUG-001: Float処理パスにPeak Limiter追加

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — ハードクリップ歪みによる音質劣化 |
| **ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` |
| **挿入位置** | NaN/Inf Scrub後、`applyFixedLatencyDelay` の前、Hard Clamp（`juce::jlimit`）の前 |
| **可視性** | ✅ `peakLimiter` は `AudioEngine::DSPCore` のメンバ（AudioEngine.h:960）。DSPCoreIO.cpp は DSPCore のメンバ関数を実装しているため直接アクセス可能。追加の参照やインクルードは不要 |

**修正内容**:
```cpp
// DSPCoreIO.cpp processOutput() 内、NaN/Inf Scrubブロックの後、
// applyFixedLatencyDelay() の前に追加:
constexpr double kPLThreshold = 0.8413951287507587;  // -1.5dBFS
constexpr double kPLKnee = 0.108748;
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
```

**テスト注意点**:
- Float/Doubleのビット一致比較は**行わない**こと（処理パスが異なるため）
- メトリクス値（RMS、ピークレベル、THD+N）の整合性または許容誤差による判定を使用すること

---

### B2. BUG-002/003: Float処理パスにLoudness Meter / TruePeak Detector追加

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — LUFS/TruePeakメーターがFloatパスで機能しない |
| **ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` |
| **挿入位置** | DC Blocker適用後、NaN/Inf Scrub前 |

**修正内容**:
```cpp
// DSPCoreIO.cpp processOutput() 内、DC Blocker適用直後に追加:
auto& dc = dcBlockers();
dc.outputL.process(dataL, numSamples);
if (dataR) dc.outputR.process(dataR, numSamples);

truePeakDetector.processBlock(dataL, dataR, numSamples);
loudnessMeter.processBlock(dataL, dataR, numSamples);

// 以降、既存の NaN/Inf scrub → Dither → Hard Clamp の流れは変更なし
```

---

### B3. BUG17: overflowDurationMs の単位修正

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 慢性OVF検出が意図の1000分の1（5ms）で発動 |
| **ファイル** | `src/audioengine/AudioEngine.Retire.cpp:136` |
| **修正量** | 1行の定数変更 |
| **根拠** | MSVC `steady_clock::duration` = nanoseconds（Microsoft公式ドキュメント確認済み）。`now`・`overflowStart` とも同じ `steady_clock::count()` 由来でナノ秒。`/1000` ではマイクロ秒にしかならない |

**修正内容**:
```cpp
// AudioEngine.Retire.cpp:136
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;       // → マイクロ秒
chronicByDuration = (overflowDurationMs > 5000);                        // 5ms で発動

// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;  // → ミリ秒
chronicByDuration = (overflowDurationMs > 5000);                        // 5000ms = 5秒
```

---

### B4. BUG16: ISRRetire Overflow Timestamp の単位統一

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — overflowAgeWarnコードパスがデッドコード |
| **ファイル** | `src/audioengine/ISRRetire.cpp:55-57`（Producer） |
| **修正量** | 1行のduration_cast追加 |
| **根拠** | フィールド名 `overflowTimestampUs`（マイクロ秒）に合わせ、Producer側でマイクロ秒に変換してから保存する |

**修正内容**:
```cpp
// ISRRetire.cpp:56
// 変更前:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count()), 0};
//                      ↑ ナノ秒を保存

// 変更後:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
//                      ↑ マイクロ秒に変換して保存
```

**注意点**: Consumer側（`ISRRuntimePublicationCoordinator.cpp:279-284`）は既に `duration_cast<microseconds>` を使用しているため変更不要。

---

### B5. BUG10: DeferredDeletionQueue の差分計算修正

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — 32-bitカウンタラップ時にキューが誤満杯/空判定 |
| **ファイル** | `src/DeferredDeletionQueue.h:80,120,172` |
| **修正量** | 3行の型変更（`intptr_t`→`int32_t`） |

**修正内容**:
```cpp
// DeferredDeletionQueue.h — 3箇所すべて
// 変更前:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 変更後:
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

**該当箇所一覧**:
| 関数 | 行 | 役割 |
|------|-----|------|
| `enqueue()` | 80 | スロット可用性チェック |
| `reclaim()` | 120 | エントリ準備完了チェック |
| `drainAllUnsafe()` | 172 | エントリ準備完了チェック |

---

## 1-C: LOW（後回し可）

### C1. BUG-009: 9ファイルに `_mm256_zeroupper()` 追加

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — AVX→SSE遷移ペナルティ（性能） |
| **修正量** | 9ファイル×各AVX2ブロック末尾に1行追加 |
| **優先度** | 正当性の核心ではない。性能修正。後回し可 |

**修正対象ファイル一覧**:
| # | ファイル | 挿入位置 |
|---|---------|----------|
| 1 | `src/CustomInputOversampler.cpp` | `isBadSampleV`, `loadStride2`, FIRカーネル各AVX2ブロック末尾 |
| 2 | `src/convolver/ConvolverProcessor.Runtime.cpp` | `#if defined(__AVX2__)` 各ブロックの `#endif` 直前 |
| 3 | `src/MKLNonUniformConvolver.cpp` | AVX2使用FFT/conv関数末尾 |
| 4 | `src/SpectrumAnalyzerComponent.cpp` | スペクトル計算AVX2ブロック末尾 |
| 5 | `src/eqprocessor/EQProcessor.Processing.cpp` | `applyGainRamp_AVX2` 関数末尾 |
| 6 | `src/audioengine/AudioEngine.EQResponse.cpp` | `__m256d` 使用各ブロック末尾 |
| 7 | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | NaN/Inf Scrubブロック後（BUG-004に同じ） |
| 8 | `src/convolver/ConvolverProcessor.LoaderThread.cpp` | Loader処理AVX2ブロック末尾 |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | ※2026-07-26時点のコードではAVX2不使用。関数名に"AVX2"とあるがスカラーコードのため対象外 |

```cpp
// 各AVX2ブロックのスカラーコード遷移直前に追加:
_mm256_zeroupper();
```

---

### C2. BUG-005〜008: `static_cast<size_t>` 欠落の是正（24箇所）

| 項目 | 内容 |
|------|------|
| **リスク** | LOW — MSVCでは実質安全。コーディング規約の一貫性 |
| **対象** | 4ファイル、24箇所 |
| **修正パターン** | `memset/memcpy(..., expr * sizeof(T))` → `memset/memcpy(..., static_cast<size_t>(expr) * sizeof(T))` |
| **優先度** | 最下位。機械的修正。まとめて処理 |

**完全棚卸し結果**:

| ファイル | 行 | コード |
|---------|-----|--------|
| `MKLNonUniformConvolver.cpp` | 1030 | `memset(tempTime, 0, l.fftSize * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1037 | `memcpy(tempTime, irSrc + copyStart, copyLen * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1045 | `memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1080-1089 | `memcpy(swapSoA, realF, l.complexSize * sizeof(double))` etc. (6行) |
| `MKLNonUniformConvolver.cpp` | 1156 | `memcpy(scratch, src, scratchSize * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1384 | `memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1417 | `memset(l.accumBuf, 0, l.partStride * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1488 | `memset(dst, 0, n * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1501 | `memset(dst + toRead, 0, (n - toRead) * sizeof(double))` |
| `MKLNonUniformConvolver.cpp` | 1538,1540 | `memcpy/memset` 関連 |
| `MKLNonUniformConvolver.cpp` | 1628,1637 | `memset/memcpy` 関連 |
| `ConvolverProcessor.Runtime.cpp` | 388 | `memcpy(buf + wPos, src, samplesFirst * sizeof(double))` |
| `ConvolverProcessor.Runtime.cpp` | 390 | `memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double))` |
| `ConvolverProcessor.Lifecycle.cpp` | 248 | `memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double))` |
| `ConvolverProcessor.Lifecycle.cpp` | 249 | `memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double))` |
| `ConvolverProcessor.h` | 821 | `memcpy(l.get(), irData[0], irDataLength * sizeof(double))` |
| `ConvolverProcessor.h` | 822 | `memcpy(r.get(), irData[1], irDataLength * sizeof(double))` |

---

# 2. 未確定事項 — 追加設計検討・確認が必要な項目

> このセクションの項目は**実装着手前に追加の設計検討または確認が必要**。
> 各項目に「現状の理解」と「未確定な点」を明記する。

---

## 2-A. BUG13: SafeStateSwapper swap() の epoch 退避期間問題 ★ 設計再検討中

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — BUG12修正後にUAFが顕在化する可能性 |
| **ステータス** | ❌ **v1の修正案（SWAP→BUMP）は却下**。代替案を検討中 |
| **BUG12との関係** | BUG12（enterStateReader no-op）が現在マスク。BUG12修正後に顕在化する |

### 現状の理解

`SafeStateSwapper::swap()` の現在の順序:
```
epoch1 = fetchAdd(globalEpoch)  // bump#1
fetchAdd(globalEpoch)           // bump#2
Exchange(activeState, newState) // swap → retire with epoch1
```

問題のインターリーブ:
| Writer | Reader | Reclaimer |
|--------|--------|-----------|
| bump#1 → N+1 | | |
| | enterReader → epoch=N+1 記録 | |
| bump#2 → N+2 | | |
| swap → retire oldState(N) | | |
| | getState() → 旧ポインタを見る | |
| | | isOlder(N, N+1) = true → **解放→UAF** |

### 未確定な点

1. **修正案A**: `retireEpoch = epoch2`（bump#2後の値 = epoch1+2）を使用する
   - 問題: reader が epoch1+2 を記録しても swap 前に旧ポインタを見る可能性
2. **修正案B**: swap → bump（ただしbump数は1回のみ、epochはswap後の値）
   - 問題: RCU退避期間が弱まる（レビュー指摘）
3. **修正案C**: 現在の設計はメモリ順序保証により安全であることを証明する
   - 要: acq_rel chain の完全な証明

**推奨アクション**: 実装着手前に、案A〜Cのいずれか（または新案）を選択し、SafeStateSwapper単体テストで検証すること。

---

## 2-B. BUG-010: retireEQStateDeferred 戻り値破棄の扱い ★ 修正方針確定待ち

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH — MKLメモリリーク（17箇所） |
| **ステータス** | ⚠️ **v1のvoid化案は設計変更のため非推奨**。カウンタ追加案を推奨中だが確定していない |
| **RTスレッド安全性** | ✅ `retireEQStateDeferred` / `retireBandNodeDeferred` はNonRTパスからのみ呼ばれる（EQProcessor.Processing.cppでは不使用） |

### 現状の理解

```cpp
// EQProcessor.Core.cpp:99,108
bool EQProcessor::retireEQStateDeferred(EQState* state) noexcept
{
    // → Coordinator → Router → DeferredDeleteQueue の3段階で退役
    return enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch);
}
```

全17箇所の呼び出し元で `(void)retireEQStateDeferred(oldState)` として戻り値を破棄。

### 未確定な点

1. **カウンタ追加案**: `if (!retireEQStateDeferred(oldState)) m_retireDropCount++`
   - ✅ 設計変更不要、診断可能
   - ❌ リーク自体は防止しない
2. **フォールバックdelete案**: 失敗時に直接 `deleteEQStatePtr(state)` を呼ぶ
   - ✅ リークを完全防止
   - ❌ Coordinator/RouterのISR epoch追跡をバイパスする
3. **現状維持**: `(void)` のまま
   - リークは発生しうるが、運用上は稀

**推奨アクション**:
- カウンタ追加案を暫定採用し、`m_retireDropCount` の宣言場所（`EQProcessor.h`）を決定する
- リーク率の実測データが得られた時点で、フォールバックdelete案の再評価を行う

---

## 2-C. BUG9: EQProcessor シャドウ relaxed-only データ競合

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM — ARM64移植時に顕在化の可能性 |
| **ステータス** | 🟡 修正自体は明確だが優先度低い。x86では実質問題なし |

### 未確定な点

- ARM64対応の時期：ARM64対応計画と併せて修正するか、先行的に修正するかの判断
- 修正範囲：
  - NonRT書き込み側: `memory_order_relaxed` → `memory_order_release`（EQProcessor.Core.cpp:592-596,655-659）
  - RT読み取り側: `memory_order_relaxed` → `memory_order_acquire`（EQProcessor.Processing.cpp:415-417,497,509,863,1019）
  - RT書き込み側: `memory_order_relaxed` のままでもよいが、一貫性のためにreleaseに統一する判断もあり

---

# 3. Appendix — 参考情報

---

## A. 全BUG 採否判定サマリ

| # | BUG-ID | 分類 | リスク | 判定（v3） | 設計セクション |
|---|--------|------|-------|-----------|--------------|
| 1 | BUG12 | RCU | CRITICAL | ✅ 設計-1A | A1 |
| 2 | BUG11 | データ競合 | CRITICAL | ✅ 設計-1A | A2 |
| 3 | BUG4 | イベント喪失 | CRITICAL | ✅ 設計-1A | A3 |
| 4 | BUG-001 | 音質劣化 | HIGH | ✅ 設計-1B | B1 |
| 5 | BUG-002 | 計測欠落 | MEDIUM | ✅ 設計-1B | B2 |
| 6 | BUG-003 | 計測欠落 | MEDIUM | ✅ 設計-1B | B2 |
| 7 | BUG17 | 単位誤差 | HIGH | ✅ 設計-1B | B3 |
| 8 | BUG16 | デッドコード | HIGH | ✅ 設計-1B | B4 |
| 9 | BUG10 | ロックフリー | HIGH | ✅ 設計-1B | B5 |
| 10 | BUG-009 | パフォーマンス | MEDIUM | 🟡 設計-1C | C1 |
| 11 | BUG-004 | パフォーマンス | MEDIUM | 🟡 設計-1C | C1 |
| 12 | BUG-005〜008 | 規約 | LOW | 🟢 設計-1C | C2 |
| 13 | BUG13 | RCU設計 | HIGH | ❌ **未確定-2A** | 2-A |
| 14 | BUG-010 | メモリリーク | HIGH | ⚠️ **未確定-2B** | 2-B |
| 15 | BUG9 | データ競合 | MEDIUM | 🟡 **未確定-2C** | 2-C |

---

## B. 修正推奨順序

```
Phase 1 — 設計セクション1-A（CRITICAL 3件）
  ├ A1. BUG12: ConvolverProcessor.h enterStateReader委譲          [45min]
  ├ A2. BUG11: AudioEngine.h atomic<DSPCore*>化                    [1.5h]
  └ A3. BUG4:  xRunBuffer.push戻り値チェック                       [30min]

Phase 2 — 設計セクション1-B（HIGH 5件）
  ├ B1. BUG-001: DSPCoreIO.cpp peakLimiter追加                     [2h]
  ├ B2. BUG-002/003: DSPCoreIO.cpp meter/detector追加              [1h]
  ├ B3. BUG17: AudioEngine.Retire.cpp /1000000                     [15min]
  ├ B4. BUG16: ISRRetire.cpp duration_cast追加                     [20min]
  └ B5. BUG10: DeferredDeletionQueue.h int32_t                     [30min]

Phase 3 — 未確定事項の設計確定（2-A, 2-B, 2-C）
  ├ 2-A. BUG13: swap順序の再設計 + 検証                           [要設計検討]
  ├ 2-B. BUG-010: カウンタ追加 or フォールバックdelete決定         [要判断]
  └ 2-C. BUG9: relaxed ordering 修正時期判断                       [要判断]

Phase 4 — 設計セクション1-C（後回し）
  ├ C1. BUG-009/004: 9ファイルzeroupper追加                       [1h]
  └ C2. BUG-005〜008: static_cast<size_t> 24箇所                   [1h]
```

## C. 調査方法

本計画書の各項目は以下の調査に基づく:

| ツール | 使用目的 |
|--------|---------|
| **ripgrep/grep/sed/awk/find (WSL)** | 全パターン検索・コード解析 |
| **Python3 (WSL)** | 全memset/memcpyサイトの自動棚卸し |
| **AiDex MCP** | コードインデックス・シンボル検索 |
| **serena MCP** | シンボル探索・型情報取得 |
| **semble-search** | コードコンテキスト検索 |
| **Web Search** | MSVC chronoドキュメント確定、Intel AVX-SSE penalty docs |
| **PowerShell Select-String** | 補完的Windows側検索 |

## D. 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1 | 2026-07-26 | 初版。全18件の修正計画 |
| v2 | 2026-07-26 | レビュー反映。採用/保留/却下の再分類。BUG13案却下。BUG-010カウンタ案に修正 |
| v3 | 2026-07-26 | 設計/未確定事項/Appendix の3部構成に再編。実装者向けに作業手順を明確化 |
