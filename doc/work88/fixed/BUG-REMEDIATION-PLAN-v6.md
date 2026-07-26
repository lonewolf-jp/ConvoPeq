# バグ修正改修計画書 v6 — ConvoPeq work88（確定版）

**作成日**: 2026-07-26
**前版**: v5
**最終レビュー通過**: 2026-07-26
**状態**: ✅ 確定済み — 全項目の実装詳細確定

---

## 目次

1. **設計** — 実装着手可能な修正指示
   - 1-A: ✅ 確定着手可能
   - 1-B: 📋 レビュー承認済み実装案
   - 1-C: 🟡 後回し可（実装詳細確定済み）
2. **設計確定済み** — 実装詳細確定（元・未確定事項）
   - 2-A: BUG13 — epoch2案 確定
   - 2-B: BUG-010 — カウンタ案 確定
   - 2-C: BUG9 — 保留（ARM64次第）
3. **Appendix** — 参考情報

---

# 1. 設計 — 実装着手可能な修正指示

> **凡例**:
> - ✅ **確定着手可能**: レビュー通過。即座に実装を開始してよい。
> - 📋 **レビュー承認済み実装案**: 方向性は確定。行単位の最終照合または実動作確認を推奨。
> - 🟡 **後回し可**: 修正方針・実装詳細とも確定。優先度低いため後回し。

---

## 1-A: ✅ 確定着手可能

### A1. BUG12: ConvolverProcessor.h enterStateReader/exitStateReader → SafeStateSwapper委譲

**リスク**: CRITICAL — RCU reader tracking無効によるUse-After-Free
**ファイル**: `src/ConvolverProcessor.h:268-269`
**修正**: 4行の委譲追加

```cpp
void enterStateReader(int readerIndex) const noexcept
{
    rcuSwapper.enterReader(readerIndex);
}
void exitStateReader(int readerIndex) const noexcept
{
    rcuSwapper.exitReader(readerIndex);
}
```

---

### A2. BUG11: AudioEngine.h activeRuntimeDSPSlot / fadingRuntimeDSPSlot → std::atomic<DSPCore*> 化

**リスク**: CRITICAL — Non-Atomicポインタのデータ競合（C++ UB）
**ファイル**: `src/audioengine/AudioEngine.h:1996-1999`, アクセサ2001-2017

```cpp
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
std::atomic<DSPCore*> fadingRuntimeDSPSlot{nullptr};

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

**最終ビルド確認推奨**: `DSPTransition.h:93` の sentinel 判定（`reinterpret_cast<uintptr_t>(prevRaw) == ~0`）が `std::atomic` 化後も正しく動作することを1回確認すること。

---

### A3. BUG4: xRunBuffer ACTIVATE イベントの戻り値チェック追加

**リスク**: HIGH — キュー満杯時にACTIVATEイベントが通知なく消失
**ファイル**: `AudioBlock.cpp:605`, `BlockDouble.cpp:572`

```cpp
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

---

### A4. BUG10: DeferredDeletionQueue 差分計算修正

**リスク**: HIGH — 32-bitカウンタラップ時にキュー誤動作
**ファイル**: `DeferredDeletionQueue.h:80,120,172`（3箇所）

```cpp
// kQueueSize(=4096) ≪ INT32_MAX により int32_t 減算で安全。
// 差分が queue size を超えないことを前提とする。
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

---

## 1-B: 📋 レビュー承認済み実装案

### B1. BUG-001/002/003: Float処理パス（DSPCoreIO.cpp）に不足機能を追加

**リスク**: HIGH（BUG-001: 音質劣化）/ MEDIUM（BUG-002/003: 計測欠落）
**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
**挿入位置**: NaN/Inf Scrub #2（AVX2ブロック）直後、`applyFixedLatencyDelay` の前

```cpp
    // NaN/Inf Scrub #2 の AVX2 ブロック直後
    _mm256_zeroupper();
    truePeakDetector.processBlock(dataL, dataR, numSamples);
    loudnessMeter.processBlock(dataL, dataR, numSamples);
    constexpr double kPLThreshold = 0.8413951287507587;
    constexpr double kPLKnee = 0.108748;
    peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
    // 以降は既存コード（applyFixedLatencyDelay → HardClamp）変更なし
```

### B2. BUG17: overflowDurationMs 単位修正

**ファイル**: `AudioEngine.Retire.cpp:136`

```cpp
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;        // 5ms
// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;  // 5秒
```

### B3. BUG16: ISRRetire Overflow Timestamp 単位統一

**ファイル**: `ISRRetire.cpp:55-57`（Producer）

```cpp
// 変更前: count() → ナノ秒を保存
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count()), 0};
// 変更後: duration_cast<microseconds> → マイクロ秒を保存
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
```

---

## 1-C: 🟡 後回し可（実装詳細確定済み）

> 以下の2項目は修正方針・実装詳細とも確定している。優先度が低いため後回しとする。

### C1. BUG-009/004: `_mm256_zeroupper()` 欠落の是正（9ファイル）

**方針**: AVX2を使った関数の末尾、スカラー/SSE/JUCEコードへ降りる直前に1回だけ入れる。

**テンプレート**:
```cpp
#if defined(__AVX2__)
    _mm256_zeroupper();
#endif
```

**配置ルール**（3つだけ）:
1. AVX2ブロックの**最後**に置く
2. **1関数につき1回**にする
3. すでに `_mm256_zeroupper()` がある場所は**触らない**

**既存の正しい配置例**（`DSPCoreDouble.cpp:741-742`）:
```cpp
// AVX→legacy SSE 境界: _mm256_zeroupper() を配置
_mm256_zeroupper();
juce::FloatVectorOperations::copy(buffer.getWritePointer(0, 0), dataL, numSamples);
```

**修正対象ファイル一覧**:
| # | ファイル | 挿入位置 |
|---|---------|----------|
| 1 | `CustomInputOversampler.cpp` | `isBadSampleV`, `loadStride2`, FIRカーネル各AVX2ブロック末尾 |
| 2 | `ConvolverProcessor.Runtime.cpp` | `#if defined(__AVX2__)` 各ブロックの `#endif` 直前 |
| 3 | `MKLNonUniformConvolver.cpp` | AVX2使用FFT/conv関数末尾（残差のみ） |
| 4 | `SpectrumAnalyzerComponent.cpp` | スペクトル計算AVX2ブロック末尾 |
| 5 | `EQProcessor.Processing.cpp` | `applyGainRamp_AVX2` 関数末尾 |
| 6 | `AudioEngine.EQResponse.cpp` | `__m256d` 使用各ブロック末尾 |
| 7 | `AudioEngine.Processing.DSPCoreIO.cpp` | NaN/Inf Scrub #2直後（BUG-004, B1と同時修正） |
| 8 | `ConvolverProcessor.LoaderThread.cpp` | Loader処理AVX2ブロック末尾 |

**除外するファイル**:
- `DSPCoreFloat.cpp` — 2026-07-26時点のコードではAVX2不使用。関数名に"AVX2"とあるが実体はスカラーコードのため対象外

---

### C2. BUG-005〜008: `static_cast<size_t>` 欠落の是正（24箇所）

**方針**: 機械的な残差修正。すでに `static_cast<size_t>` が入っている箇所は触らない。残りの箇所のみを埋める。

**改修ルール**:
```cpp
// 基本パターン（n は int 型のサイズ変数）:
memcpy(dst, src, static_cast<size_t>(n) * sizeof(double));
memset(dst, 0, static_cast<size_t>(n) * sizeof(double));

// 複合式では掛け算の前にキャスト（int * int を先に評価しない）:
memcpy(dst, src, static_cast<size_t>(complexSize) * 2 * sizeof(double));
```

**禁止パターン**: `static_cast<size_t>(expr * sizeof(T))` — 後段でまとめてキャストすると `int` 溢れのリスクが残る。

**完全棚卸し結果（24箇所）**:

| ファイル | 行 | 変更前 | 変更後 |
|---------|-----|--------|--------|
| `MKLNonUniformConvolver.cpp` | 1030 | `memset(tempTime, 0, l.fftSize * sizeof(double))` | `memset(..., static_cast<size_t>(l.fftSize) * sizeof(double))` |
| 同 | 1037 | `memcpy(tempTime, irSrc + copyStart, copyLen * sizeof(double))` | `memcpy(..., static_cast<size_t>(copyLen) * sizeof(double))` |
| 同 | 1045 | `memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double))` | `memcpy(..., static_cast<size_t>(l.complexSize) * 2 * sizeof(double))` |
| 同 | 1080-1089 | `memcpy(swapSoA, realF, l.complexSize * sizeof(double))` (6行) | 各々 `static_cast<size_t>(l.complexSize)` |
| 同 | 1156 | `memcpy(scratch, src, scratchSize * sizeof(double))` | `memcpy(..., static_cast<size_t>(scratchSize) * sizeof(double))` |
| 同 | 1384 | `memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double))` | `memcpy(..., static_cast<size_t>(l.partStride) * sizeof(double))` |
| 同 | 1417 | `memset(l.accumBuf, 0, l.partStride * sizeof(double))` | `memset(..., static_cast<size_t>(l.partStride) * sizeof(double))` |
| 同 | 1488 | `memset(dst, 0, n * sizeof(double))` | `memset(dst, 0, static_cast<size_t>(n) * sizeof(double))` |
| 同 | 1501 | `memset(dst + toRead, 0, (n - toRead) * sizeof(double))` | `memset(dst + toRead, 0, static_cast<size_t>(n - toRead) * sizeof(double))` |
| 同 | 1538,1540 | `memcpy/memset` 関連 | 同パターン |
| 同 | 1628,1637 | `memset/memcpy` 関連 | 同パターン |
| `ConvolverProcessor.Runtime.cpp` | 388 | `memcpy(buf + wPos, src, samplesFirst * sizeof(double))` | `memcpy(..., static_cast<size_t>(samplesFirst) * sizeof(double))` |
| 同 | 390 | `memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double))` | `memcpy(..., static_cast<size_t>(samplesSecond) * sizeof(double))` |
| `ConvolverProcessor.Lifecycle.cpp` | 248 | `memcpy(irL.get(), conv->irData[0], conv->irDataLength * sizeof(double))` | `memcpy(..., static_cast<size_t>(conv->irDataLength) * sizeof(double))` |
| 同 | 249 | `memcpy(irR.get(), conv->irData[1], conv->irDataLength * sizeof(double))` | 同 |
| `ConvolverProcessor.h` | 821 | `memcpy(l.get(), irData[0], irDataLength * sizeof(double))` | `memcpy(..., static_cast<size_t>(irDataLength) * sizeof(double))` |
| 同 | 822 | `memcpy(r.get(), irData[1], irDataLength * sizeof(double))` | 同 |

---

# 2. 設計確定済み（元・未確定事項）

> 以下の項目は**レビューにより実装詳細が確定した**。未確定ではなくなった。
> 2-C のみ ARM64対応時期との兼ね合いで時期未定。

---

## 2-A. BUG13: SafeStateSwapper swap() — epoch2 を retire に使用 ★ 実装詳細確定

**リスク**: HIGH — BUG12修正後にUAF顕在化の可能性
**状態**: ✅ **実装詳細確定** — epoch2案を採用

### 最終確定した修正内容

`SafeStateSwapper::swap()` において、retire に `epoch1` ではなく **`epoch2`（bump#2後の値）** を使用する。

```cpp
void swap(ConvolverState* newState) noexcept
{
    const uint64_t epoch1 = convo::fetchAddAtomic(globalEpoch, 1u, std::memory_order_acq_rel);
    const uint64_t epoch2 = convo::fetchAddAtomic(globalEpoch, 1u, std::memory_order_acq_rel);

    ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, std::memory_order_acq_rel);
    if (oldState == nullptr)
        return;

    // リングバッファに積む（既存コードのまま）
    ...

    // ★ 変更点: retired エントリには epoch1 ではなく epoch2 を記録
    convo::publishAtomic(retiredBuffer[t].state, oldState, std::memory_order_release);
    // retired エントリには「swap 後の保護境界」を記録する。
    // bump ウィンドウ中に oldState を観測した reader を保護するため、epoch2 を使う。
    convo::publishAtomic(retiredBuffer[t].epoch, epoch2, std::memory_order_release);
    ...
}
```

### この修正の根拠

- bump#1 と swap の間に enterReader した reader は epoch = epoch1+1 または epoch1+2 を保持する
- 従来の `epoch1` 退役では `isOlder(epoch1, epoch1+1) = true` となり解放されてしまう
- `epoch2`（= epoch1+1）退役では `isOlder(epoch1+1, epoch1+1) = false` となり、bump ウィンドウ中の reader が保護される
- 2-step bump の構造を壊さず、退役境界だけを後ろへずらせる

### 単体テスト（必須）

**本修正は単体テストを必須条件とする。** `BUG12` の reader tracking を有効にした上で、以下の確認を行う。

**テスト名**: `SafeStateSwapperEpochGapTests`

**確認点（1つで十分）**:
> Reader が exit する前には oldState が reclaim されないこと。

テストシナリオ:
1. `BUG12` 修正済みの状態で `enterReader(0)` を呼ぶ
2. `swap()` の bump#1 と exchange の間に割り込む形で reader を配置
3. `tryReclaim()` が oldState を返さないことを確認
4. `exitReader(0)` を呼んだ後、再試行で oldState が reclaim されることを確認

### 依存関係

- BUG12（enterStateReader委譲）を**先に修正すること**
- 本修正（epoch2）は BUG12 の直後に適用すること
- BUG12 + BUG13 の組でテスト通過を確認すること
- `getSafeEpoch()` は現状「未使用」のため当面修正不要

---

## 2-B. BUG-010: retireEQStateDeferred 戻り値破棄 → カウンタ追加 ★ 実装詳細確定

**リスク**: HIGH — MKLメモリリーク（17箇所）
**状態**: ✅ **実装詳細確定** — カウンタ追加案を採用

```cpp
// 変更前（17箇所すべて）:
(void)retireEQStateDeferred(oldState);

// 変更後:
if (!retireEQStateDeferred(oldState))
{
    convo::fetchAddAtomic(m_retireDropCount, uint64_t{1}, std::memory_order_relaxed);
}
```

**要決定事項**: `m_retireDropCount` の宣言場所 → `EQProcessor.h` に `std::atomic<uint64_t> m_retireDropCount{0};` を追加する。

---

## 2-C. BUG9: EQProcessor シャドウ relaxed-only データ競合

**リスク**: MEDIUM — ARM64移植時に顕在化の可能性（x86では実質問題なし）
**状態**: 🟡 **ARM64対応時期との兼ね合いで時期未定**（修正方針は確定）

修正方針:
- NonRT書き込み: `memory_order_relaxed` → `memory_order_release`
- RT読み取り: `memory_order_relaxed` → `memory_order_acquire`

---

# 3. Appendix

---

## A. 全BUG 確定度サマリ

| # | BUG-ID | リスク | 確定度 | 設計セクション |
|---|--------|-------|--------|--------------|
| 1 | BUG12 | CRITICAL | ✅ **確定着手可能** | 1-A1 |
| 2 | BUG11 | CRITICAL | ✅ **確定着手可能** | 1-A2 |
| 3 | BUG4 | HIGH | ✅ **確定着手可能** | 1-A3 |
| 4 | BUG10 | HIGH | ✅ **確定着手可能** | 1-A4 |
| 5 | BUG-001/002/003 | HIGH/MED | 📋 **レビュー承認済み実装案** | 1-B1 |
| 6 | BUG17 | HIGH | 📋 **単位確認済み修正案** | 1-B2 |
| 7 | BUG16 | HIGH | 📋 **方向性確認済み修正案** | 1-B3 |
| 8 | BUG-009/004 | MEDIUM | 🟡 **後回し可（実装詳細確定）** | 1-C1 |
| 9 | BUG-005〜008 | LOW | 🟢 **後回し可（実装詳細確定）** | 1-C2 |
| 10 | BUG13 | HIGH | ✅ **実装詳細確定（epoch2案）** | 2-A |
| 11 | BUG-010 | HIGH | ✅ **実装詳細確定（カウンタ案）** | 2-B |
| 12 | BUG9 | MEDIUM | 🟡 **時期未定（ARM64次第）** | 2-C |

---

## B. 推奨修正順序

```
Phase 1 — 1-A（4件, 約3.5h）
  ├ A1. BUG12: ConvolverProcessor.h enterStateReader委譲
  ├ A2. BUG11: AudioEngine.h atomic<DSPCore*>化
  ├ A3. BUG4:  xRunBuffer.push戻り値チェック
  └ A4. BUG10: DeferredDeletionQueue.h int32_t化

Phase 2 — 2-A BUG13（BUG12の直後）
  └ BUG13: SafeStateSwapper epoch2退役 + SafeStateSwapperEpochGapTests

Phase 3 — 1-B（3件, 約3h）
  ├ B1. BUG-001/002/003: DSPCoreIO.cpp 一括修正
  ├ B2. BUG17: AudioEngine.Retire.cpp /1000000
  └ B3. BUG16: ISRRetire.cpp duration_cast追加

Phase 4 — 2-B BUG-010
  └ BUG-010: カウンタ追加（17箇所）

Phase 5 — 1-C（後回し）
  ├ C1. BUG-009/004: 9ファイルzeroupper追加
  └ C2. BUG-005〜008: static_cast<size_t> 24箇所

Phase 6 — 2-C（時期未定）
  └ BUG9: relaxed→release/acquire
```

---

## C. 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1 | 2026-07-26 | 初版 |
| v2 | 2026-07-26 | 採用/保留/却下の再分類 |
| v3 | 2026-07-26 | 設計/未確定事項/Appendixの3部構成 |
| v4 | 2026-07-26 | BUG16/17/11の追加検証完了 |
| v5 | 2026-07-26 | 確定度ラベル精密化 |
| v6 | 2026-07-26 | C1/C2/BUG13の実装詳細確定。全項目が「確定」または「実装詳細確定」に |
