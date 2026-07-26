# バグ修正改修計画書 v5 — ConvoPeq work88（確定版）

**作成日**: 2026-07-26
**前版**: v4
**最終レビュー通過**: 2026-07-26
**状態**: ✅ 確定済み — 実装着手可能

---

## 目次

1. **設計** — 実装着手可能な修正指示
   - 1-A: 確定着手可能
   - 1-B: レビュー承認済み実装案（最終照合推奨）
   - 1-C: 後回し可
2. **未確定事項** — 追加設計検討・確認が必要な項目
3. **Appendix** — 参考情報

---

# 1. 設計 — 実装着手可能な修正指示

> **凡例**:
> - ✅ **確定着手可能**: レビュー通過。即座に実装を開始してよい。
> - 📋 **レビュー承認済み実装案**: 方向性は確定。行単位の最終照合または実動作確認を推奨。
> - 🟡 **後回し可**: 修正方針は確定。優先度低い。

---

## 1-A: ✅ 確定着手可能

> 以下の修正はレビューを通過しており、即座に実装を開始できる。

### A1. BUG12: ConvolverProcessor.h enterStateReader/exitStateReader → SafeStateSwapper委譲

**リスク**: CRITICAL — RCU reader tracking無効によるUse-After-Free
**ファイル**: `src/ConvolverProcessor.h:268-269`
**修正**: 4行の委譲追加
**状態**: ✅ **確定着手可能**

```cpp
// 変更前（空のスタブ）:
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

**注意点**: `ConvolverProcessor` が `SafeStateSwapper rcuSwapper` メンバを持つことは確認済み。

---

### A2. BUG11: AudioEngine.h activeRuntimeDSPSlot / fadingRuntimeDSPSlot → std::atomic<DSPCore*> 化

**リスク**: CRITICAL — Non-Atomicポインタのデータ競合（C++ UB）
**ファイル**: `src/audioengine/AudioEngine.h:1996-1999`, アクセサ2001-2017
**修正**: 型変更 + アクセサ置換
**状態**: ✅ **確定着手可能**（最終ビルドでセンチネル経路の再確認を推奨）

**全呼び出し経路（15箇所）**:

| カテゴリ | 呼び出し元 | スレッド |
|---------|-----------|---------|
| 書込 | `PrepareToPlay.cpp:262`, `CtorDtor.cpp:131`, `ReleaseResources.cpp:139` | NonRT |
| 交換 | `CtorDtor.cpp:133`, `Timer.cpp:880,1002,1554`, `ReleaseResources.cpp:142` | NonRT/Timer |
| 読取 | `Latency.cpp:84`, `Timer.cpp:969`, `PrepareToPlay.cpp:270,276`, `ReleaseResources.cpp:130,136,171`, `CtorDtor.cpp:118,128,138` | Timer/NonRT/UI |

全経路で `DSPCore*` としてのみ取り扱われ、sentinel値（`~0`）との比較は `reinterpret_cast<uintptr_t>(ptr) == ~0` で行われる。`std::atomic<DSPCore*>::load()` が返すポインタに対しても同様に動作する（ユーザー空間アドレスが `~0` になることはない）。

```cpp
// AudioEngine.h:1996-1999
// 変更前:
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };

// 変更後:
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
std::atomic<DSPCore*> fadingRuntimeDSPSlot{nullptr};

// アクセサ（2001-2017）
// 変更前: get() / operator= ベースの手動更新
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

**実装時の確認推奨事項**:
- `DSPTransition.h:93` の `reinterpret_cast<uintptr_t>(prevRaw) == (~static_cast<uintptr_t>(0))` が、`std::atomic` 化後も正しく sentinel を検出することをビルド＋単体テストで確認すること

---

### A3. BUG4: xRunBuffer ACTIVATE イベントの戻り値チェック追加

**リスク**: HIGH — キュー満杯時にACTIVATEイベントが通知なく消失（UBではなく通知欠損）
**ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:605`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:572`
**修正**: 2行のif文追加
**状態**: ✅ **確定着手可能**

```cpp
// 変更前:
xRunBuffer.push(ev);

// 変更後:
if (!xRunBuffer.push(ev))
{
    convo::fetchAddAtomic(rtAuxMutable_.xRunDropCount,
        uint64_t{1}, std::memory_order_relaxed);
}
```

---

### A4. BUG10: DeferredDeletionQueue 差分計算修正

**リスク**: HIGH — 32-bitカウンタラップ時にキュー誤動作
**ファイル**: `src/DeferredDeletionQueue.h:80,120,172`（3箇所）
**修正**: `intptr_t` → `int32_t` モジュラ減算
**状態**: ✅ **確定着手可能**

**確認結果**:
- `kQueueSize = 4096`（最大滞留エントリ数）
- `4096 ≪ INT32_MAX(2.1×10⁹)` であるためint32_tで差分を表現する前提は安全
- コメントで前提条件を明示すること

```cpp
// 変更前（3箇所すべて）:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 変更後:
// kQueueSize(=4096) ≪ INT32_MAX により int32_t 減算で安全。
// 差分が queue size を超えないことを前提とする。
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

---

## 1-B: 📋 レビュー承認済み実装案（最終照合推奨）

> 以下の修正は**設計としての方向性はレビューを通過している**。
> ただし、行単位の最終照合または実動作確認を経てから「確定着手可能」に昇格させることを推奨する。

### B1. BUG-001/002/003: Float処理パス（DSPCoreIO.cpp）に不足機能を追加

**リスク**: HIGH（BUG-001: 音質劣化）/ MEDIUM（BUG-002/003: 計測欠落）
**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
**状態**: 📋 **レビュー承認済み実装案** — Doubleパスの処理順序との照合完了。515行目近傍への挿入で合意。

**Doubleパス（DSPCoreDouble.cpp）の処理順序**:
```
DCBlocker → NaNScrub → Dither/Gain → truePeakDetector → loudnessMeter
→ peakLimiter → AVX Clamp → _mm256_zeroupper → FloatVectorOps::copy
```

**Floatパス（DSPCoreIO.cpp）の現在の処理順序**:
```
DCBlocker → NaNScrub#1 → pushAdaptiveCapture → ANSCoeff → Dither/Gain
→ NaNScrub#2 → applyFixedLatencyDelay → HardClamp (scalar)
```

**修正後のFloatパス処理順序**:
```
DCBlocker → NaNScrub#1 → pushAdaptiveCapture → ANSCoeff → Dither/Gain
→ NaNScrub#2 → _mm256_zeroupper → truePeakDetector → loudnessMeter
→ peakLimiter → applyFixedLatencyDelay → HardClamp (scalar, unchanged)
```

**挿入コード（NaN/Inf Scrub #2 の直後）**:
```cpp
    // NaN/Inf Scrub #2 の AVX2 ブロック直後
    _mm256_zeroupper();

    // truePeak検出（BS.1770-4/5準拠）
    truePeakDetector.processBlock(dataL, dataR, numSamples);

    // LUFSブロック平均電力（BS.1770-4/5 + EBU R128）
    loudnessMeter.processBlock(dataL, dataR, numSamples);

    // ★ [P1-1] Simple Peak Limiter（ソフトニー）
    constexpr double kPLThreshold = 0.8413951287507587;  // -1.5dBFS
    constexpr double kPLKnee = 0.108748;
    peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);

    // 以降は既存コード変更なし
    applyFixedLatencyDelay(dataL, dataR, numSamples);
    ...
```

**テスト注意点**: Float/Doubleのビット一致比較は行わない。メトリクス値（RMS/THD+N/TruePeak/LUFS）の許容誤差内一致で判定すること。

---

### B2. BUG17: overflowDurationMs 単位修正

**リスク**: HIGH — 慢性OVF検出が意図の1000分の1で発動
**ファイル**: `src/audioengine/AudioEngine.Retire.cpp:136`
**修正**: 1行の定数変更
**状態**: 📋 **単位確認済み修正案** — MSVC chrono文献＋コード解析で整合性確認済み。実機動作確認を推奨。

**検証結果**:
```
overflowStart = retireRuntime_.overflowStartTimestamp()
  → ISRRetire.cpp:71 の count()  → ナノ秒（MSVC steady_clock::duration = nanoseconds）

now = steady_clock::now().time_since_epoch().count()
  → 同上、ナノ秒

overflowDurationMs = (now - overflowStart) / 1000
  → ナノ秒/1000 = マイクロ秒。変数名はMs（ミリ秒）だが実際はマイクロ秒

>5000 → 5000 μs = 5ms（コメントは「>5秒」だが実際は5msで発動）
```

```cpp
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;
chronicByDuration = (overflowDurationMs > 5000);

// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;
chronicByDuration = (overflowDurationMs > 5000);  // 5000ms = 5秒
```

---

### B3. BUG16: ISRRetire Overflow Timestamp の単位統一

**リスク**: HIGH — overflowAgeWarnコードパスが事実上無効化
**ファイル**: `src/audioengine/ISRRetire.cpp:56`（Producer）
**状態**: 📋 **方向性確認済み修正案** — 実動作確認による確定を推奨

**検証結果**:
```
Producer（ISRRetire.cpp:56）:
  → steady_clock::now().time_since_epoch().count() でナノ秒を保存

Consumer（Coordinator.cpp:279-284）:
  → duration_cast<microseconds> でマイクロ秒として読取

ナノ秒(10^9) と マイクロ秒(10^6) の比較が成立するのは起動後約292年経過後。
overflowAgeWarnCallback_ コードパスは事実上無効化されている。
```

```cpp
// 変更前:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count()), 0};

// 変更後:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
```

Consumer側（`ISRRuntimePublicationCoordinator.cpp:279-284`）は既に `duration_cast<microseconds>` を使用しているため変更不要。

---

## 1-C: 🟡 後回し可

### C1. BUG-009/004: `_mm256_zeroupper()` 欠落の是正（9ファイル）

**状態**: 修正方針確定済み。性能改善のため後回し可。

### C2. BUG-005〜008: `static_cast<size_t>` 欠落（24箇所）

**状態**: 修正方針確定済み。コーディング規約上の問題。最下位優先度。

---

# 2. 未確定事項 — 追加設計検討・確認が必要な項目

> 以下の項目は設計検討が完了するまで実装着手不可。

---

## 2-A. BUG13: SafeStateSwapper swap() の epoch 退避期間問題 ★ 設計再検討中

**リスク**: HIGH — BUG12修正後にUAF顕在化の可能性
**状態**: v1の「SWAP→BUMP」案は却下。設計再検討中。

**現状**: `SafeStateSwapper::swap()` の現在の順序：
```
epoch1 = fetchAdd(globalEpoch)  // bump#1
fetchAdd(globalEpoch)           // bump#2（epoch2 = epoch1+1）
Exchange(activeState, newState) // swap
// retire oldState with epoch1
```

**問題のインターリーブ**:
| Writer | Reader | Reclaimer |
|--------|--------|-----------|
| bump#1 → N+1 | | |
| | enterReader → epoch=N+1 | |
| bump#2 → N+2 | | |
| swap → retire oldState(N) | | |
| | getState() → 旧ポインタ | |
| | | N < N+1 → 解放→UAF |

**選択肢**:
- **案A**: `retireEpoch = epoch2`（bump#2後の値）を使用
- **案B**: 現在の設計はメモリ順序保証により安全であることを証明
- **案C**: 新たなRCU方式を設計

**判断**: BUG12が修正された後に初めて影響が顕在化する。優先度は中程度。

---

## 2-B. BUG-010: retireEQStateDeferred 戻り値破棄の扱い ★ カウンタ案で調整中

**リスク**: HIGH — MKLメモリリーク（17箇所）
**状態**: void化案は設計変更のため非推奨。カウンタ追加案を暫定推奨。

```cpp
// 変更前:
(void)retireEQStateDeferred(oldState);

// 変更後（カウンタ追加案: 設計変更不要、診断可能）:
if (!retireEQStateDeferred(oldState))
{
    convo::fetchAddAtomic(m_retireDropCount, uint64_t{1}, std::memory_order_relaxed);
}
```

**要決定事項**: `m_retireDropCount` の宣言場所、リーク率の実測データ収集方針。

---

## 2-C. BUG9: EQProcessor シャドウ relaxed-only データ競合

**リスク**: MEDIUM — ARM64移植時に顕在化の可能性（x86では実質問題なし）

---

# 3. Appendix

---

## A. 全BUG 確定度サマリ

| # | BUG-ID | リスク | 確定度 | 設計セクション |
|---|--------|-------|--------|--------------|
| 1 | BUG12 | CRITICAL | ✅ **確定着手可能** | 1-A1 |
| 2 | BUG11 | CRITICAL | ✅ **確定着手可能**（最終ビルド確認推奨） | 1-A2 |
| 3 | BUG4 | HIGH | ✅ **確定着手可能** | 1-A3 |
| 4 | BUG10 | HIGH | ✅ **確定着手可能** | 1-A4 |
| 5 | BUG-001 | HIGH | 📋 **レビュー承認済み実装案** | 1-B1 |
| 6 | BUG-002 | MEDIUM | 📋 **レビュー承認済み実装案** | 1-B1 |
| 7 | BUG-003 | MEDIUM | 📋 **レビュー承認済み実装案** | 1-B1 |
| 8 | BUG17 | HIGH | 📋 **単位確認済み修正案** | 1-B2 |
| 9 | BUG16 | HIGH | 📋 **方向性確認済み修正案** | 1-B3 |
| 10 | BUG-009 | MEDIUM | 🟡 後回し可 | 1-C1 |
| 11 | BUG-004 | MEDIUM | 🟡 後回し可 | 1-C1 |
| 12 | BUG-005〜008 | LOW | 🟢 後回し可（最下位） | 1-C2 |
| 13 | BUG13 | HIGH | ❌ **未確定**（設計再検討中） | 2-A |
| 14 | BUG-010 | HIGH | ⚠️ **未確定**（カウンタ案調整中） | 2-B |
| 15 | BUG9 | MEDIUM | 🟡 **未確定**（ARM64次第） | 2-C |

---

## B. 推奨修正順序

```
Phase 1 — 設計 1-A（4件, 約3.5h）
  ├ A1. BUG12: ConvolverProcessor.h enterStateReader委譲          [45min]
  ├ A2. BUG11: AudioEngine.h atomic<DSPCore*>化                    [1.5h]
  ├ A3. BUG4:  xRunBuffer.push戻り値チェック                       [30min]
  └ A4. BUG10: DeferredDeletionQueue.h int32_t化                   [30min]

Phase 2 — 設計 1-B（4件, 約3.5h）
  ├ B1. BUG-001/002/003: DSPCoreIO.cpp 一括修正                   [2h]
  ├ B2. BUG17: AudioEngine.Retire.cpp /1000000                     [15min]
  └ B3. BUG16: ISRRetire.cpp duration_cast追加                    [20min]

Phase 3 — 未確定事項の設計確定
  ├ 2-A. BUG13: swap順序の再設計 + 検証
  ├ 2-B. BUG-010: カウンタ追加実装
  └ 2-C. BUG9: 修正時期判断

Phase 4 — 設計 1-C（後回し）
  ├ C1. BUG-009/004: 9ファイルzeroupper追加                       [1h]
  └ C2. BUG-005〜008: static_cast<size_t> 24箇所                   [1h]
```

---

## C. 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1 | 2026-07-26 | 初版 |
| v2 | 2026-07-26 | 採用/保留/却下の再分類 |
| v3 | 2026-07-26 | 設計/未確定事項/Appendixの3部構成 |
| v4 | 2026-07-26 | BUG16/17/11の追加検証完了 |
| v5 | 2026-07-26 | 確定度ラベルを精密化（確定着手可能/レビュー承認済み実装案/後回し可）。BUG10を1-Aへ昇格 |
