# バグ修正改修計画書 v4 — ConvoPeq work88（設計者向け最終案）

**作成日**: 2026-07-26
**前版**: v3
**レビュー後確定**: 2026-07-26

---

## 目次

1. **設計** — 実装着手可能な修正指示
   - 1-A: 確定着手可能（CRITICAL）
   - 1-B: 確定着手可能（HIGH）
   - 1-C: 後回し可（LOW）
2. **未確定事項** — 追加設計検討・確認が必要な項目
3. **Appendix** — 参考情報（全BUG一覧・推奨順序・改訂履歴）

---

# 1. 設計 — 実装着手可能な修正指示

> このセクションに記載する修正は、**レビューを通過し確定したもののみ**。
> 各項目にはファイルパス・行番号・修正コードを含む。

---

## 1-A: 確定着手可能（CRITICAL）

### A1. BUG12: ConvolverProcessor.h enterStateReader/exitStateReader → SafeStateSwapper委譲

**リスク**: CRITICAL — RCU reader tracking無効によるUse-After-Free
**ファイル**: `src/ConvolverProcessor.h:268-269`
**修正**: 4行の委譲追加
**状態**: ✅ **確定** — レビュー承認済み

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
**状態**: ✅ **確定** — 呼び出し経路単位の確認完了

**全呼び出し経路確認結果**:

| カテゴリ | 呼び出し元 | スレッド | 影響 |
|---------|-----------|---------|------|
| 書込 | `PrepareToPlay.cpp:262` | NonRT | `setActiveRuntimeDSP()` |
| 書込 | `CtorDtor.cpp:131`, `ReleaseResources.cpp:139` | NonRT | `setActiveRuntimeDSP(nullptr)` |
| 交換 | `CtorDtor.cpp:133`, `Timer.cpp:880,1002,1554`, `ReleaseResources.cpp:142` | NonRT/Timer | `exchangeFadingRuntimeDSP()` |
| 読取 | `Latency.cpp:84` | UI Timer | `getActiveRuntimeDSP()` |
| 読取 | `Timer.cpp:969` | Timer | `getActiveRuntimeDSP()` |
| 読取 | `PrepareToPlay.cpp:270,276` | NonRT | `getActiveRuntimeDSP()` |
| 読取 | `ReleaseResources.cpp:130,136,171` | NonRT | `getActiveRuntimeDSP()` |
| 読取 | `CtorDtor.cpp:118,128,138` | 単一スレッド | `getActiveRuntimeDSP()` |

全経路で `DSPCore*` として取り扱い。sentinel値との比較（`DSPTransition.h:93`）は `reinterpret_cast<uintptr_t>(ptr) == ~0` で行われ、`std::atomic<DSPCore*>::load()` が返す `DSPCore*` でも同様に動作する（ユーザー空間アドレスが `~0` になることはない）。

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

---

### A3. BUG4: xRunBuffer ACTIVATE イベントの戻り値チェック追加

**リスク**: HIGH（CRITICALではない）— キュー満杯時にACTIVATEイベントが通知なく消失。UBではなく通知欠損
**ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:605`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:572`
**修正**: 2行のif文追加（隣接するXRUNイベントと同一パターン）
**状態**: ✅ **確定** — レビューで方向性承認済み

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

## 1-B: 確定着手可能（HIGH）

### B1. BUG-001/002/003: Float処理パス（DSPCoreIO.cpp）に不足機能を追加

**リスク**: HIGH（BUG-001: 音質劣化）/ MEDIUM（BUG-002/003: 計測欠落）
**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`
**状態**: ✅ **確定** — 現行ソースのDoubleパス処理順序と照合済み

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

**挿入箇所（DSPCoreIO.cpp 515行目直後）**:
```cpp
    // NaN/Inf Scrub #2 の AVX2 ブロック直後（line 515の `}` の後）
    _mm256_zeroupper();

    // truePeak検出（BS.1770-4/5準拠）
    truePeakDetector.processBlock(dataL, dataR, numSamples);

    // LUFSブロック平均電力（BS.1770-4/5 + EBU R128）
    loudnessMeter.processBlock(dataL, dataR, numSamples);

    // ★ [P1-1] Simple Peak Limiter（ソフトニー）
    constexpr double kPLThreshold = 0.8413951287507587;  // -1.5dBFS
    constexpr double kPLKnee = 0.108748;
    peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);

    // 以降は既存コード（applyFixedLatencyDelay → HardClamp）変更なし
    applyFixedLatencyDelay(dataL, dataR, numSamples);
    ...
```

**注意点**:
- `peakLimiter` は `AudioEngine::DSPCore` のメンバ（AudioEngine.h:960）。DSPCoreIO.cpp も同じクラスのメンバ関数を実装しているため追加宣言不要
- `truePeakDetector` / `loudnessMeter` も `AudioEngine::DSPCore` のメンバ
- テストではビット一致比較ではなくメトリクス値（RMS/THD+N/TruePeak/LUFS）の許容誤差内一致で判定すること

---

### B2. BUG17: overflowDurationMs 単位修正

**リスク**: HIGH — 慢性OVF検出が意図の1000分の1（5ms）で発動
**ファイル**: `src/audioengine/AudioEngine.Retire.cpp:136`
**修正**: 1行の定数変更
**状態**: ✅ **確定** — 翻訳単位内の根拠確認完了

**検証結果**:
```
コード内で使用されている式:
  overflowStart = retireRuntime_.overflowStartTimestamp()  // ISRRetire.cpp:71 の count() → ナノ秒
  now = steady_clock::now().time_since_epoch().count()     // ナノ秒（MSVC）
  overflowDurationMs = (now - overflowStart) / 1000        // ナノ秒/1000 = マイクロ秒

MSVC steady_clock::duration は nanoseconds である（Microsoft公式ドキュメント確定）。
「overflowDurationMs」という変数名とコメント「>5秒」は誤りで、実際の計算結果はマイクロ秒。
したがって threshold 5000 は 5000 μs = 5ms を意味し、意図（5秒）の1000分の1で発動する。
```

```cpp
// 変更前:
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;
chronicByDuration = (overflowDurationMs > 5000);  // 実際は5msで発動

// 変更後:
const uint64_t overflowDurationMs = (now - overflowStart) / 1'000'000;
chronicByDuration = (overflowDurationMs > 5000);  // 5000ms = 5秒
```

---

### B3. BUG16: ISRRetire Overflow Timestamp の単位統一

**リスク**: HIGH — overflowAgeWarnコードパスがデッドコード
**ファイル**: `src/audioengine/ISRRetire.cpp:56`（Producer）
**修正**: Producer側で `duration_cast<microseconds>` を使用
**状態**: ✅ **確定** — Producer/Consumer両方の実コード確認完了

**検証結果**:
```
Producer（ISRRetire.cpp:56）:
  RetireOverflowEntry{..., static_cast<uint64_t>(
      steady_clock::now().time_since_epoch().count()), 0};
  → ナノ秒を保存 ❌（フィールド名 overflowTimestampUs = マイクロ秒）

Consumer（Coordinator.cpp:279-284）:
  const uint64_t nowUs = duration_cast<microseconds>(
      steady_clock::now().time_since_epoch()).count();
  if (entry.overflowTimestampUs > 0 && nowUs > entry.overflowTimestampUs)
  → マイクロ秒として読み取り

ナノ秒(10^9) ≫ マイクロ秒(10^6) のため比較が起動後292年間成立せず → デッドコード
```

```cpp
// ISRRetire.cpp:56 変更前:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count()), 0};

// 変更後:
RetireOverflowEntry entry{localIntent, static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()), 0};
```

Consumer側（`ISRRuntimePublicationCoordinator.cpp:279-284`）は既に `duration_cast<microseconds>` を使用しているため変更不要。

---

### B4. BUG10: DeferredDeletionQueue 差分計算修正

**リスク**: HIGH — 32-bitカウンタラップ時にキュー誤動作
**ファイル**: `src/DeferredDeletionQueue.h:80,120,172`（3箇所）
**修正**: `intptr_t` → `int32_t` モジュラ減算
**状態**: ✅ **確定**

**確認結果**:
- `kQueueSize = 4096`（レビューでは16384とあったが実際は4096）
- 最大滞留エントリ数は理論上4096、実運用上は数十〜数百
- `4096 ≪ INT32_MAX(2.1×10⁹)` であるため、int32_t で差分を表現する前提は安全
- ただしコードコメントで「差分が kQueueSize を超えないことを前提とする」ことを明示すべき

```cpp
// 変更前（3箇所すべて）:
intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

// 変更後:
// int32_t 減算: kQueueSize(=4096) ≪ INT32_MAX により安全
int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
```

---

## 1-C: 後回し可（LOW）

### C1. BUG-009/004: `_mm256_zeroupper()` 欠落の是正（9ファイル）

**状態**: ✅ 修正方針確定済み。正当性の核心ではなく性能改善のため後回し可。
**修正**: 9ファイルの各AVX2ブロック末尾に `_mm256_zeroupper()` 追加（各1行）。

### C2. BUG-005〜008: `static_cast<size_t>` 欠落（24箇所）

**状態**: ✅ 修正方針確定済み。コーディング規約上の問題。最下位優先度。
**修正**: 24箇所の `memset/memcpy` サイズ引数に `static_cast<size_t>` 追加。

---

# 2. 未確定事項 — 追加設計検討・確認が必要な項目

> 以下の項目は**設計検討または追加確認が完了するまで実装着手不可**。

---

## 2-A. BUG13: SafeStateSwapper swap() の epoch 退避期間問題 ★ 設計再検討中

**リスク**: HIGH — BUG12修正後にUAF顕在化の可能性
**状態**: v1の「SWAP→BUMP」案は却下済み。代替案を検討中。

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
| | getState() → 旧ポインタを見る | |
| | | isOlder(N, N+1)=true → 解放→UAF |

**選択肢**:
- **案A**: `retireEpoch = epoch2`（bump#2後の値 = epoch1+1）を使用
- **案B**: 現在の設計はメモリ順序保証により安全であることを証明
- **案C**: 新たなRCU方式を設計（要議論）

**判断基準**: BUG12が修正された後に初めて影響が顕在化するため、優先度は中程度。

---

## 2-B. BUG-010: retireEQStateDeferred 戻り値破棄の扱い ★ カウンタ案で調整中

**リスク**: HIGH — MKLメモリリーク（17箇所）
**状態**: void化案は設計変更のため非推奨。カウンタ追加案を暫定推奨。

**推奨案**:
```cpp
// 変更前:
(void)retireEQStateDeferred(oldState);

// 変更後（カウンタ追加案: 設計変更不要、診断可能）:
if (!retireEQStateDeferred(oldState))
{
    convo::fetchAddAtomic(m_retireDropCount, uint64_t{1}, std::memory_order_relaxed);
}
```

**要決定事項**:
- `m_retireDropCount` の宣言場所（`EQProcessor.h`）
- リーク率の実測データ収集方針
- 長期的にはフォールバックdelete案の再評価も視野

---

## 2-C. BUG9: EQProcessor シャドウ relaxed-only データ競合

**リスク**: MEDIUM — ARM64移植時に顕在化の可能性
**状態**: 修正自体は明確だが、ARM64対応時期との兼ね合い。
**x86影響**: なし（StoreLoad順序が強いため実質問題なし）。

---

# 3. Appendix

---

## A. 全BUG 確定度サマリ

| # | BUG-ID | リスク | 確定度 | 設計セクション | レビュー評価 |
|---|--------|-------|--------|--------------|-------------|
| 1 | BUG12 | CRITICAL | ✅ **確定** | 1-A1 | 「妥当。rcuSwapperへの委譲に置き換える方針は正しい」 |
| 2 | BUG11 | CRITICAL | ✅ **確定** | 1-A2 | 「妥当。std::atomic<DSPCore*>化は理にかなっている」 |
| 3 | BUG4 | HIGH | ✅ **確定** | 1-A3 | 「方向性は正しい。運用上重要な欠落修正」 |
| 4 | BUG-001 | HIGH | ✅ **確定** | 1-B1 | 「既存のdoubleパスの処理順序をfloatへ揃えるという意味で妥当」 |
| 5 | BUG-002 | MEDIUM | ✅ **確定** | 1-B1 | （同上、BUG-001と同時修正） |
| 6 | BUG-003 | MEDIUM | ✅ **確定** | 1-B1 | （同上） |
| 7 | BUG17 | HIGH | ✅ **確定** | 1-B2 | 「修正案自体は筋が通っている」←翻訳単位の確認完了 |
| 8 | BUG16 | HIGH | ✅ **確定** | 1-B3 | 「方向は妥当」←Producer/Consumer両側の実コード確認完了 |
| 9 | BUG10 | HIGH | ✅ **確定** | 1-B4 | 「かなり有力。int32_t前提をコメントで明示すべき」 |
| 10 | BUG-009 | MEDIUM | 🟡 後回し | 1-C1 | 「低リスクな性能対策としては妥当」 |
| 11 | BUG-004 | MEDIUM | 🟡 後回し | 1-C1 | （同上） |
| 12 | BUG-005〜008 | LOW | 🟢 最下位 | 1-C2 | 「優先度としては最下位でよい」 |
| 13 | BUG13 | HIGH | ❌ **未確定** | 2-A | 「未確定事項へ落とした判断が正しい。SWAP→BUMPの単純反転は危険」 |
| 14 | BUG-010 | HIGH | ⚠️ **未確定** | 2-B | 「void化は設計変更。まずは返り値を捨ててよい経路か確認が必要」 |
| 15 | BUG9 | MEDIUM | 🟡 **未確定** | 2-C | ARM64対応時期との兼ね合い |

---

## B. 推奨修正順序

```
Phase 1 — 設計 1-A（3件, 約3h）
  1. BUG12: ConvolverProcessor.h enterStateReader委譲
  2. BUG11: AudioEngine.h atomic<DSPCore*>化
  3. BUG4:  xRunBuffer.push戻り値チェック

Phase 2 — 設計 1-B（5件, 約4h）
  4. BUG-001/002/003: DSPCoreIO.cpp 一括修正（peakLimiter/meter/detector/zeroupper）
  5. BUG17: AudioEngine.Retire.cpp /1000000
  6. BUG16: ISRRetire.cpp duration_cast追加
  7. BUG10: DeferredDeletionQueue.h int32_t化

Phase 3 — 未確定事項の設計確定（2-A, 2-B, 2-C）
  8. BUG13: swap順序の再設計 + BUG12修正との結合テスト
  9. BUG-010: カウンタ追加実装
  10. BUG9: 修正時期判断

Phase 4 — 設計 1-C（後回し）
  11. BUG-009/004: 9ファイルzeroupper追加
  12. BUG-005〜008: static_cast<size_t> 24箇所
```

---

## C. 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v1 | 2026-07-26 | 初版。全18件の修正計画 |
| v2 | 2026-07-26 | レビュー反映。採用/保留/却下の再分類。BUG13案却下。BUG-010カウンタ案 |
| v3 | 2026-07-26 | 設計/未確定事項/Appendix の3部構成。実装者向けに整理 |
| v4 | 2026-07-26 | BUG16/17のProducer/Consumer実コード確認完了。BUG11全呼び出し経路確認完了。BUG4優先度修正。確定度サマリ追加 |
