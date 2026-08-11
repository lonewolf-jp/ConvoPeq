# -*- coding: utf-8 -*-
"""
INTEGRATED_BUG_FIX.md に 2026-08-11 別視点調査の確定結果を反映するスクリプト。

対象: 13 セクション
- 実施確定セクション: 3-1(1-1), 3-6(1-6), 3-13(2-3), 3-14(2-4), 3-19(2-9),
  3-20(2-10), 3-24(3-4), 3-25(3-5), 3-27(3-7)
- 設計判断セクション: 3-12(2-2), 3-15(2-5), 3-33(R-新規B), 3-34(R-新規C)

CRLF 改行を維持するため、バイナリモードで読み書きする。
各置換は「開始見出し行 → 終了見出し行の直前」を新しい内容で置き換える。
"""
import sys

PATH = r'c:\VSC_Project\ConvoPeq\doc\work88\big_bug\INTEGRATED_BUG_FIX.md'


# ---------------------------------------------------------------------------
# 新しいセクション内容（\n 区切り。行末 \r なしで記述する）
# ---------------------------------------------------------------------------

S_3_1 = """### 3-1. Bug 1-1 `nucHCMode` / `nucLCMode` がセッション永続化から欠落（High / P0）

**現状（Confirmed）**: `ConvolverProcessor::getState()`（StateAndUI.cpp:202-248）はテール関連を書き出すが `nucHCMode`/`nucLCMode` を欠落。`setState()`（:289-364）も読み込みなし。実行時は `pendingOverride`/`snapshot` 間で同期済み（:142-143, 194-199）、ハッシュ計算に含まれる（:55-56, 861-862）。

**別視点調査（2026-08-11）で確定**:
- **getState の読取ソースは `pendingOverride`（:194-199）**。`snapshot`（:55-56）はハッシュ計算・等価判定用の複製であり、永続化すべきは `pendingOverride` 側（UI 反映済みの最新値）。旧記載の `snapshot.nucHCMode` は誤り
- **enum 実値（OutputFilter.h:75-87）**: `HCMode{Sharp=0, Natural=1, Soft=2}` / `LCMode{Natural=0, Soft=1}`。**`LCMode::Sharp` は存在しない**（上限は `LCMode::Soft`=1）。旧記載の jlimit 上限（HCMode::Natural / LCMode::Sharp）は誤り
- **ゲッター名は `getNUCHCMode()` / `getNUCLCMode()`**（`getNucHCMode` は存在しない）。setter は `setNUCFilterModes(HCMode, LCMode)`
- **jlimit 上限の正しい値**: HCMode は `Soft`(2)、LCMode は `Soft`(1)

**修正方針（実施確定）**:
```cpp
// getState() に追加（StateAndUI.cpp:242 付近：maxCacheEntries の後、irFileLock の前）
v.setProperty("nucHCMode", static_cast<int>(pendingOverride.nucHCMode), nullptr);
v.setProperty("nucLCMode", static_cast<int>(pendingOverride.nucLCMode), nullptr);

// setState() に追加（StateAndUI.cpp:362 付近：maxCacheEntries の後、irPath の前）
if (v.hasProperty("nucHCMode") && v.hasProperty("nucLCMode")) {
    const int hcVal = static_cast<int>(v.getProperty("nucHCMode"));
    const int lcVal = static_cast<int>(v.getProperty("nucLCMode"));
    const auto hc = juce::jlimit(static_cast<int>(convo::HCMode::Sharp),
                                 static_cast<int>(convo::HCMode::Soft), hcVal);
    const auto lc = juce::jlimit(static_cast<int>(convo::LCMode::Natural),
                                 static_cast<int>(convo::LCMode::Soft), lcVal);
    setNUCFilterModes(static_cast<convo::HCMode>(hc),
                      static_cast<convo::LCMode>(lc));
}
```

**テスト**: 保存→再読込後に `getNUCHCMode()` / `getNUCLCMode()` が元値と一致。
**リスク**: Low（pure addition、ロジック変更なし）。
"""

S_3_6 = """### 3-6. Bug 1-6 IPP FFT 戻り値無視（Medium / P0）

**現状（Confirmed）**: `MklFftEvaluator.h:270-271, 425-426` で `ippsFFTFwd_RToCCS_64f()` の戻り値無視。`FFTBackend.cpp:130,146` は修正済み。`ippsFFTInit_R_64f`（:78-92）と MKL DFT はチェック済み。

**別視点調査（2026-08-11）で確定**:
- **:270-271 は `Result evaluate(...) noexcept`（:244）に属し、戻り値は `Result{}`**。:425-426 は `void computeFft(...) noexcept`（:410）に属し、出力ゼロクリア + `return;`。**両関数とも noexcept でポインタ戻りではないため `return nullptr;` は不適（旧記載の誤り）**
- 既存の `fftSpec==nullptr` ガード（:270/:425）と**同一のフォールバック**を使用するのが一貫的

**修正方針（実施確定）**:
```cpp
// :270-271 の修正（evaluate）— 既存 :256 の fftSpec==nullptr ガードと同一形式
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed (st1=" + juce::String(static_cast<int>(st1)) + ", st2=" + juce::String(static_cast<int>(st2)) + ")");
    return Result{};   // ★ nullptr ではない（evaluate の戻り値は Result）
}

// :425-426 の修正（computeFft）— 既存 :425-427 の fftSpec==nullptr ガードと同一形式
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed (st1=" + juce::String(static_cast<int>(st1)) + ", st2=" + juce::String(static_cast<int>(st2)) + ")");
    // 出力ゼロクリア（computeFft は void なので return;）
    juce::FloatVectorOperations::clear(spectrumLeft, fftSize);
    if (spectrumRight != nullptr) juce::FloatVectorOperations::clear(spectrumRight, fftSize);
    return;
}
```

**テスト**: 無効な fftSpec でエラー検出（既存ガードと同一のフォールバック動作）。
**リスク**: Low（エラーハンドリング追加）。
"""

S_3_12 = """### 3-12. Bug 2-2 入力 DC ブロッカー NaN/Inf 非対称（Low / P2）

**現状（Confirmed）**: 入力側は DC ブロッカー前にのみ `sanitizeFiniteChunk()`（DSPCoreIO.cpp:231-232, 282-283）。出力側は完全なスクラブ（ConvolverProcessor.Runtime.cpp:722）。

**別視点調査（2026-08-11）で確定**:
- **DC ブロッカー実装は `convo::UltraHighRateDCBlocker`（src/UltraHighRateDCBlocker.h、ヘッダオンリー）**。2段カスケード 1次 IIR ハイパス（`y = x − 1次LP(x)` の差分構成）、double 全経路、`alpha = 1 − exp(−ω)`（std::expm1 で小値域の桁落ち防止）
- **カットオフ実値**: input/output = **3.0 Hz**、oversampled = **1.0 Hz**（AudioEngine.h:643-651 `DCBlockerRuntimeState::init` で固定）。**「~20Hz」は実値と不一致（旧記載の誤り）**
- **「Q=0.707」は技術的に不可能な記述**（Q は2次フィルタのパラメータ。1次フィルタに Q は存在しない）。実コードにも Q パラメータなし
- フィルタ構造・係数・カットオフは**現状維持が妥当**（高速・低位相歪み、RT 安全。処理コスト: 2段 ≈20 FLOP/サンプル/CH、48kHz/2ch/1024 サンプル ≈8μs、全体の <1% 増加）
- 実バグ側面: 入力側は DC ブロッカー**後**にスクラブなし（:250-251, 303-304）

**修正方針（設計判断 → 実施確定・低リスク）**: フィルタ実装は維持し、入力側 DC ブロッカー呼び出し直後（DSPCoreIO.cpp:252 と 305 付近）にスクラブ追加。
```cpp
// DSPCoreIO.cpp:252 / :305 付近（dc.inputL / dc.inputR の process 直後）
sanitizeFiniteChunk(alignedL.get(), numSamples);
sanitizeFiniteChunk(alignedR.get(), numSamples);
```
**テスト**: NaN/Inf 入力を DC ブロッカーに通した後、出力に NaN/Inf なし。
**リスク**: Low（データ完全性向上）。
"""

S_3_13 = """### 3-13. Bug 2-3 ユニットテストの矛盾条件（Low / P2）

**現状（Confirmed）**: `EQProcessorMaxGainTests.cpp:355-358` で `for (delta < 1e-6)` と `if (delta > 1e-6)` が同時に真になり得ない（cppcheck: oppositeInnerCondition）。

**別視点調査（2026-08-11）で確定**:
- `logBound` 初期値は 0.0。ループは `delta < 1e-6` のため delta は 1e-7 まで（max=1e-7）。内側 `if (delta > 1e-6)` は**常に false = デッドコード**（cppcheck 指摘どおり）
- **内側 if 削除後**: `logBound` に log1p(1e-15)+…+log1p(1e-7) ≈ **1.11e-7** が蓄積 → `ubDb = (20/ln10)×1.11e-7 ≈ 9.6e-7`（有限・正）。テストが実質的に検証するようになる（現在は logBound=0 で空振り）
- **削除範囲は :357-358 の if 行＋代入行のみ**（:355-356 の for と { は維持、閉じ括弧を1つ上げる）
- 現行は `isfinite` のみで 0 でも通過するため、**`check(ubDb > 0.0)` の追加が必要**

**修正方針（実施確定）**:
```cpp
:354      double logBound = 0.0;
:355      for (double delta = 1e-15; delta < 1e-6; delta *= 10.0)
:356      {
:357          logBound += std::log1p(delta);   // 内側 if(:357-358)を削除
:358      }
:359      const double ubDb = kTwentyOverLog10 * logBound;
:360      check(std::isfinite(ubDb), "log1p upperBound: tiny delta finite");
:361      check(ubDb > 0.0, "log1p upperBound: tiny delta positive");   // ★追加
```
**テスト**: テストが `logBound > 0` を検証するようになる（`check(ubDb > 0.0)` 追加で保証）。
**リスク**: Low（テスト修正）。
"""

S_3_14 = """### 3-14. Bug 2-4 CacheManager strict-aliasing 違反（Low / P2）

**現状（Confirmed）**: `CacheManager.cpp:267` で `const uint8_t*` を `reinterpret_cast<const double*>` で読み替え（UB）。

**別視点調査（2026-08-11）で確定**:
- `:260` の `tdStart = dataStart + header.dataSize`（`const uint8_t*`、mmap データ）。`:264` の `tdSrc = reinterpret_cast<const double*>(tdStart)` で double スケールのポインタ演算（:268-269）と memcpy ソースに使用 → **strict-aliasing 違反**
- `raw`（:203/:241 の `uint8_t*`）は warm-up ループのローカル変数で無関係
- **要素単位 memcpy（旧記載の `double val; memcpy(&val, raw + byteOffset, ...)`）よりも、チャネル単位のバイト領域 memcpy の方が簡潔・高速**（tdSrc/idx を完全除去）

**修正方針（実施確定）**: チャネル単位バイト memcpy 方式に修正。
```cpp
// :264-272 を置換（double* の reinterpret を全廃し、バイト領域で memcpy）
            const size_t samplesPerCh = static_cast<size_t>(header.timeDomainNumSamples);
            const size_t bytesPerCh   = samplesPerCh * sizeof(double);
            for (int ch = 0; ch < static_cast<int>(header.timeDomainChannels); ++ch)
            {
                std::memcpy(tdBuffer->getWritePointer(ch),
                            tdStart + (static_cast<size_t>(ch) * bytesPerCh),  // byteOffset をバイト領域で計算
                            bytesPerCh);
            }
```
**テスト**: strict-aliasing 警告なしでビルド。
**リスク**: Low（メモリ安全性向上）。
"""

S_3_15 = """### 3-15. Bug 2-5 LockFreeRingBuffer::size() データ競合（Medium / P2）

**現状（Confirmed）**: `LockFreeRingBuffer.h:76-81` の `size()` が writeIndex/readIndex を別々に acquire 読み。SPSC で実害は稀だが負/巨大値を返す可能性。

**別視点調査（2026-08-11）で確定**:
- **SPSC 専用**（ヘッダ先頭コメント・`MpscBoundedRing.h:7` で明記）。index は `std::atomic<size_t>` = **64bit で uint32 ラップ問題なし**（x64）
- **現状は writeIndex を先に、readIndex を後に読む**（w 先読み）。この順序では `w − r` は**実占有数以下（保守側・過小評価）**。負値・Capacity 超過は 64bit 実運用上発生しない
- `.md` 旧記載の「間に Producer が writeIndex を進めると w − r が Capacity を超える」は **readIndex 先読み実装の場合の話**であり、現状コード（w 先読み）では成立しない
- 唯一のコンシューマは `AudioEngine.Timer.cpp:1445` の診断ログ近似値（`approxOcc`）で**機能的影響ゼロ**

**修正方針（設計判断 → 実施確定・防衛的）**: クランプ追加は妥当（将来の 32bit 化への保険）。`convo::consumeAtomic`（acquire）維持でオーダリング変更不要。
```cpp
size_t size() const noexcept {
    // acquire × 2: push/pop の release と HB し、一貫した（ベストエフォート）占有数を算出。
    // ★ w 先読みのため常に過小評価（安全側）。クランプは将来の 32bit 化等への防衛的保険。
    const auto w = writeIndex.load(std::memory_order_acquire);
    const auto r = readIndex.load(std::memory_order_acquire);
    return (w >= r) ? (w - r) : 0;
}
```
**テスト**: SPSC ストレステストで負にならない。
**リスク**: Low（防衛的）。
"""

S_3_19 = """### 3-19. Bug 2-9 タイミング計算 uint64 underflow（Medium / P1）

**現状（Confirmed）**: `AudioBlock.cpp:624,630,664` / `BlockDouble.cpp:589,594,626` で uint64 減算がアンダーフロー（**BlockDouble の実在行番号。旧記載 :586/591/623 は +3 ズレ**）。

**別視点調査（2026-08-11）で確定**:
- `getCurrentTimeUs()`（TimeUtils.h:14）は `std::chrono::steady_clock` ベースで `uint64_t` を返す（単調時計）
- **安全なサイト**: `callbackMs = (t1_end − t0_start)`（AudioBlock :543 / BlockDouble :515）は**同一関数内で t1_end を後から取得**するため steady_clock では逆転し得ない → **修正不要**
- **要修正の実在リスク**: `− cbPrevEndUs`（別コールバック由来の atomic 読取）と `− matchedPublishEndUs`（リング由来）は時系列順序が保証されず underflow で巨大値化の可能性
- **第4サイト追加**: `intervalMs`（AudioBlock :541 / BlockDouble :513）も `cbPrevEndUs` 減算のため underflow 対象（`cbPrevEndUs > 0` ガードのみでは防げない）

**修正方針（実施確定・全サイト saturating subtraction）**:
```cpp
// AudioBlock.cpp:624 / BlockDouble.cpp:589
const uint64_t callbackUs64 = (nowUs >= cbStartUs) ? (nowUs - cbStartUs) : 0;
const uint32_t callbackUs = static_cast<uint32_t>(std::min<uint64_t>(callbackUs64, UINT32_MAX));

// AudioBlock.cpp:630 / BlockDouble.cpp:594
const uint64_t intervalUs64 = (cbStartUs >= cbPrevEndUs) ? (cbStartUs - cbPrevEndUs) : 0;
const uint32_t intervalUs = static_cast<uint32_t>(std::min<uint64_t>(intervalUs64, UINT32_MAX));

// AudioBlock.cpp:664 / BlockDouble.cpp:626
const uint64_t observeLatencyUs =
    (observeUs >= matchedPublishEndUs) ? (observeUs - matchedPublishEndUs) : 0;

// AudioBlock.cpp:541 / BlockDouble.cpp:513（★第4サイト追加）
intervalMs = (t0_start >= cbPrevEndUs)
    ? (static_cast<double>(t0_start - cbPrevEndUs) / 1000.0) : 0.0;
```
**テスト**: タイミング逆転シナリオで巨大値なし。
**リスク**: Low（防衛的計算）。
"""

S_3_20 = """### 3-20. Bug 2-10 NoiseShaperType enum キャスト検証欠如（Medium / P1）

**現状（Confirmed）**: `AudioEngine.StateIO.cpp:90` で `setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"))` — 範囲チェックなし（enum: Psychoacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3）。

**別視点調査（2026-08-11）で確定**:
- `setNoiseShaperType` シグネチャ: `void setNoiseShaperType(NoiseShaperType type);`（AudioEngine.h:1413）
- enum 定義は `src/core/Types.h` で `NoiseShaperType{Psychoacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3}`（範囲 [0,3]）を確認済み
- `(NoiseShaperType)(int)` の C スタイルキャストは範囲チェックなし。範囲外値（4, -1 等）で UB → **現行方針（範囲チェック付き）で正しい**

**修正方針（実施確定・変更なし）**:
```cpp
const int value = static_cast<int>(state.getProperty("noiseShaperType"));
if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic) &&
    value <= static_cast<int>(NoiseShaperType::Fixed15Tap))
    setNoiseShaperType(static_cast<NoiseShaperType>(value));
```
**テスト**: 範囲外値（4, -1）でクラッシュなし。
**リスク**: Low（入力検証追加）。
"""

S_3_24 = """### 3-24. Bug 3-4 MKL アライメント判定競合（Medium / P3）

**現状（Confirmed）**: `MKLNonUniformConvolver.cpp:1571-1581` で `aligned` フラグを入口で1回計算。`dst`/`src` は 64-byte アラインなので常に true。mkl_malloc 非アラインは実運用で起こらない。

**別視点調査（2026-08-11）で確定**:
- `addFallback` ラムダ（:1567-）は :1597（addScaledFallback 内 `addFallback(n, dst, src)`）と :1611（`addFallback(toAdd, output, m_directOutBuf)`）から呼ばれる。`m_directOutBuf` は `DIAG_MKL_MALLOC(..., 64)`（:704）で 64-byte アライン。`output` は呼び出し元バッファ
- マスク `& 31` は 32-byte アライン判定で `_mm256_load_pd/store_pd` の要件と整合。load/store で同一フラグ・`i` が 4 刻み（32B）のため反復中もアライメント不変 → フラグ計算は正しい
- 「競合」の実体は実行時チェック（非アライン救済）と mkl_malloc(64) の保証付きアラインの**契約の不整合**（出力バッファは呼び出し元依存）。機能バグではなく契約の明文化が目的

**修正方針（実施確定）**: 入口でポインタ検証アサート追加（挿入位置: :1569 の `#if defined(__AVX2__)` 直後、:1570 の `int i = 0;` の前）。
```cpp
:1568  {
:1569  #if defined(__AVX2__)
         // ★ 64B アライン契約を明文化（通常動作では発火しない診断用）
         jassert((reinterpret_cast<std::uintptr_t>(dst) & 63u) == 0);
         jassert((reinterpret_cast<std::uintptr_t>(src) & 63u) == 0);
:1570      int i = 0;
```
**テスト**: 非アラインドポインタを検出。
**リスク**: Low（アサート追加）。
"""

S_3_25 = """### 3-25. Bug 3-5 CacheManager volatile sink（Medium / P3）

**現状（Confirmed）**: `CacheManager.cpp:203,241` で `volatile uint8_t sink`。MSVC ではメモリバリアにならず、C++20 で非推奨。

**別視点調査（2026-08-11）で確定**:
- `CacheManager.cpp` にも `CacheManager.h` にも **`<atomic>` が無い** → `std::atomic_signal_fence` 使用には **`#include <atomic>` の追加が必須**
- **`.md` 旧記載の「`atomic_signal_fence` のみに置換」をそのまま適用すると `raw[i]` 読み出しが消失し、ページウォームアップが機能しなくなる**（誤り）。正しくは「ダミー accumulator（`sink`）への XOR を維持 + ループ内にバリア」で volatile だけを除去

**修正方針（実施確定）**:
```cpp
// CacheManager.cpp 先頭に追加
#include <atomic>

// 両関数共通の置換パターン（:202-206 / :240-244）
// Before
    constexpr size_t kPage = 4096;
    volatile uint8_t sink = 0;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    for (size_t i = 0; i < dataSize; i += kPage)
        sink ^= raw[i];
    (void)sink;
// After（volatile 廃止・コンパイラバリア化・読み出し維持）
    constexpr size_t kPage = 4096;
    uint8_t* raw = reinterpret_cast<uint8_t*>(dst);
    uint8_t sink = 0;
    for (size_t i = 0; i < dataSize; i += kPage)
    {
        sink ^= raw[i];
        std::atomic_signal_fence(std::memory_order_seq_cst);  // ループ除去・読み出し保持を保証
    }
    (void)sink;
```
**テスト**: ページウォームアップが機能。
**リスク**: Low（最適化抑止パターン更新）。
"""

S_3_27 = """### 3-27. Bug 3-7 SpectrumAnalyzer alignas ループ内配置（Low / P3）

**現状（Confirmed）**: `SpectrumAnalyzerComponent.cpp:474` でループ内に `alignas(64) float mags[8]`。MSVC で毎回スタック 64-byte アライン。

**別視点調査（2026-08-11）で確定**:
- `_mm256_store_ps` の要件は **32-byte アライン**。`alignas(64)` は過剰（オーバーアライメント）
- ループ内宣言は毎反復でアライン確保（コンパイラがホイストする場合もあるが保証なし）。配列は毎反復 `_mm256_store_ps` で全要素書き換えるため**反復間依存なし** → ループ外へ出しても安全
- 挿入位置: **:459（vScale）と :461（for）の間**。`:474` の宣言行を削除。`alignas(32)` で十分（store_ps の 32B 要件）

**修正方針（実施確定）**: ループ外へ移動 + `alignas(32)` に変更。
```cpp
:459      const __m256 vScale = _mm256_set1_ps(FFT_MAGNITUDE_SCALE);
         alignas(32) float mags[8];            // ★ ループ外へ移動 + alignas(64)→(32)
:461      for (; i < vEnd; i += 8)
:462      {
...
            // :474 の alignas(64) float mags[8]; を削除
:475          _mm256_store_ps(mags, mag);
```
**テスト**: 出力が変わらない。
**リスク**: Low（パフォーマンス最適化）。
"""

S_3_33 = """### 3-33. R-新規B Incremental rebuild 未接続（P2）— 設計判断 → **確定（休眠維持）**

**現状（Confirmed）**: `rebuildJob` 未確保・`beginIncrementalRebuild`/`advanceIncrementalRebuild`/`resetIncrementalRebuild` は宣言のみ未定義（ConvolverProcessor.h:491-493）・`runIncrementalBuildStep`/`runIncrementalFinalizeStep`/`setUseIncrementalRebuild`/`isIncrementalRebuildEnabled` は呼び出し元ゼロ・Stage 状態機械は Prepared/FinalizingPrepare 未生成。

**別視点調査（2026-08-11）で確定**:
- `rebuildJob` は**どこからも確保されない**（`make_unique<IncrementalRebuildJob>` は 0 件）・未定義 3 関数は呼び出すとリンクエラー
- 一括 rebuild は既に非 RT スレッド（rebuildThreadLoop）で実行され、crossfade/retire 機構で切替安全性を確保済み。**Audio 処理は rebuild 中も継続するため incremental 化の実害回避効果は限定的**（rebuild 完了待ちのブロッキング短縮のみ）
- **リスク**: 設計未完のまま有効化すると R-新規A（NUC リーク）が顕在化する
- **実装コスト**: 中規模（約200〜400行）・中リスク（既存の一括 rebuild と並立するため状態遷移の整合が必須）

**修正方針（設計判断 → 確定）**:
- **Option 1 を確定（推奨）**: 休眠のまま維持し、以下を実施
  1. `reset()`（Rebuild.cpp:113-117）の `~StereoConvolver()+aligned_free` を `retireStereoConvolver(pendingConv, 0)` に変更（R-新規A、4行・低リスク・将来有効化の前提条件）
  2. 未定義 3 関数（ConvolverProcessor.h:491-493）に `jassertfalse` スタブを置く（リンクエラー要因の除去）
  3. ConvolverProcessor.h:491-493 に「休眠（将来拡張）」のコメント明記
- **Option 2（一式削除）は不採用**: 将来機能として必要になった際の再実装コストが大きいため
**リスク**: Medium（設計判断。`invalidatePendingLoads` が live 呼び出し（PrepareToPlay.cpp:287）で no-op である点に注意）。
"""

S_3_34 = """### 3-34. R-新規C bad_alloc 時リーク（P3）— 設計判断 → **確定（runSynchronously 優先）**

**現状（Confirmed）**: `rebuildAllIRsSynchronous`→`runSynchronously`→`applyNewState(async=false)`（LoaderThread.cpp:367-383 / LoadPipeline.cpp:680）で bad_alloc 時に新エンジンがリーク。例外は rebuildThreadLoop で握り潰されサイレント（OOM 時のみ）。

**別視点調査（2026-08-11）で確定**:
- **リーク経路**: ①`runSynchronously`（LoaderThread.cpp:376-377）の `make_unique<AudioBuffer>`、②`applyNewState`（LoadPipeline.cpp:687）の `make_unique<PendingCommit>` が `bad_alloc` を投げると、`conv`（`newConv` の新 StereoConvolver、生ポインタ）が誰にも所有されずリーク
- 例外は `rebuildThreadLoop` の try（RebuildDispatch.cpp:818）/catch（:1234-1242）で捕捉・握り潰し（DBG ログのみ）→ クラッシュしないが**サイレントリーク**
- **優先順位: ① > ②**（①の runSynchronously 側で防衛すれば ①② 両経路をカバーできる）

**修正方針（設計判断 → 確定）**: `runSynchronously` を try/catch で包み、失敗時に `conv` を `retireStereoConvolver` する。
```cpp
// LoaderThread.cpp:367-386（① runSynchronously）
if (result.success)
{
    auto* conv = std::exchange(result.newConv, nullptr);
    stepResult.newConv = nullptr;
    try
    {
        auto loadedIR  = std::make_unique<juce::AudioBuffer<double>>(std::move(result.loadedIR));
        auto displayIR = std::make_unique<juce::AudioBuffer<double>>(std::move(result.displayIR));
        owner.applyNewState(conv, std::move(loadedIR), result.loadedSR, result.targetLength,
                            isRebuild, file, result.scaleFactor, std::move(displayIR), /*async=*/false);
    }
    catch (...)
    {
        owner.retireStereoConvolver(conv, 0);   // ★ リーク防止（NUC エンジン解放）
        throw;                                  // rebuildThreadLoop が捕捉
    }
}

// LoadPipeline.cpp:687（② applyNewState — ①実施でカバーされるが保険として）
std::unique_ptr<PendingCommit> commit;
try { commit = std::make_unique<PendingCommit>(); }
catch (...) { retireStereoConvolver(newConv, 0); throw; }
```
**リスク**: Low（OOM 限定のため実害低。ただし将来の安定性向上に有効）。
"""


# ---------------------------------------------------------------------------
# 置換定義: (開始見出しプレフィックス, 終了見出しプレフィックス, 新しい内容)
# ---------------------------------------------------------------------------

REPLACEMENTS = [
    ("### 3-1. Bug 1-1", "### 3-2. Bug 1-2", S_3_1),
    ("### 3-6. Bug 1-6", "### 3-7. Bug 1-7", S_3_6),
    ("### 3-12. Bug 2-2", "### 3-13. Bug 2-3", S_3_12),
    ("### 3-13. Bug 2-3", "### 3-14. Bug 2-4", S_3_13),
    ("### 3-14. Bug 2-4", "### 3-15. Bug 2-5", S_3_14),
    ("### 3-15. Bug 2-5", "### 3-16. Bug 2-6", S_3_15),
    ("### 3-19. Bug 2-9", "### 3-20. Bug 2-10", S_3_19),
    ("### 3-20. Bug 2-10", "### 3-21. Bug 3-1", S_3_20),
    ("### 3-24. Bug 3-4", "### 3-25. Bug 3-5", S_3_24),
    ("### 3-25. Bug 3-5", "### 3-26. Bug 3-6", S_3_25),
    ("### 3-27. Bug 3-7", "### 3-28. Bug 3-8", S_3_27),
    ("### 3-33. R-新規B", "### 3-34. R-新規C", S_3_33),
    ("### 3-34. R-新規C", "### 3-35. R-新規D", S_3_34),
]


def build_lines(content):
    """\n 区切りの内容を、各行末に \r を付けたリスト（CRLF）に変換する。"""
    return [ln + '\r' for ln in content.split('\n')]


def main():
    with open(PATH, 'rb') as f:
        data = f.read()
    text = data.decode('utf-8')
    lines = text.split('\n')  # 各行は \r で終わる（CRLF）

    # ヘッダ（1-5行）を出力用に保持（CRLF のまま）
    header = '\n'.join(lines[:0])  # 空（不要）

    for start_prefix, end_prefix, content in REPLACEMENTS:
        start_idx = None
        end_idx = None
        for i, ln in enumerate(lines):
            stripped = ln.rstrip('\r')
            if start_idx is None and stripped.startswith(start_prefix):
                start_idx = i
            elif start_idx is not None and stripped.startswith(end_prefix):
                end_idx = i
                break
        if start_idx is None or end_idx is None:
            print(f'[FAIL] not found: {start_prefix} -> {end_prefix}')
            sys.exit(1)
        new_block = build_lines(content)
        # 置換: 開始行から終了行の直前まで
        lines[start_idx:end_idx] = new_block
        print(f'[OK] {start_prefix} .. {end_prefix} ({end_idx - start_idx} 行 -> {len(new_block)} 行)')

    out_text = '\n'.join(lines)
    with open(PATH, 'wb') as f:
        f.write(out_text.encode('utf-8'))
    print('DONE: written')


if __name__ == '__main__':
    main()
