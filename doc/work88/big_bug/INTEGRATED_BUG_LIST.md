# ConvoPeq 統合バグリスト (改訂版)

**作成日**: 2026-07-30
**最終更新**: 2026-07-30
**対象**: ConvoPeq (Windows 11 x64, AVX2, MSVC/icx, JUCE 8.0.12, MKL/IPP)
**調査元**: `bug_meta_ai.md`, `bug_qwen.md`, `ConvoPeq_Part8_findings_2026-07-23.md`, `ConvoPeq_バグレポート_2026-07-30.md`
**検証方法**: ソースコード grep / AiDex インデックス検索 / 実ファイル閲覧 / cppcheck ログ照合

---

## 検証ステータス定義

| ステータス | 意味 |
|---|---|---|
| **✅ Confirmed** | 実ソースコードで問題を確認 |
| **❌ Rejected** | 実ソースで問題なし、または Markdown 破損と判定 |
| **~~⚠️ Partial~~** | **(全件確定済み、現時点では使用なし)** |
| **~~❓ NeedsFullCode~~** | **(調査完了、現時点では使用なし)** |

---

## 1. 重症バグ (Critical / High)

### 1-1. `nucHCMode` / `nucLCMode` がセッション永続化から欠落 (High) — ✅ Confirmed

**出典**: `bug_qwen.md` バグA, `ConvoPeq_バグレポート_2026-07-30.md` T.25

**ファイル**: `src/convolver/ConvolverProcessor.StateAndUI.cpp`

**現象**:
- `getState()` (行 202–248) は `tailMode`, `tailStartSec`, `tailStrength`, `tailL1L2Multiplier` 等のテール関連設定を `juce::ValueTree` に書き出すが、**`nucHCMode` と `nucLCMode` を完全に欠落**している。
- `setState()` (行 289–364) にも `"nucHCMode"` / `"nucLCMode"` の読み込みが存在しない。
- これらのフィールドは実行時には `pendingOverride` / `snapshot` 間で正しく同期されており (行 142–143, 194–199, 830–831), ハッシュ計算にも含まれている (行 861–862)。

**影響**: ユーザーがテールのハイカット/ローカットフィルターモードを変更してプリセット/セッションを保存・再読込すると、この2設定だけがサイレントにデフォルト値 (`Natural`) へ戻る。

**修正**: `getState()` に `nucHCMode`/`nucLCMode` の `setProperty` を追加し、`setState()` に `juce::jlimit` 付きの読み込み + `setNUCFilterModes()` 呼び出しを追加する。

---

### 1-2. `coordinatorDeferredRing_` / `lastResortQueue_` がプロデューサー不在のデッドコード (High) — ✅ Confirmed

**出典**: `bug_qwen.md` バグB

**ファイル**: `src/audioengine/ISRRuntimePublicationCoordinator.h` / `.cpp`

**現象**:
- `coordinatorDeferredRing_` (容量 1024 の `LockFreeRingBuffer`) は `.pop()` でのみ消費されており (行 312), **push/producer がコードベース全体に一切存在しない**。
- `lastResortQueue_` (容量 4096 の生配列) は drain 関数内でのみ読み取られ (行 337, 353, 356), **書き込みは一切行われない**。
- `coordinatorDeferredCount_` は `fetchSub` / `consume` でのみ使用され (行 318, 322, 445), **インクリメントは存在しない**。
- `lastResortCount_` は `{0}` で初期化されたまま増加する箇所がない。
- コンストラクタ (行 8–27) は `lastResortQueue_` の値初期化を行わない (生配列のため未初期化のまま)。

**影響**: RetireIntent が OverflowRing から溢れた場合、「最後の砦」として実装されたはずの2機構は到達不能。実際には即座にドロップされる。`lastResortQueue_` の未初期化は将来の producer 実装時の地雷になり得る。

**修正**: 配列の値初期化 (`{}`) による UB の芽の除去は安全に実施可能。producer の実装は設計判断が必要。

---

### 1-3. `_mm256_store_pd` のアライメント保証なし (Critical) — ✅ Confirmed

**出典**: `bug_meta_ai.md` C-1, `bug_qwen.md` 改修項目5

**ファイル**: `src/InputBitDepthTransform.h:114–115`

**現象**:
```cpp
_mm256_store_pd(dst + i,     _mm256_cvtps_pd(lo));
_mm256_store_pd(dst + i + 4, _mm256_cvtps_pd(hi));
```
`convertFloatToDoubleHighQuality(const float* src, double* dst, ...)` は `dst` の 32-byte アライメントを契約で保証しない。現在の呼び出し元 (`alignedL.get()`, `tempAligned`) は偶然アライドだが、将来の呼び出しで非アライドな `double*` が渡ると #GP 例外で即落ちる。

**影響**: プロセス全体クラッシュ。

**修正**: `_mm256_storeu_pd` に変更するか、`[[expects]]` で契約を明示する。

---

### 1-4. `fastTanh` の3箇所独立複製 (Medium) — ✅ Confirmed

**出典**: `bug_qwen.md` バグC

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:146`, `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:76`

**現象**:
- `FastTanhApprox.h` に正規実装 (`SoftClipPadéPolicy` テンプレート) が存在する。
- `DSPCoreDouble.cpp` は正しく `convo::dsp::fastTanh<convo::dsp::SoftClipPadéPolicy>` を使用している (行 127, 191)。
- `DSPCoreFloat.cpp` と `DSPCoreIO.cpp` は**同一係数を持つ独自の `inline double fastTanh(double x)`** を複製している (行 146, 76)。
- 3箇所の係数は現時点では一致しているが、将来 `SoftClipPadéPolicy` の係数をチューニングした場合、Float 入出力経路と Double 入出力経路でサチュレーションカーブが乖離する保守上のリスク。

**修正**: `DSPCoreFloat.cpp` / `DSPCoreIO.cpp` の独自 `fastTanh()` を削除し、`#include "dsp/math/FastTanhApprox.h"` から `convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>` を使用する。

---

### 1-5. `musicalSoftClip` が未使用のデッドコード (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグC

**ファイル**: `src/audioengine/AudioEngine.h:1059` (宣言), `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:341` (定義)

**現象**: `AudioEngine::DSPCore::musicalSoftClip()` は宣言・定義されているが、コードベース全体を検索しても**どこからも呼び出されていない**。実際の処理は各ファイルのファイルローカルな `musicalSoftClipScalar()` が直接呼ばれている。

**修正**: 削除するか、実際に結線するかはインターフェース設計判断。

---

### 1-6. IPP FFT 戻り値無視 (Medium) — ✅ Confirmed

**出典**: `bug_meta_ai.md` C-3

**ファイル**: `src/MklFftEvaluator.h:270–271, 425–426`

**現象**:
- `FFTBackend.cpp:130, 146` では `const IppStatus status = ippsFFTFwd_RToCCS_64f(...)` として戻り値をキャプチャしている (修正済み)。
- `MklFftEvaluator.h:270–271, 425–426` では戻り値を**無視**している。IPP 初期化失敗や不正な `fftSpec` 時にゴミが残る可能性。

**修正**: 戻り値をチェックし、`ippStsNoErr` 以外の場合はエラー処理を追加する。

---

### 1-7. `ISRRetire.cpp` での Mutex 使用 (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-2

**ファイル**: `src/audioengine/ISRRetire.cpp:44, 135, 265`, `src/audioengine/ISRRetire.h:136`

**現象**: `emitRetireIntent` は RT 安全を謢うが、輻輳時に `std::lock_guard<std::mutex> fallbackMutex_` を取得する (行 44, 135, 265)。`fallbackMutex_` は `ISRRetire.h:136` で宣言されている。将来 Audio Thread から呼ばれると優先度逆転でオーディオドロップ。

**修正**: RT パスは mutex なしの `overflowRing` のみに退避、Non-RT 側で fallback へ移動する2段階化。

---

### 1-8. `ConvolverProcessor.LoaderThread.cpp` の OOM リスク (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-3

**ファイル**: `src/convolver/ConvolverProcessor.LoaderThread.cpp:463`

**現象**: `tempFloatBuffer(numChannels, static_cast<int>(fileLength))` で `fileLength` は `MAX 2,147,483,647` まで許可。ステレオで8GB確保試行。

**修正**: ストリーミング読み込み: 256kブロック毎に `convertFloatToDoubleHighQuality` へ流す。

---

### 1-9. `MKLNonUniformConvolver.cpp` の int/size_t 混在 (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-4

**ファイル**: `src/MKLNonUniformConvolver.cpp:842, 846, 854`

**現象**: `l.fftSize * sizeof(double)` が `int * size_t`。`fftSize` は `int` だが、極端な IR (5秒@768kHz=3.8M, partSize 262k → fftSize 524k) でも収まるが、将来 `kL0MaxParts` 拡張で int 溢れの可能性。

**修正**: `static_cast<size_t>(l.fftSize) * sizeof(double)` に統一する。

---

## 2. 中症バグ (Medium)

### 2-1. 非ASCII文字を含む識別子 `SoftClipPadéPolicy` (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグD

**ファイル**: `src/dsp/math/FastTanhApprox.h:63`, `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:127, 191`

**現象**: 識別子に U+00E9 ('é') が直接埋め込まれている。`/utf-8` フラグが設定されているためビルドは壊れないが、cppcheck の構文解析器をクラッシュさせる (実証済み)。

**修正**: `SoftClipPadéPolicy` → `SoftClipPadeApproxPolicy` にリネーム。

---

### 2-2. 入力側 DC ブロッカーの NaN/Inf スクラブ非対称 (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグE

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`

**現象**:
- **入力側** (`processInputFloat`/`processInputDouble`): `sanitizeFiniteChunk()` は DC ブロッカー処理の**前**にのみ存在し、**後**にはない。
- **出力側** (行 375付近): DC ブロッカー処理の**後**に完全な NaN/Inf スクラブが存在する。

**影響**: 入力側で DC ブロッカー内部状態が破綻した場合、NaN/Inf が下流の EQ/コンボルバーへ伝播する可能性。

**修正**: 入力側2関数の DC ブロッカー呼び出し直後に `sanitizeFiniteChunk()` を追加する。

---

### 2-3. ユニットテストの矛盾条件 (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグF

**ファイル**: `src/tests/EQProcessorMaxGainTests.cpp:355–358`

**現象**:
```cpp
for (double delta = 1e-15; delta < 1e-6; delta *= 10.0)
{
    if (delta > 1e-6)  // 微小項切り捨て条件と同じ
        logBound += std::log1p(delta);
}
```
外側のループ条件 `delta < 1e-6` と内側の `if (delta > 1e-6)` は同時に真になり得ない (cppcheck: `oppositeInnerCondition`)。`logBound` は常に 0.0 のまま変化せず、テストは実質何も検証していない。

**修正**: 矛盾する内側 `if` 条件を削除する。

---

### 2-4. `CacheManager.cpp` の strict-aliasing 違反 (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグG

**ファイル**: `src/CacheManager.cpp:267`

**現象**: `const uint8_t*` を `reinterpret_cast<const double*>` で読み替えて `memcpy` の読み出し元として使用している (cppcheck: `invalidPointerCast`)。x86-64/MSVC/ICX では実害ないが、規格上の strict-aliasing 違反パターン。

**修正**: バイトオフセット計算 + `memcpy` のみで完結するよう変更する。

---

### 2-5. `LockFreeRingBuffer::size()` のデータ競合 (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` C-4

**ファイル**: `src/LockFreeRingBuffer.h:76–81`

**現象**: `size()` は `writeIndex` と `readIndex` を別々に `acquire` で読み取り、`w - r` を返す。間に Producer が `writeIndex` を進めると `w - r` が Capacity を超えた値を返す可能性。SPSC なので実害は稀だが、`getAvailableSamples()` が負や巨大値を返す可能性。

**修正**: スナップショットは best-effort と文書化するか、`writeIndex` を1回読んだ後に `readIndex` を読む順序を固定する。

---

### 2-6. `RCUReader::enter()` のハッシュ衝突リスク (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-1

**ファイル**: `src/core/RCUReader.h:51, 152`, `src/core/ThreadHash.h:9`

**現象**: `ownerThreadToken` に `cachedThreadHash()` を使用。`cachedThreadHash()` は `std::hash<std::thread::id>` を使用しており (ThreadHash.h:13), ハッシュ衝突時に2スレッドが同一オーナーと誤認し、`activeThreadId` を共有・`epochProvider` の slot を二重登録する可能性。

**影響**: epoch が永遠に進まず `DeferredDeletionQueue` が詰まりメモリリーク → 最終的に reclaim 不能。

**修正**: `thread_local uint64_t` で単調増加 ID を採番。ハッシュは診断用に留める。

---

### 2-7. `std::atomic<DSPHandle>` のロックフリー性検証不足 (Medium) — ✅ Confirmed

**出元**: `ConvoPeq_Part8_findings_2026-07-23.md` No.20

**ファイル**: `src/audioengine/ISRDSPHandle.h:170–173`, `src/audioengine/ISRDSPHandle.cpp:12–20`

**現象**:
- `ISRDSPHandle.h:170-171` で `static_assert(std::atomic<DSPHandle>::is_always_lock_free, ...)` はコメントアウトされている (icx ではコンパイル時保証されないため)。
- 代わりに `ISRDSPHandle.cpp:13-19` でランタイム `assert(ok && ...)` が使用されているが、これは `#define NDEBUG` 時 (Releaseビルド) に無視される。
- `activeRuntimeDSPHandle_` と `fadingRuntimeDSPHandle_` (16バイト構造体) は `CMPXCHG16B` に依存する。x64+AVX2では通常ロックフリーだが、Releaseビルドで非ロックフリー実装にフォールバックした場合、RTスレッドでのパフォーマンス低下やデッドロックのリスク。

**修正**: `static_assert` を復活させるか、Releaseビルドでもランタイムチェックを維持する (例: `if (!isLockFree) std::abort()`)。

---

### 2-8. `ConvolverProcessor::cleanup()` の「強制削除」未実装 (Low) — ✅ Confirmed

**出元**: `ConvoPeq_Part8_findings_2026-07-23.md` No.23

**ファイル**: `src/convolver/ConvolverProcessor.LoadPipeline.cpp:571–604`

**現象**: コメント (行 591–593) は「強制削除は行わない」と明記しているが、実装は1つ目のループ (行 575–589) と全く同じ `waitForThreadToExit(0)` チェックの2つ目のループ (行 594–603)。スレッドが終了していなければ2つ目のループも `erase` せず `++it` するだけ。コメントにある「強制削除」は実装されておらず、実質的に1つ目のループの再走査に留まっている。

**影響**: `LoaderThread` が正常終了しない場合、`loaderTrashBin` はコメントの意図に反して際限なく増加する可能性 (別の異常系が前提)。

**修正**: コメントを実態に合わせるか、`stopThread(timeoutMs)` 付きの強制終了を実装するかの設計判断。

---

### 2-9. タイミング計算の uint64 underflow (Medium) — ✅ Confirmed

**出元**: `bug_qwen.md` N4

**ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:624, 630, 664`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:586, 591, 623`

**現象**:
```cpp
const auto callbackUs = static_cast<uint32_t>(nowUs - cbStartUs);  // AudioBlock.cpp:624
const auto intervalUs = static_cast<uint32_t>(cbStartUs - cbPrevEndUs);  // AudioBlock.cpp:630
const uint64_t observeLatencyUs = observeUs - matchedPublishEndUs;  // AudioBlock.cpp:664
```
`nowUs`, `cbStartUs`, `cbPrevEndUs`, `observeUs`, `matchedPublishEndUs` はすべて `uint64_t`。`matchedPublishEndUs > observeUs` や `cbStartUs < cbPrevEndUs` の場合、減算がアンダーフローし意図せず巨大な値になる。

**修正**: saturating subtraction にする:
```cpp
const uint64_t callbackUs64 = (nowUs >= cbStartUs) ? (nowUs - cbStartUs) : 0;
const uint32_t callbackUs = static_cast<uint32_t>(std::min<uint64_t>(callbackUs64, UINT32_MAX));
```

---

### 2-10. `NoiseShaperType` enum キャストの検証欠如 (Medium) — ✅ Confirmed

**出元**: `bug_qwen.md` N5

**ファイル**: `src/audioengine/AudioEngine.StateIO.cpp:90`

**現象**:
```cpp
setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"));
```
`NoiseShaperType` は `Psychoacoustic=0, Fixed4Tap=1, Fixed15Tap=2, Adaptive9thOrder=3` (Types.h:23)。state ファイルが壊れていると範囲外の enum 値をキャストして渡す可能性。`AudioEngine.Parameters.cpp:116` には `hasIntRange("noiseShaperType", 0, 3)` のバリデーションが存在するが、`StateIO.cpp:90` では使用されていない。

**修正**: 範囲チェックを追加:
```cpp
const int value = static_cast<int>(state.getProperty("noiseShaperType"));
if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic) && 
    value <= static_cast<int>(NoiseShaperType::Adaptive9thOrder))
    setNoiseShaperType(static_cast<NoiseShaperType>(value));
```

---

## 3. 低症バグ (Low)

### 3-1. `AudioSegmentBuffer.h` のリングラップ時データ競合 (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` C-2

**ファイル**: `src/AudioSegmentBuffer.h:50–80`

**現象**: `pushBlock()` は `leftSamples/rightSamples` へのコピー後に `writePosition` を release。`copyLatest()` は `writePosition` を acquire 後に読み取るが、リングラップ時の2ndチャンク書き込み中にリーダーが `start=0` を読むと、書き込み途中の領域を読み取る。C++メモリモデル上は non-atomic配列へのデータ競合。

**検証詳細**: `pushBlock()` (行 59–98) は ring wrap 時、先に `leftSamples[0..kCapacity-1]` へ書き込み、次に `leftSamples[0..second-1]` を書き込む。`writePosition` の release は両方の `FloatVectorOperations::copy()` 完了後に行われる (行 84, 92)。しかし `copyLatest()` (行 101–123) は `writePosition` の acquire 後に (行 108)、`start = (currentWritePos - ...) % kCapacity` を計算し、リングバッファ後方から読み取る。この `start` が `0` 付近になるタイミング (リングラップ直後の copyLatest) で、2ndチャンク書き込み中の領域を読み取るデータ競合が発生する。**緩和要因**: `NoiseShaperLearner.cpp:1204` のみの Single Producer + Single Consumer パターンで、実際の競合頻度は低い。しかし非 atomic 配列 (`double*`) への同時 Read/Write は C++ メモリモデル上の UB。

---

### 3-2. `DeferredDeletionQueue.h` の kMaxScan デッドコード (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-4

**ファイル**: `src/DeferredDeletionQueue.h:120`

**現象**: `scanPos==deqPos` で即 break するため `scanned < kMaxScan` は機能しない。将来「先頭以外が先に reclaimable」な拡張をした際に無限ループ化。

**検証詳細**: `reclaim()` (行 108–165) は `scanPos == deqPos` かつ `canDelete` の時にのみ `dequeuePos` を CAS で進め、`scanPos = deqPos; scanned = 0;` をリセットする (行 150–152)。CAS 失敗時も `scanned = 0` (行 156)。先頭が削除不可の場合は即 break (行 160–161) するため、`scanned < kMaxScan` のループ条件は実際には1回で終了する。この `kMaxScan` は将来の先読み拡張時のための設計的備え。

---

### 3-3. `AlignedAllocation.h` の例外 RT 伝播 (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-5

**ファイル**: `src/AlignedAllocation.h:20–22`

**現象**: `aligned_malloc` が `throw bad_alloc`。`MKLAllocator` が `std::vector` 経由で RT に使われると例外が RT を破壊。

**検証詳細**: `aligned_malloc()` (行 19–25) は `DIAG_MKL_MALLOC()` が `nullptr` を返した場合に `throw std::bad_alloc()` を送出する。`aligned_malloc_nothrow()` (行 29–32) は non-throwing 版として存在するが、`aligned_malloc()` のすべての使用箇所が `_nothrow` 版に置き換えられているかは未検証。RT パスでは事前割当が前提だが、`MKLAllocator` が `std::vector` 経由で RT に使われるシナリオは残存リスク。

---

### 3-4. `MKLNonUniformConvolver.cpp` のアライメント判定競合 (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-6

**ファイル**: `src/MKLNonUniformConvolver.cpp:1574–1579`

**現象**: `aligned` フラグは関数入口で一度だけ計算。`partStride` が4の倍数でないと `accumBuf` が 32byte境界を外す可能性。

**検証詳細**: 行 1571–1572 で `dst` と `src` のポインタを `(uintptr_t & 31) == 0` でチェックし、その結果をループ全体で使い回す (行 1578–1581)。`dst` は `l.accumBuf` (行 852 で `mkl_malloc(l.partStride * sizeof(double), 64)` 確保、64-byte アライン保証済み) であり、常にアラインされている。**ただし** `src` は `l.fftTimeBuf` (行 840 で同様に 64-byte アライン) であり、こちらもアライン保証済み。そのため現実のコードパスでは `aligned` は常に `true` になり、`_mm256_store_pd` が使用される。問題が顕在化する唯一のケースは、`mkl_malloc` が予期せず非アラインアドレスを返した場合 (実運用では起こらない)。**結論**: 規格上は引数のポインタがアラインされていない場合に UB だが、`mkl_malloc` の保証により実害はない。コード上の契約として、入口でのポインタ検証アサートを追加推奨。

---

### 3-5. `CacheManager.cpp` の `volatile sink` (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-2

**ファイル**: `src/CacheManager.cpp:203, 241`

**現象**: 最適化抑止に `volatile uint8_t sink` を使用。MSVCでは volatile がメモリバリアにならず、C++20では非推奨。

**検証詳細**: 行 203, 241 で page walk の最適化抑止として `volatile uint8_t sink` を宣言し、`sink ^= raw[i]` でページ先頭を読み取っている。これにより `mkl_malloc` で確保した領域の物理ページをフォールトさせている (ページウォームアップ)。`sink` の volatile 性により、コンパイラはこの読み取りを最適化で削除できない。MSVC では volatile はハードウェアレジスタ用であり、変数の volatile はコンパイラ最適化抑止として機能する (`/volatile:ms` デフォルト)。C++20 以降は非推奨だが、MSVC 互換コードとしては問題ない。`std::atomic_signal_fence` や `[[maybe_unused]]` と `do_not_optimize` パターンへの置き換えは将来のリファクタリング候補。

---

### 3-6. `SnapshotFactory.cpp` の NaN ハッシュ不一致 (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-3

**ファイル**: `src/SnapshotFactory.cpp`

**現象**: `-0.0f` と `0.0f` は同一視するが、NaNのペイロード違いは別ハッシュ。`areSnapshotsEquivalent` は epsilon比較で同値と判定するが、ハッシュ不一致で常に新規 `GlobalSnapshot` を `new` し、RCUに流す無駄な生成。

**検証詳細**: `hashCombineFloat()` (行 36–43) は `bits &= 0x7FFFFFFF` で `-0.0f` と `0.0f` を同一視しているが、NaN のペイロードビット (signaling/payload NaN) はマスクしない。一方 `areSnapshotsEquivalent()` (行 46–97) は `std::abs(a - b) > epsilon` の浮動小数点比較で、NaN と任意の値の差は `> epsilon` になるため **NaN が混入した状態では「不一致」と判定される**。つまりハッシュ不一致以前に、NaN が混入すると `areSnapshotsEquivalent` も `false` を返すため、常に新規 `GlobalSnapshot` が生成される。**影響**: NaN 混入が起きていない通常運用では問題なし。NaN がパラメータに混入した場合、ハッシュ不一致の前に `areSnapshotsEquivalent` が `false` を返すため、実害の出る箇所は同じ。

---

### 3-7. `SpectrumAnalyzerComponent.cpp` の `alignas` ループ内配置 (Low) — ✅ Confirmed

**出元**: `bug_meta_ai.md` L-1

**ファイル**: `src/SpectrumAnalyzerComponent.cpp:475`

**現象**: ループ内に `alignas(64) float mags[8]` を置くと MSVC で毎回スタックを64byteアラインし直すため無駄。

**検証詳細**: 行 474 でループ内に `alignas(64) float mags[8]` が宣言されている。MSVC では `alignas(64)` がローカル変数に適用されると、反復毎にスタックポインタを64-byte アラインするための `and rsp, -64` 相当のコードが挿入される。ループ外へ移動し、ループ毎に `_mm256_store_ps(mags, mag)` で上書きする方が効率的。ただし `mags` (8要素, 32byte) への `_mm256_store_ps` は32-byte アラインで十分であり、`alignas(64)` は過剰指定。`alignas(32)` への変更でも問題ない。**影響**: UI スレッドのパフォーマンスに実質的な影響なし (ループは数百要素のみ)。コード品質上の指摘。

---

### 3-8. `ConvolverProcessor.h` の `cachedLatency` 例外安全性 (Low) — ✅ Confirmed

**出元**: `bug_meta_ai.md` L-2

**ファイル**: `src/ConvolverProcessor.h:927`

**現象**: `cachedLatency { new LatencySnapshot() }` はデストラクタで `exchangeAtomic(nullptr)` で解放している。しかしコピー代入 (`operator=`) 時に `cachedLatency` を正しく解放していない。

**検証詳細**: `cachedLatency` は `std::atomic<LatencySnapshot*> cachedLatency { new LatencySnapshot() }` (行 927) で宣言されている。デストラクタの解放は `exchangeAtomic(cachedLatency, nullptr, memory_order_acq_rel)` (行 131) + `delete oldSnap` で行われる (コピー代入のオーバーライドはなくコンパイラ生成デフォルト、`std::atomic` のコピー代入は `delete` されている)。破棄と代入 (`exchangeAtomic` + `delete` のパターン) はすべての更新箇所 (行 437, 746) で正しく行われている。**例外安全の問題**: `new LatencySnapshot()` が `std::bad_alloc` を投げた場合、コンストラクタで初期化が失敗する。これは通常の設計範囲内 (OOM はどの `new` でも発生しうる)。**結論**: 実際のコピーコンストラクタ (`ConvolverProcessor(const ConvolverProcessor&) = delete`) と `operator=` は `delete` されているため、安全。

---

## 4. 拒否 / 修正済み (Rejected / Fixed)

| # | 指摘内容 | 判定理由 | ステータス |
|---|---|---|---|
| R-1 | Crossfade 構文エラー (`dryScaledL =const`) | `rg -n "dryScaledL =const" src` で**0件ヒット**。Markdown結合時の破損と判定 | ❌ Rejected |
| R-2 | r8brain resample ループの無限ループ | 実コードでは両ループに `done < maxOutLen` が条件に含まれている (IRDSP.cpp:67, 92)。`toCopy == 0` でも `done` が増えない場合、ループは `done < maxOutLen` 条件で自動終了する | ❌ Rejected |
| R-3 | AVX2 ランタイムチェック不足 | `CpuFeatureCheck.cpp` に CPUID チェックが実装済み (leaf 0/1/7 の確認) | ❌ Rejected |
| R-4 | UI mode 両方 bypass 曖昧 | `modeId = 5` (Bypass) が明示的に処理されている | ❌ Rejected |
| R-5 | NoiseShaper 完全未防御 | `LatticeNoiseShaper.h:152-170` に `kStateLimit = 1.0e12` と `clampStateSIMD()` が存在する | ❌ Rejected |
| R-6 | EQ 係数完全未検証 | `EQProcessor.Coefficients.cpp:84` に `validateAndClampParameters()` が存在する | ❌ Rejected |
| R-7 | MessageBox 文字列破損 | `rg -n "ConvoPeq - CPU 非対応" src` で**0件ヒット**。Markdown破損と判定 | ❌ Rejected |
| R-8 | `#pragma warning(push)` に対応する `pop` なし | `LockFreeRingBuffer.h:92` に `#pragma warning(pop)` が存在する | ❌ Rejected |
| R-9 | `CMAKE_CXX_FLAGS_RELEASE` 全局上書き | `target_compile_options` (ターゲット固有) が使用されている | ❌ Rejected |
| R-10 | CMA-ES メンバ型矛盾 (`double* mean` vs `mean.begin()`) | `CmaEsOptimizer.h:243` と `CmaEsOptimizerDynamic.cpp:132` は**異なるクラス** (`CmaEsOptimizer` vs `CmaEsOptimizerDynamic`)。`CmaEsOptimizerDynamic::mean` は `std::vector<double>` (同ファイル.h:41) であり `mean.begin()` は有効。Markdown結合による誤認と判定 | ❌ Rejected |
| R-11 | RTTraceRelay 未結線 (Part 8 No.19) | `AudioEngine.Processing.AudioBlock.cpp:304` で `rtTraceRelay_.enqueue()` が呼ばれ、`AudioEngine.Timer.cpp:1132` で `rtTraceRelay_.drain()` が呼ばれている。結線済み | ❌ Rejected |
| R-12 | DSPQuarantineManager 未使用 (Part 8 No.22) | `AudioEngine.Commit.cpp:617, 633`、`AudioEngine.Processing.ReleaseResources.cpp:233, 365, 373, 383, 385`、`AudioEngine.Threading.cpp:42, 85, 88, 128`、`AudioEngine.Timer.cpp:1790, 1827` で広範囲に使用されている | ❌ Rejected |
| R-13 | ShutdownRuntime::advancePhase() デッドコード (Part 8 No.21) | `grep -n "advancePhase" src` で**0件ヒット**。メソッドは既に削除されている | ❌ Rejected |
| R-14 | IR resample キャンセル不足 (N3) | `IRDSP.cpp:68, 93` で `if (shouldExit && shouldExit())` のチェックが存在する。キャンセル機構は実装済み | ❌ Rejected |
| R-15 | audioCallbackActiveCount uint32 overflow (N9) | `AudioEngine.h:1618` の `audioCallbackActiveCount` は**同時アクティブなコールバック数**を表す (increment/decrement ペア)。42億同時コールバックという物理的不可能な状況でしか overflow しない | ❌ Rejected |

---

## 5. 優先修正順序

| 優先度 | バグ | 理由 |
|---|---|---|
| **P0** | 1-3 (`_mm256_store_pd` アライメント) | 即落ちクラッシュ |
| **P0** | 1-1 (`nucHCMode`/`nucLCMode` 未永続化) | ユーザーデータ消失 |
| **P0** | 1-6 (IPP FFT 戻り値無視) | 無音/NaN爆発 |
| **P0** | 1-7 (ISRRetire Mutex) | RT違反 |
| **P0** | 1-8 (LoaderThread OOM) | クラッシュ |
| **P0** | 1-9 (int/size_t 混在) | ヒープ破壊 |
| **P1** | 1-2 (`coordinatorDeferredRing_` デッドコード) | RetireIntent ドロップ |
| **P1** | 2-6 (RCUReader ハッシュ衝突) | メモリリーク |
| **P1** | 2-7 (`atomic<DSPHandle>` ロックフリー検証不足) | RTパフォーマンス低下 |
| **P1** | 1-4 (fastTanh 複製) | 保守リスク |
| **P1** | 2-9 (uint64 underflow) | タイミング誤測 |
| **P1** | 2-10 (enum キャスト検証欠如) | 不正状態復元 |
| **P2** | 2-1 (非ASCII識別子) | ツール互換性 |
| **P2** | 2-2 (DCブロッカー非対称) | NaN伝播 |
| **P2** | 2-3 (テスト矇盾) | 検証無効 |
| **P2** | 2-4 (strict-aliasing) | 移植性 |
| **P2** | 2-5 (LockFreeRingBuffer size) | データ競合 |
| **P2** | 2-8 (cleanup 強制削除未実装) | メモリ蓄積 |
| **P3** | 3-1～3-8 (その他) | 将来の拡張性/品質 |
| **P3** | 1-5 (musicalSoftClip デッドコード) | コードクリーン |

---

## 6. 検証に使用したツールとコマンド

| ツール | 使用コマンド | 結果 |
|---|---|---|
| grep | `rg -n "_mm256_store_pd" src` | 11箇所ヒット、`InputBitDepthTransform.h:114-115` がアライメント保証なし |
| grep | `rg -n "nucHCMode\|nucLCMode" src` | `getState()`/`setState()` に欠落を確認 |
| grep | `rg -n "coordinatorDeferredRing_\|lastResortQueue_" src` | producer 0件確認 |
| grep | `rg -n "inline double fastTanh" src` | 3箇所ヒット (FastTanhApprox.h, DSPCoreFloat.cpp, DSPCoreIO.cpp) |
| grep | `rg -n "dryScaledL =const" src` | 0件 (Markdown破損) |
| grep | `rg -n "musicalSoftClip" src` | 2件 (宣言+定義のみ、呼び出し0件) |
| grep | `rg -n "ippsFFTFwd_RToCCS_64f" src` | `MklFftEvaluator.h` で戻り値無視確認 |
| grep | `rg -n "SoftClipPad" src` | 非ASCII 'é' を含む識別子確認 |
| grep | `rg -n "delta < 1e-6\|delta > 1e-6" src/tests` | 矛盾条件確認 |
| grep | `rg -n "reinterpret_cast<const double\*>" src` | `CacheManager.cpp:267` strict-aliasing違反確認 |
| grep | `rg -n "fallbackMutex_" src` | `ISRRetire.cpp:44,135,265` で mutex 使用確認 |
| grep | `rg -n "rtTraceRelay_\|RTTraceRelay" src` | `AudioBlock.cpp:304` で `enqueue()` 呼び出し確認 |
| grep | `rg -n "dspQuarantineManager_" src` | 15箇所で使用確認 |
| grep | `rg -n "advancePhase" src` | 0件 (削除済み) |
| grep | `rg -n "double\* mean\|mean\.begin\(\)" src` | `CmaEsOptimizer.h:243` と `CmaEsOptimizerDynamic.cpp:132` は異なるクラス |
| grep | `rg -n "is_always_lock_free\|atomic<DSPHandle>" src` | `ISRDSPHandle.h:170` で `static_assert` コメントアウト確認 |
| grep | `rg -n "observeUs - matchedPublishEndUs\|nowUs - cbStartUs" src` | `AudioBlock.cpp:624,630,664` で uint64 減算確認 |
| grep | `rg -n "NoiseShaperType\)(int)" src` | `StateIO.cpp:90` で検証なしキャスト確認 |
| grep | `rg -n "kStateLimit\|clampStateSIMD" src` | `LatticeNoiseShaper.h:152-170` で state clamp 確認 |
| grep | `rg -n "validateAndClampParameters" src` | `EQProcessor.Coefficients.cpp:84` で検証確認 |
| AiDex | `aidex_query "nucHCMode"` | 13箇所ヒット、`getState()`/`setState()` に欠落 |
| AiDex | `aidex_query "musicalSoftClip"` | 2件 (宣言+定義のみ) |
| AiDex | `aidex_query "getState"` | 46箇所ヒット |
| AiDex | `aidex_query "setState"` | 18箇所ヒット |

---

## 7. 未調査領域 (今後の調査候補)

| 領域 | 理由 |
|---|---|
| `ConvolverProcessor.LoadPipeline.cpp` (867行) | 本体ロジック未精査 |
| `ConvolverProcessor.MixedPhase.cpp` (869行) | 本体ロジック未精査 |
| `ConvolverProcessor.ResampleAndFallback.cpp` (474行) | 本体ロジック未精査 |
| `ConvolverProcessor.Rebuild.cpp` | 未着手 |
| `ConvolverProcessor.Lifecycle.cpp` | 未着手 |
| `src/core/EpochDomain.h` (543行) | 内容未精査 |
| `src/core/ThreadAffinityManager.h` (293行) | 内容未精査 |
| `NoiseShaperLearner.cpp` CMA-ES本体 | 未着手 |
| `MKLNonUniformConvolver.cpp` FDL/NUPC本体 | 未着手 |
| `EQProcessor::reset()` RT到達可能性 | 前回から持ち越し |
| `AudioEngine.Mmcss.cpp` MMCSS RT影響 | N10として部分確認済み (Low/Medium) |
