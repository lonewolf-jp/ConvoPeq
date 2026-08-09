# ConvoPeq 統合バグリスト (改訂版)

**作成日**: 2026-07-30
**最終更新**: 2026-08-07 (unchecked mini-bugs 22件検証 → 21件 Fixed/Confirmed、1件は Bug 1-6 と重複。新規 R-17〜R-38 追加)
**対象**: ConvoPeq (Windows 11 x64, AVX2, MSVC/icx, JUCE 8.0.12, MKL/IPP)
**調査元**: `bug_meta_ai.md`, `bug_qwen.md`, `ConvoPeq_Part8_findings_2026-07-23.md`, `ConvoPeq_バグレポート_2026-07-30.md`, `REPAIR_PLAN3.md`
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
**追加調査**: `_mm256_store_pd` は他に `TruePeakDetector.cpp:85`, `EQProcessor.Processing.cpp:37`, `MKLNonUniformConvolver.cpp:1319, 1580` にも存在する。これらの呼び出しサイトもアライメント契約を検証する必要がある。

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
- **追加調査**: `FastTanhApprox.h:28` に `DefaultFastTanhPolicy` が存在する (27/9 の Padé近似)。`EQProcessor.Processing.cpp:104` は `fastTanh<>()` (デフォルトポリシー) を使用。DSPCoreFloat/IO は `SoftClipPadéPolicy` (10395/1260/21 係数) と一致するポリシーを使用すべき。

**修正**: `DSPCoreFloat.cpp` / `DSPCoreIO.cpp` の独自 `fastTanh()` を削除し、`#include "dsp/math/FastTanhApprox.h"` から `convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>` を使用する。

---

### 1-5. `musicalSoftClip` が未使用のデッドコード (Low) — ✅ Confirmed

**出典**: `bug_qwen.md` バグC

**ファイル**: `src/audioengine/AudioEngine.h:1066` (宣言), `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:341` (定義)

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

**ファイル**: `src/audioengine/ISRRetire.cpp:44, 135, 265`, `src/audioengine/ISRRetire.h:169`

**現象**: `emitRetireIntent` は RT 安全を謢うが、輻輳時に `std::lock_guard<std::mutex> fallbackMutex_` を取得する (行 44, 135, 265)。`fallbackMutex_` は `ISRRetire.h:169` で宣言されている。将来 Audio Thread から呼ばれると優先度逆転でオーディオドロップ。

**修正**: RT パスは mutex なしの `overflowRing` のみに退避、Non-RT 側で fallback へ移動する2段階化。

---

### 1-8. `ConvolverProcessor.LoaderThread.cpp` の OOM リスク (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-3

**ファイル**: `src/convolver/ConvolverProcessor.LoaderThread.cpp:463`

**現象**: `tempFloatBuffer(numChannels, static_cast<int>(fileLength))` で `fileLength` は `MAX 2,147,483,647` まで許可。ステレオで `2 * 2B * 4bytes ≈ 16GB` の確保試行。`MAX_FILE_LENGTH` ガード (LoaderThread.cpp:450, ResampleAndFallback.cpp:293) は整数オーバーフローを防止するがメモリ容量制限はしない。

**修正**: ストリーミング読み込み: 256kブロック毎に `convertFloatToDoubleHighQuality` へ流す。

---


### 1-9. `MKLNonUniformConvolver.cpp` の int/size_t 混在 (High) — ✅ Confirmed

**出元**: `bug_meta_ai.md` H-4

**ファイル**: `src/MKLNonUniformConvolver.cpp:843, 847, 853` (診断用: 843, 847; 実際のアロケーション: 853)

**現象**: `l.fftSize * sizeof(double)` が `int * size_t`。`fftSize` は `int` だが、極端な IR (5秒@768kHz=3.8M, partSize 262k → fftSize 524k) でも収まるが、将来 `kL0MaxParts` 拡張で int 溢れの可能性。

**修正**: `l.fftSize` を `int` から `int64_t` に変更し、`static_cast<size_t>(l.fftSize) * sizeof(double)` に統一する。`static_cast<size_t>` のみでは `fftSize` が `int` で溢れた時点で既に不正値になっているため、型そのものの変更が必要。

---

### 1-10. `m_pendingIRChange` フラグの公開前クリアによる IR 変更要求消失 (High) — ✅ Confirmed

**出元**: `bug_qwen.md` バグC3

**ファイル**: `src/audioengine/AudioEngine.Snapshot.cpp:95`

**現象**:
```cpp
const bool promoteToStructural = convo::exchangeAtomic(m_pendingIRChange, false, std::memory_order_acq_rel);
```
`m_pendingIRChange` は `AudioEngine.h:1449` の `setIRChangeFlag()` で `true` が公開される。スナップショット構築中に `exchangeAtomic(..., false, ...)` により**公開完了前**にフラグがクリアされる。`SnapshotFactory::createImpl()` が `nullptr` を返す (ハッシュ一致 + 等価判定) 場合、IR変更要求は**永久に失われる** (再試行機構なし)。呼び出し場所: `AudioEngine.Timer.cpp:770` で `consumeAtomic` 読取。

**影響**: IR変更フラグがセットされた状態でスナップショットが等価判定を通過した場合、IR変更がサイレントに無視される。

**修正**: `createImpl` の結果に応じてクリアタイミングを分岐、または `m_pendingIRChange` のクリアを公開完了後 (NonRT) に遅延する。

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

**ファイル**: `src/audioengine/ISRDSPHandle.h:184–186`, `src/audioengine/ISRDSPHandle.cpp:13–19`

**現象**:
- `ISRDSPHandle.h:186` で `static_assert(std::atomic<DSPHandle>::is_always_lock_free, ...)` は `#if !defined(_MSC_VER)` ガード下にあり、非 MSVC コンパイラでは**アクティブ**である。MSVC/icx では `#else` 分岐でコメントアウトされる。
- MSVC/icx では `#else` 分岐で `ISRDSPHandle.cpp:13-19` がランタイム `assert(ok && ...)` を使用するが、これは `#define NDEBUG` 時 (Releaseビルド) に無視される。
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
`NoiseShaperType` は `Psychoacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3` (Types.h:23)。state ファイルが壊れていると範囲外の enum 値をキャストして渡す可能性。`AudioEngine.Parameters.cpp:116` には `hasIntRange("noiseShaperType", 0, 3)` のバリデーションが存在するが、`StateIO.cpp:90` では使用されていない。

**修正**: 範囲チェックを追加:
```cpp
const int value = static_cast<int>(state.getProperty("noiseShaperType"));
if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic) && 
    value <= static_cast<int>(NoiseShaperType::Fixed15Tap))
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

### 3-6. `core/SnapshotFactory.cpp` の NaN ハッシュ不一致 (Medium) — ✅ Confirmed

**出元**: `bug_meta_ai.md` M-3

**ファイル**: `src/core/SnapshotFactory.cpp`

**現象**: `-0.0f` と `0.0f` は `hashCombineFloat()` で `bits &= 0x7FFFFFFF` により同一視されるが、NaNのペイロード違いは別ハッシュ。`areSnapshotsEquivalent()` は epsilon比較 (`std::abs(a - b) > epsilon`) を使用するが、**NaN と任意の値の差は NaN になり `NaN > epsilon` は `false` となる**ため、NaN が混入したフィールドは「不一致」を検出できず等価と判定されてしまう。ハッシュ不一致により無駄な `GlobalSnapshot` 生成が発生する可能性。

**検証詳細**: `hashCombineFloat()` (行 36–43) は `bits &= 0x7FFFFFFF` で `-0.0f` と `0.0f` を同一視しているが、NaN のペイロードビット (signaling/payload NaN) はマスクしない。一方 `areSnapshotsEquivalent()` (行 46–97) は `std::abs(a - b) > epsilon` の浮動小数点比較を使用する。**NaN と任意の値の差は NaN になり、`NaN > epsilon` は常に `false` となる**。したがって NaN が混入したフィールドは `areSnapshotsEquivalent` において**「不一致」を検出できず**、NaN を任意の値と等価と判定してしまう (逆にハッシュ不一致が先に判定される場合もある)。**影響**: ハッシュ不一致から新規 `GlobalSnapshot` が生成される (通常運用では影響なし)。ハッシュ一致 + NaN ペイロード一致 → `areSnapshotsEquivalent` が `true` を返し、別の内部状態のスナップショットを「等価」と誤判定する可能性 (極めて稀)。NaN を正しく検出するには `std::isnan()` チェックを追加すべき。

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

### 3-9. CMakeLists.txt Release ビルドの `/fp:fast` による浮動小数点精度低下 (Medium) — ✅ Confirmed

**出元**: `bug_qwen.md` バグH20

**ファイル**: `CMakeLists.txt:1143, 1219`

**現象**:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi /utf-8")
set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
```
`/fp:fast` は浮動小数点演算の再結合や特殊値(NaN/Inf)の最適化を許容する。DSPコード (特にコンボルバー、フィルタ係数計算) でこれにより数値精度が低下し、ノイズや歪みが発生する可能性。`/fp:precise` にするか、対象ターゲットのみ `/fp:fast` を適用すべき。

**修正**: `set(CMAKE_CXX_FLAGS_RELEASE ...)` から `/fp:fast` を削除し、`target_compile_options(ConvoPeq PRIVATE /fp:precise)` または特定ファイルのみ許容。


### 3-10. CMakeLists.txt Release ビルドの `/QxCORE-AVX2` による AMD CPU 非互換性 (Medium) — ✅ Confirmed

**出元**: `bug_qwen.md` バグH21

**ファイル**: `CMakeLists.txt:1219`

**現象**:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8")
```
`/QxCORE-AVX2` は **Intel CPU専用**の命令生成フラグ。AMD Ryzen (AVX2対応) でも `VPCOMPRESSD` 等の命令が異なるため、Intel CPUで生成されたバイナリが AMD で実行時にクラッシュする可能性。`ConvoPeq` ターゲット (行 1235) では `/arch:AVX2` (標準的) が既に設定されているが、グローバル `CMAKE_CXX_FLAGS_RELEASE` への `/QxCORE-AVX2` はすべてのターゲットに影響する。

**修正**: `set(CMAKE_CXX_FLAGS_RELEASE ...)` から `/QxCORE-AVX2` を削除し、Intel向け最適化はターゲット固有に適用する。

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
| R-9 | `CMAKE_CXX_FLAGS_RELEASE` 全局上書き | CMakeLists.txt:1143 と :1219 で `set(CMAKE_CXX_FLAGS_RELEASE ...)` が明示的にグローバル設定されている。`target_compile_options` (ターゲット固有) も使用されているが、グローバル上書きにより**すべてのターゲット**に `/fp:fast` や `/QxCORE-AVX2` が適用される | ✅ Confirmed |
| R-10 | CMA-ES メンバ型矛盾 (`double* mean` vs `mean.begin()`) | `CmaEsOptimizer.h:243` と `CmaEsOptimizerDynamic.cpp:132` は**異なるクラス** (`CmaEsOptimizer` vs `CmaEsOptimizerDynamic`)。`CmaEsOptimizerDynamic::mean` は `std::vector<double>` (同ファイル.h:41) であり `mean.begin()` は有効。Markdown結合による誤認と判定 | ❌ Rejected |
| R-11 | RTTraceRelay 未結線 (Part 8 No.19) | `AudioEngine.Processing.AudioBlock.cpp:304` で `rtTraceRelay_.enqueue()` が呼ばれ、`AudioEngine.Timer.cpp:1132` で `rtTraceRelay_.drain()` が呼ばれている。結線済み | ❌ Rejected |
| R-12 | DSPQuarantineManager 未使用 (Part 8 No.22) | `AudioEngine.Commit.cpp:617, 633`、`AudioEngine.Processing.ReleaseResources.cpp:233, 365, 373, 383, 385`、`AudioEngine.Threading.cpp:42, 85, 88, 128`、`AudioEngine.Timer.cpp:1790, 1827` で広範囲に使用されている | ❌ Rejected |
| R-13 | ShutdownRuntime::advancePhase() デッドコード (Part 8 No.21) | `grep -n "advancePhase" src` で**0件ヒット**。メソッドは既に削除されている | ❌ Rejected |
| R-14 | IR resample キャンセル不足 (N3) | `IRDSP.cpp:68, 93` で `if (shouldExit && shouldExit())` のチェックが存在する。キャンセル機構は実装済み | ❌ Rejected |
| R-15 | audioCallbackActiveCount uint32 overflow (N9) | `AudioEngine.h:1618` の `audioCallbackActiveCount` は**同時アクティブなコールバック数**を表す (increment/decrement ペア)。42億同時コールバックという物理的不可能な状況でしか overflow しない | ❌ Rejected |
| R-16 | `/fp:fast` Release ビルド浮動小数点不正確性 | CMakeLists.txt:1143,1219 で `/fp:fast` が `CMAKE_CXX_FLAGS_RELEASE` に含まれている。DSPコードの数値精度低下リスク。 | ✅ Confirmed |
| R-17 | CustomInputOversampler::processDown パススルー時のバッファオーバーリード (BUG-039) | CustomInputOversampler.cpp:834-841 で `targetSamples` 分だけ `upsampledBlock` から読み取る。`upsampledBlock` のサイズが小さい場合にメモリ超過読み取り。→ **修正確認**: 840-841 で `std::min(targetSamples, upsampledBlock.getNumSamples())` を使用 | ✅ Fixed |
| R-18 | CmaEsOptimizer::sanitize が NaN/Infinity を処理しない (BUG-016) | CmaEsOptimizer.h:201-204, CmaEsOptimizerDynamic.h:50 で `sanitize(x)` が NaN/Inf を通過させる。→ **修正確認**: 両者とも `(!std::isfinite(x) || std::abs(x) < 1e-15) ? 0.0 : x` に修正済み (CmaEsOptimizer.h:208, CmaEsOptimizerDynamic.h:50) | ✅ Fixed |
| R-19 | enqueueWithRetry 戻り値無視による QueuePressure ドロップ (BUG-015) | ISRRetireRouter.cpp:148-155, SnapshotCoordinator.cpp:33-38,86-89 で `enqueueWithRetry` の戻り値 `(void)` キャスト。→ **修正確認**: 全サイトで `const auto result = enqueueWithRetry(...)` で受け取りチェック済み | ✅ Fixed |
| R-20 | DSPTransition Emergency Override が exchangeFadingRuntimeDSP をスキップ (BUG-029) | DSPTransition.h:54-74 で Emergency パスが `exchangeFadingRuntimeDSP` を呼ばず `oldDSP` が dangling 化。→ **修正確認**: DSPTransition.h:63-65 で `auto* prevRaw = engine_.exchangeFadingRuntimeDSP(oldDSP)` 呼び出し追加、prevRaw の retire 処理追加 (lines 68-71) | ✅ Fixed |
| R-21 | juce::String currentDeviceTypeName_ の CoW データ競合 (BUG-014) | AudioEngine.h:2278-2279 で Message Thread 書込 / Audio Thread 読込が CoW で UAF。→ **修正確認**: `juce::String currentDeviceTypeName_` 削除、setter で `MmcssPolicy` enum に変換して `std::atomic<MmcssPolicy>` に publish (AudioEngine.h:2354-2366, static_assert で lock-free 保証) | ✅ Fixed |
| R-22 | SpectrumAnalyzerComponent +6 dB 過大表示 (BUG-038) | SpectrumAnalyzerComponent.h:74 で `FFT_MAGNITUDE_SCALE = 4.0f/N` が複素FFT で +6dB 過大。→ **修正確認**: 既に `2.0f / NUM_FFT_POINTS` が正しい値で使用されている (Bug レポート自体が古いバージョン基準) | ✅ Fixed |
| R-23 | finalizeNUCEngineOnMessageThread で irL/irR リーク (BUG-036) | LoadPipeline.cpp:616-618 で `release()` を `init()` 前に評価し失敗時にリーク。→ **修正確認**: LoadPipeline.cpp:640-648 で `.get()` で取得し、`init()` 成功時にのみ `.release()` するパターンに修正済み | ✅ Fixed |
| R-24 | applyComputedIR() 世代不一致で isLoading 固着 (BUG-035) | LoadPipeline.cpp:329-334 で早期 return 時に isLoading を false に戻さず。→ **修正確認**: `ApplyComputedIRLoadingGuard` RAII クラス (LoadPipeline.cpp:325-338) で関数スコープ全体で isLoading_ を管理 | ✅ Fixed |
| R-25 | TruePeakDetector int オーバーフロー (BUG-019) | TruePeakDetector.cpp:96-111 で `numSamples * 2/4` を int に格納。→ **修正確認**: TruePeakDetector.cpp:102-103 で `static_cast<size_t>(numSamples) * 2/4` に修正済み | ✅ Fixed |
| R-26 | 浮動小数点 `!= 1.0` 完全一致比較 (BUG-018) | LoadPipeline.cpp:347, DSPCoreDouble.cpp:440, MKLNonUniformConvolver.cpp:1048 で FP の `!= 1.0` 比較。→ **修正確認**: 全 3 サイトで `!= 1.0` パターンが消滅済み (grep で 0 hits) | ✅ Fixed |
| R-27 | timerCallback が RCU reader guard なしで engine アクセス (BUG-021) | Lifecycle.cpp:150-169 で `timerCallback()` が `enterGlobalReader` なし。→ **修正確認**: Lifecycle.cpp:144-151 で GlobalGuard パターンが追加済み | ✅ Fixed |
| R-28 | prepareToPlay が RCU reader なしで engine アクセス (BUG-022) | Lifecycle.cpp:228-274 で `prepareToPlay()` が RCU 保護なし。→ **修正確認**: Lifecycle.cpp:211-217 で GlobalGuard パターンが追加済み | ✅ Fixed |
| R-29 | updateAudioThreadSnapshotFade スタブ関数 (BUG-031) | AudioEngine.h:3696-3706 でハードコードされたスタブ。→ **修正確認**: AudioEngine.h:3880 で「★ [DELETED] 2026-07-28: updateAudioThreadSnapshotFade は Dead Code のため削除」とコメント、関数削除済み | ✅ Fixed |
| R-30 | BlockDouble クロスフェードが dryScale 未適用 (BUG-033) | BlockDouble.cpp:400-427 で `useDryAsOld=true` 時に dryScale を適用せず。→ **修正確認**: BlockDouble.cpp:420-427 で `const double dryScale = useDryAsOld ? crossfadeRuntime_.getDryScaleGain().getNextValue() : 1.0;` 追加済み、コメント「★ BUG-033/C-1: float版と同様に useDryAsOld 時に dryScale を適用」明記 | ✅ Fixed |
| R-31 | CmaEsOptimizer Rule of Five 違反 (BUG-042) | CmaEsOptimizer.h で生ポインタ所有 + コピー/ムーブ制御なし。→ **修正確認**: CmaEsOptimizer.h:43-46 で `= delete` 宣言 4 種追加済み (copy ctor, copy assign, move ctor, move assign) | ✅ Fixed |
| R-32 | IRConverter::convertFile resample failure サンプルレート誤ラベル (BUG-045) | IRConverter.cpp:258-281 で `converted.getNumSamples() <= 0` 時に `actualSampleRate = config.targetSampleRate` 誤代入。→ **修正確認**: IRConverter.cpp:269-270 で `converted = ir; actualSampleRate = sourceRate;` (失敗時は sourceRate を維持)、コメント「Previously this mislabeled as targetSampleRate, ...」明記 | ✅ Fixed |
| R-33 | CrossfadeRuntime::complete が stale flags を残す (BUG-028) | CrossfadeRuntime.h で dryScaleTarget_/startDelayBlocks_/dryHoldSamples_ フラグの不整合。→ **修正確認**: CrossfadeRuntime.h:106-110 コメント「★ BUG-028 fix (work88): dryScaleTarget_/startDelayBlocks_/dryHoldSamples_ の ... dryScaleGain_ が stale target を保持し得た (五次レビュー §8 指摘)」、107, 134, 138 で `publishAtomic(dryScaleTarget_, 1.0, ...)` 設定 | ✅ Fixed |
| R-34 | NoiseShaperLearner VLA (可変長配列) スタック破壊 (BUG-041) | NoiseShaperLearner.cpp:643 で `alignas(64) double tanhBuffer[totalCoeffs] = {}` (VLA)。→ **修正確認**: grep で `alignas(64)\s+double\s+tanhBuffer` パターン 0 hits。std::vector または heap 確保に置換済みと推定 | ✅ Fixed |
| R-35 | IPP FFT 戻り値無視 (MKLNonUniformConvolver.cpp 7サイト) (BUG-034) | MKLNonUniformConvolver.cpp:1043, 1060, 1376, 1436, 1570, 1637, 1750 で IPP FFT 戻り値無視。→ **確認結果**: 当該ファイルには `ippsFFTFwd_RToCCS_64f` 呼び出しなし (コメントのみ)、実態は `MklFftEvaluator.h:270-271, 425-426` (Bug 1-6 と同一) | ⚠️ Bug 1-6 参照 |
| R-36 | SnapshotFadeState advance() vs resetToIdle() 競合による counter 不整合 (BUG-024) | SnapshotFadeState.h:41-67 で advance() の残量書き込みが resetToIdle() のゼロクリアと競合。→ **修正確認**: line 67-73 で「★ state 再確認（resetToIdle 競合対策）」と「★ ABA generation 再確認」を追加、`fadeGeneration_` による generation check で競合を検出 | ✅ Fixed |
| R-37 | loaderTrashBin 内スレッドが ConvolverProcessor 破棄後に dangling reference (BUG-037) | LoadPipeline.cpp:51-55, 551-579 で activeLoader の thread が owner dangling を保持。→ **修正確認**: StateAndUI.cpp:977-987 で `forceCleanup()` が `loadersToDelete.swap(loaderTrashBin)` でローカル変数に移し、各スレッドに `stopThread(500)` を呼んでから破棄するパターンに修正 | ✅ Fixed |
| R-38 | PsychoacousticDither Rule of Five 違反 (BUG-046) | PsychoacousticDither.h:55-587 で生 owning ポインタ + ユーザー宣言 dtor + move 未宣言。→ **修正確認**: line 98-105 で dtor + 4 種すべて `= delete` 明示 (copy ctor, copy assign, move ctor, move assign) | ✅ Fixed |

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
| **P1** | 1-10 (m_pendingIRChange early clear) | IR 変更要求消失 |
| **P1** | 2-9 (uint64 underflow) | タイミング誤測 |
| **P1** | 2-10 (enum キャスト検証欠如) | 不正状態復元 |
| **P2** | 2-1 (非ASCII識別子) | ツール互換性 |
| **P2** | 2-2 (DCブロッカー非対称) | NaN伝播 |
| **P2** | 2-3 (テスト矇盾) | 検証無効 |
| **P2** | 2-4 (strict-aliasing) | 移植性 |
| **P2** | 2-5 (LockFreeRingBuffer size) | データ競合 |
| **P2** | 2-8 (cleanup 強制削除未実装) | メモリ蓄積 |
| **P3** | 3-1～3-10 (その他) | 将来の拡張性/品質 |
| **P3** | R-9 (CMAKE_CXX_FLAGS_RELEASE グローバル上書き) | /fp:fast + /QxCORE-AVX2 が全ターゲットに適用 |
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
| grep | `rg -n "is_always_lock_free\|atomic<DSPHandle>" src` | `ISRDSPHandle.h:186` で `#if !defined(_MSC_VER)` ガード下 static_assert 確認 (MSVC では `#else` でコメントアウト) |
| grep | `rg -n "m_pendingIRChange" src/audioengine/AudioEngine.Snapshot.cpp` | `Snapshot.cpp:95` で `exchangeAtomic(..., false, ...)` 確認 — 公開前クリア |
| grep | `rg -n "fp:fast" CMakeLists.txt` | `CMakeLists.txt:1143,1219` で `/fp:fast` 確認 |
| grep | `rg -n "QxCORE-AVX2" CMakeLists.txt` | `CMakeLists.txt:1219` で Intel専用フラグ確認 |
| grep | `rg -n "enum class NoiseShaperType" src/core/Types.h` | `Types.h:25-30` で `Adaptive9thOrder=2, Fixed15Tap=3` 確認 |
| grep | `rg -n "set(CMAKE_CXX_FLAGS_RELEASE" CMakeLists.txt` | 2箇所でグローバルフラグ上書き確認 (R-9) |
| grep | `rg -n "observeUs - matchedPublishEndUs\|nowUs - cbStartUs" src` | `AudioBlock.cpp:624,630,664` で uint64 減算確認 |
| grep | `rg -n "NoiseShaperType\)(int)" src` | `StateIO.cpp:90` で検証なしキャスト確認 |
| grep | `rg -n "kStateLimit\|clampStateSIMD" src` | `LatticeNoiseShaper.h:152-170` で state clamp 確認 |
| grep | `rg -n "validateAndClampParameters" src` | `EQProcessor.Coefficients.cpp:84` で検証確認 |
| AiDex | `aidex_query "nucHCMode"` | 13箇所ヒット、`getState()`/`setState()` に欠落 |
| AiDex | `aidex_query "musicalSoftClip"` | 2件 (宣言+定義のみ) |
| AiDex | `aidex_query "getState"` | 46箇所ヒット |
| AiDex | `aidex_query "setState"` | 18箇所ヒット |

---


### 1-8. `ConvolverProcessor.LoaderThread.cpp` の OOM リスク

**Root Cause**: `tempFloatBuffer(numChannels, static_cast<int>(fileLength))` (LoaderThread.cpp:463) で `fileLength` は `MAX_FILE_LENGTH = 2147483647` (INT32_MAX) まで許可。ステレオ float で最大約 16GB の確保を試みる。`MAX_FILE_LENGTH` ガード (同ファイル:450, ResampleAndFallback.cpp:293) は整数オーバーフローを防止するがメモリ容量制限はしない。

**Fix Approach**: ストリーミング読み込み — 256kサンプルブロック毎に `convertFloatToDoubleHighQuality` へ流す。
```cpp
constexpr int64_t STREAMING_CHUNK = 256 * 1024;
for (int64_t offset = 0; offset < fileLength; offset += STREAMING_CHUNK) {
    const int64_t chunk = std::min(STREAMING_CHUNK, fileLength - offset);
}
```

**Testing**: 2GB を超える巨大 IR ファイルでメモリ使用量が一定範囲内に収まることを確認。

**Risk**: High — 読み込みロジックの大幅変更。

---

### 1-9. `MKLNonUniformConvolver.cpp` の int/size_t 混在

**Root Cause**: `l.fftSize * sizeof(double)` が `int * size_t` (MKLNonUniformConvolver.cpp:843, 847, 853, 855)。C++ の昇格規則により `int` が `size_t` に変換されるが、将来的に `fftSize` が `INT_MAX` を超える場合、`int` 自体のオーバーフロー。

**Fix Approach**: `fftSize` を `int64_t` に変更し、`static_cast<size_t>()` を明示。

**Testing**: 5秒@768kHz (fftSize 524288) でメモリ割り当てサイズが正しいことを確認。

**Risk**: Low — 型の昇格による安全性向上。

---

### 1-10. `m_pendingIRChange` フラグの公開前クリア

**Root Cause**: `AudioEngine.Snapshot.cpp:95` で `exchangeAtomic(m_pendingIRChange, false, ...)` がスナップショット構築開始時にフラグをクリア。`SnapshotFactory::createImpl()` が `nullptr` を返す場合、IR変更要求は永久に失われる。

**Fix Approach**: クリアタイミングを公開完了後 (NonRT) に遅延。
```cpp
const bool pendingIrChange = convo::consumeAtomic(m_pendingIRChange, std::memory_order_acquire);
// publish 成功後:
if (publishSuccess) convo::publishAtomic(m_pendingIRChange, false, std::memory_order_release);
```

**Testing**: IR変更後スナップショットが等価判定を通過した場合でも IR変更が再試行されることを確認。

**Risk**: Medium — フラグライフサイクルの変更。

---

### 2-1. 非ASCII文字を含む識別子 `SoftClipPadéPolicy`

**Root Cause**: `FastTanhApprox.h:63` の `struct SoftClipPadéPolicy` は U+00E9 ('é') を含む。cppcheck の構文解析器をクラッシュさせる。

**Fix Approach**: `SoftClipPadéPolicy` → `SoftClipPadeApproxPolicy` にリネーム。

**Testing**: cppcheck が正常に解析できることを確認。

**Risk**: Low — リネームのみ。

---

### 2-2. 入力側 DC ブロッカーの NaN/Inf スクラブ非対称

**Root Cause**: 入力側は DC ブロッカー処理の前にのみ `sanitizeFiniteChunk()` が存在する (DSPCoreIO.cpp:231-232, 282-283)。後にはない。出力側は DC ブロッカー処理後に完全なスクラブが存在する (ConvolverProcessor.Runtime.cpp:722)。

**Fix Approach**: 入力側 DC ブロッカー呼び出し直後に `sanitizeFiniteChunk()` を追加。

**Testing**: NaN/Inf 入力を DC ブロッカーに通した後、出力に NaN/Inf が残らないことを確認。

**Risk**: Low — データ完全性向上。

---

### 2-3. ユニットテストの矛盾条件

**Root Cause**: `EQProcessorMaxGainTests.cpp:355-358` で `for (delta < 1e-6)` と `if (delta > 1e-6)` は同時に真になり得ない (cppcheck: `oppositeInnerCondition`)。`logBound` は常に 0.0。

**Fix Approach**: 内側の `if (delta > 1e-6)` 条件を削除。

**Testing**: テストが `logBound > 0` を検証するようになることを確認。

**Risk**: Low — テスト修正。

---

### 2-4. `CacheManager.cpp` の strict-aliasing 違反

**Root Cause**: `CacheManager.cpp:267` で `const uint8_t*` を `reinterpret_cast<const double*>` で読み替え。規格上の UB。

**Fix Approach**: バイトオフセット計算 + `memcpy` のみ。
```cpp
double val;
std::memcpy(&val, raw + byteOffset, sizeof(double));
```

**Testing**: strict-aliasing 警告なしでビルドできることを確認。

**Risk**: Low — メモリ安全性向上。

---

### 2-5. `LockFreeRingBuffer::size()` のデータ競合

**Root Cause**: `LockFreeRingBuffer.h:76-81` の `size()` は `writeIndex` と `readIndex` を別々に `acquire` で読み取り。SPSC なので実害は稀だが、`getAvailableSamples()` が負や巨大値を返す可能性。

**Fix Approach**: 読み取り順序を固定し、負の結果をクランプ。
```cpp
size_t size() const noexcept {
    const auto w = writeIndex.load(std::memory_order_acquire);
    const auto r = readIndex.load(std::memory_order_acquire);
    return (w >= r) ? (w - r) : 0;
}
```

**Testing**: SPSC ストレステストで `size()` が負にならないことを確認。

**Risk**: Low — 防衛的プログラミング。

---

### 2-6. `RCUReader::enter()` のハッシュ衝突リスク

**Root Cause**: `RCUReader.h:51, 152` と `ThreadHash.h:9` で `cachedThreadHash()` (=`std::hash<std::thread::id>`) を使用。ハッシュ衝突時に2スレッドが同一オーナーと誤認。

**Fix Approach**: `thread_local uint64_t` で単調増加 ID を採番。
```cpp
thread_local uint64_t threadInstanceId = []() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}();
```

**Testing**: 100+ スレッドでハッシュ衝突がないことを確認。

**Risk**: Low — thread_local カウンタの追加。

---

### 2-7. `std::atomic<DSPHandle>` のロックフリー性検証不足

**Root Cause**: `ISRDSPHandle.h:186` で `static_assert` は `#if !defined(_MSC_VER)` ガード下で非 MSVC コンパイラではアクティブ。MSVC/icx では `#else` 分岐でコメントアウトされ、`ISRDSPHandle.cpp:13-19` でランタイム `assert` を使用するが、Releaseビルド (NDEBUG) では無視される。

**Fix Approach**: `if (!isLockFree) std::abort()` パターンで Release ビルドでも検証を維持。
```cpp
if (!ok) std::abort();
```

**Testing**: Releaseビルドで非ロックフリー環境で abort することを確認。

**Risk**: Low — abort パターンの導入。

---

### 2-8. `ConvolverProcessor::cleanup()` の「強制削除」未実装

**Root Cause**: `ConvolverProcessor.LoadPipeline.cpp:571-604` の `cleanup()` は2つのループを持つが、両方とも `waitForThreadToExit(0)` で即座にチェック。`forceCleanup()` (StateAndUI.cpp:965) は `stopThread(500)` を呼ぶが `cleanup()` は呼ばれない。

**Fix Approach**: `cleanup()` にタイムアウト付き強制終了を追加。
```cpp
(*it)->stopThread(timeoutMs);
if ((*it)->waitForThreadToExit(0)) { it = loaderTrashBin.erase(it); }
```

**Testing**: スレッドが終了しない場合に `stopThread` が呼ばれることを確認。

**Risk**: Medium — スレッド終了ロジックの変更。

---

### 2-9. タイミング計算の uint64 underflow

**Root Cause**: `AudioEngine.Processing.AudioBlock.cpp:624,630,664` と `BlockDouble.cpp:586,591,623` で `uint64_t` 減算がアンダーフロー。`nowUs - cbStartUs` が負の結果を `uint64_t` として巨大値に。

**Fix Approach**: saturating subtraction。
```cpp
const uint64_t callbackUs64 = (nowUs >= cbStartUs) ? (nowUs - cbStartUs) : 0;
const uint32_t callbackUs = static_cast<uint32_t>(std::min<uint64_t>(callbackUs64, UINT32_MAX));
```

**Testing**: タイミング逆転シナリオで巨大値が出力されないことを確認。

**Risk**: Low — 防衛的計算。

---

### 2-10. `NoiseShaperType` enum キャストの検証欠如

**Root Cause**: `AudioEngine.StateIO.cpp:90` で `setNoiseShaperType((NoiseShaperType)(int)state.getProperty("noiseShaperType"))` — 範囲チェックなし。enum は `Psychacoustic=0, Fixed4Tap=1, Adaptive9thOrder=2, Fixed15Tap=3` (Types.h:25-30)。

**Fix Approach**: 範囲チェックを追加 (upper bound は `Fixed15Tap` = 3)。
```cpp
const int value = static_cast<int>(state.getProperty("noiseShaperType"));
if (value >= static_cast<int>(NoiseShaperType::Psychoacoustic) &&
    value <= static_cast<int>(NoiseShaperType::Fixed15Tap))
    setNoiseShaperType(static_cast<NoiseShaperType>(value));
```

**Testing**: 範囲外値 (4, -1) でクラッシュしないことを確認。

**Risk**: Low — 入力検証追加。

---

### 3-1. `AudioSegmentBuffer.h` のリングラップ時データ竅突

**Root Cause**: `AudioSegmentBuffer.h:50-123` の `pushBlock()` は ring wrap 時に2回の `FloatVectorOperations::copy()` を行う。`copyLatest()` が ring wrap 直後に 2ndチャンク書き込み中の領域を読み取る可能性 (non-atomic `double*`)。SPSC で緩和されている。

**Fix Approach**: ダブルバッファリングまたはバージョンカウンタで整合性保証。

**Testing**: TSan で ring wrap 時にデータ競合がないことを確認。

**Risk**: Medium — データ構造の変更。

---

### 3-2. `DeferredDeletionQueue.h` の kMaxScan デッドコード

**Root Cause**: `DeferredDeletionQueue.h:120` の `reclaim()` は `scanPos == deqPos` かん `canDelete` の時のみ進み、先頭が削除不可の場合即 `break`。`scanned < kMaxScan` は実質1回で終了。将来の先読み拡張の備え。

**Fix Approach**: 現状維持。コメントを明確にするか、将来の拡張に備えて kMaxScan を文書化。

**Testing**: 既存テスト通過を確認 (変更なし)。

**Risk**: N/A — デッドコードだが安全。

---

### 3-3. `AlignedAllocation.h` の例外 RT 伝播

**Root Cause**: `AlignedAllocation.h:19-25` の `aligned_malloc` が `throw std::bad_alloc()`。`aligned_malloc_nothrow()` (line 29-32) が存在するが、RT パスでの使用保証が不十分。

**Fix Approach**: RT パスで `aligned_malloc_nothrow` を使用するか、事前割当の徹底。

**Testing**: RT パスで bad_alloc が RT に伝播しないことを確認。

**Risk**: Medium — メモリ管理の統一。

---

### 3-4. `MKLNonUniformConvolver.cpp` のアライメント判定競合

**Root Cause**: `MKLNonUniformConvolver.cpp:1571-1581` では `aligned` フラグを関数入口で1回計算。`dst` は `l.accumBuf` (64-byte アライン), `src` は `l.fftTimeBuf` (64-byte アライン) なので `aligned` は常に `true`。`mkl_malloc` が非アラインを返すケースは実運用では起こらない。

**Fix Approach**: 入口でのポインタ検証アサートを追加。

**Testing**: 非アラインドポインタを検出できることを確認。

**Risk**: Low — アサート追加のみ。

---

### 3-5. `CacheManager.cpp` の `volatile sink`

**Root Cause**: `CacheManager.cpp:203, 241` で `volatile uint8_t sink` を使用。MSVC では volatile がメモリバリアにならない。C++20 では非推奨。

**Fix Approach**: `std::atomic_signal_fence(std::memory_order_seq_cst)` への置き換え。

**Testing**: ページウォームアップが機能することを確認。

**Risk**: Low — 最適化抑止パターンの更新。

---

### 3-6. `core/SnapshotFactory.cpp` の NaN ハッシュ不一致

**Root Cause**: `hashCombineFloat()` (行 36-43) は `bits &= 0x7FFFFFFF` で `-0.0f` と `0.0f` を同一視するが、NaN のペイロードビットはマスクしない。`areSnapshotsEquivalent()` (行 46-97) は `std::abs(a - b) > epsilon` 比較を使用する。**NaN と任意の値の差は NaN になり、`NaN > epsilon` は `false` となる**。NaN が混入したフィールドは `areSnapshotsEquivalent` において「不一致」を検出できず、NaN を任意の値と等価と判定してしまう。

**Fix Approach**: `areSnapshotsEquivalent` に `std::isnan()` チェックを追加。
```cpp
if (std::isnan(params.saturationAmount) || std::isnan(snapshot.saturationAmount))
    return false;
```

**Testing**: NaN 入力で `areSnapshotsEquivalent` が `false` を返すことを確認。

**Risk**: Low — NaN ハンドリング追加。

---

### 3-7. `SpectrumAnalyzerComponent.cpp` の `alignas` ループ内配置

**Root Cause**: `SpectrumAnalyzerComponent.cpp:474` でループ内に `alignas(64) float mags[8]` が宣言されている。MSVC では毎回スタックを64-byte アラインする。`_mm256_store_ps` は 32-byte アラインで十分。

**Fix Approach**: ループ外へ移動し、`alignas(32)` に変更。

**Testing**: 出力が変わらないことを確認。

**Risk**: Low — パフォーマンス最適化。

---

### 3-8. `ConvolverProcessor.h` の `cachedLatency` 例外安全性

**Root Cause**: `ConvolverProcessor.h:927` の `cachedLatency` は `std::atomic<LatencySnapshot*>`。コピーコンストラクタと `operator=` は `delete` されている (h:927 近辺)。`std::atomic` のコピー代入は `delete` されたため安全。`new LatencySnapshot()` が `std::bad_alloc` を投げる場合は OOM。

**Fix Approach**: `aligned_make_unique` 使用を検討。現状維持も可。

**Testing**: 既存テスト通過を確認 (変更なし)。

**Risk**: N/A — 安全確認済み。

---

### 3-9. CMakeLists.txt の `/fp:fast`

**Root Cause**: `CMakeLists.txt:1143, 1219` で `set(CMAKE_CXX_FLAGS_RELEASE ...)` に `/fp:fast` が含まれている。浮動小数点再結合と NaN/Inf 最適化により DSP 数値精度が低下。

**Fix Approach**: `CMAKE_CXX_FLAGS_RELEASE` から `/fp:fast` を削除し、`target_compile_options(ConvoPeq PRIVATE /fp:precise)` を追加。

**Testing**: `/fp:precise` ビルドで既存テストの数値結果が変わらないことを確認。

**Risk**: Low — コンパイラフラグ変更。

---

### 3-10. CMakeLists.txt の `/QxCORE-AVX2`

**Root Cause**: `CMakeLists.txt:1219` で `set(CMAKE_CXX_FLAGS_RELEASE ...)` に `/QxCORE-AVX2` が含まれている。Intel CPU専用。AMD Ryzen でクラッシュ。`ConvoPeq` ターゲット (line 1235) では `/arch:AVX2` が設定済み。

**Fix Approach**: `CMAKE_CXX_FLAGS_RELEASE` から `/QxCORE-AVX2` を削除。Intel向け最適化はターゲット固有に適用。
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG /Gy /Zi /utf-8")
```

**Testing**: AMD CPU でビルドしたバイナリが実行できることを確認。

**Risk**: Low — フラグ分離。

---

### R-9. `CMAKE_CXX_FLAGS_RELEASE` 全局上書き

**Root Cause**: `CMakeLists.txt:1143` と `:1219` で `set(CMAKE_CXX_FLAGS_RELEASE ...)` が明示的にグローバル設定されている。これにより**すべてのターゲット**に `/fp:fast`, `/QxCORE-AVX2`, `/O2` or `/O3` が適用される。`target_compile_options(ConvoPeq PRIVATE /arch:AVX2)` も使用されているが、グローバル上書きによりターゲット固有の設定が部分的に無効化される可能性。

**Fix Approach**: `CMAKE_CXX_FLAGS_RELEASE` をデフォルトに戻し (または最小限に)、すべての最適化フラグを `target_compile_options` でターゲット固有に指定。
```cmake
# Remove: set(CMAKE_CXX_FLAGS_RELEASE "...")
# Use per-target:
target_compile_options(ConvoPeq PRIVATE /O2 /Ob2 /DNDEBUG /arch:AVX2 /Gy /Zi)
```

**Testing**: すべてのターゲットが期待通りのフラグでビルドされることを確認。

**Risk**: Medium — ビルド設定の大幅変更。

---


---

## 7. 詳細修正設計 (Detailed Fix Design)

以下は各バグに対する詳細設計。Root Cause分析、修正アプローチ、コードパターン、テスト方針、リスク評価を含む。

### 1-1. `nucHCMode` / `nucLCMode` がセッション永続化から欠落

**Root Cause**: `ConvolverProcessor::getState()` (StateAndUI.cpp:202) は `juce::ValueTree` にテイル関連プロパティを `setProperty()` で書き出すが、`nucHCMode`/`nucLCMode` を追加していない。`setState()` (行 289) も同様。これらのフィールドは実行時には `pendingOverride`/`snapshot` 間で同期されており (行 142-143, 194-199, 830-831)、ハッシュ計算にも含まれている (行 861-862)。

**Fix Approach**:
```cpp
// getState() に追加 (StateAndUI.cpp:242 付近)
v.setProperty("nucHCMode", static_cast<int>(snapshot.nucHCMode), nullptr);
v.setProperty("nucLCMode", static_cast<int>(snapshot.nucLCMode), nullptr);

// setState() に追加 (StateAndUI.cpp:362 付近)
if (v.hasProperty("nucHCMode") && v.hasProperty("nucLCMode")) {
    const int hcVal = static_cast<int>(v.getProperty("nucHCMode"));
    const int lcVal = static_cast<int>(v.getProperty("nucLCMode"));
    const auto hc = juce::jlimit(static_cast<int>(convo::HCMode::Sharp),
                                 static_cast<int>(convo::HCMode::Natural), hcVal);
    const auto lc = juce::jlimit(static_cast<int>(convo::LCMode::Natural),
                                 static_cast<int>(convo::LCMode::Sharp), lcVal);
    setNUCFilterModes(static_cast<convo::HCMode>(hc),
                      static_cast<convo::LCMode>(lc));
}
```

**Testing**: 保存→再読込後に `getNucHCMode()` / `getNucLCMode()` が元値と一致することを確認。

**Risk**: Low — pure addition, no logic change.

---

### 1-2. `coordinatorDeferredRing_` / `lastResortQueue_` がプロデューサー不在のデッドコード

**Root Cause**: `coordinatorDeferredRing_` (容量 1024 の LockFreeRingBuffer) は `.pop()` でのみ消費される (ISRRuntimePublicationCoordinator.cpp:330)。push/producer がコードベース全体に**存在しない**。`lastResortQueue_` (容量 4096 の生配列) は drain 関数内でのみ read/compaction が行われる (行 355, 371, 374)。new entry の追加は**存在しない**。`coordinatorDeferredCount_` は `fetchSub`/`consume` でのみ使用される (行 336, 340, 463)。`lastResortCount_` は `{0}` 初期化後増加しない。コンストラクタは `lastResortQueue_` の値初期化を行わない (生配列のため未初期化のまま)。

**Fix Approach**:
1. `lastResortQueue_` の値初期化: `RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity]{};`
2. producer 実装の検討: `emitRetireIntent` overflow 時に `lastResortQueue_` への enqueue を追加
3. デッドコード削除: producer を実装しない場合、`coordinatorDeferredRing_` と `lastResortQueue_` を削除

**Testing**:
- TSan で `coordinatorDeferredRing_` / `lastResortQueue_` への同時アクセスなしを確認
- drain 関数で `lastResortCount_ == 0` の場合即 return を確認

**Risk**: Medium — producer 追加は複雑な同期ロジックが必要。

---

### 1-3. `_mm256_store_pd` のアライメント保証なし

**Root Cause**: `convertFloatToDoubleHighQuality()` (InputBitDepthTransform.h:108) は `double* dst` の 32-byte アライメントを保証しない。`_mm256_store_pd` は 32-byte アライメントが必要。現在の呼び出し元は偶然アライドだが将来の変更で非アライドになる可能性。

**Additional sites**: `_mm256_store_pd` は他に `TruePeakDetector.cpp:85`, `EQProcessor.Processing.cpp:37`, `MKLNonUniformConvolver.cpp:1319, 1580` にも存在する。Bug 3-4 で MKLNonUniformConvolver のアライメントチェックがあるが、他の3サイトは未検証。

**Fix Approach**:
```cpp
// Option A: アンアライドストア (最も安全)
_mm256_storeu_pd(dst + i, _mm256_cvtps_pd(lo));
_mm256_storeu_pd(dst + i + 4, _mm256_cvtps_pd(hi));

// Option B: 契約の明示 (C++20 preconditions)
[[expects: reinterpret_cast<uintptr_t>(dst) % 32 == 0]]
```

**Testing**: 非アライドバッファを渡して #GP が発生しないことを確認 (AddressSanitizer + 非アライド割り当て)。

**Risk**: Low — `_mm256_storeu_pd` はわずかに遅いが安全。

---

### 1-4. `fastTanh` の3箇所独立複製

**Root Cause**: `FastTanhApprox.h:101-107` にテンプレート版 `fastTanh<Policy>` が存在する (default `DefaultFastTanhPolicy`)。`SoftClipPadéPolicy` (line 63) は 10395/1260/21 係数。`DSPCoreDouble.cpp:127,191` は `SoftClipPadéPolicy` を使用。`DSPCoreFloat.cpp:146` と `DSPCoreIO.cpp:76` は独自の `inline double fastTanh(double x)` を複製。`EQProcessor.Processing.cpp:104` は `fastTanh<>()` (デフォルトポリシー) を使用。

**Fix Approach**:
1. `FastTanhApprox.h:63` の `SoftClipPadéPolicy` を `SoftClipPadeApproxPolicy` にリネーム
2. `DSPCoreFloat.cpp` / `DSPCoreIO.cpp` の独自 `fastTanh` を削除
3. `#include "dsp/math/FastTanhApprox.h"` を追加
4. 呼び出しを `convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>(...)` に変更

```cpp
// DSPCoreFloat.cpp:186 / DSPCoreIO.cpp:116 — Before:
const double clipped = threshold + knee * fastTanh((abs_x - threshold) / knee);
// After:
const double clipped = threshold + knee * convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>((abs_x - threshold) / knee);
```

**Testing**: Float パスと Double パスのサチュレーション出力が一致することを確認 (最大許容差 1e-12)。

**Risk**: Medium — 係数の不一致により既存のサウンドが変化する可能性。

---

### 1-5. `musicalSoftClip` が未使用のデッドコード

**Root Cause**: `AudioEngine::DSPCore::musicalSoftClip()` (h:1066, DSPCoreIO.cpp:341-343) は宣言・定義されているが、コードベース全体で**呼び出されない**。実際の処理はファイルローカルな `musicalSoftClipScalar()` が使用されている (DSPCoreFloat.cpp:165,186, DSPCoreDouble.cpp:107,117, DSPCoreIO.cpp:95,116)。

**Fix Approach**: クラスメソッド `musicalSoftClip()` を削除するか、実際に呼び出し側に結線する。DSPCoreDouble.cpp:217 では `musicalSoftClipScalar` が直接呼ばれており、クラスメソッド版は冗長。

**Testing**: クラスメソッド削除後、ビルド成功 + 既存テスト通過を確認。

**Risk**: Low — デッドコード削除。

---

### 1-6. IPP FFT 戻り値無視

**Root Cause**: `MklFftEvaluator.h:270-271, 425-426` で `ippsFFTFwd_RToCCS_64f()` の戻り値 (`IppStatus`) を無視。IPP 初期化失敗や不正な `fftSpec` 時に無効なデータが残る。`FFTBackend.cpp:130,146` は既に戻り値をキャプチャ済み (修正済み)。`MklFftEvaluator.h:78-92` の `ippsFFTInit_R_64f` は戻り値をチェック済み `[Bug 2 fix]` コメント付き。MKL DFT 関数 (`DftiComputeForward` 等) は全て `!= DFTI_NO_ERROR` チェック済み。

**Fix Approach**:
```cpp
// Line 270-271 Before:
ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
// After:
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    DBG("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed (st1=" + juce::String(static_cast<int>(st1)) + ", st2=" + juce::String(static_cast<int>(st2)) + ")");
    return nullptr;
}
```

**Testing**: 無効な fftSpec でエラーが正しく検出されることを確認。

**Risk**: Low — エラーハンドリング追加。

---

### 1-7. `ISRRetire.cpp` での Mutex 使用

**Root Cause**: `emitRetireIntentRT()` (ISRRetire.cpp:94) は RT から呼ばれる可能性があるが、内部で `emitRetireIntent()` (line 102) を呼び出し、`fallbackMutex_` (h:169) を取得する (line 44, 135, 265)。RT スレッドでの mutex 取得は優先度逆転でオーディオドロップを引き起こす。コードコメント (line 97) は「実装は emitRetireIntent() を素通しし、輻輳時に std::mutex をロックする」と自認している。

**Fix Approach**:
1. RT パス (`emitRetireIntentRT`) は `overflowRing_` のみに退避 (mutex なしロックフリー)
2. Non-RT パス (`emitRetireIntent`) は `overflowRing_` から drain し、必要時のみ `fallbackMutex_` を取得
3. `emitRetireIntentRT` が `fallbackMutex_` にアクセスしないことを保証

```cpp
// emitRetireIntentRT — RT path, no mutex
void LifetimeState::emitRetireIntentRT(const RetireIntent& intent) noexcept {
    if (overflowRing_ != nullptr && overflowRing_->tryPush(encodeIntent(intent))) {
        return;
    }
    overflowDroppedCount_.fetch_add(1, std::memory_order_relaxed);
    // RT パス: mutex 取得を絶対回避
}
```

**Testing**: RT スレッドから `emitRetireIntentRT` を連続呼出しし、mutex が取得されないことを確認 (TSan)。

**Risk**: High — RT パスの設計変更。

---


## 8. 未調査領域 (今後の調査候補)

| 領域 | 理由 |
|---|---|
| `ConvolverProcessor.LoadPipeline.cpp` (867行) | **部分調査済み**: Bug 1-8 で `cleanup()`/`waitForThreadToExit` を確認。Bug 1-6 で `DftiCreateDescriptor` チェック済み |
| `ConvolverProcessor.MixedPhase.cpp` (869行) | **部分調査済み**: Bug 1-6 で `DftiCreateDescriptor`/`DftiComputeForward` のエラードリブを確認 (line 180, 278, 761, 811, 812) |
| `ConvolverProcessor.ResampleAndFallback.cpp` (474行) | **部分調査済み**: Bug 1-8 で `MAX_FILE_LENGTH` ガード確認 (line 293)。Bug 1-6 で `DftiComputeForward` チェック済み (line 395, 433) |
| `ConvolverProcessor.Rebuild.cpp` | 未着手 |
| `ConvolverProcessor.Lifecycle.cpp` | **調査済み**: Bug 1-1 で `nucHCMode`/`nucLCMode` 使用箇所確認 (line 265-266) |
| `src/core/EpochDomain.h` (543行) | 未着手 |
| `src/core/ThreadAffinityManager.h` (293行) | 未着手 |
| `NoiseShaperLearner.cpp` CMA-ES本体 | **部分調査済み**: Bug R-10 で `CmaEsOptimizer` vs `CmaEsOptimizerDynamic` のクラス分離を確認 |
| `MKLNonUniformConvolver.cpp` FDL/NUPC本体 | **部分調査済み**: Bug 1-9 で `fftSize * sizeof(double)` 検証 (line 843, 847)。Bug 3-4 でアライメントチェック確認 (line 1571-1581)。`_mm256_store_pd` at line 1319, 1580 も確認 |
| `EqProcessor::reset()` RT到達可能性 | 前回から持ち越し |
| `AudioEngine.Mmcss.cpp` MMCSS RT影響 | **調査済み**: N10 として部分確認 (thread_local HANDLE, no destructor, single init per thread) |
| `SnapshotCoordinator.h` | **調査済み**: REPAIR_PLAN3.md BUG-015 (SafeStateSwapper パターン) |
| `SRSnapshotFadeState.h` | **調査済み**: REPAIR_PLAN3.md BUG-024 |
| `ObservedRuntime.h` | **調査済み**: REPAIR_PLAN3.md BUG-026 |
