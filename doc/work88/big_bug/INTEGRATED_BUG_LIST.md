# ConvoPeq 統合バグリスト (改訂版)

**作成日**: 2026-07-30
**最終更新**: 2026-08-11（§10 別視点調査追加: 未調査領域の確定・Rebuild.cpp 別視点調査（R-新規A〜E）・各バグの現状再確認・CMake フラグ前回変更反映）
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

---

## 9. 六次レビュー追加（2026-08-09）— REPAIR_PLAN2-dash の残課題反映

### 9-1. 🔴 `submitRecoveryRequest` push 失敗の INV-5 違反 — ✅ 修正済み

**ファイル**: `src/audioengine/ISRRuntimePublicationCoordinator.cpp:643-669`
**状態**: ✅ 修正済み（2026-08-08）
**詳細**: `recoveryIntentQueue_`（SPSC, 256）が full のとき push 失敗を無視 → Recovery drop + `pendingIntentCount_` 不整合（shutdown ハング）。push 戻り値チェック + `recoveryIntentDropCount_` 記録 + backpressure 統合で修正。**詳細改修設計は REPAIR_PLAN2-dash.md に反映済み（修正済みのため dash からは削除済み）**。

### 9-2. 🟡 per-type admission policy の統一機構未実装 — **P3 に降格（レビュー指摘）**

**ファイル**: `ISRRuntimePublicationCoordinator.h` / `.cpp`（`submitObserve` / `submitQuarantine` / `submitRecoveryRequest` / `enqueuePublicationIntent`）
**状態**: ⚠️ 機能的には計画書を満たすが統一機構なし → **P3 アーキテクチャ整理（バグではない）**
**詳細改修設計**: REPAIR_PLAN2-dash.md 1.1 — `IntentAdmissionPolicy` ヘルパー導入。**★ レビュー指摘（2026-08-09）**: 「統一」は表層的（`actionFor` は決定のみ共通化。副作用 = ownerChannel 回収 / Critical 昇格 / drop カウンタは type 固有のまま）。既存バグを修正しないため **P2 対象外**。
**★ 実装照合の注意**: `Recovery` の overflow は `recoveryIntentQueue_` に **fallback ring がない**ため `DropWithCounter`（drop カウンタ + Critical 昇格）が正しい。`FallbackRing` は `Quarantine` のみ。
**対応 Phase**: P3（Phase 5）。

### 9-3. 🟡 MpscBoundedRing producer hole と cross-type FIFO — **PopStatus 案は撤回 + test seam 必要（二次レビュー）**

**ファイル**: `src/MpscBoundedRing.h` / `ISRRuntimePublicationCoordinator_ProcessIntent.cpp`
**状態**: ⚠️ 許容範囲（遅延 1ms 未満）。**★ 一次レビュー（2026-08-09）**: Empty と producer hole は **sequence protocol では区別不能**（`sequences_[i]=i` 初期値と、producer が CAS 予約後未 publication の状態で `diff` が同一値になる）。`PopStatus` の導入は**設計不成立**のため**撤回**。
**详细改修設計**: REPAIR_PLAN2-dash.md 1.2 — **`PopStatus` を追加せず、現行 `bool pop()` 契約（false = empty または unpublished reservation）を明文化**。cross-type FIFO は「FIFO-preserving backpressure / head-of-line blocking」（順序逆転なし）。
**🔴 二次レビュー（2026-08-09）— test seam 必須**: 現行 `push()` は CAS → payload → publication を**一つの呼び出しに内包**しているため、「CAS 成功後・publication 前に別スレッドを停止する」テストポイントが公開 API にない。テスト実装には **CONVO_TESTING フック**（`#ifdef CONVO_TESTING` で有効化、本番ビルドでは消滅）が必要。テスト4本: ①予約→遅延→pop false→publish→pop true ②A 予約・B 先 publish で B を消費しない ③空キュー連続 pop false ④payload publication ordering（memory-order invariant）。
**対応 Phase**: Phase 6（テスト追加 + CONVO_TESTING フックのみ・本番コード変更なし）。

### 9-4. 🔴 setPendingIntentCount(0) と pendingIntentCount_ の線形化点 — **二次レビューで全面再設計**

**ファイル**: `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:43` / `AudioEngine.Threading.cpp:117` / `ReleaseResources.cpp`
**状態**: ⚠️ 既存設計の問題
**详细改修設計**: REPAIR_PLAN2-dash.md 1.3 — **二次レビュー最重要指摘**:
- **`fetch_add after push` は underflow する**: 「push 成功 → fetch_add が pop より先」という保証がない（queue の publication と counter の RMW は別同期変数）。`P: push → C: pop → C: fetch_sub → P: fetch_add` の順序で count が UINT64_MAX へ underflow する
- **正しい設計 = residency reservation → push → failure rollback**: `fetch_add(1)` を push **前**に置き、push 失敗時のみ `fetch_sub(1)` で rollback。fallback 経路（observeDeferredRing_ / quarantineFallbackQueue_）成功時は reservation を維持
- **意味論を確定**: `pendingIntentCount_` = **Observe + Quarantine + Recovery の residency**（Publish は含まない — `publicationBacklogCount_` 側）。`enqueuePublicationIntent`（ISRRuntimePublicationCoordinator.h:273-278）は pendingIntentCount_ を更新しない
- **🔴 現行矛盾を解消**: `AudioEngine::isFullyDrained()`（Threading.cpp:117）は `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)` で **Publish の deferred commit を pendingIntentCount_ に混入**している。この上書きを廃止
- `isFullyDrained` は **queue emptiness を source of truth**（admission closed 後に限定）。`fetch_sub` は skip（epoch-FIFO）も減算対象に含める（9-14 参照）
- ISR/RT: `static_assert(std::atomic<uint64_t>::is_always_lock_free)` + cache-line bouncing 注意（診断 counter として扱い RT 主要制御変数にしない）
**対応 Phase**: Phase 6。

### 9-5. 🟡 Recovery の coalesce（マージ）が未実装 — **P3 に降格（レビュー指摘）**

**ファイル**: `ISRRuntimePublicationCoordinator.cpp`（`submitRecoveryRequest`）
**状態**: ⚠️ 計画書:860 との乖離 → **P3 最適化（drop セマンティクス確立後）**
**详细改修設計**: REPAIR_PLAN2-dash.md 1.4 — `lastRecoveryHandle_` tracking による coalesce（`LockFreeRingBuffer` に走査 API を追加せず、最新1件 tracking で**連続する同一 handle** をマージ）。**★ レビュー指摘（2026-08-09）**: tracking 方式は A→B→A の再登場をマージしない（「保留中 Recovery を1件に」の計画書要件を満たさない）。coalesce セマンティクス（世代・buildSource の採用ルール）未確定。**重複 push 自体は正しい動作（FIFO・drop 禁止遵守）でバグではない**ため P2 から降格。
**対応 Phase**: P3（Phase 5）。

### 9-6. 🟡 テストカバレッジ欠落（INV-3/INV-5）— **レビュー承認（実装可）**

**ファイル**: `src/tests/invariant_INV3_INV5.cpp`（新規）/ `CMakeLists.txt`
**状態**: ⚠️ `invariant_*.cpp` 形式のテスト未存在 → **レビュー承認**
**詳細改修設計**: REPAIR_PLAN2-dash.md 1.5 — retire 順序 / pendingReclaim 再試行 / Recovery 発行 / recoveryIntentDropCount の回帰テスト。**★ 一次レビュー（2026-08-09）**: ABA / state-ownership テスト（Quarantined → requestReclaim が Retired に上書きしない、isRetired ガード）を明記。既存 INV-3-1 に含む。
**🔴 二次レビュー（2026-08-09）— INV-5 を「silent loss 禁止」に再定義**: 現行コードは「★ INV-5: Recovery drop 禁止」（:661）とコメントしながら full 時に `recoveryIntentDropCount_++` + telemetry する semantic mismatch。INV-5 = **Recovery request loss must never be silent**（Normal: enqueue success → Builder consumes / Saturation: enqueue failure → recoveryIntentDropCount++ + Critical health / Forbidden: no telemetry・pendingIntentCount++・false success）。真に drop 絶対禁止なら bounded 256 SPSC ring 自体の再設計が必要（単なる P2 修正ではない）。
**対応 Phase**: Phase 6。

### 9-7. 🟡 shutdown 時の pendingReclaimHandles_ エッジケース — **force reclaim 案は撤回（レビュー指摘）**

**ファイル**: `AudioEngine.CtorDtor.cpp:185-224`
**状態**: ⚠️ 異常系のみ（stuck Reader 時）
**详细改修設計**: REPAIR_PLAN2-dash.md 1.6 — **force reclaim を実装しない**。**★ 一次レビュー（2026-08-09）**: `activeReaderCount()==0` ≠ 当該 handle の reclaim 安全性（EBR 判定は `retireEpoch < minReaderEpoch`）。旧案の `pendingReclaimHandles_.clear()` 無条件実行は requestReclaim false 時に **handle loss（leak）** になる。stuck reader は強制削除ではなく **reclaim 失敗として Faulted で可視化**。正常 reclaim パイプライン（reader 停止 → epoch settle → 再試行）を最後まで通す。
**🔴 二次レビュー（2026-08-09）— Faulted ≠ memory safety 保証**: Faulted は「正常 shutdown invariant を満たせなかった」という診断状態であり、メモリ安全性の保証ではない。テストでは **Faulted 遷移 + pendingReclaimHandles_ を無条件 clear しない + 未 reclaim handle を再利用可能状態に戻さない**まで確認する。
**対応 Phase**: Phase 6（テスト固定のみ・コード変更なし）。

### 9-8. 🟡 2つの ShutdownPhase enum の非1:1対応 — **レビュー承認（実装可）**

**ファイル**: `AudioEngine.h:2521`（AudioEngine::ShutdownPhase, int）と `ISRShutdown.h:25`（convo::isr::ShutdownPhase, uint8_t）
**状態**: ⚠️ 既存問題。**★ 実装照合: 非1:1** — `StopWorkers` が isr の3フェーズ（ObserverDrained → RetireClosed → EpochSettled）を駆動。`StopAudio` は isr を直接遷移させない。
**详细改修設計**: REPAIR_PLAN2-dash.md 2.1 — 遷移シーケンス（順序）を invariant テストで固定（`switch` 網羅では検出不可）。
**対応 Phase**: Phase 6。

### 9-9. 🟡 BlockDouble の finalizeCrossfadeMixPath(..., false) で dryScaleGain_ 未リセット — **レビュー承認（対象外）**

**ファイル**: `AudioEngine.Processing.BlockDouble.cpp:434`（false）/ `AudioBlock.cpp:458`（true）
**状态**: ⚠️ 事前存在の差異（work88 対象外）
**详细改修設計**: REPAIR_PLAN2-dash.md 2.4 — Phase 6（soak）で BUG-028 の RT-only ownership 契約との整合確認。
**対応 Phase**: Phase 6。

### 9-10. 🟡 PublishReceiptWaiter の high-water mark（厳密な per-seqId FIFO ではない）— **レビュー承認（現状維持）**

**ファイル**: `AudioEngine.h:3607`
**状態**: ⚠️ 設計上許容（SPSC 前提）
**详细改修設計**: REPAIR_PLAN2-dash.md 2.2 — MPSC 化（Phase 5）後に per-seqId FIFO へ拡張。
**対応 Phase**: Phase 5（MPSC 化時）。

### 9-11. 🟡 bootstrap publishWorld 失敗の ignoreUnused — **レビュー承認（診断のみ）**

**ファイル**: `AudioEngine.Init.cpp:55`
**状态**: ⚠️ 軽微なロバストネス事項
**详细改修設計**: REPAIR_PLAN2-dash.md 2.5 — `jassert` 追加（Debug のみ）。
**対応 Phase**: Phase 9。

### 9-12. 残余リスク（R1/R4/R5/R6）

- **R1**: `recoveryIntentQueue_` は SPSC（:434）。将来 Timer 等から直接呼ぶ場合は MPSC 化が必要（Phase 5 将来拡張）
- **R4**: retire 順序逆転は quarantine fallback で UAF/リーク排除。runtime 経路は `requestReclaimHandle` 化済み。`shutdownReclaim` の全廃（大規模リファクタ）は保留
- **R5**: bootstrap ignoreUnused null world リスク — 軽微
- **R6**: BlockDouble finalizeCrossfadeMixPath(false) — Phase 6 確認
- **（解決済みで dash から削除）**: R2（observe デッドメンバ削除済み）、R3（Recovery 経路接続済み）

### 9-13. 🟡 ビルド環境メモ（2026-08-09）

- **`build-icx-old-broken` ディレクトリ（後始末待ち）**: `build-icx` ディレクトリが WSL/Windows FS 競合で破損（`.obj` がアクセス拒否で削除不能）したため、`build-icx-old-broken` にリネームした。**後始末**: 管理者権限で `icacls` / `takeown` 後に削除するか、OS 再起動後に削除を試行する。新しい `build-icx` は正常にビルド成功（ConvoPeq.exe 48.5MB / AudioEngineHarness.exe 12.6MB / テスト群）。
- **`AudioEngine.Processing.AudioBlock.cpp` の一時 FAILED**: フルクリーン後の初回ビルド（build-warn3.log:299/407）で一時的に FAILED したが、リトライで成功。git 上未変更（recent fixes と無関係）で、原因はビルドディレクトリ破損（WSL/Windows FS 競合）と判断。**対応**: 次回クリーンビルドで再現しないことを確認（Phase 1 完了条件の一部）。
- **`build-warn.log` / `build-warn2.log` / `build-warn3.log` / `build-verbose.log` / `run-icx-warn*.bat` / `run-icx-warn*.ps1` / `icx-cmd*.txt`**: ビルド検証中に生成した一時ファイル。**後始末**: 確認後に削除推奨（`doc/work88/big_bug/` 以外のルートに残存）。

### 9-14. 🟡 REPAIR_PLAN2-dash.md 妥当性検証で確定した事項（2026-08-09 追加）

**検証方法**: REPAIR_PLAN2-dash.md の各改修設計を実コード照合（MpscBoundedRing.h / ISRRuntimePublicationCoordinator.cpp / ProcessIntent.cpp / ISRDSPHandle.h / ReleaseResources.cpp）で検証。

- **🔴🔴 MpscBoundedRing の `diff` 符号解釈 — 旧確定は誤り（2026-08-09 レビューで撤回）**: 旧記述は「`diff < 0` = producer hole、`diff > 0` = empty」と確定していたが**誤り**。**Empty と producer hole は `diff` が同一値になり区別不能**（初期状態 `sequences_[i]=i` と、producer が CAS 予約後未 publication の状態は同じ観測値）。例: `dequeuePos=0, sequences_[0]=0` のとき、空キューでも producer hole でも `diff = 0 - 1 = -1`。**→ `PopStatus` 導入は設計不成立（9-3 参照）**。push 側の `diff = seq - pos`（Full 判定）と pop 側の `diff = seq - (pos+1)` のオフセット違いは正しいが、pop 側の符号だけで Empty/hole を判別するのは不可。
- **🟡 Observe の2経路 pop と epoch-FIFO skip**: `submitObserve` は `intentQueue_` と `observeDeferredRing_` の両方で `pendingIntentCount + 1`。`ObserveIntentHandler`（ProcessIntent.cpp:65）と `drainObserveDeferred`（:52）は epoch-FIFO skip 時に intent を pop 済みとして扱う。**`fetch_sub` 方式（dash §1.1）では skip も減算対象に含める必要がある**。
- **🔴🔴 pendingIntentCount_ は3系統が混在（2026-08-09 実コード調査で確定）**: `pendingIntentCount_` への書き込み元は5箇所あり、**3つの異なる意味論**が競合している: ①intent 系 residency（submitObserve:553,562 / submitQuarantine:714,724 / submitRecoveryRequest:667 の `load()+1`、popRecoveryRequest:686 の `load()-1`、processIntent:43 の `0`）②**Publish 系**（isFullyDrained Threading.cpp:117 の `hasDeferredCommit ? 1 : 0`）③**RetireIntent 系**（Commit.cpp:462,604 の `lifetime().pendingIntentCount()` — retireBacklogCount_ と同値）。特に Commit.cpp の RetireIntent 混入は二次レビューで未指摘の新発見。**本改修で ②③ の混入を廃止し、①に一本化**。加えて Coordinator 側 `quarantineResidentCount_`（:713,723 +1）は pop 側減算が存在せず isFullyDrained の実測値上書き（Threading.cpp:131）に依存 — **Quarantine pop 時に -1 を追加**（RetireRuntimeEx 側 EpochControl::quarantineResidentCount_ とは別系統。こちらは Epoch ドメインの物理滞留数で reclaim 時 fetch_sub）。
- **🔴🔴🔴 isFullyDrained は全カウンタを実測値に上書き（2026-08-09 三次レビューで確定）**: `AudioEngine::isFullyDrained()`（Threading.cpp:114-136）は pendingIntentCount だけでなく **publicationBacklog / fallbackBacklog / retireBacklog / deferredRetire / quarantineResident の全カウンタを「その瞬間の実測値」で上書き**してから `runtimePublicationBridge_.isFullyDrained()` を呼ぶ。`retireBacklogCount_` も enqueueRetire（:136-160）で load→store 非RMW。**カウンタの増減は isFullyDrained 呼び出し時に消去される**。→ **三次レビュー設計判断**: 混入廃止（Publish/RetireIntent）と hard reset 廃止のみを本改修で実施。isFullyDrained の他の実測上書きは「drain 判定の正しさ」を担保している面があり現状維持 + queue emptiness で補強（実測上書きの全廃は別タスク）（dash §1.1/§1.2 に反映）。
- **🔴🔴🔴 全 producer は NonRT（2026-08-09 三次レビューで確定）**: submitObserve（DSPTransition.h:156 / AudioEngine.Timer.cpp:896,1029,1568）、submitQuarantine（Timer.cpp:1788,1826）、submitRecoveryRequest（AudioEngine.h:4277）、enqueuePublicationIntent（AudioEngine.h:4413）— **全て Timer/Transition/Commit（NonRT）**。CoordinatorLoop 周期 = 1ms（kIntervalMs=1, ISRCoordinatorLoop.h:32）。**二次レビューの「RT producer が high-frequency fetch_add で cache-line bouncing」懸念は現状の実コードでは成立しない**（将来の RT enqueue 設計のみ注意）（dash §1.1 に反映）。
- **🟡 force reclaim の既存フォールバック**: `~AudioEngine`（CtorDtor.cpp:194-221）は既に **Graceful Drain ポーリング（最大 5000ms, publishEpoch + tryReclaim）** を実装。**「force しない正常 reclaim 促進」は既にコードで実践済み**（dash A-2.2 に反映）。
- **🟡 Recovery coalesce は SPSC 単一 Producer 前提**: `submitRecoveryRequest` は CoordinatorLoop 単一スレッドからのみ呼ばれるため、`lastRecoveryHandle_` は plain メンバで十分（`std::atomic` 不要、MPSC 化（R1）時に atomic + CAS へ変更）。`DSPHandle::operator==`（slot + generation）で coalesce 判定。**ただし 9-5 のとおり P3 に降格（連続同一 handle のみ・A→B→A 不可）**（dash A-1.2 に反映）。
- **🟡 dual ShutdownPhase は非1:1対応**: `AudioEngine::ShutdownPhase`（**6値**, AudioEngine.h:2521 `enum class:int`）と `isr::ShutdownPhase`（11値, ISRShutdown.h:25）は非1:1。`StopWorkers` が isr の3フェーズ（ObserverDrained → RetireClosed → EpochSettled）を駆動し、`StopAudio` は isr を直接遷移させない（ReleaseResources.cpp:73-537 実測）。**遷移シーケンス（順序）で invariant 検証する必要がある**（`switch` 網羅では検出不可）（dash A-3.1 に反映）。**三次レビューで行番号・値数を修正（旧記述の7値は誤り）**。

**対応 Phase**: Phase 6（dash §1.1 residency accounting 再設計 / §1.4 テスト / §1.3 test hook + テスト / §1.2 isFullyDrained queue emptiness / A-3.1 dual ShutdownPhase）。

### 9-15. 🟡 REPAIR_PLAN2-dash.md 二次レビュー総括（2026-08-09 追加）

**レビュー対象**: REPAIR_PLAN2-dash.md の P2 残課題・改修設計
**レビュー日**: 2026-08-09（一次・二次）

| 項目 | 二次レビュー判定 | 反映 |
|------|-------------|------|
| 9-2 per-type admission 統一 | 妥当だが「統一」は表層的。バグではない | **P3 に降格** |
| 9-3 PopStatus 導入 | **設計不成立**（Empty と producer hole は diff 同一値） | **撤回**。テスト固定のみ |
| 9-3 producer hole テスト | **現行 API では hole を生成不能**（push が reservation→payload→publication を内包） | **CONVO_TESTING test hook 必須**。テスト4本 |
| 9-4 pendingIntentCount | **fetch_add-after-push は underflow**（queue publication と counter RMW は別同期変数） | **residency reservation → push → rollback に再設計** |
| 9-4 意味論 | **pendingIntentCount_ は Publish を含まない**（Publish は publicationBacklogCount_） | **意味論修正** + isFullyDrained の Publish 混入上書き（Threading.cpp:117）廃止 |
| 9-4 queue emptiness | 正しいが **admission closed 後に限定** | **shutdown 順序 invariant 化** |
| 9-4 ISR/RT | RMW は lock-free だが cache-line bouncing 注意 | **is_always_lock_free 確認 + 診断 counter 扱い** |
| 9-5 Recovery coalesce | 連続同一 handle のみ。**連続 coalesce も今は入れない** | **P3 に降格** + latest-wins 等の仕様確定を先送り |
| 9-6 テストカバレッジ | 妥当（ABA/state-ownership テスト明記）。**INV-5 は silent loss 禁止に再定義** | **承認** |
| 9-7 force reclaim | **採用不可**（activeReaderCount==0 ≠ reclaim 安全。clear() は loss） | **撤回**。Faulted ≠ memory safety 保証を明記 |
| 9-8〜9-11（2.x 系） | 妥当（2.2 は reorder を許す admission/deferred 設計が本質） | **承認** |
| 9-14 diff 符号解釈 | **旧確定は誤り**（Empty/hole 区別不能） | **撤回・修正** |

**最終実装対象（P2）**: 9-4（residency accounting 再設計）+ 9-6（テスト）+ 9-3（test hook + テスト）+ isFullyDrained queue emptiness（admission closed 後）。**その他は P3 または撤回**。

### 9-16. 🟡 REPAIR_PLAN2-dash.md 三次レビュー総括（2026-08-09 別視点検証）

**レビュー観点**: dash の行番号・実装可能性・既存コードとの整合・ISR/RT 前提の実在性を**別視点**から実コードで検証。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映（新構成） |
|---|---------|------------------------|----------|
| T1 | MpscBoundedRing seq 遷移・diff 解釈 | sequences_[i]=i(:54) → push pos+1(:81) → pop pos+Capacity(:109)。diff(:103) は !=0 で false。empty と producer hole の区別不能・正常動作で diff>0 発生せず — **§1.3 の分析は正確** | 修正不要 |
| T2 | processIntent の行番号 | quarantineFallbackQueue_(:32) → intentQueue_(:36) → drainObserveDeferred(:39) → setPendingIntentCount(0)(:43) — **dash の記述は正確** | 修正不要 |
| T3 | pendingIntentCount_ の書き込み元 | 5箇所3系統（intent系/Publish系/RetireIntent系）— **§1.1 の 3系統混在は実在** | 修正不要 |
| T4 | **isFullyDrained は全カウンタ実測上書き** | Threading.cpp:117-131 は pendingIntentCount 以外も publicationBacklog/fallbackBacklog/retireBacklog/deferredRetire/quarantineResident を**全て実測値で上書き**。retireBacklogCount_ も load→store 非RMW（:136-160） | **§1.1/§1.2 に反映**（混入廃止のみ、他は現状維持 + queue emptiness 補強） |
| T5 | **producer は全て NonRT** | submitObserve（DSPTransition.h:156 / Timer.cpp:896,1029,1568）、submitQuarantine（Timer.cpp:1788,1826）、submitRecoveryRequest（AudioEngine.h:4277）、enqueuePublicationIntent（AudioEngine.h:4413）— 全て NonRT。CoordinatorLoop 周期 = 1ms | **§1.1 の ISR/RT 注意を明記**（RT 経路の bouncing 懸念は現状不成立） |
| T6 | force reclaim の既存フォールバック | Graceful Drain ポーリング（最大 5000ms, publishEpoch+tryReclaim）が既存（CtorDtor.cpp:200-217） | **§3.2 に追記** |
| T7 | ShutdownPhase | AudioEngine.h:2521（6値, enum class:int）/ ISRShutdown.h:25。遷移シーケンス（ReleaseResources.cpp:73-537）は dash 記述と完全一致 | **行番号 2510→2521・7値→6値 修正**（§4.1） |
| T8 | bootstrap / R4 行番号 | Init.cpp:54-55、AudioEngine.h:2027、ReleaseResources.cpp:415,420 — 正確 | 修正不要 |
| T9 | recoveryIntentQueue_ 型 | LockFreeRingBuffer<RecoveryIntent, 256>（ISRRuntimePublicationCoordinator.h:433-434）SPSC — 正確 | 修正不要 |

**三次レビュー総合判定**: dash の技術的根拠は約 90% 正確。修正を要したのは **T4（全カウンタ上書きの追記）・T5（RT 経路の不存在）・T7（ShutdownPhase 行番号）** の3点。**最終実装対象は不変**: §1.1（residency accounting 再設計）+ §1.4（テスト）+ §1.3（test hook + テスト）+ §1.2（isFullyDrained queue emptiness）。

### 9-17. 🟡 REPAIR_PLAN2-dash.md 四次レビュー総括（2026-08-09 ISR/MPSC/shutdown 観点）

**レビュー観点**: ISR（Immutable Snapshot Runtime）・MPSC/SPSC メモリモデル・shutdown/drain・ownership/lifetime の観点から改修案を検証。最新ソースで確認した事実: `ShutdownScheduler::isFullyDrained()` は7カウンタのみ、`AudioEngine::isFullyDrained()` は pendingIntentCount_/publicationBacklogCount_ を hasDeferredCommit で上書き、`publicationSequenceCounter_`（AudioEngine.h:2189）は fetchAddAtomic で seqId 割当、`shutdownReclaim` は ReleaseResources.cpp:415,420 に残存、`dspQuarantineManager_.residentCount()` は quarantine lane の実在 DSP 数。

| # | 検証項目 | 四次レビュー判定 | dash 反映 |
|---|---------|----------------|----------|
| F1 | P2-1 reservation→push→rollback | **GO**（基本原理は正しい） | §1.1 維持 |
| F2 | load→store lost update 廃止 | **GO**（必須） | §1.1 維持 |
| F3 | push後 fetch_add の underflow 回避 | **GO**（reservation-before-publish が正しい） | §1.1 維持 |
| F4 | fallback reservation 維持 | **GO** | §1.1 維持 |
| F5 | setPendingIntentCount(0) 廃止 | **GO**（必須） | §1.1 維持 |
| F6 | Publish/Retire の混入廃止 | **GO** | §1.1 維持 |
| F7 | **quarantineResidentCount_ を pop で減算** | **NO-GO**（別の意味のカウンタ — 実際の quarantine lane DSP 数） | **§1.1.5 で削除**（P3 で別カウンタ化） |
| F8 | **publicationBacklogCount_ = Publish residency** | **NO-GO**（現コードでは hasDeferredCommit 由来） | **§1.1.5 で修正**（publicationIntentResidencyCount_ 新設は P3） |
| F9 | queue emptiness を drain 判定に追加 | **GO**（ただし consumption 禁止・shutdown ordering 条件付き） | §1.2 修正 |
| F10 | MPSC producer-hole 契約明文化 | **GO** | §1.3 維持 |
| F11 | PopStatus 導入撤回 | **GO** | §3.1 維持 |
| F12 | producer-hole test hook | **GO with modification**（2スレッド必須） | §1.3.2 修正 |
| F13 | INV-3 test | **GO** | §1.4 維持 |
| F14 | INV-5 silent-loss test | **GO**（Critical は HealthMonitor 経由で検証） | §1.4.3 修正 |
| F15 | IntentAdmissionPolicy | **P3としてGO**（staged admission に要修正） | §2.1 修正（AdmissionStep） |
| F16 | **Recovery coalesce** | **NO-GO**（正当な Recovery を永久抑制するバグ） | **§2.2 全面修正**（削除・代替3案） |
| F17 | force reclaim を実装しない | **GO**（EBR上正しい） | §3.2 維持 |
| F18 | Faulted と reclaim safety の分離 | **GO**（重要） | §3.2 維持 |
| F19 | ShutdownPhase 対応表 | **GO**（正常系だけでは不十分） | §4.1 修正（Normal/Timeout/Emergency） |
| F20 | PublishReceiptWaiter 現状維持 | **CONDITIONAL**（seq 完了順序の証明が必要） | §4.2 修正（completion monotonicity） |
| F21 | LinearRamp 文書化 | **GO** | §4.3 維持 |
| F22 | BlockDouble false | **GO**（実測前に変更しない） | §4.4 維持 |
| F23 | bootstrap jassert | **GO**（Success 判定確認済み） | §4.5 維持 |
| F24 | memory_order（relaxed） | **GO**（counter は numeric accounting）だが **convo wrapper に統一** | §1.1.2 修正（fetchAddAtomic/fetchSubAtomic） |
| F25 | fetch_sub の cur>0 ガード削除 | **GO**（reservation invariant 成立後。ガードは不整合を隠す） | §1.1.6 修正 |
| F26 | shutdownReclaim 残存 | **確認**（二系統の認識。P2 で統合しない判断は正しい） | §3.2 追記 |

**四次レビュー総合判定**: **部分採用 / 一部 NO-GO**。
- **GO**: P2-1 residency accounting（要 convo wrapper 統一・cur>0 ガード削除）、P2-4 queue emptiness（要 shutdown ordering）、P2-3 producer-hole test（要 2スレッド化）、P2-2 INV-3/INV-5、force reclaim 撤回、ShutdownPhase 対応表
- **NO-GO（修正必須）**: ①quarantineResidentCount_ の pop 減算（別の意味のカウンタ）、②publicationBacklogCount_ = Publish residency（現コードと不一致）、③Recovery coalesce（silent loss バグ）
- **CONDITIONAL**: PublishReceiptWaiter（completion seq monotonicity の証明が必要）

**ISR 上の優先順位（四次レビュー推奨）**: ①pendingIntentCount_ の正確な residency accounting → ②queue emptiness を shutdown source of truth へ追加 → ③shutdown producer 停止順序を固定 → ④MPSC hole test → ⑤INV-3/INV-5 → ⑥Admission Policy 整理 → ⑦Recovery coalesce（別設計）。

### 9-18. 🟡 REPAIR_PLAN2-dash.md 五次レビュー総括（2026-08-09 実装可能性・API 実在性検証）

**レビュー観点**: これまでのレビューが検証していない「実装可能性・API 実在性・既存テストとの整合」を実コードで検証。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| V1 | convo::fetchAddAtomic/fetchSubAtomic の存在 | AtomicAccess.h:91,100 に定義。`std::atomic<T>&, U, memory_order = acq_rel` | §1.1.2 で memory_order 扱い確定 |
| V2 | queue emptiness の API 実在 | MpscBoundedRing::sizeApprox()（:115）/ LockFreeRingBuffer::size()（:76）とも消費なし | §1.2.2 で4キュー×API 確定 |
| V3 | **2つの RuntimePublicationCoordinator** | convo::isr（Intent, ISRRuntimePublicationCoordinator.h:68）と convo::（Publish, core/RuntimePublicationCoordinator.h:24）は別クラス。本設計は前者。core 側は publishWorld のみ | §1.2 に実装場所（ShutdownScheduler::isFullyDrained）追記 |
| V4 | PublishStageResult の値 | core/RuntimePublicationCoordinator.h:15-19 で **Success/Rejected/Failed 3値** | §4.5 で3値・result != Success 追記 |
| V5 | 既存 testProducerHoleDoesNotJumpAhead | 既存（MpscBoundedRingTests.cpp:238）だが真の hole 生成なし。命名衝突注意 | §1.3.3 で命名確定 |
| V6 | 既存テストのスレッド使用 | <thread> 使用（:44）・並行 push 実装済み | §1.3.3 で2スレッド統合性確認 |
| V7 | requestReclaim の依存 | DSPHandleRuntime&/ISRRetireRouter& 依存。pendingReclaimHandles_ は AudioEngine.h:4616 | §1.4.2 で INV-3 スタブ要件追記 |
| V8 | test hook 挿入位置 | push() の CAS 成功直後・entries_ 書込み前に #ifdef ブロック | §1.3.2 で確定 |
| V9 | isFullyDrained 返り値 !hasDeferredCommit | Threading.cpp:135。上書き廃止後も維持 | §1.1.5 で返り値維持追記 |
| V10 | submitObserve の2段階 fallback | intentQueue_ → observeDeferredRing_。reservation は fallback 成功時維持 | §1.1.6 と整合 |

**五次レビュー総合判定**: 改修設計は**実装可能**。API 実在性を全て確認。修正5点: V3（Coordinator 区別）/ V4（PublishStageResult 3値）/ V5（既存テスト衝突）/ V7（INV-3 スタブ）/ V8（test hook 挿入位置）。**設計の中心（P2-1 residency accounting）は変更なしで GO**。

### 9-19. 🟡 REPAIR_PLAN2-dash.md 六次レビュー総括（2026-08-09 最終整合）

**レビュー観点**: `REPAIR_PLAN2-dash.md` と `Practical Stable ISR Bridge Runtime.md` を突き合わせ、ISR/RCU/MPSC/shutdown の観点から最終整合を検証。実コード確認: `isRetired()` ガード（ISRDSPHandle.h:155）が INV-3 の ABA 防止に対応、`recoveryIntentDropCount` が RuntimeBackpressureTelemetry（AudioEngine.h:1546）で HealthMonitor 入力、`shutdownReclaim`（ISRDSPHandle.h:171）が二系統の一翼。

| # | 検証項目 | 六次レビュー判定 | dash 反映 |
|---|---------|----------------|----------|
| S1 | P2-1 residency accounting | **GO** | §1.1 維持 |
| S2 | **invariant「push完了後 == actual」は強すぎる** | **修正必要**（Producer が push() から戻った時点で Consumer が既に pop 済みの可能性） | **§1.1.2 で修正**（基本不変条件 = counter >= actual、== は quiescent point のみ） |
| S3 | P2-4 queue emptiness | **GO** | §1.2 維持 |
| S4 | **queue emptiness は shutdown ordering が必須** | **コード上の invariant として固定** | **§1.2 でコード invariant 化を追記** |
| S5 | P2-3 producer-hole test | **GO** | §1.3 維持 |
| S6 | P2-2 INV-3 / INV-5 | **GO** | §1.4 維持 |
| S7 | **INV-5: drop == memory corruption と定義しない** | **修正推奨**（runtime safety と functional recovery を分離） | **§1.4.3 で追記**（「Recovery Intent の silent loss 禁止」と再定義） |
| S8 | quarantineResidentCount_ | **GO**（意味論が異なる） | §1.1.5 NO-GO 維持 |
| S9 | publicationBacklogCount_ | **GO**（現コードと不一致） | §1.1.5 NO-GO 維持 |
| S10 | Recovery coalesce 撤回 | **GO** | §2.2 NO-GO 維持 |
| S11 | force reclaim 撤回 | **GO**（EBR/RCU safety を破壊） | §3.2 NO-GO 維持 |
| S12 | shutdownReclaim 残存は別問題 | **妥当**（二系統。P2 に混ぜない） | §3.2 追記済み |
| S13 | PublishReceiptWaiter | **CONDITIONAL**（P2 counter bug とは別問題のため今は触らない） | §4.2 維持 |
| S14 | ShutdownPhase 強化 | **GO（P3）** | §4.1 維持 |
| S15 | IntentAdmissionPolicy 統一 | **GO（P3）** | §2.1 P3 維持 |
| S16 | producer は全て NonRT | **確認**（cache-line contention は RT deadline を直接侵害しない） | §1.1 維持 |
| S17 | memory_order relaxed vs acq_rel | **wrapper default（acq_rel）に統一支持** | §1.1.2 反映済み |

**六次レビュー総合判定**: **P2-1 / P2-2 / P2-3 / P2-4 は実装 GO**。実装開始前に2点を明文化:
- **A. `pendingIntentCount_` の基本 invariant は `counter >= actual residency`。`==` は producer quiescence 後に限定**
- **B. queue emptiness は `AdmissionClosed → producer停止/join → queue観測` の順序保証を前提条件**

dash の設計は Practical Stable ISR Bridge Runtime の原則（RT は待たない・所有しない・判断しない / Retire は Epoch を通る / Shutdown は完全 Drain / Overflow は silent loss にしない / Authority を一箇所に集約）と整合し、現行 ConvoPeq に安全に適用できる。

### 9-20. 🟡 REPAIR_PLAN2-dash.md 七次レビュー総括（2026-08-09 実装詳細の完全性）

**レビュー観点**: 実装直前の「実装詳細の完全性」を検証 — reservation→push→rollback の全 call site 網羅、RT 経路の最終確認、CONVO_TESTING の実装影響、staged admission と dispatch の整合、INV-5 の drop テスト方法。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| W1 | setPendingIntentCount 全 call site | :553,562（Observe）:667（Recovery）:714,724（Quarantine）の +1 / :686（popRecoveryRequest）の -1 / processIntent の 0 / Threading:117・Commit:462,604 の上書き | 修正不要 |
| W2 | **intentQueue_ に Publish 混在 → pop 時 fetchSub の非対称** | intentQueue_ は Observe/Publish/Recovery/Quarantine の4種混在（:201-206）。Publish は pendingIntentCount_ 対象外のため、**Publish pop で fetchSub(1) すると非対称 -1**（過小評価 → isFullyDrained が見逃し） | **§1.1.6 で「Publish pop は fetchSub 対象外」を追記（重大発見）** |
| W3 | RT 経路の最終確認 | processBlockDouble（BlockDouble.cpp:27）内に submitXxx / setPendingIntentCount なし。全 producer は NonRT 確定 | 修正不要 |
| W4 | waitForDrain 時点の Coordinator/Builder 停止 | ReleaseResources.cpp:189（shutdownCoordinatorLoop）:190（stopRebuildThread）→ :447（waitForDrain）。両方停止済み | **§1.2 で shutdown ordering の実コード照合を追記** |
| W5 | queue emptiness と pendingIntentCount_ の独立判定 | intentQueue_ は Publish 含む全4種。queue emptiness は Publish 残留も捕捉 | **§1.2 で独立判定を追記** |
| W6 | CONVO_TESTING のメモリレイアウト | MpscBoundedRing は alignas(64) の cache-line 分離。フック用 atomic は #ifdef で消滅 → production レイアウト不変 | **§1.3.2 で private 末尾配置を追記** |
| W7 | staged admission と dispatch 整合 | kDispatchTable は4種 1:1 網羅 + static_assert（ISRIntentDispatcher.h:58-68） | 修正不要 |
| W8 | INV-5 の drop テスト方法 | 既存 testRecoveryRequestEnqueueAndPop（ISRSemanticValidationTests.cpp:608）は 1-hop 輸送検証済み。INV-5 は 256 回 submit → full → 257 回目 drop を検証 | **§1.4.2 で drop テスト方法を追記** |
| W9 | processIntent の3キュー drain | quarantineFallbackQueue_(:32) → intentQueue_(:36) → drainObserveDeferred(:39)。Publish pop 除外を追加 | §1.1.6 で反映 |

**七次レビュー総合判定**: 実装詳細はほぼ完全。修正5点: W2（Publish pop の fetchSub 非対称 — 重大発見）/ W4（shutdown ordering 実コード照合）/ W5（queue emptiness の独立判定）/ W6（CONVO_TESTING レイアウト）/ W8（INV-5 drop テスト方法）。**P2-1/P2-2/P2-3/P2-4 は引き続き実装 GO**。

### 9-21. 🟡 REPAIR_PLAN2-dash.md 八次レビュー総括（2026-08-09 コード責務・実装完了条件）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準実装として、コード上の責務・スレッドモデル・ISR/RCU・MPSC・shutdown/lifetime の観点から再検証。`Practical Stable ISR Bridge Runtime.md` も照合。

**実装完了条件（八次 §25 — 3点の絶対条件）**:
| # | 絶対条件 | 崩すと |
|---|---------|--------|
| C1 | `pendingIntentCount_ == actual residency` を**常時要求しない**（基本 invariant は `counter >= actual residency`。`==` は producer quiescence 後のみ） | 並行中の counter 不整合を誤検出 |
| C2 | **Publish pop では decrement しない**（intentQueue_ に Publish 混在のため） | 非対称 -1 → isFullyDrained が見逃し |
| C3 | **queue emptiness は producer quiescence 後のみ authoritative**（AdmissionClosed + all producers joined をコードで assert / phase guard） | 通常動作中の誤った drain 判定 |

**追加要求**: 条件A（reservation counter は `queue residency + producer-side enqueue reservation` であって「成功した push の数」ではない。コードコメント・テスト名・設計資料で同一に）・条件B（fallback 含めて「1 intent = 1 reservation」）・PublishReceiptWaiter は Producer serialization が seqId monotonicity を保証することをテスト固定（今すぐ API 変更不要）・shutdown lifetime contract を将来タスクとして明文化。

**実コード検証**: `PublishReceiptWaiter::complete`（AudioEngine.h:3604-3614）は mutex 保護の high-water mark。seqId は `fetchAddAtomic(publicationSequenceCounter_, 1)`（:3412）で割当。コメント「executePublish は intentQueue_ を FIFO で処理するため seqId は単調増加で完了する（順序性前提）」（:3605）。

**八次レビュー総合判定**: **実装着手可能（概ね90%以上の設計精度）**。P2-1〜P2-4 は GO。実装完了条件として C1/C2/C3 をコードとテストの双方で固定。ISR 整合性: residency と semantic state の分離 / reservation-before-publication / Publish を pendingIntentCount から除外 / queue emptiness を shutdown 後の transport-level source of truth に / EBR/Retire authority を迂回しない — いずれも Practical Stable ISR Bridge Runtime と整合。

### 9-22. 🟡 REPAIR_PLAN2-dash.md 九次レビュー総括（2026-08-09 残課題の明確化）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(2)`（八次反映版）と `Practical Stable ISR Bridge Runtime.md` を突き合わせて検証。

**総合判定**: **「P2-1〜P2-4 は実装 GO。ただし REPAIR_PLAN2-dash(2) を『完全修正版』とは扱わない」**。

**実装完了条件（九次 §21 — 4条件）**: 条件1（pendingIntentCount_ = queue residency + producer-side enqueue reservation、successful push count ではない）・条件2（1 Intent = 1 reservation）・条件3（Publish pop は pending counter を触らない）・条件4（queue empty は AdmissionClosed + all producers joined + Coordinator stopped + Builder stopped の後だけ drain 判定に使う）。

**残課題6件（九次 §23 — P2 後検証対象・優先度高）**:
| # | 残課題 | 検証優先度 |
|---|--------|-----------|
| X1 | **Recovery Intent の full-drop そのもの**（現状は drop + Critical telemetry。AudioEngine.Retire.cpp:192-196, :223。「Recovery 保証」ではない） | **最優先**（Recovery 保証に直結） |
| X2 | **Publish completion sequence monotonicity の実装保証** | **高**（Shutdown/Receipt correctness に直結） |
| X3 | shutdownReclaim の二系統 | 中（別タスク） |
| X4 | RuntimePublicationCoordinator の authority 二重化（**二段階: X4-A rename / X4-B ownership topology 変更**） | 中（Authority Singularization・INV-X4-1〜5） |
| X5 | Publish Intent residency の専用 counter 未導入 | 中（P3） |
| X6 | quarantine intent residency と quarantine resident の semantic 分離 | 中（P3） |

**X1 の検証（実コード）**: recoveryIntentQueue_（SPSC, 256）が full で drop + `recoveryIntentDropCount_++`（ISRRuntimePublicationCoordinator.cpp:671）→ HealthMonitor が delta 監視し Critical 昇格（AudioEngine.Retire.cpp:192-196, :223）。INV-5 は「drop を正しく記録できるか」の検証であり「絶対 drop しない」保証ではない。将来的には Recovery 専用の durable admission state（primary queue → retry/coalescing state → Critical failure only when recovery guarantee itself is impossible）が望ましい。

**評価**: load/store counter → residency accounting / push→increment → reservation→push→rollback / counter only → counter + actual queue occupancy / Recovery coalesce → NO-GO / Publish pop → pending counter から除外 / queue emptiness → producer quiescence 後のみ authority。RT/audio callback に新しい lock・allocation・ownership・decision path を導入するものではなく ISR の境界を悪化させない。

**推奨実装順序**: counter semantics → Publish pop exclusion → reservation → consumer decrement + hard reset 全廃 → queue emptiness → shutdown phase assertion → producer-hole test → INV-3/INV-5 → seqId monotonicity test → Recovery drop 別タスク。

### 9-23. 🟡 REPAIR_PLAN2-dash.md 十次レビュー総括（2026-08-09 具体的コード変更の実行可能性）

**レビュー観点**: 実装直前の「具体的コード変更が実装コードの構造と整合するか」を検証 — reservation→push→rollback の観測カウンタとの関係、quarantineResidentCount_ の現行維持、fetchSub 挿入位置、SPSC/private アクセス。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| U1 | `observeOverflowCounter_`（:559）の扱い | 「intentQueue_ full → observeDeferredRing_ 退避」の観測カウンタ。reservation とは独立に既存位置（primary push 失敗直後）へ維持。診断カウンタとして residency に含めない | **§1.1.6 で追記** |
| U2 | `submitQuarantine` の `quarantineResidentCount_` +1（:713,723）の現行維持 | reservation は pendingIntentCount_ のみ。quarantineResidentCount_ の +1 は既存の意味論（実在 DSP 数の近似）を維持するため push 成功時（primary または fallback）に既存どおり実行。P3 で quarantineIntentResidencyCount_ に分離 | **§1.1.6 で追記** |
| U3 | fetchSub の挿入位置（pop 成功直後・skip 前） | drainObserveDeferred（ProcessIntent.cpp:50-55）は pop 成功直後・epoch-FIFO skip 判定前に fetchSub。ObserveIntentHandler（intentQueue_ 経由）は processIntent の while ループで fetchSub 済み（ハンドラ内では fetchSub しない）→ 二重計上防止 | **§1.1.6 で追記** |
| U4 | Recovery reservation のスレッド | submitRecoveryIntent（AudioEngine.h:4274）は QuarantineIntentHandler（ProcessIntent.cpp:102）から呼ばれ CoordinatorLoop 内。popRecoveryRequest（Builder Loop）との SPSC 契約維持 | 修正不要 |
| U5 | ShutdownScheduler の private queue アクセス | ShutdownScheduler は RuntimePublicationCoordinator の nested class。C++ では nested class は外側の private メンバにアクセス可能 | 修正不要 |
| U6 | submitRecoveryRequest の SPSC | recoveryIntentQueue_ は SPSC（Producer=CoordinatorLoop 単一）。reservation は CoordinatorLoop 内で行われ競合なし | §1.1.3 正確 |

**十次レビュー総合判定**: 実装コードとの整合は**完全**。修正3点: U1（observeOverflowCounter_ の位置維持）/ U2（quarantineResidentCount_ の現行維持）/ U3（fetchSub 挿入位置・二重計上防止）。**P2-1〜P2-4 は実装 GO**。

### 9-24. 🟡 REPAIR_PLAN2-dash.md 十一次レビュー総括（2026-08-09 残課題 X1-X6 の詳細設計）

**レビュー観点**: `REPAIR_PLAN2-dash(3)`（十次反映版）を最新案として、`ConvoPeq(20260809-022629).md` と `Practical Stable ISR Bridge Runtime.md` を突き合わせて検証。

**総合判定**: **P2-1〜P2-4 は実装 GO。ただし「ISR Runtime 全体を完全に健全化する改修案ではない」**。残課題 X1〜X6 は個別バグ修正ではなく、ISR の「唯一の Authority / Intent residency / Publish completion / shutdown-reclaim」の意味論を最終的に閉じるための設計。

**acceptance criteria 3条件（十一次 §21）**:
- C1: pendingIntentCount_ の意味をコード上で固定（Observe + Quarantine + Recovery の queue residency + producer-side enqueue reservations。Publish / Retire / Quarantine resident excluded）
- C2: queue emptiness は phase-gated（AdmissionClosed + all producers joined + Coordinator stopped + Builder stopped を assert）
- C3: Recovery full-drop を「成功扱い」にしない（drop を Health/diagnostic layer まで伝播。将来は Accepted / Queued / Dropped を区別）

**X1-X6 詳細設計（dash §6）**:
| # | 対象 | 設計方針 | 不変条件 |
|---|------|---------|---------|
| X1 | Recovery Durable Admission | durable Pending state + retry/coalesce（queue 拡張は NO-GO） | INV-X1-1〜4 |
| X2 | Publish completion monotonicity | CAS による monotonic watermark | INV-X2-1〜4 |
| X3 | shutdownReclaim 二系統 | Reclaim Authority は一つ、Safety Precondition が二種類（RuntimeEBR / ShutdownQuiescent） | phase assertion |
| X4 | authority 二重化 | **二段階: X4-A（IntentCoordinator / RuntimePublishAuthority に明示命名・分離）+ X4-B（RuntimeStore ownership topology 変更・RuntimeWorldAuthority を write authority に）**。クラス統合 NO-GO | Authority matrix / INV-X4-1〜5 |
| X5 | Publish residency 専用 counter | publicationIntentResidencyCount_ 新設（deferred と分離） | INV-X5-1 |
| X6 | Quarantine Intent/Resident 分離 | **4 semantic 分離: intentQueue_（intent）/ quarantineFallbackQueue_（ring・実体確定）/ DSPQuarantineManager（resident）/ RetireQuarantineStore（retireQuarantine）** | 状態遷移 / INV-X6-4 |

**実コード検証**: PublishExecutor::executePublish（RuntimePublishExecutor.h:19-20）sole gateway / getCurrentBuildSnapshotForRecovery()（AudioEngine.h:4265）/ onPublishCommitted（RuntimePublishExecutor.h:84 → Orchestrator.h:146）/ quarantineResidentCount_ は ISRRetireRuntimeEx（:219,222,237）と Coordinator（:713,723）の2系統。

**RT 安全性**: RT allocation / delete / mutex / wait / World mutation / publish decision / ownership transfer / crossfade decision / Epoch bypass は全て「追加なし」→ ISR 境界を悪化させない。

**実装順序**: X5/X6 → X2 → X4 → X1 → X3（X1 の設計自体は先に決める）。

**中心原則**: **Queue residency / deferred state / committed state / resident object / reclaimable object を、それぞれ別の semantic state として扱う**。

### 9-25. 🟡 REPAIR_PLAN2-dash.md 十二次レビュー総括（2026-08-09 X1-X6 の具体的コード挿入位置）

**レビュー観点**: §6 の X1-X6 詳細設計が、実装コードのどこに挿入されるか（具体的コード挿入位置）と、P2 実装（§1.1-1.4）と競合しないかを検証。

| # | 検証項目 | 検証結果（実コード照合） | dash 反映 |
|---|---------|------------------------|----------|
| V11 | X3: 既存呼び出し元への影響 | requestReclaim（AudioEngine.h:4248, AudioEngine.Retire.cpp:83）→ RuntimeEBR / shutdownReclaim（AudioEngine.h:2027, ReleaseResources.cpp:415,420）→ ShutdownQuiescent。requestReclaim を reclaim(ReclaimMode) に拡張、shutdownReclaim（ISRDSPHandle.h:171）廃止 | **§6.3 追記** |
| V12 | X5: enqueuePublicationIntent は inline | enqueuePublicationIntent（ISRRuntimePublicationCoordinator.h:273-278）は inline で push のみ。X5 はここに reservation→push→rollback | **§6.5 追記** |
| V13 | X6: submitQuarantine の挿入位置 | submitQuarantine（:690-732）に quarantineIntentResidencyCount_ 追加。quarantineResidentCount_ +1（:713,723）は X6 で撤去 | **§6.6 追記** |
| V14 | X1: submitRecoveryRequest の durable state | submitRecoveryRequest（:647-673）の push 失敗時に durable Pending state へ保持。pendingRecoveryAdmission_ 追加 | **§6.1 追記** |
| V15 | X2: PublishReceiptWaiter との統合 | complete（AudioEngine.h:3604-3614）を monotonic watermark + CAS に変更 | **§6.2 追記** |
| V16 | X4: 型エイリアスと使用箇所 | convo::RuntimePublicationCoordinator（core）は AudioEngine.h:3509 の using / convo::isr:: は7ファイルで使用 | **§6.4 追記** |

**十二次レビュー総合判定**: X1-X6 の詳細設計は実装可能で、挿入位置は全て実コードと整合。修正6点（V11-V16）を dash §6 に追記。**P2-1〜P2-4 は実装 GO。X1-X6 は P2 後の独立タスクとして実施（P2 と競合しない）**。

### 9-26. 🟡 X1-X6 詳細設計の精緻化（2026-08-09 実コード調査）

**調査観点**: X1〜X6 の実装対象コードを詳細に調査し、各 X の詳細設計を実装可能なレベルまで精緻化した。

| # | 対象 | 実コード調査結果 | dash §6 反映 |
|---|------|----------------|-------------|
| R1 | X1: Builder 消費ループ | popRecoveryRequest は RebuildDispatch.cpp:911 の while で消費。recoveryPending（AudioEngine.h:2581,4283）で lost-wakeup 防止 | **§6.1 で Builder durable state 消費を追記** |
| R2 | X2: 3箇所の同期変数 | m_lastObservedSequence（Orchestrator.h:246）と PublishReceiptWaiter::lastCompleted_（AudioEngine.h:3634）の2箇所の watermark + waitFor cv | **§6.2 で3箇所変更を追記** |
| R3 | X3: requestReclaim の epoch 判定 | requestReclaim（:573-608）: retireEpoch < minReaderEpoch 判定（:589）。reclaim は private（ISRDSPHandle.h:188）+ friend Coordinator | **§6.3 で reclaim(ReclaimMode) 設計を追記** |
| R4 | X4: core Coordinator 一時生成 | PublishExecutor（RuntimePublishExecutor.h:63-66）で makeRuntimePublicationCoordinator().publishWorld() 一時生成 | **§6.4 で追記** |
| R5 | X5: processIntent の type 分岐 | processIntent（:36）で Publish は publicationIntentResidencyCount_-- / 他は pendingIntentCount_-- に分岐 | **§6.5 で追記** |
| R6 | X6: quarantineResidentCount_ の source of truth | DSPQuarantineManager::residentCount()（ISRDSPQuarantine.h:50）が唯一の source of truth | **§6.6 で追記** |

**精緻化の結果**: X1-X6 は全て実装可能な詳細設計に到達。特に X1（Builder durable state 消費）、X2（3箇所の同期変数）、X3（reclaim(ReclaimMode) のシグネチャ）、X6（residentCount source of truth）は実装時にそのまま利用できる精度。テスト計画（§6.8）も既存テストの拡張位置を確定。

### 9-27. 🟡 REPAIR_PLAN2-dash.md 十四次レビュー総括（2026-08-09 P2 GO・X1-X6 要修正）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(4)` を ISR/Audio Thread 境界・Immutable Snapshot Runtime・Authority Singularization・ownership/lifetime/reclaim・queue residency と semantic state の分離・shutdown 安全性・publication/completion ordering まで含めて評価。

**総合判定**: **P2-1〜P2-4 は GO。X1〜X6 は「このまま実装してよい完成設計」とするのは NO-GO**。

| X | 判定 | 必須修正 |
|---|------|---------|
| X1 Recovery | 🟡 | single slot をやめ、Recovery generation / durable state に再設計。reservation を push 前 |
| X2 Completion | 🔴 最優先 | watermark と per-request receipt を分離。wraparound semantics を決定 |
| X3 Reclaim | 🟡 | shutdown precondition を retire 前に評価。reader re-entry 不可まで保証 |
| X4 Authority | 🟡 | rename は有効。RuntimeWorldAuthority → RuntimeStore write path を実際に一本化 |
| X5 Publish residency | 🟢 | そのまま採用可能 |
| X6 Quarantine | 🟡 | Intent/Ring/Resident を3分離。quarantineResidentCount_ の意味を再定義 |

**dash §6 反映**: X2 は「completion を何と定義するか」から再設計（contiguous completion 前提で store で十分・sparse は将来・wraparound は案A・INV-X2-5 sole completion writer）。X1 は `PendingRecoveryAdmission`（pending/recoveryGeneration/buildSource）に再設計・reservation を push 前。X3 は precondition を retire 前に評価・reader re-entry impossible 追加。X4 は RuntimeStore の friend Owner（core/RuntimeStore.h:81）構造を確認し RuntimeWorldAuthority を publication authority surface に。X5 は GO 承認。X6 は quarantineIntentResidencyCount_/quarantineRingResidencyCount_/quarantineResidentCount_ の3分離。

**実コード検証（十四次）**: PublishReceiptWaiter::waitFor（AudioEngine.h:3628）は `seqId <= lastCompleted_` の contiguous completion 前提。PublishExecutor が sole gateway のため現行は正しい。RuntimeStore<World, Owner>（core/RuntimeStore.h:12-83）は friend Owner（:81）のみ acquireWriteAccess 可、WriteAccess は move-only（:21-28）。core Coordinator が Owner（:34）。

**最優先修正**: 1. X2（completion watermark と receipt の完全分離）2. X1（single-slot 再設計）3. X1（reservation を push 前）4. X6（3分離）5. X3（precondition を retire 前）6. X4（write authority 実体一本化）。

### 9-28. 🟡 X1-X6 追加調査による詳細設計の精緻化（2026-08-09）

**調査観点**: X1-X6 の設計根拠となる実装コードをさらに深く調査し、未確定事項を確定。

| # | 対象 | 追加調査結果 | dash §6 反映 |
|---|------|-------------|-------------|
| T1 | X1: recoveryGeneration の実体 | rebuildRequestGeneration（AudioEngine.h:2423）は requestRebuild（RebuildDispatch.cpp:643）で ++。isRebuildObsolete（:2464）。Builder の Recovery 消費（:965-967）で recoveryGeneration を取得 | **§6.1 に追記**（coalesce は generation 単位） |
| T2 | X2: Committed と Completion の区別 | lastCommittedPublicationSequence_（AudioEngine.h:2191, Commit.cpp:398）は Committed。PublishReceiptWaiter::lastCompleted_（:3634）は Completion。2つを分離し INV-X2-2 を明示 | **§6.2 に追記** |
| T3 | X3: reader re-entry impossible の API | ISRRetireRouter（:71-74）: registerReaderThread / reserveReaderThread / enterReader / exitReader / activeReaderCount（:67）/ minReaderEpoch（:75） | **§6.3 に追記**（isQuiescent は readerRegistrationClosed を含む） |
| T4 | X4: RuntimeStore の Owner 実体 | RuntimeStore<World, Owner>（core/RuntimeStore.h:12）は friend Owner（:81）のみ acquireWriteAccess（:83）。core Coordinator が Owner（:34） | §6.4 反映済み |
| T5 | X5: seqId 割当 | reserveRuntimePublicationIdentity（AudioEngine.h:3406-3414）で fetchAddAtomic + 1 | §6.5 反映済み |

**確定結果**: X1 の coalesce は rebuildRequestGeneration 単位（T1）、X2 は Committed（lastCommittedPublicationSequence_）と Completion（lastCompleted_）の2 sequence を分離（T2）、X3 の isQuiescent は readerRegistrationClosed を含む（T3）。全て実装時に利用可能な精度に確定。

### 9-29. 🟡 REPAIR_PLAN2-dash.md 十五次レビュー総括（2026-08-09 P2/X5 GO・X1-X6 条件付き GO）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(5)` を ISR/Immutable Snapshot Runtime/Authority Singularization/ownership・lifetime/shutdown/completion ordering の観点から再検証。

**総合判定**: **P2 = GO、X5 = GO。X1/X2/X3/X4/X6 = 設計方向 GO、実装は条件反映後 GO**。dash(5) は dash(4) までの問題点をかなり正確に潰している。**過去レビュー由来の記述が一部残っており、最新版の結論と旧版の記述が同一文書内で混在**（実装時は A-2.14 以降の記述を正とする）。

| X | 判定 | 実装条件 |
|---|------|---------|
| X1 Recovery | 🟡 条件付き GO | pendingIntentCount_ と durable Recovery admission の semantic boundary を明文化（transport residency と recoveryAdmissionPending_ を分離） |
| X2 Completion | 🔴 最優先 | 4層の sequence 分離（publicationSequenceCounter_ / lastCommittedPublicationSequence_ / lastCompletedSequence_ / per-request receipt）+ completion is FIFO を architectural invariant に |
| X3 Reclaim | 🟡 条件付き GO | readerRegistrationClosed + all producers stopped + Audio stopped まで shutdown precondition |
| X4 Authority | 🟡 条件付き GO | rename は GO。最終目標は RuntimeWorldAuthority を actual RuntimeStore write authority に |
| X5 Publish residency | 🟢 GO | そのまま採用可能 |
| X6 Quarantine | 🟡 条件付き GO | 3 semantic 厳密固定。quarantineResidentCount_ に aggregate 値を絶対に入れない |

**dash §6 への反映**: X1 は pendingIntentCount_ を transport residency に限定し recoveryAdmissionPending_ を独立（push 失敗時は pendingIntentCount_ から rollback し切替）。X2 は4層の sequence 分離。X6 は3 semantic 厳密固定。実装順序は X1 を X4 より先に（X1/X2 は correctness の根幹）。

**acceptance criteria**: INV-X1〜INV-X6 をコードレベル不変条件として固定 + X1/X2/X3 は正常系だけでなく queue-full / out-of-order / shutdown race / reader re-entry の adversarial test を必須。

### 9-30. 🟡 X1-X6 追加調査による更なる詳細設計の精緻化（2026-08-09）

**調査観点**: X1-X6 の実装対象コードの正確な構造（EpochDomain / RCUReader / RuntimeWorldAuthority / waitForPublishReceipt 呼び出し元）を深く調査し、実装詳細を確定。

| # | 対象 | 追加調査結果 | dash §6 反映 |
|---|------|-------------|-------------|
| U1 | X3: reader 登録の実体 | EpochDomain::kMaxReaders = 64（core/EpochDomain.h:22）。registerReaderThread（:45-65）は slot を CAS 確保。RCUReader::enter（core/RCUReader.h:65）→ acquireThreadSlot → registerReaderThread。audioThreadRcuReader（AudioEngine.h:4529）が Audio callback（BlockDouble.cpp:151）で enter/exit | **§6.3 に追記**（readerRegistrationClosed は EpochDomain の shutdown フラグで実現） |
| U2 | X1: submitRecoveryIntent の起床統合 | submitRecoveryIntent（AudioEngine.h:4274-4287）: submitRecoveryRequest → recoveryPending = true（:4283）→ rebuildCV.notify_all()（:4286）。recoveryAdmissionPending_ は submitRecoveryRequest 内で set し、起床は既存 recoveryPending が担保 | **§6.1 に追記** |
| U3 | X4: RuntimeWorldAuthority の commit 委譲 | RuntimeWorldAuthority（RuntimeWorldAuthority.h:78-123）は coordinator_（core Coordinator&）参照に commit を委譲（:112）。X4 最終形は coordinator_ を RuntimeStore::WriteAccess に置換（Owner 変更は大規模リファクタ） | **§6.4 に追記** |
| U4 | X2: per-request receipt の呼び出し元 | waitForPublishReceipt(seqId, timeout) は commitRuntimePublication（AudioEngine.h:4450）で呼ばれる。seqId は Producer 自身の割当。タイムアウトしても Transferred 扱い | **§6.2 に追記** |

**確定結果**: X3 の readerRegistrationClosed は EpochDomain の shutdown フラグで registerReaderThread が失敗することを保証（U1）。X1 の recoveryAdmissionPending_ は submitRecoveryIntent の既存起床経路（recoveryPending + rebuildCV）と統合（U2）。X4 は coordinator_ 参照の WriteAccess 置換（U3）。X2 の per-request receipt は commitRuntimePublication:4450 を対象（U4）。全て実装時に利用可能な精度に確定。

### 9-31. 🟡 REPAIR_PLAN2-dash.md 十六次レビュー総括（2026-08-09 最終設計レビュー・X4 NO-GO）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、`REPAIR_PLAN2-dash(6)` を ISR/RCU・ownership/lifetime・publication ordering・queue residency・shutdown・Authority Singularization の観点から検証。

**総合判定**: **P2-1〜P2-4 は GO。X1/X2/X3/X6 は条件付き GO。X4 は現状 NO-GO（rename だけでなく RuntimeStore ownership topology の再設計が必要）**。

| X | 判定 | 実装条件 |
|---|------|---------|
| X1 Recovery | 🟠 条件付き GO | reservation / coalesce state machine を確定。1 logical Recovery admission = exactly one reservation（INV-X1-5）。durable admission を queue residency と二重計上しない（INV-X1-6） |
| X2 Completion | 🟠 条件付き GO / 最優先 | FIFO completion invariant を実装契約として強制（INV-X2-6: completion order == publication sequence order） |
| X3 Reclaim | 🟠 条件付き GO | readerRegistrationClosed を shutdown state machine に統合（INV-X3-4） |
| X4 Authority | 🔴 現状 NO-GO | RuntimeStore ownership topology を再設計（INV-X4-3: publishAndSwap は RuntimeWorldAuthority-owned WriteAccess のみ） |
| X5 Publish residency | 🟢 GO | そのまま採用可能 |
| X6 Quarantine | 🟠 条件付き GO | Ring/Intent/Resident/RetireQuarantine を4分離（INV-X6-4） |

**追加 INV**: INV-X1-5 / INV-X1-6 / INV-X2-6 / INV-X3-4 / INV-X4-3 / INV-X6-4（dash §6 に反映）。

**実コード検証**: RetireQuarantineStore（RetireQuarantineStore.h:60, kMaxQuarantinedEntries=512）が retire 対象の quarantine 退避を管理（:69 quarantine / :77 drain / :157 residentCount）。RuntimeWorldAuthority は coordinator_ 参照に commit 委譲（:112）し、本当の Store owner ではない。

**最終評価**: dash(6) は「完成設計」ではなく「実装直前の最終設計レビュー版」。核心は **Intent → Admission → Transport Residency → Execution → Committed → Completed → Resident → Retired → Reclaimable → Deleted** の semantic state machine を一つずつ別の状態として閉じること。現時点では X1 と X4 の state machine 境界がまだ完全には閉じていない。

### 9-32. 🟡 X1-X6 新設 counter の宣言位置確定（2026-08-09）

**調査観点**: X1〜X6 の新設 counter / durable state の**宣言位置**（どのクラスのメンバとして追加するか）を、実装対象クラスのメンバ構造から確定。

| # | 対象 | 宣言位置 | dash §6 反映 |
|---|------|---------|-------------|
| D1 | X5: publicationIntentResidencyCount_ | ISRRuntimePublicationCoordinator.h:383（publicationBacklogCount_ の隣） | §6.5 に追記 |
| D2 | X6: quarantineIntentResidencyCount_ / quarantineRingResidencyCount_ | ISRRuntimePublicationCoordinator.h:388（quarantineResidentCount_ の隣） | §6.6 に追記 |
| D3 | X6: retireQuarantineResidentCount_ | **新 counter は追加しない**（十八次別視点3）。RetireQuarantineStore の既存 size_（:175）/ residentCount()（:157）を source of truth に。isFullyDrained は retireQuarantineStore().residentCount()==0 を評価 | §6.6 に追記 |
| D4 | X1: PendingRecoveryAdmission / recoveryAdmissionPending_ | ISRRuntimePublicationCoordinator.h:437（recoveryIntentDropCount_ の隣） | §6.1 に追記 |

**確定結果**: X5 の publicationIntentResidencyCount_ は publicationBacklogCount_（:383）の隣（Publish 系 counter 集約）。X6 の intent/ring counter は quarantineResidentCount_（:388）の隣、retireQuarantineResidentCount_ は RetireQuarantineStore 側。X1 の PendingRecoveryAdmission は plain 構造体（SPSC のため atomic 不要）、recoveryAdmissionPending_ は atomic<bool>（isFullyDrained が NonRT から読むため）。全て実装時の宣言位置が確定。

### 9-33. 🟡 X1-X6 詳細設計の実コード精緻化（2026-08-09 十八次調査）

**調査観点**: X1〜X6 の詳細設計を、実装対象コードの正確な実装（関数シグネチャ・挿入位置・counter 更新箇所・テスト対象）と突き合わせて精緻化。未確定事項を確定。

**確定事項**（dash A-2.21 / §6 に反映済み）:
1. **X1 矛盾解消**: 「push 失敗時に pendingIntentCount_ rollback しない（reservation 維持）」の旧記述を、十五次 §14 の最新判断（rollback + recoveryAdmissionPending_ 切替）に統一
2. **X1 PendingRecoveryAdmission に handle/epoch/intentId 追加**（RecoveryIntent :166-179 と一致）。消費時 RebuildDispatch.cpp:917 の isNull 検証に必要
3. **X1 takePendingRecoveryAdmission 実装確定**: SPSC で競合なし。durable 化済みのため pendingIntentCount_ を触らない。Builder 消費ループ（RebuildDispatch.cpp:911 後）に追記
4. **X2 PublishReceiptWaiter::complete() 検証**: mutex ガード付き `if (seqId > lastCompleted_)`。contiguous completion 前提で案1（最小変更・mutex 維持）確定。m_lastObservedSequence と lastCompleted_ の同期をテスト対象に
5. **X3 readerRegistrationClosed 実装確定**: EpochDomain に registrationClosed_ atomic<bool> + closeReaderRegistration()。registerReaderThread/reserveReaderThread 冒頭で拒否。「kMaxReaders 全消費」案は NO-GO。既存 reader は exit まで動作継続
6. **X5 fallback queue に Publish が入らないことを確定**: quarantineFallbackQueue_ は Quarantine 専用。publicationIntentResidencyCount_ 減算は intentQueue_ ループ（:36-37）のみ
7. **X6 submitQuarantine の intent/ring counter 移動確定**: fallback 移動時は intent-- → ring++（同時に 1 にならない）。pop 減算は processIntent の while ループで一元管理（handler は純 routing）
8. **Publish intent は pendingIntentCount_ に含まれない**（ISRSoakTests.cpp:70 コメント確認）

**結果**: X1〜X6 は実装可能な精度に到達。残る未確定事項なし（保留は dash §6.9 / A-2.20 の将来タスクとして明記）。

### 9-34. 🟡 X1-X6 別視点調査（スレッド所有権・外部 setter 干渉・メモリオーダリング）（2026-08-09 十八次・別視点）

**調査観点**: 前回の関数シグネチャ・挿入位置に加え、スレッド所有権・外部 setter との干渉・メモリオーダリング・isFullyDrained 実装・RCUReader 経路の別視点から検証。

**確定事項**（dash A-2.22 / §6 に反映済み）:
1. **外部 setter 干渉（最重要）**: `AudioEngine::isFullyDrained()`（Threading.cpp:114-136）が Coordinator counter を外部 setter で強制上書き:
   - `setPendingIntentCount(hasDeferredCommit ? 1u : 0u)`（:117）— X1（recoveryAdmissionPending_）/ X6（quarantineIntentResidencyCount_）と衝突
   - `setQuarantineResidentCount(ringResident + dspQuarantineResident)`（:131）— X6 の分離と矛盾
   - **X1〜X6 実装時は外部上書きを廃止**し、Coordinator 内部で純粋 accounting（§6.7）
2. **X2 complete() の thread 所有権**: CoordinatorLoop::run（ISRCoordinatorLoop.cpp:31-43）の単一スレッド。waitFor は Producer スレッド。mutex は cv 同期に必要。shutdown 中は complete が来ない → waitFor は timeout で Transferred
3. **X3 の2つの ShutdownPhase enum**: AudioEngine（:2521-2530）と ISRShutdown（:25-41）。CloseReaderRegistration は既存列挙に無い → StopAudio 完了時の副作用として registrationClosed_ を set（enum 変更最小化）
4. **X4 friend 関係**: PublishExecutor（friend struct :3580）は runtimeStore 直接アクセス → X4-B で worldAuthority() 経由に置換。WriteAccess の friend Owner 変更は PublishExecutor に影響しない
5. **メモリオーダリング**: fetchAdd/fetchSub は default acq_rel（AtomicAccess.h:91-105）。X5/X6 の新設 counter も acq_rel 明示。Producer/Consumer の2スレッド間 RMW で race なし
6. **X6 residentCount() 実装**: DSPQuarantineManager::residentCount()（ISRDSPQuarantine.cpp:103-111）は quarantineActiveFlags_ 走査。X6 は走査を維持（新 atomic 追加しない）、isFullyDrained は NonRT で直接読む（kMaxSlots=256 許容）

**結果**: 別視点からも X1〜X6 の詳細設計が確定。**外部 setter 干渉は X1/X6 実装時の必須対応**。残る未確定事項なし。

### 9-35. 🟡 X1-X6 共通パス・テスト基盤調査（2026-08-09 十八次・別視点2）

**調査観点**: Publish の共通 enqueue 経路（enqueuePublicationIntentForRuntimeCommit → submitPublishRequest）、deferred 再 enqueue、observeDeferredRing_ の counter 管理、テスト基盤（CMakeLists・既存テスト配置）から検証。

**確定事項**（dash A-2.23 / §6 に反映済み）:
1. **X5 の Publish intent 全 enqueue 経路（3経路）**: 経路1（RebuildThread・通常 rebuild）/ 経路2（Builder Loop・Recovery publish）/ 経路3（RebuildThread・deferred 再 enqueue）がすべて `enqueuePublicationIntent`（:273）に集約。X5 の counter は単一箇所 reservation で二重計上なし。`enqueuePublicationIntentForRuntimeCommit`（:688）は直接 push しない
2. **X1 と X5 の相互作用**: Recovery publish（RebuildDispatch.cpp:971 → 経路2）も X5 の counter を +1。X1 の durable admission と X5 は独立。Recovery build 中の「build gap」はどちらの counter にも含まれない
3. **observeDeferredRing_ の pendingIntentCount_ 減算（P2 接続点）**: submitObserve（:561-562）は deferred ring にも +1 するが drainObserveDeferred（ProcessIntent.cpp:47-56）は減算しない。現状は setPendingIntentCount(0)（:43）で整合。**P2 で setPendingIntentCount(0) 廃止時は drainObserveDeferred の pop でも -1 必要**
4. **テスト基盤**: ISRSemanticValidationTests / ISRSoakTests / RuntimeWorldAuthorityProjectionTests が CMakeLists 登録済み。X1/X3/X5/X6 はヘッドレスで直接テスト可。X2 の PublishReceiptWaiter は AudioEngine private メンバのため AudioEngineHarness で統合テスト
5. **X4 静的検査コマンド確定**: `rg -n "publishAndSwap\(" src/` / `rg -n "\.commit\((PublishAuthority|Granted)" src/audioengine/` / `rg -n "RuntimeStore<RuntimePublishWorld" src/`

**結果**: 共通パス・テスト基盤の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-36. 🟡 X1-X6 実装詳細・Producer 前提・reclaim 管理調査（2026-08-09 十八次・別視点3）

**調査観点**: RetireQuarantineStore 実装・shutdownReclaim 呼び出し元・requestReclaim 内部・sequence 採番スレッド・submitRecoveryRequest Producer 前提を実コードで検証。

**確定事項**（dash A-2.24 / §6 に反映済み）:
1. **X6 retireQuarantineResidentCount_ は新 counter 不要（設計変更）**: RetireQuarantineStore は既に `size_`（:175, mutex 保護）+ `residentCount()`（:157-161）を持つ。quarantine() で ++size_ / drain() で size_=w / drainAllUnsafe() で size_=0。**既存 residentCount() を source of truth にする**（2重 source の不整合リスク回避）。isFullyDrained は `retireQuarantineStore().residentCount() == 0` を評価
2. **X1 lost-wakeup 整合**: Builder は recoveryPending クリア（:905）後に popRecoveryRequest（:911）→ takePendingRecoveryAdmission。消費中に新規 durable 化があれば recoveryPending 再 set + notify で次サイクル再消費。recoveryAdmissionPending_ が真実
3. **X3 shutdownReclaim 呼び出し元の順序**: CacheMap::~CacheMap は delete 先（:2026）→ shutdownReclaim（:2027）。ReleaseResources は retire（:414,419）→ shutdownReclaim（:415,420）。X3 移行時も順序維持（再 retire は冪等）
4. **X3 reclaimInFlightCount_ 管理**: +1（:592）は遅延 / 0（:606）は完了の簡略設計。ShutdownQuiescent は epoch 判定スキップのためカウンタを呼ばない
5. **X2 sequence 採番スレッド**: reserveRuntimePublicationIdentity（:3407）は RuntimeBuilder.cpp:81,183（RebuildThread）で呼ばれる。採番→commit が同一スレッドで Producer serialization 成立
6. **X1 Producer 単一スレッド前提の完全確認**: submitRecoveryRequest の呼び出し元は 1 箇所（AudioEngine.h:4277）。RecoveryIntentHandler（ProcessIntent.cpp:126-132）は dead code。Producer = CoordinatorLoop 単一スレッド

**結果**: X6 の retireQuarantineResidentCount_ を既存 residentCount() 使用に変更。X1 Producer 前提・X3 reclaim 管理・X2 採番スレッドを完全確定。残る未確定事項なし。

### 9-37. 🟡 十九次レビュー反映（X4-B の currentWorld_ 意味論修正・実装着手可能 GO）（2026-08-09）

**レビュー観点**: dash(8) の X4 を実装可能性・ownership/lifetime・publication ordering・RCU/ISR・Authority Singularization・shutdown の観点から再検証。

**総合判定**: **X4-A = GO。X4-B = 条件付き GO（修正3点で実装着手可能）**。

**主要指摘**:
1. **`currentWorld_` は metadata cache でなく第二の publication/read surface**: `commit()`（:109-112）が pubWorld->publication を書込み currentWorld_ を更新。getCurrent/getVersion/currentPublicationEpoch/currentPublicationSequenceId は currentWorld_ から導出
2. **`getCurrent()` を `consumeWorldHandle(runtimeStore)` の置換先にすることは NO-GO**: 別の atomic source（currentWorld_ vs RuntimeStore::current）
3. **`publish()` を「一つの atomic publication」と定義しない**: semantic transaction の唯一の execution boundary のみ
4. **X4/X2 境界**: commit-before-swap ordering は X4 / completion monotonicity は X2

**反映**（dash A-2.25 / §6.4-X4 に反映済み）:
- **INV-X4-6 / INV-X4-7 追加**（dual-pointer consistency / 独立 source 禁止）
- **RuntimeWorldAuthority の move/copy 禁止**（static_assert）
- **Bootstrap/shutdown は lifecycle-controlled publish**（sole physical publish gateway = RuntimeWorldAuthority）
- **Test 7 → commit-before-swap ordering / Test 9（dual-pointer consistency）/ Test 10（INV-X4-7）追加**（8本→10本）
- **実装順序 X4-0〜X4-10 → X4-0〜X4-11**（X4-7 publish() 導入独立・X4-9 全 direct caller 排除）

**優先修正3点**（getCurrent ≠ consumeWorldHandle 明文化 / commit-before-publishAndSwap invariant / dual-pointer consistency test）を反映し、**X4-B は実装着手可能な GO**。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜7 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B）→ X3。

### 9-38. 🟡 X1-X6 deferred 経路・OwnerChannel・LifetimeState・overflow ring 調査（2026-08-09 十八次・別視点4）

**調査観点**: deferred publish 経路の詳細・OwnerChannel 実装・LifetimeState::pendingIntentCount と X5/X6 の関係・overflow ring と X6 の関係・deferred と X2 completion の整合。

**確定事項**（dash A-2.26 / §6 に反映済み）:
1. **OwnerChannel は SPSC**（OwnerChannel.h:38-118）: enqueue = Non-RT 単一 / take = ISR 単一。capacity 256。key = (sequenceId, epoch, mappedGeneration)。enqueue は key 重複 reject / take は single-transfer drain。**owner.get() は non-owning / std::move(owner) のみ transfer**（X4 Test 8 の根拠）
2. **LifetimeState::pendingIntentCount() は retire intent の pending 数**（ISRRetire.cpp:182-189: enqueueTicket_ - dequeuePos_ + fallbackCount_）。Commit.cpp:462,604 が Coordinator pendingIntentCount_ に混入。**X5/X6 の transport counter と完全に別 semantic**。setRetireBacklogCount への同値設定は妥当
3. **deferred は単一スロット**（Orchestrator.cpp:360-409: deferredSlot_ + hasDeferred_）。deferredPublicationCount は 0/1 のみ
4. **overflow ring は retire 系**（RetireOverflowRing / coordinatorDeferredRing_ / lastResortQueue_）: X6 の quarantineRingResidencyCount_ と独立。drainOverflowRing の再注入は LifetimeState 側
5. **deferred publish と contiguous completion**（REPAIR_PLAN2.md:914）: deferred は単一スロット（上書き）。INV-X2-6 は「deferred 非発生時」前提と明示。cancel された古い seqId の receipt は来ず waitFor はタイムアウト（250ms）で Transferred
6. **recoveryIntentQueue_ の SPSC 確認**（:434, 256）: Producer=CoordinatorLoop / Consumer=Builder Loop。X1 と整合

**結果**: deferred・OwnerChannel・LifetimeState・overflow ring の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-39. 🟡 二十次レビュー反映（X4-B の Store ownership・publish() 責務・INV-X4-8・Test 定義修正）（2026-08-09）

**レビュー観点**: X4 詳細改修計画（十七次〜十九次反映版）を `ConvoPeq(20260809-022629).md` と `Practical Stable ISR Bridge Runtime.md` を基準に再検証。

**総合判定**: X4-A/X4-B 分離と「RuntimeWorldAuthority を物理 write authority にする」方針は妥当。**「Authority が Store を所有」と「AudioEngine の中で安全に所有」は別問題**。必須修正4点を反映すれば実装 GO。

**反映**（dash A-2.27 / §6.4-X4 に反映済み）:
1. **Store ownership / constructor 確定**: コンストラクタは外部 Store を受け取らず、Authority 自身が Store identity を形成（`runtimeStore_()` + `writeAccess_(runtimeStore_.acquireWriteAccess())`）
2. **publish() の責務限定**: 現行 publishWorld（:100-141）は seal/validate/swap/didPublish/willRetire/retire まで担う。publish() は「validate + commit + release + swap + return oldWorld」に限定。didPublish/willRetire/retire は PublishExecutor → completion → LifetimeState へ委譲
3. **read API 新設**: getCurrent()（ISR metadata source）と分離し、physical RuntimeStore read 専用 API（observePublishedWorld / acquireReadToken / consumeWorldHandle(ReadToken)）を Authority に新設
4. **Test 7 は commit-before-swap ordering**（X2 側は completion monotonicity と分離）
5. **INV-X4-8 追加**（source-role separation）: currentWorld_ = metadata observation alias / RuntimeStore::current = physical publication source。delete/retire/unique_ptr/shared_ptr 変換を禁止
6. **INV-X4-6 の identity 構成要素確定**: sequenceId + publicationEpoch + mappedGeneration（version/boundary は metadata）
7. **Test 3 厳密化**: `RuntimeStore<RuntimePublishWorld, ...>::WriteAccess` まで追う。allowed = RuntimeWorldAuthority / forbidden = RuntimeIntentCoordinator, PublishExecutor, RuntimePublishAuthority, AudioEngine, Builder, DSPTransition
8. **Test 6 修正**: write-capable Store のみ禁止。read-only Store reference（const RuntimeStore&）は read API が保持してよい
9. **RuntimePublishAuthority は WriteAccess を所有してはいけない**（二階層化禁止・INV-X4-3 強化）
10. **Bootstrap/shutdown clear は publish() と統合しない**（clearPublishedRuntimeSnapshotsNonRt は別 API。shutdown semantic は X3）
11. **実装順序を X4-B-0〜X4-B-11 に細分化**（rollback point 多数確保）

**最終評価**: 4点反映後、**X4-B = 実装 GO**。X4 の目的は「physical write capability を RuntimeWorldAuthority に一意化すること」と固定（currentWorld_ / RuntimeStore::current の二重性自体は X4 で解消しない）。

**次回の実装手順**: 設計固定（X2/X1/X4/X3 invariant・INV-X4-1〜8 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3。

### 9-40. 🟡 X1-X6 キュー基盤・generation・free list・複数 Producer 調査（2026-08-09 十八次・別視点5）

**調査観点**: X5/X6 の core である intentQueue_ の基盤（MpscBoundedRing）の内部構造・producer hole・RuntimeBuildSnapshot の generation・X2 receipt の複数 Producer・X3 reclaim の free list ロックを実コードで検証。

**確定事項**（dash A-2.28 / §6 に反映済み）:
1. **MpscBoundedRing の producer hole と X5/X6 counter の整合**: push は reservation（CAS :76）→ payload 書込み（:80）→ seq release（:81）の2段階。**push は publication 完了後に return** するため producer hole は push 内で完結。X5 の fetchAdd 先行設計（counter +1 → push → 失敗 rollback）は、producer hole 中に counter が一瞬過大になるが push の return 後は必ず収束
2. **RuntimeBuildSnapshot.generation と X1 recoveryGeneration**: buildSource.generation は消費時（RebuildDispatch.cpp:968-969）に現在の rebuildRequestGeneration で上書き。**coalesce の generation 判定は PendingRecoveryAdmission.recoveryGeneration を使う**（buildSource.generation ではない）
3. **X2 waitFor の複数 Producer（5箇所）**: PrepareToPlay.cpp:155,277 / ReleaseResources.cpp:175 / Timer.cpp:918 / Transition.cpp:25 / PublicationExecutor.cpp:53。mutex は複数 Producer 間の cv 同期にも必要
4. **X3 reclaim() の freeListMutex_ ロック**: reclaim()（ISRDSPHandle.cpp:129-148）は freeListMutex_（:133）をロックし freeSlots_ 返却（:146-147）。両モードとも同一の reclaim() を使用。RT からは呼ばない
5. **kDispatchTable の 1:1 mapping 確認**（ISRIntentDispatcher.h:60-65）: QuarantineIntentHandler は intentQueue_ と quarantineFallbackQueue_ の両方から dispatch（ProcessIntent.cpp:32-33 + :36-37）

**結果**: キュー基盤・generation・free list・複数 Producer の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-41. 🟡 二十一次レビュー反映（X4-B の型・constructor・member declaration 固定 + CRTP コンパイル検証）（2026-08-09）

**レビュー観点**: X4 計画を `ConvoPeq(20260809-022629).md` を基準に照合。**X4-A = GO / X4-B = 設計として GO 寄り**。実装開始前に4点（member declaration order / RuntimeStore の template dependency / publish() ownership transfer / Bootstrap・shutdown clear の初期化破棄順序）を実コードで固定する必要あり。

**反映**（dash A-2.29 / §6.4-X4-B に反映済み）:
1. **① member declaration order**: runtimeStore_ → writeAccess_ → ownerChannel_ → lifetime_ → registry_（WriteAccess が生きている間に Store が破棄されない）
2. **② CRTP 的 template 依存の実コンパイル検証（g++ -std=c++20）**:
   - `RuntimeStore<World, Self>` の CRTP は既存実績あり（RuntimePublicationCoordinator.h:34）
   - `friend Owner`（:81）は incomplete type でも動作 / `static_assert(is_class_v<Owner>)`（:16）は incomplete でも well-formed
   - **`Store::Owner` はコンパイル不可**（RuntimeStore に using Owner が無い）→ **`using OwnerType = Owner` を追加**すれば Test 1 が成立（COMPILE_OK / RUN_OK 検証済み）
   - member としての WriteAccess は Store の complete type が必要（RuntimeWorldAuthority.h が core/RuntimeStore.h を include）。RuntimeState は forward decl で十分
   - 循環依存なし（RuntimeStore.h は AtomicAccess.h のみ include）
3. **③ publish() の ownership transfer（失敗経路・null経路）**: null owner → Failed / validate 失敗 → Rejected / null→null swap → Failed。null 公開は shutdown clear（clearPublishedRuntimeSnapshotsNonRt）が担当
4. **④ Bootstrap / shutdown clear の初期化・破棄順序**: ctor（runtimeStore_ + writeAccess_ acquire）→ Bootstrap publish → shutdown clear → CoordinatorLoop join → worldAuthority_ 破棄（writeAccess_ → runtimeStore_ 逆順）

**結論**: X4-B の CRTP 懸念（incomplete-type / header include / nested WriteAccess）は全て解消済み。**X4-B は設計 GO・実装開始可能**。`RuntimeStore.h` への `using OwnerType = Owner` 追加が唯一の前提追加。

### 9-42. 🟡 X1-X6 shutdown 順序・保留再試行・shutdown 相互作用調査（2026-08-09 十八次・別視点6）

**調査観点**: shutdown シーケンスの実コード詳細（releaseResources / ~AudioEngine の2系統）・X3 の pendingReclaimHandles_ 再試行機構・shutdown と X1/X5/X6 の相互作用を実コードで検証。

**確定事項**（dash A-2.30 / §6 に反映済み）:
1. **X3 CloseReaderRegistration 挿入位置（2系統）**: 系統1 releaseResources（:34-404）・系統2 ~AudioEngine（CtorDtor.cpp:92-231）。**両系統とも graceful drain（activeReaderCount==0 待ち）の前に closeReaderRegistration() を呼ぶ**（DrainRetire フェーズ開始前）
2. **X3 drainDeferredRetireQueues の保留再試行機構**（Retire.cpp:41-114）: setReclaimInFlightCount(1)→tryReclaim→reclaim→setReclaimInFlightCount(0) + pendingReclaimHandles_ 抽出→isRetired ガード（:75）→epoch 確認（:79）→requestReclaim（:83）→失敗時再登録。**X3 の reclaim(RuntimeEBR) はこの機構を維持**
3. **X1 shutdown 中の durable admission 破棄**: Builder 消費ループ（:901）は shutdown 中スキップ。**requestShutdown() 時に recoveryAdmissionPending_ = false + pendingRecoveryAdmission_ クリア**（isFullyDrained の `recoveryAdmissionPending == false` を成立）
4. **X6 shutdown 時 quarantine counter 自然収束**: destroyForShutdown（:387）→ destroyQuarantineSlot（:390）→ lifetime().reclaim（:392）。quarantineResidentCount_ は 0 / RetireQuarantineStore::residentCount() は drainAllUnsafe（:376）で 0
5. **X5 shutdown 時 counter 収束**: CoordinatorLoop は isShutdownInProgress() が false の間 processIntent を続行し intentQueue_ を drain。join 後残留は isFullyDrained が false（正常）

**結果**: shutdown 順序・保留再試行・shutdown 相互作用の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-43. 🟡 二十二次レビュー反映（X4-B の commit ownership・Test 9 identity 限定・RuntimePublishAuthority 一切所有禁止）（2026-08-09）

**レビュー観点**: 6.4-X4 詳細改修計画を `ConvoPeq.md` を基準に一次レビュー。**X4-A/X4-B 分離・RuntimeStore 物理 ownership 移動・getCurrent() 非流用は妥当**。実装前に必須修正3点。

**反映**（dash A-2.31 / §6.4-X4-B に反映済み）:
1. **commit 二重化の危険（必須修正1）**: 現行 PublishExecutor（RuntimePublishExecutor.h:42-57）は既に authority.commit() を実行。**案A（publish() が transaction boundary）を採用**: PublishExecutor から authority.commit() を完全削除し、publish() 内部（validate → commit → owner.release() → publishAndSwap() → return previous）に内包
2. **Test 9 は identity equality に限定（必須修正2）**: pointer equality（currentWorld_.load() == runtimeStore.current.load()）を要求しない。PublicationIdentity（sequenceId + publicationEpoch + mappedGeneration）の一致のみ検証
3. **RuntimePublishAuthority は一切所有しない（必須修正3）**: Store / WriteAccess / publishAndSwap 直接呼 / 代替 authority / Store に対する write capability を一切所有しない。**production code から RuntimePublishAuthority::create() 自体を削除**（factory が Store を生成できると INV-X4-3/X4-5 を破る）
4. **Test 6 の write-capable 条件厳密化**: `RuntimeStore<RuntimePublishWorld, Owner>` の write-capable instance について `Owner == RuntimeWorldAuthority` を要求
5. **旧記述削除**: 「getCurrent() が consumeWorldHandle() の置換先になり得る」は既に削除済み（NO-GO 規範のみ維持）

**最終評価**: 設計方針 GO。必須修正3点を反映した dash(8) は **X4-B 実装着手可能**。

### 9-44. 🟡 X1-X6 read path・epoch 判定・rebuild 相互作用調査（2026-08-09 十八次・別視点7）

**調査観点**: X3 の core（EpochDomain の getMinReaderEpoch / tryReclaim / detectStuckReaders）、X1 の isRebuildObsolete と通常 rebuild の相互作用、X2/X4 の read path（makeRuntimeReadHandle / observePublishedWorld）、X3 の ISRRetireRouter 経由の reader 登録を実コードで検証。

**確定事項**（dash A-2.32 / §6 に反映済み）:
1. **X3 getMinReaderEpoch（:199-233）**: quarantined Reader は safe-epoch 計算から除外（:211-215）/ depth==0 は除外（:220-221）/ 最小 epoch を取る。X3 の reclaim 判定の核心。readerRegistrationClosed で新規登録を封じれば minReaderEpoch が新規 reader で下がらない
2. **X3 tryReclaim（:371-381）+ detectStuckReaders（:426-470）**: tryReclaim = deferredDeletionQueue.reclaim(getMinReaderEpoch())。detectStuckReaders は3パス評価（Chronic→Warning→EpochGap）で stuck Reader 検出 → quarantine → getMinReaderEpoch から除外
3. **X3 ISRRetireRouter 経由の readerRegistrationClosed 伝播**: ISRRetireRouter::registerReaderThread（:71）は EpochDomain に委譲。**EpochDomain::registrationClosed_ 設定で router / RCUReader 経由の全登録が自動的に封じられる**。audioThreadRcuReader は EpochDomain を直接 provider に持つ
4. **X1 通常 rebuild と Recovery の相互作用**: isObsolete（:976-978）は build 前（:980）と build 後（:1011）の2箇所。Recovery 消費は通常 rebuild の前に実行され isObsolete チェックの対象外。Recovery は recoveryGeneration = currentRebuildRequestGeneration（:966-967）で暗黙的に最新化
5. **X4 makeRuntimeReadHandle の read path（:3099-3139）**: acquireReadToken(runtimeStore) + consumeWorldHandle（:3116-3117）+ 単調性監視（:3128-3135）。X4-B-9 の置換対象は read token 2箇所、単調性監視は AudioEngine 側で維持。呼び出し元多数
6. **X1 isRebuildObsolete（:2464）**: generation != currentRebuildRequestGeneration の単純不一致判定

**結果**: read path・epoch 判定・rebuild 相互作用の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-45. 🟡 二十三次レビュー反映（RecoveryAdmissionClosed・X2 timeout semantics・X4 swap failure・Phase 0・必須 Acceptance Criteria）（2026-08-09）

**レビュー観点**: 最新ソース `ConvoPeq(20260809-022629).md` を基準に、dash の最新版設計（X1〜X6 / X4-A・X4-B）を `Practical Stable ISR Bridge Runtime.md` の設計原則と照合。

**総合判定**: **P2-1〜P2-4 = 実装GO / X5 = 実装GO / X1/X2/X3/X6 = invariant 固定後実装GO / X4 = 設計GO（段階実装必須）**。

**新規反映**（dash A-2.33 / §6 に反映済み）:
1. **X1 RecoveryAdmissionClosed（§4.3）**: `recoveryAdmissionPending_` だけでは不十分。`AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped` を shutdown state machine に含める（build gap 中の isFullyDrained 早期 true 防止）
2. **X2 timeout semantics（§6.1）**: `timeout ≠ publish failure`。timeout を failure と誤解して rollback すると double ownership / double publish。lifecycle を `Allocated → Transferred → Committed → Completed` として固定
3. **X4 swap failure（§20）**: `swap failure is architecturally impossible / handled` を acceptance criterion に。publishAndSwap は単一原子 exchange で失敗しない。null→null swap は異常（validate で事前検出）
4. **Phase 0（§22）**: invariant/specification freeze を最優先。実装順序を Phase 0〜Phase 7 に明確化（X4 は X2/X1/X3 の意味確定後に触る）
5. **必須 Acceptance Criteria 表（§23）**: X1-5/X1-6/X2-6/X3-4×2/X4-3/X4（3項目）/X5/X6（2項目）を architectural test に必須化

**最終評価**: 改修案は採用可能。**X1/X2/X3/X4/X6 の invariant をコード化・テスト化してから実装する**という条件付き GO。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak。

### 9-46. 🟡 X1-X6 Bridge 型・物理削除・StateOwner ledger・buildSource 供給調査（2026-08-09 十八次・別視点8）

**調査観点**: X4 の core Coordinator の Bridge 型（RuntimePublicationBridge）、X3 の DSPLifetimeManager 物理削除（enqueueWithRetry / destroyRolledBackDSP）、X2 の StateOwner ledger、X1 の currentBuildSnapshot_ 供給を実コードで検証。

**確定事項**（dash A-2.34 / §6 に反映済み）:
1. **X4 RuntimePublicationBridge（:3446-3500）**: validate / didPublish / willRetire / retire を担う。**X4-B 後も Bridge は残る**（didPublish/willRetire/retire を PublishExecutor の Execution tail から呼ぶ）。publish() は validate に Bridge を使うが、didPublish/willRetire/retire は publish() の外。Bridge は AudioEngine 側に残る
2. **X3 reclaim と物理削除の分離**: 物理削除（DSPCore* delete）は retire path の enqueueWithRetry（DSPLifetimeManager.cpp:49, destroyDSPCoreNode を deferred delete）が担当。reclaim は slot 遷移 + free list 返却のみ。X3 の reclaim は物理削除を含まない
3. **X2 StateOwner ledger**: onSubmitted/onBuilt/onValidated/onPublished/onRetired/onReclaimed 等を記録。**X2 の completion watermark と独立**（診断 ledger）。onPublished は trySubmitImpl で記録、onPublishCommitted とは別タイミング
4. **X1 currentBuildSnapshot_ 供給**: enqueuePublicationIntentForRuntimeCommit が sealedSnapshot 受領時に更新（Commit.cpp:704-705）。初期状態 sealed=false の Recovery は Builder の :917 チェックで skip（正しい動作）。X1 の durable admission は sealed=false を入れる前に検証すべき

**結果**: Bridge 型・物理削除・StateOwner ledger・buildSource 供給の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-47. 🟡 二十四次レビュー反映（isFullyDrained measurement predicate・pendingIntentCount_ 命名・X4 dual-pointer 暫定正常状態）（2026-08-09）

**レビュー観点**: 情報源 `ConvoPeq(20260809-022629).md` を基準コードとして再参照し、dash(10) を ISR/Immutable Snapshot Runtime/RCU/ownership/lifetime/publication ordering/shutdown の観点から検証。

**総合判定**: **P2-1〜P2-4 = GO / X5 = GO / X1/X2/X3/X6 = 条件付きGO / X4-A = GO / X4-B = GO（段階実装）**。一括実装 = NO-GO。Phase 0 の invariant freeze を実施してから P2→X5→X6→X2→X1→X4→X3 の順に実装（条件付き承認）。

**新規反映3点**（dash A-2.35 / §6 に反映済み）:
1. **isFullyDrained() を単独の truth source にしない**: `ShutdownPhase + ProducerQuiescence + AdmissionClosed + RecoveryAdmissionClosed + BuilderStopped + isFullyDrained()` を組み合わせる。**isFullyDrained() は measurement predicate であり shutdown authority そのものではない**（§6.7）
2. **pendingIntentCount_ の命名・コメント固定**: Observe+Quarantine+Recovery の queue residency + producer reservation（Publish と RetireIntent 除外）。将来は transportIntentResidency_ 等への改名を検討。コードコメントで「This counter excludes Publish and RetireIntent.」を固定（§1.1）
3. **X4 dual-pointer を「暫定正常状態」として明示**: `X4-B: write authority singularization / Future: read-source singularization` と分離。publish transaction 完了後 INV-X4-6（同一 PublicationIdentity）を保証するため正常動作として許容（§6.4-X4-B）

**最終評価**: dash(10) は改修方向として妥当。**「設計方針として採用可能。Phase 0 の invariant freeze を実施してから P2→X5→X6→X2→X1→X4→X3 の順に実装する」という条件付き承認**。

**次回の実装手順**: Phase 0（invariant/specification freeze）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak。

### 9-48. 🟡 X1-X6 receipt 状態・テスト基盤・admission 判定調査（2026-08-09 十八次・別視点9）

**調査観点**: X2 の pendingReceipt_ / markReceiptReclaimComplete の関係、X1 の既存テスト基盤（testRecoveryRequestEnqueueAndPop）、X5 の PublicationAdmission::evaluate（admission 判定と counter の関係）を実コードで検証。

**確定事項**（dash A-2.36 / §6 に反映済み）:
1. **X2 pendingReceipt_ と PublishReceiptWaiter の区別**: AudioEngine に2種類の receipt が存在。receipt #1 `pendingReceipt_`（:4683）= Timer の retire 用（storeReceipt:1157 / resetReceipt:1176 / retirePublishedDSP:Timer.cpp:1774 / markReceiptReclaimComplete:ProcessIntent.cpp:41）。**X2 の completion watermark と無関係**。receipt #2 `PublishReceiptWaiter`（:3613-3635）= X2 の completion watermark（onPublishCommitted → notifyPublishReceipt → complete()）。**X2 の設計は receipt #2 のみを対象**
2. **X1 testRecoveryRequestEnqueueAndPop（:609-624）**: `submitRecoveryRequest → popRecoveryRequest` の 1-hop transport のみ検証（buildSource.sealed=true、2回目 pop が null = 1-hop 保証）。**X1 の拡張は queue full → durable 化 → takePendingRecoveryAdmission の流れを追加**
3. **X5 admission 判定と counter**: PublicationAdmission::evaluate（:6-61）は5段階の admission（Shutdown/Generation/HealthState/Pressure/Fading→Deferred）。**X5 の counter は admission Accepted 後の enqueue で増加**。Rejected/Deferred は counter に影響しない。Deferred は単一スロット（再 enqueue 時に counter +1）

**結果**: receipt 状態・テスト基盤・admission 判定の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-49. 🟡 X1-X6 ReaderSlot 構造・destroyForShutdown・intentId 採番・IR 転送調査（2026-08-09 十八次・別視点10）

**調査観点**: X3 の ReaderSlot 構造（EpochDomain.h:531-547）、X6 の DSPQuarantineManager::destroyForShutdown（ISRDSPQuarantine.cpp:130-155）、X1 の nextRecoveryIntentId_ 採番（ISRRuntimePublicationCoordinator.cpp:657）と RuntimeBuilder の IR 転送（RuntimeBuilder.cpp:447）を実コードで検証。

**確定事項**（dash A-2.37 / §6 に反映済み）:
1. **X3 ReaderSlot 構造（:531-547）**: epoch / depth / enterCount / residencyStartTimestampUs / ownerThreadId / ownerTag / quarantineFlags（0x01 quarantined / 0x02 pending）。**closeReaderRegistration() は新規登録のみ拒否し、既存 ReaderSlot は解放しない**（exitReader で epoch = kInactiveEpoch に戻るのみ）。graceful drain は activeReaderCount()==0（全 slot depth==0）を待つ。quarantineFlags(0x01) の Reader は getMinReaderEpoch から除外。X3 の isQuiescent = readerRegistrationClosed AND activeReaderCount == 0
2. **X6 destroyForShutdown（:130-155）**: quarantineActiveFlags_[slot] を false（:141）→ auditLog の未解決エントリを resolved に（:146-151）→ compactAuditLogLocked（:152）。**quarantineResidentCount_（= residentCount() の走査）が自然に 0 へ**。X6 の新たな shutdown 処理は不要
3. **X1 nextRecoveryIntentId_ 採番と IR 転送**: nextRecoveryIntentId_（:435, atomic）は submitRecoveryRequest（:657）で fetch_add(1, relaxed)。**durable 化後も採番は継続**。IR 転送（RuntimeBuilder.cpp:447 transferIRStateFrom(engine.getConvolverProcessor())）は build 時に現在の UI processor から取得。**buildSource は IR データを内包しない**（metadata/fingerprint のみ）ため durable admission は軽量。coalesce 後も IR 実体は build 時に転送されるため stale IR の懸念なし

**結果**: ReaderSlot 構造・destroyForShutdown・intentId 採番・IR 転送の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-50. 🟡 二十五次レビュー反映（INV-ISR-01〜07・X1 ShutdownDiscard・X3 意味論先行固定・最終判定）（2026-08-09）

**レビュー観点**: `ConvoPeq(20260809-022629).md` を基準コードとして、REPAIR_PLAN2-dash を L1〜L4（C++/Thread Safety → Queue/Counter/State Machine → ISR/RCU/Lifetime → Architectural Authority Singularization）の4層で検証。

**総合判定**: **P2-1〜P2-4 = GO（そのまま実装してよい）/ X1/X2/X6 = 条件付きGO / X3 = GO / X4 = GO（段階実装）/ X5 = GO**。Recovery coalesce / force reclaim = NO-GO（撤回が正しい）。一括実装 = NO-GO。

**新規反映**（dash A-2.38 / §6 に反映済み）:
1. **INV-ISR-01〜07（§23・最上位 ISR 不変条件）**: INV-ISR-01（isFullyDrained 完全条件）/ 02（pendingIntentCount_ = residency+reservation）/ 03（semantic 混同禁止）/ 04（ShutdownQuiescent は readerRegistrationClosed 必須）/ 05（committed ≠ completed）/ 06（currentWorld_ は non-owning）/ 07（dual pointer identity consistency 検証可能）— §6 冒頭に追加
2. **X1 shutdown discard を「ShutdownDiscard」として明示（§8.1）**: `Recovery lost` と `ShutdownDiscard` を同じ意味にしない。Running 中の queue full → durable pending = loss ではない（INV-5 保証が機能）。Shutdown 中の durable pending → explicit lifecycle discard。**Telemetry 上で2つを分ける**
3. **X3 の意味論を X4 より先に固定（§22）**: Lifetime correctness（X3）を先に固定してから Publication authority topology（X4）を変更。X3 の意味論（INV-X3-4 / INV-ISR-04）は Phase 0 で X4 より先に固定
4. **X2 の「CAS 化すれば安全」は誤り**: completion semantic が重要（既に INV-X2-6 で固定済み）

**残余リスク（§14）**: X4-B 完了後も currentWorld_ / RuntimeStore::current の dual publication/read surface は残る。X4-B 完了 = publication write authority の Singularization（read-source singularization は Future）。

**最終評価**: **「P2-1〜P2-4 は実装GO。X1〜X6 は設計として概ね正しいが、段階実装が必要。特に X3 の lifetime closure と X4 の dual publication surface を最終的な ISR Architecture Review で再確認すること」**。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1 → X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak。

### 9-51. 🟡 X1-X6 cv 動作・retire 実装・BuildError 種類調査（2026-08-09 十八次・別視点11）

**調査観点**: X2 の PublishReceiptWaiter の cv 動作詳細、X3 の ISRRetireRouter::retire 実装、X1 の RuntimeBuilder::build の BuildError 種類、X4 の read API 現状を実コードで検証。

**確定事項**（dash A-2.39 / §6 に反映済み）:
1. **X2 cv 動作（:3613-3635）**: complete は mutex 下で lastCompleted_ 更新 → cv_.notify_all()。waitFor は `cv_.wait_until(lock, deadline, [&]{ return seqId <= lastCompleted_; })`。**wait_until は predicate 付きのため lost wakeup 安全**（notify 前に waitFor が始まっても即復帰）。deadline 到達後 predicate が false なら false
2. **X3 ISRRetireRouter::retire（:149-158）**: enqueueWithRetry(ptr, deleter, currentEpoch(), Generic) に委譲。enqueueWithRetry（:161-179）は通常 enqueue → 失敗時 tryReclaim → enqueue を最大2回リトライ。QueuePressure 以外（Shutdown 等）は即時終了。**reclaim と retire は独立**（X3 は reclaim 側のみ変更）
3. **X1 BuildError 種類（:55-71）**: InvalidInput/ResourceUnavailable/MKLFailure/ConvolverFailure/PrepareFailure/WarmupFailed/InternalError。**一時的 failure は次サイクルで自然に再試行**、永続的 failure は drop 相当。**X1 の新たな retry ループは不要**
4. **X4 read API 現状**: RuntimeWorldAuthority に read API は未実装。現状は全て RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore) を直接呼ぶ。**X4-B-9 で専用 read API に置換**

**結果**: cv 動作・retire 実装・BuildError 種類の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-52. 🟡 X1-X6 Execution tail・compactAuditLog・DeferredDeletionQueue reclaim 調査（2026-08-09 十八次・別視点12）

**調査観点**: X2 の DSPTransition（publish-completion の execution tail）、X6 の compactAuditLogLocked（ISRDSPQuarantine.cpp:158-172）、X3 の DeferredDeletionQueue::reclaim（DeferredDeletionQueue.h:108-119）を実コードで検証。

**確定事項**（dash A-2.40 / §6 に反映済み）:
1. **X4 Execution tail の3構成要素**: tail-1 `ctx.transition.onPublishCompleted(...)`（DSPTransition.h:49-90）= DSP activate/crossfade/retire（Crossfade Registration Authority・registerCrossfade は DSPTransition のみ）。tail-2 `advanceRetireEpoch()` = EBR。tail-3 `onPublishCommitted(seqId)` = X2 の completion。**tail-1 と tail-3 は独立**。X4-B の publish() は tail を含まない
2. **X6 compactAuditLogLocked（:158-172）**: kCompactThreshold = 1024（:161）、resolved エントリが1024超えた場合のみ compaction。先頭の resolved 連続を削除（:167-171）。auditLog_ は vector。X6 の新規介入は不要
3. **X3 DeferredDeletionQueue::reclaim（:108-119）**: isOlder(entry.epoch, minReaderEpoch) のエントリのみ deleter 実行。**FIFO 前提**（先頭が不安全なら break）。isOlder（:399-402）= static_cast<int64_t>(a-b) < 0（wraparound 対応）。X3 の reclaim は slot 遷移、物理削除は DeferredDeletionQueue::reclaim が epoch 安全に実行。ShutdownQuiescent では全 Retire が安全判定

**結果**: Execution tail・compactAuditLog・DeferredDeletionQueue reclaim の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-53. 🟡 X1-X6 通常 rebuild 後半・quarantineReader・crossfade decision 調査（2026-08-09 十八次・別視点13）

**調査観点**: X1 の通常 rebuild 後半処理（RebuildDispatch.cpp:1024-1138）、X3 の quarantineReader / unquarantineAllReaders / verifyReaderInvariants（EpochDomain.h:264-367）、X2 の trySubmitImpl の crossfade decision と deferred の関係を実コードで検証。

**確定事項**（dash A-2.41 / §6 に反映済み）:
1. **X1 通常 rebuild 後半**: IR rebuild（:1039, rebuildAllIRsSynchronous）→ Warmup（:1051, validateWarmup + retryable 判定 :1054）→ refreshLatency + fadeIn（:1085,1088）→ 投影値更新（:1104-1115）→ Commit（:1138, enqueuePublicationIntentForRuntimeCommit）。**通常 rebuild と Recovery は同一 commit 関数（:1138 vs :971）**。X1 の Recovery 消費はこの後半の前に実行
2. **X3 quarantineReader / unquarantineAllReaders / verifyReaderInvariants**: quarantineReader（:264-311, depth==0 即座 quarantine / depth>0 pending→exitReader で昇格）。unquarantineAllReaders（:313-321, 全 slot flags 0 に）。verifyReaderInvariants（:338-367）。**X3 の readerRegistrationClosed は quarantine と独立**（quarantine は stuck 解除、registrationClosed は新規登録封鎖）
3. **X2 trySubmitImpl の crossfade decision と deferred**: cfDecision = CrossfadeAuthority::evaluate（:193-206）→ needsCrossfade なら transitionActive（:221-227）→ executor_.publish（:263）。**crossfade decision は Deferred の発生源**（hasFadingRuntimeInWorld → DeferredFadingActive → deferredSlot_ 退避）。**deferred 中は新 publish の completion は発生しない**（re-enqueue 後）

**結果**: 通常 rebuild 後半・quarantineReader・crossfade decision の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-54. 🟡 二十六次レビュー反映（X1 lease 方式・X3 INV-X3-5・X4 INV-X4-A/B/C）（2026-08-10）

**レビュー観点**: `REPAIR_PLAN2-dash(20260810-002710)` を最新版として、`ConvoPeq.md` の最新ソースコードと照合して再評価。前回版よりかなり完成度が上がり、**X4-B は条件を満たせば実装GO できる設計に到達**。ただし実装前に修正すべき重要論点2つ。

**総合判定**: **P2-1〜P2-4 = GO / X5 = GO / X6 = 条件付きGO / X2 = 条件付きGO / X3 = 条件付きGO / X4-B = GO（strict acceptance criteria）/ X1 = 修正後GO**。

**必須修正2点・強く推奨1点**（dash A-2.42 / §6 に反映済み）:
1. **X1 の Pending/Building 矛盾（必須修正1・§9-13）**: `takePendingRecoveryAdmission()` が state クリア（destructive dequeue）なのに build 失敗で「Pending 維持」という矛盾。**take を lease（state transition）に変更**: PendingRecoveryAdmission に `State` enum（NoAdmission/DurablePending/Building）を追加。take は DurablePending → Building へ遷移（クリアしない）。build 失敗（transient）は Building → DurablePending へ戻す。obsolete は Discarded。build success は PublishTransport。**INV-X1-1（exactly one durable state）が lease 方式で常に成立**
2. **X3 の reclaimInFlightCount_ 近似 counter（必須修正2・§16-17）**: `reclaimInFlightCount_ == 0` だけで shutdown drain を判定しない。**INV-X3-5 を追加**: ShutdownQuiescent completion requires `pendingReclaimHandles_.empty() AND reclaimInFlight == 0`。`pendingReclaimHandles_`（:4616, mutex 保護）が reclaim pending の実際の source of truth。isFullyDrained に pendingReclaimHandles.empty() を追加
3. **X4 の INV-X4-A/B/C（強く推奨・§29）**: `currentWorld_ = observation-only` / `RuntimeStore::current = sole physical RuntimeWorld source` / `No RT API may derive RuntimeWorld ownership/lifetime from currentWorld_`。**Audio Thread は currentWorld_ を RuntimeWorld 取得元として使わない**

**最終判定**: **「設計の骨格は妥当で、P2 と X5 は実装開始可能。X4-B も今回の修正で実装GO まで到達。ただし X1 の Pending/Building 状態矛盾と X3 の pending reclaim accounting は、ISR shutdown correctness に直接関係するため、実装着手前に必ず修正する」**。

**次回の実装手順**: Phase 0（invariant/specification freeze・INV-ISR-01〜07 / INV-X3-5 / INV-X4-A〜C 含む）→ P2-1〜P2-4 → X5 → X6 → X2 → X1（lease 方式）→ X4（X4-A→X4-B-0〜B-11）→ X3 → 統合 shutdown/soak。

### 9-55. 🟡 X1-X6 epoch 取得元・queue capacity・enterReader 詳細・kMaxSlots 調査（2026-08-10 十八次・別視点14）

**調査観点**: X1 の submitRecoveryRequest の epoch 取得（currentWorld_ 由来）、X5/X6 の全 transport queue capacity、X3 の enterReader/exitReader 詳細、X6 の kMaxSlots を実コードで検証。

**確定事項**（dash A-2.43 / §6 に反映済み）:
1. **X1 epoch 取得元（:650-652）**: submitRecoveryRequest は `consumeAtomic(currentWorld_)` → `world->publication.epoch` で epoch 取得。**PendingRecoveryAdmission.epoch は currentWorld_ から取得**。X4 の INV-X4-A/C との整合: NonRT（CoordinatorLoop）から currentWorld_ を metadata observation（epoch 取得）として使用するのは正当
2. **X6 queue capacity**: intentQueue_ = 4096（MPSC）/ quarantineFallbackQueue_ = 1024 / recoveryIntentQueue_ = 256（SPSC）/ observeDeferredRing_ = 1024。各 X の対象 queue と capacity が確定
3. **X3 enterReader / exitReader**: enterReader（:106-130）は epoch を depth++ より先に store（BUG-050）→ depth++。ネスト時は epoch 再設定なし。exitReader（:133-168）は depth-- → 0 で epoch = kInactiveEpoch → pending quarantine（0x02）昇格（CAS）。**enter/exit は registrationClosed の影響を受けない**（既存 Reader の enter/exit は shutdown 中も継続可能）
4. **X6 kMaxSlots = 256**（ISRDSPQuarantine.h:36）: residentCount() の走査は 256 要素（NonRT で許容）。QuarantineReason::ReceiptReset は X2 の receipt #1 と関係。X6 の設計で kMaxSlots 変更は不要

**結果**: epoch 取得元・queue capacity・enterReader 詳細・kMaxSlots の観点からも X1〜X6 が確定。残る未確定事項なし。

### 9-56. 🟡 X1-X6 evaluateDeferred・QueuePressure 移送・read API 置換対象調査（2026-08-10 十八次・別視点15）

**調査観点**: X2 の evaluateDeferred の stale-discard 判定（PublicationAdmission.cpp:69-91）、X3 の enqueueWithRetry の QueuePressure 移送（ISRRetireRouter.cpp:182-203）、X4 の read API 置換対象一覧を実コードで検証。

**確定事項**（dash A-2.44 / §6 に反映済み）:
1. **X2 evaluateDeferred（:69-91）**: 1. Shutdown → Discard（ShutdownDiscard）/ 2. TTL（30s）超過 → Discard（StaleDiscard）/ 3. Generation 不一致 → Discard（StaleDiscard）/ 4. Sequence 後戻り → Discard（StaleDiscard）→ Ready。**deferred の cancel 条件が確定**。StaleDiscard された deferred の seqId は re-enqueue されない → completion は発生しない（INV-X2-6 の deferred 例外と整合）
2. **X3 enqueueWithRetry の QueuePressure 移送（:182-203）**: QueuePressure/QueueFull 時は m_retireQuarantine.quarantine（:190-192）で RetireQuarantineStore へ移送。**queue full は RT 参照中の可能性が高いため即時解放は UAF**。store full 時は delete を絶対しない（:195-199）。Future: runtimeHealth_->notifyQueuePressure（:202）
3. **X4 read API 置換対象**: RuntimeWorldAuthority に read API は未実装（X4-B-9 で新設）。現状は RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore) 等を直接呼ぶ（AudioEngine.h:1331/2119/3116/3383/3691 等）。**X4-B-9 は worldAuthority().readAPI() に一括置換**（getCurrent() は置換先にしない）。単調性監視は AudioEngine 側で維持

**結果**: evaluateDeferred・QueuePressure 移送・read API 置換対象の観点からも X1〜X6 が確定。残る未確定事項なし。

---

## 10. 別視点調査（2026-08-11）— 未調査領域・未確定事項の確定

**調査観点**: セクション 8「未調査領域」の未着手項目と、各バグの未確定設計判断（producer 実装 / デッドコード削除 / CMake フラグ）を、スレッド所有権・メモリオーダリング・例外安全性・RT 到達可能性の別視点から再調査・確定。

**使用ツール**: AiDex（411ファイル/6054 methods インデックス）、cocoindex (ccc)、semble、graphify、WSL rg/sed/awk、serena

### 10-1. セクション 8 の未着手項目 — 確定結果

| 未調査領域 | 確定結果 | 根拠 |
|---|---|---|
| `src/core/ThreadAffinityManager.h`（293行） | **問題なし（調査完了・確定）** | CPU affinity 管理は起動時 1 回の `initialize()` / `detectCoreTopology()` のみ。`applyCurrentThreadPolicy()` の `AudioRealtime` は MMCSS に委譲し SetThreadPriority を呼ばない（二重適用防止）。`std::countr_zero`（C++20）、防御的チェック（Mask!=0 / GroupCount==1 / physicalCoreCount 整合）済み。**RT スレッドからは呼ばれない**（AudioThread affinity は Timer.cpp の applyMmcssPriority が担当） |
| `EqProcessor::reset()` RT到達可能性 | **デッドコード化を確定**（前回から持ち越しの確定） | `doc/work89/INTEGRATED-BUG-LIST.md` 8.2 で全数確認済み: `DSPCore::reset()`（DSPCoreLifecycle.cpp:335）の呼び出し元ゼロ、`EQProcessor::reset()`（EQProcessor.Core.cpp:259）の呼び出し元ゼロ（grep 全ソース + git 履歴 + テスト）。UI リセットは全て `resetToDefaults()`（EQProcessor.Core.cpp:206）経由で、rt シャドウ変数に触れない安全パターン。**`EQProcessor::reset()` は実質デッドコード** |
| `ConvolverProcessor.Rebuild.cpp` | 調査完了（10-2 参照） | 別視点調査の結果を 10-2 に記載 |
| `src/core/EpochDomain.h` | 部分調査済み → 確定 | 9-49（ReaderSlot 構造）/ 9-53（quarantineReader）/ 9-55（enterReader/exitReader 詳細）で確定済み |

### 10-2. `ConvolverProcessor.Rebuild.cpp` の別視点調査

**ファイル概要**: `rebuildAllIRs()` / `postCoalescedChangeNotification()` / `rebuildAllIRsSynchronous()` / `runIncrementalBuildStep()` / `runIncrementalFinalizeStep()` / `setUseIncrementalRebuild()` / `invalidatePendingLoads()` を含む（全 296 行）。

**実行経路の事前整理（最重要の構造的事実）**:

| 関数 | 行 | 実行スレッド | 呼び出し元 | 状態 |
|---|---|---|---|---|
| `rebuildAllIRs()` | 13 | Message Thread | `executePendingCommit` 内の callAsync（LoadPipeline.cpp:814） | **LIVE** |
| `postCoalescedChangeNotification()` | 21 | 任意（内部で callAsync） | 多数（Runtime/StateAndUI/LoadPipeline） | **LIVE** |
| `rebuildAllIRsSynchronous()` | 44 | Rebuild Thread | `rebuildThreadLoop` 3箇所（RebuildDispatch.cpp:953/1031/1131） | **LIVE** |
| `IncrementalRebuildJob::reset()` | 110 | — | `setUseIncrementalRebuild`/`invalidatePendingLoads`/`rebuildAllIRsSynchronous` 経由 | **呼ばれるが実質 no-op** |
| `runIncrementalBuildStep()` | 136 | — | **呼び出し元ゼロ** | **DEAD** |
| `runIncrementalFinalizeStep()` | 237 | — | **呼び出し元ゼロ** | **DEAD** |
| `setUseIncrementalRebuild()` | 279 | — | **呼び出し元ゼロ** | **DEAD** |
| `isIncrementalRebuildEnabled()` | 286 | — | **呼び出し元ゼロ** | **DEAD** |
| `invalidatePendingLoads()` | 291 | Message Thread | PrepareToPlay.cpp:287 | **LIVE（だが no-op）** |

**最重要の構造的事実**: `rebuildJob` はソースコード上**どこにも確保されていない**（`rebuildJob = std::make_unique<IncrementalRebuildJob>()` は src 配下にゼロ件）。`beginIncrementalRebuild` / `advanceIncrementalRebuild` / `resetIncrementalRebuild` はヘッダ（ConvolverProcessor.h:491-493）に**宣言のみで未定義**。したがって **incremental rebuild サブシステム全体は完全に休眠状態**。

**別視点分析**:
1. **スレッド所有権**: `rebuildAllIRsSynchronous()` は Rebuild Thread 専用（M-1 コメント明記・3 呼び出し元は全て rebuildThreadLoop）。`rebuildAllIRs()` は Message Thread 専用。`rebuildJob` は**非 atomic のプレーンなメンバ**（ConvolverProcessor.h:889）で、仮に incremental を有効化すると `invalidatePendingLoads`（Message Thread）と Rebuild Thread のアクセスが**データ競合**（現状は rebuildJob==null で良性）
2. **データ競合**: `changeNotificationPending` の `exchangeAtomic(acq_rel)` → `publishAtomic(release)` は HB を正しく構成。`currentSampleRate` / `currentBufferSize` / `currentIRScale` / `isLoading` の acquire/release 対応は全て整合的。**競合なし**
3. **例外安全性**: `rebuildAllIRsSynchronous` は noexcept でない。例外は `rebuildThreadLoop` の try/catch（RebuildDispatch.cpp:819-1237）で捕捉され**クラッシュしないがリソースを失う**形態（P3-1 参照）
4. **リソースリーク**: `acquireIRState()` / `releaseIRState()` のペアリングは正しい（二重解放なし、遅延 retire 方式）。`runIncrementalBuildStep` の `retireStereoConvolver`（L219）は `retired` フラグで二重 retire 防止済み。**ただし `IncrementalRebuildJob::reset()`（L114-115）は `~StereoConvolver() + aligned_free` で直接破棄し NUC エンジンをリーク**（R-新規A）
5. **WeakReference / ライフタイム**: `postCoalescedChangeNotification` の `weakThis` と callAsync 失敗時の `publishAtomic(pending,false)` 復帰は正しい（lost notification レースは正常動作時なし）。`executePendingCommit` の `weakThis`（LoadPipeline.cpp:719）も破棄後安全
6. **IncrementalRebuildJob 状態機械**: `Stage::Prepared` / `FinalizingPrepare` を生成するコードが存在しない（`runIncrementalBuildStep` は Building と FinalizingApply のみ遷移）。`advanceIncrementalRebuild` 未定義のため駆動ループなし（R-新規B）

**確定**: 実行経路（rebuild / 通知）は堅牢で、データ競合・通知消失・二重解放は検出されず。**主要な問題は「休眠中の incremental rebuild サブシステムの設計未完」**（R-新規A/B）と「実行経路上の例外時リーク」（R-新規C、OOM 限定）。**`invalidatePendingLoads` が live 呼び出しされている点に注意**（現状 no-op だが、将来 `rebuildJob` を確保・有効化する際は `reset()` のリーク修正と排他アクセスが必須）。

#### 10-2-1. 新規バグ（R-新規A〜E）— 別視点調査で確定

| 提案ID | 重要度 | ステータス | 内容 |
|---|---|---|---|
| **R-新規A** | Medium (P2) | ✅ Confirmed（潜伏） | `IncrementalRebuildJob::reset()`（Rebuild.cpp:110-134）が `pendingConv` を `~StereoConvolver() + aligned_free` で直接破棄 → **NUC エンジンリーク**。`StereoConvolver` デストラクタは空（NUC 解放は `destroyStereoConvolver()` に実装）。正しくは `retireStereoConvolver(pendingConv, 0)`。現状はデッドコード経路のため未発火（Debug ビルドでは `~StereoConvolver` 内の `jassert(nucConvolvers[0]==nullptr)` も発火） |
| **R-新規B** | Medium (P2) | ✅ Confirmed | **Incremental rebuild 一式が未接続**: `rebuildJob` 未確保・`beginIncrementalRebuild`/`advanceIncrementalRebuild`/`resetIncrementalRebuild` は宣言のみ未定義（将来のリンクエラー要因）・`runIncrementalBuildStep`/`runIncrementalFinalizeStep`/`setUseIncrementalRebuild`/`isIncrementalRebuildEnabled` は呼び出し元ゼロ・Stage 状態機械は `Prepared`/`FinalizingPrepare` 未生成で不完全。Rebuild.cpp:136-296 / ConvolverProcessor.h:491-493,579-587 |
| **R-新規C** | Low (P3) | ✅ Confirmed | `rebuildAllIRsSynchronous`→`runSynchronously`→`applyNewState(async=false)` で **bad_alloc 時に新エンジンがリーク**（例外は rebuildThreadLoop に握り潰されサイレント、OOM 時のみ）。Rebuild.cpp:93 / LoaderThread.cpp:367-383 / LoadPipeline.cpp:680 |
| **R-新規D** | Low (P3) | ✅ Confirmed（コメント不整合） | `executePendingCommit` の「Message Thread のみ」契約コメントが **Rebuild Thread からの同期実行**（`applyNewState(async=false)`）と不一致。機能バグなし・文書修正推奨。LoadPipeline.cpp:766 |
| **R-新規E** | Info | ✅ Confirmed（確定事項） | `setUseIncrementalRebuild` の「常に false」過去バグは修正済み（L281 で enable を正しく使用）。`postCoalescedChangeNotification` の合体パターン・atomic 整合・IRState ペアリングは問題なし（監査結果として記録） |

**修正方針**（R-新規A〜E）:
- **R-新規A**: `IncrementalRebuildJob::reset()` の `pendingConv` 破棄を `retireStereoConvolver(pendingConv, 0)` に変更（二重 retire フラグの `retired` exchange と整合）。`reset()` は noexcept のまま維持
- **R-新規B**: incremental rebuild を「現状は休眠（将来拡張）」と明示するコメントを ConvolverProcessor.h:491-493 に追加。`beginIncrementalRebuild` 等の未定義関数は、削除するか「将来実装予約」コメントを付与（**呼び出し元ゼロなので削除も可**。削除しない場合は未定義のまま放置せず、`[[noreturn]]` か `jassertfalse` のスタブを置いてリンクエラーを防ぐ）
- **R-新規C**: `runSynchronously`（LoaderThread.cpp:367-383）の `applyNewState` / `make_unique` を try/catch で包み、失敗時に `conv` を `retireStereoConvolver` する（OOM 時のサイレントリーク防止）
- **R-新規D**: `executePendingCommit`（LoadPipeline.cpp:766）のコメントを「Message Thread または Rebuild Thread（applyNewState async=false の同期経路）から呼ばれる。内部は全てスレッド安全」に修正
- **R-新規E**: 監査結果として記録（変更不要）

**リスク**: R-新規A/B はデッドコード経路のため現状実害なし（将来の有効化時に顕在化）。R-新規C は OOM 時のみ。**対応は R-新規A（4行）と R-新規D（コメント）が低コストで即実施可、R-新規B/C は設計判断**。

### 10-3. 各バグの現状再確認（別視点）

#### 10-3-1. Bug 1-2 `coordinatorDeferredRing_` / `lastResortQueue_` デッドコード — 方針確定

**現状再確認（2026-08-11）**: `coordinatorDeferredRing_`（:515）/ `lastResortQueue_`（:518）/ `coordinatorDeferredCount_`（:516）/ `lastResortCount_`（:519）は依然として消費のみ（drainOverflowRing の pop / compaction のみ）で push/producer なし。

**別視点の確定**（REPAIR_PLAN2-dash.md:4798 で設計方針確認）:
- X6 の設計で **overflow ring（RetireOverflowRing / coordinatorDeferredRing_ / lastResortQueue_）は retire intent の overflow 用**であり、X6 では設計変更なし（producer 追加なし）
- `drainOverflowRing` の再注入は LifetimeState 側（`emitRetireIntent`）で Coordinator の `pendingIntentCount_` と独立

**確定方針**:
1. **`lastResortQueue_` の値初期化（`RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity]{};`）は安全に実施可** — UB の芽（未初期化配列）を除去
2. **producer は実装しない**（X6 設計方針と整合）— `coordinatorDeferredRing_` / `lastResortQueue_` は「retire intent overflow 用の将来予約領域」として維持し、コードコメントで到達不能であることを明示
3. **削除はしない**（drainOverflowRing のロジックと X1-X6 の将来設計に依存するため）

#### 10-3-2. Bug 1-5 `musicalSoftClip` デッドコード — 削除方針確定

**現状再確認（2026-08-11）**: `AudioEngine::DSPCore::musicalSoftClip()`（AudioEngine.h:1068 宣言 / DSPCoreIO.cpp:341 定義）は呼び出しゼロ。実際の処理は各ファイルローカル `musicalSoftClipScalar()`（DSPCoreIO.cpp:95 / DSPCoreFloat.cpp:165 / DSPCoreDouble.cpp:107）が使用。

**確定方針**: **クラスメソッド `musicalSoftClip()` を削除**（ローカル関数 `musicalSoftClipScalar` が全パスで使用済みのため、削除してもビルド・挙動に影響なし）。将来 Double パスと Float パスのサチュレーション統一（Bug 1-4 の fastTanh 統合）を行う際に、`musicalSoftClipScalar` 自体も `dsp/math/` への共通化を検討。

#### 10-3-3. Bug 2-8 `cleanup()` の「強制削除」未実装 — 役割分担確定

**現状再確認（2026-08-11）**: `cleanup()`（LoadPipeline.cpp:571）は「終了したスレッドのみ削除（`waitForThreadToExit(0)` は非ブロック）」。`forceCleanup()`（StateAndUI.cpp:969）は `loader->stopThread(500)` で強制停止。コメント「強制削除は行わない（stopThreadによる...）」明記。

**確定**: **現状の役割分担は意図どおり正常**。`cleanup()` は破棄時（デストラクタ経由）の安全な回収、`forceCleanup()` は明示的な強制停止（R-37 の loaderTrashBin dangling 修正と連動）。**Bug 2-8 の「cleanup() にタイムアウト付き強制終了を追加」は不要**（二重停止のリスクを回避するため、現状維持が正しい）。

#### 10-3-4. Bug 3-3 `AlignedAllocation.h` の例外 RT 伝播 — 現状確定

**現状再確認（2026-08-11）**: `aligned_malloc`（throw bad_alloc）/ `aligned_malloc_nothrow`（nullptr 返却、noexcept）の 2 系統。`makeAlignedArray` / `ScopedAlignedPtr` の使用箇所は全て Non-RT（StateAndUI.cpp:612/656/657 = UI、ResampleAndFallback.cpp = LoaderThread、Lifecycle.cpp:360 = prepareToPlay、EQProcessor.h メンバ = prepareToPlay で事前確保）。**RT パス（AudioEngine.Processing.*.cpp の process）には `aligned_malloc` / `new` / `setSize` は存在しない**（事前割当制）。

**確定**: **RT パスで例外伝播は発生しない**（事前割当制が徹底済み）。`aligned_malloc_nothrow` は RT 用の安全 API として提供済み。将来の RT パス変更時の契約として「RT パスでは aligned_malloc_nothrow のみ使用」をコメントで明示することを推奨（対応不要のまま維持可）。

#### 10-3-5. Bug 3-8 `cachedLatency` 例外安全性 — 現状確定

**現状再確認（2026-08-11）**: `cachedLatency` は `src/ConvolverProcessor.h` に `std::atomic<LatencySnapshot*>` メンバ（現行名は `n`）。`updateLatencyCache()` が `new LatencySnapshot()` → `exchangeAtomic(n, newSnap, acq_rel)` で公開（StateAndUI.cpp）。消費側は `consumeAtomic(n, acquire)`。

**確定**: **現状維持で安全**（3-8 の「現状維持も可」に確定）。`new LatencySnapshot()` の bad_alloc は Non-RT（UI/Message Thread）で発生し、公開前なので RT に影響しない。`std::atomic` のコピー delete により二重解放リスクなし。

#### 10-3-6. R-9 / 3-9 / 3-10 CMake フラグ — 前回変更を反映し残課題を確定

**前回変更（2026-08-11 Release icx ビルド修正）**: icx 版 `CMAKE_CXX_FLAGS_RELEASE` を `/O3` → `/O2` に変更（JUCE 巨大ファイルで LLVM out of memory 回避）。MSVC 版は既に `/O2`（L1283）。

**残課題の確定**:
- **3-9 `/fp:fast`**: 依然 `CMAKE_CXX_FLAGS_RELEASE` に残存（L1283 MSVC / L1360 icx）。DSP 数値精度の観点で **`target_compile_options` への移行を推奨**（グローバル `/fp:fast` を削除し、DSP 専用ターゲットに `/fp:fast`、それ以外は `/fp:precise`）。**ただし icx では `/fp:fast` がデフォルトかつ性能上の要件**（コメント明記）のため、**現状維持（グローバル /fp:fast）も許容**
- **3-10 `/QxCORE-AVX2`**: icx 版のみ。Intel CPU 専用フラグ。AMD 非互換の観点で **`CMAKE_CXX_FLAGS_RELEASE` から除去し、`target_compile_options` に移行することを推奨**（RuntimeBuilder 等の AVX2 依存コードは `__AVX2__` マクロでガード済みか確認が必要）
- **R-9 CMAKE_CXX_FLAGS_RELEASE グローバル上書き**: `/fp:fast` と `/QxCORE-AVX2` が全ターゲットに適用される構造は依然残存。**3-9/3-10 のターゲット固有化が完了すれば R-9 も解消**（段階的対応）

### 10-4. 別視点調査の総括

| 項目 | 確定内容 | 対応 |
|---|---|---|
| ThreadAffinityManager.h | 問題なし | 対応不要 |
| EqProcessor::reset() / DSPCore::reset() | デッドコード（work89 8.2 確定） | BUG-065 は案 A（rt シャドウ書込削除）で将来安全化 |
| ConvolverProcessor.Rebuild.cpp | 実行経路は堅牢（競合・通知消失・二重解放なし）。**incremental rebuild サブシステムは休眠状態**（R-新規A〜E 参照） | R-新規A（リーク修正 4行）+ R-新規D（コメント修正）は即実施可。R-新規B/C は設計判断 |
| R-新規A IncrementalRebuildJob::reset() リーク | `~StereoConvolver()+aligned_free` で NUC エンジンリーク（潜伏） | `retireStereoConvolver(pendingConv, 0)` に変更 |
| R-新規B Incremental rebuild 未接続 | `rebuildJob` 未確保・`beginIncrementalRebuild` 等は宣言のみ未定義 | 削除 or 将来予約コメント + スタブ化 |
| R-新規C 例外時リーク（OOM） | `applyNewState` / `make_unique` の bad_alloc で新エンジンリーク | try/catch + `retireStereoConvolver` |
| R-新規D executePendingCommit コメント不整合 | 「Message Thread のみ」が Rebuild Thread 実行と不一致 | コメント修正 |
| R-新規E setUseIncrementalRebuild 過去バグ | 修正済み（enable を正しく使用） | 監査記録のみ |
| Bug 1-2 coordinatorDeferredRing_ / lastResortQueue_ | producer 実装せず、値初期化のみ | `lastResortQueue_` に `{}` 初期化 |
| Bug 1-5 musicalSoftClip | デッドコード削除を確定 | クラスメソッド削除 |
| Bug 2-8 cleanup 強制削除 | 役割分担は正常（現状維持） | 対応不要 |
| Bug 3-3 AlignedAllocation 例外 | RT パスで確保なし（事前割当制） | コメント明示のみ |
| Bug 3-8 cachedLatency | 現状維持で安全 | 対応不要 |
| R-9 / 3-9 / 3-10 CMake フラグ | /O3→/O2 変更反映。/fp:fast・/QxCORE-AVX2 のターゲット固有化は残課題 | 段階的対応（推奨） |
