# ConvoPeq ソースコード バグ解析レポート

ソースコードを詳細に調査した結果、以下のバグおよび潜在的な問題を特定しました。重大度順に分類しています。

---

## 🔴 重大（Critical）

### Bug-1: `StereoConvolver::init()` における `irData` の二重 `release()` による nullptr 代入

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`

```cpp
// init() 内:
convo::ScopedAlignedArray<double> newIrL(irL);
// ...
if (!newNuc0->SetImpulse(newIrL.release(), length, ...))  // ← 1回目の release()
{
    return false;
}
// ...
irData[0] = newIrL.release();  // ← 2回目の release() → nullptr を返す！
irData[1] = newIrR.release();  // ← 同様に nullptr
```

**問題:** `SetImpulse()` は `const double*` を受け取る（所有権を取らない）。`newIrL.release()` で ScopedAlignedArray の所有権が放棄された後、`SetImpulse` はデータをコピーするだけで元のポインタを保持しない。その後 `irData[0] = newIrL.release()` は **nullptr** を返す。

**影響:**
- `irData[0]` / `irData[1]` が nullptr になる
- `clone()` 内で `irData[0] && irData[1]` チェックが false となり、IR データなしのクローンが生成される
- 元の IR バッファは `release()` 後に誰も所有せず **メモリリーク** する

**修正案:**
```cpp
if (!newNuc0->SetImpulse(newIrL.get(), length, ...))  // get() を使用
    return false;
// ...
irData[0] = newIrL.release();  // ここで正しく所有権を移譲
```

---

### Bug-2: `enqueueDeferredDeleteWithFallback()` 失敗時のメモリリーク

**ファイル:** `src/eqprocessor/EQProcessor.Coefficients.cpp` 他

```cpp
// 呼び出し側:
if (oldState) {
    (void)retireEQStateDeferred(oldState);  // 戻り値を無視！
}
```

`retireEQStateDeferred()` → `enqueueDeferredDeleteWithFallback()` が `false` を返した場合（キュー満杯/シャットダウン中）、`oldState` ポインタは誰も解放せず **永久リーク** する。

**影響:** 高負荷時やシャットダウン時に EQState / BandNode がリークし、メモリ使用量が単調増加する。

---

### Bug-3: `LoaderThread::FlagResetter` — スレッドキャンセル時のフラグ未リセット

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`

```cpp
~FlagResetter() {
    if (!success && !t.threadShouldExit()) {  // ← キャンセル時はスキップ
        // フラグをリセットする処理
    }
}
```

**問題:** `signalThreadShouldExit()` でスレッドがキャンセルされた場合、`threadShouldExit()` が true となり、`isLoading` / `isRebuilding` フラグが **true のまま残留** する。

**影響:**
- UI が `isLoadingIR()` を永久に true と判定し、新しい IR ロードがブロックされる
- `rebuildAllIRs()` が `isRebuilding` チェックで永久にスキップされる

---

## 🟠 高（High）

### Bug-4: `DeferredDeletionQueue::reclaim()` — FIFO 先頭ブロッキングによる無制限メモリ成長

**ファイル:** `src/DeferredDeletionQueue.h`

```cpp
} else {
    // ★ 先頭エントリが削除不可 → FIFO順序のため即座に脱出
    break;
}
```

**問題:** 先頭エントリの epoch が `minReaderEpoch` より新しい場合（Reader がまだアクティブ）、**後続の全エントリが永久にブロック** される。Reader がスタックした場合、キューは無限に成長する。

**影響:** 長時間動作時にメモリ使用量が単調増加し、最終的に OOM に至る可能性がある。

---

### Bug-5: `ConvolverProcessor::process()` における `conv` ポインタの TOCTOU 可能性

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`（process 関数）

```cpp
auto* conv = loadActiveEngine(std::memory_order_acquire);
if (!conv) return;
// ... conv を使用 ...
```

RCU リードロック（`RCUReaderGuard`）で保護されているが、`loadActiveEngine()` の戻り値 `conv` を使用している間に `exchangeActiveEngine(nullptr)` が呼ばれた場合、RCU の grace period 内であれば `conv` は有効。しかし、`retireStereoConvolver()` が grace period 前に呼ばれる設計上の誤りがあると UAF になる。

**現状:** RCU 設計上は正しいが、`retireStereoConvolver()` の呼び出しタイミングが RCU grace period を尊重しているかの検証が不十分。

---

### Bug-6: `AudioEngine::captureAudioThreadParameterSnapshot()` — `world` nullptr 時のフォールバック不完全

**ファイル:** `src/audioengine/AudioEngine.h`

```cpp
inline EngineParameterSnapshot captureAudioThreadParameterSnapshot(
    const RuntimePublishWorld* world, ...) const noexcept
{
    EngineParameterSnapshot snapshot {};
    if (world != nullptr) {
        snapshot.saturationAmount = static_cast<float>(world->automation.saturationAmount);
        // ...
    } else {
        // フォールバック: atomic から読み取り
        // しかし saturationAmount のフォールバックがない！
    }
}
```

**問題:** `world == nullptr` のフォールバックパスで `saturationAmount`、`inputHeadroomGain`、`outputMakeupGain`、`convolverInputTrimGain` のフォールバック読み取りが不足している場合、デフォルト値（0.0 / 1.0）が使用され、ユーザー設定が無視される。

---

## 🟡 中（Medium）

### Bug-7: `AudioEngine::makeEngineRuntimeState()` — `runtimeWorld` nullptr 時の `retire` フィールド未設定

**ファイル:** `src/audioengine/AudioEngine.h`

```cpp
inline convo::EngineRuntime makeEngineRuntimeState(..., const RuntimePublishWorld* runtimeWorld) noexcept
{
    // ...
    if (runtimeWorld == nullptr) {
        // retire.retireBacklog / retire.deferredResidency のフォールバックがない
    }
}
```

`runtimeWorld == nullptr` 時、`runtime.retire.retireBacklog` と `runtime.retire.deferredResidency` が 0 のままになり、実際のバックログ情報が失われる。

---

### Bug-8: `ConvolverProcessor::LoaderThread::run()` — `MessageManager::callAsync` 失敗時のスレッド安全性

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`

```cpp
if (!queued) {
    if (auto* o = wp.get()) {
        convo::publishAtomic(o->isLoading, false, std::memory_order_release);
        convo::publishAtomic(o->isRebuilding, false, std::memory_order_release);
    }
}
```

`callAsync` が失敗した場合（MessageManager シャットダウン中）、ローダー线程から直接 atomic に書き込む。atomic なのでスレッドセーフだが、`wp.get()` が有効なポインタを返しても、オブジェクトがデストラクタ実行中の場合、メンバアクセスは UB になる可能性がある（JUCE の WeakReference はデストラクタ開始時に nullptr を返す設計だが、タイミングウィンドウが存在する）。

---

### Bug-9: `Fixed15TapNoiseShaper::processSample()` — `saturateAVX2` の per-sample 呼び出し

**ファイル:** `src/Fixed15TapNoiseShaper.h`

```cpp
const double clampedError = saturateAVX2(error, -2.0 * scale, 2.0 * scale);
```

`saturateAVX2` は SSE2 命令を使用するが、**サンプルごとに1回** 呼ばれている。SIMD の利点が完全に失われており、スカラー分岐より遅い可能性がある。

---

### Bug-10: `CustomInputOversampler::decimateStage()` — `loadStride2` の境界アクセス

**ファイル:** `src/CustomInputOversampler.cpp`

```cpp
inline __m256d loadStride2(const double* ptr) noexcept
{
    __m128d v0 = _mm_loadu_pd(ptr - 6);  // ptr の 6 要素前をアクセス
    // ...
}
```

`prepareStage()` で `historyDownKeep` に +6 マージンを追加しているが、`decimateStage()` 内の `base = keep + (n << 1)` 計算で `base - convParity` が `historyDownKeep` の境界を超える場合、`loadStride2` がバッファ外を読み取る。

**条件:** `keep` が `convParity + ((convCount - 1) << 1) + 6` より小さい場合に発生。`prepareStage()` の計算で防止されているが、`convCount` が大きい場合に境界条件が脆弱。

---

### Bug-11: `AudioEngine::processBlockDouble()` — `maxInternalBlockSize` の非 atomic 読み取り

**ファイル:** `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`

```cpp
if (static_cast<size_t>(numSamples) > static_cast<size_t>(maxInternalBlockSize))
```

`maxInternalBlockSize` は `EQProcessor::prepareToPlay()` で設定される plain int。JUCE は `prepareToPlay()` がオーディオスレッド処理中に呼ばれないことを保証するが、`prepareToPlay()` の呼び出しとオーディオスレッドの開始/停止の間に競合ウィンドウが存在する可能性がある。

---

### Bug-12: `ConvolverProcessor::applyNewState()` — `MessageManager::callAsync` 失敗時のリソースリーク

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`

```cpp
const bool queued = juce::MessageManager::callAsync([weakThis, commitPtr]() { ... });
if (!queued) {
    auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
    ownedCommit->releaseEngine();
}
```

`callAsync` が失敗した場合、`PendingCommit` の `releaseEngine()` は呼ばれるが、`loadedIR` と `displayIR` の `unique_ptr` は `PendingCommit` のデストラクタで解放される。これは正しいが、`commitPtr` が `release()` された後の `unique_ptr` 再構築は、`commitPtr` が既に nullptr の場合 UB になる。

---

## 🟢 低（Low）

### Bug-13: `DeferredDeletionQueue::reclaim()` — `kMaxScan` が実質的に無意味

**ファイル:** `src/DeferredDeletionQueue.h`

```cpp
constexpr int kMaxScan = 1024;
// ...
while (scanned < kMaxScan) {
    // ...
    } else {
        break;  // 先頭が reclaim 不可なら即 break
    }
}
```

コメントでは「先頭が reclaim 不可の場合は即 break」とあるが、`kMaxScan` のループ上限は実際には到達しない。コードの意図と実装が不一致。

---

### Bug-14: `AudioEngine::captureAudioThreadParameterSnapshot()` — double→float 精度損失

**ファイル:** `src/audioengine/AudioEngine.h`

```cpp
snapshot.saturationAmount = static_cast<float>(world->automation.saturationAmount);
```

`RuntimePublishWorld` では `double` で保持している `saturationAmount` を `float` に変換している。0.0〜1.0 の範囲では精度損失は微小だが、設計上の不一致。

---

### Bug-15: `ConvolverProcessor::StereoConvolver::clone()` — Bug-1 の連鎖的影響

**ファイル:** `src/convolver/ConvolverProcessor.Lifecycle.cpp`

```cpp
if (irDataLength > 0 && irData[0] && irData[1])  // Bug-1 により irData[0] == nullptr
{
    // このブロックがスキップされる
}
return newConv.release();  // IR データなしのクローンが返される
```

Bug-1 の影響で `irData[0]` が nullptr のため、`clone()` は IR データをコピーせず、空の NUC エンジンを持つクローンを返す。

---

### Bug-16: `AudioEngine::makeEngineRuntimeState()` — `jassert` と `if` の二重チェック

**ファイル:** `src/audioengine/AudioEngine.h`

```cpp
jassert(runtimeWorld != nullptr);  // Debug のみ
if (runtimeWorld == nullptr)       // Release でもチェック
{
    // フォールバック
}
```

`jassert` は Release ビルドで除去されるが、`if` チェックは残る。これは正しい防御的プログラミングだが、`jassert` が Debug ビルドでクラッシュするため、開発中に意図しないクラッシュが発生する。

---

### Bug-17: `RuntimePublicationCoordinator::publishWorld()` — `const_cast` による sealRecursively 呼び出し

**ファイル:** `src/audioengine/RuntimePublicationCoordinator.h`

```cpp
const_cast<World*>(worldOwner.get())->sealRecursively();
```

`aligned_unique_ptr<const World>` から `const_cast` で非 const ポインタを取得し、`sealRecursively()` を呼んでいる。オブジェクトは元々非 const として生成されているため実用上は安全だが、C++ 標準上は UB（const オブジェクトの const_cast 後の変更は UB）。

---

### Bug-18: `AudioEngine::enqueueDeferredDeleteWithFallback()` — `m_retireRouter` nullptr チェックなし

**ファイル:** `src/audioengine/AudioEngine.h`（推定）

```cpp
bool AudioEngine::enqueueDeferredDeleteWithFallback(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    if (ptr == nullptr || deleter == nullptr) return true;
    auto result = m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, ...);
    // m_retireRouter が nullptr の場合、nullptr デリファレンス
}
```

`m_retireRouter` が初期化前に呼ばれた場合、nullptr デリファレンスでクラッシュする。

---

## 📋 設計上の懸念（Design Concerns）

### Concern-1: `RuntimePublishWorld` の `const` 所有権と `sealRecursively()` の矛盾

`aligned_unique_ptr<const RuntimePublishWorld>` として所有しながら、`sealRecursively()` で内部状態を変更する設計は、`const` の意味を損なっている。`seal()` 前は非 const、`seal()` 後は const という二段階所有権モデルが望ましい。

### Concern-2: `DeferredDeletionQueue` の FIFO 制約とメモリ成長

先頭ブロッキングにより、スタックした Reader が1つあるだけで全キューがブロックされる。`kMaxScan` の先読みスキャンが実装されていないため、メモリ成長の上限がない。

### Concern-3: `m_rtBypassShadow` の非 atomic 設計

`setBypassFromRT()` と `process()` が同じオーディオスレッドで呼ばれる前提で非 atomic にしているが、将来のマルチオーディオスレッド対応時にデータレースになるリスクがある。

---

## 修正優先度のまとめ

| 優先度 | Bug ID | 概要 | 影響 |
|--------|--------|------|------|
| P0 | Bug-1 | `irData` 二重 release → nullptr | IR クローン失敗、メモリリーク |
| P0 | Bug-2 | enqueue 失敗時のメモリリーク | 長時間動作で OOM |
| P0 | Bug-3 | スレッドキャンセル時フラグ残留 | IR ロード永久ブロック |
| P1 | Bug-4 | FIFO 先頭ブロッキング | メモリ無制限成長 |
| P1 | Bug-5 | RCU grace period 検証不十分 | 潜在的 UAF |
| P1 | Bug-6 | world nullptr フォールバック不完全 | 設定値の消失 |
| P2 | Bug-7〜12 | 中程度の問題 | 機能低下、精度損失 |
| P3 | Bug-13〜18 | 低優先度 | コード品質、設計改善 |