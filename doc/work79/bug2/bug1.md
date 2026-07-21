# ConvoPeq コードベース バグ報告

コーディング規約・ISR Bridge Runtime設計指針・ソースコードを精査した結果、以下のバグを特定しました。

---

## Bug 1: `processBlock` / `processBlockDouble` — DSP null時に出力バッファ未クリア 【重大】

**対象ファイル:**
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`

**問題箇所:**

```cpp
// AudioBlock.cpp
const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
if (runtimeWorld == nullptr)
{
    bufferToFill.clearActiveBufferRegion();  // ← ここはクリアしている
    return;
}
// ...
DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
if (!dsp)
    return;  // ← BUG: バッファをクリアせずにreturn
```

```cpp
// BlockDouble.cpp — 同一パターン
if (!dsp)
    return;  // ← BUG: buffer.clear() がない
```

**発生条件:**
`runtimeWorld` は存在するが `engine.current`（DSPCore*）が null の場合。
初期化直後（World公開後、初回DSP publish前）またはシャットダウン中（DSP退役後、Worldクリア前）に発生し得る。

**影響:**
出力バッファに前回の処理結果（stale data）が残留し、**音声グリッチ・ノイズ**として出力される。

**修正:**
```cpp
if (!dsp)
{
    bufferToFill.clearActiveBufferRegion();  // AudioBlock.cpp
    // buffer.clear();                       // BlockDouble.cpp
    return;
}
```

**根拠:** コーディング規約「Audio Thread内では待機・確保禁止」の趣旨に加え、JUCEの `getNextAudioBlock` はバッファの事前クリアを保証しないため、null DSP時は明示的にゼロクリアが必要。

---

## Bug 2: `processBypassWithLatencyCompensation` — delayBuffer null時に出力未クリア 【中】

**対象ファイル:** `src/ConvolverProcessor.h`（`processBypassWithLatencyCompensation`）

**問題箇所:**
```cpp
double* delayBuf[2] = { delayBuffer[0].get(), delayBuffer[1].get() };
int activeDelayCapacity = delayBufferCapacity;
if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
    return;  // ← BUG: block をクリアせずに return
```

**発生条件:**
`prepareToPlay()` 未呼び出し、または `releaseResources()` 後にバイパス遷移中で `process()` が呼ばれた場合。

**影響:**
バイパス遷移中にdelayBufferが未確保の場合、出力バッファにstale dataが残留。

**修正:**
```cpp
if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
{
    block.clear();  // 追加
    return;
}
```

---

## Bug 3: `AudioSegmentBuffer::copyLatest` — 読み取り順序のTOCTOUリスク 【低〜中】

**対象ファイル:** `src/AudioSegmentBuffer.h`

**問題箇所:**
```cpp
int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
{
    // Writer側: writePosition(release) → totalSamples(release) の順で更新
    // Reader側:
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);   // ①先に読む
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire); // ②後に読む
    // ...
}
```

**問題:**
Writerは `writePosition` → `totalSamples` の順でreleaseするが、Readerは `totalSamples` → `writePosition` の順でacquireする。release-acquireの連鎖により理論上は安全だが、**読み取り順序がWriterの書き込み順序と逆**であり、将来の修正で順序が崩れた場合に即座にデータ不整合が発生する脆弱な設計。

**推奨修正:**
Writerの書き込み順序とReaderの読み取り順序を一致させる（`writePosition` → `totalSamples` の順で読む）、または両方を単一のatomic構造体にまとめる。

---

## Bug 4: `LoaderThread::performLoad` — 例外発生時の `newConv` リーク可能性 【低】

**対象ファイル:** `src/ConvolverProcessor.h`（`LoaderThread::performLoad`）

**問題箇所:**
```cpp
LoadResult performLoad(juce::Thread* thread)
{
    // ... 各ステップで stepResult.newConv が設定される ...
    try {
        // stepResult.newConv が設定された後に例外が発生しうる
    } catch (const std::bad_alloc&) {
        stepResult.errorMessage = "...";
        // stepResult.newConv のクリーンアップがない
    } catch (...) {
        stepResult.errorMessage = "...";
        // stepResult.newConv のクリーンアップがない
    }
    return std::move(stepResult);
}
```

**緩和要因:**
呼び出し元 `run()` が `!result.success` 時に `retireStereoConvolver(std::exchange(result.newConv, nullptr), 0)` でクリーンアップするため、実害は限定的。ただし `performLoad` 内部で例外がキャッチされた場合、`stepResult.newConv` がリークする経路が存在する。

**推奨修正:**
catch ブロック内で `stepResult.newConv` を明示的にクリーンアップする。

---

## Bug 5: `makeEngineRuntimeState` — `runtimeWorld` nullチェック欠如 【低】

**対象ファイル:** `src/audioengine/AudioEngine.h`

**問題箇所:**
```cpp
inline convo::EngineRuntime makeEngineRuntimeState(
    DSPCore* current, DSPCore* next,
    convo::TransitionPolicy policy, double fadeTimeSec, bool active,
    const RuntimePublishWorld* runtimeWorld) noexcept
{
    // runtimeWorld->retire.retireBacklog 等に直接アクセス
    // null チェックなし（jassert のみ）
}
```

**問題:**
Releaseビルドでは `jassert` はno-opのため、`runtimeWorld` がnullの場合に**クラッシュ**する。呼び出し元は常にnon-nullを渡すが、防御的プログラミングの観点からnullチェックが望ましい。

---

## 重要度のまとめ

| # | バグ | 重要度 | 影響 |
|---|------|--------|------|
| 1 | processBlock/BlockDouble: DSP null時バッファ未クリア | **重大** | 音声グリッチ・stale data出力 |
| 2 | processBypassWithLatencyCompensation: delayBuffer null時未クリア | **中** | バイパス遷移中のグリッチ |
| 3 | AudioSegmentBuffer::copyLatest: 読み取り順序の脆弱性 | **低〜中** | 将来修正時のデータ不整合リスク |
| 4 | LoaderThread::performLoad: 例外時newConvリーク | **低** | 呼び出し元で緩和済み |
| 5 | makeEngineRuntimeState: nullチェック欠如 | **低** | Releaseビルドでクラッシュ可能性 |

Bug 1 が最も影響が大きく、初期化・シャットダウン時に確実に発生し得るため、最優先での修正を推奨します。