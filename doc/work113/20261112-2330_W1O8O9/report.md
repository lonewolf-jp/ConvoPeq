# W1-O8（LoaderThread の参照保持）と W1-O9（quiescence）の READ-ONLY 監査

作業日時: 2026-11-12 23:30〜
基準 commit: `173e6995`（HEAD = origin/main）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 結果

```
W1-O8  = CLOSED（参照メンバーではなく値メンバー。外部バッファ条件も clear）
W1-O9  = 順序は CONFIRMED / quiescence は NOT GUARANTEED（stopThread のタイムアウトが残る）
(b)    = 単一 atomic<uint64_t> へのパックを推奨（2 値の世代混在を排除）
#7     = guard 対象に含める（Owner 判断を支持。writer が 3 系統のため）
実装 GO = 未承認
```

W1-O8 は Owner の見立てどおりで、`sourceIR` は値メンバーであり参照を保持しない。JUCE のコピー ctor が外部データを共有するのは「参照バッファ」の場合だけで、IR 経路には参照バッファが存在しない（`setDataToReferTo` は audio バッファ 3 箇所のみ）。これで W1-O8 は閉じる。

W1-O9 は、停止順は正しいが、停止が `stopThread(タイムアウト)` の**ベストエフォート**であり、戻り値を見ていない。タイムアウト時は読み手が生き残るため、quiescence は「保証」ではなく「期待」である。ここが残る。

---

## 1. W1-O8: LoaderThread は IRState を参照し続けない

### 1.1 コードで確認した事実

| 事実 | 根拠 |
|---|---|
| `sourceIR` は**値メンバー** | `ConvolverProcessor.LoaderThreadInline.h:83` `juce::AudioBuffer<double> sourceIR;` |
| ctor は `sourceIR(src)` でコピー構築 | `ConvolverProcessor.LoaderThread.cpp:37`（`:33` の再構築用 ctor） |
| loader は自分のコピーだけを使う | `:390` `stepResult.loadedIR = std::move(sourceIR);` |
| `IRState::ir` は常に所有バッファを指す | `Lifecycle.cpp:37` `newState->ir = newState->irOwner.get();`（`irOwner` は `unique_ptr<AudioBuffer>`、`updateIRState` が `make_unique<AudioBuffer>(newIR)` で作る） |
| IR 経路に参照バッファが無い | `setDataToReferTo` の使用は `Lifecycle.cpp:371,375,379` の dry / smoothing / oldDry のみ。IR の生成は `make_unique` / move / resample のみ（`CacheManager.cpp:411`、`IRConverter.cpp:565`、`LoaderThread.cpp:337,661`、`LoadPipeline.cpp:498,803`、`Rebuild.cpp:270`） |

### 1.2 JUCE のコピー ctor の条件

`JUCE/modules/juce_audio_basics/buffers/juce_AudioSampleBuffer.h:132-159`

* 「このバッファは他方のデータを**自分でコピーする**。ただし外部データバッファで作られた場合は、両方が同じブロックを指す」（`132-134` のコメント）。
* 実装は `allocatedBytes == 0` のとき `allocateChannels(other.channels, 0)`（= 共有）、それ以外は `allocateData()` + `FloatVectorOperations::copy`（= 深いコピー）。

IR 経路のバッファはすべて所有（`allocatedBytes != 0`）なので、`sourceIR(src)` は**深いコピー**になる。したがって LoaderThread の ctor 完了後、loader が IRState のメモリを参照することはない。

### 1.3 ただし構築後にも IRState を読む箇所がある

`ConvolverProcessor.Rebuild.cpp:88-103`

```cpp
LoaderThread loader(*this, *(state->ir), state->sampleRate, ...);   // :88
loader.runSynchronously();                                          // :95
...
juce::Logger::writeToLog("... len=" + juce::String(state->ir->getNumSamples())   // :101
                          + " ch=" + juce::String(state->ir->getNumChannels())   // :102
                          + " srcSR=" + juce::String(state->sampleRate, 1));      // :103
```

Owner の指摘どおり、LoaderThread 構築後にも `state->ir` を読む。guard を付ける場合、範囲は `acquireIRState()`（`:64`）から `releaseIRState(state)`（`:109`）までである。

### 1.4 W1-O8 の判定

**CLOSED。** 参照保持ではなく、コピー元も所有バッファである。残すのは「`IRState::ir` は常に所有バッファ」という invariant の明文化（将来 IR に `setDataToReferTo` を持ち込まない）だけで、これは W1-O8a として記録する。

---

## 2. W1-O9: 停止順は正しいが、quiescence は保証されない

### 2.1 停止順（実測）

| 段階 | 内容 | 根拠 |
|---|---|---|
| 1 | `shutdownCoordinatorLoop()`（Coordinator Worker を join） | `AudioEngine.Processing.ReleaseResources.cpp:237` |
| 2 | `stopRebuildThread()`（RebuildThread を join） | `:238`。コメントに「join で Builder 完全終了」 |
| 3 | `joinProducers()` の retry loop | `:250` |
| 4 | `uiConvolverProcessor.releaseResources()` | `:558` |
| 5 | `stopUpgradeThread()`（`cancel` → `stopThread(2000)` → `reset`） | `ConvolverProcessor.LoadPipeline.cpp:101-109`、呼出しは `Lifecycle.cpp:471` |
| 6 | `forceCleanup()`（全 loader を `stopThread(500)`） | `ConvolverProcessor.StateAndUI.cpp:1035-1053`、呼出しは `Lifecycle.cpp:472` |
| 7 | `activeLoader.reset()`（`forceCleanup` が既に move しているため通常は no-op。残っていれば `~LoaderThread` が `stopThread(500)`） | `Lifecycle.cpp:473`、`StateAndUI.cpp:1045-1046`、`LoaderThread.cpp:42-48` |
| 8 | `currentIRState` を `nullptr` に交換 → `~IRState()` → `mkl_free` | `Lifecycle.cpp:506-511` |

デストラクタも同型である。

| 段階 | 内容 | 根拠 |
|---|---|---|
| 1 | `~AudioEngine`: `shutdownCoordinatorLoop()` → `stopRebuildThread()` | `AudioEngine.CtorDtor.cpp:176-177` |
| 2 | メンバ破棄により `~ConvolverProcessor` | 同上（dtor body の後） |
| 3 | `stopUpgradeThread` → `forceCleanup` → `activeLoader.reset` → IRState 即時 free | `Lifecycle.cpp:112-131` |

`updateIRState` の provider 不在分岐（`Lifecycle.cpp:58-61` の即時 `deleter(oldState)`）は、`setRcuProvider` が engine 構築時（`AudioEngine.CtorDtor.cpp:70`）と DSPCore 構築時（`AudioEngine.h:713`）に呼ばれるため、**読み手が存在しない段階**にしか到達しない。

### 2.2 何が保証されていないか

`stopThread(timeout)` は「タイムアウトまでに止まらなければ false を返す」API であり、**戻り値が無視されている**（`LoadPipeline.cpp:106`、`StateAndUI.cpp:1051`、`LoaderThread.cpp:44`）。`reset()` でスレッドオブジェクトを破棄するため、Release では assert も出ない。

タイムアウト時に生き残る読み手は実在する。

| 生き残り得る読み手 | 読む場所 |
|---|---|
| upgrade thread（`ProgressiveUpgradeThread.cpp:208` → `applyComputedIR`） | `LoadPipeline.cpp:539-567` の IR 走査 |
| loader thread | `LoaderThread.cpp:918-921` の `computeScaleFactor(stepTrimmed, currentIr, ...)` |
| rebuild thread | 停止は engine 側（§2.1 の 2）なので、通常は先に止まる |

つまり「順序は正しいが、停止の完了を待っていない」ので、W1-O9 の quiescence は**証明できない**。証明するには次のどちらかが必要である。

1. `stopThread` の戻り値を検査し、`false`（タイムアウト）なら IRState を即時解放せず、engine の deferred delete へ移送する（安全側）。
2. 停止完了を別の同期（join 可能なスレッドハンドル、カウンタ）で確認してから解放する。

加えて、`releaseResources` が **message thread で呼ばれる**という JUCE の契約が前提にある（message の読み手は呼出し元自身なので直列）。この契約を invariant として明記する必要がある（W1-O9b）。

### 2.3 W1-O9 の判定

**順序は CONFIRMED、quiescence は NOT GUARANTEED。** W1-O9a（タイムアウト時の解放延期）と W1-O9b（message thread 契約）を義務として残す。

---

## 3. (b) 2 つの float の整合条件

* 2 値を別々の atomic にすると、Reader は**異なる IRState 世代**の値を組み合わせて読める（各 atomic の load は独立で、2 つを結ぶ同期が無い）。
* 用途は `AudioEngine.RebuildDispatch.cpp:750-751` で、`analysis.irFreqPeakGainDb` と `analysis.additionalAttenuationDb` に入り、ゲイン推定に使われる。**混在はメモリ安全の問題ではなく、推定値が同一 IR のメタデータでなくなる意味論の問題**である。
* したがって、許容するか否かを設計で決める必要がある（Owner の指摘どおり）。推奨は**単一の `std::atomic<std::uint64_t>` に float32 2 個をパックする**方式である。両フィールドは `float`（`ConvolverProcessor.h:1245-1246`）なので、ビット表現をそのまま詰められ、1 回の load で同一世代の 2 値が取れる。追加コストは store/load 各 1 回で、2 つの atomic と変わらない。
* 代替は seqlock（generation 付き）。2 値ならパックの方が小さい。

**判定: 単一語パックを設計条件として推奨。** 2 値独立を許容する場合は「混在しても推定誤差の範囲」と明文化する必要がある。

---

## 4. guard 範囲の確定案

Owner の表に、`#7` の guard 化と、`Rebuild.cpp` の構築後参照を反映する。

| # | 対象 | 範囲（guard を保持する区間） | 根拠 |
|---|---|---|---|
| 1 | `getIrAdditionalAttenuationDb` | 不要（atomic 複製へ移行） | 1 フィールド読み |
| 2 | `getIrFreqPeakGainDb` | 不要（atomic 複製へ移行） | 同上 |
| 3 | `getIRGeometry` | `acquireIRState()` → 戻り値構築完了 | IR 全体の走査 |
| 4 | `transferIRStateFrom` | `source.acquireIRState()` → `updateIRState` のコピー完了 | IR のコピー |
| 5 | `loadImpulseResponse`（`:30-33`） | `acquireIRState()` → 判定完了 | `state->ir->getNumSamples()` |
| 6 | `loadImpulseResponse`（`:73-86`） | `acquireIRState()` → `LoaderThread` ctor 完了 | `*(state->ir)` のコピー（§1） |
| 7 | `applyComputedIR`（`:539-567`） | `acquireIRState()` → 走査完了 | writer が 3 系統のため直列化は未証明 |
| 8 | `LoaderThread.cpp:918` | `acquireIRState()` → `computeScaleFactor` 完了 | loader の読み |
| 9 | `Rebuild.cpp:64` | `acquireIRState()` → `releaseIRState`（`:109`） | 構築後（`:101-103`）にも読む |

* provider は engine ドメイン（`getRcuProvider()->getRetireRouter()`）。対象はすべて非 RT なので、フェンスや RT コストの議論は不要。
* `#7` を guard 対象に含めるのは妥当である。`applyComputedIR` の writer（`:640`）と同一スレッドであることは、upgrade thread の writer（`ProgressiveUpgradeThread.cpp:208`）との排他を意味しない。

---

## 5. 残る義務

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| W1-O8a | IRState の所有バッファ invariant | `ir` は常に `irOwner` を指す。IR に参照バッファを持ち込まない | 明文化待ち |
| W1-O9a | 停止タイムアウト時の解放延期 | `stopThread` の戻り値を検査し、失敗時は deferred delete へ移送 | 未（設計） |
| W1-O9b | message thread 契約 | `releaseResources` が message thread で呼ばれることを invariant 化 | 未 |
| (b)-1 | 2 値の単一語パック | 設計決定 | 未（推奨提示） |
| #7 | guard 化 | 直列化が証明できれば再評価 | 未（設計） |
| W1-O6 | snapshot 2 経路の順序修正 | 独立した実装単位 | 承認済み・未実装 |
| A-O1〜A-O5 | フェンス配置ほか | 前回のまま | 未 |
| N6 / N7 / N9 | 前回のまま | ガード無し deref / force drain の quiescence / 別機構 | 未 |

---

## 6. 証跡

`doc/work113/20261112-2330_W1O8O9/evidence/`

| ファイル | 内容 |
|---|---|
| `w1o89_excerpts.md` | 本稿が依存するコードの逐語抜粋（10 ファイル / 840 行 / 37,619 文字、SHA-256 付き、HEAD blob 基準。JUCE のコピー ctor を含む） |

---

## 7. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** W1-O8 は閉じたが、W1-O9 はタイムアウトの扱いが決まるまで閉じない。snapshot の実装も開始していない。実装 GO・Gate 1 承認は付与しない。
