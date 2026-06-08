# ConvoPeq バグ検証報告書 — 修正対象整理

> 本ドキュメントは、バグリストのソースコード検証結果に基づき、**誤検出以外**の項目を修正優先順位順に整理したものです。
> 検証日: 2026-06-08

---

## 優先順位一覧

| 優先度 | バグ# | 概要 | 影響 |
|--------|-------|------|------|
| **P0** | 4 | `updateConvolverState()` の排他制御不備 | 競合状態による状態破損 |
| **P1** | 11 | `drainAllRetired()` の強制回収リスク | 特定条件下でのUse-After-Free |
| **P2** | 12 | `getSafeEpoch()` 戻り値 `0` と `kIdleEpoch` 衝突 | 将来的な誤解放の種 |
| **P3** | 7 | `AudioSegmentBuffer::totalSamples` の非アトミックRMW | 理論上のカウント消失 |
| **P3** | 9 | `EQProcessor` の例外安全性 | 理論上のメモリリーク |
| **P3** | 3 | `globalEpoch` 初期値とRCU回収タイミング | 理論上の早期解放 |
| **Info** | 13 | `NonOwningPtr` ポインタ provenance | CHERI等移植性のみ |

---

## P0 — 至急修正推奨

### バグ4. `updateConvolverState()` の排他制御不備

**① バグの内容**
`exchangeAtomic` によるロック取得がCASではなく、Releaseビルドで競合状態が検出不可になる。

**② 発生個所**

- ファイル: `src/convolver/ConvolverProcessor.StateAndUI.cpp`
- 関数: `ConvolverProcessor::updateConvolverState()`
- 行: 1003（ロック取得）、1010/1022（ロック解放）

**③ バグの詳細**

現在のコード:

```cpp
jassert(!convo::exchangeAtomic(writerActive, true, std::memory_order_acquire));
```

問題点:

1. **`exchangeAtomic` は常に `true` を書き込む** — すでにロック取得済み（`writerActive == true`）でも強制的に `true` を上書きする。
2. **戻り値の意味** — `exchangeAtomic` の戻り値は「古い値」。古い値が `true` なら `jassert(!true)` でアサート失敗するが、**すでに `true` が書き込まれた後**。つまり排他は機能していない。
3. **Releaseビルドでの消滅** — `jassert` は Release では `#define jassert(x) ((void)0)` となる。競合状態でも黙って処理が継続される。
4. **`memory_order_acquire` のみでは不十分** — ロック取得には `acq_rel` または `seq_cst` が適切。`acquire` だけではこのフラグで保護されるデータの可視性を保証できない。

想定シナリオ:

- Message Thread 上で `updateConvolverState()` が通常呼ばれるが、複数のコードパスから再入する可能性がある（タイマー発火 + 非同期更新の同時実行など）。
- `JUCE_ASSERT_MESSAGE_THREAD` は同一スレッドチェックのみで、再入防止にはならない。
- 2回目の呼び出しで `writerActive` が `true` のまま `exchangeAtomic` され、`jassert` が消滅したReleaseビルドで2スレッド（再入）が同時にクリティカルセクションを実行する。

**④ 解決方法**

`exchangeAtomic` を `compareExchangeAtomic`（CAS）に置き換え、「`false` なら `true` に書き換える」というアトミックなロック取得を行う。

```cpp
// 修正後:
bool expected = false;
if (!convo::compareExchangeAtomic(writerActive, expected, true,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire))
{
    // ロック取得失敗 — 別のライターが進行中
    jassertfalse;
    // Releaseビルドでは廃棄/スキップ
    std::unique_ptr<convo::ConvolverState> owned{newState}; // 未使用なら解放
    return;
}
```

または簡潔に:

```cpp
// よりタイトな修正（writerActiveを使い続ける場合）
if (!convo::compareExchangeAtomic(writerActive, expected, true,
                                   std::memory_order_acq_rel,
                                   std::memory_order_acquire))
{
    jassertfalse;
    return; // または適切なフォールバック
}
```

---

## P1 — 早急な対応推奨

### バグ11. `DeferredFreeThread::drainAllRetired()` の強制回収リスク

**① バグの内容**
デストラクタでRCU保護を無視した強制解放が、Audio Thread完全停止前に実行される可能性がある。

**② 発生個所**

- ファイル: `src/DeferredFreeThread.h`
- 関数: `drainAllRetired()`、`shutdownAndDrain()`、`~DeferredFreeThread()`
- 行: 103-107（`drainAllRetired`）、76-80（`shutdownAndDrain`）、54-56（デストラクタ）

**③ バグの詳細**

```cpp
void drainAllRetired() noexcept
{
    while (auto* ptr = swapperRef.tryReclaim(std::numeric_limits<uint64_t>::max()))
    {
        std::unique_ptr<convo::ConvolverState> owned{ptr}; // RAII delete
    }
}
```

- `std::numeric_limits<uint64_t>::max()` を `tryReclaim()` に渡すことで、**すべての** retired エントリを強制的に解放可能と判定させる。
- `shutdownAndDrain()` は `stop()` → `thread.join()` → `drainAllRetired()` の順に実行する。この時点でAudio Threadが停止していることが前提。
- しかし、C++のデストラクタ呼び出し順序はメンバ宣言順の逆であり、`DeferredFreeThread` が `SafeStateSwapper` より先に破棄される場合、`swapperRef` は有効だが `swapperRef` を参照する他のコンポーネントがまだ稼働中である可能性がある。
- `ConvolverProcessor` の `releaseResources()` がデストラクタよりも先に呼ばれることが保証されていれば問題ないが、この依存関係がコード上で明示的に強制されていない。

**④ 解決方法**

**オプションA（推奨）**: `drainAllRetired()` の呼び出し元で、強制回収前にAudio Threadが確実に停止していることを表明するコメントとアサーションを追加する。

```cpp
void shutdownAndDrain() noexcept
{
    stop();
    if (thread.joinable())
        thread.join();
    // この時点で DeferredFreeThread のスレッドは停止。
    // Audio Thread が停止していることは呼び出し側（ConvolverProcessor など）で
    // 保証されている前提。（releaseResources → デストラクタ の呼び出し規約）
    drainAllRetired();
}
```

**オプションB（強化）**: デストラクタで `juce::Logger::writeToLog` などでログを残し、万が一Audio Thread稼働中の強制解放をトレース可能にする。

**オプションC（設計変更）**: `drainAllRetired()` を公開APIから削除し、`shutdownAndDrain()` を `finalize()` などにリネームして「終了処理の最終段階で必ず呼ぶこと」を契約として明確化する。

---

## P2 — 対応推奨（将来的な問題の種）

### バグ12. `getSafeEpoch()` 戻り値 `0` と `kIdleEpoch` の衝突

**① バグの内容**
`getSafeEpoch()` が `0` を返すケースで、`kIdleEpoch`（非参加Readerを示す特別値 `0`）と意味が衝突する。

**② 発生個所**

- ファイル: `src/SafeStateSwapper.h`
- 関数: `getSafeEpoch()`
- 行: 269-273

**③ バグの詳細**

```cpp
uint64_t getSafeEpoch() const noexcept
{
    const uint64_t current = convo::consumeAtomic(globalEpoch, ...);
    if (current < 2) return 0;           // ← kIdleEpoch (0) と同じ値
    return current - 2;
}
```

- `kIdleEpoch = 0` は「Readerが非参加（idle）」を意味する特別値。
- `globalEpoch = 1`（初期状態）のとき、`getSafeEpoch()` は `0` を返す。
- この `0` が `kIdleEpoch` と同一値であるため、「安全なエポック値としての `0`」と「idle状態としての `0`」の区別がつかない。
- 現在 `getSafeEpoch()` はコード内で呼び出されていないため実害はないが、将来使用する際に意図しない動作を引き起こす可能性がある。

**④ 解決方法**

```cpp
uint64_t getSafeEpoch() const noexcept
{
    const uint64_t current = convo::consumeAtomic(globalEpoch, ...);
    // kIdleEpoch (0) との衝突を避けるため、最小値を 1 にする
    if (current < 3) return 1;
    return current - 2;
}
```

または、`getSafeEpoch()` の戻り値の意味を文書化し、`kIdleEpoch` との比較を行わないことを契約として明記する（ただし防御的には値の変更が望ましい）。

---

## P3 — 将来的な改善推奨

### バグ7. `AudioSegmentBuffer::totalSamples` の非アトミックRMW

**① バグの内容**
`totalSamples` 更新が非アトミックなRead-Modify-Writeであり、複数Producerからの同時呼び出しでカウントが消失する。

**② 発生個所**

- ファイル: `src/AudioSegmentBuffer.h`
- 関数: `pushBlock()`
- 行: 51-53

**③ バグの詳細**

```cpp
const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
```

- `consumeAtomic`（read）→ 加算 → `publishAtomic`（write）の3ステップはアトミックではない。
- 2つのスレッドが同時に実行すると、両方が同じ `currentTotal` を読み取り、一方の加算が消失する（いわゆる「ABA問題の軽量版」）。
- **ただし現在の実装では `pushBlock` は単一スレッド（NoiseShaperLearnerのワーカースレッド）からのみ呼ばれているため、実際の競合は発生しない。**

**④ 解決方法**

単一Producerのままなら修正不要。将来マルチProducer対応が必要になった場合のみ、`fetchAddAtomic` を使用してアトミックRMWに変更する:

```cpp
// マルチProducer対応時の修正:
const int clampedAdd = std::min(numSamples, kCapacity - convo::consumeAtomic(totalSamples, ...));
convo::fetchAddAtomic(totalSamples, clampedAdd, std::memory_order_acq_rel);
// ただし kCapacity 上限の処理が複雑になるため、別途 clamp ロジックが必要
```

---

### バグ9. `EQProcessor` の状態更新における例外安全性

**① バグの内容**
`new EQState(*oldState)` で例外発生時に `newState` の所有権処理が不完全。

**② 発生個所**

- ファイル: `src/eqprocessor/EQProcessor.Parameters.cpp`
- 関数: `setBandFrequency()`、`setBandGain()`、`setBandQ()`、`setBandEnabled()` など全パラメータセッター
- 行: 24-35 など

**③ バグの詳細**

```cpp
auto oldState = loadCurrentState(std::memory_order_acquire);
if (oldState == nullptr) return;
auto newState = new EQState(*oldState);  // ← std::bad_alloc の可能性
newState->bands[band].frequency = freq;  // ← 単純代入、非例外
auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel);
```

- `new EQState(*oldState)` が `std::bad_alloc` を投げた場合、関数は例外で終了する。リークは発生しない（newState未割当）。
- `*oldState` のコピー中に別スレッドが `exchangeCurrentState` で状態書き換えを行うと、コピー結果が新旧混在の不整合状態になる（いわゆる「torn snapshot」）。
- ただし `loadCurrentState` はatomicポインタ読み取りであり、コピー開始後も別スレッドが書き換える可能性は常にある — これはコピー方式のロックフリーパターンに内在する制約。

**④ 解決方法**

**オプションA**: コピー中の変更検出を諦め、現状維持（現在の設計意図通り）。コメントで制約を明記。

**オプションB**（強化案）: `exchangeCurrentState` 後に世代番号を比較し、コピー中に変更があった場合は再試行する:

```cpp
retry:
auto oldState = loadCurrentState(std::memory_order_acquire);
const auto genBefore = oldState ? oldState->generation : 0;
auto newState = new EQState(*oldState);
// ... パラメータ変更 ...
auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel);
const auto genAfter = prev ? prev->generation : 0;
if (genBefore != genAfter) {
    // コピー中に変更あり → 新しい状態で再試行
    retireEQStateDeferred(prev);
    // newState を破棄して retry
    delete newState;
    goto retry;
}
```

※ ただし `EQState` の内部構造（generationメンバの有無）に依存するため、要調査。

---

### バグ3. `globalEpoch` 初期値と `getMinReaderEpoch()` の組み合わせ

**① バグの内容**
初期エポック設計により、最初のswap直後にReader不在時でもretiredオブジェクトが解放されうる。

**② 発生個所**

- ファイル: `src/SafeStateSwapper.h`
- コンストラクタ: `globalEpoch(1)`
- 関数: `getMinReaderEpoch()`（Reader不在時は `globalEpoch` を返す）
- 行: 57、287-300

**③ バグの詳細**

- `globalEpoch = 1` で初期化される。
- 最初の `swap()` で epoch1=1（fetchAdd前の値）、globalEpoch=3（2-step bump後）。
- 最初の `swap()` で `oldState = nullptr` の場合、即座にreturnするため実際にretireされるのは2回目の `swap()` 以降。
- 2回目の `swap()` では epoch1=3（すでに安全な値）、Reader不在時は `getMinReaderEpoch()` が `globalEpoch=5` を返す。
- `isOlder(3, 5) = true`。つまり **Readerが1度も参加していなくても解放される**。

これは実際には **正しい動作** である。Readerが誰も参加していなければ、誰も古い状態を参照していないため、RCU安全契約に反しない。問題が発生するのは「Reader参加直後でまだエポック記録前」のような極めて稀なタイミングのみだが、`enterReader()` の `consumeAtomic(globalEpoch, acquire)` が適切な同期を提供する。

**④ 解決方法**

現状の設計で実用上問題ない。もし完全性を高めるなら、`getMinReaderEpoch()` のReader不在時戻り値を `globalEpoch - 2`（`getSafeEpoch()` 相当）に変更する方法もあるが、`isOlder` 判定がより保守的になるだけで本質的な安全性向上にはならない。

理論上の懸念として、`getSafeEpoch()` のバグ12と組み合わせて `kIdleEpoch=0` が意図しない比較を引き起こす可能性はあるが、現在 `getSafeEpoch()` は未使用のため実害なし。

---

## Info — 参考情報

### バグ13. `NonOwningPtr` のポインタ provenance 問題

**① バグの内容**
ポインタを `std::uintptr_t` に変換して保持する手法が、C++標準の厳密解釈では実装定義動作にあたる。

**② 発生個所**

- ファイル: `src/audioengine/AtomicAccess.h`
- クラス: `NonOwningPtr<T>`
- 行: 12-37

**③ バグの詳細**

```cpp
constexpr NonOwningPtr(T* ptr) noexcept
    : bits(static_cast<std::uintptr_t>(reinterpret_cast<std::uintptr_t>(ptr))) {}
```

- `reinterpret_cast<std::uintptr_t>(ptr)` はポインタを整数に変換する操作であり、C++標準の厳密な「ポインタ provenance（出所）」ルールでは実装定義（implementation-defined）とされる。
- 主要ターゲット（x64 Windows/MSVC）では問題なく動作する。
- CHERI（Capability Hardware Enhanced RISC Instructions）などの新しいアーキテクチャや、Strict Pointer Provenance (SPP) を適用するツールでは問題が検出される可能性がある。

**④ 解決方法**

x64 Windowsが唯一のターゲットである現在は **修正不要**。将来CHERI等のアーキテクチャ対応が必要になった場合に、`std::atomic<T*>` の直接使用に変更することを検討する。

---

## 修正推奨サマリー

| # | バグ | 優先度 | 難易度 | リスク | 推奨アクション |
|---|------|--------|--------|--------|---------------|
| 4 | updateConvolverState排他 | **P0** | 低（CAS置換） | 中（Release競合） | **compareExchangeAtomicに修正** |
| 11 | drainAllRetired強制解放 | **P1** | 低（コメント強化） | 低〜中（DTOR順序依存） | コメントとアサーション追加 |
| 12 | getSafeEpoch値衝突 | **P2** | 低（戻り値変更） | 低（未使用） | `return 1` に変更 |
| 7 | totalSamples RMW | **P3** | 低 | なし（単一スレッド） | 現状維持。将来MP時に修正 |
| 9 | EQState例外安全性 | **P3** | 中（設計変更） | なし | 現状維持。コメントで制約明記 |
| 3 | globalEpoch初期値 | **P3** | 低 | なし | 現状維持 |
| 13 | NonOwningPtr provenance | Info | — | なし | 現状維持 |

**即時対応**: バグ4（P0）の `exchangeAtomic` → `compareExchangeAtomic` 修正
**推奨対応**: バグ11（P1）のコメント強化、バグ12（P2）の戻り値修正
**将来対応**: P3 の3件は現状維持で問題なし
