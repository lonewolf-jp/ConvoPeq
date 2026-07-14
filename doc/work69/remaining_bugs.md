# ConvoPeq 残存バグリスト

> 作成日: 2026-07-15 (v2.8)
> ベース: `doc/work69/bug_final_report.md` (v1.29)
> 根拠: 第一次監査(5件) + 第二次監査(4件) + 設計書レビュー指摘 + 深堀調査
>
> 本リストは「実際のバグまたは修正すべきコード上の問題」のみを収録。
> 設計差分やスタイル選択は「設計差分一覧」として別管理する。

---

## 凡例

| 記号 | 意味 |
|------|------|
| 🔴 P0 | 致命的：即対応必須 |
| 🟡 P1 | 重要：次回改修スコープ |
| 🟢 P2 | 軽微：改善提案 |
| 🔵 P3 | 情報：現状問題なし、監査結果の記録 |

---

## P1 — 要修正

### RB-01: `pendingIntentCount()` が fallback queue を計上していない 🟡 P1

**発見経緯**: 第二次監査 #1。graphify path 確認 + semble 検索で発見。

**現状**:
```cpp
// ISRRetire.cpp
uint64_t RetireRuntime::pendingIntentCount() const noexcept
{
    const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, ...);
    const uint64_t consumed = convo::consumeAtomic(dequeuePos_, ...);
    return (enqueued > consumed) ? (enqueued - consumed) : 0;
    // ↑ main queue のみ。fallback 不計上
}
```

**影響**: `setRetireBacklogCount()` → UI バックログ表示が不正確。HealthMonitor の `droppedIntentCount()` は main+fallback をカバーするため影響軽微だが、`pendingIntentCount()` ベースのメトリクスが過少評価になる。

**ISR Runtime 視点**: Observer 層の欠陥。fallback queue の Pending Intent が Observer に反映されず、バックログ観測値が実際より少なく報告される。ISR の観測系は実状態の正確な観測が前提であり、過少報告は HealthMonitor の閾値判定や UI 表示に影響する。

**修正案**:
```cpp
std::uint64_t RetireRuntime::pendingIntentCount() const noexcept
{
    const uint64_t enqueued = convo::consumeAtomic(enqueueTicket_, std::memory_order_acquire);
    const uint64_t consumed = convo::consumeAtomic(dequeuePos_, std::memory_order_acquire);
    const uint64_t mainPending = (enqueued > consumed) ? (enqueued - consumed) : 0;
    const uint64_t fbPending = convo::consumeAtomic(fallbackCount_, std::memory_order_relaxed);
    return mainPending + fbPending;
}
```

**ファイル**: `src/audioengine/ISRRetire.cpp` (pendingIntentCount)、`ISRRetire.h` (approxQueueDepth も同様)

**因果関係**: ✅ B14 改修の一部。既存コードでは `retireIntentHead_/Tail_` で維持していた fallback カウントが Vyukov 移行で欠落。

---

### RB-11: `setProcessingOrder()` に `sendChangeMessage()` 欠落 🟡 P1

**発見経緯**: 元文書 `bug_final_report.md` Appendix K で確定された B0 バグ。
第一次監査では見落とし。第二次監査でコード確認により確定。

**現状**:
```cpp
// AudioEngine.Parameters.cpp:268-275
void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    submitRebuildIntent(...);
    applyDefaultsForCurrentMode();
    // ★ sendChangeMessage() 欠落
}
```

同等の setter は全件 `sendChangeMessage()` を持つ:
- `setEqBypassRequested()` — line 161 ✅
- `setConvolverBypassRequested()` — line 172 ✅

**影響**: `setProcessingOrder()` 呼出後に UI が変更を検知できない。処理順序変更が即座に UI に反映されず、ユーザーに「変更が効いていない」ように見える。

**修正案**: `applyDefaultsForCurrentMode();` の直後に `sendChangeMessage();` を追加。

**ファイル**: `src/audioengine/AudioEngine.Parameters.cpp`

**因果関係**: 🔴 B14/B13/B20 等とは独立した既存バグ。work69 スコープ確認時に発見され放置。

---

## P2 — 改善提案

### RB-05: delayLineBuf 書き込みラップリスク (capacity < partSize) 🟢 P2

**発見経緯**: 第二次監査 #4 → 第三次レビューで実コード引用により確認。

**コード事実 1 — Add/Get の呼び出し順序**:

`ConvolverProcessor.Runtime.cpp:1154-1155`:
```cpp
nucConvolvers[channel]->Add(in, numSamples);  // ← 先に Add
const int got = nucConvolvers[channel]->Get(out, numSamples);  // ← すぐ後に Get
```
同一コールバック内で `Add()` → `Get()` の直列順序が確定。Add() 内の delayLineWrite 後、即座に Get() が走る。

**コード事実 2 — delayLineWrite() の実装**:

`MKLNonUniformConvolver.cpp:1727-1736`:
```cpp
void MKLNonUniformConvolver::delayLineWrite(Layer& l, const double* src, int n) noexcept
{
    const size_t writeOffset = static_cast<size_t>(
        l.delayWriteCursor % static_cast<uint64_t>(l.delayLineCapacity));
    const int remain = l.delayLineCapacity - static_cast<int>(writeOffset);
    const int first = std::min(n, remain);
    juce::FloatVectorOperations::copy(l.delayLineBuf + writeOffset, src, first);
    if (first < n)    // ★ n > capacity の場合、ここでラップして先頭を上書き
        juce::FloatVectorOperations::copy(l.delayLineBuf, src + first, n - first);
    l.delayWriteCursor += static_cast<uint64_t>(n);
}
```
- **`n` (`partSize`) が `capacity` を超える場合でも一切のガードなし**
- `first = min(n, remain)` でバッファ終端まで書き、残りは先頭にラップ
- **`jassert(n <= capacity)` や `jassert(capacity >= partSize)` は存在しない**
- → `n > capacity` では同一書き込み内でリング先頭へ折り返す

**コード事実 3 — パラメータの関係**:

`MKLNonUniformConvolver.cpp:807-809`:
```cpp
const int l0Part = nextPowerOfTwo(max(blockSize, 64));      // = blockSize
const int l1Part = l0Part * tailL1L2Mult;                     // = blockSize × 8 (+
const int l2Part = l1Part * tailL1L2Mult;
```

`MKLNonUniformConvolver.cpp:1102-1106`:
```cpp
l.outputDelaySamples = prevLayerTotalSamples;  // 先行レイヤーの IR 総長
l.delayLineCapacity = ((prevLayerTotalSamples + m_maxBlockSize * 3 + 15) / 16) * 16;
```

**capacity < partSize が成立する条件**:

```
Lx.partSize > outputDelaySamples + 3×blockSize
```
L1 の場合: `L1.partSize = 8×blockSize` なので:
```
8×blockSize > outputDelaySamples + 3×blockSize
→ outputDelaySamples < 5×blockSize
```

つまり、**先行レイヤー (L0) の IR 総長が 5×blockSize 未満**の場合に `capacity < L1.partSize` となる。これは短い IR や小さい `tailStartSeconds` の設定で現実的なシナリオである。

**例** (blockSize=512):
| L0 パーティション数 | L0_len | L1.outputDelaySamples | capacity | L1.partSize | capacity < partSize? |
|---|---|---|---|---|---|
| 1 | 512 | 512 | 2048 | 4096 | **✅ 成立** |
| 2 | 1024 | 1024 | 2560 | 4096 | **✅ 成立** |
| 4 | 2048 | 2048 | 3584 | 4096 | **✅ 成立** |
| 5 | 2560 | 2560 | 4096 | 4096 | 境界 (等しい) |
| 8 | 4096 | 4096 | 5632 | 4096 | 安全 |

**注意**: `capacity < partSize` は delayLineWrite 内での書き込みラップ（リング先頭への折り返し）を引き起こすが、これが直ちに音声破綻を意味するわけではない。実際の影響は `delayLineReadAdd()`（Get 側）が `delayWriteCursor` と `delayReadCursor` をどう管理するかに依存する。ただし `capacity < partSize` となる設計は保守上理解しづらく、`capacity >= partSize` を保証する方が設計意図が明確になる。

**修正案**:
```cpp
// l.partSize は line 847 で設定済み (delayLineCapacity 計算より後でも問題なし)
l.delayLineCapacity = ((prevLayerTotalSamples + l.partSize + m_maxBlockSize + 15) / 16) * 16;
```

これにより `capacity >= partSize` が保証され、**1回の delayLineWrite 呼び出しでバッファ容量を超える状態は解消される**。リングバッファとしての通常の折り返し（writeOffset がバッファ終端に達した場合の先頭へのラップ）は引き続き発生し得る。

**コード事実 4 — delayLineWrite の呼出箇所は1箇所のみ、第3引数は常に l.partSize**:

`ripgrep` による全コードベース検索結果:
```
src/MKLNonUniformConvolver.cpp:1630:
    delayLineWrite(l, l.tailOutputBuf, l.partSize);   // 唯一の呼出し
src/MKLNonUniformConvolver.h:402:
    void delayLineWrite(Layer& l, const double* src, int n) noexcept;  // 宣言
```
- `delayLineWrite()` の呼出しは **この1箇所のみ**
- 第3引数は **常に `l.partSize`**（`min()` 等による削減なし）
- `l.partSize` はレイヤー依存: L0=blockSize, L1=L0×8, L2=L1×8
- `delayLineBuf != nullptr` の条件により、L1/L2 でのみ呼ばれる (L0 は非対象)

**コード事実 5 — `prevLayerTotalSamples` の正体（実 IR サンプル数であることを確認）**:

`MKLNonUniformConvolver.cpp:811-827, 833, 1120`:
```cpp
int prevLayerTotalSamples = 0;                               // line 833
// ...
l0Len = min(irLen, ...);          // 実 IR サンプル数 (nextPowerOfTwo 非使用)
l1Len = max(0, min(irLen - l0Len, ...));  // 残りの実 IR サンプル数
l2Len = max(0, irLen - l0Len - l1Len);     // 残りの実 IR サンプル数
LayerCfg cfgs[] = { {0, l0Len, ...}, {l1Offset, l1Len, ...}, {l2Offset, l2Len, ...} };
// ...
prevLayerTotalSamples += cfgs[li].len;                        // line 1120
```

- `cfgs[li].len` は **実 IR サンプル数**（`nextPowerOfTwo` や `partSize×numParts` ではない）
- したがって `outputDelaySamples = prevLayerTotalSamples` は「先行レイヤーの IR 実サンプル数」
- 上記の表 (`L0_len = 512, 1024, 2048, ...`) の値は正確である

**ファイル**: `src/MKLNonUniformConvolver.cpp` (capacity 計算行)

**因果関係**: ✅ B13 改修で新規実装。

---

### RB-02: `goto final_drop` の構造的問題 🟢 P2

**発見経緯**: 第二次監査 #2。ast-grep 構造検出。

**現状**: `goto final_drop` が `if` の `else` ブロック内のラベルにジャンプする構造。
C++ 上 UB ではないが制御フローが追いにくい。

**修正案**: `final_drop` ラベルを if/else の外に移動し、else ブロックを独立 if に書き直す。

**ファイル**: `src/audioengine/ISRRetire.cpp` emitRetireIntent()

**因果関係**: ✅ B14 改修で新規に発生。

---

## P3 — 軽微な改善

### RB-07: `dryBypassBufferFloatL/R` 完全なデッドコード 🟢 P3

**発見経緯**: 第一次監査 #6 / 第二次監査 #6。semble 検索 + 全参照確認。

**現状**:
```cpp
// AudioEngine.h:989-990
convo::ScopedAlignedPtr<float> dryBypassBufferFloatL;   // ★ 完全未使用
convo::ScopedAlignedPtr<float> dryBypassBufferFloatR;   // ★ 完全未使用
```

Float 版 bypass blend は `dryBypassBufferDouble` を使用するため、これらのメンバは未使用。
`DSPCoreLifecycle.cpp` でも確保されていない。

**影響**: 各 DSPCore インスタンスに未使用の `ScopedAlignedPtr<float>` が 2 つ存在。16 バイトのメモリ増加は軽微だが、将来の保守者が「Float 版バッファが存在するなら使うべき」と誤解するリスクが大きい。特に DSPCore 構造体は clone/publish 対象であり、予期しないディープコピーや初期化漏れの原因になりうる。

**修正案**: 宣言ごと削除。

**ファイル**: `src/audioengine/AudioEngine.h` DSPCore 構造体内

**因果関係**: ✅ B01 改修で Float 版バッファを Double に統合した際の残骸。

---

### RB-03: `fallbackQueuePeak_` CAS loop が mutex 下で冗長 🟢 P3

**発見経緯**: 第二次監査 #3。

**現状**: mutex 保護下で CAS ループを実行しているが、mutex により他スレッドの変更が排除されているため単なる publishAtomic で十分。

**修正案**: `convo::publishAtomic(fallbackQueuePeak_, fbCountVal, memory_order_release)` に単純化。

**ファイル**: `src/audioengine/ISRRetire.cpp` emitRetireIntent()

**因果関係**: ✅ B14 改修で新規に発生。

---

## 解決済

### RB-08: MT-NUPC-03 Partition Boundary テストが Debug で異常終了 → 解決済 ✅

**発見経緯**: B13 自動測定実装時に確認。特定の IR 長（2047 等）で Debug ビルドが異常終了。

**原因**: `measureLayerDelays()` の出力バッファサイズ計算にバグ。`totalOutputSamples = irLength * 2` は blockSize の倍数とは限らず（例: irLen=2047, blockSize=512 → 4094）、while ループ `totalProcessed < totalOutputSamples` で `Get()` が blockSize=512 を返すため最終的な `totalProcessed` が `totalOutputSamples` を超過する（4096 > 4094）。超過分（output[4094], output[4095]）がピーク分析ループで範囲外アクセスとなり、Debug ビルドの `/RTC1` スタックチェックで検出されクラッシュ。

**なぜ irLen=1024 は異常なし?**: `2048 = blockSize×4` の倍数であり、`totalProcessed` が正確に一致するため。

**修正 (2箇所)**:
1. バッファサイズを blockSize の倍数に切り上げ: `totalOutputSamples = ((irLen*2 + blockSize-1) / blockSize) * blockSize`
2. 防御的に分析ループの上限を `min(output.size(), totalProcessed)` で制限

**検証**: Debug ビルド全テスト通過 (`=== ALL PASSED ===`)。

**ファイル**: `src/tests/MT-NUPC-Measurement.cpp` `measureLayerDelays()`
- irLen=2047 では `l0Len = 2047`, `l1Len = 0`, `l2Len = 0` → `m_numActiveLayers = 1`
- つまり L1/L2 が生成されず delay line も未確保
- `jassert(outputDelaySamples > 0)` は発火条件に該当しない
- 原因は未特定。JUCE MessageManager の初期化タイミングか、Debug assertion の別経路の可能性

**次のステップ**:
1. `__debugbreak()` または `OutputDebugString` で正確な落ち箇所を特定
2. `NUC_DEBUG_GUARDS` が Debug ビルドで有効か確認
3. `juce::FloatVectorOperations::clear` が nullptr で呼ばれていないか確認

**ファイル**: `src/tests/MT-NUPC-Measurement.cpp`
**因果関係**: 🔵 B13 導入前の既存挙動。非一様レイヤー構成の設計特性。

---

## 参考: 設計差分一覧（バグではない）

以下の項目は `bug_final_report.md` の仕様書と実装の差分ですが、動作に影響しないため
残存バグリストから除外し、設計差分として別管理します。

| ID | 項目 | 理由 |
|----|------|------|
| ~~RB-04~~ | `ScheduledRetireIntent` / `RetireBatch` 未使用 | `stable_sort` でも動作完全成立。実装例の選択 |
| ~~RB-06~~ | `isDelayCompatibleWith` 未実装 | Runtime 生成時に Layer 構成固定のため実害なし |
| ~~RB-09~~ | `StoredConfig` 構造体不使用 | 個別メンバでも機能同等。スタイルの問題 |
| ~~RB-10~~ | `static_assert` 型一致未実装 | 型不一致はコンパイル時に自然検出。防御策の有無 |

---

## サマリ（修正対象のみ）

| ID | 項目 | 重要度 | 原因 | ファイル | 工数 |
|----|------|--------|------|---------|------|
| **RB-01** | `pendingIntentCount()` fallback 不計上 | 🟡 P1 | B14 改修起因 | ISRRetire.cpp | 10分 |
| **RB-11** | `setProcessingOrder` sendChangeMessage 欠落 | 🟡 P1 | 既存バグ | Parameters.cpp | 5分 |
| **RB-05** | delayLineBuf capacity < partSize（コード事実確認） | 🟢 P2 | B13 改修起因 | MKLNonUniformConvolver.cpp | 15分 |
| **RB-02** | `goto final_drop` 構造的問題 | 🟢 P2 | B14 改修起因 | ISRRetire.cpp | 5分 |
| **RB-07** | `dryBypassBufferFloatL/R` デッドコード | 🟢 P3 | B01 改修残骸 | AudioEngine.h | 5分 |
| **RB-03** | CAS loop 冗長 | 🟢 P3 | B14 改修起因 | ISRRetire.cpp | 5分 |
| **RB-08** | MT-NUPC-03 Debug 異常終了 | ✅ 解決済 | バッファ範囲外アクセス | MT-NUPC-Measurement.cpp | 修正済 |
