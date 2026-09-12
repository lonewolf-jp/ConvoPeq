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

### RB-01: `pendingIntentCount()` が fallback queue を計上していない ✅ 解決済み（2026-09-12 現行ソース再判定）

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

**解決（2026-09-12 現行ソース再判定 — read-only audit）**:
- `LifetimeState::pendingIntentCount()`（`ISRRetire.cpp:183-190`）は **`mainPending + fbPending` で fallback 計上済み** — 本修正案と同一形が実装済み。
- `approxQueueDepth()`（`ISRRetire.h:100-107`・「★ B14: Queue Pressure 診断」）も同様に `mainPending + fbPending`。
- `fallbackQueueDepth` も Threading 側で「★ P1-9: ring+fallback 合計」として実装済み。
- 回帰カバレッジ: `ShutdownRetireIntentDrainTests.cpp`（pendingIntentCount drain 契約 — CTest 40/40 内）。
- Required action: NONE。stale bug record。

---

### RB-11: `setProcessingOrder()` に `sendChangeMessage()` 欠落 ✅ 解決済み（2026-09-12 現行ソース再判定）

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

**解決（2026-09-12 現行ソース再判定 — read-only audit）**:
- 現行 `setProcessingOrder`（`AudioEngine.Parameters.cpp:268-275`）には **`sendChangeMessage()` が :274 に存在**（記録時の構造から変更済み）。
- 他 setter（:161/:172/:188）と同形の対称実装。唯一の呼び出し元は `MainWindow.cpp:1372/1377`（Non-RT）。
- Required action: NONE。stale bug record。

---

## P2 — 改善提案

### RB-05: delayLineBuf 書き込みラップリスク (capacity < partSize) ✅ 解決済み（2026-09-12 検証閉包）

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

**解決（2026-09-12 検証閉包 — work57 B13 Policy R 最終監査）**:

- capacity 式は **修正案どおり実装済み**（`MKLNonUniformConvolver.cpp:1007` —
  `((prevLayerTotalSamples + l.partSize + m_maxBlockSize + 15) / 16) * 16`）。
  実測: **L1 = 2624 ≥ partSize 512、L2 = 38976 ≥ partSize 4096** — `capacity < partSize` は現行構成で不成立。
- work57 B13 Policy R 修復（2026-09-12）で **I3 構造 gate**（`cap ≥ o_L − lead + 2P`）を SetImpulse に追加 —
  全テスト case で OK（T3 余裕 0・T7 L2 余裕 64）。
- リングの正常折り返し（write wrap）は T7 run3 実測で **L1 39.61 周 / L2 2.63 周** 発生 — wrap 下で
  oPE = 0（全 anchor）・M1 波形 null −309〜−311 dB 通過。音声破綻なし。
- 残置メモ（別枠・現行 48kHz/64 仕様では不発）: `capacity % blockSize != 0` の構成では read 側
  二分割（wrap-crossing read）が発火し得るが、T7 run3 実測の wrap-crossing read は **0 件**
  （2624 / 38976 はともに B=64 の倍数のため `readOffset + B ≤ cap` が常に成立）。
  RB-05 の本体懸念（同一書き込み内 `n > capacity` 上書き）は capacity ≥ partSize 保証で解消済み。
- サマリ表のステータスも ✅ 解決済 に更新。

---

### RB-02: `goto final_drop` の構造的問題 ✅ 解決済み（2026-09-12 現行ソース再判定 — 構造自体が不存在）

**発見経緯**: 第二次監査 #2。ast-grep 構造検出。

**現状**: `goto final_drop` が `if` の `else` ブロック内のラベルにジャンプする構造。
C++ 上 UB ではないが制御フローが追いにくい。

**修正案**: `final_drop` ラベルを if/else の外に移動し、else ブロックを独立 if に書き直す。

**ファイル**: `src/audioengine/ISRRetire.cpp` emitRetireIntent()

**因果関係**: ✅ B14 改修で新規に発生。

**解決（2026-09-12 現行ソース再判定 — read-only audit）**:
- **`final_drop` は現行ソースに存在しない**: `rg "final_drop" src/` = 0 件・`ConvoPeq.md`（2026-09-12 16:00:44 版・HEAD 2451d964 と同一内容）でも 0 件。**`goto` 自体が src/audioengine/ 全体で 0 件** — 台帳の修正案（ラベル移動）を超えて goto を使わない直線構造に再設計済み。
- 現行 `emitRetireIntent`（`ISRRetire.cpp:23-60`）: Vyukov ticket → bounded spin（64）→ 失敗時 tombstone + fallback enqueue（mutex + capacity チェック + overflowCount）。制御フロー追いにくさの原因は消滅。
- Non-RT 境界は `emitRetireIntentNonRT`（cpp:94-103・work92 big 1-7 リネーム）で明文化 — 呼び出し元 `AudioEngine.Commit.cpp:485` は全て非 RT 確認済み（`ISRRuntimePublicationCoordinator.cpp:352/364/390` は OverflowRing drain 系・同じく非 RT）。
- 回帰カバレッジ: `ShutdownRetireIntentDrainTests.cpp`（Case 1-8）+ `PriorityIntegrationTests.cpp`（CTest 40/40 内）。
- Required action: NONE。stale bug record（symptom removed）。

---

## P3 — 軽微な改善

### RB-07: `dryBypassBufferFloatL/R` 完全なデッドコード ✅ 解決済み（2026-09-12 現行ソース再判定 — メンバ削除済み）

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

**解決（2026-09-12 現行ソース再判定 — read-only audit）**:
- **メンバ自体が現行ソースから削除済み**: `rg "dryBypassBufferFloat" src/` = 0 件・`ConvoPeq.md`（16:00:44 版）も 0 件。
- Float bypass path は **dryBypassBufferDoubleL/R を使用**（`DSPCoreFloat.cpp:254-258`「★ B01: Float 版 dry 信号保存 (Double 版と同じ dryBypassBufferDouble を使用)」+ :409-438 blend 経路・`dryBypassCapacityDouble` は AudioEngine.h:1017 で現行使用中）。
- historical dead-code 判定（P3-CONFIRMED）は正当で、削除が実施済み。Required action: NONE。

---

### RB-03: `fallbackQueuePeak_` CAS loop が mutex 下で冗長 ✅ historical claim 解決済み + 現行観察（2026-09-12 再判定）

**発見経緯**: 第二次監査 #3。

**現状**: mutex 保護下で CAS ループを実行しているが、mutex により他スレッドの変更が排除されているため単なる publishAtomic で十分。

**修正案**: `convo::publishAtomic(fallbackQueuePeak_, fbCountVal, memory_order_release)` に単純化。

**ファイル**: `src/audioengine/ISRRetire.cpp` emitRetireIntent()

**解決 + 現行観察（2026-09-12 現行ソース再判定 — read-only audit）**:
- **historical claim（CAS 冗長）は解決済み**: 現行 write は 1 箇所（`ISRRetire.cpp:49`）で **publishAtomic = 単純 store**（`AtomicAccess.h:52-57` — `std::atomic_store_explicit`・CAS ではない）。write は fallbackMutex_ lock 下（:44-50）。
- **現行観察 1 — peak 意味論の不一致**: 現行 store は「enqueue 成功時の fallbackCount_ 現在値」の無条件 store で、**単調非減少（high-watermark）ではない**（drain 後の再 enqueue で値が下がる）。`fallbackHighWatermark()` / `fallbackQueuePeak_` の名前に反する。
- **現行観察 2 — 消費者 0 件**: `fallbackHighWatermark()` / `fallbackOccupancy()` の呼び出し元は src/ 全体で 0 件（HealthMonitor / telemetry 未接続）— **dead metric**（symbol 7 件すべて declaration/definition — 分類 C）。
- **Disposition（2026-09-12 RB-03 Disposition Audit）**: **Option C — dead metric として削除候補に確定**。削除スコープ: `fallbackQueuePeak_`（h:167）+ `fallbackHighWatermark()`（h:75/cpp:236-238）+ write 行（cpp:49・enqueue 自体は維持）。fallbackOccupancy は別枠（同 consumer 0 件だが fallbackCount_ の 1 行 wrapper）。**実装は別 commit**（P1-x-I commit に混ぜない — 責務分離）。ISR-OBS-001/HM-001/ATM-001/002 いずれも PASS（metric 処置により不変条件は変化しない）。

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
| **RB-01** | `pendingIntentCount()` fallback 不計上 | ✅ 解決済 | fallback 計上実装済み（cpp:183-190・h:100-107）— work57 RB audit で閉包 | ISRRetire.cpp | 閉包 |
| **RB-11** | `setProcessingOrder` sendChangeMessage 欠落 | ✅ 解決済 | sendChangeMessage 実装済み（Parameters.cpp:274）— work57 RB audit で閉包 | Parameters.cpp | 閉包 |
| **RB-05** | delayLineBuf capacity < partSize（コード事実確認） | ✅ 解決済 | capacity 式実装済み（cpp:1007）+ I3 gate 実測（work57 B13 監査で閉包） | MKLNonUniformConvolver.cpp | 閉包 |
| **RB-02** | `goto final_drop` 構造的問題 | ✅ 解決済 | goto/final_drop 自体が現行ソースに不存在（work57 RB audit で閉包） | ISRRetire.cpp | 閉包 |
| **RB-07** | `dryBypassBufferFloatL/R` デッドコード | ✅ 解決済 | メンバ自体が現行ソースから削除済み — work57 RB audit で閉包 | AudioEngine.h | 閉包 |
| **RB-03** | CAS loop 冗長 / dead metric | ✅ 解決済（Gate 6 commit `0428951e` で削除・Gate 7 PASS・clang-tidy error 0）| ISRRetire.cpp | 閉包 |
| **RB-08** | MT-NUPC-03 Debug 異常終了 | ✅ 解決済 | バッファ範囲外アクセス | MT-NUPC-Measurement.cpp | 修正済 |
