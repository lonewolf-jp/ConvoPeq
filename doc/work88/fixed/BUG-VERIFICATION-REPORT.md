# BUG 検証レポート — 2026-07-26

**検証者**: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)
**検証日**: 2026-07-26
**プロジェクト**: ConvoPeq
**ブランチ**: main

## 検証方法

- **ソースコード検索**: ripgrep (WSL)、AiDex MCP
- **コード解析**: serena MCP、semble-search
- **型・シンボル解析**: WSL grep/awk/find
- **技術文献調査**: Web検索 (Intel AVX-SSE penalty docs, C++ memory model, Vyukov MPMC queue)

---

## 検証サマリ

| # | BUG-ID | カテゴリ | リスク | 検証結果 | 修正優先度 |
|---|--------|---------|-------|---------|-----------|
| 1 | BUG-001 | 信号処理/音質劣化 | **HIGH** | ✅ Confirmed | 🔴 高 |
| 2 | BUG-002 | 計測機能欠落 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
| 3 | BUG-003 | 計測機能欠落 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
| 4 | BUG-004 | パフォーマンス | **LOW** | ✅ Confirmed | 🟢 低 |
| 5 | BUG-005 | 整数オーバーフロー | **LOW** | ⚠️ Partially Confirmed | 🟢 低 |
| 6 | BUG-006 | 整数オーバーフロー | **LOW** | ⚠️ Partially Confirmed | 🟢 低 |
| 7 | BUG-007 | 整数オーバーフロー | **LOW** | ⚠️ Partially Confirmed | 🟢 低 |
| 8 | BUG-008 | 整数オーバーフロー | **LOW** | ⚠️ Partially Confirmed | 🟢 低 |
| 9 | BUG-009 | パフォーマンス | **MEDIUM** | ✅ Confirmed | 🟡 中 |
| 10 | BUG-010 | メモリリーク | **HIGH** | ✅ Confirmed | 🔴 高 |
| 11 | BUG4 | イベント喪失 | **CRITICAL** | ✅ Confirmed | 🔴 高 |
| 12 | BUG9 | データ競合 | **MEDIUM** | ✅ Confirmed | 🟡 中 |
| 13 | BUG10 | ロックフリー破綻 | **HIGH** | ✅ Confirmed | 🔴 高 |
| 14 | BUG11 | データ競合 (UB) | **CRITICAL** | ✅ Confirmed | 🔴 高 |
| 15 | BUG12 | Use-after-free | **CRITICAL** | ✅ Confirmed | 🔴 高 |
| 16 | BUG13 | 設計欠陥 (UAF) | **HIGH** | ✅ Confirmed | 🔴 高 |
| 17 | BUG16 | 単位不一致/デッドコード | **HIGH** | ✅ Confirmed | 🟡 中 |
| 18 | BUG17 | 単位不一致/誤検出 | **HIGH** | ✅ Confirmed | 🔴 高 |

---

## 詳細検証結果

### BUG-001: Float処理パスにPeak Limiterが未実装

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.Processing.DSPCoreIO.cpp:506-514` |

**検証内容**:
- `rg "peakLimiter.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` → Line 710: **存在** ✅
- `rg "peakLimiter.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` → **0 hits** ❌
- Floatパスは `juce::jlimit(-kOutputHeadroom, kOutputHeadroom, ...)` によるハードクリップのみ

**ソースコード対比**:

```cpp
// DSPCoreDouble.cpp:710 — Doubleパス（正しい）
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);

// DSPCoreIO.cpp:506-514 — Floatパス（バグ）
for (int i = 0; i < numSamples; ++i)
    dstL[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]));
// peakLimiter.processBlock() が存在しない
```

**影響**: 入力が -1.5dBFS を超えた場合、Doubleパスではソフトニー圧縮がかかるが、Floatパスでは直接ハードクリップ → 高調波歪み・エイリアシングノイズ。

**補足**: `DSPCoreIO.cpp` に `peakLimiter` メンバ変数がそもそも存在しない可能性がある。修正にはメンバ追加または共通化の設計判断が必要。

---

### BUG-002: Float処理パスでLoudness Meterが未実行

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.Processing.DSPCoreIO.cpp:375` 周辺 |

**検証内容**:
- `rg "loudnessMeter.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` → Line 701: **存在** ✅
- `rg "loudnessMeter.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` → **0 hits** ❌
- DSPCoreIO.cpp 全体で `loudnessMeter` 自体が0参照

**影響**: FloatパスでLUFSメーターが常に0または未更新。EBU R128準拠のラウドネス測定が機能しない。

---

### BUG-003: Float処理パスでTruePeak Detectorが未実行

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.Processing.DSPCoreIO.cpp:376-378` 周辺 |

**検証内容**:
- `rg "truePeakDetector.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` → Line 698: **存在** ✅
- `rg "truePeakDetector.processBlock" src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` → **0 hits** ❌

**影響**: BS.1770準拠のTruePeak測定がFloatパスで機能しない。BUG-002と同一箇所での欠落。

---

### BUG-004: Float処理パスに `_mm256_zeroupper()` が欠落

| 項目 | 内容 |
|------|------|
| **リスク** | LOW |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.Processing.DSPCoreIO.cpp:470-505` NaN/Inf Scrub後 |

**検証内容**:
- `rg "_mm256_zeroupper" src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` → Lines 741-742: **存在** ✅ (コメント付き)
- `rg "_mm256_zeroupper" src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` → **0 hits** ❌

**Web検証**: Intel「Avoiding AVX-SSE Transition Penalties」— AVX→SSE遷移時には `VZEROUPPER` が必要。Skylake以降もペナルティは軽減されているが完全には除去されていない。HackerNews (2016):「SSE code 6 times slower without VZEROUPPER on Skylake」

**影響**: Haswell/Broadwell: ~60 cycle/命令のペナルティ。小バッファ (~32 samples) での影響が相対的に大きい。

---

### BUG-005〜008: `static_cast<size_t>` 欠落による整数昇格の疑義

| # | ファイル | 行 | コード | 状態 |
|---|---------|-----|-------|------|
| BUG-005 | `MKLNonUniformConvolver.cpp` | 1488 | `memset(dst, 0, n * sizeof(double));` | ❌ キャストなし |
| BUG-005 | `MKLNonUniformConvolver.cpp` | 1501 | `memset(dst + toRead, 0, (n - toRead) * sizeof(double));` | ❌ キャストなし + アンダーフローリスク |
| BUG-006 | `ConvolverProcessor.h` | 821-822 | `memcpy(..., irDataLength * sizeof(double));` | ❌ キャストなし |
| BUG-007 | `ConvolverProcessor.Runtime.cpp` | 388-390 | `memcpy(..., samplesFirst * sizeof(double));` | ❌ キャストなし |
| BUG-008 | `MKLNonUniformConvolver.cpp` | 1030,1037,1045,1080-1089,1156,1384,1417,1488,1501,1538,1540,1628,1637 | 各種 memcpy/memset | ❌ キャストなし (13箇所) |

**検証結果**: ⚠️ **Partially Confirmed**

**理由**:
- ソースコード上で `static_cast<size_t>` 未使用のパターンが存在することは確認
- ただしMSVC 64-bitでは `int * size_t` → `size_t` への暗黙昇格が正しく機能するため、現実的なオーバーフローリスクは極めて低い
- 同一ファイル内に `static_cast<size_t>` を使用した安全なコードと混在しており、**コーディング規約の一貫性** の観点からの問題

**特に注意**: Line 1501 の `(n - toRead)` は、`toRead > n` の場合に負のintとなり、`size_t` 昇格後に巨大なunsigned値として `memset` に渡る可能性がある（バッファオーバーフロー）。

---

### BUG-009: 4ファイルで `_mm256_zeroupper()` が不足

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **検証結果** | ✅ **Confirmed** |

**検証結果**:

| ファイル | AVX2使用 | `_mm256_zeroupper()` | 状態 |
|---------|---------|---------------------|------|
| `ConvolverProcessor.Runtime.cpp` | Lines 465-480, 522-530, 617-645 | **なし** | ❌ |
| `CustomInputOversampler.cpp` | `isBadSampleV`, `loadStride2`, big-tap FIR | **なし** | ❌ |
| `DSPCoreFloat.cpp` | `applyGainRampBlockAVX2`, `softClipBlockAVX2` | **なし** | ❌ |
| `AudioEngine.EQResponse.cpp` | 全体で `__m256d` 多用 | **なし** | ❌ |

**正常に使われているファイル**: `DSPCoreDouble.cpp:742`, `LoudnessMeter.cpp:93`, `TruePeakDetector.cpp:181`

---

### BUG-010: retireEQStateDeferred/retireBandNodeDeferred の戻り値無視

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `EQProcessor.Core.cpp`, `EQProcessor.Parameters.cpp`, `EQProcessor.Coefficients.cpp` |

**検証内容**:
- `retireEQStateDeferred()` = `bool` 戻り値 (Line 93: `bool EQProcessor::retireEQStateDeferred`)
- `retireBandNodeDeferred()` = `bool` 戻り値 (Line 102)
- **全呼び出し元で `(void)` キャスト**: 約20箇所確認
  - `EQProcessor.Core.cpp`: Lines 137, 142, 234, 570, 623, 805
  - `EQProcessor.Parameters.cpp`: Lines 29, 48, 67, 90, 114, 134, 167, 189, 214, 248
  - `EQProcessor.Coefficients.cpp`: Line 75
- ソース内コメント: `// [work37 Phase 1.4] bool 返しに変更。全呼び出し元で (void) キャストして既存動作を維持。`

**影響**: Deferred Deletion Queue が満杯時に `false` が返されても無視される → MKL-allocated EQState/BandNode がリーク。EQパラメータ変更の多いシナリオ（オートメーション・スライダー高速操作）で顕在化。

---

### BUG4: xRunBuffer ACTIVATE イベント無条件プッシュ

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioBlock.cpp:605`, `BlockDouble.cpp:572` |

**検証内容**:
- XRUNイベントプッシュ: `AudioBlock.cpp:579`, `BlockDouble.cpp:546` → ✅ 戻り値チェックあり + `xRunDropCount` インクリメント
- ACTIVATEイベントプッシュ: `AudioBlock.cpp:605`, `BlockDouble.cpp:572` → ❌ `xRunBuffer.push(ev)` 戻り値無視
- キュー容量: `LockFreeRingBuffer<XRunEvent, 64>` (capacity=64)

**影響**: キュー満杯時にACTIVATEイベントが通知なく消失。RuntimeWorldのgeneration変化がTimer側で検出不能に。`xRunDropCount` もインクリメントされないため監視不能。

---

### BUG9: EQProcessor シャドウ relaxed-only データ競合

| 項目 | 内容 |
|------|------|
| **リスク** | MEDIUM |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `EQProcessor.Core.cpp:592-596,655-659` / `EQProcessor.Processing.cpp` |

**検証内容**:

影響変数:
| 変数 | 型 | NonRT書き手 | RT書き手 | RT読み手 |
|------|-----|-------------|-----------|-----------|
| `rtBypassedShadow` | `atomic<bool>` | Core:592,655 | Proc:506,517,1008,1010 | Proc:497,509,1019 |
| `rtActiveStructureShadow` | `atomic<FilterStructure>` | Core:593,656 | Proc:919,926 | Proc:863 |
| `rtAgcCurrentGainShadow` | `atomic<double>` | Core:594,657 | Proc:436,586,1070 | Proc:417 |
| `rtAgcEnvInputShadow` | `atomic<double>` | Core:595,658 | Proc:434,587,1071 | Proc:415 |
| `rtAgcEnvOutputShadow` | `atomic<double>` | Core:596,659 | Proc:435,588,1072 | Proc:416 |

**すべて `memory_order_relaxed` のみ使用。** C++標準上、relaxed順序付けはhappens-beforeを形成せず、データ競合は未定義動作。

**Web検証**: cppreference.com — `memory_order_relaxed` は「異なるスレッド間で値の順序に関する合意を形成しない」。x86ではStoreLoadが強いため実害は稀だが、ARM64では問題となる可能性がある。

**影響**: バイパストグル直後に一過性のオーディオグリッチ（~1ブロック分）。構造切り替え直後にミスマッチ。

---

### BUG10: DeferredDeletionQueue uint32_t ラップアラウンド

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `DeferredDeletionQueue.h:80,120,172` |

**検証内容**:
- `enqueuePos` / `dequeuePos` = `std::atomic<uint32_t>` (32-bit)
- `sequences[]` = `std::atomic<uint32_t>` (32-bit)
- 差分計算: `intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);` ❌

**問題**: 32-bitモジュラ減算を `intptr_t` (64-bit) で行うと、ラップアラウンド時に誤った符号を生む。
- 正解: `int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));`
- 3箇所すべて (enqueue:80, reclaim:120, drainAllUnsafe:172) で同じ問題

**Web検証**: Dmitry Vyukov のオリジナルMPMC実装は `size_t` を使用。ConvoPeq実装は `uint32_t` に変更したが差分計算を `int32_t` に修正していない。

**影響**: 連続運用中のカウンタラップ（約25時間後〜）でキューが誤って満杯/空判定。リタイア要求がフォールバックまたはドロップ。

---

### BUG11: activeRuntimeDSPSlot / fadingRuntimeDSPSlot Non-Atomic Data Race

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.h:1996-1999`, `AtomicAccess.h:31-33` (constexpr plain load/store) |

**検証内容**:
```cpp
// AudioEngine.h:1996-1999
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };  // 非アトミック
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };  // 非アトミック

// AtomicAccess.h:31-33 — constexpr = プレーンロード
constexpr T* get() const noexcept { return reinterpret_cast<T*>(bits); }

// AtomicAccess.h:25-28 — constexpr = プレーンストア
constexpr NonOwningPtr& operator=(T* ptr) noexcept { bits = ...; return *this; }
```

- NonRT Message Thread: setActiveRuntimeDSP() / exchangeFadingRuntimeDSP() → 書き込み
- Timer Thread / UI Thread: getActiveRuntimeDSP() → 読み取り
- **同期ゼロ** — 教科書的なデータ競合 (UB)

**影響緩和**: RT Audio Thread は `RuntimePublicationCoordinator` 経由でDSP状態を参照するため、音声パスでの直接的なUAFはマスクされている。しかしコンパイラ最適化により読み取りがスキップされる可能性がある。

**推奨修正**: `std::atomic<DSPCore*>` への変更。`memory_order_release/acquire` のペア。

---

### BUG12: SafeStateSwapper enterStateReader / exitStateReader がNo-Op

| 項目 | 内容 |
|------|------|
| **リスク** | CRITICAL |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `ConvolverProcessor.h:268-269` |

**検証内容**:
```cpp
// ConvolverProcessor.h:268-269 — 空のスタブ！
void enterStateReader(int /*readerIndex*/) const noexcept {}
void exitStateReader(int /*readerIndex*/) const noexcept {}
```

`SafeStateSwapper::enterReader(index)` / `exitReader(index)` への委譲が行われていない。
- `readerEpochs[]` が常に `kIdleEpoch` (0)
- `getMinReaderEpoch()` が常に `globalEpoch` を返す
- `tryReclaim()` が全エントリを即時解放可能と判定

**影響**:
- `isCacheEntrySafeToDelete()` (LoadPipeline.cpp:213-214) — readerIndex 2
- `createSnapshotFromCurrentState()` (Snapshot.cpp:24-25) — readerIndex 1
これらの呼び出し元で use-after-free の可能性。

---

### BUG13: SafeStateSwapper Epoch Bump Before Pointer Swap

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `SafeStateSwapper.h:106-109` |

**検証内容**:
```cpp
// SafeStateSwapper.h:106-109 — エポックバンプ→スワップ（誤った順序）
const uint64_t epoch1 = convo::fetchAddAtomic(globalEpoch, 1, acq_rel);  // bump #1
/* newEpoch = */ convo::fetchAddAtomic(globalEpoch, 1, acq_rel);          // bump #2
ConvolverState* oldState = convo::exchangeAtomic(activeState, newState, acq_rel); // SWAP
```

**問題のインターリーブ**:
| Time | Writer | Reader | Reclaimer |
|------|--------|--------|-----------|
| t0 | epoch1 = fetchAdd → N+1 | | |
| t1 | | enterReader → epoch N+1 記録 | |
| t2 | fetchAdd → N+2 | | |
| t3 | exchangeAtomic(...) | | |
| t4 | retire oldState (epoch=N) | | |
| t5 | | get() → 旧ポインタを見る | |
| t6 | | | minReaderEpoch = N+1 |
| t7 | | | entryEpoch(N) < N+1 → 解放 |
| t8 | | **使用→UAF** 💥 | |

**補足**: BUG12 (enterStateReader no-op) により現在はマスクされている。BUG12修正後に顕在化する。

**修正**: `swap` → `epoch bump` の順序に変更。

---

### BUG16: Retire Overflow Timestamp 単位不一致（ナノ秒保存・マイクロ秒読取）

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `ISRRetire.cpp:55-57` (Producer) / `ISRRuntimePublicationCoordinator.cpp:279-284` (Consumer) |

**検証内容**:
```cpp
// Producer (ISRRetire.cpp:55-57) — ナノ秒を保存
RetireOverflowEntry entry{
    localIntent,
    static_cast<uint64_t>(std::chrono::steady_clock::now()
        .time_since_epoch().count()),  // ← MSVC: nanoseconds!
    0};

// Consumer (ISRRuntimePublicationCoordinator.cpp:279-284) — マイクロ秒として読取
const uint64_t nowUs = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
if (entry.overflowTimestampUs > 0 && nowUs > entry.overflowTimestampUs)
    // ナノ秒〜10^9 ≫ マイクロ秒〜10^6 → 起動後292年間成立しない
```

- フィールド名: `ISRRetireOverflowRing.h:45` — `uint64_t overflowTimestampUs`（`Us` = マイクロ秒と命名）
- `overflowAgeWarnCallback_` コードパスは事実上**デッドコード**

---

### BUG17: overflowDurationMs が実際はマイクロ秒（1000倍の誤差）

| 項目 | 内容 |
|------|------|
| **リスク** | HIGH |
| **検証結果** | ✅ **Confirmed** |
| **発見箇所** | `AudioEngine.Retire.cpp:134-137` |

**検証内容**:
```cpp
const auto now = static_cast<uint64_t>(
    std::chrono::steady_clock::now().time_since_epoch().count());  // ナノ秒
const uint64_t overflowDurationMs = (now - overflowStart) / 1000;  // ÷1000 = マイクロ秒！
chronicByDuration = (overflowDurationMs > 5000);  // コメント: ">5秒"
```

- `now` はナノ秒（MSVCの `steady_clock::duration` = `nanoseconds`）
- `/1000` では **マイクロ秒** にしかならない
- 変数名 `overflowDurationMs` とコメント「>5秒」は誤り
- 実際の閾値: 5000 μs = **5 ms**（意図は5000 ms = 5秒）
- **慢性オーバーフロー検出が1000倍早く発動** → 不必要なスロットリング

---

## 重要度評価と修正推奨順序

| 優先順位 | BUG-ID | リスク | 理由 |
|---------|--------|-------|------|
| 🔴 1 | BUG11 | CRITICAL | Non-Atomic pointer data race (C++ UB)。コンパイラに依存する未定義動作 |
| 🔴 2 | BUG12 | CRITICAL | SafeStateSwapper RCUが機能せず、use-after-freeが発生し得る |
| 🔴 3 | BUG4 | CRITICAL | ACTIVATEイベント喪失によりRuntimeWorld追跡が不正確に |
| 🔴 4 | BUG13 | HIGH | BUG12修正後にUAFが顕在化。swap順序の設計欠陥 |
| 🔴 5 | BUG10 | HIGH | 32-bitカウンタラップでキューが誤動作。25h+連続運用で顕在化 |
| 🔴 6 | BUG01 | HIGH | Floatパスでソフトニーリミッター欠落 → 音質劣化 |
| 🔴 7 | BUG17 | HIGH | 慢性OVF検出が1000倍早く発動 → 不要スロットリング |
| 🔴 8 | BUG010 | HIGH | 退役失敗時のMKLメモリリーク |
| 🟡 9 | BUG16 | HIGH | 単位不一致によるデッドコード |
| 🟡 10 | BUG09 | MEDIUM | relaxed-onlyデータ競合（ARM64で顕在化リスク） |
| 🟡 11 | BUG02 | MEDIUM | FloatパスでLUFSメーター未更新 |
| 🟡 12 | BUG03 | MEDIUM | FloatパスでTruePeak未検出 |
| 🟡 13 | BUG009 | MEDIUM | AVX→SSE遷移ペナルティ（小バッファ時） |
| 🟢 14 | BUG04 | LOW | AVX→SSE遷移ペナルティ（影響限定的） |
| 🟢 15 | BUG05-08 | LOW | 整数昇格の規約違反（MSVCでは実質安全） |

---

## 補足: 検証に使用したツール

| ツール | 使用目的 | 状態 |
|--------|---------|------|
| **ripgrep (WSL)** | パターン検索・コードクロスリファレンス | ✅ |
| **AiDex MCP** | コードインデックス・シンボル検索 | ✅ |
| **serena MCP** | シンボル探索・型情報取得 | ✅ |
| **semble-search** | コードコンテキスト検索 | ✅ |
| **Web Search** | Intel AVX docs, C++ memory model, Vyukov MPMC | ✅ |
| **WSL find/grep/awk** | ファイル検索・テキスト解析 | ✅ |
