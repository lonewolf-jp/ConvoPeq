# ConvoPeq バグ改修計画書（改訂版 v2）

**作成日**: 2026-07-22
**改訂**: v2（フィードバック反映版）
**前版からの主な変更**:
- C-2: exchange案撤回を明記
- C-10: 既存 SIMD helper の汎化方針に変更
- C-7: applyNewState 全行解析に基づく責務分離設計に変更
- C-6: writerActive 依存 assert 撤回、コメント/ドキュメント方針に変更
- M-12: 計画から除外（Feature Change として別タスク化）
- 工数の再評価

---

## 📋 改修対象一覧

| Bug ID | 深刻度 | 判定 | 修正方針 | 評価 |
|--------|--------|------|----------|------|
| **H-13** | High | CONFIRMED | 全 read/write site を irFileLock で統一 | A |
| **C-2** | Critical | CONFIRMED | reset() に ptr==p ガード追加（exchange案は誤り） | A+ |
| **C-10** | Critical | CONFIRMED | 既存 helper の SIMD 汎化 + state ガード追加 | B+ |
| **C-7** | Critical | CONFIRMED | applyNewState 全行解析 → 責務分離 | B |
| **C-6** | Critical | PARTIALLY_CONFIRMED | コメント/ドキュメントで Single Writer 契約を明示 | B |
| ~~M-12~~ | ~~Medium~~ | ~~DESIGN_CHOICE~~ | ~~計画から除外（Feature Change）~~ | ~~除外~~ |

---

## 🔴 H-13: irName データレース — 全アクセスサイト保護

### 概要

`irName`（`juce::String`、non-atomic）が `irFileLock` で保護されていない。`currentIrFile` は正しく保護されているが、`irName` は複数スレッドから裸でアクセスされる。

### 全 read/write サイト（完全版）

#### irName アクセスサイト

| # | ファイル:行 | 種別 | ロック | スレッド | 修正要否 |
|---|------------|------|-------|---------|---------|
| 1 | `ConvolverProcessor.h:1022` | 宣言 | — | — | — |
| 2 | `ConvolverProcessor.h:386` | **READ** | ❌ ロックなし | 任意 | ✅ 修正 |
| 3 | `LoadPipeline.cpp:455` | **WRITE** | ✅ irFileLock | Message Thread | — |
| 4 | `LoadPipeline.cpp:667` | **WRITE** | ❌ ロックなし | 任意（applyNewState内） | ✅ 修正 |
| 5 | `StateAndUI.cpp:260` | **READ** | ❌ ロックなし | 任意（captureBuildSnapshot内） | ✅ 修正 |
| 6 | `StateAndUI.cpp:282` | **WRITE** | ❌ ロックなし | 任意（applyBuildSnapshot内） | ✅ 修正 |
| 7 | `StateAndUI.cpp:406` | **WRITE** | ❌ ロックなし | Message Thread（syncStateFrom内） | ✅ 修正 |
| 8 | `AudioEngine.UIEvents.cpp:60` | READ | ❌ ロックなし | Message Thread | ✅ 修正 |
| 9 | `AudioEngine.UIEvents.cpp:129` | READ | ❌ ロックなし | Message Thread | ✅ 修正 |
| 10 | `ConvolverControlPanel.cpp:1334` | READ | ❌ ロックなし | Message Thread | ✅ 修正 |

#### currentIrFile アクセスサイト（参考 — 既に保護済み）

| # | ファイル:行 | ロック | 備考 |
|---|------------|-------|------|
| 1 | `LoadPipeline.cpp:239` | ✅ irFileLock | — |
| 2 | `LoadPipeline.cpp:457` | ✅ irFileLock | — |
| 3 | `LoadPipeline.cpp:665` | ✅ irFileLock | — |
| 4 | `StateAndUI.cpp:245` | ✅ irFileLock | — |
| 5 | `StateAndUI.cpp:265` | ✅ irFileLock | — |
| 6 | `StateAndUI.cpp:280` | ✅ irFileLock | — |
| 7 | `StateAndUI.cpp:374` | ✅ irFileLock | — |
| 8 | `StateAndUI.cpp:404` | ✅ irFileLock | ⚠ other.currentIrFile はロック外で読み取り |
| 9 | `Lifecycle.cpp:180` | ✅ irFileLock | — |

### 修正方針

**全 read/write site を `irFileLock` で統一する。** 既存の `currentIrFile` と同様のパターン。

```cpp
// getIRName() — ロック追加
[[nodiscard]] juce::String getIRName() const
{
    const juce::ScopedLock sl(irFileLock);
    return irName;
}

// applyNewState() L665-667 — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    currentIrFile = file;
    irName = file.getFileNameWithoutExtension();  // ← ここに移動
}

// captureBuildSnapshot() L260 — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    snapshot.irFile = currentIrFile;
    snapshot.irName = irName;  // ← ここに移動
}

// applyBuildSnapshot() L280-282 — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    currentIrFile = snapshot.irFile;
    irName = snapshot.irName;  // ← ここに移動
}

// syncStateFrom() L404-406 — 既存 irFileLock ブロックに追加
{
    const juce::ScopedLock sl(irFileLock);
    currentIrFile = other.currentIrFile;
    irName = other.irName;  // ← ここに追加
}
```

### テスト

- 既存テスト回帰
- IR ロード中の UI 表示更新が正常に動作することを確認

### 予定工数: 2〜3時間

---

## 🔴 C-2: ScopedAlignedPtr::reset 自己代入保護

### 概要

`reset()` に `ptr == p` ガードがなく、自己代入時に Use-After-Free が発生する。

### 対象ファイル

- `src/AlignedAllocation.h` — `ScopedAlignedPtr::reset()` (line ~88)

### 改修案（exchange案は誤り、明示的ガードに統一）

```cpp
void reset(T* p = nullptr) noexcept
{
    static_assert(std::is_trivially_destructible_v<T>,
                  "ScopedAlignedPtr only supports trivially destructible types (POD arrays)");
    if (ptr == p) return;     // ← 自己代入ガード追加
    if (ptr)
    {
        aligned_free(ptr);
    }
    ptr = p;
}
```

### 根拠

- `exchange` 方案は `std::exchange` を使用するが、`ScopedAlignedPtr` は move-only で `std::exchange` の意味的に不透明
- 明示的ガードの方が意図が明確で、コードレビューで認識されやすい
- デストラクタ `~ScopedAlignedPtr()` は `reset(nullptr)` を呼ぶ — `nullptr != ptr` なので安全
- ムーブ代入は `if (this != &o)` ガード済み — 変更不要

### テスト

- 自己代入テスト: `ptr.reset(ptr.get())` が安全に動作
- 既存 `AlignedAllocationTests` 回帰

### 予定工数: 30分

---

## 🔴 C-10: processBandStereo NaN 伝播 — 既存 helper 汎化

### 概要

`processBandStereo()` に状態変数 `ic1eq`/`ic2eq` の NaN/Inf ガードがなく、NaN がループ内で伝播する。

### 既存 SIMD helper 調査結果

| ヘルパー | 場所 | タイプ | NaN/Inf対応 |
|---------|------|--------|------------|
| `isFinite(double)` | DspNumericPolicy.h | scalar | ✅ |
| `isFiniteAndAbsInRangeMask(double, double, double)` | EQProcessor.Processing.cpp | scalar (SSE2内蔵) | ✅ |
| `replaceNonFiniteWithZero(double)` | DspNumericPolicy.h | scalar | ✅ |
| `killDenormalV(__m128d)` | DspNumericPolicy.h | SIMD (Release: NO-OP) | ❌ |
| `isFiniteAndAbsInRangeMaskV` | — | **未実装** | — |

**結論**: SIMD (`__m128d`) 版の `isFiniteAndAbsInRangeMask` は未実装。新規作成が必要。

### 改修案: 既存パターンに適合する SIMD helper 追加

**匿名名前空間に `sanitizeFiniteInRangeV` を追加**（`isFiniteAndAbsInRangeMask` と同等の SIMD 版）:

```cpp
// SSE2 ベクトル NaN/Inf 範囲チェック
// processBand の isFiniteAndAbsInRangeMask と同等の機能を __m128d で2レーン同時実行
inline __m128d sanitizeFiniteInRangeV(
    __m128d value,
    __m128d minAbsInclusive,
    __m128d maxAbsExclusive) noexcept
{
    // isFinite: self-subtract trick
    const __m128d diff = _mm_sub_pd(value, value);
    const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

    // abs(value) via sign-bit clear
    const __m128d signMask = _mm_set1_pd(-0.0);
    const __m128d absV = _mm_andnot_pd(signMask, value);

    // abs >= min && abs < max
    const __m128d geMinMask = _mm_cmpge_pd(absV, minAbsInclusive);
    const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxAbsExclusive);

    const __m128d validMask = _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask));
    return _mm_and_pd(value, validMask);
}
```

**processBandStereo での使用**:

```cpp
const __m128d vMinZero = _mm_setzero_pd();
const __m128d vMaxRange = _mm_set1_pd(1.0e15);

// output ガード（既存 self-subtract trick を置換）
output = sanitizeFiniteInRangeV(output, vMinZero, vMaxRange);

// ★ state 変数ガード（新規追加 — processBand と同等）
ic1eq = sanitizeFiniteInRangeV(ic1eq, vMinZero, vMaxRange);
ic2eq = sanitizeFiniteInRangeV(ic2eq, vMinZero, vMaxRange);
```

### 設計判断

| 判断 | 理由 |
|------|------|
| 匿名名前空間に配置 | `isFiniteAndAbsInRangeMask` / `isFiniteNoLibm` と同等のスコープ |
| `__m128d` のみ | `processBandStereo` は L+R を2レーンにパック |
| `#if defined(__AVX2__)` ガード不要 | `_mm_*` は SSE2 命令のみ |
| ループ内で毎サンプル実行 | state 変数の NaN/Inf は次サンプルに伝播するため |

### テスト

- 既存 `EQProcessorMaxGainTests` 回帰
- NaN 入力時の動作確認

### 予定工数: 1時間

---

## 🔴 C-7: LoaderThread::runSynchronously スレッド規約違反 — 全行解析に基づく責務分離

### 概要

`runSynchronously()` がワーカースレッドから直接 `applyNewState()` を呼び、JUCE の Message Thread 前提の操作を実行する。

### applyNewState() 全行解析結果

| 処理 | カテゴリ | 根拠 |
|------|---------|------|
| `updateIRState()` | RT_FORBIDDEN | `mkl_malloc`, RCU enqueue, heap alloc |
| `irFileLock` + `currentIrFile` | RT_FORBIDDEN | CriticalSection mutex |
| `irName` 書き込み | RT_FORBIDDEN | String allocation |
| `publishAtomic(currentIRScale)` | ANY_THREAD | Atomic store |
| `createWaveformSnapshot()` | RT_FORBIDDEN | ScopedLock, vector allocation |
| `createFrequencyResponseSnapshot()` | RT_FORBIDDEN | MKL DFTI, FFT, malloc |
| **`switchEngineOnMessageThread()`** | **MESSAGE_THREAD** | `advanceRetireEpoch()` — RCU epoch publication |
| `publishAtomic(irLength)` | ANY_THREAD | Atomic store |
| `publishAtomic(currentSampleRate)` | ANY_THREAD | Atomic store |
| `publishAtomic(irFinalized)` | ANY_THREAD | Atomic store |
| **`refreshLatency()`** | ANY_THREAD（callAsync 内蔵） | RCU guard + atomic + `requestHostDisplayUpdate` |
| `publishAtomic(isLoading=false)` | ANY_THREAD | Atomic store |
| `publishAtomic(isRebuilding=false)` | ANY_THREAD | Atomic store |
| `rebuildAllIRs()` (via callAsync) | MESSAGE_THREAD | `callAsync` で遅延dispatch |
| `updateLatencyCache()` | RT_FORBIDDEN | Heap alloc (`new LatencySnapshot`) |
| **`requestHostDisplayUpdate()`** | MESSAGE_THREAD | `callAsync` dispatch |
| `postCoalescedChangeNotification()` | ANY_THREAD | 内部で `callAsync` 使用 |

### 根本問題

`switchEngineOnMessageThread()` が **唯一の MESSAGE_THREAD 必須処理**。他の処理は多くが ANY_THREAD か RT_FORBIDDEN。

**`switchEngineOnMessageThread()` の内部:**
1. `exchangeActiveEngine(newEngine)` — ANY_THREAD（atomic exchange）
2. `advanceRetireEpoch()` — **MESSAGE_THREAD**（RCU epoch publication point）
3. `retireStereoConvolver(oldEngine)` — RT_FORBIDDEN（heap dealloc）

### 改修案: callAsync による switchEngine の遅延dispatch

**applyNewState() 内の `switchEngineOnMessageThread()` を `callAsync` 経由に変更**:

```cpp
void ConvolverProcessor::applyNewState(...) noexcept
{
    // ... 重い処理（any thread）...

    // switchEngine を Message Thread で実行
    auto* engine = newConv;
    juce::MessageManager::callAsync([this, engine]() {
        switchEngineOnMessageThread(engine);
    });

    // ... 残りの処理（any thread）...
}
```

### 注意点

- `callAsync` は非同期なので、engine swap が遅延する可能性がある
- ただし、`updateConvolverState()` は既に `JUCE_ASSERT_MESSAGE_THREAD` + CAS で保護されている
- `refreshLatency()` は内部で `callAsync` を使用しており、呼び出しスレッドに依存しない
- `postCoalescedChangeNotification()` も内部で `callAsync` を使用しており安全

### テスト

- 既存テスト回帰
- rebuild 中の UI 操作が正常に動作
- rebuild 完了後の状態が正しい

### 予定工数: 6〜8時間（ソース解析 + 呼び出し経路確認 + RT制約確認 + 回帰テスト + Audio テスト）

---

## 🟠 C-6: SafeStateSwapper Single Writer 契約の明示化

### 概要

`SafeStateSwapper::swap()` に内部 CAS 保護がないが、呼び出し元（`ConvolverProcessor::updateConvolverState()`）が `writerActive` CAS + `JUCE_ASSERT_MESSAGE_THREAD` で保護している。`writerActive` は SafeStateSwapper の外部にあるため、内部 assert は実装できない。

### 調査結果

- `writerActive` は `ConvolverProcessor.h:1209` に定義（SafeStateSwapper の外部）
- `swap()` の唯一の呼び出し元は `updateConvolverState()` のみ
- `updateConvolverState()` は `JUCE_ASSERT_MESSAGE_THREAD` + `compareExchangeAtomic(writerActive)` で2重保護
- `tryReclaim()` には `jassert(reclaimThreadIdDebug == currentThreadId)` で Single Consumer 契約を確認する前例がある

### 改修案: コメント/ドキュメントでの契約明示化

```cpp
class SafeStateSwapper {
public:
    // Requires: Single Writer — caller must serialize swap() calls.
    // Currently enforced by ConvolverProcessor::updateConvolverState() via:
    //   1. JUCE_ASSERT_MESSAGE_THREAD (debug build assertion)
    //   2. compareExchangeAtomic(writerActive) (release build runtime guard)
    void swap(ConvolverState* newState) noexcept
    {
        // ... 既存コード ...
    }
};
```

### 根拠

- `writerActive` に依存した assert は SafeStateSwapper からは実装できない
- `tryReclaim()` の `reclaimThreadIdDebug` パターンは内部スレッド ID を持つ場合のみ有効
- コメント + 呼び出し元の assert が現在のコードベースの標準パターン

### 予定工数: 30分

---

## ❌ M-12: 計画除外（Feature Change）

「補間廃止 → プリセット切替」はバグ修正ではなく仕様変更。NoiseShaper の特性ジャンプが発生する可能性がある。別タスク（Week4 NoiseShaper redesign）として扱う。

---

## 📅 改修スケジュール（改訂版）

### Phase 1: 即座に修正（1-2日）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-2 | `reset()` に自己代入ガード追加 | 30分 |
| C-10 | `sanitizeFiniteInRangeV` 追加 + state ガード | 1時間 |
| H-13 | `irName` 全 site を `irFileLock` で保護 | 2〜3時間 |

### Phase 2: 設計検討後修正（1週間）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-7 | `switchEngineOnMessageThread` を callAsync 経由に変更 | 6〜8時間 |
| C-6 | Single Writer 契約のコメント/ドキュメント明示化 | 30分 |

---

## 🔧 テスト計画

### 修正後必須テスト

1. **Debug Build + CTest**: `cmake --build build --config Debug && ctest -C Debug --output-on-failure`
2. **既存テスト回帰**: `DeferredDeletionQueueReclaimTests`, `EQProcessorMaxGainTests`, `GainStagingContractTests`
3. **CI/CD**: プッシュ後に GitHub Actions で全テスト実行

### 追加テスト（推奨）

| テスト | 対象 | 内容 |
|--------|------|------|
| 自己代入テスト | C-2 | `ScopedAlignedPtr::reset(ptr.get())` が安全に動作 |
| NaN 入力テスト | C-10 | NaN を含むオーディオデータの処理 |
| マルチスレッドテスト | H-13 | IR ロード中の UI アクセス |
| rebuild 完了テスト | C-7 | `runSynchronously` 後の状態整合性 |

---

## 📊 リスク評価（改訂版）

| Bug | 改修リスク | 回帰リスク | 推奨 |
|-----|-----------|-----------|------|
| C-2 | 低（1行追加） | 低 | 即座に修正 |
| C-10 | 低（SIMD helper 追加） | 低 | 即座に修正 |
| H-13 | 中（10箇所の修正） | 低 | 即座に修正 |
| C-7 | 中高（callAsync 変更） | 中 | 設計検討後に修正 |
| C-6 | 低（コメントのみ） | 低 | 防御的修正 |
