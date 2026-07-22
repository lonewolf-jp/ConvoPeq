# ConvoPeq バグ改修計画書（改訂版 v3）

**作成日**: 2026-07-22
**改訂**: v3（feedback v2 反映版）
**前版からの主な変更**:
- H-13: lock ordering convention の明文化、`syncStateFrom` の `other.irFileLock` アクセス問題を修正
- C-7: `switchEngineOnMessageThread` の単独 callAsync 化を撤回 → **atomic commit** アプローチに変更

---

## 📋 改修対象一覧

| Bug ID | 深刻度 | 判定 | 修正方針 | 評価 |
|--------|--------|------|----------|------|
| **H-13** | High | CONFIRMED | 全 site を irFileLock で保護 + lock ordering 明文化 | A |
| **C-2** | Critical | CONFIRMED | reset() に ptr==p ガード追加 | A+ |
| **C-10** | Critical | CONFIRMED | sanitizeFiniteInRangeV 追加 + state ガード | A- |
| **C-7** | Critical | CONFIRMED | applyNewState を atomic commit に変更 | B+ |
| **C-6** | Critical | PARTIALLY_CONFIRMED | コメント/ドキュメントで Single Writer 契約明示化 | A |

---

## 🔴 H-13: irName データレース — 全 site 保護 + lock ordering

### 修正方針: irFileLock で全 site を保護

```cpp
// getIRName() — ロック追加
[[nodiscard]] juce::String getIRName() const
{
    const juce::ScopedLock sl(irFileLock);
    return irName;
}

// applyNewState() — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    currentIrFile = file;
    irName = file.getFileNameWithoutExtension();
}

// captureBuildSnapshot() — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    snapshot.irFile = currentIrFile;
    snapshot.irName = irName;
}

// applyBuildSnapshot() — irFileLock ブロック内に統合
{
    const juce::ScopedLock sl(irFileLock);
    currentIrFile = snapshot.irFile;
    irName = snapshot.irName;
}
```

### syncStateFrom() の修正: other.irFileLock の取得

```cpp
void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    const BuildSnapshot snapshot = other.captureBuildSnapshot();
    // other.pendingOverrideLock → other.irFileLock を取得→解放済み

    applyBuildSnapshot(snapshot);
    // this->pendingOverrideLock → this->irFileLock を取得→解放済み

    if (const IRState* otherState = other.acquireIRState()) { ... }

    // ★ other.irFileLock + this->irFileLock の2つを取得
    {
        const juce::ScopedLock sl1(other.irFileLock);
        const juce::ScopedLock sl2(irFileLock);
        currentIrFile = other.currentIrFile;
        irName = other.irName;
    }
    // ... 他のフィールドの同期 ...
}
```

### lock ordering convention

```
★ ConvolverProcessor Lock Ordering Convention ★

順序 (Level 0 → Level 3):

  Level 0: cacheMutex              (最外側)
  Level 1: pendingOverrideLock     (パラメータ層)
  Level 2: irFileLock              (IR ファイル層)
  Level 3: visualizationDataLock   (UI 可視化層、最内側)

ルール:
  1. 高レベルロック内で低レベルロックを取得してはならない
  2. 同レベルのロックを2つ同時に保持してはならない（例外: syncStateFrom）
  3. 複数インスタンスの同期時は other → this の順で取得する
  4. ロック取得は always 上記順序に従う
```

### テスト

- 既存テスト回帰
- IR ロード中の UI 表示更新が正常に動作

### 予定工数: 3時間

---

## 🔴 C-2: ScopedAlignedPtr::reset 自己代入保護

```cpp
void reset(T* p = nullptr) noexcept
{
    static_assert(std::is_trivially_destructible_v<T>,
                  "ScopedAlignedPtr only supports trivially destructible types (POD arrays)");
    if (ptr == p) return;     // ← 自己代入ガード追加
    if (ptr) { aligned_free(ptr); }
    ptr = p;
}
```

### 予定工数: 30分

---

## 🔴 C-10: processBandStereo NaN 伝播 — sanitizeFiniteInRangeV

### 新規 SIMD helper（匿名名前空間）

```cpp
// SSE2 ベクトル NaN/Inf 範囲チェック
inline __m128d sanitizeFiniteInRangeV(
    __m128d value,
    __m128d minAbsInclusive,
    __m128d maxAbsExclusive) noexcept
{
    const __m128d diff = _mm_sub_pd(value, value);
    const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
    const __m128d signMask = _mm_set1_pd(-0.0);
    const __m128d absV = _mm_andnot_pd(signMask, value);
    const __m128d geMinMask = _mm_cmpge_pd(absV, minAbsInclusive);
    const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxAbsExclusive);
    const __m128d validMask = _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask));
    return _mm_and_pd(value, validMask);
}
```

### processBandStereo での使用

```cpp
const __m128d vMinZero = _mm_setzero_pd();
const __m128d vMaxRange = _mm_set1_pd(1.0e15);
output = sanitizeFiniteInRangeV(output, vMinZero, vMaxRange);
ic1eq = sanitizeFiniteInRangeV(ic1eq, vMinZero, vMaxRange);
ic2eq = sanitizeFiniteInRangeV(ic2eq, vMinZero, vMaxRange);
```

### 予定工数: 1時間

---

## 🔴 C-7: LoaderThread::runSynchronously — Publish 順序保証

### 根本問題

v2 の `switchEngineOnMessageThread` のみ callAsync 化では **Publish 順序が壊れる**:

```
current:  engine_swap → publish(irFinalized=true) → refreshLatency() [正しい順序]
v2案:    callAsync(engine_swap) → publish(irFinalized=true) → refreshLatency()
           ↑ engine swap が遅延するため、refreshLatency() は旧 engine のレイテンシを返す
```

### v3 改修案: Atomic Commit（callAsync で全コミットを遅延）

```cpp
void ConvolverProcessor::applyNewState(...) noexcept
{
    // ... 重い初期化処理（任意スレッド）...
    // updateIRState, currentIrFile, irName, createWaveformSnapshot, etc.

    // ★ すべての publish を Message Thread で実行（engine swap とアトミックに）
    auto* engine = newConv;
    auto len = targetLength;
    auto sr = loadedSR;
    auto sf = scaleFactor;

    juce::MessageManager::callAsync([this, engine, len, sr, sf]() {
        // === Message Thread で実行 ===

        // Phase 1: Engine swap
        switchEngineOnMessageThread(engine);

        // Phase 2: Publish（engine swap 完了後）
        publishAtomic(irLength, len, std::memory_order_release);
        publishAtomic(currentSampleRate, sr, std::memory_order_release);
        publishAtomic(currentIRScale, sf, std::memory_order_release);
        publishAtomic(irFinalized, true, std::memory_order_release);

        // Phase 3: Latency refresh（new engine のレイテンシを正しく読む）
        refreshLatency();

        // Phase 4: 状態クリア
        publishAtomic(isLoading, false, std::memory_order_release);
        publishAtomic(isRebuilding, false, std::memory_order_release);

        // Phase 5: UI 通知
        updateLatencyCache();
        requestHostDisplayUpdate();
        postCoalescedChangeNotification();
    });
}
```

### Publish 順序保証

```
callAsync 前:                    callAsync 後 (Message Thread):
  irFinalized = false              ① switchEngine (engine swap + epoch advance)
  isLoading = true                 ② publish(irLength, currentSampleRate)
  activeEngine = OLD               ③ publish(irFinalized = true) ← FINAL COMMIT
                                   ④ refreshLatency() ← NEW engine のレイテンシ
                                   ⑤ publish(isLoading = false)
                                   ⑥ publish(isRebuilding = false)
                                   ⑦ UI 通知
```

### Happens-before 関係

| 操作 | Thread | Happens-After | Depends on |
|------|--------|---------------|------------|
| switchEngine | Message | callAsync dispatch | — |
| publish(irLength) | Message | switchEngine 完了 | engine swap |
| publish(irFinalized=true) | Message | publish(irLength) | engine swap |
| refreshLatency() | Message | publish(irFinalized=true) | engine swap + irFinalized |
| publish(isLoading=false) | Message | refreshLatency() | refreshLatency 完了 |

### 遷移ウィンドウの安全性

callAsync 前（LoaderThread 上）:
- `irFinalized = false` → Audio Thread は旧 engine を使用（正しい）
- `isLoading = true` → Timer Thread は rebuild をスキップ（正しい）
- `activeEngine = OLD` → Audio Thread は旧 engine を使用（正しい）

callAsync 後（Message Thread 上）:
- 全操作が単一スレッドで順序実行 → happens-before が保証される

### テスト

- 既存テスト回帰
- rebuild 完了後の状態整合性
- Audio テスト（rebuild 中のオーディオ途切れなし）

### 予定工数: 8〜10時間（ソース解析 + atomic commit 設計 + 回帰テスト + Audio テスト）

---

## 🟠 C-6: SafeStateSwapper Single Writer 契約の明示化

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

### 予定工数: 30分

---

## 📅 改修スケジュール（v3）

### Phase 1: 即座に修正（1-2日）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-2 | `reset()` に自己代入ガード追加 | 30分 |
| C-10 | `sanitizeFiniteInRangeV` 追加 + state ガード | 1時間 |

### Phase 2: 1週間以内

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| H-13 | `irName` 全 site を `irFileLock` で保護 + lock ordering 明文化 | 3時間 |
| C-6 | Single Writer 契約のコメント/ドキュメント明示化 | 30分 |

### Phase 3: 設計検討後（2週間）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-7 | applyNewState を atomic commit に変更 | 8〜10時間 |

---

## 📊 リスク評価（v3）

| Bug | 改修リスク | 回帰リスク | 推奨 |
|-----|-----------|-----------|------|
| C-2 | 低 | 低 | 即座に修正 |
| C-10 | 低 | 低 | 即座に修正 |
| H-13 | 中 | 低 | 1週間以内 |
| C-6 | 低 | 低 | 1週間以内 |
| C-7 | 中高 | 中 | 設計検討後に修正 |
