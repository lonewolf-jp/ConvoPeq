# ConvoPeq バグ改修計画書（改訂版 v4）

**作成日**: 2026-07-22
**改訂**: v4（feedback v3 反映版）
**前版からの主な変更**:
- H-13: lock ordering の代わりに **snapshot-first アプローチ** に変更（ABBA deadlock 完全回避）
- C-7: atomic commit を **CommitTransaction** パターンに拡張（寿命管理・失敗ハンドリング・コミット境界を明確化）

---

## 📋 改修対象一覧

| Bug ID | 深刻度 | 判定 | 修正方針 | 評価 | 状態 |
|--------|--------|------|----------|------|------|
| **C-2** | Critical | CONFIRMED | reset() に ptr==p ガード追加 | A+ | ✅ 実装可能 |
| **C-10** | Critical | CONFIRMED | sanitizeFiniteInRangeV 追加 + state ガード | A | ✅ 実装可能 |
| **C-6** | Critical | PARTIALLY_CONFIRMED | コメント/ドキュメントで Single Writer 契約明示化 | A | ✅ 実装可能 |
| **H-13** | High | CONFIRMED | snapshot-first アプローチ + irFileLock 統一 | A | ✅ 軽微修正後実装可能 |
| **C-7** | Critical | CONFIRMED | CommitTransaction パターンで設計完成 | B+ | ⚠️ 設計レビュー1回必要 |

---

## ✅ C-2: ScopedAlignedPtr::reset 自己代入保護（A+）

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

## ✅ C-10: processBandStereo NaN 伝播 — sanitizeFiniteInRangeV（A）

```cpp
// SSE2 ベクトル NaN/Inf 範囲チェック（匿名名前空間）
inline __m128d sanitizeFiniteInRangeV(
    __m128d value, __m128d minAbsInclusive, __m128d maxAbsExclusive) noexcept
{
    const __m128d diff = _mm_sub_pd(value, value);
    const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
    const __m128d signMask = _mm_set1_pd(-0.0);
    const __m128d absV = _mm_andnot_pd(signMask, value);
    const __m128d geMinMask = _mm_cmpge_pd(absV, minAbsInclusive);
    const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxAbsExclusive);
    return _mm_and_pd(value, _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask)));
}

// processBandStereo での使用
const __m128d vMinZero = _mm_setzero_pd();
const __m128d vMaxRange = _mm_set1_pd(1.0e15);
output  = sanitizeFiniteInRangeV(output,  vMinZero, vMaxRange);
ic1eq   = sanitizeFiniteInRangeV(ic1eq,   vMinZero, vMaxRange);
ic2eq   = sanitizeFiniteInRangeV(ic2eq,   vMinZero, vMaxRange);
```

### 予定工数: 1時間

---

## ✅ C-6: SafeStateSwapper Single Writer 契約の明示化（A）

```cpp
class SafeStateSwapper {
public:
    // Requires: Single Writer — caller must serialize swap() calls.
    // Currently enforced by ConvolverProcessor::updateConvolverState() via:
    //   1. JUCE_ASSERT_MESSAGE_THREAD (debug build assertion)
    //   2. compareExchangeAtomic(writerActive) (release build runtime guard)
    void swap(ConvolverState* newState) noexcept { /* ... */ }
};
```

### 予定工数: 30分

---

## 🔴 H-13: irName データレース — snapshot-first アプローチ（A）

### 問題点

v3 の `other.irFileLock` + `this->irFileLock` 二重取得では、`A.syncStateFrom(B)` と `B.syncStateFrom(A)` の同時実行時に **ABBA deadlock** が発生する可能性がある。

### 解決策: snapshot-first アプローチ

**2つのロックを同時に取得せず、スナップショットを経由する。**

```cpp
void ConvolverProcessor::syncStateFrom(const ConvolverProcessor& other)
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Step 1: other の状態をスナップショット化
    //   captureBuildSnapshot() は other.pendingOverrideLock → other.irFileLock の順で
    //   正しくロック取得し、全フィールドをコピーしてから解放する
    const BuildSnapshot snapshot = other.captureBuildSnapshot();

    // Step 2: this に適用
    //   applyBuildSnapshot() は this->pendingOverrideLock → this->irFileLock の順で
    //   正しくロック取得し、全フィールドを書き込む
    applyBuildSnapshot(snapshot);

    // Step 3: IR データの同期（原子的）
    if (const IRState* otherState = other.acquireIRState()) {
        const IRState* oldState = convo::exchangeAtomic(
            currentIRState, otherState, std::memory_order_acq_rel);
        if (oldState) { /* retire */ }
    }

    // Step 4: その他のフィールド同期（atomic load/store）
    convo::publishAtomic(irLength,
        convo::consumeAtomic(other.irLength, std::memory_order_acquire),
        std::memory_order_release);
    convo::publishAtomic(currentSampleRate,
        convo::consumeAtomic(other.currentSampleRate, std::memory_order_acquire),
        std::memory_order_release);
}
```

### 根拠

- `captureBuildSnapshot()` は既に `other.irFileLock` で `other.currentIrFile` と `other.irName` を正しく保護している
- `applyBuildSnapshot()` は既に `this->irFileLock` で `this->currentIrFile` と `this->irName` を正しく保護している
- 2つのロックは同時に保持されない → ABBA deadlock が不可能
- ISR の **Authority を増やさない** 設計思想に合致

### getIRName() の修正（既存サイトの保護）

```cpp
[[nodiscard]] juce::String getIRName() const
{
    const juce::ScopedLock sl(irFileLock);
    return irName;
}
```

### テスト

- 既存テスト回帰
- IR ロード中の UI 表示更新が正常に動作

### 予定工数: 2時間

---

## 🟠 C-7: LoaderThread::runSynchronously — CommitTransaction パターン（B+）

### 問題点（v3 からの改善）

v3 の単純な callAsync 化では以下の問題が未解決:

1. **Snapshot 整合性**: callAsync 前に `createWaveformSnapshot` / `createFrequencyResponseSnapshot` を実行するが、engine はまだ旧
2. **callAsync 失敗**: processor 破棄時に callAsync が失敗した場合のリソース管理
3. **Commit Transaction 概念の欠如**: 「Build → Commit → Publish」の明確な境界がない
4. **オブジェクト寿命**: WeakReference による保護が必要

### v4 改修案: CommitTransaction パターン

#### CommitTransaction 構造体

```cpp
struct CommitTransaction
{
    // ── 構築時にキャプチャ（不変） ──
    StereoConvolver* newEngine = nullptr;      // swap 後は nullptr に
    std::unique_ptr<juce::AudioBuffer<double>> loadedIR;
    std::unique_ptr<juce::AudioBuffer<double>> displayIR;
    int targetLength = 0;
    double sampleRate = 0.0;
    double scaleFactor = 1.0;
    bool isRebuild = false;
    juce::File irFile;

    // ── Phase 1: Pre-swap（任意スレッド、engine 依存なし） ──
    void prepareVisualizations(ConvolverProcessor& cp)
    {
        if (displayIR && displayIR->getNumSamples() > 0) {
            cp.createWaveformSnapshot(*displayIR);
            cp.createFrequencyResponseSnapshot(*displayIR, sampleRate);
        }
    }

    // ── Phase 2: Commit（Message Thread のみ） ──
    void commit(ConvolverProcessor& cp)
    {
        // 2a. IR メタデータ更新
        if (!isRebuild)
            cp.updateIRState(*loadedIR, sampleRate);
        cp.publishAtomic(cp.currentIRScale, scaleFactor, std::memory_order_release);

        // 2b. Engine swap（排他的）
        cp.switchEngineOnMessageThread(newEngine);
        newEngine = nullptr;  // 所有権譲渡

        // 2c. Publish（engine swap 完了後）
        cp.publishAtomic(cp.irLength, targetLength, std::memory_order_release);
        cp.publishAtomic(cp.currentSampleRate, sampleRate, std::memory_order_release);
        cp.publishAtomic(cp.irFinalized, true, std::memory_order_release);

        // 2d. Latency refresh（new engine のレイテンシを正しく読む）
        cp.refreshLatency();

        // 2e. 状態クリア
        cp.publishAtomic(cp.isLoading, false, std::memory_order_release);
        cp.publishAtomic(cp.isRebuilding, false, std::memory_order_release);
    }
};
```

#### applyNewState での使用

```cpp
void ConvolverProcessor::applyNewState(...) noexcept
{
    // Phase 1: Pre-swap（任意スレッド）
    CommitTransaction txn;
    txn.newEngine = newConv;
    txn.targetLength = targetLength;
    txn.sampleRate = loadedSR;
    txn.scaleFactor = scaleFactor;
    txn.isRebuild = isRebuild;
    txn.irFile = file;
    txn.loadedIR = std::move(loadedIR);
    txn.displayIR = std::move(displayIR);
    txn.prepareVisualizations(*this);

    // Phase 2: Commit（Message Thread で実行）
    auto* txnPtr = new CommitTransaction(std::move(txn));
    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);

    const bool queued = juce::MessageManager::callAsync(
        [weakThis, txnPtr]() {
            if (auto* self = weakThis.get()) {
                txnPtr->commit(*self);
                delete txnPtr;
            } else {
                // processor 破棄済み — newEngine のみ解放
                if (txnPtr->newEngine)
                    StereoConvolver::retireStereoConvolver(txnPtr->newEngine, nullptr);
                delete txnPtr;
            }
        });

    if (!queued) {
        // MessageManager シャットダウン中 — 同期的にクリーンアップ
        if (txn.newEngine)
            StereoConvolver::retireStereoConvolver(txn.newEngine, nullptr);
        delete txnPtr;
    }
}
```

### Happens-before 関係（明確化）

```
Pre-swap (任意スレッド):     Commit (Message Thread):
  createWaveformSnapshot()     ① updateIRState()
  createFrequencySnapshot()    ② publish(currentIRScale)
  txn 構築                     ③ switchEngine() ← engine swap
                                ④ publish(irLength, currentSampleRate)
                                ⑤ publish(irFinalized = true) ← FINAL COMMIT
                                ⑥ refreshLatency() ← NEW engine のレイテンシ
                                ⑦ publish(isLoading=false, isRebuilding=false)
                                ⑧ UI 通知
```

### 寿命管理

| シナリオ | 処理 |
|---------|------|
| 正常完了 | `weakThis.get()` → `commit()` → `delete txnPtr` |
| processor 破棄済み | `weakThis.get() == nullptr` → `retireStereoConvolver(newEngine)` → `delete txnPtr` |
| MessageManager シャットダウン | `callAsync` returns false → 同期的に `retireStereoConvolver` + `delete` |
| `newEngine` のみ解放 | `retireStereoConvolver(newEngine, nullptr)` — RCU 経由で遅延解放 |

### テスト

- 既存テスト回帰
- rebuild 完了後の状態整合性
- processor 破棄中の callAsync ハンドリング
- Audio テスト（rebuild 中のオーディオ途切れなし）

### 予定工数: 10〜12時間

---

## 📅 改修スケジュール（v4）

### Phase 1: 即座に修正（1-2日）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-2 | `reset()` に自己代入ガード追加 | 30分 |
| C-10 | `sanitizeFiniteInRangeV` 追加 + state ガード | 1時間 |

### Phase 2: 軽微な修正後（1週間）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| H-13 | snapshot-first アプローチ + getIRName ロック追加 | 2時間 |
| C-6 | Single Writer 契約のコメント/ドキュメント明示化 | 30分 |

### Phase 3: 設計レビュー後（2週間）

| Bug | 修正内容 | 工数 |
|-----|----------|------|
| C-7 | CommitTransaction パターンの実装 | 10〜12時間 |

---

## 📊 リスク評価（v4）

| Bug | 改修リスク | 回帰リスク | 推奨 |
|-----|-----------|-----------|------|
| C-2 | 低 | 低 | 即座に修正 |
| C-10 | 低 | 低 | 即座に修正 |
| H-13 | 低（snapshot-first） | 低 | 1週間以内 |
| C-6 | 低 | 低 | 1週間以内 |
| C-7 | 中 | 中 | 設計レビュー後に修正 |

---

## 🔧 残りの要調査事項

| 項目 | 内容 | 優先度 |
|------|------|--------|
| C-7 設計レビュー | CommitTransaction の実装詳細を詰める（IR データ所有権、表示スナップショットの engine 依存性） | 高 |
| H-13 テスト | snapshot-first アプローチの動作確認 | 中 |
| 全体回帰テスト | Debug Build + CTest 実行 | 高 |
