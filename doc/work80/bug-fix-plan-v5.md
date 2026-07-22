# ConvoPeq バグ改修計画書（改訂版 v5）

**作成日**: 2026-07-22
**改訂**: v5（feedback v4 反映版）
**前版からの主な変更**:
- C-7: CommitTransaction → **PendingCommit** に設計変更（データ保持とコミット実行を分離）

---

## 📋 実装状況

| Bug ID | 評価 | 状態 |
|--------|------|------|
| **C-2** | A+ | ✅ 実装完了 |
| **C-10** | A | ✅ 実装完了 |
| **C-6** | A | ✅ 実装完了 |
| **H-13** | A | ✅ 実装完了 |
| **C-7** | B | ⚠️ 設計レビュー中 |

---

## ✅ C-2 / C-10 / C-6 / H-13: 実装完了

4つのバグ修正は既にソースコードに適用済み。

---

## 🟠 C-7: PendingCommit デザイン（ISR Runtime 整合版）

### 問題点（v4 からの改善）

v4 の `CommitTransaction` では:
1. 処理（Build + Visualization + Commit + Publish + Cleanup）が一つのオブジェクトに抱え込まれていた
2. WeakReference を導入したが、ISR の Owner → Publish → Retire ライフサイクルと不整合
3. callAsync 失敗時の Rollback が不完全（newEngine のみ retire、他は destroy）

### 解決策: PendingCommit（データ保持）+ ConvolverProcessor（コミット実行）

**ISR Runtime の設計思想に合致する责務分離:**

```
LoaderThread:     PendingCommit 作成（データ保持のみ）
                       ↓
callAsync:        ConvolverProcessor::executePendingCommit() 呼び出し
                       ↓
Processor:        Engine swap + Publish + Latency + UI 通知（全処理を Processor が実行）
                       ↓
Processor:        PendingCommit 破棄（Processor が所有権を持つ）
```

### PendingCommit 構造体（データのみ）

```cpp
// データ保持のみ。処理は ConvolverProcessor 側が実行する。
struct PendingCommit
{
    StereoConvolver* newEngine = nullptr;
    std::unique_ptr<juce::AudioBuffer<double>> loadedIR;
    std::unique_ptr<juce::AudioBuffer<double>> displayIR;
    int targetLength = 0;
    double sampleRate = 0.0;
    double scaleFactor = 1.0;
    bool isRebuild = false;
    juce::File irFile;

    // Rollback: newEngine のみ安全に解放
    void rollback()
    {
        if (newEngine) {
            StereoConvolver::retireStereoConvolver(newEngine, nullptr);
            newEngine = nullptr;
        }
        // loadedIR / displayIR は RAII で自動解放
    }

    ~PendingCommit() { rollback(); }
};
```

### ConvolverProcessor 側のコミット実行

```cpp
// Processor 側が全処理を実行（Authority は Processor に一箇所）
void ConvolverProcessor::executePendingCommit(std::unique_ptr<PendingCommit> commit)
{
    if (!commit || !commit->newEngine) return;

    // Phase 1: IR メタデータ更新
    if (!commit->isRebuild)
        updateIRState(*commit->loadedIR, commit->sampleRate);

    // Phase 2: 可視化スナップショット（engine 依存なし）
    if (commit->displayIR && commit->displayIR->getNumSamples() > 0) {
        createWaveformSnapshot(*commit->displayIR);
        createFrequencyResponseSnapshot(*commit->displayIR, commit->sampleRate);
    }

    // Phase 3: Engine swap（排他的）
    switchEngineOnMessageThread(commit->newEngine);
    commit->newEngine = nullptr;  // 所有権譲渡

    // Phase 4: Publish
    publishAtomic(irLength, commit->targetLength, std::memory_order_release);
    publishAtomic(currentSampleRate, commit->sampleRate, std::memory_order_release);
    publishAtomic(currentIRScale, commit->scaleFactor, std::memory_order_release);
    publishAtomic(irFinalized, true, std::memory_order_release);

    // Phase 5: Latency refresh
    refreshLatency();

    // Phase 6: 状態クリア
    publishAtomic(isLoading, false, std::memory_order_release);
    publishAtomic(isRebuilding, false, std::memory_order_release);

    // Phase 7: UI 通知
    updateLatencyCache();
    requestHostDisplayUpdate();
    postCoalescedChangeNotification();

    // PendingCommit は RAII で自動解放（loadedIR / displayIR）
}
```

### applyNewState での使用

```cpp
void ConvolverProcessor::applyNewState(...) noexcept
{
    // Phase 1: PendingCommit 作成（任意スレッド）
    auto commit = std::make_unique<PendingCommit>();
    commit->newEngine = newConv;
    commit->targetLength = targetLength;
    commit->sampleRate = loadedSR;
    commit->scaleFactor = scaleFactor;
    commit->isRebuild = isRebuild;
    commit->irFile = file;
    commit->loadedIR = std::move(loadedIR);
    commit->displayIR = std::move(displayIR);

    // Phase 2: Message Thread でコミット実行
    auto* commitPtr = commit.release();  // 所有権を callAsync に渡す

    const bool queued = juce::MessageManager::callAsync(
        [this, commitPtr]() {
            // this は Message Thread で有効（timerCallback が停止済み）
            auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
            executePendingCommit(std::move(ownedCommit));
        });

    if (!queued) {
        // MessageManager シャットダウン中 — 同期的にロールバック
        auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
        ownedCommit->rollback();
    }
}
```

### WeakReference を使わない理由

- ISR の Owner → Publish → Retire ライフサイクルに統一するため
- `callAsync` の `this` は Message Thread 上で有効（`timerCallback` 停止済み）
- `~ConvolverProcessor()` は Message Thread のみで実行（JUCE 要件）
- WeakReference は GUI コンポーネント用。Engine 側では不要

### callAsync 失敗時の Rollback

| シナリオ | 処理 |
|---------|------|
| 正常完了 | `executePendingCommit()` → `PendingCommit` 自動解放 |
| MessageManager シャットダウン | `ownedCommit->rollback()` → `retireStereoConvolver(newEngine)` + RAII で解放 |
| Processor 破棄 | 不可能（`callAsync` は Message Thread のみ、`~ConvolverProcessor` も Message Thread のみ） |

### Happens-before 関係

```
PendingCommit 作成 (任意スレッド):
  loadedIR, displayIR, newEngine をキャプチャ

callAsync → Message Thread:
  ① updateIRState()
  ② createWaveformSnapshot()
  ③ switchEngineOnMessageThread() ← engine swap
  ④ publish(irLength, currentSampleRate, currentIRScale)
  ⑤ publish(irFinalized = true) ← FINAL COMMIT
  ⑥ refreshLatency() ← NEW engine のレイテンシ
  ⑦ publish(isLoading=false, isRebuilding=false)
  ⑧ UI 通知
```

### 予定工数: 6〜8時間

---

## 📊 最終評価

| Bug | 評価 | 状態 |
|-----|------|------|
| C-2 | A+ | ✅ 実装完了 |
| C-10 | A | ✅ 実装完了 |
| C-6 | A | ✅ 実装完了 |
| H-13 | A | ✅ 実装完了 |
| C-7 | B+ | ⚠️ PendingCommit 設計完成、実装待ち |

**総合評価: A-（90〜92点）**
