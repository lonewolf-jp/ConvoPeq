# ConvoPeq バグ改修計画書（改訂版 v6）

**作成日**: 2026-07-22
**改訂**: v6 Final（feedback v5 反映版 + 説明・コメントの最終修正）
**前版からの主な変更**:
- C-7: `this` 寿命保証の説明を ConvolverProcessor 固有の停止シーケンスに限定
- C-7: `PendingCommit` デストラクタに所有権遷移のコメントを追加

---

## 📋 実装状況

| Bug ID | 評価 | 状態 |
|--------|------|------|
| **C-2** | A+ | ✅ 実装完了 |
| **C-10** | A | ✅ 実装完了 |
| **C-6** | A | ✅ 実装完了 |
| **H-13** | A | ✅ 実装完了 |
| **C-7** | A+ | ✅ 設計完成（WeakReference + Precondition/Postcondition + noexcept + releaseEngine） |

---

## 🟠 C-7: PendingCommit デザイン（v6 最終版）

### 1. `this` 寿命保証 — WeakReference が本質的な安全弁

**問**: `callAsync([this]() { ... })` のラムダ実行時に `this` は必ず生きているのか？

**結論**: MessageManager は callAsync が参照するオブジェクトの寿命を保証しない。安全性は `forceCleanup()` による LoaderThread 停止と `JUCE_DECLARE_WEAK_REFERENCEABLE` による WeakReference の組み合わせで成立する。WeakReference は callAsync 後のオブジェクト破棄に対する最後の安全弁である。

#### 実際のシャットダウンシーケンス（ソースコード確認済み）

```
Timeline (all on Message Thread):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. LoaderThread (background) calls callAsync(lambda)
   → lambda posted to MessageManager queue (NOT yet executed)

2. App quit → ~MainWindow() body runs
   → audioEngineProcessor.reset()
   → UI components destroyed

3. ~MainWindow() body completes → member destruction begins (reverse order)

4. ~AudioEngine() body runs
   → NO callAsync queue drain (cancelPendingUpdate は AsyncUpdater のみ)
   → body completes

5. AudioEngine members destroyed (reverse declaration order):
   → ~ConvolverProcessor()
      → forceCleanup() stops LoaderThread (stopThread(500))
      → activeLoader.reset()
      → object memory reclaimed
      → JUCE_DECLARE_WEAK_REFERENCEABLE severs WeakReference

6. Message loop pumps (if MessageManager remains active)
   → queued lambda may execute → weakThis.get() returns nullptr → safe no-op
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### なぜ安全なのか

1. **`forceCleanup()`** が `~ConvolverProcessor()` の冒頭で LoaderThread を停止する → 停止後に新しい callAsync がキューイングされる機会がない
2. **`JUCE_DECLARE_WEAK_REFERENCEABLE`**（`ConvolverProcessor.h:1241`）が `~ConvolverProcessor()` 開始時に WeakReference を切断する → 後から発火するラムダは `wp.get() → nullptr` で安全にスキップ
3. **同一スレッド上でのシリアライゼーション** — callAsync dispatch と `~ConvolverProcessor()` は同じ Message Thread 上で実行されるため、ラムダがデストラクタの**実行中に**発火することはない

#### 設計上の位置付け

- WeakReference は「保険」ではなく**本質的な安全弁**
- `postCoalescedChangeNotification()` は既に WeakReference パターンを使用
- ConvolverProcessor は `JUCE_DECLARE_WEAK_REFERENCEABLE` を宣言済み
- **WeakReference を使わない設計は安全でない** — MessageManager は callAsync 順序を保証しない

### 2. PendingCommit デザイン（最終版）

```cpp
struct PendingCommit final
{
    StereoConvolver* newEngine = nullptr;
    std::unique_ptr<juce::AudioBuffer<double>> loadedIR;
    std::unique_ptr<juce::AudioBuffer<double>> displayIR;
    int targetLength = 0;
    double sampleRate = 0.0;
    double scaleFactor = 1.0;
    bool isRebuild = false;
    juce::File irFile;

    // releaseEngine: newEngine のみ安全に解放
    // 通常系では switchEngineOnMessageThread() 内で所有権が譲渡され
    // newEngine = nullptr になるため、デストラクタは安全に何もしない。
    // この関数は異常系（callAsync 失敗時）でのみ呼ばれる。
    void releaseEngine() noexcept
    {
        if (newEngine) {
            StereoConvolver::retireStereoConvolver(newEngine, nullptr);
            newEngine = nullptr;
        }
    }

    // Destructor releases engine only if ownership was never transferred.
    // Normal path: switchEngineOnMessageThread() transfers ownership → newEngine = nullptr → no-op.
    // Abnormal path: callAsync failed → releaseEngine() retires the engine.
    ~PendingCommit() noexcept { releaseEngine(); }
};
```

### 3. ConvolverProcessor 側のコミット実行

```cpp
void ConvolverProcessor::executePendingCommit(std::unique_ptr<PendingCommit> commit) noexcept
{
    if (!commit || !commit->newEngine) return;

    // Phase 1: IR メタデータ更新
    if (!commit->isRebuild)
        updateIRState(*commit->loadedIR, commit->sampleRate);

    // Phase 2: 可視化スナップショット
    // ★ 可視化は Engine に依存しない（IR データから直接計算するため）。
    //   Engine swap 前に実行することで、swap 時間を短縮する。
    //   Engine swap 後でも意味は同じだが、swap 中の UI レスポンスを
    //   避けるため、Swap 前に実行するのが望ましい。
    if (commit->displayIR && commit->displayIR->getNumSamples() > 0) {
        createWaveformSnapshot(*commit->displayIR);
        createFrequencyResponseSnapshot(*commit->displayIR, commit->sampleRate);
    }

    // Phase 3: Engine swap（排他的）
    // Precondition:  commit->newEngine != nullptr (checked at function entry)
    //                switchEngineOnMessageThread() is noexcept (ConvolverProcessor.h:608)
    // Postcondition: commit->newEngine has been consumed by exchangeActiveEngine()
    //                and must no longer be accessed by caller.
    //                Old engine (if any) is retired via retireStereoConvolver().
    switchEngineOnMessageThread(commit->newEngine);
    commit->newEngine = nullptr;  // 所有権譲渡（Postcondition: consumed by exchangeActiveEngine）

    // Phase 4: Publish（engine swap 完了後）
    publishAtomic(irLength, commit->targetLength, std::memory_order_release);
    publishAtomic(currentSampleRate, commit->sampleRate, std::memory_order_release);
    publishAtomic(currentIRScale, commit->scaleFactor, std::memory_order_release);
    publishAtomic(irFinalized, true, std::memory_order_release);

    // Phase 5: Latency refresh（new engine のレイテンシを正しく読む）
    refreshLatency();

    // Phase 6: 状態クリア
    publishAtomic(isLoading, false, std::memory_order_release);
    publishAtomic(isRebuilding, false, std::memory_order_release);

    // Phase 7: UI 通知
    updateLatencyCache();
    requestHostDisplayUpdate();
    postCoalescedChangeNotification();
}
```

### 4. applyNewState での使用（WeakReference ベース）

```cpp
void ConvolverProcessor::applyNewState(...) noexcept
{
    auto commit = std::make_unique<PendingCommit>();
    commit->newEngine = newConv;
    commit->targetLength = targetLength;
    commit->sampleRate = loadedSR;
    commit->scaleFactor = scaleFactor;
    commit->isRebuild = isRebuild;
    commit->irFile = file;
    commit->loadedIR = std::move(loadedIR);
    commit->displayIR = std::move(displayIR);

    auto* commitPtr = commit.release();
    auto weakThis = juce::WeakReference<ConvolverProcessor>(this);

    const bool queued = juce::MessageManager::callAsync(
        [weakThis, commitPtr]() {
            auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
            if (auto* self = weakThis.get())
            {
                // 正常パス: Processor がコミットを実行
                self->executePendingCommit(std::move(ownedCommit));
            }
            // 破棄済みの場合: releaseEngine() は ~PendingCommit で自動実行
        });

    if (!queued) {
        // MessageManager シャットダウン中 — 同期的にクリーンアップ
        auto ownedCommit = std::unique_ptr<PendingCommit>(commitPtr);
        ownedCommit->releaseEngine();
        // loadedIR / displayIR は RAII で自動解放
    }
}
```

### 5. 工数の再評価

| 項目 | 時間 |
|------|------|
| PendingCommit 構造体 + executePendingCommit 実装 | 2時間 |
| applyNewState 変更 + callAsync パターン | 2時間 |
| ライフサイクルテスト（シャットダウン試験） | 2時間 |
| オーディオ回帰テスト | 2時間 |
| MessageThread 回帰テスト | 1時間 |
| コードレビュー | 1時間 |
| **合計** | **10時間（1.5営業日）** |

---

## 📊 最終評価

| Bug | 評価 | 状態 |
|-----|------|------|
| C-2 | A+ | ✅ 実装完了 |
| C-10 | A | ✅ 実装完了 |
| C-6 | A | ✅ 実装完了 |
| H-13 | A | ✅ 実装完了 |
| C-7 | A+ | ✅ 設計完成 |

**総合評価: A（96点）**

> **注記**: `executePendingCommit() noexcept` は、内部で呼ぶ `createWaveformSnapshot` / `createFrequencyResponseSnapshot` / `updateLatencyCache` / `requestHostDisplayUpdate` が全て noexcept であることを前提とする。実装時にソースコードで確認が必要。

---

## 📝 変更履歴

| 版 | 主な変更 |
|----|---------|
| v1 | 初版（73件のバグリスト作成） |
| v2 | 検証結果反映（38件REFUTED、12件DESIGN_CHOICE） |
| v3 | lock ordering / Publish ordering / CommitTransaction 導入 |
| v4 | snapshot-first / CommitTransaction / releaseEngine |
| v5 | PendingCommit / this 寿命証明 / 可視化順序根拠 |
| v6 | **WeakReference ベースの寿命管理に修正**（MessageManager は callAsync 順序を保証しない） |
