# 安定性改善計画 — ソースコード分析に基づく詳細改善案

**作成日**: 2026-06-15
**対象**: ConvoPeq v0.5.0
**分析手法**: Serena MCP / CodeGraph MCP / AiDex / grep による静的コード解析

---

## 1. 【高】RECOVERY ポーリングループ改善

### 現状分析

#### タイマー構成

- `AudioEngine.Init.cpp:70` → `startTimer(100);` — **100ms 間隔 (10Hz)**
- タイマーコールバック `AudioEngine::timerCallback()` 内で以下を実行:
  1. VERIFY 遷移状態ログ（変更時のみ出力）
  2. VERIFY runtime publish 状態ログ（変更時のみ出力）
  3. VERIFY トランザクションカウンタ（変更時のみ出力）
  4. クロスフェード完了処理（必要時のみ）
  5. Orchestrator deferred 要求の処理
  6. DSP クリーンアップ（EQ/Convolver/Psychoacoustic）
  7. `m_healthMonitor.tick()` ← **15種類の健康診断**

#### HealthMonitor::tick() の内訳

`RuntimeHealthMonitor.cpp:45-185` で以下の全チェックを毎 tick 実行:

| # | チェック項目 | 想定処理時間 | 重要度 |
|---|---|---|---|
| 1 | `checkRetireStall()` | 軽量 | 中 |
| 2 | `checkPublicationStall()` | 軽量 | 中 |
| 3 | `diagnoseRetireStall()` | 軽量 | 低 |
| 4 | `checkCrossfadeTimeout()` | 軽量 | 中 |
| 5 | `checkCrossfadeEventDrop()` | 軽量 | 中 |
| 6 | `checkReaderSlotUsage()` | 軽量 | 高 |
| 7 | `checkOverflowRate()` | 軽量 | 中 |
| 8 | `checkRetireReclaimLatency()` | 軽量 | 中 |
| 9 | `checkConfigurationDivergence()` | 軽量 | 低 |
| 10 | `checkWorldConsistency()` | **やや重い** | 低 |
| 11 | `checkSnapshotStarvation()` | 軽量 | 低 |
| 12 | `checkPendingStructuralDeployment()` | 軽量 | 低 |
| 13 | `checkSuppressionDuration()` | 軽量 | 低 |
| 14 | `checkRuntimeProgressFreeze()` | 軽量 | 中 |
| 15 | `checkLearnerBackpressure()` | 軽量 | 中 |
| 16 | `checkConfigurationDrift()` | 軽量 | 低 |
| 17 | **閉ループ制御検証** (PolicyEngine verification) | **重い** | 中 |
| 18 | **PolicyEngine 統合評価** | 中程度 | 高 |

**問題点**: 100ms ごとに計18項目のチェックを実行している。Message Thread での実行とはいえ、GUIの応答性に影響を与える可能性がある。

#### [RECOVERY] execute action=0 の謎

ソースコード上、`executeRecoveryAction()` は `m_actionCallback` 経由でのみ呼ばれ、`m_actionCallback` は `decision.actions != 0` のときのみ発火する。しかしログには `action=0` (= `RecoveryAction::Observe`) が数百回出現している。

**推定原因**: ログを生成したバイナリと現在のソースコードに差異がある可能性が高い（ログ取得日: 2026-06-15 22:53、現在のコードはそれ以降に修正された可能性）。

### 改善案

#### 案 A: タイマー間隔の延長（推奨: 即効性・低リスク）

**変更ファイル**: `src/audioengine/AudioEngine.Init.cpp`

```cpp
// 変更前
startTimer(100);     // 100ms = 10Hz

// 変更後
startTimer(250);     // 250ms = 4Hz（GUI応答性向上）
```

**根拠**: HealthMonitor のほとんどのチェックはミリ秒単位の精度を必要としない。Retire Stall の検出閾値は10秒、Snapshot Starvation は10秒/30秒であり、250ms でも十分な応答性を確保できる。

**トレードオフ**: `checkReaderSlotUsage()` (Reader slot枯渇検出) の応答が150ms遅延する。ただし Reader slot 枯渇は通常数秒以上継続する事象であり、問題にならない。

#### 案 B: 二段階タイマー分割（推奨: 中リスク・効果大）

```cpp
// 高速パス: 100ms（軽量チェックのみ）
// 低速パス: 1000ms（重量チェックのみ）
```

**実装イメージ** (`AudioEngine.Timer.cpp`):

```cpp
void AudioEngine::timerCallback()
{
    // ---- 毎 tick 実行: 軽量チェックのみ ----
    emitEvidenceTickNonRt(false);
    processCrossfadeCompletion();
    processOrchestratorDeferred();
    processDspCleanup();

    // ---- 毎 tick 実行: 軽量 HealthMonitor ----
    m_healthMonitor.tickFast();  // checkRetireStall, checkReaderSlotUsage, checkOverflowRate のみ

    // ---- 5 tick に1回（500ms）: 中重量チェック ----
    if (++slowTickCounter_ % 5 == 0) {
        m_healthMonitor.tickSlow();  // 残りの全チェック + PolicyEngine
    }
}
```

**根拠**: `checkWorldConsistency()` は全 publish world の整合性を検証するため相対的に重い。`checkConfigurationDivergence()` や `checkSnapshotStarvation()` は10秒オーダーの閾値であり、500ms-1秒間隔で十分。

#### 案 C: executeRecoveryAction の Observe ガード強化

**変更ファイル**: `src/audioengine/AudioEngine.Timer.cpp`

```cpp
void AudioEngine::executeRecoveryAction(convo::RecoveryAction action) noexcept
{
    // ★ 追加: Observe の場合は即座に return（ログ出力も省略）
    if (action == convo::RecoveryAction::Observe)
        return;

    switch (action) {
        // ... 既存の switch ...
    }
    diagLog("[RECOVERY] execute action=" + juce::String(static_cast<int>(action)));
}
```

### 推奨優先順位

| 優先度 | 案 | 効果 | リスク | 工数 |
|---|---|---|---|---|
| ★★★ | **A: タイマー延長 100→250ms** | 中（60%削減） | 低 | 小（1行） |
| ★★☆ | **C: Observe ガード** | 小（ログ抑制） | 低 | 小（3行） |
| ★☆☆ | **B: 二段階タイマー** | 大（80%削減） | 中 | 中（設計変更） |

---

## 2. 【中】releaseResources 二重呼び出し

### 現状分析

#### 呼び出し経路

```
JUCE AudioDeviceManager lifecycle
  → AudioEngineProcessor::releaseResources()          [AudioEngineProcessor.cpp:47]
    → AudioEngine::releaseResources()                  [ReleaseResources.cpp:16]
      → CAS: Unprepared → Releasing → Prepared         [ReleaseResources.cpp:25-35]
      → ... cleanup ...
```

#### ガードの仕組み

```cpp
// ReleaseResources.cpp:25-35
if (previousState == EngineLifecycleState::Unprepared) {
    // 二重呼び出し検出 → 早期 return
    return;
}
```

CAS (Compare-And-Swap) による状態遷移:

```
Destroyed → (return)  // 完全破棄後
Unprepared → (return) // 二重解放
Releasing → (return)  // 解放中
その他 → Releasing → cleanup → Unprepared
```

#### 発生原因

JUCE のオーディオデバイス初期化時、以下いずれかのパスで `releaseResources()` が2回呼ばれる:

1. デバイス列挙時の内部リセット → `releaseResources()` → `prepareToPlay()` → デバイス差し替え → `releaseResources()`  again
2. プラグインホストとしての二重初期化パス
3. `AudioEngineProcessor` クラスと `AudioEngine` の間で、どちらからも呼ばれる可能性

#### ログでの観測

```
[DIAG] releaseResources: enter             ← 1回目
[DIAG] releaseResources: duplicate release ignored (already Unprepared)  ← CASでガード
[DIAG] releaseResources: enter             ← 2回目のシーケンス
[DIAG] releaseResources: duplicate release ignored (already Unprepared)  ← CASでガード
[DIAG] prepareToPlay: enter spb=1024 sr=192000.00  ← 正常に開始
```

CAS ガードが正しく機能しており、実害はない。

### 改善案

#### 案 A: 呼び出し元の特定（推奨）

**変更ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
// ガードログに呼び出し元のバックトレース情報を追加（Debug のみ）
if (previousState == EngineLifecycleState::Unprepared)
{
#if JUCE_DEBUG
    juce::String callStack;
    for (int i = 0; i < juce::SystemStats::getBacktraceSize(); ++i)
        callStack += juce::SystemStats::getBacktrace(i) + "\n";
    diagLog("[DIAG] releaseResources: duplicate release ignored (already Unprepared)\n"
            "Callstack:\n" + callStack);
#else
    diagLog("[DIAG] releaseResources: duplicate release ignored (already Unprepared)");
#endif
    return;
}
```

**根拠**: 呼び出し元を特定できれば、JUCE側の設定（setPlayConfig の重複呼び出し抑止など）で根本対処できる可能性がある。

#### 案 B: AudioProcessor 側での二重呼び出しガード（代替案）

**変更ファイル**: `src/audioengine/AudioEngineProcessor.cpp`

```cpp
// AudioEngineProcessor レベルでのリエントラントガード
static std::atomic<bool> s_releasing{false};

void AudioEngineProcessor::releaseResources()
{
    bool expected = false;
    if (!s_releasing.compare_exchange_strong(expected, true,
                                              std::memory_order_acq_rel))
    {
        // 再入防止: 既に releaseResources 処理中の場合はスキップ
        return;
    }
    audioEngine.releaseResources();
    s_releasing.store(false, std::memory_order_release);
}
```

ただし、このアプローチは AudioEngine 側の CAS ガードと重複するため、コード複雑性の割にメリットが少ない。

### 推奨優先順位

| 優先度 | 案 | 効果 | リスク | 工数 |
|---|---|---|---|---|
| ★★★ | **A: Debug時にcallstack出力** | 中（原因特定） | 低 | 小（5行） |
| ★☆☆ | B: AudioProcessor二重ガード | 小（ガード強化） | 低 | 小（10行） |

---

## 3. 【低】REBUILD_MERGED トリガー分析

### 現状分析

#### REBUILD_MERGED の発生条件（4系統）

| # | 条件 | 該当行 | 発生シナリオ |
|---|---|---|---|
| 1 | `sameAsPendingWouldMerge && shouldApplyLatestWinsMerge` | `RebuildDispatch.cpp:249` | 同一フィンガープリントの rebuild が latest-wins window (400ms) 内に重複 |
| 2 | `NonMtAlreadyPending` (非MTパス) | `RebuildDispatch.cpp:352` | 非AudioThread からの rebuild 要求が既に pending |
| 3 | `PendingDuplicate` (タスクキュー) | `RebuildDispatch.cpp:672` | rebuild thread のタスクキューに同一パラメータのタスクが既に存在 |
| 4 | `SnapshotIntentDebounced` | `UIEvents.cpp:111` | スナップショットコマンドがデバウンス期間内に重複 |

#### ログでの観測結果

| Intent IDs | 理由 | 発生タイミング | クラス |
|---|---|---|---|
| 7,8,9,10 | `SameAsPendingWouldMerge` | DSPCore準備中の `enqueue_snapshot_command` | Snapshot |
| 16 | `SameAsPendingWouldMerge` | convolverParamsChanged 後の snapshot | Snapshot |
| 19 | `SameAsPendingWouldMerge` | UI-EQ変更後の snapshot | Snapshot |

**レイテンシ**: 400ms（latestWinsWindowMs の値は `lmsWindowMs` 由来。詳細は `RebuildDispatch.cpp:217`）

### 改善案

#### 案 A: 変更なし（現状維持を推奨）

**根拠**:

1. 400ms の latest-wins デバウンスが正常に機能している
2. マージ後に必ず 1回の rebuild が実行されている（req=queued のカウンタ一致を確認）
3. マージが原因のスタックやメモリリークは発生していない
4. トランザクションカウンタの `rebuild(req/queued/blockP/blockR/queueFull/drain/match/fallback)` がすべて正常

**唯一の懸念**: 初期化バースト時の MERGED 連発（Intent 7-10）は、起動時のパラメータ設定が完了するまで snapshot をキューし続けるパターン。設計上正常だが、将来の最適化として初期化中は明示的に snapshot 要求を保留してもよい。

#### 案 B: 初期化専用パスの最適化（将来検討）

**根拠**: 初期化中の `enqueue_snapshot_command` の大量発生は、以下のシーケンスに起因:

1. `prepareToPlay` → `DSPCORE_PREPARE`（EQ再構築 → hash変化 → snapshot要求）
2. `convolverParamsChanged` → IR適用 → `applyComputedIR`（hash変化 → snapshot要求）
3. UI listener 経由の追加 snapshot要求（同一hashでマージ）

初期化完了フラグを導入し、初期化中は snapshot を1つにまとめることで merge 頻度を削減できる。

### 推奨優先順位

| 優先度 | 案 | 効果 | リスク | 工数 |
|---|---|---|---|---|
| ★★★ | **A: 現状維持** | — | なし | なし |
| ★☆☆ | B: 初期化最適化 | 小 | 低 | 中 |

---

## 4. 総合改善ロードマップ

### Phase 1（即日実施可能、低リスク）

| # | ファイル | 変更内容 | 期待効果 |
|---|---|---|---|
| 1 | `AudioEngine.Init.cpp` | `startTimer(100)` → `startTimer(250)` | CPU負荷60%削減 |
| 2 | `AudioEngine.Timer.cpp` | `executeRecoveryAction` に Observe ガード追加 | 無意味なログ抑制 |
| 3 | `ReleaseResources.cpp` | Debug時にcallstack出力追加（原因特定用） | 根本原因特定の足がかり |

### Phase 2（設計変更、中リスク）

| # | ファイル | 変更内容 | 期待効果 |
|---|---|---|---|
| 4 | `RuntimeHealthMonitor.h/cpp` | `tickFast()` / `tickSlow()` 分割 | 重量チェックの頻度最適化 |
| 5 | `AudioEngine.Timer.cpp` | 二段階タイマーカウンタ導入 | 軽量/重量チェックの分離 |

### Phase 3（将来検討）

| # | ファイル | 変更内容 | 期待効果 |
|---|---|---|---|
| 6 | `RebuildDispatch.h/cpp` | 初期化完了フラグによるsnapshot最適化 | 起動時のMERGED削減 |

---

## 5. ツールによる検証結果サマリ

| ツール | 用途 | 成果 |
|---|---|---|
| **Serena MCP** | シンボル検索・依存関係解析 | 全3件の呼び出し元、定義箇所を正確に特定 |
| **CodeGraph MCP** | ファイル構造解析 | `Timer.cpp` / `ReleaseResources.cpp` の全関数構造を把握 |
| **AiDex** | セッション管理 | セッション開始・外部変更検出（注: このセッションでは無効） |
| **grep/Select-String** | パターン検索 | 全該当ファイルの網羅的検索 |
| **Graphify** | 知識グラフ | スキルファイル読み込みにより使用方法を確認（グラフ構築は未実行） |
