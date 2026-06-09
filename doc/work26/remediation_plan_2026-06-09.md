# Practical Stable ISR Bridge Runtime 改修計画書 (v19)

**作成日**: 2026-06-09
**最終更新**: 2026-06-10 (v19: 最終レビュー反映)
**参考**: `doc/work26/harden_all_items_2026-06-09.md` (全項目100%確度確定)
**ベース**: ConvoPeq main branch
**達成目標**: Practical Stable ISR Bridge Runtime (実運用で破綻しにくい ISR Bridge Runtime)
**現在の達成度**: 約60〜65%
**v19実施後の達成度**: **94〜97%** (6項目補強後)

---

## 全体工程 (Tier S+/S/A/B/C 実運用優先順位 v19)

| Tier | 内容 | 項目 | 達成度 |
| --- | --- | --- | --- |
| **Tier S+** | StateOwner純化+Telemetry分離+Ledger完成 | StateOwner(State+Ledgerのみ, TelemetryRecorder分離), Ledger一次+Progress副産物+drop時間追跡, CorrelationID(128bit: struct {engineInstanceId,localCounter}) | 65% → 80% |
| **Tier S** | Authority封鎖+7段階ForwardProgress | P0-2/3, P0-5(compile-time主, grep参考のみ), OrchestratorForwardProgress(7段階+stage-gap detection) | 80% → 86% |
| **Tier A** | 可観測性+レート制限+Shutdown+DeferredHealth | P1-8(二層化+reason別トークンバケット), P1-5, P-New(Stall+ring固定), ShutdownScope(shutdownGeneration拘束), P1-7(activeReaderCount条件追加), DeferredHealth(deferredCount+oldestAge+overwrite) | 86% → 92% |
| **Tier B** | Executor+因果追跡+Audit | P0-1, P2-2(RetireTimeline固定4096) | 92% → 95% |
| **Tier C** | 運用監視 | P1-6(AdaptiveBP), P0-4(ReadHandle返却) | 95% → 97% |
| 保留 | 危険+未確定 | Recovery全体, ReaderIsolate, P1-4, P2-3, P1-1, P3-3, P3-2, P1-2, P1-3, P1-9 | — |

### リングバッファ固定サイズ一覧 (v17統一)

| バッファ | 固定サイズ | 用途 |
| --- | --- | --- |
| `FailureRecord` | **512** | 軽量失敗レコード (常時) |
| `FailureSnapshot` | **64** | 詳細失敗スナップショット (閾値超過時のみ) |
| `ProgressRecord` | **4096** | 出版進捗レコード (Ledger副産物) |
| `RetireTimeline` | **4096** | 退役タイムライン |
| `OrchestratorHealth` | **256** | Orchestrator健全性スナップショット |

Evidence出力時は各バッファから**最新N件のみ**を出力。

---

# Tier S+: StateOwner純化 + Telemetry分離 + Ledger完成 (v19)

**目標: StateOwner は State + Ledger のみに徹する。Telemetry (Failure/Progress/Health) は別コンポーネント `TelemetryRecorder` に分離し、StateOwner が巨大 GodObject 化するのを防止する。**

---

## StateOwner 書込契約 + TelemetryRecorder 分離 (v19: 再設計)

### 問題

v18 では StateOwner が State 更新 + Ledger + ProgressRecord + FailureRecord + Health + Correlation 全てを `onXxx()` に集約していた。このままだと `onPublished()` が肥大化し、長期的に GodObject へ変質する。

### 目標

**StateOwner は State 更新のみ。** Telemetry は `TelemetryRecorder` が担当。Ledger は State 内に保持。

```
Orchestrator
  ├── stateOwner.onPublished()       ← State + Ledger 更新のみ (軽量)
  └── telemetryRecorder.recordProgress(...)  ← 副産物 (別コンポーネント)
```

### 設計

```cpp
// ============================================================
// ★ StateOwner: State + Ledger のみ (軽量)。GodObject化防止
// ============================================================
class RuntimePublicationStateOwner {
    friend class RuntimePublicationOrchestrator;  // ★ 唯一の書込権限者
public:
    // onXxx() は State の更新のみ。Telemetry/Progress/Failureは呼ばない
    void onSubmitted(...);      // state_.ledger.submittedCount++
    void onBuilt(...);          // state_.ledger.builtCount++
    void onValidated(...);      // state_.ledger.validatedCount++
    void onPublished(...);      // state_.ledger.publishedCount++
    void onRetired(...);        // state_.ledger.retiredCount++
    void onReclaimed(...);      // state_.ledger.reclaimedCount++
    void onRejected(...);       // state_.ledger.rejectedCount++
    void onValidationFailed(...);// state_.ledger.validationFailedCount++
    void onExecutorFailed(...); // state_.ledger.executorFailedCount++

    [[nodiscard]] const RuntimePublicationState& state() const noexcept { return state_; }

private:
    RuntimePublicationState state_;  // Ledger 内包
};

// ============================================================
// ★ TelemetryRecorder: 副産物専用。StateOwner から完全分離
// ============================================================
class TelemetryRecorder {
public:
    void recordProgress(uint64_t correlationId, PublishStage stage, ...);
    void recordFailure(const FailureRecord& record);
    void recordHealth(const OrchestratorHealthSnapshot& snapshot);
    void recordDeferredHealth(const DeferredHealth& health);

    // Evidence 出力用スナップショット
    [[nodiscard]] TelemetrySnapshot captureSnapshot() const;
};
```

**更新契約ルール (v19):**

| ルール | 内容 |
| --- | --- |
| **R1** | State への書込は `StateOwner::onXxx()` のみ。`state_.xxx++` 直接操作禁止 |
| **R2** | StateOwner の `onXxx()` は **State + Ledger 更新のみ**。Telemetry/Progress/Failure は呼ばない |
| **R3** | Telemetry は `TelemetryRecorder` が独立して記録。Orchestrator が両方を呼ぶ |
| **R4** | `TelemetryRecorder` は StateOwner に依存しない。const 参照のみ読み取り可 |
| **R5** | `CorrelationId` は全記録に必須パラメータ。Orchestrator が払い出す |
| **R6** | `onXxx()` 呼出し権限は Orchestrator のみ。Executor/Transition/Shutdown は Orchestrator 経由 |

**呼出し経路:**

```
Executor::publish()
  └─ PublishResult
       └─ Orchestrator::onPublishCompleted()
            ├── stateOwner.onPublished()          ← 軽量 (State+Ledgerのみ)
            └── telemetryRecorder.recordProgress() ← 副産物 (別コンポーネント)
```

### 改修手順

1. `RuntimePublicationState` 構造体を定義 (Ledger + drop時間追跡を内包)
2. `RuntimePublicationStateOwner` クラスを実装 (friend Orchestratorのみ, State+Ledger更新に限定)
3. `TelemetryRecorder` クラスを新設 (ProgressRecord/FailureRecord/HealthSnapshot/DeferredHealthを保持)
4. Orchestrator が stateOwner.onXxx() + telemetryRecorder.recordXxx() の両方を呼ぶ
5. CI: `state_.` 直接書込検出 + `TelemetryRecorder` が StateOwner を直接書かないことを保証

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationState.h` | (新規) State + Ledger 定義 (friend Orchestratorのみ) |
| `src/audioengine/RuntimePublicationState.cpp` | (新規) onXxx() 実装 (State+Ledgerのみ) |
| `src/audioengine/TelemetryRecorder.h` | (新規) TelemetryRecorder (Progress/Failure/Health/Deferred) |
| `src/audioengine/TelemetryRecorder.cpp` | (新規) 実装 |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | StateOwner + TelemetryRecorder の両方を保持・呼出 |
| `src/audioengine/PublicationExecutor.h/.cpp` | PublishResult を Orchestrator に返す (直接onXxx()呼ばない) |

---

## PublicationLedger 一次情報化 + ProgressRecord 副産物化 (v19: ドロップ時間追跡)

### 問題

v16 では `ProgressRecord` と `PublicationLedger` が両方存在。これらは将来的に乖離する。例: `publishedCount=100` だが `ProgressRecord(Published)` が99件しか残っていない。リングが満杯の場合、Ledger 更新成功 → Progress push 失敗 が発生し、Evidence 欠落が解析不能になる。ドロップ数だけでは「いつ欠落したか」が分からない。

### 目標

**Ledger を一次情報源にする。** ProgressRecord は副産物として `TelemetryRecorder` が生成する。リング満杯によるドロップは `droppedProgressRecordCount` + `firstProgressDropUs` + `lastProgressDropUs` として Ledger に記録する。

### 設計

```cpp
// ★ Ledger は StateOwner の state_ 内。外部からは const 読み取りのみ
struct PublicationLedger {
    uint64_t submittedCount;
    uint64_t builtCount;
    uint64_t validatedCount;
    uint64_t publishedCount;
    uint64_t retiredCount;
    uint64_t reclaimedCount;
    uint64_t rejectedCount;
    uint64_t validationFailedCount;
    uint64_t executorFailedCount;
    uint64_t droppedProgressRecordCount;     // リング満杯によるドロップ累積数
    uint64_t firstProgressDropUs;            // ★ v19: 初回ドロップ時刻
    uint64_t lastProgressDropUs;             // ★ v19: 最終ドロップ時刻
};
```

**ドロップ記録ロジック (TelemetryRecorder):**

```cpp
void TelemetryRecorder::recordProgress(...) {
    if (!ringBuffer_.tryPush(...)) {
        stateOwner_.notifyProgressDrop(now);  // Ledger の drop 情報更新
    }
}

// StateOwner (軽量。Ledger更新のみ):
void RuntimePublicationStateOwner::notifyProgressDrop(uint64_t nowUs) {
    state_.ledger.droppedProgressRecordCount++;
    if (state_.ledger.firstProgressDropUs == 0)
        state_.ledger.firstProgressDropUs = nowUs;
    state_.ledger.lastProgressDropUs = nowUs;
}
```

**Evidence 出力例:**

```json
{
  "publishedCount": 100,
  "droppedProgressRecordCount": 3,
  "firstProgressDropUs": 1718000000000,
  "lastProgressDropUs": 1718005000000
}
```

### 改修手順

1. `PublicationLedger` に `firstProgressDropUs` / `lastProgressDropUs` 追加
2. `RuntimePublicationStateOwner::notifyProgressDrop()` 追加 (Ledger 更新のみ)
3. `TelemetryRecorder::recordProgress()` 内で `tryPush` 失敗時に通知
4. Evidence 出力時にドロップ時間を含める

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationState.h` | firstProgressDropUs/lastProgressDropUs 追加 |
| `src/audioengine/TelemetryRecorder.h/.cpp` | recordProgress() で tryPush + ドロップ通知 |

---

## Publication Correlation ID (v19: 128bit構造体)

### 問題

全 Telemetry を `(sequenceId, generation, worldId)` で統一する方針は良いが、これら3つを追跡するだけでは「1件の publish がどこで止まったか」を1本の線で追跡できない。3つ組は publish 毎に変化する。また `(engineInstanceId << 32) | localCounter` では 32bit localCounter が 2^32 を超えると再利用される (wrap問題)。

### 目標

1件の publish 要求に対して発行される**不変の `correlationId`** を全 Telemetry に付与する。**wrap 禁止のため 128bit 相当の内部表現を使用する。**

### 設計

```cpp
// ★ 128bit 相当の相関ID。内部保持は struct 2xuint64_t。
//    Telemetry 出力時だけ短縮表現 (hex または下位64bit) を使用。
struct CorrelationId {
    uint64_t engineInstanceId;   // 上位: Engine インスタンス識別子 (再生成後もユニーク)
    uint64_t localCounter;       // 下位: 単調増加カウンタ (wrap禁止。2^64 ≒ 1.8e19 で実質無制限)
};

// 採番:
CorrelationId nextCorrelationId() noexcept {
    return { engineInstanceId_, ++localCounter_ };  // 64bitカウンタなのでwrap不可
}

// 出力時短縮 (Evidence等):
uint64_t correlationIdShort() const noexcept {
    return correlationId.localCounter;  // 固定長。追跡には engineInstanceId も併記
}
```

**付与対象:**

```
Submitted(✓) → Built(✓) → Validated(✓) → Published(✓) → Retired(✓) → Reclaimed(✓)
  └─ Rejected(✓) └─ ValidationFailed(✓) └─ ExecutorFailed(✓)
       └─ Warning(✓) └─ Stall(✓) └─ Failure(✓) └─ Deferred(✓)
```

**運用例:** engineInstanceId + localCounter で grep するだけで、該当 publish の全生涯イベントが1列で取得可能。Engine 再生成後も engineInstanceId が異なるため衝突しない。

### 改修手順

1. `CorrelationId` 構造体を定義 (engineInstanceId, localCounter)
2. `TelemetryRecorder` に 64bit カウンタ追加 (wrap不可)
3. `submitPublishRequest()` 内で採番、全記録メソッドに渡す
4. 全レコード構造体に `CorrelationId` または出力用 short 値を追加
5. Evidence 出力時は engineInstanceId + localCounter の両方または short 値を含める

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | correlationId 採番 (struct) + 伝搬 |
| `src/audioengine/TelemetryRecorder.h` | CorrelationId 型定義 |

---

## ShutdownScope (v19: shutdownGeneration拘束, thread固定解除)

### 問題

v18 では `ownerThreadId` でスレッド拘束していたが、shutdown シーケンスが将来 MessageThread → ShutdownWorker → AudioThread停止 へ変化した場合に壊れる。スレッド固定は将来変更を難しくする。

### 目標

`ownerThreadId` を削除し、**`shutdownGeneration` + `engineInstanceId` + `expiration` の三要素**で拘束する。スレッド変更に耐性のある設計にする。

### 設計

```cpp
class ShutdownScope {
public:
    ShutdownScope(uint64_t engineInstanceId, uint64_t shutdownGeneration, uint64_t expiration)
        : engineInstanceId_(engineInstanceId)
        , shutdownGeneration_(shutdownGeneration)  // ★ threadId の代わり
        , expiration_(expiration)
        , active_(true) {}

    ~ShutdownScope() { active_ = false; }

    // ★ consume() 内で三要素検査 (thread非依存)
    ShutdownDrainToken consume() noexcept {
        if (!active_) return {};                    // 既に破棄
        if (now > expiration_) return {};           // 期限切れ
        // ★ shutdownGeneration 一致確認 (threadId 非依存)
        //   shutdownGeneration は AudioEngine の shutdown 毎に更新
        active_ = false;
        return ShutdownDrainToken{engineInstanceId_, shutdownGeneration_, ...};
    }

    ShutdownScope(const ShutdownScope&) = delete;
    ShutdownScope& operator=(const ShutdownScope&) = delete;

private:
    uint64_t engineInstanceId_;
    uint64_t shutdownGeneration_;  // ★ threadId → shutdownGeneration に変更
    uint64_t expiration_;
    bool active_;
};
```

**三要素の役割:**

| 要素 | 役割 | 変更耐性 |
| --- | --- | --- |
| `engineInstanceId` | Engine インスタンス識別 | 再生成後もユニーク |
| `shutdownGeneration` | Shutdown 毎に単調増加。Token がどの shutdown サイクルで発行されたか識別 | スレッド構成変更に影響されない |
| `expiration` | 期限切れ Token の悪用防止 | — |

### 改修手順

1. `ShutdownScope` から `ownerThreadId_` 削除 → `shutdownGeneration_` に置換
2. `consume()` から `GetCurrentThreadId()` 検査削除
3. `ShutdownDrainToken` に `shutdownGeneration` を追加 (Token の有効性確認用)
4. AudioEngine 側で `shutdownGeneration` を shutdown 毎に更新

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/ShutdownScope.h` | (新規) shutdownGeneration拘束版 (thread非依存) |
| `src/audioengine/AudioEngine.h/.cpp` | shutdownGeneration 管理追加 |

---

**目標: Coordinator生成権限の拡散を防止し、全publish経路でAdmission通過を強制する。**

---

## P0-2/3 (統合): Authority封鎖 (Admission bypass除去 + Coordinator生成封鎖)

**実運用で最も破綻リスクが高い。Coordinator生成権限の拡散が最大のAuthority問題。**

### 現状 (v2実コード確認)

**重要: P0-2 と P0-3 は分離できない。** Coordinator生成封鎖とAdmission強制は同一作業である。

`makeRuntimePublicationCoordinator()` が public であり、任意のコードが Coordinator を生成可能。

呼び出し元10箇所:

| # | ファイル | 用途 |
| --- | --- | --- |
| 1 | `AudioEngine.h:2733` | publishWorld() 内部 (削除予定) |
| 2 | `AudioEngine.CtorDtor.cpp:127` | ~AudioEngine shutdown |
| 3 | `AudioEngine.Init.cpp:46` | bootstrap publish |
| 4 | `PrepareToPlay.cpp:124` | prepareToPlay |
| 5 | `PrepareToPlay.cpp:236` | prepareToPlay (2nd) |
| 6 | `ReleaseResources.cpp:124` | releaseResources |
| 7 | `ReleaseResources.cpp:196` | releaseResources drain |
| 8 | `AudioEngine.Timer.cpp:404` | クロスフェード完了後 |
| 9 | `DSPTransition.h:115` | onTransitionComplete |
| 10 | `PublicationExecutor.cpp:15` | publish() (維持) |

### 目標

通常 publish 経路は全経路で `PublicationAdmission::evaluate()` が必須となる。

**ただし shutdown 経路 (#2: `~AudioEngine()`) は Admission bypass を許可する。**
理由: shutdown 中の最終排出用 publish が Admission で Rejected されると、DSPリソースが解放できずメモリリークの原因となる。

Coordinator の**短寿命・値オブジェクト設計 (move-only, 毎回生成) は維持。Stateful 化は行わない。**

### 改修手順

1. `makeRuntimePublicationCoordinator()` を private に移動 (friend Orchestrator)
2. `AudioEngine::publishWorld()` を private 化 (内部でAdmissionを呼ばない経路を削除)
3. 全10箇所の Coordinator 呼び出しを `RuntimePublicationOrchestrator` 経由に変更:
   - **#2 (shutdown)**: **Admission bypass許可**。**ShutdownDrainToken + ShutdownScope 二重条件**。Tokenには `shutdownGeneration`, `generation`, `shutdownEpoch`, `expiration`, `engineInstanceId`（★ AudioEngine再生成時の誤使用防止）を持たせる。Tokenは **move-only** (copy deleted) + **`consume()` 一回限り利用** + **`consume()` 内で `now > expiration` 検査必須**（期限切れTokenの悪用防止）。さらに `AudioEngine::ShutdownScope` を追加。Scope active中のみ submitShutdownDrain() を許可。Token漏洩だけでは bypass 不可。
   - #3 (bootstrap): Orchestrator の bootstrap publish 機能
   - #4/#5 (prepareToPlay): Orchestrator::submitPublishRequest()
   - #6/#7 (releaseResources): Orchestrator::submitPublishRequest() または shutdown 連動
   - #8 (timer): Orchestrator::notifyTransitionComplete()
   - #9 (DSPTransition): Orchestrator 経由に変更
   - #10 (PublicationExecutor): **維持。Executor は Orchestrator の委譲先**
4. 移行完了後に `AudioEngine::publishWorld()` を削除
5. `friend` 宣言されたクラス以外からの Coordinator 生成を CI で検出 (P0-5)

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/AudioEngine.h` | makeRuntimePublicationCoordinator() private化、publishWorld() private化、friend宣言追加 |
| `src/audioengine/AudioEngine.Init.cpp` | Orchestrator経由に変更 |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | Orchestrator経由に変更 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | Orchestrator経由に変更 |
| `src/audioengine/AudioEngine.Timer.cpp` | Orchestrator経由に変更 |
| `src/audioengine/DSPTransition.h` | Orchestrator経由に変更 |
| `src/audioengine/RuntimePublicationOrchestrator.h` | 新規public APIの追加 |
| `.github/scripts/` | (新規) Authority Audit スクリプト |

---

## P0-4: runtimeStore.observe() 直接利用除去 (Store private化は不要)

### 現状 (harden確定 — Store書込は既にprotected)

**P0-4 は「Store private化」ではない。Store書込権限は既に protected。**

`RuntimeStore<World, Owner>::acquireWriteAccess()` は **private** (`friend Owner` でのみアクセス可)。Owner = `RuntimePublicationCoordinator`。

つまり「Store公開 ≠ Store書込可能」。Store自体の変更は不要。真のAuthority問題は `makeRuntimePublicationCoordinator()` の公開状態にあり、P0-2/3で解決する。

`acquireReadToken(runtimeStore)` が `AudioEngine.h` 内4箇所で使われている:

- `getRuntimeGraph()` (line 906)
- `makeRuntimeReadHandle()` (line 2334)
- `computeRuntimePublishComputation()` (line 2600)
- `logRuntimeTransitionEvent()` (line 2768)

`acquireReadToken` は `RuntimePublicationCoordinator::acquireReadToken(const Store&)` の静的メソッドで、空の `ReadToken{}` を返す (実質 no-op)。

また `Orchestrator.cpp:67` で `engine_.runtimeStore.observe()` による直接観測が1箇所存在する。

### 目標

**Store自体の変更は不要** (書込は既に friend Owner で保護済み)。

必要なのは以下のみ:

- Coordinator生成封鎖で間接的に WriteAccess 取得を制限
- Orchestrator からの `runtimeStore.observe()` を getter 経由に変更

### 改修手順

1. `runtimeStore` は public 維持 (ReadToken API 互換性のため)
2. Orchestrator からの `engine_.runtimeStore.observe()` → `engine_.observePublishedWorld()` getter 経由に変更
3. Coordinator 生成が private 化されることにより、WriteAccess 取得は自動封鎖
4. 読み取り専用の static getter を追加 (`consumePublishedWorld` のラッパー)

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | runtimeStore.observe() → getter経由 |
| `src/audioengine/AudioEngine.h` | observePublishedWorld() getter追加 |

### 依存関係

- P0-2/3 (Coordinator生成封鎖) 完了が前提。Coordinator が private 化されれば Store の WriteAccess 取得も自動制限される

---

## P0-4: runtimeStore.observe() 直接利用除去

**P0-4 は「Store private化」ではない。** `RuntimeStore::acquireWriteAccess()` は既に `friend Owner` でprotected。

実際に必要なのは Orchestrator からの `engine_.runtimeStore.observe()` 1箇所を getter 経由に変更するのみ。

Store private化は不要。

---

## P0-5: Publication Authority Audit (compile-time主防御)

Coordinator封鎖だけでは不十分で、将来の追加による封鎖崩壊を防止する。

**grep監査は補助に留める。** `using PubCoord = RuntimePublicationCoordinator` で容易に回避可能なため、grep は警告扱い。CI Fail にはしない。

### 改修手順

1. **主防御は compile-time**: `RuntimePublicationCoordinator` のコンストラクタを private にし、`friend` 限定で生成。これが唯一の確実な防御。
2. **grep は警告のみ**: `makeRuntimePublicationCoordinator(`, `RuntimePublicationCoordinator::`, `acquireWriteAccess(`, `publishWorld(` の直接利用を補助監査するが、CI Fail にはしない。警告としてレポート。
3. **friend増殖監査**: `friend RuntimePublicationOrchestrator`, `friend PublicationExecutor`, `friend RuntimePublicationStateOwner` 以外の friend 追加を CI Fail。friend 追加はコードレビュー必須。
4. **ShutdownDrainToken move-only**: `ShutdownDrainToken(const&)=delete`, `operator=(const&)=delete`, `consume()` 一回限り利用 + **`consume()` 内で `now > expiration` 検査必須**。
5. CI スクリプトで違反を検出し Fail

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/RuntimePublicationCoordinator.h` | constructor private 化、friend 宣言 |
| `src/audioengine/AudioEngine.h` | friend宣言の明示・制限 |

---

# Tier A: 可観測性(失敗追跡)

**目標: publish失敗理由を構造化レコードとして保持し、障害解析を可能にする。「防ぐより観測できる」が最優先。**

---

## P1-8: Publication Failure Telemetry (Tier A最優先)

**実運用観点で最も重要なのは「なぜ publish が失敗したか」の追跡。Tier Aの先頭で実施する。**

### 現状 (v6確定)

**`FailureRecord` は未実装。** `PublicationFailureTaxonomyVerifier` はスキーマ検証器であり障害レコードではない。

`PublishResult` は `Success/ValidationFailed/PublishFailed/BridgeFailed` を定義しているが、**なぜ失敗したか**の詳細が残らない。`lastRejectReason()` は Closure/Tier の2種類のみ。

### 目標

`PublishStageResult` + 失敗理由を構造化レコードとして保持し、Evidence に出力する。

### 改修手順

1. `PublicationFailureRecord` 構造体を定義:

   ```cpp
   struct PublicationFailureRecord {
       uint64_t publicationSequenceId;  // 軽量版（常時記録）
       FailureStage stage;
       FailureReason reason;
       const char* origin;
       uint64_t timestampUs;
   };
   ```

2. **二層化**: 通常は軽量 `FailureRecord` のみ記録。異常閾値超過時のみ `FailureSnapshot`（全フィールド）を取得。

   ```cpp
   struct FailureSnapshot {  // 詳細版（閾値超過時のみ）
       uint64_t correlationId;             // ★ 全Telemetry共通追跡ID
       uint64_t publicationSequenceId;
       uint64_t generation;
       uint64_t worldId;
       FailureStage stage;
       FailureReason reason;
       const char* origin;
       uint64_t threadId;
       CoordinatorState coordinatorState;
       ShutdownPhase shutdownPhase;
       PublicationClass publicationClass;
       uint64_t activeReaderCount;
       uint64_t minReaderEpoch;
       uint64_t currentEpoch;
       uint64_t timestampUs;
   };
   ```

3. **FailureSnapshot サンプリング制御**: 大量失敗時の暴走防止。以下の条件が**両方**満たされた場合のみ Snapshot を生成:

   ```cpp
   // ★ v19: 全FailureReasonで共有1バケット → FailureReason単位の独立バケットに変更。
   //   例: 1秒で ValidationFailure 50,000件 → 最初の10件だけ保存して59秒間保存なし
   //   を防止。FailureReason 毎に独立バケットを持つことで全種別を均等にサンプリング。
   //
   // Snapshot生成条件 (全て必須):
   //   1. FailureRecord の件数が N 超過
   //   2. 最後の Snapshot 生成から 1秒以上経過 (lastSnapshotAge > 1 sec)
   //   3. トークンバケット (FailureReason 単位): maxSnapshotsPerMinute = 10 / reason
   //
   //   これにより ValidationFailure / PublishFailure / BridgeFailure / ShutdownFailure
   //   がそれぞれ独立にサンプリングされ、障害解析能力が大幅に向上。
   class FailureSnapshotController {
       static constexpr uint32_t kMaxSnapshotsPerMinute = 10;
       struct ReasonBucket {
           std::atomic<uint32_t> snapshotCountThisMinute_{0};
           std::atomic<uint64_t> minuteStartUs_{0};
       };
       ReasonBucket buckets_[static_cast<size_t>(FailureReason::Count)];
   public:
       bool shouldTakeSnapshot(FailureReason reason, uint64_t failureCount, uint64_t nowUs) noexcept {
           auto& bucket = buckets_[static_cast<size_t>(reason)];
           if (nowUs - bucket.minuteStartUs_.load() > 60'000'000) {
               bucket.minuteStartUs_.store(nowUs);
               bucket.snapshotCountThisMinute_.store(0);
           }
           return failureCount > FailureSnapshotThreshold
               && (nowUs - lastSnapshotTimestampUs) > 1'000'000
               && bucket.snapshotCountThisMinute_.fetch_add(1) < kMaxSnapshotsPerMinute;
       }
   };
   ```

4. FailureRecord リングバッファ (**512**固定) を `RuntimePublicationOrchestrator` に保持
5. publish 失敗時に FailureRecord を記録、条件合致時のみ FailureSnapshot 生成
6. Evidence (`publication_failure_log.json`) に出力

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | FailureRecord追加 |
| `src/audioengine/ISREvidenceExporter.cpp` | 障害ログ出力 |

---

## P-New: Publication Stall Detection (新規:Tier A)

### 問題

現状の計画には `submitPublishRequest()` → Builder成功 → Executor投入 → 永遠にpublishされない経路の検出がない。Admission/Failure/ReclaimはあるがPublish Stallがない。

### 目標

Publication の各段階進捗を追跡し、一定時間停滞したら警告を発火する。

```
Submitted → Built → Validated → Published → Retired → Reclaimed
```

### 改修手順

1. `PublicationProgressRecord` を定義:

   ```cpp
   // ★ 全Telemetry共通: (sequenceId, generation, worldId) で統一
   enum class PublishStage : uint8_t {
       Submitted, Built, Validated, Published, Retired, Reclaimed
   };
   struct PublicationProgressRecord {
       uint64_t publicationSequenceId;
       uint64_t generation;
       uint64_t worldId;
       PublishStage stage;
       uint64_t timestampUs;
   };
   ```

2. `RuntimePublicationOrchestrator` で各段階のタイムスタンプを記録
3. **PublicationClass 別のStall閾値**: 固定5秒では巨大IRロード・大規模再構築で誤検出。以下に分類:

   ```cpp
   enum class PublicationClass : uint8_t {
       FastPath,      // 5s  — 通常publish
       HeavyBuild,    // 30s — 大規模IR再構築
       Shutdown       // 60s — shutdown publish
   };
   ```

4. `RuntimePublicationOrchestrator` で各段階のタイムスタンプとPublicationClassを記録
5. クラス別閾値超過で `PublicationStallWarning` 発火
6. Evidence (`publication_progress_log.json`) に出力

---

## RetireStallWarning (Stall検出の補完)

Published→Retired まで進んでも Reclaimed で止まるケースがRCU系の典型障害。これを別途監視する。

**条件例**: `retireEpoch 記録後30秒経過 && reclaimEpoch == 0`

RetireTimelineRecord の定期チェックで検出。**Reader情報も同時保存**:

```cpp
struct RetireStallSnapshot {
    uint64_t publicationSequenceId;
    uint64_t generation;
    uint64_t retireEpoch;
    uint64_t reclaimEpoch;     // 0=未完了
    uint64_t activeReaderCount;  // ★ 原因特定用
    uint64_t minReaderEpoch;     // ★ Reader停滞 or Retire queue fault の判別
    uint64_t currentEpoch;
    uint64_t pendingRetireCount;
};
```

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | ProgressRecord追加、Stall検出、RetireStall |
| `src/audioengine/ISREvidenceExporter.cpp` | 進捗ログ+RetireStall出力 |

---

## P-New2: Orchestrator Health Monitor (新規:Tier S+)

**RuntimePublicationOrchestrator が SPOF 化するリスクに対処。Orchestrator 自身の健全性監査が存在しない。**

### 改修手順

1. `RuntimePublicationStateOwner` を新設:

   `RuntimePublicationState` の唯一の所有者。Coordinator と Orchestrator は **read only** で参照。書き込みは StateOwner のみが行う。

   ```
   RuntimePublicationStateOwner
       ↓ (write)
   RuntimePublicationState
       ↓ (read only)
   Coordinator / Orchestrator
   ```

2. `OrchestratorForwardProgress` を追加:

   ```cpp
   struct OrchestratorForwardProgress {
       // ★ v19: 7段階カウンタ。現行 Coordinator は publish/retire/reclaim まで保持。
       //   retired/reclaimed 追加により「Publishedで止まった」と「Retiredで止まった」を区別可能。
       uint64_t submittedCount;    // Orchestrator が受付
       uint64_t builtCount;        // Builder が構築完了
       uint64_t validatedCount;    // Validator 通過
       uint64_t executedCount;     // Executor 実行
       uint64_t publishedCount;    // Coordinator publish 成功
       uint64_t retiredCount;      // ★ v19: retire 完了
       uint64_t reclaimedCount;    // ★ v19: reclaim 完了 (GC完了)
       uint32_t executorQueueDepth;
       uint64_t lastProgressTimestampUs;
   };
   ```

3. `RuntimePublicationOrchestrator` で各オペレーション実行時にカウンタ更新
4. **Stage-gap detection (v19: 7段階):** retired/reclaimed を追加し、RCU系の停滞も特定可能:

   ```cpp
   // 停止位置の診断 (7段階)
   if (submitted > built)     → StuckStage::Builder;
   if (built > validated)     → StuckStage::Validator;
   if (validated > executed)  → StuckStage::Executor;
   if (executed > published)  → StuckStage::Coordinator;
   if (published > retired)   → StuckStage::Retire;      // ★ v19: Retire停滞
   if (retired > reclaimed)   → StuckStage::Reclaim;     // ★ v19: Reclaim停滞 (RCU)
   ```

5. Stuck 診断情報として `executorQueueDepth` + `stuckStage` を HealthSnapshot に含める

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | 段階別カウンタ+Stage-gap detection追加 |

---

## P-New3: Publication Balance Monitor (新規:Tier S+)

**v12のStall Detectionは順方向のみ。Published→Retired→Reclaimed の収支監査がない。**

### 改修手順

1. `PublicationLedger` を追加（全段階の累積値+Rejected系）:

   ```cpp
   struct PublicationLedger {
       uint64_t submittedCount;
       uint64_t builtCount;
       uint64_t validatedCount;
       uint64_t publishedCount;
       uint64_t retiredCount;
       uint64_t reclaimedCount;
       uint64_t rejectedCount;            // ★ Admission reject
       uint64_t validationFailedCount;     // ★ Validation failure
       uint64_t executorFailedCount;       // ★ Executor failure
   };
   ```

2. 各段階のカウンタを `RuntimePublicationOrchestrator` で保持
3. 異常条件:
   - `builtCount - publishedCount > 閾値` → BuildLeak
   - `publishedCount - retiredCount > 閾値` → PublishLeak
   - **`oldestRetireAge > 30s`** → RetireLeak（件数固定Nではなく経過時間を主判定）
4. Evidence (`publication_ledger_log.json`) に出力

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | BalanceMonitor追加 |

---

## Recovery 戦略 (⚠️ 危険。Tier C に降格。自動Recoveryは保留推奨)

v13では `retryDeferredPublish` / `restartExecutorSession` / `readerIsolation` を提案したが、**これらは実運用で危険。**

- `restartExecutorSession()`: Executorのpending publish/sequence/retire stateとの整合性が保証されない
- `readerIsolation()`: RCU系でreader隔離はUse After Freeに直結
- `retryDeferredPublish()`: 比較的安全だが状態確認必須

**推奨方針**: `Recover` より `Detect → Evidence → Faulted` の方が安全。Recovery機構は十分に検証されるまで保留。

---

## P1-5: Evidence System 実データ化

**Evidence の空洞化が実運用で致命的。publish失敗→原因不明を解消する。**

### 現状 (harden確定)

**Coordinator→Executor エラー伝播設計が確定。`publishWorld()` の戻り値は void。**

`RuntimePublicationCoordinator::publishWorld()` の実際の戻り値は **`void`**:

```cpp
// src/core/RuntimePublicationCoordinator.h
void publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept
{
    if (!worldOwner) return;

    worldOwner->sealRecursively();

    if constexpr (requires(...) { bridge.validatePublicationNonRt(world); })
    {
        if (!bridge_.validatePublicationNonRt(*worldOwner))
        {
            auto* rejectedWorld = worldOwner.release();
            bridge_.retireRuntimePublishWorldNonRt(rejectedWorld, false);
            return;  // ← void return, エラーを呼び出し元に伝播しない
        }
    }
    // ... publishAndSwap, didPublish, willRetire, retire ...
}
```

つまり実装前に解決すべき課題:

Coordinator.publishWorld() → (validation failure) → Coordinator は void で return → Executor へエラー伝播が不可能

### PublishResult 現在の実態

```cpp
// PublicationExecutor.h:9-12
enum class PublishResult { Success, ValidationFailed, PublishFailed, BridgeFailed };

// PublicationExecutor.cpp:7-29
//   実際の戻り値: null world → PublishFailed, それ以外 → Success
//   ValidationFailed, BridgeFailed は一度も返されていない
```

**変更影響範囲**: Coordinator テンプレート (`RuntimePublicationCoordinator<World, Handle, Bridge>`) の単一インスタンスのみ。

### 目標

**`PublishOutcome` 型を新設し、Executor が Coordinator の成否を把握できるようにする。**

```cpp
enum class PublishOutcome : uint8_t {
    Success,
    ValidationFailed,
    BridgeFailed,
    ShutdownRejected
};
```

Executor パイプライン:

```
PublicationExecutor::publish()
  ├── coordinator.publishWorld()        ← PublishOutcome を返す
  │     └── bridge.validatePublicationNonRt()  ← Validation権限はBridge側のみ
  ├── exportEvidence()                   ← Evidence 出力
  └── return appropriate PublishResult
```

**設計原則:**

- **Validator は Executor で再実行しない。** Coordinator が `bridge_.validatePublicationNonRt()` を既に持つ。
- **retire 権限も Coordinator のみ。**
- Executor の責務は「Coordinator の実行結果を PublishResult にマッピングして Orchestrator に返す」こと。二重検証は行わない。

### 改修手順

1. `PublishStageResult` 列挙型を定義 (`Success`, `Rejected`, `Failed`)
2. `RuntimePublicationCoordinator::publishWorld()` の戻り値を `void` → `PublishStageResult` に変更
3. `PublicationExecutor::publish()` 内で `coordinator.publishWorld()` の戻り値を判定
4. `PublishResult` を Coordinator の outcome に基づいて設定
5. Evidence 出力を統合 (`exportEvidence()`)
6. 全 `PublishResult` 分岐が実際に返されることを確認

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/RuntimePublicationCoordinator.h` | publishWorld()戻り値をvoid→PublishStageResultに変更 |
| `src/audioengine/PublicationExecutor.h` | PublishStageResultに対応したPublishResult判定 |
| `src/audioengine/PublicationExecutor.cpp` | coordinator.publishWorld()の戻り値を評価、FailureReasonはExecutor側で生成 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | Executor呼び出しの調整 |

### リスク

- `publishWorld()` の戻り値変更は Coordinator テンプレート (`RuntimePublicationCoordinator<World, Handle, Bridge>`) の**単一インスタンスのみ**に影響（本プロジェクトでは1種類のみ使用）
- 戻り値変更はコンパイルエラーで全呼び出し元を検出可能

### Orchestrator パイプライン全体 (設計前提)

| Phase | 担当 | 内容 |
| --- | --- | --- |
| 1 | Orchestrator | Admission::evaluate() |
| 2 | Orchestrator | RuntimeBuilder::buildRuntimePublishWorld() |
| 3 | Orchestrator | CrossfadeAuthority::evaluate() |
| 4 | Executor | coordinator.publishWorld() → PublishStageResult |
| 5 | Orchestrator/Transition | DSPTransition::onPublishCompleted() |
| 6 | Orchestrator | advanceRetireEpoch() |

---

---

## P1-7: Reclaim Progress Monitoring (Monitor→Recover→Audit)

### 現状

`tryReclaim()` の呼び出しに完全依存。回収停滞が発生しても検知できない。Timer監視のみでは Timer停止中・Shutdown中・prepareToPlay前に穴がある。

### 目標

Reclaim の進捗を**複数箇所で分散監視**し、停滞時に**自動回復**する。

### 改修手順 (3段階)

**Monitor (監視):**

1. `pendingRetireCount()` の定期監視機構追加 (Timer は補助)
2. `maxRetireAge` (最古未回収エントリの経過時間) の追跡
3. 監視ソースを分散: `tryReclaim()`, `publish()`, `shutdown()` でも進捗確認

**Recover (自動回復):**
6. **Recoverは行わない**: `forceAdvanceEpoch()` はRCU系でUAF候補になるため完全削除。`Detect → Evidence → Warning → Faulted` の方針。
7. **Faulted条件に `activeReaderCount == 0` を追加**: reclaim 停滞は Reader が長時間処理中でも発生する。Reader が存在する場合は正常動作の可能性があるため、Faulted には至らない。`activeReaderCount == 0` かつ `oldestRetireAge > threshold` の場合のみ Faulted 遷移。

**Audit (監査):**
5. 回復失敗時: `DrainAudit` 発火 → Evidence出力
6. Evidence (`retire_latency_report.json`) への実データ反映

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/AudioEngine.Timer.cpp` | 定期監視追加 (補助) |
| `src/audioengine/AudioEngine.Commit.cpp` | publish/shutdown時の進捗確認追加 |
| `src/audioengine/ISREvidenceExporter.cpp` | retire_latency_report.json 実データ化 |
| `src/core/DeferredDeletionQueue.h` | 最大滞留時間追跡追加 |

---

## DeferredHealth: Deferred Publish 監査 (v19新設: Tier A)

### 問題

現行 Orchestrator は `DeferredPublishSlot` を持つが、計画では Deferred 滞留件数・寿命の監査が弱い。Deferred が枯渇せず滞留し続けると、publish 経路の隠れたデッドロック原因になる。

### 目標

Deferred publish の健全性を定量的に監視する。

### 設計

```cpp
struct DeferredHealth {
    uint64_t deferredCount;           // 現在の滞留件数
    uint64_t oldestDeferredAgeMs;     // 最古滞留時間 (ms)
    uint64_t overwriteCount;          // 上書き破棄回数 (累積)
    DiscardReason lastDiscardReason;  // 最終破棄理由
    uint64_t lastDiscardTimestampUs;  // 最終破棄時刻
};
```

**Evidence 出力 (`deferred_health.json`):**

```json
{
  "deferredCount": 1,
  "oldestDeferredAgeMs": 4500,
  "overwriteCount": 3,
  "lastDiscardReason": "SupersededDiscard",
  "lastDiscardTimestampUs": 1718005000000
}
```

### 改修手順

1. `DeferredHealth` 構造体を定義
2. `TelemetryRecorder` に DeferredHealth 記録メソッド追加 (`recordDeferredHealth()`)
3. Orchestrator の deferred 操作時 (enqueue/consume/clear/overwrite) に DeferredHealth 更新
4. Evidence 出力

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/TelemetryRecorder.h/.cpp` | DeferredHealth 記録追加 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | deferred 操作時の Health 更新 |
| `src/audioengine/ISREvidenceExporter.cpp` | deferred_health.json 出力 |

---

## P1-9: Publication Sequence Integrity (Warning+Evidence+Rollbackのみ) — 保留

### 現状 (v7確定)

**系統A (commit内のハードFaulted):** 既存。`sequenceId <= previousSequenceId` で即Faulted。変更不要。

**系統B (makeRuntimeReadHandle内の監視カウンタ):** `observeMonotonicRollbackRequested_` は既に `AudioEngine.Commit.cpp:352` で消費され `retireRuntimeEx_.requestRollback()` を呼んでいる。

つまり現在のコードは以下で十分機能している:

| 違反検出 | 現在の対応 |
| --- | --- |
| commit内 (系統A) | 即Faulted |
| observe内 (系統B) | requestRollback (soft recovery) |

### 不足分: `observeMonotonicViolationCount_` の読み取りとEvidence出力

**Faulted 遷移条件の追加は行わない。** Faulted は commit() 内の系統A単一路線で十分。系統Bに Faulted を追加すると条件二重化による運用難易度上昇・誤停止リスクが発生する。

### 改修手順 (縮小)

1. `observeMonotonicViolationCount_` の値を Evidence (`publication_sequence_report.json`) に出力
2. Warning ログを追加
3. 系統A (commit() 内の即 Faulted) は現状維持、変更しない

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/ISREvidenceExporter.cpp` | monotonicity違反Evidence出力 |

---

# Tier B: Executor + 監査

**目標: Executor完成と監査基盤の整備。**

---

## P0-1: PublicationExecutor 完成 (+ PublishStageResult 採用)

**Coordinatorは `PublishStageResult {Success,Rejected,Failed}` のみ返す。FailureReasonはExecutr側。**

Coordinator = 実行器、Executor = 解釈器、Orchestrator = ポリシー管理。

Coordinator は以下を知らない:

- システムポリシー (Admission)
- shutdown policy
- FailureReason の詳細分類

従って Coordinator が返すのは最小限の `PublishStageResult` のみ。

### 責務分離

```
Coordinator.publishWorld()
    → PublishStageResult (Success / Rejected / Failed)

Executor
    → PublishStageResult を受け取り、FailureReason を生成
    → Evidence/Telemetry 出力

Orchestrator
    → PublishResult にマッピングして上位に返す
```

Coordinator は `validate→publish→retire` の実行責務。Coordinatorが返すのは最小限:

```cpp
enum class PublishStageResult : uint8_t {
    Success,
    Rejected,
    Failed
};
```

FailureReason の詳細分類は **FailureRecord側・Executor側** に集約。Coordinatorは結果の事実のみを返し、解釈は行わない。

### 現状 (v2実コード確認)

`PublicationSequenceId` は既に存在する (`RuntimePublishWorld::publication.sequenceId`)。しかし `DeletionEntry` には紐付け情報がなく、publication と retire の因果を追跡できない。

```cpp
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*);
    uint64_t epoch = 0;
    DeletionEntryType type = Generic;
    // ★ publicationSequenceId なし
};
```

### 重要: trivially copyable 制約 (harden確認)

```cpp
// DeferredDeletionQueue.h — 既存の静的チェック:
static_assert(std::is_trivially_copyable_v<DeletionEntry>,
    "DeletionEntry must be trivially copyable for lock-free queue operations");
static_assert(std::is_trivially_destructible_v<DeletionEntry>,
    "DeletionEntry must be trivially destructible");
```

**`uint64_t publicationSequenceId` の追加は安全。** POD型のみのstructにuint64_t追加なら trivially copyable は維持される。

**サイズ影響**: 現状32bytes → uint64_t追加後40bytes (アライメントpadding含む)。kQueueSize=4096 で 32KB→40KB の増加、許容範囲。

`RetireId / ReclaimId` の新規導入は行わない。既存の `PublicationSequenceId` のみで十分追跡可能。

### 目標

`DeletionEntry` に `uint64_t publicationSequenceId` を追加し、publication → retire → reclaim の因果を追跡可能にする。`RetireTimelineRecord` を新設し、reclaim完了まで追跡。

### 改修手順

1. `DeletionEntry` に `uint64_t publicationSequenceId` と `uint64_t generation` を追加（障害解析時にsequenceId+generationの両方で追跡可能）
2. enqueue 時に publicationSequenceId と generation を記録
3. **`RetireTimelineRecord` を新設**:

   ```cpp
   struct RetireTimelineRecord {
       uint64_t publicationSequenceId;
       uint64_t generation;
       uint64_t worldId;               // ★ 追加: 全Telemetry共通
       uint64_t retireEpoch;
       uint64_t reclaimEpoch;
   };
   ```

4. `tryReclaim()` 成功時に reclaimEpoch を記録
5. Evidence (`retire_timeline.json`) に publish/retire/reclaim の時系列を出力

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/DeferredDeletionQueue.h` | DeletionEntry に publicationSequenceId 追加 |
| `src/audioengine/ISREvidenceExporter.cpp` | retire_timeline.json 拡張 (RetireTimelineRecord出力) |

---

# Tier C: 実運用監視

**目標: Runtime 実運用監視と軽微なクリーンアップ。**

---

## P1-6: QueuePressure 段階制御 (Adaptive Backpressure) [Tier C]

### 現状

ConvoPeq には既に `CoordinatorState::Pressure` が存在する。v12では Pressure Report 中心だったが、Pressure 発生後に「何を止めるのか」が曖昧。

### 目標

Ready → Pressure → RejectLowPriority → RejectMostRequests の段階制御。Admissionへ統合。

### 改修手順

1. **Ready**: 通常運用
2. **Pressure** (slope > 8): `retirePressurePublicationThrottleActive_` 有効化
3. **RejectLowPriority** (pressure継続): timer/crossfade publish を拒否
4. **RejectMostRequests** (critical): bootstrap以外の全publish拒否
5. 回復後は段階的に通常復帰

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/PublicationAdmission.h/.cpp` | Pressureレベル統合 |
| `src/audioengine/AudioEngine.Retire.cpp` | 段階制御ロジック |

---

# Tier B: Executor + 追跡

**目標: Executor完成と出版-退役の因果追跡基盤。**

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` | ウィンドウ監視 + Faulted遷移追加 |
| `src/audioengine/ISREvidenceExporter.cpp` | 違反情報出力 |

---

## P2-3: ObserveToken 拡張 (generation getterのみ)

### 現状 (harden確定 — generationのみ取得可)

```cpp
// GlobalSnapshot.h で利用可能なフィールド:
struct GlobalSnapshot {
    uint64_t generation = 0;  // ★ generation は存在
    // ★ publicationSequenceId なし
    // ★ worldId なし
    // ★ epoch なし
};

// SnapshotCoordinator::observeCurrentRuntime() — ptrのみ設定
ObservedRuntime observeCurrentRuntime(RCUReader& reader) const noexcept {
    ObservedRuntime observed(reader);
    observed.ptr = m_slots.loadCurrent(std::memory_order_acquire);
    return observed;  // ← ptr のみ。metadata一切なし
}
```

**generation** は `GlobalSnapshot::generation` から ObserveToken が ptr 経由で間接参照可能。
**publicationSequenceId / worldId / epoch** は SnapshotCoordinator に存在しない。

### 目標

ObserveToken に generation の簡易 getter を追加する（最小変更）。full metadata (pubSeqId/worldId) は RuntimeReadHandle 側の責務とする。

### 改修手順

1. `ObservedRuntime` に `generation()` 簡易 getter を追加 (`return ptr ? ptr->generation : 0;`)
2. publicationSequenceId/worldId の取得は RuntimeReadHandle への付加を検討（ObserveToken の責務範囲外）

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/ObservedRuntime.h` | generation() getter追加のみ |

---

## P1-4: EpochDomain 直接依存除去

### 現状 (harden確定 — RouterにdrainAll不在)

**`ISRRetireRouter` に `drainAll()` メソッドは存在しない。** (Serena/find_symbol で全メソッド確認済み)

```cpp
// ISRRetireRouter.h — 全メソッド:
// Epoch: snapshotEpoch, publishEpoch, currentEpoch, activeReaderCount
// Reader: registerReaderThread, reserveReaderThread, enterReader, exitReader, minReaderEpoch
// Retire: enqueueRetire(x4), enqueueRetire(x3), tryReclaim, pendingRetireCount
// ★ drainAll() 不在!
// EpochDomain.h:220-223 — drainAll はこちらにのみ存在
```

AudioEngine の `m_epochDomain` 直接呼び出しは **2箇所 (drainAll) のみ**:

| # | ファイル | 呼び出し | 対応方法 |
| --- | --- | --- | --- |
| 1 | `AudioEngine.CtorDtor.cpp:131` | `m_epochDomain.drainAll()` | Router に委譲メソッド追加 |
| 2 | `AudioEngine.Processing.ReleaseResources.cpp:208` | `m_epochDomain.drainAll()` | Router 経由に変更 |

RCUReader初期化 (変更不可) — EpochDomain への参照自体は不可避:

- `AudioEngine.h:3370`: `audioThreadRcuReader { m_epochDomain }`
- `AudioEngine.h:3372`: `messageThreadRcuReader { m_epochDomain }`

コンストラクタ注入:

- `CtorDtor.cpp:21`: `m_coordinator(m_epochDomain)` → `m_coordinator(*m_retireRouter)` に変更可能
- `CtorDtor.cpp:26`: `make_unique<ISRRetireRouter>(m_epochDomain)` — やむを得ない

### 目標

AudioEngine からの EpochDomain 直接操作 (`drainAll`) をゼロにする。

### 改修手順

1. **`ISRRetireRouter` に `drainAll()` 委譲メソッドを追加**（必須、現時点で不在）:

   ```cpp
   void drainAll() noexcept {
       assert(epochDomain_ != nullptr);
       epochDomain_->drainAll();
   }
   ```

2. `m_coordinator(m_epochDomain)` → `m_coordinator(*m_retireRouter)` に変更 (IEpochProvider 互換確認済み)
3. 全 `m_epochDomain.drainAll()` → `m_retireRouter->drainAll()` に変更

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/ISRRetireRouter.h` | drainAll()委譲追加 (**必須**) |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | m_coordinator変更、drainAll変更 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | drainAll変更 |

---

# Tier D: 将来対策

---

## P1-1: RuntimePublicationValidator 実装完成

**最下位優先。** 実運用事故は Authority崩壊 / Shutdown崩壊 / Retire停滞 / Publication経路分裂から発生する。Topology Placeholder は実運用事故原因ではない。

Validator は既に `validatePublication()` で呼ばれている。Placeholder は存在するが、現時点では実害なし。

### 現状

4段階のバリデーションのうち3段階がダミー:

| メソッド | 状態 |
| --- | --- |
| `validateSemanticConsistency()` | ✅ 実装済み (crossfade params check) |
| `validateTopology()` | ❌ `return true; // Placeholder` |
| `validateResources()` | ❌ `return true; // Placeholder` |
| `checkNoConflictingTransitions()` | ❌ `return true; // Placeholder` |

### 改修手順

1. `validateTopology()` の実装
2. `validateResources()` の実装
3. `checkNoConflictingTransitions()` の実装

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/audioengine/RuntimePublicationValidator.cpp` | 全バリデーション実装 |

---

## P3-1: Reader Stuck Detection (複合判定) [Tier A後半]

**実運用事故率は Authority > Reader Stuck > Shutdown の順。Tier A後半に引き上げ。**

### 現状

`ReaderSlot` にカウンタがない。クラッシュ/デッドロック時に Reader が exit せず、reclaim が永久停止する可能性がある。

### 注意: enterCount 単独では検出精度が低い

enterCount増加なし = 正常アイドル と Reader死亡 を区別できない。複合判定が必要。

**推奨: `enterCount` + `currentEpoch` + `minReaderEpoch` + `pendingRetireCount` の複合判定。**

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    std::atomic<uint64_t> enterCount { 0 };  // ★ enter 回数のみカウント（軽量）
    // ★ enterTimestamp は追加しない（audio thread overhead回避）
};
```

### 改修手順

1. `EpochDomain::ReaderSlot` に `std::atomic<uint64_t> enterCount` を追加
2. `enterReader()` 内で `fetchAddAtomic(enterCount, 1, memory_order_relaxed)` のみ
3. `detectStuckReaders()` メソッドを追加: **enterCount停滞 + epoch長時間滞留 + minReaderEpoch停滞 + pendingRetireCount増加 + currentEpoch - readerEpoch > threshold** の複合判定
4. **ReaderFault検出時**: `ReaderFaultRecord` 生成 + Evidence出力 + Warning。**隔離(markReaderFault)は保留** — RCU系でreader隔離はUse After Freeリスクがあるため
5. 診断結果を Evidence に出力
6. 診断結果を Evidence に出力

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/EpochDomain.h` | ReaderSlotにenterCount追加、detectStuckReaders(複合判定)追加 |

---

## P3-3: ReaderSlot 使用状況テレメトリ

ReaderSlot 使用率の診断のみ追加。`ReaderSlotPool` の導入は過剰設計 (現在 11/64 使用で枯渇リスクなし)。

### 改修手順

1. `EpochDomain` に ReaderSlot 使用率の診断メソッドを追加
2. 枯渇が近づいた場合の warning ログ出力
3. Evidence に ReaderSlot 使用状況を出力

---

## P3-2: RuntimeReaderContext 型安全化 (最下位優先)

### 現状

理論上は誤った reader/channel の組み合わせがコンパイル可能。ただし実際の誤用は確認されていない (2箇所の直接構築は共に正しい)。実運用事故を防ぐ効果は非常に小さい。

### 改修手順

1. `AudioReaderContext` / `MessageReaderContext` / `PublicationReaderContext` を個別の型として定義
2. `makeRuntimeReadHandle()` を各型に対応するオーバーロードに分割

### 変更ファイル

| ファイル | 変更内容 |
| --- | --- |
| `src/core/RuntimeReaderContext.h` | 型分離 |

---

# P1-2/3: 保留項目

## ISRRetireRouter 実体化 (P1-2) / RetirePolicy 実装 (P1-3)

**保留 (harden確定 — 3Policyは前方宣言のみで実体ゼロ)。**

```cpp
// ISRRetireRouter.h:26
class DSPRetirePolicy;          // 前方宣言のみ
class SnapshotRetirePolicy;     // 前方宣言のみ
class DeferredRetirePolicy;     // 前方宣言のみ

// ISRRetireRouter.h:143
// [work21 P0-1] Future: delegate to DSPRetirePolicy / SnapshotRetirePolicy / DeferredRetirePolicy
```

- **実際の使用箇所: 0**（参照ゼロ。grep/Serena/find_symbol で確認）
- 現在のルーティング: `ISRRetireRouter::enqueueRetire()` → 直接 `EpochDomain::enqueueRetire()` に委譲
- 実運用要求が明確になるまで本件は保留

---

## 工程表 (Tier S+/S/A/B/C 実運用優先順位 v15)

```text
Tier S+ (Orchestrator前進監視+PublicationLedger) [最優先]:
  P-New2        — Orchestrator Forward Progress (lastProgressTimestamp。executed/reclaimed/retiredで更新)
  P-New3        — Publication Ledger (9段階: submit/build/validate/publish/retire/reclaim+rejected+valifail+execfail)

Tier S (Authority封鎖+State統合):
  P0-2/3 (統合) — Coordinator生成封鎖 + Admission強制 (Token move-only+consume+ShutdownScope+engineInstanceId)
  P0-5          — Authority Audit (直接API監査主: makeRuntimeCoordinator/Coordinator::/acquireWriteAccess/publishWorld)
  RuntimeState  — RuntimePublicationState抽出。CoordinatorとOrchestrator両方が参照。段階的移行

Tier A (可観測性+Reader):
  P1-8          — FailureRecord拡張 (+activeReaderCount+minReaderEpoch+currentEpoch)
  P1-5          — Evidence実データ化 + shutdown_drain_audit.json
  P-New         — Stall Detection + RetireStallWarning (oldestRetireAge主判定)
  P3-1↑         — Reader Stuck Detection (検出+Evidence+Warning。隔離保留)
  P1-7          — Reclaim Monitoring (Detect→Evidence→Warning→Faulted。forceAdvance完全削除)

Tier B (Executor+因果追跡+ShutdownScope):
  P0-1          — Executor完成 + PublishStageResult (Coordinator=実行器)
  P2-2          — DeletionEntry+RetireTimelineRecord ((seqId,gen,worldId)統一)
  ShutdownScope — ShutdownScope追加 (Token move-only+consume+engineInstanceId)

Tier C (運用監視):
  P1-6          — Adaptive Backpressure (Ready→Pressure→RejectLow→RejectMost)
  P0-4          — runtimeStore.observe()直接利用除去

保留 (危険+未確定):
  Recovery全体  — 過剰導入リスク。Detect→Evidence→Faultedが安全
  ReaderIsolate — markReaderFault()はUAF危険。保留
  P1-4, P2-3, P1-1, P3-3, P3-2, P1-2, P1-3, P1-9
```

---

## 検証に使用したツール

| ツール | 用途 |
| --- | --- |
| Serena MCP (find_symbol, find_referencing_symbols) | シンボル定義・依存関係の特定 |
| CodeGraph MCP (analyze_module_structure, reindex) | モジュール構造解析 (12,739 entities) |
| grep/Select-String | 全コードベースの網羅的パターン検索 |
| 直接ファイル読取 | 主要ファイルの実装確認 (publishWorld戻り値, acquireReadToken, DeletionEntry等) |

---

## 参考: 現状の主要ファイル一覧

| ファイル | 役割 | 改修Phase |
| --- | --- | --- |
| `src/core/RuntimePublicationCoordinator.h` | Coordinatorテンプレート | Phase B (PublishOutcome) |
| `src/audioengine/AudioEngine.h` | エンジン本体 | Phase A (全般) |
| `src/audioengine/PublicationExecutor.h/.cpp` | 出版実行 | Phase B |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | 出版オーケストレータ | Phase A/B |
| `src/audioengine/ISREvidenceExporter.h/.cpp` | エビデンス出力 | Phase C |
| `src/audioengine/PublicationAdmission.h/.cpp` | 出版許否判定 | Phase A |
| `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` | ISR版Coordinator | Phase D (P1-9) |
| `src/audioengine/RuntimePublicationValidator.h/.cpp` | 出版検証 | Phase F |
| `src/audioengine/ISRRetireRouter.h` | Retire Router | Phase E (drainAll委譲) |
| `src/core/EpochDomain.h` | Epoch管理 | Phase E/F |
| `src/core/ObservedRuntime.h` | 観測トークン | Phase E |
| `src/core/RuntimeReaderContext.h` | Readerコンテキスト | Phase F |
| `src/core/DeferredDeletionQueue.h` | 遅延削除キュー | Phase C |
| `src/core/RuntimeStore.h` | Runtime Store | Phase A |

---

# 現状と計画の重大ギャップ (v19 最終レビュー反映)

| # | 項目 | v18問題 | v19対応 |
| --- | --- | --- | --- |
| 1 | **StateOwner GodObject化** | onXxx()にTelemetry全部→肥大化 | **StateOwner=State+Ledgerのみ**。TelemetryRecorderを分離 |
| 2 | **CorrelationId 32bit wrap** | (engineInstanceId<<32)\|counter → 42億超で再利用 | **128bit struct**: `CorrelationId {engineInstanceId, localCounter}`。64bit counterでwrap不可 |
| 3 | **ShutdownScope thread固定** | GetCurrentThreadId()→将来変更で壊れる | **shutdownGeneration拘束**: thread非依存。三要素(engineInstanceId+shutdownGeneration+expiration) |
| 4 | **FailureSnapshot 1バケット集中** | 全FailureReason共有→ValidationFailureが独占 | **FailureReason別独立バケット**: 10/min/reason。全種別均等サンプリング |
| 5 | **ForwardProgress 5段階不足** | publishedまで→Retire/Reclaim停滞見逃し | **7段階**: retiredCount/reclaimedCount追加。Retire/Reclaim停滞も特定可能 |
| 6 | **DeferredHealth不在** | Deferred滞留の監査なし | **DeferredHealth新設**: deferredCount+oldestAge+overwriteCount+discardReason |
| 7 | **ProgressRecord drop時間不明** | ドロップ数だけ→いつ欠落したか不明 | **firstProgressDropUs/lastProgressDropUs追加**。Evidenceに欠落時間を出力 |

**達成度**: 現状60〜65% → v19完了後94〜97%。

## 残課題 (実運用到達99%のために)

| # | 項目 | 重要度 | 理由 |
| --- | --- | --- | --- |
| 1 | Orchestrator SPOF 対策 | 低 | 全publishがOrchestrator集中。実運用での破綻確率は低く、監視(Tier A)で検出可 |
| 2 | 自動Recovery (保留) | 低 | Detect→Evidence→Warning→Faultedで十分。自動Recoveryは過剰かつ危険 |
| 3 | ReaderIsolate (markReaderFault) | 危険 | UAFリスク。実運用要求が明確になるまで永久保留推奨 |

**結論**: v19 は Practical Stable ISR Bridge Runtime の全リスク(Authority/StateOwner/Telemetry/Stuck診断/同期保証/Deferred監査)をカバー。
S+/S/A/B/C の全 Tier で設計完了。実装開始可能。
