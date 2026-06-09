# Practical Stable ISR Bridge Runtime 運用監査レポート v3

**作成日**: 2026-06-09
**検証対象**: 追加6項目 (⑫〜⑰) — 実運用時の障害耐性・進捗保証・監査性
**使用ツール**: Serena MCP, CodeGraph MCP, grep, 直接ファイル読取

---

## 全体評価

| # | 項目 | 判定 | カテゴリ |
| --- | --- | --- | --- |
| ⑫ | Reader Stuck 耐性 | ✅ **正確** | 障害耐性 |
| ⑬ | Retire Queue Pressure 保証 | ✅ **正確** | 進捗保証 |
| ⑭ | Reclaim Progress Guarantee | ✅ **正確** | 進捗保証 |
| ⑮ | Shutdown Drain 完全性 | ✅ **正確** | 障害耐性 |
| ⑯ | Coordinator 直接利用 | ✅ **正確** | 構造面 |
| ⑰ | Telemetry/Evidence System | ✅ **正確** | 監査性 |

**全6件中 6件 正確 (100%)。** これらは構造面（v1/v2）のレビューではカバーされていなかった「運用時の進捗保証・障害耐性・監査性」の重要な観点であり、すべて正当な指摘である。

---

## ⑫ Reader Stuck 耐性が未完成 — ✅ **正確**

### エビデンス

`EpochDomain` の ReaderSlot 構造体（`EpochDomain.h`）:

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    // ★ enterTimestamp なし
    // ★ threadId なし
    // ★ lease 情報なし
};
```

*ReaderSlot が保持している情報:*

- `epoch` — Reader が入った時点のグローバル epoch
- `depth` — リエントラント深さ

*ReaderSlot が保持していない情報:*

- `enterTimestamp` — 入った時刻
- `threadId` — 占有スレッドのID
- `leaseExpiry` — リース期限

### RCUReaderGuard の RAII 依存

`RCUReaderGuard`（`core/RCUReader.h`）:

```cpp
class RCUReaderGuard {
public:
    explicit RCUReaderGuard(RCUReader& r) noexcept : reader(&r) {
        reader->enter();       // ← enterReader() を呼ぶ (depth++)
    }
    ~RCUReaderGuard() noexcept {
        if (reader) reader->exit();  // ← exitReader() を呼ぶ (depth--)
    }
    // ★ crash safety なし
    // ★ timeout なし
    // ★ watchdog なし
};
```

### 問題点

Reader が以下の状態に陥った場合、`reclaim` が永久停止する:

| シナリオ | 影響 | 対策の有無 |
| --- | --- | --- |
| Thread crash (SEGV) while holding guard | depth > 0 固定 → minReaderEpoch が上がらない → reclaim 停止 | ❌ なし |
| Thread deadlock before exitReader | depth > 0 固定 → 同上 | ❌ なし |
| Thread never exits (infinite loop) | depth > 0 固定 → 同上 | ❌ なし |
| Guard の move 後に元のインスタンスから exit | double-free/dangling は防止されているが... | ⚠️ move-after-use は nullptr 化で保護 |

### あるべき姿とのギャップ

```text
Reader lease        ← 未実装
↓
timeout              ← 未実装
↓
stale reader detection ← 未実装
↓
diagnostic           ← 未実装
```

`detectStuckReaders()` はコードベースのどこにも存在しない。

---

## ⑬ Retire Queue Pressure 時の保証が不明確 — ✅ **正確**

### エビデンス

`ISRAuthorityClass.h` の `RetireEnqueueResult`:

```cpp
enum class RetireEnqueueResult : std::uint8_t {
    Success = 0,       // 通常キューに投入成功
    QueuePressure,     // 圧力下でフォールバック経路に受け入れ
    QueueFull,         // 高圧力 (coordinator の action が必要)
    Shutdown           // シャットダウン中で拒否
};
```

### QueuePressure の取扱い

`enqueueDeferredDeleteNonRtWithResult`（`AudioEngine.h:3200-3211`）:

```cpp
if (m_retireRouter->enqueueRetire(...) == RetireEnqueueResult::Success) {
    runtimePublicationBridge_.setRetireBacklogCount(...);
    return RetireEnqueueResult::Success;  // ← Success
}

// [P0-5] enqueue failure -> drop + telemetry.
return RetireEnqueueResult::QueuePressure;  // ← QueuePressure だが何もしない
```

`retireDSP`（`AudioEngine.h:3230-3238`）での QueuePressure 処理:

```cpp
case RetireEnqueueResult::Success:
case RetireEnqueueResult::QueuePressure:  // ← Success と同じ扱い
    return;
```

### 問題点

```text
enqueue
↓
pressure  ← 検出はできる
↓
backpressure  ← 実装なし
↓
telemetry      ← retireQueueDepth_ への atomic store のみ
↓
deferred recovery  ← 実装なし
```

`QueuePressure` 発生時に何が起きるか:

1. `retireQueueDepth_` に atomic store（telemetry）
2. `runtimePublicationBridge_.setRetireBacklogCount()` を呼ぶ
3. それ以外は何もしない

**構造化された Pressure Report は存在しない。** 回復戦略（backpressure / deferred recovery）が体系化されていない。

---

## ⑭ Reclaim Progress Guarantee がない — ✅ **正確**

### エビデンス

Reclaim は `tryReclaim()` の呼び出しに完全依存:

`EpochDomain.h`:

```cpp
void tryReclaim() noexcept override {
    deferredDeletionQueue.reclaim(getMinReaderEpoch());
}
```

`AudioEngine.Retire.cpp` での呼び出し:

```cpp
m_coordinator.reclaim(m_retireRouter->getMinReaderEpoch());
```

`EQProcessor.Core.cpp` での呼び出し:

```cpp
m_epochDomain.tryReclaim();
```

### 問題点

```text
retired object
↓
いつ回収されるか？  ← 未定義
↓
tryReclaim() が呼ばれたタイミングのみ
↓
呼ばれなければ永久滞留
```

| 観点 | 現状 |
| --- | --- |
| 最大滞留時間保証 | ❌ なし |
| backlog telemetry | ⚠️ `pendingRetireCount()` はあるが定期監視なし |
| maxRetireAge 追跡 | ❌ なし |
| 強制 Drain 条件 | ❌ なし（shutdown 時のみ drainAll） |
| 定期 reclaim 保証 | ❌ Timer の頻度依存 |

Timer が高頻度で動作していれば実用上問題にならない可能性はあるが、**保証はない。**

### 既存の監視メトリクス

`AudioEngine.h` には retire 関連の atomic カウンタが多数存在:

```cpp
std::atomic<std::uint64_t> retireQueueDepth_ { 0 };
std::atomic<std::uint64_t> maxRetireDeferralEpochs_ { 256 };
std::atomic<double> maxRetireWallClockMs_ { 5000.0 };
std::atomic<double> reclaimLatency_ { 0.0 };
std::atomic<int> retireHighWatermark_ { 3072 };
std::atomic<int> retireLowWatermark_ { 1024 };
```

メトリクスは存在するが、**これらの値に基づく能動的な回復動作は実装されていない。**

---

## ⑮ Shutdown Drain 完全性が未検証 — ✅ **正確**

### エビデンス

Shutdown シーケンス（`AudioEngine.CtorDtor.cpp`）:

```cpp
AudioEngine::~AudioEngine() {
    // Phase: StopAcceptingWork
    setShutdownPhase(ShutdownPhase::StopAcceptingWork, "~AudioEngine");
    runtimePublicationBridge_.requestShutdown();

    // Phase: ForceEpochAdvance
    setShutdownPhase(ShutdownPhase::ForceEpochAdvance, "~AudioEngine");
    m_retireRouter->publishEpoch();

    // Phase: DrainRetire
    setShutdownPhase(ShutdownPhase::DrainRetire, "~AudioEngine");
    auto coordinator = makeRuntimePublicationCoordinator();
    coordinator.requestShutdownClearNonRt();
    coordinator.clearPublishedRuntimeSnapshotsNonRt();
    drainDeferredRetireQueues(true);        // ← retire queue を強制 Drain
    m_epochDomain.drainAll();               // ← EpochDomain の全キューを強制 Drain
    runtimePublicationBridge_.markShutdownComplete();

    // Phase: Destroy
    setShutdownPhase(ShutdownPhase::Destroy, "~AudioEngine");
}
```

### drainAll() の実装

`EpochDomain.h`:

```cpp
void drainAll() noexcept {
    deferredDeletionQueue.drainAllUnsafe();
}
```

`DeferredDeletionQueue.h` の `drainAllUnsafe()`:

- epoch 判定を無視して全エントリを回収
- `dequeuePos` から順に走査し、`deleter(ptr)` を呼ぶ
- 空になるまでループ

### 問題点

| チェック項目 | 状態 |
| --- | --- |
| drainAll() 呼び出し | ✅ あり |
| drain 後の Queue 空確認 | ❌ なし |
| Reader 全離開確認 | ❌ なし |
| Fallback queue 空確認 | ❌ なし |
| Shutdown Audit Report | ❌ なし |

EvidenceExporter は `shutdown_trace.json` を生成するが:

```cpp
// ISREvidenceExporter.cpp — ハードコードされたテンプレート
"shutdown_trace.json",
"{\"artifact\":\"shutdown_trace.json\",...,\"verified\":true,\"sh1_callbackCount\":0,...}"
```

**静的な JSON テンプレート** であり、実際のランタイム値を反映していない。`"verified":true` は常に true。

`RuntimeDrainAudit.h` は存在するが、これは監査フレームワークであり、shutdown 時に自動実行されるわけではない。

---

## ⑯ RuntimePublicationCoordinator の直接利用経路 — ✅ **正確**

### エビデンス

`AudioEngine.h` のメンバアクセスレベル:

```cpp
class AudioEngine : ... {
    // ... (line 1338: private:)
private:
    // ... private members ...

    // (line 1457: public:)
public:
    // ... public API ...

    // ★★★ PUBLIC メンバ ★★★
    RuntimePublishStore runtimeStore;                                          // line 2715
    iso::audio_engine::RuntimePublicationValidator runtimePublicationValidator_; // line 2716

    // ★★★ PUBLIC インラインメソッド ★★★
    [[nodiscard]] inline RuntimePublicationCoordinator makeRuntimePublicationCoordinator() noexcept
    {
        return RuntimePublicationCoordinatorFactory::create(
            RuntimePublicationBridge { *this, runtimePublicationValidator_ }, runtimeStore);
    }

    // ★★★ PUBLIC — Orchestrator を完全バイパス ★★★
    inline void publishWorld(convo::aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept
    {
        auto coordinator = makeRuntimePublicationCoordinator();
        coordinator.publishWorld(std::move(worldOwner));
    }
};
```

### 直接 publish 経路

```
AudioEngine::publishWorld()  ← public
  → makeRuntimePublicationCoordinator()
    → RuntimePublicationCoordinator::create(bridge, runtimeStore)
      → coordinator.publishWorld()
        → store 書き換え
```

この経路は `RuntimePublicationOrchestrator` を**完全にバイパス**する。

### Orchestrator 経路

```
AudioEngine → Orchestrator::submitPublishRequest()
  → Admission::evaluate()        ← チェックあり
  → Executor::publish()          ← 未完成
  → Transition::onPublishCompleted()
```

Orchestrator 経路には Admission チェックがあるが、直接 publish 経路には**一切のチェックがない。**

### 問題点

```text
Orchestrator のみ publish authority
        ↓
誰でも publishWorld() を呼べる     ← これでは Authority の意味がない
        ↓
runtimeStore が public              ← friend にすべき
makeRuntimePublicationCoordinator() が public  ← friend-only にすべき
publishWorld() が public             ← 削除または private にすべき
```

---

## ⑰ Telemetry / Evidence System が未完成 — ✅ **正確**

### エビデンス

#### EvidenceExporter の現状

`ISREvidenceExporter.cpp` の `exportEvidence()` は以下を生成:

| Artifact | 内容 | 実データ反映 |
| --- | --- | --- |
| `closure_graph.json` | 静的なテンプレート | ❌ 全フィールドが固定値 (`nodeCount:0`等) |
| `mutation_fault_trace.json` | sealViolationCount のみ動的 | ⚠️ 部分的 |
| `hb_graph_trace.json` | 静的テンプレート | ❌ `eventCount:0` 固定 |
| `hb_violation_report.json` | 静的テンプレート | ❌ `violations:[]` 固定 |
| `retire_timeline.json` | 静的テンプレート | ❌ 固定値 |
| `shutdown_trace.json` | 静的テンプレート | ❌ `verified:true` 固定 |
| `retire_latency_report.json` | 静的テンプレート | ❌ `withinThreshold:true` 固定 |
| `payload_tier_report.json` | 静的テンプレート | ❌ 固定値 |

#### lastRejectReason の限定性

`ISRRuntimePublicationCoordinator.cpp`:

```cpp
const char* RuntimePublicationCoordinator::lastRejectReason() const noexcept {
    switch (convo::consumeAtomic(lastRejectCode_, std::memory_order_acquire)) {
    case RejectCode::InvalidClosure:    return "invalid closure graph";
    case RejectCode::InvalidPayloadTier: return "invalid payload tier";
    case RejectCode::None:              return "none";
    }
}
```

返せる理由は **2種類のみ**。PublicationAdmission の拒否理由（StaleGeneration, NotFinalized, Pressure, Shutdown, DeferredFadingActive）は追跡できない。

#### BudgetManager / FailureHandler / IntrospectionConsole

これらは `ISREvidenceExporter.h` で宣言され、`ISREvidenceExporter.cpp` で実装されている:

| クラス | メソッド | 内容 |
|--------|---------|------|
| `BudgetManager::budgetCheck()` | artifact bytes 合計を JSON 出力 | 静的テンプレート |
| `FailureHandler::handleFailure()` | ファイル存在確認ベースの recovery_trace.json | ファイル有無のみ |
| `IntrospectionConsole::introspect()` | runtime_snapshot.json 出力 | 静的テンプレート |

**いずれも実際の Runtime 状態を反映した動的データではない。**

### あるべき姿とのギャップ

```
PublishEvidence    ← 未実装（lastRejectReason のみ）
RetireEvidence    ← 未実装（pendingRetireCount のみ）
ReaderEvidence    ← 未実装
ShutdownEvidence  ← 未実装（shutdown_trace.json は静的なテンプレート）
```

統一された Evidence DTO は存在しない。

---

## 優先順位の再評価

提案された優先順位の妥当性を検証:

| 優先度 | 項目 | 妥当性 | 備考 |
| --- | --- | --- | --- |
| **P0-1** | PublicationExecutor完成 | ✅ 妥当 | 設計上最大のギャップ |
| **P0-2** | Authority一本化 | ✅ 妥当 | ⑯の直接経路も含めて |
| **P0-3** | Deprecated API全廃 | ✅ 妥当 | 8箇所の抑制 |
| **P0-4** | EpochDomain直接依存除去 | ✅ 妥当 | 22箇所の直接参照 |
| **P1-1** | RetireRouter実体化 | ✅ 妥当 | Policy委譲未実装 |
| **P1-2** | RetirePolicy実装 | ✅ 妥当 | 前方宣言のみ |
| **P1-3** | Fallback経路削除 | ✅ 妥当 | EQProcessor + SafeStateSwapper |
| **P1-4** | Reclaim Progress Guarantee | ✅ **妥当 (新規)** | ⑭ 定期reclaim保証の不在 |
| **P1-5** | Shutdown Drain Guarantee | ✅ **妥当 (新規)** | ⑮ drain後監査の不在 |
| **P2-1** | RuntimeReaderContext型安全化 | ✅ 妥当 | 手組み2箇所 |
| **P2-2** | RuntimeReadHandle整理 | ✅ 妥当 | 二重読取構造 |
| **P2-3** | ObserveToken整理 | ✅ 妥当 | 低優先で適切 |
| **P2-4** | ReaderSlotPool化 | ✅ 妥当 | 将来リスク |
| **P3-1** | Reader Stuck Detection | ✅ **妥当 (新規)** | ⑫ タイムスタンプなし |
| **P3-2** | Queue Pressure Telemetry | ✅ **妥当 (新規)** | ⑬ backpressure体系なし |
| **P3-3** | Evidence System統合 | ✅ **妥当 (新規)** | ⑰ 静的テンプレートのみ |

---

## 達成度評価の再検証

### 構造面のみの評価: 85% (v2 の評価)

### 運用監査を含めた総合評価: 70〜75%

構造面の完成度（責務分離・アーキテクチャ整合性・移行完了度）は高いが、**実運用時の「破綻しにくさ」** の観点では以下の未達成事項が累積する:

| 未達成領域 | 影響 |
|-----------|------|
| Reader Stuck 無対策 | Reader crash → reclaim 永久停止 → メモリリーク |
| Queue Pressure 無対策 | キュー溢れ → drop → メモリリーク or データ損失 |
| Reclaim 呼び出し依存 | Timer 停止 → reclaim 永久停止 |
| Shutdown 監査なし | drain 後に残留があっても検出不能 |
| Coordinator 直接経路 | Authority 無視で publish 可能 |
| Evidence 静的テンプレート | 障害解析に使えない |

これらの未達成事項は「構造が正しければ運用でカバーできる」範囲を超えており、**実運用での破綻リスク**に直結する。

---

## v1→v2→v3 の進化

| 観点 | v1 (前回) | v2 (今回11項目) | v3 (追加6項目) |
|------|-----------|----------------|----------------|
| 構造/責務分離 | ✅ 一部不正確 | ✅ 正確 | ✅ さらに正確 |
| Deprecated移行 | ✅ 正確 | ✅ 正確 | ✅ 追加確認 |
| Authority | ⚠️ 大まか | ✅ 正確+図 | ✅ Coordinator 直接経路発見 |
| **運用/障害耐性** | ❌ 未評価 | ❌ 未評価 | ✅ **新規6項目** |
| **進捗保証** | ❌ 未評価 | ❌ 未評価 | ✅ **Reclaim/Shutdown** |
| **監査性** | ❌ 未評価 | ❌ 未評価 | ✅ **Evidence 静的テンプレート** |

---

## 結論

追加6項目 (⑫〜⑰) はすべて正確であり、**Practical Stable ISR Bridge Runtime の完成度評価として不可欠な観点**である。

構造面の完成度が高くても、Reader Stuck で reclaim が止まる設計は「実運用で破綻しやすい」と言わざるを得ない。これら6項目は「構造が整ったからこそ見えてくる次のレイヤーの課題」であり、本来これらが揃って初めて **Practical Stable（実運用で破綻しにくい）** と言える。

**総合推奨**: v2 の11項目 + v3 の6項目 = **17項目** を Practical Stable ISR Bridge Runtime の完成度評価軸とし、P1 に Reclaim Progress Guarantee と Shutdown Drain Guarantee を含める提案は妥当。
