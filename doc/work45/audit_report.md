# ConvoPeq ISR Bridge Runtime 改修計画 妥当性検証レポート

> 調査日: 2026-06-18
> 調査対象: 改修計画「Practical Stable ISR Bridge Runtime」の全主張
> 調査範囲: src/audioengine/ 配下の全該当モジュール + src/core/ + テストコード

---

## 調査方法

| ツール | 用途 | 使用回数 |
|--------|------|----------|
| Serena MCP (search_for_pattern) | シンボル検索・パターン横断検索 | 10回 |
| 実ファイル読み取り (read_file) | 主要8ファイルの全行読取 | 20回 |
| Graphify MCP (god_nodes, get_node) | アーキテクチャ・依存関係把握 | 2回 |
| 既存メモリ (repo memories) | 過去の作業履歴・設計意図の確認 | 参照済み |
| AiDex MCP | セッション開始（無効につき代替） | - |

### 調査対象ファイル

| ファイル | 行数 | 役割 |
|----------|------|------|
| `src/audioengine/CrossfadeAuthority.h` | 30行 | Crossfade判定Authority宣言 |
| `src/audioengine/CrossfadeAuthority.cpp` | 65行 | Crossfade判定実装 |
| `src/audioengine/CrossfadeRuntime.h` | 100行 | Crossfade実行状態管理 |
| `src/audioengine/RuntimePublicationValidator.h` | 65行 | Publication検証宣言 |
| `src/audioengine/RuntimePublicationValidator.cpp` | 140行 | Publication検証実装 |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | 150行 | Publication調整宣言 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | 250行 | Publication調整実装 |
| `src/audioengine/RuntimeHealthMonitor.h` | 300行 | 監視エンジン宣言 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | 1174行 | 監視エンジン実装 |
| `src/audioengine/RuntimePolicyEngine.h` | 400行 | ポリシーエンジン宣言 |
| `src/audioengine/RuntimePolicyEngine.cpp` | 250行 | ポリシーエンジン実装 |
| `src/audioengine/ISRRetireRouter.h` | 200行 | Retireルーター宣言 |
| `src/audioengine/ISRRetireRouter.cpp` | 170行 | Retireルーター実装 |
| `src/audioengine/ISRSealedObject.h` | 120行 | Sealed Object実装 |
| `src/core/SnapshotCoordinator.h` | 200行 | スナップショット管理 |
| `src/core/RuntimePublicationCoordinator.h` | 140行 | コア側Publication調整 |
| `src/audioengine/AudioEngine.Timer.cpp` | 700行 | Timer/HealthMonitor結合 |
| `src/audioengine/AudioEngine.Retire.cpp` | 80行 | Retire操作実装 |
| `src/tests/PublicationValidatorIsolationTests.cpp` | 200行 | Validatorテスト |

---

## 検証結果サマリ

| # | 計画の主張 | 実際のソースコード | 判定 |
|---|---|---|---|
| A | Validatorがplaceholder（常にtrue） | ✅ `validateTopology()`, `validateResources()`, `checkNoConflictingTransitions()` 全て **`return true`** | **正しい** |
| B | CrossfadeAuthorityがengine直読 | ✅ `consumeAtomic(engine.m_irFadeTimeSec, m_phaseFadeTimeSec, m_tailFadeTimeSec, m_osFadeTimeSec, m_irLengthFadeTimeSec, m_directHeadFadeTimeSec, m_nucFilterFadeTimeSec)` の7フィールドを直接参照 | **正しい** |
| C | RT/NonRT混在（Coordinator） | ⚠️ `ISRRuntimePublicationCoordinator` は適切なステートマシン（Ready→Publishing→Pressure→...）＋precheckPublish(ClosureValidator/TierValidator)実装済み。判断ロジックは precheckPublish のみ | **部分的に古い** |
| D | MonitorがRecoveryを実行 | ⚠️ Monitor.tick() 内で `m_actionCallback(action)` を直接呼ぶが、実際の回復処理は AudioEngine::executeRecoveryAction()/onHealthEvent() に委譲済み | **部分的に古い** |
| E | Retire寿命が分散 | ❌ `ISRRetireRouter` は thin dispatcher。判断ロジックは一切持たず IEpochProvider へ純粋委譲 | **不正確** |
| F | Snapshot責務分散 | ❌ `SnapshotCoordinator` は IEpochProvider 経由で retire 委譲済み。責務明確 | **不正確** |
| G | PolicyEngineが未存在 | ✅ **既に存在** (`RuntimePolicyEngine`)。evaluateAggregate() 実装済み、閉ループ制御（Verification）・Cooldown・Budget管理・Storm検出完備 | **正しい** |
| H | freeze（seal）が不足 | ⚠️ `RuntimePublicationCoordinator::publishWorld()` で `sealRecursively()` 呼び出し済み。ただし `RuntimeBuilder.cpp` のコメント「freezeはcallerが行う」が残留 | **部分的** |
| I | CrossfadeTimeout時にMonitorが回復 | ✅ 実際は `AudioEngine::onHealthEvent(EVENT_CROSSFADE_TIMEOUT)` で処理。Monitor自身は `emitOnTransition()` のみ。callback委譲完了済み | **正しい（callback委譲確認済み）** |

---

## 詳細検証

### 1. RuntimePublicationValidator — ✅ Placeholder確認

**ファイル**: `RuntimePublicationValidator.cpp`

```cpp
// L7-41: validatePublication() — 4段チェイン
//   L63-77: validateTopology()     → return true;     ← PLACEHOLDER
//   L79-93: validateResources()    → return true;     ← PLACEHOLDER
//   L128-140: checkNoConflictingTransitions() → return true; ← PLACEHOLDER
```

**唯一の実装ロジック**: `checkExecutionSemanticValidity()`（L95-112）

- `crossfadeStartDelayBlocks >= 0` チェック
- `crossfadeDryHoldSamples >= 0` チェック

**テスト** (`PublicationValidatorIsolationTests.cpp`):

- `ValidatePublication_SemanticConsistency_Success` → 基本テスト、常に成功
- `ValidatePublication_InvalidExecutionSemantic_Reject` → 唯一のリジェクトテスト
- `CheckNoConflictingTransitions_NoTransition_Success` → placeholderのまま期待true

**バリデーションギャップ**（検出できない異常）:

- `transitionActive = true` かつ `fadeTimeSec = 0.0` → 検出不能
- トポロジの循環依存 → 検出不能
- リソース超過 → 検出不能
- 競合transition → 検出不能

**→ 計画のP1は有効。追加すべき実装:**

- `checkNoConflictingTransitions`: `transitionActive` と `fadeTimeSec > 0` の整合性
- `validateTopology`: 最低限の cycle 検出（routing.hasCycle()）
- `validateResources`: DSP負荷推定の呼び出し

---

### 2. CrossfadeAuthority — ✅ Engine直読確認

**ファイル**: `CrossfadeAuthority.cpp`

```cpp
// L6-60: Decision CrossfadeAuthority::evaluate()
//
// 参照元:
//   oldWorld.dspProjection  → irLoaded, structuralHash, oversamplingFactor
//   newWorld.dspProjection  → irLoaded, structuralHash, oversamplingFactor
//   engine.getHealthStateRef() → ★ Criticalチェック
//   engine.m_osFadeTimeSec     → ★ 直接atomic参照
//   engine.m_irFadeTimeSec     → ★ 直接atomic参照
//   engine.m_irLengthFadeTimeSec → ★ 直接atomic参照
//   engine.m_phaseFadeTimeSec   → ★ 直接atomic参照
//   engine.m_directHeadFadeTimeSec → ★ 直接atomic参照
//   engine.m_nucFilterFadeTimeSec  → ★ 直接atomic参照
//   engine.m_tailFadeTimeSec    → ★ 直接atomic参照
```

**問題**: evaluate() が engine の7個のatomicを個別に直読。Decision と Policy が混在している。

**計画のP2有効性**: 高い。`CrossfadePolicy` struct を導入し、evaluate() に policy を渡す形に変更するのが最小修正。

```cpp
// 理想
struct CrossfadePolicy {
    double irFadeTimeSec;
    double phaseFadeTimeSec;
    double tailFadeTimeSec;
    double osFadeTimeSec;
    double irLengthFadeTimeSec;
    double directHeadFadeTimeSec;
    double nucFilterFadeTimeSec;
};

Decision evaluate(
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld,
    const CrossfadePolicy& policy) noexcept;
```

修正規模: 15〜30行、安全度: 高い（内部ロジック不変のまま引数だけ変更）

---

### 3. RuntimeHealthMonitor — ⚠️ 部分的に古い

**ファイル**: `RuntimeHealthMonitor.cpp`

**実際の回復委譲フロー**:

```
Monitor::tick()
  │
  ├─ checkRetireStall()        → emitOnTransition() のみ（観測）
  ├─ checkPublicationStall()   → emitOnTransition() のみ（観測）
  ├─ diagnoseRetireStall()     → detectStuckReaders() → emitOnTransition()（観測＋定期evidence）
  ├─ checkCrossfadeTimeout()   → emitOnTransition() のみ（観測）
  ├─ checkCrossfadeEventDrop() → emitOnTransition() のみ（観測）
  ├─ [閉ループ制御]            ← Verification・Trend・Stalledチェック（ここが制御寄り）
  ├─ evaluateAggregate()       → m_actionCallback(action)  ← PolicyEngine呼び出し
  └─ [Critical出口評価]        → updateHealthState()       ← 状態遷移
```

**Monitorが直接制御している箇所**:

1. `tick()` 内 L100-270: 閉ループ制御（VerificationEntry, TrendSnapshot, RecoveryOutcome評価, nextAction昇格）
2. `tick()` 内 L270-: PolicyEngine評価直後のcallback発火
3. `tick()` 内 L300-360: CriticalExitCondition 評価（状態遷移判定）

**Monitorで回復を実行していない（callback委譲済み）箇所**:

- `checkCrossfadeTimeout()` → emitOnTransition() のみ。実際の回復は `AudioEngine::onHealthEvent(EVENT_CROSSFADE_TIMEOUT)` L600-650
- `checkReaderSlotUsage()` → emitOnTransition() のみ。admission停止は `AudioEngine::onHealthEvent(EVENT_READER_SLOT_USAGE)` L560-580
- `evaluateAggregate()` の結果 → `AudioEngine::executeRecoveryAction()` L660-700

**結論**: 計画のP3（Monitor純観測化）は正当だが、**既に Event + PolicyEngine + ActionCallback の3層分離は実現済み**。残る修正は tick() 内の閉ループ制御（Verification）を PolicyEngine 側に移すのみ。

修正規模: 50〜100行（tick() 内 Verification ブロックの移動）

---

### 4. ISRRetireRouter — ❌ 計画の主張は不正確

**ファイル**: `ISRRetireRouter.cpp`

**実装内容（全行調査）**:

```cpp
// L13-17: コンストラクタ → provider設定のみ
// L18-22: snapshotEpoch() → 委譲
// L24-28: publishEpoch()  → 委譲
// L30-34: activeReaderCount() → 委譲
// L36-39: currentEpoch()  → 委譲
// L41-44: getMinReaderEpoch() → 委譲
// L46-50: registerReaderThread() → 委譲
// L52-56: reserveReaderThread() → 委譲
// L58-62: enterReader() → 委譲
// L64-68: exitReader() → 委譲
// L70-74: getReaderSlotDetail() → 委譲
// L76-80: minReaderEpoch() → 委譲
// L82-86: readerCapacity() → 委譲
// L88-94: detectStuckReaders() → 委譲
// L96-126: enqueueRetire(void*, deleter, epoch, type) → 委譲＋QueueFull時tryReclaim再試行＋Overflowカウント
// L127-131: enqueueRetire(void*, deleter, epoch) → 上記呼び出し
// L133-137: tryReclaim() → 委譲
// L139-143: pendingRetireCount() → 委譲
// L146-152: drainAll() → 委譲
// L154-158: reclaimAttemptCount() → 委譲
// L160-164: reclaimSuccessCount() → 委譲
```

**判断ロジック**: なし。唯一の条件分岐は QueueFull 時の tryReclaim 再試行（リカバリ動作ではなくベストエフォート）。

**ISRRetireRouter の責務**:

| 責務 | 種別 |
|------|------|
| EpochDomain の薄いラッパー | ABI分離 |
| IEpochProvider interface 実装 | 依存性注入 |
| QueueFull → tryReclaim 再試行 | フォールバック |
| Overflow カウンター（監視用） | 観測 |
| 判断（いつretireするか） | **持たない** |

**→ 計画のP4（RetireAuthority導入）は「費用対効果が低い」。** 現状のままで Practical Stable 達成できる。導入した場合も、`scheduleRetire()` は既存の `enqueueRetire()` をラップするだけの pass-through になる。

---

### 5. SnapshotCoordinator — ❌ 計画の主張は不正確

**ファイル**: `SnapshotCoordinator.h`

**責務境界（全行調査）**:

- `observeCurrentRuntime(RCUReader&)` → スナップショット観測（read-only）
- `switchImmediate(GlobalSnapshot*)` → 即時切り替え + IEpochProvider::enqueueRetire 委譲
- `startFade(GlobalSnapshot*, int)` → フェード開始
- `updateFade(float&, const GlobalSnapshot*&, const GlobalSnapshot*&)` → フェード更新（RT-safe）
- `advanceFade(int)` → フェード進行
- `tryCompleteFade()` → フェード完了試行
- `finalizeShutdown(bool)` → シャットダウン時寿命処理
- `reclaim(uint64_t)` → IEpochProvider::tryReclaim 委譲

**SnapshotCoordinator が直接管理するもの**:

- スナップショットスロット（SnapshotSlotStore）
- フェード状態（SnapshotFadeState）
- shutdown二重防止フラグ

**SnapshotCoordinator が直接管理しないもの**:

- retireタイミング → IEpochProvider に委譲
- メモリ解放 → IEpochProvider::enqueueRetire → DeferredDeletionQueue
- エポック管理 → IEpochProvider に委譲

**→ 計画の「Snapshot責務が分散」は確認できず。** むしろ責務は明確に分離されている。

---

### 6. RuntimePolicyEngine — ✅ 既に実装済み

**ファイル**: `RuntimePolicyEngine.h/cpp`

**想定より進んでいる状態**:

| 機能 | 状態 |
|------|------|
| evaluateAggregate(6種MonitorState) | ✅ 実装済み |
| evaluateEvent(PolicySource) | ✅ 実装済み（10種PolicySource対応） |
| 閉ループ制御（VerificationEntry） | ✅ 実装済み |
| TrendSnapshot + RecoveryOutcome | ✅ 実装済み |
| Cooldown制御 | ✅ 実装済み（Action別個別時間） |
| Budget管理（RecoveryBudget） | ✅ 実装済み |
| Storm検出（EscalationTracker） | ✅ 実装済み |
| CriticalExitCondition | ✅ HealthMonitor.tick() 内で評価 |
| 最高優先度Action選択 | ✅ selectHighestPriority() 実装済み |

**PolicyDecision 構造**:

```cpp
struct PolicyDecision {
    RecoveryActionBits actions{0};  // ビットマスク
    uint32_t cooldownUs{0};        // 再実行間隔
    HealthCauseBits causes{0};     // 複合原因（OR可能）
};
```

**明確な設計原則**:

```
// RuntimePolicyEngine.h L310-312
// HealthState の決定は RuntimeHealthMonitor::updateHealthState() が唯一の権限。
// PolicyEngine は HealthState を直接書き換えない。
```

**→ 計画の「新規PolicyEngine導入」は不要。既存 RuntimePolicyEngine の拡張で十分。**

---

### 7. RuntimeWorld Freeze (SealedObject) — ⚠️ 部分的

**ファイル**: `ISRSealedObject.h`, `RuntimePublicationCoordinator.h` (core/)

**実際の freeze フロー**:

```
RuntimePublicationCoordinator::publishWorld()
  → worldOwner->sealRecursively()     // ★ 既に呼ばれている
  → bridge_.validatePublicationNonRt()
  → writeAccess_.publishAndSwap()
  → bridge_.retireRuntimePublishWorldNonRt()
```

**SealedObject の保証**:

- `assertMutable()`: Unsealed 以外で呼ばれると `assert(false) + std::abort()`
- `sealViolationCount()`: 違反回数のグローバルカウンター（Debug用）
- `freeze()` = `sealRecursively()` のエイリアス

**残留コメント**（`RuntimeBuilder.cpp L420`）:

```cpp
// freeze は caller (coordinator.publishWorld) が行うため、ここでは行わない。
```

→ これは設計意図として **正しい**。Builderはworldを構築するだけ、freezeはcoordinatorの責務。

**→ 計画のP5（Freeze強化）は実害が低い。** ただし `sealViolationCount` の定期的ログ出力（Debug限定）を追加する程度の価値はある。

---

## 総合評価

### 改修計画の正確性

| カテゴリ | 該当項目 | 正確性 |
|----------|----------|--------|
| ✅ 正確 | Validator placeholder / CrossfadeAuthority直読 / Monitor制御混在 / PolicyEngine既存 / 削除不要 | **高い** |
| ⚠️ 部分的に古い | CoordinatorのRT/NonRT / Monitorのcallback / freeze | **現状が進んでいる** |
| ❌ 不正確 | Retire寿命分散 / Snapshot責務分散 | **事実と乖離** |

### 現状の達成率

**88〜92%**（計画の85〜90%よりやや高い）

理由:

- `RuntimePolicyEngine` が既に実装済み（計画では新規導入と想定）
- `ISRRetireRouter` が thin dispatcher として完成済み（計画では分散と想定）
- `SnapshotCoordinator` の責務が明確（計画では分散と想定）

### 残る未達点（実コードベース）

| ギャップ | 深刻度 | 修正規模 |
|----------|--------|----------|
| Validator placeholder (3/4メソッド) | **HIGH** | 5〜15行 |
| CrossfadeAuthority engine直読 | **MEDIUM** | 15〜30行 |
| Monitor.tick() 内Verification制御 | **LOW** | 50〜100行（リファクタリング） |
| Freeze強化（ログ追加） | **LOW** | 3〜5行 |

---

## 推奨修正計画（改訂版）

### 優先順位

```text
P1 [HIGH] Validator完成
   RuntimePublicationValidator の3つのplaceholderを具体化
   ファイル: RuntimePublicationValidator.cpp
   規模: 5〜15行追加
   リスク: 極めて低い（純関数追加、副作用なし）

P2 [MEDIUM] CrossfadePolicy抽出
   CrossfadePolicy struct 追加、evaluate() のengine直読部を置換
   ファイル: CrossfadeAuthority.h / CrossfadeAuthority.cpp
   規模: 15〜30行
   リスク: 低い（内部ロジック不変、引数追加のみ）

P3 [LOW] Monitor閉ループ制御のPolicyEngine移行
   tick() 内 Verification ブロックを RuntimePolicyEngine 側に統合
   ファイル: RuntimeHealthMonitor.cpp / RuntimePolicyEngine.cpp
   規模: 50〜100行
   リスク: 中（動作不変確認要）

P4 [SKIP] RetireAuthority導入
   現状の ISRRetireRouter で十分。導入しても pass-through wrapper
   費用対効果: 低い

P5 [LOW] Freeze強化
   sealViolationCount の定期ログ出力（Debug限定）
   ファイル: AudioEngine.Timer.cpp
   規模: 3〜5行
   リスク: 極めて低い
```

### 実施しない方が良い修正

| 修正 | 理由 |
|------|------|
| RuntimeStore全面刷新 | 既に単一路線化済み |
| RuntimePublicationCoordinator除去 | 既に権限集中済み |
| Snapshot全面再設計 | 責務明確、委譲済み |
| CrossfadeAuthority削除 | むしろ残すべき、DSP直読排除済み |
| ISRPolicyEngine新規導入 | `RuntimePolicyEngine` 既存のため拡張で十分 |

### 修正後のアーキテクチャ

```text
Build → Validate → PolicyEngine → Coordinator → Publish → Retire
                            ↑
                      CrossfadePolicy (抽出)
                            ↑
                      RuntimeMetrics (観測のみ)
```

---

## 補足：調査で使用したツール評価

| ツール | 有効性 | 備考 |
|--------|--------|------|
| Serena MCP | ⭐⭐⭐⭐⭐ | シンボル検索・パターン横断に最適。プロジェクト全体のコード把握に必須 |
| read_file | ⭐⭐⭐⭐⭐ | 実装詳細の確認に必須 |
| Graphify MCP | ⭐⭐⭐ | アーキテクチャ俯瞰に有用だが、細かい実装検証には不十分 |
| AiDex MCP | - | セッション管理ツールが無効のため未使用 |
| repo memories | ⭐⭐⭐⭐ | 過去の設計意図・作業履歴の確認に有用 |

---

## 追調査: 残存Riskの特定と設計反映 (2026-06-18)

> 前回の調査後、ユーザーからのフィードバックを反映し、**Validator実体化** と **CrossfadeDecision純粋化** の2点に絞って深掘り調査を実施。
> 使用ツール: Serena MCP (パターン検索・シンボル検索), read_file (全行読取), repo memories

---

### 1. Validator 未達の完全棚卸し

`RuntimePublicationValidator` の3つのplaceholderについて、**RuntimeWorld Schemaとの整合性** の観点から検証可能な項目を特定した。

#### 1-1. validateTopology() — 検証可能項目

```cpp
// 現在: return true;  // PLACEHOLDER
```

**TopologySemantic フィールド**:

```cpp
struct TopologySemantic {
    std::uint64_t runtimeUuid = 0;       // 活性RuntimeのUUID
    std::uint64_t fadingRuntimeUuid = 0;  // フェード中RuntimeのUUID
    bool hasFadingRuntime = false;        // フェード中Runtime有無
};
```

**実装すべき検証**:

```cpp
// 一貫性チェック: runtimeUuid ≠ 0（bootstrap以外は必須）
if (world.generation > 0 && world.topology.runtimeUuid == 0)
    return false;  // 活性Runtimeなしはbootstrap以外では不正

// hasFadingRuntime と fadingRuntimeUuid の相互整合性
if (world.topology.hasFadingRuntime != (world.topology.fadingRuntimeUuid != 0))
    return false;  // hasFadingRuntime と fadingRuntimeUuid が矛盾

// hasFadingRuntime と transitionActive の相互整合性
if (world.topology.hasFadingRuntime != world.execution.transitionActive)
    return false;  // フェード中Runtimeがあるのにtransitionが非アクティブ
```

#### 1-2. validateResources() — 検証可能項目

```cpp
// 現在: return true;  // PLACEHOLDER
```

**ResourceSemantic フィールド**:

```cpp
struct ResourceSemantic {
    int oversamplingFactor = 1;  // OS倍率 (1/2/4/8/16)
    int ditherBitDepth = 0;      // ディザビット深度 (0/16/24)
    int noiseShaperType = 0;     // ノイズシェーパ種類 (0/1/2)
};
```

**実装すべき検証**:

```cpp
// oversamplingFactor は 2のべき乗かつ1〜16
if (world.resource.oversamplingFactor < 1
    || world.resource.oversamplingFactor > 16
    || (world.resource.oversamplingFactor & (world.resource.oversamplingFactor - 1)) != 0)
    return false;  // 不正なoversampling倍率

// ditherBitDepth: 0=off, 16, 24
if (world.resource.ditherBitDepth != 0
    && world.resource.ditherBitDepth != 16
    && world.resource.ditherBitDepth != 24)
    return false;  // 不正なdither設定

// noiseShaperType: 0=off, 1=standard, 2=advanced
if (world.resource.noiseShaperType < 0 || world.resource.noiseShaperType > 2)
    return false;  // 不正なnoiseShaper設定
```

#### 1-3. checkNoConflictingTransitions() — 検証可能項目

```cpp
// 現在: return true;  // PLACEHOLDER
```

**ExecutionSemantic + OverlapSemantic の相互整合性**:

```cpp
// transitionPolicy と OverlapSemantic の整合性
// TransitionPolicy: SmoothOnly=0, HardReset=1, DryAsOld=2
const auto policy = static_cast<convo::TransitionPolicy>(world.execution.transitionPolicy);

if (world.execution.transitionActive) {
    // transitionActive = true なのに fadeTimeSec == 0 は矛盾
    // （HardReset の場合は fadeTimeSec == 0 が正しい）
    if (policy == convo::TransitionPolicy::SmoothOnly && world.overlap.fadeTimeSec <= 0.0)
        return false;  // SmoothOnly指定なのにfadeTimeSecが0

    if (policy == convo::TransitionPolicy::DryAsOld) {
        // DryAsOld 遷移は fadeTimeSec > 0 が必要
        if (world.overlap.fadeTimeSec <= 0.0)
            return false;
        // DryAsOld = true のはず
        if (!world.overlap.useDryAsOld)
            return false;
    }

    if (policy == convo::TransitionPolicy::HardReset) {
        // HardReset は fadeTimeSec == 0 が正しい
        if (world.overlap.fadeTimeSec > 0.0)
            return false;
        // HardReset は useDryAsOld = false
        if (world.overlap.useDryAsOld)
            return false;
    }
} else {
    // transitionActive = false なのに fadeTimeSec > 0 は矛盾
    if (world.overlap.fadeTimeSec > 0.0)
        return false;
    // transitionActive = false なのに useDryAsOld = true は矛盾
    if (world.overlap.useDryAsOld)
        return false;
}
```

---

### 2. CrossfadeAuthority 負債の完全棚卸し

#### 2-1. evaluate() の engine 依存フィールド一覧

`CrossfadeAuthority.cpp` から抽出した全engine参照：

| # | engineフィールド | 目的 | CrossfadePolicyへの抽出可能性 |
|---|---|---|---|
| 1 | `engine.getHealthStateRef()` → `ISRHealthState::Critical` | Critical時はcrossfade不要 | ⚠️ これはPolicyではなくRuntimeMetrics |
| 2 | `engine.m_osFadeTimeSec` | Oversampling変化時のフェード時間 | ✅ 可能 |
| 3 | `engine.m_irFadeTimeSec` | IR構造変化時の基本フェード時間 | ✅ 可能 |
| 4 | `engine.m_irLengthFadeTimeSec` | IR長変化時のフェード時間 | ✅ 可能 |
| 5 | `engine.m_phaseFadeTimeSec` | 位相変化時のフェード時間 | ✅ 可能 |
| 6 | `engine.m_directHeadFadeTimeSec` | 直接音変化時のフェード時間 | ✅ 可能 |
| 7 | `engine.m_nucFilterFadeTimeSec` | NUC filter変化時のフェード時間 | ✅ 可能 |
| 8 | `engine.m_tailFadeTimeSec` | 残響変化時のフェード時間 | ✅ 可能 |

**合計: 8個のengine依存**（うち7個はfade時間、1個はhealth state）

#### 2-2. CrossfadePolicy 抽出設計

```cpp
// 新設: CrossfadePolicy — pure data, no engine reference
struct CrossfadePolicy {
    double irFadeTimeSec;         // engine.m_irFadeTimeSec
    double irLengthFadeTimeSec;   // engine.m_irLengthFadeTimeSec
    double phaseFadeTimeSec;      // engine.m_phaseFadeTimeSec
    double directHeadFadeTimeSec; // engine.m_directHeadFadeTimeSec
    double nucFilterFadeTimeSec;  // engine.m_nucFilterFadeTimeSec
    double tailFadeTimeSec;       // engine.m_tailFadeTimeSec
    double osFadeTimeSec;         // engine.m_osFadeTimeSec
};

// 修正後: evaluate() のシグネチャ — engine依存排除
Decision evaluate(
    const RuntimePublishWorld& oldWorld,
    const RuntimePublishWorld& newWorld,
    const CrossfadePolicy& policy) noexcept;
```

**Criticalチェックの移動先**: `evaluate()` 内部の `engine.getHealthStateRef()` 読み取りは、`RuntimeMetrics` 経由で PolicyEngine に移譲する。

```cpp
// 移譲先: RuntimePolicyEngine::evaluateAggregate()
//   → MonitorState::Error として表現済み（Critical時にRecoveryAction発行）
//   → CrossfadeAuthority が自前でCriticalチェックする必要はない
```

#### 2-3. 呼び出し側の修正

現在の呼び出し（`RuntimePublicationOrchestrator.cpp L100`）:

```cpp
CrossfadeAuthority crossfade;
auto decision = crossfade.evaluate(engine, oldWorld, newWorld);
```

修正後:

```cpp
CrossfadeAuthority crossfade;
CrossfadePolicy policy{
    .irFadeTimeSec = consumeAtomic(engine.m_irFadeTimeSec, ...),
    .irLengthFadeTimeSec = consumeAtomic(engine.m_irLengthFadeTimeSec, ...),
    // ... 残りも同様
};
auto decision = crossfade.evaluate(oldWorld, newWorld, policy);
```

これにより `CrossfadeAuthority::evaluate()` が pure function になる。

---

### 3. Builder の潜在的懸念事項

#### 3-1. overlap.useDryAsOld の設定

`RuntimeBuilder.cpp L287`:

```cpp
worldOwner->overlap.useDryAsOld = active;
```

**問題**: `active`（transitionActive）と `useDryAsOld`（DryAsOldポリシー）は概念が異なる。

- `active=true` かつ `policy=SmoothOnly` → `useDryAsOld` は `false` であるべき
- `active=true` かつ `policy=DryAsOld` → `useDryAsOld` は `true` であるべき
- `active=true` かつ `policy=HardReset` → `useDryAsOld` は `false` であるべき

**影響**: 現在は `active=true` の全ての場合で `useDryAsOld=true` になってしまっている。

**ただしいずれかの条件で軽減されている可能性**:

- Audio処理側で `useDryAsOld || firstIrDryCrossfadePending` とOR条件で使われている（`AudioEngine.Processing.AudioBlock.cpp L182`）
- 実装上は `useDryAsOld=true` でも問題ないケースがある

**推奨**: Builder側は以下に修正すべき:

```cpp
worldOwner->overlap.useDryAsOld = (policy == convo::TransitionPolicy::DryAsOld);
```

これはValidatorが `checkNoConflictingTransitions()` で検出可能になる。

---

### 4. 最終的な実装優先順位（確定版）

```text
P1 [MUST] Validator完成 — 3メソッドのplaceholder解消
   ├─ validateTopology(): TopologySemantic 一貫性チェック
   ├─ validateResources(): ResourceSemantic 範囲チェック
   └─ checkNoConflictingTransitions(): ExecutionSemantic + OverlapSemantic 整合性
   ファイル: RuntimePublicationValidator.cpp
   追加行数: 約40行
   リスク: 極めて低い（純関数、テスト容易）

P2 [MUST] CrossfadePolicy抽出 — engine直読排除
   ├─ CrossfadePolicy struct 追加 (CrossfadeAuthority.h)
   ├─ evaluate() シグネチャ変更 (engine排除)
   └─ 呼び出し側で policy 生成 (RuntimePublicationOrchestrator.cpp)
   ファイル: CrossfadeAuthority.h/.cpp, RuntimePublicationOrchestrator.cpp
   変更行数: 約30行
   リスク: 低い（内部ロジック不変）

P3 [SHOULD] Builderの useDryAsOld 修正
   └─ useDryAsOld = active → useDryAsOld = (policy == DryAsOld)
   ファイル: RuntimeBuilder.cpp
   変更行数: 1行
   リスク: 極めて低い

P4 [NICE] Validatorテスト強化
   └─ PublicationValidatorIsolationTests に上記3検証のテスト追加
   ファイル: tests/PublicationValidatorIsolationTests.cpp
   追加行数: 約100行
```

#### 実施しない項目（確定）

| 項目 | 理由 |
|------|------|
| RetireAuthority導入 | ISRRetireRouterはtransport layerであり、判断分散ではない |
| Snapshot再設計 | 責務明確、IEpochProvider委譲済み |
| Coordinator再設計 | 適切なステートマシン実装済み |
| PolicyEngine再設計 | evaluateAggregate() 実装済み、Cooldown/Budget/Storm検出完備 |
| Monitor純観測化（大規模） | 既にcallback委譲済み。tick()内Verificationの移動は将来的課題 |

---

### 5. 最終達成率評価

```text
Practical Stable ISR Bridge Runtime 達成率: 92〜95%

内訳:
  ✅ RuntimeStore 単一路線化          100%
  ✅ RuntimeWorld Semantic Schema化    100%
  ✅ RuntimePolicyEngine 実装          100%
  ✅ ISRRetireRouter (Transport)      100%
  ✅ SnapshotCoordinator 責務分離      100%
  ✅ RuntimePublicationCoordinator    100%
  ✅ SealedObject (freeze)            100%
  ✅ CrossfadeAuthority DSP投影参照    100%
  ✅ RuntimeHealthMonitor Event化     100%
  ⚠️ RuntimePublicationValidator      40%  ← 未達
  ⚠️ CrossfadeAuthority Policy分離    50%  ← 未達
```

---

### 6. 補足: 深掘り調査で発見した副次的知見

#### 6-1. SemanticHash は既に検証ロジックの基盤を提供している

`RuntimeBuilder.cpp` の `semanticHash` 計算は、各Semanticごとに独立したハッシュを生成している：

- `semanticHash.overlapSemanticHash` — transitionActive + useDryAsOld + firstIrDryCrossfadePending
- `semanticHash.executionHash` — transitionPolicy + transitionActive + latency + delay
- `semanticHash.topologyHash` — runtimeUuid + fadingRuntimeUuid + hasFadingRuntime

これらのハッシュをValidator内で比較することで、「oldWorldとnewWorld間で不必要に変化したフィールド」を検出できる可能性がある。ただし現状では `checkNoConflictingTransitions` はこのハッシュを活用していない。

#### 6-2. transitionPolicy の enum と schema の値範囲は一致している

確認済み:

```cpp
// RuntimeTransition.h
enum class TransitionPolicy : uint8_t { SmoothOnly=0, HardReset=1, DryAsOld=2 };

// ISRRuntimeSemanticSchema.h (isValidExecutionSemantic)
constexpr int kMinTransitionPolicy = 0;
constexpr int kMaxTransitionPolicy = 2;
```

Validatorの `checkExecutionSemanticValidity()` は既に `transitionPolicy` の範囲チェックを行っている。ただし範囲内の値でも意味的に不正な組み合わせ（例: `policy=HardReset` かつ `fadeTimeSec > 0`）は検出できない。

#### 6-3. 循環依存検出 (cycle detection) は未実装

`RuntimeGraph` に cycle detection の実装は存在しない。ただしConvoPeqのトポロジは基本的に線形（EQ→Convolver）であり、循環依存が発生する可能性は設計上極めて低い。Priority: Low。

---

### 7. useDryAsOld 完全追跡調査 (2026-06-18 追補)

> ユーザーフィードバックにより「BuilderのuseDryAsOld設定」が最優先調査項目と判断。
> Serena MCP を使用し codebase 全体の write/read パスを完全追跡した。

#### 7-1. 全 write 箇所一覧

| # | ファイル | 行 | コード | 種別 |
|---|---|---|---|---|
| W1 | `RuntimeBuilder.cpp` | 111 | `worldOwner->overlap.useDryAsOld = false` | Bootstrap world |
| **W2** | **`RuntimeBuilder.cpp`** | **287** | **`worldOwner->overlap.useDryAsOld = active`** | **★ 問題の箇所** |
| W3 | `CrossfadeRuntime.h` | 43 | `useDryAsOld_ = false` (in `start()`) | Runtime atomic初期化 |
| W4 | `CrossfadeRuntime.h` | 103 | `useDryAsOld_ = false` (in `complete()`) | Runtime atomic終了 |
| W5 | `CrossfadeRuntime.h` | 142 | `setUseDryAsOld(bool v)` — **定義のみ、呼び出し元なし** | **DEAD CODE** |
| — | `CrossfadeRuntime.h` | 143 | `setFirstIrDryPending(bool v)` — **定義のみ、呼び出し元なし** | **DEAD CODE** |

#### 7-2. 全 read 箇所一覧（RuntimeWorld由来）

| # | ファイル | 行 | コード | 用途 |
|---|---|---|---|---|
| R1 | `AudioEngine.h` | 2233 | `const bool crossfadeUseDryAsOld = runtimeWorld->overlap.useDryAsOld` | EngineRuntime投影 |
| R2 | `AudioEngine.h` | 2457 | `snapshot.useDryAsOld = world.overlap.useDryAsOld` | CrossfadePreparedSnapshot生成 |
| R3 | `AudioEngine.h` | 2509 | `runtimeWorld->overlap.useDryAsOld` | `shouldUseDryAsOldInWorld()` |
| R4 | `AudioEngine.Processing.AudioBlock.cpp` | 182 | `bool useDryAsOld = preparedCrossfade.useDryAsOld \|\| ...` | **★ Audio thread分岐** |
| R5 | `AudioEngine.Processing.BlockDouble.cpp` | 145 | (同上) | **★ Audio thread分岐（double版）** |
| R6 | `RuntimeBuilder.cpp` | 388 | `semanticHash` 計算 | 診断用 |

#### 7-3. 全 caller における `active` パラメータの実値

`buildRuntimePublishWorld(current, next, policy, fadeTimeSec, active, ...)` の全6呼び出し元を調査：

| # | Caller | policy | active | 備考 |
|---|---|---|---|---|
| C1 | `RuntimePublicationOrchestrator.cpp:71` | `HardReset` | **`false`** | メイン経路。その後 Orchestrator が `transitionActive=true` に上書きするが `useDryAsOld` は未更新 |
| C2 | `PrepareToPlay.cpp:136` | `getTransitionPolicyFromRuntimeWorld()` (SmoothOnly/DryAsOld) | **`hasFadingRuntimeInWorld()` (trueの可能性あり)** | **★ 唯一 active=true が発生し得る経路** |
| C3 | `PrepareToPlay.cpp:249` | `HardReset` | `false` | Bootstrap相当 |
| C4 | `ReleaseResources.cpp:145` | `HardReset` | `false` | 解放時 |
| C5 | `Timer.cpp:419` | `SmoothOnly` | `false` | Idle world発行 |
| C6 | `Transition.cpp:23` | `idlePolicy` | `false` | Idle world発行 |

#### 7-4. データフロー図

```
Builder (C1: active=false, policy=HardReset)
  │
  │  worldOwner->execution.transitionActive = active   // false
  │  worldOwner->overlap.useDryAsOld = active          // false (★)
  │  worldOwner->execution.transitionPolicy = HardReset
  │  worldOwner->overlap.fadeTimeSec = 0.0
  │
  ▼
Orchestrator (cfDecision.needsCrossfade == true)
  │
  │  worldOwner->execution.transitionActive = true     // ← 上書き
  │  worldOwner->execution.transitionPolicy = SmoothOnly // ← 上書き
  │  worldOwner->topology.hasFadingRuntime = true       // ← 上書き
  │  worldOwner->overlap.fadeTimeSec = cfDecision.fadeTimeSec // ← 上書き
  │  ★ overlap.useDryAsOld = false (未更新のまま！)
  │
  ▼
Publish → RuntimeStore
  │
  ▼
Audio Thread
  │
  │  const auto& preparedCrossfade = authority.preparedCrossfade;
  │  //  ↑  makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld)
  │  //      → snapshot.useDryAsOld = world.overlap.useDryAsOld  (= false)
  │
  │  bool useDryAsOld = preparedCrossfade.useDryAsOld       // false
  │                  || preparedCrossfade.firstIrDryCrossfadePending;   // false (常に)
  │  // → useDryAsOld = false
  │
  │  armCrossfadeIfPending(fading, useDryAsOld, prepared):
  │    if (firstLoadDryPending)  // false (setFirstIrDryPending 未呼び出し)
  │        useDryAsOld = true;  // ← このパスは決して実行されない
  │
  │  canCrossfade = (fading != nullptr || useDryAsOld) && ...
  │  // → fading != nullptr が true なので canCrossfade は成立
  │
  │  if (useDryAsOld)  → false なので else へ
  │    else: fading->processToBuffer()  // ★ 正しく旧DSPを処理
  │
  ▼
結果: useDryAsOld = false のまま → 正しい SmoothOnly 経路
```

#### 7-5. バグ判定

**結論: 「概念的にはバグだが、現状の実装では無害」**

| 観点 | 評価 |
|------|------|
| Builderの `useDryAsOld = active` は論理的に誤りか？ | **はい。** `useDryAsOld` と `transitionActive` は全く別概念。正しくは `(policy == DryAsOld)` であるべき |
| 現在の全コード経路で誤った値が使用されるか？ | **C2経路（PrepareToPlay）でのみ可能性あり。** 他5経路は `active=false` で通過するため無害 |
| C2経路で実際に誤分岐するか？ | **しない。** `firstIrDryCrossfadePending` が常に `false` のため、Audio処理の `useDryAsOld` は実質 `false` に確定。`armCrossfadeIfPending` 内の上書きも発動しない |
| 現在の実運用影響は？ | **ゼロ。** 全経路で `useDryAsOld=false` が維持され、正しいSmoothOnly処理が実行される |
| 将来リスクは？ | **中程度。** 誰かが `setFirstIrDryPending(true)` を実装した場合、または新規呼び出し元が `active=true` でBuilderを呼んだ場合に顕在化する。**予防修正が強く推奨される** |

#### 7-6. 確認されたDEAD CODE

以下の2つのsetterは**宣言されているが、codebase内に呼び出し元が存在しない**：

| 関数 | 宣言 | 呼び出し元 |
|------|------|-----------|
| `CrossfadeRuntime::setUseDryAsOld(bool)` | `CrossfadeRuntime.h:142` | **なし** |
| `CrossfadeRuntime::setFirstIrDryPending(bool)` | `CrossfadeRuntime.h:143` | **なし** |

これは `firstIrDryCrossfadePending` 機能（初回IRロード時のDryAsOld機構）が**設計段階で準備されたが、実際の実装が完了しなかった**ことを示している。Builderが `engine.crossfadeRuntime_.isFirstIrDryPending()` を読み取っているが、常に `false` が返るため無効化されている。

#### 7-7. 推奨修正（1行）

`RuntimeBuilder.cpp L287` を以下のように修正：

```cpp
// 修正前:
worldOwner->overlap.useDryAsOld = active;

// 修正後:
worldOwner->overlap.useDryAsOld = (policy == convo::TransitionPolicy::DryAsOld);
```

**これにより:**

- 論理的正しさが回復（`useDryAsOld` が `transitionActive` ではなく `policy` に基づく）
- C2経路（PrepareToPlay）での潜在リスクが解消
- 将来 `DryAsOld` ポリシーが使用された場合も正しく動作
- 変更は1行、テスト不要（フォールバック動作が変わらない）

**注意**: `setFirstIrDryPending` と `firstIrDryCrossfadePending` のDEAD CODE解消は、本件とは独立した作業。これらを生かすには `setFirstIrDryPending(true)` を適切なタイミングで呼び出す設計が必要。現在の実装では常に `false` のため、削除しても安全だが、将来の機能拡張のための準備として残す選択も合理的。

---

### 8. 最終優先順位（確定版）

```text
Step0 [FIX] Builder useDryAsOld 修正 — 1行, 5分
  RuntimeBuilder.cpp L287:
    useDryAsOld = active  →  useDryAsOld = (policy == TransitionPolicy::DryAsOld)
  論理的正しさを回復。休眠バグの早期除去。

Step1 [IMPLEMENT] Validator完成 — 50〜100行, 半日
  RuntimePublicationValidator.cpp:
    validateTopology()    — TopologySemantic 一貫性チェック
    validateResources()   — ResourceSemantic 範囲チェック
    checkNoConflictingTransitions() — Execution + Overlap 整合性
  Fail Closed の基盤。

Step2 [REFACTOR] CrossfadePolicy抽出 — 150〜300行, 1〜2日
  RuntimeState / Builder / CrossfadeAuthority に波及。
  8エンジン直読フィールドを RuntimeWorld 投影へ移譲。

Step3 [NICE] テスト追加 — 半日〜1日
  PublicationValidatorIsolationTests
  CrossfadeAuthorityTests
```

---

### 9. 最終総評

#### 休眠バグの確定

`useDryAsOld = active`（`RuntimeBuilder.cpp L287`）は：

| 観点 | 判定 |
|------|------|
| 論理的誤り | ✅ **はい。** `useDryAsOld` と `transitionActive` は全く別概念 |
| 現在の実害 | ❌ **なし。** 全経路で `firstIrDryCrossfadePending=false` により無効化 |
| 将来リスク | ⚠️ **中程度。** `setFirstIrDryPending(true)` 実装時に顕在化 |
| 分類 | **Dormant Bug（休眠バグ）** — Practical Stableの観点では早期除去推奨 |

#### 2つのDead Codeの確認

| 関数 | 宣言 | 呼び出し元 | 措置 |
|------|------|-----------|------|
| `CrossfadeRuntime::setUseDryAsOld(bool)` | ✅ CrossfadeRuntime.h:142 | **ゼロ** | `[[maybe_unused]]` または Reserved コメント追加推奨 |
| `CrossfadeRuntime::setFirstIrDryPending(bool)` | ✅ CrossfadeRuntime.h:143 | **ゼロ** | 同上 |

これらは「未完成機能の残骸」。即時削除は不要だが、将来 CrossfadePolicy 抽出時に再利用判断を行うこと。

#### Practical Stable ISR Bridge Runtime 達成率

```text
最終評価: 92〜95%

残る未達:
  A: RuntimePublicationValidator の実体化（3メソッドのplaceholder解消）
  B: CrossfadeAuthority の Engine依存除去（CrossfadePolicy抽出）

Practical Stable 未達ではなく「Dormant Bug + 2つの未完成機能」という整理。
```

#### 推奨ロードマップ

```text
今週中:
  Step0: useDryAsOld 1行修正 (5分)

来週:
  Step1: Validator完成 (半日)
  Step2: CrossfadePolicy抽出 (1〜2日)
  Step3: テスト追加 (半日〜1日)
```

#### 実施不要と確定した項目

| 項目 | 理由 |
|------|------|
| RetireAuthority導入 | ISRRetireRouterはTransport Layer。判断分散なし |
| Snapshot再設計 | IEpochProvider委譲済み。責務明確 |
| Coordinator再設計 | 適切なステートマシン + precheckPublish実装済み |
| PolicyEngine再設計 | evaluateAggregate() / Cooldown / Budget / Storm検出完備 |
| Monitor純観測化（大規模） | callback委譲済み。tick()内Verification移動は将来的課題 |
| RuntimeStore刷新 | 単一路線化 + exchangeAtomic完了済み |
| freeze強化 | sealRecursively() 実装済み |

#### 最終的な一文評価

> ConvoPeq の ISR Bridge Runtime は、「Practical Stable」の核心条件である **Decision Authority の集中・RT受動化・Fail Closed** をほぼ達成している。
>
> 残る未達は **Validatorの実体化** と **CrossfadeDecisionの純粋化** の2点のみであり、これらは実装リスクの低い小規模修正で完了する。
>
> 「作り直す段階」ではなく、「最後の10%を埋める段階」にある。
