# ISR_BUG.md レビュー検証レポート

**検証日**: 2026-06-27
**検証方法**: ソースコード全ファイル調査（AiDex MCP, Serena MCP, grep/Select-String 併用）

---

## ① Overflowしても失われない — ✅ **レビュー正しい**

### レビューの主張
> Overflow がデータ喪失に直結しない が要求されているが、RetireIntent Queue と Fallback Queue の両方が満杯になると Intent を破棄する。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/ISRRetire.cpp:33-47` — MPSC Queue 満杯 → Fallback Queue。Fallback も満杯 → `droppedIntentCount_++` で**Intentを破棄**
- `src/core/DeferredRetireFallbackQueue.h:39` — `if (queue_.size() >= kFallbackHardLimit) return false;` — **強制ドロップ**
- `src/audioengine/ISRDSPQuarantine.h/cpp` — `DSPQuarantineManager` は存在するが、Overflow時の隔離先としては使われていない

**補足**: `DSPQuarantineManager` は既に実装済みだが、Overflow パスからの移送は行われていない。レビューの「Quarantineへの移送」提案は妥当な改善案である。

---

## ② HealthMonitorが自己回復まで到達していない — ❌ **レビューは不正確（outdated）**

### レビューの主張
> 現在の RuntimeHealthMonitor はほぼ Detect → Callback であり、Recover → Verify まで到達していない。

### 検証結果: 不正確 — **閉ループ制御は既に実装済み**

**該当コード**:
- `src/audioengine/RuntimeHealthMonitor.cpp:tick()` 内:
  - `computeTrend()` で Recovered/Improving/Stalled/Worsening を判定
  - `nextAction()` で Ladder 昇格（Throttle→Recover→Restore→Safe→Critical）
  - `CriticalExitCondition` で60秒安定確認後のCritical出口評価
  - `VerificationState::PendingVerification` でVerify完了を確認
- `src/audioengine/RuntimePolicyEngine.h`:
  - 6段階のRecoveryAction階層（Observe/Throttle/Recover/Restore/Safe/Critical）
  - `EscalationTracker` によるStorm検出
  - `BudgetManager` による予算管理

**補足**: レビュー作成時点よりも後の実装（work37/work39 Phase）で閉ループ制御が追加されている。ただし、個別のReader Stuckに対する直接的なForceReclaim等はPolicyEngine委譲であり、その点はレビューの懸念が一部妥当。

---

## ③ Shutdown完全Drain保証が未達成 — ✅ **レビュー正しい**

### レビューの主張
> Shutdown 完了条件に pendingPublication=0, pendingRetire=0, activeReader=0, deferredPublish=0, quarantine=0 が含まれていない。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/RuntimeDrainAudit.h:26-27` — `quarantineResident` は明示的に**「監査のみ（完了条件にしない）」**
- `isAllZero()` は `pendingPublication`, `pendingRetire`, `activeCrossfadeCount`, `deferredPublish`, `routerPendingRetire` のみチェックし、quarantine はチェックしない
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:394` — `"Drain complete but quarantine residents remain"` — quarantine があっても Shutdown 継続

**補足**: `getPrimaryBlockingReason()` には `QuarantineResident` が列挙されているが、実際の完了判定には使われていない。レビューの指摘は正確。

---

## ④ Reader Stuckは検出できるが解消権限がない — ✅ **レビュー正しい**

### レビューの主張
> Reader Stuck診断は高度だが、どう解消するかが実装されていない。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/RuntimeHealthMonitor.cpp:diagnoseRetireStall()` — `detectStuckReaders(10)` で検出、`ownerTag`/`ownerThreadId`/`residencyTimeUs` まで取得
- しかし**Readerを強制解消するコードは存在しない**
- `RetireBlockerSnapshot` は `src/core/IEpochProvider.h:27` のコメントのみで未実装
- RecoveryAction はシステムレベルの指標（pendingRetire等）に基づき、特定Readerを対象としない

**補足**: 検出は高度だが解消手段がない。レビューの指摘は正確。

---

## ⑤ RuntimeWorld Immutable が100%ではない — ✅ **レビュー正しい**

### レビューの主張
> 設計上は RuntimeWorld immutable だが、実際には setHealthStateRef 等の外部参照注入が Build 時に残っている。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/RuntimeBuilder.h:32` — `setHealthStateRef()` で**外部参照注入が存在**
- `src/audioengine/PublicationAdmission.h:44` — 同様に `setHealthStateRef()`
- `src/audioengine/AudioEngine.h` の `kFieldDescriptors` — 全フィールドが `MutabilityClass::MutablePrePublish`
- `dspProjection` / `resource` は Build 過程で段階的に設定
- `FrozenRuntimeWorld` クラスは**存在しない**（コードベース全体で0件）

**補足**: 設計意図は immutable-after-publish だが、実装は pre-publish mutation を許容している。レビューの指摘は正確。

---

## ⑥ Retire Authority はほぼ達成だが完全ではない — ✅ **レビュー正しい**

### レビューの主張
> DSPLifetimeManager が router_->enqueueRetire(...) を直接呼んでおり、Coordinator 層が欠如している。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/DSPLifetimeManager.h:72` — `router_->enqueueRetire(...)` を**直接呼び出し**
- `RuntimeRetireCoordinator` クラスは**存在しない**（コードベース全体で0件）
- 現在のチェーン: `DSPLifetimeManager → ISRRetireRouter → EpochDomain`

**補足**: 理想チェーン `DSPLifetimeManager → RetireCoordinator → ISRRetireRouter → EpochDomain` の Coordinator 層が欠如している。レビューの指摘は正確。

---

## ⑦ Shutdown Authority に Quarantine が含まれていない — ✅ **レビュー正しい**

### レビューの主張
> RuntimeDrainAudit は quarantineResident を監査のみ扱いにしており、Shutdown成功条件に含めるべき。

### 検証結果: 正確

**該当コード**:
- `src/audioengine/RuntimeDrainAudit.h:26-27` — `quarantineResident` は「監査のみ」
- `getPrimaryBlockingReason()` には `QuarantineResident` が列挙されているが、`isAllZero()` はチェックしない
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:267-290` — Shutdown時に quarantine cleanup は実行されるが、blocker にはしない

**補足**: レビューの「QuarantineResident を ShutdownBlocker に昇格すべき」は妥当な改善案である。

---

## 総合評価

| # | 項目 | レビュー評価 | 検証結果 | 備考 |
|---|------|------------|---------|------|
| ① | Overflowデータ喪失 | 未達成 | ✅ 正しい | 実際にドロップ発生。Quarantine移送は未実装 |
| ② | HealthMonitor自己回復 | 未達成 | ❌ 不正確 | 閉ループ制御・PolicyEngine・CriticalExit は既に実装済み |
| ③ | Shutdown完全Drain | 未達成 | ✅ 正しい | quarantine が blocker でない |
| ④ | Reader Stuck解消権限 | 未達成 | ✅ 正しい | 検出のみで解消手段なし |
| ⑤ | RuntimeWorld Immutable | 不完全 | ✅ 正しい | pre-publish mutation が存在。FrozenRuntimeWorld 未実装 |
| ⑥ | Retire Authority一元化 | 不完全 | ✅ 正しい | Coordinator層が欠如 |
| ⑦ | Shutdown+Quarantine | 未達成 | ✅ 正しい | audit-only で blocker 非対応 |

**全7項目中6項目が正確、1項目が不正確**。

②のHealthMonitor自己回復については、レビュー作成時点よりも後の実装（work37/work39 Phase）で閉ループ制御・PolicyEngine・CriticalExitCondition が追加されており、レビューの評価は outdated である。ただし、個別Readerに対する直接的なForceReclaim等は未実装であり、その点は引き続き改善余地がある。
