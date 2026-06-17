# PR1: Quarantine 解放パス追加設計

- **作成日**: 2026-06-17
- **優先度**: Critical
- **対象ファイル**: `src/audioengine/AudioEngine.Commit.cpp`
- **関連ファイル**: `src/audioengine/ISRDSPQuarantine.{h,cpp}`, `src/audioengine/ISRRetireRuntimeEx.{h,cpp}`

---

## 1. 問題の根本原因

`AudioEngine::commitRetireWorld()` のリタイア処理は3経路に分岐する：

```
if (exceededDeferralThresholds) {          // Case A: 猶予超過→エスカレーション
    quarantineSlot(...);
    if (canReclaimAfterEscalation(...))
        retireRuntimeEx_.reclaim(pendingSlot);
}                                          // ← 条件不成立: quarantine のみ、解放なし
else if (canTransitionRetirePendingToFree(...)) { // Case B: 正常解放
    retireRuntimeEx_.reclaim(pendingSlot);
}                                          // ✅ 解放される
else {                                     // Case C: 猶予未超過かつ条件未成立
    quarantineSlot(...);
}                                          // ← ★★★ 解放なし！二度と回収されない
```

**Case C** の致命的な欠陥:

1. `quarantineSlot()` で DSPスロットを隔離 (3系統: QuarantineManager + DSPHandleRuntime + RetireRuntimeEx)
2. しかし `reclaim()` は**一切呼ばれない**
3. このスロットは次回 `commitRetireWorld()` の `pendingIntents` に再登場しない（`dequeuePendingRetireIntents()` は新規Intentのみ返す）
4. 結果: **quarantineスロットが永久に滞留** → 10個蓄積（ログ: `quarantine=10 oldestAgeMs=378750`）

### 1.1 ログ上の証拠

```
[ISR][Shutdown] Drain incomplete:
  quarantine=10                        ← ★ 10個滞留
  oldestAgeMs=378750                   ← ★ 最古6.3分前から解放されず
```

`[VERIFY]` ログの経過:

```
tx counters lifecycle(pub/ret/reclaim)=14/0/0
```

`ret=0` のまま。つまり全14世代のパブリッシュに対して **1度もreclaimが成功していない**。
→ `quarantineResidentCount_` が単調増加していることを示す。

---

## 2. ソースコード調査による確定事項

### 2.1 Quarantine 管理の3系統（確認済み）

`AudioEngine::quarantineSlot()` は以下の3系統を1トランザクションとして更新する（`AudioEngine.Threading.cpp:34-48`）：

```
quarantineSlot(slot, generation, reason):
    ├─ Step 1: DSPQuarantineManager::quarantineHandle()  — 監査用フラグ + auditLog
    ├─ Step 2: DSPHandleRuntime::quarantineSlot()         — スロット状態を Quarantined に
    └─ Step 3: RetireRuntimeEx::quarantine()              — レーン設定 + residentCount++
```

解放時は逆順で3系統すべてを解除する必要がある。

### 2.2 既存APIの実装状況（grep/Serena調査で確定）

| API | ファイル | 実装 | 可視性 |
|-----|---------|------|--------|
| `DSPQuarantineManager::reclaimSlot(slot, gen)` | ISRDSPQuarantine | ✅ 実装済み | public |
| `DSPQuarantineManager::isActive(slot)` | — | ❌ **未実装（追加必須）** | — |
| `DSPQuarantineManager::kMaxSlots` | ISRDSPQuarantine.h | ✅ 実装済み | **private→public化必須** |
| `DSPQuarantineManager::compactAuditLog()` | ISRDSPQuarantine | ✅ 実装済み | **private→public化必須** |
| `DSPHandleRuntime::MAX_DSP_SLOTS` | ISRDSPHandle.h | ✅ 実装済み | public ✅ |
| `DSPHandleRuntime::destroyQuarantineSlot(slot, gen)` | ISRDSPHandle.cpp | ✅ **実装済み・未使用** | public ✅ |
| `RetireRuntimeEx::reclaim(slot)` | ISRRetireRuntimeEx.cpp | ✅ 実装済み | public ✅ |
| `RetireRuntimeEx::laneOf(slot)` | ISRRetireRuntimeEx.cpp | ✅ 実装済み | public ✅ |

### 2.3 `dequeuePendingRetireIntents()` の動作（確認済み）

`ISRRetire.cpp:67-87` の実装を確認。MPSCキューを head→tail まで全量ドレインする。
一度消費されたIntentは二度と戻ってこない。
→ **Case Cでquarantineされたスロットが次回の `pendingIntents` に再登場することはない**。

### 2.4 `destroyForShutdown()` の呼び出し元（確認済み）

**grep結果: 呼び出し元ゼロ**。
宣言：`ISRDSPQuarantine.h:54`
定義：`ISRDSPQuarantine.cpp:116`
呼び出し：**なし**（設計書 `doc/work24/refactoring_plan_p7.md:2470` でも「❌ 未実装」と記録済み）

---

## 3. 修正設計（確定版）

### 3.1 修正方針

**方針**: `onRuntimeRetiredNonRt()` の pendingIntents ループ末尾にquarantineスロット再評価ループを追加する。
quarantine済みスロットは「既に隔離された後」であるため、解放条件は **grace完了のみ** で十分。

### 3.2 採用: 案A（Commitループ内quarantine再評価）

#### 再評価条件（確認済み）

```cpp
// 条件: Audio Threadがこのworldの世代より先に進んだ
const bool graceCompleted = retireRuntimeEx_.isGracePeriodCompleted(
    static_cast<uint64_t>(world->generation),  // pendingGeneration
    maxObservedGeneration,
    callbackActiveCount);
```

**Root of Trust**: quarantine済みスロットは既に隔離完了しており、
本来のCase A/Bで要求される `authoritativeOwnershipReleased` や `!hasAnyPendingTransition` は
quarantine状態自体が担保する。grace完了のみで解放安全。

#### 解放シーケンス（確定版 — 要修正）

**⚠️ 重要: 従来設計に重大な欠陥を発見。`destroyQuarantineSlot` の呼び出しが必須。**

`DSPHandleRuntime::create()` の実装を確認した結果：

```cpp
DSPHandle DSPHandleRuntime::create(void* dspInstance) {
    for (size_t slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
        if (state == DSPState::Reclaimed) {  // ← Reclaimed のみチェック
            ...slot assigned...
        }
    }
    assert(false && "DSP registry exhausted");  // ← 256スロット枯渇でアサート！
}
```

`create()` は `state == Reclaimed` のスロットのみ再利用する。
quarantineSlot() は state を `Quarantined` に設定するが、`reclaim()` + `reclaimSlot()` だけでは
`Reclaimed` に戻らない。結果としてスロットが永遠に再利用不能になり、
256回のquarantine蓄積で **DSP registry exhausted** が発生する。

**修正**: 再評価ループでは3系統すべてを解放する。`dspHandleRuntime_.destroyQuarantineSlot()`
を必ず呼ぶ：

```
// 3系統解放（quarantineSlot の逆順）— 確定版
retireRuntimeEx_.reclaim(slot);                          // 系統③: レーン解放 + residentCount--
dspHandleRuntime_.destroyQuarantineSlot(slot, 0);         // 系統①: Quarantined→Reclaimed
dspQuarantineManager_.reclaimSlot(slot, 0);              // 系統②: フラグ解放 (gen=0: 不問)
```

**安全性の根拠**:

- `destroyQuarantineSlot()` は active/fading/crossfade に関与していないことを確認する
- `retireDSPHandleForRuntime()` が既に `reclaim(handle)` で instance=nullptr を設定済み
- DSP実体の削除は EBR (`enqueueDeferredDeleteNonRtWithResult`) が管理しており、
  `destroyQuarantineSlot()` の instance=nullptr はEBRのdeferred queueとは独立して安全

### 3.3 必要となるAPI変更

```cpp
// ISRDSPQuarantine.h: private→publicに移動
public:
    static constexpr size_t kMaxSlots = 256;

    // 新規追加
    bool isActive(uint32_t slot) const noexcept;

    // compactAuditLog を public に移動（PR2バッチ解放用）
    void compactAuditLog() noexcept;

private:
    // kMaxSlots は public に移動したので private には残さない
```

`isActive()` の実装：

```cpp
bool DSPQuarantineManager::isActive(uint32_t slot) const noexcept {
    if (slot >= kMaxSlots) return false;
    return convo::consumeAtomic(quarantineActiveFlags_[slot], std::memory_order_acquire);
}
```

### 3.4 `reclaimSlot()` generation=0 対応

```cpp
void DSPQuarantineManager::reclaimSlot(uint32_t slot, uint64_t generation) {
    if (slot >= kMaxSlots) return;
    bool found = false;
    for (auto& entry : auditLog_) {
        if (entry.slot == slot && !entry.resolved) {
            if (generation != 0 && entry.generation != generation) {
                return;  // generation不一致→新しい隔離を誤消し防止
            }
            entry.resolved = true;
            found = true;
            break;
        }
    }
    if (!found) return;
    convo::publishAtomic(quarantineActiveFlags_[slot], false, std::memory_order_release);
    compactAuditLog();
}
```

### 3.5 `AudioEngine.Commit.cpp` の修正

`onRuntimeRetiredNonRt()` の pendingIntents ループ直後：

```cpp
// ★ PR1: Quarantineスロット再評価
{
    const auto maxObservedGeneration = convo::consumeAtomic(
        youngestObservedGeneration_, std::memory_order_acquire);
    const auto callbackActiveCount = convo::consumeAtomic(
        rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire);

    for (uint32_t slot = 0; slot < DSPHandleRuntime::MAX_DSP_SLOTS; ++slot) {
        if (!dspQuarantineManager_.isActive(slot))
            continue;
        if (retireRuntimeEx_.laneOf(slot) != convo::isr::RetireLane::Quarantine)
            continue;

        const bool graceCompleted = retireRuntimeEx_.isGracePeriodCompleted(
            static_cast<uint64_t>(world->generation),
            maxObservedGeneration,
            callbackActiveCount);
        if (!graceCompleted)
            continue;

        // 3系統解放（quarantineSlot の逆順）
        retireRuntimeEx_.reclaim(slot);                          // 系統③: レーン解放
        dspHandleRuntime_.destroyQuarantineSlot(slot, 0);        // 系統①: Reclaimedに遷移
        dspQuarantineManager_.reclaimSlot(slot, 0);              // 系統②: フラグ解放
    }
}
```

> **Note**: `destroyQuarantineSlot(slot, 0)` は `generation=0` で世代チェックをスキップする。
> 再評価時点では正確な世代が不明だが、quarantineフラグとlaneによる二重チェックで安全。
> `destroyQuarantineSlot` の内部で active/fading/crossfade の安全確認も行われる。

---

## 3. 実装詳細

### 3.1 `AudioEngine.Commit.cpp` の修正

`commitRetireWorld()` 内、既存の `pendingIntents` ループ直後に以下を挿入：

```cpp
// ★ PR1: Quarantineスロット再評価 — 前回Case Cで隔離されたスロットの解放条件を再確認
{
    const auto* currentPublished = RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore);
    const bool authoritativeOwnershipReleased = (currentPublished != world);
    const auto maxObservedGeneration = convo::consumeAtomic(youngestObservedGeneration_, std::memory_order_acquire);
    const auto callbackActiveCount = convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire);
    const bool hasAnyPendingTransition = world->topology.hasFadingRuntime || !pendingIntents.empty();

    for (uint32_t slot = 0; slot < convo::isr::DSPQuarantineManager::kMaxSlots; ++slot) {
        if (!dspQuarantineManager_.isActive(slot))
            continue;

        const auto lane = retireRuntimeEx_.laneOf(slot);
        if (lane != convo::isr::RetireLane::Quarantine)
            continue;

        // 再評価: Case B と同じ条件
        // graceCompleted は maxObservedGeneration > worldGeneration で判定
        // quarantineスロットの generation は現在の world->generation より古い
        const bool graceCompleted = retireRuntimeEx_.isGracePeriodCompleted(
            static_cast<uint64_t>(world->generation),
            maxObservedGeneration,
            callbackActiveCount);
        const bool pendingIntentOwned = true;  // quarantine中はIntent所有確定
        // authoritativeOwnershipReleased: 現在のPublishedとworldが不一致なら解放済み
        if (retireRuntimeEx_.canTransitionRetirePendingToFree(
                graceCompleted, pendingIntentOwned, authoritativeOwnershipReleased)) {
            retireRuntimeEx_.reclaim(slot);
            // QuarantineManager側のフラグも解放 (generation=0: generation不問)
            dspQuarantineManager_.reclaimSlot(slot, 0);
        }
    }
}
```

### 3.2 `ISRDSPQuarantine.h` の修正

```cpp
// kMaxSlots を public に (enum/constexprとして)
static constexpr size_t kMaxSlots = 256;

// isActive() 追加
bool isActive(uint32_t slot) const noexcept
{
    if (slot >= kMaxSlots) return false;
    return convo::consumeAtomic(quarantineActiveFlags_[slot], std::memory_order_acquire);
}
```

### 3.3 `ISRDSPQuarantine.cpp` の修正

`reclaimSlot()` の generation一致チェックを緩和するパラメータ追加：

```cpp
// 現在: generation一致必須
void DSPQuarantineManager::reclaimSlot(uint32_t slot, uint64_t generation)
{
    // ★ PR1: generation=0 の場合は generation チェックをスキップ
    // (commitRetireWorld再評価時は正確なgenerationを保持していないため)
    if (slot >= kMaxSlots) return;

    bool found = false;
    for (auto& entry : auditLog_) {
        if (entry.slot == slot && !entry.resolved) {
            if (generation != 0 && entry.generation != generation) {
                return;  // generation不一致→新しい隔離を誤消し防止
            }
            entry.resolved = true;
            found = true;
            break;
        }
    }
    if (!found) return;

    convo::publishAtomic(quarantineActiveFlags_[slot], false, std::memory_order_release);
    compactAuditLog();
}
```

---

## 4. リスクとトレードオフ

| リスク | 影響 | 対策 |
|--------|------|------|
| quarantine再評価ループでCPU負荷 | 256スロット×atomic loadのみで軽微 | 必要に応じてactiveスロット数のキャッシュ追加 |
| generation=0での誤解放 | 新しく隔離されたスロットを誤って解放 | `isActive()` と `laneOf()` のダブルチェックで防御 |
| 競合状態 | RetireRuntimeEx::reclaimとQuarantineManager::reclaimSlotで不整合 | reclaimSlotをreclaimの前に呼ぶ（先にQuarantineManagerのフラグを確認） |

---

## 5. 検証方法

1. **単体テスト**: `DSPQuarantineManager` にquarantine→条件変更→reclaimのシナリオテスト追加
2. **動作確認**: 変更前 `quarantine=10` → 変更後 `quarantine=0` をログで確認
3. **ストレステスト**: 長時間動作でquarantine数が単調増加しないことを確認
4. **退行テスト**: 既存のリタイア/パブリッシュの単体テストが全てPASSすること
