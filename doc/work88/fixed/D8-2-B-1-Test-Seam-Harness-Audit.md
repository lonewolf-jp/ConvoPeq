# D8-2-B-1 — Test Seam/Harness Audit: Retire Path Disposition Observability & Control

- **実施日**: 2026-08-26
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0**）
- **判定**: **CONDITIONAL PASS** — existing stubs/seams allow D/Q/E/T observation for T1-T8; DSPLifetimeManager requires architectural seam (AudioEngine& concrete dependency); delete-count counter missing.
- **基準**: `src/audioengine/ISRRetireRouter.h:1-427` | `src/core/SnapshotCoordinator.h:1-195` | `src/audioengine/DSPLifetimeManager.h` | `src/core/IEpochProvider.h` | `src/audioengine/RetireQuarantineStore.h:1-236` | `src/tests/RetireGraceSemanticsTests.cpp:1-750`
- **ConvoPeq.md 再確認**: 2026-08-25 22:11 再生成版 — SnapshotCoordinator.h §(76421-76560) と ISRRetireRouter.h §(30778-) はソースと一致。

---

## 1. 背景 — D8-2-B-1 の目的

D102-C3/C4 で所有権 conservedion chain D→Q→E→T を確認した上で、**T1–T10**（retire disposition contract tests）を実装できるテストシームを監査する。

各テストの観測・制御要件:

| Test | Subject | Observable (what to assert) | Controllable (what to stub) |
|---|---|---|---|
| T1 | SnapshotCoordinator | result == Success (D owns) | IEpochProvider.enqueueRetire → true |
| T2 | SnapshotCoordinator | result == QueuePressure (Q owns) | IEpochProvider.enqueueRetire → false; observe ISRRetireRouter.quarantineResidentCount |
| T3 | ISRRetireRouter | result == EmergencyQuarantine (E owns) | ISRRetireRouter's Q full (512) → E path taken; observe emergencyQuarantineResidentCount |
| T4 | ISRRetireRouter | result == TerminalReclaim (T owns) | Q+E full → T path taken; observe terminalReclaimResidentCount |
| T5 | ISRRetireRouter | result == QueueFull (Q rejected) | Q full + E full + T... |
| T6 | ISRRetireRouter | result == Shutdown | enqueueWithRetry called after shutdown flag set |
| T7 | SnapshotCoordinator | ownership conserved (no double-delete) | Counting deleter |
| T8 | ISRRetireRouter | ownership conserved (no double-delete) | Counting deleter |
| T9 | DSPLifetimeManager | delete-count observable | — |
| T10 | SnapshotCoordinator RT boundary | resetFadeStateAndRetireTarget only calls enqueueRetire (D), never quarantine | RT context simulation |

---

## 2. SnapshotCoordinator テストシーム分析

### 2.1 既存シーム

| Element | Location | Type | テスト可能 |
|---|---|---|---|
| `SnapshotCoordinator(IEpochProvider&)` ctor | `SnapshotCoordinator.h:166` | **Public** — inject ISRRetireRouter or stub | ✅ |
| `setRetireSink(ISRRetireRouter*)` | `SnapshotCoordinator.h:135` | **Public** — inject real ISRRetireRouter as quarantine sink | ✅ |
| `startFade()`, `switchImmediate()`, `completeFade()`, `advanceFade()`, `tryCompleteFade()` | `SnapshotCoordinator.h:92-121` | **Public** — invocation entry points | ✅ |
| `finalizeShutdown()`, destructor | `SnapshotCoordinator.h:73-81` | **Public** — shutdown entry points | ✅ |
| `enqueueWithRetry()` static | `SnapshotCoordinator.h:151` | **Private static** — calls `IEpochProvider::enqueueRetire` (bool) | ❌ (indirect control via IEpochProvider stub) |
| `quarantineRetireSink()` | `SnapshotCoordinator.h:145` | **Private** — calls `ISRRetireRouter::quarantineRetire` | ❌ (indirect: control via D-retire failure) |
| `resetFadeStateAndRetireTarget()` | `SnapshotCoordinator.h:138` | **Private** — dead code (no callers, updateFade deleted) | ❌ |
| `RetireGraceSemanticsTestAccess` | `SnapshotCoordinator.h` | ❌ **Does not exist** | ❌ |

### 2.2 ディスポジション制御経路

```
SnapshotCoordinator::startFade / switchImmediate / completeFade / retireCurrentAndTarget
    → enqueueWithRetry (private static, SnapshotCoordinator.h:151)
        → IEpochProvider::enqueueRetire (returns bool)         ← STUB CONTROL POINT
        → IEpochProvider::tryReclaim (returns void)             ← STUB CONTROL POINT
        → IEpochProvider::enqueueRetire (retry)
    → if false → quarantineRetireSink (private)
        → ISRRetireRouter::quarantineRetire                      ← REAL OBJECT (setRetireSink)
        → RetireQuarantineStore::quarantine (mutex + array[512])
```

### 2.3 制御戦略

| Disposition | How to force | Observable |
|---|---|---|
| **D (Success)** | Stub `IEpochProvider::enqueueRetire` returns `true` | `pendingRetireCount() > 0` (D queue), `quarantineResidentCount() == 0` |
| **Q (QueuePressure)** | Stub returns `false` (both attempts) → `quarantineRetireSink` called → ISRRetireRouter real Q (cap 512) | `quarantineResidentCount() > 0` |
| **E (EmergencyQuarantine)** | Fill Q to 512, then stub returns `false` → ISRRetireRouter internally calls `emergencyQuarantine` | `emergencyQuarantineResidentCount() > 0` |
| **T (TerminalReclaim)** | Fill Q (512) + E (512), stub returns `false` → ISRRetireRouter calls `terminalReclaim` (growable) | `terminalReclaimResidentCount() > 0` |
| **Shutdown** | `snapshotCoordinator::~SnapshotCoordinator()` (calls `retireCurrentAndTarget`) | `quarantineOverflowCount() == 0` |

### 2.4 観測可能メソッド

ISRRetireRouter が提供する observability (SnapshotCoordinator はこれを `m_retireSink` 経由で取得):

| Method | ISRRetireRouter.h | 観測可能 |
|---|---|---|
| `quarantineResidentCount()` | :240 | ✅ Q 滞留 |
| `quarantineOverflowCount()` | :242 | ✅ Q full 拒否数 |
| `emergencyQuarantineResidentCount()` | :317 | ✅ E 滞留 |
| `terminalReclaimResidentCount()` | :326 | ✅ T 滞留 |
| `worldReclaimCount()` | :245 | ✅ World-type deleter 実行数 |
| `pendingRetireCount()` | :226 | ✅ D キュー |
| `residentCountAtomic()` | :349 | ✅ Q+E+T atomic 合計 (RT-safe) |
| `overflowCount()` | :250 | ✅ D キュー overflow |

### 2.5 ギャップ

- **SnapshotCoordinator に対する friend class が不存在** — `RetireGraceSemanticsTestAccess` は `ISRRetireRouter` にのみ存在する（下記 3.1 参照）。SnapshotCoordinator の private static `enqueueWithRetry` に直接アクセスできないが、`IEpochProvider` stub の制御により間接的にテスト可能。
- **No existing SnapshotCoordinator test file** — T1/T2/T10-T12 は新規作成必要。
- **delete-count counter 缺落** — 削除回数を直接観測するカウンタは存在しない。World-type のみ `worldReclaimCount` があるが、Generic deleter の delete count は不可観測。テストは counting deleter パターン（下記 4 参照）で代替可能。

### 2.6 RT バウンダリ (resetFadeStateAndRetireTarget)

`resetFadeStateAndRetireTarget()` (`SnapshotCoordinator.h:138`, `SnapshotCoordinator.cpp:81`):

```cpp
void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept {
    GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
    if (target) {
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);  // ← D path only, NO quarantine
    }
    m_fade.resetToIdle();
}
```

- **Dead code**: `updateFade` は 2026-07-28 に dead code として削除 (`SnapshotCoordinator.h:118` コメント参照)。呼び出し元ゼロ。
- **コメント**: `SnapshotCoordinator.h:150` — 「resetFadeStateAndRetireTarget(L67) は RT(updateFade) から呼ばれ得るため除外。」
- **RT 安全性**: `enqueueRetire()` は lock-free（`IEpochProvider` インターフェース経由）。`enqueueRetire` は bool を返す — QueuePressure 時は `false` が返るが、caller は **quarantine せず** (NonRT 操作を回避)。ptr は単に leak する（UAF なし）。
- **T10 検証方針**: SnapshotCoordinator はこのメソッドを public API として露出させるか、friend class でテストアクセスを追加する必要あり。現状は private かつ dead code。

---

## 3. ISRRetireRouter テストシーム分析

### 3.1 既存シーム

| Element | Location | Type | テスト可能 |
|---|---|---|---|
| `ISRRetireRouter(IEpochProvider&, WorldRetirementReferenceObserver*)` ctor | `ISRRetireRouter.h:172` | **Public** — inject stub provider | ✅ |
| `RetireGraceSemanticsTestAccess` | `ISRRetireRouter.h:419-420` | **friend class** — test-only access to private members | ✅ (existing) |
| `enqueueRetire(ptr, deleter, epoch, type)` | `ISRRetireRouter.h:206` | Public — returns `RetireEnqueueResult` | ✅ |
| `enqueueWithRetry(ptr, deleter, epoch, type)` | `ISRRetireRouter.h:214` | Public — returns `RetireEnqueueResult` (D→Q→E→T チェーン) | ✅ |
| `quarantineRetire(...)` | `ISRRetireRouter.h:235` | Public — direct Q injection | ✅ |
| `emergencyQuarantine(...)` | `ISRRetireRouter.h:312` | Public — direct E injection | ✅ |
| `terminalReclaim(...)` | `ISRRetireRouter.h:322` | Public — direct T injection | ✅ |
| Observability: `quarantineResidentCount()`, `emergencyQuarantineResidentCount()`, `terminalReclaimResidentCount()`, `worldReclaimCount()` | `ISRRetireRouter.h:240-245` | Public | ✅ |
| Observability: `drainQuarantineStore()`, `drainEmergencyAndTerminal()` | `ISRRetireRouter.h:387-389` | **Private** | ❌ (friend 経由可) |
| `provider_` member | `ISRRetireRouter.h:395` | Private | friend 経由 ✅ |

### 3.2 RetireGraceSemanticsTestAccess (existing friend class)

`RetireGraceSemanticsTests.cpp` の anonymous namespace:

```cpp
class TestProvider : public convo::IEpochProvider {
    // ...
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override {
        return false;  // always forces QueuePressure → Q path
    }
    // ...tryReclaim, drainAll, pendingRetireCount など
};
```

- `TestProvider` は `enqueueRetire` を `false` で固定 → D キューに入れず即座に Q へ移送。
- **Per-call control 缺落**: `true`/`false` を動的に切り替える仕組み（`std::function` コールバックやカウンタベース）はない → T1 (D path) テストには `TestProvider` を拡張する必要あり。

### 3.3 enqueueWithRetry ディスポジション制御チェーン

`ISRRetireRouter.cpp:200-340` の `enqueueWithRetry()`:

```
Stage 1: provider_->enqueueRetireTyped(ptr, deleter, epoch, type)
    → true  → return Success (D owns)            [T1]
    → false → retry loop (Stage 2)

Stage 2: retry (kMaxRetry=2)
    provider_->tryReclaim()
    drainEmergencyAndTerminal()
    re-enqueue via enqueueRetire → true → return Success (D owns)

Stage 3: result == QueuePressure → m_retireQuarantine.quarantine(ptr, ...)
    → stored → return QueuePressure (Q owns)      [T2]
    → Q full → Stage 4 (EmergencyQuarantine)

Stage 4: m_emergencyQuarantine.quarantine(ptr, ...)
    → stored → return QueuePressure (E owns)      [T3]
    → E full → Stage 5 (TerminalReclaim)

Stage 5: m_terminalReclaim.store(ptr, ...)
    → ALWAYS stored (growable) → return TerminalReclaim (T owns)  [T4]
```

| Disposition | Trigger | Observable |
|---|---|---|
| **Success (D)** | `provider_->enqueueRetireTyped` → true | `pendingRetireCount() > 0`, result == Success |
| **QueuePressure (Q)** | enqueueRetireTyped → false, Q has space | `quarantineResidentCount() > 0`, result == QueuePressure |
| **EmergencyQuarantine (E)** | Q full (512), enqueueRetireTyped → false | `emergencyQuarantineResidentCount() > 0` |
| **TerminalReclaim (T)** | Q+E full, enqueueRetireTyped → false | `terminalReclaimResidentCount() > 0` |

### 3.4 RetireEnqueueResult 列挙体の監査

`ISRAuthorityClass.h:28-34`:

```cpp
enum class RetireEnqueueResult : std::uint8_t {
    Success = 0,
    QueuePressure,
    QueueFull,
    Shutdown,
    TerminalReclaim
};
```

| Value | Returned by enqueueWithRetry? | Path |
|---|---|---|
| `Success` | ✅ Yes | Stage 1 / Stage 2 |
| `QueuePressure` | ✅ Yes | Stage 3 / Q or E accepted |
| `QueueFull` | ❌ **No** (dead code) | `ISRRetireRouter.cpp:339` に `if (result == QueuePressure \|\| result == QueueFull)` があるが、`enqueueRetire` は `QueueFull` を返さないため到達不能 |
| `Shutdown` | ❌ **No** (dead code) | `enqueueRetire` は `Shutdown` を返さない。`enqueueWithRetry` の最後の `return result;` は `QueuePressure` が返る |
| `TerminalReclaim` | ✅ Yes | Stage 5 |

**Finding**: `QueueFull` と `Shutdown` は宣言されているが **現在のコードでは返されない** dead enum values。T5 (QueueFull) と T6 (Shutdown) はこれらを返すコードパスを作成するリファクタリングが必要、または `quarantineRetire` / `emergencyQuarantine` の直接呼び出しで store-full 状況をシミュレートする。

### 3.5 ギャップ

- **Per-call enqueueRetire control**: `TestProvider` が `false` を固定返す。D-path (T1) テストには `enqueueRetire` を `true` に切り替える拡張が必要。
- **Delete-count counter**: `worldReclaimCount()` は World-type のみ。Generic deleter の delete count は不可観測。
- **QueueFull/Shutdown paths**: dead code — これらの disposition をテストするためには、`enqueueWithRetry` 内部で `QueueFull`/`Shutdown` を返すコードパスを追加するか、`quarantineRetire`/`emergencyQuarantine` の直接呼び出しで store-full をシミュレートする必要あり。

---

## 4. DSPLifetimeManager テストシーム分析

### 4.1 既存シーム

| Element | Location | Type | テスト可能 |
|---|---|---|---|
| `DSPLifetimeManager(AudioEngine& engine)` ctor | `DSPLifetimeManager.h` (1st ctor) | Public — engine owns router | ❌ (engine_ concrete) |
| `DSPLifetimeManager(AudioEngine& engine, ISRRetireRouter* router)` ctor | `DSPLifetimeManager.h` (2nd ctor) | Public — inject specific router | ⚠️ (router injectable but ISRRetireRouter concrete) |
| `activate(void* dsp)` | `DSPLifetimeManager.cpp:173` | Public | ❌ (calls engine_.registerDSPHandleForRuntime) |
| `retire(void* dsp)` | `DSPLifetimeManager.cpp:185` | Public | ❌ (calls engine_.retireDSPHandleForRuntime) |
| `retireByHandle(DSPHandle)` | `DSPLifetimeManager.cpp:222` | Public | ❌ |
| `currentRetiringGeneration_` | `DSPLifetimeManager.h` | private | ❌ |

### 4.2 retire() フロー

```cpp
void DSPLifetimeManager::retire(void* dsp) noexcept {
    const bool retired = engine_.retireDSPHandleForRuntime(...);  // ← AudioEngine& (CONCRETE, not stubbable)
    if (!retired) return;                                          // ← early exit, untestable
    const auto result = router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode, ...);  // ISRRetireRouter concrete
    juce::ignoreUnused(result);                                    // ← result IGNORED!
}
```

### 4.3 ギャップ

- **AudioEngine& 具象依存**: `engine_` は `AudioEngine&` (参照型, concrete class)。AudioEngine のインターフェースを抽出する抽象クラスは存在しない。テストには完全な AudioEngine インスタンスが必要。
- **ISRRetireRouter 具象依存**: `router_` は `ISRRetireRouter*` (concrete class pointer)。仮想デストラクタはあるが、モック/サブクラス化用の仮想メソッドはない（`enqueueWithRetry` は非仮想）。
- **結果無視 (T7 violation risk)**: `DSPLifetimeManager::retire()` は `router_->enqueueWithRetry()` の結果を `juce::ignoreUnused(result)` で無視。これは仕様通り（comment: "再移送済みなので何もしない"）だが、テストで disposition を検証するには `router_->enqueueWithRetry()` の戻り値を観測する別途の方法（ISRRetireRouter の observability counters）が必要。

---

## 5. Delete-Count Observability 缺落

### 5.1 Production code

| Component | Delete-count counter | Type |
|---|---|---|
| `RetireQuarantineStore` | `worldReclaimCount_` (only World-type) | ✅ limited |
| `TerminalReclaimAuthority` | `reclaimCount_` (World only, via recordWorldReclaim) | ✅ limited |
| `ISRRetireRouter` | — (delegates to stores) | ❌ |
| `DeferredDeletionQueue` | — | ❌ |
| `SnapshotCoordinator` | — | ❌ |
| `DSPLifetimeManager` | — | ❌ |

### 5.2 Test-only counting deleter pattern

`src/tests/D8_1_WrapperCacheTests.cpp:14-18`:

```cpp
// D8_1_T-5.6: counting deleter
static std::atomic<int> s_deleteCount{0};
static void countingDeleter(void* ptr) noexcept {
    ++s_deleteCount;
    delete static_cast<TestObj*>(ptr);
}
```

- テストは counting deleter を通じて delete count を測定可能だが、**production code に delete-count counter は存在しない**。
- T7/T8 (ownership conservation) は counting deleter で検証可能。
- T9 (DSP delete-count observability) は `AudioEngine::destroyDSPCoreNode` の deleter をカウントするカウンタを追加するか、counting deleter パターンでテストする必要あり。

---

## 6. Existing Test File Inventory

| File | Tests | Key Stubs |
|---|---|---|
| `RetireGraceSemanticsTests.cpp` | `RetireGraceSemanticsTests` | `TestProvider` (IEpochProvider stub, enqueueRetire=false固定) |
| `D8_1_WrapperCacheTests.cpp` | `D8_1_T-5` series | counting deleter (`s_deleteCount`), `TestEpochProvider` stub |
| `TerminalTelemetryContractTests.cpp` | T1-T5.6 | counting deleter, `TestProvider` variant |
| `AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | Integration tests | Full AudioEngine harness |

### 6.1 TestEpochProvider (existing stub in D8_1_WrapperCacheTests.cpp)

```cpp
// D8_1_WrapperCacheTests.cpp — TestEpochProvider
class TestEpochProvider : public convo::IEpochProvider {
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override {
        return true;  // always D-success → T1 pathのみテスト可能
    }
    // ...
};
```

- `TestEpochProvider` は `enqueueRetire` を `true` 固定 — D-path (T1) 専用。
- `TestProvider` (RetireGraceSemanticsTests.cpp) は `false` 固定 — Q-path (T2) 尚安。
- **結諡**: per-call 切り替え可能な `ConfigurableEpochProvider` が必要。

---

## 7. 結論 — テストシーム要件定義

### 7.1 クリア (ready to test)

| Test | Status | 理由 |
|---|---|---|
| **T1** (Snapshot D-success) | ✅ Ready | `TestEpochProvider` (enqueueRetire=true) + SnapshotCoordinator ctor injectable |
| **T2** (Snapshot Q-transfer) | ✅ Ready | `TestProvider` (enqueueRetire=false) + SnapshotCoordinator + setRetireSink(ISRRetireRouter) |
| **T3** (ISRRetireRouter E-escalation) | ✅ Ready | Real ISRRetireRouter + TestProvider(false) + fill Q to 512 |
| **T4** (ISRRetireRouter T-terminal) | ✅ Ready | Real ISRRetireRouter + TestProvider(false) + fill Q+E to 1024 |
| **T7** (Snapshot ownership conservation) | ✅ Ready | Counting deleter + SnapshotCoordinator |
| **T8** (ISRRetireRouter ownership conservation) | ✅ Ready | Counting deleter + ISRRetireRouter |

### 7.2 要シーム追加 (needs new seam)

| Test | Required Change | Location |
|---|---|---|
| **T5** (QueueFull) | Add `quarantineRetire` direct-call + fill Q to trigger store-full return `false` | ISRRetireRouter — already public ✅ |
| **T6** (Shutdown) | Add `Shutdown` return path to `enqueueWithRetry` (currently dead code) | ISRRetireRouter.cpp — **code change needed** |
| **T9** (DSP delete-count) | Either (a) counting deleter wrapping `AudioEngine::destroyDSPCoreNode`, or (b) add delete-count counter to ISRRetireRouter | New test infrastructure |
| **T10** (RT boundary) | Make `resetFadeStateAndRetireTarget` testable (friend class or public exposure) | SnapshotCoordinator — test-only change |
| **Configurable IEpochProvider** | Per-call `enqueueRetire` return control | New test stub |

### 7.3 Per-call controlled IEpochProvider stub — 推奨実装

```cpp
class ConfigurableEpochProvider : public convo::IEpochProvider {
public:
    std::function<bool(void*, void(*)(void*), uint64_t)> enqueueRetireFn;
    std::atomic<int> tryReclaimCount{0};
    std::atomic<int> publishEpochCount{0};
    std::atomic<uint64_t> currentEpoch_{0};
    std::array<uint64_t, 64> readerEpochs_{};

    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override {
        if (enqueueRetireFn) return enqueueRetireFn(ptr, deleter, epoch);
        return enqueueRetireResult_;  // default: true
    }
    void tryReclaim() noexcept override { ++tryReclaimCount; }
    uint64_t publishEpoch() noexcept override { ++publishEpochCount; return ++currentEpoch_; }
    // ... rest of IEpochProvider methods (registerReaderThread, enterReader, etc.)
};
```

この stub により T1-T4, T10 のすべての disposition を per-call 制御可能。

---

## 8. アクションアイテム

1. **D8-2-B-2**: Implement `ConfigurableEpochProvider` test stub in a shared `TestHarness.h` (consolidate `TestProvider` + `TestEpochProvider` pattern). — **code change**
2. **D8-2-B-3**: Add `RetireGraceSemanticsTestAccess` friend to `SnapshotCoordinator` (for T10 RT boundary test). — **test-only change**
3. **D8-2-B-4**: Add `Shutdown` return path to `ISRRetireRouter::enqueueWithRetry` for T6 testability. — **code change (dead code elimination or new path)**
4. **D8-2-B-5**: Verify `RetireEnqueueResult::QueueFull` is dead code and either remove it or implement the path. — **code audit follow-up**
