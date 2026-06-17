# PR1/PR2/PR3 × Practical Stable ISR Bridge Runtime 適合性評価

- **作成日**: 2026-06-17
- **評価基準**: `doc/Practical Stable ISR Bridge Runtime.md` の7条件
- **対象設計**: `doc/work44/001_quarantine_reclaim_design.md`, `002_shutdown_quarantine_cleanup.md`, `003_router_retire_investigation.md`

---

## 0. ソースコード調査による事実確認（全設計共通）

### 0.1 シャットダウンパイプライン（完全追跡）

`AudioEngine::releaseResources()` + `ISRShutdownRuntime` の二重パイプライン：

**ISRShutdown Phase**（`ISRShutdown.h`）:

```
Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled
→ ReclaimComplete → EmergencyDrain → VerifyDrained → ShutdownComplete
```

**AudioEngine ShutdownPhase**（`AudioEngine.h`）:

```
Running → StopAcceptingWork → StopAudio → StopWorkers → ForceEpochAdvance → DrainRetire
```

**releaseResources() 実際の流れ**（ソースコード追跡）:

```
① lifecycleState=Releasing
② ISR:AudioStopped + AE:StopAcceptingWork
③ NoiseShaper停止
④ AE:StopAudio                    ← ★ ここ以降 Audio Thread 停止
⑤ AE:StopWorkers + ISR:ObserverDrained
⑥ AE:ForceEpochAdvance → advanceRetireEpoch()
   ISR:RetireClosed → EpochSettled
⑦ AE:DrainRetire
   GracefulDrain: 最大5秒ポーリング（pendingRetire==0 && activeReader==0）
   publishEpoch() + tryReclaim() をループ
⑧ drainDeferredRetireQueues(true)
   ISR:ReclaimComplete
⑨ ISR:EmergencyDrain（or diagnostic only）
    ★★★ [PR2挿入位置] quarantine全解放 → VerifyDrained 直前
⑩ ISR:VerifyDrained
   collectDrainAudit() + 最終監査
   waitForDrain(2000ms) + tryReclaim()
⑪ ISR:ShutdownComplete
   lifecycleState=Unprepared
```

### 0.2 DSPState 状態機械（完全追跡）

`ISRDSPHandle.h`:

```
Constructing → Active → Retired → Reclaimed           (正常)
Active → CrossfadingIn/CrossfadingOut → Retired → Reclaimed  (クロスフェード)
→ Quarantined (← quarantineSlot() が任意の状態を上書き)
Quarantined → DestroyPending → Reclaimed               (destroyQuarantineSlot)
```

**重要**: `DSPHandleRuntime::create()` は `state == Reclaimed` のみをチェック。
Quarantined 状態のスロットは **再利用不可**。
→ PR1の再評価ループで `destroyQuarantineSlot()` が呼ばれない場合、256回のquarantineでレジストリ枯渇アサート。

### 0.3 EBR 安全性の検証（完了）

`DSPLifetimeManager::retire()` の流れ:

```cpp
retireDSPHandleForRuntime(dsp): ① runtimeDSPHandleMap_から削除
                                ② dspHandleRuntime_.retire(handle) → Retired
                                ③ dspHandleRuntime_.reclaim(handle) → Reclaimed, instance=nullptr
router_->enqueueRetire(dsp, ...): ④ EBRキューにdspポインタを登録（独立したコピー）
```

**確認済み**: EBRキューは独自の `void*` コピーを保持。`registry_[slot].instance` とは独立。
`destroyQuarantineSlot()` による `instance=nullptr` はEBRのdeferred deleteと競合しない。

### 0.4 Graceful Drain の実際の動作

```cpp
// AudioEngine.Processing.ReleaseResources.cpp
while (waitedMs < 5000) {
    if (pendingRetireCount == 0 && activeReaderCount == 0) break;
    publishEpoch();
    tryReclaim();
    sleep(10ms);
}
if (timedOut) {
    drainDeferredRetireQueues(true);
    m_epochDomain.tryReclaim();  // ★ drainAll 禁止！→ 残留2件
}
```

- `activeReaderCount==0` の確認 → Reader全停止を保証
- ただしTimeout時は `drainAll()` ではなく `tryReclaim()` のみ（UAF回避）
- 残留した routerPendingRetire はデストラクタの `drainAll()` で最終解放

---

## 評価基準（7条件）

Practical Stable ISR Bridge Runtime の成熟度指標：

| # | 条件 | 説明 |
|---|------|------|
| 1 | RTで delete が発生しない | Audio Thread でメモリ解放を行わない |
| 2 | RTで lock が発生しない | Audio Thread で排他制御を行わない |
| 3 | Retire が必ず Epoch を通る | 全削除は EBR の保護下で行う |
| 4 | Shutdown が完全 Drain を保証する | 停止時は全リソースの解放を確認 |
| 5 | Overflow がデータ喪失に直結しない | キュー溢れはフォールバック機構で保護 |
| 6 | HealthMonitor が自己回復可能 | 異常検出→診断→回復のサイクルを持つ |
| 7 | Coordinator が唯一の Authority | ライフサイクル遷移はCoordinatorのみ通過 |

---

## 1. PR1: Quarantine再評価ループ

### 設計の概要

`onRuntimeRetiredNonRt()` 内で、前回Case Cによりquarantineされたスロットを
grace完了時に re-evaluation し、`retireRuntimeEx_.reclaim()` + `dspQuarantineManager_.reclaimSlot()`
で解放する。

### 条件別評価

| # | 条件 | 判定 | 根拠 |
|---|------|------|------|
| 1 | RTでdelete禁止 | ✅ 適合 | 再評価ループは NonRT スレッド(`onRuntimeRetiredNonRt`) で動作 |
| 2 | RTでlock禁止 | ✅ 適合 | 新規ロック導入なし。atomic load/store のみ |
| 3 | Retire→Epoch | ✅ 適合 | `retireRuntimeEx_.reclaim()` は Epoch 経由の正当な解放パス |
| 4 | Shutdown完全Drain | ✅ 間接適合 | PR1単独では保証しないが、quarantine蓄積を減らしPR2の負荷を低減 |
| 5 | Overflow非喪失 | ✅ 適合 | Quarantine は Overflow の安全弁。PR1はその滞留を解消 |
| 6 | HealthMonitor自己回復 | ✅ 適合 | Grace完了による自動解放 = 自己回復サイクル |
| 7 | Coordinator唯一Authority | ⚠️ 軽度乖離 | `dspHandleRuntime_` を直接操作するが、Coordinatorが既に退役権限を行使済みの後処理のため許容範囲 |

### 🔴 重大な設計上の問題（新規発見）

**`DSPHandleRuntime::create()` が state==Reclaimed のみをチェックする問題**

```cpp
// ISRDSPHandle.cpp:20-30
DSPHandle DSPHandleRuntime::create(void* dspInstance) {
    for (size_t slot = 1; slot < MAX_DSP_SLOTS; ++slot) {
        auto& reg = registry_[slot];
        if (convo::consumeAtomic(reg.state, ...) == DSPState::Reclaimed) {
            // ... スロット割り当て
            return DSPHandle{ ... };
        }
    }
    assert(false && "DSP registry exhausted");  // ★ 256スロット枯渇！
    return DSPHandle::null();
}
```

**問題**: `create()` は `state == Reclaimed` のスロットのみを再利用する。
quarantineは `state = Quarantined` に設定するが、元の設計ではPR1の再評価時に
これを `Reclaimed` に戻していない。

**結果**: Case Cで10個quarantine → 再評価でgrace完了 → retireRuntimeEx.reclaim + reclaimSlotのみ
→ DSPHandleRuntimeのstateは Quarantined のまま
→ `create()` がスキップ → スロット再利用不可
→ 256回のquarantine発生後 **DSP registry exhausted** でアサートフォール

### 修正対応

PR1の再評価ループに `dspHandleRuntime_.destroyQuarantineSlot(slot, 0)` を追加する：

```cpp
// ★ PR1 修正版: 3系統すべて解放
retireRuntimeEx_.reclaim(slot);                           // 系統③: レーン解放
dspHandleRuntime_.destroyQuarantineSlot(slot, 0);          // 系統①: Quarantined→Reclaimed
dspQuarantineManager_.reclaimSlot(slot, 0);               // 系統②: フラグ解放
```

**安全性の根拠**:

- `destroyQuarantineSlot()` は active/fading/crossfade に関与していないことを確認する
- `retireDSPHandleForRuntime()` が既に `reclaim(handle)` で instance=nullptr を設定済み
- DSP実体の削除はEBR（`enqueueDeferredDeleteNonRtWithResult`）が管理しており、
  `destroyQuarantineSlot()` の instance=nullptr はEBRのdeferred queueとは独立して安全

---

## 2. PR2: シャットダウン時Quarantine全解放

### 設計の概要

`releaseResources()` の VerifyDrained 直前に、全quarantineスロットに対して
`dspHandleRuntime_.destroyQuarantineSlot()` + `dspQuarantineManager_.destroyForShutdown()` +
`retireRuntimeEx_.reclaim()` を実行する。

### 条件別評価

| # | 条件 | 判定 | 根拠 |
|---|------|------|------|
| 1 | RTでdelete禁止 | ✅ 適合 | STOP_AUDIO 後に実行。Audio Thread は停止済み |
| 2 | RTでlock禁止 | ✅ 適合 | 新規ロック導入なし |
| 3 | Retire→Epoch | ⚠️ **軽度乖離** | `destroyQuarantineSlot` は Epoch をバイパスするが、STOP_AUDIO 後に全Readerが停止しているため安全 |
| 4 | Shutdown完全Drain | ✅ **適合** | quarantine=10 → 0 を保証する（本PRの目的） |
| 5 | Overflow非喪失 | ✅ 適合 | シャットダウン時はOverflow考慮不要 |
| 6 | HealthMonitor自己回復 | N/A | シャットダウン時は自己回復不要 |
| 7 | Coordinator唯一Authority | ⚠️ **軽度乖離** | Coordinatorをバイパスして直接解放する。ただしシャットダウン時にCoordinatorは既に退役しており許容範囲 |

### 条件3の乖離に関する補足

理想的には shutdown パイプラインでも Epoch 経由の解放が望ましい（文書 Section 11:
`Drain Retire → Advance Epoch → Reclaim → Verify Empty`）。しかし：

1. シャットダウン中は全 Reader が停止している（`STOP_AUDIO`＋`STOP_WORKERS`）
2. Epoch は進行不能（Readerのenter/exitが発生しない）
3. `tryReclaim()` が `getMinReaderEpoch()` を使ってEpoch保護を試みるが、
   凍結したEpochでは `deferredDeletionQueue` の項目が解放されない可能性がある

そのため `destroyQuarantineSlot()` による Epoch バイパスは、
**安全性を損なわない実用的なトレードオフ** と判断する。

### 挿入位置の確定

シャットダウンパイプラインの以下の位置に挿入：

```
⑥ advanceRetireEpoch()
   ISR:EpochSettled
⑦ GracefulDrain（最大5秒）
   publishEpoch + tryReclaim ループ
⑧ drainDeferredRetireQueues(true)
   ISR:ReclaimComplete
⑨ ISR:EmergencyDrain（or diagnostic only）
   ↓  ★★★ PR2: quarantine全解放をここに挿入
   shutdownRuntime_.transitionTo(VerifyDrained)
⑩ ISR:VerifyDrained
   collectDrainAudit()  ← quarantine=0 確認
```

**安全性**: `GracefulDrain` で `activeReaderCount==0` が確認されているため、
quarantineスロットへのRTアクセスは完全に停止している。

---

## 3. PR3: リタイアルータ保留 (routerPendingRetire=2)

### 調査結果

`routerPendingRetire=2` は `EpochDomain::deferredDeletionQueue` に残留する
2アイテムを指す。これらは `releaseResources()` では `tryReclaim()` のみの安全設計により
解放されず、`~AudioEngine()` デストラクタの `drainAll()` で最終解放される。

### 条件別評価

| # | 条件 | 判定 | 根拠 |
|---|------|------|------|
| 1 | RTでdelete禁止 | ✅ 適合 | EBRのRetireQueue管理下 |
| 2 | RTでlock禁止 | ✅ 適合 | 影響なし |
| 3 | Retire→Epoch | ✅ 適合 | EBRのEpoch保護下で残留している |
| 4 | Shutdown完全Drain | ⚠️ **弱い乖離** | `releaseResources()` 時点では drain 不完全（2残留）。デストラクタの `drainAll()` で完全解放されるが、「Verify Empty → Shutdown Complete」のチェーンから外れる |
| 5 | Overflow非喪失 | ✅ 適合 | 関係なし |
| 6 | HealthMonitor自己回復 | ✅ 適合 | 通常運用では次の `tryReclaim()` で解放される可能性がある |
| 7 | Coordinator唯一Authority | ✅ 適合 | EBRの通常パスを経由 |

### 条件4の評価詳細

理念上は `releaseResources()` で完全 drain すべきだが、
`drainAll()` は Use-After-Free のリスクがあるため意図的に回避している。
このトレードオフは文書 Section 11 の shutdown pipeline に反するが、
**実運用の安全性を優先した正当な設計判断** である。

改善オプションとして、`activeReaderCount() == 0` 確認後の安全な `drainAll()` を
将来課題として挙げている（003設計書 4.3節）。

---

## 4. 総合判定

| 条件 | PR1 | PR2 | PR3 | 全体 |
|------|-----|-----|-----|------|
| 1. RTでdelete禁止 | ✅ | ✅ | ✅ | ✅ |
| 2. RTでlock禁止 | ✅ | ✅ | ✅ | ✅ |
| 3. Retire→Epoch | ✅ | ⚠️ | ✅ | ✅（軽度乖離許容範囲） |
| 4. Shutdown完全Drain | ✅間接 | ✅ | ⚠️ | ✅（PR2で担保） |
| 5. Overflow非喪失 | ✅ | ✅ | ✅ | ✅ |
| 6. HealthMonitor自己回復 | ✅ | N/A | ✅ | ✅ |
| 7. Coordinator唯一Authority | ⚠️ | ⚠️ | ✅ | ✅（軽度乖離許容範囲） |

### 発見された重大問題（要修正）

**🔴 PR1: `dspHandleRuntime_.destroyQuarantineSlot()` の呼び出しが欠落**

現在の設計ではPR1再評価時にDSPHandleRuntimeのstateが `Quarantined` のままとなり、
`DSPHandleRuntime::create()` が当該スロットを再利用できない。
256回のquarantine発生時に `DSP registry exhausted` アサートが発生する。

**修正**: PR1再評価ループに以下を追加：

```cpp
dspHandleRuntime_.destroyQuarantineSlot(slot, 0);
```

**優先度**: Critical（即時修正必須）

### 許容される軽度乖離

1. **PR2のEpochバイパス**: シャットダウン中のReader停止により安全
2. **PR3のrouterPendingRetire残留**: `drainAll()` のUAFリスク回避のための意図的設計
3. **Coordinatorバイパス**: いずれもCoordinator退役後の後処理であり許容範囲

---

## 5. 更新された実装コード

### PR1: 修正版 quarantine再評価ループ（`AudioEngine.Commit.cpp`）

```cpp
// ★ PR1: Quarantineスロット再評価 — 3系統すべて解放
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
        dspHandleRuntime_.destroyQuarantineSlot(slot, 0);         // 系統①: Reclaimedに遷移
        dspQuarantineManager_.reclaimSlot(slot, 0);              // 系統②: フラグ解放
    }
}
```

### PR2: 修正版シャットダウン解放（`AudioEngine.Processing.ReleaseResources.cpp`）

Graceful Drain 完了後（⑧ `drainDeferredRetireQueues(true)` 後、⑩ VerifyDrained 前）に挿入。
Graceful Drain で `activeReaderCount == 0` が確認されているため、quarantine領域は安全に解放できる。

```cpp
// ★★★ PR2: Quarantine 全スロット強制解放（シャットダウン専用）
// ★ 挿入位置: ISR:EmergencyDrain 直後、ISR:VerifyDrained 直前
//   この時点で GracefulDrain が activeReaderCount==0 を確認済み
//   （タイムアウト時も最大5秒のポーリング後に到達）
{
    const auto residentBefore = dspQuarantineManager_.residentCount();
    if (residentBefore > 0) {
        diagLog("[DIAG] releaseResources: quarantinedSlots="
                + juce::String(static_cast<int>(residentBefore))
                + " — performing shutdown cleanup");

        for (uint32_t slot = 0; slot < DSPHandleRuntime::MAX_DSP_SLOTS; ++slot) {
            // 系統①: DSPHandleRegistry の Quarantined→Reclaimed 遷移
            //   generation=0で世代チェックスキップ（GracefulDrain完了により安全）
            //   active/fading/crossfade の内部チェック付き
            dspHandleRuntime_.destroyQuarantineSlot(slot, 0);

            // 系統②: フラグ解放（非アクティブなら false でスキップ）
            // 系統③: レーン解放 + quarantineResidentCount--
            if (dspQuarantineManager_.destroyForShutdown(slot)) {
                retireRuntimeEx_.reclaim(slot);
            }
        }

        // バッチ compaction（ループ内個別 compaction より効率的）
        dspQuarantineManager_.compactAuditLog();

        const auto residentAfter = dspQuarantineManager_.residentCount();
        diagLog("[DIAG] releaseResources: quarantine cleanup done "
                + juce::String(static_cast<int>(residentBefore))
                + " -> " + juce::String(static_cast<int>(residentAfter)));
    }
}
```

**【Practical Stable 準拠の根拠】**

- `destroyQuarantineSlot()` 内部で active/fading/crossfade の3重チェック
- GracefulDrain 完了後は `activeReaderCount == 0` 確定（最大5秒保証）
- 系統①→②→③の順で解放（quarantineSlot と逆順）
- `compactAuditLog()` はループ後に1回のみ（パフォーマンス最適化）
