# PR2: シャットダウン時 Quarantine 全解放設計

- **作成日**: 2026-06-17
- **優先度**: High
- **対象ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- **関連ファイル**: `src/audioengine/ISRDSPQuarantine.{h,cpp}`, `src/audioengine/AudioEngine.h`

---

## 1. 問題の根本原因

`DSPQuarantineManager::destroyForShutdown(uint32_t slot)` は定義されているが、**どこからも呼ばれていない**。

```
ISRDSPQuarantine.h:
  bool destroyForShutdown(uint32_t slot);  // 宣言あり

ISRDSPQuarantine.cpp:
  bool DSPQuarantineManager::destroyForShutdown(uint32_t slot) { ... }  // 実装あり

  // ★ 呼び出し元: なし！（grep結果ゼロ）
```

加えて、シャットダウン時に quarantine スロットは以下の2つのストアに二重管理されている：

| 管理対象 | ロケーション | シャットダウン時の処理 |
|----------|-------------|----------------------|
| Quarantineフラグ | `DSPQuarantineManager::quarantineActiveFlags_[]` | `destroyForShutdown()` 未使用のため残留 |
| RetireRuntimeEx Lane | `RetireRuntimeEx::laneBySlot_[]` | 消去されず `Quarantine` のまま残留 |

ログでは `quarantine=10` が観測され、シャットダウン後も解放されない。

---

## 2. ソースコード調査による確定事項

### 2.1 Quarantine の3系統管理（確認済み）

`AudioEngine::quarantineSlot()` は以下の3系統を同時に更新する：

| 系統 | クラス | メソッド | 内容 |
|------|--------|---------|------|
| ① Truth | DSPHandleRuntime | `quarantineSlot(slot)` | レジストリ状態を Quarantined に |
| ② Audit | DSPQuarantineManager | `quarantineHandle(slot, gen, reason)` | フラグ + 監査ログ |
| ③ Lane | RetireRuntimeEx | `quarantine(slot)` | レーン設定 + residentCount++ |

解放時も3系統すべてを解除する必要がある。

### 2.2 既存APIの調査結果（grep/Serena確認）

#### `DSPHandleRuntime::destroyQuarantineSlot()` — ✅ **実装済み・未使用**

`ISRDSPHandle.cpp:137-175` で完全実装済み。Quarantined→DestroyPending→Reclaimed の2段階遷移＋
instance解放を行う。active/fading/crossfade に関与していないことを確認してから解放するセーフティ機構付き。

```cpp
void DSPHandleRuntime::destroyQuarantineSlot(uint32_t slot, uint64_t expectedGeneration) noexcept {
    // Phase 0: generation保護（expectedGeneration==0でスキップ）
    // Phase 1: active/fading/crossfadeチェック
    // Phase 2: Quarantined → DestroyPending（CAS）
    // Phase 3: instance = nullptr
    // Phase 4: DestroyPending → Reclaimed
}
```

#### `DSPQuarantineManager::destroyForShutdown()` — ✅ **実装済み・未使用**

`ISRDSPQuarantine.cpp:115-139` で完全実装済み。フラグ解除＋監査ログ解決＋compaction。

#### `kMaxSlots` / `MAX_DSP_SLOTS` — 可視性確認

- `DSPQuarantineManager::kMaxSlots = 256` — **private**（public化必須）
- `DSPHandleRuntime::MAX_DSP_SLOTS = 256` — **public**（そのまま使用可能）

#### `compactAuditLog()` — ✅ **実装済み・private**

ループ内で毎回呼ぶと非効率。PR2でpublic化してループ後1回呼びに最適化する。

### 2.3 設計の経緯（work24参照）

`doc/work24/refactoring_plan_p7.md:2470` には以下の記録がある：

```
| destroyForShutdown(slot) | A-1 | ❌ 未実装 |
```

work24時点では `destroyForShutdown` 自体が未実装だったが、
その後の実装作業でコード自体は書かれたものの**呼び出し元の追加が失念された**。
また `DSPHandleRuntime::destroyQuarantineSlot()` も同様に実装済み未使用。

---

## 3. 修正設計（確定版）

### 3.1 修正方針

`AudioEngine::releaseResources()` の VerifyDrained フェーズ直前に、
全quarantineスロットを3系統すべて解放する。

### 3.2 挿入箇所（確定）

`AudioEngine.Processing.ReleaseResources.cpp` の `releaseResources()` 内、
`drainDeferredRetireQueues(true)` 直後、`VerifyDrained` 遷移の直前：

```
releaseResources() の流れ（変更後）:

1. shutdownPhase = STOP_ACCEPTING_WORK
2. shutdownPhase = STOP_AUDIO
3. stopRebuildThread
4. shutdownPhase = FORCE_EPOCH_ADVANCE
5. shutdownPhase = DRAIN_RETIRE
6. drainDeferredRetireQueues(true)
   ★★★ [追加] quarantine 全解放（3系統） ← ここに挿入
7. shutdownPhase = VerifyDrained  ← quarantine=0 で通過
8. collectDrainAudit() → quarantine=0 確認
...
```

**安全性の根拠**: shutdownPhase=STOP_AUDIO 以降は Audio Thread が停止しており、
quarantineスロットへのRTアクセスは発生しない。

### 3.3 実装コード（確定版）

```cpp
// ★ PR2: Quarantine 全スロット強制解放（シャットダウン専用）
// 3系統（DSPHandleRuntime + DSPQuarantineManager + RetireRuntimeEx）をすべて解放
{
    const auto residentBefore = dspQuarantineManager_.residentCount();
    if (residentBefore > 0) {
        diagLog("[DIAG] releaseResources: quarantinedSlots="
                + juce::String(static_cast<int>(residentBefore))
                + " — performing shutdown cleanup");

        for (uint32_t slot = 0; slot < DSPHandleRuntime::MAX_DSP_SLOTS; ++slot) {
            // ★ 系統①: DSPHandleRegistry の Quarantined 状態を解除
            //    generation=0 で世代チェックをスキップ（シャットダウン時は安全）
            dspHandleRuntime_.destroyQuarantineSlot(slot, 0);

            // ★ 系統②: DSPQuarantineManager のフラグ解除
            //    非アクティブなら false で何もしない
            if (dspQuarantineManager_.destroyForShutdown(slot)) {
                // ★ 系統③: RetireRuntimeEx のレーン解放（residentCount-- 含む）
                retireRuntimeEx_.reclaim(slot);
            }
        }

        const auto residentAfter = dspQuarantineManager_.residentCount();
        diagLog("[DIAG] releaseResources: quarantine cleanup done "
                + juce::String(static_cast<int>(residentBefore))
                + " -> "
                + juce::String(static_cast<int>(residentAfter)));
    }
}
```

**重要**: 従来設計から `dspHandleRuntime_.destroyQuarantineSlot()` の呼び出しを追加。
`destroyQuarantineSlot` は active/fading/crossfade に関与していないことを確認する
セーフティチェックを持つため、シャットダウン時の誤解放リスクをさらに低減する。

### 2.4 `DSPQuarantineManager::destroyForShutdown()` の現状確認

```cpp
// 現行実装（呼び出しがないだけで内容は正しい）
bool DSPQuarantineManager::destroyForShutdown(uint32_t slot) {
    if (slot >= kMaxSlots) return false;
    bool active = convo::consumeAtomic(quarantineActiveFlags_[slot], std::memory_order_acquire);
    if (!active) return false;          // 非アクティブ＝何もしない

    convo::publishAtomic(quarantineActiveFlags_[slot], false, std::memory_order_release);  // フラグ解除
    for (auto& entry : auditLog_) {     // 監査ログ解決
        if (entry.slot == slot && !entry.resolved) {
            entry.resolved = true;
            break;
        }
    }
    compactAuditLog();
    return true;
}
```

**修正不要** — 現状の実装で正しく動作する。ただし、`compactAuditLog()` 呼び出しがループ内で毎回走るため、効率が悪い。

### 3.4 最適化: バッチ compactAuditLog

ループ内で毎回 `compactAuditLog()` を呼ぶ代わりに、ループ終了後に1回だけ呼ぶ：

```cpp
// ★ PR2 最適化: ループ後に1回だけ compaction
for (uint32_t slot = 0; slot < DSPHandleRuntime::MAX_DSP_SLOTS; ++slot) {
    dspHandleRuntime_.destroyQuarantineSlot(slot, 0);
    if (dspQuarantineManager_.destroyForShutdown(slot)) {
        retireRuntimeEx_.reclaim(slot);
    }
}
dspQuarantineManager_.compactAuditLog();  // ★ ループ後に一括 compaction
```

このため、`compactAuditLog()` を public にする必要がある。

### 3.5 `ISRDSPQuarantine.h` の修正

```cpp
// public セクションに移動
public:
    static constexpr size_t kMaxSlots = 256;
    bool isActive(uint32_t slot) const noexcept;
    void compactAuditLog() noexcept;

private:
    // kMaxSlots は public に移動済み。private には残さない
```

### 3.6 `RetireRuntimeEx::reclaim()` の動作確認

`ISRRetireRuntimeEx.cpp:204-220` の実装を確認。
`reclaim(slot)` は以下の処理を行う：

1. `lifecycleStateBySlot_[slot]` を `ReclaimEligible` → `Reclaimed` に遷移
2. `laneBySlot_[slot]` を `Reclaim` に設定
3. 直前のレーンが `Quarantine` の場合、`quarantineResidentCount_` をデクリメント

これにより RetireRuntimeEx 側のquarantine残留カウントも適切に減少する。

---

## 4. リスクとトレードオフ

| リスク | 影響 | 対策 |
|--------|------|------|
| シャットダウン中に Audio Thread が quarantine スロットを参照 | データ競合 | shutdownPhase=STOP_AUDIO 後に実行（Audio Thread は既に停止済み） |
| RetireRuntimeEx::reclaim と DSPQuarantineManager::destroyForShutdown の二重解放 | 不整合あり | destroyForShutdown が先にフラグ確認、reclaim は後続で安全 |
| `compactAuditLog()` ループ内呼び出しのパフォーマンス | 軽微（最大256回） | バッチ compactAuditLog で最適化 |

---

## 5. 検証方法

1. **ログ確認**: 変更前 `[ISR][Shutdown] quarantine=10` → 変更後 `quarantine cleanup done 10 -> 0` 確認
2. **Audit確認**: `[ISR][Shutdown] Drain complete but quarantine residents remain` のログが消えること
3. **退行テスト**: 既存の全単体テストがPASSすること
4. **メモリリークチェック**: shutdown後のquarantineスロット解放によるメモリ解放確認
