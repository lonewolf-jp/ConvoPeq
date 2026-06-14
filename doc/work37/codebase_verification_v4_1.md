# 改修計画書 v4.1 実コード検証報告

> **日付**: 2026-06-14
> **検証方法**: Serena MCP, CodeGraph MCP (Full Index: 51K entities), graphify MCP (15K nodes), grep/Select-String, 直接ファイル読み取り
> **確認範囲**: 15+ ファイル, 2000+ 行

---

## 1. 全フェーズの実コード照合結果

### 凡例

- ✅ **正確** — 設計書の記述がコードと完全一致
- ⚠️ **不正確** — 設計書の記述に誤差あり（以下に詳細）
- ❌ **誤り** — 設計書の記述がコードと不一致
- 🆕 **未記載** — コードに存在するが設計書に未記載

---

### Phase 0: PolicyEngine 基盤

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| `tick()` 内の9種類の check* 関数 | ✅ **正確** | `RuntimeHealthMonitor.cpp:9-19` で全9関数の呼び出し確認済み |
| MonitorState 6系統 | ✅ **正確** | `.h:107-115` で6つの `m_prev*State` 確認済み |
| `updateHealthState()` が単一権限 | ✅ **正確** | `.cpp:103-133` で確認。PolicyDecision を受け取らない点のみ設計と差異 |
| イベントコード定数一覧 | ⚠️ **不足あり** | コードには `EVENT_RETIRE_AGE_WARNING(1010)` が存在するが設計書に未記載 |
| Action優先順位 | ✅ **正確** | 既存の `getPrimaryBlockingReason()` の優先順位と一致 |
| `PolicyContext` 型 | ✅ **前回調査で削除決定済み** | コードに存在せず、削除は正しい |
| Cooldown スレッド安全性 | ✅ **正確** | `tick()` は Timer callback (Non-RT単一スレッド) のみから呼ばれる |

#### ⚠️ 不正確: updateHealthState() のシグネチャ

設計書 Phase 4.1 のコード例:

```cpp
void RuntimeHealthMonitor::updateHealthState(const PolicyDecision& decision) noexcept {
```

実際のコード:

```cpp
void RuntimeHealthMonitor::updateHealthState() noexcept  // 引数なし！
```

設計書は PolicyDecision を受け取る新しいシグネチャを提案しているが、既存コードは引数なし。**Phase 4.1 実装時には新規オーバーロード追加が必要。**

#### 🆕 未記載: updateHealthState() 内の優先順位

実際の `updateHealthState()` は以下の優先順位で ISRHealthState を決定する:

| 優先度 | 条件 | 結果 |
|--------|------|------|
| 1 | `m_prevRetireState == Error` | Critical |
| 2 | `m_prevPublicationState == Error` | Critical（retireより優先度高） |
| 3 | `m_prevOverflowRateState == Error` | Critical（既にCriticalなら変更なし） |
| 4 | `m_prevReaderSlotState == Error` | Critical（同上） |
| 5 | `m_prevRetireAgeState == Error` | Critical（同上） |
| 6 | 上記 Warning + Healthy 該当 | Degraded |

設計書はこの優先順位を明文化していない。

---

### Phase 1: enqueueRetire 戻り値契約の修復

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| 1.1 DSPLifetimeManager::retire() | ✅ **正確** | `DSPLifetimeManager.h:44` で戻り値未チェック確認済み |
| 1.2 SnapshotCoordinator | ⚠️ **再評価が必要** | 後述 |
| 1.3 RefCountedDeferred::release() | ✅ **正確** | `RefCountedDeferred.h:23` で戻り値未チェック確認済み |
| 1.4 EQProcessor retire 戻り値伝播 | ✅ **正確**（ただし呼出元数訂正済み） | `retireEQStateDeferred` は void、呼出元13箇所 |
| 1.5 フォールバックキュー統合 | ✅ **正確** | `DeferredRetireFallbackQueue.h` 確認済み |

#### ⚠️ 不正確: Phase 1.2 SnapshotCoordinator のスレッド安全性

設計書 v4.1 の「再評価: 実施可能。P0→P1」は**部分的に誤り**。

実際の呼び出し経路:

- `startFade()`: Non-RT (Timer callback) ✅ → **enqueueWithRetry 可能**
- `resetFadeStateAndRetireTarget()`: **`updateFade()` 経由で RT から呼ばれ得る** ❌ → **tryReclaim 不可**
- `completeFade()`: Non-RT (Timer callback → tryCompleteFade) ✅ → **enqueueWithRetry 可能**
- `switchImmediate()`: Non-RT ✅ → **enqueueWithRetry 可能**
- `retireCurrentAndTarget()`: Non-RT (releaseResources) ✅ → **enqueueWithRetry 可能**

**結論**: Phase 1.2 は「部分実施」が正しい。`resetFadeStateAndRetireTarget()` (L67) は RT から呼ばれる可能性があるため tryReclaim を追加できない。`startFade()` (L36), `completeFade()` (L87), `switchImmediate()` (header), `retireCurrentAndTarget()` (header) は Non-RT 安全。

| 経路 | Non-RT? | enqueueWithRetry 可能? |
|------|---------|----------------------|
| `startFade()` L36 | ✅ Yes | ✅ Yes |
| `resetFadeStateAndRetireTarget()` L67 | ❌ No (updateFade→RT) | ❌ **No** |
| `completeFade()` L87 | ✅ Yes | ✅ Yes |
| `switchImmediate()` header | ✅ Yes | ✅ Yes |
| `retireCurrentAndTarget()` header L161-162 | ✅ Yes | ✅ Yes |

**修正**: 設計書の4経路→実際は5経路。`switchImmediate()` にも enqueueRetire がある。`resetFadeStateAndRetireTarget()` は RT からのため除外。

---

### Phase 2: Reader Stuck 判定改善

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| 現在epoch差のみ | ✅ **正確** | `EpochDomain.h:265-302` で確認。epoch差のみで判定 |
| stuckThreshold=10 | ✅ **正確** | `RuntimeHealthMonitor.cpp:165` で `detectStuckReaders(10)` 確認 |
| residency収集あり | ✅ **正確** | `EpochDomain.h:286` で `residencyUs` 計算確認済み |
| 修正後の複合判定 | ✅ **設計は妥当** | 実際のコードと比較して妥当な改善 |

#### 🆕 未記載: severe 条件の既存実装

実際のコード `RuntimeHealthMonitor.cpp:167`:

```cpp
const bool severe = (stuckInfo.pendingRetireCount > 100 || stuckInfo.residencyTimeUs > 30'000'000);
```

設計書には pendingRetireCount > 100 の severe 条件が記載されていないが、既に実装済み。

---

### Phase 3: ShutdownResult 導入

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| emitShutdownTrace JSON 内容 | ✅ **正確**（healthState不足は既知） | `ISRShutdown.cpp:135-210` でJSON出力確認 |
| collectDrainAudit の healthState | ✅ **既に実装済み** | `AudioEngine.Threading.cpp:84` で `.healthState = m_healthMonitor.getHealthState()` 確認 |
| ShutdownBlockingReason | ✅ **正確** | `ISRShutdown.h:46-56` で8値+Unknown確認 |
| collectResult 不在 | ✅ **正確** | コードに存在せず |
| ShutdownResult 型不在 | ✅ **正確** | コードに存在せず |

#### 🆕 未記載: emitShutdownTrace の全出力フィールド

実際のJSON出力:

```json
{
  "schema": "shutdown_trace_v2",
  "phase": "...",
  "phaseName": "...",
  "blockingReason": "...",
  "blockingReasonCode": ...,
  "transitionViolations": ...,
  "sh1_callbackCount": ...,
  "sh2_activeCrossfade": ...,
  "sh3_pendingRetire": ...,
  "sh4_observerCount": ...,
  "sh5_lateCallbackCount": ...,
  "sh6_postStopEnqueueCount": ...,
  "verified": true/false
}
```

欠落: `healthState`, `durationMs`, `lastNonTerminalPhase`

---

### Phase 4: Policy Engine 統合

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| 既存 onHealthEvent 4種 | ✅ **正確** | `AudioEngine.Timer.cpp:541-653` で確認 |
| EVENT_RETIRE_STALL | ✅ **正確** | L588 で確認 |
| EVENT_PUBLICATION_STALL | ✅ **正確** | L563 で確認 |
| EVENT_READER_SLOT_USAGE | ✅ **正確** | L549 で確認 |
| EVENT_CROSSFADE_TIMEOUT | ✅ **正確** | L598 で確認 |
| EVENT_READER_STUCK（診断のみ） | ✅ **正確** | L641 で確認 |

#### 🆕 未記載: EVENT_RETIRE_AGE_CRITICAL の併合処理

実際の onHealthEvent L588:

```cpp
if ((event.eventCode == convo::EVENT_RETIRE_STALL
     || event.eventCode == convo::EVENT_RETIRE_AGE_CRITICAL)  // ← 両方を処理
    && event.severity == convo::HealthEvent::Severity::Error)
```

設計書は `EVENT_RETIRE_STALL` のみ記載しているが、実際には `EVENT_RETIRE_AGE_CRITICAL` も同様に処理されている。

---

### Phase 4.4: 背圧機構の統一（**最重要 不正確**）

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| 経路1の存在 | ✅ **正確** | `AudioEngine.Retire.cpp:118-154` で確認 |
| 経路2の存在 | ✅ **正確** | `AudioEngine.Timer.cpp:549` で確認 |
| 経路3の存在 | ✅ **正確** | `AudioEngine.Timer.cpp:588` で確認 |

#### ❌ 誤り: 経路1の発火条件と動作

設計書 ファクトF の記述:

```
経路1 ... → retirePressureAdmissionStrict_ = true (AudioEngine.Retire.cpp:154)
       発火条件: フォールバックキューサイズ > kFallbackWarningThreshold
```

**実際のコードでは全く異なる**: 経路1は **2段階の複合処理** である:

```
Step 1: evaluateRetirePressureLevelNoRt(retireDepth, hwm) → Level 0-3
        applyRetirePressurePolicyNoRt() が Level>=3 で admissionStrict_ を設定

Step 2: overflow 検出ブロック (Retire.cpp:121-154)
        overflowLevel = (droppedDelta>0 || chronicByDuration>5s || chronicByFrequency>3.0/sec) ? 3 : 0
        effectiveLevel = max(step1_level, overflowLevel)
        → retirePressureAdmissionStrict_ = (effectiveLevel >= 3)
        → 上書き！（step1 の結果を上書き）
```

**発火条件**: retireDepth/hwm 比率 (Level 3 = `kRetirePressureSeverePercent`%) **および** overflow 状態（3条件のOR）

設計書は「フォールバックキューサイズ > kFallbackWarningThreshold」と簡略化しているが、実際は retireDepth(ring queue) + overflow(複合条件) である。

#### ❌ 誤り: 「3つの独立した経路」の記述が不正確

経路1は実は**内部的に2つの独立したサブ経路を持つ**:

- サブ経路1a: `evaluateRetirePressureLevelNoRt` → `applyRetirePressurePolicyNoRt`
- サブ経路1b: overflow 検出ブロック → `effectiveLevel` で上書き

これらは同じ関数 `drainDeferredRetireQueues()` 内で順次実行されるが、`applyRetirePressurePolicyNoRt` の結果を overflow ブロックが上書きする設計になっている。

**正しい理解**: 実質的に **4つの独立した条件** が `retirePressureAdmissionStrict_` を設定しうる:

1. retireDepth/hwm 比率 (Level 3) — Retire.cpp:284
2. overflow 検出 — Retire.cpp:148-154
3. EVENT_READER_SLOT_USAGE — Timer.cpp:557
4. EVENT_RETIRE_STALL — Timer.cpp:588

---

### Phase 5-8

| 設計項目 | 検証結果 | 詳細 |
|----------|----------|------|
| Phase 5 BlockingReason 多値化 | ✅ **P3格下げ妥当** | 既存の単一値 + getPrimaryBlockingReason で十分 |
| Phase 6 Deferred Publish TTL | ✅ **設計は妥当** | DeferredPublishSlot に enqueueTimestampUs 既存 |
| Phase 7 WorldConsistency | ✅ **設計は妥当** | verifyWorldConsistency() 実装済み (`RuntimeDrainAudit.h:80`) |
| Phase 8.1 Fallback OOM | ✅ **設計は妥当** | 2000件ハードリミット + notifyOverflow |
| Phase 8.2 EmergencyDrain | ✅ **設計は妥当**（不整合修正が必要） | 後述 |
| Phase 8.3 TTL通知 | ✅ **設計は妥当** | |

#### ⚠️ 不正確: Phase 8.2 EmergencyDrain advancePhase の不整合

設計書の記述:

```
advancePhase は ifdef なしでスキップ
```

実際のコード `ISRShutdown.cpp:80`:

```cpp
case ShutdownPhase::ReclaimComplete:
    next = ShutdownPhase::VerifyDrained;  // EmergencyDrain スキップ
    break;
```

これには `#ifdef` がなく、常に EmergencyDrain をスキップする。一方 `releaseResources()` は常に EmergencyDrain に遷移する（194行目）。しかしその後の `transitionTo(VerifyDrained)` で上書きされるため、実質的に EmergencyDrain は ReleaseResources.cpp 内の `#ifdef` ブロックのみで意味を持つ。

**現状の動作**: EmergencyDrain phase への遷移は行われるが、すぐに VerifyDrained に進む。EmergencyDrain の実処理は ReleaseResources.cpp の `#ifdef` ブロック内でのみ実行される。これは意図した動作かどうかを確認する必要がある。

---

## 2. 新規発見: 設計書に未記載の重要ファクト

### ファクトG: updateHealthState() の優先順位が設計書に未記載

`RuntimeHealthMonitor.cpp:103-133` の実際の優先順位:

1. Retire Error → Critical
2. Publication Error → Critical（最優先）
3. OverflowRate Error → Critical（既存Critical維持）
4. ReaderSlot Error → Critical（同上）
5. RetireAge Error → Critical（同上）
6. 各 Warning + Healthy → Degraded

**設計書への反映**: Phase 0 の異常分類テーブルに優先順位を追記すべき。

### ファクトH: 経路1は内部的に2段階（背圧→overflow上書き）

`AudioEngine.Retire.cpp:118-154` の実際のフロー:

```
① evaluateRetirePressureLevelNoRt → Level 0-3
② applyRetirePressurePolicyNoRt → flags設定
③ overflow検出 → overflowLevel (0 or 3)
④ effectiveLevel = max(Level, overflowLevel)
⑤ ALL flagsをeffectiveLevelで上書き ← ②の結果を上書き
```

これにより、overflow が発生した場合は retireDepth の評価結果を上書きする。

### ファクトI: EVENT_RETIRE_AGE_CRITICAL も onHealthEvent で admissionStrict_ を設定

設計書は `EVENT_RETIRE_STALL` のみ記載しているが、実際のコードでは `EVENT_RETIRE_AGE_CRITICAL` も同じ条件分岐内で処理されている。

---

## 3. 設計書の修正すべき箇所一覧

| # | 箇所 | 誤り | 修正内容 |
|---|------|------|----------|
| 1 | 0.3 ファクトF 経路1 | 発火条件が「フォールバックキュー > threshold」 | retireDepth/hwm比率 + overflow複合条件に訂正 |
| 2 | 0.3 ファクトF | 「3つの独立した経路」 | 実質4条件。経路1は内部的に2段階(背圧+overflow上書き) |
| 3 | Phase 1.2 | 「実施可能。P0→P1」 | 「一部実施可能。resetFadeStateAndRetireTargetはRT除外」に修正 |
| 4 | Phase 1.2 | 3経路+4経路 | 実際は5経路。switchImmediate() にも enqueueRetire あり |
| 5 | Phase 4.1 tick() コード例 | updateHealthState(decision) | 新規オーバーロードとして明示 |
| 6 | Phase 4.4 | 経路1の注入方法 `injectBackpressureSignal` | 経路1内のoverflow検出ロジックもPolicyEngineに統合要 |
| 7 | Phase 4.4 削除対象 | 削除行番号不明確 | 具体的な置換コードを提示 |
| 8 | 異常分類テーブル | EVENT_RETIRE_AGE_CRITICAL の記載なし | テーブルに追記 |
| 9 | 設計定数一覧(2箇所) | kReaderSlotCriticalThreshold=0.75 | 実際: Warning=0.50, Critical=0.75 だが設計書は0.75のみ |
| 10 | section 4.6 | kStuckResidencyThresholdUs=1秒 | 設計提案のまま。既存コードでは severe=30秒のみ |
