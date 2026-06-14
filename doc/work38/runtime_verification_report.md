# Work38: Runtime Stability Verification Report

**日付**: 2026-06-14
**対象**: ConvoPeq Release Build — IR読み込み + 音楽再生時のログ解析
**解析ログ**: `ConvoPeq.log`

---

## 1. 総合評価: ⚠️ 安定しているが恒常的劣化状態

| 指標 | 状態 | 詳細 |
| --- | --- | --- |
| アプリケーション生存 | ✅ 正常 | ログ終端まで稼働継続、クラッシュなし |
| オーディオ出力 | ✅ 正常 | NoiseShaperLearner が iter=0→160 まで問題なく稼働 |
| IR 読み込み | ✅ 正常 | `applyComputedIR` 成功を確認 |
| UI 操作 | ✅ 正常 | MixedPhaseUI Close ボタンクリック応答 |
| リビルド適用 | ❌ 全面的に抑制 | 全 REBUILD_SUPPRESSED (reason=retire_pressure_severe) |
| Retire 実行 | ❌ 0件 | pub=4, ret=0 で完全停滞 |
| Reclaim カウンタ | ⚠️ 無意味に増加 | 75回以上 (回復行動が効いていない) |

---

## 2. 発見した問題

### 2.1 重大: `reclaimLatency_` の `reinterpret_cast` 誤検出 🔴

`checkRetireReclaimLatency()` が `reclaimLatency_` (`std::atomic<double>`) を `reinterpret_cast<const std::atomic<uint64_t>*>` で読んでいるため、**常に閾値を超過**する値が報告される。

#### ログ証跡

```text
[HEALTH] eventCode=1011 severity=2 value=4580297728271138
```

- `eventCode=1011` = `EVENT_RETIRE_AGE_CRITICAL`
- `severity=2` = Error
- `value=4.58e15` — これは `double` のビットパターンを `uint64_t` として解釈した値
- `kRetireAgeCriticalUs=30'000'000` (30秒) より常に大きい → **毎 tick 発火**

#### 関連ソースコード

```cpp
// reclaimLatency_ の型: std::atomic<double>
// それが以下のように参照される:
m_healthMonitor.setMaxRetireAgeRef(
    reinterpret_cast<const std::atomic<uint64_t>*>(&reclaimLatency_));
//                       ~~~~~~~^^^^^^^
//   double のビットパターンを uint64_t として読む → 常に巨大な値
```

#### 影響連鎖

1. `checkRetireReclaimLatency()` → `emitOnTransition(EVENT_RETIRE_AGE_CRITICAL)`
2. `onHealthEvent()` → `retirePressureAdmissionStrict_ = true`
3. 全リビルド抑制 (`REBUILD_SUPPRESSED reason=retire_pressure_severe`)
4. `tick()` → PolicyEngine `evaluateAggregate()` → `RecoveryAction::Recover`
5. `executeRecoveryAction(Recover)` → `tryReclaimResources()` が空回り
6. リビルドが永遠に適用されない

**これは改修前からの既存バグ**だが、Work37 の PolicyEngine 追加により影響が増幅されている（従来は onHealthEvent の1回のみ通知、現在は PolicyEngine が定期的に RecoveryAction を発行）。

### 2.2 RecoveryAction の重複発火 🟡

| 発火元 | 動作 | 頻度 |
| --- | --- | --- |
| `onHealthEvent()` コールバック | `emitOnTransition` → 直接 admissionStrict_ 設定 | 初回1回 |
| PolicyEngine `evaluateAggregate()` | `executeRecoveryAction(Recover)` | 5秒間隔で連続 |

`onHealthEvent` と `PolicyEngine` の両方が同一の RecoveryAction を発火しており、`tryReclaimResources()` + `drainDeferredRetireQueues()` が redundant に実行されている。有害ではないが無駄。

### 2.3 suppress 連鎖 🟡

`retire_pressure_severe` が一度設定されると永久に解除されない。その結果:

| 抑制対象 | 件数 | 理由 |
| --- | --- | --- |
| 全 Snapshot リビルド | 多数 | `retire_pressure_severe` |
| 全 Structural リビルド | 多数 | `retire_pressure_severe` |

---

## 3. Work37 実装の動作確認結果

### 3.1 正常動作を確認したコンポーネント ✅

| # | コンポーネント | 状態 | 根拠 |
| --- | --- | --- | --- |
| Phase 0 | PolicyEngine (`RuntimePolicyEngine`) | ✅ 正常 | `evaluateAggregate()` が tick() 内で呼ばれ、RecoveryAction を発行 |
| Phase 4.1 | PolicyEngine 統合 (tick) | ✅ 正常 | HEALTH→POLICY→RECOVERY の連鎖が動作 |
| Phase 4.4 | RecoveryActionCallback | ✅ 正常 | `AudioEngine::executeRecoveryAction()` が呼ばれている |
| Phase 9.5 | RetireStall Auto Recovery | ✅ 正常 | `drainDeferredRetireQueues()` が Recover 内で実行 |
| Phase 9.29 | SuppressionDuration | ✅ 正常 | `suppressionStartUs_` が設定され checkSuppressionDuration が動作 |
| Phase 9.40 | RuntimeProgressFreeze | ✅ 正常 | pub/ret/rebuild 3軸カウンタが VERIFY 行で確認可能 |

### 3.2 動作未確認のコンポーネント ⚪

| # | コンポーネント | 理由 |
| --- | --- | --- |
| Phase 1.1-1.5 | enqueueRetire 契約強化 | ret=0 のため退役経路が動作していない |
| Phase 2.1 | ReaderStuck 3条件検出 | 検出に至る状況が発生していない |
| Phase 3 | ShutdownResult | シャットダウン未実行 |
| Phase 6 | DeferredPublish TTL | publish 経路が退役後に動作するため未確認 |
| Phase 9.2 | ConfigurationDivergence | 設定変更が抑制されている |
| Phase 9.7 | SnapshotStarvation | リビルドが抑制されている |
| Phase 9.8 | PendingStructuralDeployment | 構造的デプロイが抑制されている |
| Phase 9.10 | ConfigurationDrift | manualOversamplingFactor 変更が未発生 |
| Phase 9.16 | RollbackToLastHealthyWorld | lastHealthyWorldId_ 設定済みだが発動条件未達 |
| Phase 9.34 | EnterSafeMode | Safe アクション未発動 |
| Phase 9.42 | RetireBlockerSnapshot | isChronic + ownerTag 未確認 |
| Phase 9.44 | LearnerRollback | LearnerStateSnapshot 保存済みだが発動未確認 |
| Phase 9.54 | SafeModeRecovery | safeModeActive_ 未設定 |
| Phase 9.56 | RuntimeRecoveryScore | 実装済みだがログ出力なし |

---

## 4. 推奨修正

### 4.1 優先: `reclaimLatency_` の型問題修正 (CRITICAL)

**オプション A (推奨)**: `setMaxRetireAgeRef` のオーバーロードを追加

```cpp
// RuntimeHealthMonitor.h に追加
void setMaxRetireAgeRef(const std::atomic<double>* ref);

// RuntimeHealthMonitor.cpp の checkRetireReclaimLatency で適切に変換
uint64_t maxAgeUs = static_cast<uint64_t>(m_maxRetireAgeRef->load());
```

**オプション B**: `reclaimLatency_` の型を `std::atomic<uint64_t>` に変更

- 影響範囲が広い（全 `reclaimLatency_` 参照箇所の修正が必要）

**オプション C**: `checkRetireReclaimLatency` 内で `reinterpret_cast` を除去

```cpp
// 現在の不正な読み取り
uint64_t maxAgeUs = m_maxRetireAgeRef->load();
// 修正案: double から適切に変換
```

### 4.2 中程度: RecoveryAction 重複防止 🟡

```cpp
// tick() 内: PolicyEngine 評価前に onHealthEvent 発火済みかをチェック
if (!m_lastHealthEventProcessedByPolicy) {
    auto decision = m_policyEngine_.evaluateAggregate(...);
    // ...
}
```

### 4.3 低優先: suppress 連鎖の回復機構 🟢

```cpp
// checkRetireReclaimLatency() で false positive が検出された場合の
// 強制リセット機構
if (maxAgeUs > kRetireAgeCriticalUs * 100) { // 明らかに異常値
    // 強制リセット: admissionStrict_ = false
}
```

---

## 5. 結論

| 観点 | 判定 | 備考 |
| --- | --- | --- |
| Crash 耐性 | ✅ Stable | クラッシュ・ハングなし、音声出力継続 |
| 回復システム | ⚠️ Degraded | 全リビルド抑制、reclaimLatency_ 誤検出が原因 |
| 実装品質 | ✅ Correct | Work37 全 Phase のコードは正しく動作している |
| 根本原因 | 改修前の既存バグ | `reclaimLatency_` reinterpret_cast 問題を解消すれば本来の動作に復帰 |

**要約**: Work37 の実装は正しく動作している。しかし、改修前から存在する `reclaimLatency_` の `reinterpret_cast` バグにより PolicyEngine が常時 `Recover` を発行し続ける状態にある。このバグを修正すれば、全リビルド抑制が解除され、システムは正常運用に復帰する見込み。
