# D101-32-D — Vestigial Counter / Semantic Event API Removal（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: 実装（D101-32-C §7 削除境界の機械的・domain-consistent 反映）
- **判定**: **PASS**（全 PASS 条件充足、詳細 §6）
- **基準**: ConvoPeq.md 2026-08-25 10:33 再生成版（作業開始時）／最終再生成 **2026-08-25 10:49 版**
- **前提**: D101-31-D / D101-32-A / D101-32-C すべて PASS

---

## 1. 削除実施内容（semantic domain 単位の一括削除）

### Domain A — fallback

| 種別 | 対象 | ファイル:位置（削除時点） |
|---|---|---|
| setter 宣言+定義 | `setFallbackBacklogCount` | Coordinator.h:146 / .cpp:277-279 |
| counter field | `fallbackBacklogCount_` | h:562 / cpp ctor :20 |
| getter | `getFallbackBacklogCount` | h:159 / cpp:456-458 |
| event ×2 | `onFallbackAccepted` / `onFallbackConsumed` | h:133-134 / cpp:208-219 |
| drain 条件 | `fallbackBacklogCount_ == 0` | cpp:533 |

### Domain B — deferred retire residency

| 種別 | 対象 | ファイル:位置（削除時点） |
|---|---|---|
| setter 宣言+定義 | `setDeferredRetireResidencyCount` | h:148 / cpp:285-287 |
| counter field | `deferredRetireResidencyCount_` | h:564 / cpp ctor :22 |
| getter | `getDeferredRetireResidencyCount` | h:160 / cpp:460-462 |
| event ×2 | `onDeferredRetireAccepted` / `onDeferredRetireConsumed` | h:135-136 / cpp:221-232 |
| drain 条件 | `deferredRetireResidencyCount_ == 0` | cpp:535 |

### Domain C — Coordinator quarantine resident

| 種別 | 対象 | ファイル:位置（削除時点） |
|---|---|---|
| setter 宣言+定義 | `setQuarantineResidentCount` | h:149 / cpp:289-291 |
| counter field | `quarantineResidentCount_`（**Coordinator のみ**） | h:574 |
| getter | `getQuarantineResidentCount`（**Coordinator のみ**） | h:161 / cpp:464-466 |
| drain 条件 | `quarantineResidentCount_ == 0` | cpp:542 |

### 追加 — reclaim test-reset setter

| 種別 | 対象 | 根拠 |
|---|---|---|
| setter 宣言+定義 | `setReclaimInFlightCount` | production accounting は onReclaimBegin/End に完全移行済み。test reset 用途も消滅（D101-32-C §2.2） |

**部分削除なしを確認**: 各 domain で setter/counter/getter/event/drain 条件を一括除去。
orphan API・dead initialization・参照切れは残存ゼロ（§3 grep 監査）。

### Test code 修正（src/tests/ISRSemanticValidationTests.cpp）

- :329-331 / :365-367(旧) / :435-437(旧) の `set{FallbackBacklog,ReclaimInFlight,DeferredRetireResidency}Count(0)`
  を削除し、fresh instance 初期値 0 が初期状態を保証する旨の注記コメントに置換。
- **維持**: `setRetireBacklogCount(0)/(1)/(9)` 全行（Pressure FSM 駆動 + drain violation 注入）、
  `setPublicationBacklogCount` / `setPendingIntentCount` / `setSwapPending`（scope 外）。
- `invariant_INV3_INV5.cpp` は無変更（onReclaimBegin/End + getReclaimInFlightCount 方式のため影響なし）。

---

## 2. KEEP 域の保持確認（変更禁止項目）

| 対象 | 状態 |
|---|---|
| `setRetireBacklogCount` / `retireBacklogCount_` / `getRetireBacklogCount` | ✅ 未触碰（cpp 内参照3箇所、test 内9箇所を grep で確認） |
| `noteRetireBacklogChanged` / `previousRetireBacklogCount_` / `pressureNormalizedWindows_` / `kPressureSlopeThreshold` / `kPressureNormalizeWindows` | ✅ 未触碰（合計14箇所の参照が元のまま） |
| `onRetireAccepted` / `onRetireConsumed` | ✅ 未触碰（retireBacklogCount_ / Pressure FSM とセットで保持） |
| `onReclaimBegin` / `onReclaimEnd` / `reclaimInFlightCount_` / `getReclaimInFlightCount` | ✅ 未触碰（production wired: Retire.cpp:62/66/:356/359 + Coordinator:736/747 相当経路） |
| `AdmissionPackedState`（packedState_ / tryAdmit/release/closeAdmission/joinProducers） | ✅ ISRShutdown.{h,cpp} 未触碰 |
| `publicationBacklogCount_` / `setPublicationBacklogCount` / `getPublicationBacklogCount` / `setPendingIntentCount` | ✅ scope 外として未触碰（次回独立タスク） |

---

## 3. grep 監査（§9 orphan / resurrection 検査）

```
対象シンボル（非コメント実参照） in src/{audioengine,core,convolver,eqprocessor}:
  setFallbackBacklogCount / setReclaimInFlightCount / setDeferredRetireResidencyCount /
  setQuarantineResidentCount / onFallbackAccepted / onFallbackConsumed /
  onDeferredRetireAccepted / onDeferredRetireConsumed / getFallbackBacklogCount /
  getDeferredRetireResidencyCount / fallbackBacklogCount_ / deferredRetireResidencyCount_
  → すべて 0 件 ✅

quarantineResidentCount_ のクラス区別（単純 grep ゼロ判定は不使用）:
  RuntimeIntentCoordinator::quarantineResidentCount_
      → src/audioengine/ISRRuntimePublicationCoordinator* および src/tests 内 0 件 ✅（完全削除）
  EpochControl::quarantineResidentCount_（ISRRetireRuntimeEx.h:104 / .cpp）
      → 存続 ✅（別 semantic: Q+EmergencyQ 滞留）
  ISRRetireRouter::quarantineResidentCount()（h:240、Threading.cpp:142 が使用）
      → 存続 ✅（別 semantic: router Q 滞留）
```

---

## 4. semantic drain audit（§10）

### ShutdownScheduler::isFullyDrained() 最終構造（Coordinator.cpp）

```text
intentQueue_.sizeApprox()==0
observeDeferredRing_.size()==0
quarantineFallbackQueue_.sizeApprox()==0
recoveryIntentQueue_.size()==0
retireBacklogCount_==0            ← KEEP / reserved semantic（テスト注入可能なため恒真ではない）
publicationBacklogCount_==0       ← untouched（今回 scope 外）
publicationIntentResidencyCount_==0   (INV-X5-1)
pendingIntentCount_==0
reclaimInFlightCount_==0          (wired event)
quarantineIntentResidencyCount_==0    (INV-X6-4)
quarantineRingResidencyCount_==0      (INV-X6-4)
!recoveryAdmissionPending_            (INV-X1-1/2)
```

→ 指定された semantic 構造と一致。削除した3条件は恒真（writer ゼロ）であり
**drain proof の情報量欠落なし**。

### AudioEngine::isFullyDrained()（Layer 1）

**一切未変更**。実測 source of truth 全7項目
（pendingRetireCount / lifetime pendingIntentCount / overflow ring resident /
DSPQuarantineManager resident / EpochControl quarantine resident / Terminal resident /
bridge isFullyDrained）を維持。

---

## 5. invariant 確認

| INV | 内容 | 結果 |
|---|---|---|
| INV-1 | tryAdmit production caller = Publication / Recovery / Build のみ | ✅ Orchestrator.cpp:69 / RebuildDispatch.cpp:319 / AudioEngine.h:4443（grep 再確認、Retire 追加なし） |
| INV-2 | reclaim-in-flight accounting は onReclaimBegin/End のみ（setter へ回帰なし） | ✅ setter 削除済み、event 経路のみが writer |
| INV-3 | retire/fallback/quarantine の実在資源 authority を vestigial counter に戻さない | ✅ Layer 1 実測方式を維持、Coordinator 側 counter は復活させず |

---

## 6. テスト結果

### 優先テスト（指定3種）

| テスト | 結果 |
|---|---|
| AdmissionPackedState (#21) | ✅ 内部 12/12 PASS |
| ISRSemanticValidationRejects (#19) | ✅ PASS |
| InvariantINV3INV5 (#20) | ✅ PASS |

### ビルド

| Config | 結果 |
|---|---|
| Debug (MSVC / Ninja Multi-Config) | ✅ `[4/4] Checking build artifacts...` PASS（error C/LNK ゼロ） |
| Release | ✅ 同上 PASS |

### 全 CTest（Debug）

```
100% tests passed out of 38
```

（CTest エントリ総数不変 — 本タスクは既存テストの内部修正のみのため）

---

## 7. Domain 別 削減エビデンス表（§13 指定フォーマット）

| Domain | Removed API | Removed field | Removed event | Removed drain condition | Remaining authority |
|---|---|---|---|---|---|
| fallback | setFallbackBacklogCount / getFallbackBacklogCount | fallbackBacklogCount_ | onFallbackAccepted / onFallbackConsumed | fallbackBacklogCount_==0 | queue/ring measurement（quarantineFallbackQueue_.sizeApprox() + Layer 1 ringResident） |
| deferred | setDeferredRetireResidencyCount / getDeferredRetireResidencyCount | deferredRetireResidencyCount_ | onDeferredRetireAccepted / onDeferredRetireConsumed | deferredRetireResidencyCount_==0 | observeDeferredRing_.size() / router 実測 |
| Coordinator quarantine | setQuarantineResidentCount / getQuarantineResidentCount(Coordinator) | quarantineResidentCount_(Coordinator) | （該当なし — 元々 event なし） | quarantineResidentCount_==0 | Layer 1（DSPQuarantineManager::residentCount()）/ 専用 residency domains（intent/ring）/ EpochControl・Router（別クラス・存続） |
| reclaim | **KEEP**（setReclaimInFlightCount のみ削除 — test reset 用途消滅） | **KEEP** reclaimInFlightCount_ | **KEEP** onReclaimBegin/End（wired） | **KEEP** reclaimInFlightCount_==0 | onReclaimBegin/End（唯一の authoritative writer） |
| retire | **KEEP** setRetireBacklogCount / getRetireBacklogCount | **KEEP** retireBacklogCount_ | **KEEP** onRetireAccepted/Consumed | **KEEP** retireBacklogCount_==0 | reserved semantic / Pressure FSM / Layer 1 実測（pendingRetireCount 等） |

---

## 8. PASS 条件判定

| 条件 | 判定 |
|---|---|
| Debug build PASS | ✅ |
| Release build PASS | ✅ |
| CTest 100%（38/38） | ✅ |
| production setter reference = 0（削除4種） | ✅ 非コメント実参照 0 |
| removed event reference = 0（fallback/deferred 4種） | ✅ 0 |
| retire domain unchanged | ✅ grep 数一致（KEEP 域14参照 + test 9呼び出し） |
| reclaim domain unchanged | ✅ counter/event/wired 経路すべて存続 |
| Admission domain unchanged | ✅ ISRShutdown 未触碰、tryAdmit 3-path 不変 |
| コメント以外の orphan / resurrection なし | ✅ §3 監査 |

**VERDICT: PASS**

---

## 9. 補足

- 変更ファイル: ISRRuntimePublicationCoordinator.h / .cpp、ISRSemanticValidationTests.cpp（3ファイルのみ）。
- ConvoPeq.md を作業後に再生成（2026-08-25 10:49 版）。削除済みシンボルは同版から消滅。
- AiDex インデックス更新済み（3ファイル）。
- 次タスク: **D101-32-E — Removal Integrity Audit + Debug/Release + CTest**
  （本報告書の検証を独立視点で再監査）→ **D101-33-A Path B Publication Admission Audit**。
  並行して publicationBacklogCount_/setPendingIntentCount の inventory audit を
  独立タスクとして実施可能（D101-32-C §7.2 候補）。
