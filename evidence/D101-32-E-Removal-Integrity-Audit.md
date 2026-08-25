# D101-32-E — Removal Integrity Audit + Debug/Release + 全CTest（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only gate audit（D101-32-D 実装結果の独立検証 / ソースコード変更 **0**）
- **判定**: **PASS**（E1〜E10 全 Gate 充足）
- **基準**: ConvoPeq.md **2026-08-25 11:38 再生成版**（本監査内でローカル実ファイルから再生成・source of truth 確定）

---

## E-1. 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 11:38)
git status                             → 変更 = D101-31-D 分（ISRShutdown×2 / AudioEngine.h /
                                         CMakeLists / RebuildDispatch / Orchestrator /
                                         ReleaseResources + AdmissionPackedStateTests 新規2）
                                         ＋ D101-32-D 分（Coordinator.h/.cpp +
                                         ISRSemanticValidationTests.cpp）のみ。
                                         意図しない変更なし ✅
git diff --check                       → whitespace エラーなし ✅
```

注記: File Library 側に 10:49 版が反映されていなかったため、指示どおりローカル実ファイルから
再生成し 11:38 版を基準化した。

---

## E-2〜E-5. 削除シンボル resurrection / orphan 監査（production / header / test 3領域分割）

| Domain | シンボル | production | header | test | 判定 |
|---|---|---|---|---|---|
| fallback | setFallbackBacklogCount / getFallbackBacklogCount / fallbackBacklogCount_ / onFallbackAccepted / onFallbackConsumed | **0** | **0** | **0** | ✅ resurrection = 0 |
| deferred | setDeferredRetireResidencyCount / getDeferredRetireResidencyCount / deferredRetireResidencyCount_ / onDeferredRetireAccepted / onDeferredRetireConsumed | **0** | **0** | **0** | ✅ resurrection = 0 |
| Coordinator quarantine | setQuarantineResidentCount / getQuarantineResidentCount(Coordinator) / RuntimeIntentCoordinator::quarantineResidentCount_ | **0** | **0** | **0** | ✅ resurrection = 0 |
| reclaim setter | setReclaimInFlightCount | **0** | **0** | **0** | ✅ 回帰なし |

残存ヒットは**すべて削除理由ドキュメントコメント**（Coordinator.h:563 / .cpp:541/:996/:277-280/
:692、h:142、Retire.cpp:60/:355 の B0-5 記録）であり、コード参照ゼロを確認。

### 同名別 semantic の区別（単純 grep ゼロ判定不使用・指示遵守）

以下は削除対象外として正しく**存続**していることを個別確認:

| 存続シンボル | 所在 | semantic |
|---|---|---|
| `EpochControl::quarantineResidentCount_` | ISRRetireRuntimeEx.h:104 / .cpp:169-301（writer 有: :219/:222/:237） | Q + EmergencyQ 滞留 |
| `EpochControl::getQuarantineResidentCount()` | ISRRetireRuntimeEx.h:59 / .cpp:300、ISRRetire.h:135 経由 | 同上（AudioEngine.Retire.cpp:156 が使用） |
| `ISRRetireRouter::quarantineResidentCount()` | ISRRetireRouter.h:240 | router Q 滞留（Threading.cpp:143 が使用） |
| `quarantineIntentResidencyCount_` / `quarantineRingResidencyCount_` | Coordinator.h:571-572 | X6 transport residency |

---

## E-3/E-6. Drain proof integrity

### ShutdownScheduler::isFullyDrained() 最終条件（実測抜粋）

```text
intentQueue_.sizeApprox()==0            [authoritative: queue 実測]
observeDeferredRing_.size()==0          [authoritative: queue 実測]
quarantineFallbackQueue_.sizeApprox()==0[authoritative: queue 実測]
recoveryIntentQueue_.size()==0          [authoritative: queue 実測]
retireBacklogCount_==0                  [KEEP/reserved: test 注入可能なため恒真でない]
publicationBacklogCount_==0             [untouched — scope 外]
publicationIntentResidencyCount_==0     [authoritative: INV-X5-1 reservation]
pendingIntentCount_==0                  [authoritative: push fetchAdd/pop fetchSub]
reclaimInFlightCount_==0                [authoritative: wired event]
quarantineIntentResidencyCount_==0      [authoritative: INV-X6-4]
quarantineRingResidencyCount_==0        [authoritative: INV-X6-4]
!recoveryAdmissionPending_              [authoritative: INV-X1-1/2]
```

→ D101-32-E 指示の authoritative 条件リストと完全一致。

### 削除3 domain の「semantic quantity カバー確認」（単なる条件消失でないこと）

| 削除された quantity | 代わりの authoritative measurement | 確認 |
|---|---|---|
| fallback backlog | `quarantineFallbackQueue_.sizeApprox()==0`（同一 drain 条件内）＋ Layer 1 overflow ring resident | ✅ |
| deferred retire residency | `observeDeferredRing_.size()==0`（同一 drain 条件内）＋ router 実測 | ✅ |
| Coordinator quarantine resident | Layer 1 `DSPQuarantineManager::residentCount()`（Threading.cpp dspQuarantineResident） | ✅ |

→ **drain proof の情報量欠落なし**。削除項目は恒真（writer ゼロ）だったことが
D101-32-C で確定済みであり、本監査で writer 不存在を再確認。

---

## E-4. Layer 1 / Layer 2 二層 integrity

`git diff HEAD --stat` の結果:

```
AudioEngine.Threading.cpp           → 差分なし（Layer 1 isFullyDrained 未変更）
AudioEngine.Retire.cpp              → 差分なし（onReclaimBegin/End wiring 未変更）
AudioEngine.Processing.ReleaseResources.cpp → +10/-3（D101-31-D 以前からの B0-6/X6 コメント系
                                              変更。D101-32-D は未触碰）
```

Layer 1 の7実測 source of truth を全件存続確認:
pendingRetireCount / LifetimeState::pendingIntentCount / overflow ring resident /
DSPQuarantineManager::residentCount() / EpochControl quarantine resident / Terminal resident /
runtimePublicationBridge_.isFullyDrained()

> **明示判定: D101-32-D の Coordinator counter 削除により Layer 1 の実測 authority は
> 一切欠落していない。** 削除対象 counter は Layer 1 から参照されていない（参照していたのは
> Layer 2 drain 条件の恒真項目のみ）。

---

## E-5. Retire / Reclaim / Admission 境界回帰監査

| 項目 | 結果 |
|---|---|
| tryAdmit production caller = Publication(Orchestrator.cpp:69) / Recovery(RebuildDispatch.cpp:319) / Build(AudioEngine.h:4443) のみ、Retire 追加なし | ✅ |
| reclaim accounting: onReclaimBegin/End のみが writer（Coordinator.cpp:211/:224 定義、内部 :693/:704、Retire.cpp:62/66/:356/359）。setReclaimInFlightCount への回帰なし | ✅ |
| AdmissionPackedState: packedState_ API（tryAdmit/release/closeAdmission/joinProducers）未触碰。ISRShutdown.cpp の差分は D101-31-D 分（nextVersion wrap fix + test seam）のみで、D101-32-D による変更なし | ✅ |

補助確認: `invariant_INV3_INV5.cpp` は `coordinator.requestReclaim(...)`（production 公開経路、
内部で onReclaimBegin/onReclaimEnd を駆動）経由で `getReclaimInFlightCount()` 0/1/0 を検証する
方式を維持（:132/:140/:168/:176/:182/:186/:282）。setter 非依存のため影響なし。

---

## E-7. テストコード意味回帰

| 項目 | 結果 |
|---|---|
| 削除 reset（set{FallbackBacklog,ReclaimInFlight,DeferredRetireResidency}Count(0)）| **0 件**（grep count = 0）✅ |
| setRetireBacklogCount(0) | ✅ :326 / :400 / :406 / :409 / :412 / :432 維持 |
| setRetireBacklogCount(1)（drain violation 注入） | ✅ :362 維持 |
| setRetireBacklogCount(9)（Pressure FSM 注入） | ✅ :394 維持 |
| invariant_INV3_INV5.cpp | ✅ 無変更（requestReclaim 方式のため D101-32-D 影響なし） |

---

## E-8. CMake / build target integrity

```
CMakeLists.txt への差分（vs HEAD）= D101-31-D 分のみ:
  - add_executable(AdmissionPackedStateTests ...) 追加ブロック
  - target_compile_definitions(AdmissionPackedStateTests PRIVATE CONVOPEQ_UNIT_TESTS=1) (:276)
ISRSemanticValidationTests (:181) / invariant_INV3_INV5Tests (:213) の
target 定義への変更 = なし ✅
CONVOPEQ_UNIT_TESTS seam（CMakeLists:276/:1842、ISRShutdown.h:19/:375、AudioEngine.h:3622）維持 ✅
AdmissionPackedStateTestAccess（src/tests/AdmissionPackedStateTestAccess.h）存続 ✅
```

---

## E-9. Build

| Config | compile error | link error | artifact check |
|---|---|---|---|
| Debug (`build.bat Debug`) | **0** | **0** | ✅ `[4/4] Checking build artifacts...` PASS |
| Release (`build.bat Release`) | **0** | **0** | ✅ `[4/4] Checking build artifacts...` PASS |

---

## E-10. テスト

| テスト | 結果 |
|---|---|
| #21 AdmissionPackedState | ✅ Passed（内部 12/12） |
| #19 ISRSemanticValidationRejects | ✅ Passed |
| #20 InvariantINV3INV5 | ✅ Passed |
| **全 CTest（Debug）** | ✅ **100% tests passed out of 38**（実数記録・38固定ではない） |

---

## E-11. 最終 Gate

| Gate | 条件 | 判定 |
|---|---|---|
| E1 | 最新 ConvoPeq.md 再生成・基準化 | ✅ 2026-08-25 11:38 版 |
| E2 | fallback resurrection = 0 | ✅ |
| E3 | deferred resurrection = 0 | ✅ |
| E4 | Coordinator quarantine resurrection = 0（別クラス存続を区別済み） | ✅ |
| E5 | reclaim setter resurrection = 0 | ✅ |
| E6 | Layer 1 / Layer 2 drain authority 不変 | ✅ Threading.cpp/Retire.cpp diff なし |
| E7 | Retire / Reclaim / Admission 境界不変 | ✅ tryAdmit 3-path、wired events、packedState_ 契約維持 |
| E8 | Pressure FSM / retire KEEP domain 不変 | ✅ FSM 参照 cpp:14+h:5、注入テスト (0)/(1)/(9) 維持 |
| E9 | Debug + Release build PASS | ✅ error 0 / artifact check PASS |
| E10 | 全 CTest PASS | ✅ 38/38（100%） |

# VERDICT: D101-32-E = **PASS**

---

## 次のステップ（推奨順序どおり）

```
D101-32-E  PASS ← 本報告書
      ↓
D101-32-F  publicationBacklogCount / setPendingIntentCount independent inventory audit
      ↓
D101-33-A  Path B Publication Admission Audit
      ↓
D101-33-B  Path B transaction design
```

D101-33-A は未着手（指示遵守）。publicationBacklogCount_/setPendingIntentCount/
pendingIntentCount_ は本監査でも scope 外として一切触碰していない。
