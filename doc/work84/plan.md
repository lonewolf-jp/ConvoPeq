# ConvoPeq Runtime OS 改修計画書 — 現状コードベース → basic_plan.md 最終到達点

**策定日**: 2026-07-24
**改訂日**: 2026-07-25（レビュー反映版）
**現状基準**: v0.6.10 (277 source files, ~3.17MB)
**最終到達点**: doc/work84/basic_plan.md 記載の13フェーズ完了状態
**前提設計**: Practical Stable ISR Bridge Runtime（RTは判断しない・所有しない・解放しない / Authority単一 / Observer副作用禁止）

**改訂内容（レビュー反映）**:
- Phase 5 を Publish/Retire/Crossfade の3サブフェーズに分割（5A/5B/5C）し、friend class削除は5Dとして後ろ倒し
- AuthorityToken を nonce 前提の認証型から、最小限の CoordinatorPrivateTag に縮小
- CI 指標を「個数」ベースから「責務・配置」ベースに変更（mutableは RuntimeWorld 配下のみ、Coordinator外部 atomic 減少、循環依存ゼロ）
- RuntimeCapability の導入時期を Phase 12→Phase 2 に前倒し（Builder/Validator/Coordinator 全に影響するため上流が適切）
- RuntimeIntent と RuntimeIntentJournal を分離（Intent は Runtime の責務、Journal は Diagnostics の責務）

---

## 0. 目次

1. 現状定量解析（検証確定値）
2. 最終到達点サマリ
3. ギャップ分析 全13フェーズ
4. 依存関係と実装順序
5. フェーズ別詳細計画
6. CIゲート定義書
7. リスクと対策
8. 進捗トラッキング

---

## 1. 現状定量解析（2026-07-24 コードベース実測）

凡例: `[実測値] / [最終到達点目標値]`

```
AudioEngine.h 4424行 / 4000→3000目標 ❌ 増加トレンド
class/struct 624定義（全ソース.grep "class |struct " 時）
  AudioEngine.h 68 / 50目標
AuthorityClass:: 70件
publishWorld 38件
retire 711件（全ソース合計、含むコメント）
CoordinatorState 23件（コードベース） 7状態
RuntimeStore 20件
AudioEngine.h atomic 235箇所
  retire 24 / crossfade+latency 17 / publication 12 / generation 6 / audioThread 11 / other 170
Thread注釈 388件超（計数方法依存）
MemoryOrder 2313件
  acquire 1075 / release 754 / acq_rel 253 / relaxed 233
Mutex 122件 / 70目標
alignas(64) 60件
is_always_lock_free static_assert 3件
FixedRingBuffer 512/64/4096/4096 SPSCRingBuffer 32/1024
DBG/Logger 375件
noexcept 2168件
constexpr/static_assert 811件
Complexity>20 8件（calcEQResponseCurve 47 / loadFromTextFile 47 / emitShutdownTrace 40 / 他5）
AudioEngine.h include 72（53プロジェクト + 19システム）/ 40目標 ❌ 増加トレンド
TODO/FIXME 0件 ✅
```

### ファイル構成現状

凡例: ✅ 現存 / ❌ 未作成 / ⚠ 別名/別責務で現存

| ファイル | 状態 | 備考 |
|---|---|---|
| **core/** | | |
| RuntimeStore.h | ✅ | Token必須は未実装 |
| EpochDomain.h | ✅ | 64 readers |
| DeletionQueue.h | ✅ | Vyukov MPMC 4096 |
| CommandBuffer.h | ✅ | SPSC 1024 |
| RuntimePublicationCoordinator.h | ✅ | 既存 |
| IEpochProvider.h / IReaderEpochProvider.h / IRetireProvider.h | ✅ | Provider pattern |
| RCUReader.h | ✅ | RAII reader |
| SnapshotCoordinator.h / SnapshotFactory.h / SnapshotAssembler.h | ✅ | Snapshot subsystem |
| GlobalSnapshot.h / FadeEngine.h / SnapshotFadeState.h | ✅ | Crossfade snapshot |
| RuntimeCoordinator.h | ❌ | 作成必要 |
| RuntimeEpoch.h | ❌ | 作成必要 |
| RuntimeDiff.h | ❌ | 作成必要 |
| RuntimeCapability.h | ❌ | 作成必要（Phase 2 で導入） |
| AuthorityToken.h | ❌ | 作成必要（CoordinatorPrivateTag） |
| RuntimeContract.h | ❌ | 作成必要 |
| RuntimeInvariant.h | ❌ | 作成必要 |
| RuntimeSchema.h | ❌ | 作成必要 |
| DiagnosticsDomain.h | ❌ | 作成必要 |
| ObserverEventBus.h | ❌ | 作成必要 |
| DeletionWorker.h | ❌ | 作成必要 |
| **audioengine/** | | |
| CrossfadeAuthority.h | ✅ | dspProjection使用（DSPCore非依存） |
| CrossfadeRuntime.h | ✅ | SPSC 32 |
| TelemetryRecorder.h | ✅ | 512/4096/4096 |
| RuntimePublicationState.h | ✅ | Ledger 12 |
| ISRShutdown.h | ✅ | 11 phases |
| RuntimeHealthMonitor.h | ✅ | 7状態 FSM |
| RuntimeBuilder.h | ⚠ | 現状Builder + Publish混在 |
| RuntimeGraph.h | ✅ | 95行 |
| FrozenRuntimeWorld.h | ✅ | 一部Immutable化済 |
| PublicationExecutor.h | ✅ | Publish責務 |
| ISRRetire.h / ISRRetireRouter.h / ISRRetireRuntimeEx.h | ✅ | Retire pipeline |
| ISRDSPHandle.h | ✅ | 8状態 |
| RuntimePublicationValidator.h | ✅ | 一部実装 |
| RuntimePublicationOrchestrator.h | ✅ | Coordinator |
| RuntimeIntent.h | ❌ | 作成必要 |
| RuntimeIntentJournal.h | ❌ | 作成必要（RuntimeIntent とは分離。Diagnostics の責務） |
| RuntimeGraphBuilder.h | ❌ | 作成必要（RuntimeBuilder→分離） |
| RuntimeCoordinatorFacade.h | ❌ | 作成必要 |
| DeletionWorker.h | ❌ | 作成必要 |

### CIゲート現状

| ADR | 条件 | 判定 | 優先度 |
|---|---|---|---|
| ADR-001 | publishAndSwap→RuntimeCoordinator限定 | ✅ PASS | — |
| ADR-002 | mutable禁止（RuntimeWorld 配下のみ） | ❌ FAIL（全4箇所検出。ただし RuntimeWorld 配下は 0 件） | S |
| ADR-003 | RuntimeGraph.h <250行 | ✅ PASS | — |
| ADR-004 | generation単独比較禁止 / isAfter()必須 | ⚠ NOT_VERIFIED | A |
| ADR-005 | friend class削除（Phase 5D 完了後に評価） | ❌ FAIL（13箇所） | — |
| ADR-006 | Validator10種 + Schema変換テスト | ⚠ 一部実装（5種実装済み、残5種未実装） | A |
| ADR-007 | Diagnostics逆依存禁止 | ❌ NOT_IMPLEMENTED | B |
| ADR-008 | Failure Injection 9系統 | ❌ NOT_IMPLEMENTED | B |
| ADR-009 | 循環依存ゼロ | ⚠ 未検証 | S |
| ADR-010 | Coordinator外部 atomic ≤ 100 | ⚠ 未検証 | A |

---

## 2. 最終到達点サマリ（basic_plan.md全13フェーズ完了時）

### Authority Matrix（最終形）

| コンポーネント | 唯一の責務 | 禁止事項 |
|---|---|---|
| RuntimeIntent | UI/Automation/Preset/IR → Intent変換 | 直接Builder呼び出し |
| RuntimeBuilder | RuntimeWorld構築のみ | Publish/Retire/Crossfade/Store |
| RuntimePublicationValidator | RuntimeWorld検証のみ | 副作用（Publish/Retire/Delete） |
| RuntimeCoordinator | Publish/Crossfade/Retire/Rollback唯一決定 | DSP実行/delete |
| RuntimeStore | RuntimeWorld保持・Atomic Publishのみ | Policy/Crossfade/Retire/Validation |
| Audio Thread | RuntimeWorld読取・DSP実行のみ | 所有/解放/判断 |
| CrossfadeRuntime | CrossfadePlan実行のみ | Decision保持 |
| EpochDomain | Retire/Reclaimのみ | delete |
| DeletionWorker | deleteのみ | 判断 |
| HealthMonitor | 観測/診断/回復/検証のみ | 直接Publish |

### 最終ファイル構成

```
core/
  RuntimeCoordinator.h — Publish/Crossfade/Retire唯一決定 + CoordinatorPrivateTag発行
  RuntimeStore.h — Token必須 publishAndSwap
  RuntimeEpoch.h — 四次元因果 generation/publishSequence/retireSequence/timestampUs
  RuntimeGraph.h — Execution Graphのみ（512byte以下）
  RuntimeDiff.h — Flag 6種 ChangedIR/Latency/Oversampling/EQ/Routing
  RuntimeCapability.h — constexpr Baseline/Full/SafeMode（Phase 2 導入）
  AuthorityToken.h — CoordinatorPrivateTag（最小限の権限タグ。nonce/issuer 不要）
  RuntimeContract.h — schemaVersion + capability + invariant + compatibility
  RuntimeInvariant.h — 7flags Strict
  RuntimeSchema.h — v1/v2変換 + SchemaMigrator
  DiagnosticsDomain.h — Telemetry/Health/EventBus/Journal/Exporter三層
  ObserverEventBus.h — Publish/Retire/Delete禁止
  EpochDomain.h — 64 readers quarantine（現状維持）
  DeletionQueue.h — 4096 Vyukov MPMC（現状維持）
  DeletionWorker.h — delete唯一
  CommandBuffer.h — SPSC 1024（現状維持）
audioengine/
  RuntimeIntent.h — IntentKind + struct RuntimeIntent（Runtime の責務）
  RuntimeIntentJournal.h — Intent Journal（DiagnosticsDomain 内。Diagnostics の責務）
  RuntimeCoordinatorFacade.h — AudioEngine互換API委譲（移行後削除）
  RuntimeGraphBuilder.h — 唯一Builder（RuntimeBuilderから分離）
  CrossfadeAuthority.h — Diffのみ参照（DSPCore直読ゼロ）
  CrossfadeRuntime.h — SPSC 32（現状維持）
  TelemetryRecorder.h — 512/4096/4096（現状維持）
  RuntimePublicationState.h — Ledger 12（現状維持）
  ISRShutdown.h — 11 phases TinyRingBuffer<64>（現状維持）
  RuntimeHealthMonitor.h — FSM 7状態（現状維持）
```

### CIゲート（最終形 — 責務・配置ベース）

```
ADR-001 ⇔ publishAndSwap → RuntimeCoordinator 限定 ✅
ADR-002 ⇔ RuntimeWorld/FrozenRuntimeWorld 配下のみ mutable std::mutex 禁止 ✅
         （World 外の mutable は const correctness の正当な使用として許容）
ADR-003 ⇔ RuntimeGraph.h <250行 + sizeof<=512 static_assert ✅
ADR-004 ⇔ generation 単独比較禁止。isAfter() 必須 ✅
ADR-005 ⇔ friend class 削除（Phase 5D 完了後に評価） ✅
ADR-006 ⇔ 10種の Validator 関数 + Schema変換テスト ✅
ADR-007 ⇔ DiagnosticsDomain.h が AudioEngine.h を include しない ✅
ADR-008 ⇔ 9系統の Failure Injection 自動試験 ✅
ADR-009 ⇔ 循環依存ゼロ（.h 間の相互 include なし） ✅
ADR-010 ⇔ Coordinator 以外の atomic が 100 以下 ✅
```

---

## 3. ギャップ分析 全13フェーズ

凡例: ◎ 現状から直行可 / ○ 事前Phase完了が必要 / △ 部分的に現存 / ✗ 未着手

### Phase 0 — 現状固定（Authority Matrix文書化）
- **ギャップ**: Authority Matrixはdoc/work84/plan.mdに設計図として存在するが、CIで検証可能な形式ではない
- **依存**: なし
- **難易度**: 低
- **リスク**: 低

### Phase 1 — RuntimeIntent層新設
- **ギャップ**: UI/Automation/Preset/IR Loadが直接Builderを呼ぶ経路が複数存在。RuntimeIntent型が存在しない
- **現状確認**: AudioEngineパラメータ変更経路は `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.Commit.cpp` に分散
- **依存**: Phase 0完了
- **難易度**: 中
- **リスク**: 中（既存呼び出し経路すべての置換が必要）

### Phase 2 — RuntimeBuilder純化 + RuntimeCapability導入
- **ギャップ**: RuntimeBuilderは既にPublish/Retire/Storeに依存しない ✅。RuntimeCapability型が存在せず、DSP側がversion分岐を直接記述
- **現状確認**: `RuntimeBuilder.cpp` にpublishWorld/retireIntent/RuntimeStore呼び出しなし ✅。version分岐は `CacheManager.cpp` / `DeviceSettings.cpp` 等に散在（`if (version == 1)` 等）
- **依存**: Phase 1完了
- **難易度**: 低〜中
- **リスク**: 低

### Phase 3 — RuntimePublicationValidator強化
- **ギャップ**: 現状のValidatorは 5種を実装済み。Projection/Latency/Routing/DSPGraph/Crossfade の5種を追加する必要がある
- **現状確認**: `RuntimePublicationValidator.h/.cpp` に211行の実装。5種の検証関数が存在 ✅
- **依存**: Phase 2完了（RuntimeCapability + 純化Builderの出力を受け取る必要あり）
- **難易度**: 中
- **リスク**: 低

### Phase 4 — RuntimeTransaction導入
- **ギャップ**: RuntimeIntent/RuntimeWorld/ValidationResult/CrossfadePlan/RetirePlan/PublishToken を束ねるTransaction型が存在しない
- **現状確認**: `SemanticTransactionState` enum が `ISRRuntimeSemanticSchema.h:540` に存在 ⚠
- **依存**: Phase 1〜3完了
- **難易度**: 高
- **リスク**: 高（既存のPublish経路全体の置換）

### Phase 5A — Publish Authority（Coordinator一本化 第一段階）
- **ギャップ**: RuntimeCoordinator/RuntimeStore にAuthorityが分散。RuntimeCoordinatorが唯一のPublish決定者になる状態がない
- **現状確認**: RuntimePublicationCoordinator は現存。PublishAuthority/RetireAuthority/ShutdownAuthority enum class Grant=1 が存在。Token は最小限の CoordinatorPrivateTag として開始
- **依存**: Phase 4完了（Transaction→Coordinator.Commit() の流れが必要）
- **難易度**: 高
- **リスク**: 高（Publish経路の再配線）

### Phase 5B — Retire Authority（Coordinator一本化 第二段階）
- **ギャップ**: Retire 決定権がCoordinator / EpochDomain / DSPTransition に分散
- **依存**: Phase 5A完了
- **難易度**: 高
- **リスク**: 高（Retire は RT パス。誤った変更は XRUN 直結）

### Phase 5C — Crossfade Authority（Coordinator一本化 第三段階）
- **ギャップ**: Crossfade 決定権がCoordinator / CrossfadeAuthority / Builder に分散
- **依存**: Phase 5A完了
- **難易度**: 中
- **リスク**: 中

### Phase 5D — friend class 削除（Coordinator 一本化完了後）
- **ギャップ**: friend class 13箇所が存続。Authority統合完了後に可視性の問題として整理
- **現状確認**: friend は Authority の問題ではなく**可視性の問題**。Authority統合前に friend を削除すると Accessor（getX/setY/mutable）が大量発生する
- **依存**: Phase 5A-5C完了
- **難易度**: 中
- **リスク**: 中（Accessor 増加の抑制が必要）

### Phase 6 — RuntimeWorld完全Immutable
- **ギャップ**: RuntimeWorld/FrozenRuntimeWorld の const 保証が不完全。**ISRWorld 配下のみが対象**
- **現状確認**: FrozenRuntimeWorld は mutable 0件 ✅
- **依存**: Phase 5A-5C完了
- **難易度**: 中
- **リスク**: 低

### Phase 7 — Crossfade Authority純化
- **ギャップ**: CrossfadeAuthority::evaluate() は既に dspProjection 使用済み。残るは CrossfadePlan 生成責務の境界整理
- **現状確認**: CrossfadeAuthority.cpp:evaluate() は DSPCore 直読なし ✅
- **依存**: Phase 2完了
- **難易度**: 低
- **リスク**: 低

### Phase 8 — Retire Runtime統合
- **ギャップ**: 5層Retireキューのうち2層が設計と乖離
- **現状確認**: DeferredRetireFallbackQueue (vector+mutex, 50MB) ❌ SPSC ring でない
- **依存**: Phase 5B完了
- **難易度**: 高
- **リスク**: 中

### Phase 9 — RuntimeStore簡素化
- **現状確認**: RuntimeStore.h は publishAndSwap のみ。Phase 5A 完了後の確認フェーズ
- **依存**: Phase 5A完了
- **難易度**: 低
- **リスク**: 低

### Phase 10 — Runtime Health拡張
- **現状確認**: RuntimeHealthMonitor 7状態FSM ✅。RecoveryAction/RecoveryOutcome/computeTrend() 実装済み ✅
- **依存**: Phase 5A完了
- **難易度**: 中
- **リスク**: 低

### Phase 11 — Shutdown Pipeline統一
- **現状確認**: ISRShutdown 11 phases ✅ / TinyRingBuffer<64> ✅
- **依存**: Phase 5A, 8完了
- **難易度**: 低〜中
- **リスク**: 低

### Phase 12 — Runtime Schema導入（旧 RuntimeCapability → Phase 2 へ移動済み）
- **ギャップ**: RuntimeSchema v1/v2 変換 + SchemaMigrator。RuntimeCapability は Phase 2 で導入済み
- **依存**: Phase 6完了
- **難易度**: 低
- **リスク**: 低

### Phase 13 — Runtime Generation管理
- **ギャップ**: generation の一元管理が未実装。AudioEngine.h に atomic generation が6箇所分散
- **依存**: Phase 6完了
- **難易度**: 低
- **リスク**: 低

---

## 4. 依存関係と実装順序

> **注意**: basic_plan.md は実装優先度として「Phase 5→Phase 8→Phase 6→Phase 4→Phase 7→Phase 1〜3」を推奨している。しかし本計画では依存関係分析に基づき **Phase 1〜3 → Phase 4 → Phase 5 の直列チェーン**を採用する。理由: Phase 4（RuntimeTransaction）は Phase 1（Intent）・Phase 2（Builder + Capability）・Phase 3（Validator）の成果物を構成要素とするため、Phase 5（Coordinator一本化）は Transaction.Commit() を前提とする。

```
Phase 0（現状固定）
  │
  ├──→ Phase 1（RuntimeIntent）
  │         │
  │         ▼
  │    Phase 2（Builder純化 + RuntimeCapability導入）
  │         │
  │         ▼
  │    Phase 3（Validator強化）
  │         │
  │         ▼
  │    Phase 4（Transaction導入）
  │         │
  │         ▼
  │    Phase 5A（Publish Authority）
  │         │
  │    ┌────┤
  │    │    │
  │    ▼    ▼
  │  5B   5C（Retire / Crossfade Authority）  ← 並列可
  │    │    │
  │    └────┘
  │         │
  │         ▼
  │    Phase 5D（friend class 削除）
  │         │
  │    ┌────┼─────────────────────┐
  │    │    │                     │
  │    ▼    ▼                     ▼
  │  Phase 6（Immutable）   Phase 8（Retire統合）
  │    │                          │
  │    ├──→ Phase 12（Schema）    ├──→ Phase 11（Shutdown）
  │    └──→ Phase 13（Generation）│
  │                               │
  │    Phase 7（Crossfade責務整理）← Phase 2 完了後独立可
  │
  │    Phase 9（Store確認）← Phase 5A 完了後
  │    Phase 10（Health拡張）← Phase 5A 完了後
  ▼
完了
```

### 推奨実装順序

| 優先順位 | Phase | 理由 |
|---|---|---|
| **S-1** | Phase 0 | 全ての前提。Authority Matrixの文書化とCI化 |
| **S-2** | include削減 + 循環依存チェック（ADR-009対応） | 即時改善可能。RuntimeCoordinatorFacade導入の第一歩 |
| **A-1** | Phase 1 (Intent) | 上流から着手。Builderへの入力を統一 |
| **A-2** | Phase 2 (Builder + RuntimeCapability) | Intent受信後、Builder純化 + Capability を上流で導入 |
| **A-3** | Phase 3 (Validator強化) | 純化Builderの出力を検証 |
| **A-4** | Phase 4 (Transaction) | 上記3Phase完了後、Transactionで統合 |
| **A-5** | Phase 5A (Publish Authority) | ★最重要第一段階。Transaction完了後、Publish 権限を Coordinator に集約 |
| **A-6** | Phase 5B + 5C (Retire + Crossfade) | Phase 5A完了後、並列で実行可能 |
| **B-1** | Phase 5D (friend class 削除) | Coordinator一本化完了後に可視性の問題として段階的削除 |
| **B-2** | Phase 7 (Crossfade責務整理) | Phase 2完了後、独立実行可 |
| **B-3** | Phase 8 (Retire統合) | Phase 5B完了後 |
| **B-4** | Phase 6 (World Immutable) | Phase 5A-5C完了後 |
| **C-1** | Phase 9 (Store確認) | Phase 5A完了後の確認フェーズ |
| **C-2** | Phase 10 (Health拡張) | Phase 5A完了後 |
| **C-3** | Phase 11 (Shutdown統一) | Phase 5A + Phase 8 完了後 |
| **C-4** | Phase 12 (Schema) | Phase 6完了後 |
| **C-5** | Phase 13 (Generation) | Phase 6完了後 |

---

## 5. フェーズ別詳細計画

### Phase 0 — 現状固定（Authority Matrix文書化 + 即時対応CI導入）

**目的**: 現状のAuthority分散を可視化し、以後のPhaseで改善するベースラインを固定する。

**作業**:
1. `config/authority_inventory.json`（既存）を拡張し、以下を機械可読形式で網羅
   - Publish呼び出し箇所一覧
   - Retire呼び出し箇所一覧
   - Crossfade判定箇所一覧
   - Atomic Load/Store箇所一覧（コンポーネント別）
   - RuntimeWorld更新箇所一覧
   - 循環依存チェーン一覧
2. CIスクリプト `tools/` に以下を追加
   - `check_circular_includes.sh` — ADR-009 循環依存ゼロ CIゲート自動化
   - `check_mutable_world.sh` — ADR-002 RuntimeWorld配下 mutable のみ CIゲート自動化
   - `check_coordinator_atomic.sh` — ADR-010 Coordinator外部 atomic CIゲート自動化
3. ベースラインスナップショット作成
   - 全メトリクスを `doc/work84/baseline_2026-07-24.json` に保存

**新規CIゲート**（Phase 0追加）:
```bash
# ADR-009: 循環依存ゼロ（.h 間の相互 include を検出）
check_circular_includes: python3 tools/check_circular_includes.py src/ → 0 サイクル

# ADR-002: RuntimeWorld/FrozenRuntimeWorld 配下のみ mutable 禁止
check_mutable_world: grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h src/core/RuntimeWorld.h 2>/dev/null | wc -l → 0

# ADR-010: Coordinator 以外の atomic 減少トレンド
check_coordinator_atomic: grep -rn "std::atomic" src/ --include="*.h" \
  | grep -v "RuntimeCoordinator\|RuntimePublicationCoordinator" | wc -l → 100以下
```

**受け入れ条件**:
- [ ] Authority Inventory JSONが tools/ でパース可能
- [ ] 3つのCIゲートスクリプトが動作し、現状のFAILを正しく検出
- [ ] ベースラインスナップショットが作成済み

**ファイル変更**: `config/authority_inventory.json`（拡張）, `tools/check_*.sh`（新規3ファイル）

---

### S-2: include削減 + 循環依存チェック（ADR-009対応）

**目的**: AudioEngine.hの#include 72ファイルを整理し、循環依存がないことを確認する。

**即時削減対象**（AudioEngine.hから削除してよいinclude — 実測確認済みのみ掲載）:
```cpp
"LatticeNoiseShaper.h"    // DSPCore::adaptiveNoiseShaperメンバで使用。前方宣言＋.cpp移動の可能性のみ
"TruePeakDetector.h"      // 使用箇所が限定的
"LoudnessMeter.h"         // 上に同じ
"SimplePeakLimiter.h"     // include自体は存在
"GenerationManager.h"     // include確認済み。使用箇所確認要
"UltraHighRateDCBlocker.h" // DSPCore::DCBlockerRuntimeStateで使用。削除不可 → 即時削減対象から除外
```

**循環依存チェック**: 全 .h ファイル間の include リレーションを解析し、サイクルが存在しないことを確認

**注意**: include 数は設計品質と相関しないため、**数値目標（66以下）は撤廃**。代わりに「循環依存ゼロ」を CI ゲートとする。

**受け入れ条件**:
- [ ] AudioEngine.hの#include が減少トレンド
- [ ] 循環依存がゼロ（CIゲート PASS）
- [ ] 削除後にDebug/Release両ビルドが通る

---

### Phase 1 — RuntimeIntent層新設

**目的**: UI/Automation/Preset/IR Loadが直接Builderを呼ぶ経路をRuntimeIntentで統一する。

**新規作成ファイル**:
```
src/audioengine/RuntimeIntent.h
  enum class IntentKind { Parameter, Preset, IRLoad, Automation, Host, Midi, Osc, System };
  struct RuntimeIntent { IntentKind kind; variant<...> payload; uint64_t generation; CorrelationId correlation; bool isHighPriority; };
```

**RuntimeIntent と RuntimeIntentJournal の分離**:
- `RuntimeIntent.h`（audioengine/）: Intent の定義と発行。**Runtime の責務**
- `RuntimeIntentJournal.h`（DiagnosticsDomain 内）: Intent の記録/検索/Undo/Replay。**Diagnostics の責務**
- ISR に必要なのは Intent のみ。Journal は Diagnostics の責務であり、分離することで責務が明確になる

**既存ファイル変更**:
- `AudioEngine.RebuildDispatch.cpp` — Intent発行に置換
- `AudioEngine.Commit.cpp` — 同上
- `AudioEngine.Parameters.cpp` — submitRebuildIntent(Structural) の21箇所を RuntimeIntent 発行に置換
- `RuntimeBuilder.h/.cpp` — Intentを受け取るインターフェースに変更

**CIゲート**:
```
grep -r "submitRebuildIntent" src/audioengine/ --include="*.cpp" | grep -v "RuntimeIntent"
→ 0（全てRuntimeIntent経由）
```

**受け入れ条件**:
- [ ] RuntimeIntent.h 作成済み
- [ ] RuntimeIntentJournal.h が RuntimeIntent.h から分離済み
- [ ] 既存のBuilder呼び出し経路がすべてRuntimeIntent経由に置換済み
- [ ] 既存テストが全件PASS

---

### Phase 2 — RuntimeBuilder純化 + RuntimeCapability導入

**補正: RuntimeBuilder に Publish/Retire/Store の呼び出しは既にない。RuntimeCapability を上流で導入する。**

**目的**: RuntimeBuilder の責務境界を文書化し、RuntimeCapability constexpr を導入して DSP 側の version 分岐を消滅させる。

**RuntimeCapability 導入の理由**:
- RuntimeCapability は Builder / Validator / Coordinator 全体に影響する可能性がある
- ただし現状のコードベースでは DSP 機能分岐の迫切性は低い（version 分岐はシリアライズ/キャッシュフォーマット版本が主。付録E-1 参照）
- Phase 2 着手時に、DSP 側に constexpr 分岐が必要な箇所が実際に存在するかを再確認
- **必要でなければ Phase 2 では着手せず、将来必要になった時点で導入**

**新規作成ファイル**:
```
src/core/RuntimeCapability.h
  struct RuntimeCapability { ... };
  inline constexpr kBaselineCapability = Baseline();
  inline constexpr kFullCapability = Full();
```

**既存ファイル変更**:
- `RuntimeBuilder.cpp` — `publishWorld` コメント参照（L416）削除
- `RuntimeBuilder.h` — 責務境界のコメント明確化 + Capability 分岐対応
- DSP各所の `if(version)` → `if constexpr (capability.supportsX)` に置換

**CIゲート**:
```
grep -r "publishWorld|retireIntent|RuntimeStore" src/audioengine/RuntimeBuilder.cpp
→ 0（現状 PASS ✅）
grep -rn "if.*version" src/ --include="*.cpp" | grep -v "if constexpr" | wc -l → 0
```

**受け入れ条件**:
- [ ] RuntimeBuilder が Publish/Retire/Store を呼ばない（現状 ✅）
- [ ] RuntimeCapability.h 作成済み（Phase 2 着手時に DSP 分岐の必要性を確認。不要なら保留）
- [ ] 既存テストが全件PASS

---

### Phase 3 — RuntimePublicationValidator強化

**現状確認**: Validator は 5種の検証関数を実装済み ✅。5種を追加し10種に拡張。

**既存ファイル変更**:
- `RuntimePublicationValidator.h/.cpp` — 検証関数追加
- 新規検証項目:
  - `validateProjection()` — dspProjection の整合性
  - `validateRouting()` — routing の正当性
  - `validateLatency()` — latency 制約
  - `validateDSPGraph()` — DSPGraph 構造
  - `validateCrossfade()` — crossfade 条件

**CIゲート**:
```
grep -c "validateProjection|validateRouting|validateLatency|validateDSPGraph|validateCrossfade" \
  src/audioengine/RuntimePublicationValidator.cpp → 5以上
```

**受け入れ条件**:
- [ ] 5検証項目すべて実装済み
- [ ] Validation失敗時にPublishが確実にブロックされる
- [ ] 既存テストが全件PASS

---

### Phase 4 — RuntimeTransaction導入

**目的**: RuntimeIntent → RuntimePublishWorld → Validation → CrossfadePlan → RetirePlan をRuntimeTransactionとして束ねる。

**新規作成ファイル**:
```
src/core/RuntimeTransaction.h
  struct CrossfadePlan { bool needsCrossfade; double fadeTimeSec; int startDelayBlocks; CrossfadeId id; ... };
  struct RetirePlan { vector<RetireIntent> worldRetires, dspRetires; uint64_t retireEpoch; };
  struct PublishToken { PublicationSequenceId sequenceId; PublicationEpoch epoch; uint64_t mappedGeneration; };
  struct RuntimeTransaction {
    RuntimeIntent intent;
    RuntimePublishWorld world;
    ValidationResult validation;
    CrossfadePlan crossfadePlan;
    RetirePlan retirePlan;
    PublishToken token;
    CorrelationId correlation;
    void freeze() noexcept;
  };
  enum class TransactionResult { Committed, Rejected, RolledBack, Deferred };
```

**CIゲート**:
```
grep -r "mutable|post-publish|in-place" src/core/RuntimeTransaction.h → 0
```

**受け入れ条件**:
- [ ] RuntimeTransaction型が定義済み
- [ ] Commit/Reject/Rollback/Deferred の4状態が実装済み
- [ ] freeze() でWorldが確実にsealされる
- [ ] 既存テストが全件PASS

---

### Phase 5A — Publish Authority（Coordinator一本化 第一段階）

**目的**: RuntimeCoordinator だけが Publish を決定できる唯一の Authority になる。

**新規作成ファイル**:
```
src/core/RuntimeCoordinator.h
  enum class AuthorityLevel { Publish, Retire, Shutdown, Emergency };
  struct CoordinatorPrivateTag { AuthorityLevel level; };  // 最小限の権限タグ
  class RuntimeCoordinator {
    RuntimeTransaction currentTransaction;
    CoordinatorPrivateTag issueTag(AuthorityLevel);
    TransactionResult commit(RuntimeTransaction&& tx);
    void executePublish(CoordinatorPrivateTag, RuntimeStore&, RuntimePublishWorld*);
    TransactionResult tryRollback(RuntimeTransaction& tx);
  };
```

**CoordinatorPrivateTag の設計判断**:
- ISR で本当に欲しいのは **唯一性** であって **認証** ではない
- `AuthorityToken { nonce; issuer; timestamp; }` は Capability System になりすぎ
- `CoordinatorPrivateTag` 程度でも十分。nonce は将来必要になるまで不要
- 現状の `PublishAuthority { Granted = 1 }` から最小限拡張するだけ

**既存ファイル変更**:
- `ISRRuntimePublicationCoordinator.cpp` — RuntimeCoordinator に統合（または削除）
- `RuntimeStore.h` — Token 検証を追加
- `AudioEngine.h` — Publish 経路を RuntimeCoordinator 経由に再配線
- `ISRDebugRuntime.cpp` — ShadowCompare Production 化

**Phase 5A 注意**: `SnapshotCoordinator`（core/SnapshotCoordinator.h）はRT層のFade進行担当であり、Publication Authority ではない。**削除対象ではない**。

**削除対象**（責務移管完了後）:
- `ISRRuntimePublicationCoordinator.h/.cpp`（RuntimeCoordinator へ移行）

**CIゲート（新設）**:
```
grep -r "publishAndSwap" src/ --include="*.cpp" --include="*.h" \
  | grep -v "RuntimeCoordinator" → 0
```

**受け入れ条件**:
- [ ] RuntimeCoordinator.h 作成済み
- [ ] CoordinatorPrivateTag 機構実装済み
- [ ] RuntimeStore.publishAndSwap() が Token 必須になった
- [ ] SnapshotCoordinator との責務境界が文書化されている
- [ ] ShadowCompare が Production ビルドで動作（ISRDebugRuntime に実装済み。追加作業不要。付録E-3 参照）
- [ ] 既存テストが全件PASS

---

### Phase 5B — Retire Authority（Coordinator一本化 第二段階）

**目的**: Retire 決定権を Coordinator に一元化する。

**既存ファイル変更**:
- `DSPTransition.cpp` — Coordinator.executeRetire() 経由に置換
- `EpochDomain.h` — RetireSequence 管理を Coordinator 経由に
- AudioEngine.h — Retire 経路を RuntimeCoordinator 経由に再配線

**CIゲート**:
```
grep -r "retireIntent" src/ --include="*.cpp" \
  | grep -v "RuntimeCoordinator" → 0
```

**受け入れ条件**:
- [ ] Retire 決定が全て RuntimeCoordinator 経由
- [ ] 既存テストが全件PASS
- [ ] XRUN 増加なし

---

### Phase 5C — Crossfade Authority（Coordinator一本化 第三段階）

**目的**: Crossfade 決定権を Coordinator に一元化する。

**既存ファイル変更**:
- Crossfade 決定経路を RuntimeCoordinator 経由に再配線

**CIゲート**:
```
grep -r "endCrossfade" src/ --include="*.cpp" \
  | grep -v "RuntimeCoordinator" → 0
```

**受け入れ条件**:
- [ ] Crossfade 決定が全て RuntimeCoordinator 経由
- [ ] 既存テストが全件PASS

---

### Phase 5D — friend class 削除（Coordinator 一本化完了後）

**目的**: `friend class` 13箇所を段階的に削除する。

**なぜ Phase 5A-5C 完了後か**:
- friend は **Authority の問題ではなく可視性の問題**
- Authority 統合前に friend を削除すると、逆に Accessor（getX()/setY()/mutable）が大量発生する
- Coordinator 一本化後であれば、Accessor 不要の公開 API で代替できる

**対象箇所**:
```
src/core/RuntimePublicationCoordinator.h:30      RuntimePublicationCoordinator<World, Handle, Bridge>
src/core/RuntimeStore.h:48                        RuntimeStore
src/audioengine/AudioEngine.h:142                 AudioEngine
src/audioengine/AudioEngine.h:143                 convo::RuntimeBuilder
src/audioengine/AudioEngine.h:579                 convo::RuntimeBuilder
src/audioengine/AudioEngine.h:2051                AudioEngine
src/audioengine/AudioEngine.h:3428                convo::isr::RuntimePublicationOrchestrator
src/audioengine/AudioEngine.h:3429                convo::isr::PublicationExecutor
src/audioengine/AudioEngine.h:3430                convo::isr::DSPTransition
src/audioengine/AudioEngine.h:3887                NoiseShaperLearner
src/audioengine/AudioEngine.h:3888                EQEditProcessor
src/audioengine/RuntimePublicationState.h:92      RuntimePublicationStateOwner
src/audioengine/RuntimePublicationState.h:101     RuntimePublicationOrchestrator
```

**段階的削除戦略**:
- 第一段階: NoiseShaperLearner, EQEditProcessor → 公開APIで代替可能
- 第二段階: RuntimeBuilder, DSPTransition → RuntimeCoordinatorFacade 経由に置換
- 第三段階: PublicationOrchestrator, PublicationExecutor → Coordinator 経由に置換
- 最終段階: RuntimePublicationCoordinator, RuntimeStore → Coordinator 一本化により自然消滅

**CIゲート**:
```
grep -rn "friend class" src/ --include="*.h" | wc -l → 0（段階的に減少）
```

**受け入れ条件（段階的）**:
- [ ] 第一段階: 13→11
- [ ] 第二段階: 11→9
- [ ] 第三段階: 9→4
- [ ] 最終段階: 0

---

### Phase 6 — RuntimeWorld完全Immutable

**目的**: RuntimeWorld/FrozenRuntimeWorld から mutable を全廃。**ISRWorld 配下のみが対象。**

**対象範囲**:
- FrozenRuntimeWorld.h — mutable 0件 ✅（既に達成）
- RuntimeWorld.h（Phase 5 完了後に作成）— mutable 0件

**ISRWorld 外の mutable（許容）**:
- ISRRetire.h:136 — Retire パイプラインのフォールバック（World とは無関係）
- ConvolverProcessor.h:884 — DSP プロセッサ内部（const correctness の正当な使用）
- DeferredRetireFallbackQueue.h:100 — キュー内部（World とは無関係）

**CIゲート**:
```
grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h src/core/RuntimeWorld.h 2>/dev/null | wc -l → 0
```

**受け入れ条件**:
- [ ] RuntimeWorld/FrozenRuntimeWorld に mutable 0件
- [ ] 全テストPASS

---

### Phase 7 — Crossfade Authority純化

**補正: CrossfadeAuthority は既に dspProjection のみ使用（設計合致）。**

**目的**: CrossfadePlan 生成責務の境界を文書化する。

**CIゲート**:
```
grep -r "DSPCore|dsp->|getConvolverRt|isIRLoaded|getStructuralHash" src/audioengine/CrossfadeAuthority.cpp
→ 0（既に PASS ✅）
```

**受け入れ条件**:
- [ ] CrossfadeAuthority::evaluate() が DSPCore を直接読まない（現状 ✅）
- [ ] CrossfadePlan 生成責務の境界が文書化されている
- [ ] 既存テスト全件PASS

---

### Phase 8 — Retire Runtime統合

**目的**: RetireIntent → EpochDomain → DeletionWorker の一本化。

**既存ファイル変更**:
- `DeferredRetireFallbackQueue.h` — SPSC化またはDeferredDeletionQueueとの統合
- `ISRRetireRuntimeEx.h/.cpp` — 不要なキュー層の削除
- `EpochDomain.h` — retireSequence管理の追加

**CIゲート**:
```
grep -r "aligned_free|mkl_free" src/ --include="*.cpp" | grep -v "DeletionWorker" → 0
```

**受け入れ条件**:
- [ ] 5層Retireキューが設計通り4層+DeferredDeletionQueueに整理済み
- [ ] delete が DeletionWorker のみ
- [ ] 既存テスト全件PASS

---

### Phase 9 — RuntimeStore簡素化

**補正: 既に Publish責務に特化済み。Phase 5A 完了後の確認フェーズ。**

**CIゲート**:
```
grep -c "policy|crossfade|retire|validate" src/core/RuntimeStore.h → 0（既に PASS ✅）
```

**受け入れ条件**:
- [ ] RuntimeStore が Policy/Crossfade/Retire/Validation メソッドを持たない（現状 ✅）
- [ ] 既存テスト全件PASS

---

### Phase 10 — Runtime Health拡張

**残ギャップ**: 自己修復シナリオ（RetireStallRecovery 等）の具体実装。

**CIゲート**:
```
grep -c "RecoveryOutcome" src/audioengine/RuntimeHealthMonitor.cpp → 4以上
```

**受け入れ条件**:
- [ ] 少なくとも1つの自己修復シナリオが実装済み
- [ ] RecoveryBudget が10分窓で枯渇時にCritical遷移する（現状 ✅）

---

### Phase 11 — Shutdown Pipeline統一

**CIゲート**:
```
grep -c "Running|AudioStopped|ObserverDrained|RetireClosed|EpochSettled|ReclaimComplete|EmergencyDrain|VerifyDrained|TimedOut|Failed|ShutdownComplete" \
  src/audioengine/ISRShutdown.h → 11以上
```

**受け入れ条件**:
- [ ] 11段階のShutdownパイプラインが全コンポーネントで一貫使用されている

---

### Phase 12 — Runtime Schema導入

**補正**: RuntimeCapability は Phase 2 で導入済み。本 Phase は Schema のみ。

**新規作成ファイル**:
```
src/core/RuntimeSchema.h
  struct RuntimeSchema { ... };
  class SchemaMigrator { ... };
```

**CIゲート**:
```
grep -c "SchemaMigrator|RuntimeSchema" src/core/ | wc -l → 1以上
```

**受け入れ条件**:
- [ ] RuntimeSchema.h 作成済み
- [ ] v1→v2 変換が実装済み
- [ ] 既存テスト全件PASS

---

### Phase 13 — Runtime Generation管理

**新規作成ファイル**:
```
src/core/RuntimeEpoch.h
  struct RuntimeEpoch { uint64_t generation; uint64_t publishSequence; uint64_t retireSequence; uint64_t timestampUs; ... };
```

**CIゲート**:
```
grep -rn "atomic.*generation|generation.*atomic" src/audioengine/AudioEngine.h | wc -l → 0
```

**受け入れ条件**:
- [ ] RuntimeEpoch.h 作成済み
- [ ] generation 参照が全て RuntimeEpoch 経由
- [ ] isAfter() が generation 単独比較を禁止
- [ ] 既存テスト全件PASS

---

## 6. CIゲート定義書

改訂版CIゲート一覧。**指標は「個数」ベースから「責務・配置」ベースに変更。**

| ID | CI条件 | 導入Phase | コマンド |
|---|---|---|---|
| ADR-001 | publishAndSwap→RuntimeCoordinator限定 | Phase 5A | `grep publishAndSwap src/ \| grep -v RuntimeCoordinator \| wc -l → 0` |
| ADR-002 | RuntimeWorld配下mutable禁止（World外は許容） | Phase 0 | `grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h src/core/RuntimeWorld.h 2>/dev/null \| wc -l → 0` |
| ADR-003 | RuntimeGraph.h <250行 + sizeof<=512 | Phase 0（既存✅） | `wc -l src/audioengine/RuntimeGraph.h → 250未満` |
| ADR-004 | generation単独比較禁止 | Phase 13 | `grep -rn "generation ==" src/ \| grep -v "epoch\|isAfter" \| wc -l → 0` |
| ADR-005 | friend class削除（Phase 5D完了後に評価） | Phase 5D | `grep -rn "friend class" src/ --include="*.h" \| wc -l → 0` |
| ADR-006 | Validator10種 + Schema変換 | Phase 3 | `grep -c "validate\|check" src/audioengine/RuntimePublicationValidator.cpp → 10以上` |
| ADR-007 | Diagnostics逆依存禁止 | Phase 0 | `grep -rn '#include.*AudioEngine.h' src/core/DiagnosticsDomain.h \| wc -l → 0` |
| ADR-008 | Failure Injection 9系統 | Phase 3 | `grep -c "FailureInject\|faultInject" src/tests/ \| wc -l → 9以上` |
| ADR-009 | 循環依存ゼロ（旧: include減少トレンド） | Phase 0 | `python3 tools/check_circular_includes.py src/ → 0 サイクル` |
| ADR-010 | Coordinator外部atomic≤100（旧: 全atomic 減少） | Phase 0 | `grep -rn "std::atomic" src/ \| grep -v "RuntimeCoordinator\|RuntimePublicationCoordinator" \| wc -l → 100以下` |
| PHASE-0 | CIゲート全スクリプト動作 | Phase 0 | `tools/run_all_ci_gates.sh → exit 0` |
| PHASE-1 | RuntimeIntent経路統一 | Phase 1 | `grep -r "submitRebuildIntent" src/audioengine/ \| grep -v "RuntimeIntent" \| wc -l → 0` |
| PHASE-2 | Builder純化 + Capability | Phase 2 | `grep -r "publishWorld\|RuntimeStore" src/audioengine/RuntimeBuilder.cpp \| wc -l → 0` |
| PHASE-3 | Validator全種実装 | Phase 3 | `grep -c "validate" src/audioengine/RuntimePublicationValidator.cpp → 10以上` |
| PHASE-4 | Transaction freeze | Phase 4 | `grep -c "freeze\|sealRecursively" src/core/RuntimeTransaction.h \| wc -l → 1以上` |
| PHASE-5A | Publish Authority 一本化 | Phase 5A | `grep -r "publishAndSwap" src/ \| grep -v RuntimeCoordinator \| wc -l → 0` |
| PHASE-5B | Retire Authority 一本化 | Phase 5B | `grep -r "retireIntent" src/ \| grep -v RuntimeCoordinator \| wc -l → 0` |
| PHASE-5C | Crossfade Authority 一本化 | Phase 5C | `grep -r "endCrossfade" src/ \| grep -v RuntimeCoordinator \| wc -l → 0` |
| PHASE-5D | friend class削除完了 | Phase 5D | `grep -rn "friend class" src/ --include="*.h" \| wc -l → 0` |
| PHASE-6 | World Immutable | Phase 6 | `grep -rn " mutable " src/audioengine/FrozenRuntimeWorld.h src/core/RuntimeWorld.h 2>/dev/null \| wc -l → 0` |
| PHASE-7 | Crossfade DSPCore非依存（既に達成 ✅） | Phase 7 | `grep -r "DSPCore" src/audioengine/CrossfadeAuthority.cpp \| wc -l → 0` |
| PHASE-8 | DeletionWorker唯一delete | Phase 8 | `grep -r "aligned_free\|mkl_free" src/ \| grep -v "DeletionWorker" \| wc -l → 0` |
| PHASE-9 | Store簡素化（既に達成 ✅） | Phase 9 | `grep -c "policy\|crossfade\|retire\|validate" src/core/RuntimeStore.h → 0` |
| PHASE-10 | Health修復 | Phase 10 | `grep -c "RecoveryOutcome" src/audioengine/RuntimeHealthMonitor.cpp → 4以上` |
| PHASE-11 | Shutdown統一 | Phase 11 | `grep -c "ShutdownPhase" src/audioengine/ISRShutdown.h → 11以上` |
| PHASE-12 | Schema導入 | Phase 12 | `grep -c "SchemaMigrator\|RuntimeSchema" src/core/ \| wc -l → 1以上` |
| PHASE-13 | Generation一元管理 | Phase 13 | `grep -rn "atomic.*generation\|generation.*atomic" src/audioengine/AudioEngine.h \| wc -l → 0` |

---

## 7. リスクと対策

| リスク | 影響Phase | 確率 | 影響度 | 対策 |
|---|---|---|---|---|
| Phase 5A Coordinator統合で既存Publish経路を壊す | 5A | 高 | 最大 | ShadowCompare で新旧経路を二重実行。差分検出で回帰をキャッチ |
| Phase 5D friend削除で Accessor/mutable 増加 | 5D | 中 | 中 | 段階的削除。各段階で mutable 増加を CI で監視 |
| include削減でリンクエラー多発 | S-2 | 中 | 中 | 1ファイルずつ削除+ビルド確認の反復。CIで回帰検出 |
| Phase 5B Retire統合中のXRUN増加 | 5B | 中 | 大 | Retireパス変更はISRRetireテストで事前検証 |
| Phase 4 Transaction導入で責務再配分が複雑化 | 4→5A | 中 | 中 | 移行期間は既存経路と新経路の共存を許容 |
| Phase 2 RuntimeCapability導入が上流で影響範囲を広げる | 2 | 中 | 中 | constexpr のみ。既存コードに影響しない段階的移行 |
| 全Phase完了までにコードベースが肥大化 | 全般 | 高 | 中 | Phase 0 ベースラインを半年毎に再取得 |
| 開発リソース（1人）では全Phase完了に数年規模 | 全般 | 高 | 大 | Phase 5A 完了を第一のマイルストーンに |

---

## 8. 進捗トラッキング

| Phase | ステータス | 開始日 | 完了日 | CIゲート | 備考 |
|---|---|---|---|---|---|
| Phase 0 | 📝 未着手 | — | — | 未導入 | 最初に着手 |
| S-2 include削減 | 📝 未着手 | — | — | 未導入 | Phase 0と並行 |
| Phase 1 | 📝 未着手 | — | — | 未導入 | Intent→Builder |
| Phase 2 | 📝 未着手 | — | — | 未導入 | Builder純化 + RuntimeCapability |
| Phase 3 | 📝 未着手 | — | — | 未導入 | Validator強化 |
| Phase 4 | 📝 未着手 | — | — | 未導入 | Transaction導入 |
| Phase 5A | 📝 未着手 | — | — | 未導入 | Publish Authority ★最重要 |
| Phase 5B | 📝 未着手 | — | — | 未導入 | Retire Authority |
| Phase 5C | 📝 未着手 | — | — | 未導入 | Crossfade Authority |
| Phase 5D | 📝 未着手 | — | — | 未導入 | friend class削除 |
| Phase 6 | 📝 未着手 | — | — | 未導入 | World Immutable |
| Phase 7 | 📝 未着手 | — | — | 未導入 | Crossfade責務整理（DSPCore非依存は既に達成 ✅） |
| Phase 8 | 📝 未着手 | — | — | 未導入 | Retire統合 |
| Phase 9 | ✅ 達成済み | — | — | ✅ PASS | Store簡素化（Phase 5A確認フェーズ） |
| Phase 10 | 📝 未着手 | — | — | 未導入 | Health拡張 |
| Phase 11 | 📝 未着手 | — | — | 未導入 | Shutdown統一 |
| Phase 12 | 📝 未着手 | — | — | 未導入 | Schema導入 |
| Phase 13 | 📝 未着手 | — | — | 未導入 | Generation管理 |

凡例: 📝 未着手 / 🔄 進行中 / ✅ 完了 / ❌ 差戻し

### マイルストーン

| Milestone | 定義 | 予測時期 |
|---|---|---|
| **M0** | Phase 0完了 + S-2 + CIゲート基本セット稼働 | Phase 0着手後 |
| **M1** | Phase 1〜3完了（Intent/Builder+Capability/Validator の上流整備） | Phase 1着手後 |
| **M2** | Phase 4, 5A完了（Transaction + Publish Authority ★最重要） | Phase 4着手後 |
| **M3** | Phase 5B-5D完了（Retire/Crossfade Authority + friend削除） | Phase 5A着手後 |
| **M4** | Phase 6〜8完了（Immutable / Crossfade / Retire統合） | Phase 6着手後 |
| **M5** | Phase 10〜13完了（Health/Shutdown/Schema/Generation + 最終CI全PASS） | Phase 10着手後 |

---

## 付録A: 新規作成ファイル一覧

| ファイル | 責務 | 作成Phase |
|---|---|---|
| `src/core/RuntimeCoordinator.h` | Publish/Crossfade/Retire唯一決定 + CoordinatorPrivateTag発行 | Phase 5A |
| `src/core/RuntimeEpoch.h` | 四次元因果 generation/publishSequence/retireSequence/timestampUs | Phase 13 |
| `src/core/RuntimeDiff.h` | Flag ChangedIR/Latency/Oversampling/EQ/Routing | Phase 0（設計文書）→ Phase 4（実装） |
| `src/core/RuntimeCapability.h` | constexpr Baseline/Full/SafeMode | **Phase 2**（上流導入） |
| `src/core/AuthorityToken.h` | CoordinatorPrivateTag（最小限の権限タグ） | Phase 5A |
| `src/core/RuntimeContract.h` | schemaVersion + capability + invariant + compatibility | Phase 0（設計文書）→ Phase 6（実装） |
| `src/core/RuntimeInvariant.h` | 7flags Strict | Phase 6 |
| `src/core/RuntimeSchema.h` | v1/v2変換 + SchemaMigrator | Phase 12 |
| `src/core/DiagnosticsDomain.h` | Telemetry/Health/EventBus/Journal/Exporter三層 | Phase 0（設計文書）→ Phase 10（実装） |
| `src/core/ObserverEventBus.h` | Publish/Retire/Delete禁止 | Phase 10 |
| `src/core/DeletionWorker.h` | delete唯一 | Phase 8 |
| `src/core/RuntimeTransaction.h` | Intent+World+Validation+Plan+Token | Phase 4 |
| `src/audioengine/RuntimeIntent.h` | enum IntentKind + struct RuntimeIntent | Phase 1 |
| `src/audioengine/RuntimeIntentJournal.h` | Intent Journal（DiagnosticsDomain 内） | Phase 10（DiagnosticsDomain 内） |
| ~~`src/audioengine/RuntimeGraphBuilder.h`~~ | ~~RuntimeGraph構築専念~~ | **不要**（調査結果: Builder の責務範疇内で十分。付録E-2 参照） |
| `src/audioengine/RuntimeCoordinatorFacade.h` | AudioEngine互換API委譲（移行後削除） | Phase 0（S-2） |
| `tools/check_circular_includes.sh` | ADR-009 循環依存ゼロ CIゲート | Phase 0 |
| `tools/check_mutable_world.sh` | ADR-002 RuntimeWorld配下 mutable CIゲート | Phase 0 |
| `tools/check_coordinator_atomic.sh` | ADR-010 Coordinator外部 atomic CIゲート | Phase 0 |
| `tools/run_all_ci_gates.sh` | 全CIゲート一括実行 | Phase 0 |

## 付録B: 既存ファイル変更一覧

| ファイル | 変更内容 | 該当Phase |
|---|---|---|
| `src/audioengine/AudioEngine.h` | include削減 / Publish/Retire/Crossfade 経路再配線 / generation統合 | S-2, Phase 5A-5C, 13 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | Intent経由に置換 | Phase 1 |
| `src/audioengine/AudioEngine.Commit.cpp` | Intent経由に置換 | Phase 1 |
| `src/audioengine/RuntimeBuilder.cpp` | Intent受付 + RuntimeCapability分岐対応 | Phase 1, 2 |
| `src/audioengine/RuntimePublicationValidator.cpp` | 検証範囲拡張（10種） | Phase 3 |
| `src/audioengine/CrossfadeAuthority.cpp` | 変更不要（既にdspProjection使用 ✅）責務境界文書化 | Phase 7 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | RuntimeCoordinatorへ移行（Phase 5A完了後削除） | Phase 5A |
| `src/audioengine/ISRDebugRuntime.cpp` | ShadowCompare Production化 | Phase 5A |
| `src/audioengine/RuntimeHealthMonitor.cpp` | Recoverサイクル追加 | Phase 10 |
| `src/audioengine/ISRShutdown.cpp` | パイプライン統一 | Phase 11 |
| `src/core/RuntimeStore.h` | Token必須化 | Phase 5A |
| `src/core/SnapshotCoordinator.cpp` | 変更不要（RT実行層のFade進行担当。削除しない） | — |
| `src/core/EpochDomain.h` | retireSequence管理追加 | Phase 8 |
| `src/core/DeferredRetireFallbackQueue.h` | SPSC ring化または統合 | Phase 8 |
| `config/authority_inventory.json` | 拡張 | Phase 0 |

## 付録C: 削除対象ファイル

| ファイル | 削除Phase | 理由 |
|---|---|---|
| `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` | Phase 5A | RuntimeCoordinatorへ統合 |
| `src/core/SnapshotCoordinator.h/.cpp` | — | **削除しない**（RT実行層のFade進行担当） |
| `src/audioengine/RuntimeCoordinatorFacade.h` | 全Phase完了後 | 移行用。最終的には不要 |

---

## 付録D: レビュー反映サマリ（2026-07-25）

本改訂版で反映したレビュー指摘事項:

| # | 指摘内容 | 対応 |
|---|---|---|
| ① | Phase5が巨大すぎる | Phase 5 を 5A (Publish) / 5B (Retire) / 5C (Crossfade) / 5D (friend削除) の4サブフェーズに分割 |
| ② | AuthorityToken の nonce/issuer は過剰 | CoordinatorPrivateTag に縮小。ISR に必要なのは「唯一性」であって「認証」ではない |
| ③ | friend削除は早すぎる | Phase 5D（Coordinator一本化完了後）に後ろ倒し。可視性の問題であり Authority の問題ではない |
| ④ | mutable CI は誤検知 | RuntimeWorld/FrozenRuntimeWorld 配下のみを対象に変更。World外の mutable は許容 |
| ⑤ | include数の目標は危険 | 「66以下」の数値目標を撤廃。代わりに「循環依存ゼロ」を CI ゲートに変更 |
| ⑥ | atomic数の目標は不適切 | 「全atomic 減少」から「Coordinator 以外の atomic ≤ 100」に変更。Coordinator 内部は正規な使用 |
| ⑦ | RuntimeCapability の導入が遅すぎる | Phase 12→Phase 2 に前倒し。Builder/Validator/Coordinator 全に影響するため上流が適切 |
| ⑧ | RuntimeIntentJournal は分離すべき | RuntimeIntent（Runtime の責務）と RuntimeIntentJournal（Diagnostics の責務）を分離 |
| ⑨ | Phase 7 の修正は妥当 | Builder 側責務整理の方向性を維持 |
| ⑩ | 実装順の依存関係は妥当 | down-top アプローチを維持。リスク最小化の観点で別案も記載 |

---

## 付録E: 要調査事項の確定（2026-07-25 コードベース調査結果）

レビュー後のコードベース調査により、以下の要調査事項を確定した。

### E-1: RuntimeCapability の導入根拠 — 調査結果

**調査**: `src/` 全体で `if.*version` 分岐を検索。

**結果**: 現状のコードベースに存在する version 分岐は以下の通り。

| ファイル | パターン | 用途 |
|---|---|---|
| `CacheManager.cpp:141` | `if (headerV1.version == 1)` | キャッシュフォーマット版本番号 |
| `CacheManager.cpp:159` | `else if (headerV1.version == 2)` | キャッシュフォーマット版本番号 |
| `CacheManager.cpp:280` | `if (header.version < 2)` | キャッシュフォーマットマイグレーション |
| `DeviceSettings.cpp:930` | `if (version == 1)` | 設定シリアライズ格式版本 |
| `MixedPhasePersistentCache.cpp:100` | `if (header.magic != kMagic \|\| header.version != kVersion)` | キャッシュ整合性チェック |
| `ISRRuntimeSemanticSchema.h:540` | `SemanticTransactionState` FSM | Transaction 状態遷移 |

**結論**: 現状の version 分岐は DSP 機能分岐ではなく、**シリアライズ/キャッシュフォーマット版本** が主。RuntimeCapability（constexpr Baseline/Full/SafeMode）は将来の DSP 機能分岐に備えるものであり、**現状コードベースでの迫切性は低い**。

**対応**: Phase 2 での導入は維持するが、**任意フェーズに降格**（Phase 2 と並行して着手可能だが、必須ではない）。Phase 2 着手時に、DSP 側に constexpr 分岐が必要な箇所が実際に存在するかを再確認する。

### E-2: RuntimeGraphBuilder の必要性 — 調査結果

**調査**: `RuntimeBuilder.cpp` が RuntimeGraph をどう構築しているか確認。

**結果**: `RuntimeBuilder.cpp:217` で `engine.makeRuntimeGraphState(engineState)` を呼び出し、`L224` で `worldOwner->graph = graphState` に設定。RuntimeGraph の構築は Engine 側の `makeRuntimeGraphState()` に委譲されており、Builder 自体は「グラフの設定」のみ行っている。

**結論**: RuntimeGraphBuilder は **不要**。RuntimeBuilder の責務範疇内（World 構築の一環として graph を設定）で十分。Phase 2 の「新規作成ファイル」から RuntimeGraphBuilder.h を削除。

### E-3: ShadowCompare の Production 化 — 調査結果

**調査**: ShadowCompare の実装状態を確認。

**結果**: `ISRDebugRuntime.cpp` に完全な実装あり（`recordShadowCompareObservation()` / `emitShadowCompareCadenceReport()`）。`AudioEngine.Commit.cpp:372` から呼び出されている。evidence ファイル（`shadow_compare_cadence.json`）も出力。

**結論**: ShadowCompare は **ISRDebugRuntime に実装済み**。Phase 5A の「ShadowCompare が Production で動作」という受け入れ条件は、ISRDebugRuntime からの分離（ISRDebugRuntime が Production でも有効になること）を意味する。現状は Production ビルドでも動作するため、**追加作業不要**。受け入れ条件を「ShadowCompare が Production ビルドで動作」に修正。

### E-4: SemanticTransactionState との整合性 — 調査結果

**調査**: `ISRRuntimeSemanticSchema.h:540` の `SemanticTransactionState` FSM を確認。

**結果**: `Building → Validated → Committed → Published` と `→ Rejected`（各段階から遷移可）の FSM が定義済み。`isValidSemanticTransactionTransition()` 関数も実装済み。

**結論**: Phase 4 の RuntimeTransaction は、この既存の FSM を基に拡張する。**新しい FSM を作り直すのではなく、既存の `SemanticTransactionState` を `RuntimeTransaction` の内部状態として統合する** 方針で確定。

### E-5: CrossfadePlan 生成責務 — 調査結果

**調査**: `RuntimeBuilder.cpp:276-300` で Builder が Crossfade フィールドをどう設定しているか確認。

**結果**: Builder は `spec.crossfade.*`（startDelayBlocks, dryHoldSamples, dryScaleTarget, firstIrDryCrossfadePending）を読み取り、`RuntimePublishWorld` の `dspProjection` フィールドに設定。CrossfadeAuthority はこの `dspProjection` を参照して判定。

**結論**: CrossfadePlan 生成は **Builder 責務として維持**。Builder は Specification → World の写像を行う責務であり、Crossfade パラメータの写像もその一部。Phase 7 の責務境界文書化で確定。

### E-6: 循環依存の現状 — 調査結果（調査要）

**調査**: `.h` 間の include リレーションを解析する必要がある。

**現状**: Phase 0 の `check_circular_includes.sh` で検出する前提。**Phase 0 着手時に正式検証**。現在は未検証だが、コンパイルエラーが発生していないことから、致命的な循環依存は存在しないと推定。

### E-7: Coordinator 外部 atomic 数 — 調査結果

**調査**: コンポーネント別の `std::atomic` 数を計数。

| コンポーネント | atomic 数 | 備考 |
|---|---|---|
| AudioEngine.h | 235 | 最大。AudioThread の publication/retire/crossfade/generation |
| RuntimeHealthMonitor.h | 40 | 監視用カウンタ |
| ISRShutdown.h | 20 | Shutdown フェーズ管理 |
| ISRRetire.h | 16 | Retire パイプライン |
| CrossfadeRuntime.h | 12 | Crossfade 実行 |
| EpochDomain.h | 10 | Epoch 管理 |
| TelemetryRecorder.h | 9 | テレメトリ |
| CommandBuffer.h | 2 | コマンドバッファ |
| RuntimePublicationCoordinator.h | 1 | nonce のみ |
| RuntimeGraph.h | 0 | — |
| DeletionQueue.h | 0 | — |

**合計**: 約 345 箇所。Coordinator（RuntimePublicationCoordinator + RuntimeCoordinator）を除くと **約 344 箇所**。Phase 5A で Coordinator に atomic を集約した後、AudioEngine.h の 235 が最大の削減対象。

**結論**: ADR-010 の目標「Coordinator 外部 ≤ 100」は現状 344 から大幅な削減が必要。**Phase 5A 完了後に再評価**し、現実的な目標値を設定する。
