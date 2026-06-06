# Practical Stable ISR Bridge Runtime — 詳細改修計画 v3

作成日: 2026-06-06
ベース: `doc/work19/notfinished6.md` 検証結果 + `doc/work19/notfinished6_validation_report.md` レビュー改訂
         + `doc/work19/refactoring_plan_v2.md` に対する第三者レビュー指摘
監査者: GitHub Copilot (AI Assistant)
到達率: 85〜90%（レビュー改訂後）

---

## 改訂履歴

| 日付 | 版 | 変更内容 |
| --- | --- | --- |
| 2026-06-06 | 2.0 | 初版。Authority 純化計画（PR-1〜PR-8） |
| 2026-06-06 | **3.0** | レビュー指摘を反映し全面改訂。PR-0 追加、PR-1/PR-2 順序逆転、PR-2/PR-4/PR-6 の過剰改修回避、PR-8 インライン化 |

---

## 0. 現状サマリ

### 到達率: 85〜90%

| レイヤ | 状態 | 残余 |
| --- | --- | --- |
| Legacy Commit Path 除去 | ✅ 完了 | — |
| RuntimeWorld Immutability | ✅ 実装済み | 運用強制は別監査 |
| Coordinator 導入 | ✅ 完了 | — |
| RuntimePublicationOrchestrator | ✅ 完了 | publishWorld() 直呼び散在 |
| RuntimeBuilder / RuntimeWorld | ✅ 完了 | Builder が DSPCore 依存（Authority と Execution Object の混同） |
| CrossfadeAuthority 導入 | ✅ 完了 | evaluateFromWorlds() 未使用、DSPTransition バイパス |
| DSPLifetimeManager 導入 | ✅ 完了 | AudioEngine の façade 止まり |
| PublicationAdmission / PublicationExecutor | ✅ 完了 | — |
| **DSPCore → RuntimeWorld Semantic Authority 移行** | ❌ **未完了** | 7項目（優先度A） |
| **Authority 純化** | ❌ **未完了** | 3項目（優先度B） |
| **整理・分離** | ❌ **未完了** | 4項目（優先度C） |

### Phase 定義（v3.1 改訂）

```
Phase1: Authority Pure Runtime
  → DSPCore を Semantic Authority から外す（判断 Authority の RuntimeWorld 移行）
  → PR-0/PR-2/PR-1/PR-5/PR-4/PR-7 で完了
  → PR-3: Admission DSPCore 直読排除（Authority 修正として完了）
  → PR-3A: 除外（Authority 修正ではない）

Phase2: ISR Runtime 整理
  → Semantic Authority 純化後の型安全性・責務整理フェーズ
  → PR-3A（Execution Path Handle Normalization）が Phase2 の第1タスク
  → C3-LatencyService 分離も Phase2 後半以降
  → C3-WarmupService: No-op（Builder 責務のまま維持）
```

### 根本原則（改訂）

```
Practical Stable ISR Bridge Runtime の目的は
  「DSPCore をコードベースから消すこと」では**ない**。

目的は
  「DSPCore を Semantic Authority から外すこと」
  すなわち
  「判断（Decision）の Authority を RuntimeWorld に移すこと」
  であり、Execution Object としての DSPCore は引き続き必要。
```

この原則を誤ると過剰改修・設計破壊・不要な大規模変更を招く。

### Root Cause（確定）

```text
Root Cause: DSPCore が Semantic Authority（判断権威）として残存
  └─ RuntimeWorld.dspProjection は「権威」ではなく「DSPCore の投影」
  └─ 判断ロジックが DSPCore* を直接参照
  └─ Execution Object としての DSPCore は削除不要
```

---

## 1. 現状のデータフロー（Before）

```text
                    ┌─────────────────────────────┐
                    │ 各 publish 起点              │
                    │ ・Init                       │
                    │ ・PrepareToPlay              │
                    │ ・ReleaseResources           │
                    │ ・Timer                      │
                    │ ・DSPTransition              │
                    │ ・PublicationExecutor        │
                    └──────────┬──────────────────┘
                               │ coordinator.publishWorld()
                               ▼
                    ┌─────────────────────────────┐
                    │ RuntimePublicationCoordinator│
                    │  publishWorld(world)         │
                    │  → seal → publishAndSwap     │
                    └─────────────────────────────┘

  RebuildDispatch
    enqueuePublicationIntentForRuntimeCommit
      → RuntimePublicationOrchestrator::submitPublishRequest()
          → Admission::evaluate(req)           ← req.newDSP は DSPCore*
          → CrossfadeAuthority::evaluateOnly(  ← DSPCore* ベース
              engine_, oldDSP, newDSP)
          → RuntimeBuilder::buildRuntimePublishWorld(  ← DSPCore* ベース
              newDSP, oldDSP, ...)
              → dspProjection を DSPCore から直読
          → PublicationExecutor::publish()
              → coordinator.publishWorld()     ← 再委譲
          → DSPTransition::onPublishCompleted()
              → crossfadeAuthorityRuntime_.registerCrossfade()  ← Authority バイパス
          → engine_.advanceRetireEpoch()       ← AudioEngine 依存
```

---

## 2. ターゲットアーキテクチャ（After）

```text
                    ┌─────────────────────────────────┐
                    │ AudioEngine (Facade)             │
                    │  requestPublication(result)      │
                    └──────────┬──────────────────────┘
                               │ submitPublishRequest()  ← 単一入口
                               ▼
┌─────────────────────────────────────────────────────┐
│ RuntimePublicationOrchestrator (Coordinator)         │
│                                                      │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ Admission         │  │ CrossfadeAuthority      │  │
│  │ evaluate(req)     │  │ evaluateFromWorlds()    │  │
│  │   ← DSPHandle     │  │   ← RuntimeWorld        │  │
│  └───────┬──────────┘  └──────────┬──────────────┘  │
│          │                        │                  │
│          ▼                        ▼                  │
│  ┌──────────────────────────────────────────────┐   │
│  │ RuntimeBuilder                                │   │
│  │ buildRuntimePublishWorld(sealedSnapshot,      │   │
│  │   transitionDecision) → RuntimeWorld          │   │
│  │   ← DSPHandle のみ（DSPCore* 不保持）          │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │                                │
│                     ▼                                │
│  ┌──────────────────────────────────────────────┐   │
│  │ PublicationExecutor                          │   │
│  │ publish(world) → store.publishAndSwap()       │   │
│  │   ← coordinator.publishWorld() を内蔵          │   │
│  └──────────────────┬───────────────────────────┘   │
│                     │                                │
│                     ▼                                │
│  ┌──────────────────────────────────────────────┐   │
│  │ DSPTransition                                 │   │
│  │ onPublishCompleted()                          │   │
│  │   → CrossfadeAuthority::register()            │   │
│  │   → DSPLifetimeManager::activate/handle/      │   │
│  │       retire                                  │   │
│  │   → RetireEpochAuthority::advance()           │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ PublicationEvent                             │   │
│  │ onPublished(world)                            │   │
│  │   → LatencyService::adjust()                  │   │
│  │   → UINotifier::notify()                      │   │
│  │   → WarmupService                             │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 3. 事前準備: PR-0 監査フェーズ（新設）

**PR-1/PR-2 着手前に必須の事前調査。**

### 3.1 Crossfade Decision Input 棚卸し

`CrossfadeAuthority::computeDecision()` および `evaluateFromWorlds()` が参照する全フィールドを一覧化し、`RuntimeWorld.dspProjection` で全てカバーできることを確認する。

| # | タスク | ファイル | リスク |
| --- | --- | --- | --- |
| 0-1 | `computeDecision()` の全分岐で参照している DSPCore フィールドを列挙 | `CrossfadeAuthority.cpp` | 低 |
| 0-2 | 列挙したフィールドが `RuntimeWorld.dspProjection` に存在するか確認 | `RuntimeBuilder.cpp` + `RuntimeBuildTypes.h` | **重要** |
| 0-3 | 不足フィールドがある場合、`RuntimeBuildSnapshot` と `dspProjection` に投影フィールドを追加設計 | `RuntimeBuildTypes.h` | 低 |
| 0-4 | 棚卸し結果を `doc/work19/crossfade_input_inventory.md` に文書化 | 新規 | 低 |
| 0-5 | **buildRuntimePublishWorld() が CrossfadeDecision / TransitionPolicy に依存している箇所を棚卸し**<br>evaluate→build 順序が成立することを確認 | `RuntimeBuilder.cpp` | **重要** |
| 0-Audit | **Admission Input Audit: `PublishRequest` 全フィールドについて Semantic 用途 / Execution 用途を分類し、DSPCore 依存が Semantic 経路に存在しないことを確認** | `PublicationAdmission.cpp` + `RuntimePublicationOrchestrator.cpp` | **重要** |
| 0-6 | **evaluateFromWorlds() の参照フィールド一覧と dspProjection のフィールド一覧を自動比較する仕組みを設計**<br>将来の退化防止のため | `CrossfadeAuthority.cpp` + `RuntimeBuildTypes.h` | 低 |
| 0-7 | **getActiveRuntimeDSP() 全使用箇所を Execution用途 / Semantic用途 に事前分類**<br>PR-3/PR-6 の完了条件ベースとなる | `src/audioengine/**` | **重要** |
| 0-8 | **Snapshot Authority Inventory: RuntimeBuildSnapshot に不足している全フィールドを棚卸し**<br>CrossfadeAuthority + Latency系 + PublicationAdmission + その他Decision系を含む | `RuntimeBuildTypes.h` + 各参照元 | **重要** |
| 0-9 | **Observe Source Audit: `getActiveRuntimeDSP` / `getActiveRuntimeDSPHandle` / `setActiveRuntimeDSP` / `exchangeFadingRuntimeDSP` / `resolveActiveRuntimeDSPFromRuntimeWorldOnly` / `resolveFadingRuntimeDSPFromRuntimeWorldOnly` の全使用箇所を Execution用途 / Semantic用途 に分類** | `src/audioengine/**` | **重要** |
| 0-10 | **RuntimeWorld 全フィールド Decision Input Inventory: CrossfadeAuthority 以外の Decision が将来参照し得る RuntimeWorld 全フィールドを棚卸し**（`automation` / `semanticHash` / `coefficient` / `execution` / `overlap` / `routing` / `topology` / `timing` / `resource` 等）。CrossfadeAuthority だけ移行しても別 Decision が DSPCore 参照を再発させるのを防ぐ | `RuntimeBuilder.cpp` + `RuntimeBuildTypes.h` | **重要** |
| 0-11 | **CrossfadeAuthority Output Inventory: Decision 出力（`needsCrossfade` / `fadeTimeSec`）の伝搬先を一覧化**<br>RuntimeBuilder / DSPTransition / PublicationExecutor への影響を事前把握 | `RuntimePublicationOrchestrator.cpp` + `DSPTransition.h` + `RuntimeBuilder.cpp` | **重要** |
| 0-12 | **Handle Resolution Authority の決定: DSPHandle → DSPCore 解決は RuntimePublicationOrchestrator が担当する**<br>Admission は判断 Authority、DSPTransition は Build 前に解決が必要なため不適切。Orchestrator が唯一の適切な解決責任者 | `RuntimePublicationOrchestrator.cpp` | **重要** |
| 0-13 | **Decision Candidate Inventory: 将来 Decision 化され得る RuntimeWorld 全フィールドを棚卸し**<br>現在参照していなくても、将来 Decision 入力として使われる可能性のある全フィールドを列挙 | `RuntimeBuildTypes.h` + `RuntimeBuilder.cpp` | 中 |
| 0-14 | **CrossfadeAuthority Dual-Path Audit: `evaluateOnly()` と `evaluateFromWorlds()` の判定結果を比較**<br>`needsCrossfade` / `fadeTimeSec` について両者の結果が常に一致することを確認。不一致がある場合は PR-1 切替え時にクロスフェード条件が変わる危険があるため、事前に把握する | `CrossfadeAuthority.cpp` | **重要** |
| 0-15 | **Builder Projection Coverage Audit: `RuntimeWorld.dspProjection` の各フィールドについて「供給元 DSPCore メンバ → Snapshot 格納先 → Build 使用箇所」のマッピングが 100% 存在することを確認**<br>CrossfadeAuthority 監査だけでは PR-2 の完全性を保証できない。Builder 側の全投影フィールドに対するマッピング監査が必須 | `RuntimeBuilder.cpp` + `RuntimeBuildTypes.h` | **重要** |

**本タスクをスキップして PR-1 に着手すると機能退化が発生する。**

**0-5 の詳細**: 現実装では `buildRuntimePublishWorld(..., TransitionPolicy policy, double fadeTimeSec, bool active)` と `CrossfadeDecision` が分離不可能な可能性がある。棚卸し結果に応じて「build前軽量判定」と「build後本判定」の2段階設計を検討する。

**0-6 の詳細（Gate 条件）**: `evaluate()` が参照する全フィールドが `dspProjection` に全て存在すること（被覆率100%）を機械的に検証する機構を導入する。注意: フィールド数の一致ではなく「参照フィールドが供給フィールドに包含されていること」が検査条件。現状 `evaluate()` は `irLoaded` / `structuralHash` / `oversamplingFactor` の3フィールドのみ参照し、`dspProjection` は6フィールドを供給する（`irLoaded` / `irFinalized` / `structuralHash` / `oversamplingFactor` / `sampleRate` / `baseLatencySamples`）。供給 > 利用 が正常である。

**実装方式**: C++ では関数内部でどのフィールドを参照したかをコンパイル時に抽出できないため、コンパイル時アサーションは現実的でない。代わりに現行 ConvoPeq ソースの既存方式（`FieldDescriptor` / `AuthorityInventory` / `validateDecisionCoverageContract()`）に準拠し、CrossfadeAuthority 側と dspProjection 側のフィールド記述を一元管理する。**専用ファイル（例: `CrossfadeAuthorityCoverageContract.h`）を新設するか、既存 AuthorityInventory 系に統合するかは Gate-0 で決定する**。いずれの方式でも `kDecisionRelevantFieldNames` 方式で `static_assert(validateDecisionCoverageContract())` により**供給フィールド一覧と参照フィールド一覧の包含関係**を保証する。この自動検査は **Gate-0 の完了条件** として位置づけ、PR-1 着手前に実装しておく。

**0-7 の詳細**: 現状 `getActiveRuntimeDSP()` は Semantic 用途（Crossfade 判定入力）と Execution 用途（DSP スロット操作）が混在。PR-3/PR-6 の完了条件として Semantic 用途ゼロを目指すため、事前に全使用箇所を分類しておく。

**0-8 の詳細**: 現状 `RuntimeBuildSnapshot` は `generation` / `buildInput` / `convolverFingerprint` / `rebuildFingerprint` のみ。PR-2 で追加が必要なフィールドを網羅的に棚卸しする。棚卸しは以下の2文書に分離する:

1. **`doc/work19/decision_input_inventory.md`**: CrossfadeAuthority が判断入力として参照する全フィールド（DSPCore 由来: `irLoaded` / `structuralHash` / `oversamplingFactor` 等 + Atomic 設定値: `m_osFadeTimeSec` / `m_irFadeTimeSec` / `m_irLengthFadeTimeSec` / `m_phaseFadeTimeSec` / `m_directHeadFadeTimeSec` / `m_nucFilterFadeTimeSec` / `m_tailFadeTimeSec` 等）
2. **`doc/work19/snapshot_authority_inventory.md`**: RuntimeBuildSnapshot に追加が必要な全投影フィールド（CrossfadeAuthority + Latency系 + PublicationAdmission + その他 Decision 系）

**0-9 の詳細**: Semantic Leakage の主要候補である4つの DSP 操作系 API を事前に全使用箇所調査し、Execution（DSP スロット操作 / ハンドル管理）と Semantic（判断入力 / Authority 参照）に分類する。この結果が PR-3/PR-6 の完了条件ベースとなる。

**0-10 の詳細**: 将来の新たな Decision が DSPCore を直読することを防ぐため、RuntimeWorld の全フィールド（`dspProjection` / `topology` / `routing` / `execution` / `automation` / `coefficient` / `overlap` / `semanticHash` / `timing` / `resource` / `generationSemantic` 等）について「現在の供給元が DSPCore か Snapshot か」を一覧化する。CrossfadeAuthority だけ移行しても別の Decision が DSPCore 参照を再発させる構造的リスクを排除する。dspProjection フィールドについては別途「供給元 DSPCore メンバ → Snapshot 格納先 → Build 使用箇所」のマッピング表も作成する。

**0-11 の詳細**: `CrossfadeAuthority::Decision { needsCrossfade, fadeTimeSec }` が現在どこへ伝搬しているかを追跡する。少なくとも `buildRuntimePublishWorld(TransitionPolicy, fadeTimeSec, active)` と `DSPTransition::onPublishCompleted(decision)` と `PublicationExecutor` に影響する。この伝搬経路を Output Inventory として文書化し、PR-1 で評価経路を RuntimeWorld 化した際の影響範囲を管理する。

**0-12 の詳細（再評価済み）**: PR-3 は Authority 修正として完了し、残務は低優先度 PR-3A へ分離された。PR-3A 実施時には `PublishRequest` の `void* newDSP` → `DSPHandle` 変更が必要となる。その際、Handle から DSPCore への解決を誰が担当するか決定する必要がある。Admission は判断 Authority であり Handle 解決は責務外。DSPTransition は Build 前に DSPCore が必要なため解決が遅すぎる。したがって `RuntimePublicationOrchestrator` が唯一の適切な解決責任者。ただし `resolveDSPHandle()` のような public API 追加は Handle→DSPCore 逆変換の拡散リスクがあるため、Orchestrator だけが解決可能な構造（friend または内部委譲）を PR-3A 着手時に設計する。

**0-13 の詳細（Current Decision Input Inventory に縮退）**: CrossfadeAuthority は既に `dspProjection` のみで判断できており、新規型（CrossfadeDecisionProjection 等）は不要。「将来 Decision 化され得る全フィールド」は定義不能であるため、以下の現実的な範囲に縮退する: **現在 RuntimeWorld に存在する全フィールドについて、Decision 入力として利用されているか（現在利用中 / 利用予定なし）のみを棚卸しする**。「将来利用可能性あり」の3値分類は行わない。対象フィールド: `dspProjection` / `topology` / `routing` / `execution` / `automation` / `coefficient` / `overlap` / `semanticHash` / `timing` / `resource` / `generationSemantic`。

**0-14 の詳細（Dual-Path Audit）**: 現ソースでは `evaluateOnly(DSPCore*)` と `evaluateFromWorlds(RuntimeWorld)` が共存している。両者の判定結果（`needsCrossfade` / `fadeTimeSec`）が常に一致することを確認する。不一致がある場合、PR-1 で `evaluate(RuntimePublishWorld)` に切り替えた瞬間にクロスフェード条件が変わる危険がある。少なくとも以下の4ケースで比較すること: (a) IR未ロード→ロード、(b) IR構造変更（hash変化）、(c) Oversampling変更、(d) 両方 null。

**0-15 の詳細（Builder Projection Coverage Audit）**: PR-2 の本体は `dspProjection` の DSPCore 直読除去である。CrossfadeAuthority 監査（0-14）だけでは Builder 側の全投影フィールドに対する完全性を保証できない。`RuntimeWorld.dspProjection` の各フィールドについて「供給元 DSPCore メンバ → Snapshot 格納先 → Build 使用箇所」のマッピングが 100% 存在することを確認する。この監査により PR-2 完了時に「dspProjection 全フィールドが Snapshot 由来であること」を確実にする。

**0-Audit の詳細（Admission Input Audit）**: `PublishRequest` の全フィールド（`newDSP` / `generation` / `sealedSnapshot`）について、各フィールドが Semantic 用途（判断入力）と Execution 用途（Builder/Transition への引数）のどちらで使われているかを分類する。実コード調査の結果、`req.newDSP` の全5使用箇所は Execution 用途のみであり、Semantic 用途は既にゼロ（3-4 で Admission の DSPCore 直読を sealedSnapshot 経由に修正済み）。この監査結果は PR-3 の位置付け判断（Authority修正 or 型安全性改善）の根拠となる。

### 3.2 PR-0 を Gate-0 として位置づける

PR-0 は単なる「事前調査」ではなく、**後続の全 PR の設計を確定するゲート** として位置づける。PR-0 の終了報告および承認後にのみ PR-2/PR-3 の着手を許可する。

**PR-0 (Gate-0) の完了条件**:

1. Crossfade Decision Input Inventory 完了
2. Snapshot Authority Inventory 完了
3. **RuntimeBuildSnapshot 追加フィールド確定**（PR-2 成立条件）
   - **現時点で判明している追加候補（全て PR-2 で確定するものではなく、候補リスト）**:
     - `irLoaded`（DSPCore::convolverRt().isIRLoaded() → Snapshot）— CrossfadeAuthority が参照
     - `irFinalized`（DSPCore::convolverRt().isIRFinalized() → Snapshot）— dspProjection に存在
     - `structuralHash`（DSPCore::convolverRt().getStructuralHash() → Snapshot）— CrossfadeAuthority が参照
     - `oversamplingFactor`（DSPCore::oversamplingFactor → Snapshot）— CrossfadeAuthority が参照
     - `sampleRate`（DSPCore::sampleRate → Snapshot）— dspProjection に存在
     - `baseLatencySamples`（engine.estimateRuntimeLatencyBaseRateSamples() → Snapshot）— **CrossfadeAuthority は未参照。LatencyService 用の可能性あり。PR-2 での追加が確定的ではないため、棚卸し後に必要性を判断**
   - 上記以外に不足が発見された場合は `snapshot_authority_inventory.md` に追記
4. **0-6 自動比較機構 実装完了**: `evaluate()` の参照フィールドが全て `dspProjection` に存在すること（被覆率100%）を担保する機械的検査。実装方式は現行 ConvoPeq 方式（`FieldDescriptor` / `AuthorityInventory` / `validateDecisionCoverageContract()`）に準拠。注意: フィールド数一致ではない（供給 > 利用 が正常）
5. **Dual-Path Audit 完了**（evaluateOnly vs evaluateFromWorlds 一致確認）
6. **publishWorld 呼び出し8箇所分類完了**
7. **PR-4/PR-7 実施順決定**: Admission の deferred 状態の実態に基づき、`PR-3→PR-7→PR-4` と `PR-3→PR-5→PR-4→PR-7` のいずれかを確定
8. Observe Source Audit 完了
9. Handle Resolution Authority 決定完了（Orchestrator 担当確定）
10. Decision Candidate Inventory 完了
11. **`getActiveRuntimeDSP()` Semantic 用途ゼロの Gate 条件確立**: Observe Source Audit の結果を Gate 条件として形式化
12. **Builder Projection Coverage Audit 完了**: dspProjection 全フィールドの供給元→Snapshot→Build マッピング 100% 確認

### 3.3 build → evaluate 順序の成立性（ソース調査結果）

**結論: `build → evaluate` 順序は成立するが、`evaluate → build → update` の3段階がより安全。**

#### 調査結果

現実装の `RuntimePublicationOrchestrator::trySubmit()` の順序:

```
1. evaluateOnly(engine_, oldDSP, newDSP)       ← DSPCore ベースの Crossfade 判定
2. buildRuntimePublishWorld(newDSP, oldDSP,     ← TransitionPolicy/fadeTimeSec/active を入力として受け取る
     cfDecision.needsCrossfade ? SmoothOnly : HardReset,
     cfDecision.fadeTimeSec, cfDecision.needsCrossfade, &snapshot)
3. executor_.publish()
4. transition_.onPublishCompleted()
5. engine_.advanceRetireEpoch()
```

`buildRuntimePublishWorld()` は以下のフィールドに CrossfadeDecision を反映する:

- `worldOwner->execution.transitionPolicy = static_cast<int>(policy);`
- `worldOwner->execution.transitionActive = active;`
- `worldOwner->overlap.fadeTimeSec = fadeTimeSec;`

これらは publish 前に更新可能な RuntimeWorld フィールドである。したがって以下の2案が成立する。

#### 選択肢A: evaluate → build → update → publish（推奨）

```
1. evaluateFromWorlds(oldWorld, temporaryViewFromSnapshot)  ← 判定のみ
2. buildRuntimePublishWorld(...)  ← 判定結果を入力としてビルド
3. publish
```

**課題**: `evaluateFromWorlds()` は既に `oldWorld.dspProjection` / `newWorld.dspProjection` のみを参照しており、`routing` / `resource` / `timing` / `topology` / `generationSemantic` は一切使われていない。`CrossfadeDecisionProjection` のような新規型は不要。API は現ソースの実態に合わせ **`evaluate(const DSPProjection& oldProj, const DSPProjection& newProj)`** に固定する（RuntimePublishWorld 案は不採用）。理由: (1)現ソースの実態と一致、(2)RuntimePublishWorld(RuntimeState)が非デフォルト構築可能のため仮World生成が困難、(3)テスト容易。

#### 選択肢B: build → evaluate → update → publish（代替案）

```
1. buildRuntimePublishWorld(..., HardReset, 0.0, false)  ← デフォルト値で仮ビルド
2. evaluateFromWorlds(oldWorld, tempWorld)  ← 投影値を比較して判定
3. tempWorld.execution.transitionPolicy = policy  ← 判定結果を反映
4. tempWorld.execution.transitionActive = active
5. tempWorld.overlap.fadeTimeSec = fadeTimeSec
6. publish
```

**課題**: 仮ビルドの無駄が発生する。また DSPCore から投影値を取得した後に CrossfadeDecision を反映する二重構築になる。

#### 推奨

**選択肢A** を採用する。PR-2 により snapshot 由来の投影値で判定可能になった後、PR-1 で `evaluate() = old evaluateFromWorlds()` に切り替える。この際 `computeDecision()` の fadeTime パラメータ（`m_osFadeTimeSec`, `m_irFadeTimeSec` 等）は engine atomic からの読み取りであり、DSPCore 非依存のため問題ない。

```diff
+ 決定: 選択肢A（evaluate → build → publish）を採用
+ 決定: evaluateFromWorlds は oldWorld (runtimeStore.observe()) と
+       sealedSnapshot 由来の投影ビューを比較する
+ 決定: API は evaluate(const DSPProjection&, const DSPProjection&) に固定
```

### 3.3 CrossfadeSemanticView の要否（ソース調査結果）

**結論: CrossfadeSemanticView は不要。**

#### 調査結果

`evaluateFromWorlds()` が `dspProjection` から読み取るフィールド:

| フィールド | 読み取り元 | 備考 |
| --- | --- | --- |
| `dspProjection.irLoaded` | `oldWorld` / `newWorld` | ✅ dspProjection に存在 |
| `dspProjection.oversamplingFactor` | `oldWorld` / `newWorld` | ✅ dspProjection に存在 |
| `dspProjection.structuralHash` | `oldWorld` / `newWorld` | ✅ dspProjection に存在 |

`evaluateFromWorlds()` が engine atomic から読み取るフィールド（DSPCore 非依存）:

| フィールド | 種類 |
| --- | --- |
| `engine.m_osFadeTimeSec` | 設定値（atomic） |
| `engine.m_irFadeTimeSec` | 設定値（atomic） |
| `engine.m_irLengthFadeTimeSec` | 設定値（atomic） |
| `engine.m_phaseFadeTimeSec` | 設定値（atomic） |
| `engine.m_directHeadFadeTimeSec` | 設定値（atomic） |
| `engine.m_nucFilterFadeTimeSec` | 設定値（atomic） |
| `engine.m_tailFadeTimeSec` | 設定値（atomic） |

**判定**: `evaluateFromWorlds()` は既に `dspProjection` のみを参照しており、DSPCore 直読は行われていない。`CrossfadeSemanticView` を導入しても一段ラップに過ぎず、Authority 純化に寄与しない。したがって導入不要。

```diff
- 条件付き: PR-0 棚卸しで dspProjection のフィールドが不足している場合のみ導入
+ 不要: evaluateFromWorlds() は既に dspProjection のみを参照しており、
+       DSPCore 直読は行われていない。導入不要。
```

### 3.4 publishWorld() 呼び出し8箇所の分類

### 3.2 publishWorld() 呼び出し8箇所の分類

`coordinator.publishWorld()` の各呼び出しを以下の3カテゴリに分類する。

| カテゴリ | 定義 | 該当呼び出し候補 |
| --- | --- | --- |
| **RuntimePublication** | 通常の DSP 切り替え publish | Timer, DSPTransition |
| **LifecyclePublication** | 初期化(Bootstrap)・解放(Shutdown)時のライフサイクル publish | Init (Bootstrap), ReleaseResources (Shutdown) |
| **TransitionPublication** | クロスフェード完了後の後処理 publish | notifyTransitionComplete (Timer), PrepareToPlay(特殊ケース) |
| **内部委譲** | Executor 内での publish（Coordinator経由維持） | PublicationExecutor |

分類結果に基づき、PR-4 の統一方針を決定する。Bootstrap/Shutdown は `submitPublishRequest()` 経由が適切でない可能性がある。

| # | タスク | ファイル | リスク |
| --- | --- | --- | --- |
| 0-5 | `publishWorld()` 呼び出し8箇所のカテゴリ分類 | 全呼び出し元 | 低 |
| 0-6 | 分類結果を `doc/work19/publish_calls_classification.md` に文書化 | 新規 | 低 |

---

## 4. 改修計画: PR 分解

### PR-2（先施行）: RuntimeBuilder Snapshot Authority 化

**目標**: `buildRuntimePublishWorld()` の dspProjection 構築を DSPCore 直読から RuntimeBuildSnapshot 経由に変更する。

**なぜ PR-1 より先か**: `evaluateFromWorlds()` が参照する `world.dspProjection` は現状 DSPCore 投影である。PR-2 を先に実施し、投影元を Snapshot に変更してから PR-1 で評価経路を切り替えないと、Authority 純化にならない。

#### 4.1 変更内容 — Semantic Authority のみ、Execution Object は維持

```diff
- 誤った目標: DSPCore* を RuntimeBuilder から完全排除する
+ 正しい目標: dspProjection の値供給源を DSPCore 直読から Snapshot に変更する
+ Execution Object としての DSPCore 引数は維持してよい
```

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 2-1 | `dspProjection` 構築を DSPCore 直読から `RuntimeBuildSnapshot` 経由に変更 | `RuntimeBuilder.cpp:219-224` | **High** |
| 2-2 | `RuntimeBuildSnapshot` に不足フィールドがある場合のみ追加（PR-0 棚卸し結果に基づく） | `RuntimeBuildTypes.h` | 中 |
| 2-3 | DSPCore* 引数は Execution Object として維持（**削除しない**） | `RuntimeBuilder.h` | 低 |
| 2-4 | `buildRuntimePublishWorld()` 内で DSPCore を Execution Object として使用する経路を確認し、変更不要を確認 | `RuntimeBuilder.cpp` | 中 |

#### 4.2 変更後コードイメージ

```cpp
// Before (RuntimeBuilder.cpp:219-224)
worldOwner->dspProjection.irLoaded = current->convolverRt().isIRLoaded();
worldOwner->dspProjection.irFinalized = current->convolverRt().isIRFinalized();
worldOwner->dspProjection.structuralHash = current->convolverRt().getStructuralHash();
worldOwner->dspProjection.oversamplingFactor = static_cast<int>(current->oversamplingFactor);
worldOwner->dspProjection.sampleRate = current->sampleRate;
worldOwner->dspProjection.baseLatencySamples = engine.estimateRuntimeLatencyBaseRateSamples(current, false);

// After: Snapshot を値供給源とする
worldOwner->dspProjection.irLoaded = snapshot.irLoaded;
worldOwner->dspProjection.irFinalized = snapshot.irFinalized;
worldOwner->dspProjection.structuralHash = snapshot.structuralHash;
worldOwner->dspProjection.oversamplingFactor = snapshot.oversamplingFactor;
worldOwner->dspProjection.sampleRate = snapshot.sampleRate;
worldOwner->dspProjection.baseLatencySamples = snapshot.baseLatencySamples;
```

```cpp
// シグネチャは DSPCore* を維持（Execution Object として必要）
// 引数から DSPCore* を排除する改修は行わない
buildRuntimePublishWorld(AudioEngine::DSPCore* current,
                         AudioEngine::DSPCore* next, ...);
```

#### 4.3 完了条件

- [ ] dspProjection の全フィールドが DSPCore 直読ではなく Snapshot 由来である
- [ ] **`dspProjection` 全フィールド（`irLoaded` / `irFinalized` / `structuralHash` / `oversamplingFactor` / `sampleRate` / `baseLatencySamples`）について値供給源を列挙し、DSPCore 直読残存ゼロを確認**
- [ ] **CrossfadeAuthority が参照する `irLoaded` / `structuralHash` / `oversamplingFactor` の100%が Snapshot 由来であること**（かつ将来の拡張に備え dspProjection 全体を Snapshot 化）
- [ ] DSPCore* 引数は Execution Object として維持されている（削除しない）
- [ ] `RuntimeBuildSnapshot` に不足フィールドがあれば追加されている

---

### PR-1（後施行）: CrossfadeAuthority RuntimeWorld 化

**目標**: `evaluateFromWorlds()` を活性化し、DSPCore ベースの判断を廃止する。

**前提**: PR-2 により `dspProjection` の値供給元が Snapshot に変更済みであること。

#### 1.1 oldWorld/newWorld 取得方法の具体設計（新設）

現状の Orchestrator は `engine_.getActiveRuntimeDSP()` で oldDSP を取得するが、PR-1 では oldWorld が必要。
さらに crossfade 判定のタイミングは World 構築前か後かで構造が変わる。

**設計判断**: Crossfade 判定は World 構築前に行う（oldWorld + snapshot projection で判定可能なため）。

```text
1. runtimeStore.observe() で oldWorld を取得
2. sealedSnapshot から投影値を抽出
3. evaluateFromWorlds(oldWorld, snapshotProjection) で判定
4. 判定結果（TransitionPolicy, fadeTimeSec, active）を buildRuntimePublishWorld() に渡す
5. buildRuntimePublishWorld() で newWorld を構築
6. publishWorld(newWorld)
```

また、PR-0 で確認した通り `evaluateFromWorlds()` は既に `dspProjection` のみを参照しており、新規型は不要と確定した。

**API 設計上の注意**: `evaluateFromWorlds()` が実際に参照するのは `dspProjection`（`irLoaded` / `structuralHash` / `oversamplingFactor`）のみであり、`routing` / `resource` / `timing` / `topology` / `generationSemantic` は一切使われていない。また現行 `RuntimePublishWorld` は `RuntimeState` のエイリアスであり `static_assert(!std::is_default_constructible_v)` が入っているため、未構築の lightweight view を生成する `RuntimePublishWorld::fromSnapshot()` は成立しない可能性が高い。

したがって **PR-1 の API 形式は PR-0 終了時点で確定する**。現時点では以下の2案を併記し、PR-0 の調査結果（Crossfade が必要とするフィールド一覧 + RuntimePublishWorld の構築制約）に基づいて選択する:

| 案 | API | コスト | 拡張性 |
| --- | --- | --- | --- |
| **A（最有力候補）** | `evaluate(const DSPProjection& oldProj, const DSPProjection& newProj)` | 低（既存 evaluateFromWorlds から投影抽出を分離するのみ） | 中（新規フィールド追加時は API 変更が必要） |
| **B（現時点では不採用）** | `evaluate(const RuntimePublishWorld& oldWorld, const RuntimePublishWorld& newWorld)` | 高（RuntimePublishWorld の部分構築経路または lightweight view 型の新設が必要） | 高（RuntimeWorld 全体を渡すため将来拡張に強い） |

**推奨方針（Gate-0 完了後に確定）**: 現時点では **案A（DSPProjection ベース）が最有力候補**。理由は現ソースの実態一致・RuntimePublishWorld の非デフォルト構築問題回避・テスト容易性。ただし Gate-0 の 0-10（RuntimeWorld 全フィールド棚卸し）・0-13（Current Decision Input Inventory）完了後に「DSPProjection で十分であること」を確認した上で API を最終確定する。案B は RuntimePublishWorld の構築制約が解決されない限り不採用。

```diff
+ 決定: 新規型（CrossfadeDecisionProjection / CrossfadeSemanticView）は導入しない
+ 決定: 処理順序は「evaluate → build → publish」（選択肢A）を採用
+ 決定: PR-1 API は Gate-0 完了後に確定（現時点では DSPProjection ベースが最有力候補）
+ 決定: RuntimePublishWorld ベースは現時点では不採用
+       （RuntimeState が非デフォルト構築可能のため）
```

#### 1.2 変更内容

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 1-1 | Orchestrator の処理順序を「evaluate → build → publish」に変更（選択肢A） | `RuntimePublicationOrchestrator.cpp` | **中** |
| 1-2 | `runtimeStore.observe()` で oldWorld を取得する経路を追加 | `RuntimePublicationOrchestrator.cpp` | 低 |
| 1-3 | `evaluateFromWorlds(oldWorld, newWorld)` → `evaluate(const DSPProjection&, const DSPProjection&)` に API 変更（Gate-0 完了後に確定。現時点では DSPProjection ベースが最有力候補） | `CrossfadeAuthority.h/.cpp` | 低 |
| 1-4 | `CrossfadeAuthority::evaluateOnly()` / `evaluateAndRegister()` を削除 | `CrossfadeAuthority.h/.cpp` | 低 |
| 1-5 | `CrossfadeAuthority::computeDecision(DSPCore*, DSPCore*)` を削除 | `CrossfadeAuthority.cpp` | 中 |
| 1-6 | DSPCore 直読ロジックが残っていないことを確認 | `CrossfadeAuthority.cpp` | 低 |

#### 1.3 変更後コードイメージ

```cpp
// Before (RuntimePublicationOrchestrator.cpp:34)
auto* oldDSP = engine_.getActiveRuntimeDSP();
CrossfadeAuthority crossfade;
auto cfDecision = crossfade.evaluateOnly(engine_, oldDSP,
    static_cast<AudioEngine::DSPCore*>(req.newDSP));

// After: evaluate は DSPProjection を受け取る（DSPCore は直接参照しない）
// API は DSPProjection ベースに確定。RuntimePublishWorld ベースは不採用
// Step 1: Extract projections for evaluation
auto& oldWorld = *runtimeStore.observe();
auto& oldProj = oldWorld.dspProjection;
auto newProj = DSPProjection::fromSnapshot(req.sealedSnapshot);
auto cfDecision = crossfade.evaluate(oldProj, newProj);

// Step 2: Build world using evaluation result
auto worldOwner = worldBuilder.buildRuntimePublishWorld(newDSP, oldDSP,
    cfDecision.needsCrossfade ? convo::TransitionPolicy::SmoothOnly : convo::TransitionPolicy::HardReset,
    cfDecision.fadeTimeSec, cfDecision.needsCrossfade,
    &req.sealedSnapshot);

// Step 3: Publish
// (executor_.publish が coordinator.publishWorld を呼ぶ)
```

#### 1.4 完了条件

```text
CrossfadeAuthority が参照している全フィールド一覧を作成し、
その全てが dspProjection に存在することを確認する。
不足時のみ追加。

evaluate は DSPCore を直接参照しないこと。
API は Gate-0 完了後に確定（現時点では DSPProjection ベースが最有力候補）。
```

- [ ] Crossfade Decision Input 棚卸し結果と dspProjection のフィールドが一致している（PR-0 実施済み）
- [ ] Gate-0 で確定した API 形式が Orchestrator の主経路として使用されている
- [ ] Crossfade 判定が World 構築前に行われている
- [ ] `evaluateOnly()` / `evaluateAndRegister()` / `computeDecision(DSPCore*)` / `evaluateFromWorlds()` が削除されている
- [ ] **CodeGraph find_callers で `evaluateOnly` / `evaluateAndRegister` / `computeDecision(DSPCore*)` の呼び出し元がゼロであることを確認**
- [ ] **CrossfadeAuthority の public API に DSPCore* 引数が存在しない**（Gate-0 で確定した API 形式のみ。API 表面から DSPCore が消えていることで、将来の DSPCore 直読への回帰を構造的に防止）
- [ ] **CrossfadeAuthority の public ヘッダに `AudioEngine.h` または `DSPCore` 型の include が不要であること**（DSPCore 型へのコンパイル時依存の断絶）
- [ ] DSPCore 直読ロジックが CrossfadeAuthority 内に残存しない
- [ ] 全テストが通過すること

---

### PR-3: Admission DSPCore 直読排除（完了）

**Gate-0 判断**: ✅ **Authority 修正として完了**。残務は低優先度の PR-3A へ分離。

**根拠**:

- `req.newDSP` の全5使用箇所は Execution 用途のみ、Semantic 用途ゼロ
- Admission の DSPCore 直読 (`isIRLoaded/isIRFinalized`) は sealedSnapshot 経由に修正済み（3-4）
- Decision 系クラスの DSPCore 直読ゼロ
- `getActiveRuntimeDSP()` Semantic 用途ゼロ

**完了条件（Authority 修正として）**:

- [x] `PublishRequest` が Semantic Decision Source として DSPCore* を保持しない ✅
- [x] Admission の DSPCore 直読排除 ✅
- [x] 全 Decision 系クラスの DSPCore 直読ゼロ ✅

---

### PR-3A: Execution Path Handle Normalization（Phase2: ISR Runtime 整理）

**位置付け**: Non-Authority Refactoring（型安全性改善）。Phase2（ISR Runtime 整理）の第1タスク。

**Gate-0 結論**: `PublishRequest.newDSP` の全使用箇所は Execution 用途のみで Semantic 用途ゼロ。Authority 修正としては不要。
→ Authority Pure Runtime（Phase1）から除外し、Phase2 の中核リファクタリングとして再定義。

**目的**: `PublishRequest::newDSP` を `void*` → `DSPHandle` に変更し、型安全性を向上させる。
PublishRequest を「Semantic Payload」として純化し、DSP 実体を Handle 経由で間接参照する。

**設計**: ユーザーレビューにより **Option B（DSPHandle 化）を採用**。

##### Option B（採用）: DSPHandle 化

```cpp
// PublishRequest
struct PublishRequest {
    DSPHandle newDSP;       // Handle 経由で DSP 実体を間接参照
    int generation = 0;
    RuntimeBuildSnapshot sealedSnapshot;
};

// Commit 時: handle 事前登録
auto handle = registerDSPHandleForRuntime(newDSP);
req.newDSP = handle;

// Orchestrator: handle → DSPCore* 解決
auto* newDSPResolved = engine_.resolveDSPHandle(req.newDSP);

// AudioEngine: resolve API
inline DSPCore* resolveDSPHandle(DSPHandle handle) noexcept
{
    if (handle.isNull()) return nullptr;
    const auto resolved = dspHandleRuntime_.resolve(handle);
    if (!resolved.valid || resolved.isStale) return nullptr;
    return static_cast<DSPCore*>(resolved.instance);
}
```

##### Option A（不採用）: RuntimeDSPPtr 型導入

`DSPHandle` の代わりに単純なラッパー型 `RuntimeDSPPtr { DSPCore* ptr; }` を導入する案。
型安全性は向上するが、Handle による間接化・lifetime 管理は行われない。
不採用理由: Option B が既存の `DSPHandleRuntime` インフラを活用でき、より強固な安全性を提供するため。

##### 変更内容（Option B）

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 3A-1 | `PublicationAdmission::PublishRequest` の `void* newDSP` → `DSPHandle newHandle` | `PublicationAdmission.h` | 低 |
| 3A-2 | Orchestrator 内の Handle 解決 (`resolveDSPHandle`) と Builder/Transition への DSPCore* 引き渡し | `RuntimePublicationOrchestrator.cpp` | 低 |
| 3A-3 | commit 時に handle 事前登録 (`registerDSPHandleForRuntime`) | `AudioEngine.Commit.cpp` | 低 |
| 3A-4 | AudioEngine に `resolveDSPHandle(DSPHandle) → DSPCore*` 追加 | `AudioEngine.h` | 低 |

**完了条件**:

- [x] `PublishRequest::newDSP` が `DSPHandle` であること（`void*` ではない）
- [x] commit 経路で `registerDSPHandleForRuntime()` が呼ばれていること
- [x] Orchestrator が `resolveDSPHandle()` 経由で DSPCore* を取得していること
- [x] `static_cast<AudioEngine::DSPCore*>(req.newDSP)` がコードベースに存在しないこと
- [x] `AudioEngine::resolveDSPHandle()` が定義されていること
- [x] ビルド通過

---

### PR-5: Crossfade Registration 規約化（優先度A: A7）

**目標**: Crossfade 登録は DSPTransition のみが行うという規約を確立する。CrossfadeAuthority への責務統合は行わない。

**理由**: CrossfadeAuthority の本来責務は Decision Authority である。一方 `registerCrossfade()` は Execution（DSPHandleRuntime の状態遷移）である。両者を CrossfadeAuthority に統合すると責務逆流が発生する。

#### 5.1 変更内容

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 5-1 | `DSPTransition.h:35` の `engine_.crossfadeAuthorityRuntime_.registerCrossfade()` 呼び出しを維持したまま、規約として「registerCrossfade を呼べるのは DSPTransition のみ」を文書化 | `DSPTransition.h`（コメント） | 低 |
| 5-2 | 他の箇所から `registerCrossfade()` が呼ばれていないことを grep + Serena で確認 | 全ファイル | 低 |
| 5-3 | PR-4 完了後に再度確認（publish 経路変更に伴う新規呼び出しがないこと） | 全ファイル | 低 |

```diff
- 誤った目標: CrossfadeAuthority に Registration を統合する
+ 正しい目標: registerCrossfade() の呼び出し権限を DSPTransition のみに規約化する
+             CrossfadeAuthority は Decision Authority に特化する
```

#### 5.2 完了条件

- [ ] `engine_.crossfadeAuthorityRuntime_.registerCrossfade()` の呼び出し元が DSPTransition のみであることが確認されている
- [ ] Decision（CrossfadeAuthority）と Execution（registerCrossfade）の責務が分離されたままである
- [ ] `registerCrossfade()` の呼び出し元が DSPTransition のみであることを grep + Serena で確認する
- [ ] **`registerCrossfade()` が非 DSPTransition 呼び出し元ゼロであること**（friend 指定による保護は必須としない。規約 + 監査による防御で十分。friend 導入は将来の TransitionCoordinator / CrossfadeService 導入時に再設計が発生するため避ける）
- [ ] 規約がコードコメントとして文書化されている

---

### PR-4: publishWorld() 直接呼び出しの統一（優先度A: A6）

**目標**: `coordinator.publishWorld()` の直接呼び出しを整理し、入口を統制する。

**重要**: 事前に PR-0 で分類した「Publication / Bootstrap / Shutdown」の区分に従う。

#### 4.1 分類別方針

| カテゴリ | 方針 |
| --- | --- |
| **RuntimePublication** | `submitPublishRequest()` 経由に変更 |
| **LifecyclePublication** | Bootstrap/Shutdown 専用経路として維持（通常 RuntimePublication と分離） |
| **TransitionPublication** | 専用経路として維持（通常 RuntimePublication と分離） |
| **内部委譲** | PublicationExecutor 内部では `Coordinator::publishWorld()` を維持（store 直結は行わない）。Coordinator が publish transaction boundary を保持するため |

#### 4.2 変更内容

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 4-1 | Publication カテゴリの呼び出しを `submitPublishRequest()` 経由に変更 | 該当ファイル | 中 |
| 4-2 | Bootstrap/Shutdown カテゴリは専用モードまたは現状維持 | 該当ファイル | 低 |
| 4-3 | `PublicationExecutor::publish()` の `coordinator.publishWorld()` は維持（store 直結は行わない） | `PublicationExecutor.cpp` | 低 |
| 4-4 | デッドコード `AudioEngine::publishWorld()` を削除 | `AudioEngine.h` | 低 |

#### 4.3 完了条件

- [ ] Publication カテゴリの `coordinator.publishWorld()` 直接呼び出しがゼロ
- [ ] Bootstrap/Shutdown の専用経路が文書化されている
- [ ] `AudioEngine::publishWorld()` が削除されている
- [ ] 全テストが通過すること

---

## Appendix-A: PR-6 RetireEpochAuthority（本計画では未実施、Phase2 参照用）

本計画からは除外する。参考として設計案のみ記載する。

**背景**: `advanceRetireEpoch()` は単なる `m_epochDomain.advanceEpoch()` の委譲であり、`RetireEpochAuthority` を新設しても責務分離効果は小さい。実装コスト・ファイル増加に対して利益が限定的なため Phase2 送りとした。本 Appendix は将来の実装時の参照用。

### 設計案

```cpp
class RetireEpochAuthority {
    EpochDomain& domain_;  // AudioEngine ではなく EpochDomain を保持
public:
    explicit RetireEpochAuthority(EpochDomain& domain) noexcept : domain_(domain) {}
    uint64_t advance() noexcept { return domain_.advanceEpoch(); }
};
```

AudioEngine が所有し、Orchestrator は `engine_.advanceRetireEpoch()` 経由で操作（現状維持）。

### 完了条件（参考）

- `advanceRetireEpoch()` の Authority が AudioEngine 内の独立クラスに委譲されている
- `RetireEpochAuthority` は AudioEngine への参照を保持しない（EpochDomain& のみ）
- [ ] DSPLifetimeManager の AudioEngine 依存は現状維持（別フェーズ）
- [ ] **Semantic 用途の `getActiveRuntimeDSP()` 残存ゼロ**（Execution 用途のみ許容）

---

### PR-7: Deferred Queue 移設 — Coordinator/Orchestrator 管理へ（優先度B: B3）

**目標**: Deferred publish キューを Coordinator/Orchestrator 管理下に移す。RuntimeStore への移設は行わない。

**理由**: Deferred publish は Admission 結果に直接紐付く状態であり、Store（データ保管）より Orchestrator（処理調整）の管理下が自然。

**依存**: PR-4（publishWorld 統一）完了後でなければならない。理由: Deferred Queue は Publication Admission State の一部であり、Queue ・ Publish入口 ・ Admission が一体のため。publish 入口統一前に移設すると PR-4 で再度 Queue 接続を触る可能性がある。

**順序に関する注意**: 現計画では PR-3 → PR-5 → PR-4 → PR-7 としているが、Deferred Queue が `PublicationAdmission` に残っている場合、PR-4 で入口統一後に Admission を触ることになる。実装観点では **PR-3 → PR-7 → PR-4** の方が差分が小さい可能性がある。この順序は PR-0 の調査結果（Admission の deferred 状態の実態）で最終決定する。PR-0 の Gate-0 報告時に「PR-4/PR-7 の実施順」を明記する。

#### 7.1 変更内容

| # | 変更 | ファイル | リスク |
| --- | --- | --- | --- |
| 7-1 | `RuntimePublicationOrchestrator` に `PendingPublicationQueue` を新設（PublishRequest は DSPHandle 化済み） | `RuntimePublicationOrchestrator.h/.cpp` | 低 |
| 7-2 | `PublicationAdmission::deferredRequest_` / `hasDeferred_` を削除し、Orchestrator 管理のキューを参照 | `PublicationAdmission.h` | 低 |
| 7-3 | Coordinator が Orchestrator のキュー経由で deferred publish を管理 | `RuntimePublicationOrchestrator.cpp` | 低 |

#### 7.2 完了条件

- [ ] `PublicationAdmission` が deferred 状態を保持しない
- [ ] 全 deferred 管理が Orchestrator 管理下にある
- [ ] `PublishRequest` が DSPHandle 化済みであること（PR-3 前提）

---

### サービス分離・副作用整理（優先度C→D: C1〜C4、詳細は 4a 参照）

**重要**: 独立した PR として最後に実施するのではなく、PR-2〜PR-6 の**各 PR 内で副作用を整理しながら実施**する。後回しにすると二度改修になる。ただし C3-Latency は Phase2 後半以降、C3-Warmup は No-op と判断された。

#### 残余整理タスク

| 該当PR | 副次的に整理すべき項目 |
| --- | --- |
| PR-2 (RuntimeBuilder) | レイテンシ調整 (`estimateRuntimeLatencyBaseRateSamples`)、Warmup (`validateWarmup`) |
| PR-4 (publishWorld 統一) | デッドコード (`AudioEngine::publishWorld()`) |
| PR-6 (DSPLifetimeManager) | `enqueueLearningCommand()`、`sendChangeMessage()` の publish pipeline からの分離検討 |

#### 残余整理タスク

| # | タスク | ステータス | 該当PR |
| --- | --- | --- | --- |
| C1 | デッドコード `publishRuntimeStateNonRt()` を削除 | ✅ 完了 | PR-4 |
| C2 | `sendChangeMessage()` / `triggerAsyncUpdate()` の publish pipeline 混在確認 | ✅ 問題なし | — |
| C3-Latency | `LatencyService` 分離設計 | **Phase2後半以降**（Authority 改善効果なし。Audio Thread 再監査リスク大） | — |
| C3-Warmup | `WarmupService` 分離設計 | **実施不要（No-op）**（RuntimeBuilder の自然な責務。分離により凝集度低下） | — |

---

## 4a. 残余タスク優先順位（Phase 定義反映）

### Phase1: Authority Pure Runtime（完了）

| PR | 状態 | 分類 |
| --- | --- | --- |
| PR-0 (Gate-0) | ✅ 完了 | 事前監査 |
| PR-2 (Snapshot Authority) | ✅ 完了 | Authority 純化 |
| PR-1 (CrossfadeAuthority) | ✅ 完了 | Authority 純化 |
| PR-3 (Admission 直読排除) | ✅ 完了（Authority修正） | Authority 純化 |
| PR-5 (Registration 規約化) | ✅ 完了 | Authority 規約化 |
| PR-4 (publishWorld 統一) | ✅ 完了 | 入口統制 |
| PR-7 (Deferred Queue 移設) | ✅ 完了 | Runtime Authority |

### Phase2: ISR Runtime 整理

| 優先度 | タスク | 分類 | 理由 |
| --- | --- | --- | --- |
| **P0** | **PR-3A: Execution Path Handle Normalization** | **型安全性改善** | ✅ **本実装で完了**。`void*` → `DSPHandle` 化 + resolve API + commit 時事前登録。既存 DSPHandleRuntime インフラを活用した Option B を採用。 |
| **P1** | C3-LatencyService 分離 | Optional Refactoring | Phase2後半。Authority 改善効果なし。AudioThread 再監査リスク大。 |
| **P2** | C3-WarmupService 分離 | **No-op** | Builder 責務のまま維持。分離すると凝集度低下。 |

---

## 5. PR 間依存関係（改訂）

```text
PR-0 (事前監査フェーズ: 新設)
  Crossfade Input 棚卸し
  publishWorld 呼び出し分類
  │
  ├──→ PR-2 (RuntimeBuilder Snapshot Authority 化: A3/A4)
  │       │
  │       └──→ PR-1 (CrossfadeAuthority RuntimeWorld 化: A1/A2) ← 順序逆転!
  │               │
  │               └──→ PR-5 (Crossfade Registration 規約化: A7) ← 責務統合せず
  │
  │
  ├──→ PR-4 (publishWorld 統一: A6) ← PR-0 の分類結果が前提（PR-3 非依存）
  │       │
  │       └──→ PR-7 (Deferred Queue 移設: B3) ← PR-4 完了後
  │
  ├── [PR-3: 完了] Admission DSPCore 直読排除（Authority修正クローズ）
  │
  └──→ 副作用整理 (C1完了/C2完了/C3-Latency Phase2/C3-Warmup No-op)
```

**推奨実施順（最終）**:

### Phase1: Authority Pure Runtime（全完了）

1. **PR-0 (Gate-0)** — ✅ 完了
2. **PR-2** — ✅ 完了
3. **PR-1** — ✅ 完了
4. **PR-3** — ✅ 完了（Authority修正）
5. **PR-5** — ✅ 完了
6. **PR-4** — ✅ 完了
7. **PR-7** — ✅ 完了

### Phase2: ISR Runtime 整理

1. **PR-3A: Execution Path Handle Normalization** — ✅ **本実装で完了**

---

## 6. リスクと注意点（改訂）

| # | リスク | 深刻度 | 確率 | 対策 |
| --- | --- | --- | --- | --- |
| R1 | PR-2 の dspProjection 変更で `RuntimeBuildSnapshot` に不足フィールドがある場合、Crossfade 判断が機能退化する | **High** | 中 | PR-0 で全フィールド棚卸しを必須化 |
| R2 | PR-1 を PR-2 より先に実施すると、dspProjection が DSPCore 投影のまま Authority 移行にならない | **High** | 低 | 実施順を PR-2 → PR-1 に固定 |
| R3 | DSPCore* を完全排除しようとすると過剰改修になる | 中 | 中 | 「Execution Object としての DSPCore は維持」を原則化 |
| R4 | Bootstrap/Shutdown の publishWorld 呼び出しを `submitPublishRequest` に無理に統一すると設計がねじれる | 中 | 中 | PR-0 の分類で専用経路を許容 |
| R5 | PR-6 で AudioEngine メソッド削除を要求すると依存が多すぎて破綻する | 中 | 低 | 委譲パターンを許容 |
| R6 | サービス分離を最後の独立PRにすると二度改修になる | 中 | 高い | 各PR内で副作用整理をインライン化 |

---

## 7. 各PRの想定工数（改訂）

| PR | 規模 | 新規ファイル | 変更ファイル | 推定工数（実績ベース） |
| --- | --- | --- | --- | --- |
| PR-0 | 調査 | 4 (文書) | 調査のみ | **1.5日** |
| PR-2 | **大** | 0 | ~8 | **1.5日** |
| PR-1 | 中 | 0 | 4 | **0.75日** |
| PR-3 | ✅ **完了** | 0 | 1 (Admission修正) | **0.25日** |
| PR-3A | 小（任意） | 0 | 3 | **0.5日** |
| PR-5 | 小 | 0 | 2 | **0.25日** |
| PR-4 | 中 | 0 | ~7 | **1.0日** |
| PR-6 | 小 | — (Phase2 送りのため計画から除外) | — | **0日** |
| PR-7 | 小 | 0 | 2 | **0.5日** |
| 副作用整理(C1/C2/C3調査) | — | 0 | ~5 | PR各内 |
| **合計** | — | **〜5** | **〜32** | **〜6日** |

---

## 8. 成功基準（全体・改訂）

1. **RuntimeWorld Authority**: DSPCore* が Semantic Decision Source として使われていない
   - **Orchestrator が Semantic Decision 用途で `getActiveRuntimeDSP()` を使用しない**（Execution 用途は許容）
   - Gate-0 で確定した API 形式が唯一の crossfade 判断経路
   - **`RuntimeWorld.dspProjection` の値供給元が 100% RuntimeBuildSnapshot である**（部分移行による `irLoaded→snapshot, sampleRate→DSPCore` のような混在を禁止）
   - `PublishRequest` が Semantic Decision Source として DSPCore* を保持しない（Gate-0 の Admission 入力棚卸しで必要性確認後）
   - **CrossfadeAuthority の public API に DSPCore* 引数が存在しない**（API 表面からの DSPCore 型の断絶）
   - **CrossfadeAuthority の public ヘッダに `AudioEngine.h` の include が不要であること**（DSPCore 型へのコンパイル時依存の完全断絶）
2. **Single Publication Entry（Publication 系のみ）**: Publication カテゴリの publish が `submitPublishRequest()` 経由に統一
   - Bootstrap/Shutdown は専用経路を許容
3. **Single Registration Authority（規約化）**: `registerCrossfade()` の呼び出し権限が DSPTransition のみに限定
4. **Authority 移譲（本計画では PR-6 を除外、Phase2 送り）**
5. **`getActiveRuntimeDSP()` Semantic 用途ゼロ**: Orchestrator / CrossfadeAuthority 含む全判断経路で Semantic 用途がゼロ、Execution 用途（DSP スロット操作 / ハンドル管理）のみ
6. **デッドコード削除**: `publishRuntimeStateNonRt()` / `AudioEngine::publishWorld()` が削除されている

---

## 9. 各PR完了時のAuthority監査（新設）

各PR完了時に以下を実施し、残存を確認する。

| 監査項目 | 手法 | 該当PR |
| --- | --- | --- |
| `DSPCore*` Semantic 参照の残存 | `grep` + `Serena find_symbol + find_referencing_symbols` | PR-1, PR-2, PR-3 |
| `getActiveRuntimeDSP` 呼び出し残存 | `grep` + `CodeGraph find_callers` | PR-3, PR-6 |
| `getActiveRuntimeDSP` Semantic 用途残存 | `grep` + `Serena find_referencing_symbols` + 用途分類 | PR-3, PR-6 |
| `getActiveRuntimeDSPHandle` Semantic/Execution 分類 | `grep` + `Serena find_referencing_symbols` + 用途分類 | PR-0, PR-3, PR-6 |
| `setActiveRuntimeDSP` 呼び出し残存 | `grep` + `CodeGraph find_callers` + 用途分類 | PR-3, PR-6 |
| `publishWorld` 直接呼び出し残存 | `grep` + `Serena search_for_pattern` | PR-4 |
| `registerCrossfade` 直接呼び出し残存 | `grep` + `Serena search_for_pattern` | PR-5 |
| `advanceRetireEpoch` Authority 確認 | `grep` + `CodeGraph find_callees` | PR-6 |
| ビルド通過確認 | `Build_CMakeTools` (Debug/Release) | 全PR |
| ユニットテスト通過確認 | `ctest` | 全PR |

```text
重要: grep だけでは呼び出し経路の残存を検出できない。
Serena (Find Symbol + Find References) と CodeGraph (find_callers / find_callees) を
併用すること。
```

### 9.1 Authority Regression Gate（新設）

各PR完了時に、以下の指標が「増えていないこと」を確認する。CI ゲートとして自動化することを推奨する。

| 指標 | 現在値 | 目標 | 確認手法 |
| --- | --- | --- | --- |
| DSPCore* を判断入力として読む箇所数 | 調査中（PR-0） | **非増加** | Serena + CodeGraph |
| `publishWorld()` の直接呼び出し箇所数 | 8箇所以上（PR-0確定） | **非増加** | grep + Serena |
| `registerCrossfade()` の DSPTransition 以外からの呼び出し数 | 0（PR-0確認） | **非増加** | grep + Serena |
| `evaluateOnly()` / `computeDecision(DSPCore*)` / `evaluateFromWorlds()` の残存呼び出し数 | 1（Orchestrator の evaluateOnly） | **非増加** | CodeGraph find_callers |
| `evaluate()` 参照フィールド数と `dspProjection` 供給フィールド数の一致度 | 一致（PR-0確認） | **一致維持** | PR-0 0-6 の自動比較機構 |
| `getActiveRuntimeDSP()` の Semantic 利用箇所数 | 調査中（PR-0） | **非増加** | Serena find_referencing_symbols |
| **Decision 系クラス（CrossfadeAuthority / PublicationAdmission / RuntimePublicationOrchestrator / RuntimePublicationValidator）の DSPCore 直読箇所数** | 調査中（PR-0） | **非増加** | Serena + CodeGraph find_callers |

---

## 10. フェーズ完了条件マトリクス（改訂）

| フェーズ | 完了条件 | 依存 |
| --- | --- | --- |
| **PR-0** | Gate-0 完了条件11項目全て完了（Crossfade Input Inventory / Snapshot Authority Inventory / 0-6 自動比較機構 / Dual-Path Audit / publishWorld 分類 / PR-4/PR-7 順序決定 / Observe Source Audit / Handle Resolution Authority / Decision Candidate Inventory / getActiveRuntimeDSP Semantic Gate） | なし（Gate-0） |
| **PR-2** | dspProjection が Snapshot 由来、DSPCore* 引数は Execution Object として維持 | PR-0（Gate-0） |
| **PR-1** | Gate-0 で確定した API 形式が主経路、旧 `evaluateOnly/computeDecision(DSPCore*)` 削除、CodeGraph で呼び出し元ゼロ確認 | PR-2（Gate-0 完了後） |
| **PR-3** | ✅ **Authority 修正として完了**。Admission DSPCore 直読排除、Decision 系 DSPCore 直読ゼロ確認済み | PR-0（Gate-0） |
| **PR-3A** | `PublishRequest::newDSP` を `DSPHandle` に変更（型安全性改善、低優先度・任意） | Gate-0 結果依存（不要判断可） |
| **PR-5** | `registerCrossfade` 呼び出し元が DSPTransition のみ（責務統合せず） | PR-1 推奨 |
| **PR-4** | Publication 系の `publishWorld` 直接呼び出しゼロ、デッドコード削除 | PR-0（PR-3 非依存） |
| **PR-6** | 除外（Phase2 送り、本計画では実施しない） | — |
| **PR-7** | `PublicationAdmission` が deferred 状態を保持しない | PR-4 完了後 |
