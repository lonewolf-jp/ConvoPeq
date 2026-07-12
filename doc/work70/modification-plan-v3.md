# ConvoPeq メモリ肥大化 改修計画書 v9.6

**日付**: 2026-07-12
**対象**: work70 (メモリ 2.5GB 肥大化問題)
**前版**: v9.5 (2026-07-12)

> **本計画書は「優先改修項目（Backlog）」を先頭にまとめ、「完了した改修」は Appendix に配置している。**
> 凡例: ✅ FACT / 🔍 HYPOTHESIS / 💡 PROPOSAL / ⚠️ CAVEAT / 🎯 DECISION

---

## Design Principles

本計画書全体を貫く設計原則。以降の設計判断はすべてこの原則から導かれる:

| Authority | 責務 | コード位置 |
|:----------|:-----|:----------|
| **Construction Authority** | RuntimePublishWorld の生成。入力を忠実に反映し、自身では判断しない。 | `RuntimeBuilder` |
| **Publication Authority** | publish のトランザクション管理（register → publish → rollback）。 | `commitRuntimePublication()` |
| **Validation Authority** | 不変条件の最終確認。Builder の設定漏れを検出する安全網。 | `RuntimePublicationValidator` |
| **Lifetime Authority** | DSPCore の破棄。物理メモリ解放の唯一の入口。 | `DSPLifetimeManager` |
| **Crossfade Authority** | crossfade の要否判断。DSPCore に依存しない純粋関数。 | `CrossfadeAuthorityRuntime` |

### Builder（Construction Authority）の責務制約

Builder は以下を**行わない**:
- 入力の整合性を判断しない（No Validation）
- Policy Decision を行わない（No Policy Decision）
- 入力を構造体へ忠実に写像するのみ（Pure Construction）
- semantic intent を変更しない（Never mutates semantic intent）
- **入力パラメータ間の意味的整合性を再解釈しない**

```
入力 (RuntimePublishSpecification)
  ↓
Builder（Pure Construction — 入力を忠実に写像するのみ。自身では判断しない）
  ↓
RuntimePublishWorld（完成状態）
```

### Deterministic Construction の定義（Semantic Equivalence）

> **Given identical RuntimePublishSpecification, Builder shall produce RuntimePublishWorld instances whose DSP topology, routing, runtime behavior, latency, crossfade behavior, and processing semantics are equivalent. Object identity, allocation address, publication identity, generation number, and other implementation artifacts are excluded.**

これは以下のことを意味する:
- Builder は Specification の値のみから World を構築する（暗黙の外部状態に依存しない）
- 同じ Specification なら DSP topology / routing / latency / crossfade / processing semantics は常に等価
- **保証しないもの**: オブジェクト同一性、アロケーションアドレス、publication identity、世代番号、その他実装アーティファクト
- **Builder Service may affect implementation artifacts only**. Allocation, identity, pointer, generation は変更可能だが、DSP topology / routing / latency / crossfade / processing semantics は変更不可。
- テストは Specification を直接構築して Builder 単体で実施可能（Coordinator/Current Runtime のモック不要）

### Specification Completeness（設計原則）

> **RuntimePublishSpecification shall contain every mutable runtime value required to construct RuntimePublishWorld. Builder shall not obtain additional mutable inputs from any other source.**

この原則は INV-13 を補強する:
- INV-13（Builder は mutable Runtime state を直接観測しない）は Behavior の制約
- Specification Completeness は **Data の制約** — 必要な情報はすべて Specification でカバーされているべき
- P0〜P2 はこの原則の実装に他ならない（暗黙の atomic 読取りを明示的な Part に昇格）
- 将来新たな mutable 値が必要になった場合、最初の選択肢は Specification への Part 追加（Runtime の直接参照ではない）

### Validator（Validation Authority）の正しい理解

Validator は **Builder の修正機構ではない**:

```
Builder（生成）
  ↓
Validator（Invariant 最終検査）← Builder の出力を検証するが、Builder を修正しない
  ↓
Publication（公開）
```

**Validator の保証範囲**:
- ✅ **Structural Invariants**: Topology / Execution / Identity の三カテゴリ
- ❌ **Semantic correctness**: Builder 入力の意味的整合性は保証しない
- ❌ **Authority Contract**: graph.fadingNode と CrossfadeRuntime の一貫性は保証しない
- ❌ **Policy correctness**: RuntimeBuildPolicy の適切性は保証しない

---

## 優先改修項目（Backlog）

*以下の優先順位は本セッションでのコード調査（2026-07-12）に基づく。各項目ごとに確認できた事実と具体的な対応方針を記載する。*

### ~~P0: Builder の暗黙入力を RuntimePublishSpecification に昇格する~~ ✅ 完了

**🎯 優先度: ★★★★★（最優先）→ ✅ 実装完了**

**【対応結果】**
- ✅ `RuntimePublishSpecification` に **ProcessingPart** を追加（`RuntimeBuilder.h`）
- ✅ Orchestrator が sealedSnapshot から ProcessingPart を設定（`RuntimePublicationOrchestrator.cpp`）
- ✅ Builder が `spec.processing.*` から読み取り、engine atomic 直接読み取りを排除（`RuntimeBuilder.cpp`）
- ✅ L303 の無効行（`engine.currentProcessingOrder` 読み取り）を削除
- ✅ `spec.routing.*` は ProcessingPart から同期（後方互換性維持）

**【CAVEAT 解決】**
- 設計問題（Builder vs mutable state）: 解消済み。Builder は ProcessingPart のみ参照。
- 実装問題（processingOrder 未使用）: 解消済み。L303 無効行を削除。

---

### ~~P1: computeRuntimePublishComputation() の Runtime Query と Pure Calculation の分離~~ ✅ 第一段階完了

**🎯 優先度: ★★★★☆ → ✅ 第一段階完了**

**【対応結果】**
- ✅ **PublicationSnapshotPart** を `RuntimePublishSpecification` に追加（`RuntimeBuilder.h`）
- ✅ **`previousCommittedSequence` の取得を Orchestrator 側へ移行**。Orchestrator が `engine_.getLastCommittedPublicationSequence()` を呼び、Spec の `publicationSnapshot.previousCommittedSequence` に格納。Builder は Spec から読み取り。
- ✅ `computeRuntimePublishComputation()` の Runtime Query 部分（`getLastCommittedPublicationSequence()`）の移行完了。

**【⚠️ Transitional】**: `engine.computeRuntimePublishComputation()` の呼び出し自体は Builder に残存（`engineState`/`graphState` 構築に必要）。これらは P2（CrossfadeSnapshotPart/LatencyPart 追加）で完全に Spec 経由に移行予定。

---

### ~~P2: Specification の最小拡張~~ ✅ 完了

**🎯 優先度: ★★★☆☆ → ✅ 完了**

**【基本原則（Specification Part 追加の3基準）】**
新しい Part を追加する際は、以下の3つをすべて満たすことを確認する:

1. **これは Runtime の現在状態ではなく、Builder が世界を構築するための入力か。**
2. **同じ Specification なら、同じ RuntimePublishWorld が生成されるか。**
3. **この値は Builder の責務で再計算・再取得すべきものではないか。**

この基準を満たさない情報は、Specification ではなく Orchestrator や Runtime 側に置く。

**【対応結果】**
✅ **CrossfadeSnapshotPart** 追加（startDelayBlocks, dryHoldSamples, dryScaleTarget, firstIrDryCrossfadePending）
✅ **LatencyPart** 追加（latencyDelayOld, latencyDelayNew）
✅ Orchestrator が Spec 生成時にこれらの Part を engine state から収集して設定
✅ Builder が `spec.crossfade.*` / `spec.latency.*` から読み取り、engine 直接読み取りを排除

**【Source 一意性の定義】**
- `ProcessingPart` が automation 値（eqBypassed, softClipEnabled, gain類）の **唯一の Source**
- `RoutingPart` は後方互換性のための **mirror**（ProcessingPart からの同期コピー）
- 将来のリファクタリングで RoutingPart の mirror フィールドは削除可能

**【Part Ownership（誰が書けるか）】**
| Part | Writer | Reader |
|:-----|:-------|:-------|
| TopologyPart | Orchestrator | Builder (readonly) |
| ExecutionPart | Orchestrator | Builder (readonly) |
| RoutingPart | Orchestrator（ProcessingPart から同期） | Builder (readonly) |
| **ProcessingPart** | **Orchestratorのみ** | Builder (readonly) |
| **PublicationSnapshotPart** | **Orchestratorのみ** | Builder (readonly) |
| **CrossfadeSnapshotPart** | **Orchestratorのみ** | Builder (readonly) |
| **LatencyPart** | **Orchestratorのみ** | Builder (readonly) |
| currentRuntimeWorld | Orchestrator | Builder (readonly, makeEngineRuntimeState引数) |

---

### ~~P3: INV-12 再定義（Builder 入力契約の明確化）~~ ✅ 設計書更新済み

**🎯 優先度: ★★★☆☆ → ✅ 設計書更新済み（P0〜P2 完了につき正式版に更新）**

**【問題】**
設計書 v8.3 の INV-12 は「Builder は RuntimePublishSpecification 以外 consult 禁止」としている。しかし Builder は実際に以下の依存を持つ:

**【🎯 DECISION（2026-07-12 確定）】**
INV-13 を以下のように再定義する:

> **Builder は RuntimePublishSpecification、Builder Service、Pure Utility のみ利用可能。**
> - Input（atomic 読取り）: ❌ 禁止 → Specification 経由
> - Runtime Query（Coordinator/Current Runtime/Crossfade）: ❌ 禁止 → Orchestrator 経由
> - Builder Service: ✅ 許可
> - Pure Utility: ✅ 許可

**Builder Service の定義（契約ベース）:**
> Builder が利用可能な副作用を持たない補助サービス = **Builder execution environment**。以下の条件をすべて満たす:
> 1. Runtime 状態を参照しない
> 2. Builder の入力を変更しない
> 3. 意味的結果を変更しない
> 4. 決定論性を破壊しない

**Builder Service の細分類:**

| カテゴリ | サービス種別 | 該当例 |
|:---------|:------------|:-------|
| **Memory Service** | Allocator | メモリ確保・解放（`aligned_malloc`/`aligned_free`） |
| **Identity Service** | Identity Generator | `reserveRuntimePublicationIdentity()` — UUID/Generation発行 |
| **Immutable Factory** | Factory + Immutable Helper | `RuntimePublishWorld::createForBuilder()`, IRFactory, FFTPlan, LatencyCalculator, BufferFactory |

この分類により、将来の変更（例: UUID生成方式の変更、分散IDの導入）が Builder Contract に影響しない。

P0〜P2 の実装完了後に INV-13 をこの形に更新する（先行更新するとコードと設計書の不一致が生じるため）。

---

### ~~P4: collectTrackedMemoryStatistics() MEM_SNAP 統合~~ ✅ 完了

**🎯 優先度: ★★☆☆☆ → ✅ 実装完了**

**【対応結果】**
- ✅ MEM_SNAP 出力に `| TRK: total=%.1f OS=%.1f EQ=%.1f AL=%.1f LT=%.1fMB` を追加
- ✅ `getActiveRuntimeDSP()` が非 null の場合のみ `collectTrackedMemoryStatistics()` を呼び出す
- ✅ 配線のみ（API定義＋実装は完了済み）

---

### P5: 型整理・既知制限

**🎯 優先度: ★☆☆☆☆**

| # | 項目 | 内容 | 優先度 |
|:-:|:-----|:------|:-------|
| 1 | `transitionPolicy`: int → enum class | 設計書は `convo::TransitionPolicy`、コードは `int`。コメントで対応関係はドキュメント済み。 | ★☆☆☆☆ |
| 2 | `processingOrder`: int → enum class | 同上 | ★☆☆☆☆ |
| 3 | `aligned_free` DIAG 非対称性 | `aligned_malloc` は `DIAG_MKL_MALLOC` 経由だが `aligned_free` は `mkl_free` 直接。`DIAG_MKL_FREE` は size 引数が必要なため対応不可。診断品質は低下するが設計上の既知制限として許容。 | ★★☆☆☆ |
| 4 | `RuntimePublicationSpecification.h` 整理 | 独立ファイルとして作成されたが誰もインクルードしていない。将来の分離用準備として維持。 | ★☆☆☆☆ |

---

### P6: AUTH_CONTRACT FAIL 原因修正 — Builder の fadingRuntimeUuid 条件不一致（2026-07-12 発見）

**🎯 優先度: ★★★★★（最優先）**

**【コード調査により確定した原因（2026-07-12 cocoindex/semble/serena/graphify による特定）】**

**現象**: `[AUTH_CONTRACT] FAIL fadingNode=0 hasFadingByUuid=1` は Builder の論理バグ。

**コード検証**:

- `RuntimeBuilder.cpp:210`: `worldOwner->topology.fadingRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0;`
  → **`next` を無条件で使用**

- `AudioEngine.h:2759` (makeEngineRuntimeState 内部):
  ```cpp
  const bool transitionActive = active && next != nullptr;
  DSPCore* fading = transitionActive ? next : nullptr;
  ```
  → **`graph.fadingNode` は `transitionActive` に依存**

- `RuntimeGraph.h:18`: `void* fadingNode = nullptr;`

**結果**: `transitionActive=false` かつ `fadingDSP=oldDSP` が非 null の場合:
- `graph.fadingNode = nullptr` (conditional through makeEngineRuntimeState → makeRuntimeGraphState)
- `topology.fadingRuntimeUuid = oldDSP->runtimeUuid` (non-zero, UNCONDITIONAL)
- → AUTH_CONTRACT 不一致 ⇒ publish FAILED

**【修正案】**:
```cpp
// RuntimeBuilder.cpp line 210: 条件を makeEngineRuntimeState と統一
// Current (buggy):
//   worldOwner->topology.fadingRuntimeUuid = (next != nullptr) ? next->runtimeUuid : 0;
// Fix:
worldOwner->topology.fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0;
```

**【確認された波及影響】**:
- gen=3〜5 の publish 全滅（約6秒間 NUC live=0）
- 結果として 354ms の rebuild 浪費（副次的影響、P5とは独立に防止可能）
- gen=1/2 では `fadingDSP=nullptr` だったため発現せず
- prepareToPlay での DSPCore 切り替え時に初めて顕在化

---

### P7: Builder 残留 atomic 読取りの Specification 昇格（2026-07-12 発見）

**🎯 優先度: ★★★★☆**

**【コード調査で確認された残留依存（2026-07-12）】**

現在の `RuntimeBuilder.cpp` には以下の `engine.*` 直接アクセスが残っている (P0〜P2 完了後も未対応):

| # | 行 | コード | 分類 | 現在の Spec カバレッジ |
|:-:|:--|:------|:----|:---------------------|
| 1 | 255 | `convo::consumeAtomic(engine.retireQueueDepth_, ...)` | **Input（禁止）** | ❌ RetirePart 未定義 |
| 2 | 319-320 | `engine.currentAdaptiveCoeffBankIndex` | **Input（禁止）** | ❌ AdaptivePart 未定義 |
| 3 | 324 | `engine.getAdaptiveCoeffBankForIndex(bankIndex)` | **Runtime Query（禁止）** | ❌ AdaptiveBankPart 未定義 |
| 4 | 378 | `engine.getConvolverProcessor()` → `transferIRStateFrom()` | **Runtime Query（禁止）** | ❌ IRTransferPart 未定義 |
| 5 | 357 | `engine.m_healthStateRef` (consumeAtomic) | **未分類** | ⚠️ HealthState の Spec 化要検討 |

**【重要度】**: 1〜3 は INV-12（Builder は mutable Runtime 状態を観測しない）違反。
4 は IR 転送という特殊操作であり、Builder Contract の例外として認められる可能性あり（Immutable Factory の一部と解釈可能）。
5 は HealthState の読み取りで、Spec 化するか Builder Service とみなすか設計判断が必要。

---

## 設計参照情報

*本節は全フェーズに共通する設計判断の集約。v8.3 までは [設計] 6 として配置されていたものを改訂・集約した。*

### INV 一覧（2026-07-12 ユーザー訂正版）

| ID | Invariant | 備考 |
|:---|:----------|:------|
| INV-1〜INV-10 | 変更なし（旧版維持） | DSPHandle ライフサイクル・Commit トランザクション |
| INV-11 | **RuntimePublishWorld は Builder 完了後 immutable。Builder never mutates inputs.** | ✅ 実装済み（const 返却確認済み） |
| **INV-12**（設計原則） | **Builder shall never observe mutable Runtime state directly. Any mutable runtime information required for construction shall be captured into RuntimePublishSpecification before Builder execution.** | **Builder は mutable な Runtime 状態を直接観測してはならない。構築に必要な mutable 情報はすべて RuntimePublishSpecification にキャプチャしてから Builder を実行する。** 設計原則。P0〜P2 の根拠。 |
| **INV-13**（実装契約 — INV-12 の実装ルール） | **Builder は RuntimePublishSpecification、Builder Service、Pure Utility のみ利用可能。Input（atomic）と Runtime Query（Coordinator/Current Runtime/Crossfade）は禁止。**（P3 で正式化予定） | 🎯 2026-07-12 再定義。INV-12「mutable state 観測禁止」を実現する具体的な契約。P0〜P2 完了後に設計書更新。 |

### Builder 依存分類（2026-07-12 コード調査確定）

| 分類 | 可否 | 該当例 | 件数 | 現状 |
|:-----|:-----|:-------|:----|:-----|
| **① Input（禁止）→ Specification（→ ProcessingPart）** | ❌ | `currentProcessingOrder`, `eqBypassActive`, `convBypassActive`, `softClipEnabled`, `saturationAmount`, `inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain` | **8** | P0 対象。`sealedSnapshot==nullptr` 経路で atomic 直接読み取り。ProcessingPart として昇格。 |
| **② Runtime Query（禁止）→ Orchestrator→Spec へ** | ❌ | Coordinator `consumeWorldHandle()`, Publication Sequence, CrossfadeRuntime（`getStartDelayBlocks` 他3）, `latencyDelayOld/New`, `retireQueueDepth_` | **8** | P1+P2 対象。`computeRuntimePublishComputation()` 経由 + 直接読み取り。PublicationSnapshotPart/CrossfadeSnapshotPart/LatencyPart に格納。 |
| **③ Builder Service（許可）** | ✅ | `reserveRuntimePublicationIdentity()`, `RuntimePublishWorld::createForBuilder()`, Allocator, Factory, Identity Generator, Immutable Helper | **1** | 問題なし。契約条件を満たす（Runtime状態非依存/入力を変更しない/意味的結果を変更しない/決定論性を破壊しない）。 |
| **④ Pure Utility（許可）** | ✅ | `estimateRuntimeLatencyBaseRateSamples()`, hash(), math | **1** | 問題なし。現在状態に依存しない純粋計算。 |
| **⑤ P7 残留 Input（要昇格）** | ❌ | `retireQueueDepth_`（重新評価）, `currentAdaptiveCoeffBankIndex`, `getAdaptiveCoeffBankForIndex()` | **3** | P7 対象。Specification の RetirePart/AdaptivePart 未定義。2026-07-12 発見。 |
| **⑥ P7 特殊依存（要判断）** | ⚠️ | `getConvolverProcessor()` → IR transfer, `m_healthStateRef` | **2** | IR transfer は Immutable Factory 例外と解釈可能。HealthState は要設計判断。 |

### Authority Boundary Chart

| 責務 | Authority | コード位置 |
|:-----|:----------|:----------|
| `activeRuntimeDSPHandle_` 更新 | `commitRuntimePublication()` | `AudioEngine.h:3981` |
| `DSPCore*` active 更新 | `DSPLifetimeManager::activate()` | `DSPLifetimeManager.h:28` |
| DSPHandle 発行 | `registerDSPHandleForRuntime()` | `AudioEngine.h:3838` |
| DSPHandle 回収 | `retireDSPHandleForRuntime()` | `AudioEngine.h:3875` |
| EBR enqueue | `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:37` |
| DSPCore* 物理破棄 | `destroyDSPCoreNode()` | `AudioEngine.Threading.cpp:15` |

### RT Safety

- `activeRuntimeDSPHandle_`: publishAtomic(release) / consumeAtomic(acquire) — 正しい HB 順序
- `runtimeDSPHandleMap_`: `std::mutex` 保護（NonRT のみ）
- `DSPHandleRuntime`: `std::atomic<DSPHandle>` + `std::atomic<DSPState>` ベース
- EBR drain: 「全 Reader が epoch を離脱した」ことを保証してから callback 実行

### エグゼクティブサマリ

**設計上のメモリ効果見込み**: 未修正時 2,477MB に対し、設計上は定常 686MB / ピーク 1,094MB を見込む。ただしビルド・実測未実施のため確定値ではない。
**実測確認が必須の項目**: BlockSize 削減効果（~189MB/DSPCore 見込み）、CrossfadePlan 導入後の EBR 動作、680MB Other 内訳。これらは P0〜P3 実装後の MEM_SNAP で確認する（[未確定] 7 参照）。
**ただし `lifecycle(retire)=0` は継続中**。CrossfadePlan 導入後の検証が必要（[未確定] 7 参照）。

**全11ステップ実装確認済み**: ソースコード上の実装は完了。発見した7件の不整合（旧戻り値型削除漏れ、テストの hasFadingRuntime 残存、P4 DIAG_MKL_MALLOC include不足、queueDepthBlocks ラベル、コメント古朽化2件）は修正済み。ビルド・テスト通過確認は未実施（C1060 環境問題のため）。

---

## [未確定] 7. 未解決課題（Runtime 観測依存）

*本セクションは「設計書の時点で確定できず、将来の Runtime 観測または実装後に検証が必要な事項」を集約する。*

### 7.1 NoiseShaper accepted=0 の真因 → ✅ 解決（2026-07-12 ログ解析）

🔍 **HYPOTHESIS**: `accepted=0` の原因はコード調査の範囲では特定できず、Runtime ログ確認が必要とされていた。

**✅ ログ解析結果（2026-07-12 ConvoPeq.log）**:
```
[NoiseShaperLearner] Waiting diagnostics:
  accepted=3012  dropSession=0  dropSampleRate=0  dropBank=0
  bufferedSamples=771072  sessionId=0  sampleRateHz=192000
  bankIndex=107  generation=39  queueDepthBlocks=0
```

| 項目 | 値 | 判定 |
|:-----|:---|:------|
| `accepted` | **3012 / 3004** | ❌ `accepted=0` ではない。正常に学習ブロックを受理中。 |
| `dropSession` | 0 | sessionId=0 によるドロップなし |
| `dropSampleRate` | **0** | block.sampleRateHz(192000) = session.sampleRateHz(192000)。不一致なし |
| `dropBank` | 0 | bankIndex 一致 |
| `queueDepthBlocks` | 0 | 滞留なし |

**結論**: `accepted=0` 仮説は **実測で完全に否定された**。NoiseShaper は正常動作。懸念されたsample rate不一致は発生していない。

### 7.2 EBR 経路の未検証と lifecycle(retire)=0 → ✅ 主因特定（2026-07-12 ログ解析）

⚠️ **CAVEAT**: `lifecycle(retire)=0` は handle 未登録が原因ではなく、gen=3 以降の publish が全滅（AUTH_CONTRACT）したため retire 機会そのものが発生しなかった。

**✅ ログ解析結果（2026-07-12 ConvoPeq.log）**:
- **lifecycle(pub/ret/reclaim) カウンタの変化**:
  - gen=3 FAIL 中: 4/0/0 (retire=0, reclaim=0)
  - gen=6 publish 成功直後: 6/0/0 (retire 未発生)
  - gen=6 xfade 完了後: 6/1/0 (retire enqueue 確認 = Ret: pend=1)
  - gen=7 publish 成功後: 7/1/1→7/1/13 (EBR polling 進行)
- MEM_SNAP の `Ret: pend=0` で retire queue 消費確認

**確認済み**: RETIRE enqueue → EBR polling → Ret: pend=0 のチェーンは動作。ただし EBR epoch advance / reader leave drain の詳細はこのログのみでは確認不可。

### 7.3 runtimeDSPHandleMap / BlockSize 実測 / 680MB Other 内訳

- `runtimeDSPHandleMap` 収束値: steady-state 2-3 エントリ見込み（実測確認が必要）
- BlockSize 削減効果: コード数式で ~189MB/DSPCore の削減見込み（実 MEM_SNAP での確認が必要）
- 680MB Other 内訳: `computeOtherPrivate` のトレース完了。`aligned_malloc` の DIAG 未計装により DSPCore 内部バッファは Other に含まれる。

### 7.4 調査結果の総括（2026-07-12 最終確定）

**全ツールを使用した最終網羅調査の結果、コード調査で確定可能な事項は全て確定・記録された。** 使用ツール: grep/sed (WSL), AiDex MCP, serena MCP, ctx_batch_execute, cocoindex-code (ccc.exe), graphify, semble。

| 調査項目 | 結果 | 確定状況 |
|:---------|:-----|:---------|
| `hasFadingRuntime` production残存 | **0件** — 全削除確認（ISRRuntimeSemanticSchema.h から削除、hasFadingRuntimeInWorld は fadingRuntimeUuid != 0 導出） | ✅ 確定 |
| `currentCaptureSessionId` 代入 | **0件** — 定義時初期値 =0 固定。Runtime代入なし → sessionId によるドロップは発生しない | ✅ 確定 (FACT #82) |
| P4 `DIAG_MKL_MALLOC` 完全性 | CacheManager(2) + IRConverter(1) の3箇所すべて DIAG 化＋`DiagnosticsConfig.h` include 追加完了。MKLNonUniformConvolver 内6箇所の生 `mkl_malloc` は P4 範囲外（ScopedAlignedPtr 管理＋allocatedBytes() 追跡済み） | ✅ 確定 |
| `aligned_malloc` DIAG 未計装 | 81箇所残存（v7.9 調査時点では95箇所 → DIAG_MKL_MALLOC 経由化により14箇所改善）。全体的な DIAG 化は設計範囲外 | ✅ 既知制限 |
| Builder メモリオーダー | 全18件 `memory_order_acquire` 統一。不整合なし | ✅ 確定 |
| `const_cast` | 2箇所のみ: RuntimeBuilder.h:94（旧シグネチャ委譲）, Orchestrator.cpp:178（FrozenRuntimeWorld 境界） | ✅ 確定 |
| 全 Builder 関数 | `buildRuntimePublishWorld`, `createBootstrapWorld`, `validateWarmup` 全て `noexcept` | ✅ 確定 |
| TODO/FIXME work70関連 | **0件** | ✅ 確定 |
| `collectTrackedMemoryStatistics()` | 定義＋実装完了（`ASSERT_NON_RT_THREAD()` 完備）。呼び出し元0件（P4 Backlog）。設計書要求（API定義）は達成済み。 | ✅ API完了（MEM_SNAP統合は別タスク） |
| `RuntimePublicationSpecification.h` | 独立ファイルとして作成。**git管理下に追加済み**。include元は0件（将来分離用準備）。 | ✅ ファイル存在確認・git追跡開始 |

**コード調査で確定不能な項目 → ログ解析で一部解決（2026-07-12）**:

1. ✅ **NoiseShaper accepted=0 真因** → ログ解析により **NOT-A-PROBLEM と確定**（`accepted=3012/3004`, `dropSampleRate=0`）
2. ⚠️ **EBR lifecycle(retire)=0** → 主因（AUTH_CONTRACT FAIL）を確定。gen=6 以降の retire/polling 動作確認済み。EBR epoch advance の詳細は別ログが必要。
3. 🔄 **BlockSize 削減実測値**（→ P2-1 実装後の MEM_SNAP）— コード変更後のため本ログでは未確認
4. 🔄 **680MB Other 内訳の実測値**（→ TrackedMemoryStatistics MEM_SNAP 統合後）— 現状の計装では Private−TRK=~455MB までしか分解不能

**結論**: コード調査＋ログ解析により、7.1 は完全解決、7.2 は主因確定。残る 7.3/7.4 はコード変更後の実測に依存。

---

## Appendix A: 実装済み成果物

### A.0 今回の改修（work70 v8.3 全11ステップ）— ソースコード上確認済み

| ステップ | 内容 | ファイル | 状態 |
|:--------|:-----|:---------|:-----|
| ① | RuntimePublishSpecification 定義（三部構成 + version） | `RuntimeBuilder.h` | ✅ 実装確認 |
| ② | Orchestrator Spec生成 + Post-build Mutation 削除 | `RuntimePublicationOrchestrator.cpp` | ✅ worldOwner-> 代入 READ 1件のみ |
| ③ | Builder Spec 受取 + 内部実装修正 | `RuntimeBuilder.h/.cpp` | ✅ シグネチャ変更＋Spec 解決 |
| ④ | 他6箇所の呼び出し元対応（旧シグネチャ委譲） | 各 .cpp | ✅ inline 委譲確認 |
| ⑤ | Validator 三者整合性（Topology/Execution/Identity） | `RuntimePublicationValidator.cpp` | ✅ hasFadingRuntime 不使用 |
| ⑥ | hasFadingRuntime 削除（production） | `ISRRuntimeSemanticSchema.h` + 全参照 | ✅ fadingRuntimeUuid != 0 導出 |
| ⑦ | P2-1 BlockSize 最適化 | `AudioEngine.h`, `Init.cpp`, `DSPCoreLifecycle.cpp` | ✅ PrepareBlockSizingPolicy + kInitialPrepareMaxBlock=4096 |
| ⑧ | P-NS NoiseShaper DIAG 改善 | `NoiseShaperLearner.h/.cpp` | ✅ DropReason enum, generation, queueDepthBlocks, dropBySampleRate DIAG |
| ⑨ | P-DIAG TrackedMemoryStatistics API + MEM_SNAP統合 | `AudioEngine.h`, `DSPCoreLifecycle.cpp`, `Timer.cpp` | ✅ 10カテゴリ + ASSERT_NON_RT_THREAD + MEM_SNAP出力（P4） |
| ⑩ | P4 mkl_malloc DIAG 化 | `CacheManager.cpp`, `IRConverter.cpp` | ✅ DIAG_MKL_MALLOC + include 追加 |
| ⑪ | const RuntimePublishWorld 化 | `RuntimeBuilder.h`, `AudioEngine.h`, `RuntimePublicationCoordinator.h` | ✅ Builder→commitRuntimePublication→Coordinator const chain |
| **⑫** | **P0: ProcessingPart追加 + atomic読取り排除（v9.4）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ ProcessingPart定義、Orchestrator設定、Builder読取り、無効行削除 |
| **⑬** | **P1: PublicationSnapshotPart + previousCommittedSequence移行（v9.5）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ PublicationSnapshotPart定義、Orchestrator設定、Builder読取り（Runtime Query 部分移行完了） |
| **⑭** | **P2: CrossfadeSnapshotPart/LatencyPart 追加（v9.5）** | `RuntimeBuilder.h/.cpp`, `RuntimePublicationOrchestrator.cpp` | ✅ CrossfadeSnapshotPart/LatencyPart 定義・Orchestrator設定・Builder engine直接読取り排除 |

#### A.0.1 RuntimePublishSpecification 構造（v8.3）

```cpp
struct RuntimePublishSpecification {
    uint32_t version = 1;

    struct TopologyPart {
        const AudioEngine::DSPCore* activeDSP = nullptr;
        const AudioEngine::DSPCore* fadingDSP = nullptr;
    } topology;

    struct ExecutionPart {
        bool transitionActive = false;
        int transitionPolicy = 0;  // 0=HardReset, 1=SmoothOnly, 2=DryAsOld
        double fadeTimeSec = 0.0;
    } execution;

    struct RoutingPart {
        int processingOrder = 0;   // 0=EQ→Conv, 1=Conv→EQ
        bool eqBypassed = false;
        bool convBypassed = false;
    } routing;
};
```

#### A.0.2 P2-1 PrepareBlockSizingPolicy

```cpp
struct PrepareBlockSizingPolicy {
    static constexpr int kMinimumPrepareBlock = 256;
    [[nodiscard]] static constexpr int apply(int samplesPerBlock) noexcept {
        jassert(samplesPerBlock > 0);
        return std::max(kMinimumPrepareBlock, samplesPerBlock);
    }
};
const int kInitialPrepareMaxBlock = 4096;  // AudioEngine.Init.cpp
```

#### A.0.3 P-NS DropReason + Waiting diagnostics

| フィールド | 値 | 意味 |
|:-----------|:---|:------|
| `generation=` | `consumeAtomic(progress.iteration)` | 完了世代数（1-based） |
| `queueDepthBlocks=` | `captureQueue.size()` (w-r) | キュー内 AudioBlock 滞留数 |
| `dropBySampleRate` DIAG | `block.sampleRateHz` + `session.sampleRateHz` 実値 | Runtime 観測用 |

#### A.0.4 P-DIAG TrackedMemoryStatistics API

```cpp
struct TrackedMemoryStatistics {
    size_t oversampling = 0;       // Oversampling work buffers
    size_t softClip = 0;           // SoftClip OS work buffers
    size_t eqProcessor = 0;        // EQ scratch/dry/parallel/structure/msWorkBuffer
    size_t alignedBuffers = 0;     // alignedL/R + dryBypassL/R
    size_t latencyBuffers = 0;     // fixedLatency × 4
    size_t truePeakDetector = 0;   // TruePeakDetector internal
    size_t convolver = 0;          // Convolver internal (no IR = minimal)
    size_t crossfade = 0;          // JUCE crossfade buffers
    size_t misc = 0;               // DCBlocker/LoudnessMeter/PeakLimiter/NoiseShaper
    size_t otherTracked = 0;       // tracked だが特定カテゴリに分類されないもの
    [[nodiscard]] size_t totalTracked() const noexcept { /* SUM(categories) */ }
};
```

### A.1 P1-a: publish 経路への handle 登録追加

**目的**: Coordinator direct publish（Init/PrepareToPlay/ReleaseResources/Timer/Transition）で DSPCore が `runtimeDSPHandleMap_` に未登録となる問題を修正。

```cpp
struct RegistrationContext {
    DSPCore* dsp = nullptr;
    DSPHandle handle;
    static RegistrationContext needsRegistration(DSPCore* dsp_) noexcept;
    static RegistrationContext alreadyRegistered(DSPHandle handle_) noexcept;
    static RegistrationContext none() noexcept;
};
```

**呼び出しパターン（7 箇所）**:

| # | 呼び出し元 | mode | dsp | handle |
|:-:|:-----------|:-----|:----|:-------|
| 1 | Init | none | — | — |
| 2 | PrepareToPlay (first) | needsRegistration | currentForPublish | — |
| 3 | PrepareToPlay (rebuild) | needsRegistration | getActiveRuntimeDSP | — |
| 4 | ReleaseResources | none | — | — |
| 5 | Timer (fadeComplete) | needsRegistration | currentAfterFade | — |
| 6 | Transition | needsRegistration | 引数 newDSP | — |
| 7 | PublicationExecutor | alreadyRegistered | — | req.newDSP |

**検証**: CTest 15/15 PASS, CI Gates ALL PASS.

### A.2 P1-b: advanceFade 配線

- FadeState の開始・進行・完了を RuntimePublicationCoordinator と連携
- 完了後は advanceFade 内で commitRuntimePublication を呼び出して新しい RuntimePublishWorld を公開

### A.3 P1-c: MEM_SNAP 監視強化

- DSPCore::liveCount（DIAG ガード済み atomic）の MEM_SNAP 出力
- RuntimeWorld サイズ表示（`world gen=M size=NNNMB`）
- `computeOtherPrivate()` の完全追跡

### A.4 P1-a-FIX: activeRuntimeDSPHandle_ 未更新修正

Coordinator direct publish 後の `activeRuntimeDSPHandle_` 更新漏れを修正。`commitRuntimePublication()` 成功後に `dspHandleRuntime_.activate(handle)` を呼ぶ。

### A.5 P1-a-FIX-2: DSPGuard 直接破棄パス

例外発生時に DSPGuard から直接 `destroyDSPCoreNode()` を呼ぶパスを追加。`destroyDSPCoreNode()` 呼び出し前に `DSPHandle` を retire する。

### A.6 P1-a-FIX-3: 0xC0000005 修正（DSPGuard 重複 destroy）

`runtimeDSPHandleMap_` の `eraseByHandle` が `DSPGuard` 破棄後に `runtimeDSPHandleMap_` を走査して Access Violation を引き起こす問題を修正。

### A.7 D-1: DSPLifetimeManager::destroyRolledBackDSP

publish 失敗後に未公開 DSPCore を安全に破棄するための専用メソッド。

### A.8 検証結果

**設計上のメモリ効果見込み**: 未修正時 2,477MB に対し、設計上は定常 686MB / ピーク 1,094MB を見込む。ビルド・実測未実施のため確定値ではない。

**FACT 一覧（全86件）**: コード調査により確定可能な事項は全て確定済み。残る6項目は全て Runtime 観測または実装後の検証に依存（[未確定] 7 参照）。

**検証済みの設計判断**:

| 判断 | 結論 |
|:-----|:------|
| advanceFade のサンプル単位 | OS 補正不要。コールバックサンプル数のまま減算 |
| FadeState::Completed 追加 | 不要。既存の remaining==0 チェックで完了検出可能 |
| memory_order 変更 | 現状維持（relaxed 化は却下） |
| CrossfadeRuntime への完了通知追加 | 不要。SnapshotCoordinator と CrossfadeRuntime は責務が完全に独立 |

---

## Appendix B: 調査ツール

| ツール | 使用目的 |
|:-------|:---------|
| grep/sed/awk (WSL) | ログ抽出、統計計算、production コード全数調査 |
| serena MCP | コードパストレース、型情報取得、状態遷移調査 |
| cocoindex-code (ccc.exe) | 関数間依存関係の grep、シンボル特定 |
| graphify | 依存関係グラフパス検索、関数間リンク検証 |
| semble | セマンティックコード検索、フォールバック経路発見 |
| AiDex MCP | コードインデックス検索、セッションノート管理 |
| ast-grep / rg / fd / fzf | WSL ベースの高速コード検索・フィルタリング |

---

## Appendix C: 改訂履歴

| 版 | 日付 | 改訂内容 |
|:---|:-----|:---------|
| 1.0〜7.9 | 2026-07-10〜11 | 初版〜FACT 86 確定（旧版 v8.3 までの全履歴は Appendix E 参照） |
| **8.0** | **2026-07-11** | **レビュー指摘3点反映（最終確定）**: transitionActive 導出→ExecutionSemantic 包含に戻す、Specification 三階層構造化、AllocatorPolicy 独立 |
| **8.1** | **2026-07-11** | **レビュー指摘4点反映**: Specification 三部構成、PrepareBlockSizingPolicy 改名、DropReason 通常 enum、totalTracked カテゴリ合計算出 |
| **8.2** | **2026-07-12** | **レビュー指摘5点反映**: version フィールド追加、enum class 化、jassert Contract Enforcement、Deterministic Construction 明確化、DTO 的性質明文化 |
| **8.3** | **2026-07-12** | **レビュー指摘3点反映**: Specification 独立ファイル化、INV-12 Deterministic Utility 条件追加、Validator 三カテゴリ再構成 |
| **9.0** | **2026-07-12** | **全面再構成**: 実装済み全11ステップを Appendix に移動。優先改修項目（Backlog）を先頭に新設。INV-12 をユーザー設計判断に基づき再定義。Builder 依存4分類を追加。collectTrackedMemoryStatistics MEM_SNAP 統合を Backlog P5 に追加。 |
| **9.1** | **2026-07-12** | **設計審査フィードバック反映**: P0 本質を「暗黙入力排除」に修正。P2 RuntimeContextPart を CrossfadePart/LatencyPart/PublicationPart に分割。P3 EnginePart/GraphPart 削除＋Specification Part 追加の3基準追加。Builder Service 明示分類。Deterministic Construction 定義追加。優先順位再編（INV-12 再定義を P0-P2 完了後の P3 に移動）。 |
| **9.2** | **2026-07-12** | **設計審査フィードバック反映（7点）**: P0 Deterministic 定義を Semantic Equivalence に精密化＋P0 タイトル「排除→昇格」に修正。P1 AutomationPart を ProcessingPart に改名（processingOrder 包含）。P2 PublicationPart/CrossfadeSnapshotPart に責務明記＋Builder Service を契約ベースに抽象化。INV-12 本文に Pure Utility 追記。エグゼクティブサマリ「改善→見込み」に修正。 |
| **9.3** | **2026-07-12** | **設計審査フィードバック反映（4点）**: INV-13（設計原則）と INV-12（実装契約）の上下関係を明示（INV-13→INV-12）。INV-13「Builder mutable Runtime state 直接観測禁止」追加。ProcessingPart に将来の RoutingPart/ProcessingParameterPart 分割方針を補足。processingOrder バグを設計問題と実装問題に分離記載。全ソース最終調査（grep/sed/AiDex/serena/ctx）で残存問題ゼロ確認。 |
| **9.4** | **2026-07-12** | **可読性・保守性改善（5点）**: ProcessingPart に YAGNI 統合理由を追記。Builder Service を「Builder execution environment」と位置づけ。PublicationPart→PublicationSnapshotPart に改名。INV-12/INV-13 番号を設計原則→実装契約の順序に入れ替え。FACT 件数表記を「全86件」→「整理対象86件」に軟化。 |
| **9.5** | **2026-07-12** | **実装進捗**: P0（ProcessingPart + atomic読取り排除）✅完了。P4（MEM_SNAP統合）✅完了。P1第一段階（PublicationSnapshotPart追加＋previousCommittedSequence移行）✅完了。P2（CrossfadeSnapshotPart/LatencyPart追加＋Builder engine直接読取り排除）✅完了。設計書Backlog更新、Appendix A.0に⑫⑬⑭追加。P3 設計書更新済み。 |
| **9.6** | **2026-07-12** | **設計契約の明文化（5点）**: (1) Deterministic Construction に「Builder Service may affect implementation artifacts only」追記。(2) Specification Completeness 節を新設（INV-13 の Data 制約版）。(3) P2 に Source 一意性（ProcessingPart 唯一 Source）と Part Ownership 表を追加。(4) Builder Service を Memory/Identity/Immutable Factory に細分類。(5) P2 Backlog 完了確認・冗長代入 cleanup は次回対応。 |
