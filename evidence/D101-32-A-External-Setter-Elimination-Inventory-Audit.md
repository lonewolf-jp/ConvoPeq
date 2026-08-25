# D101-32-A — External Setter Elimination Inventory Audit（実施報告書）

- **実施日**: 2026-08-25
- **性質**: read-only audit（コード変更禁止 / 実績: ソースコード変更 **0**）
- **判定**: **PASS**（B2 削除条件を満たす production call site = 全 setter で **0件**。ただし vestigial counter の扱いを B2 への確定事項として明示）
- **基準**: ConvoPeq.md 2026-08-25 09:48 再生成版（本監査内で再生成・source of truth 確定）
- **前提**: D101-31-D PASS 済み（AdmissionPackedState 契約固定）

---

## 0. エグゼクティブサマリー

ユーザー提示の前提「production 側に外部 setter が残存」は**現在のコードには当てはまらない**。
dash2 §1.4 (B0) + work88 (X6 §6.6) の先行実装により、対象5 setter の production call site
は既に全廃され、各箇所は削除理由コメントに置換されている。

| 項目 | 現状 |
|---|---|
| 対象5 setter の production call site | **0**（コメントのみ残存） |
| setter 定義 | `ISRRuntimePublicationCoordinator.cpp:289-316` に残存、`⚠️ TEST-ONLY` 明記 |
| semantic event API（B1 相当） | **実装済み**: onRetireAccepted/Consumed, onFallbackAccepted/Consumed, onDeferredRetireAccepted/Consumed, onReclaimBegin/End（underflow ガード → Faulted 付き） |
| production wiring 済み event | `onReclaimBegin/onReclaimEnd` のみ |
| vestigial counter（常時0・無 writer） | `retireBacklogCount_`, `fallbackBacklogCount_`, `deferredRetireResidencyCount_`, `quarantineResidentCount_` — drain 条件に ==0 判定が残存（恒真） |
| quarantine domain mixing | **解消済み**（X6 §6.6 — 詳細 §3） |

したがって D101-32-B（Semantic ++/-- Implementation）で「新規に実装すべき ++/--」は
存在せず、次フェーズの実質的な作業は **D101-32-C/D（Integrity Audit → API Removal /
vestigial counter の削除-or-wire 決定）** となる。

---

## 1. A-1: 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 09:48)
git status                             → 変更は D101-31-D 分のみ（未コミットの既存分）
git diff --check                       → whitespace エラーなし（LF/CRLF 警告のみ）
```

以降の監査は本版 ConvoPeq.md を source of truth とする。

---

## 2. A-2: 外部 setter 全 call site 棚卸し

### 2.1 対象5 setter の現存 call site（fresh grep、コメント除外）

検索: `src/{audioengine,core,convolver,eqprocessor}` 内の非コメント呼び出し →

**結果: 0件**（全ファイル空）。

残存するのは以下のみ:

| 箇所 | 性質 |
|---|---|
| `ISRRuntimePublicationCoordinator.h:143,146-149` | 宣言（`// TEST-ONLY` 明記） |
| `ISRRuntimePublicationCoordinator.cpp:289-316` | 定義（`⚠️ TEST-ONLY（dash2 §1.4）` コメント付き） |
| `src/tests/ISRSemanticValidationTests.cpp` | テスト初期化リセット用途（18箇所 — production inventory 外） |
| 各 production ファイルのコメント | 削除理由の記録（下表参照） |

### 2.2 元 call site（撤去済み）とその代替の対応表

| # | caller（撤去時点） | thread | 測定していた値 | semantic domain | 本来の owner | 増加イベント | 減少イベント | snapshot overwrite | domain mixing | B1 置換 API / 現状 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AudioEngine.Commit.cpp:481（現 :489 コメント） | NonRT（commit path） | lifetime().pendingIntentCount() スナップショット | retire | RetireRouter（実測 pendingRetireCount）+ Coordinator 内部 | onRetireAccepted | onRetireConsumed | **有** | 有（RetireIntent を retireBacklog に混入） | 廃止 → Layer 1 が Threading.cpp isFullyDrained で router 実測直接判定 |
| 2 | AudioEngine.Commit.cpp:624（現 :633 コメント） | NonRT | 同上 | retire | 同上 | 同上 | 同上 | **有** | 同上 | 同上 |
| 3 | AudioEngine.Threading.cpp:126-128（現 :124-128 コメント） | Timer/NonRT（isFullyDrained 計算時） | fallbackDepth / retireDepth スナップショット | fallback / retire / deferred | Layer 2 Coordinator（queue emptiness + 内部カウンタ）/ Router 実測 | onFallbackAccepted 等 | onFallbackConsumed 等 | **有** | 有 | 廃止 → isFullyDrained が実測値（pendingRetireCount / lifetime().pendingIntentCount()）を直接判定 |
| 4 | AudioEngine.Retire.cpp:113-115（撤去済み） | NonRT（drain path） | fallbackDepth / retireDepth | 同上 | 同上 | 同上 | 同上 | **有** | 有 | 同上 |
| 5 | AudioEngine.Retire.cpp:48/:52（現 :60 コメント） | NonRT（drainDeferredRetireQueues） | reclaim in-flight 1/0 | reclaim | Coordinator（reclaim transaction） | **onReclaimBegin** | **onReclaimEnd** | 有 | なし | ✅ **onReclaimBegin()/onReclaimEnd() に置換済み・wired**（Retire.cpp:62/66） |
| 6 | AudioEngine.Retire.cpp:316/:319（現 :355 コメント） | NonRT（emergency reclaim boost） | 同上 | reclaim | 同上 | 同上 | 同上 | 有 | なし | ✅ 同置換済み（Retire.cpp:356/359） |
| 7 | ReleaseResources.cpp:291（現 :306-311 コメント） | NonRT（shutdown drain） | ringResident（overflow ring resident） | **quarantine（誤）** | DSPQuarantineManager::residentCount() | （該当なし—誤ドメイン） | （同左） | **有** | **有（INV-X6-4 違反）** | 削除。Layer 1 が dspQuarantineManager_.residentCount() を直接判定（§3 参照） |
| 8 | AudioEngine.h:4152/:4162（現 :4229/:4240 コメント） | NonRT（inline helper） | retireDepth スナップショット | retire | 同 #1 | 同上 | 同上 | 有 | 有 | 廃止 |

### 2.3 semantic event の production wiring 実績

| Event | 定義 | production caller | wire 状態 |
|---|---|---|---|
| onRetireAccepted | Coordinator.cpp:190 | Coordinator.cpp:171（enqueueRetireIntent 内 — **同メソッド自体が production 未使用**、将来の retire 経路用と明記） | 未接続（意図的） |
| onRetireConsumed | Coordinator.cpp:198 | **なし** | vestigial（retireBacklogCount_ 常時0。Layer 1 実測判定が authoritative） |
| onFallbackAccepted/Consumed | Coordinator.cpp:210/214 | **なし** | vestigial 同上 |
| onDeferredRetireAccepted/Consumed | Coordinator.cpp:223/227 | **なし** | vestigial 同上 |
| onReclaimBegin/End | Coordinator.cpp:236/249 | Retire.cpp:62/66, :356/359 + Coordinator 内部 :736/:747（requestReclaim deferred→ACK） | ✅ **完全 wired** |

underflow ガード仕様（実装確認済み）: Consumed 系は fetch_sub 前 `old > 0` 検証、
違反時 `CoordinatorState::Faulted` へ遷移（Proof 生成不能）。例外的に onReclaimEnd のみ
「defer なし単発成功」が正常系のため Faulted 化せず no-op（INV-3-1 / Coordinator.cpp:239-247 コメント）。

### 2.4 関連: 対象外だが同クラスの TEST-ONLY setter

`setPublicationBacklogCount`(dead counter 明記) / `setPendingIntentCount` も TEST-ONLY 化済み。
`pendingIntentCount_` は reservation ベース（push 成功 fetchAdd / pop 成功 fetchSub）で維持され、
ProcessIntent.cpp:73-76 で setPendingIntentCount(0) リセット廃止を確認。

---

## 3. A-3: setQuarantineResidentCount() 重点監査 — domain mixing 解消状況

### 3.1 問題（旧実装）

```text
ReleaseResources.cpp（旧 :291）
    runtimePublicationBridge_.setQuarantineResidentCount(ringResident);
        → retire 系の overflow ring resident 数を
          Coordinator の quarantine resident カウンタへ絶対値上書き
        = INV-X6-4 違反（semantic 混同）
```

### 3.2 現在のコード上の確定事項

1. **writer 廃止**: 当該箇所は ReleaseResources.cpp:306-311 の削除理由コメントに置換済み。
   「overflow ring resident（retire 系）を quarantine カウンタに混ぜるのは INV-X6-4 違反」と明記。
2. **source of truth 確定**: 実在 quarantine DSP 数 = `DSPQuarantineManager::residentCount()`
   （ISRDSPQuarantine.cpp:103）。AudioEngine::isFullyDrained（Threading.cpp:139-143）が直接判定する。
3. **Coordinator 側カウンタの現役務**: `quarantineResidentCount_`（Coordinator.h:574）は
   「X6 以降 submitQuarantine が +1 しない」「常時 0」と header に明記された vestigial counter。
   ShutdownScheduler::isFullyDrained（Coordinator.cpp:587）の ==0 判定は恒真成立。
4. **semantic 分離の完成形（X6 §6.6）**: Quarantine は 3 レーンに分離済み —
   - `quarantineIntentResidencyCount_` = intentQueue_ 内 Quarantine Intent（primary transport）
   - `quarantineRingResidencyCount_` = quarantineFallbackQueue_ 残留（fallback/ring）
   - 実在 DSP = DSPQuarantineManager（Coordinator 外、Layer 1 直接判定）
5. **同名異義の注意**: ISRRetireRuntimeEx.cpp:169-301 / .h:104 の `quarantineResidentCount_`
   は別クラス（RetireRuntime/EPOCH 制御）のメンバで、Q + EmergencyQ の滞留数という
   **別 semantic**。混同ではなく意図的分離（Threading.cpp:142 で `retireQuarantineResident`
   として独立判定）。さらに Terminal 層も isFullyDrained に含まれる（Threading.cpp:147-151,
   15-P-5 — premature waitForDrain 成功の防止）。

### 3.3 確定

> **値自体の semantic correction は完了している。**
> ringResident は quarantine quantity ではないため置換先イベントは存在せず、
> 正しい解は「writer 削除 + Layer 1 実測直接判定」であり、それが現行実装。
> B1 での追加実装は不要。B2 では Coordinator 側 vestigial `quarantineResidentCount_`
> の削除可否を判断する（§5）。

---

## 4. A-4: B1 実装順序（dataflow 設計）

B1（snapshot set(X) → authoritative increment/decrement）の変換は**設計済みかつ大部分実装済み**。

### 4.1 確定済み dataflow（reclaim domain — 唯一の production wired 例）

```text
[producer] drainDeferredRetireQueues / emergency boost / requestReclaim(defer)
    ↓
onReclaimBegin()  … fetchAdd(reclaimInFlightCount_)      [Coordinator.cpp:236]
    ↓
reclaim ACK（epoch 安全化完了）
    ↓
onReclaimEnd()    … old>0 ガード付き fetchSub             [Coordinator.cpp:249]
    ↓
Coordinator observation: ShutdownScheduler::isFullyDrained()
    && reclaimInFlightCount_==0                            [Coordinator.cpp:579]
```

### 4.2 retire/fallback/deferred domain の確定方針（dash2 §1.4 設計）

```text
producer event（retire 受諾 / consume）
    ↓
semantic owner = Layer 2 Coordinator（event counter）…だが authoritative 判定は:
Layer 1 AudioEngine::isFullyDrained() が「実測値」を直接判定:
    - m_retireRouter->pendingRetireCount()            （retire depth）
    - worldAuthority_.lifetime().pendingIntentCount() （RetireIntent 滞留）
    - getOverflowRing()->residentCount()              （ring 残留）
    - dspQuarantineManager_.residentCount()           （実在 quarantine DSP）
    - m_retireRouter->quarantineResidentCount()       （Q+EmergencyQ）
    - terminalReclaimResident                         （Terminal 層）
    && runtimePublicationBridge_.isFullyDrained()     （Layer 2 queue/counters）
```

**B1 追加実装順序（確定）**:

1. 不要（reclaim domain）— wired 完了
2. 不要（retire/fallback/deferred）— Layer 1 実測直接判定が dash2 §1.4 の確定設計であり、
   現行コードがそれに合致。event API は将来の retire 経路用プレースホルダとして維持
3. B2 へ直行（API 削除 + vestigial 処理）

### 4.3 B1 相当の残課題 = なし

snapshot overwrite は 8/8 サイトで廃止済み。domain mixing も X6 §6.6 で解消済み。

---

## 5. A-5: B2 削除条件の明文化

### 5.1 acceptance gate（ユーザー指定の通り、現時点で既に充足）

```text
production call sites（非コメント実呼び出し）:
setFallbackBacklogCount              = 0  ✅
setRetireBacklogCount                = 0  ✅
setDeferredRetireResidencyCount      = 0  ✅
setReclaimInFlightCount              = 0  ✅
setQuarantineResidentCount           = 0  ✅
```

テストコードの test-access 用途（ISRSemanticValidationTests.cpp の18箇所）は
production inventory から分離済みとして扱う（テスト初期化リセットは P2 教訓により許容）。

### 5.2 B2（D101-32-D）での削除対象の確定

| 対象 | 処理 | 根拠 |
|---|---|---|
| 5 setter の宣言+定義 | 削除（または test 専用ヘッダへ移設）し、テストは reset 用途を semantic event 併用 or 専用 test seam に置換 | production 参照 0 をコンパイル時に保証（header コメントの「コンパイル時参照 = 0 を維持」契約） |
| vestigial counters（retireBacklogCount_ / fallbackBacklogCount_ / deferredRetireResidencyCount_ / quarantineResidentCount_） | **要決定**: (a) 削除して drain 条件から外す、または (b) 将来 retire 経路用に保持 | 現状 drain 条件（Coordinator.cpp:571-593）の ==0 は恒真で意味を持たない。ただし onRetireAccepted が将来経路用に保持されているため、counter 保持と対になる。**推奨: (b) 保持 + drain 条件からの恒真判定にはコメントで vestigial である旨を明記（最小変更）** |
| onRetireConsumed 等未接続 event API | 保持（将来 retire 経路の対イベント） | enqueueRetireIntent の将来利用前提 |

### 5.3 B2 の blocking issue（なし / 条件付き）

- **BLOCKER なし。** 削除は pure mechanical。
- 注意: `setRetireBacklogCount` のみ noteRetireBacklogChanged(pressure slope 共通化) を
  呼ぶため、削除時に pressure 更新経路が event 経由のみになることを C で再確認すること。

---

## 6. A-6: AdmissionPackedState 境界の再確認（D101-31-D 契約の維持）

| 項目 | 確認結果 |
|---|---|
| AdmissionReservation = Publication + Recovery + Build | ✅ tryAdmit production call site は正確に3: RuntimePublicationOrchestrator.cpp:69（Publication）/ AudioEngine.RebuildDispatch.cpp:319（Recovery）/ AudioEngine.h:4443（Build）。fresh grep で再確認 |
| Retire は AdmissionReservation 対象外 | ✅ Retire 経路に tryAdmit なし |
| packedState_.reservationCount を他カウンタと統合しない | ✅ 統合・参照の試みなし（ISRShutdown.{h,cpp} に閉じる。本監査で未触碰） |
| publicationIntentResidencyCount_ / pendingIntentCount_ / retireBacklogCount_ / reclaimInFlightCount_ / quarantineResidentCount_ の統合禁止 | ✅ 各カウンタ独立のまま（X5/X6 の semantic 分離と整合） |

---

## 7. 完了条件チェックリスト

| # | 条件 | 判定 |
|---|---|---|
| 1 | 最新 ConvoPeq.md 再生成済み | ✅ 2026-08-25 09:48 版 |
| 2 | 対象 setter の production call site 全件列挙 | ✅ §2（現存 0 + 撤去済み 8 site の対応表） |
| 3 | semantic owner 確定 | ✅ §2.2 表 |
| 4 | +/- イベント確定 | ✅ §2.2 / §2.3（reclaim のみ wired、其余 Layer 1 実測） |
| 5 | setQuarantineResidentCount domain mixing 解消方針確定 | ✅ §3（解消済み・source of truth = DSPQuarantineManager::residentCount()） |
| 6 | B1 置換順序確定 | ✅ §4（追加実装不要と確定） |
| 7 | B2 削除対象確定 | ✅ §5.2 |
| 8 | AdmissionReservation 3-path 境界不変 | ✅ §6 |
| 9 | コード変更 0 | ✅ 本監査セッション中の src/CMake/build.bat 変更なし（working tree の差分は D101-31-D 分のみ） |
| 10 | PASS または blocking issue 明示 | ✅ **PASS**（blocking issue なし。§5.2 の vestigial counter 方針決定を B2 へ持ち越し） |

---

## 8. 次フェーズへの引き継ぎ事項

1. **D101-32-B は実質スコープ縮小**: 新規 ++/-- 実装は不要。「既存 semantic event 実装の
   integrity 監査（C）」へ統合可能と判断される（進行判断はユーザー裁量）。
2. **D101-32-C で確認すべき点**:
   - setRetireBacklogCount 削除時の noteRetireBacklogChanged（pressure slope）経路の帰趨
   - vestigial counter の (a) 削除 or (b) 保持+明記 の最終決定
   - ISRSemanticValidationTests の reset 用 setter の置き替え先（test seam 化）
3. **Path B（D101-33）との関係**: ユーザー指示通り、setter authority 収束（本系列）を
   Path B admission transaction 設計より先に完了させる。Path B は単純な state_ 読み取りでは
   shutdown race を解消できず admission transaction + linearization point を要する（既存監査）。

## 9. 使用ツール記録

rg(WSL), grep/sed(WSL), git, AiDex(session/query/update/note),
ctx_batch_execute/ctx_search（生出力の sandbox 処理 — 旧セッションデータの stale 検出と
fresh 再検証を実施）。serena/ccc/graphify/semble は本監査では identifier 解決を
AiDex+rg で完結できたため不使用（原則使用の例外を明記）。
