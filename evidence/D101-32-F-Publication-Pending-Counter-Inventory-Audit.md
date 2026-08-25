# D101-32-F — publicationBacklogCount / setPendingIntentCount Independent Inventory Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0**）
- **判定**: **PASS**（D101-33-A への引き渡し条件充足 — §7）
- **基準**: ConvoPeq.md **2026-08-25 12:07 再生成版**
- **前提**: D101-31-D / D101-32-A / C / D / E すべて PASS

---

## 1. F-1 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 12:07)
git status                             → 変更ファイル = 既存セッション分のみ（追加変更なし）
git diff --check                       → whitespace エラーなし
```

---

## 2. F-2/F-3 writer / reader / authority matrix

### 2.1 `publicationBacklogCount_`

| 種別 | 内容 |
|---|---|
| writer | **setPublicationBacklogCount（TEST-ONLY）のみ** — production writer は存在しない（header 自ら「TEST-ONLY（dead counter）」と明記 h:147）。AudioEngine::isFullyDrained からの上書きも不存在（B0 系撤去済み、resurrection なし） |
| reader | ① ShutdownScheduler::isFullyDrained cpp:527（**恒真 ==0**）② Coordinator::getPublicationBacklogCount cpp:430 → AudioEngine.h:3480（P1-6/8 診断公開）→ Threading.cpp:78 `.pendingPublication`（診断スナップショット）③ Orchestrator.h:239 経由 → **RuntimeHealthMonitor.cpp:342 checkPublicationStall の第3項** ④ テストハーネス wait ループ（DeferredFlowIntegrationTests:93/:144、SoakPublishIntegrationTests:187/:226/:252） |
| semantic quantity | 「溜まった未処理 publish」のつもり（HealthMonitor コメント）— だが実際は常に 0 |
| authority | **なし（dead）**。Publish 滞留の実測 authority は `publicationIntentResidencyCount_`（X5 §6.5）が担う |
| 判定 | counter/setter = **C（REMOVE 候補）**、getter chain = **B（REWIRE）**（§5） |

### 2.2 `pendingIntentCount_`

| 種別 | 内容 |
|---|---|
| writer | **semantic event ベースの authoritative accounting（LIVE）**: <br>• +1 `submitObserve` cpp:626（reservation-before-push）<br>• −1 同 rollback cpp:645（全層溢れ drop 時 — カウンタ不変を保証）<br>• +1 `submitRecoveryRequest` cpp:850 ／ −1 rollback cpp:861（queue full → durable admission へ。INV-X1-2/5/6）<br>• +1 quarantine intent enqueue cpp:998 ／ 各種 rollback（X6 §6.6）<br>• −1 pop 成功時: processIntent ProcessIntent.cpp:36/:60/:63、drainObserveDeferred :89、popRecoveryRequest cpp:1023 等（計8 site）<br>• setPendingIntentCount（TEST-ONLY）— production caller ゼロ |
| underflow protection | fetchSub 前 old>0 ガードは**意図的に廃止**（cpp:944-949 コメント: ガードは不整合を silently hide するため危険）。代わりに reservation-before-push 構造で「pop 成功数 == push 成功数」を構造的に保証（underflow 不変条件） |
| reader | drain 条件 cpp:532 ✅ authoritative ／ getPendingIntentCount cpp:439 → RuntimeHealthMonitor.cpp:340（hasPendingWork — LIVE）／ invariant_INV3_INV5.cpp（capacity/push/pop 会計検証 :308/:319/:325/:384/:401/:407/:429/:436）／ ISRSoakTests.cpp:185/:233/:257（kIntentCap+kFallbackCap 容量会計）／ ISRSemanticValidationTests.cpp:819（N intent 注入確認） |
| semantic quantity | **非 Publish Intent（Observe / Recovery / Quarantine）の transport 予約残数**（INV-ISR-02: Publish と RetireIntent は混入禁止 — h:552-555 明記） |
| authority | ✅ **Layer 2 の authoritative reservation accounting** |
| 判定 | counter/getter = **A（KEEP）**、setter = **C（REMOVE 候補・reset-only）** |

### 2.3 `publicationIntentResidencyCount_`

| 種別 | 内容 |
|---|---|
| writer | **単一 choke point**: enqueuePublicationIntent 内 h:372（fetchAdd, reservation-before-push）→ push 成功で維持 / push 失敗で h:375 rollback。全 3 enqueue 経路（通常 rebuild / Recovery publish / deferred 再 enqueue）がここに集約（X5 §6.5）。pop 成功時 ProcessIntent.cpp:56 fetchSub |
| reader | getter cpp:436 / drain 条件 cpp:531（INV-X5-1 — queue emptiness 単独では Publish 残留を捕捉できないため独立判定） |
| semantic quantity | **Publish Intent 専門の transport residency + producer reservation**（並行中は >= 、producer quiescence 後は ==） |
| authority | ✅ **Layer 2 Publish lane の authoritative residency** |
| 判定 | **A（KEEP）** — X5 新設の生きたカウンタ |

---

## 3. F-4 setPublicationBacklogCount の確定

1. **production caller: 存在しない**（定義 + TEST-ONLY 宣言のみ。resurrection / obsolete wiring なし — 最新ソースで再確認）
2. **AudioEngine::isFullyDrained からの setter 上書き: 存在しない**（Layer 1 は publicationBacklogCount_ を一切参照しない。Threading.cpp の参照は getter 経由の診断出力のみ）
3. **test-only caller: 3箇所**（ISRSemanticValidationTests.cpp:327/:363/:433 — 全て reset-to-0）＋ RuntimeHealthMonitorTierTests.cpp:226 は getter の stub（return 0）
4. **破壊可能性**: 非 zero 注入すれば drain 条件(:527)永久 false + HealthMonitor 誤検知を起こし得るが、production 参照 = 0 のため現状不可能（コンパイル時参照=0 契約を維持中）
5. **==0 が drain proof に持つ情報: ゼロ**（writer ゼロのため恒真）。Publish 滞留の必要情報は `publicationIntentResidencyCount_==0`(INV-X5-1) が既に提供済み

> 過去設計資料の「isFullyDrained から setPublicationBacklogCount を呼ぶ旧実装」は現行ソースに
> 存在しないことを再確認（resurrection 監査完了）。

---

## 4. F-5 setPendingIntentCount の確定

| 項目 | 確定内容 |
|---|---|
| authoritative writer | semantic event 群（§2.2 の fetchAdd/fetchSub 12 site — 全て reservation-before-push 構造） |
| increment event | Observe/Recovery/Quarantine intent の enqueue reservation |
| decrement event | consumer pop 成功（processIntent / drainObserveDeferred / popRecoveryRequest）＋ push 失敗 rollback |
| underflow protection | 構造的保証（reservation-before-push。cur>0 ガードは silent-hide 危険のため意図的廃止 — P2-1 §1.1.4） |
| producer join / lifetime | X1 durable admission は transport と二重計上しない（INV-X1-6）、coalesce でも reservation 増やさない（INV-X1-5）→ producer quiescence 後に ==0 が成立 |
| isFullyDrained での必要性 | **必須**（非 Publish intent の drain 条件として authoritative） |
| test-only setter の必要性 | **低い** — 使用は reset-to-0 ×3 のみ（:328/:364/:434）。非零注入テストは存在せず、invariant/soak 系は実アカウンティング（enqueue/pop）で駆動 |
| setter 削除で失われるもの | 上記 reset 3行のみ（fresh instance で代替可 — D101-32-D と同一基準）。失われる invariant/test なし |

> **pendingIntentCount_ 自体は C と断じていない**: counter・getter・event 群は
> A（KEEP）— INV-ISR-02 / P2-1 §1.1 の authoritative accounting であり、
> INV3・Soak・SemanticValidation の実テスト契約がある。

---

## 5. F-6 3者比較表（二重計上なしの確認）

| 観点 | publicationBacklogCount_ | pendingIntentCount_ | publicationIntentResidencyCount_ |
|---|---|---|---|
| 何を数えるか | （設計意図: 未処理 publish。実態: 常時0） | 非 Publish Intent（Observe/Recovery/Quarantine）の予約残数 | **Publish Intent 専門**の queue residency + 予約 |
| +1 | setter のみ（test） | 各 intent enqueue reservation | Publish enqueue reservation（単一 choke point） |
| −1 | なし | pop 成功 / push 失敗 rollback | push 失敗 rollback / processIntent pop 成功 |
| zero の意味 | 無情報（恒真） | 非 Publish transport の完全 drain | Publish lane の完全 drain（producer quiescence 後） |
| shutdown drain 必要性 | 不要（恒真項目） | 必須 | 必須（INV-X5-1） |
| producer admission との関係 | なし | reservation-before-push で直接連動 | reservation-before-push で直接連動 |
| 二重計上 | —（死んでいる） | Publish は含まない（INV-ISR-02） | Publish のみ（lane 分離）→ **互いに素、二重計上なし** |

---

## 6. F-7 Layer 帰属と F-8 テスト依存性

### Layer 帰属

```
Layer 1（AudioEngine::isFullyDrained — 実測 authority）:
    publication/pending 系 Coordinator counter は一切参照しない
    （lifetime().pendingIntentCount() は RetireRuntime 側の別カウンタ — 混同注意）

Layer 2（bridge.isFullyDrained — transport/reservation accounting）:
    pendingIntentCount_            … A KEEP（非 Publish lane）
    publicationIntentResidencyCount_ … A KEEP（Publish lane, X5）
    publicationBacklogCount_       … C REMOVE候補（恒真項目）

Bridge 設計思想との整合: RT は所有・判断せず、Bridge/Coordinator 側で観測可能な
状態として完全 Drain を保証 — 本構造はこれに合致。Layer 1 実測と Layer 2 accounting
の混同なし。
```

X1/X5/X6 干渉: pendingIntentCount_ は X1（Recovery rollback→durable）および X6
（quarantine rollback/fallback 移動）と正しく協調（コメントで INV-X1-2/5/6、INV-X6-4 明記）。
publicationIntentResidencyCount_ は X5 専用。干渉なし。

### テスト依存性分類

| シンボル | テスト用途 | 分類 |
|---|---|---|
| setPublicationBacklogCount(0) ×3 | reset-only | fresh instance 代替可 → **削除候補** |
| setPendingIntentCount(0) ×3 | reset-only | fresh instance 代替可 → **削除候補** |
| getPendingIntentCount | INV3（容量/予約会計 8件）・Soak（容量会計 3件）・SemanticValidation:819（N intent）・HealthMonitorTier | **意味テストの本体 — KEEP** |
| getPublicationBacklogCount | DeferredFlow/SoakPublish の wait ルール一部（他条件が live なため単独では恒真）・HealthMonitorTier stub | reader 側で REWIRE 要検討 |
| RuntimeHealthMonitorTierTests.cpp:226 | getter の tier stub（return 0） | getter 変更時は同時修正 |

---

## 7. F-9/F-10 A/B/C 判定表 + D101-33-A 引き渡し

| API / field | 判定 | 理由 | 次作業 |
|---|---|---|---|
| `publicationBacklogCount_` | **C — REMOVE 候補** | writer ゼロ（dead counter・header 自認）。恒真 drain 項目。実 semantic は X5 residency が担当済み | 実装タスク（本監査では削除しない） |
| `setPublicationBacklogCount` | **C — REMOVE 候補** | production caller ゼロ・test reset-only ×3 | 実装タスク（test reset 行の整理込み） |
| `getPublicationBacklogCount`（Coordinator/AudioEngine/Orchestrator の3段チェーン） | **B — REWIRE** | reader が生データを期待： HealthMonitor stall 検出第3項（現在恒偽）、Threading.cpp:78 診断、Harness wait ループ。REWIRE 先 = publicationIntentResidencyCount_ 経由の live 値、または項目削除の設計判断 | **D101-33-A 前に方針確定を推奨** |
| `pendingIntentCount_` | **A — KEEP** | INV-ISR-02 authoritative reservation accounting。INV3/Soak/Semantic の実テスト契約 | 変更なし |
| `getPendingIntentCount` | **A — KEEP** | HealthMonitor hasPendingWork（live）＋意味テスト本体 | 変更なし |
| `setPendingIntentCount` | **C — REMOVE 候補** | reset-only ×3・fresh instance 代替可・絶対値リセット廃止方針（P2-1 §1.1.2）と整合 | 実装タスク |
| `publicationIntentResidencyCount_` | **A — KEEP** | X5 §6.5 の live authoritative residency（INV-X5-1）。単一 choke point・二重計上なし | 変更なし |

### D101-33-A への引き渡し条件判定

> **✅ 確定した。** Path B Publication Admission Audit が依拠すべき authority 境界：
>
> 1. **Publish admission の transport authority = `publicationIntentResidencyCount_`**
>    （単一 choke point: enqueuePublicationIntent の reservation→push→rollback）+
>    `state_ == ShuttingDown` gate（h:362、defense-in-depth 二次防衛）。
> 2. **`publicationBacklogCount_` は live accounting ではない** — Path B の admission
>    transaction 設計は本 counter に依拠してはならない（依拠する場合は先に B-REWIRE を完了）。
> 3. **`pendingIntentCount_` ≠ `publicationIntentResidencyCount_`**（lane 分離: 非 Publish /
>    Publish）。Path B は Publish lane を扱うため後者を使用。
> 4. HealthMonitor の `getPublicationBacklogCount()>0` 項は現在恒偽 — Path B 設計時に
>    stall 検出条件の live 化（B-REWIRE）を同時検討すること。

---

## 8. PASS 条件チェックリスト

- [x] 最新 ConvoPeq.md 再生成・基準化（2026-08-25 12:07 版）
- [x] publication backlog の writer/reader 全件列挙（§2.1）
- [x] pending intent の writer/reader 全件列挙（§2.2 — atomic ops 12 site 特定）
- [x] test-only setter 全件確認（各3箇所 reset-only）
- [x] pendingIntentCount_ と publicationIntentResidencyCount_ の semantic 分離確認（§5 — lane 分離・二重計上なし）
- [x] publicationBacklogCount_ の drain proof 観点判定（無情報・恒真）
- [x] setPublicationBacklogCount の production/test 境界確定（§3）
- [x] setPendingIntentCount の production/test 境界確定（§4）
- [x] Layer 1 / Layer 2 authority 境界確認（§6）
- [x] X1/X5/X6 干渉なし確認（§6）
- [x] 削除候報ありだが削除せず inventory audit のみ
- [x] ソースコード変更 0
- [x] D101-33-A への authority boundary 明文化（§7）

# VERDICT: D101-32-F = **PASS**

## 9. リスク・注意事項

- `getPublicationBacklogCount` チェーンの B-REWIRE は**読み値の意味が変わる**変更
  （恒偽→live）のため、HealthMonitor の stall 検出閾値・Harness 待ちループへの影響を
  実装前に評価すること（本監査では実施しない）。
- `pendingIntentCount_` の fetch_sub には old>0 ガードがないが、これは意図的設計
  （reservation-before-push 構造的保証）。将来の変更者が「バグ」と誤認しないよう、
  既存コメント（cpp:944-949）の維持を推奨。
