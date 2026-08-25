# D101-32-C — Semantic Event / Vestigial Counter Integrity Audit（実施報告書）

- **実施日**: 2026-08-25
- **性質**: read-only audit（コード変更禁止 / 実績: ソースコード変更 **0**）
- **判定**: **PASS**（BLOCKER なし。D101-32-D の削除境界を §7 テーブルとして確定）
- **基準**: ConvoPeq.md **2026-08-25 10:05 再生成版**（D101-32-A 使用版 09:48 より新しいため本版を基準）
- **前提**: D101-31-D PASS / D101-32-A PASS

---

## 1. C-1: 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 10:05)
git status                             → src/CMakeLists/build.bat の変更は D101-31-D 分のみ（監査中の追加変更なし）
git diff --check                       → whitespace エラーなし
```

---

## 2. C-2: 5 setter の TEST-ONLY 利用 全件監査

### 2.1 test call site 全件（17箇所・`src/tests/ISRSemanticValidationTests.cpp` のみ）

| 行 | 呼び出し | 目的（semantic purpose） |
|---|---|---|
| :326, :400, :406, :409, :412, :432 | `setRetireBacklogCount(0)` | baseline reset / pressure 正規化ステップ駆動 |
| :361 | `setRetireBacklogCount(1)` | **drain 条件違反の注入**（isFullyDrained()==false を強制） |
| :394 | `setRetireBacklogCount(9)` | **Pressure FSM 遷移注入**（slope=9 > kPressureSlopeThreshold=8） |
| :329, :364, :435 | `setFallbackBacklogCount(0)` | reset-to-0 のみ（**非零注入なし**） |
| :330, :365, :436 | `setReclaimInFlightCount(0)` | reset-to-0 のみ |
| :331, :366, :437 | `setDeferredRetireResidencyCount(0)` | reset-to-0 のみ |
| — | `setQuarantineResidentCount` | **呼び出しゼロ**（テストにも存在しない） |

補足: `invariant_INV3_INV5.cpp:764` はコメント言及のみ。setter を一切使わず
`onReclaimBegin/End` を直接呼んで `getReclaimInFlightCount()`（getter）で検証する
正しい semantic event 方式（:140/:176/:186/:282）。

### 2.2 setter ごとの削除影響と可否

| setter | getter/drain への依存 | 削除時に壊れるテスト | test seam 置換 | API 削除可否 |
|---|---|---|---|---|
| setRetireBacklogCount | 有（drain 条件 + Pressure FSM 入力） | drain違反注入(:361)、Pressure遷移/正規化/swapPending(:394-412) 全滅 | 可だが高コスト（friend seam でも結局 publishAtomic+note 呼びが必須 = 実体は現 setter と同一） | **不可（KEEP 推奨）** — Pressure FSM の唯一の決定論的駆動手段 |
| setFallbackBacklogCount | 恒真 drain 条件のみ | なし（reset は fresh instance で代替可） | 不要 | **可** |
| setReclaimInFlightCount | 恒真（production は event 経由で更新） | なし（invariant 系は event 直接使用） | 不要 | **可** |
| setDeferredRetireResidencyCount | 恒真 | なし | 不要 | **可** |
| setQuarantineResidentCount | 恒真 | なし（**呼び出し元皆無**） | 不要 | **可（最優先）** |

> 「単純にテストだからOK」ではない点の明確化: `setRetireBacklogCount` は
> **絶対値注入がテストの意味本体**（slope 違反・swapPending 抑制は fetch_add の連打では
> 決定論的に作れない）であり、これは正当な test-only 契約。一方 fallback/deferred/
> reclaim/quarantine の reset 用途は fresh instance で完全代替可能で、残存理由がない。

---

## 3. C-3: setRetireBacklogCount → noteRetireBacklogChanged 副作用の重点監査

### 3.1 実装確認（ISRRuntimePublicationCoordinator.cpp:257-283）

```text
noteRetireBacklogChanged(count):
    slope = max(0, count - previousRetireBacklogCount_)
    previousRetireBacklogCount_ ← count
    if slope > kPressureSlopeThreshold(=8, h:707):
        pressureNormalizedWindows_ ← 0; state_ ← Pressure        // cpp:263-266
    elif !isSwapPending():
        state==Pressure → window++、kPressureNormalizeWindows(=3, h:708) 到達で Ready 復帰  // cpp:269-281
        state==Publishing && count==0 → Ready                     // cpp:282
```

### 3.2 production call graph（実測）

```text
noteRetireBacklogChanged の caller:
  (1) setRetireBacklogCount            … TEST-ONLY（production 呼び出しゼロ）
  (2) onRetireAccepted (cpp:195)       … 唯一の caller は enqueueRetireIntent(cpp:171)
                                          ＝ コード自身が「production で未使用」と明記

CoordinatorState::Pressure の reader:
  noteRetireBacklogChanged 内部 (cpp:265/:271) のみ
  ＝ production のどこからも Pressure を観測していない
     （RuntimeHealthMonitor / CoordinatorLoop / AudioEngine に参照なし）

CoordinatorState::Publishing の reader:
  cpp:111(setter側遷移) / cpp:282(note内) のみ
```

### 3.3 必須判定

> **判定: A — `noteRetireBacklogChanged()` は現在の production semantic に不要
> （production 不到達かつ結果不観測の慣性 FSM）。**
>
> B（authoritative event/measurement からの rewire）は**採らない**:
> production には「retire backlog の絶対値」を生成する authoritative producer が
> 存在しない（Layer 1 が router 実測直接判定のため、dash2 §1.4 設計どおり）。
> fetch_add イベント列から slope を再構築するのは windowing 再設計を要する割に、
> 誰も Pressure を読まないため利益がない。

**D101-32-D での扱い**: `noteRetireBacklogChanged` + Pressure FSM 一式
（previousRetireBacklogCount_ / pressureNormalizedWindows_ / kPressureSlopeThreshold /
kPressureNormalizeWindows）は **KEEP（テスト網羅された設計機構として保持）**。
ただし header に「production では現在不到達（onRetireAccepted 接続時に活性化）」の
注記追加を D で行うことを推奨（doc-only）。setter 削除時もこの経路は onRetireAccepted
経由で生存するため、**機能欠落はない**。

---

## 4. C-4: vestigial counter 4種の個別判定

追跡形式: writer → counter → getter → isFullyDrained/shutdown/pressure/diagnostics → necessity

| counter | writer（全列挙） | reader | 判定 |
|---|---|---|---|
| `retireBacklogCount_` | setRetireBacklogCount(test) / onRetireAccepted(prod不到達) | drain条件 cpp:571（prod恒真・test注入で有効化）/ getter(getRetireBacklogCount, 外部caller なし) / previousRetireBacklog(pressure FSM) | **KEEP — 将来契約として明示的に予約**（onRetireAccepted 将来retire経路 + Pressure FSM 入力 + テスト依存。「将来用」だけでなく稼働中テスト契約がある点が他と異なる） |
| `fallbackBacklogCount_` | set...(test reset-to-0のみ) / onFallback*(callerゼロ) | drain条件 cpp:578（恒真）/ getter(外部caller なし) | **REMOVE — dead/vestigial**（非zero注入テストすら不存在。fallback 実測は Layer 1 ringResident + quarantineFallbackQueue_.sizeApprox() が担当済み） |
| `deferredRetireResidencyCount_` | set...(test reset-to-0のみ) / onDeferredRetire*(callerゼロ) | drain条件 cpp:581（恒真）/ getter(外部caller なし) | **REMOVE — dead/vestigial**（deferral 実測は observeDeferredRing_.size() / router 側が担当済み） |
| `quarantineResidentCount_`(Coordinator) | **setQuarantineResidentCount の呼び出し元が皆無**（writer ゼロ） | drain条件 cpp:587（恒真）/ getter(外部caller なし — Retire.cpp:156 の同名 getter は EpochControl の別物) | **REMOVE — pure dead** |

補足（対象外だが同時検討推奨）:
- `publicationBacklogCount_`: header 自ら「TEST-ONLY（dead counter）」と明記。
  writer=setter(test)のみ、reader=getter(AudioEngine.h:3481 で diagnostic 的に露出)+drain条件:572(恒真)。
  D101-32-D で getter 露出(:3481)の帰趨と合わせて削除判定すること。
- `previousRetireBacklogCount_` / `pressureNormalizedWindows_`: §3 の通り KEEP（FSM 一式）。

---

## 5. C-5: semantic event API 8種の integrity

| Event | production caller | 発生させる実際の semantic event | counter observation | 接続判断 | 残す/消す根拠 |
|---|---|---|---|---|---|
| onRetireAccepted | なし（enqueueRetireIntent 内のみ＝prod未使用メソッド） | retire 受諾（将来retire経路用とコード明記 cpp:168-171） | retireBacklogCount_→Pressure FSM | **接続しない** | authoritative な受諾イベント発生源が現状存在しない（Layer 1 実測方式が確定設計）。enqueueRetireIntent 自体が将来経路用プレースホルダ |
| onRetireConsumed | なし | retire 消費 | 同上 | 接続しない | Accepted と対。underflow guard→Faulted 実装済み。**KEEP(予約)** |
| onFallbackAccepted | なし | — | fallbackBacklogCount_ | 接続しない | **REMOVE** — counter 削除なら orphan。fallback は Layer 2 queue emptiness で観測済み。将来必要ならその時点の設計で再作成（dash2 §1.4 の後退ではなく Layer 1 実測方式への収束完了） |
| onFallbackConsumed | なし | — | 同上 | 接続しない | **REMOVE**（対） |
| onDeferredRetireAccepted | なし | — | deferredRetireResidencyCount_ | 接続しない | **REMOVE** — deferred 実測は observeDeferredRing_.size() |
| onDeferredRetireConsumed | なし | — | 同上 | 接続しない | **REMOVE**（対） |
| onReclaimBegin | ✅ Retire.cpp:62(drain)/:356(emergency boost) + Coordinator内部 :736(requestReclaim defer) | reclaim 保留開始（fetch_add） | reclaimInFlightCount_ → drain 条件 | **接続済み** | **KEEP — production wired** |
| onReclaimEnd | ✅ Retire.cpp:66/:359 + Coordinator内部 :747(ACK) | 保留解消（old>0 ガード付き fetch_sub、単発成功は正常系 no-op — INV-3-1） | 同上 | 接続済み | **KEEP** |

整合性チェック: fallback/deferred の event を REMOVE する場合は counter・setter・getter・
drain 条件項目と**ドメイン単位で一括削除**すること（部分削除はコンパイル破壊または
orphan を生む）。

---

## 6. C-6: Drain 条件の意味の再検証

### 6.1 ShutdownScheduler::isFullyDrained()（Coordinator.cpp:569-593）

| 条件項目 | 分類 |
|---|---|
| intentQueue_ / observeDeferredRing_ / quarantineFallbackQueue_ / recoveryIntentQueue_ の sizeApprox()==0 | ✅ authoritative（queue 実測） |
| publicationIntentResidencyCount_==0 | ✅ authoritative（reservation ベース、INV-X5-1） |
| pendingIntentCount_==0 | ✅ authoritative（push 成功 fetchAdd / pop 成功 fetchSub） |
| reclaimInFlightCount_==0 | ✅ authoritative（wired event による唯一の非恒真カウンタ） |
| recoveryAdmissionPending_==false | ✅ authoritative（INV-X1-1/2） |
| quarantineIntentResidencyCount_==0 / quarantineRingResidencyCount_==0 | ✅ authoritative（X6 新設の transport residency。writer あり — submitQuarantine 経路） |
| retireBacklogCount_==0 | ⚠️ prod 恒真（test 注入時のみ有効化）— KEEP 予約に付随して維持 |
| publicationBacklogCount_==0 | ❌ 恒真（dead counter）— D で同時検討 |
| fallbackBacklogCount_==0 / deferredRetireResidencyCount_==0 / quarantineResidentCount_==0 | ❌ 恒真 — REMOVE 対象（削除時に条件行ごと除去） |

呼び出し経路: production は `AudioEngine::isFullyDrained()` → `runtimePublicationBridge_.isFullyDrained()`（Threading.cpp:173）経由のみ。Coordinator::isFullyDrained ラッパ（cpp:506）の直接 caller はテスト3箇所（ISRSemanticValidationTests.cpp:334/:369/:440）。

### 6.2 AudioEngine::isFullyDrained()（Threading.cpp:114-174）

全項目が実測直接判定で **vestigial 項目を含まない**:

```text
!hasDeferredCommit && pendingReclaimHandles_.empty()
&& retireDepth==0(router) && lifetimeRetireIntentPending==0
&& ringResident==0(overflow ring) && dspQuarantineResident==0(DSPQuarantineManager)
&& retireQuarantineResident==0(EpochControl Q+EmergencyQ)
&& terminalReclaimResident==0(Terminal 層 — 15-P-5 premature waitForDrain 防止)
&& runtimePublicationBridge_.isFullyDrained()   ← Layer 2 transport 視点の重複ではなく補完
```

二層構造の意味: Layer 1 = 実在資源の実測、bridge(Layer 2) = Intent transport の残留。
重複ではなく直和。両者が揃って初めて「producer join 済み時点での完全 drain」を証明する。

### 6.3 quarantine 4 domain の分離確認（混同なし）

| domain | 観測手段 | drain 条件内の項目 |
|---|---|---|
| Quarantine Intent（primary transport） | quarantineIntentResidencyCount_（Coordinator） | ✅ 独立項目 |
| Quarantine Intent（fallback/ring） | quarantineRingResidencyCount_（Coordinator） | ✅ 独立項目 |
| 実在 quarantine DSP | DSPQuarantineManager::residentCount()（Layer 1 直接） | dspQuarantineResident |
| Q + EmergencyQ 滞留 | EpochControl::quarantineResidentCount()（ISRRetireRuntimeEx.h:104 — Coordinator とは別クラス・別semantic） | retireQuarantineResident |
| Terminal 層 | terminalReclaimResident（15-P-5） | terminalReclaimResident |

REMOVE 対象の `quarantineResidentCount_`（Coordinator メンバ）は上記のいずれでもなく、
domain mixing 解消（X6 §6.6）により writer を失った残骸であることを確認。

---

## 7. C-7: D101-32-D 削除境界テーブル（最終成果物）

| API / field | Decision | 理由 | 次作業 |
|---|---|---|---|
| `setRetireBacklogCount` | **KEEP（TEST-ONLY 明示化）** | Pressure FSM(slope>8 注入)・drain違反注入(:361) の唯一の決定論的駆動手段。production 参照 0 はコンパイル時に保証済み。seam 化は実体が同一になるため無益 | D で header 注記強化（doc-only 可） |
| `setFallbackBacklogCount` | **REMOVE** | test 用途 = reset-to-0 のみ（fresh instance で代替）。writer なし domain の一部 | D |
| `setReclaimInFlightCount` | **REMOVE** | 同上。production は onReclaimBegin/End が authoritative（wired） | D |
| `setDeferredRetireResidencyCount` | **REMOVE** | 同上 | D |
| `setQuarantineResidentCount` | **REMOVE（最優先）** | 呼び出し元ゼロ（test 含む）。pure dead | D |
| `retireBacklogCount_` | **KEEP — 将来契約として明示的に予約** | onRetireAccepted 将来経路 + Pressure FSM 入力 + 稼働中テスト契約（:361/:394 系）。恒真 drain 項目は注記付き維持 | 変更なし |
| `fallbackBacklogCount_` | **REMOVE** | writer=未接続 event + test-reset のみ。reader=恒真 drain 項目のみ | D（drain 条件行とセット） |
| `deferredRetireResidencyCount_` | **REMOVE** | 同上 | D（同上） |
| `quarantineResidentCount_` | **REMOVE** | writer ゼロ。X6 §6.6 mixing 解消の残骸。4 quarantine domain のいずれでもない | D（同上） |
| `onRetireAccepted/Consumed` | **KEEP（予約・明文化）** | enqueueRetireIntent（将来 retire 経路）との対。underflow guard 実装済み。retireBacklogCount_ KEEP と整合 | 変更なし |
| `onFallbackAccepted/Consumed` | **REMOVE（counter と一括）** | caller ゼロ。counter 削除で orphan 化。fallback 観測は queue emptiness + Layer 1 で完結 | D |
| `onDeferredRetireAccepted/Consumed` | **REMOVE（counter と一括）** | 同上（observeDeferredRing_ 実測で完結） | D |
| `onReclaimBegin/End` | **KEEP** | production wired（Retire.cpp:62/66/:356/359 + Coordinator:736/:747） | 変更なし |

### 7.1 D101-32-D への付帯指示（確定事項）

1. **ドメイン単位の一括削除**: fallback domain = {setFallbackBacklogCount,
   fallbackBacklogCount_, onFallbackAccepted/Consumed, getter, drain条件行}。
   deferred domain 同様。quarantine domain = {setQuarantineResidentCount,
   quarantineResidentCount_, getQuarantineResidentCount(Coordinator), drain条件行}。
   部分削除禁止（コンパイル破壊/orphan 防止）。
2. **触らないもの**: retireBacklogCount_ 域、Pressure FSM 一式、onRetire*、
   onReclaim*、publicationBacklogCount_（§7.2）、pendingIntentCount_ 系。
3. **テスト修正**: ISRSemanticValidationTests.cpp の該当 reset 行（:329-331/:364-366/
   :435-437）を削除または fresh-instance 化。:326/:361/:394-412/:400-432（retireBacklog）は**維持**。
4. **検証**: Debug/Release build + 全 CTest（D101-32-E で実施）。
5. **同時検討候補（スコープ判断は次回指示時）**: publicationBacklogCount_（dead counter、
   getter が AudioEngine.h:3481 で露出）/ setPublicationBacklogCount /
   setPendingIntentCount の取り扱い。

---

## 8. PASS 条件チェックリスト

- [x] 最新 ConvoPeq.md 再生成（2026-08-25 10:05 版を基準化）
- [x] 5 setter の test-only call site 全件確認（17箇所 + setQuarantineResidentCount は 0件）
- [x] setRetireBacklogCount → noteRetireBacklogChanged の production semantics 確定（**判定 A: 現 production 不要・慣性 FSM**）
- [x] 4 vestigial counter を個別分類（KEEP予約1 / REMOVE3）
- [x] 8 semantic event の production wiring 個別確認（wired 2 / KEEP予約 2 / REMOVE 4）
- [x] isFullyDrained() の各 ==0 条件の意味確認（authoritative 7 / 恒真 5、二層構造の直和性確認）
- [x] quarantine 各 residency domain の混同なし確認（5 domain 分離表）
- [x] D101-32-D 削除境界の確定（§7 テーブル + 付帯指示5項目）
- [x] コード変更 0
- [x] **PASS / BLOCKER なし**

## 9. リスクと注意

- **最大のリスク**: REMOVE 判定域を「使われていないから」と楽観して部分削除すると
  未接続 event が counter を参照したまま残りコンパイルが壊れる（逆方向は orphan）。
  §7.1-1 のドメイン一括ルールを厳守すること。
- setRetireBacklogCount を KEEP する理由は「将来用」ではなく**稼働中テストの決定論的
  駆動手段**である点（ユーザーの「将来用だけでは不十分」基準を満たす区分）。
- 本監査で使用: rg/grep/sed(WSL), git, AiDex(session/note), ctx_batch_execute/ctx_search
  （stale セッションデータを検出し fresh grep で全面再検証）。
