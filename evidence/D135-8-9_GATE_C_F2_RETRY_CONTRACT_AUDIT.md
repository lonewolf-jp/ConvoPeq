# D135-8/9 Gate C-F2 — Deferred Retry Semantic Contract Audit

Date: 2026-08-30
Scope: **production/test source 0変更**。設計契約の確定のみ。F1（evidence/D135-8-9_GATE_C_F1_FORENSIC_AUDIT.md）の実測トレースを入力とする。
Verdict: **F2 = 成立（10/10 項目確定）。推奨 = Option A（fading-active re-drive を retry budget 外＝保持意味論に変更、既存 30s TTL + generation supersession + shutdown を safety bound として明示）。実装は承認後。**

---

## 1. 現行 retry state machine（実ソース照合済み）

```text
[RebuildThread] build完了 → submitPublishRequest(req)
  └ 主 admission evaluate → DeferredFadingActive (fading 判定はここだけ, PublicationAdmission.cpp:51-58)
      └ enqueueDeferred(req) (Orch:437)
          ├ retry accounting (Orch:470-503)
          │    sameObligation(gen一致) → ++deferredRetryCount_
          │    else → generation_=req.generation, count_=0
          │    if (count_ > kMax=2) → retireDSPHandleForRuntime(req.newDSP) → return   ←【破棄終端】
          ├ slot struct-replace (INV-DEFERRED-2) + hasDeferred_=true
[1ms Coordinator tick] publishRetryReady=true → rebuildCV wake
[RebuildThread wake] (RebuildDispatch:850-920)
  ├ wasRecoveryWake = exchangeAtomic(recoveryRetryReady,false)   ← provenance (Step 6)
  ├ drainDeferredClearIfRequested()                              ← Step 9 latch
  └ processDeferredAdmission(wasRecoveryWake) (Orch:673)
      ├ if (wasRecoveryWake) resetDeferredRetryBudget()          ← START で reset (Step 7 / P3)
      ├ peekDeferred → evaluateDeferred (Orch:601, PA.cpp:72-101)
      │    判定順序: Shutdown → TTL(30s) → Generation → Sequence   ← ★fading は見ない
      │    → fresh なら Ready (ほぼ常時)
      ├ Ready → view.consume() (=move-out + finishView: slot reset, hasDeferred_=false)
      └ submitPublishRequest(req) → 主 admission
          ├ Accepted        → publish →【Success 終端】(fading クリアで到達)
          ├ DeferredFadingActive → enqueueDeferred → count++ → ループ    ←【churn loop】
          └ Rejected*       → telemetry のみ、**再 enqueue なし** → 要求消滅（後述 F2-5 注記）
```

要点: ループ 1回転ごとに (a) slot-level evaluateDeferred は **Ready**（fading を評価しないため）、
(b) 主 admission が **DeferredFadingActive** を返し、(c) `count_` が **1 増える**。
すなわち **fading が続く限り count は ~1ms ごとに増加し、T_exhaustion ≈ 4×tick ≈ 3-4ms**（F1 実測）。

## 2. retry / re-drive の定義（F2-1）

| 種別 | 定義 | 現行コードの扱い |
| --- | --- | --- |
| **Type A — 真の retry** | publish attempt が失敗し、同一要求を再試行する | 専用経路 **存在しない**（Ready→submit 失敗時は再 enqueue されず要求消滅。§5 注記） |
| **Type B — Deferred retention / re-drive** | 条件未成立（fading active）により同一要求を保持し再評価する | **Type A と同一の budget で ++count_ される**（Orch:470-478） |

F1 実測の cycle 1 は完全に Type B（要求は一度も publish attempt に失敗していない。admission が
Deferred を返し続けただけ）。**現行コードは Type A/B を区別する情報を一切保持していない**
（admission の Decision は enqueueDeferred に渡されない）。

## 3. `kMaxDeferredRetries` の off-by-one（F2-2 照合結果）

**4者の契約記述が 2 派に分裂している:**

| 出典 | 条件 | 意味論 |
| --- | --- | --- |
| **実コード** Orch.cpp:489 | `if (deferredRetryCount_ > kMaxDeferredRetries)` | count=3 で破棄 |
| ヘッダコメント Orch.h:289「2回再駆動許容、3回目で諦め」 | （`>` と整合） | 3回目の re-drive で死ぬ |
| **D135-1 設計文書** doc/work88/D135-1_IMPLEMENTATION.md:25 | **`count >= kMax(2)`**、ログ例 `retryCount=2` | count=2 で破棄 |
| enum コメント RuntimePublicationState.h:17「retry-cap に**到達**したため」 | （`>=` と整合） | count=2 で死ぬ |

**F1 実測ログ**（diag run1/run2）: `0 → 1 → 2 → 3 → [HEALTH] starved retryCount=3` — 実コード `>` と整合。
ユーザー指摘の「`>=` で 2回目で discard」は **working tree とは不一致**（文書側の記述）。

数式化（実コード `>`、kMax=2、初回 defer 込み）:

```text
enqueue #1 (初回 defer)      count=0   保持
re-drive #1 → enqueue #2    count=1   保持
re-drive #2 → enqueue #3    count=2   保持
re-drive #3 → enqueue #4    count=3 > 2 → RetryExhaustedDiscard（破棄）
```

つまり「**初回 defer + 2回の re-drive は生存、3回目の re-drive の enqueue が拒絶され要求破棄**」。
コードと h:289 コメントは自己整合、**D135-1 文書と enum コメントが `>=` 意味論で +1 のドリフト**。
→ 実装前に契約を一箇所に統一する必要がある（どちらを採っても本質問題 §2 は残るため、
統一は実装パッチ内で行う。推奨は文書を `>` 意味論に合わせ現コードを正とする）。

**帰属に関する未確定事項**: D135-1 文書 §7-D は「ctest -C Release 40/40 PASS（AudioEngineHarness 14.99s）」
を主張 — accounting + kMax=2 導入済みの状態で Release が PASS していたことになる。現ツリー（D135-8 Steps
1-9 追加後）は決定論的 FAIL。差分は (i) D135-8 での codegen 変化による 1ms vs 1ms レースの反転、
(ii) D135-1 当時の kMax 実効値ドリフト（D135-3 監査記録: 当時 source h:279 は kMax=10、spec=2）、のいずれか。
→ 実装フェーズで D135-1 時点ビルドの再現確認（1回の harness 実行）により特定可能。F2 の結論に影響しない
（いずれにせよ T_exhaustion 3-4ms vs 観測窓 45s の構造的衝突が本質）。

## 4. ownership / terminal transition（F2-5）

| Terminal | トリガ | slot/要求 | DSP handle の行き先 | ownership 判定 |
| --- | --- | --- | --- | --- |
| **Success** | 主 admission Accepted → publish | consume() で move-out + finishView（slot 解放） | live runtime world へ昇格 | ✅ 正常 |
| **SupersededDiscard** | enqueueDeferred の上書き（Orch:460-468） | 新 slot が struct-replace | **旧 slot の req.newDSP を retire**（resolveDSPHandle→retireDSPHandleForRuntime） | ✅ 明示 retire 済み |
| **ShutdownDiscard** | clearDeferredForShutdown（Orch:534）| slot reset | **retire なし**（要求は slot ごと破棄） | ⚠ shutdown 全体 teardown に依存（per-request retire しない）。shutdown 文脈では許容だが依存関係として明記すべき |
| **StaleDiscard** | evaluateDeferred (Generation/Sequence) | view.discard → finishView（slot ごと破棄） | **retire なし**（slot 内要求ごと破棄） | ⚠ 要 follow-up: 破棄された build の DSPCore の回収経路（rebuild-flow 側の obsolete retire）を別監査で確認 |
| **Expired** | — | — | — | **未使用**。TTL 超過は `StaleDiscard` として返る（PA.cpp:76-77、注記「Expired を別 enum 化可能」）。dead enum 値 |
| **RetryExhaustedDiscard** | Orch:489-502 | **slot に入る前**に拒絶（return、slot 無関係） | **明示 retire**: handle map erase → `Coordinator::requestReclaim`（waitReaders(epoch 安全) → reclaim、失敗時 RetireQuarantineStore。AudioEngine.h:4329-4351） | ✅ ownership release は正規 retire pipeline を通る（NonRT 隔離・epoch 安全・quarantine fallback — I4/INV-DEFERRED-2/D131-G2 整合） |

**RetryExhaustedDiscard の ownership 証明**: 破棄対象は「slot 未介入の incoming req.newDSP」であり、
retire は Coordinator 専用 reclaim 経路（DELETE-1/P0-4B 契約）を通る。**ownership invariant 違反ではない**。
問題は ownership ではなく **terminal になるべきでない要求が terminal 化していること**（§2 の意味論混同）。

**§5 注記（別経路の義務喪失リスク）**: Ready→consume→submitPublishRequest が `RejectedPressure` 等
で拒絶された場合、要求は**再 enqueue されず消滅**する（Orch:403-414: rearm/telemetry のみ）。
deferred obligation の「消滅経路」は kMax 以外にも存在する。F2 の修正設計ではこの経路も同じ
意味論枠組みで扱うこと（本監査では範囲外として記録）。

## 5. ordinary / recovery wake の4分類（F2-7）

| Wake 種別 | 現行の budget 扱い | 根拠 |
| --- | --- | --- |
| ordinary wake（publishRetryReady） | **increment**（churn がここで count++） | Orch:472-474（gen 一致） |
| recovery wake（recoveryRetryReady=true） | **reset**（peek 前に resetDeferredRetryBudget） | Orch:687-688（Step 7/P3、F1 で未発火を確認） |
| fading-active retention（Type B） | **increment**（← 本監査の主問題。保持を失敗として数える） | 同上経路 |
| Ready→publish failure | **increment なし・要求消滅**（budget に触れない終端） | Orch:427-432（RejectedPublishFailure、再 enqueue なし） |

recovery wake と ordinary wake の分離（D135-8 provenance）は正しく機能しており、変更不要。
問題は3行目のみ。

## 6. F2-3: starvation bound は何を bound しているか（4定義）

| 項目 | 現行の定義 | 提案する確定定義 |
| --- | --- | --- |
| **Obligation** | `req.generation`（`++rebuildRequestGeneration`、RebuildDispatch:655）＝1件 | 同左（世代 = obligation identity。D135-1 文書 §2 も generation-keyed を明示） |
| **Retry** | 「同一 generation の enqueue 2回目以降」＝**Type B も含む** | **publish attempt の再試行のみ**（Type A）。Ready→submit 失敗で初めて定義される |
| **Re-drive** | coordinator 1ms tick による processDeferredAdmission 再実行 | 同左。**保持（retention）であって失敗ではない** |
| **Exhaustion** | `count_ > kMax` → 要求破棄（RetryExhaustedDiscard） | 「同一 obligation が bound を超えても publish に至らない」こと。**対象は retry であり、保持ではない** |

設計意図の裏取り: RuntimePublicationState.h:16-20 は RetryExhaustedDiscard を
「**D133-1 の 28/19-iteration busy-loop を有限回に停止させる唯一の ownership-release path**」と定義する。
つまり cap の本来の目的は **無限 churn の停止**であり、それは (i) count-cap でも (ii) 時間 bound でも達成できる。
**既に存在する別の bound**: 30s TTL（dwell bound, PA.cpp:75-77）、generation supersession（新 rebuild で
StaleDiscard）、shutdown clear。ゆえに count-cap を churn から外しても無限 re-drive は成立しない（F2-8 参照）。

## 7. F2-4: T_observable vs T_exhaustion（実測パラメータ）

| パラメータ | 実測/根拠 |
| --- | --- |
| coordinator tick | ~1ms（RebuildDispatch:910 コメント + F1 ログの re-defer 間隔） |
| T_exhaustion | **≈ 4 × tick ≈ 3-4ms**（初回 defer + 3 re-drive、F1 実測 0→1→2→3→破棄が同一秒内） |
| test poll interval | **1ms**（DeferredFlowIntegrationTests.cpp:51, sleep 1ms） |
| Builder latency | 28-490ms（F1 実測）— 初回 defer 到達を遅らせるのみ、churn には無関係 |
| fade-clear latency | ~0（テストは観測直後に hook=false） |
| 観測窓（テスト契約） | 最大 45s |

**結論: T_exhaustion（3-4ms）≪ T_observable（45s）。** 1ms poll vs 1ms tick のフェアレースで
poll が 4連勝する確率は構造的に低く（実測: 素 Release 0/3、diag 2/3 が coordinator 勝ち）、
**「45秒待つ前に obligation が消える」は偶然ではなく現行パラメータからの帰結**（ユーザー指摘の確認）。
Debug/Release 差は codegen によるレース境界の移動で設計上説明可能（どちらの config でも
T_exhaustion ≪ 45s は不変）。

## 8. Option A〜D 比較（F2-6・実装なし）

| 案 | 内容 | 命中する問題 | 新規リスク | 評価 |
| --- | --- | --- | --- | --- |
| **A — fading-active re-drive を budget 外** | DeferredFadingActive 由来の再 enqueue は count++ しない（保持） | Type B を Type A budget から分離（本監査の主論点に直撃） | 無限 re-drive → **既存 30s TTL + generation supersession + shutdown が bound**（§6）。dwell 観測は既存 `maxDeferredAgeMs_`/DeferredHealth が担う | **第一候補**。意味論が正しく、新規 bound の追加が不要（既存 bound の明示だけで足りる） |
| B — exhaustion しても Discard しない | cap 超過後も保持 + health escalation | 破棄は止まる | **実質永久滞留**（TTL 30s でようやく終端。それまで coordinator が 1ms で回り続ける = D133-1 busy-loop の再発）。escalation 経路の新設が必要 | **不採用推奨**。churn停止という cap 本来の目的を自ら壊す |
| C — retry を Ready 後の失敗に限定 | DeferredFadingActive は budget 触らず。Ready→submit 失敗で count++ | 意味論は最も明快 | 現状 Ready→submit 失敗は**再 enqueue されない**ため count++ する箇所が存在しなくなる（cap が dormat 化）。§5 注記の Rejected 経路の扱いとセットで設計が必要 | **A の意味論を厳密化した形**。A と実質同一路線（A を採るなら将来の Type A 経路導入時に C の定義を採用） |
| D — kMax 2→10 | 観測窓を ~3-4ms → ~11ms に延長 | 表面のみ | **根本修正でない**（F1 実測: pre-D135 の 10 でも長期保持は保証されない。T_exhaustion が伸びるだけ）。しかも契約ドリフト（§3）を残したまま数値だけ動かすことになる | **最後評価・不採用推奨** |

## 9. 推奨案（1つ）

> **Option A を採用** — `DeferredFadingActive` による再 enqueue は「保持（retention）」として
> retry budget に数えない。safety bound は既存の **30s TTL（StaleDiscard 終端）＋ generation
> supersession ＋ shutdown clear** を明示的に依存する。`kMaxDeferredRetries` は将来の Type A
> （真の publish 再試行）経路のための契約として残し、その際は Option C の定義（Ready 後の失敗で
> count++）を採用する。併せて §3 の契約ドリフト（`>` vs `>=`）を文書側で統一する。

根拠: (1) F1 で証明された失敗機構（Type B を失敗として数える）を直接解消する、(2) 新規 bound の
設計が不要（既存 TTL/generation/shutdown で無限 re-drive は構造的に不可能）、(3) D133-1 busy-loop
停止という cap 本来の目的を損なわない（churn は TTL で 30s 以内に必ず終端する。cap より緩いが有限）、
(4) テスト契約（fade クリアまで生存）と production 契約（dwell 30s 上限）が両立する、
(5) ownership 経路（§4）に触れないため I4 影響が最小。

**残るトレードオフ（設計レビューで承認が必要）**: Option A 下では fading が 30s 続いた場合、
要求は TTL で破棄される（現行: kMax で ~4ms で破棄）。ユーザー体感としては「fade 中の publish は
最大 30s 待って成立するか、30s で破棄（telemetry 記録）」という契約になる。D133-1 の busy-loop
（1ms × 30000回）を許容するコストと引き換え。

## 10. 推奨案を実装する場合の変更箇所一覧（実装は未実施）

| # | ファイル:箇所 | 変更内容 |
| --- | --- | --- |
| 1 | Orchestrator.cpp:377-378（DeferredFadingActive → enqueueDeferred 呼出） | retention 由来を enqueueDeferred に伝える（例: `enqueueDeferred(req, /*isRetention=*/true)` または ReDriveKind enum 引数。private、呼出箇所はここ1か所） |
| 2 | Orchestrator.cpp:470-503（accounting block） | `isRetention=true` の場合は generation 記録のみで count++ しない。`>` 破棄判定は retention 以外の enqueue（将来の Type A）にのみ適用。破棄時の retire 経路（:490-492）は現行維持 |
| 3 | Orchestrator.h:283-289（kMax コメント・メンバ注記） | 契約の再定義（retention は count 対象外、bound=TTL/generation/shutdown）を 1箇所に統一 |
| 4 | RuntimePublicationState.h:16-20（enum コメント） | 「cap に到達」表現を実コード `>` 意味論に統一（または実装側を `>=` に寄せて文書に合わせる — **どちらか一方に統一**。推奨は現コード `>` を正とする） |
| 5 | doc/work88/D135-1_IMPLEMENTATION.md:25 | `>=`/`retryCount=2` 記述を実装契約に合わせ訂正（履歴注記として） |
| 6 | （任意・別提案扱い）PublicationAdmission.cpp:76-77 | TTL 超過を `Expired` に（dead enum の活性化）— 本パッチの範囲外として分離可能 |

**テストソース変更: 不要**（`testDeferredBacklogDrainsCompletely` の契約は現行のまま）。

## 11. 実装前に必要なテストケース一覧

1. **既存回帰**: `testDeferredBacklogDrainsCompletely`（2-cycle）Debug+Release PASS — 修正の直接受け皿。
2. **長期 retention**（新規・最重要）: fading を (kMax+1)×tick より十分長く（例: 50-100ms）保持後に解除 →
   deferred publish が**破棄されずに**成立すること。F1 の再現シナリオが PASS に転じることの直接証明。
   （DIAG: `[D135] re-defer (new)` 1回のみで retry が出ない、または出ても count が増えない、
   `[HEALTH] starved` が不発、のアサーション）
3. **TTL 終端**: fading を TTL 超過（30s）保持 → StaleDiscard 終端 + DSP reclaim 確認。
   （30s 実待ちを避けるなら TTL 圧縮テストフックは**別提案**（production 変更を伴うため本パッチでは禁止）。30s 実測でも構わない）
4. **kMax 意味論の単体確認**: PublicationAdmissionTests に accounting の数式（§3 の表）を固定する
   unit test（Synthetic 同一 generation 連続 enqueue で count と破棄タイミングを検証）— 契約ドリフト再発防止。
5. **recovery wake 分離の回帰**: D135-8 P3 chain（recovery → reset → ordinary → increment）が
   不変であること（既存 `PublicationAdmissionTests`/`RetrySchedulerTests` で担保、変更不要の確認）。
6. **全体回帰**: ctest Debug+Release 40/40（Gate C 再実施）。
7. **決定論性確認**: 修正後 harness を Release で ≥5 回直接実行し、レース非依存（全 PASS）を確認
   （F1 で判明した 1ms vs 1ms レースが契約上無害化されたことの検証）。

## 12. F2-8 合格条件チェックリスト

- [x] retry の意味が定義された（§2: Type A = publish 再試行のみ）
- [x] re-drive の意味が定義された（§2/§6: 条件未成立の保持、失敗でない）
- [x] obligation identity が定義された（§6: `req.generation`、D135-1 §2 と整合）
- [x] kMax の off-by-one が解消された（§3: 実コード `>`、文書 `>=` の +1 ドリフトを確定・統一方針提示）
- [x] exhaustion の terminal condition が定義された（§4/§6: retry bound の対象を Type A に限定、retention は TTL/generation/shutdown が bound）
- [x] RetryExhaustedDiscard の ownership が証明された（§4: 正規 retire pipeline、I4 整合）
- [x] 45s test observation window との関係を説明できた（§7: T_exhaustion 3-4ms ≪ 45s、構造的必然）
- [x] ordinary wake と recovery wake が区別された（§5: 4分類、provenance は正常・変更不要）
- [x] infinite re-drive 防止策が別途存在する（§6: 30s TTL + generation supersession + shutdown — 既存）
- [x] Release/Debug 差が設計上説明可能（§7: codegen によるレース境界移動、T_exhaustion ≪ 45s は両 config 不変）

## Artifacts（本監査で新規作成したもの）

- 本文書のみ。production/test source 変更 0、ビルド 0、テスト実行 0。
