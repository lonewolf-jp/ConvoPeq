# D149 — T3b Adjudication/Resolve/Reuse Race Re-proof（Work Report）

**Status: T3b = NO-GO（結論分岐 B）。naive T3 の NO-GO は不変。変更 0・実装 0。**
**詳細:** `evidence/D149_T3B_ADJUDICATION_RACE_REPROOF.md` / **基準:** ConvoPeq.md `15:32:58`（mtime 監査で現行ソースより新しいことを実証）。D147/D148 の証明を前提とせず現行コードから再導出。

## 中核の発見: tagged CAS は consumer 側 apply を保護していない
反証構成（実在命令のみ・§5）:
```text
t2 CL: CAS word (O,1)→(O,0) 成功・w={O,1} 保持
t3 CL: state==Live 通過
t4 RebuildThread: Route A resolve(O, Published)（Orchestrator.cpp:320）→ counter.store(0)（h:429）
t5 CL: apply → delivery=None + counter.fetch_add(1) → **terminal O に counter=1**
```
→ **STOP #5「resolve の counter reset が old adjudication に上書きされる」が提案配置のまま到達可能**。D148 §3 の Case B/C は「exchange 後の producer signal」しか解析しておらず、「exchange→check→resolve→apply」の **apply 側 stale** が未解析だった。resolve は id を変えないため、id 再検証論法では terminal 化を排除できない（terminal 化は state のみ変化・id 不変）。

## Case C（旧 O→N 混入）は「到達不能」だが理由は CAS ではない
tryInsert は本番 call site cpp:947（submitRecoveryRequest・**CL**）のみ、adjudicate も CL 提案 → 同一スレッド program order で apply 前に reuse は割り込めない。**保護の本体は thread affinity**であり、tagged word は extraction 瞬間の id にしか寄与しない。指示の第 2 構成（resolve→tryInsert→apply）は提案配置では不成立だが、成立条件（adjudication の CL 外配置 / drain・apply の phase 分離）は契約に未記載 — Phase 2 で破られ得る。**STOP #6: identity は affinity 依存であり CAS では証明不能**。
なお**現行コードでは Case C が実際に到達可能**（markTransientFailure=RebuildThread vs tryInsert=CL、同一 oblId の重複 transport 表現は cpp:962-969 が明示的に許容）。T3b の CL 移設はこれを閉じるが、#5/#6 を満たさない。

## 追加検証の回答
- **delivery**: `state check→resolve→delivery=None` の terminal への write は**成立**（inert — 全 reader は Live ゲート付き）。ただし T3b の delivery CL 一元化は現行 W5 の**実 data race（cpp:1071 RebuildThread vs cpp:979/1001/1159/1167 CL・h:318-324 の主張は現行で偽）**を閉じる改善点。
- **counter**: resolve `store(0)`（任意スレッド）と apply `fetch_add`（CL）は別 atomic domain — 上書き順序成立（上記反証）。
- **markTransientFailure 分断論（中心論点）**: 現行の「単一処理」は契約文言（h:475 "Atomically performs"）に反し実装は check-then-act（cpp:1066→1071/1074）で**元々 atomic domain でない**。T3b は既存の分断を再結合せず、signal 移送のみ tagged 化し apply 窓を残した。

## トポロジー訂正（D148 へ）
`onPublishCommitted` の実呼び出しは ISR（audio callback）ではなく **CoordinatorLoop**（processIntent→PublishIntentHandler→executePublish・RuntimePublishExecutor.h:114、AudioEngine.h:3727）。h:468「Callable from ISR」は契約上の許可。naive T3 反証は CL+RebuildThread 並行で依然成立（判定不変）。「ISR 非接触」の lock-pool 論拠はより強固に。

## race matrix 判定
A ✓ / B′ ✗（#5）/ C 到達不能（affinity 依存・明記必須）/ D ✓（telemetry 分類ギャップ軽微）/ E ✓ / F 到達不能（reuse ゲート h:397）/ G liveCount ✓・counter invariant ✗。

## 結論と次アクション
**T3b NO-GO → Phase 2 実装凍結継続。D150: T3c 設計再証明（read-only）へ。**
T3c 要求: `(obligationId, state, pending, adjudicated)` を単一 16B lifecycle word CAS に統合 — postSignal/adjudicate/resolve/tryInsert の全 lifecycle 遷移を同一 CAS ドメインへ（reset と加算の競合を構造排除、drain+apply を単一遷移化）。不変条件 I-1〜I-4（−1 authority / adjudication=CL・drain+apply 非分離 / tryInsert=CL call site 1 / delivery 書込=CL）を契約明記 + grep 強制。詳細は evidence §8。

**STOP — 実装 0。D150（T3c 設計再証明）の指示を待つ。P5/P6 非着手。**
