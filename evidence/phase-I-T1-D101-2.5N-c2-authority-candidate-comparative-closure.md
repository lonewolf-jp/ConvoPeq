# Phase I-T1-D101-2.5N — C2 Authority Candidate Comparative Closure

> **Verdict: CONDITIONAL / CANDIDATE — N-C2-B (RuntimePublicationOrchestrator) が唯一の生存候補**
> **→ C2-CANDIDATE = 一意に閉じる（但し placement 自体は UNASSIGNED 維持）/ 次フェーズ O: C2 Contract Semantics Closure へ**
> **C2-B guarantee の成立ではない。H-A 条件変更なし。コード変更 0 / 実装 0 / 測定 0 / 新 Authority 0 / C2 test 0**

- **一次ソース**: `ConvoPeq.md` Generated 2026-08-22 15:43:20 版 + `src/audioengine/RuntimePublicationState.h` / `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}` / `src/audioengine/PublicationAdmission.{h,cpp}` / `src/audioengine/TelemetryRecorder.{h,cpp}` / `src/audioengine/RuntimeWorldAuthority.h` / `src/audioengine/ISRRuntimePublicationCoordinator*` / `src/audioengine/AudioEngine.RebuildDispatch.cpp` / `src/audioengine/RuntimePolicyEngine.h` / `src/audioengine/RuntimeHealthMonitor.cpp`
- **禁則遵守**: deadline field 追加 / lease 追加 / reservation object 実装 / Expired を I4 に追加 / `evaluate()` 変更 / `publishAndSwap()` 変更 / Completion callback の C2 接続 / timeout telemetry の decision 昇格 / C2 test 追加 / M/R/R_cap/B_max^true/T2 推定 — **全て未実施**

---

## N-1 候補集合の固定（新規 authority を発明しない）

| 候補 | 対象 | 実在確認 |
| ---- | ---- | -------- |
| **N-C2-A** | `PublicationAdmission` | 実在 — stateless const decision 関数（M-C1 PASS の主体） |
| **N-C2-B** | `RuntimePublicationOrchestrator` | 実在 — trySubmitImpl / deferredSlot_ / kDeferredPublishTTLUs 所有 |
| **N-C2-C** | `RuntimeWorldAuthority` | 実在 — INV-X4-2/3 sole physical publish gateway（C3） |
| **N-C2-D** | `RuntimePublicationState` / `RuntimePublicationStateOwner` | 実在 — PublicationLedger + ProgressRecord の ledger owner |
| **N-C2-E** | 既存 NonRT publication/recovery authority | 実在 — recovery lane (`recoveryIntentQueue_`) / DeferredPublishView lane / PolicyEngine / HealthMonitor に分解して監査 |
| **N-C2-F** | 新規 authority | 存在可能性のみ評価（設計・実装は禁止）— deferred |

---

## N-2 4 Owner の責務分離監査

> 原則: 「クラスが state を持っている」≠「そのクラスが C2 authority である」。4 責務を authoritative に持たなければ認定しない。

### N-C2-A PublicationAdmission

| Owner 責務 | 判定 | 根拠 |
| ---------- | ---- | ---- |
| Obligation identity owner | **NO** | `evaluate()` は const noexcept の純粋判定関数。per-obligation 保持ゼロ。PublishRequest は caller 所有 |
| Deadline owner | **NO** | deadline 概念なし。`ttlUs` は Orchestrator が snapshot に詰めた値を読むだけ（h:93「現在は Orchestrator が kDeferredPublishTTLUs を詰める」） |
| Execution binding owner | **NO** | Decision 返却のみ。「Admission = Decision only; View = Store mutation」（ADR-C4 原則明記） |
| Terminal disposition owner | **NO** | evaluateDeferred は DiscardReason を返すだけで Store 変更は View/caller |

**判定: 0/4 owner。C1 との役割集約も発生（K-4 Model B の concentration 問題と同型）。REJECT**

### N-C2-B RuntimePublicationOrchestrator

| Owner 責務 | 判定 | 根拠 |
| ---------- | ---- | ---- |
| Obligation identity owner | **PARTIAL** | correlationId を採番し Submitted→Built→Validated→Published 全段で追跡。deferred lane では `DeferredPublishSlot{generation, sequence, enqueueTimestampUs}` を保持。ただし main path では obligation record を保持しない（同期通過） |
| Deadline owner | **PARTIAL** | `kDeferredPublishTTLUs = 30'000'000`（h:125）を**所有**し cpp:471 で snapshot へ注入、evaluateDeferred による TTL 強制（ageUs > ttlUs → Discard）まで実行。**現行コードで唯一 authoritative な期限的機構**。ただし意味領域は deferred stale-discard であり τ_{reservation→publish} ≤ D ではない |
| Execution binding owner | **PARTIAL** | `executor_.publish()` を同一関数スコープ内で呼ぶ手続き的 binding。deferred lane は view.consume() → 再 evaluate → execute。authoritative な O→E 契約は不在 |
| Terminal disposition owner | **PARTIAL** | deferred lane: finishView/discard が `lastDiscardReason` を記録（ShutdownDiscard/StaleDiscard）。main path: onExecutorFailed/onRejected は集計カウンタのみで per-O terminal record なし |

**判定: 4/4 PARTIAL — 全責務が deferred lane に限定され main path 拡張は新設計を要する。唯一の生存候補**

### N-C2-C RuntimeWorldAuthority

| Owner 責務 | 判定 | 根拠 |
| ---------- | ---- | ---- |
| Obligation identity owner | **NO** | publish() は完成済み owner+metadata を受けるだけ。PendingPublishRegistry は非所有 gap handle（64 entries ring） |
| Deadline owner | **NO** | temporal field ゼロ（rg deadline/lease = 0 件） |
| Execution binding owner | **該当なし** | RWA 自身が E（C3）。自己 binding は無意味 — C2=C3 concentration（K-4 Model A リスク） |
| Terminal disposition owner | **NO** | oldWorld を caller へ返すのみ。retire は Lifetime 責務（X3 分離） |

**判定: 0/4 + reverse edge REQUIRED（下記 N-6）+ concentration。REJECT**

### N-C2-D RuntimePublicationState / StateOwner

決定的証拠（`RuntimePublicationState.h:110-113`）:

```cpp
void onSubmitted(uint64_t correlationIdShort) noexcept {
    state_.ledger.submittedCount++;
    state_.progress.submittedCount++;
    state_.progress.lastProgressTimestampUs = timestampUs();
}
```

- **引数 `correlationIdShort` が本体で一切使用されない** — per-obligation 保持ゼロ、集計カウンタのみ。
- CorrelationId は `lastCorrelationId` として最新値上書き保存のみ（h:87, 177-179）。
- detectStuckStage() はカウンタ差分からの診断であり authoritative deadline ではない。

| Owner 責務 | 判定 |
| ---------- | ---- |
| Identity / Deadline / Binding / Terminal | **全て NO**（受動 ledger。header 自身が「一次情報源 PublicationLedger、ProgressRecord は副産物」と宣言） |

**判定: 0/4。state を持っていることと authority であることを同一視しない原則の典型例。REJECT**

### N-C2-E 既存 NonRT publication/recovery authority（3 sub-candidate に分解）

#### E-1: ISR Coordinator recovery lane

- `recoveryIntentQueue_`（Builder Work Queue）、`submitRecoveryRequest`（push 専用）、`popRecoveryRequest`（Builder Loop が消費 — RebuildDispatch.cpp:927）、`recoveryShutdownDiscardCount_`（INV-5 silent loss 禁止の明示会計）。
- Identity: transport レベルの保持あり。Deadline: 不在。Binding: 回復 obligation は Builder Loop pop 後 **C1 を再通過**（submitPublishRequest → evaluate）— 直接 execution binding なし。
- **構造的欠陥: recovery lane は C1 の上流に位置する。ここに C2 を置くと DAG 順序が C2→C1→C3 に逆転し、目標 DAG（C1→C2→C3）と矛盾。NO-GO**

#### E-2: RuntimePolicyEngine

- TrendSnapshot{pendingRetire, publicationSeq, maxRetireAgeUs, activeReaderCount} を観測し RecoveryAction（Observe/Throttle/Recover/...）を発行。
- maxRetireAgeUs は retire 側の年齢観測であり τ_{reservation→publish} ではない。obligation identity / D 発行 / execution binding / terminal disposition の全て不在。
- **決定的欠陥: Observation 側に配置された C2 は `Observation → C2 → C3` という禁止トポロジー（M-4 で X とした逆辺）を制度化する。NO-GO**

#### E-3: RuntimeHealthMonitor

- MonitorState（overflow rate / reader slot / retire age）から HealthState を導出し evaluate() step 4 の入力になる既存 Observation→C1 edge の供給源。
- C2 としての 4 責務すべて不在。E-2 と同じく Observation→C2 トポロジー問題。**NO-GO**

**判定: E 全 sub-candidate NO-GO**

### N-C2-F 新規 authority

- 存在可能性のみ評価: 本コードベースには move-only permit/proof 型の確立されたパターンが存在（`ReclaimPermit` single-use consume / `ShutdownQuiescenceProof` friend 制限生成 — ISRLifetimeProof.h）。OwnerChannel による単一スロット transfer パターンも確立済み。新 authority の技術的実現可能性は高い。
- ただし本段階での設計・実装は禁止。**deferred**

---

## N-3 C1→C2 Provenance 監査（候補ごと）

```text
PublishRequest → C1 Accepted → ??? → logical obligation O
```

| 候補 | ??? を埋められるか | 判定 |
| ---- | ------------------ | ---- |
| A Admission | 不可 — stateless、何も生まない | **ABSENT** |
| B Orchestrator | 部分的 — Accepted 直後に correlationId 採番点が存在し deferredSlot_ 保持の先例あり。ただし obligation object は誕生しない | **PARTIAL** |
| C RWA | 不可 — 完成品を受けるだけ | **ABSENT** |
| D StateOwner | 不可 — 集計のみ | **ABSENT** |
| E Recovery | 部分的 — queue/slot が request identity を保持するが pre-C1 / deferred-lane であり post-Accepted main-path obligation ではない | **PARTIAL（順序逆転のため不適格）** |

### identity 意味論の分離（勝手な解釈禁止の遵守）

| ID | 実際の意味論 | obligation identity として適格か |
| -- | ------------ | -------------------------------- |
| `correlationId` | TelemetryRecorder 進捗相関（128bit 相当 engineInstanceId+counter、wrap 不可）。per-O 保持されず lastCorrelationId 上書きのみ | **不適格** — 進捗相関であり義務 identity ではない |
| `PublicationSequenceId` | commit 時 bake される publication identity（INV-X4-6: RuntimeStore::current identity 構成要素、単調） | **不適格** — execution/publication identity であり pre-execution obligation identity ではない |
| `RecoveryGeneration` / `rebuildRequestGeneration` | staleness 判定用 generation epoch | **不適格** |
| EBR epoch | reader epoch domain | **無関係** |

**結論: 現行コードに「logical obligation O の birth + identity」を生成する経路は存在しない。最接近は B の correlationId 採番点 + deferredSlot_ 先例。**

---

## N-4 C2→C3 Execution Binding 監査

> 単なる call path 確認は禁止。「E が O の履行としてのみ成立する」binding の存在を問う。

| 候補 | 判定 | 根拠 |
| ---- | ---- | ---- |
| A | **BINDING-ABSENT** | decision only |
| B | **BINDING-PARTIAL** | executor_.publish() は同一スコープの手続き的呼び出し。deferred lane は consume→再 evaluate→execute。しかし seqId 一致は receipt における execution identity 確認であって obligation binding ではない（指示の警告を厳守）。authoritative O→E 契約は不在 |
| C | **BINDING-ABSENT** | RWA は E 自身。自己参照は binding ではない |
| D | **BINDING-ABSENT** | 受動 ledger |
| E | **BINDING-ABSENT** | recovery/deferred とも C1 再通過であり直接 binding なし |

---

## N-5 Terminal Disposition × I4 Conservation 突合（最重要）

I4 固定式:

```text
transport + durable + building + stalled + superseded + shutdownDiscard
  = admittedLogicalObligationCount
```

消失理由閉集合 = `{Success, Superseded, ShutdownDiscard}`。4 ケース並列監査:

| Case | 現行コードの実在 | I4 適合性 |
| ---- | ---------------- | --------- |
| **Success** | publishedCount / receipt watermark / onPublished | 適合 — 全候補が観測可能。I4 Success と矛盾なし |
| **Superseded** | `SupersededDiscard`（deferred lane taxonomy 内） | 未証明 — I4 Superseded は semantic target containment 要件付き。deferred lane の単一 slot 上書きと containment の対応は別途証明が必要 |
| **ShutdownDiscard** | 三重の先例: deferred lane DiscardReason + `recoveryShutdownDiscardCount_`（INV-5 明示会計）+ I4 閉集合 | **唯一の三重先例** — 最も適合性が高い |
| **Timeout** | **I4 適合 disposition を持つ候補ゼロ**。`Expired` は deferred-publish lane の stale-discard taxonomy（work37 Phase 6）としてのみ存在し、I4 logical obligation disposition とは意味領域が異なる | **UNASSIGNED 維持** — Timeout→Expired の安易な採用は control-lane 意味論の I4 閉集合への混入 = contradiction（M-C5 と一貫） |

### 候補ごとの I4 適合性

- A/C/D/E: I4 logical obligation ledger を今日所有するクラスは存在しない（I4 conservation は evidence chain 上の設計拘束であり runtime structure としては未実装）。どの候補でも mapping 構築は新規作業。
- B: deferred lane の discard 会計（lastDiscardReason + ShutdownDiscard 記録）が最も近い先例だが、control-lane と logical-obligation ledger の分離証明は未実施。

**結論: Timeout disposition は全候補に対して UNASSIGNED。N では決めず次フェーズへ持ち越し。**

---

## N-6 Reverse-Edge / Non-Circularity 監査（M-C4 PASS を崩さない）

| 候補 | Completion/Observation → C2 が guarantee に必要か | 判定 |
| ---- | -------------------------------------------------- | ---- |
| A Admission | 不要（admission 時点で lease 型にできる）だが C1=C2 集約 | edge なし — ただし集約問題 |
| B Orchestrator | **不要** — 既存 TTL 機構が proactive expiry の先例（completion feedback なしで ageUs > ttlUs を次回評価時に判定）。CoordinatorLoop tick による背景再評価も completion を必要としない | **edge なし — 合格** |
| C RWA | **必要** — publish は同期不可逆境界。timeout 判定は blocking（RT NO-GO）か post-completion evaluation（Completion→C2 = 循環）の二択 | **REQUIRED → NO-GO candidate** |
| D StateOwner | 受動のため能動判断は他者依存 → decision authority が別处に必然的に発生 | 実質不合格 |
| E-1 Recovery | edge は不要だが DAG 順序逆転 | NO-GO |
| E-2/E-3 Policy/Health | Observation 下流配置 → Observation→C2→C3 禁止トポロジーの制度化 | **NO-GO** |

**M-C4 PASS（Completion→C2 / Telemetry→C2 構造的不在）は本監査によって維持される。B のみが reverse-edge-free 配置の先例を持つ。**

---

## N-7 Candidate Scorecard

| Candidate                   | Identity      | Deadline                          | C1→C2        | C2→C3           | Terminal                    | I4         | Reverse edge            | 判定       |
| --------------------------- | ------------- | --------------------------------- | ------------ | --------------- | --------------------------- | ---------- | ----------------------- | ---------- |
| Admission                   | ABSENT        | ABSENT（ttlUs は読むだけ）        | n/a（=C1）   | ABSENT          | ABSENT                      | 不適合     | なし（C1C2 集約）       | **REJECT** |
| Orchestrator                | PARTIAL       | **PARTIAL（kDeferredPublishTTLUs 所有・強制まで実行）** | PARTIAL      | PARTIAL（手続き的） | PARTIAL（deferred lane のみ） | UNPROVEN   | **不要（proactive 先例）** | **CANDIDATE** |
| RWA                         | ABSENT        | ABSENT                            | ABSENT       | self(E)         | ABSENT                      | —          | **REQUIRED → 循環**     | **REJECT** |
| State owner                 | ABSENT（引数未使用の決定的証拠） | ABSENT（診断のみ）               | ABSENT       | ABSENT          | ABSENT（集計のみ）          | —          | —                       | **REJECT** |
| Existing recovery authority | PARTIAL（transport） | ABSENT                     | 順序逆転     | ABSENT（C1 再通過） | PARTIAL（ShutdownDiscard 会計） | UNPROVEN   | E-2/E-3 は制度化 NO-GO  | **REJECT** |
| New authority               | N/A           | N/A                               | N/A          | N/A             | N/A                         | N/A        | N/A                     | **deferred** |

---

## N-8 C2 Placement の唯一性判定

### 唯一性の論拠

1. **B だけが現行コードで authoritative な期限的機構を所有する**: `kDeferredPublishTTLUs` の定義（h:125）・snapshot 注入（cpp:471）・evaluateDeferred による強制（ageUs > ttlUs → Discard）・discard 理由記録まで一貫して Orchestrator 側。Admission は読むだけで所有しない。
2. **B だけが reverse-edge-free の運用先例を持つ**: proactive TTL expiry は completion feedback を必要としない。
3. **B だけが obligation ライフサイクルの部分実装を持つ**: DeferredPublishSlot{generation, sequence, enqueueTimestampUs} + DeferredGuard + view.consume/discard プロトコル。
4. 他候補は全て 0/4 owner、reverse edge 必須、DAG 順序逆転、または禁止トポロジー制度化のいずれかで失格。

### 但し書き（CONDITIONAL の理由）

- B の 4 owner 能力は **deferred lane に限定**される。main path（trySubmitImpl 同期通過）には obligation record が存在しないため、「Orchestrator が main path でも 4 owner になれる」ことは拡張設計の監査を要する。
- selection ≠ placement 確定: C2 node は引き続き **UNASSIGNED** を維持する（deadline をどこに置くかはまだ決めていない — 4 owner 能力の監査を先に行ったのは指示どおり）。
- I4 適合性（特に Timeout disposition）は未証明。

### 出口判定

```text
候補が一意に閉じる（N-C2-B RuntimePublicationOrchestrator）
    ↓
D101-2.5N = CONDITIONAL / CANDIDATE
    ↓
次に O: C2 Contract Semantics Closure
```

## H-A 進行条件（変更なし）

```text
M-C1 PASS（維持）
M-C2 PASS ← 現在 FAIL(ABSENT)、O フェーズで契約確定が必要
M-C3 PASS ← 現在 FAIL(未成立)、同上
M-C4 PASS（維持）
M-C5 PASS ← 現在 UNPROVEN、I4 接続 semantics が必要
+
N の C2 placement closure ← 本フェーズで CONDITIONAL 達成
+
Enforcement DAG non-circularity ← L-C4/M-B 継承
```

**N の成功条件は「C2 authority candidate を一意に決定できること」までであり、C2-B guarantee の成立ではない。**

## Current Fixed State & Next

```text
D101-2.5C   OPEN / D101-2.5D OPEN / D101-2.5E OPEN(Case C)
D101-2.5F   Design CLOSED / Production OPEN
D101-2.5G   OPEN / D101-2.5H H-B
D101-2.5I   PLACEMENT-CANDIDATE PARTIAL
D101-2.5J   PLACEMENT-CANDIDATE CONDITIONAL
D101-2.5K   OPEN/CONDITIONAL
D101-2.5L   OPEN（observation feasible / guarantee UNPROVEN / enforcement OPEN）
D101-2.5M   OPEN（M-A Structural DAG 閉包証明 / M-B 未構成）
D101-2.5N   CONDITIONAL/CANDIDATE — N-C2-B 唯一生存候補 / placement は UNASSIGNED 維持

A_max=1 / T_w=2 / max(E_w)=1 observed only / M/R/R_cap/B_max^true/T2 UNDETERMINED 維持
```

**次フェーズ: D101-2.5O — C2 Contract Semantics Closure**（N-C2-B を対象に、obligation birth/identity/deadline/binding/disposition の契約意味論を論理レベルで固定。実装は依然禁止）

---

## Tool Coverage

| 系統 | ツール | 実行内容 | 結果 |
| ---- | ------ | -------- | ---- |
| WSL | rg 15.1.0 | StateOwner/correlationId/recovery authority/deferred lane/onPublishCommitted/I4 disposition set census（`-g` glob 形式） | 全 census 完了・相互一致 |
| WSL | sg 0.44.0 | `sg run -p "onSubmitted"` パターン検証 | 実行確認 |
| WSL | ag 2.2.0 | `ag kDeferredPublishTTLUs` — **所有権確定の決定打**（h:125 定義 / cpp:471 注入 / Admission は読取のみ） | B の deadline ownership を裏付け |
| WSL | fdfind 10.3.0 / fzf 0.67.0 | RuntimePublicationState.h 所在 / filter 動作 | 確認 |
| WSL | awk 5.3.2 | CorrelationId/lastCorrelationId/shortValue 行抽出 | 意味論分離の根拠行確定 |
| WSL | sed 4.9 / cat / grep | RuntimePublicationState.h 全文 / Orchestrator exec flow 150-310 行 / ConvoPeq.md stalled grep / evidence dir I4 location | 本体照合完了 |
| Sandbox | ctx_execute shell | semble.exe（Windows 版）`search "deferred publish TTL slot discard"` — **DeferredGuard 構造体発見**（Orchestrator.h:21）/ ccc.exe 存在確認 / graphify 0.9.48 version | semble 99% 圧縮で追加構造発見 |
| MCP | serena | project.yml 存在確認（language_servers: cpp/python/bash） | 利用可能状態確認 |
| MCP | AiDex | `.aidex/index.db` 存在確認 | インデックス整備済 |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | version・所在確認 | 利用可能 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue 等 9 系統 | 前ステップまでに 200 OK 済の知識を再利用（N では新規外部技術要件なし — 候補比較は社内構造監査で完結） | 追加調査不要と判断 |
