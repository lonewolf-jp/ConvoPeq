# Phase I-T1-D101-2.5O-2 — Deadline Contract Closure

> **Verdict: OPEN — D_obligation = UNASSIGNED 維持 / D_deferred は完全特性化 / 両者の等価性は意味領域分離により反証 / 三段論法を形式的に禁止 / Expired dead enum 発見**
> **コード変更 0 / TTL 値変更 0 / evaluate・evaluateDeferred 変更 0 / Expired の I4 昇格 0**

- **一次ソース**: `ConvoPeq.md` Generated 2026-08-22 15:43:20 版 + `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}` / `src/audioengine/PublicationAdmission.{h,cpp}` / `src/audioengine/RuntimePublicationState.h` / `src/audioengine/AudioEngine.h:4600-4619` / `src/audioengine/TelemetryRecorder.h`
- **継承**: O-1 判定（birth = C1 Accepted 条件付き固定 / identity UNRESOLVED / DeferredPublishSlot = control-lane state）を前提とする

---

## Scope Freeze 遵守

TTL 値変更なし / `kDeferredPublishTTLUs` を O deadline と断定せず / `Expired = Timeout` と断定せず / field 追加なし / test 追加なし。成果物は semantic contract と proof obligations のみ。

---

## A. D_deferred の完全特性化（既存機構の正確な記述）

```text
deferred enqueue
      ↓
TTL (lazy check)
      ↓
stale/discard
```

| 構成要素 | 実在する値 | 根拠 |
| -------- | ---------- | ---- |
| 対象 subject | pre-admission deferred request（DeferredFadingActive = NOT Accepted） | O-1 分類済み |
| anchor t0 | `enqueueTimestampUs`（slot 構築時に設定） | Orchestrator.cpp:403 |
| 境界区間 | enqueue → 次 RebuildThread peek/evaluateDeferred | processDeferredAdmission flow |
| 値 D | `kDeferredPublishTTLUs = 30'000'000` us（30 秒） | h:125、Orchestrator 所有 |
| 強制点 enforcer | `evaluateDeferred()`（Admission）が `DeferredAdmissionSnapshot.ttlUs` を読み `ageUs > ttlUs` で判定 | PublicationAdmission.cpp:78-80 |
| snapshot 注入 | `buildDeferredAdmissionSnapshot()` が 5 値 POD {currentGeneration, lastSequence, shutdown, nowUs, ttlUs} を構築し ttlUs に定数を注入 | Orchestrator.cpp:463-474 |
| check timing | **lazy** — RebuildThread が peek した時点でのみ評価。timer 駆動の deadline ではない | Threading.cpp:265 / Single Thread Owner 契約 |
| expiry action | `Discard(StaleDiscard)` — **Expired ではない**（下記決定的発見） | PublicationAdmission.cpp:80 |
| 観測補助 | `maxDeferredAgeMs_`（overwrite 時に CAS 更新、getter cpp:557）— 観測メトリックのみで強制なし | h:254 / cpp:390-393 |

### 決定的発見 — Expired は dead enum

`rg "DiscardReason::Expired|Expired"` の全出現は 3 件のみ:

```text
RuntimePublicationState.h:15   Expired   // ★ work37: TTL 超過     ← 宣言のみ
RuntimePublicationState.h:9    コメント                            ← 言及のみ
PublicationAdmission.cpp:80    return {Discard, StaleDiscard};     ← 実際の戻り値は StaleDiscard
                               // ★ work37: Expired を別 enum 化可能 ← コメント
```

- **`DiscardReason::Expired` は一度も設定されない**。実際の TTL 超過は generation/sequence staleness と同じ `StaleDiscard` を返す
- deferred lane 内にすら独立した「timeout」disposition は実質存在しない
- 含意: 将来 Timeout→Expired 写像を作るには (a) I4 適合性の証明 に加えて (b) dead enum の活性化という二重の設計作業が必要。`Expired→Timeout` rename は deferred lane taxonomy の意味論を静かに変えるため禁止どおり未実施

---

## B. D_obligation の構成要素監査

C2 logical obligation deadline:

```text
O birth
      ↓
D
      ↓
C3 execution
```

| 構成要素 | 判定 | 根拠 |
| -------- | ---- | ---- |
| anchor t0 | **UNASSIGNED** | O birth = C1 Accepted は条件付き固定だが entity 未誕生（O-1）。Accepted 時点の authoritative timestamp は存在しない（telemetry recordProgress(Submitted, nowUs) は観測であり authority ではない） |
| 値 D | **UNASSIGNED** | main path に deadline 定数/field は存在しない（rg deadline/lease = 0 件は N で確定済み） |
| 強制点 enforcer | **UNASSIGNED** | main path の age を検査する経路ゼロ。trySubmitImpl は同期通過で保留状態を持たない |
| expiry disposition | **UNASSIGNED** | N-5 で Timeout ケース全候補不適格を確定済み |

---

## C. 等価性証明: D_obligation = D_deferred か？

### 意味領域分離表

| 次元 | D_deferred | D_obligation | 一致？ |
| ---- | ---------- | ------------ | ------ |
| subject | pre-admission request | admitted logical obligation | **不一致** |
| anchor | enqueue timestamp | C1 Accepted birth（未誕生） | **不一致** |
| 境界区間 | enqueue → 再評価 | birth → publish completion | **不一致** |
| 値 | 30s 定数 | UNASSIGNED | **不一致** |
| 強制 | lazy peek 時 stale-discard | UNASSIGNED | **不一致** |
| expiry 意味論 | stale-discard（control lane） | latency guarantee（τ ≤ D） | **不一致** |
| 保証強度 | best-effort 清掃 policy | guarantee 契約 | **不一致** |

### 判定

```text
D_obligation = D_deferred  →  UNPROVEN（むしろ反証済み）
```

- 全 7 次元が不一致。subject・anchor・expiry 意味論のいずれか一つでも異なれば契約として同一視できない
- 数値が仮に等しくても（例え両方が 30s でも）値の一致と契約の同一は別 — identity of value ≠ identity of contract
- **D_obligation = UNASSIGNED を維持する**

---

## D. 三段論法の形式的禁止

```text
P1: ∃ TTL mechanism (kDeferredPublishTTLUs)        [TRUE — 実在]
P2: TTL mechanism ≡ C2 obligation deadline          [UNPROVEN — 意味領域分離により FALSE]
∴  τ_reservation→publish ≤ 30s guaranteed           [INVALID — 推論不成立]
```

推論は P2 で破綻する。ある lane に存在する時間機構の存在は、別の lane における deadline 契約の存在を構成しない。この三段論法を採用することは N の最大の成果（B 候補の deadline ownership が deferred lane 限定である事実）を壊す行為であり、本フェーズ以降も恒久禁止とする。

---

## E. 追加の時間的機構 census（誤同定防止の全数調査）

| 機構 | 実在 | 意味論 | D_obligation と同一視可能か |
| ---- | ---- | ------ | -------------------------- |
| `maxDeferredAgeMs_` | 実在（h:254） | overwrite 時滞留時間の最大値観測 — 強制なし | NO（観測メトリック） |
| `kPublishReceiptWaitTimeoutMs = 250ms` | 実在（AudioEngine.h:4600） | producer が receipt を待つ caller 側 wait bound — publish を拘束しない waiter 側の bound | NO（wait bound ≠ obligation deadline） |
| TelemetryRecorder PublishStage timestamps | 実在（Submitted/Built/Validated/Published each nowUs） | stage 間 duration の観測能力 — τ 観測は feasible（Phase L の L-A）だが observation ≠ guarantee | NO（観測） |
| HealthMonitor retire-age 監視 | 実在 | retire 側 pipeline の年齢観測 | NO（別 pipeline） |
| CoordinatorLoop 1ms tick | 実在 | 周期駆動 — deadline checker の候補足場ではあるが現状 publication age を検査していない | NO（機構のみ） |

**結論: publication path 上に D_obligation と同一視できる時間機構はゼロ。**

---

## F. O-3 Exit 判定

```text
D_deferred  : 完全特性化完了（8 構成要素 + lazy 強制 + StaleDiscard 実態）
D_obligation: UNASSIGNED 維持（anchor/value/enforcer/expiry 全て未割当）
等価性       : 反証済み（7 次元不一致）
三段論法     : 形式的に禁止（P2 破綻を明示）
Expired     : dead enum と確定（一度も設定されない）
```

Verdict: **OPEN** — D_obligation の割当は O birth/identity の閉包（O-1 継承）と binding/disposition 契約（O-4/O-5）に依存するため、本フェーズ単独では閉じない。

## Current Fixed State & Next

```text
D101-2.5N     CONDITIONAL/CANDIDATE（N-C2-B 唯一候補）
D101-2.5O-1   OPEN（definition/birth 条件付き固定 / identity UNRESOLVED / slot = control-lane）
D101-2.5O-2   OPEN（D_deferred 特性化完了 / D_obligation UNASSIGNED / 等価性反証 /
              Expired dead enum 確定 / 三段論法恒久禁止）

A_max=1 / T_w=2 / max(E_w)=1 observed only / M/R/R_cap/B_max^true/T2 UNDETERMINED 維持
```

**次フェーズ: D101-2.5O-3 — Execution Binding Contract Closure**（O→E binding の既存意味論のみでの成立可否。新 permit/token 設計は引き続き禁止）

---

## Tool Coverage

| 系統 | ツール | 実行内容 | 結果 |
| ---- | ------ | -------- | ---- |
| WSL | rg 15.1.0 | `rg "DiscardReason::Expired\|Expired"` 全出現 3 件列挙 — **dead enum 確定** / `maxDeferredAgeMs_\|kPublishReceiptWaitTimeoutMs\|StaleDiscard` consumers census | 決定的証拠取得 |
| WSL | sed 4.9 | buildDeferredAdmissionSnapshot / finishView / consume / discard 本体（cpp:460-520）/ TelemetryRecorder.h 先頭 | 強制点と ownership release 口の照合 |
| Sandbox | ctx_batch_execute | 4 コマンド並列（expired usage / recorder structure / temporal consumers / deferred admission impl）+ queries 抽出 | 全データ 1 往復で収集 |
| 継承 | Phase N/L/M/O-1 の全ツール結果 | kDeferredPublishTTLUs 所有権（ag）、slot 構造（sed/cat）、identity 意味論（awk/semble） | 再利用・再確認 |
| 文献 | crossbeam-epoch / rigtorp 等 9 系統 | 前ステップまでに 200 OK 済の知識を再利用（O-2 では新規外部技術要件なし — 意味領域分離は社内形式論で完結） | 追加調査不要と判断 |
