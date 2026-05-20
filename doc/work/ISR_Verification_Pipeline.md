# ConvoPeq ISR Verification Pipeline

## 目的

本書は **R18: CI Verification Pipeline** の authoritative specification である。

ISR の形式保証を「設計規律 + runtime assert」から
**CI mandatory merge blocker** へ移行するための検証ステージを定義する。

---

## 基本方針

```text
warning ではなく CI fail = merge reject
```

すべての V1-V10 ステージが merge blocker である。
「警告として記録する」ことは禁止。検出 = merge reject。

追加原則（runtime proof integration）:

- validator は **runtime-generated evidence producer** として動作する
- CI は **artifact evaluator** として証跡を評価する
- 証跡未生成は検証失敗として扱う（warning不可）
- runtime correctness は artifact existence ではなく runtime invariants で成立する

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## 検証ステージ定義

### V1: Atomic Dot-Call Scan

**対象**: `.load()/.store()/.exchange()/.compare_exchange()/.fetch_add()/.fetch_sub()/.fetch_or()/.fetch_and()` の直接呼び出し

**既存スクリプト**: `.github/scripts/check-src-atomic-dotcall.ps1`

**ルール**: `convo::consumeAtomic` / `publishAtomic` / `exchangeAtomic` を使用すること。
直接 dot-call は CI fail。

**ステータス**: 実装済み（既存 CI で稼働中）

---

### V2: Seal Integrity Check

**対象**: publish 済みオブジェクトへの post-publish mutation

**ルール**:

- publish 後に mutating call が検出された場合、CI fail
- `ISR_Immutability_Enforcement_Spec.md` E3/E4 に対応

**実装要件**:

- Debug build: seal violation で assert + diagnostic log
- Release build: mutation fault counter increment（silent ignore 禁止）
- CI: post-publish mutation fuzz test を実行

---

### V3: Recursive Closure Validation

**対象**: RuntimePublication の publish 前に行う closure graph 検証

**ルール**: `validateClosureGraph(root)` が `ClosureValidationResult::Valid` を返さない場合、CI fail

**実装要件**:

- `ISR_Runtime_Closure_Descriptor.md` の `validateClosureGraph()` 呼び出しが必須
- CI Graph Dump（`closure_graph.json`）の自動生成と archive

---

### V4: Payload Tier Validation

**対象**: publish closure 内の Forbidden/RTLocalOnly tier オブジェクト

**ルール**:

- Forbidden tier が publish closure 内に存在する場合、CI fail
- RTLocalOnly が publish closure 内に存在する場合、CI fail

**実装要件**:

- `ISR_Payload_Tier_Model.md` の `validatePayloadCapabilities()` 呼び出しが必須
- tier 割り当て表に従い静的チェックツールを実装

---

### V5: HB Reorder Simulation

**対象**: `ISR_Minimal_HB_Failure_Model.md` の HB-01 〜 HB-04 シナリオ

**ルール**: failure catalog の任意シナリオが trigger 可能と判定された場合、CI fail

**実装要件**:

- HC1-HC4 に対応する acquire/release ペアの存在確認（静的解析）
- reorder simulation による failure catalog 検証

---

### V6: Shutdown FSM Verification

**対象**: shutdown phase 順序と barrier 実行の正当性

**ルール**: `verifyShutdownFSM(trace)` が `Valid` を返さない場合、CI fail

**実装要件**:

- `ISR_Shutdown_State_Machine.md` の `verifyShutdownFSM()` 呼び出しが必須
- `shutdown_trace.json` の archive が必須

---

### V7: Retire Latency Audit

**対象**: RetireIntent の enqueue から reclaim 完了までのレイテンシ

**ルール**:

- Audio Thread での reclaim/delete/free/mkl_free 呼び出しが検出された場合、CI fail
- RetireIntent が completionEpoch 到達前に reclaim された場合、CI fail

**実装要件**:

- `ISR_Deferred_Retire_Intent_Bridge.md` の RT retire 禁止ルール CI 強制
- retire latency の計測と threshold 超過検出

---

### V8: UAF Suspicion Detector

**対象**: retire 後のポインタ参照、stale snapshot 参照

**ルール**: UAF suspicion が検出された場合、CI fail

**実装要件**:

- Address Sanitizer（Debug build `/fsanitize=address`）の結果を CI artifact として保存
- retire 済み DSPHandle への参照を静的解析でフラグ

---

### V9: Forbidden Capability Scan

**対象**: `PayloadCapability::RTLocal` を持つオブジェクトの publish closure への混入

**ルール**: RTLocal capability が publish closure に混入した場合、CI fail

**実装要件**:

- `ISR_Payload_Tier_Model.md` の Forbidden tier reject ルールと統合
- `ISR_Runtime_Closure_Descriptor.md` の reject 対象リストに対応する静的スキャン

---

### V10: Ownership Cycle Detection

**対象**: ClosureGraph 内の cycle（DAG 不変条件の違反）

**ルール**: `validateClosureGraph()` が `CycleDetected` を返した場合、CI fail

**実装要件**:

- `ISR_Runtime_Closure_Descriptor.md` 不変条件 C3 の実装
- graph dump の DFS/BFS cycle 検出

---

## ステージ実装ステータス

| Stage | 名称                         | ステータス        |
| ----- | ---------------------------- | ----------------- |
| V1    | Atomic Dot-Call Scan         | 実装済み（稼働中）|
| V2    | Seal Integrity Check         | 未実装            |
| V3    | Recursive Closure Validation | 未実装            |
| V4    | Payload Tier Validation      | 未実装            |
| V5    | HB Reorder Simulation        | 未実装            |
| V6    | Shutdown FSM Verification    | 未実装            |
| V7    | Retire Latency Audit         | 部分実装          |
| V8    | UAF Suspicion Detector       | 部分実装（ASan）  |
| V9    | Forbidden Capability Scan    | 未実装            |
| V10   | Ownership Cycle Detection    | 未実装            |

---

## CI Artifact 要件

各ステージが生成する artifact を CI で archive すること。
命名・schema は `ISR_Proof_Artifact_Schema_Registry.md` の canonical を使用する。

- `closure_graph.json` (V3)
- `payload_tier_report.json` (V4)
- `hb_constraint_report.json` (V5)
- `shutdown_trace.json` (V6)
- `retire_latency_report.json` (V7)
- `asan_report.txt` (V8)
- `hb_graph_trace.json` (V5, runtime trace)
- `retire_timeline.json` (V7)
- `mutation_fault_trace.json` (V2)

### Producer / Evaluator 対応

| Producer（Runtime/Validator） | Artifact | Evaluator（CI） |
| --- | --- | --- |
| ClosureValidator | `closure_graph.json` | closure schema + invariants check |
| PayloadTierValidator | `payload_tier_report.json` | forbidden/rtlocal violation check |
| HBVerifier | `hb_constraint_report.json`, `hb_graph_trace.json` | HB constraints + failure catalog check |
| ShutdownVerifier | `shutdown_trace.json` | phase/barrier sequence check |
| RetireAudit | `retire_latency_report.json`, `retire_timeline.json` | latency bound + lane violation check |
| SealIntegrityCheck | `mutation_fault_trace.json` | post-publish mutation violation check |

`artifact missing` / `artifact parse error` / `schema mismatch` はすべて CI fail とする。

---

## 実装優先順

Phase 0 で優先実装すべき：

1. V3 (Recursive Closure Validation) — `ISR_Minimal_Phase0_Recommended.md` P0-1 と連動
2. V4 (Payload Tier Validation) — P0-2 と連動
3. V6 (Shutdown FSM Verification) — P0-4 と連動
4. V2 (Seal Integrity Check) — P3/R1/R13 と連動
5. V7 (Retire Latency Audit) — R14 と連動

---

## 関連正本

- `ISR_Runtime_Closure_Descriptor.md` — V3/V10 の実装基盤
- `ISR_Payload_Tier_Model.md` — V4/V9 の実装基盤
- `ISR_Minimal_HB_Failure_Model.md` — V5 の実装基盤
- `ISR_Shutdown_State_Machine.md` — V6 の実装基盤
- `ISR_Deferred_Retire_Intent_Bridge.md` — V7 の実装基盤
- `ISR_Immutability_Enforcement_Spec.md` — V2 の実装基盤
- `ISR_Formal_Guarantee_Package.md` P8 — 統合保証パッケージ参照
- `ISR_Proof_Artifact_Schema_Registry.md` — artifact 命名・schema contract

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R18 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（V1以外 未実装）
