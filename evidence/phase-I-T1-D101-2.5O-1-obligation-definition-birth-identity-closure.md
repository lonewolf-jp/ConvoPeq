# Phase I-T1-D101-2.5O-1 — Obligation Definition & Birth/Identity Closure

> **Verdict: OPEN — O definition/birth は作業定義として条件付き固定、identity = UNRESOLVED（既存値全て不適格）、DeferredPublishSlot = control-lane state と分類完了**
> **deadline / disposition / binding には触れない（指示どおり）。次フェーズ O-2〜O-8 へ持ち越し。**
> コード変更 0 / field 追加 0 / class 追加 0 / test 追加 0

- **一次ソース**: `ConvoPeq.md` Generated 2026-08-22 15:43:20 版 + `src/audioengine/RuntimePublicationOrchestrator.{h,cpp}` / `src/audioengine/PublicationAdmission.h` / `src/audioengine/AudioEngine.h:3447-3463` (RuntimePublicationIdentity) / `src/audioengine/RuntimeBuilder.cpp:63,165` / `src/audioengine/RuntimePublicationState.h` / `src/audioengine/TelemetryRecorder.h`
- **一次ソース更新**: 本フェーズ中に semble が `reserveRuntimePublicationIdentity()` を新規発見（AudioEngine.h:3454）— identity 監査に追加した

---

## O-0 Scope Freeze（固定禁止事項の明文化）

| 禁止項目 | 遵守 |
| -------- | ---- |
| コード変更 | 0 |
| field 追加 | 0 |
| class 追加 | 0 |
| `evaluate()` / `evaluateDeferred()` 変更 | 0 |
| `publishAndSwap()` 変更 | 0 |
| `RuntimePublicationOrchestrator` 実装変更 | 0 |
| C2 test 追加 | 0 |
| TTL 値変更 | 0 |
| `Expired` の I4 disposition 昇格 | 未実施 |
| `correlationId` の obligation identity 化 | 未実施 |
| `PublicationSequenceId` の obligation identity 化 | 未実施 |
| `RecoveryGeneration` の流用 | 未実施 |
| timeout telemetry の decision 化 | 未実施 |

成果物は semantic contract と proof obligations のみ。

---

## O-1 Obligation Definition（C2 が管理する O は何か）

### 作業定義（working definition — 最終閉包は O-8）

```text
O := C1 Accepted 後、
     C3 Execution によって履行されるべき
     logical publication obligation
```

論理モデル:

```text
PublishRequest
      │
      ▼
C1 Admission
      │
      ▼
[ O birth ]
      │
      ▼
C2
      │
      ▼
C3 Execution
```

### O-1 Exit 文の検証

```text
C1 Accepted の各 logical obligation に対して、
C2 が管理すべき O が exactly one 生じる。
```

判定: **OPEN（YES も NO も証明不能）**

構造的観察（現行コードの事実）:

- main path: `trySubmitImpl` は Accepted 後、正確に一度の build → validate → `executor_.publish()` を同期実行する（`RuntimePublicationOrchestrator.cpp:40-310`）。**Accepted と execution attempt の手続き的 1:1 は今日存在する**
- しかし永続的な O entity はどこにも誕生しない（correlationId は per-O 保持されず、StateOwner は集計のみ、build 完了後は request 情報が破棄される）
- executor publish 失敗時は `RejectedPublishFailure` / shutdown race 時は `RejectedShutdown`（15-P-6）— post-Accepted に非 Success 終点が存在することは将来 disposition 設計の制約となるが本フェーズでは扱わない
- deferred lane の request は **DeferredFadingActive = NOT Accepted** であり O 定義の対象外

結論: 「exactly one」を YES と証明するには O entity の存在が前提だが未誕生。NO と証明するには 1:1 定義の不可能性が必要だが構造的 1:1 が成立している。**定義可能性は確認、閉包は未達。**

---

## O-2 Birth Event / Identity Contract

### Birth event 候補の評価

| 候補 | 判定 | 根拠 |
| ---- | ---- | ---- |
| PublishRequest creation | **NO** | pre-admission。Rejected/Deferred になり得るため obligation birth としては早すぎる |
| **C1 Accepted** | **CONDITIONALLY FIXED** | 唯一の適格点。post-admission・pre-execution。作業定義の O と整合。最終確定は O-8 |
| correlationId allocation | **NO** | `nextCorrelationId()` は telemetry 相関のための採番（TelemetryRecorder.h:264）。obligation ledger ではない |
| deferred enqueue | **NO** | DeferredFadingActive（= NOT Accepted）の request を control-lane に保持する操作。admitted obligation の birth ではない |
| executor invocation | **NO** | execution start（E の開始）であり birth では既にない |

### Identity 候補の全判定

> 原則: ID が存在することと obligation identity であることを分離する。

| 候補 | 判定 | 根拠 |
| ---- | ---- | ---- |
| `correlationId` | **NO** | telemetry 進捗相関（128bit 相当 engineInstanceId+counter、wrap 不可）。per-O 保持ゼロ — `onSubmitted(correlationIdShort)` は本体で引数未使用（RuntimePublicationState.h:110-113）、`lastCorrelationId` は最新値上書きのみ |
| `PublicationSequenceId` | **NO** | commit 時 bake される publication identity（INV-X4-6: RuntimeStore::current identity 構成要素）。execution/publication identity であり pre-execution obligation identity ではない。deferred slot 内でも `guard.sequence = getLastCommittedPublicationSequence()` は watermark snapshot であり per-request 値ですらない |
| `Generation` / `RecoveryGeneration` | **NO** | staleness 判定用 generation epoch。複数 request が同一 generation を共有するため一意性なし |
| EBR epoch | **NO** | reader epoch domain。publication pipeline とは意味領域が異なる |
| `PublishRequest` identity | **ABSENT** | identity field 自体が存在しない。`struct PublishRequest { DSPHandle newDSP; int generation; RuntimeBuildSnapshot sealedSnapshot; BuildAnalysis; OversamplingResult; BuildDiagnostics; }`（rg requestId/obligationId = 0 件）。DSPHandle+generation の組合せも同世代同 DSP で衝突する |
| `RuntimePublicationIdentity` 【新規発見】 | **NO** | semble が発見（AudioEngine.h:3447-3463）。`{generation, worldId, publicationSequence}` の複合 identity を `reserveRuntimePublicationIdentity()` が採番（worldId 専用 generator + atomic sequence counter）。呼び出し元は RuntimeBuilder.cpp:63,165 — **C1 Accepted 後の build 開始時**。publication artifact をスタンプする identity であり意味論的には PublicationSequenceId 同族の execution/publication identity。決定的な差異: 採番点が build start であり obligation birth 点（C1 Accepted）とずれている。**この gap こそ obligation record が生まれるべき場所である** |

### Identity 判定の統合

```text
C2 identity = UNRESOLVED
```

- 既存の全 ID 候補を不適格として排除した（勝手な昇格は実施していない）
- 最接近は `RuntimePublicationIdentity` だが採番タイミング（build start）と意味論（artifact stamp）の両面で obligation identity ではない
- 無理に correlationId 等を昇格させない判断は N の最大の成果（identity 意味論の分離）を維持する

### DeferredPublishSlot の意味領域分類（最優先課題）

構造（RuntimePublicationOrchestrator.h:26-33）:

```cpp
struct DeferredPublishSlot {
    PublicationAdmission::PublishRequest request;
    DeferredGuard guard;                    // {int generation; PublicationSequenceId sequence;}
    PublicationAdmission::DeferredPublishMetadata metadata{};  // {generation, sequence, enqueueTimestampUs}
    DiscardReason lastDiscardReason{DiscardReason::None};
    uint64_t enqueueTimestampUs{0};
};
```

判定: **control-lane state — C2 obligation record ではない**

根拠（6 点）:

1. **pre-admission request の保持**: slot に入るのは DeferredFadingActive（NOT Accepted）の request のみ。admitted obligation の記録ではない
2. **main-path obligation は何処にも記録されない**: Accepted 後の同期通過 path には永続 record が存在しない
3. **silent overwrite semantics**: 新たな deferral が既存 slot を上書きし、`deferredOverwriteCount_`（h:253）を増やすだけ。上書きされた旧 request の disposition は一切記録されない — obligation ledger なら許容されない挙動
4. **discard taxonomy が stale-discard control 意味論**: TTL/generation/sequence/shutdown による stale-discard であり logical obligation disposition ではない
5. **consume path が C1 を再通過**: peek → evaluate → consume/discard → finishView → submitPublishRequest（RebuildThread Single Thread Owner 契約、Threading.cpp:265）。slot state ≠ admitted obligation state（再 admission される）
6. **guard.sequence は watermark snapshot**: enqueue 時点の last committed sequence であり当該 request の identity ではない

「もう C2 が実装されている」という錯覚を本分類により排除した。

### C1→O provenance

```text
provenance = ABSENT（今日）
```

- Accepted 後に誕生する永続 entity は存在しない（correlationId は保持されず、build 完了で request 情報消滅）
- 但し構造的 1:1（Accepted → execution attempt）が存在するため、birth point を C1 Accepted に固定すれば provenance 契約は定義可能
- M-C2 FAIL(ABSENT) の再確認であり、O フェーズ後半で契約として閉じる必要がある

### O→C3 binding premise

後段（指示の表どおり）。本フェーズでは扱わない。

---

## 判定表（成果物）

| 項目                    | 問い                           | 判定    |
| --------------------- | ---------------------------- | ----- |
| O definition          | C2 が管理する obligation は何か      | OPEN（作業定義固定: post-Accepted pre-execution logical publication obligation）   |
| Birth event           | どの event で O が生まれるか          | CONDITIONALLY FIXED（C1 Accepted — 最終閉包は O-8）   |
| Identity              | O を一意に識別する既存値は何か             | UNRESOLVED（既存値全て不適格）   |
| correlationId         | O identity か                 | NO（telemetry 相関・per-O 保持なし・引数未使用の決定的証拠）   |
| PublicationSequenceId | O identity か                 | NO（commit 時 bake の publication identity / INV-X4-6・deferred slot 内では watermark）   |
| Generation            | O identity か                 | NO（staleness epoch・一意性なし）   |
| RuntimePublicationIdentity | O identity か            | NO（build start 採番の artifact stamp — birth 点とずれ・新規発見）   |
| DeferredPublishSlot   | Oそのものか / control-lane stateか | control-lane state（6 根拠で分類完了 — O ではない）   |
| C1→O                  | provenance が存在するか            | ABSENT（構造的 1:1 あり・契約定義は後段）   |
| O→C3                  | binding の前提を定義できるか           | 後段   |

## O-1 Exit 文への回答

```text
「C1 Accepted の各 logical obligation に対して、
  C2 が管理すべき O が exactly one 生じる」

→ OPEN: YES 証明不能（O entity 未誕生）/ NO 証明不能（構造的 1:1 成立）
  作業定義下では definable = YES candidate、閉包は O-8
```

## Current Fixed State & Next

```text
D101-2.5N   CONDITIONAL/CANDIDATE（N-C2-B 唯一候補）
D101-2.5O-1 OPEN — O definition/birth 条件付き固定 / identity UNRESOLVED /
            DeferredPublishSlot = control-lane 分類完了 /
            RuntimePublicationIdentity 新規発見・不適格判定
            deadline / disposition / binding は未着手（指示どおり）

A_max=1 / T_w=2 / max(E_w)=1 observed only / M/R/R_cap/B_max^true/T2 UNDETERMINED 維持
```

**次フェーズ: D101-2.5O-2 以降** — Deadline Contract（D_deferred と D_obligation の分離・三段論法禁止の遵守）→ Binding Contract → Terminal Disposition × I4 → Conservation Mapping → Reverse-Edge Proof → O-8 Contract Closure Verdict（CLOSED/CONDITIONAL/OPEN 3 値）

---

## Tool Coverage

| 系統 | ツール | 実行内容 | 結果 |
| ---- | ------ | -------- | ---- |
| WSL | rg 15.1.0 | deferredSlot_/hasDeferred_/finishView/clearDeferredForShutdown lifecycle census / reserveRuntimePublicationIdentity callers / id generators / PublishRequest identity field 欠落確認 | 全 census 完了 |
| WSL | sg 0.44.0 | `sg run -p "struct DeferredPublishSlot"` パターン検証 | 実行確認 |
| WSL | ag 2.2.0 | `ag deferredOverwriteCount_|maxDeferredAgeMs_` — silent overwrite 証拠行（h:253-254）確定 | slot 分類の決定打 |
| WSL | fdfind 10.3.0 / fzf 0.67.0 | Orchestrator 所在 / filter 動作 | 確認 |
| WSL | awk 5.3.2 | slot field 行抽出（h:23-39,57,107,248-249） | 構造把握補助 |
| WSL | sed 4.9 / cat | Orchestrator.h 1-135 行全文 / cpp 380-480 行（enqueue/overwrite/clearDeferredForShutdown 本体）/ PublicationAdmission.h 10-30 行 | 本体照合完了 |
| Sandbox | ctx_execute shell | semble.exe `search "obligation identity publish request"` — **RuntimePublicationIdentity + reserveRuntimePublicationIdentity を新規発見**（99% 圧縮） | identity 監査の網羅性を向上 |
| MCP | serena / AiDex / CLI graphify 0.9.48 / ccc | project.yml / index.db 存在 / version 確認 | 利用可能状態確認 |
| read_file | AudioEngine.h 3420-3500 | RuntimePublicationIdentity 構造体 + reserve 関数本体 + worldId/sequence generators 確認 | 新規発見の完全照合 |
| 文献 | crossbeam-epoch / rigtorp 等 9 系統 | 前ステップまでに 200 OK 済の知識を再利用（O-1 では新規外部技術要件なし — 意味論監査は社内構造分析で完結） | 追加調査不要と判断 |
