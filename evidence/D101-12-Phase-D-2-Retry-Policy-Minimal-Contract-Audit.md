# D101-12 Phase D-2 — Retry Policy Minimal Contract / Scheduling Boundary Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-2` — Retry Policy 最小契約の監査・確定（**audit-only・契約書のみ**）
**Prerequisite:** D101-10 D-0 CLOSED / D101-11 D-1 CLOSED（`BuildError` 8値 / `kBuildErrorDefaultTable` 網羅 / Authority Matrix C判定）
**Unique source:** 実ワークツリー（`src/audioengine/RuntimeBuilder.h/.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`, `src/audioengine/AudioEngine.h`, `src/core/WorkerThread.cpp`）を一次資料とする

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | A `rg -n 'classifyBuildError\|kBuildErrorDefaultTable\|RetryDisposition\|FailureClassification'` / B `rg -n 'submitRebuildIntent\|submitRecoveryRequest\|settlePendingRecoveryAdmission\|retry'` / C `rg -n 'backoff\|exponential\|WaitableEvent\|steady_clock\|milliseconds'` / D `rg -n 'bool.*retry\|retryable\|shouldRetry\|canRetry'` | A: policy consumer 1箇所のみ（`RebuildDispatch.cpp:1098` log）、C: `RetryBackoff→scheduler` edge **0**、D: `bool retry` は `shouldRetryWarmupFailure` と `settle(bool retry)` の2箇所のみ |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'classifyBuildError($X)'` / `sg run -p 'RetryDisposition::$V'` / `sg run -p 'BuildOutcome'` / `sg run -p 'shouldRetryWarmupFailure($A)'` | 構造検索で `classifyBuildError` 1呼出、`RetryDisposition::*` は `RuntimeBuilder.h` のみに限定 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp . src` (324 files) → `xargs grep -l 'isLoadingIR'` / `fdfind -e h -e cpp 'RuntimeBuilder'` | `RuntimeBuilder.h/.cpp` のみに BuildError 家族が集約 |
| WSL | `ag` (silver searcher) | `ag -n 'submitRebuildIntent'` / `ag -n 'settlePendingRecovery'` / `ag -n 'kMaxRecovery'` | `rg` と一致、差異なし |
| WSL | `fzf` 0.67.0 | `fdfind ... \| fzf --filter='RebuildDispatch'` | パイプライン動作確認 |
| WSL | `sed` / `awk` | `sed -n '131,180p' RuntimeBuilder.h` / `awk '/BuildOutcome/{print}'` | `BuildOutcome` 3-field + `kBuildErrorDefaultTable` 8要素を抽出 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常 |
| CLI | `cocoindex` (`ccc.exe`) | `ccc status` (90068 chunks / 1737 files) / `ccc grep 'classifyBuildError'` / `ccc grep 'RetryBackoff'` | `rg` と一致、追加の隠れ consumer なし |
| CLI | `graphify` 0.9.39 | `graphify query 'BuildError'` / `graphify query 'classifyBuildError'` / `graphify path 'BuildError' 'submitRebuildIntent'` | `BuildError → classifyBuildError → BuildOutcome` の有向パスを確認、`BuildError → submitRebuildIntent` の直接 edge は存在せず |
| CLI | `semble` 0.5.3 | `semble search 'classifyBuildError' .` (3 hits) / `semble search 'RetryDisposition' .` (RuntimeBuilder.h のみ) / `semble search 'BuildOutcome' .` | `rg` と一致 |
| MCP | `AiDex` (index.db 26M) | `aidex_query term="classifyBuildError" mode="exact"` → 3 matches（定義/コメント/1呼出） / `aidex_query term="BuildOutcome" mode="contains"` | 差異なし |
| sandbox | `context-mode` `ctx_execute` | `ctx_execute(language: "javascript", code: ...)` で `src/` 全ファイルを横断 | 追加の隠れ producer なし |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep` 相当を WSL bash 経由（`rtk: No such file` 時は `grep` 直呼出しにフォールバック） | 出力差異なし |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` 断片は `context-mode` 仮想化で処理、headroom 圧縮は不要と判定 | フォールバック方針（headroom 不調時は context-mode 優先）を遵守 |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`ctx_execute` が一致）。ツール間の差異なし。

---

## D-2-1. 最初に固定する semantic chain（5層 + α）

```text
BuildError
    │  failure cause（静的: 8値、producer は build()/validateWarmup() のみ）
    ▼
FailureClassification
    │  failure category（Permanent / Transient / Infrastructure / Fatal）
    ▼
RetryDisposition
    │  retry scheduling policy（NoRetry / RetryBackoff / RetryImmediate）
    ▼
Retry Admission
    │  current-state / lifecycle eligibility（isLoadingIR() / Recovery lease / backpressure）
    ▼
Retry Scheduling
    │  when to execute（delay=0 / backoff delay / no schedule）
    ▼
submitRebuildIntent / scheduler（SPSC RebuildIntentQueue → rebuildThreadLoop）
```

### 境界の明文化

- **`RetryDisposition ≠ Retry Admission`** — 前者は「どうスケジュールすべきか」の policy、後者は「現在 retry 可能か」の eligibility。`WarmupFailed → RetryImmediate` でも `!isLoadingIR()` なら admission で reject（D-1 §D-1-2）。
- **`RetryDisposition ≠ Retry Scheduling`** — 前者は policy 型、後者は実行タイミング。`RetryBackoff` は `steady_clock`/`milliseconds`/`WaitableEvent` 等の scheduler 実体へ渡されるまで実行されない。現行は C 検索で `RetryBackoff → scheduler` edge **0**（未配線）。
- **`bool retryable` への縮退禁止** — `RetryDisposition`（3値）を `bool` に潰すと `RetryBackoff` vs `RetryImmediate` の区別が失われる。D-1 で `shouldRetryWarmupFailure() → bool` は warmup-specific eligibility として併存正当だが、Phase D の一般 policy を `bool` に縮退する API を新設してはならない（INV-D2-6）。

---

## D-2-2. `BuildContext` は「型」ではなく「契約」を先に決める

### 現行ソースの記述（`RuntimeBuilder.h:122-123, H:171`）

```cpp
//   実装は BuildError + BuildContext → FailureClassification → RetryDisposition の順で解決する
//   （BuildContext は将来拡張。現行はデフォルト表のみ — §1.8.12 未解決課題）。

// ★ §1.8.5.2 / H.11.2: BuildError → デフォルト分類の解決。
//   将来的に BuildContext（一時的 resource exhaustion / persistent config）で上書きする。
```

`rg BuildContext src/ --type cpp --type h` → **コメント3箇所のみ**（`RuntimeBuilder.h:122,123,171`）。`struct BuildContext` / `class BuildContext` / `using BuildContext` の定義は存在しない（D-0-9 と一致）。

### BuildContext が将来担えるもの（契約のみ、型は発明しない）

```text
BuildError
    +
execution/build context（例: 一時的 exhaustion か persistent 条件か）
    ↓ override
effective BuildOutcome（effective classification + effective disposition）
```

概念例（`REPAIR_PLAN2-dash2` の意図を契約として具体化、フィールドは発明しない）:

```text
default policy
  ResourceUnavailable → Transient / RetryBackoff

        ↓ context override（将来）

temporary resource exhaustion → Transient / RetryBackoff（維持）
persistent configuration/resource condition → Permanent/Fatal / NoRetry（上書き）
```

### なぜ今は型を発明しないか

- 現行の production producer は `InvalidInput`/`ResourceUnavailable`/`InternalError`/`WarmupFailed` の4種のみで、いずれも `BuildError` 単独で分類が一意に定まる（D-0-6）。
- `MKLFailure`/`ConvolverFailure`/`PrepareFailure` は producer なしの保険分類（D-0-6）。
- `ResourceUnavailable` に対して「一時的か永続的か」を区別する production input（例: `bad_alloc` の再試行可能性、config 検証結果）が現行 `RuntimeBuilder::build()` の `try/catch` 境界で観測可能でない。
- したがって **D-2 では BuildContext の具体的フィールドを発明しない**（終了条件）。Contract Test（D-3）では default table のみを検証し、effective policy は `classifyBuildError()` の将来 overload として予約する。

---

## D-2-3. 「default policy」と「effective policy」を分離する

### 現行

```cpp
// RuntimeBuilder.h:172
[[nodiscard]] inline BuildOutcome classifyBuildError(BuildError error) noexcept {
    const auto idx = static_cast<size_t>(error);
    if (idx >= sizeof(kBuildErrorDefaultTable)/sizeof(BuildOutcome))
        return { BuildError::InternalError, FailureClassification::Fatal, RetryDisposition::NoRetry };
    return kBuildErrorDefaultTable[idx]; // ← default table の値をそのまま返す
}
```

### 契約としての分離（D-2 で明文化、production API 変更は D-4/D-5）

```text
default classification（kBuildErrorDefaultTable）
        │  BuildError → BuildOutcome（table lookup、範囲外は InternalError/Fatal/NoRetry に安全側丸め）
        ▼
effective policy（将来 BuildContext override を挟む層）
        │  (BuildError, BuildContext) → BuildOutcome（現行は default と同一）
        ▼
RetryDisposition（effective BuildOutcome.retry）
```

- D-2 では `effective policy` を **契約として予約** するのみ。`classifyBuildError(BuildError, BuildContext)` の overload 型は設計書に記載するが、production 実装は行わない（INV-D2-7）。
- `BuildContext` が導入されるまで `effective == default` であることを invariant として固定する。

---

## D-2-4. Warmup retry の扱いを明文化する（D-1 継承）

```text
WarmupFailed
    │
    ├─ Build policy（classifyBuildError）
    │      Transient / RetryImmediate
    │      「retry may be immediate」（policy）
    │
    └─ runtime eligibility（shouldRetryWarmupFailure）
           isLoadingIR()
           「retry is currently admissible」（実行条件）
```

- `validateWarmup()` は `isIRLoaded() && !isIRFinalized()` を `WarmupFailed` として返す（`RuntimeBuilder.cpp:457`）— failure 原因の判定。
- `shouldRetryWarmupFailure()` は `isLoadingIR()`（`ConvolverProcessor.h:390` の `isLoading` atomic acquire）のみを見る — retry eligibility の判定。D-1 §D-1-2 で異なる semantic domain として確定。
- したがって `classifyBuildError(WarmupFailed)` から直接 `submitRebuildIntent()` する設計にしてはならない（禁止事項）。現行 `RebuildDispatch.cpp:1152-1168` の `retryable = shouldRetryWarmupFailure(*newDSP)` gate を維持する。
- 将来 `BuildContext` が `isLoadingIR()` を包摂する場合でも、D-2 では warmup retry の現行 behavior を変更しない（INV-D2-8）。

---

## D-2-5. Recovery retry を RetryPolicy に吸収しない

```text
BuildError policy
       │
       └── Build/rebuild failure semantics
           （build() failure → log → continue / warmup failure → eligibility gate → submitRebuildIntent）

Recovery lease
       │
       └── DurablePending ↔ Building lifecycle
           （takePendingRecoveryAdmission() → Building → settle(true/false) → DurablePending/NoAdmission）
           INV-X1-2: queue full ≠ Recovery lost
           kMaxRecoveryConsecutiveFailures=4（スピン防止、build/warmup failure 共通）
```

- `settlePendingRecoveryAdmission(true)`（`ISRRuntimePublicationCoordinator.cpp:968`）は `BuildError` の値を一切参照せず、単に「今回の Recovery 処理が失敗したので lease を再成立させる」状態遷移である（D-1 §D-1-4）。
- `BuildError → RetryDisposition` と Recovery lease は **直交する authority**（D-1 Authority Matrix #3 vs #5）。
- したがって `settlePendingRecoveryAdmission(true)` を `applyRetryDisposition(...)` に置換する設計は禁止。Recovery は recovery lifecycle authority のまま残す（INV-D2-5）。

---

## D-2-6. `RetryBackoff` の最小 scheduling contract を定義する

### 契約（3値の意味）

| Disposition | 契約 | scheduling 意味 |
|-------------|------|-----------------|
| `NoRetry` | retry request を発行しない | schedule なし（discard / `continue`） |
| `RetryImmediate` | admission が許可した場合、次の許可された execution point へ immediate retry を要求 | **scheduler delay = 0**（ただし「現在の関数内で再帰的に build」ではない） |
| `RetryBackoff` | admission が許可した場合、scheduler に delay/backoff policy を渡す | scheduler が `steady_clock` + `milliseconds` + `WaitableEvent` 等で delay を管理（D-5 で配線） |

### `RetryImmediate` の精密化

> **「Immediate = 現在の関数内で再帰的に build」ではない。**

現行 warmup retry は `submitRebuildIntent(Structural, RebuildThreadWarmupRetry)` を経由して **次の rebuild cycle（SPSC RebuildIntentQueue → `rebuildThreadLoop`）** に回る（`AudioEngine.RebuildDispatch.cpp:1165`）。したがって D-2 では `RetryImmediate` の意味を **`"next rebuild admission cycle"（scheduler delay = 0 の enqueue）`** とする。

- `submitRebuildIntent` は `AudioEngine.h:2962` で定義され、`RebuildIntentQueue`（`ISRRuntimePublicationCoordinator.h:629` — SPSC, Producer = CoordinatorLoop のみ）に enqueue する。
- 将来 `RetryBackoff` を配線する際は、同じ `submitRebuildIntent` 経路に `delay` パラメータを付与するか、別 scheduler（`WorkerThread` / `Timer`）経由で delay 後に `submitRebuildIntent` するかのいずれか（D-4/D-5 で設計）。

### 現行の配線状況

- `rg backoff|exponential|retry.*delay|retry.*timer` → `src/audioengine` では `RetryDisposition::RetryBackoff` のコメント（`RuntimeBuilder.h:133`）以外 **0**（D-2-8 C）。
- `steady_clock` / `milliseconds` は `core/WorkerThread.cpp`, `AudioEngine.h:3748`, `EpochDomain.h` 等で既に使用されているが、`RetryBackoff` との接続は存在しない。
- したがって `RetryBackoff → actual scheduler` の edge は **0**（D-1 §D-1-1 と一致）。D-2 では contract のみを定義し、実装は D-5。

---

## D-2-7. 推奨する最小 contract

### Layer 表

| Layer | Authority | Input | Output | D-2 status |
|-------|-----------|-------|--------|------------|
| Failure | `BuildError` | build/validation（`build()` / `validateWarmup()`） | `BuildError` (8値) | **existing**（`RuntimeBuilder.cpp:417-458`、static_assert 保証） |
| Classification | `classifyBuildError()` | `BuildError` | `FailureClassification` (4値) | **existing**（`RuntimeBuilder.h:172`） |
| Policy | default table (`kBuildErrorDefaultTable`) | `BuildError` | `RetryDisposition` (3値) / `BuildOutcome` | **existing**（8要素、範囲外は `InternalError/Fatal/NoRetry` に安全側丸め） |
| Context override | `BuildContext` | context（将来） | effective `BuildOutcome` | **contract only**（型未実装、effective==default） |
| Eligibility | warmup/recovery-specific authority | runtime/lifecycle state（`isLoadingIR()` / Recovery lease） | admit/reject | **existing**（`shouldRetryWarmupFailure()` / `take`/`settle` lease） |
| Scheduling | Retry scheduler | `RetryDisposition` | execution timing（delay=0 / backoff delay / no schedule） | **contract only**（`RetryBackoff` 未配線、`RetryImmediate` は next-cycle enqueue） |
| Execution | rebuild admission | scheduled request | retry attempt（`submitRebuildIntent` → `RebuildIntentQueue` → `rebuildThreadLoop`） | **existing**（warmup path） / **future wiring**（`ResourceUnavailable→RetryBackoff` 等） |

### D-2 invariant（8件）

```text
INV-D2-1  BuildError は retry policy を直接保持しない。
          （BuildError は failure 原因のみ。policy は classifyBuildError() が table から解決）

INV-D2-2  FailureClassification は retry scheduling を直接実行しない。
          （classification は category のみ。NoRetry/RetryBackoff/RetryImmediate は RetryDisposition が持つ）

INV-D2-3  RetryDisposition は retry scheduling policy を表すが、retry execution 自体を行わない。
          （policy 型であり、submitRebuildIntent / settle 等の実行は別 layer）

INV-D2-4  Eligibility は RetryDisposition とは別 authority である。
          （WarmupFailed: RetryImmediate でも !isLoadingIR() なら admission で reject）

INV-D2-5  Recovery lease retry は RetryDisposition とは別 authority である。
          （settle(true) は recovery lifecycle。BuildError 分類を参照しない）

INV-D2-6  RetryBackoff を bool に縮退してはならない。
          （RetryDisposition 3値 → bool への縮退で Backoff vs Immediate の区別が失われる）

INV-D2-7  BuildContext は D-2 では production implementation しない。
          （struct 未定義のまま。default==effective として contract のみ予約）

INV-D2-8  Warmup retry の現行 behavior は D-2 では変更しない。
          （shouldRetryWarmupFailure() gate + submitRebuildIntent 経路を維持）
```

---

## D-2-8. 実施する監査 — 4検索の結果

### A. Policy consumer 全列挙

```bash
rg -n 'classifyBuildError|kBuildErrorDefaultTable|RetryDisposition|FailureClassification' src/ --type cpp --type h
```

| ファイル | 行 | 内容 | 判定 |
|----------|----|------|------|
| `RuntimeBuilder.h:124` | `enum class FailureClassification` | 定義 | existing |
| `RuntimeBuilder.h:131` | `enum class RetryDisposition` | 定義（3値） | existing |
| `RuntimeBuilder.h:137` | `struct BuildOutcome` | 定義（3-field） | existing |
| `RuntimeBuilder.h:146` | `kBuildErrorDefaultTable` (8要素) | default policy 定義 | existing |
| `RuntimeBuilder.h:172` | `classifyBuildError()` | default 分類解決（範囲外は `InternalError/Fatal/NoRetry`） | existing |
| `RuntimeBuilder.h:181` | `classifyBuildErrorToString()` | 文字列表現（table 駆動） | existing |
| `RuntimeBuilder.cpp:57` | `toString()` | `classifyBuildErrorToString` への委譲 | existing |
| `AudioEngine.RebuildDispatch.cpp:1098` | `classifyBuildError(buildResult.error)` | **唯一の production 呼出**（`outcome.classification`/`outcome.retry` を `diagLog` に記録、retry 未適用） | telemetry only |

`sg run -p 'classifyBuildError($X)'` / `semble search 'classifyBuildError'` / `AiDex classifyBuildError` / `cocoindex grep 'classifyBuildError'` / `graphify query 'classifyBuildError'` 全てで **1呼出**に一致。

### B. Retry execution consumer 全列挙

```bash
rg -n 'submitRebuildIntent|submitRecoveryRequest|submitRecoveryIntent|settlePendingRecoveryAdmission|retry' src/ --type cpp --type h
```

- `submitRebuildIntent` — `AudioEngine.Parameters.cpp` (15箇所、通常の構造変更) + `AudioEngine.RebuildDispatch.cpp:149` (定義), `:447` (durable recovery success 後の再投入), `:1165` (warmup retry) + `AudioEngine.Timer.cpp` + `AudioEngine.Init.cpp` 等。Phase D の `BuildError → retry` 経路で使われるのは `:1165` の warmup retry のみ。
- `submitRecoveryRequest` — `ISRRuntimePublicationCoordinator.cpp:855` 定義、`AudioEngine.h:4473` 経由で `QuarantineIntentHandler` が呼出。Recovery 経路専用。
- `settlePendingRecoveryAdmission` — `RebuildDispatch.cpp:1011,1022,1044,1062` の4箇所（durable recovery lease）+ `ISRRuntimePublicationCoordinator.cpp:968` 定義 + テスト `ISRSemanticValidationTests.cpp`。
- `retry` 一般 — `DeferredRetireFallbackQueue.h:retryCount`, `ISRShutdown.cpp:kMaxRenameRetries` 等、Phase D と無関係な retry（確認済み、除外）。

### C. Backoff 実体確認

```bash
rg -n 'backoff|exponential|retry.*delay|retry.*timer|Timer|WaitableEvent|steady_clock|milliseconds' src/audioengine src/core --type cpp --type h
```

| 検索語 | 結果 |
|--------|------|
| `backoff` | `RuntimeBuilder.h:133` コメント `RetryBackoff, // exponential backoff 付き retry` のみ |
| `exponential` | 同上のみ |
| `retry.*delay` / `retry.*timer` | **0** |
| `WaitableEvent` | **0**（`src/audioengine` / `src/core` に存在せず） |
| `steady_clock` / `milliseconds` | `core/WorkerThread.cpp:68,89`, `AudioEngine.h:3748`, `core/TimeUtils.h:17`, `EpochDomain.h:136` 等で使用されるが、`RetryBackoff` との接続は **0** |

**結論:** `RetryBackoff → actual scheduler` の edge は **0**（D-1 §D-1-1 と一致）。Backoff timer / exponential backoff の実装は存在しない。

### D. bool collapse 検索

```bash
rg -n 'bool.*retry|retryable|shouldRetry|canRetry' src/audioengine --type cpp --type h
```

| ヒット | ファイル | 意味 | 判定 |
|--------|----------|------|------|
| `bool shouldRetryWarmupFailure(const DSPCore&)` | `RebuildDispatch.cpp:78` | `isLoadingIR()` のみを見る eligibility guard | **併存正当**（D-1 §D-1-2）、false positive ではなく semantic boundary として確定 |
| `const bool retryable = shouldRetryWarmupFailure(*newDSP)` | `RebuildDispatch.cpp:1155` | 上記の呼出し（`RetryDisposition` ではなく `bool` への縮退だが、warmup-specific のため許容） | `RetryImmediate` を即時 `submitRebuildIntent` に変換する admission decision |
| `void settlePendingRecoveryAdmission(bool retry)` | `ISRRuntimePublicationCoordinator.h:277` | `bool retry` は `Building → DurablePending` (true) vs `Building → NoAdmission` (false) の lease 状態遷移 | Recovery lifecycle の `bool` であり、Phase D の `RetryDisposition` と無関係（INV-D2-5） |
| `bool canRetry` / `canRetry` | — | **0** | — |
| `shouldRetry(BuildError)` 的な汎用 `BuildError` 対象の `bool` | — | **0** | D-1 禁止事項を遵守 |

`RetryDisposition` を `bool` に縮退する汎用 API は存在しない。

---

## D-2-9. 最重要の設計判断 — 案A vs 案B

### 案A

```text
classifyBuildError()
    ↓
RetryDisposition
    ↓
scheduler
```

`RetryDisposition` 単独を scheduler に渡す。`FailureClassification` と `BuildError` は捨てる。

### 案B

```text
classifyBuildError()
    ↓
BuildOutcome { BuildError, FailureClassification, RetryDisposition }
    ↓
retry admission（eligibility gate）
    ↓
scheduler
```

`BuildOutcome` 一体を caller に渡し、admission で `RetryDisposition` を参照しつつ、logging / telemetry / 将来の `BuildContext` override で `BuildError` / `FailureClassification` も利用可能に保つ。

### 検証

| 観点 | 案A | 案B | 判定 |
|------|-----|-----|------|
| **caller contract** | `RetryDisposition foo = classifyBuildError(err).retry` — 1 field のみ | `BuildOutcome o = classifyBuildError(err)` — 3 field 一体 | 案B: `RebuildDispatch.cpp:1098` は既に `BuildOutcome outcome` 全体を `diagLog` で `classification`/`retry` 共に利用。案A にすると `classification` が失われ、将来 `BuildContext` で classification を上書きする際に caller 側の再設計が必要 |
| **ownership / lifetime** | `RetryDisposition` は `uint8_t` enum（値型、trivially copyable） | `BuildOutcome` は `BuildError(4B) + FailureClassification(1B) + RetryDisposition(1B)` の trivially copyable な値型（`RuntimeBuilder.h:137`）— heap なし、lifetime 問題なし | 両案とも trivial。差異なし |
| **threading boundary** | `classifyBuildError()` は `inline` / `noexcept` / `constexpr` table 参照のみ（`RuntimeBuilder.h:172`）— どのスレッドから呼んでも安全（atomic なし、lock なし） | 同左（`BuildOutcome` も同様に inline table lookup） | 両案とも thread-safe。差異なし |
| **既存実装との適合性** | `kBuildErrorDefaultTable` は `BuildOutcome[]` として定義済み（H:146）。案A では table の `FailureClassification` が未使用になる | `BuildOutcome` が既に table 要素型として存在し、`classifyBuildError()` は `BuildOutcome` を返す。既存コードと完全一致 | **案B が既存実装と整合** |
| **Contract Test の網羅性** | `RetryDisposition` のみを検証（3値）— `FailureClassification` の誤りは検出できない | `BuildOutcome` 全体を検証（`BuildError` → `FailureClassification` × `RetryDisposition` の8組）— default table の完全性を1テストで固定 | **案B が網羅的**（D-3 で8値の契約表を固定する要件と一致） |
| **将来の `BuildContext` 拡張** | `BuildContext` が `FailureClassification` を上書きする場合、caller が `RetryDisposition` しか持たないと再分類できない | `BuildOutcome` 全体を持つため、`effective = classifyBuildError(err, ctx)` で `classification` と `retry` を一体で上書き可能 | **案B が拡張性を持つ** |
| **bool collapse 防止** | `RetryDisposition` 単独でも `bool` への縮退は防げる | `BuildOutcome` は `bool` への縮退を構造的に防ぐ（`BuildOutcome` → `bool` の暗黙変換なし） | 両案とも INV-D2-6 を満たすが、案B がより構造的に防ぐ |

### Ratify

**案B を ratify する。**

```text
classifyBuildError(BuildError) → BuildOutcome → retry admission → scheduler
```

理由: `BuildOutcome` が既に `BuildError` / `FailureClassification` / `RetryDisposition` を一体として表現しており（`RuntimeBuilder.h:137`）、caller contract・ownership・threading boundary の全てで案A に対する不利益がなく、Contract Test の網羅性と将来の `BuildContext` 拡張で優位である。`graphify path 'BuildError' 'submitRebuildIntent'` が直接 edge を持たないことも、間に `BuildOutcome` / admission / scheduler の layer を挟む案B の設計を支持する。

---

## D-2 終了判定

```
[x] BuildError → Classification → Disposition の semantic chain 確定     — §D-2-1（5層）
[x] default policy と effective policy の境界確定                        — §D-2-3（default==effective、将来 override 予約）
[x] BuildContext の必要性を再監査                                        — §D-2-2（現行 production は不要、型未定義を確認）
[x] BuildContext の production implementation = 0                         — rg BuildContext → コメント3箇所のみ、struct 0
[x] RetryDisposition と Retry Admission の境界確定                        — §D-2-1, §D-2-4（policy vs eligibility）
[x] RetryDisposition と Scheduling の境界確定                             — §D-2-6（policy vs execution timing）
[x] RetryBackoff の scheduler interface 必要条件確定                      — §D-2-6（delay/backoff policy を scheduler に渡す、delay=0 vs backoff 分離）
[x] RetryBackoff の実装 = 0                                               — §D-2-8 C（edge 0、timer/backoff なし）
[x] Warmup eligibility authority 維持                                     — §D-2-4（shouldRetryWarmupFailure gate 維持、INV-D2-8）
[x] Recovery lifecycle authority 維持                                     — §D-2-5（settle lease は別 authority、INV-D2-5）
[x] bool retryable への policy collapse 禁止を確定                        — §D-2-8 D（汎用 bool 縮退 0、INV-D2-6）
[x] BuildOutcome の扱いを ratify                                          — §D-2-9（案B ratified）
[x] D-3 Contract Test の入力/出力ケースを列挙                             — 下記申送り（8値 + enum/table coverage + consistency）
[x] production code 変更 = 0                                              — git diff 0（audit-only）
```

**D-2 判定: B = minimal contract ratified → D-3 へ**

---

## D-3 への明確な申送り

D-2 が **B = minimal contract ratified** となったため、次は **D-3 Contract Test**。

### D-3 で固定する契約（`kBuildErrorDefaultTable` と一致）

```
InvalidInput        → Permanent      / NoRetry
ResourceUnavailable → Transient      / RetryBackoff
MKLFailure          → Fatal          / NoRetry
ConvolverFailure    → Infrastructure / RetryBackoff
PrepareFailure      → Infrastructure / RetryBackoff
WarmupFailed        → Transient      / RetryImmediate
InternalError       → Fatal          / NoRetry
None                → Permanent      / NoRetry （成功時は retry 不要）
```

### D-3 テスト要件（契約書から導出）

| Test | 検証 |
|------|------|
| `None` | `Permanent` / `NoRetry` |
| `InvalidInput` | `Permanent` / `NoRetry` |
| `ResourceUnavailable` | `Transient` / `RetryBackoff` |
| `MKLFailure` | `Fatal` / `NoRetry` |
| `ConvolverFailure` | `Infrastructure` / `RetryBackoff` |
| `PrepareFailure` | `Infrastructure` / `RetryBackoff` |
| `WarmupFailed` | `Transient` / `RetryImmediate` |
| `InternalError` | `Fatal` / `NoRetry` |
| enum/table coverage | 全 `BuildError` (8値) が分類表に存在（`static_assert` と独立にテストで検証） |
| classification consistency | classification と retry の組み合わせが契約違反でない（例: `Fatal → NoRetry` のみ、`Transient/Infrastructure → Retry*` のみ） |

テストは `RuntimeHealthMonitorTierTests` 型の **小さな独立した契約テスト** として `src/tests/` に追加し、`BuildOutcome` 全体（案B）を検証する。`ResourceUnavailable → RetryBackoff` の **実装**（scheduler 配線）は D-3 では行わず、契約の固定のみ。

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'classifyBuildError|kBuildErrorDefaultTable|RetryDisposition|FailureClassification' src/ --type cpp --type h
rg -n 'submitRebuildIntent|submitRecoveryRequest|submitRecoveryIntent|settlePendingRecoveryAdmission|retry' src/ --type cpp --type h
rg -n 'backoff|exponential|retry.*delay|retry.*timer|WaitableEvent|steady_clock|milliseconds' src/audioengine src/core --type cpp --type h
rg -n 'bool.*retry|retryable|shouldRetry|canRetry' src/audioengine --type cpp --type h
grep -rn 'BuildContext' src/ --include='*.h' --include='*.cpp'
sed -n '118,260p' src/audioengine/RuntimeBuilder.h
sed -n '131,180p' src/audioengine/RuntimeBuilder.h
awk '/BuildOutcome/{print FNR": "$0}' src/audioengine/RuntimeBuilder.h
fdfind -e h -e cpp . src | xargs grep -l 'isLoadingIR\|isIRLoaded\|isIRFinalized'
ag -n 'submitRebuildIntent' src/
ag -n 'settlePendingRecovery' src/
echo 'RuntimeBuilder.h' | fzf --filter='Runtime'
sg run -p 'classifyBuildError($X)' --lang cpp src/
sg run -p 'RetryDisposition::$V' --lang cpp src/
sg run -p 'shouldRetryWarmupFailure($A)' --lang cpp src/
sg run -p 'settlePendingRecoveryAdmission($X)' --lang cpp src/
sg run -p 'BuildOutcome' --lang cpp src/

# cocoindex
ccc status
ccc grep 'classifyBuildError'
ccc grep 'RetryBackoff'
ccc grep 'shouldRetryWarmupFailure'
ccc search "BuildError"

# graphify
graphify query "BuildError"
graphify query "classifyBuildError"
graphify path "BuildError" "RetryDisposition"
graphify path "BuildError" "submitRebuildIntent"

# semble
semble search "classifyBuildError" . --max-snippet-lines 5
semble search "RetryDisposition" . --max-snippet-lines 5
semble search "BuildOutcome" . --max-snippet-lines 5
semble search "shouldRetryWarmupFailure" . --max-snippet-lines 8

# AiDex
aidex_query term="classifyBuildError" mode="exact"
aidex_query term="BuildOutcome" mode="contains"
aidex_query term="BuildContext" mode="contains"

# serena
find_symbol name_path_pattern="shouldRetryWarmupFailure" relative_path="src/audioengine/AudioEngine.RebuildDispatch.cpp"

# context-mode sandbox
ctx_execute(language: "javascript", code: "fs.readdirSync('src/audioengine').filter(f=>fs.readFileSync(...).includes('BuildError'))")

# RTK (WSL版)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "BuildError" src/'

# headroom — 大きな断片は context-mode で仮想化、圧縮不要と判定
```

## 参照

- `src/audioengine/RuntimeBuilder.h:107-210` — BuildError 家族の定義集約（§1.8 / H.11 契約の一次資料）
- `src/audioengine/RuntimeBuilder.h:137` — `BuildOutcome` 定義（案B の核心）
- `src/audioengine/RuntimeBuilder.h:146-177` — `kBuildErrorDefaultTable` + `classifyBuildError()`（default policy）
- `src/audioengine/RuntimeBuilder.cpp:402-461` — 唯一の producer（`build()` / `validateWarmup()`）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:78-81` — `shouldRetryWarmupFailure()` 定義（eligibility guard）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1091-1168` — Path A（build failure log）+ Path B（warmup retry + eligibility gate）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:996-1062` — Path D（durable recovery lease + `settle(true)` + `kMaxRecoveryConsecutiveFailures=4`）
- `src/audioengine/ISRRuntimePublicationCoordinator.h:277` / `.cpp:968` — `settlePendingRecoveryAdmission()` 定義
- `src/audioengine/AudioEngine.h:2962` — `submitRebuildIntent()` 定義（SPSC RebuildIntentQueue）
- `evidence/D101-10-Phase-D-0-Audit-Report.md` — D-0 監査報告書
- `evidence/D101-11-Phase-D-1-Retry-Policy-Authority-Audit.md` — D-1 監査報告書（Authority Matrix C判定）
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 / H.11 参照コメントの一次資料）
