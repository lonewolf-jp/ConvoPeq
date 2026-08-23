# D101-11 Phase D-1 — Retry Policy Authority / Eligibility Boundary Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-1` — Retry Policy Authority の監査・契約確定（**変更禁止・audit-only**）
**Prerequisite:** D101-10 D-0 CLOSED（`BuildError` 8値 / `kBuildErrorDefaultTable` 網羅済み）、5-VI-F CLOSED 維持
**Unique source:** 実ワークツリー（`src/audioengine/RuntimeBuilder.h/.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`, `src/audioengine/AudioEngine.h`）を一次資料とする

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | `rg -n 'submitRebuildIntent\|submitRecovery\|settlePendingRecoveryAdmission\|shouldRetryWarmupFailure\|kMaxRecoveryConsecutiveFailures\|RetryImmediate\|RetryBackoff\|NoRetry\|retryable\|validateWarmup\|buildResult\.error\|classifyBuildError' src/ --type cpp --type h`（600行） | retry 有向グラフの全 edge を列挙（下記 §D-1-1） |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'shouldRetryWarmupFailure($A)'` / `sg run -p 'settlePendingRecoveryAdmission($X)'` / `sg run -p 'RetryDisposition::$V'` / `sg run -p 'classifyBuildError($X)'` | 構造検索で `shouldRetryWarmupFailure` 1定義1呼出し、`settle` 4呼出し、`RetryDisposition::*` は `RuntimeBuilder.h` のみに限定 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp . src` (324 files) → `xargs grep -l 'isLoadingIR\|isIRLoaded\|isIRFinalized'` / `fdfind -e cpp . src \| fzf --filter='RebuildDispatch'` | IR状態3述語の定義・使用箇所を網羅（`ConvolverProcessor.h`/`RuntimeBuilder.cpp`/`RebuildDispatch.cpp`/`Parameters.cpp` 等） |
| WSL | `ag` (silver searcher) | `ag -n 'submitRebuildIntent' src/` / `ag -n 'settlePendingRecovery' src/` / `ag -n 'kMaxRecovery' src/` | `rg` と一致、差異なし |
| WSL | `fzf` 0.67.0 | `fdfind ... \| fzf --filter='RebuildDispatch'` | パイプライン動作確認 |
| WSL | `sed` / `awk` | `sed -n '700,900p' AudioEngine.RebuildDispatch.cpp` / `grep -n 'shouldRetryWarmupFailure' -A 40 -B 5` / `awk '/RetryDisposition/'` | `shouldRetryWarmupFailure` 本体（`isLoadingIR()` のみ）を抽出 |
| MCP | `serena` 1.7.0 | `find_symbol name_path_pattern="shouldRetryWarmupFailure" relative_path="src/audioengine/AudioEngine.RebuildDispatch.cpp"` → `(anonymous namespace)/shouldRetryWarmupFailure` (L77-80) | symbol 索引が retry 述語を正しく捕捉 |
| CLI | `cocoindex` (`ccc.exe`) | `ccc status` (90068 chunks / 1737 files) / `ccc grep 'shouldRetryWarmupFailure'` / `ccc grep 'settlePendingRecoveryAdmission'` | `rg` と一致する唯一性を確認 |
| CLI | `graphify` 0.9.39 | `graphify query "shouldRetryWarmupFailure"` / `graphify query "BuildError"` | `shouldRetryWarmupFailure` は orphan ノード（`BuildError` graph と有向パスなし）— semantic domain 分離の傍証 |
| CLI | `semble` 0.5.3 | `semble search "shouldRetryWarmupFailure" .` (2 hits: L78 def, L1155 call) / `semble search "settlePendingRecoveryAdmission" .` (10 hits) / `semble search "RetryDisposition" .` (RuntimeBuilder.h のみ) | WSL grep と一致 |
| MCP | `AiDex` (index.db 26M) | `aidex_query term="shouldRetryWarmupFailure" mode="exact"` → 2 matches / `aidex_query term="settlePendingRecoveryAdmission" mode="exact"` → 10 matches | 差異なし |
| sandbox | `context-mode` `ctx_execute` | `ctx_execute(language: "javascript", code: ...)` で `src/` 全ファイルを `BuildError` でフィルタ | 追加の隠れ producer なし（D-0 と一致） |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep` 相当を WSL bash 経由で実行（`rtk: No such file` 時は `grep` 直呼出しにフォールバック） | 出力差異なし |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` 断片は `context-mode` 仮想化で処理、headroom 圧縮は不要と判定 | フォールバック方針（headroom 不調時は context-mode 優先）を遵守 |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`serena`=`ctx_execute` が一致）。ツール間の差異なし。

---

## D-1-1. Retry decision の全経路 — 有向グラフ

### 対象キーワード（指示された全検索語）

```
submitRebuildIntent / submitRecoveryRequest(=submitRecoveryIntent) / settlePendingRecoveryAdmission
shouldRetryWarmupFailure / kMaxRecoveryConsecutiveFailures
RetryImmediate / RetryBackoff / NoRetry / retryable
warmup / buildResult.error / validateWarmup
classifyBuildError / BuildOutcome / FailureClassification / RetryDisposition
```

`rg` 全出力（600行）を集約し、**Build failure → retry decision → retry action** の有向グラフを再構成した。

### グラフ（production のみ・テストは括弧で注記）

```text
                    ┌─ Build Path A: build() failure ──────────────────────┐
                    │                                                       │
  RuntimeBuilder::build()                                                   │
    ├─ InvalidInput ──────────────┐                                         │
    ├─ ResourceUnavailable ───────┤                                         │
    └─ catch(...) → InternalError ┤                                         │
                                  ▼                                         │
                         BuildResult{runtime==nullptr, error}                │
                                  │                                         │
                                  ▼                                         │
              AudioEngine.RebuildDispatch.cpp:1091                          │
              if (buildResult.runtime == nullptr) {                         │
                outcome = classifyBuildError(buildResult.error)  ──► log only│
                // classification / retry を diagLog に記録                  │
                // retry 実適用なし（将来 D-5 で backoff と併せて拡張予定） │
                continue;  // ← NoRetry（実質 discard）                    │
              }                                                             │
                    └───────────────────────────────────────────────────────┘

                    ┌─ Build Path B: warmup failure (main Rebuild) ─────────┐
                    │                                                       │
  RuntimeBuilder::validateWarmup(*newDSP)                                   │
    └─ isIRLoaded() && !isIRFinalized() → WarmupFailed                     │
                                  │                                         │
                                  ▼                                         │
              AudioEngine.RebuildDispatch.cpp:1152                          │
              warmupError = validateWarmup(*newDSP)                         │
              if (warmupError != None) {                                    │
                retryable = shouldRetryWarmupFailure(*newDSP)               │
                          = isLoadingIR()  // ← runtime state 判定         │
                diagLog(..., retryable, irLoaded, irFinalized, isLoading)   │
                if (retryable)                                              │
                  submitRebuildIntent(Structural, RebuildThreadWarmupRetry) │
                //   └─► RebuildIntentQueue (SPSC, CoordinatorLoop)       │
                //       └─► 次サイクルの build() で再試行（即時 admission）│
                continue;  // 未コミット DSP は destroyDSPCoreNode で破棄   │
              }                                                             │
                    └───────────────────────────────────────────────────────┘

                    ┌─ Build Path C: recovery (popRecoveryRequest) ─────────┐
                    │                                                       │
  runtimePublicationBridge_.popRecoveryRequest()  ──► RecoveryIntent        │
    ├─ build() failure → log + continue (retry なし、DSP破棄)              │
    └─ warmup failure → log + destroyDSPCoreNode + continue (retry なし)  │
              // この経路では WarmupFailed の retry は行わない              │
              // （transient でも discard — 次の recovery pop で別 intent） │
                    └───────────────────────────────────────────────────────┘

                    ┌─ Build Path D: durable recovery (lease) ──────────────┐
                    │                                                       │
  runtimePublicationBridge_.takePendingRecoveryAdmission()                  │
    // DurablePending → Building の lease（INV-X1-2: queue full ≠ lost）   │
    ├─ qHandle invalid → settle(false) → Discarded                         │
    ├─ build() failure → settle(true) → DurablePending へ戻す（retry）     │
    │                  └─ kMaxRecoveryConsecutiveFailures=4 でスピン防止  │
    │                     上限超過 → break（次サイクルへ委譲、lossなし）    │
    ├─ warmup failure → destroyDSPCoreNode + settle(true) → retry         │
    │                  └─ 同じく 4回上限                                    │
    └─ success → settle(false) → クリア + enqueuePublicationIntent         │
              // settle(true) は「Recovery lease の再成立」であり             │
              // BuildError 分類を参照しない（recovery lifecycle authority）│
                    └───────────────────────────────────────────────────────┘
```

### 経路別の `ResourceUnavailable` / `WarmupFailed` 追跡

| BuildError | 発生箇所 | 分類（default table） | 実際の retry 経路 | 配線状況 |
|------------|----------|-----------------------|-------------------|----------|
| `InvalidInput` | `RuntimeBuilder.cpp:417` | `Permanent/NoRetry` | Path A: `continue`（discard）— retry なし | 分類と挙動が一致（NoRetry） |
| `ResourceUnavailable` | `RuntimeBuilder.cpp:443` (`bad_alloc`) | `Transient/RetryBackoff` | Path A: `continue`（discard）— **retry なし**（backoff 未実装） / Path D: `settle(true)` で durable recovery のみ retry | **未配線**（`RetryBackoff` は定義のみ、timer/backoff なし） |
| `MKLFailure` | producer なし（保険） | `Fatal/NoRetry` | 経路なし | 休眠 |
| `ConvolverFailure` | producer なし（保険） | `Infrastructure/RetryBackoff` | 経路なし | 休眠 |
| `PrepareFailure` | producer なし（保険） | `Infrastructure/RetryBackoff` | 経路なし | 休眠 |
| `WarmupFailed` | `RuntimeBuilder.cpp:458` | `Transient/RetryImmediate` | Path B: `shouldRetryWarmupFailure()` → `submitRebuildIntent`（即時） / Path D: `settle(true)`（次サイクル） | Path B で **state-gated 即時 retry** が実装済み |
| `InternalError` | `RuntimeBuilder.cpp:448` | `Fatal/NoRetry` | Path A: `continue`（discard） | 分類と挙動が一致（NoRetry） |
| `None` | 成功時 | `Permanent/NoRetry` | 正常系（`runtime != nullptr` で後続へ） | — |

**要点:**

- `ResourceUnavailable → RetryBackoff` は **分類上は retry だが実行経路が存在しない**（backoff timer 未実装、D-0-8 と一致）。
- `WarmupFailed → RetryImmediate` は **唯一 retry が実行される BuildError** だが、その実行は `shouldRetryWarmupFailure()` の gate を経由する。
- `InternalError` / `InvalidInput` の `NoRetry` は Path A の `continue`（discard）として正しく実装されている。

---

## D-1-2. `shouldRetryWarmupFailure()` の意味を確定

### 定義（唯一の定義）

`src/audioengine/AudioEngine.RebuildDispatch.cpp:78` — anonymous namespace

```cpp
bool shouldRetryWarmupFailure(const AudioEngine::DSPCore& dsp) noexcept
{
    return dsp.convolverRt().isLoadingIR();
}
```

`serena find_symbol` / `AiDex` / `semble` / `sg` / `rg` 全てで **1定義・1呼出し**（L1155）のみ。

### 呼出し側（唯一の呼出し）

`AudioEngine.RebuildDispatch.cpp:1152-1168` — main Rebuild path のみ

```cpp
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
if (warmupError != convo::BuildError::None)
{
    const bool retryable = shouldRetryWarmupFailure(*newDSP); // ← isLoadingIR()
    diagLog("[DIAG] ... warmup failed error=" + toString(warmupError)
            + " retryable=" + int(retryable)
            + " irLoaded=" + int(newDSP->convolverRt().isIRLoaded())
            + " irFinalized=" + int(newDSP->convolverRt().isIRFinalized())
            + " irLoading=" + int(newDSP->convolverRt().isLoadingIR()));
    if (retryable)
        submitRebuildIntent(Structural, RebuildThreadWarmupRetry, ...);
    continue;
}
```

### 入力の比較

| 述語 | 入力 | 判定内容 | semantic domain |
|------|------|----------|-----------------|
| `validateWarmup(*dsp)` → `WarmupFailed` | `isIRLoaded() && !isIRFinalized()` | **failure 原因**（IRが loaded だが finalized されていない異常状態） | **BuildError classification**（failure cause） |
| `shouldRetryWarmupFailure(*dsp)` → `bool` | `isLoadingIR()` | **retry 実行条件**（IRが現在 loading 中か = transient な未完了か） | **retry eligibility**（runtime state） |

両者は **異なる入力・異なる述語・異なる domain** である:

- `WarmupFailed` は `isIRLoaded && !isIRFinalized` で立つ（IRが存在するが未完成）。
- `isLoadingIR` は `convo::consumeAtomic(isLoading, acquire)` で読む独立した atomic（`ConvolverProcessor.h:390`、LoaderThread の release と HB）。`LoadPipeline.cpp:564` で `irFinalized` とは別に管理される。
- `isLoadingIR()==true` なら「loading が進行中なので warmup retry が成功する可能性がある」→ `submitRebuildIntent` で即時再試行。
- `isLoadingIR()==false` なら「IRは loaded だが finalized されず、かつ loading もしていない = 永続的破損」→ retry しても回復しないので discard。

### 同一 authority か否か

**No — 異なる semantic domain である。**

```text
BuildError::WarmupFailed          — failure 原因（静的分類）
        ≠
shouldRetryWarmupFailure(isLoadingIR) — retry eligibility（動的状態 guard）
```

`classifyBuildError(WarmupFailed) → {Transient, RetryImmediate}` は「WarmupFailed は retry の対象になり得る」という **policy（方針）** を表す。一方 `shouldRetryWarmupFailure()` は「現在の IR loading 状態で retry が成功し得るか」という **eligibility（実行条件）** を表す。両者は直交する:

- policy が `RetryImmediate` でも、eligibility が `false`（`!isLoadingIR()`）なら retry しない。
- eligibility が `true` でも、policy が `NoRetry` なら retry しない（ただし `WarmupFailed` の policy は `RetryImmediate` なので現行は両者が一致）。

したがって **「二重 authority だから削除」は誤り**。削除すれば、loading 中でない永続的 WarmupFailed に対しても無条件で retry する過剰 retry、あるいは loading 完了後の transient な WarmupFailed を retry し損ねる、のいずれかが生じる。

**D-1 判定:** `shouldRetryWarmupFailure()` は `BuildError` classification とは異なる責務（state-dependent admission guard）であり、**併存が正当**。

ただし、名称・コメントによる責務境界の明文化が望ましい（後述 §D-1-5）。

---

## D-1-3. `RetryDisposition` の semantic boundary を確定

### 3層の分離

```text
FailureClassification          — 失敗の性質（永続 / 一時 / 環境依存 / 致命的）
        │
        ▼
RetryDisposition               — 推奨される retry のスケジューリング方針
        │
        ▼
Retry Admission / Scheduling   — 実際に retry を発行するか・いつ発行するか
```

### 各 `RetryDisposition` 値の意味

| 値 | 意味 | semantic layer |
|----|------|----------------|
| `NoRetry` | **「retry してはならない」** — retry 自体の禁止（policy） | `RetryDisposition` 自体が決定 |
| `RetryBackoff` | **「retry してよいが、exponential backoff を挟むべき」** — scheduling 方針 | `RetryDisposition` が scheduling を示唆 |
| `RetryImmediate` | **「retry してよく、即時でよい」** — scheduling 方針（latency-sensitive） | `RetryDisposition` が scheduling を示唆 |

`RetryDisposition` は **「retryしてよい」だけでなく「どうスケジュールするか」まで含む型** である（`RuntimeBuilder.h:132-134` コメント `exponential backoff 付き` / `immediate retry（latency-sensitive）` が明示）。

### caller 側の `bool retryable` との混同

`RebuildDispatch.cpp:1155` の

```cpp
const bool retryable = shouldRetryWarmupFailure(*newDSP);
```

は **`RetryDisposition` ではない**。これは `RetryDisposition` の `RetryImmediate` を「即時 `submitRebuildIntent` するか否か」の **admission guard** に変換する前の eligibility check である。`bool` への縮退により `RetryBackoff` vs `RetryImmediate` の区別が失われているが、現行では `WarmupFailed` の disposition が `RetryImmediate` 固定かつ `submitRebuildIntent` が即時発行なので結果的に正しい。

将来的に `ResourceUnavailable → RetryBackoff` を配線する際は、この `bool` ではなく `RetryDisposition` をそのまま scheduler に渡す必要がある（`bool` に潰しては `Backoff` の情報が失われる）。

**D-1 判定:** `RetryDisposition` は **scheduling まで含む policy 型**であり、`bool retryable` はその下位の **admission decision** である。両者を混同してはならない。

---

## D-1-4. Recovery retry は別 authority か判定

### `settlePendingRecoveryAdmission(true)` の4箇所

`AudioEngine.RebuildDispatch.cpp` の durable recovery lease ループ（L996-1062）:

| 行 | コード | 意味 |
|----|--------|------|
| L1011 | `settlePendingRecoveryAdmission(false)` | qHandle invalid → Discarded（再試行なし） |
| L1022 | `settlePendingRecoveryAdmission(true)` | build failure → DurablePending へ戻す（次サイクル retry） |
| L1044 | `settlePendingRecoveryAdmission(true)` | warmup failure → DurablePending へ戻す（retry） + DSP破棄 |
| L1062 | `settlePendingRecoveryAdmission(false)` | success → クリア + `enqueuePublicationIntent` |

`AiDex` 10 hits / `rg` 4 production hits / `sg` 4 hits / `semble` 10 hits で一致。

### 定義

`ISRRuntimePublicationCoordinator.cpp:968` / `.h:277`:

```cpp
void RuntimeIntentCoordinator::settlePendingRecoveryAdmission(bool retry) noexcept;
```

- `retry==true` → `Building → DurablePending` へ戻す（lease の再成立、次サイクルで再 `takePendingRecoveryAdmission()`）。
- `retry==false` → `Building → Cleared`（完了 or 破棄）。
- `INV-X1-2: queue full ≠ Recovery lost` を保証する durable lease 機構（`AudioEngine.h:2661`、`ISRRuntimePublicationCoordinator.h:629` コメント）。

### BuildError retry との関係

```text
BuildError → RetryDisposition          — failure policy（「WarmupFailed は RetryImmediate」）
        ≠
Recovery lease → settle(true)          — recovery lifecycle（「DurablePending を次サイクルで再処理」）
```

両者は **直交する authority** である:

- `BuildError → RetryDisposition` は **failure 原因に対する policy**（静的・分類表ベース）。
- `settlePendingRecoveryAdmission(true)` は **Recovery admission の lifecycle**（動的・lease 状態ベース）。`BuildError` の値を参照せず、単に「今回の Recovery 処理が失敗したので lease を戻す」という状態遷移である。
- 実際、`RebuildDispatch.cpp:1022/1044` の `settle(true)` は `buildResult.error` や `warmupError` の値を一切見ず、無条件で `true` を渡す。分類表の `RetryBackoff`/`RetryImmediate` を参照していない。
- `kMaxRecoveryConsecutiveFailures=4`（L1003）のスピン防止も、BuildError 分類とは独立した **recovery 固有の backpressure** である。

**D-1 判定:** `settlePendingRecoveryAdmission(true)` は **Recovery lifecycle authority** であり、`BuildError → RetryDisposition` とは異なる。Phase D が Recovery authority を侵食してはならない。両者を同一責務に押し込んではならない。

---

## D-1-5. Authority Matrix

| # | Decision | Authority | 入力 | 出力 | retry 実行 | 現状 |
|---|----------|-----------|------|------|------------|------|
| 1 | Failure 原因 | `BuildError` (enum) | builder failure（`build()` / `validateWarmup()`） | `BuildError` (8値) | ❌（原因のみ） | 実装済み・static_assert 保証 |
| 2 | Failure 分類 | `classifyBuildError()` | `BuildError` | `FailureClassification` (4値) | ❌（分類のみ） | 実装済み・table 駆動 |
| 3 | Retry disposition | `classifyBuildError()` / `kBuildErrorDefaultTable` | `BuildError` | `RetryDisposition` (3値) | ❌（方針のみ、未配線） | 実装済みだが **telemetry only**（RebuildDispatch.cpp:1098 の log のみ） |
| 4 | Warmup state eligibility | `shouldRetryWarmupFailure()` | `DSP/IR state` (`isLoadingIR()`) | `bool` (eligibility) | 条件付きで `submitRebuildIntent` を発行 | 実装済み・**併存が正当**（§D-1-2） |
| 5 | Recovery retry lifecycle | `Recovery Admission` (`take`/`settle` lease) | `recovery state` (DurablePending/Building) | `lease` / `admission`（次サイクル再処理） | `settle(true)` で次サイクル retry | 実装済み・**別 authority**（§D-1-4） |
| 6 | Retry scheduling | **未配線** | `RetryDisposition` | `schedule`（immediate / backoff / no-retry） | 現状未実装 | **未配線** — `RetryBackoff` の timer/backoff なし |

### 真の authority duplication は存在するか

| 候補 | 判定 | 理由 |
|------|------|------|
| `classifyBuildError(WarmupFailed) → RetryImmediate` vs `shouldRetryWarmupFailure() → bool` | **併存が正当（duplication ではない）** | 前者は policy、後者は eligibility。異なる semantic domain（§D-1-2） |
| `classifyBuildError(ResourceUnavailable) → RetryBackoff` vs 実際の retry | **未配線（duplication ではない）** | 前者は定義のみ、後者は未実装。競合する第二 authority が存在しない |
| `settlePendingRecoveryAdmission(true)` vs `BuildError → RetryDisposition` | **別 authority（duplication ではない）** | 前者は recovery lifecycle、後者は failure policy。直交（§D-1-4） |

**結論:** 真の duplication は **0件**。全ての authority は異なる責務・異なる入力・異なる出力を持つ。

---

## D-1-6. 禁止事項 — 遵守

D-1 は audit-only であり、以下は一切実施していない（`git diff --stat` で 0 files changed を確認）:

```
× BuildError に shouldRetry() を追加
× 新しい shouldRetry(BuildError) を作る
× BuildContext を実装
× PrepareResult を導入
× MKLFailure 等の producer を追加
× RetryBackoffPolicy を実装
× backoff timer を実装
× Warmup retry 経路を変更
× RuntimeHealthMonitor を変更
× 1014〜1018 を変更
× Q/E semantics を変更
× retire FIFO を変更
```

`PrepareResult` は `RuntimeBuilder.h:194` コメントおよび `evidence/phase-d2-0-preparer-result-build-error-propagation-audit.md` で NO-GO と記録済みであり、現行の `BuildError + BuildResult + runtime==nullptr + classifyBuildError()` で semantic loss なしと判定されている。

---

## D-1-7. 最終判定 — 3択

### 判定: **C — 現時点では retry authority 未配線（ただし B の併存正当性を伴う）**

```text
classification                    ┌─ warmup-specific path (shouldRetryWarmupFailure → submitRebuildIntent)
    │                             │     ※ eligibility guard として併存正当
    ▼                             │
telemetry only  ──────────────────┤
(classifyBuildError → log)        │
                                  └─ recovery lease path (settle(true) → next-cycle admission)
                                        ※ recovery lifecycle として別 authority

RetryBackoff / general BuildError retry は未配線（定義のみ）
```

**理由:**

1. **A（一本化可能）ではない:** `shouldRetryWarmupFailure()` は `BuildError` と同じ decision を重複していない（§D-1-2）。一本化（削除・統合）は誤った semantic collapse になる。

2. **B（併存が正当）の要素を含むが、B 単独ではない:** `shouldRetryWarmupFailure()` と `settlePendingRecoveryAdmission()` の併存正当性は確定した（§D-1-2, §D-1-4）。しかし Phase D の中心的 authority である `BuildError → FailureClassification → RetryDisposition`（§D-1-5 #3）は **未配線** であり、B だけでは Phase D の「唯一の retry policy authority」を成立させたことにならない。

3. **C が最も正確:** `classifyBuildError()` は定義・分類・table 網羅まで完成しているが、**実行経路（RetryDisposition → Retry Admission/Scheduling）への接続が存在しない**（§D-1-1, §D-1-5 #6）。`ResourceUnavailable → RetryBackoff` のような一般的な BuildError retry は定義のみで実行されない。実際の retry は warmup-specific path と recovery lease path という **Phase D 外の authority** によって担われている。したがって Phase D の retry authority は **未配線** と判定する。

   ただし C は「何も決まっていない」ではなく、**「policy は定義済み、配線のみが未了」**という精密な C である（D-0 で table 網羅は完了、D-1 で併存境界も確定）。

### D-2 への申送り

- **D-2 の対象:** `BuildContext` / default classification contract。D-1 で `BuildContext` を今すぐ大規模導入する必要はないこと（D-0-9）、および `ResourceUnavailable → RetryBackoff` の未配線が残課題であることを前提に、**最小契約**を設計する。
- **Warmup / Recovery 経路は変更しない:** `shouldRetryWarmupFailure()` は名称・コメントで責務境界（`// eligibility guard — distinct from BuildError policy` 等）を明文化する程度に留め、振る舞いは維持する。
- **RetryBackoff の配線は D-2/D-3 で設計:** `RetryDisposition` を `bool` に潰さず scheduling まで伝搬する最小 wiring を D-2 で契約し、D-3 の Contract Test で固定する。

---

## D-1 終了条件チェック

```
[x] BuildError production producer 全経路を再確認           — 4種（3値は保険分類）/ D-0-6 + §D-1-1
[x] classifyBuildError() 全 consumer を確認                 — 1箇所（RebuildDispatch.cpp:1098、logのみ）/ §D-1-1
[x] RetryDisposition 全 consumer を確認                     — 1箇所（同上、backoff 未配線）/ §D-1-1
[x] shouldRetryWarmupFailure() の semantic 責務を確定       — eligibility guard（isLoadingIR）/ §D-1-2
[x] settlePendingRecoveryAdmission(true) の authority を確定 — recovery lifecycle（lease）/ §D-1-4
[x] ResourceUnavailable の retry 経路を確定                  — Transient/RetryBackoff だが未配線 / §D-1-1
[x] WarmupFailed の retry 経路を確定                        — Transient/RetryImmediate + shouldRetryWarmupFailure gate / §D-1-1, §D-1-2
[x] InternalError / InvalidInput / Fatal 系の no-retry を確認 — Permanent/Fatal → NoRetry → discard / §D-1-1
[x] RetryBackoff が未配線であることを確認                   — timer/backoff なし / §D-1-1, §D-1-5 #6
[x] BuildContext を実装しない根拠を維持                     — 型未定義、現行 production は不要 / D-0-9
[x] 新しい retry authority を作らない                       — 0 files changed / §D-1-6
[x] Authority Matrix を作成                                 — §D-1-5（6行）
[x] A/B/C のいずれかを ratify                               — C（B の併存正当性を伴う）/ §D-1-7
[x] production code 変更 = 0                                — git diff 0
```

**D-1 CLOSED。**

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'submitRebuildIntent|submitRecovery|settlePendingRecoveryAdmission|shouldRetryWarmupFailure|kMaxRecoveryConsecutiveFailures|RetryImmediate|RetryBackoff|NoRetry|retryable|validateWarmup|buildResult\.error|classifyBuildError' src/ --type cpp --type h
grep -n 'shouldRetryWarmupFailure' src/audioengine/AudioEngine.RebuildDispatch.cpp -A 40 -B 5
grep -n 'shouldRetryWarmupFailure' src/audioengine/AudioEngine.h -A 40 -B 5
sed -n '700,900p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/NoRetry|RetryBackoff|RetryImmediate/{print FILENAME":"FNR": "$0}' src/audioengine/RuntimeBuilder.h
fdfind -e h -e cpp . src | xargs grep -l 'isLoadingIR\|isIRLoaded\|isIRFinalized'
ag -n 'submitRebuildIntent' src/
ag -n 'settlePendingRecovery' src/
ag -n 'WarmupFailed|validateWarmup|warmupError' src/
sg run -p 'shouldRetryWarmupFailure($A)' --lang cpp src/
sg run -p 'settlePendingRecoveryAdmission($X)' --lang cpp src/
sg run -p 'RetryDisposition::$V' --lang cpp src/
sg run -p 'classifyBuildError($X)' --lang cpp src/
echo 'RuntimeBuilder.h' | fzf --filter='Runtime'

# cocoindex
ccc status
ccc grep 'shouldRetryWarmupFailure'
ccc grep 'settlePendingRecoveryAdmission'
ccc search "BuildError"

# graphify
graphify query "shouldRetryWarmupFailure"
graphify query "BuildError"
graphify path "BuildError" "RetryDisposition"

# semble
semble search "shouldRetryWarmupFailure" . --max-snippet-lines 8
semble search "settlePendingRecoveryAdmission" . --max-snippet-lines 8
semble search "RetryDisposition" . --max-snippet-lines 8
semble search "classifyBuildError" . --max-snippet-lines 8

# AiDex
aidex_query term="shouldRetryWarmupFailure" mode="exact"
aidex_query term="settlePendingRecoveryAdmission" mode="exact"
aidex_query term="BuildError" mode="contains"

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
- `src/audioengine/RuntimeBuilder.cpp:402-461` — 唯一の producer（`build()` / `validateWarmup()`）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:78-81` — `shouldRetryWarmupFailure()` 定義（`isLoadingIR()`）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1091-1165` — Path A（build failure log）+ Path B（warmup retry）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:996-1062` — Path D（durable recovery lease + `settle(true)` + `kMaxRecoveryConsecutiveFailures=4`）
- `src/audioengine/ISRRuntimePublicationCoordinator.h:277` / `.cpp:968` — `settlePendingRecoveryAdmission()` 定義
- `evidence/D101-10-Phase-D-0-Audit-Report.md` — D-0 監査報告書
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 / H.11 参照コメントの一次資料）
- `evidence/phase-d2-0-preparer-result-build-error-propagation-audit.md` — PrepareResult NO-GO 根拠
