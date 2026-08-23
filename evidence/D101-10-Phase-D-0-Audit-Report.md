# D101-10 Phase D-0 — BuildError / FailureClassification / RetryDisposition 現状監査

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D` 現状再監査（**変更禁止・audit-only**）
**Prerequisite:** 5-VI-F CLOSED 維持（`RuntimeHealthMonitor` / 1014-1018 / Q/E split / retire FIFO は非対象）
**Unique source:** `ConvoPeq.md` + 実ワークツリー（`src/audioengine/RuntimeBuilder.h/.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp` を一次資料とする）

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容（代表） | 結果要旨 |
|------|--------|------------------|----------|
| WSL | `rg` (ripgrep) | `rg -n 'BuildError\|FailureClassification\|RetryDisposition\|BuildOutcome\|kBuildErrorDefaultTable\|BuildContext' src/ --type cpp --type h` | 45 hits（RuntimeBuilder.h に定義集約、RebuildDispatch.cpp に唯一の分類呼出し） |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'BuildError::$ERR' --lang cpp src/audioengine/RuntimeBuilder.h` / `sg run -p 'classifyBuildError($X)' --lang cpp src/` | enum 8値・呼出し1箇所を構造検索で確認 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp . src` → `xargs rg -l 'BuildError'` / `fdfind RuntimeHealthMonitor` | Hit files: `RuntimeBuilder.h/.cpp`, `AudioEngine.RebuildDispatch.cpp` のみに限定 |
| WSL | `ag` (silver searcher) | `ag -n 'BuildError' src/` / `ag -c terminalPeakResident` | `rg` と一致 |
| WSL | `fzf` 0.67.0 | `echo 'RuntimeBuilder.h' \| fzf --filter='Runtime'` | パイプライン動作確認 |
| WSL | `sed` / `awk` | `sed -n '107,210p' RuntimeBuilder.h` / `awk '/BuildError::/{print FNR": "$0}' RuntimeBuilder.cpp` | enum 全8値・producer 3箇所を抽出 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認、symbol索引が `RuntimeBuilder` を正しく捕捉 | インデックス正常 |
| CLI | `cocoindex` (`ccc.exe`) | `ccc status` (90068 chunks / 1737 files) / `ccc grep 'BuildError'` / `ccc search "BuildError"` | `BuildError` は cpp 6310 chunks 中で RuntimeBuilder.h に集約 |
| CLI | `graphify` 0.9.39 | `graphify query "BuildError"` / `graphify query "classifyBuildError"` / `graphify path "BuildError" "RetryDisposition"` | `BuildError` ノード → `kBuildErrorDefaultTable` → `classifyBuildError` の有向パスを確認 |
| CLI | `semble` 0.5.3 | `semble search "classifyBuildError" .` / `semble search "BuildOutcome"` / `semble search "BuildContext"` | `BuildContext` は3コメントのみ、`BuildOutcome` は RuntimeBuilder.h のみに限定 |
| MCP | `AiDex` (index.db 26M) | `aidex_query term="BuildError" mode="contains"` → 45 matches / `aidex_query term="BuildContext"` → 3 matches（全てコメント） | WSL grep と一致 |
| sandbox | `context-mode` `ctx_execute` | `ctx_execute(language: "javascript", code: ...)` で `src/` 全ファイルを `BuildError` でフィルタ | 追加の隠れ producer なし |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep` 相当を WSL bash 経由で実行（proxy が `rtk: No such file` の場合は `grep` 直呼出しにフォールバック） | 出力差分なし |
| MCP | `headroom` | 大きな `ConvoPeq.md` 断片は `context-mode` 側で仮想化し headroom 圧縮は不要と判定 | フォールバック方針（headroom 不調時は context-mode 優先）を遵守 |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`ctx_execute` が一致）。ツール間の差異なし。

---

## D-0-1. `BuildError` の全 enum 値（8値）

`src/audioengine/RuntimeBuilder.h:107`

```cpp
enum class BuildError {
    None,                // 0 成功
    InvalidInput,        // 1 不正入力（sampleRate/blockSize 等）
    ResourceUnavailable,  // 2 std::bad_alloc
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗（保険分類）
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗（保険分類）
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗（保険分類）
    WarmupFailed,        // 6 warmup 失敗（IR未finalized）
    InternalError        // 7 その他例外・フォールバック
};
```

`static_assert(sizeof(kBuildErrorNames)/sizeof(*)) == InternalError+1` で件数検証済み（H:166）。

---

## D-0-2. `FailureClassification` の全 enum 値（4値）

`RuntimeBuilder.h:124`

```cpp
enum class FailureClassification : uint8_t {
    Permanent,       // retry 無意味（InvalidInput）
    Transient,       // retry 有効（ResourceUnavailable / WarmupFailed）
    Infrastructure,  // retry 有効・環境依存（ConvolverFailure / PrepareFailure）
    Fatal            // retry 無意味・異常終了（InternalError / MKLFailure）
};
```

コメントに `BuildError は failure 原因のみを表し、retryability は caller が推測していた`（8/14レビュー指摘、H:119）が明記。

---

## D-0-3. `RetryDisposition` の全 enum 値（3値）

`RuntimeBuilder.h:131`

```cpp
enum class RetryDisposition : uint8_t {
    NoRetry,         // retry 禁止（Permanent / Fatal）
    RetryBackoff,    // exponential backoff 付き retry（Transient / Infrastructure の一部）
    RetryImmediate   // immediate retry（WarmupFailed 等 latency-sensitive）
};
```

---

## D-0-4. `kBuildErrorDefaultTable` の全要素（8要素・網羅）

`RuntimeBuilder.h:146` — `constexpr BuildOutcome kBuildErrorDefaultTable[]`

| # | BuildError | FailureClassification | RetryDisposition | 備考 |
|---|------------|-----------------------|------------------|------|
| 0 | `None` | `Permanent` | `NoRetry` | 成功時は retry 不要（NoRetry 固定） |
| 1 | `InvalidInput` | `Permanent` | `NoRetry` | 入力不正は再試行無意味 |
| 2 | `ResourceUnavailable` | `Transient` | `RetryBackoff` | 一時的 resource exhaustion |
| 3 | `MKLFailure` | `Fatal` | `NoRetry` | **保険分類**（現行 producer なし） |
| 4 | `ConvolverFailure` | `Infrastructure` | `RetryBackoff` | **保険分類**（現行 producer なし） |
| 5 | `PrepareFailure` | `Infrastructure` | `RetryBackoff` | **保険分類**（現行 producer なし） |
| 6 | `WarmupFailed` | `Transient` | `RetryImmediate` | latency-sensitive 即時再試行 |
| 7 | `InternalError` | `Fatal` | `NoRetry` | catch-all |

`static_assert(sizeof(kBuildErrorDefaultTable)/sizeof(BuildOutcome) == InternalError+1)`（H:156）で網羅性をコンパイル時保証。`kBuildErrorNames[]` も同一件数の `static_assert`（H:166）を持つ。**第四者レビュー §21「デフォルト分類であり固定 lookup table ではない」**の警告コメント（H:121）が併記。

---

## D-0-5. `BuildOutcome` の生成箇所（1箇所のみ・ログ用途）

```cpp
// RuntimeBuilder.h:137
struct BuildOutcome {
    BuildError error = BuildError::None;
    FailureClassification classification = FailureClassification::Fatal;
    RetryDisposition retry = RetryDisposition::NoRetry;
};

// RuntimeBuilder.h:172 — 唯一の生成関数
[[nodiscard]] inline BuildOutcome classifyBuildError(BuildError error) noexcept {
    const auto idx = static_cast<size_t>(error);
    if (idx >= sizeof(kBuildErrorDefaultTable)/sizeof(BuildOutcome))
        return { BuildError::InternalError, FailureClassification::Fatal, RetryDisposition::NoRetry };
    return kBuildErrorDefaultTable[idx];
}
```

**呼出しサイト（production）:**

| ファイル | 行 | 用途 | 実際に retry を適用しているか |
|----------|----|------|-------------------------------|
| `AudioEngine.RebuildDispatch.cpp:1098` | `const auto outcome = convo::classifyBuildError(buildResult.error);` | `buildResult.runtime == nullptr` 時の `buildResult.error` を分類し `diagLog` に `classification`/`retry` を記録 | **No** — ログのみ。コメント `retry 方針の実適用（backoff 等）は D-5 RetryBackoffPolicy tuning と併せて将来拡張 — 1.8.9 実装手順`（H:1098コメント）が明記 |

`rg classifyBuildError src/` の production hit はこの1箇所のみ。他は `RuntimeBuilder.h` 定義本体とテストのみ。

`BuildResult`（build の戻り値）との区別:

```cpp
// RuntimeBuilder.h:189
struct BuildResult {
    AudioEngine::DSPCore* runtime = nullptr;
    BuildError error = BuildError::None;
    bool prepared = false;
};
```

`BuildResult` が `runtime==nullptr` で失敗を運び、`BuildOutcome` はその `BuildError` を分類した**派生型**である。

---

## D-0-6. `BuildError` の全 production producer（3箇所 + 1 warmup）

`RuntimeBuilder.cpp:build()`（H:402-449）と `validateWarmup()`（H:453-461）のみが producer。`MKLFailure`/`ConvolverFailure`/`PrepareFailure` は **producer なし（保険分類）**。

| Producer | ファイル:行 | 条件 | 生成される BuildError |
|----------|-------------|------|-----------------------|
| 早期入力検証 | `RuntimeBuilder.cpp:417` | `sampleRate <= 0 \|\| blockSize <= 0` | `InvalidInput` |
| `std::bad_alloc` catch | `RuntimeBuilder.cpp:443` | `catch (const std::bad_alloc&)` | `ResourceUnavailable` |
| `catch (...)` | `RuntimeBuilder.cpp:448` | その他全例外 | `InternalError` |
| warmup 検証 | `RuntimeBuilder.cpp:458` | `isIRLoaded() && !isIRFinalized()` | `WarmupFailed` |

**保険分類（enum+分類表+toString のみ、生成コードなし）:**

- `MKLFailure` — 将来 MKL 初期化・FFT 計画失敗で使う想定（C-2）
- `ConvolverFailure` — 将来 Convolver Build 失敗で使う想定
- `PrepareFailure` — 将来 `DSPCore::prepare()` 失敗で使う想定

これらは `evidence/phase-d2-0-preparer-result-build-error-propagation-audit.md`（H:194-201 コメント参照）で **NO-GO と audit-only クローズ済み**（PrepareResult 導入は 10+ prepare サブシステムの status 化を要する大規模侵入的変更で利益ゼロと判定）。

---

## D-0-7. `BuildError` → retry 判断を caller が直接行っている箇所

### (a) `buildResult.runtime == nullptr` 経路（`classifyBuildError` を呼ぶが retry 非適用）

`AudioEngine.RebuildDispatch.cpp:1091-1101`

```cpp
if (buildResult.runtime == nullptr) {
    const auto outcome = convo::classifyBuildError(buildResult.error);
    diagLog("[DIAG] rebuildThreadLoop: ... error=" + toString(buildResult.error)
            + " classification=" + int(outcome.classification)
            + " retry=" + int(outcome.retry) + " source=task-snapshot");
    continue; // ← retry なし（分類はログのみ）
}
```

**Authority 観点:** `BuildError` 自体に `shouldRetry()` 的な並列 authority は作られていない（禁止事項を遵守）。`BuildError → outcome.retry` の経路は存在するが、現行は**観測（log）のみ**で**実行（backoff/retry）に接続されていない**。

### (b) `warmupError != None` 経路（分類表を迂回する並列 retry 判定）

`AudioEngine.RebuildDispatch.cpp:1153-1165` と recovery 経路（H:956, H:1033）で共通パターン:

```cpp
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
if (warmupError != convo::BuildError::None) {
    const bool retryable = shouldRetryWarmupFailure(*newDSP); // ← 並列 authority
    diagLog("[DIAG] ... warmup failed error=" + toString(warmupError)
            + " retryable=" + int(retryable) + ...);
    if (retryable)
        submitRebuildIntent(RebuildKind::Structural, RebuildTelemetryReason::RebuildThreadWarmupRetry, ...);
    continue;
}
```

`shouldRetryWarmupFailure()` は `RebuildDispatch.cpp` 内のローカル述語（IR状態に基づく `isIRLoaded`/`isIRFinalized`/`isLoadingIR` 判定）であり、`kBuildErrorDefaultTable[WarmupFailed] = {Transient, RetryImmediate}` とは**独立した第二の retry authority**である。`rg shouldRetry` でこの1関数（+ recovery/durable recovery の同型2箇所）のみが hit。

**監査所見:** `WarmupFailed` に対して **2つの retry 経路が併存**している（`classifyBuildError(WarmupFailed) → RetryImmediate` と `shouldRetryWarmupFailure() → bool`）。いずれも `RetryImmediate` 相当の即時 `submitRebuildIntent` に収束するが、**authority の二重化**として D-1 で一本化の要否を判断すべき。

### (c) `recovery` / `durable recovery` 経路

- `RebuildDispatch.cpp:956` (`recoveryWarmup != None`) — `continue` のみ（retry なし、quarantined DSP は破棄）
- `RebuildDispatch.cpp:1033` (`durable recoveryWarmup != None`) — `settlePendingRecoveryAdmission(true)` で DurablePending へ戻す（lease方式、次サイクルで再 take — retry を構造的に保証）+ `kMaxRecoveryConsecutiveFailures=4` でスピン防止

これらは `WarmupFailed` の retry を **durable lease で保証**しており、`shouldRetryWarmupFailure` の判定を経由しない（常に `settle(true)`）。

### 総括 — retry authority の重複度

| Authority | 位置 | 対象 BuildError | 適用状況 |
|-----------|------|-----------------|----------|
| `classifyBuildError() → BuildOutcome.retry` | `RuntimeBuilder.h:172` | 全8値 | **生成されるが適用されない**（ログのみ） |
| `shouldRetryWarmupFailure()` | `RebuildDispatch.cpp:1154` 付近 | `WarmupFailed` のみ | **適用される**（即時 retry） |
| `settlePendingRecoveryAdmission(true)` | `RebuildDispatch.cpp:1033` 付近 | `WarmupFailed` / `ResourceUnavailable` 的失敗（durable recovery の build/warmup 失敗） | **適用される**（次サイクル retry）だが `BuildError` 分類を参照しない |

**結論:** `shouldRetry(BuildError)` のような `BuildError` を直接引数に取る汎用 retry 関数は存在しない（D-1 禁止事項を遵守）。ただし `WarmupFailed` に限って `classifyBuildError` と `shouldRetryWarmupFailure` が**二重 authority**を形成している。

---

## D-0-8. `FailureClassification` / `RetryDisposition` が既に使用されている箇所

`rg RetryDisposition|FailureClassification src/` の production hit は以下に限定:

| ファイル | 箇所 | 用途 |
|----------|------|------|
| `RuntimeBuilder.h:124-135` | enum 定義本体 | 定義 |
| `RuntimeBuilder.h:146-154` | `kBuildErrorDefaultTable` 8要素 | デフォルト分類定義 |
| `RuntimeBuilder.h:137-140` | `BuildOutcome` 3フィールド | 型定義 |
| `RuntimeBuilder.h:172-177` | `classifyBuildError()` | 分類解決 |
| `AudioEngine.RebuildDispatch.cpp:1098` | `classifyBuildError(buildResult.error)` 呼出し + `outcome.classification`/`outcome.retry` のログ出力 | **唯一の使用箇所**（観測のみ） |

他に `RetryBackoff` / `RetryImmediate` / `NoRetry` を switch して backoff を実装している箇所は **0件**。`FailureClassification` を分岐に使う箇所も **0件**。すなわち **Phase D の分類基盤は「定義済みだが未配線」**である。

---

## D-0-9. `BuildContext` の存在・未実装範囲

`aidex_query BuildContext` → 3 matches、全て `RuntimeBuilder.h` の**コメントのみ**。`struct BuildContext` / `class BuildContext` / `using BuildContext` の定義は存在しない（`rg 'struct BuildContext'` → 0件、WQL grep 0件）。

該当コメント（H:122-123, H:171）:

```cpp
//   実装は BuildError + BuildContext → FailureClassification → RetryDisposition の順で解決する
//   （BuildContext は将来拡張。現行はデフォルト表のみ — §1.8.12 未解決課題）。

// ★ §1.8.5.2 / H.11.2: BuildError → デフォルト分類の解決。
//   将来的に BuildContext（一時的 resource exhaustion / persistent config）で上書きする。
```

**監査所見:**

- `BuildContext` は **型として未実装**。`§1.8.12 未解決課題` として明示的に defer されている。
- 現行の `classifyBuildError(BuildError)` は `BuildError` 単独で `kBuildErrorDefaultTable` を引く**デフォルト分類**であり、将来 `classifyBuildError(BuildError, BuildContext)` への拡張が想定されている。
- `BuildContext` が導入される場合に想定される分岐（コメント・レビュー指摘からの推定）:
  - `ResourceUnavailable` が一時的 exhaustion なのか persistent な config 欠落なのか
  - `MKLFailure` / `ConvolverFailure` が再試行で回復可能か
  - しかし現行の **production code は context-sensitive classification を必要としていない**（D-0-6 の通り、実際に発生する failure は `InvalidInput`/`ResourceUnavailable`/`InternalError`/`WarmupFailed` の4種のみで、いずれも `BuildError` 単独で分類が一意に定まる）

**結論:** D-2 の指示通り、**BuildContext を今すぐ大規模導入する必要はない**。Phase D の最小完成形は default classification table の authority 化に留めるのが適切。

---

## D-0-10. Build failure が `InternalError` 等へ潰されている箇所

| 箇所 | コード | 潰し方 | 分類への影響 |
|------|--------|--------|--------------|
| `RuntimeBuilder.cpp:448` | `catch (...) { result.error = BuildError::InternalError; }` | 未知例外を全て `InternalError` に丸める | `Fatal/NoRetry` に分類（`kBuildErrorDefaultTable[InternalError]`） |
| `RuntimeBuilder.h:175-176` | `if (idx >= tableSize) return {InternalError, Fatal, NoRetry};` | 不正な `BuildError` 値（将来 enum 拡張時の未初期化値等）を `InternalError` に丸める | 同上 |
| `RuntimeBuilder.h:184-186` | `if (idx >= namesSize) return "Unknown";` | 不正値の文字列表現を `Unknown` に丸める（分類には影響なし） | — |
| 将来の `MKLFailure`/`ConvolverFailure`/`PrepareFailure` | enum は存在するが producer がないため、仮に将来 `catch (...)` に流れても `InternalError` に丸められる | `§1.8.5.3` の「一時的 vs InternalError 丸め」不一致は設計レベルで休眠（`evidence/phase-d2-0` で audit-only クローズ） | 休眠 |

`InternalError` への丸めは **意図的**であり、未知 failure を `Fatal/NoRetry` として安全側に倒す設計である（retry しない）。丸め自体は問題ないが、D-2 で `MKLFailure` 等の保険分類が将来 producer を持つ場合に `catch (...)` から正しい `BuildError` へ分岐する設計を要するかは検討対象。

---

## D-0 総括 — REPAIR_PLAN2-dash2 §1.8 との差分

> REPAIR_PLAN2-dash2 文書自体は `doc/` / `evidence/` に単独ファイルとして存在しないが、`ConvoPeq.md` および `RuntimeBuilder.h` のコメントが §1.8 / H.11 / §1.8.5.x / §1.8.8.x / §1.8.10.3 / §1.8.12 を逐条参照しており、これらを契約の一次資料とする。

| §1.8 想定 | 現行実装 | 差分 | D-1/D-2 での対応 |
|-----------|----------|------|-------------------|
| `BuildError` 8値 + `FailureClassification` 4値 + `RetryDisposition` 3値 + `BuildOutcome` + `kBuildErrorDefaultTable` + `classifyBuildError()` | **実装済み**（H:107-186、static_assert 含む） | なし（**二重実装禁止**を遵守） | D-1 では再定義しない。既存 table を authority として追認するのみ |
| `BuildError → FailureClassification → RetryDisposition` の3段パイプライン | 型・table は存在するが **配線は logs のみ**（RebuildDispatch.cpp:1098） | **配線欠落**（retry が実行されない） | D-1 で authority を確定し、D-3/D-5 で配線を最小実装 |
| `BuildError + BuildContext → Classification → Retry` | `BuildContext` はコメントのみ・未実装（H:122） | **未実装（意図的 defer）** | D-2 で「大規模導入せず、default table の authority 化に留める」を確定 |
| `PrepareResult` 導入（10+ prepare サブシステムの status 化） | **NO-GO**（H:194-201、evidence/phase-d2-0 で audit-only クローズ） | 導入しない | D-2 で再確認のみ |
| `WarmupFailed` の retry | `shouldRetryWarmupFailure()` と `classifyBuildError(WarmupFailed)` が**二重 authority** | 要整理 | D-1 で一本化の要否を決定（禁止: 新たな `shouldRetry(BuildError)` を作ること） |
| `InternalError` 丸め | `catch (...)` → `InternalError`（Fatal/NoRetry）に丸め | 設計通りだが、将来の保険分類との整合は要監視 | D-2 で軽微な改善の要否を検討 |

### 既存実装済みのものを再実装しないこと — 遵守状況

- `BuildError` / `FailureClassification` / `RetryDisposition` / `BuildOutcome` / `kBuildErrorDefaultTable` / `kBuildErrorNames` / `classifyBuildError()` / `toString()` は **全て実装済み**。D-1 以降で再定義・再生成してはならない。
- 新たな `shouldRetry(BuildError)` 的な並列 authority を作ることは **禁止**（D-1 指示）。

---

## D-0 判定 — 監査結果サマリ

```
D-0-1  BuildError 全8値                         PASS（static_assert 済み）
D-0-2  FailureClassification 全4値               PASS
D-0-3  RetryDisposition 全3値                    PASS
D-0-4  kBuildErrorDefaultTable 全8要素           PASS（網羅・同期保証済み）
D-0-5  BuildOutcome 生成箇所                    PASS（1箇所・ログのみ、retry未配線）
D-0-6  BuildError 全production producer         PASS（4種、3値は保険分類で producer なし）
D-0-7  BuildError → retry 直接判断箇所          PASS（2箇所、WarmupFailed で二重 authority 検出）
D-0-8  FailureClassification/RetryDisposition使用 PASS（1箇所のみ、backoff 未実装）
D-0-9  BuildContext 未実装範囲                  PASS（型未定義、コメント3箇所のみ、D-2 defer 妥当）
D-0-10 InternalError 等への潰し                 PASS（2箇所、意図的 Fatal/NoRetry 丸め）

総合: D-0 AUDIT PASS — コード変更なしで D-1 へ進行可
```

**Phase C / R4 について:** 指示通り実装しない。現行コードでは C-0 監査により R4 の FIFO 強化は `INV-EPOCH-1/2` が担保するため NO-GO と整理済み。Phase D の監査が `retire FIFO` / `pendingReclaimHandles_` / `currentWorld_` 等に触れていないことを `rg` / `graphify` で確認済み。

---

## 次工程への申送り（D-1 への入力）

1. **D-1 retry policy の Authority を確定:** `BuildError` 自体に retry を持たせない方針は現行コメント（H:119）と一致。`BuildError → FailureClassification → RetryDisposition` の3段を唯一の authority とし、`shouldRetryWarmupFailure()` の扱い（統合/併存/廃止）を決定する。
2. **D-2 BuildContext 問題:** 現行 production は context-sensitive classification を必要としないため、**BuildContext の大規模導入は見送り**、default table の authority 化に留める案を第一候補とする。`MKLFailure` 等の保険分類は将来 producer 出現時に最小 wiring（status 伝播 + failure check 強化 + telemetry）で対応する方針（H:194-201）を維持。
3. **D-3 Contract Test:** `None`/`InvalidInput`/`ResourceUnavailable`/`MKLFailure`/`ConvolverFailure`/`PrepareFailure`/`WarmupFailed`/`InternalError` の8値×分類×retry の契約表を独立した小さなテスト（`RuntimeHealthMonitorTierTests` 型の契約テスト）として追加する準備が整っている。

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'BuildError|FailureClassification|RetryDisposition|BuildOutcome|kBuildErrorDefaultTable|BuildContext' src/ --type cpp --type h
grep -rn 'enum class BuildError\|enum class FailureClassification\|enum class RetryDisposition' src/ --include='*.h' --include='*.cpp'
sed -n '107,210p' src/audioengine/RuntimeBuilder.h
awk '/BuildError::/{print FNR": "$0}' src/audioengine/RuntimeBuilder.cpp
fdfind -e h -e cpp . src | xargs rg -l 'BuildError'
ag -n 'BuildError' src/
echo 'RuntimeBuilder.h' | fzf --filter='Runtime'
sg run -p 'BuildError::$ERR' --lang cpp src/audioengine/RuntimeBuilder.h
sg run -p 'classifyBuildError($X)' --lang cpp src/

# cocoindex
ccc status
ccc grep 'BuildError'
ccc search "BuildError"

# graphify
graphify query "BuildError"
graphify query "classifyBuildError"
graphify path "BuildError" "RetryDisposition"

# semble
semble search "classifyBuildError" . --max-snippet-lines 8
semble search "BuildOutcome" . --max-snippet-lines 8
semble search "BuildContext" . --max-snippet-lines 8

# AiDex
aidex_query term="BuildError" mode="contains"
aidex_query term="BuildContext" mode="contains"

# context-mode sandbox
ctx_execute(language: "javascript", code: "fs.readdirSync('src/audioengine').filter(f=>fs.readFileSync(...).includes('BuildError'))")

# serena
# .serena/project.yml 確認 + find_symbol (RuntimeBuilder)

# RTK (WSL版)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "BuildError" src/'
```

## 参照

- `src/audioengine/RuntimeBuilder.h:107-210` — BuildError 家族の定義集約
- `src/audioengine/RuntimeBuilder.cpp:402-461` — 唯一の producer
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1091-1165` — 唯一の `classifyBuildError` 呼出し + 並列 `shouldRetryWarmupFailure` 経路
- `evidence/phase-d2-0-preparer-result-build-error-propagation-audit.md` — PrepareResult NO-GO の根拠
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 / H.11 参照コメントの一次資料）
