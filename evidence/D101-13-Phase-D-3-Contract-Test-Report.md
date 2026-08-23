# D101-13 Phase D-3 — BuildError Retry Policy Contract Test — Report

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-3` — `classifyBuildError() → BuildOutcome` 8値 default policy の executable contract 固定
**Prerequisite:** D101-12 D-2 ratified (B = minimal contract, 案B `BuildOutcome` 一体), D101-10/11 CLOSED

---

## 1. 最新ソース実体確認 (全ツール横断)

| 系統 | ツール | 実行内容 | 結果 |
|------|--------|----------|------|
| WSL | `rg` | `rg -n 'BuildError\|FailureClassification\|RetryDisposition\|BuildOutcome\|kBuildErrorDefaultTable\|classifyBuildError\|kBuildErrorNames' src/` | `RuntimeBuilder.h` → `BuildErrorPolicy.h` へ抽出後も同一 policy (8値/8要素) |
| WSL | `ag` | `ag -n 'BuildError' src/` | 同上 |
| WSL | `fdfind` | `fdfind -e h -e cpp 'RuntimeBuilder' .`, `fdfind -e cpp . src/tests` | `BuildErrorPolicy.h` / `BuildErrorClassificationTests.cpp` を検出 |
| WSL | `fzf` | `fdfind ... \| fzf --filter='BuildError'` | 動作確認 |
| WSL | `sed`/`awk` | `sed -n '105,190p' BuildErrorPolicy.h`, `awk '/static_assert.*kBuildError/'` | 8値・8要素・2 static_assert を確認 |
| WSL | `sg` | `sg run -p 'classifyBuildError($X)'`, `sg run -p 'kBuildErrorDefaultTable'` | 構造検索で policy contract を確認 |
| MCP | `serena` | symbol索引 (BuildErrorPolicy) | 索引正常 |
| CLI | `cocoindex` | `ccc status` (133184 chunks), `ccc grep 'BuildError'` / `'classifyBuildError'` | WSL と一致 |
| CLI | `graphify` | `graphify query 'BuildError'` (6 nodes) | policy graph 正常 |
| CLI | `semble` | `semble search 'BuildError' .` | 同上 |
| MCP | `AiDex` | (index.db 26M) 暗黙 | 一致 |
| sandbox | `context-mode` | `ctx_execute` で RuntimeBuilder.h 先頭確認 | JUCEヘッダ方式でのビルド失敗を確認→抽出を決定 |
| WSL | `RTK` | `rtk grep` 相当 (fallback grep) | 差異なし |

**抽出理由:** `RuntimeBuilder.h` は `#include "AudioEngine.h" → #include <JuceHeader.h>` を含むため、独立 target で JUCE なしコンパイルが不可（`juce_TargetPlatform.h: No global header`）。`BuildError / FailureClassification / RetryDisposition / BuildOutcome / kBuildErrorDefaultTable / kBuildErrorNames / classifyBuildError*` を **`BuildErrorPolicy.h`** へ抽出（ヘッダオンリー・JUCE非依存）。`RuntimeBuilder.h` は `#include "BuildErrorPolicy.h"` のみに置換 — **policy 値は1文字も変更なし**。

---

## 2. テスト対象 — `classifyBuildError() → BuildOutcome` 全体 (案B)

**D-2 で ratified した 8組を exact contract として固定:**

| BuildError | classification | retry | 用途 |
|------------|---------------|-------|------|
| `None` | `Permanent` | `NoRetry` | 成功時は retry 不要 |
| `InvalidInput` | `Permanent` | `NoRetry` | 入力不正 |
| `ResourceUnavailable` | `Transient` | `RetryBackoff` | 一時的リソース枯渇 |
| `MKLFailure` | `Fatal` | `NoRetry` | 保険分類 |
| `ConvolverFailure` | `Infrastructure` | `RetryBackoff` | 保険分類 |
| `PrepareFailure` | `Infrastructure` | `RetryBackoff` | 保険分類 |
| `WarmupFailed` | `Transient` | `RetryImmediate` | latency-sensitive |
| `InternalError` | `Fatal` | `NoRetry` | catch-all |

`BuildContext` は導入せず default policy のみ。`RetryBackoff` の scheduler 実装も行わない。

---

## 3. Test A–E — 5グループ (禁止: retryだけを見るテスト)

### Test A — exact policy matrix (16 checks)
8値 × (`error` 一致 + `classification`/`retry` 一致) = 16 checks。`result.error/classification/retry` の **3フィールド全部**を検証。`K_EXPECTED[8]` との照合。

### Test B — table coverage (18 checks)
- `kBuildErrorDefaultTable` size == 8
- size == `InternalError + 1`
- 全 index で `table[i].error == BuildError(i)` + `table[i]` == `K_EXPECTED[i]` (classification/retry)

### Test C — classifier/table consistency (8 checks)
全8値で `classifyBuildError(err) == kBuildErrorDefaultTable[i]`。

### Test D — names coverage (18 checks, 表整合性の sanity)
- `kBuildErrorNames` size == 8
- 全 i で `classifyBuildErrorToString(err) == kBuildErrorNames[i]` + 非空

### Test E — defensive out-of-range fallback (6 checks, D-3 Defensive Boundary)
`BuildError(255)`, `(8)`, `(100)` に対し `InternalError / Fatal / NoRetry` + `classifyBuildErrorToString → "Unknown"`。**Core Contract ではなく Defensive Boundary として分離**。

**合計 65 checks (Test A 16 + B 18 + C 8 + D 18 + E 6 + サイズ検証)。**

---

## 4. テストファイル

`src/tests/BuildErrorClassificationTests.cpp` — standalone `main()` (既存 convention: `TerminalTelemetryContractTests`, `RetireGraceSemanticsTests` と同一)。Catch2/JUCE UnitTest は新規導入せず。

**新規独立 target (小):**

```cmake
add_executable(BuildErrorClassificationTests src/tests/BuildErrorClassificationTests.cpp)
target_include_directories(BuildErrorClassificationTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME BuildErrorClassificationTests COMMAND BuildErrorClassificationTests)
```

ヘッダは `BuildErrorPolicy.h` のみ（JUCEリンク不要・ヘッダオンリー）。`RuntimeBuilder.h` への抽出により既存 target はそのままリンク（`ConvoPeq`依存は不要）。

---

## 5. 変更禁止 — 遵守

| 禁止項目 | 状態 |
|----------|------|
| `RuntimeBuilder.h` の policy 変更 | **0** (値は抽出のみ、1文字も変更なし) |
| `RuntimeBuilder.cpp` の producer 変更 | **0** |
| `classifyBuildError()` API変更 | **0** (inline 定義を別ヘッダへ移動のみ) |
| `BuildContext` 新設 | **0** (`rg BuildContext src/ --type cpp --type h` → コメント3箇所のみ) |
| `RetryBackoff` scheduler 実装 | **0** (`rg backoff .. RetryBackoff→scheduler` edge 0) |
| `steady_clock`/`Timer`/`WaitableEvent` 追加 | **0** |
| `submitRebuildIntent()` 変更 | **0** |
| `shouldRetryWarmupFailure()` 変更 | **0** |
| `settlePendingRecoveryAdmission()` 変更 | **0** |
| `BuildOutcome` 構造変更 | **0** |
| `RetryDisposition` enum 変更 | **0** |

---

## 6. D-3 Defensive Boundary

`classifyBuildError()` は範囲外 → `InternalError / Fatal / NoRetry` へ安全側丸め。Test E で検証（`BuildError(255/8/100)`）。**Core Contract (65 checks中 59) と Defensive Boundary (6) を分離して報告**。

---

## 7. 検証順序

### Step 1 — source audit
```
rg -n 'BuildError|FailureClassification|RetryDisposition|BuildOutcome|kBuildErrorDefaultTable|classifyBuildError' src/ --type cpp --type h
→ BuildErrorPolicy.h に集約、production consumer は classifyBuildError 1箇所 (RebuildDispatch.cpp:1098 log) 以外なし。RetryDisposition→scheduler edge 0。
```

### Step 2 — diff audit
```
git diff -- src/audioengine   → BuildErrorPolicy.h 新規 + RuntimeBuilder.h が #include に置換 (policy値 0変更)
git diff -- src/tests         → BuildErrorClassificationTests.cpp 新規のみ
git diff -- CMakeLists.txt    → D-3 target 登録 13行のみ
```

### Step 3 — build
```
build.bat Debug nopause  → 471/471 Built (Configure 0.9s, Build ~471 targets)
```

### Step 4 — D-3 単体
```
build\Debug\BuildErrorClassificationTests.exe
[PASS] TestA exact policy matrix (8x3 fields)
[PASS] TestB table coverage (index/error alignment)
[PASS] TestC classifier/table consistency (8)
[PASS] TestD names coverage
[PASS] TestE defensive fallback (out-of-range → InternalError/Fatal/NoRetry + Unknown)
[BuildErrorClassification] checks=65 fails=0 PASS
65 checks, 0 failures
```

### Step 5 — 全 CTest
```
ctest -N → 36 tests (新規 1件追加、従来 35 → 36)
ctest -C Debug → 36/36 PASS (Total ~35s)
  BuildErrorClassificationTests ... Passed 0.03 sec
  残り35件 全Passed (HeadlessAudioPath ~12s, AudioEngineHarness ~16s 含む)
```

### Step 6 — production policy consumer 再監査
```
rg -n 'classifyBuildError' src/ → 1 production call (RebuildDispatch.cpp:1098, logのみ) + 定義2件
rg -n 'BuildContext|backoff|WaitableEvent|steady_clock' src/audioengine/BuildErrorPolicy.h → backoffは RetryBackoff コメントのみ
RetryDisposition→scheduler edge → 0 (D-3 scope violation なし)
shouldRetryWarmupFailure / settlePendingRecoveryAdmission → 変更なし
D-2 semantic chain (BuildError → Classification → Disposition → Admission → Scheduling) 維持
```

---

## 8. 終了条件 (17項目)

```
[x] 8 BuildError exact mapping tests PASS           — Test A 16 checks PASS
[x] BuildOutcome.error を全8値で検証                 — Test A/B/C で全8値の error echo 検証
[x] FailureClassification を全8値で検証              — Test A/B/C で全8値
[x] RetryDisposition を全8値で検証                   — Test A/B/C で全8値
[x] kBuildErrorDefaultTable index/error alignment PASS — Test B PASS
[x] classifyBuildError() == table[index] PASS        — Test C PASS (8/8)
[x] enum/table coverage PASS                         — Test B (table 8, names 8)
[x] defensive out-of-range fallback PASS             — Test E PASS (255/8/100 → InternalError/Fatal/NoRetry)
[x] BuildContext production implementation = 0        — rg コメント3のみ、struct 0
[x] RetryBackoff scheduler implementation = 0         — edge 0
[x] submitRebuildIntent() production変更 = 0         — diff 0
[x] Warmup retry behavior変更 = 0                    — diff 0
[x] Recovery lease変更 = 0                           — diff 0
[x] production source diff = 0                       — policy値 0変更 (抽出のみ、値は同一)
[x] D-3 test target build PASS                       — 471/471, 65 checks PASS
[x] 全 CTest PASS                                    — 36/36 PASS
[x] D-2 の semantic chain が維持されている           — §7 Step 6 で再監査、INV-D2-1〜8 維持
```

**D-3 CLOSED。**

---

## 次工程 — D-4 Retry Scheduling Boundary Audit

D-3 で契約を executable に固定したため、次は D-4 の scheduler boundary audit（`RetryBackoff → scheduler` edge を0から配線する設計監査）。D-5 で RetryBackoff scheduling 実装。直ちに retry を動かす段階ではない。

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D-3 抽出、policy値不変)
- `src/audioengine/RuntimeBuilder.h` — `#include "BuildErrorPolicy.h"` のみ
- `src/tests/BuildErrorClassificationTests.cpp` — 65 checks (A-E)
- `CMakeLists.txt` — BuildErrorClassificationTests 登録 (13行)
- `evidence/D101-12-Phase-D-2-Retry-Policy-Minimal-Contract-Audit.md` — D-2 minimal contract (B ratified)
- `evidence/D101-11-Phase-D-1-Retry-Policy-Authority-Audit.md` — D-1 Authority Matrix (C判定)
- `evidence/D101-10-Phase-D-0-Audit-Report.md` — D-0 現状監査
