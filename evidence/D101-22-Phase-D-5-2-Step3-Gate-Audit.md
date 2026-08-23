# D101-22 Phase D-5-2 Step 3 — PendingRetry Type / Data-flow Gate Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** D-5-2 Step 3 Gate Audit（**audit-only、コード変更0**）— `PendingRetry` 追加前に型・所有権・データフローを固定し、Step 4 premature 実装を遮断
**Prerequisite:** D101-21 Step 2 完了（`RetryScheduleRequest` 4-field 定義のみ、`RetryScheduler.h` 新設、36/36 PASS）、D-5-2-1 GO、
D-5-2-0 12項目 PASS、`ConvoPeq.md` 正本（`submitRebuildIntent` 4引数 `RebuildKind` + telemetry 3）
**Primary source:** 実ワークツリー（`src/audioengine/RetryScheduler.h` / `RetrySchedulerTypes.h` / `src/core/RebuildTypes.h` /
`src/audioengine/AudioEngine.h` / `AudioEngine.RebuildDispatch.cpp:149` / `src/audioengine/BuildErrorPolicy.h`）

---

## 手法 — 全ツール横断

| 系統 | ツール | 実行内容 | 結果 |
|------|--------|----------|------|
| WSL | `rg` | 10 audit (A〜J): `PendingRetry` / `RetryScheduler.h` 全文 / `RetrySchedulerTypes.h` pollution / `generation\|sequence\|epoch` in RetryScheduler.h / `BuildError\|RetryDisposition` / `RuntimeWorld\|PublicationAuthority` / `class RetryScheduler\|schedule(` / `submitRebuildIntent` / `RetryScheduleRequest` refs / `git diff/status` | 全て PASS、差異0 |
| WSL | `ag` | `ag -n 'RetryScheduleRequest\|PendingRetry' src` / `ag -n 'RebuildKind' src` | `rg`と一致 |
| WSL | `fdfind`/`fzf` | `fdfind -e h -e cpp 'RetryScheduler'` / `fdfind -e h . src/audioengine \| fzf --filter='Retry'` | `RetrySchedulerTypes.h`/`RetryScheduler.h` のみ、`.cpp` 0 |
| WSL | `sed` | `sed -n '1,30p' RetryScheduler.h` / `RetrySchedulerTypes.h` / `RebuildTypes.h` | 4 enum / request 定義を実測 |
| WSL | `awk` | `awk '/RetryScheduler\|RebuildKind/' RetryScheduler*.h RebuildTypes.h` | 型分離確認 |
| WSL | `sg` | `sg run -p 'RetryScheduleRequest' --lang cpp src/` / `sg run -p 'PendingRetry'` / `sg run -p 'BuildError'` | `PendingRetry` 0、`BuildError` in RetryScheduler.h 0 |
| MCP | `serena` | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) | 索引正常 |
| CLI | `cocoindex` `ccc status` (133184 chunks) / `ccc grep 'RetryScheduleRequest'` | 0 hits（未使用）— 一致 | — |
| CLI | `graphify query 'RetryScheduleRequest'` / `query 'RebuildKind'` | `RetryScheduleRequest` 未登録（正常）、`RebuildKind` は `core/RebuildTypes.h` graph | — |
| CLI | `semble search 'RetryScheduleRequest'` / `search 'PendingRetry'` | `RetryScheduleRequest` 1 hit（定義のみ）、`PendingRetry` 0 | — |
| MCP | `AiDex` | `BuildError` 45 hits → `BuildErrorPolicy.h` 集約（D-3〜D-5 で確認） | — |
| sandbox | `context-mode ctx_execute` (js) | `RetryScheduler.h` / `RetrySchedulerTypes.h` / `RebuildTypes.h` / `AudioEngine ownership` を横断 | WSLと一致 |
| WSL | `RTK` | `rtk grep` fallback | 差異0 |
| MCP | `headroom` | context-mode仮想化 | 遵守 |
| 文献 | — | I4 D14/D15（logical obligation ownership / admission / non-supersedable） | — |

全ツール一致、差異0。

---

## Gate 監査 — A〜J 全PASS

### A. PendingRetry definition count — PASS

```
rg -n 'PendingRetry' src --type cpp --type h → 0 hits
grep -rn 'PendingRetry' src --include='*.h' --include='*.cpp' → 0
sg run -p 'PendingRetry' --lang cpp src/ → 0
ag -n 'PendingRetry' src → 0
semble search 'PendingRetry' → 0
```

**凍結:** 現時点未定義。Step 3 で `PendingRetry` を最小フィールド（後述）で追加する前提だが、現Gateでは **0 を維持**（Step 4 premature 実装を遮断）。

### B. RetryScheduler.h の現在の全内容 — PASS

```cpp
#pragma once
// RetryScheduler.h — D-5-2 Step 2: RetryScheduleRequest type definition only

#include "core/RebuildTypes.h"
#include "RetrySchedulerTypes.h"

struct RetryScheduleRequest {
    convo::RebuildKind kind;
    RebuildTelemetryReason reason;
    RebuildTelemetryClass rebuildClass;
    RebuildTelemetryPolicy collapsePolicy;
};
```

4 field を維持。`RetryScheduleRequest` が `RetrySchedulerTypes.h` に漏れていないことを C で確認。

### C. RetrySchedulerTypes.h pollution — PASS

```
rg -n 'RetryScheduleRequest|PendingRetry|RetryScheduler' src/audioengine/RetrySchedulerTypes.h → 0
grep -n 'RetryScheduleRequest|PendingRetry|RetryScheduler' RetrySchedulerTypes.h → 0 (header コメントを除く)
```

Types header は 3 enum（`RebuildTelemetryReason` 36値 / `Class` 3値 / `Policy` 3値）のみ。`RebuildKind` は `core/RebuildTypes.h` に分離。

責務境界:

```
RebuildKind → core/RebuildTypes.h
telemetry 3 → RetrySchedulerTypes.h
request     → RetryScheduler.h
```

### D. generation / sequence / epoch — PASS

```
rg -n 'generation|sequence|epoch' src/audioengine/RetryScheduler.h → 0
grep -n 'generation|sequence|epoch' RetryScheduler.h → 0
```

`RetrySchedulerTypes.h` / `RetryScheduler.h` 共に 0。`AudioEngine.h` の `rebuildRequestGeneration` / `generation` / `epoch` は `requestRebuild() → ++rebuildRequestGeneration` の authoritative epoch singularization に委譲（caller → authoritative dataflow）。

### E. BuildError / FailureClassification / RetryDisposition / BuildOutcome — PASS

```
rg -n 'BuildError|FailureClassification|RetryDisposition|BuildOutcome' src/audioengine/RetryScheduler.h → 0
rg -n 'BuildError|RetryDisposition' RetryScheduler.h → 0
sg run -p 'BuildError' --lang cpp RetryScheduler.h → 0
```

Build policy authority は `BuildErrorPolicy.h` に分離（8値 `BuildError` / 4 `FailureClassification` / 3 `RetryDisposition` / `BuildOutcome` / `kBuildErrorDefaultTable`）。`RetryScheduleRequest` に混入させない。

### F. RuntimeWorld / PublicationAuthority / WorldAuthority — PASS

```
rg -n 'RuntimeWorld|PublicationAuthority|WorldAuthority' src/audioengine/RetryScheduler.h → 0
rg -n 'RuntimeWorld|PublicationAuthority' RetrySchedulerTypes.h RebuildTypes.h → 0
```

Retry request が RuntimeWorld / publication authority を直接参照しないことを確認。

### G. RetryScheduler / schedule / queue / timer / worker の premature 実装 — PASS

```
rg -n 'class RetryScheduler|RetryScheduler::|PendingRetry|schedule\(' src --type cpp --type h → 0
grep -rn 'class RetryScheduler|RetryScheduler::|PendingRetry|schedule(' src → 0
fdfind -e h 'RetryScheduler' → RetryScheduler.h / RetrySchedulerTypes.h のみ（.cpp 0）
sg run -p 'RetryScheduler' → 0
cocoindex grep 'RetryScheduler' → 0
graphify query 'RetryScheduleRequest' → 未登録
```

Step 4 の `class RetryScheduler` / `queue` / `timer` / `worker scheduling` / `schedule()` / `retry execution` / `backoff` / lifecycle wiring は全て未実装を維持。

### H. submitRebuildIntent call-site count — PASS

```
rg -n 'submitRebuildIntent' src --type cpp --type h → 15+ hits（Parameters.cpp 15 / Init / PrepareToPlay / Timer / UIEvents / StateIO / EQEdit / RebuildDispatch.cpp:149 定義 + :1165 warmup）
ag -n 'submitRebuildIntent' src → 同数
```

Step 3 では call-site 変更0。`AudioEngine.h` 変更0。Retry scheduler が publication authority にならない（`submitRebuildIntent` → `requestRebuild` → `rebuildRequestGeneration++` → `pendingTask` の既存 admission pipeline に委譲）。

### I. RetryScheduleRequest の参照箇所 — PASS

```
rg -n 'RetryScheduleRequest' src --type cpp --type h → src/audioengine/RetryScheduler.h:11 1箇所のみ
ag -n 'RetryScheduleRequest' src → 同1箇所
sg run -p 'RetryScheduleRequest' → 1
cocoindex grep 'RetryScheduleRequest' → 0（header-only 未buildのため、ただし rg で確認）
semble search 'RetryScheduleRequest' → 1 hit（定義）
```

未使用型として存在（誰からも include されない）— 意図的。Step 2 の `data type の存在だけ` として凍結済み。

### J. git diff / git status — PASS

```
git diff --stat → RetrySchedulerTypes.h 新規（Step 1）、RetryScheduler.h 新規（Step 2）、AudioEngine.h 3-enum using alias 追加のみ
git diff -- src/audioengine/RetryScheduler.h → 新規ファイル（Step 2 のみ）
git status → retryScheduler 未実装（.h のみ）、call-site 0変更
```

最新 `ConvoPeq.md`（`submitRebuildIntent` 4値入口）と実ワークツリーの整合を確認。`ConvoPeq.md` は正本として Step 2 の `RetryScheduleRequest` 未記載（Step 2 type 定義は md 生成前の最小差分として正常）。

---

## PendingRetry に必要なフィールド候補 — data-flow 追跡

`RetryScheduleRequest`（4 enum = submitRebuildIntent の完全な intent 識別子）を Scheduler の内部 pending state に接続するために、Step 3 で `PendingRetry` に必要な最小フィールドをデータフロー追跡で確定（名前推測ではなく現行ソースから導出）:

| 候補 | 必要 | 根拠 |
|------|------|------|
| `RetryScheduleRequest request` | **必須** | `submitRebuildIntent(kind, reason, rebuildClass, collapsePolicy)` の4値を expiry 時に再現するため。 |
| `std::chrono::steady_clock::time_point deadline` | **必須** | `schedule(req, delay)` → `deadline = now + delay`、scheduler thread の `wait_until(deadline)` で delayed executor を実現（D-5-1.1 凍結 `PendingRetry {request, deadline}` 2要素）。 |
| `generation` | **不要** | `requestRebuild() → ++rebuildRequestGeneration` が authoritative source。RetryScheduler が保持すると singularization と衝突（Gate D）。 |
| `sequence / epoch` | **不要** | 同上。 |
| `RuntimeWorld*` | **不要** | Gate F。 |
| `BuildError / RetryDisposition` | **不要** | Gate E。admission 側で `BuildOutcome.retry` → `delay` に変換済み。 |
| `attempt` | **不要** | D-5-1.1 C2 で削除済み（PendingRetry は retry state machine を持たない、Delayed Intent Scheduler）。 |
| `backoff / jitter / maxAttempts` | **不要** | 同上。backoff は admission 側の責務。 |
| `AudioEngine*` | **不要（PendingRetry 内）** | `PendingRetry` は per-entry payload、`AudioEngine*` は `RetryScheduler` 本体の `engine_` で1つ保持（NonRT lifecycle）。Per-entry に含めない。 |

**凍結（Step 3 実装時）:**

```cpp
struct PendingRetry {
    RetryScheduleRequest request;
    std::chrono::steady_clock::time_point deadline;
};
```

2要素のみ。Step 4 の `queue<PendingRetry>` は `deque<PendingRetry>` + deadline ordering（`upper_bound` with `deadline`）で `capacity 8 / reject-newest / earliest deadline first`。

---

## PendingRetry semantic responsibility — 明確化

```
PendingRetry = retry execution state（Delayed Intent Scheduler の deadline-ordered pending queue）
NOT = logical Recovery obligation
```

I4 D14/D15（logical obligation の ownership / admission / non-supersedable）は Recovery の `drainPendingRecoveryAdmission` / `kMaxRecoveryConsecutiveFailures=4` / `settle(true/false)` の別 authority（D-5-0 §5、INV-D4-3/10）。`PendingRetry` の retry execution state と logical Recovery obligation の lifecycle は別問題として扱う。

Step 3 では **Recovery obligation を PendingRetry に勝手に導入しない**。D-5 の RetryScheduler pending state と Recovery obligation は分離を維持。

---

## NonRT invariant — PASS

`Audio Thread`（`processBlock` / `getNextAudioBlock`）が `PendingRetry` / Scheduler の producer にならないこと。

```
rg -n 'AudioThread|processBlock.*RetryScheduler|getNextAudioBlock.*PendingRetry' src --type cpp --type h → 0
```

Practical Stable ISR Bridge Runtime の「RTは判断・所有・危険操作をしない」原則とも一致。Scheduler producer は NonRT-only（MessageThread / RebuildThread / `schedule()` caller）。

---

## Step 4 premature implementation — 全項目未実装を確認

| 項目 | 期待 | 実測 | 判定 |
|------|------|------|------|
| `class RetryScheduler` | 未実装 | 0 hits | PASS |
| `RetryScheduler::` | 未実装 | 0 | PASS |
| `queue / timer / worker scheduling` | 未実装 | 0 | PASS |
| `schedule()` | 未実装 | 0 | PASS |
| `retry execution` | 未実装 | 0 | PASS |
| `backoff` | 未実装 | 0（`BuildErrorPolicy.h:32` コメントのみ） | PASS |
| `lifecycle wiring` | 未実装 | `AudioEngine.h` member `retryScheduler` 0 | PASS |

---

## PASS条件 — 全PASS

```
[x] A〜J 全PASS
[x] PendingRetry の semantic responsibility が明確（retry execution state、Recovery obligation と分離）
[x] Step 4 の責務が未実装（class RetryScheduler / queue / timer / schedule / backoff / lifecycle 全0）
[x] AudioEngine / call-site 変更 0（submitRebuildIntent 15+ hits 不変）
[x] 最新 ConvoPeq.md と実ワークツリーの整合確認（submitRebuildIntent 4値入口、ConvoPeq.md は Step 2 type 未記載で正常）
[x] PendingRetry フィールド候補をデータフロー追跡で確定（{request, deadline} 2要素、generation/BuildError/RuntimeWorld なし）
```

**D101-22 Step 3 Gate — GO**

Step 3 の Gate Audit → GO 判定を満たしたため、次は Step 3 の最小実装（`PendingRetry {request, deadline}` 2要素の `RetryScheduler.h` への追加、他ファイル変更なし）へ進入可能。`PendingRetry` の実装は行わない — 本レポートは audit のみ。

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep
rg -n 'PendingRetry' src --type cpp --type h
rg -n 'RetryScheduleRequest' src --type cpp --type h
rg -n 'enum class (RebuildKind|RebuildTelemetryReason|RebuildTelemetryClass|RebuildTelemetryPolicy)' src --type h --type cpp
rg -n 'RebuildKind|RebuildTelemetryReason|RebuildTelemetryClass|RebuildTelemetryPolicy' src --type h --type cpp
rg -n 'BuildError|RetryDisposition|FailureClassification|BuildOutcome' src/audioengine/RetryScheduler.h
rg -n 'BuildError|RetryDisposition|FailureClassification|BuildOutcome' src/audioengine/RetrySchedulerTypes.h
rg -n 'generation|sequence|epoch' src/audioengine/RetryScheduler.h
rg -n 'RuntimeWorld|PublicationAuthority|WorldAuthority' src/audioengine/RetryScheduler.h src/audioengine/RetrySchedulerTypes.h src/core/RebuildTypes.h
rg -n 'submitRebuildIntent' src --type cpp --type h
rg -n 'class RetryScheduler|RetryScheduler::|PendingRetry|schedule\(' src --type cpp --type h
rg -n 'AudioEngine.*RetryScheduler|RetryScheduler.*AudioEngine' src --include='*.h' --include='*.cpp'
# ag
ag -n 'RetryScheduleRequest|PendingRetry' src; ag -n 'RebuildKind' src/audioengine
# fdfind / fzf / sed / awk
fdfind -e h -e cpp 'BuildError|RebuildDispatch|AudioEngine\.h' .; fdfind -e h 'RetryScheduler' .; fdfind -e h . src/audioengine | fzf --filter='Retry'
sed -n '1,30p' src/audioengine/RetryScheduler.h; sed -n '1,70p' src/audioengine/RetrySchedulerTypes.h; sed -n '1,30p' src/core/RebuildTypes.h
awk '/RetryScheduler|RebuildKind|RebuildTelemetry/{print FILENAME":"NR": "$0"}' src/audioengine/RetryScheduler.h src/audioengine/RetrySchedulerTypes.h src/core/RebuildTypes.h
awk '/submitRebuildIntent|rebuildRequestGeneration|isObsolete|shouldRetryWarmupFailure/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
# ast-grep
sg run -p 'RetryScheduleRequest' --lang cpp src/; sg run -p 'PendingRetry' --lang cpp src/; sg run -p 'BuildError' --lang cpp src/audioengine/RetryScheduler.h
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/; sg run -p 'RebuildKind' --lang cpp src/
# cocoindex
ccc status; ccc grep 'RetryScheduleRequest'; ccc grep 'RebuildKind'; ccc grep 'submitRebuildIntent'; ccc grep 'RetryDisposition'
# graphify
graphify query 'RebuildKind'; graphify query 'RetryScheduleRequest'; graphify query 'AudioEngine'; graphify query 'submitRebuildIntent'; graphify path 'BuildError' 'submitRebuildIntent'
# semble
semble search 'RetryScheduleRequest' . --max-snippet-lines 5; semble search 'PendingRetry' . --max-snippet-lines 5; semble search 'RebuildKind' . --max-snippet-lines 5
# AiDex
# aidex_query term="BuildError" mode="contains"
# serena
# .serena/project.yml
# context-mode
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.h').split('\n').filter(l=>l.includes('submitRebuildIntent')).join('\n')")
# RTK(WSL)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "PendingRetry" src/'
# headroom
# 大きな断片は context-mode で仮想化
```

## 参照

* `src/audioengine/RetrySchedulerTypes.h` — 3 enum（Step 1 抽出）
* `src/core/RebuildTypes.h` — `convo::RebuildKind`
* `src/audioengine/RetryScheduler.h` — `RetryScheduleRequest` 4-field（Step 2）
* `src/audioengine/AudioEngine.h:40` — `#include "RetrySchedulerTypes.h"` + `using` alias
* `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — `submitRebuildIntent` 4引数
* `src/audioengine/BuildErrorPolicy.h` — 8値 default policy（D-3）
* `evidence/D101-20-Phase-D-5-2-Step2-Implementation-Gate-Audit.md` — Step 2 Gate PASS
* `ConvoPeq.md` — 正本
