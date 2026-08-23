# D101-20 Phase D-5-2 Step 2 — Implementation Gate / Type & Data-flow Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** D-5-2 Step 2 `RetryScheduleRequest` 実装前の Gate Audit（**audit-only、コード変更0**）
**Prerequisite:** D101-19 Step 0 PASS (12 symbol), D-5-2-1 GO (20項目), D-5-2 Step 1 完了（`RetrySchedulerTypes.h` 抽出、36/36 PASS）
**Primary source:** 実ワークツリー（`src/audioengine/RetrySchedulerTypes.h`, `src/core/RebuildTypes.h`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:149`, `src/audioengine/BuildErrorPolicy.h`）と `ConvoPeq.md` 正本

---

## 手法 — 全ツール横断

| 系統 | ツール | 実行 | 結果 |
|------|--------|------|------|
| WSL | `rg` | `RetryScheduleRequest` / `enum class RebuildKind\|RebuildTelemetry*` / `BuildError\|RetryDisposition` in RetrySchedulerTypes.h / `submitRebuildIntent` / `class RetryScheduler` / `RuntimeWorld\|PublicationAuthority` / `generation\|epoch\|sequence` | 全て再実行、Gate判定の一次資料 |
| WSL | `ag` | `ag -n 'RebuildKind' src`, `ag -n 'RetryScheduleRequest' src` | `rg`と一致 |
| WSL | `fdfind`/`fzf` | `fdfind -e h 'RetryScheduler'`, `fdfind -e h -e cpp 'RebuildTypes'` | `RetrySchedulerTypes.h` / `RebuildTypes.h` 検出、`RetryScheduler.h/.cpp` 0件 |
| WSL | `sed` | `sed -n '1,70p' RetrySchedulerTypes.h`, `sed -n '1,30p' RebuildTypes.h` | 4 enum 実体抽出 |
| WSL | `awk` | `awk '/RetryScheduler\|RebuildKind\|RebuildTelemetry/'` | 型分離確認 |
| WSL | `sg` | `sg run -p 'RetryScheduleRequest'`, `sg run -p 'RebuildKind'`, `sg run -p 'BuildError'` | `RetryScheduleRequest` 0 hits |
| MCP | `serena` | `.serena/project.yml` | `language_servers: [cpp,python,bash]` 正常 |
| CLI | `cocoindex` | `ccc status` (133184 chunks), `ccc grep 'RetryScheduleRequest'` 0, `ccc grep 'RebuildKind'` | 一致 |
| CLI | `graphify` | `graphify query 'RebuildKind'`, `query 'RetryScheduleRequest'` | `RebuildKind` は `core/RebuildTypes.h` graph、`RetryScheduleRequest` 未登録 |
| CLI | `semble` | `semble search 'RetryScheduleRequest'` 0 | 一致 |
| MCP | `AiDex` | `BuildError` 45 hits | `BuildErrorPolicy.h` 集約 |
| sandbox | `ctx_execute` (js) | `AudioEngine.h` 4 enum scope / `RebuildTypes.h` / `AudioEngine ownership` / `NonRT` | 一致 |
| WSL | `RTK` | `rtk grep` fallback | 差異0 |
| MCP | `headroom` | context-mode仮想化 | 遵守 |

全ツール一致、差異0。

---

## Gate 監査 — 10項目

### 1. `RetryScheduleRequest` の唯一の定義場所 — PASS

**実測:**
- `rg -n 'RetryScheduleRequest' src --type cpp --type h` → **0 hits**（`ag`, `sg`, `cocoindex grep`, `semble`, `graphify query` 全て0）
- `RetryScheduler.h` / `RetryScheduler.cpp` 未存在（`fdfind -e h 'RetryScheduler'` → `RetrySchedulerTypes.h` のみ、`RetryScheduler.h` 0）

**凍結:**
- 唯一の定義場所は **`src/audioengine/RetryScheduler.h`** とする（Step 2 で新設、Types header には置かない）。
- `RetrySchedulerTypes.h` は enum 分離の責務のみに限定（Step 1 PASSの `RetryScheduleRequest/ PendingRetry/ RetryScheduler/ BuildError` 混入なしを維持）。
- `RetryScheduleRequest` を Types header に入れると、Types header が `RetryScheduler` の request 概念を持つことになり責務混同。Step 1 の `入れてはいけないもの` リストを厳守。

**判定: PASS — 現時点未定義、定義場所は `RetryScheduler.h` に一意に確定**

### 2. field 最小集合 — PASS

**凍結:**

```cpp
// src/audioengine/RetryScheduler.h
struct RetryScheduleRequest {
    convo::RebuildKind kind;                 // core/RebuildTypes.h
    RebuildTelemetryReason reason;           // RetrySchedulerTypes.h
    RebuildTelemetryClass rebuildClass;      // RetrySchedulerTypes.h
    RebuildTelemetryPolicy collapsePolicy;   // RetrySchedulerTypes.h
    // generation / sequence / epoch / BuildError / RetryDisposition なし
};
```

- `submitRebuildIntent(kind, reason, rebuildClass, collapsePolicy)` の4引数と1:1で対応（`AudioEngine.RebuildDispatch.cpp:149` `void AudioEngine::submitRebuildIntent(RebuildKind, RebuildTelemetryReason, RebuildTelemetryClass, RebuildTelemetryPolicy) noexcept`）。
- `submitRebuildIntent` の4引数は `generation` を外部から受け取らない（`requestRebuild()` 内 `++rebuildRequestGeneration`）。したがって request に `generation` は不要。
- フィールド追加（例: `delay`, `deadline`）は `RetryScheduleRequest` に入れず、`schedule(req, delay)` の第2引数 `delay` として渡す（D-5-1.1 で `PendingRetry {request, deadline}` 2要素に凍結済み）。request 自体は純粋な intent 識別子に留める。

**判定: PASS — 4フィールド最小集合を凍結**

### 3. generation / sequence / epoch を新たに保持しない — PASS

**実測:**
- `rg -n 'generation|epoch|sequence' src/audioengine/RetrySchedulerTypes.h` → 0 hits（`ag` 同様）
- `RebuildTypes.h` は `enum RebuildKind` のみ（`generation` なし）
- 既存 authoritative source: `AudioEngine.h:2506` `std::atomic<int> rebuildRequestGeneration{0}` / `2654 rebuildThreadShouldExit` / `isObsolete(generation) { return generation != consumeAtomic(rebuildRequestGeneration) }` — caller → authoritative epoch singularization が既に存在（D-5-2 Steps 0/1 で確認）

**凍結:**
- `RetryScheduleRequest` に `generation / sequence / epoch / RuntimeWorld::generation / publication epoch` を追加しない。`RetryScheduler` 内の `PendingRetry` にも `generation` を持たせない（D-5-1.1 C2で `attempt` と共に削除済み）。
- 生成/epoch の singularization と衝突する再導入を禁止。`submitRebuildIntent → requestRebuild → ++rebuildRequestGeneration` に委譲。

**判定: PASS**

### 4. `BuildError` / `FailureClassification` / `RetryDisposition` を混入させない — PASS

**実測:**
- `rg -n 'BuildError|RetryDisposition|FailureClassification|BuildOutcome' src/audioengine/RetrySchedulerTypes.h` → 0 hits
- `rg -n 'BuildError|RetryDisposition' src/audioengine/RetrySchedulerTypes.h` → 0（`ag` 同様、`sg run -p 'BuildError'` 0）
- `BuildErrorPolicy.h` は `BuildError 8値 / FailureClassification 4値 / RetryDisposition 3値 / BuildOutcome / kBuildErrorDefaultTable / classifyBuildError()` を独立した policy authority として保持（D-3 65 checks PASS）

**凍結:**
- `RetryScheduleRequest` に `BuildError` / `RetryDisposition` / `FailureClassification` を入れない。REPAIR_PLAN2 の BuildError/RetryDisposition 分離方針（`BuildErrorPolicy` が policy authority）に従い、Scheduler は admitted delayed intent の executor に限定（invocation 側で `classifyBuildError → if (outcome.retry != NoRetry) schedule(req, delay)` を済ませてから `schedule`）。
- 将来 `ResourceUnavailable → RetryBackoff` 経路でも、Scheduler 側で再分類せず caller 側の admission で `delay` を決定して渡す。

**判定: PASS**

### 5. ownership / lifetime が NonRT-only — PASS

**実測:**
- `submitRebuildIntent` call sites（`rg submitRebuildIntent` 15+ hits）: `AudioEngine.Parameters.cpp` 15箇所（UI Setter, MessageThread）、`RebuildDispatch.cpp:1165` warmup retry（RebuildThread）、`Init.cpp:94` / `PrepareToPlay.cpp`（MessageThread）、`Timer.cpp:794,894`（MessageThread Timer）、`UIEvents.cpp` 等 — 全て NonRT（MessageThread / RebuildThread / Worker）。
- `Audio Thread` (`processBlock` / `getNextAudioBlock`) から `submitRebuildIntent` / `schedule` の呼出は0（`rg AudioThread.*submitRebuildIntent` 0、`rg AudioThread.*RetryScheduler` 0、D-5-2-0 12項目 PASS）。
- `RetryScheduler` 所有: `AudioEngine` が `unique_ptr<RetryScheduler>` で所有、`engine_` は `AudioEngine*` non-owning（D-5-2-1 GO 20項目で `shutdown → join` 順序凍結済み）。

**凍結:**
- `RetryScheduler::schedule()` は NonRT-only（`processBlock` から呼出禁止）。`pendingCount()`, `shutdown()` も NonRT-only。`run()` の `submitRebuildIntent` 呼出しも NonRT thread（scheduler thread）。

**判定: PASS**

### 6. `RebuildKind` と telemetry 3 enum の責務境界 — PASS

**実測:**
- `RebuildKind` authoritative source: `src/core/RebuildTypes.h`（`#pragma once` / `#include <cstdint>` / `namespace convo { enum RebuildKind: uint32_t { None, Structural, Runtime } }`）— JUCE/AudioEngine 独立、D-5-2 Step 1で維持。
- 3 telemetry: `src/audioengine/RetrySchedulerTypes.h`（`#include <cstdint>` のみ、`class AudioEngine;` 削除済み、global `enum RebuildTelemetryReason/Class/Policy : uint8_t` 36/3/3値）— `AudioEngine.h:40` `#include "RetrySchedulerTypes.h"` + `using RebuildTelemetryReason = ::RebuildTelemetryReason;` aliasで `AudioEngine::RebuildTelemetryReason` 外部参照互換を維持。
- `AudioEngine.h` 内 `enum RebuildTelemetryReason` 定義は削除済み（`rg enum class RebuildTelemetryReason` → `RetrySchedulerTypes.h` のみ、`core/RebuildTypes.h` に `RebuildKind` のみ）。
- Formatter / switch (`toTelemetryReasonString` / `toTelemetryClassString` / `toTelemetryPolicyString`) は `AudioEngine.h` に残留（`*String` 関数）、Types header には enum declaration のみ — 境界維持。

**凍結:**
- `RebuildKind` は `core/RebuildTypes.h` を authoritative source として維持（変更なし）。
- 3 telemetry は `RetrySchedulerTypes.h` に分離、`AudioEngine.h` は `using` alias で互換維持。`RetryScheduleRequest` は `convo::RebuildKind`（RebuildTypes.h）+ 3 telemetry（RetrySchedulerTypes.h）を組み合わせる。
- `RetryScheduler.h` は `RetrySchedulerTypes.h` + `core/RebuildTypes.h` を include し、`AudioEngine.h` / `BuildErrorPolicy.h` / `WorkerThread` を含めない（D-5-2-0 12項目 PASSの循環回避を維持）。

**判定: PASS**

### 7. `RetryScheduler` がまだ未実装 — PASS

**実測:**
- `rg -n 'class RetryScheduler|RetryScheduler::' src --type cpp --type h` → 0 hits（`ag`, `sg`, `cocoindex grep RetryScheduler` 0、`semble search RetryScheduler` 0、`graphify query RetryScheduler` 0 nodes、`fdfind -e h RetryScheduler` → Types.h のみ）
- `src/audioengine/RetryScheduler.h` / `RetryScheduler.cpp` 未存在（`fdfind` 0、`ls -la src/audioengine/RetryScheduler*` → Types.h のみ）

**凍結:** D-5-2 Step 2 で `RetryScheduler.h` を新設する前提で、現時点未実装を維持。Step 2 scope creep（`PendingRetry` / `AudioEngine member` / `thread` 等の先走り）を禁止（D-5-2 Step 1 禁止リスト準拠）。

**判定: PASS**

### 8. existing `submitRebuildIntent()` との data-flow 重複なし — PASS

**実測:**
- `submitRebuildIntent(RebuildKind, RebuildTelemetryReason, RebuildTelemetryClass, RebuildTelemetryPolicy) noexcept`（`AudioEngine.h:2962`, `RebuildDispatch.cpp:149`）— 4引数、generation 外部渡しなし、`void` fire-and-forget。
- 既存 pipeline: 各種 setter → `submitRebuildIntent` → `rebuildAdmissionPendingIntent_` latest-wins / `NonMtAlreadyPending` マージ → `requestRebuild(sr,bs)` → `pendingTask`（`rebuildMutex` + `hasPendingTask` 1-slot queue, `rebuildRequestGeneration++`）— `RebuildKind` の実体は MPSC SPSC-like（`retryScheduler_->schedule` 追加で1 producer 追加、SPSC invariant は既に崩れているため影響なし — D-5-0 §3 MPSC）。

**凍結:**
- `RetryScheduleRequest` → `RetryScheduler::schedule(req, delay)` → `deadline wait` → `engine_->submitRebuildIntent(req.kind, req.reason, req.rebuildClass, req.collapsePolicy)` の一方向のみ。`submitRebuildIntent` → `RetryScheduler::schedule` の逆方向循環なし。
- Data-flow 重複なし：RetryScheduler は既存 admission を置換せず前段の delayed executor として限定（`schedule` は admitted request の `delay` 待ちのみ）。

**判定: PASS**

### 9. Retry request が RuntimeWorld / publication authority を直接参照しない — PASS

**実測:**
- `rg -n 'RuntimeWorld|PublicationAuthority|WorldAuthority' src/audioengine/RetrySchedulerTypes.h src/core/RebuildTypes.h` → 0 hits
- `rg -n 'RuntimeWorld|PublicationAuthority' src/audioengine/RetrySchedulerTypes.h` → 0（`ag` 同様）
- `RetrySchedulerTypes.h` は `<cstdint>` のみ include、JUCE/AudioEngine/BuildErrorPolicy/WorkerThread 非依存（Step 1 S1-05 PASS）

**凍結:**
- `RetryScheduleRequest` は `RuntimeWorld*` / `PublicationAuthority` / `worldAuthority` / `publication epoch` を直接参照しない。generation/epoch の authoritative source 再導入は禁止（Gate 3 と同根）。Retry request は rebuild semantic type（4 enum）のみで RuntimeWorld とは疎結合。

**判定: PASS**

### 10. Step 2 で変更するファイルと call-site を事前確定 — PASS

**事前確定（変更ファイル）:**

| ファイル | 操作 | 内容 |
|----------|------|------|
| `src/audioengine/RetryScheduler.h` | **新設** | `RetryScheduleRequest` 唯一の定義（4フィールド）+ `RetryScheduler` forward declare 相当の型境界（但し scheduler 本体はまだ未実装 — Step 2 では request 定義のみ） |
| `src/audioengine/RetrySchedulerTypes.h` | **変更なし** | Step 1 抽出済み、4 enum のみ維持（`RetryScheduleRequest` をここに入れない） |
| `src/core/RebuildTypes.h` | **変更なし** | `RebuildKind` authoritative 維持 |
| `src/audioengine/AudioEngine.h` | **変更なし**（Step 2 では） | `RetrySchedulerTypes.h` include + using alias 維持、`submitRebuildIntent` 4引数維持 |
| `CMakeLists.txt` | **変更なし**（Step 2 では） | `RetryScheduler.h` は header-only 型定義のためビルド登録不要（Step 4 RetryScheduler core 実装時に登録） |

**Call-site:**
- Step 2 では `RetryScheduleRequest` 定義のみで call-site 変更なし。`submitRebuildIntent` / `requestRebuild` / `warmup retry` / `prepareToPlay` / `destructor` の call-site は全て Step 1 と同一を維持（`rg submitRebuildIntent` 15+ hits 不変）。
- Step 3 以降で `RetryScheduler.h` の `PendingRetry` / Step 4 `RetryScheduler core` / Step 5 `AudioEngine lifecycle wiring` と段階的に接続（D-5-2 実装順序）。

**判定: PASS — 変更ファイルは `RetryScheduler.h` 新設のみ、call-site 0変更を事前確定**

---

## 総合判定

```
1. 唯一の定義場所                          PASS — RetryScheduler.h に一意
2. field 最小集合                          PASS — 4フィールド（kind/reason/class/policy）
3. generation/sequence/epoch 保持なし       PASS — authoritative epoch と衝突回避
4. BuildError/RetryDisposition 混入なし     PASS — BuildErrorPolicy 独立維持
5. NonRT-only                              PASS — Audio Thread producer 0
6. RebuildKind vs telemetry 3 enum 境界    PASS — RebuildTypes.h / RetrySchedulerTypes.h 分離
7. RetryScheduler 未実装                    PASS — 現時点0、Step 2 scope creep なし
8. submitRebuildIntent との重複なし         PASS — 一方向 delayed executor
9. RuntimeWorld 直接参照なし                PASS — 疎結合維持
10. 変更ファイル/call-site 事前確定         PASS — RetryScheduler.h 新設のみ、他0
```

**D-5-2 Step 2 Implementation Gate — PASS / GO**

`generation` を `RetryScheduleRequest` に入れない判断は、caller → authoritative epoch singularization との衝突回避として妥当。Step 1 GO に続き Step 2 も GO とし、`RetryScheduleRequest` 最小定義の実装へ進入可能。

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep
rg -n 'RetryScheduleRequest' src --type cpp --type h
rg -n 'enum class (RebuildKind|RebuildTelemetryReason|RebuildTelemetryClass|RebuildTelemetryPolicy)' src --type h --type cpp
rg -n 'RebuildKind|RebuildTelemetryReason|RebuildTelemetryClass|RebuildTelemetryPolicy' src --type h --type cpp
rg -n 'BuildError|RetryDisposition|FailureClassification|BuildOutcome' src/audioengine/RetrySchedulerTypes.h
rg -n 'generation|sequence|epoch' src/audioengine/RetrySchedulerTypes.h
rg -n 'RuntimeWorld|PublicationAuthority|WorldAuthority' src/audioengine/RetrySchedulerTypes.h src/core/RebuildTypes.h
rg -n 'submitRebuildIntent' src --type cpp --type h
rg -n 'class RetryScheduler|RetryScheduler::' src --type cpp --type h
rg -n 'AudioEngine.*RetryScheduler|RetryScheduler.*AudioEngine' src --include='*.h' --include='*.cpp'
# ag
ag -n 'RetryScheduleRequest' src; ag -n 'RebuildKind' src/audioengine; ag -n 'BuildError' src/audioengine/RetrySchedulerTypes.h
# fdfind / fzf / sed / awk
fdfind -e h -e cpp 'BuildError|RebuildDispatch|AudioEngine\.h' .; fdfind -e h 'RetryScheduler' .; fdfind -e h . src/audioengine | fzf --filter='Retry'
sed -n '1,70p' src/audioengine/RetrySchedulerTypes.h; sed -n '1,30p' src/core/RebuildTypes.h
sed -n '78,85p' src/audioengine/AudioEngine.RebuildDispatch.cpp; sed -n '149,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/RetryScheduler|RebuildKind|RebuildTelemetry/{print FILENAME":"NR": "$0"}' src/audioengine/RetrySchedulerTypes.h src/core/RebuildTypes.h
awk '/submitRebuildIntent|rebuildRequestGeneration|isObsolete|shouldRetryWarmupFailure/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
# ast-grep
sg run -p 'RetryScheduleRequest' --lang cpp src/; sg run -p 'RebuildKind' --lang cpp src/; sg run -p 'BuildError' --lang cpp src/
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/
# cocoindex
ccc status; ccc grep 'RetryScheduleRequest'; ccc grep 'RebuildKind'; ccc grep 'submitRebuildIntent'; ccc grep 'RetryDisposition'
# graphify
graphify query 'RebuildKind'; graphify query 'RetryScheduleRequest'; graphify query 'AudioEngine'; graphify query 'submitRebuildIntent'; graphify path 'BuildError' 'submitRebuildIntent'
# semble
semble search 'RetryScheduleRequest' . --max-snippet-lines 5; semble search 'RebuildKind' . --max-snippet-lines 5
# AiDex
# aidex_query term="BuildError" mode="contains"
# serena
# .serena/project.yml
# context-mode
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.h').split('\n').filter(l=>l.includes('RebuildKind')).join('\n')")
# RTK(WSL)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "RebuildKind" src/'
# headroom
# 大きな断片は context-mode で仮想化
```

## 参照

* `src/audioengine/RetrySchedulerTypes.h` — 3 enum（Step 1 抽出、<cstdint> のみ）
* `src/core/RebuildTypes.h` — `RebuildKind` authoritative（`namespace convo`, `uint32_t`）
* `src/audioengine/AudioEngine.h:40` — `#include "RetrySchedulerTypes.h"` + `using RebuildTelemetry*` alias / `2962 submitRebuildIntent`
* `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — `submitRebuildIntent` 4引数 `noexcept`
* `src/audioengine/BuildErrorPolicy.h` — 8値 default policy（D-3）
* `evidence/D101-19-Phase-D-5-2-0-Implementation-Step0-Audit.md` — Step 0 PASS（12 symbol）
* `ConvoPeq.md` — 最新ソーススナップショット
