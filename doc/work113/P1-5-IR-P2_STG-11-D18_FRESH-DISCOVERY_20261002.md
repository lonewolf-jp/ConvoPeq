# STG-11-D18 Fresh Discovery — Lifecycle / Authority / Queue / RT Boundary Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D18_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D18** — D17 までの結果を前提とした新規 defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-142258).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `9f152d9d2c5f37414643659275d2a4d15425159b`（D17 docs）
  - `Generated: 2026-10-02 23:19:21` / 5,941,824 B
  - SHA-256: `3A51AB3414DBA90005B99540376A93D6AB76AE1A96A754CE9E7050E2E0593045`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery** で開始し、NO-GO 確定のため
  production / test / build ファイルの変更は一切行っていない。

---

## 0. 判定

```
STG-11-D18 — NO-GO（concrete defect 0 件、observation 1 件）
```

RuntimeWorld / Snapshot / DSPProjection の ownership・lifetime、
Publish → Crossfade → Retire → Delete の authority singularization、
identity の一意性、二重表現、publish 全 state transition、execution tail と commit の整合性、
queue overflow / fallback / quarantine / drop の ownership、shutdown の停止順序と
Drain / Epoch / reclaim、RCU reader と callback と worker lifetime、RT boundary、
atomic 規約、validator bypass、D7〜D17 の再発、D16 O-1 / D17 O-2 を現行ソースで確認した。
Owner の 7 条件をすべて満たす新規 defect は存在しない。

---

## 1. 追跡結果

### 1.1 deferred terminal disposition の一回性

全 terminal 経路（rejected-stale-generation / not-finalized / pressure / shutdown /
deferred-overwrite / retry-exhausted / timer-clear-midrun / redrive-budget-exhausted /
deferred-discard / shutdown-clear）は `retireRegisteredDSP` helper
（`RuntimePublicationOrchestrator.cpp:767-784`、null-safe・resolve 失敗時 no-op）に
一本化されている。slot は常時 1 req のみ保持し、各 terminal は consume するため、
二重 retire の構造は無い。`finishView`（`:789-814`）と `invalidateDeferredObligation`
（`:818-824`）の terminal / retention 分離も設計どおりである。

### 1.2 Execution tail と commit 状態

`RuntimePublishExecutor.h:55-115` の単一 site を再確認。`committed==false` 時の
unconditional tail は producer 保証＋FIFO により到達不能であり、反例を構成できない
（D17 O-2 と同一結論）。

### 1.3 retire intent queue の MPSC 安全性

`LifetimeState::emitRetireIntent`（`ISRRetire.cpp:23-91`）は ticket 式 MPSC であり、
fallback queue と overflow ring への退避は `fallbackMutex_` の内側である。
audio thread は現行コードで emit しないため RT 波及は無い。
容量 256 + 4096 で drop counter 付きであり、silent loss の構造は無い。

### 1.4 shutdown / RCU / callback / identity / RT / atomic / validator

いずれも D16/D17 の結論と同一であり、現行 tree（production 無変更）で再発は無い。
D15-1 mutex の NonRT-only、DSPCore process 系の lock-free、atomic wrapper 規約、
単一 production Bridge はいずれも維持されている。

### 1.5 D7〜D17 の再発確認（該当なし）

### 1.6 D16 O-1 / D17 O-2 の扱い

いずれも具体的な ownership・lifetime・authority・RT safety への帰結を確認できないため、
Observation のまま残し、production 修正は行わない。

---

## 2. Observation O-3: deferred `retry-exhausted-discard` の dormant 性（defect 計上なし）

`enqueueDeferred` 内の `deferredRetryCount_ > kMaxDeferredRetries` 分岐
（`RuntimePublicationOrchestrator.cpp:558-570`）は、F6-6 により現行 production では
発火しないことがコード上に記録されている（Type-A retry 経路が存在しないため
count は常に 0）。将来 Type-A が追加された場合の dormant guard として保持されている。
発火不能な分岐の存在自体は defect ではない。記録のみ残す。

---

## 3. 本 D18 の作業記録

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = 9f152d9d（= origin/main）
```
