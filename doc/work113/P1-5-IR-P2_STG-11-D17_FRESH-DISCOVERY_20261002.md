# STG-11-D17 Fresh Discovery — Lifecycle / Authority / Queue / RT Boundary Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D17_FRESH-DISCOVERY_20261002.md`
- Work item: **STG-11-D17** — D16 までの修正を前提とした新規 defect 探索
- Date: 2026-10-02
- Authority: リポジトリルート `ConvoPeq.md`（監査 AI が参照する `ConvoPeq(20261002-142258).md` と同一ファイル。
  ファイル名の違いを理由に別世代として扱わない。Owner 確認済みの同一最新統合ソース）
  - baseline commit: `a1ce01b24796904622d5205ca6d09b66dc5eaf6f`（D16 docs）
  - `Generated: 2026-10-02 23:19:21` / 5,941,824 B
  - SHA-256: `3A51AB3414DBA90005B99540376A93D6AB76AE1A96A754CE9E7050E2E0593045`
  - `--check`: `NEWER_SRC_COUNT = 0` / **FRESH**（調査開始時に確認）
  - 過去資料・記憶・旧行番号は根拠として使用していない。結論はすべて現行ソースから導出。
- 本 work item は **read-only Fresh Discovery** で開始し、NO-GO 確定のため
  production / test / build ファイルの変更は一切行っていない。

---

## 0. 判定

```
STG-11-D17 — NO-GO（concrete defect 0 件、observation 1 件）
```

RuntimeWorld / Snapshot の ownership と lifetime、Publish → Crossfade → Retire → Delete の
authority singularization、identity の一意性、OwnerChannel / Registry / Intent /
deferred / quarantine の二重表現、failure / timeout / rollback / retry、
queue overflow、shutdown drain、RCU reader、非同期 callback、RT boundary、
atomic 規約、validator bypass、D7〜D16 の再発、D16 O-1 を現行ソースで確認した。
Owner の 7 条件をすべて満たす新規 defect は存在しない。

---

## 1. 追跡結果

### 1.1 Execution tail の singularization

`commit → unregister → onPublishCompleted → advanceRetireEpoch → onPublishCommitted` は
`RuntimePublishExecutor.h:55-115` の 1 箇所のみである。
`ConvolverProcessor` 側の `advanceRetireEpoch`（別 subsystem の convolver-local epoch）は
対象が異なり重複ではない。順序逆転の構造は無い。

### 1.2 二重表現の所有権処理（OwnerChannel / Registry / Intent）

`executePublish` は owner / registry / payload の 3 経路を優先順位付きで解決し
（`RuntimePublishExecutor.h:42-45`）、commit 後に registry を unregister する（`:80`）。
`committed==false` 時の `*newWorld` deref は producer 保証＋FIFO により到達不能であることが
コード上に記録されている（`:65-67`）。到達可能な反例を構成できないため計上しない。

### 1.3 identity の一意性

`reserveRuntimePublicationIdentity`（`AudioEngine.h:3763-3772`）は 3 値を atomic 採番し
再利用しない。`publicationSequence` は初回から非ゼロであり facade の拒否条件と整合する。

### 1.4 retire intent queue の MPSC 安全性

`LifetimeState::emitRetireIntent`（`ISRRetire.cpp:23-91`）は ticket 式 Vyukov MPSC であり、
fallback queue と overflow ring への退避は `fallbackMutex_` の内側である。
SPSC の overflow ring が複数スレッドから直接叩かれる経路は無い。
audio thread は現行コードで `emitRetireIntent` を呼ばない（process ファイルに呼出し無し）ため、
fallback mutex が RT に波及することも無い。容量は 256 + 4096 であり、
全段枯渇には drop counter が付く。silent loss の構造は無い。

### 1.5 RCU reader / 非同期 callback

read handle は move-only RAII の scoped 使用であり、token の lifetime 逸脱は無い。
autosave callback は `safeThis`＋dtor クリア、health monitor callback は member 間であり、
UAF 経路は無い。

### 1.6 RT boundary / atomic 規約 / validator bypass

DSPCore process 系に mutex / lock / alloc / delete は無く、
setter / restore 系に raw atomic API は無い（D13/D16 と同一結論を現行 tree で再確認）。
production Bridge は単一であり bypass は無い。

### 1.7 D7〜D16 の再発確認（該当なし）

`setDitherBitDepth` / `setNoiseShaperType` の guard、D7 の StateIO guard、
`ownerChannelProducerMutex_` の 2 箇所の lock はいずれも現行ソースで維持されている。

### 1.8 D16 O-1 の扱い

`currentBuildSnapshot_` の非 lock 読出し（`Timer.cpp:1165`）について、
ownership・lifetime・authority・RT safety への具体的帰結は本 R でも確認できない。
plain int/double の混合 snapshot であり自己修復するため、Observation のまま残し、
production 修正は行わない（Owner 指示どおり）。

---

## 2. Observation O-2: `committed==false` 時の Execution tail（defect 計上なし）

`RuntimePublishExecutor.h:104-114` の Execution tail は `committed` の値に関わらず実行される。
`committed==false` が到達可能であれば、publish 内で破棄済みの world に対する
`didPublish` なしの `onPublishCompleted` 実行となり得る。
しかし到達条件（seqId==0 または commit Faulted）は producer 保証＋FIFO により
排除されており、反例を構成できないため計上しない。到達不能クレームの記録として残す。

---

## 3. 本 D17 の作業記録

```text
変更ファイル: 0 件（production / test / build / CMakeLists.txt / ConvoPeq.md すべて無変更）
追加ファイル: 本書 1 件

実施した操作:
  - 読み取り（git show / rg / Read）: 多数
  - ビルド / テスト: なし（ソース未変更のため不要）
  - reset / clean / rebase / amend / squash / force push: いずれも不使用

main checkout (C:\VSC_Project\ConvoPeq): 未改変（HEAD = bda43034）
作業 worktree: HEAD = a1ce01b2（= origin/main）
```
