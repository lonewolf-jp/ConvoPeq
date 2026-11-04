# Reader memory safety 再監査と修正（R-13・2026-11-04）

Owner 指示（次段階の残存技術リスク監査）に基づく、最新 main での Reader memory safety の再確認と、確認された欠陥の修正記録。

## 1. 最新 main の確認（Task 1）

| 項目 | 結果 |
|---|---|
| `main` / `origin/main` | `6eeb1d42` / `6eeb1d42`（一致） |
| 作業ツリー | clean（0 件）、worktree 1 本 |
| `ConvoPeq.md` | 6,476,830 B / sha256 `d8a0ce7d…`、Generated 2026-11-03 22:53:52、`--check` で **FRESH** |
| 残存課題 | 台帳 §11.4 のとおり（U5 有効化 未承認、R-101 初回ミス時、フレーク監視、`[U5ATTR]` 保持） |

## 2. Reader memory safety の現状（Task 2）

### 2.1 プロトコル（健全な部分）

| 要素 | 実装 |
|---|---|
| 公開 | `RuntimeStore::WriteAccess::publishAndSwap` が `acq_rel` exchange |
| 観測 | `RuntimeStore::observe` が `acquire` load |
| Reader 登録 | `RCUReader`/`ScopedEpochReader` が load の**前**に enter（`EpochDomain::enterReader` は epoch store → depth++ の順） |
| Retire | `markRetireEpoch()` = `publishEpoch()`（**epoch を前進させた値を retire epoch にする**） |
| 解放判定 | `DeferredDeletionQueue::reclaim` が `isOlder(retireEpoch, minReaderEpoch)` のときのみ deleter 実行 |
| 音声スレッド | `RCUReaderGuard` で `processBlock` 全体を保護し、enter 失敗時は fail-closed（V-11 修正） |

retire が epoch を前進させるため、reader が enter した時点の epoch `E` と、その後に retire される `R` には `E < R` が成立する。reader が scope 内にいる限り `minReaderEpoch ≤ E < R` となり、当該オブジェクトは解放されない。**この部分に欠陥はない。**

### 2.2 確認された欠陥 — 「load してから enter」の 3 箇所

| # | 箇所 | 内容 | 影響 |
|---|---|---|---|
| 1 | `AudioEngine::makeRuntimeReadHandle`（AudioEngine.h） | world を `consumeWorldHandle` で load・deref した後、末尾で `observeCurrentRuntime`（= enter）を呼んでいた | handle 経由の全 reader。audio は外側 guard で保護されるが、worker/UI 経路では load→deref の窓が未登録 |
| 2 | `RuntimePublicationOrchestrator::trySubmitImpl` | Message / **RebuildThread** / CoordinatorLoop worker から実行されるが、`observePublishedWorld()` に guard が無い | RebuildThread 実行時、message thread の reclaimer が world を解放し得る |
| 3 | `AudioEngine::makePublishDecisionSnapshot` | 同じく producer スレッドで `observePublishedWorld()` を無防備に読む | 同上 |

いずれも「enter → load → deref」という文書化された契約に反しており、`ScopedEpochReader.h` が説明する UAF クラスそのものである。message thread 上の読みは reclaimer と同一スレッドで直列化されるため顕在化しないが、#2/#3 は RebuildThread／worker で動くため実際に競合し得る。

## 3. 修正（Task 3）

| # | 修正 |
|---|---|
| 1 | `makeRuntimeReadHandle`: `observeCurrentRuntime` を world load の**前**へ移動。`rootEnterSucceeded()` が false のときは world を公開しない（fail-closed） |
| 2 | `trySubmitImpl`: 関数スコープで局所 `RCUReader`＋`RCUReaderGuard` を取得し、enter 成功時のみ world を読む（失敗時は `nullptr` として crossfade 判定を安全側に倒す） |
| 3 | `makePublishDecisionSnapshot`: 同様の局所 reader で保護（`const` を外し `getRetireRouter()` を使用） |
| 4 | 回帰防止: `ScopedEpochReaderStructuralTests` に「guard が world load より前」「fail-closed」の 4 チェックを追加 |

修正は **非 RT 経路のみ**。音声コールバックに新しい atomic・分岐は追加していない（#1 は enter の位置の移動で、enter 自体は従来も呼ばれていた）。

## 4. 検証

| 検証 | 結果 |
|---|---|
| Release ビルド | rc=0（AudioEngineHarness 588 s／アプリ全体 1,162 s） |
| `ScopedEpochReaderStructuralTests` | **PASS（15 チェック、新規 4 含む）** |
| `AudioEngineHarness` スイート | rc=0・297.12 s・100% tests passed（V11 reader テスト含む） |
| ASan / TSan（CI） | @@CI@@ |

## 5. 残存リスクと判断

- `getRuntimeGraph()`（呼び出し元なし）、`getActiveRuntimeSnapshot()`（テストのみ）、`currentPublicationEpoch()` など message-thread 専用の読みは、reclaimer と同一スレッドであるため直列化される。今回は変更していない。
- `RuntimeBuilder`／`Orchestrator` の world 読みは #2 で保護した。`makePublishDecisionSnapshot` は #3。
- スレッドモデル（message thread が reclaimer、RebuildThread が producer）を前提とした判断であり、将来 producer が増える場合は同じ契約（enter → load）を守る必要がある。構造テストがその一部を強制する。

## 6. 残課題

1. 同一スレッド前提の読み（`currentPublicationEpoch` 等）は、将来 worker から呼ばれる場合に guard が必要。呼び出し元の追加時に構造テストへ登録する。
2. `[U5ATTR]` は測定専用ブランチで保持（変更なし）。U5 製品有効化は未承認・既定 OFF。
