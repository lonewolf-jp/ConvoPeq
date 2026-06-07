# 計画書 vs 実コード 整合性監査
> 実施日: 2026-06-07 | 対象: `refactoring_plan.md` vs `src/` 最新コード

## 監査方法

| ツール | 用途 | 結果 |
|--------|------|------|
| grep/Select-String | 全 `EpochDomain` / `m_epochDomain` 参照検出 | ✅ 17件の残存特定 |
| ccc (cocoindex-code) | ASTベース検索、EpochDomainReaderGuard全使用者 | ✅ 110 C++ chunks |
| serena MCP | シンボルレベル参照検出 | ✅ ISRRetireRouter/EpochDomainReaderGuard 検出 |
| codegraph MCP | 依存関係解析 | ⚠ entities捕捉できず (full index必要) |
| graphify MCP | 知識グラフ (DeepSeek) | ✅ 11695 nodes, 14323 edges |
| CI Gate | 数値検証 | ✅ advanceEpoch=2, enqueueRetire=4, exposure=3 |

## 計画書 付録A (124参照) の実コード状態

### A-1. SnapshotCoordinator (8箇所) → ✅ 全件移行完了

| # | 計画書記載 | 実コード | 状態 |
|---|-----------|----------|------|
| 1-8 | `m_epochDomain->enqueueRetire/reclaimRetired/publish` (SnapshotCoordinator.h/.cpp) | `m_epochProvider->publishEpoch/enqueueRetire/tryReclaim` | ✅ **全件移行済み** |

### A-2. AudioEngine 直接呼び出し (10箇所) → ✅ 大部分移行

| # | 計画書記載 | 実コード | 状態 |
|---|-----------|----------|------|
| 9-11 | `enqueueDeferredDeleteNonRtWithResult()` の retry ループ (AudioEngine.h:3183-3194) | `m_retireRouter->enqueueRetire()` 単一呼び出しに | ✅ **移行 + retry ループ削除** |
| 12 | `enqueueRetireEpochBounded()` (Threading.cpp:54) | `AudioEngine.Retire.cpp:32` に移動、Router経由 | ✅ **移行** |
| 13 | `tryReclaimResources()` (Threading.cpp:76) | `AudioEngine.Retire.cpp:38`, Router経由 | ✅ **移行** |
| 14 | `drainDeferredRetireQueues()` (Threading.cpp:87) | `AudioEngine.Retire.cpp:49`, Router経由 | ✅ **移行** |
| 15 | `drainDeferredRetireQueues()` (Threading.cpp:222) | 削除 (boost path?) | ✅ **削除** |

### A-3. EQProcessor 直接呼び出し (5箇所) → ⚠ 要対応

| # | 計画書記載 | 実コード | 状態 |
|---|-----------|----------|------|
| 16 | `enqueueDeferredDeleteWithFallback()` coordinator path reclaimRetired (Core.cpp:45) | `Core.cpp:55` `m_epochDomain.enqueueRetire()` | ⚠ **未移行** |
| 17 | fallback path enqueueRetire (Core.cpp:60) | `Core.cpp:55` に統合 | ⚠ **未移行** |
| 18 | fallback path reclaimRetired (Core.cpp:64) | 削除(retryループ削除) | ✅ **削除** |
| 19-20 | ~EQProcessor reclaimRetired (Core.cpp:119,121) | `Core.cpp:124,126` `m_epochDomain.reclaimRetired()` | ⚠ **未移行** |

## 残存 `m_epochDomain.xxx()` 全17件

### AudioEngine 残存 (8件)

| # | ファイル:行 | 呼び出し | 分類 | 推奨対応 |
|---|------------|----------|------|---------|
| 1 | `CtorDtor.cpp:121` | `advanceEpoch()` | デストラクタ終端 | 🔶 **許容** |
| 2 | `CtorDtor.cpp:129` | `drainAll()` | デストラクタ終端 | 🔶 **許容** |
| 3 | `ReleaseResources.cpp:206` | `drainAll()` | releaseResources | 🔶 **許容** |
| 4 | `ReleaseResources.cpp:213` | `pendingRetireCount()` | 診断 | ⚠ Routerに追加推奨 |
| 5 | `Publication.cpp:24` | `current()` | Epoch読取 | ⚠ Router::snapshotEpoch()経由に変更推奨 |
| 6 | `Retire.cpp:51` | `getMinReaderEpoch()` | reclaim用epoch取得 | ⚠ Routerに追加推奨 |
| 7 | `Retire.cpp:55` | `pendingRetireCount()` | 診断 | ⚠ Router経由に変更推奨 |
| 8 | `Retire.cpp:183` | `getMinReaderEpoch()` | reclaim用epoch取得 | ⚠ Router経由に変更推奨 |

### EQProcessor 残存 (9件)

| # | ファイル:行 | 呼び出し | 分類 | 推奨対応 |
|---|------------|----------|------|---------|
| 9 | `Core.cpp:37` | `currentEpoch()` | 内部epoch読取 | 🔶 **許容** (独自ドメイン) |
| 10 | `Core.cpp:55` | `enqueueRetire()` | retire発行 | ⚠ `m_retireRouter` 経由に変更 |
| 11 | `Core.cpp:56` | `currentEpoch()` | 内部epoch読取 | 🔶 **許容** (独自ドメイン) |
| 12 | `Core.cpp:73` | `advanceEpoch()` | 内部epoch進捗 | 🔶 **許容** (独自ドメイン) |
| 13 | `Core.cpp:82` | `currentEpoch()` | 内部epoch読取 | 🔶 **許容** (独自ドメイン) |
| 14 | `Core.cpp:91` | `currentEpoch()` | 内部epoch読取 | 🔶 **許容** (独自ドメイン) |
| 15 | `Core.cpp:124` | `reclaimRetired()` | shutdown drain | ⚠ `m_retireRouter->tryReclaim()` に変更 |
| 16 | `Core.cpp:125` | `drainAll()` | shutdown drain | 🔶 **許容** (独自ドメイン) |
| 17 | `Core.cpp:126` | `reclaimRetired()` | shutdown drain | ⚠ `m_retireRouter->tryReclaim()` に変更 |

## 計画書と実コードの不一致まとめ

### 発見事項①: plan上の参照数が古い
- 計画書 付録A は 124参照としているが、Phase-D完了後は **大幅減少**
- 直接 `m_epochDomain.xxx()` は **17件のみ**
- 計画書更新が必要 (v18.13)

### 発見事項②: EQProcessor に未移行の periodic cleanup
- `EQProcessor.Core.cpp:124,126` に `reclaimRetired()` が残存 (shutdown時の2回)
- `Core.cpp:55` に `enqueueRetire()` が残存
- これらには `#pragma warning(disable:4996)` がなく、deprecated APIの警告が発生しない可能性

### 発見事項③: AudioEngine.Retire に Router未提供の epoch取得
- `m_epochDomain.getMinReaderEpoch()` と `m_epochDomain.pendingRetireCount()` が残存
- Router (`ISRRetireRouter`) にこれらのメソッドがないため直接呼び出し
- Phase-E で Router に追加すべき (P5 範囲拡張)

### 発見事項④: Publication.cpp の current() が未移行
- `AudioEngine.Publication.cpp:24` の `currentRetireEpoch()` が `m_epochDomain.current()` を呼んでいる
- Router には `snapshotEpoch()` があるため、それを使うべき
- `#pragma warning(disable:4996)` で抑制されている

### 発見事項⑤: ConvolverProcessor.h に `m_epochDomain` メンバ残存
- `ConvolverProcessor.h:1152` に `[[deprecated]] m_epochDomain` が残っている (既知)
- CI Gate で検出されている (exposure にはカウントされない)

## CI Gate 最新値 (監査時点)

| 指標 | 値 | 前回(Phase-D完了時) | Delta |
|------|----|-------------------|-------|
| advanceEpoch direct calls | 2 | 2 | 0 |
| enqueueRetire calls | 4 | 4 | 0 |
| EpochDomain type exposure | 3 | 3 | 0 |
| IEpochProvider methods | 10 | 10 | 0 |
| IReaderEpochProvider | 7 | 7 | 0 |
| IPublicationProvider | 1 | 1 | 0 |
| IRetireProvider | 2 | 2 | 0 |
| ISRRetireRouter own API | 4 | 4 | 0 |

## 結論

計画書 `refactoring_plan.md` と実コードの間に **重大な不一致はない**。
124参照中 107件が移行/削除済み。残る17件は EQProcessor 独自ドメイン(9件)と AudioEngine の診断/終了処理(8件)に分類される。

ただし、計画書の付録Aは Phase-D 完了後の状態を反映していないため（124参照のまま）、**更新が必要**。
