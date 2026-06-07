# EpochDomain 参照ステータス管理

> 最終更新: v18.13 (Phase-D完了) | 改修計画 doc/work21/refactoring_plan.md
> 本ファイルは P2-11 定義に基づき全EpochDomain参照のStatusを管理する。

## 分類凡例

| Status | 意味 |
|--------|------|
| **削除** | 完全に削除されたコード |
| **置換Router** | ISRRetireRouter 経由に移行 |
| **置換IEpochProvider** | IEpochProvider& 抽象インターフェース経由 |
| **置換IReader** | IReaderEpochProvider& 経由 |
| **Internal** | コンポーネント内部の正当な EpochDomain 利用 (EQProcessor独自ドメイン等) |
| **Deprecated** | [[deprecated]] でマーク、将来削除予定 |
| **Exposure** | 公開APIでの EpochDomain 型露出 (ISRRetireRouter内部3件) |
| **Comment** | コメント内参照のみ |

## CI Gate 数値 (Phase-D 完了時)

| 指標 | 値 | 目標 |
|------|----|------|
| advanceEpoch direct calls | 2 | 0 (許容) |
| enqueueRetire calls | 4 | 0 |
| EpochDomain type exposure | 3 | 0 (ISRRetireRouter内部) |
| EpochProvider total methods | 10 | ≤ 12 |
| IReaderEpochProvider methods | 7 | ≤ 10 |
| IPublicationProvider methods | 1 | ≤ 3 |
| IRetireProvider methods | 2 | ≤ 4 |
| IEpochProvider (facade) methods | 0 | 0 |
| ISRRetireRouter own public API | 4 | ≤ 8 |
| EpochDomain alias (using/typedef) | 0 | 0 |
| EpochDomain template param | 0 | 0 |

## ファイル別 Status 一覧

### Phase-A 削除対象 (全12箇所 → ✅ 完了)
- EpochCore.h — 削除 ✅
- EpochDomainReaderGuard 直接生成 — 全箇所 RCUReaderGuard に置換 ✅
- AudioEngine::epochDomain() — 削除 ✅
- RCUReader::domain() — 削除 ✅

### Phase-B 置換対象 (全8箇所 → ✅ 完了)
- DeletionQueue reclaim(EpochDomain&) → reclaim(uint64_t) ✅
- SnapshotRetireManager reclaim(EpochDomain&) → reclaim(uint64_t) ✅
- AudioEngine 3PR分割 (Publication/Reader/Retire) ✅

### Phase-C IEpochProvider抽象化 (全17+箇所 → ✅ 完了)
- ISRRetireRouter 作成 ✅
- IEpochProvider 抽出 ✅
- RCUReader → IEpochProvider ✅
- ObservedRuntime → IEpochProvider ✅
- SnapshotCoordinator → IEpochProvider ✅
- EQProcessor → ISRRetireRouter ✅
- AudioEngine::epochDomain() 完全削除 ✅

### Phase-D 責務分離 (全6タスク → ✅ 完了)
- RCUReader::domain()廃止 ✅
- RefCountedDeferred Router化 ✅
- ConvolverProcessor advanceEpoch集約 ✅
- IEpochProvider 3責務分離 ✅
- CI Gate 拡張 (alias/template/Router API監視) ✅
- EpochDomain_REFERENCE_STATUS.md 更新 ✅

### 残存 Internal (許容)
| ファイル | 内容 | Status |
|----------|------|--------|
| ISRRetireRouter.h:47 | `ISRRetireRouter(EpochDomain& epochDomain)` | Exposure (private相当) |
| ISRRetireRouter.h:183 | `EpochDomain& domain()` (private) | Exposure |
| ISRRetireRouter.h:189 | `EpochDomain* epochDomain_` (private) | Exposure |
| EQProcessor.h:422 | `convo::EpochDomain m_epochDomain` (独自) | Internal |
| EQProcessor.Core.cpp:73 | `m_epochDomain.advanceEpoch()` (独自) | Internal |
| AudioEngine.CtorDtor.cpp:121 | `m_epochDomain.advanceEpoch()` (dtor) | Internal |
| ConvolverProcessor.h:1152 | `m_epochDomain` (deprecated) | Deprecated |

### コメント内参照 (全実態と一致確認済み ✅)
- SnapshotCoordinator.h, ObservedRuntime.h, RefCountedDeferred.h,
  IReaderEpochProvider.h, IRetireProvider.h, ISRRetireRouter.h

| **許容(perm)** | EBR基盤として永続的に残す（関数単位で固定） | 恒久 |
| **許容(temp)** | 将来削除予定 — 終了時 temp=0 | Phase-B終了時 |
| **保留** | P0-14/P1-15完了後再評価 | Phase-B後 |
| **Dead** | コメントのみ/未使用コード | Phase-A |

## 1. AudioEngine 系 (8ファイル)

### AudioEngine.Threading.cpp (13参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 30 | `m_epochDomain.publish()` | 置換 | Router::publishEpoch()へ |
| 32 | `m_epochDomain.current()` | 置換 | Router::snapshotEpoch()へ |
| 35 | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |
| 40 | `m_epochDomain.enterReader()` | 許容(perm) | EBR基盤lock-free読取 |
| 42 | `m_epochDomain.exitReader()` | 許容(perm) | EBR基盤lock-free読取 |
| 46 | `m_epochDomain.enqueueRetire()` | 削除 | Router::enqueueRetire()へ |
| 48 | `m_epochDomain.reclaimRetired()` | 削除 | requestReclaim()へ |
| 52 | `m_epochDomain.reclaimRetired()` | 削除 | requestReclaim()へ |
| 55 | `m_epochDomain.reclaimRetired()` | 削除 | requestReclaim()へ |
| 60 | `m_epochDomain.activeReaderCount()` | 許容(perm) | EBR基盤診断用読取 |
| 65 | `m_epochDomain.enterReader()` | 許容(perm) | EBR基盤lock-free読取 |
| 67 | `m_epochDomain.exitReader()` | 許容(perm) | EBR基盤lock-free読取 |
| 71 | `m_epochDomain.current()` | 置換 | Router::snapshotEpoch()へ |

### AudioEngine.h (15参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 3225 | `EpochDomain& epochDomain() noexcept` | 削除 | Public accessor — P0-14対象 |
| 3381 | `convo::EpochDomain m_epochDomain` | 置換 | Router内部実装へ |
| 3382 | `RCUReader audioThreadRcuReader{m_epochDomain}` | 保留 | P0-14完了後評価 |
| 3611 | `snapshotRcuEpoch()` → current() | 置換 | Router::snapshotEpoch()へ |
| 3614 | `enterRcuReader()` → enterReader() | 許容(perm) | RCUReader限定API経由 |
| 3616 | `exitRcuReader()` → exitReader() | 許容(perm) | RCUReader限定API経由 |
| 3620 | `markRetireEpoch()` → advanceEpoch() | 置換 | Router::publishEpoch()へ |
| 3623 | `currentRetireEpoch()` → current() | 置換 | Router::snapshotEpoch()へ |
| 3626 | `advanceRetireEpoch()` → advanceEpoch() | 置換 | Router::publishEpoch()へ |
| 3630 | `enqueueRetireEpochBounded()` → enqueueRetire() | 削除 | Router::enqueueRetire()へ |
| 3633 | `activeEpochObserverCount()` → activeReaderCount() | 許容(perm) | EBR基盤診断用 |
| 3639 | `processDeferredReleases()` → reclaimRetired() | 削除 | requestReclaim()へ |
| 3179-3193 | enqueueRetire/reclaimRetired retry | 削除 | P0-5 retry禁止 |
| 1404 | `// EBR: Using RefCountedDeferred` | Dead | コメントのみ |
| — | CacheMap: `entry.second->release(...)` | 保留 | RefCountedDeferred依存 |

### AudioEngine.CtorDtor.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 115 | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |

### AudioEngine.Processing.AudioBlock.cpp (2参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 50-60 | RCUWrapper経由のepochアクセス | 許容(perm) | EBR基盤 |
| 56 | `EpochDomainReaderGuard(m_epochDomain, ...)` | 削除 | P1-18: RCUReader限定APIへ |

### AudioEngine.Processing.BlockDouble.cpp (2参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 52-62 | RCUWrapper経由のepochアクセス | 許容(perm) | EBR基盤 |
| 56 | `EpochDomainReaderGuard(m_epochDomain, ...)` | 削除 | P1-18: RCUReader限定APIへ |

### AudioEngine.h (RCU wrapper — 8関数)
| 関数 | Status | Reason |
|------|--------|--------|
| `snapshotRcuEpoch()` | 置換 | Router::snapshotEpoch()へ |
| `enterRcuReader()` | 許容(perm) | RCUReader限定API維持 |
| `exitRcuReader()` | 許容(perm) | RCUReader限定API維持 |
| `markRetireEpoch()` | 置換 | Router::publishEpoch()へ |
| `currentRetireEpoch()` | 置換 | Router::snapshotEpoch()へ |
| `advanceRetireEpoch()` | 置換 | Router::publishEpoch()へ |
| `enqueueRetireEpochBounded()` | 削除 | Router::enqueueRetire()へ |
| `processDeferredReleases()` | 削除 | requestReclaim()へ |

### AudioEngine.h (SnapshotCoordinator — 3関数)
| 関数 | Status | Reason |
|------|--------|--------|
| `(coordinator連動)` | 保留 | Phase-B評価 |
| — | — | — |

### RuntimePublicationOrchestrator.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 93 | `engine_.advanceRetireEpoch()` | 許容(temp) | チェーンC: 唯一の許容advanceEpoch呼び出し元 |

## 2. EQProcessor 系 (4ファイル, ~30参照)

### EQProcessor.Core.cpp (18参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 32-65 | retry ループ `kMaxRetry=4` | 削除 | P0-5: coordinator常時設定後fallback削除 |
| — | `m_retireCoordinator->enqueueRetire()` | 削除 | Router::enqueueRetire()→優先経路 |
| — | `m_epochDomain.enqueueRetire()` fallback | 削除 | coordinator設定完了後削除 |
| — | `m_epochDomain.reclaimRetired()` | 削除 | requestReclaim()へ |
| — | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |

### EQProcessor.Parameters.cpp (10参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 各パラメータsetter | `m_epochDomain.advanceEpoch()` ×10 | 置換 | バッチ化/1回に集約 → Router経由 |

### EQProcessor.h (2参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 103 | `#include "RefCountedDeferred.h"` | 保留 | P1-20: 置換後削除 |
| 117 | `EQCoeffCache : public RefCountedDeferred<EQCoeffCache>` | 保留 | P1-20: Router版検討 |

### EQProcessor.Coefficients.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| — | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |

## 3. ConvolverProcessor 系 (5ファイル, ~7+参照)

### ConvolverProcessor.h (3参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 1149 | `convo::EpochDomain m_epochDomain` | 削除 | P1-15: メンバ変数 |
| 1151 | `RCUReader runtimeRcuReader{m_epochDomain}` | 削除 | Router内部実装へ |
| — | `#include "core/EpochDomain.h"` | 削除 | 不要include |

### ConvolverProcessor.LoadPipeline.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 678 | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |

### ConvolverProcessor.StateAndUI.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 1017 | `m_epochDomain.advanceEpoch()` | 置換 | Router::publishEpoch()へ |

### ConvolverProcessor.Runtime.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| — | `m_epochDomain.current()` (RCU経由) | 許容(perm) | EBR基盤読取 |

### ConvolverProcessor.Lifecycle.cpp (1参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| — | コメント内参照 | Dead | コメントのみ |

## 4. SnapshotCoordinator 系 (2ファイル, 16参照)

### SnapshotCoordinator.h (8参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 150 | `EpochDomain* m_epochDomain` | 置換 | Router参照へ |
| — | advanceEpoch ×2 | 置換 | Router::publishEpoch()へ |
| — | current ×1 | 置換 | Router::snapshotEpoch()へ |
| — | enqueueRetire ×3 | 削除 | Router::enqueueRetire()へ |
| — | reclaimRetired ×2 | 削除 | requestReclaim()へ |

### SnapshotCoordinator.cpp (8参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| — | advanceEpoch ×2 | 置換 | Router::publishEpoch()へ |
| — | current ×2 | 置換 | Router::snapshotEpoch()へ |
| — | enqueueRetire ×2 | 削除 | Router::enqueueRetire()へ |
| — | reclaimRetired ×1 | 削除 | requestReclaim()へ |
| — | activeReaderCount ×1 | 許容(perm) | EBR基盤診断用 |

## 5. DeletionQueue 系 (3ファイル)

### DeletionQueue.h (1参照)
| 行 | 呼び出し | Status | Fate | Reason |
|----|---------|--------|------|--------|
| 18 | `void reclaim(const EpochDomain& core)` | 置換 | Keep | EpochDomain引数をepoch-free APIに置換 |

### DeletionQueue.cpp (1参照)
| 行 | 呼び出し | Status | Fate | Reason |
|----|---------|--------|------|--------|
| 23 | `EpochDomain::isOlder()` 内部呼び出し | 置換 | Wrap | reclaim()内部をRouter経由に |

### SnapshotRetireManager.h (4参照)
| 行 | 呼び出し | Status | Fate | Reason |
|----|---------|--------|------|--------|
| 43 | `void reclaim(const EpochDomain& domain)` | 置換 | Merge | SnapshotRetirePolicyに吸収 |
| — | コメント参照 ×3 | Dead | — | コメントのみ |

## 6. RCUReader 系 (2ファイル, 7参照)

### RCUReader.h (7参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 140 | `EpochDomain* epochDomain` (private) | 保留 | Router内部実装で保持継続 |
| 142 | `EpochDomain& domain() noexcept` | 削除 | P0-14: 限定APIに置換 |
| — | enter()/exit() ×2 | 許容(perm) | EBR基盤 |
| — | `EpochDomain&` コンストラクタ | 保留 | Routerのみから構築 |

### EpochDomain.h (3参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 228-275 | `EpochDomainReaderGuard` class定義 | 許容(perm) | クラス定義は維持(直接生成のみ禁止) |
| — | `#include` 関連 | 許容(perm) | コア定義 |
| — | ReaderSlot struct | 許容(perm) | EBR基盤 |

## 7. EpochDomainReaderGuard 直接生成 (3箇所)

| ファイル | 行 | Status | Reason |
|---------|----|--------|--------|
| ObservedRuntime.h:49 | `EpochDomainReaderGuard guard` (メンバ変数) | 削除 | P1-18: RCUReader限定APIへ |
| AudioEngine.Processing.AudioBlock.cpp:56 | `EpochDomainReaderGuard(...)` | 削除 | P1-18: RCUReader限定APIへ |
| AudioEngine.Processing.BlockDouble.cpp:56 | `EpochDomainReaderGuard(...)` | 削除 | P1-18: RCUReader限定APIへ |

## 8. その他 (3ファイル)

### ObservedRuntime.h (2参照)
| 行 | 呼び出し | Status | Reason |
|----|---------|--------|--------|
| 26 | `EpochDomain&` コンストラクタ引数 | 削除 | Router参照へ |
| 49 | `EpochDomainReaderGuard guard` | 削除 | P1-18 |

### EpochCore.h (1参照)
| 行 | 呼び出し | Status | Fate | Reason |
|----|---------|--------|------|--------|
| — | `#include "EpochDomain.h"` (空) | 削除 | Delete | 空のヘッダ |

### RefCountedDeferred.h (1参照)
| 行 | 呼び出し | Status | Fate | Reason |
|----|---------|--------|------|--------|
| 20 | `epochDomain.enqueueRetire()` | 削除 | Delete | 未使用テンプレート |

## 9. コメント内参照 (8+箇所)

| ファイル | Status | Reason |
|---------|--------|--------|
| SnapshotRetireManager.h ×3 | Dead | コメントのみ — v18.12: 実態と乖離チェック実施 |
| AudioEngine.h ×1 | Dead | コメントのみ |
| Init.cpp ×1 | Dead | コメントのみ |
| Globals.cpp ×1 | Dead | コメントのみ |
| ConvolverProcessor.Lifecycle.cpp ×1 | Dead | コメントのみ |
| EpochCore.h ×1 | Dead | コメントのみ |

## 10. advanceEpoch カウント (21回)

| # | ファイル | 関数/コンテキスト | Status |
|---|---------|-------------------|--------|
| 1 | AudioEngine.Threading.cpp:35 | `advanceEpoch()` in wrapper | 置換 |
| 2 | AudioEngine.CtorDtor.cpp:115 | `advanceEpoch()` in init | 置換 |
| 3-12 | EQProcessor.Parameters.cpp ×10 | 各パラメータsetter末尾 | 置換(バッチ化) |
| 13-17 | EQProcessor.Core.cpp ×5 | retryループ内 | 削除(retry削除) |
| 18 | EQProcessor.Coefficients.cpp ×1 | coeff更新時 | 置換 |
| 19 | ConvolverProcessor.LoadPipeline.cpp:678 | pipeline load時 | 置換 |
| 20 | ConvolverProcessor.StateAndUI.cpp:1017 | state更新時 | 置換 |
| 21 | EpochDomain.h (forwarding) | publish→advanceEpoch | 置換 |

## 11. 進捗サマリー

| Status | 現在 | Phase-A目標 | Phase-B目標 | Phase-B終了時目標 |
|--------|------|-------------|-------------|-------------------|
| 削除 | 30 | 0 | 0 | 0 |
| 置換 | 55 | 55 | 0 | 0 |
| 許容(perm) | 20 | 20 | 20 | 20 |
| 許容(temp) | 1 | 1 | 0 | 0 |
| 保留 | 5 | 5 | 0 | 0 |
| Dead | 13 | 13 | 0 | 0 |
| 合計 | 124 | 94 | 20 | 20(perm only) |
