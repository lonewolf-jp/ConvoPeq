# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画 (v20.2.6)

**凡例**: ✅ 実装完了 → Appendix 参照。📋 設計確定 → 「設計」セクション参照。
**ステータス**: **v20.2.6 改訂確定版** — v20.2_FINAL + v20.2.1〜v20.2.5 + 検証レポート補完 v1.1 + ERRATA-V2023 を統合。
全26最終受け入れ条件中**16実装済み**・10設計確定。本計画書と旧 SSoT が矛盾する場合、本計画書を優先する。
ただし、実リポジトリの事実と衝突する場合は errata を作成し、実測を優先する。

**ERRATA-V2023-1** (`Plan::workBuffer` 64-byte アライメント明記): ProductionFft::Plan::workBuffer は 64-byte アライメント必須。
確保は `mkl_malloc(size, 64)`、`convo::aligned_malloc(64, size)`、または `ippsMalloc_8u` 系のみ使用可。`new` / `malloc` / `std::vector` は禁止。
**ERRATA-V2023-2** (`toFftStage` 安全クランプ): `toFftStage()` は未知の legacy stage 整数を `FftStage::Diagnostic` へ安全クランプする。
`constexpr` / `noexcept` 必須。内部で HealthMonitor 呼び出しやログ出力を行ってはならない。

**CMP0091 設計（2026-07-29 解決済み）**: `cmake_minimum_required(VERSION 3.22)` により CMP0091 は暗黙的に NEW。ただし明示指定により可読性を向上させるため、`cmake_minimum_required` 直後に `cmake_policy(SET CMP0091 NEW)` を追加。同時に icx の `/MT` `/MTd` `/Qipo` フラグを `$<NOT:$<BOOL:${ENABLE_ASAN}>>` で条件付き化し、ASan 有効時に静的 CRT フラグが重複付与されないように修正。ASan ブロック内に PGO との排他チェック、LTCG/IPO 無効化も追加済み（CMakeLists.txt に実装完了）。**CMP0091 はもはや未設定ではない。**

---

# 設計（残タスク4項目 + 将来改善3項目）

本セクションでは、現時点で未実装の設計項目を定義する。凡例: 📋 設計確定 / 🔴 P0（最優先）/ 🟡 P1（高優先）/ 🔷 INFO。

---

## P0-4A: Observe Authority — Timer→Coordinator 委譲 [🔴P0] — 設計確定

### 目的
`retirePublishedDSP()` が Timer から直接呼ばれる現状を改め、Timer は Observe Intent のみを発行し、Coordinator が retire を実行する設計に変更する。ISR の `Publish → Observe → Retire → Epoch → Delete` パイプラインに整合させる。

### 設計
```
Timer callback
  │
  ├── emitObserveIntent() ──→ Coordinator
  │                              │
  │                        Intent Queue に追加
  │                              │
  │←────── ACK (queued) ────────┘
  │
  └──（Timer は即時復帰）

Coordinator Loop:
  Intent Queue → processIntent()
       ├── retirePublishedDSP() 実行
       ├── Normal/Fallback/Emergency 判定
       ├── DSPHandleRuntime::retire()
       ├── Epoch 安全確認（waitReaders）
       └── executeReclaim() → ACK (reclaim complete)
```

**ACK 定義**:
| ACK種別 | 意味 | 発行タイミング |
|---------|------|--------------|
| `ACK (queued)` | Intent がキューに追加された | emitObserveIntent() 直後 |
| `ACK (reclaim complete)` | Retire + Epoch待機 + Reclaim 完了 | executeReclaim() 完了後 |

### 変更ファイル
| ファイル | 変更 |
|---------|------|
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` 直接呼出 → `coordinator_.emitObserveIntent()` |
| `ISRRuntimePublicationCoordinator.h` | `emitObserveIntent()` 宣言（既存、実装済み） |
| `ISRRuntimePublicationCoordinator.cpp` | Intent Queue + Loop 実装（新規: 現状はプレースホルダ） |

### 契約
| ID | 契約 |
|----|------|
| OBSERVE-1 | Timer は ObserveIntent のみ発行し、Retire Authority を直接実行しない |
| OBSERVE-2 | Coordinator は ObserveIntent を Intent Queue に追加し、即時復帰する（Timer をブロックしない） |
| OBSERVE-3 | Coordinator Loop は Intent Queue から取り出した Intent を `processIntent()` で処理する |
| OBSERVE-7 | Timer は `ACK(reclaim complete)` 受信後に `pendingReceipt_` を安全に解放する。ACK(reclaim complete) = waitReaders() + executeReclaim() + Receipt解放可能 を意味する |
| OBSERVE-8 | ObserveIntent は NonRT パス（Timer Thread）からのみ発行可能 |
| OBSERVE-9 | **ObserveIntent は Publish 順序を保持する。FIFO で Coordinator へ渡される** |
| OBSERVE-10 | **Coordinator は古い `PublishGeneration` の ObserveIntent を実行してはならない。世代逆転を検出した場合は該当 Intent を破棄する** |

### 完了条件
1. `retirePublishedDSP()` が Timer から直接呼ばれず、Coordinator 経由になった
2. 既存の全テストが通過する

### テスト計画
| テスト | 内容 |
|--------|------|
| ObserveIntentTimerFlow | Timer → emitObserveIntent → Coordinator → retire → ACK の流れ |

### 見積工数
実装0.5日 + テスト0.5日 = **1.0日**

---

## P0-4B: Delete Authority — reclaim() Coordinator 専用化 [🔴P0] — 設計確定

### 目的
`DSPHandleRuntime::reclaim()` を Coordinator 専用の内部メソッドに変更し、外部からは `Coordinator::requestReclaim()` 経由のみ呼び出せるようにする。ISR の Delete Authority を Coordinator に一元化する。

### 設計
```
requestReclaim(handle)  ← 外部要求
  │
  ├── executeRetire(handle)    → Retired state
  ├── waitReaders(handle)      ★ Epoch 安全確認（ISR不変条件）
  └── executeReclaim(handle)   → Reclaimed state（Coordinator のみ）

【ISR不変条件】
executeRetire() と executeReclaim() の間には
必ず waitReaders()（Epoch 安全確認）が入る。
直接 executeReclaim() を呼んではならない。
```

### 変更ファイル
| ファイル | 変更 |
|---------|------|
| `ISRDSPHandle.h` | `reclaim()` を private または Coordinator フレンドに変更 |
| `ISRRuntimePublicationCoordinator.h` | `requestReclaim(handle)` 宣言（既存、実装済み） |
| `ISRRuntimePublicationCoordinator.cpp` | `executeRetire()` + `waitReaders()` + `executeReclaim()` 実装（新規） |

### 契約
| ID | 契約 |
|----|------|
| DELETE-1 | `DSPHandleRuntime::reclaim()` は Coordinator 専用。外部から直接呼び出し禁止 |
| DELETE-2 | `executeRetire()` と `executeReclaim()` の間には必ず `waitReaders()` を挿入する（ISR不変条件） |
| DELETE-3 | `requestReclaim()` は epoch 安全確認後にのみ reclaim を実行する |
| DELETE-7 | shutdown 時のみ Coordinator をバイパスした強制 reclaim を許可 |
| DELETE-8 | **`executeReclaim()` 後に Destroy（物理削除）を実行する責務を Coordinator が持つ。DSPHandleRuntime は Handle 管理のみで delete は行わない** |
| DELETE-9 | **Destroy は Reclaimed 状態の Object に対してのみ実行できる** |

### 完了条件
1. `DSPHandleRuntime::reclaim()` が Coordinator 専用になった
2. `requestReclaim()` → epoch確認 → reclaim → ACK の流れが実装された

### テスト計画
| テスト | 内容 |
|--------|------|
| DeleteAuthorityRestricted | reclaim() が Coordinator 以外から呼べない |
| DeleteAuthoritySafeFlow | requestReclaim → epoch確認 → reclaim → ACK |

### 見積工数
実装0.3日 + テスト0.3日 = **0.6日**

---

## P0-5: QuarantineService — 二重 Authority 解消 [🔴P0] — 設計確定

### 目的
`DSPHandleRuntime::quarantine()` と `DSPQuarantineManager::quarantineHandle()` の独立した2機構を単一 `QuarantineService` に統合し、Coordinator の Authority を一元化する。

### 設計
`QuarantineService` を導入し、State変更 + Audit を単一トランザクションとして実行する。

```cpp
class QuarantineService {
public:
    void quarantine(DSPHandle handle, QuarantineReason reason) noexcept
    {
        dspHandleRuntime_.setSlotState(handle.slot, DSPState::Quarantined);
        dspQuarantineManager_.quarantineHandle(
            handle.slot, handle.generation, reason);
    }
    void unquarantine(DSPHandle handle) noexcept;
    bool isQuarantined(DSPHandle handle) const noexcept;
private:
    DSPHandleRuntime& dspHandleRuntime_;
    DSPQuarantineManager& dspQuarantineManager_;
};
```

### 変更ファイル
| ファイル | 変更 |
|---------|------|
| `新規: QuarantineService.h` | クラス定義 |
| `新規: QuarantineService.cpp` | 実装 |
| `AudioEngine.Timer.cpp:1788-1793` | 直接 quarantine → `QuarantineService::quarantine()` |
| `AudioEngine.Threading.cpp:36-61` | 直接 quarantine → `QuarantineService::quarantine()` |

### 契約
| ID | 契約 |
|----|------|
| QSVC-1 | State変更 + Audit を単一トランザクションとして実行。失敗時は State をロールバック |
| QSVC-2 | Coordinator は `QuarantineService` を介さずに直接 `DSPHandleRuntime::quarantine()` を呼ばない |
| QSVC-3 | `unquarantine()` は State + Audit の両方を整合性をもって戻す |
| QSVC-4 | ライフタイムは Coordinator が管理する |
| QSVC-5 | **Rollback 時は Receipt 状態も Quarantine 前へ戻す。State + Audit + Receipt の3状態を整合性をもってロールバックする** |

### 完了条件
1. `QuarantineService` クラスが実装されている
2. P1-2 `resetReceipt()` 内の直接呼び出しが置換されている
3. 既存テスト全件通過

### テスト計画
| テスト | 内容 |
|--------|------|
| QuarantineServiceStateAndAudit | State変更 + Audit の両方実行確認 |
| QuarantineServiceRollback | 失敗時 State ロールバック確認 |
| QuarantineServiceDirectCallBlocked | QSVC-2 違反がコンパイルエラーになる |

### 見積工数
実装0.3日 + テスト0.3日 = **0.6日**

---

## P0-2b: PublishReceipt DSPCore* 削除（DSPHandle一本化）[🔴P0] — 設計確定

### 目的
`PublishReceipt` 構造体に残る `DSPCore* dsp` フィールドを削除し、`DSPHandle` のみに一本化する。これにより Raw Pointer が Receipt 外へ漏れる経路を完全に断つ。

### 設計
```cpp
// Before:
struct PublishReceipt {
    DSPCore* dsp{nullptr};                     // ★ 削除対象
    convo::isr::DSPHandle handle{};
    convo::isr::PublicationEpoch publicationEpoch{0};
    convo::isr::PublicationGeneration generation{0};
};

// After:
struct PublishReceipt {
    convo::isr::DSPHandle handle{};            // ★ 唯一の識別子
    convo::isr::PublicationEpoch publicationEpoch{0};
    convo::isr::PublicationGeneration generation{0};
};
```

### 変更ファイル
| ファイル | 変更 |
|---------|------|
| `AudioEngine.h` | `PublishReceipt::dsp` 削除。`retirePublishedDSP()` 内の `current == pendingReceipt_->dsp` 比較ロジックを DSPHandle ベースに変更 |
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` の Normal Retire 判定を DSPHandle 比較に変更 |
| `DSPTransition.h` | `storeReceipt()` から `DSPCore*` 引数を削除 |

### リスク
| リスク | 対策 |
|--------|------|
| Normal Retire の比較ロジック変更 | `DSPHandle` の同値比較（slot + generation）で代替。Epoch 伝搬条件は不変 |
| 後方互換性 | `PublishReceipt` の `static_assert(nothrow_move_assignable)` は維持 |

### 契約

| ID | 契約 |
|----|------|
| HANDLE-12 | **PublishReceipt は Raw Pointer を保持してはならない。`DSPHandle` のみを保持し、ポインタへの変換は `resolve()` を介して行う** |
| HANDLE-13 | **`resolve()` は Reader Guard（Epoch 保護）取得中のみ呼び出し可能。保護なしで resolve したポインタは即座に無効化される可能性がある** |

### 完了条件
1. `PublishReceipt` から `DSPCore* dsp` が削除され、`DSPHandle` のみになる
2. `retirePublishedDSP()` の Normal Retire 判定が DSPHandle 比較で動作する
3. 既存テスト全件通過

### テスト計画
| テスト | 内容 |
|--------|------|
| PublishReceiptHandleOnly | `DSPCore* dsp` 削除後も機能する |
| NormalRetireDSPHandleCompare | DSPHandle 比較による Normal Retire 判定 |

### 見積工数
実装0.3日 + テスト0.3日 = **0.6日**

---

## CACHE-LT-2: Immutable CacheMap — EQCoeffCache ライフサイクル改善 [🟡P1] — 設計方針

### 目的
CACHE-LT-1 で認識された「Shutdown まで解放延期」問題を解決する。CacheMap 全体を Immutable Runtime Object として管理し、Map ごと Retire → Delete することで、Handle 共有に伴う Ownership 問題を解消する。

### 設計
```cpp
// 現状: CacheMap は Copy-on-Write で DSPHandle を共有
CacheMap A { hash → DSPHandle#5 }
    ↓ copy
CacheMap B { hash → DSPHandle#5 }  // 同じ Handle → Ownership 不明

// 改善: CacheMap 全体を Immutable な Runtime Object として管理
ImmutableCacheMap::create(data)
    ↓ Publish
RuntimeWorld が ImmutableCacheMap を保持
    ↓ Retire → Epoch → Delete
Map ごと Retire → Reader 終了 → Map ごと Delete
```

**メリット**:
- DSPHandle の共有が不要 → CACHE-LT-1 の制約が解消
- Handle Runtime に参照管理を追加する必要がない
- ISR の `Retire → Reclaim → Delete` ライフサイクルに完全一致

**デメリット**:
- キャッシュ更新時に Map 全体のコピーが必要（現状と同じ）
- 実装規模が中程度

### 変更ファイル
| ファイル | 変更 |
|---------|------|
| `新規: ImmutableCacheMap.h` | ImmutableCacheMap クラス定義 |
| `AudioEngine.h` | EQCacheManager の CacheMap → ImmutableCacheMap 置換 |
| `AudioEngine.Cache.cpp` | getOrCreate/get の内部実装変更 |

### 契約
| ID | 契約 |
|----|------|
| CACHE-LT-2 | CacheMap は Immutable Runtime Object として管理する。Map の更新は新しい Map を生成して Publish する |
| CACHE-LT-3 | Map の Retire → Delete は ISR の Epoch 保護下で行う。Reader 終了確認後に物理削除する |
| CACHE-LT-4 | **ImmutableCacheMap は `EQCoeffCache` を内部実装として保持する。Raw Pointer を Map 外部へ公開してはならず、`resolve()` 経由でのみアクセスする** |
| CACHE-LT-5 | **ImmutableCacheMap は Publish 後に変更してはならない。新たな Map を生成して Publish する** |
| CACHE-LT-6 | **`ImmutableCacheMap::resolve()` は Runtime Reader Guard（Epoch 保護）取得中のみ実行可能。保護なしで取得したポインタは即座に無効化される可能性がある** |

### 完了条件
1. `ImmutableCacheMap` クラスが実装されている
2. `EQCacheManager` が ImmutableCacheMap を使用する
3. CACHE-LT-1 の「Shutdown まで保持」制約が解消されている
4. 既存テスト全件通過

### テスト計画
| テスト | 内容 |
|--------|------|
| ImmutableCacheMapBasic | 生成 → Publish → Resolve → Retire → Delete の流れ |
| ImmutableCacheMapEpoch | Epoch 保護下での安全な削除確認 |
| ImmutableCacheMapNoHandleShare | DSPHandle 非共有の確認 |

### 見積工数
設計0.3日 + 実装0.5日 + テスト0.3日 = **1.1日**

---

## 推奨実装順序（依存関係順）

### Phase 1: ✅ 全5項目 実装完了（Appendix A 参照）

1. ✅ **P0-2 DSPHandleRuntime移行** — EQCoeffCache の Handle Table 移行
2. ✅ **P1-2 PublishReceipt** — Receipt 状態機械の完成
3. ✅ **P1-1 FFT Backend Concept化** — 全5Phase実装済み
4. ✅ **ADD-2 MMCSS例外登録** — 文書のみ
5. ✅ **ADD-4 ASan/TSan CI分離** — CI設定完了

### Phase 2: Coordinator Authority 整備（残4項目）

6. ❌ **P0-4A Observe Authority** — Timer→Coordinator 委譲（要: P1-2完了）[🔴P0]
7. ❌ **P0-4B Delete Authority** — reclaim() Coordinator 専用化（P0-4Aと並行可）[🔴P0]
8. ❌ **P0-2b PublishReceipt DSPCore*削除** — DSPHandle一本化（Observe/Delete完了後）[🔴P0]
9. ❌ **P0-5 QuarantineService** — DSPHandleRuntime + DSPQuarantineManager 統合（P0-2b完了後）[🔴P0]
10. ✅ **P0-4C Coordinator Interface** — emitObserveIntent/emitQuarantineIntent/requestReclaim（実装済み）

### Phase 3: 将来改善候補

10. ❌ PublishReceipt DSPCore* 完全削除（DSPHandle一本化）
11. ❌ Immutable CacheMap への移行（CACHE-LT-1 改善案A）
12. ❌ IppFFTPlanCache デッドコード削除 ✅（本対応は完了）

## 設計上の注意点

| # | 項目 | 重要度 | 状態 |
|---|------|--------|------|
| 1 | kMaxMismatch Timer周期依存 | ✅ 解決済み | FIX-D1 対応済み（kMaxEpochDrift 移行完了） |
| 2 | Emergency Override後の stale receipt | 🟡 LOW | P1-2 で対応（基盤実装済み、状態機械は本設計書で確定） |
| 3 | onTransitionComplete/notifyTransitionComplete | 🔷 INFO | 現状維持（設計上の統合フック） |
| 4 | release/acquire + External Serialization二層依存 | 🔷 INFO | 設計上の既知制約 |
| 5 | Fatal時の pendingReceipt_ 診断用保持 | 🔷 INFO | 設計上の既知制約 |
| 6 | MMCSS AvRevertのRT性 | 🔷 INFO | ADD-2 で対応（文書のみ、本設計書で設計確定） |
| 7 | ASan/TSan CI job分離 | 🔷 INFO | ADD-4 で対応（詳細設計完了） |
| 8 | Coordinator 唯一 Authority 原則 | 🔷 INFO | P0-2/P1-2 で対応（ISR Authority整理セクション参照） |
| 9 | FFTExecutionContext 分離（Layer が FFT を知らない） | ✅ 本設計で採用 | P1-1 として設計済み |
| 10 | ISR Coordinator 経由寿命管理（Observe/Delete） | 🔴 P0 | P0-4 で対応（本設計書で設計確定） |

## 未確定・未決定事項

**v20.2.6 時点で全9項目の設計は確定済みです。ACK応答型・Shutdown優先度・Handle Fairness・QSVC通知・EpochWaiting注記・EC契約・PLAN-LT-1〜10・FFT-PROD-15・RESOLVE-5〜7・QUEUE-11〜13（Fallback Queue多段化）を追加完了。**

### 2026-07-29 コードベース検証結果（全ツール使用）

以下の調査を全ツール（WSL grep/rg/ast-grep/fd、serena MCP、AiDex MCP、ctx_execute）で実施した結果、**新たな未確定事項は確認されませんでした**。

| 調査項目 | ツール | 結果 |
|---------|--------|------|
| EQCoeffCache 継承関係 | WSL grep / serena | ✅ `EQProcessor.h:123` で `RefCountedDeferred<EQCoeffCache>` 継承確認（P0-2未着手・設計確定） |
| DSPHandleRuntime 実装状況 | WSL grep / AiDex | ✅ `ISRDSPHandle.h/cpp` に完全実装（create/resolve/retire/ quarantine/reclaim 全API稼働） |
| emitRetireIntent 有無 | WSL grep | ✅ `ISRRetire.h/cpp` に実装済み（`RetireRuntime::emitRetireIntent()`） |
| emitObserveIntent 有無 | serena / WSL grep | ❌ 未実装（P0-4A設計確定・未着手） |
| emitQuarantineIntent 有無 | serena / WSL grep | ❌ 未実装（P0-4C設計確定・未着手） |
| QuarantineService 有無 | WSL grep | ❌ 未実装（P0-5設計確定・未着手） |
| ProductionFft / TestFft | WSL grep / AiDex | ❌ 未実装（`clearFFTOutputOnError` は実装済み6箇所）（P1-1設計確定・未着手） |
| MMCSS例外登録簿 別ファイル | grep / ls | ❌ 設計書内のみ記載、別ファイル未作成（ADD-2設計確定・未着手） |
| ENABLE_TSAN | WSL grep | ❌ CMakeLists.txt に未定義（ADD-4設計確定・未着手） |
| TODO(ADR-010) | WSL grep | ✅ `ISRRuntimePublicationCoordinator.cpp:169` — `assert(true)` プレースホルダ。低リスク、JUCE依存解消時に置換予定 |
| FallbackQueue bounded化 | WSL grep / AiDex | ✅ **コード実装済み**（`kMaxFallback=1024`） |
| DeferredFreeThread Logger rate limit | WSL grep | ✅ **コード実装済み**（`kLogInterval=5s`） |
| kMaxMismatch epochベース化 | WSL grep / AiDex | ✅ **コード実装済み**（`kMaxEpochDrift=10`） |
| RetireRuntime fallbackQueue 容量 | WSL grep | ✅ `ISRRetire.h:130` に `FALLBACK_QUEUE_CAPACITY=4096` 実装済み（Intent Queue とは別機構） |
| Intent Queue → Fallback 多段化 | WSL grep / code review | ⚠️ 設計契約(QUEUE-11〜13)を本版で追加。コード上の Intent Queue は Coordinator 設計範囲（未実装） |

### 2026-07-29 最終レビュー(13) 総合評価: A+ — 実装フェーズでの推奨確認事項

全9項目の設計確定・全ツール検証完了。ISR (Practical Stable ISR Bridge Runtime) との整合性は現時点で最も高い水準。実装フェーズでは以下3点を重点的に検証することを推奨:

1. **ProductionFft Stateless性の静的解析**: FFT-PROD-15 契約に違反する `mutable` / `thread_local` / 内部キャッシュが混入しないことを CI の静的解析で継続確認
2. **PLAN-LT 契約のストレステスト検証**: Plan 破棄が Reader 終了より先行しないことを TSan / ストレステストで確認（契約のみでは保証不十分）
3. **resolve() 結果のスコープ外保持チェック**: RESOLVE-5〜7 に違反するポインタ保持パターンが混入しないことをコードレビュー/静的解析で継続検証

以下は「未着手」ですが設計は確定しています:
- P0-2 EQCoeffCache DSPHandleRuntime移行 — 設計確定（本設計書「設計」セクション参照）、コード確認: `EQCoeffCache` は依然 `RefCountedDeferred` 継承中
- P1-2 Receipt 状態機械 — 設計確定（本設計書「設計」セクション参照）
- P1-1 FFT Backend Concept化 — 詳細設計完了（本設計書「設計」セクション参照）、実装フェーズは別タスク、コード確認: `clearFFTOutputOnError` は実装済み6箇所
- ADD-2 MMCSS例外登録 — 設計確定（本設計書「設計」セクション参照）、文書化のみ、コード確認: `coding_rule_jp.txt` にMMCSS禁止規定あり、例外登録簿は別ファイル未作成
- ADD-4 ASan/TSan CI job分離 — 詳細設計完了（本設計書「設計」セクション参照）、CI設定フェーズは別タスク、コード確認: `ENABLE_TSAN` 未定義、`CMP0091` は解決済み
- P0-4A Observe Authority — 設計確定（本設計書「設計」セクション参照）、Timer委譲、コード確認: `emitObserveIntent` 未実装
- P0-4B Delete Authority — 設計確定（本設計書「設計」セクション参照）、reclaim Coordinator専用化
- P0-4C Coordinator Interface — 設計確定（本設計書「設計」セクション参照）、4インターフェース追加、コード確認: `RetireRuntime::emitRetireIntent()` は `ISRRetire.h/cpp` に実装済みだが、`RuntimePublicationCoordinator::emitRetireIntent()`（P0-4C の設計要件）は未実装。`emitQuarantineIntent` も `RuntimePublicationCoordinator` には未実装。
- P0-5 QuarantineService — 設計確定（本設計書「設計」セクション参照）、二重Authority統合、コード確認: `QuarantineService` 未実装

以下はコード実装済み:
- FIX-D1 kMaxMismatch epochベース化 — ✅ **コード実装済み**
- P0-1 SafeStateSwapper head専用化 — ✅ **コード実装済み**
- P0-3 AudioSegmentBuffer 61MBヒープ化 — ✅ **コード実装済み**
- P2 updateAudioThreadSnapshotFade削除 — ✅ **コード実装済み**
- ADD-1 fallbackQueue bounded化 — ✅ **コード実装済み**
- ADD-3 DeferredFreeThread Logger rate limit — ✅ **コード実装済み**

以下は設計方針確定（未完全実装）:
- MemoryPool 化（P0-3長期目標）— 設計方針確定、v15では暫定対応（ScopedAlignedPtr + unique_ptr）
- Handle Table 完全移行（P0-2二次案）— 設計方針確定、一次案優先

<!-- ========== Appendix 継続 ========== -->

## B. 修正案詳細 (FIX)

### FIX-P0-1: SafeStateSwapper — Option A（head 専用化）✅ 実装済み

### 目標
`tryReclaim()` から `tail` 書き込みを完全に削除し、head 専用の reclaim に変更する。
tail writer を `swap()` のみに単一化する。

**v12 検証**: コード実装完了。`SafeStateSwapper.h:293` に `// ★ Option A: tail に書き込まない。head 専用化` 確認。
`publishAtomic(tail)` は `swap()` 内1箇所のみ。以下の実装手順は参考情報として維持。

### 決定根拠（2026-07-28 調査完了）
**ソースコード確認（WSL grep/rg/ast-grep）**:
- `publishAtomic(tail, ...)` が `swap()` 内1箇所のみ — ✅ tail 1-writer 確認済み（v12時点 L140）
- `swap()` caller: `ConvolverProcessor.StateAndUI.cpp:1017` の1箇所のみ — ✅ Single Producer
- `tryReclaim()` caller: `DeferredFreeThread.h:143,158` のみ — ✅ Single Consumer

δ案（現状維持）は、`swap()` caller 単一性だけを証明していた。
必要な証明「tail を書く主体が単一である」には `tryReclaim()` からの tail 書き込み削除が必須。

### 実装手順（SafeStateSwapper.h tryReclaim 修正）

#### 変更: tryReclaim() の head 専用化
```cpp
ConvolverState* tryReclaim(uint64_t minReaderEpoch) noexcept
{
    // [Single Consumer debug assert — 変更なし]

    // 1. fallbackQueue を先に確認
    { std::lock_guard<std::mutex> lock(fallbackMutex);
      if (!fallbackQueue.empty()) {
          const auto entry = fallbackQueue.top();
          if (entry.epoch < minReaderEpoch) {
              if (entry.state != nullptr) {
                  fallbackQueue.pop();
                  return entry.state;
              }
          }
      }
    }

    // 2. ring head を確認
    // ★ head はローカル変数 h で追跡。head atomic と h は以下のルールで同期:
    //    next = increment(h)  →  publishAtomic(head, next)  →  h = next
    //    この3ステップは必ず一組で扱う（単独で h だけ更新しない）。
    size_t h = convo::consumeAtomic(head, std::memory_order_acquire);
    if (h == convo::consumeAtomic(tail, std::memory_order_acquire))
        return nullptr;

    // 3. null slot skip (bounded loop)
    for (size_t i = 0; i < kMaxRetired; ++i)
    {
        const uint64_t entryEpoch = convo::consumeAtomic(
            retiredBuffer[h].epoch, std::memory_order_acquire);
        ConvolverState* ptr = convo::consumeAtomic(
            retiredBuffer[h].state, std::memory_order_acquire);

        if (ptr == nullptr || entryEpoch == 0) {
            // null slot: head を進めて次の slot へ
            // ★ 同期ルール: next → publishAtomic(head, next) → h = next
            // ★ 実装推奨: advanceHead(h) helper に集約することで更新規則を1箇所に閉じ込める
            //    例: h = advanceHead(h);  // 内部で next→publish→h=next を実行
            const size_t nextH = (h + 1) % kMaxRetired;
            convo::publishAtomic(head, nextH,
                std::memory_order_release);
            h = nextH;  // ローカル追跡も同期
            if (h == convo::consumeAtomic(tail, std::memory_order_acquire))
                return nullptr;
            continue;
        }

        if (isOlder(entryEpoch, minReaderEpoch)) {
            // reclaim 可能
            convo::publishAtomic(retiredBuffer[h].state, nullptr,
                std::memory_order_release);
            // ★ 同期ルール: next → publishAtomic(head, next)
            const size_t nextH = (h + 1) % kMaxRetired;
            convo::publishAtomic(head, nextH,
                std::memory_order_release);
            // (return のため h=nextH は不要)
            return ptr;
        }

        // reclaim 不可 — ★ tail へ回転しない
        break;
    }
    return nullptr;
}
```

#### 削除するコード
`tryReclaim()` 内の以下のブロックを完全に削除:
```cpp
// ★ 削除: head を進めて tail 側へ回転する
const size_t t = convo::consumeAtomic(tail, std::memory_order_acquire);
...
convo::publishAtomic(tail, nextTail, std::memory_order_release);
```

#### null slot skip ポリシー
| 条件 | 動作 |
|------|------|
| `state == nullptr` | head を進めて skip（bounded loop） |
| `epoch == 0` | head を進めて skip（bounded loop） |
| `epoch < minReaderEpoch` | reclaim（state を返す） |
| `epoch >= minReaderEpoch` | nullptr を返す（tail へ回転しない） |

### CI 3層化

| Layer | コマンド | 成功条件 |
|-------|---------|---------|
| L1: rg | `rg -n "publishAtomic\(tail" src/SafeStateSwapper.h` | swap() 内のみ |
| L2: ast-grep | `tryReclaim` 内の `publishAtomic.*tail` 禁止 | 0 matches |
| L3: contract | `SafeStateSwapperTailWriterSingleTests` 他 | all green |

### テスト追加

```text
SafeStateSwapperTailWriterSingleTests     — publishAtomic(tail) が swap() にのみ存在
SafeStateSwapperHeadOnlyReclaimTests      — tryReclaim() が head のみ更新
SafeStateSwapperNullSlotSkipTests         — null slot を安全に skip
SafeStateSwapperEpochOrderTests           — epoch < minReaderEpoch のみ reclaim
SafeStateSwapperHeadBlockingTests         — head non-reclaimable で後続を触らない
SafeStateSwapperFullFallbackTests         — ring full 時に fallbackQueue へ退避
SafeStateSwapperFallbackOverflowTests     — fallback overflow で quarantine / health
SafeStateSwapperReaderStuckTests          — reader stuck 時に reclaim 停止、UAF なし
```

### リスクと対策
| リスク | 対策 | 重要度 |
|--------|------|--------|
| null slot 連続で ring が詰まる | bounded loop で最大 kMaxRetired まで skip。上限到達時は fallback | LOW |
| fallbackQueue 溢れ | ADD-1 で kMaxFallback 導入 | MEDIUM |
| head 専用化で epoch 逆転 | epoch 単調増加により発生しない（INV-EPOCH-MONOTONIC） | LOW |

### 見積工数
実装1日＋テスト1日＋CI追加0.5日 = **2.5日**

---

## FIX-P2: updateAudioThreadSnapshotFade 削除（旧FIX-HW-3）✅ 実装済み

### 目標
Dead Code 確定に伴い、`updateAudioThreadSnapshotFade()` と `updateFade()` を削除する。

**v12 検証**: コード実装完了。`AudioEngine.h:3738` に DELETED コメント確認。`src/core/SnapshotCoordinator.h:111` も同様。

### 決定根拠
- ✅ 全ツール（grep/ast-grep/rg/cocoindex/semble/graphify）で呼び出し元ゼロを確認
- ✅ `advanceFade()` は `AudioBlock.cpp:475` から LIVE 呼び出しあり → 維持
- ✅ 将来復元は Git 履歴から容易

### 変更内容
1. `AudioEngine.h:3731-3740` — `updateAudioThreadSnapshotFade()` 関数ブロック削除
2. `src/core/SnapshotCoordinator.h:111-138` — `updateFade()` 削除（他からの呼び出しがないことを確認済み）
3. `AudioBlock.cpp:475` — `advanceFade()` 呼び出しは維持。コメントに `[LIVE]` と追記

### 見積工数
30分

---

## FIX-P1-1: FFT Backend Concept 化 + explicit instantiation（旧FIX-U-6）

> **注意**: このセクションは FFTExecutionContext 分離設計採用前の旧実装計画である。
> 最終的な実装設計は上記「P1-1: FFT Backend Concept化」セクション（FFTExecutionContext 分離 + Builder Authority 集中）
> に従うこと。本セクションは reference としてのみ残す。

### 目標
FFT エラー時の異常系テストを実装する。RT パスへのオーバーヘッドはゼロとする。

### 決定根拠（2026-07-28 調査）
- **テスト基盤**: カスタムテストフレームワーク（GoogleTest/GMock/GMock なし）
- **FFT API**: Intel IPP 直接呼び出し（`ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f`）
- GMock 非依存のため、virtual + MockFft 方式は不適切
- → **Concept 方式を採用**（virtual dispatch ゼロ、RT-safe 確定）

### 実装手順

#### Step 1: FFT Backend Concept
```cpp
template <typename FftBackend>
concept FftBackendConcept = requires(FftBackend& b, const double* in, double* out) {
    { b.forward(in, out) } -> std::same_as<IppStatus>;
    { b.inverse(in, out) } -> std::same_as<IppStatus>;
};
```

#### Step 2: ProductionFft
```cpp
class ProductionFft {
public:
    explicit ProductionFft(IppsFFTSpec_R_64f* spec) noexcept : fftSpec_(spec) {}

    IppStatus forward(const double* in, double* out) noexcept {
        return ippsFFTFwd_RToCCS_64f(in, out, fftSpec_, workBuf_);
    }

    IppStatus inverse(const double* in, double* out) noexcept {
        return ippsFFTInv_CCSToR_64f(in, out, fftSpec_, workBuf_);
    }

    void setWorkBuffer(Ipp8u* buf) noexcept { workBuf_ = buf; }

private:
    IppsFFTSpec_R_64f* fftSpec_;
    Ipp8u* workBuf_ = nullptr;
};

static_assert(FftBackendConcept<ProductionFft>);
```

#### Step 3: TestFft（エラー注入可能なテスト用）
```cpp
class TestFft {
public:
    IppStatus forward(const double*, double*) noexcept { return result_; }
    IppStatus inverse(const double*, double*) noexcept { return result_; }

    void setResult(IppStatus s) noexcept { result_ = s; }
    void setResultOnCall(IppStatus fwd, IppStatus inv) noexcept {
        resultForward_ = fwd; resultInverse_ = inv;
    }

private:
    IppStatus result_{ippStsNoErr};
    IppStatus resultForward_{ippStsNoErr};
    IppStatus resultInverse_{ippStsNoErr};
};

static_assert(FftBackendConcept<TestFft>);
```

#### Step 4: MKLNonUniformConvolver の修正 — ★ Layer 単位テンプレート化推奨

`IppsFFTSpec_R_64f*` を直接保持する代わりに、テンプレートパラメータとして FFT 実装を受け取る。
ProductionFft をデフォルトテンプレート引数に指定。

**ISR 観点での注意**: **`Layer` 構造体単位のみのテンプレート化を第一推奨とする。**
クラス全体を `template<class FFT>` にするとコンパイル依存が増大し、インスタンス爆発のリスクがある。
可能なら `Layer` 構造体単位のみテンプレート化し、クラス全体のテンプレート化は避ける:

```cpp
// 推奨: Layer 単位のみテンプレート化
template <FftBackendConcept FftBackend>
struct Layer { /* ... */ };

// 非推奨（コンパイル爆発）:
template <typename FftBackend>
class MKLNonUniformConvolver { /* ... */ };
```

#### Step 5: テスト追加
- TestFft が `ippStsNoErr` を返す正常系
- TestFft が `ippStsErr` を返す異常系（`clearFFTOutputOnError` の動作確認）
- 6箇所全ての FFT 呼び出しをカバー

#### Step 6: explicit instantiation

バイナリ肥大とコンパイル時間対策:

```cpp
// MKLNonUniformConvolverLayer.cpp — Production 型のみ explicit instantiation
template class MKLNonUniformConvolverLayer<ProductionFft>;

// テストファイル — TestFft での instantiation
// (テスト target でのみコンパイル)
```

### ProductionFft 契約

```text
FFT-PROD-1: ProductionFft は `IppsFFTSpec_R_64f*` を保持してよい（所有はしない）。
FFT-PROD-2: spec の生成 / 破棄は NonRT のみ。
FFT-PROD-3: forward / inverse は RT から呼び出し可能。
FFT-PROD-4: forward / inverse は noexcept。
FFT-PROD-5: 失敗時は IppStatus を返す。
FFT-PROD-6: RT 内で allocation / free / exception / log を発生させない。
```

### Fail-Closed 契約

```text
FFT-FAIL-1: FFT が non-success を返したら出力をゼロクリアする。
FFT-FAIL-2: stage を ready にしない。
FFT-FAIL-3: stale な結果を publish しない。
FFT-FAIL-4: RT 内で retry しない。
FFT-FAIL-5: error flag / counter は atomic relaxed でよい。
FFT-FAIL-6: log は NonRT へ委譲する。
```

### 注意点
- Concept 方式は静的ポリモーフィズムのため、virtual dispatch は完全にゼロ
- `FftBackendConcept` 経由の呼び出しはコンパイル時に解決される
- MKLNonUniformConvolver のテンプレート化により、テスト時のみ TestFft を注入可能
- `unique_ptr<FftPolicy>` は使用禁止（virtual dispatch 発生のため）
- **Layer 単位のみテンプレート化** を推奨。クラス全体のテンプレート化はコンパイル依存増大のため避ける

### RT パス影響評価
Concept 方式（virtual ゼロ）のため、RT パスへの影響はゼロ。

### テスト追加

```text
FftProductionInstantiationTests   — ProductionFft のみ release binary に含まれる
FftTestBackendInjectionTests      — TestFft でエラー注入可能
FftForwardErrorFailClosedTests    — forward エラー時 fail-closed
FftInverseErrorFailClosedTests    — inverse エラー時 fail-closed
FftNullSpecTests                  — null spec ハンドリング
FftSizeMismatchTests              — サイズ不一致
FftNoPublishOnErrorTests          — エラー時 publish なし
FftNoMklLeakTests                 — MKL リソースリークなし
```

### 見積工数
設計0.5日 + 実装1日 + テスト0.5日 = **2日**

---

## FIX-D1: kMaxMismatch Epoch ベース検出への移行 ✅ 実装済み

### 目標
Timer 呼び出し回数ベースの `kMaxMismatch = 5` を epoch 差分ベースに変更する。

**v12 検証**: コード実装完了。
- `AudioEngine.Timer.cpp:1800`: `publicationEpochDistance(currentEpoch, receiptEpoch) > kMaxEpochDrift` 確認
- `AudioEngine.h:4354-4355`: `kMaxEpochDrift = 10` + `kMaxMismatch` deprecated 確認

### 変更内容
`AudioEngine.Timer.cpp` の `retirePublishedDSP()` 内、不一致検出ロジック：
```cpp
// 現在（Timer 呼び出し回数ベース）:
uint32_t cnt = mismatchCount_.fetch_add(1, std::memory_order_relaxed) + 1;
if (cnt >= kMaxMismatch) { fatal_ = true; }

// 修正後（epoch 差分ベース）:
// pendingReceipt_->publicationEpoch と router_->currentEpoch() の差で判定
// ★ publicationEpochDistance() helper 経由で将来の epoch policy 変更に備える
const auto currentEpoch = engine_.currentPublicationEpoch();  // ISR Coordinator の最新 epoch
const auto receiptEpoch = pendingReceipt_->publicationEpoch;
if (publicationEpochDistance(currentEpoch, receiptEpoch) > kMaxEpochDrift) { fatal_ = true; }
```

### 定数定義
```cpp
// AudioEngine.h に追加
static constexpr uint64_t kMaxEpochDrift = 10;  // 最大許容 epoch 差
// kMaxMismatch は deprecated として残す（後方互換性のため。外部参照がある場合）
```

### 注意点
- `publicationEpochDistance()` helper: 将来の多次元 Epoch を考慮し、対象を明示。
  `(a >= b) ? (a - b) : 0` のような安全な差分計算をラップ
- epoch 差が `uint64_t` の wraparound を起こさない前提が必要
- ISR Runtime の epoch は実質的に wraparound しない（64bit、単調増加）
- `kMaxMismatch` は削除せず deprecated として残す（外部参照がある場合）

### 見積工数
1時間

---

## FIX-P1-2: Stale receipt quarantine 状態機械（旧FIX-D2 拡張）

### 問題
`DSPTransition::onPublishCompleted()` の Emergency Override パス（HealthState Critical）は `storeReceipt()` を呼ばないため、以前の receipt が `pendingReceipt_` に残留する。

また、`resetReceipt()` だけでは retire 義務が消失するリスクがある。
stale receipt の oldDSP は quarantine へ移し、retire 義務を確実に履行する。

### 既存の Handle Table + Quarantine Infrastructure

`src/audioengine/ISRDSPHandle.h` に **`DSPHandleRuntime`**（完全な Handle Table）が既に実装済み:

```cpp
class DSPHandleRuntime {
    DSPHandle create(void*);         // 登録
    ResolvedDSP resolve(DSPHandle);  // 検証
    void retire(DSPHandle);          // retire 遷移
    void quarantine(DSPHandle);      // ★ quarantine 遷移（既存）
    void reclaim(DSPHandle);         // 解放
};
```

`src/audioengine/ISRDSPQuarantine.h` に **`DSPQuarantineManager`** も実装済み:
- `quarantineHandle(slot, generation, reason)` — slot + generation 単位の隔離
- `reclaimSlot(slot, generation)` — 隔離解除
- `AudioEngine.Threading.cpp:42` で実際に使用中（未使用ではない）
- `DSPQuarantineManager(maxSlots = 256)` — コンストラクタデフォルト引数で上限を規定（`ISRDSPQuarantine.h:20`）

**ISR 原則**: Quarantine Authority は **1個** に統一する。
既存の `DSPHandleRuntime::quarantine(DSPHandle)` と `DSPQuarantineManager::quarantineHandle(slot, gen, reason)` は
**独立した2つの機構**である（ソース確認済み）:
- `DSPHandleRuntime::quarantine()` → slot の `DSPState` を `Quarantined` に遷移（atomic, generation不問）
- `DSPQuarantineManager::quarantineHandle()` → slot+generation 一致確認 + audit log 記録

両者は complementary であり、両方を呼び出すことで slot 状態遷移 + audit の両方を実現する。

`PendingReceiptQuarantine` の新設は **二重 Authority** となるため、**廃止**。

### 推奨状態機械（設計セクション v16 確定版と同一。DSPState enum に統合）

```
Empty ──storeReceipt()──→ Ready ──normal retire──→ Consumed
                            │
                            ├──stale/emergency/mismatch──→ StaleExported
                            │                                 │
                            │                          quarantine 実行
                            │                                 │
                            │                                 ▼
                            │                     Quarantined ← DSPState::Quarantined
                            │                                 │
                            │                          DestroyPending
                            │                                 │
                            │                          Reclaimed
                            │
                            └──resetReceipt()──→ Empty（通常のリセット）
```

**DSPState 対応**: `StaleExported`=`Retired`, `Quarantined`=`DSPState::Quarantined`,
`DestroyPending`=`DSPState::DestroyPending`, `Reclaimed`=`DSPState::Reclaimed`

重要: **Quarantined は自動 free しない。** reader / fade / epoch の安全確認なしに free してはならない。
Quarantined→DestroyPending→Reclaimed の遷移は Coordinator の `waitReaders()` 通過後にのみ実行される。

**設計注記**: `DSPState` 列挙には明示的な `EpochWaiting` 状態は存在しないが、`Retired→DestroyPending` 間の Epoch 待機期間を Coordinator の `waitReaders()` が表現する。実装上は `Retired` 状態かつ `activeReaderCount() > 0` が暗黙の EpochWaiting 相当である。

### 実装

#### Step 1: Quarantine エントリ型
```cpp
struct alignas(64) DSPQuarantineEntry {
    DSPCore* dsp{nullptr};
    convo::isr::PublicationEpoch epoch{0};
    uint64_t quarantinedAtTick{0};
};
```

#### Step 2: resetReceipt → DSPHandleRuntime 経由の quarantine

`PendingReceiptQuarantine` の新設は **二重 Authority** となるため**廃止**。
代わりに既存の `DSPHandleRuntime` を経由する:

```cpp
// ★ ISR 推奨: PublicationReceipt 自体が DSPHandle を保持する
//    lookupDSPHandleForRuntime() による逆引きは Authority 分散のため非推奨。

// ── 現状（コード実態: DSPCore* + DSPHandle の過渡的併存）──
// DSPCore* は retirePublishedDSP 比較用。移行完了後は DSPHandle に一本化する（P0-2）。
struct PublishReceipt {
    DSPCore* dsp{nullptr};                              // ★ P1-2: retirePublishedDSP 比較用（移行完了後 DSPHandle に一本化）
    convo::isr::DSPHandle handle{};                     // ★ P1-2: quarantine 用 Handle
    convo::isr::PublicationEpoch publicationEpoch{0};
    convo::isr::PublicationGeneration generation{0};
};

// ── 設計上の目標（変更後: DSPHandle のみ）──
// struct PublishReceipt {
//     convo::isr::DSPHandle handle{};                  // ★ Handle 保持
//     convo::isr::PublicationEpoch publicationEpoch{0};
//     convo::isr::PublicationGeneration generation{0};
// };

// resetReceipt では Handle 経由で直接 quarantine:
void resetReceipt() noexcept {
    if (pendingReceipt_.has_value()) {
        // retire 義務を DSPHandleRuntime 経由で quarantine へ移転
        // ★ Handle を直接保持しているため逆引き不要
        handleRuntime_.quarantine(pendingReceipt_->handle);
    }
    pendingReceipt_.reset();
    receiptReady_.store(false, std::memory_order_relaxed);
}
```

**DSPHandleRuntime + DSPQuarantineManager 協調**:
両者は独立した機構であり、quarantine 時には両方を呼び出す:
1. `DSPHandleRuntime::quarantine(handle)` — slot state を `Quarantined` に遷移
2. `DSPQuarantineManager::quarantineHandle(slot, gen, reason)` — audit log 記録

※ ソース確認: `DSPHandleRuntime::quarantine()` は `DSPQuarantineManager` を内部的に呼ばない。

#### Step 3: Quarantine Lifecycle 全体像

**既存の `DSPHandleRuntime` の `DSPState` 列挙**:

```text
Constructing → Active → Retired → Quarantined → DestroyPending → Reclaimed
                    ↘ CrossfadingIn/Out ↗
```

これが ISR の完全な Lifecycle であり、receipt quarantine もこの中に含まれる。

```text
Published (storeReceipt)
    ↓
Retired (retirePublishedDSP - Normal Retire)
    ↓ (stale/emergency)
Quarantined (DSPHandleRuntime::quarantine)
    ↓ (shutdown / safe drain)
DestroyPending → Reclaimed
```

#### Step 3: Retire 義務移転ルール

```text
RECEIPT-1: pendingReceipt_ を reset する前に、oldDSP を quarantine へ移す。
RECEIPT-2: quarantine された DSP は、reader / fade / epoch の安全確認なしに free しない。
RECEIPT-3: evidence export は NonRT で行う。
RECEIPT-4: RT 内で file I/O / logger / exception を発生させない。
RECEIPT-5: quarantine 増加は diagnostic counter と health event に記録する。
RECEIPT-6: shutdown 時は drain を試み、不可能なら leak-safe に quarantine する。
```

### テスト追加

```text
ReceiptStaleExportTests                  — stale receipt evidence export
ReceiptQuarantineTransitionTests         — quarantine 遷移確認
ReceiptResetDoesNotDropRetireObligationTests — retire 義務消失防止
ReceiptEmergencyOverrideTests            — Emergency Override 時 quarantine
ReceiptShutdownDrainTests                — shutdown drain
```

### 見積工数
設計0.5日 + 実装0.5日 + テスト0.5日 = **1.5日**

---

## FIX-ADD-1: fallbackQueue bounded 化 ✅ 実装済み

### 問題
`SafeStateSwapper.h` の `std::priority_queue<FallbackEntry> fallbackQueue` は unbounded。
reader stuck や retire stall 時に無限に成長する可能性がある。

**v12 実装確認**: 実際のコードは `kMaxFallback=1024` 上限 + `fallbackOverflowCount_` atomic increment。
overflow 時の Coordinator通知は `getPendingRetiredCount()` の外部ポーリングに委譲（quarantine までは未実装）。

### 実装

```cpp
// SafeStateSwapper.h (v12 実装確認: 実コード SafeStateSwapper.h:119-135)
// overflow 時は quarantine ではなく fallbackOverflowCount_ を atomic increment。
// Coordinator への通知は getPendingRetiredCount() の外部ポーリングに委譲。
static constexpr size_t kMaxFallback = 1024;

// overflow 時の処理:
std::lock_guard<std::mutex> lock(fallbackMutex);
if (fallbackQueue.size() >= kMaxFallback) {
    // ★ overflow counter（relaxed atomic、diagnostic only）
    fallbackOverflowCount_.fetch_add(1, std::memory_order_relaxed);
    // Coordinator への通知は外部ポーリング（getPendingRetiredCount()）に委譲
} else {
    fallbackQueue.push({oldState, epoch2});
}
```

### ルール

```text
FALLBACK-1: fallbackQueue は NonRT でのみ使用する。
FALLBACK-2: 上限 kMaxFallback = 1024。
FALLBACK-3: 上限到達時は新規 push を拒否（drop）。fallbackOverflowCount_ を atomic increment して記録。
FALLBACK-4: fallback overflow 通知は SafeStateSwapper::getPendingRetiredCount() の外部ポーリングに委譲。
FALLBACK-5: overflow 時の quarantine は未実装（将来課題）。現状は leak-safe に counter 記録のみ。
```

### 見積工数
0.5日

---

## FIX-ADD-2: MMCSS AvRevert 例外登録

### 問題
`coding_rule_jp.txt` では Audio Thread 内の MMCSS 設定を禁止しているが、
`revertMmcssOnAudioThread()` が Audio callback 内で `AvRevertMmThreadCharacteristics` を呼ぶ。

### 調査結果
- `AudioEngine.Mmcss.cpp:204` — `revertMmcssOnAudioThread()` が `::AvRevertMmThreadCharacteristics(t_mmcssHandle)` を呼ぶ
- `AudioEngine.h:2303-2305` — MMCSS shutdown は flag 経由で Audio Thread に委譲
- 設計コメントに「ASIO thread entry をフックできない場合のみ例外として許可」と記載あり

### 対応

```text
MMCSS-EX-1: Audio Thread 内での MMCSS API 呼び出しは、
            ASIO thread entry をフックできない場合のみ例外として許可する。
MMCSS-EX-2: 呼び出しは thread_local guard により一度だけとする。
MMCSS-EX-3: RT 内で log しない（#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS でガード済み）。
MMCSS-EX-4: 失敗しても音声を止めない。
MMCSS-EX-5: 例外登録簿に記載する。
```

### 例外登録簿への記載例

```text
| # | 機能 | ファイル | 行 | 理由 | 承認日 |
|---|------|---------|----|------|-------|
| 1 | AvRevertMmThreadCharacteristics | AudioEngine.Mmcss.cpp | 201-204 | ASIO thread entry 非フック可能時のみ | 2026-07-28 |
```

### 見積工数
30分（文書更新のみ）

---

## FIX-ADD-3: DeferredFreeThread Logger rate limit ✅ 実装済み

### 問題
`DeferredFreeThread.h:168` で backlog 警告を毎ループ出力している。
`kPendingRetiredWarnThreshold` 到達時は毎 iteration でログ出力が発生する。

### 対応

```cpp
// DeferredFreeThread.h に rate limit 追加
std::chrono::steady_clock::time_point lastLogTime_;
static constexpr auto kLogInterval = std::chrono::seconds(5);

// ログ出力部分
if (pendingRetired >= kPendingRetiredWarnThreshold) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastLogTime_ >= kLogInterval) {
        juce::Logger::writeToLog("[DIAG] DeferredFreeThread backlog pending="
                                 + juce::String(static_cast<juce::int64>(pendingRetired)));
        lastLogTime_ = now;
    }
}
```

### ルール

```text
LOG-1: DeferredFreeThread の log は rate limit する（5秒間隔以上）。
LOG-2: 同一条件の連続 log は間引く。
LOG-3: critical な場合のみ error log（通常は DIAG level）。
```

### 見積工数
15分

---

## FIX-ADD-4: ASan / TSan CI job 分離

### 問題
現在の CMakeLists.txt には ASan 設定が含まれるが、Debug ビルド（/MTd）とは非互換。
ASan と TSan は同時に使えないため、別 job に分離する必要がある。

### 調査結果
`CMakeLists.txt:1049` — ASan 設定存在（`/fsanitize=address`）。ASan ブロックは L1049-1081。
ただし Debug は `/MTd`（静的CRT）で ASan 非対応。

### 推奨 CI 構成

| Config | CRT | Sanitizer | 備考 |
|--------|-----|-----------|------|
| Debug | /MTd | なし | 既存の Debug タスク |
| Debug-ASan | /MDd | AddressSanitizer | 新規 CI job |
| Debug-TSan | dynamic CRT | ThreadSanitizer | 新規 CI job（要 Clang） |
| Release | /MT | なし | 既存の Release タスク |
| Release-PGO | /MT | なし | 既存の PGO タスク |

### CMakeLists.txt 変更案

```cmake
# ASan / TSan は専用ターゲットでのみ有効化
option(ENABLE_ASAN "Enable AddressSanitizer (Debug ASan job)" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (Debug TSan job, Clang only)" OFF)

if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASan and TSan are mutually exclusive. Enable only one.")
endif()

if(ENABLE_TSAN AND MSVC)
    message(FATAL_ERROR "TSan requires Clang (MSVC not supported). Use Clang or WSL Clang.")
endif()

if(ENABLE_ASAN)
    # ASan 必須: 動的 CRT（/MDd for Debug, /MD for Release）
    # 静的 CRT（/MT /MTd）は MSVC ASan と非互換（LNK2038）
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(ENABLE_TSAN)
    # TSan は Clang の -fsanitize=thread で有効化
    target_compile_options(ConvoPeq PRIVATE -fsanitize=thread)
    target_link_options(ConvoPeq PRIVATE -fsanitize=thread)
    set_property(TARGET ConvoPeq PROPERTY MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
```

### 見積工数
CI 設定1日

---

## FIX-D3: onTransitionComplete / notifyTransitionComplete デッドコード処理（現状維持）

### 選択肢

#### ISR 観点: 削除が原則だが保留中

ISR では「Authority が存在しないコードは削除」が原則。
`onTransitionComplete()` と `notifyTransitionComplete()` は以下の状態:

- `notifyTransitionComplete()`: **メソッド本体は実装済み**（`RuntimePublicationOrchestrator.cpp:392`）、4責務の処理ロジックを持つ
- **ただし外部呼び出し元はゼロ**（AiDex 検証済み）。`notifyTransitionComplete` 自体はどこからも呼ばれていない
- `onTransitionComplete()`: `DSPTransition.h:132` に**定義済み**（宣言のみではない）。`notifyTransitionComplete`（`RuntimePublicationOrchestrator.cpp:398`）から呼び出されているが、`notifyTransitionComplete` 自体が呼ばれていないため間接的に到達不能

#### Option A: 完全削除（推奨）※ただし本体実装済みのため注意
関数本体が実装済みであるため、単なる宣言削除ではない。削除する場合は実装コードも削除する。

#### Option B: Reserved Hook（現状維持）
コメントに「将来の統合フック」として維持。本設計書では現状維持とする。

### 推奨: Option B（現状維持）
呼び出し元不在だが、設計上の統合ポイントとして責務定義を保持する（コードコメント L383-391 に明記）。

### 見積工数
15分

---

## A. 実装済み事項一覧（全37件）

### HW-1: Publication Metadata Propagation ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-030（拡張） |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | 7ファイル |
| **ステータス** | ✅ **実装完了・テスト通過（19/19）** |

**設計の核**: `DSPTransition` が `oldDSP` を `PublishReceipt` として保存し、Timer の retire パスで `publicationEpoch` を伝搬する。Retire は Normal/Fallback/Emergency の3分類。

**実装ファイル**:

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimeSemanticSchema.h` | `PublicationGeneration` 型エイリアス追加 |
| `ISRRuntimePublicationCoordinator.h` | `currentPublicationEpoch()` getter |
| `AudioEngine.h` | `PublishReceipt` struct + receipt管理メンバ + `storeReceipt()`/`retirePublishedDSP()`/診断カウンタ |
| `AudioEngine.Timer.cpp` | `retirePublishedDSP()` 定義（3分類＋診断カウンタ）＋3 CAS パス更新 |
| `DSPLifetimeManager.h` | `retire(DSPCore*, uint64_t epoch)` overload |
| `DSPTransition.h` | `storeReceipt(oldDSP, epoch)` + CAS パス更新 |
| `RuntimePublicationOrchestrator.cpp` | （初回実装後に削除 → storeReceipt は DSPTransition に移動） |

**設計上の重要な決定**:

| 判断 | 根拠 |
|------|------|
| Retire の3分類: Normal / Fallback / Emergency | Normal のみ publicationEpoch 伝搬。Fallback/Emergency は runtimeEpoch |
| Publication Metadata Propagation は Normal Retire のみ対象 | Invariant として明文化 |
| 診断カウンタ: normal/fallback/emergencyRetireCount_ | `publishCount ≈ normal + emergency`。fallback は startup/shutdown で少数発生 |
| release/acquire 二層保護 | External Serialization（設計層）＋ atomic 操作（言語層） |
| Fatal 時も current を retire | リーク防止。pendingReceipt_ のみ診断用保持 |

### C-4: TruePeakDetector int→size_t ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-019 |
| **重要度** | 🟢 LOW |
| **ファイル** | `src/TruePeakDetector.cpp`, `src/TruePeakDetector.h` |
| **ステータス** | ✅ **実装完了** |

**修正内容**:
- `kStage0LOffset`/`kStage0ROffset`/`kStage1LOffset`/`kStage1ROffset`: `int` → `size_t`
- `interpolateStage()` 第3引数: `int inputSamples` → `size_t inputSamples`
- ループ変数: `int n` → `size_t n`（`ptrdiff_t` で安全演算）
- `scanPeak()` 呼び出し: `static_cast<int>(up2Samples)` で警告抑制

### グループA: 即時実施可能（13件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII `ApplyComputedIRLoadingGuard` 導入 |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp` | `irL.release()`/`irR.release()` を init 成功時に移動 |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp`（6箇所） | `clearFFTOutputOnError()` ヘルパー導入 |
| A-5 | BUG-011/012/013 | `CmaEsOptimizer.h`, `CmaEsOptimizerDynamic.h`, `CmaEsOptimizerDynamic.cpp` | `sigma = std::clamp(s, sigmaMin, sigmaMax)` 5箇所 |
| A-6 | BUG-029 | `DSPTransition.h` | Emergency Override で `exchangeFadingRuntimeDSP` を使用 |
| A-7 | BUG-028 | `CrossfadeRuntime.h` | `complete()` で全フラグリセット（pending/useDryAsOld/等） |
| A-8 | BUG-015 | `ISRRetireRouter.cpp` | `n` でリトライロジック内蔵＋戻り値確認 |
| A-9 | BUG-016 | `CmaEsOptimizer.h`, `CmaEsOptimizerDynamic.h` | `sanitize()` で NaN/Inf→0.0 クランプ |
| A-10 | BUG-042/044/046 | 各クラス | Rule of Five（`=delete`/`=default`） |
| A-11 | BUG-045 | `IRConverter.cpp` | resample 失敗時に `actualSampleRate = sourceRate` |
| A-12 | BUG-039 | `CustomInputOversampler.cpp` | `std::min(targetSamples, static_cast<int>(upsampledBlock.getNumSamples()))` |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp` | `sampleRateHz > 0 ? ... : 48000` フォールバック |

### グループB: 設計確定済み（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| B-1 | BUG-030 | `AudioEngine.h`, `DSPTransition.h`, `AudioEngine.Timer.cpp` | `claimFadingRuntimeDSP` CAS-only 実装 | ✅ 完了 |
| B-4 | BUG-032 | `SnapshotCoordinator.h:122` | `getCurrentSnapshot()` インターフェース追加 | ✅ 完了 |
| B-5 | BUG-024 | `SnapshotFadeState.h` | `fadeGeneration_` ABA 対策（generation比較） | ✅ 完了 |
| B-6 | BUG-037 | `ConvolverProcessor.h:883`, `ConvolverProcessor.Lifecycle.cpp:107` | `loaderGeneration_` UAF 防止（デストラクタ先頭 fetch_add） | ✅ 完了 |

### グループC: 計画的対応（7件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| C-1 | BUG-033 | `AudioEngine.Processing.BlockDouble.cpp:421` | `dryScale` ラムダキャプチャ追加 | ✅ 完了 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:38` | `n` 化 | ✅ 完了 |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-5f` | ✅ 完了 |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` | ✅ 完了（HW-1 関連で本編C-4も同時完了） |
| C-5 | BUG-020 | `ConvolverProcessor.LoaderThread.cpp:151-152` | `if(targetLength<=0)return 0;` | ✅ 完了 |
| C-6 | BUG-021/022 | `ConvolverProcessor.Lifecycle.cpp:147-150` | RCU `GlobalGuard` 追加（2箇所） | ✅ 完了 |
| C-7 | BUG-026 | `ObservedRuntime.h:49` | `rootEnterSucceeded()`確認 | ✅ 完了 |

### グループD: 余裕時（4件完了）

| ID | バグ | ファイル | 修正内容 | ステータス |
|----|------|----------|----------|-----------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:649` | VLA→`makeAlignedArray` ヒープ割当 | ✅ 完了 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 | ✅ 完了 |
| D-3 | BUG-027 | `SnapshotCoordinator.cpp:15` | `target==null` 時 state 再確認 | ✅ 完了 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10 に含む（Rule of Five） | ✅ 完了 |

### 解決済み未確定事項

| ID | 内容 | 解決日 |
|----|------|--------|
| U-1 | `getCurrentSnapshot()` インターフェース確認（`SnapshotCoordinator.h:122`） | ✅ 2026-07-27 |
| U-4 | Publication Metadata Propagation to Retire Path | ✅ 2026-07-28（→ HW-1 実装完了） |
| U-5 | B-6 Generation インクリメントタイミング（デストラクタ先頭） | ✅ 2026-07-27 |

### v12 新規追加実装済み事項（6件）

以下の6項目は v11 計画書では「⚠️ 未実装」と記載されていたが、v12 ソースコード検証（AiDex/AST-grep/grep/serena 他）により実装完了を確認。

| ID | 内容 | ファイル | 確認内容 |
|----|------|----------|----------|
| ✅ **P0-1** | SafeStateSwapper tail 2-writer 解消（head 専用化） | `SafeStateSwapper.h` | `tryReclaimSlot()` + `advanceHead()` + `ReclaimResult` enum 実装済み。`publishAtomic(tail)` は `swap()` のみ |
| ✅ **P0-3** | AudioSegmentBuffer 61MB ヒープ化 | `AudioSegmentBuffer.h`, `NoiseShaperLearner.h` | `ScopedAlignedPtr` heap + factory + Rule of Five + `static_assert(sizeof<1024)` |
| ✅ **P2** | updateAudioThreadSnapshotFade 削除 | `AudioEngine.h:3738`, `src/core/SnapshotCoordinator.h:111` | DELETED コメント確認。`advanceFade()` は維持 |
| ✅ **ADD-1** | fallbackQueue bounded化 | `SafeStateSwapper.h:448` | `kMaxFallback=1024` + overflow counter 実装済み |
| ✅ **ADD-3** | DeferredFreeThread Logger rate limit | `DeferredFreeThread.h:169,184-185` | `kLogInterval=5s` + `lastLogTime_` 実装済み |
| ✅ **FIX-D1** | kMaxMismatch epochベース化 | `AudioEngine.Timer.cpp:1800`, `AudioEngine.h:4355` | `kMaxEpochDrift=10` + `publicationEpochDistance()`

### v20.2.6 新規追加実装済み事項（7件）

以下の7項目は v20.2.5 設計書では「⚠️ 未着手/設計確定」と記載されていたが、v20.2.6 実装フェーズでコード実装を完了。

| ID | 内容 | 成果ファイル | 確認内容 |
|----|------|-------------|----------|
| ✅ **P1-1** | FFT Backend Concept 全5Phase | `FFTBackend.h/cpp`, `FFTExecutionContext.h`, `ConvolverBuilder.h`, `MKLNonUniformConvolver.h/cpp` | `FftStatus`/`FftStage` enum, `FftBackendConcept`, `ProductionFft`(createPlan/destroyPlan/forward/inverse), `TestFft`(error injection), `FFTExecutionContext`(processLayerFwd/Inv+nullptr guard), `ConvolverBuilder`, Layer `m_fftPlan`/`m_fftCtx`統合, 6FFT呼出全置換, `releaseAllLayers` Plan破棄, `areFftDescriptorsCommitted`→`m_fftCtx[li].isPlanValid()`, `FFTBackendTests`(7テスト) |
| ✅ **P0-2** | EQCoeffCache DSPHandleRuntime移行 | `EQProcessor.h`, `AudioEngine.h`, `AudioEngine.Cache.cpp`, `RefCountedDeferred.h` | `EQCoeffCache`→`RefCountedDeferred`継承削除, `CacheMap`→`DSPHandle`化, `getOrCreate()`/`get()`→`DSPHandleRuntime::create()`/`resolve()`統合, エラーパス `delete cache`追加, `RefCountedDeferred` deprecated化 |
| ✅ **P1-2** | Receipt状態機械 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `ISRDSPQuarantine.h` | `resetReceipt()`実装(quarantine+reset), `QuarantineReason::ReceiptReset`追加 |
| ✅ **ADD-2** | MMCSS例外登録簿 | `doc/coding_rule_jp.txt` | MMCSS-EX-1〜5契約, 例外登録簿テーブル追加 |
| ✅ **ADD-4** | ASan/TSan CI設定 | `CMakeLists.txt`, `.github/workflows/sanitizer-ci.yml` | `ENABLE_TSAN`オプション追加(ASan排他+Clangのみ), debug-asan+debug-tsan CI workflow |
| ✅ **P0-4C** | Coordinator Interface拡充 | `ISRRuntimePublicationCoordinator.h/cpp` | `emitObserveIntent()`, `emitQuarantineIntent()`, `requestReclaim()` 宣言+実装 |
| ✅ **CACHE-LT-1** | キャッシュライフタイム契約 | `doc/work88/REPAIR_PLAN.md` | 通常時`retire()`のみ/Shutdown時`resolve→delete→reclaim`の契約明文化 |

## C. レビュー履歴

| 版 | レビュー | 主な変更 |
|----|---------|----------|
| v1 | — | 初版。Phase 1〜4 分類 |
| v2 | 1次 | グループA/B/C/D 再分類。W-6/W-8/W-10 設計変更 |
| v3 | 1次 | 全調査確定。3部構成。B全6件設計確定 |
| v4 | 2次 | B-1: acquiringFadingSlot→CAS+exchange。B-2: publish順序。他 |
| v5 | 3次 | B-1: CAS+exchange→CAS-only。A-4: CCS/FFT サイズ差異明記。B-2: 安全性証明追記 |
| v6 | 4次 | A-4: 「7箇所」→「6箇所」修正。異常系テスト方法変更。A-2 isLoading競合確認。B-2「未確定」に格下げ |
| v7 | 2026-07-27 | 全実装状況をコードベース調査により確認。実装済み29件をAppendixに移動。未実装5件の設計を新設 |
| **v8** | **2026-07-28** | **HW-1/C-4 実装完了に伴い全未解決事項を「残課題」に集約。実装済み37件をAppendix Aに統合。設計上の注意点5項目を追加。全7回のレビューサイクル完了** |
| **v9** | **2026-07-28** | **全残課題を最終調査・確定。HW-2(Single Producer確認→δ案)、HW-3(Dead Code確定→削除)、U-6(CRTP方式決定)、設計上の注意点5項目全件調査完了。全ツール(WSL grep/ast-grep/rg/cocoindex/semble/graphify/serena/AiDex)使用。** |
| **v10** | **2026-07-28** | **レビュー指摘を全面的に反映。P0-1: δ案→Option A(head専用化)＋HEAD-4/5/6＋INV-AUTHORITY。P0-2: retired flag案廃止→DSPHandleRuntime移行＋HANDLE-5/6/7。P0-3: 61MBヒープ化＋Rule of Five。P1-1: FFT Backend Concept化＋Layer単位テンプレート。P1-2: quarantine状態機械＋RECEIPT-6。ADD-1〜4: fallbackQueue bounded/MMCSS/Logger/ASan。全7項目の設計上の注意点更新。** |
| **v11** | **2026-07-28** | **P1-1/ADD-4 詳細設計を新設・旧設計セクションより分離。全11項目の設計確定。** |
| **v12** | **2026-07-28** | **全コードベース検証（AiDex/AST-grep/grep/serena/semble/cocoindex/graphify使用）。P0-1/P0-3/P2/ADD-1/ADD-3/FIX-D1の6項目がコード実装済みであることを確認、ステータス修正。P0-2/P1-2の一部実装確認。未完了は残5項目（P0-2一部/P1-2一部/P1-1/ADD-2/ADD-4）。P1-1/ADD-4は詳細設計完了、実装フェーズは別タスク。onTransitionCompleteは宣言のみではなく定義済み（DSPTransition.h:132）、notifyTransitionCompleteから呼び出されているが間接到達不能。** |

## D. 調査結果詳細

### C.1 HW-1: Publication Metadata Propagation 調査結果

**調査ツール**: AiDex/grep/semble/cocoindex/serena/ast-grep/rg

✅ **確定した事実**:
- `RetireIntent` 構造体（`ISRRetire.h`）には既に `retireEpoch` フィールドが存在する
- `commit()` 関数（`ISRRuntimePublicationCoordinator.cpp`）は `PublicationEpoch epoch` パラメータを受け取る
- DSPLifetimeManager::retire(DSPCore*, uint64_t) overload 追加により epoch 伝搬が可能に

### C.2 P0-1: SafeStateSwapper tail 2-writer 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/awk + コード実査

✅ **修正済みの事実（v12 時点）**:
- `swap()` は publish 順序: `publishAtomic(state, release) → publishAtomic(epoch, release) → publishAtomic(tail, release)` — **正しい**
- `tryReclaim()` からの `publishAtomic(tail, ...)` は **削除済み**
- `tryReclaim()` は **Single Consumer 前提**（コードコメント L270 に明記）
- `getState()` は `activeState` のみ読み `retiredBuffer` を直接読まない ✅

履歴: v9=δ案(現状維持) → v10=Option A(head専用化) → v12=実装完了確認

### C.3 HW-3: updateAudioThreadSnapshotFade 調査結果

**調査ツール**: AiDex/grep/ast-grep/rg/sed/cocoindex/semble/graphify

🔴 **確定**: `updateFade()` は未呼び出し、`snapshotAlpha` 等は DSP 処理パスのどこからも未参照。
SnapshotFade の結果は全く使用されていない ≈ Dead Code。

### C.4 P1-1: FFT clearFFTOutputOnError 調査結果（旧U-6）

**調査ツール**: AiDex/grep/rg

- ✅ A-4（`clearFFTOutputOnError`）実装済み（`MKLNonUniformConvolver.cpp` 内6箇所）
- ✅ `unique_ptr<FftPolicy>` は存在しない（virtual dispatch なしの状態維持）
- ❌ FFT エラー時の異常系テストが未実装
- ❌ explicit instantiation 未対応

### C.5 P0-2: RefCountedDeferred tryAddRef 調査結果

**調査ツール**: AiDex/grep/rg/WSL grep

✅ **確定した事実**:
- `tryAddRef()` は既に **CAS loop** を実装（`RefCountedDeferred.h:48-56`）
- `compareExchangeAtomic` で count 0 への increment は atomic に防止される
- ❌ `retired_` flag がない — retire 済みオブジェクトへの tryAddRef が成功し得る（resurrection）
- ❌ RCU 保護契約が文書化されていない

### C.6 P0-3: AudioSegmentBuffer 61MB 調査結果

**調査ツール**: AiDex/grep/rg/WSL grep

✅ **修正済みの事実（v12 時点）**:
- `AudioSegmentBuffer.h` — ヒープ化完了済み（`ScopedAlignedPtr` + factory `create()`）
- 合計約 **61.44 MB** はスタック→ヒープに移行済み
- `NoiseShaperLearner.h:278` で `std::unique_ptr<AudioSegmentBuffer> segmentBuffer` として保持
- `static_assert(sizeof(AudioSegmentBuffer) < 1024)` でスタック禁止を保証

### C.7 ADD-1〜4 調査結果（v12 更新）

| ID | 項目 | v11 状態 | v12 状態 | 詳細 |
|----|------|----------|----------|------|
| ADD-1 | fallbackQueue bounded | ❌ unbounded | ✅ **実装済み** | `kMaxFallback=1024` + overflow counter |
| ADD-2 | MMCSS例外登録 | ❌ 未登録 | ❌ 未登録 | コードは存在、例外登録簿への記載のみ未完了 |
| ADD-3 | Logger rate limit | ❌ 未実装 | ✅ **実装済み** | `kLogInterval=5s` + `lastLogTime_` |
| ADD-4 | ASan/TSan CI | ❌ 未分離 | ❌ 未分離 | CI設定フェーズ未着手（詳細設計完了） |

### C.8 追加調査で確定した事実

**P0-2: tryAddRef 呼び出し元ゼロ判定**:
- `src/RefCountedDeferred.h:48` で定義されているが、`.cpp` / `.h` の**いずれからも呼び出しなし**
- `MKLNonUniformConvolver.h:284` の `refCount` は別の軽量参照カウンタ（RefCountedDeferred非使用）
- **結論: `tryAddRef()` は Dead Code。修正は予防措置**

**P0-1: SafeStateSwapper tryReclaim Single Consumer 検証**:
- `SafeStateSwapper::tryReclaim()` の真の呼び出し元: **`DeferredFreeThread.h:143` のみ**
- `ISRRetireRouter::tryReclaim()` → `provider_->tryReclaim()` は `IEpochProvider*` 経由で **`EpochDomain`（別RCU実装）** を呼ぶ
- `EQProcessor.Core.cpp` の `m_epochDomain.tryReclaim()` も **EpochDomain 独自**
- ✅ **SafeStateSwapper は真に Single Consumer**

**P1-2: DSPQuarantineManager / DSPHandleRuntime 既存確認**:
- `src/audioengine/ISRDSPQuarantine.h` に `DSPQuarantineManager` が **既に実装済み**
- API: `quarantineHandle(slot, generation, reason)` / `reclaimSlot(slot, generation)` / `isActive(slot)` / `destroyForShutdown(slot)`
- `kMaxSlots = 256`（上限固定）
- `AudioEngine.Threading.cpp:42` で **実際に使用中**（未使用ではない）
- `src/audioengine/ISRDSPHandle.h` に **`DSPHandleRuntime`**（完全な Handle Table）が実装済み
  - API: `create/resolve/retire/quarantine/reclaim` — 全ライフサイクル管理
  - `DSPHandle{slot, generation}` — ABA 防止
  - `DSPState` 列挙: `Constructing→Active→Retired→Quarantined→DestroyPending→Reclaimed`
  - これが ISR の理想的 Handle Table パターン
- ❌ `DSPQuarantineManager` は (slot, generation) ペアで動作。receipt の (DSPCore*, epoch) とは型が合わない
- ⚠️ `DSPHandleRuntime::quarantine(DSPHandle)` と `DSPQuarantineManager` は**独立した別機構**:
  - `DSPHandleRuntime::quarantine()`: slot state を `Quarantined` に遷移（generation不問）
  - `DSPQuarantineManager::quarantineHandle()`: generation一致確認後 audit log 記録
  - 両者は相互に呼び出さない。必要に応じて両方を呼ぶ設計が必要

## E. 調査で使用したツール

| ツール | 用途 |
|--------|------|
| WSL grep | 全テキスト検索・全実装項目のコードベース確認 |
| ast-grep | 構造パターン検索（`engine_.storeReceipt`, `retirePublishedDSP` 等） |
| rg (ripgrep) | 高速フィルタリング検索 |
| cocoindex (ccc.exe) | 構造的grep（receiptReady_, fatal_, mismatchCount_ 等の全参照網羅） |
| semble | セマンティックコード検索 |
| graphify | ナレッジグラフ解析（RuntimePublicationCoordinator ノード確認） |
| serena MCP | プロジェクト構成確認 |
| AiDex MCP | プロジェクトインデックス管理・ステータス確認 |

---

*本設計書は ISR Runtime OS 設計原則に基づく。v10-18: ISR段階的改善。v19: QUEUE-3〜8。**v20.2.6: ERRATA-V2023統合・FftStatus/FftStage型安全・FftBackendConcept static/instance分離・CMake CMP0091・workBufferアライメント・EnqueueResult拡張・ShutdownState機械・Queue-9 reserved slots・QuarantineService QSVC・HealthMonitorイベント・Traceability Matrix・文書体系整備・EC契約・PLAN-LT-1〜10・FFT-PROD-15・RESOLVE-5〜7・QUEUE-11〜13。全26最終受け入れ条件中6実装済み・20設計確定。***
**【2026-07-29 検証レポート】**: 22ファイルの全テストファイル存在確認。CMake ビルドターゲット統合は未完了。コード検証（WSL grep/rg/ast-grep 他）により10の設計/実装事実を確認。`setWorkBuffer()` 不在、`m_layers[3]` 範囲外アクセス記述、TestFft `IppStatus` 型ミスマッチ、ResolvedDSP 戻り値型記述の不正確さを修正。icx Debug `/MT` vs `/MTd` 乖離を注記。`CMP0091` 未設定を確認後、設計して CMakeLists.txt に実装済み（解決）。

---

# Appendix C: 補完セクション (v20.2.4 統合)

## C-1: 拡張 Enum 定義

### AckResult — Intent Queue ACK 用

Intent Queue（QUEUE-5）で使用する ACK 応答型。EnqueueResult とは別用途（キューイング受付確認用）。

```cpp
enum class AckResult : int
{
    Accepted = 0,     // Intent がキューに受理された
    QueueFull,         // Intent Queue が満杯
    ShuttingDown       // Shutdown 中で新規 Intent を受付不可
};
```

### EnqueueResult — 汎用 Enqueue 結果

```cpp
enum class EnqueueResult : int
{
    Success = 0,
    QueueFull,
    QueueFullCritical,   // ★ critical command が reserved slot にも enqueue 不可
    Shutdown,
    InvalidArgument,
    NotReady,
    Duplicate,
    RejectedByPolicy,
    RejectedByAdmission,
    InternalError
};

// 全 enqueue 関数は [[nodiscard]] noexcept
[[nodiscard]] EnqueueResult enqueue(...) noexcept;
```

**契約**:
| ID | 契約 |
|----|------|
| ENQUEUE-1 | enqueue は `noexcept` |
| ENQUEUE-2 | enqueue は RT から呼ばれても wait-free bounded |
| ENQUEUE-3 | `EnqueueResult` は `[[nodiscard]]` |
| ENQUEUE-4 | `Success` 以外でも RT はブロックしない |
| ENQUEUE-5 | `QueueFull` は non-critical command に対して coalesce / drop 可能 |
| ENQUEUE-6 | `QueueFullCritical` は critical command が enqueue できないことを示す |
| ENQUEUE-7 | `Shutdown` は終端状態。以降の enqueue は拒否される |
| ENQUEUE-8 | `RejectedByAdmission` は reader slot / retire queue / backpressure による拒否 |
| ENQUEUE-9 | `InternalError` は HealthMonitor へ報告する |

## C-2: Shutdown 状態機械

```cpp
enum class ShutdownState : int
{
    Running = 0,
    ShutdownRequested,
    Draining,
    EpochWaiting,
    Reclaiming,
    Quarantined,
    ShutdownCompleted,
    Faulted
};
```

**状態遷移**:
```
Running → ShutdownRequested → Draining → EpochWaiting → Reclaiming → ShutdownCompleted
                                               |
                                          timeout → Quarantined → ShutdownCompleted
```

**閾値**: `EPOCH_WAIT_NORMAL_MS = 100`, `EPOCH_WAIT_SHUTDOWN_MS = 1000`, `EPOCH_WAIT_HARD_LIMIT_MS = 3000`

**契約**:
| ID | 契約 |
|----|------|
| SHUTDOWN-1 | Shutdown は最高優先度。fairness の対象外 |
| SHUTDOWN-2 | Shutdown 要求後は新規 enqueue を拒否する |
| SHUTDOWN-3 | Shutdown 中は DSPState の retire / epoch reclaim を安全に行う |
| SHUTDOWN-4 | Shutdown 完了前にオブジェクトを破壊しない |
| SHUTDOWN-5 | Shutdown 完了後は `EnqueueResult::Shutdown` を返す |
| SHUTDOWN-6 | Shutdown は冪等 |
| SHUTDOWN-7 | Shutdown 中に fatal error が起きたら `Faulted` へ遷移する |

## C-3: HealthMonitor イベント種別

```text
EVENT_FFT_ERROR                 — FFT 呼び出し失敗
EVENT_QUEUE_FULL                — 通常キュー満杯
EVENT_QUEUE_FULL_CRITICAL       — critical reserved slot も満杯
EVENT_EPOCH_WAIT_TIMEOUT        — EpochWaiting タイムアウト
EVENT_QUARANTINE_ENTERED        — quarantine 登録
EVENT_QUARANTINE_RECLAIMED      — quarantine から回収成功
EVENT_QUARANTINE_ABANDONED      — quarantine 放棄（process exit 時）
EVENT_QUARANTINE_LIMIT_EXCEEDED — quarantine 上限超過
EVENT_QUARANTINE_SERVICE_FAILURE— QuarantineService State+Audit トランザクション失敗
EVENT_READER_SLOT_USAGE         — reader slot 使用率閾値超過
EVENT_PUBLICATION_MISMATCH      — publication epoch 不一致
EVENT_RETIRE_OVERFLOW           — retire queue overflow
EVENT_ADMISSION_STOPPED         — backpressure による admission 停止
EVENT_SHUTDOWN_REQUESTED        — shutdown 開始
EVENT_SHUTDOWN_COMPLETED        — shutdown 完了
EVENT_FAULTED                   — Faulted 状態遷移
```

**契約**:
| ID | 契約 |
|----|------|
| HEALTH-1 | HealthMonitor への enqueue は bounded |
| HEALTH-2 | RT は HealthMonitor へ non-blocking に enqueue するだけ |
| HEALTH-3 | HealthMonitor の集計は NonRT で行う |
| HEALTH-4 | UI 表示は pull 型 |
| HEALTH-5 | 診断ログは `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード |
| HEALTH-6 | Release の RT パスでログを出さない |
| HEALTH-7 | 重大イベントは `Faulted` / `SafeState` と連携する |

## C-4: Traceability Matrix

| BUG/リスク | 設計項目 | 契約 | テスト | 完了条件 |
|---|---|---|---|---|
| FFT 異常系未検証 | P1-1 | FFT-PROD / FFT-TEST / FFT-FAIL | FftErrorInjectionTests | clearFFTOutputOnError 発火、silent output |
| ASan CRT 衝突 | ADD-4 | ASan-CMAKE-1〜8 | debug-asan build | link success、ASan clean |
| TSan 非現実性 | ADD-4 | TSAN-ALT-1〜7 | stress / audit | race 代替検証 |
| critical drop 矛盾 | QUEUE-9 | QUEUE-9-1〜10 | queue stress | QueueFullCritical、critical drop なし |
| epoch timeout 未定義 | Epoch/Quarantine | EPOCH-1〜9 / QUARANTINE-1〜10 | shutdown drain | quarantine or reclaim |
| MMCSS RT 呼び出し | ADD-2 | MMCSS-EX-1〜5 | doc + audit | exception registry |
| DSPState UAF | DSPState/P0-4 | DSPSTATE-1〜8 / RETIRE-1〜6 | publication tests | UAF なし |
| Receipt stale | P1-2 | RECEIPT-1〜8 | receipt state test | quarantine/reset via intent |
| Handle Authority 分散 | P0-4/P0-5 | ISR-AUTH-1〜6 | coordinator tests | Authority 単一化 |
| Qipo 二重定義 | ADD-4 | ASan-CMAKE-5 | CMake inspect | 条件付き単一定義 |
| clearFFTOutputOnError 移行 | P1-1 | FFT-STAGE / FFT-STATUS | migration test | legacy stage 互換 |
| Quarantine 二重 Authority | P0-5 | QSVC-1〜4 | quarantine tests | State+Audit 単一管理 |
| workBuffer alignment | P1-1 | FFT-PROD-11〜14 | debug assert / ASan | 64-byte alignment |
| toFftStage 範囲外 | P1-1 | FFT-STAGE-6〜9 | unit test | Diagnostic clamp |

## C-5: 付属文書（実装時に作成/更新）

```text
doc/exception_registry.md            — MMCSS 例外登録簿（関数名・ファイル名・契約・承認日必須）
doc/health_monitor_events.md         — HealthMonitor イベント定義
doc/fft_backend_concept.md           — FftBackendConcept / FftStatus / FftStage 完全仕様
doc/quarantine_lifecycle.md          — Quarantine ライフサイクル詳細
doc/ci_asan_matrix.md                — ASan CI 設定マトリックス
doc/errata/v20.2-errata.md           — 設計と実装の乖離を記録する errata
```

## C-6: Errata 運用

実装中に以下のような乖離が見つかった場合は、コードを無理に設計書へ合わせず、以下を行う。

```text
1. 事実を実測する
2. v20.2.4 へ errata を追記する
3. 契約番号を振る
4. テストを追加する
5. 実装へ反映する
```

典型例:

```text
- clearFFTOutputOnError の呼び出し箇所数が異なる
- FftStage に対応できない既存 stage 番号がある
- icx が /MTd を受け付けない
- ASan と MKL/IPP の組み合わせで false positive が出る
- reserved slot 64 では不足する
- quarantine 上限が不適切
```

ERRATA 命名規則: `ERRATA-{Phase}-{番号}`（例: `ERRATA-PHASE0-1`）
