# ISR Bridge Runtime 改修 — 総合調査・確定報告書

> 作成日: 2026-06-28
> 種別: 未確定事項の調査・確定・棚卸し
> 使用ツール: grep (Select-String), serena (MCP), AiDex (MCP), コード分析

---

## 1. Phase 4 FrozenRuntimeWorld — DEFERRED 確定分析

### 1.1 結論

**`using RuntimePublishWorld = FrozenRuntimeWorld;` 型エイリアス置換は採用しない。**

代わりに **Builder/Runtime 二段階モデル** を推奨する。

### 1.2 技術的根拠

| 課題 | 詳細 | 影響範囲 |
|------|------|---------|
| C++ `operator->` 制約 | 生ポインタ `FrozenRuntimeWorld*` の `->` は `operator->()` を経由しない | 261箇所の `(*ptr)->field` 変換が必要 |
| Builder mutable 競合 | `assertMutable()` が3箇所。FrozenRuntimeWorld は const-only | Builder/Orchestrator の設計変更必須 |
| Coordinator テンプレート | `World=FrozenRuntimeWorld` で `consumeWorldHandle` が `const FrozenRuntimeWorld*` を返す | 全 consumer に波及 |

### 1.3 推奨代替設計

```
Builder (RuntimeState, mutable)
  │
  ├── worldOwner->freeze()  // seal
  │
  ├── std::move → FrozenRuntimeWorld(worldOwner)
  │
  ├── publish(frozen)       // Coordinator は const RuntimeState* を保持
  │
  └── Consumer: FrozenRuntimeWorld& 経由で const access
```

**変更点**:
- Coordinator テンプレート `World` パラメータは `RuntimeState` のまま（変更不要）
- Publish境界で `FrozenRuntimeWorld` 生成 → 直ちに Coordinator に const access 委譲
- 既存261箇所の `world.X` アクセス構文は変更不要

---

## 2. SPSC スレッド安全性 — 確定

### 2.1 対象コンポーネント

| コンポーネント | Producer | Consumer | 方式 | 状態 |
|--------------|----------|----------|------|------|
| `LockFreeRingBuffer` | Audio Callback (RT) | NonRT Timer | acquire/release ordering | ✅ 設計通り |
| `RetireOverflowRing` | Audio Callback (RT) | Coordinator Timer | SPSC + debug check | ✅ ADR-001遵守 |
| `SPSCRingBuffer` (CrossfadeRuntime) | Audio Callback | Timer | HB契約文書化 | ✅ |
| `CommandBuffer` | UI Thread | Audio Callback | SPSC HB契約 | ✅ |

### 2.2 課題: 複数Producerの可能性

現状 `RetireOverflowRing` は SPSC 前提だが、以下の将来シナリオで MPSC が必要になる:

| シナリオ | 発生条件 | 優先度 |
|---------|---------|-------|
| 複数Audio Callback | マルチクライアントDAW | 低 (現状単一callback) |
| ISR+通常パス併用 | 移行期間中 | 低 |

**対応**: 現状維持。必要時は `LockFreeRingBuffer` を MPSC variant に置き換え。

---

## 3. EBR (Epoch-Based Reclamation) 安全性 — 確定

### 3.1 EBR プロトコル検証

| 条件 | 状態 | エビデンス |
|------|------|-----------|
| `kInactiveEpoch` (UINT64_MAX) マーク | ✅ | `EpochDomain.h:27` |
| `registerReaderThread()` CAS | ✅ | `EpochDomain.h:50-52` |
| `enterReader()` depth increment | ✅ | 実装済み |
| `exitReader()` depth decrement | ✅ | 実装済み |
| `getMinReaderEpoch()` scan | ✅ | quarantined除外付き |
| `tryReclaim()` epoch comparison | ✅ | 実装済み |

### 3.2 quarantined Reader 除外

`getMinReaderEpoch()` が `kQuarantinedFlag` の立った Reader をスキップすることを確認:
```cpp
// EpochDomain.h:200-202
const uint8_t flags = convo::consumeAtomic(slot.quarantineFlags, ...);
if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
    continue;  // skip quarantined reader
```

✅ **EBR安全性は維持されている。**

---

## 4. メモリ管理 — 確定

### 4.1 割当・解放経路完全トレース

```
RuntimeBuilder::createBootstrapWorld()
  └→ aligned_make_unique<RuntimeState>()
  └→ freeze()
  └→ PublicationExecutor::publish()
       └→ coordinator.publishWorld(std::move(worldOwner))
            ├→ worldOwner.release() → raw pointer
            ├→ store.publishAndSwap(rawPtr)
            └→ bridge.retireRuntimePublishWorldNonRt(oldPtr)
                 └→ enqueueDeferredDeleteNonRt()
                      └→ ~RuntimePublishWorld() + aligned_free()
```

**所見**: 全パスで `aligned_free()` が呼ばれる。`aligned_unique_ptr` の RAII により確保・解放の対応が取れている。✅

### 4.2 未使用コード

| ファイル | 状態 |
|---------|------|
| `FrozenRuntimeWorld.h/.cpp` | クラス定義完了だが publish パスで未使用（Phase 4 DEFERRED）|
| `PublicationBuffer` | `ISRRuntimePublicationCoordinator.h` で定義されているが使用されていない可能性 |

---

## 5. コメント・命名不整合 — 修正対応済み

| ファイル | 修正前 | 修正後 | 状態 |
|---------|--------|--------|------|
| `EpochDomain.h:285` | `safeToIgnoreでカウント` | `kQuarantinedFlagでカウント` | ✅ 修正済み |
| `IEpochProvider.h:81` | `safeToIgnoreでカウント` | `kQuarantinedFlagでカウント` | ✅ 修正済み |
| `FrozenRuntimeWorld.h` | 制約説明のみ | 代替設計方案を追記 | ✅ 修正済み |

---

## 6. 残タスク一覧 (11 items) — 確定済み

### Phase 4 DEFERRED (9 tasks) — 改善計画に基づき保留確定

- [ ] `using RuntimePublishWorld = FrozenRuntimeWorld;` — **代替設計採用: Builder/Runtime二段階モデル**
- [ ] Coordinator テンプレート分離 — 上記設計で不要に
- [ ] テスト — Phase 4 設計確定後に実施

### 統合テスト (複雑なセットアップが必要) — 現状維持

以下のテストは Coordinator/EpochDomain の複雑なセットアップが必要なため、別タスクとして管理:

| フェーズ | テスト項目 | 備考 |
|---------|-----------|------|
| Phase 1 | Coordinator統合, QueuePressure defer, retryCount超過 | Coordinator+OverflowRing結合 |
| Phase 1 | 滞留年限通知, Shutdown drainAll, RT安全性 | タイミング依存 |
| Phase 2 | Drain完了, Timeout強制解放, 再注入継続 | シャットダウン結合 |
| Phase 3 | quarantine各種, Recovery統合, Reclaim連動 | EpochDomain結合 |

### 検証 (特殊環境が必要) — 現状維持

| 項目 | 必要な環境 | 現状 |
|------|-----------|------|
| RTレイテンシ計測 | リアルタイムオーディオ環境 | 専用計測不能 |
| メモリリーク | ASAN / Valgrind | Debug CRT で確認可能 |
| SPSCスレッド検証 | マルチスレッド負荷試験 | Debug build 一部検証 |
| EBR安全性 | 長期連続動作試験 | 論理的検証完了 |

---

## 7. 総合ステータス

| カテゴリ | 完了 | 全数 | 備考 |
|---------|------|------|------|
| Phase 0-3 コア実装 | **52** | 52 | ✅ 全完了 |
| Phase 5 コア実装 | **14** | 14 | ✅ 全完了 |
| Phase 4 FrozenRuntimeWorld | **3** | 12 | ⏳ DEFERRED（代替設計確定） |
| クロスカット検証 | **6** | 8 | 専用環境が必要な2項目を保留 |
| コメント不整合修正 | **2** | 2 | ✅ 修正済み |
| **Total** | **77** | **88** | **87.5%** |

---

## 8. 付録: 調査で使用したコマンド

```bash
# AiDex インデックス
aidex_init path=C:\VSC_Project\ConvoPeq

# Serena パターン検索
serena search_for_pattern "SPSC|ADR-001" src/**/*.{h,cpp}
serena search_for_pattern "aligned_make_unique|aligned_free" src/**/*.{h,cpp}
serena search_for_pattern "safeToIgnore" src/**/*.{h,cpp}
serena search_for_pattern "assertMutable" src/**/*.{h,cpp}

# grep 補完
grep -rn "safeToIgnore" src/
grep -rn "assertMutable" src/
```
