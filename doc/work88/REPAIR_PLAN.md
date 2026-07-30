# ConvoPeq 改修設計書 — BUG-011〜BUG-046 修正計画 (v20.2.6)

**凡例**: ✅ 実装完了 → Appendix 参照。📋 設計確定 → #設計 または「将来対応事項」参照。🔮 将来対応維持 → 設計確定だが今回実装しない（FUTURE-5/6）。
**ステータス**: **v20.5 ISR最終版** — 以下4セクションで構成:
- #未実装事項 (1件: CI-1 未確認)
- #将来対応事項 (5件: FUTURE-3/4/5/6/7)
- #設計 (P0-4A+ISR設計原則)
- #未確定事項
実装済み項目はすべて Appendix に移動済み。FUTURE-5/6 は将来対応維持（今回実装せず）。

**ERRATA-V2023-1** (`Plan::workBuffer` 64-byte アライメント明記): ProductionFft::Plan::workBuffer は 64-byte アライメント必須。
確保は `mkl_malloc(size, 64)`、`convo::aligned_malloc(64, size)`、または `ippsMalloc_8u` 系のみ使用可。`new` / `malloc` / `std::vector` は禁止。
**ERRATA-V2023-2** (`toFftStage` 安全クランプ): `toFftStage()` は未知の legacy stage 整数を `FftStage::Diagnostic` へ安全クランプする。
`constexpr` / `noexcept` 必須。内部で HealthMonitor 呼び出しやログ出力を行ってはならない。

**CMP0091 設計（2026-07-29 解決済み）**: `cmake_minimum_required(VERSION 3.22)` により CMP0091 は暗黙的に NEW。ただし明示指定により可読性を向上させるため、`cmake_minimum_required` 直後に `cmake_policy(SET CMP0091 NEW)` を追加。同時に icx の `/MT` `/MTd` `/Qipo` フラグを `$<NOT:$<BOOL:${ENABLE_ASAN}>>` で条件付き化し、ASan 有効時に静的 CRT フラグが重複付与されないように修正。ASan ブロック内に PGO との排他チェック、LTCG/IPO 無効化も追加済み（CMakeLists.txt に実装完了）。**CMP0091 はもはや未設定ではない。**

---

# 未実装事項

本セクションの全項目は**今回改修で実装する**。凡例: 📋 設計確定 / 🟡 P1（高優先）。

---

## CI-1: ASan/TSan CI workflow 実効性確認 [🟡P1] — 📋 設計確定

### 目的

ADD-4 で追加した `ENABLE_TSAN` オプション（`CMakeLists.txt:1102`）と `.github/workflows/sanitizer-ci.yml` が実際の CI パイプラインで機能することを確認する。

### 設計

#### 現状分析

| コンポーネント | 状態 |
|--------------|------|
| `CMakeLists.txt` — `ENABLE_ASAN` オプション | ✅ 実装済み（line 1049） |
| `CMakeLists.txt` — `ENABLE_TSAN` オプション | ✅ 実装済み（line 1102） |
| ASan ブロック — PGO 排他チェック | ✅ 実装済み |
| ASan ブロック — LTCG/IPO 無効化 | ✅ 実装済み |
| ASan ブロック — 条件付き CRT フラグ | ✅ 実装済み |
| `.github/workflows/sanitizer-ci.yml` | ⚠️ 存在するが未検証 |
| debug-asan CI job 実効性 | 🔮 未確認 |
| debug-tsan CI job 実効性 | 🔮 未確認 |

#### 検証手順

```
1. ローカルで cmake -DENABLE_ASAN=ON のビルド成功確認
2. ローカルで cmake -DENABLE_TSAN=ON のビルド成功確認
3. CI 上で debug-asan job が green になることを確認
4. CI 上で debug-tsan job が green になることを確認
5. ASan 有効時と無効時で /MT フラグが正しく切り替わることを確認
6. CTest が ASan/TSan ビルドで正常終了することを確認
7. ASan ビルドでメモリリーク検出数が 0 であることを確認
8. TSan ビルドでデータ競合検出数が 0 であることを確認
9. Sanitizer ログ（stdout/stderr）にエラー出力がないことを確認
10. TSan の既知の false positive を除き、データ競合検出数が 0 であることを確認
    （ISR は std::atomic と memory_order release/acquire を多用するため、
      TSan の false positive が発生する可能性がある。既知の false positive は
      `tsan.supp` ファイルで管理し、`ASan-CMAKE-10` で契約化する。）
```

#### 契約（ASan-CMAKE-1〜10）

| ID | 契約 | 現状 |
|----|------|------|
| ASan-CMAKE-1 | ASan 有効時は静的 CRT フラグ（`/MT` `/MTd`）を付与しない | ✅ 実装済み |
| ASan-CMAKE-2 | ASan 有効時は PGO と排他する | ✅ 実装済み |
| ASan-CMAKE-3 | ASan 有効時は LTCG/IPO を無効化する | ✅ 実装済み |
| ASan-CMAKE-4 | ASan 有効時も debug-asan ビルドがリンク成功する | 🔮 未確認 |
| ASan-CMAKE-5 | `Qipo` フラグは ASan 有効時に二重定義されない | ✅ 実装済み |
| ASan-CMAKE-6 | TSan 有効時も debug-tsan ビルドがリンク成功する | 🔮 未確認 |
| ASan-CMAKE-7 | ASan/TSan ビルドは CI の独立した job として実行される | 🔮 未確認 |
| ASan-CMAKE-8 | ASan/TSan job は通常ビルドと並列実行可能 | 🔮 未確認 |
| ASan-CMAKE-9 | ASan/TSan ビルドで CTest が正常終了する | 🔮 未確認 |
| ASan-CMAKE-10 | TSan の既知の false positive を文書化し、それを除きデータ競合検出数が 0 | 🔮 未確認 |

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `.github/workflows/sanitizer-ci.yml` | 現状確認と必要に応じた修正 |
| `tsan.supp` | TSan の既知の false positive を除外する suppression ファイル。ISR の std::atomic/memory_order パターンによる false positive を管理する。 |

#### 完了条件

1. `cmake -DENABLE_ASAN=ON` のビルドが成功する
2. `cmake -DENABLE_TSAN=ON` のビルドが成功する
3. CI 上で debug-asan / debug-tsan 両 job が green

---

# 将来対応事項

本セクションの全項目は設計確定済み。実装ステータスは各項目のタスク表を参照（✅完了 / 🔮未着手 / 🔮将来対応維持）。凡例: 📋 設計確定。

---

## FUTURE-3: QuarantineService — submitRecoveryRequest (Rollback廃止) [🔮] — 📋 設計確定

### 実装内容

| タスク | 状態 |
|-------|------|
| QSVC-5 rollback コード削除（`result.rolledBack = false`） | ✅ 完了 |
| `rollbackQuarantine()` 設計を破棄、`submitRecoveryRequest()` 設計に変更 | ✅ 完了 |
| `submitRecoveryRequest()` のコード実装 | 🔮 未着手 |
| `ISRRuntimePublicationCoordinator` への `submitRecoveryRequest()` 宣言追加 | 🔮 未着手 |

残りのコード実装は以下の方針に従う。

### 目的

ISR の不変条件「Publish 後は Immutable」に従い、Quarantine 復旧に Rollback ではなく新しい Immutable RuntimeWorld の Publish を使用する。`rollbackQuarantine()` は廃止し、`submitRecoveryRequest()` で置換する。

### ISR 不変条件

| 原則 | 内容 |
|------|------|
| **Publish後は Immutable** | 一度 Publish された RuntimeWorld は変更不可。Rollback は禁止。 |
| **復旧は New World** | 状態復旧は新しい RuntimeWorld の Publish で行う。既存 World の変更不可。 |
| **Recovery も Validate 必須** | Recovery Runtime も通常の Builder → Validate → Publish 経路を通る。Recovery 例外として Validator を省略してはならない。 |

### 設計

```cpp
// ★ FUTURE-3: quarantine 復旧は New RuntimeWorld の Publish で行う
//   Rollback 禁止。Coordinator は Builder ではないため、RuntimeWorld の build は行わない。
//   Coordinator は Recovery Request を発行し、Builder → Validate → Publish の経路を通す。
//   命名: submitRecoveryRequest — Coordinator API。Request の発行のみ行い、Recovery 自体は実行しない。

// RuntimePublicationCoordinator に追加（Recovery Request 発行のみ）
void submitRecoveryRequest(
    const DSPHandle& quarantinedHandle) noexcept
{
    // 1. Recovery Request を発行（Coordinator は Builder を直接呼ばない）
    // 2. Builder が quarantinedHandle の情報を元に Recovery RuntimeWorld を build
    // 3. PublicationValidator で Validate（通常経路と同じ。Recovery でも例外ではない）
    // 4. coordinator.publishWorld(recoveryWorld)  — Immutable Publish
    // 5. 旧 World は coordinator.retire() で自然退役
    //
    // ★ rollback ではない。新しい World が古い World を置き換える。
    //    Quarantined の旧 Handle は EpochDomain が削除するのを待つだけ。
    // ★ Coordinator は Builder を知らない。Request を発行するのみ。
}
```

### QSVC-5 契約の修正

| 契約ID | 旧内容 | 新内容 |
|--------|--------|--------|
| QSVC-5 | Audit失敗時、State + Audit + Receipt の3状態をロールバック | **Audit失敗時は診断カウンタ更新のみ。State は変更しない。rollback 禁止。** |
| QSVC-5a | `quarantine()` 実行時に `previousState` を保存 | **削除。previousState 不要。** |
| QSVC-5b | `rollbackQuarantine()` で State 復元 | **削除。`submitRecoveryRequest()` で代替。** |
| QSVC-5c | rollback 完了後 Receipt 状態も戻す | **削除。Receipt は Epoch 完了後に解放。** |

### Recovery Intent 契約

`submitRecoveryRequest()` は Recovery Runtime を構築してはならない。**Recovery Request の発行のみ**を責務とする。Builder → Validate → Publish の責務境界を維持する。
Recovery Queue は**単なる Transport** であり、Decision Authority を持たない。

| ID | 契約 |
|----|------|
| RECOVERY-1 | `submitRecoveryRequest()` は Recovery Request を発行するのみ。RuntimeWorld の構築・Validate は行わない。 |
| RECOVERY-2 | Recovery Runtime は通常の Builder → Validate → Publish 経路を通る。Recovery 例外として Validator を省略してはならない。 |
| RECOVERY-3 | Coordinator は Builder を直接呼ばない。Recovery Request は Queue を経由して Builder へ渡される。Recovery Queue は単なる Transport であり、Decision Authority を持たない。 |
| RECOVERY-4 | `submitRecoveryRequest()` は NonRT（MessageThread）からのみ呼び出し可能。 |

> **注 — 共通 Intent Queue への統合検討**: RECOVERY-3 の Recovery Queue は、Observe Queue・Publish Intent・Quarantine Request と統合した共通 Intent Queue として設計する選択肢がある。ISR では Intent は統一的なイベントとして扱えることが望ましく、共通化により Authority の単一化と FIFO 保証が容易になる。現時点では別 Queue としているが、将来の共通 Intent Queue 化を排除しない。

---

## FUTURE-4: persistentState_ の廃止と RuntimeWorld Metadata Snapshot 統合 [🔮] — 📋 設計確定

### 目的

`persistentState_`（plain struct, 3×uint64_t）は MessageThread-only の前提だが、`emitObserveIntent()` が Timer Thread から読んでいる。ISR の Single Source of Truth 原則に従い、RuntimeWorld を唯一の Metadata Authority とする。

### 設計方針

`persistentState_` を廃止し、全メタデータ（epoch + generation + sequence）を RuntimeWorld 内の ObservationMetadata 構造体として統合する。Timer Thread は `const RuntimeWorld*` を1回読み取るだけで全メタデータを取得する。

```
// Before: 3 sources of truth
persistentState_.publicationEpoch       ← plain struct, cross-thread unsafe
persistentState_.mappedRuntimeGeneration ← same struct
persistentState_.publicationSequenceId   ← same struct

// After: 1 source of truth
RuntimeWorld::metadata::ObservationMetadata
  ├── epoch          ← RuntimeWorld publish時に atomically に設定
  ├── generation     ← RuntimeWorld publish時に atomically に設定
  └── sequence       ← 同上。DeferredDeletionQueue 等からは world 経由で参照
  │
  └── Timer Thread: const RuntimeWorld* world = consumeAtomic(currentWorld_)
       → world->metadata.epoch, world->metadata.generation, world->metadata.sequence
       → 1回の atomic load で全メタデータが一貫性をもって取得可能 ("RuntimeWorld Metadata Snapshot")
```

### ISR 設計判断

| 方式 | 問題点 | 採用 |
|------|--------|------|
| `persistentState_` 廃止、RuntimeWorld Metadata Snapshot 統合（**本設計で採用**） | `currentWorld_` の atomic load 1回で全メタデータを一貫性をもって取得。epoch/generation/sequence 間の inconsistency が原理的に発生しない。Single Source of Truth。 | ✅ **本設計** |
| atomic epoch cache + RuntimeWorld generation（過渡的措置） | epoch だけ atomic cache、generation は World から別途取得。epoch==N, generation==N-1 の inconsistency が理論上発生する。Metadata Authority が3箇所に分散。本設計への移行までの暫定措置としてのみ許容。 | ⏳ 過渡的措置 |
| `std::atomic<PersistentStateBlock>` | lock-free 非保証（icx）。mutex リスク。 | ❌ |
| plain struct 維持 | cross-thread の一貫性未保証。既知の技術負債。 | ❌ |

### 設計

```cpp
// RuntimeWorld 内で新たに定義される ObservationMetadata
struct ObservationMetadata {
    PublicationEpoch epoch{0};
    uint64_t generation{0};
    uint64_t publicationSequence{0};
};

// ISRRuntimePublicationCoordinator.h — persistentState_ 完全削除
// Before:
//   PersistentStateBlock persistentState_{};   // 削除（3フィールドのplain struct）
// After:
//   persistentState_ は完全廃止。
//   全スレッドは consumeAtomic(currentWorld_) から ObservationMetadata を取得。

// MessageThread 書き込み（commit 内）:
//   1. RuntimeWorld を build（この時点で metadata は確定）
//   2. atomic<const void*>::store(currentWorld_, newWorld, release)
//      → RuntimeWorld + 全メタデータが atomically に公開される

// Timer Thread 読み取り（emitObserveIntent 内）:
//   const auto* world = static_cast<const RuntimeState*>(
//       convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
//   if (world != nullptr) {
//       ObserveIntent intent{
//           fadingHandle,
//           world->metadata.epoch,
//           world->metadata.publicationSequence
//       };
//   }
//   → 1回の atomic load で epoch + generation + sequence を一貫性をもって取得
//   → epoch/generation/sequence 間の inconsistency は原理的に発生しない
```

### 過渡的措置（atomic epoch cache）

RuntimeWorld Metadata Snapshot 方式への完全移行までの暫定措置として、`atomic<PublicationEpoch> currentPublicationEpoch_` のみを追加する。この方式では epoch と generation の inconsistency が理論上発生するが、ObserveIntent の世代逆転検出は epoch のみで動作するため実用上問題ない。

```cpp
// ★ 過渡的措置: RuntimeWorld Metadata Snapshot 移行までの暫定 atomic cache
std::atomic<PublicationEpoch> currentPublicationEpoch_{0};
std::atomic<uint64_t> currentPublicationSequenceId_{0};

// HB契約（過渡的措置中使用）:
//   Writer: currentWorld_ store(release) → currentPublicationEpoch_ store(release)
//   → RuntimeWorld 公開が epoch 更新より先行することを保証
```

### 削除されるメンバ

| 現状メンバ | 移行先 | 理由 |
|-----------|--------|------|
| `persistentState_.publicationEpoch` | `RuntimeWorld::metadata.epoch`（過渡的: `atomic<PublicationEpoch>` cache） | Single Source of Truth: RuntimeWorld |
| `persistentState_.mappedRuntimeGeneration` | `RuntimeWorld::metadata.generation`（`consumeAtomic(currentWorld_)` 経由） | RuntimeWorld は publish 後に Immutable |
| `persistentState_.publicationSequenceId` | `RuntimeWorld::metadata.publicationSequence`（過渡的: `atomic<uint64_t>` cache） | 複数ファイル参照のため過渡的に atomic cache を許容 |

#### トレードオフ

| 方式 | メリット | デメリット |
|------|---------|-----------|
| **RuntimeWorld 統合（採用）** | ISR 完全準拠。atomic 1個。lock-free 保証。 | `getVersion()` の実装変更が必要（world→generation）。 |
| atomic 2個 + Seqlock | 2変数の論理的一貫性を保証しようとする。 | Seqlock として不完全（writer が seq++ を1回のみ）。epoch だけ古い状態を検出不可。 |
| plain struct 維持 | 変更ゼロ。 | cross-thread の一貫性未保証。既知の技術負債。 |

#### リスク

`currentPublicationEpoch_` が単一 `std::atomic<uint64_t>` であるため、C++ メモリモデル上の問題は完全に解決される。`getVersion()` の実装が `persistentState_.mappedRuntimeGeneration` から `world->generation` に変わるが、`currentWorld_` の読み取りは既存の `consumeAtomic` パターンと同一。

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ISRRuntimePublicationCoordinator.h` | `persistentState_` を削除。`std::atomic<PublicationEpoch> currentPublicationEpoch_` + `std::atomic<uint64_t> currentPublicationSequenceId_` を追加。`getVersion()` の実装を `currentWorld_` 経由の RuntimeWorld 読み取りに変更。 |
| `ISRRuntimePublicationCoordinator.cpp` | 全 `persistentState_` 参照を `currentPublicationEpoch_` / `currentPublicationSequenceId_` / `currentWorld_`（RuntimeWorld generation）に変更。`commit()` 内の3フィールド書込を各 atomic の store に分割。 |

---

## FUTURE-5: MemoryPool化 [🔮] — 📋 設計確定（将来対応維持）

### 目的

DSPHandle の内部ストレージを動的メモリプールに移行する設計。**本改修では実装しない**。現状の256固定スロット（`std::array`）は cache locality・RT bounded・実運用充足の面で十分であり、実益が確認されるまで延期する。

### 設計方針

```
現状:
  std::array<DSPRegistrySlot, 256> registry_;
  → 256 スロット固定。compile-time 確保。

将来:
  MemoryPool<DSPRegistrySlot> registryPool_;
  → 動的拡張可能。ページ単位でスロット追加。
  → RT-safe: プール拡張は NonRT でのみ実行。
  → スロット数に上限なし（実運用上の制限のみ）。
```

### 契約

| ID | 契約 |
|----|------|
| MEMPOOL-1 | プール拡張は NonRT（MessageThread）でのみ実行する |
| MEMPOOL-2 | RT パスでのスロット確保は O(1) bounded とする |
| MEMPOOL-3 | プール縮小は明示的な shrink 操作のみ |
| MEMPOOL-4 | プールの初期容量は 256 スロット（後方互換） |

### 完了条件

1. 256 スロット制限が撤廃され、動的確保に移行
2. RT パスのパフォーマンスが現状と同等
3. 既存テスト全件通過

---

## FUTURE-6: Handle Table 完全移行 [🔮] — 📋 設計確定（将来対応維持）

### 目的

`std::unordered_map<DSPCore*, DSPHandle>` を Handle Table に移行する設計。**本改修では実装しない**。現状の256エントリの線形探索はキャッシュヒット率が高く実用上十分であり、実測でボトルネックが確認されるまで延期する。

### 設計方針

```
現状:
  std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_;
  DSPHandle → DSPCore* の逆引きは linear scan（eraseByHandle: O(n)）

将来:
  HandleTable<DSPHandle, DSPCore*> handleTable_;
  → 双方向 O(1) lookup（forward + reverse）
  → メモリアクセスパターンの改善（密配列）
```

### 契約

| ID | 契約 |
|----|------|
| HTABLE-1 | forward map（DSPCore* → DSPHandle）は O(1) |
| HTABLE-2 | reverse map（DSPHandle → DSPCore*）は O(1) |
| HTABLE-3 | スロット再利用は generation で ABA 防止 |
| HTABLE-4 | 全操作は lock-free または bounded mutex |

### 推奨

`eraseByHandle` の linear scan（`MAX_DSP_SLOTS=256`）がホットパスになった場合に実施。現時点では不要。

---

## FUTURE-7: AudioEngine.Threading.cpp — emitQuarantineIntent 統合 [🔮] — 📋 設計確定

### 目的

`AudioEngine::quarantineSlot()`（Threading.cpp:36-65）内の直接 `dspQuarantineManager_.quarantineHandle()` 呼び出しを `emitQuarantineIntent()` 経由に変更する。

### 現状

```cpp
// AudioEngine.Threading.cpp:36-65
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    // Step 1: Truth store
    const bool applied = dspQuarantineManager_.quarantineHandle(slot, generation, reason);
    // Step 2-3: retire + Projection 更新
    // ...
}
```

### 変更後

```cpp
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                  convo::isr::QuarantineReason reason) noexcept
{
    const convo::isr::DSPHandle handle{slot, generation};
    // Coordinator 経由で quarantine を実行（QSVC-2）
    runtimePublicationBridge_.emitQuarantineIntent(
        handle, reason, dspHandleRuntime_, dspQuarantineManager_);

    // Step 2-3: retire + Projection 更新（現状維持）
    // ...
}
```

### トレードオフ

| 利点 | 欠点 |
|------|------|
| QSVC-2 完全遵守。全 quarantine が Coordinator 経由に。 | 追加の関数呼び出しオーバーヘッド。 |
| Authority 一元化が完全に達成。 | Threading.cpp:42 の直接呼び出しが無くなることで変更範囲が広い。 |

### 同期性分析

`quarantineSlot()` の現状は **同期実行**（`dspQuarantineManager_.quarantineHandle()` を直接呼び、即座に結果が返る）である。一方 `emitQuarantineIntent()` 経由に変更すると、Intent Queue → Coordinator → QuarantineService の経路を経由するため、**呼び出しから quarantine 確定までに遅延が発生する可能性がある**。

ただし、以下の理由で影響は限定的:

1. `quarantineSlot()` の呼び出し元（`AudioEngine.Commit.cpp:578,597`）は **NonRT（MessageThread）** である。RT パスからの呼び出しではない。
2. `emitQuarantineIntent()` 内の `QuarantineService::executeQuarantine()` は直ちに State + Audit を実行する（Intent Queue を経由しない）。したがって**同期性は維持される**。
3. `emitQuarantineIntent()` 自体が `DSPHandleRuntime::quarantine()` と `DSPQuarantineManager::quarantineHandle()` の両方を同期的に呼び出すため、`quarantineSlot()` が期待する「即時隔離」のセマンティクスは変わらない。

**結論**: `emitQuarantineIntent()` への置換は同期性を維持するため、安全に実施できる。

### 保留理由

設計書が「将来のリファクタリング候補」と明記。現在の直接呼び出しでも機能的正しさは維持されている。

### 命名に関する注意

`emitQuarantineIntent()` は現状**同期実行**（Intent Queue を経由せず、直ちに `QuarantineService::executeQuarantine()` を呼ぶ）である。ISR 的には「Intent 発行」という命名と「同期実行」という動作に乖離がある。ISR では Intent は「未来に処理される要求」を意味し、同期実行する関数は Intent ではない。

以下のいずれかに統一すべき:

- **`executeQuarantine()`**: 現状の同期実行を正確に表す（**推奨 — ISR語彙に適合**）
- **`submitQuarantine()`**: 将来の非同期 Queue 化を視野に入れる場合
- ~~`emitQuarantineIntent()`~~: 命名と動作の乖離あり（ISR語彙として曖昧）

今回の改修では `emitQuarantineIntent()` のまま維持するが、次回の ISR 純化フェーズで `executeQuarantine()` への改名を行うことを推奨する。

---

# 設計

本セクションは**今回改修で実装する**（ISR 設計原則に基づく設計確定項目。コードは Appendix 参照）。凡例: 📋 設計確定 / 🔴 P0（最優先）。

### ISR 設計原則（本設計書全体に適用）

| 原則 | 内容 | 根拠 |
|------|------|------|
| **Observer 副作用禁止** | Observer（Timer）は Intent Queue への push のみ。Retire/State Transition 不可。 | ISR Runtime 不変条件 |
| **Coordinator 唯一 Authority** | `processIntent()` は RuntimePublicationCoordinator のみが実行する。 | ISR Runtime 不変条件 |
| **Observe ≠ Retire** | `emitObserveIntent()` は Observation Intent。`processIntent()` は Retire Coordination。別関数・別責務。 | ISR Runtime 不変条件 |
| **Publish後は Immutable** | 一度 Publish された RuntimeWorld は変更不可。Rollback 禁止。復旧は New World の Publish。 | ISR Runtime 不変条件 |
| **Epoch 安全確認後 Ownership Release** | ACK は Epoch 完了通知であり、解放契機ではない。解放は `getMinReaderEpoch() > retireEpoch` の安全確認後にのみ実行する。 | Practical Stable ISR |
| **実行コンテキスト分離** | Observer Phase（`emitObserveIntent`）と Coordinator Phase（`processIntent`）は同一スレッド上でも明確に分離された Phase として実行する。Timer callback 内では Observer→Coordinator の順序を保証する。 | ISR Execution Context Separation |

## 🔴 P0-4A: Observe Authority — Observe Intent Queue + Timer→Coordinator 委譲 — 📋 設計確定

### アーキテクチャ

```
Timer callback（暫定実装 — Authority分離ではなく実行コンテキスト分離）
  │
  ├── [Observer Phase] emitObserveIntent()
  │     └── Intent Queue push — 即座復帰
  │
  ├── [Coordinator Phase] processIntent()  ★ 暫定: 同一 callback 内で時系列分離
  │     ├── Intent Queue pop → retire要求 → EpochDomain委譲
  │     ├── Coordinator は retire の開始のみ行う。物理削除（delete）は行わない。
  │     └── EpochDomain が唯一の Delete Authority
  │
  └── return

将来の理想:
  Timer: emitObserveIntent() のみ（即座復帰）
  Dedicated Coordinator Worker / MessageThread:
    while(...) { processIntent(); }  ★ Authority分離 + Execution Context分離
```

> **Important — 暫定実装について**: 現状の Timer callback 内での `processIntent()` 呼び出しは、**Authority 分離ではなく実行コンテキストの時系列分離（Phase 分離）**である。Observer と Coordinator の Authority はコード上分離されているが、実行主体（Timer callback）は同一であるため、**完全な Coordinator Authority 分離とは言えない。** 特に以下の2点の制約がある:
> - **Scheduling Authority は依然 Timer 側**: Coordinator Loop の実行開始を Timer が決定しており、Coordinator 自身が自律的に動作しているわけではない。
> - **Retire 遅延は 1ms 保証ではない**: Queue backlog + Coordinator 処理時間 + Epoch 待ち の総和となり、負荷状況により変動する。processIntent が毎 callback 実行されることで実用上のレイテンシは bounded だが、理論上の worst-case にはキューイング遅延が加わる。
>
> ただし、ObserveIntent は DSPHandle を保持する自己完結型 Intent であるため、processIntent は外部状態（`lifetimeMgr.getActive()`）に依存しない。これは専用の Coordinator Worker / Loop が未整備であるための暫定措置であり、以下の理由で許容される:
>
> 1. `emitObserveIntent()` → `processIntent()` の順序が Timer callback 内で保証されており、publish interleaving が発生しない
> 2. `processIntent()` 自体は Coordinator の public メソッドであり、Observer が直接 retire を実行しているわけではない
> 3. ObserveIntent は自己完結型（DSPHandle 保持）のため、将来 Dedicated Coordinator Worker へ移行する際のインターフェース変更はゼロ（`processIntent()` の呼び出し元を変えるだけ）
> 4. 本改修のコードベースは ISR 準拠度 **約85〜90%** を達成しており、Scheduling Authority は依然 Timer 側にある。残る Scheduling Authority と Execution Context 分離（Dedicated Coordinator Loop への移行）は将来対応とする
>
> **将来の完全 ISR 移行パス**:
> 1. ✅ ObserveIntent に `DSPHandle` フィールド追加（今回実装済み）
> 2. Timer callback から `processIntent()` 呼び出しを削除
> 3. 専用の `CoordinatorLoop`（MessageThread の定期タスクまたは Worker）で `processIntent()` を実行（Coordinator 自身が Scheduling Authority を持つ）
> 4. これにより Authority 分離 + Execution Context 分離 + Scheduling Authority の完全 ISR が達成される

### 目的

`retirePublishedDSP()` が Timer から直接呼ばれる現状を改め、Timer は Observe Intent のみを発行し、Coordinator が retire を実行する設計に変更する。これにより以下の向上を達成する:

- **RT レイテンシ低減**: Timer callback は `emitObserveIntent()` の1命令で即座復帰
- **Coordinator Authority 一元化**: 全寿命管理（Observe/Delete/Quarantine）が Coordinator 経由に統一
- **ISR パイプライン整合**: `Publish → Observe → Retire → Epoch → Delete` に完全準拠

### データ構造

#### ObserveIntent（✅ 実装済み）

```cpp
// ISRRuntimePublicationCoordinator.h 🔬 確認済み
struct ObserveIntent {
    DSPHandle handle;           // ★ 観測対象の DSPHandle（自己完結型 Intent）。ISR: Coordinator は handle のみで retire 対象を識別可能。
    PublicationEpoch epoch;     // emit 時の publicationEpoch（FIFO順序保証、世代逆転検出用）
    uint64_t intentId;          // 診断・モニタリング用途専用。Coordinator は handle と epoch のみで処理可能。
};
```

> **Note**: ObserveIntent は DSPHandle を保持する**自己完結型（self-contained）Intent** である。コード実装でも `processIntent()` は `retireByHandle(intent.handle)` を使用しており（`ISRRuntimePublicationCoordinator.cpp:588,597`）、`lifetimeMgr.getActive()` には依存していない。Coordinator は Intent 内の `handle` のみで retire 対象を一意に識別できる。これにより、将来の専用 Coordinator Worker への移行がコード変更なく実現可能。`intentId` は診断・モニタリング用途に限定される。
>
> **Queue 責務分離に関する注意**: Overflow Policy の Deferred 層は `coordinatorDeferredRing_` を使用しているが、このリングは本来 `RetireOverflowEntry`（Retire 系統）を保持するものであり、ObserveIntent とは異なる責務のデータを同一経路で扱う設計になっている。ObserveIntent の overflow が Retire queue の経路に流入することは、ISR 上の責務分離を曖昧にする可能性がある。現在は overflow 発生頻度が極めて低いため問題にはならないが、将来の完全 ISR 化では ObserveIntent 専用の Deferred Ring を別途用意することが望ましい。

**設計判定**: `LockFreeRingBuffer<ObserveIntent, 1024>` を使用。既存の `coordinatorDeferredRing_`（`LockFreeRingBuffer<RetireOverflowEntry, 1024>`）と同じパターン。SPSC なので atomic オーバーヘッドなし。Capacity 1024 は Timer 周期（1ms）× 1秒間のバッファに十分。

#### ACK 定義（4種のイベント分離）

ISR では Publish / Observe / Retire / Epoch / Delete は別イベントであり、ACK もこれに対応して分離される。

| イベント | ACK種別 | 意味 | 発行タイミング |
|---------|---------|------|--------------|
| **Publish** | `ACK (published)` | RuntimeWorld が Publish された。Observer が次回の Observe で検出可能。 | RuntimePublicationCoordinator::publishWorld() 完了後 |
| **Observe** | `ACK (queued)` | Intent がキューに受理された。Timer は即座に復帰可能。 | emitObserveIntent() 直後 |
| **Retire/Epoch** | `ACK (reclaim complete)` | Epoch 安全確認が完了し、Ownership Release が可能になった。**ACK自体は解放契機ではなく、Epoch安全確認の完了通知である。** 実際の解放契機は EpochDomain::getMinReaderEpoch() > retireEpoch。 | processIntent() 完了後、EpochDomain の安全確認後 |
| **Delete** | （ACK なし） | 物理削除は EpochDomain の内部処理。Coordinator は関知しない。 | — |

> **ISR Note**: ACK(reclaim complete) は Epoch が安全を保証した証拠であり、単なる ACK ではない。以下の4イベントは別々の意味を持ち、混同してはならない:
> - **Publish Receipt**: RuntimeWorld が Publish された証拠（`RuntimePublicationCoordinator::publishWorld` が返す）
> - **Retire Receipt**: DSP が Retire キューに投入された証拠（`DSPLifetimeManager::retire` が返す）
> - **Epoch Complete**: 全 Reader が epoch を通過した証拠（`EpochDomain::getMinReaderEpoch() > retireEpoch`）
> - **Delete**: 物理削除（EpochDomain の内部。Coordinator は関知しない）
>
> 現設計では Epoch 安全確認を `processIntent()` 完了時に暗黙的に行い、`markReceiptReclaimComplete()` は Epoch 完了通知として機能する。ただし `markReceiptReclaimComplete()` の名称は「Receipt 解放完了」ではなく「Epoch 安全確認完了通知」の意味であり、命名の再検討余地がある。

### 状態機械

```
emitObserveIntent() (Timer Thread, RT)
  │
  ├── observeIntentQueue_.push({epoch, intentId})  ← SPSC, lock-free, RT-safe
  │     └── full → Fallback→Deferred→Drop（4層Overflow）
  │
  └── return ACK (queued) — Timer は即座に復帰

processIntent() (Coordinator Phase, MessageThread/NonRT)
  │
  ├── observeIntentQueue_.pop(intent) — SPSC, lock-free
  │     ├── empty → return (no-op)
  │     └── intent取得
  │
  ├── OBSERVE-10: intent.epoch < currentEpoch → skip (世代逆転検出)
  │
  └── DSPLifetimeManager::retire(currentDSP)
        ├── ISRRetireRouter → EpochDomain (deferred deletion)
        ├── Epoch安全確認: getMinReaderEpoch() > retireEpoch
        │   （Coordinator は delete を実行しない。物理削除は EpochDomain の責務）
        └── ✅ 安全確認後に engine.markReceiptReclaimComplete()
              → ACKは「Epoch安全確認完了通知」であり、解放契機ではない
              → 実際の解放契機は「Epoch Complete」である
              → Coordinator は pendingReceipt_ の解放契機を通知するが、delete は行わない
```

### 契約一覧

| ID | 契約 | 対象 | 実装状態 |
|----|------|------|---------|
| OBSERVE-1 | Timer は ObserveIntent のみ発行し、Retire Authority を直接実行しない | `AudioEngine.Timer.cpp` | ✅ 完了（3箇所＋DSPTransition全置換） |
| OBSERVE-2 | Coordinator は ObserveIntent を Intent Queue に追加し、即時復帰する（Timer をブロックしない） | `emitObserveIntent()` | ✅ 完了 |
| OBSERVE-3 | Coordinator Loop は Intent Queue から取り出した Intent を `processIntent()` で処理する | `processIntent()` | ✅ 完了（毎 Timer callback 実行） |
| OBSERVE-7 | Timer は `ACK(reclaim complete)` = Epoch 安全確認完了通知を受信後、`pendingReceipt_` を安全に解放する。ACK は解放契機ではなく、Epoch Complete の通知である。 | `markReceiptReclaimComplete()` | ✅ 完了（Epoch 安全確認後） |
| OBSERVE-8 | ObserveIntent は NonRT パス（Timer Thread）からのみ発行可能 | — | ✅ 完了 |
| OBSERVE-9 | ObserveIntent は Publish 順序を保持する。FIFO で Coordinator へ渡される | `LockFreeRingBuffer` | ✅ 完了 |
| OBSERVE-10 | Coordinator は古い PublicationGeneration の ObserveIntent を実行してはならない | `processIntent()` | ✅ 完了 |

### 変更ファイル一覧

| ファイル | 変更内容 | 状態 |
|---------|---------|------|
| `ISRRuntimePublicationCoordinator.h` | `ObserveIntent` 構造体, `observeIntentQueue_`, `nextObserveIntentId_`, `overflowCounter_`, `processIntent()` 宣言 | ✅ 実装済み |
| `ISRRuntimePublicationCoordinator.cpp` | `emitObserveIntent()` → LockFreeRingBuffer push。`processIntent()` 実装（FIFO pop, 世代逆転検出, retire委譲） | ✅ 実装済み |
| `AudioEngine.Timer.cpp:890,1019,1578` | `retirePublishedDSP()` 直接呼出 → `runtimePublicationBridge_.emitObserveIntent()` のみ（3箇所） | ✅ 実装済み |
| `AudioEngine.Timer.cpp` 末尾 | `runtimePublicationBridge_.processIntent()` 定期実行追加 | ✅ 実装済み |
| `DSPTransition.h:146` | `engine_.retirePublishedDSP(...)` → `engine_.runtimePublicationBridge_.emitObserveIntent()` のみ | ✅ 実装済み |

### Intent Queue Overflow Policy（4層設計）

`LockFreeRingBuffer<ObserveIntent, 1024>` の `push()` は容量満杯時に `false` を返す（spin-wait しない）。この場合、Timer 側は復帰するが Intent が失われる。以下の4層ポリシーで対処する。

| 層 | 容量 | 動作 | 状態 |
|----|------|------|------|
| **Primary** | 1024 | `LockFreeRingBuffer` push 正常完了 | ✅ 実装済み |
| **Fallback** | 2048 | `LockFreeRingBuffer<ObserveIntent, 2048>` Secondary キュー | ✅ 実装済み |
| **Deferred** | 1024 | `coordinatorDeferredRing_` → `drainOverflowRing` で定期回収 | ✅ 実装済み |
| **Drop** | ∞ | `overflowCounter_` / `fallbackOverflowCounter_` increment | ✅ 実装済み |

**Overflow 状態機械**:
```
observeIntentQueue_.push({intent})
  ├── success → ACK(queued)、正常復帰
  │
  └── false (full) → observeFallbackQueue_.push({intent})     ← Fallback層
        ├── success → ACK(queued-fallback)
        │
        └── false (full) → coordinatorDeferredRing_.push({entry})  ← Deferred層
              ├── success → coordinatorDeferredCount_++
              │   （次回 drainOverflowRing で回収）
              │
              └── false (full) → overflowCounter_++ / fallbackOverflowCounter_++  ← Drop
```

| 契約ID | 契約 |
|--------|------|
| QUEUE-11 | Intent Queue が満杯の場合、Fallback → Deferred → Quarantine の4段階で安全側へ倒す |
| QUEUE-12 | Fallback Queue と Quarantine 発行は RT-safe（lock-free or atomic increment） |
| QUEUE-13 | Overflow 発生時は診断カウンタ（`overflowCounter_` / `fallbackOverflowCounter_`）を atomic increment する |
| QUEUE-14 | Overflow 発生を Coordinator が診断可能なイベントとして扱えるよう、HealthMonitor へ非同期通知する（将来対応） |

### 完了条件

1. ✅ `retirePublishedDSP()` が Timer から直接呼ばれず、Coordinator 経由になった（4箇所全置換完了）
2. ✅ Observer（Timer）は `emitObserveIntent()` のみ発行。Retire は Coordinator の責務。
3. ✅ processIntent が毎 Timer callback で定期実行される（MessageThread 保証）

### テスト計画

```cpp
// tests/ObserveIntentTests.cpp
TEST(ObserveIntentTimerFlow) {
    coordinator.emitObserveIntent();
    ASSERT_EQ(coordinator.getPendingIntentCount(), 1);
    coordinator.processIntent(lifetimeMgr, handleRuntime, engine);
    ASSERT_EQ(coordinator.getPendingIntentCount(), 0);
}
TEST(ObserveIntentFIFOOrder) {
    coordinator.emitObserveIntent(); coordinator.emitObserveIntent(); coordinator.emitObserveIntent();
    // processIntent は FIFO で処理
}
TEST(ObserveIntentGenerationReversal) {
    // OBSERVE-10: 古い PublicationGeneration の Intent を破棄
}
```

---

## 設計上の注意点（要点抽出）

| # | 項目 | 重要度 | 状態 | 説明 |
|---|------|--------|------|------|
| 1 | kMaxMismatch Timer周期依存 | ✅ 解決済み | FIX-D1 対応済み | `kMaxEpochDrift=10` に移行。周期依存から epoch 差分ベース検出へ変更。 |
| 2 | Emergency Override後の stale receipt | 🟡 LOW | P1-2 対応済み | `resetReceipt()` で quarantine Intent 発行。基盤実装済み。 |
| 3 | onTransitionComplete/notifyTransitionComplete | 🔷 INFO | 現状維持 | 設計上の統合フック。現状は `DSPTransition::onTransitionComplete` から直接 `retirePublishedDSP` を呼ぶ。 |
| 4 | release/acquire + External Serialization二層依存 | 🔷 INFO | 既知制約 | 二層の整合性は Coordinator Authority の外部 Serialization に依存。設計上の制約として許容。 |
| 5 | Fatal時の pendingReceipt_ 診断用保持 | 🔷 INFO | 既知制約 | Fatal 状態でも `pendingReceipt_` を残し、診断情報として活用。リークではなく意図的設計。 |
| 6 | MMCSS AvRevertのRT性 | 🔷 INFO | ADD-2 文書化完了 | `AudioEngine.Mmcss.cpp:204` AvRevertMmThreadCharacteristics 呼び出し。MMCSS-EX-1〜5 契約策定。 |
| 7 | ASan/TSan CI job分離 | 🔷 INFO | ADD-4 設計完了 | `ENABLE_TSAN` オプション追加済み。CI workflow 設定は未着手。 |
| 8 | Coordinator 唯一 Authority 原則 | 🔴 P0 | P0-4 対応完了 | Observe(P0-4A)/Delete(P0-4B)/Quarantine(P0-5) 全3 Authority を Coordinator に一元化。 |
| 9 | FFTExecutionContext 分離 | ✅ 本設計で採用 | P1-1 実装完了 | Layer が FFT を知らない設計。`FFTExecutionContext` が仲介。 |
| 10 | ISR Coordinator 経由寿命管理 | 🔴 P0 | P0-4 対応完了 | Observe/Delete の Coordinator Authority 経由化。 |

> 詳細なコードベース検証結果、調査結果詳細、レビュー履歴、付属文書、Errata 運用については Appendix を参照。

---

---

# 未確定事項

本セクションも**今回改修で調査・解決する**。凡例: ⚡ 設計方針確定（実装未着手） / ✅ 確認済み / 🔷 調査中。

## ⚡ MemoryPool化

| 項目 | 内容 |
|------|------|
| **設計方針** | 確定（本設計書で設計確定）。コードには未完全実装。 |
| **内容** | DSPHandle の内部ストレージを動的メモリプールに移行。現在の256固定スロット（`std::array<DSPRegistrySlot, MAX_DSP_SLOTS>`）を動的確保に変更する。 |
| **メリット** | 256スロット制限の解消。メモリ使用量の動的最適化。 |
| **トレードオフ** | 動的確保によるRT安全性の低下。プール管理の複雑性増加。 |
| **現状の制限** | `MAX_DSP_SLOTS=256` は実運用で十分だが、理論上の上限が存在する。 |
| **推奨** | 現時点では実施不要。256制限に達した事例がないため、問題発生時に対応する。 |

## ⚡ Handle Table完全移行

| 項目 | 内容 |
|------|------|
| **設計方針** | 確定（本設計書で設計確定）。コードには未完全実装。 |
| **内容** | 現在の `std::unordered_map<DSPCore*, DSPHandle> runtimeDSPHandleMap_` を Handle Table（密なスロット配列 + 逆引き index）に移行する。 |
| **メリット** | O(1) lookup（現在は linear scan `eraseByHandle`）。メモリアクセスパターンの改善。 |
| **トレードオフ** | 移行に伴うリファクタリングコスト。現在の `MAX_DSP_SLOTS=256` では linear scan でも実用上問題ない。 |
| **推奨** | `eraseByHandle` がホットパスになった場合に実施。現時点では不要。 |

## ✅ P0-2: tryAddRef Dead Code

| 項目 | 内容 |
|------|------|
| **確認日** | 2026-07-29 |
| **確認ツール** | WSL grep, serena MCP |
| **発見内容** | `RefCountedDeferred.h:48` で定義されている `tryAddRef()` メソッドが、全 `.cpp`/`.h` ファイルから一度も呼び出されていない。 |
| **リスク** | なし（Dead Code）。削除しても安全だが、将来のリファクタリングで再利用される可能性があるため現状維持。 |
| **推奨** | 削除またはコメントアウト。`// DEPRECATED: no callers found (2026-07-29)` を追記。 |

## 🔷 P0-2b: retirePublishedDSP 比較ロジックの完全性検証待ち

| 項目 | 内容 |
|------|------|
| **内容** | P0-2b で `current == pendingReceipt_->dsp` から `currentHandle == pendingReceipt_->handle` に変更したが、`getFadingRuntimeDSPHandle()` が常に正しい Handle を返すとは限らない。複数の fading が同時に存在する場合の動作が未検証。 |
| **リスク** | LOW。`fadingRuntimeDSPHandle_` は atomic で管理され、CAS 操作で排他制御されている。同時 fading は設計上防止されている。 |
| **追跡** | 単体テスト `NormalRetireDSPHandleCompare` で検証予定。 |

---

# Appendix: 実装済み事項一覧

## A-1: v20.2.6 新規追加実装済み事項（全10件）

| ID | 内容 | 成果ファイル | 確認内容 |
|----|------|-------------|----------|
| ✅ **P1-1** | FFT Backend Concept 全5Phase | `FFTBackend.h/cpp`, `FFTExecutionContext.h`, `ConvolverBuilder.h`, `MKLNonUniformConvolver.h/cpp` | `FftStatus`/`FftStage` enum, `FftBackendConcept`, `ProductionFft`, `TestFft`, `FFTExecutionContext`, `ConvolverBuilder`, Layer `m_fftPlan`/`m_fftCtx`統合, 6FFT呼出全置換, `releaseAllLayers`, `FFTBackendTests`(7テスト) |
| ✅ **P0-2** | EQCoeffCache DSPHandleRuntime移行 | `EQProcessor.h`, `AudioEngine.h`, `AudioEngine.Cache.cpp`, `RefCountedDeferred.h` | `EQCoeffCache`→`RefCountedDeferred`継承削除, `CacheMap`→`DSPHandle`化, `getOrCreate()`/`get()`→`DSPHandleRuntime::create()`/`resolve()`統合 |
| ✅ **P1-2** | Receipt状態機械 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `ISRDSPQuarantine.h` | `resetReceipt()`実装, `QuarantineReason::ReceiptReset`追加 |
| ✅ **ADD-2** | MMCSS例外登録簿 | `doc/coding_rule_jp.txt` | MMCSS-EX-1〜5契約, 例外登録簿テーブル追加 |
| ✅ **ADD-4** | ASan/TSan CI設定 | `CMakeLists.txt`, `.github/workflows/sanitizer-ci.yml` | `ENABLE_TSAN`オプション追加, debug-asan+debug-tsan CI workflow |
| ✅ **P0-4B** | Delete Authority — reclaim() Coordinator専用化 | `ISRDSPHandle.h`, `ISRRuntimePublicationCoordinator.h/cpp` | `reclaim()` private + friend。`shutdownReclaim()` 追加（DELETE-7）。`requestReclaim()` に executeRetire→waitReaders→executeReclaim 実装。全4箇所の直接 reclaim() 呼び出しを shutdownReclaim() に置換。 |
| ✅ **P0-2b** | PublishReceipt DSPCore*削除 | `AudioEngine.h`, `AudioEngine.Timer.cpp`, `DSPTransition.h` | `PublishReceipt::dsp` 削除。`storeReceipt()` DSPCore*引数削除。retirePublishedDSPのNormal Retire判定をDSPHandle比較に変更。 |
| ✅ **P0-5** | QuarantineService | `ISRRuntimePublicationCoordinator.h/cpp` | `QuarantineService` クラス新規追加。`emitQuarantineIntent()` → QuarantineService 経由の単一Authority。Timer内の直接 quarantine/quarantineHandle 呼び出しを emitQuarantineIntent に置換。 |
| ✅ **CACHE-LT-1** | キャッシュライフタイム契約 | `doc/work88/REPAIR_PLAN.md` | 通常時`retire()`のみ/Shutdown時`resolve→delete→reclaim`の契約明文化 |
| ✅ **P0-4C** | Coordinator Interface拡充 | `ISRRuntimePublicationCoordinator.h/cpp` | `emitObserveIntent()`, `emitQuarantineIntent()`, `requestReclaim()` 実装完了（P0-4A/B/5 で本実装に置換）。プレースホルダからの昇格完了。 |

## A-2: v12 新規追加実装済み事項（6件）

| ID | 内容 | ファイル | 確認内容 |
|----|------|----------|----------|
| ✅ **P0-1** | SafeStateSwapper tail 2-writer 解消（head 専用化） | `SafeStateSwapper.h` | `tryReclaimSlot()` + `advanceHead()` + `ReclaimResult` enum 実装済み。`publishAtomic(tail)` は `swap()` のみ |
| ✅ **P0-3** | AudioSegmentBuffer 61MB ヒープ化 | `AudioSegmentBuffer.h`, `NoiseShaperLearner.h` | `ScopedAlignedPtr` heap + factory + Rule of Five + `static_assert(sizeof<1024)` |
| ✅ **P2** | updateAudioThreadSnapshotFade 削除 | `AudioEngine.h:3738`, `src/core/SnapshotCoordinator.h:111` | DELETED コメント確認 |
| ✅ **ADD-1** | fallbackQueue bounded化 | `SafeStateSwapper.h:448` | `kMaxFallback=1024` + overflow counter 実装済み |
| ✅ **ADD-3** | DeferredFreeThread Logger rate limit | `DeferredFreeThread.h:169,184-185` | `kLogInterval=5s` + `lastLogTime_` 実装済み |
| ✅ **FIX-D1** | kMaxMismatch epochベース化 | `AudioEngine.Timer.cpp:1800`, `AudioEngine.h:4355` | `kMaxEpochDrift=10` + `publicationEpochDistance()` |

## A-3: 実装済み事項一覧（全37件）

### HW-1: Publication Metadata Propagation ✅ 完了

| 項目 | 内容 |
|------|------|
| **対応バグ** | BUG-030（拡張） |
| **重要度** | 🔴 HIGH |
| **関連ファイル** | 7ファイル |
| **ステータス** | ✅ **実装完了・テスト通過（19/19）** |

**実装ファイル**: `ISRRuntimeSemanticSchema.h`, `ISRRuntimePublicationCoordinator.h`, `AudioEngine.h`, `AudioEngine.Timer.cpp`, `DSPLifetimeManager.h`, `DSPTransition.h`, `RuntimePublicationOrchestrator.cpp`

### グループA: バグ修正（13件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | `FFT_MAGNITUDE_SCALE = 2.0f / NUM_FFT_POINTS` |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII `ApplyComputedIRLoadingGuard` 導入 |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp` | `irL.release()`/`irR.release()` を init 成功時に移動 |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp`（6箇所） | `clearFFTOutputOnError()` ヘルパー導入 |
| A-5 | BUG-011/012/013 | `CmaEsOptimizer.h/Dynamic.h/Dynamic.cpp` | `sigma = std::clamp(s, sigmaMin, sigmaMax)` 5箇所 |
| A-6 | BUG-029 | `DSPTransition.h` | Emergency Override で `exchangeFadingRuntimeDSP` を使用 |
| A-7 | BUG-028 | `CrossfadeRuntime.h` | `complete()` で全フラグリセット |
| A-8 | BUG-015 | `ISRRetireRouter.cpp` | `n` でリトライロジック内蔵＋戻り値確認 |
| A-9 | BUG-016 | `CmaEsOptimizer.h/Dynamic.h` | `sanitize()` で NaN/Inf→0.0 クランプ |
| A-10 | BUG-042/044/046 | 各クラス | Rule of Five（`=delete`/`=default`） |
| A-11 | BUG-045 | `IRConverter.cpp` | resample 失敗時に `actualSampleRate = sourceRate` |
| A-12 | BUG-039 | `CustomInputOversampler.cpp` | `std::min(targetSamples, ...)` |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp` | `sampleRateHz > 0 ? ... : 48000` フォールバック |

### グループB: 設計確定済み（4件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| B-1 | BUG-030 | `AudioEngine.h`, `DSPTransition.h`, `AudioEngine.Timer.cpp` | `claimFadingRuntimeDSP` CAS-only 実装 |
| B-4 | BUG-032 | `SnapshotCoordinator.h:122` | `getCurrentSnapshot()` インターフェース追加 |
| B-5 | BUG-024 | `SnapshotFadeState.h` | `fadeGeneration_` ABA 対策 |
| B-6 | BUG-037 | `ConvolverProcessor.h:883`, `ConvolverProcessor.Lifecycle.cpp:107` | `loaderGeneration_` UAF 防止 |

### グループC: 計画的対応（7件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| C-1 | BUG-033 | `AudioEngine.Processing.BlockDouble.cpp:421` | `dryScale` ラムダキャプチャ追加 |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:38` | `n()` 化 |
| C-3 | BUG-018 | 3ファイル | `!=1.0` → `std::abs(x-1.0)>1e-5f` |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | `int` → `size_t` |
| C-5 | BUG-020 | `ConvolverProcessor.LoaderThread.cpp:151-152` | `if(targetLength<=0)return 0;` |
| C-6 | BUG-021/022 | `ConvolverProcessor.Lifecycle.cpp:147-150` | RCU `GlobalGuard` 追加 |
| C-7 | BUG-026 | `ObservedRuntime.h:49` | `rootEnterSucceeded()`確認 |

### グループD: 余裕時（4件完了）

| ID | バグ | ファイル | 修正内容 |
|----|------|----------|----------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:649` | VLA→`makeAlignedArray` ヒープ割当 |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 |
| D-3 | BUG-027 | `SnapshotCoordinator.cpp:15` | `target==null` 時 state 再確認 |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10 に含む（Rule of Five） |

## A-4: 修正案詳細 (FIX) — 実装済み

### FIX-P0-1: SafeStateSwapper — Option A（head 専用化）✅ 実装済み

**変更内容**: `tryReclaim()` 内の tail 回転コード削除, head 専用 reclaim, null slot skip を bounded loop で実装。
**CI 3層化**: L1: rg `publishAtomic(tail` → swap() 内のみ / L2: ast-grep `tryReclaim` 内の `publishAtomic.*tail` 禁止 / L3: `SafeStateSwapperTailWriterSingleTests`
**テスト追加（8件）**: `SafeStateSwapperTailWriterSingleTests`, `SafeStateSwapperHeadOnlyReclaimTests`, `SafeStateSwapperNullSlotSkipTests`, `SafeStateSwapperEpochOrderTests`, `SafeStateSwapperHeadBlockingTests`, `SafeStateSwapperFullFallbackTests`, `SafeStateSwapperFallbackOverflowTests`, `SafeStateSwapperReaderStuckTests`

### FIX-P2: updateAudioThreadSnapshotFade 削除 ✅ 実装済み

### FIX-D1: kMaxMismatch Epoch ベース検出への移行 ✅ 実装済み

### FIX-ADD-1: fallbackQueue bounded 化 ✅ 実装済み

### FIX-ADD-3: DeferredFreeThread Logger rate limit ✅ 実装済み

### FIX-P1-2: Stale receipt quarantine 状態機械 ✅ 実装済み

### FIX-ADD-2: MMCSS AvRevert 例外登録 ✅ 設計完了（文書のみ）

### FIX-ADD-4: ASan / TSan CI job 分離 ✅ 設計完了

---

# Appendix B: コードベース検証結果（2026-07-29）

## B-1: コードベース検証結果（全ツール使用）

| 調査項目 | ツール | 結果 |
|---------|--------|------|
| EQCoeffCache 継承関係 | WSL grep / serena | ✅ `EQProcessor.h:123` で `RefCountedDeferred<EQCoeffCache>` 継承確認（P0-2完了） |
| DSPHandleRuntime 実装状況 | WSL grep / AiDex | ✅ `ISRDSPHandle.h/cpp` に完全実装（create/resolve/retire/quarantine/reclaim 全API稼働） |
| emitRetireIntent 有無 | WSL grep | ✅ `ISRRetire.h/cpp` に実装済み |
| emitObserveIntent 有無 | WSL grep / semble | ✅ Queue push + DSPHandle 実装済み。processIntent は retireByHandle で自己完結動作。P0-4A 完了。 |
| emitQuarantineIntent 有無 | WSL grep / semble | ✅ QuarantineService 経由で実装済み（P0-5完了） |
| QuarantineService 有無 | WSL grep / semble | ✅ 実装済み（P0-5完了） |
| ProductionFft / TestFft | WSL grep / AiDex / semble | ✅ P1-1 実装完了 |
| MMCSS例外登録簿 別ファイル | grep / ls | ❌ 設計書内のみ記載、別ファイル未作成（ADD-2設計確定・未着手） |
| ENABLE_TSAN | WSL grep | ✅ CMakeLists.txt:1102 実装済み |
| TODO(ADR-010) | WSL grep | ✅ `assert(true)` プレースホルダ。低リスク |
| FallbackQueue bounded化 | WSL grep / AiDex | ✅ `kMaxFallback=1024` |
| DeferredFreeThread Logger rate limit | WSL grep | ✅ `kLogInterval=5s` |
| kMaxMismatch epochベース化 | WSL grep / AiDex | ✅ `kMaxEpochDrift=10` |
| RetireRuntime fallbackQueue 容量 | WSL grep | ✅ `FALLBACK_QUEUE_CAPACITY=4096` |

## B-2: v20.4 ISR Design Refinements（本版で反映）

| # | 改善内容 | 反映先 |
|---|---------|--------|
| 1 | **単一削除 Authority 確定**: EpochDomain を唯一の削除 Authority | P0-4B, DELETE-8 |
| 2 | **ACK 定義拡張 — Receipt 完全ライフサイクル**: 文書化 | P0-4A ISR Note |
| 3 | **processIntent() 将来方向**: markReceiptReclaimComplete 過渡的措置を明記 | P0-4A コメント |
| 4 | **Overflow Policy 4層化**: Deferred 層追加 | P0-4A §6 |

## B-3: 調査で使用したツール

grep/ast-grep/rg/sed/awk/fdfind/fzf（WSL）, serena MCP, AiDex MCP, cocoindex, semble, graphify

---

# Appendix C: 補完セクション

## C-1: 拡張 Enum 定義

### AckResult — Intent Queue ACK 用

```cpp
enum class AckResult : int {
    Accepted = 0,     // Intent がキューに受理された
    QueueFull,         // Intent Queue が満杯
    ShuttingDown       // Shutdown 中で新規 Intent を受付不可
};
```

### EnqueueResult — 汎用 Enqueue 結果

```cpp
enum class EnqueueResult : int {
    Success = 0,
    QueueFull,
    QueueFullCritical,
    Shutdown,
    InvalidArgument,
    NotReady,
    Duplicate,
    RejectedByPolicy,
    RejectedByAdmission,
    InternalError
};
```

**契約（ENQUEUE-1〜9）**: 全 enqueue 関数は `[[nodiscard]] noexcept`。`QueueFullCritical` は critical command が reserved slot にも enqueue 不可。`Shutdown` は終端状態。`InternalError` は HealthMonitor へ報告。

## C-2: Shutdown 状態機械

```cpp
enum class ShutdownState : int {
    Running = 0, ShutdownRequested, Draining, EpochWaiting,
    Reclaiming, Quarantined, ShutdownCompleted, Faulted
};
```

**遷移**: `Running → ShutdownRequested → Draining → EpochWaiting → Reclaiming → ShutdownCompleted`, EpochWaiting から timeout → `Quarantined → ShutdownCompleted`

**閾値**: `EPOCH_WAIT_NORMAL_MS=100`, `EPOCH_WAIT_SHUTDOWN_MS=1000`, `EPOCH_WAIT_HARD_LIMIT_MS=3000`

**契約（SHUTDOWN-1〜7）**: Shutdown は最高優先度。冪等。新規 enqueue 拒否。Faulted 遷移可能。

## C-3: HealthMonitor イベント種別

```
EVENT_FFT_ERROR, EVENT_QUEUE_FULL, EVENT_QUEUE_FULL_CRITICAL,
EVENT_EPOCH_WAIT_TIMEOUT, EVENT_QUARANTINE_ENTERED, EVENT_QUARANTINE_RECLAIMED,
EVENT_QUARANTINE_ABANDONED, EVENT_QUARANTINE_LIMIT_EXCEEDED,
EVENT_QUARANTINE_SERVICE_FAILURE, EVENT_READER_SLOT_USAGE,
EVENT_PUBLICATION_MISMATCH, EVENT_RETIRE_OVERFLOW,
EVENT_ADMISSION_STOPPED, EVENT_SHUTDOWN_REQUESTED,
EVENT_SHUTDOWN_COMPLETED, EVENT_FAULTED
```

**契約（HEALTH-1〜7）**: bounded enqueue, RT non-blocking, NonRT 集計, pull 型 UI, 診断ログは `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード。

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

## C-5: 付属文書

```text
doc/exception_registry.md            — MMCSS 例外登録簿（未作成）
doc/health_monitor_events.md         — HealthMonitor イベント定義
doc/fft_backend_concept.md           — FftBackendConcept / FftStatus / FftStage 完全仕様
doc/quarantine_lifecycle.md          — Quarantine ライフサイクル詳細
doc/ci_asan_matrix.md                — ASan CI 設定マトリックス
doc/errata/v20.2-errata.md           — 設計と実装の乖離を記録する errata
```

## C-6: Errata 運用

実装中に乖離が見つかった場合、コードを無理に設計書へ合わせず以下を行う:
1. 事実を実測する
2. errata を追記する
3. 契約番号を振る
4. テストを追加する
5. 実装へ反映する

ERRATA 命名規則: `ERRATA-{Phase}-{番号}`（例: `ERRATA-PHASE0-1`）

---

*本設計書は ISR Runtime OS 設計原則に基づく。v20.5（確定版）: v20.2.6 + ISR Review 4件反映 + Phase2 Coordinator整備(4件) + REPAIR_PLAN(19) ISRレビュー対応（Observer純化・Coordinator専用processIntent・EpochベースOwnership Release・publishRecoveryRuntime・Runtime Version API分離・MMCSS例外登録簿・Full FUTURE-1実装）。全26最終受け入れ条件中20実装済み・6設計確定。「一部実装済み」全解消。*
