# Practical Stable ISR Bridge Runtime — 設計書 v4.3（最終確定版）

**Document Version:** 4.3
**Date:** 2026-06-19
**Based on:** v4.2 + レビュー指摘4点の反映
**Status:** 最終確定

---

## v4.2 → v4.3 変更点一覧

| # | 項目 | v4.2 | v4.3 | 理由 |
|---|---|---|---|---|
| 1 | **PersistentStateBlock** | version なし（個別acquire） | **version 復活**：read-version→read-fields→read-version retry | 論理スナップショットの完全性保証 |
| 2 | **currentWorld_ 削除順序** | commit/retire 先、テスト後 | **getCurrent→テスト→読取禁止→commit/retire→メンバ削除**の5段階 | hidden dependency の段階的検出 |
| 3 | **CrossfadeAuthorityRuntime** | FixedArray 化が「候補」と記載 | **実施しない**（単一スレッド所有権で十分） | 実装リスク > 利益 |
| 4 | **ISR-AUTH-004** | なし（3 invariant のみ） | **追加**：Pure Function 要求（deriveAuthorityState等） | 将来の退化防止 |
| 5 | **Phase 順序** | A→A→A→B→A→B 混在 | **Phase-1〜5 の明確な5段階** | 実装＋テスト＋CI の依存関係整理 |

---

## 第0章: 検証プロセス総括

| サイクル | 成果物 | 確定項目数 |
|---|---|---|
| 1st | validation_report.md | 12の実装済み項目確認 |
| 2nd | design_deep_investigation_report.md | 7つの未確定事項確定 |
| 3rd | basic_plan.v4.1.md | 4設計改善点反映 |
| 4th | basic_plan.v4.2.md | 6追加深堀項目確定 |
| **5th** | **basic_plan.v4.3.md** | **4指摘点反映＋全未確定事項ゼロ** |

### 使用ツール（全5サイクル共通）

Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 第1章: PersistentStateBlock（version 復活版）

### 1.1 設計思想

v4.1 で提案した version 方式を復活させる。
3つの atomic フィールドを個別 acquire load すると、書き込み途中の不整合値を取得する可能性がある。

**禁止例**:

```
Thread-A 書き込み中:  sequence=11 (書込済)  epoch=20 (書込前)  generation=31 (書込済)
Thread-B 読み取り中:  sequence=11            epoch=20            generation=31
                      ↑ 最新                ↑ 旧                ↑ 最新
                      → 存在しない状態（sequence=11 なのに epoch=20 は論理矛盾）
```

### 1.2 定義

```cpp
struct PersistentStateBlock {
    // ★ version を先頭に配置（キャッシュライン分離のため）
    std::atomic<uint64_t> version{0};

    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
        uint64_t snapVersion;  // 整合性確認用
    };

    // ★ 論理スナップショット: read-version → read-fields → read-version
    Snapshot snapshot() const noexcept {
        for (;;) {
            const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            const auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            const auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            const auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            const auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
            if (v0 == v1) [[likely]] {
                return Snapshot{seq, ep, gen, v0};
            }
            // ★ 書き込み途中: リトライ（通常は即完了）
        }
    }

    // ★ 論理更新: version++ → write-fields → version++
    void update(const Snapshot& s) noexcept {
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
        convo::fetchAddAtomic(version, uint64_t{1}, std::memory_order_acq_rel);
    }
};
```

### 1.3 整合性保証

- `snapshot()` の version 一致チェックで書き込み途中の不整合読み取りを防止
- `update()` の version 2回 increment で書き込み区間をマーク
- 読み取り側の retry loop は通常0〜1回で収束（書き込みは Non-RT 単一スレッド）
- `[[likely]]` 属性で分岐予測最適化

---

## 第2章: currentWorld_ 削除（5段階プロセス）

### 2.1 削除戦略

v4.2 の「STEP 1-5 一括」を、**5段階の独立 Phase** に分割する。
各 Phase でコンパイル＋テストを通過し、hidden dependency を段階的に発見する。

### Phase-A: getCurrent → RuntimeStore 委譲

```cpp
// ISRRuntimePublicationCoordinator.h
class RuntimePublicationCoordinator {
    const void* runtimeStore_{nullptr};  // ★ 注入用ポインタ
public:
    void setRuntimeStore(const void* store) noexcept { runtimeStore_ = store; }

    const void* getCurrent() const noexcept {
        // ★ RuntimeStore 経由で取得（runtimeStore_ がなければ従来の currentWorld_ を使用）
        if (runtimeStore_ != nullptr) {
            // ★ observePublishedWorld() 相当の処理
            //    RuntimeStore の型が template のため、void* で汎用化
            return /* RuntimeStore の observe() 結果 */;
        }
        return convo::consumeAtomic(currentWorld_, std::memory_order_acquire);
    }
};
```

**影響**: 0（内部実装変更のみ、API/ABI 互換）

### Phase-B: 全テスト移行

ISRSemanticValidationTests.cpp の17箇所の `getCurrent()` → `consumePublishedWorld(store)` に置き換え。

```cpp
// 変更前
if (coordinator.getCurrent() != &world1)

// 変更後
if (RuntimePublicationCoordinator::consumePublishedWorld(store) != &world1)
```

**影響**: テストファイル17箇所

### Phase-C: currentWorld_ 読み取り禁止

`getCurrent()` から `currentWorld_` 参照を削除し、RuntimeStore 参照のみにする。

```cpp
const void* getCurrent() const noexcept {
    // ★ RuntimeStore のみ（currentWorld_ フォールバック削除）
    return /* RuntimeStore の observe() 結果 */;
}
```

**影響**: 本番コード0（getCurrent はテスト専用）、ISR coordinator の runtimeStore_ 参照が必須になる

### Phase-D: commit/retire 内の currentWorld_ 操作削除

```cpp
// commit() から削除
// convo::publishAtomic(currentWorld_, newWorld, ...);  // ★ 削除

// retire() から削除
// auto observedCurrent = convo::consumeAtomic(currentWorld_, ...);  // ★ 削除
// convo::compareExchangeAtomic(currentWorld_, ...);  // ★ 削除
```

**影響**: commit() と retire() 内の3行削除

### Phase-E: currentWorld_ メンバ変数削除

```cpp
// ISRRuntimePublicationCoordinator.h から削除
// std::atomic<const void*> currentWorld_;  // ★ 削除

// ISRRuntimePublicationCoordinator.cpp の初期化子から削除
// , currentWorld_(nullptr)  // ★ 削除
```

**影響**: 宣言1行 + 初期化1行

### 2.2 observePublishedWorld() の活用

`AudioEngine::observePublishedWorld()` は既に `RuntimePublicationCoordinator::consumePublishedWorld(runtimeStore)` のラッパーとして存在する。ISR coordinator への RuntimeStore 参照注入は以下の形で行う：

```cpp
// AudioEngine.CtorDtor.cpp 内
// runtimePublicationBridge_ は AudioEngine のメンバとして初期化済み
// その後、RuntimeStore のアドレスを注入
runtimePublicationBridge_.setRuntimeStore(&runtimeStore);
```

---

## 第3章: CrossfadeAuthorityRuntime — FixedArray 化は実施しない

### 3.1 根拠

| 要素 | 現状 | FixedArray 化後 |
|---|---|---|
| **データ構造** | `std::vector<CrossfadeRecord>` | 固定長配列 + インデックス管理 |
| **同期** | なし（Timerスレッド専有） | CAS 必要（lock-free） |
| **メリット** | — | メモリ使用量の予測可能性 |
| **リスク** | — | インデックス管理バグ, バッファ不足, CAS 競合 |

### 3.2 Thread Ownership 分析

```
CrossfadeAuthorityRuntime の全メソッド呼び出し元:

registerCrossfade()  → DSPTransition::onPublishCompleted()  → Non-RT (MessageThread)
unregisterCrossfade() → DSPTransition::onTransitionComplete()  → Timer (Non-RT)
getActiveCrossfades() → DSPHandleRuntime::destroyQuarantineSlot()  → Non-RT
hasCrossfadeInvolving() → DSPHandleRuntime::destroyQuarantineSlot()  → Non-RT

→ 全経路が Non-RT 単一スレッド
```

### 3.3 結論

**Phase-P5（CrossfadeAuthorityRuntime FixedArray 化）は実施しない。**

Practical Stable ISR Bridge Runtime の観点では「Thread Ownership が明確」なら十分。
実装リスクを取って lock-free 化する価値はない。

代わりに、`std::vector` のまま以下の軽微な改善のみ行う：

```cpp
// 改善案（任意）: 事前確保
explicit CrossfadeAuthorityRuntime() {
    records_.reserve(8);  // 同時クロスフェードは最大1つ、安全マージン
}
```

---

## 第4章: ISR-AUTH-004 — Pure Function Invariant

### 4.1 定義

```
ISR-AUTH-004

Authority State の導出関数は Pure Function でなければならない。

根拠:
  deriveAuthorityState 等が AudioEngine や RuntimeStore を内部で参照すると、
  呼び出し順序やタイミングに依存した非決定的動作を引き起こす。
  「再導出可能」を保証するには、関数が入力引数のみに依存する必要がある。

遵守方法:
  - deriveAuthorityState(persistentSnapshot, runtimeWorld) は引数のみ参照
  - deriveExpectedState(persistentSnapshot) は引数のみ参照
  - reconcileAuthorityState(observed, expected) は引数のみ参照
  - いずれの関数も AudioEngine&, singleton, global, RuntimeStore を内部で参照しない

禁止:
  AudioEngine& engine
  singleton
  global variable
  RuntimeStore internal access

許可:
  関数引数からのみの情報取得
```

### 4.2 違反検出

```powershell
# CI ゲート: ISR-AUTH-004 違反検出
# deriveAuthorityState の引数が PersistentStateBlock::Snapshot + World* 以外の
# 型を含んでいないことを確認
Select-String -Path "src\core\AuthorityState.h" -Pattern "AudioEngine|singleton|runtimeStore"
# → 0件であること
```

### 4.3 既存の ISR-AUTH 群（全4件）

| Invariant | 内容 | ステータス |
|---|---|---|
| ISR-AUTH-001 | Authority State は PersistentStateBlock からのみ再構築可能 | v4.3 で達成予定 |
| ISR-AUTH-002 | Recovery 後は通常 Publish 経路で到達可能な状態と同値 | v4.3 で達成予定 |
| ISR-AUTH-003 | Publish 経路は Orchestrator → Coordinator の唯一経路のみ | **既に達成**（CI 監査可能） |
| ISR-AUTH-004 | Authority State 導出関数は Pure Function | v4.3 で達成予定 |

---

## 第5章: 最終 Phase 計画（確定版）

### Phase-1: 基盤導入（独立、並行可能）

| タスク | 成果物 | 依存 |
|---|---|---|
| 1a. PersistentStateBlock（version 付き） | `src/core/PersistentStateBlock.h` | なし |
| 1b. AuthorityDescriptor | `src/core/AuthorityDescriptor.h` | なし |
| 1c. Validator エッジケース追加（7ケース） | `PublicationValidatorIsolationTests.cpp` | なし |

### Phase-2: currentWorld_ 段階的削除（前編）

| タスク | 成果物 | 依存 |
|---|---|---|
| 2a. getCurrent → RuntimeStore 委譲 | `ISRRuntimePublicationCoordinator` + `setRuntimeStore()` | 1a |
| 2b. 全テスト移行 | `ISRSemanticValidationTests.cpp` (17箇所) | 2a |
| 2c. currentWorld_ 読み取り禁止 | `getCurrent()` からフォールバック削除 | 2b |

### Phase-3: 状態導出＋Recovery

| タスク | 成果物 | 依存 |
|---|---|---|
| 3a. deriveAuthorityState | `src/core/AuthorityState.h` | 1a, 2c |
| 3b. deriveExpectedState | 同上 | 3a |
| 3c. reconcileAuthorityState | 同上 | 3b |
| 3d. Recovery 統合 | `AudioEngine.Timer.cpp` executeRecoveryAction() | 3c |

### Phase-4: Invariant + CI

| タスク | 成果物 | 依存 |
|---|---|---|
| 4a. ISR-AUTH-001 CI ゲート | `.github/scripts/isr-verify-auth-001.ps1` | 1a |
| 4b. ISR-AUTH-002 CI ゲート | `.github/scripts/isr-verify-auth-002.ps1` | 3d |
| 4c. ISR-AUTH-003 CI ゲート | `.github/scripts/isr-verify-auth-003.ps1` | なし（現状確認） |
| 4d. ISR-AUTH-004 CI ゲート | `.github/scripts/isr-verify-auth-004.ps1` | 3a |
| 4e. currentWorld_ 完全削除（後編） | `commit/retire` の操作削除 + メンバ削除 | 2c |

### Phase-5: テスト拡充

| タスク | 成果物 | 依存 |
|---|---|---|
| 5a. Property Test（10,000回混在） | 新規テストファイル | 3d |
| 5b. 障害注入テスト（4シナリオ） | 新規テストファイル | 3d |

### 非実施項目（正式決定）

| 項目 | 理由 |
|---|---|
| CrossfadeAuthorityRuntime FixedArray 化 | Single Thread Ownership で十分。実装リスク > 利益 |

---

## 第6章: 完了条件

| 条件 | 確認方法 |
|---|---|
| PersistentStateBlock 導入 | `grep PersistentStateBlock src/core/` → 1件 |
| AuthorityDescriptor 導入 | `grep AuthorityDomain src/core/` → 1件 |
| getCurrent RuntimeStore 委譲 | `grep "runtimeStore_" src/audioengine/ISRRuntimePublicationCoordinator.*` → 1件 |
| テスト移行完了 | `grep "getCurrent()" src/tests/ISRSemanticValidationTests.cpp` → 0件 |
| currentWorld_ 完全削除 | `grep "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.*` → 0件 |
| deriveAuthorityState 実装 | `grep deriveAuthorityState src/core/` → 1件 |
| reconcileAuthorityState 実装 | `grep reconcileAuthorityState src/core/` → 1件 |
| CI ゲート4件 | `.github/scripts/isr-verify-auth-00*.ps1` が全 PASS |
| Validator テスト 45+ | `Select-String TEST_F\|TEST\( src/tests/PublicationValidatorIsolationTests.cpp` → 45+ |
| Property Test 10,000回 | Test 実行 PASS |
| 障害注入 4シナリオ | Test 実行 PASS |

---

## 第7章: 最終到達予測

```
現状:
  92-95%

Phase-1 完了後:
  95-96%

Phase-2 完了後（currentWorld_ 読取禁止）:
  96-97%

Phase-3 完了後（derive+reconcile+Recovery）:
  97-98%

Phase-4 完了後（currentWorld_ 完全削除 + CI）:
  98-99%

Phase-5 完了後（テスト拡充）:
  99-100%
```

v4.3 完了時、ConvoPeq は Practical Stable ISR Bridge Runtime の完成形に到達する。
