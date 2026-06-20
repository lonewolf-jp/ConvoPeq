# Practical Stable ISR Bridge Runtime — 設計書 v4.6（実装開始版）

**Document Version:** 4.6
**Date:** 2026-06-20
**Based on:** v4.5 + レビュー指摘6点の反映
**Status:** 実装開始版（全設計判断完了）

---

## v4.5 → v4.6 変更点一覧

| # | 項目 | v4.5 | v4.6 | 根拠 |
|---|---|---|---|---|
| 1 | **PersistentStateBlock** | seqlock 必須（version + guard + optional） | **2方式提示：簡易版(atomic交換) / 本格版(seqlock)** | 実装負荷と必要性のトレードオフ |
| 2 | **snapshot() 失敗モデル** | `nullopt` → Recovery break | **`expected<Snapshot, SnapshotError>`** で理由を返す | Recovery の fail-dead 防止 |
| 3 | **ISR-AUTH-004 CI** | 関数スコープ regex | **暫定CI（regex）+ 将来AST移行計画** | regex の脆さを認識した上での暫定運用 |
| 4 | **ISR-AUTH-006** | なし | **新設**：PersistentStateBlock ↔ RuntimeStore 整合性 | ISR-AUTH-001 と ISR-AUTH-002 の橋渡し |
| 5 | **seqlock 決定ポイント** | 明示せず | **Phase-0 として事前評価項目に昇格** | 実装着手前に方式を確定 |

---

## 第0章: 検証プロセス総括（全8サイクル完了）

| サイクル | 成果物 | 特記事項 |
|---|---|---|
| 1st | validation_report.md | 12の実装済み項目確認 |
| 2nd | design_deep_investigation_report.md | 7つの未確定事項確定 |
| 3rd | basic_plan.v4.1.md | reconcileAuthorityState + 論理スナップショット |
| 4th | basic_plan.v4.2.md | 6追加深堀（Orchestrator/Shutdown/Admission） |
| 5th | basic_plan.v4.3.md | 4指摘反映（version復活/削除順序/Pure Function） |
| 6th | basic_plan.v4.4.md | 深堀3項目（無限リトライ/CI/統合コード） |
| 7th | basic_plan.v4.5.md | 6改善（optional/偶数判定/ScopedGuard/AUTH-005） |
| **8th** | **basic_plan.v4.6.md** | **seqlock再評価＋AUTH-006＋失敗モデル明確化** |

### 使用ツール（全8サイクル）

Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 第1章: Phase-0 — seqlock 必要性の事前評価

### 1.1 調査結果

現行コードの metadata フィールドへのアクセスパターンを調査した。

| アクセス元 | スレッド | 内容 |
|---|---|---|
| `commit()` 読取 (line 81-83) | Non-RT (MessageThread) | monotonicity チェック用に全3フィールドを個別acquire読取 |
| `commit()` 書込 (line 106-108) | Non-RT (MessageThread) | 全3フィールドを個別release書込 |
| `getVersion()` (line 169) | **テストのみ** | mappedRuntimeGeneration_ のみ読取 |
| `isFullyDrained()` | Non-RT (MessageThread) | backlog カウンタのみ読取（metadata 非依存） |
| `getCurrent()` | **テストのみ** | currentWorld_ のみ読取 |

**結論**: 全アクセスが **同一 Non-RT スレッド** で発生する。
Single Writer + Single Reader（同一スレッド）のため、seqlock は理論上不要。

### 1.2 2方式の比較

| 観点 | 方式A: 簡易版（atomic<Snapshot>交換） | 方式B: 本格版（seqlock） |
|---|---|---|
| **実装** | `std::atomic<PersistentStateSnapshot>` を commit 時に store | version + ScopedGuard + optional の seqlock |
| **複雑さ** | 低（3 atomic → 1 atomic） | 中（version 管理 + retry + RAII ガード） |
| **スレッド安全性** | 同一スレッドで十分 | 将来の並行読取にも耐性 |
| **論理整合性** | commit 単位で保証 | read 側でも検証可能 |
| **メモリ使用量** | 24 bytes（1 atomic） | 32 bytes（4 atomic） |
| **リスク** | 将来の並行読取追加時に破綻 | 実装バグの余地 |
| **推奨** | **現状では十分** | 防御的設計 |

### 1.3 推奨

**Phase-0 で以下を決定する：**

- 現状の Single-Thread Ownership を今後も維持する → **方式A（簡易版）**
- 将来の並行読取の可能性を見込む → **方式B（本格版）**

本設計書では **両方式の詳細設計を提示** し、Phase-0 で選択可能とする。

---

## 第2章: PersistentStateBlock（2方式）

### 2.1 方式A: 簡易版（atomic<Snapshot> 交換）

```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

// PersistentStateSnapshot: 3つの永続フィールドを1つの atomic 構造体に集約
// ★ 同一スレッド内で全ての読み書きが発生する前提
//   commit() でまとめて store し、monotonicity チェックも同一スレッドで行う
struct PersistentStateSnapshot {
    uint64_t sequenceId{0};
    uint64_t epoch{0};
    uint64_t mappedGeneration{0};

    bool operator==(const PersistentStateSnapshot& o) const noexcept {
        return sequenceId == o.sequenceId && epoch == o.epoch
            && mappedGeneration == o.mappedGeneration;
    }
};

// ★  std::atomic<PersistentStateSnapshot> はロックフリー？
//    sizeof(PersistentStateSnapshot) = 24 > 16 (x64 HW atomic limit: CMPXCHG16B)
//    → MSVC STL が内部ミューテックスに fallback する可能性あり
//    → しかし commit() 単一スレッドのため問題なし
//    → 必要に応じて padding + 64byte align で cache line 分離

struct alignas(64) PersistentStateBlock {
    std::atomic<PersistentStateSnapshot> current{};

    // ★ commit() から呼ばれる：3フィールドを一括 atomic store
    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        convo::publishAtomic(current,
            PersistentStateSnapshot{seq, ep, gen},
            std::memory_order_release);
    }

    // ★ deriveAuthorityState / Recovery から呼ばれる：一括 atomic load
    PersistentStateSnapshot snapshot() const noexcept {
        return convo::consumeAtomic(current, std::memory_order_acquire);
    }

    // ★ monotonicity チェック（commit() 内で使用）
    //   事前に snapshot() で読み取った値と新しい値を比較
    static bool isMonotonic(const PersistentStateSnapshot& prev,
                            uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        const bool hasPrev = prev.sequenceId != 0 || prev.epoch != 0
            || prev.mappedGeneration != 0;
        if (hasPrev && seq <= prev.sequenceId) return false;
        if (hasPrev && ep <= prev.epoch) return false;
        if (hasPrev && gen <= prev.mappedGeneration) return false;
        return true;
    }
};

} // namespace convo
```

**commit() 統合**:

```cpp
void RuntimePublicationCoordinator::commit(...) {
    // 変更前
    const auto prevSeq = convo::consumeAtomic(publicationSequenceId_, ...);
    const auto prevEp  = convo::consumeAtomic(publicationEpoch_, ...);
    const auto prevGen = convo::consumeAtomic(mappedRuntimeGeneration_, ...);

    // 変更後（方式A）
    const auto prev = persistentState_.snapshot();
    if (!PersistentStateBlock::isMonotonic(prev, sequenceId, epoch, mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, ...);
        return;
    }
    // ...
    // 変更前
    convo::publishAtomic(publicationSequenceId_, sequenceId, ...);
    convo::publishAtomic(publicationEpoch_, epoch, ...);
    convo::publishAtomic(mappedRuntimeGeneration_, mappedGeneration, ...);

    // 変更後（方式A）
    persistentState_.commitFields(sequenceId, epoch, mappedGeneration);
}
```

### 2.2 方式B: 本格版（seqlock）

v4.5 の設計を踏襲するが、以下の改善を行う：

- `snapshot()` の戻り値を `std::expected<Snapshot, SnapshotError>` に変更
- `SnapshotError` で失敗理由を明確化

```cpp
enum class SnapshotError : uint8_t {
    None,
    WriterBusy,     // version が奇数のまま（書き込み中）
    RetryExceeded,  // 最大リトライ回数超過
};

// ★ std::optional ではなく expected で理由を返す
//   Recovery が状況に応じた判断を行えるようにする
std::expected<Snapshot, SnapshotError> snapshot() const noexcept {
    for (int attempt = 0; attempt < kMaxSnapshotRetries; ++attempt) {
        const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
        if ((v0 & 1u) != 0) {
            // ★ 書き込み中 → リトライ
            if (attempt == kMaxSnapshotRetries - 1)
                return std::unexpected(SnapshotError::WriterBusy);
            continue;
        }
        // ... fields load ...
        const auto v1 = convo::consumeAtomic(version, std::memory_order_acquire);
        if ((v1 & 1u) == 0 && v0 == v1)
            return Snapshot{seq, ep, gen, v0};
    }
    return std::unexpected(SnapshotError::RetryExceeded);
}
```

---

## 第3章: snapshot() 失敗モデルの明確化

### 3.1 エラー種別と Recovery での対応

| SnapshotError | 意味 | Recovery での対応 |
|---|---|---|
| `WriterBusy` | commit() 実行中（version 奇数） | リトライ（通常1回で成功） |
| `RetryExceeded` | writer が異常に長時間停止中 | `publishIdleWorldOnly()` で強制回復後リトライ |

### 3.2 Recovery 統合コード

```cpp
case convo::RecoveryAction::Restore: {
    const auto snapResult = persistentState_.snapshot();
    if (!snapResult) {
        switch (snapResult.error()) {
        case SnapshotError::WriterBusy:
            // commit() 実行中 → 即リトライ（通常は即完了）
            diagLog("[RECOVERY] PersistentStateBlock busy, retrying...");
            break;  // timer の次回 tick で再試行
        case SnapshotError::RetryExceeded:
            // writer が異常停止 → 緊急回復
            diagLog("[RECOVERY] PersistentStateBlock stalled, forcing recovery");
            // RuntimeStore の状態を確認
            const auto* runtimeWorld = observePublishedWorld();
            if (runtimeWorld == nullptr) {
                // world も空なら安全
                break;
            }
            // world があるなら Idle World を強制 publish
            publishIdleWorldOnly(getActiveRuntimeDSP(),
                convo::TransitionPolicy::HardReset);
            break;
        }
        break;  // snapshot 再試行は次回 tick で
    }

    const auto& snap = *snapResult;
    const auto* runtimeWorld = observePublishedWorld();
    const auto observed = deriveAuthorityState(snap, runtimeWorld);
    const auto expected = deriveExpectedState(snap);
    const auto rec = reconcileAuthorityState(observed, expected);
    // ... 修復処理 ...
    break;
}
```

### 3.3 Derive 関数の引数型統一

方式A/B どちらを選んでも、deriveAuthorityState の引数は統一する：

```cpp
// ★ 方式A でも方式B でも共通の Snapshot 型
//   方式A: PersistentStateSnapshot
//   方式B: PersistentStateBlock::Snapshot
//   どちらも sequenceId/epoch/mappedGeneration を持つ

// deriveAuthorityState は共通のスナップショット型を受け取る
template <typename World>
[[nodiscard]] AuthorityState deriveAuthorityState(
    const PersistentStateSnapshot& persistentState,
    const World* runtimeWorld) noexcept;
```

---

## 第4章: ISR-AUTH-006（新規）

### 4.1 定義

```
ISR-AUTH-006

PersistentStateBlock の内容は RuntimeStore の内容と矛盾してはならない。

根拠:
  sequenceId > 0 なのに runtimeStore.observe() == nullptr は論理矛盾。
  deriveAuthorityState() でこの矛盾を検出し、Recovery が publishIdleWorldOnly()
  で修復する必要がある。

遵守方法:
  - deriveAuthorityState() 内で persistentState と runtimeWorld の整合性をチェック
  - 矛盾検出時は AuthorityState::runtimeMissing = true を設定
  - Recovery が runtimeMissing を検出したら publishIdleWorldOnly() で修復

違反例:
  sequenceId=100, epoch=100, generation=100 なのに
  runtimeStore.observe() == nullptr
  → 矛盾。Recovery が必要。

整合例:
  sequenceId=0, epoch=0, generation=0 かつ runtimeStore.observe() == nullptr
  → Bootstrap 状態。正常。
```

### 4.2 AuthorityState 拡張

```cpp
struct AuthorityState {
    // ...（既存フィールド）...

    // ★ ISR-AUTH-006: PersistentStateBlock ↔ RuntimeStore 整合性
    bool runtimeMissing{false};  // persistent が世界を示すが world が存在しない
    bool persistentMissing{false}; // world が存在するが persistent が空
};
```

### 4.3 deriveAuthorityState 拡張

```cpp
template <typename World>
AuthorityState deriveAuthorityState(
    const PersistentStateSnapshot& ps,
    const World* runtimeWorld) noexcept
{
    AuthorityState result;
    result.publicationSequenceId = ps.sequenceId;
    result.publicationEpoch = ps.epoch;
    result.mappedRuntimeGeneration = ps.mappedGeneration;
    result.hasActiveRuntime = (runtimeWorld != nullptr);

    // ★ ISR-AUTH-006: 整合性チェック
    //   persistent が世界を示す (ps.sequenceId > 0) が world がない
    result.runtimeMissing = (ps.sequenceId > 0 && runtimeWorld == nullptr);
    //   world が存在するが persistent が空 (ps.sequenceId == 0)
    result.persistentMissing = (runtimeWorld != nullptr && ps.sequenceId == 0);

    result.hasPendingPublication = result.runtimeMissing;
    if (runtimeWorld != nullptr) {
        result.hasActiveCrossfade = runtimeWorld->execution.transitionActive;
    }
    return result;
}
```

### 4.4 CI ゲート

```powershell
# .github/scripts/isr-verify-auth-006.ps1
# ISR-AUTH-006: deriveAuthorityState 内で runtimeMissing / persistentMissing をチェックしていること
$targetFile = "src\core\AuthorityState.h"
$content = Get-Content (Join-Path $RepoRoot $targetFile) -Raw -Encoding UTF8
if ($content -match 'runtimeMissing' -and $content -match 'persistentMissing') {
    Write-Host "[PASS] ISR-AUTH-006"
} else {
    Write-Host "[FAIL] ISR-AUTH-006: runtimeMissing/persistentMissing not found"
    exit 1
}
```

---

## 第5章: ISR-AUTH-004 CI（暫定版 + AST 移行計画）

### 5.1 現状認識

v4.5 の関数スコープ限定 regex で誤検出は大幅に削減されたが、以下の限界がある：

| パターン | regex 検出 | 判定 |
|---|---|---|
| 関数本体直接の `publishAtomic(...)` | ✅ 検出 | 正 |
| ラムダ内の `publishAtomic(...)` | ❌ 未検出 | **誤（見逃し）** |
| ヘルパー関数呼び出し | ❌ 未検出 | **誤（見逃し）** |

### 5.2 暫定運用方針

| フェーズ | CI 方式 | 期間 |
|---|---|---|
| Phase-4d 直後 | **v4.5 の関数スコープ regex**（暫定） | 〜1ヶ月 |
| 次のイテレーション | **clang-tidy カスタムチェッカー** | 1〜3ヶ月後 |
| 最終形 | **clang-query AST matcher** | 安定後 |

### 5.3 clang-query による Pure Function 検証（参考）

```cpp
// clang-query スクリプト案（将来）
// 関数 deriveAuthorityState 内での atomic 操作を禁止
let fnDecl = functionDecl(hasName("deriveAuthorityState"))
let atomicCalls = declRefExpr(to(functionDecl(
    hasAnyName("publishAtomic", "consumeAtomic", "fetchAddAtomic")
)))
// fnDecl の子孫に atomicCalls が存在するか
// → 存在すれば違反
```

---

## 第6章: 全 Invariant 一覧（6件）

| # | Invariant | 内容 | CI ゲート |
|---|---|---|---|
| **001** | Authority State 再構築可能性 | PersistentStateBlock からのみ再構築可能 | `isr-verify-auth-001.ps1` |
| **002** | Recovery 状態同値性 | Recovery 後は通常経路で到達可能な状態と同値 | `isr-verify-auth-002.ps1` |
| **003** | Publish 経路唯一性 | Orchestrator → Coordinator の唯一経路 | `isr-verify-auth-003.ps1`（既存） |
| **004** | Pure Function | 導出関数は引数のみ参照（atomic 操作禁止） | `isr-verify-auth-004.ps1`（暫定regex） |
| **005** | 唯一永続メタデータ源 | PersistentStateBlock 以外の永続状態禁止 | `isr-verify-auth-005.ps1` |
| **006** | RuntimeStore 整合性 | PersistentStateBlock ↔ RuntimeStore の矛盾禁止 | `isr-verify-auth-006.ps1` |

---

## 第7章: 完了条件

```
【基盤】
grep "PersistentStateBlock\|PersistentStateSnapshot" src/core/ → 1件以上
grep "AuthorityDomain" src/core/ → 1件以上

【currentWorld_】
grep "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.* → 0件

【状態導出】
grep "deriveAuthorityState" src/core/ → 1件以上
grep "reconcileAuthorityState" src/core/ → 1件以上

【Invariant 違反ゼロ】
isr-verify-auth-001.ps1 → PASS
isr-verify-auth-002.ps1 → PASS
isr-verify-auth-003.ps1 → PASS
isr-verify-auth-004.ps1 → PASS
isr-verify-auth-005.ps1 → PASS
isr-verify-auth-006.ps1 → PASS

【テスト】
Validator tests: 45+ test cases PASS
Property Test: 10,000回 PASS
Fault Injection: 4 scenarios PASS
```
