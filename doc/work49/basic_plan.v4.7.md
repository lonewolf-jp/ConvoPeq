# Practical Stable ISR Bridge Runtime — 設計書 v4.7（実装開始版）

**Document Version:** 4.7
**Date:** 2026-06-20
**Based on:** v4.6 + 深堀4項目の確定
**Status:** 実装開始版（全設計判断・技術制約確定）

---

## v4.6 → v4.7 変更点一覧

| # | 項目 | v4.6 | v4.7 | 根拠 |
|---|---|---|---|---|
| 1 | **std::expected 使用** | C++23 `std::expected` 前提 | **自作 `Expected<T,E>` を使用**、または `Result<Snapshot>` 構造体 | プロジェクトが **C++20**（CMakeLists.txt:310） |
| 2 | **方式A atomic 制約** | `std::atomic<PersistentStateSnapshot>` を提示 | 制約明記：**sizeof=24 > 16 → MSVC STL 内部 mutex**（BlockingReasonStats と同様） | 既存コードの知見（ISRShutdown.h:63）を流用 |
| 3 | **currentWorld_ 参照** | 「6箇所」と記載 | **ISRRuntimePublicationCoordinator 内6箇所のみ**。他ファイル0件を確認 | 全ファイルスキャン完了 |
| 4 | **AuthorityDomain マッピング** | 概念のみ | **具体的な publishWorld 呼び出し元 × Domain マッピング** | 6箇所の呼び出し元を特定 |
| 5 | **non-atomic 読み取り** | 未考慮 | **方式A の 24byte atomic 問題を回避する non-atomic 代替案**を追加 | 最軽量オプションの提供 |

---

## 第0章: 検証プロセス総括（全9サイクル完了）

| サイクル | 成果物 | 確定項目数 |
|---|---|---|
| 1st–8th | v4.0–v4.6 | 全 Invariant・Phase 確定 |
| **9th** | **basic_plan.v4.7.md** | **4技術制約確定（C++20/atomic sizeof / currentWorld_網羅 / Domain実マッピング）** |

### 使用ツール（全9サイクル）

Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 第1章: Phase-0 事前評価 — 技術制約の確定

### 1.1 C++標準の確定

```
CMAKE_CXX_STANDARD = 20（CMakeLists.txt:310）
CMAKE_CXX_STANDARD_REQUIRED = ON（CMakeLists.txt:311）
全テストターゲット: cxx_std_20
```

**結論**: `std::expected`（C++23）は使用不可。
代わりに以下2案のいずれかを採用：

| 案 | 方式 | メリット | デメリット |
|---|---|---|---|
| **案A** | 自作 `Expected<T,E>` template | `std::expected` と同じ使用感 | 約50行の実装が必要 |
| **案B** | `struct Result { bool valid; SnapshotError error; Snapshot data; }` | 実装0行（struct定義のみ） | 戻り値のチェック漏れリスク |

**推奨: 案B**（struct Result）。ConvoPeq のコーディング規約に沿い、最小実装。

### 1.2 atomic 制約の確定

```
sizeof(PersistentStateSnapshot) = 24 > 16 (x64 HW atomic limit: CMPXCHG16B)
→ std::atomic<PersistentStateSnapshot> は MSVC STL が内部ミューテックスに fallback
→ 既に BlockingReasonStats（ISRShutdown.h:66）で同じ制約が文書化済み
```

**方式A の影響**: commit() 単一スレッドで使用するため、内部 mutex による性能影響はゼロ。
`alignas(64)` の指定は推奨（cache line 分離）。

### 1.3 currentWorld_ 参照の完全網羅

全ファイルスキャン結果：

```
src/audioengine/ISRRuntimePublicationCoordinator.h (6行: 宣言＋参照)
src/audioengine/ISRRuntimePublicationCoordinator.cpp (6行: init/commit/retire/getCurrent)
src/tests/ISRSemanticValidationTests.cpp (17行: テスト内 getCurrent())
```

**他ファイルでは0件確認。** `currentWorld_` の参照は ISRRuntimePublicationCoordinator 内に完全に閉じている。

### 1.4 AuthorityDomain 実マッピング

publishWorld の全実呼び出し元：

| 呼び出し元 | ファイル | Domain | Reason |
|---|---|---|---|
| Bootstrap 起動 | `AudioEngine.Init.cpp:48` | User | UserParameter |
| PrepareToPlay（サンプルレート変更） | `AudioEngine.Processing.PrepareToPlay.cpp:150,263` | User | UserParameter |
| ReleaseResources（シャットダウン） | `AudioEngine.Processing.ReleaseResources.cpp:151` | Shutdown | — |
| Timer publishIdleWorldOnly | `AudioEngine.Timer.cpp:435` | Recovery | TimerRecovery |
| DSPTransition onTransitionComplete | `AudioEngine.Transition.cpp:26` | DSPLifecycle | DSPTransition_Complete |
| PublicationExecutor（通常publish） | `PublicationExecutor.cpp:18` | User | UserParameter |

---

## 第2章: PersistentStateBlock（3方式）

### 2.1 方式A: 簡易版 atomic<Snapshot> 交換

v4.6 からの変更なし。但し以下の制約を明記：

- `std::atomic<PersistentStateSnapshot>` は sizeof=24 > 16 により **MSVC STL 内部 mutex** に fallback
- commit() 単一スレッドのため性能影響なし
- `alignas(64)` 推奨

```cpp
struct alignas(64) PersistentStateBlock {
    std::atomic<PersistentStateSnapshot> current{};
    // ...
};
```

### 2.2 方式B: seqlock（C++20 対応版）

v4.6 の seqlock 設計を C++20 に対応させるため `std::expected` を `Result` 構造体に置き換える：

```cpp
struct SnapshotResult {
    bool valid{false};
    SnapshotError error{SnapshotError::None};
    PersistentStateBlock::Snapshot data{};
};

// ★ C++20 対応: std::expected の代わりに Result 構造体
SnapshotResult snapshot() const noexcept {
    for (int attempt = 0; attempt < kMaxSnapshotRetries; ++attempt) {
        const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
        if ((v0 & 1u) != 0) {
            if (attempt == kMaxSnapshotRetries - 1)
                return {false, SnapshotError::WriterBusy, {}};
            continue;
        }
        const auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
        const auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
        const auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
        const auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
        if ((v1 & 1u) == 0 && v0 == v1)
            return {true, SnapshotError::None, {seq, ep, gen, v0}};
    }
    return {false, SnapshotError::RetryExceeded, {}};
}
```

### 2.3 方式C: non-atomic 読み取り（新規・最軽量）

**追加**: 24byte atomic の制約を回避するため、`PersistentStateSnapshot` を
単なる 3×uint64_t の非 atomic 構造体として保持し、commit() でのみ書き換える方式。

```cpp
struct PersistentStateBlock {
    // ★ non-atomic: commit() 単一スレッドからのみ書き換えられる
    //   読み取りも同一スレッド（またはテスト）からのみ
    //   スレッド安全性の問題は一切発生しない
    PersistentStateSnapshot current{};

    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        current.sequenceId = seq;
        current.epoch = ep;
        current.mappedGeneration = gen;
    }

    PersistentStateSnapshot snapshot() const noexcept {
        return current;
    }
};
```

**メリット**: atomic 操作ゼロ、ミューテックスゼロ、オーバーヘッドゼロ
**デメリット**: 将来の並行読取追加時に破綻する
**適用条件**: Phase-0 で「Single-Thread Ownership 維持」を決定した場合のみ

### 2.4 3方式の比較

| 観点 | 方式A: atomic<Snapshot> | 方式B: seqlock | 方式C: non-atomic |
|---|---|---|---|
| 複雑さ | 低 | 中 | **最小** |
| atomic 操作数 | 1 (store/load) | 4 (version×2 + fields×2) | **0** |
| 論理整合性保証 | commit 単位 | read 側で検証 | **暗黙（同一スレッド）** |
| 将来の並行読取耐性 | 低 | **高** | なし |
| MSVC mutex fallback | あり（性能影響なし） | なし | なし |
| 推奨シナリオ | 汎用 | 防御的設計 | **最軽量・現状最適** |

---

## 第3章: Recovery での snapshot() 失敗ハンドリング（確定）

```cpp
case convo::RecoveryAction::Restore: {
    const auto snapResult = persistentState_.snapshot();
    if (!snapResult.valid) {
        switch (snapResult.error) {
        case SnapshotError::WriterBusy:
            // commit() 実行中 → 次回 timer tick で再試行
            diagLog("[RECOVERY] PersistentStateBlock busy, retrying next tick");
            break;
        case SnapshotError::RetryExceeded:
            // writer が異常停止 → 緊急回復
            diagLog("[RECOVERY] PersistentStateBlock stalled, force recovery");
            if (observePublishedWorld() == nullptr) {
                break;  // world も空 → 安全
            }
            publishIdleWorldOnly(getActiveRuntimeDSP(),
                convo::TransitionPolicy::HardReset);
            break;
        }
        break;
    }
    // ... 通常の derive → reconcile → 修復 ...
    break;
}
```

---

## 第4章: Phase-2.5 監査機構（具体化）

### 4.1 週次監査ログ

```powershell
# .github/scripts/isr-audit-currentworld-refs.ps1
$logFile = "evidence/currentworld_audit.log"
$date = Get-Date -Format "yyyy-MM-dd"

# ISRRuntimePublicationCoordinator 内の参照数
$headerRefs = (Select-String -Path "src/audioengine/ISRRuntimePublicationCoordinator.h" -Pattern "currentWorld_").Count
$cppRefs = (Select-String -Path "src/audioengine/ISRRuntimePublicationCoordinator.cpp" -Pattern "currentWorld_").Count
$total = $headerRefs + $cppRefs

# 他ファイルの新規参照チェック
$otherRefs = @()
Get-ChildItem -Path "src" -Recurse -Include "*.h","*.cpp" | ForEach-Object {
    $path = $_.FullName
    if ($path -notmatch "ISRRuntimePublicationCoordinator") {
        $refs = (Select-String -Path $path -Pattern "currentWorld_").Count
        if ($refs -gt 0) { $otherRefs += "$path : $refs" }
    }
}

$logEntry = "$date coordinator.h=$headerRefs coordinator.cpp=$cppRefs total=$total"
if ($otherRefs) { $logEntry += " OTHER=" + ($otherRefs -join "; ") }

Add-Content -Path $logFile -Value $logEntry
Write-Host $logEntry
```

### 4.2 CI ゲート

```powershell
# isr-enforce-no-new-currentworld-ref.ps1
$violations = @()
Get-ChildItem -Path "src" -Recurse -Include "*.h","*.cpp" | ForEach-Object {
    $path = $_.FullName
    if ($path -match "ISRRuntimePublicationCoordinator|ISRSemanticValidationTests") { return }
    $content = Get-Content $path -Raw -Encoding UTF8
    if ($content -match "currentWorld_") { $violations += $path }
}
if ($violations) { Write-Error "currentWorld_ ref in unexpected file"; exit 1 }
```

---

## 第5章: 全 Invariant（6件確定）

| # | 名称 | CI ゲート |
|---|---|---|
| 001 | Authority State 再構築可能性 | `isr-verify-auth-001.ps1` |
| 002 | Recovery 状態同値性 | `isr-verify-auth-002.ps1` |
| 003 | Publish 経路唯一性 | `isr-verify-auth-003.ps1`（既存） |
| 004 | Pure Function（暫定 regex + 将来 AST） | `isr-verify-auth-004.ps1` |
| 005 | 唯一永続メタデータ源 | `isr-verify-auth-005.ps1` |
| 006 | RuntimeStore 整合性 | `isr-verify-auth-006.ps1` |

---

## 第6章: Phase 計画（確定版）

```
Phase-0: seqlock方式決定（方式A/B/Cから選択）
  └─ 実装着手前に1回のみ

Phase-1: 基盤導入（並行可能）
  1a: PersistentStateBlock（選択した方式で実装）
  1b: AuthorityDescriptor + Telemetry
  1c: Validator エッジケース（7 tests）

Phase-2: currentWorld_ 段階的削除（前編）
  2a: getCurrent → RuntimeStore 委譲
  2b: 全17テスト移行
  2c: getCurrent() の currentWorld_ フォールバック削除

Phase-2.5: currentWorld_ 監査（新規）
  監査ログ + CI ゲート

Phase-3: 状態導出 + Recovery
  3a: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
  3b: Recovery 統合

Phase-4: Invariant CI + currentWorld_ 完全削除
  4a-f: ISR-AUTH-001～006 CI ゲート
  4g: commit/retire 内 currentWorld_ 操作削除
  4h: メンバ削除（grep 0件確認）

Phase-5: テスト拡充
  5a: Property Test（10,000回混在）
  5b: 障害注入テスト（4シナリオ）
```
