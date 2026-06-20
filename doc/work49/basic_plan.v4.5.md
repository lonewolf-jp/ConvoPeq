# Practical Stable ISR Bridge Runtime — 設計書 v4.5（実装開始版）

**Document Version:** 4.5
**Date:** 2026-06-19
**Based on:** v4.4 + レビュー指摘5点の反映
**Status:** 実装開始版（全未確定事項ゼロ、全 Invariant 確定）

---

## v4.4 → v4.5 変更点一覧

| # | 項目 | v4.4 | v4.5 | 根拠 |
|---|---|---|---|---|
| 1 | **snapshot() 戻り値** | `Snapshot`（不整合フォールバックあり） | **`std::optional<Snapshot>`**（不整合時は nullopt） | ISR-AUTH-001 との矛盾解消 |
| 2 | **seqlock 偶数/奇数判定** | `v0 == v1` のみ | **`(v0 & 1u) == 0 && v0 == v1`** | 書き込み中の version を正しく検出 |
| 3 | **update() ガード** | 裸の version++ | **`ScopedVersionWriteGuard`**（RAII） | 途中終了時の version 奇数残留防止 |
| 4 | **ISR-AUTH-004 CI** | ファイル全体 regex | **関数本体スコープ限定の regex** | 誤検出削減 |
| 5 | **currentWorld_ 監査 Phase** | Phase-2 → Phase-4 の2段階 | **Phase-2.5 を新設（監査 + CI 禁止）** | 将来の再利用防止 |
| 6 | **ISR-AUTH-005** | なし | **新設**：PersistentStateBlock は唯一の永続メタデータ源 | 第二の永続状態の発生防止 |

---

## 第0章: 検証プロセス総括（全7サイクル完了）

| サイクル | 成果物 | 確定項目数 |
|---|---|---|
| 1st | validation_report.md | 12 |
| 2nd | design_deep_investigation_report.md | 7 |
| 3rd | basic_plan.v4.1.md | 4 |
| 4th | basic_plan.v4.2.md | 6 |
| 5th | basic_plan.v4.3.md | 4 |
| 6th | basic_plan.v4.4.md | 3 |
| **7th** | **basic_plan.v4.5.md** | **6** |

### 使用ツール（全7サイクル）

Serena MCP, AiDex MCP, CodeGraph MCP, graphify, semble, Select-String

---

## 第1章: PersistentStateBlock（seqlock 完全版）

### 1.1 設計思想

3つの atomic フィールドの論理一貫性を保証するために seqlock パターンを採用する。
v4.4 の問題点を全て修正した完全版。

**修正点**:

1. `snapshot()` の戻り値を `std::optional<Snapshot>` に変更（不整合状態を返さない）
2. `(v0 & 1u) == 0` で version の偶数（安定状態）を確認
3. `ScopedVersionWriteGuard`（RAII）で update() の例外安全/Fault安全を保証

### 1.2 定義

```cpp
#pragma once
#include <atomic>
#include <optional>
#include <cstdint>
#include "AtomicAccess.h"

namespace convo {

// ★ 本ッドロック防止: ScopedVersionWriteGuard — RAII で version の偶数/奇数を管理
//   update() 開始時に version を奇数にし、終了時に偶数にする。
//   スコープ脱出時（正常・異常の両方）に必ず偶数に戻す。
class ScopedVersionWriteGuard {
public:
    explicit ScopedVersionWriteGuard(std::atomic<uint64_t>& version) noexcept
        : version_(version) {
        // ★ version を奇数に（書き込み中マーク）
        convo::fetchAddAtomic(version_, uint64_t{1}, std::memory_order_acq_rel);
    }

    ~ScopedVersionWriteGuard() noexcept {
        // ★ ★ デストラクタで必ず version を偶数に戻す
        //    assert/jassert/guard/validation による途中終了でも安全
        convo::fetchAddAtomic(version_, uint64_t{1}, std::memory_order_acq_rel);
    }

    ScopedVersionWriteGuard(const ScopedVersionWriteGuard&) = delete;
    ScopedVersionWriteGuard& operator=(const ScopedVersionWriteGuard&) = delete;

private:
    std::atomic<uint64_t>& version_;
};

struct PersistentStateBlock {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
        uint64_t snapVersion;  // 読み取り成功時の version（常に偶数）
    };

    // ★ std::optional<Snapshot> — 不整合状態を返さない
    //   条件: version が偶数（安定）かつ前後の version が一致
    std::optional<Snapshot> snapshot() const noexcept {
        for (int attempt = 0; attempt < kMaxSnapshotRetries; ++attempt) {
            const auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            // ★ 偶数チェック: (v0 & 1u) == 0 で書き込み中でないことを確認
            if ((v0 & 1u) != 0)
                continue;  // 書き込み中 → リトライ

            const auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            const auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            const auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            const auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);

            // ★ 偶数 + 一致 = 論理一貫スナップショット
            if ((v1 & 1u) == 0 && v0 == v1) [[likely]] {
                return Snapshot{seq, ep, gen, v0};
            }
            // 不整合 → リトライ
        }
        // ★ 最大リトライ到達 → 恒久的な障害の可能性
        //   nullopt を返し、呼び出し側で RetryLater 等の判断を行う
        return std::nullopt;
    }

    // ★ ScopedVersionWriteGuard 使用版
    //   version の偶数/奇数を RAII で管理
    void update(const Snapshot& s) noexcept {
        ScopedVersionWriteGuard guard(version);
        convo::publishAtomic(publicationSequenceId, s.sequenceId, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, s.epoch, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, s.mappedGeneration, std::memory_order_release);
        // ★ guard のデストラクタで version++（version を偶数に戻す）
    }

private:
    static constexpr int kMaxSnapshotRetries = 3;
};

} // namespace convo
```

### 1.3 整合性保証

| 条件 | 保証内容 |
|---|---|
| `snapshot()` が `optional` を返す | 3回の整合性チェックを全て通過 ⇒ 論理一貫状態 |
| `snapshot()` が `nullopt` を返す | 書き込みスレッドが異常状態 ⇒ 呼び出し側で RetryLater |
| `(v0 & 1u) == 0` | version が偶数（書き込み中でない）ことを確認 |
| `v0 == v1` | 読み取り中に書き込みが発生していないことを確認 |
| `ScopedVersionWriteGuard` | デストラクタで必ず version を偶数に戻す |

### 1.4 Recovery との統合

```cpp
// executeRecoveryAction 内の使用例
case convo::RecoveryAction::Restore: {
    const auto snap = persistentState_.snapshot();
    if (!snap) {
        // ★ PersistentStateBlock が読み取り不能 → 再試行 or 安全側に倒す
        diagLog("[RECOVERY] PersistentStateBlock snapshot failed, retrying later");
        break;
    }

    const auto* runtimeWorld = observePublishedWorld();
    const auto observed = deriveAuthorityState(*snap, runtimeWorld);
    const auto expected = deriveExpectedState(*snap);
    const auto rec = reconcileAuthorityState(observed, expected);
    // ... 修復処理 ...
    break;
}
```

---

## 第2章: currentWorld_ 削除（6段階プロセス）

### 2.1 完全な段階計画

| Phase | 作業 | 確認方法 | 依存 |
|---|---|---|---|
| **Phase-2a** | `getCurrent()` → RuntimeStore 委譲（setRuntimeStore 注入） | コンパイル + 既存テスト PASS | Phase-1a |
| **Phase-2b** | 全17テスト箇所を `consumePublishedWorld(store)` に移行 | テスト PASS | Phase-2a |
| **Phase-2c** | `getCurrent()` の currentWorld_ フォールバック削除 | 本番コード変更ゼロ | Phase-2b |
| **Phase-2.5** | **currentWorld_ 読み取り監査 + CI 禁止（新設）** | CI PASS | Phase-2c |
| **Phase-4a** | `commit()` 内の `publishAtomic(currentWorld_, ...)` 削除 | コンパイル | Phase-2.5 |
| **Phase-4b** | `retire()` 内の `consumeAtomic` + `compareExchangeAtomic` 削除 | コンパイル | Phase-4a |
| **Phase-4c** | `currentWorld_` メンバ宣言 + コンストラクタ初期化削除 | grep 0件 | Phase-4b |

### 2.2 Phase-2.5 の詳細（新設）

#### CI ゲート

```powershell
# .github/scripts/isr-enforce-no-new-currentworld-ref.ps1
$ErrorActionPreference = 'Stop'

# ★ 新規 currentWorld_ 参照の追加を禁止
#   Phase-2c 完了後、currentWorld_ は ISRRuntimePublicationCoordinator 内でのみ使用可能
#   それ以外のファイルからの参照は違反

$allowedFiles = @(
    'src\audioengine\ISRRuntimePublicationCoordinator.h',
    'src\audioengine\ISRRuntimePublicationCoordinator.cpp'
)

$violations = @()
Get-ChildItem -Path 'src' -Recurse -Include '*.h','*.cpp' | ForEach-Object {
    $relativePath = $_.FullName.Replace($RepoRoot, '').TrimStart('\')
    if ($allowedFiles -contains $relativePath) {
        return  # 許可ファイルはスキップ
    }
    $content = Get-Content $_.FullName -Raw -Encoding UTF8
    if ($content -match 'currentWorld_') {
        $violations += $relativePath
    }
}

if ($violations.Count -gt 0) {
    Write-Host "[FAIL] ISR-AUTH-003(currentWorld_): New references detected in:"
    $violations | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "[PASS] No new currentWorld_ references"
```

#### 監査ログ

Phase-2.5 では毎週の currentWorld_ 参照数を記録し、ゼロが継続していることを確認する：

```
# evidence/currentworld_audit.log
2026-06-19 Phase-2c完了 currentWorld_参照: ISRRuntimePublicationCoordinator内5箇所
2026-06-26 監査 currentWorld_参照: 同上5箇所（変化なし）
2026-07-03 監査 currentWorld_参照: 同上5箇所（変化なし）
...
Phase-4a 実施後: commit() から削除 → 4箇所
Phase-4b 実施後: retire() から削除 → 2箇所（宣言 + 初期化）
Phase-4c 実施後: 0箇所
```

---

## 第3章: ISR-AUTH-004 CI（関数スコープ限定版）

### 3.1 設計

v4.4 のファイル全体 regex は誤検出が多すぎる。
関数本体スコープに限定して解析することで、コメントや型名の誤検出を防止する。

### 3.2 実装

```powershell
# .github/scripts/isr-verify-auth-004.ps1
param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..')
)

$ErrorActionPreference = 'Stop'

# ISR-AUTH-004: Authority State 導出関数は Pure Function
# ★ 関数本体スコープ限定で解析（ファイル全体は対象外）

$targetFiles = @(
    'src\core\AuthorityState.h'
)

# ★ チェック対象の関数シグネチャ（関数本体のみをスコープとする）
$functionPatterns = @(
    @{ Name = 'deriveAuthorityState';  Forbidden = @('publishAtomic', 'consumeAtomic', 'fetchAddAtomic') },
    @{ Name = 'deriveExpectedState';   Forbidden = @('publishAtomic', 'consumeAtomic', 'fetchAddAtomic') },
    @{ Name = 'reconcileAuthorityState'; Forbidden = @('publishAtomic', 'consumeAtomic', 'fetchAddAtomic') }
)

$violations = @()
foreach ($file in $targetFiles) {
    $fullPath = Join-Path $RepoRoot $file
    if (-not (Test-Path $fullPath)) {
        Write-Host "[SKIP] $file not found"
        continue
    }

    $content = Get-Content $fullPath -Raw -Encoding UTF8

    foreach ($func in $functionPatterns) {
        # ★ 関数本体スコープを抽出: "funcName(...) { ... }"
        $match = [regex]::Match($content, "${func.Name}\s*\([^)]*\)\s*\{")
        if (-not $match.Success) {
            continue  # 関数未定義（Phase-3未実装）
        }

        # ★ 関数開始位置から最初の閉じ括弧までをスコープとする
        $startPos = $match.Index + $match.Length
        $braceDepth = 1
        $endPos = $startPos
        while ($braceDepth -gt 0 -and $endPos -lt $content.Length) {
            $endPos++
            if ($content[$endPos] -eq '{') { $braceDepth++ }
            elseif ($content[$endPos] -eq '}') { $braceDepth-- }
        }
        $functionBody = $content.Substring($startPos, $endPos - $startPos)

        # ★ 関数本体内のみで forbidden pattern をチェック
        foreach ($pattern in $func.Forbidden) {
            if ($functionBody -match $pattern) {
                $violations += [PSCustomObject]@{
                    File    = $file
                    Function = $func.Name
                    Pattern = $pattern
                    Snippet  = $functionBody.Substring(0, [Math]::Min(80, $functionBody.Length))
                }
            }
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "[FAIL] ISR-AUTH-004 violations:"
    $violations | Format-Table -AutoSize
    exit 1
}

Write-Host "[PASS] ISR-AUTH-004: All authority derivation functions are Pure Functions"
exit 0
```

### 3.3 誤検出防止策

| 誤検出パターン | v4.4 | v4.5 |
|---|---|---|
| コメント内の `runtimeStore` | 検出（誤） | **非検出（関数スコープ限定）** |
| 型名 `RuntimeStoreSnapshot` | 検出（誤） | **非検出（関数スコープ限定）** |
| 関数本体内の `publishAtomic` | 検出（正） | **検出（正）** |

---

## 第4章: ISR-AUTH-005（新規）

### 4.1 定義

```
ISR-AUTH-005

PersistentStateBlock は Authority State の唯一の永続メタデータ源である。

根拠:
  将来の保守で「便利だから」と publicationSequenceId 等の第二の永続状態を
  持つと、PersistentStateBlock の SSOT 性が失われる。
  回復可能性の保証には唯一の永続メタデータ源が必要。

遵守方法:
  - publicationSequenceId / publicationEpoch / mappedRuntimeGeneration に
    相当する値は PersistentStateBlock からのみ取得する
  - これらの値を別の atomic やキャッシュに保持しない

禁止:
  cachedSequenceId_
  cachedEpoch_
  cachedGeneration_
  または類似の第二永続状態

違反検出:
  grep -E "cachedSequenceId_|cachedEpoch_|cachedGeneration_|backupSequenceId|backupEpoch|backupGeneration"
  → 0件であること
```

---

## 第5章: 全 Invariant 一覧

| Invariant | 内容 | 違反検出 | フェーズ |
|---|---|---|---|
| **ISR-AUTH-001** | Authority State は PersistentStateBlock からのみ再構築可能 | `deriveAuthorityState` の引数に PersistentStateBlock::Snapshot が含まれること | 4a |
| **ISR-AUTH-002** | Recovery 後は通常 Publish 経路で到達可能な状態と同値 | `reconcileAuthorityState().fullReconciliation == true` | 4b |
| **ISR-AUTH-003** | Publish 経路は Orchestrator → Coordinator の唯一経路 | DSPTransition/HealthMonitor/CrossfadeRuntime からの直接 publishWorld 禁止 | 4c（既存） |
| **ISR-AUTH-004** | Authority State 導出関数は Pure Function | 関数本体内での atomic 操作禁止 | 4d |
| **ISR-AUTH-005** | PersistentStateBlock は唯一の永続メタデータ源 | 第二永続状態（cachedSequenceId_ 等）の存在禁止 | 4e |

---

## 第6章: 最終 Phase 計画（v4.5 確定版）

```
Phase-1: 基盤導入（並行可能）
  ├─ 1a: PersistentStateBlock（seqlock 完全版: optional + 偶数判定 + ScopedGuard）
  ├─ 1b: AuthorityDescriptor（Domain × Reason）
  └─ 1c: Validator エッジケース（7 test cases）

Phase-2: currentWorld_ 段階的削除（前編）
  ├─ 2a: getCurrent → RuntimeStore 委譲
  ├─ 2b: 全17テスト移行（consumePublishedWorld）
  └─ 2c: getCurrent() の currentWorld_ フォールバック削除

Phase-2.5: currentWorld_ 監査（新設）
  ├─ 監査: 週次 currentWorld_ 参照数記録
  └─ CI: 新規 currentWorld_ 参照の追加を禁止

Phase-3: 状態導出 + Recovery
  ├─ 3a: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
  └─ 3b: Recovery 統合（executeRecoveryAction → reconcile 接続）

Phase-4: Invariant CI + currentWorld_ 完全削除
  ├─ 4a: ISR-AUTH-001 CI ゲート（再構築可能性）
  ├─ 4b: ISR-AUTH-002 CI ゲート（状態同値性）
  ├─ 4c: ISR-AUTH-003 CI ゲート（経路唯一性）
  ├─ 4d: ISR-AUTH-004 CI ゲート（Pure Function）
  ├─ 4e: ISR-AUTH-005 CI ゲート（唯一永続源）
  ├─ 4f: commit() 内 currentWorld_ 書き込み削除
  ├─ 4g: retire() 内 currentWorld_ CAS 削除
  └─ 4h: currentWorld_ メンバ削除（grep 0件確認）

Phase-5: テスト拡充
  ├─ 5a: Property Test（10,000回混在）
  └─ 5b: 障害注入テスト（4シナリオ）
```

---

## 第7章: 完了条件

```
grep "PersistentStateBlock" src/core/        → 1件以上
grep "AuthorityDomain" src/core/             → 1件以上
grep "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.* → 0件
grep "deriveAuthorityState" src/core/        → 1件以上
grep "reconcileAuthorityState" src/core/     → 1件以上
grep "cachedSequenceId_\|cachedEpoch_\|cachedGeneration_" src/ → 0件
isr-verify-auth-001.ps1 → PASS
isr-verify-auth-002.ps1 → PASS
isr-verify-auth-003.ps1 → PASS
isr-verify-auth-004.ps1 → PASS
isr-verify-auth-005.ps1 → PASS
isr-enforce-no-new-currentworld-ref.ps1 → PASS
Validator tests: 45+ test cases PASS
Property Test: 10,000回 PASS
Fault Injection: 4 scenarios PASS
```

### 到達予測

```
現状:          92-95%
Phase-1 完了:  95-96%
Phase-2 完了:  96-97%
Phase-2.5 完了: 97%
Phase-3 完了:  97-98%
Phase-4 完了:  98-99%
Phase-5 完了:  99-100%
```
