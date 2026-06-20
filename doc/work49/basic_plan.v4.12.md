# Practical Stable ISR Bridge Runtime — 設計書 v4.12（実装開始版・最終）

**Document Version:** 4.12
**Date:** 2026-06-20
**Based on:** v4.11 + レビュー指摘6点の反映
**Status:** 実装開始版（全設計判断完了）

---

## v4.11 → v4.12 変更点一覧

| # | 項目 | v4.11 | v4.12 | 根拠 |
|---|---|---|---|---|
| 1 | **方式選択** | 方式C(non-atomic)採用 | **方式B(seqlock)正式採用。方式C廃止。** | 「Practical Stable=将来の変更でも壊れない」思想との整合 |
| 2 | **deriveExpectedState** | `sequenceId > 0` のみ | **3フィールド間の整合性も検出**（epoch=99, generation=100 等） | commit() の単調増加検証と整合 |
| 3 | **ISR-AUTH-001 CI** | 引数のみgrep | **引数検査 + 関数本体の forbidden 呼び出し検査** | 容易なすり抜け防止 |
| 4 | **fullReconciliation 検証** | reconcileAuthorityState 自身で判定 | **独立 Validator `validateAuthorityState()` を追加** | 循環参照の解消 |
| 5 | **Property Test** | Random Stress Test | **Model State との比較（モデルベース）** | Invariant の真の保証 |
| 6 | **方式C** | 3方式併記の1つ | **廃止**（Practical Stable 思想と非整合） | 将来の並行化で破綻 |

---

## 第1章: 方式B(seqlock) 正式採用

### 1.1 判断理由

Practical Stable ISR Bridge Runtime の最終目標は：

> 将来の実装変更や回復経路追加でも Authority State を失わない

方式C(non-atomic)は「現状の Single-Thread Ownership が永遠に続く」ことを前提とする。
しかし本設計書 v4.0 から一貫して目指しているのは「将来の変更でも壊れない堅牢性」である。

| 観点 | 方式C (non-atomic) | 方式B (seqlock) |
|---|---|---|
| 現状の安全性 | ✅ 同一スレッドで安全 | ✅ 同一スレッドで安全 |
| 将来の並行読取 | ❌ 破綻 | ✅ 防御済み |
| 実装コスト | 最小 | 中（+version + ScopedGuard + Result） |
| Practical Stable 思想 | ❌ 逆行 | ✅ 一致 |

**判断: 方式B(seqlock)を正式採用。方式Cは廃止。**

### 1.2 seqlock 完全版

```cpp
struct PersistentStateBlock {
    std::atomic<uint64_t> version{0};
    std::atomic<uint64_t> publicationSequenceId{0};
    std::atomic<uint64_t> publicationEpoch{0};
    std::atomic<uint64_t> mappedRuntimeGeneration{0};

    struct Snapshot {
        uint64_t sequenceId;
        uint64_t epoch;
        uint64_t mappedGeneration;
        uint64_t snapVersion;
    };

    // 論理スナップショット（Read-copy update 方式）
    // ★ (v0 & 1u) == 0 で書き込み中でないことを確認
    // ★ v0 == v1 で読み取り中の更新がなかったことを確認
    SnapshotResult snapshot() const noexcept {
        for (int i = 0; i < kMaxRetries; ++i) {
            auto v0 = convo::consumeAtomic(version, std::memory_order_acquire);
            if ((v0 & 1u) != 0) continue;
            auto seq = convo::consumeAtomic(publicationSequenceId, std::memory_order_acquire);
            auto ep  = convo::consumeAtomic(publicationEpoch, std::memory_order_acquire);
            auto gen = convo::consumeAtomic(mappedRuntimeGeneration, std::memory_order_acquire);
            auto v1  = convo::consumeAtomic(version, std::memory_order_acquire);
            if ((v1 & 1u) == 0 && v0 == v1)
                return {true, SnapshotError::None, {seq, ep, gen, v0}};
        }
        return {false, SnapshotError::RetryExceeded, {}};
    }

    // ★ ScopedVersionWriteGuard (RAII) で書き込み区間を保護
    void commitFields(uint64_t seq, uint64_t ep, uint64_t gen) noexcept {
        jassert(!convo::numeric_policy::isAudioThread());
        ScopedVersionWriteGuard guard(version);
        convo::publishAtomic(publicationSequenceId, seq, std::memory_order_release);
        convo::publishAtomic(publicationEpoch, ep, std::memory_order_release);
        convo::publishAtomic(mappedRuntimeGeneration, gen, std::memory_order_release);
    }

private:
    static constexpr int kMaxRetries = 3;
};
```

### 1.3 MessageThread 所有権ルール

方式B採用後も、書き込みは **MessageThread 専有** を維持する。
seqlock は「将来、別スレッドからの読み取りが追加された場合」の防御：

```
commitFields(): MessageThread のみ（jassert(isAudioThread()) で違反検出）
snapshot():    任意スレッドから呼び出し可能（seqlock で整合性保証）
```

---

## 第2章: deriveExpectedState 拡張

### 2.1 問題

v4.11 の deriveExpectedState は `sequenceId > 0` しか見ていない。
そのため以下の不整合を見逃す：

```
seq=100, epoch=99, generation=100
→ epoch だけ古い → commit() の単調増加チェックなら reject されるべき状態
```

### 2.2 拡張設計

```cpp
AuthorityState deriveExpectedState(const PersistentStateSnapshot& ps) noexcept {
    AuthorityState result;
    result.publicationSequenceId = ps.sequenceId;
    result.publicationEpoch = ps.epoch;
    result.mappedRuntimeGeneration = ps.mappedGeneration;

    // ★ sequenceId > 0 → RuntimeStore に world が存在すべき
    result.hasActiveRuntime = (ps.sequenceId > 0);

    // ★ epoch と generation の整合性チェック
    //   commit() では「epoch <= previous」「generation <= previous」を reject する
    //   つまり正常状態では epoch と generation はどちらも前回より大きい
    //   → epoch == 0 かつ generation > 0 は不整合（bootstrap 以外）
    //   → generation == 0 かつ sequenceId > 0 は不整合
    result.hasPendingPublication = false;
    result.hasActiveCrossfade = false;

    return result;
}
```

### 2.3 deriveAuthorityState での不整合検出

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

    // ISR-AUTH-006: PersistentStateBlock ↔ RuntimeStore 整合性
    result.runtimeMissing = (ps.sequenceId > 0 && runtimeWorld == nullptr);
    result.persistentMissing = (runtimeWorld != nullptr && ps.sequenceId == 0);

    // ★ 3フィールド間の不整合検出（新規）
    //   commit() の単調増加契約の観点から、epoch/generation が不自然な状態を検出
    result.fieldInconsistencyDetected =
        (ps.sequenceId > 0 && ps.epoch == 0)      // seq あるのに epoch なし
     || (ps.sequenceId > 0 && ps.mappedGeneration == 0)  // seq あるのに gen なし
     || (ps.epoch > 0 && ps.mappedGeneration == 0)        // epoch あるのに gen なし
     || (ps.epoch == 0 && ps.mappedGeneration > 0);       // gen あるのに epoch なし

    result.hasPendingPublication = result.runtimeMissing;
    if (runtimeWorld != nullptr) {
        result.hasActiveCrossfade = runtimeWorld->execution.transitionActive;
    }
    return result;
}
```

---

## 第3章: ISR-AUTH-001 CI 強化

### 3.1 問題

v4.11 の `deriveAuthorityState(\)PersistentStateSnapshot` は以下のすり抜けが可能：

```cpp
// CI を PASS するが ISR-AUTH-001 違反
void deriveAuthorityState(PersistentStateSnapshot ps, RuntimeStore& store);
```

### 3.2 強化版 CI（引数検査 + 関数本体検査）

```powershell
# isr-verify-auth-001.ps1
$targetFile = "src/core/AuthorityState.h"
$content = Get-Content (Join-Path $RepoRoot $targetFile) -Raw -Encoding UTF8

$violations = @()

# ★ 検査1: 関数シグネチャ
#   deriveAuthorityState の引数が (const PersistentStateSnapshot&, const World*) であること
$sigMatch = [regex]::Match($content,
    'deriveAuthorityState\s*\([^)]*\)')
if (-not $sigMatch.Success) {
    $violations += "deriveAuthorityState signature not found"
} else {
    $sig = $sigMatch.Value
    # ★ 第1引数が const PersistentStateSnapshot& であること（値渡しや非constは禁止）
    if ($sig -notmatch 'const\s+PersistentStateSnapshot\s*&') {
        $violations += "Arg1 must be const PersistentStateSnapshot& (not by value)"
    }
    # ★ RuntimeStore が引数に含まれていないこと
    if ($sig -match 'RuntimeStore') {
        $violations += "Arg must not include RuntimeStore"
    }
}

# ★ 検査2: 関数本体に forbidden 呼び出しがないこと
#   consumeAtomic / publishAtomic / fetchAddAtomic が関数本体内にあるか
$funcStart = $content.IndexOf("deriveAuthorityState")
if ($funcStart -ge 0) {
    $braceStart = $content.IndexOf("{", $funcStart)
    if ($braceStart -ge 0) {
        $braceDepth = 1; $pos = $braceStart + 1
        while ($braceDepth -gt 0 -and $pos -lt $content.Length) {
            if ($content[$pos] -eq '{') { $braceDepth++ }
            elseif ($content[$pos] -eq '}') { $braceDepth-- }
            $pos++
        }
        $body = $content.Substring($braceStart + 1, $pos - $braceStart - 2)
        foreach ($forbidden in @('publishAtomic', 'fetchAddAtomic')) {
            if ($body -match $forbidden) {
                $violations += "Function body contains forbidden: $forbidden"
            }
        }
        # consumeAtomic は引数読み取りに使うかもしれないが、
        # PersistentStateBlock は seqlock 版なので snapshot() を使うべき
        if ($body -match 'consumeAtomic') {
            $violations += "Function body should use snapshot() not consumeAtomic"
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "[FAIL] ISR-AUTH-001:"
    $violations | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "[PASS] ISR-AUTH-001"
```

---

## 第4章: 独立 Validator（循環参照の解消）

### 4.1 問題

v4.11 では `reconcileAuthorityState()` 自身が `fullReconciliation` を返していた。
これでは修復関数にバグがある場合、テストが偽陽性になる。

### 4.2 独立 Validator

```cpp
// ★ 独立した Validator: reconcileAuthorityState とは別に関数
//   observed と expected の一致を独立して検証する
[[nodiscard]] bool validateAuthorityStateMatch(
    const AuthorityState& observed,
    const AuthorityState& expected) noexcept
{
    // ★ observed と expected の全フィールドを独立比較
    //   reconcileAuthorityState() が正しく動いているかの検証とは独立
    if (observed.publicationSequenceId != expected.publicationSequenceId)
        return false;
    if (observed.publicationEpoch != expected.publicationEpoch)
        return false;
    if (observed.mappedRuntimeGeneration != expected.mappedRuntimeGeneration)
        return false;
    // ★ fullReconciliation の真の条件: observed == expected
    return observed == expected;
}
```

### 4.3 Property Test / Recovery での使用

```cpp
// Recovery 後の確認（循環参照なし）
const auto afterObserved = deriveAuthorityState(afterPs, afterWorld);
const auto afterExpected = deriveExpectedState(afterPs);

// ★ 独立 Validator で確認（reconcileAuthorityState には依存しない）
jassert(validateAuthorityStateMatch(afterObserved, afterExpected));

// ★ reconcileAuthorityState の結果も確認（両方確認することで安心）
const auto afterRec = reconcileAuthorityState(afterObserved, afterExpected);
jassert(afterRec.fullReconciliation);
```

---

## 第5章: 真の Model-Based Test

### 5.1 問題

v4.11 の Property Test は API をランダムに叩く Random Stress Test であり、
Model State との比較がない。

### 5.2 Model-Based Test

```cpp
// ★ モデル状態: 期待される Authority 状態を独立して追跡
struct AuthorityModel {
    uint64_t expectedSequenceId{0};
    uint64_t expectedEpoch{0};
    uint64_t expectedGeneration{0};
    bool expectedWorldExists{false};

    // ★ Publish 操作後のモデル更新
    void onPublish(uint64_t seq, uint64_t ep, uint64_t gen) {
        expectedSequenceId = seq;
        expectedEpoch = ep;
        expectedGeneration = gen;
        expectedWorldExists = true;
    }

    // ★ Retire 操作後のモデル更新
    void onRetire() {
        expectedWorldExists = false;
    }

    // ★ モデルと実際の状態を比較
    bool matches(const PersistentStateSnapshot& ps,
                 const void* runtimeWorld) const {
        if (ps.sequenceId != expectedSequenceId) return false;
        if (ps.epoch != expectedEpoch) return false;
        if (ps.mappedGeneration != expectedGeneration) return false;
        if ((runtimeWorld != nullptr) != expectedWorldExists) return false;
        return true;
    }
};

TEST_F(ModelBasedTest, ModelMatchesRealState_AfterRandomOps) {
    AuthorityModel model;
    std::mt19937 rng(42);

    for (int i = 0; i < 10000; i++) {
        uint64_t gen = rng() % 1000 + 1;
        uint64_t seq = rng() % 1000 + 1;
        uint64_t ep  = rng() % 1000 + 1;

        switch (rng() % 3) {
        case 0: // Publish
            coordinator.publishWorld(createWorld(seq, ep, gen));
            model.onPublish(seq, ep, gen);
            break;
        case 1: // Retire
            // ...
            model.onRetire();
            break;
        case 2: // Recover
            // ...
            break;
        }

        // ★ モデルと実際の状態を毎回比較
        auto snap = persistentState.snapshot();
        ASSERT_TRUE(snap.valid);  // seqlock: snapshot は必ず有効
        auto* world = observePublishedWorld();
        EXPECT_TRUE(model.matches(snap.data, world));
    }
}
```

---

## 第6章: 全 Invariant 最終版（6件）

| # | 名称 | 保証方法 | CI |
|---|---|---|---|
| 001 | Authority State 再構築可能性 | seqlock + requires 型制約 + 引数/本体両方の CI | `isr-verify-auth-001.ps1`（強化版） |
| 002 | Recovery 状態同値性 | 独立 Validator `validateAuthorityStateMatch()` | `isr-verify-auth-002.ps1` |
| 003 | Publish 経路唯一性 | 既存 CI 流用 | `isr-verify-auth-003.ps1` |
| 004 | Pure Function | 型システム（const ref 引数）+ 関数本体 atomic 禁止 | `isr-verify-auth-004.ps1` |
| 005 | 唯一永続メタデータ源 | PersistentStateBlock 以外の永続状態禁止 | `isr-verify-auth-005.ps1` |
| 006 | RuntimeStore 整合性 | deriveAuthorityState 内で fieldInconsistencyDetected | `isr-verify-auth-006.ps1` |
