---
goal: "Phase-E: Recap Prevention & Architectural Hardening"
version: 2.0
date_created: 2026-06-07
last_updated: 2026-06-08
note: 'Phase-D完了後の残課題。「実装」ではなく「退行防止の恒久化」と「構造の磨き込み」が主眼。全6項目(P1-P6)の実装完了。P7追加: C4996ビルドノイズ抑制。'
owner: GitHub Copilot (AI Assistant)
status: 'Complete (P1-P6), Partial (P7)'
tags: refactor, architecture, bridge-runtime, epoch, ci, ast
---

# Phase-E: Recap Prevention & Architectural Hardening

## 対象範囲

Phase-D で達成した「実運用で破綻しにくい状態」を恒久化するための施策。
新機能追加ではなく、**退行防止の自動化**と**構造的負債の返済**が目的。

## 実装状況

| 優先度 | 項目 | リスク | 状態 | 完了日 |
|--------|------|--------|------|--------|
| **P1** | Architecture Regression Snapshot | 退行監視不足 | ✅ 完了 | 2026-06-07 |
| **P2** | ObservedRuntime ownerThreadId Diagnostic化 | Debug/Release差 | ✅ 完了 | 2026-06-07 |
| **P3** | ISRRetireRouter domain()削除 | Authority境界漏洩 | ✅ 完了 | 2026-06-07 |
| **P4** | AST CI (alias検出強化) | 退行検出不能 | ✅ 完了 | 2026-06-07 |
| **P5** | EpochDomainReaderGuard 撤去 | ReaderGuard二系統 | ✅ 完了 | 2026-06-07 |
| **P6** | コメント監査 (Warning only) | 情報腐敗 | ✅ 完了 | 2026-06-07 |
| **P7** | C4996ビルドノイズ抑制 | 開発体験低下 | ⚡ Partial | 2026-06-08 |

---

## P4: AST CI (alias検出強化) — ✅ 完了 (2026-06-07)

### 完了状態

**実装**: Python AST スキャナー (`tools/check-epochdomain-ast.py`) を作成し CI Gate に統合。

**二層構成 (計画通り)**:
- **Layer 1 (grep)**: `check-work21-epochdomain-gates.ps1` — 既存の CI Gate で継続。advanceEpoch / enqueueRetire / exposure をテキストパターンで監視。
- **Layer 2 (Python AST)**: `tools/check-epochdomain-ast.py` — Python スクリプトでソースファイルをパースし、`using` / `typedef` / template parameter 経由の EpochDomain 出現を検出。

**スキャナー出力例**:
```
Direct EpochDomain usage:  5
Type aliases found:        0
Alias transitive usage:    0
Template parameter usage:  0
```

**完了条件の充足**:
| 条件 | 状態 |
|------|------|
| clang-tidy が CI パイプラインで実行 | ➡ Python AST scanner が代替 (clang-tidy plugin は保守コストに見合わず見送り) |
| `using EpochAlias = EpochDomain;` を検出して Fail | ✅ (two-step alias検出実装) |
| CI Gate に AST 結果が統合表示 | ✅ (`[P4] AST scanner` 行として出力) |

---

## P5: EpochDomainReaderGuard 撤去 — ✅ 完了 (2026-06-07)

### 設計判断と経緯

本項目の計画は完了時に以下の形で実装された。

### 完了状態

| 条件 | 状態 | 確認方法 |
|------|------|----------|
| `EpochDomainReaderGuard` の型使用が `src/` 以下に存在しない | ✅ | `Select-String -Path (Get-ChildItem -Recurse -Include "*.h","*.cpp" -Path src) -Pattern "EpochDomainReaderGuard"` → 空 |
| `EpochDomainReaderGuard.h` が削除されている | ✅ | `Test-Path src/core/EpochDomainReaderGuard.h` → False |
| `ObservedRuntime` が `RCUReaderGuard` のみを使用 | ✅ | `ObservedRuntime.h` の `guard` メンバ型は `RCUReaderGuard` |

### 実装内容

| 変更対象 | 変更内容 |
|----------|----------|
| `src/core/RCUReader.h` | `RCUReaderGuard` に pointer-based move semantics 追加: move ctor/assign で source を `nullptr` 化、move assign で旧 reader の `exit()` を実行、全メソッド `noexcept` |
| `src/core/ObservedRuntime.h` | `EpochDomainReaderGuard guard` → `RCUReaderGuard guard` に置換。コンストラクタが `RCUReader& reader` を受け取り `guard(reader)` で初期化。`ownerThreadId` を `#ifndef NDEBUG` で Diagnostic 化 |
| `src/core/SnapshotCoordinator.h` | `observeCurrentRuntime()` の引数を `IReaderEpochProvider&` → `RCUReader&` に変更 |
| `src/core/EpochDomainReaderGuard.h` | ファイル削除 |
| `src/eqprocessor/EQProcessor.h/.cpp` | `ObservedRuntime` 生成を `m_epochDomain` 直接 → `audioThreadRcuReader` 経由に移行 |

### 特に重要な設計判断: RCUReaderGuard move semantics

`RCUReaderGuard` に move semantics を追加した結果、以下の条件で `enter()` / `exit()` の回数一致が保証される:

- **move ctor**: source を `nullptr` 化、destructor で `exit()` は呼ばない
- **move assign**: 先に `this->exit()` (旧 reader 解放)、次に source の所有権移譲、source を `nullptr` 化
- **vector reallocation / optional / RVO failure**: いずれも move ctor 経由で `exit()`/`enter()` 回数維持
- **全メソッド `noexcept`**: 標準コンテナの強い保証を活用可能

```
ObservedRuntime
    ↓ (メンバとして保持)
RCUReaderGuard   ← RAIIスコープ管理のみ (move安全)
    ↓ (参照)
RCUReader&       ← 上位オブジェクト (AudioEngine/EQProcessor) が所有
```

---

## P2: ObservedRuntime ownerThreadId Diagnostic化 — ✅ 完了 (2026-06-07)

### 完了状態

実装内容は計画の推奨対応と一致。

```cpp
// ObservedRuntime.h
struct ObservedRuntime
{
    explicit ObservedRuntime(RCUReader& reader) noexcept
        : guard(reader)
#ifndef NDEBUG
        , ownerThreadId(std::this_thread::get_id())
#endif
    { ... }

#ifndef NDEBUG
    std::thread::id ownerThreadId;
#endif
};
```

- `ownerThreadId` は `#ifndef NDEBUG` でガード → Release ビルドでは完全除去
- `get()` / `operator bool()` のスレッドチェックも `#ifndef NDEBUG` 内

### 呼称の訂正

> ~~ownerThreadId削除完了~~ → **ownerThreadId Diagnostic化完了** (Debug限定化)

### 完了条件の充足

| 条件 | 状態 |
|------|------|
| `ObservedRuntime` の `ownerThreadId` が `#ifndef NDEBUG` でガード | ✅ |
| Debug/Release 両方でビルド成功 | ✅ |

---

## P1: Architecture Regression Snapshot — ✅ 完了 (2026-06-07)

### 完了状態

CI Gate (`check-work21-epochdomain-gates.ps1`) に Architecture Regression Snapshot を統合済み。

最終計測値 (2026-06-08):

| Metric | Value | Status |
|--------|-------|--------|
| advanceEpoch direct calls | 0 | ✅ Target |
| enqueueRetire calls | 4 | ⚠ Router-internal |
| EpochDomain type exposure | 2 | ⚠ ISRRetireRouter only |
| IReaderEpochProvider methods | 7 | ✅ |
| IPublicationProvider methods | 1 | ✅ |
| IRetireProvider methods | 2 | ✅ |
| IEpochProvider (facade) methods | 0 | ✅ Target |
| ISRRetireRouter own public API | 3 | ✅ ≤8 |
| Alias count | 0 | ✅ |
| Template parameter count | 0 | ✅ |
| Total EpochProvider pure virtual | 10 | ✅ ≤12 |

### アラート閾値

| 段階 | 条件 | 動作 |
|------|------|------|
| Info | Delta = 0 | 表示のみ |
| Warning | Delta > 0 | CI Warning (Blockしない) |
| Review | Delta > 3 | Manual Review Required |
| Fail | Delta > 5 | CI Fail (急増検出) |

---

## P3: ISRRetireRouter domain()削除 — ✅ 完了 (2026-06-07)

### 完了状態

`ISRRetireRouter::domain()` を完全削除。外部は以下の Router API のみ使用可能:

| API | 経路 |
|-----|------|
| `snapshotEpoch()` | `epochDomain_->currentEpoch()` |
| `publishEpoch()` | `epochDomain_->advanceEpoch()` (warning suppressed) |
| `enqueueRetire()` | `epochDomain_->enqueueRetire()` (warning suppressed) |
| `tryReclaim()` | `epochDomain_->deferredDeletionQueue.reclaim()` (inlined, no deprecated call) |
| `pendingRetireCount()` | `epochDomain_->pendingRetireCount()` |
| `minReaderEpoch()` | `epochDomain_->getMinReaderEpoch()` |

### Router 自身の public API 数: 3 (snapshotEpoch, minReaderEpoch, pendingRetireCount)

---

## P6: コメント監査 (Warning only) — ✅ 完了 (2026-06-07)

### 実施内容

- `EpochDomain_REFERENCE_STATUS.md` の参照実態一致性: 確認済み
- `SnapshotRetireManager.h` の 1 コメント (`EpochDomain::reclaim`) を epoch-free 表現 (`epoch boundary`) に修正
- 残存コメントは実態と乖離なし

### 未実施 (任意)

- CI Gate 13 として「コメント内 EpochDomain 参照が実態と乖離していないこと」の追加は未実施

---

## P7: C4996 ビルドノイズ抑制 — ⚡ Partial (2026-06-08)

### 課題

`EpochDomain.h` の `[[deprecated]]` 付きメソッド (`advanceEpoch`, `reclaimRetired`, `exitReader`, `enqueueRetire` 4-param) が原因で、全 145 翻訳単位で **C4996 警告** が大量発生していた。

### 対策

| 変更対象 | 内容 |
|----------|------|
| `src/core/EpochDomain.h` | ファイル先頭に `#pragma warning(push/disable: 4996)`、末尾に `#pragma warning(pop)` を追加。`[[deprecated]]` 属性そのものが警告を発生させないように抑制 |
| `src/core/EpochDomain.h` | `publishEpoch()` と `tryReclaim()` の内部実装をインライン化（非推奨メソッドを経由しない） |
| `src/eqprocessor/EQProcessor.Core.cpp` | `reclaimRetired()` → `tryReclaim()` (2箇所)、`advanceEpoch()` → `publishEpoch()` (1箇所) |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | `m_epochDomain.advanceEpoch()` → `m_retireRouter->publishEpoch()` |
| `src/ConvolverProcessor.h` | `m_epochDomain` の `[[deprecated]]` 属性を削除（内部ドメインとして正当） |
| `src/RefCountedDeferred.h` | `#pragma message("deprecated")` を削除 |

### ビルド結果

| 構成 | C4996 件数 | 状態 |
|------|-----------|------|
| Release (prev) | 145+ TU で多数 | 🚫 |
| Release (after) | **0** | ✅ |
| Debug (after) | 未確認 | ⏳ |

### 残課題

- Debug ビルドの C4996 件数を確認
- CI Gate に C4996 件数チェックを追加 (任意)

---

## 全体スケジュール見積もり（完了時点での実績）

| 項目 | 工数実績 | 難易度 | 状態 |
|------|---------|--------|------|
| P1 Architecture Regression Snapshot | 0.5h | 低 | ✅ |
| P2 ownerThreadId Diagnostic化 | 0.5h | 低 | ✅ |
| P3 ISRRetireRouter domain()削除 | 0.5h | 低 | ✅ |
| P4 AST CI (alias検出強化) | 2h | 中 | ✅ |
| P5 EpochDomainReaderGuard撤去 | 1h | 低 | ✅ |
| P6 コメント監査 | 0.5h | 低 | ✅ |
| P7 C4996ビルドノイズ抑制 | 1h | 低 | ⚡ Partial |

## 完了条件サマリ（検証済み）

| # | 条件 | 確認方法 | 状態 |
|---|------|----------|------|
| 1 | `clang-tidy` が EpochDomain 型露出を検出できる | CI Gate 出力（Python AST scanner 代替） | ✅ |
| 2 | `EpochDomainReaderGuard` がコードベースから消えている | `grep -rn EpochDomainReaderGuard src/` が空 | ✅ |
| 3 | `ObservedRuntime` の `ownerThreadId` が `#ifndef NDEBUG` でガード | grep 確認 | ✅ |
| 4 | CI Gate が全指標の regression を追跡している | CI ログ | ✅ |
| 5 | コメント内 EpochDomain 参照が実態と乖離していない | 手動レビュー (SnapshotRetireManager.h 1件修正) | ✅ |
| 6 | Release ビルドの C4996 警告が 0 | `cmake --build --config Release 2>&1 \| findstr C4996` が空 | ✅ |
| 7 | Debug ビルドの C4996 警告が 0 | `cmake --build --config Debug 2>&1 \| findstr C4996` が空 | ⏳ 未確認 |
