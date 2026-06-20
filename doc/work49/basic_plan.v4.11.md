# Practical Stable ISR Bridge Runtime — 設計書 v4.11（実装開始版・最終決定）

**Document Version:** 4.11
**Date:** 2026-06-20
**Based on:** v4.10 + 深堀5項目の確定
**Status:** 実装開始版（全未確定要素・全技術的懸念事項ゼロ）

---

## v4.10 → v4.11 変更点一覧

| # | 項目 | v4.10 | v4.11 | 根拠 |
|---|---|---|---|---|
| 1 | **commit() の (void) version** | 未調査 | **単なる未使用パラメータ抑制。PersistentStateBlock 導入時に削除可能** | sembl 調査で確認 |
| 2 | **getVersion() の影響** | 未調査 | **本番コードで未使用（テストのみ）**。移行後も互換性問題なし | 全ファイル grep 確認 |
| 3 | **isFullyDrained 等の互換性** | 未調査 | **backlog カウンタのみ参照。PersistentStateBlock 非依存。影響なし** | コード確認 |
| 4 | **2種類の Coordinator** | 概念のみ | **AudioEngine.h が両方を include し、typedef で template版を参照**。移行パス確定 | コード確認 |
| 5 | **semble/cocoindex** | 未使用 | **semble 正常動作確認。cocoindex は未インストール** | 実機確認 |

---

## 第0章: 未使用ツールの確認結果

### cocoindex code: 未インストール

```
cocoindex : コマンド名またはパラメータが間違っています。
```

プロジェクトの Python 仮想環境にインストールされていない。インストール手順は以下：

```powershell
pip install cocoindex-code
# または
uv tool install cocoindex-code
```

### semble: 正常動作確認

```powershell
$env:PYTHONUTF8="1"; semble search "commit (void) version" . --top-k 3
```

正常に結果を返した。

---

## 第1章: commit() の (void) version 行

### 現状

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t version,          // ← この version パラメータ
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    // ...
    (void) version;  // ← 未使用パラメータ警告抑制（line 105）
    // ...
}
```

### 調査結果

- `(void) version;` は **単純な未使用パラメータの警告抑制**
- 関数パラメータの `version` は `std::uint64_t` 型の値
- PersistentStateBlock の `version` フィールドとは**無関係**
- 最初のオーバーロード `commit(... version)` で `version` 引数のみを詰めている名残

### 影響

| 影響 | 判定 |
|---|---|
| PersistentStateBlock 導入時の競合 | **なし（単なるパラメータ名）** |
| 削除可否 | **Phase-1a で (void) version 行も同時削除可能** |

---

## 第2章: getVersion() の使用実態

### 現状

```cpp
// ISRRuntimePublicationCoordinator.cpp:168
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    return convo::consumeAtomic(mappedRuntimeGeneration_, std::memory_order_acquire);
}
```

### 調査結果: 本番コードでの呼び出しゼロ

全ソースファイルを grep した結果、`getVersion()` は **テストコードでのみ使用**：

| ファイル | 用途 |
|---|---|
| `ISRSemanticValidationTests.cpp` | テスト内で coordinator.getVersion() を呼び出し |
| 本番コード (src/audioengine/) | **呼び出しなし** |

### 影響

| 影響 | 判定 |
|---|---|
| getVersion の変更 | **本番コードに影響なし** |
| テストコード | 17箇所の getCurrent() と同時移行可能 |
| PersistentStateBlock 移行後 | getVersion は `persistentState_.snapshot().mappedGeneration` を返すよう変更 |

---

## 第3章: isFullyDrained 等の互換性

### 調査結果

`runtimePublicationBridge_` のメソッド呼び出しは以下のみ：

| メソッド | 呼び出し元 | 参照するフィールド |
|---|---|---|
| `commit()` | AudioEngine.Commit.cpp:371 | **全3フィールド + currentWorld_**（変更対象） |
| `retire()` | AudioEngine.Commit.cpp:415 | currentWorld_（変更対象） |
| `getPublicationBacklogCount()` | AudioEngine.h:2723 | backlog カウンタ（**変更不要**） |
| `setRetireBacklogCount()` | 複数 | backlog カウンタ（**変更不要**） |
| `setPendingIntentCount()` | 複数 | pendingIntentCount_（**変更不要**） |
| `setFallbackBacklogCount()` | 複数 | backlog カウンタ（**変更不要**） |
| `setReclaimInFlightCount()` | 複数 | reclaimInFlightCount_（**変更不要**） |
| `setDeferredRetireResidencyCount()` | 複数 | deferredRetireResidencyCount_（**変更不要**） |
| `isFullyDrained()` | AudioEngine.Threading.cpp:108 | backlog カウンタ（**変更不要**） |
| `precheckPublish()` | AudioEngine.h:2839 | ClosureValidator（**変更不要**） |
| `requestShutdown()` | 複数 | state_（**変更不要**） |
| `markShutdownComplete()` | AudioEngine.CtorDtor.cpp:208 | state_（**変更不要**） |

### 結論

PersistentStateBlock の影響範囲は **commit() / retire() / getCurrent() / getVersion() のみ**。
他の全メソッドは backlog カウンタや state のみを参照するため変更不要。

---

## 第4章: 2種類の Coordinator — 移行パス確定

### 現状

```cpp
// AudioEngine.h:76
#include "core/RuntimePublicationCoordinator.h"  // template版

// AudioEngine.h:94
#include "ISRRuntimePublicationCoordinator.h"    // ISR版（古い方）

// AudioEngine.h:2732（typedef）
using RuntimePublicationCoordinator = convo::RuntimePublicationCoordinator<
    RuntimePublishWorld, DSPCore*, RuntimePublicationBridge>;

// AudioEngine.h:2723（メンバ）
// runtimePublicationBridge_ は ISR版（convo::isr::RuntimePublicationCoordinator）
convo::isr::RuntimePublicationCoordinator runtimePublicationBridge_;
```

### 移行パス（確定）

```
Phase-1a 後:
  runtimePublicationBridge_.persistentState_ を使用
  → commit/retire 内の 3つの個別 atomic を PersistentStateBlock に置き換え

Phase-2 後:
  getCurrent() → consumePublishedWorld(store) に置き換え
  → ISR版の currentWorld_ フィールド削除可能に

Phase-4 後:
  currentWorld_ 完全削除
  → ISR版は Pure Metadata Manager に純化
```

---

## 第5章: Phase 計画（最終決定版）

```
Phase-0: 方式決定
  └─ 方式C (non-atomic) 採用。方式B (seqlock) は将来の安全網として維持

Phase-1: 基盤導入
  1a: PersistentStateBlock（方式C + jassert ガード）
      └─ commit() 内の (void) version 行削除
      └─ 3つの個別 atomic → PersistentStateBlock に置き換え
  1b: AuthorityDescriptor + Telemetry（TelemetryRecorder 統合）
  1c: deriveAuthorityState / deriveExpectedState / reconcileAuthorityState
      └─ C++20 requires 型制約
      └─ RepairConfidence（Recovery storm 防止）
  1d: Validator エッジケース（7 tests）

Phase-2: currentWorld_ 段階的削除（前編）
  2a: getCurrent → RuntimeStore 委譲（setRuntimeStore 注入）
  2b: 全17テスト + getVersion() テスト移行
  2c: getCurrent / getVersion のフォールバック削除
  2.5: 監査 + CI 禁止（currentWorld_ 新規参照禁止）

Phase-3: Recovery 統合
  3a: executeRecoveryAction → reconcileAuthorityState 接続
  3b: RepairConfidence による修復強度制御
  3c: ISR-AUTH-002 fullReconciliation 確認

Phase-4: Invariant CI + currentWorld_ 完全削除
  4a-f: ISR-AUTH-001〜006 CI ゲート
  4g: commit() 内 currentWorld_ 書き込み削除
  4h: retire() 内 currentWorld_ CAS 削除
  4i: currentWorld_ メンバ削除
  4j: (void) version 行が削除されたことを CI で確認

Phase-5: テスト拡充
  5a: Model-Based 状態遷移テスト
  5b: Fault Injection 6シナリオ
```
