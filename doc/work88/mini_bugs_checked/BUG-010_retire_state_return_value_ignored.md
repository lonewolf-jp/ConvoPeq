# BUG-010: retireEQStateDeferred/retireBandNodeDeferred の戻り値無視によるMKLメモリリーク（~20箇所）

- **発見日**: 2026-07-26
- **カテゴリ**: リソースリーク / メモリ管理
- **関連**: EQProcessor.Core.cpp , EQProcessor.Coefficients.cpp , EQProcessor.Parameters.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`EQProcessor::retireEQStateDeferred()` と `EQProcessor::retireBandNodeDeferred()` は
`bool` を返し、Deferred Deletion Queue が満杯またはシャットダウン済みの場合に `false` を
返す設計である。しかし、**すべての呼び出し元で `(void)` キャストによって戻り値が**
**破棄されている**。これにより、MKL で確保された EQ State / Band Node が
削除されずにリークする可能性がある。

## 該当箇所

### 呼び出し元一覧 (EQProcessor.Core.cpp)
| 行 | コード |
|----|--------|
| 137 | `(void)retireEQStateDeferred(oldState);` |
| 142 | `(void)retireBandNodeDeferred(n);` |
| 234 | `(void)retireEQStateDeferred(oldState);` |
| 570 | `(void)retireEQStateDeferred(oldState);` |
| 623 | `(void)retireBandNodeDeferred(oldNode);` |
| 805 | `(void)retireBandNodeDeferred(oldNode);` |

### 呼び出し元一覧 (EQProcessor.Coefficients.cpp)
| 行 | コード |
|----|--------|
| 75 | `(void)retireBandNodeDeferred(oldNode);` |

### 呼び出し元一覧 (EQProcessor.Parameters.cpp)
| 行 | コード |
|----|--------|
| 29,48,67,90,114,134,167,189,214,248 | `(void)retireEQStateDeferred(oldState);` |

## 問題の詳細

```cpp
// EQProcessor.Core.cpp:93
bool EQProcessor::retireEQStateDeferred(EQState* state) noexcept
{
    if (state == nullptr)
        return true;

    const uint64_t epoch = m_epochDomain.currentEpoch();
    return enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch);
}

// EQProcessor.Core.cpp:26-63 — 内部実装
bool EQProcessor::enqueueDeferredDeleteWithFallback(...) noexcept
{
    ...
    // 初回: Coordinator 経由
    auto result = m_retireCoordinator->enqueueRetire(...);
    if (result == Success) return true;

    // 2回目: stackRouter 経由で再試行
    result = stackRouter.enqueueWithRetry(...);
    return result == RetireEnqueueResult::Success;
}
```

1. `m_retireCoordinator->enqueueRetire()` が失敗すると → **false** が戻る
2. `stackRouter.enqueueWithRetry()` も失敗すると → **false** が戻る
3. 呼び出し元の `(void)` で切り捨て → **ptr に渡した MKL-allocated オブジェクトは破棄されないまま**

全 20 箇所以上が `(void)` でキャストされており、仮にいずれかの Deferred Deletion Queue が
満杯になった場合、毎回の EQ パラメータ更新（ダイヤル操作・プリセット変更など）で
EQState/BandNode のごみ集めが再帰的に発動する。

## 影響

- 通常時 : Retrofit Queue は低負荷でほぼ毎回 `true` を返す → 検出されていない
- 負荷時 : 大量の EQ パラメータ変更（オートメーション・スライダー素早い操作）で
  Deferred Queue が溢れ、漏洩が活性化
- MKL による確保メモリは巨大で（数百万 doubles/チャンネルして 1 EQP 状態で ~1MB+）、
  メモリ劇的に上昇 → Android サポート廃止 / メモリ制限プラグインで不安定

## 元のコメント

```cpp
// [work37 Phase 1.4] bool 返しに変更。全呼び出し元で (void) キャストして既存動作を維持。
```

隊員が既に認識済み、`(void)` を追加して今の動作を維持したコメントがある。
これは実質的に「issue として認識」しているが、「修正されていない」状態。

## 修正案

オプション 1: `retireEQStateDeferred` / `retireBandNodeDeferred` を **void** 化し、
内部で failure を **直接 delete する**（退役失敗しても即解放）：

```cpp
void EQProcessor::retireEQStateDeferredNoWait(EQState* state) noexcept
{
    if (!state) return;
    const uint64_t epoch = m_epochDomain.currentEpoch();
    if (!enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch))
    {
        deleteEQStatePtr(state);  // fallback: 直接解放
    }
}
```

オプション 2: 呼び出し元で返り値をチェックし、失敗時に `getShortestRecordedEpoch` 監視タイマー
経由で再トライ

オプション 3: ステータス公開用の `std::atomic<uint64_t> retiringDropCount` を追加して
失敗時にカウントアップする（監視用、本質的にはリーク）

推奨: **オプション 1** — 退役失敗時は直接 `deleteEQStatePtr()` を呼ぶ方が絶対に安全
（quoi? 遅延退役でなく直接開放するとオーディオスレッドが参照している可能性がある）
→ 実際はピアで Audio Thread と Message Thread の共存検証が必要。