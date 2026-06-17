# PR3: リタイアルータ保留 (routerPendingRetire=2) 調査

- **作成日**: 2026-06-17
- **優先度**: Low (情報収集)
- **対象ファイル**: `src/core/EpochDomain.h`, `src/audioengine/ISRRetireRouter.cpp`
- **関連ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

---

## 1. 問題の概要

シャットダウン時のログ:

```
[ISR][Shutdown] Drain incomplete:
  routerPendingRetire=2   ← リタイアルータに2アイテム滞留
```

`routerPendingRetire` は `ISRRetireRouter::pendingRetireCount()` → `EpochDomain::pendingRetireCount()` → `deferredDeletionQueue.sizeApprox()` で取得される。

---

## 2. ソースコード調査による確定事項

### 2.1 滞留メカニズム（確認済み）

`EpochDomain::tryReclaim()` (`src/core/EpochDomain.h:220-231`) は EBR (Epoch-Based Reclamation) に基づき、
**最小 Reader Epoch より新しい epoch を持つアイテムは解放しない**：

```cpp
void tryReclaim() noexcept override {
    const auto n = deferredDeletionQueue.reclaim(getMinReaderEpoch());
    //                                     ↑
    //           minReaderEpoch 以下の epoch のアイテムのみ解放
    //           minReaderEpoch より新しいアイテムは次回以降に持ち越し
    reclaimSuccessCount_.fetch_add(n, std::memory_order_relaxed);
}
```

### 2.2 `drainAll()` の実装（確認済み）

```cpp
// EpochDomain.h:239-241
void drainAll() noexcept override {
    deferredDeletionQueue.drainAllUnsafe();  // epoch不問で全強制解放
}
```

`drainAllUnsafe()` は `DeferredDeletionQueue.h` で定義され、
**Reader Epoch に関係なく全エントリを即時解放**する。

### 2.3 `drainAll()` の全呼び出し元（grep確認）

| 呼び出し元 | ファイル | 行 | コンテキスト |
|-----------|---------|---|------------|
| `m_epochDomain.drainAll()` | `AudioEngine.CtorDtor.cpp` | 206 | **デストラクタ** — 全Reader停止後 |
| `m_epochDomain.drainAll()` | `EQProcessor.Core.cpp` | 137 | EQProcessorデストラクタ |
| `provider_->drainAll()` | `ISRRetireRouter.cpp` | 150 | ルーター経由（委譲） |

`releaseResources()` では意図的に **`drainAll()` を呼ばない**（コメント: `// ★ P1-2: drainAll 禁止 → 安全な tryReclaim`）。

### 2.4 デストラクタでの完全解放（確認済み）

`AudioEngine.CtorDtor.cpp:203-206`:

```cpp
drainDeferredRetireQueues(true);
m_epochDomain.drainAll();  // ここで全アイテム強制解放
```

つまり `routerPendingRetire=2` は **リソースリークではなく、シャットダウンシーケンスの設計上の残余** である。
`releaseResources()` では安全のため `tryReclaim()` のみ実行し、残余はデストラクタまで持ち越される。

### 2.5 `collectDrainAudit()` における位置づけ（確認済み）

`RuntimeDrainAudit.h:32` に明記：

```cpp
uint64_t quarantineResident;    // 監査のみ
```

`quarantineResident` は `isAllZero()` の完了条件から除外されており、
shutdown をブロックしない設計。確認済み。

---

## 3. ノイズとの関連性

### 3.1 直接的な関連性: **低い**

`deferredDeletionQueue` に滞留しているアイテムは「削除待ちのメモリ領域」であり：

- エポックが保護中の場合、Reader (Audio Thread) がまだ参照している可能性がある
- 解放されない = 不正メモリアクセスは起きていない
- よって **直接的な可聴ノイズの原因ではない**

### 3.2 間接的な関連性: **可能性あり**

ただし以下の場合はノイズにつながる可能性がある：

| シナリオ | 発生条件 | ノイズ確率 |
|----------|---------|-----------|
| 滞留アイテムが Reader Epoch の進行を阻害 | Reader Thread が epoch を進めず stopped 状態 | 低 |
| tryReclaim の未解放メモリが次回起動に影響 | プロセス終了するので無関係 | なし |
| 滞留が原因で他のリソース解放が遅延 | シャットダウンタイムアウトの間接要因 | 低 |

---

## 4. 推奨アクション

### 4.1 現状維持（推奨）

現状の設計は意図的であり、安全性の観点から **変更不要**。`drainAll()` 禁止の方針は正しい。

### 4.2 監視強化（オプション）

シャットダウン時に `routerPendingRetire` の値が異常に大きい場合のみ警告を強化する：

```cpp
// 現状: 通常ログ
diagLog("[ISR][Shutdown] Drain incomplete: ... routerPendingRetire=" + ...);

// 強化案: 滞留数が閾値を超えた場合のみ Warning レベルのログ
if (audit.routerPendingRetire > 10) {
    diagLog("[WARNING] [ISR][Shutdown] Abnormal routerPendingRetire="
            + juce::String(static_cast<int64>(audit.routerPendingRetire))
            + " — possible leak");
}
```

### 4.3 drainAll 安全呼び出しの検討（将来課題）

もし `releaseResources()` の時点で全 Reader が停止していることが確認できれば、`drainAll()` を安全に呼び出せる：

```cpp
// releaseResources() 内の shutdownPhase=STOP_AUDIO 以降:
if (m_retireRouter->activeReaderCount() == 0) {
    // 全 Reader 停止確認済み → drainAll 安全
    m_epochDomain.drainAll();
    diagLog("[DIAG] releaseResources: drainAll safe — "
            + juce::String(static_cast<int>(pendingBefore)) + " items reclaimed");
} else {
    // Reader 生存中 → 安全な tryReclaim のみ
    m_epochDomain.tryReclaim();
}
```

ただし現状の `activeReaderCount() == 0` の確認は、
シャットダウンシーケンスのタイミング依存が大きく、誤判断リスクがあるため **今回は見送る**。

---

## 5. 結論

| 項目 | 判定 |
|------|------|
| ノイズの直接原因 | ❌ 可能性は低い |
| 設計上の問題 | ❌ 意図的な安全設計 |
| 監視強化の余地 | △ 閾値超過時の警告追加は有効 |
| 対応優先度 | Low — 問題3は単独では対応不要 |

本Issueの主目的は、`routerPendingRetire=2` が **ノイズ原因ではないことの確認** と、
今後の調査のための **ベースライン情報の記録** である。
