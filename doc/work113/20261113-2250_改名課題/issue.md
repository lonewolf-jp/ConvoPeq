# 課題: `retireBacklogCount_` の改名と更新経路の整理

- 起票日: 2026-11-13
- 種別: **独立課題**（C3-D / C9-A とは独立。両者の成果に混在させない）
- 状態: **起票のみ**。改名の実装は未承認（別途 Owner の GO が必要）
- 根拠: Owner 指示 2026-11-13（改名の別課題起票 GO）／設計検討 §8

---

## 1. 背景

`retireBacklogCount_` という名前は「残存件数（backlog）」を連想させるが、実測では
**累積 accepted 数**としてしか動いていない。

- 加算: `onRetireAccepted()`（`ISRRuntimePublicationCoordinator.cpp:198`）が +1。
- 減算: `onRetireConsumed()`（同 `:206`）は**production の呼出し元が 0**（`src` 全体を検索）。
- 結果: 1 実行で 143〜164 に単調増加し、0 に戻らない。
- 2026-11-13 の C9-A で、Coordinator の drain 述語から `retireBacklogCount_ == 0` を外した。

## 2. 調査対象

### 2.1 現在の意味論（確認すること）

| 観点 | 現状 |
| --- | --- |
| 意味 | 累積 accepted 数（残存ではない） |
| 減算 | 実装はあるが呼出し元 0 |
| 用途 | pressure の slope 検出（`noteRetireBacklogChanged`） |
| drain 判定 | C9-A で除外済み |

### 2.2 更新経路（production）

```
EQProcessor.Core.cpp:60
  → RuntimeIntentCoordinator::enqueueRetire()          （:149-182）
    → onRetireAccepted()                               （:198）
      → retireBacklogCount_ += 1
      → noteRetireBacklogChanged(newCount)             （:242）
```

- `onRetireConsumed()` の呼出し元は 0。
- `setRetireBacklogCount()`（`:274`）は TEST-ONLY。

### 2.3 参照経路

| 参照元 | 用途 |
| --- | --- |
| `noteRetireBacklogChanged()` | pressure slope（増分のみ） |
| `getRetireBacklogCount()` | 診断・テスト |
| `ShutdownScheduler::isFullyDrained()` | **C9-A で削除済み** |

## 3. 確認対象（起票条件 3）

1. `onRetireAccepted` / `onRetireConsumed` の全呼出し元（production / テスト）。
2. production update path とコード内コメントの整合。特に
   `ISRRuntimePublicationCoordinator.cpp:177` の「本メソッドは production で未使用」は
   **事実と違う**（EQ が本番で呼ぶ）。要修正。
3. pressure 検出の実入力が「累積 accepted の増分」で妥当か（残存でなくてよいか）。
4. `onRetireConsumed()` を削除するか、減算契機を定義するか（全経路会計が前提）。

## 4. 改名候補と変更対象（起票条件 4）

| 項目 | 内容 |
| --- | --- |
| 改名候補 | `retireAcceptedCount_`（次点: `retireAcceptedTotal_`） |
| 変更対象 | メンバ変数、getter、TEST-ONLY setter、関連コメント |
| 関連テスト | `testPressureStateNormalizationContract`（`setRetireBacklogCount` を使用）、`testC9A_*` |
| 移行時の注意 | シリアライズ・外部公開名・ログ項目に旧名が無いことを確認する |

## 5. 移行時の制約（起票条件 5）

- **名称変更に伴って、現在の意味論（累積 accepted）や drain 判定を無断で変更しない。**
- 改名と意味変更（減算の導入など）は別の変更単位に分ける。
- C3-D / C9-A の実装と同じ変更に混ぜない。

## 6. 受け入れ条件（案）

| # | 条件 |
| --- | --- |
| 1 | 全参照が新名に統一され、旧名が残らない |
| 2 | 意味論（累積 accepted・pressure slope）が変わらない |
| 3 | `ctest` 61 件と C9-A のテストが維持される |
| 4 | コメントの事実誤認（production 未使用）が解消される |

## 7. 承認状態

| 項目 | 状態 |
| --- | --- |
| 課題起票 | 承認済み（2026-11-13） |
| 改名の実装 | **未承認**（別途 GO） |
| 減算契機の導入 | 未承認（本課題の調査結果を見て判断） |
