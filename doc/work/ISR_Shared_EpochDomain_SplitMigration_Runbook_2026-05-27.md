# ISR Shared EpochDomain Split Migration Runbook

作成日: 2026-05-27
対象: R10 Shared Epoch Canonical 前提の移行コスト固定

---

## 1. 目的

shared epoch が canonical ではない前提で、split migration へ切り替える際の

- 切替手順
- ロールバック手順
- 判定条件
- 証跡
を単一文書に固定する。

---

## 2. 発動条件

以下のいずれかを満たした場合に split migration を検討する。

- shared 継続時に latency / jitter / reclaim burst のいずれかが許容閾値を継続超過
- retire burst が shared domain で吸収しきれない
- shutdown drain の収束時間が運用許容を超える
- shared 方式の方が split 方式より安定性で劣後しないことを確認できない

---

## 3. 切替手順（最小）

1. 計測窓を固定する

   - 同一ビルド
   - 同一負荷条件
   - 同一 host 設定
2. shared / split の両方を同じ scenario で実行する
3. 取得指標を保存する

   - callback latency
   - jitter
   - reclaim burst
   - retire queue depth
   - shutdown drain
4. 判定表に結果を記録する
5. Go の場合のみ split へ進める

### 3.1 比較入力 JSON フォーマット（`isr-compare-shared-split-epoch.ps1`）

比較スクリプトの入力は以下キーを持つ JSON とする。

- `latencyMs`
- `jitterMs`
- `reclaimBurst`
- `shutdownDrainMs`

サンプル:

- `doc/work/samples/shared_epoch_metrics_sample.shared.json`
- `doc/work/samples/shared_epoch_metrics_sample.split.json`

上記サンプルを複製して実測値へ置換し、
`isr-compare-shared-split-epoch.ps1` の `-SharedMetricsPath` / `-SplitMetricsPath` へ渡して比較表を更新する。

---

## 4. ロールバック手順

1. split migration を停止する
2. shared domain へ authority を戻す
3. retire queue を shared 既定へ戻す
4. shared scenario を再実行し、収束を確認する
5. rollback 事由を記録する

---

## 5. 記録テンプレート

### 5.1 実行条件

- 日時:
- ビルド構成:
- host:
- scenario:
- 参照コミット:

### 5.2 判定

- shared:
- split:
- 判定:
- 理由:

### 5.3 エスカレーション

- 閾値超過項目:
- 一次対応:
- 次回実施予定:

---

## 6. Go/No-Go

### Go(shared継続)

- shared が split に対して latency / jitter / burst の全軸で劣後しない

### Go(split移行)

- split が shared より 1軸以上で安定性優位
- 他軸で許容内

### No-Go

- いずれの方式も A1 / A2 / A4 を満たさない

---

## 7. 参照

- `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- `.github/scripts/isr-compare-shared-split-epoch.ps1`
- `.github/scripts/isr-record-shared-split-go-no-go.ps1`
- `doc/work/ISR_Shared_EpochDomain_Shared_vs_Split_Comparison_2026-05-27.md`
- `doc/work/ISR_Completeness_Risk_Backlog.md`
