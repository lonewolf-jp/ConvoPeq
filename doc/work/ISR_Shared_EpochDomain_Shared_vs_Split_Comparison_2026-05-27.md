# ISR Shared vs Split Comparison

作成日: 2026-05-27
対象: R10 latency / jitter / reclaim burst 比較表

---

## 1. 比較軸

| 軸 | shared | split | 判定コメント |
| --- | --- | --- | --- |
| latency | 未測定 | 未測定 | 5分窓の中央値 / P95 を比較 |
| jitter | 未測定 | 未測定 | callback jitter を比較 |
| reclaim burst | 未測定 | 未測定 | retire burst のピーク / 継続時間を比較 |
| shutdown drain | 未測定 | 未測定 | bounded completion への影響を比較 |

---

## 2. 判定

| 判定 | 条件 |
| --- | --- |
| Go(shared継続) | shared が split に対して全軸で劣後しない |
| Go(split移行) | split が 1軸以上で安定性優位、他軸が許容内 |
| No-Go | いずれの方式も A1 / A2 / A4 を満たさない |

---

## 3. 判定記録

- 日時:
- 判定:
- 理由:
- 追跡チケット:
