# STG-11-D7 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D7_POST-IMPLEMENTATION-GATE_20260930.md`
- Work item: **STG-11-D7-1** — session 復元時の未検証 enum cast による RT 配列 OOB
- Predecessor: Contract Audit（GO） / Implementation Record
- Date: 2026-09-30
- HEAD: `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- Branch: `main...origin/main [ahead 9]`, staged = 0

---

## 判定

```
READY FOR COMMIT
```

**commit / push は未実施。** Owner の commit GO を待つ。

---

## 1. Authority 再確認

| 項目 | 値 |
| --- | --- |
| 着手時 | `4014C78F…` / 5,854,098 B / Gen `2026-09-30 22:17:20` / NEWER 0 / FRESH |
| 再生成後 | `29CC8D88…` / 5,864,923 B / Gen `2026-09-30 23:29:19` / NEWER 0 / FRESH（exit 0） |

D7 の production・test が反映されていることを確認。`ConvoPeq.md` は編集していない。

---

## 2. Defect / repair の判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| 実コード上の defect | **PASS** | 未検証 cast 6 件（StateIO.cpp:40/165/169/171/173）＋ RT index 使用（OutputFilter.cpp:226-228）＋ validator caller 0 件 |
| 具体的な failure | **PASS** | 範囲外値 → RT 配列 OOB（誤係数／AV crash）。negative control で `changed to 5` を実証 |
| 再現可能な sequence | **PASS** | D7-T1（deterministic、外部依存なし） |
| 修正後の invariant | **PASS** | D7-T1/T2/T3（拒否・維持・境界） |
| 最小 repair（1 defect / 1 unit） | **PASS** | StateIO.cpp のみ。他 defect の混入なし |

## 3. Regression の判定

| 項目 | 判定 |
| --- | --- |
| D7-T1 / T2 / T3（Debug / Release） | PASS |
| Negative control | PASS（FAIL→PASS の両方向実証） |
| D3 / D4 / D5 sub-test | PASS |
| D1 / D2 回帰（値も不変） | PASS |
| full Debug CTest | PASS 45/45 |
| full Release CTest | PASS 45/45 |
| raw atomic / mutex / alloc audit | PASS（0 件） |
| 静的解析（clang-tidy D7 hunk 0 / cppcheck StateIO 0） | PASS |
| ASAN | 環境 block（別問題として記録し PASS と扱わない） |
| ConvoPeq regeneration + freshness | PASS |
| Diff scope（D1〜D5 production 無変更） | PASS |
| D1〜D6 の保持 | PASS |
| commit / push | **未実施** |

```
READY FOR COMMIT
```

---

## 4. 観測事項（未修正）

- `validatePresetStateTreeForDebug` は依然 caller 0 件。D7 は restore-site guard で
  対応したため、validator の配線は行わない（重複機構を作らない）。将来の集中検証が
  必要になった場合の投入候補として記録する。
- D6 の NO-GO（ownership/RT 系）・R8 flake は引き続き未解決として残る。
