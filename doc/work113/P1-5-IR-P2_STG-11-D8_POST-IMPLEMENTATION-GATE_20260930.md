# STG-11-D8 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D8_POST-IMPLEMENTATION-GATE_20260930.md`
- Work item: **STG-11-D8-1** — NaN IR length の無条件 store による IR 破壊
- Predecessor: Contract Audit（GO） / Implementation Record
- Date: 2026-10-01
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
| 着手時 | D7 マーカー 10 件 / NEWER 0 / FRESH |
| 再生成後 | `821D9215…` / 5,872,152 B / Gen `2026-10-01 00:31:49` / NEWER 0 / FRESH（exit 0） |

D8 の production・test が反映されていることを確認（`STG-11-D8` 9 件）。
`ConvoPeq.md` は編集していない。

---

## 2. Defect / repair の判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| 実コード上の defect | **PASS** | 無条件 store 2 箇所＋jlimit の NaN 素通し＋compute の INT_MIN→1 chain |
| 具体的な failure | **PASS** | NaN 格納→IR 1-sample trim（negative control で `stored (nan)` を実証） |
| 再現可能な sequence | **PASS** | D8-T1（deterministic） |
| 修正後の invariant | **PASS** | D8-T1/T2/T3 |
| 最小 repair（1 defect / 1 unit） | **PASS** | Runtime.cpp のみ。他 defect の混入なし |

## 3. Regression の判定

| 項目 | 判定 |
| --- | --- |
| D8-T1 / T2 / T3（Debug / Release） | PASS |
| Negative control | PASS（FAIL→PASS の両方向実証） |
| D3 / D4 / D5 / D7 sub-test | PASS |
| D1 / D2 回帰（値も不変） | PASS |
| full Debug CTest | PASS 45/45 |
| full Release CTest | PASS 45/45 |
| raw atomic / mutex / alloc audit | PASS（0 件） |
| 静的解析（clang-tidy D8 hunk 0 / cppcheck Runtime 0） | PASS |
| ASAN | 環境 block（別問題として記録し PASS と扱わない） |
| ConvoPeq regeneration + freshness | PASS |
| Diff scope（D1〜D7 production 無変更） | PASS |
| D1〜D7 の保持 | PASS |
| commit / push | **未実施** |

```
READY FOR COMMIT
```

---

## 4. 観測事項（未修正）

- D8 監査で安全確定した項目（dB/saturation の abs-gate、dither の validator、
  adaptive の clampCoeff、learner の tanh＋stability、int-domain の変換＋jlimit、
  setMaxCacheEntries の clamp）は修正不要として記録する。
- tailStart/tailStrength/mixedF 系の無条件 store は NaN 格納の可能性があるが、
  下流被害の証明がないため本 D8 の範囲外として記録のみ（将来の harm 証明時に再評価）。
- D6 の NO-GO・R8 flake は引き続き未解決として残る。
- D8 作業中に既知 flake 署名の事象を 2 件観測（D167 admission FAIL 1 件、
  I2T 早期 crash 1 件）。いずれも再実行で rc=0・D8 全 PASS のため D8 無関係。
  R8 追跡に含めることを推奨する。
