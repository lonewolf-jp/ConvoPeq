# D102-C2-3 — Numerical Result（O_denom 確定後の即時計算・parametric）

- **実施日**: 2026-08-25
- **作業種別**: read-only numerical application（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + `evidence/D102-C2-3-A-Harness-Capability-Audit.md` + `evidence/D102-C2-3-Odenom-Campaign-Raw-Evidence.md`
- **状態**: **CLOSE PASS** — O_denom=1 を campaign-wide maximum として実測確定、K_min/R_required 数値確定済み。TBD なし

---

## 1. 固定入力（D102-C2-2 確定・変更禁止）

```text
λ_prod_bound = 13 events/s
G_bound      = 1.0 s
K_starve     = 1.0 s
T_sampler    = 100 ms
M_scope      = 4120  (= 4096 + 13*1.0 + 11)
R_cap,bounded = 5120 (= 4096(D) + 512(Q) + 512(E))
```

## 2. O_denom 定義（campaign-wide maximum）

```text
O_denom = max(windowMax(w)) over all eligible windows w

eligible = valid==1 && counterWrapped==0 && missedTickCount==0
           && windowTag==Normal && sampleCount>=2
           && campaignStart <= windowStart <= windowEnd <= campaignEnd
```

取得経路: `telemetry.lastClosedSnapshot().windowMax`（`ISRWorldRetirementTelemetry.h:194`）を全 eligible Closed window について `max` 集約（外部集約、telemetry 変更なし）

## 3. O_denom 確定後の即時計算（D102-C2-3 §8）

```text
K_min      = ceil(M_scope / O_denom) = ceil(4120 / O_denom)
R_required = 1 + K_min
```

### Parametric table（M_scope=4120 固定）

| O_denom | K_min = ceil(4120/O_denom) | R_required = 1+K_min | Terminal dep = max(0,R-5120) | bounded 判定 |
|---:|---:|---:|---:|---|
| 1 | 4120 | 4121 | 0 | **PASS** |
| 2 | 2060 | 2061 | 0 | **PASS** |
| 4 | 1030 | 1031 | 0 | **PASS** |
| 8 | 515 | 516 | 0 | **PASS** |
| 13 | 317 | 318 | 0 | **PASS** |
| 20 | 206 | 207 | 0 | **PASS** |
| 50 | 83 | 84 | 0 | **PASS** |
| 100 | 42 | 43 | 0 | **PASS** |
| 500 | 9 | 10 | 0 | **PASS** |
| 1000 | 5 | 6 | 0 | **PASS** |
| 4120 | 1 | 2 | 0 | **PASS** |

### 閾値証明

```text
R_required <= 5120
  ↔ 1 + ceil(4120/O_denom) <= 5120
  ↔ ceil(4120/O_denom) <= 5119
  ↔ O_denom >= ceil(4120/5119) = 1
```

∴ **M_scope=4120 の下では、構造的下界 O_denom≥1 を満たす任意の実測値で bounded compatibility は PASS。最大要求 4121 に対し headroom 999。**

Terminal は `std::vector` growable（`ISRRetireRouter.h:138`）だが、本判定では無限容量として PASS にせず、bounded 5120 で先に証明したため Terminal 依存は 0。将来 M_scope が 5119 を超える変更があった場合のみ Terminal の finite-memory 評価が必要（D102-C3 §8）。

## 4. 実測値入力欄（campaign 実行後に埋める）

```text
campaign start:  104614058528 (2026-08-25 23:40 campaignStartUs)
campaign end:    104623794884 (2026-08-25 23:40 campaignEndUs, duration 9736356us = 9.736s)
eligibleWindowCount:   10
excludedWindowCount:   1
O_denom (measured max): 1
argmax windowId:        2

Derived:
K_min      = ceil(4120 / 1) = 4120
R_required = 1 + 4120 = 4121
R_cap,bounded = 5120
Terminal dependency = max(0, 4121-5120) = 0
Numerical compatibility = PASS bounded
```

**診断情報（denominator にしない — 平均・P95・P99 は参考のみ）:**

```text
min(windowMax)    = 1
mean(windowMax)   = 1.00
median(windowMax) = 1.00
P95               = 1
P99               = 1
max(windowMax)    = O_denom = 1
```

## 5. 最終報告テーブル（D102-C2-3 用・O_denom 実測後に確定）

| Parameter | 最終状態 |
|---|---:|
| λ_prod_bound | 13 events/s |
| G_bound | 1.0 s |
| K_starve | 1.0 s |
| T_sampler | 0.1 s |
| N_timer | 11 |
| M_scope | 4120 |
| O_denom | **1 (実測 campaign-wide maximum)** |
| K_min | **4120** (`ceil(4120/1)`) |
| R_required | **4121** (`1+4120`) |
| R_cap,bounded | 5120 |
| Terminal dependency | **0** (`max(0,4121-5120)`) |
| Numerical compatibility | **PASS bounded** (`4121 <= 5120`) |

## 6. D102-C2-3 PASS 条件（G1-G15）対応

| Gate | 条件 | 現状 |
|---|---|---|
| G1 | 22:11:33 ソースと実 `src/` が一致 | **PASS** (`git diff HEAD -- src/` = 0, `git diff --stat` = ConvoPeq.md 2 +- のみ) |
| G2 | production source modification = 0 | **PASS** |
| G3 | campaign 開始・終了時刻を記録 | **PASS** (104614058528 → 104623794884) |
| G4 | 全 Closed window を記録 | **PASS** (11 windows) |
| G5 | eligibility を事前固定条件で機械判定 | **PASS** (§2 式を事前固定、OdenomCampaignTests.cpp で機械適用) |
| G6 | excluded window と理由を全件記録 | **PASS** (1 WarmupExclusion) |
| G7 | eligible window が複数存在 | **PASS** (10) |
| G8 | workload observed rate を contract rate と分離 | **PASS** (Observed 4.11/s vs contract 13/s) |
| G9 | `O_denom = max(windowMax)` を全 eligible に対して算出 | **PASS** (1) |
| G10 | `O_denom >= 1` | **PASS** (1) |
| G11 | `K_min = ceil(4120/O_denom)` | **PASS** (4120) |
| G12 | `R_required = 1 + K_min` | **PASS** (4121) |
| G13 | `R_required <= 5120` | **PASS** (4121 ≤ 5120) |
| G14 | Terminal dependency = 0 | **PASS** |
| G15 | campaign raw evidence を保存 | **PASS** (evidence/D102-C2-3-B-Campaign-Execution-Report.md + OdenomCampaign_console_2026-08-25.log) |

### 総合

```text
D102-C2-3-A Harness Audit         : PASS (本監査)
D102-C2-3 Raw Evidence            : PASS (11 windows, 実測値で確定)
D102-C2-3 Numerical Result        : PASS — O_denom=1, K_min=4120, R_required=4121
D102-C2-3 全体                    : PASS (G1-G15 全 PASS, production 0, bounded compat PASS)
```

## 7. やってはいけないこと — 遵守確認（D102-C2-2 § 続）

| 禁止事項 | 遵守 |
|---|---|
| `O_denom=1` を保守的だからと仮採用 | ✅ parametric で全値を列挙、実測待ち |
| 単一 window で確定 | ✅ campaign-wide max を要求 |
| `R_required` から O_denom 逆算 | ✅ O_denom → K_min → R_required 順序厳守 |
| Terminal growable で自動 PASS | ✅ bounded 5120 で先に証明 |
| 実測値を λ/G の safe bound に昇格 | ✅ observed と contract を分離記録 |
| ソース変更で測定都合を作る | ✅ production 0 変更、harness-only のみ |

## 8. 次ステップ（O_denom 実測）

```text
1. AudioEngineHarness 再ビルド (build.bat / cmake --build)
2. runOdenomCampaignDefault() 実行 (10+1 windows, 4 pubs/window, 60ms interval, 100ms sampler)
3. Raw Evidence 表を実測 snap で埋める
4. 本ファイル §4 は実測 O_denom=1 / K_min=4120 / R_required=4121 で確定済み
5. G3-G15 を実測値で再チェックし、D102-C2-3 を CLOSE
```

---

*本ファイルは 2026-08-25 23:40 campaign 実測 (O_denom=1, 10 eligible windows) で確定し、CLOSE した。production source 変更 0、O_denom の恣意的仮確定 0。*
