# D102-C2-7 — R_required / Bounded-Retirement Numerical Decision Gate

- **実施日**: 2026-08-26
- **基準版**: `ConvoPeq.md Generated: 2026-08-26 20:06:47`（C2-6 と同一基準）
- **作業種別**: **read-only / decision-only**（production 変更 **0** / test 変更 **0** / CMake 変更 **0** / contract 変更 **0** / telemetry 変更 **0**）
- **位置づけ**: C2-6 で閉じた数値チェーンを decision record として正式固定する gate。D2-4 の Terminal NO-GO は継承し、K_terminal 導入の意味ではない。
- **前提**: D2 series（D2-0 freeze / D2-1 NO-GO / D2-2 PASS-B / D2-3 Gate 不成立 / D2-4 closure）は **CLOSED**。D2 へは戻らない（§5 参照）。

---

## 1. 数値決定 — 正式 decision values

```
M_scope       = 4120                          … 4096 + 13×1.0 + 11（C2-6 C6-1 凍結）
O_denom       = 1                             … campaign-wide max(windowMax)（10 eligible 窓全て windowMax=1）
K_min         = ceil(M_scope / O_denom)
              = ceil(4120 / 1)
              = 4120

R_required    = 1 + K_min
              = 1 + 4120
              = 4121

R_cap         = 5120  (= 4096(D) + 512(Q) + 512(E))

TerminalDep   = max(R_required - R_cap, 0)
              = max(4121 - 5120, 0)
              = 0

Headroom      = R_cap - R_required
              = 5120 - 4121
              = 999
```

### bounded-retirement 判定

```
R_required <= R_cap
      4121 <= 5120   → PASS
```

→ **bounded 5120 のみで充足、Terminal finite 容量を要しない**。
本判定は成長可能な Terminal を無限容量として PASS にしたものではなく、
bounded 容量で先に証明したため Terminal 依存は 0 である
（D102-C2-3 §3 コメント「Terminal は growable だが本判定では無限容量として PASS にせず」）。

---

## 2. 「Terminal dependency = 0」の厳密な意味

**意味するもの**:
> `R_required` の数値的充足に Terminal の容量を必要としない。

**意味しないもの**（誤解防止 — 本 gate で固定）:

- ❌ Terminal が不要
- ❌ Terminal の runtime authority を削除できる
- ❌ Terminal を fixed-capacity に変更できる

**継承する D2-4 結論**（変更なし）:

```
Terminal        = growable（std::vector, store() は常に true — ISRRetireRouter.cpp:24-25, h:138）
K_terminal      = 未導入
QueueFull       = dead enum のまま維持（D2-0 freeze）
P-4 ownership   = 維持（enqueueWithRetry は caller に ownership を残さない）
```

Terminal は bounded 証明が PASS したから不要になるのではなく、
D/Q/E の上流監視（Q/E cap 512 + overflowCount + health）と P-4 最終 authority の設計で
必要であり続ける。

---

## 3. Headroom の扱い

```
headroom = R_cap - R_required = 999
```

- 本 gate の **safety margin** として記録する。
- 「Terminal を bounded 化する余裕」と表現しない。
- bounded retirement capacity に対する数値的余裕である。
- 将来 `M_scope` または `O_denom` の契約変更時に headroom が縮小するかの一次指標として使用する。

---

## 4. 将来条件の formal record（再評価 trigger）

現在（`M_scope=4120, O_denom=1, R_cap=5120`）は `TerminalDep=0`。

同じ worst-case `O_denom=1` を仮定すると `R_required = 1 + M_scope` なので:

```
M_scope ≤ 5119   → R_required ≤ 5120   → TerminalDep = 0  （現行と同じ）
M_scope = 5120   → R_required = 5121   → TerminalDep = 1  （bounded 超過 1）
M_scope > 5119   → 再評価が必要
```

> **Formal trigger**: `M_scope` が **5119 を超える契約変更**を将来行う場合、
> **D102 numeric gate を再評価する**。
>
> その時点で bounded 超過の有無、すなわち Terminal 有限化の要否を
> 本 C2-7 の式（`R_required = 1+ceil(M_scope/O_denom)`）で再判定する。
> 「その時点で Terminal bounded 化する」とは決めない — 再評価対象とするだけである。

---

## 5. D102-C2-7 Decision Matrix（固定）

| Decision item | Result |
|---|---:|
| `M_scope` | **4120** |
| `O_denom` | **1** |
| `K_min` | **4120** |
| `R_required` | **4121** |
| `R_cap` | **5120** |
| `TerminalDep` | **0** |
| `Headroom` | **999** |
| bounded numerical feasibility | **PASS** |
| Terminal finite-capacity requirement | **NONE** |
| **D102 numeric gate** | **GO** |

---

## 6. 追加確認（D7-1 〜 D7-5）

| # | 確認事項 | 判定 | 根拠 |
|---|---|---|---|
| D7-1 | `R_required=4121` が worst-case `O_denom=1` に基づく | ✅ PASS | campaign-wide max=1（10 窓全て windowMax=1）は最悪値。parametric table（O=1..4120）で O=1 が R 最大となることを独立再計算で確認 |
| D7-2 | `O_denom=1` が 10 eligible windows の全 windowMax=1 と一致 | ✅ PASS | `evidence/D102-C2-3-B-Campaign-Execution-Report.md:89-99` — 11 窓中 warmup 1 除外、残り 10 窓全て windowMax=1, sampleCount=5, tag=Normal。Numerical-Result §5 `min=1 mean=1.00 median=1` |
| D7-3 | `R_cap=5120 = 4096+512+512` | ✅ PASS | Numerical-Result:18「4096(D)+512(Q)+512(E)」/ OdenomCampaignTests.cpp:387 `Rcap=5120` |
| D7-4 | `TerminalDep=0` 算出に growable 容量を未使用 | ✅ PASS | `max(4121-5120,0)` は bounded 5120 のみで算出。growable は判定に持ち込んでいない（§1 参照） |
| D7-5 | D2-4 と C2-7 の境界が崩れていない | ✅ PASS | 本 gate は §4 の通り D2 トピック（K_terminal/QueueFull/caller fallback/ignoreUnused）に一切戻っていない |

全 5 項目 PASS。D2 closure を侵食する要素なし。

---

## C2-7 後の次工程

C2-7 が **PASS/GO** のため、**D102-C2 数値系列は正式 CLOSE** とする。

```
D2-4  Terminal bounded = NO-GO / CLOSED
C2-6  O_denom / R_required = GO
C2-7  bounded numerical decision = GO   ★ 本 gate
       ↓
D102-C2 CLOSE
       ↓
C3/C4 remaining gates inventory（最新 ConvoPeq.md 基準で棚卸し）
       ↓
implementation prerequisite audit
       ↓
実装 gate の有無を決定
```

**本 gate では production implementation を開始しない。**
`R_required=4121 / R_cap=5120 / TerminalDep=0 / headroom=999` を正式 decision record として閉じ、
次に **D102-C3/C4 の残存 gate を最新 ConvoPeq.md 基準で棚卸し**するのが次の工程である。

---

## 変更有無

```
production source : 0
test source       : 0
CMake             : 0
contract          : 0
telemetry         : 0
QueueFull         : 変更なし（dead 維持）
Terminal impl     : 変更なし（growable 維持）
```

## 参照

- `evidence/D102-C2-3-Numerical-Result.md`（parametric table, G1-G15 PASS, headroom 999）
- `evidence/D102-C2-3-B-Campaign-Execution-Report.md`（11 窓 raw 全記録）
- `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp:1-7, 372-420`（契約固定・式実装・NOT redefined 出力・compatibility 判定）
- `evidence/D102-C2-6-Numeric-Gate-Odenom-Evidence-Closure-Audit.md`（C6-1〜C6-5）
