# R3 — 独立行レベルレビュー RESULT

Status: **STOP CONDITION MET**
Checks: **17 / 18 PASS**（R3-1 が genuine FAIL）
Reviewed artefacts: **変更なし**
CDB executions: **0**

---

## 1. レビュー対象

```text
checker  C:\VSC_Project\ConvoPeq\tmp\rz2c_instruments.py
         968377BEA8A6FE6A007EDF0DD18444931AB5C3943207CDD594459D9293667A47   MATCH
driver   C:\VSC_Project\ConvoPeq\tmp\rr_evaluate_real.py
         930727D3051C139B19388D9981EA857C...
reviewer C:\VSC_Project\ConvoPeq\tmp\r3_independent_review.py  （本レビュー）
```

本レビューは**判定を checker の印字から採らない**。結論は `ast` によるソース構造と
first-principles な数値再構成から導く。偽実化は使い捨て COPY 上で行い、reviewer が削除する。

---

## 2. R3-1 — self-test trace と RR-phase trace の分離：**FAIL（genuine）**

### 2.1 静的解析結果（行レベル）

```text
checker 側（self-test）
  L855  <module>  Store   EVAL_TRACE = []
  L879  evaluated_protected_model  Load  return [t for t in EVAL_TRACE if t[0] in prot]
  run() 内の append は ast 上 Attribute Load として現れ、mutator 検出で enclosing
         function = run と特定された

driver 側（RR phase）
  L138  <module>  Load    Z.EVAL_TRACE[:] = [t for t in Z.EVAL_TRACE]   （no-op）
  run_real() 内の Z.EVAL_TRACE.append(...)
```

```text
containers : checker が list を 1 つ定義し、driver はその同じ list に append する
logical separation : 呼び出し箇所による（checker run() / driver run_real()）  分離あり
data separation    : phase ごとの独立 storage                                 なし
```

### 2.2 判定

```text
Owner R3-1 は「論理的・データ構造的に区別されていること」を要求する。
論理的分離   あり
データ構造的分離 なし
-> R3-1 FAIL
```

**同一容器を共有している以上、phase を誤って混入しても区別する手段が無い。**
現在のところ混入していないことは R3-2 で確認したが、**構造的に区別できる状態ではない。**

---

## 3. R3-2 — occurrence #20 の報告値を独立に再構成：**PASS**

```text
checker を単独で import した直後（self-test 段階のみ）
    EVAL_TRACE entries            120   （synthetic model での評価）
    protected-model evaluations     0

RR phase を first principles で再現（real model での 40 instrument 評価）
    protected-model evaluations    40
    of which from the self-test     0
    of which from the RR phase     40
    RR verdict                 18 / 18 PASS
    i_A (substring) dispatches     0
```

```text
R3-2a  self-test protected-model evaluations == 0      PASS
R3-2b  RR-phase protected-model evaluations == 40      PASS
R3-2c  40 は RR phase からのみ、self-test からのものは 0     PASS
R3-2d  再現した RR verdict は報告値と一致 18/18            PASS
```

**occurrence #20 の「40」は RR phase 由来のみであることが独立に確認された。
self-test の verdict 計算へ混入していない。ラベルは誤りだったが、値は正しかった。**

---

## 4. R3-3 — verdict independence：**PASS**

```text
driver 内の verdict 代入は 1 箇所のみ
  L151  VERD[cid] = ok
verdict 式が読む名前 : ['ok']      （= instrument の return value の AND）
Forbidden 名の intersection : 空
    EVAL_TRACE / evaluated_protected_model / TRACE / RECHASH / records_for / EV
```

```text
run_real call site    L[149]
records_for call site L[152, 190]
すべての records_for が、いずれかの run_real より後                    PASS
```

**reporting label → verdict の逆流は存在しない。**

---

## 5. R3-4 — evidence / verdict の生成順序：**PASS**

```text
i_* body 内の ev() call site        120
run() は dispatch の戻り値を verdict とする（dispatch が先に evidence を記録する）
driver 側の evidence consumer      L[103,108,137,152,185,190]
                                 印字する、または FIELD NAME のみを比較する
```

**evidence が記録され、→ verdict が返る → reporting が読む。順序は逆っていない。
report された value から verdict を生成する経路は無い。**

---

## 6. R3-5 — execution identity：**PASS**（1 回目の判定は偽 FAIL → §11 参照）

```text
run_real body の call :
    TRACE.append((mi["rr"], mi["type"], mi["key"], fn.__name__))
    Z.EVAL_TRACE.append((id(m), mi["rr"], mi["key"]))
    raise AssertionError("run_real requires the REAL model")
    return bool(fn(m, mi["section"], mi["params"], mi["key"]))
process-launch 名（subprocess / Popen / cdb / debuggee / spawn / exec 等）  なし
```

```text
dispatch target 6 個を AST 定義として検査
    i_F / i_E / i_D / i_C / i_G / i_A   すべて process 名 0
```

### 6.1 execution identity の記録（Owner §7 の要求）

```text
R1 probe execution
    = cdb を debuggee に対して R1 script と共に起動すること。      count 0
    本回・RB1/RB2 回とも一切発生していない。

RR phase の run_real
    = 契約の parsed model に対する Python predicate の評価。
      cdb を起動しない。debuggee を起動しない。image を実行しない。
      probe expression を評価しない。process を生成しない。

両者は同一物ではない。R1 execution authorization は本結果によって発生しない。
```

---

## 7. R3-6 — 偽実化 control（使い捨て COPY 上、reviewer が削除）：**PASS**

```text
R3-6a  無改変 driver の実行                              18/18
R3-6b  誤ラベル文言のみを変更                             18/18
R3-6c  EVAL_TRACE に 999 件の毒入れ、EV に 99 件の毒入れ   18/18
```

**ラベル文言、trace 内容、evidence 内容のいずれを弄っても verdict は不変。**
RR verdict はこれら表示値に依存していない。

ただし R3-6c は同時に、**共有容器の構造的欠陥が「誤った trace を混入させても
判別できない」ことを示している。** R3-1 の FAIL を補強する。

---

## 8. R3-7 — model invariants（reviewer 自身が再導出）：**PASS**

```text
descriptor count 18   digest E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442
instrument count 40   digest A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4
contract              3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53
i_A dispatch          0
```

checker の印字を read せず、reviewer が model から再計算した。

---

## 9. occurrence #21 — reviewer 自身の計器欠陥（R3-5b の 1 回目の偽 FAIL）

| # | 内容 | 場所 |
| --- | --- | --- |
| **#21** | R3-5b が **live 関数オブジェクトに `ast.FunctionDef` 型判定を適用**したため、dispatch target 6 個全てを「impure」と誤判定した。検査対象は AST 定義であるべきなのに Live object を検査していた | R3 reviewer script |

```text
誤った検査 : all(isinstance(v, ast.FunctionDef) and ... for v in Z.DISPATCH.values())
実際の対象 : CT.body 内の i_* FunctionDef .ast
誤判定     : R3-5b FAIL（6 targets）
訂正後     : i_F / i_E / i_D / i_C / i_G / i_A すべて process 名 0  -> PASS
```

**#1〜#20 と同じ欠陥族の 21 番目。判定器の形が検査対象の形式と一致していなかった。**
自己申告を排除するため reviewer は静的解析で動くが、その静的解析にも同じ欠欠陥族童族の混入を混入した。

---

## 10. 停止判定

```text
Owner §8 停止条件
    ・self-test と RR trace の境界が明確でない            -> R3-1 FAIL で該当
    ・RB1/RB2 修正以外の計器モデル変更が必要になる        -> 該当しない（model digest 不変）
    ・descriptor / instrument digest が変化する            -> 該当しない
    ・contract 変更が必要になる                          -> 該当しない

R3 checks  17 / 18 PASS
R3-1       FAIL（genuine）
```

```text
Contract Amendment Review = NOT TRUSTED.   RR verdict = NONE / HOLD.
```

reviewer は review であり repair ではない。**review 対象の 2 ファイルは変更していない。**
修正が必要なら新たな Owner 裁定を要する。

---

## 11. RB3 の最終分類 — 本レビューでは確定しない

Owner §6 により RB3 は R3 完了後に裁定される。本レビューは R3 の判定材料のみarth供する。

```text
contract  -> descriptor  -> primary mechanism -> instrument -> fixture
          -> evidence    -> verdict
```

各段階の接続は R3-2 / R3-3 / R3-4 で確認した。ただし R3-1 の構造的欠陥があるため、
`fixture`（評価対象データ）が `instrument` へ正しく供給されていることの**構造的**保証は
trace 経路についてのみ欠けている。instrument への入力供給そのものは R3-2 の
数値再構成で 18/18 として確認済みである。

**最終分類は Owner の裁定による。本記録は分類しない。**

---

## 12. 状態

```text
RZ2C                         VERIFIED / RB1-RB2
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT TRUSTED
RR-1..RR-18                 18/18 PASS（Owner 受領済み。ただし R3 STOP 中は HOLD）
RR result                    ACCEPTED AS NEW FULL REDERIVATION
RB1 / RB2                    ACCEPTED
RB3                          DEFERRED
occurrence #20               OPEN  — 値は独立確認済み。構造的分離が未充足
occurrence #21               OPEN  — reviewer 自身の偽 FAIL、訂正済み・記録
CR-3                         PENDING
R1 execution                NOT AUTHORIZED
Implementation              FORBIDDEN
Phase 2-B                   NOT STARTED
```
