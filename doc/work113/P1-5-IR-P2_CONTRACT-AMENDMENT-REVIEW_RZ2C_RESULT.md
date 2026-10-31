# Contract Amendment Review — RZ2C 実施記録

Status: **RZ2C VERIFIED**
Contract Amendment Review: **NOT CLOSED**
RR verdict: **NONE — 評価していない**
Owner ruling: RZ2C + RZ2B
CDB executions: **0**

---

## 1. 状態は維持されている

```text
Contract Amendment          ESTABLISHED
Contract Amendment Review   NOT CLOSED
RR                          15/18 PASS, 3 FAIL     <- 変更していない
CR-3                        PENDING
R1 execution                NOT AUTHORIZED
```

本記録は RZ2C の結果であり、**RR verdict を含まない**。

---

## 2. 対象（不変）

```text
path   doc/work113/P1-5-IR-P2_Step5EK-R1_Pre-Execution_Contract.md
size   23,533 B
sha256 3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53
```

checker 起動時に `assert` により照合した。

---

## 3. RZ2C の成果物

```text
checker path            C:\VSC_Project\ConvoPeq\tmp\rz2c_instruments.py
checker size            48,285 B
checker sha256          717E957D232AB814A0048A8914DEF78F6FFB406D418F9ECC8A370F4035304C10
descriptor count        18
descriptor digest       E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442
instrument count        40
instrument digest       A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4
checklist source        section 14, machine extracted, 18 items
hand-written needles    0
hand-written region     0
hand-written hash       0
real-model evaluations  0
```

descriptor digest は `rr \t type \t section \t params` のソート行の SHA-256。
instrument digest は `rr \t type \t section \t key` のソート行の SHA-256。
**この 2 つの digest により、次回の Review が「同じ checker を使ったか」を機械的に照合できる。**

---

## 4. §14 の順序（RZ2C §14）に対する結果

| # | step | 結果 |
| --- | --- | --- |
| 1 | §14 checklist extraction | **PASS**  18 items, RR-1..RR-18, 機械抽出 |
| 2 | property classification | **PASS**  18 descriptors, C/D/E/F/G のみ |
| 3 | instrument generation | **PASS**  40 instruments / 18 RR |
| 4 | instrument identity / coverage | **PASS**  全 RR に主計器が 1 つずつ存在 |
| 5 | N/N normaliser self-test | **PASS**  5/5 |
| 6 | structural / positional self-test | **PASS**  C 12/12, D 2/2, G 6/6 |
| 7 | positive / negative / boundary | **PASS**  40 / 40 / 40 |
| 8 | false-positive audit | **PASS**  load-bearing 40/40 |
| 9 | false-negative audit | **PASS**  反転 0 |
| 10 | GATE 3 | **PASS**  3A..3G 7/7 |
| — | checker identity / hash recorded | **PASS** |
| — | real-model guard | **ENFORCED**  clone に対して raise することを検証 |

**1〜10 すべて PASS。したがって RZ2C VERIFIED。**

### 4.1 GATE 3 内訳

```text
OK  3A  descriptor and property-type validation
OK  3B  text-presence instruments (supporting only)
OK  3C  structural instruments
OK  3D  positional instruments
OK  3E  ordering/dependency instruments
OK  3F  quantitative/identity instruments
OK  3G  generated control / positive / negative / boundary
```

---

## 5. property type 配分

```text
C  structural              5   RR-7 RR-11 RR-12 RR-13 RR-17
D  positional              2   RR-6 RR-8
E  quantitative            5   RR-3 RR-4 RR-10 RR-15 RR-18
F  identity / hash         4   RR-1 RR-2 RR-5 RR-16
G  ordering / dependency   2   RR-9 RR-14
A  semantic text-presence  0   主計器として使用されていない
B  exact phrase            0   同上
```

**substring HIT から C/D/E/F/G の verdict へ到達する経路は存在しない。**

---

## 6. §18 の境界 — 実測で強制した

RZ2C は **real contract に対して 1 つも instrument を評価していない**。
これは主張ではなく、次の 2 .Mechanism で実現している。

```text
1. run() は real model を受け取ると RealModelForbidden を送出する
2. その guard は real model の clone に対して実際に raise することを検証した
   （検証して初めて GATE 3 へ進む。検証に失敗したら RZ2C NOT VERIFIED で停止する）
```

自己テスト 120 ケース（40 positive / 40 negative / 40 boundary）はすべて
**synthetic model** 上で実行された。synthetic model は構成により性質を充足するように
作られており、それを Parsing する parser 自体も同時に検証される。

---

## 7. false PASS の防止機構

```text
false-positive audit
  型指向の構造 mutation を各 instrument に 1 つずつ適用し、
  40/40 が False へ反転した。例外なし。すべての instrument が load-bearing である。

false-negative audit
  無視できる差異（hash / size / path のみ）だけを変えた第二の synthetic variant で
  40/40 が True のまま保たれた。過適合でない。
```

**この 2 監査は RZ2C で新設された。** 従来の checker には存在しなかった。
従来の self-test は「意図した事例で checker が動くか」しか確認していない。
false PASS は productive に PASS 而言 Talarid には気付けない。

---

## 8. RZ2C 実施中に同定した新規欠陥（6 件）

```text
#13  parser: 最初の ## より前の header fence を丸ごと捨てていた
#14  parser: §12 の禁止行が同 label で衝突し 1 件しか保持されなかった
#15  instrument: marker_coverage が len(marker)==len(probe) を要求し過厳
#16  instrument: void_condition が N() 無しで case-sensitive 比較
#17  descriptor: 禁止項目数に 15 という手書き定数（実数 16）
#18  violation: permutation_invariance の mutation が no-op
```

**#13 / #14 は「検査対象そのものの誤要約」であり、instrument の型分解では検出できない。
instrument test が検出した。** 詳細は process-defect 記録 §12 に保持した。

### 8.1 構築反復の記録

```text
反復 1  positive FAIL 8 / negative 未検出 13 / boundary 反転 7 / STEP 8 判定式が逆
反復 2  positive FAIL 1 / negative 未検出 1 / boundary 反転 1
反復 3  positive FAIL 1 / negative 未検出 1 / boundary 反転 1
反復 4  positive 40/40  negative 40/40  boundary 40/40  STEP 8 PASS
反復 5  guard を主張から実強制に変更（出力の虚偽記載を修正）
反復 6  guard の検証位置・import 衝突を修正  ->  RZ2C VERIFIED
```

反復 4 で全 instrument test が PASS した時点でも、**出力に「guard enforced」と書かれて
いたが guard は存在しなかった。** 検出は自己検査ではなく、process-defect 記録
§1 の欠陥族（判定器の形が主張と一致しない）を 自己適用したInspect 結果である。

---

## 9. 禁止事項の遵守

```text
CDB launch                 0        source modification       0
debuggee launch            0        build                     0
R1 probe                   0        production runtime        0
g / go / gc                0        .cdb modification         0
Git operation              0        Contract Amendment        0
                                       Amendment 1            0
```

amended contract / Amendment 1 / §15 / G-1..G-7 / AB-1..AB-8 / C1..C4 / DF-1..DF-3 /
Step5EI / T0P / T0Q / probe script / probe order / probe markers / CDB identity /
debuggee identity / SHAPE A-B / invocation / artifact rules / no-retry rule /
authority・A1-B3-C1 / `ConvoPeq.md` / `.cdb` corpus 16 / Release harness
——すべて不変。

---

## 10. 現状

```text
DF-1   ACCEPTED / CLOSED
DF-2   CLOSED BY AMENDMENT
DF-3   CLOSED BY AMENDMENT

Contract Amendment          ESTABLISHED
Contract Amendment Review   NOT CLOSED
RZ2C                        VERIFIED
RR                          15/18 PASS, 3 FAIL     （変更なし・未再評価）

CR-3                  PENDING
14/14                 NOT MET
Step5EK-R1 execution  NOT AUTHORIZED
Implementation        FORBIDDEN
Phase 2-B             NOT STARTED
```

## 11. 次の Owner 判断

§18 により、**現段階では R1 execution authorization を要求する段階にない。**

```text
次の判断対象は「RR-1..RR-18 を amended contract から全件再導出するかどうか」である。
Owner が別途継続を指示した場合にのみ実施する。
RR-1..RR-18 の PASS 无论如何、Contract Amendment Review の CLOSED は
R1 execution authorization を意味しない。
```
