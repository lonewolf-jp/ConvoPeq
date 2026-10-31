# RR-1..RR-18 全件再導出 — RESULT（RB1 + RB2）

Status: **RR 18/18 PASS — Contract Amendment Review の CLOSED 判定は Owner に委ねる**
Owner ruling: RB1 + RB2（RB3 deferred）
CDB executions: **0**

---

## 1. 二層 identity（Owner §6）

```text
model identity        descriptor count  18   E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442   不変
                      instrument count  40   A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4   不変
implementation id     checker sha256         968377BEA8A6FE6A007EDF0DD18444931AB5C3943207CDD594459D9293667A47   新規
                      修正前 checker        B14350D7AF5FAED1D20B73233E6A3ED005CB6AAD1E52276EDB66C479259D1AC4
```

model digest は checker 内部の `assert` で固定されている。最初の実行で
instrument count が 42 になり `INSTRUMENT DIGEST MOVED` を出力して停止した
（KEYS を暗黙集合から明示 40 鍵集合へ戻して解消）。**model digest の移動は技術的に防止される。**

変更していないもの:

```text
property classification   18 descriptors, C/D/E/F/G のみ
primary instrument assignment   不変
descriptor / instrument digest  不変
hand-written needles / regions / hashes   0 / 0 / 0
```

## 2. 全事前検証（Owner §8）1..15 — 15/15 PASS

```text
 1  amended contract identity             23,533 B  36587255…
 2  RZ2C checker identity                 968377BE…
 3  descriptor count                      18
 4  descriptor digest                     E6E75878…  unchanged
 5  instrument count                      40
 6  instrument digest                     A5D93923…  unchanged
 7  property classification               18 descriptors, types CDEFG
 8  instrument coverage                   40 instruments / 18 RR, all non-empty
 9  normaliser self-test                  5/5, A/B unreachable as a primary
10  structural / positional self-test     positive 40/40
11  positive / negative / boundary        40 / 40 / 40
12  false-positive audit                  load-bearing 40/40
13  false-negative audit                  flipped 0
14  real-model guard / property-to-leg    Legs 1-4 PASS, postcondition True
15  evidence completeness                 40/40 complete, 360 records
```

## 3. RB1 — RR-6

```text
property   : invocation の最終 token が debuggee 位置を占めること。
             表記は契約が定義する placeholder 形式。§11 が artefact filename の
             hand-transcription を禁じているため、この形式であることが正である。
evidence   : 契約定義の invocation 表現における token position と placeholder semantics。
verdict    : 上記の構造的条件が成立する場合に限り PASS。

observed   tokens        ["tmp\cdb.exe", "-cf", "<script>", "-logo", "<log>", "<debuggee>"]
           final_slot    "<debuggee>"
           token_count   6
           final_four    ["<script>", "-logo", "<log>", "<debuggee>"]
           placeholder   true
           nothing_follows true
verdict    HOLDS
```

「literal が無いから PASS という」という判定にはしていない。placeholder 形式を positive 条件として
要求しており、literal を final slot に置く mutation は negative 40/40 で検出された。

## 4. RB1 — RR-18

```text
fixture wiring   _pre_region 595 B   _pre_x 16 entries

prohibition_count     pre_count 16  post_count 16  multiset_equal true
byte_identity_to_pre  pre_supplied true  pre_non_empty true
                      pre_bytes 595  post_bytes 595  identical true
verdict               HOLDS / HOLDS -> AND -> PASS
```

§18 の real-model boundary は維持:

```text
Leg 1 PASS  値等価 copy は拒否される
Leg 2 PASS  Leg 1 の後も REAL は保護集合に残る
Leg 3 PASS  REAL 自身は拒否される
Leg 4 PASS  self-test 段階で保護対象の model は一度も評価されていない
3I    PASS  suite 終了後も postcondition が維持される
```

`REAL` に加わったのは**データ**（PRE 参照）であり、guard の登録は 1 件のままである。
real contract は self-test に持ち込まれていない。

## 5. RB2 — evidence traceability

```text
instrument body --> ev("target"|"expected"|"actual") --> EV
run()           --> 同じ evaluation から verdict
reporting       --> Z.records_for(rr,key) を読むだけ。再計算 0
```

前回報告した 14 件の粗い観測表示を解消。前回「section 7 body: 41 lines」としか
出なかった計器は、現在は `c1_text` / `parser_accepted` / `evaluation_failed`
等の構造的事実を返す。evidence completeness **40/40**。

## 6. RR-1..RR-18 — 全件再導出 18/18

```text
RR-1  PASS  F  identity_field / path_field / resolution_rule
RR-2  PASS  F  machine_field                        0x8664
RR-3  PASS  E  excluded_harness_paths               EXCLUDED 4 lines, harness 3
RR-4  PASS  E  enumerated_reasons                   reasons 3, fixed decision present
RR-5  PASS  F  identity_field / size_field / requirement_classified_assertion
RR-6  PASS  D  invocation_final_token               placeholder 形式・最終位置
RR-7  PASS  C  abort_names_forbidden                AB-6
RR-8  PASS  D  forbidden_verbs                      verbs に禁止語 0
RR-9  PASS  G  role_order / negatives_precede_target
RR-10 PASS  E  marker_coverage                      10 = marker 5 + probe 4 + term 1
RR-11 PASS  C  class_label_set / exclusivity_assertion / absence_named
RR-12 PASS  C  syntax_class / evaluation_class / trap_assertion / full_form_keying
RR-13 PASS  C  parser_accepted_classes / reason_given
RR-14 PASS  G  label_contiguity / void_condition / prerequisite_range /
               permutation_invariance                512 assignments x 6 permutations
RR-15 PASS  E  abort_label_set 8 / no_retry_budget / terminal_state /
               retry_repair_forbidden
RR-16 PASS  F  artifact_name / must_not_preexist / hash_resolution / no_cdb
RR-17 PASS  C  execution_not_authorized / prohibition_denies_execution
RR-18 PASS  E  prohibition_count / byte_identity_to_pre
```

前回 16/18 の内訳は引き継がず、18 件すべてを新規に導出した。
各 RR について以下を提示:

```text
RR id / property type / contract-derived descriptor / primary evidence mechanism /
instrument identity / evidence record digest / 判定経路（target・expected・actual・
verdict を計器記録から読み戻し）/ verdict
```

## 7. Post-phase audit

```text
instrument dispatches                  40
dispatch functions used                i_C, i_D, i_E, i_F, i_G
i_A (substring) dispatches             0        <- substring HIT は判定経路に未混入
evidence records from the RR phase     120
evidence printed by recomputation      0
instruments with incomplete evidence   0
real-model guard still armed           True
```

## 8. occurrence #20 — 報告ラベルの不一致（自国の欠陥として開示）

```text
出力行  protected-model evaluations, self-test : 40
```

**このラベルは誤りである。** 正しい値は:

```text
self-test 段階の保護対象評価数    0    （checker 内部の Leg 4 と GATE 3I が検証）
RR 段階の real-model 評価数    40    （Owner 授権下、run_real 経由）
guard の armed 状態             True （REAL_MODELS は 1 件。run_real は arm を緩めない）
```

driver が RR 段階の 40 dispatch を `EVAL_TRACE` に追記した直後に同じ関数を呼ぶため 40 が返った。
印字ラベルが 이를 self-test 段階と書いたため誤った。**判定値には影響しない。**
§1〜§4 の欠陥族（計器の形が主張する性質と一致しない）の 20 番目として記録済み。
R3 の行レベル review 対象にも含める。

## 9. 禁止事項の遵守

```text
CDB 0  debuggee 0  g/go/gc 0  R1 probe 0  production runtime 0  source 0  build 0
.cdb 0  Git operation 0  Contract Amendment 0  Amendment 1 0  ConvoPeq.md 0
Phase 2-B 未開始
amended contract 36587255… 不変 / Amendment 1 B163DF96… 不変
ConvoPeq.md C5873522… 不変 / .cdb 16 件不変 / HEAD e565019d 不変 / プロセス 0
```

## 10. 状態

```text
RZ2C                         VERIFIED（RB1/RB2 版、Owner 再受領待ち）
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT CLOSED   RR 18/18 PASS
RR verdict                   18 / 18（全件再導出）
RB3                          DEFERRED
CR-3                         PENDING
R1 execution                 NOT AUTHORIZED
Implementation               FORBIDDEN
Phase 2-B                    NOT STARTED
```

**本記録は Contract Amendment Review を CLOSED と判定しない。**
RB3（RR-6 / RR-18 の最終分類）と Review の CLOSED 判定はいずれも Owner の裁定による。

## 11. 次の Owner 判断

```text
(a) RB1/RB2 版 checker 968377BE… の受領
(b) occurrence #20 の記録の受領
(c) RR-6 / RR-18 の最終分類（RB3）
(d) Contract Amendment Review の CLOSED 判定
(e) R1 execution authorization（別個の授権）
```
