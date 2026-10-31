# Contract Amendment Review — RESULT

Status: **NOT CLOSED**
Result: **RR 15/18 PASS, 3 FAIL**
Owner ruling in force: RY2
CDB executions: **0**

---

## 1. 対象

```text
path   doc/work113/P1-5-IR-P2_Step5EK-R1_Pre-Execution_Contract.md
size   23,533 B
sha256 3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53
```

対象は amended contract のみ。旧 Review（第 1 回）の結果はいかなる項目でも再利用していない。
RR-1..RR-18 の全 18 項目を amended text から再導出した。

比較対象として用いた PRE text:

```text
doc 内には存在しない。tmp/r1_pre_amendment.md を用いた。
19,268 B  9690F7C5C29BC283084FB242FD4970F5CA919730CFB42A5CA12812718F621C9F
```

この digest は historical 値 `9690F7C5…` と一致し、PRE コピーの同一性を確認した。

---

## 2. 3 回の Review の推移

| 回 | 対象 | checker の比較形態 | 結果 |
| --- | --- | --- | --- |
| 第 1 回 | PRE contract | RAW / RAW | RR 18/18 PASS |
| 第 2 回 | amended contract | RAW / RAW（normaliser 除去） | RR 16/18 — RR-11, RR-18 false FAIL |
| 第 3 回 | amended contract | RAW / N（half-normalisation） | **RR 15/18 — RR-1, RR-8, RR-11 false FAIL** |

第 1 回の 18/18 は RAW/RAW が偶然成立した結果であり、比較仕様として健全ではなかった。
第 2 回で normaliser を除去したことが regression iagnosed された（occurrence #6）。
RY2 により normaliser を「復元」したが、**両端同時適用ではなく haystack 側にのみ適用**したため、
case と whitespace の両軸で新たな false FAIL が発生した（occurrence #7）。

第 1 回と第 3 回の RR-18 の差が示すこと: RR-18 の needle
`cdb execution under this contract` は元来すべて小文字・単一空白であり、
RAW/N を素通過するneedle である。normaliser が RR-18 を直し得るのは偶然であって、
normaliser が RR-18 を直すことは normaliser の正当性の証拠にならない。

---

## 3. normaliser control — 8 点

| # | 点 | 結果 |
| --- | --- | --- |
| 1 | needle normalization の有無 | **FAIL — needle に N() が適用されていない** |
| 2 | haystack normalization の有無 | OK |
| 3 | case sensitivity の意図 | **FAIL — needle 側は case-sensitive のまま** |
| 4 | whitespace normalization の意図 | **FAIL — needle 側は 2 空白のまま** |
| 5 | normalization が false-positive を生成しないこと | OK（4/4） |
| 6 | positive self-test | OK（9/9） |
| 7 | negative self-test | OK（6/6、既存 control 維持） |
| 8 | control case | OK（11/11、第 1 試行 FAIL の後、§2.2 の誤前提を是正） |

**点 1 / 3 / 4 が FAIL であるため、この checker の RR verdict は trusted されない。**
self-test 6 点目と 7 目が PASS したのは、self-test の needle が**すべて正規化済みで書かれていた**ため。
checker 実 needle は 2 空白・一部大文字であり、self-test はその needle 集合を検査していない。
§3.1 の制約「control case は checker の実際の needle 集合から導出せよ」に該当し、充足していない。

---

## 4. RR-1 .. RR-18

```text
RR-1   FAIL  ★ case:  needle 'F8B20772…47C9' は大文字、needle 'resolved by SHA-256 at
                execution time' は 'SHA-256' が大文字。N(haystack) は小文字化される。
                第 3 needle 'never hand-transcribed' は HIT。
RR-2   PASS
RR-3   PASS
RR-4   PASS
RR-5   PASS
RR-6   PASS
RR-7   PASS
RR-8   FAIL  ★ whitespace:  needle 'bp or bu at any address.  the gate sets no breakpoints'
                は 2 空白。contract の該当行も 2 空白だが、N(haystack) は 1 空白に畳まれる。
RR-9   PASS
RR-10  PASS  markers=[P3,P1,P2,P4] echo=5 probe=4 lines=10, q terminator
RR-11  FAIL  ★ whitespace:  'c1  parse_reject' / 'c2  eval_failed_non_syntax' /
                'c3  value' / 'c4  no_output' の 4 needle はいずれも 2 空白。
                'mutually exclusive' と 'this is not a pass' は HIT しており、
                case 軸は本回 PASS。失敗しているのは whitespace 軸のみ。
RR-12  PASS
RR-13  PASS
RR-14  PASS  DF-2 P5 が order-independence を提供
RR-15  PASS
RR-16  PASS
RR-17  PASS
RR-18  PASS  needle 'cdb execution under this contract' が
                'CDB execution under this contract.' を HIT
```

**PASS 15 / FAIL 3**

### 4.1 3 件の FAIL はいずれも同一机制・両軸

| 項目 | 軸 | needle | N(haystack) |
| --- | --- | --- | --- |
| RR-1 | case | 大文字 hash / `SHA-256` | 小文字 |
| RR-8 | whitespace | 2 空白 | 1 空白 |
| RR-11 | whitespace | 2 空白 × 4 | 1 空白 |

**contract に該当内容は存在する。** 3 件とも、N/N で比較すれば 3/3 HIT する
（第 2 回の診断スクリプトが N/N で確認した結果と整合する）。

### 4.2 撤回済みの診断

RR-1 / RR-8 / RR-11 の診断、第 1 試行は以下を誤った。

- `F8B20772…` は line-wrap  Shine[不整合] であるとした → **誤り。1 行に収まっている。**
- 7 needle すべて OK とした → **誤り。診断が needle も正規化していたため、checker 実挙動を再現していない。**

occurrence #9 として `P1-5-IR-P2_RY2_PROCESS-DEFECT_Gate-Checker-Normaliser.md` §2.3 に記録した。
RR-1 の真因は **case** であり wrap ではない。確認:

```text
needle as written (uppercase) in N(hay)        -> False
needle after N()          (lowercase) in N(hay) -> True
```

---

## 5. DF-1 / DF-2 / DF-3

```text
DF-1   ACCEPTED / CLOSED      §15 無変更。contract 変更なし
DF-2   CLOSED BY AMENDMENT    V-5 により 10 項目充足
DF-3   CLOSED BY AMENDMENT    V-6 / V-7 により充足
```

DF-2 / DF-3 の amendment 自体に欠落はない。V-1..V-10 は 10/10 PASS であり、
本 Review の 3 件の FAIL はすべて checker 側の比較仕様に帰属する。

---

## 6. Owner §7 の要求 — recurrence confirmation

| 項目 | positive | negative | control | 結果 |
| --- | --- | --- | --- | --- |
| RR-10 | markers/echo/probe/lines/q を実 script text から導出 | なし（数的一致のみ） | なし | PASS |
| RR-11 | `mutually exclusive` / `this is not a pass` HIT | C1..C4 の 2 空白 needle MISS | §2 の control 11/11 | **FAIL（whitespace 軸）** |
| RR-12 | trap 文言・full-form・C2 parser-accepted HIT | なし | なし | PASS |
| RR-14 | VOID-not-a-pass・G-1..G-4・order-independence HIT | なし | なし | PASS |
| RR-15 | AB-1..AB-8・VOID terminal・no-retry HIT | なし | なし | PASS |
| RR-18 | `cdb execution under this contract` HIT | 残り 14 needle HIT | §2 の control 11/11 | PASS |

**Owner §7 の要求「normaliser 復元が他の RR 項目を壊していないこと」→ 充足されていない。**
RR-1 と RR-8 は第 1 回 PASS、第 2 回 PASS であり、本回で false FAIL した。
**新規の false FAIL が 2 件発生している。** false PASS は発生していない
（第 3 回の 15 PASS はすべて同じ mechanism により生じた真 PASS である）。

---

## 7. 状態

```text
DF-1   ACCEPTED / CLOSED
DF-2   CLOSED BY AMENDMENT
DF-3   CLOSED BY AMENDMENT

Contract Amendment          ESTABLISHED   (V-1..V-10 = 10/10)
Contract Amendment Review   NOT CLOSED    (RR 15/18、RR-1 / RR-8 / RR-11 が checker 由来)

CR-3                  PENDING
14/14                 NOT MET
Step5EK-R1 execution  NOT AUTHORIZED
Implementation        FORBIDDEN
Phase 2-B             NOT STARTED
```

CDB は起動していない。R1 probe は実行していない。debuggee は起動していない。

---

## 8. 本記録の範囲

```text
新規            : 本書 1 件、RY2 process-defect 記録 1 件
変更なし        : amended contract、Amendment 1、G-1..G-7、AB-1..AB-8、C1..C4、
                  DF-1..DF-3、Step5EI predicate、T0P / T0Q、authority / A1-B3-C1、
                  CR-3、14/14、§15、probe script、probe order、probe markers、
                  CDB identity、debuggee identity、SHAPE A/B、invocation、
                  artifact rules、no-retry rule
CDB executions  : 0
debuggees       : 0
images executed : 0
source changes  : 0
build           : 0
.cdb changes    : 0
Git operations  : 0
```

---

## 9. Owner に委ねる事項

本 Review は 18/18 に達していない。**agent は自己裁定しない。**

```text
  RZ1  RR-1 / RR-8 / RR-11 の false FAIL を checker 欠陥と裁定し、normaliser を
       needle と haystack の両端に適用する形で再判定する。contract は再編集しない。

  RZ2  RZ1 に加え、本欠陥を RY2 §3 の mandatory control の改訂として記録し、
       「needle 集合の単一定義元」を設ける。§3.1 の制約を、
       「side-by-side 手書きを禁ずる」から「機械的導出を必須とする」へ格上げする。

  RZ3  上記 notwithstanding、checker tooling の連続的 defect を踏まえ、
       以後の gate verdict に対して checker の hash 固定と
       「checker の needle 集合の出力」を verdict の前提としてさせる。

  RZ4  RZ1..RZ3 のいずれも採らず、本 Review を NOT CLOSED のまま保持し、
       R1 execution authorization の要求段階に進まない。
```

推奨は **RZ2** である。RZ1 単独では、本欠陥が 2 回目のamnest 化される構造が残る。
