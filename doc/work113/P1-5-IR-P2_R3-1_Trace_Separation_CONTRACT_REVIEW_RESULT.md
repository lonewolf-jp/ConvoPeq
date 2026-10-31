# R3-1 Trace Separation Repair Contract — CONTRACT REVIEW RESULT

Status: **PASS — for the Owner to rule**
Reviewer self-test: **PASS** (10 negative controls, 1 boundary, 1 positive)
R3-1-P1 .. P8: **8 / 8 PASS**
Adversarial ADV-1 .. ADV-8: **8 / 8 PASS**
Files modified by this review: **none**
CDB executions: **0**

---

## 1. レビュー対象

```text
document  doc/work113/P1-5-IR-P2_R3-1_Trace_Separation_Repair_CONTRACT.md
size      13,139 B
sha256    28805836D44165ECD0E8C13772654214361E10F047D146C6A90DE7CA14A8B53B
clauses   36 parsed across sections 0..13
```

契約は **DRAFTED** 状態であり、本 Review はその受入である。
**本 Review は実装を授権せず、いかなるファイルも変更していない。**

## 2. Review checklist — Owner 由来

以下は Owner が提示した R3-1-P1 .. R3-1-P8 を逐語撥译したものであり、
**reviewer が定義したものではない**。契約も section 12 で自らの受理基準を
定義することを明示的に拒否している（ADV-3 で検証済み）。

```text
R3-1-P1  independent storage
R3-1-P2  independent producer
R3-1-P3  independent consumer
R3-1-P4  object identity separation
R3-1-P5  cross-contamination detection
R3-1-P6  lifecycle separation
R3-1-P7  no verdict dependence on reporting data
R3-1-P8  no real-model execution during self-test
```

## 3. Reviewer self-test（GATE）

```text
negative  R3-1-P1  one container name absent from A.1                detected
negative  R3-1-P1  distinct-object requirement weakened              detected
negative  R3-1-P2  producer permitted to be aimed                    detected
negative  R3-1-P3  a two-container consumer permitted                detected
negative  R3-1-P4  identity replaced by a name comparison            detected
negative  R3-1-P5  one contamination direction removed               detected
negative  R3-1-P5  verdict invariance admitted as the argument       detected
negative  R3-1-P6  a write after close permitted                     detected
negative  R3-1-P7  verdict inputs widened to reporting data          detected
negative  R3-1-P8  the §18 boundary relaxed                         detected
boundary  a re-wrapped, re-cased, re-worded contract                 all 8 still hold
positive  the contract as written                                   all 8 hold
GATE      PASS
```

各計器は成功と失敗の両方を示した後のみ verdict を出した。

## 4. R3-1-P1 .. P8

```text
R3-1-P1  PASS  type F + E
  property   : 二つの phase 所有 container が宣言され、distinct object として要求され、
               第三の container が evaluation record を保持することは禁じられる
  target     : clause group A  (A.1, A.2, A.3, A.4, A.5)
  observation: 5 clause。両 container 名を A.1 で宣言。distinct object を要求。
               alias と第三 container を禁止。container と record の phase tag を要求
  falsifier  : 名が欠ける / A.2 が alias を許容 / 第三 container を許容 / phase tag 不要求

R3-1-P2  PASS  type C + E
  property   : 各 producer は一つの container に hard-bound され、caller が狙い選べず、
               foreign phase tag を拒否する
  target     : clause group B  (B.1, B.2, B.3, B.4)
  observation: 4 clause。container 引数を禁止。binding は構造的であり call discipline ではない
  falsifier  : container 引数を許す producer / B.3 または B.4 欠落

R3-1-P3  PASS  type C
  property   : 各 consumer はちょうど一つの container を名指し、両方を读的 consumer は無い。
               報告される count は報告箇所で phase を明示する
  target     : clause group C  (C.1, C.2, C.3, C.4)
  observation: 4 clause。C.2 と C.3 はそれぞれ一つの container を名指し、C.4 が phase を必須化
  falsifier  : 両 container を股的两た count / C.4 欠落

R3-1-P4  PASS  type F
  property   : 分離は container object の identity 検査で主張され、
               name / length / content / digest の 4 つの代理指標は排除される
  target     : clause group D  (D.1, D.2, D.3)
  observation: 3 clause。D.1 が identity assertion、D.2 が 4 代理を排除、
               D.3 が transition ごとに再実行を要求
  falsifier  : name / length / content / digest による分離 / transition での再実行なし

R3-1-P5  PASS  type E
  property   : 汚染は両方向で TraceContamination の raise によって検出され、
               verdict invariance も count invariance も根拠として使えない
  target     : clause group E  (E.1, E.1a, E.1b, E.2, E.3, E.4)
  observation: 6 clause。E.1a と E.1b が各々 injection と必須 raise を持ち、
               E.2 と E.4 が 2 つの弱い根拠を排除し、E.3 が実行を要求
  falsifier  : 一方向の欠落または空振り / verdict invariance による分離 /
               count による検出 / test が未実行

R3-1-P6  PASS  type G + E
  property   : phase 境界は検証される OPEN/CLOSED lifecycle であり、close 後の write は raise、
               self-test container は RR container 生成前に close される
  target     : clause group F  (F.1, F.2, F.2a..F.2f, F.3, F.4, F.5)
  observation: 11 clause と順序が固定された 6 段階の lifecycle
  falsifier  : lifecycle 段階の欠落・順序違反 / close 後 write 許容 /
               self-test close 前に RR container 生成

R3-1-P7  PASS  type C + G
  property   : どの verdict も gate も trace container を読まず、
               G は A〜F と双方向に独立であることが要求される
  target     : clause group G  (G.1, G.2, G.3)
  observation: 3 clause。両方の性質が独立して成立することを要求
  falsifier  : G.1 の弱体化 / G と A〜F の独立性要求の欠落

R3-1-P8  PASS  type C
  property   : §18 の real-model boundary は invariant として再掲され、緩和されず、
               self-test 段階の real-model 評価を禁じ、execution 禁止事項が継続される
  target     : section 11 と section 13
  observation: §11 が境界を再掲し self-test の real-model 評価を禁止、§13 が禁止リストを保持
  falsifier  : §18 の再掲の削除・弱体化 / self-test での real-model 評価許可 /
               禁止事項の欠落
```

## 5. Adversarial section — 8/8 PASS

```text
ADV-1  verdict invariance は分離の根拠として拒否される        E.2 で確認
ADV-2  G は A〜F と独立であることが要求される                G.3 で確認
ADV-3  契約は自らの受理基準を定義することを拒否              §12 で確認
ADV-4  clause group A〜G すべてに falsifier がある            7 / 7
ADV-5  契約は Owner's checklist を名指し、自前のものは作らない  R3-1-P8 参照を確認
ADV-6  契約は実装順序を固定し、その前で停止する              別途授権を要求
ADV-7  契約は shared / aliased container を禁じる            A.2 で確認
ADV-8  契約は count による汚染検出を禁じる                  E.4 で確認
```

## 6. occurrence #22 — reviewer 自身の計器欠陥（今回の実行中に 4 件）

| # | 内容 | 症状 |
| --- | --- | --- |
| **#22** | Contract Review の reviewer が、**4 件の計器欠陥**を抱えて GATE に出した | 下記 |

```text
(1) parser が fence 内を不透明扱いし、契約中の 36 clause を 1 件も読めなかった
    -> 計器が検査対象を 0 件と見て positive が成立しなかった
    -> fence は表示上の装置であり意味的境界ではない。契約を編集せず parser を修正

(2) 節抽出の lookahead に後続節しか 없어、最終節（§13）を一切抽出できなかった
    -> P8 の禁止事項チェックが false になり positive が成立しなかった

(3) P3 の needle が動詞形を誤った（契約 "shall name exactly one container" に対し
    計器 "names exactly one container"）
    -> 折り返し・表記の相違で positive が成立しなかった

(4) P5 の verdict-invariance needle を clause group に対して張っていたため、
    E.3 の "does not satisfy E" と衝突し、E.2 の弱体化を検出できなかった
    -> negative control 10 件中 1 件が NOT DETECTED
    -> 併せて fixture 2 件（E.1b の方向削除が raise 行だけ消していた／
       E.2 の拒否句そのものを消していなかった）も不正であり、修正した
```

**4 件すべて計器側の欠陥であり、契約の欠落ではない。**
**契約は一度も編集していない。** 契約 digest は Review 前後で
`28805836D44165ECD0E8C13772654214361E10F047D146C6A90DE7CA14A8B53B` のまま。

**#1〜#21 と同じ欠陥族の 22 番目。** reviewer は自らの GATE で 4 件を検出したが、
検出できるまでには 4 回の書き直しを要した。**検出率も収束していない。**

## 7. 数量の訂正

上記 4 件の (1)〜(4) に加え、本 reviewer は 表記の誤り
（`sequential` / `Lisp` 混在）を 1 件出力に含めていた。出力の節を
「順序が固定された 6 段階の lifecycle」に訂正済み。

## 8. この Review の限界 — 隠さず記載する

```text
L1  契約と Review を同一 agent が書いた。P1..P8 は conformance check であり、
    その証拠力は衝突によって弱まる。対立的に構築されているのは ADV-1..ADV-8 と
    negative control 10 件である。

L2  契約 section 2 の「用語の二重使用」は機械化していない。
    checked として主張しない。open review limitation として列挙する。

L3  本 Review は契約の実装可能性と自己検査可能性を評価するものであり、
    実装の正当性は評価していない。実装の受入は別個.Owner 裁定による。

L4  本 Review はいかなるファイルも変更していない。
```

## 9. 禁止事項の遵守

```text
CDB 0  debuggee 0  g/go/gc 0  R1 probe 0  production runtime 0  source 0  build 0
.cdb 0  Git operation 0  Contract Amendment 0  Amendment 1 0  ConvoPeq.md 0
Phase 2-B 未開始
trace 分離の実装 0    checker 変更 0    RR 再評価 0
amended contract 36587255… 不変 / Amendment 1 B163DF96… 不変
.cdb 16 件不変 / HEAD e565019d 不変 / cdb・harness プロセス 0
```

## 10. 状態

```text
R3-1 Repair Contract      DRAFTED
R3-1 Contract Review      PASS  (P1..P8 8/8, ADV 8/8, GATE PASS)
R3-1 implementation       NOT AUTHORISED
Contract Amendment Review NOT CLOSED / NOT TRUSTED
RR-1..RR-18              18/18（Owner 受領済み、HOLD 中）
RB3                       DEFERRED
occurrence #20            OPEN
occurrence #21            OPEN
occurrence #22            OPEN  reviewer の計器欠陥 4 件、訂正済み・記録
CR-3                      PENDING
R1 execution             NOT AUTHORIZED
```

**ここで停止する。** Contract Review は PASS したが、
**PASS をもって実装へ進まない。** 次の Owner 判断を要する。
