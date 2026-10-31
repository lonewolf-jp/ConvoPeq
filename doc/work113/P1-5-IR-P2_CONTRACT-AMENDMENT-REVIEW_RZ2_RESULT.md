# Contract Amendment Review — RZ2 実行記録

Status: **NOT CLOSED / NOT TRUSTED**
Verdict emitted: **NONE** — blocked by RZ2-4 at GATE 3
Owner ruling: RZ2
CDB executions: **0**

---

## 1. 状態は RZ2 裁定どおり昇格させていない

```text
Contract Amendment          ESTABLISHED
Contract Amendment Review   NOT CLOSED
RR                          15/18 PASS, 3 FAIL      <- 15/18 のまま保持
CR-3                        PENDING
R1 execution                NOT AUTHORIZED
```

本記録は RZ2 実行の結果であり、**verdict を含まない。**
GATE 3 が FAIL したため、checker は GATE 3 の直後で停止し、
`RR-1..RR-18` は **一度も評価されていない**。

---

## 2. 対象

```text
path   doc/work113/P1-5-IR-P2_Step5EK-R1_Pre-Execution_Contract.md
size   23,533 B
sha256 3658725552EBCA3D5FF398CBE28C850491B3AFE8E7A988E00F19A096B5E05B53
```

checker 起動時に `assert` により照合した。subject は変更されていない。

```text
PRE  tmp/r1_pre_amendment.md  19,268 B  9690F7C5…  assert により照合
```

---

## 3. RZ2-10 checker evidence

```text
checker path            C:\VSC_Project\ConvoPeq\tmp\rr_review_rz2.py
checker size            29,577 B
checker sha256          8433186C8FAA51FDD26482B4E47E8E132CCB575234284AC8FBC8123CED48D3A7
normaliser form         N(needle) in N(haystack)
                        N = whitespace-run collapse + case fold
                        RAW/N half-normalisation は廃止
needle-set count        111
needle-set identity     runtime set == self-test set == REG,  object identity (id()) により確認
needle-set digest       NOT EMITTED
```

`needle-set digest` が無いのは **RZ2-4 が正しく機能したため**である。
digest 出力ブロックは GATE 3 の後に置かれており、GATE 3 が FAIL した時点で
到達しない。digest のない verdict を出さない、という要求は充足されている。

### 3.1 RZ2-2 / RZ2-3 の構造

```text
DEF(key, hay, text)  が REG に追加する唯一の経路
checker / self-test / control case はすべて REG の同一オブジェクトを参照
has(nd) は未宣言 needle に対して AssertionError を送出
自前の side-by-side needle list は存在しない
```

### 3.2 RZ2-5 needle-set identity（GATE 2、実測）

```text
checker runtime set  : 111
self-test set        : 111
REG authoritative set: 111
runtime == self-test          : True
runtime subset of REG         : True
REG covered by runtime        : True
registered but never evaluated: none
```

**この 3 点はすべて成立した。** identity check 自体は PASS である。
GATE 3 の FAIL は identity 欠陥によるものではない。

---

## 4. GATE 1 — normaliser self-test: **PASS 7/7**

```text
OK  N() is idempotent
OK  N() is symmetric -- N(x) computed the same way on either end
OK  N() lower-cases
OK  N() collapses whitespace runs to one space
OK  N(needle) is precomputed once per registered needle
OK  no registered needle is an empty or whitespace-only string
OK  every registered needle resolves to a declared haystack
```

---

## 5. GATE 3 — control / positive / negative: **FAIL**

```text
positive   needles exercised 111 / 111   FAIL 1
             RR-17.THREE_SEPARATE
negative   mutants exercised 333          FAIL 1
             RR-10.TERMINATOR   one-char   '0'
case       needles carrying upper case : 82 / 111
             assertion 5 (n._n == n.text.lower()) が FALSE
             assertions 1-4 (真の case axis) は全て成立
whitespace needles carrying repeated spaces : 57 / 111
             N('a  b') == N('a b')            : True
             両 spacing で HIT                  : True
```

### 5.1 三件は checker 側の欠陥であり、contract 内容の欠落ではない

**(1) hay key の誤割当 — occurrence #10**

```text
needle  'this document is the FIRST of those three things.'
所在    L22、## 0. Three authorizations, kept separate 節内
割当    hay = "HEADER" = B[:15] = L1..L15
結果    N(needle) in N(HEADER) -> False
```

needle は §0 にあり header fence にはない。registry の hay key 誤り。

**(2) 1 文字 needle の under-determination — occurrence #11**

```text
needle    'q'
変異      '0'
N('0') in N(SCRIPT) -> True    (0x30010 / 0xfff / 0x34040 が '0' を含む)
```

1 文字 needle は substring presence test を支えられない。negative self-test が報告しているのは
**この needle が判別力を持ちえない**という真の性質。
「script が q で終端する」は位置依存の性質であり、構造的計器を要する。

**(3) case assertion の軸混合 — occurrence #12**

```text
assertion 1..4  N(t) in N(h) / N(t.lower()) in N(h) / N(t.upper()) in N(h) / n._n == N(t)  -> 成立
assertion 5     n._n == t.lower()                                                          -> 偽
```

`n._n` は case fold に加えて空白畳み込みも含む。assertion 5 は 2 軸を混合している。

```text
contract 側の欠落 : 0 件
checker 側の欠陥  : 3 件
```

---

## 6. RZ2-4 の作動

```text
GATE 1  PASS
GATE 2  PASS
GATE 3  FAIL
        ↓
VERDICT BLOCKED BY RZ2-4.  Contract Amendment Review = NOT TRUSTED.
RR-1..RR-18 are NOT evaluated.  No PASS/FAIL from this run may be adopted.
```

checker は `sys.exit(0)` で停止した。**RR-1..RR-18 の 1 項目も出力されていない。**
追加 re-check（DF-2 P1..P5 / DF-3 LEVEL 1..4 / G-1..G-7）も未評価。
false-PASS audit も未実行（verdict 出力より後段にある）。

---

## 7. 禁止事項の遵守

```text
CDB launch                 0        source modification   0
debuggee launch            0        build                 0
R1 probe                   0        production runtime    0
g / go / gc                0        .cdb modification     0
Git operation              0        Contract Amendment    0
                                         Amendment 1        0
```

amended contract / Amendment 1 / §15 / G-1..G-7 / AB-1..AB-8 / C1..C4 / DF-1..DF-3 /
Step5EI / T0P / T0Q / probe script / probe order / probe markers / CDB identity /
debuggee identity / SHAPE A-B / invocation / artifact rules / no-retry rule /
authority・A1-B3-C1 / `ConvoPeq.md` / `.cdb` corpus 16 / Release harness
——すべて不変。

---

## 8. Owner に委ねる事項

agent は自己裁定しない。RZ2-12 により停止する。

```text
  RZ2A  3 件の checker 欠陥（hay key 誤割当 / 1 文字 needle / case assertion 軸混合）を
        修正し、GATE 3 を通過させてから RR-1..RR-18 を評価する。
        contract は再編集しない。

  RZ2B  RZ2A に加え、「位置依存の性質には構造的計器を用いる」という規則を §3 に追加し、
        needle 集合から構造的検査への分離を明示する。

  RZ2C  RZ2A/B  notwithstanding、substring needle を主計器とする Review checker を
        採用せず、§14 の checklist 項目から計器を機械的に生成する方式へ移行する。

  RZ2D  いずれも採らず NOT CLOSED のまま保持し、R1 authorization 要求に進まない。
```

**推奨は RZ2B である。** RZ2A 単独では occurrence #10〜#12 が 4 回目の再帰となりうる。
RZ2C は本 workstream で 5 回目の停止が続いている状況に対する唯一の構造的対処であり、
§14 の checklist 定義（contract 本文自身が持つ）を計器生成の入力にすることで、
「needle を手書きする」という欠陥族（RA-3, occurrence #8, #9, #10）そのものを消せる。

---

## 9. 現状

```text
DF-1   ACCEPTED / CLOSED
DF-2   CLOSED BY AMENDMENT
DF-3   CLOSED BY AMENDMENT

Contract Amendment          ESTABLISHED
Contract Amendment Review   NOT CLOSED  /  NOT TRUSTED

CR-3                  PENDING
14/14                 NOT MET
Step5EK-R1 execution  NOT AUTHORIZED
Implementation        FORBIDDEN
Phase 2-B             NOT STARTED
```
