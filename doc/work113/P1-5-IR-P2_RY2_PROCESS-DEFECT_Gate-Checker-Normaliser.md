# PROCESS DEFECT — Gate checker の normaliser 欠落／half-normalisation による false FAIL

Status: RECORDED, OPEN as a process control
Owner ruling: RY2
Scope: gate **checker** tooling and verification process. **No contract, authority, source, build, `.cdb` corpus, or Git state was touched.**

---

## 1. 欠陥の記述

```text
PROCESS DEFECT:
Gate checker における normaliser の欠落／除去／半適用により、
case variation および whitespace variation を含む semantic presence check が
false FAIL となった。
```

Checker の比較仕様には 3 形態があり、**両端を同じ階段に置くこと**が必要である。

| 形態 | needle | haystack | 結果 |
| --- | --- | --- | --- |
| RAW / RAW | 生 | 生 | case variation で false FAIL（RR-11, RR-18 の元凶） |
| **RAW / N** | 生 | 正規化済 | **case と whitespace の両方で false FAIL（RR-1, RR-8, RR-11）** ← 今回採用した形態 |
| N / N | 正規化済 | 正規化済 | 正しい |

**RAW/N は RAW/RAW および N/N より弱い。** 判定力を失わない_NORMALISED_段階を haystack 側にのみ
適用した場合、needle 側の case と whitespace がそのまま誤差として残る。normaliser を「復元した」
という_labels_ は、その両端同時適用を意味しない。

normaliser の適用対象は **checker ではなく checker 内の両端（needle と haystack）** である。

---

## 2. 発生履歴

| # | ゲート | checker の形 | 誤判定 | 段階数 |
| --- | --- | --- | --- | --- |
| 1 | Step5EJ | needle 未正規化 | CR-9 / CR-10 / CR-12 false FAIL | 3 |
| 2 | Contract Review（第 1 回） | needle 未正規化 | RR-11 false FAIL | 1 |
| 3 | RA-3 | 手書き語彙リスト（実測対象と非対応） | false verdict | 1 gate |
| 4 | RA-7 | 引用と supersession の同一視 | false verdict | 1 gate |
| 5 | V-6 / V-7 | 領域抽出 index 誤り | false FAIL | 1 gate |
| 6 | Contract Review（第 2 回、RY2 前） | **normaliser 除去による regression** | RR-11 / RR-18 false FAIL | 2 |
| 7 | Contract Review（第 3 回、RY2 後） | **half-normalisation（RAW/N）** | RR-1 / RR-8 / RR-11 false FAIL | 3 |
| 8 | Contract Review（normaliser self-test 第 1 試行） | control case の前提誤り | self-test false FAIL | 1 |
| 9 | RR-1/RR-8/RR-11 の診断（第 1 試行） | 診断が needle も正規化し、checker の実挙動を再現せず | 診断 6 miss を 0 miss と報告 | 1 |

### 2.1 #6 は regression である

第 1 回の Contract Review（PRE contract 対象）では **RR-11 と RR-18 は PASS していた**。
両者は RAW/RAW 比較であり、case variation を含む 2 needle を除いて偶然成立していた。
verifier を書き換えた際、**これら 2 つを保護していた比較仕様を意図せず失った**。
raw text への切り替えは「より厳密化为」として行われたが、結果は判定力の喪失であった。

### 2.2 #8 は self-test が自分の対象を測っていなかった

RY2 で導入した normaliser self-test の control case は、
「RR-11 / RR-18 の needle は amended text に HIT し、PRE text に MISS する」と主張した。
**§7 と §12 は amendment 非変更領域である**ため、両方の text に存在するのが正しい。
この control は「差が無い場所に差を主張する」control であり、それ自体が欠陥であった。
self-test は own の対象を測れていないため、Review への進入を正しく BLOCK した。

### 2.3 #9 は診断が checker と一致しなかった

RR-1/RR-8/RR-11 の診断スクリプトが `N(needle) in N(hay)` すなわち N/N で比較したため、
checker 実挙動の RAW/N と一致せず、6 件の miss を 0 件と報告した。
同時に hash の line-wrap という存在しない原因を主張した。
**この診断は撤回済み。** 両誤りとも「計器が測った対象と形が一致しない」という同一欠陥族である。

### 2.4 単一欠陥族としての収束

上記 9 件は 3 種に収束する。いずれも **判定器の形がрации它の主張する性質と一致しない**。

1. **比較両端の非対称** — #1, #2, #6, #7（case / whitespace）
2. **検査対象と検査尺の非対応** — #3, #4, #8, #9
3. **領域取り違え** — #5

3 種すべては「**その検査が成立するための前提を、測定器自身が検査していない**」という同型である。

---

## 3. 以後の gate checker に対する mandatory control（RY2 追加措置）

**本項は amended contract の内容変更ではない。gate checker の検証プロセスの強化である。**

各 checker review は以下 8 点を **報告可能に**しRET 才能その verdict を trusted とする。

| # | 点 | PASS 条件 |
| --- | --- | --- |
| 1 | needle normalization の有無 | needle にも N() が適用されている |
| 2 | haystack normalization の有無 | haystack にも N() が適用されている |
| 3 | case sensitivity の意図 | 意図が明示され、needle 側と一致している |
| 4 | whitespace normalization の意図 | 意図が明示され、needle 側と一致している |
| 5 | normalization が false-positive を生成しないこと | 異なる語・1 文字増・無関係句が case-insensitive 下でも MISS |
| 6 | positive self-test | 既知 PRESENT が HIT |
| 7 | negative self-test | 既知 ABSENT が MISS、既存 control が維持される |
| 8 | control case | checker 自身の needle 集合から導出した control が、その property が要求する分類を返す |

### 3.1 #8 に対する追加制約（本欠陥から導出）

**control case は checker の実際の needle 集合から機械的に導出しなければならない。
side-by-side で手書きした needle リストを使ってはならない。**

根拠:

- #3（RA-3）は手書き語彙リストが実測対象と非対応であった
- #8 は手書き self-test needle が checker の実 needle と非対応であった
- #9 は手書き診断が checker の実挙動と非対応であった

**3 件すべて、手書きの needle リストが「検査対象の器」と一致しなかった。**
self-test は「PASS する事例の集合」ではなく「**checker が実際に呼ぶ関数を、
checker が実際に使う引数で**呼ぶこと」でなければ checker の挙動を保証しない。

### 3.2 適用順序

```text
1. checker の needle 集合を 1 か所に定義する（唯一の定義元）
2. self-test / control case はその定義元から導出する（手書きしない）
3. self-test が PASS するまで verdict を出さない
4. self-test が PASS しても、self-test の needle が checker の needle と
   同一であることを別途表示する
5. 途中で self-test が FAIL した場合、Review の再判定へ進まない（Owner §3）
```

---

## 4. 本記録の範囲

```text
本記録の変更          : 新規文書 1 件の追加
amended contract     : 変更なし
Amendment 1          : 変更なし
authority / A1-B3-C1 : 変更なし
CDB executions       : 0
debuggees            : 0
images executed      : 0
source changes       : 0
build                : 0
.cdb changes         : 0
Git operations       : 0
```

本欠陥は **Open** とする。§3 の mandatory control が Owner 裁定により採用されるまで、
および以降の gate checker で §3 の 8 点が実際に充足されるまで CLOSED にしない。

---
---

# 追記 — RZ2 裁定の反映（2026-09-27）

**本追記は §1〜§4 の事実記録を書き換えない。上の 9 件の occurrence はすべて有効である。**
§3 の mandatory control は RZ2 により下記 §6 のとおり強化された。§3 の原文も削除しない。

## 5. RZ2 で採用された裁定と、その適用結果

```text
RZ2 Owner ruling accepted.

Contract Amendment          ESTABLISHED
Contract Amendment Review   NOT CLOSED
RR                          15/18 PASS, 3 FAIL        <- 昇格させない
CR-3                        PENDING
R1 execution                NOT AUTHORIZED
```

RZ2 は「15/18 を PASS に昇格させない」ことを明示した。normaliser を両端に適用した
再判定（RR-1 / RR-8 / RR-11 の解消）をもって Review を閉じない。
実行順序は §6 の gate 順序に固定された。

## 6. Mandatory control strengthened（RZ2 追記分）

```text
1. normalisation must be symmetric on needle/haystack
2. checker needle set has one authoritative definition
3. self-test derives from that definition
4. control cases derive from that definition
5. verdict is blocked until self-test passes
6. needle-set identity is independently exposed and checked
```

§1 の 3 形態表は本追記でも有効である。RZ2-1 は **RAW/N を廃止**し **N/N** を採用した。

### 6.1 gate 順序（固定）

```text
checker implementation
        ↓
normaliser self-test                       GATE 1
        ↓
needle-set identity check                  GATE 2   (mechanical, by object identity)
        ↓
control / positive / negative tests        GATE 3
        ↓
PASS
        ↓
RR-1..RR-18 evaluation
```

いずれかが FAIL の場合:

```text
Contract Amendment Review = NOT TRUSTED
RR-1..RR-18 の結果を PASS/FAIL として採用してはならない
```

## 7. RZ2 実行時に同定された新規 occurrence（3 件）

RZ2 の checker `tmp/rr_review_rz2.py`（29,577 B /
`8433186C8FAA51FDD26482B4E47E8E132CCB575234284AC8FBC8123CED48D3A7`）は
GATE 3 を FAIL させ、**RZ2-4 により verdict を BLOCK した**。RR-1..RR-18 は評価されていない。

| # | ゲート | checker の形 | 誤判定 | 段階数 |
| --- | --- | --- | --- | --- |
| 10 | Review checker（RZ2） | **hay key の誤割当** — needle が L22 にあるのに haystack を L1..L15 に指定 | positive self-test false FAIL（RR-17.THREE_SEPARATE） | 1 |
| 11 | Review checker（RZ2） | **1 文字 needle の under-determination** — `q` の 1 文字変異 `0` が haystack 内に存在 | negative self-test false FAIL（RR-10.TERMINATOR） | 1 |
| 12 | Review checker（RZ2） | **case assertion が whitespace axis を混入** — `n._n == n.text.lower()` は空白畳み込み fallacy を含むため繰り返し空白のある needle で偽 | case leg false FAIL | 1 |

### 7.1 #10 は authoring error であり contract 内容の欠落ではない

```text
needle   'this document is the FIRST of those three things.'
所在     L22、## 0. Three authorizations, kept separate 節内
割当     hay = "HEADER" = B[:15] = L1..L15
結果     N(needle) in N(HEADER) -> False
```

needle は §0 にあり、header fence にはない。**registry の hay key 誤り**であり、
contract に当該文言は存在する。Owner §1 の「amended contract は再編集しない」に照らし、
contract 側は変更せず、checker 側のみが誤りである。

### 7.2 #11 は real property であり comparator の defect ではない

```text
needle    'q'          1 文字
変異      '0'
N('0') in N(SCRIPT) -> True    (0x30010 / 0xfff / 0x34040 がいずれも '0' を含む)
```

1 文字 needle は substring presence test を支えられない。negative self-test が報告しているのは
**この needle が判別力を持ちえない**という真の性質である。
「script が q で終端する」は **位置依存の性質**であり、presence needle ではなく
**構造的計器**が必要である。これは §2 の欠陥族 3「検査対象と検査尺の非対応」の第 2 例である。

### 7.3 #12 は case axis の assertion 取り違え

```text
assertion 1..4  N(t) in N(h) / N(t.lower()) / N(t.upper()) / n._n == N(t)   -> 全て成立
assertion 5     n._n == t.lower()                                            -> 繰り返し空白で FALSE
```

`n._n` は case fold に加えて空白畳み込みも含むため、`t.lower()` とは一致しない。
assertion 5 は 2 軸を混合しており、正しい case assertion は assertion 4 である。

### 7.4 3 件とも contract 内容の欠落ではない

```text
GATE 3 の 3 件の FAIL はいずれも checker 側に帰属する。contract 側の欠落は 0 件。
GATE 3 は RZ2-3 / RZ2-4 が求めるままに機能した。自らの性質を測らない計器の上を
verdict を通さなかった。
```

## 8. 5 回の停止に関する所見（事実の記録）

本 workstream で Review verdict の成立を阻止した回数は **5 回**である。

```text
1  V-6 / V-7        領域抽出 index 誤り
2  RR-11 / RR-18    normaliser 除去（regression）
3  RR-1 / RR-8 / RR-11   RAW/N half-normalisation
4  （本次）GATE 3    hay key 誤割当 / 1 文字 needle / case assertion 混合
```

各回で「新設した gate が、直前に私が書いた checker の計器欠陥を検出した」。
**検査が自らを検査しない構造のもとでは、その gate 自身が最初の実実行で
3 件の計器欠陥を検出し、verdict を 1 度も出さずに止めた。**
この事実は §1 の欠陥族が未解決であることではなく、**substring needle を主計器とする
gate checker という測定器そのものの妥当性**に関するものである。

本記録は §1〜§4 の occurrence 1〜9 と、本追記の occurrence 10〜12 を保持する。
**RZ2 で Review は CLOSED していない。CDB は起動していない。R1 execution は未授権。**

---
---

# 追記 2 — RZ2C 裁定の反映（checker architecture / instrumentation model change）

**本追記は §1〜§4 と 追記 1（§5〜§8）の事実記録を書き換えない。occurrence #1〜#12 はすべて有効である。**

## 9. RZ2C の性質

```text
本件は individual checker bug repair ではない。
checker architecture / instrumentation model change である。
```

§1 の欠陥族は「判定器の形が、その判定器が主張する性質と一致しない」ことである。
needle を 1 本ずつ直しても、判定器の形は変わらない。判定器の形を変えた。

### 9.1 直接的な根拠として保持する 3 件

| # | 内容 | RZ2C での扱い |
| --- | --- | --- |
| **#10** | **hay key misassignment** — needle は L22 にあるのに haystack を L1..L15 に指定した | checker 側の region を section 番号で機械導出する方式に変更。needle と haystack の対応は宣言されない |
| **#11** | **positional property を 1 文字 substring needle で表現した** — `'q'` の変異 `'0'` が `0x30010` により HIT した | 位置依存の性質は型 D の計器で測定する。`q` は needle ではなく、解析済み command 列の終端要素として扱う |
| **#12** | **case assertion が whitespace normalisation を混入した** | 正規化軸を 2 軸に分離。case 軸の assertion は `n._n == N(n.text)` のみ。whitespace 軸は別の self-test Leg とする |

## 10. property type の導入（mandatory control）

計器選択の前に property type を宣言する。

| 型 | 意味 | 主計器として可否 |
| --- | --- | --- |
| **A** | semantic text-presence | **不可**（補助のみ） |
| **B** | exact / normalized phrase | **不可**（補助のみ） |
| **C** | structural（解析済み label 集合とその関係） | 可 |
| **D** | positional（解析済み command / token 位置） | 可 |
| **E** | quantitative（解析済み要素数） | 可 |
| **F** | identity / hash（解析済み 64-hex フィールド） | 可 |
| **G** | ordering / dependency（解析済み順序と順列不変性） | 可 |

```text
mandatory control（追加分、RZ2B §3）

  property type must be declared before instrument selection
  the instrument must be generated from the contract checklist/property definition
  positional or structural properties must not use substring presence as their primary instrument
  instrument coverage must be complete: every RR item has exactly one declared primary
      evidence mechanism
  secondary textual needles may support an RR item but may not replace its required
      primary evidence mechanism
```

RZ2C で実測した配分:

```text
C  5 items   RR-7  RR-11  RR-12  RR-13  RR-17
D  2 items   RR-6  RR-8
E  5 items   RR-3  RR-4  RR-10  RR-15  RR-18
F  4 items   RR-1  RR-2  RR-5  RR-16
G  2 items   RR-9  RR-14
A  0 items   主計器として使用されたものはない
B  0 items   同上
```

**substring HIT から C/D/E/F/G の verdict へ到達する経路は、コード上存在しない。**

## 11. false PASS の構造的防止

従来の self-test は false FAIL 検出を主眼としていたが、RZ2C は false PASS を先に潰す。

```text
false-positive audit
    各 instrument に、構造を型指向で違反させる mutation が存在し、
    その mutation 適用後 instrument が False になること（load-bearing であること）

false-negative audit
    無視できる差異だけを変えた第二の synthetic variant に対して
    どの instrument も反転しないこと（過適合でないこと）
```

両者は **synthetic model** 上で実行する。real contract は §18 により評価してはならないため、
`run()` は real model を受け取ると `RealModelForbidden` を送出する。
**この guard は主張ではなく、real model の clone に対して実際に raise することで検証した。**

## 12. RZ2C 実施中に同定された新規 occurrence（6 件）

RZ2C の instrument model 構築は 6 回の反復を要した。途中で同定した欠陥の記録である。

| # | 内容 | 場所 |
| --- | --- | --- |
| **#13** | 最初の `##` より前の fence が **header block** だが、`cur` が None のため **丸ごと捨てられていた** | parser |
| **#14** | items を label でキー化したため、§12 の禁止行が **すべて同じ label** で衝突し、最後の 1 件しか保持されなかった | parser |
| **#15** | `marker_coverage` が `len(marker) == len(probe)` を要求したが、script は終端 marker を含むため 5 対 4 になる | instrument（過厳） |
| **#16** | `void_condition` が **N() を適用せず**、大文字の解析済みテキストに case-sensitive で比較した | instrument（軸混同） |
| **#17** | `RR-18` の禁止項目数に **15 という手書き定数**を置いていたが、実数は 16 であった | descriptor |
| **#18** | `RR-14 permutation_invariance` の violation が **no-op** であった | violation operator |

### 12.1 #13 / #14 は「検査対象そのものの誤要約」

```text
#13  最初の `##` より前の fence（header）はいずれの section にも属さない。
     parser はこれを破棄していた。RR-17 の「NOT authorized」宣言は header にあり、
     したがって 해당 instrument は原理的に充足され得なかった。
#14  §12 の禁止行はすべて同じ label を持つ。dict は衝突し最後の 1 件だけが残った。
     禁止項目数の instrument は「1 != 15」で false した。
```

**instrument は正しく、入力の要約が壊れていた。**
§1 の欠陥族は checker だけの問題ではなく、**checker が測っている対象そのものが誤要約されている**
場合にも生じる。instrument の type 分解だけでは parser の欠陥は検出できない。
instrument test がその 2 件を検出したことが、RZ2C の設計が機能した証拠である。

### 12.2 #17 は「instrument 側の過来信」

```text
正しい instrument は「PRE が持ち越した禁止集合は何か」を測るべきである。
しかし RR-18 の descriptor は 15 という数字を checker 側が持ち込んでいた。
実数は 16 であり、15 は 1 件の取りこぼしである。
instrument を「POST の禁止集合と PRE の禁止集合の一致」に書き換え、
数字を一切持たなくなった。
```

### 12.3 #18 は violation operator 自体の欠陥

```text
旧 mutation : 「c4 == C4 -> CR-3 = PASS」を先頭に差し込む
真の.evaluate : pre4 が「どの c_i も C4 でない」を要求するため、c4 == C4 のとき
                pre は常に False。default VOID が既にこれを賄う。
                よって差し込んだ規則は到達不能で verdict map は不変。
                violation ではなく no-op である。

新 mutation : 「c4 in {C2, C3} -> CR-3 = FAIL」を先頭に差し込む
                pre が成立し c4 が C2/C3 のとき、規則の順序により FAIL / PASS が変わる。
                真の order-dependence。instrument がこれを検出した。
```

**no-op を violation として数えると、false-positive audit は空振りする。**
audit 自身にも「その mutation が実際に何かを変えるか」の検査が必要である。

## 13. RZ2C 完了状態

```text
§14 checklist -> machine extraction     PASS   18 items, RR-1..RR-18
property classification                PASS   18 descriptors, C/D/E/F/G のみ
instrument coverage                   PASS   40 instruments over 18 RR items
needle/haystack N/N                   PASS   A/B のみが到達可能、かつ主計器として不使用
structural instrument self-test        PASS   12/12
positional instrument self-test        PASS   2/2
ordering/dependency self-test          PASS   6/6
quantitative/identity self-test        PASS   20/20
positive / negative / boundary         PASS   40 / 40 / 40
false-positive audit                  PASS   load-bearing 40/40
false-negative audit                  PASS   反転 0
checker identity / hash recorded      PASS
real-model guard                      ENFORCED, clone に対して raise することを検証

手書き needle            0
手書き region slice      0
手書き hash literal      0
real-model 評価          0
```

## 14. §8 の所見の更新

§8 で 5 回の停止を記録した。RZ2C はその 5 回目の後、**新規 6 件**を同定した。
検査が出力を止めた回数は 6 に達する。

```text
ただし RZ2C は、同一の gate をさらに 1 段追加して就这样に止まったのではない。
欠陥を continue 生み出したのではなく、**欠陥を構造的に発生させなくした計器**である。
手書き needle の集合が消え、negative test は文字列変異ではなく型指向の構造 mutation になり、
false-positive audit と false-negative audit が新規に入った。
guard は主張ではなく検証された。
```

本記録は §1〜§4、追記 1（§5〜§8、occurrence 1〜12）、追記 2（§9〜§14、occurrence 13〜18）を保持する。
**Contract Amendment Review は NOT CLOSED。RR-1..RR-18 は未評価。R1 execution は未授権。**

---
---

# 追記 3 — RR 全件再導出の事前ゲート A9 失敗と occurrence #19

**本追記は §1〜§4、追記 1（§5〜§8）、追記 2（§9〜§14）の事実記録を書き換えない。
occurrence #1〜#18 はすべて有効である。**

## 15. 出来事

Owner が RR-1..RR-18 の全件再導出を授権した。着手前に Owner §6 の事前ゲート 11 項を
評価したところ、**A9（real-model guard）が FAIL** した。

```text
A9  real-model guard : armed=False  raises=True
```

`raises=True` だが `armed=False`、すなわち

```text
guard は probe clone に対して raise した。
しかし Z.REAL は保護集合に含まれていなかった。
```

RZ2C 実行時の出力は

```text
real-model guard   ENFORCED   (run() raises RealModelForbidden on the real model;
                               probe-verified on a clone: True)
```

と記していた。**この記載は誤りであった。**

## 16. occurrence #19 — guard の自己検査が、自らが守っていたものを検査していない

| # | 内容 | 場所 |
| --- | --- | --- |
| **#19** | guard の自己検査が「raise が発生した」ことしか確認せず、**「保護対象が保護集合に残っている」ことを確認していなかった**。その結果、guard 自体は実在するが守りたいものを守っていなかった | guard + self-test |

### 16.1 機序

```text
line 713  FORBIDDEN.append(REAL)
line 718  _probe = copy.deepcopy(REAL)
line 719  FORBIDDEN.append(_probe)
line 721  run(INSTS[0], _probe)        -> RealModelForbidden      <- ここで True
line 727  FORBIDDEN.remove(_probe)
```

`list.remove(x)` は **値の一致**で最初の要素を削除する。
`_probe` は `REAL` の `deepcopy` であり、`REAL["_pre_region"]` と `REAL["_pre_x"]` は
どちらも `None` のままなので、**両者は値が等しい**。

```text
remove(_probe) は FORBIDDEN[0] == REAL を削除した。
削除後に残ったのは _probe であり、REAL ではない。

FORBIDDEN == [REAL]            ... append 後
FORBIDDEN == [REAL, _probe]    ... probe append 後
FORBIDDEN == [_probe]          ... remove 後    <- REAL が保護から消えた
```

`run()` は `any(m is f for f in FORBIDDEN)` で判定するため、以後 `REAL` を渡しても
raise しない。**guard は SELF-TEST 中に自らを無効化された。**

### 16.2 なぜ self-test が 이를検出できなかったか

自己検査が問うたのは

```text
「raise が発生したか」
```

だけであった。問うべきだったのは

```text
「raise が発生し、かつ REAL が保護集合に残っているか」
```

である。**検査の形が、检查するはずの性質と一致していない。**
これは §1 の欠陥族そのものが出現した位置が、本欠陥族への修正の内部であったことを示す。

## 17. 影響範囲の境界

### 17.1 無効なもの

```text
RZ2C 出力の「real-model guard ENFORCED」という記載   -> 偽
```

### 17.2 無効でないもの（実測で確認）

```text
計器モデル            無効でない。descriptor 18 / instrument 40、両 digest が
                      Owner 受受領 artefact と一致（事前ゲート A4..A7 すべて MATCH）。

RZ2C self-test 結果    無効でない。static audit により、run() の呼び出し site は 3 つ
                      （SYN[0] / violation of SYN[0] / SYN[1]）であり、
                      REAL を渡す site は 1 つも存在しない。
                      したがって real contract が一度も計器評価の対象になっておらず、
                      self-test 結果は汚染されていない。
```

static audit の実測:

```text
line 945   run(mi, SYN[0])        model source: SYN
line 951   run(mi, vm)            model source: SYN
line 952   run(mi, SYN[1])        model source: SYN
REAL を渡す呼び出し site            0
```

## 18. 停止の根拠

```text
Owner §6   いずれかが不一致なら RR verdict を出さずに停止      -> 適用
Owner §10  計器モデルそのものの変更が必要になったら停止し Owner へ戻す  -> 適用

RR verdict                NONE（PHASE C 未到達）
Contract Amendment Review NOT TRUSTED
```

RZ2C ファイルは Owner が `717E957D…` として受領した artefact である。
guard の修正はファイル内容を変SIZE、A1 の固定 digest を無効化する。
したがって agent は修正を適用せず、裁定を求めた。

## 19. 修正案（適用していない）

```text
(1) 保護判定を object identity のみに依存させる。値比較の経路を完全に無くす。

      def run(mi, m):
          if m is REAL or any(m is f for f in FORBIDDEN):
              raise RealModelForbidden(...)

(2) probe の後始末を identity で行う。remove() を使わない。

      FORBIDDEN[:] = [x for x in FORBIDDEN if x is not _probe]

(3) self-test に第 3 の leg を追加する。これが本欠陥の直接の修正である。

      leg 1  run(probe) が raise する
      leg 2  leg 1 の後、REAL が保護集合に残っている   <- 今回欠けていた leg
      leg 3  REAL を実際に run() に渡して raise する
```

leg 3 がないと「raise した」ことだけで充足してしまう。一旦 True と判定される。
**#19 の 一般則：guard / self-test は、 자신이守っている対象が残っていることをも検査せねばならない。**

## 20. 現状

```text
RZ2C                       VERIFIED（Owner 受領済み、ただし §15 の誤記載を含む）
RR-1..RR-18                 NOT EVALUATED
Contract Amendment Review  NOT TRUSTED
RR                         15/18 PASS, 3 FAIL（変更なし・未再評価）
CR-3                       PENDING
R1 execution               NOT AUTHORIZED
CDB executions 0   debuggees 0   images executed 0   builds 0   .cdb edits 0   Git operations 0
```

本記録は §1〜§4、追記 1（§5〜§8、#1〜#12）、追記 2（§9〜§14、#13〜#18）、
追記 3（§15〜§20、#19）を保持する。

---
---

# 追記 4 — RA2: occurrence #19 の修正、Property-to-Leg mandatory control、RZ2C 再 VERIFIED

**本追記は §1〜§4、追記 1（§5〜§8）、追記 2（§9〜§14）、追記 3（§15〜§20）を書き換えない。
occurrence #1〜#19 はすべて有効である。#19 は本追記において「修正済み」と記録するが、
検出記録・原因記録は削除しない。**

## 21. occurrence #19 の閉包

### detected

```text
RR 全件再導出の事前ゲート A9 が FAIL。
    A9  real-model guard : armed=False  raises=True
RZ2C 実行時の出力「real-model guard ENFORCED」は偽であった。
```

### cause

```text
保護集合が list であり、guard の自己検査がそれを変更していた。

  list.remove(x) は値の一致で最初の一致要素を削除する。
  self-test の probe は REAL の deepcopy であり、REAL と値が等しい
  （REAL["_pre_region"] と REAL["_pre_x"] は両方 None のまま）。
  したがって remove(_probe) は FORBIDDEN[0] == REAL を削除した。

  結果: FORBIDDEN == [_probe]。REAL は保護集合から消えた。
  run() は any(m is f ...) で判定するため、以後 REAL を渡しても raise しなかった。

  検査が問うたのは「raise が発生したか」だけだった。
  問うべきだったのは「raise が発生し、かつ REAL が保護集合に残っているか」。
  検査の形が検査するはずの性質と一致していなかった。
```

### repair

```text
(1) 保護判定を object identity のみに依存。値比較の経路を完全に排除した。
      def is_protected(m, extra=()):
          for x in REAL_MODELS:
              if m is x: return True
          for x in extra:
              if m is x: return True
          return False

(2) probe を保護集合に格納しない。extra_protected 引数で渡す。
    したがって後始末処理が存在せず、復元すべき状態が存在しない。
    値一致のコピーが保護対象主意を代行することも、押し出すこともできない。

(3) self-test を leg 単位の性質検証に変更し、Leg 2（保護状態の存続）を必須化した。
    加えて Leg 4（保護対象の model が一度も評価されなかったこと）を新設した。

(4) fixture の状態保存事後条件を新設した。STEP 5-7 の前後で SYN[0]/SYN[1] の
    canonical digest を比較し、violate() が共有 fixture を変更していないことを検証する。
```

### verification

```text
実測（本体）:
    Leg 1  PASS   RealModelForbidden raised
    Leg 2  PASS   REAL still in the protected set by identity: True
    Leg 3  PASS   RealModelForbidden raised
    Leg 4  PASS   protected-model evaluations in the trace: 0
    3I     PASS   guard postcondition after the full suite
    STEP 7 postcondition  PASS   SYN[0] / SYN[1] とも UNCHANGED
    GATE 3 PASS   3A..3I  9/9
    陰性対照     4 leg すべてが #19 を検出し、RZ2C NOT VERIFIED を出力

陰性対照（#19 の再導入）:
    修正済み checker の COPY に #19 をverbatim で再導入し実行した。
      FAIL Leg 1  RealModelForbidden raised
      FAIL Leg 2  REAL still in the protected set by identity: False
      FAIL Leg 3  no exception -- the guard did not reject the real model
      FAIL Leg 4  protected-model evaluations in the trace: 0
      RZ2C NOT VERIFIED -- the section-18 boundary is not established and preserved.
    -> 修正は存在するだけでなく、#19 を検出できることを実証した。
    対照用 copy は削除済み。本体 checker は一度も変更していない。
```

### residual risk

```text
R1  Leg 2 は REAL_MODELS の要素数ではなく identity 所属を検査する。
    将来 REAL_MODELS  자체が再代入された場合、Leg 2 は新しいリストを検査して PASS する。
    緩和: 検査対象は run() が実際に参照する同一个のリストである必要がある。
    現状は単一モジュールで bind されており、この経路は未使用である。

R2  Leg 4 は EVAL_TRACE による事後的検出である。
    trace 改竄で偽装できる構造は残る。ただし trace への追記は guard 通過後にのみ
    行われるため、保護対象が評価されたなら必ず 1 件残る。

R3  descriptor digest / instrument digest は不変であることを確認した。
    ただし本修正が guard 以外の経路を意図せず変更していないことは、
    digest の一致と 3A..3G の PASS によって担保されているにすぎない。
    行レベル差分の独立レビューは Owner 裁定の領域である。
```

## 22. Mandatory Control — Property-to-Leg Correspondence（RA2 で追加）

すべての self-test leg について次を必須とする。

```text
1. leg が検証する property を明示する
2. property に対応する observable evidence を明示する
3. evidence が property を十分に検証していることを確認する
4. leg 実行後に、その property が維持される必要がある場合は
   state-preservation / postcondition を検証する
5. 「一時的に成立した事実」と「検証終了時にも成立している保証」を同一視しない
```

明示的に区別する 3 状態:

```text
    action succeeded
            !=
    property established
            !=
    property preserved after action
```

### 22.1 occurrence #19 がこの制御を要求した理由

```text
    run(probe) -> raise

は「拒否動作が一度発生した」という事実しか保証していない。

    REAL remains protected

を保証していなかった。両者は別状態であり、Leg 1 のみでは 3 番目の状態は判定できない。
```

### 22.2 guard の 4 leg への適用

| Leg | property | evidence | evidence 失敗時の verdict |
| --- | --- | --- | --- |
| 1 | probe model が real-model guard により拒否されること | `run()` が値等価 copy に対して `RealModelForbidden` を送出 | RZ2C NOT VERIFIED |
| 2 | Leg 1 の実行後も REAL が保護集合に残っていること | identity による REAL membership が成立 | RZ2C NOT VERIFIED |
| 3 | REAL 自体を `run()` に渡した場合にも拒否されること | `run(REAL)` が `RealModelForbidden` を送出 | RZ2C NOT VERIFIED |
| 4 | 保護対象の model が一度も評価されなかったこと | `EVAL_TRACE` に保護対象の id が現れない | RZ2C NOT VERIFIED |

**Leg 2 は補助 assertion ではない。guard の必須性質である。**
Leg 2 なくしては Leg 1 の一過的な成立を恒久的な保証として報告することになる。

### 22.3 fixture への適用（今回新設）

```text
property : self-test が共有 fixture を変更しない
evidence : STEP 5-7 の前後で SYN[0] / SYN[1] の canonical digest が一致する
postcond : 同一条件を suite 終了時にも再検査する（3I）
```

## 23. RZ2C 再 VERIFIED

```text
checker path            C:\VSC_Project\ConvoPeq\tmp\rz2c_instruments.py
checker size            54,212 B
checker sha256          B14350D7AF5FAED1D20B73233E6A3ED005CB6AAD1E52276EDB66C479259D1AC4
                        （修正前 717E957D232AB814A0048A8914DEF78F6FFB406D418F9ECC8A370F4035304C10）

descriptor count        18                                  UNCHANGED
descriptor digest       E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442   UNCHANGED
instrument count        40                                  UNCHANGED
instrument digest       A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4   UNCHANGED
hand-written needles    0
```

Owner §5 の再 VERIFIED 確認項目の結果:

```text
guard identity semantics             PASS   is_protected() は identity のみ、値比較なし
Leg 1                                PASS
Leg 2                                PASS
Leg 3                                PASS
property-to-leg correspondence       PASS   4 leg すべて property / evidence / verdict を明示
real-model rejection                 PASS   Leg 1 および Leg 3
real-model never evaluated           PASS   Leg 4、trace 0 件
RZ2C descriptor count                18
RZ2C instrument count                40
descriptor digest                   unchanged
instrument digest                   unchanged
```

他 step の結果:

```text
STEP 1  PASS      STEP 2  PASS     STEP 3  PASS    STEP 4  PASS
STEP 5/6/7 positive  PASS  40/40
STEP 7 negative     PASS  40/40  detected
STEP 7 boundary     PASS  40/40
STEP 7 postcondition PASS  SYN[0] / SYN[1] UNCHANGED
STEP 8  PASS  load-bearing 40/40
STEP 9  PASS  反転 0
GATE 3  PASS  3A..3I  9/9
再現性  3 回連続で checker sha256 / 両 digest / 全 leg 同一
```

## 24. 状態

```text
RZ2C                         RE-VERIFIED（Owner 再受領待ち）
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT TRUSTED
RR verdict                   NONE
RR                           15/18 PASS, 3 FAIL（旧 Review 結果としてのみ保持）
CR-3                         PENDING
R1 execution                 NOT AUTHORIZED
Implementation               FORBIDDEN
Phase 2-B                    NOT STARTED
```

本記録は §1〜§4、追記 1（#1〜#12）、追記 2（#13〜#18）、追記 3（#19）、
追記 4（#19 の閉包 + mandatory control + 再 VERIFIED）を保持する。

---
---

# 追記 5 — RB1/RB2 実施、RR-1..RR-18 全件再導出 18/18、occurrence #20

**本追記は §1〜§4、追記 1〜4 を書き換えない。occurrence #1〜#20 はすべて有効である。**

## 25. RB1 — RR-6 の計器配線を修正

### 25.1 修正前のinstrument

```text
期待 : toks[-1] == 解決済み debuggee パス（§2 の非 build パス）
実測 : toks[-1] == '<debuggee>'
結果 : FAIL
```

### 25.2 修正後の instrument（Owner 指示に基づく）

```text
property
    invocation の最終 token が debuggee 位置を占めること。
    表記は契約が定義する placeholder 形式であり、§11 が artefact filename の
    hand-transcription を禁じているため、この形式であることが正である。

evidence
    契約定義の invocation 表現における token position と placeholder semantics。
    tokens / final_slot / final_four_tokens / final_slot_is_placeholder /
    nothing_follows_final_slot を、計器内部で実際に用いた値として記録する。

verdict
    上記の構造的条件が成立する場合に限り PASS。
```

**「literal が書かれていないから PASS」という判定という判定にはしていない。**
placeholder 形式そのものを positive 条件として要求している。
literal を final slot に置く mutation は negative として 40/40 で検出された。

### 25.3 修正後の実測

```text
tokens        ["tmp\\cdb.exe", "-cf", "<script>", "-logo", "<log>", "<debuggee>"]
final_slot    "<debuggee>"
token_count   6
final_four    ["<script>", "-logo", "<log>", "<debuggee>"]
placeholder   true
nothing_follows true
verdict       HOLDS
```

## 26. RB1 — RR-18 の評価 fixture 配線を修正

```text
原因 : 評価対象 fixture に PRE 参照が供給されていなかった。
      instrument は M["_pre_region"] is not None を要求するが、
      旧 RZ2C は REAL["_pre_region"] = None を意図的に設定していた。
修正 : 評価対象データとして PRE 参照を明示的に供給する。
      _pre_region 595 B、_pre_x 16 entries。
      これはデータであって代替 model ではない。
```

### 26.1 §18 の real-model boundary は維持された

```text
Leg 1  PASS   値等価 copy は拒否される
Leg 2  PASS   Leg 1 の後も REAL は保護集合に残る
Leg 3  PASS   REAL 自身は拒否される
Leg 4  PASS   self-test 段階で保護対象の model は一度も評価されていない
3I     PASS   suite 終了後も postcondition が維持される
```

`REAL` に PRE 参照という**データ**が加わったが、guard の登録は 1 件（`REAL` 自身）のままであり、
self-test は synthetic fixture のみを評価する。real contract を self-test に持ち込んでいない。

### 26.2 修正後の実測

```text
prohibition_count    pre_count 16  post_count 16  multiset_equal true
byte_identity_to_pre pre_supplied true  pre_non_empty true
                     pre_bytes 595  post_bytes 595  identical true
```

## 27. RB2 — evidence traceability

### 27.1 構造

```text
instrument body  -->  ev("target"|"expected"|"actual", value)  -->  EV レコード
run()            -->  同じ evaluation から verdict を返す
reporting layer  -->  Z.records_for(rr, key) を読むだけ。再計算しない
```

Owner §5「instrument result と evidence record を同じ evaluation result から生成し、
表示側だけが別の値を再計算する構造を避ける」を満たす。

### 27.2 evidence completeness

```text
40 / 40 instruments が target / expected / actual を保持
self-test 段階で 360 records、RR 段階で 120 records
不完全 instrument  0
```

これにより、前回報告した 14 件の粗い観測表示は解消した。
前回「section 7 body: 41 lines」としか出なかった計器は、
現在は `c1_text` / `parser_accepted` / `evaluation_failed` 等の構造的事実を返す。

## 28. 計器モデルの不変性（Owner §6 の二層 identity）

```text
model identity        descriptor count 18   E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442  不変
                      instrument count 40   A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4  不変
implementation id     checker sha256        968377BEA8A6FE6A007EDF0DD18444931AB5C3943207CDD594459D9293667A47  新規
                      (修正前 B14350D7AF5FAED1D20B73233E6A3ED005CB6AAD1E52276EDB66C479259D1AC4)
```

checker 内部に digest 固定の `assert` を置き、model digest が動いた時点で停止する設計とした。
 실제로この assert は作動し、KEYS を暗黙の集合から明示 40 鍵の集合へ戻す強制修正 /
 instrument count が 42 になっていた時点で INSTRUMENT DIGEST MOVED を出力して停止した。
 この.False positive は model を改変しないことへの防護として記録する。

## 29. occurrence #20 — 報告ラベルが報告対象の性质と一致しない

| # | 内容 | 場所 |
| --- | --- | --- |
| **#20** | 出力行 `protected-model evaluations, self-test : 40` の**ラベルが誤り**。実際は Owner 授権下の RR 段階 40 回の real-model 評価であり、self-test 段階の値は 0（checker 内部の Leg 4 と GATE 3I が保証） | driver の報告ラベル |

```text
driver は RR 段階の 40 dispatch を Z.EVAL_TRACE に追記する。
その直後に同じ evaluated_protected_model() を呼ぶと 40 が返る。
印字ラベルはこれを「self-test阶段的」と書いたため、誤り。

正しい値
    self-test 段階の保護対象評価数   0    （checker 内部で検証済み）
    RR 段階の real-model 評価数   40    （Owner 授権下、run_real 経由）
    guard の armed 状態            True （REAL_MODELS は 1 件、run_real は arm を緩めない）
```

**#1〜#19 と同じ欠陥族（判定器の形が主張する性質と一致しない）の 20 番目。**
今回は判定値には影響せず、印字ラベルだけの問題である。R3 の行レベル review 対象にも含める。

## 30. RR-1..RR-18 全件再導出の結果

```text
pre-verification 1..15   15 / 15 PASS
evidence completeness   40 / 40
RR-1..RR-18             18 / 18 PASS
```

前回 16/18 の内訳は引き継がず、18 件すべてを新規に導出した。

```text
前回 FAIL した RR-6 / RR-18 がいずれも
    contract property -> descriptor -> primary mechanism -> instrument
    -> observed evidence -> verdict
の連鎖が接続された状態で PASS した。
```

**ただし本記録は Contract Amendment Review を CLOSED と判定しない。**
RB3（instrumentation defect であったか否かの最終分類）は Owner が保留している。

## 31. 状態

```text
RZ2C                         VERIFIED（RB1/RB2 版、Owner 再受領待ち）
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT CLOSED   RR 18/18 PASS   <- 判定は Owner に委ねる
RR verdict                   18 / 18（全件再導出。16/18 は引き継がれていない）
CR-3                         PENDING
R1 execution                 NOT AUTHORIZED
Implementation               FORBIDDEN
Phase 2-B                    NOT STARTED
```

本記録は §1〜§4、追記 1（#1〜#12）、追記 2（#13〜#18）、追記 3（#19）、
追記 4（#19 閉包）、追記 5（#20、RB1/RB2、18/18）を保持する。

---
---

# 追記 6 — R3 独立レビュー。occurrence #20 の独立確認と occurrence #21

**本追記は §1〜§4、追記 1〜5 を書き換えない。occurrence #1〜#21 はすべて有効である。**

## 32. R3 の位置づけ

```text
R3 は review であり repair ではない。
review 対象は 2 ファイル（RB1/RB2 checker と RR driver）で、どちらも変更していない。
結論は checker の印字から採らない。ast によるソース構造と first-principles な
数値再構成から導く。偽実化は使い捨て COPY 上で行い reviewer が削除する。
```

## 33. occurrence #20 の独立確認（R3-2）— 値は正しかった

```text
checker を単独で import（self-test 段階のみ）
    EVAL_TRACE entries            120   synthetic model での評価
    protected-model evaluations     0

RR phase を first principles で再現
    protected-model evaluations    40
    self-test 由来                  0
    RR phase 由来                  40
    RR verdict                 18 / 18
```

**「40」は RR phase 由来のみ。self-test の verdict 計算へ混入していない。**
ラベルの文言だけが誤りであり、値は正しかった。

ただし §34 のとおり構造的分離が は未充足である。

## 34. occurrence #20 の構造的判定（R3-1）— FAIL

```text
checker 側
  L855  <module>  EVAL_TRACE = []
  L879  evaluated_protected_model  return [t for t in EVAL_TRACE if t[0] in prot]
  run() が EVAL_TRACE.append を行う

driver 側
  L138  <module>  Z.EVAL_TRACE[:] = [t for t in Z.EVAL_TRACE]   no-op
  run_real() が Z.EVAL_TRACE.append を行う

containers            同一 list を共有
logical separation    呼び出し箇所による分離                あり
data separation       phase ごとの独立 storage               なし
```

**Owner R3-1 は論理的・データ構造的な双方の区別を要求する。**
論理的には区別できるが、**構造的には区別できない。**
phase を誤って混入しても判別する手段が無い。

補強材料（R3-6c）。使い捨て COPY に

```text
Z.EVAL_TRACE に 999 件の毒入れ
Z.EV        に  99 件の毒入れ
```

を行い、verdict は 18/18 のままだった。これは「現在の verdict が trace に依存しない」ことを
示すと同時に、**共有容器では誤った trace と正しい trace を区別できない**ことも示す。
R3-1 の FAIL を補強する材料である。

## 35. verdict independence（R3-3 / R3-4）— PASS

```text
driver 内の verdict 代入は 1 箇所 L151  VERD[cid] = ok
verdict 式が読む名前は ['ok'] のみ。
EVAL_TRACE / TRACE / RECHASH / records_for / EV はいずれも読まれていない。

run_real call site    L[149]
records_for call site L[152, 190]     すべて run_real より後

i_* body 内の ev() call site  120
run() は dispatch の戻り値を verdict とする
consumer は印字、または FIELD NAME のみを比較
```

**reporting label -> verdict の逆流なし。evidence -> verdict の逆流なし。**

## 36. 偽実化 control（R3-6）— PASS

```text
無改変 driver                                18/18
誤ラベル文言のみ変更                          18/18
trace と evidence を毒入れ                    18/18
```

**ラベル・trace・evidence のいずれを弄しても verdict は不変。**

## 37. execution identity（R3-5）— PASS

```text
R1 probe execution
    cdb を debuggee に対して R1 script と共に起動すること。      count 0

RR phase の run_real
    契約の parsed model に対する Python predicate の評価。
    cdb を起動しない / debuggee を起動しない / image を実行しない /
    probe expression を評価しない / process を生成しない。
```

Owner §7 の要求により、両者を同一視しない execution identity として記録した。
**R1 execution authorization は本結果によって発生しない。**

## 38. occurrence #21 — reviewer 自身の計器欠陥

| # | 内容 | 場所 |
| --- | --- | --- |
| **#21** | R3-5b が **live 関数オブジェクトに `ast.FunctionDef` 型判定を適用**し、dispatch target 6 個全てを impure と誤判定した。検査対象は AST 定義であるべきところ Live object を検査していた | R3 reviewer |

```text
誤った検査 : all(isinstance(v, ast.FunctionDef) and ... for v in Z.DISPATCH.values())
実際の対象 : CT.body 内の i_* FunctionDef.ast
誤判定     : R3-5b FAIL (6 targets)
訂正後     : i_F / i_E / i_D / i_C / i_G / i_A すべて process 名 0 -> PASS
```

**#1〜#20 と同じ欠陥族の 21 番目。**
自己申告を排除する目的で静的解析を用いたが、その静的解析にも同じ欠陥が混入した。
reviewer に exceptions はない。

## 39. R3 総合

```text
R3 checks  17 / 18 PASS
FAIL        R3-1  （genuine：構造的 trace 分離が未充足）

Contract Amendment Review = NOT TRUSTED.   RR verdict = NONE / HOLD.
review 対象 2 ファイルは変更していない。
```

## 40. 状態

```text
RZ2C                         VERIFIED / RB1-RB2
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT TRUSTED
RR-1..RR-18                 18/18 PASS（Owner 受領済み、R3 STOP 中は HOLD）
RB1 / RB2                    ACCEPTED
RB3                          DEFERRED
occurrence #20               OPEN  値は独立確認済み、構造的分離が未充足
occurrence #21               OPEN  reviewer 自身の偽 FAIL、訂正済み・記録
CR-3                         PENDING
R1 execution                NOT AUTHORIZED
Implementation              FORBIDDEN
Phase 2-B                   NOT STARTED
```

本記録は §1〜§4、追記 1（#1〜#12）、追記 2（#13〜#18）、追記 3（#19）、
追記 4（#19 閉包）、追記 5（#20、RB1/RB2、18/18）、追記 6（#20 独立確認、#21、R3）を保持する。

---
---

# 追記 7 — R3-1 Trace Separation Repair Contract とその Contract Review

**本追記は §1〜§4、追記 1〜6 を書き換えない。occurrence #1〜#22 はすべて有効である。**

## 41. 成果物

```text
CONTRACT  doc/work113/P1-5-IR-P2_R3-1_Trace_Separation_Repair_CONTRACT.md
          13,139 B  28805836D44165ECD0E8C13772654214361E10F047D146C6A90DE7CA14A8B53B
REVIEW    doc/work113/P1-5-IR-P2_R3-1_Trace_Separation_CONTRACT_REVIEW_RESULT.md
          11,428 B  E9D1B5138A4A026492DE9D727E3DA93930E7FF530A48F771D4CABF61AA35D9AD
```

Owner 指示の遵守:

```text
実施したもの   Repair Contract の作成、Contract Review のみ
実施していない  trace 分離の実装、checker 変更、RR 再評価、R1 execution
契約の編集     0（Review 前後で digest 同一）
```

## 42. 契約が明文化した内容

```text
A  independent storage      2 つの phase 所有 container、distinct object、
                             alias と第三 container を禁止、phase tag を要求
B  independent producer     producer は container 引数を取れず、foreign tag を拒否
C  independent consumer     各 consumer は 1 containerのみ、報告は phase を明示
D  object identity          identity assertion のみ。name / length / content /
                             digest は代理指標として排除。transition ごとに再実行
E  cross-contamination      両方向、raise による検出。verdict invariance と
                             count invariance は根拠として禁止
F  lifecycle                OPEN/CLOSED、6 段階の順序、close 後 write は raise、
                             self-test container が RR container より先に close
G  no verdict dependence    verdict / gate は trace を読まない。
                             G は A〜F と独立
```

## 43. Contract Review の結果

```text
checklist     R3-1-P1 .. P8   Owner 由来，逐語撥
GATE          PASS   negative control 10 件、boundary 1 件、positive 1 件
P1 .. P8      8 / 8 PASS
ADV-1 .. 8    8 / 8 PASS
```

ADV は契約に都合の悪い文言が欠落していないことを検査する対抗項である。

```text
ADV-1  verdict invariance は分離の根拠として拒否される
ADV-2  G は A〜F と独立
ADV-3  契約は自らの受理基準を定義することを拒否
ADV-4  clause group A〜G すべてに falsifier がある      7/7
ADV-5  Owner's checklist を名指し
ADV-6  実装順序を固定し、その前で停止
ADV-7  shared / aliased container を禁じる
ADV-8  count による汚染検出を禁じる
```

## 44. occurrence #22 — Contract Review reviewer の計器欠陥 4 件

| # | 内容 |
| --- | --- |
| **#22** | Contract Review の reviewer が 4 件の計器欠陥を抱えて GATE に出した |

```text
(1) parser が fence 内を不透明扱いし、契約の 36 clause を 1 件も読めなかった
(2) 節抽出の lookahead に後続節しか無く、最終節 §13 を抽出できなかった
(3) P3 の needle の動詞形が契約と不一致（names / name）
(4) P5 の verdict-invariance needle が group 単位で張られ、E.3 と衝突。
    negative control 10 件中 1 件が NOT DETECTED。
    併せて fixture 2 件も不正だった
```

**4 件すべて計器側の欠陥。契約は一度も編集していない。**
**#1〜#21 と同じ欠陥族の 22 番目。**
developer の GATE が 4 件を検出したが、検出できるまでに 4 回の書き直しを要した。

### 44.1 累積

```text
Review verdict の成立を阻止した回数        9 回
  1  V-6 / V-7        領域抽出 index 誤り
  2  RR-11 / RR-18    normaliser 除去（regression）
  3  RR-1 / RR-8 / RR-11   RAW/N half-normalisation
  4  GATE 3           hay key 誤割当 / 1 文字 needle / case assertion 混合
  5  RZ2C 構築 6 件   parser / instrument / descriptor / violation operator
  6  事前ゲート A9    guard が REAL を守っていなかった（#19）
  7  RR-6 / RR-18     計器配線 2 件
  8  R3-1             構造的 trace 分離（genuine、未修正）
  9  Contract Review  reviewer 計器 4 件（#22、本追記で訂正済み）

うち genuine な未解決は 1 件のみ: R3-1（trace 構造分離）
うち checker 側の欠陥   8 件
```

**8 回の停止はすべて「判定器の形が検査対象の形式と一致しない」ことの帰結である。**
欠陥族は未解決であり、検出率も収束していない。

## 45. この Review の限界

```text
L1  契約と Review を同一 agent が書いた。P1..P8 は conformance check であり証拠力は弱い。
    対立的に構築されているのは ADV-1..8 と negative control 10 件のみ。
L2  契約 section 2 の用語の二重使用は機械化していない。checked として主張しない。
L3  本 Review は実装可能性と自己検査可能性を評価したものであり、実装の正当性は
    評価していない。
L4  本 Review はいかなるファイルも変更していない。
```

## 46. 状態

```text
R3-1 Repair Contract      DRAFTED
R3-1 Contract Review      PASS  (P1..P8 8/8, ADV 8/8, GATE PASS)
R3-1 implementation       NOT AUTHORISED
Contract Amendment Review NOT CLOSED / NOT TRUSTED
RR-1..RR-18              18/18（Owner 受領済み、HOLD 中）
RB3                       DEFERRED
occurrence #20            OPEN
occurrence #21            OPEN
occurrence #22            OPEN  reviewer 計器 4 件、訂正済み・記録
CR-3                      PENDING
R1 execution             NOT AUTHORIZED
```

本記録は §1〜§4、追記 1（#1〜#12）、追記 2（#13〜#18）、追記 3（#19）、
追記 4（#19 閉包）、追記 5（#20、RB1/RB2、18/18）、追記 6（#20 独立確認、#21、R3）、
追記 7（#22、Repair Contract、Contract Review）を保持する。

---
---

# 追記 8 — occurrence #23: provenance / artifact-retention process defect

**本追記は §1〜§4、追記 1〜7 を書き換えない。occurrence #1〜#22 はすべて有効である。
#23 は #1〜#22 とは別系統として記録する。**

## 46.1 occurrence 台帳

| # | 系統 | 状態 |
| --- | --- | --- |
| **#23** | provenance / artifact-retention process defect（#1〜#22 とは別系統） | OPEN / UNRECOVERED |

## 47. 分類

`	ext
provenance / artifact-retention process defect
checker defect ではない
#1〜#22（判定器の形と検査対象の形の不一致）とは別系統
`

## 48. 事実関係

`	ext
Owner-authorized baseline が存在した
        968377BEA8A6FE6A007EDF0DD18444931AB5C3943207CDD594459D9293667A47
        Owner が受領し、R3-1 実装の基準として固定した

baseline identity は報告された
        Contract Review 記録、RR 再導出記録、R3 reviewer の pin 値はすべて 968377BE

implementation が artefact を変更した
        tmp\rz2c_instruments.py を R3-1 実装で上書きした
        実装後 identity  ECCB4396210ED103C79B449EE9EE7EF2BD4775206E618BAFABF6BB698BA4C195

pre-implementation artefact が保持されなかった
        上書き前に byte-exact copy を残さなかった
        残る tmp\rz2c_instruments.PRE-RA2.bak は 717E957D... であり 968377BE ではない

正確な baseline-to-after diff が不可能になった
        968377BE -> ECCB4396 の厳密な差分が存在しない
`

## 49. 禁止された扱い

`	ext
717E957D を 968377BE の代用品として扱う            禁止
717E957D -> ECCB4396 の diff を baseline diff とする   禁止
推測による reconstruction                            禁止
「実質的に同じだった」という扱い                    禁止
#23 の削除・軽減・ #1〜#22 への統合                 禁止
`

## 50. provenance recovery gate の結果

read-only 探索。既存データのみ。新規 baseline なし。reconstruction なし。

`	ext
探索範囲                                        候補      結果
repository 全ファイル 8 KiB..4 MiB                38242    exact 0
repository copy-like 名                            15    exact 0
opencode temp                                       33    exact 0
user temp root                                   17689    exact 0
recycle bin                                     読取不可    -
git object store  blob 0（tmp/ は .gitignore 対象）    0    exact 0
harness session store（opencode.db / log / WAL）      6    exact 0
    6 件は target digest を TEXT として含む参照であり、digest 自体が artefact ではない
`

`	ext
自前の bytes が target と一致する file : 0 件
CASE A  byte-exact baseline recovered   : NO
CASE B  OPEN / UNRECOVERED              : YES
968377BE -> ECCB4396 exact diff         : NOT PROVEN
`

### 50.1 残る経路は reconstruction であり、Owner が禁じている

```text
実測 : 717E957D に文字列 KEYS = set([ は無く、実装後ファイル ECCB4396 にはある。
      tmp/patch_rz2c_1.py には KEYS の文字列が一切無い。
      つまり 717E957D から 968377BE への編集は、ファイル化されていない
      inline shell コマンドとして実行されたものであり、磁盘上のいかなる
      patch script にも記録されていない。

968377BE を再び作る手段は、
  (a) その inline 編集を shell history あるいは session log から取り出し再実行する
  (b) 差分を推測して書き戻す
のいずれかであり、どちらも reconstruction である。

Owner は reconstruction を明示的に禁じている。
実施していない。試していない。試行の意図もない。
```

### 50.2 本記録自体の訂正

本追記の初稿は「両状態の差は記録された編集（patch_rz2c_1.py）」と書いた。
**これは誤りである。** 実測により patch_rz2c_1.py に KEYS が無いことを確認した。
訂正後の記述が上記の 50.1 である。訂正の事実そのものも記録する。

この追記の初稿はさらに 1 箇所の fence が shell heredoc のエスケープで
` + backslash + `text` および単一バッククォートに壊れていた。上記 50.1 の
差替えと同時に正しい fence へ戻した。破損は追記 8 の範囲内だけに残っていた。

## 51. 影響

`	ext
R3-1 実装そのものの clause A〜G 適合性     別途 Post-Implementation Review で検証可能
Owner 受領 baseline からの正確な diff      NOT PROVEN（回復不能）
「変更が契約の 2 つの条項のみに限られた」ことの証明   NOT PROVEN
model digest 不変（descriptor 18 / instrument 40）    検証済み、これは別系統の証拠
`

model digest が不変であることは実測した。ただしそれは model が不変であることを示すものであり、
implementation ファイルの変更範囲が小さかったことを示すものではない。

## 52. 状態

`	ext
#20  OPEN   構造的 trace separation（R3-1 実装済み、Post-Implementation Review 未実施）
#21  RECORDED / CORRECTED
#22  RECORDED / CORRECTED
#23  OPEN / UNRECOVERED   provenance / artifact-retention process defect
R3-1 implementation                     DONE   ECCB4396...
R3-1 Post-Implementation Review          NOT STARTED
Contract Amendment Review                NOT CLOSED / NOT TRUSTED
RR-1..RR-18                             18/18 HOLD（再利用しない）
RB3                                      DEFERRED
CR-3                                     PENDING
R1 execution                             NOT AUTHORIZED
`

本記録は §1〜§4、追記 1〜7、追記 8（#23）を保持する。
---

# 追記 9 — R3-1 Post-Implementation Review で発見した Reviewer 自身の instrument defect

本追記は §1〜§4、追記 1〜8 を書き換えない。occurrence #1〜#23 の状態は変更しない。

これらは #1〜#22 と同系統の

`	ext
判定器の形が、それが検査すると主張している検査対象の形と合っていない
`

という欠陥の新しい实例である。**#23 とは別系統である。**（#23 は provenance /
artifact-retention process defect であり、本追記とは性質が異なる。）

番号は Owner の決定事項である。本追記は new occurrence として番号を付与しない。

## 53. 発生件数

`	ext
誤った verdict を出した instrument   9 件
    うち vacuous PASS（空集合に対して OK を返した）  1 件  = C.4
    うち false FAIL                                  8 件
    = A.3b, A.3a(第一版), A.4, C.1, D.3, F.2, F.5, END-3
実行を停止させた欠陥                  6 件
    = クラス内 def の探索不備, identity_holds の arity 誤り,
      clause F の probe 順序, falsifier plant の anchor,
      retention helper の再生成漏れ, G.2b の対象誤り
self-audit が自分の source を捕まえた件  2 箇所
    = SA-2、28 文字のリテラルを raw text に in で当てている（#22 の形）
`

## 54. 最も重要だった 1 件

`	ext
C.4  count を報告する行を探す instrument が print を行単位で判定していた。
     実際の print は
         print("  self-test records %d   RR records %d   ..."
               % (self_test_record_count(), rr_record_count()))
     と複数行に及应用されており、count 呼び出しは次の行にあった。
     結果 0 行が見つかり、phase 名なしの報告も 0 件，于是 OK を返した。

     つまり空集合に対する vacuous PASS であった。
     Owner's rule「absent string は FAIL」 に正面から反する形である。

修正: print call を AST で取り、count 関数を参照する print が 1 件以上存在することを
      必須条件にした。Phase 名の不在依然是 FAIL。
`

## 55. 8 件の false FAIL の原因

`	ext
すべて「instrument の形が検査対象の形と合っていない」ことによる。
いずれも 対象 artefact の欠陥ではなく 判定器の誤りであり、
誤った FAIL は 1 件も 結果に残していない（訂正後に再実行した）。

A.3b   container 名を「文言として含む」だけで違反とした。文字列のタプルもラベルも
       報告用 alias も該当した。何も bind していない。 → A.3a に置換
A.3a   「2 つ以上の phase 関数が到達する global」を違反とした。所有者オブジェクト LIFE は
       list を持たず経路を開かない。 → record を受け取れるか、container を bind するかを問う
A.4    self.phase = phase の形だけを想定した。SealedTrace は tuple unpack を使う。
C.1    「両 container を名指す関数」を全部調べた。consumer は record から値が導出される関数
       であり、identity_holds は record を読まない。 → .records 読者に限定
D.3    counter が [0,1,2,3,4,5] と期待した。assert の中で増えるので実際は [1,2,3,4,5]。
       性质は満たされていた。期待値が誤っていた。
F.2    拒否された probe の後に state を記録し INIT が重複した。
F.5    close の「前」に sealed 読取を評価していた。
END-3  現在の pristine を「pre-change 版」として数えていた。構成上それは pre-change ではない。
`

## 56. #23 の再発防止が実際に発動した

`	ext
Owner §7 は「reviewer 自身を変更する場合、変更前 artefact を byte-exact に保持せよ」と定めた。
この control は reviewer 自身に実装し、4 回発動した。

r3_1_post_impl_review.v1.PRISTINE.py   88444953B96029AC...   v2 の存在前に v1 を保存
r3_1_post_impl_review.PRISTINE.1.py     9D1836F3777FB7BD...   最初の v2
r3_1_post_impl_review.PRISTINE.2.py     41BE92B0B3951B42...   第 1 回の instrument 修正後
r3_1_post_impl_review.PRISTINE.3.py     59C302863CAFE95E...   第 2 回の修正後
r3_1_post_impl_review.PRISTINE.py       BF538B69F09D0BE3...   本結果を生んだ版

上書きされた版は 1 件も無い。#23 の原因となった事象は再発していない。
`

## 57. 状態

`	ext
#1〜#22   変更なし
#23       OPEN / UNRECOVERED   変更なし
本追記    #1〜#22 と同系統、#23 とは別系統   番号は Owner 決定待ち
`

詳細 : doc/work113/P1-5-IR-P2_R3-1_POST-IMPLEMENTATION_REVIEW_RESULT.md §6
