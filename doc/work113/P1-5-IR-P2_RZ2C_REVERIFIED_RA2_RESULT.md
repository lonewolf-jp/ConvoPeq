# RZ2C 再 VERIFIED — RA2 実施記録

Status: **RZ2C RE-VERIFIED**（Owner 再受領待ち）
Contract Amendment Review: **NOT TRUSTED**
RR verdict: **NONE — 評価していない**
Owner ruling: RA2 + RA3
CDB executions: **0**

---

## 1. 状態

```text
RZ2C                         RE-VERIFIED  ->  Owner 再受領待ち
Contract Amendment           ESTABLISHED
Contract Amendment Review    NOT TRUSTED
RR verdict                   NONE
RR                           15/18 PASS, 3 FAIL   旧 Review 結果としてのみ保持
CR-3                         PENDING
R1 execution                 NOT AUTHORIZED
Implementation               FORBIDDEN
Phase 2-B                    NOT STARTED
```

---

## 2. 修正版 checker

```text
path              C:\VSC_Project\ConvoPeq\tmp\rz2c_instruments.py
size              54,212 B
sha256            B14350D7AF5FAED1D20B73233E6A3ED005CB6AAD1E52276EDB66C479259D1AC4
pre-RA2 sha256    717E957D232AB814A0048A8914DEF78F6FFB406D418F9ECC8A370F4035304C10
backup            tmp\rz2c_instruments.PRE-RA2.bak
```

### 2.1 計器モデル — 変更されていない

```text
descriptor count   18    UNCHANGED
descriptor digest  E6E758784020CE82F7408C387F1D5CDB0AAEF0A26D142E8F382E88B3EC63F442   UNCHANGED
instrument count   40    UNCHANGED
instrument digest  A5D93923D73B5E4226A146ECB66B3612E076D67B8868811F9A30DC2CA2345ED4   UNCHANGED
hand-written needles   0
```

Owner §4 で禁止された項目（RR descriptor の再設計、property type の変更、
instrument type の変更、主計器変更、needle / region / hash の復活、
substring-based primary instrument の復活）のいずれも**変更していない**。
両 digest が一致することで機械的に確認した。

### 2.2 変更した 3 つ（Owner 承認範囲と一致）

```text
1  real-model guard
2  self-test completeness
3  property-to-leg correspondence
```

---

## 3. occurrence #19 の閉包

### 3.1 detected

```text
RR 全件再導出の事前ゲート A9 が FAIL。
  A9  real-model guard : armed=False  raises=True
RZ2C 出力「real-model guard ENFORCED」は偽であった。
```

### 3.2 cause

```text
保護集合が list で、guard の自己検査がそれを変更していた。
list.remove() は値比較で最初の一致を削除する。
probe は REAL の deepcopy で REAL と値が等しいため、
remove(_probe) は FORBIDDEN[0] == REAL を削除した。

FORBIDDEN == [REAL] -> [REAL, _probe] -> [_probe]
REAL が保護集合から消えた。

自己検査が問うたのは「raise が発生したか」だけだった。
問うべきだったのは「raise が発生し、かつ REAL が保護集合に残っているか」。
```

### 3.3 repair

```text
(1) 保護判定を object identity のみに依存。値比較の経路を排除。
      is_protected(m, extra) は `is` のみを用いる。

(2) probe を保護集合に格納しない。extra_protected 引数で渡す。
    後始末処理が存在せず、復元すべき状態が存在しない。
    値一致のコピーが保護対象を代行することも押し出すこともできない。

(3) self-test を leg 単位の性質検証へ変更。Leg 2 を必須化、Leg 4 を新設。

(4) fixture 状態保存事後条件を新設。
    STEP 5-7 の前後で SYN[0] / SYN[1] の canonical digest を比較。
```

### 3.4 verification — 本体実測

```text
Leg 1  PASS   the probe model is rejected by the real-model guard
             evidence : run() raises RealModelForbidden on a value-equal copy
             observed : RealModelForbidden raised
Leg 2  PASS   REAL remains protected AFTER Leg 1 has run
             evidence : identity membership of REAL still holds
             observed : REAL still in the protected set by identity: True
Leg 3  PASS   REAL itself is rejected by run()
             evidence : run(REAL) raises RealModelForbidden
             observed : RealModelForbidden raised
Leg 4  PASS   no protected model was ever evaluated
             evidence : no EVAL_TRACE entry carries a protected model id
             observed : protected-model evaluations in the trace: 0

3I     PASS   guard postcondition after the full self-test suite
S7p    PASS   fixture postcondition: SYN[0] UNCHANGED, SYN[1] UNCHANGED
GATE 3 PASS  3A..3I  9/9
```

### 3.5 verification — 陰性対照（#19 を再導入）

```text
修正版 checker の COPY に #19 を verbatim で再導入し実行。

  FAIL Leg 1  RealModelForbidden raised
  FAIL Leg 2  REAL still in the protected set by identity: False
  FAIL Leg 3  no exception -- the guard did not reject the real model
  FAIL Leg 4  protected-model evaluations in the trace: 0
  RZ2C NOT VERIFIED -- the section-18 boundary is not established and preserved.

  legs that reported FAIL : Leg 1, Leg 2, Leg 3, Leg 4
  NEGATIVE CONTROL       : PASS
```

**修正は存在するだけでなく、#19 を検出できることを実証した。**
対照用 copy は削除済み。**本体 checker は陰性対照を通じて一度も変更していない。**

---

## 4. Mandatory Control — Property-to-Leg Correspondence

全 self-test leg に以下を適用した。

```text
1. leg が検証する property を明示する
2. property に対応する observable evidence を明示する
3. evidence が property を十分に検証していることを確認する
4. 実行後に property の維持が必要なら state-preservation / postcondition を検証する
5. 「一時的に成立した事実」と「検証終了時にも成立している保証」を同一視しない
```

```text
    action succeeded
            !=
    property established
            !=
    property preserved after action
```

### 4.1 guard への適用

| Leg | property | evidence | evidence 失敗時 |
| --- | --- | --- | --- |
| 1 | probe が guard に拒否されること | `run()` が値等価 copy に `RealModelForbidden` | RZ2C NOT VERIFIED |
| 2 | Leg 1 の後も REAL が保護集合に残ること | identity による REAL membership | RZ2C NOT VERIFIED |
| 3 | REAL 自身が `run()` に拒否されること | `run(REAL)` が `RealModelForbidden` | RZ2C NOT VERIFIED |
| 4 | 保護対象の model が一度も評価されなかったこと | trace に保護対象 id がない | RZ2C NOT VERIFIED |

**Leg 2 は補助 assertion ではない。** Leg 2 なくしては Leg 1 の一過的成立を
恒久的保証として報告することになる。

### 4.2 fixture への適用（今回新設）

```text
property   : self-test が共有 fixture を変更しない
evidence   : STEP 5-7 の前後で SYN[0] / SYN[1] の canonical digest が一致する
postcond   : suite 終了時にも同一条件を再検査する（GATE 3I）
```

---

## 5. Owner §5 の再 VERIFIED 確認項目

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

## 6. 他 step の結果

```text
STEP 1  PASS   §14 抽出 18/18
STEP 2  PASS   18 descriptors、C/D/E/F/G のみ
STEP 3  PASS   40 instruments / 18 RR
STEP 4  PASS   N/N 5/5、A/B は主計器として到達不能
STEP 5/6/7 positive   PASS  40/40
STEP 7  negative      PASS  40/40  detected
STEP 7  boundary      PASS  40/40
STEP 7  postcondition PASS  SYN[0] / SYN[1] UNCHANGED
STEP 8  PASS  load-bearing 40/40
STEP 9  PASS  反転 0
GATE 3  PASS  3A..3I  9/9
再現性  3 回連続で checker sha256 / 両 digest / 全 leg 同一
```

## 7. residual risk（ Owner が求める記載）

```text
R1  Leg 2 は要素数ではなく identity 所属を検査する。将来 REAL_MODELS 自体が
    再代入された場合、Leg 2 は新リストを検査して PASS する。現状は単一 module で
    bind されておりこの経路は未使用。

R2  Leg 4 は EVAL_TRACE による事後検出であり、trace 改竄で偽装できる構造は残る。
    ただし trace への追記は guard 通過後にのみ行われるため、保護対象が評価されたなら
    必ず 1 件残る。

R3  descriptor / instrument digest の一致と 3A..3G の PASS は、本修正が guard 以外の
    経路を意図せず変更していないことを担保するにとどまる。行レベル差分の独立レビューは
    Owner 裁定の領域である。
```

---

## 8. 禁止事項の遵守

```text
CDB launch                 0        source modification       0
debuggee launch            0        build                     0
g / go / gc                0        production runtime        0
R1 probe execution         0        .cdb modification         0
Git operation              0        Contract Amendment        0
RR-1..RR-18 evaluation     0        Amendment 1 / ConvoPeq.md 0
```

amended contract / Amendment 1 / §15 / G-1..G-7 / AB-1..AB-8 / C1..C4 / DF-1..DF-3 /
Step5EI / T0P / T0Q / probe script / probe order / probe markers / CDB identity /
debuggee identity / SHAPE A-B / invocation / artifact rules / no-retry rule /
authority・A1-B3-C1 / `.cdb` corpus 16 / Release harness ——すべて不変。

**RR-1..RR-18 の評価は実施していない。** 順序は Owner §7 のとおり
`#19 修正 → property-to-leg control → RZ2C 再 VERIFIED → STOP → Owner 判断` を守った。

## 9. 次の Owner 判断

```text
次の判断対象は「RR-1..RR-18 全件再導出・再評価」である。
本承認は RR 再導出の再開許可ではない。
Owner の判断なしには RR 評価へ進まない。
```
