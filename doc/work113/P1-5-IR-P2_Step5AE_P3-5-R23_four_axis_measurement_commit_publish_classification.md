# P1-5-IR-P2 — Step 5-AE / P3-5-R23: Four-Axis Measurement / Commit-Publish Classification

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R23）
- **判定**: **R23-A**。4軸（take／bld／cmt／seq）実測により
  B2前／B2後／commit後の三層分離が成立した。原因帰属はしない。
- **方法**: R22 vehicle（`a6f84a17` 後継 R23 binary）F episode #1。
  test-only `drp` field 追加のみ。production 追加変更なし。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take＋R19 B2＋R22 commit counter＋R10 accessor＋T6/T9＋delta観測 保持
ConvoPeq.md 再生成済み（topology：writer 各1・順序 take→B2→commit→sequence）。
production差分＝承認範囲のみ。
```

## 2. Latest ConvoPeq.md reconciliation

- take／B2／commit の writer・getter・順序を live source で再確認し、
  R18／R21／R22 の監査内容と一致することを確認した。
- `lastDroppedGeneration` は lifecycle getter の既存 field であることを確認し、
  新規 drop 計装なしで `drp` 読出しを追加した（§10）。

## 3. R22 vehicle reconciliation

- R22 binary＋`cmt` field（main-site 到達のみ）＋`drp` field（既存値）。
- 待機条件不変（30 s・sleepPump(800)・500 ms・warm4＋cap12・F順序）。
- 新規 logger／CLI／timeout／retry／health／pressure 計装なし。

## 4. Measurement protocol

- `qdelta tag=pre／post` に req／que／dup／take／bld／cmt／drp／seq／blo。
  delta は監査側算出。絶対値の記録のみで同期・settle 追加なし。

## 5. Raw pre/post observations（episode #1・clean exit）

```text
pair1 am-20 sc0（strict）:
  pre:  req=2 que=2 dup=0 take=2 bld=2 cmt=2 drp=0 seq=5 blo=0
  post: req=5 que=5 dup=0 take=5 bld=3 cmt=2 drp=0 seq=5 blo=0
  → WARN publish not confirmed（timeout・pair無効・failures=1）
pair2 am-6 sc0（inherit）:
  pre:  req=6 que=6 dup=0 take=6 bld=4 cmt=3 drp=0 seq=8 blo=0
  post: req=8 que=8 dup=0 take=7 bld=4 cmt=3 drp=0 seq=8 blo=0
pair2 am-6 sc1（inherit）:
  pre:  req=10 que=10 dup=0 take=9 bld=7 cmt=5 drp=0 seq=8 blo=0
  post: req=12 que=12 dup=0 take=10 bld=7 cmt=5 drp=0 seq=8 blo=0
summary cases=1 failures=1／gainDb -13.0276／-13.0276（7回目の再現）
```

## 6. Delta table

| window | reqΔ | queΔ | takeΔ | bldΔ | cmtΔ | drpΔ | seqΔ | 分類 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pair1-sc0 | +3 | +3 | +3 | +1 | 0 | 0 | 0 | **B2後・commit前（Case B形）** |
| gap→pair2-pre | +1 | +1 | +1 | +1 | +1 | 0 | +3 | 参考（window外） |
| pair2-sc0 | +2 | +2 | +1 | 0 | 0 | 0 | 0 | **B2前（build-stage）** |
| gap→sc1-pre | +2 | +2 | +2 | +3 | +2 | 0 | 0 | 参考：**cmt有・publish無** |
| pair2-sc1 | +2 | +2 | +1 | 0 | 0 | 0 | 0 | **B2前（build-stage）** |

## 7. take/bld/cmt/seq classification

- pair1-sc0： taken=3 のうち1が B2 到達（bld+1）。cmtΔ=0・seqΔ=0。
  → **B2後・commit-enqueue前（Case B形）で window 内停止**。
  R20 の同 window（seq+1）との差は window-edge の完了時刻非決定性として記録（§8）。
- pair2 両 pass： taken=1・B2 未達・cmt 0・seq 0 → **B2前で安定**
  （R20 と同一形状＋cmt=0 の追加確定）。
- gap（sc0-post→sc1-pre）： cmt+2・seq+0・drp+0 → **Case C 形
 （commit-enqueue 到達後・publish 未達）を gap 区間で観測**。
  drop 記録なしのため commit-drop 断定はしない（§9）。
- Case D（cmt>0＋seq>0 の同 window 共存）は本 episode の window 内では未観測。
  gap（pair1-post→pair2-pre）では take+1／bld+1／cmt+1／seq+3 が共存するが
  per-task 対応付け不能のため参考記録に留める。

## 8. R20 comparison

- R20 pair2（take=1／bld=0／seq=0）と R23 pair2（＋cmt=0）は同一形状。
  cmt=0 の追加により B2前確定が強化された。
- R20 pair1（take+3／bld+1／seq+1）と R23 pair1（take+3／bld+1／cmt0／seq0）：
  bld 到達は同一。seq の window 内到達は run 間で非決定的（R20:+1／R23:0）。
  R23 の cmt=0 により「B2到達分は window 内で main-site commit 未到達」が
  新規確定（Case B形）。gap 内で両 run とも seq→8 に収束。
- R17→R20→R23 で同一 vehicle 形状が 3 run 連続再現している。

## 9. lastDroppedGeneration observation

- 全6点で `drp=0`（Δ=0）。monotonicity 系 drop 記録なし。
- gap の cmt 有・seq 無と併せても drop 理由の断定はしない（§10 遵守）。
  非 monotonicity 系（health／pressure／fading／not-finalized）および
  coordinator 滞留・defer 継続は残余として残る。

## 10. window-boundary race

- post-read は瞬時値であり、trailing の take／build／commit／publish は
  window 外に落ちる（gap の cmt+1／seq+2 が実例）。
- `cmt<=bld<=take` は全 snapshot で成立する（2<=3<=5・3<=4<=6・3<=4<=7・
  5<=7<=9・5<=7<=10）。違反なし。
- `seq<=cmt`： window 内は全て seqΔ=0 のため seq>cmt は発生しない。
  gap→pair2-pre では seq+3 > cmt+1 であり、main-site 以外の publish 経路
  （recovery／idle 等）の存在と整合する（cmt は main-site のみ計数。R21 §8 の想定内）。
- gap→sc1-pre の bld+3 > take+2 は、cumulative 制約（3<=4<=7 → 5<=7<=9）を
  満たす遅延 build の catch-up であり、違反ではない。
  race 潰しのための同期追加はしない。

## 11. Per-task attribution limitation

- takeΔ>1 の window では個々の task へ対応付けない（R15 既知制限の維持）。
- generation linkage なしのため、どの take が build／commit したかは不可知である。
  window-level の分類（B2前／B2後／commit後）に留める。

## 12. Unresolved causes（断定しないリスト）

```text
build-fail 内訳／validate-fail／obsolete／exception／slow-build／
commit-side health-pressure-fading／coordinator stall／defer／
gap publishes の発行主体／per-task 対応／fade 0.06 由来
```

## 13. R23 Gate

```text
R23-A： ADOPTED
  4軸測定成功／take-bld-cmt-seq 比較可能／B2前・B2後・commit後の三層分類可能／
  既存 counter topology と矛盾なし（§10 の設計差を明記）／原因早期断定なし。
  in-window 観測： Case A（pair2×2）＋Case B（pair1: bld+1／cmt0／seq0）。
  Case C 形（cmt>0／seq=0）は gap 区間で観測（drp 0）。Case D は in-window 未観測。
  → cmt=1／seq=0 形の実在は gap で確認済み（task 単位確認は per-task 制限により未達）。
    coordinator／admission 側の最小観測点の要否判断へ（別 gate）。
R23-B／R23-C： 非該当。
```

## 14. STOP（本 Step 終了）

- R vehicle build／実行・F/R 比較・P3-1-D・limiter／stale／crossfade 帰属なし。
- source は R23 vehicle のまま保持する。revert なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
