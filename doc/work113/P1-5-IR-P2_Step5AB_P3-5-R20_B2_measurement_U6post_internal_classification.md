# P1-5-IR-P2 — Step 5-AB / P3-5-R20: B2 Measurement / U6-post Internal Classification

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R20）
- **判定**: **R20-A**。U6-post を B2 前後に分類した。原因特定はしない。
- **方法**: R19 vehicle（F・`8222b969` 後継 R19 binary）1 episode。
  production 追加変更なし。60/120 s 実行なし。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16 take counter＋R19 B2 counter＋R10 accessor＋T6/T9＋delta観測 保持
ConvoPeq.md 再生成済み（R19実装含有を確認）。
chain 再確認（live source）：take :906-909 → build :1211 → null-check :1215 →
  B2 increment :1235 → obsolete :1232系 → rebuildAllIRs → validate :1281 →
  commit :1417 → sequence bump。R18行番号のまま信用せず再特定した。
```

## 2. 最新 ConvoPeq.md source reconciliation

- `rebuildTakeCount_` 宣言1／getter 1／writer 1（take直後）。
- `rebuildBuildResultCount_` 宣言1／getter 1／writer 1（B2）。
- R18報告書との差異なし（production 無変更のため）。

## 3. R19 counter topology確認

- take counter： take完了直後のみ（queue／wake／backlog／build／commit／publish 非加算）。
- B2 counter： null-check通過直後のみ（validation／commit／publish 非加算）。
- test reader： `emitQDelta` の `take`／`bld` field（既存 WARN 行）。
  build-result／commit／health／pressure の新規計装なし。

## 4. measurement vehicle

R17 vehicle＋bld field（R19 binary・F順序・fresh process）。
待機戦略不変（30 s wait・sleepPump(800)・500 ms post-settle・warm4＋cap12）。
`qdelta tag=pre／post` に req／que／dup／take／bld／seq／blo を記録し、
delta は監査側で算出する。

## 5. 各caseの pre/post raw values（episode #1・clean exit）

```text
pair1 am-20 sc0（strict）:
  pre:  req=2 que=2 dup=0 take=2 bld=2 seq=5 blo=0
  post: req=5 que=5 dup=0 take=5 bld=3 seq=6 blo=0
  → WARN publish not confirmed（timeout・pair無効・failures=1）
pair2 am-6 sc0（inherit）:
  pre:  req=6 que=6 dup=0 take=6 bld=4 seq=8 blo=0
  post: req=8 que=8 dup=0 take=7 bld=4 seq=8 blo=0
pair2 am-6 sc1（inherit）:
  pre:  req=10 que=10 dup=0 take=9 bld=7 seq=8 blo=0
  post: req=12 que=12 dup=0 take=10 bld=7 seq=8 blo=0
summary cases=1 failures=1／gainDb -13.0276／-13.0276（6回目の再現）
```

## 6. delta表

| window | reqΔ | queΔ | takeΔ | bldΔ | seqΔ | 分類 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pair1-sc0 | +3 | +3 | +3 | +1 | +1 | **Case C相当＋B2到達**（§7） |
| pair2-sc0 | +2 | +2 | +1 | 0 | 0 | **Case A（build-stage）** |
| pair2-sc1 | +2 | +2 | +1 | 0 | 0 | **Case A（build-stage）** |
| gap（pair1-post→pair2-pre） | +1 | +1 | +1 | +1 | +2 | window外・参考値 |

## 7. `takeΔ / bldΔ / seqΔ` classification

- pair1-sc0： taken=3 のうち1が B2 到達＋1 publish（window内）。
  残り2 takes は window 内 B2 未達。**B2 前後に跨る**。
  wait-boolean（timeout）と counter-evidence（seq+1）の乖離は、
  joint 条件（seq前進＋backlog 0＋250 ms 安定）の poll-timing 由来として記録する。
  wait 戻り値で publish 有無を判定しない（R6規律の維持）。
- pair2-sc0／sc1： taken=1・B2 未達・publish なし → **B2前（build-stage）**。
  両 pass で同一形状（安定パターン）。
- Case D（queuedΔ=0）：未観測。Case A（ §4 定義の take=1／bld=0／seq=0）は
  pair2 両 pass が該当する（takeΔ=+1 の bound 内）。
- `bldΔ <= takeΔ` は全 window で成立（1<=3・0<=1・0<=1）。違反なし。

## 8. window-boundary raceの記録

- post-read は瞬時値であり、trailing の take／build／publish は window 外に落ちる。
  gap の take+1／bld+1／seq+2 はこの race と整合する（正常非同期遅延として記録）。
- race を潰すための同期・sleep・settle 追加は行わない（指示どおり）。
- per-task attribution（どの take がどの build に対応するか）は
  generation linkage なしのため不可のまま（R15 既知制限）。

## 9. R17 U6-postとの対応

- R17 pair1（take 2→5・seq flat）と R20 pair1（take 2→5・seq 5→6）は
  take pattern が同一で publish timing のみ相違する。
  両 run とも pair2-pre で seq=8 に収束する。
- 結論：take→build→publish の完了時刻が 30 s window 境界を跨いで
  非決定的に分布する。同一 total（5→8）への収束は両 run で一致する。
- よって「timeout＝never-publish」ではなく「window-edge race」を含むと記録する。
  ただし 60／120 s 延長根拠にはしない（R1-B 維持。完了時刻の分布は未測定）。

## 10. B2前／B2後の帰属

- **B2前（build-stage）**：pair2-sc0／sc1（taken=1・B2未達）。
  内訳（build-fail／obsolete-pre／exception-in-build／slow-build）は不可分。
- **B2後（post-build）**：pair1 の 1 take（B2到達＋publish）。
  内訳（validate／commit／admission）は当該 take については publish 済みのため対象外。
  pair1 の残り2 takes は B2前残余として残る。
- いずれも原因の早期断定なし（§7 指示どおり）。
  `bld=1／seq=0` 形は本 episode では観測されず。
  観測されたのは `bld=1／seq=1`（pair1）および `bld=0／seq=0`（pair2）である。

## 11. 未解決事項

- per-task attribution（generation linkage なし）。
- gap publishes の発行主体・内容（3 publishes の内訳）。
- pair1 の残り2 takes の B2 未達理由（build-fail／obsolete／exception／slow の内訳）。
- pair2 takes の B2 未達理由（同上）。
- 上記はいずれも B2 以降／周辺の追加観測点（承認後）を要し、本 Step では触れない。

## 12. STOP＋R20 Gate

```text
R20-A： ADOPTED
  B2 counter の test-only 読出し成功／vehicle measurement 成功／
  take-bld-publish 比較可能／U6-post の B2 前後分類可能／早期断定なし／
  production変更 0。
  → 次は P3-5-R21（R20-A の受領後）。
R20-B／R20-C： 非該当。
```

- F/R実行（本 episode で完了）以上の run なし。R vehicle 実行なし。
- P3-1-D・limiter／stale／crossfade 帰属なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。
