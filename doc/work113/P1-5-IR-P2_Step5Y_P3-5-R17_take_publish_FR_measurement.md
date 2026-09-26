# P1-5-IR-P2 — Step 5-Y / P3-5-R17: Take/Publish Separation F Measurement

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R17）
- **判定**: **R17-A**。ただし R 実行は行わない（U6-post 確定 → build-result boundary 設計へ）。
- **方法**: Step 0 照合 → test-only delta 観測追加 → build → F episode #1。
  production 追加変更なし。60/120 s 実行・build-result 追加計装なし。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a／kP15FullMatrix true／OFF
R16実装（counter＋getter＋take点）保持／R10 accessor・T6/T9 保持
production差分＝R16承認範囲のみ（§2）／F vehicle＋R1 diagnostic 保持
```

## 2. R16 source reconciliation（Step 0・PASS）

- `ConvoPeq.md` を `output_sourcecode_markdown.py` で再生成。
- 確認：`rebuildTakeCount_` 宣言1／`getRebuildTakeCount` 1／
  take increment 1（ownership transfer 直後）／`rebuildBacklog_`・
  `queuedCount`・`getActiveRuntimeSnapshot` 存在。
- R16-SOURCE-RECONCILIATION 不発（不一致なし）。

## 3. Measurement vehicle（test-only 差分）

`P1PolyphaseGainCharacterization.cpp` の `runCase` に追加（既存 WARN 行のみ）：

```text
emitQDelta(pre)： configureChain 前（req／que／dup／take／seq／blo）
emitQDelta(post)： wait 終了直後（strict timeout path を含む全分岐）
```

- 新 logger／prefix family なし。新規 CLI／wait／sleep／settle／retry／
  build-result／commit／health／pressure 計装なし（§5遵守）。
- production 差分は R16 承認範囲のまま（§1 再確認）。

## 4. Observation definitions（delta 規約）

- カウンタ累積のため絶対値ではなく **delta** を判定値にする。
- `queuedΔ`＝queue insertion accepted／`takenΔ`＝worker ownership acquired／
  `publishedΔ`＝sequence 前進。queued／taken の語義重複なし（R16 位置制約）。

## 5. F/R results（episode #1・R17 binary `8222b969`・clean exit）

```text
pair1 am-20 sc0（strict）:
  pre:  req=2 que=2 dup=0 take=2 seq=5 blo=0
  post: req=5 que=5 dup=0 take=5 seq=5 blo=0
  → reqΔ=+3／queΔ=+3／dupΔ=0／takeΔ=+3／publishedΔ=0
  → WARN publish not confirmed（timeout・pair無効・failures=1）
pair2 am-6 sc0（inherit）:
  pre:  req=6 que=6 dup=0 take=6 seq=8 blo=0
  post: req=8 que=8 dup=0 take=7 seq=8 blo=0
  → reqΔ=+2／queΔ=+2／takeΔ=+1／publishedΔ=0
pair2 am-6 sc1（inherit）:
  pre:  req=10 que=10 dup=0 take=9 seq=8 blo=0
  post: req=12 que=12 dup=0 take=10 seq=8 blo=0
  → reqΔ=+2／queΔ=+2／takeΔ=+1／publishedΔ=0
summary cases=1 failures=1／gainDb -13.0276／-13.0276（5回目の再現）
```

- inter-pass req／que／take 増加（例：sc0-post→sc1-pre で+2/+2/+2）は
  `ensureTestIr` の IR load が request→queue→take を駆動することの証拠である。
  load 経路の DSPCore rebuild 非依存（R12 §4）と両立する。
- post-read 直後の worker take（race）により trailing edge の±1は
  未確定として記録する（§6の core 判定には影響しない）。

## 6. queuedΔ/takenΔ/publishedΔ table

| Episode | queuedΔ | takenΔ | publishedΔ | seqBefore | seqAfter | timeout | 判定 |
| --- | ------: | -----: | ---------: | --------: | -------: | ------- | --- |
| #1 pair1-sc0 | +3 | +3 | 0 | 5 | 5 | 有 | **Case B（U6-post）** |
| #1 pair2-sc0 | +2 | +1 | 0 | 8 | 8 | —（継承） | Case B（U6-post） |
| #1 pair2-sc1 | +2 | +1 | 0 | 8 | 8 | —（継承） | Case B（U6-post） |

- Case A（1/0/0）：未観測。takeΔ>0 が全 window で成立したため、
  never-consumed（worker 飢餓）は本 vehicle では排除される。
- Case C（1/1/1）：未観測。
- Case D（queuedΔ=0）：未観測（admission／queue 受理は全 window で成立）。
- 1/0/0 vs 1/1/0 の識別規則は成立し、観測値は 1/1/0 側に分類される。

## 7. U6 classification

- **U6-pre（never-consumed）**: 本 episode では排除（takeΔ>0 全面）。
- **U6-post（taken-without-published）**: **確定**（3 window 全て）。
  build／validation／commit-drop／exception／slow-build の内訳は不可分（次段）。
- per-task attribution（どの take がどの request に対応するか）は
  generation linkage なしのため不可（R15 既知制限）。window-level の判定に留める。

## 8. R17-A/B/C gate

```text
R17-A： ADOPTED
  source-binary 一致／F execution 成功（clean exit＋全delta取得）／
  queuedΔ・takenΔ・publishedΔ 取得成功／識別規則成立／production変更 0。
  ただし分岐図に従い R 実行ではなく build-result boundary 設計へ進む。
  （taken=1・published=0 → U6-post。F pair1-invalid は継続中のため
   order 比較は依然ブロックされる。）
R17-B／R17-C： 非該当。
```

## 9. Next-step boundary

- 次段：`taken=1, published=0` の内訳を分ける build-result boundary の設計
  （計装は承認後。先回り追加なし）。
- 60／120 s 実行・F/R 再実行・P3-1-D・limiter／stale／crossfade 帰属なし。
- R4 境界・保留事項・§7解釈制約・H-B 対象外を維持する。

## 10. STOP（本 Step 終了）

- R vehicle build／実行は行っていない（U6-post 確定により order 比較は非先行）。
- source は F vehicle＋delta 観測のまま保持する。revert なし。
