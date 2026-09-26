# P1-5-IR-P2 — Step 5-H / P3-5: Test-Only Vehicle Execution（F-invalid STOP）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5）
- **判定**: **F vehicle integrity FAIL → STOP**。R は実行していない。
  order comparison（Pattern A/B/C）には到達していない。
- **変更**: `src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp` の1ファイルのみ
  （P3-4 §11範囲）。production/CMake/JUCE＝0 diff を維持。
- **raw evidence**: `tmp/p35_F_run.log`（166行・summary有・clean exit）、
  `tmp/p35_F.exe`（F binary保存）、`tmp/p35_F_build.log`。

---

## 1. State Freeze（実装前・PASS）

```text
HEAD                         1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix               true（維持）
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF（cache確認）
production / CMake / JUCE    0 diff
```

pre-existing の4ファイル差分と今回diffの混同なし（§2で分離確認）。

## 2. test-only diff（F vehicle・1ファイル）

対象：`src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp`
（numstat 493 insertions／17 deletions＝pre-existing 427/16＋今回約66/1）：

```text
a. constexpr kP15G0OS1Only=true・kP35PostSettleMs=500 追加（新規CLIなし）
b. runCase: requirePublish 末尾param＋第1 strict-honor（WARN＋return false）＋
   第2継承確認（backlog既存WARNのみ）＋既存sleepPump(800)保持＋gated post-settle
c. runPair: firstInVehicle param＋初回sc0へのみ伝播（topology不変）
d. section A–H,J–M 抑止（bare blockは if(!kP15G0OS1Only)、H/J/K/Lはgate追加）
e. section I: g0単一・os{1}・adb{-20,-6}（F順序）＋firstPair伝播
f. runPair/ensureTestIr×2/sc0/sc1/diffStats 構造不変
```

禁止確認（0件）：新CLI・新logger prefix・limiter accessor・limiter reset・
既存sleep変更（`sleepPump(800)` 保持）・production/CMake/JUCE差分。
第2ケースに `published` の語義拡張なし。`getCurrentEnvelope` 参照なし。

## 3. production diff = 0

`git diff --name-only` に production／CMake／JUCE の新規差分なし
（表示されるのは P3-3 以前からの持ち越しのみ）。
`kP15FullMatrix=true`・OFF を再確認。

## 4. build identity（F）

```text
target   AudioEngineHarness / Release / OFF（vcvars64＋oneAPI setvars経由・A3先例準拠）
result   BUILD_EXIT=0（1 TU再compile＋link）
path     build/Release/AudioEngineHarness.exe → tmp/p35_F.exe に保存
SHA-256  f472381550414f51a16d74782ecd1176bd01fa49ab385ddb6d1ebfa67ee8b9ad
prefix   f472381550414f51（A3 b2c39a9a… と意図どおり相違）
p15ir string  1（含有）
CONVOPEQ_CORRECT_POLYPHASE_GAIN  OFF
```

identity不一致なし → 実行へ進んだ。

## 5. F実行結果（fresh process・`tmp/p35_F.exe --p1-char`）

```text
[P1CHAR] flag_macro=0 (0=OFF build / 1=ON build)
[P1CHAR] WARN publish not confirmed os=1 sc=0
[P1CHAR] p15ir id=g0_os1_am-6 os=1 n=0 ampDb=-6.0
         gainDb_sc0=-13.0276 gainDb_sc1=-13.0276
         limitingEngaged_sc0=0 limitingEngaged_sc1=0
         hardClamp_sc0=0 hardClamp_sc1=0
         clipEngagement=1400 clipEngMax=0.000000
[P1CHAR] summary flag_macro=0 cases=1 failures=1
```

- pair1（am-20）：sc0 strict-honor が 30 s timeout → WARN → capture skip → failures=1。
  am-20 行は存在しない。
- pair2（am-6）：継承分岐で capture 実施 → 1行出力 → cases=1。
- clean exit（crash・heap corruption・ASan なし。H-Bマーカーなし）。

## 6. F vehicle integrity（Gate P3-5-A：FAIL）

```text
F/Rとも正常実行        : Fのみ実行・pair1欠落のため FAIL
g0/os1対象行が存在      : am-6 のみ存在・am-20 不在のため FAIL
第1ケース publish PASS  : timeout（WARN確認） のため FAIL
第2ケース inherited     : capture実施（条件付きPASS・単独では無意味）
crashなし               : PASS（clean exit＋summary）
```

→ **integrity FAIL。比較解析に進まない。R は実行しない**（P3-5 §8遵守）。

### timeout原因の read-only 整理（断定なし・次gate材料）

- fresh process 初回 publish が 30 s を超過、または dispatch 未発生。
  backlog WARN なし（waitBacklogZero は通過）＝ loader backlog は空。
  よって configureChain 起因 build の publish が 30 s 以に commit されなかった。
- 候補（いずれも未確定）：cold-start build 遅延／初回重初期化コスト／
  publish pipeline の初回遅延。no-dispatch 説は no-guard setter 3 本の存在と整合しない。
- 設計への示唆：第1ケースの待機 budget 30 s が cold start に対して不足の可能性。
  budget 延長・warm-up pair 追加はいずれも test-only 再設計事項であり、
  本 Step では実施しない（次gate判断）。

## 7. R実行結果（NOT EXECUTED）

F 異常のため R build（adb 1 token 変更）・R 実行は実施していない。
source は F vehicle 状態のまま（adb Forward 順序保持）。

## 8. R vehicle integrity（NOT EVALUATED）

## 9. measurement validity（Gate P3-5-B：comparison 対象外）

am-6 単行としては capture 成功・4096 解析・dftDb/gainDb 取得が成立している。
しかしペア欠落のため order comparison は禁止する（Gate 未到達）。

### OBSERVED（帰属なし・記録のみ）

1. am-6 `gainDb_sc0=-13.0276` は A3 と exact 一致。
   先行 context（pair1 configureChain 適用済み・capture skip）が異な条件下での一致である。
2. `gainDb_sc1=-13.0276` は sc0 と exact 一致（A3 sc1=-15.5220 とは不一致）。
   `clipEngagement=1400 / clipEngMax=0.000000`（A3: 4096 / 0.163382）。
   sc1 pass が sc0 と実質同一 world＋history で処理されたことと整合的だが、
   因果の断定はしない（merge-no-dispatch 機構の観測例候補として記録）。
3. g0/os1 限定 vehicle は clean exit し、p15hrnl crash 域を回避した。
   crash 非発生を H-B の因果証拠とは扱わない（P3-5 §11遵守）。

## 10. F/R comparison（Gate P3-5-C：未到達）

ΔF（−20 不在のため算出不能）・ΔR（未実行）のため比較不能。

## 11. crash/exception状況

```text
0xc0000005／0xc0000374／heap corruption／crash／ASan failure： なし
p15hrnl到達： なし（section抑止により構造的に到達不能）
H-B領域への侵入： なし（H-Bマーカー不在・対象外維持）
IR geometry変更： なし（g0資産同一）
第1ケースpublish timeout： 有（§5・§6。STOP条件ではなく run-invalid 条件として処理）
capture前提崩壊： pair1のみ（設計どおり無効化・pair2は前提内）
```

## 12. Pattern A/B/C（NOT CLASSIFIED）

比較 gate 未到達のため分類しない（Pattern A/B/C のいずれにも入れない）。

## 13. attribution conclusion（断定可能範囲のみ）

- 断定できること：F vehicle（g0/os1・publish-honored）は pair1 の cold-start publish
  未確認により無効化され、order 比較は不成立。本 vehicle 形での比較には
  第1ケース待機の再設計が必要。
- 断定しないこと：limiter／state-history／order の帰属一切（§9 OBSERVED は材料止まり）。
- P3-2 M3・P3-3 V-B・P3-4 G-A の各判定は本結果により覆らない。
  覆るのは「現 vehicle 形で F/R 比較が実行可能」という実行可能性の前提のみである。

## 14. 次Gate

1. 第1ケース待機の再設計（budget 延長 or warm-up pair or publish ポーリング診断）を
   test-only 変更として次gateに諮る。いずれも P3-4 §11 範囲内の追加と位置づける。
2. R build／実行は F-valid 確立後。現 source（F vehicle）は保持し、 revert しない。
3. P3-1-D（logger/accessor）は開始しない。E3 不要の判断を維持する。
4. §7解釈制約（新core／publish戻り値無視の一般則／envelope持続の肯定・否定禁止）を維持する。
