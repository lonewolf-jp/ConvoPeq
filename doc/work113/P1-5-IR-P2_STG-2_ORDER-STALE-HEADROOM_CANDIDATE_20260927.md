# STG-2 Candidate — Order/Bypass Change Leaves Input Headroom Above Mode Ceiling

```text
Authority HEAD : 6dc032133f365ed4ad6a525fe8353c54b7124d8f
Authority file : ConvoPeq.md 5,557,334 B
SHA-256        : 3C61996501C6DA27AD85F36998E746301BC6CBF1BBED48F338C11015A508CD7A
Generated      : 2026-09-27 23:12:13
Phase          : Discovery only (no source/test modification)
```

## 1. Defect ID / title

```text
STG-2 — Processing-order / bypass change leaves inputHeadroomDb above the
new mode ceiling (stale manual staging, max 6 dB excess into convolver-first chain)
```

STG-1（bypass setter が staging を mode default で上書き）と対称の残存欠陥。
STG-1 修正は上書きを除去したが、再検証（re-clamp）を追加していないため、
逆方向の不変条件破れ（古い値の残留）が残る。新規 ID として扱い、
STG-1 / H-01〜03 / C-01 / N2-OPEN-1 / D105-R5-9 / work92 / I-5 系の再オープンはしない。

## 2. Current source locations（current authority）

```text
setter/clamp : src/audioengine/AudioEngine.Parameters.cpp L223-241
  setInputHeadroomDb: convIsFirst = !convBypassed && (order == ConvolverThenEQ || eqBypassed)
  maxDb = convIsFirst ? -6.0f : 0.0f; jlimit(-12.0f, maxDb, db)
order mutator (no re-validate):同 L267-274 setProcessingOrder
bypass mutators (no re-validate, post-STG-1):同 L154-172
dead re-validator:同 L295-343 applyDefaultsForCurrentMode（caller 0：宣言 AudioEngine.h L2733 のみ）
snapshot (no clamp):src/audioengine/AudioEngine.RebuildDispatch.cpp L55
spec copy (no clamp):src/audioengine/RuntimePublicationOrchestrator.cpp L129,
  src/audioengine/RuntimeBuilder.h L169, RuntimeBuilder.cpp L297（manual 経路は素通し）
DSP consume : src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp L235,
  src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp L334
StateIO     : src/audioengine/AudioEngine.StateIO.cpp L38-59（mode 先行復元）,
  L76-77（staging 復元＝clamp 再適用）, L197（保存）
debug validator（wide のため検出不能）:AudioEngine.Parameters.cpp L113 (-12.0..0.0)
design intent（参考・authority 外）:doc/gain_staging_analysis_2026-06-21.md L66-67, L112-118
  Conv系 -12〜-6 dB / PEQ系 -12〜0 dB。役割＝先頭 Convolver 時の畳み込み後クリップ防止。
```

## 3. Trigger / precondition

```text
- AutoGainStaging OFF（手動 staging が DSP に届く状態。UI の manual 編集で OFF になる正規状態）
- order = EQThenConvolver, conv active, eq active
  → convIsFirst = false → maxDb = 0.0
- setInputHeadroomDb(0.0) → 0.0 が正当に格納される（gain 1.0）
- その後 setProcessingOrder(ConvolverThenEQ) を呼ぶ（または conv unbypass で convIsFirst が false→true になる遷移）
  → staging に一切触れない
```

## 4. Actual behavior

```text
- getInputHeadroomDb() は 0.0 のまま（新 mode の上限 -6.0 を 6 dB 超過）。
- 次回 rebuild の snapshot/spec は atomic を無加工で DSP へ流す
  （RebuildDispatch L55 → Orchestrator L129 → Builder L297 → automation.inputHeadroomGain = 1.0）。
- getCurrentState() は order=ConvThenEQ と headroom=0.0 の組合せを保存する。
- 同じ state を requestLoadState() すると Step1 で order 復元後に
  setInputHeadroomDb(0.0) が -6.0 に clamp される → save/load 非冪等（0.0 → -6.0）。
```

## 5. Expected behavior

```text
- setter が強制する mode ceiling（conv-first 系は -6 dB）は mode 遷移後も維持される。
  すなわち order/bypass 変更時に staging を新 mode で re-clamp（または等価の正規化）する。
- design doc の定義（Conv系 -12〜-6 / PEQ系 -12〜0、入力保護）が全到達状態で成立する。
```

## 6. Concrete consequence（observable）

```text
- Convolver-first チェーンが ceiling -6 dB（gain 0.501）ではなく 0 dB（gain 1.0）で駆動
  → 最大 6 dB（電圧比 2x）過大入力。設計上の「畳み込み後クリップ防止」が外れる。
- 保存 state がその mode では setter が拒否する値を含む不正 state になる。
- save → load で値が変わる（0.0 → -6.0）。round-trip 非冪等は state が不正だったことの独立した証拠。
```

## 7. Deterministic reproduction / counterexample（code trace、動的実行なし）

```text
e.setAutoGainStagingEnabled(false);
e.setProcessingOrder(EQThenConvolver);   // + both bypass false
e.setInputHeadroomDb(0.0f);             // OK: maxDb 0.0 → stored 0.0
assert(e.getInputHeadroomDb() == 0.0f);
e.setProcessingOrder(ConvolverThenEQ);   // staging untouched
assert(e.getInputHeadroomDb() == 0.0f);  // ← 新 ceiling -6.0 違反（6 dB 超過）
const auto saved = e.getCurrentState(); // order=ConvThenEQ, headroom=0.0 を保存
e.requestLoadState(saved);              // Step2 で clamp → -6.0
assert(e.getInputHeadroomDb() == -6.0f); // ← 保存値と不一致。state が不正だった証拠。
```

bypass 経路の対称例：conv bypassed（max 0.0）で 0.0 を格納 → conv unbypass（order ConvThenEQ 系）→ 同様に 0.0 残留。

## 8. Existing coverage（not covered）

```text
- GainStagingContractTests.cpp：AutoGainPlanner::plan() の reference 検証のみ。manual setter 不変条件なし。
- ConvolverStateRoundTripTests STG-1：意図的に全 mode で valid な -8.0/+3.0/-4.0 を使用
  （L2242-2243 コメント）。mode ceiling を跨ぐ値（0.0 等）の order 遷移は未試験。
- requestLoadState 経路は mode 先行→staging clamp のため load 時に normalize されるが、
  live の setProcessingOrder / bypass setter には対応する re-clamp が存在しない。
- debug validator L113（-12..0）は stale 0.0 を通過させるため検出不能。
- cocoindex semantic search「re-validation after order/bypass change」：production の再検証経路なし（docs のみ hit）。
- WSL rg：setInputHeadroomDb の caller は StateIO/DeviceSettings/UI のみ。order/bypass setter からの呼出しなし。
```

## 9. Why existing contract does not permit

```text
- setter コメント「コンボルバーが先頭に来る場合 (Conv→PEQ / Conv only) は -6dB 上限で入力保護」
  ＋ jlimit 強制は、0.0-in-conv-first をコード自身が invalid とみなす証拠。
- AudioEngine.h の setter 宣言群に stale 値の許容文書なし。
- STG-1 が確立したのは preservation（bypass で壊さない）であり、天井超過値の保持許可ではない。
- applyDefaultsForCurrentMode は caller 0 の dead code。mode 遷移時の正規化責務が宙に浮いている。
```

## 10. CLOSED 項目との非重複

```text
STG-1（bypass 上書き）とは逆方向（残留）。H-01/02/03・C-01・N2-OPEN-1・D105-R5-9・
work92・I-5/I-5-R1 のいずれの成立条件とも一致しない。current-authority の新規 trace による standalone 候補。
```

## 11. Minimal repair direction（提案のみ・未実施）

```text
Option A（推奨・最小）：setProcessingOrder＋両 bypass setter の末尾で
  setInputHeadroomDb(getInputHeadroomDb()) 相当の re-clamp を1行追加
 （既存 clamp ロジックを再利用。AutoGain ON 時は Builder が上書きするため無害。
   m_isRestoringState 配慮：requestLoadState は mode 先行復元＋個別 clamp のため二重 clamp は冪等）。
Option B：captureBuildParameterSnapshot 側で clamp（単一 choke point だが DSP と UI 表示の乖離が残る）。
いずれも production 1-3 行＋regression test。Discovery phase のため実施しない。
```

## 12. Risk / ISR impact

```text
- RT path／ISR Bridge／RuntimeWorld／Publication／Crossfade／Retire-Epoch／Recovery-Rebuild
  ownership／atomic wrapper 契約：変更なし（値のみ正規化、構造不変）。
- 影響は NonRT parameter path のみ。全 setter は ASSERT_NON_RT_THREAD 維持。
- 既存 valid 値（-8.0 等）への影響なし（冪等）。AutoGain ON 経路への影響なし（Builder が上書き）。
```

## 13. Phase gate

```text
production source: 0
test: 0
build: 0
commit: NOT STARTED
push: NOT AUTHORIZED
dynamic reproduction: deferred（regression test 新設は Owner GO 待ち）
```
