# STG-7-D1 Repair Gate — autoGainStagingEnabled 復元欠落の修正記録（2026-09-28）

## 1. Root cause
- `AudioEngine::requestLoadState()`（`src/audioengine/AudioEngine.StateIO.cpp`）は
  `autoGainStagingEnabled` プロパティを読み取り、手動ゲイン復元の可否判定にのみ使用し、
  live flag 自体を復元していなかった。
- `getCurrentState()` は同プロパティを保存するため、`save → load` で live flag が
  保存値に戻らず、`save → load → save` で保存値が反転する semantic non-idempotence があった。

## 2. Source location
- 欠落箇所: `src/audioengine/AudioEngine.StateIO.cpp` `requestLoadState()` Step 2
  （`autoGainEnabled` 判定直後、manual gain 復元分岐の前）。
- setter: `src/audioengine/AudioEngine.h:1416` `setAutoGainStagingEnabled()`。
- 保存側: `src/audioengine/AudioEngine.StateIO.cpp:207`（`getCurrentState`）。
- 回帰テスト: `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`
  `checkSTG71AutoGainFlagRoundTrip()`（新規、既存 harness 内 subtest。CTest 新規登録なし）。

## 3. setAutoGainStagingEnabled() 副作用監査（Phase B）
実読対象: `AudioEngine.h:1416-1432`、`EQProcessor.Parameters.cpp:127-146`、
`AudioEngine.Parameters.cpp:310-316`、全 caller（`DeviceSettings.cpp:388`、
`StateIO.cpp:78`（本修正）、UI トグル経路）。
1. flag のみか → 否。3 作用: (a) `autoGainStagingEnabled` atomic publish、
   (b) `getEQProcessor().setAGCEnabled(!enabled)`（Bug#4: 二重ゲイン補正防止）、
   (c) `submitRebuildIntent(Structural, Replaceable)`（BUG-10: ON/OFF で
   ProcessingPart の手動値/AutoGainPlanner 選択が変わるため）。
2. parameter staging を直接変更するか → 否。`inputHeadroomDb` 等への直接 publish なし。
   選択切替は次回 rebuild 時の RuntimeBuilder 経由。
3. rebuild submit を発生させるか → 発生させる（Structural / Replaceable / Snapshot）。
   `requestLoadState` 末尾の最終 Structural 意図と合流・置換されるため bulk restore と競合しない。
4. publication を直接発生させるか → 否。deferred publish pipeline 経由のみ。
5. planner/manual gain state を直接変更するか → 否。
6. bulk restore 中の安全性 → 安全。`RestoreStateGuard`（`m_isRestoringState`）は関数全域で
   `applyDefaults` を抑制し続ける。setter 自体は `noexcept` かつ ASSERT なしで restore 文脈から呼出可能。
   変更なし時は setter 内 early-return のため副作用ゼロ（idempotence を保存）。
7. 復元順序への適合 → 適合（下記 Phase C）。

## 4. requestLoadState() 復元順序（Phase C）
確定順序: Step 1（mode/order/bypass）→ Step 2（autoGain flag 復元【本修正】→ manual gain 条件復元 →
dither/enum ガード）→ Step 3（その他グローバル）→ Step 4（EQ/Convolver サブツリー復元）
→ 最終 Structural rebuild → `sendChangeMessage()`。
- flag 復元は manual gain スキップ判定の直前（`if (!autoGainEnabled)` の前）に配置。
  ON 保存時は手動ゲインを上書きせず（Bug#3 維持）、OFF 保存時は手動ゲインを復元する。
- Step 4 より前の配置の根拠: 新規 preset は engine flag と EQ subtree（`agcEnabled = !engine flag`）
  が自己無矛盾に保存されるため、Step 4 の subtree 値で最終 DSP 状態が保存値と一致する。
  保存 DSP バイトをverbatim 保持する立場を取った。
- 既知の edge: 旧 preset（`autoGainStagingEnabled` プロパティなし）は engine=false 扱いで
  手動ゲインを復元する（v14.41 互換維持）。この場合の engine/DSP ペアの不変条件は
  本 defect の scope 外（製品判断事項）であり、Step 4 後に flag を置く再設計は行わない。

## 5. 修正内容（Phase D・最小修正）
```cpp
// ★ STG-7-D1: 保存された AutoGain flag 自体を復元する（判定のみに使って
//   終わりにしない）。EQ サブツリー復元（Step 4）より前なので、DSP AGC の
//   不整合があれば Step 4 の subtree 値で上書きされ、最終状態は一致する。
//   変更なし時は setter 内 early-return のため副作用なし。
setAutoGainStagingEnabled(autoGainEnabled);
```
- 追加位置: `autoGainEnabled` 読取直後、`if (!autoGainEnabled)` 手動ゲイン復元の前。
- 契約: `saved autoGainStagingEnabled → requestLoadState() → live autoGainStagingEnabled` を一致させる。
- `oversamplingFactor` 等の明示的 StateIO 対象外への拡張なし。

## 6. 回帰テスト（Phase E）
`checkSTG71AutoGainFlagRoundTrip()`:
- Case A: ON→save→OFF→load→ON を `isAutoGainStagingEnabled()` で検証。
- Case B: OFF→save→ON→load→OFF を検証。
- `save → load → save` で `autoGainStagingEnabled` プロパティの semantic 一致を検証。
- テスト付随の安定化: `stg61SettlePublication()` を backlog ゼロ＋1500ms 安定要求に強化。
  理由: deferred rebuild tail（200ms window＋rebuild＋publish）の消化を確実化するため。
  判定の緩和ではなく厳格化であり、STG-6-D1 の契約変更なし（初回実行で baseline tail の
  残存による `seq=3->5` を観測し、強化後に `seq=5->5` で安定 PASS）。

## 7. Regression 結果（Phase F/I）
- Debug harness（`stg1_run.bat` 相当）: STG-7-D1 PASS / STG-1（6/6）PASS /
  STG-2（3/3＋A/B/C＋R4/R5/R6）PASS / STG-4-1 PASS / STG-6-D1 PASS（`seq=5->5`）/
  既存 StateIO 系（AC-A1、SR01、M01/M02/M04、H01/H02）PASS /
  loader 系（WORK102 IRLoadAdmission、PREV01）PASS。`FAIL: check...` なし。
- Release harness（`build/Release/AudioEngineHarness.exe`）: 同上すべて PASS、
  `AudioEngineHarness: all publish pipeline tests PASS`。
- CTest（`i5r1_ct.bat`、`ctest -C Release`）: **42/42 PASS**（基準維持）。
  `AudioEngineHarness`（78.10s）を含む全テスト PASS。

## 8. Persistence invariant（Phase G）
- `getCurrentState()` → 保存 flag → `requestLoadState()` → `isAutoGainStagingEnabled()` の
  semantic 一致を Case A/B で確認。
- `save → load → save` で flag 反転なしを確認。
- STG-1/STG-2 回帰の維持により、autoGain flag と manual staging gain / input headroom /
  mode・order・bypass の相互上書きなしを確認（Bug#3 スキップ分岐は不変）。

## 9. ISR / authority audit（Phase H）
- 変更ファイル: `src/audioengine/AudioEngine.StateIO.cpp`（NonRT StateIO のみ）、
  `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`（test のみ）、
  `ConvoPeq.md`（再生成物）。
- RT path / RuntimeWorld / RuntimeStore / Publication / Crossfade / Retire-Epoch /
  Coordinator の変更: なし。
- 新規 raw atomic: 0（追加行に `std::atomic` なし。新規 call site は既存 setter のみ）。
- StateIO は NonRT 処理であり RT contract 変更なし。

## 10. Authority（Phase A/J）
- 修正前 authority: SHA-256 `A7C03CD4FA02311BD1A29F5D9940F944BD26F00431DF0E98C1F59DCF025239DC` /
  size 5,586,461 B / Generated 2026-09-28 11:58:08 / `--check` FRESH /
  HEAD `9afc71deb161b0fd862b21b4f19b4189856cc3b7`（実装開始時に一致確認済み）。
- 修正後 authority: SHA-256 `5DB50053B2B4B98692FBBC04654F9C4A068CD8979D0896450B79D9DFB73B61C1` /
  size 5,589,694 B / Generated 2026-09-28 13:10:22 / `--check` FRESH /
  NEWER_SRC_COUNT 0 / HEAD `9afc71deb161b0fd862b21b4f19b4189856cc3b7`（不変）。
- diff scope: `src/audioengine/AudioEngine.StateIO.cpp`（+6）、
  `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp`（+70・内 settle 強化）、
  `ConvoPeq.md`（再生成・+78）。その他 production file の変更なし。
  （`AGENTS.md` / `headroom-proxy-start.ps1` の作業ツリー差分は本作業前から存在する
  無関係な変更であり、非接触。）

## 11. Worktree status / commit policy
- commit: 未実施（Repair Gate 承認待ち）。push: NOT AUTHORIZED（継続）。
