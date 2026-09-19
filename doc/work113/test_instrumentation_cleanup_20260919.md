# WORK113 test-instrumentation cleanup: TransitionMetrics / BassBuzz harness 品質修正（2026-09-19）

独立 work item（**Phase 2-2 closure には含めない**）。ユーザー判定（2026-09-19）:「計器欠陥は別 work item として残す」に基づく。

## 0. 発端（実害）

2026-09-19 の WORK113 Phase 2-2 検証（`bypass_mirror_committed_projection_20260919.md` §5-6）で、以下の計器欠陥が**実際に誤測定・誤判断を誘発**した:

- `--buzz-eq=1` / `--buzz-conv=1` が silent に bypass 扱いされ、EQ/conv とも無効のまま 4 capture を取得（PROBE_CFG 行で判明 until 再実行）。
- `rmsTrMax` が常に 0.000000 で出力され、データ上は 0.2185 があるにもかかわらず計器が無音と報告する状態を awk 再計算まで遡って検証する羽目になった。

## 1. 修正内容（test-only・production 変更なし）

### 1-1. `TransitionMetrics.h` — rmsTrMax の result 代入欠落

- 症状: ローカル `transientRmsMax` を計算したまま `result.` に代入しておらず、**常に 0.000000** を出力（他指標は正しい）。
- 修正: `result.transientRmsMax = transientRmsMax;` を result 詰め替えブロックに追加。
- 影響: transition 窓内の 1ms RMS 最大値が初めて正しく出力される。

### 1-2. `BassBuzzMeasurement.cpp` — boolean CLI parser の fail-closed 化

- 対象: `--buzz-eq=` / `--buzz-conv=` / `--buzz-direct=` / `--buzz-flip-eqbypass=`。
- 旧: `(substr == "on") ? 1 : 0` — `"1"` 等の値が**黙って 0(bypass)** になる。
- 新: `parseOnOff(flag, v)` ヘルパーを追加し、`"on"`/`"off"` 以外は `[BUZZ] FAIL: <flag> expects 'on' or 'off' (got '<v>')` を出して `exit(2)`（fail-closed）。
- スコープ外（ observation のまま変更せず）: `--buzz-order=` の `(=="etc")?1:0` は boolean ではなく列挙系で既存スクリプト互換があるため触らない。`parseHcIdx`/`parseLcIdx` は `std::stoi` 例外で fail するため silent 誤解釈はない。

### 1-3. `BassBuzzMeasurement.cpp` — metrics 窓の capture 実効 sample rate binding

- 症状: `computeTransitionMetrics(out, opt.sr, ...)` は capture 実効レート == `opt.sr` を仮定するが、tap の capture レートはデバイス/エンジン構成依存（2026-09-19 実測: engine 384kHz・capture 2.0s 指定に対し ~1.76M samples = 実効 ~880k samples/s・`opt.sr` 既定 192k）で窓位置がずれる。
- 修正: `effectiveCaptureRate = out.size() / probeCaptureSec`（probe capture は 2.0s 固定）を算出し、これを `computeTransitionMetrics` の sampleRate 引数に渡す。flip 時刻（runCapture 開始からの wall clock）と同一タイムラインに bind される。
- 性質: 平均レートによる bind であり、コールバックレートが非一様な場合は近似的（計器の目的に対して十分）。

### 1-4. `BassBuzzMeasurement.cpp` — rigcheck=eq の設定と criterion の整合（モード別 PASS 窓）

- 症状: rigcheck=eq は `kDefaultBands +3dB × total −2dB × saturation 0.05` を設定しながら、identity 判定窓 `[0.880,0.897]` + `THD<−80dB` を要求 → **FAIL が構造的**（実測 ratio 0.5640 / THD −53.6dB）。
- 修正(設定側): 設定を `configureProbeFlatEQ(e)`（全 band 無効・total 0dB・AGC off・saturation 0 = EQ チェーン恒等変換）+ Parallel structure + engine `setAutoGainStagingEnabled(false)` に統一。
- 修正(criterion 側): PASS 窓をモード別に分離 — bare/ir は従来の bypass 基準 `[0.880,0.897]` を維持、eq は **EQ-on identity 実測基準 `[0.486,0.496]`**（中心 0.4912）を採用。判定ロジックは 1 本化（`verdictPass`）。
- **OPEN（未帰属）**: EQ-on 経路は bypass 経路（0.891 = −1dB passthrough）に対して **−5.17dB 減成**（0.4912 = 0.891 × 0.5513）を示す。全 band 無効・total 0dB（`eqParams.totalGainDb` 既定 0.0f も確認）・saturation 0・EQ AGC off・engine staging off の状態で二連続実測 0.4912 一定 → EQ パス固有の構造減成と推定されるが内訳は未特定。criterion は regression tripwire（EQ チェーン構造変化の検出）として機能する。帰属は EQ DSP オーナーの後続作業。

## 2. 検証（実施済み・全 PASS）

| 検証 | 結果 |
| --- | --- |
| fail-closed parser | `--buzz-eq=1` → `[BUZZ] FAIL: --buzz-eq expects 'on' or 'off' (got '1')` + exit 2。`--buzz-eq=on` は parse 通過後 harness 初期化へ進行 |
| Release 再ビルド | exit 0（commit `3ea3208` 後の stamp M2 欠陥 hit → 既知手順どおり stamp 再作成で回復） |
| rigcheck=bare | `ratio=0.8846 thd=-149.3dB -> PASS`（従来基準維持・不変を確認） |
| rigcheck=eq | `ratio=0.4912 thd=-152.7dB -> PASS`（EQ-on identity 校準窓 `[0.486,0.496]`）+ mirror 状態行: `eqBypassReq=0 eqBypassActive=0 convBypassReq=1 convBypassActive=1 seq=5`（Phase 2-2 mirror 遷移の継続観測点として機能） |
| LC flip capture（`--buzz-flip-lc=1`・eq=on/conv=on/sr=384000/sine50） | `jumpPre=0.000079 jumpTr=0.000079 rmsPre=0.067760 rmsTrMax=0.095163 ampPre=0.096084 ampPost=0.088888 nonFinite=0 check=1` — **rmsTrMax が実値を出力**（fix 1+3 の効果）・窓が flip 実位置に一致（effectiveCaptureRate=880,640Hz で flipIndex 誤差 <0.5%）・check=1 は LC 変化に伴う DC シフトの正当な要確認フラグ |
| capture | `acc_11317b_C2_lc_fixed.csv` |

## 3. 状態

- 実装: **完了**（test-only・`src/tests/AudioEngineHarness/{TransitionMetrics.h,BassBuzzMeasurement.cpp}` のみ）
- 検証: **完了**（上表）
- commit: 本 commit に包含（`test(harness)` 系列・Phase 2-2 commit `3ea3208` とは別管理）。監査記録への受入判定追記（§9）は `docs(work113)` 系列で別 commit。
- ユーザーレビュー判定（2026-09-19）: 全項目 PASS・`--buzz-order` は OBSERVATION/OPEN（enum-like のため既存互換優先で未変更・将来的に `parseProcessingOrder()` 等の別 cleanup 候補）・**EQ-on −5.17dB は OPEN / EQ DSP owner へ引継ぎ**（本 cleanup の責務は測定器の信頼性回復であり原因究明ではない・rigcheck=eq 校準値は regression tripwire として固定）
- 評価注記: capture-rate binding は平均実効レートによる bind であり非一様 callback rate の完全補正ではない → 現行 transition probe の目的には十分。timestamp-based capture が必要になった時点で別課題
- OPEN 引継ぎ: EQ-on 経路 −5.17dB 減成の内訳未帰属（§1-4）— EQ DSP オーナーの後続作業
- 観察（変更せず記録のみ）: `--buzz-order=` の `(=="etc")?1:0` も silent-fallback 系だが boolean ではなく列挙系のため既存互換を優先し未変更
