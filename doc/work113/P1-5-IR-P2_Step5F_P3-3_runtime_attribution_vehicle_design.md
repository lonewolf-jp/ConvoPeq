# P1-5-IR-P2 — Step 5-F / P3-3: Runtime Attribution Vehicle Design Audit（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-3）
- **種別**: 完全 read-only / 実験設計監査。実行・build・source変更なし。
- **目的**: 実験ではなく「既存runtime機構だけで何を独立化できるか」を確定する。
  帰属軸 R（world state）/ H（DSP history）/ N（nonlinear state）を同時変動させない
  最小 vehicle の設計可否を E1/E2/E3 に分類し、V-A/B/C/D の1つに判定する。
- **前提**: P3-1-C=STOP / R1=COMPLETE(C4) / P3-2=COMPLETE(M3) /
  Reverse=NOT EXECUTED / P3-1-D=NOT STARTED / H-B=OPEN・対象外。

---

## 1. State Freeze

```text
HEAD                         1e9e63e3（1e9e63e34bed7adb9342ebc81259ded9689fc48a）
kP15FullMatrix               true（:43・:746 adb順序 Forward のまま）
CONVOPEQ_CORRECT_POLYPHASE_GAIN OFF
A3 binary / backup           b2c39a9a0e3b1fe3 / b2c39a9a0e3b1fe3
production / CMake / JUCE / settle-sleep / measurement-semantics  0 diff
```

禁止事項（すべて遵守・0件）:
`source変更・test source変更・build・Forward/Reverse再実行・logger追加・CLI追加・
settle/sleep変更・limiter reset追加・crash dump変更・H-B再実行`。

---

## 2. Attribution question

P3-2（M3）で残った問い：`Δ=+1.4759 dB`（g0/os1/-20→-6）は測定器誤差ではない（M1棄却）。
しかし以下は未確定のまま交絡している：

```text
amp（入力レベル）
＋ previous case（p15eq-os8 → −20 → −6 の履歴位置）
＋ OS（8→1 遷移は−20の前のみ）
＋ rebuild timing（dispatch／merge／publish の成否と時刻）
＋ crossfade（旧core混合の有無と窓）
＋ convolver history（前ケース tail の加算）
```

識別軸（P3-3 §1）：

```text
Axis R — Runtime world state（R0 新world安定／R1 crossfade中／R2 stale world）
Axis H — DSP history（H0 前history排除／H1 残留の可能性）
Axis N — nonlinear state（N0 limiter/clamp排除可／N1 排除不可）
```

R/H/N を同時に変える実験は設計しない。現行 P1CHAR は全軸同時変動のため帰属不能である。

---

## 3. Runtime-world state model（B1 結論）

### 3-1. publication 段階と sequence/counter

source確定の段階（RebuildDispatch／RuntimeBuilder／Publication／AudioBlock）：

```text
request（submitRebuildIntent: Structural/Snapshot, Replaceable）
→ build（RuntimeBuilder::build → 新DSPCore生成 :469）
→ publish（publication sequence  bump・world公開）
→ old world（fading core として保持・RCU）
→ crossfade（等電力 old/new mix・§4）
→ stable new world（ramp完了→Timer解決→endCrossfade→retire）
```

harness観測可能な量（`runCase` が既読のもの）:

- `getLastCommittedPublicationSequence()`（最終commit sequence・:411取得）
- `getPublicationBacklogCount()`（未処理 backlog・:414/:93 ポーリング）

`[PUBLISH]` / `[REBUILD_TELEMETRY]` / `[CONV_IR]` は本vehicle非出力（stderr logger未設置・P3-1-A確定）。
よって harness が読めるのは上記2カウンタのみである。

### 3-2. `waitWorldPublished()` の保証範囲（P3-3 §4 回答）

定義（:86-103）：`cur != before && backlog==0` を30 s ポーリング。成功は以下を保証する：

```text
保証する：ポーリング時点で sequence が seqBefore から進み、backlog が 0 であること
保証しない：
  (a) timeout時（false を返すが runCase は無視して capture へ進む :417）
  (b) 進んだ sequence が「今回の」configureChain の成果であること
      → P3-1-B §2-2 の merge により今回 intent が dispatch されず、
        前ケースの build で sequence が進んだ場合も「成功」になる
  (c) crossfade ramp の完了（publication ≠ 混合終了。§4）
  (d) DSP内部 settle（convolver tail・limiter release・§7-2系）
```

`sleepPump(800)` は wall-clock 800 ms の message pump であり、
stable-world の判定機構ではない。これを安定保証と解釈してはならない（指示どおり）。

### 3-3. merge-no-dispatch 時の振舞い（本設計の核心・source確定）

P3-1-B §2-2 の規則：同一 signature＋Replaceable＋debounce窓内 →
`REBUILD_MERGED` を emit して return（dispatch なし）。
この場合 publish は発生しないため、後続の `waitWorldPublished` は
`cur==before` のまま **30 s timeout（false）** する。
現行 `runCase` は戻り値を無視するため、timeout は WARN すら出さず capture へ進む。

p15ir `-20 → −6` 境界への適用（source確定の対応表）：

| 境界 | configureChain差 | intent |dispatch| wait帰結 |
| --- | --- | --- | --- |
| p15eq-os8 → −20 | os/eqBoost/convBypass/IR 変化 | 有（複数） | 有 | 成功の可能性（build次第） |
| −20 → −6 | **差なし**（ampDbはcapture側でパラメータ外） | 同一signature → **merge** | **なし**（窓内なら） | **timeout確定**（成功しえない） |

すなわち −6 行は、同一パラメータの world（＝−20 行の world、前方依存の timing 付き）で
処理されることが source 上の既定動作である。R軸は本境界では作動しない可能性が高く、
行間差の候補は H（history/tail）・N（limiter blind spot）に絞られる構造である。
ただし merge は debounce 窓条件付きのため「確定」ではなく「既定動作」と表現する
（窓値の実測は本 audit 範囲外・値の特定はしていない）。

---

## 4. Crossfade source audit（B2 結論）

実装（AudioBlock.cpp:286-477・AudioEngine.h:4154-4195・CrossfadeRuntime.h）：

```text
old DSPCore（fading・RCU保持）／ new DSPCore（active）
mix開始条件：armCrossfadeIfPending（pending && isPending・:4162）→ canCrossfade
  ＝ (fading!=null || dry) && ramp smoothing中 && buffer確保 (:384-388)
fade duration：world overlap の fadeTimeSec（既定 0.030 s・CrossfadeRuntime.h:218,130）。
  48 kHz×0.030 s ≒ 1440 samples ≒ 2.8 blocks（512/block）
fade完了条件：ramp remaining 1→0 エッジ → notifyRampComplete (:465-466)
  → Timer が解決 → endCrossfade → retire（ramp完了≠retire許可・D129契約）
captureとの位置関係：
  captureは configureChain 後の warm4＋cap12 blocks の先頭12を取る（tap :192-194）。
  解析窓はその末尾4096＝blocks 4–11。
  よって publish直後の ramp（〜3 blocks）は解析窓外に落ちる幾何だが、
  capture開始後の publish（build遅延時）は blocks 4+ に混合が入る余地がある。
  どちらが起きたかは本vehicleのログでは判定不能（PUBLISH非出力）。
```

結論：crossfade混入は source 上 reachable だが、p15ir両行での有無は現evidenceで確定不能。
混入窓（〜3 blocks）と解析窓（blocks 4–11）の幾何関係は V2 設計の根拠とする（§8）。

---

## 5. Convolver-history source audit（B3 結論・推測なし）

`RuntimeBuilder::build`（RuntimeBuilder.cpp:454-509）の確定順序：

```text
新DSPCore生成（aligned_make_unique :469）
→ applyBuildSnapshot（metadata のみ・AudioBufferは含まない :471-472 コメント明記）
→ transferIRStateFrom（実IRデータ転送 :473）
→ prepare(...)（:474-480）
→ eq totalGain適用・IR形状契約照合（:485-509）
```

`transferIRStateFrom`（ConvolverProcessor.h:1269-1288）は IRState
（IR AudioBuffer・sampleRate・attenuation・freqPeak・blockSize）のみを
`updateIRState` で複写する。overlap／delay-line等の runtime 履歴の複写経路は存在しない。

| State | new DSPCore | transfer対象 | source確定? |
| --- | --- | --- | --- |
| IR coefficients | 転送される | YES（updateIRState） | YES |
| convolution overlap／delay line | 転送されない | NO（経路なし） | YES（非転送まで確定。初期値のzero/clearは本audit未追跡→UNKNOWNとして残す） |
| limiter envelope | fresh（member-init 1.0・§6） | NO | YES |
| crossfade state | world側（CrossfadeRuntime） | —（core外） | YES |

結論：旧coreの信号履歴は新coreへ**移らない**（source確定）。
よって rebuild完了後の新coreに旧tailは存在しない。
一方 rebuild未完了／merge-no-dispatch時（§3-3）は旧core＋旧historyのまま capture が進む。
両分岐のどちらが各行で起きたかは現ログで確定不能 → H軸は M3 候補として残る。

---

## 6. DSPCore / limiter lifecycle audit（B4 結論）

```text
new DSPCore → SimplePeakLimiter はメンバ default 構築（envelope=1.0・SimplePeakLimiter.h:89）
→ prepare(sampleRate,100.0) は releaseCoeff のみ設定（:19-24。reset()を呼ばない）
→ reset() の呼出しは全sourceで0件（P3-1-A確定・維持）
→ first process から envelope 追跡開始（attack即時・release 100 ms）
```

`new world で envelope=1` の保証範囲：
rebuild が完了し新coreが active になった場合に限り成立する。
merge-no-dispatch・publish未完・crossfade混合中は旧envelope系の出力が混入しうる。
「だから今回のΔはlimiterではない」という結論は出さない（指示どおり）。

---

## 7. Existing harness capabilities（E判定の土台）

- `--p1-char` は引数なし（argc/argv無視・Step5_A1確定）。section選択CLIは存在しない。
- harness観測可能：P1CHAR行（gainDb系・counter系・§P3-2-§6）＋2カウンタ（§3-1）。
  `PUBLISH`/ramp状態/limiter envelope の観測手段は存在しない。
- sc0/sc1ペア（runPair）は同一条件の非線形stage on/off を既存出力で比較可能だが、
  sc1はsc0＋reloadの後続位置のため履歴が交絡する（P3-2 §11相当・partial制約として記録）。

---

## 8. V1–V4 vehicle design（実行ではなく設計評価）

| Vehicle | previous state | target | 目的 | E判定 |
| --- | --- | --- | --- | --- |
| V1 同一状態で−20→−6 | 現行どおり | −20→−6 | 現象再現 | **E1**（既存 `--p1-char` as-is。70分＋p15hrnl crash域通過＝非最小・交絡全残）／最小形はE2 |
| V2 新world安定化 | publish成功＋ramp終了を待てる状態 | −20→−6 | crossfade影響分離 | **E2**（§10の最小変更1〜3。production 0） |
| V3 historyなし | 新core確定（旧history非継承・§5） | −20→−6 | convolver tail影響分離 | **E2**（V2と同一機構で達成。fresh process両行も同等・§10） |
| V4 順序反転 | −6→−20 | −6→−20 | order dependence | **E2**（`:746` 1 token・P3-1-Bで実装可能性確定済み） |

V2/V3収束点：§3-3・§5より、publish成功の確保は同時に (a) crossfade rampの時間的余裕と
(b) 新core（旧historyなし）の確定を与える。すなわち**一つの test-only 機構でR軸とH軸を同時安定化**できる。
N軸は sc0/sc1ペア（既存出力）で観測継続する。

---

## 9. Existing-only feasibility（V-A 判定材料）

既存harnessだけでR/H/orderのいずれかを独立化できる vehicle は存在しない：

- `--p1-char` 全行程は全軸同時変動（§2）のまま。A3/Forwardの2 run一致はrun間再現性であり軸分離ではない。
- sc0/sc1比較はN軸のpartial制約に留まる（履歴交絡・§7）。
- 2カウンタの値は現行vehicleではログに出ない（読めるが記録されない）。

よって **V-A（Existing-only feasible）は不成立**。

---

## 10. Test-only change requirements（V-B 仕様固定・未実施）

production 0 diff を保つ最小変更案（固定のみ・実装しない）：

```text
1. :746 adb順序（V4用・1 token）。V2/V3単独には不要。
2. runCase で waitWorldPublished の戻り値をhonorする（bounded retry/timeout計上）。
   結果（success/timeout・merge-no-dispatch示唆）をP1CHAR行へ記録する。
   ※ 行field追加は test TU のlog変更であり、次gateの明示承認を要するものとして記録。
3. publish成功後の bounded post-settle（ramp〜3 blocks＋余裕の時間的カバー）。
   ※ settle-adjacent変更であり、現行P3 STOP条件と抵触しうる旨を明記して次gateへ。
4. g0/os1ペア限定実行（hardcode gate・70分→数分・p15hrnl crash域除外）。
   ※ section選択のhardcodeであり、新規CLIは作らない。
```

推奨次実験（単一・最小・次gate判断用）：fresh process × g0/os1ペア限定 ×
publish-honored settle を F（−20→−6）とR（−6→−20）の2 runに適用し、
R安定・H fresh の同条件で order を比較する。sc0/sc1は維持しN軸観測を継続する。

---

## 11. Production instrumentation requirements（V-C 判定材料）

- 上記§10で実行に不足はないため、production計装は**不要**と判定する。
- 証明級（per-caseのramp完了・world世代の確定記録）には ramp状態・world世代の
  accessor/logが要るが、それは実行要件ではなく証明要件である。
  よって **V-C（instrumentation required）は不成立**。P3-1-Dは開始しない。

---

## 12. Gate classification

```text
V-A（Existing-only feasible）: REJECTED（§9）
V-B（Test-only change required）: ADOPTED（§10の仕様固定・実装はしない）
V-C（Runtime instrumentation required）: REJECTED（§11）
V-D（Structurally underdetermined）: REJECTED（test-onlyで分離可能・§8）
```

## 13. Recommended next experiment

V-B の仕様（§10）を承認gateへ諮り、承認後に単一最小vehicle（§10推奨）を1回だけ実行する。
本 P3-3 では設計確定とSTOPで終わる。Reverse単独の先行実行は推奨しない
（R/H未安定のままでは order と state が分離できない・P3-3 §7遵守）。

## 14. Prohibited actions（遵守記録）

```text
source変更 0 / test source変更 0 / build 0 / Forward再実行 0 / Reverse再実行 0
logger追加 0 / CLI追加 0 / settle/sleep変更 0 / limiter reset追加 0
crash dump変更 0 / H-B再実行 0 / p15hrnl含有run 0（70分vehicle未実行）
```

STOP条件に触れる事項は発生しなかった（本 audit は read-only で完結）。
H-B との因果主張なし。§7解釈制約（RuntimeBuilder新core／publish戻り値無視／
envelope持続の肯定・否定禁止）を維持する。
