# P1-5-IR-P2 — Measurement Evidence (BLOCKED / NOT PASS)

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2
- **判定**: **STOP(BLOCKED) — P2 PASS にせず停止**。理由: **protocol deviation が発生**(§15)。凍結 matrix(同一 IR / OS=1x / SC=OFF / −20・−6 dBFS / 3+ run)を既存 flags だけで表現できなかった。harness 拡張は実施していない。
- **raw evidence**: `tmp/p1_5_ir_p2_raw/`(14 run の完全ログ + 実行スクリプト `tmp/p1_5_ir_p2_runs.bat`)

---

## 1. Binary identity

| 項目 | 値 | 分類 |
| --- | --- | --- |
| 実行 exe | `build\Release\AudioEngineHarness.exe` | — |
| exe SHA-256[:16] | `d1fd59789f43afee` | OBSERVED |
| CMakeCache | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF` | OBSERVED |
| 構成 | OFF K2 build(Step 3 由来) | OBSERVED |

## 2. IR asset identity

| 項目 | 値 | 分類 |
| --- | --- | --- |
| IR file | `tmp/p15_ir_g0.wav`(8236 bytes) | OBSERVED |
| IR SHA-256[:16] | `2e526f4de5a97afb` | OBSERVED |
| 実測 geometry | **delta IR: loadedLen=1 / copySamples=1**(ピーク 9.824e-01 @ sample 0 → 192000×1s へ pad) | OBSERVED(`[IR_TAIL_GEOM] gen=0 loadedSr=192000 loadedLen=1 targetLength=192000 copySamples=1`) |

## 3. Command / flags(実行した組合せ)

```text
AudioEngineHarness.exe --buzz --buzz-rigcheck=irwet1 --buzz-probe=real --buzz-conv=on
  --buzz-eq=off --buzz-order=cte --buzz-ir=tmp\p15_ir_g0.wav --buzz-probe-level=<0.1|0.5011872>
  --buzz-out=<runId>.csv
```

- `--buzz-rigcheck=irwet1`: OS=1x pinning の唯一の既存旗(rigCheckMode[5]='1' → `setOversamplingFactor(1)`)
- `--buzz-probe=real --buzz-ir=…`: probe flow が実 IR をロードする組合せ(Probe flow :2204-2216)
- `--buzz-quiet` は**指定せず既定値 150000ms を使用**(settle 無変更)

## 4. Matrix A raw runs — 実行結果

- **Matrix A の実測は成立しなかった**。14 run すべてが **`--buzz-rigcheck=irwet1` の独自 FAIL gate で測定前に中断**:
  ```text
  [BUZZ] RIGCHECK(irwet1) sine50-6dBFS inPeak=0.5000 outPeak=0.1645 ratio=0.3290 thd=-150.9dB ultra=<-136.1〜-138.5>dB -> FAIL
  ```
  - 14 run 全てで ratio=0.3290 / thd=−150.9 dB が bit 一致(**決定論**)。ultra 指標のみ −136.1〜−138.5 dB の実行間変動(計器系ジッタ — Step 4 の計器系分離と同型)。
  - `[PROBE_CFG]`(probe flow 開始)に到達した run = **0 件**。CSV 生成 = **0 件**。
- 中断位置: rigcheck FAIL → 即 `releaseResources` → shutdown。**probe flow(CSV/level sweep)は 1 run も実行されていない**。

## 5. Matrix B raw runs

- 同構成のため Matrix B も rigcheck 中断 → **順序試験は未実施**。

## 6. `isIRFinalized` observations(rigcheck=irwet1 flow から取得できたもの)

- `waitIrFinalized(e, 300000)` は 14 run 全てで FAIL(timeout)を出さず通過 → **after finalize = true(OBSERVED・間接)**。
- 明示的な 4 時点記録(before load / after request / after finalize / before measurement)は buzz ランの既存ログには出力されない → **そのうち直接記録できたのは load 完了後の状態のみ・残り UNKNOWN**。
- `still=true` 系列(stale candidate)は観測対象外(本 composition では到達しなかった)。

## 7. publication sequence observations

`[PUBLISH] seq=<commit seq> gen=<worldId>`:
```text
[PUBLISH] seq=3 gen=3  → (OS変更 world)
[PUBLISH] seq=4 gen=4  → osFactor=1 の world([CONV_STATUS] gen=7 irLoaded=0 irLen=0)
[PUBLISH] seq=6 gen=6  → IR 適用後 world([CONV_STATUS] generation=8 irLoaded=1 irLen=192000)
```
- **seq 前進と IR-structural world の対応は取得できた**: IR finalize → `[DIAG] convolverParamsChanged: requestRebuild Structural hash=0x3a4f1ba2e464d6dd irName=p15_ir_g0` → `[CONV_IR] transferIRStateFrom: IR transferred ch=2 len=1 sr=192000.0 block=1024 gen=6` → `[CONV_REBUILD] rebuildAllIRsSynchronous: engine rebuilt len=1 ch=2 srcSR=192000.0` → `[CONV_STATUS] generation=8 irLoaded=1 irLen=192000` → `[PUBLISH] seq=6`。**telemetry/gen/irLen の対応が取れた行は IR-structural world として特定可能(OBSERVED)**。

## 8. IR geometry observations

- g0 test IR は **1 サンプルの delta IR**(nominal 0 dB ラベル・8236 bytes)。LoaderThread が 192k×1s へ padding(`[IR_TAIL_GEOM] targetLength=192000 copySamples=1`)。
- `[IR_RATE_GEN]` 第 1 ロード: `sourceSr=48000 targetSr=192000 ... resampled=yes` — UI processor の prepared SR(= device rate 192000)を target として処理。
- `[IR_CHAIN] F_scale scaleFactor=0.50894/0.50898(phaseMode=0)` — **IR スカラー利得はロード構成に依存**(P0 既知の構成依存性の再確認 OBSERVED)。
- `[L0_WRITE] geom part=1024 numIR=23 numParts=32 fft=2048 imm=1 irLen=192000 / slot=22 peak=0.500000 bin=0` ✓

## 9. Runtime IR identity observations

- **`[CONV_IR] transferIRStateFrom: IR transferred ch=2 len=1 sr=192000.0 block=1024 gen=6`** — UI 側 state(gen=6)→ RT convolver への転送が ch/len/sr/block/gen 付きで 1 行記録される(**P1 契約の L2 観測手段は実在・動作確認 OBSERVED**)。
- 転送後: `[CONV_REBUILD] rebuildAllIRsSynchronous: engine rebuilt len=1 ch=2 srcSR=192000.0` → `[CONV_STATUS] irLoaded=1 irLen=192000 osFactor=1` ✓
- 初回(placeholder world)は `[CONV_IR] no IR data to transfer`(複数回)= **OBSERVED**。

## 10. output DFT observations

- **Matrix A/B の DFT 出力は未生成**(全 run rigcheck 中断)。
- rigcheck=irwet1 自身の測定(sine50 @ −6 dBFS・inPeak=0.5000)のみ: `outPeak=0.1645 / ratio=0.3290 / thd=-150.9dB` — 14 run で完全再現(ultra 指標のみ変動)。

## 11. 1.48 dB reproduction status

- **未再測定**(matrix 未実行のため)。既知値は measurement-level delta としてのみ維持: amp=−20 → −14.5035 / amp=−6 → −13.0276 / Δ≈1.48 dB = **OBSERVED(記録値)** / IR internal gain error = **UNKNOWN(不変)**。
- 本 P2 の実測で 1.48 dB に新規の帰属を与えた記述は存在しない。

## 12. 1-run-lag status

- **NOT EVALUATED**(matrix 未実行)。ただし rigcheck=irwet1 の **in-process フローでは整定済みで測定に到達している**(IR world gen=8 の publish(seq=6)が rigcheck sine50 測定より前に完了 — ログ順 OBSERVED)→ 当該 vehicle では **1-run-lag = NOT REPRODUCED(この flow 内)**。irwet 経路の状態遅延問題は本 flow では観測されなかった。

## 13. OBSERVED / INFERRED / NOT REPRODUCED / UNKNOWN

| 項目 | 分類 |
| --- | --- |
| binary/IR asset identity | OBSERVED |
| `[CONV_IR]` transfer trace の動作(ch/len/sr/block/gen) | **OBSERVED — P1 契約の L2 instrumentation は現行 binary で機能する** |
| IR lifecycle トレース一式(`IR_RATE_GEN/IR_CHAIN/IR_TAIL_*/L0_WRITE/CONV_STATUS/PUBLISH/REBUILD_TELEMETRY`) | OBSERVED — P1 の観測契約は既存ログで充足される |
| 1-run-lag(rigcheck=irwet1 flow 内) | NOT REPRODUCED(本 flow では整定済みで測定に到達) |
| 1.48 dB | UNKNOWN(matrix 未実行) |
| rigcheck=irwet1 FAIL(ratio=0.3290)の機因 | **INFERRED: rigcheck 判定窓と WORK104 chain(staging=1/hdr=−6dB/makeup=+12dB/softClip=1 sat=0.1 — `[EQ_RTPATH]` world 行 OBSERVED)の不一致**。1.48 dB とは無関係 |

## 14. Anomalies

1. `[BUZZ] RIGCHECK(irwet1) ... -> FAIL`(14/14 run) — 判定窓と chain 構成の不一致。**既存挙動の観測であり、本 P2 で新規に生じた欠陥ではない**。
2. `[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete` — shutdown 時の残 intent 警告(observation only)。新規計測阻害なし。
3. 音声デバイスが **192000 Hz** で開かれている(`[DIAG] prepareToPlay: enter spb=1024 sr=192000.00`)— opt.sr 既定 48000 と不一致は既存 harness の device 選択挙動(matrix 条件への影響は OS 表現の問題と連動・下記 deviation)。

## 15. Protocol deviations(1 件 → P2 NOT PASS)

**Deviation-1(唯一): 凍結 matrix の条件組(OS=1x × SC=OFF × −20/−6 sweep × 同一 IR 反復)を既存 flags のみで同時に表現できない。**

- OS=1x pinning の唯一の既存旗 = `--buzz-rigcheck=irwet1` だが、この vehicle は (a) 自身の PASS/FAIL gate(ratio 判定窓)で **測定前に abort** する、(b) world の chain が WORK104 既定(staging/hdr/makeup/softClip — 出力は thd −150.9 dB で飽和実質無しとは言えるが、param state は softClip=1)であり、**probe flow(:2240 `setSoftClipEnabled(false)` が保証する SC=OFF とは別物)**。
- SC=OFF が構造的に保証される唯一の vehicle = Probe flow(`--buzz-probe=real` で実 IR 使用可)だが、**OS pinning の旗が存在しない**(OS=Auto 解決・`[OS] resolved` 行で記録は可能)。
- よって「OS=1x × SC=OFF × level sweep × 同一 IR 反復」の同時表現 = 既存 flags では不能 → **事前ゲートの停止条件に該当**。

**結論: P2 = STOP(BLOCKED)・NOT PASS。** raw evidence(14 log + bat)は `tmp/p1_5_ir_p2_raw/` に保存済み。production/CMake/harness/default/calibration/settle/commit の変更は 0。

### ユーザーへの次判断材料(本レポートの結論)

1. **harness 拡張の承認**(例: `--buzz-os=<n>` pinning 旗の追加 — test-only)→ P2 を再実行
2. **probe-only vehicle の受入**(OS=Auto の deviation を許容する matrix の再定義)→ P2 を再実行
3. matrix の条件見直し(例: rigcheck=irwet1 の判定窓と chain を整合させる既存経路の調査)

いずれも本 P2 の範囲外であり、ユーザー判断を待つ。
