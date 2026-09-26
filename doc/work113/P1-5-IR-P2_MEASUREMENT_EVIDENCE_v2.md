# P1-5-IR-P2 — Frozen Matrix Measurement Evidence（Step 4-A〜4-F）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（再開ラン）
- **性質**: 測定のみ（production / CMake / harness / default / calibration / settle 変更 0）
- **前提**: Step 3-J PASS（H-A CLOSED）。本ランは 3-J の teardown 修正後の binary で実施。
- **raw evidence**: `tmp/p1_5_ir_p2j/{A1,A2,A3,B1,B2,B3}_lvl*.log`（6 run 完全ログ）

---

## Step 4-A — State Freeze（read-only・変更 0）

| 項目 | 値 | 判定 |
| --- | --- | --- |
| HEAD | `1e9e63e3` | ✅ |
| production source diff | 0 | ✅ |
| CMake diff | 0 | ✅ |
| JUCE source diff | 0 | ✅ |
| default / calibration diff | 0 | ✅ |
| 新規 test-source modification | **0**（既存 M 3 ファイルは 3-J 以前からの持ち越し。3-J の変更は `BassBuzzMeasurement.cpp` の `static` 1 token 除去のみ） | ✅ |
| settle / sleep / retry 変更 | 0 | ✅ |
| measurement semantics 変更 | 0 | ✅ |
| binary | `build/Release/AudioEngineHarness.exe` SHA-256[:16] = `2054d970da4ec91c`（= 3-J PASS 版） | ✅ |
| flag cache | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF` | ✅ |

### Binary / IR identity

| 項目 | 値 |
| --- | --- |
| exe SHA-256[:16] | `2054d970da4ec91c`（3-J 適用版） |
| IR file | `tmp/p15_ir_g0.wav`（8236 bytes） |
| IR SHA-256[:16] | `2e526f4de5a97afb`（**先行 P2 文書と一致**＝同一資産） |
| 実行 cwd | `C:\VSC_Project\ConvoPeq` |

### 実行時の注意（IR 解決の silent fallback）

`BassBuzzMeasurement.cpp:1697-1700` は IR 解決に silent fallback を持つ:

```cpp
juce::File irFile(opt.irPath);
if (opt.irPath.empty() || !irFile.existsAsFile())
    irFile = juce::File::getCurrentWorkingDirectory().getChildFile("sampledata/impulse.wav");
```

- 本 P2 の 6 run は **forward-slash 形 `--buzz-ir=tmp/p15_ir_g0.wav`** で起動し、全 run の先頭行が
  `[BUZZ] IR: C:\VSC_Project\ConvoPeq\tmp\p15_ir_g0.wav` を出力（= **凍結 IR が実際に使用された**ことを確認）。
- 記録: 3-J の V1/V3 実行時は backslash 形が shell で解決されず本 fallback が働き
  `sampledata/impulse.wav` が使われた。**3-J の teardown 判定（exit code / heap corruption）には影響しない**
  （teardown 経路は IR 非依存）が、証跡の正確性のためここに記録する。本 P2 では fallback は発生していない。

---

## Step 4-B — Frozen Matrix（6 run）

条件（P1 凍結条件をそのまま使用・追加旗なし）:

```text
--buzz-os=1 --buzz-probe=real --buzz-conv=on --buzz-eq=off --buzz-order=cte
--buzz-ir=tmp/p15_ir_g0.wav --buzz-probe-level=<0.1 | 0.5011872>
--buzz-quiet は未指定（既定 150000 ms）
```

| run | level | 相当 | exit code |
| --- | --- | --- | --- |
| A1 | 0.1 | −20 dBFS | 0 |
| B1 | 0.5011872 | −6 dBFS | 0 |
| A2 | 0.1 | −20 dBFS | 0 |
| B2 | 0.5011872 | −6 dBFS | 0 |
| A3 | 0.1 | −20 dBFS | 0 |
| B3 | 0.5011872 | −6 dBFS | 0 |

- run order = A1→B1→A2→B2→A3→B3（§4.B 順序1 相当）。
- 全 6 run exit code 0。**crash / heap corruption / ASan error / capture corruption = 0**。

---

## Step 4-C — Observables（各 run）

**全 6 run が完全決定論**: A1≡A2≡A3、B1≡B2≡B3（全観測値 bit 一致）。

| observable | A1/A2/A3（−20 dBFS） | B1/B2/B3（−6 dBFS） |
| --- | --- | --- |
| input level | 0.100 | 0.5011872 |
| **outPeak** | **0.005327** | **0.026701** |
| energy | 0.00009524 | 0.00239224 |
| peakIdx | 1028 | 1028 |
| sigTaps / outsideTaps | 14 / 7 | 465 / 458 |
| headFrac | 0.9837 | 0.9837 |
| verdict | BROADENED | BROADENED |
| THD（`[PROBE_METRICS]`） | 57.45 / 61.84 / 58.07 dB（run 間変動） | 60.20 / 64.78 / 64.94 dB（run 間変動） |
| IR length（runtime） | 96000 | 96000 |
| IR sample rate | 192000 | 192000 |
| IR block | 1024 | 1024 |
| IR generation | gen=14（`[CONV_IR] IR transferred`） | gen=14 |
| publicationSeq（final） | seq=8 gen=8 | seq=8 gen=8 |
| CONV generation（final） | generation=9 irLoaded=1 irLen=96000 | generation=9 irLoaded=1 irLen=96000 |
| IR finalized | `waitIrFinalized(300000)` 通過（FAIL なし） | 同左 |
| OS factor | **osFactor=1**（pin 成立） | osFactor=1 |
| processingRate | 192000.0 | 192000.0 |
| L0 geometry | `part=1024 numIR=23 numParts=32 fft=2048 imm=1 irLen=96000` / `slot=22 peak=0.500000 bin=0` | 同左 |
| REBUILD_TELEMETRY structural 件数 | 35 | 35 |
| settle | `[PROBE] quiet 150000 ms for DSP-side IR rebuild...` | 同左 |
| exit code | 0 | 0 |

補助観測（A1・B1 で同一）:

```text
[IR_RATE_GEN] gen=0 sourceSr=48000 targetSr=192000 actualSr=192000 sourceLen=1 convertedLen=1 ratio=1.0000 resampled=yes srcArgmax=0 convArgmax=0
[IR_TAIL_GEOM] gen=0 loadedSr=192000 loadedLen=1 targetLength=96000 copySamples=1 fadeSamples=0 fadeStart=1 fadeEnd=1 fadeMs=0.0000 fadeDisabled=0
[IR_CHAIN] A_resampled n=1 sr=192000 peak=9.82491042e-01 ...
[IR_CHAIN] F_scale scaleFactor=0.50897711 phaseMode=0
```

- 凍結 IR `p15_ir_g0` は **1 サンプルの delta IR**（`loadedLen=1`）で、`targetLength=96000` へ pad される。
- `sigTaps` が 14 → 465 と増えるのは、`BassBuzzMeasurement.cpp:2394` の有意タップ閾値が
  **絶対値 `|v| > 1.0e-4`** であるため。出力が線形に 5.0119 倍されれば固定閾値を超えるサンプル数が増えるのは
  期待どおりであり、**異常ではない**（BROADENED 判定も両 level で同一）。
- THD のみ run 間で変動（計器系ジッタ）。先行 P2 文書の `ultra` 変動と同型。

### 時系列（A1・`[GEOM before-capture]` を含む）

```text
:648  [CONV_IR] transferIRStateFrom: IR transferred ch=2 len=1 sr=192000.0 block=1024 gen=14
:714  [CONV_STATUS] generation=9 irLoaded=1 irLen=96000 convBypass=0 sr=192000.0 osFactor=1 processingRate=192000.0
:715  [PUBLISH] seq=8 gen=8 worldId=8
:719  [BUZZ] probe-irload: world seq 6 -> 8 committed
:720  [PROBE] quiet 150000 ms for DSP-side IR rebuild...
:738  [GEOM before-capture] IR(sr=192000 block=1024 len=96000 gen=14) UIprep(sr=192000 block=1024)
      ENG(buildRate=192000 irLen=96000 block=1024 layers=2 L0part=1024 L0numIR=23 L0numParts=32 L0fft=2048 ring=8192)
:740  [PROBE] kind=real level=0.100 outPeak=0.005327 ...
```

---

## Step 4-D — 1-run-lag 判定（凍結定義に基づく）

凍結定義（P1 §6b・4 点チェック）に従い、**`publicationSeq` の前進のみでは判定しない**。
Runtime IR の同定は `[CONV_IR]` / generation / `[IR_TAIL_GEOM]` / `[GEOM before-capture]` の相関で行った。

| 4 点 | 観測 | 分類 |
| --- | --- | --- |
| requested IR = X | `tmp/p15_ir_g0.wav`（`[BUZZ] IR:` 行で確認） | OBSERVED |
| UI finalized = X? | `waitIrFinalized(e, 300000)` が FAIL せず通過（全 6 run） | OBSERVED（間接） |
| Runtime IR = X? | `[CONV_IR] IR transferred ch=2 len=1 sr=192000.0 block=1024 gen=14`（**同一 run 内**） | OBSERVED |
| output = X? | `[GEOM before-capture] IR(... gen=14)` → 直後に `[PROBE]` 測定。gen 一致 | OBSERVED |

- **run N の観測に Runtime(Y)（= 別 run/別 IR の状態）が現れる事象は全 6 run で 0 件**。
- 各 run は独立プロセス起動であり（P1 §5）、run 間の状態持ち越しは構造上存在しない。
- Runtime IR は **同一 run 内で測定前に確定**（IR transfer → CONV_STATUS irLoaded=1 → PUBLISH seq=8 →
  150000 ms quiet → `[GEOM before-capture]` → 測定）。
- したがって:

```text
1-run-lag = NOT REPRODUCED（凍結 P2 vehicle 内）
```

- 先行 P2 文書 §12（rigcheck=irwet1 flow 内でも NOT REPRODUCED）と整合。

---

## Step 4-E — 1.48 dB 判定

### 測定（level 正規化 transfer gain）

入力は `SignalGen::next() = nextRaw() * level_`（`BassBuzzMeasurement.cpp:185`）で
**厳密に `probeLevel` 倍**されるため、`outPeak / level` は level 非依存の実効 transfer gain として比較可能。

| 量 | −20 dBFS | −6 dBFS |
| --- | --- | --- |
| transfer gain = outPeak/level | 0.05327000 | 0.05327550 |
| gainDb | −25.47035 dB | −25.46945 dB |
| **Δ(B−A)** | **+0.00090 dB** | |

- 丸め不確かさ（outPeak 6 桁）から Δ の範囲は **−0.00008 〜 +0.00188 dB**（上限 ≈ 0.002 dB）。
- **独立検証（energy・level² 則）**: energy 比 25.118018 対 期待 (level 比)² = 25.118861 →
  残差 **−0.00015 dB**。ピーク・エネルギー両指標が level 完全比例を支持。
- 参照値 1.48 dB は測定上限の **約 789 倍**。

```text
1.48 dB = NOT REPRODUCED（凍結 P2 probe matrix・各 level 3 run・完全決定論）
```

### 分類の維持（P0/P1 の凍結を変更しない）

```text
1.48 dB（記録値）      = OBSERVED（先行記録として維持。本ランで新規帰属は与えない）
mechanism              = INFERRED（不変）
IR internal gain error = UNKNOWN（不変）
```

- 本ランは **1.48 dB を「IR gain error」と呼ばない**。`[IR_CHAIN] F_scale scaleFactor=0.50897711` と
  `[L0_WRITE] slot=22 peak=0.500000` は **OBSERVED な geometry/scale 事実として記録するのみ**で、
  gain 帰属・補正・normalization の検討には入らない（P1 §9-5 遵守）。

### 3-J teardown 修正との関係（明示的切断）

- **3-J は IR DSP の変更ではない**（test-only の lifetime 1 token 変更）。本ランで anomaly が
  再現しなかったことを **3-J が IR 原因を修正した証拠として解釈しない**。
- 3-J の効果は H-A（JUCE static teardown double-free）の閉止に限定される。

### ⚠ §9-2 caveat（本判定に付随する最重要留保）

P1 §9-2 は「整定時間・sleep・settle の変更によって anomaly が消えた状態での計測
（bug disappeared と protocol masked の区別が不能）」を最重要禁止としている。本ランは
**settle を変更していない**（既定 150000 ms を使用）ため手続き的には適合する。

しかし、参照値の出所と本 vehicle の間には **settle 構造の差**が OBSERVED されている:

| vehicle | IR finalize 待ち | OS 変更後の settle |
| --- | --- | --- |
| 本 P2 probe flow（`--buzz-probe=real`） | `waitIrFinalized(e, 300000)` **あり** | `[PROBE] quiet 150000 ms` + `waitBacklogZero` + `waitWorldPublished` |
| 参照値の出所（`--p1-char` / `runCase`・`P1PolyphaseGainCharacterization.cpp:404-418`） | `ensureTestIr` は `isIRFinalized()` poll を持つが **`runCase` 内に `waitIrFinalized` なし** | `waitBacklogZero` + `waitWorldPublished` + **`sleepPump(800)`（800 ms）のみ** |

- さらに先行文書 `p1_5_adoption_characterization_20260922.md` §5 は、参照値
  （amp=−20 → −14.5035 / amp=−6 → −13.0276）を出した測定自体を **「測定不成立（invalid）」** と
  判定している（理由: IR load/finalize と OS レート変更に伴う再準備の整定不足・「本 TU にはそれが無い」）。
- したがって本ランの NOT REPRODUCED は、**「anomaly が存在しない」と「settle により protocol masked」を
  区別できない**。この区別は本ランの範囲外であり、**P3 の root-cause attribution に直ちに進む根拠にならない**。

---

## Step 4-F — H-B 分離（記録のみ・修正なし）

全 6 run で以下を観測（**記録のみ・その場で修正に入っていない**）:

```text
[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — residual intents may remain in System 1 queues   （1 件/run）
routerPendingRetire=2                                                                                                            （全 run 同値）
```

- H-B が引き起こした **process crash / heap corruption / ASan error / capture corruption は 0 件**
  → P2 の STOP 条件に該当しない。**4-F = PASS（分離維持）**。

```text
H-A = CLOSED
H-B = OPEN
IR measurement = independent observation
```

---

## Step 4 判定

| Step | 結果 |
| --- | --- |
| 4-A State Freeze | **PASS** |
| 4-B 6-run matrix | **COMPLETE**（6/6 run・exit 0・完全決定論） |
| 4-C evidence | **COMPLETE**（観測表 + 時系列 + raw log 6 本） |
| 4-D lag classification | **COMPLETE — 1-run-lag = NOT REPRODUCED** |
| 4-E 1.48 dB | **COMPLETE — NOT REPRODUCED**（Δ=+0.0009 dB・上限 0.002 dB・§9-2 caveat 付き） |
| 4-F H-B separation | **PASS** |

```text
Step 4 = COMPLETE（PASS、ただし 1.48 dB の解釈に §9-2 caveat を付す）
production fix / IR geometry fix / gain compensation / settle 変更 / P2 harness 変更 = 0
```

---

## 次段階の判断（P3 へ進むか / 追加 read-only audit か）

**推奨: 追加の read-only audit を先に行い、P3 root-cause attribution にはまだ進まない。**

根拠（OBSERVED の突き合わせ）:

1. 1.48 dB は凍結 P2 vehicle（settle 完備）で **NOT REPRODUCED**（決定論・3 run/level）。
2. 1-run-lag も **NOT REPRODUCED**（Runtime IR は同一 run 内で測定前に確定）。
3. 参照値の出所測定は先行文書自身が **「測定不成立（invalid）」** と判定済み。
4. 両 vehicle の settle 構造差は OBSERVED だが、**§9-2 により「消えた」と「masked」を区別できない**。

したがって必要な追加作業は次の 2 点（いずれも **read-only / 既存条件の再実行のみ**）:

- **A-1（推奨・protocol-clean）**: 参照値の出所 vehicle（`--p1-char`）を **条件変更なしで** 現 binary 上で
  再実行し、anomaly が **その vehicle 自身で再現するか** を確認する。settle を触らないため §9-2 に抵触しない。
  - 再現する → 同一 vehicle 内で anomaly が生存。P3 attribution の対象が確定する。
  - 再現しない → anomaly は現 binary で観測不能。P0/P1 の「OBSERVED（記録値）」を
    **NOT REPRODUCED（現 binary）** へ更新する判断材料になる。
- **A-2（source read-only）**: 両 vehicle の settle 構造差と、参照値測定の手順をソースから完全に確定する
  （本レポート §4-E の表を一次資料で裏付ける）。**settle 次元の A/B は §9-2 の禁止領域に触れるため、
  実施するなら明示的なユーザー承認が必要**。

**P3（root-cause attribution）に進む条件**: A-1 で anomaly が同一 vehicle 内に再現し、
かつ A-2 で手順差が確定した時点。それまでは P3 の対象が「現行 vehicle で再現しない事象」であり、
attribution の土台が成立しない。

**禁止の継続**: production fix / IR geometry fix / gain compensation / normalization /
settle 変更 / P2 harness 変更 / H-B 修正。
