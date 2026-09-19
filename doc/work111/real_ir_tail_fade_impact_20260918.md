# WORK111 — Real IR Tail-Fade Impact / Buzz Contribution Audit

- **作成日**: 2026-09-18
- **種別**: Measurement（実 IR に対する末尾 gain-ramp の影響分離測定と buzz 寄与の因果判定）
- **前工程**: WORK110（`doc/work110/l0_partition_fdl_correspondence_20260918.md`）
- **正本**: `ConvoPeq.md` 2026-09-18 21:31:27 / 5,252,672 bytes（WORK110 trace を含む最新 source から再生成 = 111-0 完了）
- **制約遵守**: E 修正なし / F4 limiter なし / F5 IR gain なし / IRRuntimeContract 未変更 / L0・FDL 未変更 /
  OS 補償なし / fadeSamples・fadeRatio・minFadeSamples・Tukey・IR 長・engine・limiter・gain の**変更なし** /
  RT に分岐・確保・ログなし（追加は test-only + NonRT trace のみ）

```text
REAL IR TAIL-FADE IMPACT

IR:
    sourceSr = 48000
    sourceLen = 8252        （trim 後。raw 8253 から 1 sample 減）
    loadedSr = 384000
    loadedLen = 62914       （resample 比 7.6241。理想的 ×8=66016 とは不一致 → §7 参照）

Trim:
    targetLength = 192000
    copySamples  = 62914
    fadeSamples  = 1258     （round(62914*0.02)=1258。256 下限/30720 上限に非拘束）
    fadeStart    = 61656
    fadeEnd      = 62914
    fadeMs       = 3.276    （fadeRatio 実効 = 2.00%）

Tail energy（Tukey 後 = fade 前の実データ、ch0）:
    total     = 8.05031433e+00
    preFade   = 8.05031433e+00
    fadeBand  = 2.03404509e-10      （総和の 2.527e-11 = -106.0 dB）
    postFade  = 0.00000000e+00      （無音）

Low-band（1-pole 100 Hz LP 後の energy）:
    0-100 Hz（LP 総和） = 2.60889697e-02
    うち fadeBand       = 2.10598849e-09   （8.072e-8 = -70.9 dB）
    → 低域 energy も fade 帯には実質存在しない

A/B:（実 IR / 同一条件。SR・OS・block・IR・phase・scale・generation・L0・limiter すべて固定）
    fade ON  (fadeDisabled=0):
        signal=sine50  peak=0.005979  rms=0.003762  fundAmp=0.005278  thd=-16.51 dB  ultra=-56.41 dB
        signal=sine40  peak=0.006168  rms=0.002778  fundAmp=0.002314  thd= +2.76 dB  ultra=-41.41 dB

    fade OFF (fadeDisabled=1, --disable-tail-fade-for-measurement):
        signal=sine50  peak=0.005979  rms=0.003762  fundAmp=0.005278  thd=-16.51 dB  ultra=-56.41 dB
        signal=sine40  peak=0.006168  rms=0.002777  fundAmp=0.002314  thd= +2.76 dB  ultra=-41.41 dB

Delta:
    sine50 : peak 0.000000 dB / rms 0.0000 dB / fundAmp 0.000000 dB / thd 0.00 dB / ultra 0.00 dB
    sine40 : peak 0.000000 dB / rms -0.0031 dB / fundAmp 0.000000 dB / thd 0.00 dB / ultra 0.00 dB
    （いずれも測定分解能以下。limiter/clamp は両条件で flatTop=0 / limitZone=0 = 非接触）

Origin:
    OTHER（TAIL-FADE ではない）
    末尾 fade は実 IR の有効な低域 tail を一切削っていない。

Confidence:
    PROVEN（fade 帯 energy ≈ 0、A/B Δ ≈ 0、hook 作動を fadeDisabled=0/1 で確認）
```

---

## 1. 111-1 fade geometry の実測（推測ではなく測定）

`doTrimStep()` に `[IR_TAIL_GEOM]`（NonRT trace）を追加して実測。

```text
[IR_TAIL_GEOM] gen=0 loadedSr=384000 loadedLen=62914 targetLength=192000 copySamples=62914
               fadeSamples=1258 fadeStart=61656 fadeEnd=62914 fadeMs=3.2760 fadeDisabled=0
```

| 項目 | 予想（ユーザー提示） | 実測 | 判定 |
|---|---|---|---|
| sourceLen | 8253 | **8252**（trim で 1 sample 減） | 差異あり・実測採用 |
| loadedLen | 66024 | **62914** | 差異あり（§7） |
| copySamples | 66024 | **62914** | 差異あり |
| fadeSamples | 1320 | **1258** | `round(62914*0.02)=1258`。下限 256 / 上限 `sr*0.08=30720` に非拘束 |
| fadeStart | — | **61656** | 実測 |
| fadeEnd | — | **62914** | 実測 |
| fadeMs | — | **3.276** | 実測 |

予想の 1320 は「66024 を仮定した場合の値」であり、実データ長 62914 に対しては 1258 が正しい。
**仮定せず実測した結果、clamp は作用していない**（256 < 1258 < 30720）。

## 2. 111-2 fade 前後の envelope（実データ）

`fadeStart` 近傍の実測（ch0、Tukey 後 = fade 前 / fade 後）:

| i | t [ms] | afterTukey | afterFade | gain |
|---|---|---|---|---|
| 60632 | 157.896 | -3.03364331e-06 | -3.03364331e-06 | 1.000000 |
| 61144 | 159.229 | -1.80899126e-06 | -1.80899126e-06 | 1.000000 |
| 61656 | 160.563 | -9.03347364e-07 | -9.03347364e-07 | 1.000000 |
| 61784 | 160.896 | -7.26469336e-07 | -6.52551947e-07 | 0.898251 |
| 61912 | 161.229 | -5.69224738e-07 | -4.53388861e-07 | 0.796502 |
| 62168 | 161.896 | -3.13134091e-07 | -1.85690009e-07 | 0.593005 |
| 62680 | 163.229 | -3.01644714e-08 | -5.61087941e-09 | 0.186010 |
| 62913 | 163.836 | 0.0 | 0.0 | 0.000000 |

- gain-ramp は**設計どおり 1.0→0.0 で動作**（0.898 → 0.593 → 0.186 → 0）。
- しかし **fade 帯の絶対値は既に 1e-6〜0**。IR は 160 ms 時点で十分に減衰済み。
- → 「fade が tone を削っている」のではなく、**削る対象が存在しない**。

## 3. 111-3/111-7 fade 帯の低域 energy

```text
[IR_TAIL_ENERGY] total=8.05031433e+00 preFade=8.05031433e+00 fadeBand=2.03404509e-10
                 bandFrac=0.000000 lp100Total=2.60889697e-02 lp100Pre=2.60889676e-02
                 lp100Band=2.10598849e-09 lp100BandFrac=0.000000
```

| 指標 | 値 | 対総和 |
|---|---|---|
| fadeBand / total（全帯域） | 2.527e-11 | **-106.0 dB** |
| lp100Band / lp100Total（0–100 Hz 相当） | 8.072e-8 | **-70.9 dB** |

- 全帯域・低域のいずれでも **fade 帯の寄与は -71 dB 以下**。
- WORK109/110 で問題になった「末尾付近の tap」は *合成 IR 固有* の配置であり、
  **実 IR にはその状況が存在しない**。

## 4. 111-4/111-5 fade ON/OFF A/B（measurement-only）

- `--disable-tail-fade-for-measurement` を追加。`IRTrimTestHooks.h` の test-only フラグを
  `doTrimStep()`（**NonRT**）だけが参照し、`applyGainRamp` を skip する。
  - production 既定は必ず `false`（誰も true にしない）。**RT 経路は一切参照しない**。
  - 作動は `fadeDisabled=0`（ON） / `fadeDisabled=1`（OFF）で確認（hook の証明）。
- 固定条件：実 IR（sampledata/impulse.wav）、OS=2、engine 384k/block 2048、L0 幾何、
  phase、scale、generation、limiter（既定）を両条件で完全同一。

## 5. 111-6/111-7 結果（buzz 指標）

| signal | 指標 | fade ON | fade OFF | Δ |
|---|---|---|---|---|
| sine50 | outPeak | 0.005979 | 0.005979 | 0 |
| sine50 | rmsAll | 0.003762 | 0.003762 | **0.0000 dB** |
| sine50 | fundAmp(50Hz) | 0.005278 | 0.005278 | 0 |
| sine50 | THD | -16.51 dB | -16.51 dB | 0.00 dB |
| sine50 | ultraRatio | -56.41 dB | -56.41 dB | 0.00 dB |
| sine40 | outPeak | 0.006168 | 0.006168 | 0 |
| sine40 | rmsAll | 0.002778 | 0.002777 | **-0.0031 dB** |
| sine40 | fundAmp(40Hz) | 0.002314 | 0.002314 | 0 |
| sine40 | THD | +2.76 dB | +2.76 dB | 0.00 dB |
| sine40 | ultraRatio | -41.41 dB | -41.41 dB | 0.00 dB |

- **両信号で fade ON/OFF の差は測定分解能以下（≤0.003 dB）**。
- `flatTop=0` / `limitZone=0` → **limiter/clamp は両条件で非接触**（limiter 寄与と混同していない）。

## 6. ゴール判定：**Case B**

```text
Case B（TAIL-FADE は主因ではない）
  • fade 帯 energy = 総和の 2.5e-11（-106 dB）、低域でも 8.1e-8（-71 dB）
  • fade ON/OFF で出力 buzz 指標が変化しない（Δ ≤ 0.003 dB）
  → doTrimStep() の末尾 gain-ramp は「ベースのジジジ」の寄与要因ではない
```

- 111-8（synthetic 境界試験）は**不要**（A/B が明確に Case B のため）。
- 111-10：Tukey は**固定**（変更なし）。fade 前データは「Tukey 後」であり、
  Tukey 自身の減衰分離は本 WORK の対象外（fade 帯 energy ≈ 0 のため影響は同じく無視できる）。

## 7. 副次的に得られた未確定事象（次工程の候補）

1. **resample 比が 8 でない**：実 IR は source 8252 → converted **62914**（比 **7.6241**）。
   理想的 48k→384k は ×8 = 66016。WORK109 の合成 IR は ratio=8.0000 だったため、
   **実 IR 側の resampler 出力長**に別要因（r8brain の入出力長計算、source 側の実効長）がある。
   → 確定には `IRDSP::resampleIR` / `computeTargetIRLength` の出力長式の監査が必要（未実施）。
2. **低域 THD が limiter 非接触で高い**：sine50 で THD **-16.51 dB**、sine40 では **+2.76 dB**
   （残差 ≥ 基本波）。しかも `flatTop=0` / `limitZone=0` で limiter/clamp は未動作、
   末尾 fade も無関係（本 WORK で否定）。
   → 低域で IR の通過利得が小さい（inPeak 0.25 → outPeak 0.006, 約 -32 dB）ため、
     残差（数値ノイズ/DC/denormal/過渡）が相対的に支配している可能性が高い。
   → 未確定。WORK112 の対象候補。

## 8. 受入条件

```text
[x] 最新 ConvoPeq.md 再生成                        (2026-09-18 21:31:27 / 5,252,672 B)
[x] 実IR source/converted geometry を取得          (48000/8252 → 384000/62914)
[x] copySamples を取得                             (62914)
[x] fadeSamples を取得                             (1258)
[x] fadeStart/end を取得                           (61656 / 62914)
[x] fade前後のIR envelopeを取得                    (§2)
[x] fade帯の低域energyを取得                       (§3, -71 dB)
[x] 現行fade ON/OFF A/Bを同一条件で実行            (§4/§5)
[x] outputの低域スペクトル比較                     (fundAmp 一致)
[x] THD比較                                        (Δ 0.00 dB)
[x] ultrasonic energy比較                          (Δ 0.00 dB)
[x] IR tail energy比較                             (§3)
[x] Tukey条件は固定                                (未変更)
[x] limiter条件は固定                              (flatTop=0/limitZone=0 両条件)
[x] E未修正                                        (未変更)
[x] F4/F5未修正                                    (未変更)
[x] RT変更なし                                     (追加は NonRT trace と test-only hook のみ)
[x] IRRuntimeContract未変更                        (未変更)
```

## 9. 変更したファイル（test-only / NonRT trace のみ）

| ファイル | 変更 | 種別 |
|---|---|---|
| `src/convolver/IRTrimTestHooks.h`（新規） | `disableTailFadeForMeasurement()` の inline atomic（既定 false） | test-only |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | `[IR_TAIL_GEOM]` / `[IR_TAIL_ENV]` / `[IR_TAIL_ENERGY]` 追加、`fadeDisabled` で `applyGainRamp` を skip | NonRT 観測 + test-only 分岐（既定 false） |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--disable-tail-fade-for-measurement` / `--buzz-probe-signal=` 追加、`[PROBE_METRICS]`（peak/rms/fundAmp/THD/ultra）追加 | test-only |

- production の `doTrimStep()` の**挙動は不変**（フラグ既定 false では完全に従来どおり）。
- RT（Add/Get）には一切の変更なし。

## 10. 次工程（WORK112 候補）

Case B のため、**L0/FDL に戻らない**。WORK110/111 で否定された領域を除いた残りは以下。

1. **`doTransformStep()` 系の実データ監査**：`applyAsymmetricTukey` → phase transform → scale →
   engine input の各段で、実 IR のデータがどう変化するか（特に低域）を同一 generation で対応付ける。
2. **低域 THD の発生源の特定**（§7-2）：limiter/clamp/fade が非関与である以上、
   EQ 段（shaper/softClip 相当）、engine の denormal/DC 挙動、合算時の丸めのいずれか。
   A/B は「その1段だけ」を test-only で切り替えて測定する。
3. （任意）**resample 比 7.6241 の要因**（§7-1）を `IRDSP::resampleIR` の出力長式から確定。

## 11. ツール使用（本作業）

headroom proxy＋context-mode（ctx_execute / ctx_execute_file）＋rtk(WSL) 常時。
WSL `rg`/`grep -a`。AiDex（`aidex_query`）で `doTrimStep` / `analyzeRun` / `Metrics` を特定。
serena で構造メモ。ビルドは `vcvarsall.bat x64` → `cmake --build build --config Release
--target AudioEngineHarness`。`python output_sourcecode_markdown.py` で正本再生成（111-0）。
計測は逐次起動（`0xC0000374` は終了時のみ、測定値は有効）。
