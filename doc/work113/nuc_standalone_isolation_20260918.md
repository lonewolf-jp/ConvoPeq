# WORK113-6 — NUC Standalone Isolation Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（AudioEngine を経由しない NUC 単独駆動）
- **前工程**: `doc/work113/block_rate_sidebands_20260918.md`, `doc/work113/engine_null_impulse_baseline_20260918.md`
- **制約遵守**: NUC algorithm / partition / FDL / block size / resampler / scale / limiter / softClip /
  smoothing / wet-dry / OS 補償 は**いずれも未変更**。production AudioEngine へは観測フックを追加していない。

```text
# WORK113-6 — NUC Standalone Isolation Audit

HEAD:
    git              = 0654e7b5
    ConvoPeq.md      = 5,276,742 bytes / 2026-09-18 22:33:38
    instrumentation  = t_getGot / t_getShort（Get 戻り値, relaxed telemetry）
                       harness: --nuc6=<freq>（standalone）, --buzz-probe-signal=, printDeltaNull, GEOM getGot/getShort
Build = OK（AudioEngineHarness / 0 error）

Input:
    sine50, sine40   （振幅 0.25, 4 秒, x384 を直接生成）

IR:
    delta（384k index 488 に 1.0, irLen=192000, それ以外 0）
    ※ L0 のみ（FilterSpec.tailEnabled=false / tailMode=2 Bypass）→ 直列 L0 immediate path を分離

NUC:
    sampleRate = 384000
    blockSize  = 2048
    scale      = 1.0（NUC 単体。engine の scaleFactor は使わない）
    駆動       = Add(2048) → Get(2048) を 750 block（Get の return は毎回 2048）

R:  （time-domain reference: R[n] = scale * x[n-488]）
    h1          = 2.50000574e-01
    residualRms = 3.55796607e-09   (res/fund = -153.9 dB)
    sidebands   = 137.5Hz:1.22e-06 / 237.5Hz:3.42e-08 / 325.0Hz:3.93e-08  → 実質ゼロ

E:  （NUC standalone 出力）
    sine50: h1 = 1.42262068e-01, residualRms = 4.84000448e-02 (res/fund = -6.36 dB)
            sidebands = 137.5Hz:4.787e-02 / 237.5Hz:2.771e-02 / 325.0Hz:2.025e-02 / 425.0Hz:1.549e-02
            resPeaks  = 134.8 / 140.6 / 240.2 / 234.4 / 322.3 Hz（bin 5.859 Hz）
            h3=2.73e-04, h5=1.59e-04（奇数高調波も僅かに存在）
    sine40: sidebands = 147.5Hz:4.564e-02 / 227.5Hz:2.959e-02 / 335.0Hz:2.009e-02 / 415.0Hz:1.622e-02

E - R:
    sine50: residualRms = 4.84000448e-02, residualPeak = 1.14745916e-01
            137.5 Hz = 4.78658267e-02
            237.5 Hz = 2.77119808e-02
            325.0 Hz = 2.02510100e-02
            147.5 Hz = (n/a, f_in=50)
            227.5 Hz = (n/a)
            335.0 Hz = (n/a)
    sine40: 147.5 Hz = 4.56377129e-02 / 227.5 Hz = 2.95897882e-02 / 335.0 Hz = 2.00943933e-02
    → E - R ≒ E（R が -154 dB のため）。差の全量が NUC 由来。

Verdict:
    E ≒ R ?                      NO  （E: 4.84e-02 vs R: 3.56e-09、約 7 桁差）
    block-rate sideband in E-R ? YES （n*187.5 +- f_in が E と E-R の双方に存在）

C1 = PASS   （block-rate 側帯波は **NUC 単独で生成される**）
C2 = FAIL   （post-process 起因ではない）
```

---

## 1. 113-6-0 実施前の固定内容

| 項目 | 値 |
|---|---|
| git HEAD | `0654e7b5` |
| ConvoPeq.md | 5,276,742 B / 2026-09-18 22:33:38 |
| NUC telemetry | `t_getGot`（Get 戻り値）, `t_getShort`（`got < numSamples` 回数）— relaxed store のみ |
| harness | `--nuc6=<freq>`（standalone）, `--buzz-probe-signal=`, `printLfResidual`, `printDeltaNull`, GEOM `getGot/getShort` |
| 変更禁止の遵守 | partition / FDL / block size / resampler / scale / limiter / softClip / smoothing / wet-dry / OS 補償 未変更 |

## 2. 手法（RT 外・AudioEngine 非経由）

`convo::MKLNonUniformConvolver` を harness 内で**直接インスタンス化**し、
実 API（`SetImpulse(const double*, int, int, double, bool, const convo::FilterSpec*)` /
`Add(const double*, int)` / `Get(double*, int)`）のみで駆動した。production の RT 経路へは
一切のフックを追加していない（113-6-5 の要求）。

- `SetImpulse(G, 192000, 2048, 1.0, false, &spec)`、`spec.sampleRate=384000`、`tailEnabled=false`
- `x384[i] = 0.25·sin(2π f i / 384000)`、`R[n] = x384[n−488]`
- 解析は `[LF_RESIDUAL]`（LS 基本波除去 + flat-top DFT 高調波 + residual FFT band/peaks）
  と明示的な `[NUC6_SIDEBAND]`（Hann 単一ビン）を併用

## 3. 結果（113-6-3 の明示的側帯波抽出）

### sine50（block rate 187.5 Hz）

| f | 期待 | E (NUC) | R (reference) | E−R |
|---|---|---|---|---|
| 137.5 Hz | 1×187.5 − 50 | **4.787e-02** | 1.218e-06 | **4.787e-02** |
| 237.5 Hz | 1×187.5 + 50 | **2.771e-02** | 3.424e-08 | **2.771e-02** |
| 325.0 Hz | 2×187.5 − 50 | **2.025e-02** | 3.930e-08 | **2.025e-02** |
| 425.0 Hz | 2×187.5 + 50 | **1.549e-02** | 7.569e-09 | **1.549e-02** |

`[LF_RESIDUAL] NUC_E resPeaks: 134.8Hz:6.80e+02 140.6Hz:6.51e+02 240.2Hz:3.94e+02 234.4Hz:3.77e+02 322.3Hz:2.88e+02`
（bin = 5.859 Hz。134.8/140.6 は 137.5 Hz 一本の窓広がり）

### sine40（block rate 187.5 Hz）

| f | 期待 | E (NUC) | R (reference) | E−R |
|---|---|---|---|---|
| 147.5 Hz | 1×187.5 − 40 | **4.564e-02** | 1.325e-06 | **4.564e-02** |
| 227.5 Hz | 1×187.5 + 40 | **2.959e-02** | 8.571e-08 | **2.959e-02** |
| 335.0 Hz | 2×187.5 − 40 | **2.009e-02** | 6.412e-08 | **2.009e-02** |
| 415.0 Hz | 2×187.5 + 40 | **1.622e-02** | 1.740e-08 | **1.622e-02** |

- R は両周波数で **-154 dB**（側帯波なし）→ reference は清浄。
- E と E−R は**同一**（差の全量が NUC 由来）。
- 期待周波数との差は bin 幅（5.859 Hz）内。

## 4. 判定

```text
E ≈ R ?                      NO
block-rate sideband in E-R ? YES

C1 = PASS   （NUC 内部起因）
C2 = FAIL   （post-process 起因ではない）
```

- 本試験は **L0 のみ**（`tailEnabled=false`）で実施したため、発生箇所は
  **L0 の immediate path**（`Add` → L0 partition push/FFT → FDL 書込 → 全 partition 積算 →
   interleave → IFFT → `ringWrite` → `Get` の `ringRead`）に局在する。
- `getGot=2048 / getShort=0`（WORK113 続行）は維持されており、**ゼロ埋めは原因ではない**。
- 参考：E には h3=2.73e-04 / h5=1.59e-04 の奇数高調波も僅かに存在（副次成分）。

## 5. 次段（NUC 内部 audit）の対象

`E−R ≒ E` が block 周期で変調されているため、次は以下を**1 段ずつ**観測する
（いずれも NonRT/test-only の範囲で、production RT は変更しない）。

```text
Add
 ↓
L0 partition push / forward FFT（prevInputBuf / inputAccBuf の overlap-save 組立）
 ↓
FDL 書込 + mirror（fdlReal/fdlImag の index = linStart + p）
 ↓
全 partition 積算（accumReal/accumImag）
 ↓
interleave → IFFT
 ↓
ringWrite(fftOutBuf + partSize, partSize)   ← 有効前半/後半の選択
 ↓
Get = ringRead
```

- 仮説候補：①overlap-save の組立（prev/current の入れ替え）が 1 block ごとに不連続を作る、
  ②`ringWrite` に渡す有効領域（`fftOutBuf + partSize`）の選択、③`linStart` の
  `- numPartsIR + 1 + numParts` が partition 数 32 のとき block ごとに折返す位置のずれ。
- いずれも WORK110 の「partition correspondence」とは別軸（**時間軸の連続性**）である点に注意。

## 6. 変更したファイル（RT-safe / test-only のみ）

| ファイル | 変更 |
|---|---|
| `src/MKLNonUniformConvolver.cpp` | `t_getGot` / `t_getShort`（Get の戻り値観測、relaxed store のみ） |
| `src/MKLNonUniformConvolver.h` | `GeometryTrace` に `lastGetGot` / `getShortCount` |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `--nuc6=`（standalone ドライバ）, `specAmpHann`, `printDeltaNull`, GEOM `getGot/getShort` |

- production AudioEngine / RT 経路への観測フック追加は**なし**。
- 追加した RT コードは `publishAtomic` / `fetchAddAtomic`（relaxed）のみ（分岐・ログ・確保・I/O なし）。

## 7. 未確定・保留

- NUC 内部の**どのサブ段**が block 周期の不連続を生むか（§5）。
- 137.7 / 146.5 Hz の帰属は `n×187.5 ± f_in` で説明済み（113-6-6 の frequency sweep は不要）。
- block sweep（113-7）と resample 比 7.6241（WORK114）は**未着手**（指示どおり）。
