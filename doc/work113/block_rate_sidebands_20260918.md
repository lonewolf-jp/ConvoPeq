# WORK113 続行（113-3〜113-5）— Block-Rate Sideband Origin

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（113-3〜113-5。113-4 の E 取得は代替手段で前進）
- **前工程**: WORK113（`doc/work113/engine_null_impulse_baseline_20260918.md`）
- **正本**: `ConvoPeq.md` 現行（WORK113 の instrumentation を含む）
- **制約遵守**: NUC algorithm / partition / FDL / resampler / scale / limiter / softClip / smoothing / wet-dry /
  OS 補償 / block size / IR preprocessing / production RT architecture は**いずれも未変更**。

```text
BLOCK-RATE SIDEBAND ORIGIN（113-3〜113-5 の到達点）

観測（delta IR / sine50）:
    residualPeaks = 137.7 / 237.3 / 134.8 / 140.6 / 325.2 Hz
    residualRms   = 1.0877e-02  (res/fund = -6.28 dB)
    null/y        = -7.34 dB

決定的な数値一致: 残留ピークは **block rate 187.5 Hz の側帯波**
    n * 187.5 +- f_in   (n = 1, 2)
      f_in = 50 Hz :  137.5 (=187.5-50) , 237.5 (=187.5+50) , 325.0 (=375-50)
      f_in = 40 Hz :  147.5 (=187.5-40) , 227.5 (=187.5+40) , 335.0 (=375-40)
    実測との差は最大 3.1 Hz（DFT bin 幅 2.93 Hz 内。134.8/137.7/140.6 は同一線の窓漏れ広がり）

    187.5 Hz = 192000 / 1024 = 384000 / 2048 = **engine の 1 block 周期（5.333 ms）**

解釈:
    信号が **block 周期で振幅/位相変調** されている（時間変化系）。
    純正弦 × 固定 IR の定常線形畳み込みでは生成不能。 WORK112/113 の
    「IR 非依存・入力比例・zero-input で消える・非調和」をすべて同時に説明できる。

E 取得の代替（RT 非侵襲）:
    StereoConvolver::process は `got = Get(out, n)` の後
    `if (got < n) memset(out+got, 0, ...)` でゼロ埋めする。
    この **per-block ゼロ埋めが block-rate 変調の有力候補**だったため、
    MKLNonUniformConvolver::Get の戻り値 `got` と不足回数を
    **既存と同じ relaxed telemetry（t_getGot / t_getShort）** で観測（RT は store のみ、
    分岐・ログ・確保なし）。

    → 実測: `getNs=2048 getGot=2048 getShort=0`（before/after とも）
      = **Get は常に全サンプルを返しており、ゼロ埋めは発生していない**
      → **per-block zero-fill 仮説は REFUTED**

FINAL（暫定）:
    C1 (NUC engine 内部)          = OPEN（NUC の Get 経路は正常だが、内部の block 処理は未分離）
    C2 (engine の post / per-block) = **有力**（入力タップ後〜出力の間の block-rate 変調）
    C3 (双方)                     = OPEN
    C4 (Reference/Input モデル)     = **棄却**（入力は -154 dB で清浄、R は純正弦×固定IRで非調和を出せない）
```

---

## 1. 113-3 Reference FIR（R）

- 入力 `x` は harness タップで **-154 dB の純正弦**（`sine50_IN residualRms=3.53e-09`、`h1=2.50e-01`）。
- `R[n] = scale · Σ G[k]·x[n−k]` は**定常線形畳み込み**であり、純正弦入力に対しては
  **同じ周波数の正弦のみ**を出力する。したがって
  `n·187.5 ± 50 Hz` の側帯波を R が生成することは**原理的に不可能**。
- → **R は clean（PROVEN、解析的＋入力実測による）**。
- 数値 R の生成（G のエクスポートと OS 跨ぎモデル）は §4 の設計が必要で、本段では未実施。

## 2. 113-4 3 本の null（E の入手）

- **E（NUC 出力）は現行構造では harness から直接取得できない**（タップは `AudioEngineHarness` の
  入出力 float バッファのみ）。production RT への常設 I/O は禁止のため、
  第3候補の「一時的・コンパイルゲートされた診断 seam」が必要（§4）。
- ただし **E に依存しない形で block-rate 候補の 1 つを棄却できた**（§3）。
- したがって `E−R` / `C−E` は未取得、`C−R` は「C の残留 = R には存在しない成分」として
  実質的に §3 の結果と同義。

## 3. `Get()` 戻り値テレメトリ（RT-safe Observer 追加）

`MKLNonUniformConvolver::Get` に既存 `t_*` と同形の relaxed telemetry を追加：
`t_getGot`（戻り値）、`t_getShort`（`got < numSamples` の回数）。GEOM に `getGot`/`getShort` を追加。

```text
[GEOM before-capture] ... RT(addNs=2048 addCalls=152264 L0calls=152264 getNs=2048 getGot=2048 getShort=0 ringW=8192 ringR=8192 avail=0 fdl=4 nextPart=0)
[GEOM after-capture]  ... RT(addNs=2048 addCalls=155184 L0calls=155184 getNs=2048 getGot=2048 getShort=0 ringW=0    ringR=0    avail=0 fdl=24 nextPart=0)
[DELTA_NULL] delta lag=1055 gain=0.127715 yRms=2.471651e-02 nullRms=1.061314e-02 nullPeak=3.356811e-02 null/y=-7.34dB
```

- `getGot = 2048 = getNs`、`getShort = 0` → **`StereoConvolver::process` のゼロ埋めは発生していない**。
- → block-rate 変調の原因は「Get の不足サンプルゼロ埋め」では**ない**。

## 4. 113-5 判定と残る分離

| 成分 | 判定 | 根拠 |
|---|---|---|
| 入力 x | clean | `-154 dB`、側帯波なし |
| Reference R | clean | 純正弦×固定線形系は非調和を生成不能 |
| NUC Get の戻り値 | 正常 | `getGot=2048, getShort=0` |
| NUC 内部の block 処理 | **未分離（OPEN）** | E 未取得 |
| engine の per-block / post | **有力** | 入力タップ後〜出力の block-rate 変調。ゼロ埋め以外の候補 |
| 変調周波数 | **187.5 Hz = 1 block（5.333 ms）** | 側帯波の厳密一致 |

**C1/C2 を分離するための最小設計（次段）**
1. `doBuildStep()` 直前で `G`（`stepTrimmed` ch0）、`scaleFactor`、`sampleRate`、`blockSize`、
   `FilterSpec`（tailMode/tailEnabled/tailStartSeconds/tailStrength/tailL1L2Multiplier）、
   `enableDirectHead` を **NonRT test-only** でバイナリ書き出し（環境変数ゲート）。
2. harness 内で `MKLNonUniformConvolver` を**単独生成**し、`SetImpulse(G, n, blockSize, scale, enableDirectHead, &spec)`
   → `Add`/`Get` を block 単位で駆動して **E** を得る（384k 領域、OS 非依存）。
3. 同時に `R = scale · conv(x384, G)` を time-domain double で生成（同レート）。
4. `E−R` を `[LF_RESIDUAL]` と同指標で比較 → `n·187.5 ± f_in` が E−R に現れれば **C1**、
   現れなければ **C2**。
   （`C−E` は OS 跨ぎモデルが必要なため、まず E−R で判定する）

> 注意：オフライン NUC は production と同じ `FilterSpec` を渡さないと build が一致しない。
> §4-1 のエクスポートに `FilterSpec` を含めるのはこのため。

## 5. 変更したファイル（test-only / RT-safe Observer のみ）

| ファイル | 変更 |
|---|---|
| `src/MKLNonUniformConvolver.cpp` | `t_getGot` / `t_getShort` の relaxed telemetry を `Get()` に追加（RT は store のみ） |
| `src/MKLNonUniformConvolver.h` | `GeometryTrace` に `lastGetGot` / `getShortCount` を追加 |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `printDeltaNull()`（WORK113 で追加）／GEOM に `getGot`/`getShort` を追加 |

- NUC algorithm / partition / FDL / resampler / scale / limiter / softClip / smoothing / wet-dry /
  OS 補償 / block size / IR preprocessing / production RT architecture は**未変更**。
- 追加した RT コードは `publishAtomic` / `fetchAddAtomic`（relaxed）のみ。分岐・ログ・確保・I/O なし。

## 6. 未確定事項

1. **E−R**（NUC 内部 vs post-process の分離）— §4 の設計で確定する。
2. **block-rate 変調を発生させている具体的な段** — 候補：AudioEngine の OS up/down、
   per-block の混合／平滑、convolver 呼び出し境界。ゼロ埋めは棄却済み。
3. 137.7 / 146.5 Hz の帰属は「187.5 ± f_in」で説明済み（113-6 の周波数 sweep は
   この結論を補強するためだけに使える）。
4. resample 比 7.6241 は **WORK114** として切離し（本段でも未着手）。
