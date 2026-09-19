# WORK113（= WORK112-5）— Engine Null / Impulse Response Baseline Audit

- **作成日**: 2026-09-18
- **種別**: Read-only / measurement / test-only（113-1〜113-5 のうち 113-1・113-2 を実施、113-3〜113-5 は設計提示のみ）
- **前工程**: WORK112（`doc/work112/real_ir_transform_chain_audit_20260918.md`）
- **正本**: `ConvoPeq.md` 2026-09-18 21:59:59 / 5,269,690 bytes（113-0 完了）
- **制約遵守**: production algorithm / MKL engine algorithm / partition geometry / FDL / resampler /
  limiter / Tukey / phase / scale policy / OS 補償 の**いずれも未変更**。RT に分岐・確保・ログなし。
  追加は test-only 解析（`printDeltaNull`）のみ。

```text
ENGINE BASELINE（113-1 / 113-2 の結論）

Zero input（level=0, 実IR・delta の双方）:
    in   = 完全に 0（全サンプル 0）
    out  = residualRms 2.33e-10 / residualPeak 4.66e-10
           帯域は全帯域で ~1e-14〜4e-11（float 床）
           resPeaks は高域の散乱ノイズ（~1.1e-7）で 137/237/325 Hz は**存在しない**
    → **engine + capture に付加的自己残留はない（PROVEN）**

Silence IR:
    → **実行不能**。全零 IR は loader が拒否し `[PROBE] FAIL: IR load timeout`（§2）

Delta IR + sine（最重要）:
    sine50: out h1=3.169e-2, residualRms=1.088e-2, res/fund=-6.28 dB
            null test（out − gain×遅延入力）: null/y = **-7.23 dB**（lag=1040, gain=0.1288）
    sine40: out h1=2.379e-2, residualRms=1.085e-2, res/fund=-3.81 dB
            null test: null/y = **-5.13 dB**（lag=561, gain=0.0957）
    → delta IR でも output は「遅延入力のスカラー倍」にならない（PROVEN）

IR 非依存性（決定的）:
    残留ピーク周波数は delta と実IR で**完全に一致**
      sine50: 137.7 / 237.3 / 134.8 / 140.6 / 325.2 Hz
      sine40: 146.5 / 149.4 / 228.5 / 225.6 / 334.0 Hz
    帯域の形も同一（100-200 ≈ 200-2k ≫ 2k-20k）。振幅比のみ ~382× 異なる
    → **残留は IR の畳み込み（L0/FDL/partition）から生成されていない（PROVEN）**

判定:
    Engine の付加的ノイズ      … 棄却（zero-input で 2.3e-10）
    入力に比例する成分         … PROVEN（level=0 でゼロ）
    IR 依存の大きさ × 固定スペクトル形 … PROVEN
    Case C1 / C2 / C3 / C4 の最終分離 … **OPEN（113-3〜113-5 が必要）**
```

---

## 1. 113-0 engine input → engine output → capture の経路

```text
ConvolverProcessor::process(AudioBlock<double>&)          (Runtime.cpp:238)
  └ StereoConvolver::process(ch, in, out, n)              (Runtime.cpp:1227)
       └ nucConvolvers[ch]->Get(out, n)                   (Runtime.cpp:1242)   ← NUC 出力（double）
  → wet/dry・crossfade/smoothing・makeup・softClip・limiter 等（既存の安全鎖/混合段）
  → AudioEngine 出力（float）
  → harness tap（session.outCapL）→ [LF_RESIDUAL]/[DELTA_NULL] 解析
```

- IR 側は `SetImpulse(const double*)` まで double。scale は engine 内周波数領域（`cblas_dscal`）。
- **engine 出力と capture の間には混合／平滑／安全鎖が存在**する（WORK112 §2 でいずれも今回条件では非動作を示唆：
  `flatTop=0` `limitZone=0`、softClip 閾値 0.9275 に対し |out|≈5e-3）。
- 113-3〜113-5 ではこの区間を「E = NUC 出力」と「C = capture」に分けて null を取る必要がある（§6）。

## 2. 113-1 Silence IR baseline

**結果: 実行不能**。全零 IR は loader に拒否される。

```text
[CONV_IR] transferIRStateFrom: no IR data to transfer   (×6)
[PROBE] FAIL: IR load timeout
```

- 理由（コード）：`applyAsymmetricTukey`/`validateBuffer` 系は `maxAbs > 1e-12` を要求し、
  `resampleIR` は `SilentIR` を返して `"IR is silent (all samples near zero)."` で失敗する。
- これは**新規の観測事実**（全零 IR は graceful に拒否される）。buzz の原因ではない。

**代替として「入力ゼロ baseline」を実施**（`--buzz-probe-level=0`）。IR は通常どおりロードされる。

```text
[PROBE_METRICS] signal=sine50 inPeak=0.000000 outPeak=0.000000 rmsAll=0.000000
[LF_RESIDUAL] sine50_IN  : h1=0.0 … 全 0
[LF_RESIDUAL] sine50     : fundRms=4.08e-13 h1=1.56e-12 residualRms=2.330e-10 residualPeak=4.663e-10
                           resBand: 0-20=1.0e-14 20-50=1.8e-14 50-100=2.0e-14 100-200=4.1e-14 200-2k=8.3e-13 2k-20k=4.2e-11
                           resPeaks: 40702Hz:1.18e-07 93873Hz:1.18e-07 …（散乱＝ノイズ床）
[DELTA_NULL] delta lag=0 gain=0.000000 yRms=2.32e-10 nullRms=2.32e-10
```

- 実 IR / delta IR の**双方**で同一（差は 1% 以内）。
- **137/237/325 Hz は一切現れない**。
- → **engine と capture は付加的な自己残留・自己発振を持たない。**
  113-5 の「Engine baseline failure」は**否定**。

## 3. 113-2 Delta IR baseline + null test

synthetic delta（48k idx61 → 384k argmax488、resample 後 energy 6.738）で sine を入力。

| | sine50 | sine40 |
|---|---|---|
| outPeak | 0.052224 | 0.043954 |
| h1 (fund) | 3.1692e-02 | 2.3789e-02 |
| residualRms | **1.0883e-02** | **1.0847e-02** |
| residualPeak | 3.3554e-02 | 3.3697e-02 |
| res/fund | **-6.28 dB** | **-3.81 dB** |
| **null/y** | **-7.23 dB** | **-5.13 dB** |
| null lag / gain | 1040 / 0.1288 | 561 / 0.0957 |
| subnormal / nonfinite | 0 / 0 | 0 / 0 |

- delta IR の理想出力は「入力の遅延スカラー倍」＝純正弦。しかし **null は -7 dB しか下がらない**
  （`out − gain×遅延入力` が出力の 44%/55%）。
- h2..h5 は極小（2e-7 以下）なので、この null は高調波歪ではない（WORK112 と同じ）。
- → **delta（= ほぼ恒等）でも engine 出力は入力の忠実なコピーにならない。**

## 4. IR 非依存性（本 WORK の最重要観測）

| | delta IR | 実 IR |
|---|---|---|
| sine50 residualRms | 1.088e-02 | 5.563e-04 |
| sine50 res/fund | -6.28 dB | -16.53 dB |
| sine50 resPeaks | 137.7 / 237.3 / 134.8 / 140.6 / 325.2 Hz | **同一** |
| sine50 band 100-200 / 200-2k / 2k-20k | 4.69e4 / 4.44e4 / 4.22e3 | 1.227e2 / 1.162e2 / 1.10e1 |
| band 比（100-200 : 200-2k : 2k-20k） | 1.00 : 0.95 : 0.09 | 1.00 : 0.95 : 0.09 |

- **ピーク周波数も帯域の形も完全に一致**し、**振幅のみ ~382× 異なる**。
- sine40 でも同様に delta/実IR でピークが一致（146.5 / 149.4 / 228.5 / 225.6 / 334.0 Hz）。
- → **残留は IR（＝partition/FDL/partSize/IR 長）由来ではない。**
  WORK110 の L0/FDL correspondence が正常であることと整合。

## 5. 113-5 判定ゲート（暫定）

| Case | 判定 | 根拠 |
|---|---|---|
| **C1: NUC engine 起因** | **有力（OPEN）** | delta でも residual（§3）。ただし engine 単体か後段かは未分離 |
| **C2: Engine 以外の post-process** | **OPEN（同等に有力）** | 経路に wet/dry・smoothing・makeup・softClip・limiter が存在（§1）。今回は非動作だが分離未実施 |
| **C3: Reference 自体に residual** | 未判定 | 113-3/113-4 未実施 |
| **C4: 両者とも residual** | 未判定 | 同上 |
| **Engine baseline failure（付加ノイズ）** | **棄却** | zero-input で 2.33e-10（§2） |

**現時点で PROVEN なこと**
1. engine + capture に付加的自己残留はない（zero-input）。
2. 残留は入力に比例する（level=0 で消える）。
3. 残留のスペクトル形は IR に依存しない（振幅のみ IR 依存）。
4. denormal ではない（subnormal=0）、float capture 量子化でもない（WORK112 §4-3）。

**未確定（113-3〜113-5 で確定する）**
- その「固定スペクトル形」が engine 内部で作られるのか、engine の外側（混合／平滑／安全鎖）か。

## 6. 113-3 / 113-4 / 113-5 の実施設計（次段・未実施）

```
G = engine input（= stepTrimmed の ch0、scale 前）
R = Reference FIR（time-domain, double）
E = NUC 出力（StereoConvolver::process の out）
C = capture（harness outCapL）
```

1. **G のエクスポート（test-only, NonRT）**：`doBuildStep` 直前に `stepTrimmed` ch0 と
   `stepResult.scaleFactor`、`sampleRate` をバイナリ（header + f64）で書き出す。環境変数で ON/OFF。
   → 192000×8 B = 1.5 MB。NonRT のみ。
2. **E のエクスポート（test-only, NonRT 相当の観測）**：`StereoConvolver::process` の `Get` 直後に
   コピーするフック（RT には追加しない設計が望ましいため、まずは `Get` の戻りを
   harness 側 tap で代替できるか検討）。RT 変更が避けられない場合は **C のみで判定**する。
3. **Reference FIR**：`R[n] = scale × Σ_k G[k]·x[n−k]`（time-domain, double）。
   **OS を跨ぐ場合は up/down のモデルが必要**なため、まずは
   `cap = 0` の理想条件で `R` と `E` を作り、`E−R` を比較する。
4. **null 3 本**：`E−R` / `C−E` / `C−R` を `[LF_RESIDUAL]` と同じ指標で出力。

> 注意：`x`（engine 入力）は harness tap の `inCap` と一致するかを確認する必要がある
> （generator → AudioEngine 入力 → engine input のどこで値が変わるか）。

## 7. 受入条件

```text
[x] 最新 ConvoPeq.md 再生成          (2026-09-18 21:59:59 / 5,269,690 B)
[x] silence IR baseline              → 実行不能（loader 拒否）。代替の zero-input baseline を実施
[x] delta IR baseline                (§3)
[x] delta null test                  (§3, null/y=-7.23/-5.13 dB)
[ ] 同一 G の reference FIR          → 未実施（§6 の設計）
[ ] NUC output E                     → 未実施
[x] capture output C                 (§2/§3)
[ ] E-R null / C-E null / C-R null   → 未実施
[x] 40/50 Hz residual comparison     (§4)
[ ] block 1024/2048/4096 comparison  → 指示により 113-5 の結果後
[x] production変更 0                 (変更なし)
[x] RT変更 0                         (変更なし)
```

## 8. 変更したファイル（test-only のみ）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `printDeltaNull()`（lag/gain 推定 + null RMS/peak/dB + null の `[LF_RESIDUAL]`）を追加し delta probe で呼び出し |

- production / MKL engine / partition / FDL / resampler / limiter / Tukey / phase / scale は**未変更**。
- `IRRuntimeContract` 未変更。RT 未変更。

## 9. 次の判断材料として残る問い

- **137.7 Hz（50 Hz 入力）/ 146.5 Hz（40 Hz 入力）** が何に由来するか。input 周波数に依存して移動し、
  IR に依存しない。block 1024@192k = 187.5 Hz、ring 16384@384k = 23.4 Hz のいずれの整数倍でもない。
- 入力に比例し、IR に比例しない「固定スペクトル形」の発生源は、engine 内部か engine 外側か（§6 で確定）。
- 参考：WORK112 の resample 比 7.6241（`8252 → 62914`）は原因候補から切り離し、**WORK114** として扱う。

## 10. 指示

113-1・113-2 を完了し、113-3〜113-5 は**実施設計まで**を提示した状態で、指示どおり**ここで一旦停止**します。
113-3〜113-5（G/E のエクスポートと 3 本 null）に進むか、先に 113-6（入力周波数 sweep）／113-7（block sweep）
を回すかの判断を仰ぎます。
