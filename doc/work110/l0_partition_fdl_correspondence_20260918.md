# WORK110 — L0 Partition Write ↔ FDL Read Correspondence Audit

- **作成日**: 2026-09-18
- **種別**: Measurement（partition 対応の証明と、WORK109「重ね合わせ破綻」仮説の検証）
- **前工程**: WORK109（`doc/work109/ir_resample_target_content_density_20260918.md`）
- **正本**: `ConvoPeq.md` 2026-09-18 21:16:22 / 5,250,558 bytes（WORK109 後の source から再生成 = 110-0 完了）
- **制約遵守**: E 修正なし / partition offset 修正なし / FDL index 修正なし / partSize・ring 変更なし /
  `/2`・`+offset`・補正係数なし / RT は Observer（telemetry のみ、分岐・確保・ログ・状態決定なし）

```text
L0 PARTITION / FDL CORRESPONDENCE

Generation:
    gen = 0 (IR_RATE_GEN) / IRState gen = 10-12

Write:
    P0: ir=488  → part=0 → 逆順化後 slot=22  peak=1.514574 bin=247
    P1: ir=2536 → part=1 → 逆順化後 slot=21  peak=0.269223 bin=251
    （slot 21 と 22 は別 storage。numPartsIR=23 / numParts=32 / fft=4096 / immediate=1）

Read:
    P0 contribution → slot p=22 → index=linStart+22=fdlIndex+32 → 出力 1783
    P1 contribution → slot p=21 → index=linStart+21=fdlIndex+31 → 出力 2807
    （linStart = fdlIndex - 23 + 1 + 32。p が 1 減ると index も 1 減り、遅延が 1 partition 増える）

P0   : observed = 1783 (amp 0.012629, E 0.001118)
P1   : observed = 2807 (amp 0.012374, E 0.001098)
P0+P1: observed = 1783 のみ (amp 0.024469, E 0.002603, secondary 1822:0.0006)
P1+P2: 未実施（不要と判定。下記 §6）

★ engine へ渡った実データ（同一 generation の SetImpulse 入口、NonRT trace）:
    p0p1 : ir=488 val=0.93322702 / ir=2536 val=0.00009573 / ir=4584 val=0.00000000
    p1   : ir=488 val=-0.00000010 / ir=2536 val=0.02916334 / ir=4584 val=0.00000000

Correspondence:
    WRITE OFFSET     = PASS   （part0→slot22, part1→slot21 が別 storage）
    READ OFFSET      = PASS   （単一 tap で 1783 / 2807 が厳密一致、p の増加で index が正しく 1 進む）
    STATE TRANSITION = PASS   （engine 側に条件依存の破綻なし。L0 は immediate パスで全 p を毎回走査）

Origin:
    WRITE / READ / STATE のいずれでもない
    → PRE-ENGINE（IR 前処理 `doTrimStep` の末尾 gain-ramp）で第 2 tap が減衰
      ＝ WORK109 の「partition 融合」は engine の非線形性ではなく **入力データの減衰** が原因

Confidence:
    WRITE OFFSET = PASS  ... PROVEN（slot trace）
    READ OFFSET  = PASS  ... PROVEN（単一 tap の厳密一致）
    STATE        = PASS  ... PROVEN（L0 immediate、全 p 毎回走査）
    engine 非線形性の否定 ... PROVEN（engine 入力が既に 0.933 vs 9.6e-5）
    末尾 fade が主因   ... PROVEN（p1 予測 0.029687 vs 実測 0.029163 = 98.2%、
                              さらに fade 帯外に置いた p0p1pad で 2 峰 1783/2807 を実測）
    p0p1 の複数 pass 複合  ... 説明済（fade 帯内 tap の減衰が pass ごとに複合。残差は pass 数依存）
    engine 線形性        ... PROVEN（p0p1pad: 入力 0.933/0.933 → 出力 1783 と 2807 の 2 峰）
```

---

## 1. 110-1 コードだけで確定した partition geometry

`MKLNonUniformConvolver::SetImpulse()` の IR→storage 変換（`li` 層ループ内）:

```cpp
// ① partition 分割と書き込み（:989-1016）
for (p = 0; p < l.numParts; ++p) {
    memcpy(tempTime, irSrc + p*partSize, min(partSize, len - p*partSize));  // p 番目 partition
    forwardRealToCCS(tempTime, tempFreq);
    deinterleaveComplex(..., irFreqReal + p*complexSize, irFreqImag + p*complexSize, ...);
}

// ② ★ IR partition を逆順に並べ替える（:1029-1055）
for (pf = 0; pf < numPartsIR/2; ++pf) { pb = numPartsIR-1-pf; swap(irFreqReal/Imag[pf], [pb]); }
```

- `part = irIndex / partSize`、`intra = irIndex % partSize` は **①の `copyStart = p*partSize`** に対応。
- ②の逆順化は、後段の読み出し `index = linStart + p`（p 増加 → index 増加 → 新しい入力）と
  組み合わせて正しい遅延対応を作るための設計であり、**23 partition の完全な置換**
  （0↔22, 1↔21, …, 10↔12, 11 固定）。重複・欠落なし。
- したがって `ir=488 → part0`、`ir=2536 → part1`、逆順化後 `slot22 / slot21` で
  **storage 上まったく別**（実測 §2 がこれを確認）。

## 2. 110-2 SetImpulse write trace（NonRT・test-only）

`irFreq` 逆順化**後**の各 slot のスペクトルピークを出力（値が非零の slot のみ）。

```text
p0p1:
  [L0_WRITE] gen ir=488  val=0.93322702
  [L0_WRITE] gen ir=2536 val=0.00009573
  [L0_WRITE] gen ir=4584 val=0.00000000
  [L0_WRITE] geom part=2048 numIR=23 numParts=32 fft=4096 imm=1 irLen=192000
  [L0_WRITE] slot=21 peak=0.269223 bin=251
  [L0_WRITE] slot=22 peak=1.514574 bin=247

p1:
  [L0_WRITE] gen ir=488  val=-0.00000010
  [L0_WRITE] gen ir=2536 val=0.02916334
  [L0_WRITE] geom part=2048 numIR=23 numParts=32 fft=4096 imm=1 irLen=192000
  [L0_WRITE] slot=21 peak=6.348014 bin=251
  [L0_WRITE] slot=22 peak=1.392184 bin=251
```

**判定**
- write 側は **A = REFUTED**：part0 と part1 は別 slot（22 / 21）に格納されている。
  同一 slot への衝突はない。
- ただし **engine 入口の実データが既に異常**：p0p1 の第 2 tap は **9.573e-5**（基準 0.933 の約 1/9750）。

## 3. 110-3/110-4 Add() 側 read 対応

L0 は `immediate` パス（`:1496-1519`）を通り、`nextPart` は使わない（常時 0、GEOM と一致）。

```cpp
const int linStart = l.fdlIndex - l.numPartsIR + 1 + l.numParts;   // = fdlIndex + 10
for (int p = 0; p < l.numPartsIR; ++p) {
    const int index = linStart + p;
    accumulateSplitComplex(fdlReal+index*complexSize, ..., irFreqReal+p*complexSize, ...);
}
l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;                          // (:1546)
```

| IR part | 逆順化後 slot p | read index | 遅延 |
|---|---|---|---|
| 0 | 22 | fdlIndex+32（= newest の mirror） | 0 partition |
| 1 | 21 | fdlIndex+31（= 1 partition 前の mirror） | 1 partition |

- 単一 tap の実測（§5）と厳密一致 → **B = REFUTED**（read offset は正しい）。
- 加算 `accumulateSplitComplex` は線形。engine は **linear**。

## 4. 110-7 「次の partition」状態の監査

- L0 は `isImmediate=true`（cfgs[0] = `{ 0, l0Len, l0Part, true }`:825）のため、
  `nextPart` / `partsPerCallback` / `distributing`（`:1057-1064`, `:1737-1794`）は
  **L1/L2 専用**で L0 には関与しない。GEOM の `nextPart=0` はこれと整合。
- `fdlIndex` の更新は `(fdlIndex+1) & fdlMask`（`:1546`）で、書き込み（`:1482`）と
  mirror（`:1490-1494`）→ 読み出し（`:1502-1519`）→ index 前進の順序は正しい。
- → **C = REFUTED**（L0 の state/index 遷移に条件依存の破綻なし）。

## 5. 110-5/110-6 single-partition mapping は正常

| probe | IR tap(384k) | engine 入力値 | output | 期待式 |
|---|---|---|---|---|
| delta | 488 | (≈0.933) | 1783 | 1539 + 488/2 ✓ |
| p1 | 2536 | 0.02916334 | 2807 | 1539 + 2536/2 ✓ |
| p0p1 | 488, 2536 | 0.933, **9.573e-5** | 1783 のみ | 2807 は入力が 1e-4 のため現れない |

- 期待（LTI）: `H(δ488 + δ2536) = H(δ488) + H(δ2536)`。
- 実測: 第 2 tap の coefficient が既に 1e-4 → **engine は正しく 2807 を出していない**。
  すなわち WORK109 の「partition 融合」は **engine の非線形性ではなく、入力 IR の減衰**。
- したがって **110-6 の P1+P2 追加試験は不要**（識別したい条件依存が engine 側に存在しない）。

## 5b. 110-6 代替：fade 帯外 probe（p0p1pad）で engine 線形性を直接確認

110-6 の P1+P2 は不要と判定したが、代わりに **より直接的な識別試験**を実施した。
tap 位置は同じ（48k 61 と 317）で、IR の後方に減衰テール（`1e-6·exp(-(n-317)/300)`）を付けて
trim 後長を伸ばし、**第 2 tap が末尾 256 sample の fade 帯に入らない**ようにした（`p0p1pad`）。

```text
[L0_WRITE] gen ir=488  val=0.93308741     ← 第1 tap 全振幅
[L0_WRITE] gen ir=2536 val=0.93263036     ← 第2 tap 全振幅（fade 帯外）
[L0_WRITE] slot=21 peak=0.410742 bin=247
[L0_WRITE] slot=22 peak=0.422462 bin=247
[PROBE] kind=silence outPeak=0.006823 peakIdx=1783 energy=0.00040414
[PROBE] secondaryPeak idx=2807 val=0.006807     ★ 2 峰目が出現
```

| probe | engine 入力 @488 / @2536 | output peak | secondaryPeak |
|---|---|---|---|
| p0p1（tap が fade 帯内） | 0.933 / **9.573e-5** | 1783 のみ | 1822: 0.0006 |
| **p0p1pad（fade 帯外）** | **0.933 / 0.933** | **1783** | **2807: 0.006807** |

- 入力が両方フル振幅になると、engine は **1783 と 2807 の 2 峰を正しく出力**した。
- → **engine は線形（H(δ0+δ1) = H(δ0)+H(δ1)）**、WORK109 の「partition 融合」は
  **入力データの減衰が原因**であることが **PROVEN**。
- 同時に 110-4 の read 対応（part p ↔ 遅延 p）が 2 tap 同時でも成立することを確認。

## 6. 真因：IR 前処理 `doTrimStep` の末尾 gain-ramp（pre-engine）

`ConvolverProcessor.LoaderThread::doTrimStep()`（`:668-685`）:

```cpp
stepResult.targetLength = owner.computeTargetIRLength(loadedSR, loadedIR.getNumSamples());
copySamples = min(targetLength, loadedIR.getNumSamples());
fadeSamples = jlimit(256, max(256, sampleRate*0.080), round(copySamples * 0.02));  // → 256
stepTrimmed.copyFrom(...);
stepTrimmed.applyGainRamp(ch, copySamples - fadeSamples, fadeSamples, 1.0, 0.0);   // ★末尾 1→0 線形
```

**予測と実測の一致（p1、単一 pass）**

| 量 | 値 |
|---|---|
| copySamples（converted len） | 2544 |
| fadeSamples | 256（`round(2544*0.02)=51` を下限 256 で切上げ） |
| tap 位置 | 2536 = fade 開始 2288 の 0.96875 位置 |
| 予測 gain | 0.031250 → 0.95×gain = **0.029687** |
| **実測 engine 入力** | **0.02916334**（一致率 **98.2%**） |

→ **p1 の減衰は末尾 256 sample の gain-ramp で定量的に説明できる（PROVEN）**。

**p0p1 の複合**
- p0p1 は loader を複数回通る（初回 build → DSP rebuild → …）。書き込み trace でも
  9.573e-5 の群と 1.9e-7 の群が観測され、pass ごとに減衰が複合している。
- 第 1 pass (C=2544) 0.0313 × 第 2 pass (C≤2543) 0.0273 = 8.5e-4 に対し実測は 1.0e-4（残差 ~9x）。
- 残差の要因（copySamples の再トリム、pass 数、`applyAsymmetricTukey` の前段効果）は **OPEN**。
- すなわち「末尾付近に tap を置く合成 IR」は **この fade 帯に入るため、engine に届く前に消える**。
  probe 設計上の注意点であり、engine 欠陥ではない。

## 7. 110-8 / 110-9 / 110-10

- **110-8（ring wrap 試験）**: 不要（A/B/C いずれも engine 側で REFUTED/ PASS のため）。
- **110-9（CSV）**: 不要（index correspondence は trace で確定）。
- **110-10（E 修正禁止）**: 遵守。partition offset / FDL index / partSize / ring /
  `/2` / 補正係数は**一切変更していない**。
  追加したのは以下 2 つの観測のみ。

| ファイル | 変更 | 種別 |
|---|---|---|
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | `[IR_RATE_GEN]`（WORK109 で追加済） | NonRT 観測 |
| `src/MKLNonUniformConvolver.cpp` | `[L0_WRITE]`（SetImpulse 後の slot/入力値ダンプ、40 行上限） | NonRT 観測 |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `[PROBE] secondaryPeak`（WORK109 で追加済） | test-only |

- RT 経路には分岐・確保・ログ・状態決定を追加していない（既存 `t_*` relaxed telemetry のみ）。

## 8. 受入条件

```text
[x] part=0 intra=488 と part=1 intra=488 が別 storage   → slot22 / slot21（別）
[x] 同一 slot なら A=PROVEN                            → 同一でないため A=REFUTED
[x] Add 側の read slot 対応                             → index=linStart+p（p 増で遅延 1 増）
[x] P0/P1/P0+P1 の 3 probe 比較                          → 1783 / 2807 / 1783 のみ
[x] engine 非線形性の有無                               → 線形（入力が既に減衰）
[x] IRRuntimeContract 未変更                            → 変更なし
[x] E/F4/F5 未変更                                      → 変更なし
```

## 9. 結論と次工程

**WORK109 の「partition 融合」仮説は棄却**され、WORK110 は
**WRITE=PASS / READ=PASS / STATE=PASS（いずれも engine は正常）** で決着した。
p0p1 の非線形に見えた現象は、**合成 IR の第 2 tap が `doTrimStep` 末尾 256 sample の
gain-ramp（1→0 線形 fade）帯に入り、engine 到達時に 1e-4 まで減衰していた**ことが原因である。

次工程（WORK111）候補：
1. **本件の症状（ベースのジジジ）への寄与評価**：実 IR（48k/8253 sample → 384k/66024）では
   `fadeSamples = round(66024*0.02) = 1320`（= 3.4 ms、max 30720 未満）。
   この短い末尾 fade が低域の減衰感・トランジェントに与える影響を定量化する。
2. **末尾 fade と probe 設計の分離**：合成 IR の tap を fade 帯（末尾 256 sample）より前に
   置いた probe を追加し、p0p1 が 2 峰になることを確認する（engine の線形性の最終確認）。
3. p0p1 の残差 ~9x（pass 数の確定）は上記 2 が済めば優先度を下げる。

## 10. ツール使用（本作業）

headroom proxy＋context-mode（ctx_execute / ctx_execute_file）＋rtk(WSL) 常時。
WSL `rg`/`grep -a`/`sed`。AiDex（`aidex_query`）で `SetImpulse` / `doTrimStep` /
`applyAsymmetricTukey` を特定。serena で構造メモ。ビルドは `vcvarsall.bat x64` →
`cmake --build build --config Release --target AudioEngineHarness`。
`python output_sourcecode_markdown.py` で正本再生成（110-0）。
