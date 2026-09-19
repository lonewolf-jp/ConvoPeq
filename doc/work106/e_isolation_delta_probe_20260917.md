# WORK106 — E の最小再現・構造隔離（単一デルタ IR probe）

- **作成日**: 2026-09-17
- **種別**: Measurement（最小再現による E の範囲確定）
- **前工程**: WORK105（`doc/work105/`）
- **指示**: F5/F4 より先に E を一段深く隔離する。第一実験は「単一デルタ IR → convolver → 出力」。

```text
判定: E は NUC の【単一partition 畳み込み】では再現しない（正しい）
      E は【複数partition（IR長 > partSize）】でのみ再現する
      ＋【初回以降の IR ロードがエンジンに反映されない】副次欠陥を発見
```

---

## 1. 実験装置（harness 拡張・test-only）

| 追加 | 内容 |
|---|---|
| `BuzzSignal::Impulse` | 単一サンプル 1.0（システム同定用） |
| `writeSyntheticIrFile(kind)` | `Delta`（index61 に 1.0）/ `Boost9dB` / `Silence` の float32 WAV 合成 |
| `--buzz-probe=delta\|boost\|silence\|real` | 最小構成（AsIs/EQ bypass/softclip off/autoGain off/0dB）でインパルス→出力を測定 |
| `--buzz-quiet=<ms>` | DSP 側 IR 再構築の静穏期間 |
| `--buzz-probe-level=<x>` | プローブ入力レベル（既定 0.25） |
| 解析 | peakIdx / peak / energy / sigTaps / outsideTaps（±3 外）/ headFrac / タップダンプ |
| probe 時は既定 IR ロードを省略 | probe IR を「初回ロード」にする（初回反映仮説の検証） |

## 2. 実測結果

### 2.1 初回ロード = デルタ IR（IR長 496 sample → L0 単一 partition）

```text
[CONV_IR] transferIRStateFrom: IR transferred ch=2 len=496 sr=384000.0 block=2048 gen=10
[PROBE] kind=delta outPeak=0.012629 peakIdx=1783 energy=0.00111786 sigTaps=241 outsideTaps=234 headFrac=0.5570
 nearPeak: 1778:-0.002589 1780:0.005563 1781:0.009976 1782:0.012569 1783:0.012629 1784:0.010464 ...
```

**期待値との一致**：プローブ入力 0.25、IR scale≈0.5012（energy正規化）、出力段 HC(LR4 22kHz)+LC(18Hz)
のインパルス応答ピーク≈0.1146、dither headroom 0.891 とすると
`0.25 × 0.5012 × 0.1146 × 0.891 = 0.0128` → **実測 0.012629（誤差 1.3%）**。
形状も HC/LC フィルタで説明できる帯域制限パルス（タップ数 241 は resample の ringing＋フィルタ）。

→ **単一 partition の畳み込みは正しい**（NUC primitive は健全）。

### 2.2 実 IR（IR長 62914 sample → 31 partition + L1）

```text
[CONV_IR] IR transferred ch=2 len=62914 sr=384000.0 block=2048
[PROBE] kind=real outPeak=0.003416 peakIdx=1783 energy=0.00005092 sigTaps=63 outsideTaps=56 headFrac=0.9051
```

期待（同じ式で実 IR peak 0.499 を使用）`0.25 × 0.499 × 0.1146 × 0.891 = 0.0128`（※実IRも
energy正規化で peak≈0.499）→ **実測 0.003416 = 期待の 0.27 倍（-11.5dB）**、タップ数も 63 と少なく
尾部が欠落。→ **複数 partition の畳み込みが破綻**。

### 2.3 副次欠陥：初回以降の IR ロードが反映されない

初回に実 IR をロード済みの状態で、delta IR／boost IR／実 IR のいずれを再ロードしても
**出力がビット同一**（0.003416 / peakIdx 1783 / energy 5.092e-5 / タップ列まで一致）で、
`--buzz-quiet=70000` を与えても変化しなかった。

→ IR ロード（UI 側 `isIRFinalized` は成立、`[CONV_IR]` transfer ログも出る）は成功しているが、
**DSP 側エンジンが再構築されず、初回の IR のまま動作し続ける**。WORK105 で入れた
`IR_CONTRACT`（形状照合）は rate/block のみを見るため、この「内容の未反映」は検出しない。

### 2.4 B13 構造不変条件は健全

```text
[B13-GATE] L1: P=16384 o_L=47104 lead=22528 | I2=OK I2g=OK I5=OK I3(cap 65536 >= need 57344)=OK
```

384k/2048・tailMode=1（tailStart=0.12）での実測。違反なし＝E は構造上限界則の違反ではない。

## 3. NUC 幾何の理論値（要求4点の対応表・384k/2048・IR長62914）

| 項目 | 値 | 根拠 |
|---|---|---|
| host / processing rate | 192000 / **384000** | OS=2（F1 で UI も一致） |
| call quantum b | 2048 | processingBlockSize |
| **L0** partSize | 2048 | `l0Part = nextPow2(max(b,64))` |
| L0 被覆 l0Len | **47104**（23 partition） | `tailStart=0.12` → 46080 → 2048 の倍数へ切上=47104 |
| L0 numParts / fdlMask | 32 / 31 | nextPow2(23) |
| **L1** partSize | 16384 | l0Part×tailL1L2Mult(8) |
| L1 offset / len / numPartsIR | 47104 / **15810** / 1 | IR残差 |
| **L1 outputDelaySamples (o_L)** | **47104** | = 先行層 IR 総長（=l0Len） |
| L1 delayLineCapacity | 65536 | `ceil((o_L+P+b+15)/16)*16` |
| L1 lead（gate） | 22528 | `b*(bpp+distCbs-2)` |
| **L2** | 生成されず（len=0） | IR残差ゼロ |
| **L0 ring 長** | 16384 | `max(nextPow2(numParts*2+b), nextPow2(4*l0Part+4*b))` |
| m_latency | 2048 | =L0 partSize |
| **IR partition 格納順** | `irFreqReal/Imag` を **numPartsIR 内で逆順にスワップ**（SoA, stride=complexSize） | `MKLNonUniformConvolver.cpp:1006-1032` |

**デルタ IR の構造的差異（決定的）**: IR長 496 < L0 被覆 47104 かつ ≤ partSize×1 なので
`numPartsIR=1`、`l1Len=max(0,496-47104)=0` → **L0 のみ・単一 partition**。
実 IR は `numPartsIR=31` かつ L1 が生成される。→ E は「複数 partition の FDL 蓄積」または
「L1 テール層」のどちらか（または両方）に限定される。

## 4. E の有力仮説（WORK107 の起点）

1. **FDL（frequency-domain delay line）の index 対応**：`Add()` 内の履歴スロットと IR partition の
   掛け合わせ順序。単一 partition では index 0 のみ使用のため破綻が顕在化しない。
2. **IR partition 逆順ソート**（`:1006-1032`）：numPartsIR>1 のときのみ実行される。ここが誤ると
   「先頭 partition と末尾 partition が入れ替わる」→ 実 IR のエネルギー（先頭 70 sample に 99%）が
   末尾 partition へ移動し、-11.5dB の減衰と歪として観測される…という機序が E の観測と整合。
3. **L1 テール層の遅延整合**（`:1051-1066` の delayLineBuf / outputDelaySamples）：
   B13 gate は OK だが、gate は上限条件であり実際の読み出し位相は別途要検証。
4. **副次**：初回以降のロードで DSP 側 engine が再構築されない（`rebuildAllIRsSynchronous` の
   実行／結果の可視化が必要）。

## 3.5 WORK107 追試：partition 数・層配置の切り分け（同一条件・期待値 0.0128）

デルタの挿入位置だけを変えた 3 プローブ＋実 IR（すべて初回ロード、quiet=60s）。

| probe | 48k idx | 384k idx | L0 partition | L1 | outPeak | ×期待 | 判定 |
|---|---|---|---|---|---|---|---|
| delta | 61 | 488 | **1** | なし | 0.012629 | **1.00** | 正常 |
| deltaMid | 4000 | 32000 | **16** | なし | 0.008797 | **0.69**（−3.3dB） | 軽度劣化 |
| deltaFar | 7000 | 56000 | 28 | **あり**(L1 idx8896) | 0.014770 | 1.15（L1 gain1.375考慮で0.84） | ほぼ正常 |
| real IR | — | — | 31 | あり | 0.003416 | **0.27**（−11.5dB） | 強度劣化 |

確定した知見：

1. **単一 partition は完全に正しい**（1.00）。NUC の基本演算・HC/LC 後段は健全。
2. **L1 単独は概ね正しい**（0.84〜1.15）。L1 の遅延整合そのものは主因ではない。
3. **IR エネルギーが複数 partition に分散する場合に強い劣化**（実 IR 0.27）：
   - 単一 partition 集中（delta）→ 無症状
   - 1 partition 内に集中（deltaMid, #16）→ 軽度（0.69）＋**出力位置が理論値 34048 に対し 17539 と大きくずれる**
   - 多数 partition に分散（実 IR）→ 強度（0.27）＋THD −16.5dB＋超音波 +82dB
4. 遅延位置のずれ（理論値と実測の大差）は、**partition ↔ 入力履歴（FDL）の対応付け**が
   誤っていることを示す。`MKLNonUniformConvolver` の IR partition 逆順ソート
   （`:1006-1032`）と `Add()`／`processLayerBlock()` の FDL index が第一容疑。
   partition 間の相対位相が誤ると、振幅スペクトルは保たれるが**相互に部分打消し**し、
   観測どおり「レベル低下＋コム状歪＋広帯域ハッシュ」になる（単一 partition では
   partition 間加算が発生しないため無症状、という事実と完全に整合）。
5. B13 gate は全構成で OK（I2/I2g/I5/I3）＝境界則違反ではない。

## 5.5 WORK107 の残タスク（優先順）

1. **2 タップ IR（同一 partition 内: 384k idx 488 と 490）** → 単一 partition 内の複数タップ健全性。
2. **隣接 2 partition に 1 タップずつ**（384k idx 488 と 2048+488）→ partition 間加算の直接検証。
   ここで 0.5×＋打消しが出れば FDL/逆順ソートで確定。
3. **非RT トレースの追加**：`Add()` の `fdlIndex` / 使用 partition / `ringWrite` 位置・
   `Get()` の `m_outputSamplesProcessed` / `ringRead` を 1 ブロックだけダンプする
   テスト専用フック（RT 変更なし・NonRT 出力）で、index 対応を実測で可視化。
4. 副次欠陥（初回以降のロードが未反映）の確定：`rebuildAllIRsSynchronous` の実行有無・
   `IRState.ir` 長・scale を NonRT ログ化。


1. **デルタを L0 第2 partition に置く**（48k IR で index ≥ 2048×8=16384 に 1.0）：
   `numPartsIR≥2`・L1 なし → **L0 複数 partition（FDL/逆順）単独**を切り分ける。
   - 失敗 → FDL／逆順ソート確定（仮説1/2）
   - 成功 → L1 テール層確定（仮説3）
2. **2 タップ IR（例: index 488 と 490、同一 partition）** → 単一 partition 複数タップの健全性。
3. **L1 単独**（IR長を L0 被覆直後まで伸ばす）→ tailMode=Bypass との比較で L1 寄与を分離。
4. **再ロード反映の可視化**：`rebuildAllIRsSynchronous` の実行有無・`IRState.ir`・scale を
   NonRT ログで1行出力（副次欠陥の確定）。
5. その後の F5（B2）→ F4（limiter）。

## 6. 装置側の不具合（修正済み）と教訓

- `--buzz-quiet` の substr off-by-one（14→13）→ 静穏期間が 0/5s になり **初回ロード測定を誤認**。
  これが「delta と real がビット同一」という副次欠陥の発見につながった。
- probe の `inPeak` が out から計算されていた（表示のみの問題）→ ラベル修正。
- 教訓：DSP 側の IR 再構築完了を UI 側シグナルだけで推定してはならない（WORK104-R2 の
  150s 静穏と同根）。**エンジン世代を直接観測できる NonRT カウンタ**が必要。

## 7. ツール使用（本作業）

headroom proxy＋context-mode（batch/execute）＋rtk 常時。WSL rg/ast-grep/ag/sed/awk。
AiDex（query: `writeBoostIrFile` 周辺・`MKLNonUniformConvolver` の層構成）、semble。
ビルドは `vcvarsall x64`＋`cmake --build --target AudioEngineHarness --parallel 3`。
`AudioEngineHarness` 既定スイート・`IRRuntimeContractTests`・`BuildErrorClassificationTests` は
WORK105 のまま緑。
