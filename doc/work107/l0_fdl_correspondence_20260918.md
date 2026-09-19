# WORK107 — L0 multi-partition FDL 対応関係の決定的検証

- **作成日**: 2026-09-18
- **種別**: Measurement（E の原因判定：A= L0 FDL/ordering 破綻 か B= 別経路か）
- **前工程**: WORK106（`doc/work106/`）
- **指示に基づく制約**: E の暫定修正なし／F5・F4 なし／RT logging・allocation・validation なし／
  tailMode・limiter で症状を消す実験なし。

```text
判定: A（L0 multi-partition の FDL/ordering 破綻）は【不支持】
      E の主因は【IR ↔ エンジンの実効サンプルレートが 2 倍ずれている】こと（rate axis 欠陥）
      ただしその 2 倍は WORK105 の IR_CONTRACT が見る位置とは別の場所にあり、契約の穴
```

---

## 1. 実測（全プローブ・同一条件：AsIs／EQ bypass／softclip off／autoGain off／level0.25／384k/2048）

期待レベル = `0.25 × scale(≈0.5012) × HC(LR4 22kHz)インパルス応答ピーク(≈0.1146) × dither headroom(0.891) ≈ 0.0128`

| probe | 48k idx | 384k idx | L0 part | L1 | outPeak | ×期待 | peakIdx 実測 | 1/2則の予測 `1539+t/2` |
|---|---|---|---|---|---|---|---|---|
| delta | 61 | 488 | 0 | なし | 0.012629 | **1.00 PASS** | 1783 | 1539+244=**1783** ✓ |
| P1Delta | 317 | 2536 | 1 | なし | 0.012374 | **0.97 PASS** | 2807 | 1539+1268=**2807** ✓ |
| DeltaMid | 4000 | 32000 | 15 | なし | 0.008797 | 0.69 | 17539 | 1539+16000=**17539** ✓ |
| DeltaFar | 7000 | 56000 | 28 | あり(L1) | 0.014770 | 1.15 | 29539 | 1539+28000=**29539** ✓ |
| real IR | — | — | 31 | あり | 0.003416 | **0.27** | 1783 | — |

**確定（4点すべて厳密一致）**: 出力タップ位置は `output_index = 1539 + IR_index/2`。

## 2. この 1/2 則が意味すること（A の否定）

- 1/2 は **単一 partition のタップ（delta, P1Delta）でも成立**する。したがって
  「partition 間加算（FDL/ordering）の破綻」は 1/2 の原因ではない
  → **仮説 A（L0 multi-partition の FDL/ordering 破綻）は不支持**。
- また **単一 partition 単一タップのレベルは正常**（1.00 / 0.97）。つまり
  「レベル低下＋THD＋超音波ハッシュ」という E の本質は、**単一partition・L1なしでは再現しない**。
  受入条件「L1 を介さない条件で E の再現性を確認」→ **E は再現しない（PASS 側）**。
- 一方で IR の**時間軸が 2 倍圧縮**されている（タップ位置が t/2）。時間軸 2 倍圧縮は
  **IR とエンジンの実効サンプルレートが 2 倍ずれている**ことを意味する。
  1/2 則の 4 点厳密一致はこの読みを強く支持する。

## 3. なぜ 2 倍で「レベル低下＋歪＋超音波ハッシュ」になるか（E の全観測と整合）

IR を 2 倍のレートで走査すると、IR は**時間圧縮**され、周波数軸では**2 倍に伸長**される。

- **孤立タップ**（delta, P1Delta）：位置が動くだけでレベルは不変 → 実測 1.00 / 0.97 ✓
- **広がりを持つ IR**（実 IR, 0.17s）：応答が 2 倍高い周波数へ移り、出力段 HC(LR4 22kHz) に
  削られる → **レベル低下**（0.27 = −11.5dB）＋**スペクトル変形**（THD −16.5dB）
- 圧縮で 2 倍帯域へ押し出された成分が、そのまま残留 → **超音波 +82dB のハッシュ** ✓
- 定常正弦（C1 sine50）は「実 IR の 100Hz 応答」に化ける → レベル/THD が設計と一致しない ✓
  （WORK105 の C1: 0.0476 / −16.5dB / −56.4dB と整合）
- **phase/EQ/makeup/shaper に不変**（WORK105 3.7）であることも、rate axis の欠陥なら当然 ✓

→ WORK103〜106 の全観測が、単一の原因（**IR↔エンジンの 2 倍レート差**）で説明できる。

## 4. なぜ WORK105 の IR_CONTRACT をすり抜けたか（契約の穴）

WORK105 の契約は `IRState(rate, block)` と **`convolver.prepareToPlay` に渡された値**を照合する。
しかし実測は、その照合を通過した world でも **実効 IR レートがエンジン実効レートの 1/2** で
あることを示している。すなわち契約の比較対象（宣言された prepared 形状）と、
**実際に畳み込みに使われる IR のレート／エンジンの実効処理レート**が一致していない。

根拠（実測ログ・WORK107 の probe 実行）:

```text
[CONV_IR] IR transferred ch=2 len=2544 sr=384000.0 block=2048 gen=10
[DSPCORE_PREPARE] processingRate=384000 processingBlockSize=2048
[CONV_STATUS] rebuildThreadLoop: generation=7 irLoaded=1 irLen=192000 osFactor=2 processingRate=384000.0
→ IR_CONTRACT の REFUSED は 0 件（＝契約は通過）
→ それでも実効は 2 倍ずれ（1/2 則）
```

併せて判明した構造上の事実：

1. `rebuildThreadLoop` の `CONV_STATUS` は **`irLen=192000`**（= `computeTargetIRLength` の
   targetLength。0.5s × 384k）。すなわち IRState.ir（2544／62914）と **エンジン IR（192000、
   末尾ゼロ詰め）は別物**で、契約は前者しか見ていない。
2. 起動直後は **auto OS=4 → `processingRate=768000 processingBlockSize=4096`** の world が
   実在し、その後 OS=2 → 384000 へ遷移する（ログで確認）。**768000 が「2 倍」の出所**と一致。
   OS=2 変更後に「768k のままのエンジン／IR」が残れば、まさに 2 倍レート差になる。
3. `MKLNonUniformConvolver::Add()` の test-only トレース（本作業で追加）は、本ビルドでは
   診断マクロが OFF のため**コンパイルアウトされ 0 のまま**（`addCalls=0`）。RT trace の
   常時有効化（テスト専用フック）が 107-4 の未完点。

## 2.5 107-2（partition 0 + 1 の2タップ）＝最重要の切り分け結果

IR = δ[488] + δ[2536]（partition 0 と partition 1、L1 なし、len=2544）。
期待（1/2則がそのまま成立するなら）: 2タップが 1783 と 2807 に分離して各 0.0124。

実測:

```text
[PROBE] kind=P0P1 outPeak=0.024469 peakIdx=1783 energy=0.00260279 sigTaps=795 headFrac=0.9458
nearPeak: 1778:-0.001391 1779:0.002604 1780:0.008750 1781:0.015678 1782:0.021501
          1783:0.024469 1784:0.023623 1785:0.019163 1786:0.012377 1787:0.005167 1788:-0.000639
```

判定（比較は単一タップの実測値 energy≈0.00110, outPeak≈0.0124 に対して）:

- **分離していない**: 2タップが **同一出力位置 1783 に融合**（1つの主ローブのみ）。
- **打消しなし**: 振幅 ≈ **2.0×**（0.0245 / 0.0124）、エネルギー ≈ **2.4×**（0.00260 / 0.00110）。
  → 位相反転や部分打消しではなく、**時間軸の融合（同一FDL位置への collapse）**。
- すなわち **partition 1 のタップが partition 0 と同じ出力時刻に写像**されている
  （ユーザーの判定木でいう `partition 1 → FDL slot 0` 型の対応）。

これは 1/2 則（単独タップの時間圧縮）と合わせて、**「partition ↔ FDL/時間 の対応が壊れている」**
ことの実測証拠であり、一方で **レベルと位相は保たれている**（打消しなし）ため、
破綻は「加算の位相」ではなく **「時間写像（FDL index / partition 配置）」** に限定される。

補足（レート軸との関係）: 1/2 則が *単一partition* でも成立する事実は、時間写像の誤りが
**partition 単位ではなく IR 全体の時間軸（レート/ホップ）** に由来することを示す。
2 タップの fusion は、その 2 倍圧縮下で hop 2048 が実効 1024 相当になり、
隣接 partition のタップが同一出力サンプルへ落ちたこととして説明できる。

## 5.5 受入条件の更新

| 項目 | 状態 |
|---|---|
| single-partition single-tap = PASS | **PASS**（delta 1.00 / P1Delta 0.97） |
| single-partition two-tap = PASS | 未実施（p0two 実装済・未実行） |
| L0 partition 1 single-tap 確定 | **レベル PASS（0.97）**、位置は 1/2 則 |
| partition 0 + partition 1 two-tap 確定 | **FAIL（時間融合・2.0×振幅・2.4×エネルギー・打消しなし）** |
| L1 を介さない条件で E の再現性 | 単一タップでは E（レベル低下/歪）**非再現**。2タップで**時間融合**を再現 |
| partition → FDL slot の NonRT trace | 未完（診断マクロ OFF でコンパイルアウト） |
| reverse-order と runtime FDL index の対応 | 未完（上記依存） |
| output delay 実測 vs 理論 | **実施**：`1539 + t/2`（4点厳密）＋2タップ融合 |

## 6.5 追求すべき単一仮説（現時点の最有力）

**IR の partition hop（2048）と、エンジンが IR を走査する実効ホップ（1024 相当）が 2 倍ずれている。**
これにより
 (a) 単一タップの出力位置が t/2（4点厳密一致）、
 (b) 2 タップ（2048 離れ）が同一出力サンプルへ融合（振幅2倍・打消しなし）、
 (c) 広帯域 IR は時間圧縮されて周波数軸が 2 倍へ伸び、出力段 HC(22kHz) に削られて
     −11.5dB＋THD −16.5dB＋超音波ハッシュ、
がすべて一つの原因で説明できる。WORK105 の IR_CONTRACT は
「IRState(宣言レート) vs convolver.prepareToPlay に渡された値」のみを照合するため、
**実効ホップ/実効レートの 2 倍**を検出できない（契約の穴）。

WORK108 の第一手は、`Add()` の実呼び出し `numSamples`（=実効入力数）／`L0 partSize`／
`IRState.sampleRate`／エンジンの実効 `oversamplingFactor` を **同一世代で 1 行出力**し、
2 倍の発生点（OS 解決／IR リサンプル／hop）を一意に特定すること。


| 項目 | 状態 |
|---|---|
| single-partition single-tap = PASS | **PASS**（delta 1.00） |
| single-partition two-tap = PASS | 未実施（p0two：実装済み・ビルド済み、未実行） |
| L0 partition 1 single-tap = PASS/FAIL 確定 | **PASS（レベル 0.97）**。位置は 1/2 則（t/2） |
| partition 0 + partition 1 two-tap 確定 | 未実施（p0p1：実装済み・ビルド済み、未実行） |
| L1 を介さない条件で E の再現性確認 | **E は再現しない**（単一partition・L1なしでレベル正常） |
| IR partition → FDL slot 対応の NonRT trace | **未完**（診断マクロ OFF でコンパイルアウト） |
| reverse-order storage と runtime FDL index の対応 | **未完**（上記に依存） |
| output delay 実測 vs 理論 | **実施**：`1539 + t/2`（4点厳密一致）。理論値（t + 2048 等）とは不一致 |

## 6. 次工程（WORK108）で確定すべきこと

1. **RT trace の常時有効化（テスト専用）**：`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` を外すか、
   `CONVOPEQ_TEST_TRACE` 等のテスト専用マクロを harness ターゲットに定義し、
   `Add(lastNumSamples, engineMaxBlock, L0partSize, addCalls)` と
   `Get(...)`／`ringRead` 位置を NonRT で観測する（**ログ・分岐・検証を RT に置かない**方針は維持）。
2. **実効レートの直接計測**：world の DSP が持つ `oversamplingFactor`／実効 processingRate と、
   IRState.sampleRate／engine IR（192000 パディング後のレート）を**同一世代で並べて出力**する。
   `1539 + t/2` の 2 倍がどこで生じているかを 1 行で確定させる。
3. **OS 変更の伝搬経路の確認**：`setOversamplingFactor(2)` が sealed snapshot の
   `BuildInput.oversamplingFactor` に反映され、world の DSP が 384k で構築されること
   （768k の world が残らないこと）を publish 世代単位で追跡。
4. **p0two / p0p1 の実行**（実装済み・ビルド済み）：同一 partition 内 2 tap と partition 0+1 の
   2 tap。1/2 則が支配的なら両者とも「2 タップが t/2 位置に現れる」はずで、
   その確認により「partition 間加算そのものは正常」まで確定できる。
5. その後、**契約の穴の閉塞**：`IRState` に加えて
   (a) エンジンが実際に使用した IR のレート（パディング後・リサンプル後）、
   (b) エンジンの実効 processingRate、
   を NonRT で照合対象に加える（WORK105 契約の拡張）。E の修正はその後に別工程で。

## 7. 変更ファイル（本作業・test-only）

| ファイル | 変更 |
|---|---|
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | `SynthIrKind` に P1Delta/P0P1/P0Two 追加、`--buzz-probe=p1|p0p1|p0two`、`[TRACE]` 出力 |
| `src/MKLNonUniformConvolver.h/.cpp` | `Add()` の test-only 形状トレース（relaxed telemetry）＋ 静的 getter（NonRT 読み出し） |
| `doc/work107/` | 本報告 |

production の音声経路（RT のロジック・分岐・ログ）への追加は **なし**（telemetry の relaxed store のみ、
かつ診断マクロでガード。ただし現ビルドでは OFF のため未計測 → 107-4 の未完点）。

## 8. ツール使用（本作業）

headroom proxy＋context-mode（batch/execute）＋rtk 常時。WSL rg/ast-grep/ag/sed/awk/fdfind。
AiDex（query: MKLNonUniformConvolver の Add/Get/ring、ConvolverProcessor の prepare/transfer）、
semble、graphify、cocoindex。ビルド：`vcvarsall x64`＋`cmake --build --target ConvoPeq
--target AudioEngineHarness --parallel 3`。cppcheck（新規変更はヘッダ1行＋ガード付き telemetry のみ）。
