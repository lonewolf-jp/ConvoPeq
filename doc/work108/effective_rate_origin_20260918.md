# WORK108 — Effective Geometry / Rate Origin の決定的追跡

- **作成日**: 2026-09-18
- **種別**: Measurement（`t → t/2` の発生点の一意特定）
- **前工程**: WORK106 / WORK107（`doc/work106/`, `doc/work107/`）
- **制約遵守**: E の修正なし / F5・F4 なし / partition ordering・FDL・partSize の変更なし /
  OS 補正なし / 症状補正（/2）なし / RT validation・branch・allocation・logging なし

```text
EFFECTIVE-RATE ORIGIN

IRState:
    sr = 384000
    block = 2048
    generation = 10
    len(engine irLength) = 192000  （= targetLength 0.5s@384k。source は 504/2544/… のtrim済み）

Engine:
    buildRate (filterSpec.sampleRate) = 384000
    block = 2048
    IR len (SetImpulse irLen) = 192000
    L0: part=2048 numIR=23 numParts=32 fft=4096
    layers = 2 (L0 + L1)
    ring = 16384
    generation = 10

World:
    UIprep sr = 384000  block = 2048
    (起動直後の OS=4 / 768000 / 4096 world は存在するが、probe world は 384000/2048)

Add:
    numSamples = 2048   （call 数 ≈ 1.2〜1.4e5、L0calls 同数）
    Get.numSamples = 2048
    ringW/ringR/avail は整合（avail=0）
    fdl = 16〜25 を巡回、nextPart=0

Observed:
    outputIndex = 1539 + (48kHz_index × 4)
    ※ 384k 換算 t で書くと output = 1539 + t/2 と等価

2x origin:
    [ engine rebuild / OS resolution / Add quantum / partition traversal ] を棄却
    → 残るは「IR の resample 内容密度」= 48k→192k(×4) で作られた IR が
      384k エンジンで走査されている（＝コンテンツが 192k 相当）

Confidence:
    Add quantum / OS / rebuild / L0 幾何 : PROVEN（不一致なし）
    192k コンテンツ仮説              : STRONGLY SUPPORTED（4点厳密 + 4倍則）
    発生点の最終確定                  : OPEN（resample target の世代記録が未取得）
```

---

## 1. 108-0 正本更新

`python output_sourcecode_markdown.py` で `ConvoPeq.md` を再生成。

```text
ConvoPeq.md : 5,243,329 bytes / 2026-09-18 19:56:58 （WORK107 変更反映済）
```

## 2. 108-2 test-only trace の常時有効化

`MKLNonUniformConvolver` に `GeometryTrace` 構造体と `getGeometryTrace()`（静的・NonRT 読み出し）を
追加し、診断マクロを**外して常時コンパイル**した。RT 側は **relaxed telemetry のみ**
（`publishAtomic` / `fetchAddAtomic`。ログ・分岐・検証なし）で、Observer 原則
（Observer は metrics/logging/telemetry のみ）に限定。

- engine build 形状: `SetImpulse`（NonRT）で記録
- runtime: `Add()`（numSamples / call数 / L0 FDL index / nextPart）と
  `Get()`（numSamples / ringW / ringR / avail）で記録

## 3. 108-3/108-5：同一 generation の 1 行（決定的）

p0two（同一 partition 0 内 2 tap, IR len=504）実行時の実測:

```text
[GEOM before-capture] IR(sr=384000 block=2048 len=192000 gen=10) UIprep(sr=384000 block=2048)
  ENG(buildRate=384000 irLen=192000 block=2048 layers=2 L0part=2048 L0numIR=23 L0numParts=32
      L0fft=4096 ring=16384)
  RT(addNs=2048 addCalls=139688 L0calls=139687 getNs=2048 ringW=8192 ringR=8192 avail=0
     fdl=20 nextPart=0)
[GEOM after-capture]  RT(addNs=2048 addCalls=142304 L0calls=142304 getNs=2048 fdl=16 nextPart=0)
```

**確定事項**

| 項目 | 実測 | 判定 |
|---|---|---|
| `Add(numSamples)` | **2048** | 「実効 quantum 二重化（4096）」仮説を**棄却** |
| `Get(numSamples)` | **2048** | 入出力量子は一致 |
| engine `buildRate` | **384000** | 「768k エンジン残存」仮説を**棄却** |
| engine `block` | **2048** | IRState.block と一致 |
| `L0.partSize` / `fft` | **2048 / 4096** | partSize = hop = 2048（2× の混入なし） |
| `L0.numIR` / `numParts` | **23 / 32** | L0 被覆 47104（=23×2048）と一致 |
| `IRState.sr` / `UIprep.sr` | **384000 / 384000** | 宣言形状は全て 384k |
| ring | 16384, W/R 整合, avail=0 | ring 幾何も整合 |

→ **ユーザーの 108-5 の第2分岐に合致**：「`Add(2048)` なのに `t/2` が発生する場合は、
Add → IR partition traversal → FDL index の内部で sample coordinate が変換されている」。
ただし本件ではさらに踏み込み、下記 §5 の「コンテンツ密度」仮説を支持する。

## 4. 108-1：p0two（同一 partition 内 2 tap）→ **A（時間順序保持）**

IR = δ[48k 61] + δ[48k 62]（= 384k 488 と 496。両者とも partition 0、IR len=504）。

```text
[PROBE] kind=P0Two outPeak=0.024981 peakIdx=1784 energy=0.003075 sigTaps=921 headFrac=0.933
 nearPeak: 1779:0.0020 1780:0.0069 1781:0.0129 1782:0.0188 1783:0.0232 1784:0.0250 1785:0.0238 ...
```

- 振幅 ≈ **2.0×**（0.0250 / 0.0126）、エネルギー ≈ **2.8×** → 2 tap が近接して重畳。
- 予測（後述の 4 倍則）: 61→1783、62→1787（4 サンプル差）→ 主ローブは 1779〜1789 に広がり
  ピークは中央 1784 → **実測と整合**。
- したがって **A**：single partition 内の tap→output mapping は**一貫（順序保持・単調）**。
  partition 境界以前の sample-index traversal 自体は破壊されていない。
- 併せて 107-2（P0+P1, 488 と 2536）の「同一位置への融合」は、4 倍則では
  61→1783 と 317→2807 となり **fusion ではない**（107 の測定は 1783 の単峰のみ）。
  → 107-2 の解釈は本節の 4 倍則で再評価が必要（下記 §6 残課題）。

## 5. 4 倍則（1/2 則の正体）と最有力の発生点

実測 4 点（すべて L0 内、Add=2048、384k エンジン）:

| 48k idx | 384k idx | 実測 output | `1539 + 48k×4` |
|---|---|---|---|
| 61 | 488 | 1783 | **1783** ✓ |
| 317 | 2536 | 2807 | **2807** ✓ |
| 4000 | 32000 | 17539 | **17539** ✓ |
| 7000 | 56000(L1) | 29539 | **29539** ✓ |

**4 点厳密一致**。ここから「48k IR が **192k（×4）** として扱われている」と読める。
すなわち **IR のコンテンツ密度が 192k 相当**で、384k エンジンを走査している
（エンジン側の量子・幾何はすべて 384k/2048 で正しい）。

**最有力の発生点（候補の絞り込み）**

| 候補 | 判定 |
|---|---|
| OS 解決 | 棄却（buildRate=384000、OS=2） |
| engine rebuild | 棄却（irLen/block/L0 幾何すべて整合） |
| Add quantum | 棄却（2048 実測） |
| partition traversal / FDL | 棄却（part=2048, fft=4096, fdl 巡回正常、ring 整合） |
| **IR resample の実効ターゲット** | **残存（最有力）**：IR が 48k→**192k** で作られ、
  その後の DSP 側 resample（→384k）が「同一レート」と誤認して素通しになった、
  または IRState が 384k と宣言されつつ内容が 192k（WORK104 の `sr=192000` transfer と同根） |

CONV_STATUS/GEOM は「宣言・構築」が 384k であることを示すが、**IR コンテンツの密度**は
本 trace では直接測っていない（`IRState.ir` の長さは transfer ログの 504/2544/… のままで、
パディング前）。この 1 点（resample target の世代記録）が未取得であり、`OPEN` の理由。

## 6. 108-4 / 108-6 / 108-7 の状態

- **108-4（OS=4→2 世代）**: 起動直後に `768000/4096` の world が存在することはログで確認済み。
  しかし **probe world は 384000/2048**（本 trace で確定）。「768k が残存して 2× を生む」
  仮説は本測定では**棄却**。世代単位の OS 遷移表（G0..G3）の完全な記録は未完。
- **108-6（IRState vs engine IR copy）**: `IRState.sr(384000) == engine.buildRate(384000)`、
  `engine.irLen(192000) == irLength(192000)`（パディング後 targetLength）。
  `IRState.ir->getNumSamples()`（source, trim済み）は別途 transfer ログに出る（504 等）で、
  **source とパディング後は別物**という二重性は確認どおり。宣言レートは両者一致。
- **108-7/108-8（契約拡張）**: **未実施**（指示どおり。まず declared / actual engine /
  effective traversal の 3 者を確定させる段階）。`IRRuntimeContract` は変更していない。
  「effective traversal rate」はコード上の明示状態として**存在しない**ため、契約候補としては
  `resample target（build）`・`IRState.sampleRate` と `engine.buildRate`・`Add quantum`・
  `L0 partSize/fft` から**決定論的に検証**する形が適切（架空の状態を作らない）。

## 7. 受入条件の状態

| 項目 | 状態 |
|---|---|
| `ConvoPeq.md` を WORK107 後ソースで再生成 | **完了**（5,243,329B / 2026-09-18 19:56:58） |
| p0two 実行 | **完了**（A：時間順序保持） |
| P0+P1 再実行 | **実行中に OS メモリ不足で中断**（GEOM before-capture は取得済・幾何は p0two と同一） |
| test-only trace 常時有効化 | **完了**（診断マクロ外し・RT は relaxed telemetry のみ） |
| `Add(numSamples)` 実測 | **完了 = 2048** |
| `L0.partSize` 実測 | **完了 = 2048（fft=4096）** |
| `IRState.sampleRate` 実測 | **完了 = 384000** |
| engine IR sample rate 実測 | **完了 = 384000（buildRate）** |
| effective processing rate 実測 | **完了 = 384000（UIprep/engine）** |
| OS factor 実測 | **完了 = 2** |
| generation 付与 | **完了**（IRState gen=10 を併記） |
| OS=4→2 世代遷移の追跡 | 部分（768k world の存在は確認、遷移表は未完） |
| `t/2` の発生点を一箇所まで絞る | **「IR resample の実効ターゲット（コンテンツ密度 192k）」まで絞り込み**（最終確定は OPEN） |

## 8. 次工程（WORK109 前の残作業）

1. **IR resample target の世代記録**（唯一の未取得項目）：
   `LoaderThread` の構築時 `sampleRate`（resample 目標）と `IRState.sampleRate` を
   ペアで NonRT 記録し、IR build 時に実際に使われた目標値を確定する。
   （これで「コンテンツ 192k」が PROVEN になる）
2. **P0+P1 の再測定**（前回はメモリ不足で中断）：107-2 の「融合」が 4 倍則の下で
   どう解釈されるかを再確認（4 倍則では 1783 と 2807 に分離するはず）。
3. **108-4 の世代遷移表**（G0..G3）の完全記録。
4. その上で WORK105 契約の拡張（declared ではなく **build 時の resample target** を含める）を
   WORK109 の設計対象にする。**E の修正はさらに後**。
5. 実行環境メモ：長時間の連続 harness 実行で OS メモリが逼迫したため、以降は
   プロセスを逐次終了させてから次を起動する（並列起動を避ける）。

## 9. ツール使用（本作業）

headroom proxy＋context-mode（batch/execute）＋rtk 常時。WSL rg/sed/awk。AiDex（query:
MKLNonUniformConvolver の SetImpulse/Add/Get、ConvolverProcessor の transfer/rebuild）。
ビルド：`vcvarsall x64`＋`cmake --build --target ConvoPeq --target AudioEngineHarness --parallel 3`。
`output_sourcecode_markdown.py` で正本再生成。cppcheck（変更は test-only trace のみ）。
