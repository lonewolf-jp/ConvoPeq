# WORK109 — IR Resample Target / Content-Density Origin Audit

- **作成日**: 2026-09-18
- **種別**: Measurement（resample target と実データ密度の実測確定）
- **前工程**: WORK108（`doc/work108/effective_rate_origin_20260918.md`）
- **制約遵守**: `IRRuntimeContract` 未変更 / E・F4・F5 未変更 / RT に validation・branch・allocation・logging なし /
  `effectiveTraversalRate` 等の新規状態を追加しない / 症状補正（/2 等）なし

```text
EFFECTIVE-RATE ORIGIN

Source:
    sr  = 48000
    len = 62 (delta[61]) / 318 (p1[317]) / 318 (p0p1[61,317]) / 63 (p0two[61,62])

Resample:
    targetSr     = 384000
    convertedLen = 496 / 2544 / 2544 / 504
    ratio        = 8.0000 (all probes)   ← 48k→384k は ×8 で正しい
    srcArgmax    = 61 / 317 / 61 / 61
    convArgmax   = 488 / 2536 / 2536(max) / 492

IRState:
    sr         = 384000
    len        = 496 / 2544 / 2544 / 504   (★ パディング前。engine へは irLen=192000)
    generation = 10-12

Engine:
    buildRate  = 384000
    irLen      = 192000
    block      = 2048  (= base 1024 × OS 2)
    layers     = 2 (L0 + L1)
    L0         = part 2048 / numIR 23 / numParts 32 / fft 4096 / ring 16384
    generation = 10-12

Observed:
    source(48k) → converted(384k) → capture(192k)
      61  →  488  →  1783      (delta 単独)
     317  → 2536  →  2807      (p1 単独)
    mapping: output(192k) = 1539 + 4×idx48k = 1539 + convIdx/2
    ★ /2 は OS=2 の decimation。384k エンジン出力を 192k base で観測しているため。
      = 正しい挙動（欠陥ではない）

Origin:
    RESAMPLE ... なし（ratio=8.0000、argmax 488/2536 が ×8 と厳密一致）
    COPY ...... なし（IRState sr=384000 / len=496-2544 が converted と一致）
    ENGINE .... ★ あり（L0 partition 間の重ね合わせ破綻。下記 §5）
    → 分類: ENGINE（partition traversal）

Confidence:
    192k-content-density 仮説 : REFUTED（PROVEN。resample は正しい）
    ×4 = capture-domain      : PROVEN（1539 + 4×idx48k が単一 tap で厳密一致）
    P0+P1 重ね合わせ破綻      : STRONGLY SUPPORTED（4 条件の実測行列）
    発生点の最終行特定        : OPEN（L0 partition 書き込み/読み出しオフセット）
```

---

## 1. 109-0 targetSampleRate の生成元（コード追跡）

`ConvertConfig::targetSampleRate` の設定元は 2 系統のみ。

| 経路 | 設定箇所 | 値の由来 |
|---|---|---|
| File→Converter（UI ライブラリ/RCU） | `src/IRConverter.cpp:384` `cfg.targetSampleRate = sampleRate;`（`convertToHighRes`） | 呼び出し元が渡す `sampleRate` |
| Lifetime/低解像度（`loadIR`） | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:280` `cfg.targetSampleRate = sr;` | `sr = consumeAtomic(currentSampleRate)`（`:245`） |
| LoaderThread（probe が通る経路） | `src/convolver/ConvolverProcessor.LoaderThread.cpp:566` の比較対象 `sampleRate` | `processingSampleRate`（LoadPipeline.cpp:57/86、Rebuild.cpp:77/88）＝ `consumeAtomic(currentSampleRate)` |

- `convertFile()` は `sourceRate != targetSampleRate` のときのみ
  `IRDSP::resampleIR(ir, sourceRate, config.targetSampleRate, ...)` を呼び、
  成功時 `actualSampleRate = config.targetSampleRate`、失敗時は **元 IR へフォールバックし
  `actualSampleRate = sourceRate`**（`IRConverter.cpp:258-280`）。この fallback 分岐は実在する。
- ただし probe（`loadImpulseResponse`）は **LoaderThread 経路**であり `convertFile` を通らない。

## 2. 109-1/109-2 実測：resample 世代ペアとサンプル数比（決定打）

`ConvolverProcessor.LoaderThread::doTrimStep()` の resample 境界に
**NonRT 数値 trace `[IR_RATE_GEN]`**（stderr）を追加して実測した
（変更は trace のみ。RT 経路・挙動・契約は一切変更なし）。

```text
delta  : gen=0 sourceSr=48000 targetSr=384000 actualSr=384000 sourceLen=62  convertedLen=496  ratio=8.0000 resampled=yes srcArgmax=61  convArgmax=488
p1     : gen=0 sourceSr=48000 targetSr=384000 actualSr=384000 sourceLen=318 convertedLen=2544 ratio=8.0000 resampled=yes srcArgmax=317 convArgmax=2536
p0p1   : gen=0 sourceSr=48000 targetSr=384000 actualSr=384000 sourceLen=318 convertedLen=2544 ratio=8.0000 resampled=yes srcArgmax=61  convArgmax=2536
p0two  : gen=0 sourceSr=48000 targetSr=384000 actualSr=384000 sourceLen=63  convertedLen=504  ratio=8.0000 resampled=yes srcArgmax=61  convArgmax=492
```

**判定**

| 判定項目 | 結果 |
|---|---|
| `ratio == 4`（192k コンテンツ） | **NO**（全 probe で 8.0000） |
| `ratio == 8`（384k コンテンツ） | **YES** |
| `srcArgmax → convArgmax` の ×8 対応 | **厳密一致**（61→488、317→2536） |
| **Case A（192k content を 384k とラベル）** | **REFUTED** |
| **Case B（384k content を 384k エンジン）** | **CONFIRMED** |
| Case C（resample failure fallback） | 発生せず（`resampled=yes`、fallback ログなし） |

さらに DSP 側 rebuild でも整合を確認：

```text
[CONV_IR]         transferIRStateFrom: IR transferred ch=2 len=496/2544 sr=384000.0 block=2048 gen=10-12
[IR_RATE_GEN]     gen=0 sourceSr=384000 targetSr=384000 actualSr=384000 ... ratio=1.0000 resampled=no
[CONV_REBUILD]    rebuildAllIRsSynchronous: engine rebuilt len=2544 ch=2 srcSR=384000.0
```

→ **IRConverter / LoaderThread / IRState / engine のレート宣言と実データ密度はすべて一致**。
WORK106〜108 の「192k コンテンツを 384k とラベル」仮説は**棄却**。

## 3. 109-3 実データ位置：`source → converted → engine → output` の厳密対応

WORK108 の `output = 1539 + 4×index48k` の **`4×` の正体**を、単一 tap で厳密に確定した。

| probe | source(48k) | converted(384k) | output(capture) | 期待式 | 一致 |
|---|---|---|---|---|---|
| delta | 61 | 488 | **1783** | 1539 + 488/2 = 1783 | ✓ 厳密 |
| p1 | 317 | 2536 | **2807** | 1539 + 2536/2 = 2807 | ✓ 厳密 |
| mid(WORK108) | 4000 | 32000 | 17539 | 1539 + 32000/2 = 17539 | ✓ 厳密 |
| far(WORK108) | 7000 | 56000 | 29539 | 1539 + 56000/2 = 29539 | ✓ 厳密 |

- `convIdx = 8 × idx48k`（resample が ×8、実測どおり）
- `output  = 1539 + convIdx/2`（**capture は 192k base**。エンジンは 384k＝OS 2）
- したがって `output = 1539 + 4×idx48k` は
  **「48k → 192k の正しい写像」**であり、`t/2` の異常ではない。

**結論**：WORK106〜108 が異常とした `t/2` は、**観測ドメイン（192k）を 384k と取り違えた
解析側のラベリング誤り**であった。エンジンのサンプル座標変換は正しい。
（`UX_PREP block=2048 = base 1024 × OS 2` がこのドメイン関係を裏付ける。）

## 4. 109-4 Origin 分類

| 候補 | 実測 | 判定 |
|---|---|---|
| RESAMPLE（target が 192k） | ratio=8.0 / targetSr=384000 | **棄却** |
| COPY（IRState と engine IR の不一致） | len=496/2544・sr=384000 が一致 | **棄却** |
| ENGINE（rate/量子/幾何） | buildRate=384000 block=2048 part=2048 fft=4096 ring=16384 | **棄却**（WORK108 と同じ） |
| ENGINE（**L0 partition 間の重ね合わせ**） | §5 の実測行列 | **残存（最有力）** |

## 5. 109-5 P0+P1 逐次再測定 → **重ね合わせ破綻を発見**

メモリ逼迫のため**逐次起動**（1 プロセスずつ終了）で 4 条件を同一バイナリ・同一設定で取得。

| probe | source taps(48k) | converted(384k) | output peak | 振幅 | energy | secondaryPeak |
|---|---|---|---|---|---|---|
| delta | 61 | 488 | 1783 | 0.012629 | 0.001118 | 1749: 0.0025 |
| p1 | 317 | 2536 | **2807** | 0.012374 | 0.001098 | — |
| p0two | 61,62 | 488,496 | 1784 | 0.024981 | 0.003075 | — |
| **p0p1** | **61,317** | **488,2536** | **1783 のみ** | **0.024469** | **0.002603** | **1822: 0.0006（2807 に有意峰なし）** |

**線形性（重ね合わせ）の検証**

- 期待（LTI）: p0p1 = delta + p1 → **1783 と 2807 の 2 峰**、各振幅 ≈ 0.0126、energy ≈ 0.0022。
- 実測: **1783 の 1 峰のみ**、振幅 **≈2×**（0.0245）、energy 2.3×。
- p1 単独では 317→2807 が正しく出る。**61 tap を追加した瞬間に 317 tap の峰が消え、
  そのエネルギーが 1783 に融合する**（= superpositon の破綻）。
- 幾何（ENG/RT/ring/fdl）は 4 条件で同一。generation 差（10 vs 12）以外に差はない。

**推定メカニズム（未確定・要検証）**

```
L0 partition 1 の tap（intra-index 488 = 2536 - 2048）が、
partition 0 の intra-index 488（= 61 tap）と同じ出力位置に放出されている
  → partition offset (p × partSize) が L0 読み出しで適用されていない可能性
```

- この機構なら p0p1 は「1783 に 2 tap 融合」となり、振幅 2× を説明できる。
- ただし p1 単独では 2807 に正しく出るため、
  「partition 0 が空のときだけ offset が効く」という条件依存があり、
  単純な offset 欠落ではない。**L0 FDL の書き込み/読み出しスロット対応**の調査が必要。
- この現象は低域 IR の後部反射を頭部へ折り畳むため、**ユーザー症状「ベースのジジジ」と
  整合する時間軸スメア**を生む（E の実体候補）。

## 6. 109-6 / 109-7

- **109-6（OS G0..G3）**: 凍結（probe world は 384000/2048、base 192000/1024 で確定）。
- **109-7（契約変更禁止）**: 遵守。`IRRuntimeContract` は**未変更**。
  新規状態 `effectiveTraversalRate` 等も**追加していない**。
  既存の `sourceRate / targetSampleRate / actualSampleRate / IRState.sampleRate /
  engine buildRate / sourceLength / convertedLength / generation` だけで
  本 WORK の判定（ratio・argmax 対応）は**決定論的に導出できた**。
  すなわち **契約拡張は不要**（Build→Validate→Publish の責務分離は現状で足りる）。

## 7. 受入条件の達成状況

```text
[x] sourceSr を取得                         (48000)
[x] targetSampleRate を取得                 (384000)
[x] actualSampleRate を取得                 (384000)
[x] sourceLen を取得                        (62/318/318/63)
[x] convertedLen を取得                     (496/2544/2544/504)
[x] generation を一致させる                 (gen=0 を同一行で取得)
[x] source→converted の sample 数比を算出   (8.0000)
[x] converted 実データの non-zero index を確認 (srcArgmax→convArgmax の ×8 厳密対応)
[x] engine SetImpulse に渡った IR の対応を確認 (transferIRStateFrom len/sr、rebuild len/srcSR)
[x] P0+P1 を逐次プロセスで再測定            (4 条件を逐次取得)
[x] 192k content-density の PROVEN/REFUTED  (REFUTED = 384k が正しいと PROVEN)
[x] IRRuntimeContract は未変更              (変更なし)
[x] E/F4/F5 は未変更                        (変更なし)
```

## 8. 次工程（WORK110 候補）

1. **最優先**: L0 partition 間の重ね合わせ破綻の行特定。
   `SetImpulse` の NonRT trace を拡張し、(a) L0 各 partition に対する書き込み先
   (part, intra-index) と (b) `Add` 時の FDL 読み出しスロット（`fdlIndex` と
   partition offset の適用）を同一 generation で 1 行に出す。
   → 「書き込み側の offset 欠落」か「読み出し側の offset 欠落」か「条件依存の分岐」かを確定。
2. p0+p1 の捕捉出力を全サンプル CSV 化（`--buzz-out` は probe 経路未対応のため、
   probe 経路にも CSV 出力を追加）して 2 峰の有無を独立検証。
3. 上記で確定後、WORK105 契約の**拡張要否**を判断（現時点では不要と判定）。
4. E の修正は**さらに後**（現象の行特定が先）。

## 9. ツール使用（本作業）

headroom proxy＋context-mode（ctx_execute/ctx_batch_execute）＋rtk(WSL) 常時。
WSL `rg`/`grep -a`/`sed`。AiDex（`aidex_query`）で `LoaderThread` / `IRConverter` /
`MKLNonUniformConvolver::SetImpulse` を特定。serena 設定（`memory`）で構造を記録。
ビルドは `vcvarsall.bat x64` → `cmake --build build --config Release --target
AudioEngineHarness / ConvoPeq`（`build.bat` 単独は `cl` 不在で失敗するため vcvars 必須）。
`build.bat` 一括ビルドは無関係ターゲット `RuntimeHealthMonitorTierTests` の `mkl.h` 不在で
失敗するため、必要なターゲットを個別指定した。

## 10. 追加した trace（変更点の全量）

| ファイル | 変更 | 種別 |
|---|---|---|
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | `doTrimStep()` に `[IR_RATE_GEN]` 数値 trace（resample 前後の sr/len/argmax/ratio/generation） | NonRT 観測のみ |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | probe 解析に `[PROBE] secondaryPeak`（global peak ±32 除外の最大値）を追加 | test-only |

- いずれも **RT 経路に分岐・確保・ログ・状態変更を追加しない**。
- 既存の幾何（`ENG/RT`）・`[CONV_IR]`・`[CONV_REBUILD]` は WORK108 のまま。
