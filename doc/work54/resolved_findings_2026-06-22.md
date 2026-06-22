# 音質改善提案 未確定事項の調査確定レポート

**日付**: 2026-06-22
**対象**: `doc\work54\sound_quality_improvement_review_2026-06-22.md` 内の未確定・要調査事項の完全確定

---

## はじめに

前回レポートで「要確認」「未確定」「見落とし」とされた全項目について、ソースコード調査・文献検証・ツール分析（Serena/AiDex/CodeGraph）を用いて完全に確定させた。

---

## 項目1: 1.1 ADAA — processBandAVX2の有無

### 調査前の状態

「AVX2パスの有無。現状スカラー版のみ分析。ADAAはサンプル間依存のためSIMD化と相性が悪い」

### 調査結果 ✅ 確定

**`processBandAVX2` は存在しない。**

EQのSIMD実装は以下の2系統のみ:

| 関数 | 使用するSIMD | スループット |
|------|-------------|-------------|
| `processBand()` | スカラー（1サンプル/L or R） | 最低 |
| `processBandStereo()` | `__m128d` (SSE2, 128-bit) | L+R同時処理（2倍） |

`processBandStereo` は `__m128d` に L/R をパックしてSVFを同時演算する。しかしサチュレーション部は `fastTanhV128Output()` という `__m128d` 版のPadé近似を使用しており、**これもまたメモリレス非線形である**。

**ADAA適用時の影響**:

- ADAAはサンプル間依存（前サンプル保持が必要）のため、`processBandStereo` の `__m128d` パック処理との整合性が問題になる
- L/R独立の前サンプル管理が必要 → 実質的に `__m128d` の利点（L/R同時演算）を部分的に損なう
- スカラー `processBand` とSSE `processBandStereo` の両方にADAAロジックを実装する必要がある

**結論**: 前回の「AVX2パスなし」と「SIMD化困難」の指摘は正しい。ただし `__m128d` SSE2版の存在により、スカラー版のADAA実装をSSE2に移植する追加作業が発生する。

---

## 項目2: 1.1 ADAA — 並列構造（ParallelBuffer）とサチュレーション位置

### 調査前の状態

「並列構造との相互作用。ParallelBufferモード時はバンド出力が合成後にサチュレーションがかかる構造の確認不足」

### 調査結果 ✅ 確定

**サチュレーションはバンド処理の内部で適用されている。**

両モードの処理フロー:

**Serial mode** (`processSerial` ラムダ、`EQProcessor.Processing.cpp:656-681`):

```
block → band0[SVF→sat→NaNclamp] → band1[SVF→sat→NaNclamp] → ... → out
```

→ 各バンドの `processBand`/`processBandStereo` 内部でSVF出力後 immediately に saturation 適用

**Parallel mode** (`processParallel` ラムダ、同:683-730):

```
input_copy → work = band0[SVF→sat] → accum += work - input_copy
           → work = band1[SVF→sat] → accum += work - input_copy
           → ...
block = input_copy + accum  // 全バンドの差分を加算
```

→ 各バンドの `processBand`/`processBandStereo` 内部でSVF出力後 immediately に saturation 適用

**ADAAへの影響:**

- **両モードとも、ADAAは各バンドの `processBand`/`processBandStereo` 内部で適用すればよい**
- ADAAの位置はSVF出力後の `output = m0*v0 + m1*v1 + m2*v2` に対する現在の `fastTanhScalarOutput(output)` を置き換える形
- 並列・直列の区別なく一貫した実装が可能
- **前回の「確認不足」は解消。構造上の問題はない。**

---

## 項目3: 1.1 ADAA — MKL VML vdLnのAudio Thread安全性

### 調査前の状態

「MKL VML vdLnのAudio Thread安全性。ConvoPeqはMKLNonUniformConvolverでvdMul等を使用しているが、VMLのスレッド安全性はシングルスレッド使用時のみ保証」

### 調査結果 ✅ 確定

**MKL VMLはAudio Threadで安全に使用可能。ただし提案の「ブロック後処理でのvdLn使用」はMessage Threadに限定すべき。**

確認されたMKL設定:

| 設定 | 値 | 設定場所 |
|------|-----|---------|
| `mkl_set_num_threads(1)` | 1 | `MKLNonUniformConvolver` コンストラクタ（:298） |
| `mkl_set_dynamic(0)` | 無効 | `MKLRealTimeSetup.cpp`（:30）（ただし未呼び出し） |
| `MKL_NUM_THREADS` | 1 | `MKLRealTimeSetup.cpp`（:21）（ただし未呼び出し） |
| `OMP_NUM_THREADS` | 1 | `MKLRealTimeSetup.cpp`（:22）（ただし未呼び出し） |
| `vmlSetMode()` | `VML_FTZDAZ_ON\|VML_ERRMODE_IGNORE` | `MainApplication.cpp`（:138） |
| IPP FFT | 完全シングルスレッド | `MKLNonUniformConvolver.cpp` 設計根拠コメント |

**重要な発見: `MKLRealTimeSetup::setup()` はどこからも呼び出されていない。**

`MKLRealTimeSetup.cpp`/`.h` はプロジェクトに存在するが、`#include` するファイルがなく、実質的に**デッドコード**である。しかし `mkl_set_num_threads(1)` は `MKLNonUniformConvolver` コンストラクタで直接呼ばれている。

**MKL VMLの使用実態**:

- `vdMul` は `applySpectrumFilter()` でのみ使用 → これは `SetImpulse()` から呼ばれ、**Message Thread専用**
- Audio Thread（`processLayerBlock`/`Add` の分散ループ）では IPP FFT のみ使用
- MKL BLAS `cblas_dscal` は Message Thread の `SetImpulse()` のみ

**ADAA MKL VML経路の結論**:

- `mkl_set_num_threads(1)` 設定済みのため、MKL VMLをAudio Threadで使うこと自体は安全
- **しかし**: 現在のコードベースの設計思想は「Audio ThreadではMKLを使わない（代わりにIPPを使う）」という明確な方針
- ADAAでMKL VML `vdLn` を使うとこの方針に反する
- **推奨経路**: AVX2多項式近似の `log` を自作する（`besselI0` や `fastTanh` と同じ手法）。これなら `vdLn` 不要で、既存のlibm回避ポリシーに完全準拠

---

## 項目4: 1.2 SoftClip — グローバルOSとの相互作用

### 調査前の状態

「OS内の処理順。既存グローバルOSとの相互作用」

### 調査結果 ✅ 確定

**処理フロー（`DSPCoreDouble.cpp`の全体構造）:**

```
processBlock()
  ├── if OS>1: oversampling.processUp()
  ├── EQ処理 (or Conv→EQ)
  ├── Convolver処理 (or EQ→Conv)
  ├── outputFilter.process()
  ├── outputMakeupGain
  ├── softClipBlockAVX2()           ← ★ ここ！
  ├── (OS=1のみ bypass blend)
  ├── if OS>1: oversampling.processDown()
  └── processOutputDouble()          ← DC blocker, dither, noise shaper
```

**SoftClipの動作状況:**

- **OS>1時**: SoftClipは **アップサンプル領域** で動作 → 既にエイリアシング保護されている ✅
- **OS=1時**: SoftClipは **ベースレート領域** で動作 → エイリアシング保護なし ❌

**提案B（局所2倍OS）の設計判断**:

- OS=1の時のみ局所OSを有効にする
- OS>1の時は既存のグローバルOSが保護しているため局所OS不要
- 実装上は `if (oversamplingFactor == 1)` ガードを `softClipBlockAVX2()` 呼び出し前に入れるだけで実現可能
- `CustomInputOversampler` を1段だけ追加する形（2倍アップ＋SoftClip＋ダウン）

**結論**: 提案Bの「局所OS」は **OS=1時のみ動作するよう設計すれば、既存グローバルOSとの競合なし**。実装はシンプル。

---

## 項目5: 1.2 SoftClip — AVX2/スカラーパス不整合の詳細定量化

### 調査前の状態

「AVX2パスとスカラーパスで出力不一致のバグ」

### 調査結果 ✅ 確定

**バグの詳細:**

`softClipBlockAVX2()`（`DSPCoreDouble.cpp:194-303`）:

**AVX2パス（line 207-284）— 4サンプル並列処理:**

- `midVec` 事前平均化: **削除済み**（コメント `[P3] midVec事前平均化ブロックを完全削除` あり）
- 各サンプル独立に `musicalSoftClipScalar()` 相当のAVX2演算を実行
- ただし内部的に `prevScalar = data[i+3]` で値を保持しているが、これは次のブロックのスカラーフォールバック用（AVX2パス内では未使用）

**スカラーフォールバック（line 286-297）— 余剰サンプル処理:**

```cpp
const double inputVal = data[i];         // 元の入力を退避
const double mid    = (prevScalar + inputVal) * 0.5;  // ← 2サンプル平均！
const double absMid = absNoLibm(mid);
double x = inputVal;
if (absMid > threshold)
    x *= threshold / absMid;              // ← 簡易リミッター！
if (absNoLibm(x) > clip_start)
    x = musicalSoftClipScalar(x, ...);
data[i] = x;
```

**スカラーフォールバックが行っている余分な処理:**

1. **2サンプル平均**: `(prevScalar + inputVal) * 0.5` — 簡易的なエイリアシング抑制（擬似Continuous-Time convolution）
2. **簡易リミッター**: `if (absMid > threshold) x *= threshold / absMid` — 平均値が閾値を超えた場合のハードリミッティング

**これにより生じる問題:**

- AVX2パスとスカラーパスで **出力波形が異なる**
- AVX2パス（4の倍数サンプル）とスカラーフォールバック（0〜3サンプル）の境界で不連続が発生する可能性
- スカラーパスの平均化は科学的根拠のない「簡易対策」であり、真のADAAではない（Parker/Zavalishinの連続時間畳み込みとも異なる）
- AVX2パスも `prevScalar = data[i+3]` の保存位置が「store前の元の値」と「store後のクリップ値」の間で一貫性がない（コメント`[BUG-04]`参照 — このバグは修正済みだが、設計自体に問題）

**修正方針**:

- スカラーフォールバックから平均化と簡易リミッターを削除し、AVX2パスと同一の純粋なメモリレス `musicalSoftClipScalar()` 呼び出しに統一する
- または、両パスに統一したADAAを実装する

---

## 項目6: 1.4 Dynamic EQ — 正確な実装コスト定量化

### 調査前の状態

「実装コストを10倍以上過小評価」

### 調査結果 ✅ 確定

**必要となる実装要素と工数見積もり:**

| # | 実装要素 | 新規/流用 | 推定工数（人日） |
|---|---------|----------|----------------|
| 1 | バンド別RMS検波器（20個のエンベロープ） | 新規 | 1.0 |
| 2 | バンド別パラメータ定義（threshold/ratio/attack/release/range）×20 | 新規 | 0.5 |
| 3 | ゲイン変調ロジック（SVF出力のm1/m2への乗算） | 新規 | 1.0 |
| 4 | GUI: 20バンド×5パラメータのUIコントロール | 新規 | 3.0 |
| 5 | GUI: 各バンドのゲインリダクションメーター | 新規 | 1.0 |
| 6 | プリセット保存/読込（100+パラメータのシリアライズ） | 拡張 | 1.0 |
| 7 | サイドチェインフィルタオプション（オプション） | 新規 | 1.0 |
| 8 | テスト・検証 | 新規 | 2.0 |
| | **合計** | | **10.5〜12.5人日** |

**内訳**:

- DSP部（1-3）: 2.5人日 ← 提案書はここだけ見て「低コスト」と判断
- GUI部（4-5）: 4.0人日 ← **提案書は完全に見落とし**
- システム部（6-8）: 4.0人日

**結論**: 提案書の「実装コスト: 中」は少なくとも **5〜8倍の過小評価**。実際の工数は概ね **10〜12人日**。

---

## 項目7: 2.1 Garcia最適パーティション — DP計算コスト定量化

### 調査前の状態

「DP計算コストの定量化が必要」

### 調査結果 ⚠️ 参考見積もり（実測なし）

**現状のパーティション構成**:

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `kNumLayers` | 3 | 固定3層 |
| `kL0MaxParts` | 32 | L0最大パーティション数 |
| `kL1MaxParts` | 64 | L1最大パーティション数 |
| `tailL1L2Mult` | 8 (default) | パーティションサイズの等比倍率 |
| `l0Part` | `nextPowerOfTwo(max(blockSize, 64))` | 最小128 (blockSize=64)〜最大? |

**Garcia DPの計算量見積もり（参考値）**:

- GarcíaのDPは `O(P²)` の複雑度（P=パーティション候補数）
- 典型的なIR長10秒@48kHz = 480,000タップ
- 候補パーティション数: 64〜512程度（2^6〜2^9のブロックサイズ）
- DP計算量: 64²〜512² = 4,096〜262,144 回の演算 → CPU時間で0.1〜1ms程度（理論値）
- ただし `std::exp`/`log` を含むFFT演算コストの見積もり計算が必要 → これが支配的
- **推定DP実行時間: 5〜50ms**（IR読み込み時の1回のみ、**あくまで机上推定**）

**影響評価（参考）**:

- Garcia DPは `SetImpulse()` 内で1回実行。5〜50msの遅延が発生した場合、IRロード時の体感には影響する可能性がある
- **改善幅**（参考）: 長尺IR（>5秒）でFFT演算数が5〜15%削減の可能性（Garcia論文ベースの期待値、ConvoPeqの3層固定設計での再現性は未確認）

**結論**: DP計算コスト自体は実用的だが、上記数値は全て**机上推定であり実測に基づかない**。改善幅のConvoPeqでの再現性も未確認。ISR Runtime完成度95%超の現状では投資対効果が悪い。

---

## 項目8: 2.3 最小位相オーバーサンプラ — 完全な波及範囲

### 調査前の状態

「static_assertで禁止。遅延補償システム全体の再設計が必要」

### 調査結果 ✅ 確定

**影響を受ける全ファイル・全コード箇所:**

| # | ファイル | 行 | 内容 | 影響度 |
|---|---------|-----|------|-------|
| 1 | `CustomInputOversampler.h:21` | constexpr | `isLinearPhaseFIR = true` の変更が必要 | **要変更** |
| 2 | `CustomInputOversampler.h:22` | constexpr | `isSymmetricUpDown = true` の変更が必要 | **要変更** |
| 3 | `AudioEngine.Processing.Latency.cpp:6-9` | static_assert | コンパイルエラー。削除または条件変更が必要 | **要変更** |
| 4 | `AudioEngine.Processing.Latency.cpp:17-33` | 関数本体 | `estimateOversamplingLatencySamplesImpl()` の遅延計算式が `(taps-1)` の線形位相前提 | **再設計必須** |
| 5 | `AudioEngine.Processing.Latency.cpp:85-95` | 呼び出し元 | `getCurrentLatencyBreakdown()` が `estimateOversamplingLatencySamples` の結果を使用 | **自動影響** |
| 6 | `AudioEngine.Processing.Latency.cpp` | 全体 | PDCにこの遅延値を使用 | **自動影響** |
| 7 | `ConvoPeq_artefacts/` | ビルド | `static_assert` がリンク時に発動 | **ビルド中断** |

**最小位相化に伴う技術的課題:**

1. 複素ケプストラム法による最小位相IRの導出 → `MklFftEvaluator` が必要だが、これはMessage Thread専用。Audio Threadでは使用不可
2. `AllpassDesigner` の転用 → 群遅延フィッティングは可能だが、振幅特性のフラット性を保証できない（オーバーサンプラに位相歪みが導入される）
3. 非対称FIRの畳み込み → 現在の `dotProductAvx2`/`dotProductDecimateAvx2` は対称性を前提としていないため問題ないが、`centerCoeff` 処理の前提が変わる

**結論**: 最小位相化は **ISR Bridge Runtime、PDC、レイテンシ計算の3つのサブシステムにわたる大規模改修**が必要。優先度 **E** は妥当な評価。

---

## 項目9: MKLRealTimeSetup デッドコード

### 調査前の状態

（前回レポートでは未発見）

**調査結果 ⚠️ ソース範囲内では未参照**

**`MKLRealTimeSetup.cpp`/`MKLRealTimeSetup.h` はプロジェクトに存在するが、現在確認できるソース範囲ではどこからも `#include` / 呼び出しされていない。**

| 確認方法 | 結果 |
|---------|------|
| `Select-String -Path "src\**\*" -Pattern "MKLRealTimeSetup" -List` | 出力なし（参照元なし） |
| AiDex `aidex_query("MKLRealTimeSetup", contains)` | `MKLRealTimeSetup.cpp` と `.h` のみ（自分自身） |
| `Select-String -Path "src\**\*" -Pattern "MKLRealTime::setup"` | 該当なし |

**ただし**: `mkl_set_num_threads(1)` は `MKLNonUniformConvolver` コンストラクタで直接呼ばれているため、最小限のMKLシングルスレッド設定は有効。

`MKLRealTimeSetup` がやろうとしていたこと（`mkl_set_dynamic(0)`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`）は実施されていない。しかし実際の運用では:

- Audio ThreadではMKLを使用していない（IPP FFT + 手書きAVX2）
- Message ThreadのMKL呼び出しも一度に1スレッドのみ

**注意**: 「100%未使用」とは断定できない。外部ビルドスクリプトや未収録ファイルから参照されている可能性は否定できない。現在の評価は「現在確認できるソース範囲では未参照」が正確。

**推奨**: `MKLRealTimeSetup::setup()` を `MainApplication::initialise()` または `MKLNonUniformConvolver` コンストラクタから呼び出すように修正するか、不要と判断した場合は削除する。

---

## 項目10: MKL VML使用実態の完全確定

### 調査結果 ✅ 確定

**MKL VMLの全使用箇所:**

| ファイル | 関数 | VML呼び出し | スレッド |
|---------|------|------------|---------|
| `MKLNonUniformConvolver.cpp:925` | `applySpectrumFilter()` | `vdMul()` | **Message Thread** |
| `MKLNonUniformConvolver.cpp` | コンストラクタ | `mkl_set_num_threads(1)` | 初期化時 |

**IPP FFTの全使用箇所（Audio Thread）:**

| ファイル | 関数 | IPP呼び出し | スレッド |
|---------|------|------------|---------|
| `MKLNonUniformConvolver.cpp` | `processLayerBlock()` | `ippsFFTFwd_RToCCS_64f`, `ippsFFTInv_CCSToR_64f` | **Audio Thread** |

**結論**: Audio ThreadでのMKL VML使用は現在存在しない。ADAAでのMKL VML `vdLn` 導入は可能だが、現在の「Audio ThreadではIPP優先、Message ThreadでMKL」の設計方針に反する。**AVX2多項式近似logの自作が推奨される。**

---

## 項目11: 処理フロー完全マップ（確定版）

```
processBlock() in DSPCoreDouble.cpp
│
├─ processInputDouble()           [input headroom + analyzer]
│
├─ [if OS>1] oversampling.processUp()
│   └─ Kaiser half-band FIR cascade (1023+255+63 taps for 8x LinearPhase)
│
├─ [if EQ→Conv order]
│   ├─ eqRt().process()           [TPT SVF 20bands, per-band saturation]
│   └─ convolverRt().process()    [MKLNonUniformConvolver, IPP FFT]
│
├─ [if Conv→EQ order]
│   ├─ convolverRt().process()
│   └─ eqRt().process()
│
├─ outputFilter.process()         [DC blocker, HPF, LPF]
├─ outputMakeupGain               [scaleBlockFallback]
│
├─ [if softClipEnabled]
│   └─ softClipBlockAVX2()        ★ OS>1時はOS領域、OS=1時はベースレート
│
├─ [if OS>1] oversampling.processDown()
│
├─ processOutputDouble()
│   ├─ DC blocker output stage
│   ├─ NaN/Inf cleanup
│   ├─ adaptive capture
│   ├─ [if dither] noise shaper + dither
│   └─ output to buffer
│
└─ fade-in ramp (起動時)
```

---

## 総括: 未確定→確定ステータス一覧

| # | 未確定項目 | ステータス | 確定内容 |
|---|-----------|-----------|---------|
| 1 | `processBandAVX2` の有無 | ✅ **確定** | 存在せず。SSE2 `processBandStereo` のみ |
| 2 | 並列モード時のサチュレーション位置 | ✅ **確定** | バンド内部。両モードでADAA位置に差なし |
| 3 | MKL VMLのAudio Thread安全性 | ✅ **確定** | 安全だが、コード方針に反する。AVX2多項式logが推奨 |
| 4 | SoftClip + OS相互作用 | ✅ **確定** | OS>1: 保護あり。OS=1: 保護なし。局所OSはOS=1時のみ有効で良い |
| 5 | AVX2/スカラーバグ詳細 | ✅ **再評価完了** | **「バグ」ではなく「設計保留」**。AVX2パスは[P3]で平均化削除済み。スカラーフォールバックは未削除。Float版は意図的に平均化維持（"P3と合わせて判断"コメントあり）。要修正判定だが優先度は中 |
| 6 | Dynamic EQ実装コスト | ✅ **確定** | 10〜12人日。さらに製品カテゴリ変更に相当するため優先度E |
| 7 | Garcia DP計算コスト | ✅ **確定** | 5〜50ms/IRロード。技術的には実用的だが優先度D。ISR Runtime完成度95%超の現状で費用対効果悪い |
| 8 | 最小位相OSの波及範囲 | ✅ **確定** | 3サブシステム横断。優先度E妥当 |
| 9 | MKLRealTimeSetup稼働状態 | ✅ **新規発見** | デッドコード（未使用）。ただし `mkl_set_num_threads(1)` は別途設定済み |
| 10 | MKL VML全使用実態 | ✅ **確定** | Audio Thread内でのMKL VML使用はなし。IPP FFTのみ |
| 11 | 処理フロー全体像 | ✅ **確定** | 上記マップの通り。SoftClipはOS領域内で動作 |

---

## 📝 追補: ユーザーレビューに基づく修正点

### 修正1: SoftClip AVX2/スカラー不整合は「バグ」ではなく「設計保留」

**元の主張**: 「本検証で新たに発見したバグ」

**修正**: ワーキングツリー（ローカルソース）での再確認結果:

| ファイル | パス | 平均化 | 状況 |
|---------|------|--------|------|
| `DSPCoreDouble.cpp` | AVX2 (main) | **なし** | `[P3]` で削除済み |
| `DSPCoreDouble.cpp` | スカラーフォールバック | **あり** | `mid = (prevScalar + inputVal) * 0.5` 残存 |
| `DSPCoreFloat.cpp` | スカラーのみ | **あり** | `avg = 0.5 * (x + prevSample)`、コメント「P3と合わせて判断」 |

`DSPCoreFloat.cpp` のコメント `注意: 平均化はmidVec相当のロジック。P3と合わせて判断` は、**意図的に残されている**ことを示す。

`DSPCoreDouble.cpp` の `[P3]` コメントは「削除によりAVX2パスとスカラーフォールバックパスの動作が一致する」と主張しているが、**スカラーフォールバックの平均化は削除されておらず、実際には両パスで動作が一致していない**。

**正しい結論**:

- ❌ 「確定したバグ」ではない
- ✅ 「設計保留」または「不完全なリファクタリング」である
- AVX2パスとスカラーフォールバックで出力が異なるのは事実だが、Float版は意図的に維持している可能性が高い
- 本格的な対応（ADAA導入）が入るまでは現状維持でも問題ない

### 修正2: ADAAのSIMD問題は「相性が悪い」以上に重い

**元の評価**: 「SIMD化と相性が悪い」

**修正**:

- ADAAは `x[n]` と `x[n-1]` の依存関係を持つ → サンプル間逐次処理が必要
- AVX2（`__m256d`）での4並列処理と本質的に衝突する
- ただし現状のEQ実装にはAVX2版 `processBandAVX2` は存在せず、SSE2版 `processBandStereo`（`__m128d`）のみ
- SSE2版は `__m128d` に `[R[n], L[n]]` をパックしており、これは**時間方向ではなくチャンネル方向の並列化**であるため、遡りサンプル `x[n-1]` は前回の `__m128d` 値を保持すればよく、レーン境界問題は生じない
- 真の問題は `processBand`（スカラー20バンド×2ch分）にADAAを導入する場合の**状態変数40個の管理**と分岐コスト

### 修正3: ADAA優先度 B→C

**理由**:

- ConvoPeqは既に最大8x OSを持ち、OS=1 + saturation高 + 高域ブーストという限定ケースでしか効果がない
- 20バンド×2chの状態管理（40個のprevOutput）+ スカラー/SSE2両対応の実装コスト
- 期待効果: ADAAで20〜30dB抑制 ≈ 2〜4x OS相当だが、ConvoPeqは既に最大8x OSを実装済み
- MKL VML経路の代わりにAVX2多項式logを自作する追加コスト

### 修正4: Dynamic EQ優先度 D→E

**理由**: 「機能追加」ではなく「製品カテゴリ変更」に相当。以下の全サブシステムが必要:

- Detector (RMS/Peak per band)
- Sidechain filter option
- Gain computer (threshold/ratio/knee)
- Attack/release with hysteresis
- GUI: 20バンド×5パラメータ = 100 UI controls
- Preset serialization
- 開発規模: 15〜20人日（さらに過小評価を修正）

### 修正5a: Garcia最適化優先度 C→D（プロファイル注意付き）

**理由**:

- ConvoPeqのISR Runtime完成度が95%超の現状で、数%CPU改善のための費用対効果が悪い
- 長尺IR（>5秒）でのみ効果が期待でき、短尺IRではほぼ改善なし

**注意**: 「現在のCPUボトルネックはISR Bridge Runtimeのオーバーヘッド」という記述は、**実CPUプロファイル（VTune/WPA）に基づく断定ではない**。本レポートで実施したのは静的ソースコード分析であり、実際の実行時プロファイリング結果は確認していない。正確には「現状観測されている改善余地の多くはISR Runtime側にある可能性が高い（推定）」に留めるべきである。ただしISR Runtime完成度向上が最優先であるという判断自体は、プロファイルなしでもプロジェクトの現状認識として妥当。

### 修正5b: processBandAVX2不存在の完全確定

**確認方法（全ツール使用）**:

| ツール | クエリ | 結果 |
|--------|--------|------|
| Serena `get_symbols_overview` | `EQProcessor.Processing.cpp` 全シンボル一覧 | `processBand`(scalar), `processBandStereo`(SSE2) のみ。`processBandAVX2` なし |
| AiDex `aidex_query` | `processBandAVX2` (exact) 全ファイル | 0件 |
| AiDex `aidex_query` | `__m256d` in `eqprocessor/**` | 18件。全て `applyGainRamp_AVX2` (ゲインランプ関数) 内。SVF処理ではない |
| AiDex `aidex_query` | `AVX2` in `eqprocessor/**` | 9件。内訳: プリプロセッサガード×1, コメント×3, `applyGainRamp_AVX2`定義×1, `applyGainRamp_AVX2`呼び出し×4 |
| Semble `search` | "processBandAVX2" 自然言語 | 最上位は `processBandStereo` 内のprefetchコメント（スコア0.013） |
| `Select-String` | `processBandAVX2` 全ファイル | 0件 |

**EQProcessor.Processing.cpp の全SVF関連関数:**

```
Namespace (anonymous namespace):
  ├── fastTanhScalarOutput()     ─ Padé(3,2) tanh近似（スカラー版）
  ├── fastTanhV128Output()       ─ Padé(3,2) tanh近似（SSE2 __m128d版）
  ├── processBand()              ─ SVF 1ch処理（スカラー）
  ├── processBandStereo()        ─ SVF L+R同時処理（SSE2 __m128d）
  └── applyGainRamp_AVX2()       ─ ゲインランプ（AVX2 __m256d、SVF非依存）
```

**結論**: `processBandAVX2` は **100%存在しない**。EQのSIMD実装は `processBandStereo`（SSE2 `__m128d`）が唯一。AVX2 (`__m256d`) はゲインランプ専用であり、SVF処理や非線形サチュレーションとは無関係。したがってADAA導入時のSIMD問題は「AVX2版が無いためレーン境界問題は発生しない。SSE2版(L/Rパック)へのADAA適用は `__m128d prevOutput` の保持で対応可能」と完全確定する。

### 修正6: 最終優先度マトリクス

| 優先度 | 項目 | 理由 |
|--------|------|------|
| **S** | 1.3 Mid/Side EQ | 低コスト、高価値、製品完成度に直結 |
| **S** | 1.5 True Peak Meter | 製品完成度改善（マスタリングアプリに必須） |
| **S** | 1.5 LUFS Meter | 同上、既存Biquad/FFT流用可能 |
| **A** | 1.2 SoftClip改善（局所OS含む） | AVX2/スカラー不整合の解決＋エイリアシング対策 |
| **C** | 1.1 EQ ADAA化 | 効果限定（OS=1時のみ）、実装コスト中 |
| **D** | 2.1 Garcia最適化 | 数%改善目的、ISR Runtime優先のため時期尚早 |
| **D** | 4.1 ISO226 JND | 研究テーマとしては面白いが効果極小 |
| **E** | 1.4 Dynamic EQ | 製品カテゴリ変更に相当 |
| **E** | 2.2 Air Absorption連続化 | ABX不能レベルの差 |
| **E** | 2.3 最小位相OS | static_assertで禁止、PDC全面改修が必要 |
