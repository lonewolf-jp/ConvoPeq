# Step 0: NUPC content-mapping audit（delayLine 位置 ↔ 出力波形の対応確定）

- **日付**: 2026-09-11
- **基準ソース**: `ConvoPeq.md` Generated 2026-09-11 20:16:16（`MKLNonUniformConvolver.cpp` 抽出内の行番号は ConvoPeq.md 全体行番号ではなく抽出セクション内の相対行番号）
- **目的**: `MKLNonUniformConvolver` の L1/L2 分散パスを 1 回完全に追跡し、「delayLine 論理位置 `p` に書かれた内容が数学的に何であるか」を確定する。`content_time(j) = 512j + IR_offset_layer`（Gardner Null Test v2.3/v2.4 §2.2）の証明。
- **位置づけ**: 手順書 v2.4 の Step 0（レビュー第5ラウンド要求）。Step 0 が確定しないまま実測すると、`outputPlacementError` が異常値を示した際に「B13 read alignment の問題」と「content_time 定義の誤り」を区別できない。

---

## 1. 対象構成（48 kHz / blockSize = 64 / IR = 5000 / tailMode = 1）

| 項目 | 値 | 出典 |
|---|---|---|
| L1 partSize / fftSize | 512 / **1024**（= 2×partSize） | cpp:780-781 |
| L1 numPartsIR / numParts | 6 / 8（nextPowerOfTwo） | cpp:787-788 |
| L1 offset / len | 2048 / 2952 | cpp:750-751 |
| L1 outputDelaySamples | 2048（暫定値） | cpp:1009 |
| L1 delayLineCapacity | (2048+512+64+15)/16×16 = **2624**（整数除算: 2639/16=164） | cpp:1010 |
| complexSize / partStride | 513 / 520（=(513×2+7)&~7） | cpp:784-785 |
| IR パーティション | H1_q = DFT_1024(IR[2048+512q .. 2048+512q+512) を 1024 にゼロパッド) | cpp:922-948 |

## 2. 追跡の前提（確認済みコード事実）

1. **Forward FFT 入力の組立**（Add() 内 L1/L2 パス、cpp:1547-1549）:
   ```cpp
   copy(l.fftTimeBuf,             l.prevInputBuf, partSize);   // [0,512)  ← 前回ブロック
   copy(l.fftTimeBuf + partSize,  l.inputAccBuf,  partSize);   // [512,1024) ← 今回ブロック
   copy(l.prevInputBuf, l.inputAccBuf, partSize);              // prev ← current
   ```
   → ブロック j（j 番目の 512 蓄積）の Forward FFT 入力は `x[512(j−1) .. 512(j+1))`（1024 長 Overlap-Save 窓）。
   記号: `X_b := DFT_1024(x[512(b−1) .. 512(b+1)))`。

2. **FDL 格納と mirror**（cpp:1552-1575）:
   - `X_j` は SoA スロット `fdlIndex` と `fdlIndex + numParts`（mirror）に**同時書き込み**（cpp:1557-1560, 1563-1570）→ SoA 空間 `[0, 2·numParts)` は `[0, numParts)` と常に同内容の複製。
   - `l.fdlIndex = (l.fdlIndex+1) & l.fdlMask`（cpp:1572）、`l.baseFdlIdxSaved = (fdlIndex−1+numParts) & fdlMask`（cpp:1575）= ブロック j のスロット番号 `j mod numParts`。

3. **IR パーティション生成と逆順 swap**:
   - cpp:924-932: tempTime をゼロクリア後、`IR[2048 + 512q .. +512)` のみコピー（IR 範囲外はゼロ）→ Forward FFT → deinterleave（cpp:945-948）。
   - cpp:960-988: SoA の slot `pf` と slot `numPartsIR−1−pf` を swap → **`irFreqSoA[p] = H1_{numPartsIR−1−p}`**。

4. **複素演算**:
   - `deinterleaveComplex` / `interleaveComplex`: 標準 CCS↔SoA 変換（cpp:135-151）。
   - `accumulateSplitComplex`: 分割複素の正規複素積＋累積（cpp:153-、AVX2 FMA / スカラー同値）。

## 3. 分散積算のインデックス解決

分散ループ（cpp:1590-1611、`distributing == true` のコールバックで毎回 `partsPerCallback` 分）:

```cpp
const int baseFdlIdx = l.baseFdlIdxSaved;                       // = j mod numParts
const int linStart   = baseFdlIdx − l.numPartsIR + 1 + l.numParts;   // cpp:1592
const int index      = linStart + p;                            // cpp:1598
accumulate(X_{index}, irFreqSoA[p]);                            // cpp:1610
```

- `index` は SoA 空間 `[0, 2·numParts)` のスロット番号。`index ≡ (j − numPartsIR + 1 + p) (mod numParts)` であり、mirror 複製（前提 2）により SoA `index` の内容 = FDL 実スロット `(j − numPartsIR + 1 + p) mod numParts` の内容。
- FDL depth `numParts = 8` ≥ 使用幅 `numPartsIR = 6`: 連続 6 ブロック `j−5 .. j` は 8 スロットリング内で**衝突なし** → スロット `(j−5+p) mod 8` の内容は確かにブロック `j−5+p` の `X`。
- IR 逆順置換: `irFreqSoA[p] = H1_{numPartsIR−1−p}`。
- 置換 `q := numPartsIR−1−p` により:

```
accum_j = Σ_{p=0}^{numPartsIR−1} X_{j−numPartsIR+1+p} · irFreqSoA[p]
        = Σ_{q=0}^{numPartsIR−1} X_{j−q} · H1_q          （逆順同士の相殺）
```

境界チェック（IR=5000, numParts=8, j mod 8 = 0 の worst case）: `index = (j mod 8) + 3 + p = 3+p`、p=5 → index=8（mirror slot 0）⊂ [0,16)。mirror slot 0 は「直近 Forward FFT（ブロック j、slot 0）の同時複製」= `X_j`。期待値 `X_{j−0}` = `X_j` ✓ 一致。全 p で成立。

## 4. IFFT と有効区間の時間領域展開

- `accumBuf = interleave(accum_j)`（cpp:1615-1616）、`fftOutBuf = IFFT_1024(accumBuf)`（cpp:1622）。
- `tailOutputBuf = fftOutBuf[512 .. 1024)`（cpp:1626、`fftOutBuf + l.partSize` から partSize コピー）。

**循環折り返し解析**: 各項 `(x-window_{j−q} ⊛_circ h1_q)[n] = Σ_{m=0}^{511} h1_q[m] · x[512(j−q−1) + ((n−m) mod 1024)]`。有効区間 `n = 512+τ`（τ ∈ [0,512)）では `n−m ∈ [1, 1023]` で **mod 折り返しなし**（h1_q は 512 長、窓は 1024 長）。したがって:

```
tailOutput_j[τ] = Σ_{q} Σ_{m=0}^{511} h1_q[m] · x[512(j−q−1) + 512 + τ − m]
                = Σ_{q} Σ_{m=0}^{511} IR[2048+512q+m] · x[512(j−q) + τ − m]
                = Σ_{p=0}^{3071} IR[2048+p] · x[512j + τ − p]
```

（置換 `p = 512q + m`。`p ≥ 2952` の項は IR パーティション生成時にゼロパッド → 寄与 0。実効 `p ∈ [0, 2951]`。）

**結論（数学的確定）**:

```
tailOutputBuf[0..512) = (h1 ⋆ x)[512j .. 512j+512)     ただし h1[p] = IR[2048+p]
```

`x` の負インデックス（Reset 直後）は `prevInputBuf`/`inputAccBuf`/FDL のゼロ初期化（cpp:888-904）により寄与 0。

## 5. delayLine 書き込み位置との対応

- `delayLineWrite(l, tailOutputBuf, 512)`（cpp:1630-1631 → 1730-1739）: 位置 `delayWriteCursor mod 2624` から 512 コピー、`delayWriteCursor += 512`。
- **書き込みは直列正順**: 分散パスは `distributing == true` の間、IFFT 完了まで新しい分散を開始せず（cpp:1588 の if 条件）、IFFT 完了（cpp:1619）→ write → `distributing = false`（cpp:1633）の直列化。したがって write はブロック j の昇順に発生し、write event は `delayWriteCursor_before == 512j ∧ delayWriteCursor_after == 512(j+1)` で定義できる。
- **不変条件（1 callback 1 write）**: `numSamples = blockSize = 64 < partSize = 512` であるため、1 つの Add() callback で `inputPos` が partSize に到達するのは最大 1 回 → Forward FFT 最大 1 回 → 分散ループも callback あたり最大 1 回呼ばれ（cpp:1588）、IFFT+`delayLineWrite` は **1 callback あたり最大 1 回**。したがって `W_before == 512j ∧ W_after == 512(j+2)` のような 2 ブロック一括書き込みの callback は構造的に発生しない。（partsPerCallback=2/8 は「1 callback で累積するパーティション数」であり、IFFT 完了回数とは別。）

```
delayLine 論理位置 p に格納される内容 = (h1 ⋆ x)[p]        ← ★ Step 0 の主結論
```

- リング物理位置 = `p mod 2624`。内容整合の成立条件は write/read 間ラグ（`writeLag`）+ 読み出し幅 < 2624（容量式が `outputDelaySamples + partSize + maxBlockSize` を cover する設計: RB-05 修正済み。2624 ちょうど、境界寄りだが定常ラグ ≈ 数百 〜 2048 以下で成立）。

## 6. reference 上の正しい時刻（content_time）の確定

NUC 全体出力 `y[n] = Σ_m IR[m]·x[n−m]` の L1 寄与:

```
y1[n] = Σ_{m≥2048} IR[m]·x[n−m] = Σ_{p≥0} h1[p]·x[n−2048−p] = (h1 ⋆ x)[n − 2048]
```

したがって delayLine 論理位置 `p` の内容 `(h1⋆x)[p]` が出力時刻 `t` に加算されるのが正しい ⟺ `t = p + 2048`。

```
content_time(p) = p + IR_offset_layer      L1: IR_offset = 2048（= l0Len）
                                       L2: IR_offset = 34816（= l0Len + l1Len）
```

**→ Gardner Null Test v2.3 §2.2 の `content_time(j) = 512j + 2048` は実装追跡により確定（証明済み）。**

## 7. L0（即時パス）との対比

`processLayerBlock`（cpp:1336-1427）は L0（isImmediate=true）の同一数学構造（同一 fftTimeBuf 組立 cpp:1347-1349、同一積算パターン cpp:1382-1399、同一 IR 逆順 swap）で、IFFT 後半を `ringWrite` に直行する（cpp:1423）。L0 は `IR_offset = 0` なので `content_time = p` であり遅延補償不要 — ストリーム遅延 0 の確定と整合。

## 8. 仮定と限界（監査記録としての留意点）

1. 本追跡は「ブロック j ≥ 0 の定常動作」を対象とし、Reset 直後の warm-up（FDL ゼロ初期化完了後）を含む。
2. `mirror SoA` の内容一貫性は「直近 Forward FFT での同時書き込み」（cpp:1563-1570）に依存。FDL depth（numParts）≥ numPartsIR は暗黙前提（L1: 8 ≥ 6 ✓。L2 は L2 長から同様に確認する必要がある — IR=40000 で L2 numPartsIR = ceil(5184/4096) = 2、numParts = 2 で depth ちょうど。連続 2 ブロックは衝突なし ✓）。
3. `enableDirectHead = false` を前提（true の場合 `impulseForFft` 先頭 32 タップがゼロ化され direct/L0 の二重加算が回避される設計: cpp:17613-17616）。
4. 数値誤差（FFT 丸め・FMA 順序）は本監査の対象外（M1 で定量）。
5. L2 についても同一構造（差分は partSize=4096、fftSize=8192、numPartsIR、IR_offset=34816 のみ）であることを CMake/コード構造から確認済み。

## 9. 結論と後続 Step への引き渡し

| Step | 状態 | 結果 |
|---|---|---|
| Step 0 | **完了（本書）** | `delayLine 論理位置 p ↔ (hIR_offset_layer⋆x)[p]`、`content_time = p + IR_offset_layer` 確定 |
| Step 1 | 完了（§3-§5 に包含） | delayLine write block j の数学的 content = `(h1⋆x)[512j..512j+512)` |
| Step 2 | 完了（§6 に包含） | `content_time(j) = 512j + 2048`（L1）/ `512j·(l2Part/l1Part 別計算) + 34816`（L2） |
| Step 3 | 未実施 | M2 accessor/test 実装（NUPCTestAccess.h + MT-NUPC-Measurement.cpp v2.4 仕様） |
| Step 4 | 未実施 | T1〜T8 実測 |
| Step 5 | 未実施 | M1/M2/M3 相互検証 |

L2 の content_time 記載訂正: L2 ブロック番号は L2 partSize（4096）単位なので `content_time(j) = 4096·j + 34816`。L1 は `512j + 2048`。
