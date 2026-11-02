# STG-11-D15 Fresh Discovery — DSP Execution Boundary & Buffer-Safety Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D15_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D15** — DSP 実行境界・バッファ安全性監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用。
  D12〜D14 の結論の流用なし）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

正常 runtime の Audio Thread 実行における buffer／size／pointer／SIMD／FFT／
resample／delay／NaN 生成／failure fallback を実コードで監査した結果、
D15 GO 条件 7 項目をすべて満たす concrete defect は存在しない。
Implementation / Gate は作成しない。

D1〜D11 の未 commit 変更・D12〜D15 監査成果物はすべて保持する。

---

## 1. Buffer / sample-count boundary map（§2）

- Entry guards（`AudioBlock.cpp:152-164`）：`numSamples<=0‖>1M`・
  `startSample` 範囲・world nullptr のいずれも `clearActiveBufferRegion` で復帰。
  D15-1（N=0）安全。D15-2（host block ≠ 内部）は entry で clamp。
  D15-3（channel 数）は `std::min(2, ...)`＋nullptr ガード付き pointer 取得。
  D15-4（stereo/mono）は chCount 分岐＋各 ch の存在確認。安全。

## 2. SIMD / alignment map（§4）＋ loop remainder（§5）

| site | pattern | verdict |
| --- | --- | --- |
| TruePeakDetector peak／dot | `alignas(32) tmp`＋loadu(x)／load(coeffs 64B aligned)＋`i<=n-16/+12`＋`i<=n-4`＋scalar 残余 | SAFE |
| CustomInputOversampler dot | 同上（コメントに alignment 契約明記） | SAFE |
| OutputFilter pack/unpack | `alignas(16) tmp[2]`＋loadu/load（state は struct 配置） | SAFE |
| EQ sumSq | `alignas(32) temp`＋`vEnd=n/4*4`＋scalar 残余 | SAFE |
| MKL direct | 64B alloc＋k step 8（k+4 範囲内）＋scalar 残余 | SAFE |
| MKL accumBuf denormal | `partStride=(cplx*2+7)&~7`（8 の倍数）＋64B alloc | SAFE |
| MKL addFallback | 実行時 alignment 判定（load／loadu 切替）＋残余 | SAFE |
| Lattice clampStateSIMD | `loadu/storeu`＋残差 `std::clamp(state[8])` | SAFE |
| Dither/processBand | loadu のみ／scalar | SAFE |

`loadu` 安全宣言だけを根拠にせず、aligned 側の allocation 起源・
offset・remainder を個別確認した。`partStride` の 8-倍数化により
末尾超過なし。N=0/1/width±1 の remainder は全 site で scalar 処理あり。
unsigned underflow（`N-1`）パターンなし。

## 3. FFT / partition map（§6）＋ delay / ring map（§8）

- Layer geometry：`partStride` 8-倍数、`delayLineCapacity` に maxBlock 項を含むため
  `numSamples<=maxBlock<capacity` が成立。`delayLineReadAdd` は nullptr／capacity／
  uint64 wrap（t0<delay 先行判定）／I2 availability／二分割 read
  （`first=min(n,cap-ro)`＋残差）で防御。`capacity==0` は alloc 失敗で layer 構築失敗
  （`return false`）のため到達不能。
- `l0LenByTailStart` の INT_MIN 経路は D9 で session 側を遮断済み。
  有限値では `jlimit(l0Part, l0MaxLen, ...)` で有界。
- Resample（`resampleIR`）：SR 正値・`abs diff>1e-6`・ch/len 正値・peak・
  `maxOut>0`・SilentIR／Cancelled／Error の Result 列挙。呼び出し側は
  LoaderThread step で処理。output 不足・end 超過・未初期化の証拠なし。

## 4. NaN / Inf generation map（§9 — valid state からの新規生成）

| 演算 | 検証 |
| --- | --- |
| `decibelsToGain` | 入力は clamp 済み有限値（D8/D11 確定）のみ |
| `pow/sqrt/exp`（SVF/biquad） | `isfinite` ガード→bypass 係数（全 type） |
| `1/q`（q=0） | q は `[DSP_MIN_Q>0, ...]` に clamp |
| `1/(1+g(g+k))` | `abs<1e-15` ガード→bypass |
| dither `pow(2,N-1)` | N は validator 通過分のみ（publish）。placeholder は bypass |
| `nextPowerOfTwo` | `max(x,64/1)` 付き呼び出しのみ |
| 除算・剰余一般 | ゼロ除算ガードまたは正値保証を確認 |
| RT `processBand` | per-sample 有限 clamp＋state reset（最終防衛） |

valid state からの NaN/Inf 新規生成経路なし。

## 5. DSP failure / fallback map（§10）

- `validateBuffer`（有限＋energy）が mixed/min-phase 出力を関門化。不合格は前段維持。
- SVF/biquad の failure は bypass 係数（無音化ではなく unity 維持）。
- Loader の decode/alloc/read/cancel 失敗は error＋部分 object retire（D13 で確定）。
- `isLoading/isRebuilding` flag は例外時も reset。
- failure の success 扱い・前回 buffer 誤再利用・flag 漏れの証拠なし。

## 6. なぜ NO-GO か

D15 GO 条件 7 項目に照らし、「既存 guard で完全に中和されていない」
「具体的なユーザー影響がある」を満たす候補が存在しない。
最も精査した SIMD alignment・ring read・NaN 生成のいずれも、
allocation 起源・境界式・中和ゲートが実コードで確定している。

## 7. 推奨

- D6 NO-GO・R8 flake・R4（P3）・dead code 一括整理は引き続き未解決として残る。

## 8. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12〜D15 監査成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
