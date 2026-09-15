# SR-02 Implementation Gate / Contract Freeze (Pre-Audit)

- **日付**: 2026-09-14
- **基準**: commit `9ff83609`（SR-01(B) CLOSED 後）の作業ツリー = 最新 `ConvoPeq.md`（FRESH 検証済）相当
- **Type**: read-only Pre-Audit / Contract Freeze（production code 変更 0）
- **監査根拠**: doc/audit/ConvoPeq_SampleRate_Support_Audit_2026-09-13.md §2・ConvoPeq_SampleRate_Reverification_2026-09-13.md §3・ConvoPeq_BugList_and_FixPlan_2026-09-13.md §2.3/§5 Step 5・doc/work57/*（null_test_procedure_v2 / b13_repair_design / step4 結果）
- **凍結案**: BugList 推奨 **Option A + C**（秒ベース l0MaxLen + UI 可視化）。B（kL0MaxParts 定数拡大）は SR 依存式を定数に埋め込む変形であり不採用

---

## 1. 現行実装（確定・コード 1:1）

```738:745:src/MKLNonUniformConvolver.cpp
const int l0Part = nextPowerOfTwo(max(blockSize, 64));
const int l0MaxLen = kL0MaxParts * l0Part;          // kL0MaxParts = 32（h:452 固定）
const int l0LenByTailStart = llround(tailStartSec × sampleRateForTail);
const int l0LenTarget = jlimit(l0Part, l0MaxLen, l0LenByTailStart);
const int l0Len = min(irLen, tailEnabled ? l0LenTarget : l0MaxLen);
```

- `numPartsIR = ceil(cfg.len / partSize)`（cpp:784）→ L0 の numPartsIR = パーティション数そのもの
- tailMode clamp（cpp:649-670）: AirAbsorption は `tailStartSec ≥ 0.055`、LayerTailContouring は `≥ 0.12`、Bypass は tailEnabled=false → l0Len=l0MaxLen 固定
- SetImpulse は **NonRT（Message/Loader thread）専用**（h:34 契約）。Add()/Get() は RT で確保ゼロ（h:11 契約、FDL/ring/accum は SetImpulse 時確定済み）

## 2. SR-02 の再証明（Reverification §3 と実測一致）

`l0MaxLen = 32 × l0Part` は**秒換算で SR に反比例**。tailStart の意図（mode 後 clamp 値）に対する実効 L0 カバー率:

| 処理レート | bs | l0Part | l0MaxLen | 意図 ts（mode1 clamp 後 0.12s） | 実効 L0 | カバー |
|---|---|---|---|---|---|---|
| 48k | 256 | 256 | 8,192 | 5,760 | 5,760 | 100% |
| 96k | 512 | 512 | 16,384 | 11,520 | 11,520 | 100% |
| 192k | 256 | 256 | 8,192 | 23,040 | 8,192 | **35.6%** |
| 384k | 512 | 512 | 16,384 | 46,080 | 16,384 | **35.6%** |
| 705.6k | 512 | 512 | 16,384 | 84,672 | 16,384 | **19.4%** |
| **768k** | **512** | 512 | 16,384 | **92,160** | 16,384 | **17.8%** |

**High 確認**（Reverification: 「192 kHz / bs=256 から既に顕在」）。監査表の 0.085s 基準に対し、production の mode-1 実 clamp は 0.12s のため**実欠損は監査数値より深い**。早期反射が L1（分散 ppc・大パーティション）へ退避し、L0 の時間分解意図（低遅延高解像度層）と AirAbsorption/Contour の層ゲイン設計前提が崩れる。

### 2.1 連動機構（更新必須範囲の確定）

- **MT-NUPC-Measurement**（ctest #39・work57）は同一理論モデルを模倣（:71-93 `computeLayerCfg`、`kL0MaxParts=32` 直書き:51）。48k/bs64 で `l0MaxLen=2048` 全 case 固定、期待値 oPE=0 / **effL1=1984 (=o_L−B)** / effL2=34752 はこの幾何の関数。**上限式を変えればケース表は失効**（例: ts=0.12→required 90 parts→l0MaxLen=5760 で T3(2049)/T4(5000) の L1 生成境界そのものが移動）。→ BugList §5「SR-02 → work57 テスト更新（同時に）」を**必須共変**として凍結
- **B13 構造 gate**（cpp:1103-1135、li=1..）: I2/I5/I3 — l0Len↑ は l1Len↓（numPartsIR(L1)↓ → distCbs↓ → lead↓）と prevLayerTotalSamples↑（delayLineCapacity:1007↑）へ作用し、**I3 の need は下がり cap は上がる**（余裕改善方向）。o_L（layer 境界由来）は MT-NUPC 同様に再計算を要する

### 2.2 RT 影響モデル（processLayerBlock cpp:1358-1461）

L0 は 1 callback あたり **forward FFT ×1 + inverse FFT ×1 + numPartsIR 回の SoA 複素乗算蓄積**（×complexSize、AVX2、prefetch 済）。確保・lock・分岐追加なし。増加分見積（上限 256 parts）:

| 構成 | parts | hop | L0 partition ops / hop | 概算 |
|---|---|---|---|---|
| 48k/bs64（ts0.12→90 parts、FFT128） | 32→90 | 1.33ms | 90×65×2ch ≈ 11.7K bin-mac | ~2-3µs（無視可能） |
| 768k/bs512（→180 parts、FFT1024） | 32→180 | 0.67ms | 180×513×2ch ≈ 185K bin-mac ≈ 0.74 MFLOP | ~10-15µs ≪ hop budget |
| 768k/bs64（extreme hop 83µs、required 1020→cap 256） | 32→256 | 83µs | 256×65×2ch ≈ 33K bin-mac | ~5-8µs / 83µs（≤10%） |

→ **上限付き拡大は RT 予算内に収まる**（extreme hop 構成でも ≤10% 水準）。ただしこの見込みは本 freeze の計測前提であり、実装時に work57 相当の wall-clock 実測（MT-NUPC 実行時間回帰）で確認する義務を契約化する（§5 T-SR02-6）。

### 2.3 メモリ影響（容量 proof）

L0 irFreq SoA = parts × complexSize × 8B × 2：768k/bs512 で 262KB→**1.48MB**/ch、2ch 計 +2.4MB。L1 delayLine +0.6MB。`MAX_IR_LATENCY(2^21)`/`DELAY_BUFFER_SIZE(2^22)`/**IR 長上限**は一切不変 — SR-01(B) の capacity invariant と直交・非抵触。

---

## 3. 契約凍結 — 上限式（Option A + C）

```
kL0PartsLegacy := 32                      // 下限（フロア）: 現行幾何以上を常に保証（縮小回帰の排除）
kL0PartsHard   := 256                     // RT ceiling（§2.2 実測根拠付き）
coverageSec    := SetImpulse 実効 tailStartSec（mode clamp 後: air≥0.055 / contour≥0.12、上限 0.80）
required       := ceil(coverageSec × sr / l0Part)
l0MaxParts     := tailEnabled ? clamp(required, kL0PartsLegacy, kL0PartsHard)
                             :  kL0PartsLegacy          // Bypass は tailStart 非依存（現行不変）
l0MaxLen       := l0MaxParts × l0Part      // partSize 倍数へ切り上げ（pow2 不要・alignment は partSize pow2 で充足）
l0LenTarget    := jlimit(l0Part, l0MaxLen, l0LenByTailStart)   // 式不変（天井のみ変化）
```

### 境界条件（凍結）

| 条件 | 契約 |
|---|---|
| 低SR回帰 | **48k/bs256、96k/bs512 等、required ≤ 32 の構成は l0MaxParts == 32 で現行幾何（ビット一致）を維持する。48k/bs64 は MT-NUPC の意図的な例外であり、required = 90 のため l0MaxParts = 90。work57 の期待値表を同一サイクルで更新する。** |
| 高 SR | 192k/bs256→90、384k/bs512→90、768k/bs512→180 parts（ts=0.12）、768k/bs256→360→cap 256（カバー 85.3ms）| 
| tailStart max 0.80 | 48k/bs64 で required=600 → cap 256 → 実効 0.341s。cap 到達は NonRT ログ＋UI 表示で可視化（C-2/C-3 相当） |
| tailEnabled=false | 現行と同一（32×l0Part） |
| directHead ON | 幾何の l0MaxLen 定義に変更不要（direct taps は別領域） |
| channel | 不変（2ch） |
| 状態互換 | tailStartSec/pendingOverride の永続窓不変。層幾何は状態非永続（load 再構築で確定論）— 永続互換リスクなし |
| 変更禁止 | MAX_IR_LATENCY / DELAY_BUFFER_SIZE / IR 長上限・clamp 系（SR-01(B) 契約）/ process() 本文 / RT 確保・delete・lock・判断 / Publish・Crossfade・Retire・Epoch / Coordinator / Host PDC / H-01 dryAlign / H-02 peak / SR-03 gate / 新規 CTest target |

### 凍結スコープ（実装で許容する変更箇所）

- **S-1** `MKLNonUniformConvolver::SetImpulse`（cpp:742-745）— 上記式（l0MaxParts 算出）。ヘッダに `kL0PartsLegacy=32`（既存 kL0MaxParts の別名/置換）と `kL0PartsHard=256` 定数
- **S-2** MT-NUPC-Measurement — モデル追従（`computeLayerCfg` 同一式化）＋ **case 表の再設計**（新アンカー l0MaxLen=5760 @48k/bs64 基準で T2'/T3'… 再配置、期待値は理論モデル単一系統で再生成 → 実測照合。手編集禁止）
- **S-3** UI 可視化（C）— tailStart 近傍表示または IR info: 実効 L0 coverage `min(coverageSec, 256×l0Part/sr)` と cap 到達通知（message thread、新規 public API 追加なし・既存 accessor 範囲で実装）
- **S-4** cap 発動/拡大発動の NonRT ログ（`[SR-02] L0 parts 32→N (coverage X ms)`）

## 4. RT/ISR impact audit

- 変更不要箇所: `process()`・RT 側ループコード（processLayerBlock は numPartsIR 読取のみで変更なし）
- SetImpulse 内純粋な定数計算 → NonRT。numPartsIR 増分は RT ループ反復回数へ反映されるが確保ゼロ・lock ゼロ・分岐形状不変（§2.2）
- B13 gate（I2/I5/I3）は li=1.. の遅延層のみ — 拡大後の全構成で通過をテスト契約化（§5 T-SR02-4）
- 層 ownership・publish 経路・retire ルート・FDL/Plan 生命周期は無変更（work95 と同一原則）

## 5. テスト契約（実装より先に凍結）

| ID | 内容 |
|---|---|
| T-SR02-1 | 式テーブル: (sr, bs, tailMode, tailStart) グリッドで l0MaxParts == clamp(ceil(ts×sr/l0Part), 32, 256)・tail disabled→32・低 SR（48k/bs256, 96k/bs512）は 32 据置（現行幾何一致） |
| T-SR02-2 | 生産ジオメトリ: 768k/bs512 contour デフォルトで L0 = 92,160 samples（180 parts）確定（SetImpulse 後 numPartsIR(L0)==180、directHead 両状態）|
| T-SR02-3 | tailEnabled=false（Bypass）で l0MaxLen=32×l0Part 完全一致（旧挙動回帰）|
| T-SR02-4 | B13 構造 invariant グリッド: 高 SR×bs×irLen 全構成で I2/I5/I3 OK かつ gate ログ [NG] 0（MT-NUPC 内 or harness 既存経路で回帰確認）|
| T-SR02-5 | MT-NUPC case 表再生成後の structuralFailures == 0（既存 ctest #39 を維持・target 追加なし）|
| T-SR02-6 | RT コスト回帰: MT-NUPC 実行 wall-clock を実施前基準（SR-01(B) 後記録）比 **+50% 以内**に収める（§2.2 見込みの測定検証。超過時は STOP 再ゲート）|
| T-SR02-7 | UI 可視化（S-3）の計算式一致: effective coverage = min(ts, 256×l0Part/sr)（message thread 純関数として検証可能形に）|
| T-SR02-8 | cap 発動ログ（S-4）: tailStart=0.80@48k/bs64 で parts==256・ログ発火・カバレッジ 0.341s 申告 |

## 6. 判定

```text
SR-02 Pre-Audit              PASS（§1-§2 実コード再証明・Reverification/Support と数値一致）
RT cost model                PASS（上限付きで予算内・ただし T-SR02-6 実測を条件付き義務化）
Capacity proof               PASS（+~3MB/ch クラス・IR 長上限不変）
Contract Freeze              PASS（§3 式・境界・スコープ S-1..S-4）
Test contract                FROZEN（T-SR02-1..8、work57 表再生成は S-2 の必須共変）
SR-02 Implementation Gate    GO
```

**条件**:
1. work57/MT-NUPC 期待値表の更新を**同一サイクル**で行わない変更は却下（順序制約: 式変更と表更新を分離して gate を通さない）
2. T-SR02-6 wall-clock 回帰が +50% 超なら即 STOP（kL0PartsHardCap 再設計へ戻る）
3. 監査原文（Support §2.4 / BugList §2.3）の「2 冪切り上げ」案は凍結式（partSize 倍数切り上げ）で差し替え。pow2 化は l0MaxLen をさらに最大 2× 膨張させ RT 見積を悪化させるため不採用（根拠 §2.2/§2.3）

---

## 7. 実装記録（同一サイクル内で発生した共変・是正 — 承認済みスコープ内）

| # | 項目 | 内容 | 判定 |
|---|------|------|------|
| R-1 | S-4 ログの `%s` 問題 | `juce::String::formatted` の `%s` はワイド解釈（B13-GATE 同注記）→ "hard cap" サフィックスは連結構築へ是正 | 実測 OK |
| R-2 | Phase1 coverage expected 境界 | o_L=5760（128 mod 512）で `contentTime == totalStream` の厳密尾が顕在化。旧幾何（o_L が 512 倍数）では空差。expected/missing 双方の境界を `< totalStream`（run 内到達の定義に忠実）へ共変。B13 手順書 §P0-1 の「run 時間内のみ」注記と整合 | T5/T7 8 件 → 0 |
| R-3 | MT-NUPC 追加ケース | B1 (irLen=5761 = 新 l0MaxLen+1 境界)。T3/T4 は SR-02 後単層（L0 のみ）となり L1 生成回归は B1/T5/T6/T7/R2bL2 が担保 | structuralFailures=0 |
| R-4 | T-SR02-7（UI 式一致） | UI は production `kL0PartsHardCap`（NUC 公開定数）を直接参照。式 `min(ts, cap×l0Part/sr)` の数値権威は M4 式表（production `computeL0MaxParts`）と共用。UI 側に新規 public API なし | inspection + M4 |
| R-5 | **I5 アライン是正（契約 §3 式に反映）** | B13 grid に air mode を追加した際、`l0LenByTailStart = llround(ts×sr)`（air 48k で 4080）が partSize 非整除 → o_L 非整除 → **I5 NG**（3 件）。landmine は SR-02 前から潜在（旧幾何は target 常時 l0MaxLen=pow2 由来で非顕在）。是正 = カバレッジを **l0Part の倍数へ切り上げ**（`l0CoverageAligned`）— pow2 天井は膨張（768k で 180→256 parts）を招くため不採用とし、step 2 指示「partSize 倍数で十分」を厳守。b(pow2 ホップ) \| l0Part より I5 が構造的に成立 | air grid 含む [NG]=0 |

**実測サマリ（Release icx / 3 回）**: wall-clock {1.175, 0.876, 0.930}s、中央値 0.930s 対 baseline 1.048s → **Δ=−11%**（T-SR02-6 PASS）。M2 フェーズ単体 0.246s。M4 式表 9 行・geometry 3 構成・B13 grid 6 構成すべて PASS、[SR-02] ログ expand/hardCap とも発火確認。
