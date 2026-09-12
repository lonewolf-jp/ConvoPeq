# B13 Repair Design / Stream-Time Mapping Proof（D1〜D4 / Rev 4）

- **日付**: 2026-09-12
- **版**: Rev 4（第15ラウンド監査反映 — F1 unsigned underflow 防止契約・F2 I3 の R(t) 定義・F3 deterministic safety guard 明記・基準版再生成。監査履歴は §10）
- **基準**: `doc/work57/null_test_procedure_v2.md`（v2.9）、`doc/work57/null_test_step3_results_20260912.md`（v1.2、B13 FAILURE CONFIRMED）、`doc/work57/content_mapping_audit.md`（Step 0）、`doc/work69/remaining_bugs.md`（RB-05）
- **判定基準ソース**: `ConvoPeq.md` **Generated 2026-09-12 09:06:50（Rev 4 で再生成 — Step 3 実装込み: friend 宣言 + NUPCTestAccess.h + MT-NUPC-Measurement.cpp 置換。production `delayLineReadAdd()` は依然旧実装 `max(delayReadCursor, maxRead)` — Step 4 未着手）**（cpp: 行番号は抽出相対。実装時は付録 B のマッピングで実ソース行に引き直す）
- **本書のスコープ**: B13 修復の**設計と数学的証明（D1〜D4）**。コード実装は含まない（Step 4 実装は本書の検収後）。
- **Architectural 制約（Practical Stable ISR Bridge Runtime）**: B13 修復を理由に RT 側へ新しい状態判断・動的確保・lock・ownership を導入**しない**。Observer（NUPCTestAccess）は読み取りのみを維持。

---

## 0. 対象の確定事項（実測 — Step 3 報告書 v1.2）

| 項目 | 確定値 |
|---|---|
| oPE（現行実装・定常定数） | T3 −1600 / T4 −1280 / T5 −1280 / T6 −1216 / T7 L1 −1152 / T7 L2 −30720 |
| effectiveDelay（定常定数） | 384 / 704 / 704 / 768 / 832 / 4032 |
| readMode | 初回 MAXREAD → 定常 AUTONOMOUS（forcedSkip = 0） |
| coverage | 1.0（write/read-anchor 欠落なし → 「read 飛び・write 欠落」説明の排除） |
| 現行設定の実効性 | outputDelaySamples（2048 / 34816）は現行 read policy の下で定常 read cursor を拘束していない |

**確定した本質**: delayLine の**論理位置**と**ストリーム時刻**の意味論不整合。content は論理位置 `p` に書かれるが、その content が属すべき出力時刻は `p + IR_offset_layer`。現行 read policy（`max(R, maxRead)`）は自律進行する `R`（writeCursor 追従）をそのまま読むため、content を「論理位置の時刻」に配置してしまう。

---

## 1. 時刻体系の定義（D1 前提 — 原点の固定）

**ストリーム時刻 `t` の原点**: Reset 後の最初の audio callback が処理する入力ストリームの先頭サンプルを `t = 0` とする。callback `c` の時刻区間は `[cB, (c+1)B)`（B = blockSize = 64）、`t_write_callback(c) = t_output_callback(c) = cB`（いずれも Add/Get 引数ブロックの先頭サンプルの絶対インデックス）。

**5 つの時刻の定義**（layer L、partSize `P`、IR_offset `o_L`）:

| 時刻 | 定義 |
|---|---|
| input sample time | 入力ストリーム上の絶対サンプル位置 `n`（原点同上） |
| partition content time | ブロック j の内容 `(h_L ⋆ x)[jP .. jP+P)` が reference 出力 `y` に属すべき時刻区間 `[jP + o_L, jP + o_L + P)`。代表点 `content_time(j) = jP + o_L`（**n0 に依存しない** — Step 0 確定） |
| IFFT completion time | ブロック j の全パーティション累積完了 → IFFT 実行が完了した callback の時刻（`tailOutputBuf` コピー完了点） |
| delay-line write time | `t_write(j)` = `t_ifft(j)`（`delayLineWrite` は IFFT 直後・同 callback 内 — Step 0 §5 直列化・1 callback 1 write）。write event は `W_before == jP ∧ W_after == (j+1)P` で観測 |
| stream output time | 該当 content が `Get()` の dst に加算された callback の時刻 `t_output(j)` |

**`t_write(j)` と `t_output(j)` は同一視しない**（第12ラウンド監査要求）。

### 1.1 現行実装の数式化（乖離の証明）

Step 0（分散スケジュール）+ Step 3 実測より:

```
blocksPerPart bpp = P / B
dist_cbs          = ceil(numPartsIR / partsPerCallback)      // 分散に要する callback 数
t_write(j)        = jP + B × (bpp + dist_cbs − 2)
                  （ブロック j の Forward FFT は蓄積完了 callback C(j/bpp × bpp + bpp − 1) で
                    発生し分散 1 回目を同 callback 内で実行、write は + (dist_cbs − 1) callback 後）
t_output(j)       = t_write(j)        // 実測: write と read-anchor が同一 callback
                    （write 直後の Get で actualReadStart = max(R, maxRead) = R = jP）
```

**現行の outputPlacementError**:

```
oPE = t_output(j) − content_time(j) = B × (bpp + dist_cbs − 2) − o_L
```

これは「**パイプライン lead**（content の計算完了が生論理位置 `jP` より `B×(bpp + dist_cbs − 2)` サンプル早く到達すること）」が、配置に必要な `o_L` と一致しないことを意味する。実測値: L1 lead = 768（T4）/ 832（T6）/ 896（T7）、L2 lead = 4096。`o_L` = 2048 / 34816 に対し先行差分 −1280〜−1600 / −30720。

**乖離の本質（1 行）**: content は `p + o_L` に置かれるべきなのに、現行 read policy は「論理位置 p」を「時刻 p」に対応づけて読む。

---

## 2. D1 — read policy の数学的定義（Policy R）

### 2.1 目標不変式

```
I1（配置正確性）:  ∀j:  t_output(j) = content_time(j) = p_content(j) + o_L
I2（計算先行不変条件）:  lead_L = B × (bpp + dist_cbs − 2) ≤ o_L − B
   （content の計算完了が配置時刻より B 以上早いこと — 未計算 content を読まない十分条件）
I3（capacity）:  cap ≥ o_L − lead_L + 2P
   （Rev 3 訂正: delayLineWrite は **P サンプル**書く（B ではない）— 証明は §3）
```

### 2.2 Policy R — stream-time fixed-offset read（**I4 Get Clock Invariant** 含む）

**I4 — Get Clock Invariant**: `t0 = m_outputSamplesProcessed`（Reset 後 0、uint64 単調増加）。この Get が扱う出力ストリームの先頭時刻は **`t0`** とする。L1/L2 の読み出し位置計算には **`t0` を使用**し、`t0 + got`（この Get の戻り値）は読み出し位置に**使用しない**（off-by-B 防止）。Get 完了後に `m_outputSamplesProcessed += got`（**numSamples ではなく got で累積** — 実ソースは `got = ringRead(...)` を返却する）。`numSamples` と `got` は明確に区別する（本番は Add(B)→Get(B) で `got == numSamples` だが、仕様上は時計は「L1/L2 加算と同じブロック先頭」に固定）。

```
Get() 内（RT・layer L の L1/L2 加算時）:
  t0 = m_outputSamplesProcessed                    // この Get ブロックの先頭時刻（I4）
  if (t0 < o_L):
      return                                       // Phase 0 — 判定は減算前（F1: uint64 では
                                                   //   readStart = t0 − o_L を先に計算すると
                                                   //   0−2048 がラップアラウンドし readStart < 0
                                                   //   が成立しない。t0=0, o_L=2048 で検証済み）
  readStart = t0 − o_L                             // ここでは underflow しない
  if (readStart + numSamples ≤ W(t)):
      delayLine[readStart mod cap] から numSamples 加算
  else:
      // F3 — deterministic safety guard: 証明済み前提条件（I2）の違反検出
      //   → fail-closed no-add → diagnostic counter++（metrics/telemetry のみ）
      //   RT の policy decision ではない（「どの policy を使うか」を決めない）
      no-add + I2 違反カウンタ++
  // Get 完了後: m_outputSamplesProcessed += got（L1/L2 read より前に置かない）
```

**I4 前提（本番プロトコル）**: `got == numSamples` は本番プロトコル（Add(B)→Get(B)、L0 が毎 callback B サンプルを書く）が**保証する前提**であることを明記する（`got < numSamples` の場合、L1 は numSamples 分加算するのに時計が got 分しか進まず次 read で gap が生じる — 現行プロトコルでは発生しないが、実装コメントに前提を残す）。

- `readStart` は**純粋に t0 と o_L の関数**（状態遷移なし、Message Thread 依存なし、確保なし）— Architectural 制約内。
- `delayReadCursor` の自律進行（`max()`）を廃止。cursor は観測用の残存値とする（Getter 経由の coverage 監査は維持可能）。
- **I2 違反カウンタ**は既存診断パターン（atomic counter、Message Thread 観測）で追加 — metrics/telemetry は Observer 原則で許可。

### 2.3 証明 1 — 配置正確性（I1、前提 **I5 Alignment**）

**I5 — Alignment（前提不変条件）**: `P % B == 0` ∧ `o_L % B == 0`。現行値: B=64、P=512/4096、o_L=2048/34816 — すべて成立。

`readStart(cB) = cB − o_L`。I5 の下で `readStart` は B の倍数であり、ブロック j の content `p = jP` を読む callback は `cB − o_L = jP ⟺ cB = jP + o_L = content_time(j)` を満たす。`readStart` は callback ごとに B ずつ増えるため、**各 j について `readStart = jP` となる callback がちょうど 1 回存在する（∀j, ∃! c）**。したがって `t_output(j) = content_time(j)` が全 j で成立。∎（**I5 が破れる構成ではこの一意性が崩れる** — gate で検証、§7.5）

### 2.4 証明 2 — warm-up 欠落なし

`t < o_L` の区間では `readStart < 0` → 加算なし。この区間の L 寄与（reference 側）は `y_L[n] = (h_L⋆x)[n − o_L]` で `n − o_L < 0` の項は `x` の負インデックス = 0（Reset 後ゼロ初期化、Step 0 §2.4）。→ NUC が加算しないことが正しい（欠落なし）。∎

### 2.5 証明 3 — 可否・欠落なし（I2 の使用）

`readStart + B ≤ W` の成立要件: 定常で `W(t) ≥ t − lead_L + P`（直近 write で content 時刻 `t − lead` まで書き込み済み）なので、`t − o_L + B ≤ t − lead + P ⟺ B ≤ o_L − lead + P ⟺ lead ≤ o_L − B`。これは I2 と同値。**I2 が成立する構成では未計算 content の読み出し（欠落）は発生しない**。∎

### 2.6 I2 の全現行構成チェック（シミュレーション検証済み）

| 構成 | P | bpp | dist_cbs | lead | I2（lead ≤ o_L − B） | 修復後 oPE（シミュレーション） | I2 違反 |
|---|---|---|---|---|---|---|---|
| T3 | 512 | 8 | 1 | 448 | 448 ≤ 1984 ✓ | **0**（11 anchor） | 0 |
| T4 | 512 | 8 | 6 | 768 | 768 ≤ 1984 ✓ | **0**（11 anchor） | 0 |
| T6 | 512 | 8 | 7 | 832 | 832 ≤ 1984 ✓ | **0**（11 anchor） | 0 |
| T7 L1 | 512 | 8 | 8 | 896 | 896 ≤ 1984 ✓ | **0**（11 anchor） | 0 |
| T7 L2 | 4096 | 64 | 2 | 4096 | 4096 ≤ 34752 ✓ | **0**（4 anchor、warm-up 544 callback 後） | 0 |

I2 の一般形（`dist_cbs ≤ bpp` の ppc 設計: cpp:991-996 により保証）: `lead ≤ B×(2·bpp − 2) = 2P − 2B`。I2 ⟺ `2P − B ≤ o_L`。全現行構成で成立（T3〜T7 L1: 960 ≤ 2048 ✓、T7 L2: 8128 ≤ 34816 ✓）。**gate への追加を推奨**（§5 検収）。**注意（Rev 2）**: `2P − B ≤ o_L` は**現行分散スケジューラの `dist_cbs ≤ bpp`（cpp:991-996）を前提とする構成 gate** であり、B13 の普遍的な物理法則ではない — scheduler 変更時は本 gate を再導出する。

---

## 3. D2 — capacity re-proof

新 policy の write/read 間ラグ上限:

```
ラグ = W − R ≤ (t − lead + P) − (t − o_L) = o_L − lead + P
読み出し中の上書き防止は **I3 — Capacity Safety（overwrite inequality・Rev 3 訂正）**:
```
I3（overwrite inequality・訂正）:  ∀t:  W(t) + P ≤ R(t) + cap
   （`delayLineWrite` は **P サンプル**（L1: 512、L2: 4096）を書く — Rev 2 の「B サンプル」は誤り。
     未読区間 [R, W) と新規 write 区間 [W, W+P) の非重なりを保証する）
```
必要 cap ≥ (W − R) + P ≤ (o_L − lead + P) + P = **o_L − lead + 2P**
```

**I3 の `R(t)` の定義（Rev 4 — F2）**: I3 の `R(t)` は**論理リードヘッド** `R(t) = t − o_L`（数学上の定義）であり、メンバ `delayReadCursor` とは**別物**として定義する。修復後、`delayReadCursor` は「実際に読み出した位置の telemetry / observation」に格下げ（policy cursor ではない）— I3 と I4 の責務分離。

```
RB-05 形式の現行 cap = o_L + P + B ≥ o_L − lead + 2P  ⟺  o_L + B − 2P ≥ lead。**現行構成では全 case 成立（capacity 変更不要）**:

| 構成 | lead | need = o_L − lead + 2P | 現行 cap | 余裕 | 判定 |
|---|---|---|---|---|---|
| T3 L1（numPartsIR=1） | 448 | 2624 | 2624 | **0** | ✓（境界 — 注記 1） |
| T4 L1 | 768 | 2304 | 2624 | 320 | ✓ |
| T6 L1 | 832 | 2240 | 2624 | 384 | ✓ |
| T7 L1 | 896 | 2176 | 2624 | 448 | ✓ |
| T7 L2 | 4096 | **38912** | **38976** | **64** | ✓（境界寄り — 注記 1） |

- **注記 1**: T3（余裕 0）と T7 L2（余裕 64）は境界寄り。一般式は定常を仮定した保守的上界であり、T3 は write 1 回のみの構成（実効 need ≈ 1024、余裕大）だが、**gate での I3 検証（cap ≥ o_L − lead + 2P の毎回確認）を推奨**（第14ラウンド監査要求）。
- **シミュレーション検証**: Policy R + 訂正 I3 の離散事象シミュレーションで全 case I3 違反 0（上書き 0）を確認。

RB-05 の capacity proof は**式訂正込みで再確認済み**（構造変更なし、数値境界更新）。

---

## 4. D3 — phase 分離（初回 / warm-up / steady state）

| Phase | 区間 | 挙動 | reference との整合 |
|---|---|---|---|
| Phase 0（warm-up） | `t < o_L` | L1/L2 加算なし | L 寄与は `n < o_L` で 0（Step 0 §2.4）→ 整合 |
| Phase 1（steady） | `t ≥ o_L` | 常時加算、oPE = 0 | 配置正確性（証明 1） |
| 遷移点 | `t = o_L` | 最初の content `p = 0` を読む（`t_write(0) = lead ≤ o_L − B`（I2）より書き込み済み） | 不連続なし |

- 現行実装の「初回 MAXREAD → AUTONOMOUS」遷移（policy 支配項の変化）は **Policy R では存在しない**（readStart は単一の閉じた式）。Phase 遷移は「加算開始時刻」のみ。
- **I6 — Phase Coverage**: `t < o_L` では L 寄与なし（reference と一致）、`t ≥ o_L` では **各 callback でちょうど 1 つの論理 content 位置が消費される**。coverage の集計は **Phase 1 のみ**（`coverage(Phase 1) = 1.0`）— Phase 0 の意図的 non-add を欠落として誤認しない（第13ラウンド監査要求）。
- 修復後の M2 測定では Phase 0（silent）を coverage 集計から除外する（read-anchor は Phase 1 で発生）。

---

## 5. D4 — 代表ケース検証計画（Step 4 実装後の検収）

| Case | 構成 | 検収基準（**Primary: oPE == 0** — Rev 2 で effDelay を補助診断に格下げ） |
|---|---|---|
| **T4**（IR=5000） | L1: numPartsIR=6 / ppc=1 / dist_cbs=6 / lead=768 / o_L=2048 | **Primary**: oPE == 0（全 Phase-1 anchor）・I2 違反 = 0・forcedSkip = 0・coverage(Phase 1) = 1.0・**M1 RMS < −90 dB** |
| **T6**（IR=12000） | L1: numPartsIR=20 / ppc=3 / dist_cbs=7 / lead=832 / o_L=2048 | 同上（ppc>1 での確認） |
| **T7 L2**（IR=40000） | L2: numPartsIR=2 / ppc=1 / partSize=4096 / dist_cbs=2 / lead=4096 / o_L=34816 | 同上（大 partSize での確認）+ **L1 の oPE も 0（両レイヤー独立確認）** |

- 検収は **M2 cursor 観測**（CSV 再計算）で行い、M1（gain 反映）と M3 を補助証拠として照合。
- M1 期待値: T4/T6/T7 の RMS < −90 dB（時間ズレ消失後、gain 反映 reference との差は FFT 丸め/加算順序のみ）。
- **Step 4 実装時には H1（`effConst[3]` layer 別化）・H3（T8 EXCLUDED ラベル）を同一サイクルで実施し、H2（cppcheck / clang-tidy）で静的解析ゲートを解消する**。
- **Rev 2 訂正**: effectiveDelay（`t − R_after`）の定常値は Policy R では **`o_L − B`**（L1: 1984、L2: 34752）となる。Rev 1 の「effDelay == 0」は誤り（effectiveDelay と oPE は別の量 — 第13ラウンド精検で指摘）。effDelay は**補助診断**に格下げし、**Primary gate は oPE == 0** のみとする。

---

## 6. 実装指針（Step 4 — 本設計の検収後に着手）

**実装順序（Rev 3 で確定 — 第14ラウンド監査提議を採用）**:

```
Step 4-A   m_outputSamplesProcessed（member / Reset = 0 / Get 後 += got — L1/L2 read より前に置かない）
Step 4-B   delayLineReadAdd 読み出し式変更（maxRead 廃止 / R 自律進行依存廃止 /
           readStart = t0 − o_L / negative → no add / unavailable → skip + diagnostic counter）
Step 4-C   I2 / I5 gate（SetImpulse 後診断ログ）
Step 4-D   H1 effConst[3] layer 別化 + H3 T8 EXCLUDED ラベル
Step 4-E   cppcheck / clang-tidy（STATIC-ANALYSIS PENDING 解消）
Step 4-F   CTest 40/40（Release）
Step 4-G   D4 実測（T4 / T6 / T7 L2）
Step 4-H   M2 → Primary、M1/M3 → Cross-validation の判定
Step 4-I   ConvoPeq.md 再生成
```

**実装後の監査重点（第14ラウンド監査指示）**: `t0` の取得位置、`got` の加算位置、`delayLineReadAdd()` に `numSamples` を渡すことの整合性、I2/I5 gate の実装位置、`maxRead`/`delayReadCursor` の残存依存がないこと。

1. **`MKLNonUniformConvolver` に stream 時刻カウンタ導入**: `m_outputSamplesProcessed`（uint64、Reset で 0、Get の got 分ずつ加算）。RT 内の単調カウンタのみ（状態判断の新設なし、確保なし、lock なし — Architectural 制約内）。
2. **`delayLineReadAdd` の読み出し位置式を変更**: `actualReadStart = max(R, maxRead)` → `readStart = tStream − outputDelaySamples`（`maxRead`/`max` 廃止）。負値区間は加算スキップ。I2 違反時は加算スキップ + カウンタ（防御）。**既存 obj 世代・依存は Ninja が追跡**（ビルドゲートの clean リカバリは dirty ツリー運用に従う）。
3. **`outputDelaySamples` の値は変更しない**: 現行値（`prevLayerTotalSamples` = 2048 / 34816）は「論理位置 → ストリーム時刻のオフセット」として**正しい**（D1 証明 1）。壊れているのは読み出し policy のみ — work69「実測値から決定」規定に対しては、本設計の lead 実測（L1=768 / L2=4096）と I2 不変条件の確認をもって履行とする。
4. **H1/H3 を同サイクルで実施**（`effConst[3]` layer 別化、T8 EXCLUDED ラベル）→ H2（cppcheck / clang-tidy）→ Step 4 実測（D4）。
5. **observer（NUPCTestAccess）は無変更**（読み取りのみ。`effConst` のメンバ名変更には accessor 追従）。

---

## 7. 検収基準（本設計）

| # | 項目 | 基準 |
|---|---|---|
| R1 | D1 証明 | 配置正確性（証明 1）・warm-up 欠落なし（証明 2）・可否（証明 3）が数学的に閉じていること（本書 §2.3-2.5） |
| R2 | D2 | capacity 変更不要の証明（本書 §3 の数値表） |
| R3 | D3 | phase 分離の定義と遷移不連続なし（本書 §4） |
| R4 | I2 一般形 | `2P − B ≤ o_L` を gate（cpp 起動時診断または SetImpulse 後診断ログ）に追加すること |
| R5 | Step 4 実装 | Policy R 実装 + H1/H3 + cppcheck/clang-tidy + CTest 40/40 |
| R6 | D4 実測 | **Primary: oPE == 0（全 Phase-1 anchor）**。Safety: I2 違反 0・forcedSkip 0・coverage(Phase 1) 1.0。Cross-validation: M1 RMS < −90 dB、effDelay 定常 = `o_L − B`（L1 1984 / L2 34752 — 補助診断）、M3 L1 一致 |

---

## 8. 実装契約 — 不変条件 I1〜I6（Step 4 実装との一対一対応・第13ラウンド監査提議を採用）

| ID | 不変条件 | Step 4 実装上の対応 |
|---|---|---|
| **I1** Placement | `t_output(j) = jP + o_L`（全 j） | `delayLineReadAdd` の readStart 式（`t0 − o_L`） |
| **I2** Availability | `lead = B(bpp + dist_cbs − 2) ≤ o_L − B`（**現行 scheduler の `dist_cbs ≤ bpp` 前提の構成 gate** — 普遍法則ではない） | SetImpulse 後の診断ログ + gate（scheduler 変更時は再導出）。違反時の no-add は **deterministic safety guard**（fail-closed + diagnostic — F3）であり RT policy decision ではない |
| **I3** Capacity | `cap ≥ o_L − lead + 2P`（overwrite inequality `W(t) + P ≤ R(t) + cap` — write は **P サンプル**、Rev 3 訂正。**R(t) = t − o_L は論理リードヘッド、`delayReadCursor` は observation（F2）**） | 現行 RB-05 式で成立（**変更なし**。T3 余裕 0・T7 L2 余裕 64 — gate で毎回確認推奨） |
| **I4** Get Clock | `t0 = m_outputSamplesProcessed`、読み出し位置計算に `t0` を使用、Get 後 `+= got`（numSamples と got を区別。**got == numSamples は本番プロトコル（Add(B)→Get(B)）が保証する前提**）。**F1（Rev 4）: Phase 0 判定は減算前 — `if (t0 < o_L) return;` の後に `readStart = t0 − o_L`（uint64 では `readStart < 0` は成立しない）** | Get 内カウンタ導入（Reset で 0） |
| **I5** Alignment | `P % B == 0 ∧ o_L % B == 0`（証明 1 の前提） | SetImpulse 後の診断ログ（gate） |
| **I6** Phase Coverage | `t < o_L`: no L contribution（reference と一致）／`t ≥ o_L`: exactly one logical content position consumed — **current Step 4 contract（numSamples == B）における限定**（将来 numSamples > B を許す場合は 1 callback が複数 P 境界を跨ぐ可能性があるため invariant を分離） | M2 coverage 集計は **Phase 1 のみ** |

- **I2 violation counter は policy decision state にしない**: 安全側 skip + diagnostic counter のみ（Observer/metrics の範囲）。
- **Architectural**: 修復は Publish/Retire/RCU 系 Authority に触れない（`MKLNonUniformConvolver` 内の B13 read policy に局所化）。

---

## 9. リスクと代替案

| リスク | 対応 |
|---|---|
| I2 違反（将来の構成変更で lead > o_L − B） | gate に I2 不変条件（`2P − B ≤ o_L`）を追加し、違反構成は SetImpulse で拒否 or 診断ログ |
| 分散完了遅延の変動（ partsPerCallback 変更） | lead 式は構成から決定論的に算出 → gate で毎回検証可能 |
| `m_outputSamplesProcessed` と L0 ringRead の同期ずれ | got 分のみ加算（L0 が提供したサンプル数と同一）— T1/T2 の −311 dB が同期の健全性を既に実証 |
|将来の numSamples ≠ blockSize | readStart は t ベースで計算されるため式は不変（blockSize 依存は lead 式のみ） |
| 代替案（maxRead 固定 / スナップ） | 不採用（再読み出し / residual +256 — Step 3 報告書 §8 で実測・シミュレーション済み） |

---

## 10. 監査履歴（本設計書）

| ラウンド | 判定 | 主要指摘 | 対応 |
|---|---|---|---|
| 第13ラウンド（監査 + 精密検証の 2 文書） | **D1〜D4 CONDITIONALLY APPROVE**（設計審査通過 — Step 4 実装可） | (1) **I4 Get Clock Invariant** の明文化（`t0` 使用・Get 後 `+= got`・numSamples と got の区別・off-by-B 防止） (2) **I5 Alignment**（`P%B==0 ∧ o_L%B==0`）を証明 1 の前提に（∀j, ∃! c） (3) I2 `2P−B≤o_L` は**現行 scheduler 前提の構成 gate** と明記 (4) D2 に overwrite inequality（I3）を明記 (5) coverage は **Phase 1 のみ** (6) R6 primary gate を **oPE==0** に（**effDelay==0 は誤り — Policy R では `o_L−B`**） | **Rev 2 で全件反映** |
| 第15ラウンド（監査 + 精密検証の 2 文書） | 監査: **CONDITIONAL APPROVE**（F1 必須 + F2/F3 推奨 + 基準版確認）／精検: **APPROVED / Step 4 GO** | (1) **F1 — unsigned underflow 防止**: `m_outputSamplesProcessed` が uint64 のため `readStart = t0 − o_L; if (readStart < 0)` は**永遠に成立しない**（t0=0, o_L=2048 でラップアラウンド検証済み）→ **Phase 0 判定を減算前**（`if (t0 < o_L) return;` の後に減算）に契約固定 (2) **F2 — I3 の `R(t)` を論理リードヘッド `R(t) = t − o_L` と定義**し、`delayReadCursor`（observation/telemetry）と分離 (3) **F3 — I2 violation の no-add は deterministic safety guard**（proven precondition violation → fail-closed → diagnostic）であり RT policy decision ではないと明記 (4) **基準版**: 監査側アップロード（11:20:30）とディスク版（20:16:16）が相違 → **ConvoPeq.md を現ソースから再生成（2026-09-12 09:06:50、Step 3 実装込み）で基準固定** (5) production は旧実装のまま（設計承認・Step 4 実装前として扱う） | **Rev 4 で全件反映（本版）** |

- **独立検証（第14ラウンド）**: Policy R + I4 時計込みの再シミュレーションで全 case oPE={0}・I2 違反 0・silent = `o_L/B`（L1 32 / L2 544 callback）・初回 read は `t = o_L`（`p=0`、`t_write(0)=lead ≤ o_L−B` で書き込み済み）を確認。**effectiveDelay は Policy R で `o_L − B`（L1 1984 / L2 34752）** となるため、Rev 1 の「effDelay == 0」は成立しない — Rev 2 で補助診断に格下げ（維持）。
- **正確な現在環境の解釈**: 現行 `outputDelaySamples = 2048 / 34816` は「無意味」ではなく、**現行 read policy の下で定常 read cursor の固定ストリーム遅延パラメータとして機能していない**（Policy R では `o_L` が正しいオフセット）。
- **バージョン**: Rev 1（2026-09-12 作成）→ Rev 2（第13ラウンド監査反映）→ Rev 3（第14ラウンド監査反映）→ Rev 4（第15ラウンド監査反映・本版）。

## 11. 正式ステータス（第15ラウンド監査で確定）

```
B13 Repair Design (D1〜D4) Rev 4: 設計審査 APPROVED — Step 4 実装 GO
  — Policy R（readStart = t0 − o_L、I4 時計込み）は全 case で oPE = 0（数学 + シミュレーション）
  — I3 訂正済み（W + P ≤ R + cap、cap ≥ o_L − lead + 2P）→ capacity 変更不要
  — F1: Phase 0 判定は減算前（uint64 underflow 防止契約）
  — F2: I3 の R(t) = t − o_L（論理リードヘッド）と delayReadCursor（observation）を分離
  — F3: I2 violation の no-add は deterministic safety guard（fail-closed + diagnostic）
  — outputDelaySamples 値変更不要 / I6 は numSamples == B contract の限定
  — 基準版: ConvoPeq.md 2026-09-12 09:06:50（Step 3 実装込み・Step 4 未着手）で固定

Step 4 実装順序（固定）: 4-A → 4-B → 4-C → 4-D → 4-E → 4-F → 4-G → 4-H → 4-I
検収: D4（T4/T6/T7 L2・Primary oPE==0）+ CTest 40/40 + cppcheck/clang-tidy
Architectural 制約: Authority 構造変更なし・RT に状態判断/確保/lock/ownership の新設なし
```

## 12. 附属資料

| 資料 | 位置 |
|---|---|
| Policy R シミュレーション（本書 §2.6 表の出所） | 本検証の離散事象シミュレーション（T3/T4/T6/T7 L1/T7 L2・oPE 全 0・I2 違反 0・warm-up silent 確認） |
| 現行実測（乖離の証拠） | `.auto/nulltest/nupc_v29_csv/M2_*.csv`（21 run）・`.auto/nulltest/smoke_run.log` |
| 手順書・報告書 | `doc/work57/null_test_procedure_v2.md`（v2.9）・`doc/work57/null_test_step3_results_20260912.md`（v1.2） |
| Step 0 監査記録 | `doc/work57/content_mapping_audit.md` |
