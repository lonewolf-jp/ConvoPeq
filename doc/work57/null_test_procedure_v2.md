# Gardner Null Test 実施手順書 v2.9（現行ソース準拠版）

- **日付**: 2026-09-11（v2.9）
- **基準ソース**: `ConvoPeq.md` Generated 2026-09-11 20:16:16（cpp: 行番号は ConvoPeq.md 抽出セクション**相対**。実ソース対応は**付録 B**、offset は箇所により −1〜−3 で一様ではない。Step 3 の実装・コミットでは実ソース行を用いる）
- **先行文書**: `doc/work57/null_test_procedure.md`（v1）、`doc/work57/content_mapping_audit.md`（Step 0）、`doc/work69/bug_final_report.md`、`doc/work69/remaining_bugs.md`
- **改訂履歴**:
  - v2〜v2.5: （再設計・差分式廃止・M2 主測定化・三時刻モデル・イベント定義・Step 0・layerGain 反映・capacity 2624・修復差し替え）
  - v2.6（2026-09-11）: content_time から n0 除去、T7 L2 期待値 −30720、判定帯 64 量子化整合、P2=704、Step 3 実装ゲート。
  - **v2.7（2026-09-11）**: 第8ラウンドレビュー（2 文書: 監査 + 詳細検証）反映。**全指摘をシミュレーション再検証で確認（全件一致）**:
    - (1) **検収基準の §6 統一（必須）**: §8-2 の「修復後 `|error| ≤ 64` 検収許容」を削除。**修復後の合格条件も §6 と同一（`error == 0`）**。「callback 粒度 ±64」は診断上の分類のみで合格条件ではない。
    - (2) **t_write 一般式を修正**: v2.6 の `p + 448 + (numPartsIR−1)×64` は ppc=1 限定。一般式 `t_write(j) = p_content(j) + 64 × (blocksPerPart + dist_cbs − 2)`、`blocksPerPart = partSize/blockSize`、`dist_cbs = ceil(numPartsIR/ppc)`。
    - (3) **case 別期待値表に差し替え**: oPE は ppc に依存するため case 別（T3=−1600 / T4=−1280 / T5=−1280 / T6=−1216 / T7 L1=−1152 / T7 L2=−30720）。effectiveDelay も case 別（384/704/704/768/832/4032）。
    - (4) **readMode を修正**: 「常に AUTONOMOUS」は誤り。**初回 read は MAXREAD**（`maxRead=0 ∧ R=0 → actual==maxRead`）、以降 AUTONOMOUS。P1 を「定常では AUTONOMOUS」に修正。初回 MAXREAD を異常と誤判定しない。
    - (5) **M2 は impulse 位置ごとに独立 run**: 同一ストリームに複数インパルスを同時配置すると delayLine 内容が重ね合わせになり write event の n0 帰属が不可能。M2 の 1 run は 1 インパルス。M3 は複数インパルスの集計を許容。
    - (6) 付録 B 修正（outputDelaySamples 1009→実 1006 等、欠落行追加）、T5 目的変更、§6 に「1..63 = M2 再構成式バグ」明記。
  - **v2.8（2026-09-11）**: 第9ラウンドレビュー（両文書とも **IMPLEMENTATION-READY / Step 3 実装開始を承認**）反映。軽微 2 点のみ:
    - (1) **`writeLag` の記述訂正**: 「定常定数」ではなく**定常鋸歯**（シミュレーション確認: L1 は write 直後 448 → 64 ずつ減少 → 0 → 次 write で 448 反復。L2 は 0..4032 の鋸歯）。監査補助指標であり主判定には使わない。
    - (2) **M2 独立 run の根拠を書き分け**: `outputPlacementError` 本体は write/read スケジュールのみで決まり delayLine 内容（重ね合わせ）に依存しないため、oPE 測定自体は重ね合わせに非依存。独立 run が効くのは **M1（gain 反映 reference との差分がクリーン）・M3（ピーク位置同定の分離）・CSV の n0 帰属**のため。方針（独立 run）は維持。
  - **v2.9（2026-09-11）**: 第10ラウンドレビュー（2 文書とも **v2.8 = IMPLEMENTATION-READY / Step 3 GO**、全項目 PASS・不整合なし）反映。装飾的指摘 3 点のみ: 改訂履歴の番号重複解消（本修正）、判定履歴の現行版更新、`writeLag` への L2 周期（64 callback）追記。**仕様・数値の変更なし**。
- **判定履歴**: 第8ラウンド: **v2.6 CONDITIONALLY IMPLEMENTATION-READY**（条件: §8-2 統一 — v2.7 で反映）＋ 詳細検証の修正要求（t_write 式・case 別期待値表・readMode — v2.7 で反映）→ 第9ラウンド: **v2.7 IMPLEMENTATION-READY**（writeLag 鋸歯・M2 独立 run 根拠の 2 点軽微修正のみ — v2.8 で反映）→ 第10ラウンド: **v2.8 IMPLEMENTATION-READY / Step 3 GO**（全項目 PASS・不整合なし、装飾的指摘 3 点 — v2.9 で反映）。**主判定（`|oPE| ≥ 128 → B13 不成立`）は case 間の差（−1152〜−1600 / −30720）に依存しないため、測定そのものは着手可能**。

---

## 0. v2.9 の位置づけ — work69 規定の未履行状態の解消

`doc/work69/bug_final_report.md` L787-797:

> **MT-NUPC 未完了の場合は Phase 1 以降の実装を一切禁止する。** 特に Phase 2 の `outputDelaySamples` は実測値から決定し、`Σ(partSize × numPartsIR)`（IR 長）を暫定値として実装してはならない。

**現状**: ① B13 は暫定値のまま実装済み（cpp:1009 `prevLayerTotalSamples`）② `MT-NUPC-Measurement.cpp` は実質プレースホルダ（`irLength/3` ハードコード、`peakPos >= 0` 常時 true）③ Null Test は未実施。

**実施順序**:

| Step | 内容 | 状態 |
|---|---|---|
| Step 0〜2 | content-mapping audit・content_time 確定 | **完了**（第6/7/8ラウンドで独立再導出により再確認済み） |
| **Step 3** | M2 accessor/test 実装 | **IMPLEMENTATION-READY**（本 v2.9 仕様。第9/10ラウンドで実装開始を承認 — GO） |
| Step 4 | T1〜T8 実測 | 未実施 |
| Step 5 | M1/M2/M3 相互検証 | 未実施 |

**次にやるべきことは修正ではなく測定**（Step 3 → Build → CTest 40/40 → Step 4 → M1/M2/M3 → 実測で B13 を確定）。

**役割分担**: **M1 = 波形の正しさ**、**M2 = 時間軸の正しさ（主判定）**、**M3 = 位置同定による独立証拠（補助）**。

**適用範囲**: 48 kHz / blockSize = 64 専用。

## 0.5 検証対象の切り分け

1. `outputDelaySamples` の値の正当性（暫定値 2048）
2. `delayLineReadAdd()` の読み出し時系列で、その値が L1/L2 を L0 と同じ出力時間軸へ配置しているか ← **本質**
3. delayLine 書き込み位置の内容の意味論 — **Step 0 で確定済み**

(2) の本質: `actualReadStart = max(delayReadCursor, maxRead)` の `max()` により、`delayReadCursor` 自律進行が先行すると `outputDelaySamples` は固定遅延として働かない（cpp:1749-1753, 1778）。cursor は delayLine **論理位置**であって reference 時刻でも Get 時刻でもない — **3 つは別物**。

**本番との整合**: `ConvolverProcessor.Runtime.cpp`（実 ~1173-1174）で `Add` → `Get` の順。テストも同一順序・**有効出力バッファ必須**（`Get(nullptr)` は L1/L2 読み出しをスキップ: 実 1712）。`SetImpulse` 成功で `m_ready=true`（実 1099）。

---

## 1. 層構成・gain（48 kHz / blockSize = 64）

| 項目 | 値 | 出典（相対 → 実ソース） |
|---|---|---|
| l0Part / l1Part / l2Part | 64 / 512 / 4096 | cpp:741→738, 742→739, 743→740 |
| kL0MaxParts / l0MaxLen | 32 / 2048 | h:444→**h:441**, cpp:745 |
| l0LenTarget クランプ | 2048（常時） | cpp:747 |
| L1 offset / partSize | 2048 / 512 | cpp:750-760 |
| L1 `outputDelaySamples` | **2048**（暫定） | cpp:1009→**1006** |
| L1 `delayLineCapacity` | **2624** | cpp:1010→**1007** |
| L2 `delayLineCapacity`（IR=40000） | **38976**（=34816+4096+64） | 同式 |
| L1 layerGain（tailMode=1） | **1.4375**（+3.152 dB） | cpp:671→**668**, 686→**683** |
| L2 layerGain | **1.100**（+0.828 dB） | cpp:672, 687 |

- tailMode=1 クランプ（`max(tailStartSec, 0.12)`）により 48 kHz/blockSize=64 では `l0LenTarget` 常時 2048。v1 の層構成前提（IR=2000 → L0 のみ、IR=5000 → L0+L1=2048+2952）は**正しい**。実効境界 **2048/2049**。
- `strength01 = jlimit(0,1, userTailStrength×0.5) = 0.5`（cpp:641）、`tailStrength = max(1.0,1.25) = 1.25`（cpp:665）。
- `Get()`: `layerGain = m_tailEnabled ? m_tailLayerGain[li] : 0.0`（cpp:1717→1714）。tailEnabled=false → L1/L2 無音。

### IR 長別層構成表

| IR 長 | L0 | L1 | L2 | L1 numPartsIR/ppc | L1 outputDelaySamples（暫定） |
|---|---|---|---|---|---|
| 2000 | 2000 | 0 | 0 | — | — |
| 2048 | 2048 | 0 | 0 | — | — |
| 2049 | 2048 | 1 | 0 | 1 / 1 | 2048 |
| 5000 | 2048 | 2952 | 0 | 6 / 1 | 2048 |
| 8000 | 2048 | 5952 | 0 | 12 / 2 | 2048 |
| 12000 | 2048 | 9952 | 0 | 20 / 3 | 2048 |
| 20000 | 2048 | 17952 | 0 | 36 / 5 | 2048 |
| 40000 | 2048 | 32768 | 5184 | 64 / 8 | 2048（L2: 34816） |

---

## 2. レイテンシ構造と時間軸意味論

### 2.1 基本挙動

1. **L0 は即時 Overlap-Save**（cpp:1336-1427）: ストリームレイテンシ **0**。
2. **L1/L2 は分散処理**（cpp:1538-1636）: Forward FFT は partSize 蓄積時、累積は毎 callback ppc パーティション、完了後 IFFT → `delayLineWrite(partSize)`。**1 callback あたり write 最大 1 回**。
3. **Get()**: `ringRead`（L0）＋ `delayLineReadAdd`（L1/L2, layerGain）。
4. **delayLineReadAdd**（cpp:1744-1779）:
   ```
   maxRead         = delayWriteCursor − outputDelaySamples   // アンダーフロー時 0
   actualReadStart = max(delayReadCursor, maxRead)
   if (actualReadStart + numSamples > delayWriteCursor) return;
   l.delayReadCursor = actualReadStart + numSamples;
   ```

### 2.2 時間軸意味論（**Step 0 で証明・第6/7/8ラウンド独立再導出で再確認**）

```
delayLine 論理位置 p の L1 内容 = (h1 ⋆ x)[p]      h1[p] = IR[2048+p]（L2: h2[p] = IR[34816+p]）
content_time(j) = p_content(j) + IR_offset_layer   ← n0 に依存しない
p_content(j)    = j × partSize_layer               L1: 512j、L2: 4096j
```

n0 は **M3 のピーク位置予測**（`n0 + IR_offset + peak_in_h1`）にのみ使用。

### 2.3 検証仮説（**第8ラウンド独立シミュレーションで全 case 確定 — 実測は Step 4**）

定常イベント系列（Ck = 時刻 64k、write/read-anchor は同一 callback）:

| Case | numPartsIR/ppc | first FFT | first write | **oPE（定数）** | **effectiveDelay（t−R_after）** |
|---|---|---|---|---|---|
| T3（IR=2049） | 1 / 1 | C7 | C7 | **−1600** | **384** |
| T4（IR=5000） | 6 / 1 | C7 | C12 | **−1280** | **704** |
| T5（IR=8000） | 12 / 2 | C7 | C12 | **−1280** | **704** |
| T6（IR=12000） | 20 / 3 | C7 | C13 | **−1216** | **768** |
| T7 L1（IR=40000） | 64 / 8 | C7 | C14 | **−1152** | **832** |
| T7 L2 | 2 / 1（partSize=4096） | C63 | C64 | **−30720** | **4032** |

**t_write 一般式（v2.7 修正）**:

```
blocksPerPart = partSize_layer / blockSize        // L1: 8、L2: 64
dist_cbs      = ceil(numPartsIR / partsPerCallback)
t_write(j)    = p_content(j) + 64 × (blocksPerPart + dist_cbs − 2)
oPE(j)        = 64 × (blocksPerPart + dist_cbs − 2) − IR_offset_layer   // p に依存しない定数
```

（導出: 蓄積完了 callback `C(p/64 + blocksPerPart−1)` で FFT と分散 1 回目が同時に発生し、write は `+ (dist_cbs − 1)` callback。ppc=1 のとき `dist_cbs = numPartsIR` で v2.6 の式と一致するが、ppc>1 では v2.6 の式は不正。シミュレーション検証: 全 6 case 一致。）

**仮説 P1（修正）**: `readMode` は**初回 read のみ MAXREAD**（`maxRead=0 ∧ R=0 → actual==maxRead`）、**以降（定常）は AUTONOMOUS**。`forcedSkip = 0`（ジャンプなし）。初回 MAXREAD は正常であり、「maxRead 支配の混在」と誤判定しないこと。

**仮説 P2**: `effectiveDelay` は case 別定数（上表）。
**仮説 P3**: `outputPlacementError` は case 別定数（上表）。`outputDelaySamples` を変えても不変（maxRead が自律進行を引っ張らない）— Step 4 で検証。

### 2.4 過去レビューの立場

- work69 設計は `max()` 機構を「遅延整合を保証する」としたが、構造問題（自律進行が勝つ）は考慮されていない。
- work57 静的解析（≈1312 samples）はオーダーとして整合。
- **RB-05**: コード修正済み。`remaining_bugs.md` は旧式引用のまま陳腐化 → 独立に status 更新推奨。

---

## 3. テストケース

| Case | IR 長 | tailEnabled | 期待層構成 | カテゴリ | 目的 |
|---|---|---|---|---|---|
| T1 | 2000 | true | L0 のみ | M1/M3 | FFT コア健全性基準 |
| T2 | 2048 | true | L0 のみ（ちょうど境界） | M1/M3 | 境界で L1 非発生確認 |
| T3 | 2049 | true | L0 + L1(1 sample) | M2/M3 | 境界動作 + 初回スパイク到着（期待 oPE=−1600・eff=384） |
| T4 | 5000 | true | L0=2048 + L1=2952 | M1/M2/M3 | **主テスト**: L1 三時刻（期待 oPE=−1280・eff=704） |
| T5 | 8000 | true | L0=2048 + L1=5952 | M1/M2/M3 | **ppc=2 でも oPE が T4 と同一（−1280）であることの確認**（dist_cbs が等しいため。変化しないことを明示的に検証する） |
| T6 | 12000 | true | L0=2048 + L1=9952 | M1/M2/M3 | ppc=3（期待 oPE=**−1216**・eff=768。**T4 とは異なる値** — 期待値表を case 別に照合すること） |
| T7 | 40000 | true | L0 + L1=32768 + L2=5184 | M1/M2 | L1（期待 **−1152**/832）と L2（期待 **−30720**/4032）を**独立カーソルで別々に判定** |
| T8 | 5000 | **false** | L0=2048 のみ | **仕様確認** | IR truncation + L1/L2 無音（layerGain=0）確認 |

- **入力信号と run 構成（v2.8 で根拠を書き分け — 方針は v2.7 維持）**: **M2 は impulse 位置 n0 ごとに独立 run**（Run A: n0=0、Run B: n0=4096、Run C: n0=8192、Run D: n0=16384 — それぞれ Reset から開始する別テストラン）。なお `outputPlacementError` 本体は write/read スケジュールのみで決まり delayLine 内容（重ね合わせ）に依存しないため、**oPE 測定自体は複数インパルスの重ね合わせに非依存**である。独立 run が効くのは: **M1**（gain 反映 reference との差分がクリーン）、**M3**（ピーク位置同定が単一応答で分離）、**CSV の n0 帰属**（「どのインパルス起因の波形がいつ出たか」を M3 と対応付けるとき）。M3 は複数インパルス（1 run 内に複数配置 or 複数 run 集計）を許容する。M1 のスイープは参考。
- 保存: reference / NUC / difference を 64-bit double WAV（または raw）で run ごとに保存。

## 4. リファレンス畳み込み（M1 用 — layerGain 反映）

```
y_ref[n] = (h0 ⋆ x)[n]
         + g1 · (h1 ⋆ x)[n − 2048]
         + g2 · (h2 ⋆ x)[n − 34816]         // T7 のみ
```

- `g1`/`g2` は **`layerTailGain` accessor で実測値を読む**（expected/actual/gainError を分離ログ、§7）。
- `(h⋆x)` は v1 §3.3 の IPP overlap-add スケルトンをレイヤーごとに。出力長 `inputLen + irLen − 1`、入力長 ≥ irLen×2。
- `SetImpulse` は `scale=1.0`、`enableDirectHead=false`、`filterSpec=nullptr`。

## 5. 測定方法（M1 / M2 / M3）

### M1 — 全体出力 Null Test（layerGain 反映・波形の正しさ）

reference（§4）vs NUC を **align = 0** で比較。指標: ① RMS error ② Peak error ③ 帯域別誤差スペクトル（20–200 Hz / 200 Hz–1 k / 1–10 k / 10–20 k）④ サンプル位置別誤差（L0/L1 境界 2048 付近に局在するか）。

### M2 — NUC 内部三時刻観測（**主測定**・時間軸の正しさ）

```
content_time ──┬─ writeArrivalDelay = t_write_callback − content_time   （正常量 — oPE と同値になる）
               └─ outputPlacementError = t_output_callback − content_time   （配置誤差 — 主指標）
```

**3 つの時空間**: `p_content`（delayLine 論理位置）／ `t_write_callback`（write event を観測した Add 側 callback 時刻 = Add 引数ブロック先頭サンプルの絶対インデックス）／ `t_output_callback`（read-anchor event を観測した Get 側 callback 時刻）／ `content_time = p_content + IR_offset_layer`。

**一般式**:

```
p_content(j)       = j × partSize_layer
content_time(j)    = p_content(j) + IR_offset_layer
write event (j)    : W_before == p_content(j) ∧ W_after == p_content(j) + partSize_layer
read-anchor event  : readExecuted == true ∧ actualReadStart == p_content(j)
                     （block j の先頭 64 samples を読み始めた callback。全体を 1 callback で読む意味ではない）
```

**不変条件**: 1 Add callback あたり `delayLineWrite` 最大 1 回（numSamples=64 < partSize）。

**観測機構**:

- `MKLNonUniformConvolver.h` の `private:` 直後に `friend struct convo::NUPCTestAccess;` を 1 行追加（本番オーバーヘッド 0）
- `src/tests/NUPCTestAccess.h` 新規:
  ```cpp
  struct NUPCTestAccess final {
      // ★ 読み出しのみ（setter・状態変更・publish・retire 禁止 — ISR Bridge "Observer は副作用を持たない" 原則）
      static int  numActiveLayers(const MKLNonUniformConvolver& c) noexcept;
      static int  layerPartSize        (const MKLNonUniformConvolver& c, int li) noexcept;
      static int  layerNumPartsIR      (const MKLNonUniformConvolver& c, int li) noexcept;
      static int  layerPartsPerCallback(const MKLNonUniformConvolver& c, int li) noexcept;
      static int  layerOutputDelaySamples(const MKLNonUniformConvolver& c, int li) noexcept;
      static int  layerDelayLineCapacity(const MKLNonUniformConvolver& c, int li) noexcept;
      static double layerTailGain      (const MKLNonUniformConvolver& c, int li) noexcept;
      static uint64_t layerDelayWriteCursor(const MKLNonUniformConvolver& c, int li) noexcept;
      static uint64_t layerDelayReadCursor (const MKLNonUniformConvolver& c, int li) noexcept;
  };
  ```
- 記録はテストアプリ側: `Add()` 直前・直後（write event 検出）と `Get()` 直後（同一スレッド、本番と同じ Add→Get 順序、有効出力バッファ）。

**CSV 記録項目**:

```
run_id（n0 ごとに独立）, impulse_pos(n0), t, layer, layerGain（実測）,
delayWriteCursor_before/after, delayReadCursor_before/after,
maxRead, actualReadStart, readExecuted(0/1), readMode(MAXREAD/AUTONOMOUS/NONE),
forcedSkip, skippedTotal, outputDelaySamples, partSize, numPartsIR, partsPerCallback
```

`maxRead`/`actualReadStart`/`readExecuted` はテスト側で実装と同一の式から再構成（**blockSize を使用**）:

```cpp
const uint64_t maxRead = (W >= outputDelay) ? (W - outputDelay) : 0;
const uint64_t actualReadStart = std::max(R_before, maxRead);
const bool readExecuted = actualReadStart + static_cast<uint64_t>(blockSize) <= W;
const char* readMode = !readExecuted ? "NONE" : (actualReadStart == maxRead ? "MAXREAD" : "AUTONOMOUS");
```

**導出指標**:

| 指標 | 定義 | 期待値（case 別 — §2.3 表参照） |
|---|---|---|
| `content_time(j)` | `p_content(j) + IR_offset_layer` | n0 非依存 |
| `t_write_callback(j)` / `t_output_callback(j)` | write/read-anchor event の callback 時刻 | `p_content(j) + 64×(blocksPerPart + dist_cbs − 2)`（両者同一 callback） |
| **`outputPlacementError(j)`** | `t_output_callback(j) − content_time(j)` | **case 別定数**（T3 −1600 / T4 −1280 / T5 −1280 / T6 −1216 / T7 L1 −1152 / T7 L2 −30720） |
| `writeArrivalDelay(j)` | `t_write_callback(j) − content_time(j)` | oPE と同一 |
| `forcedSkip(t)` / `skippedTotal` | `max(0, actualReadStart − R_before)` と累積 | 0 |
| `readMode(t)` | MAXREAD / AUTONOMOUS / NONE | **初回 read のみ MAXREAD、以降 AUTONOMOUS** |
| `effectiveDelay(t)` | `t − delayReadCursor_after` | **case 別定数**（384 / 704 / 704 / 768 / 832 / 4032） |
| `writeLag(t)` | `delayWriteCursor − delayReadCursor` | **定常鋸歯**（シミュレーション確認: L1 は write 直後 448 → 64 ずつ減少 → 0 → 次 write で反復。L2 は 0..4032、**周期 64 callback**（= partSize/blockSize）。capacity 2624 / 38976 に対し十分小さく内容整合は成立）。**監査補助指標であり主判定には使わない** |
| coverage | 観測済み event 数 / 発生ブロック数 | 1.0 |

### M3 — インパルス応答の位置同定（**補助証拠**・位置同定による独立証拠）

- インパルス入力に限る。k\* はセグメント単独計算:
  - `k*(A)`: ref = IR[0..2047] vs NUC `[0, 2048+Dmax)` 窓。**`k*(A) ∈ [−2, +2]`**
  - `k*(B)`: ref = IR[2048..] vs NUC 全範囲。L0+L1 混在のため L1 純粋でない — M2 の**独立な補助証拠**（同格にしない）
  - **ピーク位置理論予測**: `n0 + IR_offset + peak_in_h1`
  - T7 の L2 も独立計算（gain は相関形状に影響しない）
- M2 と M3 が一致すれば測定系の信頼性が確定。

## 6. 判定基準（64 量子化整合）

**判定帯（事前規定・後付け禁止）**: 規定プロトコルでは n0/p_content/IR_offset/callback 時刻がすべて 64 の倍数 → **oPE は常に 64 の倍数**:

```
outputPlacementError == 0      → B13 正当（sample 単位で正確）          ← **唯一の合格条件**
1 ≤ |error| ≤ 64               → 要精査（**合格ではない**）。実質 |error|=64 のみ発火（1 callback 位相ずれ）
                                  ※ 1..63 が出た場合は M2 再構成式のバグ（測定器側不具合）として扱う
|error| ≥ 128                  → B13 補償不成立（§2.3 期待値表: −1152〜−1600 / −30720）
```

| 条件 | 判定 |
|---|---|
| T1 M1 RMS < −90 dB かつ `k*(A) ∈ [−2,+2]` | コア健全 → Gardner 判定へ進む |
| T4〜T7 で M2 `outputPlacementError` 定常 == 0 かつ M1（gain 反映）RMS < −90 dB | B13 正当 → **Gardner 棄却** |
| T4〜T7 で M2 `outputPlacementError` が `≥128` の定常値（§2.3 期待値表の case 別定数と一致）をとり、M1 誤差が L1/L2 区間に局在 | **B13 遅延補償の実効不成立確定**（期待値表どおりの定数なら `max()` 構造問題の裏付け）→ §8 修復を実施 |
| 期待値表と異なる定常値（例: −1216 ではなく −1152 が T6 で観測） | モデル修正（分散タイミング等）として精査 — 主判定（≥128 → 不成立）は変わらないが、記録を取る |
| `skippedTotal > 0` かつ 2 段階確認で欠落確定 | 欠落バグとして別起票 |
| M1 誤差が全帯域にほぼ均一 | FFT 丸め/加算順序差。Gardner とは別問題 |

## 7. 実装方法（Step 3）

- **既存 `MT-NUPC-Measurement.cpp` は完全置換**。推奨構造:
  ```
  NUPCTestAccess.h（topology / gain / cursor accessor）
        │
        ▼
  NUPCNullTestV2.cpp
        ├─ T1..T8（M2 は n0 ごとに独立 run）
        ├─ M1（gain 反映 reference）
        ├─ M2（三時刻 CSV）
        ├─ M3（位置同定）
        └─ CSV/WAV 出力
  ```
- `MTNUPCMeasurement` ターゲット（CMakeLists.txt 実ファイル ~1058 行、`CONVOPEQ_ENABLE_ISR_TESTS`）を流用。
- 本番変更は **friend 1 行のみ**。accessor は読み出しのみ。
- **スタートアップゲート（最初のテスト実行時に log）**:
  ```
  sampleRate=48000、blockSize=64
  T4: numActiveLayers=2、L0 partSize=64、L1 partSize=512、numPartsIR=6、ppc=1、
      outputDelaySamples（log のみ・assert 禁止）、delayLineCapacity=2624、tailGain（log）
  T7: numActiveLayers=3、L2 partSize=4096、L2 IR offset=34816、L2 delayLineCapacity=38976、tailGain（log）
  ```
  `assert(outputDelaySamples == 2048)` は**禁止**（Phase 1 の目的自体を前提にしてしまう）。
- **layerGain は expected/actual/gainError を分離ログ**（将来の tail contour 式変更に耐える）。
- 実装ゲート: cppcheck / clang-tidy / CTest 40/40 → Step 4 実測。

## 8. 実施後のアクション（Step 4〜5）

1. T4〜T7 の M2 三時刻を確定（期待値は §2.3 case 別表）。
2. **outputPlacementError が判定帯外と確定した場合**:
   - 不採用修復（第6/7/8ラウンド検証済み）: maxRead 固定（再読み出し）、maxRead スナップ+連続読み（residual +256）— いずれも `error == 0` にならない。
   - 修復方向（ストリーム時刻ベース再設計）: ① `t_output = p + IR_offset_layer` を直接実現 ② `outputDelaySamples` を「delayLine 論理 0 が出力軸上で遅延すべきストリームサンプル数（パイプライン lead + IR offset の合算）」と再定義し M2 で `error == 0` になる値を Phase 1 実測で確定 ③ 論理位置↔ストリーム時刻対応表 + 固定ストリーム遅延
   - `delayLineCapacity` を RB-05 形式から再計算。
3. **修復後の検収基準は §6 と同一**: `error == 0 → 正当`、`1..64 → 要精査（合格ではない）`、`≥128 → 不成立`。「callback 粒度 ±64」は診断上の分類としてのみ意味を持ち、**合格条件にはならない**（v2.7 で §6 と統一）。
4. 欠落確定分は別起票。
5. 閉包後、`convolver_timing_verification_report.md` の A ランク未解決事項を更新。

---

## 付録 A: 修正履歴サマリ（v1 → v2.8 の差分）

| # | 対象 | 現行（v2.8）での扱い |
|---|---|---|
| 1-21 | v1〜v2.5 の修正（align=0、M1/M2/M3、層構成、境界 2048/2049、outputDelaySamples 実測、方法 B、差分式廃止、M3 補助、複数インパルス、T3/T8 分離、effectiveDelay 判定廃止、maxRead/readMode CSV、forcedSkip、時間軸構造、friend、write/read event、検証仮説、skip 2 段階、M3 限界、n0 除去、−30720、64 量子化、Step 0） | 前版どおり維持 |
| 27-33 | v2.5 必須修正（一般式・不変条件・layerGain・capacity 2624・writeArrivalDelay・effectiveDelay・修復差し替え） | 維持 |
| 34-36 | 本番 Add→Get 順序・適用範囲・CMakeLists | 維持 |
| 37-40 | v2.6 の n0 除去・−30720・判定帯・P2=704 | 維持 |
| 41-44 | Step 3 ゲート・L2 capacity・行番号・RB-05 | 維持（行番号は 46 で修正） |
| 45 | M1/M2/M3 役割分担 | 維持 |
| **46** | **付録 B の不整合** | **修正**: outputDelaySamples 1009→実 1006、m_ready（実 1099）、Get nullptr（実 1712）、layer1Gain（実 668）、m_tailLayerGain[1]（実 683）、kL0MaxParts（h:441）を追加。maxRead は offset −2 |
| **47** | **§8-2 検収 `|error| ≤ 64` 許容** | **削除・§6 統一**（`error == 0` のみ合格） |
| **48** | **t_write 式（ppc>1 で不正）** | **一般式に修正**: `t_write(j) = p_content(j) + 64×(blocksPerPart + dist_cbs − 2)` |
| **49** | **「L1 −1280 / eff 704」の全 case 一般化** | **case 別期待値表に差し替え**（T3 −1600/384、T4 −1280/704、T5 −1280/704、T6 −1216/768、T7 L1 −1152/832、T7 L2 −30720/4032） |
| **50** | **readMode「常に AUTONOMOUS」** | **修正**: 初回 read のみ MAXREAD、以降 AUTONOMOUS。P1 を「定常では AUTONOMOUS」に修正 |
| **51** | **複数インパルス同時配置** | **M2 は n0 ごとに独立 run**（write event の n0 帰属のため）。M3 は複数インパルス許容 |
| **52** | **T5 の目的** | 「ppc=2 でも oPE が T4 と同一であることの確認」に変更 |
| **53** | **§6 の 1..63** | 「M2 再構成式のバグ（測定器側不具合）」として扱う旨を明記 |
| **54** | **v2.7 `writeLag` = 「定常定数」** | **訂正 → 定常鋸歯**（L1: 0..448、L2: 0..4032、write 直後ピーク→64 ずつ減少→0）。監査補助指標のみ | 
| **55** | **v2.7 M2 独立 run の根拠文** | 方針（独立 run）は維持。根拠を「oPE は重ね合わせに非依存。独立 run は M1/M3 と CSV 帰属のため」に書き分け |

## 付録 B: 行番号マッピング（ConvoPeq.md 相対 → 実ソース）

| シンボル | 本文引用（相対） | 実ソース | offset |
|---|---|---|---|
| `l0Part` | 741 | 738 | −3 |
| `l1Part` | 742 | 739 | −3 |
| `kL0MaxParts` | h:444 | **h:441** | −3 |
| `outputDelaySamples =` | 1009 | **1006** | −3 |
| `delayLineCapacity` | 1010 | **1007** | −3 |
| `layer1Gain`（tailMode=1） | 671 | **668** | −3 |
| `m_tailLayerGain[1] =` | 686 | **683** | −3 |
| `layerGain =`（Get） | 1717 | **1714** | −3 |
| `Get` の `output != nullptr` | 1715 | **1712** | −3 |
| `maxRead` | 1749 | **1747** | **−2** |
| `m_ready = true`（publish） | 1102 | **1099** | −3 |
| 本番 Add→Get | — | Runtime.cpp **1173-1174** | — |

offset は**一様でない（−1〜−3）**。Step 3 の実装・コミットでは**実ソース行番号のみ**を用いること。
