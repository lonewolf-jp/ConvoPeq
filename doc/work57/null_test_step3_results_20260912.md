# Gardner Null Test Step 3 実測報告書（B13 遅延補償の実効性検証）

- **日付**: 2026-09-12
- **版**: v1.2（第12ラウンド監査反映 — 監査履歴は §11）
- **実施**: Step 3（測定系実装）+ smoke run（Step 4 解析用の全データ取得）
- **仕様**: `doc/work57/null_test_procedure_v2.md`（v2.9）
- **先行監査**: 第1〜第10ラウンドの外部レビュー（v1 → v2.9 確定）
- **総合判定（v1.1）**: **B13 遅延補償：実効不成立。NUPC のレイヤー出力に大規模な時間軸配置誤差（先行）を確認 — Gardner 型時間ズレ仮説を支持する。** 全 case で `outputPlacementError` が **|oPE| ≥ 128**（判定帯 FAIL）の定常定数を示し、仮説 P1〜P3（手順書 §2.3）と**全件一致**した。
- **ステータス（第11/12ラウンド監査で確定）**: **B13 FAILURE CONFIRMED / Step 3 FUNCTIONAL GATE PASS / STATIC ANALYSIS PENDING / 次工程 = D1〜D4 Repair Design（修復実装は D1 の read policy 数学的定義まで未開放）**
- **Step 3 のゲート**:

  | Gate | 状態 |
  |---|---|
  | Step 3 実装 | PASS |
  | icx Release clean ビルド（GATE-OK・error 0） | PASS |
  | CTest 40/40 | PASS |
  | smoke run（structural failures = 0） | PASS |
  | cppcheck | **PENDING**（Step 4 実装時に実施） |
  | clang-tidy | **PENDING**（同上） |

  → **Functional Step 3 gate: PASS** / **Static-analysis gate: PENDING**（「Step 3 ゲート完全 PASS」とは記録しない）

---

## 1. 実装内容

| ファイル | 変更 | 備考 |
|---|---|---|
| `src/MKLNonUniformConvolver.h` | `private:` 直後に `friend struct NUPCTestAccess;` を追加（**本番変更 1 行のみ**） | qualified 名（`convo::NUPCTestAccess`）は事前宣言が必要なため unqualified — 初回ビルドで発見・修正 |
| `src/tests/NUPCTestAccess.h`（新規） | 純粋読み取り accessor 9 種（topology / B13 構成 / layerTailGain / cursor） | setter・publish・retire なし（ISR Bridge Observer 原則）。friend 宣言は `private:` 直後 |
| `src/tests/MT-NUPC-Measurement.cpp` | **完全置換**（旧プレースホルダ: `irLength/3` ハードコード・`peakPos>=0` 常時 true） | T1〜T8（M2 は n0 独立 run）、M1（gain 反映 reference）、M2（三時刻 CSV）、M3（相関位置同定）。出力は `std::cout`/`std::ofstream` に統一 |
| `CMakeLists.txt` | **変更なし**（既存 `MTNUPCMeasurement` ターゲットが同一ソースパスをビルド） | |

## 2. ビルド検証

- **icx Release clean ビルド**: 成功（556 ターゲット・error 0・`[GATE-OK]`・`[SUCCESS] Executable created successfully`）。
  - `MTNUPCMeasurement.exe`（9,877,504 bytes）生成。
  - `build_identity_gate.py` の COHERENCE-4（dirty ツリー `104200ec+dirty`）は fail-closed が正しく機能し、ゲート指定の正攻法リカバリ（明示 clean）で通過。
- **MSVC Debug ビルドは不可**（既存問題・本変更と無関係）: (a) `RuntimeHealthMonitorTierTests` が JuceHeader 未生成ターゲットから `AudioEngine.h` を include (b) `AudioEngine.EQResponse.cpp` の AVX intrinsic が Debug 構成で AVX フラグなし。過去の CTest 40/40 実績と同一の **icx Release** を採用。

## 3. CTest（Release・40 件）

**40/40 PASS（100%、32.71 秒）**。

- 初回のみ `MTNUPCMeasurement` が `0xc0000374`（ヒープ破壊）で失敗。原因: `computeBandSpectrum` の **MKL DFTI INPLACE real FFT（CCS 形式）は N+2 doubles を出力するのに N サイズのバッファを確保しており 2 doubles あふれ**。`n + 2` 確保に修正し 40/40 回復。**既存 39 テストには影響なし**（friend 追加はコンパイル時のみの変更）。

## 4. 実測結果 — M2 三時刻観測（主判定）

テスト条件: 48 kHz / blockSize = 64 / filterSpec=nullptr（tailMode=1）/ enableDirectHead=false / scale=1.0。
M2 は impulse 位置 n0 ごとに独立 run（Reset から開始）。

### 4.1 case 別の定常値（全 run 一致）

| Case | numPartsIR/ppc | read-anchor 件数 | **oPE 実測（定数）** | 期待値（§2.3） | 照合 | **effectiveDelay（定常）** | 期待値 | 判定帯 |
|---|---|---|---|---|---|---|---|---|
| T3（IR=2049） | 1 / 1 | 35+ | **−1600** | −1600 | **MODEL-MATCH** | **384** | 384 | FAIL(≥128) |
| T4（IR=5000） | 6 / 1 | 35（run0） | **−1280** | −1280 | **MODEL-MATCH** | **704** | 704 | FAIL |
| T5（IR=8000） | 12 / 2 | — | **−1280** | −1280 | **MODEL-MATCH** | **704** | 704 | FAIL |
| T6（IR=12000） | 20 / 3 | — | **−1216** | −1216 | **MODEL-MATCH** | **768** | 768 | FAIL |
| T7 L1（IR=40000） | 64 / 8 | 171（run0） | **−1152** | −1152 | **MODEL-MATCH** | **832** ※ | 832 | FAIL |
| T7 L2 | 2 / 1（partSize=4096） | 21（run0） | **−30720** | −30720 | **MODEL-MATCH** | **4032** | 4032 | FAIL |

※ T7 L1 の effDelay はテストの表示バグ（`effConst` がレイヤー別配列でなく L2 値 4032 で上書き表示）— **CSV から再計算した正値は 832**（1365 callback 分定常一致）。

### 4.2 CSV 再計算による定常値の詳細（M2_T4_run0 / M2_T7_run0）

- **T4 L1**: read-anchor oPE = **−1280 × 35 件（全 anchor 完全一致）**。effectiveDelay = **704 × 274 件**（warm-up 11 callback 分 0..640 を除き定常）。
- **T7 L1**: read-anchor oPE = **−1152 × 171 件**。effectiveDelay = **832 × 1365 件**。
- **T7 L2**: read-anchor oPE = **−30720 × 21 件**。effectiveDelay = **4032 × 1315 件**（warm-up 63 callback 分 0..3968 に 1 件ずつ）。

### 4.3 仮説 P1〜P3 の検証（手順書 §2.3）

| 仮説 | 実測 | 判定 |
|---|---|---|
| **P1**: readMode は初回 read のみ MAXREAD、以降（定常）AUTONOMOUS。forcedSkip = 0 | 全 case で初回 MAXREAD → 以降 AUTONOMOUS、forcedSkip = 0 | **成立** |
| **P2**: effectiveDelay は outputDelaySamples と無関係の定数 | T4=704 / T7 L1=832 / T7 L2=4032（すべて case 別定数、outputDelaySamples と無関係） | **成立** |
| **P3**: oPE は case 別定数（先行・負値）、write と read-anchor は同一 callback | T3=−1600 / T4=−1280 / T5=−1280 / T6=−1216 / T7 L1=−1152 / T7 L2=−30720（全件定数） | **成立** |
| 現行設定値では `maxRead` が定常 read cursor を拘束していない | outputDelaySamples=2048（L1）/ 34816（L2）で oPE は構造定数のみに従う | **成立**（maxRead が自律進行を引っ張らない構造の実測裏付け）。**注意（v1.1）**: 「outputDelaySamples を変えても oPE 不変」の一般化は **A/B 実験未実施のため実測事実としては記録しない** — 本報告書の実測事実は「現行設定値で拘束されていない」まで |

### 4.4 coverage

全 case・全 layer で write event 数 == read-anchor 数 == 発生ブロック数（**coverage = 1.0**、欠落 event なし）。`STRUCTURAL-FAIL` は 0 件。

## 5. 実測結果 — M1 layerGain 反映 Null Test（align=0・波形の正しさ）

| Case | RMS [dB] | ゲート (< −90 dB) | Peak [dB] | 帯域別誤差 [dB]（0–200/200–1k/1k–10k/10k–24k） | セグメント RMS [dB]（L0/L1/L2） |
|---|---|---|---|---|---|
| T1（IR=2000） | **−311.39** | **PASS** | −291.70 | −311.9 / −312.3 / −311.5 / −311.3 | −311.5 / — / — |
| T2（IR=2048） | **−311.37** | **PASS** | −291.76 | −311.9 / −312.0 / −311.5 / −311.3 | −311.4 / — / — |
| T3（IR=2049） | −23.29 | FAIL | +14.62 | −24.0 / −23.3 / −23.3 / −23.3 | −26.3 / **0.00** / — |
| T4（IR=5000） | +0.91 | FAIL | +17.07 | −2.3 / +0.7 / +0.8 / +1.0 | **−0.47** / **+1.54** / — |
| T5（IR=8000） | +1.58 | FAIL | +16.93 | −0.0 / +1.6 / +1.5 / +1.7 | −0.47 / +2.14 / — |
| T6（IR=12000） | +1.87 | FAIL | +17.49 | +0.7 / +1.5 / +2.1 / +1.8 | −0.70 / +2.40 / — |
| T7（IR=40000） | +1.99 | FAIL | +21.24 | +1.5 / +2.0 / +2.0 / +2.0 | −0.92 / +2.49 / **0.00** |
| T8（IR=5000, bypass） | −10.08 | **EXCLUDED**※ | +9.67 | −1.9 / −15.3 / −29.8 / −8.1 | −10.08 / — / — |

- **T1/T2 PASS（−311 dB）**: L0 単層の FFT/加算コアは sample-accurate に正確 — **Gardner 判定の前提（コア健全性）が成立**。
- **T4〜T7 FAIL**: 期待どおり。L0 区間の誤差（−0.47〜−0.92 dB）は **L1 成分が先行（−1280 など）して L0 区間末尾に混入**するため。L1/L2 区間の誤差（+1.5〜+2.5 dB / 0.00 dB）は時間ズレ＋gain 差による。T7 L2 seg 0.00 dB は L2 区間の ref と NUC が全く別の位置に配置され同レベルのエネルギーが存在することを示す。
- **T8 ※（v1.1 で格下げ — M1 FAIL ではない）**: **EXCLUDED / 比較不能（filterSpec mismatch）**。NUC 側は `applySpectrumFilter`（Natural HC/LC コサインロールオン）適用済み IR、reference 側は raw IR という**異なる信号モデルの比較**であり、B13 の M1 failure ではない。T8 の本来の検証対象（`numActiveLayers == 1`、L1/L2 layerGain = 0、IR truncation = 2048）は **M2/topology ログで PASS** と記録。

## 6. 実測結果 — M3 位置同定（補助証拠）

| Case | k\*(B) 実測 | n0 + IR_offset | diff（=oPE 相当） | 期待 oPE | 照合 |
|---|---|---|---|---|---|
| T3 L1 | 448 | 2048 | **−1600** | −1600 | **一致** |
| T4 L1 | 768 | 2048 | **−1280** | −1280 | **一致** |
| T5 L1 | 768 | 2048 | **−1280** | −1280 | **一致** |
| T6 L1 | 832 | 2048 | **−1216** | −1216 | **一致** |
| T7 L1 | 896 | 2048 | **−1152** | −1152 | **一致** |
| T7 L2 | 2114 | 34816 | **−32702** | −30720 | **不一致（−1982）** |

- **L1 の M3 は全 case で M2 と完全一致** — 測定系の相互検証が成立。
- **T7 L2 の M3 が不一致**: L2 セグメント（減衰テールのノイズ IR）の波形自己相関が高く、相関ピークの位置が曖昧になったもの（手順書 §5 M3 の限界に明記済みの L0+L1/L2 混在・非純粋性に加え、IR の L2 区間に顕著なフィンガープリントがない）。**M2（カーソル直接観測）を主判定とする手順書の方針どおり、T7 の B13 判定は M2 の −30720 を採用する**。

## 7. B13 判定（手順書 §6 判定帯の適用）

```
outputPlacementError == 0      → B13 正当（唯一の合格条件）
1 ≤ |error| ≤ 64               → 要精査（合格ではない）
|error| ≥ 128                  → B13 補償不成立
```

**実測: 全 case・全 run で oPE が |≥128| の定常定数（−1152〜−1600 / −30720）を示した。**

→ **B13 遅延補償（NUPC レイヤー間遅延アライメント）は実効不成立**と確定（**B13 FAILURE CONFIRMED**）。

- **因果関係の表現分離（v1.1）**: 本実測で確定しているのは **「B13 が sample-accurate なストリーム時間配置を実現していない」こと**（M2 による直接 cursor 観測）。「Gardner 型時間ズレ」という名称・因果解釈は仮説の命名であり、本実測（M1/M2/M3 の組合せ）はそれを**支持する**という表現を正式記録とする（測定結果から数学的に必要な命名ではない）。

- 仮説 P3 の予測（§2.3: `oPE = 64×(blocksPerPart + dist_cbs − 2) − IR_offset`）は**符号・値とも全件一致** — `max()` 自律進行が `maxRead` を常に支配する構造問題（§0.5）が実測で裏付けされた。
- `outputDelaySamples = 2048 / 34816`（暫定値）は**現行 read policy の下では、定常状態の固定ストリーム遅延を規定するパラメータとして機能していない**（§4.3 の限定どおり、A/B 実験未実施 — 実測事実は oPE が現行設定値に依存しない定数であることまで）。work69「outputDelaySamples は実測値から決定」規定の観点でも、暫定値のまま実装された B13 は修正が必要。
- **T4（IR=5000）の音響的帰結**: IR[2048..5000) のテール成分が本来の位置より **1280 サンプル（26.7 ms @48kHz）早く +3.15 dB（layer1Gain 1.4375）で出力に混入**。L2 が発動する IR（>34816）では 30720 サンプル（640 ms）早く +0.83 dB で混入。work57 静的解析（2026-06-25）が指摘した「低音ジジジ」等の時間整合性問題の原因候補として整合。

## 8. 既知の注記・限界

1. **T7 L1 の effectiveDelay 表示バグ**: `effConst` がレイヤー別配列でないため、T7 の L1 行に L2 の値（4032）を誤表示。CSV 再計算で正値 **832** を確認（1365 callback 定常一致）。実測データ自体は CSV から完全に復元可能（本報告書 §4.2）。次回改修で layer 別配列化（`effConst[3]`）。
2. **T7 L2 の M3 不一致**（§6）: 波形ベース補助証拠の限界。M2 主判定に影響なし。
3. **T8 の M1 = EXCLUDED**: filterSpec mismatch（NUC = フィルタ適用 IR、reference = raw IR という異なる信号モデルの比較）。T8 の本来の確認（truncation・無音）は topology で PASS（v1.1 で §5 と整合 — M1 failure ではない）。
4. **cppcheck / clang-tidy 未実施**（Step 3 ゲートの一部）: DFTI バッファ修正済みだが、次回サイクル（Step 4 実装時）に実施すること。
5. **スイープ入力（M1 補助）と L2 の M3 は未実装/限定**: 手順書 §3 の「periodic impulse」も未実装（オプション項目）。
6. **commit 未実施**: 本番変更（friend 1 行）+ テスト 2 ファイル + doc 3 件（手順書 v2.9・監査記録・本報告書）は未 commit（commit 判断はユーザー側）。

### 8.1 Step 4 開始前のハーネス修正（第12ラウンド監査で確定 — 修復実装に先立って実施）

| # | 修正 | 内容 |
|---|---|---|
| **H1** | `effConst` → layer 別保持（`effConst[3]`） | T7 L1 の console summary が L2 値（4032）で上書き表示される問題。CSV から復元可能（正値 832）だが、監査資料として console summary と CSV の不整合を残さない |
| **H2** | `computeBandSpectrum` の N+2 修正を cppcheck / clang-tidy で検証 | 0xc0000374 は**テストハーネス自身のメモリ安全性問題**（INPLACE real FFT は CCS 形式で N+2 doubles 必要なのに N 確保 → 2 doubles あふれ）。修正は済んだが、「既存 39 テストに影響なしだから問題なし」で終わらせず、**Step 4 前に静的解析を通す**（STATIC-ANALYSIS PENDING の解消） |
| **H3** | T8 のハーネス表記を EXCLUDED に | smoke ログ上の T8 は FAIL 表記のまま（ハーネス未修正）— 報告書 v1.1 の EXCLUDED 解釈とハーネス出力を一致させる |

## 9. 次の一手（Step 4 — **B13 Repair Design / Stream-Time Mapping Proof を先に固定**）

**v1.1 注意**: §8 の修復方向は本実測から導かれた**必要条件の観測**であり、実装式までは確定していない。**修復実装には着手せず、先に B13 Repair Design / Stream-Time Mapping Proof を行う**（第11ラウンド監査方針）:

- **D1 — B13 stream-time mapping proof**: 各 layer について `p_content(j) / t_write(j) / t_output(j) / content_time(j)` を再定義し、**`t_output(j) = content_time(j)` が成立する read policy を数学的に定義**する（logical `p` → stream `t` の写像が全 L1/L2 callback sequence に対して一意に定義できるか）。**最初に stream time の原点 `t = 0` を固定**し、5 つの時刻 — input sample time / partition content time / IFFT completion time / delay-line write time / stream output time — を明示定義する。特に `t_write(j)` と `t_output(j)` を**同一視しない**（本実測で確定した問題は「content の論理位置」と「stream 上でその content が出力される位置」の不整合である）。
- **D2 — capacity re-proof**: read policy を変更する場合、RB-05 の capacity proof（`outputDelaySamples + partSize + maxBlockSize` 形式）を**必ず再実施**する。
- **D3 — warm-up / first-read proof**: 初回（MAXREAD）→ warm-up → steady state を分離して証明（本実測で確認された初回 MAXREAD → AUTONOMOUS 遷移が修復後も整合するか）。
- **D4 — 代表ケース**: 修復設計検証は **T4（ppc=1）/ T6（ppc=3）/ T7 L2（partSize=4096）** の 3 ケースでよい（情報量最大。全ケースのフル実行は毎回不要）。

修復方向（必要条件の観測として維持）:

不成立の原因は `delayLineReadAdd()` の `actualReadStart = max(delayReadCursor, delayWriteCursor − outputDelaySamples)` において **`delayReadCursor` の自律進行（writeCursor 追従）が `maxRead` を常に先行する**構造にある。修復方向（手順書 §8、maxRead 固定/スナップ方式は不採用 — 再読み出し・residual +256 が実測/シミュレーションで確認済み）:

1. **ストリーム時刻ベース**: `t_output = p_content + IR_offset_layer` を直接実現する読み出し（`p = t − IR_offset` の位置を読む）
2. **`outputDelaySamples` の再定義**: 「delayLine 論理 0 が出力軸上で遅延すべきストリームサンプル数（パイプライン lead + IR offset の合算）」とし、Phase 1 実測（本報告書 §4 の lead 実測値: L1=768、L2=4096）から確定
3. 書き込み側で論理位置↔ストリーム時刻の対応を保持し、読み出し側で固定ストリーム遅延

修復後の検収は §6 統一基準（**`error == 0` のみ合格**）。`delayLineCapacity` は RB-05 形式から再計算。

## 10. 附属資料

| 資料 | 位置 |
|---|---|
| M2 CSV（21 run） | `.auto/nulltest/nupc_v29_csv/M2_*.csv` |
| smoke run stdout | `.auto/nulltest/smoke_run.log` |
| ビルドログ（icx Release clean） | `.auto/nulltest/build_icx_rel_clean.log` |
| CTest ログ（Release 40/40） | `.auto/nulltest/rebuild_ctest.log` |
| 実行バイナリ | `build-icx/MTNUPCMeasurement_artefacts/Release/MTNUPCMeasurement.exe` |
| 手順書（v2.9） | `doc/work57/null_test_procedure_v2.md` |
| Step 0 監査記録 | `doc/work57/content_mapping_audit.md` |

## 11. 監査履歴（本報告書）

| ラウンド | 判定 | 主要指摘 | 対応 |
|---|---|---|---|
| 第11ラウンド（監査 + 精密検証の 2 文書） | **B13 実効不成立： ACCEPT** | (1) Step 3 gate 完全 PASS 表記 → **Functional PASS / Static-analysis PENDING に分離** (2) T8 M1 → **EXCLUDED** に格下げ (3) Gardner 命名の表現分離（確定は「sample-accurate な配置未実現」まで） (4) outputDelaySamples A/B 実験未実施の明記（「変えても不変」は実測事実としない） (5) §9 修復実装 → **Repair Design / Stream-Time Mapping Proof 先行**（D1〜D4） | **v1.1 で全件反映** |
| 第12ラウンド（監査 + 精密検証の 2 文書） | **v1.1 ACCEPT**（B13 FAILURE CONFIRMED 確定 / Step 4 repair implementation は D1 証明まで未開放） | (1) **Step 4 前のハーネス修正 2 点**: H1 `effConst` layer 別保持、H2 DFTI N+2 修正を cppcheck/clang-tidy で検証（メモリ安全性問題として終わらせない） (2) **D1 に stream time 原点の固定を追加**（5 つの時刻の明示定義、`t_write` と `t_output` の非同一視） (3) T8 ハーネス表記の EXCLUDED 化 (4) §7 の outputDelaySamples 表現を §4.3 の限定に整合 | **v1.2 で全件反映（本版）**。H1/H2/H3 は Step 4 開始前の実装タスク（§8.1） |

- **独立検証（第11ラウンド）**: CSV 21 本を監査側で再計算し、M2 全値（oPE 定数・effDelay 定数・readMode 遷移・forcedSkip=0・coverage）、P1〜P3、topology（capacity 2624 / 38976、ppc、numPartsIR、gain 1.4375 / 1.100）を確認。件数の ±1 差（本報告書 274 vs 再計算 273 等）は集計境界差で結論に影響なし。
- **独立検証（第12ラウンド）**: 21 CSV 全走査で `outputDelaySamples` の値変動なし（L1={2048} のみ、L2={34816} のみ — A/B 実験未実施の裏付け）、T8 の `FilterSpec` 経路（cpp 277-280/543）と `applySpectrumFilter` 未反映の確認、cppcheck/clang-tidy 成果物の不在（PENDING が事実）を確認。
- **バージョン**: v1.0（2026-09-12 作成）→ v1.1（第11ラウンド監査反映）→ v1.2（第12ラウンド監査反映・本版）。

## 12. 正式ステータス（第12ラウンド監査で確定）

```
B13 FAILURE CONFIRMED
  — 全 case |oPE| ≥ 128 の定常定数（−1152〜−1600 / −30720）
  — sample-accurate なレイヤー時間配置は未実現（M2 cursor 直接観測）

Step 3 FUNCTIONAL GATE: PASS
  — 実装 / icx Release clean / CTest 40/40 / smoke (structural=0)

Step 3 STATIC-ANALYSIS GATE: PENDING
  — cppcheck / clang-tidy 未実施（Step 4 前に H2 とともに解消）

次の一手: B13 Repair Design / Stream-Time Mapping Proof (D1〜D4)
  — 修復実装は D1 の read policy 数学的定義の後
  — Architectural 制約: B13 修復を理由に RT 側へ状態判断・動的確保・
    lock・ownership を導入しない（Observer は副作用・ownership を持たない）
```
