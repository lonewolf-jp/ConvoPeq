# M-01 Pre-Audit / Failure Contract Audit (read-only)

- **日付**: 2026-09-15
- **対象**: 2026-09-13 監査系 M-01「L1/L2 pre-IFFT デノーマル guard 欠落」
- **根拠文書**: doc/audit/ConvoPeq_IR_Audio_Path_Audit_2026-09-13.md §M-01 /
  ConvoPeq_Bug_Verification_2026-09-13.md §6（判定: 確認・弱） / ConvoPeq_BugList_and_FixPlan_2026-09-13.md §5 Step 6
- **前例**: doc/work97（M-04）・doc/work98（M-02）。本 work は M-02 実測
  （§8-6: Debug 発火 = L1 非ガード経由）で**再確認された独立 latent DSP hardening** として扱う。
  M-02 の修正拡張ではない（別 commit 境界）。
- **方式**: read-only（production 変更 0 / commit 0）。baseline origin/main `140b2079`
- **基準**: ConvoPeq.md 2026-09-15 23:22:48（FRESH・commit 140b2079 相当と一致検証済みの実ソース連結）

---

## 1. Source Trace — 実コード現状

### 1.1 pre-IFFT guard の所在（非対称性の確定）

NUC の IFFT 入力点は RT パスに 2 箇所のみ:

| 経路 | 関数 | accum→IFFT | guard |
|---|---|---|---|
| **L0（immediate）** | `processLayerBlock`（MKLNonUniformConvolver.cpp:1422–1513） | interleave（:1488）→ **AVX2: `killDenormalV` ループ（:1493–1498）/ scalar: `killDenormal`（:1500–1502）** → `processLayerInv`（:1504） | **あり** |
| **L1/L2（分散）** | `Add`（:1584–1725）完了枝 | interleave（:1702）→ `processLayerInv`（:1708） | **なし** |

L1/L2 は `if (l.nextPart >= l.numPartsIR)` 完了枝でのみ IFFT（:1705–1721）。
tailOutputBuf（:1712）・delayLineWrite（:1717）はそのまま無防備、最終 wet は
M-02 で凍結済みの出力 scrub（Runtime.cpp:787）が唯一の最終防塞。

### 1.2 「同一処理」と呼んでいるものの実体は 3 系統（責務混同の解明 — 本監査中核）

`DspNumericPolicy.h` の既存 primitive は 3 種の意味論を持つ:

| primitive | Release | Debug | NaN | ±Inf | flush 境界 |
|---|---|---|---|---|---|
| `killDenormal`（scalar :189–204） | **no-op**（#if で消える） | 有効 | **保持** | **保持** | 厳密 IEEE subnormal（bit 判定・**閾値なし**） |
| `killDenormalV`（AVX2 :274–287） | **no-op** | 有効 | **ゼロ化**（`_CMP_GE_OQ` 偽→mask 0） | **保持**（GE 真） | \|x\| < `kDenormThresholdDouble = 1.0e-20`（policy・-380 dBFS 級） |
| `isFiniteAndAboveThresholdMask`（NUC cpp:118–129・direct path :1396） | 常に有効 | 有効 | ゼロ化 | **ゼロ化**（`v-v==0` 判定） | 同上 1e-20 |

- L0 の production（AVX2）実効 guard = **killDenormalV 意味論**（policy 閾値 + NaN ゼロ化副作用 + Inf 保持）。
- BugList/監査の「L0 と同じ処理を L1/L2 へ移植」は意味論を特定しておらず、
  **M-02（非有限責任）と M-01（denormal 責任）の境界を跨ぎうる** — §4 決定点。
- **M-02 §8-6/§8-11 との整合**: Debug の T-M02 発火は「NaN が L0 で潰れ L1 で生存する」事実に依存。
  L1/L2 へ `killDenormalV` をそのまま移植すると **M-02 の Debug 発火路径が構造的に消滅**（連係衝突）。
  scalar `killDenormal`（NaN 保持）であれば衝突 0。
- **棚卸し（記録のみ・本 work 対象外）**: 3 系統意味論の不整合自体と、Release/Debug で
  no-op/実動が分岐する primitive 群の docs 統一は hygiene work 候補。

### 1.3 FTZ/DAZ 設定の全経路（production 影響の実測根拠）

- MainApplication.cpp:159–160（起動時）/ MKLRealTimeSetup.cpp:29–30 / ProgressiveUpgradeThread.cpp:78–79 —
  各専用スレッド起動時に `_MM_SET_FLUSH_ZERO_MODE/_DENORMALS_ZERO_MODE` ON。
- RT: ConvolverProcessor::process エントリ `juce::ScopedNoDenormals`（Runtime.cpp:288）、
  AudioEngine RT entries（BlockDouble:101「スレッド起動時に1回だけ設定」・DSPCoreDouble:606 コメント R-2 検証済）。
- **帰結**: production Release では全 FFT/SSE 演算が HW レベル FTZ/DAZ に覆盖され、
  L1/L2 の subnormal は accumBuf 到達前に 0 化される → **guard なしの観測可能影響は Release でゼロ**。
  影響領域は Debug（JUCE_DEBUG/_DEBUG）または CONVOPEQ_DEBUG_DENORMALS のビルドのみ
  （L0 は guard 有効・L1/L2 は無効、という非対称が実体化する領域）。

## 2. 汚染持続性（B 項目: propagation / 持続性）

```
input → inputAccBuf → [prev|cur] → FwdFFT → fdl slot → accum(窓) → [pre-IFFT guard?] → IFFT
  → L0: ringWrite / L1,L2: tailOutputBuf + delayLineWrite → Get → wetOut（M-02 scrub）
```

- **denormal の発生機序**: 長減衰 IR 後尾部・ゲラント下 tiny 値の畳み込み積が
  subnormal 領域へ減衰する経路のみ（入力が normal なら accum 和も normal 域で留まる帯域あり）。
  Debug（FTZ off）では subnormal が accumBuf/tailOutputBuf/ring/delayLine を **乗算加算鎖で
  通常域へ再生されず減衰しながら流れる** → SSE subnormal アシスト遅延（マイクロコード補助）が
  IPP IFFT/加算カーネルに集中 — **性能劣化のみで音声破綻ではない**（値自体は -6140 dBFS 級）。
- **持続性**: denormal は毎ブロック memset 再生成の accum（:1663/:1701）と FDL 窓回転
  （:1658）で受動的に流出。M-02 と違い**状態残留の増幅ループなし**。
- **NaN/Inf との分離**: pre-IFFT guard の責任範囲は **denormal のみ**。NaN/Inf は
  ①エンジン入力エッジ sanitize（V1）・②IR 負荷時検証（V2）・③direct path の
  isFiniteAndAbove…（:1396）・④出力 scrub（M-02）・⑤FFT 障害時 clearFFTOutputOnError で
  既に層構造が成立しており、pre-IFFT guard に責任を持たない（§4 の責務分離条項）。

## 3. RT safety / 予算（C 項目）

- 追加候補の本体は**既存 primitive の同一位置適用のみ**:
  allocation 0 / lock 0 / atomic 0 / log 0 / 分岐回復 0 / 状態遷移 0 / telemetry 接続 0。
  **M-02 telemetry・recovery・HealthMonitor/policy・Publish/Retire/Crossfade/Epoch と接続しない**（禁止条項）。
- **Release**: 意味論をどちらに確定しても Release 影響は**命令増 0**（primitive が #if で
  消える — 呼出文の SIMD ループもスカラーループも最適化により除去される同一書式を L0 が証明済）。
  DeterministicBuildVerifier / semantic-hash 等価 gate で回帰確認（§5 T-M01-4）。
- **Debug**: partStride（= complexSize×2 + pad、L2 最大 32,776 級）の 1 linear pass を
  **完了枝 1 回/レイヤーブロック**のみ。512-block・mult8 幾何で amortize ≒ 128 演算/callback、
  同一ブロックの IPP IFFT（O(N log N)）比で無視可能。**RT 判断主体の追加なし**（純データ衛生）。
- placement 契約: `processLayerInv`（L1/L2 側 :1708）**直前**、対象 `accumBuf[0..partStride)`。
  分散ループ本体（:1682–1702）には手を入れない（毎 callback の interleave には適用しない）。

## 4. M-01 契約凍結案（Implementation GO 時の変更スコープ）

```
G-M01-1  配置: Add() 完了枝 processLayerInv 直前（MKLNonUniformConvolver.cpp:1705–1708 の間）に
         pre-IFFT guard を追加。適用域 l.accumBuf[0, partStride)。bypass/分散本体/accum 再構成なし。
G-M01-2  primitive（決定点 D-M01、§6 参照）:
         (β) 既存 scalar killDenormal のループ適用 — 厳密 IEEE subnormal のみ・閾値新設なし・
             NaN/Inf 責任不関与・M-02 連係衝突なし【監査推奨】
         (α) 既存 killDenormalV の AVX2 ループ適用 — L0（AVX2 実効）と同形・同一構造を最優先する
             が、Debug で NaN ゼロ化副作用 → M-02 T-M02-1 の Debug 発火路径が消滅（衝突明示）
G-M01-3  閾値新設 0: policy 定数（kDenormThreshold*）も新設しない。(β) は判定不要（bit 判定）。
G-M01-4  責務分離: pre-IFFT guard は denormal のみ。NaN/Inf は §2 の既存層（V1/V2/scrub）の
         責任とし、guard 経由の暗黙 NaN 処理を増やさない（=混同の禁止条項）。
G-M01-5  対象外（棚卸し・記録のみ、混入禁止）:
         (a) L0 の AVX2/scalar 枝の意味論差（§1.2）
         (b) direct path isFiniteAndAbove…（Inf まで潰す独自意味論・:1396）
         (c) NonRT 側 DftiComputeBackward（ResampleAndFallback:414/451、MixedPhase:568/854）
         (d) 3 系統意味論の docs/レジストリ統一 hygiene work
禁止: M-02 scrub/counter/reporter 仕様、M-04 gate、SR-01/02/03、H-01/H-02、
     Publish/Retire/Crossfade/Epoch、Coordinator、エンジン境界 sanitize、MAX_* 容量、
     killDenormal/V の定義自体（呼出追加のみ）、新規 CTest target
```

## 5. テスト契約（先行凍結）

| ID | 内容 |
|---|---|
| T-M01-1 | denormal-only 入力（subnormal 級 impulse・例 1e-310）を L1/L2 活性幾何（irLen > l0Len+kL1MaxParts・l1Part で L2 起動、48k で ≒270k sample IR）で drive → 全出力サンプルが「0 または |x| ≥ DBL_MIN」検証（Debug=guard、Release=FTZ/DAZ — 両ビルドで同一主張） |
| T-M01-2 | 通常信号 + L1/L2 活性幾何の正常ブロック出力が対照（同一入力 clean シーケンス）と有限・決定的（subnormal 出現 0）— 既存 40/40 の幾何（4800 IR）は L1 のみ/単層で継続、新規 target 追加なし（同 TU 内） |
| T-M01-3 | **M-02 非退行**: checkM02NonFiniteTelemetry が Debug でも Release でも無変更で PASS（= D-M01 選択の衝突検証ゲート。(α) 選択時は本項目が Fail → M-02 テスト契約の再改定が別途必要） |
| T-M01-4 | Release 等価性: guard 追加の Release バイナリは semantic-hash/DeterministicBuild verifier 群で回帰 PASS（no-op 除去の構造保証） |

## 6. 重大度の再判定（監査所見）

- BugList v1.3「Medium・確認（弱）」→ **Low（latent hygiene — Debug ビルド限定の実効差、
  production は FTZ/DAZ で無影響）** への訂正を求める。根拠 §1.3/§3。
- 導入価値: Debug 開発時の性能一貫性 + 将来の非 FTZ カバレッジ経路（新 RT entry 等）への
  構造防衛。費用は呼出追加のみ（primitive 新設 0・閾値新設 0）。M-04/M-02 と同じ
  「低頻度・病的条件だが、不可視の非対称を可視でなく**解消**する最小費用」型 hardening。
- **ただし混同注意（ユーザー指摘項目の答え）**: 「L0 と同じ処理」の素直な実装 (α) は
  NaN 責任の越境であり、M-02 で凍結した「検出は scrub、 hygiene は denormal」の層を壊す。
  同一性は **位置と責務**で担保する（(β) がその解）。

## 7. 出口判定

```text
Source Trace          完了（guard 所在 §1.1・意味論 3 系統 §1.2・FTZ 全経路 §1.3）
Failure Model         denormal 持続は受動減衰・性能のみ影響・音声破綻なし §2
RT Safety Boundary    呼出追加のみ・Release 命令増 0・Debug amortize ≒128 演算/callback §3
Contract Freeze       §4（primitive 選択 D-M01 のみ未確定・衝突検証 T-M01-3 を凍結）
Severity Reassessment Medium → Low（production 無影響の実証 §1.3/§3）
Implementation Gate   **提示のみ・PENDING** — ユーザー裁定事項:
  D-M01  (β) scalar killDenormal【推奨: 責務純正・M-02 衝突 0・閾値判定なし】
         (α) killDenormalV AVX2【L0 同形、ただし Debug で M-02 発火路径が消滅
             → T-M01-3 Fail を承知して M-02 テスト再改定を同 or 別 commit で行うこと】
  cond-1 重大度 Medium→Low 訂正の承認
  cond-2 §4 G-M01-5 の隣接 (a)〜(d) 混入禁止
  cond-3 M-02 commit 140b2079 のテスト・契約へ触れない（T-M01-3 は非退行 gate）
```

以上。

---

## 8. 実装確定事項（2026-09-16 Implementation — D-M01=β 承認 GO に基づく）

Implementation Gate 承認（D-M01=β / Severity Low 訂正 / T-M01-3 最重要 gate / commit・push は
Final Gate 後まで HOLD）による確定記録。production 変更は **`MKLNonUniformConvolver.cpp` の
呼出追加 1 hunk のみ**（完了枝 processLayerInv 直前・`accumBuf[0,partStride)`・scalar `killDenormal`
同形）。primitive 定義・DspNumericPolicy.h・閾値・M-02/M-04/SR/H 系は変更 0（承認済み変更禁止リスト）。

1. **実測による位置づけの精密化（正直記録）**: RT 実行時は `ConvolverProcessor::process` の
   `ScopedNoDenormals`（Runtime.cpp:288・Debug/Release 共通）と MainApplication/MKLRealTimeSetup の
   per-thread FTZ/DAZ 設定により、**guard 追加前から HW が subnormal 演算結果を flush している**。
   したがって M-01 の gap は通常の RT 経路では観測不能であり、guard は **software 層の
   defense-in-depth（意味論的一貫性・将来の非 FTZ 経路対策）**として導入される。
   §3「Debug のみ実効差」→「実効差は FTZ 無効時のみ（現状の RT 経路では重畳保証）」に読み替え。
   重大度 Low（latent hygiene）は不変・むしろ強化。
2. **契約検証の二層設計（T-M01-1・弱体化禁止対応）**:
   - (a) primitive 層: guard 本体と同一 `#if` 条件で、Release=恒等（命令増 0 の構造保証＝T-M01-4）・
     Debug=subnormal→0 / normal・NaN・Inf・±0 保持（責務分離 G-M01-4 の直接証明）。観察可能で
     vacuous でない。
   - (b) DSP 層: L1 活性幾何（4800 IR/512 block → l1Len=704）へ subnormal 専用入力（1e-310/-1e-311、
     finiteness を満たす値）を注入し、全出力サンプルの**厳密不変式「0 または |x| >= DBL_MIN」**を
     両ビルドで主張（契約回帰ロック）。空証防止に warmup 非零応答アサーション併設。
3. **T-M01-3 強制方法**: runner 配線を checkM02NonFiniteTelemetry → checkM01DenormalHygiene の順に
   固定。Release/Debug ctest の双方で M-02 チェックが PASS しなければsuite が fail するため、
   β の非連係（NaN 保持＝発火路径不変）が gate として成立する。
4. **テスト永続化の教訓反映（work98 §8-9）**: 新規テストの全書式 index 計算は %512、
   exit 経路で releaseResources。subnormal リテラルは前提検証（kSub!=0 && |kSub|<DBL_MIN）で
   ツールチェーンの定数 flush を検出。
5. **単独境界**: M-01 は M-02（140b2079）と**別 commit**（production NUC cpp / テスト /
   ConvoPeq.md / work99 の 4 ファイル）。M-02 既存物への接触 0。

以上を以て M-01 実装は §4 G-M01-1〜5 の範囲に閉じている。

## 9. 検証記録（2026-09-16 Implementation 完了時）

| # | 項目 | 結果 |
|---|---|---|
| ① | T-M01-1 | **PASS**: (a) primitive 契約（Release=恒等・Debug=subnormal→0/normal・NaN・Inf・±0 保持）+ (b) subnormal 注入不変式「0 or \|x\|>=DBL_MIN」両構成 |
| ② | T-M01-2 | **PASS**: L1 活性幾何での warmup 非零応答 + clean 長尺区間の不変式継続 |
| ③ | T-M01-3 (M-02 非退行) | **PASS**: checkM02NonFiniteTelemetry が Release/Debug とも無変更で PASS（runner 配線順で強制。β の NaN 保持＝発火路径不変が成立） |
| ④ | T-M01-4 (Release 等価) | **PASS**: primitive `#if` 恒等（①a で直接検証）+ semantic-hash-drift / deterministic-build / semantic-equivalence verifier 3 連続 PASS（evidence/M01_VERIFIERS_LOG.txt） |
| ⑤ | Release ctest | **40/40 PASS**（REL_CTEST_EXIT=0・フルビルド含む REL/DBG_BUILD_EXIT=0） |
| ⑥ | Debug ctest | **40/40 PASS**（DBG_CTEST_EXIT=0）+ 安定性再走 Release×2/Debug×2 exit 0 |
| ⑦ | clang-tidy（NUC 対象・9 TU） | exit 0。実質警告 1 = bugprone-branch-clone @MKLNonUniformConvolver.cpp:658/686（**SetImpulse tailMode の既存行・M-01 diff 外**・監査台帳 clang_tidy_audit_report.json は同ファイルを既登録＝既存計上） |
| ⑧ | cppcheck（変更 2 ファイル） | 指摘 0 |
| ⑨ | RT safety（raw atomic dot-call scan） | **PASS**（guard は pure data hygiene — atomic/lock/alloc/log 追加 0） |
| ⑩ | ConvoPeq.md | 再生成 2026-09-16 01:00:43 / FRESH・NEWER_SRC_COUNT=0 / guard 記述の埋め込み確認 |
| ⑪ | diff/scope boundary | production hunk = **1 のみ**（Add 完了枝 +9/−0）。テスト = +179/−0（include 1 + checkM01DenormalHygiene + runner 配線）。ConvoPeq.md / doc/work99。git diff --check PASS。PRESERVE 3 件 staged のまま非接触。M-02 commit 140b2079 の全ファイル（Runtime/Lifecycle/ConvolverProcessor.h）接触 0 |

evidence: `evidence/M01_BUILD_TEST_LOG.txt` / `evidence/M01_VERIFIERS_LOG.txt` / `evidence/M01_STATIC_ANALYSIS_LOG.txt`

出口: commit/push は Final Gate 承認後（承認時境界: src/MKLNonUniformConvolver.cpp /
src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp / ConvoPeq.md /
doc/work99/m01_pre_audit_failure_contract_freeze_20260915.md の 4 ファイル・M-02 と独立 commit）。
