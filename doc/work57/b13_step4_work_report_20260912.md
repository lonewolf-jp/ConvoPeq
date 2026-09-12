# B13 修復作業報告書（Step 4 — Policy R 実装から D4 実測・残ゲート完了まで）

- **日付**: 2026-09-12
- **作業範囲**: Step 4（B13 Repair 実装 + H1/H3 + ビルド/CTest 検証 + D4 実測）+ 報告書 v1.6 更新（第15〜18ラウンド監査対応 + 残ゲート全完了）
- **設計基準**: `doc/work57/b13_repair_design_20260912.md`（Rev 4 — 設計審査 APPROVED / Step 4 GO）
- **関連文書**: `doc/work57/null_test_procedure_v2.md`（v2.9 手順書）、`doc/work57/null_test_step3_results_20260912.md`（v1.2 — Step 3 実測/B13 FAILURE CONFIRMED）、`doc/work57/content_mapping_audit.md`（Step 0 監査記録）
- **総合状態（v1.6 — 残ゲート全完了）**:

```
B13 Functional Repair（Policy R）: PASS 確定
  — D4 実測: 全 case・全 run で oPE == 0（Primary gate 合格）
  — M1 波形 Null Test: T1〜T7 全 PASS（−309〜−311 dB）
  — M3 L1: M2 と完全一致（diff = 0）
  — i2Violations = 0（deterministic safety guard 発動なし）
  — 旧 policy dependency（maxRead / max(R, maxRead)）: 完全除去

Step 4 残ゲート: 全完了（§7 実施結果）
  — P0-1 coverage Phase 1 基準化: 完了（missing=0 / dup=0 / coveragePhase1=1.000000 全 run）
  — P0-2 cppcheck C++ モード: syntaxError 0（clang-tidy はツールチェーン課題で PENDING 継続）
  — P0-3 Full Release build（-j1）: PASS（ConvoPeq.exe 48.2MB 生成）
  — P0-4 CTest 40/40: PASS（100%・44.59 sec）
  — P1-1 Ring wrap stress: 定量記録（T7 run3 write wrap L1 39.61 周 / L2 2.63 周・read split 0 件実測・oPE=0 維持）
  — P1-2 D4 再確認: 完了（v16 smoke・oPE=0 ×22・effDelay 1984/34752 回復）
  — ConvoPeq.md 再生成: 2026-09-12 13:07:53 版を最終基準に昇格
  — commit: ユーザー判断
```

---

## 1. 作業の背景と目的

Step 3 実測（`null_test_step3_results_20260912.md` v1.2）で **B13 遅延補償の実効不成立**が確定した:

- 旧 read policy（`actualReadStart = max(delayReadCursor, maxRead)`）の下で、L1/L2 の `outputPlacementError` が **oPE = −1152〜−1600（L1）/ −30720（L2）の定常定数**（先行ズレ）を示した。
- 原因は「delayLine 論理位置」と「ストリーム時刻」の意味論不整合 — content は `p + IR_offset` に置かれるべきなのに、旧 policy は「論理位置 p」を「時刻 p」に対応づけて読む。

Step 4 の目的は、設計審査 APPROVED の **Policy R（stream-time fixed-offset read: `readStart = t0 − o_L`）** を実装し、**D4 実測で `oPE == 0`（全 case・全 run）を検収**すること。Architectural 制約（RT に lock/確保/ownership/publish-retire decision の新設なし）を遵守する。

## 2. 実施状況サマリ（Step 4-A〜4-I）

| Step | 内容 | 状態 |
|---|---|---|
| **4-A** | `m_outputSamplesProcessed`（I4 stream clock）+ `m_delayI2ViolationCount`（atomic counter）導入 | **完了** |
| **4-B** | `delayLineReadAdd` を Policy R 式に変更（F1 契約: 減算前 Phase 0 判定 / F3 fail-closed no-add / `maxRead`・`max(R, …)` 廃止 / `delayReadCursor` を observation に格下げ） | **完了** |
| **4-A** | `Get()` で I4 順序を厳守（`t0` 取得 → L1/L2 read → `+= got`）、`Reset()` で clock/counter リセット | **完了** |
| **4-C** | SetImpulse 末尾に I2/I5/I3 構造不変条件 gate 診断ログ（`[B13-GATE]`） | **完了** |
| **4-D** | ハーネス更新（H1 `effConst[3]` layer 別化 / H3 T8 EXCLUDED / `observeCursor` Policy R 式 / CSV 列更新） | **完了** |
| **4-E** | ビルド: `MTNUPCMeasurement` **成功**（error 0）／本体は SoakPublish icx OOM → **`-j1` 再試行 PENDING** | 部分完了 |
| **4-F** | CTest Release: **39/40**（MTNUPC のみ Failed — coverage 集計 Phase 1 対応前の構造的失敗フラグ） | 部分完了 |
| **4-G/H** | D4 実測（T4/T6/T7 L2）+ 判定 | **完了**（oPE = 0 全 case） |
| **4-I** | ConvoPeq.md 再生成 | **PENDING**（コード変更停止後） |

## 3. 実装詳細（本番コード変更一覧）

| ファイル | 変更内容 |
|---|---|
| `src/MKLNonUniformConvolver.h` | ① `m_outputSamplesProcessed`（uint64 stream clock、Reset で 0、Get の got 分ずつ加算 — **I4 Get Clock: L1/L2 read より前に `+= got` しない**）② `m_delayI2ViolationCount`（atomic uint32 — **deterministic safety guard の diagnostic counter、RT policy decision ではない**）③ `delayLineReadAdd` シグネチャに `std::uint64_t t0` 追加 ④ `delayReadCursor` のコメントを observation/telemetry 定義に修正（第17ラウンド軽微指摘） |
| `src/MKLNonUniformConvolver.cpp` | ① `delayLineReadAdd` を Policy R 式に変更: **Phase 0 判定は減算前**（F1）、**`maxRead`/`max(R, maxRead)` 廃止**、読み出し位置 `readStart = t0 − o_L`、I2 違反時は fail-closed no-add + diagnostic counter（F3）、**既存の二分割リング読み出し（wrap 対応）は維持**、`delayReadCursor` は telemetry 更新のみ ② `Get()` で `t0 = m_outputSamplesProcessed` 取得（ringRead 前）→ L1/L2 read → `m_outputSamplesProcessed += got`（**I4 順序厳守**）③ `Reset()` で clock/counter リセット ④ `SetImpulse` 末尾に **I2/I5/I3 構造不変条件 gate**（`[B13-GATE]` ログ + jassertfalse — outputDelaySamples は測定対象のため assert しない） |
| `src/tests/NUPCTestAccess.h` | accessor 追加: `outputSamplesProcessed` / `delayI2ViolationCount`（**`convo::consumeAtomic` wrapper 経由** — 第17ラウンド監査の raw `.load()` 指摘を統一）。読み取りのみ（Observer 原則） |
| `src/tests/MT-NUPC-Measurement.cpp` | **H1**（`effConst[3]` layer 別化）/ **H3**（T8 M1 `EXCLUDED (filterSpec mismatch)`）/ `observeCursor` を Policy R 式に（**PHASE0 / POLICY-R / UNAVAILABLE**）/ CSV 列更新（`t0`/`readStart`/`phase0`/`i2ViolationCount` 追加、`maxRead`/`forcedSkip` 廃止）/ cases 表を修復後期待値に更新 |
| `CMakeLists.txt` | 変更なし |

## 4. ビルド検証（4-E）

| 項目 | 結果 |
|---|---|
| `MTNUPCMeasurement`（icx Release、Policy R 実装込み） | **成功**（error 0） |
| 本体（ConvoPeq 全ターゲット、icx Release `-j2`） | **SoakPublishIntegrationTests.cpp で icx OOM（LLVM ERROR: out of memory）** — 環境リソース枯渇と整合するが、**再ビルド未完了のため本体 Release build gate は未確定**（変更非起因の断定は再現試験まで保留）。`-j 1` で再試行が必要 |
| CTest（Release・40 件） | **39/40（32.78 秒）** — 失敗は MTNUPCMeasurement のみ（coverage 集計 Phase 1 対応前の構造的失敗フラグ、§6）。他の 39 テストは本変更の影響なし |

## 5. D4 実測結果（4-G/H — 修復効果の検収）

### 5.1 M2 Primary gate（`outputPlacementError == 0`）

| Case | 構成 | read-anchor（Phase 1） | **oPE 実測** | 期待値 | 照合 | 判定帯 |
|---|---|---|---|---|---|---|
| T3 L1（IR=2049） | 1/1 | 21 | **0** | 0 | **MODEL-MATCH** | **PASS(B13-ALIGNED)** |
| T4 L1（IR=5000） | 6/1 | 32+ | **0** | 0 | **MODEL-MATCH** | **PASS** |
| T5 L1（IR=8000） | 12/2 | — | **0** | 0 | **MODEL-MATCH** | **PASS** |
| T6 L1（IR=12000） | 20/3 | — | **0** | 0 | **MODEL-MATCH** | **PASS** |
| T7 L1（IR=40000） | 64/8 | — | **0** | 0 | **MODEL-MATCH** | **PASS** |
| T7 L2（P=4096） | 2/1 | 21 | **0** | 0 | **MODEL-MATCH** | **PASS** |

全 case・全 run・全 Phase-1 anchor で **oPE = 0**（1 値に収束、分布なし — 監査側独立再計算でも一致）。

### 5.2 M1 波形 Null Test（gain 反映 reference、align=0）

| Case | RMS [dB] | ゲート |
|---|---|---|
| T1（IR=2000） | −311.39 | **PASS** |
| T2（IR=2048） | −311.37 | **PASS** |
| T3（IR=2049） | −311.38 | **PASS** |
| T4（IR=5000） | **−310.03** | **PASS** |
| T5（IR=8000） | −309.92 | **PASS** |
| T6（IR=12000） | −309.88 | **PASS** |
| T7（IR=40000） | −309.86 | **PASS** |
| T8（bypass） | −10.08 | **EXCLUDED**（filterSpec mismatch — truncation・無音は topology で PASS） |

Step 3（修復前）の T3〜T7 FAIL（−23〜+2 dB）から **−309〜−311 dB へ回復**。M2（位置）と M1（波形）が独立に同じ結論を示す。

### 5.3 M3 位置同定（補助証拠）

| Case | k\*(B) | n0 + IR_offset | diff | 照合 |
|---|---|---|---|---|
| T3〜T7 L1 | `n0 + 2048`（全 case） | `n0 + 2048` | **0** | **M2 と完全一致** |
| T7 L2 | 2330 | 34816 | −32486 | 相関限界（減衰 IR の自己相関 — M2 主判定を採用） |

### 5.4 仮説検証（修復後）

| 仮説 | 実測 | 判定 |
|---|---|---|
| **I1 Placement**: `t_output(j) = jP + o_L`（全 j） | 全 anchor で oPE = 0 | **成立** |
| **I2 Availability**: lead ≤ o_L − B の下で欠落なし | i2Violations = 0、Phase 1 anchor 正常 | **成立** |
| **I4 Get Clock**: t0 = ブロック先頭、+= got は read 後 | 全 anchor で oPE = 0（off-by-B なし） | **成立** |
| **I5 Alignment**: P % B == 0 ∧ o_L % B == 0 → ∀j ∃! anchor | 各 j に 1 anchor | **成立** |
| **I6 Phase Coverage**: Phase 0 は no contribution（reference 整合） | readMode first = PHASE0、silent = o_L/B（32 / 544 callback） | **成立** |
| **Phase 1 で expected read-anchor が全て存在** | coverage（Phase 1 anchor）= 発生ブロック数と一致 | **成立** |

### 5.5 修復前後の対比（因果系列の確定）

```
旧 Policy（max(R, maxRead) 自律進行）
  ↓  oPE = −1152〜−1600（L1 先行）/ −30720（L2 先行）
  ↓  M1: T3〜T7 FAIL（−23〜+2 dB）
Policy R（readStart = t0 − o_L、F1 減算前判定、F3 fail-closed）
  ↓  oPE = 0（全 case・全 run・全 anchor）
  ↓  M1: T1〜T7 全 PASS（−309〜−311 dB）
  ↓  M3 L1: diff = 0（全 case）
```

## 6. Architectural 制約の遵守確認（第16〜18ラウンド監査）

| 制約 | 状態 |
|---|---|
| RT に lock / 動的確保 / ownership / publish-retire decision の新設なし | ✓（追加は uint64 clock + atomic counter のみ） |
| `m_outputSamplesProcessed` は RT-local stream clock（単調、Reset で 0、got 分加算） | ✓ |
| `m_delayI2ViolationCount` は **deterministic safety guard の diagnostic counter**（**RT policy decision ではない** — F3 契約） | ✓ |
| `delayReadCursor` は observation/telemetry（**policy cursor 依存の完全除去** — 第16ラウンド監査要求） | ✓ |
| `NUPCTestAccess` は読み取りのみ（Observer 原則: metrics/logging/telemetry 許可、publish/retire/crossfade 変更禁止） | ✓ |
| `delayI2ViolationCount` accessor は **`convo::consumeAtomic` wrapper 経由**（raw `.load()` 廃止 — 第17ラウンド監査の raw atomic 指摘を統一） | ✓ |
| Authority 構造（Publish/Retire/RCU）への変更なし | ✓ |

## 7. 残ゲート実施結果（v1.6 — 全ゲート完了）

第17/18ラウンド監査で確定した実施順序（P0-1 → P0-2 → P0-3 → P0-4 → P1-1 → P1-2 → ConvoPeq.md 再生成）に従い、**全ゲート実施・完了**。

### P0-1 — coverage 集計を Phase 1 基準へ修正（ハーネス）→ **完了**

判定式の緩和ではなく、I6 契約を壊さない形で Phase 1 のみを厳密判定する形に修正した:

```
Phase 0（t < o_L）: read contribution = 0（意図された no-add）— coverageOk に含めない
Phase 1（t ≥ o_L）: expected = run 時間内に content_time(j) が到達する論理位置のみ
coverageOk = (expected > 0 ∧ missing == 0 ∧ duplicate == 0)
```

**expected の限定が本修正の核心**。初回実装では `expected = numBlocks`（全論理位置）としていたため、run 時間外の論理位置が missing 3 として誤計上された（T3）。expected を「`jP + o_L ≤ totalStream` のみ」に限定した結果、**T3: expected 24 → 21、missing 3 → 0** となり、全 case で **missing = 0 / dup = 0 / coveragePhase1 = 1.000000**。

実装は anchorCbs map（j → callback list）による集計に一本化し、write 逐次照合ループは H 診断（writeEventCb 記録）として維持（coverageOk 判定からは除外 — 二重定義の排除）。

### P0-2 — cppcheck / clang-tidy（B13 変更範囲 4 ファイル）→ **完了（cppcheck）/ PENDING（clang-tidy）**

**cppcheck（C++ モード）** — 第19ラウンド指摘（C モードで C++ を解析していた問題）を解消:

```
コマンド: cppcheck --language=c++ --std=c++17 --enable=warning,style,performance
          --inline-suppr --suppress=missingIncludeSystem --suppress=unusedFunction
          --error-exitcode=2 -I src src\tests\NUPCTestAccess.h src\tests\MT-NUPC-Measurement.cpp
結果: syntaxError = 0（C++ モードでの解析成功）
```

残警告の評価:
- **style 系 6 件**（FFTBackend.h / DspNumericPolicy.h / AtomicAccess.h）— **Step 4 変更外の既存コード**、スコープ外として記録
- **uninitMemberVarNoCtor 7 件**（MT-NUPC-Measurement.cpp `Case` struct）— 集約初期化で全メンバ指定のため実害なし。ただし**デフォルト初期化子を追加して消去**（§7 P1-2 対応の v1.6 ビルドに反映）
- **uninitvar（`evs`）1 件** — **false positive**。`std::vector<Ev> evs[3]` はデフォルト構築（336 行）で初期化は不要

**clang-tidy** — icx compile_commands.json 非互換（`/Qstd:c++20` 等の icx 固有引数を clang が解析不能）。MSVC build/ でも同様。**ツールチェーン課題として STATIC-ANALYSIS PENDING を継続**（次サイクル技術課題）。

### P0-3 — Full Release build（`-j1`）→ **PASS**

第19ラウンドで icx OOM が発生した SoakPublishIntegrationTests.cpp を含む本体全体を `-j1` で再ビルド:

```
cmake --build build-icx --config Release -- -j 1 → V16B_BUILD_EXIT=0
ConvoPeq_artefacts/Release/ConvoPeq.exe 生成確認（48,272,896 bytes・Sep 12 11:59）
```

OOM は **環境リソース枯渇（並列コンパイル時のメモリ圧迫）が妥当な説明**であり、変更コード起因の証拠は無いことを確認（ログ: `.auto/nulltest/v16b_build_ctest.log`）。

### P0-4 — CTest 40/40 → **PASS**

coverage 修正・Case struct 修正後の最終コードで Release CTest を再実行:

```
100% tests passed out of 40（Total Test time = 44.59 sec・CTEST_EXIT=0）
Test #39 MTNUPCMeasurement ... Passed（0.47 sec）
Test #40 AudioEngineHarness ... Passed（33.01 sec）
```

B13 Primary gate の同時維持は P1-2（D4 再確認・v16 smoke）で確認 — **oPE == 0 ∧ i2Violations == 0 ∧ Phase-1 coverage == 1.0** の 3 条件成立。

### P1-1 — Ring wrap stress → **完了（定量記録）**

wrap を意図的に作成する新 case の追加は不要と判断 — **既存 T7 run3（n0=16384、irLen=40000）が実質的に wrap stress を含有**していることを CSV 実測で定量化:

```
T7 run3（totalStream ≈ 103,936 samples）:
  L1: 論理 write cursor 最大 103,936 / capacity 2,624  → 39.6 周 wrap
  L2: 論理 write cursor 最大 102,400 / capacity 38,976 → 2.63 周 wrap
  → wrap 存在下で oPE = 0・effDelay const（1984/34752）・coveragePhase1 = 1.000000・i2Violations = 0
```

L1 の 39.61 周 / L2 の 2.63 周は **write cursor の周回（write wrap 発生）** を直接証明する。
**【F2 最終監査で訂正（2026-09-12）】**: wrap-crossing read（readStart % cap + B が cap を跨ぎ
二分割 read が発火するケース）は CSV 実測で **0 件**（L1 1,602 reads / L2 1,090 reads とも単一
セグメント） — capacity 2624 / 38976 がともに B=64 の倍数のため `readOffset + B ≤ cap` が常に成立
する構造。したがって「first segment + wrapped second segment のサンプル値連続性」は T7 run3 では
**試験不能**で、値連続性の証明は M1 波形 null test（全波形 −309.86 dB・write wrap 領域を含む）に
よる**間接証拠のみ**。split を意図的に発生させる検証は test case 追加（コード変更）を要するため
STOP CODE CHANGES の下では実施せず、次サイクル候補として記録。実装の二分割 read は防御コードと
して維持（現行 48kHz/64 構成では不発）。

### P1-2 — D4 再確認（v16 smoke）→ **完了**

Case struct 修正込みの再ビルド後、smoke を再実行（ログ: `.auto/nulltest/v16_build_smoke.log`）:

```
SMOKE_EXIT = 0・structural failures = 0
M2 全 22 run: oPE = 0（MODEL-MATCH）・effDelay = 1984（L1×18）/ 34752（L2×4）・const
coverage: 全 layer 全 run で missing = 0 / dup = 0 / coveragePhase1 = 1.000000
i2Violations = 0（全 run）
M1: T1〜T7 PASS（−309.86〜−311.39 dB）・T8 EXCLUDED (filterSpec mismatch)
M3: L1 diff = 0（expected oPE = 0 と整合）
```

**effDelay 表示退行の修復も本 run で確認** — coverage 修正時に anchorCbs ループへ書き換えた際に消失していた effVal 更新を、read 実行 callback 全体から `t − R_after` を計算する専用ループに分離して復元（1984/34752 を expected どおり回復）。

### 最後 — ConvoPeq.md 再生成 → **完了（13:07:53 版）**

コード変更停止後に再生成:

```
python output_sourcecode_markdown.py → Generated: 2026-09-12 13:07:53
マーカー確認: Policy R 関連 7 件（m_outputSamplesProcessed・新 delayLineReadAdd シグネチャ）
             NUPCTestAccess 27 件 / Phase 1 基準ハーネス（coveragePhase1）5 件
```

**本 13:07:53 版を Step 4 実装込みの最終基準に昇格**。commit はユーザー判断。

## 8. 監査対応履歴（第13〜18ラウンド）

| ラウンド | 判定 | 主要指摘と対応 |
|---|---|---|
| 第13ラウンド | D1〜D4 **CONDITIONALLY APPROVE** | **I4 Get Clock**（t0 使用・+= got・off-by-B 防止）**I5 Alignment**（P%B==0 ∧ o_L%B==0 → ∀j ∃! c）I2 は現行 scheduler 前提の構成 gate・I3 overwrite inequality・coverage は Phase 1 のみ・R6 primary gate は **oPE==0**（**effDelay==0 は誤り — Policy R では o_L−B**）→ **Rev 2 で反映** |
| 第14ラウンド | Rev 2 **APPROVE / Step 4 GO**（I3 1 点訂正条件付き） | **I3 訂正**: write は B でなく **P サンプル**（`delayLineWrite(..., partSize)`）→ `cap ≥ o_L − lead + 2P`（T3 余裕 0・T7 L2 余裕 64 — gate で毎回確認）。I4 に got==numSamples 前提、I6 を numSamples==B contract に限定、Step 4-A〜4-I 順序固定 → **Rev 3 で反映** |
| 第15ラウンド | Rev 3 **CONDITIONAL APPROVE** | **F1 — unsigned underflow**: uint64 では `readStart = t0 − o_L` の `readStart < 0` が**永遠に成立しない**（t0=0, o_L=2048 で 18446744073709549568 を実証）→ **Phase 0 判定を減算前**に契約固定。**F2 — I3 の R(t) を論理リードヘッド `R(t) = t − o_L` と定義**し delayReadCursor（observation）と分離。**F3 — I2 violation の no-add は deterministic safety guard**。基準版再生成（11:20:30 版と 20:16:16 版の相違解消）→ **Rev 4 で反映** |
| 第16ラウンド | Rev 4 **APPROVED / Step 4 GO** | I1〜I6/F1/F2/F3/Architectural 全 PASS。**Step 4 gate 表現の厳格化**（CTest PASS 扱い禁止・icx OOM 断定保留・P0-3 として ConvoPeq.exe 生成確認）。Ring wrap は「コード構造 PASS / stress 未実施」 |
| 第17ラウンド | **v1.3 検証: 妥当** | coverage の原因説明訂正（**write > read 差分 3〜7 件、Phase 0 起因** — 「writeEventCount=0」は誤記）を確認。`rebuild_ctest.log` が Step 3 旧ログである点も確認 |
| 第18ラウンド | **v1.4 検証: 妥当**（軽微指摘 2 件の対応確認・再実測で oPE=0 維持） | **残ゲート P0-1〜P1-2 の実施順序確定**（本報告書 §7）。「B13 機能修復は確定 PASS、Step 4 全ゲートは未完了」の状態維持 |

## 9. 附属資料

| 資料 | 位置 |
|---|---|
| **B14 最終判定書（固定・監査・基準化 — FINDING 0）** | `doc/work57/b14_final_audit_20260912.md`（B14-A〜E・ISR 35 項目・4 分類判定） |
| D4 smoke run stdout（Policy R 修復後・v1.6 最終） | `.auto/nulltest/v16_build_smoke.log`（Case struct 修正込み再ビルド+smoke）/ `.auto/nulltest/v14_build_smoke.log`（v1.4 時点） |
| M2 CSV（Policy R・22 run） | `.auto/nulltest/nupc_v29_csv/M2_*.csv`（Ring wrap 定量の抽出元） |
| cppcheck C++ モードログ | `.auto/nulltest/v15_build_smoke_cppcheck.log` |
| 残ゲート ビルド/CTest ログ | `.auto/nulltest/v16b_build_ctest.log`（Release build -j1 exit 0 + CTest 40/40）／`step4_build.log`（初回・本体 OOM）／`step4_rebuild2.log`（-j2・CTest 39/40） |
| 設計書 | `doc/work57/b13_repair_design_20260912.md`（Rev 4） |
| Step 3 実測報告書（修復前） | `doc/work57/null_test_step3_results_20260912.md`（v1.2） |
| Step 4 実測報告書（本作業の詳細実測版） | `doc/work57/b13_step4_results_20260912.md`（v1.3） |
| 手順書 | `doc/work57/null_test_procedure_v2.md`（v2.9） |
| Step 0 監査記録 | `doc/work57/content_mapping_audit.md` |
| 実行バイナリ | `build-icx/MTNUPCMeasurement_artefacts/Release/MTNUPCMeasurement.exe`・`build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe`（v1.6 ビルド後） |

---

## 10. 結論

1. **Step 4 実装（4-A〜4-D）を完了**: Policy R（stream-time fixed-offset read）を本番コードに実装（本番変更は `MKLNonUniformConvolver.h/.cpp` の clock・counter・read 式に限定、**Authority 構造変更なし**）。
2. **D4 実測で修復効果を検収**: 全 case・全 run・全 Phase-1 anchor で **oPE = 0**、M1 波形 Null Test も T1〜T7 全 PASS（−309〜−311 dB）。**「B13 は sample-accurate stream-time placement を実現していなかった（Step 3 確定）」という問題を Policy R で修復し、M2 + M1 で実測確認した。**
3. **残ゲートを全完了**（§7 実施結果）: coverage Phase 1 基準化（missing=0 / dup=0 / 1.000000 全 run）→ cppcheck C++ モード（syntaxError 0）→ Full Release build `-j1` PASS → **CTest 40/40 PASS** → Ring wrap stress 定量記録（L1 39.6 周 / L2 2.63 周含有で oPE=0 維持）→ D4 再確認（oPE=0 ×22・effDelay 1984/34752 回復）→ ConvoPeq.md 再生成（13:07:53 版・最終基準に昇格）。
4. **現段階の宣言**: 「**B13 の機能修復は確定 PASS。Step 4 全ゲートは完了**」— clang-tidy のみ STATIC-ANALYSIS PENDING（ツールチェーン課題・次サイクル）。残る行為は **commit（ユーザー判断）** のみ。
