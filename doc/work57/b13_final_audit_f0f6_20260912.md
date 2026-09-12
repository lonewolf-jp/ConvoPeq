# B13 Final Audit Phase 判定書（F0〜F6 — Baseline Freeze 〜 Commit Readiness）

- **日付**: 2026-09-12
- **性質**: **コード変更禁止状態での最終監査**（本監査中の source 変更 0 件・文書更新のみ）
- **前提文書**: `b13_step4_work_report_20260912.md`（v1.6）/ `b14_final_audit_20260912.md`（B14 FINDING 0）
- **HEAD**: `104200ec3b41b1cce0a25934e3785b24806e3ac7`（監査時点・未 commit）

---

## 前提 — 13:07:53 版の基準確保（監査側参照差の解消）

監査側が参照できた `ConvoPeq.md` は **2026-09-12 02:20:04 UTC（= 11:20:04 JST）版** — これは
13:07:53 版の**前の世代**（11:19〜11:20 JST 生成・v1.5 前後）であり、最終基準ではない。

**こちら側での基準確保の実施**:

| 項目 | 実測値（受領ファイルの照合に使用可） |
|---|---|
| `Generated:` スタンプ | **`2026-09-12 13:07:53`** |
| SHA256（ConvoPeq.md） | `f6f4182a9a8df8cd212af3d680cd3931257b8dc64405a6763d703fa6038715fb` |
| SHA256（MKLNonUniformConvolver.h） | `3ae21280db3cb4e2ef5514597ee09af30be5e5a8038b1fe9deb1d05978be09e2` |
| SHA256（MKLNonUniformConvolver.cpp） | `e28f5fde71a78c7b2d1feb40c85b33192c62cc9fb143d72e632b002e3cf1697a` |
| SHA256（NUPCTestAccess.h） | `d3404e359ae30ed63586254ece095872233c23a364990ee28f3059f167261071` |
| SHA256（MT-NUPC-Measurement.cpp） | `efbc880355d1cd65f70265040e792464e63d2d312930d1669cf6465400b86fa8` |

**スナップショット↔実ソース機械照合（B14 Step 1 実施済み）**: 4 ソース（cpp 1,892 / h 520 /
NUPCTestAccess 52 / harness 781 行）を ConvoPeq.md 13:07:53 版から抽出して比較 → **すべて IDENTICAL
（差分 0 行）**。**監査環境へのアップロード/登録はユーザー操作** — 上記 SHA256 で受領ファイルの
同一性を検証可能。ディスク上の 13:07:53 版を正式基準として固定した。

---

## F0 — Baseline Freeze → **完了**

コード変更禁止で開始（監査中の source 変更 0 件・`ninja: no work to do` で exe↔source 一致を並行証明）。

```text
git status（監査時点）:
  M ConvoPeq.md / M doc/work57/null_test_procedure.md / M doc/work69/remaining_bugs.md（閉包更新）
  M doc/work57/convolver_timing_verification_report.md（閉包更新）
  M src/MKLNonUniformConvolver.cpp（89 行）/ M src/MKLNonUniformConvolver.h（24 行）
  M src/tests/MT-NUPC-Measurement.cpp（905 行）
  ?? src/tests/NUPCTestAccess.h（新規）/ ?? doc/work57/*（8 文書）/ ?? .auto/ / ?? .tgrep/
  ?? doc/work68/automatic_sound_test_plan_v7.4_validation_20260911.md（B13 スコープ外・work68 検証文書）

git diff --stat: 5 files changed, 1687 insertions(+), 398 deletions(-)（ConvoPeq.md 更新含む）
git diff --check: exit 0（whitespace エラー 0 件）
HEAD: 104200ec3b41b1cce0a25934e3785b24806e3ac7
ConvoPeq.md generation timestamp: 2026-09-12 13:07:53
CMakeLists.txt / build.bat: 変更なし（git status に含まれず — B13 はビルド設定に触れていない）
B13 変更対象 4 ファイル: 報告書どおり（h / cpp / NUPCTestAccess.h / MT-NUPC-Measurement.cpp）を確認
```

**Baseline 固定**: `ConvoPeq.md = 13:07:53 final` / `B13 report = v1.6` / `Design = Rev 4 APPROVED`。

---

## F1 — Source ↔ Report 最終照合 → **全項目 PASS**

**ConvoPeq.md 13:07:53 版に対する機械照合**（行番号は md 内の行）:

| 照合項目 | 要求 | 実測（ConvoPeq.md） | 判定 |
|---|---|---|---|
| Policy R — t0 | `t0 = m_outputSamplesProcessed` | md:18577 `const std::uint64_t t0 = m_outputSamplesProcessed;` | **PASS** |
| Policy R — Phase 0 | `t0 < o_L` を**減算前に**判定 | md:18683 `if (t0 < static_cast<std::uint64_t>(l.outputDelaySamples))` → md:18687 `readStart = t0 - static_cast<...>(l.outputDelaySamples)`（判定→減算の順） | **PASS** |
| I4 — 順序 | t0 取得 → L1/L2 read → `+= got`。**`+= got` が read より前に存在しない** | md:18577（t0）→ md:18643（`delayLineReadAdd(l, output, numSamples, t0, ...)`）→ md:18648（`m_outputSamplesProcessed += static_cast<std::uint64_t>(got);`）。`+=` の実装行は md:18648 の 1 箇所のみ（md:18576 は契約コメント） | **PASS** |
| 旧 Policy 完全除去 | `maxRead` / `max(R, maxRead)` が機能経路に残存しない | md 内 `maxRead` 出現は md:18670（廃止コメント）のみ。他は `maxReaderResidencyUs`（無関係の別機能・md:41620/64468） | **PASS** |
| Cursor 分離 | `delayReadCursor` = observation/telemetry。**Policy R の read 決定に cursor を使用しない** | read 位置は `readStart = t0 − o_L`（t0 のみから計算）。cursor は md:18717-18718（telemetry 更新 `readStart + numSamples`）・md:17181（Reset）・md:19171/19196（Layer 宣言）・md:97147（test getter）のみ。read 分岐での使用 0 件 | **PASS** |
| Atomic | `m_delayI2ViolationCount` は `convo::consumeAtomic` 経由。raw `.load()` 等なし | NUPCTestAccess.h 内 `.load(/.store(/fetch_add(` = **0 件**（:46 は wrapper 原則のコメント）・:48 `convo::consumeAtomic(..., acquire)`。ISR-ATM-001/002 整合 | **PASS** |

前段（B14 Step 1）の**全セクション機械照合 IDENTICAL** と併せ、**Source ↔ Report Reconciliation = PASS**。

---

## F2 — Gate Evidence Reconciliation → **全 gate 1対1 照合 PASS（wrap 記述を 1 件訂正）**

報告書数値 vs ログ/CSV の照合（`b14_regression.log` / `v16_build_smoke.log` / `v16b_build_ctest.log` / CSV）:

| Gate | 最終判定 | 証拠（1対1） |
|---|---|---|
| B13 Functional Repair | **PASS** | v1.6 §7 / B14-A〜E / F1 本書 |
| M2 oPE | **0 × 全 22 run** | `b14_regression.log`: MODEL-MATCH 22 件・MODEL-DIFF 0 件 |
| M1 T1〜T7 | **PASS** | −309.86〜−311.39 dB（T8 は EXCLUDED・filterSpec mismatch） |
| M3 L1 | **diff = 0** | 全 case `diff=0 (expected oPE=0)` |
| i2Violations | **0** | × 22 run（uniq 計測） |
| Phase-1 coverage | **1.000000 × 22** | missing = 0 / dup = 0 × 22（uniq 計測） |
| Release `-j1` | **exit 0** | `V16B_BUILD_EXIT=0`・ConvoPeq.exe 48,272,896 bytes 生成。B14-E で `ninja: no work to do`（exe↔source 一致） |
| CTest | **40/40** | `100% tests passed out of 40` × 2 回（v16b・B14-E） |
| Ring wrap | **T7 run3** | 下記（発生証明と値証明を分離） |
| D4 再確認 | **PASS** | B14-E smoke — v1.6 報告時と完全同一・変動なし |

### Ring wrap — 発生証明と値証明の分離（監査強化 + 1 件の訂正）

**(a) wrap の発生 — 直接証明（cursor 数学・CSV 実測）**:

```text
T7 run3: L1 論理 write cursor 最大 103,936 / cap 2,624 → 39.61 周
         L2 論理 write cursor 最大 102,400 / cap 38,976 → 2.63 周
```

**(b) read split（二分割 read 発火）— F2 で新規計測: 0 件**:

```text
L1: reads = 1,602 / wrapCrossing(readOffset + B > cap) = 0
L2: reads = 1,090 / wrapCrossing = 0
```

capacity 2624 / 38976 がともに B=64 の倍数のため `readOffset + B ≤ cap` が常に成立 —
**現行 48kHz/64 テスト構成では二分割 read は構造的に不発**。

**(c) サンプル値連続性 — 間接証拠のみ（訂正）**: 前回報告の「二分割 read ケースが多数含まれ、
first + wrapped second segment の連続性を間接検証」は**誤り** — (b) のとおり split 発火 0 件のため
T7 run3 では試験不能。値連続性の証明は **M1 波形 null test（全波形 −309.86 dB・write wrap 領域を
含む）による間接証拠のみ**。split 境界サンプルの直接照合は CSV にサンプル値が記録されないため
既存データからは不能で、**意図的 split 発生は test case 追加（コード変更）を要する** —
STOP CODE CHANGES の下では実施せず**次サイクル候補**として記録。
v1.6 報告書 §7 P1-1 と B14 判定書 B14-E に訂正注記を反映済み。

---

## F3 — clang-tidy の扱い → **TOOLCHAIN BLOCKED / PENDING 固定（PASS へ変更しない）**

```text
STATIC ANALYSIS
  cppcheck   : PASS（C++ モード・syntaxError 0）
  clang-tidy : PENDING / TOOLCHAIN BLOCKED
               — error: unknown argument: '-Qstd:c++20'（icx 固有 flag を clang が解析不能）
               — error: 'mkl.h' file not found（oneAPI include path 依存）
```

- clang-tidy を通すための本番コード変更は**禁止**（実施していない）
- 次サイクルに **clang / clang-cl 互換 compile_commands を別途構築して read-only 再試行**（別タスク）
- **「clang-tidy PASS」と偽って記録していない** — B13 Functional PASS とは明確に分離

---

## F4 — Practical Stable ISR Bridge Architectural Audit → **PASS（FINDING 0・新設計変更なし）**

RT 境界（B13 変更部分について再確認 — 詳細は B14-B の 35 項目監査）:

```text
RT
 ├─ lock なし             → delayLineReadAdd / Get 変更部に lock 0 件
 ├─ allocation なし       → new/malloc 0 件（本体全行確認）
 ├─ delete なし           → 0 件
 ├─ ownership 変更なし    → delayLineBuf は read のみ・所有権操作 0 件
 ├─ publish decision なし → publish への分岐・接続 0 件
 └─ retire decision なし  → retire への分岐・接続 0 件
```

pipeline 責務分離（Build → Validate → Publish → Crossfade → Retire → Epoch → Delete）: B13 の変更
関数は **Get / delayLineReadAdd / SetImpulse（gate）/ Reset の 4 箇所のみ**で、上記 pipeline の
authority 系ファイルには 1 件も接触していない（git status 機械証明・B14-B で ISR 35 項目 +
§11 10 項目 + 最重要 3 原則 全 PASS）。**本監査で新しい設計変更は開始していない**。

---

## F5 — Final Audit Verdict

```text
B13 Functional Repair        PASS
D4 Primary Gate              PASS   （oPE == 0 × 22 run）
M1                           PASS   （T1〜T7 −309〜−311 dB / T8 EXCLUDED）
M3                           PASS   （L1 diff = 0）
I2/I4/I5/I6                  PASS   （i2Violations=0・I4 順序・I5 gate・coverage 1.0）
Release Build                PASS   （-j1 exit 0・ConvoPeq.exe 生成）
CTest 40/40                  PASS   （2 回実行とも 100%）
Ring-wrap stress             PASS   （write wrap 直接証明・read split 0 件実測・値連続性は間接証拠 — F2 訂正済み）
cppcheck                     PASS
clang-tidy                   PENDING（TOOLCHAIN BLOCKED）
Architectural Audit          PASS   （B14-B FINDING 0）
Source/Report Reconciliation PASS   （F1・機械照合 IDENTICAL 含む）
```

**最終状態の正式表現**:

> **「Step 4 全ゲート PASS」ではなく、**
> **「Step 4 の実装・機能・Release・CTest・実測ゲートは PASS。clang-tidy のみ toolchain pending」**

---

## F6 — Commit Readiness → **READY（commit 実行はしない — ユーザー判断待ち）**

commit 直前チェック（本監査で実施済み — 再実行推奨は `git diff --check` / `git status` / `git diff --stat`）:

```text
git diff --check → exit 0（whitespace エラー 0）
```

**commit 対象**:

| 区分 | ファイル |
|---|---|
| 本番 | `src/MKLNonUniformConvolver.h` / `src/MKLNonUniformConvolver.cpp` |
| テスト | `src/tests/NUPCTestAccess.h`（新規）/ `src/tests/MT-NUPC-Measurement.cpp` |
| 文書（work57） | `b13_repair_design_20260912.md` / `b13_step4_results_20260912.md` / `b13_step4_work_report_20260912.md` / `b14_final_audit_20260912.md` / `b13_final_audit_f0f6_20260912.md`（本書）/ `content_mapping_audit.md` / `null_test_procedure_v2.md` / `null_test_step3_results_20260912.md` / `null_test_procedure.md`（v1 注記更新）/ `convolver_timing_verification_report.md`（閉包更新） |
| 文書（work69） | `remaining_bugs.md`（RB-05 閉包更新） |
| 基準 | `ConvoPeq.md`（13:07:53 版） |

**commit しないもの**: `.auto/`（ログ・CSV 証跡）/ `.tgrep/` / ビルド成果物

**判断待ち事項（ユーザー棚卸し）**: `doc/work68/automatic_sound_test_plan_v7.4_validation_20260911.md`
（未追跡・work68 計画書検証の成果物 — **B13 commit スコープ外**。B13 と分離して commit するか
同時に含めるかはユーザー判断）

**推奨 commit message**（レビュー②案ベース）:

```text
fix(nupc): align delay reads to output stream time

Replace max(R, maxRead) autonomous read with readStart = t0 - o_L.
Add I4 stream clock, F1 uint64 underflow guard, F3 I2 safety counter.
D4: oPE==0 all cases; M1 T1-T7 pass (-309..-311 dB); CTest 40/40.
Docs: B13/B14/final-audit reports, RB-05 & timing-report closure.
```

**本監査では commit を実行していない**（レビュー①推奨どおり・ユーザー判断で実施）。

---

## 附属資料

| 資料 | 位置 |
|---|---|
| F0〜F2 証跡 | 本書（git status/diff --check/HEAD/SHA256・md 行番号・CSV 集計） |
| B14 判定書（ISR 35 項目等） | `doc/work57/b14_final_audit_20260912.md` |
| B13 報告書 v1.6 | `doc/work57/b13_step4_work_report_20260912.md` |
| regression ログ | `.auto/nulltest/b14_regression.log` |
| clang-tidy 証拠 | F3（`-Qstd:c++20` / `mkl.h`）+ `.auto/nulltest/p02b_clangtidy.log` |
| 閉包更新文書 | `doc/work69/remaining_bugs.md`（RB-05 ✅）/ `doc/work57/convolver_timing_verification_report.md`（Aランク 3 項目 ✅） |

---

## CP 追補 — Commit Preparation 結果（2026-09-12・CP-1〜CP-6 完了 / CP-7 承認待ち）

### CP-1 — Commit スコープ分離 → **確定**

**除外（B13 スコープ外）**:

```text
doc/work68/automatic_sound_test_plan_v7.4_validation_20260911.md   ← work68 計画書検証（別 work item）
.auto/                                                              ← ログ・CSV 証跡
.tgrep/                                                             ← ツールキャッシュ
（ビルド成果物 build-icx/ も対象外）
```

### CP-2 — B13 Commit Manifest（因果系列基準で固定）

| 区分 | ファイル | 因果系列の根拠 |
|---|---|---|
| [Production] | `src/MKLNonUniformConvolver.h`（24 行）/ `src/MKLNonUniformConvolver.cpp`（89 行） | B13 Policy R 本体実装 |
| [Test] | `src/tests/NUPCTestAccess.h`（新規 52 行）/ `src/tests/MT-NUPC-Measurement.cpp`（905 行） | D4 実測ハーネス + getter 専用 Observer（B13 検収の証跡装置） |
| [Documentation] | work57 新規 6 文書: `b13_repair_design` / `b13_step4_results` / `b13_step4_work_report` / `b14_final_audit` / `b13_final_audit_f0f6` / `null_test_procedure_v2` | B13 設計→実装→検収→監査の因果系列本体 |
| [Documentation] | work57 新規 2 文書: `content_mapping_audit`（Step 0）/ `null_test_step3_results`（Step 3 修復前実測） | B13 FAILURE CONFIRMED の根拠（修復の因果起点） |
| [Documentation] | `null_test_procedure.md`（v1・3 行追記） | **因果系列必要** — v1 が未実施で `align=64` が B13 実装後無効である旨の参照注記。v1 単独では B13 閉包チェーンの誤用を招く |
| [Documentation] | `convolver_timing_verification_report.md`（29 行） | **因果系列必要** — Aランク「Gardner 未解決」3 項目を B13 修復 + D4 実測で閉包する記録（閉包対象の旧基準文書） |
| [Documentation] | `doc/work69/remaining_bugs.md`（19 行） | **因果系列必要** — RB-05（capacity 式/I3）を B13 監査実測で閉包する記録 |
| [Documentation] | `ConvoPeq.md`（13:07:53 版） | 最終基準スナップショット（実ソースと機械照合 IDENTICAL） |

### CP-3 — diff の最終意味監査 → **スコープ逸脱 0 件**

`git diff --check` exit 0。チェックリスト（コード品質ではなくスコープ逸脱の観点）:

```text
Authority 変更なし        ✓ publish/retire/crossfade/validator 系ファイル未変更（git status 機械証明）
CMake 変更なし            ✓ git status に不在
Build 設定変更なし        ✓ build.bat 未変更
RT lock 追加なし          ✓ diff 内 lock/mutex 0 件（B14-B ハンク全列挙）
RT allocation 追加なし    ✓ diff 内 new/malloc 0 件
RT ownership 変更なし     ✓ 所有権操作 0 件（delayLineBuf は read のみ）
Publish 経路追加なし      ✓ publishAtomic は既存 m_ready + counter リセットのみ・新 publish 経路 0 件
Retire 経路追加なし       ✓ 0 件
```

テスト差分（追加 720 行）のスコープ逸脱パターン走査（publish/retire/crossfade/authority/fetch_add/.load(/.store(/new/malloc/delete）: **一致 0 件** — テストハーネスは production state を変更しない getter 路のみ。
ヘッダ diff 4 ハンク: friend 開口 / `delayReadCursor` コメント（旧「唯一のRead Authority」廃止）/ `delayLineReadAdd` シグネチャ / 2 メンバ宣言のみ。cpp diff 8 ハンク: B14-B 表のとおり（SetImpulse gate / Get t0 / 引数 / `+= got` / read 本体 / I2 guard / telemetry / Reset）。

### CP-4 — clang-tidy PENDING 固定 → **維持（偽 PASS 記録 0 件）**

3 判定書の `clang-tidy` × `PASS` 行走査: 偽 PASS 記録 **0 件**（すべて「PASS へ変更しない」「偽っていない」「toolchain pending」の文脈）。正式状態は `cppcheck PASS / clang-tidy PENDING (TOOLCHAIN BLOCKED)` のまま commit 後も維持。ソース改変による通過は**次サイクルの独立タスク**。

### CP-5 — Ring-wrap 最終表現固定 → **確定**

```text
write wrap:                          proven
read split:                          0 occurrences in current test topology
split-segment value continuity:      not directly proven
```

`39.61 周` は wrap（write cursor 周回）の証明であり **read split の実行証明ではない** — 本訂正で監査上の過大主張は解消済み。**今の B13 をさらに修正する理由にはならない**。split 自体の将来検証は **P1-x: explicit read-split test topology**（本番コードではなくテストトポロジー設計から）として別タスク化。

### CP-6 — Commit 前の最終実行 → **F6 = READY 維持**

```text
git diff --check: exit 0
git status: 変更 7 + 未追跡（manifest どおり・work68/.auto/.tgrep は除外対象）
git diff --stat: 7 files changed, 1730 insertions(+), 403 deletions(-)
commit 対象 = B13 source 2 + B13 test 2 + B13 docs + ConvoPeq.md のみを確認
```

### CP-7 — Commit → **ユーザー承認待ち（実行はしていない）**

推奨メッセージ（確定済み）:

```text
fix(nupc): align delay reads to output stream time

Replace max(R, maxRead) autonomous read with readStart = t0 - o_L.
Add I4 stream clock, F1 uint64 underflow guard, F3 I2 safety counter.
D4: oPE==0 all cases; M1 T1-T7 pass (-309..-311 dB); CTest 40/40.
Docs: B13/B14/final-audit reports, RB-05 & timing-report closure.
```

commit 完了後は **B13 CLOSED** とし、次サイクルは B13 に戻らず以下の優先順位で進める:

1. **clang-tidy toolchain closure**（最優先・B13 と分離）— clang/clang-cl と icx の compile database 整備 + `mkl.h` include path 含め read-only static analysis を成立
2. **P1-x: explicit read-split test topology** — `readOffset + B > capacity` を意図的に成立させるテスト設計（本番コードではなくテストトポロジー設計から）
3. **次の残存バグ監査** — `remaining_bugs.md` OPEN のみ抽出: **P1 = RB-01（`pendingIntentCount()` fallback 不計上・ISRRetire.cpp）/ RB-11（`setProcessingOrder()` sendChangeMessage 欠落・Parameters.cpp）**、P2 = RB-02、P3 = RB-07 / RB-03。次 P0/P1 の選定はこの OPEN 群から
