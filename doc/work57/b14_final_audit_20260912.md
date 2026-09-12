# B14 最終判定書 — 最終基準化・Read-only Architecture Audit

- **日付**: 2026-09-12
- **性質**: **修復フェーズ終了後の固定・監査・基準化**（B13 コード変更禁止状態で実施 — 本監査中の source 変更 0 件）
- **基準文書**: `doc/Practical Stable ISR Bridge Runtime.md`（不変条件 ISR-RT/WORLD/SNAP/XF/PUB/RET/OBS/HM/VAL/ATM × 35 項目 + §11 review 項目 + 最重要 3 原則）
- **前置き**: B13 報告書 `doc/work57/b13_step4_work_report_20260912.md`（v1.6）の全ゲート完了を受け、B13 を固定した状態で最終基準ソースに対する独立監査を実施
- **HEAD**: 104200ec（監査時点・ソース未 commit）

---

## Step 1 — 最終 `ConvoPeq.md` の同一性確認 → **確定**

### 11:19:21 / 13:07:53 差の解消

| 項目 | 実測 |
|---|---|
| ディスク上 `ConvoPeq.md` の `Generated:` | **`2026-09-12 13:07:53`**（4.97 MB・mtime 13:07） |
| 監査添付側が参照した値 | 11:19:21 — **stale な添付コピー**（ディスク実体には存在しない世代。本監査で 11:19:21 版の実体確認は不能・不要） |
| 判定 | **ディスク実体は 13:07:53 版が唯一の最新生成物。最終基準として固定** |

### スナップショット↔実ソースの機械照合（13:07:53 版の全セクション抽出 vs ディスク）

| ファイル | 照合結果 |
|---|---|
| `src/MKLNonUniformConvolver.cpp` | **IDENTICAL**（1,892 行） |
| `src/MKLNonUniformConvolver.h` | **IDENTICAL**（520 行） |
| `src/tests/NUPCTestAccess.h` | **IDENTICAL**（52 行） |
| `src/tests/MT-NUPC-Measurement.cpp` | **IDENTICAL**（781 行） |

CRLF 正規化後の全行一致（照合スクリプト: ConvoPeq.md の `### 📄` マーカー配下コードフェンスを抽出して diff — 4 ファイルとも差分 0 行）。

### 生成時点のソース固定性

- ソース 4 ファイルの mtime はすべて ConvoPeq.md 生成（13:07:53）以前 → **生成後にソース変更なし**
- `git status`（監査時点）: `M ConvoPeq.md` / `M doc/work57/null_test_procedure.md` / `M src/MKLNonUniformConvolver.cpp`（89 行差分）/ `M src/MKLNonUniformConvolver.h`（24 行）/ `M src/tests/MT-NUPC-Measurement.cpp`（905 行）/ `?? src/tests/NUPCTestAccess.h`（新規）/ `?? doc/work57/*`（7 文書）/ `?? .auto/ .tgrep/`
- `CMakeLists.txt`・`build.bat` 変更なし（B13 はビルド設定に触れていない）

**→ 最終基準 = `ConvoPeq.md` 2026-09-12 13:07:53 版（実ソースと完全同一であることを機械照合で証明済み）として固定。**

---

## B14-A — Policy R 最終 read-only 監査 → **全項目 PASS**

実ソースからの再確認（行番号は現行 `src/MKLNonUniformConvolver.cpp` / `.h`）。

### I4 — Get Clock 順序

```text
cpp:1691   const std::uint64_t t0 = m_outputSamplesProcessed;   ← Get ブロック先頭時刻
cpp:1694   const int got = ringRead(output, numSamples);        ← L0
cpp:1757   delayLineReadAdd(l, output, numSamples, t0, layerGain);  ← L1/L2（t0 使用）
cpp:1762   m_outputSamplesProcessed += static_cast<std::uint64_t>(got);  ← read 後
cpp:1764   return got;
```

順序 **t0 取得 → ringRead → L1/L2 read → `+= got`** を確認。off-by-B 防止の契約（設計 Rev 4 §2.2）どおり。Reset で `cpp:1881 m_outputSamplesProcessed = 0`。

### F1 — 減算前 Phase 0 判定

```text
cpp:1797   if (t0 < static_cast<std::uint64_t>(l.outputDelaySamples)) return;   ← 減算前に符号なし比較
cpp:1801   const std::uint64_t readStart = t0 - static_cast<std::uint64_t>(l.outputDelaySamples);
```

uint64 ラップアラウンド（0−2048 問題）の防御が**減算前**に存在。旧 policy（`maxRead` / `max(R, maxRead)` 自律進行）は実装から完全除去 — `rg "maxRead" src/` は cpp:1784 の廃止コメント 1 件のみ（`maxReaderResidencyUs` は無関係の別機能）。

### F2 — R(t) と delayReadCursor の分離

- 論理リードヘッド R(t) = t − o_L は `readStart`（ローカル計算・I1 Placement 写像）としてのみ存在
- `cpp:1832 l.delayReadCursor = readStart + numSamples` — **observation/telemetry 更新のみ**（read 位置の決定には一切使用しない・旧コメント「唯一のRead Authority」は削除済み — h:388 のコメントも observation/telemetry 定義に修正済み）

### F3 — I2 violation は deterministic safety guard

```text
cpp:1806-1810   if (readStart + numSamples > l.delayWriteCursor) {
                    convo::fetchAddAtomic(m_delayI2ViolationCount, 1, std::memory_order_acq_rel);
                    return;   // fail-closed no-add
                }
```

no-add + diagnostic counter のみで、**policy authority に昇格していない**（B14-C で経路証明）。

### 附加 — I2/I5/I3 構造 gate（cpp:1099-1131 付近・SetImpulse 末尾）

lead ≤ o_L − B / P%B==0 ∧ o_L%B==0 / cap ≥ o_L − lead + 2P を SetImpulse（非 RT 準備パス）で診断ログ + `jassertfalse`（Debug のみ）。outputDelaySamples は測定対象のため assert しない（設計 §7/§8 契約どおり）。

---

## B14-B — Practical Stable ISR Bridge Architecture Audit → **FINDING 0（侵食なし）**

B13 変更ハンクの全列挙（`git diff HEAD` 実測）:

| # | 位置 | 内容 |
|---|---|---|
| 1 | h:312-319 | `friend struct NUPCTestAccess`（unqualified・getter 専用テスト access の開口） |
| 2 | h:382-390 | `delayReadCursor` コメント修正（observation/telemetry 定義） |
| 3 | h:432-446 | `delayLineReadAdd` シグネチャに `t0` 追加 |
| 4 | h:496-503 | `m_outputSamplesProcessed`（plain uint64・RT 内専用）+ `m_delayI2ViolationCount`（atomic）宣言 |
| 5 | cpp:1099-1131 | SetImpulse 末尾の I2/I5/I3 構造 gate（診断ログ + jassertfalse） |
| 6 | cpp:1685-1764 | Get(): t0 取得・呼び出し引数・`+= got` |
| 7 | cpp:1779-1833 | delayLineReadAdd 本体（Policy R 式・F1/F3 guard・二分割 read 維持・telemetry 更新） |
| 8 | cpp:1877-1882 | Reset(): clock/counter リセット |

変更関数は **Get / delayLineReadAdd / SetImpulse / Reset の 4 箇所のみ**。Publication / Crossfade / Retire / RuntimeWorld / Snapshot / Validator 系ファイルは 1 件も変更していない（git status で機械証明）。

### 不変条件チェックリスト（35 項目）

**A. B13 が直接触れる領域 — 行レベル判定**

| 項目 | 不変条件 | 判定 | 証拠 |
|---|---|---|---|
| ISR-RT-001 | RT は状態を決定しない（`if(needCrossfade())` / `if(shouldRetire())` / `if(policy.requiresXXX())` 禁止） | **PASS** | B13 は publish/retire/crossfade/policy 状態への分岐を RT に 1 件も追加していない。`t0 < o_L` / `readStart+n > W` は音響 read 可否（availability）であり状態遷移決定ではない。F3 契約（deterministic safety guard は policy authority ではない）は設計 Rev 4 で確定・20 ラウンド監査承認済み |
| ISR-RT-002 | RT はメモリ所有権を持たない | **PASS** | B13 変更部に所有権取得/移転/解放 0 件。`delayLineBuf` は read のみ |
| ISR-RT-003 | RT は動的確保禁止 | **PASS** | delayLineReadAdd / Get 変更部に new/malloc/alloc 0 件（本体全行確認） |
| ISR-RT-004 | RT はログ出力禁止 | **PASS** | RT read path（Get/delayLineReadAdd）にログ 0 件。`[B13-GATE]` ログは SetImpulse（非 RT 準備パス） |
| ISR-OBS-001 | Observer は副作用禁止（許可: metrics/logging/telemetry。禁止: publish/retire/crossfade 変更） | **PASS** | counter increment と delayReadCursor 更新は telemetry 許可カテゴリのみ。publish/retire/crossfade への接続 0 件（B14-C） |
| ISR-OBS-002 | Observer は所有権禁止 | **PASS** | `NUPCTestAccess` は getter のみ（h:316-319 コメント契約・shared_ptr なし） |
| ISR-ATM-001 | atomic 直接呼び出し禁止 | **PASS** | B13 差分内の直接 `.load()/.store()/.fetch_add()` 0 件。テスト側 raw `.load()` は第19ラウンド指摘で wrapper 化済み |
| ISR-ATM-002 | Wrapper 経由のみ（`consumeAtomic()` / `publishAtomic()` / `fetchAddAtomic()`） | **PASS** | 実使用: `fetchAddAtomic`（cpp:1808）/ `publishAtomic`（cpp:1882）/ `consumeAtomic`（NUPCTestAccess.h:48）— 全部 wrapper |
| ISR-ATM-003 | MemoryOrder 統一（独自指定禁止） | **PASS** | wrapper 既定形のみ（acq_rel / release / acquire）。B13 独自 memory order の新設なし |

**B. B13 が触れない領域 — 変更ファイル列挙による除外証明（24 項目）**

| グループ | 項目 | 判定 | 根拠 |
|---|---|---|---|
| RuntimeWorld | ISR-WORLD-001〜004（Immutable / 完全構築後 Publish / 修正経路なし / 単一所有者） | **PASS（侵食なし）** | B13 差分は RuntimeWorld 系コードに触れない。原則 1（Publish 後完全 Immutable）への接触 0 |
| Snapshot | ISR-SNAP-001〜004（読み取り専用 / 生成元 1 箇所 / 差異なし / 寿命管理なし） | **PASS（侵食なし）** | Snapshot 系コード変更 0 件 |
| Crossfade | ISR-XF-001〜004（判定 1 箇所 / Pure Function / DSP 状態参照なし / Duration は Policy） | **PASS（侵食なし）** | Crossfade 系コード変更 0 件・B13 は判定箇所を増やしていない |
| Publish | ISR-PUB-001〜004（経路 1 本 / Validation 必須 / Validator 空実装禁止 / Rollback 禁止） | **PASS（侵食なし）** | Publication 管控系（PublicationAdmission.cpp 等）変更 0 件・Validator 迂回なし |
| Retire | ISR-RET-001〜004（判定 1 箇所 / retire-delete 分離 / RetireQueue 寿命 / World 破棄 Deferred） | **PASS（侵食なし）** | Retire 系コード変更 0 件・retire 経路を増やしていない |
| HealthMonitor | ISR-HM-001〜003（観測のみ / World 変更禁止 / Crossfade 変更禁止） | **PASS（侵食なし）** | Monitor 系コード変更 0 件 |
| Validation | ISR-VAL-001〜003（失敗時 Publish 禁止 / 決定論的 / 副作用禁止） | **PASS（侵食なし）** | Validator 系コード変更 0 件 |

### §11 Architectural Review 項目（10 項目 — 追加コードが…）

| 項目 | 判定 |
|---|---|
| RT でメモリ確保していないか | **No（していない）** |
| RT で delete していないか | **No** |
| Publish 経路を増やしていないか | **No** |
| Crossfade 判定箇所を増やしていないか | **No** |
| RuntimeWorld を書き換えていないか | **No** |
| Snapshot を書き換えていないか | **No** |
| Observer に副作用がないか | **Yes（ない — telemetry のみ）** |
| retire 経路を増やしていないか | **No** |
| Validator を迂回していないか | **No** |
| atomic 直接呼び出ししていないか | **No（wrapper のみ）** |

### 最重要 3 原則

| 原則 | 判定 | 証拠 |
|---|---|---|
| 原則 1 — RuntimeWorld は Publish 後完全 Immutable | **維持** | B13 は RuntimeWorld に接触なし |
| 原則 2 — RT は実行のみ・判断しない | **維持** | B13 の RT 変更は read 位置計算（引数 t0 由来）と availability guard のみ。publish/retire/crossfade 決定 0 件（ISR-RT-001 参照注記） |
| 原則 3 — Publish/Crossfade/Retire の決定権は 1 箇所 | **維持** | 新経路・迂回 0 件 |

---

## B14-C — B13 診断コードの本番責務からの分離 → **PASS**

`m_delayI2ViolationCount` の**全参照 4 箇所**（`rg` による完全列挙）:

| 箇所 | 役割 | 許可分類 |
|---|---|---|
| h:503 宣言 `std::atomic<std::uint32_t> { 0 }` | 診断 counter（コメント明記: "RT policy decision ではない"） | 宣言 |
| cpp:1808 `fetchAddAtomic(..., 1, acq_rel)` | I2 violation 検出時の telemetry 加算 | ISR-OBS-001 許可（metrics/telemetry） |
| cpp:1882 `publishAtomic(0, release)` | Reset 時の観測値リセット | 観測リセット |
| NUPCTestAccess.h:48 `consumeAtomic(..., acquire)` | テスト Observer の読み取り（getter のみ） | ISR-OBS-001 許可 |

**policy 変更・publish / retire / crossfade への分岐経路: 0 件** — counter を読んで動作を変える production コードは存在しない（`rg` 全文検索で確定）。`m_outputSamplesProcessed` も同様に全参照 5 箇所（宣言・t0 取得・`+= got`・Reset・test getter）で、RT read 位置の入力としてのみ機能し、状態遷移決定には使用しない。

同系メンバとの整合: `m_outputSamplesProcessed` / `delayReadCursor` / `delayWriteCursor` は既存の plain uint64 cursor 群と同一のスレッド模型（RT 内で読み書き、Reset は audio 停止下の準備パスで実行 — 既存契約に従う）であり、**新規の cross-thread 共有を作っていない**。

---

## B14-D — clang-tidy → **TOOLCHAIN BLOCKED / PENDING 固定**

無理に PASS に変更しない（監査指示どおり）。失敗の具体的原因を 2 種確定（実行証拠・2026-09-12）:

```text
error: unknown argument: '-Qstd:c++20' [clang-diagnostic-error]        ← icx 固有 flag を clang が解析不能
error: 'mkl.h' file not found                                          ← oneAPI 環境 include path 依存
（clang-tidy -p build-icx src/tests/NUPCTestAccess.h → "Found compiler error(s)"・2 errors）
```

MSVC build/ の compile_commands.json も失敗（CLANGTIDY_MSVC_EXIT=1 — `.auto/nulltest/p02b_clangtidy.log`）。**対処には icx 用 compile database の生成 or `--driver-mode=cl` + MKL include path 補完が必要 → 別タスク（STATIC-ANALYSIS PENDING）**。この PENDING と B13 Functional PASS は混同しない。

---

## B14-E — 変更禁止状態での最終 regression → **PASS（監査中 source 変更 0 件）**

ログ: `.auto/nulltest/b14_regression.log`（2026-09-12 実測）

| Gate | 結果 |
|---|---|
| Build up-to-date | `ninja: no work to do` + `B14_BUILD_EXIT=0` — **監査時点の exe が現行ソースと一致**（変更禁止状態の証明を兼ねる） |
| D4 smoke | `B14_SMOKE_EXIT=0`・structural failures = 0 |
| Primary oPE | **MODEL-MATCH × 22 run**（MODEL-DIFF 0 件） |
| effDelay | **1984（L1 × 18）/ 34752（L2 × 4）全 const**（= o_L − B・expected 一致） |
| coverage(Phase 1) | **coveragePhase1 = 1.000000 × 22 run・missing = 0 / dup = 0 × 22** |
| i2Violations | **0 × 22 run** |
| CTest | **`100% tests passed out of 40`・`B14_CTEST_EXIT=0`** |

T3〜T7 L1（1984）と T7 L2（34752）の再確認を含む。**結果は v1.6 報告時と完全同一 — 監査中の変動なし。**

Ring wrap の精緻化（F2 訂正 2026-09-12）: write wrap（cursor 周回 L1 39.61 / L2 2.63 周）は直接証明、**wrap-crossing read（二分割 read 発火）は T7 run3 実測 0 件**（L1 1,602 / L2 1,090 reads・capacity が B=64 の倍数のため構造的に不発）。split boundary のサンプル値直接照合は現行テストでは試験不能 — 値連続性は M1 波形 null（−309.86 dB）による間接証拠。split 意図発生は次サイクル候補（コード変更を要する）。

---

## B14 最終判定

```text
┌──────────────────────────────────────────────────┐
│ B13 Functional Repair（Policy R）                │
│   PASS                                           │
│   — oPE == 0 ×22・effDelay = o_L − B const       │
│     ・M1 T1〜T7 PASS・M3 L1 diff=0               │
├──────────────────────────────────────────────────┤
│ B13 Validation / Regression                      │
│   PASS                                           │
│   — CTest 40/40・D4 smoke 全 gate・coverage 1.0  │
│     ・i2Violations = 0・Ring wrap 定量           │
│     （L1 39.6 周 / L2 2.63 周含有）              │
├──────────────────────────────────────────────────┤
│ Architecture Invariant Audit（B14-A〜C）         │
│   PASS — FINDING 0                               │
│   — ISR 35 項目・§11 10 項目・最重要 3 原則      │
│     全 PASS・B13 による境界侵食なし              │
├──────────────────────────────────────────────────┤
│ clang-tidy                                       │
│   TOOLCHAIN BLOCKED / PENDING                    │
│   — icx flag + MKL include の非互換（証拠固定）  │
│   — B13 Functional PASS とは分離                 │
└──────────────────────────────────────────────────┘
```

**総合: B14 監査完了 — FINDING 0。**

- **最終基準**: `ConvoPeq.md` 2026-09-12 13:07:53 版（実ソースと機械照合で完全同一を証明）を固定。11:19:21 差は stale 添付と解消済み
- **B13 のコードは固定**（監査中 source 変更 0 件・build up-to-date で証明）
- **commit は本監査終了後のユーザー判断**（対象: 本番 2 ファイル + テスト 2 ファイル + doc 8 文書 + スナップショット）
- **別管理タスク**: clang-tidy の icx compile_commands 対応（STATIC-ANALYSIS PENDING）

### 附属資料

| 資料 | 位置 |
|---|---|
| B14-E regression ログ | `.auto/nulltest/b14_regression.log` |
| ConvoPeq.md↔ソース機械照合（本監査で実施） | 4 ファイル IDENTICAL（抽出 diff 0 行） |
| clang-tidy 失敗証拠 | 本書 B14-D（icx `-Qstd:c++20` + `mkl.h`）+ `.auto/nulltest/p02b_clangtidy.log` |
| ISR 不変条件原典 | `doc/Practical Stable ISR Bridge Runtime.md`（不変条件 §1〜§10・§11 review 項目・最重要 3 原則） |
| B13 報告書（v1.6） | `doc/work57/b13_step4_work_report_20260912.md` |
| B13 設計書（Rev 4） | `doc/work57/b13_repair_design_20260912.md` |
