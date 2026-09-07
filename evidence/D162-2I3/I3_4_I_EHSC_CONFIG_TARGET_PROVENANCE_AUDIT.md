# D162-2-I3-4-I — `/EHsc` Configuration / Target Provenance Audit

```text
Date:        2026-09-06
Type:        read-only audit (I0-I10) — 変更 0 (production/test/CMake/build.bat/source/tools すべて 0)
Baseline:    I3-4-H5 GO (RWDI /utf-8 fix, CMakeLists +10 行, commit 9cacee1f+dirty)
判定:        **I1-I7 全証明 → Case A (NO DEFECT)。/EHsc の修正実装は不要。
             I8 = NO-CHANGE 契約、I9 = 実装対象なし。総合 = I3-4-I GO。**
```

---

## 0. 要旨

> **`/EHsc` は Release / Debug / RelWithDebInfo の全 CXX 492 edge に存在する（492/492/492）。
> RWDI の `set(CMAKE_CXX_FLAGS_RELWITHDEBINFO ...)` 文に `/EHsc` が無いことは edge に影響しない。
> 理由: base `CMAKE_CXX_FLAGS = /DWIN32 /D_WINDOWS /EHsc` が configuration-independent に
> 全 CXX edge へ `/EHsc` を供給しており、これは CMake 4.4.3 の MSVC platform default
> (`Modules/Platform/Windows-MSVC.cmake`: `_FLAGS_CXX "${_GR} /EHsc"`、CMP0117 NEW → `_GR=""`、
> CMP0092 NEW → `_W3=""`) である。H5 が書いた「/EHsc は RWDI にも欠落している」(L1532) は
> set() 文字列レベルでは正しいが edge レベルでは誤り。
> JUCE C1189 ガード (`juce_BasicNativeHeaders.h:92`, `_CPPUNWIND`) は fixture で実証したが、
> 全 CXX edge に /EHsc があるため現在の構成では到達不能（リスク不活性）。
> C flags への /EHsc 追加も不要（C に例外モデルは無い・fixture Case D で確認）。**

---

## I0 — Baseline freeze

```text
git status            CMakeLists.txt M (+10/-0 = H5 のみ) / それ以外 untracked evidence・doc のみ
git HEAD              9cacee1f (+dirty: H5 RWDI /utf-8 block)
CMakeLists.txt        sha16 = 5bdc2a2a... (freeze: evidence/D162-2I3/i3_4_i0_freeze/CMakeLists.txt)
impl-*.ninja          build/CMakeFiles/{impl-Release,impl-Debug,impl-RelWithDebInfo}.ninja
                      mtime 2026-09-06 07:08 (H5 修正後の reconfigure 产物 — freeze.json 参照)
rules.ninja           build/CMakeFiles/rules.ninja (rules 内 /EHsc 定義 0 — flags は edge 変数経由)
.build_identity       build-diag/CMakeFiles/.build_identity (stamp: cl 19.51 / Ninja Multi-Config /
                      source_revision 9cacee1f+dirty / configuration_family Debug;Release;RelWithDebInfo)
CMakeCache.txt        CMAKE_CXX_FLAGS:STRING=/DWIN32 /D_WINDOWS /EHsc  ★ base に /EHsc あり
                      CMAKE_C_FLAGS:STRING=/DWIN32 /D_WINDOWS (base に /EHsc なし — C は別系統)
                      cache の *_RELEASE/_DEBUG/_RELWITHDEBINFO は CMake 既定値のまま
                      (CMakeLists の非キャッシュ set() が generate 時に上書きするため)
ConvoPeq.md           Generated: 2026-09-05 12:02:18 — H5 編集 (09-06 00:00) より古い = stale。
                      本監査は CMakeLists.txt 生ファイルを基準に実施 (棚卸し: §付記 2)
I3-4-H5 変更範囲      CMakeLists.txt L1526-1535 の +10 行のみ (RWDI /utf-8 set × 2 + コメント 6 行)
```

対象 build tree: `build\` (build.bat `BUILD_ROOT=build`、msvc mode → `-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`)。
`build-icx\` (IntelLLVM, 2026-08-26 snapshot) は副系統として比較に使用。

---

## I1 — `/EHsc` source provenance census

CMakeLists.txt 全 47 行 (grep -c `/EHsc` = 47) を分類。完全表は `i3_4_i1_occurrence_table.txt`。

| 分類 | 件数 | 内容 |
| --- | ---: | --- |
| A. CMAKE_CXX_FLAGS_* (set) | 3 | L1519 Release (MSVC block) / L1524 Debug (MSVC block) / L1606 Release (IntelLLVM block)。**RWDI set (L1534) には /EHsc なし** |
| B. CMAKE_C_FLAGS_* (set) | 3 | L1520 / L1525 / L1607。RWDI set (L1535) にはなし |
| C. target_compile_options | 33 | L322…951 群 (31 target) + L1100 MTNUPCMeasurement (if(MSVC) → cl/icx 両方) + L1506 ConvoPeq + L1886 AudioEngineHarness |
| D. generator expression | 0 | — |
| E. IntelLLVM 専用 target options | 1 | L1575 ConvoPeq (elseif IntelLLVM block 内) |
| F. コメントのみ | 5 | L920 / L1517 / L1518 / L1532 (H5 追加・本監査が訂正) / L1605 |
| G. その他 | 2 | L1807 / L1816 = CLANG_TIDY_CMD `--extra-arg=/EHsc` (clang-tidy 解析用, compile edge ではない) |
| 合計 | 47 | — |

CMakeLists 以外: `JUCE/extras/Build/CMake/JUCEHelperTargets.cmake:123` が `juce_recommended_config_flags`
INTERFACE target に `/EHsc` を供給するが、**project CMakeLists は `juce_recommended_config_flags` を
0 回しか参照しない** → 本構成では不活性（どの edge にも寄与しない）。
rules.ninja / build.bat に `/EHsc` 定義は 0。

### base CMAKE_CXX_FLAGS の出所（本監査の核心）

`/EHsc` が全 config 全 edge に届く仕組みは 3 方法で裏取りした:

1. **4 tree の cache 一致**: build / build-diag (09-06 新規 configure) / build-ci-check (08-08) /
   build-asan-msvc の CMakeCache がすべて `CMAKE_CXX_FLAGS:STRING=/DWIN32 /D_WINDOWS /EHsc`、
   `CMAKE_C_FLAGS:STRING=/DWIN32 /D_WINDOWS`。stale cache 残留ではなく再現可能な既定値。
2. **CMake モジュールソース**: `C:\Program Files\CMake\share\cmake-4.4\Modules\Platform\Windows-MSVC.cmake`
   L259 `set(_FLAGS_CXX "${_GR} /EHsc")` → L538 `string(APPEND CMAKE_${lang}_FLAGS_INIT " ... /D_WINDOWS${_W3}${_FLAGS_${lang}}")`。
   CMP0117 NEW → `_GR=""` (/GR は cl の既定 ON のため不要)、CMP0092 NEW → `_W3=""`。
   → MSVC desktop branch の既定 INIT は正確に `/DWIN32 /D_WINDOWS /EHsc`。
3. **fresh configure fixture (Case E)**: %TEMP% に project flag を一切書かない最小 CMake project を
   configure → cache が `CMAKE_CXX_FLAGS:STRING=/DWIN32 /D_WINDOWS /EHsc` を再現。

---

## I2 — 3 configuration × 全 CXX edge census

`build/CMakeFiles/impl-{Release,Debug,RelWithDebInfo}.ninja` の全 compile edge を解析
(方法 = H6 と同じ ninja statement 単位 FLAGS スキャン。raw data: `i3_4_i2_raw_census.json`、
表: `i3_4_i2_edge_census.txt`)。

| Config | CXX edges | /EHsc あり | /EHsc なし | 対象 target 数 | /utf-8 (H6 突合) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Release | 492 | **492** | **0** | 40 | 492/492 ✓ |
| Debug | 492 | **492** | **0** | 40 | 492/492 ✓ |
| RelWithDebInfo | 492 | **492** | **0** | 40 | 492/492 ✓ (H5 修正後と一致) |

**/EHsc 無し CXX edge = 0 件（3 config 合計 0）。列挙すべき欠落 edge は存在しない。**

補助データ:

- **C edges (16/config)**: Release 16/16、Debug 16/16、RWDI **15/16**。
  RWDI で欠落する 1 edge は `PublicationAdmissionTests` の `juce_graphics_Sheenbidi.c.obj`
  (C 言語)。C には例外モデルが無いため意味的影響は 0 (I6 Case D で確認)。
  他 15 C edge に /EHsc があるのは target_compile_options が C 言語にも効くため（冗長・無害）。
- **重複**: Release/Debug の edge は base + config-level set で `/EHsc` ×2 を持つ
  (例 Release D8_1: `/DWIN32 /D_WINDOWS /EHsc /Zm400 ... /utf-8 /EHsc`)。
  target-level /EHsc を持つ 33 target は最大 ×3。MSVC は同一 option の重複を受理
  (H5-H8 の clean rebuild rc=0 が実証)。RWDI edge は ×1 (base のみ)。
- **独立照合**: `build/compile_commands.json` (492×3 CXX) — /EHsc 無し CXX = **0 件**。
- **IntelLLVM (build-icx, 08-26 snapshot)**: 3 config とも CXX 492/492 に /EHsc あり
  (同一 CMake 既定 base + L1575/L1100 target options)。icx でも /EHsc coherence は成立。

## I3 — target provenance

40 CXX target の内訳（full table: `i3_4_i2_edge_census.txt`）:

- **33 target**: target-level /EHsc あり (L322-951 群・ConvoPeq・MTNUPCMeasurement・AudioEngineHarness)。
- **7 target**: target-level /EHsc なし — **base flag のみで供給**:
  `D8_1_WrapperCacheTests / DeferredDeletionQueueReclaimTests / EQAnalysisUnitTests /
  EQBoundExcessBenchmark / EQProcessorMaxGainTests / GainStagingContractTests / PublicationAdmissionTests`
  (RWDI edge FLAGS 実測: `/DWIN32 /D_WINDOWS /EHsc /Zi /O2 /Ob1 /DNDEBUG /utf-8 -std:c++20 -MD`
  — /EHsc の位置が base flag 群に一致)。
- H5 で /utf-8 修復した 7 target との関係: 6 target は /EHsc も target-level に持たないが
  base flag で補完済み。**FFTBackendTests は target-level /EHsc を単独で持っていた** (L626) —
  H4 の指摘どおり「/utf-8 が付いた ≠ /EHsc も付いた」の非対称が実在したが、
  欠落側 (/utf-8) のみが H5 で修復され、/EHsc 側は最初から欠落していなかった。
- JUCE 経由: JUCE module source は project target 内に compile されるため
  その target の flags に従う (= /EHsc あり)。juce_recommended_config_flags 経由の供給は 0 (I1)。

## I4 — 根因判定

**Case A — NO DEFECT**（Release / Debug / RWDI 全 CXX edge が /EHsc あり）。

- `CMAKE_CXX_FLAGS_RELWITHDEBINFO` への `/EHsc` 追加は**不要**（指示 Case A のとおり）。
- C flags への `/EHsc` 追加も**不要**（Case D: C に例外モデルなし・fixture Case D 実証）。
- H4 の「RWDI にも /EHsc が欠落」という前提は **set() 文字列レベルでは真、edge レベルでは偽**。
  H 監査自身の census (`i3_4_h_h1_flag_provenance.txt` L24-25) が H5 修正前の RWDI edge
  (x483 + x9) すべてに `/DWIN32 /D_WINDOWS /EHsc` を記録しており、本監査の census と整合。
  base flag は H5 前から全 config 全 edge に /EHsc を供給していた。
- 傍証: H5 の RWDI broken-build エラー集合 (C1083/C2143/C4430/C2888/C2065/C2061/C1075/C4819)
  に **C1189 が含まれない** — /EHsc が欠落していれば Case C1 のとおり即座に C1189 が出たはずで、
  出ていないことが「欠落していなかった」の実行時証拠になる。

## I5 — JUCE C1189 実証（因果チェーン）

```text
JUCE/modules/juce_core/native/juce_BasicNativeHeaders.h:92
  #if JUCE_MSVC → #ifndef _CPPUNWIND → #error "You're compiling without exceptions enabled! ..."
juce_CompilerSupport.h (MSVC branch): #if ! _CPPUNWIND → #define JUCE_EXCEPTIONS_DISABLED 1
        ↓ 対象 target = JUCE module を compile する全 project target
        ↓ compile command = impl-*.ninja edge FLAGS (base /EHsc により _CPPUNWIND 定義済み)
        ↓ /EHsc 有無 = 全 CXX edge にあり (I2: 492/492/492)
        ↓ 実際の JUCE compilation = 現行全 build log に C1189 0 件 (evidence 全検索:
          C1189 を含むのは CMakeLists コメントと本監査 fixture ログのみ)
```

fixture (I6 Case C1/C2) が同一ガード機構 (`_MSC_VER && !_CPPUNWIND → #error`) を検証:
/EHsc 無し → fatal error C1189 再現、/EHsc あり → 通過。**ガードは real だが、
現在の構成では全 edge に /EHsc があるため到達不能**。
コメント L1517-1518 の因果 (「/EHsc 欠落 → JUCE C1189」) は機構として正しいが、
「Release だけ欠落していた」という記述は base flag の存在と整合しない
（歴史的経緯は commit c8ca439b = work88 Phase 7 に由来。過去の失敗再現は本監査では行わない）。

## I6 — adversarial fixture (%TEMP%\ehsc_i34i、cl 19.51 / VS18 Enterprise)

| Case | 構成 | 結果 |
| --- | --- | --- |
| A | C++ throw/catch + `/EHsc` | compile rc=0、実行 `CAUGHT: boom` rc=42 — **PASS** |
| B | 同じく `/EHsc` 無し | compile rc=0 だが **warning C4530**（「アンワインド セマンティクスは有効にはなりません。/EHsc を指定してください」）— compiler が unwind を保証しない状態を明示 |
| C1 | JUCE-style guard (`_MSC_VER && !_CPPUNWIND → #error`) `/EHsc` 無し | **fatal error C1189**（期待どおり #error 発火 — JUCE ガードと同一機構） |
| C2 | 同 guard + `/EHsc` | rc=0、実行 rc=7 — ガード通過 |
| D | `.c` + `/EHsc` | rc=0・実行正常 — **C 言語に /EHsc は無意味（受理される no-op）** |
| E | %TEMP% に fresh CMake (Ninja Multi-Config + cl, project flag 記述 0) | cache `CMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc` 再現 → RWDI build rc=0。**base /EHsc が CMake 既定であることの決定証拠** |

Case B/C の結果は ConvoPeq の修正必要性の判断には使用していない（指示どおり）。
判断は I2/I3 の edge census と I4 の CMake semantics で行った。
log: `i3_4_i6_fixture_log.txt`。

## I7 — `/utf-8` と `/EHsc` の因果分離

- **/utf-8**: base `CMAKE_CXX_FLAGS` に**存在しない**。Release/Debug は config-level set (L1519/1524)、
  32 target は target-level で補完。RWDI はどちらにも無く H5 まで 9 edge / 7 target が欠落 —
  H5 の修正 (config-level set 追加) はこの欠落のみを修復した。H5 evidence が証明するのはこの分。
- **/EHsc**: base `CMAKE_CXX_FLAGS` に**最初から存在**（CMake MSVC 既定）。config-level set の
  有無に関係なく全 config 全 CXX edge に届く。H5 前後で edge レベルの /EHsc は 492/492 で不変
  (H 監査 census L24-25 = 修正前実測 / 本監査 = 修正後実測)。
- 分離の自然実験: H 監査 census の RWDI 9 edge は「`/EHsc` あり × `/utf-8` なし」—
  2 flag が独立した供給経路（base vs config/target-level）を持つことの直接証拠。
- **「/utf-8 修正で RWDI build が成功した ⇒ /EHsc も問題だった」という推論は不成立**。
  RWDI は /EHsc について何も失っておらず、追加も不要。
- 失敗モードも別系統: /utf-8 欠落 = C4819/mojibake (CP932 解釈)・C1083 連鎖、
  /EHsc 欠落 = C4530 警告 + unwind 保証喪失、JUCE では C1189。H5 の broken-build ログに
  C1189/C4530 が無いことが「/EHsc は健在だった」の傍証。

## I8 — 変更契約

**Defect 不成立のため、修正実装は存在しない。** 以下は将来の CMake flag 触及時の契約:

```text
必須:   (実装なし — Case A)
        ・/EHsc を RWDI (または任意 config) の set() に追加しないこと。
          追加しても edge は変化しない (base flag が既に供給) — 重複だけが増える。
        ・base CMAKE_CXX_FLAGS (/DWIN32 /D_WINDOWS /EHsc) を破壊しないこと。
          明示 set や -DCMAKE_CXX_FLAGS=... で上書きすると CMake 既定が失われ、
          その時に限り /EHsc が全 edge から消える (Case C1 = JUCE C1189 が活性化する)。

優先:   (ドキュメント衛生 — 次回 CMakeLists 触及許可時に限る・本監査では未実施)
        ・L1531-1533 の H5 追加コメント「/EHsc は RWDI にも欠落しているが…」を訂正
          (真実: base CMAKE_CXX_FLAGS により RWDI edge にも /EHsc あり — 本報告参照)
        ・L1996 の「RelWithDebInfo の CMAKE_CXX_FLAGS には /utf-8 が無い」は H5 後 stale

禁止:   × 7 target への ad-hoc /EHsc 追加 (不要)
        × test source 変更 / JUCE source 変更
        × gate による隠蔽
        × C1189 コメントだけからの /EHsc 必要性推定 (機構は実証済だが前提が崩れている)
        × juce_recommended_config_flags への依存追加 (現状未 link・不活性)
```

## I9 — regression contract

実装フェーズが存在しないため**新規 regression contract は不適用**。
既存の回帰基準はそのまま維持:

```text
/utf-8 census:   3 config とも 492/492 (H6 基準の継続維持)
/EHsc census:    3 config とも CXX 492/492 (本監査で新規基準化 — 将来の flag 変更時に再測定すること)
I3-4-G negative: 8/8 MATCH (gate 無変更のため自然維持)
change scope:    本監査 = source 変更 0
```

## I10 — GO/NO-GO

| Gate | 条件 | 判定 |
| --- | --- | --- |
| I1 | /EHsc 全 occurrence provenance (47 行分類 + base flag 出所 3 重裏取り) | **PASS** |
| I2 | Release/Debug/RWDI 全 CXX edge census (492×3・欠落 0・独立照合 compile_commands) | **PASS** |
| I3 | target → edge provenance 完全追跡 (33 target-level + 7 base-only・C edge 16/16/15) | **PASS** |
| I4 | 根因分類 = **Case A (NO DEFECT)**・RWDI /EHsc 追加不要・C flags 追加不要 | **PASS** |
| I5 | JUCE exception semantics 実証 (guard 存在確認 + fixture C1189 再現 + build log 0 件) | **PASS** |
| I6 | adversarial fixture Case A-E 完走 (%TEMP% のみ・production 0) | **PASS** |
| I7 | /utf-8 と /EHsc の因果分離 (独立供給経路の実証・横展禁止の根拠) | **PASS** |
| I8 | 修正契約 = NO-CHANGE (必須 0・将来契約と禁止事項を明記) | **PASS** |
| I9 | regression contract = 実装なし / 既存基準維持 | **PASS** (N/A 明示) |
| I10 | source 変更 0 (production/test/CMake/build.bat/tools・fixture は %TEMP%) | **PASS** |

**総合判定 = I3-4-I GO（I1-I7 証明済み、I8 方針確定 = 修正不要）。**
次フェーズ I3-4-J（/EHsc 修正実装）は**実施不要**として閉じる。

---

## 付記（棚卸し・保留事項 — 本監査では修正しない）

1. **icx (IntelLLVM) の Debug/RWDI `/utf-8` coherence** — build-icx (08-26 snapshot) で
   Debug/RWDI が 375/492 (/utf-8 なし 117 edge・target-level /utf-8 の無い target 群)。
   /EHsc は icx でも 492/492 で成立。CMakeLists の IntelLLVM block は Release のみ global
   `/utf-8` set を持ち Debug/RWDI を欠く。**/EHsc とは別問題** — icx は既定で UTF-8 ソース
   charset を解釈するため実害の有無は別途確認が必要。後続監査候補 (I3-4-K 系)。
   なお build-icx tree は 08-26 の snapshot であり、現在の MSVC tree の状態を代表しない。
2. **ConvoPeq.md が stale** (Generated 2026-09-05 12:02 < H5 編集 09-06 00:00)。
   H5 の RWDI set 行を含まない。次回の派生スナップション更新時に
   `python output_sourcecode_markdown.py` で再生成すること (本監査は未実施・read-only)。
3. **build/ tree に .build_identity stamp が現存しない** (build-diag にはあり)。
   次回 build.bat 実行時に gate が再生成する。census には影響なし。
4. cppcheck / clang-tidy / Dr.Memory: 本監査は CMake flag 定義と生成物 (ninja/cache) の
   provenance 監査であり、C++ セマンティクス解析・実行計装の対象コード変更が無いため適用なし
   (H5 報告と同一理由)。ccc / graphify / semble: /EHsc はビルド構成テキストの完全列挙課題であり
   セマンティック検索の対象でない (grep 系で網羅性が定義上保証される)。serena は初期指示読込に使用。

## 添付 evidence (evidence/D162-2I3/)

```text
I3_4_I_EHSC_CONFIG_TARGET_PROVENANCE_AUDIT.md   本書
i3_4_i0_freeze/ (CMakeLists.txt + freeze.json)   I0 baseline freeze
i3_4_i1_occurrence_table.txt                     I1 全 47 occurrence 分類表
i3_4_i2_raw_census.json                          I2 census raw (3 config × msvc + icx)
i3_4_i2_edge_census.txt                          I2/I3 per-target provenance 表
i3_4_i6_fixture_log.txt                          I6 fixture 全ログ + Case E cache 証跡
```
