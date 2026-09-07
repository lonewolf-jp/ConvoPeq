# D162-2-I3-4-K — IntelLLVM `/utf-8` Configuration Coherence Audit

```text
Date:        2026-09-06
Type:        read-only audit (K0-K10) — 変更 0 (production/test/CMake/build.bat/source/tools すべて 0・
             icx 現行生成は %TEMP% の scratch tree で取得し repository の build tree は不 touch)
Baseline:    I3-4-I GO (Case A: /EHsc NO DEFECT) / commit 9cacee1f+dirty (H5 RWDI /utf-8 含む)
判定:        **K7 = Case B — Configuration incoherence / no current build failure。
             icx Debug/RWDI は Release と /utf-8 供給源が不整合 (CXX 375/492・C 9/16) だが、
             icx は UTF-8 (BOM 無し) を既定 charset として解釈するため build failure は発生しない
             (fixture Case A/B/C/D/E + 08-26 snapshot の実 compile 実績で実証)。
             修正は機能的に不要。統一する場合の契約のみ K8 に記録。総合 = I3-4-K GO。**
```

---

## 0. 要旨

> **IntelLLVM (icx) では、Release のみ config-level `/utf-8` set (CMakeLists L1606/L1607) が存在し、
> Debug / RelWithDebInfo は CMake 既定 flags (`/Zi /Ob0 /Od /RTC1`・`/Zi /O2 /Ob1 /DNDEBUG`) に
> フォールバックして `/utf-8` を欠く。その結果、現行 CMakeLists での icx 現行生成において
> Debug/RWDI は CXX 492 edge 中 375 のみ /utf-8 付き (117 edge 欠落・26 target)、
> C 16 edge 中 9 (7 target 欠落)。欠落 target は (a) ISR-family 19 target — その /utf-8 供給が
> L850-871 の `if(MSVC AND NOT ... IntelLLVM)` block 内にのみ存在、(b) ISR test 7 target —
> target-level /utf-8 を一切持たず cl では H5 の config-level set で補完されるが icx では
> 該当 set が存在しない。ただし cl (MSVC) と異なり icx は UTF-8 ソースを既定で UTF-8 と
> 解釈するため、/utf-8 無しでも BOM 無し UTF-8 source (日本語コメント/識別子/文字列・
> Shift-JIS 危険文字含む) は compile・実行とも正しく、failure は 0。**

---

## K0 — Baseline freeze

```text
git status            CMakeLists.txt M (+10/-0 = H5 のみ) / それ以外 untracked evidence・doc のみ
CMakeLists.txt        sha16 = (i3_4_k0_freeze/CMakeLists.txt に保存・I 監査と同一 5bdc2a2a 系)
build-icx 現行生成     impl-*.ninja = 2026-08-26 22:13 / CMakeCache = 2026-08-23 19:01
                      → 現行 CMakeLists より古い snapshot (★ K2 で現行生成を別途取得・比較)
ツールチェーン         icx = Intel oneAPI DPC++/C++ 2026.1.0 Build 20260617
                      (C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe)
                      generator = Ninja Multi-Config / ninja = WinGet Links (1.13.2 系)
                      icx mode は build.bat L137-158: setvars.bat intel64 (+MKLROOT/IPPROOT/LIB 追加)
build option          ENABLE_ASAN=OFF / ENABLE_TSAN=OFF / CONVOPEQ_PGO_INSTRUMENT=OFF /
                      CONVOPEQ_PGO_USE=OFF / CONVOPEQ_ENABLE_ISR_TESTS=ON / CONVOPEQ_ENABLE_CLANG_TIDY=OFF
icx cache flags        CMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc (base — /utf-8 無し)
                      CMAKE_C_FLAGS=/DWIN32 /D_WINDOWS
                      *_DEBUG=/Zi /Ob0 /Od /RTC1・*_RELEASE=/O2 /Ob2 /DNDEBUG・*_RWDI=/Zi /O2 /Ob1 /DNDEBUG
                      (すべて CMake 既定 — IntelLLVM block からの config set は cache に現れない=
                       非キャッシュ set は generate 時にのみ効く)
ConvoPeq.md           09-05 生成 = H5 以前の stale source → 本監査の基準に使用しない (K0 指示どおり)
production/test/source 変更: 0
```

## K1 — IntelLLVM `/utf-8` provenance 完全追跡

全 `/utf-8` 出現箇所の条件チェーン解析 (full: `i3_4_k1_provenance_chain.txt`)。icx では
`MSVC=TRUE` (MSVC-like frontend)・`CMAKE_CXX_COMPILER_ID="IntelLLVM"`・`ENABLE_ASAN=OFF`。

| 供給源 | 行 | icx 適用 | 内容 |
| --- | --- | --- | --- |
| base CMAKE_CXX_FLAGS | (cache 既定) | **/utf-8 無し** | `/DWIN32 /D_WINDOWS /EHsc` |
| config-level CXX Release | L1606 (IntelLLVM block) | **適用** | `/O2 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8 /EHsc` |
| config-level C Release | L1607 (IntelLLVM block) | **適用** | 同上 /utf-8 あり |
| config-level CXX/C Debug | **存在しない** (IntelLLVM block) | — | icx Debug は CMake 既定のみ → /utf-8 なし |
| config-level CXX/C RWDI | **存在しない** (IntelLLVM block) | — | 同上 |
| target_compile_options (/utf-8) | L174, L322-812 群, L1100 (if(MSVC)), L1886 | **適用** | PublicationAdmission/D8_2/ISRSoak/Owner/Mpsc/MTNUPC/AudioEngineHarness 等 14 target |
| target_compile_options (/utf-8) | L1576 (IntelLLVM block) | **適用** | ConvoPeq 本体 |
| target_compile_options (/utf-8) | **L850-871 群** | **不適用** ★ | `if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")` 内 → icx では効かない |
| config-level set (MSVC block) | L1519/L1524/L1534 (H5) | **不適用** ★ | cl 専用 block 内 |
| GenEx / ASan | L1997 | 不適用 | `ENABLE_ASAN` ガード (OFF) + MSVC-only |
| JUCE interface | — | **不在** | `juce_recommended_config_flags` は project から 0 参照 (K5) |
| CMake/IntelLLVM platform 既定 | — | /utf-8 無し | Case C-E fixture で実証 |

**「IntelLLVM block は Release のみ global /utf-8 set を持つ」を現行生成物で再確認**: Yes。
icx Release edge FLAGS に L1606 由来の `/O2 /DNDEBUG /QxCORE-AVX2 /fp:fast /Gy /Zi /utf-8 /EHsc` が
現れ、Debug/RWDI edge FLAGS には現れない (K3 FLAGS 実測)。

## K2 — 3 configuration × 全 compile edge census

**現行生成** (本監査内で `%TEMP%\icx_i34k\bld` に icx scratch configure — 現行 CMakeLists
@ 9cacee1f+dirty を -B のみで消費、repository には書き込みなし。configure log: `i3_4_k2_current_configure.log`):

| Config | CXX edges | /utf-8あり | /utf-8なし | C edges | /utf-8あり | /utf-8なし |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Release | 492 | **492** | **0** | 16 | **16** | **0** |
| Debug | 492 | **375** | **117** | 16 | **9** | **7** |
| RelWithDebInfo | 492 | **375** | **117** | 16 | **9** | **7** |

- /EHsc は 3 config とも 492/492 (I3-4-I の Case A 結論は icx でも維持)。
- **08-26 snapshot (build-icx) との edge 単位比較: 508 edge × 3 config で差分 0。**
  H5 の MSVC block 修正は icx 生成に一切影響しないことが実証され、
  旧 snapshot 観測値 (375/492・117 欠落) が**現行値としても正確**であることが確認された
  (指示どおり「仮説として扱い、現在値として再取得した」)。
- JUCE bootstrap (juceaide) は configure 内で正常 build (icx・rc=0)。

## K3 — target provenance (なぜその target 群だけ欠落するか)

117 欠落 CXX edge は **26 target** に集約。2 群に分かれる:

```text
群1 (19 target / ISR-family・L850-871 block 対象):
  TerminalTelemetryContractTests(10) ISRRuntimeIdentityTests(1) RuntimePublicationCoordinatorTests(1)
  ISRSemanticValidationTests(17) invariant_INV3_INV5Tests(18) AdmissionPackedStateTests(18)
  RetireGraceSemanticsTests(10) ShutdownRetireIntentDrainTests(11) NormalRetireDSPHandleCompareTests(2)
  RuntimeSemanticSchemaValidationTests(1) ObservePathSingleSourceTests(1) OverlapAuthoritySingularTests(1)
  ShadowCompareContractTests(1) CrossfadeExecutorLocalContractTests(1) RuntimeWorldAuthorityProjectionTests(1)
  PartialPublicationRejectTests(1) RebuildAdmissionRegressionTests(1) BuildInputSemanticContractTests(1)
  PriorityIntegrationTests(11)
  → これらの唯一の /utf-8 供給が L850-871 の if(MSVC AND NOT ... IntelLLVM) 内 target options。
    icx では条件が false となり供給 0。JUCE module source を多数 compile するため edge 数が多い
    (例: invariant_INV3_INV5Tests 18 edge = JUCE モジュール群)。

群2 (7 target / target-level /utf-8 を一切持たない):
  D8_1_WrapperCacheTests(1) DeferredDeletionQueueReclaimTests(1) EQAnalysisUnitTests(1)
  EQBoundExcessBenchmark(1) EQProcessorMaxGainTests(1) GainStagingContractTests(1) FFTBackendTests(3)
  → cl では H5 の config-level set (L1519 Release / L1524 Debug / L1534 RWDI) で補完されるが、
    いずれも if(MSVC AND NOT ... IntelLLVM) 内 → icx Debug/RWDI では補完経路が存在しない。
    FFTBackendTests のみ target-level /EHsc (L626) を持つが /utf-8 は target-level に無い。

C edge 欠落 (7 target・Sheenbidi .c): 群1 のうち C ソース (juce_graphics_Sheenbidi.c) を
  compile する 7 target。target options (/utf-8) が icx で効かないため C edge も欠落。
  Release では L1607 の config-level set が C 言語にも効くため 16/16。
```

言語分離: 同一 target でも CXX と C で状態は一致 (供給は target-level → 両言語共通)。

実測 FLAGS (現行 icx Debug):

```text
欠落側  invariant_INV3_INV5Tests: /DWIN32 /D_WINDOWS /EHsc /Zi /Ob0 /Od /RTC1 -Qstd:c++20 -MDd /EHsc /bigobj
欠落側  D8_1_WrapperCacheTests:    /DWIN32 /D_WINDOWS /EHsc /Zi /Ob0 /Od /RTC1 -Qstd:c++20 -MDd
供給側  PublicationAdmissionTests: ...(同head)... /utf-8 /bigobj          (L174 target option)
供給側  ConvoPeq:               ...(同head)... /EHsc /utf-8 ...           (L1576 IntelLLVM block)
供給側  MTNUPCMeasurement:      ...(同head)... /utf-8 /EHsc ...           (L1100 if(MSVC))
対照 (cl build/ Debug): D8_1 = /D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc (H5 config set)
```

MSVC cross-check: **26 欠落 target は cl (build/ tree Debug) では全員 /utf-8 を持つ** (実測 True)。

## K4 — icx source encoding semantics (compiler requirement の分離評価)

fixture: BOM 無し UTF-8・日本語コメント + UTF-8 識別子 `変数` + Shift-JIS 危険文字を含む
文字列 `"日本語テスト表ソ十構"` (表/ソ/十/構 = CP932 で 2nd byte 0x5C を含む組)。log: `i3_4_k4_k6_fixture_log.txt`。

```text
Case A  icx + /utf-8    : compile rc=0・IDENT_OK=42・LEN=30・bytes = E6 97 A5 ... E6 A7 8B (UTF-8 そのもの)
Case B  icx /utf-8 無し : compile rc=0・実行出力 = Case A と完全同一 (UTF-8 bytes がそのまま保持)
Case B2 cl  /utf-8 無し : warning C4819 (codepage 932 で表示不可) + error C2065 ('g_text' 未定義 =
                          日本語コメントが CP932 誤読されパース崩壊) → rc=2 compile 失敗 ★対比
```

→ **icx は /utf-8 無しでも BOM 無し UTF-8 source を UTF-8 として解釈する** (clang 系の既定
input charset = UTF-8)。/utf-8 は icx にとって no-op。一方 cl は ACP(932) 既定のため
同一 source が compile 不能 — H5 で修復した defect クラスは cl 固有。

## K5 — JUCE interaction

- `juce_recommended_config_flags` (JUCEHelperTargets.cmake:123, /EHsc+/Od,/Ox+/MP): project CMakeLists
  から **0 参照** → edge への寄与なし。実測: icx Debug edge の `/Od` 508 件は CMake 既定
  `CMAKE_CXX_FLAGS_DEBUG=/Zi /Ob0 /Od /RTC1` 由来、Release `/Ox` 0 件 — JUCE interface 特有の
  供給 signature は一切現れない。「使っているはず」仮定は棄却。
- JUCEUtils.cmake の `/utf-8`・`/EHsc` 参照: 0 — JUCE 側から /utf-8 は供給されない。
- JUCE module source は project target に組み込まれて compile されるため、その target の flags
  に従う (= 欠落 target 内の JUCE edge は /utf-8 なしで compile されている)。

## K6 — fixture (CMake IntelLLVM per-config)

最小 CMake project (project(utf8_probe CXX)・flag 記述 0) を icx で configure:

```text
Case C Release        : FLAGS = /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG -MD        (/utf-8 無し)
Case D Debug          : FLAGS = /DWIN32 /D_WINDOWS /EHsc /Zi /Ob0 /Od /RTC1 -MDd      (/utf-8 無し)
Case E RelWithDebInfo : FLAGS = /DWIN32 /D_WINDOWS /EHsc /Zi /O2 /Ob1 /DNDEBUG -MD    (/utf-8 無し)
→ 3 config とも /utf-8 無しで BOM 無し UTF-8 fixture が build rc=0・実行 LEN=30・bytes 完全一致。
  CMake/IntelLLVM 既定は /utf-8 を供給せず、それでも icx は UTF-8 を正しく扱う。
```

(注意: fixture の rc 出力順序の都合上、Case C の Release run 出力は Case E セクションに連続表示。
build rc=0 は 3 config 全て個別に記録済み。)

## K7 — 根因分類

**Case B — Configuration incoherence / no current build failure**

- **configuration coherence**: 不成立。icx Release は L1606/L1607 (config-level)、Debug/RWDI は
  供給源なし (base も target-level も無い対象 26 target)。同一 semantic source から全 config が
  /utf-8 を得ている状態ではない。
- **compiler semantic requirement**: icx は /utf-8 を要求しない (K4/K6 実証)。BOM 無し UTF-8 が
  既定解釈され、実行時 bytes も /utf-8 ありと完全一致。
- **build failure**: なし。08-26 snapshot の build-icx には欠落 26 target の obj が実在
  (/utf-8 無しで compile 済みの実績)、Case C-E でも 3 config とも rc=0。
- 判定は generated compile edge を最終 authority として行った (K2 現行生成 census)。

## K8 — 修正契約 (実装は本監査では行わない・CMakeLists.txt は未変更)

```text
必須:   (Case B — 機能的必須項目は無し)

選択肢 (統一する場合 — H5 と同じ configuration-level semantic source 統一を第一候補):
  案1  IntelLLVM block 内に cl block (L1519/L1524/L1534) と対になる config-level set を追加
       (例: set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Ob0 /Od /RTC1 /utf-8") 等 CXX+C × Debug/RWDI)
       → icx 3 config が /utf-8 を config-level で統一受けする。Release (L1606) と対称。
  案2  icx では /utf-8 が不要であることを根拠に、L1606/L1607 から /utf-8 を外す
       → 供給源統一 (icx = target-level のみ)。ただし Release set の変更を伴い影響面が広い。
  推奨: 実害が 0 のため「修正しない (現状維持 + 本報告を公式記録とする)」を基本線とし、
       統一を進める場合は案1。いずれも次フェーズ (K-impl) でユーザー判断を要する。

禁止:
  × icx Debug/RWDI への ad-hoc target_compile_options (26 target 個別対応)
  × source への BOM 追加
  × test source 変更
  × gate whitelist / compiler workaround
  × MSVC block (L1519/L1524/L1534) への影響 (H5/H8 の 492/492/492 を変えない)
```

## K9 — regression contract

Case B につき今回の closure で修正不要。将来 K-impl (案1) を実施する場合の基準:

```text
IntelLLVM (icx):
  /utf-8 census  : Release/Debug/RWDI = CXX 492/492/492・C 16/16/16
  build          : 3 config rc=0 (少なくとも clean configure + 対象 26 target の rebuild)
MSVC (cl) — 不変条件:
  /utf-8 census  : 492/492/492 維持 (I3-4-H5/H8 基準)
  /EHsc census   : 492/492/492 維持 (I3-4-I 基準)
  CTest          : Release/Debug 40/40 (config-level set に触れないため影響想定なし)
I3-4-G negative  : 8/8 MATCH (gate 無変更)
change scope     : CMakeLists IntelLLVM block のみ・src/tests/build.bat/tools 変更 0
```

## K10 — GO/NO-GO

| Gate | 条件 | 判定 |
| --- | --- | --- |
| K1 | /utf-8 provenance 完全追跡 (条件チェーン付き全出現箇所) | **PASS** |
| K2 | 3 config × 全 edge census (**現行 icx 生成を %TEMP% scratch で取得**・snapshot と edge diff 0) | **PASS** |
| K3 | 欠落 117 edge = 26 target の供給経路完封 (群1 19 + 群2 7・MSVC cross-check) | **PASS** |
| K4 | icx compiler semantics 分離評価 (Case A/B 同一出力・Case B2 cl は失敗 = 対比) | **PASS** |
| K5 | JUCE interaction (interface 不在の実証・/Od は CMake 既定由来) | **PASS** |
| K6 | fixture Case C-E (CMake icx 既定に /utf-8 無し・3 config build rc=0) | **PASS** |
| K7 | 根因分類 = **Case B (incoherence / no build failure)** | **PASS** |
| K8 | 修正契約 (実装なし・案1/案2 と禁止事項を記録) | **PASS** |
| K9 | regression contract (Case B closure・将来実装時基準を記録) | **PASS** |
| K10 | 変更 0 (CMakeLists.txt 未 touch・repository build tree 未 touch) | **PASS** |

**総合判定 = I3-4-K GO。K7 = Case B のため K8 の実装 (案1) は任意 — ユーザー判断待ち。**

---

## 付記

1. **本監査の K2 で repository の build-icx tree は再 configure していない** (read-only 維持)。
   現行生成は `%TEMP%\icx_i34k\bld` (cmake -S <repo> -B <temp>) で取得し、生成物は監査後も
   削除していない (再検証可能)。edge diff 0 のため build-icx snapshot の census 値は現行値と
   同一として扱える。
2. icx 環境: oneAPI setvars.bat intel64 のみでは Windows SDK LIB が付かず configure の
   compiler ABI check が LNK1104 で失敗する。build.bat (L137-158) と同様の SDK probe
   (Windows Kits の LIB/INCLUDE 追加) が必要 — 手順を configure log に記録。
3. **インベントリ (未実装新規 CR) の現状**: BuildError retry backoff/count/telemetry と
   CW-8 PublishedWorldObservation atomic snapshot contract の 2 系統。I3-4-K 後も
   勝手な実装着手はせず、CR として判断を分ける (D159 freeze の Phase-II recovery episode 系も同様)。
   Practical Stable ISR Bridge Runtime 原則 (Build/Validate/Publish/Retire/Delete の責務分離・
   RT 観測限定) に照らし、今回の監査でも source-side workaround は実施していない。
4. ツール使用記録: census・chain・census 解析 = ctx-mode (ctx_execute python) + grep/sed/find。
   icx 現行生成 = oneAPI setvars + cmake 4.4.3 (scratch %TEMP%)。serena = 初期指示読込。
   cppcheck / clang-tidy / Dr.Memory: C++ コード変更 0 の flag provenance 監査のため適用なし
   (H5/I 報告と同一理由)。ccc / graphify / semble: 完全列挙課題は grep 系で網羅するため不使用。

## 添付 evidence (evidence/D162-2I3/)

```text
I3_4_K_INTELLLVM_UTF8_CONFIG_COHERENCE_AUDIT.md   本書
i3_4_k0_freeze/ (CMakeLists.txt + freeze.json)     K0 baseline freeze
i3_4_k1_provenance_chain.txt                       K1 /utf-8 条件チェーン全表
i3_4_k2_current_configure.log                      K2 現行 icx scratch configure ログ
i3_4_k2_census.json                                K2 現行 + snapshot census (raw)
i3_4_k3_target_provenance.json                     K3 欠落 target 分類 (26 + C 7)
i3_4_k4_k6_fixture_log.txt                         K4/K6 fixture 全ログ (Case A/B/B2/C/D/E)
i3_4_k4_fixture_source.cpp                         fixture source (BOM 無し UTF-8)
```
