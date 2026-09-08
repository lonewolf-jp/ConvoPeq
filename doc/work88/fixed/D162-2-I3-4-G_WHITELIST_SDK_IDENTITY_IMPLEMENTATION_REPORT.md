# D162-2-I3-4-G — Whitelist + Windows SDK Identity Implementation Report

```text
Date:            2026-09-05
Type:            build-system contract implementation (G0 audit -> G1-G9)
Changes:         src/tools/build_identity_gate.py  v1 -> v2 (再実装・223 行 -> 約 430 行)
                 build.bat                          変更 0（v1 CLI 契約を維持するため不要と判明）
                 src/** runtime: 0 / tests: 0 / CMakeLists.txt: 0
Baseline:        I3-4-F GO 12/12 / build-diag (E0 凍結系譜)
判定:            **G1-G12 全 PASS → I3-4-G GO。fail-closed gate は warning-level の
                 whitelist を獲得し、92 obj 事故クラスの保護は完全に維持。**
```

---

## 0. 要旨

> **`#deps 0` は値では判断しない。production link set に属する obj は常に FAIL、
> test obj は transitive include closure で「非システム依存の不在」を証明できた
> 場合のみ WARN+ALLOW、それ以外は FAIL。**
> 加えて Windows SDK identity（dir / version / include-tree fingerprint / source）
> を stamp に追加し、同一バージョン番号の in-place servicing も identity 照合で
> 検出できるようになった。

実装後の実物ビルドでは、I3-4-D 以降 Release build を塞いでいた 10 test TU が
**10 × ZERO_DEPS_SYSTEM_ONLY → 10 × WARN + ALLOW** に転換し、Release/Debug の
build・link・CTest 40/40・harness exit 0x0 ×3・layout coherence がすべて回復した。

---

## G0 — 実装前監査（`i3_4_g_g0_preimpl_audit.txt`）

1. **build.bat**: msvc モードに明示 vcvarsall 呼出しは無い（開発シェルの vcvars 継承）。
   → **gate 起動時に `WindowsSdkDir` / `WindowsSDKVersion` / `INCLUDE` は生存**（実証）。
   gate は `:configure_cmake_ok` 直後（configure と build の間）。
2. **gate v1**: identity schema 1・`--check` 全差分 rc=3・`#deps 0` 一律 rc=3・
   `RELEVANT_OBJ_RE = /(?:Release|Debug|RelWithDebInfo)/src/`・rc=3 経路 5 系統を確認。
3. **check_layout_offsets.py**: gate と非干渉（独立 PE スキャナ・build-diag パス内蔵）。
4. **impl-*.ninja**: TU→target→object は build edge 出力パスと edge 変数から安全に取得可。
5. **production link set**: `build ConvoPeq_artefacts\<cfg>\ConvoPeq.exe` edge の obj 入力
   （3 config とも 134 obj・src/ 116・rspfile なし）から決定的に構築できる。
6. **F4 fixture 再利用**: fixture に (a) `CMakeFiles/impl-Release.ninja`（compile edge +
   必要なら production link edge）(b) `CMakeFiles/rules.ninja`（CP932 prefix）
   (c) 実 CMakeCache 形式 `key:TYPE=value` の最小 cache を置けば gate が無修正で動く。
   → **whitelist は 10 TU のファイル名リストにしない**設計どおり実装。

## G1 — Windows SDK identity（stamp schema 2）

```json
 "windows_sdk_dir": "C:\\Program Files (x86)\\Windows Kits\\10",
 "windows_sdk_fingerprint_source": "environment",
 "windows_sdk_include_fingerprint": "2c5e417e4aee1303",
 "windows_sdk_version": "10.0.26100.0"
```

- 真実源は F5 確定どおり vcvarsall 環境（`WindowsSdkDir` / `WindowsSDKVersion`）。
- fingerprint = SDK include tree（ucrt/um/shared/winrt/cppwinrt）の
  `relpath|size|mtime_ns` canonical manifest の SHA256[:16]。
  **同一 version の in-place servicing も mtime/size 変化で検出**（F5 実測 4,771 files・0.05s）。
- env が無い場合（build.bat 外で gate を起動）は `unknown` となり stamp 照合で fail-closed
  （正しい挙動: gate は build.bat と同一環境で走るべき）。

## G2/G3 — zero-deps classifier と provenance 境界

分類器（`classify_zero_deps`）と closure エンジン（`tu_closure`）を実装:

```text
#deps == 0 の relevant obj
  ├─ production link set 属する → PRODUCTION_ZERO_DEPS → FAIL（例外なし）
  └─ test obj
       ├─ closure に非システム依存あり   → ZERO_DEPS_SUSPICIOUS → FAIL
       ├─ closure に unresolved include  → ZERO_DEPS_UNRESOLVED  → FAIL
       └─ closure が全システム & 解決済み → ZERO_DEPS_SYSTEM_ONLY → WARN + ALLOW
```

- **TU identity は完全 source path + edge identity**（basename key 禁止 — F2 で実測した
  衝突を回避）。`-I` は **impl-<Config>.ninja の edge 変数**から取得（target 別の真実）。
- angle include の解決は **INCLUDE env**（vcvars 系）まで行い、解決不能は無条件 FAIL。
- `--explain-zero-deps` を追加（identity 検証後・rc=0 診断専用・fail-closed を損なわない）。
  `--strict` は不採用（F8 判定どおり）。

## G4 — adversarial regression（`i3_4_g_g4_fixture_results.txt`）

| Case | 期待 | 実測 |
| --- | --- | --- |
| system-only | WARN + ALLOW | **MATCH (rc=0)** |
| project direct | PASS | **MATCH (rc=0)** |
| project transitive | PASS | **MATCH (rc=0)** |
| project + prefix mismatch | **FAIL** | **MATCH (rc=3 SUSPICIOUS)** |
| unresolved include | **FAIL** | **MATCH (rc=3 UNRESOLVED)** |
| production + `#deps 0` | **FAIL** | **MATCH (rc=3 PRODUCTION)** |
| test-only + system-only | WARN + ALLOW | **MATCH (rc=0)** |
| test-only に project dependency 混入 | **FAIL** | **MATCH (rc=3 SUSPICIOUS)** |

**8/8 MATCH。Case D（prefix mismatch → project TU が `#deps 0`）が確実に FAIL になる
ことが gate 実装レベルで再実証された。**

## G5 — 実物 Release build（build-diag）

```text
1. stamp migration schema 1 -> 2       rc=0（SDK 4 項目を初記録・WARN 表示）
2. gate --check Release                rc=0: 337 objs checked / 10 WARN+ALLOW /
                                       production #deps 0 = 0 / suspicious = 0 / unresolved = 0
3. cmake --build --config Release      rc=0（no work to do — coherent tree）
4. touch-cycle: ObservePathSingleSourceTests.cpp のみ mtime touch（内容 hash 不変を検証）
   -> obj 再 compile + link -> gate --check rc=0（再び 10 WARN+ALLOW）
   = compile -> deps 記録(0 by design) -> 分類(WARN+ALLOW) の全サイクルが実物で成立
```

過去の 92 obj 事故クラス（I3-4-D D10-C）の再発は無い。production #deps 0 は常に 0。

## G6 — Debug / RelWithDebInfo

```text
Debug:           build rc=0（JuceHeader.h 再生成由来の全面再 compile 含む）
                 gate --check rc=0: 337 objs / 10 WARN+ALLOW / production 0
RelWithDebInfo:  gate --check rc=0: 139 objs / 6 WARN+ALLOW / production 0
                 （RWDI は 10 TU のうち 6 obj に deps 記録がある状態で検証）
```

- configuration-specific whitelist になっていないことを確認（同一 classifier・同一閾値）。
- **pre-existing の発見（本監査では未修正・スコープ外）**: RWDI の compile FLAGS には
  `/utf-8` が無い（Release/Debug には存在。CMake の per-config options の設定漏れ）。
  UTF-8 (BOM 無し) ソースを CP932 解釈で読むため、RWDI の全面再 build は
  `AtomicAccess.h(55)` 等で C1083/C2143 により失敗する（実録:
  `i3_4_g_build_RWDI.log`）。これは gate とは無関係の既存ギャップであり、
  CMakeLists.txt 変更 0 の本スコープでは修正しない → **次回作業項目として起票**。
  gate の RWDI 分類検証自体は既存 coherent tree 上で成立している。

## G7 — identity negative tests（`i3_4_g_g7_identity_negatives.txt`）

| Case | stamp 改変 | 結果 |
| --- | --- | --- |
| G7-A SDK version mismatch | `10.0.99999.0` | **rc=3 検出** |
| G7-B SDK fingerprint mismatch（in-place servicing 模擬・version 同一） | `deadbeefdeadbeef` | **rc=3 検出** |
| G7-C compiler mismatch | `MSVC 19.99` | **rc=3 検出** |
| G7-D codepage mismatch | `65001` | **rc=3 検出** |
| G7-E generator mismatch | `Visual Studio 18 2026` | **rc=3 検出** |

stamp は全ケースで元の hash（42ae358299e07111）に**完全復元**され、復元後の
`--check` は rc=0。G7-B により「version 同一・fingerprint 異常」の in-place servicing
が確実に build 拒否になることを実証した。

## G8 — build.bat integration の順序

```text
configure (:configure_cmake_ok)
  ↓ gate 呼出し（build.bat 197-214 行・変更 0 で維持）
  ↓ gate main(): ①identity 検証（stamp 照合・不一致は即 rc=3）
  ↓              ②deps extraction（ninja -t deps）
  ↓              ③zero-deps semantic classification
  ↓              ④FAIL があれば停止（rc=3・recovery 表示・自動 clean なし）
build / link
```

- identity mismatch を classifier より後に判定する経路は存在しない（main() の構造で保証）。
- 実測ログでも identity 行が分類出力に常に先行（`i3_4_g_check_release.log`）。
- **build.bat は無変更**（gate v2 が v1 の CLI 契約 `--build-dir/--check/--show/--config`
  を維持したため。G スコープ原則「変更は build.bat + gate に閉じる」をより厳密に達成）。

## G9 — regression

```text
Release harness ×3:  exit 0x0 / 0x0 / 0x0
Debug   harness ×3:  exit 0x0 / 0x0 / 0x0
CTest Release:       100% tests passed, 40/40
CTest Debug:         100% tests passed, 40/40
layout offsets:      PASS（Release 0x1290880:4・旧 crash offset 0x12A7640 は全 config 0）
変更監査:            本セッションの変更は src/tools/build_identity_gate.py のみ
                     （build.bat の +17 行は I3-4-D 時点の既存差分・本セッション未変更）
                     src/** runtime 0 / tests 0 / CMakeLists.txt 0
                     （ObservePathSingleSourceTests.cpp は mtime touch のみ・
                      内容 hash 74b48b54665b9314 が touch 前後で不変を検証）
```

## GO 条件対合（G1-G12）

| ID | 条件 | 判定 |
| --- | --- | --- |
| G1 | SDK identity 4 項目が stamp に記録される | **PASS**（schema 2 実物） |
| G2 | SDK version mismatch → rc=3 | **PASS**（G7-A） |
| G3 | SDK fingerprint mismatch → rc=3 | **PASS**（G7-B） |
| G4 | `#deps 0` semantic classifier 実装 | **PASS**（G4 8/8） |
| G5 | system-only test TU → WARN + ALLOW | **PASS**（fixture + 実物 10 TU） |
| G6 | project direct → PASS semantics 正常 | **PASS**（fixture B + 実物 327 objs） |
| G7 | project transitive → 誤許可しない | **PASS**（fixture C + DepsLog transitive 真実照合） |
| G8 | prefix mismatch → **FAIL** | **PASS**（fixture D・F4 連鎖） |
| G9 | unresolved → **FAIL** | **PASS**（fixture E） |
| G10 | production `#deps 0` → 例外なく FAIL | **PASS**（fixture F） |
| G11 | Release/Debug/RWDI + CTest 回帰 PASS | **PASS**（40/40 ×2・harness 6/6・layout PASS） |
| G12 | runtime/test/CMake source 変更 0 | **PASS**（gate script のみ） |

**総合判定 = I3-4-G GO（12/12）。**

## 残留事項（次回起票）

1. **RWDI `/utf-8` 欠落**（pre-existing・CMakeLists の per-config options 漏れ）:
   RWDI 全面再 build が不可。修正には CMakeLists.txt 変更が必要（G スコープ外）。
2. **build/ ディレクトリ**: stamp 未作成のまま。次回 `build.bat Release` 時に
   stamp-create → classifier が走る。build/ の .ninja_deps（Sep 1 時点）に
   production #deps 0 が残っていれば fail-closed が作動し、契約どおり
   `build.bat Release clean` を要求する（D10-C と同じ保護）。

## 添付 evidence（evidence/D162-2I3/）

```text
I3_4_G_WHITELIST_SDK_IDENTITY_IMPLEMENTATION_REPORT.md   本書
i3_4_g_g0_preimpl_audit.txt                               G0 監査
i3_4_g_g0_build_identity_gate_v1_backup.py                v1 退避
i3_4_g_g0_build_bat_backup.bat                            build.bat 退避（未変更の証明）
i3_4_g_g4_fixture_results.txt                             G4 8/8 matrix
i3_4_g_g5_g6_realbuild.txt                                G5/G6 実物 build + 分類
i3_4_g_check_{release,Debug,RWDI}.log                     config 別 gate 出力
i3_4_g_build_{release,Debug,RWDI}.log                     config 別 build 出力
i3_4_g_g7_identity_negatives.txt                          G7 5/5 negative + 復元
i3_4_g_explain_zero_deps.log                              --explain-zero-deps 実証
fixture: %TEMP%/i3_4_g_g4/*（8 契約・再実行可）
```
