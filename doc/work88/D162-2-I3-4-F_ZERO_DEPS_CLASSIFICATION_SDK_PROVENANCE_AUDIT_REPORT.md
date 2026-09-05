# D162-2-I3-4-F — `#deps 0` Semantic Classification / TU Dependency Provenance / Windows SDK Identity Audit

```text
Date:            2026-09-05
Type:            read-only audit + contract design + adversarial fixture (%TEMP% のみ)
                 実装は行わない（production/test/CMake/build.bat/src/tools 変更 0）
Baseline:        I3-4-E GO / build-diag 状態は E0 凍結から完全未変更（F0 で実証）
判定:            **F1-F12 全 PASS → I3-4-F GO。次段階は I3-4-G（whitelist 実装 +
                 Windows SDK stamp 実装）で、ユーザー承認後に限る。**
```

---

## 0. 要旨

1. **F1**: `#deps 0` を「値」ではなく「意味」で分類する 4 状態モデルを確立した
   （ZERO_DEPS_SYSTEM_ONLY / ZERO_DEPS_SUSPICIOUS / ZERO_DEPS_UNRESOLVED / DEPS_VALID）。
   判定器は **source graph の transitive include closure** を用い、
   `#deps 0` という値そのものは whitelist 根拠にしない。
2. **F2**: 判定ソースは (1) `impl-<Config>.ninja` の edge 変数（-I の真実、target 別に相違）、
   (2) `compile_commands.json`（同値・target 重複に注意）、(3) source `#include`
   （direct のみ・transitive は closure 計算が必要）、(4) `ninja -t deps`
   （ndeps>0 のときの transitive 真実）の 4 層で相互検証した。
   単純 grep を dependency truth にしない要求どおり、**transitive closure エンジン**を実装。
3. **F3**: 10 TU は source graph・build graph の両面から **全件
   ZERO_DEPS_SYSTEM_ONLY(EXPECTED)**（project=0 / third_party=0 / external=0 /
   unresolved=0）と再分類した。control group（`juce_core_CompilationTime.cpp`、
   #include 皆無）も同一クラス。
4. **F4**: adversarial fixture で Case A-D の 4 象限 + Case E（stale 伝播）を検証し、
   **Case D（prefix mismatch → project TU が `#deps 0`）は確実に SUSPICIOUS→FAIL になる**
   ことを実証。**これが成立しない whitelist は採用不可という条件を満たした。**
5. **F5**: Windows SDK identity は `WindowsSdkDir` + `WindowsSDKVersion` 環境変数
   （vcvarsall が設定 = MSVC が実際に使用する選択）を真実とし、
   **SDK include tree の directory fingerprint**（4,771 files / stat-walk 0.05 秒 /
   in-place update を mtime 変化で検出）を stamp に追加する方案を確定。
6. **F7**: codepage contract（I3-4-C）は一切退行しないことを fixture で再確認した。

---

## F0 — baseline / preservation

`evidence/D162-2I3/i3_4_f_f0_baseline.txt`:
- `.ninja_deps` / `.ninja_log` / rules.ninja / impl-{Release,Debug,RelWithDebInfo}.ninja /
  `.build_identity` / CMakeCache.txt / build.ninja — **E0 凍結ハッシュと完全一致（UNCHANGED）**
- 10 affected objs — E0 保存物と同一ハッシュ（SAME-GEN、clean rebuild なし）
- stamp: codepage=932 / prefix_sha256=b84b3da87752fa51 / ninja 1.13.2 / MSVC 14.51.36231

## F1 — `#deps 0` 意味論の 4 状態分類（設計）

```text
DEPS_VALID              #deps > 0（ninja が非システム依存を記録済み）
                        → PASS（build graph が transitive 真実を保持）

ZERO_DEPS_SYSTEM_ONLY   #deps == 0 かつ source-graph transitive closure に
                        非システムヘッダが 1 つも無い（unresolved も無い）
                        → ninja 設計上 EXPECTED。test-only scope では WARN+ALLOW

ZERO_DEPS_SUSPICIOUS    #deps == 0 かつ closure に非システムヘッダが存在
                        （prefix mismatch / DepsLog 損失 / stale のいずれか）
                        → FAIL-CLOSED（I3-4-C の事故クラスそのもの）

ZERO_DEPS_UNRESOLVED    #deps == 0 かつ closure 計算が解決不能な include を含む
                        → FAIL-CLOSED（証明不能なものは許可しない）
```

`#deps 0` という値は分類の入力であって許可根拠ではない。許可根拠は
「**closure によって非システム依存の不在が証明された**」ことのみ。

## F2 — TU→project dependency 判定方法（4 ソース比較）

| 優先 | ソース | 役割 | 実測での限界 |
| --- | --- | --- | --- |
| 1 | `impl-<Config>.ninja` edge 変数 | `-I` 探索 dirs の真実（target 別に異なる） | なし（edge ごとに存在） |
| 2 | `compile_commands.json` | 同値の別表現 | 同一 .cpp が複数 target に属するとき basename key では衝突する（full path key 必須・実測で 1 回事故） |
| 3 | source `#include` 走査 | direct include の抽出 | **transitive は別途 closure 計算が必須**・プリプロセッサ非対応のため条件依存 include を過大に含む（過大方向＝安全側） |
| 4 | `ninja -t deps` / DepsLog | ndeps>0 のときの transitive 真実 | `#deps 0` TU には真実が無い（だから F1 の closure が必要） |

探索順序の実装仕様（MSVC 規約）:
- quoted include: 当該ファイルの dir → `-I` dirs
- angle include: `-I` dirs → INCLUDE env（vcvarsall 由来の SDK/MSVC roots）

**検証結果（Release 327 TUs）**: エンジン closure と DepsLog の差分は
(a) エンジン過大: `#if` プラットフォーム条件内ヘッダ（例: `juce_Audio_ios.h`）1,273 件
    ＋ JuceHeader.h 4 件 — プリプロセッサ非対応による過大 approximation（安全側）
(b) DepsLog 過大: JUCE のマクロ集約 include でしか到達しないヘッダ 119,477 件
    （例: `juce_IIRFilter.h` 382 TUs 分）— 正規の動作（ninja は実際に開いた全ヘッダを記録）
この 2 方向の差分は **`#deps 0` TU の分類に影響しない**: 分類器は
(i) ndeps>0 では DepsLog をそのまま真実とし、
(ii) ndeps==0 では closure が空で unresolved が無い場合のみ EXPECTED とし、
(iii) それ以外は常に fail-closed に倒すため。

## F3 — 10 TU 再分類（`i3_4_f_f2_f3_provenance.txt`）

| TU | direct project include | transitive project include | system-only | target | #deps |
| --- | ---: | ---: | --- | --- | ---: |
| ObservePathSingleSourceTests | 0 | 0 | **YES** | test exe 専用 | 0 |
| OverlapAuthoritySingularTests | 0 | 0 | **YES** | 〃 | 0 |
| CrossfadeExecutorLocalContractTests | 0 | 0 | **YES** | 〃 | 0 |
| RebuildAdmissionRegressionTests | 0 | 0 | **YES** | 〃 | 0 |
| RuntimeWorldAuthorityProjectionTests | 0 | 0 | **YES** | 〃 | 0 |
| GainStagingContractTests | 0 | 0 | **YES** | 〃 | 0 |
| EQProcessorMaxGainTests | 0 | 0 | **YES** | 〃 | 0 |
| BuildInputSemanticContractTests | 0 | 0 | **YES** | 〃 | 0 |
| EQAnalysisUnitTests | 0 | 0 | **YES** | 〃 | 0 |
| EQBoundExcessBenchmark | 0 | 0 | **YES** | 〃 | 0 |
| （control）juce_core_CompilationTime.cpp | 0 | 0 | **YES**（#include 皆無） | JUCE bundle | 0 |

根拠は source（`#include <stdexcept>` 等のみ・census 済）と build graph（DepsLog ndeps=0・
link edge は自テスト exe のみ）の**両方**。I3-4-E の結論を実証的に再確認した。

## F4 / F7 — adversarial fixture matrix（`i3_4_f_f4_fixture_results.txt`）

| Case | 構成 | #deps | 分類器判定 | 期待 | 結果 |
| --- | --- | ---: | --- | --- | --- |
| A | system-only TU・prefix 一致（CP932） | 0 | ZERO_DEPS_SYSTEM_ONLY(WARN+ALLOW) | WARN+ALLOW | **MATCH** |
| B | project 直 include（P1.h）・一致 | 2 | DEPS_VALID(PASS) | PASS | **MATCH** |
| C | project transitive include（CH1→CH2）・一致 | 2 | DEPS_VALID(PASS) | PASS | **MATCH** |
| D | **project TU + prefix mismatch（baked CP932 vs console CP65001）** | 0 | **ZERO_DEPS_SUSPICIOUS(FAIL)** | MUST FAIL | **MATCH** |
| E | project TU 正常 compile（ndeps=1）→ CP65001 再 compile（ndeps=0）→ **header 内容変更（cpp 不変）** | 0 | （ninja: **no work to do** — obj は旧 VAL のまま永続 stale） | MUST FAIL | **hazard 実証** |

- **Case D が必須という条件を成立**：classifier はエンジン closure（P1.h+P2.h）で
  非システム依存の存在を検出し、mismatch による `#deps 0` を**許可しない**。
- **Case E**: D162-2 の事故機序そのもの（stale obj が header 変更を永久に無視）を
  fixture で再現。`#deps 0` project TU に対する fail-closed の存在意義を再証明した
  （I3-4-D gate が 92 obj を捕まえたのと同じ保護）。
- **F11 非退行**: Case B/C（codepage 一致時）は正しく transitive deps を記録し
  DEPS_VALID となる — codepage contract の正系は無傷。

## F5 — Windows SDK identity（`i3_4_f_f5_sdk_identity.txt`）

| 候補 | 実測 | 採否 |
| --- | --- | --- |
| CMakeCache `CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION` / `CMAKE_SYSTEM_VERSION` | **absent**（Ninja+cl では記録されない） | ✗ 情報なし |
| registry `HKLM\...\Microsoft SDKs\Windows\v10.0` | インストール済み一覧のみで選択を反映しない | 補助のみ |
| `WindowsSdkDir` / `WindowsSDKVersion` / `UCRTVersion` 環境変数 | **`C:\Program Files (x86)\Windows Kits\10\` / `10.0.26100.0`**（vcvarsall が選択。INCLUDE env の SDK 5 dirs と完全一致＝MSVC が実際に使用する SDK） | **✓ 採用（一次）** |
| cl `/showIncludes` 出力中の SDK パス | `...\Windows Kits\10\include\10.0.26100.0\ucrt\corecrt.h` 等 — **コンパイラが実際に開いた SDK を attestation** | **✓ 採用（裏取り検証用）** |
| SDK include tree directory fingerprint | 4,771 files / `(relpath,size,mtime_ns)` の SHA256 / **stat-walk 0.05 秒** | **✓ 採用（in-place update 検出）** |

**I3-4-G stamp 拡張スキーマ（提案・実装は次フェーズ）**:

```text
windows_sdk_dir:                C:\Program Files (x86)\Windows Kits\10\
windows_sdk_version:            10.0.26100.0        (WindowsSDKVersion env、trailing \ 正規化)
windows_sdk_include_fingerprint: <sha256[:16]>       (ucrt/um/shared/winrt/cppwinrt の
                                                     relpath|size|mtime_ns manifest、0.05s)
windows_sdk_fingerprint_source: environment          (environment | compile-output)
```

fingerprint は **SDK の in-place servicing（バージョン番号不変のヘッダ更新）を
mtime 変化として検出する**。コスト実測 0.05 秒（SDK）+ 0.02 秒（MSVC include）で
gate の --check に組み込み可能。取得は build.bat の vcvarsall 呼出し後の
`%WindowsSDKVersion%` 環境変数から行う（gate は build.bat から起動されるため env が生存）。

## F6 — whitelist scope 契約（提案・実装は次フェーズ）

```text
Production object（ConvoPeq.exe link 集合に属する obj）:
    #deps 0 → ALWAYS FAIL（例外なし）

Test object（テスト exe 専用 target の obj）:
    #deps 0
      ├─ ZERO_DEPS_SYSTEM_ONLY（closure 証明済み）→ WARN + ALLOW
      ├─ ZERO_DEPS_UNRESOLVED                     → FAIL
      └─ ZERO_DEPS_SUSPICIOUS                     → FAIL

Third-party / JUCE / resource artifacts:  gate scope 外（D4 維持）
```

> 許可の根拠は「test-only だから」ではなく
> **「test-only かつ closure により非システム依存不在が証明されたから」**。
> 判定は stamp の codepage/prefix hash 照合（COHERENCE-4）の**後**にのみ実行される
> （identity 不一致時は分類以前に fail）。

## F8 — gate 実装仕様（設計のみ・コード変更なし）

```text
build_identity_gate.py v2 フロー（現行の上に分類層を追加）:

  identity validation（現行どおり stamp 照合 → 不一致は即 fail）
        ↓
  deps extraction（現行: ninja -t deps パース）
        ↓
  zero-deps classification（新設）
      obj ごとに:
        production link set に属する？ ──yes→ #deps 0 は FAIL（例外なし）
                │no
        TU の transitive closure 計算（impl edge の -I + INCLUDE env）
                │
        ┌───────┴────────┐
   nonSystem>0 or     nonSystem==0
   unresolved>0       and unresolved==0
        │                 │
      FAIL           WARN + ALLOW
                     ( "[WARN] system-only TU (no project dependency): <obj>" )

  出力集計: FAIL が 1 件でも → rc=3（現行どおり fail-closed・自動 clean なし）
```

診断モードの評価:

| モード | 評価 |
| --- | --- |
| `--check`（現行） | 維持。fail-closed 性を犠牲にしない |
| `--explain-zero-deps` | **採用価値あり**。classification の根拠（closure 一覧・unresolved 一覧・判定理由）を標準出力する読み取り専用モード。rc は 0 固定（診断は分岐しない） |
| `--strict` | **採用しない**。production `#deps 0` の FAIL は常に厳格であり、モード分岐は意味論を二重化するだけ。system-only WARN の抑制（`--quiet-warn`）も見送り — WARN は見えるべき情報 |

パフォーマンス: 分類は `#deps 0` の obj のみで閉包計算するため通常 build では
10 TU × closure（小）＋ SDK/MSVC fingerprint 0.05+0.02 秒 — 現行 gate（秒単位）と同程度。

## F9 — upstream issue

I3-4-E の結論を維持し**保留**。優先順位は (1) gate semantics 正当化（本 F）→
(2) SDK provenance 閉包（本 F5 設計）→ (3) codepage protection 維持 →
(4) upstream design discussion（任意・blocker ではない）。

## F10 — GO/NO-GO 対合（ユーザー指定 F1-F12）

| ID | 条件 | 判定 |
| --- | --- | --- |
| F1 | `#deps 0` semantic classification 確立 | **PASS**（4 状態モデル・F1 節） |
| F2 | TU→project dependency 判定方法確立 | **PASS**（4 ソース比較 + closure engine・F2 節） |
| F3 | 10 TU 全件再分類 | **PASS**（全件 SYSTEM_ONLY・両 graph 一致・control group あり） |
| F4 | system-only positive fixture | **PASS**（Case A → WARN+ALLOW） |
| F5 | project-header positive fixture | **PASS**（Case B/C → DEPS_VALID・transitive 含む） |
| F6 | prefix mismatch → `#deps 0` → FAIL | **PASS**（Case D → SUSPICIOUS(FAIL)） |
| F7 | transitive project dependency の検証 | **PASS**（Case C + DepsLog 119k 件の実ビルド集計） |
| F8 | Windows SDK identity の取得方法確立 | **PASS**（env 真実 + fingerprint・0.05s・F5 節） |
| F9 | production `#deps 0` は fail-closed 維持 | **PASS**（F6 契約: ALWAYS FAIL・実装は現行 gate のまま維持中） |
| F10 | test-only whitelist の境界定義 | **PASS**（F6 契約: 許可根拠=closure 証明・test-only は必要条件に過ぎない） |
| F11 | I3-4-C の codepage protection 非退行 | **PASS**（Case B/C/D/E で正系と異系の両方を検証・stamp 照合は先行して必ず実行） |
| F12 | production/test/CMake source 変更 0 | **PASS**（git: 本監査で source 変更なし・fixture は %TEMP% のみ・build-diag は読み取りのみ） |

**総合判定 = I3-4-F GO（12/12）。**
次は **I3-4-G（whitelist 実装 + windows_sdk stamp 実装）** — ユーザーの明示 GO 後に限る。

## 添付 evidence（evidence/D162-2I3/）

```text
I3_4_F_ZERO_DEPS_CLASSIFICATION_SDK_PROVENANCE_AUDIT.md   本書
i3_4_f_f0_baseline.txt                                     F0 凍結一致検証
i3_4_f_f2_f3_provenance.txt                                F2 検証 + F3 10 TU 再分類
i3_4_f_f4_fixture_results.txt                              F4/F7 adversarial matrix
i3_4_f_f5_sdk_identity.txt                                 F5 SDK identity 実測
fixture: %TEMP%/i3_4_f_f4_{A,B,C,D}, i3_4_f_f4d_e_neg, i3_4_f_f4b_d ほか（再実行可）
```

## 変更ファイル

```text
src/**: 0 / tests: 0 / CMakeLists.txt: 0 / build.bat: 0 / src/tools/*: 0
（build-diag は ninja -t deps 読み取りのみ・clean rebuild なし・F0 で完全一致を記録）
```
