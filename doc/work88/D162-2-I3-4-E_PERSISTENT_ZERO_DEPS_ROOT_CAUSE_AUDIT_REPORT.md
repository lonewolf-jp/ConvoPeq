# D162-2-I3-4-E — Persistent Zero-Dependency / Ninja DepsLog Root-Cause Audit

```text
Date:            2026-09-05
Type:            read-only build-system forensic audit (E0-E10)
Production:      0 / Test: 0 / CMake source: 0 / build.bat: 0 / tools: 0
Baseline:        I3-4-D GO (fail-closed gate 維持) / build-diag stamp 0aeb22c+dirty
判定:            **分類 A — Ninja DepsLog 側の設計機序 confirmed（heuristic by-design、
                 かつ gate との契約衝突）。whitelist は test-only scope のみ条件付き許容、
                 gate 緩和はしない。**
```

---

## 0. 最終結論（要旨）

> **10 test TU の persistent `#deps 0` は事故状態ではない。ninja (v1.13.2) の
> `CLParser::IsSystemInclude` ヒューリスティックが、`/showIncludes` 出力のうち
> パスに "program files" / "microsoft visual studio" を含む行 — すなわち
> MSVC 標準ライブラリ・Windows SDK の全ヘッダ — を deps 記録から静かに除外する
> 設計であるため、プロジェクトヘッダを 1 つも include しない TU は
> 常に `#deps 0` として記録される。**
>
> 10 TU は全て `#include <stdexcept> <fstream> <sstream> <string> <filesystem>` 等の
> **システムヘッダのみ**の TU であり（プロジェクトヘッダ 0 件・実測）、
> cl 単体では 91-159 行の正しい CP932 prefix 行を出力するが、ninja は
> 全行を廃棄して空の deps entry を書く。ninja は `#deps 0` を VALID として扱い、
> 何も警告しない。
>
> 一方、I3-4-C で確定した codepage mismatch（configure と build の console codepage
> 差 → prefix bytes 不一致 → **プロジェクト include も含めて全部**消失）は
> 別個の第一メカニズムであり、過去の stale-obj 混在事故（D162-2）の実原因。
> 本件 10 TU は codepage を修正しても残る**第二の（by-design の）機序**である。

分類: **A（Ninja/DepsLog 側機序 confirmed）**。ただし「1.13.2 固有のバグ」ではなく
2015 年頃から存在する意図的ヒューリスティック（clparser.cc に `TODO: this is a
heuristic` の注記あり）。ninja upstream issue 化は **条件付き GO**（§E8）。

**I3-4-D の fail-closed gate は本監査を通じて一切変更していない。**

---

## E0 — 現状固定（clean なし・preservation 完了）

```text
evidence/D162-2I3/i3_4_e_e0_freeze/
  ninja_files/  build.ninja / common.ninja / build-Release.ninja / build-Debug.ninja /
                build-RelWithDebInfo.ninja / CMakeFiles/rules.ninja / impl-{Release,Debug,
                RelWithDebInfo}.ninja / .build_identity / CMakeCache.txt / .ninja_deps
                (8,622,200 B, 2026-09-05 20:09) / .ninja_log (324,844 B)
  affected_objs/  10 TU の .obj 実物 (sha256[:16] 記録)
  affected_srcs/  10 TU の .cpp 実物
evidence/D162-2I3/i3_4_e_e0_snapshot.txt   バージョン・ハッシュ・mtime 一式
```

| 項目 | 値 |
| --- | --- |
| generator / build dir | Ninja Multi-Config / `build-diag`（`build.bat` BUILD_ROOT=build、diag 用に複製利用） |
| ninja | **1.13.2**（`.ninja_log` は v7 形式: `start end mtime output cmdhash`） |
| cmake / compiler | 4.4.3 / MSVC 14.51.36231 (19.51) |
| stamp | console_codepage=932 / msvc_deps_prefix_sha256=b84b3da87752fa51 |
| msvc_deps_prefix | CP932 bytes `83 81 83 82: 83C 83 93 83N 83 8b 81[ 83h 20 83t 83@ 83C 83 8b: 20 20`（「メモ: インクルード ファイル:  」30 bytes、rules.ninja に 1 件のみ・重複定義なし） |

## E1 — 10 TU provenance 完全化（`i3_4_e_e1_inventory.txt` / `i3_4_e_e1_include_census.txt`）

| TU | ルール | deps= | /showIncludes | command hash | include 数（system のみ） |
| --- | --- | --- | --- | --- | --- |
| ObservePathSingleSourceTests | CXX_COMPILER__*_unscanned_Release | msvc | ✓ | 9d2d0856e97aac0a | 5 |
| OverlapAuthoritySingularTests | 同上 | msvc | ✓ | b06d338b9bd12903 | 5 |
| CrossfadeExecutorLocalContractTests | 同上 | msvc | ✓ | 8c88bec3632cac5e | 5 |
| RebuildAdmissionRegressionTests | 同上 | msvc | ✓ | 2de8070f2e1c671f | 7 |
| RuntimeWorldAuthorityProjectionTests | 同上 | msvc | ✓ | 342db181fcdc0bb5 | 6 |
| GainStagingContractTests | 同上 | msvc | ✓ | 553f26049bb34df | 5 |
| EQProcessorMaxGainTests | 同上 | msvc | ✓ | a71b2d4d424ea43b | 7 |
| BuildInputSemanticContractTests | 同上 | msvc | ✓ | a7fad46ce938be47 | 6 |
| EQAnalysisUnitTests | 同上 | msvc | ✓ | 648d4bdf5df29972 | 9 |
| EQBoundExcessBenchmark | 同上 | msvc | ✓ | 8dab82c44f05a136 | 14 |

- **command hash は 10 TU ですべて異なる**（D10-E の「hash 衝突クラス」仮説は棄却）。
- 10 TU の include は **プロジェクトヘッダ 0 件**、`<stdexcept> <fstream> <sstream>
  <string> <filesystem> <cmath> <iostream> ...` の system headers のみ。
- 同一 failure class 確認: Release/Debug/RelWithDebInfo **3 config 全部**で同一挙動
  （DepsLog 上 3 config 分の obj node がすべて 0）。
- 対比の良性クラス: `juce_core_CompilationTime.cpp.obj`（#include 皆無 TU）も
  16+ 個常に `#deps 0` — 同一機序の自明なケース。

## E2 — cl 単体 vs Ninja 完全分離（決定実験）

%TEMP% fixture（影響 TU と健全 TU を同一 ninja プロセス・同一 prefix・同一ルールで並置）:

```text
E2-A cl.exe 単体（vcvars64 + chcp 932、exact command、stdout を raw bytes 保存）:
  affected (ObservePath):  stdout 20,974 B / 161 行 / prefix 行 159 / stderr 0 行
                           prefix bytes は rules.ninja 焼き込みと byte-for-byte 一致
                           非prefix 行は 1 行のみ（ソース名エコー）
  healthy  (D8_1):         stdout 11,671 B / 93 行 / prefix 行 91 / stderr 0 行

E2-B 同一 fixture を ninja 実行 → ninja -t deps:
  healthy.obj : #deps 1   = プロジェクトヘッダ ISRAuthorityClass.h のみ記録
  affected.obj: #deps 0   = 159 行すべて廃棄 → 空リストで記録（REPRODUCED）
```

**compiler output generation は正常、DepsLog ingestion が行を選別して廃棄 — 確定。**

## E3 — codepage 仮説の再検証

```text
CP932 → cl 単体:    prefix 正常 159 行（§E2-A）
CP932 → ninja:      #deps 0（fixture・影響 TU）
CP65001 → ninja:    #deps 0（同一影響 TU、build.bat 相当の console）
同一 console 内の対照 healthy TU: #deps 1〜3 が常に記録される
```

→ **「cl 単体では正しいのに Ninja 内だけ #deps 0」は codepage に依存せず再現**。
I3-4-C の codepage mismatch が説明するのは過去の stale-object generation（第一メカニズム）
であり、本 10 TU は別機序 — E 監査の冒頭指示どおり二者を混同しないことを実証。

## E4 — DepsLog binary forensic（`i3_4_e_e4_depslog_forensic.txt`）

`.ninja_deps` を v4 形式で完全パース（3,459 path records / 4,027 deps records /
全 8,622,200 bytes 消費・record checksum まで整合）:

```text
record format (v4):
  path record: [int32 size][path(size-4 bytes, 4B alignment padding)][int32 checksum = ~id]
  deps record: [int32 size | 0x80000000][int32 out_id][int64 mtime][int32 dep_id × n]
```

- 10 TU × 3 config の **全 26 node に deps record が存在し、初回構築時から
  常に ndeps=0**（過去に正常な deps list が存在した痕跡は一度もない）。
- `.ninja_log` v7 に各 edge の実行記録あり（cmdhash は 10 TU で相互に異なる）→
  「edge が走って空リストを書いた」= D10-E の観測を正として確定。
- deps record 内の dep id 衝突・別 edge との混線・path id エイリアスは **0 件**
  （corruption / keying 異常なし）。
- 健全対比: `ConvoPeq.dir/Release/.../AudioEngine.Timer.cpp.obj` は ndeps=751、
  その 751 件は **`../JUCE/...` 等 build-tree 相対パスのみ** —
  `C:\Program Files\...` は **0 件**（後述の機序で完全一致する結果）。

## E5 — restat 切り分け

```text
E5-A restat = 1 fixture:  compile 実行 → #deps 0 → 2nd build "no work to do"
E5-B restat なし fixture: compile 実行 → #deps 0 → 2nd build "no work to do"
```

→ **判定表どおり「restat ON/OFF 両方 #deps 0 = restat 仮説棄却」**。
（参考: rules.ninja の `CXX_DYNDEP__*` ルールに restat=1 があるが、10 TU は
unscanned ルールであり dyndep 経路に乗っていない。）

## E6 — command / edge identity 相関（`i3_4_e_e6_identity_correlation.txt`）

- FLAGS/DEFINES 署名で Release .cpp.obj 全 492 edge を分類:
  - 影響 10 TU は 2 つの署名に分散（UNICODE+`/utf-8` 重複あり/なし）
  - **同一署名の健全 TU が存在**（ISRRuntimeIdentityGeneratorsTests ndeps=3、
    RuntimePublicationCoordinatorTests ndeps=8、D8_1 ndeps=1、DeferredDeletionQueueReclaimTests ndeps=4 など）
  → FLAGS/DEFINES/ルール差異は原因ではない
- 決定的な相関は **include クラス**: ndeps>0 の全 obj の deps は
  build-tree 相対パスのみ。ndeps=0 の obj は「システムヘッダのみの TU」
  （10 test TU + CompilationTime）に完全一致。

## E7 — 最小再現 fixture（一連のプロップ実験・全て %TEMP% 限定）

| Prop | 構成 | 結果 |
| --- | --- | --- |
| bisect | `<cassert>` `<vector>` `<filesystem>` `<format>` … **18 種の STL ヘッダを 1 個ずつ** include する最小 TU ×18 | **18/18 が `#deps 0`**（ヘッダの内容・種類に依存しない） |
| healthy 對照 | D8_1（project 1 include + system 90） | `#deps 1`（project 1 件のみ記録、system 90 は廃棄） |
| SameDrive | `%TEMP%` 直下の自作ヘッダ（quoted include） | `#deps 1` 記録される（cwd 側の相対化は成功） |
| CP65001 | 影響 TU を CP65001 console で再 compile | `#deps 0`（codepage 非依存の再確認） |
| full-flags serial | 影響 1 + 健全 2 を本物の DEFINES/FLAGS で -j1 | aff=#deps 0 / heal=#deps 1,3（並列性・追加 flags の影響なし） |
| DepsLog direct | fixture `.ninja_deps` を binary パース | `affected.obj` = ndeps 0 の正規 record（mtime は出力 obj の mtime と一致） |

**最小構成（`int main(){return 0;}` + `#include <cassert>` 1 行 + deps=msvc ルール）で
完全再現**。原因は TU が参照するヘッダの**所在パス**のみで決まる。

## 機序の最終確定（ninja v1.13.2 上流ソース照合）

`src/clparser.cc`（tag v1.13.2 から直接取得・`i3_4_e_e8_mechanism_note.txt` に全文引用）:

```cpp
// static
bool CLParser::IsSystemInclude(string path) {
  transform(path.begin(), path.end(), path.begin(), ToLowerASCII);
  // TODO: this is a heuristic, perhaps there's a better way?
  return (path.find("program files") != string::npos ||
          path.find("microsoft visual studio") != string::npos);
}
...
if (!IsSystemInclude(normalized))
  includes_.insert(normalized);
```

- `IncludesNormalize::Normalize` は include パスを cwd（build dir）基準で相対化する
  （SameDrive の場合）。build tree 内ヘッダは `../...` 相対パスになり記録される。
- その後 `IsSystemInclude` が「program files / microsoft visual studio を含むパス」を
  システムヘッダとみなし**廃棄**する。MSVC 標準ライブラリ・Windows SDK は
  いずれも `C:\Program Files...` 配下のため、**例外なく廃棄される**。
- 上流の設計意図: システムヘッダは build tree の寿命中に不変とみなす
  （DepsLog 肥大化防止）。結果として「システムヘッダのみを含む TU」は
  `#deps 0` が**正しい・意図された記録**になる。
- これが ConvoPeq の 241 生産 obj（JUCE/proj ヘッダを含む → ndeps 738-753）と
  10 test TU（system のみ → ndeps 0）の差分を完全に説明する。

## E8 — upstream issue 判定

| GO 条件 | 判定 |
| --- | --- |
| 1. cl standalone → 正常 showIncludes | ✓ E2-A |
| 2. Ninja → #deps 0 | ✓ E2-B |
| 3. exact command 正しい | ✓ E1（deps=msvc + /showIncludes + prefix 一致） |
| 4. codepage mismatch では説明不能 | ✓ E3 |
| 5. minimal fixture で再現 | ✓ E7（1 ヘッダ TU） |
| 6. clean build でも再現 | ✓ E4（初回構築から常に 0）+ fixture |
| 7. restat 等既知要因の切り分け | ✓ E5 / E6 / hash 衝突棄却 |
| 8. 1.13.2 固有挙動の証拠 | **✗ — 逆に固有ではなく設計挙動と確定** |

→ **判定: 「Ninja bug 認定」は不可**。機序は documented-by-code の設計
（clparser.cc 内 TODO 付きヒューリスティック、2015 年頃から存在）。
upstream issue は **bug ではなく design-discussion として条件付き GO**:
提案テーマ =「deps=msvc で system-only TU が #deps 0 を記録することの
ドキュメント化」と「IsSystemInclude 文字列ヒューリスティックの硬コード
（`C:\Program Files` 以外のインストール先 — 例: `X:\libs`、`D:\Program Files` —
では逆に SDK ヘッダが全件記録される非対称性）」。提出は LOE 小・ユーザー判断に委ねる。
既存上流 issue は #1766 / #1669（locale クラス）まで確認し、本機序の重複報告は無し。

## E9 — 10 test-only TU の whitelist 可否判定

契約の 4 条件を対合:

| 条件 | 判定 |
| --- | --- |
| test-only（production exe に絶対 link されない） | **✓ 実証** — `ConvoPeq.exe` link edge（134 .obj inputs）に 10 TU は含まれない。各 TU は自テスト exe 専用 |
| header dependency failure が production artifact に伝播しない | **✓（条件付き）** — 残る依存は VS/SDK ヘッダのみ。VS update は identity stamp（compiler_version）で COHERENCE-4 検出→clean 要求。**残留リスク: Windows SDK のみの in-place update は stamp 項目になく未検出** |
| 意図的に deps-less である設計上の理由 | **✗ ない**（偶然の stdlib-only TU。CompilationTime は意図的 no-include） |
| `#deps 0` の意味の再定義 | **✓** — ninja の意味論では「deps 追跡の結果、記録対象が無い」。I3-4-C の意味（prefix mismatch で全依存消失）とは異なる状態を同じ値が表す、という**表現力の欠如**が gate との衝突の正体 |

**判定: test-only scope に限り whitelist-with-warning は妥当（条件付き GO）。**
ただし本監査は実装をしない（実装禁止契約）。I3-4-F 以降の推奨実装:

1. gate の `RELEVANT_OBJ_RE` scope は維持したまま、`#deps 0` の obj について
   **エッジの TU がプロジェクトヘッダを含むかを静的判定**（`#include` 走査 or
   impl-*.ninja の TU↔target 対応表）し、「project-include なし TU」のみ
   `#deps 0 = EXPECTED(system-only)` として警告付き通過。リスト外は fail-closed 維持。
2. 残留リスク（SDK in-place update）は stamp に `windows_sdk_version` を追加する
   ことで閉じられる（I3-4-F 提案、E0 stamp の拡張）。
3. 代替案（非推奨）: 10 TU にダミーの project header include を 1 行追加する
   （test source 変更。deps≥1 が記録され gate は無修正で通るが、
   gate の意味論を崩さないために whitelist 方が誠実）。

## E10 — 最終分類

```text
A: Ninja/DepsLog defect confirmed  ← これ（ただし「defect」は設計ヒューリスティック
                                      と gate 契約の衝突を指す。誤動作ではない）
B: build configuration defect      — ならず（同一 config の健全 TU が実在）
C: compiler/showIncludes defect    — ならず（cl 出力は byte-perfect で正しい）
D: root cause unresolved           — ならず
```

**D ではないため、gate 緩和は whitelist 手続き（E9 の条件）を経由してのみ行う。
本監査による実装変更は 0。**

---

## I3-4-E GO 条件対合

| Gate | 条件 | 判定 |
| --- | --- | --- |
| E1 | 10 TU 全件 provenance 完全化 | **PASS**（3 config・hash・mtimes・include 実測） |
| E2 | cl単体/Ninja経路分離 | **PASS**（E2-A/E2-B 決定実験） |
| E3 | codepage仮説再検証 | **PASS**（CP932/65001 とも再現・第一機序と分離） |
| E4 | DepsLog entry構造確定 | **PASS**（v4 binary 完全パース・履歴全件） |
| E5 | restat切り分け | **PASS**（棄却） |
| E6 | command/edge identity切り分け | **PASS**（hash 衝突・flags 差異棄却） |
| E7 | 最小fixture再現 | **PASS**（1-include TU で再現） |
| E8 | upstream issue可否判定 | **PASS**（bug 認定不可・design-discussion 条件付き GO） |
| E9 | whitelist可否判定 | **PASS**（test-only 条件付き GO・実装は次フェーズ） |
| E10 | root cause分類 | **PASS（分類 A）** |

## 添付 evidence（evidence/D162-2I3/）

```text
I3_4_E_PERSISTENT_ZERO_DEPS_ROOT_CAUSE_AUDIT.md        本書
i3_4_e_e0_snapshot.txt                                  E0 バージョン/ハッシュ/mtime
i3_4_e_e0_freeze/                                       E0 preservation 一式
i3_4_e_e1_inventory.txt                                 E1 10 TU provenance
i3_4_e_e1_include_census.txt                            E1 include 分類（system-only 実証）
i3_4_e_e1_ninja_deps_{Release,Debug}.txt                E1 ninja -t deps 全量
i3_4_e_e4_depslog_forensic.txt                          E4 DepsLog binary forensic
i3_4_e_e6_identity_correlation.txt                      E6 FLAGS/DEFINES 署名相関
i3_4_e_e8_mechanism_note.txt                            E8 clparser.cc 機序引用
fixture: %TEMP%/i3_4_e_e2b_fixture ほか 6 契約（再実行可）
```

## 変更ファイル

```text
src/**: 0 / tests: 0 / CMakeLists.txt: 0 / build.bat: 0 / src/tools/*: 0
（新規は evidence と doc/work88 の報告書のみ。build-diag は fixture 実験での
  正規 compile 2 回（Release/Debug deps 取得用 ninja -t deps は読み取り専用）以外
  触っていない。%TEMP% に 6 契約の実験 fixture。）
```
