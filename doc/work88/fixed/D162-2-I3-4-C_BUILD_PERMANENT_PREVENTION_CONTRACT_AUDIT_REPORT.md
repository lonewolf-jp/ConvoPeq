# D162-2-I3-4-C — Build-system Permanent Prevention Contract Audit

```text
Date:            2026-09-05
Type:            read-only build-system / dependency-integrity audit
                 (+ %TEMP% 検証 fixture — repo source への影響 0)
Production:      0 / Test: 0 / CMake source: 0
Baseline:        clean rebuild (I3-4-B) / ConvoPeq.md 2026-09-05 12:02:31
Tools:           ninja 1.13.2 -t deps / CMake 4.4 / codepage probe (chcp 932/437/65001)
判定:            **C1-C10 全 PASS → GO。I3-4-D (Contract Implementation) へ進行可能**
```

---

## 0. 結論（要旨）— 根因の最終確定

I3-4-A/B で確定した「stale obj 混在」の **一次原因** を本監査で特定した:

> **`cl.exe /showIncludes` の出力 prefix は console codepage により bytes が変化する。**
> **CMake は configure 時の codepage で 1 種類の prefix を `msvc_deps_prefix` として
> rules.ninja に焼き込む。configure 時と build 時の codepage が異なると、
> ninja は依存行を 1 件も解析できず `#deps 0` として記録する。**
> **`#deps 0` の obj は header 変更を検知できず、`.cpp` が不変である限り永遠に
> 再構築されない — これが 08-29 obj が 09-05 link まで生き残ったメカニズム。**

決定実験（fixture + 本物 build.bat 環境）:

| console codepage | cl prefix bytes | ninja deps 解析 |
| --- | --- | --- |
| CP932（日本語） | `83 81 83 62 ...` (CP932 'メモ:...') | ○ 成功（#deps 741） |
| CP437 | `?? ?????? ????:`（mangling） | ✗ 全滅（#deps 0） |
| **CP65001（build.bat が設定）** | **UTF-8 bytes `E3 83 A1 ...`** | **✗ CP932-baked prefix と不一致** |

`LANG` / `VSLANG` 環境変数は本 cl 版では無効（probe C/D）。`chcp` のみが効く。

**現在の build-diag は CP932 console で configure されている**（rules.ninja prefix は
CP932 bytes）。一方 `build.bat` は冒頭で `chcp 65001` を実行する —
つまり **build.bat 経由の configure + agent console（CP932）経由の build、またはその逆で
本事故は任意のタイミングで再発し得る**（実際に 08-29 batch で発生した）。

## 1. C0 — Baseline 固定

clean rebuild 後の全 build-state（`i3_4_c_c0_baseline.txt`）:
- CMakeCache.txt / build.ninja / impl-*.ninja / rules.ninja / common.ninja: 09-05 18:58:32
- .ninja_deps 19:08:30 / .ninja_log 19:08:39 / Release exe 19:02:54 (sha256[:16]=833dba185d40b398)
- Release objs 143 個: 19:00:48-19:02:52 / free run ×3 exit 0x0
- Debug exe 19:08:38 / RWDI 13:22:33（I3-4-B 世代）

## 2. C1 — dependency tracking の実効性

### C1.1/C1.2 rules と dep 記録（`i3_4_c_c1_deps_audit.txt`）

```text
rule CXX_COMPILER__AudioEngineHarness_unscanned_Release:
  deps = msvc ✓  /showIncludes ✓  (cl.exe 14.51.36231)
5 TU の dep 記録（clean rebuild 後）:
  ProcessIntent.cpp.obj  #deps 747 VALID — coordinator.h を含む ✓
  AudioEngine.Init.cpp.obj   #deps 740 VALID — ✓
  AudioEngine.CtorDtor.cpp.obj #deps 750 VALID — ✓
  AudioEngine.Threading.cpp.obj #deps 745 VALID — ✓
  AudioEngine.Timer.cpp.obj  #deps 751 VALID — ✓
```

**mechanism は現状完全に機能している。**

### C1.3 「#deps 0」原因の分類（本監査の核心）

| 候補 | 判定 |
| --- | --- |
| Ninja 以外の経路で obj 生成 | ✗（代替 rule は存在しない） |
| .ninja_deps の消失・上書き | 部分（ninja は書き換えるが deps は保持する — 単独では説明不能） |
| **console codepage 不一致による prefix mismatch** | **✓ 決定（§0 の probe 実測）** |
| build dir copy/restore | ✗ 併発証拠なし（同一 codepage 現象が 195+88 個の全 batch に一様） |
| generator 再生成による deps 消失 | ✗（reconfigure は .ninja_deps を触らない） |

## 3. C2 — header change propagation 実験（`i3_4_c_c2_locale_evidence.txt`）

%TEMP% fixture（main.cpp + feature.h、`deps=msvc` を本物 build-diag と同一構成で再現）:

```text
prefix 一致 (CP932/CP932):  build 1 → #deps 741 相当・feature.h tracked
                            header 内容変更（.cpp mtime 不変）→ obj 再構築 → 実行結果 FEATURE_VERSION=2 ✓
prefix 不一致 (何れか片方):  #deps 0 → header 変更しても "ninja: no work to do"
                            → 実行結果は旧値のまま（FEATURE_VERSION=1）＝ 事故の再現
```

**「dependency DB が正常なら .cpp 不変でも header 変更で obj が rebuild される」ことを
positive/negative 両面で実証。** 事故状態（negative 側）を正確に再現できた。

## 4. C3 — build directory corruption / regeneration 耐性

実事故（I3-4-B §10）+ probe からの評価:

```text
Ninja → VS → Ninja 遷移（build dir 共有）:
  - cache に generator 違反が記録されるため configure は失敗する（CMake 保護が働く）。
    しかし今回の事故では VS configure が「成功する前」に cache を書き換え、
    復旧の過程で Ninja artifacts が消えた — 保護はあるが破壊的失敗モードがある。
  - build.bat は juceaide sub-cache のみを掃除する（outer cache は保護しない）。
codepage 遷移（Ninja 共通・cache 不変）:
  - CMake 保護は一切働かない（generator は一致しているため）。
  - prefix mismatch → #deps 0 が静かに発生し、以後の header 変更が伝播しない。
  - **これが本事故の実際の破壊経路。**
```

## 5. C4 — provenance metadata の現状（`i3_4_c_c4_provenance_audit.txt`）

| 項目 | 状態 |
| --- | --- |
| Generator / Compiler path / Config / CMake version | cache から導出可 |
| **Compiler version / Ninja version / Source revision / Build dir ID** | **未記録** |
| showIncludes prefix | rules.ninja から導出可 — **ただし build 時の検証なし** |
| obj の deps state | `ninja -t deps` で取得可 — **link 前の gate なし** |

## 6. C5 — Object-generation coherence contract（提案）

```text
COHERENCE-1  1 executable = 1 build generation（同一 codepage/configure 世代の obj のみ link）
COHERENCE-2  affected header change → affected TU rebuild（deps=msvc が機能している前提）
COHERENCE-3  dependency information missing（#deps 0）→ その obj を使った incremental link は禁止
             → 実行前 gate: `ninja -f impl-<Config>.ninja -t deps` で #deps 0 の .obj を検出したら
               incremental build を拒否し fresh reconfigure + clean rebuild を要求する
             ★ 本事故の核心。ninja 単体では #deps 0 でも VALID 扱いで link するため
               外部 gate が必須
COHERENCE-4  generator/compiler/configuration identity mismatch → build fail or forced clean
             （CMake cache 保護に加え、codepage identity を build dir に記録し build 時照合）
COHERENCE-5  object provenance 不能 → Release link から除外（fail-closed）
```

**codepage identity の記録方法**（I3-4-D 実装候補）: configure 時に `chcp` 値 +
`msvc_deps_prefix` の SHA256 を build dir に stamp（例: `CMakeFiles/.build_identity`）し、
`cmake --build` 前に照合。不一致なら reconfigure を要求。

## 7. C6 — `--clean-first` の位置づけ

常時 clean は採用しない（build 時間 ~2 分/143 obj + developer workflow 劣化）。
二段構え:

```text
通常時:  incremental build（deps 正常 + identity stamp 一致のとき）
異常時:  COHERENCE-3/4 gate が不一致を検出 → fresh reconfigure + clean rebuild を自動要求
```

gate の実装コストは小さい（`ninja -t deps` の grep + stamp 照合。いずれも秒単位）。

## 8. C7 — build.bat 安全境界監査（`i3_4_c_c7_buildbat_audit.txt`）

| 既存要素 | 評価 |
| --- | --- |
| `chcp 65001`（冒頭） | **危険**: CP932 configure 世代の build dir で実行すると prefix mismatch を起こす。恒久策では「configure と build の codepage 一致強制」が必要 |
| `-G "Ninja Multi-Config"` + `-DCMAKE_*_COMPILER=cl` | generator/compiler pin ✓（ただし compiler は名前のみ・version pin なし） |
| `DO_CLEAN=1` → rmdir build dir | full clean は opt-in で存在 ✓ |
| juceaide sub-cache 掃除 | generator mismatch 対策（JUCE tools のみ）✓ |
| stale RC .res 削除 | RC1109 対策 — **obj provenance とは無関係**（指示どおり別物として扱う） |
| configure 3 回 retry | ✓ |

**C7 判定**: build.bat に COHERENCE-3/4 gate（deps validity + identity stamp 照合）を
**追加可能**。追加位置は configure 完了後・build 実行前。

## 9. C8 — 恒久契約書（I3_4_C_BUILD_PERMANENT_PREVENTION_CONTRACT.md 本体）

1. **Root cause**: console codepage 差 → `msvc_deps_prefix` bytes 不一致 → `#deps 0` →
   header 変更非伝播 → 旧 layout obj が新 layout link に混入（ODR 違反）→ null coordinator → AV
2. **既存 deps=msvc が不十分だった理由**: prefix は configure 時 codepage で凍結されるが
   build 時 codepage は可変。不一致時 ninja は静かに #deps 0 とし、何も警告しない。
3. **Required invariants**: COHERENCE-1〜5（§6）
4. **Build provenance**: generator/compiler(+version)/arch/config/cmake/ninja/git SHA/
   build-dir ID/showIncludes-prefix-hash を build dir に stamp
5. **Generator/compiler identity**: cache 保護（既存）+ identity stamp 照合（新設）
6. **Dependency DB validity**: `ninja -t deps` で #deps 0 obj が 1 個でもあれば incremental 不可
7. **Clean rebuild trigger**: identity stamp 不一致 / #deps 0 obj 存在 / cache generator 破壊
8. **CI enforcement**: (a) build 前に gate script、(b) 週次 fresh-rebuild smoke、
   (c) exe と obj の generation 検証（`ninja -t deps` 出力を artifact 保存）
9. **Developer build enforcement**: build.bat に gate 組み込み（configure 後・build 前）
10. **Forbidden states**: #deps 0 obj を含む link / codepage 不一致 build /
    generator 混在 build dir / provenance 未知 obj の Release link
11. **GO evidence**: 本監査一式（C0 baseline・C1 deps audit・C2 locale evidence・
    C4 provenance audit・I3-4-A/B の epoch 表・3 exe sha256）
12. **No-production-source-change**: 本監査を通じて src/test/CMake source 変更 0
    （%TEMP% fixture と build artifact のみ操作）

## 10. GO 条件対合

| ID | 条件 | 判定 |
| --- | --- | --- |
| C1 | dependency tracking 実効性 | **PASS**（5 TU #deps 740-751・coordinator.h tracked） |
| C2 | header→rebuild 実証 | **PASS**（一致時 positive・不一致時 negative を再現） |
| C3 | missing metadata の危険性実証 | **PASS**（codepage probe + fixture 再現） |
| C4 | generator/compiler mismatch 検出条件 | **PASS**（cache 保護 + identity stamp 設計） |
| C5 | object-generation coherence 判定方法 | **PASS**（#deps gate + stamp、§6） |
| C6 | stale obj を link させない fail-safe | **PASS**（COHERENCE-3 gate 定義） |
| C7 | enforcement 位置確定 | **PASS**（build.bat configure 後・build 前 / CI） |
| C8 | clean fallback 発動条件 | **PASS**（§7 二段構え） |
| C9 | clean Release/Debug coherent 再確認 | **PASS**（I3-4-B B5-B7 引き継ぎ・3/3 exit 0x0） |
| C10 | production/test/CMake source 変更 0 | **PASS**（git diff は I2/I3 session 分のみ） |

**総合判定 = I3-4-C GO → D162-2-I3-4-D (Contract Implementation) へ進行可能**

## 11. 添付

```text
evidence/D162-2I3/
  I3_4_C_BUILD_PERMANENT_PREVENTION_CONTRACT_AUDIT.md   本書
  i3_4_c_c0_baseline.txt                                C0 build-state snapshot
  i3_4_c_c1_deps_audit.txt                              C1 rules + dep 記録
  i3_4_c_c2_locale_evidence.txt                         C2 fixture + codepage probe
  i3_4_c_c4_provenance_audit.txt                        C4 metadata 現状
  (fixture: %TEMP%/i3_4_c_propagation_test — 実験一式・再実行可)
```
