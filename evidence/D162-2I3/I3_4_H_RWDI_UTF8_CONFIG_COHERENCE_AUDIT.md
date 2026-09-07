# D162-2-I3-4-H — RelWithDebInfo `/utf-8` Configuration Coherence Audit

```text
Date:            2026-09-05
Type:            read-only audit (H0-H4) — CMakeLists.txt 実装（H5）は本監査 GO 後
                 production/test source 変更 0 / CMakeLists.txt 変更 0 / fixture は %TEMP% のみ
Baseline:        I3-4-G GO 12/12 / ConvoPeq.md 2026-09-05 12:02 (内部 Generated 12:02:18,
                 CMakeLists.txt 2026-08-26 20:40 より後に生成 → 現行ソースを反映した基準)
判定:            **根因確定（CMake semantics 欠落）+ fixture 完全再現 + 修正予行検証済み
                 → H1-H4 GO。H5（CMakeLists 最小修正）への進行を推奨。**
```

---

## 0. 要旨

> **根因（第1）: `CMakeLists.txt` は `set(CMAKE_CXX_FLAGS_RELEASE ... /utf-8 ...)`（L1519）
> と `set(CMAKE_CXX_FLAGS_DEBUG ... /utf-8 ...)`（L1524）を持つが、
> `CMAKE_CXX_FLAGS_RELWITHDEBINFO` の set 文が存在しない。**
> RWDI は CMake デフォルト `/Zi /O2 /Ob1 /DNDEBUG` にフォールバックし、
> config-level の `/utf-8` を欠く。
>
> **根因（第2）: RWDI の `/utf-8` 補完経路が target-level にしか無い**。32 target は
> `target_compile_options(... /utf-8)`（L174-1100 群）で補完されるが、
> ASan ループ（L1964 `if(ENABLE_ASAN)` 内の 29 target）は **`ENABLE_ASAN:BOOL=OFF`**
> のため不活性。その結果 **RWDI の 492 CXX edge のうち 9 edge（7 target）だけが
> `/utf-8` 無し**で compile され、UTF-8 (BOM 無し) ソースを CP932 解釈して
> C4819 → mojibake 構文エラー（`AtomicAccess.h(55)`）で失敗する。
>
> **ConvoPeq 固有の target 問題ではない**: 同一 CMakeLists・同一ソースで Release は
> diagnostics 0 で成功し、fixture（broken CMakeLists 複製）でも RWDI のみ失敗を再現。
> fixture の fixed 版（RWDI set 追加）では 3 config 全部成功 — **H5 修正の予行検証済み**。

---

## H0 — 現状固定（`i3_4_h_h0_snapshot.txt` + `i3_4_h_h0_freeze/`）

| 項目 | 値 |
| --- | --- |
| ConvoPeq.md | mtime 2026-09-05 12:02:31 / 内部 Generated 12:02:18 / CMakeLists.txt (08-26 20:40) より新しい → **現行** |
| CMakeLists.txt | 2,004 行・sha[:16]=5fbd56e1b58684ca（freeze 保存） |
| impl-RelWithDebInfo.ninja | sha[:16]=6b1a28745aaeb5c2（09-05 18:58 生成・Release/Debug と同時 → stale generation ではない） |
| CMAKE_CXX_FLAGS（cache 共通） | `/utf-8` を含まず・全 config 同一 |
| CMAKE_CXX_FLAGS_RELEASE (cache) | `/O2 /Ob2 /DNDEBUG`（**/utf-8 無し** — L1519 は非キャッシュ set で上書きされるため cache に反映されない。これが「cache に無いのに edge に有る」の説明） |
| CMAKE_CXX_FLAGS_DEBUG (cache) | `/Zi /Ob0 /Od /RTC1`（同上） |
| CMAKE_CXX_FLAGS_RELWITHDEBINFO (cache) | `/Zi /O2 /Ob1 /DNDEBUG`（= CMake デフォルトそのまま・**生成 edge と完全一致**） |
| stamp | schema 2・codepage 932 / prefix b84b3da8… / SDK 10.0.26100.0 / fingerprint 2c5e417e… |
| ENABLE_ASAN:BOOL | **OFF** |

## H1 — flag provenance 完全追跡（`i3_4_h_h1_flag_provenance.txt`）

**source level（CMakeLists.txt）**:

```text
L1497-1509  target_compile_options(ConvoPeq PRIVATE /utf-8 /W4 /wd4100 /wd4189 /MP1 /EHsc /Zm400 /bigobj)
            → target-level・config 非依存（ConvoPeq 本体は RWDI でも /utf-8 を持つ）
L1519-1520  set(CMAKE_CXX_FLAGS_RELEASE "/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /fp:fast /Gw /Gy /Zi /utf-8 /EHsc")
L1524-1525  set(CMAKE_CXX_FLAGS_DEBUG   "/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc")
            → config-level・全 target に効く（非キャッシュ変数の上書き）
（該当なし）  CMAKE_CXX_FLAGS_RELWITHDEBINFO の set 文は CMakeLists.txt 全 2,004 行中に存在しない
L1964-2002  ASan ループ: if(ENABLE_ASAN) → target_compile_options(${tgt} PRIVATE ... /utf-8 ...)
            （ENABLE_ASAN=OFF のため不活性・L1985-1986 のコメントが欠落を明示的に記述済み）
L174-1100   個別 test target の target_compile_options(... /utf-8)（32 target）
```

**generated level（全 CXX compile edge 統計）**:

| Config | CXX edges | /utf-8 あり | /utf-8 無し | 欠落 target |
| --- | ---: | ---: | ---: | ---: |
| Release | 492 | **492** | 0 | 0 |
| Debug | 492 | **492** | 0 | 0 |
| RelWithDebInfo | 492 | 483 | **9** | **7** |

**欠落 7 target**（RWDI edge FLAGS `/DWIN32 /D_WINDOWS /EHsc /Zi /O2 /Ob1 /DNDEBUG -std:c++20 -MD`）:

```text
D8_1_WrapperCacheTests          （explicit 32 list・ASan 29 list のどちらにも無い）
DeferredDeletionQueueReclaimTests（ASan 29 list のみ → ENABLE_ASAN=OFF で不活性）
EQAnalysisUnitTests             （同上）
EQBoundExcessBenchmark          （同上）
EQProcessorMaxGainTests         （同上）
FFTBackendTests                 （同上）
GainStagingContractTests        （同上）
```

→ **答え: 「RWDI の全 TU が /utf-8 欠落」ではなく「7 target / 9 edge のみ欠落」**。
483 edge は target-level options（explicit 32 + ConvoPeq + JUCE 系 target 経由）で補完済み。
G6 で失敗した `DeferredDeletionQueueReclaimTests.cpp` はこの 7 target の 1 つ。

## H2 — 原因候補 8 項目の切り分け

| # | 候補 | 判定 |
| --- | --- | --- |
| 1 | `CMAKE_CXX_FLAGS_RELWITHDEBINFO` の未設定 | **✓ 第1根因**（set 文不在・CMake デフォルトへフォールバック） |
| 2 | generator expression による構成漏れ | **部分寄与**: ASan ループの /utf-8 は GenEx でなく config 非依存だが、`if(ENABLE_ASAN)` ガードが全体不活性（=OFF）。/utf-8 自体に GenEx は関与しない |
| 3 | `target_compile_options()` の適用対象差 | **✓ 第2根因**: /utf-8 の RWDI 補完が target-level にしか存在せず、対象が 32 target に限られる |
| 4 | test target と ConvoPeq target の差 | ✓ 補完: ConvoPeq は target-level /utf-8 を持ち RWDI でも安全。test target は explicit 32 のみ保護 |
| 5 | `CMAKE_CXX_FLAGS` 初期値／継承差 | ✗（共通 flags は全 config 同一・/utf-8 を含まない） |
| 6 | configure 時の toolchain/cache 状態 | ✗（cache 初期値は 3 config 同様。RELEASE/DEBUG も cache に /utf-8 は無い — 非キャッシュ set の上書き機構で統一説明できる） |
| 7 | 生成済み Ninja の stale generation | ✗（impl-RWDI.ninja は Release/Debug と同一時刻 09-05 18:58 生成・差分は CMakeLists 設定の忠実な写実） |
| 8 | 別 option による打ち消し | ✗（付与されないだけ。Release で /utf-8 が 2 回重複するのは config+target の二重付与で無害） |

## H3 — 最小 fixture による再現と分離（`%TEMP%/i3_4_h_h3_fixture`）

構成: `set(CMAKE_CXX_FLAGS_RELEASE/DEBUG ... /utf-8)` のみで RWDI set 無し（CMakeLists 該当
パターンを忠実に複製）+ `target_compile_options(withopt PRIVATE /utf-8)` の target + **実物の
`AtomicAccess.h` / `DeferredDeletionQueue.h` / `ISRWorldRetirementReference.h` /
`ISRWorldRetirementTelemetry.h` をverbatim 取り込んだ TU**。

```text
broken CMakeLists（現行複製）:
  Release        rc=0  diagnostics 0
  Debug          rc=0  diagnostics 0
  RelWithDebInfo rc=2  C4819 (CP932) + AtomicAccess.h(55): C2143/C4430/C2888/C2065/C2061
                       + AtomicAccess.h(75): C1075
                       → G6 の実物 RWDI 失敗と同一エラー集合で完全再現
fixed CMakeLists（RWDI set に /utf-8 を追加したのみの修正版）:
  Release / Debug / RelWithDebInfo すべて rc=0   ← H5 修正の予行検証成功
```

→ **CMake configuration semantics の問題**であることを ConvoPeq 固有 target から分離して
実証。compile 失敗の有無はソース内容のバイト偶発に依存するため（日本語のみの軽量ソースは
CP932 誤読でも parse が生き延びる — 負対照として確認済み）、実物ヘッダを用いた再現とした。

## H4 — 修正契約（H5 実装前に固定）

### 必須

```text
Release / Debug / RelWithDebInfo の全 config が、同一 semantic source から /utf-8 を得ること
  → MSVC ブロック内（L1497 if 文・NOT IntelLLVM）に
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "... /utf-8 ...") を追加する形
  → source encoding = UTF-8 を configuration-independent にする
```

### 推奨（同時に審議）

```text
RWDI の CMake デフォルト flags には /EHsc も無い。L1517-1518 のコメントは
「Release の /EHsc 欠落が JUCE C1189 を起こした既存バグ」を記録しており、
RWDI は同型の未発火リスク。/utf-8 と同時に /EHsc を含めることを推奨
（JUCE module obj を RWDI で compile する場合の予防。fixture fixed 版は /utf-8 単独で検証済み）。
```

### 禁止（ユーザー指示どおり）

```text
× RWDI だけへの ad-hoc /utf-8 追加（7 target 個別の target_compile_options 付け足し）
× test source 変更 / BOM 追加
× compiler codepage 変更による回避
× gate whitelist / build_identity_gate.py での隠蔽
（本件は source/build configuration coherence の問題であり、gate を弱めることは許されない）
```

### 実装時の付随作業（H5 以降の検収に含める）

```text
1. AdmissionPackedStateTests の target_compile_options 重複（L855/L856 同一 2 行）の除去検討
2. RWDI clean rebuild + gate --check（H6）・3 config 回帰（H8）・negative regression（H9）
3. 修正後の impl-RelWithDebInfo.ninja 再 census: 492/492 edges with /utf-8 を確認
```

## H10 — GO 条件対合（本監査は H1-H4 を判定・H5 以降は実装後）

| ID | 条件 | 判定 |
| --- | --- | --- |
| H1 | RWDI `/utf-8` 欠落の原因を CMake→Ninja edge で証明 | **GO**（L1519/L1524 vs 不在 + census 9 edge/7 target） |
| H2 | Release/Debug/RWDI の flag provenance 比較 | **GO**（Release 492/492・Debug 492/492・RWDI 483/492 の差分完封） |
| H3 | 最小 UTF-8 fixture で再現 | **GO**（実物ヘッダで同一エラー集合再現・fixed 版で解消） |
| H4 | 修正方針を実装前に契約化 | **GO**（上記契約・禁止事項・付随作業を固定） |
| H5-H10 | 実装・clean rebuild・統合・回帰 | **未実施（本監査の GO を前提に次フェーズで実施）** |

**総合判定 = H1-H4 GO → I3-4-H（H5 実装フェーズ）へ進行可能。**

## 添付 evidence（evidence/D162-2I3/）

```text
I3_4_H_RWDI_UTF8_CONFIG_COHERENCE_AUDIT.md   本書
i3_4_h_h0_snapshot.txt                        H0 固定
i3_4_h_h0_freeze/                             CMakeLists.txt + ninja files + stamp 保存
i3_4_h_h1_flag_provenance.txt                 H1/H2 provenance + census
fixture: %TEMP%/i3_4_h_h3_fixture/（broken/fixed CMakeLists・実物ヘッダ・全 build ログ）
```

## ツール使用記録

- 解析主軸: python（CP932 バイトを含む ninja/cache の安全解析・全 edge census）＋ WSL grep/rg 相当のテキスト抽出
- CMakeLists 解析は serena/AiDex のシンボル検索対象外（CMake テキスト）のため python/regex で実施
- cppcheck / clang-tidy / Dr.Memory は C++ 静的解析・メモリ検証ツールであり、本監査の主題
  （CMake flag provenance）には適用対象コードパスが無いため不使用（根拠を明記）
```
