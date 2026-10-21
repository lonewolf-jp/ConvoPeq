# STG-11-D18 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D18_POST-IMPLEMENTATION-GATE_20261001.md`
- Work item: **STG-11-D18-1**
- Contract: `doc/work113/P1-5-IR-P2_STG-11-D18_REPAIR-CONTRACT-AUDIT_20261001.md`
- Implementation: `doc/work113/P1-5-IR-P2_STG-11-D18_REPAIR-IMPLEMENTATION_20261001.md`
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一ファイル）

---

## 0. 総合判定

```
READY FOR COMMIT
```

| Gate | 結果 |
| --- | --- |
| Defect 特定 | GO（concrete defect 1 件） |
| 最小修正の定義可能性 | PASS（F-A / F-B / F-C の 3 点、NonRT 限定） |
| ISR authority boundary | PASS（RT 変更 0、新規 atomic 0、新規 mutex 0、authority 追加 0） |
| targeted regression | PASS（T-CFX-A/B/C/D） |
| Negative control | PASS（原始欠陥復元で 3 件 FAIL を実証） |
| Debug 全件 | PASS（45/45） |
| Release 全件 | PASS（45/45） |
| D1〜D17 regression | PASS（全件に含まれ PASS） |
| 静的解析 | PASS（D18 hunk 指摘 0） |
| ASan | **BLOCKED**（`0xC0000139` pre-main。PASS 扱いせず環境制約として記録） |
| ConvoPeq.md | FRESH（`2026-10-01 18:40:06`） |
| Commit / push | **未実施** |

---

## 1. Defect

**STG-11-D18-1** — 連続 publish 時の phantom crossfade record と stale completion による
retire 済み DSP の Active 復活および stale `activeRuntimeDSPHandle_` 上書き。
overlap publish が先行 fade の source DSP の retire 経路も消滅させる（orphan）。

Root cause: `onPublishCompleted` の register/begin の claim 非依存、
`endCrossfade` の無条件書き込み、`activate()` の fading-handle 無条件 null 化。

→ 詳細は Implementation §2。

---

## 2. 修正ファイル / production diff

| ファイル | 種別 | 行数（HEAD `f95f524c` 起点） |
| --- | --- | --- |
| `src/audioengine/ISRDSPHandle.cpp` | production | +30 / -4 |
| `src/audioengine/DSPTransition.h` | production | +26 / -14 |
| `src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp` | test（新規） | 279 |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | test（配線） | +3 |
| `src/tests/NormalRetireDSPHandleCompareTests.cpp` | test（契約更新） | +16 / -3 |
| `CMakeLists.txt` | build | +1 |

production の変更は 2 ファイルのみ（+56 / -18）。
（`CMakeLists.txt` と `PublishPipelineIntegrationTests.cpp` の全体 numstat は
STG-11-D1〜D18 の累積値であり、D18 の寄与分は上記 1 行 / 3 行。）

production の変更は 2 ファイルのみ。

---

## 3. 修正点

| ID | 場所 | 内容 |
| --- | --- | --- |
| F-A | `ISRDSPHandle.cpp::activate` | in-flight fade の `fadingRuntimeDSPHandle_` を null 化しない（終端は `endCrossfade`） |
| F-B | `DSPTransition.h::onPublishCompleted` | fading-slot claim を register/begin より先行。失敗時は retire のみ（`ramp.start` は無条件実行を維持） |
| F-C | `ISRDSPHandle.cpp::endCrossfade` | `toHandle→Active` は `CrossfadingIn` 時のみ、`activeHandle` は from/to 一致時のみ、`fading→null` は from/null 時のみ（`fromHandle→Retired` は無条件維持） |

---

## 4. Targeted tests

`src/tests/AudioEngineHarness/STG11D18CrossfadeTests.cpp`

| ID | 検証内容 | 結果 |
| --- | --- | --- |
| T-CFX-A | overlap completion で retire 済み DSP が復活せず、`activeHandle` が clobber されない | PASS |
| T-CFX-B | `activate()` が in-flight の fading handle を保持する | PASS |
| T-CFX-C | 通常完了 path の観測結果が変わらない（production 順を再現） | PASS |
| T-CFX-D | 構造検査（claim-first・ガード群・activate の fading 書き込みなし） | PASS |

Negative control: `ISRDSPHandle.cpp` を原始欠陥へ復元 → rc=1、
`D18-A h2 resurrected` / `D18-A active handle clobbered` /
`D18-B activate() destroyed the in-flight fading handle` /
`D18-D staleness guards missing` を確認。復元後 PASS へ復帰。

---

## 5. Debug / Release

```
Debug   build exit=0   CTest  100% tests passed out of 45   (ctest exit=0)
Release build exit=0   CTest  100% tests passed out of 45   (ctest exit=0)
```

Debug `AudioEngineHarness`（D18 sub-test を含む）を独立実行でも rc=0 を確認。

---

## 6. D1〜D17 regression

| 範囲 | 手段 | 結果 |
| --- | --- | --- |
| D1〜D5 | `STG11EQRetireTests` / `STG11D2SnapshotRetireTests` / `AudioEngineHarness`（D3/D4/D5 sub-test 含む） | PASS |
| D6 | Fresh Discovery NO-GO（production 変更なし） | N/A |
| D7〜D11 | `AudioEngineHarness`（D7/D8/D9/D10/D11 sub-test 含む） | PASS |
| D12〜D17 | Fresh Discovery NO-GO（production 変更なし） | N/A |
| R4/P3 | already repaired 確定（`ISRSemanticValidationTests` PASS） | PASS |

---

## 7. 静的解析

compile database: `build/compile_commands.json`（array 1785 entries）を
`tmp/clangdb/compile_commands.json` に dedupe（206 entries）。
検査対象: `ISRDSPHandle.cpp`、`STG11D18CrossfadeTests.cpp`、
`RuntimePublicationOrchestrator.cpp`（`DSPTransition.h` の include 者）、
`NormalRetireDSPHandleCompareTests.cpp`。

| ツール | D18 hunk の指摘 | hunk 外（既存） |
| --- | --- | --- |
| cppcheck 2.22（`--project`, warning/performance/portability, c++17） | **0** | 2 件: `ISRDSPHandle.h:109 uninitMemberVarNoCtor`、`AudioEngine.h:5297 unknownMacro`（JUCE macro ノイズ） |
| clang-tidy 23.1.1（`clang-analyzer-*,bugprone-*,performance-*,misc-*`） | **0** | `ISRDSPHandle.cpp` の `misc-const-correctness` / `misc-include-cleaner` 25 件（全て lock_guard・include の既存パターン）、test TU は harness から可視性が必須のエントリポイント 1 件のみ |

新 TU で出た `clang-analyzer-unix.Stream`（fopen/fread ループ）2 件は
`ifstream` / `ostringstream` へ置換して解消済み。残る 1 件は
`runSTG11D18CrossfadeTests` が harness から extern 宣言で呼ばれる
エントリポイントであるため意図的に残す。

---

## 8. ASan

```
ASAN_EXIT = -1073741511 (0xC0000139, STATUS_ENTRYPOINT_NOT_FOUND)
```

- 対象: `build-asan/Debug/STG11EQRetireTests.exe`（D18 の production ファイル
  `ISRDSPHandle.cpp` を含む）、`STG11D2SnapshotRetireTests.exe`
- 試行 1: 既定 PATH → `0xC0000139`
- 試行 2: LLVM の `clang_rt.asan_dynamic-x86_64.dll` を PATH へ追加 → `0xC0000139`
- 試行 3: MSVC 14.51.36231 の `clang_rt.asan_dynamic-x86_64.dll` を PATH へ追加 → `0xC0000139`

いずれも `main()` 進入前のエントリポイント解決失敗で終了し、ASan レポート自体を
生成していない。**PASS 判定不能**。ASan によるメモリ安全性の証跡は本次得られていない。
環境制約として記録する（STG-11-D10 / D11 と同じ block）。

代替の証跡: Debug / Release 両 config の 45/45 CTest、negative control、
静的解析。いずれも PASS。

---

## 9. ConvoPeq.md

```
Generated  : 2026-10-01 18:40:06
SHA-256    : 64C4AD043D052772FEF4B596701F63737032AC2C21F3100E3E98A00327535CDC
size       : 5,907,579 B
NEWER_SRC_COUNT : 0
STATUS     : FRESH
```

---

## 10. Repository state

```
HEAD    = f95f524cf6df0b3e1529425e690c02767912614e
branch  = main...origin/main [ahead 9]
staged  = 0 files
```

**commit / push は実施していない。** Owner の明示承認待ち。

---

## 11. 残存事項（次工程で検討）

1. fire-and-forget publish 経路に fading gate が無い。latest-wins / collapse の
   **仕様判断**を要する。D18-1 は「gate が無い状態でも incoherent な結果を
   残さない」ことで閉じた。gate の追加は本 STG の scope 外。
2. `Timer.cpp:1142` の `submitObserve` は通常 path では `endCrossfade` が先に
   fading handle を null 化するため到達しない可能性が高い。到達不能性の立証と
   处置は別途。
3. 既知 flake (R8): Debug `AudioEngineHarness` の散発クラッシュ。
   本次 4 ランすべて rc=0。