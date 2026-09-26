# P1-5-IR-P2-J — Step 3-J post-fix audit / evidence consolidation（JUCE teardown fix）

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-J（post-fix audit）
- **性質**: read-only 後処理（3-J の V1〜V4・H-A/H-B/P2 matrix の確定）
- **3-J 修正**: `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp:1542`
  `static juce::ScopedJuceInitialiser_GUI juceInit;` → `juce::ScopedJuceInitialiser_GUI juceInit;`
  （**token 1 個除去のみ**）。`BuzzLogger`（:1540）は未変更（3-J-2 範囲外）。

---

## 1. 3-J-0 State Freeze（再確認済み）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e3`（`git rev-parse HEAD`） |
| production source | **0** |
| CMake / build.bat | **0** |
| default/calibration | **0** |
| JUCE source | **0** |
| commit | STOP（0） |
| working tree | `M`/`D` は既存差分のみ。3-J による追加差分は `BassBuzzMeasurement.cpp` の 1 token（`static`）除去のみ |

## 2. 3-J-3 Diff 自己検査

| 変更対象 | 内容 |
| --- | --- |
| `BassBuzzMeasurement.cpp:1542` | `static` を除去 → automatic local 化 |

- 追加の +4/-1 行は**監査用コメント**（`★ P1-5-IR Step 3-J ...`）であり、コード動作に影響しない
  （`juceInit` 宣言本体の変更は `static` 除去のみ）。pre-existing 差分とは明確に分離。
- STOP チェック（以下、いずれも 0）: production source / CMake / `BuzzLogger` / `h.stop()` /
  return 経路 / settle・sleep / measurement semantics / JUCE source / 追加 teardown API。

## 3. 3-J-4 OFF binary（再ビルド・固定値記録）

| 項目 | 値 |
| --- | --- |
| build exit code | 0 |
| binary | `build/Release/AudioEngineHarness.exe` |
| new SHA-256[:16] | `2054d970da4ec91c` |
| old SHA-256[:16] | `F6C49980920C1D8E`（指示値と一致） |
| timestamp | 2026-09-22 23:40（+09:00） |
| compiler : config | `cl`（MSVC）: Release |
| `CONVOPEQ_CORRECT_POLYPHASE_GAIN` | `OFF`（`CMakeCache.txt:BOOL=OFF`） |

## 4. Validation Matrix — V1〜V4

| ID | binary | 条件 | 結果 | 判定 |
| --- | --- | --- | --- | --- |
| **V1** | OFF（`build/Release`） | `--buzz-os=1` 等（固條件） | capture 完了 → `runBassBuzzMeasurement` return → process exit。exit 0・`c0000374` 無し・CRT heap error 無し。log `v1_off_os1_20260922_234709.log` | ✅ PASS |
| **V2** | ASan（`build-asan/RelWithDebInfo/AudioEngineHarness.exe`） | V1 と同一条件（ASan 環境・再ビルド後） | `attempting double-free` 消滅・`SUMMARY: AddressSanitizer` 消滅。exit 0。log `v2_asan_os1_rerun.log` | ✅ PASS |
| **V3** | OFF（`build/Release`） | `--buzz-os=0`（他同一） | exit 0・`c0000374` 無し・error marker 0。log `v3_off_os0.log` | ✅ PASS |
| **V4** | OFF（default path） | AudioEngineHarness default（`ctest` scenario） | 既存 PASS（`M04/M02/H01/SR01` 等全件 PASS）+ process exit 0 + heap corruption なし。log `v4_default.log` | ✅ PASS |

- V2 の新鲜度: ASan binary も修正済み版で再ビルド（build exit 0・SHA `f3d122bbfd3bbb1f...`）。V1→V2→V3→V4 の順に実施。
- `--buzz-quiet` 等の追加引数はなし（Step 3-I 凍結条件を完全遵守）。

### 4-1. V1 の補足観測

V1 log の tail に以下が観測された（修正志向の範囲外）：

```text
[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — residual intents may remain in System 1 queues
[DIAG] shutdown phase: DRAIN_RETIRE -> DESTROY at ~AudioEngine
```

これは **H-B（別チケット）**であり、3-J の成功条件「heap corruption なし / exit 0」には該当しない。
`routerPendingRetire=2` / `coordinator Faulted` は P2 measurement の開示対象として今後も別管理を継続する。

## 5. 判定

```text
Step 3-J = PASS
```

| 最終定格 | 値 |
| --- | --- |
| H-A | **CLOSED**（JUCE static teardown double-free。3-I 文書の候補 A を実装、V1〜V3 で否定条件消滅） |
| H-B | **OPEN**（別チケット。`routerPendingRetire=2` / `coordinator Faulted` / 3-I §3-I-6 と同一） |
| P2 measurement | **STOP**（3-J の後、直ちに P2 matrix に戻さない。3-J は teardown fix のみ。1.48 dB / 1-run-lag / IR behavior の評価は P1-5-IR 本流で別途実施） |

## 6. 次作業への境界（明示）

- 3-J で閉じたもの: `H-A` の JUCE `ScopedJuceInitialiser_GUI` static teardown double-free。
- 3-J で閉じていないもの（**秘密解消を伴うもの**）:
  - `BassBuzzMeasurement.cpp:1540` の `BuzzLogger`（`setCurrentLogger(nullptr)` 未追加・Debug assert 懸念は別検討）。
  - H-B（P2 の shutdown 系検知）。
- 今後の P1-5-IR 本流再開前には、本書の「3-J = teardown fixのみ」という範囲を前提とすること。
- P2 matrix への戻り方:
  1. まず `H-A = CLOSED` を凍結（本書）。
  2. `H-B = OPEN / separate` を維持（別対応）。
  3. `P2 measurement = STOP` のまま、teardown 修正のみを閉じる（本書）。
  4. その後、本来の 1.48 dB / 1-run-lag / IR runtime behavior を改めて評価。
 これは teardown fix を音質・IR 挙動の原因修正と混同しないための**境界宣言**である。
