# P1-5-IR-P2-I — JUCE teardown lifetime 修正境界監査（Step 3-I）

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 2-I
- **性質**: read-only audit（production/test/CMake 変更 0・commit 0）
- **前提**: Step 3-H = PASS（`P1-5-IR-P2H_ASAN_LOCALIZATION.md`）。double-free の provenance は
  `runBassBuzzMeasurement` の function-local `static ScopedJuceInitialiser_GUI juceInit` の
  atexit dtor と、JUCE `getDeletedAtShutdownObjects()` の静的 `Array` の atexit dtor の
  破棄順序逆転として直接結合済み。
- **範囲**: `ScopedJuceInitialiser_GUI` の lifetime 変更前に、全使用箇所・呼び出し境界・
  teardown 順序・修正候補 A/B・validation matrix を確定する。修正実装はしない（3-J 待ち）。

---

## 3-I-0 — State Freeze（確認済み）

| 項目 | 値 |
| --- | --- |
| HEAD | `1e9e63e3`（`git rev-parse HEAD` で確認） |
| P2 matrix | STOP（修正未実施） |
| production source | **0**（`src/audioengine` `src/convolver` `src/dsp` `src/core` `src/runtime` の porcelain は空） |
| test source | **変更あり（pre-existing・今回触れない）**: `BassBuzzMeasurement.cpp` / `P1PolyphaseGainCharacterization.cpp` / `PublishPipelineIntegrationTests.cpp`（いずれも M 状態・3-I 以前からの作業差分。porcelain の `D src/tools/check_layout_offsets.py` は 3-I 対象外） |
| CMake | **0**（`CMakeLists.txt` `build.bat` `cmake/` の porcelain は空） |
| default/calibration | **0**（該当パスの porcelain は空） |
| commit | STOP（0） |
| `build-asan/` | 存在。3-H localization artifact として扱い、通常 binary の評価対象に混ぜない |

> 注意: 3-I-0 の freeze 要求「test source = 0」は、作業開始時点で既に M 状態の
> 3 ファイルがあるため「**今回の 3-I 作業による変更 0**」として記録する。
> 3-I 自体は read-only であり、上記差分への追記・commit は行わない。

---

## 3-I-1 — `ScopedJuceInitialiser_GUI` 全使用箇所 census

対象: 最新 source（`git grep` + `grep -rn` の両方で確認）。`ConvoPeq.md` にも同一 17 箇所が含まれることを確認（`grep -c` = 17・代表行 :88931 / :93193 / :93254 / :93289 / :93324）。

### 1-1. 同一 executable（`AudioEngineHarness`）内の function-local static — 17 箇所

`AudioEngineHarness` target（`CMakeLists.txt:1904`）は以下 3 ファイルを**同一 exe**にリンクする。
`BassBuzzMeasurement.cpp:1542` を含む全 17 箇所は **function-local `static`** であり、
namespace-scope static / global は 0 件。

| # | ファイル | 行 | 関数 | 変数名 | 種別 | lifetime |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `BassBuzzMeasurement.cpp` | 1542 | `runBassBuzzMeasurement` | `juceInit` | function-local static | 初回呼出〜process exit（atexit dtor） |
| 2 | `ConvolverStateRoundTripTests.cpp` | 744 | `checkH01DryWetAlignment` | `h01JuceInit` | function-local static | 同上 |
| 3 | `ConvolverStateRoundTripTests.cpp` | 1062 | `checkSR01SetterClamp` | `sr01JuceInit` | function-local static | 同上 |
| 4 | `ConvolverStateRoundTripTests.cpp` | 1425 | `checkM04OversizedContainment` | `m04Init` | function-local static | 同上 |
| 5 | `ConvolverStateRoundTripTests.cpp` | 1627 | `checkM02NonFiniteTelemetry` | `m02Init` | function-local static | 同上 |
| 6 | `ConvolverStateRoundTripTests.cpp` | 1839 | `checkM01DenormalHygiene` | `m01Init` | function-local static | 同上 |
| 7 | `ConvolverStateRoundTripTests.cpp` | 2062 | `measureM03DirectHeadTiming` | `m03Init` | function-local static | 同上 |
| 8 | `ConvolverStateRoundTripTests.cpp` | 2161 | `measureM03HcLcBoundary` | `m03_2Init` | function-local static | 同上 |
| 9 | `IRLoadAdmissionTests.cpp` | 385 | `checkChannelAdmissionRuntime` | `juceInit` | function-local static | 同上 |
| 10 | `IRLoadAdmissionTests.cpp` | 446 | `checkResampleBoundUsesTrimmedLength` | `juceInit` | function-local static | 同上 |
| 11 | `IRLoadAdmissionTests.cpp` | 481 | `checkResampleBoundRejectsOversized` | `juceInit` | function-local static | 同上 |
| 12 | `IRLoadAdmissionTests.cpp` | 516 | `checkResamplePathAccepted` | `juceInit` | function-local static | 同上 |
| 13 | `IRLoadAdmissionTests.cpp` | 617 | `checkPreviewChannelAdmission` | `juceInit` | function-local static | 同上 |
| 14 | `IRLoadAdmissionTests.cpp` | 672 | `checkPreviewChunkedReadEquivalence` | `juceInit` | function-local static | 同上 |
| 15 | `IRLoadAdmissionTests.cpp` | 721 | `checkPreviewResampleBoundUsesTrimmedLength` | `juceInit` | function-local static | 同上 |
| 16 | `IRLoadAdmissionTests.cpp` | 748 | `checkPreviewResampleBoundRejectsOversized` | `juceInit` | function-local static | 同上 |
| 17 | `IRLoadAdmissionTests.cpp` | 774 | `checkPreviewFailureCompletion` | `juceInit` | function-local static | 同上 |

### 1-2. 別 executable / ビルド対象外

| 場所 | 行 | 形態 | 帰属 |
| --- | --- | --- | --- |
| `tools/AutoGainBenchmark.cpp` | 240 | automatic local（`juce::ScopedJuceInitialiser_GUI scopedJuce;`・static なし） | **どの CMake target にも含まれない**（`grep AutoGainBenchmark CMakeLists.txt build.bat cmake/` = 0 件）。3-I 対象外 |
| `src/tests/MT-NUPC-Measurement.cpp` | 671 / 1113 | `juce::initialiseJuce_GUI()` / `juce::shutdownJuce_GUI()` の明示呼出（Scoped クラス不使用） | **別 exe** `MTNUPCMeasurement`（`CMakeLists.txt:1076` `juce_add_console_app`）。`AudioEngineHarness` とは process を共有しないため 3-I 対象外 |

### 1-3. 対応する JUCE shutdown（実コード確定）

`JUCE/modules/juce_events/messages/juce_MessageManager.cpp:457-477`:

```cpp
initialiseJuce_GUI() { MessageManager::getInstance(); }
shutdownJuce_GUI()   { DeletedAtShutdown::deleteAll(); MessageManager::deleteInstance(); }
static int numScopedInitInstances = 0;
ScopedJuceInitialiser_GUI::ScopedJuceInitialiser_GUI()  { if (numScopedInitInstances++ == 0) initialiseJuce_GUI(); }
ScopedJuceInitialiser_GUI::~ScopedJuceInitialiser_GUI() { if (--numScopedInitInstances == 0) shutdownJuce_GUI(); }
```

- refcount `numScopedInitInstances` により、同一 process 内の複数 static instance は
  **最初の ctor で 1 回だけ initialise、最後の dtor で 1 回だけ shutdown** する。
- したがって「複数 static instance の共存」は JUCE 設計上許容されており、
  **二重 initialise/shutdown 自体は起きない**。問題は shutdown **時点**（atexit 順序）のみ。

### 1-4. STOP 条件の判定結果

| STOP 条件 | 結果 |
| --- | --- |
| 複数 static instance が同一 exe lifetime で共存 | **该当あり → ただし STOP せず記録**。17 箇所が同一 exe に存在。ただし refcount により多重 shutdown は起きない。共存自体は既知の JUCE 許容形態。**実害経路は「--buzz 単独実行時に juceInit（:1542）のみが構築され、`objects` Array の atexit dtor より後に破棄される」順序逆転**（3-H 確定）。共存が発火条件を変えるのは「default path（:1305-1319）で複数 check が同一 process で走る場合」のみで、`--buzz` dispatch（`main:1126-1127` 即時 return）では他 16 箇所は構築されない（§3-I-2 参照）。**修正案に進まず報告する条件には該当しない**（共存≠多重 shutdown のため）が、3-J 実装時の回帰確認対象として V4 に残す |
| 明示的 `shutdownJuce_GUI()` | `AudioEngineHarness` exe 内には **0 件**。唯一の呼出は別 exe `MT-NUPC-Measurement.cpp:1113`（対象外）。**非該当** |
| `DeletedAtShutdown::deleteAll()` の直接呼出 | project `src/` + `tools/` に **0 件**（JUCE 内部 `shutdownJuce_GUI` のみ）。**非該当** |
| `DeletedAtShutdown` の登録解除の独自実装 | **0 件**。project 側で `DeletedAtShutdown` を継承するクラスは 0 件（`git grep` 空）。登録源は JUCE 内部 `ShutdownDetector`（`juce_Timer.cpp:38`・`private DeletedAtShutdown`）のみ。**非該当** |
| `std::atexit` / static dtor の独自制御 | project `src/` + `tools/` に **0 件**。**非該当** |
| `runBassBuzzMeasurement()` の static initializer が他の測定 entry と共有 | **非該当**。`juceInit`（:1542）は `runBassBuzzMeasurement` の function-local であり、他 entry と共有されていない（他 16 箇所は別関数・別変数）。**非該当** |

---

## 3-I-2 — `runBassBuzzMeasurement()` 呼び出し境界監査（実コード確定）

### 2-1. Caller（唯一）

`src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp:1126-1127`（`main` 内 dispatch loop）：

```cpp
else if (a == "--buzz" || a.rfind("--buzz-", 0) == 0)
    return runBassBuzzMeasurement(argc, argv);
```

- caller は `main` のみ。他 TU からの呼出はなし（`git grep runBassBuzzMeasurement` の定義外ヒットは宣言 `:1110` と dispatch `:1127` のみ）。
- `main` は dispatch 時に **即時 return** するため、`--buzz*` 実行時は default path
  （`:1305-1319` の `runDeferredFlowIntegrationTests` / `runConvolverStateRoundTripTests` /
  `runIRLoadAdmissionTests` 等）には到達しない。**他 16 箇所の static initializer は
  `--buzz` 実行時には構築されない**（同一 exe 内に存在するが、同一 process lifetime で共存しない）。

### 2-2. 呼出回数

`main` の dispatch は `return` で終わるため、**1 process あたり最大 1 回**。
`juceInit` static の再入・二重構築は起きない。

### 2-3. `AudioEngineHarness h` の破棄（return 前に完了）

- `h` は `runBassBuzzMeasurement` の automatic local（`:1704` `AudioEngineHarness h;`）。
- `h` 構築後の全 return 経路（`:1723-1724` `:1768-1769` `:1798-1799` `:2063-2064`(rigcheck終端)
  `:2173-2174` `:2183-2184` `:2220` `:2232` `:2274` `:2445-2446` rigcheck `:2496-2497`
  `:2519-2520`・最終 `:2698-2700`）は **すべて `h.stop()` → `return` の順**。
  `h.stop()` を経由しない `return` は `h` 構築前（`:1700` IR missing 等）のみ。
- `h` の dtor（`AudioEngineHarness.cpp:16-19`）は `stop()` を呼ぶため、
  `runBassBuzzMeasurement` return 時点で `h`（→ `engine_` → `AudioEngine`）の破棄は完了する。
- `AudioEngine::~AudioEngine`（`AudioEngine.CtorDtor.cpp:103`）は `stopTimer()` を明示呼出
  （TimerThread への listener 解除）。**engine 側の Timer 登録解除は dtor で済んでいる**。
  残るのは JUCE 側 `ShutdownDetector`/`TimerThread`（`DeletedAtShutdown` 登録済み）の破棄であり、
  これは `shutdownJuce_GUI → deleteAll` の責務。**AudioEngine の shutdown/reclaim 修正は不要**（3-I 指示どおり飛ばない）。

### 2-4. `juceInit` の lifetime と teardown 順序（3-H との結合）

```text
main
 ↓ --buzz dispatch（:1127 即時 return）
runBassBuzzMeasurement()
 ├─ :1540 static BuzzLogger logger 構築（初回呼出時・atexit 登録①）
 ├─ :1541 setCurrentLogger(&logger)
 ├─ :1542 static ScopedJuceInitialiser_GUI juceInit 構築（初回呼出時・atexit 登録②）
 │        → numScopedInitInstances 0→1 → initialiseJuce_GUI → MessageManager 生成
 ├─ :1704 AudioEngineHarness h 構築 → AudioEngine ctor
 │        → juce::Timer 基底 → TimerThread 起動 → ShutdownDetector 生成
 │        → DeletedAtShutdown::DeletedAtShutdown → objects[] に登録
 │          （この時点で getDeletedAtShutdownObjects() の静的 Array が初回構築・atexit 登録③）
 ├─ 測定…
 ├─ h.stop() → h dtor → ~AudioEngine（stopTimer 済）→ return
 ↓ runBassBuzzMeasurement return
main return
 ↓ CRT atexit（LIFO）
 ③ objects Array dtor が先 → block 解放（free #1）
 ② juceInit dtor → shutdownJuce_GUI → deleteAll → Array::clear → setAllocatedSize → free #2
    = double-free（3-H ASan stack #6 ← #4 ← #3 と一致）
```

- atexit LIFO 規則上、後に登録されたものが先に破棄される。`objects` Array（③）は
  `juceInit`（②）より**後**に初回構築されるため、LIFO では `objects` が先に破棄される。
  `deleteAll` は「Array が生きている」前提で `clear()` する設計のため、
  既に dtor 済みの Array に対する `clear → setAllocatedSize(:234)` が double-free となる。
- これは **JUCE の既知の静的 teardown 制約**（function-local static ScopedInitialiser +
  同一 TU 外の静的 Array の順序逆転）であり、production DSP / Convolver / RuntimeWorld /
  `ISRRetireRouter` / `DeferredFree` / `ShutdownRuntime` とは無関係。

### 2-5. 監査結論（3-I-2）

| 確認対象 | 結果 |
| --- | --- |
| caller | `main`（`:1127`）唯一・即時 return |
| 呼出回数 | 1 process 1 回 |
| `h` 破棄 | return 前に完了（全経路 `h.stop()` 済・dtor で二重保証） |
| `h.stop()` | あり（`AudioEngineHarness::stop` → terminal release + `releaseResources`） |
| `AudioEngine` dtor | return 前に完了（`stopTimer()` 明示済） |
| `juceInit` lifetime | function-local static（process exit まで生存・atexit dtor） |
| JUCE shutdown と CRT atexit の順序 | **逆転確定**（`objects`③ → `juceInit`② の順で破棄 → double-free） |

---

## 3-I-3 — 最小修正候補（コードは書かず比較のみ）

### Candidate A — `juceInit` を automatic local lifetime に変更

```cpp
int runBassBuzzMeasurement(...)
{
    juce::ScopedJuceInitialiser_GUI juceInit;  // static を外す
    ...
}
```

| 観点 | 評価 |
| --- | --- |
| dtor 実行時点 | `runBassBuzzMeasurement` return 前（`h` 破棄後・関数スコープ退出時）。CRT atexit より大幅に早い |
| `AudioEngine` dtor との前後 | **engine dtor の後**（`h` は `:1704` で `juceInit` より後に構築されるため、automatic の逆順破棄で `h` → `juceInit` の順に dtor が走る）。`stopTimer()` 済みの engine の後に JUCE shutdown が来る = 正しい順序 |
| `DeletedAtShutdown::objects` との前後 | `deleteAll` が `objects` Array 生存中（atexit 前）に実行される。`clear()` は有効な Array に対する操作となり double-free は消える |
| 他 test entry への影響 | `runBassBuzzMeasurement` のみに閉じる。他 16 箇所は無変更 |
| `--buzz` 以外の harness mode への影響 | なし（`--buzz` dispatch は即時 return のため、default path とは process を共有しない） |
| JUCE 初期化を要する既存 test への影響 | 関数内で JUCE を使う全処理（`juce::File` `juce::AudioBuffer` 等）は `juceInit` 構築後（`:1544` 以降）に実行されるため、automatic 化しても初期化カバレッジは変わらない。唯一の注意は **早期 return 経路**（`:1639` NUC standalone 等・`:1650` `:1656` `:1662` `:1669`）で `juceInit` 構築後に return する場合も dtor が正常に走ること（automatic のため保証される）。`parseOnOff` の `std::exit(2)` 経路（`:1578` `:1593` `:1613`）では dtor が bypass されるが、現状の static でも `exit` 時に atexit dtor は走るため挙動差はある（§残留リスク参照） |

### Candidate B — lifetime を `main()` 側へ移し `--buzz` dispatch より外側で管理

```text
main
 ├─ ScopedJuceInitialiser_GUI juceInit（automatic・dispatch より前）
 └─ --buzz → runBassBuzzMeasurement（static juceInit を削除）
```

| 観点 | 評価 |
| --- | --- |
| dtor 実行時点 | `main` return 前（`runBassBuzzMeasurement` の `return` 後・`main` スコープ退出時）。CRT atexit より早い |
| `AudioEngine` dtor との前後 | **engine dtor の後**（engine は `runBassBuzzMeasurement` 内 automatic `h` の member であり、`main` の `juceInit` より後に構築・先に破棄）。正しい順序 |
| `DeletedAtShutdown::objects` との前後 | `deleteAll` が `objects` Array 生存中に実行される。double-free は消える |
| 他 test entry への影響 | **`main` の全 dispatch が JUCE 初期化済みになる**。default path の各 check が持つ function-local static initializer は refcount により no-op 化される（`numScopedInitInstances` が既に 1 のため `initialiseJuce_GUI` は再実行されない）が、**各 check の static dtor 時の `--numScopedInitInstances` が 0 にならず、`shutdownJuce_GUI` は `main` の dtor まで遅延する**。これは正しい方向（shutdown が atexit より早まる）だが、**全 harness mode の teardown 時点が変わる**ため影響範囲が A より広い |
| `--buzz` 以外の harness mode への影響 | default path・`--p1-char`・`--t1..t4`・soak 全体で JUCE lifetime が変わる。V4 の確認範囲が拡大する |
| JUCE 初期化を要する既存 test への影響 | 初期化自体は常に有効になるため機能的退行はないが、**`--buzz` 用に常時 GUI 初期化コストを払う**全 mode への波及と、T1〜T4/soak の長時間実行における MessageManager 生存期間の延伸が副次効果 |

### A/B 比較（優劣の点数化なし・事実のみ）

- 両案とも double-free の直接原因（atexit 順序逆転）を除去できる。
- A は **影響が `runBassBuzzMeasurement` に閉じる**（最小修正の原則に適合）。
- B は **全 mode の teardown 時点を一括で atexit 前に前倒しする**（将来の同種 double-free を全 entry で予防できる）が、影響範囲が harness 全体に及ぶ。
- 残留リスク（両案共通）: `parseOnOff` 系の `std::exit(2)` 経路では automatic dtor が bypass され、
  `MessageManager`/`objects` が atexit に残る。現状 static でも `exit` 後に atexit dtor 順序問題が
  残り得るが、不正引数時の即時異常終了パスであり、V1〜V4 の正常系 validation とは分離して扱う
  （3-J 実装時に `exit` → `return` 化の要否を判断。今回は監査のみ）。

---

## 3-I-4 — `BuzzLogger` + JUCE Logger global state の lifetime census

### 4-1. 実コード確定

- `BuzzLogger` 定義: `BassBuzzMeasurement.cpp:52-60`（`namespace convo_buzz` 内・`public juce::Logger`）。
- 設置: `:1540` `static BuzzLogger logger;`（function-local static・`juceInit` の**直前**に宣言・構築）。
- 登録: `:1541` `juce::Logger::setCurrentLogger(&logger);`
- **解除（`setCurrentLogger(nullptr)`）: `BassBuzzMeasurement.cpp` 内に 0 件**。
  repo 内の `setCurrentLogger(nullptr)` は `MainApplication.cpp:207`（別 exe 本体）、
  `PublishPipelineIntegrationTests.cpp:749,905,913,921,927,939,945`（別関数）、
  `T1〜T4Measurement.cpp`（別 TU・別 entry）のみ。`runBassBuzzMeasurement` は解除しない。

### 4-2. JUCE Logger global の mechanics（実コード確定）

`JUCE/modules/juce_core/logging/juce_Logger.cpp:43-55`:

```cpp
Logger* Logger::currentLogger = nullptr;  // namespace-scope static（JUCE 側）
setCurrentLogger(newLogger) { currentLogger = newLogger; }  // 単なる代入・所有権なし
writeToLog → currentLogger->logMessage(message)
~Logger() { jassert(currentLogger != this); }  // 自身が current のまま dtor されると assert
```

- `currentLogger` は生ポインタ保持・所有権なし。JUCE shutdown（`shutdownJuce_GUI`）は
  `currentLogger` に触れない（`deleteAll` + `MessageManager::deleteInstance` のみ）。
- `static BuzzLogger logger` の atexit dtor（登録①）は、`juceInit`（②）・`objects`（③）より
  **先に登録 → atexit LIFO では後に破棄**される。破棄順は ③ → ② → ①。
  ①の時点で `currentLogger` は既に dangling（`logger` 自身を指す）が、JUCE 側は参照しないため
  実害はない。ただし `~Logger` の `jassert(currentLogger != this)` は、**current のまま static
  logger が破棄されると Debug で assert** する。現状 `setCurrentLogger(nullptr)` がないため、
  Debug ビルドの終了時にこの assert が発火し得る（Release では no-op）。
- ASan stack（3-H）に Logger は出現しておらず、**double-free の culprit ではない**。
  ただし 3-J の最小 fix と同時に `setCurrentLogger(nullptr)` を `h.stop()` 後の return 前に
  置くかは、A/B いずれの案でも併せて判断すべき随伴項目として記録する（推測で変更しない）。

### 4-3. teardown 順序の全体像

```text
③ objects Array dtor（atexit・最初）— free #1
② juceInit dtor → shutdownJuce_GUI → deleteAll → free #2（double-free）
① logger dtor（atexit・最後）— currentLogger は dangling のまま残る（実害なし・Debug assert のみ）
```

---

## 3-I-5 — Validation matrix（凍結）

smoke 条件（全ケース共通）:

```text
--buzz-os=1 --buzz-probe=real --buzz-conv=on --buzz-eq=off --buzz-order=cte
--buzz-ir=tmp\p15_ir_g0.wav --buzz-probe-level=0.5011872
```

| ID | binary | 条件 | 期待 | 目的 |
| --- | --- | --- | --- | --- |
| V1 | 通常 binary（`build/` 系・非 ASan） | 上記 smoke | capture 完了 → `runBassBuzzMeasurement` return → process exit で heap corruption なし（exit code 0・CRT ダイアログなし） | 3-J fix の主検証 |
| V2 | ASan binary（`build-asan/RelWithDebInfo/AudioEngineHarness.exe`） | 上記 smoke と同一 | `attempting double-free` が消える（ASan exit 0・SUMMARY なし） | 3-H provenance の直接消滅確認 |
| V3 | 通常 binary | no-OS control（`--buzz-os=0`・他同一） | V1 と同様に corruption なし | OS=1 固有修正になっていないことの確認（3-G §5-2 の「OS 条件に依存しない共通 teardown 経路」と整合） |
| V4 | 通常 binary | 非 buzz entry を最低 1 系統（例: default path の短時間 `ctest -R AudioEngineHarness` または `runConvolverStateRoundTripTests` 経路） | 既存 PASS が維持される（特に M-04/M-02 等の static initializer を持つ check 群） | 共有 exe 内の他 static initializer への退行がないことの確認 |

- V1〜V4 はいずれも **3-J 実装後に実行**する。3-I 時点では凍結のみ（実行しない）。
- `build-asan/` は V2 専用。V1/V3/V4 の通常 binary 評価に混ぜない。

---

## 3-I-6 — H-B は別トラックとして保持（分離宣言）

```text
H-A: JUCE DeletedAtShutdown double-free → Step 3-H PASS → 3-I lifetime remediation audit（本書）
H-B: routerPendingRetire=2 / coordinator Faulted → 別チケット → 今回の修正対象外
```

- 3-H ASan stack に `ISRRetireRouter` / `DeferredFree` / `ShutdownRuntime` は出現していない
  （分離の根拠 = ASan stack が両者を結ばない。`P1-5-IR-P2H_ASAN_LOCALIZATION.md` §4 と同一判断）。
- Practical Stable ISR Bridge Runtime の shutdown 責務順序
  （Stop → Drain Intent → Drain Retire → Epoch → Reclaim → Verify Empty → Shutdown Complete）は
  独立トラックであり、**JUCE static teardown の修正に `ISRRetireRouter` / `DeferredFree` /
  `ShutdownRuntime` の変更を混ぜない**。

---

## Step 3-I 完了条件の照合

| 条件 | 結果 |
| --- | --- |
| ScopedJuceInitialiser_GUI 全使用箇所 census | ✅ 17 箇所（同一 exe）+ 別 exe 1 箇所 + ビルド対象外 1 箇所（§3-I-1） |
| runBassBuzzMeasurement caller/lifetime 確定 | ✅ caller=`main:1127` 唯一・1 process 1 回・`h` は return 前破棄完了（§3-I-2） |
| AudioEngine destructor との順序確定 | ✅ `~AudioEngine`（`stopTimer()` 済）は `juceInit` dtor より前。engine 側修正不要（§3-I-2-2-3/2-4） |
| BuzzLogger lifetime 確認 | ✅ static（:1540）・解除なし・culprit ではない・随伴項目として記録（§3-I-4） |
| DeletedAtShutdown との関係確認 | ✅ `ShutdownDetector`（JUCE 内部）のみが登録源・project 側継承 0 件・順序逆転の mechanics 確定（§3-I-1-3/1-4・§3-I-2-4） |
| 修正候補 A/B を列挙 | ✅ コードなし・比較のみ（§3-I-3） |
| validation matrix を凍結 | ✅ V1〜V4（§3-I-5・実行は 3-J 後） |
| production source 変更 | ✅ 0 |
| test source 変更 | ✅ 0（3-I 作業による変更なし。pre-existing M 状態は §3-I-0 に記録） |
| CMake 変更 | ✅ 0 |
| commit | ✅ 0 |

## 判定

```text
Step 3-I = PASS
```

- lifetime/order はソース実コードのみで確定できた（BLOCKED 条件に該当せず）。
- `ScopedJuceInitialiser_GUI` の修正実装は行っていない。**Step 3-J（最小 test-only teardown fix）の
  実装承認待ち**とする。
- STOP 条件のうち「複数 static instance の同一 exe 共存」は該当したが、
  JUCE refcount 設計上多重 shutdown は起きず、`--buzz` 実行時は他 16 箇所が構築されないため、
  3-H の provenance（`juceInit:1542` 単独 + `objects` 順序逆転）を変えるものではない。
  よって監査継続を妨げない記録事項とし、V4 の回帰確認対象に含める。
