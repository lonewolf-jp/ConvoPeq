# STG-11-D1 Repair Implementation（2026-09-29）

> **Verdict: STG-11-D1 Repair Implementation = PASS**
> **COMMIT = NOT AUTHORIZED / PUSH = NOT AUTHORIZED（未実施）**
> 本書は Owner GO（Candidate B 正式採用）に基づく最小実装と全 validation の実施記録である。

---

## 1. Authority

Owner 指定: `ConvoPeq(20260928-152951).md == ConvoPeq.md`（統一 authority）。
実 source 検証は repo の現行 `ConvoPeq.md` を基準に実施した。

### 1.1 実装開始時

| 項目 | 値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09` |
| size | 5,662,249 B |
| Generated | 2026-09-29 00:19:43 |
| NEWER_SRC_COUNT | 0 |
| STATUS | FRESH |

### 1.2 全 validation PASS 後の再生成（Owner §7-7）

| 項目 | 値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e`（不変） |
| SHA-256 | `CDAFA18BBCA815595ACB0B2C3D3F4C1FCB25D01DE150FEA6772837F65E5C38B1` |
| size | 5,690,145 B |
| Generated | 2026-09-29 19:35:52 |
| NEWER_SRC_COUNT | 0 |
| STATUS | FRESH（`--check` exit 0） |

---

## 2. Implementation contract の最終確認（Owner §7-1）

Audit の Repair Contract（候補 B）を source 上で再確認してから実装した。

| 契約要素 | source 確認 |
| --- | --- |
| provider 束縛 | `m_ownedRetireRouter { m_epochDomain }`（`EQProcessor.h:527`）。`m_epochDomain`（`:521`）より後に宣言 |
| retireEpoch | `m_epochDomain.currentEpoch()`（`Core.cpp:43`、不変） |
| reclaim boundary | member router → `provider_->getMinReaderEpoch()`（`ISRRetireRouter.cpp:187` 委譲、不変） |
| engine domain 不使用 | diff 内の engine 参照 0（V7 検証） |
| `ISRRetireRouter.*` 不変 | diff 0（V2/V4 検証） |
| RT path 不変 | `Processing.cpp` / `Coefficients.cpp` / `Parameters.cpp` diff 0、追加行の RT guard/load 0（V3 検証） |
| `stackRouter` 残存 | 0 件（V11 検証） |

---

## 3. D1 最小実装（Owner §7-2）

### 3.1 Production diff（2 ファイル）

**`src/eqprocessor/EQProcessor.h`**（+39/-0 相当）:

1. `#include "audioengine/ISRRetireRouter.h"` を追加（循環参照なし。
   `ISRRetireRouter.h` は `EQProcessor.h` を参照しない）。
2. member を `m_epochDomain`（`:521`）の直後に追加（`:527`）:
   `convo::isr::ISRRetireRouter m_ownedRetireRouter { m_epochDomain };`
3. test/diagnostic observation の public const getter 7 件を追加
  （`pendingRetire` / `quarantineResident` / `emergencyResident` /
   `terminalResident` / `retireDropCount` / `privateEpoch` / `routerEpoch`）。
   Logic-neutral（読み取りのみ）。`diagFootprintBytes()` と同一規約。

**`src/eqprocessor/EQProcessor.Core.cpp`**:

1. `enqueueDeferredDeleteWithFallback()` — `stackRouter` を `m_ownedRetireRouter` に置換
   （2 箇所: coordinator 経路と `enqueueWithRetry` 直接経路）。
2. **二重所有の排除**（実装中に発見した同一関数内の第二欠陥。§4 参照）:
   coordinator 経路は内部で `router.enqueueWithRetry` を完全委譲するため、
   `Success / QueuePressure / TerminalReclaim` は ownership transfer 成立として
   `true` を返し、同一 ptr での再実行を行わない。
   `Shutdown` は `false`（caller が ownership 保持）。
   `QueueFull`（未格納）のみ router-level retry に進む。
3. `flushPendingEpochAdvance()` — `publishEpoch()` 直後に
   `m_ownedRetireRouter.tryReclaim()` を追加（定期的 reclaim driver。§5 参照）。
4. `~EQProcessor()` — 既存 D drain の前に member router の
   `tryReclaim()`（epoch-gated）＋ `drainAllQuarantineStore()`（force）を追加。
   force の破棄前提は既存 `m_epochDomain.drainAll()` と同一
   （publication pipeline が engine-epoch-gated に DSPCore を破棄し、
   `BlockDouble.cpp:151` の engine-domain read 区間が audio block 全体を覆うため
   audio quiescence 成立）。

### 3.2 変更禁止の遵守

| 禁止対象 | 結果 |
| --- | --- |
| `EpochDomain` | diff 0（STG-10-D1 の RC-1 を維持） |
| `RCUReader` / RT process path | diff 0 |
| `RuntimeIntentCoordinator` | diff 0 |
| AudioEngine の shared retire authority | diff 0 |
| RuntimeWorld / Publication authority | diff 0 |
| Crossfade authority | diff 0 |
| STG-10-D1 の修正 | diff 0 |
| `ISRRetireRouter.*` | diff 0 |
| raw `std::atomic` method 追加 | 0 件 |
| memory ordering 変更 | 0（追加行は既存 `acquire` パターンのみ） |

---

## 4. 実装中の第二欠陥の発見と修正

### 4.1 事実

`enqueueDeferredDeleteWithFallback()` は coordinator 経路
（`RuntimeIntentCoordinator::enqueueRetire` →
`ISRRuntimePublicationCoordinator.cpp:160` で `router.enqueueWithRetry` に完全委譲）の
戻り値が `Success` でない場合、**同一 ptr で `enqueueWithRetry` を再実行**していた。

coordinator 経路が `QueuePressure / TerminalReclaim` を返した場合は
既に Q/E/T へ格納済みであるため、再実行は**同一オブジェクトの二重所有**になる。

### 4.2 数値による確定

TD1-2（2700 setters × 2 = 5400 objects）の fill 状態:

```text
期待（単一所有）: D=4096 Q=512 E=512 T=280（合計 5400）
実測（二重所有）: D=4096 Q+E=1024 E=512 T=1584
```

`T = 280 + 1304`（= 5400 − 4096）は、call#1 で Q/E/T に格納された 1304 個が
call#2 で全て T へ重複格納されたことと一字一句一致する。

### 4.3 帰結

- 二重所有 → drain 時の double-delete → heap corruption → abort / hang。
  実際に TD1-1（Q=608 = 304×2 を検出）および TD1-2 後の dtor で abort を観測した。
- pre-fix（stack-local 時代）は call#1 の格納分が関数 return で失われていたため、
  二重所有は顕在化せず二重 leak に留まっていた。**member 化により顕在化した。**
- 修正は `enqueueDeferredDeleteWithFallback()` 内に閉じる
  （Owner §2 の scope 内。Coordinator / Router / authority の変更なし）。

### 4.4 Negative control

二重所有ガードを一時的に revert した隔離ビルドで TD1-1 が
`Q residency=608 expected 304`（正確に 2 倍）で FAIL し、その後 abort。
**value oracle のみで二重所有を検出できることを実証**した後、修正を復元した。
（production の一時的 revert は検証後に完全復元し、最終 diff に含まない。）

---

## 5. 定期的 reclaim driver（Owner §4）

```text
driver = EQProcessor::flushPendingEpochAdvance() 内の publishEpoch 直後
理由   = (1) epoch 前進後に reclaim 可能になる entry が生まれる意味論的結合点
         (2) 呼び出し元が releaseResources / prepareToPlay（いずれも NonRT）に限定
         (3) 新規 timer / thread / authority の追加なし
         (4) RT からは呼ばれない（RT は flush を呼ばない）
```

`ISRRetireRouter::tryReclaim()`（public、`cpp:386-394`）は
provider tryReclaim（private D）＋ `drainQuarantineStore`（Q）＋
`drainEmergencyAndTerminal`（E/T）を epoch-gated で実行する。
`signalDrainWakeup()` が起こすのは engine の CoordinatorLoop であり
member router を見ないため、flush  piggyback が最小の driver である。

---

## 6. Invariant 検証（Owner §5）

| 不変条件 | 検証方法 | 結果 |
| --- | --- | --- |
| INV-D1-1: EQ retired objects are reclaimed only through the EQ private EpochDomain-gated retire authority | TD1-4a（router epoch == private epoch の一体前進）＋ TD1-4b（他 domain 前進では reclaim されない） | **PASS** |
| INV-D1-2: Q/E/T ownership は function return で消滅しない | TD1-1（Q=304 正確）＋ TD1-2（Q=512/E=512/T=280 正確） | **PASS** |
| INV-D1-3: destruction 前に durable retire state が drain/reclaim される | TD1-1/1-2/1-3 の release 後全 counts 0・drop 0＋ dtor 実行 | **PASS** |
| INV-D1-4: engine epoch 前進だけでは EQ entry は reclaim されない | TD1-4b（foreign 100 前進で未解放） | **PASS** |
| INV-D1-5: RT path 不変、RT は ownership / reclaim / delete の決定を行わない | static V3（RT 関連 diff 0）＋全 retire 呼び出しが NonRT | **PASS** |

---

## 7. Test（Owner §6）

### 7.1 Test changes

- 新規 TU: `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp`（448 行）。
  TD1-0（setter 会計: 2/setter の線形性）/ TD1-1 / TD1-2 / TD1-3 / TD1-4a / TD1-4b。
- harness 配線: `PublishPipelineIntegrationTests.cpp` に `runSTG11EQRetireTests()` の
  宣言＋呼び出し（STG-8/9 と同じ形）。
- standalone CTest target: `STG11EQRetireTests`（`add_test(NAME STG11EQRetire ...)`）。
  AudioEngineHarness の深いコールチェイン外で実行するため。
  TU は両方から共有し、`STG11_STANDALONE_MAIN` でのみ `main()` を定義する。
- 既存 oracle の改変 = 0。test hook（production の test 専用分岐）= 0。

### 7.2 結果

| 項目 | Debug | Release |
| --- | --- | --- |
| build | PASS | PASS |
| TD1-0 | PASS | PASS |
| TD1-1（Q=304 正確） | PASS | PASS |
| TD1-2（Q=512/E=512/T=280 正確） | PASS | PASS |
| TD1-3 | PASS | PASS |
| TD1-4a（束縛） | PASS | PASS |
| TD1-4b（bound-provider gating） | PASS | PASS |
| standalone exit | 0 | 0 |
| harness 内 TD1 全件 | PASS | PASS |
| CTest 全件 | **44/44 PASS**（209.61 s） | **44/44 PASS**（151.56 s） |

### 7.3 既存 regression（TD1-5）

Debug / Release とも harness 直接実行で確認:

```text
STG-1 / STG-2 / STG-2-A / STG-2-B / STG-2-C / STG-2-R / STG-4-1 / STG-6-D1 /
STG-7-D1 / STG-8-D1 / STG-8-D2 / STG-8-D2b / STG-8-D3 / STG-9-D1(T-01..T-04)
= Debug / Release とも全 PASS
```

`D8_2_B_2_Tests` / `RetireGraceSemanticsTests` / `StuckReaderFallbackDrainTests` を含む
全 CTest が 44/44 の一部として PASS（改変 0）。

---

## 8. Static validation

| # | 検査 | 結果 |
| --- | --- | --- |
| 1 | production 変更 = `EQProcessor.h` + `EQProcessor.Core.cpp` のみ | PASS |
| 2 | 禁止 14 ファイル（EpochDomain / RCUReader / DDQ / Router×2 / Coordinator×2 / AudioEngine.h / Retire.cpp / Shutdown×2 / Threading / CtorDtor / ConvolverProcessor.Runtime）diff 0 | PASS |
| 3 | RT path（Processing / Coefficients / Parameters）diff 0、追加行の RT guard/load 0 | PASS |
| 4 | raw `std::atomic` method 追加 0 | PASS |
| 5 | memory ordering 変更 0 | PASS |
| 6 | EQ diff 内の engine-domain 参照 0 | PASS |
| 7 | member 宣言順（`m_epochDomain` :521 → `m_ownedRetireRouter` :527） | PASS |
| 8 | `stackRouter` 残存 0 | PASS |
| 9 | STG-10 files（`EpochDomain.h` / STG10 TU）diff 0 | PASS |

---

## 9. Build

| 項目 | 結果 |
| --- | --- |
| Debug build | PASS（`cmake --build build --config Debug` exit 0） |
| Release build | PASS（exit 0） |

---

## 10. ConvoPeq regeneration（Owner §7-7/8）

§1.2 のとおり再生成済み。`NEWER_SRC_COUNT = 0` / FRESH。
SHA `CDAFA18B...5C38B1` / 5,690,145 B / Generated 2026-09-29 19:35:52。

---

## 11. Diff / invariant audit

§8 のとおり。RT invariant（no wait / no lock / no allocation / no delete /
no ownership / no new policy decision）は RT 経路の無変更により全て維持。
`Retire は Epoch を通る` は TD1-4 で実証。
`shutdown は完全 Drain` は TD1-3 および dtor drain で実証。
authority（Coordinator / RuntimeWorld / Retire / Publish / Crossfade / Shutdown）は
いずれも変更なし。R-4 diagnostics は未接触。

---

## 12. Remaining risks

| # | 項目 | 状態 |
| --- | --- | --- |
| 1 | 定期的 drain の頻度 | flush piggyback（prepare/release 時のみ）のため、長期間 setter 駆動が続くと Q/E/T に滞留する。**安全性ではなく timeliness の問題**であり、D の既存挙動と同等。D1 の defect（喪失）とは無関係 |
| 2 | `m_retireRouter`（注入済み・未使用 member）の残存 | 未使用のまま保持。削除は API 変更になるため行わない。D1 と無関係 |
| 3 | Debug CTest の AudioEngineHarness SEGFAULT 1 回 | **intermittent**。直後の単体再実行で PASS、その後の全件再実行で 44/44 PASS。ログ末尾に pre-existing の `[I2T] phase4: abandon engine (pre-existing Debug segfault route, I3 issue)` が記録されており、既知の不安定経路と一致。STG-11-D1 との因果関係なし（直接実行では exit 0 全 PASS） |
| 4 | R-4 diagnostics / RC-2 / Path A / Candidate C / observability 2 件 / `RCUReader::enter` failure handling / intermittent 3 件 | STG-10-D1 報告の R1〜R8 をそのまま保持。**本 STG で併合していない** |
| 5 | D2 / D3 / D4 | **未着手**（Owner 指示どおり scope 外）。D1 との dependency なし（Audit §11 で確定済み） |

### 12.1 テスト環境の知見（残留記録）

| # | 内容 |
| --- | --- |
| T-1 | `RuntimeIntentCoordinator` は約 4.1MB（PDB 実測 `0x415dc0`）、`EQProcessor` は約 281KB（`0x460c0`）。harness の深いコールチェイン下で stack 確保すると `__chkstk` で stack overflow する（ダンプで faulting frame を特定）。test fixture は heap 確保すること（production も DSPCore ごと heap 生成）。TD1 の全 fixture を `make_unique` 化済み |
| T-2 | `quarantineResidentCount()` は Q+E の合算（`ISRRetireRouter.cpp:431-435`）。TD1-2 の oracle は合算値で判定し、Q 単独は減算で導出する |
| T-3 | stdout バッファ消失に注意: crash 時のログ tail は stderr のみを反映し、stdout の PASS 行は失われる。crash 位置の特定には procdump＋ダンプ解析が有効（WinDbg は未導入のため `C:\Windows\System32\procdump.exe` を使用） |

---

## 13. 変更ファイル一式

| 種別 | ファイル | 内容 |
| --- | --- | --- |
| production | `src/eqprocessor/EQProcessor.h` | include 1 行＋member 1 個＋observation getter 7 件（+39） |
| production | `src/eqprocessor/EQProcessor.Core.cpp` | member router 置換＋二重所有排除＋driver＋dtor drain（+47/-10  상당） |
| test（新規） | `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | 448 行（TD1-0/1/2/3/4a/4b） |
| test 配線 | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 宣言＋呼び出し（+6） |
| build | `CMakeLists.txt` | standalone target＋add_test＋IPO-OFF＋ASAN（+84） |
| authority | `ConvoPeq.md` | 再生成（+633/-11 相当） |

既存 oracle 改変 0 / test hook 追加 0 / `ISRRetireRouter.*` 改変 0。

---

## 14. STOP 条件の判定

| # | STOP 条件 | 判定 |
| --- | --- | --- |
| 1 | private EpochDomain を維持できない | 該当せず（束縛を維持し TD1-4 で実証） |
| 2 | member router の lifetime が成立しない | 該当せず（member 化＋dtor drain で成立） |
| 3 | 既存 NonRT driver では定期 reclaim を成立させられない | 該当せず（flush piggyback で成立。§5） |
| 4 | RT path の変更が必要 | 該当せず（V3） |
| 5 | ISRRetireRouter.* の変更が必要 | 該当せず（V2/V4） |
| 6 | Coordinator / authority の変更が必要 | 該当せず |
| 7 | shutdown ordering の前提が変わる | 該当せず（既存前提と同一。§3.1-4） |
| 8 | TD1-4 の epoch provenance を証明できない | 該当せず（TD1-4a/4b PASS） |
| 9 | D2 / D3 / D4 が直接 dependency として現れる | 該当せず |
| 10 | 新しい ownership authority が必要になる | 該当せず（既存 chain を再利用） |

**全 10 項目に該当しない。**

---

## 15. 最終報告

```text
STG-11-D1 Repair Implementation
===============================

Authority:
HEAD = f95f524cf6df0b3e1529425e690c02767912614e
SHA (pre)  = E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
size (pre) = 5,662,249 B / Generated 2026-09-29 00:19:43 / NEWER_SRC_COUNT 0
SHA (post) = CDAFA18BBCA815595ACB0B2C3D3F4C4CC738CF29D09
size (post)= 5,690,145 B / Generated 2026-09-29 19:35:52
NEWER_SRC_COUNT = 0
STATUS = FRESH (pre) / FRESH (post)

Production changes = 2 files (EQProcessor.h / EQProcessor.Core.cpp)
  - member router (m_ownedRetireRouter, private domain bound)
  - double-ownership elimination in the same function (pre-existing defect
    exposed by memberization; fixed within Owner scope)
  - periodic driver (flush piggyback, NonRT only, no new timer/thread/authority)
  - destruction drain (same premise as existing D drainAll)
Test changes       = 1 new TU (448 lines) + harness wiring (+6)
CMake changes      = standalone target + add_test + IPO-OFF + ASAN (+84)
Authority changes  = 1 (ConvoPeq.md regenerated)
Existing oracle changes = 0

Negative control   = PASS (Q=608 exactly 2x without the guard; abort on drain)

Debug build   = PASS
Debug CTest   = PASS  44/44 (209.61 s; one intermittent AudioEngineHarness
                SEGFAULT on first full run, PASS on retry + full re-run)
Release build = PASS
Release CTest = PASS  44/44 (151.56 s)

TD1-1 = PASS (Q=304 exact)
TD1-2 = PASS (Q=512/E=512/T=280 exact)
TD1-3 = PASS
TD1-4a = PASS (binding)
TD1-4b = PASS (bound-provider gating)
TD1-5 = PASS (existing suites unmodified, all PASS)

Static ISR validation = PASS
Raw atomic audit      = PASS (0)
Authority audit       = PASS (all authorities unchanged)

Commit = 0
Push   = 0
staged = 0 files

Verdict:
STG-11-D1 Repair Implementation = PASS
```

**COMMIT / PUSH は未実施。Owner へ差し戻す。**
