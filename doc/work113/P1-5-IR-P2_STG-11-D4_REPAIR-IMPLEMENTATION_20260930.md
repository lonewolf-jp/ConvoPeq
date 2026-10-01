# STG-11-D4 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D4_REPAIR-IMPLEMENTATION_20260930.md`
- Work item: **STG-11-D4 / OBS-1** — Diagnostics ON 時の `applyMmcssPriority()` success path の RT-unsafe logging 除去
- Predecessor: `P1-5-IR-P2_STG-11-D4_REPAIR-CONTRACT-AUDIT_20260930.md`（GO）
- Date: 2026-09-30
- Commit / push: **未実施**（Owner の個別 GO 待ち。D1 + D2 + D3 + D4 を作業ツリーに保持）

---

## 1. Authority（実ファイルから再取得・転記禁止）

着手時に実ファイルから取得:

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md`（Owner § の統一ファイル規定による） |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `F441875191E30FA864B06C0538AA4E5A2B775470E0FA49990A59F65E726CD4C0` |
| size | 5,761,584 B |
| Generated | `2026-09-30 01:42:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH`（exit 0） |

Owner 指示の authority 名 `ConvoPeq(20260930-111230).md` はリポジトリ内に存在しない
（`doc/ConvoPeqMD/` の実在は `.../20260930014222` を含む 10 スナップショット）。
D3 第 2 版の OBS-4 と同一 phenomenology のため、同様にルート `ConvoPeq.md` を authority とした。
古い snapshot は authority にしていない。

完了後の再生成値は Post-Implementation Gate §1 参照。

---

## 2. 実装（Contract Audit §3 / §4 どおり）

### 2.1 RT 側: lock-free atomic record のみ

`src/audioengine/AudioEngine.h` に private atomic member **7 件**を追加
（D3 の `affinityFailureReportedCount_` の直後。D3 の 6 件には触れていない）:

```cpp
// STG-11-D4: success observation (RT to NonRT diagnostic transport). ...
std::atomic<bool> successObserved_{false};            // (3) monotonic flag, last store
std::atomic<std::uint64_t> successCount_{0};          // (2) monotonic + publication marker
std::atomic<std::uint32_t> successKind_{0};           // (1) 1/2/3, last-wins
std::atomic<std::uint64_t> successA_{0};              // (1) payload, last-wins
std::atomic<std::uint64_t> successB_{0};              // (1) payload, last-wins
std::atomic<std::uint64_t> successC_{0};              // (1) payload, last-wins
std::atomic<std::uint64_t> successReportedCount_{0};  // NonRT-only once-only bookkeeping
```

private 宣言 2 件（D3 宣言の直後）:

```cpp
void recordSuccessObserved(std::uint32_t kind, std::uint64_t a, std::uint64_t b, std::uint64_t c) noexcept;
void reportSuccessIfRecorded() noexcept;
```

`AudioEngine.Timer.cpp` の recorder（**RT 到達**。D3 ordering correction と同一の順序）:

```cpp
void AudioEngine::recordSuccessObserved(
    std::uint32_t kind, std::uint64_t a, std::uint64_t b, std::uint64_t c) noexcept // NOLINT(bugprone-easily-swappable-parameters)
{
    // (1) payload
    convo::publishAtomic(successKind_, kind, std::memory_order_release);
    convo::publishAtomic(successA_, a, std::memory_order_release);
    convo::publishAtomic(successB_, b, std::memory_order_release);
    convo::publishAtomic(successC_, c, std::memory_order_release);
    // (2) count: monotonic AND the per-record publication marker.
    convo::fetchAddAtomic(successCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    // (3) observed: monotonic flag, always the last store.
    convo::publishAtomic(successObserved_, true, std::memory_order_release);
}
```

success 分岐の置換（3 箇所。`#if` guard は除去 — record は ON/OFF 両対応の RT-safe のため。
guard が不要になったことが INV-D4-3 の構造的証明の一部）:

- `if (nativeRtOk) { ... recordSuccessObserved(1, prio, procClass, savedClass); }`
- `else { recordSuccessObserved(2, audioMask, prevMask, 0); }`
- hetero `else { recordSuccessObserved(3, 0, 0, 0); }`

**すべて `convo::` wrapper のみ使用**（raw `std::atomic` API 追加 0）。
`::GetThreadPriority` / `::GetPriorityClass` の OS query 自体は RT 上に残るが、
これは既存の query であり logging backend ではない（変更しない）。

NOLINT の根拠（D3 OBS-3 と同一）: 4 引数は無符号整数で交換可能だが、
production の呼び出しは 3 箇所のみで第 1 引数にリテラル（`1` / `2` / `3`）を渡すため
交換は構造上起きない。加えて D4 機能 test は 5 ラウンドの各回で
`(kind, a, b, c)` の対応を一意に検証する。引数順序の変更はしていない。

### 2.2 NonRT 側: 既存 NonRT execution point から診断

`reportSuccessIfRecorded()` は `convo::` wrapper で snapshot を読み、
**既存** `diagLog` backend に kind 別の原文言で出力する。
`diagLog` 発行部は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で囲む
（現行の guard 位置と同一の意味。OFF 時は bookkeeping のみ進展し backend に到達しない）。
新規 timer / thread / worker / authority は**作っていない**。
呼び出し点は既存の `timerCallback()`（D3 の `reportAffinityFailureIfRecorded()` 呼び出しの
直後）1 箇所のみ。

```cpp
void AudioEngine::reportSuccessIfRecorded() noexcept
{
    if (!convo::consumeAtomic(successObserved_, std::memory_order_acquire))
        return;                                          // fast-out only
    const std::uint64_t count = convo::consumeAtomic(successCount_, std::memory_order_acquire);
    if (count == 0)
        return;
    const std::uint64_t reported =
        convo::consumeAtomic(successReportedCount_, std::memory_order_acquire);
    if (reported >= count)
        return;
    const std::uint32_t kind = ...;  // snapshot reads AFTER the count acquire
    const std::uint64_t a = ...; const std::uint64_t b = ...; const std::uint64_t c = ...;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    if (kind == 1) {
        diagLog("[NATIVE_RT] applied: win32Prio=" + ... + " count=" + ...);
    } else if (kind == 2) {
        diagLog("[AFFINITY] AudioThread pinned mask=0x" + toHexString(a) + ...);
    } else if (kind == 3) {
        diagLog("[AFFINITY] P/E cores: AudioThread affinity skipped (MMCSS Deadline QoS)" + ...);
    }
#endif
    convo::publishAtomic(successReportedCount_, count, std::memory_order_release);
}
```

成功ログの意味・内容は維持（3 文言と同一 + D3 と同じ `count=` suffix）。
`toHexString` 整形は NonRT 側で行う（RT では整数だけを写す）。

### 2.3 変更していないもの（Owner §4 / §5 遵守）

- `recordAffinityFailure()` / `reportAffinityFailureIfRecorded()` / D3 atomic member 6 件 /
  D3-T1〜T5 / D3 publication ordering / D3 failure policy — **すべて不変**
- `priorityApplied` の意味 / caller の return value handling / `t_mmcssTried` — **不変**
- `AudioEngine.Mmcss.cpp`（MMCSS 登録ログ）— **不変**（scope 外）
- `revertMmcssPriorityOnAudioThread()` / `finalizeMmcssShutdown()` — **不変**（scope 外）
- `diagLog` backend（`asyncSink` / `s_logMutex` / `LockFreeRingBuffer`）— **不変**
- Publish / Retire / Recovery / Coordinator authority — **不変**

### 2.4 変更範囲（`git diff --numstat`、D1/D2/D3 込み）

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Timer.cpp` | 191 | 23 | **D4**（D3 分を含む累積） |
| `src/audioengine/AudioEngine.h` | 34 | 0 | **D4** 19 行 + D3 15 行 |
| `src/tests/AudioEngineHarness/STG11D4SuccessObservationTests.cpp` | 新規 | — | **D4**（untracked） |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 103 | 0 | D1 + D3 + **D4**（53 行） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 21 | 0 | D1 + D2 + D3 + **D4**（5 行） |
| `CMakeLists.txt` | 134 | 0 | D1 + D2 + D3 + **D4**（1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存・保持） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存・保持） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存・保持） |
| `ConvoPeq.md` | 3634 | 234 | 再生成 |

D4 の production 変更は **2 ファイルのみ**（`AudioEngine.Timer.cpp` / `AudioEngine.h`）。
D4 の production hunk は `@@ -252`（success 分岐 3 箇所）、`@@ -311,0 +315,156`
相当（record/report 定義）、`@@ -927,0 +1087,9`（`timerCallback` hook 1 行）のみ。

---

## 3. 不変条件（implementation contract）

| ID | 内容 | 担保手段 |
| --- | --- | --- |
| INV-D4-1 | RT-reachable success logging shall not acquire a mutex. | S1/S2/S3 の brace 抽出＋旧文言の RT 関数内不在＋recorder body 検査 |
| INV-D4-2 | RT-reachable success logging shall not allocate or free memory. | 同上（`juce::String` / `new ` / `make_*` / `malloc` / `calloc` / `realloc` / `free(`） |
| INV-D4-3 | Diagnostics OFF shall not be required as a precondition. | (a) success 分岐は guard に関係なく禁止 token を検査、(b) record 呼び出し 3 件が guard 外、(c) body 内の残存 `diagLog`/`juce::String` が全件 guard 内 |
| INV-D4-4 | Success observation shall not create a new authority; ownership を輸送しない | recorder body に thread/timer/queue/publication/retire/recovery token が無いこと。輸送物は整数のみ |
| INV-D4-5 | Lossy-coalescing（payload → count → observed、count == N ⇒ record #N visible） | D4-T5a（recorder 順序）/ T5b（report 順序）/ 機能 test（5 ラウンド） |

---

## 4. テスト

`src/tests/AudioEngineHarness/STG11D4SuccessObservationTests.cpp`（新規）。
新規 CTest target は**作らない**（`AudioEngineHarness.exe` のサブテスト）。
harness main から `runSTG11D4SuccessObservationTests()` として呼び出す
（D3 の呼び出しの**前**に配置。D3 との依存はない）。

| Test | 方式 | 検証内容 |
| --- | --- | --- |
| **D4-T1** | structural | 旧 success 文言 3 種が RT 関数内に**存在しない**こと。S1 の `if (nativeRtOk)` block と S2/S3 の record 周辺 window に backend token がないこと。record 呼び出し 3 件が guard 外 |
| **D4-T2** | structural | `[NATIVE_RT] applied: win32Prio=` が report 内に存在＋guard 内＋`diagLog` backend 保持 |
| **D4-T3** | structural | `[AFFINITY] AudioThread pinned mask=0x`（+ `toHexString`）と `[AFFINITY] P/E cores` が report 内に存在＋guard 内 |
| **D4-T4** | structural | `applyMmcssPriority` body 内の残存 `diagLog`/`juce::String` が全件 guard 内。関数全体が mutex / SPSC sink を直接触らない |
| **D4-T5a** | structural | recorder の atomic access 順序（コメント除去後に列挙）。payload 4 種 < count < observed、`observed` が最後、`count` は `fetchAddAtomic`、`observed` は `publishAtomic`、新 authority なし |
| **D4-T5b** | structural | report の read 順序（snapshot が count acquire より後、全 read が `consumeAtomic` + `memory_order_acquire` 明記、raw atomic なし、`timerCallback` から呼ばれる） |
| **D4 機能** | deterministic | 5 ラウンド（kind 1→2→3→1→2、payload 一意）。毎回 `observed` / `count == k` / snapshot == record #k / `reportedCount` ちょうど +1。record なしの report は不変 |
| **D4-T6** | structural（read-only） | D3 の recorder 順序（payload → count → observed、`count` は `fetchAddAtomic`）と report 順序が不変。D3 ファイル・test・関数は**一切変更しない** |

OS 注入用の新しい production mechanism は作っていない。
preprocessor 判定は D3 と同一の条件付きスタック。順序判定は quote-aware コメント除去を先行。

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug build | BUILD_EXIT=0 |
| Release build | BUILD_EXIT=0 |
| D4 sub-test（Debug, harness 直接実行） | rc=0 / D4-T1, T2/T3, T4, T5a, T5b, 機能, T6 全 PASS |
| D4 sub-test（Release, harness 直接実行） | rc=0 / 同上 |
| **Negative control**（S2 を旧 `diagLog` へ一時復元） | D4-T1 `success wording '[AFFINITY] AudioThread pinned mask=0x' still inside RT function (ON success path reaches logging backend)` → **FAIL**、harness rc=**1**。⇒ oracle は欠陥を捕捉する。復元後は全 PASS |
| D3 sub-test（Debug / Release） | D3-T1/T1b/T2/T3/T5 全 PASS（**D3-T3 含む**。success 文言は report 内・guard 内に移動したため成立） |
| D1 回帰（`STG11EQRetireTests` Debug） | rc=0 / TD1-1 Q=304、TD1-2 D/Q/E/T = 4096/512/512/280、TD1-3/4a/4b PASS（不変） |
| D2 回帰（`STG11D2SnapshotRetireTests` Debug） | rc=0 / TD2-1 E=3、TD2-2 T=3、TD2-3 全 0（不変） |
| full Debug CTest | **45/45 PASS** |
| full Release CTest | **45/45 PASS** |
| raw `std::atomic` audit | D3+D4 追加行に raw API **0** |
| RT lock/allocation/backend audit | `recordSuccessObserved` 本体 **0 件** |
| authority audit | 新規 authority / 新規 synchronization 構造 **0 件** |
| clang-tidy（`AudioEngine.Timer.cpp`） | D4 hunk 内 finding **0**。残存 `bugprone-branch-clone` ×2 と `bugprone-unchecked-optional-access` ×6 はいずれも hunk 外の既存事象。`easily-swappable` は NOLINT 対応済み |
| cppcheck（project DB） | project 全体 135 件すべて既存ノイズ、**D4 production 2 ファイル 0 件** |
| ASAN | 環境 block（`0xC0000139`）。従来どおり別問題として記録 |

---

## 6. D4 scope 外で発見・観測した事項（**修正せず記録のみ**）

### OBS-D4-1: `AudioEngine.Mmcss.cpp` の MMCSS 登録ログは依然 RT 到達のまま

`[MMCSS-*] registered` / `already registered by JUCE/driver` / `FAILED` /
`[MMCSS] reverted on Audio Thread` は同一 first-call RT 経路から file-local `diagLog`
（`juce::Logger::writeToLog` 直接＋`juce::String` 構築）を呼ぶ。
Contract Audit §6 のとおり scope 外のため**触れていない**。次回 Discovery 候補。

### OBS-D4-2: `revertMmcssPriorityOnAudioThread()` / `finalizeMmcssShutdown()` の guarded `diagLog`

同じく scope 外のため**触れていない**。次回 Discovery 候補。

### OBS-D4-3: 復元直後の harness 実行で既知 flake 署名の早期 crash を 1 件観測

復元＋relink 直後の直接実行で rc=`0xC0000005`。ログは 28 行で
`[I2T] T-I2-3: gotWorld=1` の直後に停止しており、D4/D3 の test より遥か前の
test（I2T）で発生した。何も変えずに再実行すると rc=0・全 PASS（1179 行）。
production が fixed であること（record 呼び出し 3 件の存在）と build 正常を再確認済み。
D3 Gate の OBS-5（R8）と同一 phenomenology のため、**D4 への帰属は立証できない**。
未解決の既知 flake として記録し、R8 の調査対象に含めることを推奨する。

---

## 7. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Post-Implementation Gate は `READY FOR COMMIT`。
- D1 / D2 / D3 の未 commit 変更はそのまま保持している。
- pre-existing worktree 変更には触れていない。
