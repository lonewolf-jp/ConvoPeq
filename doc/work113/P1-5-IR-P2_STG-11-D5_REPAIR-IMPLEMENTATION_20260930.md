# STG-11-D5 Repair Implementation Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-D5_REPAIR-IMPLEMENTATION_20260930.md`
- Work item: **STG-11-D5** — MMCSS RT 到達経路の残存 RT 契約違反修正（OBS-D4-1 / OBS-D4-2）
- Predecessor: `P1-5-IR-P2_STG-11-D5_REPAIR-CONTRACT-AUDIT_20260930.md`（GO）
- Date: 2026-09-30
- Commit / push: **未実施**（D1 + D2 + D3 + D4 + D5 を作業ツリーに保持）

---

## 1. Authority

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| 着手時 SHA-256 | `678B5BA15658A7AEDCED97D0E6A59EC2D3AF4C9BBE107350EF4710CC9B3DF57C` |
| 着手時 size / Generated | 5,805,842 B / `2026-09-30 21:09:22` |
| NEWER_SRC_COUNT / `--check` | 0 / FRESH |

完了後の再生成値は Gate §1 参照。

---

## 2. 実装（Contract Audit §4 どおり）

### 2.1 RT 側: lock-free atomic record のみ

`src/audioengine/AudioEngine.h` に private atomic member **8 件**を追加
（D4 の `successReportedCount_` の直後。D3 の 6 件・D4 の 7 件には触れていない）:

```cpp
// STG-11-D5: MMCSS registration/revert observation (RT to NonRT diagnostic transport). ...
std::atomic<bool> mmcssObserved_{false};            // (3) monotonic flag, last store
std::atomic<std::uint64_t> mmcssCount_{0};          // (2) monotonic + publication marker
std::atomic<std::uint32_t> mmcssKind_{0};           // (1) 1..5, last-wins
std::atomic<std::uint64_t> mmcssA_{0};              // (1) policy, last-wins
std::atomic<std::uint64_t> mmcssB_{0};              // (1) task selector, last-wins
std::atomic<std::uint64_t> mmcssC_{0};              // (1) priority or err, last-wins
std::atomic<std::uint64_t> mmcssD_{0};              // (1) taskIndex, last-wins
std::atomic<std::uint64_t> mmcssReportedCount_{0};  // NonRT-only once-only bookkeeping
```

private 宣言 2 件（D4 宣言の直後）:

```cpp
void recordMmcssEventObserved(std::uint32_t kind, std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) noexcept;
void reportMmcssEventIfRecorded() noexcept;
```

`AudioEngine.Mmcss.cpp` の recorder（**RT 到達**。D3 ordering correction と同一の順序。
定義は site と同一 TU に置き、D3/D4 の TU には触れない）:

```cpp
void AudioEngine::recordMmcssEventObserved(
    std::uint32_t kind, ...) noexcept // NOLINT(bugprone-easily-swappable-parameters)
{
    // (1) payload ×5 → (2) count(fetchAdd/acq_rel) → (3) observed(last store)
}
```

MMCSS 5 箇所の置換（`#if` guard は除去 — record は ON/OFF 両対応の RT-safe のため）:

| site | kind | payload |
| --- | --- | --- |
| primary success | 1 | policy, 0, priority, index |
| already registered | 2 | policy, 0, err, 0 |
| fallback success | 3 | policy, taskId(1/2), priority, index |
| FAILED | 4 | policy, 0, err, 0 |
| reverted | 5 | 0, 0, 0, 0 |

`attemptFallback` lambda に `taskId` 引数を 1 件追加（local のみの変更。
呼び出し順序 fallback1 → fallback2 は不変）。

**すべて `convo::` wrapper のみ**（raw API 追加 0）。
NOLINT の根拠は D3 OBS-3 / D4 と同一（5 引数無符号整数だが呼び出し 5 箇所は
第 1 引数リテラル `1..5`。D5 機能 test が 5 ラウンドで対応を一意に検証）。

### 2.2 NonRT 側: Mmcss.cpp の既存 backend で診断

`reportMmcssEventIfRecorded()`（Mmcss.cpp 内定義）は `convo::` wrapper で snapshot を読み、
当該 TU の既存 file-local `diagLog` に kind 別の原文言（+ `count=` suffix）で出力する。
`diagLog` 発行部は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で囲む。
task 名・priority 文字列・`[MMCSS-ASIO]` / `[MMCSS-DS]` tag は記録整数から NonRT で復元する
（RT では整数だけを写す）。未知 kind は抑止。
呼び出し点は既存 `timerCallback()`（D4 hook の直後）1 箇所のみ。新規 thread/timer/queue なし。

### 2.3 変更していないもの（§6 禁止物＋ Contract §5）

- MMCSS OS API の実行場所・回数・順序（`AvSet*` / `AvRevert*` の site 数は不変。
  T4 で実証：CharacteristicsW=1 定義＋tryTask 経由、Priority=2 箇所、Revert=1 箇所）
- `thread_local` 3 変数・same-thread revert・Message-Thread-flag 設計・`t_mmcssTried` guard
- return true ×6 / false ×4 の site（T5 で実証）
- fallback chain（1531 → fb1 → fb2 の順序。T5 で実証）
- error 分類（5/183/1552 → true。T5 で実証）
- D3 / D4 の関数・member・test・ordering・policy（T6 で read-only 実証）
- `diagLog` backend 自体（両 TU）
- `revertMmcssPriorityOnAudioThread()`（dead code）/ `finalizeMmcssShutdown()`（NonRT）

### 2.4 変更範囲（`git diff --numstat`、D1〜D4 込み）

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Mmcss.cpp` | 125 | 37 | **D5** |
| `src/audioengine/AudioEngine.Timer.cpp` | 195 | 23 | D3 + D4 + **D5**（hook 1 行） |
| `src/audioengine/AudioEngine.h` | 54 | 0 | D3 15 + D4 19 + **D5** 20 行 |
| `src/tests/AudioEngineHarness/STG11D5MmcssObservationTests.cpp` | 新規 | — | **D5**（untracked） |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 159 | 0 | D1 + D3 + D4 + **D5**（56 行） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 26 | 0 | D1 + D2 + D3 + D4 + **D5**（5 行） |
| `CMakeLists.txt` | 135 | 0 | D1 + D2 + D3 + D4 + **D5**（1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存・保持） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存・保持） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存・保持） |
| `ConvoPeq.md` | 4805 | 227 | 再生成 |

---

## 3. 不変条件

| ID | 担保手段 |
| --- | --- |
| INV-D5-1 / INV-D5-2 | M1〜M5 の旧文言が RT 関数内に不在＋backend token 不在＋recorder body 検査 |
| INV-D5-3 | record 5 件が guard 外。report 発行部が guard 内。OFF 時の直接 lock なし（D5-T3 後半） |
| INV-D5-4 | recorder body に新 authority token なし。輸送物は整数のみ |
| INV-D5-5 | D5 recorder 順序＋report 順序＋5 ラウンド機能 test |

設計契約 C-D5-OS1〜OS4（OS API 残置）は T4 が構造的に保証する。

---

## 4. テスト

`src/tests/AudioEngineHarness/STG11D5MmcssObservationTests.cpp`（新規。harness サブテスト）。

| Test | 方式 | 検証内容 |
| --- | --- | --- |
| **D5-T1/T2** | structural | 旧文言 5 種が RT 関数内に不在。両 body に backend token なし。record 5 件が guard 外 |
| **D5 recorder** | structural | payload 5 種 < count < observed、`observed` が最後、`count` は `fetchAddAtomic`、新 authority なし |
| **D5 report** | structural | snapshot が count acquire より後、全 read が `consumeAtomic` + order 明記、raw なし、`timerCallback` から呼ばれる |
| **D5-T3** | structural | 原文言 5 種が report 内＋guard 内＋backend 保持。OFF 時の直接 lock なし |
| **D5-T4** | structural | `thread_local` 3 変数維持、AvRevert が revert 関数内、`t_mmcssTried` reset 維持、Av* site 数不変（W=1/Prio=2/Revert=1）、Message Thread は flag のみ（Av* なし） |
| **D5-T5** | structural | once-guard 維持、`applyMmcssPriority()` 呼び出し維持、error 分類維持、fallback 順序維持（fb1 < fb2）、return 文言数 true=6/false=4（実測確定値） |
| **D5 機能** | deterministic | 5 ラウンド（kind 1〜5、payload 一意）。毎回 observed/count==k/snapshot==record #k/reported +1。無 record の report は不変 |
| **D5-T6** | structural（read-only） | D3/D4 の recorder 順序と 3 hook の存在が不変 |

期待値の確定方法：T4 の Av* 数と T5 の return 数は初回実行で不一致が出たため、
**実ソースを数えて期待値を修正した**（CharacteristicsW=1 は tryTask 経由のため定義のみ、
return true=6/false=4 は call-site 行を含む。いずれも実測で確定）.
推測で期待値を決めていない。

---

## 5. 実施した検証

| 検証 | 結果 |
| --- | --- |
| Debug / Release build | BUILD_EXIT=0 両方 |
| D5 sub-test（Debug / Release, harness 直接実行） | rc=0 / 全 PASS 両方 |
| **Negative control**（M5 を旧 `diagLog` へ一時復元） | D5-T1 `old wording '[MMCSS] reverted on Audio Thread' still inside RT function` → **FAIL**、rc=**1**。復元後は全 PASS（record 5 件の存在を再確認） |
| D4 / D3 sub-test（Debug / Release） | 全 PASS（D3-T3・D4-T2/T3 含む） |
| D1 回帰 | TD1-1 Q=304 / TD1-2 T=280、他 PASS（不変） |
| D2 回帰 | E=3 / T=3 / 0（不変） |
| full Debug / Release CTest | **45/45 PASS 両方** |
| raw `std::atomic` audit | D5 追加行に raw API **0**（`convo::` 21 箇所） |
| RT lock/allocation/backend audit | `recordMmcssEventObserved` 本体 **0 件** |
| authority audit | 新規 authority / 同期構造 **0 件** |
| clang-tidy（Mmcss.cpp） | finding **0**（NOLINT 適用済み） |
| cppcheck | project 135 件は既存ノイズ、**Mmcss.cpp 0 件** |
| ASAN | 環境 block（`0xC0000139`）。別問題として記録（PASS と扱わない） |

---

## 6. D5 scope 外の観測（**修正せず記録のみ**）

### OBS-D5-1: `revertMmcssPriorityOnAudioThread()` は caller なしの dead code

Timer.cpp:474 定義。`finalizeMmcssShutdown()`（NonRT）と役割重複の legacy alias と見られる。
STG-11 Discovery の `resetFadeStateAndRetireTarget` と同種の dead code。
削除は scope 外のため行わない。次回 Discovery 候補。

### OBS-D5-2: shell audit の `new` 誤検出パターン

`malloc|...|free\(|...\|\bnew\b` 系の grep はコメント文中の "nothing **new** to report"
に一致する。Gate の RT audit は recorder 本体に限定して 0 件を確認した。
test の token リストは `"new "`（trailing space）であり report/recorder の code には
存在しないため oracle に影響しない。記録のみ。

---

## 7. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**。Gate は `READY FOR COMMIT`。
- D1〜D4 の未 commit 変更はそのまま保持。pre-existing 変更には触れていない。
