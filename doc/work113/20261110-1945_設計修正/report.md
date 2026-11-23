# C4267 の適用結果と unmatched 設計の修正版（2026-11-10）

## §0 範囲

- ② C4267: 承認された独立パッチを適用し、コンパイル検証と警告の消失を確認した。追加のソース変更は行っていない。
- ③ unmatched: 指摘 2 点（取消しの識別、C2 の失敗時保証）を解消した設計修正版を提示する。**実装は行っていない**（承認待ち）。
- commit／merge／release は行っていない。

## §1 ② C4267: 適用と検証

### 1.1 適用

| 項目 | 値 |
|---|---|
| 対象 | `src/InputInspection.cpp` |
| 適用前 | 8,406 バイト、SHA-256 `C7259580…` |
| 適用後 | 8,836 バイト、SHA-256 `3811875F…` |
| 適用方法 | `patch -p0`（警告・fuzz なし） |

### 1.2 コンパイル検証

```
[1/3] Building CXX object CMakeFiles\ConvoPeq.dir\Release\src\InputInspection.cpp.obj
[2/3] Linking CXX executable ConvoPeq_artefacts\Release\ConvoPeq.exe
build_exit=0
```

**C4267 の発生件数: 0**（ビルドログ全体を検索）。置換前は 1 件だったため、警告は解消した。

### 1.3 回帰

`build/` で `ctest -C Release --output-on-failure` を実行した。結果は §1.4 に示す（本節は実行後に追記）。

### 1.4 ctest

```
100% tests passed out of 60
Total Test time (real) = 357.21 sec
ctest_exit=0
```

`AudioEngineHarness` を含む 60 件すべて合格した。C4267 パッチ適用による回帰は検出されていない。

### 1.5 挙動の差分

| 入力 | 変更前 | 変更後 |
|---|---|---|
| 〜8 MiB（全経路） | 走査する | 同じ（`numBytes ≤ INT_MAX` のため分岐しない） |
| 約 715 MB 超（現行では到達不能） | `int` へ切り捨て | `-1` を返し、呼出側は `fail` |

### 1.6 ロールバック

`patch -R` で逆適用するか、`tmp/rollback/item4/InputInspection.cpp`（置換前）から復元する。ただし現在のファイルは項目 4 の置換を含むため、`tmp/rollback` の内容は「置換前」ではなく「C4267 適用前」を使う必要がある。C4267 適用前の内容は SHA-256 `C7259580…` の状態であり、必要なら `patch -R` が確実である。

## §2 ③ unmatched 設計の修正版

### 2.1 指摘 1: 取消しが既存記録を消す可能性

**原因**: 提案 C は「ポインタ単位の冪等性」に依存していたため、B 経路が既存記録を再利用した場合でも `cancelRequest(ptr)` がその記録を消し得た。

**修正**: レコーダの API を次のように変え、**自分が新規作成した記録だけを取り消す**。

```cpp
// 戻り値: requestId（0 = 記録不能）。created が非 null なら、新規作成時のみ true。
std::uint64_t recordRequest (const void* object, std::uint64_t epoch,
                             std::uint64_t minReaderEpoch = 0,
                             std::uint32_t minReaderSlot = 0xFFFFFFFFu,
                             bool* created = nullptr) noexcept;

// 指定 ID の記録だけを取り消す（object と id の両方が一致し、状態が ready のときのみ）。
void cancelRequestById (const void* object, std::uint64_t requestId) noexcept;

// RT 経路用（TLS 区画内だけを走査する O(kRangeSize) 版）。
void cancelRequestRTById (const void* object, std::uint64_t requestId) noexcept;
```

`cancelRequestById` は次の 3 条件をすべて満たす場合に限りスロットを解放する。

1. スロットの `object` が引数と一致する。
2. スロットの `id` が引数と一致する（**他経路が作った記録は id が違うため消えない**）。
3. 状態が `ready` であり、CAS で `free` に遷移できた（競合時は何もしない）。

B 経路の手順は次のとおり。

```cpp
bool created = false;
const auto id = recorder.recordRequest (ptr, epoch, minEpoch, minSlot, &created);
const auto result = router.enqueueWithRetry (ptr, deleter, epoch, type);
if (result != RetireEnqueueResult::Success && created && id != 0)
    recorder.cancelRequestById (ptr, id);   // 自分が作った記録のみ
```

### 2.2 指摘 2: C2（`resetFadeStateAndRetireTarget`）の失敗時

**確認した事実**

- `resetFadeStateAndRetireTarget` の**本番呼出元は無い**（テスト専用。`src/tests/AudioEngineHarness/V11EngineRetireProtectionTests.cpp` と `src/tests/D8_2_B_2_Tests.cpp` のみ）。
- ヘッダの注記（`SnapshotCoordinator.h:150`）は「RT（updateFade）から呼ばれ得る」としており、**RT スレッドで実行され得る**前提で設計する必要がある。
- 実装（`SnapshotCoordinator.cpp:110`）は戻り値を**無視**している。

```cpp
const uint64_t retireEpoch = m_epochProvider->publishEpoch();
m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);   // 戻り値を無視
```

- `IEpochProvider::enqueueRetire` は `DeferredDeletionQueue::enqueue` の結果を返し、**false は「D キュー満杯で未格納」**を意味する。このとき所有権は呼出元に残るが、`target` はローカル変数であり、関数を抜けると**ポインタが失われる（リーク）**。

**設計（所有権の挙動は変えない）**

- 要求は `recordRequestRT`（RT 安全版）で記録する。
- `enqueueRetire` が false を返し、かつ自分が記録を作成した場合は `cancelRequestRTById` で取り消す。完了が永遠に来ない記録を残さないためである。
- リーク自体は**既存の欠陥**であり、本設計では修正しない（所有権・retry・reclaim を変えない制約）。別途、修正提案として扱う（§3）。

```cpp
bool created = false;
const auto id = recorder.recordRequestRT (target, retireEpoch, minEpoch, &created);
const bool stored = m_epochProvider->enqueueRetire (target, snapshotDeleter, retireEpoch);
if (! stored && created && id != 0)
    recorder.cancelRequestRTById (target, id);
```

### 2.3 全経路の失敗時一覧

| 経路 | 使用 API | 失敗の意味 | 所有権 | 記録の扱い |
|---|---|---|---|---|
| B `ISRRuntimePublicationCoordinator::enqueueRetire` | `router.enqueueWithRetry` | `QueueFull`／`Shutdown` | 呼出元に残る | 自分が作成した記録のみ `cancelRequestById` |
| C1 `startFade` | `enqueueWithRetry(*m_epochProvider, …)`（静的ヘルパ） | D 満杯 | `quarantineRetireSink` → Q→E→T で必ず受領 | 取り消さない（完了が来る） |
| C2 `resetFadeStateAndRetireTarget` | `m_epochProvider->enqueueRetire` | D 満杯 | 呼出元に残る（**現状はリーク**） | 自分が作成した記録を `cancelRequestRTById` |
| C3 `completeFade` | `enqueueWithRetry(*m_epochProvider, …)` | D 満杯 | `quarantineRetireSink` → Q→E→T | 取り消さない |
| C4 `quarantineRetireSink` | Q → E → T | 最終 T は常に受領 | 常に移転 | 追加記録なし（C1/C3 で記録済み） |
| D `retireRT` | `provider_->enqueueRetire` | false（未格納） | 呼出元に残る | 実装済みの `cancelRequest` を ID 版へ置換 |

### 2.4 C4 を追加対象外とする条件の確認

| 条件（Owner） | 確認結果 |
|---|---|
| 1. `quarantineRetireSink` の全呼出元が C1/C3 に限定される | **満たす**。呼出は `startFade`（`SnapshotCoordinator.cpp:78`）と `completeFade`（同 135）の 2 か所のみ。他に呼出元は無い |
| 2. C1/C3 のすべての経路で、quarantine 移行より前に要求記録が完了している | **設計で満たす**。記録は `startFade`／`completeFade` の先頭（`enqueueWithRetry` の前）に置く。quarantine はその後 |
| 3. 将来 C4 へ直接到達する経路が追加された場合の注意を残す | **設計に明記**。`quarantineRetireSink` の先頭に「直接呼ぶ場合は呼出元で `recordRequest` すること」というコメントを残す（実装時） |

したがって、C4 を独立した記録位置にしない判断は条件を満たす。

### 2.5 不変条件

- 所有権移転の条件式（`enqueueWithRetry`・`terminalReclaim`・`quarantine`・`emergencyQuarantine` の分岐）は変更しない。
- retry・reclaim の条件（`epoch < minReaderEpoch` など）は変更しない。
- D1 の判定式（`evaluateRetireDelay`）と 50 ms・`N_req = 59` は変更しない。
- 追加するのは記録呼び出しと、失敗時の取消し（ID 一致時のみ）だけである。

### 2.6 実装時の検証計画

1. 単体: 同一ポインタを B→A の順で記録したとき、`requests` が 1 で `created` が 2 回目に false になること。
2. 単体: `cancelRequestById` に誤った id を渡しても記録が消えないこと。
3. 単体: C1〜C4 の各経路で要求と完了が 1 対 1 になること（C2 の失敗時は取消し後に pending が増えないこと）。
4. アプリ: D1 の `unmatched` が 0、world の `unmatched` が 1（bootstrap）であること。
5. 回帰: 既存 60 テスト。
6. 判定不変: `evaluateRetireDelay` に差分が無いこと。

## §3 今回新たに判明した点（別提案・未修正）

1. **C2 の戻り値無視によるリーク**。`resetFadeStateAndRetireTarget` は `enqueueRetire` の失敗で snapshot を失う。D キュー満杯時のみ発生するが、所有権の欠落である。修正は所有権挙動の変更にあたるため、本設計から切り離し、別提案とする。
2. **C2 は本番呼出元が無い**。テスト専用であるため、現時点の `unmatched = 3` の原因では**ない**（原因は C1/C3 の非記録）。設計上は RT 安全版で実装する。
3. **C1/C3 の記録は NonRT**。`startFade` は `AudioEngine.Snapshot.cpp:158`（EQ 公開経路）、`completeFade` は `tryCompleteFade`（`AudioEngine.Timer.cpp:1100`）から呼ばれ、いずれも NonRT である。

## §4 成果物と保全

- 本報告書: `doc/work113/20261110-1945_設計修正/report.md`
- C4267 パッチ: `doc/work113/20261110-1920_C4267と設計/patch/c4267_size_guard.patch`（適用済み）
- 実装 4 件の差分・バックアップ: `doc/work113/20261110-1836_4件実装/patches/`、`tmp/rollback/manifest.txt`（維持）
