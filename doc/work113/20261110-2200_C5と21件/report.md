# C5 記録追加・C3 回帰テスト・計数窓外 21 件の特定（2026-11-10）

## §0 結論

1. **`retireCurrentAndTarget` の限定パッチを実装した。D1 の `unmatched` は 0 になった。** 3 シナリオ（EQ 変更、bypass 40 回、bypass 60 回）のすべてで `unmatched = 0`・`pending = 0` である。
2. **C3 の回帰テスト T11 を追加した。** 実 `SnapshotCoordinator`・実 `EpochDomain`・実 `DeferredDeletionQueue`・実 recorder を使い、`advanceFade` と `tryCompleteFade` で C3 を駆動して要求 1 件・完了 1 件を確認する。quarantine 経由でも記録が重複しないことを併せて確認した。`D8_2_B_2_Tests` で PASS している。
3. **計数窓外の 21 件をポインタ単位で特定した。** 対象は `EQProcessor::EQState` 21 個で、`~EQProcessor` が退役させる。要求記録は shutdown 相のため拒否され、完了は同じ `~EQProcessor` の `drainAll` で起きる。重複完了は無い。
4. **C2 は Non-RT 専用案の確定設計を提出した**（`c2_nonrt_design.md`）。実装はしていない。T の確保失敗時は `std::terminate` になる既存契約であることを確認し、記載した。
5. **記録失敗の検出手段を追加した。** `recordMisses` をレコーダに追加し、D1 の JSON に出力する。`evaluateRetireDelay` はバイト一致のままである。

## §1 C5: `retireCurrentAndTarget` の限定パッチ

### 1.1 変更内容

`SnapshotCoordinator::retireCurrentAndTarget()` の current と target の退役直前に、要求記録を 1 回ずつ追加した。差分は `patches/01_c5_retire_current_and_target.patch`（+9 行）。

```cpp
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap) {
            // ★ C5 (2026-11-10): 退役の直前に要求を記録する（完了フックと同じ object で対応）。
            //   格納失敗時は quarantineRetireSink が Q→E→T で必ず受領するため取り消さない。
            convo::isr::RetireLatencyRecorder::instance().recordRequest (
                snap, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu);
            if (!enqueueWithRetry(*m_epochProvider, snap, deleter, retireEpoch))
                quarantineRetireSink(snap, deleter, retireEpoch, "retireCurrentAndTarget:queueFull");
        }
```

target 側も同じ形である。

`recordRequest` の戻り値は使わない。失敗時も退役の流れを変えない（Owner 条件）。記録できなかった場合は §1.3 の `recordMisses` で検出する。

### 1.2 include 構造と循環依存の確認

| 項目 | 結果 |
|---|---|
| 追加した include | `SnapshotCoordinator.h` に `#include "../audioengine/ISRRetireLatency.h"` |
| 相対パスにした理由 | `SnapshotCoordinator.h` は `core/` 起点と `src/` 起点の両方から include される。相対指定なら `-I` に依存しない |
| `ISRRetireLatency.h` の依存 | `<atomic> <chrono> <cstddef> <cstdint> <limits> <vector>` のみ。`convo` の他ヘッダを include しない |
| 循環依存 | なし（`ISRRetireLatency.h` は `SnapshotCoordinator.h` を参照しない） |
| ビルド影響 | ConvoPeq 本体と 24 のテストターゲットが再ビルドされ、すべて成功 |

### 1.3 記録失敗の検出（Owner 要求）

`RetireLatencyRecorder` に `recordMisses_` を追加した。`recordRequest` / `recordRequestRT` が **null 以外の object に対して 0 を返した**とき（shutdown 相・ID 枯渇・スロット満杯）に 1 増える。既存の `postShutdown_` / `idExhausted_` / `overflow_` はそのまま残す。

D1 の JSON に `"recordMisses"` を追加した（`runtime` と `final` の両段）。差分は `patches/02_recorder_record_misses.patch` と `patches/03_d1_json_record_misses.patch`（+2 行）。

`evaluateRetireDelay` の判定には使わない（バイト一致を維持するため。§3.4）。したがって `recordMisses > 0` は**判定を変えずに読者へ知らせる**情報である。

> 制約: runtime stage が未捕捉で `evaluateRetireDelay` のフォールバックが走った場合、そのローカル `stage` の `recordMisses` は 0 のままになる。フォールバックは通常発生しない（実測でも runtime stage は常に捕捉されている）。判定値には影響しない。

### 1.4 必須検証（8 項目）

| # | 検証 | 結果 | 根拠 |
|---|---|---|---|
| 1 | current の退役で要求・完了が対応する | 合格 | アプリ EQ 実行で `queue` が 100 → 101 に増え、`unmatched = 0`。T11 の後始末も同じ経路 |
| 2 | target が存在する場合も要求・完了が対応する | 合格 | T12: fade 完了前に `finalizeShutdown` を呼び、current と target の 2 件を退役させ、要求 2・完了 2・`unmatched=0` を確認 |
| 3 | 同一ポインタに既存記録がある場合、二重記録にならない | 合格 | ハーネス `[R]` と shutdown 分岐 `[S]`（`created=false` で要求が増えない） |
| 4 | queue 失敗から quarantine に移送されても完了との対応が維持される | 合格 | ハーネス `[Q]`、T11 の (2)（要求 1・完了 1・`unmatched=0`） |
| 5 | D1 `unmatched = 0` | **合格** | 3 シナリオすべてで 0（§2.1） |
| 6 | world `unmatched = 1` の維持 | 合格 | 3 シナリオすべてで 1（bootstrap 由来） |
| 7 | `evaluateRetireDelay` のバイト一致 | 合格 | 2,756 B・SHA-256 `20346AD1C2B6597EDE74FAFF90201F42` |
| 8 | 既存 60 テストの合格 | 合格 | 60/60（T11・T12 は `D8_2_B_2_Tests` 内のケースとして追加）。§2.3 |

## §2 検証

### 2.1 アプリ実行（C5 適用後・最終ビルド）

| シナリオ | requests | completions | unmatched | pending | sync | drain | drainAll | queue | recordMisses |
|---|---|---|---|---|---|---|---|---|---|
| EQ 変更（12 秒） | 187 | 187 | **0** | 0 | 1 | 1 | 84 | 101 | 0 |
| bypass 40 回 × 250 ms | 1,882 | 1,882 | **0** | 0 | 1 | 1 | 924 | 956 | 0 |
| bypass 60 回 × 2 ms | 627 | 627 | **0** | 0 | 1 | 1 | 315 | 310 | 0 |

別ビルドでの先行実行も同じ結果だった（EQ 186→187、bypass40 1,888 件・`unmatched` 0、bypassfast 586 件・0）。`requests` の総数は実行ごとに数 % 動くが、`unmatched` は 0 で安定している。

world は 3 シナリオとも `commits = destroys`、`unmatched = 1`（bootstrap 由来の既知値）である。

C5 適用前との比較（EQ シナリオ）: 186 件・`unmatched` 1・`queue` 100 → **187 件・`unmatched` 0・`queue` 101**。増えた 1 件が C5 の記録で、完了は queue 経路で対応している。

`retire_delay` の判定は `latency_exceeds_threshold` で fail のままである。これは queue 経路の最大待ち時間（4.4〜13.9 秒）によるもので、C5 の追加分（`sync`/`drain` 各 1 件、マイクロ秒台）は最大値に影響しない。この判定は本変更の対象外である。

### 2.2 C3 の回帰テスト（T11）

`src/tests/D8_2_B_2_Tests.cpp` に `test_T11_CompleteFadeRecords()` を追加した（`patches/04_t11_complete_fade_test.patch`）。

| 手順 | 内容 |
|---|---|
| 実体 | 実 `SnapshotCoordinator`・実 `EpochDomain`（実 `DeferredDeletionQueue`）・実 `ISRRetireRouter`（退避先）・実 `RetireLatencyRecorder` |
| C3 の駆動 | `startFade(a)` → `startFade(b)` → `advanceFade(1000)` → `tryCompleteFade()` |
| 正常系の確認 | 要求 +1、drain 後に「要求 = 完了」「`unmatched = 0`」「`cancelled = 0`」「`recordMisses = 0`」 |
| 後始末 | `finalizeShutdown` → `retireCurrentAndTarget`（C5 経路）で current を退役。ここでも要求 = 完了・`unmatched = 0` |
| quarantine 系 | `enqueueRetire` を常に拒否する provider で C3 を駆動し、Q が受領。要求 +1 のまま（重複記録なし）、drain 後に完了 +1・`unmatched = 0` |

実行結果は `failures=0`、`D8-2-B-2 Tests PASS`（`evidence/d8_t11_stdout.txt`）。

さらに `test_T12_FinalizeRetiresTargetAndCurrent()` を追加した。fade を完了させずに `finalizeShutdown` を呼ぶと、current と target の両方が `retireCurrentAndTarget` で退役する。要求 2 件・完了 2 件・`unmatched 0`・`recordMisses 0` を確認する。これが C5 の 2 か所（current 側と target 側）を直接検証する唯一のケースである。

### 2.3 ctest

```
100% tests passed out of 60
Total Test time (real) = 351.22 sec
ctest_exit=0
```

`AudioEngineHarness` を含む 60 ターゲットすべて合格した。T11・T12 は `D8_2_B_2_Tests` 内のケースとして実行され、同ターゲットは `D8-2-B-2 Tests PASS` で終了している。テストターゲット数は 60 のまま変わらない。

### 2.4 `evaluateRetireDelay` の不変

| 項目 | 値 |
|---|---|
| バイト数 | 2,756（実装前後で一致） |
| SHA-256 | `20346AD1C2B6597EDE74FAFF90201F42…`（一致） |

## §3 計数窓外の 21 件（ポインタ単位の特定）

### 3.1 時系列

プローブ（`[P] REPORT_BEGIN` / `REPORT_END` / 退役サイト / 完了 / 未対応）を一時的に入れ、追加操作なしのシナリオを実行した。stderr の並びは次のとおりである（`evidence/probe21b_stderr.txt`）。

```
line 652  [P] REPORT_BEGIN
line 653  [P] REPORT_END
line 654〜 EQ.retire ×21（各 4 行: EQ.retire → B.coordinator → A2.enqueueWithRetry → A.enqueueRetire）
line 738〜 [P] UNMATCHED ptr=<同じ 21 個> path=drainAll
```

- レポート生成（652 行）より前に未対応の完了は **0 件**。
- レポート生成後に EQ の退役が 21 件あり、その 21 ポインタが順序まで一致して未対応の完了になる。

### 3.2 対象・経路・deleter

| 項目 | 特定結果 |
|---|---|
| 対象 | `EQProcessor::EQState` 21 個（`~EQProcessor` が保持していた状態） |
| 退役元 | `EQProcessor::~EQProcessor()`（`src/eqprocessor/EQProcessor.Core.cpp:156`）→ `retireEQStateDeferred(oldState)`（同 :160） |
| 要求経路 | `EQProcessor::enqueueDeferredDeleteWithFallback` → `m_retireCoordinator->enqueueRetire`（B）→ `router.enqueueWithRetry`（A2）→ `enqueueRetire`（A） |
| deleter | `deleteEQStatePtr`（`delete static_cast<EQProcessor::EQState*>(p)`、同 :22） |
| 完了経路 | `~EQProcessor` の `m_epochDomain->drainAll()`（同 :187）→ `DeferredDeletionQueue::drainAll` → `path=drainAll` |
| 記録が無い理由 | レコーダは `AudioEngine::drainAll` の先頭（`markShutdownPhase`）で shutdown 相に入っている。以降の `recordRequest` は 0 を返し、`postShutdown_` と `recordMisses_` に数えるだけである |
| 記録試行回数 | 21 対象 × 2 回 = 42 回（B の初回と、QueueFull 後の A2 再試行）。プローブの `POSTSHUTDOWN_RECORD` は 42 件で、集合は 21 個と一致 |
| レポート内の値 | `postShutdownRequests` と `recordMisses` はレポート生成時点で 0。42 回の拒否はレポート生成後に起きるため、どちらの計数にも現れない |

### 3.3 重複完了・再計上の確認

| 確認 | 結果 |
|---|---|
| 21 ポインタのうち、対応する完了（`COMPLETE`）が以前にあったもの | **0 件** |
| 21 ポインタの重複 | **0 件**（すべて 1 回だけ未対応として計上） |
| レポート内の `unmatched` との混在 | なし（レポートは生成時点の値 0 を記録し、その後は更新されない） |

したがって 21 件は「二重完了」ではなく「記録されなかった退役の完了」である。

### 3.4 計数窓の整理（提案・未実装）

現在のレポートは `~AudioEngine` の本体で、`m_retireRouter->drainAll()` の直後・メンバ破棄の前に生成される。この位置では次の 2 種が窓の外に出る。

| 種別 | 例 | 現在の扱い |
|---|---|---|
| shutdown 相での退役（記録拒否） | `~EQProcessor` の EQState 21 件 | `postShutdown_` / `recordMisses_` に計上。レポート生成済みのため JSON に出ない |
| その完了 | 同上の drainAll 21 件 | `unmatchedCompletions_` に計上。同じく JSON に出ない |

窓の設計案は 3 つある。いずれも判定式の変更を伴うため、実装は Owner 判断を待つ。

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| W-1 | レポート生成をメンバ破棄の後（別オブジェクトのデストラクタ or 明示 finalize）へ移す | 終了処理全体を 1 つの窓にできる | 生成主体の移動が必要。レポートが「エンジン生存中の状態」でなくなる |
| W-2 | 第 3 段（`postTeardown`）を追加し、終了処理完了時にもう一度捕捉する | 既存の窓を壊さない。`runtime`/`final`/`postTeardown` の 3 段で比較できる | JSON と読み手の解釈が増える |
| W-3 | 「shutdown 相の要求」と「その完了」を別勘定にし、`unmatched` の対象から外す | 意味が明確（測定対象外を混ぜない） | `recordComplete` 側で要求の有無を判定する必要があり、レコーダの構造変更になる |

現時点の推奨は **W-2** である。既存の判定ロジック（`evaluateRetireDelay`）に触れず、`unmatched = 0` の判定対象を変えないまま、窓の外の件数を観測できる。

## §4 C2 Non-RT 専用の確定設計

`c2_nonrt_design.md` に分離した。要点は次のとおり。

- C2 は Non-RT 専用とする。`jassert(!isAudioThread())` と Release 用カウンタで契約違反を検出する。
- 成功時は D、失敗時は Q→E→T。移送が確定するまで呼出元が所有権を持つ。
- 記録は移送する場所で 1 回だけ行い、取消し API は使わない（移送するため完了が必ず来る）。
- T の `std::vector` 確保失敗は `std::terminate` になる。これは Q→E→T を使う既存経路すべてに共通する契約であり、C2 で新しい失敗モードは増えない。
- 呼出元は本番 0・テスト 2・ハーネス 1 で、いずれも Non-RT である。

## §5 変更ファイルとロールバック

| ファイル | 変更 | 行数 |
|---|---|---|
| `src/core/SnapshotCoordinator.h` | C5 の記録 2 か所、include 1 行 | +9 |
| `src/audioengine/ISRRetireLatency.h` | `recordMisses` の追加（カウンタ・Snapshot・StageStats・reset） | +11 |
| `src/audioengine/ISRHB.cpp` | `recordMisses` の JSON 出力 | +2 |
| `src/tests/D8_2_B_2_Tests.cpp` | T11 の追加、main への登録 | +190 |

変更前スナップショットは `tmp/rollback/c5_before/`（4 ファイル、SHA-256 付き）と `tmp/rollback/probe2_before/`（プローブ対象 11 ファイル）に取得した。プローブ解除後、11 ファイルすべてが SHA-256 で一致することを確認済みである。

ロールバックは `patches/` の 4 パッチを逆適用する（`git apply -R`）。またはスナップショットから復元する。

commit／merge／release は行っていない。

## §6 残課題

1. **計数窓の外の扱い**（§3.4）。実装は未承認。
2. **C2 の実装**は未承認（設計のみ提出）。
3. `retire_delay` の判定 fail（queue 経路の最大待ち時間）は本変更の範囲外で、未解決のままである。
