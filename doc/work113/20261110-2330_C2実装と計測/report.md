# C2 限定実装・W-2 改訂設計・`retire_delay` 追加計測（2026-11-10）

## §0 結論

1. **C2（Non-RT 専用）を承認範囲内で実装した。** RT 違反時は `jassert`・違反カウンタ・即時 return のみで、target／fade 状態／所有権／記録を一切変更しない。Non-RT では「1 回だけ記録 → D か Q→E→T へ移送」とし、取消しは行わない。テスト T13（8 ケース）が合格し、ctest は 60/60 である。
2. **W-2 の改訂設計（v2）を提出した**（`w2_post_teardown_design_v2.md`、実装なし）。Owner 推奨の案 A を採用し、基底クラス 6 つのデストラクタをソースで監査した。すべて「退役・完了を発生させない」ことを確認できたため、`postTeardown` を**メンバ破棄完了時点の最終捕捉**と定義し、`postProcess` を**保証ではなく追加の観測値**に格下げした。
3. **`retire_delay` の追加計測を実施した**（`retire_delay_measurement.md`、修正なし）。H-1（epoch 未公開）は棄却、H-2（epoch 未更新の入場経路）も主因ではない。原因は**publish より前に開始された `ScopedEpochReader` の区画が 4.4 秒保持された**ことである。
4. 計測コードは除去し、対象 5 ファイルの SHA-256 一致を確認した。

## §1 C2 の限定実装

### 1.1 変更内容

| # | 変更 | 差分 |
|---|---|---|
| 1 | `resetFadeStateAndRetireTarget` の Non-RT 専用化（RT 判定・違反カウンタ・即時 return・失敗時の Q→E→T 移送・記録 1 回・取消し廃止） | `patches/01_c2_nonrt_reset.patch`（+23/−9） |
| 2 | `s_rtViolationCount`（プロセス単位）と `rtViolationCount()` の追加、契約コメントの更新 | `patches/02_c2_counter.patch`（+13/−2） |
| 3 | 既存診断出力（`retireLatencyD1`）へ `snapshotRtViolations` を追加 | `patches/03_c2_diag_output.patch`（+4/−1） |
| 4 | テスト T13（8 ケース）とテスト用アクセサ（target スロット参照） | `patches/04_t13_c2_tests.patch`（+303/−1） |

RT 判定は次の形である。

```cpp
if (convo::numeric_policy::isAudioThread())
{
    jassert (false && "SnapshotCoordinator::resetFadeStateAndRetireTarget is Non-RT only");
    convo::fetchAddAtomic (SnapshotCoordinator::s_rtViolationCount, std::uint64_t{1}, std::memory_order_release);
    return;   // target / fade 状態 / 所有権 / 記録を一切変更しない
}
```

`#include "../DspNumericPolicy.h"` は `.cpp` 側だけに入れ、ヘッダ依存を増やしていない。

### 1.2 承認範囲との対応

| 承認項目 | 実装 |
|---|---|
| Non-RT 専用化 | `isAudioThread()` 判定を先頭に置いた |
| RT 判定時の `jassert`・カウンタ・即時 return | 上記のとおり |
| RT 違反時に target/fade/所有権/記録を変更しない | return のみ。T13 (5) で target ポインタ・fade 状態・要求数・滞留数が不変であることを確認 |
| Non-RT での再呼出しによる正常退役 | T13 (6) で要求 1・完了 1・`unmatched 0` |
| 成功・Q・Q→E・Q→E→T の所有権移送と記録 1 回 | T13 (1)〜(4) |
| RT 違反カウンタの既存診断出力への追加 | `retireLatencyD1.snapshotRtViolations` |
| 二重呼出し | T13 (7) で 2 回目の要求増加なし |

### 1.3 T13 の検証結果

```
[T13] C2 Non-RT only: success/Q/E/T/RT-violation/double-call...
  PASS: Non-RT only contract, Q/E/T chain, RT violation keeps state
D8-2-B-2 Tests PASS
```

| # | ケース | 判定の根拠 |
|---|---|---|
| 1 | Non-RT 成功 | 要求 +1、target スロットが null 化、drain 後 要求 = 完了・`unmatched 0`・`cancelled 0` |
| 2 | Q 移送 | `quarantineResidentCount` +1、要求 +1、drain 後 要求 = 完了 |
| 3 | Q 満杯 → E | Q を 512 件埋めた後、`emergencyQuarantineResidentCount` +1 |
| 4 | Q/E 満杯 → T | 両方 512 件埋めた後、T の resident +1 または同期破棄（`Path::Sync` +1）で受領を確認 |
| 5 | RT 違反 | target ポインタ不変・fade 状態不変・要求数不変・滞留数不変・`rtViolationCount` +1 |
| 6 | RT 違反後の Non-RT 再呼出し | 要求 +1、target が退役、drain 後 `unmatched 0` |
| 7 | 二重呼出し | 2 回目は要求が増えない |

ケース 4 で「resident が増えない」場合があるのは、`terminalReclaim` が epoch 安全なら同期破棄するためである。どちらの経路でも所有権は T に移るので、`Path::Sync` の完了数で受領を確認する形にした。

## §2 W-2 改訂設計（v2）

詳細は `w2_post_teardown_design_v2.md`。要点は次の 3 点である。

### 2.1 基底クラス監査（案 A）

| 基底 | デストラクタ | 退役を発生させるか |
|---|---|---|
| `juce::AudioSource` | `= default` | しない |
| `juce::ChangeBroadcaster` | 空 `{}` | しない |
| `juce::ChangeListener` | `= default` | しない |
| `ConvolverProcessor::Listener` | `= default` | しない |
| `juce::Timer` | `jassert` + `stopTimer()` | しない（`timerCallback` は呼ばれない） |
| `juce::AsyncUpdater` | 保留メッセージの破棄 | しない（`handleAsyncUpdate` は呼ばれない） |

監査の限界も明記した。`stopTimer()` と `cancelPendingUpdate()` はメッセージマネージャのロックを取り得る（ブロックはし得るが退役はしない）。JUCE 更新時は再監査が必要である。

### 2.2 段の位置づけの変更

- `postTeardown` = **最終捕捉**（メンバ破棄完了時点）。言語保証＋§2.1 の監査で根拠づける。
- `postProcess` = **追加の観測値**。Owner 指摘のとおり、静的破棄順は構築の逆順であり、レポータより先に構築された静的オブジェクトは後に破棄される。したがって「差分 0 = プロセス終了まで取りこぼしなし」とは主張しない。主張できるのは「レポータの破棄時点までに観測できた差分が 0」までである。

### 2.3 維持する項目

- 3 段の累積値と差分の併記。
- `recordMisses`（記録に失敗した**呼び出し回数**）、`postShutdownRequests`、`postShutdownCompletions` の区別。
- `evaluateRetireDelay` は `runtime` 段のみを使用。

## §3 `retire_delay` 追加計測の結果

詳細は `retire_delay_measurement.md`。

### 3.1 計測結果（停滞ドメインと健全ドメイン）

| ドメイン | `pub` | `global` | `minE` | `pend` | 最小 epoch の reader |
|---|---|---|---|---|---|
| A（停滞） | **1** | 2 | **1** | **20** | slot 0 / `ScopedEpochReader` / `e=1` / `d=1` / `ew=21` / `enter=21` |
| B（健全） | 3 → 87 | 4 → 88 | global−1 を追従 | 2 | slot 0 / `ISRRetireRouter` / epoch 追従 |

### 3.2 仮説の判定

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H-1: ドメインが epoch を公開しない | **棄却** | A の `pub=1`（publish は 1 回あり、global は 1→2 に進んだ） |
| H-2: epoch を更新しない入場経路がある | **主因ではない** | `enter=21` と `ew=21` が一致（入場のたびに epoch を書いている）。`resv=0`、`reacq=20` |
| 長時間の reader 区画 | **確定** | 停滞中 `enter` が増えない（= 新しい区画に入っていない）のに `d=1`。publish 前に開始した区画が保持され続けた |

### 3.3 原因

**publish より前に開始された `ScopedEpochReader` の区画が 4.4 秒保持され、その epoch（1）が `minReaderEpoch` を固定した。** その間に退役した 20 件（epoch=1）は reclaim できない。

`ScopedEpochReader` を EQ ドメインで取る箇所は、EQProcessor 内部の短い読み取り（6 か所）と、外部の 2 か所（`AudioEngine.EQResponse.cpp:92`、`AudioEngine.RebuildDispatch.cpp:706`）である。4.4 秒は後者に相当する。クラス単位のタグでは呼出し元までは特定できないため、次段階では guard の生成箇所ごとにタグを分ける。

## §4 検証

### 4.1 ctest

```
100% tests passed out of 60
Total Test time (real) = 336.97 sec
ctest_exit=0
```

T13 を含む `D8_2_B_2_Tests` が `D8-2-B-2 Tests PASS` で終了し、60 ターゲットすべてが合格した。

### 4.2 アプリ実行と診断出力

| シナリオ | requests | completions | unmatched | recordMisses | **snapshotRtViolations** | 最大 latency |
|---|---|---|---|---|---|---|
| bypass 40 × 250 ms | 1,849 | 1,849 | 0 | 0 | **0** | 4,407 ms |
| bypass 60 × 2 ms | 586 | 586 | 0 | 0 | **0** | 11,373 ms |
| EQ 変更 | 187 | 187 | 0 | 0 | **0** | 13,993 ms |

world は 3 シナリオとも `unmatched = 1`（bootstrap・既知）。C2 は本番呼出元が無いため、アプリの挙動は C2 実装の前後で変わらない（`unmatched 0`・`recordMisses 0`・`sync 1`・`drain 1` が一致）。新しい診断値 `snapshotRtViolations` は 0 で出力される。

### 4.3 `evaluateRetireDelay` の不変

| 項目 | 値 |
|---|---|
| バイト数 | 2,756（実装前後で一致） |
| SHA-256 | `20346AD1C2B6597EDE74FAFF90201F42…`（一致） |

## §5 変更ファイルとロールバック

| ファイル | 変更 |
|---|---|
| `src/core/SnapshotCoordinator.cpp` | C2 の Non-RT 専用化・移送・記録 |
| `src/core/SnapshotCoordinator.h` | 違反カウンタ・アクセサ・契約コメント |
| `src/audioengine/ISRHB.cpp` | `snapshotRtViolations` の出力 |
| `src/tests/D8_2_B_2_Tests.cpp` | T13 とテスト用アクセサ |

計測用の変更（`EpochDomain.h`、`IReaderEpochProvider.h`、`RCUReader.h`、`ScopedEpochReader.h`、`ISRRetireRouter.cpp`）は除去済みで、5 ファイルすべて SHA-256 一致を確認した。変更前スナップショットは `tmp/rollback/c2impl_before/`（9 ファイル）。

ロールバックは `patches/` の 4 本を逆適用する。commit／merge／release は行っていない。

## §6 残課題

1. W-2 の実装は未承認（設計 v2 を提出）。
2. `retire_delay` の修正は未承認。次段階は「区画の持ち主の特定（呼出し箇所ごとのタグ）→ 区画の短縮」の順。
3. `retire_delay` の判定 fail 自体は解決していない（計測のみ）。
