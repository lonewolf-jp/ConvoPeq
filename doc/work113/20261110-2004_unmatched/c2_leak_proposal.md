# 別提案: C2 の snapshot リーク修正（設計のみ・実装未承認）

## §1 問題

`SnapshotCoordinator::resetFadeStateAndRetireTarget`（`src/core/SnapshotCoordinator.cpp:99`）は、D キューへの格納結果を無視している。

```cpp
const uint64_t retireEpoch = m_epochProvider->publishEpoch();
m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch);   // 戻り値を使わない
```

`IEpochProvider::enqueueRetire` は格納結果を返す。

- `true` → キューが受領する。
- `false`（満杯）→ **未格納**。所有権は呼出元に残る。`target` はローカル変数である。関数を抜ければ参照が失われる。

つまり、満杯時に `GlobalSnapshot` が解放されずに消える。これがリークである。実測では未発生。キュー容量は 4096 で、通常の滞留は数十件にとどまる。それでも所有権の欠落である。

## §2 前提となる制約（コードから確認）

| 項目 | 事実 |
|---|---|
| C2 の本番呼出元 | **無い**（`src/tests/...` の 2 ファイルのみ） |
| C2 の RT 性 | ヘッダ注記（`SnapshotCoordinator.h:150`）は「RT（`updateFade`）から呼ばれ得る」としている |
| D キューの格納 | ロックフリー（`DeferredDeletionQueue::enqueue` は CAS のみ） |
| 隔離ストア（Q） | `std::mutex` を取る（`RetireQuarantineStore::quarantine`） |
| TerminalReclaimAuthority（T） | `std::mutex` を取る（`store` の `entries_.push_back`） |

したがって、**失敗時に Q や T へ流す案は、RT から呼ばれた場合にロックを持ち込む**ため、そのままでは採用できない。

## §3 案の比較

| 案 | 内容 | RT 安全 | 所有権 | 評価 |
|---|---|---|---|---|
| **A（推奨）** | SnapshotCoordinator 内に固定長の **再試行リング**（ロックフリー、例 32 スロット）を持ち、格納失敗時に ptr/deleter/epoch を積む。次の NonRT 機会（`startFade`／`completeFade`／Timer）で `quarantineRetireSink` へ移す | ○（CAS のみ） | リングが保持し、移送で移転 | 失敗を握りつぶさず、RT 制約も守れる。リング満杯時は「強制解放しない」方針でカウンタのみ増やす（既存の Q-full と同じ扱い） |
| B | C2 を NonRT 専用と明記（`jassert(!isAudioThread())`）し、失敗時は `quarantineRetireSink` へ | ○（NonRT 前提） | 移転 | 実装は小さいが、**RT から呼ばれ得る**という既存注記と矛盾する契約変更になる |
| C | D キューを growable にして失敗を無くす | ○ | 移転 | キュー実装の変更で影響が大きい。本件の範囲外 |

推奨は **A**。C2 が将来 RT から呼ばれる場合でもロックを持ち込まない。

## §4 案 A の設計

### 4.1 データ構造

```cpp
struct PendingSnapshotRetire { void* ptr; void (*deleter)(void*); uint64_t epoch; };
std::atomic<uint32_t> retryWrite{0};
std::atomic<uint32_t> retryRead{0};
PendingSnapshotRetire retryRing[32];   // 固定長・トリビアル
```

### 4.2 失敗時（RT 可）

1. リングに空きがあれば書き込み、`retryWrite` を release で進める。
2. 空きが無ければ `snapshotRetireOverflow` を 1 増やす（解放はしない）。

所有権はリングが保持する。**キューへ移るまで SnapshotCoordinator が所有者**であることを契約に明記する。

### 4.3 再試行（NonRT）

`startFade`／`completeFade`／Timer の先頭でリングを drain し、各要素を `quarantineRetireSink` へ移す（Q→E→T で必ず受領）。

### 4.4 shutdown

エンジンの shutdown 経路（Audio Thread 停止後）でリングを drain する。停止前に drain できなかった分は、shutdown の NonRT フェーズで `quarantineRetireSink` へ移す。**破棄はしない**（二重解放の回避）。

### 4.5 二重解放・リークの防止

- リングへの書き込みは「格納失敗を確認した後」に 1 回だけ行う（成功時は書かない）。
- drain は単一コンシューマ（NonRT）で、`retryRead` の CAS により 1 回だけ取り出す。
- リングから取り出した後に移転が失敗する経路は無い（Q→E→T は T が常に受領）。
- リング満杯時は解放しないため、リークは残るが**UAF は起こさない**。満杯は telemetry で検知する。

### 4.6 既存の所有権契約への影響

- `enqueueRetire` 成功時の契約（D キューが所有）は不変。
- 失敗時の契約を「呼出元が所有」から「SnapshotCoordinator の再試行リングが所有」へ**明示的に**変える。これは現在の「黙って失う」より弱い保証（リーク）から、より強い保証（後で移転）への変更である。
- 他のコンポーネント（AudioEngine・EQ・Router）の契約には触れない。

## §5 検証計画（実装時）

1. 単体: D キューを満杯にして C2 を呼び、リングに 1 件入り、次の drain で Q へ移ることを確認する。
2. 単体: リング満杯時に `snapshotRetireOverflow` が増え、解放されないことを確認する。
3. 単体: 二重 drain で同じポインタが 2 回移転しないこと。
4. shutdown: リングに残った状態で終了し、リークが無いこと（ASan は Windows でリーク検出不可のため、移送カウンタで確認）。
5. 回帰: 既存 60 テスト。

## §6 実装しない理由（今回）

- 所有権の契約変更であり、承認済みの「記録の追加・取消し」とは別種の変更である。
- C2 は本番呼出元が無く、現時点の実害は確認されていない。
- Owner の指示どおり、別提案として起票し、実装は承認を待つ。
