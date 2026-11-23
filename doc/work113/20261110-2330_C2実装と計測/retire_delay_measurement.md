# `retire_delay` 追加計測の結果（調査のみ・修正未承認）

Owner 承認（2026-11-10）の範囲で計測を追加し、H-1（epoch 未公開）と H-2（epoch 未更新の入場経路）を切り分けた。**EBR の安全性・所有権・reclaim 判定・publish 頻度・reader の入退場動作は変更していない。** 計測コードは計測後に除去した（§5）。

## §1 計測した内容

| 計測 | 実装（固定容量・atomic のみ） |
|---|---|
| reader タグ | `IReaderEpochProvider::registerReaderThreadWithTag(tag)` を追加（既定は無タグ版へ委譲）。`RCUReader` → `"RCUReader"`、`ScopedEpochReader` → `"ScopedEpochReader"`、`ISRRetireRouter` → `"ISRRetireRouter"` |
| ドメイン別 publish 計数 | `publishCount_`（`publishEpoch` のたびに +1）、`lastPublishUs_` |
| 経路別 reader 入場計数 | `enterPathCount_` / `reservePathCount_` / `reacquirePathCount_` |
| slot epoch 更新回数 | `slotEpochWrites_[i]`（`enterReader` が slot の epoch を実際に書き換えた回数） |
| ダンプ | 滞留があるときだけ 1 秒に 1 回、NonRT の `tryReclaim` から stderr へ 1 行 |

RT 経路には動的確保・mutex・ファイル出力・通常ログを追加していない（atomic の加算のみ）。ダンプは NonRT の `tryReclaim` からのみ実行し、滞留 0 のときは出力しない。

## §2 計測結果（bypass 40 回 × 250 ms、12 秒、1 実行）

停滞したドメインと健全なドメインがはっきり分かれた。

| ドメイン（役割） | `pub` | `global` | `minE` | `pend` | 最小 epoch の reader |
|---|---|---|---|---|---|
| A（停滞・EQ 系） | **1** | 2 | **1** | **20** | slot 0 / `ScopedEpochReader` / `e=1` / `d=1` / `ew=21` / `enter=21` |
| B（健全・router） | 3 → 87 | 4 → 88 | global−1 を追従 | 2 | slot 0 / `ISRRetireRouter` / e は epoch に追従 |

### 2.1 停滞ドメインの不変量

A では 4 つのサンプルすべてで次が一定だった。

- `pub=1`（publish は 1 回だけ）、`global=2`、`minE=1`、`pend=20`。
- `enter=21`、`ew=21`、`reacq=20〜21`。
- `d=1`（区画保持中）。

**`enter` が増えていない**ことが決定的である。すなわち停滞中に新しい区画へ入っていない。区画は publish より前に開始され、そのまま保持されている。

### 2.2 H-1（epoch 未公開）の判定 → **棄却**

A は `pub=1` であり、publish は 0 ではない。epoch は 1 → 2 に進んでいる。したがって「ドメインが epoch を公開しないため進まない」という説明は成立しない。

### 2.3 H-2（epoch 未更新の入場経路）の判定 → **主因ではない**

`ew=21` は、`enterReader` が slot の epoch を 21 回書き換えたことを示す。`reserveReaderThread` と `reacquireReaderSlot` の回数（`resv=0`、`reacq=20`）に対し、`enter=21` と `ew=21` が一致している。**入場のたびに epoch は書かれている。** したがって「epoch を更新しない入場経路」が主因ではない。

### 2.4 確定した原因

**publish より前に開始された `ScopedEpochReader` の区画が、数秒間保持されたままになっている。** その区画の epoch は 1 で固定され、`minReaderEpoch` を 1 に留める。A に退役した 20 件（epoch=1）は、区画が終わって reader が epoch 2 で入り直すまで reclaim できない。

これは「長時間 reader 区画」であり、前回の調査で「区画の滞留 0.1 ms」と見えたのは、サンプルが別スロット（健全なドメイン）を捉えていたためである。今回、ドメインとスロットを同時に出すことで区別できた。

## §3 区画を保持している主体

`tag=ScopedEpochReader` から、`convo::ScopedEpochReader` を使う EQ 系の読み手であることが分かる。EQ ドメインの `ScopedEpochReader` を取る箇所は次の 2 系統である。

| 箇所 | 内容 | 区画の長さ |
|---|---|---|
| `EQProcessor` 内部（`EQProcessor.Core.cpp:301,555,707,825,837,849` など） | 各 API の短い読み取り | 短い |
| 外部（`AudioEngine.EQResponse.cpp:92`、`AudioEngine.RebuildDispatch.cpp:706`） | 呼出し元が guard を保持したまま処理する | **長くなり得る** |

今回の 4.4 秒の区画は後者に相当する。どちらの呼出しかまでは、タグがクラス単位のため特定できていない。**次の計測では guard の生成箇所ごとにタグを分ける**（例: `"EQResponse"`, `"RebuildDispatch"`）ことで確定できる。

## §4 判定への影響

- `retire_delay` の最大値は「最も長い reader 区画」に等しい。今回の 4.4 秒はその実測である。
- 実行待ちは 1 ms 未満であり、reclaim 処理・FIFO・キュー容量の問題ではない。
- したがって修正候補は「長い区画を短くする」ことが第一であり、閾値・FIFO・publish 頻度の変更は、区画の短縮で足りない場合に検討する順序になる。

## §5 計測の制約の遵守と復元

| 制約 | 遵守 |
|---|---|
| EBR の安全性・所有権・reclaim 判定を変更しない | 遵守（カウンタの追加のみ） |
| `publishEpoch` の呼出し頻度を変更しない | 遵守（計数の追加のみ） |
| reader の入退場動作を変更しない | 遵守（`enterReader` に計数のみ） |
| RT 経路に動的確保・mutex・ファイル出力・通常ログを追加しない | 遵守（atomic 加算のみ。ダンプは NonRT の `tryReclaim` から 1 秒 1 回） |
| カウンタは固定容量・低オーバーヘッド | 遵守（`std::atomic` の配列、上限 64 要素） |
| 計測前後のソースハッシュと復元後のハッシュを保存 | 遵守（`tmp/rollback/c2impl_before/manifest.txt`、復元 5 ファイルの SHA-256 一致を確認） |

計測対象は `EpochDomain.h`、`IReaderEpochProvider.h`、`RCUReader.h`、`ScopedEpochReader.h`、`ISRRetireRouter.cpp` の 5 ファイルである。復元後、計測コード（`publishCount_`、`slotEpochWrites_`、`registerReaderThreadWithTag`、`dumpMeasurementIfNeeded`）がソースに残っていないことを確認した。

## §6 次の提案（未承認）

1. guard の生成箇所ごとにタグを分け、4.4 秒の区画の持ち主を確定する（1 行 × 2 か所）。
2. 特定した箇所の区画を短縮する（guard のスコープを狭める）。EBR の安全性は変わらない。
3. 区画の短縮で 50 ms を満たせない場合に限り、閾値・FIFO・publish 頻度の扱いを検討する。
