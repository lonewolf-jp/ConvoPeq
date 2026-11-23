# `retire_delay` の経路別原因分析（調査のみ・修正未承認）

C5 適用後も `retire_delay` が `latency_exceeds_threshold` で fail する。本稿は、その最大待ち時間を経路別・時間成分別に分解した調査結果である。**修正は行っていない。**

## §1 判定式の定義と意図（現行）

```cpp
if (normalCount < kNReq) return insufficient("insufficient_observations");   // kNReq = 59
if (normalMaxNs > kThresholdNs) return fail("latency_exceeds_threshold");   // 50 ms
```

| 項目 | 値 | 意味 |
|---|---|---|
| `kNReq` | 59 | 超過 0 件のとき、95% 信頼で超過率 5% 未満と言える最小標本数（`ceil(ln(0.05)/ln(0.95))`） |
| `kThresholdNs` | 50 ms | **単一要求の上限**（分位点ではない） |
| 判定対象の経路 | `sync` / `drain` / `queue` / `quarantine` | `drainAll` は除外（shutdown の強制解放） |

つまり「通常経路の退役は、最悪でも 50 ms 以内に完了する」という**最大値基準**である。1 件の外れ値で fail する設計であり、平均や分位点では救われない。

## §2 計測の内訳（EBR 待ちと実行待ち）

レコーダは要求時に `ts`、reclaim 直前の `recordSafe` で `safeTs`、完了時に `now` を取る。したがって次の 2 成分に分かれる。

| 成分 | 定義 | 意味 |
|---|---|---|
| EBR 待ち | `safeTs - ts` | 要求から「epoch 安全」と判定されるまで（queue 滞留を含む） |
| 実行待ち | `now - safeTs` | 判定後、実際に deleter を呼ぶまで |

実測（3 回、bypass 40 回 × 250 ms、12 秒）:

| 実行 | 最大 latency | そのうち EBR 待ち | 実行待ちの最大 |
|---|---|---|---|
| run1 | 4,408.2 ms | 4,408.2 ms（100%） | 0.777 ms |
| run2 | 4,415.6 ms | 4,415.6 ms（100%） | 0.740 ms |
| run3 | 4,388.0 ms | 4,388.0 ms（100%） | 0.501 ms |

**最大待ち時間はすべて EBR 待ちである。** 実行待ちは 1 ms 未満で、reclaim 処理そのものはボトルネックではない。

## §3 最大待ちを発生させた個々の request

3 回の実行で共通する特徴は次のとおり。

| 項目 | 値（run1 の例） |
|---|---|
| `requestId` | 3,515 / 3,517 / 3,519 / 3,521 …（同時刻に一括で作られた連番） |
| `path` | `queue`（D キュー経由の通常 reclaim） |
| `latencyNs` | 4,408 ms（全件ほぼ同一） |
| `ebrNs` | 4,408 ms（= latency） |
| `execNs` | 0.000 ms |
| 要求時の `epoch` | **1** |
| 要求時の `minReaderEpoch` | **1** |
| 要求時の `minReaderSlot` | **0** |
| 要求時刻 / 完了時刻 | t=30,352,475 ms / t=30,356,883 ms（差 4.4 s） |
| 完了経路 | `DeferredDeletionQueue::reclaim`（`Path::Queue`） |

すなわち「epoch 1 で退役した一群が、`minReaderEpoch` が 2 以上になるまで D キューに滞留した」ケースである。60〜800 ms の遅延も同じ署名（epoch=1、minE=1、slot=0、path=queue）で、**同一の原因の大小**である。

## §4 待ち時間の 3 分解

```
要求 → [A] epoch 公開待ち → [B] reader 再入場待ち → [C] FIFO 先頭詰まり → 安全判定 → [D] deleter 実行
        └──────────── EBR 待ち（実測 100%） ────────────┘                      └ 実行待ち（実測 <1ms）
```

| 成分 | 内容 | 実測での寄与 |
|---|---|---|
| A: epoch 公開待ち | ドメインの `globalEpoch` が進まないと `minReaderEpoch` も進まない | 中（バースト中は 250 ms ごとに進む） |
| B: reader 再入場待ち | 入場中の reader が古い epoch を持つ間は `minReaderEpoch` が固定される | **主因** |
| C: FIFO 先頭詰まり | `reclaim` は先頭が不可なら即 `break`（`DeferredDeletionQueue.h` の `else break`）。後続が安全でも待つ | 副因（後続の 60〜200 ms 群） |
| D: deleter 実行 | — | 無視できる（<1 ms） |

## §5 reader スロットの状態

停滞中の `[R]` サンプル（400 ms 間隔、一時プローブ）:

```
[R] t=30341883 dom=...BE40 minReader=1 global=2 pending=20 [0 e=1 d=1 tag=unnamed th=2 res=0.4ms]
[R] t=30342439 dom=...0D40 minReader=1 global=2 pending=20 [0 e=1 d=1 tag=unnamed th=2 res=0.1ms]
```

| 項目 | 観測 |
|---|---|
| 最小 epoch のスロット | **slot 0** が常に最小 |
| depth | 1（区画保持中） |
| 区画の滞留時間 | サンプル時点で 0.1〜0.5 ms（短い区画を繰り返している） |
| `ownerTag` | **"unnamed"**（＝タグ未設定） |
| `ownerThreadId` | 1 または 2（登録順の単調 ID） |
| ドメイン | 複数の `EpochDomain` インスタンスが併存（コンポーネントごとに独立） |

### 5.1 重要な発見: 本番の reader 登録はすべて無名

```
src/core/RCUReader.h:183              epochProvider->registerReaderThread()          // タグなし
src/core/ScopedEpochReader.h:147      provider.registerReaderThread()                // タグなし
src/audioengine/ISRRetireRouter.cpp:371  provider_->registerReaderThread()           // タグなし
```

`registerReaderThread(const char* tag)` はタグを受け取れるが、**本番の呼出しは 3 か所ともタグを渡していない**。したがって `ownerTag` は常に "unnamed" になり、ログから「どのコンポーネントが epoch を保持しているか」を特定できない。今回はこれが原因で、ブロッキング reader の帰属を**特定できなかった**。

### 5.2 「長時間 reader」の有無

区画の滞留時間はサンプル時点で 1 ms 未満であり、**1 つの区画を数秒保持する reader は観測されなかった**（100 ms 超の区画を記録するプローブでも 0 件）。一方で、停滞したドメインでは次の状態が 8.5 秒間続いた。

```
[R] t=30343972 dom=...D880 minReader=1 global=2 pending=20 [0 e=1 d=1 tag=unnamed th=2 res=0.1ms]
[R] t=30344476 dom=...D880 minReader=1 global=2 pending=20 [0 e=1 d=1 tag=unnamed th=2 res=0.1ms]
...
[R] t=30352476 dom=...D880 minReader=1 global=2 pending=20 [0 e=1 d=1 tag=unnamed th=2 res=0.1ms]
```

| 観測 | 値 |
|---|---|
| ドメインの `globalEpoch` | **2 のまま少なくとも 8.5 秒間変化なし** |
| `minReaderEpoch` | 1 のまま |
| 滞留件数 | 20 件で一定 |
| reader | slot 0、depth 1、区画の滞留 0.1〜0.2 ms（短い区画を繰り返す） |
| 解放 | 退役（t=30,352,476）の 4.4 秒後に 20 件が一括で解放（`minReader` が 2 に進んだ瞬間） |

この 2 つ（短い区画の繰り返しと、epoch が進まないこと）が両立する説明は **「そのドメインが epoch を公開していない」** である。区画は繰り返し入退場しているが、入場時に設定される epoch は `currentEpoch()`＝2 であり、`minReaderEpoch` が 1 のままなのは、**slot 0 の区画が epoch 2 に更新されていない**ことを意味する。

したがって、原因は「長時間の reader 区画」ではなく、次のいずれかである。

| 仮説 | 内容 | 判定に必要な計測 |
|---|---|---|
| H-1 | 退役先のドメインが epoch を公開しないため、`minReaderEpoch` が進まない | ドメインごとの `publishEpoch` 回数と最終時刻 |
| H-2 | reader の入場経路の一部（`reserveReaderThread` / `reacquireReaderSlot` 経由）が slot の epoch を更新しない | 経路別の入場回数と epoch 更新回数 |

H-1 は「epoch を公開しないドメインへ退役すると、shutdown の `drainAll` まで解放されない」という設計上の穴を示す。H-2 は reader 側の不具合を示す。**どちらかを確定するには §9 の追加計測が必要**であり、今回は確定できていない。

### 5.3 帰属を特定できなかった理由

本番の reader 登録は 3 か所すべてタグなし（§5.1）であり、`ownerTag` は常に "unnamed" になる。`ownerThreadId` は 1 または 2 だが、これをコンポーネント名に写像する情報が無い。したがって「どのコンポーネントの reader か」はログから特定できない。

## §6 4.4〜13.9 秒のばらつき

| 実行 | 最大 | 実行時間 | 備考 |
|---|---|---|---|
| bypass 40 × 250 ms | 4.4 s | 16 s | バースト終了付近で解放 |
| bypass 60 × 2 ms | 11.4 s | 16 s | バーストが早く終わり、待ちが長い |
| EQ 変更 | 13.9 s | 16 s | バーストなし。早期に退役した分が最後まで待つ |
| 追加操作なし | 7.5 s | 16 s | 同上 |

**ばらつきは実行条件（バーストの長さ・シナリオ）に依存する。** 待ち時間は「退役した時刻から、`minReaderEpoch` が進む時刻まで」であり、シナリオが変わると両者の相対位置が変わる。負荷やシャットダウン処理への依存は、今回の範囲では確認できなかった（`drainAll` の完了は `Path::DrainAll` で別集計され、最大値の対象外である）。

## §7 閾値の意味と fail の解釈

- 50 ms は「退役要求から解放までの上限」であり、**EBR の設計上の待ち（reader 区画の長さ）を直接反映する**。
- 現状の fail は「退役パイプラインが遅い」ことを意味しない。実測どおり実行待ちは 1 ms 未満である。
- したがって、閾値を満たすには次のいずれかが必要になる。
  1. reader 区画を短くする（数百 µs 以下に保つ）。
  2. epoch の公開頻度を上げる（`publishEpoch` を退役のたびに呼ぶ等）。
  3. FIFO 先頭詰まりを解消する（先頭が不可でも後続の安全な entry を解放する）。
  4. 閾値の定義を「実行待ち」と「EBR 待ち」に分ける（判定式の変更になるため未承認）。

いずれも判定式・キュー・reader の挙動に関わるため、**実装は Owner 判断を待つ**。

## §8 混同してはいけない点

| 用語 | 定義 | 今回の実測 |
|---|---|---|
| EBR 待ち | 要求 → epoch 安全判定 | 最大 4.4〜13.9 s（全体のほぼ 100%） |
| 実行待ち | 判定 → deleter 呼出し | 最大 0.8 ms |
| queue 滞留 | D キューに入ってから出るまで | EBR 待ちと同一区間（この実装では同じ） |
| reclaim 実行待ち | `tryReclaim` の呼出し間隔 | 400 ms 周期の観測から、待ちの主因ではない |

「queue 滞留」と「EBR 待ち」は本実装では同じ区間を指すため、分けて語る場合は「公開待ち」「reader 再入場待ち」「FIFO 詰まり」に分解する必要がある（§4）。

## §9 次の調査提案（未承認）

1. **reader 登録にタグを付ける**（`RCUReader`／`ScopedEpochReader`／`ISRRetireRouter` の 3 か所、1 行ずつ）。これで `ownerTag` から帰属を特定できる。
2. **要求ごとの分解ログを恒久化する**（現在は一時プローブ）。`path`・`epoch`・`minReaderEpoch`・`minReaderSlot`・`ebr`・`exec` を遅い順に 8 件だけ JSON に残す。観測リング（先頭 256 件）では最遅の要求が残らない。
3. **ドメインごとの `globalEpoch` 前進回数と間隔**を記録する。今回の観測では複数ドメインが併存し、どれが停滞したかの切り分けができていない。
4. 上記 1〜3 の結果で「公開待ち」と「reader 再入場待ち」の寄与を確定してから、閾値・FIFO・公開頻度の扱いを判断する。
