# C2 スナップショット退役リーク: 改訂設計（実装は未承認・実装しない）

## §0 改訂の要点

前案（固定長の再試行リング＋write/read index）は不採用となった。理由は 2 点で、いずれも前案の構造に起因する。

1. 複数 producer のスロット予約が未証明だった。
2. リング満杯時に所有権を保持する主体が無く、リークを残したまま「所有権を保持する」という目的を満たしていなかった。

改訂案は**リングを持たない**。所有権は、移送が確定するまで**現在の保持者から動かさない**。この 1 つの原則で、指摘された 6 点はすべて構造的に解消する。

## §1 前提の再確認（コードで確定）

| 項目 | 事実 | 根拠 |
|---|---|---|
| C2 の本番呼出元 | **存在しない** | `SnapshotCoordinator.h:194-199`（「no production caller (updateFade was removed 2026-07-28)」） |
| C2 の RT 呼出し | 現状は無い。ヘッダ注記（`SnapshotCoordinator.h:150`）は削除済みの `updateFade` を前提にした記述 | 同上 |
| C2 が失敗する条件 | `IEpochProvider::enqueueRetire` が false（D キュー満杯） | `SnapshotCoordinator.cpp:123` |
| Q の受領 | 512 件まで受領。満杯時は false を返し、**deleter を実行しない** | `RetireQuarantineStore.h:67,80-87` |
| E の受領 | Q 満杯時のみ到達。Q と同じ契約 | `SnapshotCoordinator.cpp:41-43` |
| T の受領 | **growable（`std::vector`）で常に受領**。失敗経路が無い。呼出し元は Non-RT 限定と明記 | `ISRRetireRouter.h:50-55` |
| T の同期破棄 | epoch safe かつ Non-RT なら即時 deleter 実行 | `ISRRetireRouter.cpp:757-771` |

この表から、**Non-RT 文脈では Q→E→T の連鎖が所有権の終端を保証する**。つまり C2 の失敗時に必要なのは「移送先を用意すること」ではなく、**移送を Non-RT で行うこと**である。

## §2 改訂案（2 経路に分ける）

### 2.1 Non-RT から呼ばれた場合

```
target を exchangeTarget(nullptr) で取り出す（所有権はこの時点でローカル）
  └ enqueueRetire 成功 → D キューが所有（従来どおり）
  └ 失敗 → quarantineRetireSink(target, …) へ渡す
        Q 受領 → Q が所有
        Q 満杯 → E 受領 → E が所有
        E 満杯 → T 受領（常に成功） → T が所有
```

失敗時にカウンタを増やすだけの経路は廃止する。**所有権は必ずどこかへ移る。**

### 2.2 RT から呼ばれた場合

RT では**所有権を動かさない**。

```
C2（RT）
  ├ m_resetRequested.store(true, release)     … 1 bit。CAS は不要（冪等なフラグ）
  ├ m_fade.resetToIdle()                       … RT 安全（既存実装）
  └ return                                     … target はスロットに残したまま

Non-RT の drain 機会（startFade / completeFade / Timer / finalizeShutdown）
  ├ m_resetRequested.exchange(false) が true のときだけ 2.1 を実行
  └ 実行しない側は何もしない（CAS により 1 者だけが実行）
```

重要: **RT 経路は target を exchange しない**。したがって所有権は `SnapshotSlotStore` の target スロットに残り続ける。関数を抜けてもポインタを保持する主体が存在する。これは「RT から安全に実行できる overflow 処理」の答えでもある — overflow を作らない。

### 2.3 フラグと `startFade` の相互作用

フラグが立ったまま `startFade` が呼ばれると、`exchangeTarget(newTarget)` が旧 target を返す。この場合、旧 target は C1 の経路で退役するため、フラグは「同じ対象をもう一度退役させる」ことになり得る。

これを避けるため、`startFade` の先頭で `m_resetRequested.exchange(false)` を消費し、旧 target の退役を C1 に一本化する。フラグが残っていても対象は 1 つ（target スロット）しかないため、二重退役は起こらない。

| 時点 | target スロット | フラグ | 次の drain の動作 |
|---|---|---|---|
| RT の C2 直後 | 旧 target が残る | true | 旧 target を exchange して移送 |
| `startFade` が先に来た | 新 target に置換（旧は C1 が退役） | false に消費 | 何もしない |
| `finalizeShutdown` が先に来た | null になる（`retireCurrentAndTarget`） | 意味を失う | 何もしない（対象が無い） |

### 2.4 契約の変更点

- C2 を RT から呼んだ場合、「target の退役は次の Non-RT drain まで延期される」ことを契約に明記する。フェード状態（`m_fade`）のリセットは即時である。
- 現行 v2 の「失敗時に取り消す」処理は廃止する。移送するため完了が必ず来るので、取り消すと `unmatched` を生む。
- 記録は「移送を実行する場所」に 1 回だけ置く。RT 経路では記録しない（移送していないため）。

### 2.5 代替案（RT 対応を捨てる場合）

C2 の本番呼出元が無く、RT 呼出しを前提とした `updateFade` も削除済みである。RT 対応を不要と判断できるなら、次の方が小さい。

1. C2 を Non-RT 専用と明記し、`jassert(!convo::numeric_policy::isAudioThread())` を置く。
2. 失敗時は `quarantineRetireSink` へ移送する（2.1 と同じ）。
3. フラグもスロット保持も不要になる。

ただし「RT から呼ばれ得る」という既存注記を契約変更で消すことになる。将来 RT から呼ぶ予定があるなら 2.1／2.2 の案を採るべきである。

## §3 指摘 6 点への回答

| # | 指摘 | 回答 |
|---|---|---|
| 1 | リング満杯時に誰が所有権を保持するか | **リングを持たないため満杯が存在しない。** Non-RT では T が常に受領する（growable・失敗経路なし）。RT では target スロットが保持し続ける（exchange しない）。どの時点でも所有者が一意に存在する |
| 2 | RT から安全に実行できる overflow 処理 | **overflow を発生させない。** RT では移送そのものを行わず、1 bit のフラグを立てるだけ（ロック・確保・待機なし、`resetToIdle` は既存の RT 安全操作）。移送は Non-RT の drain 機会が行う |
| 3 | 複数 producer のスロット予約の保証 | 共有スロットを持たないため予約が不要。Non-RT 側の移送は Q/E/T が各々 `std::mutex` 下で受領を判定する（受領は原子的）。RT 側は冪等な 1 bit フラグのみで、複数 producer が同時に立てても状態は同じ |
| 4 | shutdown 時の未移送要素の drain（誰が・どの順序で） | 未移送要素は**スロットに残った target** だけである。`releaseResources` → `SnapshotCoordinator::finalizeShutdown` → `retireCurrentAndTarget()` がスロットの current と target を順に退役させる（既存経路）。その後に `drainAll()` → `provider_->drainAll()`（D）→ `drainAllQuarantineStore()`（Q+E+T）の順で全強制解放される。新しい保持領域を追加しないため、新しい drain 順序も追加されない |
| 5 | drain と通常 enqueue の二重移送の防止 | 移送の起点は `exchangeTarget(nullptr)` の**戻り値 1 回だけ**である。exchange は原子的で、同じ target を 2 回返さない。RT 経路のフラグは `exchange(false)` の勝者だけが 2.1 を実行する。したがって同一オブジェクトが 2 つの保持者に同時に属することはない |
| 6 | Q/T への移送が失敗した場合の所有権 | Q が満杯なら E へ、E が満杯なら T へ進む。T は growable で失敗経路を持たない（`ISRRetireRouter.h:50-55`）。したがって終端で必ず受領される。**残存リスクは T の `std::vector` 確保失敗（`std::bad_alloc`）のみ**で、この場合は `noexcept` によりプロセスが終了する（ポインタを失うリークにはならない） |

## §4 記録（D1）との関係

現行 v2 の C2 記録は「格納を試みる場所で記録し、未格納なら取り消す」である。改訂案では移送が確定するため、この形は使わない。

| 項目 | 現行 v2 | 改訂案 |
|---|---|---|
| 記録する場所 | `resetFadeStateAndRetireTarget` 内（格納の直前） | 移送を実行する場所（Non-RT の drain） |
| 格納失敗時 | `cancelRequestRTById` で取り消す | 移送するので取り消さない（完了が必ず来る） |
| RT 経路 | `recordRequestRT` で記録する | 記録しない（まだ移送していない） |
| 取消し API | 使う | 使わない |

理由は単純である。取り消しは「完了が来ない」ことを前提にした処理であり、移送する設計では前提が変わる。取り消すと、移送先の完了が `unmatched` になる。

この変更により、C2 の記録は移送と同一の関数内に 1 回だけ置かれ、記録と所有権移転が同じ場所で対応する。

## §5 検証計画（実装承認後）

| # | 検証 | 方法 |
|---|---|---|
| 1 | Non-RT の失敗時に所有権が T へ移る | キュー容量を 0 にしたテスト用 provider で C2 を呼び、residency が 1 増え、リークが無いことを確認 |
| 2 | RT 経路で target がスロットに残る | RT スレッドから C2 を呼び、`isFading()==false` かつ target スロットが非 null のままであることを確認 |
| 3 | フラグの勝者 1 者だけが移送する | 2 スレッドから同時に drain を呼び、移送が 1 回だけ起きることを確認 |
| 4 | 二重移送なし | 移送後に再度 drain しても residency が増えないことを確認 |
| 5 | 記録の対応 | 記録 1 件・完了 1 件・unmatched 0 を確認 |
| 6 | 回帰 | 既存 60 テスト |

## §6 実装しない理由（今回）

- 所有権の契約変更であり、承認済みの「記録の追加・取消し」とは別種の変更である。
- C2 は本番呼出元が無く、現時点の実害は確認されていない。
- Owner の指示どおり、実装は承認を待つ。
