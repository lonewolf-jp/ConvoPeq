# RCUReader・音声 reader のインスタンス計測（2026-11-11・計測のみ）

Owner 限定承認（2026-11-11）に従い、`RCUReader` と音声 reader を `ScopedEpochReader` と**同一実行**で計測した。**EBR の入退場・epoch・所有権・reclaim 判定は変更していない。区画短縮も行っていない。**

## §0 結論

1. **4.4 秒を保持した reader は存在しない。** 3 系統の最長保持は `RCUReader` 39.4 ms、音声 reader 14.0 ms、`ScopedEpochReader` 8.1 ms である。同時取得した D1 の最大待ちは 4,405 ms だった。
2. **終了時に active な reader は 0 件**（final 段・postTeardown 段とも）。記録失敗・容量不足・集計取りこぼし・入場失敗・underflow はすべて 0 件で、計測自体は飽和していない。
3. **原因は reader の保持ではなく、EQ ドメインの reclaim ドライバの不在である。** `EQProcessor` の専用ドメインは `prepareToPlay()` / `releaseResources()` / デストラクタでしか reclaim されない。通常運転中に退役した EQ 状態は、次のそれらの機会まで解放されない。
4. この説明は、以前の計測（停滞ドメインの `publishCount=1`、滞留 20 件、`minReaderSlot=0`）と整合する。

## §1 計測の実装

### 1.1 共通レジストリ（要件対応）

`src/core/ReaderMeasureRegistry.h`（新規・固定容量 128）。

| 要件 | 実装 |
|---|---|
| 退出済みレコードの再利用 | `close()` でタグ別集計（回数・最大・合計）へ加算した**後**にレコードを解放する |
| active を上書きしない | `claim()` は state=free のレコードのみ CAS で取得する |
| 記録失敗・容量不足・集計取りこぼしの明示 | `recordFailures` / `aggMisses` / `enterFailures` / `underflows` をカウンタで出力 |
| active 一覧を段ごとに取得 | `final`（`emitViolationReport` の先頭）と `postTeardown`（レポータ）で `activeAtStage` を出力 |
| 3 系統を同一実行で計測 | `ScopedEpochReader`（自前表）＋ `RCUReader`＋音声 reader（共通レジストリ）を同じプロセスで記録 |
| 固定容量・atomic のみ | レコード 128、タグ 32、すべて `std::atomic`。出力は NonRT |

### 1.2 RCUReader の計測点

`RCUReader::enter()` / `exit()` の**最外層のみ**を 1 区画として記録する。

- 入場: `nestingDepth` の 0→1 を観測し、slot 取得と `enterReader()` が成功した後に `claim("RCUReader", tid, 1, this, provider)`。
- 退出: `previousDepth == 1`（最外層）で `close()`。ネスト（`previousDepth > 1`）では閉じない。
- 異常系: slot 取得失敗は `enterFailures`、`previousDepth == 0` の退出は `underflows` に数え、正常退出と混同しない。
- インスタンス・provider・slot・thread・開始/終了時刻・保持時間を記録する。

### 1.3 音声 reader の計測点

`AudioEngine::enterRcuReader(int)` / `exitRcuReader(int)`（`AudioEngine.Reader.cpp`）。

- スレッドローカルの深さで最外層のみを 1 区画とする（ネストは深さで数え、二重計上しない）。
- 記録は atomic のみ。動的確保・mutex・ファイル出力・通常ログは行わない。
- `readerIndex`・thread・入退場時刻・保持時間を対応づける。

### 1.4 対象の再検索（漏れの確認）

| 系統 | 生成・入退場箇所 | 計測 |
|---|---|---|
| `ScopedEpochReader` | 10 か所（前回計測。`EQProcessor.Core.cpp` 7、`EQProcessor.Coefficients.cpp:70`、`AudioEngine.EQResponse.cpp:92`、`AudioEngine.RebuildDispatch.cpp:706`） | 済 |
| `RCUReader` | `src/core/RCUReader.h` の `enter()`/`exit()`。生成は各利用側（`RuntimeReaderContext` 経由） | 済（クラス内で一括） |
| 音声 reader | `AudioEngine.Reader.cpp` の `enterRcuReader`/`exitRcuReader` → `ISRRetireRouter::enterReader/exitReader` | 済 |

`IReaderEpochProvider::enterReader` を直接呼ぶ他の経路は、再検索で見つからなかった（`ISRRetireRouter` は転送のみ）。

## §2 計測結果（bypass 40 回 × 250 ms、実測 wall 16.5 秒、exit-ms 12000）

### 2.1 3 系統の同時計測

| 系統 | 区画数 | 最長保持 | 合計保持 | active（final / postTeardown） |
|---|---|---|---|---|
| `RCUReader` | 3,835 | **39.4 ms** | 2,383 ms | 0 / 0 |
| 音声 reader（`AudioReader`） | 91 | **14.0 ms** | 562 ms | 0 / 0 |
| `ScopedEpochReader` | 512 件（うち最大は `RebuildDispatch`） | **8.1 ms** | — | 0 / 0 |

レジストリのカウンタ（final と postTeardown で同一）:

```json
{"closed": 3926, "recordFailures": 0, "aggMisses": 0, "enterFailures": 0, "underflows": 0}
```

### 2.2 D1 との対応

| 項目 | 値 |
|---|---|
| D1 最大待ち | **4,405.2 ms** |
| 最長の reader 保持 | 39.4 ms（`RCUReader`） |
| 差 | 約 4.37 秒 |

**どの reader も 4.4 秒を保持していない。** したがって「4.4 秒の区画を保持した guard インスタンス」は存在せず、特定できないのではなく**存在しない**。

### 2.3 `minReaderSlot=0` / `minReaderEpoch=1` との照合

- 最遅要求は `minReaderSlot=0`、`minReaderEpoch=1`、`epoch=1` である（従来と同じ）。
- slot 0 の保持は最長でも 39.4 ms であり、4.4 秒の保持はない。
- `minReaderEpoch` が 1 のままだったのは、**そのドメインの `globalEpoch` が 1 のままだった**ためである（`publishCount=1`）。reader が 1 を保持していたのではない。

## §3 原因（確定）

`EQProcessor` の専用 `EpochDomain` には、**通常運転中の reclaim ドライバが無い**。

| 呼出し | 箇所 | 頻度 |
|---|---|---|
| `flushPendingEpochAdvance()` → `m_ownedRetireRouter.tryReclaim()` | `EQProcessor::releaseResources()`（`Core.cpp:231`） | 停止時のみ |
| 同上 | `EQProcessor::prepareToPlay()`（`Core.cpp:790`） | 再構成時のみ |
| `m_ownedRetireRouter.tryReclaim()` / `m_epochDomain->tryReclaim()` | `~EQProcessor`（`Core.cpp:181,185,187`） | 破棄時のみ |

したがって、通常運転中に EQ ドメインへ退役したオブジェクトは、**次の `prepareToPlay` / `releaseResources` / 破棄まで D キューに留まる**。待ち時間は「退役からその機会まで」であり、シナリオと実行時間で 4.4 秒／11.4 秒／13.9 秒と変わる。

これは「EBR 待ち」という分類の内訳が誤っていたことを意味する。正確には **reclaim ドライバ待ち**である。

## §4 制約の遵守

| 制約 | 状態 |
|---|---|
| EBR の入退場・epoch・所有権・reclaim 判定を変更しない | 遵守（フックのみ。`enterReader`/`exitReader` の呼出し順は不変） |
| RT 側は atomic のみ | 遵守（音声 reader は thread_local と atomic のみ） |
| 動的確保・mutex・RT ファイル出力・通常ログ | なし |
| 集計・出力は NonRT | 遵守 |
| レコード再利用・飽和検出 | 遵守（`recordFailures`/`aggMisses` が 0、closed 3,926 で再利用が機能） |
| 変更前スナップショットと SHA-256 | `tmp/rollback/readermeas_before/`（7 ファイル）に作業前に取得 |

## §5 検証

| 項目 | 結果 |
|---|---|
| ctest | **60/60 合格（335.64 秒）** |
| アプリ実行 | D1 `unmatched=0`、最大 4,405 ms、3 系統の計測 JSON 出力あり |
| レジストリの整合 | `closed=3926`、`recordFailures=0`、`aggMisses=0`、`active=0` |
| `evaluateRetireDelay` | 変更なし（本計測は同関数に触れていない） |

## §6 次の提案（未承認）

1. **reclaim ドライバの追加**（例: EQ ドメインの `tryReclaim` を既存のタイマー周期で呼ぶ）。これが `retire_delay` の最大待ちを直接短縮する。EBR の安全性は変わらない（epoch-gated のまま）。
2. 併せて、`retire_delay` の分類を「reclaim ドライバ待ち / EBR 待ち / 実行待ち」に分けることを提案する（判定式の変更になるため未承認）。
3. 区画短縮は不要（保持は最大 39.4 ms で、閾値 50 ms 未満）。
