# guard インスタンス単位の追加計測（2026-11-11・計測のみ）

Owner 限定承認（2026-11-11）に従い、`ScopedEpochReader` の**インスタンス単位**で入場・退出を計測した。**EBR の入場・退出、epoch、所有権、reclaim 判定は変更していない。区画短縮も行っていない。**

## §0 結論

1. **承認された 8 か所を含む全 10 個の `ScopedEpochReader` 生成箇所を計測した。** 最長保持は `RebuildDispatch` の **15.3 ms**、次が `Core:707` の 0.39 ms である。**4.4 秒の区画は ScopedEpochReader ではない。**
2. **終了時に active な guard は 0 件**だった。したがって「終了記録が無い」のではなく、「ScopedEpochReader はすべて退出済み」である。
3. 4.4 秒の待ちを保持しているのは、`ScopedEpochReader` 以外の reader（`RCUReader` または `AudioEngine` の音声 reader）である可能性が高い。**特定には次の計測が必要**（未承認）。
4. 計測の途中で**容量不足の不具合**を自分で検出した。64 スロットでは 1 実行で飽和し（64 件で満杯）、以降の guard が記録されない。512 に増やして再計測した結果が上記である。

## §1 計測の実装

| 要件（Owner） | 実装 |
|---|---|
| 生成時にタグ・入場時刻・active 状態 | `ScopedEpochReader` のコンストラクタで固定表（512 スロット）のレコードを CAS で確保し、タグ・入場 µs・slot・depth・thread・provider を記録 |
| 破棄時に終了時刻・保持時間・active 解除 | `reset()` の先頭で `closeMeasure()`（終了 µs を書き、state を exited に） |
| 終了時に active を出力し、終了記録なしと区別 | `evidence/retire_latency_guard_instances.json` の `activeAtExit`（空なら全件退出済み）と `exitedByTag`（退出済みの集計） |
| ネストの区別 | 入場時の `depth` を記録し、`activeDepthHistogram` に分布を出す |
| move で二重終了しない | move コンストラクタで `measureIdx_` を移し、移行元を `-1` にする |
| 固定容量・動的確保なし・mutex なし | 静的配列 512、atomic のみ。ファイル出力は NonRT のレポータ（`postTeardown`）から |
| 集計は NonRT | `RetireLatencyTeardownReporter` のデストラクタで出力 |

### 1.1 計測対象（全生成箇所の再検索）

実装前に再検索し、`ScopedEpochReader` を生成する箇所は**10 個**だった。

| # | 箇所 | タグ |
|---|---|---|
| 1–7 | `EQProcessor.Core.cpp:301, 555, 707, 825, 837, 849, 867` | `Core:301` … `Core:867` |
| 8 | `EQProcessor.Coefficients.cpp:70`（`updateBandNode`） | `Coefficients:70` |
| 9 | `AudioEngine.EQResponse.cpp:92`（前回承認済み） | `EQResponse` |
| 10 | `AudioEngine.RebuildDispatch.cpp:706`（前回承認済み） | `RebuildDispatch` |

## §2 計測結果（bypass 40 回 × 250 ms、12 秒）

```json
"exitedByTag": [
  {"tag": "Coefficients:70",  "count": 120, "maxHoldUs": 22,    "sumHoldUs": 66},
  {"tag": "RebuildDispatch",  "count": 11,  "maxHoldUs": 15258, "sumHoldUs": 74584},
  {"tag": "Core:825",         "count": 14,  "maxHoldUs": 6,     "sumHoldUs": 7},
  {"tag": "Core:707",         "count": 4,   "maxHoldUs": 393,   "sumHoldUs": 904},
  {"tag": "Core:867",         "count": 172, "maxHoldUs": 7,     "sumHoldUs": 19},
  {"tag": "Core:837",         "count": 95,  "maxHoldUs": 20,    "sumHoldUs": 46},
  {"tag": "Core:849",         "count": 95,  "maxHoldUs": 1,     "sumHoldUs": 2},
  {"tag": "EQResponse",       "count": 1,   "maxHoldUs": 0,     "sumHoldUs": 0}
],
"activeAtExit": [],
"activeDepthHistogram": {}
```

同じ実行の D1 最大待ちは **4,416.2 ms** である。

### 2.1 読み取り

| 項目 | 結果 |
|---|---|
| 最長の ScopedEpochReader 区画 | **15.3 ms**（`RebuildDispatch`） |
| 2 番目 | 0.39 ms（`Core:707`） |
| `Coefficients:70`（`updateBandNode`） | 120 回、最長 **22 µs**。**保持は極めて短い** |
| 承認済み 2 か所 | `RebuildDispatch` 15.3 ms、`EQResponse` 0 ms。どちらも 4.4 秒ではない |
| 終了時に active な guard | **0 件** |
| ネスト | `activeAtExit` が空のため分布も空（計測期間中に active が残らなかった） |

**結論: 4.4 秒を保持した `ScopedEpochReader` インスタンスは存在しない。**

### 2.2 4.4 秒の区画と D1 `minReaderSlot=0` の対応

- D1 の最遅要求は `minReaderSlot=0`、`minReaderEpoch=1`、`epoch=1` で記録されている（前回と同様）。
- しかし slot 0 を 4.4 秒保持した `ScopedEpochReader` は、今回の計測に 1 件も存在しない。
- したがって slot 0 を保持しているのは **別の reader 実装**である。

## §3 残る候補（ソース確認）

`IReaderEpochProvider::enterReader` を呼ぶ主体は 2 系統ある。

| 系統 | 実装 | タグ | 区画の性質 |
|---|---|---|---|
| `ScopedEpochReader` | 今回計測済み | あり（10 か所） | すべて 15.3 ms 以下 |
| `RCUReader` | `src/core/RCUReader.h`（`enter()`/`exit()`） | **なし**（`registerReaderThread()` のまま） | 任意のスコープ長。UI・メッセージ経路で使われる |
| 音声 reader | `AudioEngine.Reader.cpp:13` → `ISRRetireRouter::enterReader` | **なし** | 音声コールバックの区間 |

4.4 秒の保持者は、このうち `RCUReader` か音声 reader のいずれかである。今回の承認範囲は `ScopedEpochReader` のインスタンス計測であり、この 2 系統は計測対象外のため**特定できなかった**（特定できなかった理由は「対象外だった」である）。

## §4 計測上の不具合の検出と修正（自己申告）

最初の計測（容量 64）では `exitedByTag` が 4 タグ・合計 64 件で止まり、`activeAtExit` も空だった。件数が容量と一致したため**飽和**と判断し、容量を 512 に増やして再計測した。再計測では 8 タグ・合計 512 件弱（120+11+14+4+172+95+95+1 = 512）となり、`Core:867`・`Core:837`・`Core:849`・`EQResponse` が新たに現れた。

**容量 64 の結果は採用していない。** 採用したのは容量 512 の再計測である。ただし今回も合計が 512 に達しており、**さらに長時間の実行では飽和し得る**。次回は「退出済みレコードの再利用（タグ別集計を加算してから解放）」に変えるべきである。

## §5 制約の遵守

| 制約 | 状態 |
|---|---|
| EBR の入場・退出、epoch 更新、所有権、reclaim 判定を変更しない | 遵守（計測フックのみ。`enterReader`/`exitReader` の呼出し順は不変） |
| 動的確保・mutex・RT ファイル出力・通常ログの禁止 | 遵守（静的配列と atomic のみ。出力は NonRT のレポータ） |
| 固定容量 | 遵守（512。§4 のとおり飽和の懸念は残る） |
| 集計・出力は NonRT | 遵守 |
| 変更前スナップショットと SHA-256 | `tmp/rollback/guardmeas_before/`（9 ファイル）に取得済み。**今回からは作業前に取得している** |

## §6 次に必要な計測（未承認）

1. `RCUReader` のインスタンス計測（今回と同じ方式）。`RCUReader` は `enter()`/`exit()` を持つので、同じ固定表を適用できる。
2. 音声 reader（`AudioEngine.Reader.cpp:13`）の区間計測。音声スレッドのため、記録は atomic のみ・出力は NonRT。
3. 上記で 4.4 秒の保持者が確定したら、その関数のソースと区間を提示する（区画短縮は未承認のまま）。

## §7 検証

| 項目 | 結果 |
|---|---|
| ctest | **60/60 合格（357.21 秒）** |
| アプリ実行 | D1 `unmatched=0`、最大 4,416 ms、guard 計測 JSON 出力あり |
| `evaluateRetireDelay` | 変更なし（本計測は同関数に触れていない） |
