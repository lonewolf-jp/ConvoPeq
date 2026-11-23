# A（D1 測定）の実装と CI v5 の 2 検査（2026-11-10）

## §0 今回の範囲

| 項目 | 状態 |
|---|---|
| A: 完了イベントの実装（要求 ID・完了時刻・epoch・経路） | 実装した |
| A: 測定器と集計器（全体・経路別・N_valid・insufficient） | 実装した |
| CI: v5 への 2 検査（insufficient の拒否） | 実装した（pwsh 7 で検証） |
| B: 修正案 A の恒久実装 | 未実装（最終レビュー待ち） |
| レポートの v2 化・CI 駆動スクリプトの変更 | 未実施（承認外） |
| `AGENTS.md` の更新 | 未実施（承認外） |
| commit / merge / release | 未実施 |

変更したファイルは次の 9 件である。

| ファイル | 内容 |
|---|---|
| `src/audioengine/ISRRetireLatency.h`（新規） | 退役遅延レコーダ（固定容量・アトミックのみ） |
| `src/audioengine/ISRRetireRouter.cpp` | 要求点 1 箇所、完了点 5 箇所（同期・drain・drainAll・retryable 2）、終了処理マーカー |
| `src/DeferredDeletionQueue.h` | 完了点 2 箇所（D キュー・drainAllUnsafe） |
| `src/audioengine/RetireQuarantineStore.h` | 完了点 2 箇所（隔離 drain・drainAllUnsafe） |
| `src/core/DeletionQueue.cpp` | 完了点 2 箇所（容量超過の即時解放・reclaim） |
| `src/audioengine/ISRHB.h` / `.cpp` | シナリオ結果の拡張、D1 判定、レポート出力 |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | エンジン生成時のレコーダ初期化 |
| `.github/scripts/isr-verify-v5.ps1` | insufficient を拒否する 2 検査 |

## §1 実装内容

### 1.1 レコーダ（`RetireLatencyRecorder`）

Owner の条件（一意 ID・0 不使用・ID 再使用なし・容量不足の記録・重複完了の非上書き・RT 安全）に対応した。

| 要件 | 実装 |
|---|---|
| 一意な ID | `fetch_add` で払い出し。並行発行でも一意 |
| 0 を払い出さない | 初期値 0 に +1 して使用。0 と `UINT64_MAX` は枯渇として拒否 |
| オーバーフロー時に再利用しない | ラップを検出したら払い出しを止め、`idExhausted` に計上 |
| 容量不足 | 512 スロット。空きが無ければ `overflow` に計上し、その要求は観測に含めない |
| 重複完了を上書きしない | 完了は CAS で 1 回だけ取得。2 回目は `unmatchedCompletions` に計上 |
| RT 制約 | 動的確保・ロックなし。アトミックと固定配列のみ（要求点は Non-RT 経路） |
| 対応付けのキー | ポインタ（オブジェクト同一性）。epoch は記録用 |
| 再試行の二重計上 | 同一ポインタの再 enqueue は既存の要求を再利用 |
| 終了処理 | `drainAll()` 以降の要求は `postShutdownRequests` として測定対象外 |
| 未完了 | `pending`（先頭 8 件を ID と epoch でレポートに出す） |

### 1.2 要求点と完了点

| 事象 | 位置 | 経路名 |
|---|---|---|
| 要求 | `ISRRetireRouter::enqueueRetire`（全 enqueue の漏斗） | — |
| 完了（同期） | `ISRRetireRouter::terminalReclaim` | `sync` |
| 完了（retryable 同期） | `terminalReclaimRetryable` | `sync` |
| 完了（drain） | `TerminalReclaimAuthority::drain` の 2 箇所 | `drain` |
| 完了（drainAll） | `TerminalReclaimAuthority::drainAll` と retryable | `drainAll` |
| 完了（D キュー） | `DeferredDeletionQueue::reclaim` | `queue` |
| 完了（D キュー全排水） | `DeferredDeletionQueue` の drainAll | `drainAll` |
| 完了（容量超過） | `DeletionQueue::enqueue` | `queueOverflow` |
| 完了（隔離） | `RetireQuarantineStore::drain` / drainAll | `quarantine` / `drainAll` |

### 1.3 判定

- `N_req = 59`（`ceil(ln 0.05 / ln 0.95)`、95% 信頼・超過率 5%・超過 0 件）
- `N_valid = completions`（要求と完了が 1 対 1 で対応した数）
- `pending = requests - completions`
- 判定順: `overflow`・`idExhausted` → `incomplete_requests`（pending > 0）→ `insufficient_observations`（N_valid < 59）→ `latency_exceeds_threshold`（最大 > 50 ms）→ pass
- 50 ms は変更していない

レポートには `scenarioResults` の各要素へ `reason`・`observations`・`required`・`maxLatencyNs` を追加し、トップレベルに `retireLatencyD1`（全体・経路別・生観測 256 件・pending 8 件）を追加した。スキーマ名は `hb_violation_report_v1` のままで、追加フィールドは既存検証器が無視する。

### 1.4 v5 の 2 検査

`reason == insufficient_observations` を拒否する検査と、`observations < required` を拒否する検査を追加した（Owner 承認の範囲に限定）。

## §2 検証

### 2.1 レコーダ単体（リポジトリ外ハーネス、22 項目）

`[PASS]` 22 件、`summary: failures=0`。確認した内容:

- requestId が 0 を払い出さない・一意（500 件）
- 完了 1 件だけが対応付く／経路別件数が一致
- 重複完了は completions を増やさず unmatched に計上
- 容量上限（512）で打ち切り、超過 5 件を overflow に計上
- 同一ポインタの再要求は 1 件・同じ ID
- 60 ms 遅延で overThreshold と maxNs が立つ
- 未完了は completions に入らない
- 並行 400 件（4 スレッド）で全件対応付き・unmatched 0
- reset で全カウンタが 0

### 2.2 v5（PowerShell 7）

| # | 入力 | 結果 |
|---|---|---|
| C1 | 既存 v1 pass | 受理 |
| C2 | fail + reason | 拒否 |
| C3 | pass + observations=0 | **拒否**（追加前は受理） |
| C4 | pass + reason=insufficient | **拒否**（追加前は受理） |
| C5 | result=insufficient | 拒否 |
| C6 | 未知フィールド pass | 受理 |

### 2.3 アプリ実行（Release、`--verify-runtime-evidence-live`）

| 実行 | 待機 | requests | completions | unmatched | overflow | N_valid | 判定 |
|---|---|---|---|---|---|---|---|
| 短（24 バースト×250 ms） | 20 秒 | 23 | 2 | 0 | 0 | 2 | fail（`incomplete_requests`） |
| 長（120 バースト×150 ms） | 40 秒 | 23 | 2 | 0 | 0 | 2 | fail（`incomplete_requests`） |

経路別は両実行とも `queue` の 2 件のみで、最大 6.74 ms（短）/ 8.36 ms（長）。50 ms 超は 0 件。

**補助実行（同じ実装の直前ビルド。D キューのフック追加後、終了処理マーカー導入前）** では観測数が多く、次の分布だった。

| 実行 | requests | completions | `drainAll` | `queue` | queue の 50 ms 超 | queue 最大 |
|---|---|---|---|---|---|---|
| 36 秒 | 386 | 346 | 168 件・最大 33 µs | 178 件 | **141 件** | **11.08 秒** |
| 59 秒 | 4,370 | 4,330 | 2,205 件・最大 38.7 µs | 2,125 件 | **264 件** | **10.78 秒** |

### 2.4 ctest

Release 構成で全ターゲットをビルドし、`build/` で `ctest -C Release --output-on-failure` を実行した。

```
100% tests passed out of 60
Total Test time (real) = 358.90 sec
ctest_exit=0
```

`AudioEngineHarness`（303.10 秒）、`RT` 系、`PersistentCacheV3Phase2Tests` を含め 60 件すべて合格した。レコーダ追加による回帰は検出されていない。

## §3 実装中に見つけて直した不具合

1. **レポート用ブロックを `emitTrace` に誤挿入**。最初の実装では、`hb_violation_report.json` ではなく `hb_graph_trace.json` に `retireLatencyD1` が書かれていた（トレースのサイズが増え、レポートにキーが無いことで発覚）。挿入先を `emitViolationReport` に移し、トレースは元の形に戻した。
2. **D キューの完了フック漏れ**。`src/core/DeletionQueue.cpp` だけをフックしていたが、実際に使われるのは `src/DeferredDeletionQueue.h` だった。このため最初の実行は completions=0 だった。両方に追加した。
3. **JSON 文字列リテラルへの改行混入**でビルドが失敗（C2001）。該当行を修正して再ビルドした。
4. **単体テストの不備**。同一変数のアドレスを 500 回記録してしまい「ID が一意でない」と誤検出した。テスト側を修正した（実装は正しかった）。

## §4 実装で判明した設計上の論点（重要）

1. **world の退役はルーターを通らない。** 承認時の設計は「`onRuntimeRetiredNonRt` の要求 → ルーターの解放完了」を想定していたが、実際の `runtimePublicationBridge_.retire(...)` は入力検証だけで、world は解放待ち行列に積まれない。ルーターを通るのは DSP ハンドルとライフタイムの退役である。したがって D1 は「ルーターを通る退役」の測定であり、world ポインタの解放は測っていない。
2. **`queue` 経路の遅延は EBR 待ちを含む。** 補助実行では 178 件中 141 件が 50 ms 超、最大 11.08 秒だった。これは D キューが「読者 epoch が進むまで解放しない」設計であるためで、パイプラインの遅延ではない。一方、終了時の強制解放（`drainAll`）は最大 33〜39 µs である。同じ 50 ms を両方に適用すると、`queue` 経路だけで必ず不合格になる。
3. **`pending` が残る。** 最終ビルドでは 23 件中 21 件が未完了である。レポートはシャットダウンの途中（最終破棄より前）に書かれるため、その時点で解放されていないオブジェクトが残る。このため `retire_delay` は常に `insufficient` になり、合格しない（安全側だが、判定としては使えない）。
4. **観測数の確保が不安定。** bypass バーストで 386 件観測できた実行と、23 件しか観測できない実行があった。同じ引数でも歩留まりが変わるため、`N_req = 59` を安定して満たす駆動方法は未確立である。

## §5 未確認事項と残課題

- `pending` の正体（最終破棄のタイミングとレポート出力順の関係）。
- `N_req = 59` を安定して満たす駆動（CI スクリプトの変更は今回の承認外）。
- `queue` 経路の EBR 待ちを遅延として扱うか、`sync`／`drainAll` と分けて判定するか（仕様判断）。
- 実機負荷時の分布、TSan／LSan、聴感確認。
- ctest の結果（実行中）。

## §6 承認をお願いする事項

1. **レポート出力順の変更**（最終 drain の後に `hb_violation_report.json` を書く）。現状は `pending` が残り、常に `insufficient` になる。測定コードの範囲で直せるが、出力順は挙動の変更にあたるため承認を求める。
2. **経路別の合格基準**。`sync`／`drain`／`drainAll` と `queue` を分け、`queue` は EBR 待ちとして別の閾値（または参考値）にするか。50 ms は変更しない前提で、判定に使う経路を決める必要がある。
3. **観測数を確保する駆動の追加**（CI スクリプトまたは CLI オプション）。`N_req = 59` を満たすには、1 実行で 59 件以上の退役を安定して起こす必要がある。
