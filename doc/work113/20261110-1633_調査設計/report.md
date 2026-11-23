# 次段階の調査・設計（queue 待ちの分解／RuntimeWorld 寿命／RT 経路／B 差分レビュー）

## §0 範囲

今回の承認は「調査・設計のみ」であり、ソース・テスト・閾値・設定の変更は行っていない。commit、merge、release、配布、公開告知もしない。既存の未コミット変更（前回までの D1 実装）は保持している。

作業は 4 つである。

| # | 作業 | 結果 |
|---|---|---|
| 1 | `queue` の 40 秒級待ちの原因分解 | 分解できた（支配要因は epoch 前進待ち）。追加計測の設計を提示 |
| 2 | `RuntimeWorld` 寿命の測定設計 | 設計のみ提示 |
| 3 | RT 経路の要求記録方式 | 設計のみ提示（方式比較つき） |
| 4 | B 修正案 A の最終レビュー資料 | 現行実装・差分・26 ケース・API 確認を提示（適用はしない） |

## §1 `queue` の 40 秒級待ちの分解

### 1.1 コードから確定した事実

- D キューの解放条件は `entry.epoch < minReaderEpoch`（`DeferredDeletionQueue::reclaim`）。
- `minReaderEpoch` は「現在 epoch と、参加中 reader が入場時に捕捉した epoch の最小値」（`EpochDomain::getMinReaderEpoch`）。
- 現在 epoch（`globalEpoch`）を進めるのは `publishEpoch()` だけである（`EpochDomain::publishEpoch`）。ブロックごとには進まない。
- したがって、epoch `E` で積まれたエントリは、**次の publish で epoch が `E+1` 以上になるまで解放されない**。
- drain の実行機会は 1 ms 周期のフォールバックとイベント駆動（`drainDeferredRetireQueues`、`waitForDrainSignalOrTimeout`）。実行機会は制約にならない。

### 1.2 実測（交互バースト 60 回・250 ms、待機 20 秒）

| 項目 | 値 |
|---|---|
| requests / completions / pending | 2,708 / 2,708 / 0 |
| `queue` | 1,385 件・50 ms 超 1,222 件・最大 17.47 秒 |
| `drainAll` | 最大 38.6 µs |
| 判定 | fail（`latency_exceeds_threshold`、観測 1,385 件） |
| shutdown の `oldestAgeMs` | 26,584 ms |
| `routerPendingRetire`（shutdown） | 2 |
| stuck reader の記録 | 0 件 |
| `reclaimAttemptCount` / `reclaimSuccessCount` | 690,693 / 124 |

### 1.3 epoch と遅延の関係（同じ実行の生観測 256 件）

| epoch | 件数 | 中央値 | 最大 |
|---|---|---|---|
| 1 | 120 | 301.3 ms | **17,468 ms** |
| 2 | 126 | **0.0 ms** | 0.0 ms |
| 118〜126 | 各 1 | 約 4.5 ms | 約 4.8 ms |
| 130 | 1 | 12,199 ms | 12,199 ms |

同じ実行の中で、epoch 1 のエントリだけが長く待ち、epoch 2 のエントリは即時に解放されている。drain の実行機会が足りないなら全 epoch が同じ周期で待つはずであり、この偏りは**エントリの epoch が安全になるまでの待ち**を示す。

### 1.4 分解（現時点の結論）

| 成分 | 現時点の評価 | 根拠 |
|---|---|---|
| epoch 前進待ち（EBR 安全到達） | **支配的** | epoch 1 の中央値 301 ms・最大 17.5 秒、epoch 2 は 0.0 ms。解放条件が `epoch < minReaderEpoch` で、epoch は publish でしか進まない |
| drain の実行機会不足 | 小さい | 1 ms フォールバックとイベント駆動。試行回数 690,693 回 |
| 解放処理そのもの（deleter） | 小さい | 完了は µs 台（`drainAll` 最大 38.6 µs） |
| 終了時の強制排水 | 別枠 | `drainAll` 最大 68.6 µs（別実行） |
| 未説明の残余 | 26.6 秒の `oldestAgeMs` | epoch 前進だけでは説明できない。要追加計測 |

要点は、**「EBR 上必要な待ち」と「publish が来ないことによる待ち」が同一の仕組みで、現状は区別できない**ことである。次の publish までの時間がそのまま待ちになるため、publish 間隔が長い区間では待ちが延びる。

### 1.5 追加計測の設計（承認後に実施）

エントリごとに次の 4 点を記録する。

| 項目 | 内容 | 用途 |
|---|---|---|
| `enqueueTs` | D キューへ積んだ時刻 | 滞留時間の起点 |
| `enqueueEpoch` | 積んだときの epoch | 安全条件の判定 |
| `safeTs` | `minReaderEpoch` が `enqueueEpoch` を超えたことを最初に観測した時刻 | **EBR 待ち**の終点 |
| `reclaimTs` | deleter を実行した時刻 | 解放の完了 |

分解式は次のとおり。

```
滞留 = (safeTs - enqueueTs) + (reclaimTs - safeTs)
       = EBR 待ち          + 実行待ち
```

- `safeTs` は drain のたびに「安全になったエントリ」を走査して記録する（Non-RT、O(滞留数)）。
- 実行待ち（`reclaimTs - safeTs`）が大きければ drain 機会の問題、小さければ EBR 待ちが主因と判定できる。
- 併せて「どの reader が最小 epoch を保持しているか」を、`getMinReaderEpoch` の走査で最小値を与えたスロット番号として記録する（reader ごとの滞留理由を切り分ける）。
- 観測不足（`safeTs` が取れない）は 0 として扱わず、`insufficient` に計上する。

### 1.6 未確定

- 26.6 秒の `oldestAgeMs` の内訳（1.5 の計測で確定する）。
- publish 間隔と待ちの関係（バースト間隔を変えた比較は未実施）。
- 複数 reader の最小 epoch 保持者の特定（現状は `stuckReaders` が 0 としか分からない）。

## §2 `RuntimeWorld` 寿命の測定設計（設計のみ）

### 2.1 現行の寿命（コードから確認した範囲）

1. `RuntimeBuilder::buildRuntimePublishWorld(...)` が `worldOwner`（`aligned_unique_ptr`）を生成する。
2. `commitRuntimePublication(std::move(worldOwner), ctx)` が公開を確定し、所有者を `worldAuthority_.ownerChannel()` に積む。
3. `RuntimePublishExecutor` が publication fence まで所有者を取り出し、**新しい所有者の受け入れに伴って旧所有者を解放する**。この解放が物理破棄である。
4. 旧 world の退役は `onRuntimeRetiredNonRt(world)` で通知され、`200→300` の HB エッジと `RetireIntent` が出る。
5. 終了時は `drainAllNonRt` が残りの所有者を解放する。

### 2.2 起点と終点（案）

| 定義 | 起点 | 終点 | 意味 |
|---|---|---|---|
| **W1（採用案）** | `commitRuntimePublication` が所有者を channel に積んだ時刻 | 旧所有者のデストラクタ実行時刻 | 「公開してから物理破棄まで」 |
| W2（参考） | 旧 world の参照カウントが 0 になった時刻 | デストラクタ実行時刻 | 純粋な破棄処理の時間 |

W1 を主指標とし、W2 は補助として同じ記録に残す。D1（DSP ハンドル）とは別の指標であり、混ぜない。

### 2.3 段階と待ちの区別

| 段階 | 待ちの性質 |
|---|---|
| channel への滞留 | 公開 fence・executor の巡回間隔 |
| fence 待ち | 公開シーケンスの確定待ち |
| クロスフェード保持 | 音声遷移の完了待ち（設計上の保持） |
| 参照保持 | reader が旧 world を参照している時間 |
| 破棄処理 | デストラクタの実行時間 |

### 2.4 測定対象となる経路

通常公開、公開拒否（`retireRejectedRuntimeWorldNonRt`）、初期化時の置換、再構成（reconfigure）、終了（`drainAllNonRt`）、deferred 公開。経路ごとに件数と最大を分けて記録する。

### 2.5 観測不足・不一致の扱い

- 要求と完了は world の識別子（`worldId` と世代）で対応付ける。ポインタは再利用され得るため単独では使わない。
- 完了が 0 件の要求は 0 秒として扱わない。`insufficient` に計上する。
- 二重完了は無効とし、件数を残す。
- 観測数は D1 と同じ考え方で `N_valid` を出し、必要数は別途定義する（D1 の 59 を流用しない）。

### 2.6 D1 との関係

- 測定対象が異なる（DSP ハンドル／ライフタイム 対 RuntimeWorld）。
- レポートは別の key（例: `worldLifetime`）に分け、D1 の 50 ms 判定に混ぜない。
- 実装は今回未着手。設計の承認後に、D1 と同じ 2 段階（runtime／final）で記録する。

## §3 RT 経路の要求記録方式（設計のみ）

### 3.1 現状

`retireRT(ptr, deleter)` は `provider_->enqueueRetire(...)` を直接呼び、`ISRRetireRouter::enqueueRetire` の要求フックを通らない。このため RT 経路の退役は `unmatchedCompletions`（実測 3 件）として現れる。

### 3.2 要件（Owner 条件）

RT スレッドで動的確保をしない、mutex を取らない、ログ・文字列生成をしない、実行時間を有界にする、対応付けを一意にする、既存の動作・タイミングを不必要に変えない。

### 3.3 方式比較

| 方式 | 要求記録のコスト | 完了探索 | RT 安全性 | 評価 |
|---|---|---|---|---|
| A. 既存レコーダをそのまま RT から呼ぶ | 512 スロットの線形探索（最大 512 回のアトミック読み） | ポインタ一致 | 確保・ロックなし。ただし最悪 512 回で有界だが比較的大きい | 実装が最小。RT 予算が厳しいと不利 |
| **B. TLS で確保したスロット範囲に O(1) で書く** | 自分の範囲から 1 スロット CAS（数回） | ポインタ一致（Non-RT 側で探索） | 確保・ロックなし、O(1) | **推奨** |
| C. MPSC リング（Vyukov）に publish | 1 回の CAS + 1 回の store | ポインタ一致（Non-RT 側で探索） | 確保・ロックなし、O(1) | リング満杯時の扱いが増える |

### 3.4 推奨案（B）の概要

- RT スレッドごとに固定スロット範囲（例: 64 個）を TLS で確保する。範囲は起動時に `registerReaderThread` 相当のタイミングで払い出す。
- 要求記録は「自分の範囲から空きスロットを CAS で取る → ID・epoch・時刻・ポインタを書く」の 3 手順。範囲が満杯なら `overflow` を 1 回インクリメントして諦める（確保しない）。
- 完了記録は既存どおり Non-RT 側でポインタを探索する。
- ログ・文字列生成は行わない。時刻は `steady_clock` の 1 回読み。

### 3.5 RT 安全性の根拠（提示内容）

- メモリ確保なし（固定配列のみ）、ロックなし（アトミックのみ）、ループは範囲長で有界、`fprintf` などの I/O なし。
- ただし「記録を足しても既存の RT 安全性が保たれる」ことの証明は別作業である。実装時は、(a) RT スレッドの最悪実行時間の前後比較、(b) 記録を有効にした soak での xrun・デッドライン超過の有無、(c) `retireRT` の呼出回数と記録件数の一致、で確認する。

## §4 B 修正案 A のレビュー資料（適用しない）

### 4.1 差分

**重要**: `src/InputInspection.cpp` は **git 管理外（未追跡）** のファイルである（`git ls-files` に存在せず、履歴も無い）。したがって 2026-11-08 のスナップショットには含まれておらず、公開済みの `ConvoPeq.md` から `computeMaxElementDepth` を確認できないのは当然である。レビューは作業ツリーの実ファイルに対して行う必要がある。

- 対象ファイルの実体: 6,948 バイト、SHA-256 `A4F4F01A31F2558B…`。
- レビュー用に同一内容を `patch/current_InputInspection.cpp.txt` として添付した（バイト一致）。
- 現行実装: その `computeMaxElementDepth`（13 行目から）。
- 提案実装と unified diff を `patch/b_fixA_computeMaxElementDepth.patch` に置いた（161 行、4 ハンク）。
- 一時ツリーへ `patch -p0` で適用し、**警告・fuzz なしで適用できる**ことを確認した。適用後の関数は提案ファイルとバイト一致する。
- 添付した現行ファイル（`current_InputInspection.cpp.txt`）に適用した結果も同じである。

### 4.2 同等性（パッチ適用後の関数で再確認）

26 ケースすべて一致（`summary: pass=26 fail=0`）。内訳は、空入力、自己終了タグ、通常の対、3 階層、閉じタグのみ、余分な閉じタグ、コメント、未終端コメント、CDATA、小文字 CDATA、未終端 CDATA、DOCTYPE、XML 宣言、未終端の処理命令、引用符内の `>`、単一引用符、未終端タグ、孤立した `<`、UTF-8 属性、UTF-8 タグ名、UTF-8 テキスト、BOM、前後の空白、1,000 階層、20 KB テキスト、40 KB 属性。

### 4.3 API の確認

| 項目 | 内容 |
|---|---|
| `juce::String::toRawUTF8()` | UTF-8 の先頭ポインタを返す。`String` が生存している間のみ有効（呼び出し中は引数の参照が生きている） |
| `juce::String::getNumBytesAsUTF8()` | UTF-8 のバイト数を返す。文字数（`length()`）ではない |
| `juce::String::operator[](int)` | UTF-8 では**文字位置**の参照であり、先頭からの走査を伴う（実測で O(位置) の挙動） |
| 構文判定の安全性 | 判定に使う文字（`<` `>` `/` `!` `?` `-` `]` `"` `'` と空白）はすべて ASCII。UTF-8 の継続バイトは 0x80 以上なので誤一致しない |

### 4.4 性能（同一ハーネス・同一環境）

| 入力 | 現行 | 提案 |
|---|---|---|
| 1,000 階層 | 17.5 ms | 0.008 ms |
| 20 KB テキスト | 50.9 ms | 0.019 ms |
| 40 KB 属性 | 197〜290 ms | 0.02〜0.04 ms |
| 278 KB（実ファイル、単体計測） | 12.4〜12.6 秒 | 未計測（同じ走査のため同水準と推定） |

### 4.5 レビュー項目との対応

| 項目 | 今回の提示内容 |
|---|---|
| XML 構文 | 26 ケース（自己終了・コメント・CDATA・DOCTYPE・PI）で一致 |
| 引用符 | 引用符内の `>` と単一引用符で一致 |
| UTF-8 | 属性・タグ名・テキスト・BOM で一致。不正入力（未終端）も一致 |
| 境界 | 空入力・1,000 階層・20 KB/40 KB で一致 |
| API | 4.3 のとおり |
| 同等性 | 26/26。追加候補は「DOCTYPE の内部サブセット」「コメント内のタグ風文字列」「NUL を含む入力」 |
| 性能 | O(n²) の解消を 40 KB で確認（197〜290 ms → 0.02〜0.04 ms） |
| 差分 | 現行 HEAD の関数と 1 対 1 の差分。パッチ適用で一致 |

### 4.6 未確認

- 実ファイルでの起動・終了時間（未適用のため未測定）。推定は起動 12.5 秒 → 約 2 秒、`saveSettings` 10.2 秒 → 約 0.1 秒。
- `computeMaxElementDepth` 以外の `juce::String` 添字走査（横断確認は未実施）。

## §5 成果物

- 本報告書: `doc/work113/20261110-1633_調査設計/report.md`
- B の差分: `doc/work113/20261110-1633_調査設計/patch/b_fixA_computeMaxElementDepth.patch`
- B の現行ファイル（レビュー用の同一コピー）: `doc/work113/20261110-1633_調査設計/patch/current_InputInspection.cpp.txt`
- 一時ハーネス: `C:\Users\user\AppData\Local\Temp\opencode\b0612\`（`decomp`・`equiv`・`patchtest`）
- 測定サンプル: `...\decomp\samples.csv`（100 ms 周期の `epoch_reclaim_audit`／`retire_timeline`）
