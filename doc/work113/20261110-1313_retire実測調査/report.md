# 調査 A: B-11 "retire_delay" 実測の妥当性

## §1 実施した調査・測定

### 1.1 コードの確認（変更なし）

| 対象 | 内容 |
|---|---|
| `src/audioengine/ISRHB.cpp:287` `checkRetireDelayBound(maxNs)` | 引数の `maxNs` と判定式 |
| `src/audioengine/ISRHB.cpp:213` `simulateReorderScenario` | `RetireDelay` が `checkRetireDelayBound(50'000'000u)` を呼ぶ |
| `src/audioengine/ISRHB.cpp:226` `runScenarioSuite` | 4 シナリオの並び |
| `src/audioengine/ISRHB.cpp:13` `recordEdge` | イベント時刻の取得方法 |
| `src/audioengine/AudioEngine.Commit.cpp:385/485`、`ReleaseResources.cpp:775` | `recordHBEdge` の呼出箇所 |
| `doc/` 全文 | シナリオの意図（retire 遅延の定義）を記述した仕様の有無 |

### 1.2 実測

`--cli-run --cli-exit-ms <ms> --verify-runtime-evidence-live` でアプリを実行し、`evidence/hb_graph_trace.json` と `hb_violation_report.json` を取得した。

| 実行 | 実行時間 | イベント数 | `retire_delay` | 最大イベント間隔 |
|---|---|---|---|---|
| 1 | 12 秒 | 4 | fail | 10,947 ms |
| 2 | 12 秒 | 4 | fail | 13,819 ms |
| 3 | 12 秒 | 4 | fail | 11,061 ms |
| 4 | 30 秒 | 4 | fail | 11,027 ms |
| 5（詳細） | 30 秒 | 4 | fail | 10,758 ms |

30 秒実行のイベント列（`steady_clock` の ns）:

```
#0  from=100 to=200  epoch=1→1  +0 ms
#1  from=100 to=200  epoch=2→2  +150.081 ms
#2  from=200 to=300  epoch=1→1  +0.002 ms   （retire エッジ）
#3  from=100 to=200  epoch=3→3  +10,758.603 ms
```

## §2 直接確認できた事実

1. 判定は `checkRetireDelayBound(50'000'000)` で、**引数は 50,000,000 ns（50 ms）**。実装は「イベント列の隣接 2 件の時刻差が 50 ms を超えたら false」であり、イベントが 2 件未満なら **true（無条件合格）** を返す。
2. 比較対象は **HB トレースの全イベント**（publish エッジ 100→200、retire エッジ 200→300 を含む）であり、retire 完了までの時間ではない。
3. イベント時刻は `recordEdge` 内の `std::chrono::steady_clock::now()`（単調時計、ns）。記録は publish／retire の 3 箇所のみ。
4. 実測では 1 回の実行で **イベントは 4 件**。実行時間を 12 秒から 30 秒へ延ばしても 4 件のままだった。
5. 30 秒実行の内訳では、retire エッジ（#2、epoch 1）は直前の publish エッジ（#1、epoch 2）の **0.002 ms 後**に記録されている。ただし epoch が異なるため、これは「同一 epoch の publish から retire までの遅延」ではない。最大間隔 10,758 ms は **#2 と #3（次の publish）の間**であり、retire の処理遅延ではない。
6. イベント順は epoch 1 の publish（#0）→ epoch 2 の publish（#1）→ epoch 1 の retire（#2）→ epoch 3 の publish（#3）である。**epoch 1 の retire は、次の epoch の publish より後に記録されている**。
7. 5 回すべてで fail。再現性はある。失敗の直接の原因は「隣接イベント間隔が 50 ms を超えたこと」である。これが retire 遅延の超過を意味するかは定義次第で、断定できない（§3・§5）。
8. `doc/` には本シナリオの意図（retire 遅延の定義、50 ms の根拠）を記述した仕様が見当たらない。`doc/ConvoPeqMD/` はコードのスナップショットのみ。

## §3 事実から導いた判断

- このシナリオは「retire 遅延が 50 ms 以内か」を測っていない。**測っているのは「HB イベントが 50 ms 以上空かないこと」**であり、期待値（retire 遅延）と測定対象（イベント間隔）が一致していない。
- したがって、この判定式の fail は「retire 遅延が 50 ms を超えた」ことの直接の証拠ではない。一方で、**「イベントが 10 秒以上空いた」という事実自体は記録として正しい**（時計は単調、時刻は記録済み）。
- ただし、同一 epoch（1→1）の publish エッジ #0 と retire エッジ #2 の差は **150.08 ms** であり、これを retire 遅延と解釈するなら 50 ms を超える。定義が未確定なため「超過した」とも「していない」とも断定しない。判断に必要なのは retire 要求時刻の記録である。
- イベントが 2 件未満の場合は無条件合格になるため、**観測量が足りない実行ほど合格しやすい**という構造になっている。これは「測定できていないものを合格にしない」という方針と整合しない。
- 閾値（50 ms）や判定条件を変えて PASS にする操作は行っていないし、行うべきでもない。修正するなら「測る対象」を retire 遅延に合わせる必要がある。ただし §5 のとおり、測定量を変えても合格するとは限らない。

## §4 未確認事項と制約

- シナリオの本来の意図（何を retire 遅延と呼ぶか）は仕様が無く確定できない。`RetireDelay` という名前と 50 ms という定数からの推定に留まる。
- publish エッジが 12 秒で 3 件しか出ない理由（イベント間引きの設計か、CLI 実行の特性か）は未確認。`recordHBEdge` の呼出は publish 経路の 2 箇所と retire 経路の 1 箇所であり、publish 頻度は実行内容に依存する。
- 実機のオーディオ負荷が高い状態での retire 遅延そのものは測っていない。
- 今回の測定は CLI 実行（音声デバイス使用）であり、CI の実行環境とは負荷が異なる可能性がある。

## §5 追加変更が必要な場合の提案（未実施・承認待ち）

1. **シナリオの測定量を retire 遅延に合わせる**
   - 例: 同一 epoch の publish エッジ（100→200）と retire エッジ（200→300）の時刻差を retire 遅延として測る。
   - **この定義でも今回の実測は合格しない。** 同一 epoch（1→1）の publish エッジは #0（`ts=251912749275600`）、retire エッジは #2（`ts=251912899359100`）で、差は **150.08 ms** であり 50 ms を超える。0.002 ms は #1（epoch 2 の publish）と #2（epoch 1 の retire）の差で、epoch が異なるため retire 遅延ではない。
   - 真の retire 遅延を測るには **retire 要求時刻の記録が別途必要**である（現行トレースには要求時刻が無い）。
   - 影響: 判定条件の変更は CI 契約の変更になる。値が変わるだけで合格を保証するものではない。
   - 併せて「イベント 2 件未満は不合格（観測不足）」に改める案がある。
2. **測定用のイベントを増やす**（publish を定期的に発行する、または retire 専用の計測点を追加する）
   - 影響: トレースサイズの増加、publish 経路への計測追加。
3. **シナリオを廃止し、別の検証（retire 完了の観測）に置き換える**
   - 影響: v5 の検査項目が減る。代替の検査を用意する必要がある。

いずれもソース・テスト・閾値の変更を伴うため、今回は実施していない。現行の失敗結果は維持している。
