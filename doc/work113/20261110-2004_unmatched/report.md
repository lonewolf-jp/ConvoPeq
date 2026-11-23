# unmatched 記録（設計 v2）の実装報告（2026-11-10）

## §0 範囲

承認された設計 v2 の範囲で実装した。C2 のリーク修正は含めていない（別提案 `c2_leak_proposal.md`）。commit／merge／release は行っていない。

## §1 実装内容

### 1.1 レコーダ API

| API | 内容 |
|---|---|
| `recordRequest(..., bool* created = nullptr)` | 新規作成時のみ `*created = true`。既存記録を再利用した場合は false |
| `recordRequestRT(..., bool* created = nullptr)` | RT 版。TLS 区画内の走査のみ |
| `cancelRequestById(object, id)` | object・id・状態 ready の 3 条件一致時のみ CAS で解放 |
| `cancelRequestRTById(object, id)` | 同条件を TLS 区画内で行う RT 版 |
| 既存 `cancelRequest(ptr)` | 互換のため残置（新規経路では使わない） |

再利用時は、未確定の最小 reader 情報（`minReaderSlot` が未設定）だけを補完する。既知の情報は上書きしない。

### 1.2 記録位置

| 経路 | 位置 | 使用 API | 失敗時 |
|---|---|---|---|
| A（既存） | `ISRRetireRouter::enqueueRetire` | `recordRequest` | 変更なし |
| **B** | `RuntimeIntentCoordinator::enqueueRetire`（`ISRRuntimePublicationCoordinator.cpp`） | `recordRequest`（A で再利用） | 非 Success かつ自分が作成した場合のみ `cancelRequestById` |
| **C1** | `SnapshotCoordinator::startFade` | `recordRequest` | 取り消さない（quarantine が受領） |
| **C2** | `SnapshotCoordinator::resetFadeStateAndRetireTarget` | `recordRequestRT` | 未格納かつ自分が作成した場合のみ `cancelRequestRTById` |
| **C3** | `SnapshotCoordinator::completeFade` | `recordRequest` | 取り消さない |
| **C4** | `quarantineRetireSink` | 追加なし | 将来の直接呼出しへの注意コメントを追加 |
| **D** | `ISRRetireRouter::retireRT` | `recordRequestRT` | 自分が作成した場合のみ `cancelRequestRTById` |

## §2 必須条件への対応

| 条件 | 対応 |
|---|---|
| 1. D 経路の取消しも ID 単位 | `retireRT` を「作成フラグ＋ID 一致取消し」に置換（`cancelRequestRTById`） |
| 2. 取消しと完了の競合 | 取消しは CAS（ready→free）が成功した場合のみ。失敗時は何もしない。単体で競合テストを実施 |
| 3. RT 制約の維持 | `recordRequestRT`／`cancelRequestRTById` は TLS 区画内の走査のみ。ロック・確保・待機なし |
| 4. 記録位置以外の変更なし | `evaluateRetireDelay` は実装前後でバイト一致（2,626 バイト）。C2 のリーク修正は未実施 |
| 5. 検証項目の実施 | §3 のとおり |

## §3 検証結果

### 3.1 単体（レコーダ v2、15 チェック）

| 検証 | 結果 |
|---|---|
| 同一ポインタは同じ requestId | PASS |
| 1 回目 created=true / 2 回目 false | PASS |
| B→A の二重記録で `requests` が 1 件 | PASS |
| 誤った ID では取り消せない（`cancelled` が増えない） | PASS |
| 記録は有効なまま完了と対応する | PASS |
| 正しい ID では取り消せる | PASS |
| 取消し後の完了は `unmatched` | PASS |
| 完了後の取消しは no-op（完了記録を壊さない） | PASS |
| RT 記録は created=true | PASS |
| RT 取消しが成立する | PASS |
| RT 取消し後の完了は `unmatched` | PASS |
| 競合（完了と取消し）で片方だけが成立し件数が整合する | PASS |
| 両方が成立しない | PASS |

`summary: failures=0`。

### 3.2 アプリ実行

| 実行 | requests | completions | D1 `unmatched` | world `unmatched` | 判定 |
|---|---|---|---|---|---|
| 交互バースト 40 回 | 1,846 | 1,846 | **3** | 1 | fail（最大 4,444 ms） |
| EQ 変更（バーストなし） | 184 | 184 | **3** | 1 | fail（観測 100） |

**D1 の `unmatched` は 3 のままで、検証項目「D1 unmatched = 0」を達成していない。**

### 3.3 判定式の不変

`evaluateRetireDelay` の関数テキストが、実装前のバックアップ（`tmp/rollback/item1_3/ISRHB.cpp`）とバイト一致することを確認した。

### 3.4 ctest

```
100% tests passed out of 60
Total Test time (real) = 358.59 sec
ctest_exit=0
```

`AudioEngineHarness` を含む 60 件すべて合格した。設計 v2 の記録追加による回帰は検出されていない。

## §4 未達項目の原因（重要）

### 4.1 原因

`AudioEngine::enqueueDeferredDeleteNonRtWithResult`（`src/audioengine/AudioEngine.h:4673` 付近）は、**shutdown 中に `shutdownReclaim` を直接呼ぶ**分岐を持つ。

```cpp
if (isShutdownInProgress())
{
    const uint64_t epoch = markRetireEpoch();
    const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
    return transferred ? RetireEnqueueResult::Success : RetireEnqueueResult::Shutdown;
}
```

`shutdownReclaim` → `terminalReclaim` は完了フックを持つが、**要求フック（`enqueueRetire`）を通らない**。したがって shutdown 中に deferred delete されたオブジェクトは、完了だけが記録され `unmatched` になる。

件数が実行条件によらず 3 で一定なのは、shutdown 時に defer されるオブジェクト数が固定（active／fading の DSP ハンドル等）であることと整合する。

### 4.2 C 経路のフックは今回の発火なし

B／C1〜C3 のフックを入れた前後で `requests` の総数が変わっていない（1,846 → 1,846、184 → 184）。今回のシナリオ（bypass バースト、EQ 変更）では `SnapshotCoordinator` の退役が発生しておらず、C 経路は**未発火**である。C1〜C3 の対応関係は単体では確認できるが、アプリでの実測はできていない。

### 4.3 提案（承認範囲外のため未実装）

shutdown 分岐にも 1 か所だけ記録を追加する。

```cpp
if (isShutdownInProgress())
{
    bool created = false;
    const auto id = recorder.recordRequest (ptr, epoch, ..., &created);
    const bool transferred = m_retireRouter->shutdownReclaim (ptr, deleter, epoch, type);
    if (! transferred && created && id != 0)
        recorder.cancelRequestById (ptr, id);
    return transferred ? Success : Shutdown;
}
```

これで D1 `unmatched` は 0 になる見込みである。ただし承認された記録位置（B・C1〜C4・D）の外側であり、**今回の承認範囲外**のため実装していない。

## §5 変更ファイル

| ファイル | 内容 |
|---|---|
| `src/audioengine/ISRRetireLatency.h` | `created` フラグ、`cancelRequestById`、`cancelRequestRTById`、再利用時の最小 reader 補完 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | B の記録と取消し |
| `src/core/SnapshotCoordinator.cpp` | C1／C2／C3 の記録、C2 の RT 記録・取消し、C4 の注意コメント |
| `src/audioengine/ISRRetireRouter.cpp` | D の ID 単位取消し |

## §6 ロールバック

**実装前スナップショットは取得していない**（手順の不備。今後の同種変更では事前に取得する）。実装後の状態は `tmp/rollback/itemC_v2_after/` に保全した（4 ファイルと SHA-256）。

ロールバックは次の 2 段階で行う。

| レベル | 内容 | 対象 |
|---|---|---|
| 1（挙動の復元） | 追加したフックの呼び出しを削除する。API 追加は未使用なら無害なため残せる | `ISRRuntimePublicationCoordinator.cpp` の B ブロック、`SnapshotCoordinator.cpp` の C1／C2／C3 ブロック、`ISRRetireRouter.cpp` の `retireRT` の記録・取消し、および 3 ファイルの include |
| 2（完全復元） | `ISRRetireLatency.h` の `created` 引数・`cancelRequestById`・`cancelRequestRTById` も削除する | 上記に加えて `ISRRetireLatency.h` |

レベル 1 は「記録を止める」だけであり、所有権・retry・reclaim の経路には触れていないため、削除しても挙動は実装前と同一になる（`evaluateRetireDelay` は無変更）。
