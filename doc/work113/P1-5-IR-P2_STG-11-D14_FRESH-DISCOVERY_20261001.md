# STG-11-D14 Fresh Discovery — Async Generation, Cancellation & Stale-Completion Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D14_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D14** — rebuild/generation/cancellation/async-callback/stale-completion 境界監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用。
  D12/D13 の結論の流用なし）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

generation identity / cancellation / PendingCommit ownership / retry-redrive /
publication admission / async callback lifetime を実コードで監査した結果、
D14 GO 条件 7 項目をすべて満たす concrete defect は存在しない。
Implementation / Gate は作成しない。

D1〜D11 の未 commit 変更・D12〜D14 監査成果物はすべて保持する。

---

## 1. Generation identity map（§3）

| identity | 生成 | 保存 | 比較 | 消費 | 無効化 |
| --- | --- | --- | --- | --- | --- |
| `convolverStateGeneration`（uint64） | `bumpGeneration()`（load 要求時） | `PendingCommit.requestGenId` | `isCurrentGeneration()`（commit 時） | commit 採用／破棄 | 次回 bump（単調） |
| `rebuildRequestGeneration`（int） | `++`（rebuild 要求時） | task.generation | `isRebuildObsolete()`（多 phase） | worker 継続／破棄 | 次回 increment |
| `lastCommittedRebuildGeneration`（int） | commit 確定時 | — | outstanding 判定 | drain 判定 | 次回 commit |
| `rebuildPendingAfterLoad`（bool） | commit Phase 4 | — | exchange（once） | 再 rebuild 1 回 | consume 時に false |

- D14-1：比較対象はいずれも同一 logical operation（load 要求／rebuild 要求）の
  identity であり、混用なし（convolver 系と engine-rebuild 系は別変数・別比較）。
- D14-2：uint64 世代の wraparound は実用上到達不能。0 初期値は task に付与されない
  （付与は bump 後の値のみ）。int 世代は単調増加・wraparound 未考慮だが、
  等価比較のため wraparound 時の誤判定は理論上のみ（2^31 要求は非現実的）。
- D14-3：request generation と runtime generation の混用なし（上表のとおり分離）。
- D14-4：check→publish 間の stale 割り込みは、commit 実行の thread 直列化により排除
  （下記 §2）。
- D14-5：generation check のない async completion は、idempotent 処理
  （UI 通知・latency 更新・再 rebuild 要求）のみ。state 変更を伴う commit は
  全経路で currency check あり。

## 2. Serialization 根拠（D14-4）

- uiConvolverProcessor の commit（`executePendingCommit`）は Message Thread のみ
  （async は `callAsync`、sync `runSynchronously` は DSP-owned instance 用であり
  ui instance では並行しない）。
- DSP-owned convolver の sync commit は rebuild worker（単一）上のみ。
- `runIncrementalFinalizeStep` の `getCurrentGeneration` 刻印は、
  job 固有の shouldCancel による直列化と組み合わせ（コメントに明示）。
- 以上により、check 後の stale publish の割り込み経路なし。

## 3. Cancellation map（§4）

- `LoaderThread`：`externalCancellationCheck`（threadShouldExit＋saved）の
  多点検査。cancel 後の `newConv` は retire。error path は RAII 解放＋UI 通知。
- `rebuildAllIRsSynchronous`：`shouldCancel` 伝播。`isRebuildObsolete` の多 phase 検査。
- `RetryScheduler::shutdown`：queue clear＋join。`schedule` は exit 後 reject。
- `stopRebuildThread`／`shutdownCoordinatorLoop`／`forceCleanup` の停止順序は
  dtor／releaseResources で固定。
- 「cancel requested だけ」で済ませている箇所なし（worker 観測＋cleanup あり）。

## 4. PendingCommit ownership map（§5）

- create（`applyNewState`）→ populate → consume（`executePendingCommit`）→
  destroy の全経路で単一所有（`unique_ptr`）。stale 破棄時は destructor が
  engine を retire。例外時は flag reset。`callAsync` 失敗時は同期 cleanup。
- A/B 同時存在は generation check で直列解決（stale 側は swap なし）。
  destructor の非所有 retire なし。failed commit の巻き戻しなし。

## 5. Retry / redrive map（§6）

- `RetryScheduleRequest` は enum のみ（ownership なし）。重複 dispatch は
  `submitRebuildIntent` の latest-wins merge で冪等。
- Recovery redrive は `delivery==None` のみ＋budget＋failure signal。
  二重 delivery の証拠なし。

## 6. Publication admission boundary（§7）

- rebuild worker の多 phase obsolete 検査（before prepare／prepare／rebuildIR／
  warmup）により、stale task は publish 前に破棄される。
- A-start→B-start→B-publish→A-complete シーケンスでは A の generation が
  陳腐化しているため A は破棄される（A による上書きなし）。
- `RuntimePublicationValidator` が resource／topology／transition を関門化。

## 7. Async callback lifetime（§8）

- commit chain は `WeakReference`＋破棄時 retire。`[this]` capture の
  `RetryScheduler::DispatchFn` は `shutdown()` で join され、AudioEngine より
  先に停止する（destructor 順序で固定）。
- shutdown 後実行・二重 consume・check なし state 変更の証拠なし。

## 8. なぜ NO-GO か

D14 GO 条件 7 項目に照らし、「既存 guard で完全に中和されていない」
「具体的なユーザー影響がある」を満たす候補が存在しない。
最も近い TOCTOU 仮説（§2）は thread 直列化により不成立、
世代共有仮説は instance 分離＋worker 単一性により不成立であることを実証した。

## 9. 推奨

- uint64/int 世代の wraparound は理論上のみ。記録のみ。
- R8 flake・R4（P3）・dead code 一括整理は引き続き未解決として残る。

## 10. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12〜D14 監査成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
