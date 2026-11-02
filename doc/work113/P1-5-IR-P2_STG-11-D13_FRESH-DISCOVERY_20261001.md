# STG-11-D13 Fresh Discovery — Resource-Lifecycle-Failure Boundary Audit

- Document: `doc/work113/P1-5-IR-P2_STG-11-D13_FRESH-DISCOVERY_20261001.md`
- Work item: **STG-11-D13** — resource / lifecycle / failure / shutdown / ownership 境界監査
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  D11 マーカー 8 件・FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用。
  D12 の結論の再利用なし — 必要箇所は最新ソース上で再確認した）
- Commit / push: **禁止**

---

## 0. 判定

```
NO-GO — no concrete defect
```

resource acquisition failure / loader-builder failure propagation /
shutdown ordering（C-1〜C-7）/ queue-fallback ownership / retire-epoch lifetime /
RT reachability を実コードで監査した結果、D13 GO 条件 7 項目をすべて満たす
concrete defect は存在しない。D12 と同じく推測による修正は行わない。
Implementation / Gate は作成しない。

D1〜D11 の未 commit 変更・D12/D13 監査成果物はすべて保持する。

---

## 1. Shutdown ordering audit（§2.C）

### 1.1 Member destruction order（C-3）

`AudioEngine.h` 宣言順：`m_epochDomain`（:5069）→ `m_retireRouter`（:5074）→
`m_coordinator`（:5081）→ … → `runtimePublicationBridge_`（:5241）→
`worldAuthority_`（:5243）→ `shutdownRuntime_`（:5277）。
破棄は逆順のため、provider（domain）は router より長生きする。
`m_retireRouter->provider_ == &m_epochDomain` の dangling なし。**正常。**

### 1.2 Destructor / releaseResources pipeline

- `~AudioEngine`：phase 固定（StopAcceptingWork→StopAudio→StopWorkers）、
  timer 停止（`stopTimer`）、AsyncUpdater cancel、RetryScheduler shutdown、
  Coordinator join、rebuild join、handle-based retire（pointer-value 廃止済み）、
  graceful drain（5s）→ clear → drain → Q/E/T 強制 drain → residual 監視
  （`jassert`＋diag）。**具体的な欠落なし。**
- `releaseResources` terminal：reconfigure/terminal 分離維持、graceful→final→
  Emergency→強制 drain の多段構成。**具体的な欠落なし。**
- `drainAll()` は provider＋Q＋E＋T を全カバー（`ISRRetireRouter.cpp:593`）。
  C-2（残存 ownership 未 drain）の証拠なし。
- C-1（shutdown 中の新規 work）：`RetryScheduler.shutdown` が dispatch 停止、
  rebuild/loader/ coordinator は join 済み。`submitRecoveryIntent` は
  flag＋durable 消費で lost-wakeup なし。新規 work の混入経路なし。
- C-4（stale access）：timer/cancel、WeakReference（commit chain）、
  `forceCleanup` の loader 停止の組み合わせ。証拠なし。
- C-6（flag のみで drain 待ちなし）：5s loop＋強制 drain＋residual 監視あり。なし。
- C-7（shutdown 限定の直接 delete）：`retireImmediateDuringShutdown`
  （`EQProcessor.Core.cpp:99-107`）が該当パターンだが **caller 0 件**の dead code。
  到達不能のため欠陥ではない（記録のみ）。

### 1.3 `~EQProcessor` の retire 経路

`retireEQStateDeferred`／`retireBandNodeDeferred`＋drop count。
coordinator 不在時は `false`→counter（comment に記載の設計）。
shutdown 終了時の OS 回収域であり、concrete harm なし（記録のみ）。

---

## 2. Loader / builder failure propagation audit（§2.B）

- `LoaderThread`：decode/alloc/read/cancel の全 failure で errorMessage＋
  部分 object の retire／RAII 解放（`:296-310` の callAsync-failure path を含む）。
  `runSynchronously` の失敗時は `newConv` を retire。**old valid 維持。**
- `applyNewState`：`callAsync` 失敗時は `releaseEngine` で同期 cleanup。
  stale generation は destructor retire。例外時は flag reset。
  `executePendingCommit` は swap 前に currency／metadata を確定。
  **新旧 residency の不整合なし。**
- `PendingCommit` destructor が engine を retire（所有権の終端明確）。

---

## 3. Queue / fallback ownership audit（§3）

- retire chain（D→Q→E→T）は D1/D2/D11 で確定済み。terminal は growable で常時受領。
- `signalDrainWakeup`／`waitForDrainSignalOrTimeout` は B-R3 修正済み。
  consumer（CoordinatorLoop）は 1ms timeout fallback 付き。lost-wake なし。
- `drainDeferredRetireQueues`／`pendingReclaimHandles_` の再試行経路は
  shutdown 時の明示 drain でカバー。「counter は増えるが ownership は失われる」
  型の実装なし（counter と residency の対応は D2/D11 で確定済み）。

---

## 4. Retire / epoch lifetime audit（§4）

- publish 済み object の bypass：`retireDSPHandleForRuntime` 単独誤用パターンは
  production caller が `DSPLifetimeManager` のみに限定（D6 で確定。今回再確認）。
- `DSPGuard`（`RebuildDispatch.cpp:983`）の false 経路は直接破棄（if-else）。
  過去の二重解放は修正済み。現行正しい。
- `m_retireSink == nullptr` 経路は ctor 設定により到達不能（D6 で確定）。
- `seqId=generation=0` 格納は metadata であり safety 無関係（D6 で確定）。

---

## 5. RT reachability audit（§5）

- retire/terminal/quarantine の mutex＋heap は NonRT producer のみ到達
  （B-R2-2/R3-5 の caller 列挙が authoritative。D6 で確定。今回変更なし）。
- `retireRT`／`releaseRT`／`enqueueRetireEpochBounded` は caller 0 件の dead code
  （D6 で確定）。到達不能のため欠陥ではない。

---

## 6. なぜ NO-GO か

D13 GO 条件 7 項目に照らし、「既存 guard で完全に中和されていない」
「具体的なユーザー影響がある」を満たす候補が存在しない。
最も近い `retireImmediateDuringShutdown`（C-7 パターン）は到達不能、
`~EQProcessor` の shutdown leak は OS 回収域である。

---

## 7. 推奨（将来の harm 証明時に再評価）

- `retireImmediateDuringShutdown` 等の shutdown 限定直接 delete 群は
  dead code として一括整理の候補（到達不能を確認済み）。
- `~EQProcessor` の coordinator 不在時 drop は shutdown 時のみ。記録のみ。
- D6 NO-GO・R8 flake・R4（P3）は引き続き未解決として残る。

---

## 8. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12/D13 監査成果物はすべて保持
- Implementation / Gate は作成しない（NO-GO のため）
