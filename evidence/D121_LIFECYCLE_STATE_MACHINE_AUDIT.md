# D121 — Lifecycle State-Machine / Ownership Contract Re-Proof

**Date:** 2026-08-29
**性質:** read-only 設計・コード監査。**production source 変更 0**。commit・Phase-II・production fix 凍結継続
**Frozen HEAD:** `a65ace1` + D117 観測トレース（macro-gated 2 ファイル）

---

## D121-A — DSPCore ↔ DSPHandle ↔ RuntimeWorld identity/ownership matrix（G1）

状態: `Constructing → Active → CrossfadingIn/Out → Retired →（Quarantined / DestroyPending）→ Reclaimed`

| 操作 | runtimeDSPHandleMap_ (DSPCore*→handle, mutex) | registry_ slot | activeRuntimeDSPHandle_ | fadingRuntimeDSPHandle_ | production caller |
| --- | --- | --- | --- | --- | --- |
| `create(dsp)` | — | slot pop（freelist O(1)）、gen+1、instance=、**Constructing** | — | — | registerDSPHandleForRuntime のみ |
| `registerDSPHandleForRuntime(dsp)` | find→existing / insert（**冪等**） | （create 経由） | — | — | lifetime.activate / Commit.cpp:804 / rollback(AudioEngine.h:4585) / DSPTransition.h:92 |
| `lifetime.activate(newDSP)` | **map 登録のみ** | — | **不変** | **不変** | onPublishCompleted |
| `DSPHandleRuntime::activate(h)` | — | slot→**Active** | **= h** | **= null**（副作用） | **0（unit test のみ）** |
| `beginCrossfade(from,to)` | — | from→CrossfadingOut / to→CrossfadingIn | — | **= from** | DSPTransition（到達不能経路） |
| `endCrossfade(id)` | — | from→Retired / to→Active | **= to** | **= null** | Timer.cpp:954（snapshot-fade ゲート内＝実質到達不能） |
| `retire(handle)` | — | slot→**Retired** | — | — | retireDSPHandleForRuntime |
| `reclaim(handle)` | — | instance=null / →**Reclaimed** / freelist push（gen 一致検証・二重 reclaim 防止） | — | — | Coordinator requestReclaim のみ（DELETE-2 専用） |
| `resolve(handle)` | — | gen 不一致→**stale**、Reclaimed/Quarantined→invalid、**Retired→valid**（instance 返却） | — | — | 多数 |
| `retireDSPHandleForRuntime(dsp)` | **erase** | retire + requestReclaim | — | — | DSPLifetimeManager::retire / （旧 DSPGuard フォールバック） |
| RuntimeWorld 参照 | — | — | — | — | `graph.activeNode/fadingNode`（**void\* visibility projection、ownership なし**） |

**D121-2 結論 — activate の完全分離と invariant 定義**:

- **registration**（map 登録）= `registerDSPHandleForRuntime` / `lifetime.activate`
- **activation**（active handle 公開）= `DSPHandleRuntime::activate`
- 両者は**別操作**であり、契約コメント（AudioEngine.h:4486「commitRuntimePublication が唯一の Authority」、:4536「Execution tail の activate 責務」）は後者を要求するが実装は前者しか実行しない
- `activeRuntimeDSPHandle_` の invariant（一文定義）: **「直近に commit が完了した RuntimeState の topology.activeDSP に対応する handle であり、次回 publish の oldHandle 取得の唯一の供給源」** — 現行実装はこの不変条件を満たす writer を欠く（D120-2 の再確認）
- **危険な結合**: `activate()` は active 公開と同時に `fadingRuntimeDSPHandle_ = null` を行う → 「active 公開」と「fading 解消」が 1 操作に束ねられ、crossfade 機構と結合。単純に「呼べばよい」ものではない（D119 暴走の構造的要因の一つ）

## D121-B — Publish transaction 前後状態表（G3）

| ケース | newDSP | oldDSP | oldHandle | crossfade | 現行の結果 / 必須結果 |
| --- | --- | --- | --- | --- | --- |
| 初回 publish | valid | null | null | no | new が current（activate のみ）— **正当** |
| 通常 publish | valid | valid | valid | no | **即時 retire（到達条件: oldHandle 非 null — 現行は常に null で到達不能）** |
| crossfade publish | valid | valid | valid | yes | old fading → 完了後 retire（完了駆動が存在しない — D121-C） |
| **oldDSP valid / oldHandle null** | valid | valid | **null** | any | **両分岐 skip → 漏出（現行の常態）**。契約は**未定義** |
| register failure | valid | — | — | — | rollbackHandle で rollback 済み（AudioEngine.h:4585 ScopeExit）→ ownership invariant 維持 |
| duplicate publish | — | — | — | — | map erase 済みなら retire は冪等 skip（retireDSPHandleForRuntime が false）→ **no double retire**（DSPLifetimeManager::retire は retired=false で return） |
| shutdown publish | — | — | — | — | shutdown guard（isShutdownInProgress）で publish 不発 / destroyForShutdown 経路が別管轄 |

**「oldDSP != nullptr && oldHandle == null」の判定材料**: 現行では `decision.oldHandle` が「常に null（active handle 未公開）」のために発生する**構造的常態**であり、偶発的不整合ではない。ゆえに D122 では (a) active handle 公開の配線により発生自体を消すのが本筋、(b) それでも残るケース（family #1-#5 の旧 world）は DSPTransition 内の map lookup（`registerDSPHandleForRuntime` 冪等）で復元可能 — **「復元可能状態」**として契約化するのが正しい（graph.activeNode 復元は bootstrap placeholder を誤破棄したため不採用 — D119 実測）。

## D121-C — DSP crossfade の独立状態機械は存在するか（G4/G5）

実コードで確定した1本の call graph:

```text
【START】DSPTransition::onPublishCompleted（crossfade 分岐）
  registerCrossfade（CrossfadeAuthorityRuntime）→ beginCrossfade（registry 状態+fading handle）
  crossfadeRuntime_.start(): pending_=true、gain_.reset（ramp 総ステップ設定のみ）
     ↓
【PROGRESS】RT（AudioBlock）:
  armCrossfadeIfPending: applyImmediateValueRT(0.0) + setTargetValue(1.0)（BUG-028 修正済み）
  gain(LinearRamp) 進行中は canCrossfade ミキシング（AudioBlock.cpp:384）
  ramp 到達 → isSmoothing()=false → **RT は完了イベントを発行しない**
     ↓
【存在しないリンク】完了検出 → notifyFadeComplete(id)
     ↓
【CONSUME】Timer: consumeCompletedFade → endCrossfade → fading slot CAS → retire
```

- `notifyFadeComplete` の production caller は **Timer.cpp:950 のみ**で、それは `m_coordinator.tryCompleteFade()`（Snapshot fade）のゲート内 → **寄生**
- `getFadeAgeUs()`（Practical-2 Timeout 監視用）は存在するが、超過を検出して通知する経路は未実装（HealthMonitor は drop count のみ監視）
- 結論: **「START→PROGRESS」は実装済み、「COMPLETION 検出→イベント→NonRT handling→retire」は未実装の未完成状態機械**

## D121-5 — Crossfade 完了方式の設計比較（コード変更なし）

| 軸 | A: RT ramp 完了→event→Timer | B: NonRT Timer が isSmoothing() 監視 | C: CrossfadeRuntime が state 公開→Timer consume |
| --- | --- | --- | --- |
| RT allocation = 0 | ✅（既存 SPSC `completedFadeQueue_` push のみ） | ✅ | ✅ |
| RT blocking = 0 | ✅ | ⚠️ NonRT が LinearRamp（非 atomic）を直接読むとデータレース — 間接読み設計が必要 | ✅（atomic batch publish） |
| RT delete = 0 | ✅ | ✅ | ✅ |
| ownership ambiguity | 低（id 指定イベント） | 低 | 低 |
| duplicate completion | 1 回限り push のフラグで防止可能 | 二重検出に注意 | state 遷移で防止 |
| overlap handling | CrossfadeAuthorityRuntime の直列化前提（現行 jassert ≤1）+ pending_ AND 条件（armCrossfadeIfPending 済） | 同左 | 同左 |
| generation safety | start/complete の generation anchor 済（BUG-028 五次レビュー §8） | — | 同左 |
| shutdown safety | queue はテアダウン drain 対象 | 同左 | 同左 |
| authority singularization | ✅ 検出は RT、決定は Timer、実行は DELETE-1/2/3 | ✅ | ✅ |
| I4 DELETE-1/2/3 | ✅（retire decision は Timer/NonRT、destroy は drain） | ✅ | ✅ |

**推奨: A**（既存の `notifyFadeComplete`/`consumeCompletedFade`/SPSC/generation anchor が設計済みで、未実装のリンクは「RT が ramp 完了を検出して1回 pushする」一点のみ）。B は LinearRamp 非atomic 読みのデータレース回避が本質的に困難、C は A の等価代替（既存 API 整合性で A が優位）。**SnapshotCoordinator をトリガーにする案は除外**（独立機構 — G5 確定済み）。

## D121-D — Retire authority の最終確定（G6）と Observe 分離（G7）

**retire decision authority と physical destruction authority の分離（再証明）**:

```text
retire decision: Execution tail（DSPTransition）— oldDSP identity が確定する唯一の時点
  ↓ DELETE-1: DSPLifetimeManager::retire(DSPCore*)
     retireDSPHandleForRuntime（map erase = authority acquisition）+ slot→Retired
  ↓ DELETE-2: requestReclaimHandle
     epoch 安全 → requestReclaim（waitReaders → reclaim slot 状態遷移のみ）
     不安全   → pendingReclaimHandles_ 保留 → drainDeferredRetireQueues 再試行
  ↓ DELETE-3: enqueueWithRetry(destroyDSPCoreNode)
     RetireRouter drain が epoch 通過後に物理削除（失敗時は RetireQuarantineStore 移送）
```

REPAIR_PLAN.md の原則（requestReclaim は物理削除しない / directDelete 禁止）と整合。**R-A（DSPTransition での retire decision）を採用**、R-B は完了駆動が A を前提とする実行順序の問題、R-C/R-D は不採用（Observer 副作用禁止・singularization）。

**Observe 分離の可否（G7）**: ObserveIntentHandler は現状 `retireByHandle` を呼ぶ（retire 誘発）。Observe を観測専用に戻すには:
1. Timer 3 箇所 + DSPTransition dead 1 箇所の `submitObserve`（retire 誘発目的）を撤去
2. ObserveIntentHandler の役割を telemetry/metrics へ限定し、`retireByHandle` 呼び出しを削除
3. stale epoch filter（`intent.epoch < currentEpoch → 破棄`）の意味論は Observe の観測用途として妥当化
→ **分離可能**。ただし現行で Observe が担っている「fading DSP の回収」機能は R-A の即時 retire（crossfade 非依存）により不要化される。

## D121-E — Shutdown teardown audit（G9・D119-BLOCKER-2）

実測した teardown 順序（MainApplication::shutdown:174 → ~MainWindow:1120 → ~AudioEngine:CtorDtor.cpp:103）:

```text
[CLI] Auto-exit flush（Timer::callAfterDelay）
  → juce::Logger::setCurrentLogger(nullptr)   ← ★ ここで FileLogger 切断（以後の全ログ喪失）
  → systemRequestedQuit → JUCE shutdown
MainApplication::shutdown()（ログはファイルに残らない）
  → mainWindow.reset()
    ~MainWindow: step3 telemetry off → step4 setProcessor(nullptr) → step5 stopTimer
      → step6 saveSettings → step7 removeAudioCallback → （audioEngine 破棄）
      ~AudioEngine: StopAcceptingWork → bridge.requestShutdown → StopAudio(stopTimer)
        → StopWorkers: retryScheduler_.shutdown → shutdownCoordinatorLoop(join) → stopRebuildThread
        → active/fading slot 切離し → lifetimeMgr.retire(activeToRelease/fadingToRelease)
        → drainDeferredRetireQueues / drainAll / shutdownRuntime_.release → DSPCore destroy
```

**D119-BLOCKER-2 分類: probable race（teardown phase）— 崩壊箇所は現行計測では観測不能**:
- ロガー切断が crash 観測を完全にマスク（全 run の最終行が "Auto-exit flush" で同一になる理由）
- 実測頻度: plain 6s 2/5、IR+rebuild 12s 3/3 → **rebuild 活動量（=漏出オブジェクト量）と正相関**
- shutdown 経路は健全に設計されている（retire→drain→reclaim の固定順序）ため、クラッシュ候補は (a) 停止済み coordinator/router への後着 retire、(b) audio device close と DSPCore destroy の HB、(c) ログリングテアダウン、(d) 漏出 DSPCore の shutdown reclaim 競合

**診断設計（D122 実装・temporary diagnostic only）**: CLI モードでは FileLogger を process 終了まで保持（detach を最後に移動）+ `SHUTDOWN_BEGIN/SHUTDOWN_END` トレースを各 phase に注入。これにより crash 前最終 phase が確定する。

## D121 verdict — GO 条件 1〜12

| # | 条件 | 判定 |
| --- | --- | --- |
| 1 | DSPCore* ↔ DSPHandle identity 完全証明 | **PASS**（D121-A matrix・generation/stale/freelist 含む） |
| 2 | registration / activation / publication の分離 | **PASS**（3操作の定義と契約-実装ギャップの特定） |
| 3 | oldDSP/oldHandle 全状態の定義 | **PASS**（状態表・「oldDSP valid/oldHandle null = 復元可能状態」契約案） |
| 4 | crossfade completion の独立性 | **PASS**（寄生構造の実証 + 独立状態機械は未完成と特定） |
| 5 | crossfade overlap の扱い | **PASS**（claim fail→即時 retire の現行意味論 + 重複時の未定義領域を明記） |
| 6 | retire authority 一箇所 | **PASS**（R-A 確定・決定/実行の分離再証明） |
| 7 | Observe の authority 除外可能性 | **PASS**（分離可能・手順明記） |
| 8 | DELETE-1/2/3 境界維持 | **PASS** |
| 9 | shutdown race 調査手順確定 | **PASS**（teardown 順序 + 診断設計） |
| 10 | production source = 0 | **PASS** |
| 11 | Phase-II = 0 | **PASS** |
| 12 | commit freeze 継続 | **PASS** |

**D121: PASS（12/12）。** D122（最小修正仕様）は本監査の契約決定を入力として作成すること:
1. Execution tail で `DSPHandleRuntime::activate(newHandle)` を公開（lifetime.activate と区別）
2. oldDSP/oldHandle 不一致は map lookup で復元（復元可能状態として契約化）
3. crossfade 完了は方式 A（RT ramp 完了 → notifyFadeComplete 1回 push → Timer consume → retire decision は R-A）
4. Observe の retire 誘発を撤去（観測専用化）
5. shutdown: FileLogger 保持 + SHUTDOWN trace（診断専用）
6. snapshot fade トリガー除外は確定済み

## 生成物

- 本ファイル（`evidence/D121_LIFECYCLE_STATE_MACHINE_AUDIT.md`）
- production source 変更: **0**
