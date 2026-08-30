# D120 — Publish / Activation / Crossfade Lifecycle Contract Re-Audit

**Date:** 2026-08-29
**性質:** read-only 設計・コード監査（production source 変更 0）
**凍結:** commit・Phase-II・production fix 凍結継続（D119 NO-GO 状態維持）
**Frozen HEAD:** `a65ace1` + D117 観測トレース（macro-gated 2 ファイル）

---

## D120-1 — Publish family 完全列挙（G1）

`commitRuntimePublication` / `worldAuthority_.publish` / `executor_.publish` の全 caller:

| # | 経路 | 呼び出し箇所 | oldHandle | DSP transition（onPublishCompleted） | active handle 更新 |
| --- | --- | --- | --- | --- | --- |
| 1 | Bootstrap | Init.cpp:73（worldAuthority_.publish 直接） | —（直接 publish） | **なし** | **なし** |
| 2 | PrepareToPlay | PrepareToPlay.cpp:155, 277（commitRuntimePublication） | null（idle publish） | **なし**（直 commit 経路に tail なし） | なし |
| 3 | ReleaseResources | ReleaseResources.cpp:187 | null | **なし** | なし |
| 4 | Timer idle publish | Timer.cpp:994（publishIdleWorldOnly 系） | null 固定 | **なし** | なし |
| 5 | Transition / idle 統一関数 | AudioEngine.Transition.cpp:25（publishIdleWorldOnly） | null 固定 | **なし** | なし |
| 6 | Rebuild（Orchestrator） | Orchestrator.cpp:280 → PublicationExecutor.cpp:53/57 | **getActiveRuntimeDSPHandle() = 常に null**（D120-2） | あり — ただし **RuntimePublishExecutor.h テールは CoordinatorLoop の PublishIntentHandler で実行** | **なし（本件・D120-2）** |
| 7 | Shutdown publish | ReleaseResources 系 shutdown drain | — | なし | なし |

**結論（G1）**: publish family は 7 系統。**DSP lifecycle 遷移（onPublishCompleted）に到達するのは #6 のみ**であり、#1-#5/#7 の world は DSPCore retire 対象として一切扱われない。#6 に至っては oldHandle が常時 null のため分岐到達不能（D120-2）。

## D120-2 — Active Handle の authority（G2）

| 項目 | 実コードの事実 |
| --- | --- |
| `activeRuntimeDSPHandle_` の全 writer | ① `DSPHandleRuntime::activate()`（ISRDSPHandle.cpp:100）② `endCrossfade()`（:116）— **production からの呼び出しは endCrossfade のみで、endCrossfade 自体が GlobalSnapshot fade ゲート内（Timer.cpp:954）** |
| 契約上の Authority | AudioEngine.h:4486「**activeRuntimeDSPHandle_ の更新は commitRuntimePublication() が唯一の Authority**」および :4536「activate 責務は executePublish の Execution tail（DSPTransition::onPublishCompleted → lifetime.activate）に一本化」 |
| 実装の現実 | Execution tail の `lifetime.activate(newDSP)` = **`registerDSPHandleForRuntime`（map 登録のみ）** であり `DSPHandleRuntime::activate`（active handle 公開）を**呼ばない** → **契約と実装の不一致**。「activate」という語が 2 つの異なる操作（map 登録 / active handle 公開）に使われていることが混乱の核 |
| `activate()` の caller | production: **0**（unit test NormalRetireDSPHandleCompareTests のみ） |
| `registerDSPHandleForRuntime` の caller | Commit.cpp:804 / AudioEngine.h:4585（rollback）/ DSPLifetimeManager.cpp:27（= lifetime.activate）/ DSPTransition.h:92 |
| reader | Orchestrator.cpp:82（oldHandle 取得 = **常に null**）、ReleaseResources.cpp:444（shutdown 経路も null）、resolveDSPHandle 経由 |

**一意な時系列の証明（G2/G3）**:

```text
現行の実際の時系列（#6 Rebuild publish）:
  trySubmit: oldHandle = getActiveRuntimeDSPHandle() → 【常に null（writer なし）】
  → executor_.publish → enqueue（decision.oldHandle = null 固定）
  → CoordinatorLoop: executePublish → world swap → bridge（world retire）
  → Execution tail: onPublishCompleted(new, oldDSP=null, oldHandle=null, ...)
  → DSPTransition: lifetime.activate(new)（map 登録のみ）→ 両分岐 skip（oldDSP=null）
  ⇒ DSPCore retire 不発（D117 観測と完全一致）

契約が意図した時系列（AudioEngine.h:4486/4536 の記述）:
  trySubmit: oldHandle = getActiveRuntimeDSPHandle()（前回 publish で更新済みであるべき）
  → ... → Execution tail: lifetime.activate + 【activeRuntimeDSPHandle_ 公開】
  ⇒ 次回 publish の oldHandle が正しく得られる
```

**欠陥の本質**: 「lifetime.activate（map 登録）」と「DSPHandleRuntime::activate（active handle 公開 + fading handle null 化）」が別操作であり、契約コメントは後者を要求するが実装は前者しか実行しない。**誰も activeRuntimeDSPHandle_ を更新しない**。

## D120-3 — `DSPTransition::onPublishCompleted()` 状態表（G3）

前提: `newHandle = registerDSPHandleForRuntime(newDSP)`（常に取得可能・冪等）。

| oldDSP | oldHandle | needsCrossfade | claim | 現行の結果 | ownership 評価 |
| --- | --- | --- | --- | --- | --- |
| null | null | false | — | activate のみ・何も retire しない | **正当**（初回 publish。bootstrap からの遷移では旧 world の DSP 有無に注意） |
| valid | **null** | false | — | **両分岐とも skip → oldDSP は永遠に retire されない** | **漏出**（現行の常態。oldDSP なのに handle null の case） |
| valid | valid | false | — | 即時 retire（crossfadeRuntime_.complete() → lifetime.retire(oldDSP)） | 正当（DELETE-1 起点） |
| valid | valid | true | success | claim → storeReceipt → beginCrossfade → crossfadeRuntime_.start → **完了待ち**（完了は Timer の snapshot-fade ゲート内 — D120-4） | 条件付き正当（完了駆動が別問題） |
| valid | valid | true | **fail** | 即時 `lifetime.retire(oldDSP)` | **危険**（fading slot が前回分で占有されていた場合、現行 active DSP を即 retire する — crossfade 重複時の意味論が未定義） |
| valid | stale | true | — | oldHandle は resolve 済みで渡る（BUG-054 修正済み）が、resolve 失敗なら oldDSP=null と同様に skip | 漏出リスク |
| same DSP | same handle | — | — | 現行コードに同一性ガードなし | 遷移不要ケースの扱いが未規定 |

**必須の契約決定（D121 へ）**: 「oldDSP != nullptr かつ oldHandle == null」ケースでは、**oldDSP からの handle 復元（registerDSPHandleForRuntime の map lookup）を DSPTransition 内で行う**か、**oldHandle null を invariant violation として扱う**かを決めない限り、activate をどこに置いても漏出は消えない。なお graph.activeNode からの復元は bootstrap world の placeholder node を誤破棄した（D119 実測）ため不採用。

## D120-4 — 2 つの fade 機構の分離（G4/G5）

### A. Snapshot fade（SnapshotCoordinator）

```text
startFade(newSnap, fadeSamples)   ← 唯一の起点: AudioEngine.Snapshot.cpp:145（GlobalSnapshot 経路）
  ↓ advanceFade(numSamples)       ← AudioBlock.cpp:475（RT が毎 callback 減算）
  ↓ remaining == 0
tryCompleteFade()                 ← Timer.cpp:928（message thread）
  ↓ completeFade()（snapshot 入替え）
fadeCompleted = true
```

### B. DSP crossfade（CrossfadeRuntime + CrossfadeAuthorityRuntime）

```text
CrossfadeAuthority::evaluate → decision.needsCrossfade
  ↓ registerCrossfade（CrossfadeAuthorityRuntime）
  ↓ DSPHandleRuntime::beginCrossfade（from=CrossfadingOut, fadingRuntimeDSPHandle_=from）
  ↓ CrossfadeRuntime::start（RT gain smoothing 開始）
  ↓ RT: getGain().isSmoothing() の間 canCrossfade ミキシング（AudioBlock.cpp:384）
  ↓ 【完了検出なし — RT は mixing を終めるだけ。イベントを発行しない】
notifyFadeComplete(id)            ← Timer.cpp:950 — 唯一の呼び出し元が【A の fadeCompleted ブロック内】
  ↓ consumeCompletedFade → endCrossfade → fading slot CAS → retire（submitObserve）
```

**G4/G5 の証明**:
- **B の完了は A の完了に寄生している**。`notifyFadeComplete` の production caller は Timer.cpp:950 のみで、それは `if (m_coordinator.tryCompleteFade())` — すなわち GlobalSnapshot fade が完了した tick でしか実行されない。世界 publish（IR 切替・rebuild）では `startFade` が呼ばれないため **A は決して完了せず、B も決して完了しない**。
- RT 側は gain smoothing が終わっても完了イベントを発行しない（AudioBlock 384-442 に完了検出なし）。
- よって **「crossfade を start した DSPCore は、GlobalSnapshot fade が偶発するまで永久に fading のまま」** — D119 発見②（activate 配線後の再構築ループ）の構造的原因。crossfade 完了を待つ逐次 retire は設計上成立しない。

## D120-5 — 経路到達性分類（G8）

| API / 経路 | 分類 | 根拠 |
| --- | --- | --- |
| `DSPHandleRuntime::activate()` | **Dead（production caller 0）** | ISRDSPHandle.cpp:93、unit test のみ |
| `activeRuntimeDSPHandle_` の意味ある read | **Unreachable under current production flow** | writer が実質欠落 → Orchestrator/ReleaseResources の read は常に null |
| `DSPTransition::onPublishCompleted` の retire/crossfade 分岐 | **Unreachable（現行 flow）** | oldDSP 常に null（#6 経路）+ 他 family はそもそも呼ばない |
| `claimFadingRuntimeDSP` / `storeReceipt` | **Dormant** | 分岐内だが分岐自体が到達不能 |
| `retirePublishedDSP` | **Dead（direct/indirect caller 0 — D118-7 再確認）** | かつ接続先の fadeCompleted も到達不能（下記） |
| Timer fadeCompleted ブロック（endCrossfade / fading slot CAS / notifyFadeComplete / consumeCompletedFade / publishIdleWorldOnly） | **Unreachable under world-publish flow**（GlobalSnapshot fade 時のみ Reachable） | m_fade.start の唯一起点が Snapshot.cpp:145 |
| `tryCompleteFade` / `advanceFade` | Reachable（advance は毎 RT callback） | ただし FadingIn 状態でなければ tryComplete は false |
| `submitObserve`（Timer 3 箇所 + DSPTransition dead 1 箇所） | **Unreachable（現行 flow）** — 3 箇所とも fadeCompleted/CAS ゲート内 | 仮に到達しても handle null で skip |
| `retireByHandle` / Observe→Coordinator | **Dormant**（intent が飛んでこない） | D117 トレース 0 発火 |
| `DSPLifetimeManager::retire(DSPCore*)` | Reachable — ただし現状呼び出しは shutdown/direct-retire 経路のみ（RebuildDispatch.cpp:775 orphan 経路等） | 定常運転の publish からは到達しない |
| `destroyDSPCoreNode`（enqueue→drain） | Reachable（shutdown 経由で確認・D117 の tF=0 は定常運転で 0 発火の意） | |

**総括**: DSPCore lifecycle の設計経路は、現行 production flow において **activate→…→retire の全体が休眠**。稼働しているのは world level（worldAuthority publish/retire）のみで、DSPCore は world から参照され続けたまま破棄されない。

## D120-6 — Retire Authority の再決定（G6）

| 候補 | 評価 |
| --- | --- |
| R-A: DSPTransition → DSPLifetimeManager::retire(DSPCore*) | **推奨**。oldDSP identity が確定する唯一の一点（Execution tail）であり、immediate-retire / crossfade 分岐の双方がここに収束する。ポインタ一次で resolve 検証可能（D118 DESIGN-B の identity 設計を踏襲可能） |
| R-B: Crossfade completion → Timer → DSPLifetimeManager | **不採用** — completion が寄生構造（D120-4）であり、独立駆動の設計変更が前提になる。R-A を採用すれば crossfade 完了待ちは「retire を遅らせる手段（fading slot）」として残せるが、retire 決定は R-A で行う |
| R-C: Observe → Coordinator → retireByHandle | **retire authority としない**（Observer は副作用を持たない原則）。stale intent 破棄の意味論問題（D118 追加発見）もあり、Observe は観測専用に戻す |
| R-D: 複数経路 + dedup | 不採用（authority singularization 違反） |

**決定事項の分離**:
- **retire を「決定」するのは 1 者**: Execution tail の DSPTransition（publish commit の直後 — oldDSP identity が確定する唯一の時点）
- **物理 destroy を「実行」するのは別**: RetireRouter drain（enqueueWithRetry → destroyDSPCoreNode、epoch 安全確認後）— DELETE-1/2/3 境界は現行維持
- crossfade の有無は「retire を即時に行うか、fading slot で保留するか」の実行順序の問題であり、retire 決定 authority を複数化しない。保留中 DSPCore の完了駆動は D121 の設計課題（Timer による fading slot の定期監査など、snapshot fade に寄生しない駆動）

## D120-7 — Shutdown race（G9・D119-BLOCKER-2）

**実測（frozen production binary・6s/12s CLI runs）**:
- plain 6s × 5: exit 139, 139, 0, 0, 0（2/5 crash）
- IR+rebuild 12s × 3: exit 139, 139, 139（3/3 crash）

**測定方法の重要な訂正**: D116 で「restart 6/6 exit 0」と記録した測定は `cmd /c "... & echo %ERRORLEVEL%"` を使用しており、**%ERRORLEVEL% はコマンドライン解析時に展開されるため常に事前値（0）を表示していた**。すなわち D116 の shutdown exit code 記録は無効で、**間欠クラッシュは D119 以前から存在した可能性が高い**（D116 実施中に bash が報告した 2 回の "Segmentation fault" も実クラッシュだった可能性が高い — 当時「アーティファクト」と誤判定）。icx 0xc0000005 ×3 も同族の可能性が高い。

**分類: probable race（teardown phase）** — 証拠:
- クラッシュは "Auto-exit flush: shutting down" **後**に発生。この直後 `juce::Logger::setCurrentLogger(nullptr)` で FileLogger が切断されるため、**ログでは崩壊箇所を特定できない**（全 run の最終行が同一になる理由）
- IR+rebuild ありの方が頻度が高い → teardown 時の DSPCore/NUC/queue の残存物量（＝漏出量）と正相関
- WER Event 1000 が記録されないケースが多い（icx 時は記録されていた — ビルド差分）
- 仮説候補（未検証）: (a) shutdown 中の RetireRouter drain / quarantine と coordinator loop 停止の順序、(b) asyncSink ログリングとテアダウン、(c) audio device close と DSPCore destroy の HB、(d) 破棄漏れ DSPCore（本件漏出）の shutdown 経路での reclaim 競合

**原因特定は次段階（D121 の diagnostic 計画）へ**。ロガー切断前に shutdown 完了トレースを残す改修（診断専用）を推奨。

## D120-9 — テスト（frozen binary の挙動再現）

既存データでカバー: T1/T2/T3（D117 10-pair, D119 gate3 — MEM_SNAP 時系列あり）、T4/T6（burst + IR swap — D116-5 D2/D3）、T7/T8（shutdown — 本監査で 8 run 追加・exit code 測定修正済み）。新規 trace は未追加（D120-8 のイベントセットは D121 実装時の検証用として設計済み — 下記）。

D121 用推奨トレースセット（temporary diagnostic only・macro-gated）:
```text
[PUBLISH] seq=N newDSP=X activeBefore=Ho oldDSP=Xo
ACTIVE handle=Hn        （activeRuntimeDSPHandle_ 公開点）
XFADE_START id / XFADE_COMPLETE id
RETIRE dsp= / RECLAIM handle= / DESTROY dsp=
SHUTDOWN_BEGIN / SHUTDOWN_END
```

## GO/NO-GO（G1-G12）

| Gate | 判定 |
| --- | --- |
| G1 publish caller 列挙 | **PASS**（7 系統 + lifecycle 到達表） |
| G2 active handle authority | **PASS** — 契約（commitRuntimePublication が唯一 Authority）と実装（誰も更新しない）の不一致を line-level で確定。「lifetime.activate ≠ DSPHandleRuntime.activate」の用語混同を特定 |
| G3 oldDSP ↔ oldHandle | **PASS** — 状態表で「oldDSP valid / oldHandle null → 漏出」を確定。復元は DSPTransition 内 map lookup（graph.activeNode は不採用） |
| G4 DSP crossfade completion | **PASS** — 完了駆動は存在しない（notifyFadeComplete 唯一 caller が snapshot-fade ゲート内） |
| G5 2 fade 機構の非同一性 | **PASS** — A（Snapshot）と B（DSP crossfade）の分離図と寄生関係を実証 |
| G6 retire authority | **PASS** — R-A（Execution tail の DSPTransition）を推奨。決定者と実行者を分離 |
| G7 DELETE-1/2/3 | **PASS** — 現行境界維持を確認（retire → epoch/reclaim → drain destroy） |
| G8 dead/dormant 分類 | **PASS**（D120-5 表） |
| G9 shutdown race | **PASS** — probable race + 測定アーティファクト訂正 + ログ遮断問題の特定 |
| G10 D119 暴走の因果モデル | **PASS** — 「crossfade 完了不能 × activate 配線」で再構築ループが生じる構造を D120-4 で説明（実行時再現は D121 診断トレースで確認） |
| G11 Phase-II = 0 | **PASS** |
| G12 production fix = 0 | **PASS**（working tree は D117 trace のみ） |

## 結論

**D120: PASS（12/12）。** lifecycle の真の構造は:

```text
publish（7 family）
  ↓ world swap（world level retire のみ稼働）
  ↓ 【active handle 公開 — 契約上は commitRuntimePublication Authority、実装は存在しない】
  ↓ oldHandle = 常に null
  ↓ DSPTransition 両分岐到達不能
  ↓ 【DSP crossfade 完了駆動 — 存在しない（snapshot fade 寄生）】
  ⇒ DSPCore は定常運転では誰にも retire されない
```

**D121（修正設計）への持ち込み事項**:
1. `DSPHandleRuntime::activate(newHandle)` を Execution tail に配線する（lifetime.activate と区別して「active handle 公開」を明示）
2. oldDSP 指定がない family（idle/prepare/release）の DSPCore lifetime 方針（現状: 触れない → world 経由で参照され続ける）
3. crossfade 完了の独立駆動（snapshot fade 寄生の解消）— R-A 選択時は fading slot 保留の完了監視が必要
4. Observe の stakeholder から外す（観測専用化）+ stale epoch 意味論の明確化
5. shutdown 間欠クラッシュの原因特定（logger 切断前の shutdown trace + teardown 順序監査）
6. `retirePublishedDSP` は R-A 採用時、そのまま fade-completion 用サブルーチン（receipt epoch 伝搬）として再利用可／あるいは削除

## 生成物

- 本ファイル（`evidence/D120_LIFECYCLE_AUDIT.md`）
- `evidence/D120_shutdown_probe*.log` / `D120_shutdown_ir_probe*.log`（shutdown 頻度実測 8 run）
- production source 変更: **0**
