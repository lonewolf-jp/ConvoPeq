# D122 — Lifecycle Repair Contract / Minimal Patch Specification Audit

**Date:** 2026-08-29
**性質:** read-only 設計監査・契約固定。**production source 変更 0**（診断専用 macro-gated instrumentation の仕様のみ定義）
**凍結:** commit・Phase-II・production fix 凍結継続。**D123 実装は本監査 12/12 PASS 後**
**入力:** D117（原因）→ D118（初回設計・前提誤り）→ D119（実装 NO-GO・上流欠陥発見）→ D120（lifecycle 全経路）→ D121（state-machine 再証明）

---

## 1. Current facts（確定済み事実の再掲）

| # | 事実 | 出典 |
| --- | --- | --- |
| F1 | `activeRuntimeDSPHandle_` の production writer が存在しない（`DSPHandleRuntime::activate` caller 0）→ Orchestrator の oldHandle 常に null | D120-2 |
| F2 | `lifetime.activate` = map 登録のみ。`DSPHandleRuntime::activate` = active 公開 + **fading null 化の副作用**。2 操作が「activate」として混同 | D121-2 |
| F3 | DSP crossfade の完了駆動は存在しない（`notifyFadeComplete` 唯一 caller が snapshot-fade ゲート内 = 寄生） | D121-C |
| F4 | RT は LinearRamp で ramp を進め、`remaining<=0` で収束するが完了イベントを発行しない | DspNumericPolicy.h:387-396 |
| F5 | family #2-#5（idle publish）は oldHandle **null 固定が設計意図**（同一 DSP 再 publish に retire 意図なし） | 各箇所のコメント |
| F6 | retire decision の唯一の確定点 = Execution tail（DSPTransition） | D120-6/D121-D |
| F7 | CLI ログは "Auto-exit flush" 直後に切断され、teardown が観測不能 | D120-7 |
| F8 | shutdown 間欠 exit 139: plain 2/5、IR+rebuild 3/3（frozen binary 実測） | D120-7 |
| F9 | `isSlotInCrossfade` は **crossfadeRecords_** を参照する（fading handle atomic に非依存） | ISRDSPHandle.cpp:185-194 |

## 2. Required state transitions（修正後の正規 lifecycle）

```text
publish commit 成功（Execution tail, CoordinatorLoop）
  ↓ (1) lifetime.activate(newDSP)                    = map 登録（冪等・既存）
  ↓ (2) DSPHandleRuntime::activate(newHandle)        = active handle 公開【新設配線】
  ↓ (3) oldDSP 確定
  │     ├─ oldHandle 非 null → resolve（gen 検証）
  │     └─ oldHandle null かつ oldDSP 非対象 → retire なし（idle family）
  ↓ (4) needsCrossfade:
  │     ├─ false → 即時 retire decision（DELETE-1）
  │     └─ true  → claim → storeReceipt → beginCrossfade → crossfadeRuntime_.start
  ↓ (5)【新設】RT ramp 完了 → notifyFadeComplete(id) 1回
  ↓ (6) Timer: consumeCompletedFade → endCrossfade → fading slot CAS → retire decision（DELETE-1）
  ↓ (7) DELETE-2: requestReclaimHandle（epoch 安全→slot reclaim / 不安全→pendingReclaimHandles_）
  ↓ (8) DELETE-3: enqueueWithRetry(destroyDSPCoreNode) → drain が物理削除
```

## 3. D122-A — Active-handle publication 契約（G1）

**呼び出し位置**: `DSPTransition::onPublishCompleted` 内、`lifetime.activate(newDSP)` の直後（normal path / emergency path の両方）。**Execution tail から onPublishCompleted が呼ばれるのは commit 成功時のみである**ことを実コードで確認済み（publish 失敗時は Orchestrator が `executor_.publish FAILED` を検出して `destroyRolledBackDSP` へ分岐し tail に到達しない。rollback は facade の ScopeExit（AudioEngine.h:4585）で enqueue 失敗時に完結し tail 前に終了 → **activate が rollback 後に呼ばれる経路は存在しない**）。

**newHandle 対応**: `newHandle = registerDSPHandleForRuntime(newDSP)` — map 既存 lookup（冪等）であり newDSP と 1:1。newDSP==nullptr（idle family で RegistrationContext により未登録の場合）は activate を skip。

**fading null 化副作用の安全性審査（最重要）**:

| 項目 | 評価 |
| --- | --- |
| 副作用内容 | `fadingRuntimeDSPHandle_ = null`（ISRDSPHandle.cpp:99） |
| 影響 reader | ① Timer 3 箇所の retire 提出ブロック（**D123 で retirePublishedDSP 接続に置換され handle 非依存化**）② `getFadingRuntimeDSPHandle`（DSPTransition claim 分岐の storeReceipt — activate は claim **より前に**実行され、beginCrossfade が**後から** fading handle を再設定するため順序健全）③ `ISRDSPHandle.cpp:221-224` の検証関数（active slot + fading handle + isSlotInCrossfade の複合照合 — 診断用途） |
| `isSlotInCrossfade` への影響 | **なし** — crossfadeRecords_ 参照であり fading handle atomic に非依存（F9） |
| endCrossfade の to→Active との競合 | なし — activate(新)→beginCrossfade(old,new) の順で to は CrossfadingIn に遷移し、完了時 endCrossfade が to→Active に戻す（現行設計の状態遷移を維持） |
| **D119 暴走との関係** | 暴走の原因は activate 単体ではなく「**completion 不発** × activate」の組合せ（D120-4）。本契約では D122-C により completion を独立駆動するため、fading null 化は「完了済み crossfade の state 掃除」として安全側に働く |

**結論**: `DSPHandleRuntime::activate()` を**そのまま使用**（新 API 分離は不要）。ただし呼び出し契約を明文化: 「publish commit 成功後・claim/beginCrossfade より前・newDSP registered 済みであること」。

## 4. D122-B — oldDSP valid / oldHandle null の復元契約（G2）

**採用: 案 A（map lookup 復元）— ただし適用範囲を Rebuild family に限定**。

line-level 契約:

```text
Execution tail（Rebuild family のみ到達）:
  oldDSP = resolve(decision.oldHandle)          ← D122-A 配線後は非 null が常態
  if (oldDSP == nullptr)                         ← decision.oldHandle null（残留ケース）
      → onPublishCompleted に oldDSP=null を渡す（= retire 対象なし。idle 意図と整合）
  ※ graph.activeNode からの復元は行わない（bootstrap placeholder を誤破棄 — D119 実測）
```

確認項目の検証結果:

| 項目 | 検証 |
| --- | --- |
| map lookup が ownership を生成しない | ✅ `registerDSPHandleForRuntime` は既存 entry を返すのみ。未登録の場合のみ create するが、それは「当該 DSPCore の初回登録」であり ownership は引き続き DSPLifetimeManager 管轄 |
| stale handle を生成しない | ✅ create は常に gen+1 の新鮮な handle を返す（stale 生成の概念なし）。resolve 側の gen 検証で誤参照防止 |
| erase 済みの場合の意味論 | find 失敗 → create（新 handle）→ 直後に retire で erase — **二重 retire 構造的に不可能**（map erase 済みなら `retireDSPHandleForRuntime` が false → DSPLifetimeManager::retire は静かに return） |
| bootstrap / PrepareToPlay / idle | oldHandle **null 固定**が設計意図（F5）→ 復元を行わない。同一 DSP 再 publish で retire が誤発しないよう **family による oldHandle 規約を維持** |
| graph.activeNode fallback | **禁止**（D119 実測: bootstrap placeholder 誤破棄 → crash） |

**同一 DSP 再 publish の保護**: oldHandle が non-null でかつ `oldDSP == newDSP` の場合は retire しない（onPublishCompleted 冒頭で同一性チェック — D122 契約として明記。現行コードに同一性ガードがないため D123 実装項目）。

## 5. D122-C — Crossfade completion 独立駆動契約（G3/G4/G5）

### RT 側（AudioBlock.cpp、crossfade ミキシング後）

```text
LinearRamp: remaining は getNextValue()/skip() 内でのみ減算（RT 専用・ASSERT_AUDIO_THREAD）
  remaining が 1→0 に遷移する block が「ramp 完了 block」（getNextValue: current=target 収束、DspNumericPolicy.h:387-396）
契約: RT は block 末尾で
  wasSmoothing（block 開始時 isSmoothing）== true && 現在 isSmoothing()== false
  を検出した場合、**直近の active CrossfadeId** に対して notifyFadeComplete(id) を 1 回 push
```

| 項目 | 証明 |
| --- | --- |
| 1回だけ発行 | `remaining` の 1→0 遷移は ramp 生存中に正確に 1 回（`--remaining <= 0` 収束）。検出は「直前 block まで smoothing / 今 block で非 smoothing」のエッジ検出 → exactly-once。arm 再実行（applyImmediateValueRT）で remaining が復活した場合は新 ramp として再カウント（generation で区別） |
| allocation = 0 | CompletedFadeEvent は trivially copyable、SPSCRingBuffer 固定容量 32（CrossfadeRuntime.h:213）— zero alloc |
| lock = 0 | SPSC push は lock-free（static_assert 済） |
| blocking = 0 | push 失敗時は drop + drop count（既存実装） |
| queue full 時 ownership | **イベント損失 = retire 遅延のみ**（DSPCore は fading 保留のまま。drop count は HealthMonitor 監視済み）。Timer が毎 tick 複数 consume するため実質 full にならない |
| producer/consumer ownership | producer = RT（1 writer）、consumer = Timer（1 reader）— SPSC 契約成立 |

### NonRT 側（Timer）

```text
Timer tick（m_fade ゲートから引き出す — D123 で fadeCompleted とは独立の consume ブロックに変更）:
  while (consumeCompletedFade(ev)):
      CrossfadeAuthorityRuntime で ev.id の active 検証（generation/identity）
      有効 → endCrossfade(ev.id)（from→Retired, to→Active, activeHandle=to, fading=null）
           → fading slot CAS → retirePublishedDSP(current, ...)（D118-B の receipt epoch 伝搬を再利用）
      無効/stale → ログ + 破棄（retire しない）
```

**最重要契約**: **「ramp が終了した」ことは「DSP を retire してよい」ことを意味しない**。retire は (a) endCrossfade による slot Retired 遷移、(b) fading slot CAS による identity 取得、(c) DELETE-1〜3 の順序、をすべて満たした後の **retire decision（R-A: DSPTransition/Timer の DSPLifetimeManager::retire）** によってのみ行う。ramp 完了イベントはその**トリガー**に過ぎない。

| 異常系 | 扱い |
| --- | --- |
| duplicate completion | 同一 CrossfadeId の 2 回目は authority registry に存在しない → 破棄（endCrossfade も records 走査で no-op） |
| stale completion | crossfadeId が現行世代と不一致 → 破棄 |
| shutdown 中の completion 到着 | isShutdownInProgress() → retire は shutdown drain 経路（drainAllQuarantineStore/destroyForShutdown）に委譲し Timer 側では破棄 |
| crossfade abort（Emergency Override / crossfadeRuntime_.complete()） | abort 時に active crossfade record を unregister（現行 emergency path を監査 — D123 で complete() 呼び出しと records 整合を確認） |

## 6. D122-D — Retire authority ownership chain（G4/G10）

D121-D の chain を**変更なし**で最終固定。検証済み:

- `requestReclaim` は物理削除を行わない（REPAIR_PLAN.md 明記・ISRDSPHandle.cpp:129-147 は slot 状態遷移+freelist のみ）
- epoch unsafe な reclaim は `pendingReclaimHandles_` 保留 → `drainDeferredRetireQueues` 再試行（REPAIR_PLAN2-dash2 設計どおり）
- `destroyDSPCoreNode` の唯一の実行経路 = RetireRouter drain（.enqueueWithRetry の deleter）
- **Timer / DSPTransition から直接 delete / aligned_free を行わない**（D119 禁止リスト維持）

## 7. D122-E — Observe の authority 完全除外（G6）

| 経路 | 現状 | D123 処置 |
| --- | --- | --- |
| Timer 3 箇所 submitObserve（fadeCompleted ゲート内） | retire 誘発意図だが実際は不発（null handle） | **retirePublishedDSP 接続に置換**（Observe 呼び出しを削除） |
| DSPTransition.h:156（onTransitionComplete） | dead code（caller 0） | D123 では触れない（将来削除候補として記録） |
| ObserveIntentHandler → retireByHandle | stale intent を破棄し、有効 intent を retire 誘発 | **retireByHandle 呼び出しを削除し telemetry/metrics 専用化** |
| drainObserveDeferred → retireByHandle | 同上 | 同上 |

**削除しても ownership chain に穴が開かないことの証明**: Observe 経路が現行で retire を実行した実績は 0（D117 トレース）。D122-C/D の新経路が全 retire を引き受けるため、Observe 撤去で喪失する機能はない。`retireByHandle` 自体は Observe 撤去後 caller 0 となる（dead API 化 — 削除は D124 以降に記録）。

## 8. D122-F — Publish-family ownership matrix（G7）

| family | newDSP | oldHandle | oldDSP 成り得るか | lifetime ownership の引き受け手 |
| --- | --- | --- | --- | --- |
| Bootstrap（#1） | bootstrap DSP | —（直接 publish） | — | 初期 current。**最初の Rebuild publish で oldDSP になり retire される**（D122-A/B 経路） |
| PrepareToPlay（#2/#3） | 同一 DSP 再 publish | **null 固定** | old==new（retire 意図なし） | 変更なし（DSPCore は現行 owner のまま） |
| ReleaseResources（#4） | 登録なし/同一 | null 固定 | なし | 同上 |
| Timer idle（#5）/ Transition idle（#6） | 同一（currentAfterFade） | null 固定 | なし | 同上 |
| **Rebuild（#7）** | **新規 DSPCore** | **active handle（D122-A 配線後は非 null）** | **はい** | **Execution tail が retire decision（R-A）** |
| Shutdown | — | — | — | destroyForShutdown / drainAllQuarantineStore（既存・D120-7） |

**World publish ≠ DSP publish の維持**: World lifetime は `RuntimeWorldAuthority` / bridge（world retire queue）が管轄し、DSPCore lifetime は `DSPHandleRuntime` + `DSPLifetimeManager` + RetireRouter が管轄 — 両者は既存どおり分離（I4: World は m_retireRouter の別エントリ）。本仕様は DSPCore 側のみを修復し World 側は触れない。

## 9. D122-G — Shutdown diagnostic contract（G9）

**目的**: crash 前最終 teardown phase の特定（production fix ではない・診断専用 macro-gated）。

| イベント | 注入点 |
| --- | --- |
| SHUTDOWN_BEGIN | MainApplication::shutdown() 冒頭 |
| STOP_ACCEPTING_WORK / STOP_AUDIO / STOP_WORKERS / STOP_REBUILD | ~AudioEngine の各 setShutdownPhase 呼び出し後 |
| RETIRE_DRAIN / DSP_DESTROY / WORLD_DRAIN | ~AudioEngine の retire/drain 各ステップ後 |
| LOGGER_DETACH | `setCurrentLogger(nullptr)` の**直前**（MainApplication::shutdown と Auto-exit の 2 箇所） |
| SHUTDOWN_END | MainApplication::shutdown 末尾 |

**必須要件**: CLI モードでは `setCurrentLogger(nullptr)` を**削除せず**、detach 前に別経由（OutputDebugString + `SHUTDOWN_END` 到達フラグ、または shutdown 専用セカンダリ FileLogger を detach 後に再接続）で `SHUTDOWN_END` が**logger detach 後にも観測できる**ようにする。これにより「crash した場合、最後の phase イベント = crash phase」が確定する。FileLogger 保持は shutdown authority 契約とは無関係の**観測専用変更**として扱う（I4/REPAIR_PLAN の shutdown ownership は変更しない）。

## 10. D122-H — D119 暴走防止 invariant（G8）

| Invariant | 定義 | 実装時の担保手段 |
| --- | --- | --- |
| INV-XFADE-1 | crossfade start → 必ず completion または explicit abort に到達する | D122-C の RT 完了検出（gain ramp は必ず収束する）+ Emergency abort での records unregister |
| INV-XFADE-2 | 1 CrossfadeId → completion ≤ 1 | authority registry からの消費で冪等化・重複イベントは破棄 |
| INV-XFADE-3 | completed crossfade → fading slot は最終的に解放される | consume → endCrossfade → slot CAS → retire の chain（Timer が毎 tick consume するため liveness 確保） |
| INV-XFADE-4 | fading slot 占有が rebuild feedback loop を生成しない | claim fail 時の即時 retire（現行 DSPTransition 分岐）+ completion 駆動により slot 占有が一時化 |
| INV-XFADE-5 | crossfade completion は SnapshotCoordinator に依存しない | D122-C の独立駆動（notifyFadeComplete 呼び出しを snapshot ゲートから移動） |
| INV-ACT-1 | `DSPHandleRuntime::activate` は commit 成功後・onPublishCompleted 内でのみ呼ぶ | 呼び出し位置 1 箇所（DSPTransition）に限定 |
| INV-ACT-2 | idle family（oldHandle null 固定）では retire を発火しない | family 規約維持（F5）+ onPublishCompleted の同一 DSP ガード |

## 11. Exact source files / functions to change（D123 実装範囲）

| ファイル | 変更 |
| --- | --- |
| `src/audioengine/DSPTransition.h` | ① `lifetime.activate` 後に `DSPHandleRuntime::activate(registerDSPHandleForRuntime(newDSP))`（normal/emergency 両分岐）② same-DSP ガード ③ emergency path の records 整合確認 |
| `src/audioengine/AudioEngine.Timer.cpp` | fadeCompleted ブロックの submitObserve → consume-完成crossfade→endCrossfade→retirePublishedDSP の再構成（m_fade ゲートから completion consume を分離）+ RT 完了検出は AudioBlock 側 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ramp 完了エッジ検出 → notifyFadeComplete 1 回 push（macro-gated ではなく production 機能） |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | ObserveIntentHandler / drainObserveDeferred から retireByHandle 呼び出し撤去（telemetry 化） |
| 診断専用（macro-gated） | shutdown trace（D122-G）、既存 D117 trace 維持 |

## 12. Files explicitly forbidden to change

`ISRRetireRouter.{h,cpp}` / `RetireQuarantineStore.h` / `EpochDomain` / `RuntimeWorldAuthority` / `DSPLifetimeManager.cpp` の意味論 / `FrozenRuntimeWorld` / Recovery・Phase-II 関連 / queue capacity / shutdown ownership mechanism / `icx` 関連

## 13. Test / acceptance gates（D123 用）

| Gate | 条件 |
| --- | --- |
| G1 | Compile（DIAG ON / production OFF 両方・**独立 build-diag ディレクトリ使用**） |
| G2 | CTest（test21 ×2 → full 40/40、両 config） |
| G3 | 6-publish smoke: `[D117_RETIRE]`/`[D117_DESTROY]` > 0、DC/SC/NUC live bounded、`[D119_TAIL]` oldHandleNull=0 |
| G4 | 10-publish: live count bounded・Priv 単調増加なし |
| G5 | burst-only ×10: retention なし |
| G6 | IR swap + rebuild: retention なし・XFADE_COMPLETE イベント確認 |
| G7 | ≥180s long-run: memory slope ≈ 0 |
| G8 | shutdown ×8: exit 139 頻度が frozen baseline（2/5, 3/3）と同等以下 + SHUTDOWN trace で crash phase 特定 |
| G9 | identity-preserving chain: 同一 DSPCore* の register→publish→fade CAS→retire→destroy を log で対合 |

## 14. Rollback conditions

- Gate 3 時点で DC live が publish 数に比例増加 → 即時中止・revert（D119 手順: git checkout 3-4 ファイル → 両 config 再ビルド → CTest）
- Gate 6/7 で memory slope > 0 → 中止・D122 契約の再監査
- shutdown 頻度が frozen baseline を超過 → shutdown 診断サブトラック分離
- 新規 invariant 違反（double destroy / UAF / quarantine 重複）→ 即時 NO-GO

## 15. GO / NO-GO

| Gate | 判定 |
| --- | --- |
| D122-G1 active publication API と副作用の確定 | **PASS**（activate そのまま使用可・副作用審査済み・呼び出し位置1箇所） |
| D122-G2 oldDSP/oldHandle mismatch 処理 | **PASS**（案 A 復元 + family 規約 + 同一DSP ガード） |
| D122-G3 completion 独立駆動の line-level 確定 | **PASS**（RT remaining 1→0 エッジ検出・AudioBlock 挿入点確定） |
| D122-G4 completion → retire identity chain | **PASS**（id 検証 → endCrossfade → slot CAS → retirePublishedDSP） |
| D122-G5 duplicate/stale semantics | **PASS** |
| D122-G6 Observe 非 authority | **PASS**（撤去しても chain に穴なし） |
| D122-G7 publish family ownership | **PASS**（8 family matrix） |
| D122-G8 runaway invariant | **PASS**（INV-XFADE-1〜5 + INV-ACT-1/2） |
| D122-G9 shutdown diagnostic 仕様 | **PASS** |
| D122-G10 DELETE-1/2/3 境界不変 | **PASS** |
| D122-G11 Phase-II = 0 | **PASS** |
| D122-G12 production source change = 0 | **PASS**（本監査での変更 0） |

**D122: PASS（12/12）→ D123（最小実装＋Gate 1 Compile）への進行を GO。**

主要な NO-GO 条件（activate 副作用の分離不能 / completion exactly-once 証明不能）は **いずれも成立しないことを実コードで確認**（fading null 化は isSlotInCrossfade 非依存・remaining 1→0 エッジは数学的に exactly-once）。

## 生成物

- 本ファイル（`evidence/D122_REPAIR_CONTRACT_AUDIT.md`）
- production source 変更: **0**
