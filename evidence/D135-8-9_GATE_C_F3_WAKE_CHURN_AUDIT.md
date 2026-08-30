# D135-8/9 Gate C-F3 — Deferred Retention Wake/Churn Design Audit

Date: 2026-08-30
Scope: **production source 0変更・test source 0変更・実装 0・テスト変更 0**。最新 working tree（= ConvoPeq.md 2026-08-30 版と同一ソース）からのコード追跡のみ。F1/F2（evidence/D135-8-9_GATE_C_F1_FORENSIC_AUDIT.md / F2_RETRY_CONTRACT_AUDIT.md）を入力とする。

## Verdict（要点）

> **F2 の「30s TTL が Option A の safety bound」は実コード上 偽**。`enqueueDeferred` は再 enqueue ごとに
> `enqueueTimestampUs = now` をリセットする（Orch.cpp:445,517,520）ため、連続 re-drive 下では
> `evaluateDeferred` の `ageUs ≈ 1ms` が永遠に続き **TTL は発火しない**。
> したがって **Option A 単独では、kMax cap が止めていた churn が無制限化**する
> （D133-1 の 28/19-iteration busy-loop の再発 — enum コメントが「唯一の ownership-release path」と呼ぶもの）。
> **推奨 = Option A + event-driven wake（fade-complete 通知）+ obligation-keyed TTL + watchdog 間隔化**（§11）。
> fade 完了時に deferred を起こす既存機構は**存在しない**（§6 で全 producer 網羅確認済み）。

---

## 1. 現行 wake chain（実測・全 producer 網羅）

`publishRetryReady` の全設定箇所（grep 網羅、AudioEngine.h:2718 宣言）:

| # | 箇所 | 主体 | 条件 |
| --- | --- | --- | --- |
| P1 | **Threading.cpp:280-289**（`runCoordinatorPhase`） | CoordinatorLoop（1ms tick、NonRT worker） | `!shutdown && hasDeferredRequest()` → lock(rebuildMutex) + `publishRetryReady=true` + `rebuildCV.notify_one()` — **毎 tick 無条件再通知** |
| P2 | **Timer.cpp:1744-1750**（crossfade timeout recovery） | MessageThread | recovery 発火時のみ |
| （消去） | CtorDtor:182 / PrepareToPlay:82 / ReleaseResources:175 / RebuildDispatch:866,890 | — | 各 lifecycle で false クリア |

消費側（RebuildDispatch.cpp:854-890）: predicate `hasPendingTask || publishRetryReady || recoveryPending || shouldExit` →
`wasRecoveryWake = exchangeAtomic(recoveryRetryReady)` → `doDeferredPublish = publishRetryReady; publishRetryReady = false`。

**churn loop 全周（1ms ごと）**: P1 → wake → `drainDeferredClearIfRequested` → `processDeferredAdmission(false)`
→ `peekDeferred` → `evaluateDeferred`（**fading を見ない**ので Ready）→ `consume()`（move-out + `finishView`:
slot reset・hasDeferred_=false・DeferredHealth 記録）→ `submitPublishRequest` → 主 admission
→ **DeferredFadingActive** → `enqueueDeferred`（count++、slot struct-replace、hasDeferred_=true、DeferredHealth 記録）
→ 次 tick で P1 発火。以遠無限（fade 継続中）。

## 2. 30s retention 時の最大 re-drive 回数

**無制限（TTL 失効のため）。** 根拠:
- `evaluateDeferred` の TTL 判定は `ageUs = ctx.nowUs - m.enqueueTimestampUs`（PA.cpp:75-77）。
- `m.enqueueTimestampUs` は **enqueue ごとに `now` で再設定**（Orch.cpp:445 → :517/:520）。
- re-drive 間隔 ≈ 1ms → `ageUs ≈ 1ms ≪ 30s` → **TTL 分岐は永遠に非発火**。
- 理論上限: fade が解消されず recovery も入らなければ **∞**（tick 数にして 30s で ~30,000、以上も可）。
- 付随: `maxDeferredAgeMs_`（Orch:448-458）は「上書き前の slot の寿命」の max なので churn 下では ~1ms しか見えず、**dwell 全体が telemetry からも不可視**。`oldestDeferredAgeMs=0`（:527）も同様にリセット。

## 3. 1回の re-drive が触る state / counter / telemetry

| 対象 | 操作 | 箇所 |
| --- | --- | --- |
| `deferredRetryCount_` | ++（同一 gen） | Orch:472-474 |
| `deferredSlot_` | **struct-replace（PublishRequest 一式コピー: sealedSnapshot/buildAnalysis/oversamplingResult/buildDiagnostics）** | Orch:505-521 |
| `hasDeferred_` | false→（finishView）→ true（release store ×2/tick） | Orch:637,522 |
| `DeferredHealth` telemetry | **2回/tick**（finishView の discard 記録 + enqueue の deferredCount=1 記録） | Orch:639-644,524-530 |
| `deferredOverwriteCount_` | 非増加（consume 済みのため hasDeferred_=false 状態で enqueue） | Orch:441-443 |
| `maxDeferredAgeMs_` | ~1ms 刻みで CAS 更新（実質据わり） | Orch:448-458 |
| `rebuildMutex` + `rebuildCV.notify_one` | lock/unlock + notify /tick | Threading:285-288 |
| 主 admission `evaluate` | RCU read（publicationReader）+ fading 判定 /tick | PA.cpp:51-58 |
| `evaluateDeferred` | snapshot 5値（atomic read ×4 + nowUs）/tick | Orch:611-621, PA.cpp:72-101 |
| DSP handle | 移動のみ（retire なし。`req.newDSP` は同一 handle 保持） | — |

## 4. 30s retention による CPU/churn upper bound

- 現状（kMax=2）: churn は **~4 tick（3-4ms）で終端**（要求破棄と引き換え）。
- Option A 単独: **上界なし**（§2）。1 tick あたりの仕事は §3（struct コピー×2、telemetry×2、mutex/notify、admission×2、RCU read）。RebuildThread は眠るので「CPU スピン」ではないが、**wake 頻度・メモリ帯域・telemetry レート・mutex 競合が fade 全期間 1ms 周期で発生**し続ける。CoordinatorLoop 自体は元々 1ms tick で他業務（processIntent/overflow drain 等）を回すため、増分は「deferred 再通知 + rebuild wake + 再評価一式」。
- 数式: `T_exhaustion(現) = (kMax+1)×tick ≈ 3-4ms` / `T_churn(Option A 単独) = fade 継続時間（上界なし）`。

## 5. generation supersession の実効 bound 性（F3-5）

`currentGeneration = rebuildRequestGeneration`（AudioEngine.h:1666、**要求採番時点**の atomic）。
evaluateDeferred の Generation 判定（PA.cpp:80-82）は「新しい rebuild 要求が実際に採番された場合」のみ旧 obligation を StaleDiscard する。
→ **イベント駆動であり時間駆動ではない**。fade 中にユーザー操作がなければ G+1 は来ない。
さらに単一 slot + struct-replace（Orch:505）なので、同一 obligation の churn 中に別要求が来れば SupersededDiscard で旧 DSP retire（:460-468）— これも「新要求が来た場合」のみ。
**結論: supersession は Option A の dwell bound として信頼できない**（F2 §6 の「bound の一つ」は条件付きに格下げ）。

## 6. fade-clear 起点の既存 wake mechanism の有無（F3-4）

**不存在（推測でなく全 producer 網羅で確認）。** fade 完了ブロック（Timer.cpp:928-1001）の処理:
`tryCompleteFade` → `notifyFadeComplete`/`consumeCompletedFade` → `endCrossfade`/`unregisterCrossfade` →
`fadingRuntimeDSPSlot` CAS クリア → `submitObserve`（retire intent）→ `crossfadeRuntime_.complete()` →
idle publish（`commitRuntimePublication`）→ `sendChangeMessage()`。
**`publishRetryReady`/`rebuildCV`/orchestrator への通知は一切ない。** `onPublishCommitted` 経由でも deferred は起こされない（producer 表 §1 の P1/P2 のみ）。
→ 現アーキテクチャで deferred が Ready 条件成立を知る唯一の経路は **1ms poll**。

## 7. `hasDeferred_` / `publishRetryReady` / `rebuildCV` の責務分離（F3-3）

| 要素 | 設計上の責務 | 実態 |
| --- | --- | --- |
| `hasDeferred_` | **状態**（atomic bool。テスト/DrainAudit/health の観測対象） | 正しい。predicate 非採用は妥当 |
| `publishRetryReady` | **wake フラグ**（rebuildMutex 保護、rebuild 側で read-and-clear） | 正しい |
| `rebuildCV` | wake 機構 | 正しい |
| Threading:277-279 の主張 | 「predicate に入れていないので busy-loop 防止。RebuildThread は休眠し CPU スピンしない」 | **半分正しい**: CPU spin は無い（眠る）。しかし P1 が毎 tick 再通知するため **実質 1ms polling** が成立しており、「churn なし」とは言えない。F2/F3 で分離すべきはこの2文 |

## 8. Option A 単独（retention を budget 外、他不変）

- 意味論: 正しい（Type B を失敗から分離）。
- 帰結: **churn 無制限化**（§2/§4）。kMax が担っていた「busy-loop 停止」の担い手が消える。
- 判定: **単独では不可**（F2 の推奨は §2 の発見により修正必要）。

## 9. Option A + event-driven wake

fade-complete 時に既存 handoff（`lock(rebuildMutex) + publishRetryReady=true + notify_one`）で 1 回起こす。
- **authority 観点**: producer 追加だが、**既存 P2（recovery）と同一プリミティブ・同一スレッド（MessageThread）の既存パターン踏襲**。publish/crossfade/retire の決定権は増やさない（wake は「再評価の依頼」であり決定ではない）。
- 条件成立の網羅性: Deferred になる唯一の理由は `DeferredFadingActive`（fading のみ）→ **fade-complete 通知で条件変化を完全にカバー**。stale/superseded/shutdown は既存 poll/watchdog 不要で別終端。
- 残存課題: notify 消失時の自己修復（lost-wake）→ watchdog 必須（§10 と併用）。
- 変更点: Timer.cpp fadeCompleted ブロック末尾に wake 追加（既存 P2 と同型、~4行）+ P1 の毎 tick 再通知を停止/間隔化。

## 10. Option A + bounded polling（watchdog 間隔化）

P1 を「毎 tick」から「遅延インクリメント間隔（例: 1ms→2ms→…→上限 100ms）」または固定 100ms に変更。
- 30s dwell 最悪ケースでも wake ≤ ~300 回（100ms 固定）。lost-wake 自己修復を維持。
- 単独では「条件成立後の遅延 ≤ watchdog 間隔」の応答性低下。event-driven と併用すれば遅延は通常 ~0。
- **TTL 失効問題（§2）は解決しない** — 別途 §11 の timestamp 修正が必要。

## 11. 推奨案（F3 確定）

> **Option A + event-driven wake（§9）+ watchdog 間隔化（§10）+ obligation-keyed TTL timestamp**

構成要素:
1. **retention は count しない**（F2 Option A の意味論 — 据え置き推奨）。
2. **fade-complete wake 追加**: Timer.cpp fadeCompleted ブロック末尾で既存 handoff を 1 回発火（P2 と同型・同スレッド。新 authority 無し）。
3. **P1 の毎 tick 再通知を廃止し watchdog 化**（例: 100ms 固定 or 指数 backoff）。lost-wake 自己修復を確保しつつ churn 上界を確定させる。
4. **TTL timestamp を obligation-keyed に**: `enqueueDeferred` で同一 generation の再 enqueue 時は `enqueueTimestampUs` を**引き継ぐ**（初回 defer 時刻を保持）。これにより 30s TTL が真の dwell bound として復元し、`maxDeferredAgeMs_` も dwell を正しく反映する。
5. churn 上界の再計算: fade 継続 F 秒 → wake 回数 ≈ 1（fade-complete）+ F/100ms（watchdog）。30s 滞留でも ≤ ~300 回 + TTL 終端。

## 12. 実装変更箇所（承認後のパッチ設計用。本監査では未実施）

| # | 箇所 | 変更 |
| --- | --- | --- |
| 1 | Orch.cpp:377-378 + :470-503 | retention（DeferredFadingActive 由来）を enqueueDeferred に伝達、count++ 対象外化（F2 §10 #1,#2 と同一） |
| 2 | Orch.cpp:445,505-521 | 同一 generation 再 enqueue 時に `enqueueTimestampUs` を旧 slot から引き継ぎ（metadata 側 :517 も） |
| 3 | Timer.cpp:1000 付近（fadeCompleted ブロック末尾） | 既存 handoff で deferred wake 1 回（`hasDeferredRequest()` 条件付き、P2 と同型） |
| 4 | Threading.cpp:280-289 | 毎 tick 再通知を watchdog 化（間隔カウンタ or 指数 backoff。`publishRetryReady` 意味は不変） |
| 5 | Orch.h:283-289 / State.h:16-20 / D135-1 文書 | 契約統一（`>` 意味論 + retention 除外 + TTL dwell 定義） |
| 6 | （任意）PA.cpp:76-77 | TTL 終端を `Expired` に分離（dead enum 活性化）— 別パッチ推奨 |

## 13. 必要テスト（実装前定義）

1. 既存 2-cycle 回帰（Debug+Release）。
2. **長期 retention**: fade を kMax×tick ≫（例 500ms）保持 → fade-complete wake で**即座に** publish 成立 + churn 上限（wake 回数）アサーション。
3. **TTL dwell 終端**: obligation-keyed timestamp 化後、fade 継続 30s+ で StaleDiscard/Expired 終端 + DSP reclaim（TTL 圧縮フックなしなら 30s 実測）。
4. **lost-wake watchdog**: wake 通知を意図的に消費不能にする経路は作れないため、watchdog 単独発火を DIAG で確認（fade-complete なしで dwell → watchdog 再評価回数 ≥1）。
5. kMax 数式 unit test（F2 §11 #4 と同一）。
6. ctest 40/40 Debug+Release、Release harness ≥5 回決定論確認。
7. DIAG アサーション: retention 中に `[HEALTH] starved` 不発、`[D135] re-defer (new)` のみ。

## 14. I4 / Practical Stable ISR Bridge invariant への影響

- **決定権**: 新設 wake は「再評価依頼」のみ（既存 P2 と同一プリミティブ）。publish/crossfade/retire の authority は不変。✅
- **NonRT 隔離**: 変更 3 は MessageThread（既存 recovery と同文脈）、変更 4 は CoordinatorLoop（NonRT worker）内で完結。RT 側変更なし。✅
- **Ownership/Retire**: 変更 1,2 は slot 内の timestamp/count のみ。DSP の retire 経路（正規 reclaim pipeline）不変。✅
- **単一 slot（INV-DEFERRED-2）**: 維持（struct-replace 方式は変えない。timestamp 引き継ぎのみ）。✅
- **CV predicate 不変**: `hasDeferred_` は引き続き predicate に入れない（poll 化しない）。✅
- 残存リスク: watchdog 間隔の応答性トレードオフ（条件成立検知 ≤ watchdog）。event-driven wake が主経路のため通常 ~0。

## F2 からの訂正点（明示）

- F2 §6「infinite re-drive 防止策: 30s TTL」→ **失効**（§2 で TTL が churn に無効化されることを確認）。bound は本 F3 §11 の構成（event wake + watchdog + obligation-keyed TTL）で再定義。
- F2 §6「generation supersession が bound」→ **条件付き**（§5: 新要求が来た場合のみ）。
- F2 §8 Option A 評価「第一候補」→ **条件付き候補**（ユーザー指摘通り。A の意味論は正しいが、budget 除去と churn 抑制は別問題として分離して解決する）。
