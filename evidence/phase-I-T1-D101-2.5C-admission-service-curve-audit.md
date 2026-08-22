# Phase I-T1-D101-2.5C — Admission Service-Curve Audit

> **Verdict: D101-2.5C = OPEN (Case B — Missing Admission Envelope Contract) / M = UNDETERMINED**
> **service_curve(G) としての `N_success(t,t+G) ≤ f(G)` を既存 invariant だけから導出不可**
> コード変更・測定追加なし。単一 executor 規律 ≠ 有限 rate bound を source-level で確定。

## 1. D101-2.5C Step 1 — executePublish 完全 loop 構造確定

```
CoordinatorLoop::run()                         [ISRCoordinatorLoop.cpp:31]
  while(!threadShouldExit())
    ├─ engine_.runCoordinatorPhase()           [AudioEngine.Threading.cpp:254]
    │    ├─ runtimePublicationBridge_.processIntent(*this, lifetimeMgr)  [ISRRuntimePublicationCoordinator_ProcessIntent.cpp:5]
    │    │    ├─ while(quarantineFallbackQueue_.pop(intent)) → handle(Quarantine)
    │    │    ├─ while(intentQueue_.pop(intent))              → DispatchTable handle(Publish|Quarantine|Observe|Recovery)
    │    │    │    └─ Publish: PublishExecutor::executePublish(authority, intent, ctx) [RuntimePublishExecutor.h:20]
    │    │    │         └─ authority.publish(owner, metadata) → publishAndSwap [RuntimeWorldAuthority.h:249]
    │    │    └─ drainObserveDeferred(...)                    [ProcessIntent.cpp: ★ FUTURE-8/QUEUE-16]
    │    ├─ deferred publish resubmit (flag → RebuildThread) [AudioEngine.Threading.cpp:264-278]
    │    ├─ overflow ring drain                               [285]
    │    └─ deferred retire drain (Q/E/T event wake + 1ms fallback) [295-299]
    └─ waitForDrainSignalOrTimeout(kIntervalMs=1ms)          [ISRCoordinatorLoop.cpp:47]
```

**判定:**

| 観点 | 実装 | 有限性への寄与 |
|------|------|----------------|
| 1 tick / 1 wake で処理できる publish intent 数 | `while(intentQueue_.pop(intent))` — **unbounded drain**. queue に N intent があれば N 回 `executePublish` を連続呼出 | **No per-tick cardinality bound** |
| while/for drain loop 有無 | **あり** (2つの while: fallback + main queue) | unbounded drain operand |
| 1 publication 後即時次 publication 可否 | Yes — loop 内で次 intent を即時 pop → publish | **No mandatory inter-publication wait** |
| publish ごとに必須 wait/event/handoff | publish → bridge retire (old world を queue) のみ。次 publish への wait なし | **No per-publication gate** |
| publishAndSwap 自体に temporal serialization | publisher は single serialized executor (CoordinatorLoop thread). ordering 保証だが rate 保証ではない | **Ordering ≠ Rate** |

## 2. Step 2 — service-time lower bound τ_service_min > 0 の探索

| 候補 | 不変条件 `every success consumes ≥ τ_min` の有無 |
|------|---------------------------------------------------|
| executePublish() | publishAndSwap (atomic exchange acquire-release) は nanoseconds オーダーの lock-free swap. invariant としての下界なし |
| RuntimeWorldAuthority::publish() | commit metadata bake + publishAndSwap を束ねるが両者とも lock-free atomic のみ |
| publishAndSwap() | `RuntimeStore::publishAndSwap` は atomic exchange — duration 上界なし |
| retire enqueue | old world の retire は publish 成功後の tail だが publish admission の前提条件ではない |
| receipt generation | `onPublishCommitted → notifyPublishReceipt` は publish 後の completion-notify。次 publish の admission gate ではない |
| coordinator wake | wake は drain signal (Q/E/T pending) または 1ms timeout による wake。publish admission の throttle ではない |
| intent dequeue | pop は lock-free queue の atomic dequeue — service time 契約なし |
| crossfade/decision | `PublishDecisionSnapshot` は enqueue 時に固定 (Decision Snapshot)。execution 遅延の min 契約なし |

**結論: `τ_service_min > 0` を構造的に保証する箇所なし。測定なし原則により実測値 (例:平均100µs) を bound に昇格させることも禁止。**

## 3. Step 3 — waitForDrainSignalOrTimeout() の rate-limiter 性判定

```
wait(timeout) 存在 → N_success(G) ≤ ceil(G / timeout) ?  → NO
```

| 区分 | 判定 |
|------|------|
| timeout-only | 1ms fallback は busy-wait 防止の bounded join 手段 |
| signal-wakeup | drainCv_ は drain signal (Q/E/T pending) で即時 wake — timeout 前に起床可能 |
| unbounded-drain | wake 後は `while(pop)` で複数 publication を一括処理 — 1 wake で複数 success を生成可能 |

**よって timeout は rate contract ではない。`ceil(G/1ms)` は単一 wake 内の burst を束縛しない。**

## 4. Step 4 — 1 turn 1 publish 隠れ invariant 有無確認

| 分類 | 実装 | 判定 |
|------|------|------|
| Case A: single-publication turn + bounded turn frequency | 1 turn = 1 intent であれば `N_success(G) ≤ turns(G)` 導出可 | **不成立** — `while(pop)` の unbounded drain により 1 turn で任意数 publish 可能 |
| Case B: serialized executor + unbounded drain | 単一 executor + FIFO + bounded queues + wake/timeout だが per-time bound なし | **★ 本実装の分類** |
| Case C: other structural limiter | queue capacity / registry cap は瞬間 concurrent bound (再利用可能なため累積 bound ではない — D101-2.5/2.6 で証明済み) | 該当なし |

## 5. Step 5 — commitRuntimePublication receipt wait の分離

`commitRuntimePublication()` は fire-and-forget core (`enqueueRuntimePublicationFireAndForget`) に receipt wait を被せた**同期 wrapper** (AudioEngine.h:4595-4627):

```
Producer A: enqueue → success (owners transferred, waitFor receipt kPublishReceiptWaitTimeoutMs=250ms)
Producer B: enqueue → success (独立 — A の wait 中でも enqueue 可能)
Coordinator: while(pop) で A,B 共に executePublish → commit → receipt notify
```

- receipt wait は **producer 側の待機**であり **Coordinator admission の throttle ではない**
- Producer A の wait 中も Producer B は別 key で enqueue 可能 (OwnerChannel key = {seqId, epoch, mappedGen} で衝突しない)
- Coordinator は両者を単一 wake 内で連続 publish → **receipt wait ≠ λ_max**

## 6. Step 6 — deferred / recovery / bootstrap の競合モデル再分類

| Source | 同一 executor へ投入 | 同時存在 | 再投入可能 | A_true 増加 |
|--------|---------------------:|---------:|----------:|------------:|
| normal rebuild | Yes | Yes (queue に待機) | Yes (deferred resubmit → RebuildThread) | success only |
| bootstrap | Yes | 一時的 (起動時1回) | No | success only |
| deferred | Yes | flag 駆動 (1 outstanding) | Yes (flag+CV 再起床) | success only |
| recovery | Yes | Builder Work Queue 転送 | Yes | success only |

- retry は同一 World の再試行 → `ΔA_true` 非寄与 (同 World の再 attempt を N_success から除外)
- しかし retry が別 World の成功を **starvation なく追加**できるため、retry 存在自体が per-time N_success 上界を自動的に与えるわけではない — **混同禁止** (指示どおり)

## 7. 最終判定 (5 Gate)

| Gate | 判定対象 | 判定 |
|------|----------|------|
| 2.5C-1 | successful admission LP 固定 | PASS — single LP (publishAndSwap success) |
| 2.5C-2 | executor drain cardinality 固定 | PASS — while(pop) unbounded drain として固定 |
| 2.5C-3 | service-time lower bound 有無 | **OPEN — τ_service_min > 0 の不変条件なし** |
| 2.5C-4 | producer receipt wait / wake が rate contract か | **NO — いずれも rate ではなく wait/signal の役割** |
| 2.5C-5 | finite service curve 導出可否 | **不可 — `N_success([t,t+G]) ≤ f(G)` を既存 invariant から証明不可** |

```
A: structural service curve 発見 → 不成立 (τ_min / ceil(G/τ_min) / B+λG いずれも導出不可)
B: 発見できない → single executor + FIFO + bounded queues + wake/timeout ≠ temporal admission bound を source で確定
   → D101-2.5C = OPEN / Missing Admission Envelope Contract 確定
C: structural unboundedness 実証（∀N ∃execution: ΔA_true(G)>N 構成）→ 本監査では主張しない
```

**最終支持: `finite structural bound = 未証明` (Case B 維持) — これが正確な停止状態。`sup ΔA_true(G)=∞` とは書かない。**

## 8. 禁止事項遵守

コード変更 / admission limiter 実装 / harness 変更 / 追加測定 / λ×G 仮採用 / queue capacity burst 化 / T_w→M / max(E_w)→M / R/R_cap/B_max^true 数値化 / T2 / Reservation gate — **全て未実施**

## 9. 次の確定状態

```
D101 #1       CLOSED (D101-1R2)
D101-2.5      OPEN (missing explicit admission envelope)
D101-2.5C     OPEN (service-curve 不在を再確認 — 無制限 drain ループが核心)
D101-2.6R     PARTIAL (E_ref=0 proven, E_sample は T_w=peak により λ 非依存だが数値 bound 未証明)
A_max=1 / T_w=2 / max(E_w)=1  observed only
M/R/R_cap/B_max^true UNDETERMINED / T2 NO-GO
```

**次に行うべきは missing contract (admission rate / burst envelope) の設計判断 — 新しい測定ではなく実装契約追加の判断。**

## 10. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | loop 構造 / service-time / capacity / G_sampler 全 census 一致 |
| Sandbox | node fs walk / source 直読 (ISRCoordinatorLoop.cpp full, AudioEngine.Threading.cpp:250-330, RuntimePublishExecutor.h, PublicationExecutor.h, Coordinator.h processIntent, Telemetry.h:311 kExpectedTickIntervalUs) | enumeration 正確性確定 |
| MCP | serena 一時無効 (前 Gate 多数確定済みで代替 node walk) | — |
| MCP | AiDex 一時無効 (node walk 代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK |
