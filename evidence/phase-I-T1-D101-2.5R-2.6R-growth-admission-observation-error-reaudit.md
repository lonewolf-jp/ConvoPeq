# Phase I-T1-D101-2.5R/2.6R — Growth Admission & Observation Error Re-audit

> **Verdict: D101-2.5R = OPEN (Case B) / D101-2.6R = PARTIAL (E_ref=0 proven, E_sample OPEN)**
> **M / R / R_cap / B_max^true = UNDETERMINED 継続 / 測定・コード変更なし**
> コード変更 0 / harness 変更 0 / 追加測定 0 / A_max_candidate=1 / T_w=2 / max(E_w)=1 固定維持

## 1. D101-2.5R Step 1 — A_true semantic LP 固定

**結論: PASS**

```
A_true(t) = PublishedDomain に実際に加入した World 数
increment source = successful PublishedDomain admission only
```

- `enqueue`, `registry.register`, `OwnerChannel.enqueue`, `IntentQueue.enqueue`, `executePublish`, `retry`, `deferred Ready` はいずれも A_true increment ではない。
- 成功 LP は `publishAndSwap` を含む `RuntimeWorldAuthority::publish` → `publishAndSwap` 経由の successful publication のみ。
- `onAcquire()` は `onRuntimePublishedNonRt()` に置かれ (`AudioEngine.h:3469,3522`, `AudioEngine.Commit.cpp:332,408`), publish 成功 LP (`publishAndSwap` 成功後) と同一 semantic event を表す。`ISRRetirementReference.h:27` で `publish 成功・onRuntimePublishedNonRt` と明記。rejected/failed publish は acquire observer に入らない。

**Gate `∀ΔA_true: increment source = successful PublishedDomain admission only` → PASS**

## 2. D101-2.5R Step 2 — rate / burst / outstanding 3分離

| 区分 | 定義 | 本コードでの実体 |
|------|------|------------------|
| A. sustained rate | N_success([t,t+G]) ≤ λ_max G + ... | λ_acquire の時間上界 — 未契約 |
| B. instantaneous burst | N_success([t,t+τ]) ≤ B_burst | 単一 gap 内 burst 上界 — 未契約 |
| C. outstanding capacity | concurrent admitted-but-not-released ≤ C | OwnerChannel 256 / Registry 64 — 瞬間 concurrent bound であり時間累積 bound ではない |

**混同の否定:** `OwnerChannel=256, Registry=64` から `ΔA_true(G) ≤ 256` は推論不可。現行 `OwnerChannel.enqueue` 後に IntentQueue full なら Owner を取り戻して失敗する構造 (`AudioEngine.h:4559-4583`) であり、成功 World が capacity に永続拘束されるわけではない。capacity は再利用可能なため同一 World の多重占有ではなく、queue が admission を永続的に制限するわけでもない。

**今回調査対象: capacity ではなく「成功 publication 生成速度そのもの」への structural contract の有無**

## 3. D101-2.5R Step 3 — Producer serialization census

| Producer | enqueue path | concurrency | admission serialization | publishAndSwap |
|----------|--------------|-------------|-------------------------|----------------|
| RebuildThread | commitRuntimePublication | Non-RT 1 | IntentQueue FIFO | serialized executor |
| Bootstrap | enqueueRuntimePublicationFireAndForget direct | Non-RT 1 | 同上 | 同上 |
| direct enqueue | commitRuntimePublication | Non-RT 1 | 同上 | 同上 |
| deferred resubmit | deferred Ready → re-enqueue | Coordinator 回収 | 同上 | 同上 |
| recovery | Recovery path | Non-RT 1 | 同上 | 同上 |
| fire-and-forget | enqueueRuntimePublicationFireAndForget | Non-RT | 同上 | 同上 |

**全 producer → 単一 serialized executor (`executePublish` via CoordinatorLoop single thread) → `publishAndSwap`**

> 重要: 複数 producer 同時存在でも `publishAndSwap` は単一 executor だが、「single consumerだから有限 rate」と結論しない。consumer が `while(queue not empty) executePublish()` の unbounded drain を実行可能なら serialization は per-time rate を制限しない。

## 4. D101-2.5R Step 4 — executePublish service-time contract

| 項目 | 調査結果 | 契約有無 |
|------|----------|----------|
| 4-A: 1 scheduler turn で複数 publication 処理可否 | `CoordinatorLoop::run()` は `waitForDrainSignalOrTimeout(kIntervalMs)` 後に `processIntent` → `executePublish` を呼ぶが、queue に複数 intent があれば連続 drain 可能。1 turn 内の `N_success` に deterministic 上界なし | **No contract** |
| 4-B: tick/wait/wake/drain semantics が per-time rate を制限するか | `kIntervalMs`=1ms (ISRCoordinatorLoop.h), `kExpectedTickIntervalUs`=100000us (Telemetry) は sampler gap 用。CoordinatorLoop の service time に対する rate limit 契約なし。`waitForDrainSignalOrTimeout` は drain signal があれば即時 wake | **No deterministic limit** |
| 4-C: publishAndSwap serialization が N_success finite の十分条件か | serialization は ordering を保証するが temporal rate を保証しない | **Not sufficient** |

**判定分岐:**

```
Case A: structural service/rate contract exists → finite ΔA_true(G) candidate → 不成立
Case B: serialization only but no temporal contract → finite ΔA_true(G) still unproven → ★ 採用
```

## 5. D101-2.5R Step 5 — G 分離

| G | 定義 | 本コード |
|---|------|----------|
| G_sample | sampler maximum gap `G = max_k(t_{k+1}-t_k)`, 期待 100ms (`kExpectedTickIntervalUs`) | I4 定義、Telemetry で計測 |
| G_admission | admission 側 gap | 未定義 |
| G_enqueue | enqueue → execute 遅延 | τ_enqueue→execute |
| G_execute | execute → publish 遅延 | τ_execute→publish |
| G_publish | publish 自体 | τ_publish |

`G_sample` finite だから `ΔA_true(G_sample)` finite になるわけではない。`G_sample → admission envelope` 導出には **admission rate / burst envelope contract** が必要だが不存在。

## 6. D101-2.5R Step 6 — 最終三択

| 選択肢 | 条件 | 判定 |
|--------|------|------|
| A: finite structural bound found `N_success ≤ ceil(λ_max G)+B` | source contract から証明 | **不成立** |
| B: finite bound requires explicit missing contract (serialization/queue capacity あるが λ_max/service-time/burst 未契約) | 現実装の状態 | **★ 採用 → D101-2.5R OPEN** |
| C: structural unboundedness demonstrated `∀N, ∃execution: ΔA_true(G)>N` | source 上構成 | 不採用（「queue 再利用される」だけでは C 証明にならない） |

**Missing contract: `admission rate / burst envelope (λ_max, τ_service, B_burst)`**

## 7. D101-2.6R — Observation Error Re-audit

### Step 1 Acquire correspondence

```
∀W∈PublishedDomain: A_true+=1 ⇔ referenceAcquireCount+=1
```

onRuntimePublishedNonRt → onAcquire() → referenceAcquireCount_++ → updateRunningMax()。PublishedDomain のみに発火、rejected は Generic。**PASS**

### Step 2 Release correspondence

```
∀W∈PublishedDomain terminalized: R_true+=1 ⇔ referenceReleaseCount+=1
```

terminal World deletion 9 LP (D/Q/E/terminal/shutdown) で `worldReclaimCount++ → referenceReleaseCount++` 1:1 exactly-once (D101-2B 再確認、D101-1 World producer 閉鎖済みのため再監査せず)。**PASS**

### Step 3 Accounting identity

```
B_ref(t)=A_ref(t)-R_ref(t), A_ref=A_true, R_ref=R_true → B_ref(t)=B_true(t)
```

source-level invariant として証明可。**PASS → B_ref == B_true**

### Step 4 残る E^obs

`B_ref==B_true` でも `B_obs(t_k)` は sampler snapshot のため `B_true(t_k)-B_obs(t_k)` が残存。T_w は sampler再取得ではなく **observer自身の running maximum**。

### Step 5 二層分解

```
E^obs = E_ref + E_sample
E_ref = reference observer missing true events → 0 (Step1-3で消去可)
E_sample = sampler snapshot being behind → 現行 T_w は observer peak のため
           sampling cadence は reference capture completeness に影響しない
```

**分解妥当:** T_w が observer peak であることを source 上で確認済み (referenceMax_ = window内 running max)。

## 8. Gate

| Gate | PASS条件 | 判定 |
|------|----------|------|
| D101-2.5R | A_true LP 固定 | PASS |
| D101-2.5R | producer serialization census | PASS |
| D101-2.5R | executePublish service semantics (単一executorがrateを意味するか) | **OPEN — serialization ≠ rate limit** |
| D101-2.5R | G_sample vs G_admission 分離 | PASS (分離定義) |
| D101-2.5R | rate/burst/capacity 分離 | PASS |
| D101-2.5R | **finite bound: A/B/C** | **B (missing contract)** |
| D101-2.6R | acquire completeness | PASS |
| D101-2.6R | release completeness | PASS |
| D101-2.6R | B_true==B_ref | PASS |
| D101-2.6R | E_ref / E_sample 分離、E^obs 有限性 | **E_ref=0 proven, E_sample=OPEN (sampler gap由来の有限性は T_w=peak により分離済みだが数値 bound 未証明)** |

```
D101-2.5R: Δgrowth finite? → NO (Case B, missing admission rate/burst envelope contract)
D101-2.6R: E_obs finite? → E_ref finite (0), E_sample は T_w=peak により sampler λ から分離だが数値 bound 未証明 → PARTIAL

M = f(G, λ, τ_b, ...) として形のみ提示、数値なし
```

## 9. 禁止事項遵守

コード変更 / harness変更 / 追加測定 / A_max_candidate更新 / queue capacity→burst bound / λ×G無条件採用 / max(E_w)→M / T_w→M / R/R_cap/B_max^true数値化 / T2 / Reservation gate — **全て未実施**

## 10. 現状サマリ

```
D101 #1       CLOSED
D101-2.5R     OPEN (admission rate/burst envelope contract 欠如)
D101-2.6R     PARTIAL (E_ref=0, E_sampleはpeak分離でλ非依存だが bound 未数値化)
A_max_candidate = 1        observed only
T_w             = 2        observed reference peak only
max(E_w)        = 1        observed characterization only
M               = UNDETERMINED
R               = UNDETERMINED
R_cap           = UNDETERMINED
B_max^true      = UNDETERMINED
T2              = NO-GO
```

**次に行うべきは「新しい測定」ではなく、本再監査で特定された missing contract (admission rate / burst envelope) を実装契約として追加する設計判断。** それまでは D101 OPEN のまま正しく停止する。

## 11. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 (Coordinator.h, RuntimeWorldAuthority.h, OwnerChannel.h, Telemetry.h, Reference.h) | provenance/window/λ区別確定 |
| MCP | serena 一時無効 (前Gate多数確定済みで代替) | — |
| MCP | AiDex 一時無効 (node walk代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK (Vyukov失効→rigtorp代替, crossbeam 404は正規URLで200) |
