# Phase I-T1-D101-2.5/2.6 — Growth Admission & Observation Error Proof

> **Verdict: D101-2.5 = OPEN / D101-2.6 = OPEN**
> **M / R / R_cap / B_max^true = UNDETERMINED 継続（D101 #2 は OPEN のまま恒常 registration）**
> **A_max_candidate=1 / T_w=2 / max(E_w)=1 の observed characterization を維持 — M への昇格なし**
> コード変更・harness 変更・測定追加 なし。λ×G を未証明のまま M にしない。

## 1. D101-2.5-1 — 全 publish producer / admission path census

### Protocol 上の publishProducer は 2 経路だが、実正経 path は 1 概念

| 経路 | 入口 | commit 核心 | 状態 | 実 cense 参照 |
|---|---|---|---|---|
| RebuildThread 等 async prod. | `commitRuntimePublication`（AudioEngine.h:4595）→ `enqueueRuntimePublicationFireAndForget`（:4517 fire-and-forget） | `register → registry + OwnerChannel → Intent enqueue`（:4554-4577） | **sync wrapper**（:4595 で fire-and-forget に委譲後 receipt wait）。core は「commit 呼出回数」とは独立 | AudioEngine.h:4500-4584 |
| Bootstrap/Rebuild (#2/#4/#6) 直遂 | `enqueueRuntimePublicationFireAndForget` | 同上 | 同構造の直遂 path | AudioEngine.h:4517-4524 |
| Commit 受理側 | `executePublish`（RuntimePublishExecutor.h:76） | `RuntimeWorldAuthority::publish` → `publishAndSwap` | **唯一 execution boundary**（X4-B §6.4 / INV-X4-2） | RuntimePublishExecutor.h:76・Coordinator.h:130-147 |

### Admission Chain の各 Capacity

| 機構 | kCapacity | 性質 | ΔA_true(G) への bound 性格 |
|---|---|---|---|
| `OwnerChannel` | **256**（OwnerChannel.h: kCapacity 256） | Single Producer owner-transfer channel。Single Transfer（enq/take 各一機会）。各 World が **1 key / 1 slot** を 점유 | **同一 World の複重占用は仕様上不可能だが、channel 自体には再入可能（drain 後に update slot を探る）** |
| `RuntimeIntentCoordinator` Publish queue | 不明（I4 の RWS 続き。C_owner=256 換半分の既タイムledger は `FIXME` で退路試行未了） | IntentQueue は FIFO。take 後の slot は次 intent に再用。並列 Publication Intent は sequential に possess | **channel と同様に循環 queue — 単体で finite N_success 上界にならない** |
| `PendingPublishRegistry` | **64**（RuntimeWorldAuthority.h:34） | circular cursor (`% 64`)。D101-2.5 の指摘どおり「async enqueue→commit gap の fallback」 | **限責 cap は再帰。Drain 後に同 slot を再取できる** |

## 2. D101-2.5-2 — Queue capacity ≢ Burst bound

### 証明: `min(C_owner, C_intent, C_registry)` は `ΔA_true(G)` bound ではない

1. **World ∝ slot で用滅比は 1:1 だが、slot は再利用可能** — channel/queue/registry は全て drain 後に空 slot を再配し、同 window G 内に `C_owner` 以上の successful publication を累積できる。`min(capacity)` は**瞬間 concurrent bound**であり**時間-window 累積 bound** ではない。
2. **queue bounded ≠ admission bounded** — `enqueue()` が `false` を返した場合 caller は `CallerDestroy`（時に refill 後 retry 可能）。単体 snapshot の bounded は retry/再発行の累積 `sup ΔA_true(G)` を束ねない。
3. **stale / generation / sequence discard は N_success の bound ではない** — stale World の捨ては `RuntimeWorldAuthority::publish` 内 guard（commit Refuted 内で破砕）。PublishedDomain に入る = successful publication の数を制限しない。

**結論: いずれの単体 capacity も `ΔA_true(G)` の deterministic finite bound として採用不可。**

## 3. D101-2.5-3 — ΔA_true(G) の finite bound 有無

### 対象: `sup_t ΔA_true([t, t+G])`

- **A_true は PublishedDomain への arrival**（成功 commit = Σ publishWorld success + bootstrap）。
- G_sample（sampler 構造 GAP）、G_observe、τ_enqueue→execute（enqueue への跨り）、τ_execute→publish（Executor 実遅延）、τ_publish（commit 自遅延）は **区別すべきだが現段階ではいずれも fixed 定数ではない**（source-hold 計測値として規定されていない）。
- `G < τ_enqueue→publish` の場合、sampler 非観測 window 内に複数 World が LP を通過し得る — Δgrowth(G) を `λ_acquire × G` と置くには **rate contract** が必要だが、source 上に λ_acquire の deterministic 上界を許す **structural invariant は未特定**。
- **Burst duration τ_b、jitter bound J、G の相互規定も未固定**（D101 open Item の核心）。

**Verdict D101-2.5-D: λ_acquire に structural rate limit が存在するかの三型判定 — Case C（両方なし）を選択。**

| Case | 条件 | 判定 |
|---|---|---|
| A. deterministic rate bound (`N_success(t,t+G) ≤ ceil(λ_max G)+burst_allow`) | source 上 undiscovered | ✗ |
| B. finite burst capacity (`N_success(t,t+G) ≤ B_queue`) | §2 のとおり `B_queue` は recycled で累積 bound にならない | ✗ |
| C. どちらも現 Graph 上不存在 (`sup ΔA_true(G)=∞` を排除できない) | **有番観測（ΔA_true の G 内無制限増加を止める proof 未嗜好）** | ✅ **採用** |

**D101-2.5 = OPEN 継続** — 実測で 1/20/256 が得られても safe bound には昇格させない。

### D101-2.5-5 反例 9 項目（source graph で潰す）

| 反例 | 必須判定 |
|---|---|---|
| 複数 producer が同時に publish admission | **未束縛**（各 Producer が独立 enqueue → channel/queue/Memo は独立 admit） |
| 1 producer が queue cap より高速 submit | **未束縛**（Producer は non-blocking enqueue retryなく CallerDestroy→再発行可能） |
| queue dequeue が sampler より高速 | **bound でなし**（queue は退行 lag 而已、sampler speed は grow bound に無関連） |
| publish が gap 内に連続成功 | **未 sure 避**（有番観測 normal/ jitter の E_w=1 を見ても window-gap 内の N連続を属性 proof にできない） |
| OwnerChannel 空き後の再 enqueue 可能 | **可能（再入可能） → バースト累積を一度 cap にできない** |
| queue full → retry → 再 admission | **drain 後の再 admission は累積成功に含む → bound 無** |
| deferred publish が同一 window に再 Ready | **deferred resubmit（retry 機構）は同一 World の再 attempt だが retry sample を無視して N_success 累積の source とするかを構成分離要** |
| generation/sequence discard が growth 制限か | **supply：Rejected は PublishedDomain 未入 — N_success 上界に無関連** |
| stale discard は N_success bound か | **supply：CanceledPublish は durable 成功ではない — 上界に無関連** |

## 4. D101-2.6 — E^obs の切離し（T_w 意味論確定）

### 対象模型

```
B_true → observer(B_ref) → sampler(T_w)
```

- **E1 — Reference linearization point:** 各 acquire/release の atomic event は `WorldRetirementReferenceObserver` の fetch_add / CAS window 内で max 線形化（D101-2C 共用 step 同期）。event時自身の linearization、observer B_ref の peak 在室。
- **E2 — B_ref Peak 保持:** `referenceMax_` は window 内 running maximum（T_w）を event-driven で保持（`updateRunningMax`）。単に current outstanding ではなく **accumulated 内 peak**。
- **E3 — Window-local maximum:** window 開始時に reset（1 から始動）、終了時 samplerTick で bounded に閉鎖。`B_ref,max(window)` は window 内全 event を包含（sr tick 同期）。
- **E4 — T_w の本性:** `T_w = observer's accumulated/window peak`（**sampler が reference peak を再サンプリングする一様ではない — event-driven peak を cassette 前の原像として保有**）。もし `T_w = sampled current B_ref`なら G が観測誤差に直入だが、現機構は **peak 捕捉経路なので λ の事変度は E^obs に直接寄りかからない**。

→ **sampling cadence λ は reference event capture completeness に作用しない — E1-E4 の寓明により、頭専次 GtT_w が sampled-current 拂句の錯誤であることを source 上二是分する。**

### 今回は `E_obs` 数式化を捨てる

`E^obs` の finite bound は **未証明のまま残すことが正しい停止**（I4 D101.2 の設計どおり）:

```
E_obs(G) = sup [B_true - B_ref/T_w] の有限 bound は現 source 上 undiscovered
M ≥ Δgrowth + E^obs 方向は形のみ提示、数値は入れない
```

`E_obs ≤ λG` 等を仮側から導かない。先に **T_w が何を保持しているか**を source 上で完全確認できたこと自体が本 track の closure point。

## 5. Gate（5 項目）

| Gate | 内容 | 判定 |
|---|---|---|
| D101-2.5-1 | 全 publish producer / admission path census | ✅ CLOSED（2 producer path + 3 capacity 機制を完全列挙） |
| D101-2.5-2 | queue/registry capacity と publication admission の関係 | ✅ CLOSED（capacity ≠ burst bound を証明） |
| D101-2.5-3 | ΔA_true(G) の deterministic finite bound 有無 | **OPEN（Case C — structural finite bound 未発見）** |
| D101-2.6-1 | reference observer の event/linearization/peak semantics | ✅ CLOSED（E1-E4 二分完了） |
| D101-2.6-2 | T_w が sampled-current か observer-peak かの確定 | ✅ CLOSED（**observer-peak** として sampler の λ から分離） |

## 6. 最終判定

```
D101-2.5 — Δgrowth finite structural bound: OPEN
    └─ 有限 bound を作ることが核心ではなく、finite が invariant として本当に存在するかを証明できなかったことが正しい停止
D101-2.6 — E_obs finite structural bound: OPEN
    └─ 有限 E^obs を証明する前に T_w 意味論を確定する → T_w = observer-peak 切り分けにより sampler λ と observe completeness の誤連関を解消

D101 #2（総括） = OPEN / M = UNDETERMINED / R / R_cap / T2 = UNDETERMINED
```

## 7. 禁止事項（今回も守守）

`× コード改変 / harness 改変 / 付試計測 / A_max 更新 / queue cap 安全化 / λ×G 未証明化 / max(E_w)=1→M / T_w=2→M / R 正定 / B_max^true 数値化 / T2 / Reservation gate` — 全て実施せず。

## 8. 使唱 Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | admission 全 10 指標 / queue capacity / observer semantics を複数手段一致 |
| Sandbox | node fs walk / source 直読（RuntimeWorldAuthority.h / AudioEngine.h / ISRWorldRetirementReference.h / ISRWorldRetirementTelemetry.h / RuntimeIntentCoordinator.h） | enumeration 正経性を確定 |
| MCP/CLI | serena（一次無効・前 Gate 多数確定ずみで代替） / AiDex（一時無効・代替 node walk） | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索施行 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK（Vyukov失効→rigtorp 代替、crossbeam 404 は正規 URL で 200） |

*産空: Phase I-T1-D101-2.5/2.6 — 成長と観測誤差を尺の定式化へ持つ前に「有限 bound が invariant として本当に在るか」を未証明のまま正しく止めることが本 track の所在。*

