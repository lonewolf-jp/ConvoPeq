# D147 Phase 0 — RecoveryFailure Event Production / Capacity Re-proof (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only。**Production source changes: 0. Test source changes: 0。Ring 実装・W5 移動・コメント修正・P5/P6 すべて非実施。**
**基準:** `ConvoPeq.md Generated: 2026-08-31 15:32:58`（D146 後版）。D144/D145 の数値仮定は引き継がず、現行コードから再導出。
**結論先出し:** **`512+128` は撤回**（リング方式の F_max ≈ 21504 ≫ 640 を厳密導出）。**再設計案 T3 = per-slot saturating signal counter（F_max = K×L = 128 構造的）**を推奨。Phase 1 は T3 固定で進めてよい（GO 条件 7/7 成立）。

---

## 1. Failure Site Matrix（現行コード実測・D146 後行番号）

| # | サイト | failure predicate | producer thread | 1 invocation あたり post 数 | retry loop 内複数回 | continue/break 後再処理 | settle 順序 | rearm 独立 | terminal 後 | duplicate |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RebuildDispatch:1006 | `recoveryResult.runtime == nullptr`（pop 済み intent 1 件） | RebuildThread | 1 | なし（pop 毎に 1 回） | 次 pop は**別 intent=別観測**（正当） | n/a（transport） | n/a | Live チェック no-op | なし |
| 2 | RebuildDispatch:1033 | `validateWarmup != None`（同 pop の続行） | RebuildThread | 1 | なし | 同上 | n/a | n/a | 同上 | なし |
| 3 | RebuildDispatch:1091 | durable build null（take 1 件） | RebuildThread | 1 | while ループ再 take→**失敗毎に別観測**（spin 上限 4/run） | break 後は次 wake まで停止 | **settle(true) 後 post**（順序固定） | n/a | 同上 | なし |
| 4 | RebuildDispatch:1115 | durable warmup 失敗 | RebuildThread | 1 | 同上 | 同上 | 同上 | n/a | 同上 | なし |
| 5 | Orchestrator:311 | publish executor 失敗（attempt 1 回） | RebuildThread | 1 | なし | decision 返却後再試行なし | n/a | n/a | 同上 | なし |
| 6 | Orchestrator:401 | `RejectedPressure` decision 1 回 | RebuildThread | 1 | なし | — | — | **rearm とは独立呼び**（rearm は delivery 非触达・settle のみ） | 同上 | なし |

**Sites は 6 に閉じている**（grep 実測: markTransientFailure の production caller はこの 6 箇所のみ、D143-1 と同一）。全 site RebuildThread 実行（caller 連鎖は D143-1 で確定済み、D146 で不変）。

## 2. 「1 obligation ≤ K events」を再び仮定しない（§3 指示）

分離の明示:
- **K=4** は `consecutiveFailureCount` の **adjudication 結果**（枯渇→terminal）であり、**post 数ではない**。
- 同一 obligation の intent が transport queue に複数存在し得る（C16 再 push 設計、`fillRecoveryQueue`=256 前例）。各 pop 失敗は**独立した failure observation**。adjudication は次 tick まで行われないため、**1 obligation が 1 窓で K を超えて post し得る**（最大 = その obligation の queue 内 intent 数）。
- 従って observation 数は K で上界化しない。リング方式の F_max 導出はこの事実を織り込む。

## 3. Capacity Derivation（§4 変数化・実コード制御フローから）

```text
Q = 256   recoveryIntentQueue_ 容量（h:927）
D = 1     durable スロット（単一）
L = 32    kMaxLogicalRecoveryObligations（h:363）
K = 4     kMaxObligationConsecutiveFailures（h:369）
I = 4096  intentQueue_ 容量（h:994）
Fq = 1024 quarantineFallbackQueue_ 容量（h:1002）
```
**CL の消費窓**: `runCoordinatorPhase` は 1 tick に 1 回 `processIntent`（**intentQueue_ と quarantineFallbackQueue_ を while-pop でフルドレイン** — ProcessIntent.cpp:36/49 実測）→ 窓内処理 quarantine intent ≤ I+Fq = 5120 → 成功 quarantine（stateChanged 限定、ProcessIntent:137）≤ 5120 → **submit ≤ 5120**。

**窓内 production（リング方式 T1）**:
```text
push(transport) ≤ submit ≤ 5120（+ redrive ≤ L=32: 同一 obligation は push 後 delivery!=None
              → 同一窓の再スキャンでは再 push 不能。adjudication は窓頭に済んでいるため窓内 +32 まで）
pop ≤ Q + push ≤ 256 + 5152 = 5408
T（transport 失敗 event）≤ pop ≤ 5408
success ≤ pop → publish attempt ≤ 5408 + task publish（≤ wake 数 ≤ submit 数）≤ 10560
P（publish 失敗 event）≤ 10560
durable take ≤ wake 数、失敗 ≤ take、spin ≤ 4/run → R ≤ 5408
F_max(T1) = T + P + R ≤ 5408 + 10560 + 5408 ≈ 21376（有限。ただし 640 を大幅超過）
```
→ **§5 の順序（先に F_max）に従い: F_max(T1) ≫ 640 ⇒ `512+128` は構造的十分性なし。撤回。**
（実運用の窓内 burst は桁で小さいが、証明は数学的上界でなければならない。概算「≲520」は不採用。）

## 4. 再設計 T3: per-slot saturating signal counter（推奨）

リングを置かず、**obligation slot 内の飽和 atomic** で観測を伝達する:

```text
LogicalRecoveryObligation に追加（Phase 1 仕様）:
    std::atomic<std::uint8_t> pendingFailureSignals{0};   // cap K（飽和加算）

Producer（RebuildThread・6 サイト）: postRecoveryFailureSignal(oblId)
    slot を oblId で id-scan（現 markTransientFailure と同一検索）
    state==Live でなければ no-op（stale/unknown/terminal/post-reuse 一括処理）
    pendingFailureSignals: fetch_add(1) が K 未満のときのみ反映（CAS で K に飽和）
      ─ K 超過の観測は inert: adjudication は K 到達で terminal 化するため、
         現行の直接 5 回目呼び（counter=5→terminal）と同一帰結

Consumer（CoordinatorLoop・runCoordinatorPhase、processIntent 後 redrive 前）:
    各 Live slot について n = pendingFailureSignals.exchange(0, acquire)
    n > 0 なら: delivery = None（1 回・冪等）
                consecutiveFailureCount += n（飽和）→ ≥K で resolve(Failed)（exhausted telemetry 1 回）

slot 再利用時: tryInsert で pendingFailureSignals を reset（consecutiveFailureCount reset と同位置 h:407）
```

**F_max(T3) = K × L = 4 × 32 = 128 — 到着レート非依存の構造的界。** リング・fallback・昇格すべて不要。

### 意味論の同一性証明（現行 markTransientFailure との等価性）
| 観点 | 現行（直接呼び） | T3 | 判定 |
|---|---|---|---|
| 1 observation = 1 加算 | 呼び出し毎 +1 | fetch_add 毎 +1（飽和後 inert） | ✓（飽和超過は terminal 確定後の観測と等価） |
| K=4 で terminal | 4 回目呼びで resolve | adjudication で counter≥4 → resolve | ✓ |
| delivery=None | 呼び毎（同一値） | adjudication 毎（冪等） | ✓ |
| exhausted telemetry | 枯渇 1 回/obligation | 枯渇 1 回/obligation | ✓ |
| stale/unknown oblId | id-scan no-op | id-scan no-op（同一機構） | ✓ |
| slot reuse 後 stale | id 単調性で no-op | 同一 + signals reset | ✓ |
| R18/R20/R21 既存テスト | — | 期待値不変（adjudication 結果同一） | ✓（Phase 1 で回帰実測） |

## 5. overflow semantics 3 層分離（§6）

| 層 | T1（リング 640） | T3（飽和 signal） |
|---|---|---|
| 1. capacity sufficient | ✗（F_max≈21376 > 640） | ✓（128 = 界そのもの） |
| 2. overflow impossible | ✗（理論到達可能） | ✓（飽和により構造的不能） |
| 3. overflow 時に非 silent | 昇格+counter で ✓ | 飽和時 counter（`recoveryFailureSignalSaturatedCount_`）で ✓ — かつ飽和は**意味的に inert**（terminal 確定済相当）なので情報損失なし |

T3 では層 2 が成立するため層 3 は防御のみ。T1 で層 2 を得るにはリング ≥ 21376（≈512KB）が必要で、これは T3 より劣る。

## 6. Event struct 仕様（§7）— T3 では消滅

T3 の伝達単位は `oblId`（関数引数）のみ。
- adjudication 必要値: **oblId のみ**（delivery/counter/resolve は全て slot 内で完結）— 現行 markTransientFailure が既にこの入力だけで成立していること自体が証拠。
- `seq`: 不要（順序は counter 加算の可換性で不問。telemetry 必要なら global atomic 采番で足りる）。
- `siteId`: telemetry 専用（任意引数として保持可、adjudication 意味論に非関与）。
- terminal/stale/unknown 判定: **Coordinator 側だけで成立**（id-scan + Live チェック — 現行 cpp:1070-1074 と同一機構）。

## 7. exactly-once 再定義（§8）

目標対応は **「1 failure observation ↔ 1 signal」**（「1 obligation ↔ 1 event」ではない — 後者は K 意味論を壊すため誤り）。サイト別の判定は §1 表のとおり全サイト duplicate/missing なし（各観測は自身のサイト通過のみで 1 回）。stale/terminal 後/unknown は id-scan + Live + 飽和で安全側に吸収。

## 8. W5 の現状記録（§9 — 変更なし）

- 現行 delivery writer: W1-W4/W6-W8 = CoordinatorLoop、**W5（markTransientFailure cpp:1073）= RebuildThread のまま**。
- 現行 reader: cpp:900/943/1116/1142 = CoordinatorLoop。
- **T3 移行後の intended ownership**: delivery writer = CoordinatorLoop のみ（adjudication ループに W5 相当を集約）。RebuildThread の接触は `pendingFailureSignals` の atomic 加算のみ（payload/delivery 非接触）。
- h:318-324 の「delivery 単一書込者」コメントは T3 実装後に真となる（更新は Phase 1 以降）。

## 9. 最終判定（§成果物 D — 7 条件）

```text
1. exactly-once posting            ✓（§1 表・§7。1 observation=1 signal、飽和超過は inert）
2. duplicate impossible            ✓（各観測は 1 サイトのみ通過）
3. stale/terminal/unknown safe     ✓（id-scan + Live + 飽和。slot reuse は tryInsert reset + id 単調性）
4. maximum production bounded      ✓（T3: F_max = K×L = 128 構造的・到着レート非依存）
5. selected capacity sufficient    ✓（T3 はリング不要。T1/512+128 は ✗ → 撤回）
6. overflow non-silent             ✓（飽和 counter telemetry。層 2 成立で層 3 は防御のみ）
7. W5 migration preserves semantics ✓（§4 等価性表。K=4/telemetry/stale 挙動同一）

Phase 0 verdict: GO（ただし採用契約は T3。リング 512+128 は撤回）
停止条件チェック: failure sites 6 に閉じる ✓ / 1 observation→1 signal 証明 ✓ /
duplicate post なし ✓ / F_max 有限導出 ✓（T1=21376, T3=128）/ 512+128 十分性 ✗→撤回済み /
非 silent 保証 ✓ / siteId/seq 必要性 = 不要と確定（oblId のみ）/ W5 移動の retry semantics 不変 ✓
```

**Phase 1（event contract / ring API / overflow promotion の設計固定）は「リング」ではなく T3（saturating signal + CL adjudication loop）を前提に行うこと。** リング案が必要なら F_max≥21376 の再提示が条件。

**STOP — 実装 0。Phase 1 の指示を待つ。P5/P6 非着手。**
