# D147 Phase 0 — Event Production / Capacity Re-proof（Work Report）

**Status: GO — ただし `512+128` リングは撤回、再設計案 T3（per-slot saturating signal）を推奨。**
**詳細:** `evidence/D147-PHASE0_EVENT_CAPACITY_REPROOF.md` / **基準:** ConvoPeq.md `15:32:58`（D146 後版）。**変更 0・リング実装なし・W5 移動なし。**

## Failure Site Matrix（§1）
6 サイトに閉じていることを grep 実測で再確認（RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401、全 site RebuildThread）。サイト別に predicate / 1 invocation あたり post 数 / retry loop / settle 順序 / rearm 独立 / terminal 後 / duplicate を表化 — **全サイトで 1 観測=1 呼び、duplicate 経路なし**。

## K と post 数の分離（§2）
K=4 は adjudication 結果であって post 数ではない。同一 obligation の intent が queue に複数載る（C16 再 push・fillRecoveryQueue 256 前例）ため、adjudication（次 tick）までの窓で **1 obligation は K を超えて post し得る** — D145 が反証した前提を正式に採用せず。

## Capacity Derivation（§3）— リング方式 F_max ≈ 21376
processIntent が intentQueue_(4096)+quarantineFallback(1024) を **1 tick でフルドレイン**（実測）するため、窓内 submit ≤ 5120 → push ≤ 5152 → pop ≤ 5408 → T ≤ 5408、publish 試行 ≤ 10560 → P ≤ 10560、R ≤ 5408。
**F_max(T1) ≈ 21376 ≫ 640 ⇒ 指示 §5 の順序（先に F_max）に従い `512+128` を撤回。**

## 再設計 T3（§4）— F_max = K×L = 128 構造的
リングを置かず、obligation slot に **飽和 atomic `pendingFailureSignals`（cap K）** を追加:
- Producer（6 サイト）: oblId で id-scan → Live なら飽和加算（K 超過観測は inert — adjudication が K で terminal 化するため現行の 5 回目直接呼びと同一帰結）。
- Consumer（CL、processIntent 後 redrive 前）: 各 slot の signals を exchange → n>0 なら delivery=None（1 回・冪等）+ counter+=n → ≥K で resolve(Failed)。
- tryInsert で reset（consecutiveFailureCount reset と同位置）。
**現行 markTransientFailure が既に「oblId のみ」で成立していることが入力量最小の証拠**（seq/siteId は不要と確定、§6）。等価性表（§4）で K=4・telemetry・stale/unknown・slot reuse の全観点で現行意味論と同一であることを示した。

## overflow semantics 3 層分離（§5）
T1: capacity 不十分 / overflow 理論可能 / 非 silent は昇格で担保。T3: **層 2（overflow 構造的不能）が成立**し層 3 は飽和 counter の防御のみ。T1 で層 2 を得るには ≥21376（≈512KB）必要で T3 優位。

## W5 記録のみ（§8）
delivery writer は現行 W5=RebuildThread のまま（未変更）。T3 移行後は adjudication ループに集約され delivery 単一書込者=CL が真となる（h:318-324 コメント更新は Phase 1 以降）。

## 7 条件判定（§9）
```text
1 exactly-once ✓ / 2 duplicate 不可 ✓ / 3 stale/terminal/unknown 安全 ✓ /
4 production 有界 ✓（T3=128 構造的）/ 5 容量十分 ✓（T3、T1/640 は ✗→撤回）/
6 overflow 非 silent ✓ / 7 W5 移動で意味論不変 ✓
→ Phase 0 GO。Phase 1 は T3 前提で設計固定すること（リング案は F_max≥21376 再提示が条件）。
```

**STOP — 実装 0。Phase 1（event contract / adjudication loop 設計固定）の指示を待つ。P5/P6 非着手。**
