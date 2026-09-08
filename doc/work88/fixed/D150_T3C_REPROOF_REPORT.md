# D150 — T3c Lifecycle Single-Atomic Ownership Domain Design Re-proof（Work Report）

**Status: T3c = GO（delivery を lifecycle word に fold する設計に限定）。naive T3 / T3b の NO-GO は不変。変更 0・実装 0。**
**詳細:** `evidence/D150_T3C_LIFECYCLE_WORD_REPROOF.md` / **基準:** ConvoPeq.md `15:32:58`（mtime 再実測: より新しい src 0 件）。D147/D148/D149 は候補仮説として扱い、現行ソースから再導出。

## 中核の結論: 単一 W CAS が #5/#6 を affinity 非依存で消す
`W = {obligationId, state, pending, adjudicated, delivery}` を 16B 単一 CAS ドメインに。4 遷移すべてが同一語の expected/new を共有する。
- **#5 消滅（§6/CE-3）**: counter 加算は adjudicate CAS **内**で完結。`resolve CAS(Live→T, reset) → old fetch_add` という旧構造が**存在し得ない**（resolve 後の postSignal/adjudicate は expected.state==Live で失敗）。
- **#6 消滅・affinity 非依存化（§7/CE-4）**: adjudicate/tryInsert の expected に **id を同封**。T3b の consumer は `state==Live` だけ再判定したため reuse 後の N-Live で**通過する ABA**があったが、T3c は id タグで N を弾く。tryInsert を別スレッドと仮定しても混入不能。

## load-bearing 条件: delivery は W に fold 必須（§8）
delivery を W 外 plain に残すと、terminal 化が**別スレッド（RebuildThread Route A/C）**なので「adjudicate CAS 成功 → resolve terminal → delivery=None plain 書込」が affinity で防げず残存。これは data race でなく **semantic invariant violation**（terminal obligation の所有権フィールド書換）。指示の「inert だから GO」禁止により **W 外案は §8 NO-GO**。→ delivery を W の 2bit に取り込み、delivery=None を adjudicate CAS と同一遷移化。**これが T3c GO の前提条件**。

## 現行ソース再導出の要点（一次実測）
- W 候補 field アクセスを ccc grep + rg で二重棚卸し（完全一致）。`pending` は現行に無く T3c 新設。`consecutiveFailureCount`=adjudicated 相当。
- **実 audio callback は obligation を 0 件**（AudioEngineProcessor/DSPCoreIO/BlockDouble 等 rg hit なし）。W 触达者 = {CL, RebuildThread}。Route B resolve は CoordinatorLoop（D149 訂正をソースで再確認）。
- markTransientFailure（cpp:1059-1085）は契約文言（h:475 "Atomically performs"）に反し check-then-act + cpp:1071 のみ RebuildThread delivery 書込（W5 data race）。T3c はこれを CL 化 + W fold で両方解消。

## race matrix A-J（§9）
全 Case 合格。**D/E/F は affinity を意図的に崩して検証**し、id-tagged CAS が混入を構造排除することを確認。liveCount −1 は単一 W CAS authority（§5.5/§6）。shutdown は join 後単一スレッドで同一 W CAS 経由（§13）。

## 16B 実現可能性（§12・最後に検証）
意味幅 75bit（丸め 90bit）< 128bit。MSVC x64 の `std::atomic<16B>` は lock-pool（`is_always_lock_free==false`・cppreference 実測）。`_InterlockedCompareExchange16b`(cmpxchg16b) は x64 で 16B 整列・hardware lock-free。**実 ISR 非接触を call graph で確認済み**なので lock-pool でも RT 違反しないが、将来性から intrinsic 直実装を推奨。id は u64 維持（型波及回避）。

## 11 GO 基準
全 ✓（delivery writer ownership は delivery-in-W 条件で ✓）。NO-GO 条件の残存なし。反証探索 CE-1..CE-11 で残存反証なし。

## 次アクション
**Phase 2 実装は凍結継続（本監査 read-only・実装 0）。** D150 GO により、次は **T3c 実装設計ゲート**（実装前 inventory → patch specification → read-only implementation audit → build/CTest）。実装設計で確定すべき 8 項目（W bit layout / intrinsic 選択 / tryInsert CAS 化 / 走査コスト / R18-R20 テスト適配 / telemetry 分類 / 2 スレッド回帰 / 状態 3bit 表現）を evidence §19 に列挙。

**STOP — 実装 0。T3c 実装設計ゲートの指示を待つ。P5/P6 非着手。**
