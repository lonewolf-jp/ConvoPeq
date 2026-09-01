# D151 — T3c Implementation Pre-Inventory (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only pre-implementation inventory. **Production source: 0 / Test source: 0 / Ring: 0 / W5 migration: 0 / P5/P6: 0 / Phase 2 implementation: 0。**
**基準:** `ConvoPeq.md Generated: 2026-08-31 15:32:58`（mtime 再実測: src/**.{cpp,h} でより新しいもの 0 件）。D149/D150 は候補仮説として扱い、全 site を現行ソースへ戻して実測（rg + ccc grep 二重棚卸し一致）。
**ツール:** AiDex / serena / semble / ccc grep / graphify / WSL rg・sed（rtk）/ ctx_execute / 公式資料（MicrosoftDocs cpp-docs raw・MSVC STL atomic ヘッダー raw・cppreference）。

---

## 0. 判定（先出し）

```text
D151 inventory = 確定。D152（T3c Patch Specification）進出可。
実装凍結は維持（本件は表 4 枚の確定のみ）。
D150 に対する一次資料由来の訂正 2 件 + 設計帰責 1 件（§2.4）を反映のこと。
```

---

## 表 A — W Field Contract

```cpp
struct alignas(16) RecoveryLifecycleWord {   // 16 bytes exactly; padding 常時 0（CAS は 16B 全体を比較）
    std::uint64_t obligationId;   // +0
    std::uint8_t  state;          // +8   ObligationState 全値を byte 格納（bitfield 不採用 — §2.3）
    std::uint8_t  pending;        // +9   未 adjudicate signal 0..K
    std::uint8_t  adjudicated;    // +10  adjudicate 済み 0..K-1（terminal で 0）
    std::uint8_t  delivery;       // +11  ObligationDeliveryState
    std::uint8_t  pad[4];         // +12  常に 0
};
static_assert(sizeof == 16 && alignof == 16 && is_trivially_copyable_v);
```

| field | bit 幅（実装 byte） | 意味 | 初期値 | Live 遷移 | terminal 遷移 | writer | reader |
|---|---|---|---|---|---|---|---|
| `obligationId` | 64 | 単調 identity（`nextId_` ++、h:398/451・CL 単一書込）。0=無効（cpp:1018 early-return） | 0 | tryInsert: N | **不変**（resolve は id を保持 — h:423-436 実測） | CL（tryInsert） | 全走査（scan/CAS expected） |
| `state` | 8（意味 3bit・8 値） | ObligationState h:306-316 | NoObligation | tryInsert→Live（h:401）/ coalesce Live→Live（cpp:932） | resolve→Success/Failed/Stale/Shutdown（cpp:1027-1030）/ adjudicate 枯渇→Failed | CL+RebuildThread+shutdown（CAS 経由） | findByKey h:384 / tryInsert h:397 / markT cpp:1066 / redrive cpp:1110,1136 |
| `pending` | 8（意味 3bit・0..4） | **新設**（現行に無し — 改名対象ではない） | 0 | postSignal +1（Live∧<K） | →0（adjudicate drain / resolve） | RebuildThread（postSignal） | CL（adjudicate） |
| `adjudicated` | 8（意味 3bit） | 現行 `consecutiveFailureCount`（h:361）の置換。adjudicate 済み連続失敗 | 0 | adjudicate +=pending（<K 時のみ残る） | →0（resolve/adjudicate 枯渇） | CL（adjudicate）+ tryInsert リセット | CL / telemetry accessor h:542（**テスト直読 0 実測**） |
| `delivery` | 8（意味 2bit・3 値 h:325-329） | transport 住処。**W 内へ fold（D150 §8 load-bearing 条件）** | None | tryInsert None / submit Transport・Durable / adjudicate None / redrive 付着 | **不変**（resolve は delivery 保持） | **CL のみ**（全書込が W CAS 内） | CL のみ（cpp:901/944/1112/1138） |

**列挙値の到達可能性（実測）**: 本番で格納されるのは NoObligation（初期）/Live/ResolvedSuccess/ResolvedFailed/ResolvedStaleSuperseded/ShutdownDiscarded の 6 値。`ResolvedRetry` は**格納不能**（cpp:1023-1024 で early-return・T-R18-10 が Live 維持をアサート）。`ResolvedSuperseded` は dormant（G-4.1）。→ byte 格納で全 8 値を失わずに保持（将来の Phase-II 遷移に備える）。
**K**: `kMaxObligationConsecutiveFailures = 4`（h:369）。pending 飽和上限=K、adjudicated 非 terminal 時最大 K-1。
**capacity**: 32（h:363）→ `std::array<LogicalRecoveryObligation,32>` の各要素 alignas(16) で stride 16 保証。C4324 pragma は h:971-972/994 に既存。

---

## 表 B — Transition Matrix

| 遷移 | Expected W | New W | CAS owner（スレッド） | 失敗時挙動 | liveCount 効果 |
|---|---|---|---|---|---|
| `postSignal(O)` | `{O, Live, p<K, a, d}` | `{O, Live, p+1, a, d}` | RebuildThread（6 サイト） | id≠O→no-op(stale/reuse) / state≠Live→no-op(terminal) / p≥K→no-op(saturated) — **reason 別 telemetry** | なし |
| `adjudicate` 非枯渇 | `{O, Live, c>0, a, d}` | `{O, Live, 0, a+c, None}` | CL（runCoordinatorPhase） | CAS 競合→再試行 / state≠Live→drop+telemetry | なし |
| `adjudicate` 枯渇（a+c≥K） | `{O, Live, c>0, a, d}` | `{O, Failed, 0, 0, d}` | CL | 敗者（resolve 先行）→no-op | **−1（勝者のみ）** |
| `resolve(O,T)` | `{O, Live, p, a, d}` | `{O, T, 0, 0, d}` | CL（Route B :354）/ RebuildThread（Route A :320・Route C :371）/ shutdown（cpp:1352） | id 不一致/既 terminal→false（冪等） | **−1（勝者のみ）** |
| `tryInsert(N)` | `{oldId, ≠Live, 0, 0, d}`（観測 terminal/NoObligation 語） | `{N, Live, 0, 0, None}` | CL（唯一 site cpp:947） | CAS 競合→次 slot 探索 | **+1（勝者のみ）** |
| coalesce（既存維持） | `{O, Live, p, a, d}` | 同一語（identity CAS） | CL（cpp:932） | 敗者→tryInsert 経路（D27.2 不変） | なし |
| submit/redrive delivery 付着 | `{O, Live, p, a, None}` | `{O, Live, p, a, Transport/Durable}` | CL | postSignal 競合→再試行（pending 変化で CAS 失敗しうる） | なし |

**禁止条件（明文化）**: `CAS 成功 → state 再判定 → 別 atomic counter/delivery 更新` の旧構造（markTransientFailure cpp:1066→1071→1074）を一切残さない。counter/delivery の更新は必ず W CAS 内で完結。

---

## 表 C — Source Patch Inventory（実 site 固定）

| # | file | function | site | 現行動作 | T3c 変更 | risk |
|---|---|---|---|---|---|---|
| C1 | ISRRuntimePublicationCoordinator.h | LogicalRecoveryObligation | h:347-362 | atomic id/state + plain delivery + atomic counter | 4 語を `RecoveryLifecycleWord W` に統合。identity/handle/epoch/intentId/buildSource/recoveryGeneration は plain 維持（CL 単一書込） | 構造 size/align（static_assert 追加） |
| C2 | 同 | findByKey | h:382-389 | state.load + identity== | W.load(acquire).state==Live ∧ identity | 低（CL のみ） |
| C3 | 同 | tryInsert | h:392-413 | plain store 列 + state.store(Live,release) | payload 先書→**W CAS**（観測非 Live 語→{N,Live,0,0,None}）→liveCount++ | publication 順序（§7） |
| C4 | 同 | resolve | h:423-436 | id scan + state CAS + counter store(0) 別語 | W CAS {id,Live}→{id,T,0,0,d}（reset 同語化） | 低 |
| C5 | 同 | consecutiveFailureCount(i) | h:444-446/542-544 | counter.load | W.load.adjudicated（telemetry 専用） | 低 |
| C6 | ISRRuntimePublicationCoordinator.cpp | submitRecoveryRequest coalesce | cpp:927-940 | state CAS(Live→Live) | W 全文 identity CAS（D27.2 意味不変） | 低 |
| C7 | 同 | submit delivery 付着 | cpp:979/996/1001 | plain 書込 | W CAS（delivery フィールドのみ変化、retry ループ要） | 中（postSignal と CAS 競合→再試行必須） |
| C8 | 同 | markTransientFailure | cpp:1059-1085 | scan→Live 判定→delivery plain→fetch_add→resolve | **分割**: `postRecoveryFailureSignal(oblId)`（pending++ CAS のみ）+ `adjudicateRecoveryFailureSignals()`（CL ループ）。`markTransientFailure` は TEST-ONLY 互換ラッパ（postSignal+即 adjudicate）へ降格 | **責務境界（§4）** |
| C9 | 同 | redriveDeferred* | cpp:1106-1116/1122-1167 | state/delivery plain 読書 | W load/CAS 化（付着=delivery CAS） | 中（C7 と同一パターン） |
| C10 | 同 | resolveRecoveryObligation | cpp:1016-1041 | switch→table.resolve | **変更不要**（Retry early-return cpp:1023 維持） | なし |
| C11 | 同 | shutdown close | cpp:1349-1353 | id load→resolve | 変更不要（resolve が W CAS 化される） | なし |
| C12 | AudioEngine.Threading.cpp | runCoordinatorPhase | :258-272（processIntent 終端〜redrive :272 の間） | — | **adjudicateRecoveryFailureSignals() を挿入**（processIntent 後・redrive 前 — D148 §1-D 配置） | 低（同一 tick 内順序） |
| C13 | RuntimePublicationOrchestrator.cpp | trySubmitImpl / submitPublishRequest | :311/:401 | markTransientFailure | postRecoveryFailureSignal 置換 | 低（機械的） |
| C14 | AudioEngine.RebuildDispatch.cpp | rebuildThreadLoop | :1006/1033/1091/1115 | markTransientFailure | 同上 | 低 |
| C15 | ISRRuntimePublicationCoordinator.h | telemetry | h:1006-1016 近傍 | recoveryRetryExhaustedCount_（cpp:1079 で resolve 成否前に増加） | **exhausted++ を枯渇 CAS 勝者のみへ移動**（過計上修正）+ signalSaturated/signalDropped(reason) 新設 | observability のみ |
| C16 | 同 | delivery 注釈 | h:318-324 | 「CoordinatorLoop-only・non-atomic」— **現行偽**（cpp:1071 RebuildThread） | W fold 後に「delivery field は W 内・CL 専用書込」へ真化 | なし |

**消去確認（§5 目標）**: cpp:1071 の RebuildThread delivery plain 書込は C8 分割で**消滅**。実装後 grep `\.delivery\s*=` は W 内 CAS 経由のみ（plain 書込 0）で機械検証可能。

---

## 表 D — Test Adaptation Inventory

実測: `markTransientFailure` 呼び出しはテスト側 **約 30 site・ISRSemanticValidationTests.cpp のみ**（AudioEngineHarness/invariant_INV3_INV5 に hit ゼロ）。テストからの `delivery`/`consecutiveFailureCount`/`ObligationState::` **直接アクセス 0 件**（全 grep 実測）→ 適配面は「同期アサートの意味」に限定される。

| test | 旧暗黙前提 | T3c semantics | 必要適配 |
|---|---|---|---|
| T-R18-1/2（:1785/:1809） | markT 直後 Live∧exhausted==0、coalesce 可 | postSignal だけでも成立（pending++ は Live 維持） | **なし**（ラッパ/素 postSignal 両可） |
| T-R18-3（:1827） | markT 直後 delivery==None → redrive 候補 | delivery=None は **adjudicate の帰結** | 呼び出し間に adjudicate 実行（ラッパで自動） |
| T-R18-5（:1881） | 4 回目 markT 直後 Live==0・exhausted==1 | 枯渇は adjudicate CAS | 同上 |
| T-R18-6/7/8（:1902/:1922/:1939） | 失敗経過後 resolve(Published/Shutdown/Stale) | W CAS 直列化で同一帰結 | なし |
| T-R18-9（:1956） | stranded repair（delivery None→redrive） | 同 T-R18-3 | ラッパ |
| T-R18-10（:1979） | resolve(Retry) は Live 維持 | cpp:1023 early-return 不変（C10） | なし |
| T-R18-11（:1995） | 非 Live/unknown/0 は no-op | postSignal CAS 失敗=no-op | なし |
| T-R18-12（:2020） | slot reuse で counter=0 | tryInsert W リセット（{N,Live,0,0,None}） | なし |
| T-R20-1/2/3（:2056/:2077/:2098） | 1 呼び=1 加算・4 目枯渇・delivery None | 「1 観測=1 pending 加算、adjudicate で一括 adjudicated」— **ラッパ下では 1:1 同一** | ラッパ |
| T-P2-1（:1318） | failure 単体では wake なし | postSignal/adjudicate は attach しない → wake なし | なし（ラッパ） |
| T-P2-5（:1336） | K=4 終端・終端後 wake 源なし | 同 | ラッパ |
| T-P3-1..5 + openDurableWindow（:1450-1534+） | markT で Transport/Durable→None（窓生成） | delivery=None は adjudicate CAS 内 | ラッパ |

**意味論の置換（指示の核心）**: 旧テストの暗黙前提 = 「markTransientFailure が**呼び出しスレッドで同期的に adjudicate まで完了**」。T3c での置換先 = 「postSignal（transport 観測）→ **CL adjudicate（所有権遷移）**」。テストハーネスは単一スレッドで CL 役割を兼ねるため、**TEST-ONLY ラッパ（postSignal+即 adjudicate）**で全 40 テストの意味を保存できる（D148 §12 契約の継承）。production 6 サイトはラッパを通らない（grep 強制: production 内 `markTransientFailure` 呼び出し 0）。
**新規テスト（D152 で設計）**: (i) postSignal 単体では delivery/adjudicated 不変、(ii) pending=3 の一括 drain（1 CAS）、(iii) 飽和 inert、(iv) terminal 後 adjudicate drop telemetry、(v) 2 スレッド（postSignal ∥ adjudicate ∥ resolve）の identity CAS 安全性。

---

## 2. 16B CAS primitive 契約（§2・公式資料で確定）

### 2.1 正式名称の訂正
D150/D148 が書いた `_InterlockedCompareExchange16b` は通称。**公式 intrinsic 名は `_InterlockedCompareExchange128`**（MicrosoftDocs cpp-docs `docs/intrinsics/interlockedcompareexchange128.md` 実測取得）。D152 以降はこの名称を用いること。

### 2.2 公式仕様（取得一次資料より）
- **構文**: `unsigned char _InterlockedCompareExchange128(volatile __int64* Destination, const __int64* Comparand, __int64* Exchange)` — Destination は 2×64bit を 128bit フィールドとして扱う。**交換成立なら non-zero を返し、Comparand バッファは元の Destination 値で上書きされる**（失敗時 retry ループが再 load 不要）。
- **整列**: 「The destination data must be **16-byte aligned** to avoid a general protection fault.」— **GP fault（クラッシュ）なので alignas(16) は必須契約**。
- **アーキテクチャ**: x64, ARM64（`_acq/_rel/_nf` 変種は ARM64 のみ。x64 の基本形は lock プレフィックス付き cmpxchg16b = **フルメモリバリア**）。
- **copyability**: trivially-copyable POD 要件（static_assert で担保）。Debug/Release で ABI 同一（intrinsic 直接）。

### 2.3 `std::atomic<16B>` を採用しない根拠（一次ソース実測）
MSVC STL `stl/inc/atomic`（GitHub raw 取得）:
```cpp
template <size_t _TypeSize>
constexpr bool _Is_always_lock_free = _TypeSize <= 8 && (_TypeSize & (_TypeSize - 1)) == 0;
```
→ **16 バイトは lock-free 対象外（lock pool 経路）**。cppreference の「never or sometimes lock-free なら false」と整合。よって W は `std::atomic<RecoveryLifecycleWord>` にせず、**16B 整列ストレージ + `_InterlockedCompareExchange128` ラッパ**で実装する。

### 2.4 設計帰責: load は advisory、CAS のみ commit（新規確定事項）
`_InterlockedCompareExchange128` は RMW を原子化するが、**16B の素の load は 2×u64 read であり torn read があり得る**（id と state が別世代の組になる窓）。本設計の全意思決定経路（scan→CAS）は CAS が**全文を expected 比較**するため、torn load は CAS 失敗→再試行/再スキャンで自己修復される（偽の成功は生成不能）。**契約**: 「W の load は advisory（分断可能性を許容）、commit は CAS のみ。telemetry 専用 read は観測値の整合性を保証しない」。この規律を D152 の全擬似コードに明記すること。

### 2.5 lock-pool 可否（短絡しない判定 — 指示 §7）
実 call graph 実測（D150 §2.3 継承・本件で再確認）: W 触达者 = {CoordinatorLoop, RebuildThread}。**実 audio callback（AudioEngineProcessor/DSPCoreIO/BlockDouble 等）の obligation アクセス 0 件**（rg 実測）。よって仮に lock-pool 経路でも RT 制約は違反しない — が、§2.3 の通り std::atomic<16B> は lock pool であり、**intrinsic 直実装（lock-free）を推奨**（将来の ISR 接触追加に対する構造防衛）。

---

## 3. markTransientFailure 責務境界（§4）

```text
現行（cpp:1059-1085・RebuildThread・check-then-act 4 段）
  ├─ 観測責任（delivery=None + counter++ + 枯渇 terminal）
T3c 分割
  ├─ postRecoveryFailureSignal(oblId)   [RebuildThread・6 サイト]  = W CAS pending++ のみ。payload/state/delivery/liveCount 非接触
  ├─ adjudicateRecoveryFailureSignals() [CL・runCoordinatorPhase 挿入（C12）] = slot 走査、pending>0 を単一 CAS で adjudicated 化（枯渇なら同一 CAS で terminal+−1）
  └─ markTransientFailure(oblId)        [TEST-ONLY 互換ラッパ]      = postSignal + 即 adjudicate（呼び出しスレッド実行）
```
production からの `markTransientFailure` 呼び出しは **0**（6 サイト置換後・grep 強制）。ラッパの TEST-ONLY 化は `setRetireBacklogCount` の TEST-ONLY 前例（h:147）に倣いコメントで固定。adjudicate のスレッドアフィニティ検証（CL id jassert）は D152 決定事項（§6 open-2）。

---

## 4. delivery 全アクセス棚卸し（§5・GO 条件の再確認）

**本番 11 サイト（実測）**: 書込 = h:402（tryInsert）/ cpp:979,996,1001（submit）/ cpp:1071（markT・**RebuildThread — 消去対象**）/ cpp:1159,1167（redrive）。読 = cpp:901,944（submit）/ cpp:1112,1138（redrive）。resolve は delivery 非接触（h:423-436 実測）。
**テスト直アクセス 0**（grep 実測）。
**T3c 後**: 全書込が W CAS 内・全 owner=CL（cpp:1071 は C8 で消滅）。postSignal は delivery を触らない（expected/new 同値保持）。→ 「delivery = W only・writer=CL only」は実装後 grep で機械検証可能（`\.delivery\s*=` の plain 書込 0、`slot(...)\.delivery` 直接アクセス 0）。

---

## 5. liveCount_ single-authority（§6）

| 操作 | site | gate |
|---|---|---|
| +1 | h:408（tryInsert） | W CAS 勝者のみ（T3c） |
| −1 | h:430（resolve） | W CAS(Live→T) 勝者のみ |
| −1 | **新**: adjudicate 枯渇 CAS | 勝者のみ |
| 読 | h:393（capacity）/ h:439 / cpp:556（isFullyDrained）/ テスト getter | — |

duplicate decrement 経路: resolve と adjudicate 枯渇が同一 slot を競合 → 両者 expected.state==Live → **片方の CAS のみ成功** → fetchSub 1 回。shutdown（cpp:1352）も同一 resolve 経路。**単一 authority 成立**。capacity 読（h:393）の stale 読取は無害（実 gate は slot 走査の state）。

---

## 6. tryInsert CAS 化の publication ordering（§7）

```text
1. payload plain 書込（identity/handle/epoch/intentId/buildSource/recoveryGeneration）  ← W の Live 化より sequenced-before
2. W CAS: {oldId,≠Live,0,0,d} → {N, Live, 0, 0, None}   （release 相当 — x64 では CAS 自体がフルバリア）
3. liveCount_ fetchAdd(release)
```
- 現行 h:399-401（id relaxed→state Live release）と同型だが、T3c では **id が W 内**のため「id 可視 ∧ Live 可視」が単一語で原子（現行より強い）。
- 別 thread reader（resolve/postSignal の scan）は W.load(acquire)→CAS 経路のみで Live を解釈するため、payload 参照は常に Live 観測後（CL 内 program order）。
- `nextId_`/`nextRecoveryGeneration_` は CL 単一書込維持（h:451-452）。

---

## 7. D152 への open items（実装設計で確定）

1. **ラッパ vs テスト書き換え**: 表 D の TEST-ONLY ラッパ案を基本とし、production 呼び出し 0 の grep 強制を patch spec に条文化。
2. **adjudicate のスレッド検証**: CL thread-id 登録 + jassert の可否（redriveWakePending_ は現状 id 検証なし — 前例確認要）。
3. **torn-load 規律の明文化**（§2.4）を全擬似コードに反映。
4. **exhausted++ の CAS 勝者後移動**（C15）— T-R18-5/12/20-3 のアサート値は不変（過計上経路がテストされないこと実測済）。
5. **W byte 格納 vs bitfield**: byte 推奨（§表 A）。pad[4] 常時 0 の初期化経路（コンストラクタ `{}`）を固定。
6. **alignas(16) 波及**: LogicalRecoveryObligation 全体を alignas(16)（std::array stride 保証）。C4324 pragma 既存（h:971）。
7. **新 telemetry reason 分類**（saturated/dropped(stale/terminal/reuse-overwrite)）のカウンタ名。
8. **coalesce identity CAS（C6）と G-4.3-R 監査不変条件**の整合再確認（全文 CAS でも post-CAS mutation-free は維持）。
9. **新規テスト 5 本**（表 D 末尾）のハーネス設計（production フック不要 — 2 スレッドは postSignal/adjudicate 公開 API で構成可能）。
10. **R18/R20 以外の 40 テスト**（capacity/durable/wake/retire 系）は W 内部表現に非依存（getter 経由実測）→ 変更不要見込み。

---

## 8. 結論

- 表 A-D により T3c の全波及範囲（構造 1・遷移 7・patch 16・テスト適配 約 30 site 中 実変更不要〜ラッパ依存）を確定。
- **cpp:1071 の RebuildThread delivery 書込は消去可能**（C8 分割で確認）。
- **liveCount_ 単一 authority・resolve/adjudicate 相互排他・reuse の affinity 非依存保護**は表 B の CAS 仕様で担保。
- primitive は `_InterlockedCompareExchange128`（16B 整列必須・x64 lock-free・失敗時 comparand 更新で再試行容易）。`std::atomic<16B>` は lock pool のため不採用（MSVC STL 一次ソース実測）。
- **Phase 2 実装凍結は維持**。次は D152 — T3c Patch Specification（ここで初めて exact struct/bit layout/wrapper API/擬似コードを固定）。
