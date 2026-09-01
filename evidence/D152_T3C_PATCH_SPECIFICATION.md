# D152 — T3c Patch Specification (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only patch specification. **Production source: 0 / Test source: 0 / build / CTest: 非着手。**
**基準:** ConvoPeq.md `2026-08-31 15:32:58`（D151 で mtime 実証済み、本件で追加変更なし）。公式一次資料: MicrosoftDocs cpp-docs `interlockedcompareexchange128.md`（raw 取得）、MSVC STL `stl/inc/atomic`（raw 取得）。
**目的:** D151 inventory を「実装者が解釈を挟まず適用できる」exact specification へ変換する。**本文書のコードブロックは仕様であり、実装ではない。**

---

## 0. Gate 対応表

| Gate | 内容 | 本仕様節 |
|---|---|---|
| G152-01 | W exact layout | §1 |
| G152-02 | `_InterlockedCompareExchange128` wrapper API | §2 |
| G152-03 | memory ordering | §2.4/§3 |
| G152-04 | torn-load/advisory-load 規律 | §3 |
| G152-05 | 7 CAS 遷移 expected/new | §5 T1-T7 |
| G152-06 | liveCount authority | §5 各表行 + §5.8 |
| G152-07 | delivery authority | §6 |
| G152-08 | markTransientFailure 3 責務分離 | §7 |
| G152-09 | production call site 6 件置換先 | §7.3 |
| G152-10 | thread affinity | §8 |
| G152-11 | telemetry reason/name | §9 |
| G152-12 | 既存テスト適応 | §10.1 |
| G152-13 | 新規 5 テスト | §10.2 |
| G152-14 | D153 機械検証条件 | §11 |

---

## 1. `RecoveryLifecycleWord` exact 定義（G152-01）

配置先: `ISRRuntimePublicationCoordinator.h`、`namespace convo::isr` 内・`RuntimeIntentCoordinator` クラス定義**前**（テーブルテンプレートから参照されるため）。

```cpp
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#else
#error "T3c lifecycle word requires MSVC x64 (_InterlockedCompareExchange128)"
#endif

// ★ D152 (T3c): single atomic ownership domain for one recovery obligation.
//   ALL lifecycle decisions (identity / liveness / un-adjudicated signals /
//   adjudicated failures / transport residency) commit ONLY via full 16-byte CAS.
struct alignas(16) RecoveryLifecycleWord {
    std::uint64_t obligationId = 0;   // +0  (low quad; 0 = invalid/none)
    std::uint8_t  state        = 0;   // +8  (ObligationState, full enum byte)
    std::uint8_t  pending      = 0;   // +9  (un-adjudicated signals, 0..K)
    std::uint8_t  adjudicated  = 0;   // +10 (adjudicated failures, 0..K-1 while Live)
    std::uint8_t  delivery     = 0;   // +11 (ObligationDeliveryState)
    std::uint8_t  pad[4]       = {};  // +12 (ALWAYS zero — CAS compares all 16 bytes)
};
static_assert(sizeof(RecoveryLifecycleWord) == 16);
static_assert(alignof(RecoveryLifecycleWord) == 16);
static_assert(std::is_trivially_copyable_v<RecoveryLifecycleWord>);
static_assert(std::is_standard_layout_v<RecoveryLifecycleWord>);
```

**固定事項**:
- byte 格納（bitfield 不採用 — ABI/パディングの解釈余地を排除）。`state`/`delivery` は enum class の underlying `uint8_t` を `__cast` せず `static_cast<std::uint8_t>` で格納。
- `pad[4]` は**常時 0**。生成は `{}` 値初期化 + フィールド代入のみ（pad に触れる経路を作らない）。CAS 比較は 16B 全文 → pad が非 0 の語は存在してはならない。
- 本番で格納される state 値は 6（NoObligation/Live/ResolvedSuccess/ResolvedFailed/ResolvedStaleSuperseded/ShutdownDiscarded）。`ResolvedRetry`（格納不能・cpp:1023）と `ResolvedSuperseded`（dormant）も byte で表現可能。
- K = `kMaxObligationConsecutiveFailures = 4`（h:369 不変）。pending 最大値 K、adjudicated 非 terminal 最大 K-1。

---

## 2. CAS wrapper（G152-02）

公式シグネチャ（MicrosoftDocs 実測）:
```c
unsigned char _InterlockedCompareExchange128(
    __int64 volatile * Destination,   // [in,out] 16-byte aligned（違反は GP fault）
    __int64 ExchangeHigh,             // 上位 64bit（bytes 8..15）
    __int64 ExchangeLow,              // 下位 64bit（bytes 0..7）
    __int64 * ComparandResult);       // [in,out] 失敗時、現在の Destination で上書き
// return: 0 = 不一致（交換なし） / 非0 = 交換成立
```

```cpp
// Advisory load — 2×u64 read、torn あり得る（§3 規律）。commit には使えない。
inline RecoveryLifecycleWord loadLifecycleAdvisory(const RecoveryLifecycleWord& src) noexcept {
    RecoveryLifecycleWord out;
    out.obligationId = src.obligationId;   // u64/u8 は単一コピー原子
    out.state        = src.state;
    out.pending      = src.pending;
    out.adjudicated  = src.adjudicated;
    out.delivery     = src.delivery;
    out.pad[0]=out.pad[1]=out.pad[2]=out.pad[3]=0;
    return out;
}

// Commit CAS — 唯一の遷移手段。expected は失敗時「現在値」に更新される（再試行は再 load 不要）。
inline bool casLifecycle(RecoveryLifecycleWord& expectedInOut,
                         const RecoveryLifecycleWord& desired,
                         RecoveryLifecycleWord& dst) noexcept {
    auto* dest = reinterpret_cast<volatile __int64*>(&dst);
    auto* cmp  = reinterpret_cast<__int64*>(&expectedInOut);
    const auto* exg = reinterpret_cast<const __int64*>(&desired);
    return _InterlockedCompareExchange128(dest, exg[1], exg[0], cmp) != 0;
}
```

**固定事項**:
- 関数自由形（member 不要）。`dst` は `slot(i).lifecycle`。little-endian x64: `exg[0]=bytes0-7=obligationId`、`exg[1]=bytes8-15`。
- **retry 規則**: `casLifecycle` 失敗時 `expectedInOut` に現在値が入る → ループは**再 load せず**それを次の expected として再評価する（§5 全擬似コードの `w = expectedInOut;` 相当）。
- 戻り値: true=交換成立（この遷移の勝者）、false=不一致（現在値で再評価 or 離脱）。
- 直接使用は wrapper 内 1 箇所のみ（D153 V5）。

### 2.4 memory ordering（G152-03）
- x64 の基本形 `_InterlockedCompareExchange128` は `lock cmpxchg16b` = **フルメモリバリア**（`_acq/_rel/_nf` 変種は ARM64 のみ。x64 では使わない）。→ 全 CAS は acq_rel を超える順序強度をハードウェアが保証。**追加 fence 不要**。
- `liveCount_` は現行 helper 維持: `convo::fetchAddAtomic/fetchSubAtomic(release)`・`consumeAtomic(acquire)`（AtomicAccess.h:91/60 実測）。CAS 勝者の program-order 後なので、CAS のフルバリアが payload 可視性も含めて担保。
- advisory load に fence は不要（load 単独で何も commit しないため）。

---

## 3. 「load は advisory、CAS のみ commit」規律（G152-04）

**全擬似コードに適用される単一規律**:
1. `loadLifecycleAdvisory` の結果（id/state/pending/adjudicated/delivery の組）は**分断され得る**（id=O・state=N 由来等）。分断 load 自体は禁止しない。
2. 分断 load 結果を**条件に遷移を commit してはならない**。commit は必ず `casLifecycle`（16B 全文比較）の成功のみ。
3. 走査（scan）は advisory load で候補 slot を絞り、**CAS が最終判定**する: expected が現在値と 1 バイトでも違えば失敗 → 失敗時は更新された expected を再評価（§2 retry 規則）。
4. 帰結: 偽の成功（別世代の混成語への commit）は**生成不能**。torn load は CAS 失敗として自己修復される。
5. telemetry/peek 専用 read は整合性を保証しない（観測値に非決定の窓があり得る旨をコメント明記）。

---

## 4. `LogicalRecoveryObligation` 変更仕様（D151 C1/C5 対応）

```cpp
// 旧（h:347-362）:
struct LogicalRecoveryObligation {
    std::atomic<LogicalRecoveryObligationId> id{0};          // → 削除（W 内へ）
    CoalesceIdentity identity{};                              // 維持（plain・CL）
    std::atomic<ObligationState> state{ObligationState::NoObligation}; // → 削除（W 内へ）
    DSPHandle handle{};                                       // 維持（plain・CL）
    PublicationEpoch epoch{0};                                // 維持
    std::uint64_t intentId = 0;                               // 維持
    convo::RuntimeBuildSnapshot buildSource{};                // 維持
    RecoveryGeneration recoveryGeneration{0};                 // 維持
    ObligationDeliveryState delivery{...::None};              // → 削除（W 内へ）
    std::atomic<std::uint8_t> consecutiveFailureCount{0};     // → 削除（W.adjudicated が置換）
};

// T3c:
#pragma warning(push)
#pragma warning(disable : 4324)   // alignas(16) による padding 警告（C4324）— h:971 の前例に倣う
struct LogicalRecoveryObligation {
    RecoveryLifecycleWord lifecycle{};   // ★ 単一所有権ドメイン（id/state/pending/adjudicated/delivery）
    CoalesceIdentity identity{};
    DSPHandle handle{};
    PublicationEpoch epoch{0};
    std::uint64_t intentId = 0;
    convo::RuntimeBuildSnapshot buildSource{};
    RecoveryGeneration recoveryGeneration{0};
};
#pragma warning(pop)
```

- **W 外 plain 維持**: `identity/handle/epoch/intentId/buildSource/recoveryGeneration`（全て CL 単一書込・CL 単一読取 — D151 §2.2 実測）。
- **publication ordering 明文化**: payload 書込は必ず `lifecycle` の **Live 公開 CAS より sequenced-before**（§5 T4）。Live 観測後 reader が payload を読んでも、CAS のフルバリアにより書込が可視。terminal slot の payload 上書きは安全（terminal payload の跨スレッド reader ゼロ — D151 §6）。
- accessor 変更: `consecutiveFailureCount(i)`（h:444-446）→ **`adjudicatedFailureCount(i)`**（`loadLifecycleAdvisory(slots_[i].lifecycle).adjudicated` 返却）。coordinator 側 `recoveryConsecutiveFailureCount(i)`（h:542）→ **`recoveryAdjudicatedFailureCount(i)`**。テスト使用箇所 0 実済のため波及なし。
- `slot(i)` アクセサ維持（返却型に W を含む member が増えるのみ）。

---

## 5. 7 CAS 遷移の擬似コード（G152-05/06）

全経路共通: `W = slot(i).lifecycle`、`K = kMaxObligationConsecutiveFailures`。`cas` は §2、`load` は §3 規律。

### T1 `postRecoveryFailureSignal(std::uint64_t oblId)` — owner: RebuildThread
```text
if oblId == 0: signalDroppedInvalid++; return
for i in 0..31:
    w = load(slot(i).lifecycle)
    if w.obligationId != oblId: continue
    // id 候補発見 — commit は CAS のみ
    loop:
        if w.state != Live:        signalDroppedTerminal++; return
        if w.pending >= K:         signalSaturated++;       return
        expected = w
        desired  = { oblId, Live, w.pending + 1, w.adjudicated, w.delivery }
        if cas(expected, desired, slot(i).lifecycle): return   // posted
        // 失敗: expected = 現在値
        if expected.obligationId != oblId: signalDroppedStale++; return  // reuse 済み
        w = expected; continue loop
// id 一致 slot なし
signalDroppedStale++; return
```
- liveCount 効果: **なし**。payload/state/delivery 非接触（desired は w のコピーから pending のみ変化）。
- 停止条件: posted / terminal / saturated / stale のいずれかで必ず return（無限ループなし — contention は CL の drain/attach のみ、pending は K で頭打ち）。

### T2 `adjudicateRecoveryFailureSignals()` — owner: CL（§8）
```text
for i in 0..31:
    w = load(slot(i).lifecycle)
    if w.pending == 0: continue
    loop:
        if w.state != Live: break        // terminal: pending は resolve が 0 化（T3 で計上）
        const uint8 a2 = w.adjudicated + w.pending
        expected = w
        if a2 >= K:   // 枯渇 — terminal 化と counter 消費を同一 CAS
            desired = { w.obligationId, ResolvedFailed, 0, 0, w.delivery }
            if cas(expected, desired, slot(i).lifecycle):
                liveCount_.fetchSub(1, release)
                recoveryRetryExhaustedCount_.fetchAdd(1, release)   // ★ 勝者のみ（過計上不変条件）
                break
        else:         // 非枯渇 — drain+加算+delivery=None を同一 CAS
            desired = { w.obligationId, Live, 0, a2, None }
            if cas(expected, desired, slot(i).lifecycle): break
        if expected.obligationId != w.obligationId: break   // reuse: 旧 O の残 signal は無効
        w = expected; continue loop
```
- **禁止構造の不在**: `CAS → state 再判定 → 別 atomic 更新` は存在しない（加算も None 化も terminal 化も CAS 内）。
- resolve との競合: 両者 expected.state==Live → 同一語版で**片方のみ成功**、liveCount−1 は 1 回（§5.8）。

### T3 `resolve(id, terminalState)` — owner: 任意（CL Route B / RebuildThread Route A,C / shutdown）
```text
if id == 0: return false
for i in 0..31:
    w = load(slot(i).lifecycle)
    if w.obligationId != id: continue
    loop:
        if w.state != Live: return false          // 既 terminal（冪等）
        expected = w
        desired  = { id, terminalState, 0, 0, w.delivery }   // pending/adjudicated ゼロ、delivery 保持
        if cas(expected, desired, slot(i).lifecycle):
            if w.pending > 0: signalDroppedTerminal += w.pending   // in-flight signal 破棄の観測
            liveCount_.fetchSub(1, release)
            return true
        if expected.obligationId != id: return false   // 競合中に reuse
        w = expected; continue loop
return false
```
- **STOP #5 構造消滅**: counter reset が語内。T2 の加算は T3 成功後の expected.state!=Live で発火不能。

### T4 `tryInsert(key, handle, epoch, intentId, buildSource)` — owner: CL（唯一 site cpp:947）
```text
if liveCount_.load(acquire) >= Capacity: return nullopt      // advisory（実 gate は次）
for i in 0..31:
    w = load(slot(i).lifecycle)
    if w.state == Live: continue
    const id = ++nextId_                                      // CL 単一書込
    // (1) payload plain 書込 — Live 公開 CAS より sequenced-before
    slot(i).identity = key
    slot(i).handle = handle; slot(i).epoch = epoch; slot(i).intentId = intentId
    slot(i).buildSource = buildSource
    slot(i).recoveryGeneration = ++nextRecoveryGeneration_
    // (2) W CAS: 観測非 Live 語 → {N, Live, 0, 0, None}
    expected = w
    desired  = { id, Live, 0, 0, None }
    if cas(expected, desired, slot(i).lifecycle):
        // (3) liveCount は Live 公開勝者のみ増加
        liveCount_.fetchAdd(1, release)
        return i
    // 失敗（想定外 — 挿入者は CL のみ、terminal 語への他書込なし）: expected 再評価で次 slot
return nullopt
```
- **署名変更**: 現行 `tryInsert(key)` → payload 5 引数追加。**呼び出し側 cpp:954-959 の post-insert payload 書込は本関数内へ移動**（§7 の順序要求を充足）。`oblId` は `slot(i).lifecycle.obligationId`（advisory load、CL 自身直後）で取得（現行 cpp:955 相当維持）。

### T5 coalesce identity CAS — owner: CL（cpp:927-940 置換）
```text
existing = findByKey(cid)                       // advisory: state==Live ∧ identity==key
if existing != npos:
    w = load(slot(existing).lifecycle)
    loop:
        if w.state != Live: coalesceOnLive = false; break   // 真の terminal → tryInsert 経路
        expected = w
        desired  = w                                          // 全文 identity CAS（D27.2 線形化）
        if cas(expected, desired, slot(existing).lifecycle): coalesceOnLive = true; break
        w = expected; continue loop              // ★ contention（pending/delivery 変化）は再試行
                                                 //   — tryInsert へ落ちない（L 二重計上の元）
```
- **D151 からの新規固定**: 現行の 1 発 CAS 失敗＝即 tryInsert は、T3c では postSignal の pending 変化と衝突し誤 NEW 化（ΔL 誤増）するため、**再試行ループ必須**。terminal 判定は再 load した expected の state で行う。
- COALESCE 分岐の `oblId` 取得（cpp:938）・delivery 読（cpp:944）は advisory load。

### T6 submit delivery attach — owner: CL（cpp:979/996/1001 置換）
```text
bool casDelivery(slotIdx, oblId, to):     // 共通ヘルパ（T6/T7 兼用）
    loop:
        w = load(slot(slotIdx).lifecycle)
        if w.obligationId != oblId || w.state != Live: return false   // terminal/reuse: 付着しない
        expected = w
        desired  = { oblId, Live, w.pending, w.adjudicated, to }
        if cas(expected, desired, slot(slotIdx).lifecycle): return true
        // 失敗=contention（postSignal/adjudicate との競合）→ 再評価継続
```
- cpp:979 `Transport` / cpp:996 `None`（deferred）/ cpp:1001 `Durable` を `casDelivery(..., to)` へ置換。
- **false 返却時**（obligation が既に terminal）: intent は queue/durable に残るが Builder 消費後に resolve no-op（現行の重複表現窓と同一帰結）。戻り値 true（recovery 存在）は維持。

### T7 redrive delivery attach — owner: CL（cpp:1106-1167）
- 走査（cpp:1110/1112）: `w = load(...)` で `state==Live ∧ delivery==None` 判定（advisory）。
- `redriveDeferredRecovery` の id scan（cpp:1127-1132）: advisory load の `obligationId` 比較。
- 付着（cpp:1159/1167）: `casDelivery(idx, oblId, Durable/Transport)` へ置換。
- payload 読（handle/epoch/buildSource/recoveryGeneration・cpp:1143-1157）: CL 単一読取のため plain 維持。

### 5.8 liveCount authority（G152-06 総括）
| 操作 | 発火条件 |
|---|---|
| +1 | T4 Live 公開 CAS 勝者のみ |
| −1 | T3 terminal CAS 勝者のみ / T2 枯渇 CAS 勝者のみ |
| 相互排他 | T2 枯渇と T3 は同一語の expected.state==Live を競合 → 片方のみ成功 → **二重 −1 生成不能** |

---

## 6. delivery authority 完全固定（G152-07）

```text
delivery は RecoveryLifecycleWord の一フィールド。
変更 = casLifecycle 経由のみ。書込 owner = CL（T4/T5 なし/T6/T7/T2）。
postSignal（RebuildThread）は delivery を触らない（expected/new 同値）。
resolve は delivery を保持（desired = 現在値）。
```
- **plain write 完全禁止**: `slot(...).delivery = ...` / `s.delivery = ...` / `slots_[i].delivery = ...` を 0 に（D153 V2）。現行 cpp:1071（markTransientFailure の RebuildThread 書込）は T2 の `desired.delivery=None` に吸収され**消滅**。
- 読取は advisory load（§3）。全 decision-path reader（cpp:901/944/1112/1138）は CL。
- h:318-324 コメントは「delivery は W 内フィールド・CL のみ W CAS で変更」へ更新（実装後真になる）。

---

## 7. markTransientFailure 3 責務分離（G152-08/09）

```text
Production（RebuildThread・6 サイト）
    postRecoveryFailureSignal(oblId)        // T1: pending++ のみ。他一切非接触
CoordinatorLoop（CL・tick 毎）
    adjudicateRecoveryFailureSignals()      // T2: 全 slot drain+apply+枯渇 terminal
Test only（宣言コメントに TEST-ONLY 明記 — h:147 setRetireBacklogCount 前例）
    markTransientFailure(oblId) = { postRecoveryFailureSignal(oblId);
                                    adjudicateRecoveryFailureSignals(); }
```

### 7.3 production 置換表（6 サイト・機械的）
| file | line | 現行 | T3c |
|---|---|---|---|
| AudioEngine.RebuildDispatch.cpp | 1006 | `runtimePublicationBridge_.markTransientFailure(recovery->obligationId);` | `postRecoveryFailureSignal(...)` |
| 同 | 1033 | 同 | 同 |
| 同 | 1091 | 同 | 同 |
| 同 | 1115 | 同 | 同 |
| RuntimePublicationOrchestrator.cpp | 311 | `engine_.runtimePublicationBridge_.markTransientFailure(req.recoveryObligationId);` | `postRecoveryFailureSignal(...)` |
| 同 | 401 | 同 | 同 |

- 置換後、production 内 `markTransientFailure(` 呼び出し = **0**（D153 V1）。定義（cpp:1059-1085）はラッパ 2 行に置換、宣言（h:483）は TEST-ONLY コメント付きへ更新。
- h:473-482 の契約コメント（"Atomically performs…"）は T1/T2 の実仕様に書き換え（check-then-act 記述を除去）。

---

## 8. thread affinity 決定（G152-10）

```text
adjudicateRecoveryFailureSignals() = CoordinatorLoop 専用（規約）
runCoordinatorPhase():
    processIntent(...)                              // Threading.cpp:260-264
    ↓ ★ 挿入: runtimePublicationBridge_.adjudicateRecoveryFailureSignals();
    redriveDeferredRecoveryObligations()            // :272（同一 tick で None→redrive→wake）
```
- **jassert 不採用（決定）**: TEST-ONLY ラッパがテストスレッドで adjudicate を実行するため、CL thread-id 固定アサートはテスト契約と両立しない。かつ T3c の安全性は**スレッド非依存**（全 commit が CAS、§3）— affinity は決定論・タイミングのための規約であり安全条件ではない（D150 の affinity 非依存目標と整合）。
- 代替強制力: D153 V7 の grep（call site = runCoordinatorPhase + tests のみ）+ 関数ヘッダコメントに owner 明記。
- （将来 CL 外誤配線が疑われる場合のみ、work61 前例（AudioEngine.h:1484/1796）の ThreadID キャッシュで Debug 限定アサートを追加 — 本パッチ範囲外。）

---

## 9. telemetry 仕様（G152-11）

| member（新設/変更） | 型 | 増加条件 | getter |
|---|---|---|---|
| `recoveryRetryExhaustedCount_`（既存 h:1015・**条件変更**） | atomic<u64> | **T2 枯渇 CAS 勝者のみ**（現行 cpp:1079 の resolve 成否前増加を廃止） | 既存維持 |
| `recoveryFailureSignalSaturatedCount_`（新） | atomic<u64> | T1 `pending >= K` no-op | `recoveryFailureSignalSaturatedCount()` |
| `recoveryFailureSignalDroppedStaleCount_`（新） | atomic<u64> | T1 id 不一致（scan ゼロ / CAS 失敗後 id≠O=reuse） | 同型 |
| `recoveryFailureSignalDroppedTerminalCount_`（新） | atomic<u64> | T1 state≠Live + T3 勝者時の `pending>0` 破棄分（+= pending） | 同型 |
| `recoveryFailureSignalDroppedInvalidCount_`（新） | atomic<u64> | T1 `oblId == 0` | 同型 |

- **不変条件（invariant として明記）**: 「exhausted 過計上を許さない」— 増加は必ず terminal 化の CAS 成功と対応。T2 敗者（resolve 先行）は増加しない。
- 全 getter は既存パターン（`consumeAtomic(acquire)`、h:516-540 準拠）。カウンタは coordinator 所有（slot 状態ではない — G-4.3-R の「telemetry ≠ slot state」区別維持）。
- observability と semantic correctness は分離（D150 §9 継承）: 上記カウンタの誤りは GO 判定に影響しないが、exhausted の対応関係は D153 V9 のテストで検証。

---

## 10. テスト仕様（G152-12/13）

### 10.1 既存テスト（約 30 markTransientFailure site・ISRSemanticValidationTests.cpp のみ）
- **方針**: TEST-ONLY ラッパ（§7）で全 site 無変更。ラッパ = postSignal + 即 adjudicate（呼び出しスレッド実行）が、旧「同期 adjudication」暗黙前提を 1:1 で保存（1 観測=1 pending 加算→即 1 一括 adjudicate=1 加算）。
- 検証済みの非依存性: テストの delivery/counter/ObligationState 直接アクセス 0（D151 表 D）。アサート対象は liveCount / exhausted / coalesced / redrive 挙動 / durable・transport 実体 — すべて T3c で帰結同一（D150 §4 飽和等価性 CE-11）。
- accessor リネーム波及: `recoveryConsecutiveFailureCount` テスト使用 0 → 波及なし。
- **既存 40 テストの意味は変更しない**（D148 §12 契約の継承）。

### 10.2 新規 5 テスト（exact）
観測手段: TEST-ONLY `std::optional<RecoveryLifecycleWord> peekLifecycleForTest(std::uint64_t oblId)`（advisory load のコピー返却・単一スレッド文脈で整合保証）。

| # | 構成 | 断言 |
|---|---|---|
| NT-1 | submit→pop(id)→`postRecoveryFailureSignal(id)` のみ | peek: pending==1 ∧ state==Live ∧ adjudicated==0 ∧ delivery==Transport（不変）∧ liveCount==1 ∧ exhausted==0 |
| NT-2 | postSignal×3 → `adjudicateRecoveryFailureSignals()` | peek: pending==0 ∧ adjudicated==3 ∧ delivery==None ∧ Live ∧ liveCount==1（1 回の drain で一括） |
| NT-3 | postSignal×5 | 4 回目まで pending==K、5 回目 inert（peek pending==4 ∧ saturated==+1）→ adjudicate で Failed・exhausted==+1・liveCount==0 |
| NT-4 | postSignal(id) → `resolveRecoveryObligation(id, Published)` → adjudicate | adjudicate の CAS は state≠Live で失敗（no-op）: peek state==ResolvedSuccess ∧ pending==0 ∧ adjudicated==0 ∧ liveCount==0 ∧ exhausted==0（二重減算なし・droppedTerminal==+1） |
| NT-5 | 実 2 スレッド: A=postSignal(id) ループ、B=adjudicate ループ、main=一定数後 resolve(id, Published)、join | 最終: state==terminal ∧ liveCount==0 ∧ exhausted ∈ {0,1} ∧ pending==0 ∧ adjudicated < K（Live 時）— 混成世代 commit 不能（id 単調性）と −1 単一性を不変条件として検証（干渉順序は非決定、アサートは invariant のみ） |

- NT-5 は production フック不要（公開 API のみ）。決定論は invariant アサートで担保（G-4.3-T の T11 断念理由を解消 — 2 スレッドで構成可能）。
- 登録: 既存 main regression 一覧 + CMakeLists テストターゲット（#21 ISRSemanticValidation 系に同梱、新規 exe 不要）。

---

## 11. D153 機械検証条件（G152-14）

| # | 条件 | コマンド/検査 |
|---|---|---|
| V1 | production 内 `markTransientFailure(` 呼び出し 0 | `rg -n "markTransientFailure\(" src/audioengine` → 定義・宣言・コメントのみ |
| V2 | delivery plain write 0 | `rg -n "\.delivery\s*=" src/audioengine` → `desired.delivery`（ローカル語構築）のみ |
| V3 | 旧 atomic フィールド消滅 | `rg -n "consecutiveFailureCount|std::atomic<ObligationState>|atomic<LogicalRecoveryObligationId>" src/audioengine` → 0 |
| V4 | obligation state/id 直接 atomic アクセス 0 | `rg -n "slot\((i|existing|slotIdx)\)\.(state|id)\.|slots_\[i\]\.(state|id)\.|s\.(state|id)\.(load|store)" src/audioengine` → 0（`pendingRecoveryAdmission_.state`・`state_` は別ドメインで除外） |
| V5 | intrinsic 直接使用 1 箇所 | `rg -c "_InterlockedCompareExchange128" src/audioengine` → wrapper 定義のみ |
| V6 | static_assert 4 点 + alignas(16) + pad 常時 0 | ソース検査 |
| V7 | adjudicate call site 限定 | `rg -n "adjudicateRecoveryFailureSignals\(\)" src` → Threading.cpp + tests のみ |
| V8 | 6 サイト置換完了 | RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401 に `postRecoveryFailureSignal` |
| V9 | CTest: 既存 40/40×2（Debug/Release）+ NT-1..5 合格 + exhausted 対応不変（NT-3/4） | build/CTest（D153 通過後） |
| V10 | 旧構造回帰禁止 | `rg -n "fetch_add.*consecutiveFailure|\.delivery\s*=\s*Obligation" src/audioengine` → 0 |

---

## 12. file 別 patch 一覧（実装者向け）

| file | 変更 |
|---|---|
| ISRRuntimePublicationCoordinator.h | §1 W 定義 + §2 wrapper（namespace 直下）/ §4 struct 置換（pragma 4324 追加）/ findByKey・tryInsert（署名変更）・resolve・accessor・telemetry member/getter 追加 / postSignal・adjudicate・peekLifecycleForTest 宣言 / markTransientFailure を TEST-ONLY 化 / h:318-324・h:473-482 コメント更新 |
| ISRRuntimePublicationCoordinator.cpp | T1/T2 定義追加 / markTransientFailure→ラッパ / submit: T5 coalesce ループ・T6 casDelivery×3・wasDeferredBefore advisory 化 / redrive: advisory 化・T7 / shutdown close 不変（C11）/ resolveRecoveryObligation 不変（C10） |
| AudioEngine.Threading.cpp | runCoordinatorPhase に adjudicate 挿入（processIntent 後・redrive 前） |
| RuntimePublicationOrchestrator.cpp | :311/:401 置換 |
| AudioEngine.RebuildDispatch.cpp | :1006/1033/1091/1115 置換 |
| ISRSemanticValidationTests.cpp | NT-1..5 追加 + main 登録（既存 site 無変更） |
| CMakeLists.txt / build.bat | **変更なし** |

---

## 13. リスク・注記

- **R1（最重要）**: T5 coalesce の contention 再試行を省くと、postSignal との衝突で誤 tryInsert（ΔL 誤増・L 二重計上）になる。D153 で coalesce 回帰テスト（T1/T10 系）+ V4 検査必須。
- **R2**: tryInsert 署名変更の唯一呼び出し側 cpp:947-959 の移動漏れ → payload 未設定 Live 窓。V 検査: 「Live 公開前に recoveryGeneration 書込」コードレビュー項目。
- **R3**: C4324（alignas padding 警告）— pragma で抑制（前例 h:971）。
- **R4**: `_M_X64` ガード（プロジェクトは x64 のみ — build.bat 実測）。ARM64 移植時は `_acq/_rel` 変種の採用要否を再設計。
- **R5**: T1 の contention 再試行は有界（producer は RebuildThread 単一、pending は K で飽和、CL の drain は 1ms tick）。
- **R6**: terminal 時 delivery 保持（T3/T2 枯渇）は現行（None 化）と観測差なし — 全 reader Live ゲート + tryInsert で None 再初期化。仕様として明記。
- **R7**: `peekLifecycleForTest` は TEST-ONLY（本番呼び出し 0 を V 検査に追加可）。

---

## 14. 判定

**D152 = 仕様確定（G152-01..14 全項目固定）。** 実装は依然禁止 — 次は **D153 read-only implementation audit**（この仕様と実差分を 1 項目ずつ照合）であり、D153 通過後に初めて T3c production 実装 → build/CTest へ進む。
