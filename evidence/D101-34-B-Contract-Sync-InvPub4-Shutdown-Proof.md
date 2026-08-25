# D101-34-B — Contract Synchronization + INV-PUB-4 / Shutdown Formal Proof（設計・証明確定報告書）

- **実施日**: 2026-08-25
- **作業種別**: 証明・契約同期フェーズ（ソースコード変更 **0** / I4_DESIGN_CONTRACT.md も未編集 — 次タスク D101-34-C で反映）
- **判定**: **PASS**（B-9 全条件充足。INV-PUB-4 exactly-once 証明完了、RuntimeStore shutdown contract 全終了経路で current==nullptr を証明）
- **基準**: ConvoPeq.md 15:15 版（D101-34-A）＋ 本フェーズでの fresh source trace
- **⚠️ D101-34-A 訂正**: A 報告書 §5 の「onRelease 5 sites」は `src/core/DeferredDeletionQueue.h` 内
  2 site の取りこぼし（grep 範囲漏れ）。**正しい基準値は 7 sites**（§3）。A の他の結論は変更なし。

---

## 1. B-1: I4.D101 旧状態 ↔ 現行実装 対応表（契約更新対象の全列挙）

| 契約記載（2026-08-16 版） | 現行実装の事実 | 更新要否 |
|---|---|---|
| `retireRuntimePublishWorldNonRt` が AudioEngine.h:3534 に存在 | 参照ゼロ（削除済み） | ✅ 要更新 |
| Init.cpp:67 rejectedWorld → World → onRelease（反例） | Init.cpp:67 は `retireRejectedRuntimeWorldNonRt(rejectedWorld)` → **Generic** | ✅ 要更新 |
| Step 2 API Separation = NOT STARTED | **IMPLEMENTED**（h:3533/:3549、PRECONDITION コメント付き） | ✅ 要更新 |
| INV-PUB-3 = DISPROVEN | 反例解消（code-fixed）。形式証明は本報告書 §6 で部分完了 | ✅ 要更新（§6） |
| onRelease = 3 terminal sites | 実測 **7 candidate sites**（§3） | ✅ 要更新 |
| RuntimeStore dtor contract = 構造確認のみ | **全終了経路で current==nullptr を証明**（§5） | ✅ 要更新 |
| INV-PUB-1/2 = OPEN | 変更なし（本設計の scope 外） | − |

---

## 2. B-2: INV-PUB-4 exactly-once 証明

### 2.1 storage topology（実測）

```text
retirePublishedRuntimeWorldNonRt (唯一の World producer, h:3545)
    ↓ enqueueDeferredDeleteNonRt → enqueueWithRetry (ISRRetireRouter.cpp:315-380)
    ↓ Ownership chain: D → Q → EmergencyQ(E) → TerminalReclaimAuthority
      （cpp:315 「Ownership chain」コメント。各段は排所有で受け取る:
        「ptr transferred here ⟹ caller retains NO ownership」）

D  = DeferredDeletionQueue（src/DeferredDeletionQueue.h、bounded ring）
Q  = RetireQuarantineStore（第2退避）
E  = EmergencyQuarantineStore（第3退避 — RetireQuarantineStore と同一クラス型の別インスタンス、h:402）
T  = TerminalReclaimAuthority（growable vector・常に受領可能 cpp:33-38）
```

### 2.2 onRelease candidate sites = **7**（全列挙）

| # | site | 属する storage | 発火条件 |
|---|---|---|---|
| R1 | DeferredDeletionQueue.h:154 | D | reclaim() が D 上の entry を CAS dequeue して破壊（type==World） |
| R2 | DeferredDeletionQueue.h:204 | D | drainAllUnsafe() 同上（shutdown） |
| R3 | ISRRetireRouter.cpp:87 | T | TerminalReclaimAuthority::drain() — epoch safe 化後 |
| R4 | ISRRetireRouter.cpp:114 | T | TerminalReclaimAuthority::drainAll() |
| R5 | ISRRetireRouter.h:128 | **storage 不経由** | tryReclaim 系の同期破壊（epoch safe & NonRT 判定時に handoff 即破壊）— recordWorldReclaim |
| R6 | RetireQuarantineStore.h:145 | Q / E（同一クラス型の別インスタンス双方をカバー） | reclaimBatch() |
| R7 | RetireQuarantineStore.h:182 | Q / E | drainAllUnsafe() |

acquire: Commit.cpp:409 `worldRetirementReference_.onAcquire()` — publish LP で 1回。

### 2.3 指示の 7 条件に対する証明

**(1) World entry が同時複数 storage に存在し得ない**
✅ ownership chain は「転送 = 排所有移動」。enqueueWithRetry は D→Q→E→T の順に
**挿入成功した storage のみ**が ptr を保持し（失敗した段は保持しない）、一度受領された
entry はその storage が破壊するまで移動しない（retry cycle は再 enqueue 前に
tryReclaim で破壊を試みる = 移動ではなく破壊）。単一所有者の言語仕様的保証
（「ptr transferred here ⟹ caller retains NO ownership」, cpp:20-22）。

**(2) D → quarantine 移動時の複製なし**
✅ 前項どおり、移動は「前段で破壊試行→空いた場合の新規 enqueue」または「受領失敗段を
スキップ」であり、同一 {ptr,deleter} が2つの storage に同時存在する経路はない。

**(3) reclaim() と drainAllUnsafe() の二重処理不能**
✅ D 内: reclaim() は dequeuePos への **CAS 成功者のみ**が破壊を実行し、破壊直後に
slot を無効化（ptr=nullptr, type=Generic, seq 前進 — h:154-168 相当）。
drainAllUnsafe() も同一 CAS/dequeuePos 機構を使用（:188-194）。
両者とも破壊済 slot は二度と取得できない。ライフタイム分離（reclaim=運用中 /
drainAllUnsafe=shutdown 後）も併存。

**(4) RetireQuarantineStore::reclaimBatch() と drainAllUnsafe() の相互排他**
✅ 両者とも `lock_guard(mtx_)` 下で `pending.swap(entries_)`（または size_=0 リセット）を
行うため、同一 entry を両者が取得することは不可能。破壊は lock 外だが entry は既に
store から抽出済み。Q と E は別インスタンスのため相互に独立。

**(5) Terminal 同期破壊経路と quarantine 経路の非重複**
✅ R5（同期破壊）は handoff 時点で即破壊するため Terminal storage には**挿入されない**
（recordWorldReclaim は storage 非経由の release 通知）。R3/R4 は storage 内 entry のみ
処理。挿入されない以上、R5 と R3/R4 が同一 entry を処理する組合せは存在しない。

**(6) 各 World entry → onRelease exactly one**
✅ 上記 (1)-(5) より: entry の生涯は
`[生成] → (単一 storage に常駐) → (当該 storage の唯一の破壊 site で deleter 実行 +
type==World 分岐で onRelease 1回)` または `[handoff 即破壊 → R5 1回]`。
storage 間移動は破壊を伴わない排所有移動のため onRelease を発火しない。
∴ **World entry 1個につき onRelease 実行回数 = ちょうど 1**。

**(7) onAcquire との identity 対応**
✅ onAcquire は publish LP（Commit.cpp:409）で commit された world ごとに 1回。
当該 world は必ず publishAndSwap を経由しているため PublishedDomain member であり、
退役時 retirePublished → World entry（ちょうど1個）→ onRelease 1回。
Rejected/unpublished world は onAcquire せず Generic 経路で破壊されるため onRelease もない。
∴ エンジン lifetime で `ΣonAcquire == ΣonRelease`（quiescence 時）。

> **注意**: 「sites = 7」は terminal consumer の**候補数**であり実行回数ではない。
> 証明対象は entry 1個あたりの実行回数 = 1（指示どおり）。

---

## 3. B-3 RuntimeStore shutdown contract（全終了経路で current==nullptr）

### 3.1 構造的事実

- `current` の唯一の mutation は `WriteAccess::publishAndSwap` =
  `exchangeAtomic(current, next, acq_rel)`（RuntimeStore.h）。
- `RuntimeStore` に user destructor なし → `std::atomic<T*>` デストラクタはポインタを
  破棄しない → **destructor invocation 時に current != nullptr なら leak**。
- null 化の唯一の手段 = `publishAndSwap(nullptr)` = `clearPublishedRuntimeSnapshotsNonRt()`
  （core/RPC.h:93-100）+ caller 侧 retirePublished。

### 3.2 全終了経路の列挙と証明

| # | 終了経路 | current==nullptr 到達 | 根拠 |
|---|---|---|---|
| 1 | 通常 shutdown（releaseResources → ~AudioEngine） | ✅ ReleaseResources.cpp:492 付近で clear（requestShutdownClearNonRt + clearPublishedRuntimeSnapshotsNonRt + retirePublished）| 実測 |
| 2 | ~AudioEngine 単独（releaseResources 未実行の異常系） | ✅ CtorDtor.cpp:242-248 が dtor 内で**再度** clear を実行（冪等 — 既に null なら publishAndSwap(nullptr)=nullptr → retirePublished 冪等 return）| 実測 |
| 3 | prepareToPlay failure | bootstrap validate fail → Rejected(Generic)。publish 未成立なら current==nullptr のまま → 破壊時自明に成立。再初期化で publish 済みの場合は経路 2 の dtor clear が適用 | ✅ |
| 4 | initialization failure | 同上 | ✅ |
| 5 | partial publication failure | pre-LP 失敗（CallerDestroy）により current 不変。最終破壊は経路 1/2 の clear を通る | ✅ |
| 6 | publication rejection | 同上（Rejected は current を触らない） | ✅ |
| 7 | exception-free early return | AudioEngine 関係関数は noexcept。例外による clear skip 経路は存在しない | ✅ |
| 8 | destructor-triggered shutdown | = 経路 2 | ✅ |

### 3.2.1 証明の骨格

```text
補題1: current への write は publishAndSwap のみ（RuntimeStore.h — write authority は
       acquireWriteAccess 経由の Owner のみ、friend Owner で制限）。
補題2: publishAndSwap(nullptr) 以降、producer による新規 swap-in は不可能
       （admission closed: closeAdmission 済み + producer thread join 済み —
        D101-33-C/D で確立。Recovery/Build gate も ShuttingDown で閉鎖）。
       ＊前提条件: clear 実行スレッド以外が publish LP に到達しなこと（Q2 producer join）。
補題3: clear は冪等（2回目の publishAndSwap(nullptr) は nullptr を返し retirePublished は
       null early-return）。
定理:  AudioEngine 破壊シーケンスは必ず経路 1 または 2 の clear を通り、
       補題2 の下では以降 current を書く者がいないため、
       RuntimeStore destructor invocation ⇒ current == nullptr ∎
```

⚠️ **証明の前提条件（明示）**: 補題2 の「producer join 済み」が崩れる呼び出し
（例: 破壊開始後に外部 thread から commitRuntimePublication を呼ぶ誤用）は本契約の
範囲外（caller 契約違反）。将来の強化として debug assertion（clear 後 admission state 確認）
を推奨。

---

## 4. B-4: INV-PUB-3 の状態分離

| 段階 | 状態 |
|---|---|
| Code-fixed | ✅ **完了** — API separation（retirePublished/retireRejected）+ caller provenance（D101-34-A §3: P-1〜P-6 / R-1〜R-2 全分類）により反例経路は消滅 |
| Formal-proof-complete | ⚠️ **部分** — Publication domain 側（facade/core RPC/Bootstrap/Shutdown clear）の閉包は本報告書 §2-§3 で完了。残余は「全 production path」の網羅性を将来の compiler-enforced 検査（type-state 等）で継続保証する段階 |

I4 更新時の表記: `INV-PUB-3 = Code-fixed ✅ / Formal-closure: publication domain complete / whole-engine type-state enforcement pending`

---

## 5. INV-WORLD-TYPE 確定

**確定: 成立可能かつ成立している（code level）。**

- World 型の deletion entry 生成は `retirePublishedRuntimeWorldNonRt` 単一箇所に閉じ、
  その PRECONDITION（W ∈ PublishedDomain）がコメントで契約化されている。
- 全 caller（6 site）が publishAndSwap LP 通過後の oldWorld/clearedWorld のみであることを
  D101-34-A §3 で 100% 分類済み。
- Rejected/unpublished は Generic または直接破棄。
- 将来強化: PRECONDITION の debug assertion（world.sealState 照会等）追加の余地あり。

---

## 6. PASS 条件チェックリスト

* [x] I4.D101 旧状態 ↔ 現行実装の不一致全列挙（§1 — 6 項目）
* [x] Step 2 = IMPLEMENTED をコード根拠付き確認（h:3533/:3549）
* [x] Published / Rejected API caller provenance 確定（A §3 + 本報告書）
* [x] INV-PUB-3 counterexample elimination の形式化（Code-fixed / formal 分離 — §4）
* [x] `DeletionEntryType::World` producer = 1 再確認
* [x] World entry の single-storage / single-terminal path 証明（§2.3 (1)(2)(5)）
* [x] 7 `onRelease()` site の役割個別分類（§2.2 — **A の 5 から訂正**）
* [x] INV-PUB-4 exactly-once proof 完了（§2.3 (6) — entry 1個につき実行回数 1）
* [x] RuntimeStore 全 shutdown/destruction path 列挙（§3.2 — 8 経路）
* [x] current == nullptr at destruction 証明（補題1-3 + 定理）
* [x] INV-WORLD-TYPE 確定（§5）
* [x] コード変更 0
* [x] 契約ファイル未編集（次タスク D101-34-C で反映）

# VERDICT: D101-34-B = **PASS**

---

## 7. D101-34-C（Contract Update Implementation）への引き継ぎ

実装すべき文書更新（コード変更なし）:

1. `doc/work88/I4_DESIGN_CONTRACT.md` I4.D101 節:
   - Step 2 → IMPLEMENTED（h:3533/:3549、Generic/World 分離）
   - 反例トレース削除 → 解消記録（Init.cpp:67 現状）
   - Layer 4: onRelease sites 3 → **7**（site 一覧付き）
   - INV-PUB-3 → Code-fixed / formal closure: publication domain complete
   - INV-PUB-4 → exactly-once 証明参照（本報告書 §2）
   - RuntimeStore shutdown contract → Tier 4 証明完了（§3、前提条件 Q2 明記）
2. D101 #1 状態: OPEN → **CLOSED（Step 1-3 完備）**。ただし M 導出（D101 の本来目的）
   とは別軸である旨は维持。
3. 次フェーズ: M の数学的バインド再開判断（I4.D101.4 判定条件）。
