# Phase I-T1-C1-R1 — Release Observation Authority / Double-Count Resolution

> **Status: COUNTEREXAMPLE — 同一 destruction が `releaseObserved_` を2回増加させる**
> T1-C1 = FAIL / OPEN, T1-C2..C6 = BLOCKED, T1-C7 = NO-GO, T1-C8 = NO-GO
> コード変更禁止・測定開始禁止。R1-1〜R1-6 を最新 `ConvoPeq.md` (96816行) と `src/` 実コードの両方で確定。
> 前回報告の「差分反映で正規化されるので二重加算されない」は誤り — 本R1で訂正する。

## 0. TL;DR / Verdict

| 項目 | 判定 |
|---|---|
| R1結論 | **COUNTEREXAMPLE** — 同一 `World` 破壊1件が `WorldRetirementTelemetry::releaseObserved_` を **+2** する |
| authoritative source | **DESIGN AMBIGUITY** を併発 — D86/D98/D99 と実装コメントが案A/案Bを同時に主張 |
| T1 measurement 可否 | **NO-GO** — `observedOutstanding = A - R`, `max`, `A_max candidate` は現コードでは信用できない |
| 影響範囲 | D/Q/E/Terminal/synchronous/shutdown の全 6+ drain path で同一構造 |

```
World destruction (type==World terminal deleter 成功)
   ├── storage.worldReclaimCount_++                (fetchAddAtomic 1)
   └── referenceObserver_->onRelease()
            └── telemetry_->onReleaseObserved()     (+1)

Timer sampler (Non-RT, 100ms)
   worldReclaimed = router->worldReclaimCount() // 上記 ++ を含む合算
   delta = worldReclaimed - prevReclaimed (=1)
   └── telemetry.addReleaseObserved(delta)         (+1)

=> releaseObserved_ への実反映 = 2  (期待 1 に対して 100% 過大)
```

---

## 1. 方法 / Tool Coverage（指示どおり全系統を使用）

| 系統 | ツール | 実行内容 | 結果 |
|---|---|---|---|
| WSL#1 | `rg ripgrep 15.1.0` | `rg -n "onRelease\(\)\|worldReclaimCount\|releaseObserved" src/ ConvoPeq.md` | 全ヒット列挙（§2-3） |
| WSL#2 | `ast-grep sg 0.44.0` | `sg search --pattern "onRelease" --lang cpp` / `sg search --pattern "worldReclaimCount"` | `onRelease` 2件、`worldReclaimCount` 複数確認 |
| WSL#3 | `fdfind fd 10.3.0` | `fdfind -t f "ISRWorldRetirement"` | `ISRWorldRetirementTelemetry.h` / `ISRWorldRetirementReference.h` 発見 |
| WSL#4 | `ag 2.2.0` | `ag "worldReclaimCount" src/` | `DeferredDeletionQueue.h` / `RetireQuarantineStore.h` / `ISRRetireRouter.cpp` 一致 |
| WSL#5 | `fzf 0.67.0` | `printf "onRelease\nworldReclaimCount\n" | fzf --filter "onRelease"` | フィルタ動作確認 |
| WSL#6 | `sed GNU 4.9` | `sed -n "85,105p" ISRWorldRetirementTelemetry.h` | `onReleaseObserved` / `addReleaseObserved` 定義確認 |
| WSL#7 | `awk GNU Awk 5.3.2` | `awk "/worldReclaimCount/ {print NR\":\"$0}" DeferredDeletionQueue.h` | increment 行特定 |
| MCP#1 | `serena` | `find_symbol name_path_pattern="onRelease" substring_matching=true` | `WorldRetirementReferenceObserver/onRelease (Reference.h:39)` / `WorldRetirementTelemetry/onReleaseObserved (Telemetry.h:88)` の2件を正規シンボルとして確定 |
| MCP#2 | `cocoindex ccc` | `ccc --help` / `ccc search "onRelease" --path ... --top-k 5` | `ccc` 正常起動（`C:\Users\user\.local\bin\ccc.exe`）、`onRelease` 検索で同ファイル群ヒット |
| CLI#1 | `graphifyy 0.9.48` | `graphify.exe --version` / `--help` | `graphify.exe` at `AppData\Roaming\Python\Python314\Scripts\graphify.exe` 動作確認（本リポ `graphify-out/` 未生成のため query はスキップ、代替で serena/semble/AiDex で補完） |
| CLI#2 | `semble 0.5.5` | `semble --version` / `semble search "worldReclaimCount"` | `worldReclaimCount` 検索で `DeferredDeletionQueue.h`/`RetireQuarantineStore.h`/`ISRRetireRouter` ヒット |
| MCP#3 | `AiDex` | `aidex_query term="onRelease" mode=exact` | 12 matches — 本R1の全 `onRelease` サイトと一致 |
| 文献 | `crossbeam-epoch` / `rigtorp MPMCQueue` / `serena` / `cocoindex` / `ast-grep` / `semble` / `AiDex` / `graphify` / `headroom` | `urllib` 200 OK 全9件（Vyukov は SSL失効で rigtorp 代替を明記） | D69/D86設計の外部裏付け補助（本R1は生産コード一次情報） |

> 本R1の法的根拠は `src/` と `ConvoPeq.md` の生産コード。文献は設計妥当性の補助参照のみ。

---

## 2. R1-1 — `onRelease()` 全呼び出し列挙

### 2.1 定義側

| Symbol | File | Line |
|---|---|---|
| `WorldRetirementReferenceObserver::onRelease() noexcept` | `src/audioengine/ISRWorldRetirementReference.h` | 39 |
| `WorldRetirementTelemetry::onReleaseObserved() noexcept` | `src/audioengine/ISRWorldRetirementTelemetry.h` | 88 (`fetchAddAtomic(releaseObserved_,1)`) |
| `WorldRetirementTelemetry::addReleaseObserved(count)` | `src/audioengine/ISRWorldRetirementTelemetry.h` | 94 (`fetchAddAtomic(releaseObserved_,count)`) |

`onRelease()` 内部で `telemetry_->onReleaseObserved()` を呼ぶ（Reference.h:42-43）：

```cpp
void onRelease() noexcept {
    fetchAddAtomic(referenceReleaseCount_, 1, acq_rel);
    if (telemetry_ != nullptr) telemetry_->onReleaseObserved(); // → releaseObserved_ +=1
    updateRunningMax();
}
```

### 2.2 呼び出し側（production path 全7コード地点 → 論理6分類）

| # | 分類 | File | Line | 周辺コード | 直前 `worldReclaimCount++` |
|---|---|---|---|---|---|
| 1 | **DeferredDeletionQueue — reclaim** (`D::reclaim` / `tryReclaim`) | `src/DeferredDeletionQueue.h` | 154 | `if (entryType==World){ fetchAdd(worldReclaimCount_,1); if(observer) observer->onRelease(); }` | 150 同ブロック |
| 2 | **DeferredDeletionQueue — shutdown drainAllUnsafe** | `src/DeferredDeletionQueue.h` | 204 | 同上 | 201 同ブロック |
| 3 | **RetireQuarantineStore — drain** (Q/E 共通コード、インスタンスは `m_retireQuarantine` / `m_emergencyQuarantine`) | `src/audioengine/RetireQuarantineStore.h` | 145 | `fetchAdd(worldReclaimCount_,1); observer->onRelease();` | 142 |
| 4 | **RetireQuarantineStore — drainAllUnsafe** (shutdown) | `src/audioengine/RetireQuarantineStore.h` | 182 | 同上 | 179 |
| 5 | **TerminalReclaimAuthority — drain** (epoch-safe drain) | `src/audioengine/ISRRetireRouter.cpp` | 72 | `++reclaimCount_; observer->onRelease();` | 70 同ブロック（`reclaimCount_` は `worldReclaimCount()` 合算に含まれる） |
| 6 | **TerminalReclaimAuthority — drainAll** (shutdown) | `src/audioengine/ISRRetireRouter.cpp` | 93 | 同上 | 91 |
| 7 | **TerminalReclaimAuthority — synchronous** (`recordWorldReclaim`) | `src/audioengine/ISRRetireRouter.h` | 105 | `++reclaimCount_; observer->onRelease();` + 呼び元 `ISRRetireRouter.cpp:505 recordWorldReclaim()` | 505 直前 `if(type==World) recordWorldReclaim()` |

> `RetireQuarantineStore` は1クラスで Q/E 2インスタンスに使い回されるため、コード地点は2だが論理パスは4（Q-drain / Q-drainAll / E-drain / E-drainAll）。合計論理パス = 7地点相当。

全7地点で **`worldReclaimCount` 系 increment と `onRelease()` が同一 `if (type==World)` ブロック内で隣接して実行**されている — D97.2 が固定する構造。

### 2.3 各分類の `physical destruction → storage → observer → telemetry` 表

| 分類 | `physical World destruction` | `storage counter` | `referenceObserver.onRelease?` | `telemetry.onReleaseObserved?` |
|---|---|---|---|---|
| Deferred `reclaim` | `entry.deleter(ptr)` 1回 | `worldReclaimCount_++` 1 | Yes (154) | Yes (+1) |
| Deferred `drainAllUnsafe` | 同上 | `worldReclaimCount_++` 1 | Yes (204) | Yes (+1) |
| Q `drain` | `deleter(ptr)` 1回 | `worldReclaimCount_++` 1 | Yes (145) | Yes (+1) |
| Q `drainAllUnsafe` | 同上 | `worldReclaimCount_++` 1 | Yes (182) | Yes (+1) |
| E `drain`/`drainAllUnsafe` | 同上（別インスタンス） | 同上 | Yes | Yes (+1) |
| Terminal `drain`/`drainAll` | `e.deleter(e.ptr)` 1回 | `++reclaimCount_` 1 | Yes (72/93) | Yes (+1) |
| Terminal `synchronous` | `deleter(ptr)` 1回 | `++reclaimCount_` 1 | Yes (105 via 505) | Yes (+1) |

全 path で **両方**が実行される — 片方のみの path は存在しない。

---

## 3. R1-2 — `worldReclaimCount_` 全 increment 列挙

| # | Storage | File | Line | 式 | 対応 `onRelease` |
|---|---|---|---|---|---|
| 1 | Deferred | `DeferredDeletionQueue.h` | 150 | `fetchAddAtomic(worldReclaimCount_,1,acq_rel)` | 154 同ブロック |
| 2 | Deferred | `DeferredDeletionQueue.h` | 201 | 同上 | 204 同ブロック |
| 3 | Q | `RetireQuarantineStore.h` | 142 | `fetchAddAtomic(worldReclaimCount_,1)` | 145 同ブロック |
| 4 | Q/E | `RetireQuarantineStore.h` | 179 | 同上 | 182 同ブロック |
| 5 | Terminal | `ISRRetireRouter.cpp` | 70 | `++reclaimCount_` | 72 同ブロック |
| 6 | Terminal | `ISRRetireRouter.cpp` | 91 | `++reclaimCount_` | 93 同ブロック |
| 7 | Terminal sync | `ISRRetireRouter.h` | 103 | `++reclaimCount_` in `recordWorldReclaim()` | 105 同ブロック |

### 1:1 対応表

| Increment 地点 | 対応 `onRelease()` 地点 | 同一 `if(type==World)` ? | 1:1 ? |
|---|---|---|---|
| Deferred 150 | Deferred 154 | Yes | 1:1 |
| Deferred 201 | Deferred 204 | Yes | 1:1 |
| Q 142 | Q 145 | Yes | 1:1 |
| Q/E 179 | 182 | Yes | 1:1 |
| Terminal 70 | 72 | Yes | 1:1 |
| Terminal 91 | 93 | Yes | 1:1 |
| Terminal sync 103 | 105 | Yes | 1:1 |

**例外なし — 全 increment は必ず `onRelease()` とペアで実行される。**

---

## 4. R1-3 — 1 destruction event の因果グラフ（6ケース数値化）

### 共通定義

```
S = {World terminal deleter 実行} 1件
C = storage counter increment (worldReclaimCount_ or reclaimCount_) 1
O = referenceObserver.onRelease() 1 → telemetry.onReleaseObserved() +1
T = sampler: worldReclaimCount() 合算 → delta=1 → addReleaseObserved(1) +1
R = telemetry.releaseObserved_ への最終加算
```

### Case A — Deferred `D::reclaim()`

```
World destruction =1
  C =1 (Deferred worldReclaimCount_ 150)
  O =1 → R +=1 (onReleaseObserved)
Sampler later: worldReclaimed = sum(Deferred 1 + Q0 + E0 + Terminal0)=1, delta=1 → R +=1 (addReleaseObserved)
=> R = 2  (期待 1 → +1 過大)
```

### Case B — Quarantine `Q::drain()`

```
World destruction =1
  C =1 (RetireQuarantineStore 142)
  O =1 → R +=1
Sampler: delta 1 → R +=1
=> R = 2
```

### Case C — Emergency `E::drain()`

```
同上（EmergencyQuarantineStore 142/179 は同一クラス、別インスタンス）
=> R = 2
```

### Case D — Terminal `Terminal::drain()` / `tryReclaim`

```
World destruction =1
  C =1 (Terminal reclaimCount_ 70)
  O =1 → R +=1
Sampler: worldReclaimCount() = provider + Q + E + Terminal.reclaimCount(1) → delta 1 → R +=1
=> R = 2
```

### Case E — Synchronous Terminal `recordWorldReclaim()`

```
ISRRetireRouter::terminalReclaim() epochSafe && !isRt → deleter(ptr) + recordWorldReclaim() 505
  C =1 (reclaimCount_ 103)
  O =1 → R +=1
Sampler: delta 1 → R +=1
=> R = 2
```

### Case F — Shutdown `drainAll*`

```
Deferred::drainAllUnsafe + Q::drainAllUnsafe + E::drainAllUnsafe + Terminal::drainAll
 各 entry で上と同じペアがループ内で繰り返される
=> 1 World あたり R = 2（N Worlds なら R=2N）
```

### 数値表

| Case | `physical destruction` | `worldReclaimCount delta` | `reference release event` | `telemetry R increment` | 期待 | 実際 |
|---|---|---|---|---|---|---|
| A Deferred | 1 | 1 | 1 | **2** | 1 | **+1 過大** |
| B Q | 1 | 1 | 1 | **2** | 1 | **+1 過大** |
| C E | 1 | 1 | 1 | **2** | 1 | **+1 過大** |
| D Terminal | 1 | 1 | 1 | **2** | 1 | **+1 過大** |
| E sync Terminal | 1 | 1 | 1 | **2** | 1 | **+1 過大** |
| F shutdown | 1 | 1 | 1 | **2** | 1 | **+1 過大** |

> **全ケースで `R = O + T = 2`。正常系 `R=1` からの乖離は系統的。**

Counterexample の最小コード断片（Deferred 例、他も同形）：

```cpp
// DeferredDeletionQueue.h:148-154 (reclaim) — 現コード抜粋
if (entryType == DeletionEntryType::World) {
    convo::fetchAddAtomic(worldReclaimCount_, 1, acq_rel); // C
    if (referenceObserver_ != nullptr)
        referenceObserver_->onRelease(); // O → telemetry.onReleaseObserved() +1
}
// ...
// AudioEngine.Timer.cpp:425-429 (sampler)
const uint64_t worldReclaimed = m_retireRouter->worldReclaimCount(); // C を含む
delta = worldReclaimed - prevReclaimed; // 1
telemetry.addReleaseObserved(delta); // T +1
// => 同一 destruction で R に +1+1 = +2
```

---

## 5. R1-4 — 「authoritative source」再判定

### 5.1 実コード自身の記述（一次情報）

| File | Line | 記述 |
|---|---|---|
| `ISRWorldRetirementTelemetry.h` | 71-72 | `releaseObserved: sampler が storage 側の worldReclaimCount（type==World の terminal deleter 実行数・D86）の累積差分を反映` — **案B** を明記 |
| `ISRWorldRetirementTelemetry.h` | 86-87 | `実体の更新は storage 側の worldReclaimCount_ が担う。本メソッドは sampler が差分を移すための入口` — `onReleaseObserved()` 自体を **sampler 経由の反映入口**と説明 |
| `ISRWorldRetirementReference.h` | 37-38 | `sampler の outstanding 推定を正しくするため telemetry の releaseObserved にも転送する（同一 terminal release を両観測系が観測・D100 の独立観測）` — **案Aを案Bと独立に両方加算する**と明記 |
| `DeferredDeletionQueue.h` | 146-147 | `type==World の terminal deleter 実行後 → world 破棄観測（release observation・案 B）` と同時に `reference observer に release event を通知（event-driven）` — 両方を同ブロックで肯定 |

### 5.2 設計文書 D86/D98/D99 間の不一致

```
D86: type==World の terminal deleter 実行数 = world 物理破棄数、worldReclaimCount が一次情報源（案B）
D98: reference observer は measurement only / non-owning
D99/D100: reference observer は event-driven で running max を更新
D100.4: 「sampler の outstanding 推定を正しくするため telemetry.releaseObserved にも転送（独立観測）」
```

D86 が案Bを一次情報源としつつ、D100.4 が「独立観測」として案Aの転送を正当化 — **両案が同時に正規**として文書化され、生産コードは両方を同時に実行。

### 5.3 本R1の判定

> 単に「D86だから案Bが正規」や「reference observer が正規だから sampler は補助」とは結論できない。
> 現状は **DESIGN AMBIGUITY** — authoritative release source が未確定のまま、実装は両経路を同時に `releaseObserved_` に加算している。

R1としては **COUNTEREXAMPLE が主判定**、その原因として **DESIGN AMBIGUITY を併記**する（§7 参照）。

---

## 6. R1-5 — T1 reference telemetry と T1 measurement telemetry の分離検証

### 6.1 2つのカウンタの所在

| カウンタ | 所在 | 更新源 | 用途 |
|---|---|---|---|
| `referenceReleaseCount_` | `WorldRetirementReferenceObserver` (Reference.h:??) | `onRelease()` で `fetchAdd(1)` | event-driven running max (`updateRunningMax()`) |
| `releaseObserved_` | `WorldRetirementTelemetry` (Telemetry.h:releaseObserved_) | `onReleaseObserved()` (+1) と `addReleaseObserved(delta)` (+delta) | `observedOutstandingEstimate = A - R`, `max`, window export |

### 6.2 分離しているか？

```
ReferenceObserver.referenceReleaseCount_  ← onRelease() で +1（独立）
ReferenceObserver → telemetry.onReleaseObserved() → Telemetry.releaseObserved_ +1  ← ここで結合
worldReclaimCount → sampler → Telemetry.addReleaseObserved(delta) → 同じ releaseObserved_ +1
```

**結論: 分離していない。** `referenceReleaseCount_` は独立だが、`releaseObserved_` は **両経路の合流点**。したがって `releaseObserved_` に対する double-count は回避されない。

もし設計が以下なら二重ではなかった：

```
ReferenceObserver.referenceReleaseCount  → reference max 用（telemetry に転送しない）
worldReclaimCount → sampler → Telemetry.releaseObserved  → T1 measurement 用
```

しかし現コードは `ReferenceObserver.onRelease()` が必ず `telemetry.onReleaseObserved()` を呼ぶため、この分離は成立していない。

AiDex `onRelease` 12 hits / serena `onRelease` 2 symbols も、上記結合を裏付ける（`Reference.h:39` が `telemetry_->onReleaseObserved()` を呼ぶ定義として索引されている）。

---

## 7. R1-6 — 現時点判定

```
T1-C1 = FAIL / OPEN  — COUNTEREXAMPLE 確定（全 drain path で R+2）
T1-C2 = BLOCKED      — single observation 証明は R 確定後に再実施
T1-C3 = BLOCKED      — window/sampler 整合性は R 正規化後に再検証
T1-C4 = BLOCKED      — RT safety は再監査不要だが R 依存のため保留
T1-C5 = BLOCKED      — export 監査は R 正規化後に再実施
T1-C6 = BLOCKED      — test gap 調査は R 修正設計確定後に再実施
T1-C7 = NO-GO
T1-C8 = NO-GO
```

### 判定根拠（3値のいずれか）

```
COUNTEREXAMPLE — 同一 destruction が releaseObserved_ を2回増加させる
```

- `CLOSED` ではない：2経路は異なるカウンタへ行っていない、同一 `releaseObserved_` へ合流する。
- `DESIGN AMBIGUITY` も併発するが、主判定は `COUNTEREXAMPLE`（実装が二重加算を直接実行しているため、ambiguity だけでは片付かない）。

### 影響

```
releaseObserved_ が 2倍に過大 → observedOutstanding = A - R が過小
→ observedOutstandingMax が過小 → A_max candidate が過小に測定される
→ T1 measurement は現コードのままでは信用できない
```

前回報告で「差分反映で正規化される」とした説明は、生産コードの `fetchAdd` と `onRelease` の隣接実行（§2-3）および sampler の `addReleaseObserved(delta)`（Timer.cpp:429）を同時に読み落とした誤り — 本R1で訂正する。

---

## 8. 改修案の適合性調査（コードは書かない — 設計選択肢のみ）

> 指示どおり修正コードは書かない。どちらを authoritative にするべきかだけを現行ソース適合性で棚卸しする。

### Option A — `worldReclaimCount → sampler → addReleaseObserved` を authoritative にする

| 観点 | 評価 |
|---|---|
| 現コード整合性 | `Deferred/Quarantine/Terminal` の `worldReclaimCount_++/reclaimCount_++` は既に全 path で存在し、sampler (Timer.cpp:425-430) も実装済み。D86 の一次情報源定義とも一致 |
| 変更点 | `ReferenceObserver.onRelease()` から `telemetry.onReleaseObserved()` への転送を止め、`referenceReleaseCount_` のみを更新する（`updateRunningMax()` は維持）。`T1 reference max` と `T1 measurement max` を分離 |
| RT safety | 影響なし（`onRelease()` は元々 atomic のみ） |
| リスク | `referenceReleaseCount` と `releaseObserved` の乖離を許容する設計にする必要あり（D100.4 の「独立観測」記述を「分離観測」に訂正） |

### Option B — `onRelease() → onReleaseObserved()` を authoritative にする

| 観点 | 評価 |
|---|---|
| 現コード整合性 | `referenceObserver` は全 drain path で既に呼ばれる。sampler の `worldReclaimCount` 読みは不要になるが、`worldReclaimCount_` 自体は診断用に残せる |
| 変更点 | sampler の `addReleaseObserved(delta)` を止め、`onReleaseObserved()` のみを計数する。`lastSampledWorldReclaimCount_` は不要に |
| RT safety | 影響なし |
| リスク | `worldReclaimCount` が authoritative でなくなると D86 の記述と衝突。`worldReclaimCount` を残す場合も sampler を止めると `worldReclaimCount` と `releaseObserved` の一致保証を失う |

### Option C — 両方を残しつつ `releaseObserved_` を分離（2カウンタ化）

| 観点 | 評価 |
|---|---|
| 現コード整合性 | 最小の挙動変更 — `Telemetry` に `referenceReleaseObserved` と `sampledReleaseObserved` を分離し、T1 measurement は片方のみを採用 |
| 変更点 | `Telemetry` の API 追加が必要（現 `releaseObserved_` 単一を分割） |
| リスク | 既存 `observedOutstandingEstimate()` / `max` / export の全てがどちらのカウンタを使うか再定義が必要 |

### 現行ソース適合性の暫定評価

- **最小侵襲は Option A** — `Reference.h:42-43` の `telemetry_->onReleaseObserved()` 1行を止めるだけで double-count は解消し、他ファイルの `worldReclaimCount` 機構はそのまま authoritative として使える。`referenceReleaseCount` は reference max 用に独立して残る。
- ただし D86/D98/D99 の文書訂正が必須（「独立観測」を「分離観測」に、authoritative を明記）。
- **いずれの Option でも `A_max 実測へは進まない` こと — 本R1確定後に設計レビューで authoritative を一本化してから T1-C2 以降を再開する。**

---

## 9. 棚卸し事項（未確定 → 本R1で確定したもの / 残課題）

| 区分 | 事項 | 本R1での確定 |
|---|---|---|
| 要調査 | `onRelease()` 全呼び出し | 確定 — 7地点（§2.2） |
| 要調査 | `worldReclaimCount` 全 increment | 確定 — 7地点、1:1 ペア（§3） |
| 要調査 | 6ケース因果グラフ | 確定 — 全ケース R=2（§4） |
| 要調査 | authoritative source | 確定 — DESIGN AMBIGUITY 併発、COUNTEREXAMPLE が主（§5） |
| 要調査 | Reference vs Telemetry 分離 | 確定 — 分離しておらず同一 `releaseObserved_` に合流（§6） |
| 保留 | `R_required` / `A_max` 実測 | 保留のまま — 本 counterexample 解消まで着手しない |
| 保留 | Window/sampler, RT safety, export, test gap, Build/CTest | BLOCKED のまま — R 正規化後に再検証 |

---

## 10. 付録 — 主要証拠の原文抜粋（行番号付き）

### Telemetry 定義

```cpp
// ISRWorldRetirementTelemetry.h:71-72
// releaseObserved: sampler が storage 側の worldReclaimCount（type==World の terminal deleter 実行数・D86）
//                  の累積差分を反映

// ISRWorldRetirementTelemetry.h:88-90
void onReleaseObserved() noexcept { fetchAddAtomic(releaseObserved_, 1, acq_rel); }
// 94-98
void addReleaseObserved(uint64_t count) noexcept { if(count==0) return; fetchAddAtomic(releaseObserved_, count, acq_rel); }
```

### ReferenceObserver 定義（合流点）

```cpp
// ISRWorldRetirementReference.h:39-44
void onRelease() noexcept {
    fetchAddAtomic(referenceReleaseCount_, 1, acq_rel);
    if (telemetry_ != nullptr) telemetry_->onReleaseObserved(); // ← 合流
    updateRunningMax();
}
```

### Storage 側（Deferred 例）

```cpp
// DeferredDeletionQueue.h:148-154
if (entryType == DeletionEntryType::World) {
    fetchAddAtomic(worldReclaimCount_, 1, acq_rel);
    if (referenceObserver_ != nullptr) referenceObserver_->onRelease();
}
```

### Sampler

```cpp
// AudioEngine.Timer.cpp:425-430
const uint64_t worldReclaimed = m_retireRouter ? m_retireRouter->worldReclaimCount() : 0;
const uint64_t prevReclaimed = consumeAtomic(lastSampledWorldReclaimCount_, acquire);
if (worldReclaimed > prevReclaimed) {
    telemetry.addReleaseObserved(worldReclaimed - prevReclaimed);
    publishAtomic(lastSampledWorldReclaimCount_, worldReclaimed, release);
}
```

---

*Evidence generated: Phase I-T1-C1-R1 — do not proceed to measurement until authoritative release source is unified and double-count is eliminated.*
