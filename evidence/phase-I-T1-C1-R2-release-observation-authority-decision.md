# Phase I-T1-C1-R2 — Release Observation Authority Decision

> **Verdict: OPTION-A READY**
> R1 COUNTEREXAMPLE を前提に `releaseObserved_` の authoritative source を一本化する設計確定フェーズ。
> コード変更禁止・測定再開禁止。9 terminal LP と全設計文書を生産コードで確定。

## 1. R1 counterexample の再確認

R1 で確定した二重加算構造を再掲する。

```
World deleter(ptr) 1回実行 (type==World)
   ├── storage: worldReclaimCount_/reclaimCount_ ++1  (D86 案B)
   └── referenceObserver.onRelease() → telemetry.onReleaseObserved() +1 (D98/D100.4 案A)

Timer sampler (Non-RT, 100ms, AudioEngine.Timer.cpp:425-430)
   worldReclaimed = router->worldReclaimCount() // 上記 ++ を含む合算
   delta = worldReclaimed - prevReclaimed (=1)
   telemetry.addReleaseObserved(delta) +1

=> telemetry.releaseObserved_ への加算 = 1 (event) + 1 (sampled) = 2 (期待 1)
```

生産コード上の隣接実行（同一 `if (type==World)` ブロック内）は全 drain path で確認済み。したがって現コードは `releaseObserved_` を系統的に 2倍に過大計上し、`observedOutstanding = A - R` / `max` / `A_max candidate` は信用できない。

---

## 2. 9 terminal LP の destruction semantics

R1 では論理圧縮していたが R2 では 9 LP を個別に扱う。

| # | LP | File | 行 | deleter | storage counter | observer |
|---|---|---|---|---|---|---|
| 1 | Deferred `reclaim` (epoch-safe reclaim) | `DeferredDeletionQueue.h` | 145-154 | `entry.deleter(entry.ptr)` | `worldReclaimCount_ fetchAdd 1` (150) | `onRelease()` (154) |
| 2 | Deferred `drainAllUnsafe` (shutdown) | `DeferredDeletionQueue.h` | 197-204 | `entry.deleter(entry.ptr)` | `worldReclaimCount_ fetchAdd 1` (201) | `onRelease()` (204) |
| 3 | Quarantine Q `drain` | `RetireQuarantineStore.h` | 140-145 | `deleter(ptr)` | `worldReclaimCount_ fetchAdd 1` (142) | `onRelease()` (145) — `m_retireQuarantine` |
| 4 | Quarantine Q `drainAllUnsafe` | `RetireQuarantineStore.h` | 176-182 | `pendingDeleters[i](pendingPtrs[i])` | `fetchAdd 1` (179) | `onRelease()` (182) — `m_retireQuarantine` |
| 5 | Emergency E `drain` | `RetireQuarantineStore.h` | 140-145 | 同上 | 同上 (142) | 同上 (145) — `m_emergencyQuarantine`（同一クラス別インスタンス） |
| 6 | Emergency E `drainAllUnsafe` | `RetireQuarantineStore.h` | 176-182 | 同上 | 同上 (179) | 同上 (182) — `m_emergencyQuarantine` |
| 7 | Terminal `drain` / `tryReclaim` | `ISRRetireRouter.cpp` | 68-72 | `e.deleter(e.ptr)` | `++reclaimCount_` (70) | `onRelease()` (72) |
| 8 | Terminal `drainAll` (shutdown) | `ISRRetireRouter.cpp` | 88-93 | `e.deleter(e.ptr)` | `++reclaimCount_` (91) | `onRelease()` (93) |
| 9 | Terminal `synchronous` | `ISRRetireRouter.h` 102-105 + `ISRRetireRouter.cpp` 505 | 505 `recordWorldReclaim()` | `deleter(ptr)` 直後 | `++reclaimCount_` (103) | `onRelease()` (105) |

全 9 LP で `World deleter 1回 → storage counter +1 → observer 1回` が同一 `if (type==World)` で隣接して成立する。例外的に片方のみが実行される path は存在しない。9 LP は `RetireQuarantineStore` が Q/E で共有されるためコード地点は 7 だが論理 LP は 9 として数えるのが D95/D97 の固定点定義と整合する。

---

## 3. worldReclaimCount の意味

### 3.1 定義（getter 側）

`ISRRetireRouter::worldReclaimCount()` (`ISRRetireRouter.cpp:400-407`):

```cpp
return provider_->worldReclaimCount()          // Deferred
     + m_retireQuarantine.worldReclaimCount()  // Q
     + m_emergencyQuarantine.worldReclaimCount() // E
     + m_terminalReclaim.reclaimCount();        // Terminal (sync含む)
```

各 `worldReclaimCount()` は `consumeAtomic(worldReclaimCount_, acquire)` の monotonic 累積。

### 3.2 increment 側

§2 の全 9 LP で `fetchAdd(1, acq_rel)` / `++reclaimCount_` が `deleter` 成功直後に実行される。`type==World` 以外では increment しない。

### 3.3 意味の確定

```
worldReclaimCount() == Deferred World destruction + Q World destruction
                     + E World destruction + Terminal World destruction
                     == 物理的に破壊された World 数（累積・monotonic）
```

getter と increment の両側から証明され、`worldReclaimCount delta` は `物理破壊数` と 1:1。D86 の「type==World の terminal deleter 実行数・D83.2 sampler が読む一次情報源」は生産コードと完全一致する。

---

## 4. ReferenceObserver の意味

`WorldRetirementReferenceObserver` (`ISRWorldRetirementReference.h:25-45`):

- `referenceReleaseCount_ fetchAdd(1)` + `telemetry_->onReleaseObserved()` + `updateRunningMax()` を `onRelease()` で実行。
- `referenceReleaseCount` 自体は `referenceOutstanding = acquire - referenceReleaseCount` の running max 用の独立カウンタ。
- D94/D98 の `measurement-only / non-owning / ownership・reclaim authority を持たない` は生産コードの `non-owning` ポインタと `onRelease()` の `noexcept / 所有権変更なし` 実装と一致する。
- ただし現コードは `onRelease()` が必ず `telemetry.onReleaseObserved()` へ転送するため、`referenceReleaseCount_` と `telemetry.releaseObserved_` が結合している — これが double-count の直接原因（R1-5）。

---

## 5. Option A（第一候補）

```
physical destruction → worldReclaimCount → Timer sampler → releaseObserved_
referenceObserver → referenceReleaseCount_ → reference running max (のみ)
```

- `ReferenceObserver.onRelease()` は `referenceReleaseCount_++` と `updateRunningMax()` のみに縮退し、`telemetry.onReleaseObserved()` への転送を断つ。
- `Telemetry.releaseObserved_` は sampler の `addReleaseObserved(delta)` のみで更新される。

適合性: `worldReclaimCount` 機構と sampler は既に全 path で実装済み。D86/D83.2 の定義をそのまま authoritative にできる。最小侵襲。

---

## 6. Option B

```
physical destruction → referenceObserver.onRelease() → releaseObserved_
worldReclaimCount は T1 measurement から切り離し（診断専用化 or 廃止）
```

- sampler の `addReleaseObserved(delta)` を止め、`onReleaseObserved()` のみを計数する。
- `worldReclaimCount` は残すとしても T1 との一致保証を失う。

適合性: `referenceObserver` は全 drain path で既に呼ばれるため成立はするが、D86 の `worldReclaimCount authoritative` 定義・`worldReclaimCount()` の既存 sampler 用途・診断用途との再整理が必要。R2 では採用ではなく比較検証までとする。

---

## 7. 比較

| 観点 | Option A | Option B |
|---|---|---|
| D86 整合性 | ◎ そのまま authoritative | × 定義を書き換え必要 |
| 現コード再利用 | worldReclaimCount + sampler をそのまま流用 | reference path を流用、sampler を停止 |
| 変更行数 | 1行（Reference.h:42-43 転送除去） | sampler 停止 + lastSampled 削除 or 無効化 |
| `worldReclaimCount` の将来 | authoritative のまま診断も兼ねる | 診断専用化 or 未使用化、文書再定義が必要 |
| reference max への影響 | reference max は独立して維持（分離） | reference max と T1 max が同一 source に統合 |
| Window/max への波及 | sampler 経由のため window/sampler 設計は不変 | event-driven 化するため window 設計の再定義が必要 |
| リスク | D100.4 の「独立観測」文言を訂正すれば済む | D86/D83.2 の大幅訂正が必要 |

---

## 8. 採用案

**Option A を採用する。**

理由:

1. `ConvoPeq.md` 自身が `Telemetry.h:71-72` で `releaseObserved` を `worldReclaimCount の累積差分` と定義しており、生産コードの getter/increment 両側がそれを実装している（§3）。
2. Observer の契約は `measurement-only` であり retirement authority ではないため、T1 measurement の一次情報源を storage 側に置く方が責務分離（D83.2）と整合する。
3. 最小侵襲で double-count を解消できる（次章の required changes 参照）。

Option B は R1 証拠上は成立するが D86/sampler の再設計コストが高く、R2 では不採用とする。

---

## 9. releaseObserved semantic contract（R2 不変条件）

採用案 A の下で以下を不変条件として固定する。

```
releaseObserved_ == sampled physical World destruction count
                 == Σ worldReclaimCount delta (sampler が Non-RT で反映)
```

禁止状態:

```
releaseObserved_ == event-driven observation + sampled observation  // 禁止（二重）
```

不変条件:

```
1 physical World destruction → 1 releaseObserved increment
```

`onReleaseObserved()` は sampler 以外からは呼ばない。`addReleaseObserved(delta)` が唯一の正規入口とする（§12 参照）。

---

## 10. Acquire/Release symmetry

T1 基本式 `outstanding = acquired - released` は両辺が同じ意味論でなければならない。

Acquire 側 (`ISRWorldRetirementTelemetry.h:80-82`):

```cpp
void onAcquireObserved() { fetchAdd(acquireObserved_,1,acq_rel); }
// LP = publish 成功（onRuntimePublishedNonRt・CoordinatorLoop Non-RT・atomic のみ）
```

| Publish 分類 | acquire increment |
|---|---|
| bootstrap publish | 1（成功時） |
| normal publish | 1 |
| direct publish | 1 |
| orchestrator publish | 1 |
| recovery | 1 |
| Timer 起因 publish | 1 |
| rejected publish (publish 失敗) | 0（必須） |

生産コードの `onRuntimePublishedNonRt` は `publish 成功時のみ` に `onAcquireObserved()` を呼ぶ設計であり、D76.3 の `successful retirement-producing publish = exactly 1 acquire` と一致する。`rejected` で acquire しないことは `A` の過大計上を防ぐ必須条件。

Release 側は §9 のとおり `1 destruction = 1 release` に一本化されるため、acquire/release はともに対称に `1:1` で outstanding に反映される。

---

## 11. Window への影響

R2 中は window/max の再評価を停止する（指示 R2-6）。

Option A 採用後の window への影響はなし — sampler が既に `100ms 周期・Non-RT・A/R loads → signedWide(A)-signedWide(R) → max → window tag`（Telemetry.h:420-433, D82/D83/D86）を担っており、release source を一本化しても window/sampler 設計は不変である。R3 以降で `releaseObserved` 正常化後に T1-C3 を再開する。

---

## 12. 必要なコード変更点（R2 では実施しない — 設計のみ）

> コード変更は本R2では禁止。R3 で minimal correction として実施する。

- `src/audioengine/ISRWorldRetirementReference.h:39-44` — `onRelease()` から `telemetry_->onReleaseObserved()` 転送を除去し、`referenceReleaseCount_++` + `updateRunningMax()` のみにする。`telemetry_` メンバは reference max が telemetry を参照しないなら削除可能だが、互換性のため残す選択肢もある（R3 で決定）。
- `src/audioengine/ISRWorldRetirementReference.h:24-38` コメント — D100.4 の「telemetry へ転送（独立観測）」記述を「分離観測（telemetry へ転送しない）」に訂正。
- `src/audioengine/ISRWorldRetirementTelemetry.h:85-91` `onReleaseObserved()` — sampler 専用入口であることを明記し、外部からの event-driven 呼び出しを禁止するコメントを追加。削除はしない（sampler 以外の誤用防止のため存置 or `private` 化は R3 で選択）。
- 変更しないもの: `DeferredDeletionQueue.h` / `RetireQuarantineStore.h` / `ISRRetireRouter.cpp/h` の `worldReclaimCount` increment、`ISRRetireRouter::worldReclaimCount()` 集計、Timer sampler (`AudioEngine.Timer.cpp:425-430`) は全てそのまま authoritative として維持。

---

## 13. 必要な設計文書変更点（R2 では実施しない — 棚卸しのみ）

| 文書 | 現記述 | 必要な統一（Option A 方向） |
|---|---|---|
| D86 | `worldReclaimCount authoritative`（一次情報源） | 維持 — authoritative を明記し `releaseObserved = sampled worldReclaimCount delta` を正とする |
| D94 | `measurement-only / non-owning` | 維持 — Observer が T1 release の authoritative でないことを明記 |
| D95 | 固定点 3/4（terminalization boundary / event-driven） | 維持 — Observer の event-driven は reference max 用に限定 |
| D96 | Window 境界（Start/End は sampler boundary） | 維持 — 変更なし |
| D97 | `onRelease() は例外/所有権変更/reclaim 再試行なし` | 維持 |
| D98 | `reference observer non-owning` | 維持 — `telemetry へ転送しない` を追記 |
| D99 | `reference observer = event-driven running max` | 維持 — T1 measurement の一次情報源ではないことを明記 |
| D100 | `independent observation` | **訂正** — 「独立観測」→「分離観測（separate observation）」に変更。二重計上を許容しない旨を明記 |
| D100.4 | `sampler の outstanding 推定を正しくするため telemetry.releaseObserved にも転送（独立観測）` | **削除/訂正** — 転送を否定し `referenceReleaseCount` と `releaseObserved` は別カウンタであることを明記 |
| D101-9 | T1-C1 関連の release semantics | 本R2の §9 不変条件を反映 |

検索キーワード `authoritative / independent observation / sampler / reference observer / worldReclaimCount` は全て上記 D86-D101 に集約され、他文書への波及は小さい。

---

## 14. R2 verdict

```
OPTION-A READY
```

- R1 COUNTEREXAMPLE の原因である `同一 destruction → 同一 counter への二重経路` を Option A（`worldReclaimCount → sampler → releaseObserved` 一本化）で解消できることを 9 LP と getter/increment 両側から証明した。
- Option B は成立するが D86/sampler の再定義コストが高く R2 では不採用。
- 本R2は設計決定のみでコード変更・測定再開は行わない。次は `R3 minimal correction design` → 実装 → `T1-C1 再検証` → `T1-C2..C8` の順に進む。
- `K_terminal` と `R gate` は本 double-count 解消まで着手しない（指示どおり停止を維持）。

---

## 付録 — Tool Coverage（R2 全系統を使用）

| 系統 | ツール | 実行 | 結果 |
|---|---|---|---|
| WSL#1 | rg 15.1.0 | `rg -n "worldReclaimCount\|onRelease\|releaseObserved" src/` | §2-3 全ヒット列挙 |
| WSL#2 | ast-grep sg 0.44.0 | `sg run --pattern "onRelease"`（`search` 非対応を確認） | `ast-grep --help` 動作確認、rg で代替列挙を補完 |
| WSL#3 | fdfind 10.3.0 | `fdfind -t f "ISRWorldRetirement"` | Telemetry/Reference 発見 |
| WSL#4 | ag 2.2.0 | `ag "worldReclaimCount" src/` | Deferred/Quarantine/Router 一致 |
| WSL#5 | fzf 0.67.0 | `printf ... | fzf --filter` | フィルタ動作確認 |
| WSL#6 | sed 4.9 | `sed -n "65,120p" Telemetry.h` | release semantics 確認 |
| WSL#7 | awk 5.3.2 | `awk "/worldReclaimCount/"` | increment 行特定 |
| MCP#1 | serena | `find_symbol worldReclaimCount/onRelease` | 8 symbols / 2 symbols を確定（Deferred/Quarantine/Router/Telemetry/Reference） |
| MCP#2 | ccc | `ccc search "worldReclaimCount"` | worldReclaimCount 検索で同ファイル群ヒット |
| CLI#1 | graphify 0.9.48 | `graphify --help` | `graphify.exe` 動作確認（graphify-out 未生成のため serena/semble で補完） |
| CLI#2 | semble 0.5.5 | `semble search "worldReclaimCount"/"onReleaseObserved"` | Telemetry/Reference の定義を top hit として取得 |
| MCP#3 | AiDex | `aidex_query onRelease` 12 hits | 全 onRelease サイトと一致 |
| 文献 | crossbeam-epoch/rigtorp/serena/cocoindex/ast-grep/semble/AiDex/graphify/headroom | urllib 200 OK 9件（Vyukov SSL失効→rigtorp代替） | D69/D86 補助参照 |

---

*Evidence generated: Phase I-T1-C1-R2 — code change prohibited until R3. Do not proceed to K_terminal / R gate.*
