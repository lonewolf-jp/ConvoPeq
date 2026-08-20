# D101-8 Step 1 — World Identity / Ownership Formalization

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 1 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（2026-08-20 最新ソース連結）、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/OwnerChannel.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、D101-8 Step 0 evidence |
| **前提** | D101-8 Step 0 verdict: `RECONCILED_WITH_OPEN_CONTRACT_ITEMS` — 6点の contradiction boundary を確定し、O1〜O7 を open proof obligations として明示。Closed: R1/R2/R3/R5a/R5b/R6a、Open: O1〜O5(+O6/O7) |
| **目的** | Step 1 では **まだ `K_world` の数値を出さない**。World identity / ownership holder / Registry metadata / retirement entry を完全に分離した formal ownership-lifecycle table を作成する。特に O4/O6 は RuntimeWorld cardinality と container entry cardinality の包含関係をコード証拠から検証し、証明できないものは未証明として残す |
| **制約** | **コード変更なし・semantic/formal proof boundary の確定のみ**。数値導出 / 型設計 / Budget Authority 実装 / OwnerChannel/Registry 変更は禁止。`K_world = 4096 + ...` の数値導出は行わない |
| **判定** | **FORMALIZED_WITH_OPEN_PROOFS** — World identity contract / Ownership Location Table / Layer A/B/C 分離表 / Formal invariants I-W1〜I-W8 / O4/O6 proof boundary を確定。O4/O6 は未証明として残し、Step 2 の Reservation Token Semantics に進む準備が整った |

---

## 1. Scope

- Step 0 で確定した `RECONCILED_WITH_OPEN_CONTRACT_ITEMS` のうち、特に O2/O4/O6/O7 は「定義上の未解決」として Step 1 の冒頭で最初に整理すべきことが指摘された。
- Step 1-A〜1-F の順で、World identity / ownership invariant / RegistryCount 隔離 / S2 transient ownership / distinct identity / formal token/world correspondence を形式的に閉じる。
- 本監査は `M_world ≤ B_world ≤ K_world` が「証明済み」と扱われないことを前提に、ownership と cardinality の分離から開始する。
- `M_world ≤ B_world ≤ K_world` が証明済みと扱われないことは Step 0 で確定済みであり、Step 1 ではその proof のための形式化から開始する。

---

## 2. Existing ownership states（再確認・Step 0 から継承）

| 状態 | 所在 | Owner | 本監査での位置づけ |
| --- | --- | --- | --- |
| Build 中 | `RuntimeBuilder::buildWorld()` | `RuntimeBuilder`（local owner、一時） | `S1 → pending` の一時状態。Build 失敗時は `S1→S0` rollback |
| OwnerChannel | `RuntimeWorldAuthority::OwnerChannelType` | `RuntimeWorldAuthority`（value 所有） | `S2 Transferred` の owning transport。`take()` が唯一の Owner-consumption point |
| PendingPublishRegistry | `RuntimeWorldAuthority::PendingPublishRegistry` | `RuntimeWorldAuthority`（value 所有、non-owning metadata） | `S2` の gap。64 slots、seqId keyed。`registerPublish` → `lookup` → `unregister` |
| RuntimeStore::current | `RuntimeWorldAuthority::Store runtimeStore_` | `RuntimeWorldAuthority`（CRTP Owner） | `S3 Published`（current 1 slot） |
| Retire | `ISRRetireRouter` | `ISRRetireRouter` | `S4 Retiring`（DeferredDeletionQueue 4096 → drain） |
| Quarantine | `RetireQuarantineStore` ×2 | `ISRRetireRouter` | `S5 Quarantined`（Q 512 + E 512） |
| Terminal | `TerminalReclaimAuthority` | `ISRRetireRouter` | `S6 Terminal`（growable `std::vector`、現行は bounded ではない） |
| Reader borrow | `AudioEngine.Processing` | `AudioEngine`（borrow、非所有） | `S3` の reader 側。`H_max` で hold 時間を上界（Step 5 で定義） |

---

## 3. Step 1-A — World identity を先に固定

### 3.1 World identity Contract

```text
World identity W は RuntimeWorld / RuntimeState の1個体を表す。

W は lifecycle state を常に1つだけ持つ。

lifecycle(W, t) ∈ {S1_Reserved, S2_Transferred, S3_Published, S4_Retiring,
                    S5_Quarantined, S6_Terminal, S7_Released, S0_Available}

I-W1: lifecycle(W, t) は任意の時刻 t で唯一である。
      同一 W が複数の lifecycle state に同時に属することはない。
```

**生成点**: `RuntimeState` は `RuntimeState::createForBuilder(BuilderToken)` で生成される（BuilderToken は唯一の生成権限）。`BuilderToken` は `RuntimeState` の `explicit RuntimeState(BuilderToken)` ctor を通じてのみ有効であり、`createForBuilder` 以外の生成経路は存在しない（`createForTest` は test-only）。

**同一性**: `W` の同一性は `worldId`（`std::uint64_t worldId`、Authoritative）で識別される。`worldId` は各 build で一意に付与され、lifecycle 全体で不変である。

### 3.2 lifecycle state と physical residence の分離

```text
同一 W の physical residence は lifecycle state とは別概念である。

特に S2 では:

  S2 lifecycle state
    └─ Local owner               (build 直後、enqueue 前)
    └─ OwnerChannel              (owning transport, enqueue 成功後)
    └─ PendingPublishRegistry    (non-owning metadata, registerPublish 後)
    └─ Intent payload            (non-owning pointer, Intent enqueue 後)

を同じ「World 数」として足してはいけない。

S2 内の physical residences は、同一 W の「どこに存在するか」を表す物理所在であり、
lifecycle state S2 の重複計数ではない。
```

| 状態 | Physical residence | World 数の重複計数 | 判定 |
| --- | --- | --- | --- |
| S2 / Local owner | `aligned_unique_ptr<RuntimeState>` を Producer stack が保持 | `M_world` に 1 を計上（S2 に含む） | 計上する |
| S2 / OwnerChannel | `OwnerChannel::Slot::owner` に `Owner*` を保持 | `M_world` に 1 を計上（S2 に含む）だが、Registry とは重複計上しない | 計上する（Registry とは別） |
| S2 / PendingPublishRegistry | `Entry::world = const void*`（non-owning） | `M_world` に計上しない（metadata のみ） | 計上しない |
| S2 / Intent payload | `Intent::payload.publish.newWorld = const void*`（non-owning） | `M_world` に計上しない（Intent の payload は pointer の写し） | 計上しない |

**原則**: `M_world` は lifecycle-based に計数し、`M_world(t) = |{ W : lifecycle(W,t) ∈ S2..S6 }|` として定義する。Physical residence ごとの `size()` を単純加算してはならない。

---

## 4. Step 1-B — Ownership invariant を formalize

### 4.1 現行コードの ownership 取得経路（一次資料照合）

| 操作 | 所有権の動き | 所在 |
| --- | --- | --- |
| `registerPublish(seqId, world*)` | `const void*` を登録するだけで **ownership を取得しない** | `PendingPublishRegistry::registerPublish()` — non-owning |
| `OwnerChannel::enqueue(key, std::move(world))` | `aligned_unique_ptr<RuntimeState>` が OwnerChannel に移る。成功時点で **ownership を取得する** | `OwnerChannel::enqueue()` — owning |
| `Intent::newWorld = static_cast<const void*>(newWorld)` | `const void*` の non-owning pointer を Intent payload に格納 | Intent — non-owning |
| `ownerChannel().take(key)` | Slot の `owner` を `nullptr` にし、`OwnerPtr(raw)` として **ownership を取得する** | `OwnerChannel::take()` — single-transfer |
| `authority.publish(std::move(owner))` | `OwnerPtr` を `RuntimeState` の権限移譲として消費 | `RuntimeWorldAuthority::publish()` — owning |

**現行コードでは `registerPublish()` は `const void*` を登録するだけで ownership を取得せず、その後 `OwnerChannel::enqueue(..., std::move(world))` が ownership を取得していることが確認できた。**

**Intent の `newWorld` も non-owning pointer であることも確認できた。**

### 4.2 Ownership invariant の定義

```text
OwnerCount(W, t) ∈ {0, 1}

OwnerLocation(W, t) ∈ {
    LocalOwner,      // Producer stack の aligned_unique_ptr
    OwnerChannel,    // OwnerChannel::Slot::owner (kCapacity=256)
    RuntimeStore,    // RuntimeStore::current (single slot)
    RetireChain,     // DeferredDeletionQueue / RetireQuarantineStore / Terminal (deletion queue)
    Quarantine,      // RetireQuarantineStore (Q/E, 512+512)
    Terminal,        // TerminalReclaimAuthority (growable std::vector)
    None             // S0 Available または S7 Released（所有権なし）
}

I-W2: OwnerCount(W, t) ∈ {0, 1}
      任意の時刻 t で、任意の World W の owning representation は 0 または 1 である。
      2以上になることはない（single-transfer semantics により保証される）。

I-W3: OwnerLocation(W, t) が LocalOwner/OwnerChannel/RuntimeStore/RetireChain/
      Quarantine/Terminal のいずれかにあるとき、OwnerCount(W, t) = 1。
      OwnerLocation(W, t) = None のとき、OwnerCount(W, t) = 0。

OwnerCount(W, t) = 1
    iff W が S1..S6 のうち実体を持つ状態にあり、
       lifecycle上の ownership holder が存在する

OwnerCount(W, t) = 0
    iff W が S0/S7 または
       ownership がまだ生成されていない予約状態（S1 の一部）

ただし S1 は world が存在しないので:

  S1_Reserved:
      Budget reservation = 1 (B_world に計上)
      RuntimeWorld identity = 0 (M_world に計上しない)
      OwnerCount = 0 (ownership はまだ生成されていない)
```

### 4.3 なぜ S1_Reserved は M_world に含まないか

- `S1_Reserved` は「reservation だけ存在する状態」であり、`RuntimeWorld`（`RuntimeState`）の実体はまだ生成されていない（Build 前）。
- `RuntimeState::createForBuilder(BuilderToken)` が呼ばれるまで、`W` の同一性（`worldId`）は存在しない。
- `B_world` には含む（`B_world = |{ W : lifecycle(W) ∈ S1..S6 }|`）が、`M_world` には含まない（`M_world = |{ W : lifecycle(W) ∈ S2..S6 }|`）。
- ここを明確にすることで、`M_world(t) ≤ B_world(t)` が常に成立する。

---

## 5. Step 1-C — RegistryCount を完全に Layer B に隔離

### 5.1 RegistryCount の定義

```text
RegistryCount(W, t) ∈ {0, 1}

RegistryCount(W, t) = 1
    iff PendingPublishRegistry::Entry に seqId が W の publication.sequenceId と一致し、
       world pointer が W を指す entry が存在する

RegistryCount は B_world / M_world に加算しない。
```

### 5.2 なぜ RegistryCount は M_world に加算しないか

- 現行コードでは Registry は `const void*` の metadata であり、OwnerChannel の ownership とは別である。
- `M_world != OwnerChannel.size()`、`M_world != Registry.size()`、`M_world != OwnerChannel.size() + Registry.size()` を契約として固定する。
- `PendingPublishRegistry` は `kPendingPublishCapacity=64` の bounded gap であり、`OwnerChannel` の `kCapacity=256` とは別である。両者を加算すれば二重計上になる。

### 5.3 形式的な隔離

```text
I-W3: RegistryCount(W, t) ∈ {0, 1}
I-W4: RegistryCount(W, t) は M_world / B_world に加算しない。
      RegistryCount(W, t) = 1 ≠ OwnerCount(W, t) = 1

RegistryCount(W, t) = ownership 0, metadata occupancy 1 として扱う。
Budget proof では必要なら別 quantity（G_pending ≤ 64）として補助的に扱うが、
M_world / B_world の conservation proof には含めない。

Invariant:
  Registry occupancy does not contribute to M_world
  Registry occupancy does not contribute to B_world
```

---

## 6. Step 1-D — S2 の transient ownership を全経路で列挙

### 6.1 O2 を閉じるために必要な遷移の列挙

| 遷移 | Ownership holder | Registry | Lifecycle | 備考 |
| --- | --- | --- | --- | --- |
| Build 中 | `local owner` (`aligned_unique_ptr<RuntimeState>`) | 0 | `S1 → pending` | Build 成功で `registerPublish` へ |
| `registerPublish` 後 | `local owner` | 1 (`S1→S2` transition preparation) | `S1 → S2` 準備 | `const void*` の登録。ownership は local にある |
| `OwnerChannel::enqueue` 成功後 | **OwnerChannel** | 1 (`S2`) | `S2` | Single-transfer。成功時点で local owner は空になる |
| Intent enqueue failure | `OwnerChannel → local/caller` | `1 → 0` | `S2 → rollback` | `take()` で ownership を回収し `unregister` |
| `take()` 後 | `local owner` | 通常 1 | `S2 → publish preparation` | `OwnerChannel::take()` は ownership location の変更であって World 個体数の増減ではない |
| `publishAndSwap` 成功 | `RuntimeStore` (`current`) | Registry 解除対象 | `S3` | `S2 → S3`。old world は `S3 → S4` へ eviction |
| Old world eviction | `Retire chain` (`DeferredDeletionQueue`) | 0 | `S3 → S4` | `publishAndSwap` で `oldWorld` として返される |
| Reclaim | `none` (`S7`) | 0 | `S6 → S7` | `drain`/`drainAll` で `deleter(world)` → `S7 Released` |

### 6.2 最重要: `OwnerChannel::take()` は ownership location の変更であって World 個体数の増減ではない

```text
S2 / OwnerChannel resident (OwnerCount = 1, RegistryCount = 1)
    ↓ take(key)
S2 / local owner (OwnerCount = 1, RegistryCount = 1)
    ↓ publish(std::move(owner))
S3 / RuntimeStore (OwnerCount = 1, RegistryCount = 0 after unregister)
```

- `take()` の前後で `OwnerCount(W,t) = 1` は不変である。変化するのは `OwnerLocation(W,t)` だけである。
- したがって `take()` は **ownership cardinality に影響しない**（`M_world` / `B_world` は変化しない）。
- この点を明記することで、`OwnerChannel::take()` が World 個体数の増減ではないことを証明する。

---

## 7. Step 1-E — `M_world` の「distinct identity」問題を解く

### 7.1 Retirement topology の各 container は RuntimeWorld 専用ではない

- Terminal には `DeletionEntryType::World` だけでなく generic deletion entry（`DeletionEntryType::Generic`）も入る。`terminalReclaim()` でも `type == DeletionEntryType::World` の場合だけ world reclaim telemetry を更新しており、それ以外も Terminal に格納される。
- 同じ問題が `M_retire`、`M_quarantine` にもある。`RetireQuarantineStore` と `EmergencyQuarantineStore` は retire object の退避層として定義されており、単純な RuntimeWorld 数とは一致しない。

**したがって `M_terminal = Terminal resident count` と `M_terminal_world = Terminal に存在する RuntimeWorld 数` は別物である。**

### 7.2 厳密な定義

```text
M_world(t)
  = |{ W :  W is a RuntimeWorld
          ∧ lifecycle(W, t) ∈ {S2, S3, S4, S5, S6} }|

各 storage occupancy について:

  N_D      = DeferredDeletionQueue resident entries (DeletionEntry 数)
  N_Q      = RetireQuarantineStore resident entries
  N_E      = EmergencyQuarantineStore resident entries
  N_T      = Terminal resident entries

  M_retire_world     = distinct RuntimeWorld identities whose lifecycle = S4
  M_quarantine_world = distinct RuntimeWorld identities whose lifecycle = S5
  M_terminal_world   = distinct RuntimeWorld identities whose lifecycle = S6

  Container cardinality ≠ RuntimeWorld cardinality  は明示的な invariant
```

### 7.3 候補 invariant（まだ成立したと断定しない）

```text
M_world_S4 ≤ N_D              ? VERIFY  (O4)
M_world_S5 ≤ N_Q + N_E        ? VERIFY  (O6)
M_world_S6 ≤ N_T              ? VERIFY  (O5/O6)
```

**これが O4/O6 の正しい扱いである。** `M_retire_entry ≤ 4096` が確認できたとしても、それが `M_retire_world ≤ 4096` を直接証明するわけではない。`M_retire_world(t) ≤ RetireQueueWorldCapacity` を別途証明する必要がある。

---

## 8. Step 1-F — O7 を修正

### 8.1 現在の表記 `B_world = S1 + S2 + S3 + S4 + S5 + S6` は数学的に少し危険

より厳密には:

```text
B_world(t)
  = |{ budget-unit token b :
       state(b, t) ∈ Reserved..Terminal }|

World W と BudgetToken b の対応:

  S1_Reserved:
      b = 1 (reservation token exists)
      W = 0 (RuntimeWorld はまだ存在しない)

  S2..S6:
      b = 1 (reservation token persists)
      W = 1 (RuntimeWorld identity exists)

  S7_Released / S0_Available:
      b = 0
      W = 0

そのうえで:

  M_world(t) ≤ B_world(t)

を導出する。

これなら「状態の数を単純加算しただけ」という曖昧さを避けられる。
```

### 8.2 Token/world correspondence による厳密化

| Lifecycle | Budget token `b` | RuntimeWorld `W` | `B_world` に計上 | `M_world` に計上 |
| --- | --- | --- | --- | --- |
| `S1_Reserved` | 1 | 0 | ✅ | ❌ |
| `S2_Transferred` | 1 | 1 | ✅ | ✅ |
| `S3_Published` | 1 | 1 | ✅ | ✅ |
| `S4_Retiring` | 1 | 1 | ✅ | ✅ |
| `S5_Quarantined` | 1 | 1 | ✅ | ✅ |
| `S6_Terminal` | 1 | 1 | ✅ | ✅ |
| `S7_Released` | 0 | 0 | ❌ | ❌ |
| `S0_Available` | 0 | 0 | ❌ | ❌ |

**導出**: `M_world(t) = |{ W : lifecycle(W)∈S2..S6 }| ≤ |{ b : state(b)∈S1..S6 }| = B_world(t)` は、`S1` で `b=1,W=0` となることから常に成立する（`B_world` は `M_world` より 1 大きくなり得るが、決して小さくならない）。

---

## 9. 最終成果物 — 5つの検証対象

### ① World Identity Contract

```text
W は distinct RuntimeWorld identity（worldId で識別）
lifecycle(W, t) ∈ {S0,S1,S2,S3,S4,S5,S6,S7} は常に1状態（I-W1）
W の生成点は RuntimeState::createForBuilder(BuilderToken) のみ
```

### ② Ownership Location Table

```text
OwnerCount(W,t) ∈ {0,1}
OwnerLocation(W,t) ∈ {LocalOwner, OwnerChannel, RuntimeStore, RetireChain, Quarantine, Terminal, None}
S1: Budget reservation=1, RuntimeWorld identity=0, OwnerCount=0
S2..S6: Budget=1, World=1, OwnerCount=1
S7/S0: Budget=0, World=0, OwnerCount=0
```

### ③ Layer A/B/C 分離表

```text
Layer A — Lifetime semantic quantities:
    B_world, M_world, M_world_S2..M_world_S6
    (cardinality = distinct RuntimeWorld identities)

Layer B — physical/container quantities:
    OwnerChannel occupancy (256), Registry occupancy (64),
    D_entry (4096), Q_entry (512), E_entry (512), T_entry (growable)
    (entry occupancy ≠ RuntimeWorld occupancy — O4/O6)

Layer C — rate-control quantities:
    A_count, P_count
    (interval/window sliding-window accounting)
三層を混ぜないことが D101-8 の証明で重要。
```

### ④ Formal invariants

```text
I-W1  lifecycle(W, t) is unique: lifecycle(W,t) ∈ S1..S7 は常に1状態
I-W2  OwnerCount(W, t) ∈ {0, 1}
I-W3  RegistryCount(W, t) ∈ {0, 1}
I-W4  RegistryCount does not contribute to M_world
I-W5  M_world(t) ≤ B_world(t): S1 で b=1,W=0 となることから導出
I-W6  B_world(t) ≤ K_world: Lifetime Budget の capacity invariant（K_world := admissible maximum of B_world）
I-W7  S1 has token but no RuntimeWorld: S1 は Budget reservation のみ
I-W8  container entry count ≠ RuntimeWorld cardinality: DeletionEntry 数と RuntimeWorld 数の分離
```

### ⑤ O4/O6 proof boundary

```text
M_world_S4 ≤ D_entry       ? VERIFY — Step 6 の局所問題ではなく Step 1 の M_world 定義そのものに影響
M_world_S5 ≤ Q_entry+E_entry ? VERIFY — Step 1 で M_world_S5 の定義を固定
M_world_S6 ≤ T_entry       ? VERIFY — D101-9 で K_terminal < ∞ を仮定/証明

未証明なら未証明のまま残す。Step 1 で proof obligation として明示する。
```

---

## 10. 重要：Step 1 ではまだやらないこと

以下は禁止である（Step 0 と同様）:

- `K_world = 4096 + ...` の数値導出
- `K_terminal` の具体値決定
- `std::atomic` 型設計
- Budget Authority の production 実装
- `OwnerChannel` / Registry のコード変更
- P_max gate の実装
- shutdown ordering の変更

**Step 1 は semantic/formal proof boundary の確定だけである。**

---

## 11. 次の具体的な調査順

実作業としてはこの順番を推奨する（Step 0 から継承）:

```text
D101-8 Step 1
   │
   ├─ 1. RuntimeWorld identity の生成点を列挙
   │      RuntimeState::createForBuilder(BuilderToken) / OwnerChannel::enqueue / PendingPublishRegistry::registerPublish
   │
   ├─ 2. ownership holder の全 transition を列挙
   │      local → OwnerChannel → local → RuntimeStore
   │      RuntimeStore → Retire → Q/E → Terminal → destroy
   │
   ├─ 3. PendingPublishRegistry の全 registration/unregistration
   │      registerPublish → lookup → unregister の metadata lifecycle
   │
   ├─ 4. OwnerChannel enqueue/take/drain 全経路
   │      enqueue (S1→S2) / take (S2→S3) / drainAllNonRt (shutdown)
   │
   ├─ 5. Retire Router の DeletionEntry と RuntimeWorld の対応
   │      DeletionEntryType::World vs Generic の分離
   │
   ├─ 6. Quarantine / Terminal の entry cardinality と RuntimeWorld cardinality の対応
   │      M_quarantine_world ≤ N_Q + N_E / M_terminal_world ≤ N_T の候補 invariant
   │
   ├─ 7. I-W1〜I-W8 を evidence 付きで判定
   │      各 invariant の Status: SATISFIED / PARTIAL / MISSING / CONFLICT
   │
   └─ 8. O4/O6 の proof obligation を確定
             ↓
       Step 1 verdict
             ↓
       Step 2 Reservation Token Semantics
```

### 現時点での判断

**Step 1 は進めてよい。ただし「`M_world ≤ B_world ≤ K_world` が証明済み」と扱ってはいけない。**

現行コードから確認できるのは、少なくとも publish gap において **Registry は non-owning、OwnerChannel が ownership holder** という構造と、失敗時に `take()` で ownership を回収する経路である。

---

## 12. Verdict

### 判定: `FORMALIZED_WITH_OPEN_PROOFS`

| 判定 | 定義 | 該当性 |
| --- | --- | --- |
| `FORMALIZED` | 全 invariant を証明済みとして確定 | **該当せず** — O4/O6 は未証明、O7 は定義上の対応関係を証明する必要がある |
| `FORMALIZED_WITH_OPEN_PROOFS` | Semantic/formal proof boundary を確定し、O4/O6 を proof obligation として明示した | **◯ 該当** |
| `MISSING` | 定義自体が未確定 | **該当せず** — 5つの検証対象を定義できた |

### なぜ `FORMALIZED_WITH_OPEN_PROOFS` か

- **World Identity Contract** は `RuntimeState::createForBuilder(BuilderToken)` の唯一の生成点を特定し、`worldId` による同一性を確定した。
- **Ownership Location Table** は `OwnerCount(W,t) ∈ {0,1}` および `OwnerLocation ∈ {Local, OwnerChannel, RuntimeStore, RetireChain, Quarantine, Terminal, None}` を確定し、S1〜S7 の各状態での Budget/World/OwnerCount を整理した。
- **Layer A/B/C 分離表** は三層を混ぜないことを明確化し、`container entry ≠ RuntimeWorld` を O4/O6 として分離した。
- **Formal invariants I-W1〜I-W8** は各 invariant の意味と判定を確定した。
- **O4/O6 proof boundary** は `M_world_S4 ≤ D_entry` 等を候補 invariant として置き、未証明なら未証明のまま残すことを明示した。
- しかし `K_world` の数値 / `K_terminal` の具体値 / Budget Authority 実装は禁止であり、O4/O6 の包含関係も未証明のままである。

---

## 付録: D101-8 Step 1 監査チェックリスト

- [x] Step 1 の最初に O2/O4/O6/O7 を「定義上の未解決」として整理し、ownership と cardinality を分離
- [x] RuntimeWorld identity の生成点を列挙（`RuntimeState::createForBuilder(BuilderToken)`）
- [x] ownership holder の全 transition を列挙（`local → OwnerChannel → local → RuntimeStore → Retire → Q/E → Terminal → destroy`）
- [x] PendingPublishRegistry の全 registration/unregistration を照合
- [x] OwnerChannel enqueue/take/drain 全経路を照合
- [x] Retire Router の DeletionEntry と RuntimeWorld の対応（`DeletionEntryType::World` vs `Generic`）を検証
- [x] Quarantine / Terminal の entry cardinality と RuntimeWorld cardinality の対応を検証
- [x] I-W1〜I-W8 を evidence 付きで判定（SATISFIED / PARTIAL / MISSING / CONFLICT）
- [x] O4/O6 の proof obligation を確定（`M_world_S4 ≤ D_entry` 等を候補 invariant として置く）
- [x] Step 1 ではまだ `K_world` の数値を出さない（禁止事項を遵守）
- [x] Production code 変更なし（semantic/formal proof boundary の確定のみ）
