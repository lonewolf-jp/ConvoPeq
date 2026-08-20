# D101-8 Step 0 — Contract Reconciliation / Proof Boundary Freeze

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 0 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（2026-08-19 最新ソース連結）、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/OwnerChannel.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/AudioEngine.Retire.cpp`、`src/audioengine/AudioEngine.Processing.*.cpp`、`src/audioengine/ISRWorldRetirementTelemetry.h`、D101-3/D101-4/D101-5/D101-6/D101-7 evidence |
| **前提** | D101-7 verdict: `CONTRACT_DEFINED` — Lifetime Budget State Machine（8状態 + 各遷移の唯一の Authority）、`K_world`/`A_max`/`P_max`/reservation token の invariant と ordering を定義。だが D101-7 と現行 `ConvoPeq.md` の間に6点の不整合が存在する |
| **目的** | D101-7 の `CONTRACT_DEFINED` を、現行コードと矛盾しない「証明可能な契約」に修正してから、具体的な `K_world / A_max / P_max / H_max / G_contract` の導出へ進む。`K_world` の数値導出・`std::atomic` 型設計・production code 変更は行わない |
| **制約** | **コード変更なし・Step 0 の再照合のみ**。6点（Terminal bounded/growable / `B_world`/`M_world` / `A_max` / `P_max` / OwnerChannel/Registry / shutdown ordering）を再照合し、D101-7 に contradiction が残らない契約へ修正する |
| **判定** | **RECONCILED_WITH_OPEN_CONTRACT_ITEMS — RECONCILIATION_SUBSTANTIALLY_COMPLETE + OPEN_CONTRACT_ITEMS。6点の contradiction boundary を確定し、Step 1 以降で解くべき contract obligations（O1〜O7）を明示した。ただし、R5/R6 の一部は現行実装事実と契約上の証明条件をまだ分離し切っていない。semantic definition は閉じたが、enforcement / conservation proof は未完成** |

---

## 1. Scope

- D101-7 で定義した Lifetime Budget State Machine / Contract（`K_world`/`A_max`/`P_max`/reservation token 等）と、現行 `ConvoPeq.md` の間に少なくとも4点の不整合が存在することが指摘された。本監査では6点に拡大して照合する。
- 本監査は **コード変更なし**で、D101-7 の契約を現行コードと矛盾しない「証明可能な契約」に修正する Step 0 である。
- 本 Step 0 が RECONCILED になるまで、`K_world` の具体値、`std::atomic` 型設計、production code 変更には進まない。
- 指摘の6点:

| # | 不整合 | D101-7 の記述 | 現行 `ConvoPeq.md` の実装 |
| --- | --- | --- | --- |
| 1 | Terminal bounded/growable | `TerminalReclaimAuthority<K>` bounded / `std::array<Entry, K>` / S5→S6 必ず成功 | `std::vector<Entry> entries_` growable / `store()` ALWAYS true |
| 2 | `B_world` vs `M_world` | `S1 Reserved` を `M_world` の live に含む | `Reserved` はまだ RuntimeWorld が存在しない |
| 3 | `A_max` semantics | `A_max = publish success` / interval に `A_max` を数える | Enforcement は `S0→S1 reserve`。count 対象と enforce 対象が不一致 |
| 4 | `P_max` semantics | `P_max = publish 成功数` / `S0→S1 reserve gate` で制御 | Reservation admission → build → transfer → publish intent → publishAndSwap の各段階があり、event mismatch |
| 5 | OwnerChannel / Registry ownership | `S2 stale/shutdown → S0` を「registry entry drop + reservation release」で済ませる | Ownership は `OwnerChannel` に移された後。`registry = non-owning metadata/gap` / `OwnerChannel = owning transport` の区別が必要 |
| 6 | Shutdown ordering | `Admission close → gap drain → retire drain → terminal drainAll → budget release` | 現行は `releaseResources` で timeout 付き `drainDeferredRetireQueues` → `tryReclaim` → shutdown clear → `OwnerChannel::drainAllNonRt()` → retire chain |

---

## 2. R1 — S6 Terminal の契約を修正する

### 2.1 現行コードの確認（`ConvoPeq.md` 一次資料）

```cpp
// ConvoPeq.md L54319 — class TerminalReclaimAuthority
class TerminalReclaimAuthority {
    // ★ P-4: Growable store (std::vector) — Non-RT only, heap allocation acceptable.
    //   Guarantees store() ALWAYS succeeds → no EBR-failure leak path.
    std::vector<Entry> entries_;
    mutable std::mutex mtx_;
    std::atomic<std::uint64_t> reclaimCount_{0};
    std::atomic<uint32_t> residentAtomic_{0};
    // store() ALWAYS returns true (growable store) — ownership always transfers.
    bool store(void* ptr, void (*deleter)(void*), uint64_t epoch,
               DeletionEntryType type, const char* reason) noexcept;
    void drain(uint64_t minReaderEpoch, const std::function<...>& isOlderFn) noexcept;
    void drainAll() noexcept;  // shutdown only — audio thread must be stopped
};

// L53949 — enqueueWithRetry の ownership chain
// ★ P-4: Ownership chain: D → Q → EmergencyQ → TerminalReclaimAuthority
// L54000 — Stage 5: TerminalReclaimAuthority
//   D+Q+E 全滿 → TerminalReclaimAuthority へ移送
//   (void)tstored;  // ★ P-4: 常に true（growable store）
```

- **確定**: 現行 `TerminalReclaimAuthority` は `std::vector` の growable store で、`store()` は常に `true`。全 bounded store が満杯でも Terminal が ownership を受領することで ownership leak を防いでいる。
- 現時点で `TerminalReclaimAuthority<K>` は存在しない。

### 2.2 Option A vs Option B

| 項目 | Option A | Option B |
| --- | --- | --- |
| 内容 | D101-9 の bounded 化を前倒しして D101-8 の前提にする。`S6 Terminal capacity = K_terminal < ∞`。`S5 → S6` は reservation-first により必ず slot available | D101-8 では現行 growable Terminal を正式に認める。S6 は容量無制限 |
| `K_world` の証明 | `K_world` は `Terminal` を含む有限 bound として証明可能（`K_terminal` を前提に `K_world ≤ 1+4096+1024+K_terminal+...`） | `K_world` は現状では Terminal を含む有限 bound として証明不可。`I01 / I10 / K_world finite proof` が未閉鎖 |
| 依存関係 | `D101-8: bounded Terminal が必要であることを形式化 → D101-9: bounded TerminalReclaimAuthority を具体設計 → D101-8 に戻って K_world proof を完成` | D101-8 で bounded 化を仮定せず、growable を前提に証明を進めるが、K_world の有限性は証明できない |

### 2.3 判定: Option A を採用（推奨どおり）

- **Option A を採用する方向で D101-9 と接続する。**
- D101-8 の finite-proof は `K_terminal < ∞` を未証明の前提条件として明示的にパラメータ化する。D101-8 自身が現行コードについて「bounded である」と誤認しない:

```text
D101-8 Step 0 (本監査):
    bounded Terminal が必要であることを形式化
    （現行 growable と D101-7 bounded の不整合を解消）
        │
        ▼
D101-8 Steps 1-8:
    B_world / M_world 分離、A_max / P_max / H_max / G_contract の導出
    K_world の形式的条件を導出するが、K_terminal の具体値は仮定として扱う
    （K_world ≤ 1+4096+1024+K_terminal+... の式で K_terminal を形式的パラメータとして扱う）
        │
        ▼
D101-9:
    bounded TerminalReclaimAuthority を具体設計
    （std::vector → std::array<K_terminal> + reservation-first の詳細設計）
        │
        ▼
D101-8 に戻って K_world proof を完成:
    D101-9 の K_terminal 具体設計を前提に、K_world の有限証明を閉鎖する
    （I01 / I10 / K_world finite proof が閉鎖される）
```

### 2.4 D101-7 契約の修正

| D101-7 の記述 | 修正後（本監査） |
| --- | --- |
| `S6 Terminal — TerminalReclaimAuthority<K> bounded` | **現行は growable `std::vector`。D101-8 では `K_terminal < ∞` が必要であることを形式化し、D101-9 で bounded 化を具体設計する。D101-8 の証明では `K_terminal` を形式的パラメータとして仮定する** |
| `S5→S6 は必ず成功` | **現行: S5→S6 = growable store により ownership acceptance failure はない。将来契約: bounded Terminal を導入するなら S5→S6 の acceptance を reservation-first で保証する。未証明: K_terminal < ∞ / reservation-first による slot conservation** |
| `K_world ≤ 1+4096+1024+K+...` | **D101-8 では `K_terminal` を形式的パラメータとして扱い、D101-9 で `K_terminal` の具体値と bounded 化の詳細設計後に証明を閉鎖する** |

### 2.5 現行コードとの矛盾の解消

- 本監査により、D101-7 の `S6 Terminal bounded` が **現行コードの growable 実装と矛盾する** ことが明示的に解消された。
- Option A の採用により、D101-8 は bounded Terminal の必要性を形式化し、D101-9 での具体設計を前提に証明を進めることが確定した。
- 本 Step 0 が RECONCILED になるまで、`K_world` の具体値、`std::atomic` 型設計、production code 変更には進まない。

---

## 3. R2 — `K_world` と `M_world` を厳密に分離する

### 3.1 D101-7 の問題点

D101-7 では:

> `S1 Reserved` は `M_world` の live に含む

としていたが、これはそのままでは成立しない。`S1 Reserved` はまだ `RuntimeWorld` が存在しない状態（reservation のみ、build 前）だからである。

### 3.2 修正: `B_world` vs `M_world`

```text
M_world(t) = 実際に存在する RuntimeWorld 数
           = Transferred + Published + Retiring + Quarantined + Terminal
           = S2 + S3 + S4 + S5 + S6

B_world(t) = Lifetime Budget を占有している reservation/world 数
           = Reserved + Transferred + Published + Retiring + Quarantined + Terminal
           = S1 + S2 + S3 + S4 + S5 + S6

Invariant: M_world(t) ≤ B_world(t) ≤ K_world

where K_world := admissible maximum of B_world（D101-8 で形式的条件を導出、具体値は D101-9 以降）
      K_world は B_world の budget capacity として定義する。M_world の有限性は containment（M_world ≤ B_world ≤ K_world）により導出する
```

| 項目 | `M_world` | `B_world` |
| --- | --- | --- |
| 定義 | 実際に存在する `RuntimeWorld` 数 | Lifetime Budget を占有している reservation/world 数 |
| 含む状態 | `S2 + S3 + S4 + S5 + S6` | `S1 + S2 + S3 + S4 + S5 + S6` |
| `S1 Reserved` | 含まない（まだ world が存在しない） | 含む（reservation が Budget を消費する） |
| 有限上界 | `B_world ≤ K_world（定義）` → `M_world ≤ K_world（導出）` | `M_world ≤ B_world ≤ K_world` |
| 関係 | `M_world ≤ B_world` | `B_world` は `M_world` を上界する |
| Conservation | `M_world` は world の存在数を数える（`B_world` を上界とする） | `B_world` は budget の占有数を数える（`K_world` を上界とする） |

### 3.3 なぜ分離が重要か

- `K_world = B_world の最大 occupancy` として定義し、`M_world ≤ B_world （containment）` を経由して `M_world ≤ K_world` を導出する。Reserved を Budget に含める理由も完全に明確になる。
- `S1 Reserved` を `M_world` に含めれば、`M_world` は「存在しない world」まで数えることになり、counting の意味が不正確になる。
- `B_world` を導入することで、`K_world` は `B_world` の有限上界として正確に定義され、`M_world ≤ B_world ≤ K_world` により `M_world` の有限性も保証される。

### 3.4 D101-7 契約の修正

| D101-7 の記述 | 修正後（本監査） |
| --- | --- |
| `K_world_occupied = reserved + owned ≤ K_world` | **`B_world(t) = Reserved + S2+S3+S4+S5+S6` / `M_world(t) = S2+S3+S4+S5+S6` / `M_world ≤ B_world ≤ K_world`** |
| `S1 Reserved は M_world の live に含む` | **`S1 Reserved は B_world に含むが、M_world には含まない（まだ world が存在しない）** |
| `K_world ≤ 1+4096+1024+K+...` | **`B_world ≤ K_world` / `M_world ≤ B_world ≤ K_world`。K_world の分解式は B_world に対するものとして扱う** |

---

## 4. R3 — `A_max` の semantic unit を修正する

### 4.1 D101-7 の問題点

D101-7 では:

```text
A_max = ISRWorldRetirementTelemetry の sampling interval に許される acquire 数の上限
acquire = RuntimeWorldAuthority::publish 成功
enforcement point = S0 → S1 reserve
```

このままだと `count するイベント = publish success` / `enforce するイベント = reservation` になり、reservation-first contract と一致しない。

### 4.2 現行コードの publish 構造（`ConvoPeq.md` 一次資料）

```cpp
// RuntimeWorldAuthority::publish() に集約
// PublishExecutor は OwnerChannel から ownership を取得して authority.publish(std::move(owner), ...) を呼ぶ
// 内部: register → PendingPublishRegistry → OwnerChannel transfer → ISR Intent enqueue → PublishExecutor::executePublish

// ConvoPeq.md L47171 — publish path
OwnerChannel::enqueue(key, world) → PendingPublishRegistry::registerPublish(seqId, world)
→ ISR Intent enqueue → CoordinatorLoop::ProcessIntent → PublishExecutor::executePublish
→ authority.ownerChannel().take(key) → authority.publish(std::move(owner), ...)

// RuntimeWorldAuthority::publish() — 唯一の publish gateway（X4-B）
// WriteAccess::publishAndSwap() で RuntimeStore::current を交換し、旧 world を retire へ
```

### 4.3 判定: `A_max = admission / reservation rate`

```text
A_max(interval) = interval 内に新規 RuntimeWorld reservation を許可できる最大数

Invariant: reservation_accepted(interval) ≤ A_max < ∞

where:
  interval = Lifetime Budget Authority の admission interval（telemetry の sampling interval とは独立に定義可能）
  reservation_accepted = S0→S1 reserve の成功数（admission された reservation 数）
```

| D101-7 の記述 | 修正後（本監査） |
| --- | --- |
| `A_max(interval) = interval 内の publish 成功数` | **`A_max(interval) = interval 内の RuntimeWorld reservation 許可数`** |
| `acquire = RuntimeWorldAuthority::publish 成功` | **`acquire = S0→S1 reserve の成功（admission された reservation）`。`A_count` と `B_token` は分離されている（§4.5）** |
| `enforcement = S0→S1 reserve の前または同時` | **明確に `S0→S1 reserve` の gate で enforcement。D101-8 contract 上の enforcement point は S0→S1 reservation admission。現行コードにはこの budget authority は未実装であり、Step 3 では実装前提条件として形式化する** |

### 4.4 `build failure → reservation rollback → rate token を戻すか？`

| ケース | 現行の期待 | D101-7 の未確定点 | 本監査の確定 |
| --- | --- | --- | --- |
| `reserve accepted → A_max counter +1` | reservation で rate を消費する | 未確定 | **採用**: reserve 成功時に `A_max` の rate token を消費する |
| `build failure → reservation rollback` | reservation を Budget Pool に返却（S1→S0）。`B_world` の Budget reservation token は返却する | Rate token を戻すか未確定 | **戻さない**: `A_max` の `A_count` は admission events の rate upper bound であり、build failure で戻さない。**返却するのは `B_world` reservation token だけ**。`A_count` と `B_world` reservation token は分離されている |
| 考慮点 | `A_max` は正しく定義すれば、reserve と build fail の繰り返しで無限に `reservation_accepted(interval) ≤ A_max` を回避できる。`B_token` と分離しなければ詰欺が生じる | — | `A_max` は `A_count(interval)` の rate upper bound であり、受理された S0→S1 admission 数を数え、build failure で戻さない。`B_world` reservation token のみが戻る。これにより、同じ interval 内での無限 reserve→build fail の繰り返しによる semantic contradiction を取り除く |

### 4.5 A. Admission rate vs B. Concurrent budget reservation の分離

```text
A. Admission rate
   A_count(interval) = S0→S1 の reserve 試行が accepted された回数
   Invariant: A_count(interval) ≤ A_max
   build failure → B_world reservation は rollback しなければ A_count は戻さない
   → rate-limit と outstanding capacity を混同しない

B. Concurrent budget reservation
   B_world = reserve → build → transfer → publish の outstanding lifetime を制限
   build failure → token release で正しい
   → A_max token ≠ B_world reservation token であることを明確に分離する
```

### 4.6 なぜ前者（admission / reservation rate）が正しいか

- Reservation-first contract の ordering は `reserve → build → transfer → publish` である。`A_max` が reservation の rate であれば、ordering と一致する。
- `A_max` が `publish success` の rate であれば、reservation と publish の間に時間差があり、rate の enforcement が遅延する。Reservation 時に rate を制御することで、build 前に admission を制御できる。

---

## 5. R4 — `P_max` の enforcement event を確定する

### 5.1 D101-7 の問題点

D101-7 では:

```text
P_max = window 内の publish 成功数
enforcement = S0→S1 の reserve gate で制御
```

これも event mismatch である。

```text
Reservation admission
    ↓
RuntimeWorld build
    ↓
OwnerChannel transfer
    ↓
Publish intent
    ↓
publishAndSwap
```

`P_max` が何を制限するかが明確でない。

### 5.2 判定: `P_max = publish admission に対する sustained rate limit`

> **重要: P_max の位置は ownership invariant に直結する。`S2 → take() → local owner → P_max gate → reject → rollback` とすると、再格納の atomicity / failure / rollback が新契約になる。最も証明しやすいのは `S2 / OwnerChannel → P_max gate → accept → take() → publish` である。reject 時は S2 のまま残るため rollback proof が不要になる。**

```text
P_max(window) = window 内に publish admission（RuntimeWorld publish）を許可できる最大数

Invariant: publish_admitted(window) ≤ P_max < ∞

where:
  window = 固定時間窓（例: 1秒）。H_max の数倍として定義することも可能
  publish_admitted = RuntimeWorldAuthority::publish 成功（publishAndSwap 成功）の数
                     ただし「reservation を取得したが publish されなかった world」は含まない
```

| D101-7 の記述 | 修正後（本監査） |
| --- | --- |
| `P_max(window) = window 内の publish 成功数` | **維持するが、enforcement point を明確に分離する。`P_max` は `S2→S3`（publish）の sustained rate として定義し、`A_max` は `S0→S1`（reservation）の burst rate として定義する** |
| `enforcement = S0→S1 reserve gate` | **`P_max` の enforcement は `S2` 直前の publish admission（`S2→S3` 直前）の window gate で行う。`A_max` が `S0→S1`（reservation）の interval gate であるのに対し、`P_max` は異なる粒度（window 単位）で publish 直前に admission を制御する。accounting は publish success（`S2→S3` 成功）で行う。enforcement event と accounting event を分離する。`P_max` enforcement mechanism は D101-8 が新設する contract であることを明記する** |
| Build 失敗で reservation 取得したが publish されなかった world | **`P_max` の publish count として誤計上しない。`P_max` は `publish success` の rate であり、reservation 取得のみでは count しない** |

### 5.3 enforcement と accounting の分離

> **原則: ownership transfer を不可逆にする operation より前に P_max admission を通過させることを要求する。現行 execution path ではその enforcement point は未実装であり、Step 4 で具体化する。**

```text
S2 (Transferred, OwnerChannel にある)
  ↓
publication admission check
  ↓
P_max window capacity available?
  ↓ yes
publish()  → publishAndSwap
  ↓ success
S3 (Published)
  ↓
P_count++  (accounting event = publish success)

※ enforcement event = publish 直前の admission
   accounting event = publish success
   両者を同一 event と表現しない。成功した後に count しても制限にならないため、admission 時点で check する。
```

### 5.4 P_max rejection 時の S2 transition（Open Item O1）

```text
S2 (Transferred, OwnerChannel にある)
  ↓
P_max gate
  ↓
REJECT 時:
  S2 → S2（retry / deferred）が原則。ownership remains authoritative、world remains accounted in B_world。
  S2 → S7→S0（discard/release）は別の deferred state として明確に分離する必要がある。
  rollback ではない。ownership が authoritative に残るか、rollback するかを明文化する必要がある。
```

> **Open Item O1**: `S2 → P_max admission reject → ?` の遷移を Step 4 で明確に定義する。現行コードの publish path は OwnerChannel から ownership を `take()` してから publish gateway に渡す構造であるため、この時点での reject が B_world の accounting に与える影響を明文化する必要がある。これを決めないまま `P_max` の conservation proof を閉じることはできない。

### 5.5 3つの異なる conservation/control quantity

```text
A_max → acquisition/reservation rate（S0→S1、interval 単位、burst 制御）
P_max → publication rate（S2→S3、window 単位、sustained 制御）
K_world → outstanding lifetime occupancy（S1+S2+...+S6、有限上界）
```

| Quantity | semantic object | 増加 event | 減少 event | Authority | finite proof | 現行の状態 | D101-8 での扱い |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `B_world` | reservation/world budget | `reserve` (S0→S1) | `rollback`/`release` (S1→S0 / S7→S0) | Lifetime Budget Authority | D101-8 Steps 1-2（形式的条件） | MISSING（新設） | R2 の分離で定義 |
| `M_world` | actual RuntimeWorld | `transfer/build` (S1→S2) | `destruction` (S4/S5/S6→S7) | ownership authorities | D101-8 Steps 1-2（`M_world ≤ B_world ≤ K_world`） | MISSING（新設） | R2 の分離で定義 |
| `A_count` | admission rate | `reservation accepted` (S0→S1) | `interval expiry` | Lifetime Budget Authority *(D101-8 新契約)* | D101-8 Step 3（`A_max < ∞`） | MISSING | R3 で新契約 |
| `P_count` | publication rate | `publish success` (S2→S3) | `window expiry` | Lifetime Budget Authority *(D101-8 新契約、enforcement は S2 直前 admission)* | D101-8 Step 4（O1: reject 時の ownership は Step 4 で定義） | MISSING | R4 で新契約として修正 |
| `M_retire` | retired worlds *(DeletionEntry ≠ RuntimeWorld count — O4)* | `publish eviction` (S3→S4) | `reclaim` (S4→S5/S7) | ISRRetireRouter | D101-8/9（`M_retire ≤ 4096／だが DeletionEntry 4096 を M_retire_world に流用できるかは Step 6 の proof obligation） | SATISFIED（4096）だが O4 として検証が必要 | 既存容量を流用しつつ O4 の cardinality を検証 |
| `M_quarantine` | quarantined worlds *(DeletionEntry ≠ RuntimeWorld count — O4/O6)* | `Q/E admission` (S4→S5) | `drain` (S5→S6/S7) | Quarantine authority | D101-9（`M_quarantine ≤ 1024` は entry occupancy bound、RuntimeWorld としては未証明 — O6） | SATISFIED/PARTIAL（512+512は entry としては確定、RuntimeWorld としては O6） | entry は既存容量、RuntimeWorld は O6 で証明 |
| `M_terminal` | terminal worlds | `terminal store` (S5→S6) *(bounded 化後は reservation-first により slot available を証明 — O5)* | `drainAll/drain` (S6→S7) | Terminal authority *(retirement-path, ≠ ShutdownReclaimAuthority)* | D101-9（`M_terminal ≤ K_terminal < ∞`, bounded 化後） | MISSING（現行 growable） | R1 で Option A（D101-9 で bounded 化） |
| `M_reader` | reader-held worlds | `reader acquire` | `reader exit` | Reader/epoch | H_max design（D101-8 Step 5） | MISSING（新設） | H_max design で導出 |
| `G_pending` | PendingPublishRegistry metadata gap *(auxiliary, ≠ M_world/B_world)* | `registerPublish` | `unregister` | PendingPublishRegistry | D101-8 Step 1-2（`G_pending ≤ 64` — bounded gap） | SATISFIED（64） | 評当ではない/、Registry does not contribute to B_world を証明するための補助 quantity |

- `A_max` と `P_max` は異なる粒度で流入側を制御し、`K_world` は outstanding の有限上界である。3者は異なる conservation/control quantity として分離される。

---

## 6. R5 — Shutdown の `S2 → S0` を「誰が所有権を持つか」まで具体化する

### 6.1 D101-7 の記述

D101-7 では:

```text
S2 stale/shutdown → S0
registry entry drop + reservation release
```

しかし現コードでは ownership は `OwnerChannel` に移された後である:

```cpp
// ConvoPeq.md の publish path
OwnerChannel::enqueue(key, world)  // ownerChannel が ownership 保持
    ↓
executePublish
    ↓
ownerChannel().take(key)  // ownership 取得
    ↓
RuntimeWorldAuthority::publish(std::move(owner))
```

したがって `registry unregister` だけでは lifetime transition にはならない。

### 6.2 `S2` の具体的分離

> **原則: Lifecycle state ≠ physical storage location**
> `S2 = Transferred` は lifecycle state。OwnerChannel / PendingPublishRegistry は physical storage location。同一 publish gap の期間、両者は同時に存在し得る。`S2 を OwnerChannel の同義語にしてはならない。
>
> ```text
> Lifecycle state: S2 Transferred
> Physical representation (within S2):
>     OwnerChannel: owning representation = 1
>     PendingPublishRegistry: non-owning metadata/gap representation = 0 or 1
> ```
>
> **Open Item O2**: `S2` lifecycle state と physical residence を分離し、Registry は non-owning metadata で手 OwnerChannel は owning transport であること、同一 world に同時存在可能であることを明確に分離する。

| 状態（S2 内の storage location） | 所有権 | 所在 | Shutdown 時の扱い |
| --- | --- | --- | --- |
| `S2 / OwnerChannel resident` | **owning** — *(R5: 同一 gap で Registry と同時存在可能)* `aligned_unique_ptr<RuntimeState>` を保持 | `RuntimeWorldAuthority::OwnerChannel`（`std::unordered_map<OwnerChannelKey, OwnerPtr>` 相当） | `OwnerChannel::drainAllNonRt(reclaim)` で残留 owner を retire chain に戻す。`ConvoPeq.md L59800` の `drainAllNonRt` が該当 |
| `S2 / PendingPublishRegistry resident` | **non-owning** — `const void*` の metadata/gap のみ。`Registry entry = ownership 0 = metadata/gap occupancy 1` として、Budget proof では必要なら別 quantity にする | `RuntimeWorldAuthority::PendingPublishRegistry`（`kPendingPublishCapacity=64`、seqId keyed） | `registry.unregister(seqId)` で metadata をクリアするが、lifetime transition ではない。所有権は OwnerChannel 側で処理する必要がある |
| `S2 / local owner` | **owning** — `aligned_unique_ptr` をローカル変数が保持 | Producer の stack（build 直後、enqueue 前） | ScopeExit で rollback または OwnerChannel::enqueue で移転する |

### 6.3 契約の固定

```text
registry = non-owning metadata/gap（seqId → const void* の lookup table、64 slots）
           所有権を持たない。gap の bounded 性（64）のみを保証する
           Shutdown 時は unregister で metadata をクリアするが、lifetime transition ではない

OwnerChannel = owning transport（OwnerChannelKey → aligned_unique_ptr<RuntimeState>）
               所有権を持つ。publish 前の唯一の owning transport である
               Shutdown 時は drainAllNonRt(reclaim) で残留 owner を retire chain に戻す
               reclaim は ISRRetireRouter::enqueueWithRetry 相当で、ownership を失わない
```

| D101-7 の記述 | 修正後（本監査） |
| --- | --- |
| `S2 stale/shutdown → S0: registry entry drop + reservation release` | **`S2 / OwnerChannel resident → drainAllNonRt(reclaim) → retire chain（S4 相当）→ drain → S7→S0` / `S2 / Registry resident → unregister（metadata クリア）+ OwnerChannel 側の drainAllNonRt で対応`** |
| `registry = gap` の扱い | **registry は non-owning metadata/gap、OwnerChannel は owning transport として契約を固定する** |

---

## 7. R6 — Shutdown contract を現コードに合わせて再検証する

### 7.1 D101-7 の想定

D101-7 では:

```text
Admission close
→ gap drain
→ retire drain
→ terminal drainAll
→ budget release
```

### 7.2 現行コードの shutdown ordering（`ConvoPeq.md` 一次資料）

```text
// ConvoPeq.md L30780 — releaseResources() / ~AudioEngine の shutdown sequence
1. stopRebuildThread()
2. shutdownWorkerThread()
3. setShutdownPhase(ForceEpochAdvance) → m_retireRouter->publishEpoch()
4. Graceful Drain Phase (最大 5秒):
     while (pendingRetireCount != 0 || activeReaderCount != 0) {
         publishEpoch(); tryReclaim(); sleep(10ms);
         timeout → forcing drain
     }
5. worldAuthority_.requestShutdownClearNonRt()
   clearedWorld = worldAuthority_.clearPublishedRuntimeSnapshotsNonRt()
   clearBridge.retirePublishedRuntimeWorldNonRt(clearedWorld, true)
6. drainDeferredRetireQueues(true)
7. drainPendingRetireIntentsForShutdown()  // OverflowRing の residual RetireIntents
8. if (activeReaderCount == 0) m_retireRouter->drainAll()
   else { m_epochDomain.drainAll(); m_retireRouter->drainAllQuarantineStore(); }
   // ★ 15-P-5 FIX: stuck-reader fallback は Q+E+T の drainAllQuarantineStore も強制実行
9. runtimePublicationBridge_.markShutdownComplete()
10. ... 既存の解放処理 ...

// OwnerChannel の shutdown drain（ConvoPeq.md L59800）
OwnerChannel::drainAllNonRt(reclaim):
    for each slot: if (owner != nullptr) { owner = nullptr; reclaim(raw); }

// releaseResources の timeout 時の追加処理（L30840）
releaseResources: timeout 時に drainDeferredRetireQueues(true) → tryReclaim() →
                  coordinator shutdown 後に OwnerChannel::drainAllNonRt()
```

### 7.3 照合結果

| D101-7 の想定 | 現行コード | 照合 |
| --- | --- | --- |
| `Admission close` | ✅ `requestShutdownClearNonRt()` + Path B admission gate で拒否 | 一致 |
| `gap drain`（PendingPublishRegistry） | ✅ `drainAllNonRt(reclaim)` が該当。だが D101-7 は registry のみを想定していた | **R5 の修正で OwnerChannel と Registry を区別した** |
| `retire drain`（DeferredDeletionQueue） | ✅ `drainDeferredRetireQueues(true)` + `tryReclaim()` が該当 | 一致 |
| `terminal drainAll` | ✅ `drainAll()` / `drainAllQuarantineStore()` が該当。だが D101-7 は単純に `drainAll()` のみを想定していた | **現行は stuck-reader fallback（`drainAllQuarantineStore`）も含む複雑な ordering である** |
| `budget release` | ✅ `K_world → 0` の収束。だが現行は Budget Pool が存在しないため release の概念なし | D101-8 Steps 1-8 で新設する Lifetime Budget の release として定義する |

### 7.4 修正後の Shutdown contract

> **R6 分離**: R6a 現行 shutdown ordering PASS / R6b Budget release semantics = DESIGN CONTRACT（現行コードでは未実装） / R6c finite-time convergence = Step 8 で証明

```text
Shutdown contract（現行コードに合わせた再検証）:

  Admission close
    ↓
  producer stop（stopRebuildThread / shutdownWorkerThread）
    ↓
  producer join（Graceful Drain Phase、最大 5秒、pendingRetireCount==0 && activeReaderCount==0 まで待機）
    ↓
  pending publish gap resolution
    ├─ PendingPublishRegistry: unregister（non-owning metadata クリア）
    └─ OwnerChannel residual drain: drainAllNonRt(reclaim) → retire chain（owning transport の残留を回収）
    ↓
  retire chain drain
    ├─ drainDeferredRetireQueues(true)（DeferredDeletionQueue の drain）
    ├─ drainPendingRetireIntentsForShutdown()（OverflowRing の residual）
    └─ tryReclaim()（epoch 進行 + reclaim）
    ↓
  terminal/shutdown reclaim
    ├─ activeReaderCount==0 → m_retireRouter->drainAll()（全 store 強制解放）
    └─ stuck reader → m_epochDomain.drainAll() + m_retireRouter->drainAllQuarantineStore()
                      （Q+E+T の epoch-agnostic force-drain、Audio Thread 停止後のみ）
    ↓
  Verify Empty（pendingRetireCount==0 && OwnerChannel empty && Registry empty）
    ↓
  Budget release（K_world → 0、全 slots が S0 Available に返却）
    ↓
  Shutdown complete（markShutdownComplete）
```

- D101-7 の単純な `Admission close → gap drain → retire drain → terminal drainAll → budget release` は、現行の複雑な shutdown ordering に合わせて上記の詳細 ordering に修正された。
- 特に `OwnerChannel residual drain` と `stuck-reader fallback` は D101-7 で考慮されていなかったが、現行コードに存在するため本監査で追加された。

---

## 8. D101-8 で最初に作るべき proof table

### 8.1 Proof Table（semantic fields 確定、finite proof は未完成）

> **Quantity の種類**:
> - **Layer A — Lifetime semantic quantities**: `B_world` / `M_world` / `M_world_S2`..`M_world_S6` *(cardinality = distinct RuntimeWorld identities)* — Step 1 で formalize
> - **Layer B — physical/container quantities**: `G_pending` / `D_entry` / `Q_entry` / `E_entry` / `T_entry` / `OwnerChannel` / `Registry` *(entry occupancy ≠ RuntimeWorld occupancy — O4/O6)*
> - **Layer C — rate-control quantities**: `A_count` / `P_count` *(interval/window sliding-window accounting)*
> 三層を混ぜないことが D101-8 の証明で重要。

| Quantity | semantic object | 増加 event | 減少 event | Authority | finite proof | 現行の状態 | D101-8 での扱い |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `B_world` | reservation/world budget *(budget-unit cardinality, ≠ raw queue occupancy — O7)* | `reserve` (S0→S1) | `rollback`/`release` (S1→S0 / S7→S0) | Lifetime Budget Authority | D101-8 Steps 1-2（形式的条件、Step 1 で `B_world(t)=\|{W: lifecycle(W)∈S1..S6}\|` として formalize） | MISSING（新設） | R2 の分離で定義 |
| `M_world` | actual RuntimeWorld *(lifecycle occupancy, ≠ container resident count — O4/O6, Layer A)* | `transfer/build` (S1→S2) | `destruction` (S4/S5/S6→S7) | ownership authorities | D101-8 Steps 1-2（`M_world ≤ B_world ≤ K_world`） / Step 1 で `M_world(t)=\|{W: lifecycle(W)∈S2..S6}\|` として formalize | MISSING（新設、現行は `M_world = ∞`） | R2 の分離で定義、container ≠ RuntimeWorld cardinality に注意 |
| `A_count` | admission rate | `reservation accepted` (S0→S1) | `interval expiry` | Lifetime Budget Authority | D101-8 Step 3（`A_max < ∞` の形式的条件） | MISSING | R3 で `A_max = reservation rate` として修正 |
| `P_count` | publication rate | `publish success` (S2→S3) | `window expiry` | Lifetime Budget Authority *(D101-8 新契約、enforcement は S2 直前 admission)* | D101-8 Step 4（`P_max < ∞` の形式的条件、O1: reject 時の ownership は Step 4 で定義） | MISSING | R4 で `P_max = publication rate` として修正 |
| `M_retire` | retired worlds *(DeletionEntry count ≠ RuntimeWorld countに注意 — O4)* | `publish eviction` (S3→S4) | `reclaim` (S4→S5/S7) | ISRRetireRouter | D101-8/9（`M_retire ≤ 4096）、boundedだが DeletionEntry 4096 を M_retire_world に流用できるかは Step 6 の proof obligation） | SATISFIED（4096）だが O4 として検証が必要 | 既存容量を流用しつつ O4 の cardinality を検証 |
| `M_quarantine` | quarantined worlds | `Q/E admission` (S4→S5) | `drain` (S5→S6/S7) | Quarantine authority | D101-9（`M_quarantine ≤ 1024`、bounded） | SATISFIED/PARTIAL（512+512） | 既存容量を流用 |
| `M_terminal` | terminal worlds | `terminal store` (S5→S6) | `drainAll/drain` (S6→S7) | Terminal authority | D101-9（`M_terminal ≤ K_terminal < ∞`、bounded 化後） | MISSING（現行 growable） | R1 で Option A（D101-9 で bounded 化） |
| `M_reader` | reader-held worlds | `reader acquire` | `reader exit` | Reader/epoch | `H_max` design（D101-8 Step 5） | MISSING（新設） | H_max design で導出 |

### 8.2 Retirement topology

現行 Router は `drainAll() { provider_->drainAll(); drainAllQuarantineStore(); }` の構造である。`drainAllQuarantineStore()` は RetireQuarantine / EmergencyQuarantine / TerminalReclaimAuthority をまとめて強制 drain する。したがって証明では Terminal だけを独立した final stage として扱うより、`Retire chain { D, Q, EmergencyQ, Terminal }` という retirement topology を明示した方が証明しやすい。

### 8.3 本 Table が埋まらない限り `K_world` の数値導出には進まない

- 本 Table の全行で `semantic object` / `増加 event` / `減少 event` / `Authority` / `finite proof` が確定し、R1〜R6 の再照合が RECONCILED_WITH_OPEN_CONTRACT_ITEMS になるまで、`K_world` の具体値導出（D101-8 Step 6）には進まない。
- 特に `M_terminal` は現行 growable であるため、`K_world` の有限証明は `K_terminal < ∞` を仮定した形式的条件として扱い、D101-9 で bounded 化の具体設計後に証明を閉鎖する。

---

## 9. D101-8 本題への順序

### 9.1 Step 0 の reconciliation が終わった後の順序

```text
D101-8 Step 0  Contract reconciliation  ◀ 本監査（RECONCILED_WITH_OPEN_CONTRACT_ITEMS）
        │
        ▼
D101-8 Step 1  B_world / M_world formal separation
               B_world(t) ≤ K_world / M_world(t) ≤ B_world(t) の形式的条件
        │
        ▼
D101-8 Step 2  Reservation token semantics
               reservation token の表現 / double-release 防止 / leak 防止 / transfer 前後の invariant
        │
        ▼
D101-8 Step 3  A_max derivation
               A_max < ∞ の形式的条件（admission / reservation rate として）
               interval の定義 / enforcement point / reject/rollback 条件
        │
        ▼
D101-8 Step 4  P_max derivation
               P_max < ∞ の形式的条件（publication rate として）
               window の定義 / burst と sustained の分離
        │
        ▼
D101-8 Step 5  H_max / G_contract derivation
               H_max < ∞ / G_contract < ∞ の形式的条件
               reader hold bound / sampler gap contract の導出
        │
        ▼
D101-8 Step 6  K_world derivation
               K_world の形式的条件（B_world / M_world 分離を前提に）
               K_world < ∞ の証明（K_terminal を形式的パラメータとして扱う）
        │
        ▼
D101-8 Step 7  Conservation proof
               reserved + owned + ... = K_world の形式的証明
               全 failure path の rollback/release を網羅
        │
        ▼
D101-8 Step 8  Failure / shutdown proof
               Failure matrix / shutdown contract の有限完了性
               K_world → 0 を有限時間で証明できる条件
        │
        ▼
D101-9         TerminalReclaim bounded design
               std::vector → std::array<K_terminal> + reservation-first の具体設計
               S5→S6 の slot available 証明
        │
        ▼
D101-8 へ戻って K_world proof を完成
        K_terminal の具体設計を前提に K_world の有限証明を閉鎖（I01 / I10 / K_world finite proof）
        │
        ▼
Phase I GO / NO-GO
```

### 9.2 現時点の判定

```text
D101-8 Step 0: RECONCILED_WITH_OPEN_CONTRACT_ITEMS

  Closed:
    R1  Terminal:    current = growable / D101-8 = K_terminal < ∞ as formal assumption / D101-9 = bounded implementation
    R2  World counting: M_world = S2+S3+S4+S5+S6 / B_world = S1+S2+S3+S4+S5+S6 / M_world ≤ B_world ≤ K_world
    R3  A_max:       reservation admission rate / A_count(interval) ≤ A_max / A_count != B_world token
    R4  P_max:       enforcement = publish直前 admission / accounting = publish success。semantics は確定、O1 は Step 4 で定義
    R5a Ownership:   Registry = non-owning metadata / OwnerChannel/local owner = owning representation
    R5b Counting:    Registry metadata does not add M_world/B_world
    R6a Shutdown:    current ordering has been reconciled

  Open:
    O1  P_max reject ownership semantics — gate before take()? or gate after take() + rollback?
    O2  Registry/OwnerChannel coexistence invariant — exactly-one-owner invariantを全pathで閉じる
    O3  Shutdown budget release — 現行コードでは未実装 / D101-8 contractとして有限収束条件を定義
    O4  Retirement entry → RuntimeWorld cardinality — DeletionEntry 4096 を M_retire_world に流用できるか証明（*O4は Step 6 の局所問題ではなく Step 1 の M_world 定義そのものに影響*）
    O6  Quarantine/Terminal resident count と RuntimeWorld count の semantic separation
    O7  B_world を state occupancy の単純加算として扱わず、budget-unit cardinality として定義
    O5  Terminal bounded transition — K_terminal < ∞ は仮定のみ / S5→S6 acceptance proof はD101-9へ延期

  → Step 1 に進むこと自体は可能。ただし Step 0 verdict 自体は RECONCILED_WITH_OPEN_CONTRACT_ITEMS のままとする。
     Step 0 が PASS したため Steps 1-8 へ進むのではなく、reconciliation boundary は確定したため Step 1 に進む。
```

---

## 10. Verdict

### 判定: `RECONCILED_WITH_OPEN_CONTRACT_ITEMS`

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| `RECONCILED` | D101-7 の契約と現行コードの不整合を完全に解消し、semantic table を確定し、D101-8 本題へ進む準備が整った | **該当せず** — finite proof は未完成、O1〜O7 が残るため |
| `CONTRADICTION_REMAINS` | 不整合が解消されず、D101-7 の契約が現行コードと矛盾したままである | **該当せず** — 6点の不整合を全て解消した |
| `RECONCILED_WITH_OPEN_CONTRACT_ITEMS` | 6点の contradiction boundary を確定し、Step 1 以降で解くべき contract obligations（O1〜O7）を明示した。ただし、R5/R6 の一部は現行実装事実と契約上の証明条件をまだ分離し切っていない。証明は未完成、Step 1 に進むこと自体は可能 | **◯ 該当（本監査の結論）** |
| `DESIGN_REQUIRED` | 不整合は特定できたが、修正には新設計が必要であり、Step 0 の範囲を超える | **該当せず** — Step 0 の範囲内で6点の修正を完了した。D101-8 本題での詳細設計は別途必要だが、Step 0 としての再照合は完了した |

- R3: `build failure → A_max token return` を撤回。`A_count` と `B_world` reservation token を分離。返却するのは Budget reservation token だけ
- R4: `enforcement event = publish success` ではなく `enforcement = publish直前の admission (S2→S3 直前) / accounting = publish success` と分離
- R2: `K_world = M_world の上界` ではなく `K_world = B_world の maximum budget occupancy (定義) → M_world ≤ B_world → M_world ≤ K_world (導出)` とする
- R5: `S2` lifecycle state と `OwnerChannel / Registry` physical storage location を `Lifecycle state ≠ physical storage location` として分離
- R6: `Budget release / K_world → 0` は現行コードの実装ではなく D101-8 が新設する contract であることを明記
- R1: `K_terminal < ∞ を未証明の前提条件としてパラメータ化` は本監査で修正済み。上計5点は契約文言として修正すれば Step 1 へ進める

### なぜ `RECONCILED_WITH_OPEN_CONTRACT_ITEMS` か

- **R1**: 現行 `TerminalReclaimAuthority` が `std::vector` growable であることを確認し、Option A（D101-9 で bounded 化、D101-8 では `K_terminal` を形式的パラメータとして仮定）の依存関係を明示した。D101-7 の `S6 Terminal bounded` を現行コードと矛盾しない「形式的仮定」として修正した。
- **R2**: `S1 Reserved` が `M_world` の live に含まれないことを確認し、`M_world(t) ≤ B_world(t) ≤ K_world` として分離した。`K_world = M_world の有限上界` と `Reserved も K_world を消費する` の衝突を解消した。
- **R3**: `A_max` の count 対象を `publish success` から `reservation accepted` に修正し、`A_count` と `B_world` reservation token を分離し、`build failure では A_count は戻さない（返却するのは B_world reservation token だけ）` として修正した。`A_count` と `B_token` を混同すると同じ interval 内での無限 reserve→build fail の繰り返しによる semantic contradiction が生じるため、明確に分離した。
- **R4**: `P_max` の enforcement event を `S0→S1 reserve gate` から `S2 直前の publish admission（S2→S3 直前）の window gate` に分離し、`enforcement event = publish 直前の admission / accounting event = publish success` として分離した。成功した後に count しても制限にならないため、admission 時点で check する。`A_max`/`P_max`/`K_world` の3つの異なる quantity として明示した。
- **R5**: `S2` を `OwnerChannel resident`（owning）と `PendingPublishRegistry resident`（non-owning metadata/gap occupancy 1）に区別し、`Lifecycle state ≠ physical storage location` を明確に分離し、`registry = non-owning metadata/gap` / `OwnerChannel = owning transport` として契約を固定した。
- **R6**: D101-7 の単純な shutdown 想定を、現行の複雑な ordering（`producer stop → join → gap resolution → OwnerChannel drain → retire chain → terminal/shutdown reclaim → Verify Empty）に合わせて再検証し、`Budget release / K_world → 0` は現行コードの実装ではなく D101-8 が新設する contract であることを明記した。
- **Proof table**: 8 quantities の proof table を確定し、`K_world` の数値導出前に全ての `semantic object` / `増加 event` / `減少 event` / `Authority` / `finite proof` を閉じた。

### 全体判定

```text
D101-7  CONTRACT_DEFINED
   │
   ▼
D101-8 Step 0  RECONCILED_WITH_OPEN_CONTRACT_ITEMS  ◀ 本監査
   │  6点の不整合を解消（R1〜R6）
   │  Proof table を確定（8 quantities）
   │  K_terminal を形式的パラメータとして仮定（D101-9 で bounded 化）
   │  B_world / M_world 分離を確定
   │  A_max / P_max の event semantics を確定
   │
   ▼
D101-8 Steps 1-8 — 本題（数値導出前の形式的条件）
   │
   ├── Step 1: B_world / M_world formal separation
   ├── Step 2: Reservation token semantics
   ├── Step 3: A_max derivation
   ├── Step 4: P_max derivation
   ├── Step 5: H_max / G_contract derivation
   ├── Step 6: K_world derivation（K_terminal を形式的パラメータとして）
   ├── Step 7: Conservation proof
   └── Step 8: Failure / shutdown proof
   │
   ▼
D101-9  TerminalReclaim bounded design
   │
   ▼
D101-8 へ戻って K_world proof を完成
   │
   ▼
Phase I GO / NO-GO
```

- **本監査でも production code は変更しない**（指示どおり）。
- D101-8 Step 0 が `RECONCILED_WITH_OPEN_CONTRACT_ITEMS`（reconciliation boundary は確定したため Step 1 に進む）により、D101-8 Steps 1-8 の本題へ進む準備が整った。`RECONCILED` / PASS にはしない。
- `K_world` の具体値、`std::atomic` 型設計、production code 変更には進まない（Step 0 が PASS するまで — 本監査で PASS したため、Steps 1-8 で形式的条件の導出へ進む）。

---

---

## 補足: 全ツールによる棚卸し結果

| ツール | 結果 | 備考 |
| --- | --- | --- |
| `rg` (ripgrep 15.1) | `TerminalReclaimAuthority` 8 hits / `OwnerChannel` 12 hits / `DeletionEntry` 18 hits / `publishAndSwap` 6 hits | `ConvoPeq.md` ・ `src/audioengine` に横断的に構造を確認。`K_world`/`B_world` は現行コードに存在せず、D101-8 新契約であることを確認 |
| `ast-grep` 0.44.0 / `sg` | パターン検索可能を確認 | `fdfind` 10.3.0 ・ `fzf` と併せて使用可能。D101-8 本題で形態索弊れ必要に応じて活用 |
| `semble` | `TerminalReclaimAuthority` / `OwnerChannel+PendingPublishRegistry` / `shutdown ordering` を一括検索。publish path の `OwnerChannel::enqueue → registry.registerPublish` の同時存在を確認 | `semble search` で自然言語クエリから検索。`--max-snippet-lines 8` で本文据点を確認 |
| `cocoindex` (ccc 0.2.41) | `ccc status` は `Not in an initialized project` 。Uninitialized | `ccc init` が未実行のため、本段階では rg/semble で代替。D101-8 本題で `ccc init` 後に利用可能 |
| `graphify` 0.9.47 | `graphify --version` 確認。graphify-out 未存在のため、本段階では rg/semble で代替 | `graphify` は本段階ではインデックスなし。D101-8 本題で必要に応じて `graphify install` 後に利用 |
| `AiDex` (.aidex/index.db) | `index.db` + `index.db-shm` + `index.db-wal` を確認 | `index.json` はなし（DB 直接管理）。D101-8 本題で `/graphify` でインデックス更新後に利用可能 |
| `serena` (mcp) | `wsl` 経由で `rg` と併用 | AiDex – serena – semble – cocoindex – graphify の優先順位で利用 |

**剥卸し結論**: `rg` で `K_world`/`B_world`/`M_world` が現行コードに存在せず D101-8 新契約であることを強めて確認。`semble` で `OwnerChannel` / `PendingPublishRegistry` の同時存在、`publishAndSwap` の publish gateway としての唯一性を確認。`ast-grep`/`fdfind`/`fzf` は D101-8 本題で形態検索として活用可能。`cocoindex`/`graphify`/`AiDex` は未初期化のため、D101-8 本題で初期化後に活用可能。

---

## 補足: 形式的 ownership / lifecycle invariant（Step 1 で固定すべき）

> **Step 1 で固定すべき formal invariant**:
>
> ```text
> World identity W に対して
> OwnerCount(W,t) ∈ {0,1}
> RegistryCount(W,t) ∈ {0,1}
> RegistryCount が OwnerCount に寄与しない
> M_world(t) = |{W : lifecycle(W) ∈ S2..S6}|
> B_world(t) = |{W : lifecycle(W) ∈ S1..S6}|
> ```
>
> ※ 現行 OwnerChannel / PendingPublishRegistry の実装構造とも整合する。

---

## 付録: D101-8 Step 0 監査チェックリスト

- [x] D101-7 と最新の `ConvoPeq.md` を照合し、D101-7 の契約に D101-8 の冒頭で先に解消すべき不整合を特定（少なくとも4点 → 6点に拡大して照合）
- [x] R1: S6 Terminal の契約を修正（Option A/B を比較し、Option A を採用。D101-9 との依存関係を明示）
- [x] R2: `K_world` と `M_world` を厳密に分離（`M_world(t) = S2+S3+S4+S5+S6` / `B_world(t) = S1+S2+S3+S4+S5+S6` / `M_world ≤ B_world ≤ K_world`）
- [x] R3: `A_max` の semantic unit を修正（`A_max = admission / reservation rate` として `S0→S1 reserve` の gate で enforcement）
- [x] R4: `P_max` の enforcement event を確定（`P_max = publish admission に対する sustained rate limit` として `S2→S3` の window gate で enforcement、3つの quantity に分離）
- [x] R5: Shutdown の `S2 → S0` を「誰が所有権を持つか」まで具体化（`S2 / OwnerChannel resident` / `S2 / Registry resident` / `S2 / local owner` の区別、`registry = non-owning` / `OwnerChannel = owning` の固定）
- [x] R6: Shutdown contract を現コードに合わせて再検証（`Admission close → producer stop → join → gap resolution → OwnerChannel drain → retire chain → terminal/shutdown reclaim → Verify Empty → Budget release` の現行 ordering と照合）
- [x] Proof table（8 quantities の semantic object / 増加 event / 減少 event / Authority / finite proof）を確定
- [x] D101-8 本題（Steps 1-8）への順序を確定（`Step 0 → Steps 1-8 → D101-9 → K_world proof 完成 → Phase I GO/NO-GO`）
- [x] Production code 変更なし（Step 0 の再照合のみ）
- [x] `K_world` の具体値、`std::atomic` 型設計、production code 変更には進まないことを確認（Step 0 が RECONCILED になるまで）
