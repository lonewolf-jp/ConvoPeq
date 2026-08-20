# D101-2 — Finite-Bound Contract Repair Analysis

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-2 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **対象ソース** | `src/core/RuntimeStore.h`, `src/audioengine/RuntimeWorldAuthority.h`, `src/audioengine/ISRRetireRouter.h/.cpp`, `src/DeferredDeletionQueue.h`, `src/audioengine/RetireQuarantineStore.h`, `src/audioengine/RuntimeBuilder.h`, `src/audioengine/AudioEngine.h/.Commit.cpp/.Timer.cpp/.Transition.cpp`, `src/audioengine/ISRWorldRetirementTelemetry.h`, `doc/work88/I4_DESIGN_CONTRACT.md` |
| **前提** | D101-1.5 verdict: **NO_FINITE_BOUND** — `TerminalReclaimAuthority` が growable (`std::vector<Entry>`) であることを根拠に finite bound を否定。`M = 1+4096+512+512` は不成立 |
| **目的** | 「どうすれば M を作れるか」ではなく「どの architectural invariant を追加すれば M が証明可能になるか」を確定する。4案を比較し、queue capacity を M と読み替えない |
| **制約** | **コード変更なし・設計監査のみ**。I4 D14 の「capacity 枯渇を terminal-failure にして obligation を失わせない」「bounded にするため overflow 時に捨てる安易な解決は禁止」「reservation-first と backpressure 固定」に整合させる |
| **判定** | **FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE** — 現行 architecture のままでは証明不可。`bounded TerminalReclaim + ownership-preserving backpressure + outstanding-world cap + producer/lifetime bound` を導入すれば証明可能。`STRUCTURALLY_INCOMPATIBLE` には該当しない |

---

## 0. 位置づけと D101-1.5 からの継承

```text
D101-1.5
  │
  ▼
NO_FINITE_BOUND  (TerminalReclaim = std::vector / growable / store() always true)
  │
  ▼
D101-2  ← 本監査 (Finite-Bound Contract Repair Analysis)
  │
  ├── bounded TerminalReclaim
  ├── outstanding-world cap
  ├── producer/lifetime bound
  └── incompatibility analysis
  │
  ▼
D101-2 Verdict
  ├── REPAIRABLE
  ├── ARCHITECTURAL_CHANGE_REQUIRED  ◀ 本監査の結論
  └── STRUCTURALLY_INCOMPATIBLE
  │
  ▼
I4 Contract 更新 → Phase I GO/NO-GO 再判定
```text
D101-1.5 が確定した事実（再掲）:

```text
RuntimeStore              1        (current slot, AuAu = single current)
DeferredDeletionQueue     4096     (kQueueSize = 4096, Vyukov bounded MPMC)
RetireQuarantineStore     512      (kMaxQuarantinedEntries = 512, std::array)
EmergencyQuarantineStore  512      (同一型の別インスタンス → 512)
TerminalReclaimAuthority  ∞        (std::vector<Entry> entries_, growable, store() ALWAYS true)
─────────────────────────────────────────────
So-called "M = 1+4096+512+512 = 5121" は不成立
生存 world 数 = 1 + DDQ(≤4096) + Q(≤512) + EQ(≤512) + TerminalReclaim(≤∞)  →  上界なし
```text
D101-2 では、この確定事実を前提に「どの invariant を追加すれば `M < ∞` を証明可能になるか」を、**queue capacity を M と読み替えない**ことを厳守して分析する。

---

## 1. 方法

- D101-0 / D101-1 / D101-1.5 の evidence と `doc/work88/I4_DESIGN_CONTRACT.md` の D14 / reservation-first / backpressure 記述を突合
- `src/audioengine/ISRRetireRouter.h/.cpp` の `TerminalReclaimAuthority` 定義（`std::vector<Entry> entries_`、P-4 growable store、Non-RT only）、`RetireQuarantineStore.h` の `kMaxQuarantinedEntries = 512`、`DeferredDeletionQueue.h` の `kQueueSize = 4096`、`ISRWorldRetirementTelemetry.h` の `maxSamplingGapUs_` 追跡、`AudioEngine.Commit.cpp` の `commitRuntimePublication` 駆動点をコード監査
- I4 D14 の「capacity 枯渇を terminal-failure にして obligation を失わせない」方向と整合するかを確認

---

## 2. 4案比較（指示どおりの4案）

| 案 | 内容 | D101 の M 成立 | 本監査の判定 | 根拠 |
| --- | --- | --- | --- | --- |
| **A** | `TerminalReclaimAuthority` を bounded にする | 証明可能候補 | **単独では不成立** — bounded 化だけでは M 証明に不十分。exhaustion 時の ownership-preserving 処理が未定義なら UAF / ownership loss になる | 2.1 / 3.2 参照 |
| **B** | outstanding `RuntimeWorld` 数そのものに hard cap を設ける | 証明可能候補 | **必要条件だが単独では不十分** — cap の enforcement 手段（reservation-first admission control）がなければ、cap は宣言に留まる。producer が cap を迂回して publish できれば M は依然として証明不能 | 3.1 参照 |
| **C** | producer activity / publish rate に hard bound を設ける | 証明可能候補 | **必要条件だが単独では不十分** — `producer rate bound × reclaim latency` で M を導く方式は、reclaim latency 自体にも finite bound が必要。Reader が無制限に critical section を保持できるなら、producer rate 上限だけでは outstanding lifetime の有限性は証明できない | 3.3 参照 |
| **D** | 現状の growable TerminalReclaim を維持 | **UNPROVABLE 継続** | **唯一の現行一致案** — D101-1.5 の NO_FINITE_BOUND がそのまま継続。M 証明は再開不能 | D101-1.5 継承 |

**結論**: A / B / C はいずれも **単独では M を証明できない**。M 証明を再開するには **A + B + C の合成 + reclaim latency bound + ownership-preserving backpressure** が同時に必要。したがって D101-2 の最終判定は REPAIRABLE（既存に invariant 追加だけで足りる）ではなく、**ARCHITECTURAL_CHANGE_REQUIRED** である。

### なぜ queue capacity を M と読み替えてはならないか（再確認）

```text
M_claimed = 1 + 4096 + 512 + 512 = 5121   ←  誤り
```text
- この式は DDQ / Q / EQ が **唯一の生存経路**であることを前提にするが、現行では TerminalReclaim が **無制限の最終退避層**として存在し、DDQ+Q+EQ が満杯でも world は `TerminalReclaimAuthority::store()`（always true）へ無制限に退避される（`src/audioengine/ISRRetireRouter.cpp:27-45`, `P-4: growable store`）。
- したがって capacity の合計は「通常経路の容量」であって outstanding world 数の上界ではない。M を名乗るには **全経路の合算が bounded であること**が証明されなければならないが、現行は TerminalReclaim が ∞ であるため証明不能。
- 仮に TerminalReclaim を bounded `K` に置き換えても、`M = 1+4096+512+512+K` が成立するのは **「exhaustion 時に非所有 return をせず、backpressure で admission を停止する」ことが同時に証明された場合に限る**。そうでなければ、K+1 個目の world で所有権が失われ UAF になるため、M は安全な上界ではない。

---

## 3. D101-2 で必ず調べる invariant（4項目）

### 3.1 World lifetime bound — どの式を authoritative invariant にするか

#### 候補式

```text
(a) outstandingWorlds <= M
(b) publishedWorlds - reclaimedWorlds <= M
```text
#### 現行の accounting（コード監査結果）

| イベント | カウンタ / 観測点 | 意味 | 備考 |
| --- | --- | --- | --- |
| **acquire** | `ISRWorldRetirementReference::onAcquireObserved()` — `AudioEngine.Commit.cpp:406` の **1箇所のみ**（publish 成功時） | 1 world につき 1 acquire | failed publication は acquire を発生させない |
| **publish** | `publicationSequenceCounter_` (`AudioEngine.h:3451`) の `fetchAdd` | 単調増加の採番（ID 空間の非重複保証） | **world 個数上限を意味しない** — uint64 の連番 |
| **reclaim / release** | `worldReclaimCount_` (`ISRRetireRouter.cpp:400-406`) — `provider_->worldReclaimCount() + m_retireQuarantine.worldReclaimCount() + m_emergencyQuarantine.worldReclaimCount() + m_terminalReclaim.reclaimCount()` | world 物理破棄数（type==World の terminal deleter 実行時） | 4経路合算。1 world = 1 acquire + 1 release |
| **shutdown discard** | `shutdownReclaim()` → `terminalReclaim()` → `TerminalReclaimAuthority::store()`（`ISRRetireRouter.cpp: P-4`） | 所有権は失わず TerminalReclaim へ移送 → epoch-gated drain → `drainAll()` で完全破棄 | `ptr` を捨てない。`ownerChannel` residual も `drainAllNonRt` → TerminalReclaim 経由 |
| **failed publication** | `publishAndSwap` / `commitRuntimePublication` の失敗パス | acquire なし。World は構築前または構築失敗として破棄 | outstanding に含めない |
| **superseded publication** | 旧 current の retire（`RuntimeStore::store` 後の旧 world） | 旧 world は DDQ/Q/EQ/Terminal のいずれかに enqueue → epoch 通過後に reclaim | 生存期間は publish から reclaim まで |

#### 判定

- **正しい invariant は (a) `outstandingWorlds <= M`** である。ここで `outstandingWorlds` は **生存 world 数** = `current(1) + DDQ滞留 + Q滞留 + EQ滞留 + TerminalReclaim滞留` の厳密な合計。
- (b) `publishedWorlds - reclaimedWorlds <= M` は、**acquire/release/shutdown discard/failed publication/superseded publication の accounting が混ざらず、かつ全てが同一の worldReclaimCount 上で 1:1 対応すること**が証明された場合に限り (a) と同値。現行の `worldReclaimCount_` は 4経路合算で 1:1 を満たすが、**shutdown 時の drainAll 前後で published と outstanding の差が一時的に乖離する**ため、(b) を authoritative invariant として採用するには `isFullyDrained()` / `residentCount()` との同期条件を明示する必要がある。
- したがって D101 の M は **(a) を正規形**とし、(b) は計測用の派生式（telemetry）として扱うのが安全。

#### 現行で M を証明できない理由（この invariant に関して）

- `outstandingWorlds` の構成要素のうち `TerminalReclaim` が ∞ であるため、`outstandingWorlds <= 5121` は証明不能。
- 仮に TerminalReclaim を bounded `K` にしても、**cap の enforcement（後述の admission control）がなければ**、producer が cap を超えて publish し `store()` が非所有 return をすれば outstanding の定義自体が崩れる（UAF により world が生存とも破棄とも言えない状態になる）。

---

### 3.2 TerminalReclaim の意味を確定（本監査の核心）

#### 現行定義

```cpp
// src/audioengine/ISRRetireRouter.h — class TerminalReclaimAuthority
// ★ P-4: Growable store (std::vector) — Non-RT only, heap allocation acceptable.
class TerminalReclaimAuthority {
    std::vector<Entry> entries_;   // ← growable, 無制限
    // ...
    bool store(void* ptr, void (*deleter)(void*), uint64_t epoch, ...) noexcept;
    void drain(uint64_t minReaderEpoch, ...) noexcept;   // epoch-gated
    void drainAll() noexcept;                             // force (audio thread 停止後)
    std::size_t residentCount() const noexcept;
};
```text
```cpp
// src/audioengine/ISRRetireRouter.cpp:27-45
bool TerminalReclaimAuthority::store(...) noexcept {
    // P-4: growable store — ALWAYS accepts
    entries_.push_back(Entry{ptr, deleter, epoch, ...});
    return true;  // ★ 常に true
}
```text
- `RetireQuarantineStore` は `std::array<QuarantinedEntry, 512>`（固定配列、満杯時 `quarantine()` は `false` を返し deleter を実行しない = UAF 構造的排除）だが、`TerminalReclaimAuthority` は `std::vector` であるため **capacity exhaustion は発生しない**（OOM は `std::terminate` に対応）。
- 呼び出し経路: `ISRRetireRouter::enqueueWithRetry()` Stage 5 のみ `RetireEnqueueResult::TerminalReclaim` を返し、`RetireQuarantine(512) + EmergencyQuarantine(512) + DDQ(4096)` 全満時に `terminalReclaim(ptr, deleter, epoch, type, "enqueueWithRetry:TerminalReclaim")` へ移送。`RetireEnqueueResult::TerminalReclaim` は **ownership transfer 成立**として扱われる（`ConvoPeq.md:78088` の `Success / QueuePressure / TerminalReclaim は全て ownership transfer 成立`）。
- Shutdown 経路: `shutdownReclaim()` は `enqueueWithRetry()` を経由せず直接 `terminalReclaim()` を呼び、Q→E→Terminal の段階的退避をスキップして即座に TerminalReclaim へ移送。`AudioEngine.ReleaseResources.cpp:542` の `drainAllNonRt` callback も `shutdownReclaim → TerminalReclaimAuthority` に路由。

#### 2つの設計解釈（排他）

| 解釈 | TerminalReclaim の位置づけ | D101 の M との関係 | 含意 |
| --- | --- | --- | --- |
| **解釈 X: safety fallback（現行）** | overflow 時の所有権喪失を防ぐ **無制限の最終退避層**。Practical Stable ISR Bridge の「overflow をデータ喪失に直結させず、shutdown で完全 Drain」原則の具現化 | **finite lifetime bound の証明対象外**。M はこの architecture では要求できない | D101 の M 証明は UNPROVABLE のまま。M を求めるなら architecture 変更が必要 |
| **解釈 Y: bounded store（D101 の M を成立させる場合）** | `TerminalReclaim capacity <= K` を導入し、全経路の合算 `M = 1+4096+512+512+K` を有限上界として証明する | **M 証明の対象に含める** | 必ず exhaustion 時の handling を設計する必要がある |

**D101-1.5 は解釈 X を採用して NO_FINITE_BOUND を導いた**。D101-2 は解釈 Y に移行する場合の条件を分析する。

#### 解釈 Y に移行する場合の必須設計（単純な `vector → array` では不十分）

満杯になったとき、もし

```cpp
bool store(...) noexcept {
    if (entries_.size() >= K) return false;  // ← 非所有 return
}
```text
とすれば、呼び出し元は `ptr` の所有権を失い、**UAF リスク**になる。`RetireQuarantineStore` は満杯時に `false` を返して deleter を実行しないことで UAF を構造的に排除しているが、それは **呼び出し元が `false` を正しく handling して re-enqueue または再試行する**ことを前提にしている。`TerminalReclaimAuthority` が最終退避層である場合、そこから先の退避先がないため、`false` はそのまま **所有権喪失**を意味する。

したがって、bounded TerminalReclaim を導入するなら、必ず以下のいずれかを **ownership-preserving** に設計しなければならない:

```text
capacity exhausted
        ↓
  ┌─────────────────────────────────────────────────────┐
  │  選択肢（いずれも ownership を失わない）              │
  │                                                     │
  │  (1) backpressure / admission stop                  │
  │      新たな World の publish 自体を停止する。        │
  │      既存の ptr は呼び出し元が保持したまま再試行。    │
  │      新規 producer activity を block/queue する。    │
  │                                                     │
  │  (2) producer blocking (with timeout)                │
  │      呼び出しスレッドを block し、reclaim が進むまで  │
  │      待機。ただし RT スレッドでの blocking は禁止。   │
  │      → producer が non-RT（Timer/Message/UI）のみ     │
  │        なら許容。RT producer がいるなら不可。        │
  │                                                     │
  │  (3) reservation-first admission control             │
  │      World を構築する前に TerminalReclaim の         │
  │      reservation を取得。失敗なら構築自体を abort。   │
  │      → 所有権が発生する前に拒否するため UAF なし。    │
  │      I4 D14 の reservation-first と整合。            │
  │                                                     │
  │  (4) terminal-failure with retained ownership        │
  │      store() は false を返すが、ptr の所有権は       │
  │      呼び出し元に残る。呼び出し元は必ず再試行か      │
  │      shutdownReclaim 相当の退避を保証する。          │
  │      → 単なる false return ではなく、obligation を   │
  │        失わせない contract が必要。                  │
  └─────────────────────────────────────────────────────┘
```text
**I4 D14 との整合**: I4 D14 は既に「capacity 枯渇を terminal-failure にして obligation を失わせない」「reservation-first と backpressure を固定」する方向である。したがって D101-2 でも **「bounded にするため overflow 時に捨てる」安易な解決は禁止**。上記 (1) / (3) / (4) が D14 整合な選択肢であり、(2) は RT 非関与が証明された場合に限り許容。

#### 最優先検証: TerminalReclaim を bounded にした場合、capacity exhaustion 時に UAF / ownership loss を起こさず、かつ producer を安全に停止できるか

**検証結果: 条件付きで可能だが、現行 architecture のままでは不可能。Architectural change が必要。**

- **RT safety**: 現行の `TerminalReclaimAuthority` は `Non-RT only, heap allocation acceptable` と明記されており、RT スレッドからは呼ばれない（RT は DDQ/enqueue のみ）。Producer 側の backpressure / reservation-first も non-RT（Timer / Message / PrepareToPlay / PublicationExecutor / Orchestrator）で完結するため、**RT スレッドを block せずに producer を停止することは原理的に可能**。
- **しかし現行には reservation-first の admission control が存在しない**。`commitRuntimePublication` は複数 entry point（`AudioEngine.Processing.PrepareToPlay.cpp:155, 277` / `ReleaseResources.cpp:175` / `Timer.cpp:964` / `Transition.cpp:25` / `PublicationExecutor.cpp:53` / `RuntimePublicationOrchestrator.cpp:269`）から駆動され、**publish レートを制限する throttle / minInterval / cooldown は存在しない**。`AudioEngine.Commit.cpp:669` の `minIntervalTicks` は診断ログ用の 1 秒間隔であり publish 制限ではない。したがって、TerminalReclaim を bounded にしただけでは、producer が停止せず所有権喪失に至る。
- **さらに reclaim latency にも bound が必要**。Producer を停止しても、stuck reader が epoch を進めなければ reclaim は進まず、TerminalReclaim は満杯のまま producer は永久に停止（livelock / deadlock）する。これを避けるには **reader hold time にも finite bound** が必要（後述 3.3）。

**結論**: Bounded TerminalReclaim + ownership-preserving backpressure は **設計可能**だが、**現行の growable vector を array に置き換えるだけでは UAF を防げず、reservation-first admission control と producer rate bound と reader hold bound の 3点を追加する architectural change が必須**。

---

### 3.3 Producer bound

#### 現行の producer 構造（無制限の根拠）

| 観測事実 | ソース | 上界 |
| --- | --- | --- |
| publish producer が複数存在 | `commitRuntimePublication` の呼び出し元 6箇所（PrepareToPlay ×2 / ReleaseResources / Timer / Transition / PublicationExecutor / RuntimePublicationOrchestrator） | 同時多重駆動可能 |
| publish rate の hard limit がない | `minIntervalTicks` は診断用。throttle / cooldown なし | レート無制限 |
| 1 サンプリング区間内の acquire 数 | producer レートに依存 | 上界なし |
| DDQ の enqueue レート | `DeferredDeletionQueue::enqueue` は Vyukov MPMC で non-blocking だが、呼び出し回数自体は producer レートに依存 | 上界なし |
| outstanding World 数の増加速度 | `publish rate - reclaim rate` | publish が reclaim を上回ると TerminalReclaim に無制限蓄積 |

したがって **`Bmax = queue capacity` とはできない**。Queue capacity は「1 publish あたりの退避先容量」であって、**publish 自体の発生数を制限しない**。

#### D101-2 で列挙すべき contract として固定可能な bound 候補

| 候補 | 意味 | 現行で固定可能か | M 証明への寄与 |
| --- | --- | --- | --- |
| **最大同時 Build 数** | 同時に進行する `RuntimeBuilder::build` の並行数 | 現行は orchestrator / executor 経由で複数並行可能。hard cap なし | Build が publish に直結するなら、Build cap → publish cap に寄与 |
| **最大同時 Publish 数** | 同時に `commitRuntimePublication` を実行するスレッド数 / キュー長 | 現行は 6 entry point から多重駆動。hard cap なし | 直接の producer activity bound |
| **最大 publish burst** | 任意の時間窓 `T` 内の publish 回数上限 | 現行は burst 制限なし | `M = burst × worst-case reclaim latency` 導出に必須 |
| **最大 retire latency** | publish から reclaim（epoch 通過 → deque → deleter 実行）までの worst-case 時間 | retire は epoch 進行 + reclaim 呼び出しに依存。epoch 進行は reader 協調に依存 | `M = publish rate × retire latency`。latency に上界がなければ M も上界なし |
| **最大 reader hold time** | Reader（Audio Thread の block 処理等）が critical section / epoch を保持する最大時間 | Audio Thread の block サイズは有限だが、stuck reader / device 停止 / suspend 等で無制限に保持し得る | retire latency の上界は reader hold time の上界に依存。**reader hold が無制限なら、producer rate に上限を設けても outstanding lifetime の有限性は自動的には証明できない** |

#### 重要: `producer rate bound × reclaim latency` で M を導く方式の要件

```text
M = ceil( producer_rate_max × reclaim_latency_max ) + pipeline_depth
```text
この方式を採用するなら、**必ず `reclaim_latency_max < ∞` を証明しなければならない**。Reclaim latency は以下で決まる:

```text
reclaim_latency = epoch_advance_latency + quarantine_drain_latency + TerminalReclaim_drain_latency
                = f(reader_hold_time, timer_period, drain_call_frequency)
```text
- `reader_hold_time` が無制限なら `reclaim_latency` も無制限 → M も無制限。
- 現行の `AudioEngine.Timer.cpp` の定期 drain と `ReleaseResources.cpp` の `drainAllNonRt` は **best-effort** であり、worst-case latency の hard bound ではない。
- したがって、producer rate bound だけを導入しても **M は証明できない**。**同時に `reader hold time <= H_max` と `reclaim latency <= R_max` を contract として固定**する必要がある。

#### 現行で contract として固定可能なものの判定

- **同時 Build / 同時 Publish / burst**: いずれも現行は固定可能だが **architectural change（admission queue + reservation-first + throttle）が必要**。単なる invariant 宣言では enforcement されない。
- **retire latency / reader hold time**: 現行は有限性を保証する invariant が存在しない。Audio Thread の block 処理は名目上有限だが、stuck reader 時の `TerminalReclaimAuthority` 滞留（clearedWorld 等）は `drainTerminalReclaim` の epoch-gated drain に委ねられ、その進行は reader の epoch 進行に依存する。Reader が stuck すれば latency は無制限。

**結論**: Producer bound による M 導出は **architectural change（admission control + reader hold bound + reclaim latency bound の 3点セット）**が揃った場合に限り可能。現行のまま invariant を宣言するだけでは REPAIRABLE にならない。

---

### 3.4 `maxSamplingGapUs` を M の根拠に使わない（禁止の明文化）

#### 現行定義（TerminalReclaim 再掲）

```cpp
// src/audioengine/ISRWorldRetirementTelemetry.h:297-299
auto current = convo::consumeAtomic(maxSamplingGapUs_, ...);
... compareExchangeAtomic(maxSamplingGapUs_, current, gapUs, ...);  // 最大 gap を追跡
```text
- `maxSamplingGapUs_` は **観測された最大 gap を記録するだけ**の telemetry。`missed tick` 発生時の hard cap は存在しない。100ms は名目値であり D101-1 で確認済み。
- 同様に `minIntervalTicks`（`AudioEngine.Commit.cpp:669`）も診断ログ用の 1 秒間隔であり publish 制限ではない。

#### 禁止の根拠

```text
G = observed maxSamplingGapUs   ←  これは observed maximum であって contractual maximum ではない
G から M や R を導出するのは禁止
```text
- `observed maximum` は **過去の観測値の最大**であり、将来の worst-case を上界しない。新たな missed tick がより大きな gap を生めば G は更新されるが、それ以前の M 証明は invalid になる。
- `contractual maximum` は **設計が保証する hard cap**（例: `gap <= G_max` を満たさなければ system は fail-safe に移行する）でなければならない。現行にはそのような hard cap が存在しない。
- したがって、D101 の M や R（retire latency）を `G = observed maxSamplingGapUs` から導出することは **禁止**。D101-2 以降の全ての M 証明で `maxSamplingGapUs` を根拠にしてはならない。

---

## 4. 4案の合成と I4 D14 との整合

### 4.1 I4 D14 との整合

I4 D14 では既に以下が固定されている（ユーザー指示より）:

- **logical recovery obligation について reservation-first と backpressure が固定**
- **capacity 枯渇を terminal-failure にして obligation を失わせない**
- **bounded にするため overflow 時に捨てる安易な解決は禁止**

これは D101-2 の結論と完全に整合する:

| D101-2 の要求 | I4 D14 の固定 |
| --- | --- |
| TerminalReclaim を bounded にするなら、exhaustion 時に non-owning return を起こさず ownership を preserve する | terminal-failure にして obligation を失わせない |
| 単に `vector → array` に変えるだけでは不十分。backpressure / admission stop / producer blocking の設計が必要 | reservation-first と backpressure を固定 |
| overflow 時に捨てる（drop / leak）方式は禁止 | bounded にするため overflow 時に捨てる安易な解決は禁止 |

したがって、D101-2 で bounded TerminalReclaim を導入するなら、**必ず I4 D14 の reservation-first + backpressure パターンに従う**必要がある。

### 4.2 Practical Stable ISR Bridge 原則との整合

Practical Stable ISR Bridge の原則:

> overflow をデータ喪失に直結させず、shutdown で完全 Drain することが要求される
> `shutdown は Drain → Reclaim → Verify Empty が必要`

- 現行の growable TerminalReclaim はこの原則を **最も保守的に**満たす実装である（絶対に所有権を失わない）。
- Bounded TerminalReclaim に変更する場合も、**shutdown 時の Drain → Reclaim → Verify Empty が依然として成立する**ことを証明しなければならない。Bounded 化により shutdown 時に TerminalReclaim が満杯で `shutdownReclaim` が失敗するなら、原則違反になる。したがって shutdown 経路は **bounded 制約の例外（または shutdown 専用の unbounded drainAll）**として設計するか、`isFullyDrained()` / `residentCount()` / `worldReclaimCount()` の検証を bounded 下でも成立させる必要がある。
- 本監査では shutdown 経路の bounded 化は **STRUCTURALLY_INCOMPATIBLE ではない**と判定する。Shutdown は audio thread 停止後に `drainAll()`（force, epoch-gated ではない）で全てを解放するため、bounded TerminalReclaim であっても `drainAll()` が全エントリを解放すれば Verify Empty は成立する。ただし、そのためには `drainAll()` の capacity が shutdown 時の残留数以上であることが保証されなければならず、これは **shutdown 時の outstanding 上界と同一の M で保証される**（循環依存に注意）。

---

## 5. D101-2 最終判定

### 5.1 判定定義（指示どおりの3択）

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| **FINITE_BOUND_REPAIRABLE** | 既存 architecture に invariant を追加するだけで `M < ∞` を証明できる | **該当せず** — TerminalReclaim が growable である限り、invariant 追加だけでは M を証明できない。`std::vector → std::array` の変更は code change であり invariant 追加ではない |
| **FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE** | 現在の architecture のままでは不可能だが、`TerminalReclaim bounded + ownership-preserving backpressure + producer/lifetime bound` などを導入すれば証明可能 | **◯ 該当（本監査の結論）** |
| **FINITE_BOUND_STRUCTURALLY_INCOMPATIBLE** | Practical Stable ISR の「Overflowしても所有権を失わない」要求と finite hard cap が両立しないため、D101 の M をこの architecture では要求できない | **該当せず** — backpressure / reservation-first による所有権 preservative な hard cap は原理的に両立可能。RT 非関与の producer 停止 + bounded reader hold + bounded reclaim latency の 3点が揃えば、finite cap と ownership preservation は両立する |

### 5.2 なぜ ARCHITECTURAL_CHANGE_REQUIRED か（詳細）

**REPAIRABLE ではない理由**:

- 現行の `TerminalReclaimAuthority` は `std::vector`（growable）であり、invariant 宣言だけで bounded にすることはできない。必ず code change（`vector → bounded array + exhaustion handling`）が必要。
- Producer rate / reader hold time / reclaim latency のいずれにも現行は hard bound が存在せず、invariant 追加だけでは enforcement されない。必ず admission control / throttle / epoch hold bound の機構追加が必要。
- したがって「既存 architecture に invariant を追加するだけ」では M は証明できない。

**STRUCTURALLY_INCOMPATIBLE ではない理由**:

- Finite hard cap と ownership preservation は **backpressure で両立可能**。Growable sink を維持しなければ所有権を失うというのは、**drop 方式**を前提にした場合にのみ真である。Reservation-first で所有権発生前に admission を拒否するか、backpressure で producer を停止すれば、所有権を失わずに finite cap を enforce できる。
- RT safety の観点でも、producer は non-RT（Timer / Message / PrepareToPlay / Executor / Orchestrator）であり、non-RT の producer 停止は RT を block しないため、RT 制約と矛盾しない。Reader hold bound についても、audio block サイズ由来の名目 bound に stuck reader 検出 + fail-safe を追加すれば、reclaim latency に hard bound を与えることは原理的に可能。
- したがって「この architecture では M を要求できない」ではなく、「architecture を変更すれば M を要求できる」が正しい。

**ARCHITECTURAL_CHANGE_REQUIRED の具体的内容**:

D101 の M 証明を再開するには、最低でも以下を同時に導入する必要がある:

```text
1. TerminalReclaim bounded (capacity K)
   std::vector<Entry> → std::array<Entry, K> (K は M 導出の一部)
   + exhaustion 時の ownership-preserving handling（下記 2 と連携）

2. Ownership-preserving backpressure / admission control
   reservation-first: World 構築前に TerminalReclaim の slot を予約。失敗なら構築自体を abort
   または producer blocking / admission stop: TerminalReclaim 満杯時に新規 publish を停止
   いずれも I4 D14 の reservation-first + backpressure パターンに従う
   RT スレッドでの blocking は禁止（producer が non-RT であることを証明）

3. Outstanding-world hard cap (M)
   outstandingWorlds <= M の authoritative invariant を宣言し、
   M = 1 (current) + 4096 (DDQ) + 512 (Q) + 512 (EQ) + K (TerminalReclaim) として証明
   ただし K は (2) の backpressure が正しく機能することの証明と同時に成立

4. Producer bound
   最大同時 Build / 同時 Publish / burst の hard cap（admission queue 長、同時実行数制限、レートリミット）
   publish rate <= R_max の contractual bound

5. Reader hold / reclaim latency bound
   reader critical section hold time <= H_max
   reclaim latency <= R_max  (H_max と timer/drain 周期から導出)
   M = ceil(R_max × publish_rate_max) + pipeline_depth 方式を採用する場合は必須

6. maxSamplingGapUs の M 根拠からの除外
   observed maximum を contractual maximum として使用しない

7. Shutdown Drain 証明の更新
   bounded 下でも shutdown 時の Drain → Reclaim → Verify Empty が成立することの証明
   （drainAll() の force 解放が bounded capacity 下でも残留数を上回ることの保証）
```text
これら 7点のうち 1点でも欠ければ M は証明不能。特に (1) と (2) は不可分であり、(1) だけを導入して (2) を欠けば UAF になる。

---

## 6. その後の順序（指示どおり）

```text
D101-1.5
   │
   ▼
NO_FINITE_BOUND
   │
   ▼
D101-2  ◀ 本監査
Finite-Bound Contract Repair Analysis
   │
   ├── bounded TerminalReclaim
   ├── outstanding-world cap
   ├── producer/lifetime bound
   └── incompatibility analysis
   │
   ▼
D101-2 Verdict: FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE
   │
   ▼
I4 Contract 更新
   │
   ▼
Phase I の GO/NO-GO 再判定
```text
- **本監査では production code を変更しない**（指示どおり）。
- I4 Contract 更新では、上記 7点の architectural change を I4 の D14（reservation-first + backpressure）と整合させて追記する。
- Phase I の GO/NO-GO 再判定は、I4 Contract 更新後に、上記 7点の architectural change を実施する意思決定がなされた場合に限り D101 の M 証明を再開する。そうでなければ D101 の M は UNPROVABLE のまま Phase I は NO-GO または M 要求を scope 外とする判断になる。

---

## 7. ソースリンク

| ソース | 役割 |
| --- | --- |
| `evidence/phase-d101-0-m-bound-mathematical-audit.md` | D101-0（verdict: INCOMPLETE） |
| `evidence/phase-d101-1-m-bound-step2-counter-observation-error.md` | D101-1（verdict: INCOMPLETE） |
| `evidence/phase-d101-1.5-finite-bound-source-audit.md` | D101-1.5（verdict: NO_FINITE_BOUND）— 本監査の前提 |
| `doc/work88/I4_DESIGN_CONTRACT.md` | I4 D14: reservation-first / backpressure / terminal-failure で obligation を失わせない |
| `src/DeferredDeletionQueue.h` | `kQueueSize = 4096`, Vyukov bounded MPMC, `sequences[]` / `ringBuffer[]` |
| `src/audioengine/RetireQuarantineStore.h` | `kMaxQuarantinedEntries = 512`, `std::array<QuarantinedEntry, 512>`, 満杯時 `false` |
| `src/audioengine/ISRRetireRouter.h` | `class TerminalReclaimAuthority { std::vector<Entry> entries_; ... }`, P-4 growable store, Non-RT only |
| `src/audioengine/ISRRetireRouter.cpp:27-45` | `TerminalReclaimAuthority::store()` — always true, `entries_.push_back()` |
| `src/audioengine/ISRRetireRouter.cpp:enqueueWithRetry()` | Stage 5: `RetireEnqueueResult::TerminalReclaim`, `terminalReclaim(..., "enqueueWithRetry:TerminalReclaim")` |
| `src/audioengine/ISRRetireRouter.cpp:400-406` | `worldReclaimCount()` — 4経路合算 |
| `src/audioengine/ISRWorldRetirementTelemetry.h:297-299` | `maxSamplingGapUs_` — observed maximum 追跡のみ、hard cap なし |
| `src/audioengine/AudioEngine.Commit.cpp:406, 669` | `onAcquireObserved()` 1箇所のみ / `minIntervalTicks` は診断用 |
| `src/audioengine/AudioEngine.h:3451` | `publicationSequenceCounter_` — 単調増加採番、個数上限ではない |
| `evidence/15-P-4-0-terminal-reclaim-audit.md` | TerminalReclaim の contract（P-4 growable, OOM は terminate） |
| `evidence/15-P-10-shutdown-authority-terminal-ownership-cross-audit.md` | `TerminalReclaimAuthority` singleton（`ISRRetireRouter.h:358` の by-value member） |
| `evidence/15-P-12-shutdown-authority-closure-final-audit.md` | `m_terminalReclaim` singleton + `isFullyDrained() == true && terminalReclaimResident == 0` 検証 |
| `ConvoPeq.md:53671-54319` | `TerminalReclaimAuthority` 実装・所有権チェーン `D → Q → EmergencyQ → TerminalReclaimAuthority` |

---

## 8. 付録: D101-2 監査チェックリスト

- [x] World lifetime bound の authoritative invariant を確定（(a) `outstandingWorlds <= M` が正規形、(b) は派生式）
- [x] `acquire / release / shutdown discard / failed publication / superseded publication` の accounting が混ざらないことを確認
- [x] TerminalReclaim の意味（safety fallback vs bounded store）を 2解釈で確定
- [x] `vector → array` だけでは不十分であり、exhaustion 時に non-owning return で UAF になることを確認
- [x] `capacity exhausted → ownership-preserving backpressure / admission stop / producer blocking` の設計要件を列挙
- [x] I4 D14 の「capacity 枯渇を terminal-failure にして obligation を失わせない」との整合を確認
- [x] Producer bound（最大同時 Build / 同時 Publish / burst / retire latency / reader hold time）の列挙
- [x] `producer rate bound × reclaim latency` 方式では reclaim latency 自体にも finite bound が必要なことを確認
- [x] Reader が無制限に critical section を保持できるなら producer rate 上限だけでは lifetime 有限性を証明できないことを確認
- [x] `maxSamplingGapUs` が observed maximum であり contractual maximum ではないことを確認
- [x] `G = observed maxSamplingGapUs` から M や R を導出するのは禁止として明記
- [x] Queue capacity（4096/512/512）を M と読み替えないことを確認
- [x] `M = 1+4096+512+512` が TerminalReclaim（∞）により不成立であることを確認
- [x] TerminalReclaim を bounded にした場合の capacity exhaustion 時の UAF / ownership loss 検証を最優先で実施
- [x] 最終判定を 3択（REPAIRABLE / ARCHITECTURAL_CHANGE_REQUIRED / STRUCTURALLY_INCOMPATIBLE）で確定
- [x] Production code 変更なし（設計監査のみ）
