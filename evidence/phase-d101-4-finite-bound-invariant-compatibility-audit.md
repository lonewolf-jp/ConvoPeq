# D101-4 — Finite-Bound Invariant Compatibility Audit

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-4 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（最新ソースコード）、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/DeferredDeletionQueue.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/AudioEngine.Processing.*.cpp`、`src/audioengine/AudioEngine.Timer.cpp`、`src/audioengine/ISRWorldRetirementTelemetry.h`、`src/audioengine/ISRWorldRetirementReference.h`、`doc/work88/I4_DESIGN_CONTRACT.md` |
| **前提** | D101-3 verdict: **CONTRACT_REQUIRES_NEW_INVARIANT** — 10 の Required architectural invariants を契約として定義。`M_world = M_current + M_reader + M_retire + M_quarantine + M_terminal` は契約モデルであり実装上の事実ではない |
| **目的** | D101-3 の 10 invariant について、現行コードに「既に存在する / 部分的に存在する / 契約上は定義できるが実装根拠がない / 現行設計と衝突する」を `ConvoPeq.md` を一次資料として 1項目ずつ照合する |
| **制約** | **コード変更なし・適合性監査のみ**。`M_world` 分解式の各項（特に `M_quarantine = 1024` / `M_retire = 4096`）をそのまま `M_world` の有限上限とみなさない。重点監査: `M_terminal` / reservation-first / `H_max` / `A_max` / `P_max` / `G_contract` / budget 分離 / ownership conservation / conservation chain / shutdown 有限完了性 |
| **判定** | **ARCHITECTURAL_CHANGE_REQUIRED** — 10 invariant のうち 2 は SATISFIED（既存容量）、1 は PARTIAL（quarantine chain）、7 は MISSING（`M_terminal` / reservation-first / `H_max` / `A_max` / `P_max` / `G_contract` / budget 分離）、conservation / chain / shutdown は bounded 化後の再証明を要する。現行設計との衝突（CONFLICT）はなし |

---

## 1. Scope

- D101-3 で定義した 10 の Required architectural invariants を、現行コード（`ConvoPeq.md` 一次資料）と I4 契約に照合する。
- 判定は `SATISFIED / PARTIAL / MISSING / CONFLICT` の 4 段階。各項目で Evidence と Gap を明示する。
- `M_world = M_current + M_reader + M_retire + M_quarantine + M_terminal` は **契約モデル**であり、既存容量値（4096/512/512/1024）をそのまま `M_world` の有限上限とみなさない。
- 本監査の後に D101-5 で「どの invariant をどのコード境界に実装するか」を設計する。

---

## 2. D101-3 引継ぎ — 10 invariants 一覧

| # | Invariant | D101-3 での位置づけ |
| --- | --- | --- |
| I01 | `M_terminal ≤ K < ∞` | `TerminalReclaimAuthority` を bounded に |
| I02 | `M_terminal reservation-first` | publish 前に slot 予約、失敗→backpressure |
| I03 | `M_reader ≤ f(H_max, P_max) < ∞` | Reader hold の並行度を上界 |
| I04 | `H_max < ∞` | Reader hold / reclaim latency の契約上限 |
| I05 | `A_max < ∞` | 1 interval の acquire 上限 |
| I06 | `P_max < ∞` | 時間窓あたりの publish activity 上限 |
| I07 | `G_max = G_contract < ∞` | Sampler gap の契約上限（telemetry から導出しない） |
| I08 | `RuntimeWorld lifetime budget` の明示 | `logical-obligation budget ≠ RuntimeWorld lifetime budget` |
| I09 | `Ownership conservation (RuntimeWorld)` | `publishedWorlds = liveWorlds + reclaimedWorlds` |
| I10 | `Shutdown Drain → Reclaim → Verify Empty` の bounded 下での成立 | Bounded でも shutdown が有限完了 |

---

## 3. 適合性監査 — I01〜I10

### I01 — `M_terminal` bounded

```text
I01 M_terminal bounded
    Status: MISSING
    Evidence:
      - ConvoPeq.md: class TerminalReclaimAuthority { std::vector<Entry> entries_; ... }
        P-4 コメント「GROWABLE — ALWAYS accepts — NO store full failure path」
      - ISRRetireRouter.h: 同上、std::vector<Entry> entries_; std::mutex mtx_; std::atomic<uint32_t> residentAtomic_
      - ISRRetireRouter.cpp: bool TerminalReclaimAuthority::store(...) {
            std::lock_guard<std::mutex> lock(mtx_);
            entries_.push_back(Entry{...});
            residentAtomic_.fetch_add(1); return true; // ALWAYS true
        }
      - RetireQuarantineStore.h: 対照的に std::array<QuarantinedEntry, 512> で bounded、
        満杯で store() は false（deleter 実行せず）
    Gap:
      - TerminalReclaim は唯一の unbounded authority。M_terminal = ∞。
      - Bounded 化には std::vector → std::array<Entry, K> + index 配置への置換が必要。
      - K の値は M_world 分解から導出（K = M_world - (1+4096+1024) - M_reader）。
      - 現行設計との衝突はなし（RetireQuarantineStore が既に bounded であり、同一パターンを適用可能）。
```text

**Status: MISSING** — 現行は growable であり、bounded invariant は存在しない。D101-2/D101-3 で既に確定した事実の再確認。

---

### I02 — reservation-first

```text
I02 M_terminal reservation-first
    Status: MISSING
    Evidence:
      - ConvoPeq.md / ISRRetireRouter.cpp: enqueueWithRetry() は D→Q→EQ→Terminal の順に
        ownership transfer を試みるが、TerminalReclaim が growable であるため reservation は存在しない。
        いずれかが成功すれば ownership は失われない（現行は必ず成功）。
      - I4 D14: reservation-first は logical-obligation budget に対して定義済み
       （新規 logical obligation 生成に対して exactly once、D18.4。COALESCE は新規 reservation を取得しない）。
        だが RuntimeWorld lifetime に対する reservation-first は未定義。
      - RuntimeWorldAuthority / RuntimeBuilder / AudioEngine.Commit.cpp: publish 経路に
        M_terminal slot の reservation 取得は存在しない。
    Gap:
      - RuntimeWorld publish 前の reservation 取得機構が不存在。
      - Reservation → ownership transfer → store の間に失敗可能な操作がないことの証明も不存在。
      - D 案の reservation-first（std::array + index で allocation-free）を新設する必要がある。
      - 現行設計との衝突はなし（I4 の reservation-first と同型パターンを流用可能）。
```text

**Status: MISSING** — I4 の reservation-first は logical obligation にのみ適用され、RuntimeWorld には適用されていない。

---

### I03 — `M_reader` bound

```text
I03 M_reader ≤ f(H_max, P_max)
    Status: MISSING (derived, depends on I04/I06)
    Evidence:
      - RuntimeStore.h: class RuntimeStore { std::atomic<T*> current; WriteAccess::publishAndSwap / observe() }
        current は単一スロット、observe() は borrow（非所有）参照。
      - AudioEngine.Processing.*.cpp: Reader (Audio Thread) は processBlock 単位で observe() し
        block 終了後に参照を離す。block 時間（sampleRate/blockSize）に由来する hold は有限だが、
        異常系（stall/suspend/デバッガ停止）では indefinite に延長し得る。
      - D101-3 chapter 8: M_reader ≤ f(H_max, P_max) は H_max と P_max から導出される派生 bound。
    Gap:
      - H_max (I04) と P_max (I06) が MISSING であるため、M_reader も導出不能。
      - Reader の並行度が 1 (Audio Thread 単一) であることは既存事実だが、retire drain が H_max 時間停止すれば
        その間に publish された旧 world が滞留し M_reader の寄与は H_max に比例して増大する。
      - H_max / P_max の導入後に M_reader = f(H_max, P_max) として導出可能。
```text

**Status: MISSING** — 派生 bound であり、前提の `H_max` / `P_max` が不存在のため導出不能。

---

### I04 — `H_max`（reader hold bound）

```text
I04 H_max — reader hold / reclaim latency の契約上限
    Status: MISSING
    Evidence:
      - EpochDomain.h: struct EpochDomain { uint64_t epoch; bool isOlder(uint64_t a, uint64_t b) ... }
        wraparound-safe な epoch 比較。isOlder(entry.epoch, minReaderEpoch) == true なら safe。
      - RetireQuarantineStore.h: drain(minReaderEpoch, isOlderFn) は entry.epoch < minReaderEpoch のみ deleter 実行。
        TerminalReclaimAuthority::drain も同様。
      - DeferredDeletionQueue: reclaim(minReaderEpoch) も同様の epoch 比較。
      - 現行コードに H_max / maxBlockDuration / watchdog / HealthEvent による reader hold 上限は存在しない。
      - Audio Thread の block 時間は有限だが、異常系の hold を上界する invariant は不存在。
    Gap:
      - H_max < ∞ を保証する invariant が不存在。D101-3 chapter 8 の方式 a (block 時間由来 + fail-safe) を
        新 invariant として導入する必要がある。
      - H_max 超過時の producer backpressure（流入停止）も同時に必要。
      - 現行設計との衝突はなし（block 時間由来の H_max は既存の sampleRate/blockSize から導出可能、
        異常系 fail-safe は HealthEvent 機構の拡張で実現可能）。
```text

**Status: MISSING** — 現行に `H_max` の契約は存在しない。D101-2 で指摘した「reader が無制限に hold できるなら producer rate 上限だけでは lifetime 有限性を証明できない」は依然として成立。

---

### I05 — `A_max`（admission bound）

```text
I05 A_max — 1 sampling interval の acquire 上限
    Status: MISSING
    Evidence:
      - AudioEngine.Timer.cpp / ISRWorldRetirementTelemetry.h: sampler は観測専用（T1 telemetry）。
        D76.4「T1 telemetry state is observational state and is not a reservation authority」。
        sampler 自体に admission 制御は存在しない。
      - I4 D14/D18.4 の coalesce は同一 CoalesceIdentity への重複を吸収するが、
        異なる target への publish は吸収しない。A_max の制御は存在しない。
      - 現行の publish 経路（RuntimeBuilder / OwnerChannel / PendingPublishRegistry=64）に
        interval あたりの acquire 上限は存在しない。
    Gap:
      - 1 interval の acquire 数を上界する invariant が不存在。
      - Reservation-first と連動した admission control として導入する必要がある。
      - 現行設計との衝突はなし（coalesce による吸収と admission queue による制御を併用可能）。
```text

**Status: MISSING**

---

### I06 — `P_max`（pending publication bound）

```text
I06 P_max — 時間窓あたりの publish activity 上限
    Status: MISSING
    Evidence:
      - RuntimeWorldAuthority.h: PendingPublishRegistry kPendingPublishCapacity = 64。
        enqueue→commit の async gap を bounded にするが、publish 呼び出し頻度自体は unbounded。
      - RuntimeBuilder: buildWorld() の呼び出し頻度に hard limit は存在しない。
      - I4 の kMaxLogicalRecoveryObligations (=32 候補) は logical obligation の budget であり、
        RuntimeWorld publish の P_max ではない（I08 で分離）。
    Gap:
      - 時間窓あたりの publish 回数を上界する invariant が不存在。
      - Reservation-first と連動した rate limiter / admission queue として導入する必要がある。
      - 現行設計との衝突はなし（PendingPublishRegistry 64 は gap の bounded 性を示すが、頻度自体の bound ではない）。
```text

**Status: MISSING** — `PendingPublishCapacity = 64` は gap の容量であり、publish 頻度の上限ではない。

---

### I07 — `G_contract`（sampling/progress の契約上限）

```text
I07 G_contract — sampler gap の契約上限
    Status: MISSING
    Evidence:
      - ISRWorldRetirementTelemetry.h: maxSamplingGapUs は observed maximum（telemetry）。
        sample() 間隔の最大値を記録するが、missed tick 時の hard cap は存在しない。
      - AudioEngine.Timer.cpp: timer の定期 drain は best-effort であり、worst-case latency の hard bound ではない。
      - 現行コードに G_max = G_contract の契約は存在しない。
      - D101-2/D101-3 で禁止した G_max = observed maxSamplingGapUs は現行でも不成立。
    Gap:
      - G_contract として設計時に固定する値と、gap 超過時の bounded recovery + producer admission control が不存在。
      - Telemetry (maxSamplingGapUs) と contract (G_contract) の分離も不存在。
      - 新 architectural contract として導入する必要がある。
      - 現行設計との衝突はなし（telemetry 観測と contract 固定は独立）。
```text

**Status: MISSING** — 現行は observed maximum のみであり、contractual maximum は存在しない。

---

### I08 — `M_world` と logical-obligation budget の分離

```text
I08 M_world budget separation — logical-obligation budget ≠ RuntimeWorld lifetime budget
    Status: MISSING
    Evidence:
      - I4 D14: kMaxLogicalRecoveryObligations (候補 32) は logical recovery obligation の budget。
        reservation-first は新規 logical obligation 生成に対して exactly once（D18.4）。
      - ConvoPeq.md / RuntimeWorldAuthority: RuntimeWorld lifetime に対する M_world budget は未定義。
        I4 にも RuntimeWorld lifetime budget の記述は存在しない。
      - D101-3 chapter 12 で両 budget の分離を契約として定義したが、I4 への追記は未実施。
    Gap:
      - I4 Contract への追記が必要: 「logical-obligation budget ≠ RuntimeWorld lifetime budget」を明示し、
        両者を別 invariant・別数値・別機構として契約化する。
      - 混同を防ぐための命名・ドキュメント分離も必要。
      - 現行設計との衝突はなし（I4 は前者のみ定義しており、後者の追加は拡張である）。
```text

**Status: MISSING** — I4 への追記が未実施。契約定義は D101-3 で完了しているが、I4 への反映が残る。

---

### I09 — ownership conservation の再定義

```text
I09 Ownership conservation (RuntimeWorld)
    Status: MISSING (bounded 化後の再証明を要する)
    Evidence:
      - 現行の conservation: ISRRetireRouter::enqueueWithRetry() は D→Q→EQ→TerminalReclaim の順に
        ownership transfer を試み、TerminalReclaim が growable であるため必ず成功。
        したがって現行は publishedWorlds = liveWorlds + reclaimedWorlds が成立（growable 前提）。
      - I4 D15/D18.3: admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount
        は logical obligation の conservation であり、RuntimeWorld の conservation とは別式。
      - D101-3 chapter 7 で RuntimeWorld 用の conservation を再定義:
        publishedWorlds = liveWorlds + reclaimedWorlds
        liveWorlds = currentWorlds(1) + readerHeldWorlds + retireQueueWorlds + quarantineWorlds + terminalWorlds
        だが、bounded 化後の再証明は未実施。
    Gap:
      - Bounded M_terminal での conservation 条件（reservation-first / allocation-free store /
        drain 進行保証 / shutdown drainAll）の再証明が不存在。
      - I4 の logical obligation conservation と RuntimeWorld conservation の分離を I4 に追記する必要もある。
      - 現行設計との衝突はなし（growable 前提の現行 conservation は bounded 化で再証明すれば維持可能）。
```text

**Status: MISSING** — 現行は growable 前提で成立するが、bounded 化後の再証明が未実施。

---

### I10 — shutdown / terminal drain の有限完了性

```text
I10 Shutdown Drain → Reclaim → Verify Empty の bounded 下での成立
    Status: MISSING (bounded 化後の再証明を要する)
    Evidence:
      - 現行: TerminalReclaimAuthority::drainAll() は audio thread 停止後に全 pending を強制解放。
        drain() は epoch-safe 到達後に drain、drainAll() は epoch に関わらず全解放。
        Growable であるため shutdown 時の残留数に関わらず drainAll() は成功し Verify Empty が成立。
      - ConvoPeq.md: drainAllQuarantineStore() が Q + EmergencyQ + TerminalReclaimAuthority を全て解放。
      - Bounded 化後の懸念: M_terminal = K が shutdown 時の worst-case 残留数を下回れば、
        shutdown 時に TerminalReclaim が満杯で drainAll() が残留数を上回れない可能性。
        ただし D101-3 chapter 13 の指摘どおり、K は M_world の上界と同一の値から導出されるため、
        K が worst-case 残留数を上回ることの保証には循環依存への注意が必要。
    Gap:
      - Bounded 下での drainAll() の capacity 保証の証明が不存在。
      - K の値が shutdown 時の worst-case 残留数を上回ることの保証が必要。
      - 現行設計との衝突はなし（drainAll() 自体は bounded array でも全 entry を走査して解放可能）。
```text

**Status: MISSING** — 現行は growable で成立するが、bounded 化後の有限完了性の再証明が未実施。

---

## 4. 既存容量の扱い — `M_world` 分解式との整合

### 4.1 既存 bounded 容量

| 容量 | 値 | 現行の保証 | `M_world` 分解式での位置 |
| --- | --- | --- | --- |
| `RuntimeStore::current` | 1 | 単一 `std::atomic<T*> current` | `M_current = 1` ✅ SATISFIED |
| `DeferredDeletionQueue kQueueSize` | 4096 | Vyukov MPMC bounded queue | `M_retire ≤ 4096` ✅ SATISFIED |
| `RetireQuarantineStore` | 512 | `std::array` 固定容量 | `M_quarantine` の一部 |
| `EmergencyQuarantineStore` | 512 | 同上（別インスタンス） | `M_quarantine ≤ 1024` ⚠️ PARTIAL（個別には SATISFIED だが、溢れた場合の最終所有権が growable に流れるため `M_world` の上界としては不十分） |
| `PendingPublishRegistry` | 64 | `kPendingPublishCapacity = 64` | publish 前の gap のみ。`M_world` の outstanding には含まない |

### 4.2 なぜ既存容量をそのまま `M_world` の有限上限とみなさないか

- `M_world = 1 + 4096 + 512 + 512 = 5121` は **通常経路容量**であり、`M_world` の証明ではない（D101-1.5 / D101-2 / D101-3 で繰り返し確定）。
- `M_terminal = ∞` であるため、既存容量の合計は `M_world` の上界にならない。
- 正しい `M_world` は:

```text
M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max)
```text

であり、既存容量はその一部（`1 + 4096 + 1024`）に過ぎない。残りの `K + f(H_max, P_max)` が新 invariant なしには `∞` であるため、`M_world` 全体も `∞` である。

- `M_quarantine = 1024` は `PARTIAL` と判定する: 個別の store は bounded だが、Quarantine 満杯時の退避先が growable TerminalReclaim であるため、`M_quarantine` 単独の bounded 性は `M_world` の有限性に寄与しない。

---

## 5. Conservation chain の整合性

### 5.1 現行の chain

```text
publish admission → ownership transfer (publishAndSwap) → retire (enqueueWithRetry: D→Q→EQ→TerminalReclaim)
    → drain(minReaderEpoch, isOlder) → deleter → release
```text

- 現行は TerminalReclaim が growable であるため、全ての publish された world は必ずいずれかの authority に所有され、epoch safe 到達後に必ず reclaim される（conservation は growable 前提で成立）。

### 5.2 Bounded 化後の chain（D 案）

```text
publish admission (A_max/P_max 制御)
    ↓
reservation (M_terminal slot 取得、失敗→backpressure)
    ↓
ownership transfer (publishAndSwap、旧 world を retire へ)
    ↓
retire (enqueueWithRetry: D→Q→EQ→Terminal[reserved slot]、失敗しない)
    ↓
drain(minReaderEpoch, isOlder, H_max 保証) → deleter → release
    ↓
shutdown: drainAll() → Verify Empty
```text

- Bounded 化後の chain が conservation を維持するには、I01〜I10 の全てが充足される必要がある。
- 現行の chain との整合性: 既存の `enqueueWithRetry` のフォールバック構造（D→Q→EQ→Terminal）は維持し、Terminal のみを `std::vector` から `std::array<K>` + reservation-first に置換する。chain 自体の構造は変わらないため、現行設計との衝突はなし。

---

## 6. 総合判定

### 6.1 I01〜I10 サマリ

| # | Invariant | Status | 備考 |
| --- | --- | --- | --- |
| I01 | `M_terminal ≤ K < ∞` | **MISSING** | `std::vector` growable → bounded `std::array<K>` に要置換 |
| I02 | `M_terminal reservation-first` | **MISSING** | RuntimeWorld に対する reservation-first は未定義 |
| I03 | `M_reader ≤ f(H_max, P_max)` | **MISSING** | `H_max` / `P_max` に依存する派生 bound |
| I04 | `H_max < ∞` | **MISSING** | Reader hold 上限は現行不存在 |
| I05 | `A_max < ∞` | **MISSING** | Interval acquire 上限は現行不存在 |
| I06 | `P_max < ∞` | **MISSING** | Publish activity 上限は現行不存在（PendingPublish 64 は gap 容量であり頻度上限ではない） |
| I07 | `G_max = G_contract < ∞` | **MISSING** | Sampler gap 契約は現行不存在（observed maximum のみ） |
| I08 | `RuntimeWorld lifetime budget` の分離 | **MISSING** | I4 への追記が未実施。契約定義は D101-3 で完了 |
| I09 | `Ownership conservation (RuntimeWorld)` | **MISSING** | Bounded 化後の再証明を要する |
| I10 | `Shutdown 有限完了性` | **MISSING** | Bounded 下での drainAll() 保証の再証明を要する |
| — | `M_current = 1` | **SATISFIED** | `RuntimeStore` 単一スロット |
| — | `M_retire ≤ 4096` | **SATISFIED** | `DeferredDeletionQueue` bounded |
| — | `M_quarantine ≤ 1024` | **PARTIAL** | 個別 store は bounded だが最終所有権が growable に流れる |

### 6.2 Verdict

```text
Overall: ARCHITECTURAL_CHANGE_REQUIRED
```text

| 判定 | 定義 | 本監査の該当性 |
| --- | --- | --- |
| `CONTRACT_SUFFICIENT` | 既存コード・既存 I4 契約だけで全 invariant を充足し `M_world < ∞` を証明できる | **該当せず** — 10 invariant のうち 7 が MISSING、2 が再証明待ち |
| `CONTRACT_REQUIRES_REFINEMENT` | 契約の精緻化（I4 追記等）だけで充足可能 | **該当せず** — I08 の I4 追記だけでは不十分。I01/I02/I04/I05/I06/I07 のコード変更が必須 |
| `ARCHITECTURAL_CHANGE_REQUIRED` | コード変更（bounded 化 / reservation-first / H_max / A_max / P_max / G_contract の導入）を要する | **◯ 該当（本監査の結論）** |
| `STRUCTURALLY_UNPROVABLE` | 現行 architecture では `M_world < ∞` を要求できない（有限 cap と ownership が両立しない） | **該当せず** — D 案により両立可能。衝突（CONFLICT）は 0 件 |

### 6.3 なぜ `ARCHITECTURAL_CHANGE_REQUIRED` か

- **SATISFIED は 2 のみ**（`M_current` / `M_retire`）。`M_quarantine` は `PARTIAL` であり、`M_world` の上界としては不十分。
- **MISSING は 7**（I01/I02/I04/I05/I06/I07/I08）+ **再証明待ち 2**（I09/I10）。いずれも現行コード・現行 I4 契約には存在せず、新 invariant の導入が必須。
- **CONFLICT は 0** — 現行設計と新 invariant の間に構造的衝突はなく、全て D 案（unified lifetime budget + reservation-first + backpressure）により導入可能である。したがって `STRUCTURALLY_UNPROVABLE` ではない。
- D101-2/D101-3 と一貫して `ARCHITECTURAL_CHANGE_REQUIRED` が継続する。本監査はその内訳を 10 invariant × 4 段階で精密化したものである。

---

## 7. Next gate

```text
D101-2  FINITE_BOUND_REQUIRES_ARCHITECTURAL_CHANGE
   │
   ▼
D101-3  CONTRACT_REQUIRES_NEW_INVARIANT
   │  10 Required invariants を契約として定義
   ▼
D101-4  ARCHITECTURAL_CHANGE_REQUIRED  ◀ 本監査
   │  10 invariants × 4 段階照合
   │  SATISFIED: 2 / PARTIAL: 1 / MISSING: 7+2(再証明待ち) / CONFLICT: 0
   ▼
I4 Contract 更新（blocking）
   │
   ├── I08: logical-obligation budget ≠ RuntimeWorld lifetime budget の明示
   ├── D101-3 の M_world 分解と各 bound の契約定義を I4 に追記
   └── I09/I10 の conservation / shutdown の再定義を I4 に追記
   │
   ▼
D101-5 — Invariant 実装設計（コード変更の設計）
   │
   ├── どの invariant をどのコード境界に実装するか
   ├── M_terminal bounded (K) の値と配置
   ├── reservation-first の機構設計
   ├── H_max / A_max / P_max / G_contract の具体的機構
   └── M_world の証明と backpressure progress の設計
   │
   ▼
D101-6 — 実装 + 検証
   │
   ▼
Phase I GO/NO-GO 再判定
```text

- **本監査でも production code は変更しない**（指示どおり）。
- I4 Contract 更新は D101-5 の前提となる blocking item。I08 の budget 分離を I4 に追記し、D101-3 の契約定義を I4 に反映する。
- D101-5 では、本監査で MISSING と判定した 7+2 invariant について「どのコード境界に実装するか」を設計する。`M_world = 1+4096+512+512` をそのまま上限とみなすことは依然として禁止である。

---

## 付録: D101-4 監査チェックリスト

- [x] D101-3 の 10 Required invariants を 1項目ずつ照合（I01〜I10）
- [x] 各項目を MISSING / PARTIAL / SATISFIED / CONFLICT の 4 段階で判定
- [x] 各項目で Evidence（ConvoPeq.md 一次資料）と Gap を明示
- [x] `M_terminal` の有限性（I01）を重点監査（std::vector growable の確認）
- [x] reservation-first（I02）を重点監査（RuntimeWorld に対する reservation は不存在）
- [x] `H_max`（I04）を重点監査（reader hold bound は現行不存在）
- [x] `A_max`（I05）を重点監査（admission bound は現行不存在）
- [x] `P_max`（I06）を重点監査（PendingPublish 64 は gap 容量であり頻度上限ではない）
- [x] `G_contract`（I07）を重点監査（observed maximum から導出しない）
- [x] `M_world` と logical-obligation budget の分離（I08）を重点監査（I4 への追記が未実施）
- [x] ownership conservation の再定義（I09）を重点監査（bounded 化後の再証明を要する）
- [x] `publish admission → reservation → ownership transfer → retire → release` の conservation chain の整合性を検証
- [x] shutdown / terminal drain の有限完了性（I10）を重点監査（bounded 下での drainAll() 保証の再証明を要する）
- [x] `M_world` 分解式（`M_current + M_reader + M_retire + M_quarantine + M_terminal`）が実際の所有権・容量モデルと整合するか検証
- [x] 既存容量値（4096/512/512/1024）をそのまま `M_world` の有限上限とみなさない
- [x] Production code 変更なし（適合性監査のみ）
- [x] 総合判定を 4択（`CONTRACT_SUFFICIENT / CONTRACT_REQUIRES_REFINEMENT / ARCHITECTURAL_CHANGE_REQUIRED / STRUCTURALLY_UNPROVABLE`）で確定
