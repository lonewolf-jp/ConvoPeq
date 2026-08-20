# D101-5 — Finite-Bound Authority Mapping

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-5 |
| **日付** | 2026-08-20 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md`（最新ソースコード）、`src/audioengine/ISRRetireRouter.h/.cpp`、`src/audioengine/RetireQuarantineStore.h`、`src/DeferredDeletionQueue.h`、`src/core/RuntimeStore.h`、`src/core/EpochDomain.h`、`src/audioengine/RuntimeWorldAuthority.h`、`src/audioengine/RuntimeBuilder.h`、`src/audioengine/AudioEngine.Commit.cpp`、`src/audioengine/AudioEngine.Timer.cpp`、`src/audioengine/ISRWorldRetirementTelemetry.h`、`doc/work88/I4_DESIGN_CONTRACT.md`、D101-3/D101-4 evidence |
| **前提** | D101-4 verdict: **ARCHITECTURAL_CHANGE_REQUIRED** — 10 invariant のうち 7 が MISSING、2 が再証明待ち、2 が SATISFIED、CONFLICT は 0 |
| **目的** | D101-4 の `ARCHITECTURAL_CHANGE_REQUIRED` を、**新規 invariant をどの既存責務（Authority/State/Enforcement Boundary）へ割り当てるか**の architectural mapping として具体化する。I01/I02/I08 を別々に設計しすぎず、`publish admission → reservation → ownership transfer → retire → release` を一つの lifetime-budget conservation chain として扱えるかを判定し、実装順序（案A/案B）を確定する |
| **制約** | **コード変更なし・mapping 定義のみ**。I01〜I10 の各 invariant で Authority/State/Reservation point/Consumption/Release/Enforcement boundary/Existing code/Required change を確定する |
| **判定** | **ARCHITECTURAL_CHANGE_REQUIRED（具体化完了）** — 10 invariant の mapping を確定し、lifetime-budget conservation chain を一つの invariant として扱えることを判定。実装順序は **案B**（lifetime budget authority → reservation → ownership conservation → M_terminal/M_reader/M_retire/M_quarantine → M_world）が正しい |

---

## 1. Scope

- D101-3 で定義した 10 invariants のうち、7 が MISSING、2 が再証明待ちであることを D101-4 で確定した。
- D101-5 では、各 invariant を **どの Authority が所有するのか、State をどこに置くのか、Enforcement をどの境界で行うのか**を、現行の `ConvoPeq.md` の責務構造に照合して mapping する。
- I01/I02/I08 を別々に設計しすぎず、D101-3 の D 案（`publish admission → reservation → ownership transfer → retire → release` を一つの lifetime-budget conservation chain として扱う）が成立するかを判定する。
- 最後に Implementation Order の案A/案B を D101-3/D101-4 と現行 `ConvoPeq.md` に照合して判定する。

---

## 2. Mapping Rules

| ルール | 内容 |
| --- | --- |
| **Authority は単一責任** | 一つの invariant は一つの Authority が所有する。複数 Authority が同一 invariant を所有すれば race が生じる |
| **State は Authority の内部** | invariant の状態（count / slot / epoch）は Authority の内部 State に置く。外部からの直接操作は禁止し、Authority の API 経由のみで操作する |
| **Enforcement boundary は呼び出し境界** | invariant の強制は、呼び出し元と Authority の境界で行う。Authority 内部でのみ強制しても、呼び出し元が迂回すれば invariant は破れる |
| **Reservation point は ownership transfer の前** | reservation は所有権の移転前に取得する。移転後に reservation を取得すれば、移転と reservation の間に失敗可能な操作が存在し ownership が失われる可能性がある |
| **Chain は単一 invariant** | `publish admission → reservation → ownership transfer → retire → release` は一つの lifetime-budget conservation chain として扱う。分割すれば chain の途中で invariant が破れる |

---

## 3. I01 — `M_terminal < ∞`

```text
I01 M_terminal
  Authority: ISRRetireRouter::TerminalReclaimAuthority
  State: std::array<Entry, K> entries_ + std::mutex mtx_ + std::atomic<uint32_t> residentAtomic_
         （現行は std::vector<Entry> entries_ → bounded array に置換）
  Reservation point: 新規 RuntimeWorld publish 前（I02 と統合、D 案の conservation chain 内）
  Consumption: reservation 成功時に slot を消費。store() は予約済み slot への配置であり失敗しない
  Release: drain(minReaderEpoch, isOlderFn) で epoch-safe 到達後に deleter 実行、
           residentAtomic_.fetch_sub() で slot 解放。drainAll() は shutdown 時に全 pending を強制解放
  Enforcement boundary: ISRRetireRouter::TerminalReclaimAuthority::store() の呼び出し境界。
                        reservation なしでの store() は禁止（assert / HealthEvent）
  Existing code: TerminalReclaimAuthority は std::vector growable（P-4）。RetireQuarantineStore は
                 既に std::array<512> で bounded であり、同一パターンを適用可能
  Required architectural change:
    - ISRRetireRouter.h: std::vector<Entry> → std::array<Entry, K> + index 管理に置換
    - K の値は M_world 分解から導出（K = M_world - (1+4096+1024) - M_reader）
    - 現行設計との衝突はなし（RetireQuarantineStore が既に bounded であり、同一パターンを適用可能）
  Notes:
    - K は shutdown 時の worst-case 残留数を上回る必要がある（I10 との循環依存に注意）
    - RetireQuarantineStore の std::array + mutex + index 配置と同一パターンで allocation-free に
```

**Status**: MISSING → mapping 確定。Authority は既存の `ISRRetireRouter::TerminalReclaimAuthority` を bounded 化する。State は `ConvoPeq.md` の「GROWABLE」コメントを bounded array に置換する。

---

## 4. I02 — reservation-first

```text
I02 M_terminal reservation-first
  Authority: ISRRetireRouter::TerminalReclaimAuthority（I01 と同一 Authority）
             ただし reservation の admission 制御は RuntimeWorldAuthority / AudioEngine.Commit が協調
  State: TerminalReclaimAuthority 内部の slot 管理（空き slot 数を atomic で管理）
  Reservation point: RuntimeWorld publish 前、RuntimeBuilder::buildWorld() の前
                     または AudioEngine.Commit::commitRuntimePublication() の publishAndSwap 前
                     正確には「新規 RuntimeWorld の所有権が RuntimeStore::current に移転する前」
  Consumption: reservation 成功 → slot index を取得 → publish → reserved slot へ配置
  Release: build 失敗時は reservation を release（所有権は発生していないため UAF なし）
           drain() で epoch-safe 到達後に slot を解放
           drainAll() で shutdown 時に全 slot を解放
  Enforcement boundary: RuntimeWorld publish の呼び出し境界（RuntimeWorldAuthority / AudioEngine.Commit）。
                        reservation なしでの publish は禁止
                        I4 D14 の reservation-first と同型: 新規 logical obligation 生成に対して
                        exactly once（D18.4）→ RuntimeWorld でも新規 publish に対して exactly once
  Existing code: I4 D14 の reservation-first は logical-obligation budget に対して定義済み。
                 RuntimeWorld に対する reservation-first は未定義
  Required architectural change:
    - RuntimeWorld publish 前の reservation 取得機構を新設（I4 の reservation-first と同型）
    - reservation → ownership transfer → store の間に失敗可能な操作がないことの証明（C/D 案の証明）
    - COALESC 相当の重複 publish は新規 reservation を取得しない（D18.4 と同様）
  Notes:
    - I01 と I02 は同一 Authority（TerminalReclaimAuthority）で協調して実現する。分離しない
    - I4 D18.4 の「COALESCE は新規 reservation を取得しない」を RuntimeWorld にも適用する
```

**Status**: MISSING → mapping 確定。I01 と I02 は **同一 Authority で一体として設計する**（D 案）。分離すれば reservation と store の間に race が生じる。

---

## 5. I03 — `M_reader ≤ f(H_max, P_max) < ∞`

```text
I03 M_reader
  Authority: EpochDomain（reader hold の時間的 bound）+ RuntimeWorldAuthority（publish 頻度の bound）
             派生 bound であり、単一 Authority ではなく H_max / P_max から導出される
  State: なし（派生値）。H_max と P_max の State から計算される
  Reservation point: なし（派生 bound のため reservation 自体は H_max / P_max で制御される）
  Consumption: なし（派生 bound）
  Release: なし（派生 bound）
  Enforcement boundary: AudioEngine.Processing.*.cpp の processBlock 境界。
                        Reader は observe() で borrow し block 終了後に参照を離す
  Existing code: RuntimeStore::observe() は borrow（非所有）参照。Audio Thread は単一 reader
                 だが、retire drain が H_max 時間停止すれば滞留が増大する
  Required architectural change:
    - H_max (I04) と P_max (I06) の導入後に M_reader = f(H_max, P_max) として導出
    - 現行の M_reader = 1（単一 Audio Thread）は通常時の値であり、異常時の滞留は H_max に依存
  Notes:
    - M_reader は独立した invariant ではなく、H_max と P_max の関数として M_world 分解式で扱う
    - D101-3 chapter 8 の方式 a (block 時間由来 + fail-safe) が H_max の導入方法
```

**Status**: MISSING (derived) → mapping 確定。M_reader は独立した State を持たず、H_max/P_max から導出される派生値として扱う。

---

## 6. I04 — `H_max < ∞`

```text
I04 H_max — reader hold / reclaim latency の契約上限
  Authority: EpochDomain（epoch 進行の管理）+ AudioEngine（block 時間の管理）
  State: H_max 値自体は設計時の定数（例: maxBlockDuration + ε）。異常系 fail-safe の状態は
         HealthEvent / watchdog timer で管理
  Reservation point: なし（H_max は時間的 bound であり、slot reservation ではない）
  Consumption: なし（時間的 bound）
  Release: epoch 進行により自動的に reader hold が解放される。H_max 超過時は fail-safe が発動
  Enforcement boundary: EpochDomain::isOlder() の呼び出し境界（drain の epoch 比較）。
                        AudioEngine.Processing の block 境界（reader hold の開始/終了）
  Existing code: EpochDomain は epoch 比較（isOlder）のみを提供し、H_max の契約は存在しない。
                 RetireQuarantineStore::drain() / TerminalReclaimAuthority::drain() は
                 minReaderEpoch の進行に依存するが、その進行速度の保証は存在しない
  Required architectural change:
    - H_max = maxBlockDuration + ε として定義（blockSize/sampleRate から導出）
    - 異常系（stall/suspend/デバッガ停止）では HealthEvent / watchdog で fail-safe
    - H_max 超過時の producer backpressure（流入停止）を同時に実装
    - D101-3 chapter 8 の方式 a を第一候補
  Notes:
    - H_max は単一の時間的 bound であり、M_reader と G_contract の前提となる
    - 現行の EpochDomain は H_max の機構を持たないが、epoch 進行の観測は可能
```

**Status**: MISSING → mapping 確定。Authority は `EpochDomain` + `AudioEngine` の協調。State は設計時定数 `H_max` + 異常系 fail-safe。

---

## 7. I05 — `A_max < ∞`

```text
I05 A_max — 1 sampling interval の acquire 上限
  Authority: RuntimeWorldAuthority（publish 側）+ ISRWorldRetirementTelemetry（sampling 側、観測のみ）
  State: A_max 値自体は設計時の定数。acquire count は RuntimeWorldAuthority の publish 成功数から導出
  Reservation point: なし（A_max は 1 interval の上限であり、slot reservation とは別レイヤー）
  Consumption: 1 interval 内の publish 成功数が A_max に達すれば、残りの publish は backpressure で BLOCK
  Release: interval 終了時に acquire count をリセット（次の interval で再び A_max まで許可）
  Enforcement boundary: RuntimeWorldAuthority::publish / AudioEngine.Commit の publish 境界。
                        1 interval 内の publish 成功数をカウントし、A_max 超過で BLOCK
  Existing code: ISRWorldRetirementTelemetry は観測専用（D76.4）。sampler 自体に admission 制御は存在しない。
                 I4 の coalesce は同一 CoalesceIdentity への重複を吸収するが、A_max の制御は存在しない
  Required architectural change:
    - 1 interval の acquire 上限を reservation-first と連動した admission control として導入
    - Coalesce による吸収と admission queue による制御を併用
    - G_contract (I07) と連動（gap 超過時は A_max 制御も発動）
  Notes:
    - A_max は P_max (I06) と関連するが、A_max は interval 単位、P_max は時間窓単位で異なる粒度
    - D101-3 の producer bound として定義済み
```

**Status**: MISSING → mapping 確定。Authority は `RuntimeWorldAuthority`。Telemetry は観測のみであり、A_max の強制は publish 側で行う。

---

## 8. I06 — `P_max < ∞`

```text
I06 P_max — 時間窓あたりの publish activity 上限
  Authority: RuntimeWorldAuthority（publish 頻度の管理）+ RuntimeBuilder（build 頻度の管理）
  State: P_max 値自体は設計時の定数。PendingPublishRegistry (kPendingPublishCapacity=64) は
         gap の容量であり、P_max の頻度上限ではない
  Reservation point: なし（P_max は時間窓の上限であり、個別 slot reservation とは別レイヤー）
  Consumption: 時間窓内の publish 成功数が P_max に達すれば、残りの publish は backpressure で BLOCK
  Release: 時間窓終了時に publish count をリセット
  Enforcement boundary: RuntimeBuilder::buildWorld() / RuntimeWorldAuthority::publish の呼び出し境界。
                        時間窓内の publish 成功数をカウントし、P_max 超過で BLOCK
  Existing code: PendingPublishRegistry=64 は gap の bounded 性を示すが、publish 頻度の上限ではない。
                 RuntimeBuilder::buildWorld() の呼び出し頻度に hard limit は存在しない
  Required architectural change:
    - 時間窓あたりの publish 回数を上界する invariant として導入
    - Reservation-first と連動した rate limiter / admission queue として実装
    - PendingPublishRegistry 64 は gap の容量であり、P_max の代わりにはならないことを明示
  Notes:
    - P_max と A_max (I05) は関連するが、P_max は時間窓単位、A_max は interval 単位で異なる粒度
    - H_max (I04) と P_max の積が M_world の流入側を決定する
```

**Status**: MISSING → mapping 確定。Authority は `RuntimeWorldAuthority` + `RuntimeBuilder`。PendingPublishRegistry 64 は P_max の代わりにならない。

---

## 9. I07 — `G_max = G_contract < ∞`

```text
I07 G_contract — sampler gap の契約上限
  Authority: ISRWorldRetirementTelemetry（観測）+ AudioEngine.Timer（timer 進行）
             ただし G_contract の強制は publish 側（RuntimeWorldAuthority）で行う
  State: G_contract 値自体は設計時の定数（例: block 時間の数倍）。maxSamplingGapUs は
         observed maximum であり、G_contract から導出しない
  Reservation point: なし（G_contract は時間的 bound）
  Consumption: なし（時間的 bound）
  Release: gap が G_contract を超過した場合、bounded recovery が発動（HealthEvent / producer admission control）
  Enforcement boundary: ISRWorldRetirementTelemetry の gap 検出境界と RuntimeWorldAuthority の
                        publish admission 境界。gap 超過時は新規 publish を BLOCK
  Existing code: ISRWorldRetirementTelemetry::maxSamplingGapUs は observed maximum（telemetry）。
                 AudioEngine.Timer の定期 drain は best-effort であり、worst-case latency の hard bound ではない
  Required architectural change:
    - G_contract として設計時に固定する値（例: block 時間の数倍）を定義
    - maxSamplingGapUs は telemetry として観測・記録を継続するが、G_contract の根拠にはしない
    - Gap 超過時の bounded recovery + producer admission control を実装
    - D101-2/D101-3 で禁止した G_max = observed maxSamplingGapUs は現行でも不成立
  Notes:
    - G_contract と maxSamplingGapUs は独立した値として分離する。両者を混同しない
    - H_max (I04) と G_contract は関連するが、H_max は reader hold、G_contract は sampler gap で異なる対象
```

**Status**: MISSING → mapping 確定。Authority は `ISRWorldRetirementTelemetry`（観測）+ `RuntimeWorldAuthority`（強制）。Telemetry と contract を分離する。

---

## 10. I08 — budget separation

```text
I08 RuntimeWorld lifetime budget separation
  Authority: I4 Design Contract（契約レベル）+ RuntimeWorldAuthority（実装レベル）
  State: 2 つの独立した budget:
         - kMaxLogicalRecoveryObligations (=32 候補): logical recovery obligation の budget
         - M_world: RuntimeWorld lifetime の budget
         両者は別 invariant、別数値、別機構
  Reservation point: 各 budget で独立した reservation-first
                     - Logical obligation: 新規 logical obligation 生成に対して exactly once (D18.4)
                     - RuntimeWorld: 新規 RuntimeWorld publish に対して exactly once
                     COALESCE 相当の重複は新規 reservation を取得しない（両 budget 共通）
  Consumption: 各 budget の reservation 成功時に消費
  Release: 各 budget の terminal disposition 後に解放
           - Logical obligation: successCount + supersededCount + shutdownDiscardCount
           - RuntimeWorld: drain() で epoch-safe 到達後に deleter 実行
  Enforcement boundary: I4 Contract の契約境界。各 budget の admission 制御は独立した Authority で行う
  Existing code: I4 D14 の reservation-first は logical obligation にのみ適用。
                 RuntimeWorld に対する M_world budget は未定義
  Required architectural change:
    - I4 Contract への追記: 「logical-obligation budget ≠ RuntimeWorld lifetime budget」を明示
    - 両 budget を別 invariant・別数値・別機構として契約化
    - D101-3 chapter 12 の budget 分離定義を I4 に追記
  Notes:
    - 本 invariant は契約レベル（I4 追記）であり、コード変更よりも契約変更が主体
    - D101-3 で契約定義は完了しているが、I4 への反映が残る
```

**Status**: MISSING → mapping 確定。Authority は I4 Contract + RuntimeWorldAuthority。契約レベルの分離が主体であり、コード変更よりも I4 追記が先行する。

---

## 11. I09 — ownership conservation

```text
I09 Ownership conservation (RuntimeWorld)
  Authority: ISRRetireRouter（retire 経路の ownership transfer）+ RuntimeStore（publish 側の ownership transfer）
             + EpochDomain（epoch 進行の保証）
             ただし conservation 自体は M_world 全体の chain として扱う（I01/I02/I08 と統合）
  State: publishedWorlds = liveWorlds + reclaimedWorlds
         liveWorlds = currentWorlds(1) + readerHeldWorlds + retireQueueWorlds + quarantineWorlds + terminalWorlds
         reclaimedWorlds = drainSuccessCount（deleter 実行済み）
         invariant: liveWorlds ≤ M_world < ∞
  Reservation point: I02 の reservation-first と統合（publish 前の reservation 取得）
  Consumption: I02 の reservation 成功時に消費
  Release: I01 の drain() / drainAll() で epoch-safe 到達後に deleter 実行、slot 解放
  Enforcement boundary: ISRRetireRouter::enqueueWithRetry() の D→Q→EQ→TerminalReclaim chain 境界。
                        I4 の logical obligation conservation と RuntimeWorld conservation の分離を明示
                        I4: admittedLogicalObligationCount = liveOwnershipCount + terminalDispositionCount
                        D101-3: publishedWorlds = liveWorlds + reclaimedWorlds（別式）
  Existing code: 現行は TerminalReclaim が growable であるため必ず成功し、conservation は growable 前提で成立。
                 I4 D15/D18.3 の conservation は logical obligation の式であり、RuntimeWorld の式とは別
  Required architectural change:
    - Bounded M_terminal での conservation 条件の再証明:
      1. reservation-first（I02）で publish 前に slot を確保
      2. reservation → ownership transfer → store の間に失敗可能な操作がない（I02 の証明）
      3. store は予約済み slot への配置であり失敗しない（I01 の allocation-free）
      4. drain() で epoch-safe 到達後に必ず deleter が実行される（I04 の H_max 保証）
      5. shutdown 時に drainAll() が全 pending を解放（I10）
    - I4 の logical obligation conservation と RuntimeWorld conservation の分離を I4 に追記
  Notes:
    - I09 は I01/I02/I08 と統合して一つの lifetime-budget conservation chain として扱う
    - I04 (H_max) の保証がなければ、epoch が進行せず drain が停止し conservation が破れる
```

**Status**: MISSING (bounded 化後の再証明を要する) → mapping 確定。Authority は `ISRRetireRouter` + `RuntimeStore` + `EpochDomain` の協調。I01/I02/I08 と統合して一つの chain として扱う。

---

## 12. I10 — `M_world < ∞`

```text
I10 M_world formal bound
  Authority: 上位 invariant — I01〜I09 の統合として定義される。単一 Authority ではなく、全体の chain の保証
  State: M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max) < ∞
         各項は I01〜I09 から:
           M_current = 1 (SATISFIED, RuntimeStore)
           M_retire ≤ 4096 (SATISFIED, DeferredDeletionQueue)
           M_quarantine ≤ 1024 (PARTIAL→MISSING, RetireQuarantineStore×2 だが bounded 化で再証明)
           M_terminal ≤ K (MISSING→I01/I02 で導入)
           M_reader ≤ f(H_max, P_max) (MISSING→I03/I04/I06 で導入)
  Reservation point: I02 の reservation-first と統合
  Consumption: I02 の reservation 成功時に消費
  Release: I01/I04 の drain() で解放
  Enforcement boundary: M_world 全体の chain 境界。流入（A_max/P_max + reservation-first）と
                        流出（H_max による epoch 進行保証）の両面から M_world の有限性を証明
  Existing code: 既存容量 1+4096+1024 は SATISFIED/PARTIAL だが、K + f(H_max,P_max) が ∞ であるため
                 M_world 全体は ∞。既存容量をそのまま M_world の有限上限とみなすことは禁止
  Required architectural change:
    - I01〜I09 の全てが充足された後に M_world < ∞ を形式的証明
    - 流入: publish rate ≤ P_max ∧ 1 interval acquire ≤ A_max ∧ reservation-first
    - 流出: epoch 進行速度 ≥ 1/H_max → drain() で reclaim
    - 異常系: H_max/G_contract 超過時は流入を backpressure で停止
    - Shutdown: drainAll() で有限回で完了（I10 の shutdown 有限完了性）
  Notes:
    - M_world は独立した invariant ではなく、I01〜I09 の統合として定義される上位 bound
    - D101-3 chapter 11 の流入・流出の観点からの再表現を参照
```

**Status**: MISSING (上位 invariant) → mapping 確定。M_world は I01〜I09 の統合として定義される。上位 bound であり、単一 Authority ではなく全体の chain の保証。

---

## 13. Cross-Invariant Dependency

### 13.1 依存グラフ

```text
I01 (M_terminal bounded)
  ↔ I02 (reservation-first)          — 同一 Authority で一体設計。分離すれば race
  ↔ I08 (budget separation)          — M_terminal は RuntimeWorld budget の一部
  ↔ I09 (ownership conservation)     — bounded 化後の conservation 再証明に I01/I02 が必須

I04 (H_max)                          — 全ての時間的 bound の前提
  ↔ I03 (M_reader)                   — M_reader = f(H_max, P_max) の派生
  ↔ I06 (P_max)                      — H_max × P_max が M_world の流入側
  ↔ I09 (ownership conservation)     — H_max がなければ epoch が進行せず drain が停止
  ↔ I10 (M_world)                    — H_max がなければ M_world は ∞

I05 (A_max)                           — interval 単位の流入制御
  ↔ I06 (P_max)                      — 時間窓単位の流入制御。両者は異なる粒度だが流入側で協調
  ↔ I10 (M_world)                    — 流入側の上界

I07 (G_contract)                      — sampler gap の契約
  ↔ I04 (H_max)                      — gap 超過時の bounded recovery が H_max の fail-safe と連動
  ↔ I09 (ownership conservation)     — gap 超過時の producer admission control が流入側の保証

I08 (budget separation)               — 契約レベルの分離
  ↔ I01/I02 (M_terminal)             — RuntimeWorld budget の一部として M_terminal を位置づけ
  ↔ I09 (ownership conservation)     — 2 つの budget の conservation を別式として分離

I09 (ownership conservation)          — 全 chain の conservation
  ↔ I01/I02/I04/I08/I10              — 全ての invariant が conservation の前提

I10 (M_world)                         — 上位 invariant
  ↔ I01〜I09 全て                    — 全ての invariant の統合として定義される
```

### 13.2 依存の方向（実装順序への含意）

```text
H_max (I04) ──────→ M_reader (I03) ──────→ M_world (I10)
     │                    ↑
     └────→ G_contract (I07) ─┘

M_terminal (I01) ↔ reservation-first (I02) ──→ ownership conservation (I09) ──→ M_world (I10)
     ↑                                              ↑
budget separation (I08) ────────────────────────────┘

A_max (I05) ──→ M_world (I10)（流入側）
P_max (I06) ──→ M_world (I10)（流入側）

Shutdown (I10 sub) ──→ drainAll() 有限完了性（K が worst-case 残留数を上回る）
```

---

## 14. Authority Conflicts

### 検証結果: 衝突（CONFLICT）は 0 件

| 検証項目 | 結果 | 根拠 |
| --- | --- | --- |
| `ISRRetireRouter::TerminalReclaimAuthority` の bounded 化 | 衝突なし | `RetireQuarantineStore` が既に `std::array<512>` で bounded であり、同一パターンを適用可能。`ConvoPeq.md` の P-4 コメント「GROWABLE — ALWAYS accepts」は bounded reservation-first に置換可能 |
| `EpochDomain` の `H_max` 導入 | 衝突なし | `EpochDomain` は `isOlder()` の epoch 比較のみを提供し、`H_max` の時間的 bound は上位レイヤー（`AudioEngine.Processing` の block 時間 + `AudioEngine.Timer` の watchdog）で導入する。`EpochDomain` 自体の変更は不要 |
| `RuntimeWorldAuthority` の `A_max` / `P_max` 導入 | 衝突なし | `RuntimeWorldAuthority` は `PendingPublishRegistry(64)` で gap の bounded 性を示すが、頻度自体の bound ではない。`A_max` / `P_max` は publish 境界での admission control として追加可能 |
| `ISRWorldRetirementTelemetry` の `G_contract` 導入 | 衝突なし | Telemetry は観測専用（D76.4）であり、`G_contract` は publish 側の強制として導入するため、Telemetry 自体の変更は不要。両者を分離することで衝突を回避 |
| `I4` の budget 分離 | 衝突なし | I4 は `kMaxLogicalRecoveryObligations` のみ定義しており、`M_world` budget の追加は拡張である。両 budget は別 invariant・別数値・別機構として分離するため、既存 I4 の変更は追記のみ |
| `ISRRetireRouter::enqueueWithRetry()` の chain 変更 | 衝突なし | 既存の `D→Q→EQ→TerminalReclaim` chain は維持し、TerminalReclaim のみを `std::vector` から `std::array<K>` + reservation-first に置換する。chain 自体の構造は変わらない |
| RT/Non-RT 境界 | 衝突なし | `TerminalReclaimAuthority::store()` の全 callers は Non-RT（`enqueueWithRetry` は Non-RT path）。`P-4` の前提「all callers are Non-RT」は現行で成立し、bounded 化後も Non-RT のみで完結する |

**総合**: 新 invariant の導入は全て既存 Authority の拡張または上位レイヤーでの追加として実現可能であり、既存の責務構造との衝突は存在しない。D101-4 の `CONFLICT: 0` を再確認した。

---

## 15. Implementation Order

### 15.1 案A vs 案B

**案A:**

```text
reservation
  ↓
M_terminal
  ↓
ownership conservation
  ↓
M_world
```

**案B:**

```text
lifetime budget authority
  ↓
reservation
  ↓
ownership conservation
  ↓
M_terminal / M_reader / M_retire / M_quarantine
  ↓
M_world
```

### 15.2 判定: 案Bが正しい

| 検証観点 | 案A の問題 | 案B の正しさ |
| --- | --- | --- |
| **Budget の位置づけ** | reservation を最初に置くが、reservation が何の budget に対するものかが未定義 | lifetime budget authority（`M_world` budget の定義と `logical-obligation budget ≠ RuntimeWorld lifetime budget` の分離）を最初に確定し、reservation がその budget の一部であることを明示する |
| **D101-3 の結論との整合** | D101-3 は `M_world = M_current + M_reader + M_retire + M_quarantine + M_terminal` の分解を定義し、D 案（unified lifetime budget）を第一候補とした。案A は M_terminal のみを先に扱い、分解全体の位置づけを後回しにする | 案B は lifetime budget authority を最初に確定し、その下で各項（M_terminal / M_reader / M_retire / M_quarantine）を位置づける。D101-3 の分解と D 案の考え方に整合する |
| **I08 の位置づけ** | 案A では I08（budget separation）が reservation の後に来るが、budget が何であるかを定義せずに reservation を設計することはできない | 案B では I08（budget separation）を最初に確定し、その後に reservation（I02）を設計する。契約の分離が先、機構の設計が後の順序が正しい |
| **M_world の上位性** | 案A では M_world を最後に置くが、M_terminal のみから M_world を導出することはできない（M_reader も必要） | 案B では M_terminal / M_reader / M_retire / M_quarantine の各項を ownership conservation の後に位置づけ、最後に M_world を統合する。各項の有限性が証明された後に上位 bound として M_world を定義する順序が正しい |
| **Cross-invariant dependency との整合** | 案A は I01→I02→I08→I10 の依存を無視する | 案B は 13 章の依存グラフ（`I08 → I01↔I02 → I09 → I01/I03/I05/I06/I07 → I10`）に整合する |

### 15.3 確定した実装順序（案B 詳細化）

```text
Phase 1: Contract foundation
  1. I08: RuntimeWorld lifetime budget の分離（I4 Contract 追記）
     └── logical-obligation budget ≠ RuntimeWorld lifetime budget を明示

Phase 2: Core bounded authority
  2. I01: M_terminal bounded（TerminalReclaim std::array<K> 化）
  3. I02: reservation-first（publish 前の slot 予約 + backpressure）
     └── I01 と I02 は同一 Authority で一体設計（D 案）

Phase 3: Ownership guarantee
  4. I09: Ownership conservation の再証明
     └── reservation → ownership transfer → store の間に失敗可能な操作がないこと
     └── drain() の epoch 進行保証（I04 に依存）

Phase 4: Time-bounded guarantees
  5. I04: H_max（reader hold bound、block 時間由来 + fail-safe）
     └── AudioEngine.Processing の block 境界 + AudioEngine.Timer の watchdog
  6. I03: M_reader = f(H_max, P_max) の導出（派生、I04/I06 に依存）
  7. I07: G_contract（sampler gap 契約、telemetry と分離）
     └── ISRWorldRetirementTelemetry（観測）+ RuntimeWorldAuthority（強制）の分離

Phase 5: Producer bounds
  8. I05: A_max（interval acquire 上限、reservation-first と連動）
  9. I06: P_max（時間窓 publish 上限、reservation-first と連動）
     └── PendingPublishRegistry 64 は gap 容量であり P_max の代わりにならない

Phase 6: Integration
  10. I10: M_world < ∞ の形式的証明
      └── M_world ≤ 1 + 4096 + 1024 + K + f(H_max, P_max) < ∞
      └── 流入（A_max/P_max + reservation-first）と流出（H_max による epoch 進行）の両面から証明
      └── 異常系: H_max/G_contract 超過時は流入を backpressure で停止
  11. I10-sub: Shutdown Drain → Reclaim → Verify Empty の bounded 下での有限完了性
      └── K が shutdown 時の worst-case 残留数を上回ることの保証（循環依存に注意）

Phase 7: Verification
  12. D101-6: 実装 + 検証（M_world の証明と backpressure progress の証明）
  13. Phase I GO/NO-GO 再判定
```

### 15.4 なぜ案Bが第一候補として正しいか（D101-3 との接続）

- D101-3 は D 案（unified lifetime budget）を第一候補とし、`M_world` の分解を定義した。案B はその分解を実装順序に反映したものである。
- I08（budget separation）を最初に置くことで、以降の全ての invariant が「どの budget に対するものか」を明確にした上で設計される。案A ではこの前提が欠落する。
- 現行 `ConvoPeq.md` の責務構造（`ISRRetireRouter::TerminalReclaimAuthority` / `EpochDomain` / `RuntimeWorldAuthority` / `ISRWorldRetirementTelemetry`）は、案B の各 Phase での Authority 割り当てと整合する。新たな Authority の追加は不要であり、既存 Authority の拡張で実現可能である。

---

## 16. Verdict

### 判定: `ARCHITECTURAL_CHANGE_REQUIRED` — mapping 具体化完了

| 項目 | 内容 |
| --- | --- |
| **I01〜I10 の mapping** | 全10 invariant の Authority/State/Reservation point/Consumption/Release/Enforcement boundary/Existing code/Required change を確定。I01/I02/I08 は同一 chain（lifetime-budget conservation chain）として一体設計することを判定 |
| **Conservation chain** | `publish admission → reservation → ownership transfer → retire → release` を一つの lifetime-budget conservation chain として扱えることを判定。I01/I02/I08 を別々に設計しすぎず、D 案として統合する |
| **Cross-invariant dependency** | 13 章の依存グラフを確定。H_max が全ての時間的 bound の前提であり、M_terminal/reservation-first が ownership conservation の前提であることを明示 |
| **Authority conflicts** | 0 件。全ての新 invariant は既存 Authority の拡張または上位レイヤーでの追加として実現可能であり、衝突は存在しない |
| **Implementation order** | **案B** が正しい。lifetime budget authority（I08）→ reservation（I02）→ ownership conservation（I09）→ M_terminal/M_reader/M_retire/M_quarantine → M_world（I10）の順序で実装する |

### 全体判定

```text
D101-3  CONTRACT_REQUIRES_NEW_INVARIANT
   │
   ▼
D101-4  ARCHITECTURAL_CHANGE_REQUIRED（10 invariants × 4 段階照合）
   │
   ▼
D101-5  ARCHITECTURAL_CHANGE_REQUIRED（mapping 具体化完了）◀ 本監査
   │  I01〜I10 の Authority/State/Enforcement Boundary を確定
   │  Conservation chain を一つの invariant として扱えることを判定
   │  Implementation Order を案B として確定
   ▼
I4 Contract 更新（blocking）
   │
   ├── I08: logical-obligation budget ≠ RuntimeWorld lifetime budget の明示
   ├── D101-3 の M_world 分解と各 bound の契約定義を I4 に追記
   └── I09/I10 の conservation / shutdown の再定義を I4 に追記
   │
   ▼
D101-6 — Invariant 実装設計 + 実装
   │
   ├── Phase 1: I08 Contract foundation（I4 追記）
   ├── Phase 2: I01/I02 Core bounded authority（TerminalReclaim）
   ├── Phase 3: I09 Ownership guarantee
   ├── Phase 4: I04/I03/I07 Time-bounded guarantees
   ├── Phase 5: I05/I06 Producer bounds
   ├── Phase 6: I10 M_world 形式的証明 + Shutdown 有限完了性
   └── Phase 7: 検証 + Phase I GO/NO-GO 再判定
```

- **本監査でも production code は変更しない**（指示どおり）。
- I4 Contract 更新は D101-6 の前提となる blocking item。
- D101-6 では、本監査で確定した mapping と実装順序（案B）に従い、各 invariant の具体的機構を設計・実装する。

---

## 付録: D101-5 監査チェックリスト

- [x] D101-3 の 10 Required invariants を 1項目ずつ Authority/State/Reservation point/Consumption/Release/Enforcement boundary/Existing code/Required change で mapping
- [x] I01 `M_terminal` の有限容量を誰が予約・消費・解放するかを確定（ISRRetireRouter::TerminalReclaimAuthority）
- [x] I02 reservation-first の authority と reservation 前の ownership transfer 禁止点を確定（publish 前の slot 予約）
- [x] I03 `H_max` の reader hold 開始・終了・上限をどこで定義するかを確定（EpochDomain + AudioEngine.Processing）
- [x] I04 `A_max` の admission 有限上限と overflow/backpressure の authority を確定（RuntimeWorldAuthority）
- [x] I05 `P_max` の「容量」と「生成頻度」を分離して bounded にする方法を確定（RuntimeBuilder + PendingPublishRegistry との分離）
- [x] I06 `G_contract` の契約上の bound を何として定義するかを確定（telemetry と分離、observed maxSamplingGapUs は根拠にしない）
- [x] I07 budget separation の独立性を確定（I4 Contract レベルでの分離）
- [x] I08 ownership conservation の `admission → reservation → ownership transfer → retire → release` chain を確定
- [x] I09 shutdown finite completion の bounded 下での条件を確定（drainAll() 有限完了性、K の worst-case 保証）
- [x] I10 `M_world` の各項を上位 invariant として接続する方法を確定（`M_current + M_reader + M_retire + M_quarantine + M_terminal` の分解）
- [x] I01/I02/I08 を別々に設計しすぎないことを検証（D 案として一つの lifetime-budget conservation chain として扱えることを判定）
- [x] Implementation Order の案A/案B を現行 ConvoPeq.md と D101-3/D101-4 に照合して判定（案Bが正しい）
- [x] Cross-invariant dependency（I01↔I02↔I08 / I03↔I10 / I04↔I05 / I06↔I09 等）を明示
- [x] Authority conflicts を検証（0 件）
- [x] Production code 変更なし（mapping 定義のみ）
