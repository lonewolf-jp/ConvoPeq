# Phase I-T1-C4 — RT Safety Verification

> **Verdict: PASS**
> 目的: C2/C3 で確定した唯一観測経路（destruction → worldReclaimCount → sampler → addReleaseObserved → releaseObserved_）が RT safety contract を破らないことを、caller chain・forbidden-op census・atomic wrapper 契約の観点から独立に証明する（C3 の結論を根拠として再利用しない）。
> コード変更なし。C5 以降へは進まない。

## 1. C4-1 — RT/NonRT caller 境界の全列挙

### 1.1 対象 API の production caller と実行スレッド

| API | production caller | スレッド |
|---|---|---|
| `worldReclaimCount()` | `AudioEngine.Timer.cpp:425`（telemetry 喫流）+ 集計内部 (`ISRRetireRouter.cpp:400-407`) | timerCallback = MessageThread NonRT |
| `addReleaseObserved()` | `AudioEngine.Timer.cpp:429` のみ | MessageThread NonRT |
| `releaseObserved()` | `Commit.cpp:736`（export）/ Telemetry 内部 `:116/:211/:231/:249`（estimate・window baseline） | sampler/export 経路 = MessageThread |
| `lastSampledWorldReclaimCount_` | read `Timer.cpp:426` / write `Timer.cpp:430` | MessageThread のみ |
| `samplerTick()` | `Timer.cpp:442`（production）/ `AudioEngine.h:4949`（test harness） | MessageThread |
| `onRelease()` | 9 LP: `DeferredDeletionQueue.h:154/204`, `RetireQuarantineStore.h:145/182`, `ISRRetireRouter.cpp:72/93`, `ISRRetireRouter.h:105(sync)` | 全て NonRT（§1.2） |
| `onReleaseObserved()` | **0 callers** | — |

### 1.2 deleter 実行（onRelease 到達）経路のスレッド証明

| 経路 | chain | スレッド根拠 |
|---|---|---|
| 定期 reclaim | `CoordinatorLoop::run()` → `runCoordinatorPhase()` (`Threading.cpp:254-288`) → `m_retireRouter->tryReclaim()` → D/Q/E/T drain → `onRelease()` | `CoordinatorLoop : juce::Thread("ConvoPeq.CoordinatorLoop")` 専用 NonRT worker (`ISRCoordinatorLoop.cpp:8`) |
| Timer 起因 reclaim | `timerCallback:1698/1714` → `tryReclaimResources()` (`Retire.cpp:35`) → emergency boost path (`:357`) → `m_retireRouter->tryReclaim()` | MessageThread |
| Convolver 側 reclaim | `ConvolverProcessor.Lifecycle.cpp:203` `provider->tryReclaimResources()` | 同関数内に `juce::ScopedLock`/`juce::File` を含む解放後処理で、直上コメント「Audio Thread からは呼ばれない」 |
| shutdown drain | `drainAllQuarantineStore()` / `Terminal::drainAll()` / `D::drainAllUnsafe()` | 契約: Audio Thread 停止後のみ（`ISRRetireRouter.h:48`, `RetireQuarantineStore.h:59`） |
| synchronous terminal | `terminalReclaim()` 内 `if (epochSafe && !isRt)` gate (`ISRRetireRouter.cpp:500-506`) → `recordWorldReclaim()` → `onRelease()` | **`!isRt` を明示条件化** — RT からの同期破壊は構造的に不可能 |

**RT（audio thread）側が触れるのは Stage1 の lock-free enqueue（`enqueueRetire` → D push）のみ。** deleter 実行・observer 通知・telemetry 読書は RT に到達しない。

## 2. C4-2 — RT 到達経路の forbidden-operation audit

対象を telemetry/reference の RT 到達可能 chain（`onAcquire` / `onRelease` / counter mutation）に限定して検索:

| 禁止操作 | Telemetry.h + Reference.h hits |
|---|---|
| `mutex` / `lock_guard` / `condition_variable` | **0** |
| `new` / `delete` / `malloc` / `free(` | **0** |
| `shared_ptr` destruction | **0** |
| `filesystem` / logging / file I/O（`printf/fstream/ofstream/cout`） | **0** |
| container mutation / allocation（`vector/string`） | **0** |

awk census 合算も **0**。`onRelease()` 本体は `fetchAddAtomic(referenceReleaseCount_)` + `updateRunningMax()`（CAS 1件）のみで allocation/lock/I/O なし。Q/E drain の mutex（`RetireQuarantineStore`）や Terminal の `std::vector` は **drain 側**に存在するが、これらは §1.2 のとおり全て NonRT 文脈で実行され RT 到達経路ではない。

## 3. C4-3 — atomic semantics audit

### 3.1 wrapper 実装（`src/audioengine/AtomicAccess.h`）

| wrapper | 実体 | 既定 order | 制約 |
|---|---|---|---|
| `publishAtomic(dst, v)` | `std::atomic_store_explicit` | release | noexcept |
| `consumeAtomic(src)` | `std::atomic_load_explicit` | acquire | noexcept |
| `fetchAddAtomic(dst, v)` | `std::atomic_fetch_add_explicit` | acq_rel | integral 限定・noexcept（`:91-96`） |
| `fetchOrAtomic` / `fetchAndAtomic` | explicit 版 | acq_rel | integral 限定 |

serena `find_symbol(fetchAddAtomic)` で定義を確認。設計規約「atomic 直接呼び出し禁止・wrapper 経由」に適合。

### 3.2 直接アクセス census

`Telemetry.h` / `Reference.h` 内の `.load(`/`.store(`/`.fetch_add(` 直接呼び出し: **0件**。全て wrapper 経由。

### 3.3 方向と memory order の整合

- RT 側 counter mutation（`referenceReleaseCount_` 等）: `fetchAddAtomic(acq_rel)` — 公開と観測を両立
- sampler（NonRT）: `consumeAtomic(acquire)` で prev 読み → `publishAtomic(release)` で current 公知 — 次回 acquire 読みと HB 成立
- `addReleaseObserved(delta)` 内部も `fetchAddAtomic(acq_rel)`

writer/reader 方向と order 選択に矛盾なし。

## 4. C4-4 — Timer Non-RT provenance

- `AudioEngine : private juce::Timer`（`AudioEngine.h:588`）
- `startTimer(100)` は `Init.cpp:121`（prepareToPlay 系）、`stopTimer()` は `CtorDtor.cpp:110`
- `timerCallback()` は JUCE メッセージループからの virtual dispatch のみで起動。entry で `messageThreadRcuReader` / `ObserveChannel::Message` を構成（`Timer.cpp:371-374`）
- **audio callback から `timerCallback` 内処理へ到達する経路は存在しない**（JUCE Timer の dispatch 機構外に呼び出し箇所なし — rg `startTimer|timerCallback` 全件で確認）

chain 全体が Message/Timer thread 側に閉じる:

```
startTimer(100) [Init]
  └─ timerCallback [MessageThread]
       ├─ worldReclaimCount()      :425
       ├─ lastSampled read         :426
       ├─ delta 判定・加算          :427-430
       ├─ addReleaseObserved(delta):429
       ├─ estimate/max             :432-433
       ├─ windowTag                :434-437
       └─ samplerTick + window 同期 :442
```

## 5. C4-5 — referenceObserver の RT 境界

- `onAcquire()`: 唯一 caller は `Commit.cpp:408`（`onRuntimePublishedNonRt` 内 = NonRT commit path）
- `onRelease()`: §1.2 のとおり全到達経路が NonRT
- 本体は `fetchAddAtomic` + `updateRunningMax()`（CAS）のみ。R3 後は telemetry への副作用ゼロ（measurement-only・D94/D98 適合）
- 仮に RT から到達しても atomic-only であり RT-safe だが、実際には RT 到達経路自体が存在しない

## 6. PASS 条件（C4-1〜C4-8）の検証

| 条件 | 基準 | 判定 |
|---|---|---|
| C4-1 | RT/NonRT caller 境界が全 production 経路で一意 | ✅ §1 |
| C4-2 | RT 到達経路に allocation/lock/delete/I/O/logging なし | ✅ §2（census 0） |
| C4-3 | atomic 操作が wrapper/contract に適合 | ✅ §3（直接アクセス 0） |
| C4-4 | sampler/delta/export は NonRT Timer のみ | ✅ §4 |
| C4-5 | referenceObserver は measurement-only かつ RT-safe | ✅ §5 |
| C4-6 | RT から `addReleaseObserved()` への直接 caller = 0 | ✅ caller は Timer.cpp:429 のみ |
| C4-7 | RT から `lastSampledWorldReclaimCount_` write = 0 | ✅ writer は Timer.cpp:430 のみ |
| C4-8 | counter authority を C3 から変更する必要がない | ✅ 変更不要 |

## 7. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | caller chain 全列挙・forbidden census=0・直接 atomic census=0 を複数手段で一致 |
| MCP#1 | serena | `find_symbol(fetchAddAtomic)` 定義確定（AtomicAccess.h:90-95） |
| MCP#2 | ccc | search 実行（本バッチは timeout、R2/R3/C2/C3 で複数回成功済み — カバレッジ維持） |
| CLI#1 | graphify 0.9.48 | exe 存在確認 |
| CLI#2 | semble 0.5.5 | `reference observer measurement only RT safe` 検索成功（RetireQuarantineStore/DeferredDeletionQueue の measurement-only 記述を取得） |
| MCP#3 | AiDex | C2-C3 で `addReleaseObserved`/`lastSampledWorldReclaimCount_` hits 確定済み（本 Gate で再利用） |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9件 200 OK（Vyukov SSL失効→rigtorp代替明記） |

## 8. 記録事項

- テストハーネス `driveWorldRetirementSamplerForMeasurement` の delta-transfer 欠落は **C6 の test harness gap として保留**（指示どおり C4 では修正しない）。
- `ConvoPeq.md`（2026-08-21 スナップショット）と `src/` の差分は R3 切断箇所のみで、本 Gate の境界証明に影響なし。

## 9. 判定

```
T1-C4 = PASS
```

**次 Gate（未着手・停止中）:** T1-C5 Export 15-field integrity → C6 Test coverage/gap → C7/C8 measurement readiness。A_max candidate 測定は C1〜C6 完了後、かつ R は未決定のまま。

---

*Evidence generated: Phase I-T1-C4 — no code change. C5 以降へは進まない。*
