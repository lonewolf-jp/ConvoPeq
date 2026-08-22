# Phase I-T1-C2 — Single Observation / Writer-Path Verification

> **Verdict: PASS**
> 目的: `releaseObserved_` の単一観測経路を証明する。R3後の静的因果分離に加え、全 production writer/caller を列挙し一意経路を確定。
> コード変更なし。C3以降・A_max・R gate は未決定のまま停止。

## 1. `releaseObserved_` の production writer 全列挙

| # | File | Line | 式 | メソッド |
|---|---|---|---|---|
| W1 | `ISRWorldRetirementTelemetry.h` | 90 | `fetchAddAtomic(releaseObserved_, 1, acq_rel)` | `onReleaseObserved()` |
| W2 | `ISRWorldRetirementTelemetry.h` | 98 | `fetchAddAtomic(releaseObserved_, count, acq_rel)` | `addReleaseObserved(count)` |

読み取り（参考）: `:108` getter / 定義 `:306`。**src/ 全域で他の writer は存在しない**（rg 全件ヒットが上記のみ）。

## 2. `onReleaseObserved()` の production caller 全列挙

| 調査手段 | 結果 |
|---|---|
| serena `find_referencing_symbols(onReleaseObserved)` | **参照 0件** `{}` |
| rg `onReleaseObserved src/` | 定義 `Telemetry.h:88` のみ（caller なし） |
| ag / awk / sg | 同様に caller なし |

**production caller = 0。** R3 で切断された `ReferenceObserver::onRelease()` からの呼び出しは復活していない。

> 注記: `ConvoPeq.md:58851` に `telemetry_->onReleaseObserved();` が残存するが、これは **R3 変更前のスナップショット**（2026-08-21 20:45:49 生成）である。現行 `src/` は R3 後の正であり caller は存在しない。スナップショット再生成時に行番号が更新される。

## 3. `addReleaseObserved()` の production caller 全列挙

| 調査手段 | 結果 |
|---|---|
| serena `find_referencing_symbols(addReleaseObserved)` | **参照 1件**: `AudioEngine::timerCallback` (`AudioEngine.Timer.cpp:428-429`) |
| rg / AiDex | `Timer.cpp:429`（caller）+ `Telemetry.h:94`（定義）の 2 hits のみ |

**destruction event が `releaseObserved_` に到達する実経路は sampler の `addReleaseObserved(delta)` のみ。**

## 4. `worldReclaimCount()` の caller 全列挙

| # | File | Line | 役割 |
|---|---|---|---|
| 1 | `AudioEngine.Timer.cpp` | 425 | **sampler による唯一の telemetry 喫流 caller**（`m_retireRouter->worldReclaimCount()`） |
| 2 | `ISRRetireRouter.cpp` | 400-407 | 集計本体（Deferred + Q + E + Terminal.reclaimCount） |
| 3 | `EpochDomain.h` | 410-412 | `IRetireProvider` override（deferredDeletionQueue へ委譲） |
| 4 | `IRetireProvider.h` | 58 | interface 既定実装 |
| 5-7 | `DeferredDeletionQueue.h:246` / `RetireQuarantineStore.h:209` / `ISRRetireRouter.h:217` | — | getter 宣言 |

telemetry へ喫流する caller は **Timer.cpp:425 の 1箇所のみ**。他は getter/集計/委譲であり `releaseObserved_` に寄与しない。

## 5. `referenceReleaseCount_` と `releaseObserved_` の semantic 分離

| カウンタ | writer | reader | 用途 |
|---|---|---|---|
| `referenceReleaseCount_` | `Reference.h:41`（onRelease 内 fetchAdd のみ） | getter `:78-80`, `referenceOutstanding :87` | reference running max 専用 |
| `releaseObserved_` | `Telemetry.h:90/98` のみ | getter `:108` | T1 measurement（sampler 反映） |

JSON export も別フィールド（`AudioEngine.Commit.cpp:764` `"referenceReleaseCount"` と `:736` `"releaseObserved"`）。混同・合流なし。

## 6. `onRelease()` → telemetry 経路の非復活確認

```
void onRelease() noexcept {
    fetchAddAtomic(referenceReleaseCount_, 1, acq_rel);
    updateRunningMax();
}
```

- `Reference.h` 内 `telemetry_->` 呼び出し: **0件**（rg 空）
- `onRelease()` 本体が `releaseObserved` に触れる行: **0件**

## 7. PASS 条件の検証

| 条件 | 結果 |
|---|---|
| production writer = `onReleaseObserved()` + `addReleaseObserved()` の 2メソッドのみ | ✅（§1） |
| destruction event で実際に呼ばれるのは `addReleaseObserved(delta)` のみ | ✅（§2: onReleaseObserved caller 0 / §3: addReleaseObserved caller 1=sampler） |
| ΔworldReclaimCount = N ⇒ ΔreleaseObserved = N | ✅ 構造的に成立: 9 LP 各1破壊が storage counter を +1（R2 §2-3 で1:1証明済み）→ 集計 → sampler delta → `addReleaseObserved(delta)` が 1:1 加算。`lastSampledWorldReclaimCount_` monotonic により同一 delta の二重反映は構造上不可能 |
| `ReferenceObserver.onRelease()` ⇒ ΔreleaseObserved = 0 | ✅（§6） |

## 8. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | writer/caller 全列挙・census 一致 |
| MCP#1 | serena `find_referencing_symbols` | onReleaseObserved=0 refs / addReleaseObserved=1 ref（Timer.cpp:428）— シンボルレベルで確定 |
| MCP#2 | ccc | search 実行（本リポ未 init のため rg/semble 補完） |
| CLI#1 | graphify 0.9.48 | exe 存在確認（graphify-out 未生成のため補完検索） |
| CLI#2 | semble 0.5.5 | `sampler addReleaseObserved worldReclaimCount delta` 検索実行 |
| MCP#3 | AiDex | `addReleaseObserved` 2 hits（caller+定義）/ `releaseObserved` 17 hits と一致 |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9件 200 OK（Vyukov SSL失効→rigtorp代替明記） |

## 9. 判定

```
T1-C2 = PASS
```

- 単一観測経路 `destruction → worldReclaimCount(+N) → sampler delta(N) → addReleaseObserved(N) → releaseObserved(+N)` のみが実在し、observer 経路は `releaseObserved` に寄与しない。
- R1 counterexample（+2）の構造的再発は不可能。

**次 Gate（未着手・停止中）:** T1-C3 window/sampler 整合性 → C4 RT safety → C5 export 15フィールド → C6 test coverage → C7/C8。A_max・R gate は C1〜C6 完了まで未決定。

---

*Evidence generated: Phase I-T1-C2 — no code change. C3 以降へは進まない。*
