# Phase I-T1-C3 — Window / Sampler Integrity Verification

> **Verdict: PASS**
> 目的: C2 で確定した唯一観測経路（destruction → worldReclaimCount → Timer sampler → addReleaseObserved(delta) → releaseObserved_）が、window/sample 境界・複数 destruction・lifecycle 境界を跨いでも 1:1 で保存されることを証明する。
> コード変更なし。C4 以降・A_max・K_terminal・R gate・Recovery supersession は未着手のまま停止。

## 1. C3-1 — sampler state の全 writer/caller 列挙

### 1.1 `lastSampledWorldReclaimCount_`

| # | File | Line | 操作 | 調査手段 |
|---|---|---|---|---|
| D | `AudioEngine.h` | 2233 | 定義 `std::atomic<uint64_t> {0}`（初期値 0） | rg / AiDex |
| R | `AudioEngine.Timer.cpp` | 426 | read `consumeAtomic(acquire)` | rg / AiDex / ag |
| W | `AudioEngine.Timer.cpp` | 430 | **write `publishAtomic(release)` — 唯一の writer** | rg / AiDex / awk census |

**writer は sampler 内部（timerCallback）の 1箇所のみ。** serena 参照検索・awk writer census（count=1）も一致。

### 1.2 関連 API の caller（C2 再確認）

- `worldReclaimCount()` telemetry 喫流 caller: `Timer.cpp:425` のみ
- `addReleaseObserved(delta)` caller: `Timer.cpp:429` のみ

### 1.3 sampler 実行コンテキスト

| 経路 | File | 内容 |
|---|---|---|
| **production** | `AudioEngine::timerCallback` (`Timer.cpp:371`, 100ms JUCE Timer・MessageThread Non-RT) | delta transfer block (:423-431) + `telemetry.samplerTick(windowNowUs)` (:442) + reference window 同期 |
| test-only | `driveWorldRetirementSamplerForMeasurement` (`AudioEngine.h:4946-4954`) | `samplerTick()` + reference window 同期のみ（**delta block を含まない**） |

> 観察記録（C3 判定に影響なし）: テスト用ハーネス経路は delta transfer を実行しないため、ヘッドレス実測では `releaseObserved` が storage counter と同期しない可能性がある。production 正当性には無関係だが **T1-C6/C8 で評価すべき項目**として記録する。

### 1.4 `releaseObserved()` の reader 全列挙（読み取り専用確認）

| File | Line | 用途 |
|---|---|---|
| `AudioEngine.Commit.cpp` | 736 | JSON export |
| `ISRWorldRetirementTelemetry.h` | 106 | getter |
| 同 | 116 | `observedOutstandingEstimate` |
| 同 | 211 / 231 / 249 | `beginWindow` / `samplerTick` / `endWindow` の baseline 読み取り |

全て read-only。reader からの書き込みはない。

## 2. C3-2 — delta 算出の完全な因果（10ケース）

対象コード（`Timer.cpp:423-431`）:

```cpp
const uint64_t worldReclaimed = m_retireRouter ? m_retireRouter->worldReclaimCount() : 0;
const uint64_t prevReclaimed  = consumeAtomic(lastSampledWorldReclaimCount_, acquire);
if (worldReclaimed > prevReclaimed) {
    telemetry.addReleaseObserved(worldReclaimed - prevReclaimed);
    publishAtomic(lastSampledWorldReclaimCount_, worldReclaimed, release);
}
```

| # | ケース | 挙動 | 判定 |
|---|---|---|---|
| 1 | 初期値 | `lastSampled{0}` / 各 storage `worldReclaimCount_{0}` — 対称 | ✅ |
| 2 | 初回実行 | prev=0, current=N → delta=N（起動以来の全破壊を一度だけ反映） | ✅ |
| 3 | delta == 0 | guard `>` 不成立 → addReleaseObserved 呼ばない・lastSampled 不変 | ✅ |
| 4 | delta == 1 | +1 を 1回のみ加算 | ✅ |
| 5 | delta > 1 | `addReleaseObserved(delta)` が全件を一括加算（Telemetry.h:94-99, count==0 early-return） | ✅ |
| 6 | sampler 間に複数 destruction | storage counter が累積し次 tick で合算 delta として 1回転送 | ✅ |
| 7 | sampler 間に destruction なし | delta=0 → 何もしない | ✅ |
| 8 | sampler 複数回実行 | 各回「前回 publish 値以降の増分」のみ消費（publishAtomic が linearization） | ✅ |
| 9 | counter 減少 | storage counter は fetchAdd 専用で monotonic・減算 API なし。仮に減少しても guard により skip（underflow delta 不発生） | ✅ |
| 10 | wraparound / signedness | uint64 比較 guard により current < prev 時は安全 skip。D82.2 が測定期間中 wraparound 前提なしを明記、estimate は signedWide(int64 cast) | ✅ |

## 3. C3-3 — window 境界の二重計上検証

delta transfer block は **window 状態と独立して毎 tick 実行される累積転送**である。window 境界処理（`Telemetry.h:210-222` `beginWindow`）は:

- `a0 = acquireObserved()`, `r0 = releaseObserved()` を **baseline として読むだけ**（`startAcquire_`/`startRelease_` に publish）
- reset するのは per-window 統計（`maxSamplingGapUs_`, `missedTickCount_`）のみ
- 累積カウンタ（`acquireObserved_`/`releaseObserved_`/storage counters）は不変

検証例:

```
sample N:   worldReclaimCount=10, lastSampled=10
破壊×3
sample N+1: current=13 > prev=10 → addReleaseObserved(3), lastSampled=13
            ΔworldReclaimCount=3, ΔreleaseObserved=3 ✅
sample N+2: delta=0 → 変化なし ✅
```

境界ケース（destruction → sample → destruction → sample）でも各 delta が 1回ずつ消費され 1:1 保持。window Start/End を跨いでも二重計上なし。

## 4. C3-4 — `referenceReleaseCount_` との再混入確認

sampler block（`Timer.cpp:423-442`）が触れるのは `m_retireRouter->worldReclaimCount()` / `lastSampledWorldReclaimCount_` / telemetry のみ。reference observer のカウンタには一切アクセスしない。

5 API の分離（C2 §5 の再確認）:

- `onRelease()` → `referenceReleaseCount_` のみ（telemetry 呼び出し 0件）
- `onReleaseObserved()` → production caller 0件
- `addReleaseObserved()` → sampler のみ
- `referenceReleaseCount()` / `releaseObserved()` → reader のみ

別経路なし ✅

## 5. C3-5 — counter reset / lifecycle 境界

### 5.1 reset 存在調査（rg 全件）

| 対象 | `store(0)` / reset の hits |
|---|---|
| `worldReclaimCount_`（D/Q/E/T 全 storage） | **0件** |
| `releaseObserved_` | **0件** |
| `lastSampledWorldReclaimCount_` | **0件**（writer は delta 転送時の current 値 publish のみ） |

### 5.2 類似名カウンタの誤認チェック

| メソッド | 実際に reset するもの | 累積カウンタへの影響 |
|---|---|---|
| `DeferredDeletionQueue::clearMaxRetireAge()` (:240) | `maxRetireAgeUs_` | なし |
| `RetireQuarantineStore` drainAllUnsafe 内 `residentAtomic_.store(0)` (:171) | 滞留件数 | なし（`worldReclaimCount_` は increment 専用） |
| Telemetry `beginWindow` (:215-222) | `startAcquire_/startRelease_` baseline + `maxSamplingGapUs_/missedTickCount_` | なし（累積カウンタは読むだけ） |

### 5.3 lifecycle 対称性

constructor で全カウンタ `{0}` 初期化 → AudioEngine オブジェクト寿命と共に生存 → shutdown drain（dtor 内 `m_retireRouter->drainAll()` 等）は破壊を起こして counter を **increment** するだけで reset しない → engine 再構築時は新オブジェクトで全て `{0}` から開始（対称）。

**非対称 reset は存在しない → 偽 delta は構造上発生不能。** ✅

## 6. 判定基準（9条件）の検証

| # | 条件 | 判定 |
|---|---|---|
| 1 | worldReclaimCount の増加が destruction event と 1:1 | ✅（R2 §2-3: 9 LP 各1破壊=+1） |
| 2 | sampler が delta を 1回だけ消費 | ✅（publishAtomic が linearization point） |
| 3 | `lastSampledWorldReclaimCount_` の writer が一意 | ✅（Timer.cpp:430 のみ — rg/AiDex/serena/awk 一致） |
| 4 | sample window を跨いでも二重計上なし | ✅（§3: 転送は window 非依存の累積方式） |
| 5 | delta > 1 でも全件反映 | ✅（addReleaseObserved(count) 一括加算） |
| 6 | delta == 0 では増加なし | ✅（guard `>` による skip） |
| 7 | counter reset による偽 delta なし | ✅（§5: reset 0件・lifecycle 対称） |
| 8 | `referenceReleaseCount_` との semantic 混同なし | ✅（§4） |
| 9 | `releaseObserved_` への別 production 経路なし | ✅（C2 §1-3 再確認） |

## 7. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | lastSampled 3 hits・writer census=1・reset census=0 を複数手段で一致 |
| MCP#1 | serena `find_referencing_symbols(lastSampledWorldReclaimCount_)` | member 参照 {} （inline 使用のためシンボル参照なし — rg/AiDex で補完確定） |
| MCP#2 | ccc | search 実行（未 init のため rg/semble 補完） |
| CLI#1 | graphify 0.9.48 | exe 確認（graphify-out 未生成のため補完検索） |
| CLI#2 | semble 0.5.5 | `sampler delta transfer lastSampled window` 検索実行 |
| MCP#3 | AiDex | `lastSampledWorldReclaimCount_` 3 hits（定義+read+write）と一致 |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9件 200 OK（Vyukov SSL失効→rigtorp代替明記） |

## 8. 記録事項（C6/C8 への持ち越し・C3 判定外）

- テストハーネス `driveWorldRetirementSamplerForMeasurement`（AudioEngine.h:4946）は `samplerTick` + reference window 同期のみで **delta transfer block を含まない**。ヘッドレス実測で `releaseObserved` を storage 同期させるには timerCallback 相当の delta block 実行が必要 — T1-C6（テスト gap）/C8（measurement readiness）で評価すること。
- `ConvoPeq.md` スナップショット（2026-08-21 20:45:49）は R3 前の為、`src/` 現行を正として照合済み（差分は R3 の切断箇所のみ）。

## 9. 判定

```
T1-C3 = PASS
```

**次 Gate（未着手・停止中）:** T1-C4 RT safety → C5 export 15フィールド → C6 test coverage → C7/C8。A_max・K_terminal・R gate・Recovery supersession は C1〜C6 完了まで未決定。

---

*Evidence generated: Phase I-T1-C3 — no code change. C4 以降へは進まない。*
