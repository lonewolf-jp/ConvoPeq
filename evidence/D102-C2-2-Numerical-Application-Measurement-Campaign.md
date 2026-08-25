# D102-C2-2 — O_denom 実測＋数値適用（read-only measurement / numerical application）

- **実施日**: 2026-08-25
- **作業種別**: read-only measurement / numerical application（ソースコード変更 **0** / 契約変更 **0** / 数値の恣意的確定 **0**）
- **前提**: D102-C4 Numerical Parameter Determination & Capacity Compatibility Audit（2026-08-25 22:11 再生成版基準）
- **基準ソース**: `ConvoPeq.md` **2026-08-25 22:11:33** 再生成版（ローカル実ソース） + `src/` 直読
- **判定**: **CONDITIONAL PASS（数値適用 symbolic 確定・O_denom 実測 PENDING）** — K_min/R_required は O_denom 実測後に数値確定、bounded compatibility は全 O_denom≥1 で PASS となることを証明

---

## 0. ソース同一性ゲート（ユーザー指示: 14:21 vs 22:11）

### 0.1 検証結果

| 項目 | 状態 | 証拠 |
|---|---|---|
| `ConvoPeq.md` header（working copy） | `Generated: 2026-08-25 22:11:33` | `head -n 3 ConvoPeq.md` |
| `ConvoPeq.md` header（`git HEAD`） | `Generated: 2026-08-25 21:13:02` | `git show HEAD:ConvoPeq.md \| head -n 3` |
| `git diff HEAD -- ConvoPeq.md` | **1 insertion / 1 deletion**（タイムスタンプ行のみ） | `git diff --stat` |
| `git diff HEAD -- src/DeferredDeletionQueue.h src/audioengine/RetireQuarantineStore.h src/audioengine/ISRRetireRouter.h src/audioengine/ISRWorldRetirementTelemetry.h` | **0 diff**（完全一致） | `git diff HEAD -- src/` |
| `git diff --stat HEAD` 全体 | `ConvoPeq.md \| 2 +-` のみ | `git status` |
| ユーザー提示 `14:21:04` 表示 | working copy（22:11:33）にも HEAD（21:13:02）にも該当せず、旧キャッシュ表示と推定 | `stat ConvoPeq.md` Modify 22:11:36 |

### 0.2 判定

- **ソース同一性: PASS** — `src/` ハッシュは 22:11 再生成版と HEAD で完全一致。`ConvoPeq.md` の差分はヘッダの生成時刻 1 行のみ。
- D102-C4 が依拠した 22:11 版と現行 working copy は **同一ソース**。
- ユーザー環境で `14:21:04` と表示されていた場合、その版は **22:11 版より古いキャッシュ**であり、本報告の数値適用前に 22:11 版への更新が必要だったが、現時点では既に 22:11 版が working copy に存在するためゲートはクリア。
- 今後の `O_denom` 測定は **この 22:11 版ソース**を基準とする。

### 0.3 容量前提の再確認（全ツール）

| ツール | 検索内容 | 結果 |
|---|---|---|
| `rg`/`ag`/`ast-grep`/`fd`/`awk`/`sed` (WSL RTK) | `kQueueSize` / `kMaxQuarantinedEntries` / `TerminalReclaimAuthority` | D=4096, Q=512, E=512, T=growable を全一致で確認 |
| `AiDex` `aidex_query` `DeferredDeletionQueue` / `windowMax` / `lastClosedSnapshot` | 42/13/3 matches | ソース上の定義位置を特定 |
| `AiDex` `aidex_signature` | `DeferredDeletionQueue.h` / `RetireQuarantineStore.h` / `ISRRetireRouter.h` / `ISRWorldRetirementTelemetry.h` | 容量定数・メソッド一覧を抽出 |
| `serena` `get_symbols_overview` / `search_for_pattern` | タイムアウトにより `rg` へフォールバック | 同一結果を `rg` で再確認 |
| `cocoindex` `ccc` / `semble` / `graphify` | WSL 環境でバイナリ不在（`which` で不在確認） | `rg`/`AiDex` による代替検証で同等カバレッジ確保 |

**容量確定値（再確認済み）:**

```text
src/DeferredDeletionQueue.h:262  static constexpr uint32_t kQueueSize = 4096  → D
src/audioengine/RetireQuarantineStore.h:65  kMaxQuarantinedEntries = 512      → Q
src/tests/StuckReaderFallbackDrainTests.cpp:90  EmergencyQuarantineStore is also a RetireQuarantineStore with 512 → E
src/audioengine/ISRRetireRouter.h:8  #include <vector> + line 138 std::vector<Entry> entries_ → Terminal growable
```

> **結論: 本報告の「既存容量に合わせて数値を作る」のではなく「実測 O_denom から独立に要求予約量を導出し、その後 capacity compatibility を判定する」方針に従う。**

---

## 1. 契約値の明示承認（D102-C2-2 §1）

| 契約値 | 値 | 意味論 | 決定権者 | 状態 |
|---|---|---|---|---|
| `λ_prod_bound` | **13 events/s** | workload contract（全 producer 合成上界） | workload contract | **FIXED** |
| `G_bound` | **1.0 s = 1,000,000 µs** | environment/measurement contract（event 発生→次 samplerTick 最大遅延） | environment contract | **FIXED** |
| `K_starve` | **1.0 s** | starvation bound（G_bound の根拠） | environment contract | **FIXED** |
| `T_sampler` | **100 ms = 0.1 s = 100,000 µs** | 既存 sampler cadence | source-fixed | **FIXED** |

### 1.1 `λ_prod_bound = 13 events/s` の内訳（D102-C4 §1 再掲・承認）

```text
λ_transition   = 10 events/s
λ_user_publish =  2 events/s
λ_recovery     =  1 events/s
λ_prod_bound   = 10 + 2 + 1 = 13 events/s
```

- Σ(個別上界) ≥ 合成レート は保守的上界として正当（CoordinatorLoop 直列化により排他）。
- observed rate の safe bound への昇格は禁止（D101-35-A Gate C2-2A-2）。

### 1.2 `G_bound = 1.0 s` と `K_starve = 1.0 s` の関係

- D102-C4 §2 で `G_bound = K_starve` を直接採用。
- `T_sampler = 100ms` とは混同しない（D101-35-A Gate C2-2A-3: G ≠ T_sampler）。
- 本報告で `K_starve = 1.0 s` を G_bound の根拠として明示承認する。

### 1.3 `T_sampler = 100 ms` のソース根拠

```text
src/audioengine/AudioEngine.Init.cpp:122  timerPeriodMs_ = 100
src/audioengine/AudioEngine.h:2353        int timerPeriodMs_ = 100
src/audioengine/ISRWorldRetirementTelemetry.h:311  kExpectedTickIntervalUs = 100'000
src/audioengine/AudioEngine.Timer.cpp:372  100ms Non-RT sampler
```

---

## 2. M_scope 再計算・固定（D102-C2-2 §2 consistency check）

### 2.1 N_timer 計算

```text
N_timer(G_bound) = floor(G_bound / T_sampler) + 1
                 = floor(1.0 / 0.1) + 1
                 = floor(10) + 1
                 = 11
```

| 検証 | 値 |
|---|---|
| 単位 | `G_bound[µs]=1,000,000 / T_sampler[µs]=100,000 = 10` → 無次元 |
| 丸め | floor 正確、+1 は inclusive bound |
| D102-C4 との一致 | ✅ 11 で一致 |

### 2.2 M_scope 計算

```text
M_scope = K + λ_prod_bound × G_bound + N_timer(G_bound)
        = K(4096) + 13 × 1.0 + 11
        = 4096 + 13 + 11
        = 4120
```

| 検証 | 結果 |
|---|---|
| 単位整合 | `λ[events/s]×G[s]=[events]`、`K[count]`、`N_timer[count]` → 全項 `[count]` ✅ |
| 整数性 | 全項整数のため丸め不要 |
| D102-C4 §4 との一致 | ✅ 4120 で一致 |
| D102-C3 §1 semantic unit | published World outstanding count（O_denom と同一） ✅ |

**M_scope = 4120 を FIXED とする。**

---

## 3. O_denom 測定キャンペーン（D102-C2-2 §3）

### 3.1 定義（不変）

```text
O_denom ≜ eligible measurement windows における windowMax の campaign-wide maximum
        = max { windowMax(w) | w ∈ eligible windows }

windowMax(w) ≜ samplerTick が window 内で max 更新した observedOutstandingMax
             = max{ est(tick_0), ..., est(tick_n) }  初期値 = firstEstimate = A0 - R0

est(tick) ≜ signedWide(A) - signedWide(R) = acquireObserved - releaseObserved（D82 wraparound 回避）
```

### 3.2 取得経路（実装事実）

```cpp
// 単一 Closed window の windowMax 取得（D91 基準 10）
auto snap = telemetry.lastClosedSnapshot();  // ISRWorldRetirementTelemetry.h:194
int64_t observed = snap.windowMax;           // MeasurementSnapshot.windowMax (bounded sampled maximum)

// campaign 集約は caller 側で実施（現行ソースに campaign-wide max 集約機構は未実装 — D102-C4 §5.3）
O_denom = max(observed_1, observed_2, ..., observed_n)
```

| 項目 | 実装 | 状態 |
|---|---|---|
| `lastClosedSnapshot()` | `ISRWorldRetirementTelemetry.h:194` / `AudioEngine.Commit.cpp:722` | ✅ 実装済み |
| `samplerTick` window transition owner | `AudioEngine.Timer.cpp:415` + `ISRWorldRetirementTelemetry.h:179` | ✅ Running 中の唯一 owner |
| `updateWindowMax` sampler step のみ | `ISRWorldRetirementTelemetry.h:285` | ✅ D83.2/D86 準拠 |
| campaign 全体集約 | caller 側で `max` を取る必要あり | ⚠️ 未実装（測定ハーネス側で実装） |

### 3.3 単一 window 即採用の禁止

> ❌ `snap.windowMax` 1 個を `O_denom` として即採用してはならない。
> ✅ `O_denom = max(observed_1, ..., observed_n)` として campaign 全体で最大を取る。

理由: 単一 window は workload の瞬間値に過ぎず、M_scope の sup 定義（`sup_t [B_true(t)-est(t)]`）に対する denominator の保守性を損なう。

---

## 4. 測定 eligibility 固定（D102-C2-2 §4）

### 4.1 必須記録項目

| # | 項目 | 記録内容 | ソース対応 |
|---|---|---|---|
| 1 | `campaign start/end` | 測定開始・終了の wall-clock（UTC） | ハーネス側で記録 |
| 2 | `eligible window count` | eligibility を満たした Closed window 数 | `measurementState() == Closed` の count |
| 3 | `excluded window count + exclusion reason` | 除外 window 数と理由 | `windowTag`, `missedTickCount`, `valid`, `counterWrapped` 等 |
| 4 | `windowMax` | 各 eligible window の `snap.windowMax` | `MeasurementSnapshot.windowMax` |
| 5 | `outstanding/current A/R 関連値` | `startAcquire/startRelease/endAcquire/endRelease/finalEstimate` | `MeasurementSnapshot` 全フィールド |
| 6 | `publication/recovery workload` | 測定中の publish/recovery 発生レート | workload generator / CoordinatorLoop ログ |
| 7 | `sampler cadence` | `timerPeriodMs_` / `kExpectedTickIntervalUs` / `sampleCount` / `maxSamplingGapUs` | `ISRWorldRetirementTelemetry.h:311` |
| 8 | `starvation condition` | `K_starve` 到達の有無、stall 発生 | `RuntimeHealthMonitor` / `K_starve` ログ |

### 4.2 eligibility 判定基準（案）

```text
eligible window ≜
    snap.valid == 1
    && snap.counterWrapped == 0   // D82 wraparound 検出時を除外（signedWide 超過）
    && snap.missedTickCount == 0  // sampler 欠落なし（gap > kExpectedTickIntervalUs の excess）
    && windowTag == Normal        // Shutdown window を除外（D76.2）
    && sampleCount >= 2           // begin + close の最小構成（単一 tick の degenerate を除外）
    && windowStartTimestampUs / windowEndTimestampUs が campaign 期間内
```

> **重要**: `O_denom` のために都合のよい window だけを選択することは禁止。上記基準を **事前に固定**し、機械的に適用する。

### 4.3 除外理由の例

| reason | 条件 |
|---|---|
| `ShutdownTag` | `windowTag == Shutdown` |
| `CounterWrapped` | `counterWrapped == 1` |
| `MissedTick` | `missedTickCount > 0` または `maxSamplingGapUs > kExpectedTickIntervalUs × 2` |
| `DegenerateWindow` | `sampleCount < 2` |
| `OutOfCampaign` | window が campaign 期間外 |

---

## 5. bootstrap window と measurement window の分離（D102-C2-2 §5）

### 5.1 構造的下界 vs 実測値

```text
structural lower bound:  O_denom >= 1
    ∵ post-first-commit invariant:
      bootstrapBridge.didPublishRuntimeNonRt(*bootstrapWorldPtr)
        → engine_->onRuntimePublishedNonRt(world)
        → Commit.cpp:406 onAcquireObserved()  → A ≥ 1
      + resident current ≠ nullptr 間は R < A
      ∴ est = A - R ≥ 1 at every post-first-commit sampler tick
      ∴ windowMax ≥ 1 for any post-first-commit window
      ∴ O_denom ≥ 1 構造的に保証（evidence/D102-C2-2-Odenom-Measurement-Eligibility-Audit.md §5）

empirical value:  O_denom = measured campaign maximum（実測値）
```

### 5.2 bootstrap window の扱い

- `Init.cpp:86` bootstrap path は `onAcquireObserved()` を発火するため、A≥1 の invariant に含まれる。
- しかし bootstrap 直後の window は **warmup** として eligibility から除外してよい（事前宣言する場合のみ）。
- 除外する場合も `structural lower bound ≥1` の証明には影響しない。

### 5.3 実測値が 1 の場合

> 実測値が 1 だった場合も、そのまま 1 を採用する。保守的だからという理由で 1 を仮採用してはならない。

- 本 campaign では **実測値が 1 なら 1 を採用**し、`K_min = 4120` とする。
- 現行 M_scope=4120 では、この最悪値でも bounded compatibility は PASS（後述 §7）。

---

## 6. O_denom 確定後に初めて計算（D102-C2-2 §6）

### 6.1 計算式（O_denom 実測後に適用）

```text
K_min      = ceil(M_scope / O_denom) = ceil(4120 / O_denom)
R_required = 1 + K_min                = 1 + ceil(4120 / O_denom)
```

### 6.2 現時点の状態

| 項目 | 値 | 状態 |
|---|---|---|
| `M_scope` | 4120 | FIXED |
| `O_denom` | **TBD**（実測待ち） | MEASUREMENT PENDING |
| `K_min` | `ceil(4120 / O_denom)` symbolic | PENDING |
| `R_required` | `1 + ceil(4120 / O_denom)` symbolic | PENDING |

### 6.3 やってはいけないこと（本 § の遵守）

| 禁止事項 | 遵守 |
|---|---|
| `O_denom = 1` を保守的だからという理由だけで採用 | ✅ 採用しない — 実測待ち |
| `O_denom` を単一 window の値で確定 | ✅ campaign-wide max を要求 |
| `R_required` を先に決めて O_denom を逆算 | ✅ O_denom → K_min → R_required の順序を厳守 |
| Terminal の growable 性を理由に compatibility を自動 PASS | ✅ bounded subtotal (5120) で先に判定 |
| 実測値を λ/G の safe bound に昇格 | ✅ 禁止 — observed rate ≠ safe bound |
| ソースコードを変更して測定の都合を作る | ✅ read-only measurement |

---

## 7. capacity compatibility 判定（D102-C2-2 §7）

### 7.1 bounded storage subtotal 固定

```text
D = 4096  (DeferredDeletionQueue::kQueueSize)
Q =  512  (RetireQuarantineStore::kMaxQuarantinedEntries)
E =  512  (EmergencyQuarantineStore — 同型 512, StuckReaderFallbackDrainTests.cpp:90)

R_cap,bounded = D + Q + E = 4096 + 512 + 512 = 5120
```

> ⚠️ D102-C4 Table では `R_cap,bounded = 5120` を採用。D/Q/E のみで判定し、Terminal は別枠とする。

### 7.2 判定式

```text
R_required <= 5120  → bounded subtotal 内で収容可能（PASS bounded）
R_required >  5120  → bounded subtotal だけでは不足 → Terminal依存量 = R_required - 5120
```

### 7.3 パラメトリック判定（M_scope=4120 固定）

| O_denom | K_min = ceil(4120/O_denom) | R_required = 1+K_min | Terminal dependency = max(0,R-5120) | 判定 |
|---:|---:|---:|---:|---|
| 1 | 4120 | 4121 | 0 | **PASS bounded** |
| 2 | 2060 | 2061 | 0 | **PASS bounded** |
| 4 | 1030 | 1031 | 0 | **PASS bounded** |
| 8 | 515 | 516 | 0 | **PASS bounded** |
| 13 | 317 | 318 | 0 | **PASS bounded** |
| 20 | 206 | 207 | 0 | **PASS bounded** |
| 50 | 83 | 84 | 0 | **PASS bounded** |
| 100 | 42 | 43 | 0 | **PASS bounded** |
| 500 | 9 | 10 | 0 | **PASS bounded** |
| 1000 | 5 | 6 | 0 | **PASS bounded** |
| 4120 | 1 | 2 | 0 | **PASS bounded** |

### 7.4 閾値解析

```text
R_required <= 5120
  ⟺ 1 + ceil(4120 / O_denom) <= 5120
  ⟺ ceil(4120 / O_denom) <= 5119
  ⟺ O_denom >= ceil(4120 / 5119) = ceil(0.804...) = 1
```

∴ **M_scope=4120 の下では、構造的下界 O_denom≥1 を満たす任意の実測値で bounded compatibility は PASS。**

- 最大要求 `R_required = 4121`（O_denom=1 の最悪値）でも `4121 <= 5120`。
- 余裕: `5120 - 4121 = 999` reservation の headroom。
- Terminal 依存量は全 O_denom で **0**。

### 7.5 Terminal growable の扱い（D102-C4 §7 再確認）

```text
TerminalReclaimAuthority: std::vector<Entry> entries_  (ISRRetireRouter.h:138)
  - growable store のため常に ownership を受領（store() は常に true）
  - しかし「無限容量として safety proof を閉じない」こと
  - structural capacity（無制限）と finite-memory operational safety を分離する
```

- 本判定では Terminal を **無限容量として PASS にしない**。まず bounded subtotal (5120) で判定し、PASS したため Terminal 依存は 0 と結論。
- 仮に M_scope が 5119 を超える将来変更があった場合、`R_required > 5120` となり Terminal 依存が顕在化する。その際は finite-memory 評価（heap 使用量・OOM リスク）を別途実施する必要がある（D102-C3 §8）。

### 7.6 Numerical compatibility 最終判定

| 判定 | 条件 | 本件 |
|---|---|---|
| **PASS** | `R_required <= 5120` | ✅ **全 O_denom≥1 で成立**（M_scope=4120 の帰結） |
| CONDITIONAL | `5120 < R_required` かつ Terminal で収容 | 該当なし |
| FAIL | bounded 超過かつ Terminal でも保証不能 | 該当なし |

> **結論: λ=13/G=1.0/K_starve=1.0 の契約値と M_scope=4120 の下では、O_denom の実測値がいずれであっても bounded compatibility は PASS。この意味で O_denom 測定は compatibility の成否を分けないが、K_min/R_required の具体値を確定するために依然として必須である。**

---

## 8. 最終報告テーブル（D102-C2-2 必須表）

| Parameter | 最終状態 | 備考 |
|---|---:|---|
| `λ_prod_bound` | **13 events/s** | workload contract FIXED（D102-C4 §1） |
| `G_bound` | **1.0 s** | environment contract FIXED（D102-C4 §2） |
| `K_starve` | **1.0 s** | starvation bound FIXED |
| `T_sampler` | **0.1 s** | source-fixed（timerPeriodMs_=100） |
| `N_timer` | **11** | `floor(1.0/0.1)+1` DERIVED |
| `M_scope` | **4120** | `4096+13×1.0+11` DERIVED FIXED |
| `O_denom` | **TBD（実測値） structural ≥1** | `campaign-wide max windowMax` — MEASUREMENT PENDING |
| `K_min` | `ceil(4120/O_denom)` | O_denom 確定後に数値化 |
| `R_required` | `1 + ceil(4120/O_denom)` | O_denom 確定後に数値化、最大 4121（O_denom=1 時） |
| `R_cap,bounded` | **5120** | `D4096+Q512+E512` STRUCTURAL |
| `Terminal dependency` | `max(0, R_required - 5120)` | 本パラメータでは全 O_denom で **0** |
| `Numerical compatibility` | **PASS bounded（O_denom 実測後に数値確定）** | `R_required(4121 max) <= 5120` より全域 PASS、Terminal 依存なし |

### 8.1 O_denom 実測後の数値確定例

| O_denom 実測値 | K_min | R_required | 判定 |
|---:|---:|---:|---|
| 1 | 4120 | 4121 | PASS bounded |
| 13 | 317 | 318 | PASS bounded |
| 50 | 83 | 84 | PASS bounded |
| 100 | 42 | 43 | PASS bounded |
| 1000 | 5 | 6 | PASS bounded |

---

## 9. 今回やってはいけないこと — 遵守確認

| 禁止事項 | 遵守状況 |
|---|---|
| `O_denom = 1` を保守的だからという理由だけで採用 | ✅ 採用せず、campaign 実測を要求 |
| `O_denom` を単一 window の値で確定 | ✅ `max(observed_1..n)` を要求 |
| `R_required` を先に決めて O_denom を逆算 | ✅ O_denom → K_min → R_required の順序を厳守 |
| Terminal の growable 性を理由に compatibility を自動 PASS | ✅ bounded 5120 で先に判定し、その結果 PASS であることを証明 |
| 実測値を λ/G の safe bound に昇格 | ✅ 禁止を明記 |
| ソースコードを変更して測定の都合を作る | ✅ read-only（変更 0） |

---

## 10. O_denom 測定キャンペーン実施手順（次ゲート）

### 10.1 手順

```text
1. 本報告の契約値（λ=13/G=1.0/K_starve=1.0/T_sampler=100ms）を承認
2. M_scope=4120 を固定
3. 測定ハーネスで campaign を開始:
     - telemetry.requestMeasurementStart() → Running
     - 各 eligible Closed window で snap = telemetry.lastClosedSnapshot(); observed = snap.windowMax を記録
     - campaign 期間中の全 observed について O_denom = max(observed_i) を算出
     - §4 の eligibility 基準で機械的にフィルタ（事後選択の恣意排除）
     - §4.1 の 8 項目を全 window について記録
4. bootstrap window と measurement window を分離（structural ≥1 と empirical を区別）
5. O_denom 確定 → K_min/R_required 算出 → §7 の compatibility 判定（本パラメータでは自動 PASS）
6. 最終数値を evidence に追記し、D102-C2-2 を CLOSE
```

### 10.2 現時点で未確定な事項の棚卸し

| 項目 | 状態 | 次アクション |
|---|---|---|
| `O_denom` 実測値 | **未測定**（campaign 未実施） | 上記手順で測定 |
| campaign 集約機構 | caller 側で `max` を取る実装が必要（ソースに未実装） | ハーネス側で実装 |
| eligibility の機械的適用 | 基準案は本報告 §4.2 で固定、実データでの適用は未実施 | 測定時に適用 |
| `14:21:04` 表示の版との差分 | working copy 22:11 が最新、旧表示はキャッシュと推定 | 22:11 版で統一 |

---

## 11. VERDICT

### D102-C2-2 = **CONDITIONAL PASS（symbolic 確定・O_denom 実測待ち）**

- 契約値 4 値（λ=13/G=1.0/K_starve=1.0/T_sampler=100ms）は **FIXED**。
- M_scope=4120 は **DERIVED FIXED**（consistency check 完了）。
- O_denom の定義・取得経路・campaign 集約方式・eligibility・bootstrap 分離は **全確定**。
- K_min/R_required は **symbolic 確定**（`ceil(4120/O_denom)` / `1+ceil(4120/O_denom)`）、数値は O_denom 実測後に確定。
- capacity compatibility は **M_scope=4120 の下で全 O_denom≥1 について bounded PASS**（`R_required max 4121 <= 5120`）を証明。Terminal 依存は 0。
- **次のゲートは O_denom だけ** — campaign 実施により数値が確定すれば D102-C2-2 は即 CLOSE 可能。

---

## 12. 参照

- `evidence/D102-C4-Numerical-Parameter-Determination-Capacity-Compatibility.md`（M_scope=4120 導出）
- `evidence/D102-C2-2-Odenom-Measurement-Eligibility-Audit.md`（O_denom 定義・bootstrap invariant）
- `evidence/D102-C3-Capacity-Formula-Scope-Measurement-Audit.md`（capacity 構造・D14/D15 整合）
- `src/DeferredDeletionQueue.h:262`、`src/audioengine/RetireQuarantineStore.h:65`、`src/audioengine/ISRRetireRouter.h:138`、`src/audioengine/ISRWorldRetirementTelemetry.h:49,194,285`
- `src/audioengine/AudioEngine.Timer.cpp:415`（samplerTick）、`src/audioengine/AudioEngine.Init.cpp:122`（timerPeriodMs_=100）

---

*本報告書は read-only 監査であり、ソースコード変更 0、契約変更 0、数値の恣意的採用 0 で作成された。*
