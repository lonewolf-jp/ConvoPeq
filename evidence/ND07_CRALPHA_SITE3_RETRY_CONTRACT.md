# ND-07 — CR-α Site 3 Retry Contract Finalization（実装前契約書・read-only）

```text
Date: 2026-09-01
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（--check FRESH / NEWER_SRC_COUNT=0 実測 — ND-06 と同一・開始時再確認済み）
前提: ND-06 = CONDITIONAL-GO（Site 3 のみ乖離 V-1/V-2/V-4 — evidence/ND06_BUILDERROR_RETRY_CONTRACT_AUDIT.md）
性格: 実装時の判断余地をなくす Site 3 専用 retry contract の確定。コード変更なし。
```

## 総合判定（先出し）

> ## **ND-07 = GO**
>
> 14 節の契約を以下に確定。**max retry = 3（Site 3 専用・recovery の K=4 とは別値）**、
> **backoff normative default = 10→20→40→80ms（`{10, 80, 2}`）を採用**、
> **WarmupFailed（RetryImmediate）は delay 0 + bounded 3 回**、**RetryBackoff は接続済みだが
> 現行の Site 3 failure class では dormant**（build failure retry は対象外 — 将来拡張として明示）。
> 判断余地の削除のため、retry decision を**純関数（BuildErrorPolicy.h 内）**として定義し、
> rebuildThreadLoop はその結果を実行するだけの構造とする。

---

## 1. Site 3 state owner

**所有者 = `rebuildThreadLoop`（RebuildThread 専用スレッド）の関数スコープ変数。**

- Site 3 の task は pendingTask（単一スロット・rebuildMutex 保護）から 1 iteration に 1 つ transfer される（実測: `task = pendingTask; pendingTask.currentDSP = nullptr;` — RebuildDispatch.cpp:875 付近）。retry は同一 generation の連続処理として複数 iteration に跨るため、counter は **task 単位でなく rebuildThreadLoop 関数スコープ**に置く（Site 2 の `recoveryConsecutiveFailures` 前例と同一パターン）。
- **単一所有者** = RebuildThread（rebuildThreadLoop は RebuildThread 専用・Site 2 前例どおり）。mutex / atomic 不要。
- **ScheduleRequest 側に持たせない**: `RetryScheduleRequest` は 4-field（kind / reason / rebuildClass / collapsePolicy）であり D101-24 で固定済み。変更禁止。
- **RetryScheduler に持たせない**: scheduler は time/ordering のみ所有（CtorDtor.cpp:97 コメント契約「RetryScheduler owns time/ordering only; BuildError/RetryDisposition stays caller-side」）。BuildError-aware 化禁止（BE-8）。
- generation / Replaceable collapse との対応: counter は **generation に紐付ける**（§6）。collapse（Replaceable latest-wins merge・:186-215）は intent 送信側の burst 吸収であり retry attempt 計数とは独立 — collapse で merge された intent は新しい attempt として数えない（同一 generation の retry は schedule 経由のみで発生し、collapse は新規 request に対してのみ作用するため構造的に干渉しない）。

## 2. Retry attempt semantics

- **consecutive failure の定義**: Site 3 warmup failure（`validateWarmup != None` かつ `shouldRetryWarmupFailure` == true かつ !obsolete）1 回 = attempt 1 回。
- **成功時の reset**: publish 経路に到達（warmup 成功）したら counter = 0。
- **generation 変更時の reset**: `task.generation != boundGeneration` なら counter = 0 にして boundGeneration を更新（§6）。
- **obsolete request 時の扱い**: obsolete なら schedule しない（retry 打ち切り・counter は次 generation 変更で無効化）。obsolete 判定は既存 `isObsolete()` ラムダ（generation 比較）を schedule 前に再確認する形で使用。
- **build failure（`runtime == nullptr`）は対象外**: 現行どおり task drop・retry しない（task 喪失意味論の見直しは将来拡張。CR-α では WarmupFailed retry の bound 化のみ）。
- **max retry = 3 を採用**（recovery の K=4 には揃えない）:
  - 根拠 1: dash2 Amendment 1.8 の normative default「configurable, default 3」。
  - 根拠 2: **recovery の K=4 と意図的に別値**にすることで、grep / telemetry 読解時に Site 3 counter と obligation counter の混同を構造的に検出可能にする（値の一致は混同を隠す）。
  - 定数名: `kMaxWarmupConsecutiveRetries = 3`（Site 3 専用であることを名前で明示）。

## 3. Maximum retry bound

```text
attempt 0 = 初回 warmup（retry ではない）
attempt 1..3 = warmup retry（schedule 済み再処理）
attempt 4 回目の warmup failure → exhausted（schedule しない・terminal telemetry）
```

- 上限超過後の同 generation warmup failure は schedule しない（telemetry 1 回のみ — §8）。
- `kMaxObligationConsecutiveFailures = 4`（obligation counter・Site 1/2）と `kMaxWarmupConsecutiveRetries = 3`（Site 3）は **別ドメイン・別値**。混同禁止（ND-06 §5 の context 分離を契約として明文化）。

## 4. RetryBackoffPolicy

**normative default を確定: 設計記録の 10→20→40→80ms を採用する。**

```cpp
// BuildErrorPolicy.h に追加（JUCE 非依存・standalone contract test 可能 — 同ヘッダの既存性格どおり）
struct RetryBackoffPolicy
{
    std::uint32_t initialDelayMs = 10;   // attempt 1 の delay
    std::uint32_t maxDelayMs     = 80;   // saturation 上限
    std::uint32_t multiplier     = 2;    // exponential 倍率
};
inline constexpr RetryBackoffPolicy kDefaultWarmupRetryBackoff { 10, 80, 2 };

// attempt (1-based) の delay。saturation: delay を掛けた結果が maxDelayMs を超えたら
// maxDelayMs に固定（overflow しない — 掛け算前に上限判定）。
[[nodiscard]] inline std::uint32_t retryBackoffDelayMs(
    const RetryBackoffPolicy& p, std::uint32_t attempt) noexcept
{
    if (attempt == 0) return 0;
    std::uint64_t delay = p.initialDelayMs;
    for (std::uint32_t i = 1; i < attempt; ++i) {
        delay *= p.multiplier;
        if (delay >= p.maxDelayMs) return p.maxDelayMs;   // saturation
    }
    return static_cast<std::uint32_t>(delay > p.maxDelayMs ? p.maxDelayMs : delay);
}
```

- **tuning parameter と invariant の分離**（H.11.27.6 設計どおり）:
  - **invariant（変えない）**: NonRT 実行 / non-blocking（spin しない）/ bounded（maxDelayMs 上限存在）/ backoff は RetryBackoff disposition にのみ適用 / counter は caller 側。
  - **tuning parameter（値は構成可能）**: initialDelayMs / maxDelayMs / multiplier。normative default = `{10, 80, 2}`。
- **attempt #1〜N の実際の delay（normative default 時）**: attempt 1 = 10ms / attempt 2 = 20ms / attempt 3 = 40ms（max 3 のため 80 は cap としてのみ存在）。
- **overflow / saturation**: `delay >= maxDelayMs` で早期 return（uint64 中間計算 + 上限打ち切り）により multiplier 掛算の overflow を構造的に排除。
- **RetryImmediate と RetryBackoff の境界**: `RetryImmediate` → **delay 0（backoff を適用しない）**・ただし max retry bound は等しく適用。`RetryBackoff` → `retryBackoffDelayMs(policy, attempt)`。境界は disposition 値で決まり、error 値では決まらない。
- **配置**: `BuildErrorPolicy.h`（policy 契約専用ヘッダ・JUCE 非依存・既存 `BuildErrorClassificationTests` target から standalone test 可能 → **CMake 変更不要**）。

## 5. RetryDisposition → delay mapping

```text
classifyBuildError(error).retry
    ├─ NoRetry        → schedule しない（telemetry なし）
    ├─ RetryImmediate → delay = 0ms（bound のみ適用）         ← WarmupFailed（Site 3 のみ到達）
    └─ RetryBackoff   → delay = retryBackoffDelayMs(policy, attempt)   ← 現行 Site 3 では dormant
```

- **WarmupFailed = Transient + RetryImmediate だが、無条件 retry ではない**: 実 retry には `shouldRetryWarmupFailure()`（= `isLoadingIR()`・ContextDependent 判定）が必須条件。**enum 値を無条件 retry と解釈する設計は禁止**（指示 §4）。
- **RetryBackoff は接続済みだが現行 dormant**: Site 3 で retry 対象になる failure class は WarmupFailed のみ（build failure は retry 対象外・§2）。`RetryBackoff` の consumer は存在しないため policy は定義・接続のみで dormant — 将来 build failure retry が承認された場合に使用（§12 の対象外明記と対）。
- **`classifyBuildError()` は policy source、Site 3 context（`isLoadingIR()`）は ContextDependent 判定** — 責務分離を維持。

## 6. Generation / obsolete interaction

```text
counter は generation に紐付く:
    task.generation == boundGeneration → counter 続行
    task.generation != boundGeneration → counter = 0・boundGeneration 更新（新規 request = retry state 再開）
    exhausted generation への再 wake     → !wokeByPendingTask なら D132 防御（:1135 付近）で continue
                                          ・wokeByPendingTask でも obsolete なら isObsolete で continue
exhaustion 後の同 generation 再発行:
    Replaceable collapse による隠れた無限再発行は存在しない（exhaustion 後は schedule しない =
    新規 intent を発行しない。submitRebuildIntent が再び呼ばれるのは別の明示的 request 時のみ）
```

- schedule 前に `isObsolete()` を再確認（obsolete → schedule しない・drop）。build 中に obsolete 化した場合も retry を打たない。
- 明示的な新規 rebuild request（新 generation）は counter を 0 から再開 — exhaustion は永続 mask ではない。

## 7. Scheduler interaction

- 接続方法: 既存 `retryScheduler_->schedule(req, delay)` の **delay 引数に §5 の mapping を渡す**。RetryScheduler の infrastructure（deque / worker / shutdown）は **1 行も変更しない**。
- `schedule()` は `void`（rejection 非通知）: capacity 満杯 / shutdown 時の reject は **attempt を消費した retry drop として扱う**（counter の巻き戻しはしない — 再 queue storm を避ける簡潔規則）。rejectCount telemetry は既存のまま。
- **RetryScheduler を BuildError-aware にしない**（D101-24 契約維持・BE-8）。

## 8. Exhaustion behavior

max 超過時（generation G で attempt 4 回目の warmup failure）:

```text
1. schedule しない（submitRebuildIntent を追加発行しない — collapse による隠れた無限再発行の構造的排除）
2. terminal telemetry を 1 回だけ発火（同 generation で重複発火しない — exhausted flag は
   boundGeneration == G かつ counter >= max の状態で表現。generation 変更で自動解除）
3. 以後の同 generation warmup failure は continue（telemetry なし）
4. 次の明示的 rebuild request（新 generation）で counter = 0 から再開
5. diagLog に generation / error 名 / attempt 数 / exhausted を記録（既存 diagLog 形式）
```

## 9. Telemetry contract（最小）

| 項目 | 実装 | 必須度 |
|---|---|---|
| retry attempt count | `emitRebuildTelemetry` 経由ではなく diagLog（attempt / limit / generation / error 名）+ `RebuildTelemetryReason::WarmupRetryScheduled`（enum 追加 1 値） | 必須 |
| retry exhausted | `RebuildTelemetryReason::WarmupRetryExhausted`（enum 追加 1 値）+ diagLog（generation・error・attempts=3） | 必須（terminal event） |
| BuildError / generation | diagLog に含める（既存形式の拡張） | 必須 |
| **retryStormDetected** | **必須にしない** — bounded retry（max 3 + exhaustion terminal）が storm 検出の必要性を構造的に排除するため代替判定とする | 判定: 代替 |

- `RetrySchedulerTypes.h` の `RebuildTelemetryReason` に 2 値追加（additive・既存 switch への影響は telemetry 出力の全数検査で確認 — enum↔toString 網羅の既存 static_assert パターンに従う）。
- `retryLatency` は見送り（delay 計測の価値が delay=0 の現状にない・将来 RetryBackoff が活性化した時点で再判断）。

## 10. RT / NonRT boundary（BE-8 acceptance）

CR-α 適用後も以下を維持（acceptance condition）:

```text
Audio Thread  ×  BuildError classification        （RebuildThread のみ）
              ×  RetryDecision                    （RebuildThread のみ）
              ×  backoff calculation              （RebuildThread のみ・純関数）
              ×  RetryScheduler::schedule         （RebuildThread のみ）
```

- 追加コードの設置場所: `BuildErrorPolicy.h`（純関数・NonRT/RT 問わず呼べるが production 呼び出しは RebuildThread のみ）+ `RebuildDispatch.cpp`（rebuildThreadLoop 内）。
- Practical Stable ISR Bridge 原則「RT は所有権を持たず、実行のみ・判断しない」: 変更なし（observation / read path に触れない）。

## 11. Ownership / lifetime impact

- **影響 0**: counter は RebuildThread stack（thread-local by ownership）・policy は constexpr・decision は純関数。新 authority / new queue / new atomic / new lifetime mechanism は**一切導入しない**。
- RuntimeStore / RuntimeWorldAuthority / PublishExecutor / Retire-EBR / Recovery admission topology / CW-8 / Crossfade: **非接触**。
- Retry 中も durable admission は lease 保持（Site 2 の既存機構）— Site 3 は recovery admission と無関係（task pipeline）。

## 12. Forbidden changes（CR-α 対象外の固定）

```text
Site 1/2 recovery retry の変更          ×
obligation K=4（kMaxObligationConsecutiveFailures）の変更  ×
Builder-local spin guard 4 の変更       ×
RetryScheduler infrastructure の変更    ×（delay 引数は既存）
RuntimeStore / RuntimeWorld / EBR / retire の変更           ×
MKLFailure / ConvolverFailure / PrepareFailure の生成経路追加 ×
DSPCore::prepare() 等の status propagation 改修             ×
Site 3 build failure（runtime==nullptr）の retry 化         ×（将来拡張として明示保留）
新 ownership authority / queue / atomic の導入              ×
ReadToken / CW-8 の意味変更                                 ×
RetryScheduleRequest の field 追加                          ×
```

## 13. Test contract（CR-α-2 で実装）

| ID | 検証 | 方法 |
|---|---|---|
| **T-CRα-1** | backoff delay 計算: attempt 1/2/3/4 → 10/20/40/80・saturation（attempt 5+ → 80）・attempt 0 → 0 | `BuildErrorClassificationTests` target に追加（`retryBackoffDelayMs` 純関数・**CMake 変更なし**） |
| **T-CRα-2** | normative default: `{10, 80, 2}` の constexpr 値検証 | 同上 |
| **T-CRα-3** | warmup retry decision（純関数化する場合）: attempt<max && context && !obsolete → schedule・attempt==max → exhausted・context false → no schedule | 純関数を BuildErrorPolicy.h に置く場合は同 target / RebuildDispatch.cpp 内 inline の場合はソース文字列検査（ObservePath 前例） |
| **T-CRα-4** | RetryImmediate → delay 0 / RetryBackoff → policy delay / NoRetry → schedule なし の mapping | 同上 |
| **T-CRα-5** | exhaustion で submitRebuildIntent を再発行しないこと | ソース文字列検査（exhaustion 分岐内に submit 呼び出しが無いこと）+ runtime（可能なら） |
| **T-CRα-6** | RebuildTelemetryReason 追加 2 値の enum↔toString 網羅 | 既存 static_assert パターン |

実装側の推奨構造（判断余地の削除）: retry decision を**純関数**（`warmupRetryDecision(attempt, max, contextRetryable, obsolete, disposition) → {schedule, delayMs}`）として BuildErrorPolicy.h に置き、rebuildThreadLoop は counter 維持 + 純関数呼び出し + schedule 実行のみにする。→ T-CW8-7 と同じ「decision の unit test 可能化」パターン。

## 14. GO / CONDITIONAL-GO / NO-GO

> ## **GO**
>
| 条件 | 判定 |
|---|---|
| state owner 確定（RebuildThread 関数スコープ・generation 紐付け） | ✓ §1 |
| attempt semantics 確定（warmup failure + context + !obsolete = 1 attempt） | ✓ §2 |
| max bound 確定（**3**・recovery K=4 と別値・別名） | ✓ §3 |
| backoff 式確定（`{10, 80, 2}` normative default・saturation 付き・純関数） | ✓ §4 |
| disposition→delay mapping 確定（Immediate=0 / Backoff=policy / NoRetry=none・context gate 必須） | ✓ §5 |
| generation/obsolete interaction 確定（exhaustion は generation mask・collapse 再発行なし） | ✓ §6 |
| scheduler 接続確定（delay 引数のみ・infrastructure 変更 0） | ✓ §7 |
| exhaustion behavior 確定（schedule しない・terminal telemetry 1 回） | ✓ §8 |
| telemetry 最小化確定（2 enum 値 + diagLog・retryStormDetected は bounded retry で代替） | ✓ §9 |
| BE-8 / ownership 維持 | ✓ §10/11 |
| 対象外固定 | ✓ §12 |

- 判断余地の残存: **なし**（§1-9 の各決定は実装コードに直接写像できる）。
- 実装規模見込み: production = `BuildErrorPolicy.h`（+35 行前後: policy + delay 純関数 + decision 純関数）+ `RebuildDispatch.cpp`（counter + gate 置換・+25 行前後）+ `RetrySchedulerTypes.h`（enum 2 値）。test = `BuildErrorClassificationTests.cpp` 追加のみ → **CMake 変更なし**。
- NO-GO 条件の発火: なし。

## 遷移

```text
ND-06 CONDITIONAL-GO
   ↓
ND-07 = GO（本契約書）   ← 実装時の判断余地ゼロ
   ↓
CR-α-1 最小実装（contract どおり）
   ↓
CR-α-2 targeted tests（T-CRα-1〜6）
   ↓
CR-α-3 Debug / Release build
   ↓
CR-α-4 CTest + regression（40/40 現行基準）
   ↓
CR-α closure
```
