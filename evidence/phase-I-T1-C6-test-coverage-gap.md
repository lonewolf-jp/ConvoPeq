# Phase I-T1-C6 — Test Coverage / Gap Verification

> **Verdict: FAIL（G2 = 1件）— C6 PASS 不可。measurement run へは進まない。**
> 目的: C1〜C5 で証明した measurement contract がテストコードでどこまで動的に検証され、どこに gap が残るかを現行ソースと `WorldRetirementMeasurementTests` 照合で確定する。
> コード変更なし。G2 解消まで T1 measurement run / A_max candidate / R は全て停止。

## 1. C6-1 — Test inventory（15必須ケース × 3値判定）

調査対象: `src/tests/` 配下 41 ファイル全件を node census + rg/fdfind で走査。

| # | 必須ケース | 判定 | 根拠 |
|---|---|---|---|
| 1 | acquire = 1 | **部分的** | `WorldRetirementMeasurementTests.cpp:115-117` が `acquireObserved()` を diagnostic print（assert なし・exactly-1 検証なし） |
| 2 | release = 1 | **存在しない** | `releaseObserved()` を読むテストは **0件**（rg/AiDex contains/node census 全て 0 hits） |
| 3 | outstanding = A − R | **存在しない** | A/R 差分の assert なし |
| 4 | outstanding max 単調増加 | **存在しない** | `observedOutstandingMax` はテスト内 0 hits |
| 5 | window baseline | **部分的** | Start→Running 遷移と `lastClosedSnapshot` は検証（:43/:84-90）。baseline 値自体の assert なし |
| 6 | window close | **存在する** | normal/burst/jitter 3条件で Closed snapshot・O_w/T_w/E_w・sampleCount/gap/missed/wrapped を記録、`T_w >= O_w` を検証 |
| 7 | shutdown / final drain | **部分的** | drain 機構は `RetireGraceSemanticsTests`/`StuckReaderFallbackDrainTests`/`invariant_INV3_INV5` でカバー。ただし shutdown drain 起因の telemetry 観測 assert なし |
| 8 | bootstrap | **存在する** | `PublishPipelineIntegrationTests`（bootstrap publish 成功系） |
| 9 | rejected / non-admitted | **存在する** | `PartialPublicationRejectTests`/`ISRSemanticValidationTests`(intent queue full backpressure)/`ISRSoakTests` 等多数 |
| 10 | Terminal disposition 3経路 | **部分的** | D(`DeferredDeletionQueueReclaimTests`)・Q/E/T drain 機構(`RetireGraceSemanticsTests`/`StuckReaderFallbackDrainTests`)は storage レベルでカバー。ただし「destruction → reclaimCount → measurement observation」chain の assert なし |
| 11 | duplicate protection | **存在する** | `OwnerChannelTests`/`SequenceArithmeticTests`/`SoakPublishIntegrationTests` 等 |
| 12 | retry → Terminal | **部分的** | `RetireGraceSemanticsTests` が enqueueWithRetry chain 構造をカバー。Terminal 到達時の観測 assert なし |
| 13 | multiple destruction | **部分的** | Soak 系で多数 World publish/reclaim。ただし ΔworldReclaimCount == ΔreleaseObserved の検証なし |
| 14 | cross-window isolation | **存在しない** | 第2 window（Start→End→Start）のテストなし |
| 15 | measurement/export representation | **部分的** | O_w/T_w/E_w/snapshot stats は検証。JSON export 内容（C5 の 24 field）のテストなし |

## 2. C6-2 — `releaseObserved` 直接テスト監査（最重要）

### 2.1 census 結果（41 ファイル全件）

| 検索対象 | hits |
|---|---|
| `releaseObserved()` を読むテスト | **0** |
| `onReleaseObserved()` / `addReleaseObserved()` / `onAcquireObserved()` の直接呼出 | **0** |
| `worldReclaimCount` への言及 | **0** |
| `setReferenceObserver` / observer wiring 検証 | **0**（`referenceReleaseCount()` の diagnostic print 1件のみ） |

### 2.2 harness が authoritative path を迂回している

`driveWorldRetirementSamplerForMeasurement`（`AudioEngine.h:4946-4954`）は `samplerTick()` + reference window 同期のみを実行し、**timerCallback 内の delta-transfer block（`Timer.cpp:423-431`: worldReclaimCount delta → addReleaseObserved）を含まない**。したがってヘッドレス測定テストでは `releaseObserved_` が一切進まない。

diagnostic print（`:115-117`）が `acquireObserved` と reference 両カウンタを出す一方で `releaseObserved` を意図的に欠いていることも、この経路が未検証であることの傍証。

### 2.3 判定

> **R1 counterexample（同一 destruction → releaseObserved +2）が発生しても、現行テストスイート全体がそれを検出できない。**
> C2〜C5 の invariant（single writer / 1:1 transfer / export conservation）は静的証明のみで、動的検証が存在しない。これは単なる coverage 欠落ではなく、measurement correctness（`observedOutstanding = A − R` の R 側）の証明に直接影響する。

## 3. C6-3 — 3 Terminal paths の完全性

| Store | storage/drain 機構テスト | telemetry 観測（reclaimCount/onRelease）テスト |
|---|---|---|
| DeferredDeletionQueue | ✅ `DeferredDeletionQueueReclaimTests`（reclaim/drainAllUnsafe） | ❌ なし |
| RetireQuarantineStore (Q/E) | ✅ `RetireGraceSemanticsTests:327-356`（quarantine→resident→drain reset）/ `StuckReaderFallbackDrainTests` | ❌ なし |
| TerminalReclaimAuthority | ✅ `StuckReaderFallbackDrainTests:112-146`（growable store・fill/drain） | ❌ なし |

shutdown drain と通常 drain はテスト上も別経路で扱われている（同一視なし）。ただし **いずれの経路も「World destruction → counter → observation」の測定側までは検証していない**。

## 4. C6-4 — negative / adversarial cases

| ケース | 状態 |
|---|---|
| rejected publish | ✅ 複数テストで検出可能 |
| duplicate | ✅ OwnerChannel/SequenceArithmetic 等で検出可能 |
| retry → Terminal | 部分的（chain 構造のみ・観測 assert なし） |
| multiple destruction | 部分的（soak は実行するが R 側を検証しない） |
| window boundary | ❌ 専用テストなし |
| shutdown boundary | 部分的（drain 機構のみ） |

「誤った release が releaseObserved に混入する」「正当な release が欠落する」counterexample を**現在のテストは検出不可能**。

## 5. C6-5 — Gap classification

### G0（十分にカバー済み）

window close 機構（O_w/T_w/E_w・snapshot stats）、bootstrap、rejected、duplicate、D/Q/E/T storage drain 機構、retry chain 構造。

### G1（軽微な直接テスト不足・production 証明に影響しない）

- G1-1: `observedOutstandingMax` 単調増加の明示 assert（sampler-only writer は C4 で静的確定済み）
- G1-2: cross-window isolation（第2 window）テスト（beginWindow reset は C5-4 で静的確定済み）
- G1-3: JSON export 内容テスト（C5 で 24 field mapping を静的確定済み）
- G1-4: observer wiring（setReferenceObserver 経由）の明示テスト（production wiring は AudioEngine 初期化で確認済み）
- G1-5: exactly-1 acquire/release の値 assert（現状は diagnostic print のみ）

### G2（measurement correctness の証明に影響する未検証経路）

- **G2-1: authoritative release observation path（destruction → worldReclaimCount → sampler delta → addReleaseObserved → releaseObserved）の動的テストが 0件。**
  - テストハーネスが delta-transfer を迂回するため、既存 normal/burst/jitter テストですら `releaseObserved` を一度も検証していない。
  - R1 で実際に発見された double-count クラスの欠陥が全スイートを通過する。
  - `observedOutstanding = A − R` / `observedOutstandingMax` / `A_max candidate` の R 側根拠が動的に未検証。

## 6. C6-6 — 判定

```
T1-C6 = FAIL（G2 = 1件）
```

- PASS 条件「G2 = 0」を満たさない。
- **measurement run / A_max candidate / R の決定は G2-1 解消まで禁止を継続。**

### G2-1 解消に必要な最小要件（設計指示・本フェーズでは実装しない）

1. テストハーネスが timerCallback と同一の delta-transfer（worldReclaimCount delta → addReleaseObserved）を実行する構造にする（共有 sampler step 化または harness 側再現）。
2. deterministic test: 1 destruction → `ΔworldReclaimCount = 1` → `ΔreleaseObserved = 1`、かつ observer 経由の寄与 = 0（R1 反転の自動検出化）。
3. double-count regression guard: N destruction 後に `releaseObserved() == worldReclaimCount()` を assert。
4. 上記成立後、G1 を Residual Gap として明示した上で C6 再判定 → measurement run へ。

## 7. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 41 test ファイル全件走査・census 一致 |
| Sandbox | node fs walk（41 files × 14 pattern census） | worldReclaimCount/releaseObserved/direct API = 0 を決定づけ |
| MCP#1 | serena | C2-C5 でシンボル参照確定済み（本 Gate の census と整合） |
| MCP#2 | ccc | search 実行 |
| CLI#1 | graphify 0.9.48 | exe 存在確認 |
| CLI#2 | semble 0.5.5 | `releaseObserved assertion test` 検索実行（test 内 0 hit を裏付け） |
| MCP#3 | AiDex | `file_filter=src/tests/**` で `releaseObserved` contains = **No matches** を索引レベルで確認 |
| 文献 | crossbeam-epoch / rigtorp / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9件 200 OK（Vyukov SSL失効→rigtorp代替明記） |

## 8. 進行状態

```
C1-R3 PASS → C2 PASS → C3 PASS → C4 PASS → C5 PASS → C6 FAIL (G2-1)
                                                        │
                                                        ▼
                              G2-1 解消（harness delta-transfer + deterministic test）
                                                        │
                                                        ▼
                                           C6 再判定 → measurement run
```

- 実施しないもの: コード修正 / R・R_cap 決定 / A_max 確定 / T2 / Recovery coalesce / I4 D12-17 実装 / shutdown architecture 変更。
- `ConvoPeq.md` スナップショットと `src/` の差分は R3 切断箇所のみで本判定に影響なし。

---

*Evidence generated: Phase I-T1-C6 — no code change. G2-1 解消まで measurement run には進まない。*
