# D102-C2-6 — Numeric Gate / O_denom Evidence Closure Audit

- **実施日**: 2026-08-26
- **基準版**: `ConvoPeq.md Generated: 2026-08-26 20:06:47`
- **作業種別**: **read-only / audit-only**（production 変更 **0** / test 変更 **0** / CMake 変更 **0** / contract 変更 **0**）
- **対象**: D101-35-C 確定の `M ≤ M_scope` 以降に残る D102-C2 数値ゲート（`λ_prod_bound / G_bound / M_scope / O_denom / R_required`）
- **前提**: D2 series（D2-0 freeze / D2-1 NO-GO / D2-2 PASS-B / D2-3 Gate 不成立）により Terminal growable 維持・K_terminal 未導入・QueueFull dead 維持が確定済み（本監査では Terminal へ**戻らない** — §C6-4）

---

## C6-1. 契約値の source identity — PASS

### 固定値（契約入力として凍結）

| パラメータ | 契約値 | source | 種別 |
|---|---|---|---|
| `λ_prod_bound` | **13 events/s** | `src/tests/AudioEngineHarness/OdenomCampaignTests.cpp:7` コメント + L372 `contract: lambda=13/s` 出力 | 契約入力（固定） |
| `G_bound` | **1.0 s** | 同 L7 / 同 L372 | 同上 |
| `K_starve` | **1.0 s** | 同 L7 / 同 L372 | 同上 |
| `T_sampler` | **100 ms** | 同 L7 / `AudioEngine::Init.cpp:121 startTimer(100)` / `ISRWorldRetirementTelemetry.h:311 kExpectedTickIntervalUs=100'000` | コード定数 |
| `M_scope` | **4120** | 同 L7 + 同 L372 + `evidence/D102-C2-3-Numerical-Result.md:17 (4096+13*1.0+11)` | 契約導出値（凍結） |
| `R_cap,bounded` | **5120** | `evidence/D102-C2-3-Numerical-Result.md:18 (=4096(D)+512(Q)+512(E))` / harness L390 `R_cap,bounded=5120` | bounded 合計容量（凍結） |

### 観測値と契約値の混同なし — PASS

- Odenom campaign は `observedRate ≈ 4.11 events/s (contractRate=13/s, NOT redefined)`（`OdenomCampaignTests.cpp:380`）と**明示分離**して出力。
- `λ_prod_bound=13/s` は再定義されず、`G_bound/K_starve/T_sampler/M_scope` も実測で上書きされていない。
- `git diff --stat -- src/` = **0 行**（上記 batch 検証 `no output` により確認）。契約値ドリフトなし。

---

## C6-2. O_denom provenance audit — PASS

### 定義（不変）

```
O_denom = max(windowMax(w)) over all eligible windows w
          where windowMax = telemetry.lastClosedSnapshot().windowMax
                （ISRWorldRetirementTelemetry.h:194）
          外部集約のみ — telemetry 本体は変更しない
```

### 実測 campaign プロファイル（source: OdenomCampaignTests.cpp:1-120, 270-434）

- **profile**: `4 pubs/window` × `60ms interval` × `100ms sampler`
- **campaign**: `warmup 1 + 10 eligible` ウィンドウ、外部 campaignStart/End タイムスタンプで集約
- **build**: `AudioEngineHarness.exe Debug`（`ConvoPeq.md 2026-08-25 22:11:33` 基準、
  `git diff src/ = 0` を G1 として機械検証）

### エビデンス所在

| エビデンス | パス | サイズ / 時刻 |
|---|---|---|
| 実行報告 | `evidence/D102-C2-3-B-Campaign-Execution-Report.md` | 9.9 KB / 2026-08-25 23:43 |
| Raw snapshots | `evidence/D102-C2-3-Odenom-Campaign-Raw-Evidence.md` | 7.3 KB / 2026-08-26 00:01 |
| console log | `OdenomCampaign_console_2026-08-25.log`（同報告内で保存宣言） | — |
| 数値確定 | `evidence/D102-C2-3-Numerical-Result.md` | — |

### 本監査が確認した O_denom provenance

1. **eligible window 数**: 10 / excluded 1（WarmupExclusion）— raw evidence に全 11 ウィンドウが `windowId/windowMax/sampleCount/missed/wrapped/valid/tag` 付きで記録。
2. **eligibility 条件**: `valid==1 && counterWrapped==0 && missedTickCount==0 && windowTag==Normal && sampleCount>=2 && campaignStart ≤ windowStart ≤ windowEnd ≤ campaignEnd` — OdenomCampaignTests.cpp:109-119 で機械判定（手動選別なし）。
3. **excluded reason**: 1 WarmupExclusion（campaign window 外）。理由は eligible 判定時に exclusionReason に記録。
4. **windowMax の定義**: `telemetry.lastClosedSnapshot().windowMax` = 100ms サンプル窓内の `referenceAcquire - referenceRelease` running max。
5. **O_denom = max(windowMax) の正当性**: 全 eligible 10 ウィンドウに対し `eligibleMaxes` を収集 → `maxWindowMax` = **1**。`argmax windowId=2`。統計は `min=1, mean=1.00, median=1.00, P95=1, P99=1, max=1`。
6. **複数 eligible**: 10 ウィンドウ（G7 要件「≥2」を満たす）。
7. **raw evidence 保存場所**: 上記 2 md + console log。D102-C2-3 G15 PASS として既に実証。
8. **workload observed と contract rate の分離**: observed 4.11/s を contract 13/s と**分離出力**（L380 `NOT redefined`）。`observedRate` から `λ_prod_bound` への再定義は 0 件（rg 検証）。

### O_w と O_denom の混同 — なし

- `O_w` は単一窓の `windowMax`、`O_denom` は全 eligible 窓の `max(O_w)`。campaign は O_denom のみを K_min に使用（単窓採用ではない）。

---

## C6-3. R_required 再計算監査

### 実装式（source のまま・新式提案なし）

```text
K_min           = ceil(M_scope / O_denom)     … OdenomCampaignTests.cpp:385
R_required      = 1 + K_min                   … L386
TerminalDep     = max(R_required - R_cap, 0)  … L388  (R_cap=5120)
bounded 判定    = R_required <= 5120          … L394-396
```

### 独立再計算（本監査 — M_scope=4120）

| O_denom | K_min | R_required | TerminalDep | bounded |
|---|---:|---:|---:|---|
| 1 | 4120 | **4121** | 0 | **PASS** |
| 2 | 2060 | 2061 | 0 | PASS |
| 4 | 1030 | 1031 | 0 | PASS |
| 8 | 515 | 516 | 0 | PASS |
| 13 | 317 | 318 | 0 | PASS |
| 20 | 206 | 207 | 0 | PASS |
| 50 | 83 | 84 | 0 | PASS |
| 100 | 42 | 43 | 0 | PASS |
| 500 | 9 | 10 | 0 | PASS |
| 1000 | 5 | 6 | 0 | PASS |
| 4120 | 1 | 2 | 0 | PASS |

**実測 O_denom=1（worst-case）での確定値**: `K_min=4120, R_required=4121, TerminalDep=0, headroom=999`。

### 記法一致（I4 揺れの解消）

| レイヤ | 記法 | 一致 |
|---|---|---|
| I4 contract symbolic | `R_required`（場合により `R_req`, `R` と表記揺れ — D102-C2-3-Numerical-Result §3 コメントに明記） | 式 `1 + ceil(M_scope/O_denom)` と数値 4121 で一致（定義の同一性は parametric table で確定） |
| D102-C2-2 実装式 | `K_min=(M_scope+O_denom-1)/O_denom; R_required=1+K_min` | 一致 |
| D102-C2-3 campaign output | `O_denom=1 → K_min=4120 → R_required=4121` | 一致 |

**小括**: 3 者に定義・実装・出力の不一致なし。「R_required 表記揺れ」は数値チェーンに影響しない。

---

## C6-4. D2 Closure との境界 — 遵守

本監査では以下に**一切戻っていない**（read-only / audit-only の範囲を厳守）:

- `K_terminal` / `std::vector → fixed-capacity` / `QueueFull live 化` / `enqueueWithRetry API 変更` /
  `ignoreUnused(result)` 変更 / caller fallback 追加 / `SnapshotCoordinator`・`DSPLifetimeManager` 変更 /
  Terminal `noexcept` セマンティクス変更 / artificial QueueFull/Shutdown path

Terminal を D102 numeric の入力として扱う必要が生じた場合も、
**現行 growable Terminal（store() は常に true、P-4 適合）を前提**として評価するのみ。
その前提下で本 C6-3 の `R_required=4121 ≤ 5120` が **bounded 5120 での PASS** であることは、
Terminal 依存 0 を意味し、growable が無限容量か否かに依存しない。

---

## C6-5. 最終判定

| Gate | 判定 |
|---|---|
| Contract constants | **PASS**（λ/G/K_starve/T_sampler/M_scope/R_cap 全て契約入力として凍結、ドリフト 0） |
| O_denom provenance | **PASS**（warmup+10 eligible、機械集約、raw 保存、workload/contract 分離） |
| O_denom eligibility | **PASS**（条件は機械判定、valid/wrapped/missed/tag/sampleCount/window 範囲 全て固定） |
| O_denom ≥ 1 | **PASS**（1 — 構造的下界を満たす、10 eligible 全て windowMax=1） |
| M_scope fixed | **PASS**（4120 凍結、M=M_scope として数値適用） |
| K_min recomputation | **PASS**（独立再計算 4120 で一致） |
| R_required recomputation | **PASS**（4121 で一致） |
| R_required ≤ R_cap | **PASS**（4121 ≤ 5120、headroom 999、最大要求でも PASS） |
| Terminal dependency | **0**（max(0, 4121−5120) = 0 — bounded 5120 でのみ PASS） |
| **D102 numeric** | **GO** |

### 判定根拠（GO 条件の充足）

```
O_denom ≥ 1          … 1          (PASS)
AND R_required ≤ R_cap … 4121 ≤ 5120 (PASS)
AND provenance 全 PASS … eligible 機械判定・raw 保存・contract/observed 分離
```

全 GO 条件を充足。**TBD なし。**

---

## その後の分岐（C6 GO 時の次工程）

> **D102-C2-7 — R_required / bounded-retirement numerical decision gate**

本 C6-3 で算出した

```text
R_required = 4121
R_cap      = 5120
TerminalDep = 0
```

を decision record として正式化する。
**これは K_terminal 導入を意味しない** — D2-4 の Terminal NO-GO（growable 維持）はそのまま。

決定対象:
- `R_required vs R_cap vs TerminalDep` の bounded-retirement 数値決定
- bounded 5120 で Terminal 依存 0 であることの formal record
- 将来 `M_scope > 5119` に変更される場合のみ Terminal 有限化を要する旨の条件付き注記
  （D102-C2-3 §7 参照: 数値適用前ステップで O_denom 単窓誤用を排し、headroom 999 を根拠に）

D102 数値系は C6 をもって **CLOSE** とし、次は C2-7 へ。
