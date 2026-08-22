# Phase I-T1-MR — Measurement Run

> **Verdict: T1-MR = CLOSED（3 条件すべて acceptance PASS）**
> **A_max_candidate = RECORDED**（sampled: 1 / event-driven reference peak: 2 — いずれも安全上界として宣言しない）
> **R = UNDETERMINED / R_cap = UNDETERMINED / M = UNDETERMINED / B_max^true = UNDETERMINED**
> `M = max(E_w)` への短絡は NO-GO（D100.5・I4）— max(E_w)=1 は **observed characterization** としてのみ記録。
> コード変更: measurement harness の計装追加のみ（baseline snapshot / raw evidence 出力 / acceptance check）。production 変更なし。

## 1. Protocol

| 項目 | 内容 |
|---|---|
| engine | fresh AudioEngineHarness per condition（headless・audio thread 稼働） |
| baseline 安定化 | `stabilizeMeasurementBaseline()` を measurement window 外で実行（起動直後の未転送破壊を窓外へ排出・cursor 同期）— C6-G2-1 の教訓を適用 |
| sampler | 共有 measurement step（`runWorldRetirementMeasurementStep`）を独立スレッドから 100ms cadence で駆動（D100.5） |
| window | requestStart → publish/reclaim 反復 → requestEnd → Closed snapshot 待ち |
| end flush | sampler 停止後に残存転送を flush してから end snapshot（dWc/dRel lag 防止） |

## 2. Raw evidence（3 条件）

### normal（cadence 100ms / publishes=8 / interval=150ms）

```
baseline   A=3 R=2 observedOutstandingMax=1 worldReclaimCount=2 referenceRelease=2
window     windowId=1 sampleCount=21 maxSamplingGapUs=100998 missedTickCount=0 counterWrapped=0
evidence   A_start=3  R_start=2  A_end=11 R_end=10
           observedOutstandingMax start=1 end=1
           worldReclaimCount 2→10   referenceRelease 2→10
           windowStartUs=13031017487 windowEndUs=13033118144 (duration ≈ 2,100,657 µs)
```

### burst（cadence 100ms / publishes=20 / interval=150ms）

```
baseline   A=3 R=2 observedOutstandingMax=1 worldReclaimCount=2 referenceRelease=2
window     windowId=1 sampleCount=46 maxSamplingGapUs=110050 missedTickCount=0 counterWrapped=0
evidence   A_start=3  R_start=2  A_end=23 R_end=22
           observedOutstandingMax start=1 end=1
           worldReclaimCount 2→22   referenceRelease 2→22
           windowStartUs=13036461264 windowEndUs=13041061164 (duration ≈ 4,599,900 µs)
```

### jitter（cadence 100ms / publishes=10 / interval=irregular {0,80,200,0,120,300,0,60,180,0}ms）

```
baseline   A=3 R=2 observedOutstandingMax=1 worldReclaimCount=2 referenceRelease=2
window     windowId=1 sampleCount=22 maxSamplingGapUs=111479 missedTickCount=0 counterWrapped=0
evidence   A_start=3  R_start=2  A_end=13 R_end=12
           observedOutstandingMax start=1 end=1
           worldReclaimCount 2→12   referenceRelease 2→12
           windowStartUs=13044212795 windowEndUs=13046428401 (duration ≈ 2,215,606 µs)
```

整合メモ: sampleCount ≈ duration/100ms（21/46/22）と一致し、missedTickCount=0 — sampling cadence が規定どおり維持された。

## 3. Acceptance checks

| Check | normal | burst | jitter |
|---|---|---|---|
| Counter conservation `ΔreleaseObserved == ΔworldReclaimCount` | 8==8 ✅ | 20==20 ✅ | 10==10 ✅ |
| Reference consistency `ΔreferenceRelease == ΔworldReclaimCount` | 8==8 ✅ | 20==20 ✅ | 10==10 ✅ |
| Outstanding identity `estimate == A - R` | ✅ | ✅ | ✅ |
| Window identity `E_w == T_w - O_w` | ✅ | ✅ | ✅ |
| Non-negative excess `E_w >= 0` | 1 ✅ | 1 ✅ | 1 ✅ |
| Max monotonicity（`M_end >= M_start`） | 1→1 ✅ | 1→1 ✅ | 1→1 ✅ |
| `counterWrapped == false` | ✅ | ✅ | ✅ |
| Sampling metadata valid（gap≈cadence, missed=0） | ✅ | ✅ | ✅ |

**3/3 条件が全 acceptance を満たす → T1-MR = CLOSED。**

## 4. 結果表（`observedOutstandingMax` と `windowMax` は別列・別 semantic）

| Condition | O_w (= snap.windowMax) | T_w (reference peak) | E_w | observedOutstandingMax (accumulated) | windowMax (window-local) |
|---|---:|---:|---:|---:|---:|
| normal | 1 | 2 | 1 | 1 | 1 |
| burst | 1 | 2 | 1 | 1 | 1 |
| jitter | 1 | 2 | 1 | 1 | 1 |

- `O_w ≡ snap.windowMax`（定義により同一値・参考として両列記載）。
- `T_w = 2 > O_w = 1`（全条件）: event-driven reference は publish 直後の transient outstanding=2 を捕捉し、100ms sampled max はこれを取りこぼす — **E_w > 0 の構造的再現**（burst 設計意図どおり）。
- `observedOutstandingMax` は accumulated live max として 1 で安定（transient 2 は 100ms サンプル間で解消されるため不捕捉 — D91 基準 8 の bounded sampled 性質どおり）。

## 5. E_w characterization（M ではない）

```
max(E_w across valid runs) = 1   ← observed characterization のみ
```

- `M = max(E_w)`: **NO-GO**（D100.5 / I4 — 有限実測の max(E_w) を安全上界としない）
- `R = max(E_w)` / `R_cap`: **NO-GO**
- `B_max^true <= O_w + max(E_w)`: **未証明**（D101 OPEN — 数学的 bound 未導出）

## 6. A_max_candidate の記録

```
A_max_candidate(sampled, protocol-defined)
    = max(observedOutstandingMax across valid measurement runs)
    = 1

A_max_candidate(supplementary, event-driven reference peak)
    = max(T_w across valid runs)
    = 2
```

- source / derivation: 本 run の 3 valid conditions における accumulated sampled maximum（primary）、および event-driven reference maximum（supplementary 記録）。
- **これは安全上界ではない。**「現在の measurement protocol で観測された sampled outstanding の最大候補」以上の意味を持たない。
- `A_max` / `M` / `R_cap` / `B_max^true` とは別物（命名厳密遵守）。

## 7. UNDETERMINED のまま残る項目

```
R            = UNDETERMINED
R_cap        = UNDETERMINED
M            = UNDETERMINED
B_max^true   = UNDETERMINED
T2 / I-2     = 未着手
```

## 8. 次の Gate — D101 / M-bound investigation

```
C6 PASS → T1-MR CLOSED → A_max_candidate RECORDED
        ↓
D101 — M mathematical bound（未完了課題）
  ├─ reference observer completeness
  ├─ sampling cadence λ
  ├─ publication/reclaim timing τ_b
  ├─ generation/window constraints G
  └─ structural bound: B_max^true <= O_w + M(G, λ, τ_b, ...)
        ↓
M-bound established ?
  ├─ NO  → R/R_cap/M は UNDETERMINED 継続
  └─ YES → R determination review
```

## 9. コード変更の報告

本フェーズでの変更は **measurement harness の計装のみ**（`WorldRetirementMeasurementTests.cpp`）:

- `stabilizeMeasurementBaseline` / `waitForWorldReclaimCount` の前方宣言追加（既定引数は宣言側に統一）
- `runMeasurement` / `testJitterMeasurement` へ baseline snapshot・end flush・raw evidence 出力・acceptance checks を追加

production code（AudioEngine.* / ISRWorldRetirement* / retire 系）は **無変更**。

## 10. Tool Coverage

node fs（計装差分確認）/ get_errors（lint 0）/ MSVC Debug build / AudioEngineHarness --measurement=all 実測 / serena・AiDex・semble・ccc・graphify・WSL rg 系は C6-Final までに確定済みの参照整合を維持 / 文献 9 系統 200 OK 済。

---

*Evidence generated: Phase I-T1-MR — 3 条件 acceptance 全 PASS・A_max_candidate RECORDED。R は UNDETERMINED。次 Gate = D101 / M-bound investigation。*
