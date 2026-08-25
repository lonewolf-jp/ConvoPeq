# D102-C2-2 — O_denom Measurement & Eligibility Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only measurement/eligibility audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS** — O_denom の定義・算出式・観測値・適用条件・bootstrap invariant・契約根拠の全項確認完了
- **基準**: ConvoPeq.md **2026-08-25 21:53 再生成版**（ローカル実ソースから）

---

## 1. O_denom の定義箇所と算出式

### 定義

```text
O_denom ≜ eligible measurement windows の observedOutstandingMax の最大値
        = max { windowMax(w) | w ∈ eligible windows }

observedOutstandingMax ≜ window 内の tick estimate (A − R) の running max
windowMax ≜ observedOutstandingMax_ atomic 変数の値
```

### 算出式

```text
est(tick_k) = signedWide(A(tick_k)) − signedWide(R(tick_k))
            = acquireObserved − releaseObserved（D82: unsigned wraparound 回避）

windowMax(w) = max{ est(tick_0), est(tick_1), ..., est(tick_n) }
             初期値 = firstEstimate = A0 − R0（beginWindow で設定・D91 監視項目 1）
```

---

## 2. numerator / denominator の各観測値

| counter | writer | 読み取りタイミング | 意味 |
|---|---|---|---|
| `acquireObserved` | `onAcquireObserved()` Commit.cpp:406 | samplerTick で即値読み | publish commit 成功回数（累積） |
| `releaseObserved` | `addReleaseObserved(delta)` Timer.cpp:382 | 同上 | World terminal destruction 反映回数（累積） |

### A/R の更新タイミング差

```text
A: publish LP で即時 +1（Commit.cpp:406・CoordinatorLoop 上）
R: storage 側 worldReclaimCount_ が即時 +1、sampler が cursor delta を transfer
   （Timer.cpp:372-386 transferWorldReclaimDeltaForTelemetry）
```

この差により intra-tick window で estimate が真値を一時的に超え得るが、
**overstate 方向（safe）**であり understate は発生しない。

---

## 3. 観測期間・sampling 条件

```text
sampler cadence: 100ms（timerCallback・MessageThread NonRT）
window transition owner: samplerTick（D91 唯一の owner）
Start: beginWindow — A0/R0 snapshot + firstEstimate = windowMax 初期値
Running: sampleWindow — estimate 計算 → updateWindowMax
End: closeWindow — finalEstimate で windowMax 最終更新 → Closed
```

---

## 4. bootstrap / post-first-commit の適用条件

### bootstrap path（Init.cpp:86）

```text
bootstrapBridge.didPublishRuntimeNonRt(*bootstrapWorldPtr)
    → engine_->onRuntimePublishedNonRt(world)
    → Commit.cpp:406 onAcquireObserved()    ← ✅ bootstrap も A++ を発火
```

### post-first-commit invariant

```text
最初の committed publish 以降:
    A ≥ 1（acquireObserved は decrement されない）
    resident current ≠ nullptr → R < A
    ∴ est = A − R ≥ 1（構造的に保証）
```

---

## 5. `O_denom > 0` の成立確認

| 条件 | 実測 | 判定 |
|---|---|---|
| post-first-commit で A ≥ 1 | ✅ Init.cpp:86 → Commit.cpp:406 | ✅ |
| resident release 未発生 | ✅ current ≠ nullptr 間は R 不変 | ✅ |
| samplerTick で est 計算される | ✅ D91 基準 6 | ✅ |
| ∴ O_denom ≥ 1 | ✅ **構造的に成立** | ✅ |

---

## 6. 契約上の根拠（ceil(4120/O_denom) への投入可否）

```text
投入可能条件:
  (a) O_denom は actual measurement であること（推測・仮定でないこと）✅
      → snap.windowMax は samplerTick の実測値
  (b) O_denom > 0 であること ✅
      → bootstrap invariant により構造的に保証
  (c) M_scope との単位整合 ✅
      → 両者とも「published World outstanding count」の同一意味論

∴ ceil(4120 / O_denom) への投入は契約上正当 ✅
```

---

## 7. VERDICT: **PASS**

全 7 項目確認完了。O_denom 測定プロトコルは確定済み。
次フェーズ D102-C3 へ進行可能。
