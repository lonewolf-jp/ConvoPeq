# D101-1 — M-Bound Step 2: Counter / Observation Error Formalization

| 項目 | 内容 |
| --- | --- |
| **日付** | 2026-08-15 |
| **対象** | `src/audioengine/ISRWorldRetirementTelemetry.h`, `ISRWorldRetirementReference.h`, `AudioEngine.Timer.cpp`, `AudioEngine.Commit.cpp`, `AudioEngine.Init.cpp`, `DeferredDeletionQueue.h`, `RetireQuarantineStore.h`, `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp` |
| **目的** | D101-0 の INCOMPLETE を維持したまま、証明義務 $B_{\max}^{\text{true}} \le B_{\max}^{\text{observed}} + M$ のうち「観測誤差 $\Delta = B_{\max}^{\text{true}} - B_{\max}^{\text{observed}}$ を有限の $M$ で上界できるか」をコード・契約から形式化し、判定を下す。 |
| **判定** | **INCOMPLETE** （必要な bound は存在するが、コード/契約上の証拠が不足） |

---

## 1. 4 つの量の数学的分離

D82/D101 の安全マージン証明義務を以下の 4 量に分解する。

| 記号 | 意味 | コード上の対応 |
| --- | --- | --- |
| $B_{\max}^{\text{observed}}$ | 100ms サンプラが観測した outstanding のピーク | $O_w = \texttt{snap.windowMax}$ （`MeasurementSnapshot::windowMax`） |
| $B_{\max}^{\text{true}}$ | 真の outstanding ピーク | $T_w = \texttt{referenceMax\_}$ （`WorldRetirementReferenceObserver`、イベント駆動） |
| $\Delta$ | 観測誤差 $= B_{\max}^{\text{true}} - B_{\max}^{\text{observed}}$ | $E_w = T_w - O_w$ （D100.5 で $\Delta > 0$ を証明済み） |
| $M$ | $\Delta$ を上界する定数 | 本タスクで導出を試みる対象 |

証明義務は $B_{\max}^{\text{true}} \le B_{\max}^{\text{observed}} + M$ すなわち $\Delta \le M$。
$M$ を有限に導出するには、$\Delta$ の構成要素（観測ウィンドウ内の未観測 acquire 数）を有限上界で囲う必要がある。

---

## 2. 潰すべきポイントの調査結果

### 2.1 acquire/release カウンタのアトミック性 — **CONFIRMED**

- `ISRWorldRetirementTelemetry.h`: `acquireObserved_` / `releaseObserved_` は `std::atomic<uint64_t>`。
  - `onAcquireObserved()` → `fetchAddAtomic(acquireObserved_, 1, acq_rel)`
  - `onReleaseObserved(count)` → `fetchAddAtomic(releaseObserved_, count, acq_rel)`
- `ISRWorldRetirementReference.h`: `referenceAcquireCount_` / `referenceReleaseCount_` も `std::atomic<uint64_t>`、`acq_rel` で更新。
- `DeferredDeletionQueue.h` / `RetireQuarantineStore.h`: `worldReclaimCount_` は `alignas(64) std::atomic<uint64_t>`、`fetchAddAtomic(..., acq_rel)` で単調増加。

→ 全カウンタは acq_rel アトミック。インクリメント競合はない。

### 2.2 サンプラの read order — **2 回の別ロード（read-order skew あり）**

`ISRWorldRetirementTelemetry.h` の `sampleWindow()`:

```cpp
const auto a = acquireObserved();   // 別ロード (1)
const auto r = releaseObserved();   // 別ロード (2) — (1) と (2) の間に producer が動作可能
const int64 estimate = static_cast<int64>(a) - static_cast<int64>(r);
```

- `acquireObserved()` と `releaseObserved()` は**独立した 2 つの atomic load**。
- 両者の間に producer（`onAcquireObserved` / `onReleaseObserved`）が介入可能 → 読み取り順スキューが存在する。
- これが観測誤差 $\Delta$ の主要因の一つ（intra-tick 成長 + read-skew）。

### 2.3 worldReclaimCount の差分反映方法 — **CONFIRMED（単調差分）**

`AudioEngine.Timer.cpp:420-443`:

```cpp
const uint64 worldReclaimed = m_retireRouter->worldReclaimCount();
const uint64 delta = worldReclaimed - lastSampledWorldReclaimCount_; // 単調差分
lastSampledWorldReclaimCount_ = worldReclaimed;
telemetry.addReleaseObserved(delta);   // releaseObserved_ += delta
const int64 estimate = telemetry.observedOutstandingEstimate();
telemetry.updateObservedOutstandingMax(estimate);
telemetry.samplerTick(windowNowUs);
```

- `worldReclaimCount_` は単調増加のみ。`delta` は累積値の差分として正しく反映される。
- サンプラは累積カウンタを読み、前回との差分を `releaseObserved_` に加算する設計。

### 2.4 サンプリング間隔 100ms — **実装値（保証値ではない）**

- `AudioEngine.Init.cpp:122`: `timerPeriodMs_ = 100`（ハードコードされた**名目値**）。
- `ISRWorldRetirementTelemetry.h`: `kExpectedTickIntervalUs = 100'000`（static constexpr の名目値）。
- しかし `AudioEngine.Timer.cpp:395-410` に jitter ログ、`missedTickCount`（`gapUs > kExpectedTickIntervalUs * 2` で増加）、`maxSamplingGapUs` が存在 → 間隔は**保証されていない**。
- テスト側も `WorldRetirementMeasurementTests.cpp` で 100ms スレッドだが、burst テスト（`testBurstMeasurement`: 20 publishes / 150ms）で意図的にピークを取りこぼす（$O_w=0, T_w=1, E_w=1$）。

→ 100ms は「実装上の目標間隔」であり、有限上界 $G$ としての**保証値ではない**。

### 2.5 プロデューサの 1 区間 acquire 上界 — **NOT FOUND**

- プロデューサは `AudioEngine.Commit.cpp:332` `onRuntimePublishedNonRt`（CoordinatorLoop / Non-RT）。
- 成功 publish ごと `worldRetirementTelemetry_.onAcquireObserved()` + `worldRetirementReference_.onAcquire()` を呼ぶ。
- コード上に「1 区間あたりの publish 数上限」「publish レート上限 $\lambda$」を定める定数・契約は**存在しない**。

### 2.6 プロデューサ並行度の上界 — **NOT FOUND**

- `validateDistinctRuntimeSlots`（`AudioEngine.h:3795`）は active/fading/queued の 3 ポインタの**異なりチェック（DEBUG アサート）**であり、ワールドプールの個数上限ではない。
- 同時進行 publish 数を制限するロック・セマフォ・プールサイズ定数はコード上に**存在しない**。

### 2.7 burst 継続時間の設計上界 — **NOT FOUND**

- burst の長さ・回数・間隔を規定する設計定数（burst duration $\tau_b$、burst rate $\mu_{\text{burst}}$）は**存在しない**。
- `testBurstMeasurement` は 20 件の連続 publish を想定しているが、これはテストの仮定であり上限の証明ではない。

### 2.8 counter の wraparound が差分計算に与える影響 — **NEGLIGIBLE**

- `worldReclaimCount_` / `acquireObserved_` / `releaseObserved_` はすべて `uint64_t`。
- 単調増加カウンタの差分は $2^{64}$ 回のインクリメントが必要なので、測定期間中の wraparound は実質起こらない。
- `MeasurementSnapshot::counterWrapped` は `closeWindow` で `(a1<a0)||(r1<r0)` を診断的に検出するが、短期ウィンドウでは発火しない。
- D82 契約「測定期間中は A/R が wraparound しない」は満たされる。

→ wraparound は差分計算に実質影響しない。

### 2.9 観測ウィンドウ境界での未観測イベント最大数 — **1 区間分（上界なし）**

- `beginWindow` で $A_0, R_0$ をスナップ、`closeWindow` で $A_1, R_1$ をスナップ。
- 推定値は離散 tick で計算されるため、tick 間の成長（intra-tick growth）が未観測成分となる。
- 境界（開始/終了 tick）で未観測になり得る未観測イベント数は「1 サンプリング区間 $G$ 内の acquire 数」で上界される。
- しかし 2.4–2.7 の通り、$G$ も 1 区間 acquire 数も有限上界がコード/契約に**存在しない**。

---

## 3. M の上界合成

観測誤差は
$$\Delta = B_{\max}^{\text{true}} - B_{\max}^{\text{observed}} = T_w - O_w = E_w$$
であり、$E_w$ は「1 サンプリング区間内の未観測 acquire 数（intra-tick 成長 + read-skew）」で上界される。

有限 $M$ を導出するには以下の有限上界がコード/契約に必要：

1. プロデューサ publish レート上限 $\lambda$
2. burst 継続時間 $\tau_b$ および burst レート $\mu_{\text{burst}}$
3. プロデューサ並行度上限（同時進行 publish 数）
4. 保証されたサンプリング間隔 $G$（100ms は名目値）

**現状**: 上記 1–4 はいずれもコード/契約に存在しない。

- ランタイムプール（もし有限 $N$ なら $B_{\text{true}} \le N$）のような**退化した緩い上界**は存在し得るが、それは $M \le N$ という疎な束縛であり、意図する「観測誤差（1 区間 acquire 数）」の上界として契約に確立されていない。
- D101.3 の #2–#9 は OPEN / NOT ESTABLISHED のまま。

---

## 4. 判定: INCOMPLETE

| 判定 | 適用条件 | 本件 |
| --- | --- | --- |
| PROVABLE | コード上の有限上限から $M$ を数学的に導出可能 | ✗ 該当せず |
| UNPROVABLE | producer/burst/concurrency に有限上限がなく、有限 $M$ を保証できない | △ 上限は見つからないが「存在しない」とまでは証明されていない（ランタイムプールの暗黙上限の可能性） |
| **INCOMPLETE** | **必要な bound は存在するが、まだコード/契約上の証拠が不足** | **◯ 該当** |

**理由**:

- アトミック性、read-order skew、名目 100ms 間隔、イベント駆動リファレンス（$T_w = B_{\max}^{\text{true}}$）はコードで確認済み。
- しかし $M$ を有限に導出するために必要な bound（publish レート $\lambda$、burst $\tau_b/\mu_{\text{burst}}$、並行度上限、保証間隔 $G$）がコード/契約に存在しない。
- 退化したプールサイズ上界（$M \le N$）が存在し得るが、それは意図する $M$（観測誤差の上界）として契約確立されておらず、1 区間 acquire 数を束縛しない。
- したがって「必要な bound は存在するが、コード/契約上の証拠が不足」→ **INCOMPLETE**。

---

## 5. 次のゲート

- ユーザー規定のゲート: **PROVABLE のみ** D101-2 へ進む。UNPROVABLE なら Phase I GO 判定停止。INCOMPLETE は継続（証明構築の続行）。
- 本判定は INCOMPLETE のため、D101-2 には進まず、以下の証拠を補完すること：
  1. プロデューサ publish レート上限 $\lambda$ をコード/契約に確立（または「存在しない」ことを証明）。
  2. burst 継続時間 $\tau_b$ / burst レート $\mu_{\text{burst}}$ の設計上界を確立。
  3. プロデューサ並行度上限（同時 publish 数）を確立。
  4. サンプリング間隔 $G$ を「保証値」として契約に格納（名目 100ms からの乖離を束縛）。
  5. 上記から $M \ge \Delta$ を数学的に導出し、D101.3 #2–#9 を CLOSED にする。

---

## 6. ソースリンク

- `doc/work88/I4_DESIGN_CONTRACT.md` — D101.4 末尾に D101-0 監査リンク済み（verdict: INCOMPLETE）。
- `evidence/phase-d101-0-m-bound-mathematical-audit.md` — D101-0 監査証拠（verdict: INCOMPLETE）。
- D100.5 — $\Delta > 0$（観測誤差が正）の証明。
