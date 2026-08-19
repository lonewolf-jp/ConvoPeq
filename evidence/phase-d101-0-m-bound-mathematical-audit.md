# D101-0: M-Bound Mathematical Audit

- **日付**: 2026-08-20
- **対象**: `doc/work88/I4_DESIGN_CONTRACT.md` D101 セクション（`B_max^true ≤ B_max^observed + M` の数学的バインド）
- **目的**: M（安全偺上界）の数学的バインドが PROVABLE / UNPROVABLE / INCOMPLETE のいずれかで判定する。**production code は変更しない**（監査のみ）。
- **判定**: **INCOMPLETE** — 数学的フレームワーク（D101.1-D101.4）は仕様化されているが、証明は未完了。INV-PUB-3 が DISPROVEN（counterexample 確認済み）・Tier 4（published-domain exclusion）が OPEN・Step 2（API separation）が NOT STARTED。

---

## 1. D82 の式を正式な契約として再構成

### 1.1 形式的契約

$$B_{\max}^{\text{true}} \leq B_{\max}^{\text{observed}} + M$$

| 記号 | 意味 | 定義場所 | 状態 |
| --- | --- | --- | --- |
| $B_{\max}^{\text{true}}$ | 数学上の真値（観測不能） | D101.1 | — |
| $B_{\max}^{\text{observed}}$ | 100ms sampler の観測最大値（O_w） | D82.3 / D101.1 | CLOSED |
| $M$ | safety margin（policy headroom） | D101.2 | **OPEN（未導出）** |
| $G$ | sampler の最大観測間隔 | D101.1 | CLOSED |
| $\mu_{\text{burst}}$ | retirement burst の event rate | D101.2 | CLOSED（候補） |
| $\tau_b$ | burst duration | D101.2 | CLOSED（候補） |

### 1.2 A/R counter の数学的モデル（D101.1）

```text
B_true(t)  = A_true(t)  - R_true(t)   — 数学上の真値（観測不能）
B_ref(t)   = A_ref(t)   - R_ref(t)    — reference observer が観測した値
B_obs(t_k) — sampler が tick 時刻 t_k に観測した値
```text

- $T_w = B_{\max}^{\text{reference}} = \sup_t B_{\text{ref}}(t)$ — **真値ではない**（D100.5）・reference observer は観測器
- $O_w = \max_k B_{\text{obs}}(t_k)$ — 100ms sampler の観測最大値
- $G = \max_k (t_{k+1} - t_k)$ — sampler の最大観測間隔

### 1.3 A/R 算術契約（D82.1-D82.2）

- **O = signedWide(A) - signedWide(R)** — unsigned subtraction の wraparound を回避（int64 に cast 後減算）
- **A/R の monotonic unsigned は実行期間中に wraparound しない**（uint64・2^64 回 increment に要する時間 ≫ 測定期間）
- **measurement-duration contract**: 測定期間は A/R が wraparound しない範囲に制限

### 1.4 export 命名（D82.3）

- `observedOutstandingEstimate` / `observedOutstandingMax` — T1 telemetry（authority の `outstanding` / `held_count` と明確に区別）
- `referenceMax`（T_w）— reference observer（実測時のみ・D94）
- `Ew`（E_w = T_w - O_w）— **診断のみ**（M の安全偺根拠にしない）

## 2. (M) の候補導出（D101.2）

### 2.1 M の数学的定義

$$M \geq \sup_k \Delta_k^{\text{growth}} + E^{\text{obs}}$$

| 項目 | 定義 | 説明 |
| --- | --- | --- |
| $\Delta_k^{\text{growth}}$ | $\max_{t \in [t_k, t_{k+1}]} [B_{\text{true}}(t) - B_{\text{true}}(t_k)]$ | 観測間隔内の真値増加量（interval growth） |
| $E^{\text{obs}}$ | $\sup_k [B_{\text{true}}(t_k) - B_{\text{obs}}(t_k)]$ | sampler が安全側に underestimate する可能性のある有限上界 |
| $\mu_{\text{burst}} \cdot G$ | candidate upper bound for $\Delta_k^{\text{growth}}$ | burst rate × sampler interval |

### 2.2 導出の鎖

```text
B_true(t) - B_obs(t_k)
  = [B_true(t) - B_true(t_k)] + [B_true(t_k) - B_obs(t_k)]
  ≤ Δ_k^growth + ε_k^obs
  ≤ Δ_k^growth + E^obs
```text

- $\sup_k \Delta_k^{\text{growth}}$ は D101.3 #2-#8 の envelope から
- $E^{\text{obs}}$ は D101.3 #1（reference completeness）から有限になることを証明

### 2.3 禁止される M の定義

| 禁止される定義 | 理由 |
| --- | --- |
| $M = \max(E_w)$ | D94.5/D100.5: 有限回の実測最大値は安全偺上界ではない |
| $M = \text{observed maximum}$ | 実測値は未観測 peak をカバーしない |
| $M = 1$ | E_w=1 は「O_w ≠ T_w が存在する」の証明にすぎない |
| $M = \text{sampling interval 内の observed A/R}$ | rate が observation に依存して循環論法 |
| $M = \text{sampling interval} \times \text{observed rate}$ | rate 自体が observation に依存（循環論法） |

> ⚠️ **M は測定値ではなく、実装上保証された envelope から導出される安全偺上界。**
> 実測 $E_w = 1$ から $M = 1$ を決定づめることは**禁止**（D94/D95）。

## 3. burst/jitter test が M の証明に使えるか（D100）

### 3.1 実測結果

| 条件 | O_w | T_w | E_w | sampleCount | maxSamplingGapUs | missedTick |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 3 | 4 | 1 | 21 | 100959 | 0 |
| burst | 3 | 4 | 1 | 41 | 101132 | 0 |
| jitter | 13 | 13 | 0 | 19 | 101102 | 3 |

### 3.2 証明能力の評価

| 観測 | 結果 | M の証明に貢献するか |
| --- | --- | --- |
| E_w > 0（normal/burst） | ✅ 実証済み | ❌ — O_w ≠ T_w の存在証明のみ |
| T_w ≥ O_w（全条件） | ✅ 実証済み | ❌ — reference が sampler 以上に観測することのみ |
| sampler gap / missed tick（jitter） | ✅ 観測済み | ❌ — G の実測値だけ（上界証明ではない） |
| burst duration / event rate | ⚠️ 部分的 | ❌ — 1 burst condition（20 publish / 150ms） |
| reclaim latency | ❌ 未観測 | ❌ — D101.3 #7（delayed release） |

### 3.3 結論

- burst/jitter test は **「測定器が動く」ことの検証**にはなる（D100.5: E_w > 0 実証）
- しかし、**全実行に対する上界の証明にはならない** — 有限回の実測最大値は未観測 peak をカバーしない
- **finite observation の最大値を mathematical upper bound と誤認しない**（D93.3: windowMax は sampled 下側の観測量）

## 4. wraparound / saturation の確認（D82.2 / D91.9）

### 4.1 counter wraparound

| 項目 | 状態 | 説明 |
| --- | --- | --- |
| A/R counter type | `std::uint64_t` | D82.2 CLOSED |
| wraparound period | 2^64 increments | 実用上到達不能（D82.2 measurement-duration contract） |
| measurement-duration contract | 測定期間は wraparound しない範囲 | CLOSED |
| `counterWrapped` field | `MeasurementSnapshot.counterWrapped` | D91.9 — 診断のみ（trigger にしない） |

### 4.2 saturation

| 項目 | 状態 | 説明 |
| --- | --- | --- |
| `observedOutstandingMax` | `std::int64_t` | D82.1 signedWide — int64 で十分な range |
| `referenceMax` | `std::int64_t` | D94 — signedWide |
| `windowMax` | `std::int64_t` | D91.8 — bounded sampled maximum |
| overflow risk | なし | R ≤ R_cap ≤ 4608（D46）・int64 で十分 |

### 4.3 結論

- **wraparound**: uint64 で実用上到達不能・measurement-duration contract で契約済み（CLOSED）
- **saturation**: int64 で十分な range・R ≤ 4608 で overflow なし（CLOSED）
- **問題なし** — wraparound/saturation は M の証明のブロッカーではない

## 5. M が導出可能か — D101 レビュー順序の現状（D101.3）

### 5.1 10-step review order

| # | ステップ | 内容 | 状態 |
| --- | --- | --- | --- |
| 1 | Reference completeness (4-tier) | B_ref(t) が B_true(t) を取りこぼさない 4 つの保証 | **Tier 1-3 CLOSED / Tier 4 OPEN** |
| 2 | State equation | B = A - R の状態方程式 | CLOSED（D101.1） |
| 3 | Sampler gap | G = sup_k(t_{k+1} - t_k) / missed tick / jitter / window boundary | CLOSED（D101.1 / D100） |
| 4 | Acquire/increase envelope | event rate / burst duration τ_b / μ_burst | **OPEN（D101.3 #4）** |
| 5 | Single burst bound | --- | **OPEN（D101.3 #5）** |
| 6 | Multiple acquire in one interval | E_w=1 を一般化する重要点 | **OPEN（D101.3 #6）** |
| 7 | Delayed release | reclaim latency が outstanding peak を増幅するケース | **OPEN（D101.3 #7）** |
| 8 | Shutdown / quarantine / deferred deletion | 通常経路以外も同じ bound に含める | **OPEN（D101.3 #8）** |
| 9 | Finite M proof | #1-#8 から sup_k Δ_k^growth と E^obs が有限になることの証明 | **OPEN（D101.3 #9）** |
| 10 | D102 gate | finite M が証明できれば B_max^true ≤ O_w + M が安全保証として成立 | **D102 未作成** |

### 5.2 Tier 4 — No hidden World lifetime path（D101.3.1 #4）

| 項目 | 状態 | 説明 |
| --- | --- | --- |
| 通常 publish/replace/shutdown path | ✅ CLOSED | RuntimeStore::publish() → publishAndSwap(oldWorld) → retire/deferred delete 一本化 |
| published-domain exclusion | 🔴 OPEN | RuntimeStore destructor は current を delete しない — "current == nullptr at destruction" を shutdown contract として証明する必要あり |
| failure/retry path | 🔴 OPEN | OwnerChannel/Intent queue enqueue failure, validation rejection は unpublished World → R_ref に入れない exclusion が**未形式化** |
| quarantine path | 🔴 OPEN | 4 terminal release sites 網羅済みだが、"quarantined World は published domain に属する" を type/state contract として証明 pending |
| WorldState predicate | 🔴 OPEN | Built→Owned→Published→Retired→Quarantine→Reclaimed の published-domain membership をコード上で証明 pending |

### 5.3 INV-PUB-3 — DISPROVEN（D101 #1 / I4.D101）

| 項目 | 状態 | 説明 |
| --- | --- | --- |
| Producer completeness | ✅ CLOSED | World producer = 1 site（AudioEngine.h:3534） |
| Counterexample | 🔴 DISPROVEN | `Init.cpp:67 — rejectedWorld → retireRuntimePublishWorldNonRt → World → onRelease` |
| 問題 | rejected (unpublished) World が World deletion path に入り onRelease() を呼び出す | B_ref = A_ref - R_ref の意味を破す |
| Step 2 (API separation) | 🔴 NOT STARTED | `retirePublishedRuntimeWorldNonRt` / `retireRejectedRuntimeWorldNonRt` の分離設計のみ（実装未着手） |

### 5.4 結論

- **D101.1-D101.2**: CLOSED（state equation / M definition は仕様化済み）
- **D101.3 review order**: CLOSED（10-step proof 構造は定義済み）
- **D101.3.1 reference completeness**: Tier 1-3 CLOSED / **Tier 4 OPEN**（published-domain exclusion proof pending）
- **D101 #1 producer completeness**: Step 1 CLOSED / **Step 2 NOT STARTED**（API separation 未実装）
- **INV-PUB-3**: **DISPROVEN**（counterexample 確認済み）
- **D101.3 #4-#9**: 全て OPEN（envelope 導出 / reclaim latency / finite M proof 未着手）
- **D102**: **未作成**（D101 が PROVABLE になった時に初めて作成）

## 6. reclaim latency instrumentation の監査（D101.3 #7）

### 6.1 現状

| 項目 | 状態 | 説明 |
| --- | --- | --- |
| reclaim latency | ❌ 未観測 | D101.3 #7（delayed release）— release イベント（terminal deleter 成功）から実際の destroy までの時間 |
| instrumentation | ❌ 未実装 | AudioEngine.Commit.cpp の evidence export は acquire/release カウンタのみ（reclaim latency なし） |
| 影響 | M の証明の最大のギャップ | reclaim 遅延が outstanding peak を増幅するケース（delayed release）の bound がない |

### 6.2 必要な instrumentation（監査のみ・実装はしない）

| 必要な観測 | 現状 | 備考 |
| --- | --- | --- |
| release event → deleter 実行までの時間 | ❌ 未観測 | D101.3 #7 |
| worst observed reclaim latency | ❌ 未観測 | D101.3 #7 |
| sampler window との関係 | ❌ 未観測 | D101.3 #7 |
| reclaim latency の upper bound | ❌ 未証明 | D101.3 #7 |

### 6.3 結論

- reclaim latency instrumentation は **D101 の証明に必要**（D101.3 #7）
- この段階では **instrumentation の production 実装を開始しない**（監査のみ）
- 必要な観測項目を明文化した上で、D101 が PROVABLE になった時に実装する

## 7. 判定: **INCOMPLETE**

### 判定理由

1. **D101.1-D101.2（state equation / M definition）は CLOSED** — 数学的フレームワークは仕様化されている。
2. **D101.3（review order）は CLOSED** — 10-step proof 構造は定義済み。
3. **しかし証明は未完了**:
   - **INV-PUB-3 が DISPROVEN** — rejected (unpublished) World が World deletion path に入り onRelease() を呼び出す counterexample が確認されている（I4.D101）。
   - **Tier 4（no hidden World lifetime path）が OPEN** — published-domain exclusion proof が未完了。
   - **Step 2（API separation）が NOT STARTED** — counterexample の修正のための API 分離が未実装。
   - **D101.3 #4-#9 が全て OPEN** — envelope 導出 / reclaim latency / finite M proof が未着手。
   - **D102 が未作成** — M が PROVABLE になった時に初めて作成されるゲート。
4. **M は導出可能ではない** — D101.4 の判定式によれば、M が構造上の上界として導出可能な場合のみ R_required が導出可能。現状では M は NO-GO。
5. **burst/jitter test は M の証明にはならない** — 有限回の実測最大値は未観測 peak をカバーしない（D93.3 / D94.5）。
6. **reclaim latency は完全に未観測** — D101.3 #7（delayed release）が OPEN。

### R_required 導出のブロック要因

```text
R_required = R_baseline + B_max(T_stall) + M
                     │              │
                     │              └── D101 INCOMPLETE（M 未導出）
                     └── P99.9 baseline（sustained observation 不足）
```text

**M が構造上の上界として導出可能になるまで（D101.3 #1-#10 の全ステップ完了 + Step 2 API separation + D102 gate）、R_required は導出不能**。

## 8. 推奨アクション（production code 変更なし）

### 即座の次ステップ（D101 証明完遂）

1. **INV-PUB-3 counterexample の修正（Step 2: API separation）**
   - `retirePublishedRuntimeWorldNonRt(W)` — W ∈ PublishedDomain 前提
   - `retireRejectedRuntimeWorldNonRt(W)` — W ∉ PublishedDomain → DeletionEntryType::Generic
   - これにより `onRelease()` は published-domain World のみに呼び出される

2. **Tier 4（published-domain exclusion）の形式化**
   - WorldState predicate（Built→Owned→Published→Retired→Quarantine→Reclaimed）の published-domain membership をコード上で証明
   - RuntimeStore destructor の "current == nullptr at destruction" を shutdown contract として形式化
   - failure/retry path の unpublished World exclusion を形式化

3. **D101.3 #4-#9 の証明完遂**
   - #4: Acquire/increase envelope（event rate / burst duration τ_b / μ_burst）
   - #5: Single burst bound
   - #6: Multiple acquire in one interval（E_w=1 を一般化）
   - #7: Delayed release（reclaim latency instrumentation + bound）
   - #8: Shutdown / quarantine / deferred deletion
   - #9: Finite M proof（sup_k Δ_k^growth と E^obs が有限になること）

4. **reclaim latency instrumentation（D101.3 #7）**
   - release event（terminal deleter 成功）から実際の destroy までの時間を測定する T1 instrumentation
   - worst observed reclaim latency の記録
   - sampler window との関係の分析

5. **D102 gate の作成**
   - D101 が PROVABLE になった時に初めて作成
   - `B_max^true ≤ O_w + M` が安全保証として成立 → R_required = ceil(M / O_w) の導出

### ゲート

- **R_required の確定**: D101 が PROVABLE になり D102 gate が CLOSED になった後
- **ReservationExhausted の実生成**: T2（R gate）実装後
- **Phase I 本実装**: ユーザー最終 GO 後

---

## 付録: 監査のスコープ

- **変更した production code**: なし（本監査は audit-only）
- **ゲート**: R_required の確定・ReservationExhausted の実生成・Phase I 本実装は行わない
- **参照**: `doc/work88/I4_DESIGN_CONTRACT.md`（D82, D93, D94, D100, D101, I4.D101）, `doc/work88/D2_IMPL_CHECKLIST.md`（Phase I ステータス）, `src/audioengine/ISRWorldRetirementTelemetry.h`, `src/audioengine/ISRWorldRetirementReference.h`, `src/audioengine/AudioEngine.Timer.cpp:435`, `src/audioengine/AudioEngine.Commit.cpp:723-760`, `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp`, `CMakeLists.txt:1682`
