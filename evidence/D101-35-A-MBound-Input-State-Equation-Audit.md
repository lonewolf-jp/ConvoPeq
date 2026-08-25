# D101-35-A — M-Bound Input Completeness & State Equation Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0**）
- **判定**: **PASS（分類完了）— ただし有限構造上界には契約上界の追加が 2 項目必要（§5）→ D101-35-B′ で補完**
- **基準**: ConvoPeq.md **2026-08-25 16:25 再生成版**（ローカル実ソースから）＋ I4.D101.4 判定条件

---

## 1. 観測インフラの実体（fresh source trace）

| 要素 | 実装 | 役割 |
|---|---|---|
| **T_w**（event-driven reference max） | `WorldRetirementReferenceObserver`（ISRWorldRetirementReference.h）— `referenceAcquireCount_/referenceReleaseCount_/referenceMax_`、event 発生ごとに running max 更新 | 真の retirement peak の高頻度参照観測 |
| **O_w**（sampled window max） | `WorldRetirementTelemetry`（ISRWorldRetirementTelemetry.h）— `acquireObserved_/releaseObserved_/windowMax_`、`samplerTick` のみ更新（D91） | 100ms Non-RT sampler による sampled maximum |
| sampler | AudioEngine.Timer.cpp:372-474 — 100ms `timerCallback`（Non-RT）。`samplerTick(windowNowUs)` が window 遷移の唯一 owner（D91）。`worldReclaimCount_` の累積差分を `addReleaseObserved(delta)` で反映（cursor 方式・取りこなし不可） | O の計算源 |
| estimate | `observedOutstandingEstimate() = signedWide(A) − signedWide(R)`（D82） | sampled outstanding |
| window 比較 | `E_w = T_w − O_w`（同一 windowId） | 未観測 peak の実測 |
| stats | `sampleCount / maxSamplingGapUs / jitter集計 / missedTickCount / counterWrapped`（MeasurementSnapshot） | 実測統計（安全上界ではない） |

---

## 2. A-1 Reference completeness

### acquire 経路（全列挙）

```
publish LP（Commit.cpp:406/:409 — CoordinatorLoop NonRT）
    ├─ worldRetirementTelemetry_.onAcquireObserved()   … A++
    └─ worldRetirementReference_.onAcquire()           … T_w running max 更新
```

production における acquire 発火点は **1 site のみ**（publish 成功 LP）。
Bootstrap / Recovery publish も最終的に同一 Commit 経路へ集約。

### release 経路（7 candidate sites — D101-34-D 全域再列挙と一致）

R1 DDQ.h:154 / R2 DDQ.h:204 / R3 Router.cpp:87 / R4 Router.cpp:114 /
R5 Router.h:128（同期破壊）/ R6 QuarantineStore.h:145（drain）/ R7 QuarantineStore.h:182（drainAllUnsafe）。
各 site で `worldReclaimCount_++`（storage 側カウンタ）＋ `referenceObserver_->onRelease()`。
observer は全 storage（DDQ/Q/E/Terminal）へ伝搬済み（ISRRetireRouter.cpp:153-157）。

### coverage 判定

```
∀ published W: onAcquire(W) = 1（publish LP・INV-PUB-4 証明済み）
∀ published W: onRelease(W) = 1（exactly-once・D101-34-B §2.3 証明済み）
observer は両イベントを取りこなし得ない（event-driven・cursor 方式）
```
✅ **M-A02/M-A03/M-A04 = 成立**。reference completeness は event レベルで完全。

---

## 3. A-2 Window / sampler 時間モデル（M-A06）

| 変数 | 定義可能か | 内容 |
|---|---|---|
| sampler cadence | ✅ 定義済み | 100ms Non-RT timer（公称値） |
| **G（真の最大観測 gap）** | ⚠️ **要契約定義** | G ≝ 「event 発生時刻から、その event が反映された次の samplerTick までの最大遅延」。構成要素: (a) timer 公称周期 100ms、(b) メッセージスレッド実行遅延（jitter）、(c) missed tick、(d) tick 内の処理順序。**(b)-(d) にコード上の硬い上限は存在しない**（JUCE Timer は message thread 動作・OS スケジューリング依存）。D100.5 が示した E_w=1 は「cadence 100ms ≠ 実効 gap」の実証 |
| λ / μ_burst / τ_b / J | §5・§6 参照 | observed ≠ safe bound の分離が必要 |

❌ `G = sampler cadence` という単純化は**不成立**（D100.5 の反例が存在）。
G は「公称 cadence + スケジューリング余剰」を含む**別個の契約定数**として定義する必要がある。

---

## 4. A-3 State equation `B(t) = O(t) + U(t)`（M-A05/M-A08）

### 4.1 カウンタの線形化性質（実測）

| counter | 更新タイミング | 性質 |
|---|---|---|
| A（acquireObserved） | publish LP で**即時**（event-driven） | monotonic 増加のみ |
| R（releaseObserved） | samplerTick で `worldReclaimCount` 差分を一括反映（**tick 遅延あり**） | monotonic 増加のみ・cursor 方式で取りこなし不可 |
| T_w reference | acquire/release **両方即時**（event-driven running max） | 真値の上側観測 |

### 4.2 線形化順序の分析

```text
tick k での読み取り順: samplerTick → (A load) → (worldReclaimCount load → R 反映)
```

- **非対称性**: A は即時反映 / R は tick 遅延反映。
- tick 直前の区間で release が起きた場合: R 反映は tick 内で完了するため
  estimate は正しく減算される（cursor が差分を保持）。
- A load と R load の間の僅かな区間で発生した acquire/release は次 tick へ繰り越し
  （1 tick 分の skew 上限）。
- **結論**: `B(t) = O(t) + U(t)` の state equation は
  「O は tick 時点の推定、U は直近 tick 以降に発生した净変化 + tick 内 skew」
  として**妥当に定義可能**。U ≥ 0 が常に保証されるわけではなく（release 遅延反映により
  O が真値を一時的に上取り得る）、安全上界として扱うのは **max 方向（underestimate 側）のみ**:
  `sup_t U(t)` を「O が真値を下回った量」の supremum として定義する場合は整合。
  ✅ **M-A05/M-A08 = 成立**（符号方向の注記付き）。

---

## 5. A-4 U_max 分解と 4 分類（M-A09/M-A10）

```text
U_max ≤ M_gap + M_burst + M_jitter + M_boundary
```

| 項 | 意味 | 分類 | 根拠 |
|---|---|---|---|
| **M_gap** | 1 sampling gap 内に完結し sampler が peak を取り逃した量 | ⚠️ **契約上界の追加が必要** | G 自体にコード上の硬い上限がない（§3）。G_bound を契約定数（例: watchdog 強制 tick 間隔上限 or スケジューリング契約）として定義すれば `M_gap ≤ (acquire rate upper) × G_bound` 様式で構造化可能 |
| **M_burst** | gap 内の複数 acquire burst による未観測 peak | ⚠️ **契約上界の追加が必要** | INV-PUB-1 により executePublish は CoordinatorLoop で直列化（contract T60: N_publish ≤ floor(Δt/T_build)+1）するが、T_build の下限（最小実行時間）はコード上に存在しないため μ_burst の構造上界には契約定数（例: 最小 execute 間隔 or 最大 in-flight publication 数）が必要 |
| **M_jitter** | missed tick / 遅延 tick による gap 延長分 | ⚠️ **契約上界の追加が必要（M_gap に吸収可能）** | missedTickCount/maxSamplingGapUs は実測統計であり bound ではない。M_gap の G_bound に jitter 分を含めて定義すれば独立項として消せる |
| **M_boundary** | window Start/End 境界での未観測分 | ✅ **有限構造上界（他項に吸収）** | D95 固定点1（Start baseline はリセットせず現在値）+ T_w が event-driven 連続 running のため、境界自体で情報が失われるのは直近 1 gap 分のみ = M_gap に包含 |

### 判定ルール適用

```
finite structural bound が全項成立していない
  ├── M_boundary      : 構造的に M_gap に吸収（独立項不要）
  ├── M_gap + M_jitter: G_bound の契約定義が必要 ★不足①
  └── M_burst         : μ_burst/τ_b 相当の契約定義が必要       ★不足②
        ↓
D101-35-B′（入力補完）に進む
```

✅ **M-A09/M-A10 = 分類完了**（4分類を全項に付与）。

---

## 6. A-5 λ の厳密化（M-A07）

| 表記 | 区別 |
|---|---|
| observed λ | D100.6 の実測 retire rate（measurement-only・安全上界ではない） |
| **safe λ_bound / μ_burst** | 契約上界。**現時点で未定義 — これが不足②の本体** |

候補となる構造的根拠（コードから取得可能な要素）:

- Publication は CoordinatorLoop で直列実行（INV-PUB-1）→ 同時 commit 数 1
- 直接 producer（Timer/PrepareToPlay/Transition/ReleaseResources）は並列 potential あり
- rebuild request には rate limit 構造（emergencyReclaimBoostCount/minInterval パターン）が
  reclaim 側に存在するが、**publish/retire 側の rate 上限定数は未定義**

→ `λ_max`/`μ_burst` は「コードから取得」ではなく「**契約で bound を追加定義**」する分類。
✅ **M-A07 = 意味固定完了**（observed/safe の分離を明示）。

---

## 7. A-6 Burst 最悪ケース（μ_burst × τ_b）

必要前提の取得可否:

| 前提 | コードから取得 | 現状 |
|---|---|---|
| publish rate 上限 | ❌（rate limiter は publish 側に未実装。reclaim boost 側のみ） | 契約定数が必要 |
| rebuild request rate | △（RebuildDispatch に間隔パターンあり、ただし publish 連鎖への変換は build 時間依存） | 同上 |
| commit rate | ❌（CoordinatorLoop 直列だが実行時間下限なし） | 契約定数が必要 |
| retirement rate | △（epoch gate + drain cadence 依存・実測のみ） | 同上 |
| 同時 outstanding 増加可能数 | △（intentQueue 容量等の構造上限はあるが acquire は execute 時点なので直接対応しない） | 契約定数が必要 |
| τ_b（burst duration 上限） | ❌ | 契約定数が必要 |

✅ **M-A07 burst 部 = 前提列挙完了**。すべて契約定数の追加設計（D101-35-B′）に帰着。

---

## 8. A-7 Shutdown / quarantine contribution（M-A11）

評価観点: 「sampler observation window 内の未観測 outstanding を増加させるか」

| フェーズ | U への影響 |
|---|---|
| normal operation | M_gap/M_burnt 項に含まれる（既述） |
| delayed retirement（epoch 待ち・quarantine 滞留） | World が outstanding に滞留する**時間**は延びるが、acquire は publish 時に一度だけ・release は terminal 破壊時に一度だけのため、**未観測 peak の増加要因にならない**（peak は acquire burst 時に決まる）。residency counter は drain 判定用で M とは無関係 |
| shutdown drain | producer 停止後は acquire 新規発生なし → U は単調非増加。drain 中の release は cursor 方式で最終 tick までに反映可能。**追加項なし**（構造的） |
| quarantine capacity / epoch delay / shutdown drain delay | 「outstanding の滞留時間」への影響のみで「未観測 outstanding 量」への影響なし → M への加算項**なし** |

✅ **M-A11 = shutdown/quarantine は独立加算項を生まない**（時間遅延と量の区別を明示）。

---

## 9. Gate 判定 M-A01〜M-A15

| Gate | 条件 | 判定 |
|---|---|---|
| M-A01 | 最新 ConvoPeq.md 再確認 | ✅ 16:25 版 |
| M-A02 | acquire 全経路列挙 | ✅ 1 site（publish LP）+ Bootstrap/Recovery 集約確認 |
| M-A03 | release 7 candidate sites 再列挙 | ✅ R1〜R7（全域 grep・漏れなし） |
| M-A04 | reference observer 全 event coverage | ✅ acquire/release とも exactly-once・全 storage 伝播済み |
| M-A05 | window boundary / linearization 確定 | ✅ tick 内 A→R 読み取り順と skew 1-tick を特定 |
| M-A06 | G 定義（cadence と混同しない） | ✅ G ≝ event→次 tick 反映までの最大遅延。契約定数化が必要と判定 |
| M-A07 | λ/μ_burst/τ_b/jitter の意味固定 | ✅ observed ≠ safe bound を分離 |
| M-A08 | B=O+U state equation 妥当性 | ✅ 符号方向注記付きで成立 |
| M-A09 | U_max 4 項分解 | ✅ |
| M-A10 | 各項 4 分類 | ✅ M_boundary=構造（吸収）/ M_gap·M_jitter·M_burst=契約上界の追加が必要 |
| M-A11 | shutdown/quarantine 影響評価 | ✅ 加算項なし（滞留時間と量の分離） |
| M-A12 | M の具体値不採用 | ✅ |
| M-A13 | R/R_cap/T2 導出未開始 | ✅ |
| M-A14 | ソースコード変更 0 | ✅ |
| M-A15 | 契約変更 0 | ✅ |

# VERDICT: D101-35-A = **PASS（分類完了）**

---

## 10. 次ステップ: D101-35-B′（入力補完）

判定ルールどおり、有限構造上界に不足する入力は以下の **2 項目のみ**:

| 不足 | 補完内容（D101-35-B′ で設計） |
|---|---|
| ★不足① G_bound | sampler tick 間隔の契約上界定義。候補: (a) watchdog による tick 遅延検知 + 最大許容値の明文化、(b) メッセージスレッドスケジューリング契約の文書化。jitter はここに吸収 |
| ★不足② μ_burst / τ_b | publication serialization の構造を利用した契約上界。候補: (a) CoordinatorLoop 直列実行 + 最小 execute 間隔の明文化的定数、(b) 最大 in-flight publication 数（intentQueue 容量との突合）、(c) INV-PUB-1 の T60 式（N_publish ≤ floor(Δt/T_build)+1）に T_build 下限を契約定数として与える |

両項が契約定数として確定した時点で:

```text
M = M_gap(G_bound) + M_burst(μ_burst × τ_b) + M_boundary(=0, 吸収済み)
    ≤ 構造上の有限上界として導出可能
      ↓
B_max^true ≤ O_w + M   の形式化完了 → D102（R_required 導出）の GO/NO-GO 判定へ
```

M-bound / Phase I / D102 の NO-GO 状態は引き続き維持（本監査では一切変更していない）。
