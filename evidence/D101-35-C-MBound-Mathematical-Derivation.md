# D101-35-C — M-Bound Mathematical Derivation（設計・証明確定報告書）

- **実施日**: 2026-08-25
- **作業種別**: 数学的導出（read-only / ソースコード変更 **0** / 契約変更 **0** / 具体値決定 **なし**）
- **判定**: **PASS** — `K < ∞ ∧ G < ∞ ∧ λ < ∞ ⇒ M < ∞` を証明。**M の具体値は未採用**
- **基準**: ConvoPeq.md 17:21 再生成版（ソース未变更を確認 — D101-33-F コミット `5c84ec9`/`f39fcd3` と同一内容）
- **入力**: D101-35-A（観測モデル）/ B′（G_bound・λ_prod_bound の契約導入可能性 = CONDITIONAL 確定）

---

## 0. 定義の固定（本報告書で確定する記号）

```text
K  ≜ kIntentQueueCapacity = 4096        （構造定数・コードから取得済み Coordinator.h:693）
G  ≜ G_bound             （抽象有限定数 — event 発生 → 次回 samplerTick 反映までの最大遅延。
                            jitter / missed tick を含む。値は未決・有限性のみ要求）
λ  ≜ λ_prod_bound        （抽象有限定数 — 全 publication producer の submit レート上界。
                            値は未決・適用範囲は §5 で2モデル比較）
B(t)    ≜ t 時点の真の outstanding published World 数
A(t)    ≜ acquireObserved 累計（publish LP で即時 +1・event-driven 正確）
R_c(t)  ≜ worldReclaimCount 累計（storage 側・terminal 破壊で即時 +1）
R_o(t)  ≜ releaseObserved 累計（samplerTick で R_c の差分を反映・最大 1 tick 遅延）
O(t)    ≜ sampler estimate = signedWide(A(t)) − signedWide(R_o(t))
T_w     ≜ reference running max（event-driven・真値ピークの参照観測）
O_w(w)  ≜ window w の sampled windowMax
M       ≜ sup_w [T_w(w) − O_w(w)]        ← 導出対象の未観測 peak 上界
```

---

## 1. 補題（実装事実からの整理）

| 補題 | 内容 | 根拠 |
|---|---|---|
| L1 | A(t) は publish LP で即時更新されるため、**acquire イベントと A の増分は同時刻** | Commit.cpp:406-409 |
| L2 | R_c(t) は terminal 破壊で即時更新。R_o は tick でのみ R_c 差分を反映（遅延 ≤ 1 tick 間隔） | DDQ.h:145-157/:197-204、Timer.cpp:415 |
| L3 | tick 時の estimate は「その瞬間の B」に一致（微小 in-tick race を除く）。race は O を**上方向にのみ**逸脱させる（safe 方向） | A/R 両 load が同一 tick 内 |
| L4 | B(t) の増加は execute（queue からの pop + commit）によってのみ発生 | release は B を減らすのみ |
| L5 | 区間 (t₁, t₂] 中の execute 数 ≤ \|queue\|(t₁) + arrivals((t₁,t₂]) | 各 execute は queue から 1 個消費。arrivals は admission token 保持下での受理分（reject は acquire に至らない） |
| L6 | \|queue\|(t) ≤ K = 4096（任意時点） | kIntentQueueCapacity 定数 |

---

## 2. 主定理

### Theorem（M-bound finiteness）

```text
前提: K < ∞, G < ∞, λ < ∞（すべて抽象有限定数として契約导入）
      counter wrap なし（測定期間内・uint64 — counterWrapped flag が逸脱検知）

結論: M = sup_w [T_w(w) − O_w(w)] ≤ K + λ·G < ∞
```

### 証明

任意の window w と、その window 内の T_w 更新点 τ（= 直近の acquire event 後のピーク候補時点）を取る。
release は B を減らすのみのため、running max は必ず acquire 直後に更新される → τ は acquire event 時点と仮定してよい。

`t* ≜ τ 以前で最後に estimate を記録した tick` とする（τ − t* ≤ G、∵ G は event→次 tick 反映の最大遅延であり、τ 以降の次 tick までの区間も含めて G で押さえる）。

**(a) rise の bound**:
区間 (t*, τ] での B の増加 = この区間の execute 数。
L5 より:

```text
B(τ) − est(t*) ≤ executes(t*, τ] ≤ |queue|(t*) + arrivals((t*, τ])
               ≤ K + λ·(τ − t*)
               ≤ K + λ·G
```

（arrivals は admission token 保持・queue 受理されたもののみ。reject は obligation を
生成しないため acquire にもカウントされない。）

**(b) O_w 側の下押さえ**:
windowMax は window 内の tick estimate の running max であり、`O_w(w) ≥ est(t*)`
（t* は window 内の tick であるため、その estimate は windowMax 候補に含まれる）。

**(c) 合成**:

```text
T_w(w) ≥ B(τ)
O_w(w) ≥ est(t*)
∴ T_w(w) − O_w(w) ≤ B(τ) − est(t*) ≤ K + λ·G
```

w と τ は任意 → sup をとっても不等号は維持される:

```text
M = sup_w,sup_τ [T_w − O_w] ≤ K + λ·G ≤ 4096 + λ·G < ∞   ∎
```

### Case B（T_build_min 不仮定）の明示的閉鎖

証明中で execute 数を抑えるのに使ったのは **L5（backlog + arrivals）のみ**であり、
execute 1 回あたりの所要時間 T_build には依存しない。

- T_build → 0 の極限でも、区間内 execute 数は「backlog K + 到着数 λG」に抑制される
  （backpressure: queue full → reject → obligation 不发生）。
- よって「CoordinatorLoop が直列だから有限」という誤った推論を用いずとも、
  **Case B のまま有限性が成立する**。∎

（参考: T_exec の正の下限を追加で仮定すれば bound は更に締まるが、本証明には不要。）

---

## 3. 項の再整理（二重計上の排除）

| 項 | 取扱い | 根拠 |
|---|---|---|
| `f_gap(G, λ)` | ≔ **λ·G**（gap 中の新規到着による rise 分）とし、K 項（backlog drain 分）と合算して `K + λG` | §2 証明の (a)。独立項を立てると二重計上になるため統合 |
| `M_burst` | ≤ K + λ·G（長時間 burst は同一 bound の繰り返し適用で被覆） | §2 |
| `M_boundary` | **= 0（M_gap/M_burst に吸収）** | D95 固定点 1: window Start の baseline は 0 リセットではなく現在値を引き継ぐため、境界での情報消失が発生しない。proof obligation: baseline continuity の維持（将来変更禁止項目） |
| `M_jitter` | **M_gap に吸収**（G の定義に jitter/missed tick を含済み） | §1 の G 定義 |

```text
確定形:
    f_gap(G, λ) := λ·G
    M ≤ K + λ·G = 4096 + λ_prod_bound × G_bound
```

（より緩い保守形として M ≤ K + 2·λ·G を分離表記する変種も成立するが、
§2 の厳密導出により 1 倍で十分 — 二重計上なしを §3 表が保証。）

---

## 4. λ スコープ 2 モデルの数学的比较

| Model | λ の対象 | 有限性 | 境界のタイトネス |
|---|---|---|---|
| **Model 1（conservative 全包含）** | 全 publication producer（内部自動系 Timer #5 / rebuild 完了駆動 / Recovery gate 済み + 外部 flow 系 4 site）を単一 λ で拘束 | ✅ M < ∞ | 緩い（内部自動系の構造 bound と外部契約を足し上げた値が必要）。ただし**決定事項は λ の値のみ** |
| **Model 2（分離）** | 内部自動系を構造 bound として別枠固定（Timer #5 ≤ 1/tick ≒ 10/s は timerCallback 構造から従う / Recovery は ShuttingDown gate + tryAdmit で閉鎖済み）、外部/フロー系 4 site のみ λ_ext で拘束 | ✅ M < ∞ | 境界がタイト。ただし内部経路の構造 bound を各々証明として維持する義務が生じる |

**数学的結論: 両モデルとも `M < ∞` を導く。差は境界のタイトネスのみであり、
有限性証明の成立性に影響しない。** モデル選択（どちらを契約に採用するか）は
D101-35-D 以前の意思決定事項として持ち越し。

---

## 5. 有限性の成立条件と未確定事項（最終整理）

```text
【有限性の成立条件】
    G_bound < ∞   … environment contract（message thread scheduling premise）
    λ_prod_bound < ∞ … producer submission rate contract

【未確定（D101-35-D 以前の意思決定事項）】
    ① λ_prod_bound の適用 producer scope（Model 1 vs Model 2）
    ② G_bound の environment contract wording
    （③ enforcement watchdog を将来実装するか否か — M 導出には不要）

【確定済み】
    K = 4096（コード定数）
    M_boundary = 0（吸収）
    M_jitter ⊆ M_gap（吸収）
    M ≤ 4096 + λ_prod_bound × G_bound < ∞
```

---

## 6. Case C / shutdown・quarantine の寄与（再確認）

- **Case C（admission vs close race）**: closeAdmission 後は tryAdmit 失敗 → 新規 obligation
  不发生（D101-33-C/D で実装・検証済み）。shutdown 区間で U は単調非増加。
- **quarantine / delayed retirement**: World の滞留**時間**を延ばすのみで、peak は
  acquire burst 時点で決まる。residency counters（X5/X6）は drain 判定用であり
  M の加算項ではない（D101-32-F authority separation 維持）。
- **shutdown drain**: producer 停止後 U 単調非増加。cursor 方式により drain 中の release も
  最終 tick までに反映可能。

→ **追加項なし**（D101-35-A §8 の結論を本証明の枠組みで再確認）。

---

## 7. PASS 条件チェックリスト

* [x] `G_bound` を抽象有限定数 `G` として固定（100ms cadence を bound としない）
* [x] `λ_prod_bound` を抽象有限定数 `λ` として固定（具体値不決定・2 モデル比較）
* [x] backlog 項を `K = 4096` として厳密分離（K から rate bound を導出しないことを明記）
* [x] `M_gap := f_gap(G, λ)` を定義し `M ≤ f_gap + K + λG` を導出
* [x] `M_boundary = 0` / `M_jitter ⊆ M_gap` の非再加算を proof obligation 化
* [x] 有限性のみを証明（`K,G,λ < ∞ ⇒ M < ∞`）
* [x] R_required / B_max^true の具体化 / D102 GO-NO-GO は未実施（D101-35-D へ持ち越し）
* [x] Case B を明示的に閉じる（T_build_min 不仮定で証明完了 — §2）
* [x] `onAcquireObserved` が execute/commit tail にある前提を維持（Commit.cpp:406-409 再確認）
* [x] コード変更 0 / 契約変更 0 / 具体値決定 0

# VERDICT: D101-35-C = **PASS**

---

## 8. 次ステップ: D101-35-D

```text
D101-35-C PASS（本報告書）
      ↓
【意思決定】 λ スコープ（Model 1 / Model 2）+ G_bound 前提文言
      ↓
D101-35-D: R_required / B_max^true 導出 + D102 GO-NO-GO 判定
      ↓
（GO の場合）Phase I NO-GO 解除判断へ
```

注意: D101-35-C の証明は「有限契約入力を与えれば M < ∞」までである。
G_bound / λ_prod_bound の**実際の値**が確定してはじめて M の数値が得られ、
D102 の R_required = ceil(M / O_w) が計算可能になる。
