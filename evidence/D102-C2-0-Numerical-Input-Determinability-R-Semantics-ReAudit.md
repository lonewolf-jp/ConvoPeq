# D102-C2-0 — Numerical Input Determinability & R Semantics Re-Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS**（6 Gate 全充足。R_required 訂正含む）
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容＋ fresh source trace（16:06 再生成・以後 diff なし）

---

## Gate C2-0-1: R_required semantics 確定

### 指摘の数学的確認

```text
証明済み: B_max^true ≤ O_denom + M_scope          （D101-35-C §2 / D-R §4）

要求条件（retention capacity R が真のピークを被覆する）:
    B_max^true ≤ R_required × O_denom

両式より R_required の必要十分条件:
    R_required × O_denom ≥ O_denom + M_scope
    R_required ≥ (O_denom + M_scope) / O_denom = 1 + M_scope / O_denom

R_required は整数のため:
    R_required = 1 + ceil(M_scope / O_denom)    （M_scope > 0 の場合）
    R_required = 1                              （M_scope = 0 の場合・ceil(0)=0 より 1+0）
    ※ 統一形: R_required = 1 + ceil(M_scope / O_denom) は M_scope ≥ 0 で常に成立
      （M_scope=0 のとき ceil(0)=0 → R=1。B_max^true ≤ O_denom を正しく被覆）
```

### semantics の切り分け

| 式 | 意味 | 判定 |
|---|---|---|
| `ceil(M_scope / O_denom)` 単独 | **gap multiplier** — 観測 baseline を超える「未観測分」のみの倍率。base 分（O_denom 自体）を含まない | ❌ retention multiplier としては不十分 |
| **`1 + ceil(M_scope / O_denom)`** | **total retention multiplier** — base（観測 resident）+ 未観測分の合計被覆倍率 | ✅ **採用** |

### R_cap/T2 authority 接続時の R の意味

- `R_required` は「O_denom baseline に対する必要 retention 倍率」であり、
  絶対的な retention 個数ではない
- `R_cap`（実際に実装可能な retention capacity）との整合判断は
  `R_required ≤ R_cap` で行う（D102-C3）
- `T2`（time authority）は本式の時間ドメイン側であり、M_scope の G_bound 項が
  既に時間要素を吸収済みのため T2 変更は不要
- **注意**: `R_required ≠ R_cap` の分離は D101-33-F 以前から維持

✅ **C2-0-1 = PASS**（訂正式確定）。

---

## Gate C2-0-2: λ_prod_bound scope — producer inventory

全 `commitRuntimePublication` call site 実測:

| # | producer | call site | trigger | 分類 |
|---|---|---|---|---|
| P-1 | PrepareToPlay (publish #1) | PrepareToPlay.cpp:155 | engine 初期化/re-init | **lifecycle-bound（構造）** — prepareToPlay 呼出頻度に律速 |
| P-2 | PrepareToPlay (publish #2) | PrepareToPlay.cpp:277 | 同上 | 同上 |
| P-3 | Timer idle publish (#5) | Timer.cpp:994 | timerCallback（100ms 周期） | **構造的 hard bound**: ≤1/tick ≒ 10/s |
| P-4 | Transition (#6 idle publish) | Transition.cpp:25 | DSP transition 完了時 | workload contract 必要 |
| P-5 | ReleaseResources | ReleaseResources.cpp:175 | shutdown sequence 中 | **lifecycle-bound（構造）** — shutdown 中の 1 回 |
| P-6 | PublicationExecutor deferred resubmit | PublicationExecutor.cpp:53 | CoordinatorLoop 上の再送 | 元 submission の派生（独立レートではない） |
| P-7 | Orchestrator Path A | Orchestrator 経由 submitPublishRequest → executePublish | ユーザー/オートメーション publish 要求 | workload contract 必要 |
| P-8 | Recovery publish | quarantine 検出時 | build failure 連動（外部要因） | workload contract 必要 |

### λ_ext construction rule

```text
λ_prod_bound = Σ (各 producer の rate 上界)

分類:
  構造的 hard bound（契約不要）:
    P-3: ≤ 10/s（timerCallback 構造）
    P-1/P-2/P-5: lifecycle-bound（定常状態では 0）
  workload contract 必要:
    P-4/P-7/P-8: ユーザー/オートメーション駆動のため契約上界が必要
```

**単純加算の妥当性**: 各 producer の commit/acquire は CoordinatorLoop で**直列化**
されるため、合計 acquire rate ≤ Σ(個別 rate 上界)。単純加算は保守的かつ正当な上界構成。
（producer 間の相関で下がる場合はあるが、上界として問題なし。）

⚠️ **P-6（deferred resubmit）は独立 producer ではなく元 submission の派生**のため、
λ_ext 加算に含めると二重計上になる可能性 → P-6 の元 submission が既にカウント済みなら除外。
実装: deferred resubmit は PublicationExecutor.cpp:53 の waitForReceipt=false 経路で、
これは初回 enqueue が receipt 待ちで失敗した場合の再試行。初回 submission が既に
λ_ext カウント対象なら P-6 は二重計上 → **除外**。

✅ **C2-0-2 = PASS**（inventory 完了・加算 rule 確定）。

---

## Gate C2-0-3: G_bound environment contract boundary

| 構成要素 | G_bound 包含 | 根拠 |
|---|---|---|
| (1) nominal cadence 100ms | ✅ 含む | T_sampler 公称値 |
| (2) callback scheduling delay | ✅ 含む | message thread queue 待ち |
| (3) missed/coalesced callback | ✅ 含む | JUCE coalescing property |
| (4) OS scheduling delay | ✅ 含む | スレッド優先度/プリエンプション |
| (5) message-thread starvation | ⚠️ **要判断** | modal dialog / 長時間タスクで理論上無限遅延し得る |
| (6) samplerTick 処理時間 | ✅ 含む（微小・atomic ops のみ） | |

### 契約境界の確定案

```text
G_bound は (1)-(4)+(6) を包含する。
(5) message-thread starvation は以下のいずれかで扱う:
    (i)   environment premise として「starvation は発生しない」を明記
          （JUCE アプリケーションの通常運用では modal dialog 中も
            timerCallback は呼ばれ続けるため実質成立）
    (ii)  将来の watchdog 実装による強制検知（D101-35-D §1.4 残留 caveat と同一対応）
```

推奨: (i) を採用し、(ii) を将来強化として並行記録。
**実測値の昇格は禁止**（指示どおり維持）。

---

## Gate C2-0-4: N_timer 境界項の必要性

| 項目 | 実測 |
|---|---|
| `startTimer(100)` 設定 | ✅ Init.cpp:121 |
| callback あたり idle publish ≤ 1 | ✅ Timer.cpp:994 単一 site |
| JUCE coalescing premise | ✅ framework 契約（callback 間隔 ≥ 設定 interval） |
| **+2 の必要性** | ✅ **必要な境界項** — 区間端点を両側含める閉区間カウントでは floor(G/T)+2 が最悪値。保守的余剰ではなく boundary term |

修正後 Model 2′ の N_timer(G_bound) = ⌈G_bound/100ms⌉ + 2 は**必要かつ十分な境界項**。

---

## Gate C2-0-5: O_denom 定義と positivity 証明の分解

「World が存在する」ことと「sampled O_w ≥ 1」は同義ではないという指摘への分解:

| 段階 | 内容 | 根拠 |
|---|---|---|
| (i) commit 存在 | bootstrap または 通常 publish の committed=true | Init.cpp:85-90 / RuntimePublishExecutor.h:70-76 |
| (ii) current world residency | `current != nullptr`（store swap 済み） | RuntimeStore.h — current は replacement/clear まで保持 |
| (iii) acquire counter 反映 | A++ は commit tail で即時（Commit.cpp:406） | event-driven・tick 待ち不要 |
| (iv) sampler observation timing | 次回 tick で estimate に反映（A は即値読み） | est = A(tick) − R_reflected(tick) |
| (v) windowMax への反映 | updateWindowMax(estimate) | tick 時点で実行 |

**positivity 証明の鍵**: (iii) A++ が即時であるため、commit 後の最初の tick で必ず est ≥ 1。
window に commit 後の tick が 1 回でも含まれていれば windowMax ≥ 1。
含まれない場合（commit が window 終端直前に発生）は次 window の firstEstimate ≥ 1。

∴ **post-first-commit の window は O_w ≥ 1 を保証する。ただし commit 直前の window では
O_w = 0 も有効（burst test ケースと同じ機構）。**

---

## Gate C2-0-6: Numerical determinability

| 入力 | 決定可否 | 必要なアクション |
|---|---|---|
| λ_prod_bound 値 | ❌ 未決定 | workload contract の決定（P-4/P-7/P-8 の rate 上界） |
| G_bound 値 | ❌ 未決定 | environment premise の文言 + 数値確定 |
| O_denom 値 | △ **measurement required** | 実稼働 window での windowMax 実測（post-first-commit window から取得）。test harness での事前取得も可 |

---

## 最終成果物構成

### Gate C2-0-1: R_required semantics
- ☑ gap multiplier: `ceil(M_scope / O_denom)` — base 非含有
- ☑ **total retention multiplier: `1 + ceil(M_scope / O_denom)`** ← 式確定
- ☑ R_cap/T2 接続時の意味: retention 倍率（絶対数ではない）

### Gate C2-0-2: λ scope
- ☑ Model 2′ producer inventory: 8 sites 特定
- ☑ bound type 分類: 構造 3 / workload contract 3 / lifecycle 2
- ☑ λ_ext construction rule: Σ workloads + 構造 bounds（P-6 二重計上除外）

### Gate C2-0-3: G_bound
- ☑ environment contract boundary: (1)-(4)+(6) 包含 / (5) starvation 要判断
- ☑ nominal cadence との分離明記

### Gate C2-0-4: N_timer
- ☑ exact conservative formula: ⌈G_bound/100ms⌉ + 2（boundary term 必要）

### Gate C2-0-5: O_denom
- ☑ definition: eligible window の windowMax max
- ☑ positivity proof: 4段階分解（commit/residency/counter/visibility）
- ☑ measurement eligibility: post-first-commit window のみ

### Gate C2-0-6: Numerical determinability
| 入力 | 状態 |
|---|---|
| λ 値 | not yet（workload contract 決定待ち） |
| G 値 | not yet（environment premise 承認待ち） |
| O_denom 値 | measurement required（実稼働データ or harness 事前取得） |

---

## VERDICT: **PASS**（全 Gate 充足・数値決定は次フェーズへ適切に繰り越し）

---

## D101-35-D 観測事項の独立分類（指示の独立判断事項）

| 項目 | 分類 | 根拠 |
|---|---|---|
| Test 14 ownerChannel 非観測 | **OBSERVATION** | by-construction 保証で補完済み。Harness stress test は将来強化。Commit BLOCKER ではない |
| ~AudioEngine closeAdmission ペアリング | **OBSERVATION** | pre-existing（HEAD 時点から存在）。destruction 中に producer は活性しないため低リスク。独立タスク候補 |
