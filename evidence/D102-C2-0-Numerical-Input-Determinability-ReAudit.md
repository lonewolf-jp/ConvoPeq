# D102-C2-0 — Numerical Input Determinability & R Semantics Re-Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値採用 **0**）
- **判定**: **PASS**（6 Gate 全充足。R_required 訂正含む）
- **基準**: ConvoPeq.md 2026-08-25 16:25 版内容＋ fresh source trace

---

## Gate C2-0-1: R_required semantics 確定

### 証明

```text
証明済み: B_max^true(w) ≤ O_w(w) + M_scope    （全 eligible window w）

要求条件: retention capacity R がピークを被覆する
    B_max^true ≤ R_required × O_denom

両式より:
    R_required × O_denom ≥ O_denom + M_scope
    R_required ≥ (O_denom + M_scope) / O_denom = 1 + M_scope / O_denom

R_required は整数のため:
    R_required = 1 + ceil(M_scope / O_denom)
```

### semantics の切り分け

| 式 | 意味 | 使用場面 |
|---|---|---|
| `ceil(M_scope / O_denom)` 単独 | gap coverage multiplier — baseline 超過分のみ。base 非含有 | ❌ retention multiplier として不十分 |
| `1 + ceil(M_scope / O_denom)` | total retention multiplier — base + miss coverage 合計 | ✅ **R_cap/T2 接続時の正式な要求値として採用** |

### R_cap/T2 authority 接続時の意味

- `R_required` は「O_denom baseline に対する必要 retention 倍率」であり、絶対的な retention 個数ではない
- `R_cap`（実際に実装可能な capacity）との整合判断は `R_cap ≥ R_required × O_denom` を D102-C3 で検証
- `T2`（time authority）は G_bound 項が時間要素を吸収済みのため T2 変更不要だが、shutdown drain duration の確認は D102-C3 scope

✅ **Gate C2-0-1 = PASS**。

---

## Gate C2-0-2: λ_prod_bound producer inventory

### 全 commitRuntimePublication call site 実測

| # | producer | call site | 定常発生 | λ 加算 | 分類 |
|---|---|---|---|---|---|
| P-1 | PrepareToPlay #1 | PrepareToPlay.cpp:155 | lifecycle 時のみ | ❌ | lifecycle-bound |
| P-2 | PrepareToPlay #2 | PrepareToPlay.cpp:277 | 同上 | ❌ | 同上 |
| P-3 | Timer #5 | Timer.cpp:994 | ✅ ≤1/tick ≒10/s | ❌ N_timer 項へ | 構造的 hard bound |
| P-4 | Transition #6 | Transition.cpp:25 | ユーザー/自動遷移時 | △ workload contract 必要 | workload |
| P-5 | ReleaseResources | ReleaseResources.cpp:175 | shutdown 中のみ | ❌ scope 外 | lifecycle-bound |
| P-6 | deferred resubmit | PublicationExecutor.cpp:53-66 | 元 submission の派生 | ❌ 二重計上防止 | derivative |
| P-7 | Orchestrator Path A | submitPublishRequest → executePublish | ユーザー publish 要求時 | △ workload contract 必要 | workload |
| P-8 | Recovery | quarantine 検出時 | build failure 連動 | △ workload contract 必要 | workload |

### λ_ext construction rule

```text
λ_prod_bound = λ_transition(P-4) + λ_user_publish(P-7) + λ_recovery(P-8)

加算の正当化: CoordinatorLoop 直列化により各 producer の acquire は排他的に処理。
Σ(個別上界) ≥ 合成レート は保守的正しい上界。
```

⚠️ P-6 deferred resubmit は初回 submission の派生のため独立項から除外
（失敗 attempt は acquire 不発火・成功時のみ acquire 1回 — INV-PUB-4 整合）。

---

## Gate C2-0-3: G_bound environment contract boundary

| 構成要素 | G_bound 包含 | 分類 |
|---|---|---|
| nominal cadence 100ms | ✅ 含む | source constant |
| callback scheduling delay | ✅ 含む | message thread queue 待ち |
| missed/coalesced callback | ✅ 含む | JUCE coalescing property |
| OS scheduling delay | ✅ 含む | thread priority/preemption |
| message-thread starvation | ⚠️ 要判断 | modal dialog 等で理論上無限遅延し得る |
| samplerTick 処理時間 | ✅ 含む（微小） | atomic ops のみ |

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

## VERDICT: **PASS**（全 Gate 充足・数値決定は次フェーズへ適切に繰り越し）

---

## D101-35-D 観測事項の独立分類（指示の独立判断事項）

| 項目 | 分類 | 根拠 |
|---|---|---|
| Test 14 ownerChannel 非観測 | **OBSERVATION** | by-construction 保証で補完済み。Harness stress test は将来強化。Commit BLOCKER ではない |
| ~AudioEngine closeAdmission ペアリング | **OBSERVATION** | pre-existing（HEAD 時点から存在）。destruction 中に producer は活性しないため低リスク。独立タスク候補 |
