# D101-35-B′ — M-Bound Contract Input Design / Finite-Bound Closure Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only 設計・可能性監査（ソースコード変更 **0** / 契約変更 **0** / 具体値の決定なし）
- **判定**: **B′-CONDITIONAL**
  （有限 M は理論上導出可能。ただし契約定数の**適用範囲**に未確定の決定点 1 件あり — §6）
- **基準**: ConvoPeq.md 2026-08-25 16:25 版（ソース未变更のため最新のまま有効）＋ fresh source trace

---

## 1. B′-1 G_bound の定義確定

### 必須判定（YES/NO）

| 判定項目 | 答え | 根拠 |
|---|---|---|
| Timer cadence（100ms）だけから finite G_bound を導けるか | **NO** | JUCE Timer は message thread 動作。公称周期は目安であり、OS スケジューリング/メッセージポンプ停滞に対する上限保証がコード上存在しない |
| `missedTickCount` から finite G_bound を導けるか | **NO** | 実測統計カウンタであり bound ではない（measurement ≠ safe bound） |
| `maxSamplingGapUs` を safe bound として扱えるか | **NO** | 同上。D100.5 の E_w=1 は実測 gap でも取り逃しが起きたことの証明であり、実測値の昇格は禁止（指示どおり） |
| watchdog / scheduling contract がコードに既に存在するか | **NO** | timer 遅延検知の watchdog・tick 間隔強制機構は未実装（missedTickCount の記録のみ） |
| 契約追加だけで G_bound を導入可能か | **YES** | caller/environment contract として「message thread のスケジューリング前提（例: tick 遅延は W_max 以下）」を文書化すれば、M 導出の入力としては成立する |
| 実装変更が必要か、純粋な caller/environment contract で成立するか | **純粋な契約で成立**（M 導出のためには） | enforcement（watchdog 実装）は将来の強化であり、M の数学的導出には不要 |

### 確定定義（案）

```text
T_sampler = 100 ms        （公称 cadence — 実装事実）
G         = sup(event time → 次回 samplerTick での反映完了時刻までの遅延)   ← 監査対象量
G_bound   = 契約前提: message thread スケジューリングにより G ≤ G_bound が保証される環境前提
            （jitter / missed tick を含む。値は D101-35-C 以降で確定）
```

✅ **G_bound は契約追加のみで導入可能**（M_gap の有限化に必要十分）。

---

## 2. B′-2 μ_burst / τ_b の意味分離

```text
μ_burst ≝ burst 区間中の最大 acquire 発生率（onAcquireObserved の増加率）
τ_b     ≝ 当該 burst の継続時間
M_burst ≤ μ_burst × τ_b … ただし μ_burst は retire rate の実測から作らない
```

重要な区別（A 結果より厳密化）:

- **acquire（onAcquire）は execute/commit 時点で発生し、enqueue 時点では発生しない**
  （Commit.cpp:409 = `onRuntimePublishedNonRt` 内 = executePublish の commit tail）。
  よって μ_burst の正しい対象は **publication execute rate** であり、producer の
  submit rate ではない（submit rate は queue backpressure を介して間接的に効く）。
- retire rate は release 側（epoch gate + drain cadence 依存）であり、acquire burst と
  独立。`μ_burst` を retire 実測から作るのは領域混同。

---

## 3. B′-3 Publication serialization の実効上界（source trace 実測）

| # | 必須項目 | 実測結果 |
|---|---|---|
| 1 | publication の同時実行数 | **1**（CoordinatorLoop 単一 consumer が intentQueue_ を直列処理。ProcessIntent が pop→handle を逐次実行） |
| 2 | request の queue capacity | **kIntentQueueCapacity = 4096**（Coordinator.h:693, MpscBoundedRing）。full 時 `enqueuePublicationIntent` は false → facade が CallerDestroy で reject（backpressure 成立） |
| 3 | queue 滞留だけでは onAcquire 発生？ | **NO** — `onAcquireObserved` は `onRuntimePublishedNonRt`（executePublish の commit tail）内。enqueue 時点では発生しない（ProcessIntent.cpp の type 分岐参照） |
| 4 | onAcquire の時点 | executePublish → didPublish callback（commit tail）。**queue 滞留分は未 acquire** |
| 5 | T_build の正の下限 | **存在しない**（build コストは world 内容依存。理論上の下限 0 を排除する定数・契約は現行に不存在） |
| 6 | T_build=0 排除 contract | 不存在（追加可能だが implementation-performance 主張となり enforceability が弱い — §4） |
| 7 | rebuild rate → publication rate の deterministic implication | **部分のみ** — Timer idle publish (#5) は timerCallback 内のため構造的に ≤1/tick（≈10/s）。rebuild 完了駆動 publish は build 完了率に依存し deterministic ではない |

### 重要な構造的事実（指示の論点への回答）

- **queue capacity（4096）は publication 数上界ではない** — 同時に acquire されないため。
- しかし **backpressure により飽和時の acquire throughput は消費率に制限される**:
  full → 新規 publish は拒否（acquire 不発生）。
- それでも **単位時間あたりの実行回数の有限上界は導出できない**（T_exec の正の下限が
  ないため）。→ Candidate B の Case B が適用される。

---

## 4. B′-4 T_build_min 成立性判定

```text
ケース A（T_build ≥ T_build,min > 0 を証明できる）→ 採用不可
```

理由: build/validate/commit は world payload 依存の O(n) 処理であり、正の下限を与える
コード定数・契約は現行に不存在。「ほぼゼロの world」を繰り返すことで実行率は
理論上非有限に近づき得る。

**代替経路（採用推奨）**: producer submission-rate 契約

```text
λ_prod_bound ≝ 全 publication producer の intent submit レートの契約上界
```

これを採用した場合の M_burst 構造:

```text
M_burst ≤ (gap 開始時の queue backlog) + (G_bound 中の新規 execute 数)
        ≤ kIntentQueueCapacity + λ_prod_bound × G_bound      ← すべて有限定数
```

- backlog 項 4096 は**構造的定数**（コードから取得済み）
- λ_prod_bound × G_bound は契約積
- T_build_min は**不要になる**（execute rate は producer rate と backlog で頭打ち）

---

## 5. B′-5 Burst duration の独立性

operational definition（採用案）:

```text
burst ≝ 「acquire が sampler gap より高いレートで発生している区間」
開始   : acquire レートが baseline（timer 駆動 idle publish ≈ 10/s 等）を超えた点
終了   : (a) queue saturation による reject 発生、(b) producer 停止、(c) shutdown 強制終了、
         (d) G_bound を超えて sampler が追いついた点 — のいずれか最初
```

| 検証項目 | 結果 |
|---|---|
| 最大継続時間の構造的上界 | ❌ 単独では有限化できない（producer が供給し続ければ継続）— **queue 有限だから τ_b 有限という推論は不成立**（指示どおり）。ただし backpressure により飽和後は reject へ切り替わるため、**unobserved acquires の蓄積は 4096 + 到着分で頭打ち** |
| shutdown 強制終了 | ✅ あり（closeAdmission 後 tryAdmit 失敗 → 新規 obligation 不发生 — D101-33-C/D） |
| queue saturation backpressure | ✅ あり（reject + CallerDestroy） |
| rebuild cancellation/coalescing | ✅ あり（Recovery coalesce: INV-X1-5/6、retryScheduler shutdown 等） |

τ_b 単独立項ではなく **λ_prod_bound × G_bound 形式へ統合**するのが正しい（B′-6 推奨どおり）。

---

## 6. B′-6 M の二重計上防止

```text
M = M_gap(G_bound, λ_prod_bound) + M_burst(kIntentQueueCapacity, λ_prod_bound × G_bound)
M_boundary = 0（M_gap に吸収 — D95 固定点1 baseline 連続性）
M_jitter   = M_gap に吸収（G_bound 定義に jitter/missed tick を含めて定義）
```

独立項は M_gap と M_burst の 2 つに整理。jitter/boundary は独立項として残さない。

---

## 7. B′-7 最終的な有限性判定表

| Input | 必要条件 | 現行コードから証明 | 契約追加で補完可能 | finite M に必要 |
|---|---|---:|---:|---:|
| `G_bound` | event→next tick の有限上界 | **NO**（watchdog/契約なし） | **YES**（environment contract） | YES |
| `λ_bound`(=λ_prod_bound) | 全 producer の submit rate 上界 | **NO**（rate limiter 未実装） | **YES**（caller/environment contract） | YES |
| `μ_burst` | gap 中の最大 execute rate | **NO**（T_exec 下限なし・Case B） | **YES**（λ_prod_bound + backlog 4096 から導出可能に） | YES |
| `τ_b` | burst duration 上界 | **NO**（単独では無限） | **YES**（G_bound 区間に統合 → 独立項廃止） | YES（統合形） |
| `M_boundary` | boundary loss finite | **YES**（D95 固定点1） | — | 吸収 |
| backlog cap | queue 容量 | **YES**（4096・コード定数） | — | 使用 |

---

## 8. 最終判定

# **B′-CONDITIONAL**

### 根拠

```text
有限 M は理論上導出可能:
    M ≤ [4096 + λ_prod_bound × G_bound] + M_gap(λ_prod_bound, G_bound)
    （すべて有限契約定数の関数）

ただし D101-35-C 以前に以下の「適用範囲決定」が未確定:
    ★ λ_prod_bound が拘束する producer site の範囲
      - 内部自動系: Timer idle publish #5（構造的に ≤10/tick・既知）
                  / rebuild 完了駆動 publish（build 完了率依存）
                  / Recovery publish（gate 済み）
      - 外部/フロー系: PrepareToPlay ×2 / Transition / ReleaseResources:175
      → 「内部自動系は別枠（構造 bound 済み）とし、λ_prod_bound は
         外部/フロー系 submission に適用する」等の範囲宣言が必要
    ★ G_bound の前提としての message-thread スケジューリング環境の明文化範囲
```

### 各候補との対応

| 候補 | 該当性 |
|---|---|
| B′-PASS（契約追加のみで M<∞） | 機構としては成立するが、λ_prod_bound の適用範囲決定が先決のため**即 PASS は付けない** |
| **B′-CONDITIONAL** | ✅ **採用** — 有限 M は導出可能。未確定は「契約定数の適用範囲」のみ |
| B′-BLOCKED | 該当しない（rate/duration に有限上界を与える契約自体は成立する — 特に producer submission rate と environment 前提は文書化可能） |

---

## 9. 禁止事項の遵守確認

* ソースコード変更 0 ✅ / I4_DESIGN_CONTRACT.md 編集 0 ✅
* G_bound / M の具体値決定 なし ✅ / R_required 導出 なし ✅ / D102 GO 判定 なし ✅
* 実測値（maxSamplingGapUs 等）の安全上界昇格 なし ✅
* queue capacity からの直接 rate bound 導出 なし ✅（backlog 項としてのみ使用）
* 「直列だから有限」という結論不使用 ✅（Case B 明示・§4）

---

## 10. D101-35-C への引き継ぎ

D101-35-C（M-bound mathematical derivation）開始前に確定すべき決定事項:

1. λ_prod_bound の適用範囲（内部自動系を構造 bound として別枠にするか、一律に含めるか）
2. G_bound の前提として明文化する環境スケジューリング仮定の文言
3. （参考）enforcement 用 watchdog の将来実装有無

上記確定後:

```text
D101-35-C: M = f(G_bound, λ_prod_bound, kIntentQueueCapacity=4096) の導出
      ↓
B_max^true ≤ O_w + M 形式化
      ↓
D101-35-D: R_required / D102 GO-NO-GO 判定
```

M-bound / Phase I / D102 の NO-GO は本タスクでも一切変更していない。
