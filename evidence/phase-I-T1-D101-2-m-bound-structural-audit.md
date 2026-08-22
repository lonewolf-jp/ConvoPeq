# Phase I-T1-D101-2 — M-bound structural audit

> **Verdict: D101 #2 = OPEN（M-bound investigation = NO-GO 継続 / M / R / R_cap / T2 = UNDETERMINED）**
> **A_max_candidate=1 / T_w=2 / max(E_w)=1 の observed characterization を維持 — M への昇格なし**
> コード変更・harness 変更・測定追加 なし。

## 1. D101-2A — Acquire completeness

| 項目 | 根拠 | 判定 |
|---|---|---|
| PublishedDomain への全 publish 成功 | 2 経路: RebuildThread 等の async producer は `commitRuntimePublication`（AudioEngine.h:4595-4708）、Bootstrap/Rebuild (#2/#4/#6) は `enqueueRuntimePublicationFireAndForget`（:4517-4590）。両者とも owner 晒出→OwnerChannel/registry/intent 投入 | ✅ |
| acquire observer LP | RebuildThread path: `onRuntimePublishedNonRt`（未共有）。Coordinator path: `RuntimePublicationCoordinator::publishWorld` は現行では observer を発火しないが、世帯生成直後に deterministic に commit 前に後退する Bridge-前 `RuntimeWorldAuthority::publish` が旧世界的「唯一 publish-success LP（X4-B §6.4 / INV-X4-2）」に昇格（work88 X4-B-4）。両 path とも `newWorld publication.sequenceId !=0` の all-or-nothing（失敗時は caller-owned） | ✅ |
| `∀ W ∈ PublishedDomain: exactly-one reference acquire` | 両 publish LP が 1 world につき 1 回の atomic increment のみ（publishAndSwap 支配） | ✅ |
| `W ∉ PublishedDomain ⇒ referenceAcquireCount 不寄与` | rejected publish（Coordinator.h:122-124 の ValidationFailure → `retireRejectedRuntimeWorldNonRt`）は Generic 側（D101-1 完了）。OwnerChannel 残留は Generic 化（D101-1F）。未成り立つ PublishedDomain 未満の world については acquire 未発火 | ✅ |

## 2. D101-2B — Release completeness

Terminalization boundary は single-transfer DeletionEntry 削除点（任意の 9 LP の削除成功時）であり、prototype は `DeferredDeletionQueue.h:144-156 / RetireQuarantineStore.h:140-194 / ISRRetireRouter.cpp:62-96,398-511` に跨る `entryType == DeletionEntryType::World → type==World branch → worldReclaimCount++`, `referenceObserver_->onRelease(); ++reclaimed` / pendingTypes 保存（shutdown）→ `onRelease()` / `++reclaimed` / 同地内 1:1 観測。D101 #1 `World producer = retirePublishedRuntimeWorldNonRt only` の閉鎖により、**producer provenance 再監査は重ねず observer coverage に集中**（指示どおり）。

| LP | File | 対応 |
|---|---|---|
| provider (D) | DeferredDeletionQueue.h:144-156 | 1:1 exactly-once（harness スタビライザが rack で動的検証済み） |
| quarantine (Q) | RetireQuarantineStore.h:140-194（通常 drain 前半 / shutdown 完全解放） | 1:1 exactly-once（fall-back 調査済み） |
| emergency (E) | 同（別 instance） | 同上 |
| terminal | ISRRetireRouter.cpp:62-96（stuck / shutdown 完全解放）+ 45b-delta 転送 | 1:1（stuck は delayed、shutdown は quiescence 後に全解放） |
| shutdown 全強制 | DrainDomain drains 全体 + CtorDtor drains | 上記 aggregate + drain 审计（D101-1 五次監査）で漏洩なき収束 |

```
terminal success ⇒ exactly one reference release
terminal failure / non-World ⇒ zero reference release
```

**release completeness = CLOSED（D101-1 による single-transfer 終端＋既存 StuckReader fallback 監査により達成）**

## 3. D101-2C — Window boundary consistency

`AudioEngine.h:4948-4960` 単一 `runWorldRetirementMeasurementStep()` が `transferWorldReclaimDeltaForTelemetry → estimate → max → tag → samplerTick → reference sync` を一括実行。samplerTick 後に measurementState を見て `Running → onMeasurementStart() else onMeasurementEnd()` は dirt に投棄されない rc を払う（D91.1 上書き契約の rc 上書き）:

| 要件 | 根拠 | 判定 |
|---|---|---|
| windowStart 以前の event が T_w 混入なし | RC 不在時の window switch point は samplerTick 境界 | ✅ |
| windowEnd 以後の event 混入なし | 同上（止まる tick は endTick 後の samplerTick） | ✅ |
| Start/End 冪等 | `onMeasurementStart` は CAS `running_=1` + Running 遷移同期 | ✅ |
| 同一 windowId で O_w と T_w 比較 | `world_retirement_telemetry.json` は sampld one + reference per Window 実証。kMaxRetireAgeMs の試行 Plan #7 未満 | ✅ |
| observer window state が sampler から乖離しない | shared step 内同期化（D96 実食 gate ④） | ✅ |

**重要:** `T_w == B_max^true` ではない。I4 D101.1 の三層 `B_true → B_ref → T_w` を維持。T_w は window への reference 積善 peak（log max を持った reference 計測）であり、最新 mixed-phase signal の「回復／重再生」ではなく有用性としての定着値としての限界 `M` には直進しない。

## 4. D101-2D — λ の意味一本化

`B_true(t) = A_true(t) - R_true(t)`（I4 定義）。区間 `[t_k, t_{k+1}]` で `ΔB_true = ΔA_true - ΔR_true`。

| λ 候補 | B_true 上昇量との関係 | 評価 |
|---|---|---|
| λ_payload / λ_retire / λ_reclaim | release 処理レート（drain/collect）。B_true を**低下**させる側（growth を制約しない） | ✗ 上界化に逆方向 |
| λ_acquire | **publish 受理レート**（RuntimeIntent publish の FIFO 承諾）。ΔA_true を直接制約する真の成長レート。ΔB_true ≤ ΔA_true より上界化は `Δgrowth ≤ ∫ λ_acquire` 方向で持っていける | ✅ **M-bound で唯一制約力を持つ λ は暗黙に λ_acquire** |
| λ_publish | λ_acquire と同義（publish queue 承諾側）。文脈に応じて「受理」を強調する場合の別名 | ✅ 同一視（機会的） |

**結論: λ を単一値とするなら λ = λ_acquire（validated world の承諾・publication rate）である。** 平均承諾率だけでは burst の方向を束ねられないため、単純 mean λ では不十分であることを D/E に引継ぐ。

## 5. D101-2E — τ_b / burst model（固定）

| 量 | 意味 | D101 要件 | I4 引継ぎ |
|---|---|---|---|
| τ_b | burst duration（分散の如何にかかわらず publish が恒定する継続時間） | I4 D100.7 → D101 | — |
| μ_burst | burst rate（burst 中の承諾率・λ_acquire の瞬時 peak） | — | — |
| J | jitter bound（timing ばらつきの有界性） | — | — |
| G | max sampling gap（unit gap の実効上界） | — | — |

```
ケース 1: τ_b > G      — burst が複数 sampler 間隔を跨ぐ（再 trichotomy）
ケース 2: τ_b ≤ G      — burst が単一 gap 内に収まる
        2a: gap 窓中に burst 完結（観測欠落 → Eo^obs の感受部）
        2b: burst が sampler tick 境界を跨ぐ（gap 収増分に split され bound 維持）
ケース 3: μ_burst = λ_acquire burst peer（平均 λ では不足・瞬時 peak による convex 下がり）
```

## 6. D101-2F — Reference completeness 限界（E^obs）

理想は `B_ref` に関する証明を `B_true` へ持上げ:

```
E^obs = sup_k [B_true(t_k) - B_obs(t_k)]
B_max^true ≤ O_w + (Δgrowth + E^obs)
```

I4 は growth/burst bound と `E^obs`（観測誤差）を **D101.2 で分離**する設計（D101.1 の三層）。現行 observer は atomic counter + event-driven update であるが、**sup_k 区間の有限 E^obs が現行 source 上で構造的に有限と証明されたコード証拠なし**。B_true は kernel `A_true - R_true`（厳密）の同期構造、observation tick は別スレッドの sampler tick であり、sup の worst-case は atomic の linearization とは独立に**時間窓の競合**として残る。

→ **E^obs の構造的有限 bound は未証明（有番観測 `watchBatch` 3 run だけでは bound intrinsic の起結には達しない）。**

## 7. D101-2G — M 候補式の方向のみ提示（数値を入れない）

D–F より:

```
M ≥ Δgrowth(τ_b, μ_burst, G, …) + E^obs(G, jitter, observer completeness, …)
Δgrowth ≤ acquisition/burst bound over G   （λ_acquire / τ_b / μ_burst で有界化を試みる対象）
E^obs    ≤ reference/sampler 観測差異 bound （D101-2F の未完対象）
```

```
M(G, λ_acquire, τ_b, μ_burst, J, ...)
```

として**形のみ**を意図に向ける（数値入れず）。現時点で Δgrowth の単純 `λ_acquire × G` や `μ_burst × τ_b` が無制限 publish 任意実行の下で無界である可能性を否定する残骸証明が十分ではないため、M としての定数置きは行わない。

## 8. D101-2H — Counterexample census

| Counterexample | 必須判定 | 再判定 |
|---|---|---|
| gap 中に acquire 無制限増加 | finite bound ありか | **不明（構造的 finite proof 無） — Burst は channel/queue full で屈伏するが、batch 機製としての確実上界が原子的に閉鎖されておらず、λ_acquire の瞬時 peak を平均率で近似できない。MA** |
| burst が tick 境界跨ぐ | bound 維持か | **維持可能（split 平均で gap 単位に帰着）** |
| burst が複数 tick 跨ぐ | τ_b で制約可か | **τ_b 導入により再分配は可能だが、τ_b 自体の構造的上界が非固定** |
| release 遅延で B_true 増加 | growth bound 入りか | **release 遅延は B_true を低下（ΔR 遅延は ΔB 増加方向に作用するが、遅延自体を Δgrowth bound の漏れとして burn する必要）** |
| reference が event 取り逃す | 有無 | **観測漏れ概念なし（atomic counter）。ただし atomic と同時性の sup を B_true へ持ち上げる E^obs が未証明** |
| window boundary race | T_w が窓外含むか | **supply chain 境界では搬枚起きない（共有 step 同期）** |
| observer/sampler 順序 | race/ambiguity | **source-order audit で race 無（runWorldRetirementMeasurementStep 内単一順序）** |
| B_ref < B_true | 有限 E^obs 証明可か | **未証明（D101-2F）** |

## 9. D101 #2 Gate（6 項目）

| Gate | PASS 条件 | 判定 |
|---|---|---|
| D101-2-1 | acquire completeness | ✅ CLOSED（両 publish LP が exactly-one acquire） |
| D101-2-2 | release completeness | ✅ CLOSED（9 LP 1:1 exactly-once + G2-1/2 で liveness 確認） |
| D101-2-3 | window boundary consistency | ✅ CLOSED（shared step 同期・冪等・同一 windowId 比較） |
| D101-2-4 | G/λ/τ_b/μ_burst/jitter の意味が一意 | **OPEN（λ は λ_acquire に絞るも τ_b/G の構造値が未固定）** |
| D101-2-5 | Δgrowth の構造上有限 bound | **OPEN（無制限 publish の取引束を排除する確実上界がまだ閉鎖不能）** |
| D101-2-6 | E^obs の有限 bound または未証明理由の明確化 | **OPEN（理由明確: 有番観測 3 run の有限 notion で sup を構造化不能）** |

## 10. 結論（構造監査として）

```
D101 #2
├─ reference completeness = CLOSED（A/B/C）
├─ growth bound         = OPEN（burst 無制限可能性を構造的に排除できず・D/E）
├─ observation error bound = OPEN（E^obs 未証明・F）
└─ M = f(G, λ_acquire, τ_b, μ_burst, J, ...) として形のみ提示（数値提示せず・G）
```

→

```
D101 #2 = OPEN
M = UNDETERMINED
R / R_cap / T2 = UNDETERMINED 継続
M = max(E_w)=1 への short は NO-GO を維持
```

*D101 #2 が途中で有限 bound を証明できないという結論も「失敗」ではなく正しい停止である — ここで M/R/R_cap/T2 を決定しない（指示どおり）。*

## 11. Tool Coverage

| 系統 | ツール | 結果 |
|---|---|---|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | World producer / retire API / enqueue/drain / observer 参照の複数手段一致 |
| Sandbox | node fs walk（acquire/release/observer/window census） / source 直読（Coordinator.h:130-150・AudioEngine.h:3505-4720・OwnerChannel.h:7030 全文・AudioEngine.Commit.cpp:446-600・RuntimeWorldAuthority.h:10644-10670） | provenance / window 境界 / λ 定義区別を確定 |
| MCP/CLI | serena: 一時無効（代替として rg/sg/node で補完。前 Gate までに多数シンボル確定済み） | — |
| MCP | AiDex: 一時無効（代替として node walk + rg census。具体観測 0 hits は前 Gate までに索引確認） | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実行（World producer / acquire observer 等） |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 前 Gate までに 9/10 200 OK 済（Vyukov SSL 失効→rigtorp 代替、crossbeam equiv URL 対応済み） |

## 12. 次進行

```
D101 #2 OPEN のまま次は D101 #2 の growth/observation 各 OPEN item を個別 proof obligation として恒常的課題登録
        ↓
D101-2-5 (Δgrowth): publish burst の構造的有限 bound を別 proof track として記録
D101-2-6 (E^obs): reference→true 観測誤差の有限 bound を別 track として記録
        ↓
両 track が閉鎖され次第 M = Δgrowth + E^obs として数値化可能 → M-bound成立 ? → D102 / R determination review
```

*生成: Phase I-T1-D101-2 — 追加測定・コード変更なしの構造監査として OPEN を確定。*

