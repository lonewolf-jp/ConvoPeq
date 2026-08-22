# Phase I-T1-D101-2.5G — Temporal Conservation / Reserve→Publish Displacement Audit

> **Verdict: D101-2.5G = OPEN / Case B — Finite D contract missing / M = UNDETERMINED**
> Design L0: reservation acquire = publish() 内 swap 前（I4 D48/D49）。Production L2: 未実装（src/ に WorldRetirementReservation 0件）。
> `τ_res→publish ≤ D` の有限 structural upper bound は Design/Production いずれにも存在しない。
> **N_publish(I) ≤ N_reservation(I⊕D) の temporal dilation は現行 invariant から導出不可。**
> コード変更・測定・実装なし / A_max=1 / T_w=2 / max(E_w)=1 固定維持。

## G-1. Reservation temporal endpoints 固定

```
t_reserve = reservation acquire 成功時刻
t_publish = A_true increment LP = successful PublishedDomain admission = publishAndSwap success
τ_res→publish = t_publish - t_reserve
```

| Layer | t_reserve | 根拠 |
|-------|-----------|------|
| Design L0 | t_reserve^design = WorldRetirementReservation::acquire(prevWorld)（D48: publish() 内 swap 前・I4 D49/D51） | I4 D48/D49 design contract CLOSED |
| Production L2 | t_reserve^prod = N/A（未定義） | src/ grep WorldRetirementReservation 0件（rg/ag/fdfind/semble/AiDex 全一致・C6-Finalまで） |

**曖昧にしたまま temporal bound 議論禁止 — 本監査で二層分離を厳守。** I4 design は acquire→不可逆 swap の ordering を定義するが、Production PublishExecutor（authority.publish → didPublish/willRetire/retire tail）は reservation 層を含まない。

## G-2. Temporal chain 全経路 census

| Path | reserve | build | queue | deferred | retry | publish |
|------|---------|-------|-------|----------|-------|---------|
| normal rebuild | ? (design) | ✓ | ✓ (OwnerChannel 256 + IntentQueue) | — | — | ✓ |
| bootstrap | ? | ✓ | ✓/direct | — | — | ✓ |
| direct publish | ? | ✓ | ✓ | — | — | ✓ |
| deferred | ? | 完了済 | re-enqueue | ✓ (RebuildThread起床) | ✓ | ✓ |
| recovery | ? | ✓ | Builder queue | possible | ✓ | ✓ |
| shutdown | ? | — | — | — | — | special (退避 drain) |

**順序 invariant 検証:** `t_reserve ≤ t_build_complete ≤ t_enqueue ≤ t_execute ≤ t_publish` は **actual invariant ではない**。理由:

- deferred/retry/recovery は `same reservation` 維持か `new reservation` かの契約が未定義（D101-2.5E-5 OPEN, D101-2.5F でも OPEN 維持）。
- PendingPublishRegistry (kCapacity=64) と OwnerChannel の reservation 分離は D101-2.5R で ≠ burst bound と証明済み。

## G-3. Delay component 別 upper bound 分類

```
τ_res→publish = τ_res→build + τ_build→enqueue + τ_enqueue→execute + τ_execute→publish + τ_deferred/retry
```

| 項 | 現行有限 bound | 判定 |
|----|----------------|------|
| τ_res→build | 有界性 invariant なし | **OPEN** |
| τ_build→enqueue | 有界性 invariant なし | **OPEN** |
| τ_enqueue→execute | queue bounded ≠ latency bound（Owner→IntentQueue は単一 executor FIFO だが service-curve なし） | **OPEN** |
| τ_execute→publish | publishAndSwap 自体の temporal serialization なし | **OPEN** |
| τ_deferred/retry | deferred Ready→re-enqueue の delay 契約なし | **OPEN** |

**queue capacity < ∞ から τ_enqueue→execute < ∞ を推論禁止** — 最新 source の構造（Owner 占有移譲 → IntentQueue投入 → Coordinator 後続 execute、queue full時は取り戻して失敗）からも、finite queue は latency bound ではない。

## G-4. Coordinator wake/drain ≠ delay bound

`processIntent()` drain 構造（ISRRuntimePublicationCoordinator_ProcessIntent.cpp）:

```cpp
while (quarantineFallbackQueue_.pop(...)) ...
while (intentQueue_.pop(...)) ...  // Publish|Quarantine|Observe|Recovery
```

| 命題 | 真偽 |
|------|------|
| 1 ms timeout ≠ maximum enqueue→execute latency | **真**（timeoutは fallback、wake signalで即時起床） |
| single consumer ≠ bounded execution latency | **真**（single serialized executor だが temporal bound ではない、D101-2.5C 確定を再利用） |
| FIFO ≠ bounded temporal displacement | **真** |
| bounded queue ≠ bounded waiting time | **真**（drain 全件処理で待機時間が queue 占有に依存しない） |

**結論: wake/drain 機構は τ_res→publish のどの成分にも有限 D を与えない。**

## G-5. Receipt wait 除外証明

`commitRuntimePublication()` = `enqueueRuntimePublicationFireAndForget() → waitForPublishReceipt(seqId, 250ms)` 同期 wrapper。source自身が timeout を publish failure と解釈せず ownershipは enqueue時 transferred と明記。

```
producer wait ≠ publication temporal displacement
→ receipt wait は τ_res→publish の upper bound ではない（固定）
```

Producer待機と Coordinator側 publish遅延は別軸 — 本監査で形式的に固定。

## G-6. Bounded delay なくても publication window bound 出るか

要求: `∃ finite D : ∀r, t_publish(r) - t_reserve(r) ≤ D` が source/design から証明できるか

```
∀r: success(r) ≤ 1 が design CLOSED でも τ_res→publish ≤ D がなければ
N_publish([t,t+G]) ≤ N_reservation([t-D,t+G]) の dilation は作れない
```

**有限 D の source/design contract は不存在（実測値禁止）。→ OPEN**

## G-7. Deferred/retry liveness と temporal bound 分離

| 性質 | 定義 | 現状 |
|------|------|------|
| Safety | reservation が勝手に release されない | design PARTIAL（quarantine full時の保持は I4 上 catastrophic path として明示） |
| Conservation | 1 World ≤ 1 reservation | design CLOSED / production OPEN |
| Liveness | reservation後 publish が eventually occur | **未証明**（deferred liveness は I4 上別 track） |
| Temporal liveness | publish が finite D 内に occur | **OPEN**（G-6 と同一） |

**reservation remains held ≠ publication eventually within D** — 4分離を本監査で固定。

## G-8. 最終3択

| Case | 条件 | 判定 |
|------|------|------|
| A. finite D が structural contract として存在 + N_reservation(I') ≤ λ|I'|+B | 不成立（τ成分全 OPEN） |
| B. ordering は存在するが finite D なし → D101-2.5G OPEN | **★ 該当** — F成果を踏まえた最有力予測どおり確定 |
| C. reservation identity/lifecycle 自体が破綻 | 不成立（design CLOSED / production は未実装だが破綻ではない） |

**→ Case B: D101-2.5G OPEN / temporal displacement contract missing**

## G-9. 成果物 — 14項目 proof obligation

| Proof obligation | Design | Production | D101 |
|------------------|--------|------------|------|
| t_reserve 定義 | CLOSED (D48 acquire) | N/A (未実装) | OPEN |
| t_publish 定義 | CLOSED (publishAndSwap success) | CLOSED (同) | CLOSED |
| reserve→build bound | OPEN | OPEN | OPEN |
| build→enqueue bound | OPEN | OPEN | OPEN |
| enqueue→execute bound | OPEN | OPEN | OPEN |
| execute→publish bound | OPEN | OPEN | OPEN |
| deferred displacement | OPEN | OPEN | OPEN |
| retry displacement | OPEN | OPEN | OPEN |
| recovery displacement | OPEN | OPEN | OPEN |
| shutdown exception | CLOSED (特殊退避) | CLOSED | CLOSED |
| finite D | OPEN | OPEN | OPEN |
| temporal dilation | OPEN | OPEN | OPEN |
| N_publish(I) ≤ N_reservation(I') | OPEN | OPEN | OPEN |
| service curve connection | OPEN | OPEN | OPEN |

**最終:** `D101-2.5G = OPEN`

## D101-2.5G 完了後分岐

```
D101-2.5G OPEN (finite Dなし)
  → D101-2.5H — Temporal Envelope Necessity / Minimal Contract Audit
     D101-2.5D admission envelope + D101-2.5E/F conservation + D101-2.5G temporal displacement
       → 「A_true を有限 service curve にするために最低限何を契約すべきか」を最小公理集合として整理
```

## 禁止事項遵守

`× コード変更 × reservation implementation × admission limiter × token bucket × 測定 × observed latency→contract × queue capacity→latency bound × 1ms wake→1ms service bound × receipt 250ms→publication bound × R→λ × R→B × A_max→temporal rate × T_w→M × M数値化 × T2 GO` — **全て未実施**

## Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (前 Gate 多数確定済みで代替) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK (Vyukov失効→rigtorp代替) |

*Evidence generated: Phase I-T1-D101-2.5G — finite D contract missing を確定。次は D101-2.5H で最小必要契約集合を整理。*
