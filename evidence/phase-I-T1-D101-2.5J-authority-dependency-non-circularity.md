# Phase I-T1-D101-2.5J — Authority Dependency / Non-Circularity Proof

> **Verdict: PLACEMENT-CANDIDATE PARTIAL / Non-Circularity CONDITIONAL**
> H-B 下で 4 契約の Authority 配置を仮置きし、dependency DAG の非循環性を監査。
> **コード変更・測定・実装なし / M/R/R_cap/T2 UNDETERMINED 維持**

## J-1 Authority → decision/observation/execution census

| Authority | decision | observation | execution | 所有する C |
|-----------|----------|-------------|-----------|------------|
| **RWA** (RuntimeWorldAuthority) | C1: reservation/publication identity (publishAndSwap 前 acquire) | current world / previous world (RuntimeStore::current) | `publishAndSwap` (sole physical publish gateway, INV-X4-3/6, X4-B) | C1 |
| **Admission** (PublicationAdmission) | C2/C3 可否: generation/sequence/admission state に基づく reversible admission gate | generation / sequence / admission state | admission result (accept/reject) | C2/C3 候補 |
| **Coordinator** (RuntimePublicationCoordinator / CoordinatorLoop) | **原則 decision を持たない** (dispatch のみ) | queue / wake / lifecycle state | dispatch (`while(pop)` → handler) | なし |
| **Builder** (ConvolverBuilder / RebuildThread) | build validity | build inputs | World construction | なし |

**Coordinator が C2/C3 の入力値を生成していないことの確認:**

```
Build facts (Builder) → Admission (PublicationAdmission) → RWA (publishAndSwap) → Retire/Life
Coordinator ─┬─→ Admission (transport ではない、scheduling のみ)
             └─→ RWA (dispatch のみ)
Builder ───→ Admission (build inputs を供給するが admission decision は Admission が行う)
Builder ───→ Coordinator (deferred resubmit → RebuildThread 起床のみ)
```

- Coordinator は Admission 結果を運ぶだけ (intentQueue_.pop → DispatchTable → handle) — `Admission → RWA` は非循環
- 逆向き `Admission → Coordinator scheduling → RWA → Admission` は **actual circular dependency の懸念** — 2.5I で指摘された通り、Coordinator scheduling が Admission 入力を生成する構造になっていないことを確認 (Coordinator は queue occupancy 等を Admission 入力にしない)

## J-2 C2/C3 分離 — Model A/B/C 比較

**この Gate で3モデルの正誤を決定しない** — 各モデルの必要入力・新規依存・責務侵食を比較:

| Model | 配置 | 必要入力 | 新規依存 | 責務侵食 |
|-------|------|----------|----------|----------|
| **Model A** | Admission: C3 α(G) / RWA: C1 + C2 | C2 に reservation→publish delay (τ) が必要 | C2 を RWA が所有するため physical+temporal 同一化リスク | RWA の temporal policy 侵食の可能性 |
| **Model B** | Admission: C2 + C3 / RWA: C1 | C2/C3 共に Admission 入力 (generation/sequence/admission state) | C2/C3 の Admission 集約で循環リスク低減 | 最も分離が明確 |
| **Model C** | RWA: C1 + direct F(G) (C2 redundant) / Admission: C3 | C2 不要化 → publication authority で直接 F(G) 契約 | C2 冗長の妥当性未証明 | RWA の temporal 侵食が最大 |

**評価保留:** いずれのモデルも現時点で「正しい」と断定せず、次 Gate での深掘り対象として保持。

## J-3 C3 入力 provenance 完全列挙

`α(G)` の各引数・状態の producer/owner/consumer:

| 引数・状態 | producer | owner | consumer | 禁止依存の有無 |
|------------|----------|-------|----------|----------------|
| G (window) | test harness / sampler | Telemetry | Admission | — |
| N_success | RWA (publishAndSwap success) | Telemetry | — | — |
| reservation count | WorldRetirementReservation (design) | Budget Authority (future) | — | `reservation count → rate` 禁止確認: **存在しない** |
| publication count | RWA | Telemetry | — | — |
| generation | Builder | RuntimeState::publication | Admission | — |
| sequence | Builder | RuntimeState::publication | Admission | — |
| queue occupancy | IntentQueue / OwnerChannel | Coordinator | — | `queue capacity → rate` 禁止: **存在しない** |
| deferred state | Orchestrator (single slot 0/1) | Orchestrator | — | `deferred → rate` 禁止: **存在しない** |
| retry state | PublicationAdmission stale 判定 | Admission | — | `retry → rate` 禁止: **存在しない** |
| queue capacity | MpscBoundedRing | — | — | `capacity → rate` 禁止: **存在しない** |
| reservation count | — | — | — | 禁止依存 **存在しない** |
| observed latency | 測定値 | — | — | 禁止依存 **存在しない** |

**確認:** 禁止されている `queue capacity → rate / queue occupancy → α(G) / reservation count → rate / observed latency → D` の依存関係はいずれも **存在しない** — 監査で確認。

## J-4 C1/C3 完全分離

| Contract | 定義 | Graph |
|----------|------|-------|
| **C1 reservation conservation** | `admitted → reserved → published → retired → released` の identity/lifetime conservation. `reservation → OwnerChannel → publish → retire` の状態遷移, `publishAndSwap` は不可逆境界 | Build → Admission → RWA → Retire/Life |
| **C3 admission envelope** | 時間区間 I → publication success 数の temporal bound `N_success(I) ≤ α(|I|)` | Admission の temporal bound (I → success 数の写像) |

```
C1 proves "誰が存在しているか" (identity/lifetime)
C3 proves "どれだけの頻度で成功できるか" (temporal frequency)
```

**分離成功:** `C3 → reservation → C3` の循環を排除 — C1 と C3 は別グラフとして扱うことで `reservation conservation` と `admission frequency` の混同を防止。

## J-5 C4 Identity Lineage 再監査

**分離:**

```
reservation identity ≠ recovery identity ≠ publication identity ≠ generation/sequence
```

| Identity | 定義 | 所有者 |
|----------|------|--------|
| reservation | `WorldRetirementReservation` token (design) | Budget Authority (future) |
| publication | `RuntimePublishWorld` (publishAndSwap success) | RWA |
| recovery | `handle + RecoveryEpisodeId + RecoveryGeneration + semantic target containment` (domain 一致だけでは不十分) | Recovery handler |
| generation/sequence | `RuntimeState::publication` (stale 判定用) | Admission |

**C4 を RWA/Admission の二択にしない** — 4者を混同せず deferred/retry/recovery 全てについて owner 一意性を確保。`generation equality = reservation identity` の同一視を禁止。

## J-6 Dependency DAG 最終成果物

```
             ┌──────────────┐
             │ Build facts  │  (Builder: inputs → World)
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │   Admission  │  (C3/C? — generation/sequence/admission state → accept/reject)
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │     RWA      │  (C1 / Publish — reservation identity, publishAndSwap, retire)
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Retire/Life  │  (deferred quarantine → terminal drain → release)
             └──────────────┘
```

**逆向き edge 監査:**

| Edge | 存在有無 | 分類 | 判定 |
|------|----------|------|------|
| Coordinator → Admission | あり (intent transport) | data dependency | **非循環** (Coordinator は transport のみ) |
| Coordinator → RWA | あり (dispatch) | control dependency | **非循環** |
| Builder → Admission | あり (build inputs) | data dependency | **非循環** |
| Builder → Coordinator | あり (deferred resubmit) | lifecycle dependency | **非循環** |
| Admission → RWA → Admission | なし (分離配置で回避) | — | **潜在循環として特定、配置分離で回避可能** |

**逆向き edge（RWA → Admission, Retire → Admission 等）は存在しないことを確認 — DAG は非循環。**

### Dependency 分類

| 分類 | 例 | 存在 |
|------|----|------|
| data | Build facts → Admission | あり |
| control | Coordinator dispatch → RWA publish | あり |
| observation | Telemetry ← RWA/Retire | あり（逆向きなし） |
| lifecycle | RWA publish → Retire | あり |
| authority | RWA = physical, Admission = temporal | 分離で非循環 |

## J-7 Gate 判定

| Gate | 条件 | 判定 |
|------|------|------|
| J-C1 | C1 reservation identity が RWA に単一所有 | **PASS** — D48 design contract で RWA 所有、INV-R1〜R4 CLOSED |
| J-C2 | C2 が RWA/Admission どちらに置かれても他方への循環を生成しない | **CONDITIONAL** — Model A/B/C のいずれでも循環は配置分離で回避可能だが、具体的配置未確定のため条件付き |
| J-C3 | C3 α(G) に禁止された queue capacity / observed latency / R / A_max 等からの未証明導出なし | **PASS** — 全禁止依存が存在しないことを確認 |
| J-C4 | C4 identity lineage が 4者を混同せず owner 一意 | **PASS** — 4分離を明示 |

## Overall Verdict

```
D101-2.5J = PLACEMENT-CANDIDATE CONDITIONAL
  J-C1 PASS / J-C2 CONDITIONAL / J-C3 PASS / J-C4 PASS
  → Non-circularity は配置分離により達成可能だが、具体的 Authority 配置の確定には次 Gate での詳細設計が必要
```

**コード変更・測定なしの設計監査として完了。**

## Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (代替 rg/sg) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 (一部 not found は環境差) |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK |

## State Update

```
D101-2.5H = H-B
D101-2.5I = PLACEMENT-CANDIDATE PARTIAL
D101-2.5J = PLACEMENT-CANDIDATE CONDITIONAL
           ↓
H-A/CLOSED へ進めるか判定 — J-C2 の具体的配置確定が次 Gate
```
