# Practical Stable ISR Bridge Runtime — 最終評価（2026-06-12）

**作成日**: 2026-06-12
**元文書**: `doc/work32/notfinished8.md`
**検証レポート**: `doc/work32/notfinished8_validation_report.md`
**評価者**: ユーザー（コード検証結果に基づく再評価・優先順位修正版）

---

## 評価の前提

`notfinished8_validation_report.md` のソースコード実証結果を反映し、
Practical Stable ISR Bridge Runtime の残課題を優先度順に再整理する。

---

# S — システム破綻につながる構造的問題

## S-1. Epoch 汚染 — DSPLifetimeManager::retire() が publishEpoch() を呼ぶ

**検証結果**: 確定（コード確認済み）。

現状:

```cpp
// DSPLifetimeManager.h:42
const uint64_t epoch = router_->publishEpoch();  // ← epoch を ADVANCE
```

問題の本質は、**Publish Event と Retire Event が同一の Epoch Source を共有している**点。

Practical Stable ISR Bridge Runtime では:

```
Epoch = Publication Generation
```

であるべき。しかし現在は:

```
Epoch = Publication Count + Retire Count
```

になっている。これにより:

- Epoch の意味（世代境界）が崩れる
- safe-epoch 解析および診断基盤を複雑化する
- Telemetry 上の世代解析が困難になる
- 将来の Reader Stall 診断の基盤が歪む

ただし注意すべき点として、**epoch 加速が即座に unsafe free を引き起こすわけではない**。
Reader が保持している epoch が `readerEpoch <= retireEpoch` の条件で保護されている限り、
RCU 安全性そのものは維持される可能性が高い。

したがって問題の本質は **「RCU 安全性破壊」ではなく「Epoch の意味の汚染」**である。
Health 系・Diagnostics 系の基盤全体に波及する構造的問題ではあるが、
安全上の深刻度ではなく「意味の純度」の観点から S 評価としている。

### 影響範囲

| コンポーネント | 影響 |
|---|---|
| `EpochDomain::publishEpoch()` | `fetchAddAtomic(globalEpoch, 1)` — インクリメント |
| `ISRRetireRouter::publishEpoch()` | 委譲経路 |
| `AudioEngine.Publication.cpp` | 同一経路使用 |
| `AudioEngine.CtorDtor.cpp` | graceful drain でも使用 |

### 優先度: **S**（構造的意味論の問題）

---

## S-2. Health→Control Pipeline 未完成

**検証結果**: 確定（コード確認済み）。

`PublicationAdmission` のみ HealthState が配線済みだが、以下の4経路が未接続:

| 経路 | 現状 | リスク |
|---|---|---|
| **Rebuild Admission** | `shouldRejectRebuildAdmissionForPressure()` は retire pressure のみ参照 | Critical でも Rebuild 継続 → 負荷増大 |
| **RuntimeBuilder** | `RuntimeBuilder::build()` に HealthState 参照なし | Critical でも Build 実行 → リソース競合 |
| **CrossfadeAuthority** | `CrossfadeAuthority::evaluate()` に HealthState 参照なし | Critical でも Crossfade 開始可能 → 新旧同時保持で圧力倍増 |
| **DSPTransition** | `DSPTransition::onPublishCompleted()` に HealthState 参照なし | 上記同 |

あるべき制御経路:

```
Health Monitor (ISRHealthState)
    ↓
Admission Layer (Publication / Rebuild)
    ↓
Builder (RuntimeBuilder)
    ↓
Crossfade (CrossfadeAuthority / DSPTransition)
    ↓
Publish (PublicationExecutor)
```

現在は `PublicationAdmission → HealthState` のみ配線済み。
残り4経路が未接続のため、Health Monitor が Critical を検出しても
システム全体としての防御が効かない。

### 優先度: **S**

---

# A — 実運用で顕在化する可能性がある課題

## A-1. Health→Recovery 未完成

**検証結果**: 確定（コード確認済み）。

`onHealthEvent()` (`AudioEngine.Timer.cpp:534-565`) は存在するが、
実質的にログ出力のみ。Crossfade Timeout の回復処理のみ例外。

Practical Stable ISR Bridge Runtime の理想は **Self-Protecting Runtime**:

```
Reader Exhaustion → Admission Stop
Publication Stall → Crossfade Stop
Retire Stall     → Builder Throttle
Current: Detect → Log
```

### 優先度: **A**

---

# B — 運用診断能力の課題

## B-1. Reader Ownership Telemetry 未完成

**検証結果**: 確定（コード確認済み）。

`HealthEvent` に `readerIndex` / `readerEpoch` / `readerDepth` / `residencyTimeUs`
が定義されているが、`checkReaderSlotUsage()` で実際に埋められていない。

設計者自身が構想していたが未完成のまま残っている可能性が高い。
ただし、これがなくても RCU/Reader 保護/メモリ解放は機能するため、
システム破綻要因ではない。運用診断品質の問題。

### 優先度: **B+**

---

# C — 診断品質・保守性の改善

## C-1. RuntimeDrainAudit と WorldLifecycleAudit の連携不足

**検証結果**: 確定。

`WorldLifecycleAudit` は `activeWorldCount` / `publishedCount` / `retiredCount` を保持しているが、
`RuntimeDrainAudit` が参照していない。Shutdown 完了判定の補強材料。

Practical Stable ISR Bridge Runtime の必須条件ではなく、診断品質向上の領域。

### 優先度: **C**

---

## C-2. BuildError 分類不足

**検証結果**: 現状でも `ResourceUnavailable` / `WarmupFailed` / `InternalError` まで分類済み。
不足しているのは `MKLFailure` / `ConvolverFailure` / `PrepareFailure` の細分化。
例外依存そのものより診断能力の不足が本質。

### 優先度: **C**

---

## C-3. 固定64 Reader

**検証結果**: 現状のスレッド構成では 64 は実用上十分。問題は数値でなく枯渇時の可観測性。
`64 → 128` 変更より `Ownership Telemetry` 整備が優先。

### 優先度: **C**

---

# 達成度再評価

コード検証結果を反映:

| 項目 | 達成度 |
|---|---|
| Authority 分離 | 95% |
| Publication Runtime | 95% |
| Retire Runtime | 82% |
| Reader Safety | 90% |
| Health Monitoring | 90% |
| Health→Control 連携 | 65% |
| Health→Recovery | 50% |
| Shutdown Runtime | 80% |
| Diagnostics | 80% |
| Recovery Strategy | 60% |

**総合: 約80〜85%達成**。

---

# 残る構造的課題 — Health Governance が最大の未達

Practical Stable ISR Bridge Runtime の観点では、「2.5本柱」というより
**最大の未達は Health Governance（Health→Control + Health→Recovery）であり、
Epoch 純化はそれに次ぐアーキテクチャ整合性課題** と整理するのが正確。

ConvoPeq の Health Monitoring は既にかなり成熟している。不足しているのは:

| 領域 | 現状 | あるべき姿 |
|---|---|---|
| **Health Governance** | Observe は完成、Decide は一部、Act は未完成 | Observe → Decide → Act の閉ループ |
| **Epoch Semantics** | Retire が publishEpoch() を呼ぶ | Retire は currentEpoch() 読み取り |
| **運用品質** | Reader Ownership Telemetry 未完成 / BuildError 分類不足 / DrainAudit連携不足 | 診断性・可観測性の向上 |

### 優先順位

| Tier | 課題 | ID | 理由 |
|---|---|---|---|
| **Tier 1** | Health→Control Pipeline 完成 | S-2 | Health=Critical でも負荷増大を防ぐ。最大の運用リスク。 |
| **Tier 1** | Epoch Semantics 純化 | S-1 | アーキテクチャ整合性。最小工数（1行修正）。 |
| **Tier 2** | Health→Recovery（自己防衛） | A-1 | Control の拡張版。S-2 完了後に実装。 |
| **Tier 3** | Reader Ownership Telemetry | B-1 | 診断品質。実運用価値大。 |
| **Tier 4** | C 群（DrainAudit連携 / BuildError / 固定64） | C | 保守性・診断性の向上。 |

### 最終見解

上記が解消されれば、残る課題は主に診断性・可観測性・保守性の領域となり、
アーキテクチャ上の主要な未達事項はほぼ解消されたと評価してよい。
ただし「ほぼ解消」は「完全に解消」ではない点に注意が必要。

---

# 補足

## 補足A: S-1は「安全性」より「意味論」の問題

S-1 の優先度は S のままで妥当だが、その理由は「Unsafe Free の危険」ではなく
**「Epoch Semantics の崩れ」** にある。

つまり:

- ❌ 安全性S（Reader 保護が破綻する）
- ✅ **アーキテクチャ整合性S**（Epoch の世代境界としての意味が失われる）

現状コードから確認できるのは `retireEpoch = publishEpoch()` という事実であり、
`readerEpoch <= retireEpoch` による保護が破綻することまでは証明されていない。
RCU 安全性そのものは維持される可能性が高い。

したがって「安全上の緊急度」ではなく「意味の純度の観点からの構造的課題」と
位置付けるのが正確である。

---

## 補足B: S-2は最終的に Admission Policy へ集約される

将来的な理想像としては、個別に `if (health == Critical)` を各所に散在させるより、
**AdmissionPolicy** または **RuntimeGovernancePolicy** に集約した方が保守性が高い。

例:

```cpp
HealthDecision decision = governancePolicy.evaluate(currentHealth);
```

これは現時点の未達事項ではなく、将来の設計指針として有効。

---

## 補足C: 実装順序の考察

優先度分類と実装順序は必ずしも一致しない。実運用リスクの観点では:

| 順序 | 課題 | 理由 |
|---|---|---|
| **1** | **S-2 Health→Control Pipeline** | Health=Critical なのに負荷を増やし続ける状態を防ぐ。最も直接的な運用リスク。 |
| **2** | **A-1 Health→Recovery** | 異常発生後の自己防衛。Detect→Act の閉路を作る。 |
| **3** | **S-1 Epoch 意味論** | アーキテクチャ整合性の回復。安全性破綻リスクは低いが、診断基盤の健全性のために重要。 |

S-1 が最も発見が容易で修正範囲も小さい（`publishEpoch()` → `currentEpoch()` の一行変更）ため、
工数対効果ではむしろ最初に着手しやすい可能性もある。

---

## 総評: 残り15〜20%は「機能不足」ではなく「ガバナンス不足」

成熟した Runtime は通常、**Observe → Decide → Act** の3段階を構成する。
ConvoPeq の現状:

| 段階 | 状態 |
|---|---|
| **Observe**（監視） | かなり完成 — Health Monitoring / Reader Slot / Publication Stall 等 |
| **Decide**（判断） | 一部完成 — PublicationAdmission のみ HealthState 反映済み |
| **Act**（実行） | ほぼ未完成 — ログ出力のみ、自己防衛機構なし |

残っているのは「監視した結果をどう制御に反映するか」という**運用安定化の段階**。
残り15〜20%の大半は機能不足ではなくガバナンス不足である。

ConvoPeq は既に以下を実装している:

- Publication / Crossfade / Retire / Epoch / RCU / Health Monitoring

残っているのは「監視した結果をどう制御に反映するか」という**運用安定化の段階**。
残り15〜20%の大半は機能不足ではなくガバナンス不足である。

### 最終結論

Practical Stable ISR Bridge Runtime の観点では、2.5本柱（Epoch純化 / Health→Control / Health→Recovery）が
解消されれば、残る課題は主に診断性・可観測性・保守性の領域となり、
アーキテクチャ上の主要な未達事項はほぼ解消されたと評価してよい。

---

## 付録: 検証に使用したツール

| ツール | 用途 |
|---|---|
| AiDex MCP | 識別子検索、ファイル構造把握、セッション管理 |
| grep/Select-String | パターン検索、catch(...)/publishEpoch等の網羅的抽出 |
| CodeGraph MCP | グラフベースのコード解析（global_search） |
| read_file (直接) | 該当ファイルの全文確認 |
