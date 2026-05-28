# ConvoPeq ISR Bridge Runtime Operational Survivability 基本設計 v7

- Project: ConvoPeq
- Date: 2026-05-28
- Scope: `doc/work4/` 向け統合基本設計
- Goal: **理論的な ISR purity の完成ではなく、DAW 長時間運用で破綻しにくい runtime を実装契約として固定すること**

---

## 0. 本書の目的

本書は、これまでのレビュー結果を統合し、Tier0 実装で迷わないための実装基準を定義する。

重視するのは設計の美しさではなく、以下の運用リスク抑止である。

1. admission bypass による rebuild 流入再発
2. drain authority 分散による shutdown 非決定化
3. reclaim pressure 暴走による CPU starvation
4. residency ownership 不明確化による滞留・解放漏れ

---

## 1. 最上位方針（Operational Safety First）

優先順位は以下で固定する。

1. crash / UAF 防止
2. runaway growth 防止
3. deterministic shutdown completion
4. bounded residency
5. bounded reclaim pressure
6. long-session survivability
7. ISR purity

**原則:**

- `Operational Safety > Architectural Purity`
- Tier0 では大規模再設計を禁止し、最小差分で enforceability（強制可能性）を上げる。

---

## 2. 現状リスクモデル（ConvoPeq 向け）

本フェーズでの主要リスクは以下と定義する。

- `mutable impurity` ではなく、
- `runaway growth`
- `authority ambiguity`
- `reclaim escalation`
- `admission bypass`

特に危険なのは次の連鎖である。

```text
bypass path から rebuild 流入
→ saturation 抑止不発
→ residency 増加
→ reclaim 飽和
→ shutdown 収束失敗
→ unload failure
```

---

## 3. Authority モデル（単一権威）

### 3.1 Single Admission Authority

- rebuild 要求の唯一の authority は `submitRebuildIntent()` とする。
- `requestRebuild(sr, bs, ...)` は **internal execution primitive** に降格する。
- 外部経路（timer/helper/legacy/UI/parameter 復元）からの直接 `requestRebuild(...)` を禁止する。

### 3.2 Single Drain Authority

- drain completion 判定は `coordinator.isFullyDrained()` に集約する。
- queue 個別判定・telemetry 判定を completion authority として使用しない。

---

## 4. Tier0 実装契約（OPS ルール）

## OPS-1: Single admission funnel

全 rebuild 要求は `submitRebuildIntent()` 経由のみ許可する。

## OPS-2: 実行クラスは 2値維持

Tier0 では `Replaceable / MustExecute` のみ。

- 3段階以上の execution class 拡張は禁止。
- 優先度差は execution class ではなく suppression reason table で吸収する。

## OPS-3: Bounded reclaim cadence

reclaim 実行には以下を必須化する。

- `max reclaim iterations per cycle`
- `minimum reclaim interval`
- `bounded reclaim pressure escalation cap`
- `no recursive reclaim dispatch`

## OPS-4: waitForDrain の意味論固定

`waitForDrain(timeout)` は**完全静止保証 API ではない**。

意味は次で固定する。

```text
waitForDrain()==true
= timeout 内で bounded convergence が観測された
```

## OPS-5: Single drain authority

`coordinator.isFullyDrained()` 以外を shutdown completion authority にしない。

## OPS-6: Residency ownership table 必須

residency ごとに owner / reclaim trigger / drain authority を固定定義する。

## OPS-7: Suppression reason telemetry 必須

saturation 抑止の理由を reason 別に計測し、運用解析可能にする。

## OPS-8: Tier0 での設計拡張禁止

以下は禁止。

- scheduler 大改造
- lock-free 全面 rewrite
- epoch 再実装
- shared_ptr 全面化
- purity 目的の crossfade 大規模変更

---

## 5. Suppression 設計（Tier0）

### 5.1 基本原則

- enqueue 後 reject は禁止。
- admission 入口で判定する。
- MustExecute は saturation でも通す。
- Replaceable は reason-weighted suppression を適用する。

### 5.2 SuppressionReason（推奨定義）

```cpp
enum class SuppressionReason : uint8_t
{
    RetireQueueHighWatermark,
    ShutdownInProgress,
    DuplicateReplaceable,
    ReclaimPressure,
    PublicationBacklog,
};
```

### 5.3 Telemetry

- `suppressedReasonCounters[reason]` を保持する。
- 観測結果は flow control 調整に使い、completion authority には使わない。

---

## 6. Residency Ownership 固定

Tier0 では理想統合より「所有権の固定」を優先する。

### 6.1 推奨 enum

```cpp
enum class ResidencyAuthority : uint8_t
{
    PublicationCoordinator,
    DeferredFallback,
    EpochRetire,
    EmergencyRetire,
    ShutdownDrain
};
```

### 6.2 最低契約

各 residency type について次を定義する。

- owner
- reclaim trigger
- drain authority
- shutdown completion path

---

## 7. Reclaim Budget（Global）

subsystem ごとの局所最適化を禁止し、global budget authority を持つ。

- ローカル reclaim pressure の無制限増幅を禁止
- 全体で cadence/budget を一元管理

---

## 8. Tier0 実装順序（強制順）

1. **Tier0-A**: single admission funnel 封鎖（bypass 経路遮断）
2. **Tier0-B**: saturation suppression 実装（Replaceable 抑止 / MustExecute 通過）
3. **Tier0-C**: drain SSOT 固定（`isFullyDrained()` authority 集約）
4. **Tier0-D**: bounded reclaim cadence 導入
5. **Tier0-E**: residency ownership table 固定

理由:

```text
流入を止めない限り、reclaim は勝てない
```

---

## 9. CI / Lint 強制（Enforcement）

設計規約ではなく、機械的 enforcement を必須にする。

- 外部からの `requestRebuild(...)` 直呼びを CI で fail
- rebuild 要求経路の funnel 逸脱を static check
- Tier0 完了まで bypass を fail-closed

---

## 10. 受入基準（Tier0）

以下を満たした場合のみ Tier1 へ進む。

1. 全 rebuild 要求が単一 admission funnel 経由
2. saturation 下で Replaceable 抑止が機能
3. MustExecute starvation がない
4. `waitForDrain(timeout)` が bounded convergence 契約で運用される
5. drain completion authority が単一化されている
6. reclaim cadence が bounded（暴走なし）
7. residency owner/reclaim/drain が表で固定されている

---

## 11. 非目標（Tier0 ではやらない）

- ISR purity 100% 化
- crossfade purity の全面追求
- runtime/epoch/publication の再設計
- execution class 多段化

---

## 12. 結論

ConvoPeq の現在地で最優先なのは、次の 4 点である。

- single admission enforcement
- single drain authority
- bounded reclaim pressure
- residency ownership fixation

Tier0 では「理想化」ではなく「破綻防止」を完成させる。

```text
Stop inflow first.
Then bound growth.
Then guarantee convergence.
```
