# ConvoPeq ISR Bridge Runtime 詳細設計 v7.3

- Project: ConvoPeq
- Date: 2026-05-28
- Basis:
  - `doc/work4/ConvoPeq_ISR_Bridge_Runtime_Operational_Survivability_基本設計_v7_2026-05-28.md`
  - `doc/work4/ConvoPeq_ISR_Bridge_Runtime_AI詳細設計・実装統制規約_v7.3_2026-05-28.md`
- Scope: Tier0/Tier1 の実装詳細
- Priority: **Operational Safety > Architectural Purity**

---

## 0. 目的

本書は、v7基本設計とv7.3規約を、実装可能な粒度に落とした詳細設計である。

固定するもの:

1. authority 境界（admission / drain / reclaim / residency）
2. 実装順序（Tier0優先）
3. fail-closed な CI 検証接続
4. rollback 可能な最小差分導入

---

## 1. 非目標

Tier0/Tier1 では次を行わない。

- runtime graph 全面再設計
- epoch system 再実装
- lock-free 全面 rewrite
- shared_ptr 全面導入
- crossfade architecture 全面刷新
- execution semantic 多段化（Replaceable/MustExecute 以外）

---

## 2. リスクモデル

主要リスク:

- admission bypass
- drain authority ambiguity
- reclaim escalation / starvation
- residency ownership ambiguity

破綻連鎖:

```text
bypass path から rebuild 流入
→ saturation suppression 不発
→ residency 増加
→ reclaim 慢性飽和
→ shutdown 収束失敗
→ unload failure
```

---

## 3. Authority 境界（固定）

## 3.1 Admission authority

- 唯一の外部入口: `submitRebuildIntent(...)`
- `requestRebuild(...)`: internal execution primitive
- 外部直呼び禁止（Timer/UI/helper/test/legacy/worker）

## 3.2 Drain authority

- 唯一の completion authority: `coordinator.isFullyDrained()`
- queue empty / telemetry 単独判定は authority としない

## 3.3 Reclaim authority

- global coordinator が単一 authority
- subsystem-local reclaim policy 禁止

## 3.4 Residency authority

- owner / producer / reclaim trigger / drain authority を固定
- comment-only 管理は禁止（enum/contract として定義）

---

## 4. Rebuild 詳細設計

## 4.1 Funnel 構造

```mermaid
flowchart TD
    EXT[Timer/UI/Helper/Test/Legacy] --> INTENT[submitRebuildIntent]
    INTENT --> ADMIT[Admission Gate + Suppression]
    ADMIT -->|accepted| EXEC[requestRebuild (internal only)]
    ADMIT -->|suppressed| TEL[Suppression Telemetry]
```

## 4.2 Suppression 規約

- suppression は funnel 内でのみ実行（Rule-1M）
- enqueue 後 suppression 禁止
- MustExecute reject 禁止

### 4.2.1 Suppression 定義域（混同防止）

Tier0/Tier1 では suppression を次の 3 ドメインに分離する。

1. `RebuildIntentSuppression`（admission funnel 専属）
2. `QueueAdmissionSuppression`（queue/backlog 保護）
3. `SnapshotDropSuppression`（command/snapshot buffer 保護）

設計上の拘束:

- Rule-1M の対象は `RebuildIntentSuppression` に限定する。
- `QueueAdmissionSuppression` / `SnapshotDropSuppression` を funnel 規約違反として誤判定してはならない。
- telemetry には suppression domain を必須タグとして記録する。

### 4.2.2 Tier0 固定 suppression reason

Tier0 で増設可能な reason は次に限定する。

- Saturation
- Duplicate
- Shutdown
- Obsolete
- InvalidState

上記以外の reason 追加は Tier1 review mandatory とする。

## 4.3 Hysteresis

- Enter: `retireQueueDepth_ >= retireHighWatermark_`
- Exit: `retireQueueDepth_ <= retireLowWatermark_`
- 常時: `retireHighWatermark_ > retireLowWatermark_`
- flapping 禁止

## 4.4 Collapse

- latest-generation-wins を維持
- obsolete collapse は削除不可

## 4.5 Execution semantic

Tier0/Tier1 は 2 値固定:

```text
Replaceable
MustExecute
```

実装型マッピング:

```text
Current implementation type: RebuildTelemetryPolicy
Canonical governance semantic: RebuildExecutionClass
```

---

## 5. Snapshot / Seal 詳細設計

順序固定:

```text
capture -> finalize -> seal -> publish
```

禁止:

- unsealed snapshot publish
- seal 後 mutation
- RuntimeBuildSnapshot bypass publish

worker 契約:

- sealed snapshot のみ参照
- runtime pointer 起点補正禁止

---

## 6. Drain / Shutdown 詳細設計

## 6.1 waitForDrain 意味論

`waitForDrain(timeout)` は **bounded convergence observation API**。

- true: timeout 内に bounded convergence を観測
- false: convergence 未確認
- full quiescence 保証ではない

## 6.2 非RT制約

- non-RT only
- RT thread 呼び出し禁止
- infinite blocking wait 禁止

## 6.3 phase bypass 禁止

`waitForDrain()` 成功後でも phase machine を通さない解放は禁止。

禁止:

- direct reclaim during unload
- direct queue clear
- direct coordinator reset
- force-retire without shutdown phase gate

## 6.4 isFullyDrained 集約項目

最低集約:

- publication backlog
- pending publication intents
- retire residency
- fallback residency
- reclaim in-flight
- publication coordinator staging

---

## 7. Reclaim 詳細設計

## 7.1 bounded cadence 必須

- max reclaim iterations per cycle
- min reclaim interval
- reclaim pressure upper bound

## 7.2 starvation 防止

- reclaim starvation timeout
- reclaim retry escalation ceiling
- obsolete residency prioritization

## 7.3 emergency reclaim

許可:

- cadence boost
- obsolete prioritization
- fallback aggressive drain

禁止:

- synchronous full drain
- RT wait
- stop-the-world reclaim

---

## 8. Residency 詳細設計

## 8.1 基本契約

全 residency は以下を持つ。

- producer
- owner
- reclaim trigger
- drain authority
- boundedness

## 8.2 固定テーブル

| Residency | Producer | Owner | Reclaim Trigger | Drain Authority |
| --- | --- | --- | --- | --- |
| retire queue | publication coordinator | coordinator | reclaim scheduler | coordinator |
| fallback queue | publication bridge | coordinator | drain scheduler | coordinator |
| deferred retire | epoch retire path | coordinator | epoch advance | coordinator |
| epoch retire staging | epoch retire path | coordinator | epoch advance | coordinator |

### 8.2.1 Residency Authority のコード固定（例）

```cpp
enum class ResidencyAuthority : uint8_t
{
  PublicationCoordinator,
  DeferredDeleteFallback,
  EpochRetire,
  ShutdownDrain,
};
```

### 8.2.2 Boundedness 定量契約（必須）

各 residency は次を明示する。

- hard upper bound
- warn threshold
- force-drain trigger

未定量 residency は Tier0 受入不可。

## 8.3 新規 residency 追加ルール

新規 queue/container は mandatory review。

PR 必須記載:

```text
[Residency Contract]
Name:
Producer:
Owner:
Reclaim Trigger:
Drain Authority:
Boundedness:
Shutdown Completion Path:
```

---

## 9. Telemetry 詳細設計

## 9.1 役割

telemetry は visibility 用。authority ではない。

## 9.2 write authority 固定

```text
Telemetry counters may only be mutated by their declared owner subsystem.
```

禁止:

- cross-subsystem increment
- helper-side mutation
- debug utility mutation

## 9.3 suppression telemetry（最低）

- saturation reject count
- Replaceable suppress count
- MustExecute bypass count
- duplicate collapse count

---

## 10. CI / Lint 詳細設計（v7.3準拠）

実装案ドキュメント連携:

- `doc/work4/ConvoPeq_ISR_Bridge_Runtime_v7.3_CIチェック実装案_grep_lint_2026-05-28.md`

## 10.1 チェックID

| Check ID | Purpose | Fail条件 |
| --- | --- | --- |
| CI-ADMISSION-001 | requestRebuild直呼び禁止 | allowlist外 call site 検出 |
| CI-ADMISSION-002 | execution semantic 拡張禁止 | 禁止トークン検出 |
| CI-ADMISSION-003 | suppression 分散禁止 | funnel外 suppression 検出 |
| CI-SHUTDOWN-001 | phase bypass 禁止 | unload shortcut 検出 |
| CI-RECLAIM-001 | bounded cadence 必須 | 主要パラメータ欠落 |
| CI-RESIDENCY-001 | queue増殖監視 | 未登録 residency 検出 |
| CI-TELEMETRY-001 | write authority 検証 | owner外更新検出 |

## 10.2 CI-1 厳密化

`requestRebuild(` 検出ルール:

- definition site 除外
- allowlisted funnel 実装除外
- その他 direct invocation は fail

### 10.2.1 段階適用（移行期間）

現行コードに残存する direct call を考慮し、Tier0 期間は段階適用を許可する。

- Phase-A: warn（allowlist 外を可視化）
- Phase-B: fail（allowlist 外を CI fail）

移行完了条件:

- `requestRebuild(` の外部経路直呼びが allowlist 0 件
- `submitRebuildIntent(...)` へ全収束

### 10.2.2 allowlist 管理要件

allowlist 各項目に必須:

- owner
- issue
- rationale
- expiry

expiry 超過エントリは fail-closed で CI fail とする。

## 10.3 ポリシーJSON

`.github/isr-ai-governance-policy.json`（提案）で allowlist/owner を管理する。

---

## 11. 実装順序（Tier0→Tier1）

## Tier0

1. single admission funnel 封鎖
2. saturation suppression 入口実装
3. drain SSOT 固定
4. bounded reclaim cadence
5. residency ownership/producer 固定

## Tier1

1. starvation 防止の閾値最適化
2. suppression telemetry 拡張
3. 例外 shim の整理（semantic 拡張なし）

---

## 12. 受入判定

Tier0 完了条件:

1. 全 rebuild 要求が funnel 経由
2. Replaceable suppress が saturation 下で機能
3. MustExecute starvation がない
4. `waitForDrain(timeout)` が bounded convergence 契約を満たす
5. `isFullyDrained()` が単一 authority
6. reclaim cadence が bounded
7. residency owner/producer/trigger が固定表に一致

Tier1 完了条件:

- Tier0 + reclaim starvation 指標安定
- suppression reason telemetry が継続観測可能

---

## 13. Rollback 方針

- 1PR 1目的
- 失敗時は check 単位で rollback 可能に分離
- authority 増殖につながる変更は即時差し戻し

---

## 14. 最終原則

Tier0/Tier1 の判断基準は次で固定する。

```text
Operational Safety > Runtime Survivability > Boundedness > Purity
```

AI は「きれいにする」より「壊れにくくする」を優先する。
