# ConvoPeq ISR Bridge Runtime v6.x 詳細設計（統合版）

- Project: ConvoPeq
- Date: 2026-05-28
- Basis: `doc/work3/ConvoPeq_ISR_Bridge_Runtime_v6.x_最終統制文書_2026-05-28.md`
- Status: Detailed Design Freeze Candidate (Unified)
- Priority: **DAW 実運用耐性 > 理論純度**

---

## 0. 本書の目的

本書は、v6.x 最終統制文書を実装可能な粒度に分解し、ConvoPeq の ISR Bridge Runtime を段階的に hardening するための詳細設計を定義する。

本書が固定するのは次の 3 点である。

1. 変更対象の責務境界
2. 実装順序と rollback 可能性
3. 受入判定と検証観点

---

## 0.1 統合版の位置づけ

本書は、先行版（v0.1）と改訂版（v0.2）を一本化した統合版である。

- v0.1 の基本構造を維持
- v0.2 の Critical / High 改善点を正規取り込み
- 実装時の解釈ブレを最小化

---

## 1. 非目的

以下は本詳細設計で目的化しない。

- RuntimeGraph の全面再設計
- publication coordinator の全面 rewrite
- epoch system の再実装
- 全面 lock-free 化
- DSP full immutable 化
- shared_ptr pervasive 化
- crossfade 全面刷新

本書は、既存 hardening を壊さずに operational survivability を完成させることだけを目的とする。

### 1.1 矛盾時の優先順位（詳細設計内）

本詳細設計内で矛盾が生じた場合、次を優先する。

1. §17 受入判定
2. §10 Drain / Shutdown
3. §5 Unified Admission Gate
4. その他セクション

本詳細設計と v6.x 最終統制文書が矛盾する場合は、最終統制文書 §1 の優先順位に従う。

---

## 2. Failure Mode と封じ込め方針

### 2.1 対象 failure mode

- rebuild storm
- retire saturation
- shutdown race
- unload failure
- stale runtime reuse
- publication race
- crossfade authority drift
- telemetry misinterpretation

### 2.2 封じ込めの優先軸

1. runaway prevention
2. bounded residency
3. deterministic shutdown
4. reclaim convergence
5. long-session survivability
6. RT safety
7. ISR purity

### 2.3 基本的な封じ込め方針

- 入口で止める
- 後段で救済しない
- 測定だけで終わらせない
- 既存 hardening は再実装しない
- AI が誤解しやすい境界を先に固定する

---

## 3. 設計対象

### 3.1 中核コンポーネント

- `AudioEngine`
  - admission
  - publication
  - rebuild dispatch
  - shutdown orchestration
- `RuntimePublicationCoordinator`
  - publication world 管理
  - drained 判定
  - staging / swap / retire 集約
- `RuntimeBuildSnapshot`
  - worker 入力の immutable 化
  - fingerprint / finalize / seal
- `ShutdownRuntime`
  - shutdown phase control
  - bounded teardown counters
- `RetireRuntime` / `RetireRuntimeEx`
  - retire / reclaim / quarantine の運用
- `Crossfade Execution Path`
  - prepare / activate 分離

### 3.2 スレッド責務

- Audio Thread: consume-only
- Message Thread: admission / publication / telemetry / lifecycle control
- Rebuild Worker: sealed snapshot ベース再構築
- Retire / Reclaim 処理: non-RT 側で best effort に実行

### 3.3 実行ローカル mutable の扱い

許容するのは次のローカル状態のみである。

- local fade progression
- thread-local scratch
- smoothing accumulators
- readonly cache materialization

これらは runtime authority を持たないこと。

### 3.4 shutdown authoritative state

shutdown phase の authoritative state は `shutdownPhase` とする。

- `lifecycleState`: `shutdownPhase` からの read-only mirror
- `shutdownRuntime`: state machine を駆動するコマンド実行体（state 所有者ではない）

---

## 4. 全体アーキテクチャ

### 4.1 主要フロー

```mermaid
flowchart LR
    UI[Message Thread / UI] -->|acceptsRuntimePublication()| ADMIT[Unified Admission Gate]
    ADMIT -->|capture| CAP[RuntimeBuildSnapshot capture]
    CAP -->|finalize| FIN[deterministic finalize]
    FIN -->|seal| SEAL[sealed snapshot]
    SEAL -->|handoff| WORKER[Rebuild Worker]
    WORKER -->|build| BUILD[Runtime build]
    BUILD -->|prepareCommit / executeCommit| PUB[RuntimePublicationCoordinator]
    PUB -->|publish| AUDIO[Audio Thread consume-only]
    PUB -->|retire| RETIRE[Retire / Reclaim Path]
    RETIRE -->|drain| DRAIN[coordinator.isFullyDrained()]
    DRAIN -->|shutdown complete| SHUT[ShutdownRuntime]
```

### 4.2 設計の収束点

- publication authority を分散しない
- drained 判定を分散しない
- telemetry を truth source に昇格させない
- queue topology を増やしすぎない
- Tier0 未完了で Tier2 に進まない

---

## 5. Unified Admission Gate 詳細設計

### 5.1 目的

publication / rebuild / commit の入口を単一路線化し、shutdown 後の resurrection と storm 流入を止める。

### 5.2 API

```cpp
bool acceptsRuntimePublication() const noexcept;
```

### 5.3 適用ポイント

- `requestRebuild`
- `submitRebuildIntent`
- `prepareCommit`
- `executeCommit`
- `appendPublicationIntent`

### 5.4 判定方針

- Running: accept
- それ以外: reject

### 5.4.1 atomic 契約

- `acceptsRuntimePublication()` は `lifecycleState` を `memory_order_acquire` で単一読み取りする。
- `lifecycleState` 遷移の publish 側は `memory_order_release` とし、shutdown 遷移観測を順序保証する。
- 判定の有効期間は呼び出しスコープ内のみとする。

### 5.4.2 TOCTOU 対策

- Admission 判定後の enqueue / publish 直前で、coordinator 内で再判定（double-check）を行う。
- ShuttingDown 観測後に到達した enqueue は破棄する。

### 5.5 失敗時の振る舞い

- side effect を最小化する
- queue へ積まない
- retry を勝手に行わない
- telemetry は increment のみ許可する場合がある

### 5.6 実装上の注意

admission 判定を後段にずらしてはならない。

悪い例:

```cpp
queue.push(task);
if (!acceptsRuntimePublication())
    rejectLater();
```

良い例:

```cpp
if (!acceptsRuntimePublication())
    return;
```

---

## 6. Snapshot / Seal 詳細設計

### 6.1 Snapshot lifecycle

```text
capture -> finalize -> seal -> handoff
```

### 6.2 `capture` の責務

- 入力を収集する
- runtime pointer へ依存しない
- worker 再構築に必要な情報をまとめる

### 6.3 `finalize` の責務

finalize は deterministic normalization である。

許可:

- canonical ordering
- parameter normalization
- immutable fingerprint generation
- bounded validation

禁止:

- wall clock 依存
- thread-order 依存
- allocation order 依存
- pointer identity 依存
- environment-derived mutation
- hidden patch

### 6.4 `seal` の責務

- finalize 後の immutable 契約を固定する
- debug build では assert を推奨する
- sealed 後 mutation を禁止する

### 6.5 `handoff` の責務

- sealed snapshot を worker に渡す
- runtime pointer 参照を禁じる
- snapshot mutation を禁じる

### 6.6 fingerprint versioning

`RuntimeBuildFingerprint` の構成要素が変わる場合は `fingerprintVersion` を increment する。

### 6.6.1 fingerprint 比較規則

- 比較は field-wise 比較を原則とし、`memcmp` 等のバイナリ比較に依存しない。
- `fingerprintVersion` が不一致の場合は **same-fingerprint とみなさない**。
- version 不一致の intent は collapse 対象外とし、通常評価経路へ送る。

---

## 7. Rebuild Worker 詳細設計

### 7.1 入力契約

worker は sealed snapshot のみを読む。

### 7.2 禁止事項

- active / fading runtime pointer 参照
- old runtime traversal
- runtime pointer 由来の補正
- snapshot mutation

### 7.3 許可事項

- readonly snapshot access
- execution-local scratch 生成
- deterministic cache materialization

### 7.4 obsolete collapse

worker 実行前に obsolete 判定を行い、latest-generation-wins を維持する。

### 7.5 retry 方針

- retry は bounded
- warmup retry は timeout / saturation / shutdown を観測する
- 無限 retry は不可

### 7.5.1 retry 運用定数

- max retry: 3
- backoff: 5ms, 10ms, 20ms（指数的）
- shutdown 観測時は即中断し retry しない

---

## 8. Rebuild Collapse 詳細設計

### 8.1 目的

obsolete rebuild avalanche を collapse し、latest-generation-wins に収束させる。

### 8.2 collapse 条件

safe-to-collapse は以下が全成立した場合のみ許可する。

- non-committed
- externally invisible
- same rebuild class
- same rebuildFingerprint
- newer equivalent rebuild が存在

### 8.3 collapse 禁止領域

- prepareToPlay rebuild
- topology migration rebuild
- shutdown transition rebuild
- runtime recovery / safety rebuild
- cross-class collapse

### 8.4 入口での抑止

collision が多い場合は enqueue 前で coalescing / reject を優先する。

---

## 9. Residency / Retire 詳細設計

### 9.1 目的

retired object の残留を bounded にし、shutdown completion と reclaim convergence を一本化する。

### 9.2 管理対象

- retire queue
- fallback queue
- retiredWorlds_
- quarantine
- deferred retire residency

### 9.3 禁止対象

- unbounded residency
- permanent append-only residency
- drain condition を持たない residency

### 9.4 許可される構造

- bounded append buffer
- short-lived staging queue
- bounded temporary residency
- deterministic reclaim を伴う append structure

### 9.5 実装要求

residency container を追加する場合は、必ず次を定義する。

- boundedness
- reclaim owner
- drain condition
- shutdown completion path

### 9.5.1 レビュー強制テンプレート（必須）

新規 residency の PR には以下を必須添付する。

```text
[Residency Contract]
Name:
Boundedness:
Reclaim Owner:
Drain Condition:
Shutdown Completion Path:
Observed By (Telemetry):
```

### 9.6 巨大 Manager の禁止

- manager explosion を避ける
- coordinator layering を増やしすぎない
- queue topology を発明し直さない

---

## 10. Drain / Shutdown 詳細設計

### 10.1 `coordinator.isFullyDrained()`

shutdown completion / drain completion の source of truth は単一 coordinator state とする。

内部集約対象:

- publication backlog
- pending publication intents
- retire residency
- fallback residency
- reclaim in-flight operations
- publication coordinator staging residency
- deferred retire residency
- publication swap pending state

個別 queue の独自 drained 判定を外へ漏らしてはならない。

### 10.1.1 評価規約

- `isFullyDrained()` は coordinator 内部 atomic state からの lock-free 合成読み取りを原則とする。
- 呼び出し側で mutex を取得しない。
- Message Thread からの polling 専用とし、Audio Thread / Rebuild Worker からの呼び出しを禁止する。
- `true` は「その時点の収束観測」を意味し、再流入抑止は Admission Gate 側で担保する。

### 10.2 `waitForDrain(timeout)`

non-RT 限定の best-effort bounded wait とする。

#### 意味論

- convergence attempt
- bounded operational wait
- best-effort synchronization

#### 戻り値

- drained converged: true
- timeout reached: false
- infinite wait: prohibited

#### 禁止

- retry-until-success loop
- unbounded blocking
- RT-assisted waiting
- RT thread / audio callback からの呼び出し

### 10.2.1 timeout / poll 定数契約

- default timeout: 2000ms
- max timeout: 10000ms
- polling interval: 1ms〜5ms
- spin-only 待機禁止（sleep を伴うこと）
- timeout 到達時は `false` を返し、内部で自動再待機しない

### 10.2.2 timeout 後の動作

- timeout 後に許可されるのは **1 回のみ** の emergency reclaim boost 要求。
- `waitForDrain()` の再帰的呼び出し・連鎖呼び出しは禁止する。

### 10.3 shutdown phase control

shutdown phase machine を増殖させてはならない。

統合先:

- authoritative: `shutdownPhase`
- derived mirror: `lifecycleState`
- command executor: `shutdownRuntime`

### 10.4 resurrection 防止

shutdown 開始後に以下を禁止する。

- publish
- rebuild enqueue
- retire append

---

## 11. Retire Backpressure 詳細設計

### 11.1 基本値

- `HWM = 3072`
- `LWM = 1024`
- `HWM > LWM` を常に保証する

### 11.1.1 saturation 判定入力と hysteresis

- saturation 入力は `retireResidency + fallbackResidency` の合算 depth を用いる。
- `retireSaturationActive_ = true`: 合算 depth `>= HWM`
- `retireSaturationActive_ = false`: 合算 depth `<= LWM` かつ 1 publication cycle 経過後
- `HWM <= LWM` は構成エラーとし、debug では assert、release では初期化失敗扱いとする。

### 11.2 saturation semantics

saturation 中は stabilization direction only とする。

許可:

- HWM increase
- LWM increase
- reject aggressiveness increase
- rebuild coalescing increase
- obsolete rebuild discard increase

禁止:

- HWM / LWM decrease
- reject relaxation
- rebuild expansion
- queue growth encouragement

### 11.3 memoryPressureScale の入力源

許可:

- retire queue depth
- fallback queue depth
- rebuild backlog
- quarantine resident count
- reclaim latency
- publication backlog
- allocation retry count

禁止:

- OS global memory usage
- DAW process memory usage
- allocator opaque heuristics
- external plugin state
- system pressure callbacks

### 11.4 emergency reclaim

- best-effort 非同期
- reclaim cadence boost
- aggressive fallback drain
- obsolete retire prioritization

RT reclaim / blocking reclaim / synchronous full drain は禁止。

### 11.4.1 emergency reclaim 運用定数

- emergency reclaim worker: 1（追加 worker を生成しない）
- boost cadence: 通常 cadence の 2 倍まで
- obsolete prioritization: oldest-generation-first
- boost window: 最大 500ms（無期限 boost 禁止）

---

## 12. Crossfade 詳細設計

### 12.1 方針

crossfade purity は Tier2 扱いとする。

### 12.2 責務境界

- Message Thread: prepare / stage / immutable handoff
- Audio Thread: activate / progression / consume-only

### 12.3 mutable reduction

段階的に以下を削減する。

- `latencyDelayOld_RT`
- `latencyDelayNew_RT`
- `dspCrossfade*`

### 12.3.1 実施順序（High 指摘対応）

1. `latencyDelayOld_RT` を execution-local 読み出しへ限定
2. `latencyDelayNew_RT` を publication handoff 経由へ移行
3. `dspCrossfade*` を authority narrowing（Message->Audio 単方向）へ移行

各段階は Tier0 受入判定の 1/3/8/9/10 を満たしていることを前提に進める。

### 12.4 禁止事項

- publish 後 topology mutation
- cross-runtime mutable progression 共有
- crossfade 全面 rewrite

### 12.5 許可事項

- mutable reduction
- authority narrowing
- execution-local snapshotization

---

## 13. Telemetry 詳細設計

### 13.1 運用目的

telemetry は operational visibility のためにのみ使う。
source-of-truth へ昇格させてはならない。

### 13.2 必須 counter

少なくとも以下を対象とする。

- saturationEnterCount
- saturationExitCount
- rebuildRejectCount
- collapseCount
- reclaimBoostCount

### 13.3 ownership hardening

各 counter は必ず次を持つ。

- owner subsystem
- increment condition
- decrement / reset condition
- shutdown finalization behavior

複数 subsystem が同一 counter を更新してはならない。

### 13.3.1 owner 既定割当

- saturationEnter/Exit: `RetireRuntimeEx`
- rebuildReject/collapse: `RebuildDispatch`
- reclaimBoost: `RetireRuntime`

### 13.4 finalization policy

shutdown finalization 時は、各 counter ごとに次のいずれかを明示する。

- freeze final value
- export snapshot
- explicit reset

暗黙 reset 禁止。

### 13.5 telemetry の禁止用途

以下は禁止する。

- lifecycle 判定
- shutdown completion 判定
- reclaim safety 判定

---

## 14. AI 実装統制詳細設計

### 14.1 目的

AI の誤解・暴走・過剰実装を抑止する。

### 14.2 典型的な暴走パターン

- 理論的に正しいからという理由で redesign する
- telemetry を truth source にする
- append-only residency を増やす
- shutdown completion source を分散する
- reclaim ownership を再分散する
- queue topology を再発明する
- Tier0 未完了状態で purity expansion を行う

### 14.3 AI に禁止すること

- subsystem rewrite
- ownership semantics の全面変更
- reclaim model の再発明
- queue topology の全面変更
- manager explosion
- coordinator layering
- dispatcher nesting

### 14.4 AI に優先させること

- incremental hardening
- bounded behavior
- deterministic teardown
- operational stability
- survivability under automation storm

---

## 15. Tier Execution Policy

原則として、Tier0 完了前に Tier2 / Tier3 の拡張へ進んではならない。

ただし、以下のみ例外として許可する。

- Tier0 blocker fix
- crash fix
- unload failure fix
- RT safety regression fix

例外修正は次を必須とする。

- minimum diff
- bounded scope
- no architectural expansion
- no semantic broadening

---

## 16. 実装シーケンス

### 16.1 Tier0

1. saturation admission suppression
2. residency accounting 一本化
3. `waitForDrain(timeout)`
4. emergency reclaim pressure

### 16.2 Tier1

1. quarantine telemetry 実体化
2. reclaim prioritization
3. saturation tuning

### 16.3 Tier2

1. crossfade mutable reduction
2. deeper seal propagation

---

## 17. 検証設計

### 17.1 静的検証

- Gate 未適用入口の検出
- worker 側 runtime pointer 参照の検出
- drained 後 resurrection 経路の検出
- RT path 禁則の検出
- telemetry の source-of-truth 化の検出

### 17.2 挙動検証

- shutdown 反復（start / stop 連続）
- rebuild burst（UI 連続変更）
- saturation 誘発（retire backlog 負荷）
- IR spam / automation spam
- crossfade 連続遷移

### 17.3 受入判定

以下 10 条件を全満足で完了判定する。

1. worker が runtime object を直接参照しない
2. shutdown 中 publication 不可能
3. saturation policy 実装済み
4. rebuild collapse deterministic
5. stale runtime reuse 不可能
6. crossfade authority shared mutable state 不在
7. RT path mutex / allocation 不在
8. deterministic shutdown 成立
9. fallback 含め drain deterministic
10. finalize deterministic

### 17.4 測定プロトコル（Go/No-Go 判定）

| 条件 | 試験 | Pass 基準 |
| --- | --- | --- |
| 2 | shutdown 開始後 publish 試行 N=1000 | reject 1000/1000 |
| 5 | retire 後 stale pointer 参照検査 | 検出 0 |
| 7 | RT 禁則スキャン + allocator hook | 違反 0 |
| 8 | start/stop 反復 N=100 | drain true 100/100 |
| 9 | fallback 残留観測 | tail residency 0 収束 |
| 10 | finalize 再実行比較 | fingerprint 差分 0 |

### 17.5 Tier と受入条件の対応

- Tier0 完了条件: 1, 2, 3, 8, 9, 10
- Tier1 完了条件: Tier0 + reclaim 関連メトリクス安定
- Tier2 完了条件: Tier1 + 6

---

## 18. Rule トレーサビリティ

### 18.1 v6.x 統制文書との対応

- Rule-0A / 0B / 0C: 非 rewrite / survivability 優先 / 既存 hardening 保護
- Rule-1A / 1B / 1C / 1D / 1E: saturation admission / execution class 収束 / collapse / hysteresis
- Rule-2A / 2B / 2C / 2D / 2E: residency boundedness / append-only 抑止 / emergency reclaim
- Rule-3A / 3B / 3C / 3D: waitForDrain / source of truth / resurrection 禁止 / phase 統合
- Rule-4A / 4B / 4C / 4D: snapshot seal / worker read-only / finalize determinism
- Rule-5A / 5B / 5C: crossfade を Tier2 に固定 / execution-local 化
- Rule-6A / 6B: telemetry を guard 用途に限定 / ownership hardening
- Rule-7A / 7B / 7C: RT safety / non-RT reclaim / timeout wait 禁止
- Rule-8A: Tier 例外規定
- Rule-9A / 9B / 9C / 9D / 9E: 禁止事項の総括

### 18.2 addendum との対応

- Final Operational Hardening Addendum: migration compatibility / append-only 定義 / telemetry ownership / bounded wait
- Final Clarification Addendum: `coordinator.isFullyDrained()` 単一化 / telemetry を truth source にしない / bounded wait の意味論

---

## 19. 実装統制チェックリスト

- [ ] 影響範囲（関数単位）を列挙済み
- [ ] 変更対象の thread affinity を確認済み
- [ ] RT path 禁則を確認済み
- [ ] rollback 手順を定義済み
- [ ] telemetry 追加点を先行定義済み
- [ ] `coordinator.isFullyDrained()` の内部集約対象を定義済み
- [ ] `waitForDrain(timeout)` の bounded semantics を定義済み
- [ ] drained / resurrection の否定テスト観点を定義済み
- [ ] Tier0 完了前の Tier2 進出禁止を確認済み

---

## 20. 統合時に取り込んだ改訂要点

### 20.1 Critical 対応

- 追加: §5.4.1 / §5.4.2（atomic 契約、TOCTOU 対策）
- 追加: §10.1.1（`isFullyDrained()` 評価規約）
- 追加: §10.2.1 / §10.2.2（timeout/polling 定数、timeout 後動作）
- 追加: §11.1.1（saturation 入力・hysteresis）

### 20.2 High 対応

- 追加: §6.6.1（fingerprint 比較規則）
- 追加: §9.5.1（Residency Contract テンプレート）
- 追加: §11.4.1（emergency reclaim 運用定数）
- 追加: §17.4（測定プロトコル表）
- 追加: §12.3.1（crossfade mutable reduction 実施順序）

### 20.3 統制一貫性補強

- 追加: §1.1（詳細設計内優先順位）
- 追加: §3.4 / §10.3（shutdown authoritative state の明確化）
- 追加: §13.3.1（telemetry owner 既定割当）
- 追加: §17.5（Tier 完了条件の対応）

---

## 21. 結論

ConvoPeq に必要なのは ISR purity の理論的完成ではない。
必要なのは、DAW 実運用で以下を満たす runtime である。

- runaway しない
- saturation で収束する
- unload failure を起こしにくい
- drain completion を判定できる
- 長時間セッションで破綻しにくい

したがって本詳細設計（統合版）の中心は、次の 4 点に集約される。

1. saturation suppression
2. bounded residency
3. deterministic drain
4. reclaim pressure control
