# ConvoPeq ISR Bridge Runtime Operational Survivability 基本設計

- Project: ConvoPeq
- Date: 2026-05-28
- Scope: `doc/work3/` に格納する、実運用耐性重視の基本設計
- Design Goal: **理論的完全性ではなく、DAW 長時間運用で破綻しにくい ISR Bridge Runtime の完成**

---

## 0. 本書の位置づけ

本書は、ConvoPeq における ISR Bridge Runtime 改修の**基本設計**である。

本設計は、既存の hardening を壊さず、実運用での破綻要因だけを段階的に封じ込めることを目的とする。

対象は次の failure mode である。

- rebuild storm
- retire saturation
- shutdown race
- unload failure
- stale runtime reuse
- publication race
- crossfade authority drift

---

## 1. 現状認識

現状の ConvoPeq は、既に次の領域へ到達している。

- RuntimeBuildSnapshot 化
- Unified Admission
- capture → finalize → seal
- publication validation
- shutdown phase control
- obsolete rebuild collapse
- bounded RT critical sections
- partial execution-local mutability

したがって、次の改修は ISR purity の理論追求ではなく、**operational survivability の完成**である。

---

## 2. 設計方針

### 2.1 最優先軸

1. long-session survivability
2. deterministic shutdown
3. saturation collapse
4. bounded reclamation
5. incremental migration
6. rewrite 回避

### 2.2 採用する考え方

- 既存 hardening は再実装しない
- 足りない operational guard のみを追加する
- 入口で抑止し、後段で救済しない
- 測定よりも flow control を優先する
- 理論的 purity よりも実運用安定性を優先する

### 2.3 禁止する方向

- 全面 rewrite
- 全面 lock-free 化
- DSP 全 immutable 化
- runtime graph の全面再設計
- epoch system の再実装
- shared_ptr の pervasive 化
- publication coordinator の全面 rewrite

これらは費用対効果が低く、regression surface を拡大しやすい。

---

## 3. 基本アーキテクチャ

### 3.1 既存骨格の扱い

ConvoPeq の ISR Bridge Runtime は、次の骨格を維持する。

- `AudioEngine` を中心とした admission / publication / shutdown / rebuild dispatch
- `RuntimePublicationCoordinator` による publication world 管理
- `RuntimeBuildSnapshot` による worker 入力の固定
- `ShutdownRuntime` による phase control
- `RetireRuntime` / `RetireRuntimeEx` による retire/reclaim 分離

### 3.2 スレッド責務

- Audio Thread: consume-only
- Message Thread: admission / publication / telemetry / lifecycle control
- Rebuild Worker: sealed snapshot ベース再構築
- Retire/Reclaim 処理: non-RT 側で best effort に実行

### 3.3 実行ローカル mutable の許容

次は排除対象ではなく、実運用上の許容領域とする。

- local fade progression
- thread-local scratch
- smoothing accumulators
- readonly cache materialization

ただし、これらは runtime authority を持たないこと。

---

## 4. 必須不変条件

### 4.1 Unified Admission Gate

`acceptsRuntimePublication()` を publication 経路の全入口に適用する。

適用対象の代表例:

- `requestRebuild`
- `submitRebuildIntent`
- `prepareCommit`
- `executeCommit`
- `appendPublicationIntent`

shutdown 受理フェーズ以降は publication intent を受け付けない。

### 4.2 Snapshot Seal Contract

snapshot lifecycle は次の順序に固定する。

```text
capture -> finalize -> seal -> handoff
```

finalize 後の mutation は禁止する。

### 4.3 Worker 入力契約

worker は sealed snapshot のみを受け取る。

禁止:

- active runtime pointer の参照
- fading runtime pointer の参照
- old runtime traversal
- runtime pointer 由来の補正

### 4.4 Shutdown Determinism

shutdown 完了は、少なくとも次が全成立したときにのみ認める。

- rebuild worker stopped
- publication intents empty
- retire queue empty
- fallback queue empty
- retired residency empty
- quarantine empty
- deferred free idle
- epoch domain drained

### 4.5 Residency Boundedness

retired オブジェクト群は、単なる append-only のブラックボックスにしない。

- queue depth
- fallback depth
- residency count
- reclaim latency

を deterministic に観測できることを前提とする。

---

## 5. Tier0 改修対象

### 5.1 Saturation Admission Suppression

最優先で導入する。

#### 目的

retire saturation 発生時に rebuild inflow を止める。

#### 入口方針

- Replaceable rebuild は reject 可能
- MustExecute rebuild は reject 不可
- collapse / coalescing は維持

#### 適用位置

- `submitRebuildIntent()` の入口
- `requestRebuild()` の入口

#### 抑止効果

- storm 時に queue growth を抑止
- obsolete rebuild collapse を壊さない
- saturation を「測れるだけ」にしない

### 5.2 Residency Accounting の一本化

`retire queue` / `fallback queue` / `retiredWorlds_` / `quarantine` を分散管理したままにしない。

#### 整理方針

- retired runtime residency の責務を一本化する
- bounded accounting を持たせる
- drain completion 判定に直結させる

#### 注意

巨大な新 Manager を追加するのではなく、既存の責務を整理して一本化する。

### 5.3 `waitForDrain(timeout)`

non-RT 限定で導入する。

#### 導入目的

外部から「完全 drain 完了」を待てるようにする。

#### 導入効果

- plugin unload の安全性向上
- DAW shutdown の determinism 向上
- rapid close/open の破綻防止

### 5.4 Emergency Reclaim

saturation 中に reclaim 圧を上げる。

#### 回収方針

- reclaim cadence を上げる
- obsolete retire を優先回収する
- fallback を aggressive に drain する
- rebuild suppression を強める

#### 禁止

- RT blocking
- forced synchronous stop

---

## 6. Tier1 改修対象

### 6.1 Quarantine Telemetry 実体化

quarantine を単なる名前にせず、残留量を観測可能にする。

### 6.2 Reclaim Prioritization

飽和時は、古い/obsolete なものを優先回収する。

### 6.3 Saturation Policy Tuning

HWM/LWM の関係は維持しつつ、saturation 時の安全側 tuning のみを許可する。

---

## 7. Tier2 改修対象

### 7.1 Crossfade Mutable Reduction

`latencyDelayOld_RT` / `latencyDelayNew_RT` / `dspCrossfade*` などのグローバル mutable を段階的に減らす。

ただし、ここは Tier0 の安全性確保後に進める。

### 7.2 Deeper Seal Propagation

seal / finalize の責務を必要最小限で拡張する。

ただし、DSP 全 immutable 化は行わない。

---

## 8. 実装順序

### Step 1

saturation admission suppression を導入する。

### Step 2

residency accounting を整理し、drain 判定を一本化する。

### Step 3

`waitForDrain(timeout)` を追加する。

### Step 4

emergency reclaim を導入する。

### Step 5

quarantine telemetry と reclaim prioritization を整える。

### Step 6

crossfade mutable reduction を段階導入する。

---

## 9. 受入判定

次を満たしたとき、基本設計として合格とする。

1. storm automation を長時間流しても runtime が収束する
2. retirement / fallback / quarantine の residency が bounded である
3. shutdown 後に publication resurrection が起きない
4. `waitForDrain(timeout)` で unload completion を判定できる
5. Replaceable rebuild が saturation 下で抑止される
6. MustExecute は救済経路として維持される
7. 既存 hardening を壊さずに運用安定性だけが向上する

---

## 10. 実装上の注意

### 10.1 入口で止める

Rule-A は queue に積んでから判定しない。

```cpp
if (retireSaturationActive_ && intent.policy == Replaceable)
    return Reject;
```

### 10.2 MustExecute の表現を一意化する

`forceMustExecute` と `policy == MustExecute` の二重管理は避ける。

将来的には単一の `RebuildExecutionClass` に寄せることを推奨する。

### 10.3 bounded accounting を崩さない

telemetry は「取るだけ」にせず、flow control へ結びつける。

---

## 11. リスクと抑止方針

### 11.1 過剰設計リスク

新しい Manager を作りすぎると、責務が増え、逆に破綻しやすい。

### 11.2 再設計リスク

全面 rewrite は regression surface を急拡大させる。

### 11.3 運用破綻リスク

観測だけで抑止しないと、storm は止まらず長時間運用で破綻する。

---

## 12. 結論

ConvoPeq に必要なのは ISR purity の理論的完成ではない。

必要なのは、DAW 実運用で以下を満たす runtime である。

- runaway しない
- saturation で収束する
- unload failure を起こしにくい
- drain completion を判定できる
- 長時間セッションで破綻しにくい

したがって本基本設計の中心は、次の 4 点に集約される。

1. saturation suppression
2. bounded residency
3. deterministic drain
4. reclaim pressure control
