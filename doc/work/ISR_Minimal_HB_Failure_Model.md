# ConvoPeq ISR Minimal HB Failure Model

## 目的

本書は **R16: HB Failure Spec + Reorder Simulation** の authoritative specification である。

本書は「再現テスト」ではなく、**HB が欠落したときに何が壊れるか** の仕様書である。
failure を specification として記述することで、CI HB simulator が検証すべき条件を定義する。

---

## HB Algebra

### HBEvent 定義

```cpp
enum class HBEvent
{
    Publish,    // runtime world を publish store(release) で公開
    Observe,    // observer が acquire load で runtime world を取得
    Retire,     // retire intent が enqueue される
    Reclaim,    // retire authority が epoch 確認後に reclaim を実行
    Shutdown    // shutdown FSM phase 遷移
};
```

### HBConstraint 定義

```cpp
struct HBConstraint
{
    HBEvent before;  // この event が
    HBEvent after;   // この event に happens-before でなければならない
    // before HB after が成立しない場合は violation
};
```

---

## 必須 HB Constraints

以下の constraint がすべて成立していなければ ISR は安全でない。

- HC1: **Publish HB Observe**
  - publish が完了する前に observe してはならない
- HC2: **Observe HB Retire**
  - observe 中の world を retire してはならない
- HC3: **Retire HB Reclaim**
  - retire intent 完了前に reclaim してはならない
- HC4: **Shutdown(StopAudio) HB Observe**
  - audio 停止後に新規 observe してはならない

---

## Minimal Failure Graph Catalog

HB constraint が欠落したときに発生する failure のカタログ。

### HB-01: observe/reclaim inversion（最重要）

```text
observe(old_world)
  -- [HB 欠落] --
reclaim(old_world)
→ Use-After-Free (UAF)
```

条件: HC2 または HC3 が成立しない場合

影響: Audio Thread が reclaim 済みメモリを参照する致命的 UAF

### HB-02: publish visibility gap

```text
publish store(old_world → new_world)
  -- [HB 欠落] --
observe loads old_world still
→ stale snapshot 参照
```

条件: HC1 が成立しない場合（acquire/release ペアの欠落）

影響: Audio Thread が古い state で DSP 処理を続行する（silent wrong output）

### HB-03: shutdown reclaim race

```text
Shutdown(EpochSettled 未完了)
  -- [HB 欠落] --
Reclaim(active_node)
→ UAF in shutdown
```

条件: ShutdownBarrier S3 が成立しない場合

影響: `~AudioEngine()` / `releaseResources()` 内での UAF

### HB-04: stale retire enqueue

```text
RT: emitRetireIntent(handle, epoch=N)
  -- [HB 欠落: completionEpoch 未検証] --
NonRT: reclaim at epoch < N
→ retire before epoch complete
```

条件: RetireCoordinator が completionEpoch を確認せずに reclaim した場合

影響: 他の observe が参照中の DSPHandle を reclaim する（UAF）

---

## 形式記述

各 failure は以下の形式で記述する：

```text
[Event A]
  !HB
[Event B]
→ [Violation type]
```

これは「A が B に happens-before でない場合に発生する violation」を意味する。

---

## CI HB Simulator

以下の条件で CI が HB simulator を実行すること：

1. **Static analysis**: HBConstraint (HC1-HC4) に対応する acquire/release ペアの存在確認
2. **Reorder simulation**: failure catalog (HB-01 〜 HB-04) の各シナリオを疑似実行
3. **Shutdown trace validation**: `shutdown_trace.json` の phase 順序と barrier 記録を検証

CI fail 条件：

- HC1-HC4 のいずれかを満たす acquire/release ペアが存在しない
- failure catalog の任意シナリオが trigger 可能と判定された

---

## 関連正本

- `ISR_HB_Graph_Specification.md` — Domain A/B/C の HB 仕様詳細
- `ISR_Shutdown_State_Machine.md` — HB-03 に対応する ShutdownBarrier 仕様
- `ISR_Deferred_Retire_Intent_Bridge.md` — HB-04 に対応する RetireIntent Bridge
- `ISR_Verification_Pipeline.md` V5 — HB Reorder Simulation ステージ
- `ISR_Formal_Guarantee_Package.md` P6 — 統合保証パッケージ参照

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R16 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（CI HB simulator 未実施）
